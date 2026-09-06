"""디바이스 배정과 폴백 (기획서 §5.1.5).

규칙 하나로 줄이면: L1은 학습을 방해하지 않는다. L2가 점유하지 않은 GPU를
고르고, 고를 것이 없으면 CPU로 내려가되 그래프는 살아 있게 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceChoice:
    device: str
    mode: str          # full | forward_only
    batch: int
    reason: str
    free_gb: float | None = None

    @property
    def forward_only(self) -> bool:
        return self.mode == "forward_only"


def available_devices(torch) -> list[str]:
    if torch.cuda.is_available():
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return ["mps"]
    return []


def _free_gb(torch, device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    free, _total = torch.cuda.mem_get_info(torch.device(device))
    return free / 1024**3


def pick_l1_device(torch, *, occupied: tuple[str, ...] = (), batch: int = 4) -> DeviceChoice:
    """L1이 쓸 디바이스를 고른다.

    ``occupied``는 L2 워커가 잡고 있는 디바이스다.
    """
    devices = available_devices(torch)

    if not devices:
        return DeviceChoice("cpu", "full", batch, "no accelerator")

    if devices == ["mps"]:
        # 통합 메모리다. L2가 돌면 같은 메모리를 두고 다투므로 동시 실행을 막는다(§5.1.5).
        if occupied:
            return DeviceChoice("cpu", "forward_only", 1,
                                "mps is unified memory; L2 is running")
        return DeviceChoice("mps", "full", batch, "mps")

    free = [device for device in devices if device not in occupied]
    if not free:
        # GPU를 전부 L2가 쓰고 있다 - 그래프를 죽이지 않고 CPU로 내려간다.
        return DeviceChoice("cpu", "forward_only", 1, "all GPUs held by L2")

    best = max(free, key=lambda device: _free_gb(torch, device) or 0.0)
    return DeviceChoice(best, "full", batch, "free GPU with most VRAM",
                        free_gb=_free_gb(torch, best))


def suggest_sweep_devices(torch) -> list[str]:
    """Sweep 기본 제안 = 사용 가능 GPU − 1. 하나는 L1에 남긴다(§5.1.5)."""
    devices = available_devices(torch)
    return devices[:-1] if len(devices) > 1 else devices
