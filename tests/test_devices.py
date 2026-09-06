"""디바이스 배정과 폴백 (기획서 §5.1.5)."""

import pytest

torch = pytest.importorskip("torch")

from torchflow.kernel.devices import pick_l1_device, suggest_sweep_devices  # noqa: E402


class FakeTorch:
    """GPU 구성을 흉내 낸다 - CI에 GPU가 없어도 배정 규칙은 검증되어야 한다."""

    def __init__(self, cuda: int = 0, mps: bool = False, free=None):
        self._cuda, self._mps = cuda, mps
        self._free = free or {}
        outer = self

        class _Cuda:
            @staticmethod
            def is_available():
                return outer._cuda > 0

            @staticmethod
            def device_count():
                return outer._cuda

            @staticmethod
            def mem_get_info(device):
                return int(outer._free.get(str(device), 8) * 1024**3), int(24 * 1024**3)

        class _Mps:
            @staticmethod
            def is_available():
                return outer._mps

        class _Backends:
            mps = _Mps()

        self.cuda = _Cuda()
        self.backends = _Backends()
        self.device = str


def test_no_accelerator_falls_back_to_cpu_but_keeps_backward():
    choice = pick_l1_device(FakeTorch())
    assert choice.device == "cpu" and choice.mode == "full"


def test_picks_the_gpu_with_the_most_free_memory():
    fake = FakeTorch(cuda=3, free={"cuda:0": 2, "cuda:1": 19, "cuda:2": 7})
    assert pick_l1_device(fake).device == "cuda:1"


def test_avoids_gpus_held_by_the_training_worker():
    fake = FakeTorch(cuda=2, free={"cuda:0": 20, "cuda:1": 4})
    assert pick_l1_device(fake, occupied=("cuda:0",)).device == "cuda:1"


def test_all_gpus_busy_falls_back_to_forward_only_cpu():
    """그래프는 살아 있어야 한다 - 배지가 회색으로 남을 뿐(§5.1.5)."""
    fake = FakeTorch(cuda=1, free={"cuda:0": 20})
    choice = pick_l1_device(fake, occupied=("cuda:0",))
    assert choice.device == "cpu" and choice.forward_only and choice.batch == 1


def test_mps_refuses_to_share_with_training():
    """통합 메모리에서 L1과 L2를 같이 돌리지 않는다(§5.1.5)."""
    assert pick_l1_device(FakeTorch(mps=True)).device == "mps"
    choice = pick_l1_device(FakeTorch(mps=True), occupied=("mps",))
    assert choice.device == "cpu" and choice.forward_only


def test_sweep_leaves_one_gpu_for_l1():
    assert suggest_sweep_devices(FakeTorch(cuda=4)) == ["cuda:0", "cuda:1", "cuda:2"]
    assert suggest_sweep_devices(FakeTorch(cuda=1)) == ["cuda:0"]
    assert suggest_sweep_devices(FakeTorch()) == []


def test_real_torch_returns_a_usable_device():
    choice = pick_l1_device(torch)
    assert choice.device in {"cpu", "mps"} or choice.device.startswith("cuda")
