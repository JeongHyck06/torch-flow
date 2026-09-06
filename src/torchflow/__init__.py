"""TorchFlow - 반응형 PyTorch 연구 워크벤치."""

__version__ = "0.0.1"


def __getattr__(name):
    """Attach API는 지연 로드한다 - ``import torchflow``가 torch를 끌고 오면
    hub 프로세스의 격리 불변식(§5.8)이 깨진다."""
    if name in ("watch", "hook_step", "log", "active", "Session"):
        from . import attach

        return getattr(attach, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["watch", "hook_step", "log", "active", "Session", "__version__"]
