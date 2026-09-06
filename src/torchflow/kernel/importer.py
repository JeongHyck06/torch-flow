"""인스턴스 import (기획서 §7.4 경로 3).

사용자가 준 .py를 **실행한다**. 클래스를 만들고 forward를 한 번 돌려야
그래프가 나오기 때문이다. `python train.py`를 직접 돌리는 것과 같은 신뢰
모델이고, 그래서 hub가 아니라 커널 프로세스에서만 일어난다.

인스턴스화는 `torch.device('meta')` 안에서 한다. 675M 파라미터 모델도 할당
없이 ms 단위로 만들어진다. meta에서 forward가 막히는 연산이 있으면 CPU로
한 번 물러선다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from ..attach.trace import trace_module, trace_report
from ..ir import ModuleGraph


class ImportError_(Exception):
    """import 실패. 어느 단계에서 막혔는지 붙여 보낸다."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


def load_module(path: Path):
    """파일 하나를 모듈로 읽는다. 같은 디렉터리의 형제 모듈도 import되게 한다."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"tf_import_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError_("load", f"cannot load {path}")

    module = importlib.util.module_from_spec(spec)
    parent = str(path.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError_("exec", f"{type(exc).__name__}: {exc}") from exc
    finally:
        if added:
            sys.path.remove(parent)
    return module


def build_inputs(torch, example_inputs: dict[str, Any], batch: int = 2, device="meta"):
    """``{"x": {"shape": ["B", 3, 32, 32], "dtype": "float32"}}`` 를 텐서로."""
    tensors = {}
    for name, spec in example_inputs.items():
        shape = [batch if dim == "B" else (dim if isinstance(dim, int) else 1)
                 for dim in spec.get("shape", [])]
        dtype = getattr(torch, spec.get("dtype", "float32"))
        if dtype.is_floating_point:
            tensors[name] = torch.zeros(shape, dtype=dtype, device=device)
        else:
            tensors[name] = torch.zeros(shape, dtype=dtype, device=device)
    return tensors


def import_instance(path: str | Path, factory: str,
                    example_inputs: dict[str, Any]) -> tuple[ModuleGraph, dict[str, Any]]:
    """모델을 만들고 한 번 돌려 Module Graph를 얻는다."""
    import torch

    module = load_module(Path(path))
    namespace = {name: getattr(module, name) for name in dir(module) if not name.startswith("__")}

    for device in ("meta", "cpu"):
        try:
            with torch.device(device):
                model = eval(factory, {"torch": torch, **namespace})  # noqa: S307
            model.eval()
            inputs = build_inputs(torch, example_inputs, device=device)
            ir = trace_module(model, inputs, name=type(model).__name__)
        except NotImplementedError:
            continue   # meta가 못 하는 연산이 있다. CPU로 물러선다.
        except Exception as exc:
            stage = "instantiate" if "model" not in dir() else "trace"
            raise ImportError_(stage, f"{type(exc).__name__}: {exc}") from exc

        report = trace_report(ir)
        report["device"] = device
        report["factory"] = factory
        return ir, report

    raise ImportError_("trace", "forward failed on both meta and cpu")
