from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MINIVIT = ROOT / "examples" / "minivit.tfg.json"


@pytest.fixture
def minivit():
    from torchflow.ir import load

    return load(MINIVIT)
