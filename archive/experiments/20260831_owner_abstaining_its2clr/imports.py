"""Load the frozen MultiHateLoc reimplementation without copying it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BASELINE_ROOT = REPO / "scripts/reproduction_baselines"
MULTIHATELOC_ROOT = BASELINE_ROOT / "multihateloc"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base_model = _load("frozen_multihateloc_model", MULTIHATELOC_ROOT / "model.py")
base_data = _load("frozen_multihateloc_data", MULTIHATELOC_ROOT / "data.py")

