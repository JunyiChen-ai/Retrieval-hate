"""Shared query-tree training, inference and input infrastructure."""
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "scripts/reproduction_baselines", _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
