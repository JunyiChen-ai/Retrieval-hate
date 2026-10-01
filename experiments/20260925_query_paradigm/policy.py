"""Compatibility entry point; implementation promoted to src/qtl for reuse."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qtl import policy as _implementation

sys.modules[__name__] = _implementation
