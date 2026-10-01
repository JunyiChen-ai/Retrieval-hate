"""Compatibility entry point; implementation promoted to src/qtl for reuse."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qtl import search as _implementation

if __name__ == "__main__":
    _implementation.main(train_entry=str(Path(__file__).with_name("train.py")))
else:
    sys.modules[__name__] = _implementation
