"""Revision 6 trainer: the shared QTL trainer (src/qtl/train.py); the settings come from search.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qtl.train import main

if __name__ == "__main__":
    main()
