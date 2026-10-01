"""Detached complete-study owner; paired with an independent local monitor."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from qtl.run_owner import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "search.py",
         "runs/20261002_residual_io_backbone")
