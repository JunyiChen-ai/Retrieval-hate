"""Locked residual-backbone diagnostics using the shared full-training owner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from qtl.diagnostics import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "train.py",
         "20261002_residual_io_backbone",
         {"full": {}, "nooutside": {"io_outside": False},
          "noresidual": {"io_residual": False}},
         {"backbone": "residual_io", "io_outside": True, "io_residual": True,
          "answer_source": "soft_both", "soft_levels": 8, "primary_budget": 8})
