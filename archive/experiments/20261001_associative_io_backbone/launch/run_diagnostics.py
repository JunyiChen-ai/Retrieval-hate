"""Locked-configuration structural diagnostics via the shared QTL owner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from qtl.diagnostics import inspect_trial, main

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "train.py",
         '20261001_associative_io_backbone',
         {'full': {}, 'nooutside': {'io_outside': False}, 'noattention': {'io_attention': False}},
         {'backbone': 'associative_io', 'io_outside': True, 'io_attention': True})
