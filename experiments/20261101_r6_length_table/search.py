"""Revision 6 search (README): the r5 Optuna search (four scalars, 20 trials per seed) with the shared-length-slope
anchored answer table and no copy likelihood."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qtl.search import main

EXTRA_DEFAULTS = {"answer_source": "soft_both", "node_prior": True, "answer_model": "anchored_shared",
                  "length_term": True}

if __name__ == "__main__":
    main(train_entry=str(Path(__file__).with_name("train.py")), extra_defaults=EXTRA_DEFAULTS)
