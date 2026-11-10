"""Revision 7 search (README section 8): the revision-6 Optuna search (four scalars, 20 trials per seed) with the
label-free location weight of the prior (src/qtl/prior_weight.py, mode "loc") estimated after every epoch on the
validation answers and used at validation (checkpoint choice) and test. Everything else as revision 6."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qtl.search import main

EXTRA_DEFAULTS = {"answer_source": "soft_both", "node_prior": True, "answer_model": "anchored_shared",
                  "length_term": True, "prior_weight": "loc"}

if __name__ == "__main__":
    main(train_entry=str(Path(__file__).with_name("train.py")), extra_defaults=EXTRA_DEFAULTS)
