"""Revision 7 with a validated first level (README section 10): the revision-7 search (search.py: label-free
location weight of the prior, mode "loc") plus first_level "validated" - training and checkpoint choice as revision
7; after training, the depth d of the breadth-first start of the question policy (the 2^d equal windows of tree depth
d first, then greedy EIG; d = 0 is the greedy policy) is chosen on the validation videos at every reported budget,
and the test numbers are those of the chosen d. Every trial also records every candidate d (summary "depths")."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qtl.search import main

EXTRA_DEFAULTS = {"answer_source": "soft_both", "node_prior": True, "answer_model": "anchored_shared",
                  "length_term": True, "prior_weight": "loc", "first_level": "validated"}

if __name__ == "__main__":
    main(train_entry=str(Path(__file__).with_name("train.py")), extra_defaults=EXTRA_DEFAULTS)
