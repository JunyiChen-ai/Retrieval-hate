"""Fixed Optuna search, with the same scalar space as the preceding backbone."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from qtl.search import main
from backbone import EXTRA_DEFAULTS

if __name__ == "__main__":
    main(train_entry=str(Path(__file__).with_name("train.py")), extra_defaults=EXTRA_DEFAULTS)
