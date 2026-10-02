"""The unchanged full Optuna protocol, with initial inside-outside content encoding."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qtl.search import main
from qtl.inside_outside import EXTRA_DEFAULTS

if __name__ == "__main__":
    main(train_entry=str(Path(__file__).with_name("train.py")), extra_defaults=EXTRA_DEFAULTS)
