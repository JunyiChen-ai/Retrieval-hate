"""Full QTL training with the additive inside-outside content backbone."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qtl.train import main
from backbone import EXTRA_DEFAULTS, make_model

if __name__ == "__main__":
    main(model_factory=make_model, extra_defaults=EXTRA_DEFAULTS)
