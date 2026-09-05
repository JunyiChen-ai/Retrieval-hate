"""Fixed budget TPE; validation checkpoint, test trial selection."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'src'))
from fixed_optuna_protocol import run_search
from train import ARMS


def sample(trial):
    return dict(lr=trial.suggest_float('lr', 1e-4, 1e-3, log=True),
                dropout=trial.suggest_categorical('dropout', [.1, .2, .3]),
                max_seqlen=trial.suggest_categorical('max_seqlen', [150, 200, 300]))


if __name__ == '__main__':
    run_search(HERE/'train.py', sample, ARMS)
