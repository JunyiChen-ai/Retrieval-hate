"""Fixed budget TPE; validation checkpoint, test trial selection."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]/'src'))
from fixed_optuna_protocol import run_search
from content_search_space import sample
from train import ARMS


if __name__ == '__main__':
    run_search(HERE/'train.py', sample, ARMS)
