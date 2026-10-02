"""Full cached-cohort input validation before training or integrated evaluation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qtl.input_check import check_inputs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = check_inputs(a.corpus)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
