#!/usr/bin/env python3
"""Three-seed retraining of a baseline's validation-selected configuration, scored on test once (README section 2).

Same procedure as scripts/reproduction_baselines/confirm_official_val.py (the train / inference / evaluation commands
are imported from it unchanged): read the Optuna winner best.json of the 40-trial validation search, retrain it at
seeds 234 / 2025 / 3407, write test scores, score them with eval_baseline_scores.py. It differs only in what it
records: no file digests and no commit identifiers (CLAUDE.md, hashing ban 2026-09-05); a completed seed is
recognised by its frozen_config.json and a parsable frame_eval.json.

    python experiments/20260926_dehate_external/confirm_baselines.py --method macilsd --corpus dehate \
        --tuning-root runs/20260926_dehate_external/baselines/tuning \
        --final-root runs/20260926_dehate_external/baselines/final
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(BASE))
from confirm_official_val import SEEDS, inference_command, materialize, run, train_command  # noqa: E402
from tune_official_val import DEFAULT_PYTHON  # noqa: E402

CODE_VERSION = "2026-09-26 DeHate external validation (confirm_baselines.py, first version)"


def done(out, corpus):
    path = out / "frame_eval.json"
    if not ((out / "frozen_config.json").is_file() and path.is_file()):
        return False
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return payload.get("corpus") == corpus and payload.get("split") == "test" and bool(payload.get("results"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--tuning-root", required=True)
    ap.add_argument("--final-root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    best_path = Path(args.tuning_root) / args.method / args.corpus / "best.json"
    selected = json.loads(best_path.read_text())
    values = materialize(selected["best_params"])
    root = Path(args.final_root) / args.method / args.corpus
    for seed in args.seeds:
        out = root / f"seed_{seed}"
        if done(out, args.corpus):
            print(f"already complete {args.method}/{args.corpus}/seed_{seed}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        frozen = {"method": args.method, "corpus": args.corpus, "seed": seed, "code_version": CODE_VERSION,
                  "host": socket.gethostname(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "source": str(best_path), "best_trial": selected["best_trial"],
                  "best_validation_ap": selected["best_value"], "params": values}
        (out / "frozen_config.json").write_text(json.dumps(frozen, indent=2) + "\n")
        run(train_command(args.method, args.corpus, out, values, seed, args.python), out / "train.log")
        infer = inference_command(args.method, args.corpus, out, values, args.python)
        if infer:
            run(infer, out / "infer.log")
        scores = out / args.corpus / "scores.jsonl" if args.method == "multihateloc" else out / "scores.jsonl"
        run([args.python, str(BASE / "eval_baseline_scores.py"), "--corpus", args.corpus, "--scores", str(scores),
             "--split", "test", "--require-full-coverage", "--json-out", str(out / "frame_eval.json")],
            out / "eval.log")
        print(f"completed {args.method}/{args.corpus}/seed_{seed}", flush=True)


if __name__ == "__main__":
    main()
