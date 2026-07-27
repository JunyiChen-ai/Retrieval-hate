#!/usr/bin/env python
"""
mechnov_pairverify_runner.py -- ORCHESTRATION ONLY for the MECHNOV pair-verify pregate.

Why this file exists: the first execution of mechnov_pairverify.py's own `main()`
was killed by SIGTERM part-way through (login-node process reaping), losing the
spaces that had not yet been serialised. This runner drives the SAME frozen module
one (dataset x space) cell at a time in a short-lived process and serialises each
cell immediately, so a reap costs at most one cell.

IT CHANGES NO ARM. It imports `mechnov_pairverify` (sha256
77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d, unmodified) and
calls its frozen `run_space` with the frozen constants. Same folds, same seeds, same
PCA, same models, same bars. Only the process boundary and the file granularity
differ.
"""
import argparse
import json
import os
import sys

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
import mechnov_pairverify as P      # noqa: E402  FROZEN -- not modified

PARTS = os.path.join(REPO, "scripts/analysis/mechnov_parts")

FROZEN_SHA = "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d"


def meta_for(key):
    cfg = P.DATASETS[key]
    ids, _, _, lab = P.load_cache(cfg["cache_dir"], "train", cfg["model"])
    return {
        "script": os.path.join(REPO, "scripts/analysis/mechnov_pairverify.py"),
        "script_sha256": P.sha256_of(os.path.join(REPO, "scripts/analysis/mechnov_pairverify.py")),
        "mechfix_ops_sha256": P.sha256_of(os.path.join(REPO, "scripts/analysis/mechfix_ops.py")),
        "runner": os.path.abspath(__file__),
        "dataset": key, "ds": cfg["ds"], "model": cfg["model"],
        "n_train_items": int(len(ids)), "pos_rate": round(float(lab.mean()), 4),
        "frozen": dict(K_FOLDS=P.K_FOLDS, FOLD_SEED=P.FOLD_SEED, PCA_DIM=P.PCA_DIM,
                       PCA_SOLVER=P.PCA_SOLVER, PAIR_FIT_CAP=P.PAIR_FIT_CAP,
                       MLP_HIDDEN=P.MLP_HIDDEN, MLP_EPOCHS=P.MLP_EPOCHS,
                       MLP_BATCH=P.MLP_BATCH, MLP_LR=P.MLP_LR, MLP_WD=P.MLP_WD,
                       MLP_SEED=P.MLP_SEED, LOGIT_C=P.LOGIT_C,
                       LOGIT_MAXITER=P.LOGIT_MAXITER, M_PER_CLASS=P.M_PER_CLASS,
                       TOPK_DEPLOYED=P.TOPK_DEPLOYED, MEAN_TOPQ=P.MEAN_TOPQ,
                       PATHOLOGY_RANK=P.PATHOLOGY_RANK, SPACES=list(P.SPACES),
                       MODELS=list(P.MODELS), AGGS=list(P.AGGS)),
        "test_contact": "NONE -- only the train split is loaded",
    }


def run_cell(key, space):
    import torch
    torch.set_num_threads(8)
    os.makedirs(PARTS, exist_ok=True)
    cfg = P.DATASETS[key]
    _, img, txt, lab = P.load_cache(cfg["cache_dir"], "train", cfg["model"])
    X = P.build_space(img, txt, space)
    print(f"[{key}/{space}] n={X.shape[0]} dim={X.shape[1]}", flush=True)
    R = P.run_space(X, lab, space, lambda m: print(m, flush=True))
    out = os.path.join(PARTS, f"{key}_{space}.json")
    json.dump({"meta": meta_for(key), "space": space, "result": R},
              open(out, "w"), indent=1)
    print(f"[{key}/{space}] CELL DONE -> {out}", flush=True)


def merge(key):
    """Assemble the per-dataset OUT.json in the schema the frozen main() produced.
    Prefers an already-complete space from the original run when one exists."""
    cfg_out = os.path.join(REPO, f"scripts/analysis/mechnov_pairverify_{key}_OUT.json")
    existing = json.load(open(cfg_out)) if os.path.exists(cfg_out) else None
    OUT = {"meta": existing["meta"] if existing else meta_for(key), "spaces": {}}
    OUT["meta"]["assembly"] = ("per-space cells written by mechnov_pairverify_runner.py; "
                               "frozen arms module unmodified, sha " + FROZEN_SHA)
    for space in P.SPACES:
        p = os.path.join(PARTS, f"{key}_{space}.json")
        if os.path.exists(p):
            OUT["spaces"][space] = json.load(open(p))["result"]
        elif existing and space in existing.get("spaces", {}):
            OUT["spaces"][space] = existing["spaces"][space]
    json.dump(OUT, open(cfg_out, "w"), indent=1)
    print(f"[{key}] merged spaces={list(OUT['spaces'])} -> {cfg_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(P.DATASETS))
    ap.add_argument("--space", choices=list(P.SPACES))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    assert P.sha256_of(os.path.join(REPO, "scripts/analysis/mechnov_pairverify.py")) == FROZEN_SHA, \
        "FROZEN ARMS MODULE HAS CHANGED -- refusing to run"
    if a.merge:
        merge(a.dataset)
    else:
        run_cell(a.dataset, a.space)


if __name__ == "__main__":
    main()
