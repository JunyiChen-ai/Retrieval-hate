#!/usr/bin/env python
"""KS-MNTP-2: CPU head dev screen for the S1 bidir+meanpool arm.

Trains the deployed align-fusion RGCL head on CPU over three feature arms and reads
DEV accuracy only. Paired CPU-to-CPU (recon §4.3: ERRPAT measured CPU-vs-CUDA head
drift at -0.0031, so a CPU arm may NEVER be compared against a banked GPU floor).

Arms (per dataset, 3 seeds each):
  causal    — the deployed banked cache (this is the CPU-trained floor)
  bidir     — banked F72 arm: mask flipped, deployed EOS-class text readout
  meanpool  — S1: mask flipped AND text readout = LLM2Vec mean over all positions

CLI is byte-identical to scripts/slurm/enc3seed_bidir.sbatch (the runner that produced
BOTH banked arms) except:
  --device cpu        (the banked arms used cuda)
  --num_workers 0     (CPU budget; the sampler is main-proc so no RNG stream changes)
  --group_name        (fresh group; never touches a banked group; --force stays False)

ZERO TEST-TOUCH, enforced by TWO independent belts:
  1. `load_feats_MHC` is replaced by a dev-only loader that returns (train, dev, dev),
     so the test cache file is never opened. The substitution is applied UNIFORMLY to
     every arm, so the harness is identical across arms and the pairing stays valid.
     The "test" rows run_rac then prints are dev-on-dev duplicates; they are never read.
  2. `load_feats_split` is wrapped in a hard guard that RAISES on any path containing
     "test_seen". If belt 1 ever regressed, the job dies instead of reading test.
The S1 caches have no test_seen file at all, which is the third, physical belt.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import data_loader.dataset as dataset_mod  # noqa: E402
import run_rac  # noqa: E402

# ---------------------------------------------------------------- test-touch belts
_ORIG_LOAD_SPLIT = dataset_mod.load_feats_split


def _guarded_load_split(path, *a, **kw):
    if "test_seen" in str(path):
        raise RuntimeError(
            "ZERO-TEST-TOUCH VIOLATION: something tried to load {!r}".format(path)
        )
    return _ORIG_LOAD_SPLIT(path, *a, **kw)


def _load_feats_MHC_devonly(path, model, dataset="MHC"):
    """(train, dev, dev) — test_seen is aliased to dev and never read."""
    train = _guarded_load_split("{}/{}/train_{}.pt".format(path, dataset, model))
    dev = _guarded_load_split("{}/{}/dev_seen_{}.pt".format(path, dataset, model))
    return train, dev, dev


dataset_mod.load_feats_split = _guarded_load_split
dataset_mod.load_feats_MHC = _load_feats_MHC_devonly

# ---------------------------------------------------------------- dev-metric spies
_ORIG_RETRIEVE = run_rac.retrieve_evaluate_RAC_
_ORIG_METRICS = run_rac.compute_metrics_retrieval
_STATE = {"eval_name": None, "epoch": None}
_ROWS = []


def _retrieve_spy(*a, **kw):
    _STATE["eval_name"] = kw.get("eval_name")
    _STATE["epoch"] = kw.get("epoch")
    return _ORIG_RETRIEVE(*a, **kw)


def _metrics_spy(logging_dict, labels, **kw):
    out = _ORIG_METRICS(logging_dict, labels, **kw)
    if _STATE["eval_name"] == "dev":  # DEV ONLY; the aliased "test" row is discarded
        acc, roc, pre, recall, f1, vote, lab, macro = out
        _ROWS.append({"epoch": int(_STATE["epoch"]), "acc": float(acc),
                      "roc": float(roc), "macro_f1": float(macro["macro_f1"]),
                      "macro_acc": float(macro["acc"])})
    return out


run_rac.retrieve_evaluate_RAC_ = _retrieve_spy
run_rac.compute_metrics_retrieval = _metrics_spy

# ---------------------------------------------------------------- the deployed CLI
ARMS = {
    "HateMM": {
        "causal": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
        "bidir": "Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir_HF",
        "meanpool": "Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-meanpool_HF",
    },
    "MHC_zh": {
        "causal": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
        "bidir": "Qwen2.5-VL-7B-Instruct-LoRA-bidir_HF",
        "meanpool": "Qwen2.5-VL-7B-Instruct-LoRA-bidir-meanpool_HF",
    },
}
SEEDS = [0, 1, 2]
WARMUP = 5


def base_argv(ds, model, seed):
    return [
        "run_rac.py",
        "--batch_size", "64",
        "--lr", "0.0001", "--epochs", "30", "--topk", "20", "--dataset", ds,
        "--model", model,
        "--proj_dim", "1024", "--map_dim", "1024", "--dropout", "0.2", "0.4", "0.1",
        "--fusion_mode", "align",
        "--hard_negatives_loss", "True", "--no_hard_negatives", "1",
        "--final_eval", "False", "--seed", str(seed),
        "--group_name", "mntp_s1_cpuhead",
        "--metric", "cos", "--loss", "triplet", "--batch_norm", "False",
        "--hybrid_loss", "True", "--warmup", str(WARMUP),
        "--majority_voting", "arithmetic", "--no_pseudo_gold_positives", "1",
        "--lambda_seg", "0", "--seg_mode", "full", "--num_subclips", "4",
        "--em_rounds", "2", "--consensus_topk", "10", "--consensus_margin", "0.2",
        "--exp_comment", "_{}".format(model),
        "--Faiss_GPU", "False", "--force", "False",
        "--device", "cpu",
        "--num_workers", "0",
    ]


def run_cell(ds, arm, model, seed):
    global _ROWS
    _ROWS = []
    sys.argv = base_argv(ds, model, seed)
    args = run_rac.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get("MNTP_THREADS", "6")))
    run_rac.main(args)
    rows = sorted(_ROWS, key=lambda r: r["epoch"])
    assert rows, "no dev rows captured for {}/{}/seed{}".format(ds, arm, seed)
    final = rows[-1]
    warm = [r for r in rows if r["epoch"] >= WARMUP] or rows
    best = max(warm, key=lambda r: (r["acc"], r["roc"]))
    print("[cpuhead] {} {} seed{}: FINAL ep{} dev acc {:.4f} mF1 {:.4f} | "
          "BESTDEV ep{} acc {:.4f}".format(ds, arm, seed, final["epoch"], final["acc"],
                                           final["macro_f1"], best["epoch"], best["acc"]),
          flush=True)
    return {"final": final, "best_dev": best, "n_epochs": len(rows)}


OUT_PATH = ROOT / "scripts/analysis/mntp_s1_cpuhead_OUT.json"


def load_prev():
    """Resume support: the causal/bidir arms do not need the S1 cache, so they can be
    run while the S1 extraction is still queued. Later runs merge into the same file."""
    if OUT_PATH.exists():
        with open(OUT_PATH) as fh:
            return json.load(fh)
    return {}


if __name__ == "__main__":
    only_ds = [a for a in sys.argv[1:] if a in ARMS] or list(ARMS)
    # MNTP_ARMS=causal,bidir runs only those arms (the ones that need no S1 cache).
    want_arms = [a.strip() for a in os.environ.get("MNTP_ARMS", "").split(",") if a.strip()]
    out = load_prev()
    for ds in only_ds:
        out.setdefault(ds, {})
        for arm, model in ARMS[ds].items():
            if want_arms and arm not in want_arms:
                continue
            out[ds][arm] = {}
            for s in SEEDS:
                out[ds][arm][str(s)] = run_cell(ds, arm, model, s)
        # paired deltas vs the CPU-trained causal floor, per seed then averaged.
        # Skipped for any arm not yet measured (partial runs are resumable).
        for arm in ("bidir", "meanpool"):
            if arm not in out[ds] or "causal" not in out[ds] or not out[ds].get(arm):
                continue
            d_fin = [out[ds][arm][str(s)]["final"]["acc"] - out[ds]["causal"][str(s)]["final"]["acc"]
                     for s in SEEDS]
            d_bst = [out[ds][arm][str(s)]["best_dev"]["acc"] - out[ds]["causal"][str(s)]["best_dev"]["acc"]
                     for s in SEEDS]
            out[ds].setdefault("paired", {})[arm] = {
                "delta_final_per_seed": d_fin, "mean_delta_final": float(np.mean(d_fin)),
                "delta_bestdev_per_seed": d_bst, "mean_delta_bestdev": float(np.mean(d_bst)),
            }
            m = float(np.mean(d_fin))
            verdict = ("ESCALATE-flag (>=+0.020)" if m >= 0.020 else
                       "CONTINUE (>=-0.014)" if m >= -0.014 else
                       "KILL (<=-0.05)" if m <= -0.05 else "MIDDLE (-0.05..-0.014)")
            out[ds]["paired"][arm]["verdict_final"] = verdict
            print("[cpuhead] {} {} vs CPU causal floor: mean Dfinal {:+.4f} "
                  "per-seed {} -> {}".format(ds, arm, m,
                                             ["{:+.4f}".format(x) for x in d_fin], verdict),
                  flush=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote {}".format(OUT_PATH))
