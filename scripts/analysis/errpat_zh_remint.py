#!/usr/bin/env python
"""ERRPAT MHC-ZH: CPU re-mint of the deployed ZH-LoRA head with per-item, per-epoch dumps.

WHY: the deployed floor head ckpts for job 13150 are ALL DELETED (F78 / CURATION_FORENSIC_RECON
     §2.2), and no per-item prediction dump or retrieval pkl survives. Per-item test predictions
     for the binding ZH floor are therefore UNRECOVERABLE bit-exactly.

WHAT THIS IS: a same-recipe, same-cache, same-seed CPU re-mint. Byte-identical CLI to the
     13150 sbatch (scripts/slurm/enc3seed_zh_b3.sbatch) except:
       --device cpu           (13150 used cuda)
       --group_name errpat_zh_remint   (fresh dir; never touches any banked group)
     Model init draws from the CPU RNG in both runs (build_model() then .to(device)), so the
     INITIALISATION is identical; the trajectories diverge because dropout masks are drawn from
     the CUDA RNG on GPU and from the CPU RNG here, and because matmul reduction order differs.

  ==> THIS IS A PROXY. IT IS NOT BIT-EXACT TO JOB 13150. Fidelity is measured, not assumed:
      errpat_zh_remint_fidelity.py compares the re-minted per-epoch dev/test curves against the
      banked 13150 trainlog curves.

Dumps, per seed, for every epoch 0..29 and both dev+test splits:
  query id, gold label, vote scalar, prediction, and the full top-20 (neighbour ids, cosines, labels).
"""
import os
import pickle
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUTDIR = ROOT / "scripts/analysis/errpat_remint_dumps"
OUTDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- monkeypatches
import run_rac  # noqa: E402
import utils.metrics as metrics_mod  # noqa: E402

_ORIG_RETRIEVE = run_rac.retrieve_evaluate_RAC_
_ORIG_METRICS = run_rac.compute_metrics_retrieval

_STATE = {"eval_name": None, "epoch": None}
_DUMP = []  # list of records


def _retrieve_spy(*a, **kw):
    _STATE["eval_name"] = kw.get("eval_name")
    _STATE["epoch"] = kw.get("epoch")
    return _ORIG_RETRIEVE(*a, **kw)


def _metrics_spy(logging_dict, labels, **kw):
    out = _ORIG_METRICS(logging_dict, labels, **kw)
    # NB: metrics.py returns `list_majority_voted` (the raw vote scalars) in the
    # 6th slot, NOT the rounded prediction.
    acc, roc, pre, recall, f1, vote_ret, lab, macro_val = out
    # Recompute the vote exactly as metrics.py:262-301 does, per query, so we keep the scalar.
    topk = kw.get("topk", 20)
    weight = np.arange(1, topk + 1)[::-1]
    ids, votes, nb_ids, nb_sims, nb_labs, nretr = [], [], [], [], [], []
    for qid, v in logging_dict.items():
        rl = np.asarray(v["retrieved_label"])
        rs = np.asarray([s.item() if hasattr(s, "item") else s for s in v["retrieved_scores"]])
        mapped = (rl * 2 - 1) * rs
        L = len(mapped)
        vote = float(np.sum(mapped * weight[:L]) / np.sum(weight[:L]))
        ids.append(qid)
        votes.append(vote)
        nretr.append(L)
        pad = topk - L
        nb_ids.append(list(v["retrieved_ids"]) + [""] * pad)
        nb_sims.append(np.concatenate([rs, np.full(pad, np.nan)]))
        nb_labs.append(np.concatenate([rl, np.full(pad, -1)]))
    labs = lab.detach().cpu().numpy() if hasattr(lab, "detach") else np.asarray(lab)
    votes = np.asarray(votes, dtype=np.float64)
    # parity guard: our per-query recompute must equal metrics.py's own vote list
    ret = np.asarray(vote_ret, dtype=np.float64)
    assert votes.shape == ret.shape and np.max(np.abs(votes - ret)) < 1e-12, \
        f"vote recompute parity FAILED max|d|={np.max(np.abs(votes - ret)):.3e}"
    # deployed decision rule (metrics.py:300): predict 1 iff sigmoid(vote) >= 0.5 <=> vote >= 0
    pred = (1.0 / (1.0 + np.exp(-votes)) >= 0.5).astype(np.int64)
    assert abs(float(np.mean(pred == labs)) - float(acc)) < 1e-12, "acc parity FAILED"
    _DUMP.append({
        "split": _STATE["eval_name"],
        "epoch": int(_STATE["epoch"]),
        "ids": ids,
        "gold": labs.astype(np.int64),
        "vote": votes,
        "pred": pred,
        "n_retrieved": np.asarray(nretr, dtype=np.int64),
        "nb_ids": nb_ids,
        "nb_sim": np.asarray(nb_sims, dtype=np.float32),
        "nb_lab": np.asarray(nb_labs, dtype=np.int64),
        "acc": float(acc), "roc": float(roc), "pre": float(pre),
        "recall": float(recall), "f1": float(f1),
        "macroF1": macro_val["macro_f1"], "macroP": macro_val["macro_pre"],
        "macroR": macro_val["macro_recall"],
    })
    return out


run_rac.retrieve_evaluate_RAC_ = _retrieve_spy
run_rac.compute_metrics_retrieval = _metrics_spy

# ---------------------------------------------------------------- exact 13150 CLI
BASE_ARGV = [
    "run_rac.py",
    "--batch_size", "64",
    "--lr", "0.0001", "--epochs", "30", "--topk", "20", "--dataset", "MHC_zh",
    "--model", "Qwen2.5-VL-7B-Instruct-LoRA_HF",
    "--proj_dim", "1024", "--map_dim", "1024", "--dropout", "0.2", "0.4", "0.1",
    "--fusion_mode", "align",
    "--hard_negatives_loss", "True", "--no_hard_negatives", "1",
    "--final_eval", "False",
    "--group_name", os.environ.get("ERRPAT_GROUP", "errpat_zh_remint_v2"),
    "--metric", "cos", "--loss", "triplet", "--batch_norm", "False",
    "--hybrid_loss", "True", "--warmup", "5",
    "--majority_voting", "arithmetic", "--no_pseudo_gold_positives", "1",
    "--lambda_seg", "0", "--seg_mode", "full", "--num_subclips", "4",
    "--em_rounds", "2", "--consensus_topk", "10", "--consensus_margin", "0.2",
    "--exp_comment", "_Qwen2.5-VL-7B-Instruct-LoRA_HF",
    "--Faiss_GPU", "False", "--force", "False",
    "--device", "cpu",              # <-- the only substantive deviation
    "--num_workers", "0",           # <-- CPU budget; affects no RNG stream (sampler is main-proc)
]


def run_seed(seed):
    global _DUMP
    _DUMP = []
    sys.argv = BASE_ARGV + ["--seed", str(seed)]
    args = run_rac.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get("ERRPAT_THREADS", "6")))
    run_rac.main(args)
    dst = OUTDIR / f"errpat_zh_remint_seed{seed}.pkl"
    with open(dst, "wb") as f:
        pickle.dump({"seed": seed, "argv": sys.argv, "records": _DUMP}, f)
    print(f"[errpat] seed{seed}: {len(_DUMP)} eval records -> {dst}")


if __name__ == "__main__":
    seeds = [int(x) for x in (sys.argv[1:] or ["0", "1", "2"])]
    for s in seeds:
        run_seed(s)
