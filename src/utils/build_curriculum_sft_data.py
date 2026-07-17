#!/usr/bin/env python
"""Build a retrieval-confusion CURRICULUM variant of the LoRA-SFT train set (cand-2).

This is a curriculum builder for the encoder-level LoRA-SFT regime (B3 / LoRA-HateMM):
it does NOT re-generate any SFT record. It FORKS the generic word-variant train.json
produced by build_lora_sft_data.py and re-weights ONLY its example distribution. Each
train video's sampling multiplicity is set from its frozen-encoder *confusability* c_i,
so the RGCL archive/memory (the LOO kNN vote under the banked frozen Qwen features)
constructs the encoder-adaptation curriculum. The emitted records are byte-identical to
the generic arm's records (same 8 frames, same instruction, same word target); the ONLY
manipulated variable is how often each record appears, and the total is capped to
N_train so the 3-epoch SFT step-count is IDENTICAL to the generic arm (cost-neutral).

Mining (ZERO GPU, $0 CPU over the banked cache):
  * load data/CLIP_Embedding/<DS>/train_<FROZEN_TAG>.pt (frozen Qwen2.5-VL-7B pooled
    3584-d img/text features + labels; the SAME features the deployed kNN memory uses).
  * fused LOO kNN: L2-normalise each stream, concat, renormalise (=> IP is the mean of
    the img-cosine and text-cosine), top-20 rank-weighted signed-cosine vote with
    exclude_self (leave-one-out; TRAIN ONLY, no dev/test ever enters the index).
  * confusability:
      - softconf (i-a, REGISTERED): c_i = exp(-|vote_i| / TAU)  (peaks at the boundary).
      - error   (i-b, variant):     c_i = 1{LOO vote misclassifies i}.
  * multiplicity weight w_i = 1 + LAMBDA * c_i.

Curriculum multiset (cost-neutral, DETERMINISTIC — no RNG in the softconf path):
  deterministic largest-remainder apportionment of exactly N_train slots proportional to
  w_i (floor(quota) then hand the residual slots to the largest fractional remainders;
  ties broken by larger w then lower index). Easy (low-c) videos may receive 0 copies
  (the easy tail is subsampled to make room for the duplicated confusable head), so the
  multiset shifts mass onto the boundary at the SAME budget as generic.

Output: data/lora_sft/<DS>/train_curric.json (list, byte-identical record dicts, N rows),
registered as <prefix>_lora_curric_train in dataset_info.json. Val is UNCHANGED (the
generic <prefix>_lora_val). Also writes refine-logs/CAND2_KC20_<DS>.json with the K-C2-0
mining-validity diagnostics.

Idempotent: re-running reproduces train_curric.json byte-for-byte (pinned sha256).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import faiss
import torch

# Reuse the generic builder's dataset-info registration + path constants verbatim.
_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_THIS)  # .../RGCL/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from utils.build_lora_sft_data import (  # noqa: E402
    DS_PREFIX,
    RGCL_ROOT,
    register_dataset_info,
)

faiss.omp_set_num_threads(1)  # deterministic exact search

CACHE_ROOT = os.path.join(RGCL_ROOT, "data", "CLIP_Embedding")
FROZEN_TAG = "Qwen2.5-VL-7B-Instruct_HF"  # the banked frozen-Qwen train cache to mine
LORA_SFT_ROOT = os.path.join(RGCL_ROOT, "data", "lora_sft")
KC20_OUT_DIR = os.path.join(RGCL_ROOT, "refine-logs")

# ---- PINNED curriculum hyperparameters (frozen by the prereg) ----------------
TOPK = 20            # matches the deployed head top-20 kNN + the router-gate mining
TAU = 0.20           # softconf temperature on the |signed-cosine-vote| scale
LAMBDA = 10.0        # multiplicity gain: w_i = 1 + LAMBDA * c_i
CAP_RATIO = 1.0      # curriculum size = CAP_RATIO * N_train (cost-neutral == generic)
SEED = 20260718      # nominal RNG seed (softconf+largest-remainder path uses NO RNG)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load_frozen_cache(ds):
    """Return (ids[list[str]], img[N,D] f32, text[N,D] f32, labels[N] int)."""
    d = torch.load(os.path.join(CACHE_ROOT, ds, "train_{}.pt".format(FROZEN_TAG)),
                   map_location="cpu")
    ids = d["ids"]
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    ids = [str(i) for i in ids]
    return (ids, d["img_feats"].float().numpy(), d["text_feats"].float().numpy(),
            d["labels"].long().numpy())


def _weighted_signed_vote(nb_lab, nb_sim):
    """Arithmetic rank-weighted signed-cosine vote over top-k neighbours.

    Lifted verbatim from scripts/analysis/cross_channel_router_gate.py:73-78 (the
    deployed metrics.py use_sim path)."""
    k = len(nb_lab)
    w = np.arange(1, TOPK + 1)[::-1].astype("float64")[:k]
    lm = (nb_lab.astype("float64") * 2 - 1) * nb_sim
    return float(np.sum(lm * w) / np.sum(w))


def mine_confusion(ds, mode="softconf"):
    """Per-anchor frozen-encoder confusability over the banked train cache (LOO kNN).

    Returns (cmap: {id -> c_i in [0,1]}, diag: dict of K-C2-0 mining-validity numbers)."""
    ids, img, txt, lab = _load_frozen_cache(ds)

    def _unit(x):
        x = x.astype("float32").copy()
        faiss.normalize_L2(x)
        return x

    feats = np.concatenate([_unit(img), _unit(txt)], axis=1).astype("float32")
    faiss.normalize_L2(feats)  # IP now == mean(img-cosine, text-cosine), in [-1, 1]
    index = faiss.IndexFlatIP(feats.shape[1])
    index.add(feats)
    sims, nbrs = index.search(feats, TOPK + 1)  # +1 for the self hit we drop

    votes = np.zeros(len(feats))
    preds = np.zeros(len(feats), dtype=int)
    for i in range(len(feats)):
        idx, sim = nbrs[i], sims[i]
        keep = idx != i  # leave-one-out: exclude self (TRAIN index; no dev/test present)
        idx, sim = idx[keep][:TOPK], sim[keep][:TOPK]
        v = _weighted_signed_vote(lab[idx], sim)
        votes[i] = v
        preds[i] = int(v >= 0)

    av = np.abs(votes)
    if mode == "softconf":
        c = np.exp(-av / TAU)
    elif mode == "error":
        c = (preds != lab).astype("float64")
    else:
        raise ValueError("unknown mode {!r}".format(mode))

    loo_err = float((preds != lab).mean())
    diag = {
        "n_train_cache": int(len(ids)),
        "loo_acc": round(float((preds == lab).mean()), 4),
        "loo_error_rate": round(loo_err, 4),
        "abs_vote_median": round(float(np.median(av)), 4),
        "c_gini": round(_gini(c), 4),
        "mode": mode,
        "tau": TAU,
    }
    return {ids[i]: float(c[i]) for i in range(len(ids))}, diag


def _gini(x):
    xs = np.sort(np.asarray(x, dtype="float64"))
    n = len(xs)
    if xs.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(xs) / (n * xs.sum()))


def _largest_remainder(weights, total):
    """Deterministic apportionment of `total` slots proportional to `weights`.

    floor(quota) then hand the residual to the largest fractional remainders; ties
    broken by larger weight then lower index. No RNG. Sums exactly to `total`."""
    w = np.asarray(weights, dtype="float64")
    quota = total * w / w.sum()
    base = np.floor(quota).astype(int)
    residual = int(total - base.sum())
    frac = quota - base
    order = sorted(range(len(w)), key=lambda i: (-frac[i], -w[i], i))
    mult = base.copy()
    for i in order[:residual]:
        mult[i] += 1
    return mult


def _record_id(record):
    """Recover the video id from a ShareGPT record's frame path .../<id>/frame_0.jpg."""
    return os.path.basename(os.path.dirname(record["images"][0]))


def build_curriculum(ds, mode="softconf"):
    generic_path = os.path.join(LORA_SFT_ROOT, ds, "train.json")
    if not os.path.exists(generic_path):
        raise SystemExit(
            "[curric] missing generic word-variant train set {}. Run "
            "`python src/utils/build_lora_sft_data.py --dataset {}` first.".format(
                generic_path, ds))
    with open(generic_path, "r") as f:
        records = json.load(f)
    n = len(records)
    rec_ids = [_record_id(r) for r in records]

    cmap, diag = mine_confusion(ds, mode)
    n_missing = sum(1 for rid in rec_ids if rid not in cmap)  # anchors absent from cache
    c_vec = np.array([cmap.get(rid, 0.0) for rid in rec_ids], dtype="float64")
    weights = 1.0 + LAMBDA * c_vec

    cap = int(round(CAP_RATIO * n))
    mult = _largest_remainder(weights, cap)

    out_records = []
    for rec, m in zip(records, mult):
        out_records.extend([rec] * int(m))
    assert len(out_records) == cap, (len(out_records), cap)

    out_path = os.path.join(LORA_SFT_ROOT, ds, "train_curric.json")
    with open(out_path, "w") as f:
        json.dump(out_records, f, indent=2, ensure_ascii=False)

    # ---- K-C2-0 mining-validity diagnostics ---------------------------------
    coverage = int((mult > 0).sum())
    thr70 = float(np.percentile(c_vec, 70))
    hard = c_vec >= thr70
    hard_mass = float(mult[hard].sum()) / cap
    diag.update({
        "dataset": ds,
        "n_train_sft": n,
        "n_anchor_missing_from_cache": int(n_missing),
        "lambda": LAMBDA,
        "cap_ratio": CAP_RATIO,
        "curriculum_size": int(len(out_records)),
        "size_equals_N": bool(len(out_records) == n),
        "weight_gini": round(_gini(weights), 4),
        "unique_coverage": coverage,
        "coverage_frac": round(coverage / n, 4),
        "dropped_easy": int(n - coverage),
        "max_dup": int(mult.max()),
        "dup_hist": np.bincount(mult).tolist(),
        "hard_head_top30_mass": round(hard_mass, 4),
        "hard_head_uniform_mass": round(float(hard.mean()), 4),
        "hard_head_concentration_x": round(hard_mass / float(hard.mean()), 4),
        "generic_train_json_sha256": _sha256(generic_path),
        "train_curric_json_sha256": _sha256(out_path),
    })
    # K-C2-0 pre-declared checks (informational here; the gate is applied at review):
    diag["KC20_a_nondegenerate"] = bool(0.15 <= diag["loo_error_rate"] <= 0.35)
    diag["KC20_b_concentration"] = bool(diag["c_gini"] >= 0.30)
    diag["KC20_c_differs_from_uniform"] = bool(diag["coverage_frac"] < 0.90)
    diag["KC20_PASS"] = bool(diag["KC20_a_nondegenerate"]
                             and diag["KC20_b_concentration"]
                             and diag["KC20_c_differs_from_uniform"])

    kc20_path = os.path.join(KC20_OUT_DIR, "CAND2_KC20_{}.json".format(ds))
    with open(kc20_path, "w") as f:
        json.dump(diag, f, indent=2)

    prefix = DS_PREFIX[ds]
    register_dataset_info({"{}_lora_curric_train".format(prefix): out_path})

    print("[curric] {} mode={} -> {} rows (N={}, cap={})".format(ds, mode, len(out_records), n, cap))
    print("[curric]   K-C2-0: LOO-err {loo_error_rate} (nondeg {KC20_a_nondegenerate}) | "
          "c-Gini {c_gini} (conc {KC20_b_concentration}) | coverage {coverage_frac} "
          "(differs {KC20_c_differs_from_uniform}) | hard-head x{hard_head_concentration_x} | "
          "PASS={KC20_PASS}".format(**diag))
    print("[curric]   train_curric.json sha256 {}".format(diag["train_curric_json_sha256"]))
    print("[curric]   diagnostics -> {}".format(kc20_path))
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["MHC", "MHC_zh", "HateMM"])
    ap.add_argument("--mode", default="softconf", choices=["softconf", "error"],
                    help="softconf (i-a, REGISTERED) or error (i-b, binary variant).")
    args = ap.parse_args()
    build_curriculum(args.dataset, args.mode)


if __name__ == "__main__":
    main()
