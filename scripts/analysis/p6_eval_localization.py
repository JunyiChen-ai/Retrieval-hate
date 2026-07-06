#!/usr/bin/env python
"""P6 — MLLM evidence-score localization on HateClipSeg (CPU eval).

Pre-registration: research-wiki/EXP_p6_mllm_localization.md. Reuses the EXACT
harness of eval_localization_hateclipseg.py (imported read-only): same gold,
same second->window mapping, same frame/segment/within-video estimators, same
bootstrap-CI + sign-test. P6 only swaps the [V,K] window-score matrix source.

Conditions (K=30 primary): a=memory consensus-kNN (cached npz), b=MLLM P3 scores,
c=per-video rank-average(a,b), d=random(seed0), e=broadcast of b. within-video
mean-AUC is the primary metric; frame/segment AP/AUC are supporting.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np
from scipy.stats import rankdata

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))

import eval_localization_hateclipseg as L  # noqa: E402  (frozen harness)

MLLM_DIR = os.path.join(ROOT, "data/MLLM_scores/HateClipSeg")


def load_mllm_S(video_ids, K):
    """[V,K] float matrix of P3 integer scores, aligned to video_ids order.
    Missing ids / short rows -> zero row (constant -> wv-AUC 0.5, pooled low)."""
    path = os.path.join(MLLM_DIR, "test_seen_segscoreK{}_qwen.jsonl".format(K))
    by_id = {}
    n_bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sc = r.get("scores") or []
            if len(sc) != K:
                n_bad += 1
                sc = (list(sc) + [0] * K)[:K]
            by_id[str(r["id"])] = np.asarray(sc, dtype=np.float64)
    S = np.zeros((len(video_ids), K), dtype=np.float64)
    missing = 0
    for i, v in enumerate(video_ids):
        if v in by_id:
            S[i] = by_id[v]
        else:
            missing += 1
    return S, dict(path=path, n_scored=len(by_id), n_missing=missing, n_bad_len=n_bad)


def rank_average(S_a, S_b):
    """Per-video rank-average, normalised to [0,1] per video (rank/(K-1))."""
    V, K = S_a.shape
    out = np.zeros_like(S_a)
    denom = max(K - 1, 1)
    for i in range(V):
        ra = rankdata(S_a[i], method="average") - 1.0
        rb = rankdata(S_b[i], method="average") - 1.0
        out[i] = 0.5 * (ra + rb) / denom
    return out


def eval_condition(name, S, gold, vids, vid_row, K, idx_full, idx_tox,
                   with_sig=False):
    apf, aucf = L.frame_metrics(idx_full, S)
    apt, auct = L.frame_metrics(idx_tox, S)
    (aps, aucs_), nseg = L.segment_metrics(gold, vids, vid_row, K, S)
    wv, wv_n = L.within_video_auc(gold, vids, vid_row, K, S)
    rec = {
        "frame_full": {"AP": apf, "AUC": aucf},
        "frame_toxiconly": {"AP": apt, "AUC": auct},
        "segment_full": {"AP": aps, "AUC": aucs_, "n_segments": nseg},
        "within_video_meanAUC": {"mean": wv, "n_videos": wv_n},
    }
    if with_sig:
        rec["within_video_significance"] = L.wv_significance(
            gold, vids, vid_row, K, S)
    print("  {:26s} full AP {:.4f} AUC {:.4f} | tox AP {:.4f} AUC {:.4f} | "
          "seg AP {:.4f} AUC {:.4f} | wv-AUC {:.4f} ({})".format(
              name, apf, aucf, apt, auct, aps, aucs_, wv, wv_n))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", default="30")
    ap.add_argument("--mem", default="knn_hatemm_subclip",
                    help="primary memory config for conditions a & c")
    ap.add_argument("--out_json",
                    default=os.path.join(L.OUT, "results_p6_mllm_loc.json"))
    args = ap.parse_args()
    Ks = [int(x) for x in args.Ks.split(",")]
    gold = json.load(open(L.GOLD_PATH))
    results = OrderedDict()

    for K in Ks:
        q = L.load_query(K)
        vids = list(q["video_ids"])
        vid_row = {v: i for i, v in enumerate(vids)}
        assert set(vids) == set(gold.keys())
        toxic_vids = [v for v in vids
                      if any(sum(l[1:]) > 0 for _, _, l in gold[v]["segments"])]
        idx_full = L.build_seconds(gold, vids, vid_row, K)
        idx_tox = L.build_seconds(gold, toxic_vids, vid_row, K)

        # a: memory (cached npz), primary = knn_hatemm_subclip; also video-mem
        S_mem = {}
        for mem in ["knn_hatemm_subclip", "knn_hatemm_video"]:
            npz = os.path.join(L.OUT, "scores_{}_K{}.npz".format(
                mem.replace("knn_", "knn_"), K))
            npz = os.path.join(L.OUT, "scores_{}_K{}.npz".format(mem, K))
            S_mem[mem] = np.load(npz)["S"]
        S_a = S_mem[args.mem]
        # b: MLLM
        S_b, b_meta = load_mllm_S(vids, K)
        print("K={}: MLLM scored {} / {} videos (missing {}, bad-len {})".format(
            K, b_meta["n_scored"], len(vids), b_meta["n_missing"], b_meta["n_bad_len"]))
        # c: per-video rank-average(a,b)
        S_c = rank_average(S_a, S_b)
        # d: random
        S_d = np.random.RandomState(0).random(S_a.shape)
        # e: broadcast of b
        S_e = np.repeat(S_b.mean(1, keepdims=True), K, axis=1)

        conds = OrderedDict([
            ("a_" + args.mem, (S_a, True)),
            ("a_knn_hatemm_video", (S_mem["knn_hatemm_video"], True)),
            ("b_mllm", (S_b, True)),
            ("c_rankavg", (S_c, True)),
            ("d_random", (S_d, True)),
            ("e_bcast_mllm", (S_e, False)),
        ])
        res_K = OrderedDict()
        print("K={} conditions:".format(K))
        for name, (S, sig) in conds.items():
            res_K[name] = eval_condition(name, S, gold, vids, vid_row, K,
                                         idx_full, idx_tox, with_sig=sig)
        results["K{}".format(K)] = {
            "n_videos": len(vids), "n_toxic_videos": len(toxic_vids),
            "primary_mem": args.mem, "mllm_meta": b_meta, "configs": res_K,
        }
        # pre-registered verdict print
        a = res_K["a_" + args.mem]; b = res_K["b_mllm"]
        d = res_K["d_random"]; c = res_K["c_rankavg"]
        bsig = b["within_video_significance"]
        print("\n  === K={} pre-registered checks ===".format(K))
        print("  wv-AUC: a={:.4f} b={:.4f} c={:.4f} d={:.4f}".format(
            a["within_video_meanAUC"]["mean"], b["within_video_meanAUC"]["mean"],
            c["within_video_meanAUC"]["mean"], d["within_video_meanAUC"]["mean"]))
        print("  b wv-AUC CI95 [{:.4f},{:.4f}] sign-p {:.4g}  (excludes 0.5: {})".format(
            bsig["ci95"][0], bsig["ci95"][1], bsig["sign_test_p"],
            bsig["ci95"][0] > 0.5))
        print("  (1) b>a & b>d: {} | (2) CI excl .5 & p<.05: {} | "
              "AP(b)-AP(a)={:+.4f} (bar +0.176: {}) | (4) c>=max(a,b): {}".format(
                  (b["within_video_meanAUC"]["mean"] > a["within_video_meanAUC"]["mean"]
                   and b["within_video_meanAUC"]["mean"] > d["within_video_meanAUC"]["mean"]),
                  (bsig["ci95"][0] > 0.5 and bsig["sign_test_p"] < 0.05),
                  b["frame_full"]["AP"] - a["frame_full"]["AP"],
                  (b["frame_full"]["AP"] - a["frame_full"]["AP"]) >= 0.176,
                  c["within_video_meanAUC"]["mean"] >= max(
                      a["within_video_meanAUC"]["mean"], b["within_video_meanAUC"]["mean"])))

    json.dump(results, open(args.out_json, "w"), indent=1)
    print("\nwrote", args.out_json)


if __name__ == "__main__":
    main()
