#!/usr/bin/env python
"""Zero-training temporal localization on HateClipSeg via cross-dataset kNN memory.

One experiment, two novelty pillars:
  (1) span-free localization — the consensus E-step's similarity-weighted kNN
      vote (src/utils/consensus.py `_knn_vote`, read-only reuse) scores every
      temporal window; NO span supervision, NO training, NO HateClipSeg label
      enters the scoring path (gold labels are used for METRICS only).
  (2) swappable cross-dataset memory — the memory bank is another dataset's
      train set (HateMM primary, MHC contrast); swapping it is a config change,
      no retraining.

Scoring space (declared deviation from consensus.py round-0): visual-only
raw frozen-CLIP keys (query = l2n(window feat), memory = l2n(video/subclip
feat)). consensus.py concatenates a video-level text half, but HateClipSeg has
no transcripts extracted, and a per-video-constant text half contributes ZERO
within-video temporal signal by construction — for a localization eval the
visual half is the only live part. Memory-side text is dropped too so both
sides live in the same space.

Protocols (mirror EVAL_localization_hatemm.md):
  frame-level, 1 fps, second-midpoint rule; second t = [t,t+1), label from the
  gold segment containing t+0.5 (segments tile [0,D)); positive = any toxic
  flag (multi-hot idx 1..5: hateful/insulting/sexual/violence/harm).
    protocol-full      : pool seconds of ALL 395 videos.
    protocol-toxiconly : pool seconds of videos with >=1 toxic gold second.
  segment-level (native gold granularity): each kept gold segment scored by the
  duration-weighted mean of overlapping window scores; AP/AUC pooled over all
  segments. Plus: per-video mean within-video AUC (videos with both classes) —
  the sharpest "windows carry temporal info" diagnostic (any broadcast control
  scores 0.5 by construction).

Controls: video-broadcast (per-video mean window vote broadcast to all
windows) and random (seed 0). No smoothing, no calibration, no post-processing.

Outputs: cached [V,K] window-score matrices + results JSON under
scripts/analysis/loc_out_hcs/. CPU-only; resumable.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch

ROOT = "/data/jehc223/RGCL"
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.consensus import _knn_vote, _l2n  # noqa: E402  read-only import
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

CE = "openai_clip-vit-large-patch14-336_HF"
EMB = os.path.join(ROOT, "data/CLIP_Embedding")
GOLD_PATH = os.path.join(ROOT, "data/gt/HateClipSeg/gold_segments.json")
OUT = os.path.join(ROOT, "scripts/analysis/loc_out_hcs")
TOPK = 10  # consensus_topk (validated consensus config; untouched)
CLASSES = ["normal", "hateful", "insulting", "sexual", "violence", "harm"]

os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_query(K):
    p = os.path.join(EMB, "HateClipSeg", "test_seen_subclipK{}_{}.pt".format(K, CE))
    d = torch.load(p, map_location="cpu")
    assert int(d["num_subclips"]) == K
    return d


def load_memory(kind):
    """kind in {hatemm_video, hatemm_subclip, mhc_video} ->
    (feats [N,Dv] visual-only, labels [N] int). Zero rows dropped."""
    if kind == "hatemm_video":
        w = torch.load(os.path.join(EMB, "HateMM", "train_{}.pt".format(CE)),
                       map_location="cpu")
        feats, labs = w["img_feats"].float(), w["labels"].numpy().astype(np.int64)
    elif kind == "hatemm_subclip":
        s = torch.load(os.path.join(EMB, "HateMM", "train_subclipK4_{}.pt".format(CE)),
                       map_location="cpu")
        feats, labs = s["subclip_img_feats"].float(), s["labels"].numpy().astype(np.int64)
    elif kind == "mhc_video":
        w = torch.load(os.path.join(EMB, "MHC", "train_{}.pt".format(CE)),
                       map_location="cpu")
        feats, labs = w["img_feats"].float(), w["labels"].numpy().astype(np.int64)
    else:
        raise ValueError(kind)
    keep = (feats.abs().sum(1) != 0).numpy()
    return feats[torch.as_tensor(keep)], labs[keep]


# --------------------------------------------------------------------------- #
# Window scoring (zero training, zero HateClipSeg labels)
# --------------------------------------------------------------------------- #
def window_scores(q, mem_kind, K, force=False):
    """[V,K] kNN consensus-vote matrix for memory `mem_kind`; cached."""
    cache = os.path.join(OUT, "scores_knn_{}_K{}.npz".format(mem_kind, K))
    if os.path.exists(cache) and not force:
        return np.load(cache)["S"]
    feats, labs = load_memory(mem_kind)
    query = _l2n(q["subclip_img_feats"].float())
    memory = _l2n(feats)
    own = np.full(query.shape[0], -1, dtype=np.int64)  # cross-dataset: no self
    vote = _knn_vote(query, memory, labs, own, topk=TOPK)
    S = vote.reshape(len(q["video_ids"]), K)
    np.savez(cache, S=S)
    return S


# --------------------------------------------------------------------------- #
# Second-level index
# --------------------------------------------------------------------------- #
def build_seconds(gold, vids, vid_row, K):
    """Arrays over all 1-fps seconds of `vids`: video row, window idx,
    6-col multi-hot label. Seconds not covered by any kept segment are skipped
    (counted)."""
    vi, wq, mh = [], [], []
    skipped = 0
    for v in vids:
        g = gold[v]
        D = float(g["duration"])
        segs = g["segments"]
        for t in range(int(np.floor(D))):
            mid = t + 0.5
            lab = None
            for s, e, l in segs:
                if s <= mid < e:
                    lab = l
                    break
            if lab is None:
                skipped += 1
                continue
            vi.append(vid_row[v])
            wq.append(min(K - 1, int(mid * K / D)))
            mh.append(lab)
    return (np.array(vi), np.array(wq),
            np.array(mh, dtype=np.int64), skipped)


def ap_auc(lab, sc):
    if lab.sum() == 0 or lab.sum() == len(lab):
        return float("nan"), float("nan")
    return (float(average_precision_score(lab, sc)),
            float(roc_auc_score(lab, sc)))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def frame_metrics(idx, S):
    vi, wq, mh, _ = idx
    lab = (mh[:, 1:].sum(1) > 0).astype(np.int64)
    return ap_auc(lab, S[vi, wq])


def segment_metrics(gold, vids, vid_row, K, S):
    """Duration-weighted window-score mean per kept gold segment."""
    labs, scs = [], []
    for v in vids:
        g = gold[v]
        D = float(g["duration"])
        row = S[vid_row[v]]
        bounds = [(k * D / K, (k + 1) * D / K) for k in range(K)]
        for s, e, l in g["segments"]:
            num, den = 0.0, 0.0
            for k, (ws, we) in enumerate(bounds):
                ov = max(0.0, min(e, we) - max(s, ws))
                num += ov * row[k]
                den += ov
            if den <= 0:
                continue
            labs.append(1 if sum(l[1:]) > 0 else 0)
            scs.append(num / den)
    return ap_auc(np.array(labs), np.array(scs)), len(labs)


def within_video_auc(gold, vids, vid_row, K, S):
    """Mean per-video 1-fps AUC over videos with both classes (temporal-info
    diagnostic; broadcast controls = 0.5 by construction)."""
    aucs = []
    for v in vids:
        g = gold[v]
        D = float(g["duration"])
        lab, sc = [], []
        for t in range(int(np.floor(D))):
            mid = t + 0.5
            for s, e, l in g["segments"]:
                if s <= mid < e:
                    lab.append(1 if sum(l[1:]) > 0 else 0)
                    q = min(K - 1, int(mid * K / D))
                    sc.append(S[vid_row[v], q])
                    break
        lab = np.array(lab)
        if len(lab) == 0 or lab.sum() in (0, len(lab)):
            continue
        sc = np.array(sc)
        if np.allclose(sc, sc[0]):
            aucs.append(0.5)
        else:
            aucs.append(float(roc_auc_score(lab, sc)))
    return float(np.mean(aucs)), len(aucs)


def class_slices(idx, S):
    """Per-class frame-level AP/AUC: positives = seconds of class c, negatives
    = normal-only seconds (other-toxic seconds excluded from the slice)."""
    vi, wq, mh, _ = idx
    sc_all = S[vi, wq]
    normal_only = mh[:, 1:].sum(1) == 0
    out = {}
    for c in range(1, 6):
        pos = mh[:, c] == 1
        keep = pos | normal_only
        lab = pos[keep].astype(np.int64)
        ap, auc = ap_auc(lab, sc_all[keep])
        out[CLASSES[c]] = {"AP": ap, "AUC": auc,
                           "pos_seconds": int(lab.sum()),
                           "neg_seconds": int(len(lab) - lab.sum())}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--Ks", default="4,30")
    ap_.add_argument("--force", action="store_true")
    args = ap_.parse_args()
    Ks = [int(x) for x in args.Ks.split(",")]

    gold = json.load(open(GOLD_PATH))
    results = OrderedDict()

    for K in Ks:
        q = load_query(K)
        vids = list(q["video_ids"])
        vid_row = {v: i for i, v in enumerate(vids)}
        assert set(vids) == set(gold.keys())
        # decode failures -> all-zero rows
        Z = q["subclip_img_feats"].float().reshape(len(vids), K, -1)
        dead = [vids[i] for i in range(len(vids))
                if Z[i].abs().sum() == 0]
        if dead:
            print("[WARN] K={} undecodable videos (kept, score=vote(0)=const): {}"
                  .format(K, dead))

        toxic_vids = [v for v in vids
                      if any(sum(l[1:]) > 0 for _, _, l in gold[v]["segments"])]
        idx_full = build_seconds(gold, vids, vid_row, K)
        idx_tox = build_seconds(gold, toxic_vids, vid_row, K)
        lab_full = (idx_full[2][:, 1:].sum(1) > 0)
        print("K={}: videos={} toxic-videos={} seconds={} (skipped {}) "
              "prevalence={:.4f}".format(
                  K, len(vids), len(toxic_vids), len(lab_full),
                  idx_full[3], lab_full.mean()))

        S_by_cfg = OrderedDict()
        for mem in ["hatemm_video", "hatemm_subclip", "mhc_video"]:
            S_by_cfg["knn_" + mem] = window_scores(q, mem, K, args.force)
        Sp = S_by_cfg["knn_hatemm_video"]
        S_by_cfg["vbcast_hatemm_video"] = np.repeat(
            Sp.mean(1, keepdims=True), K, axis=1)
        rng = np.random.RandomState(0)
        S_by_cfg["random"] = rng.random(Sp.shape)

        res_K = OrderedDict()
        for name, S in S_by_cfg.items():
            apf, aucf = frame_metrics(idx_full, S)
            apt, auct = frame_metrics(idx_tox, S)
            (aps, aucs_), nseg = segment_metrics(gold, vids, vid_row, K, S)
            wv, wv_n = within_video_auc(gold, vids, vid_row, K, S)
            res_K[name] = {
                "frame_full": {"AP": apf, "AUC": aucf},
                "frame_toxiconly": {"AP": apt, "AUC": auct},
                "segment_full": {"AP": aps, "AUC": aucs_, "n_segments": nseg},
                "within_video_meanAUC": {"mean": wv, "n_videos": wv_n},
            }
            print("  {:24s} full AP {:.4f} AUC {:.4f} | toxOnly AP {:.4f} "
                  "AUC {:.4f} | seg AP {:.4f} AUC {:.4f} | wv-AUC {:.4f} ({})"
                  .format(name, apf, aucf, apt, auct, aps, aucs_, wv, wv_n))

        # slices on the primary config
        for cfg in ["knn_hatemm_video", "vbcast_hatemm_video", "random"]:
            S = S_by_cfg[cfg]
            plat = {}
            for p in ["bit", "yt"]:
                pv = [v for v in vids if gold[v]["platform"] == p]
                idx_p = build_seconds(gold, pv, vid_row, K)
                apf, aucf = frame_metrics(idx_p, S)
                lab_p = (idx_p[2][:, 1:].sum(1) > 0)
                plat[p] = {"AP": apf, "AUC": aucf, "videos": len(pv),
                           "prevalence": float(lab_p.mean())}
            res_K[cfg]["platform_full"] = plat
            res_K[cfg]["class_slices_full"] = class_slices(idx_full, S)

        results["K{}".format(K)] = {
            "n_videos": len(vids), "n_toxic_videos": len(toxic_vids),
            "n_seconds_full": int(len(lab_full)),
            "prevalence_full": float(lab_full.mean()),
            "skipped_seconds": int(idx_full[3]),
            "undecodable_videos": dead,
            "configs": res_K,
        }

    outp = os.path.join(OUT, "results_hateclipseg_loc.json")
    json.dump(results, open(outp, "w"), indent=1)
    print("\nwrote", outp)


if __name__ == "__main__":
    main()
