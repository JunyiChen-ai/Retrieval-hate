#!/usr/bin/env python
"""Segment-level localization scoring for the consensus denoising method on HateMM.

Reuses (read-only) src/utils/consensus.py `_knn_vote` / `_l2n` and
src/model/classifier.py `classifier_hateClipper`. Nothing under src/ is modified.

Localization protocol (fixed by the team lead):
  * 1 fps frames; second t owns the interval [t, t+1); its label is 1 iff the
    second-midpoint (t+0.5) falls in ANY gold hate span; the final sub-1s tail
    is dropped (num_seconds = floor(duration)); duration comes from hate_spans.json.
  * protocol-full   : pool the seconds of ALL 215 test videos.
  * protocol-hateonly: pool the seconds of the 85 hateful videos with non-empty spans.
  * metrics: AP (sklearn average_precision_score) and ROC-AUC; no per-video
    averaging, no IoU, no smoothing.

Sub-clip -> time mapping (K=4): the sub-clip cache uniformly samples M=16 frames
across the whole video (frame j at video-fraction j/15) and mean-pools frames
[4k,4k+4) into sub-clip k. Sub-clip k's information is therefore centred at
video-fraction (4k+1.5)/15 (= 0.10, 0.367, 0.633, 0.90). We broadcast each
sub-clip score to a contiguous equal quarter of the duration: a second with
midpoint m is owned by sub-clip q = min(K-1, floor(K*m/duration)) (quarter
centres 0.125, 0.375, 0.625, 0.875 -- an excellent match to the sub-clip
information centres). This tiles the whole video with no gaps.

Scored configurations (each reported under both protocols):
  consensus_vote_video   : training-free raw-CLIP kNN consensus vote, memory =
                           whole-video TRAIN embeddings (faithful to consensus.py).
  consensus_vote_subclip : same vote but memory = TRAIN sub-clip embeddings with
                           inherited video labels (faithful to the brief's literal
                           "train sub-clip library").
  model_consensus        : sigmoid hate score of the trained consensus head
                           (EM round-2 val-selected, epoch 18) on each test sub-clip.
  model_selfscore        : same, self-score control head.
  vbcast_model           : whole-video sigmoid of the consensus head, broadcast to
                           every second (honest video-level-separability control).
  vbcast_consensus_mean  : per-video mean of consensus_vote_video, broadcast.
  random                 : seed-0 uniform per-second scores (sanity floor).

Resumable: per-config sub-clip scores are cached under loc_out/; rerun reuses them.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = "/data/jehc223/RGCL"
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.consensus import _knn_vote, _l2n          # noqa: E402  read-only import
from model.classifier import classifier_hateClipper  # noqa: E402  read-only import
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

CE = "openai_clip-vit-large-patch14-336_HF"
EMB = os.path.join(ROOT, "data/CLIP_Embedding/HateMM")
GOLD_PATH = os.path.join(ROOT, "data/gt/HateMM/hate_spans.json")
OUT = os.path.join(ROOT, "scripts/analysis/loc_out")
CKPT_CONSENSUS = os.path.join(OUT, "ckpt_consensus_emr1_ep18.pt")
CKPT_SELFSCORE = os.path.join(OUT, "ckpt_selfscore_emr1.pt")
TOPK = 10           # consensus_topk (validated config)
K = 4               # num_subclips

os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_all():
    gold = json.load(open(GOLD_PATH))
    tr_w = torch.load(f"{EMB}/train_{CE}.pt", map_location="cpu")
    te_w = torch.load(f"{EMB}/test_seen_{CE}.pt", map_location="cpu")
    tr_s = torch.load(f"{EMB}/train_subclipK{K}_{CE}.pt", map_location="cpu")
    te_s = torch.load(f"{EMB}/test_seen_subclipK{K}_{CE}.pt", map_location="cpu")
    d = dict(
        gold=gold,
        tr_ids=list(tr_w["ids"][0]), te_ids=list(te_w["ids"][0]),
        tr_img=tr_w["img_feats"].float(), tr_txt=tr_w["text_feats"].float(),
        tr_lab=tr_w["labels"].numpy().astype(np.int64),
        te_img=te_w["img_feats"].float(), te_txt=te_w["text_feats"].float(),
        te_vids=list(te_s["video_ids"]),
        te_sub_img=te_s["subclip_img_feats"].float(),
        te_sub_par=te_s["subclip_parent"].numpy().astype(np.int64),
        tr_vids=list(tr_s["video_ids"]),
        tr_sub_img=tr_s["subclip_img_feats"].float(),
        tr_sub_par=tr_s["subclip_parent"].numpy().astype(np.int64),
        tr_sub_lab=tr_s["labels"].numpy().astype(np.int64),
    )
    assert d["te_vids"] == d["te_ids"], "test sub-clip / whole-video id order mismatch"
    assert d["tr_vids"] == d["tr_ids"], "train sub-clip / whole-video id order mismatch"
    return d


# --------------------------------------------------------------------------- #
# Sub-clip scorers -> [V, K] matrices (rows aligned to te_ids)
# --------------------------------------------------------------------------- #
def _reshape_vk(vec, V):
    return np.asarray(vec, dtype=np.float64).reshape(V, K)


def score_consensus_vote(d, memory_kind):
    """Raw-CLIP kNN consensus vote for every test sub-clip. memory_kind in
    {'video','subclip'}. Query key = l2n([l2n(sub_img) | l2n(parent_text)])."""
    te_par = torch.as_tensor(d["te_sub_par"])
    q = _l2n(torch.cat([_l2n(d["te_sub_img"]),
                        _l2n(d["te_txt"].index_select(0, te_par))], dim=1))
    if memory_kind == "video":
        mem = _l2n(torch.cat([_l2n(d["tr_img"]), _l2n(d["tr_txt"])], dim=1))
        mem_lab = d["tr_lab"]
    else:  # subclip memory with inherited video labels; drop zero-vector clips
        keep = d["tr_sub_img"].abs().sum(1) != 0
        tr_par = torch.as_tensor(d["tr_sub_par"])
        mem = _l2n(torch.cat([_l2n(d["tr_sub_img"]),
                              _l2n(d["tr_txt"].index_select(0, tr_par))], dim=1))
        mem, mem_lab = mem[keep], d["tr_sub_lab"][keep.numpy()]
    own = np.full(q.shape[0], -1, dtype=np.int64)  # no self among train memory
    vote = _knn_vote(q, mem, mem_lab, own, topk=TOPK)
    return _reshape_vk(vote, len(d["te_ids"]))


def _build_head():
    m = classifier_hateClipper(
        image_dim=1024, text_dim=768, num_layers=3, proj_dim=1024, map_dim=1024,
        fusion_mode="align", dropout=[0.2, 0.4, 0.1], batch_norm=False,
        args=SimpleNamespace(dataset="HateMM"))
    return m


def score_model(d, ckpt):
    """Sigmoid hate score of the trained head on each test sub-clip; also returns
    the whole-video sigmoid score for the broadcast control."""
    m = _build_head()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    te_par = torch.as_tensor(d["te_sub_par"])
    sub_txt = d["te_txt"].index_select(0, te_par)
    with torch.no_grad():
        logit, _ = m(d["te_sub_img"], sub_txt, return_embed=True)
        sub_p = torch.sigmoid(logit.reshape(-1)).numpy()
        vlogit, _ = m(d["te_img"], d["te_txt"], return_embed=True)
        vid_p = torch.sigmoid(vlogit.reshape(-1)).numpy()
    return _reshape_vk(sub_p, len(d["te_ids"])), vid_p


def get_scores(d, name, force=False):
    """Return [V,K] sub-clip score matrix for `name`, caching to loc_out/."""
    cache = os.path.join(OUT, f"scores_{name}.npz")
    if os.path.exists(cache) and not force:
        z = np.load(cache, allow_pickle=True)
        return z["S"], (z["vid"] if "vid" in z.files else None)
    vid = None
    if name == "consensus_vote_video":
        S = score_consensus_vote(d, "video")
    elif name == "consensus_vote_subclip":
        S = score_consensus_vote(d, "subclip")
    elif name == "model_consensus":
        S, vid = score_model(d, CKPT_CONSENSUS)
    elif name == "model_selfscore":
        S, vid = score_model(d, CKPT_SELFSCORE)
    else:
        raise ValueError(name)
    np.savez(cache, S=S, vid=(vid if vid is not None else np.array([])))
    return S, vid


# --------------------------------------------------------------------------- #
# Second-level index + evaluation
# --------------------------------------------------------------------------- #
def build_index(videos, d):
    """Return arrays (vi, q, label) at 1 fps for the given video-id list."""
    gold, te_ids = d["gold"], d["te_ids"]
    row_vi, row_q, row_lab = [], [], []
    for vid in videos:
        vi = te_ids.index(vid)
        D = float(gold[vid]["duration"])
        spans = gold[vid]["spans"]
        n = int(np.floor(D))
        for t in range(n):
            mid = t + 0.5
            q = min(K - 1, int(mid * K / D))
            lab = 1 if any(s <= mid < e for (s, e) in spans) else 0
            row_vi.append(vi); row_q.append(q); row_lab.append(lab)
    return (np.array(row_vi), np.array(row_q), np.array(row_lab, dtype=np.int64))


def eval_config(idx, S):
    vi, q, lab = idx
    scores = S[vi, q]
    return average_precision_score(lab, scores), roc_auc_score(lab, scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    d = load_all()
    te_ids = d["te_ids"]
    gold = d["gold"]

    full_vids = list(te_ids)
    hate_vids = [v for v in te_ids
                 if gold[v].get("label") == 1 and len(gold[v]["spans"]) > 0]

    idx_full = build_index(full_vids, d)
    idx_hate = build_index(hate_vids, d)

    def stats(idx, vids):
        _, _, lab = idx
        return len(vids), len(lab), int(lab.sum()), lab.mean()
    print("=== dataset stats ===")
    for tag, idx, vids in [("protocol-full", idx_full, full_vids),
                           ("protocol-hateonly", idx_hate, hate_vids)]:
        V, sec, pos, prev = stats(idx, vids)
        print(f"  {tag:18s}: videos={V:4d}  seconds={sec:6d}  "
              f"hate_seconds={pos:5d}  prevalence={prev:.4f}")

    # base sub-clip score matrices
    Sv, _ = get_scores(d, "consensus_vote_video", args.force)
    Ss, _ = get_scores(d, "consensus_vote_subclip", args.force)
    Smc, vidc = get_scores(d, "model_consensus", args.force)
    Sms, _ = get_scores(d, "model_selfscore", args.force)

    V = len(te_ids)
    # derived broadcast controls -> [V,K] with equal columns
    S_vb_model = np.repeat(vidc.reshape(V, 1), K, axis=1)
    S_vb_consmean = np.repeat(Sv.mean(axis=1, keepdims=True), K, axis=1)

    configs = [
        ("consensus_vote_video", Sv),
        ("consensus_vote_subclip", Ss),
        ("model_consensus", Smc),
        ("model_selfscore", Sms),
        ("vbcast_model", S_vb_model),
        ("vbcast_consensus_mean", S_vb_consmean),
    ]

    print("\n=== localization results (AP / ROC-AUC) ===")
    hdr = f"{'config':26s} {'full-AP':>8s} {'full-AUC':>9s} {'hate-AP':>8s} {'hate-AUC':>9s}"
    print(hdr); print("-" * len(hdr))
    for name, S in configs:
        ap_f, auc_f = eval_config(idx_full, S)
        ap_h, auc_h = eval_config(idx_hate, S)
        print(f"{name:26s} {ap_f:8.4f} {auc_f:9.4f} {ap_h:8.4f} {auc_h:9.4f}")

    # random sanity floor (per-second uniform, seed 0)
    for tag, idx in [("random(full)", idx_full), ("random(hateonly)", idx_hate)]:
        _, _, lab = idx
        rs = np.random.RandomState(0).random(len(lab))
        ap_r = average_precision_score(lab, rs)
        auc_r = roc_auc_score(lab, rs)
        pre = "full" if "full" in tag else "hate"
        print(f"{'random':26s} "
              + (f"{ap_r:8.4f} {auc_r:9.4f} {'':>8s} {'':>9s}" if pre == "full"
                 else f"{'':>8s} {'':>9s} {ap_r:8.4f} {auc_r:9.4f}"))

    print(f"\ncaches: {OUT}/scores_*.npz")


if __name__ == "__main__":
    main()
