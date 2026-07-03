"""W2 subtask 3: forensics on consensus round-0 pruned segments (MHC-EN focus).

For every train sub-clip we recompute the round-0 consensus E-step EXACTLY as
src/utils/consensus.py::consensus_estep(model=None) does (raw frozen-CLIP space,
own-parent excluded, similarity-weighted kNN vote), keeping the raw vote value.
We then join the hateful-parent sub-clips that the consensus PRUNES from
positive supervision (ROLE_DRIFT: video label hateful, neighbours confidently
vote benign; plus low-margin ROLE_IGNORE) against the E0b MLLM archives
(data/Archive/<DS>/train_*_archive.jsonl) and ask: is the hate of the affected
videos carried by speech / on-screen-text evidence (invisible or weakly visible
to the frame-only sub-clip visual embedding) rather than visual evidence?

CPU-only, read-only: does NOT modify anything under src/ or data/.

Usage:
  python scripts/analysis/consensus_forensics.py --dataset MHC
  python scripts/analysis/consensus_forensics.py --dataset MHC_zh --cases 0
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from utils.consensus import (  # noqa: E402
    ROLE_CONFLICT, ROLE_DRIFT, ROLE_IGNORE, ROLE_NEG, ROLE_POS, ROLE_NAMES,
    _knn_vote, _l2n, assign_roles, summarize_roles,
)

EMB_DIR = "./data/CLIP_Embedding"
ARCHIVE_DIR = "./data/Archive"
MODEL = "openai_clip-vit-large-patch14-336_HF"
ARCHIVE_MODEL = "Qwen2.5-VL-7B-Instruct"


def auc(scores, labels):
    """Rank AUC without sklearn."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allsc = np.concatenate([pos, neg])
    for v in np.unique(allsc):
        m = allsc == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def evidence_channel(archive):
    """Classify where the (hateful) evidence lives, from MLLM modality_cues."""
    if not archive:
        return "no_archive"
    mc = archive.get("modality_cues") or {}
    vis = bool((mc.get("visual") or "").strip())
    spe = bool((mc.get("speech") or "").strip())
    ost = bool((mc.get("on_screen_text") or "").strip())
    if (spe or ost) and not vis:
        return "speech/text-only"
    if vis and not (spe or ost):
        return "visual-only"
    if vis and (spe or ost):
        return "mixed"
    return "no-cues"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHC")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--num_subclips", type=int, default=4)
    ap.add_argument("--cases", type=int, default=10)
    args = ap.parse_args()
    ds = args.dataset

    # ---- load caches (read-only) ----
    seg = torch.load(os.path.join(
        EMB_DIR, ds, "train_subclipK{}_{}.pt".format(args.num_subclips, MODEL)),
        map_location="cpu")
    tr = torch.load(os.path.join(
        EMB_DIR, ds, "train_{}.pt".format(MODEL)), map_location="cpu")

    sub_img = seg["subclip_img_feats"].float()          # [S, Dv]
    parents = seg["subclip_parent"].long()              # [S]
    inherited = seg["labels"].long()                    # [S]
    video_ids = seg["video_ids"]                        # [N]
    vid_img = tr["img_feats"].float()                   # [N, Dv]
    vid_txt = tr["text_feats"].float()                  # [N, Dt]
    vid_labels = tr["labels"].long().numpy()

    # sanity: subclip cache and whole-video cache must be row-aligned
    tr_ids = tr["ids"][0] if len(tr["ids"]) == 1 else tr["ids"]
    assert list(tr_ids) == list(video_ids), "video id order mismatch!"

    # ---- round-0 consensus E-step, bitwise identical to consensus_estep(model=None)
    parent_txt = vid_txt.index_select(0, parents)
    memory = _l2n(torch.cat([_l2n(vid_img), _l2n(vid_txt)], dim=1))
    query = _l2n(torch.cat([_l2n(sub_img), _l2n(parent_txt)], dim=1))
    vote = _knn_vote(query, memory, vid_labels, parents.numpy(), topk=args.topk)
    valid = (sub_img.sum(dim=1) != 0).numpy()
    roles, margins = assign_roles(inherited.numpy(), vote, valid, args.tau)
    print("== {} round-0 consensus (raw-CLIP, topk={}, tau={}) ==".format(
        ds, args.topk, args.tau))
    summarize_roles(roles, tag=" [recomputed; must match E1 job log]")
    roles = roles.numpy()

    # ---- retrieval-annotator quality ----
    sub_auc = auc(vote[valid], inherited.numpy()[valid])
    vid_vote = np.array([vote[(parents.numpy() == i) & valid].mean()
                         if ((parents.numpy() == i) & valid).any() else 0.5
                         for i in range(len(video_ids))])
    vid_auc = auc(vid_vote, vid_labels)
    print("[annotator quality] subclip-vote vs inherited label AUC = {:.4f} | "
          "video mean-vote vs video label AUC = {:.4f}".format(sub_auc, vid_auc))
    n_hate_vid = int((vid_labels == 1).sum())
    print("[train] {} videos ({} hateful), {} sub-clips".format(
        len(video_ids), n_hate_vid, len(roles)))

    # ---- is the vote segment-level or effectively video-level? ----
    pn0 = parents.numpy()
    within_std, vid_means = [], []
    for i in range(len(video_ids)):
        m = (pn0 == i) & valid
        if m.sum() >= 2:
            within_std.append(vote[m].std())
        if m.any():
            vid_means.append(vote[m].mean())
    print("[vote dispersion] mean WITHIN-video std = {:.4f} | "
          "BETWEEN-video std of per-video mean vote = {:.4f}".format(
              float(np.mean(within_std)), float(np.std(vid_means))))
    hate_rows = [i for i in range(len(video_ids)) if vid_labels[i] == 1]
    ben_rows = [i for i in range(len(video_ids)) if vid_labels[i] == 0]
    hv = np.array([vote[(pn0 == i) & valid].mean() for i in hate_rows
                   if ((pn0 == i) & valid).any()])
    bv = np.array([vote[(pn0 == i) & valid].mean() for i in ben_rows
                   if ((pn0 == i) & valid).any()])
    print("[vote level] mean video-vote: hateful={:.3f} benign={:.3f} "
          "(threshold 0.5, tau band +-{})".format(hv.mean(), bv.mean(), args.tau / 2))

    # ---- archives ----
    arch_path = os.path.join(ARCHIVE_DIR, ds,
                             "train_{}_archive.jsonl".format(ARCHIVE_MODEL))
    arch = {}
    if os.path.exists(arch_path):
        with open(arch_path) as f:
            for line in f:
                d = json.loads(line)
                arch[d["id"]] = d.get("archive")
    else:
        print("[WARN] no archive file at", arch_path)

    # ---- per-video role aggregation for hateful videos ----
    pn = parents.numpy()
    per_vid = {}
    for i, vid in enumerate(video_ids):
        if vid_labels[i] != 1:
            continue
        m = pn == i
        per_vid[vid] = {
            "row": i,
            "roles": roles[m],
            "votes": vote[m],
            "margins": margins.numpy()[m],
            "n_pos": int((roles[m] == ROLE_POS).sum()),
            "n_drift": int((roles[m] == ROLE_DRIFT).sum()),
            "n_ignore": int((roles[m] == ROLE_IGNORE).sum()),
            "channel": evidence_channel(arch.get(vid)),
            "archive": arch.get(vid),
        }

    # ---- table: evidence channel x pruning ----
    print("\n== hateful train videos: consensus pruning by MLLM evidence channel ==")
    header = ("channel", "videos", "subclips", "pos", "drift", "ignore",
              "drift%", "pos%", "vid_all_pruned")
    print("{:<18} {:>6} {:>8} {:>5} {:>6} {:>7} {:>7} {:>6} {:>14}".format(*header))
    by_ch = defaultdict(list)
    for vid, d in per_vid.items():
        by_ch[d["channel"]].append(d)
    totals = Counter()
    for ch in sorted(by_ch, key=lambda c: -len(by_ch[c])):
        vids = by_ch[ch]
        n_sub = sum(len(d["roles"]) for d in vids)
        n_pos = sum(d["n_pos"] for d in vids)
        n_dr = sum(d["n_drift"] for d in vids)
        n_ig = sum(d["n_ignore"] for d in vids)
        allpruned = sum(1 for d in vids if d["n_pos"] == 0)
        print("{:<18} {:>6} {:>8} {:>5} {:>6} {:>7} {:>6.1f}% {:>5.1f}% {:>10}/{:<3}".format(
            ch, len(vids), n_sub, n_pos, n_dr, n_ig,
            100.0 * n_dr / max(n_sub, 1), 100.0 * n_pos / max(n_sub, 1),
            allpruned, len(vids)))
        totals.update({"vids": len(vids), "sub": n_sub, "pos": n_pos,
                       "dr": n_dr, "ig": n_ig, "allp": allpruned})
    print("{:<18} {:>6} {:>8} {:>5} {:>6} {:>7} {:>6.1f}% {:>5.1f}% {:>10}/{:<3}".format(
        "TOTAL", totals["vids"], totals["sub"], totals["pos"], totals["dr"],
        totals["ig"], 100.0 * totals["dr"] / max(totals["sub"], 1),
        100.0 * totals["pos"] / max(totals["sub"], 1),
        totals["allp"], totals["vids"]))

    # ---- table: mechanism x pruning (a video can have several mechanisms) ----
    print("\n== hateful train videos: drift/pos rate by MLLM mechanism ==")
    by_mech = defaultdict(list)
    for vid, d in per_vid.items():
        mechs = (d["archive"] or {}).get("mechanism") or ["<none>"]
        for mch in mechs:
            by_mech[mch].append(d)
    for mch in sorted(by_mech, key=lambda m: -len(by_mech[m])):
        vids = by_mech[mch]
        n_sub = sum(len(d["roles"]) for d in vids)
        n_pos = sum(d["n_pos"] for d in vids)
        n_dr = sum(d["n_drift"] for d in vids)
        print("  {:<28} videos={:<4} drift%={:>5.1f}  pos%={:>5.1f}".format(
            mch, len(vids), 100.0 * n_dr / max(n_sub, 1),
            100.0 * n_pos / max(n_sub, 1)))

    # ---- fine 3-class MultiHateClip label (Hateful vs Offensive), if available ----
    RAW_ANN = {"MHC": "/data/jehc223/Multihateclip/English/annotation(new).json",
               "MHC_zh": "/data/jehc223/Multihateclip/Chinese/annotation(new).json"}
    if ds in RAW_ANN and os.path.exists(RAW_ANN[ds]):
        with open(RAW_ANN[ds]) as f:
            fine = {e["Video_ID"]: e.get("Label") for e in json.load(f)}
        print("\n== hateful(binary=1) train videos: pruning by RAW 3-class label ==")
        by_fl = defaultdict(list)
        for vid, d in per_vid.items():
            by_fl[fine.get(vid, "<missing>")].append(d)
        for fl in sorted(by_fl, key=lambda x: -len(by_fl[x])):
            vids = by_fl[fl]
            n_sub = sum(len(d["roles"]) for d in vids)
            n_pos = sum(d["n_pos"] for d in vids)
            n_dr = sum(d["n_drift"] for d in vids)
            allp = sum(1 for d in vids if d["n_pos"] == 0)
            mv = np.mean([d["votes"].mean() for d in vids])
            print("  {:<12} videos={:<4} drift%={:>5.1f}  pos%={:>5.1f}  "
                  "all_pruned={}/{}  mean-vote={:.3f}".format(
                      fl, len(vids), 100.0 * n_dr / max(n_sub, 1),
                      100.0 * n_pos / max(n_sub, 1), allp, len(vids), mv))

    # ---- explicitness ----
    print("\n== hateful train videos: drift/pos rate by explicitness ==")
    by_ex = defaultdict(list)
    for vid, d in per_vid.items():
        by_ex[(d["archive"] or {}).get("explicitness") or "<none>"].append(d)
    for ex in sorted(by_ex, key=lambda e: -len(by_ex[e])):
        vids = by_ex[ex]
        n_sub = sum(len(d["roles"]) for d in vids)
        n_pos = sum(d["n_pos"] for d in vids)
        n_dr = sum(d["n_drift"] for d in vids)
        print("  {:<12} videos={:<4} drift%={:>5.1f}  pos%={:>5.1f}".format(
            ex, len(vids), 100.0 * n_dr / max(n_sub, 1),
            100.0 * n_pos / max(n_sub, 1)))

    # ---- benign side: conflict rate (for the hardneg variant context) ----
    ben_mask = (inherited.numpy() == 0) & valid
    n_conf = int((roles[ben_mask] == ROLE_CONFLICT).sum())
    print("\n[benign side] conflict subclips = {} / {} ({:.1f}%)".format(
        n_conf, int(ben_mask.sum()), 100.0 * n_conf / max(int(ben_mask.sum()), 1)))

    # ---- cases ----
    if args.cases > 0:
        print("\n== top-{} hateful videos most aggressively pruned at round 0 ==".format(
            args.cases))
        ranked = sorted(per_vid.items(),
                        key=lambda kv: (-kv[1]["n_drift"], -kv[1]["n_ignore"],
                                        kv[1]["votes"].mean()))
        for vid, d in ranked[: args.cases]:
            a = d["archive"] or {}
            mc = a.get("modality_cues") or {}
            print("\n- video {} (label=hateful) roles={} votes={}".format(
                vid, [ROLE_NAMES[r] for r in d["roles"]],
                np.round(d["votes"], 3).tolist()))
            print("  channel={} | mechanism={} | targets={} | explicitness={}".format(
                d["channel"], a.get("mechanism"), a.get("target_groups"),
                a.get("explicitness")))
            for key, lab in (("visual", "visual"), ("speech", "speech"),
                             ("on_screen_text", "on-screen-text")):
                cue = (mc.get(key) or "").strip()
                if cue:
                    print("  {} cue: {}".format(lab, cue[:180]))
            summ = (a.get("neutral_summary") or "").strip()
            if summ:
                print("  summary: {}".format(summ[:220]))


if __name__ == "__main__":
    main()
