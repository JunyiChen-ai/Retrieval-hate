"""EXP_mm_segment_keys phase 3 probe: round-0 consensus vote quality, clip vs mm.

No training. For each candidate voting space we rebuild the EXACT round-0
E-step keys via utils.consensus.build_vote_keys (the same function the
training E-step calls), vote, assign roles, and score the vote as an
annotator on MHC-EN / MHC_zh:

  * severity alignment: mean video-level vote for RAW 3-class Hateful vs
    Offensive vs Normal (+ Spearman rank corr). The clip-space EN failure
    signature is Hateful < Offensive (anti-correlated with severity).
  * positive-supervision supply: fraction of hateful train videos with ZERO
    ROLE_POS sub-clips ("all-pruned"). Gate: mm must not exceed the clip
    space's fraction (EN clip = 94/168 = 56.0%).
  * segment-ness: mean within-video std of the vote (clip EN ~= 0.05 == the
    vote was de facto video-level).
  * evidence-channel breakdown (MLLM archive modality_cues), as in
    consensus_forensics.py.

CPU-only, read-only. Usage:
  python scripts/analysis/consensus_probe_mm.py --dataset MHC
  python scripts/analysis/consensus_probe_mm.py --dataset MHC_zh
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from utils.consensus import (  # noqa: E402
    ROLE_DRIFT, ROLE_IGNORE, ROLE_POS,
    _knn_vote, assign_roles, build_vote_keys, summarize_roles,
)

EMB_DIR = "./data/CLIP_Embedding"
ARCHIVE_DIR = "./data/Archive"
MODEL = "openai_clip-vit-large-patch14-336_HF"
ARCHIVE_MODEL = "Qwen2.5-VL-7B-Instruct"
RAW_ANN = {"MHC": "/data/jehc223/Multihateclip/English/annotation(new).json",
           "MHC_zh": "/data/jehc223/Multihateclip/Chinese/annotation(new).json"}


def auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allsc = np.concatenate([pos, neg])
    order = np.argsort(allsc, kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    for v in np.unique(allsc):
        m = allsc == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def evidence_channel(archive):
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


def probe_space(name, seg_cache, train_ns, args_ns, valid, inherited_np,
                vid_labels, parents_np, video_ids, fine, arch, tau, topk):
    memory, query, space_desc = build_vote_keys(
        seg_cache, train_ns, None, args_ns)
    vote = _knn_vote(query, memory, vid_labels, parents_np, topk=topk)
    roles, margins = assign_roles(inherited_np, vote, valid, tau)
    print("\n#### space = {} [{}] ####".format(name, space_desc))
    summarize_roles(roles)
    roles = roles.numpy()

    N = len(video_ids)
    out = {"space": name, "desc": space_desc}

    # annotator quality
    out["subclip_auc"] = auc(vote[valid], inherited_np[valid])
    vid_mean = np.full(N, np.nan)
    within = []
    for i in range(N):
        m = (parents_np == i) & valid
        if m.any():
            vid_mean[i] = vote[m].mean()
        if m.sum() >= 2:
            within.append(vote[m].std())
    ok = ~np.isnan(vid_mean)
    out["video_auc"] = auc(vid_mean[ok], vid_labels[ok])
    out["within_video_std"] = float(np.mean(within))
    out["between_video_std"] = float(np.std(vid_mean[ok]))

    # severity alignment (RAW 3-class)
    sev_map = {"Normal": 0, "Offensive": 1, "Hateful": 2}
    cls_votes = defaultdict(list)
    sev_x, sev_y = [], []
    for i, vid in enumerate(video_ids):
        if not ok[i]:
            continue
        fl = fine.get(vid)
        if fl in sev_map:
            cls_votes[fl].append(vid_mean[i])
            sev_x.append(sev_map[fl])
            sev_y.append(vid_mean[i])
    from scipy.stats import spearmanr
    rho, p = spearmanr(sev_x, sev_y) if len(sev_x) > 2 else (float("nan"),) * 2
    out["severity_spearman"] = float(rho)
    out["severity_spearman_p"] = float(p)
    out["mean_vote_by_class"] = {
        c: float(np.mean(v)) for c, v in cls_votes.items()}

    # per-hateful-video pruning + channel table
    per_vid = {}
    for i, vid in enumerate(video_ids):
        if vid_labels[i] != 1:
            continue
        m = parents_np == i
        per_vid[vid] = {
            "n_pos": int((roles[m] == ROLE_POS).sum()),
            "n_drift": int((roles[m] == ROLE_DRIFT).sum()),
            "n_ignore": int((roles[m] == ROLE_IGNORE).sum()),
            "n": int(m.sum()),
            "channel": evidence_channel(arch.get(vid)),
            "fine": fine.get(vid, "<missing>"),
        }
    n_hate = len(per_vid)
    allp = sum(1 for d in per_vid.values() if d["n_pos"] == 0)
    out["hateful_videos"] = n_hate
    out["all_pruned"] = allp
    out["all_pruned_frac"] = allp / max(n_hate, 1)
    tot_sub = sum(d["n"] for d in per_vid.values())
    out["pos_pct"] = 100.0 * sum(d["n_pos"] for d in per_vid.values()) / max(tot_sub, 1)
    out["drift_pct"] = 100.0 * sum(d["n_drift"] for d in per_vid.values()) / max(tot_sub, 1)

    by_fl = defaultdict(list)
    for d in per_vid.values():
        by_fl[d["fine"]].append(d)
    out["by_fine"] = {}
    for fl, vids in sorted(by_fl.items(), key=lambda kv: -len(kv[1])):
        ns = sum(d["n"] for d in vids)
        out["by_fine"][fl] = {
            "videos": len(vids),
            "drift_pct": 100.0 * sum(d["n_drift"] for d in vids) / max(ns, 1),
            "pos_pct": 100.0 * sum(d["n_pos"] for d in vids) / max(ns, 1),
            "all_pruned": sum(1 for d in vids if d["n_pos"] == 0),
        }
    by_ch = defaultdict(list)
    for d in per_vid.values():
        by_ch[d["channel"]].append(d)
    out["by_channel"] = {}
    for ch, vids in sorted(by_ch.items(), key=lambda kv: -len(kv[1])):
        ns = sum(d["n"] for d in vids)
        out["by_channel"][ch] = {
            "videos": len(vids),
            "drift_pct": 100.0 * sum(d["n_drift"] for d in vids) / max(ns, 1),
            "pos_pct": 100.0 * sum(d["n_pos"] for d in vids) / max(ns, 1),
            "all_pruned": sum(1 for d in vids if d["n_pos"] == 0),
        }

    print("[annotator] subclip AUC {:.4f} | video AUC {:.4f} | "
          "within-vid std {:.4f} | between-vid std {:.4f}".format(
              out["subclip_auc"], out["video_auc"],
              out["within_video_std"], out["between_video_std"]))
    print("[severity]  mean vote: " + "  ".join(
        "{}={:.3f}".format(c, out["mean_vote_by_class"].get(c, float("nan")))
        for c in ("Hateful", "Offensive", "Normal"))
        + "  | spearman rho={:.3f} (p={:.3g})".format(
            out["severity_spearman"], out["severity_spearman_p"]))
    print("[supply]    hateful videos all-pruned (zero ROLE_POS): {}/{} = "
          "{:.1f}%  | pos% {:.1f} drift% {:.1f}".format(
              allp, n_hate, 100.0 * out["all_pruned_frac"],
              out["pos_pct"], out["drift_pct"]))
    for ch, d in out["by_channel"].items():
        print("  channel {:<18} videos={:<4} pos%={:>5.1f} drift%={:>5.1f} "
              "all_pruned={}/{}".format(
                  ch, d["videos"], d["pos_pct"], d["drift_pct"],
                  d["all_pruned"], d["videos"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHC")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--num_subclips", type=int, default=4)
    ap.add_argument("--weights", type=str, default="0.3,0.5,0.7",
                    help="mm_text_weight grid.")
    ap.add_argument("--empty_modes", type=str, default="parent",
                    help="Comma list of mm_empty_text modes to probe.")
    ap.add_argument("--out_json", type=str, default="auto")
    args = ap.parse_args()
    ds = args.dataset

    seg = torch.load(os.path.join(
        EMB_DIR, ds, "train_subclipK{}_{}.pt".format(args.num_subclips, MODEL)),
        map_location="cpu")
    mm_path = os.path.join(
        EMB_DIR, ds, "train_subclipK{}_mm_{}.pt".format(args.num_subclips, MODEL))
    mmc = torch.load(mm_path, map_location="cpu")
    tr = torch.load(os.path.join(
        EMB_DIR, ds, "train_{}.pt".format(MODEL)), map_location="cpu")

    video_ids = seg["video_ids"]
    tr_ids = tr["ids"][0] if len(tr["ids"]) == 1 else tr["ids"]
    assert list(tr_ids) == list(video_ids), "video id order mismatch!"
    assert torch.equal(mmc["subclip_parent"].long(), seg["subclip_parent"].long())
    assert torch.equal(mmc["subclip_img_feats"].float(),
                       seg["subclip_img_feats"].float())

    seg_cache = {
        "subclip_img_feats": seg["subclip_img_feats"].float(),
        "subclip_parent": seg["subclip_parent"].long(),
        "labels": seg["labels"].long(),
        "subclip_txt_feats": mmc["subclip_txt_feats"].float(),
        "subclip_txt_has_text": mmc["subclip_txt_has_text"].bool(),
    }
    train_ns = SimpleNamespace(
        image_feats=tr["img_feats"].float(),
        text_feats=tr["text_feats"].float(),
        labels=tr["labels"].long(),
    )
    parents_np = seg_cache["subclip_parent"].numpy()
    inherited_np = seg_cache["labels"].numpy()
    valid = (seg_cache["subclip_img_feats"].sum(dim=1) != 0).numpy()
    vid_labels = tr["labels"].long().numpy()

    n_txt = int(seg_cache["subclip_txt_has_text"].sum())
    print("== {} probe: {} videos / {} sub-clips; {} windows with ASR text "
          "({:.1f}%) ==".format(ds, len(video_ids), len(parents_np), n_txt,
                                100.0 * n_txt / len(parents_np)))

    fine = {}
    if ds in RAW_ANN and os.path.exists(RAW_ANN[ds]):
        with open(RAW_ANN[ds]) as f:
            fine = {e["Video_ID"]: e.get("Label") for e in json.load(f)}
    arch = {}
    arch_path = os.path.join(ARCHIVE_DIR, ds,
                             "train_{}_archive.jsonl".format(ARCHIVE_MODEL))
    if os.path.exists(arch_path):
        with open(arch_path) as f:
            for line in f:
                d = json.loads(line)
                arch[d["id"]] = d.get("archive")

    def mk_args(space, w=0.5, empty="parent"):
        return SimpleNamespace(
            consensus_space=space, mm_text_weight=w, mm_empty_text=empty,
            consensus_topk=args.topk, consensus_margin=args.tau,
            consensus_space_alpha=1.0)

    results = []
    results.append(probe_space(
        "clip", seg_cache, train_ns, mk_args("clip"), valid, inherited_np,
        vid_labels, parents_np, video_ids, fine, arch, args.tau, args.topk))
    for empty in [e.strip() for e in args.empty_modes.split(",") if e.strip()]:
        for w in [float(x) for x in args.weights.split(",") if x.strip()]:
            results.append(probe_space(
                "mm_w{}_{}".format(w, empty), seg_cache, train_ns,
                mk_args("mm", w, empty), valid, inherited_np, vid_labels,
                parents_np, video_ids, fine, arch, args.tau, args.topk))

    # ---- gate verdict vs the clip space ----
    clip = results[0]
    print("\n== GATE (vs clip space) ==")
    print("{:<18} {:>7} {:>7} {:>9} {:>10} {:>9} {:>8} {:>8}".format(
        "space", "Hate", "Off", "H>=O?", "allprun%", "supply?", "rho", "wv-std"))
    for r in results:
        h = r["mean_vote_by_class"].get("Hateful", float("nan"))
        o = r["mean_vote_by_class"].get("Offensive", float("nan"))
        g1 = "PASS" if h >= o else "fail"
        g2 = ("PASS" if r["all_pruned_frac"] <= clip["all_pruned_frac"] + 1e-9
              else "fail")
        r["gate_severity"] = g1
        r["gate_supply"] = g2
        print("{:<18} {:>7.3f} {:>7.3f} {:>9} {:>9.1f}% {:>9} {:>8.3f} {:>8.4f}".format(
            r["space"], h, o, g1, 100.0 * r["all_pruned_frac"], g2,
            r["severity_spearman"], r["within_video_std"]))

    out_json = args.out_json
    if out_json == "auto":
        out_json = os.path.join(os.path.dirname(__file__), "probe_out",
                                "consensus_probe_mm_{}.json".format(ds))
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"dataset": ds, "topk": args.topk, "tau": args.tau,
                   "results": results}, f, indent=2)
    print("\n[probe] JSON -> {}".format(out_json))


if __name__ == "__main__":
    main()
