#!/usr/bin/env python3
"""Evaluate a trained MultiHateLoc checkpoint on the HateMM test split.

Produces:
  (a) video-level detection: acc / macro-F1 / ROC-AUC  (comparable to our main table)
  (b) frame-level localization: per-second hate scores exported to scores.npz, and
      -- if data/gt/HateMM/hate_spans.json exists -- frame-level mAP (average
      precision) and AUC (ROC-AUC) under the default protocol: a second is
      positive iff its midpoint (t+0.5) lies in any annotated hate span. Scores
      from all test videos are pooled ("all") and hateful-only ("hateful").
"""
import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             average_precision_score)

from dataset import HateMMFeats, collate, read_split
from model import MultiHateLoc, video_score

SPANS = "/data/jehc223/RGCL/data/gt/HateMM/hate_spans.json"


@torch.no_grad()
def run(model, loader, device, mods, k_div):
    model.eval()
    per_video = {}          # vid -> per-second np.array of fused scores
    vids_order, ys, vscore = [], [], []
    for batch in loader:
        if "v" not in mods: batch["fv"] = torch.zeros_like(batch["fv"])
        if "a" not in mods: batch["fa"] = torch.zeros_like(batch["fa"])
        if "t" not in mods: batch["ft"] = torch.zeros_like(batch["ft"])
        fv, fa, ft = batch["fv"].to(device), batch["fa"].to(device), batch["ft"].to(device)
        mask = batch["mask"].to(device)
        out = model(fv, fa, ft, mask)
        vs = video_score(out["p_fused"], mask, k_div).cpu().numpy()
        pf = out["p_fused"].cpu().numpy()
        for i, vid in enumerate(batch["vids"]):
            T = batch["lens"][i]
            per_video[vid] = pf[i, :T].astype(np.float32)
            vids_order.append(vid); vscore.append(float(vs[i])); ys.append(int(batch["y"][i]))
    return per_video, vids_order, np.array(ys), np.array(vscore)


def frame_level(per_video, labels_map, spans):
    """Build pooled per-second (score,label) arrays and compute mAP/AUC."""
    def collect(only_hateful):
        S, L = [], []
        used = 0
        for vid, scores in per_video.items():
            y = labels_map.get(vid, 0)
            if only_hateful and y == 0:
                continue
            T = len(scores)
            lab = np.zeros(T, dtype=np.int32)
            if vid in spans:
                for (s, e) in spans[vid].get("spans", []):
                    for t in range(T):
                        if s <= (t + 0.5) < e:
                            lab[t] = 1
            elif y == 1:
                # hateful video with no span entry -> cannot label frames; skip
                continue
            S.append(scores); L.append(lab); used += 1
        if not S:
            return None
        S = np.concatenate(S); L = np.concatenate(L)
        if L.sum() == 0 or L.sum() == len(L):
            return {"n_videos": used, "n_frames": int(len(L)),
                    "n_pos": int(L.sum()), "mAP": float("nan"), "AUC": float("nan")}
        return {"n_videos": used, "n_frames": int(len(L)), "n_pos": int(L.sum()),
                "mAP": float(average_precision_score(L, S)),
                "AUC": float(roc_auc_score(L, S))}
    return {"all": collect(False), "hateful": collect(True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k_div", type=int, default=3)
    ap.add_argument("--max_t", type=int, default=0, help="0 = full length (localization)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    d_model = ckpt.get("args", {}).get("d_model", 256)
    mods = ckpt.get("args", {}).get("modalities", "vat")
    model = MultiHateLoc(d_model=d_model).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch')}, mods={mods})", flush=True)

    ds = HateMMFeats(args.split, max_t=args.max_t)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate)
    per_video, vids, ys, vscore = run(model, loader, device, mods, args.k_div)

    # (a) video-level
    pred = (vscore >= 0.5).astype(int)
    vlevel = {"acc": float(accuracy_score(ys, pred)),
              "macro_f1": float(f1_score(ys, pred, average="macro")),
              "auc": float(roc_auc_score(ys, vscore)),
              "n": int(len(ys))}
    print("VIDEO-LEVEL:", json.dumps(vlevel), flush=True)

    # export per-second scores
    np.savez_compressed(os.path.join(args.out, "scores.npz"),
                        **{v: per_video[v] for v in per_video})
    with open(os.path.join(args.out, "video_preds.json"), "w") as f:
        json.dump({v: {"y": int(y), "score": float(s)}
                   for v, y, s in zip(vids, ys, vscore)}, f, indent=2)

    # (b) frame-level (if spans available)
    labels_map = {vid: y for vid, y in read_split(args.split)}
    flevel = None
    if os.path.exists(SPANS):
        spans = json.load(open(SPANS))
        flevel = frame_level(per_video, labels_map, spans)
        print("FRAME-LEVEL:", json.dumps(flevel), flush=True)
    else:
        print(f"[note] {SPANS} not found -> frame-level scoring skipped; "
              f"per-second scores exported to {args.out}/scores.npz", flush=True)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "split": args.split, "modalities": mods,
                   "video_level": vlevel, "frame_level": flevel}, f, indent=2)
    print(f"EVAL DONE -> {args.out}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
