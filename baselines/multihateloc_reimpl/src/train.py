#!/usr/bin/env python3
"""Train MultiHateLoc (reimplementation) on HateMM.

Paper defaults (Sec 4.1): Adam, lr 1e-4, batch 32, 100 epochs, lam_smooth 0.1,
lam_con 0.2, adaptive top-K with K=3 (top 33%). Best epoch is selected by
video-level ROC-AUC on our val split.
"""
import argparse, json, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from dataset import HateMMFeats, collate
from model import MultiHateLoc, total_loss, video_score


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def apply_modalities(batch, mods):
    """Zero out inputs of modalities not in `mods` (subset of {v,a,t})."""
    if "v" not in mods:
        batch["fv"] = torch.zeros_like(batch["fv"])
    if "a" not in mods:
        batch["fa"] = torch.zeros_like(batch["fa"])
    if "t" not in mods:
        batch["ft"] = torch.zeros_like(batch["ft"])
    return batch


@torch.no_grad()
def evaluate_video_level(model, loader, device, mods, k_div):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = apply_modalities(batch, mods)
        fv, fa, ft = batch["fv"].to(device), batch["fa"].to(device), batch["ft"].to(device)
        mask = batch["mask"].to(device)
        out = model(fv, fa, ft, mask)
        vs = video_score(out["p_fused"], mask, k_div)
        ps.extend(vs.cpu().tolist()); ys.extend(batch["y"].tolist())
    ys = np.array(ys); ps = np.array(ps)
    auc = roc_auc_score(ys, ps) if len(set(ys.tolist())) > 1 else float("nan")
    pred = (ps >= 0.5).astype(int)
    return {"auc": auc, "macro_f1": f1_score(ys, pred, average="macro"),
            "acc": accuracy_score(ys, pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--k_div", type=int, default=3)
    ap.add_argument("--max_t", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modalities", type=str, default="vat")
    ap.add_argument("--lam_smooth", type=float, default=0.1)
    ap.add_argument("--lam_con", type=float, default=0.2)
    ap.add_argument("--out", type=str, required=True, help="output dir")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} args={vars(args)}", flush=True)

    tr = HateMMFeats("train", max_t=args.max_t)
    va = HateMMFeats("val", max_t=args.max_t)
    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, collate_fn=collate, drop_last=False)
    va_loader = DataLoader(va, batch_size=args.batch, shuffle=False,
                           num_workers=args.workers, collate_fn=collate)

    model = MultiHateLoc(d_model=args.d_model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc, best_epoch = -1.0, -1
    history = []
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); agg = {}
        for batch in tr_loader:
            batch = apply_modalities(batch, args.modalities)
            fv, fa, ft = batch["fv"].to(device), batch["fa"].to(device), batch["ft"].to(device)
            mask = batch["mask"].to(device); y = batch["y"].to(device)
            out = model(fv, fa, ft, mask)
            loss, logs = total_loss(out, y, mask, k_div=args.k_div,
                                    lam_smooth=args.lam_smooth, lam_con=args.lam_con)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for kk, vv in logs.items():
                agg[kk] = agg.get(kk, 0.0) + vv
        n = max(1, len(tr_loader))
        agg = {k: v / n for k, v in agg.items()}
        vm = evaluate_video_level(model, va_loader, device, args.modalities, args.k_div)
        history.append({"epoch": ep, "train": agg, "val": vm})
        if vm["auc"] > best_auc:
            best_auc, best_epoch = vm["auc"], ep
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": ep, "val": vm}, os.path.join(args.out, "best.pt"))
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            print(f"ep{ep:03d} {time.time()-t0:.0f}s loss={agg.get('total',0):.4f} "
                  f"(mil={agg.get('mamil',0):.3f} sm={agg.get('smooth',0):.3f} "
                  f"con={agg.get('con',0):.3f}) | val auc={vm['auc']:.4f} "
                  f"f1={vm['macro_f1']:.4f} acc={vm['acc']:.4f} "
                  f"| best auc={best_auc:.4f}@{best_epoch}", flush=True)

    torch.save({"model": model.state_dict(), "args": vars(args),
                "epoch": args.epochs}, os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_auc": best_auc,
                   "history": history}, f, indent=2)
    print(f"TRAIN DONE best_val_auc={best_auc:.4f} @ epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
