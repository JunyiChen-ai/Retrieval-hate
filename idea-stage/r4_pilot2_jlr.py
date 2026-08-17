"""
Pilot R4-2 — B1 JLR, Jackknife Lower-Bound Rank Head.
Decision rules frozen in idea-stage/R4_PILOT_FREEZE_2026-08-10.md BEFORE this file existed.

Mechanism (frozen): five heads, head k trained only on items outside fold k. For each sampled
pos/neg pair, over the heads eligible for BOTH items:
    softplus( -( mean_eligible_margin - 1.0 * sd_eligible_margin ) ) + 0.1 * mean_eligible_BCE
Inference averages the five logits. Coefficient 1.0, BCE weight 0.1, 5 folds, architecture and
pair-sampling rule are FIXED; no grid. Epoch selection on validation macro-F1.
"""
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4_harness import Head, load_split, train_head, CLIP, QWEN  # noqa: E402
from r4_pilot1_mdl import macro_f1, pick_threshold  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
NFOLD = 5
NULL_REPS = 20
# Four pre-declared cells (freeze doc, R4-2 scope).
CELLS = [
    ("HateMM", "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF", "LORA"),
    ("MHC", QWEN, "QWEN"),
    ("MHC_zh", "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF", "LORA"),
    ("ImpliHateVid", CLIP, "CLIP"),
]
TIE_ORDER = ["ens_bce", "ens_pair_sd0", "single_pair", "single_bce"]
EPOCHS, LR, BS, WARMUP, NPAIR = 30, 1e-4, 64, 5, 2048


def train_multihead(tr, va, te, seed, mode, null_rng=None):
    """mode: 'jlr' (sd coeff 1.0) | 'sd0' (sd coeff 0) | 'bce' (plain BCE, no pairwise).

    null_rng: if given, permute each eligible head's item scores WITHIN HARD LABEL before
    assembling cross-head margins -- for the LCB term only (freeze doc null).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = tr["y"].shape[0]
    y = tr["y"].numpy()
    folds = np.zeros(n, dtype=int)
    for k, (_, vi) in enumerate(StratifiedKFold(NFOLD, shuffle=True,
                                                random_state=seed).split(np.zeros(n), y)):
        folds[vi] = k
    F = torch.tensor(folds, device=DEV)
    Xi, Xt, Y = tr["img"].to(DEV), tr["txt"].to(DEV), tr["y"].to(DEV)
    heads = [Head(tr["img"].shape[1], tr["txt"].shape[1]).to(DEV) for _ in range(NFOLD)]
    opt = torch.optim.AdamW([p for h in heads for p in h.parameters()], lr=LR)
    packs = {nm: (s["img"].to(DEV), s["txt"].to(DEV), s["y"].numpy()) for nm, s in
             (("val", va), ("test", te))}
    pos = torch.nonzero(Y > 0.5).squeeze(1)
    neg = torch.nonzero(Y < 0.5).squeeze(1)
    g = torch.Generator().manual_seed(seed)
    sd_coef = 1.0 if mode == "jlr" else 0.0
    best = (-1.0, None)

    for ep in range(EPOCHS):
        for h in heads:
            h.train()
        if mode == "bce":
            perm = torch.randperm(n, generator=g).to(DEV)
            for s in range(0, n, BS):
                idx = perm[s:s + BS]
                loss = 0.0
                for k, h in enumerate(heads):
                    m = F[idx] != k                      # head k excludes fold k
                    if m.sum() == 0:
                        continue
                    lo = h(Xi[idx][m], Xt[idx][m]).squeeze(1)
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(lo, Y[idx][m])
                opt.zero_grad(); loss.backward(); opt.step()
        else:
            for _ in range(max(1, n // BS)):
                pi = pos[torch.randint(len(pos), (NPAIR,), generator=g)].to(DEV)
                ni = neg[torch.randint(len(neg), (NPAIR,), generator=g)].to(DEV)
                sp = torch.stack([h(Xi[pi], Xt[pi]).squeeze(1) for h in heads])   # (K, NPAIR)
                sn = torch.stack([h(Xi[ni], Xt[ni]).squeeze(1) for h in heads])
                elig = ((F[pi].unsqueeze(0) != torch.arange(NFOLD, device=DEV).unsqueeze(1)) &
                        (F[ni].unsqueeze(0) != torch.arange(NFOLD, device=DEV).unsqueeze(1))).float()
                spn, snn = sp, sn
                if null_rng is not None:
                    # freeze-doc null: permute each head's item scores within hard label
                    pp = torch.tensor(np.stack([null_rng.permutation(NPAIR)
                                                for _ in range(NFOLD)]), device=DEV)
                    nn_ = torch.tensor(np.stack([null_rng.permutation(NPAIR)
                                                 for _ in range(NFOLD)]), device=DEV)
                    spn = torch.gather(sp, 1, pp)
                    snn = torch.gather(sn, 1, nn_)
                marg = spn - snn                                                  # (K, NPAIR)
                cnt = elig.sum(0).clamp(min=1)
                mu = (marg * elig).sum(0) / cnt
                var = ((marg - mu.unsqueeze(0)) ** 2 * elig).sum(0) / cnt.clamp(min=2)
                sd = torch.sqrt(var.clamp(min=1e-12))
                rank = nn.functional.softplus(-(mu - sd_coef * sd)).mean()
                bce_p = (nn.functional.binary_cross_entropy_with_logits(
                    sp, torch.ones_like(sp), reduction="none") * elig).sum() / elig.sum().clamp(min=1)
                bce_n = (nn.functional.binary_cross_entropy_with_logits(
                    sn, torch.zeros_like(sn), reduction="none") * elig).sum() / elig.sum().clamp(min=1)
                loss = rank + 0.1 * (bce_p + bce_n) / 2
                opt.zero_grad(); loss.backward(); opt.step()

        for h in heads:
            h.eval()
        with torch.no_grad():
            pr = {nm: torch.sigmoid(torch.stack([h(a, b).squeeze(1) for h in heads]).mean(0)
                                    ).cpu().numpy() for nm, (a, b, _) in packs.items()}
        if ep >= WARMUP:
            f = macro_f1(packs["val"][2], pr["val"], pick_threshold(packs["val"][2], pr["val"]))
            if f > best[0]:
                best = (f, {"val": pr["val"], "test": pr["test"], "ep": ep})
    return best[1]


def single_pairwise(tr, va, te, seed):
    """Comparator (d): single head, ordinary pairwise-AUC loss + 0.1 BCE."""
    torch.manual_seed(seed); np.random.seed(seed)
    n = tr["y"].shape[0]
    Xi, Xt, Y = tr["img"].to(DEV), tr["txt"].to(DEV), tr["y"].to(DEV)
    h = Head(tr["img"].shape[1], tr["txt"].shape[1]).to(DEV)
    opt = torch.optim.AdamW(h.parameters(), lr=LR)
    pos = torch.nonzero(Y > 0.5).squeeze(1); neg = torch.nonzero(Y < 0.5).squeeze(1)
    g = torch.Generator().manual_seed(seed)
    packs = {nm: (s["img"].to(DEV), s["txt"].to(DEV), s["y"].numpy()) for nm, s in
             (("val", va), ("test", te))}
    best = (-1.0, None)
    for ep in range(EPOCHS):
        h.train()
        for _ in range(max(1, n // BS)):
            pi = pos[torch.randint(len(pos), (NPAIR,), generator=g)].to(DEV)
            ni = neg[torch.randint(len(neg), (NPAIR,), generator=g)].to(DEV)
            sp = h(Xi[pi], Xt[pi]).squeeze(1); sn = h(Xi[ni], Xt[ni]).squeeze(1)
            loss = nn.functional.softplus(-(sp - sn)).mean() + 0.1 * 0.5 * (
                nn.functional.binary_cross_entropy_with_logits(sp, torch.ones_like(sp)) +
                nn.functional.binary_cross_entropy_with_logits(sn, torch.zeros_like(sn)))
            opt.zero_grad(); loss.backward(); opt.step()
        h.eval()
        with torch.no_grad():
            pr = {nm: torch.sigmoid(h(a, b).squeeze(1)).cpu().numpy()
                  for nm, (a, b, _) in packs.items()}
        if ep >= WARMUP:
            f = macro_f1(packs["val"][2], pr["val"], pick_threshold(packs["val"][2], pr["val"]))
            if f > best[0]:
                best = (f, {"val": pr["val"], "test": pr["test"], "ep": ep})
    return best[1]


if __name__ == "__main__":
    def log(s):
        print(s, flush=True)
    log(f"R4-2 JLR start {time.strftime('%Y-%m-%dT%H:%M:%S')} device={DEV}")
    log("Rules: idea-stage/R4_PILOT_FREEZE_2026-08-10.md (frozen before this file existed)")
    out = {"pilot": "R4-2_JLR", "freeze": "idea-stage/R4_PILOT_FREEZE_2026-08-10.md",
           "cells": {}, "null": {}}

    for ds, mt, tag in CELLS:
        tr, va, te = (load_split(ds, mt, s) for s in ("train", "val", "test"))
        yv, yt = va["y"].numpy(), te["y"].numpy()
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            r = {}
            r["JLR"] = train_multihead(tr, va, te, seed, "jlr")
            r["ens_pair_sd0"] = train_multihead(tr, va, te, seed, "sd0")
            r["ens_bce"] = train_multihead(tr, va, te, seed, "bce")
            r["single_pair"] = single_pairwise(tr, va, te, seed)
            b = train_head(tr, va, te, seed, device=DEV)
            r["single_bce"] = {"val": b["val_prob"], "test": b["test_prob"]}
            row = {"seed": seed, "methods": {}, "scores": {"y": yt.tolist()}}
            for nm, p in r.items():
                th = pick_threshold(yv, p["val"])
                row["methods"][nm] = {
                    "val_roc": float(roc_auc_score(yv, p["val"])),
                    "val_macro_f1": float(macro_f1(yv, p["val"], th)),
                    "test_roc": float(roc_auc_score(yt, p["test"])),
                    "test_macro_f1": float(macro_f1(yt, p["test"], th))}
                row["scores"][nm] = np.asarray(p["test"]).tolist()
            per_seed.append(row)
            log(f"  [{ds}/{tag}] seed {seed} val_roc="
                + " ".join(f"{k}:{v['val_roc']:.4f}" for k, v in row["methods"].items())
                + f"  ({time.time()-t0:.0f}s)  [test withheld]")
        names = [n for n in per_seed[0]["methods"] if n != "JLR"]
        mv = {n: float(np.mean([r["methods"][n]["val_roc"] for r in per_seed])) for n in names}
        top = max(mv.values())
        frozen = sorted([n for n in names if abs(mv[n] - top) < 1e-12],
                        key=lambda n: TIE_ORDER.index(n))[0]
        d_roc = float(np.mean([r["methods"]["JLR"]["test_roc"] - r["methods"][frozen]["test_roc"]
                               for r in per_seed]))
        d_f1 = float(np.mean([r["methods"]["JLR"]["test_macro_f1"]
                              - r["methods"][frozen]["test_macro_f1"] for r in per_seed]))
        out["cells"][ds] = {"encoder": tag, "per_seed": per_seed, "mean_val_roc": mv,
                            "frozen_comparator": frozen, "DeltaROC": d_roc, "DeltaF1": d_f1}
        log(f"  [{ds}/{tag}] FROZEN COMPARATOR = {frozen}")
        json.dump(out, open("idea-stage/r4_pilot2.json", "w"), indent=2)

    # ---- null: MHC-EN, 20 pre-seeded null trainings (freeze doc, unchanged -- D1 touched R4-1 only)
    log("PROGRESS phase=null dataset=MHC")
    ds, mt = "MHC", QWEN
    tr, va, te = (load_split(ds, mt, s) for s in ("train", "val", "test"))
    yv, yt = va["y"].numpy(), te["y"].numpy()
    comp = out["cells"]["MHC"]["frozen_comparator"]
    base = [r["methods"][comp]["test_roc"] for r in out["cells"]["MHC"]["per_seed"]]
    nd = []
    for rep in range(NULL_REPS):
        seed = SEEDS[rep % len(SEEDS)]
        p = train_multihead(tr, va, te, seed, "jlr",
                            null_rng=np.random.default_rng(30_000 + rep))
        nd.append(float(roc_auc_score(yt, p["test"]) - base[SEEDS.index(seed)]))
        if (rep + 1) % 5 == 0:
            log(f"  null {rep+1}/{NULL_REPS} running mean {np.mean(nd):+.4f}")
    null95 = float(np.percentile(np.maximum(0, nd), 95))
    out["null"] = {"deltas": nd, "Null95": null95}

    C = out["cells"]
    mean_roc = float(np.mean([C[d]["DeltaROC"] for d in C]))
    mean_f1 = float(np.mean([C[d]["DeltaF1"] for d in C]))
    c1 = (C["MHC"]["DeltaROC"] >= 0.010) and (C["MHC"]["DeltaROC"] >= 3 * null95)
    c2 = mean_roc >= 0.010
    c3 = (sum(C[d]["DeltaROC"] > 0 for d in C) >= 3) and all(C[d]["DeltaROC"] >= -0.005 for d in C)
    c4 = (mean_f1 >= 0.005) and all(C[d]["DeltaF1"] >= -0.005 for d in C)
    out["verdict"] = {"MeanDeltaROC": mean_roc, "MeanDeltaF1": mean_f1, "Null95": null95,
                      "c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "c4": bool(c4),
                      "GO": bool(c1 and c2 and c3 and c4)}
    json.dump(out, open("idea-stage/r4_pilot2.json", "w"), indent=2)
    log("=" * 78)
    log("FULL TEST TABLE (withheld until now)")
    for d in C:
        R = C[d]
        log(f"-- {d}/{R['encoder']}, frozen comparator = {R['frozen_comparator']}")
        for nm in list(R["per_seed"][0]["methods"]):
            rr = [s["methods"][nm]["test_roc"] for s in R["per_seed"]]
            ff = [s["methods"][nm]["test_macro_f1"] for s in R["per_seed"]]
            log(f"     {nm:<14} test ROC {np.mean(rr):.4f}+/-{np.std(rr):.4f}   "
                f"macroF1 {np.mean(ff):.4f}+/-{np.std(ff):.4f}")
        log(f"     => DeltaROC={R['DeltaROC']:+.4f}  DeltaF1={R['DeltaF1']:+.4f}")
    log(f"MeanDeltaROC={mean_roc:+.4f} MeanDeltaF1={mean_f1:+.4f} Null95={null95:.4f}")
    log(f"c1={c1} c2={c2} c3={c3} c4={c4}")
    log(f"VERDICT: {'GO' if out['verdict']['GO'] else 'KILL'}")
