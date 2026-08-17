"""
Pilot R4-1 — F1 MDL, Monotone Disagreement Lattice.
Decision rules frozen in idea-stage/R4_PILOT_FREEZE_2026-08-10.md BEFORE this file existed.
Nothing here may deviate from that document.

Protocol: train on train, select epoch on val, REPORT TEST. Seeds 0/1/2. Single submission.
"""
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4_harness import load_split, train_head, CLIP, QWEN  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
NFOLD = 5
NKNOT = 4
NULL_REPS = 200
LORA = {"HateMM": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
        "MHC": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
        "MHC_zh": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"}
CELLS = {
    "HateMM": [("CLIP", CLIP), ("QWEN", QWEN), ("LORA", LORA["HateMM"])],
    "MHC": [("CLIP", CLIP), ("QWEN", QWEN), ("LORA", LORA["MHC"])],
    "MHC_zh": [("CLIP", CLIP), ("QWEN", QWEN), ("LORA", LORA["MHC_zh"])],
    "ImpliHateVid": [("CLIP", CLIP), ("QWEN", QWEN)],   # no LoRA cache -- frozen in the freeze doc
}
# Comparator tie-break order, most conservative first (freeze doc).
TIE_ORDER = ["mlp", "logistic", "weighted", "mean_logit", "mean_prob", "single"]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def macro_f1(y, s, t):
    return f1_score(y, (s >= t).astype(int), average="macro")


def pick_threshold(yv, sv):
    """Max validation macro-F1; ties -> threshold closest to 0.5 (freeze doc)."""
    cands = np.unique(np.concatenate([sv, [np.median(sv)]]))
    if len(cands) > 400:
        cands = np.quantile(sv, np.linspace(0.01, 0.99, 400))
    best_f1, best_t = -1.0, float(np.median(sv))
    for t in cands:
        f = macro_f1(yv, sv, t)
        if f > best_f1 + 1e-12 or (abs(f - best_f1) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_f1, best_t = f, float(t)
    return best_t


# --------------------------------------------------------------------------- lattice
class MonotoneLattice(nn.Module):
    """Multilinear interpolation over a D-dim grid with NKNOT knots per axis.

    Monotone non-decreasing along every axis by construction: the vertex table is the
    cumulative sum of non-negative increments along each axis.
    """

    def __init__(self, knots):
        super().__init__()
        self.D = len(knots)
        self.register_buffer("knots", torch.tensor(np.array(knots), dtype=torch.float32))
        shape = [NKNOT] * self.D
        self.base = nn.Parameter(torch.zeros(1))
        self.raw = nn.Parameter(torch.full(shape, -3.0))  # softplus(-3) ~ 0.049

    def table(self):
        t = nn.functional.softplus(self.raw)
        for d in range(self.D):
            t = torch.cumsum(t, dim=d)
        return t + self.base

    def forward(self, x):
        """x: (N, D) raw logits. Returns (N,) lattice output."""
        n = x.shape[0]
        idx, frac = [], []
        for d in range(self.D):
            k = self.knots[d]
            xi = torch.clamp(x[:, d], float(k[0]), float(k[-1]))
            j = torch.clamp(torch.searchsorted(k.contiguous(), xi.contiguous(), right=True) - 1,
                            0, NKNOT - 2)
            lo, hi = k[j], k[j + 1]
            frac.append(((xi - lo) / (hi - lo + 1e-9)).clamp(0, 1))
            idx.append(j)
        tab = self.table()
        out = torch.zeros(n, device=x.device)
        for corner in range(2 ** self.D):
            w = torch.ones(n, device=x.device)
            flat = torch.zeros(n, dtype=torch.long, device=x.device)
            stride = 1
            for d in reversed(range(self.D)):
                bit = (corner >> d) & 1
                w = w * (frac[d] if bit else (1 - frac[d]))
                flat = flat + (idx[d] + bit) * stride
                stride *= NKNOT
            out = out + w * tab.reshape(-1)[flat]
        return out


def fit_lattice(tr_x, tr_y, va_x, va_y, ref_tr, ref_va, seed, epochs=300):
    """Pairwise logistic rank loss + BCE calibration + concordant-region identity penalty (w=1.0).

    Knots: train-OOF empirical 0, 1/3, 2/3, 1 quantiles per axis (freeze doc).
    Epoch selection on validation macro-F1.
    """
    torch.manual_seed(seed)
    D = tr_x.shape[1]
    knots = [np.quantile(tr_x[:, d], [0.0, 1 / 3, 2 / 3, 1.0]) for d in range(D)]
    knots = [np.maximum.accumulate(k + np.arange(NKNOT) * 1e-6) for k in knots]
    m = MonotoneLattice(knots).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)

    X = torch.tensor(tr_x, dtype=torch.float32, device=DEV)
    Y = torch.tensor(tr_y, dtype=torch.float32, device=DEV)
    R = torch.tensor(ref_tr, dtype=torch.float32, device=DEV)
    Xv = torch.tensor(va_x, dtype=torch.float32, device=DEV)
    # concordant region = every encoder predicts the same class at its own zero-logit
    conc = torch.tensor((np.sign(tr_x) == np.sign(tr_x[:, :1])).all(axis=1),
                        dtype=torch.float32, device=DEV)
    pos = torch.nonzero(Y > 0.5).squeeze(1)
    neg = torch.nonzero(Y < 0.5).squeeze(1)
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("degenerate labels")
    g = torch.Generator(device="cpu").manual_seed(seed)

    best = (-1.0, None)
    for ep in range(epochs):
        m.train()
        o = m(X)
        npair = min(4096, len(pos) * len(neg))
        pi = pos[torch.randint(len(pos), (npair,), generator=g)]
        ni = neg[torch.randint(len(neg), (npair,), generator=g)]
        rank = nn.functional.softplus(-(o[pi] - o[ni])).mean()
        bce = nn.functional.binary_cross_entropy_with_logits(o, Y)
        ident = (conc * (o - R) ** 2).sum() / (conc.sum() + 1e-9)
        loss = rank + bce + 1.0 * ident
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep >= 20 and ep % 5 == 0:
            m.eval()
            with torch.no_grad():
                sv = m(Xv).cpu().numpy()
            f = macro_f1(va_y, sv, pick_threshold(va_y, sv))
            if f > best[0]:
                best = (f, {k: v.detach().clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1])
    m.eval()
    return m


# --------------------------------------------------------------------------- base logits
def base_logits(dataset, enc_list, seed, cache):
    """OOF train logits + full-train val/test logits, per encoder."""
    out = {}
    for tag, mt in enc_list:
        key = (dataset, tag, seed)
        if key in cache:
            out[tag] = cache[key]
            continue
        tr = load_split(dataset, mt, "train")
        va = load_split(dataset, mt, "val")
        te = load_split(dataset, mt, "test")
        y = tr["y"].numpy()
        oof = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=seed)
        for f, (ti, vi) in enumerate(skf.split(np.zeros(len(y)), y)):
            sub = {k: (tr[k][ti] if k != "ids" else [tr["ids"][j] for j in ti])
                   for k in tr}
            hold = {k: (tr[k][vi] if k != "ids" else [tr["ids"][j] for j in vi])
                    for k in tr}
            r = train_head(sub, va, hold, seed * 100 + f, device=DEV)
            oof[vi] = logit(r["test_prob"])
        full = train_head(tr, va, te, seed, device=DEV)
        val_probs = full["val_prob"]
        # test logits from the same val-selected checkpoint
        out[tag] = {"oof": oof, "val": logit(val_probs), "test": logit(full["test_prob"]),
                    "ytr": y, "yva": va["y"].numpy(), "yte": te["y"].numpy()}
        cache[key] = out[tag]
    return out


# --------------------------------------------------------------------------- comparators
def comparators(L, tags, seed):
    """Every frozen comparator. Returns name -> (val_score, test_score)."""
    ytr, yva = L[tags[0]]["ytr"], L[tags[0]]["yva"]
    Otr = np.stack([L[t]["oof"] for t in tags], 1)
    Ova = np.stack([L[t]["val"] for t in tags], 1)
    Ote = np.stack([L[t]["test"] for t in tags], 1)
    C = {}
    # single: validation-best encoder
    best = max(tags, key=lambda t: roc_auc_score(yva, L[t]["val"]))
    C["single"] = (L[best]["val"], L[best]["test"], best)
    C["mean_logit"] = (Ova.mean(1), Ote.mean(1), None)
    sig = lambda z: 1 / (1 + np.exp(-z))
    C["mean_prob"] = (sig(Ova).mean(1), sig(Ote).mean(1), None)
    w = np.array([max(0.0, roc_auc_score(yva, L[t]["val"]) - 0.5) for t in tags])
    w = w / (w.sum() + 1e-9)
    C["weighted"] = (Ova @ w, Ote @ w, None)
    lr = LogisticRegression(max_iter=2000).fit(Otr, ytr)
    C["logistic"] = (lr.decision_function(Ova), lr.decision_function(Ote), None)
    # MLP stacker, params >= lattice (NKNOT**D + 1)
    torch.manual_seed(seed)
    hid = max(32, (NKNOT ** len(tags) + 1) // (len(tags) + 2) + 4)
    mlp = nn.Sequential(nn.Linear(len(tags), hid), nn.ReLU(), nn.Linear(hid, 1)).to(DEV)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    Xt = torch.tensor(Otr, dtype=torch.float32, device=DEV)
    Yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xv = torch.tensor(Ova, dtype=torch.float32, device=DEV)
    bestm = (-1.0, None)
    for ep in range(300):
        mlp.train()
        o = mlp(Xt).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(o, Yt)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep >= 20 and ep % 5 == 0:
            mlp.eval()
            with torch.no_grad():
                sv = mlp(Xv).squeeze(1).cpu().numpy()
            f = macro_f1(yva, sv, pick_threshold(yva, sv))
            if f > bestm[0]:
                bestm = (f, {k: v.detach().clone() for k, v in mlp.state_dict().items()})
    mlp.load_state_dict(bestm[1]); mlp.eval()
    with torch.no_grad():
        C["mlp"] = (mlp(Xv).squeeze(1).cpu().numpy(),
                    mlp(torch.tensor(Ote, dtype=torch.float32, device=DEV)).squeeze(1).cpu().numpy(),
                    None)
    return C


def run_dataset(ds, log):
    tags = [t for t, _ in CELLS[ds]]
    cache = {}
    per_seed = []
    for seed in SEEDS:
        t0 = time.time()
        L = base_logits(ds, CELLS[ds], seed, cache)
        yva, yte = L[tags[0]]["yva"], L[tags[0]]["yte"]
        ytr = L[tags[0]]["ytr"]
        C = comparators(L, tags, seed)
        ref = max(tags, key=lambda t: roc_auc_score(yva, L[t]["val"]))
        Otr = np.stack([L[t]["oof"] for t in tags], 1)
        Ova = np.stack([L[t]["val"] for t in tags], 1)
        Ote = np.stack([L[t]["test"] for t in tags], 1)
        m = fit_lattice(Otr, ytr, Ova, yva, L[ref]["oof"], L[ref]["val"], seed)
        with torch.no_grad():
            sv = m(torch.tensor(Ova, dtype=torch.float32, device=DEV)).cpu().numpy()
            st = m(torch.tensor(Ote, dtype=torch.float32, device=DEV)).cpu().numpy()
        row = {"seed": seed, "ref": ref, "methods": {}}
        for name, (a, b, _) in list(C.items()) + [("MDL", (sv, st, None))]:
            th = pick_threshold(yva, a)
            row["methods"][name] = {
                "val_roc": float(roc_auc_score(yva, a)),
                "val_macro_f1": float(macro_f1(yva, a, th)),
                "test_roc": float(roc_auc_score(yte, b)),
                "test_macro_f1": float(macro_f1(yte, b, th)),
            }
        row["scores"] = {"y": yte.tolist(), "MDL": st.tolist(),
                         **{n: np.asarray(C[n][1]).tolist() for n in C}}
        per_seed.append(row)
        # D1 ruling item 4: suppress per-cell TEST output until all predictions and comparator
        # choices are saved. Only VAL progress is logged live.
        log(f"  [{ds}] seed {seed} ref={ref} val_roc="
            + " ".join(f"{k}:{v['val_roc']:.4f}" for k, v in row["methods"].items())
            + f"  ({time.time()-t0:.0f}s)  [test metrics withheld until final table]")
    # frozen comparator: highest mean VAL roc, tie-break TIE_ORDER
    names = [n for n in per_seed[0]["methods"] if n != "MDL"]
    mv = {n: np.mean([r["methods"][n]["val_roc"] for r in per_seed]) for n in names}
    top = max(mv.values())
    cands = [n for n in names if abs(mv[n] - top) < 1e-12]
    frozen = sorted(cands, key=lambda n: TIE_ORDER.index(n))[0]
    log(f"  [{ds}] FROZEN COMPARATOR = {frozen}  (mean val ROC {mv[frozen]:.4f}; "
        + ", ".join(f"{n} {mv[n]:.4f}" for n in names) + ")")
    d_roc = np.mean([r["methods"]["MDL"]["test_roc"] - r["methods"][frozen]["test_roc"]
                     for r in per_seed])
    d_f1 = np.mean([r["methods"]["MDL"]["test_macro_f1"] - r["methods"][frozen]["test_macro_f1"]
                    for r in per_seed])
    return {"dataset": ds, "tags": tags, "per_seed": per_seed, "mean_val_roc": mv,
            "frozen_comparator": frozen, "DeltaROC": float(d_roc), "DeltaF1": float(d_f1),
            "cache_keys": [f"{k[0]}|{k[1]}|{k[2]}" for k in cache]}, cache, per_seed


def paired_bootstrap(out, log):
    """Deviation-D1 replacement for clause 2, per idea-stage/R4_DEVIATION_D1_RULING.md.

    Paired stratified joint-row bootstrap. 10,000 reps, rng(20260810). Dataset order
    HateMM, MHC-EN, MHC-ZH, ImpliHateVid; positives drawn before negatives. The SAME sampled
    joint rows are applied to MDL, the frozen comparator and every seed -- methods and encoders
    are never resampled independently. LCB95 = quantile(MeanDeltaROC_boot, 0.05, linear);
    no truncation at zero, no x3.
    """
    rng = np.random.default_rng(20260810)
    order = ["HateMM", "MHC", "MHC_zh", "ImpliHateVid"]
    pre = {}
    for ds in order:
        R = out["datasets"][ds]
        comp = R["frozen_comparator"]
        y = np.array(R["per_seed"][0]["scores"]["y"])
        pre[ds] = {
            "pos": np.where(y == 1)[0], "neg": np.where(y == 0)[0], "y": y,
            "mdl": [np.array(r["scores"]["MDL"]) for r in R["per_seed"]],
            "cmp": [np.array(r["scores"][comp]) for r in R["per_seed"]],
        }
    boot = np.empty(10_000)
    for b in range(10_000):
        per_ds = []
        for ds in order:
            P = pre[ds]
            ip = rng.choice(P["pos"], size=len(P["pos"]), replace=True)   # positives first
            ineg = rng.choice(P["neg"], size=len(P["neg"]), replace=True)
            idx = np.concatenate([ip, ineg])
            yb = P["y"][idx]
            if yb.min() == yb.max():
                per_ds.append(0.0)
                continue
            per_ds.append(float(np.mean([
                roc_auc_score(yb, P["mdl"][s][idx]) - roc_auc_score(yb, P["cmp"][s][idx])
                for s in range(len(SEEDS))])))
        boot[b] = float(np.mean(per_ds))
        if (b + 1) % 2000 == 0:
            log(f"  bootstrap {b+1}/10000 running mean {boot[:b+1].mean():+.5f}")
    return boot


if __name__ == "__main__":
    os.makedirs("logging/runs/r4_pilot1_mdl", exist_ok=True)
    def log(s):
        print(s, flush=True)

    log(f"R4-1 MDL start {time.strftime('%Y-%m-%dT%H:%M:%S')} device={DEV}")
    log(f"Rules: idea-stage/R4_PILOT_FREEZE_2026-08-10.md (frozen before this file existed)")
    out = {"pilot": "R4-1_MDL", "freeze": "idea-stage/R4_PILOT_FREEZE_2026-08-10.md",
           "datasets": {}, "null": {}}
    caches = {}
    for ds in CELLS:
        log(f"PROGRESS dataset={ds} phase=primary")
        res, cache, _ = run_dataset(ds, log)
        out["datasets"][ds] = res
        caches[ds] = cache
        json.dump(out, open("idea-stage/r4_pilot1.json", "w"), indent=2)
    log("PROGRESS phase=paired_bootstrap (deviation D1 replacement for clause 2)")
    boot = paired_bootstrap(out, log)
    lcb95 = float(np.quantile(boot, 0.05, method="linear"))
    out["bootstrap"] = {"n": 10000, "rng": 20260810, "mean": float(boot.mean()),
                        "LCB95": lcb95,
                        "pct": {p: float(np.quantile(boot, p / 100, method="linear"))
                                for p in (1, 5, 25, 50, 75, 95, 99)}}

    # ---- frozen decision rule evaluation (clause 2 per deviation D1 ruling)
    D = out["datasets"]
    mean_roc = float(np.mean([D[d]["DeltaROC"] for d in D]))
    mean_f1 = float(np.mean([D[d]["DeltaF1"] for d in D]))
    c1 = D["MHC"]["DeltaROC"] >= 0.010
    c2 = (mean_roc >= 0.010) and (lcb95 > 0.0)
    c3 = (sum(D[d]["DeltaROC"] > 0 for d in D) >= 3) and all(D[d]["DeltaROC"] >= -0.005 for d in D)
    c4 = (mean_f1 >= 0.010) and all(D[d]["DeltaF1"] >= -0.005 for d in D)
    out["verdict"] = {
        "MeanDeltaROC": mean_roc, "MeanDeltaF1": mean_f1, "LCB95": lcb95,
        "c1_MHC_EN_DeltaROC_ge_0.010": bool(c1),
        "c2_MeanDeltaROC_ge_0.010_and_LCB95_gt_0": bool(c2),
        "c3_three_of_four_positive_none_below_-0.005": bool(c3),
        "c4_MeanDeltaF1_ge_0.010_none_below_-0.005": bool(c4),
        "GO": bool(c1 and c2 and c3 and c4),
    }
    json.dump(out, open("idea-stage/r4_pilot1.json", "w"), indent=2)
    log("=" * 78)
    log("FULL TEST TABLE (withheld until now per deviation-D1 ruling item 4)")
    for d in D:
        R = D[d]
        log(f"-- {d} (encoders {'+'.join(R['tags'])}), frozen comparator = {R['frozen_comparator']}")
        for name in list(R["per_seed"][0]["methods"]):
            rr = [s["methods"][name]["test_roc"] for s in R["per_seed"]]
            ff = [s["methods"][name]["test_macro_f1"] for s in R["per_seed"]]
            log(f"     {name:<11} test ROC {np.mean(rr):.4f}+/-{np.std(rr):.4f}   "
                f"test macroF1 {np.mean(ff):.4f}+/-{np.std(ff):.4f}")
        log(f"     => DeltaROC={R['DeltaROC']:+.4f}  DeltaF1={R['DeltaF1']:+.4f}")
    log("=" * 78)
    log(f"MeanDeltaROC={mean_roc:+.4f}  MeanDeltaF1={mean_f1:+.4f}  "
        f"bootstrap LCB95={lcb95:+.5f} (mean {boot.mean():+.5f})")
    log(f"c1(MHC-EN dROC>=.010)={c1}  c2(mean>=.010 & LCB95>0)={c2}  "
        f"c3(3/4 positive)={c3}  c4(meandF1>=.010)={c4}")
    log(f"VERDICT: {'GO' if out['verdict']['GO'] else 'KILL'}")
