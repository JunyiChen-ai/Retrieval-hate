#!/usr/bin/env python
"""R11-SEG v2 runner. Executes exactly idea-stage/R11_SEG_PILOT_FREEZE_V2.md. Single submission.

Arms: A1/A2/A4 (carried, per-seed scores saved), B2 DENSE (claim a),
C0/C1/C1a/C1b/C2/C3 coverage-budget decoding (claim b), E1/E2 objective control.
Writes idea-stage/r11_seg/out/results_v2.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_pilot as R  # same directory

ROOT = Path("/home/jehc223/Retrieval-hate")
OUT = ROOT / "idea-stage/r11_seg/out"
K = R.K
SEEDS = R.SEEDS
DEV = R.DEV


# ------------------------------------------------------------------ B2 DENSE
class Dense(nn.Module):
    """Minimal MS-TCT / PAT-shaped dense action detector, causal.

    Causal dilated conv backbone + a multi-scale temporal branch (stride 1/2/4,
    upsampled back causally) + per-window multi-hot sigmoid over the 5 offensive
    categories. No background class, no top-k, no intra-video negatives.
    """

    def __init__(self, d_in, c=64, n_cls=5):
        super().__init__()
        self.proj = R.Proj(d_in)
        self.inc = nn.Conv1d(256, c, 1)
        self.blocks = nn.ModuleList([R.DilatedResidual(c, 2 ** l) for l in range(4)])
        self.scales = nn.ModuleList([nn.Conv1d(c, c, 3) for _ in range(3)])  # stride 1,2,4 via pooling
        self.fuse = nn.Conv1d(3 * c, c, 1)
        self.head = nn.Conv1d(c, n_cls, 1)

    def forward(self, x):
        h = self.inc(self.proj(x).transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        feats = []
        for s, conv in enumerate(self.scales):
            r = 2 ** s
            # causal multi-scale pooling: position t sees only frames [t-r+1, t]
            z = h if r == 1 else F.avg_pool1d(F.pad(h, (r - 1, 0)), r, stride=1)
            z = conv(F.pad(z, (2, 0)))                      # causal k=3
            feats.append(F.relu(z))
        h = F.relu(self.fuse(torch.cat(feats, dim=1)))
        return self.head(h).transpose(1, 2)                 # [B,T,5] logits


def train_dense(X, y_multi, tr, va, te, seed, y_ts, wot):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Dense(X.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=R.LR, weight_decay=R.WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, R.EPOCHS)
    Xt = torch.from_numpy(X).to(DEV)
    Yt = torch.from_numpy(y_multi.astype(np.float32)).to(DEV)
    g = np.random.default_rng(seed)
    best = (-1.0, None)
    for ep in range(R.EPOCHS):
        m.train()
        perm = g.permutation(tr)
        for s in range(0, len(perm), R.BATCH):
            b = perm[s : s + R.BATCH]
            loss = F.binary_cross_entropy_with_logits(m(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pv = 1.0 - torch.prod(1.0 - torch.sigmoid(m(Xt[va])), dim=-1)
        pv = pv.cpu().numpy()
        c = R.ts_counts(y_ts, wot, (pv >= 0.5).astype(int), va).sum(0)
        sc = R.macro_f1_acc(c)[0]
        if sc > best[0]:
            with torch.no_grad():
                pt = (1.0 - torch.prod(1.0 - torch.sigmoid(m(Xt[te])), dim=-1)).cpu().numpy()
            best = (sc, pt)
    return best


# ------------------------------------------------------------------ E2 intra-video negatives
class CTCNProj(R.CTCN):
    def forward(self, x):
        self._emb = self.proj(x)
        h = self._emb.transpose(1, 2)
        outs, cur = [], h
        for st in self.stages:
            o = st(cur); outs.append(o.transpose(1, 2)); cur = F.softmax(o, dim=1)
        self._all = outs
        return outs[-1]


def train_obj(X, y, tr, va, te, seed, y_ts, wot, intra_w):
    """E1 (intra_w=0) / E2 (intra_w>0): UniVTG-style score-derived intra-video negatives."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = CTCNProj(X.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=R.LR, weight_decay=R.WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, R.EPOCHS)
    Xt = torch.from_numpy(X).to(DEV); Yt = torch.from_numpy(y).to(DEV)
    g = np.random.default_rng(seed); best = (-1.0, None)
    nsel = max(1, int(round(0.2 * K)))
    for ep in range(R.EPOCHS):
        m.train(); perm = g.permutation(tr)
        for s in range(0, len(perm), R.BATCH):
            b = perm[s : s + R.BATCH]
            out = m(Xt[b])
            loss = 0.0
            for o in m._all:
                loss = loss + F.cross_entropy(o.reshape(-1, 2), Yt[b].reshape(-1))
                lp = F.log_softmax(o, dim=-1)
                dl = torch.clamp((lp[:, 1:] - lp[:, :-1]) ** 2, max=R.TAU ** 2)
                loss = loss + R.LAMBDA_SMOOTH * dl.mean()
            if intra_w > 0:
                sc = (out[..., 1] - out[..., 0]).detach()          # relative within-video score
                hi = torch.topk(sc, nsel, dim=1).indices
                lo = torch.topk(-sc, nsel, dim=1).indices
                e = F.normalize(m._emb, dim=-1)
                pos = torch.gather(e, 1, hi[..., None].expand(-1, -1, e.shape[-1]))
                neg = torch.gather(e, 1, lo[..., None].expand(-1, -1, e.shape[-1]))
                anch = F.normalize(pos.mean(1), dim=-1)
                lp_ = torch.einsum("bd,bkd->bk", anch, pos) / 0.07
                ln_ = torch.einsum("bd,bkd->bk", anch, neg) / 0.07
                loss = loss + intra_w * (-(lp_.logsumexp(1)
                                           - torch.cat([lp_, ln_], 1).logsumexp(1))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pv = F.softmax(m(Xt[va]), dim=-1)[..., 1].cpu().numpy()
        c = R.ts_counts(y_ts, wot, (pv >= 0.5).astype(int), va).sum(0)
        s_ = R.macro_f1_acc(c)[0]
        if s_ > best[0]:
            with torch.no_grad():
                pt = F.softmax(m(Xt[te]), dim=-1)[..., 1].cpu().numpy()
            best = (s_, pt)
    return best


# ------------------------------------------------------------------ decoders
def fit_transition(y, tr):
    T = np.ones((2, 2)) * 1.0
    for i in tr:
        for k in range(1, K):
            T[y[i, k - 1], y[i, k]] += 1
    return T / T.sum(1, keepdims=True)


def forward_filter(p, T, prior=0.5):
    """Causal HMM forward recursion; returns filtered P(y_k=1 | p_0..p_k)."""
    out = np.zeros_like(p)
    a = np.array([1 - prior, prior])
    for k in range(len(p)):
        if k > 0:
            a = a @ T
        e = np.array([1 - p[k], p[k]])
        a = a * e
        a = a / max(a.sum(), 1e-12)
        out[k] = a[1]
    return out


def decode(p, mode, budget=None, T=None, dur_min=0):
    """p: [K] scores for one video. Returns binary [K]."""
    if mode == "unconstrained":
        return (p >= 0.5).astype(int)
    if mode == "budget":          # causal quantile budget
        out = np.zeros(K, dtype=int)
        for k in range(K):
            c = float(np.clip(budget[k], 0.0, 1.0))
            thr = np.quantile(p[: k + 1], 1.0 - c) if c > 0 else 1.1
            out[k] = int(p[k] >= thr)
        return out
    if mode == "trans":
        return (forward_filter(p, T) >= 0.5).astype(int)
    if mode == "covbud":          # both, causal
        f = forward_filter(p, T)
        out = np.zeros(K, dtype=int)
        for k in range(K):
            c = float(np.clip(budget[k], 0.0, 1.0))
            thr = np.quantile(f[: k + 1], 1.0 - c) if c > 0 else 1.1
            out[k] = int(f[k] >= thr)
        return out
    if mode == "viterbi":         # OFFLINE
        c = float(np.clip(budget, 0.0, 1.0))
        lp = np.log(np.clip(np.stack([1 - p, p], 1), 1e-9, 1))
        bias = np.log(max(c, 1e-3) / max(1 - c, 1e-3))
        lp[:, 1] += bias
        lT = np.log(np.clip(T, 1e-9, 1))
        dp = np.zeros((K, 2)); bp = np.zeros((K, 2), dtype=int)
        dp[0] = lp[0]
        for k in range(1, K):
            for j in range(2):
                cand = dp[k - 1] + lT[:, j]
                bp[k, j] = int(np.argmax(cand)); dp[k, j] = cand[bp[k, j]] + lp[k, j]
        path = np.zeros(K, dtype=int); path[-1] = int(np.argmax(dp[-1]))
        for k in range(K - 1, 0, -1):
            path[k - 1] = bp[k, path[k]]
        if dur_min > 1:           # min-duration constraint on emitted runs
            k = 0
            while k < K:
                j = k
                while j + 1 < K and path[j + 1] == path[k]:
                    j += 1
                if (j - k + 1) < dur_min and 0 < k:
                    path[k : j + 1] = path[k - 1]
                k = j + 1
        return path
    raise ValueError(mode)


def main():
    R.causality_guard()
    D = R.load_all()
    # y_multi is in the frozen grid artifact but not returned by load_all(); read it directly.
    D["y_multi"] = np.load(OUT / "grid_labels.npz", allow_pickle=True)["y_multi"]
    tr, va, te = D["tr"], D["va"], D["te"]
    y_ts, wot, y_win = D["y_ts"], D["win_of_ts"], D["y_win"]
    res, meta = {}, {}
    X_ALL = R.zscore(D["ALL"], tr)
    X_V = R.zscore(D["chans"]["V"], tr)

    # causality assertion for the new modules
    x = torch.randn(4, K, 64, device=DEV); x2 = x.clone(); x2[:, 20:] += 100.0
    for nm, mod in [("B2_DENSE", Dense(64)), ("E_CTCNProj", CTCNProj(64))]:
        mod = mod.to(DEV).eval()
        with torch.no_grad():
            d = (mod(x)[:, :20] - mod(x2)[:, :20]).abs().max().item()
        assert d < 1e-4, (nm, d)
        print(f"[guard] causal OK  {nm}  max|delta_past|={d:.2e}", flush=True)

    def score_table(pm, name):
        c = R.ts_counts(y_ts, wot, (pm >= 0.5).astype(int), te)
        f, a = R.macro_f1_acc(c.sum(0))
        res[name] = dict(ts_macro_f1=f, ts_acc=a, _counts=c.tolist())
        print(f"  {name:26s} ts-MF1 {f:6.2f}  acc {a:6.2f}", flush=True)
        return c

    # ---------- carried arms, per-seed scores kept
    per_seed_scores = {}
    for tag, X in (("ALL", X_ALL), ("V", X_V)):
        for arm in ("A1_BCAST_CAUSAL", "A2_PERWIN", "A4_CTCN"):
            lam = R.LAMBDA_SMOOTH if arm.startswith("A4") else 0.0
            ps = []
            t0 = time.time()
            for sd in SEEDS:
                _, p = R.train_eval(arm, X, y_win, D["y_video"], tr, va, te, sd, lam,
                                    "ts_macro_f1", y_ts, wot)
                ps.append(p)
            P = np.stack(ps); per_seed_scores[f"{tag}/{arm}"] = P
            np.save(OUT / f"v2_probs_{tag}_{arm}.npy", P)
            score_table(P.mean(0), f"{tag}/{arm}")
            print(f"    ({time.time()-t0:.0f}s)", flush=True)

    # ---------- B2 DENSE
    for tag, X in (("ALL", X_ALL), ("V", X_V)):
        ps = []
        t0 = time.time()
        for sd in SEEDS:
            _, p = train_dense(X, D["y_multi"], tr, va, te, sd, y_ts, wot)
            ps.append(p)
        P = np.stack(ps); per_seed_scores[f"{tag}/B2_DENSE"] = P
        np.save(OUT / f"v2_probs_{tag}_B2_DENSE.npy", P)
        score_table(P.mean(0), f"{tag}/B2_DENSE")
        print(f"    ({time.time()-t0:.0f}s)", flush=True)

    # ---------- coverage budget regressor (TRAIN only)
    from sklearn.linear_model import Ridge

    cov = y_win.mean(1)
    prefix = np.cumsum(X_ALL, axis=1) / np.arange(1, K + 1).reshape(1, K, 1)
    ridge_full = Ridge(alpha=10.0).fit(X_ALL[tr].mean(1), cov[tr])
    ridge_pref = Ridge(alpha=10.0).fit(prefix[tr].reshape(-1, X_ALL.shape[-1]),
                                       np.repeat(cov[tr], K))
    bud_online = ridge_pref.predict(prefix[te].reshape(-1, X_ALL.shape[-1])).reshape(len(te), K)
    bud_offline = ridge_full.predict(X_ALL[te].mean(1))
    meta["budget_train_r"] = float(np.corrcoef(ridge_full.predict(X_ALL[tr].mean(1)), cov[tr])[0, 1])
    meta["budget_test_r"] = float(np.corrcoef(bud_offline, cov[te])[0, 1])
    print(f"[budget] coverage regressor pearson r: train {meta['budget_train_r']:.3f} "
          f"test {meta['budget_test_r']:.3f}", flush=True)

    Tmat = fit_transition(y_win, tr)
    meta["transition_matrix"] = Tmat.tolist()
    runs = []
    for i in tr:
        k = 0
        while k < K:
            j = k
            while j + 1 < K and y_win[i, j + 1] == y_win[i, k]:
                j += 1
            runs.append(j - k + 1); k = j + 1
    dur_min = int(np.percentile(runs, 5))
    meta["dur_min_p5"] = dur_min
    print(f"[budget] transition persistence {Tmat[0,0]:.3f}/{Tmat[1,1]:.3f}  dur_min_p5={dur_min}",
          flush=True)

    # ---------- decoders on A2 scores (and A4 scores, reported)
    gold_cov = np.repeat(cov[te][:, None], K, axis=1)
    for base in ("A2_PERWIN", "A4_CTCN"):
        P = per_seed_scores[f"ALL/{base}"].mean(0)
        for name, mode, bud in [("C0_UNCONSTRAINED", "unconstrained", None),
                                ("C1_COVBUD_ONLINE", "covbud", bud_online),
                                ("C1a_BUDGET_ONLY", "budget", bud_online),
                                ("C1b_TRANS_ONLY", "trans", None),
                                ("C3_ORACLE_BUDGET", "covbud", gold_cov)]:
            pred = np.stack([decode(P[j], mode, None if bud is None else bud[j], Tmat)
                             for j in range(len(te))])
            c = R.ts_counts(y_ts, wot, pred, te)
            f, a = R.macro_f1_acc(c.sum(0))
            res[f"DEC/{base}/{name}"] = dict(ts_macro_f1=f, ts_acc=a, _counts=c.tolist())
            print(f"  DEC/{base}/{name:22s} ts-MF1 {f:6.2f}  acc {a:6.2f}", flush=True)
        pred = np.stack([decode(P[j], "viterbi", bud_offline[j], Tmat, dur_min)
                         for j in range(len(te))])
        c = R.ts_counts(y_ts, wot, pred, te)
        f, a = R.macro_f1_acc(c.sum(0))
        res[f"DEC/{base}/C2_COVBUD_OFFLINE"] = dict(ts_macro_f1=f, ts_acc=a, _counts=c.tolist())
        print(f"  DEC/{base}/C2_COVBUD_OFFLINE   ts-MF1 {f:6.2f}  acc {a:6.2f}  [OFFLINE]", flush=True)

    # ---------- E1 / E2 objective control
    for name, w in (("E1_OBJ_BCE", 0.0), ("E2_OBJ_INTRA", 0.1)):
        ps = []
        t0 = time.time()
        for sd in SEEDS:
            _, p = train_obj(X_ALL, y_win, tr, va, te, sd, y_ts, wot, w)
            ps.append(p)
        P = np.stack(ps); np.save(OUT / f"v2_probs_ALL_{name}.npy", P)
        per_seed_scores[f"ALL/{name}"] = P
        score_table(P.mean(0), f"ALL/{name}")
        print(f"    ({time.time()-t0:.0f}s)", flush=True)

    # ---------- strata + bootstrap
    covt = cov[te]
    strat = {"LOW": covt < 0.25, "MID": (covt >= 0.25) & (covt < 0.75), "HIGH": covt >= 0.75}
    nrun = np.array([int(((y_win[i, 1:] == 1) & (y_win[i, :-1] == 0)).sum() + (y_win[i, 0] == 1))
                     for i in te])
    strat["MULTISPAN"] = nrun >= 2
    strat["SINGLESPAN"] = nrun == 1
    meta["strata_sizes"] = {k: int(v.sum()) for k, v in strat.items()}
    print(f"[strata] {meta['strata_sizes']}", flush=True)

    def mk(key, sub=None):
        c = np.array(res[key]["_counts"])
        if sub is None:
            return lambda idx: R.macro_f1_acc(c[idx].sum(0))[0], len(te)
        w = np.where(sub)[0]
        cc = c[w]
        return lambda idx: R.macro_f1_acc(cc[idx].sum(0))[0], len(w)

    contrasts = {}

    def add(nm, ka, kb, sub=None):
        fa, n = mk(ka, sub); fb, _ = mk(kb, sub)
        if n < 8:
            return
        d = fa(np.arange(n)) - fb(np.arange(n))
        lo, hi = R.boot_ci(fa, fb, n, np.random.default_rng(R.BOOT_SEED))
        contrasts[nm] = dict(delta=d, ci=[lo, hi], n=n)
        print(f"  [ci] {nm}: {d:+.3f} [{lo:+.3f},{hi:+.3f}]  (n={n})", flush=True)

    # gate (a)
    for tag in ("ALL", "V"):
        add(f"{tag}/B2_DENSE-A2_PERWIN", f"{tag}/B2_DENSE", f"{tag}/A2_PERWIN")
        add(f"{tag}/A4_CTCN-A2_PERWIN", f"{tag}/A4_CTCN", f"{tag}/A2_PERWIN")
        add(f"{tag}/B2_DENSE-A1_BCAST_CAUSAL", f"{tag}/B2_DENSE", f"{tag}/A1_BCAST_CAUSAL")
    # gate (b)
    for base in ("A2_PERWIN", "A4_CTCN"):
        for nm in ("C1_COVBUD_ONLINE", "C1a_BUDGET_ONLY", "C1b_TRANS_ONLY",
                   "C2_COVBUD_OFFLINE", "C3_ORACLE_BUDGET"):
            add(f"DEC/{base}/{nm}-C0", f"DEC/{base}/{nm}", f"DEC/{base}/C0_UNCONSTRAINED")
        for sname in ("MULTISPAN", "SINGLESPAN", "LOW", "MID", "HIGH"):
            add(f"DEC/{base}/C1-C0@{sname}", f"DEC/{base}/C1_COVBUD_ONLINE",
                f"DEC/{base}/C0_UNCONSTRAINED", strat[sname])
    # objective control
    add("ALL/E2-E1", "ALL/E2_OBJ_INTRA", "ALL/E1_OBJ_BCE")
    for sname in ("LOW", "MID", "HIGH", "MULTISPAN", "SINGLESPAN"):
        add(f"ALL/E2-E1@{sname}", "ALL/E2_OBJ_INTRA", "ALL/E1_OBJ_BCE", strat[sname])

    res["_contrasts"] = contrasts
    res["_meta"] = meta
    (OUT / "results_v2.json").write_text(json.dumps(res, indent=2))
    print(f"[done] wrote {OUT/'results_v2.json'}", flush=True)


if __name__ == "__main__":
    main()
