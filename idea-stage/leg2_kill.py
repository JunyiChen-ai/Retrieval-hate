#!/usr/bin/env python
"""LEG2-KILL — capped kill test of Human-Agreement leg (ii).

Agreement-weighted contrastive training (s_ij = q_i^T q_j) against the published GenSCL
label-distribution contrastive baseline (cosine label similarity) and a shuffled-q placebo.

ADAPTIVELY SELECTED HYPOTHESIS. Chosen after the frozen P-A-v2 gate failed in both languages.
Inherits no prior GO. The original gate stays failed regardless of this outcome. A positive
result grants only the label "exploratory"; a negative result permanently closes the entire
Human-Agreement family.

Decision rules are frozen in idea-stage/PILOT_FREEZE_2026-08-09.md, section "LEG2-KILL"
(sha256 14c803d1bbf408c193c0676dda7b46b7fb8f6117c3150287f1e32cade0a2f902 at implementation time)
and are NOT edited after results are seen.

Data loading, vote parsing, the feature key and the path guard are imported unchanged from the
P-A implementation. Zero test-set contact: P-A's guard is re-armed (any path component containing
"test" HALTs).

Usage:
  python idea-stage/leg2_kill.py --smoke synthetic
  python idea-stage/leg2_kill.py --smoke permuted
  python idea-stage/leg2_kill.py --out idea-stage/leg2_kill.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "idea-stage"))
import pilot_a_disagreement_retrievability as PA  # noqa: E402

Halt = PA.Halt
log = PA.log

# ---- frozen constants (PILOT_FREEZE_2026-08-09.md, LEG2-KILL) ----
CLASSES = ["Hateful", "Offensive", "Normal", "Counter Narrative"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

NFOLD = 5
INNER_NFOLD = 3
FOLD_SEEDS = [20260914, 20260915, 20260916]
PLACEBO_SEED = 20260920

H_DIM = 128
P_DIM = 64
DROPOUT = 0.2
LR = 1e-3
WD = 1e-2
STEPS = 400
TAU = 0.1
LAMBDA_GRID = [0.1, 0.3, 1.0, 3.0]
THRESH = 0.5

ARMS = ["A", "B", "C", "D"]
GATE_DELTA = 0.005

torch.set_num_threads(8)


# --------------------------------------------------------------------- data --
def build_q(ids, votes_table):
    """Empirical 4-class vote histogram, normalised. No smoothing (frozen)."""
    Q = np.zeros((len(ids), len(CLASSES)), dtype=np.float64)
    for i, v in enumerate(ids):
        vs = votes_table[v]
        if not vs:
            raise Halt("HALT_EMPTY_VOTES:%s" % v)
        for x in vs:
            if x not in CLASS_IDX:
                raise Halt("HALT_UNKNOWN_CLASS:%r" % x)
            Q[i, CLASS_IDX[x]] += 1.0
        Q[i] /= float(len(vs))
    if not np.allclose(Q.sum(axis=1), 1.0):
        raise Halt("HALT_Q_NOT_NORMALISED")
    return Q


def harm_fraction(Q):
    """Soft binary target: mass on {Hateful, Offensive}."""
    return Q[:, CLASS_IDX["Hateful"]] + Q[:, CLASS_IDX["Offensive"]]


def kernels(Q, perm):
    """Frozen label-similarity kernels. A has none (lambda = 0)."""
    nrm = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    Qn = Q / nrm
    Qp = Q[perm]
    K = {
        "B": Qn @ Qn.T,      # GenSCL cosine label similarity  (primary comparator)
        "C": Q @ Q.T,        # raw inner product = P(Y_i = Y_j) (candidate mechanism)
        "D": Qp @ Qp.T,      # shuffled-q placebo
    }
    if np.allclose(K["B"], K["C"]):
        raise Halt("HALT_KERNELS_B_C_IDENTICAL")
    if not np.allclose(np.sort(K["D"].ravel()), np.sort(K["C"].ravel())):
        raise Halt("HALT_PLACEBO_NOT_A_PERMUTATION_OF_C")
    return K


# -------------------------------------------------------------------- model --
class Head(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, H_DIM), nn.ReLU(), nn.Dropout(DROPOUT))
        self.proj = nn.Linear(H_DIM, P_DIM)
        self.clf = nn.Linear(H_DIM, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.clf(h).squeeze(1), F.normalize(self.proj(h), dim=1)


def gen_scl(z, Ksub):
    """GenSCL Eq. 2 (arXiv 2206.00384), verbatim form, full-batch anchors.

      mean_i [ -(1/|A(i)|) sum_{j in A(i)} simY(y_i,y_j) log softmax_j( z_i.z_j / tau ) ]

    No row-normalisation of simY, 1/|A(i)| prefactor kept as published.
    """
    n = z.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=z.device)
    sim = (z @ z.T) / TAU
    sim = sim.masked_fill(~off, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    w = Ksub * off
    return -(w * logp).sum(dim=1).div(n - 1).mean()


def fit_predict(Xtr, ytr, Ktr, Xte, lam, seed):
    """One frozen training run; returns held-out probabilities."""
    torch.manual_seed(int(seed))
    d = Xtr.shape[1]
    m = Head(d)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    npos = float(ytr.sum())
    nneg = float(len(ytr) - npos)
    if npos <= 0 or nneg <= 0:
        raise Halt("HALT_DEGENERATE_TRAIN_FOLD")
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / npos, dtype=torch.float32))
    Xt = torch.as_tensor(Xtr, dtype=torch.float32)
    yt = torch.as_tensor(ytr, dtype=torch.float32)
    Kt = None if Ktr is None else torch.as_tensor(Ktr, dtype=torch.float32)
    m.train()
    for _ in range(STEPS):
        logit, z = m(Xt)
        loss = bce(logit, yt)
        if Kt is not None and lam > 0:
            loss = loss + lam * gen_scl(z, Kt)
        loss.backward()
        opt.step()
        opt.zero_grad()
    m.eval()
    with torch.no_grad():
        logit, _ = m(torch.as_tensor(Xte, dtype=torch.float32))
        p = torch.sigmoid(logit).numpy().astype(np.float64)
    if not np.all(np.isfinite(p)):
        raise Halt("HALT_NONFINITE_PROBS")
    return p


def select_lambda(X, y, K, itr, seed):
    """Inner stratified 3-fold CV over the frozen lambda grid. Identical budget for B/C/D."""
    inner = StratifiedKFold(n_splits=INNER_NFOLD, shuffle=True, random_state=int(seed))
    scores = []
    for lam in LAMBDA_GRID:
        vals = []
        for a, b in inner.split(np.zeros(len(itr)), y[itr]):
            ia, ib = itr[a], itr[b]
            if len(set(y[ia])) < 2 or len(set(y[ib])) < 2:
                continue
            Ksub = None if K is None else K[np.ix_(ia, ia)]
            p = fit_predict(X[ia], y[ia], Ksub, X[ib], lam, seed)
            vals.append(f1_score(y[ib], (p > THRESH).astype(int), average="macro",
                                 zero_division=0))
        scores.append(float(np.mean(vals)) if vals else float("nan"))
    best = int(np.nanargmax(scores))          # ties -> smallest lambda
    return LAMBDA_GRID[best], scores


# ------------------------------------------------------------------ metrics --
def kl_bernoulli(f, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    f = np.clip(f, 0.0, 1.0)
    t1 = np.where(f > 0, f * np.log(np.maximum(f, 1e-12) / p), 0.0)
    t2 = np.where(f < 1, (1 - f) * np.log(np.maximum(1 - f, 1e-12) / (1 - p)), 0.0)
    return float(np.mean(t1 + t2))


def soft_f1(f, p):
    f1p = 2.0 * float((p * f).sum()) / max(float(p.sum() + f.sum()), 1e-12)
    f1n = 2.0 * float(((1 - p) * (1 - f)).sum()) / max(float((1 - p).sum() + (1 - f).sum()), 1e-12)
    return 0.5 * (f1p + f1n)


# ------------------------------------------------------------------ one run --
def run_lang(X, y, Q, seeds, arms=ARMS):
    """Per-seed OOF probabilities for every arm."""
    n = len(y)
    oof = {a: np.zeros((len(seeds), n), dtype=np.float64) for a in arms}
    book = {a: [] for a in arms}
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(PLACEBO_SEED + si)
        perm = rng.permutation(n)
        K = kernels(Q, perm)
        skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=int(seed))
        for fi, (itr, ite) in enumerate(skf.split(np.zeros(n), y)):
            rs = int(seed) + fi
            mu = X[itr].mean(axis=0)
            sd = np.maximum(X[itr].std(axis=0), 1e-8)
            Z = (X - mu) / sd
            for a in arms:
                Ka = None if a == "A" else K[a]
                if a == "A":
                    lam, isc = 0.0, None
                else:
                    lam, isc = select_lambda(Z, y, Ka, itr, rs)
                Ksub = None if Ka is None else Ka[np.ix_(itr, itr)]
                oof[a][si, ite] = fit_predict(Z[itr], y[itr], Ksub, Z[ite], lam, rs)
                book[a].append({"seed": int(seed), "fold": fi, "lambda": lam,
                                "inner_scores": isc})
        log("    seed %d done (%.1fs elapsed)" % (seed, time.time() - _T0))
    return oof, book


def analyse(oof, y, f, seeds):
    res = {}
    for a, P in oof.items():
        res[a] = {
            "macro_f1_per_seed": [float(f1_score(y, (P[si] > THRESH).astype(int),
                                                 average="macro", zero_division=0))
                                  for si in range(P.shape[0])],
            "kl_per_seed": [kl_bernoulli(f, P[si]) for si in range(P.shape[0])],
            "soft_f1_per_seed": [soft_f1(f, P[si]) for si in range(P.shape[0])],
        }
        for k in ("macro_f1", "kl", "soft_f1"):
            res[a][k] = float(np.mean(res[a][k + "_per_seed"]))
    return res


# ------------------------------------------------------------------ verdict --
def verdict(per_lang):
    """Transcribed frozen rule (PILOT_FREEZE_2026-08-09.md, LEG2-KILL):

      M(arm, s) = 0.5 * (macroF1_EN(arm,s) + macroF1_ZH(arm,s))
      d_CB(s) = M(C,s) - M(B,s);   d_CD(s) = M(C,s) - M(D,s)
      pass_CB := mean_s d_CB >= +0.005 AND d_CB(s) > 0 for all seeds
      pass_CD := mean_s d_CD >= +0.005 AND d_CD(s) > 0 for all seeds
      EXPLORATORY-GO iff pass_CB AND pass_CD ; otherwise FAMILY-CLOSED (no AMBIGUOUS branch).
    """
    langs = sorted(per_lang.keys())
    M = {}
    for a in ARMS:
        rows = np.array([per_lang[L][a]["macro_f1_per_seed"] for L in langs])  # [lang, seed]
        M[a] = rows.mean(axis=0)
    d_cb = M["C"] - M["B"]
    d_cd = M["C"] - M["D"]
    pass_cb = bool(float(d_cb.mean()) >= GATE_DELTA and bool(np.all(d_cb > 0)))
    pass_cd = bool(float(d_cd.mean()) >= GATE_DELTA and bool(np.all(d_cd > 0)))
    v = "EXPLORATORY-GO" if (pass_cb and pass_cd) else "FAMILY-CLOSED"
    return {
        "verdict": v,
        "M_per_seed": {a: [float(x) for x in M[a]] for a in ARMS},
        "M_mean": {a: float(M[a].mean()) for a in ARMS},
        "d_CB_per_seed": [float(x) for x in d_cb],
        "d_CD_per_seed": [float(x) for x in d_cd],
        "d_CB_mean": float(d_cb.mean()),
        "d_CD_mean": float(d_cd.mean()),
        "pass_CB": pass_cb,
        "pass_CD": pass_cd,
        "gate_delta": GATE_DELTA,
        "no_ambiguous_branch": True,
        "note": ("Futility rule in force (external review IDEA_REPORT 6.5/6.8): one failure "
                 "permanently closes the Human-Agreement family. A positive result grants only "
                 "the 'exploratory' label and does not revive the failed P-A/P-A-v2 gate."),
    }


DECLARATION = (
    "ADAPTIVELY SELECTED. This hypothesis was chosen after the frozen P-A-v2 gate failed in both "
    "languages, because leg (ii) is the only leg that gate did not touch. It inherits no prior "
    "GO. The original P-A/P-A-v2 gate stays failed regardless of this outcome. A positive result "
    "grants the label 'exploratory' only, never 'recommended'. A negative result permanently "
    "closes the entire Human-Agreement family (legs i, ii, iii)."
)


# ------------------------------------------------------------------- runner --
_T0 = time.time()


def main():
    global _T0
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", choices=["synthetic", "permuted"], default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    PA.arm_guard()
    _T0 = time.time()
    out = {
        "experiment": "LEG2-KILL — agreement-weighted contrastive objective, capped kill test",
        "declaration": DECLARATION,
        "freeze": "idea-stage/PILOT_FREEZE_2026-08-09.md (section LEG2-KILL)",
        "freeze_sha256_at_implementation":
            "14c803d1bbf408c193c0676dda7b46b7fb8f6117c3150287f1e32cade0a2f902",
        "mode": args.smoke or "real",
        "primary_comparator": "B = GenSCL (arXiv 2206.00384) Eq.2 with cosine label similarity",
        "arms": {"A": "BCE only (hard-label baseline head)",
                 "B": "GenSCL cosine label similarity on vote distributions",
                 "C": "raw inner product q_i^T q_j = P(Y_i = Y_j)  [candidate]",
                 "D": "shuffled-q placebo (global permutation of q, labels untouched)"},
        "hyper": {"h_dim": H_DIM, "p_dim": P_DIM, "dropout": DROPOUT, "lr": LR, "wd": WD,
                  "steps": STEPS, "tau": TAU, "lambda_grid": LAMBDA_GRID,
                  "nfold": NFOLD, "inner_nfold": INNER_NFOLD, "fold_seeds": FOLD_SEEDS,
                  "placebo_seed": PLACEBO_SEED, "threshold": THRESH,
                  "vote_smoothing": "none (frozen)"},
        "guard": "armed: any path containing 'test' HALTs",
        "per_lang": {},
    }

    if args.smoke == "synthetic":
        rng = np.random.default_rng(11)
        for L, n in (("EN", 160), ("ZH", 160)):
            X = rng.normal(size=(n, 200))
            y = (rng.random(n) > 0.7).astype(np.int64)
            cnt = rng.integers(0, 3, size=(n, 4)) + 1
            Q = cnt / cnt.sum(axis=1, keepdims=True)
            oof, _ = run_lang(X, y, Q, FOLD_SEEDS)
            out["per_lang"][L] = analyse(oof, y, harm_fraction(Q), FOLD_SEEDS)
        vb = verdict(out["per_lang"])
        log("SMOKE synthetic macro-F1: %s" % json.dumps(
            {L: {a: round(out["per_lang"][L][a]["macro_f1"], 4) for a in ARMS}
             for L in out["per_lang"]}))
        log("SMOKE synthetic verdict (meaningless, structure check only): %s" % vb["verdict"])
        log("elapsed %.1fs" % (time.time() - _T0))
        return

    # ---- real data load ----
    meta = {"files": {}}
    data = {}
    for L, cfg in PA.LANGS.items():
        ids, img, txt, ylab, origin = PA.load_lang(L)
        votes, n_alias, per_file = PA.load_votes(cfg["tsv"])
        meta["files"].update(per_file)
        missing = [v for v in ids if v not in votes]
        if missing:
            raise Halt("HALT_JOIN_FAILED:%d" % len(missing))
        Q = build_q(ids, votes)
        X = np.concatenate([PA.l2np(img), PA.l2np(txt)], axis=1)
        nv = np.array([len(votes[v]) for v in ids])
        log("%s: n=%d (train=%d val=%d) d=%d label+=%.4f meanHarmFrac=%.4f "
            "unanimous=%.4f vote-count hist=%s alias=%d"
            % (L, len(ids), origin.count("train"), origin.count("val"), X.shape[1],
               ylab.mean(), harm_fraction(Q).mean(), float((Q.max(axis=1) == 1.0).mean()),
               dict(zip(*[list(map(int, a)) for a in np.unique(nv, return_counts=True)])),
               n_alias))
        data[L] = dict(ids=ids, X=X, y=ylab, Q=Q, n_train=origin.count("train"),
                       n_val=origin.count("val"), n_alias=n_alias,
                       vote_hist={int(k): int(c) for k, c in zip(*np.unique(nv,
                                                                           return_counts=True))})

    if args.smoke == "permuted":
        rng = np.random.default_rng(999)      # NOT a frozen seed
        for L in PA.LANGS:
            d = data[L]
            p = rng.permutation(len(d["y"]))
            oof, _ = run_lang(d["X"], d["y"][p], d["Q"][p], FOLD_SEEDS[:1])
            out["per_lang"][L] = analyse(oof, d["y"][p], harm_fraction(d["Q"][p]),
                                         FOLD_SEEDS[:1])
        log("SMOKE permuted macro-F1 (targets destroyed, reveals no endpoint): %s"
            % json.dumps({L: {a: round(out["per_lang"][L][a]["macro_f1"], 4) for a in ARMS}
                          for L in out["per_lang"]}))
        log("elapsed %.1fs" % (time.time() - _T0))
        return

    # ---------------------------- REAL RUN (single submission) ----------------
    for L in PA.LANGS:
        d = data[L]
        log("%s: arms %s ..." % (L, ARMS))
        oof, book = run_lang(d["X"], d["y"], d["Q"], FOLD_SEEDS)
        r = analyse(oof, d["y"], harm_fraction(d["Q"]), FOLD_SEEDS)
        r["_meta"] = {"n": int(len(d["y"])), "n_train": d["n_train"], "n_val": d["n_val"],
                      "base_rate_label_pos": float(d["y"].mean()),
                      "vote_count_hist": d["vote_hist"],
                      "vote_alias_No_to_Normal": d["n_alias"],
                      "chosen_lambda_hist": {a: {str(x): int(sum(
                          1 for b in book[a] if b["lambda"] == x)) for x in LAMBDA_GRID}
                          for a in ARMS if a != "A"}}
        out["per_lang"][L] = r
        log("%s macro-F1  A=%.4f B=%.4f C=%.4f D=%.4f | KL A=%.4f B=%.4f C=%.4f D=%.4f"
            % (L, r["A"]["macro_f1"], r["B"]["macro_f1"], r["C"]["macro_f1"],
               r["D"]["macro_f1"], r["A"]["kl"], r["B"]["kl"], r["C"]["kl"], r["D"]["kl"]))

    out["verdict_block"] = verdict(out["per_lang"])
    out["meta"] = meta
    out["paths_touched"] = sorted(set(PA._TOUCHED))
    out["elapsed_sec"] = time.time() - _T0
    vb = out["verdict_block"]
    log("d_CB per seed = %s (mean %+.4f) | d_CD per seed = %s (mean %+.4f)"
        % ([round(x, 4) for x in vb["d_CB_per_seed"]], vb["d_CB_mean"],
           [round(x, 4) for x in vb["d_CD_per_seed"]], vb["d_CD_mean"]))
    log("VERDICT: %s" % vb["verdict"])
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log("wrote %s" % args.out)


if __name__ == "__main__":
    try:
        main()
    except Halt as e:
        log("HALT %s" % e)
        sys.exit(3)
