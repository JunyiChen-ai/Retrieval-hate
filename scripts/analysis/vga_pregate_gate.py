#!/usr/bin/env python
"""
vga_pregate_gate.py -- GATE FITTING AND READ-OUT for the VGA (C1) / VNQ (C2) $0 pregate.
Record: refine-logs/VGA_PREGATE_RECORD.md.  Spec: refine-logs/LITSWEEP6_RELGEN.md §2.

Consumes the per-item tables written by vga_pregate_emit.py.  Computes nothing that
touches a feature cache, a GPU, or the test split.

C1 / VGA
    Gate set        = items where the deployed vote and the F95 adjudication DISAGREE
                      (test-time computable: both decisions are computable without labels).
                      On agreement items the switch is a no-op by construction, so the
                      gate is fitted and evaluated exactly on the set where it can act.
    Gate target     = 1 if adjudication is right on this item (a "fix"), 0 if the
                      deployed vote is right (a "break").  Exactly one holds.
    Nesting         = for outer fold f, the gate is fitted ONLY on gate-set items whose
                      frozen F95 fold is not f; the operating point is chosen by an
                      INNER 5-fold CV inside that pool.  The gate never sees fold f.
    Emission        = adjudicated label iff the gate fires, deployed vote otherwise.

C2 / VNQ
    Selective prediction over the DEPLOYED vote's own errors.  Confidence scores are
    produced under the same nesting.  AUGRC (Traub et al., arXiv:2407.01032) is the
    primary metric; AURC and error-detection AUROC are reported alongside.

COST: CPU only, <= 8 threads.  Zero GPU / SLURM / Modal.  Zero test contact.
"""
import argparse
import hashlib
import json
import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = "/data/jehc223/RGCL"

# ------------------------------------------------------- FROZEN ANALYSIS CONSTANTS
INNER_FOLDS = 5
INNER_SEED = 17
LOGIT_C = 1.0
LOGIT_MAXITER = 1000
GBM_N = 100          # "shallow GBM" per LITSWEEP6_RELGEN §2 C1 transplant sketch (iii)
GBM_DEPTH = 2
GBM_LR = 0.05
GBM_SEED = 0
N_PERM = 200         # K-VGA-2 permutation null
PERM_SEED = 12345
TARGET_GAIN = 0.030  # the +0.030 acc bar of K-VGA-1

PRIMARY_ADJ = "adj_max_pred"      # F95 PRIMARY cell (fused x MLP x max)
SECONDARY_ADJ = "adj_mean3_pred"  # SECONDARY; cannot carry a pass
PRIMARY_GATE_MODEL = "logistic"   # lower capacity, appropriate at n_gate ~ 90-111


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def macro_f1(y, p):
    y = np.asarray(y); p = np.asarray(p)
    fs = []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(fs))


def make_model(kind):
    if kind == "logistic":
        return LogisticRegression(penalty="l2", C=LOGIT_C, solver="lbfgs",
                                  max_iter=LOGIT_MAXITER)
    if kind == "gbm":
        return GradientBoostingClassifier(n_estimators=GBM_N, max_depth=GBM_DEPTH,
                                          learning_rate=GBM_LR, random_state=GBM_SEED)
    raise ValueError(kind)


def _fit_predict(kind, Xtr, ytr, Xte):
    """Standardise on the fitting rows only, fit, return P(y=1) on Xte.
    Degenerate single-class fitting sets return the constant base rate."""
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr.mean()))
    m = make_model(kind)
    m.fit(A, ytr)
    return m.predict_proba(B)[:, 1]


def choose_threshold(scores, y):
    """Operating point = the threshold maximising net gain (fixes - breaks) among
    fired items, computed on INNER-fold out-of-sample scores only.  +inf (fire on
    nothing, net 0) is always a candidate; ties resolve to the most conservative
    (highest) threshold."""
    cand = np.concatenate([np.unique(scores), [np.inf]])
    best_t, best_net = np.inf, 0
    for t in np.sort(cand)[::-1]:
        fired = scores >= t
        net = int((y[fired] == 1).sum()) - int((y[fired] == 0).sum())
        if net > best_net:
            best_net, best_t = net, float(t)
    return best_t, best_net


def run_gate(feat, y_gate, gate_idx, fold_of, kind, perm_rng=None):
    """Nested gate.  Returns a boolean fire mask aligned with gate_idx.
    If perm_rng is given, the FITTING-pool targets are shuffled (K-VGA-2 null);
    the evaluated fold's true targets are untouched."""
    fired = np.zeros(len(gate_idx), dtype=bool)
    gf = fold_of[gate_idx]
    for f in np.unique(fold_of):
        pool = np.flatnonzero(gf != f)
        ev = np.flatnonzero(gf == f)
        if len(ev) == 0:
            continue
        Xp, yp = feat[pool], y_gate[pool].copy()
        if perm_rng is not None:
            yp = yp[perm_rng.permutation(len(yp))]
        if len(np.unique(yp)) < 2 or len(pool) < INNER_FOLDS * 2:
            continue
        # ---- inner CV -> out-of-sample scores on the pool -> operating point
        oof = np.zeros(len(pool))
        inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                                random_state=INNER_SEED)
        for tr, te in inner.split(Xp, yp):
            oof[te] = _fit_predict(kind, Xp[tr], yp[tr], Xp[te])
        thr, _ = choose_threshold(oof, yp)
        if not np.isfinite(thr):
            continue
        # ---- refit on the whole pool, apply the inner-chosen threshold to fold f
        s_ev = _fit_predict(kind, Xp, yp, feat[ev])
        fired[ev] = s_ev >= thr
    return fired


def gate_readout(D, fired, gate_idx, y_gate, tag):
    """End-to-end read of one gate arm against the deployed vote."""
    lab, dep, adj, fold = D["lab"], D["dep"], D["adj"], D["fold"]
    n = len(lab)
    emit = dep.copy()
    emit[gate_idx[fired]] = adj[gate_idx[fired]]
    nf = int(fired.sum())
    fx = int(((y_gate == 1) & fired).sum())
    bk = int(((y_gate == 0) & fired).sum())
    fold_d = []
    for f in np.unique(fold):
        m = fold == f
        fold_d.append(round(acc(lab[m], emit[m]) - acc(lab[m], dep[m]), 4))
    signs = "".join("+" if v > 0 else ("-" if v < 0 else "0") for v in fold_d)
    return {
        "arm": tag,
        "n_gate_set": int(len(gate_idx)),
        "n_fired": nf,
        "fire_rate_of_gateset": round(nf / len(gate_idx), 4) if len(gate_idx) else None,
        "fire_rate_of_items": round(nf / n, 4),
        "fired_fixes": fx, "fired_breaks": bk, "net": fx - bk,
        "gate_precision": round(fx / nf, 4) if nf else None,
        "exchange_rate_fired": round(fx / bk, 4) if bk else None,
        "required_precision_at_this_fire_count": (
            round(0.5 + (TARGET_GAIN / 2.0) * n / nf, 4) if nf else None),
        "acc_emitted": round(acc(lab, emit), 4),
        "dacc_vs_deployed": round(acc(lab, emit) - acc(lab, dep), 4),
        "mF1_emitted": round(macro_f1(lab, emit), 4),
        "dmF1_vs_deployed": round(macro_f1(lab, emit) - macro_f1(lab, dep), 4),
        "fold_deltas": fold_d, "fold_signs": signs,
        "n_folds_positive": int(sum(1 for v in fold_d if v > 0)),
        "n_folds_nonneg": int(sum(1 for v in fold_d if v >= 0)),
        "posrate_emitted": round(float(emit.mean()), 4),
    }


# ------------------------------------------------------------------ C2 / VNQ
def augrc(conf, err):
    """Area under the GENERALISED risk-coverage curve (Traub et al., arXiv:2407.01032):
    at each coverage level the risk is the rate of UNDETECTED failures relative to the
    WHOLE set, so AUGRC = mean_k (#errors among the k most confident) / N.  Lower is
    better.  AURC (selective risk, denominator k) is returned alongside."""
    o = np.argsort(-np.asarray(conf, dtype="float64"), kind="stable")
    e = np.asarray(err, dtype="float64")[o]
    N = len(e)
    cum = np.cumsum(e)
    k = np.arange(1, N + 1)
    return float((cum / N).mean()), float((cum / k).mean())


def vnq_scores(D, feats, kind="logistic"):
    """Nested confidence in the DEPLOYED prediction: fit on items outside the outer
    fold, predict the fold.  Higher = more confident."""
    fold, lab, dep = D["fold"], D["lab"], D["dep"]
    correct = (dep == lab).astype(int)
    out = np.zeros(len(lab))
    for f in np.unique(fold):
        tr = np.flatnonzero(fold != f)
        te = np.flatnonzero(fold == f)
        out[te] = _fit_predict(kind, feats[tr], correct[tr], feats[te])
    return out


def run_dataset(key, emit_path, log):
    E = json.load(open(emit_path))
    lab = np.asarray(E["lab"]); fold = np.asarray(E["fold"])
    F = {k: np.asarray(v) for k, v in E["feat"].items()}
    meta = E["meta"]
    n = len(lab)
    res = {"meta": {"dataset": key, "n": n, "emit": os.path.basename(emit_path),
                    "emit_script_sha256": meta["script_sha256"],
                    "parity": meta["parity"]["n_pass"],
                    "parity_total": meta["parity"]["n_gates"],
                    "posrate_bank": round(float(lab.mean()), 4)}}

    def block(adj_key, label):
        dep = np.asarray(E["pred"]["dep_pred"])
        adj = np.asarray(E["pred"][adj_key])
        D = {"lab": lab, "dep": dep, "adj": adj, "fold": fold}
        gate_idx = np.flatnonzero(dep != adj)
        y_gate = (adj[gate_idx] == lab[gate_idx]).astype(int)
        n_fix, n_brk = int(y_gate.sum()), int((1 - y_gate).sum())
        out = {"adjudicator": label,
               "acc_deployed": round(acc(lab, dep), 4),
               "mF1_deployed": round(macro_f1(lab, dep), 4),
               "acc_adjudicated_ungated": round(acc(lab, adj), 4),
               "dacc_ungated": round(acc(lab, adj) - acc(lab, dep), 4),
               "gate_set_N": len(gate_idx), "n_fix": n_fix, "n_break": n_brk,
               "base_rate_fix": round(n_fix / len(gate_idx), 4),
               "oracle_gate_dacc": round(n_fix / n, 4),
               "breakeven_precision_full_coverage": round(
                   0.5 + (TARGET_GAIN / 2.0) * n / len(gate_idx), 4),
               "arms": {}, "perm": {}}

        sets = {"verifier": meta["vga_feats"],
                "f47ctrl": meta["f47_feats"],
                "f47ctrl_full": meta["f47_full_feats"]}
        for sname, cols in sets.items():
            X = np.stack([F[c][gate_idx] for c in cols], axis=1)
            for kind in ("logistic", "gbm"):
                tag = f"{sname}:{kind}"
                fired = run_gate(X, y_gate, gate_idx, fold, kind)
                out["arms"][tag] = gate_readout(D, fired, gate_idx, y_gate, tag)
                log(f"    [{key}/{label}] {tag:24s} fired {int(fired.sum()):3d} "
                    f"prec {out['arms'][tag]['gate_precision']} "
                    f"dacc {out['arms'][tag]['dacc_vs_deployed']:+.4f} "
                    f"{out['arms'][tag]['fold_signs']}")

        # references: oracle ceiling and ungated (fire-on-everything)
        out["arms"]["oracle"] = gate_readout(
            D, y_gate.astype(bool), gate_idx, y_gate, "oracle")
        out["arms"]["fire_all"] = gate_readout(
            D, np.ones(len(gate_idx), bool), gate_idx, y_gate, "fire_all")

        # ---- K-VGA-2 permutation null, for every fitted arm
        for sname, cols in sets.items():
            X = np.stack([F[c][gate_idx] for c in cols], axis=1)
            for kind in ("logistic", "gbm"):
                tag = f"{sname}:{kind}"
                obs = out["arms"][tag]["dacc_vs_deployed"]
                rng = np.random.RandomState(PERM_SEED)
                null = []
                for _ in range(N_PERM):
                    fr = run_gate(X, y_gate, gate_idx, fold, kind, perm_rng=rng)
                    e = np.asarray(E["pred"]["dep_pred"]).copy()
                    e[gate_idx[fr]] = adj[gate_idx[fr]]
                    null.append(acc(lab, e) - acc(lab, dep))
                null = np.asarray(null)
                out["perm"][tag] = {
                    "observed_dacc": obs, "n_perm": N_PERM,
                    "null_mean": round(float(null.mean()), 4),
                    "null_sd": round(float(null.std()), 4),
                    "null_q95": round(float(np.quantile(null, 0.95)), 4),
                    "null_max": round(float(null.max()), 4),
                    "p_value": round(float((1 + (null >= obs - 1e-12).sum())
                                           / (N_PERM + 1)), 4)}
                log(f"    [{key}/{label}] PERM {tag:24s} obs {obs:+.4f} "
                    f"null {null.mean():+.4f}+-{null.std():.4f} "
                    f"p={out['perm'][tag]['p_value']}")
        return out

    res["primary"] = block(PRIMARY_ADJ, "mlp_max (PRIMARY)")
    res["secondary"] = block(SECONDARY_ADJ, "mlp_mean3 (SECONDARY)")

    # ---------------------------------------------------------------- C2 / VNQ
    dep = np.asarray(E["pred"]["dep_pred"])
    D = {"lab": lab, "dep": dep, "fold": fold}
    err = (dep != lab).astype(int)
    scores = {
        "vnq_fitted": vnq_scores(D, np.stack([F[c] for c in meta["vga_feats"]], 1)),
        "knnue_fitted": vnq_scores(D, np.stack([F[c] for c in meta["knnue_feats"]], 1)),
        "vote_margin": np.abs(F["dep_vote"]),
        "vnq_raw_absgap": np.abs(F["v_gap"]),
    }
    c2 = {"n": n, "n_errors": int(err.sum()), "arms": {}}
    for name, s in scores.items():
        a_g, a_r = augrc(s, err)
        per_fold = []
        for f in np.unique(fold):
            m = fold == f
            per_fold.append(round(augrc(s[m], err[m])[0], 4))
        c2["arms"][name] = {
            "AUGRC": round(a_g, 4), "AURC": round(a_r, 4),
            "AUROC_error_detection": round(float(roc_auc_score(err, -s)), 4),
            "AUGRC_per_fold": per_fold}
    for base in ("knnue_fitted", "vote_margin"):
        d = [c2["arms"][base]["AUGRC_per_fold"][i] - c2["arms"]["vnq_fitted"]["AUGRC_per_fold"][i]
             for i in range(len(np.unique(fold)))]
        c2[f"vnq_vs_{base}"] = {
            "dAUGRC_pooled_improvement": round(
                c2["arms"][base]["AUGRC"] - c2["arms"]["vnq_fitted"]["AUGRC"], 4),
            "fold_improvements": [round(v, 4) for v in d],
            "fold_signs": "".join("+" if v > 0 else ("-" if v < 0 else "0") for v in d),
            "n_folds_positive": int(sum(1 for v in d if v > 0))}
        log(f"    [{key}] VNQ vs {base}: dAUGRC "
            f"{c2[f'vnq_vs_{base}']['dAUGRC_pooled_improvement']:+.4f} "
            f"{c2[f'vnq_vs_{base}']['fold_signs']}")
    res["c2_vnq"] = c2
    return res


# ------------------------------------------------------------------- self-test
def selftest(log):
    """Machinery validity, run BEFORE the freeze on synthetic data (F95 §2.5 pattern).
    Arm A: fix/break IS a function of the features -> the harness must return a large
    positive that beats its own permutation null.  Arm B: features are pure noise ->
    the harness must return ~0 and a non-significant p.  This proves a null below is a
    property of the data, not of the code."""
    out = {}
    for arm, signal in (("A_signal", True), ("B_noise", False)):
        rng = np.random.RandomState(7)
        n, ngate = 700, 110
        lab = rng.randint(0, 2, n)
        fold = np.repeat(np.arange(5), n // 5)[:n]
        rng.shuffle(fold)
        dep = lab.copy()
        gate_idx = rng.choice(n, ngate, replace=False)
        y = rng.binomial(1, 0.47, ngate)          # ~F95 base rate of "fix"
        adj = dep.copy()
        adj[gate_idx] = 1 - dep[gate_idx]         # disagreement by construction
        dep[gate_idx[y == 1]] = 1 - lab[gate_idx[y == 1]]   # adjudication right -> fix
        adj[gate_idx] = 1 - dep[gate_idx]
        y_true = (adj[gate_idx] == lab[gate_idx]).astype(int)
        X = rng.randn(ngate, 6)
        if signal:
            X[:, 0] += 2.2 * y_true
        D = {"lab": lab, "dep": dep, "adj": adj, "fold": fold}
        r = {}
        for kind in ("logistic", "gbm"):
            fired = run_gate(X, y_true, gate_idx, fold, kind)
            ro = gate_readout(D, fired, gate_idx, y_true, kind)
            rngp = np.random.RandomState(PERM_SEED)
            null = []
            for _ in range(60):
                fr = run_gate(X, y_true, gate_idx, fold, kind, perm_rng=rngp)
                e = dep.copy(); e[gate_idx[fr]] = adj[gate_idx[fr]]
                null.append(acc(lab, e) - acc(lab, dep))
            null = np.asarray(null)
            ro["perm_p"] = round(float((1 + (null >= ro["dacc_vs_deployed"] - 1e-12).sum()) / 61), 4)
            ro["perm_null_mean"] = round(float(null.mean()), 4)
            r[kind] = ro
            log(f"  SELFTEST {arm}/{kind}: dacc {ro['dacc_vs_deployed']:+.4f} "
                f"prec {ro['gate_precision']} p={ro['perm_p']}")
        out[arm] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--datasets", default="hatemm,zh,en")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    OUT = {"meta": {"script": os.path.abspath(__file__),
                    "script_sha256": sha256_of(os.path.abspath(__file__)),
                    "frozen": dict(INNER_FOLDS=INNER_FOLDS, INNER_SEED=INNER_SEED,
                                   LOGIT_C=LOGIT_C, GBM_N=GBM_N, GBM_DEPTH=GBM_DEPTH,
                                   GBM_LR=GBM_LR, GBM_SEED=GBM_SEED, N_PERM=N_PERM,
                                   PERM_SEED=PERM_SEED, TARGET_GAIN=TARGET_GAIN,
                                   PRIMARY_ADJ=PRIMARY_ADJ,
                                   PRIMARY_GATE_MODEL=PRIMARY_GATE_MODEL),
                    "test_contact": "NONE"}}
    if a.selftest:
        OUT["selftest"] = selftest(log)
    else:
        for key in a.datasets.split(","):
            p = os.path.join(REPO, f"scripts/analysis/vga_emit_{key}_OUT.json")
            log(f"[{key}] reading {p}")
            OUT[key] = run_dataset(key, p, log)
    json.dump(OUT, open(a.out, "w"), indent=1)
    log(f"DONE -> {a.out}")
    logf.close()


if __name__ == "__main__":
    main()
