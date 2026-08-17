#!/usr/bin/env python
"""P-A-v2 — strong-baseline retest of the disagreement retrievability gate.

Decision rules are frozen in idea-stage/PILOT_FREEZE_2026-08-09.md (section "P-A-v2") and are
NOT edited after results are seen.

Data loading, vote parsing, target construction, the similarity key and the AUROC estimator are
imported unchanged from the P-A implementation so that T1/T2 are byte-identical to P-A's.

Zero test-set contact: P-A's path guard is re-armed (any path containing "test" HALTs).

Usage:
  python idea-stage/pilot_a_v2_strong_baseline.py --smoke synthetic
  python idea-stage/pilot_a_v2_strong_baseline.py --smoke permuted
  python idea-stage/pilot_a_v2_strong_baseline.py --out idea-stage/pilot_a_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "idea-stage"))
import pilot_a_disagreement_retrievability as PA  # noqa: E402

Halt = PA.Halt
log = PA.log
auroc = PA.auroc

# ---- frozen constants (PILOT_FREEZE_2026-08-09.md, P-A-v2) ----
KNN = PA.KNN                      # 20, inherited
NFOLD = 5
INNER_NFOLD = 5
FOLD_SEEDS = [20260910, 20260911, 20260912]
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
BOOT = 2000
BOOT_SEED = 20260913
NULL_SEED = PA.NULL_SEED          # 20260909, inherited
NULL_LO, NULL_HI = PA.NULL_LO, PA.NULL_HI
ARMS = ["B1", "B2", "C", "D"]
NBR_FEATS = ["s_T1", "s_T2", "u_T1", "sd_T1", "s_frac", "s_disp", "w_mean", "w_max"]


# ------------------------------------------------------------------- model ---
def _fit_select(Xtr, ytr, seed):
    """Frozen recipe: standardise on train, pick C by inner stratified 5-fold AUROC."""
    mu = Xtr.mean(axis=0)
    sd = np.maximum(Xtr.std(axis=0), 1e-8)
    Z = (Xtr - mu) / sd
    skf = StratifiedKFold(n_splits=INNER_NFOLD, shuffle=True, random_state=int(seed))
    scores = []
    for c in C_GRID:
        vals = []
        for itr, ite in skf.split(Z, ytr):
            if len(set(ytr[itr])) < 2 or len(set(ytr[ite])) < 2:
                continue
            m = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=5000,
                                   class_weight="balanced")
            m.fit(Z[itr], ytr[itr])
            vals.append(auroc(ytr[ite], m.decision_function(Z[ite])))
        scores.append(float(np.mean(vals)) if vals else float("nan"))
    best = int(np.nanargmax(scores))          # ties -> smallest C (argmax takes first)
    m = LogisticRegression(C=C_GRID[best], penalty="l2", solver="lbfgs", max_iter=5000,
                           class_weight="balanced")
    m.fit(Z, ytr)
    return m, mu, sd, C_GRID[best], scores[best]


def fit_predict(X, y, itr, ite, seed):
    m, mu, sd, c, sc = _fit_select(X[itr], y[itr], seed)
    return m.decision_function((X[ite] - mu) / sd), c, sc


def unc_block(Xb, ylab, itr, ite, seed):
    """X_unc = [H(p), |p-0.5|] with p an OOF harmful-label probability (labels only, no votes).

    Nested: C is selected once on the outer training fold (implementation detail; a full
    nested-nested search is not affordable and buys nothing here), then an inner stratified
    5-fold produces p for the training rows while the refit on the whole outer training fold
    produces p for the held-out rows.
    """
    Xtr, ytr = Xb[itr], ylab[itr]
    m_full, mu, sd, c, _ = _fit_select(Xtr, ytr, seed)
    p = np.zeros(len(Xb), dtype=np.float64)
    p[ite] = m_full.predict_proba((Xb[ite] - mu) / sd)[:, 1]
    skf = StratifiedKFold(n_splits=INNER_NFOLD, shuffle=True, random_state=int(seed))
    Z = (Xtr - mu) / sd
    for a, b in skf.split(Z, ytr):
        mm = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=5000,
                                class_weight="balanced")
        mm.fit(Z[a], ytr[a])
        p[itr[b]] = mm.predict_proba(Z[b])[:, 1]
    q = np.clip(p, 1e-9, 1 - 1e-9)
    H = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return np.stack([H, np.abs(p - 0.5)], axis=1), c


# ------------------------------------------------------- neighbourhood block --
def sim_matrix(ids, img, txt):
    key = np.concatenate([PA.l2np(img), PA.l2np(txt)], axis=1)
    sim = key @ key.T
    n = len(ids)
    lexrank = np.empty(n, dtype=np.int64)
    lexrank[np.argsort(np.array(ids, dtype=object), kind="stable")] = np.arange(n)
    return sim, lexrank


def nbr_block(sim, lexrank, pool, t1, t2, frac):
    """Fold-restricted neighbourhood features for every row.

    `pool` = the training-fold indices; neighbours are drawn only from `pool`, minus self.
    Weights are P-A's dot-product similarities; ties broken lexicographically by video_id.
    """
    n = sim.shape[0]
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) <= KNN:
        raise Halt("HALT_POOL_TOO_SMALL:%d" % len(pool))
    S = sim[:, pool].copy()                       # [n, |pool|]
    lr = lexrank[pool]
    inpool = np.full(n, -1, dtype=np.int64)
    inpool[pool] = np.arange(len(pool))
    self_col = inpool                              # -1 if not in pool
    rows = np.arange(n)
    hit = self_col >= 0
    S[rows[hit], self_col[hit]] = -np.inf
    disp = 2.0 * frac * (1.0 - frac)
    X = np.zeros((n, len(NBR_FEATS)), dtype=np.float64)
    for i in range(n):
        order = np.lexsort((lr, -S[i]))[:KNN]
        j = pool[order]
        w = S[i, order]
        if not np.all(np.isfinite(w)):
            raise Halt("HALT_NONFINITE_WEIGHT")
        den = w.sum()
        if abs(den) < 1e-9:
            raise Halt("HALT_ZERO_WEIGHT_DENOM")
        X[i] = (
            float((w * t1[j]).sum() / den),
            float((w * t2[j]).sum() / den),
            float(t1[j].mean()),
            float(t1[j].std()),
            float((w * frac[j]).sum() / den),
            float((w * disp[j]).sum() / den),
            float(w.mean()),
            float(w.max()),
        )
    return X


# ----------------------------------------------------------------- one run ---
def run_lang(ids, img, txt, ylab, t1, t2, frac, target, seeds):
    """Returns per-seed OOF score vectors for every arm, plus bookkeeping."""
    sim, lexrank = sim_matrix(ids, img, txt)
    Xb = np.concatenate([PA.l2np(img), PA.l2np(txt)], axis=1)
    n = len(ids)
    out = {a: np.zeros((len(seeds), n)) for a in ARMS}
    out["C0"] = np.zeros((len(seeds), n))
    book = {"chosen_C": {a: [] for a in ARMS}, "chosen_C_unc": []}
    for si, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=int(seed))
        for fi, (itr, ite) in enumerate(skf.split(np.zeros(n), target)):
            rs = int(seed) + fi
            Xn = nbr_block(sim, lexrank, itr, t1, t2, frac)
            Xu, cu = unc_block(Xb, ylab, itr, ite, rs)
            book["chosen_C_unc"].append(cu)
            feats = {
                "B1": Xb,
                "B2": np.concatenate([Xb, Xu], axis=1),
                "C": Xn,
                "D": np.concatenate([Xb, Xn], axis=1),
            }
            for a in ARMS:
                s, c, _ = fit_predict(feats[a], target, itr, ite, rs)
                out[a][si, ite] = s
                book["chosen_C"][a].append(c)
            out["C0"][si, ite] = Xn[ite, 0]      # P-A's raw scalar under v2 folds
        log("    seed %d done" % seed)
    return out, book


# ---------------------------------------------------------------- bootstrap --
def boot_increments(target, scores, seed):
    """Paired bootstrap over queries; identical draws for every arm.

    Per resample: AUROC per arm per fold-seed, averaged over fold-seeds; increments are
    differences of those seed-averaged values.
    """
    rng = np.random.default_rng(seed)
    n = len(target)
    keys = list(scores.keys())
    acc = {k: [] for k in keys}
    nskip = 0
    for _ in range(BOOT):
        b = rng.integers(0, n, size=n)
        tb = target[b]
        if tb.sum() == 0 or tb.sum() == n:
            nskip += 1
            continue
        for k in keys:
            acc[k].append(float(np.mean([auroc(tb, scores[k][si][b])
                                         for si in range(scores[k].shape[0])])))
    A = {k: np.array(v) for k, v in acc.items()}
    return A, nskip


def ci(a):
    if a.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ------------------------------------------------------------------ verdict --
def verdict(per_lang):
    """Transcribed frozen rule (PILOT_FREEZE_2026-08-09.md, P-A-v2):

      condition(L) := AUROC(C,L) >= AUROC(B1,L) AND lower bound of 95% CI of (D-B1) at L > 0
      GO-STRONG   -- condition holds in both languages
      GO-ZH-ONLY  -- only in ZH ; GO-EN-ONLY -- only in EN
      KILL        -- AUROC(C,L) < AUROC(B1,L) in both languages
      AMBIGUOUS   -- anything else
      precedence: GO-STRONG > GO-ZH-ONLY/GO-EN-ONLY > KILL > AMBIGUOUS
    """
    cond, cge, dlb = {}, {}, {}
    for L, r in per_lang.items():
        cge[L] = bool(r["T1"]["auroc"]["C"] >= r["T1"]["auroc"]["B1"])
        dlb[L] = float(r["T1"]["increment_ci95"]["D_minus_B1"][0])
        cond[L] = bool(cge[L] and dlb[L] > 0)
    both = all(cond.values())
    only_zh = cond.get("ZH", False) and not cond.get("EN", False)
    only_en = cond.get("EN", False) and not cond.get("ZH", False)
    kill = all(not v for v in cge.values())
    if both:
        v = "GO-STRONG"
    elif only_zh:
        v = "GO-ZH-ONLY"
    elif only_en:
        v = "GO-EN-ONLY"
    elif kill:
        v = "KILL"
    else:
        v = "AMBIGUOUS"
    return {
        "verdict": v,
        "condition_met": cond,
        "C_ge_B1": cge,
        "D_minus_B1_ci_lower": dlb,
        "kill_condition_triggered": bool(kill),
        "literal_at_least_one_flag": bool(any(cond.values())),
        "precedence_note": ("Frozen precedence GO-STRONG > GO-ZH-ONLY/GO-EN-ONLY > KILL > "
                            "AMBIGUOUS. The brief's looser literal reading ('condition in at "
                            "least one language -> GO-STRONG') is reported separately as "
                            "literal_at_least_one_flag so the record can be re-adjudicated."),
    }


# ------------------------------------------------------------------- runner --
def analyse(target, scores, book, extra):
    A, nskip = boot_increments(target, scores, BOOT_SEED)
    res = {
        "n": int(len(target)),
        "base_rate": float(target.mean()),
        "auroc": {k: float(np.mean([auroc(target, scores[k][si])
                                    for si in range(scores[k].shape[0])]))
                  for k in scores},
        "auroc_per_seed": {k: [float(auroc(target, scores[k][si]))
                               for si in range(scores[k].shape[0])] for k in scores},
        "auroc_ci95": {k: ci(A[k]) for k in A},
        "increment": {}, "increment_ci95": {}, "increment_frac_positive": {},
        "boot": {"n_resamples": BOOT, "seed": BOOT_SEED, "n_skipped_degenerate": nskip},
    }
    pairs = [("C_minus_B1", "C", "B1"), ("D_minus_B1", "D", "B1"),
             ("C_minus_B2", "C", "B2"), ("D_minus_B2", "D", "B2"),
             ("B2_minus_B1", "B2", "B1"), ("C0_minus_B1", "C0", "B1")]
    for name, a, b in pairs:
        res["increment"][name] = res["auroc"][a] - res["auroc"][b]
        d = A[a] - A[b]
        res["increment_ci95"][name] = ci(d)
        res["increment_frac_positive"][name] = float((d > 0).mean()) if d.size else float("nan")
    res.update(extra)
    res["chosen_C_hist"] = {a: {str(c): int((np.array(book["chosen_C"][a]) == c).sum())
                                for c in C_GRID if (np.array(book["chosen_C"][a]) == c).sum()}
                            for a in ARMS}
    return res


def synth(n, rng):
    ids = ["syn_%04d" % i for i in range(n)]
    img = rng.normal(size=(n, 1024))
    txt = rng.normal(size=(n, 768))
    ylab = rng.integers(0, 2, size=n)
    t1 = rng.integers(0, 2, size=n)
    t2 = (t1 * rng.integers(0, 2, size=n)).astype(np.int64)
    frac = rng.random(n)
    return ids, img, txt, ylab, t1, t2, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", choices=["synthetic", "permuted"], default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    PA.arm_guard()
    t0 = time.time()
    out = {
        "pilot": "P-A-v2 strong-baseline retest of disagreement retrievability",
        "freeze": "idea-stage/PILOT_FREEZE_2026-08-09.md (section P-A-v2)",
        "mode": args.smoke or "real",
        "knn": KNN, "nfold": NFOLD, "inner_nfold": INNER_NFOLD,
        "fold_seeds": FOLD_SEEDS, "C_grid": C_GRID,
        "boot": BOOT, "boot_seed": BOOT_SEED, "null_seed": NULL_SEED,
        "nbr_features": NBR_FEATS,
        "guard": "armed: any path containing 'test' HALTs",
        "per_lang": {},
    }

    if args.smoke == "synthetic":
        rng = np.random.default_rng(7)
        for L, n in (("EN", 200), ("ZH", 200)):
            ids, img, txt, ylab, t1, t2, frac = synth(n, rng)
            sc, book = run_lang(ids, img, txt, ylab, t1, t2, frac, t1, FOLD_SEEDS[:1])
            out["per_lang"][L] = {"T1": analyse(t1, sc, book, {})}
        log("SMOKE synthetic AUROCs: %s" % json.dumps(
            {L: out["per_lang"][L]["T1"]["auroc"] for L in out["per_lang"]}, indent=1))
        log("elapsed %.1fs" % (time.time() - t0))
        return

    # ---- real data load ----
    meta = {"files": {}}
    data = {}
    for L, cfg in PA.LANGS.items():
        ids, img, txt, ylab, origin = PA.load_lang(L)
        votes, n_alias, per_file = PA.load_votes(cfg["tsv"])
        meta["files"].update(per_file)
        t1, t2, nvotes, maj_bin = PA.targets_from_votes(ids, votes)
        frac = np.array([sum(1 for x in votes[v] if x in PA.HARM_VOTES) / len(votes[v])
                         for v in ids], dtype=np.float64)
        log("%s: n=%d (train=%d val=%d) T1=%.4f T2=%.4f label+=%.4f maj-agree=%.4f"
            % (L, len(ids), origin.count("train"), origin.count("val"),
               t1.mean(), t2.mean(), ylab.mean(), float((maj_bin == ylab).mean())))
        data[L] = dict(ids=ids, img=img, txt=txt, ylab=ylab, t1=t1, t2=t2, frac=frac,
                       n_alias=n_alias, maj_agree=float((maj_bin == ylab).mean()),
                       n_train=origin.count("train"), n_val=origin.count("val"))

    if args.smoke == "permuted":
        rng = np.random.default_rng(999)     # NOT the frozen null seed
        for L in PA.LANGS:
            d = data[L]
            p = rng.permutation(len(d["ids"]))
            sc, book = run_lang(d["ids"], d["img"], d["txt"], d["ylab"][p], d["t1"][p],
                                d["t2"][p], d["frac"][p], d["t1"][p], FOLD_SEEDS[:1])
            out["per_lang"][L] = {"T1": {"auroc": {k: float(auroc(d["t1"][p], sc[k][0]))
                                                  for k in sc}}}
        log("SMOKE permuted AUROCs: %s" % json.dumps(
            {L: out["per_lang"][L]["T1"]["auroc"] for L in out["per_lang"]}, indent=1))
        log("elapsed %.1fs" % (time.time() - t0))
        return

    # ---------------------------- REAL RUN (single submission) ----------------
    for L in PA.LANGS:
        d = data[L]
        log("%s: T1 arms ..." % L)
        sc1, bk1 = run_lang(d["ids"], d["img"], d["txt"], d["ylab"], d["t1"], d["t2"],
                            d["frac"], d["t1"], FOLD_SEEDS)
        log("%s: T2 arms (secondary) ..." % L)
        sc2, bk2 = run_lang(d["ids"], d["img"], d["txt"], d["ylab"], d["t1"], d["t2"],
                            d["frac"], d["t2"], FOLD_SEEDS)
        extra = {"n_train": d["n_train"], "n_val": d["n_val"],
                 "vote_alias_No_to_Normal": d["n_alias"],
                 "majority_vs_cache_label_agreement": d["maj_agree"]}
        r = {"T1": analyse(d["t1"], sc1, bk1, extra),
             "T2": analyse(d["t2"], sc2, bk2, {})}

        # frozen null control: arm C on permuted T1, fold seed 20260910
        rng = np.random.default_rng(NULL_SEED)
        p = rng.permutation(len(d["ids"]))
        t1p, t2p, fracp = d["t1"][p], d["t2"][p], d["frac"][p]
        scn, _ = run_lang(d["ids"], d["img"], d["txt"], d["ylab"], t1p, t2p, fracp,
                          t1p, FOLD_SEEDS[:1])
        a_null = float(auroc(t1p, scn["C"][0]))
        r["null_auroc_C_on_permuted_T1"] = a_null
        r["null_seed"] = NULL_SEED
        r["null_in_range"] = bool(NULL_LO <= a_null <= NULL_HI)

        out["per_lang"][L] = r
        t = r["T1"]
        log("%s T1  B1=%.4f B2=%.4f C=%.4f D=%.4f C0=%.4f | C-B1=%+.4f D-B1=%+.4f "
            "CI(D-B1)=[%+.4f,%+.4f] null=%.4f"
            % (L, t["auroc"]["B1"], t["auroc"]["B2"], t["auroc"]["C"], t["auroc"]["D"],
               t["auroc"]["C0"], t["increment"]["C_minus_B1"], t["increment"]["D_minus_B1"],
               t["increment_ci95"]["D_minus_B1"][0], t["increment_ci95"]["D_minus_B1"][1],
               a_null))

    out["verdict_block"] = verdict(out["per_lang"])
    bad = [L for L in out["per_lang"] if not out["per_lang"][L]["null_in_range"]]
    out["null_control_pass"] = (len(bad) == 0)
    out["null_control_failed_langs"] = bad
    if bad:
        out["verdict_block"]["verdict"] = "VOID (null control out of [0.45,0.55]): " + \
            out["verdict_block"]["verdict"]
        log("!!! NULL CONTROL OUT OF RANGE for %s -- flagged VOID per the freeze" % bad)

    out["meta"] = meta
    out["paths_touched"] = sorted(set(PA._TOUCHED))
    out["elapsed_sec"] = time.time() - t0
    log("VERDICT: %s" % out["verdict_block"]["verdict"])
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log("wrote %s" % args.out)


if __name__ == "__main__":
    try:
        main()
    except Halt as e:
        log("HALT %s" % e)
        sys.exit(3)
