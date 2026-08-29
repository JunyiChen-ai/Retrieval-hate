#!/usr/bin/env python
"""R7-2 -- on-screen-text provenance rule channel at the decision layer.

Frozen design: idea-stage/R7_OCRPROV_FREEZE.md.

All four arms are derived from the SAME 30 head runs (HateMM, A0, seeds 100-129).
The head is never retrained.  Epoch selection uses dev macro-F1 (P1) / last epoch
(P2), identical to R6/R7-1; the trainlog parser is imported verbatim from
idea-stage/r6_audit/analyze_audit.py and cross-checked against the per-item dump.

Arms
  A0        head alone, test macro-F1 @0.5 at the selected epoch
  COMB0     logistic regression fitted on the VAL split with the single feature
            [head_logit]; test macro-F1 @0.5 on the combiner output
  COMBR     ... with [head_logit, 6 frozen provenance rule indicators]
  COMBRAND  ... with [head_logit, 6 random Bernoulli indicators whose rates are
            matched, per feature, to the rule indicators' marginal trigger rates
            over train+val] -- the combiner-overfitting control

The combiner is fitted on VAL because that is the only split on which the head's
logits are out of sample (see the freeze doc, deviation D1).  It is a single global
function of (head score, rule indicators) with one weight vector shared by every
item -- not a per-item selection -- so Law III is not engaged.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = "/home/jehc223/Retrieval-hate"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "r6_audit"))
sys.path.insert(0, HERE)
import analyze_audit as AA          # noqa: E402
from rules import FEATURE_NAMES, extract_binary   # noqa: E402

DS = "HateMM"
SEEDS = list(range(100, 130))
PROTOCOLS = ["P1", "P2"]
ARMS = ["A0", "COMB0", "COMBR", "COMBRAND"]
GO_BAR = 0.005
COVERAGE_FLOOR = 0.05
N_BOOT = 20000
BOOT_SEED = 20260817
RAND_SEED_BASE = 20260817000
LR_C = 1.0                 # frozen
OCR_TRAINVAL = os.path.join(ROOT, "data/OCR/HateMM/ocr_video.jsonl")
OCR_TEST = os.path.join(ROOT, "data/OCR/HateMM/ocr_video_test.jsonl")

PAIRS = [("COMBR", "A0"), ("COMBRAND", "A0"), ("COMB0", "A0"),
         ("COMBR", "COMB0"), ("COMBR", "COMBRAND")]


def macro_f1(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    return AA.macro_f1_from_cm(tp, fp, fn, tn)


def boot_ci(d, rng, n_boot=N_BOOT, level=95.0):
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    m = d[idx].mean(axis=1)
    lo_p = (100.0 - level) / 2.0
    lo, hi = np.percentile(m, [lo_p, 100.0 - lo_p])
    return float(lo), float(hi)


def load_ocr_feats():
    """video_id -> 6 binary provenance indicators.  Label-blind."""
    feats, texts = {}, {}
    for p in (OCR_TRAINVAL, OCR_TEST):
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                texts[d["video_id"]] = d.get("text", "") or ""
    for vid, t in texts.items():
        feats[vid] = np.array(extract_binary(t), dtype=float)
    return feats


def fit_predict(Xfit, yfit, Xte):
    """Standardise on the fit split, L2 logistic regression, predict at 0.5."""
    mu = Xfit.mean(axis=0)
    sd = Xfit.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    lr = LogisticRegression(C=LR_C, max_iter=2000, solver="lbfgs")
    lr.fit((Xfit - mu) / sd, yfit)
    prob = lr.predict_proba((Xte - mu) / sd)[:, 1]
    return (prob >= 0.5).astype(int), lr, mu, sd


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logging/runs/r7_ocrprov/logs")
    ap.add_argument("--scoredir", default="logging/runs/r7_ocrprov/scores")
    ap.add_argument("--out", default="idea-stage/r7_ocrprov/results.json")
    a = ap.parse_args()
    rng = np.random.default_rng(BOOT_SEED)

    ocr = load_ocr_feats()
    K = len(FEATURE_NAMES)

    out = {"what": "R7-2 on-screen-text provenance rule channel. Frozen: "
                   "idea-stage/R7_OCRPROV_FREEZE.md",
           "grid": "HateMM, 1 head arm (A0) x 30 seeds (100..129); 4 read-out arms "
                   "derived offline from the same runs",
           "feature_names": FEATURE_NAMES, "lr_C": LR_C,
           "n_boot": N_BOOT, "bootstrap_rng_seed": BOOT_SEED,
           "coverage": {}, "arms": {}, "deltas": {}, "verdict": {}}

    # ---------- parse the 30 runs ----------
    runs, dumps, bad = {}, {}, []
    for s in SEEDS:
        lp = os.path.join(ROOT, a.logdir, "%s_A0_s%d.trainlog" % (DS, s))
        dp = os.path.join(ROOT, a.scoredir, "%s_A0_s%d.jsonl" % (DS, s))
        if not (os.path.exists(lp) and os.path.exists(dp)):
            bad.append((s, "missing log or dump")); continue
        r, err = AA.parse(lp, DS)
        if r is None:
            bad.append((s, err)); continue
        d = {}
        with open(dp) as f:
            for line in f:
                o = json.loads(line)
                d[(o["split"], o["epoch"])] = o
        if len(d) != 60:
            bad.append((s, "dump has %d records, expected 60" % len(d))); continue
        runs[s] = r; dumps[s] = d
    if bad:
        for s, w in bad:
            print("  BAD seed %s: %s" % (s, w))
        raise SystemExit("HALT: %d of %d runs unusable." % (len(bad), len(SEEDS)))

    # ---------- rule-feature coverage (label-blind, reported separately) ----------
    any_split_ids = {sp: dumps[SEEDS[0]][(sp, 0)]["ids"] for sp in ("dev", "test")}
    train_ids = [json.loads(l)["id"] for l in
                 open(os.path.join(ROOT, "data/gt/HateMM/train.jsonl"))]
    for name, ids in (("train", train_ids), ("val", any_split_ids["dev"]),
                      ("test", any_split_ids["test"])):
        M = np.array([ocr[i] for i in ids])
        out["coverage"][name] = {
            "n": int(len(ids)),
            "per_feature_rate": {fn: float(M[:, k].mean())
                                 for k, fn in enumerate(FEATURE_NAMES)},
            "any_feature_rate": float((M.sum(axis=1) > 0).mean()),
            "mean_n_features_firing": float(M.sum(axis=1).mean())}
    trainval = np.array([ocr[i] for i in train_ids + any_split_ids["dev"]])
    rand_rates = trainval.mean(axis=0)
    out["random_control_rates"] = {fn: float(r) for fn, r
                                   in zip(FEATURE_NAMES, rand_rates)}

    # ---------- per-seed, per-protocol arm scores ----------
    per = {(p, arm): [] for p in PROTOCOLS for arm in ARMS}
    coef_acc = []
    for s in SEEDS:
        r = runs[s]
        for proto in PROTOCOLS:
            e = AA.select_epoch(r["dev"], proto)
            dev = dumps[s][("dev", e)]
            tst = dumps[s][("test", e)]
            zv = np.array(dev["logits"], float).reshape(-1, 1)
            zt = np.array(tst["logits"], float).reshape(-1, 1)
            yv = np.array(dev["labels"], int)
            yt = np.array(tst["labels"], int)

            # A0: head alone at 0.5 -- must reproduce the trainlog macro-F1
            a0 = macro_f1(yt, (zt.reshape(-1) >= 0.0).astype(int))
            logged = r["test"][e]["macro_f1"]
            if abs(a0 - logged) > 1e-6:
                raise SystemExit("HALT: seed %d %s A0 from dump %.6f != trainlog "
                                 "%.6f" % (s, proto, a0, logged))
            per[(proto, "A0")].append(a0)

            Rv = np.array([ocr[i] for i in dev["ids"]])
            Rt = np.array([ocr[i] for i in tst["ids"]])
            srng = np.random.default_rng(RAND_SEED_BASE + s)
            Qv = (srng.random((len(dev["ids"]), K)) < rand_rates).astype(float)
            Qt = (srng.random((len(tst["ids"]), K)) < rand_rates).astype(float)

            for arm, Xv, Xt in (("COMB0", zv, zt),
                                ("COMBR", np.hstack([zv, Rv]), np.hstack([zt, Rt])),
                                ("COMBRAND", np.hstack([zv, Qv]), np.hstack([zt, Qt]))):
                pred, lr, _, _ = fit_predict(Xv, yv, Xt)
                per[(proto, arm)].append(macro_f1(yt, pred))
                if arm == "COMBR" and proto == "P1":
                    coef_acc.append(lr.coef_.reshape(-1).tolist())

    for proto in PROTOCOLS:
        for arm in ARMS:
            v = np.array(per[(proto, arm)], float)
            out["arms"]["%s/%s" % (proto, arm)] = {
                "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "se": float(v.std(ddof=1) / np.sqrt(len(v))),
                "per_seed": [float(x) for x in v]}

    C = np.array(coef_acc)
    out["combr_P1_mean_standardised_coefs"] = {
        n: float(C[:, k].mean()) for k, n in
        enumerate(["head_logit"] + list(FEATURE_NAMES))}

    for proto in PROTOCOLS:
        for L, R in PAIRS:
            d = (np.array(per[(proto, L)], float) - np.array(per[(proto, R)], float))
            lo, hi = boot_ci(d, rng)
            out["deltas"]["%s/%s-%s" % (proto, L, R)] = {
                "mean": float(d.mean()), "std": float(d.std(ddof=1)),
                "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                "ci95": [lo, hi], "n_pos": int((d > 0).sum()), "n": int(len(d))}

    # ---------- frozen decision rule ----------
    dR = out["deltas"]["P1/COMBR-A0"]
    dQ = out["deltas"]["P1/COMBRAND-A0"]
    dC0 = out["deltas"]["P1/COMBR-COMB0"]
    cov = out["coverage"]["test"]["any_feature_rate"]
    c1 = dR["mean"] >= GO_BAR
    c2 = dR["ci95"][0] > 0 or dR["ci95"][1] < 0
    c3 = dQ["mean"] < GO_BAR                       # random control does not rise
    c4 = dC0["mean"] > 0                           # rules beat pure recalibration
    if c1 and c2 and c3 and c4:
        verdict = "GO" if cov >= COVERAGE_FLOOR else "GO-BUT-SUBSCALE"
    elif c1 and c2 and c3 and not c4:
        verdict = "TRICK"                          # gain is recalibration, not rules
    else:
        verdict = "KILL"
    out["verdict"] = {
        "verdict": verdict,
        "P1_COMBR_minus_A0": dR, "P1_COMBRAND_minus_A0": dQ,
        "P1_COMBR_minus_COMB0": dC0,
        "test_any_feature_coverage": cov, "coverage_floor": COVERAGE_FLOOR,
        "c1_bar": bool(c1), "c2_ci_excludes_0": bool(c2),
        "c3_random_control_flat": bool(c3), "c4_beats_recalibration": bool(c4),
        "P2_COMBR_minus_A0_mean": out["deltas"]["P2/COMBR-A0"]["mean"]}

    op = os.path.join(ROOT, a.out)
    with open(op, "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== rule-feature coverage (label-blind) ===")
    for sp in ("train", "val", "test"):
        c = out["coverage"][sp]
        print(" %-5s n=%3d  any=%.3f  %s" % (sp, c["n"], c["any_feature_rate"],
              "  ".join("%s=%.3f" % (k[:9], v)
                        for k, v in c["per_feature_rate"].items())))
    print("\n=== arm means ===")
    for proto in PROTOCOLS:
        for arm in ARMS:
            k = "%s/%s" % (proto, arm)
            print(" %-3s %-9s mean %.4f  std %.4f  se %.4f"
                  % (proto, arm, out["arms"][k]["mean"], out["arms"][k]["std"],
                     out["arms"][k]["se"]))
    print("\n=== paired deltas ===")
    for proto in PROTOCOLS:
        for L, R in PAIRS:
            d = out["deltas"]["%s/%s-%s" % (proto, L, R)]
            print(" %-3s %-18s mean %+.4f  ci95 [%+.4f, %+.4f]  %d/30 pos"
                  % (proto, "%s-%s" % (L, R), d["mean"], d["ci95"][0],
                     d["ci95"][1], d["n_pos"]))
    print("\n=== combiner mean standardised coefficients (P1, COMBR) ===")
    for k, v in out["combr_P1_mean_standardised_coefs"].items():
        print("  %-18s %+.4f" % (k, v))
    print("\n=== VERDICT: %s ===" % verdict)
    print(json.dumps({k: v for k, v in out["verdict"].items()
                      if not isinstance(v, dict)}, indent=1))
    print("wrote", op)


if __name__ == "__main__":
    main()
