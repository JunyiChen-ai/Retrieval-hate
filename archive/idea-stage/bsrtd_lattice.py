"""B-SRTD shared data contract — lattice IO, cell order, finite differences, G0 gate.

Frozen by research-wiki/EXP_bsrtd_prereg.md.  Nothing in this file may be changed after the
freeze timestamp recorded there without a numbered deviation record.

CELL ORDER IS FROZEN
    index 0 = "orig"       axis A off, axis B off
    index 1 = "targetsub"  axis A ON  (attack target substituted), axis B off
    index 2 = "stancerev"  axis A off, axis B ON  (stance reversed: assert <-> condemn)
    index 3 = "both"       axis A ON, axis B ON

FINITE DIFFERENCES (frozen)
    dA  = c[1] - c[0]                       first difference along axis A
    dB  = c[2] - c[0]                       first difference along axis B
    dAB = (c[3] - c[2]) - (c[1] - c[0])     mixed second difference

The same three operators are applied to the teacher's score vector and to the student's
per-cell probability vector.  Both live in [0, 1], so the residuals are scale-free and the
loss carries no temperature hyper-parameter.

This module also contains `make_synth`, which fabricates a *synthetic* lattice + feature +
teacher-score bundle for implementation rehearsal.  Synthetic bundles are written under
`idea-stage/bsrtd_synth/` and are never read by a primary run (the pilot refuses to report a
primary verdict unless every teacher row carries a non-synthetic engine tag).
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSRTD_DIR = os.path.join(ROOT, "data", "Counterfactual", "BSRTD")
TEACHER_PATH = os.path.join(BSRTD_DIR, "teacher_scores.jsonl")

CELLS = ("orig", "targetsub", "stancerev", "both")
LANG2DS = {"en": "MHC", "zh": "MHC_zh"}
SPLIT2CACHE = {"train": "train", "val": "dev_seen"}
STUDENT_TAG = "Qwen2.5-VL-7B-Instruct_HF"
CELL_TAG = "bsrtdcells_" + STUDENT_TAG

# Teacher prompt identity: bumping this invalidates every cached score.
# T1 = single verbalized 0-100 rating (superseded 2026-08-10 on the verbalized-confidence
# challenge).  T2 = binary Yes/No judgement read from token logprobs, averaged over three
# frozen semantically-equivalent paraphrases, with a verbalized fallback.
PROMPT_ID = "BSRTD-T2"
PROMPT_VARIANTS = ("a", "b", "c")
SCORING_MODES = ("logprob", "verbalized")


# --------------------------------------------------------------------------- lattice IO
def lattice_path(split, lang, root=BSRTD_DIR):
    assert split in ("train", "val"), f"split {split!r} is not allowed (test is never touched)"
    return os.path.join(root, f"{split}_lattices_{lang}.jsonl")


def load_lattices(split, lang, root=BSRTD_DIR, require=True):
    """Read one lattice file.  Returns [] if absent and require=False."""
    p = lattice_path(split, lang, root)
    if not os.path.exists(p):
        if require:
            raise FileNotFoundError(p)
        return []
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def verify_pass(ver):
    """True iff the build agent's rubric passed on every criterion of every generated cell.

    Handles the builder's schema (`data/Counterfactual/BSRTD/BUILD_RECORD_2026-08-10.md` §4):
        {"verdicts": {"targetsub": {"axis_fidelity": "PASS", "fluency": "PASS",
                                    "label_consistency": "PASS", "minimal_pair": "PASS",
                                    "reason": "..."}, ...}, "round": 1}
    A plain {"pass": bool} is also accepted.  Absent verification info is not treated as a
    failure (the G0.a fraction then simply reads 1.0 and the clause is uninformative).
    """
    if not isinstance(ver, dict):
        return True
    if "pass" in ver:
        return bool(ver["pass"])
    vd = ver.get("verdicts")
    if not isinstance(vd, dict) or not vd:
        return True
    for crit in vd.values():
        if not isinstance(crit, dict):
            continue
        for k, v in crit.items():
            if k == "reason":
                continue
            if isinstance(v, str) and v.strip().upper() != "PASS":
                return False
            if isinstance(v, bool) and not v:
                return False
    return True


def normalise_row(r, split, lang):
    """Coerce one lattice record into the frozen internal form; raise on contract violation."""
    sid = str(r["seed_id"])
    cells = r["cells"]
    missing = [c for c in CELLS if c not in cells or not str(cells[c]).strip()]
    if missing:
        raise ValueError(f"{lang}/{split}/{sid}: missing/empty cells {missing}")
    exp = r.get("cell_expected_labels") or {}
    y_exp = [int(exp.get(c, r["seed_label"] if c == "orig" else -1)) for c in CELLS]
    ver = r.get("verify") or {}
    ok = verify_pass(ver)
    return {
        "seed_id": sid,
        "split": str(r.get("split", split)),
        "lang": str(r.get("lang", lang)),
        "seed_label": int(r["seed_label"]),
        "texts": [str(cells[c]) for c in CELLS],
        "expected": y_exp,
        "verify_pass": ok,
        "verify": ver,
    }


def text_sha(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- teacher IO
def teacher_key(lang, split, seed_id, cell):
    return f"{lang}|{split}|{seed_id}|{cell}"


def load_teacher_scores(path=TEACHER_PATH):
    """key -> row (last write wins).  Missing file -> {}."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from an interrupted run
            out[r["key"]] = r
    return out


# --------------------------------------------------------------------------- differences
def finite_differences(c):
    """c: (..., 4) tensor/array in cell order.  Returns (dA, dB, dAB) with shape (...,)."""
    if isinstance(c, torch.Tensor):
        return c[..., 1] - c[..., 0], c[..., 2] - c[..., 0], \
            (c[..., 3] - c[..., 2]) - (c[..., 1] - c[..., 0])
    c = np.asarray(c)
    return c[..., 1] - c[..., 0], c[..., 2] - c[..., 0], \
        (c[..., 3] - c[..., 2]) - (c[..., 1] - c[..., 0])


# --------------------------------------------------------------------------- G0 gate
# Floors are POOLED across the two languages, because that is how the MVE precondition in
# IDEA_REPORT.md sec 8.4 ("≥200 train + 80 val balanced 2×2 lattices") is realised by the build:
# BUILD_RECORD_2026-08-10.md sec 2.5 quotas 110 train per language (220 total) and 40 en + 44 zh
# val (84 total).  Per-language minima are set below the quota so that ordinary verification
# attrition does not HALT the pilot, while a language that collapsed still would.
G0_MIN_TRAIN = 200      # train lattices, POOLED over languages
G0_MIN_VAL = 80         # val lattices, POOLED over languages
G0_MIN_TRAIN_LANG = 75  # train lattices in EACH language
G0_MIN_VAL_LANG = 28    # val lattices in EACH language
G0_MIN_VERIFY = 0.80    # fraction of lattices passing the builder's verification rubric
G0_MIN_CLASS = 0.30     # each seed class >= 30% of lattices
G0_MIN_SIGN = 0.60      # teacher dB sign agrees with the intended direction
G0_MIN_MOVE = 0.05      # mean |dB| of the teacher score
G0_MIN_PROBE = 0.65     # axis-A vs axis-B delta probe accuracy on val
G0_MIN_RESID = 0.03     # residual sd of each response coordinate, after removing the
#                         per-gold-pattern mean.  This is the "is there anything here that
#                         the binary gold labels do not already carry?" clause: if the
#                         teacher's response tensor is a deterministic function of the gold
#                         label pattern, B-SRTD is reducible to the CAD control by
#                         construction and there is nothing to distil.


def _resid_sd(T, E):
    """sd of each of (dA, dB, dAB) after removing the mean within each gold-label pattern."""
    d = np.stack(finite_differences(T), axis=1)              # (M, 3)
    pat = [tuple(int(v) for v in row) for row in np.asarray(E)]
    out = np.zeros_like(d)
    for p in set(pat):
        m = np.array([q == p for q in pat])
        out[m] = d[m] - d[m].mean(axis=0, keepdims=True)
    return [float(out[:, j].std()) for j in range(3)]


def g0_report(bundle):
    """bundle: dict produced by bsrtd_pilot.load_bundle.  Returns (dict, bool)."""
    rep, ok = {}, True
    pooled = {"train": 0, "val": 0}
    for lang in bundle:
        b = bundle[lang]
        r = {}
        for split, need in (("train", G0_MIN_TRAIN_LANG), ("val", G0_MIN_VAL_LANG)):
            rows = b[split]["rows"]
            n = len(rows)
            pooled[split] += n
            vp = float(np.mean([x["verify_pass"] for x in rows])) if n else 0.0
            y = np.array([x["seed_label"] for x in rows]) if n else np.zeros(0)
            bal = float(min((y == 0).mean(), (y == 1).mean())) if n else 0.0
            r[split] = {"n": n, "need_this_lang": need, "verify_pass_frac": vp,
                        "min_class_frac": bal,
                        "ok": bool(n >= need and vp >= G0_MIN_VERIFY and bal >= G0_MIN_CLASS)}
            ok &= r[split]["ok"]
        # (b) teacher actually moves, and moves in the intended direction on axis B
        T = np.concatenate([b[s]["T"] for s in ("train", "val")], axis=0)
        E = np.concatenate([b[s]["expected"] for s in ("train", "val")], axis=0)
        dA, dB, dAB = finite_differences(T)
        eA, eB, _ = finite_differences(E.astype(float))
        use = np.abs(eB) > 1e-9
        sign = float(np.mean(np.sign(dB[use]) == np.sign(eB[use]))) if use.any() else 0.0
        move = float(np.mean(np.abs(dB)))
        resid = _resid_sd(T, E)
        r["teacher"] = {"signB_agree": sign, "mean_abs_dB": move,
                        "mean_abs_dA": float(np.mean(np.abs(dA))),
                        "mean_abs_dAB": float(np.mean(np.abs(dAB))),
                        "sd_dA": float(np.std(dA)), "sd_dB": float(np.std(dB)),
                        "sd_dAB": float(np.std(dAB)),
                        "resid_sd_beyond_gold": resid, "resid_floor": G0_MIN_RESID,
                        "ok": bool(sign >= G0_MIN_SIGN and move >= G0_MIN_MOVE
                                   and min(resid) >= G0_MIN_RESID)}
        ok &= r["teacher"]["ok"]
        # (c) encoder responsivity: can a linear probe tell axis-A deltas from axis-B deltas?
        r["probe"] = {"acc": b["probe_acc"], "need": G0_MIN_PROBE,
                      "ok": bool(b["probe_acc"] >= G0_MIN_PROBE)}
        ok &= r["probe"]["ok"]
        rep[lang] = r
    rep["_pooled"] = {
        "train": {"n": pooled["train"], "need": G0_MIN_TRAIN,
                  "ok": bool(pooled["train"] >= G0_MIN_TRAIN)},
        "val": {"n": pooled["val"], "need": G0_MIN_VAL,
                "ok": bool(pooled["val"] >= G0_MIN_VAL)}}
    ok &= rep["_pooled"]["train"]["ok"] and rep["_pooled"]["val"]["ok"]
    return rep, bool(ok)


# --------------------------------------------------------------------------- G3 gate
# Score-resolution gate, added at freeze time on the user's 2026-08-10 challenge that
# verbalized LLM confidences cluster on a few round numbers (0 / 50 / 90 / 95).  If the
# teacher's scores really occupy only a handful of levels then the "continuous magnitude
# beyond the binary gold label" premise is false, the finite differences are eaten by
# quantisation, and B-SRTD cannot beat CAD by construction.  G3 detects that BEFORE training.
G3_ROUND = 4                 # decimals at which two scores count as the same level
G3_MAX_TOP6_MASS = 0.90      # the 6 most frequent levels may not carry >= 90% of the mass
G3_MIN_NEFF_SCORES = 12.0    # effective number of score levels, exp(Shannon entropy)
G3_MIN_NEFF_DIFFS = 15.0     # effective number of levels among (dA, dB, dAB) pooled
G3_MAX_ZERO_DIFF_FRAC = 0.50 # fraction of finite differences that are exactly zero


def _effective_levels(vals, nd=G3_ROUND):
    """(N_eff, n_distinct, sorted descending mass) for a set of scalar scores.

    N_eff = exp(H) with H the Shannon entropy of the empirical level distribution — the
    Hill number of order 1, i.e. "how many levels is this distribution worth".  It is the
    right statistic here because it is insensitive to a long tail of one-off values and
    reacts directly to mass piling up on a few round numbers.
    """
    v = np.round(np.asarray(vals, dtype=float).reshape(-1), nd)
    _, cnt = np.unique(v, return_counts=True)
    p = cnt / cnt.sum()
    H = float(-(p * np.log(p)).sum())
    return float(np.exp(H)), int(len(cnt)), np.sort(p)[::-1]


def g3_resolution(bundle_lang):
    """Score-resolution gate for ONE language.  Returns (report, passed).

    Thresholds, and why they are what they are:

    * `top6_mass < 0.90` — the user's own framing: if the six most frequent levels carry
      90% of the mass the teacher is effectively a 6-way ordinal scale.  Kept as stated
      rather than tightened, because G0.d already tests the sharper question (is there
      variance beyond the gold pattern) and a stricter screen here risks a false HALT —
      the deviation-D1 failure mode.
    * `N_eff(scores) >= 12` — derived from the design, not picked: the 2x2 lattice has
      4 cells and the gold pattern takes 2 values (one per seed class), so there are at
      most 8 distinct "roles" a cell can occupy.  A teacher whose effective level count is
      below 8 cannot express ANY within-role variation; 12 is that bound with a 1.5x margin.
    * `N_eff(differences) >= 15` and `zero_fraction <= 0.50` — the loss consumes
      differences, not levels, and differencing destroys resolution.  Exactly-zero
      differences are the signature of the verbalized failure mode (both cells answered
      "90"); if half the tensor is identically zero there is nothing to distil.
    """
    T = np.concatenate([bundle_lang[s]["T"] for s in ("train", "val")], axis=0)
    neff_s, ndist_s, mass = _effective_levels(T)
    top6 = float(mass[:6].sum())
    d = np.concatenate([np.asarray(x).reshape(-1) for x in finite_differences(T)])
    neff_d, ndist_d, _ = _effective_levels(d)
    zero = float((np.round(d, G3_ROUND) == 0.0).mean())
    rep = {
        "n_scores": int(T.size), "n_distinct_levels": ndist_s,
        "N_eff_scores": neff_s, "N_eff_scores_floor": G3_MIN_NEFF_SCORES,
        "top6_mass": top6, "top6_mass_ceiling": G3_MAX_TOP6_MASS,
        "top_levels": [float(x) for x in mass[:6]],
        "n_distinct_diff_levels": ndist_d, "N_eff_diffs": neff_d,
        "N_eff_diffs_floor": G3_MIN_NEFF_DIFFS,
        "zero_diff_frac": zero, "zero_diff_ceiling": G3_MAX_ZERO_DIFF_FRAC,
    }
    rep["ok"] = bool(top6 < G3_MAX_TOP6_MASS
                     and neff_s >= G3_MIN_NEFF_SCORES
                     and neff_d >= G3_MIN_NEFF_DIFFS
                     and zero <= G3_MAX_ZERO_DIFF_FRAC)
    return rep, rep["ok"]


# --------------------------------------------------------------------------- G1 / G2
# Teacher-quality protection chain (added at freeze time on the user's 2026-08-10 query
# about teacher scoring error).  G1 = qualification exam, G2 = disagreement-cell filter.
G1_PASS = 0.90          # per-cell-type accuracy the teacher must reach, in EVERY cell type
G2_MIN_TRAIN = 150      # post-filter train lattices, POOLED over languages
G2_MIN_VAL = 60         # post-filter val lattices, POOLED over languages
G2_MIN_TRAIN_LANG = 60  # post-filter train lattices in EACH language
G2_MIN_VAL_LANG = 25    # post-filter val lattices in EACH language
TEACHER_BINARISE = 0.5  # frozen binarisation threshold for the exam and the filter


def g1_exam(bundle_lang):
    """Teacher qualification exam for ONE language.

    bundle_lang: dict with ["train"|"val"]["T"] (M,4) teacher scores in [0,1] and
    ["expected"] (M,4) intended hard labels in {0,1} (-1 = unspecified -> skipped).

    Returns (report, passed).  Stratified by cell type so that a systematic failure mode
    (e.g. every stance-reversed cell answered wrongly) cannot be hidden by the pooled mean.
    """
    T = np.concatenate([bundle_lang[s]["T"] for s in ("train", "val")], axis=0)
    E = np.concatenate([bundle_lang[s]["expected"] for s in ("train", "val")], axis=0)
    P = (T >= TEACHER_BINARISE).astype(int)
    rep = {"threshold": TEACHER_BINARISE, "pass_line": G1_PASS, "by_cell": {}}
    accs = []
    for ci, c in enumerate(CELLS):
        m = E[:, ci] >= 0
        n = int(m.sum())
        acc = float((P[m, ci] == E[m, ci]).mean()) if n else 0.0
        fp = float(((P[m, ci] == 1) & (E[m, ci] == 0)).mean()) if n else 0.0
        fn = float(((P[m, ci] == 0) & (E[m, ci] == 1)).mean()) if n else 0.0
        rep["by_cell"][c] = {"n": n, "acc": acc, "false_pos": fp, "false_neg": fn,
                             "ok": bool(n > 0 and acc >= G1_PASS)}
        accs.append(acc)
    m = E >= 0
    rep["overall_acc"] = float((P[m] == E[m]).mean()) if m.any() else 0.0
    rep["worst_cell_acc"] = float(min(accs)) if accs else 0.0
    rep["spread"] = float(max(accs) - min(accs)) if accs else 1.0
    rep["ok"] = bool(all(v["ok"] for v in rep["by_cell"].values())
                     and rep["overall_acc"] >= G1_PASS)
    return rep, rep["ok"]


def g2_filter(bundle_lang):
    """Disagreement-cell filter for ONE language (frozen, whole-lattice drop).

    A lattice is KEPT iff the teacher's binarised verdict equals `cell_expected_label` in
    ALL FOUR cells.  Any single disagreement drops the whole lattice from the distillation
    loss (drop, not down-weight).  Cells with expected label -1 (unspecified) are treated
    as agreeing, since there is no intent to contradict.

    Mutates nothing; returns (report, {split: boolean keep-mask}).
    """
    rep, masks = {}, {}
    for split in ("train", "val"):
        T = bundle_lang[split]["T"]
        E = bundle_lang[split]["expected"]
        P = (T >= TEACHER_BINARISE).astype(int)
        agree = (P == E) | (E < 0)
        keep = agree.all(axis=1)
        need = G2_MIN_TRAIN_LANG if split == "train" else G2_MIN_VAL_LANG
        per_cell = {c: int((~agree[:, i]).sum()) for i, c in enumerate(CELLS)}
        rep[split] = {"n_before": int(len(keep)), "n_after": int(keep.sum()),
                      "n_dropped": int((~keep).sum()), "dropped_by_cell": per_cell,
                      "floor_this_lang": need, "ok": bool(int(keep.sum()) >= need)}
        masks[split] = keep
    rep["ok"] = bool(all(rep[s]["ok"] for s in ("train", "val")))
    return rep, masks


def axis_probe(train_txt, val_txt):
    """Logistic probe: is this delta an axis-A (label 0) or an axis-B (label 1) delta?

    train_txt / val_txt: (M, 4, D) float arrays of cell text features.
    Trained on train lattices, scored on val lattices.  Pure train/val, no test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def xy(a):
        dA = a[:, 1] - a[:, 0]
        dB = a[:, 2] - a[:, 0]
        return np.concatenate([dA, dB], 0), np.concatenate(
            [np.zeros(len(dA)), np.ones(len(dB))])

    Xtr, ytr = xy(np.asarray(train_txt, dtype=np.float64))
    Xva, yva = xy(np.asarray(val_txt, dtype=np.float64))
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    return float(clf.score(sc.transform(Xva), yva))


# --------------------------------------------------------------------------- synthetic
SYNTH_DIR = os.path.join(ROOT, "idea-stage", "bsrtd_synth")


def make_synth(out_dir=SYNTH_DIR, d_img=64, d_txt=64, n_train=260, n_val=90,
               n_extra_train=240, n_extra_val=40, seed=20260810, signal="planted",
               txt_noise=0.85, img_noise=0.85, cell_noise=0.30):
    """Fabricate a complete synthetic bundle: base caches, lattices, cell caches, teacher.

    Generative story (so the rehearsal is a real end-to-end test, not a shape test):
      * a hidden direction `w_hate` in text space carries hate content;
      * axis A (target substitution) moves the text along a *different* direction `w_tgt`
        and leaves hate content roughly intact;
      * axis B (stance reversal) subtracts most of the hate component -> label flips;
      * the mixed term is non-zero: reversing the stance of a substituted target removes
        slightly *more* hate than reversing the original (an interaction the additive
        student cannot express from the labels alone);
      * the teacher score is sigma(a * hate_component + b), i.e. a smooth monotone read of
        the same hidden variable, so the response tensor is genuinely learnable.
    `signal="nosignal"` replaces the teacher score with independent uniform noise.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    lat_dir = os.path.join(out_dir, "lattices")
    emb_dir = os.path.join(out_dir, "CLIP_Embedding")
    os.makedirs(lat_dir, exist_ok=True)

    w_hate = rng.normal(size=d_txt); w_hate /= np.linalg.norm(w_hate)
    w_tgt = rng.normal(size=d_txt)
    w_tgt -= w_tgt @ w_hate * w_hate
    w_tgt /= np.linalg.norm(w_tgt)
    w_img = rng.normal(size=d_img); w_img /= np.linalg.norm(w_img)

    teacher_rows = []
    manifest = {}
    for lang in ("en", "zh"):
        ds = LANG2DS[lang]
        os.makedirs(os.path.join(emb_dir, ds), exist_ok=True)
        for split, n_lat, n_extra in (("train", n_train, n_extra_train),
                                      ("val", n_val, n_extra_val)):
            n = n_lat + n_extra
            y = (rng.random(n) < 0.5).astype(np.int64)
            # hidden hate level; positives higher
            h = rng.normal(loc=y * 2.4 - 1.2, scale=0.6, size=n)
            g = rng.normal(size=(n, d_txt)) * txt_noise
            txt = g + h[:, None] * w_hate[None, :]
            img = (rng.normal(size=(n, d_img)) * img_noise
                   + (h * 0.6)[:, None] * w_img[None, :])
            ids = [f"{lang}{split}{i:05d}" for i in range(n)]
            torch.save({"ids": [ids],
                        "img_feats": torch.tensor(img, dtype=torch.float32),
                        "text_feats": torch.tensor(txt, dtype=torch.float32),
                        "labels": torch.tensor(y)},
                       os.path.join(emb_dir, ds,
                                    f"{SPLIT2CACHE[split]}_{STUDENT_TAG}.pt"))

            # lattice over the first n_lat items
            cell_txt = np.zeros((n_lat, 4, d_txt), dtype=np.float32)
            cell_h = np.zeros((n_lat, 4), dtype=np.float64)
            lat_rows = []
            for i in range(n_lat):
                base_h = h[i]
                # axis A: substituting the target moves the text along `w_tgt` and changes
                # the hate LEVEL only mildly -- but by a per-item amount, which is exactly
                # the graded information a binary gold label cannot carry.
                a_shift = 1.1 + 0.3 * rng.normal()
                a_eff = -0.20 + 0.55 * rng.normal()
                # axis B: stance reversal removes the hate content outright (label flips)
                b_drop = 2.6 + 0.5 * rng.normal()
                inter = 0.40 + 0.30 * rng.normal()      # mixed term (non-additive)
                hh = np.array([base_h,
                               base_h + a_eff,
                               base_h - b_drop,
                               base_h + a_eff - b_drop - inter])
                for c in range(4):
                    v = g[i] + hh[c] * w_hate
                    if c in (1, 3):
                        v = v + a_shift * w_tgt
                    cell_txt[i, c] = v + rng.normal(size=d_txt) * cell_noise
                cell_h[i] = hh
                exp = {"orig": int(y[i]), "targetsub": int(y[i]),
                       "stancerev": 0, "both": 0}
                lat_rows.append({
                    "seed_id": ids[i], "split": split, "lang": lang,
                    "seed_label": int(y[i]),
                    "cells": {c: f"[synthetic {lang}/{split}/{ids[i]}/{c}]" for c in CELLS},
                    "cell_expected_labels": exp,
                    "verify": {"pass": True, "source": "synthetic"},
                })
            with open(os.path.join(lat_dir, f"{split}_lattices_{lang}.jsonl"), "w") as f:
                for r in lat_rows:
                    f.write(json.dumps(r) + "\n")
            torch.save({"ids": [[r["seed_id"] for r in lat_rows]],
                        "cells": list(CELLS),
                        "text_feats": torch.tensor(cell_txt, dtype=torch.float32),
                        "labels": torch.tensor([r["seed_label"] for r in lat_rows]),
                        "cell_expected_labels": torch.tensor(
                            [[r["cell_expected_labels"][c] for c in CELLS] for r in lat_rows]),
                        "student_tag": STUDENT_TAG, "engine": "synthetic"},
                       os.path.join(emb_dir, ds, f"{SPLIT2CACHE[split]}_{CELL_TAG}.pt"))

            if signal == "nosignal":
                T = rng.random((n_lat, 4))
            else:
                # smooth monotone read of the same hidden variable + a little rater noise,
                # so a few lattices genuinely disagree with the gold pattern and G2 has
                # something to filter.
                T = 1.0 / (1.0 + np.exp(-(1.8 * cell_h + 0.15
                                          + rng.normal(scale=0.35, size=cell_h.shape))))
            for i, r in enumerate(lat_rows):
                for ci, c in enumerate(CELLS):
                    teacher_rows.append({
                        "key": teacher_key(lang, split, r["seed_id"], c),
                        "lang": lang, "split": split, "seed_id": r["seed_id"], "cell": c,
                        "text_sha256": text_sha(r["cells"][c]),
                        "model": "synthetic-oracle", "engine": "synthetic",
                        "prompt_id": PROMPT_ID, "score": float(T[i, ci]),
                        "raw": str(int(round(T[i, ci] * 100)))})
            manifest[f"{lang}/{split}"] = {"n_items": n, "n_lattices": n_lat}

        # synthetic held-out pack (no lattices) so the rehearsal exercises the real
        # report path.  This is fabricated data, not any project test split.
        n = 200
        y = (rng.random(n) < 0.5).astype(np.int64)
        h = rng.normal(loc=y * 1.6 - 0.8, scale=0.7, size=n)
        txt = rng.normal(size=(n, d_txt)) * txt_noise + h[:, None] * w_hate[None, :]
        img = (rng.normal(size=(n, d_img)) * img_noise
               + (h * 0.6)[:, None] * w_img[None, :])
        torch.save({"ids": [[f"{lang}test{i:05d}" for i in range(n)]],
                    "img_feats": torch.tensor(img, dtype=torch.float32),
                    "text_feats": torch.tensor(txt, dtype=torch.float32),
                    "labels": torch.tensor(y)},
                   os.path.join(emb_dir, ds, f"test_seen_{STUDENT_TAG}.pt"))

    with open(os.path.join(out_dir, "teacher_scores.jsonl"), "w") as f:
        for r in teacher_rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump({"signal": signal, "seed": seed, "d_img": d_img, "d_txt": d_txt,
                   "splits": manifest}, f, indent=2)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="B-SRTD synthetic bundle generator (rehearsal only)")
    ap.add_argument("--out", default=SYNTH_DIR)
    ap.add_argument("--signal", choices=["planted", "nosignal"], default="planted")
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--txt-noise", type=float, default=0.85)
    ap.add_argument("--img-noise", type=float, default=0.85)
    a = ap.parse_args()
    p = make_synth(out_dir=a.out, signal=a.signal, seed=a.seed,
                   txt_noise=a.txt_noise, img_noise=a.img_noise)
    print(f"synthetic bundle ({a.signal}) written to {p}")
