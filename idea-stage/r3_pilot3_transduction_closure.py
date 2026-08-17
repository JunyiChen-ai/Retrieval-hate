#!/usr/bin/env python
"""R3-3 — C10 "Cross-channel evasion transduction closure": compositional path dependence.

Decision rules are FROZEN in idea-stage/R3_PILOT_FREEZE_2026-08-09.md (section "Pilot R3-3")
and are NOT edited after results are seen.  Nothing in this file may change a threshold,
a statistic or a rule; every ambiguity resolved here is recorded verbatim in the output JSON
under "interpretations".

Zero test-set contact: only data/gt/HateMM/{train,val}.jsonl,
data/ASR/HateMM/{train,dev_seen}_asrK30_whisper-large-v3.jsonl and
data/OCR/HateMM/ocr_windows_K30.jsonl.  An explicit path guard HALTs on any path whose
name contains "test".  `dev_seen` is the validation split and is allowed.

Usage:
  python idea-stage/r3_pilot3_transduction_closure.py --smoke synthetic
  python idea-stage/r3_pilot3_transduction_closure.py --smoke permuted
  python idea-stage/r3_pilot3_transduction_closure.py --out idea-stage/r3_pilot3.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from itertools import permutations
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

ROOT = Path("/home/jehc223/Retrieval-hate")

# ---------------- frozen constants (R3_PILOT_FREEZE_2026-08-09.md) ----------------
SEEDS = [0, 1, 2, 3, 4]          # 5 fixed seeds
N_FOLDS = 5                      # 5-fold, video-level
N_PERM = 20                      # 20 permutations per seed => 100 null replicates
NULL_PCT = 95.0                  # one-sided 95th percentile
GO_P = 0.35                      # condition 1: P_obs >= 0.35 clean-logit SD
GO_A = 0.15                      # condition 2: A_obs >= 0.15 clean-logit SD
NULL_MULT = 3.0                  # conditions 3/4: >= 3 x N95
GO_RETENTION = 0.90              # condition 5: >= 90% of the clean signed margin

# shared-protocol OCR filter (project standard)
MIN_CONF = 0.5
MIN_TEXT_LEN = 2

# implementation knobs (NOT frozen quantities; recorded in the JSON)
MAXF = 15000                     # TF-IDF max_features PER CHANNEL BLOCK
MIN_DF = 3
NGRAM = (3, 5)
C_REG = 1.0
NULL_SEED_BASE = 20260903

FILES = {
    "asr_train": ROOT / "data/ASR/HateMM/train_asrK30_whisper-large-v3.jsonl",
    "asr_val": ROOT / "data/ASR/HateMM/dev_seen_asrK30_whisper-large-v3.jsonl",
    "ocr_windows": ROOT / "data/OCR/HateMM/ocr_windows_K30.jsonl",
    "gt_train": ROOT / "data/gt/HateMM/train.jsonl",
    "gt_val": ROOT / "data/gt/HateMM/val.jsonl",
}


class Halt(RuntimeError):
    pass


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------ guards --
_GUARD_ARMED = False
_TOUCHED = []


def arm_guard():
    global _GUARD_ARMED
    _GUARD_ARMED = True
    log("GUARD ARMED: any path whose name contains 'test' HALTs; "
        "allowed split tokens = {train, val, dev_seen}")


def guard_path(p):
    if not _GUARD_ARMED:
        raise Halt("HALT_GUARD_NOT_ARMED")
    p = Path(p)
    low = str(p).lower()
    for part in p.parts:
        if "test" in part.lower():
            raise Halt("HALT_TEST_CONTACT:path=%s" % p)
    if "test_seen" in low or "test" in low:
        raise Halt("HALT_TEST_CONTACT:path=%s" % p)
    _TOUCHED.append(str(p))
    return p


def guard_open(p, **kw):
    return open(guard_path(p), **kw)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ------------------------------------------------------- frozen transforms --
LEET_A = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
SYM_O = {"a": "@", "e": "€", "i": "!", "o": "()", "s": "$", "t": "+"}


def normalise(s):
    """Unicode NFKC + lowercase + whitespace collapse (project OCR/text convention)."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    return " ".join(s.split())


def _map_chars(s, table):
    if not s:
        return s
    return "".join(table.get(c, c) for c in s)


def op_L_A(state):
    a, o = state
    return (_map_chars(a, LEET_A), o)


def op_L_O(state):
    a, o = state
    return (a, _map_chars(o, SYM_O))


def _period_insert(text):
    """Insert periods between the characters of every second token of length >= 4."""
    if not text:
        return text
    toks = text.split()
    elig = [j for j, t in enumerate(toks) if len(t) >= 4]
    for j in elig[1::2]:                       # "every second" = 2nd, 4th, ...
        toks[j] = ".".join(toks[j])
    return " ".join(toks)


def op_S(state):
    a, o = state
    return (_period_insert(a), _period_insert(o))


def _eligible(tok):
    """Frozen-doc term "eligible" (see interpretations I3): length >= 3 and containing
    at least one alphanumeric character."""
    return len(tok) >= 3 and any(c.isalnum() for c in tok)


def op_M(state):
    """Move every second eligible ASR token into the OCR channel, preserving token order."""
    a, o = state
    if not a:
        return (a, o)
    toks = a.split()
    elig = [j for j, t in enumerate(toks) if _eligible(t)]
    sel = set(elig[1::2])
    if not sel:
        return (a, o)
    moved = [toks[j] for j in range(len(toks)) if j in sel]
    kept = [toks[j] for j in range(len(toks)) if j not in sel]
    new_o = (o + " " + " ".join(moved)).strip() if o else " ".join(moved)
    return (" ".join(kept), new_o)


OPS = {"L_A": op_L_A, "L_O": op_L_O, "S": op_S, "M": op_M}
OP_NAMES = ["L_A", "L_O", "S", "M"]
# abstract op class: L_A and L_O are both character-level obfuscation
ABSTRACT = {"L_A": "L", "L_O": "L", "S": "S", "M": "M"}


def enumerate_paths():
    """Identity + all 4 single edges + all ordered length-2 and length-3 compositions of
    DISTINCT transformations (12 + 24 = 36 composed paths)."""
    paths = [()]
    paths += [(t,) for t in OP_NAMES]
    paths += [tuple(p) for p in permutations(OP_NAMES, 2)]
    paths += [tuple(p) for p in permutations(OP_NAMES, 3)]
    return paths


PATHS = enumerate_paths()
PNAME = ["IDENT" if not p else "->".join(p) for p in PATHS]
PIDX = {n: i for i, n in enumerate(PNAME)}
IDENT_I = PIDX["IDENT"]
SINGLE_I = [PIDX[t] for t in OP_NAMES]
COMPOSED_I = [i for i, p in enumerate(PATHS) if len(p) >= 2]

# --- FROZEN semantic-equivalence declaration (gating) ---
# The freeze document declares exactly these two endpoint classes.
EQ_CLASSES_GATING = [
    ["L_A->M", "M->L_O"],
    ["S->M->L_O", "L_A->M->S"],
]
EQ_GATING_I = [[PIDX[n] for n in cls] for cls in EQ_CLASSES_GATING]


def eq_classes_extended():
    """Descriptive (NON-GATING) extension: group every enumerated path by the multiset of
    its abstract ops (L_A, L_O -> L).  Reproduces both declared classes and covers the
    remaining length-2 / length-3 compositions."""
    buckets = {}
    for i, p in enumerate(PATHS):
        if len(p) < 2:
            continue
        key = tuple(sorted(ABSTRACT[t] for t in p))
        buckets.setdefault(key, []).append(i)
    return {"+".join(k): v for k, v in sorted(buckets.items()) if len(v) >= 2}


EQ_EXT = eq_classes_extended()


def apply_path(state, path):
    for t in path:
        state = OPS[t](state)
    return state


# -------------------------------------------------------------------- data --
def load_real():
    asr, ocr, lab = {}, {}, {}
    for key in ("gt_train", "gt_val"):
        with guard_open(FILES[key], encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                lab[str(o["id"])] = int(o["label"])
    for key in ("asr_train", "asr_val"):
        with guard_open(FILES[key], encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                wt = o.get("window_text") or []
                asr[str(o["id"])] = normalise(" ".join(w for w in wt if w))
    raw_ocr = {}
    n_det, n_kept = 0, 0
    with guard_open(FILES["ocr_windows"], encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            keep = []
            for d in r.get("texts") or []:
                n_det += 1
                t = (d.get("text") or "").strip()
                if float(d.get("conf", 0.0)) >= MIN_CONF and len(t) >= MIN_TEXT_LEN:
                    keep.append(t)
                    n_kept += 1
            raw_ocr.setdefault(str(r["video_id"]), {})[int(r["window_k"])] = " ".join(keep)
    for v, wk in raw_ocr.items():
        ocr[v] = normalise(" ".join(wk[k] for k in sorted(wk)))

    ids = sorted(lab)
    if set(asr) != set(lab):
        raise Halt("HALT_ASR_ID_MISMATCH:%d vs %d" % (len(asr), len(lab)))
    if set(ocr) != set(lab):
        raise Halt("HALT_OCR_ID_MISMATCH:%d vs %d" % (len(ocr), len(lab)))
    y = np.array([lab[v] for v in ids], dtype=np.int64)
    A = [asr[v] for v in ids]
    O = [ocr[v] for v in ids]
    meta = {"n_ocr_detections_total": n_det, "n_ocr_detections_kept": n_kept,
            "n_videos_with_empty_ocr": int(sum(1 for s in O if not s)),
            "n_videos_with_empty_asr": int(sum(1 for s in A if not s))}
    return ids, A, O, y, meta


def make_synthetic(n=140, seed=11):
    rng = np.random.default_rng(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
             "iota", "kappa", "lambda", "sigma", "omega", "tau", "rho", "chi"]
    hot = ["toxicword", "slurtoken", "hatestem"]
    ids, A, O, y = [], [], [], []
    for i in range(n):
        lab = int(rng.integers(0, 2))
        na = int(rng.integers(40, 160))
        toks = list(rng.choice(vocab, size=na))
        if lab:
            for _ in range(int(rng.integers(2, 6))):
                toks.insert(int(rng.integers(0, len(toks))), str(rng.choice(hot)))
        no = int(rng.integers(0, 30))
        otoks = list(rng.choice(vocab, size=no))
        if lab and no and rng.random() < 0.5:
            otoks.append(str(rng.choice(hot)))
        ids.append("syn_%04d" % i)
        A.append(normalise(" ".join(toks)))
        O.append(normalise(" ".join(otoks)))
        y.append(lab)
    return ids, A, O, np.array(y, dtype=np.int64), {"synthetic": True}


# ------------------------------------------------------------ fold machinery --
def _vec():
    return TfidfVectorizer(analyzer="char", ngram_range=NGRAM, min_df=MIN_DF,
                           max_features=MAXF, sublinear_tf=True, lowercase=False,
                           dtype=np.float32)


def _dedup_transform(vec, docs):
    """Transform a list of docs, vectorising each unique string once."""
    uniq, idx = {}, np.empty(len(docs), dtype=np.int64)
    order = []
    for k, d in enumerate(docs):
        j = uniq.get(d)
        if j is None:
            j = len(order)
            uniq[d] = j
            order.append(d)
        idx[k] = j
    X = vec.transform(order)
    return X[idx]


def _fit_logits(X_tr, y_tr, X_ev):
    clf = LogisticRegression(C=C_REG, solver="liblinear", dual=True, max_iter=2000,
                             tol=1e-4, fit_intercept=True)
    clf.fit(X_tr, y_tr)
    return clf.decision_function(X_ev)


def run_fold(A, O, y, tr, te, label_settings, maxf_note=None):
    """One (seed, fold).  Returns:
      f_paths : [n_label_settings, n_test, n_paths] fold-standardised logits
      f_orig  : [n_test] fold-standardised logits of the NON-augmented classifier
                (observed labels only, label_settings[0])
    """
    n_paths = len(PATHS)
    # ---- augmented training docs: originals + every single-edge transformation ----
    variants = [()] + [(t,) for t in OP_NAMES]
    tr_a, tr_o = [], []
    for v in variants:
        for i in tr:
            a, o = apply_path((A[i], O[i]), v)
            tr_a.append(a)
            tr_o.append(o)
    va, vo = _vec(), _vec()
    Xa = va.fit_transform(tr_a)
    Xo = vo.fit_transform(tr_o)
    X_tr = sp.hstack([Xa, Xo], format="csr")
    n_tr = len(tr)

    # ---- evaluation docs: every enumerated path on the held-out videos ----
    ev_a, ev_o = [], []
    for i in te:
        for p in PATHS:
            a, o = apply_path((A[i], O[i]), p)
            ev_a.append(a)
            ev_o.append(o)
    X_ev = sp.hstack([_dedup_transform(va, ev_a), _dedup_transform(vo, ev_o)], format="csr")

    # ---- non-augmented reference classifier (own fold-local vocabulary) ----
    oa, oo = _vec(), _vec()
    Xoa = oa.fit_transform([A[i] for i in tr])
    Xoo = oo.fit_transform([O[i] for i in tr])
    X_tr0 = sp.hstack([Xoa, Xoo], format="csr")
    X_ev0 = sp.hstack([oa.transform([A[i] for i in te]),
                       oo.transform([O[i] for i in te])], format="csr")

    out = np.empty((len(label_settings), len(te), n_paths), dtype=np.float64)
    for li, yv in enumerate(label_settings):
        y_tr = np.tile(yv[tr], len(variants))
        if len(np.unique(y_tr)) < 2:
            raise Halt("HALT_DEGENERATE_FOLD_LABELS")
        raw = _fit_logits(X_tr, y_tr, X_ev).reshape(len(te), n_paths)
        clean = raw[:, IDENT_I]
        mu, sd = float(clean.mean()), float(clean.std())
        if sd < 1e-9:
            raise Halt("HALT_ZERO_CLEAN_SD")
        out[li] = (raw - mu) / sd

    y0 = label_settings[0]
    raw0 = _fit_logits(X_tr0, y0[tr], X_ev0)
    mu0, sd0 = float(raw0.mean()), float(raw0.std())
    if sd0 < 1e-9:
        raise Halt("HALT_ZERO_CLEAN_SD_ORIG")
    f_orig = (raw0 - mu0) / sd0
    return out, f_orig


# ------------------------------------------------------------- statistics ----
def stat_P(F, classes_i):
    """P = median_i max over declared endpoint classes of the within-class max |Δf|."""
    n = F.shape[0]
    best = np.zeros(n)
    for cls in classes_i:
        for a in range(len(cls)):
            for b in range(a + 1, len(cls)):
                best = np.maximum(best, np.abs(F[:, cls[a]] - F[:, cls[b]]))
    return float(np.median(best)), best


def stat_A(F):
    """A = median_i [ max_{|p|>=2}|f(p)-f(x)| - max_{|p|=1}|f(p)-f(x)| ]."""
    base = F[:, IDENT_I][:, None]
    d_comp = np.abs(F[:, COMPOSED_I] - base).max(axis=1)
    d_sing = np.abs(F[:, SINGLE_I] - base).max(axis=1)
    return float(np.median(d_comp - d_sing)), (d_comp - d_sing)


# ------------------------------------------------------------------- driver --
def run_all(ids, A, O, y, n_perm, seeds, n_jobs, tag):
    n = len(ids)
    tasks = []
    for s in seeds:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        rng = np.random.default_rng(NULL_SEED_BASE + 1000 * s)
        perms = [y[rng.permutation(n)] for _ in range(n_perm)]
        label_settings = [y] + perms
        for fi, (tr, te) in enumerate(kf.split(np.arange(n))):
            tasks.append((s, fi, tr, te, label_settings))

    t0 = time.time()
    done = [0]

    def _job(k, s, fi, tr, te, ls):
        r = run_fold(A, O, y, tr, te, ls)
        return k, s, fi, tr, te, r

    log("%s: dispatching %d (seed,fold) tasks, %d label settings each, n_jobs=%d"
        % (tag, len(tasks), 1 + n_perm, n_jobs))
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0, batch_size=1,
                       return_as="generator")(
        delayed(_job)(k, *t) for k, t in enumerate(tasks))

    n_ls = 1 + n_perm
    F = {s: np.full((n_ls, n, len(PATHS)), np.nan) for s in seeds}
    Forig = {s: np.full(n, np.nan) for s in seeds}
    for k, s, fi, tr, te, (fp, fo) in results:
        F[s][:, te, :] = fp
        Forig[s][te] = fo
        done[0] += 1
        log("PROGRESS %s task=%d/%d seed=%d fold=%d elapsed=%.1fs"
            % (tag, done[0], len(tasks), s, fi, time.time() - t0))

    per_seed = []
    null_P, null_A = [], []
    for s in seeds:
        if np.isnan(F[s]).any() or np.isnan(Forig[s]).any():
            raise Halt("HALT_INCOMPLETE_OOF:seed=%d" % s)
        P_o, _ = stat_P(F[s][0], EQ_GATING_I)
        A_o, _ = stat_A(F[s][0])
        sgn = 2.0 * y - 1.0
        m_aug = float(np.mean(sgn * F[s][0][:, IDENT_I]))
        m_org = float(np.mean(sgn * Forig[s]))
        ext = {k: stat_P(F[s][0], [v])[0] for k, v in EQ_EXT.items()}
        P_ext, _ = stat_P(F[s][0], list(EQ_EXT.values()))
        rec = {"seed": s, "P": P_o, "A": A_o,
               "mean_signed_clean_margin_augmented": m_aug,
               "mean_signed_clean_margin_original": m_org,
               "clean_margin_retention_ratio": (m_aug / m_org) if m_org != 0 else float("nan"),
               "P_extended_nongating": P_ext,
               "P_extended_per_class_nongating": ext,
               "null_P": [], "null_A": []}
        for li in range(1, n_ls):
            p, _ = stat_P(F[s][li], EQ_GATING_I)
            a, _ = stat_A(F[s][li])
            rec["null_P"].append(p)
            rec["null_A"].append(a)
            null_P.append(p)
            null_A.append(a)
        per_seed.append(rec)
        log("SEED %d  P=%.4f  A=%.4f  retention=%.4f  nullP_med=%.4f nullA_med=%.4f"
            % (s, P_o, A_o, rec["clean_margin_retention_ratio"],
               float(np.median(rec["null_P"])), float(np.median(rec["null_A"]))))
    return per_seed, np.array(null_P), np.array(null_A), F, Forig


def adjudicate(per_seed, null_P, null_A):
    P_obs = float(np.mean([r["P"] for r in per_seed]))
    A_obs = float(np.mean([r["A"] for r in per_seed]))
    ret = float(np.mean([r["clean_margin_retention_ratio"] for r in per_seed]))
    n95P = float(np.percentile(null_P, NULL_PCT))
    n95A = float(np.percentile(null_A, NULL_PCT))
    cond = {
        "1_P_obs_ge_0.35": bool(P_obs >= GO_P),
        "2_A_obs_ge_0.15": bool(A_obs >= GO_A),
        "3_P_obs_ge_3xN95P": bool(P_obs >= NULL_MULT * n95P),
        "4_A_obs_ge_3xN95A": bool(A_obs >= NULL_MULT * n95A),
        "5_clean_margin_retention_ge_0.90": bool(ret >= GO_RETENTION),
    }
    return {
        "P_obs": P_obs, "A_obs": A_obs,
        "N95_P": n95P, "N95_A": n95A,
        "three_x_N95_P": NULL_MULT * n95P, "three_x_N95_A": NULL_MULT * n95A,
        "clean_margin_retention_ratio": ret,
        "P_obs_min_over_seeds": float(np.min([r["P"] for r in per_seed])),
        "A_obs_min_over_seeds": float(np.min([r["A"] for r in per_seed])),
        "retention_min_over_seeds": float(np.min(
            [r["clean_margin_retention_ratio"] for r in per_seed])),
        "n_null_replicates": int(len(null_P)),
        "conditions": {k: ("PASS" if v else "FAIL") for k, v in cond.items()},
        "verdict": "GO" if all(cond.values()) else "KILL",
    }


INTERPRETATIONS = {
    "I1_S_channel_scope": (
        "The freeze writes L_A and L_O with an explicit channel subscript but writes S "
        "without one. S is therefore applied to BOTH channels independently (each channel's "
        "own whitespace token stream)."),
    "I2_every_second_token": (
        "'every second token of length >= 4' (S) and 'every second eligible ASR token' (M) "
        "are read as: filter to the eligible subsequence, then take elements at 0-based "
        "indices 1,3,5,... (i.e. the 2nd, 4th, ... eligible token). The 1st eligible token "
        "is left untouched."),
    "I3_eligible_definition": (
        "'eligible' (M) = whitespace token with length >= 3 that contains at least one "
        "alphanumeric character. The narrower literal reading 'alphabetic tokens of length "
        ">= 3' (str.isalpha) was rejected because after L_A ('summer'->'5umm3r') or after S "
        "('s.u.m.m.e.r') no token is alphabetic any more, so M would silently degenerate to a "
        "no-op inside exactly the composed paths this pilot exists to measure. That "
        "degeneracy would INFLATE P_obs (it would make L_A->M collapse to L_A while M->L_O "
        "still migrates tokens), so the alphanumeric reading is also the conservative one."),
    "I4_M_insertion_point": (
        "Moved ASR tokens are appended to the end of the OCR channel, in their original "
        "relative order ('preserving token order')."),
    "I5_path_set": (
        "Compositions are ordered sequences of DISTINCT transformations drawn from "
        "{L_A, L_O, S, M}: 12 length-2 and 24 length-3 paths, plus the 4 single edges and "
        "the identity = 41 evaluated states per video. Repeated transformations are excluded "
        "because e.g. L_A->L_A is a no-op and would make the abstract-op bookkeeping "
        "ill-defined. No length-4 path is evaluated (the freeze names length 2 and 3 only)."),
    "I6_equivalence_classes_gating": (
        "E_i (the paths 'declared to share a semantic endpoint') is taken LITERALLY from the "
        "freeze: exactly the two declared classes {L_A->M, M->L_O} and "
        "{S->M->L_O, L_A->M->S}. P_obs takes the maximum |Δf| WITHIN each class and then the "
        "maximum over the two classes; it never compares across classes. This is the "
        "conservative reading -- any wider declaration of equivalence can only increase "
        "P_obs. A wider, principled grouping (paths sharing the same multiset of abstract "
        "ops after L_A,L_O -> L, which reproduces both declared classes) is computed and "
        "reported as 'P_extended_nongating' for information only; it does NOT enter any "
        "frozen condition."),
    "I7_classifier": (
        "Character 3-5-gram TF-IDF logistic model with TWO SEPARATE fold-local vectorisers -- "
        "one fit on the ASR-channel strings, one on the OCR-channel strings -- whose matrices "
        "are horizontally concatenated. The channel distinction is therefore real: the same "
        "n-gram carries different weights in the two blocks, so the channel-migration "
        "transformation M changes the representation even when the union of the two strings "
        "is unchanged. sublinear_tf, min_df=%d, max_features=%d PER BLOCK, "
        "LogisticRegression(liblinear, dual, C=%.2f)." % (MIN_DF, MAXF, C_REG)),
    "I8_standardisation": (
        "'clean-logit SD' units: within each fold, the held-out ORIGINAL (identity-path) "
        "decision_function values define mu and sd, and every held-out path logit of that "
        "fold is standardised as (raw - mu)/sd. Fold-local, computed separately for each "
        "label setting (observed and each permutation). Scores are continuous unbounded "
        "decision_function values throughout; no probabilities, no votes."),
    "I9_folds": (
        "5-fold KFold(shuffle=True, random_state=seed) over videos. Each video is exactly one "
        "row and one group, so plain KFold over videos IS video-level grouping; the seed only "
        "reshuffles the fold assignment."),
    "I10_margin_retention": (
        "'mean signed clean margin' = mean over held-out originals of (2y-1) * f_clean, where "
        "each classifier is standardised by ITS OWN fold-local clean mu/sd (so the ratio is "
        "scale-free). The reference 'original classifier' is trained on originals only, with "
        "its own fold-local vocabulary. Retention = margin_augmented / margin_original."),
    "I11_seed_aggregation": (
        "P_obs / A_obs / retention are reported as the MEAN over the 5 seeds of the per-seed "
        "statistic (each per-seed statistic is itself the median over all %d out-of-fold "
        "videos). The minimum over seeds is also recorded so a reader can re-adjudicate. "
        "N95 is the 95th percentile of the 100 pooled null replicate values (20 permutations "
        "x 5 seeds), each replicate being one full 5-fold OOF re-run."),
    "I12_channel_text": (
        "ASR channel text = the K=30 'window_text' entries of the cached Whisper-large-v3 "
        "transcript joined by spaces. OCR channel text = all detections of all 30 windows "
        "with conf >= %.2f and stripped length >= %d, joined by spaces in window order "
        "(the project's standard filter). Both channels are NFKC-normalised, lowercased and "
        "whitespace-collapsed BEFORE any transformation." % (MIN_CONF, MIN_TEXT_LEN)),
    "I13_null": (
        "The null permutes the 851 video labels globally, then retrains the SAME "
        "single-edge-augmented classifier on the same folds and recomputes both statistics "
        "end to end. The label-independent TF-IDF vocabulary and the evaluation matrices are "
        "shared across the 21 label settings of a fold -- an exact algebraic identity, not an "
        "approximation, because fitting the vectoriser never sees a label."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", choices=["synthetic", "permuted"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--smoke-perm", type=int, default=3,
                    help="permutations per seed in SMOKE modes only")
    args = ap.parse_args()

    arm_guard()
    t0 = time.time()
    log("paths enumerated: %d total (1 identity, %d single, %d composed)"
        % (len(PATHS), len(SINGLE_I), len(COMPOSED_I)))
    log("gating equivalence classes: %s" % EQ_CLASSES_GATING)

    out = {
        "pilot": "R3-3 C10 cross-channel evasion transduction closure",
        "freeze": "idea-stage/R3_PILOT_FREEZE_2026-08-09.md (section 'Pilot R3-3')",
        "mode": args.smoke or "real",
        "guard": "armed: any path containing 'test' HALTs (dev_seen allowed)",
        "frozen_constants": {
            "seeds": SEEDS, "n_folds": N_FOLDS, "n_permutations_per_seed": N_PERM,
            "n_null_replicates": N_PERM * len(SEEDS), "null_percentile": NULL_PCT,
            "GO_P": GO_P, "GO_A": GO_A, "null_multiplier": NULL_MULT,
            "GO_retention": GO_RETENTION,
        },
        "implementation_knobs": {
            "tfidf_analyzer": "char", "tfidf_ngram_range": list(NGRAM),
            "tfidf_max_features_per_channel_block": MAXF, "tfidf_min_df": MIN_DF,
            "tfidf_sublinear_tf": True, "logreg_C": C_REG,
            "logreg_solver": "liblinear(dual)", "null_seed_base": NULL_SEED_BASE,
            "note": ("max_features was set to %d PER BLOCK to keep the 100-replicate null "
                     "loop inside the wall-clock budget; the number of permutations was "
                     "never reduced." % MAXF),
        },
        "path_set": {
            "n_paths_evaluated_per_video": len(PATHS),
            "identity": "IDENT",
            "single_edge": OP_NAMES,
            "composed_length2": [PNAME[i] for i in COMPOSED_I if len(PATHS[i]) == 2],
            "composed_length3": [PNAME[i] for i in COMPOSED_I if len(PATHS[i]) == 3],
            "declared_semantically_equivalent_classes_GATING": EQ_CLASSES_GATING,
            "extended_equivalence_classes_NONGATING": {
                k: [PNAME[i] for i in v] for k, v in EQ_EXT.items()},
        },
        "interpretations": INTERPRETATIONS,
    }

    if args.smoke == "synthetic":
        ids, A, O, y, m = make_synthetic()
        per_seed, nP, nA, F, Fo = run_all(ids, A, O, y, args.smoke_perm, SEEDS,
                                          args.n_jobs, "SMOKE-SYN")
        out["n_videos"] = len(ids)
        out["per_seed"] = per_seed
        out["adjudication"] = adjudicate(per_seed, nP, nA)
        out["smoke_note"] = ("synthetic data, %d permutations/seed -- pipeline exercise only, "
                             "NOT a decision" % args.smoke_perm)
        log("SMOKE synthetic: %s" % json.dumps(
            {k: out["adjudication"][k] for k in
             ("P_obs", "A_obs", "N95_P", "N95_A", "clean_margin_retention_ratio",
              "conditions", "verdict")}))
        log("elapsed %.1fs" % (time.time() - t0))
        if args.out:
            Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return

    ids, A, O, y, dmeta = load_real()
    log("loaded n=%d videos, positives=%d (%.3f), %s"
        % (len(ids), int(y.sum()), float(y.mean()), json.dumps(dmeta)))

    if args.smoke == "permuted":
        rng = np.random.default_rng(4242)     # NOT the frozen null seed base
        yp = y[rng.permutation(len(y))]
        per_seed, nP, nA, F, Fo = run_all(ids, A, O, yp, args.smoke_perm, SEEDS[:1],
                                          args.n_jobs, "SMOKE-PERM")
        out["n_videos"] = len(ids)
        out["per_seed"] = per_seed
        out["adjudication"] = adjudicate(per_seed, nP, nA)
        out["smoke_note"] = ("real texts with globally permuted labels, 1 seed, %d "
                             "permutations -- pipeline exercise only, NOT a decision"
                             % args.smoke_perm)
        log("SMOKE permuted: %s" % json.dumps(
            {k: out["adjudication"][k] for k in
             ("P_obs", "A_obs", "N95_P", "N95_A", "clean_margin_retention_ratio",
              "conditions", "verdict")}))
        log("elapsed %.1fs" % (time.time() - t0))
        if args.out:
            Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return

    # ------------------------- REAL RUN (single submission) -------------------
    per_seed, nP, nA, F, Fo = run_all(ids, A, O, y, N_PERM, SEEDS, args.n_jobs, "REAL")
    adj = adjudicate(per_seed, nP, nA)

    out["n_videos"] = len(ids)
    out["n_positive"] = int(y.sum())
    out["data_meta"] = dmeta
    out["per_seed"] = per_seed
    out["adjudication"] = adj
    out["null_P_all"] = [float(x) for x in nP]
    out["null_A_all"] = [float(x) for x in nA]
    out["input_sha256"] = {k: sha256_file(v) for k, v in FILES.items()}
    out["paths_touched"] = sorted(set(_TOUCHED))
    out["elapsed_sec"] = time.time() - t0

    log("P_obs=%.4f (>=%.2f) N95(P)=%.4f 3xN95=%.4f" %
        (adj["P_obs"], GO_P, adj["N95_P"], adj["three_x_N95_P"]))
    log("A_obs=%.4f (>=%.2f) N95(A)=%.4f 3xN95=%.4f" %
        (adj["A_obs"], GO_A, adj["N95_A"], adj["three_x_N95_A"]))
    log("clean-margin retention=%.4f (>=%.2f)" %
        (adj["clean_margin_retention_ratio"], GO_RETENTION))
    for k, v in adj["conditions"].items():
        log("  condition %s : %s" % (k, v))
    log("VERDICT: %s" % adj["verdict"])
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log("wrote %s" % args.out)


if __name__ == "__main__":
    try:
        main()
    except Halt as e:
        log("HALT %s" % e)
        sys.exit(3)
