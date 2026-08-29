#!/usr/bin/env python
"""C8 — "Prosody-as-operator binding": conditional vs marginal audio estimand.

Decision rules are FROZEN in idea-stage/PILOT_FREEZE_2026-08-09.md, section "§C8".
Nothing below changes a threshold, a statistic, a dimension, or a rule. Every ambiguity that had
to be resolved is resolved to the least GO-favouring reading and is written into the output JSON
under "interpretations".

Zero test-set contact: pilot_a's path guard is armed; every file open / torch.load goes through
it, so any path component containing "test" HALTs.

Usage:
  python idea-stage/c8_prosody_operator.py --smoke synthetic
  python idea-stage/c8_prosody_operator.py --smoke permuted
  python idea-stage/c8_prosody_operator.py --out idea-stage/c8_prosody.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "idea-stage"))
import pilot_a_disagreement_retrievability as PA  # noqa: E402

Halt = PA.Halt
log = PA.log

# ---------------------------------------------------------------- frozen knobs --
# §C8.4 / §C8.3 / §C8.6 of the freeze.
SEEDS = [20260901, 20260902, 20260903]
NFOLD = 5
D_TEXT = 64
D_PROS = 16
D_INT = 8                 # 8 x 8 = 64 interaction terms
LOGREG_C = 1.0
MAX_ITER = 5000
BAND_LO, BAND_HI = 0.35, 0.65        # middle 30 % by M0 OOF probability rank
BAND_MIN_PER_CLASS = 20              # VOID clause
NPERM = 10                           # per seed -> 3 x 10 = 30 placebo replicates

# frozen decision-rule thresholds (§C8.7)
GO_DELTA = 0.010                     # (a) mean band Delta_int >= +0.010 AUC
GO_NSEED_SIGN = 3                    # (b) 3/3 seeds strictly positive
GO_PLACEBO_PCT = 95                  # (c) mean > P95 of placebo

# non-gating sensitivity bands (§C8.4)
SENS_BANDS = {"mid20": (0.40, 0.60), "mid40": (0.30, 0.70)}

PATHS = {
    "gt_train": ROOT / "data/gt/HateMM/train.jsonl",
    "clip_train": ROOT / "data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt",
    "egemaps": ROOT / "data/audio/HateMM/egemaps_v02_trainval.pt",
    "clap": ROOT / "data/audio/HateMM/clap_larger_clap_general_trainval.pt",
}

FILE_SHAS: dict[str, str] = {}


def _sha(p):
    p = str(p)
    if p not in FILE_SHAS:
        FILE_SHAS[p] = PA.sha256_file(p)
    return FILE_SHAS[p]


def _ids(raw):
    """Caches store ids as a length-1 list wrapping the real list."""
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        return list(raw[0])
    return list(raw)


# --------------------------------------------------------------------- loaders --
def load_population():
    """HateMM train split, empty-transcript rows removed (§C8.1)."""
    gt = {}
    order = []
    with PA.guard_open(PATHS["gt_train"], encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"] in gt:
                raise Halt("HALT_DUP_GT_ID:%s" % r["id"])
            gt[r["id"]] = r
            order.append(r["id"])

    clip = PA.guard_torch_load(PATHS["clip_train"])
    cids = _ids(clip["ids"])
    if cids != order:
        raise Halt("HALT_ID_ORDER_MISMATCH:clip vs gt")
    text = clip["text_feats"].float().numpy()
    y = clip["labels"].long().numpy()
    for i, v in enumerate(cids):
        if int(gt[v]["label"]) != int(y[i]):
            raise Halt("HALT_LABEL_MISMATCH:%s" % v)

    empty = [v for v in order if not gt[v]["text"].strip()]
    keep = np.array([not (not gt[v]["text"].strip()) for v in order], dtype=bool)

    audio = {}
    for tag, key, feat in (("egemaps", "egemaps", "egemaps"), ("clap", "clap", "proj")):
        d = PA.guard_torch_load(PATHS[key])
        aids = _ids(d["ids"])
        pos = {v: i for i, v in enumerate(aids)}
        missing = [v for v in order if v not in pos]
        if missing:
            raise Halt("HALT_AUDIO_JOIN_FAILED:%s:%d e.g. %r" % (tag, len(missing), missing[:3]))
        idx = np.array([pos[v] for v in order])
        A = d[feat].float().numpy()[idx]
        alab = d["labels"].long().numpy()[idx]
        if not np.array_equal(alab, y):
            raise Halt("HALT_AUDIO_LABEL_MISMATCH:%s" % tag)
        if not np.isfinite(A).all():
            raise Halt("HALT_AUDIO_NONFINITE:%s" % tag)
        audio[tag] = A

    return dict(ids=order, text=text, y=y, audio=audio, keep=keep, empty_ids=empty)


# ----------------------------------------------------------------------- model --
def _prep(Xtr, Xte, ncomp, seed):
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr), sc.transform(Xte)
    k = min(ncomp, a.shape[0], a.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(a)
    return pca.transform(a), pca.transform(b)


def _inter(Ttr, Tte, Ptr, Pte):
    """Bilinear D_INT x D_INT outer product, all scaling fitted on the training fold."""
    t_tr, t_te = Ttr[:, :D_INT], Tte[:, :D_INT]
    p_tr, p_te = Ptr[:, :D_INT], Pte[:, :D_INT]
    st = np.maximum(t_tr.std(axis=0), 1e-8)
    sp = np.maximum(p_tr.std(axis=0), 1e-8)
    t_tr, t_te, p_tr, p_te = t_tr / st, t_te / st, p_tr / sp, p_te / sp
    n_tr, n_te = t_tr.shape[0], t_te.shape[0]
    Itr = (t_tr[:, :, None] * p_tr[:, None, :]).reshape(n_tr, -1)
    Ite = (t_te[:, :, None] * p_te[:, None, :]).reshape(n_te, -1)
    sc = StandardScaler().fit(Itr)
    return sc.transform(Itr), sc.transform(Ite)


def _lr(Xtr, ytr, Xte, seed):
    m = LogisticRegression(C=LOGREG_C, max_iter=MAX_ITER, solver="lbfgs", random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def oof_probs(text, pros, y, seed, arms=("M0", "M1", "M2")):
    """5-fold stratified OOF probabilities for the requested arms."""
    n = text.shape[0]
    out = {a: np.zeros(n, dtype=float) for a in arms}
    skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(n), y):
        Ttr, Tte = _prep(text[tr], text[te], D_TEXT, seed)
        need_p = ("M1" in arms) or ("M2" in arms)
        if need_p:
            Ptr, Pte = _prep(pros[tr], pros[te], D_PROS, seed)
        if "M0" in arms:
            out["M0"][te] = _lr(Ttr, y[tr], Tte, seed)
        if "M1" in arms:
            out["M1"][te] = _lr(np.hstack([Ttr, Ptr]), y[tr], np.hstack([Tte, Pte]), seed)
        if "M2" in arms:
            Itr, Ite = _inter(Ttr, Tte, Ptr, Pte)
            out["M2"][te] = _lr(np.hstack([Ttr, Ptr, Itr]), y[tr],
                                np.hstack([Tte, Pte, Ite]), seed)
    return out


# ------------------------------------------------------------------- endpoints --
def band_mask(p0, lo=BAND_LO, hi=BAND_HI):
    """Middle band by rank of the text-only OOF probability. Ties broken by index order."""
    n = len(p0)
    order = np.lexsort((np.arange(n), p0))
    rank = np.empty(n, dtype=float)
    rank[order] = (np.arange(n) + 0.5) / n
    return (rank >= lo) & (rank < hi)


def macro_f1(y, p):
    return f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)


def run_arm(text, pros, y, seeds, do_placebo=True, tag=""):
    """Full frozen protocol for one prosody representation."""
    per_seed = []
    for s in seeds:
        pr = oof_probs(text, pros, y, s)
        m = band_mask(pr["M0"])
        nb, npos = int(m.sum()), int(y[m].sum())
        rec = dict(
            seed=s, band_n=nb, band_pos=npos, band_neg=nb - npos,
            auc_full=dict((a, PA.auroc(y, pr[a])) for a in pr),
            auc_band=dict((a, PA.auroc(y[m], pr[a][m])) for a in pr),
            auc_out=dict((a, PA.auroc(y[~m], pr[a][~m])) for a in pr),
            mf1_band=dict((a, macro_f1(y[m], pr[a][m])) for a in pr),
        )
        rec["d_int_band"] = rec["auc_band"]["M2"] - rec["auc_band"]["M1"]
        rec["d_int_out"] = rec["auc_out"]["M2"] - rec["auc_out"]["M1"]
        rec["d_marg_full"] = rec["auc_full"]["M1"] - rec["auc_full"]["M0"]
        rec["d_int_full"] = rec["auc_full"]["M2"] - rec["auc_full"]["M1"]
        rec["d_int_band_mf1"] = rec["mf1_band"]["M2"] - rec["mf1_band"]["M1"]
        rec["sens"] = {}
        for name, (lo, hi) in SENS_BANDS.items():
            mm = band_mask(pr["M0"], lo, hi)
            rec["sens"][name] = dict(
                n=int(mm.sum()),
                d_int=PA.auroc(y[mm], pr["M2"][mm]) - PA.auroc(y[mm], pr["M1"][mm]))
        per_seed.append(rec)
        log("  %s seed %d: band n=%d (+%d/-%d)  AUC band M0 %.4f M1 %.4f M2 %.4f  d_int %+.4f"
            % (tag, s, nb, npos, nb - npos, rec["auc_band"]["M0"], rec["auc_band"]["M1"],
               rec["auc_band"]["M2"], rec["d_int_band"]))

    placebo = []
    if do_placebo:
        for s in seeds:
            for k in range(NPERM):
                rng = np.random.default_rng(s * 1000 + k)
                perm = np.arange(len(y))
                for lab in (0, 1):                     # permute WITHIN label strata
                    idx = np.where(y == lab)[0]
                    perm[idx] = rng.permutation(idx)
                pr = oof_probs(text, pros[perm], y, s, arms=("M0", "M1", "M2"))
                m = band_mask(pr["M0"])
                placebo.append(dict(seed=s, perm=k,
                                    d_int_band=PA.auroc(y[m], pr["M2"][m])
                                    - PA.auroc(y[m], pr["M1"][m]),
                                    d_marg_full=PA.auroc(y, pr["M1"]) - PA.auroc(y, pr["M0"])))
            log("  %s placebo seed %d done (%d perms)" % (tag, s, NPERM))

    d = np.array([r["d_int_band"] for r in per_seed])
    pl = np.array([r["d_int_band"] for r in placebo]) if placebo else np.array([])
    res = dict(
        per_seed=per_seed, placebo=placebo,
        mean_d_int_band=float(d.mean()),
        n_seed_positive=int((d > 0).sum()),
        mean_d_int_out=float(np.mean([r["d_int_out"] for r in per_seed])),
        mean_d_marg_full=float(np.mean([r["d_marg_full"] for r in per_seed])),
        mean_d_int_full=float(np.mean([r["d_int_full"] for r in per_seed])),
        mean_d_int_band_mf1=float(np.mean([r["d_int_band_mf1"] for r in per_seed])),
        placebo_mean=float(pl.mean()) if pl.size else None,
        placebo_p95=float(np.percentile(pl, GO_PLACEBO_PCT)) if pl.size else None,
        placebo_max=float(pl.max()) if pl.size else None,
        placebo_n_positive=int((pl > 0).sum()) if pl.size else None,
        void=any(min(r["band_pos"], r["band_neg"]) < BAND_MIN_PER_CLASS for r in per_seed),
    )
    res["cond_a_delta"] = bool(res["mean_d_int_band"] >= GO_DELTA)
    res["cond_b_sign"] = bool(res["n_seed_positive"] >= GO_NSEED_SIGN)
    res["cond_c_placebo"] = bool(pl.size and res["mean_d_int_band"] > res["placebo_p95"])
    res["arm_pass"] = bool(res["cond_a_delta"] and res["cond_b_sign"] and res["cond_c_placebo"])
    return res


# ---------------------------------------------------------------------- smokes --
def smoke_synthetic():
    """Positive control: a planted text x prosody interaction the pipeline MUST detect."""
    rng = np.random.default_rng(7)
    n = 705
    # Low-rank generative structure so that the planted directions survive standardisation and
    # land inside the D_INT=8 PCA slice; otherwise the control measures PCA truncation on
    # isotropic noise rather than the pipeline's ability to see an interaction.
    u = rng.normal(size=(n, 2))                      # latent text factors
    v = rng.normal(size=(n, 2))                      # latent prosody factors
    text = u @ rng.normal(size=(2, 100)) + 0.3 * rng.normal(size=(n, 100))
    pros = v @ rng.normal(size=(2, 40)) + 0.3 * rng.normal(size=(n, 40))
    logit = 0.6 * u[:, 0] + 3.0 * u[:, 1] * v[:, 1]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    r = run_arm(text, pros, y, SEEDS[:1], do_placebo=False, tag="SYNTH")
    log("SYNTH mean band d_int = %+.4f  (must be clearly > 0)" % r["mean_d_int_band"])
    if r["mean_d_int_band"] <= 0:
        raise Halt("HALT_SMOKE_SYNTHETIC_FAILED")
    return r


def smoke_permuted(pop):
    """Negative control on real features with labels destroyed; must not clear the bar."""
    rng = np.random.default_rng(11)
    keep = pop["keep"]
    y = pop["y"][keep].copy()
    rng.shuffle(y)
    r = run_arm(pop["text"][keep], pop["audio"]["egemaps"][keep], y, SEEDS[:1],
                do_placebo=False, tag="PERMLAB")
    log("PERMLAB mean band d_int = %+.4f  (bar is +%.3f)" % (r["mean_d_int_band"], GO_DELTA))
    return r


# ------------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", choices=["synthetic", "permuted"], default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    PA.arm_guard()

    if args.smoke == "synthetic":
        smoke_synthetic()
        log("synthetic smoke OK (%.1fs)" % (time.time() - t0))
        return

    pop = load_population()
    keep = pop["keep"]
    log("population: %d train rows, %d empty-transcript excluded, %d analysable"
        % (len(pop["ids"]), len(pop["empty_ids"]), int(keep.sum())))

    if args.smoke == "permuted":
        smoke_permuted(pop)
        log("permuted-label smoke done (%.1fs)" % (time.time() - t0))
        return

    text = pop["text"][keep]
    y = pop["y"][keep]
    arms = {}
    for tag, feat in (("P_egemaps", "egemaps"), ("C_clap", "clap")):
        log("=== arm %s (%s, dim %d)" % (tag, feat, pop["audio"][feat].shape[1]))
        arms[tag] = run_arm(text, pop["audio"][feat][keep], y, SEEDS, do_placebo=True, tag=tag)

    # non-gating sensitivity: keep the 39 empty-transcript rows in
    sens_full = {}
    for tag, feat in (("P_egemaps", "egemaps"), ("C_clap", "clap")):
        r = run_arm(pop["text"], pop["audio"][feat], pop["y"], SEEDS,
                    do_placebo=False, tag=tag + "_all744")
        sens_full[tag] = dict(mean_d_int_band=r["mean_d_int_band"],
                              n_seed_positive=r["n_seed_positive"],
                              mean_d_marg_full=r["mean_d_marg_full"])

    any_void = any(a["void"] for a in arms.values())
    go = any(a["arm_pass"] for a in arms.values())
    verdict = "VOID" if any_void else ("GO" if go else "KILL")

    out = dict(
        pilot="C8_prosody_as_operator",
        freeze="idea-stage/PILOT_FREEZE_2026-08-09.md#C8",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        dataset="HateMM train split only",
        n_train_rows=len(pop["ids"]),
        n_empty_text_excluded=len(pop["empty_ids"]),
        empty_text_ids=pop["empty_ids"],
        n_analysable=int(keep.sum()),
        label_base_rate=float(y.mean()),
        features=dict(
            text="CLIP ViT-L/14-336 text_feats (768) — data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt",
            prosody_arm_P="openSMILE eGeMAPSv02 Functionals (88) — data/audio/HateMM/egemaps_v02_trainval.pt",
            prosody_arm_C="CLAP laion/larger_clap_general proj (1024) — data/audio/HateMM/clap_larger_clap_general_trainval.pt",
            excluded="Whisper-large-v3 encoder (2560) — pre-registered EXCLUDED, carries lexical content",
        ),
        frozen_knobs=dict(seeds=SEEDS, nfold=NFOLD, d_text=D_TEXT, d_pros=D_PROS, d_int=D_INT,
                          logreg_C=LOGREG_C, band=[BAND_LO, BAND_HI], nperm_total=len(SEEDS) * NPERM,
                          metric="ROC-AUC", bar=GO_DELTA, placebo_pct=GO_PLACEBO_PCT),
        arms=arms,
        sensitivity_all744_nongating=sens_full,
        verdict=verdict,
        verdict_rule="GO iff arm P PASSES or arm C PASSES; PASS = (mean band d_int >= +0.010) and (3/3 seeds > 0) and (mean > placebo P95). Otherwise KILL.",
        interpretations=[
            "Empty-transcript rows (39) excluded from the analysis population per freeze §C8.1; "
            "the all-744 re-run is reported as labelled non-gating sensitivity.",
            "Band membership is computed from M0 (text-only) OOF probabilities only, so it is "
            "identical across arms and across placebo replicates (paired comparison).",
            "Delta_int compares M2 against M1, which already contains both main effects, so the "
            "increment isolates the bilinear interaction block.",
            "Placebo permutes prosody rows within label strata, preserving any label-marginal "
            "audio information and destroying only the text-prosody pairing.",
            "AUC is the pinned gating metric; macro-F1 on the band is reported non-gating.",
        ],
        touched_paths=sorted(set(PA._TOUCHED)),
        input_sha256={str(p): _sha(p) for p in PATHS.values()},
        elapsed_s=round(time.time() - t0, 1),
    )
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        log("wrote %s" % args.out)
    log("VERDICT = %s" % verdict)
    for tag, a in arms.items():
        log("%s: mean band d_int %+.4f | seeds+ %d/3 | placebo P95 %+.4f mean %+.4f | "
            "marginal d(M1-M0) full %+.4f | PASS=%s"
            % (tag, a["mean_d_int_band"], a["n_seed_positive"], a["placebo_p95"],
               a["placebo_mean"], a["mean_d_marg_full"], a["arm_pass"]))
    log("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
