#!/usr/bin/env python
"""c09_a0_arena.py -- C09 Stage-0 (A0) analysis.  THE ONLY NEW ANALYSIS CODE.

FROZEN DESIGN OF RECORD: refine-logs/C09_A0_V17_RECORD.md (GO 0C/0H/0I, round 17),
carried into refine-logs/C09_A0_RECORD.md at freeze time.  Section references below
are to that document and every constant comes from configs/c09/c09_a0.json.

WHAT IT COMPUTES, per dataset, in three spaces (head with-null; head remove-null on
the GATE-NULL dataset; raw fused, seed-free):

  O1          reach oracle over the OOF-stable-inversion population        (§5.1)
  D-FELDMAN   AUC_strat(FULL) - AUC_strat(BASE), stratum-conditional and
              incremental, with PERM-STRUCT (marginal) and PERM-STRUCT-COND
              (within-ITEM-STRATUM) nulls and a fit-conditional item bootstrap (§5.2)
  NET         conversion at three frozen operating points, fully costed       (§5.3)
  K-DEG       threshold-degeneracy control on prediction-vector agreement     (§6.2)
  controls    SHUFFLE-POP, RANDOM-POP, UNSTABLE-POP, STRATUM_OCCUPANCY,
              FEATURE_DEGENERACY, DATA-DEFECT-OVERLAP                   (§5.5, §6.3)
  gates       nine HALT gates (§8.1) + three reporting instruments (§8.2)
  verdict     the frozen two-valued rule of §9

TEST CONTACT: NONE.  Only the 36 mint .npz, the two banked train feature caches (raw
leg), and data/gt/*/train.jsonl `text` (data-defect flags, no label read).  A
process-wide open()/torch.load guard raises on any path containing "test_seen" or
"/test"; GATE-LEDGER emits the integer counts.

DETERMINISM: DET-1/DET-2.  Thread env asserted before anything numeric; full runtime
block recorded.  COST: CPU only, <= 8 threads.  Zero GPU / SLURM / Modal.
"""
import argparse
import builtins
import inspect
import hashlib
import json
import os
import socket
import sys
import time

import numpy as np

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
os.chdir(REPO)

if not __debug__:
    raise SystemExit("REFUSING TO RUN: python -O strips the assert-based guards")

# ------------------------------------------------------------ GATE-LEDGER + guards
# One guard for the WHOLE job.  Normally already installed by sitecustomize (the
# sbatch puts scripts/analysis/c09_guard on PYTHONPATH, so the 36 mints and the 2
# fidelity runs are covered too); install() is idempotent, and importing it here
# means the arena is self-protecting even if PYTHONPATH were lost.
sys.path.insert(0, os.path.join(REPO, "scripts/analysis/c09_guard"))
import c09guard  # noqa: E402
LEDGER = c09guard.install()
_OPEN = c09guard._ORIG_OPEN

import torch  # noqa: E402
import mechfix_ops as M          # noqa: E402  frozen F89
import mechnov_pairverify as P   # noqa: E402  frozen F95
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold          # noqa: E402

_TORCH_LOAD = torch.load


def _guarded_torch_load(f, *a, **kw):
    """Belt-and-braces over c09guard's open()-level guard, which torch.load already
    bottoms out in.  Deliberately does NOT increment the ledger: c09guard counts the
    underlying open(), and counting here too would double-count dev_path_opens."""
    assert not c09guard.is_test_like(f), \
        "TEST-SPLIT GUARD (torch.load): refusing to open {}".format(f)
    return _TORCH_LOAD(f, *a, **kw)


torch.load = _guarded_torch_load

DET1_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

FROZEN_SHA = {
    "headspace_mint.py": "cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612",
    "mechfix_ops.py": "635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d",
    "mechnov_pairverify.py": "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d",
    "headspace_fidelity.py": "72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598",
}


def det1_assert(expect):
    miss = [k for k in DET1_KEYS if os.environ.get(k) != expect]
    assert not miss, "DET-1 violated: {} not exported as {}".format(miss, expect)


def runtime_block():
    import threadpoolctl, scipy, sklearn
    return {"env": {k: os.environ.get(k) for k in DET1_KEYS},
            "threadpools": threadpoolctl.threadpool_info(),
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                         "torch": torch.__version__},
            "torch_num_threads": int(torch.get_num_threads()),
            "node": socket.gethostname()}


def sha256_of(path):
    h = hashlib.sha256()
    with _OPEN(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------ helpers
def R(x):
    """Frozen half-up rounding (§5.3); numpy/python round() is banker's."""
    return int(np.floor(float(x) + 0.5))


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def mf1(y, p):
    return float(M.macro_f1(y, p))


def mw_auc(sp, sn):
    """Mann-Whitney P(pos > neg) with ties counted 0.5, rank-based (O(m log m))."""
    m, k = len(sp), len(sn)
    v = np.concatenate([sp, sn])
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    r = np.empty(len(v), dtype="float64")
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(len(v), dtype="float64")
    ranks[order] = r
    U = ranks[:m].sum() - m * (m + 1) / 2.0
    return U / float(m * k)


def strat_auc_cell(sc, y, st):
    """§5.2: within-stratum Mann-Whitney pooled with n_pos*n_neg weights.
    Returns (auc | None, total_weight, n_weighted_strata, p_w, per_stratum)."""
    num = den = 0.0
    nw = pw = 0
    per = []
    for s in np.unique(st):
        m = st == s
        sp, sn = sc[m][y[m] == 1], sc[m][y[m] == 0]
        if len(sp) == 0 or len(sn) == 0:
            per.append({"stratum": int(s), "n_pos": int(len(sp)),
                        "n_neg": int(len(sn)), "weight": 0})
            continue
        w = float(len(sp) * len(sn))
        num += mw_auc(sp, sn) * w
        den += w
        nw += 1
        pw += len(sp)
        per.append({"stratum": int(s), "n_pos": int(len(sp)), "n_neg": int(len(sn)),
                    "weight": int(w), "auc": round(float(mw_auc(sp, sn)), 6)})
    if den == 0.0:
        return None, 0.0, 0, 0, per
    return num / den, den, nw, pw, per


def pooled_auc(sc, y, st, cell):
    """Mean over (dataset, seed) cells of the within-cell stratified AUC.
    Cells whose strata are all single-class are dropped and named."""
    vals, dropped = [], []
    for c in np.unique(cell):
        m = cell == c
        a, _, nw, _, _ = strat_auc_cell(sc[m], y[m], st[m])
        if a is None or nw == 0:
            dropped.append(int(c))
        else:
            vals.append(a)
    if not vals:
        return None, dropped
    return float(np.mean(vals)), dropped


def fit_lr(Xtr, ytr, Xte, cfg):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-12] = 1.0     # not "== 0": a post-l2n norm column has sd ~1e-16
    clf = LogisticRegression(penalty="l2", C=cfg["lr_C"], solver="lbfgs",
                             max_iter=cfg["lr_max_iter"], class_weight="balanced",
                             tol=cfg["lr_tol"])
    clf.fit((Xtr - mu) / sd, ytr)
    return clf.predict_proba((Xte - mu) / sd)[:, 1]


def fit_gbm(Xtr, ytr, Xte, cfg):
    clf = HistGradientBoostingClassifier(
        max_iter=cfg["gbm_max_iter"], learning_rate=cfg["gbm_lr"],
        max_depth=cfg["gbm_max_depth"], l2_regularization=cfg["gbm_l2"],
        random_state=cfg["gbm_random_state"])
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def terciles(v):
    return np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])


def bucket(v, edges):
    return np.searchsorted(np.asarray(edges), v, side="right")


PURITY_EDGES = [0.60, 0.80, 0.95]

BASE_FEATS = ["abs_score", "c_i", "pred_purity", "sim_mean", "sim_sd", "sim_gap",
              "dens50", "own_norm"]
STRUCT_FEATS = ["first_diff_rank", "runs", "deg_mean", "deg_sd", "class_gap"]
FULL_FEATS = BASE_FEATS + STRUCT_FEATS

FEATURE_MANIFEST = {
    "abs_score": ["deployed_vote.votes"],
    "c_i": ["mean over seeds of deployed_vote.votes"],
    "pred_purity": ["deployed_vote.I -> BANK labels", "deployed_vote.preds"],
    "sim_mean": ["deployed_vote.sim"], "sim_sd": ["deployed_vote.sim"],
    "sim_gap": ["deployed_vote.sim"],
    "dens50": ["faiss k=50 sims: query key vs bank keys"],
    "own_norm": ["query key, pre-L2 norm"],
    "first_diff_rank": ["deployed_vote.I -> BANK labels"],
    "runs": ["deployed_vote.I -> BANK labels"],
    "deg_mean": ["deployed_vote.I (bank indices only)"],
    "deg_sd": ["deployed_vote.I (bank indices only)"],
    "class_gap": ["faiss per-class top-1 sims", "BANK labels"],
}



class BankLabels(object):
    """The ONLY label channel the feature phase has (GATE-BLIND, design 8.1).

    A caller must name the fold it is scoring; the object then serves labels only
    for indices inside THAT fold's frozen fitting pool.  A query-side (hold-out)
    label cannot be obtained through it -- the assertion fires first.  This is the
    legal channel of mechfix_ops.py:91 (the bank's labels are what the deployed
    vote reads); it is audited, not forbidden.
    """

    def __init__(self, lab, splits):
        self._lab = np.asarray(lab)
        self._pool = [set(map(int, np.asarray(f).tolist())) for f, _ in splits]
        self._ho = [set(map(int, np.asarray(h).tolist())) for _, h in splits]
        self.audit = {"reads": 0, "labels_served": 0,
                      "holdout_label_requests_refused": 0,
                      "pool_violations_refused": 0, "per_fold": {}}

    @property
    def n(self):
        return len(self._lab)

    def bank(self, fold, idx):
        idx = np.asarray(idx)
        st = set(map(int, idx.tolist()))
        if not st <= self._pool[fold]:
            self.audit["pool_violations_refused"] += 1
            raise AssertionError(
                "GATE-BLIND: bank-label request outside fold {}'s fitting "
                "pool".format(fold))
        bad = st & self._ho[fold]
        if bad:
            self.audit["holdout_label_requests_refused"] += 1
            raise AssertionError(
                "GATE-BLIND: hold-out label requested for fold {} "
                "({} indices)".format(fold, len(bad)))
        self.audit["reads"] += 1
        self.audit["labels_served"] += int(idx.size)
        self.audit["per_fold"].setdefault(str(fold), 0)
        self.audit["per_fold"][str(fold)] += int(idx.size)
        return self._lab[idx]


# ------------------------------------------------------------------ arena emitter
def emit_cell(X_by_fold, banklab, splits, topk, fixk_grid, drop=None,
              rawnorm_by_fold=None):
    """Deployed vote + every feature input for one (space, seed) cell.

    GATE-BLIND: this signature admits no gold-label array and no target-derived
    array.  `banklab` is a BankLabels channel that can serve only fitting-pool
    labels for the fold named in the call.
    """
    n = banklab.n
    F = {k: np.full(n, np.nan) for k in
         ("score", "pred_purity", "sim_mean", "sim_sd", "sim_gap", "dens50",
          "own_norm", "first_diff_rank", "runs", "deg_mean", "deg_sd", "class_gap")}
    pred = np.full(n, -1, dtype=int)
    fold = np.full(n, -1, dtype=int)
    fixk = {kk: np.full(n, np.nan) for kk in list(fixk_grid) + [topk]}
    # independent path: a separate M.deployed_vote call per k', which is literally
    # how §6.2 defines the fixed-k vote.  The in-line map above reuses the k=20
    # neighbour block, so on its own it could not detect a defect in the truncation
    # path that K-DEG reads; the two are compared by GATE-FIXK20.
    fixk_indep = {kk: np.full(n, np.nan) for kk in list(fixk_grid) + [topk]}
    fixk_tie = {kk: np.zeros(n, dtype=bool) for kk in list(fixk_grid) + [topk]}
    for f, (fit_idx, ho_idx) in enumerate(splits):
        fit_idx, ho_idx = np.asarray(fit_idx), np.asarray(ho_idx)
        if drop is not None:
            fit_idx = fit_idx[fit_idx != drop]
            ho_idx = ho_idx[ho_idx != drop]
        X = X_by_fold[f]
        assert np.asarray(X).dtype == np.float64, (
            "key matrix must be float64: mechfix_ops._norm32 L2-normalises a float32 "
            "input IN PLACE (np.asarray is a no-op on float32 C-contiguous), so a "
            "float32 key matrix would drift ~6e-8 on every call and perturb dens50 / "
            "class_gap. headspace_mint.py:304 writes float64; this asserts it.")
        lb = banklab.bank(f, fit_idx)
        # fresh slices on EVERY call: fancy indexing copies, so an in-place
        # renormalisation inside deployed_vote can only touch that temporary.
        v, p, I, sim = M.deployed_vote(X[fit_idx], lb, X[ho_idx], topk=topk)
        nl = lb[I].astype("float64")
        F["score"][ho_idx] = v
        pred[ho_idx] = p
        fold[ho_idx] = f
        for kk in fixk:
            w = M._rank_weights(kk)
            fixk[kk][ho_idx] = ((nl[:, :kk] * 2 - 1) * sim[:, :kk] * w).sum(1) / w.sum()
            vk, _pk, _Ik, _sk = M.deployed_vote(X[fit_idx], lb, X[ho_idx], topk=kk)
            fixk_indep[kk][ho_idx] = vk
            if kk < topk:
                fixk_tie[kk][ho_idx] = (sim[:, kk - 1] == sim[:, kk])
        F["pred_purity"][ho_idx] = (nl == p[:, None]).mean(1)
        F["sim_mean"][ho_idx] = sim.mean(1)
        F["sim_sd"][ho_idx] = sim.std(1)
        F["sim_gap"][ho_idx] = sim[:, 0] - sim[:, topk - 1]
        # own_norm = the item's own L2 norm BEFORE normalisation (BASE feature 8,
        # design 5.2).  X is already unit-norm, so measuring it here would yield
        # 1.0 +/- 1e-16; the pre-normalisation norms are carried in separately.
        F["own_norm"][ho_idx] = np.asarray(rawnorm_by_fold[f], dtype="float64")[ho_idx]
        b32, q32 = M._norm32(X[fit_idx]), M._norm32(X[ho_idx])
        k50 = min(50, b32.shape[0])
        D50, _ = M._flat_ip(b32, q32, k50)
        F["dens50"][ho_idx] = D50.astype("float64").mean(1)
        gap = np.zeros(len(ho_idx))
        for c, sgn in ((1, 1.0), (0, -1.0)):
            ic = np.flatnonzero(lb == c)
            Dc, _ = M._flat_ip(np.ascontiguousarray(b32[ic]), q32, 1)
            gap += sgn * Dc[:, 0].astype("float64")
        F["class_gap"][ho_idx] = gap
        maj = (nl.mean(1) >= 0.5).astype("float64")
        diff = nl != maj[:, None]
        fr = np.full(len(ho_idx), float(topk + 1))
        has = diff.any(1)
        fr[has] = diff[has].argmax(1) + 1.0
        F["first_diff_rank"][ho_idx] = fr
        F["runs"][ho_idx] = (nl[:, 1:] != nl[:, :-1]).sum(1).astype("float64") + 1.0
        deg = np.bincount(I.ravel(), minlength=len(fit_idx)).astype("float64")
        F["deg_mean"][ho_idx] = deg[I].mean(1)
        F["deg_sd"][ho_idx] = deg[I].std(1)
    return {"F": F, "pred": pred, "fold": fold, "fixk": fixk,
            "fixk_indep": fixk_indep, "fixk_tie": fixk_tie}


def build_features(banklab, X_by_seed_fold, rawnorm_by_seed_fold, splits, seeds,
                   topk, fixk_grid, drop):
    """Feature-construction phase, half 1.  GATE-BLIND by signature: a top-level
    function, so it cannot close over any caller-local label or target array, and
    its parameters contain none."""
    return {s: emit_cell(X_by_seed_fold[s], banklab, splits, topk, fixk_grid, drop,
                         rawnorm_by_fold=rawnorm_by_seed_fold[s]) for s in seeds}


def build_design(cells, c_i, seeds):
    """Feature-construction phase, half 2.  GATE-BLIND by signature: `cells` holds
    only vote/geometry arrays and `c_i` is a function of |score| alone."""
    XF = np.vstack([design_matrix(cells[s]["F"], c_i, FULL_FEATS) for s in seeds])
    XB = np.vstack([design_matrix(cells[s]["F"], c_i, BASE_FEATS) for s in seeds])
    return XF, XB


def design_matrix(F, c_i, feats):
    cols = [(c_i if f == "c_i" else
             (np.abs(F["score"]) if f == "abs_score" else F[f])) for f in feats]
    return np.column_stack(cols)


# --------------------------------------------------------------------- K-DEG twins
def topk_mask(vals, k, keep, n):
    """The k items with the SMALLEST vals among `keep`."""
    v = np.where(keep, vals, np.inf)
    idx = np.argsort(v, kind="mergesort")[:k]
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def pred_agree(S, twin, n_keep):
    return 1.0 - float((S ^ twin).sum()) / float(n_keep)


def k_deg_block(S, k, cells, seeds, keep, c_i, inv, n, cfg):
    nk = int(keep.sum())
    out = {"n_for_pred_agree": nk, "kill_line": cfg["deg_kill"], "twins": {}}
    scores = {s: cells[s]["F"]["score"] for s in seeds}

    # -- THRESH-SYM
    ti = topk_mask(c_i, k, keep, n)
    ps = [pred_agree(S, topk_mask(np.abs(scores[s]), k, keep, n), nk) for s in seeds]
    out["twins"]["THRESH_SYM"] = {
        "pred_agree_per_item": round(pred_agree(S, ti, nk), 6),
        "pred_agree_per_seed_mean": round(float(np.mean(ps)), 6),
        "set_overlap_per_item_descriptive": round(float((S & ti).sum()) / k, 6),
        "read_by_K_DEG": True}

    # -- THRESH-BEST (hindsight-best one-sided band; its net is NOT a finding, §6.2)
    band_sizes = {}

    def bands(sc):
        # topk_mask always returns exactly k items; if a side has fewer than k
        # members the remainder are inf-valued fillers.  §6.2 defines the twin as
        # the k items with the smallest score ON THAT SIDE, so the realised side
        # sizes are recorded and the shortfall named rather than left silent.
        pos_n, neg_n = int((keep & (sc >= 0)).sum()), int((keep & (sc < 0)).sum())
        band_sizes["side_sizes"] = {"nonneg": pos_n, "neg": neg_n, "k": k}
        band_sizes["side_short_of_k"] = bool(pos_n < k or neg_n < k)
        return [topk_mask(np.where(sc >= 0, sc, np.inf), k, keep, n),
                topk_mask(np.where(sc < 0, -sc, np.inf), k, keep, n)]

    def best_band(sc, invm):
        cands = bands(sc)
        nets = [2 * int((c & invm).sum()) - int(c.sum()) for c in cands]
        return cands[int(np.argmax(nets))]

    def best_band_meanseed(sc):
        """PINNED READING for the per-ITEM band (§6.2 says 'the one with the higher
        net_s'; net_s is per seed, and the deployed rule averages it over seeds, so
        the per-item analogue is mean_s net_s).  The alternative reading -- score the
        band by the union of the three seeds' error sets -- is computed too and both
        pred_agree values are emitted, so the choice is auditable rather than silent."""
        cands = bands(sc)
        nets = [float(np.mean([2 * int((c & inv[s]).sum()) - int(c.sum())
                               for s in seeds])) for c in cands]
        return cands[int(np.argmax(nets))]

    sc_item = np.mean([scores[s] for s in seeds], axis=0)
    inv_any = np.zeros(n, dtype=bool)
    for s in seeds:
        inv_any |= inv[s]
    pb = [pred_agree(S, best_band(scores[s], inv[s]), nk) for s in seeds]
    tb_pinned = best_band_meanseed(sc_item)
    tb_alt = best_band(sc_item, inv_any)
    out["twins"]["THRESH_BEST"] = {
        "pred_agree_per_item": round(pred_agree(S, tb_pinned, nk), 6),
        "pred_agree_per_seed_mean": round(float(np.mean(pb)), 6),
        "set_overlap_per_item_descriptive": round(float((S & tb_pinned).sum()) / k, 6),
        "read_by_K_DEG": True,
        "PINNED_READING": "per-item band scored by mean_s net_s",
        "realised_band_sizes": dict(band_sizes),
        "alternative_reading_union_of_seed_error_sets": {
            "pred_agree_per_item": round(pred_agree(S, tb_alt, nk), 6),
            "selects_same_band": bool(np.array_equal(tb_pinned, tb_alt)),
            "read_by_K_DEG": False},
        "note": "the twin's own net/d_acc select and score the twin only; NOT findings"}

    # -- FIXK.  One twin-builder, used at both scales: per seed (base = that seed's
    # deployed prediction) and per item (base = the sign of the seed-mean fixed-k
    # vote).  §6.2: "computed on BOTH scales ... K-DEG reads the maximum over the two
    # scales, the conservative direction for a gate whose job is to fire."
    def fixk_twin(fxmap, base_pred):
        sizes, best_kp, best_ratio, best_flip = {}, None, -1.0, None
        for kp in cfg["fixk_grid"]:
            flip = keep & ((fxmap[kp] >= 0).astype(int) != base_pred)
            sizes[str(kp)] = int(flip.sum())
            ratio = float((S & flip).sum()) / float(k)
            if ratio > best_ratio + 1e-12:      # ties -> smallest k' (grid ascending)
                best_kp, best_ratio, best_flip = kp, ratio, flip
        if all(v == 0 for v in sizes.values()):
            return None, best_kp, sizes, True
        dsc = np.abs(fxmap[best_kp] - fxmap[cfg["topk"]])
        nf = int(best_flip.sum())
        if nf > k:
            twin = topk_mask(np.where(best_flip, -dsc, np.inf), k, keep, n)
        elif nf < k:
            twin = best_flip.copy()
            pad = np.where(keep & ~best_flip, -dsc, np.inf)
            twin[np.argsort(pad, kind="mergesort")[:k - nf]] = True
        else:
            twin = best_flip.copy()
        return twin, best_kp, sizes, False

    per_seed, ags, sizes_all = {}, [], {}
    for s in seeds:
        twin, best_kp, sizes, all_empty = fixk_twin(cells[s]["fixk"], cells[s]["pred"])
        sizes_all[str(s)] = sizes
        if twin is not None:
            ags.append(pred_agree(S, twin, nk))
        per_seed[str(s)] = {"best_k_prime": best_kp, "flip_sizes": sizes,
                            "DEGENERATE_ALL_EMPTY": bool(all_empty)}

    # per-ITEM scale: seed-mean fixed-k vote, base = sign of the seed-mean k=20 vote
    grid_keys = list(cfg["fixk_grid"]) + [cfg["topk"]]
    mi = {kk: np.mean([cells[s]["fixk"][kk] for s in seeds], axis=0) for kk in grid_keys}
    base_item = (mi[cfg["topk"]] >= 0).astype(int)
    twin_i, best_kp_i, sizes_i, empty_i = fixk_twin(mi, base_item)
    pa_item = None if twin_i is None else round(pred_agree(S, twin_i, nk), 6)
    if ags or pa_item is not None:
        out["twins"]["FIXK"] = {
            "pred_agree_per_seed_mean": (round(float(np.mean(ags)), 6) if ags else None),
            "pred_agree_per_item": pa_item,
            "per_item": {"best_k_prime": best_kp_i, "flip_sizes": sizes_i,
                         "DEGENERATE_ALL_EMPTY": bool(empty_i),
                         "base": "sign of the seed-mean k=20 vote"},
            "per_seed": per_seed, "read_by_K_DEG": True,
            "seeds_contributing": [str(s) for s in seeds
                                   if not per_seed[str(s)]["DEGENERATE_ALL_EMPTY"]],
            "seeds_excluded_degenerate": [str(s) for s in seeds
                                          if per_seed[str(s)]["DEGENERATE_ALL_EMPTY"]],
            "PINNED_READING": "§6.2's DEGENERATE exemption is applied PER SEED: a seed "
                              "with |flip(k')| = 0 at every k' contributes no twin and "
                              "is dropped from the mean; the twin is still read by "
                              "K-DEG as long as at least one seed is non-degenerate. "
                              "The stricter reading (exempt the twin entirely unless "
                              "ALL seeds are degenerate) would only ever make K-DEG "
                              "fire more often, i.e. KILL more readily."}
    else:
        out["twins"]["FIXK"] = {"per_seed": per_seed,
                                "per_item": {"DEGENERATE_ALL_EMPTY": True},
                                "read_by_K_DEG": False,
                                "status": "DEGENERATE_ALL_EMPTY"}
    vals = [float(v) for t in out["twins"].values() if t.get("read_by_K_DEG")
            for kk, v in t.items()
            if kk.startswith("pred_agree") and v is not None]
    out["scales_read"] = ("both (per-item and per-seed-mean), maximum taken "
                          "(§6.2, §9(4))")
    out["max_pred_agree"] = round(max(vals), 6) if vals else None
    out["K_DEG_fires"] = bool(vals and max(vals) >= cfg["deg_kill"])
    return out


def cd_ds(ds):
    return P.DATASETS[ds]["ds"]


def gate_null_census(ds, cd, lab, cfg):
    """GATE-NULL (§8.2): structural-zero census on the operative train cache."""
    _, img, txt, _ = P.load_cache(cd["cache_dir"], "train", cd["model"])
    zi = np.flatnonzero(np.abs(np.asarray(img)).sum(1) == 0)
    zt = np.flatnonzero(np.abs(np.asarray(txt)).sum(1) == 0)
    return {"cache": os.path.join(cd["cache_dir"], "train_{}.pt".format(cd["model"])),
            "zero_img_rows": sorted(map(int, zi)), "zero_txt_rows": sorted(map(int, zt)),
            "labels_of_zero_rows": {int(i): int(lab[i]) for i in
                                    sorted(set(zi.tolist()) | set(zt.tolist()))},
            "is_gate_null_dataset": bool(ds == cfg["null_row_dataset"]),
            "head_space_note": "in head space the zero row is NOT zero (nn.Linear bias, "
                               "no final normalisation); the C01/C02 raw-space exact-zero "
                               "contract does not transfer and is not asserted (§8.2)"}


# ------------------------------------------------------------------ space analysis
def run_space(tag, ds, cfg, lab, splits, X_by_seed_fold, seeds, log, flags=None,
              drop=None, rawnorm_by_seed_fold=None, rng_children=None):
    t0 = time.time()
    n = len(lab)
    topk = cfg["topk"]
    keep = np.ones(n, dtype=bool)
    if drop is not None:
        keep[drop] = False
    nk = int(keep.sum())
    out = {"tag": tag, "dataset": ds, "n_scored": nk, "seeds": list(seeds),
           "seed_free_raw_space": tag == "raw"}

    banklab = BankLabels(lab, splits)
    cells = build_features(banklab, X_by_seed_fold, rawnorm_by_seed_fold, splits,
                           seeds, topk, cfg["fixk_grid"], drop)
    fold_of = cells[seeds[0]]["fold"]

    # ---- floors, GATE-ARENA, GATE-FIXK20, GATE-PARITY-FOLD
    floors = {}
    for s in seeds:
        c = cells[s]
        m = keep & (c["pred"] >= 0)
        floors[str(s)] = {
            "acc": float(acc(lab[m], c["pred"][m])), "mF1": float(mf1(lab[m], c["pred"][m])),
            "n": int(m.sum()),
            "fold_acc": [round(acc(lab[m & (fold_of == f)], c["pred"][m & (fold_of == f)]), 4)
                         for f in range(cfg["k_folds"])]}
    out["floors"] = {k: {"acc": round(v["acc"], 10), "mF1": round(v["mF1"], 10),
                         "n": v["n"], "fold_acc": v["fold_acc"]} for k, v in floors.items()}
    posrate = float(lab[keep].mean())
    maj = max(posrate, 1.0 - posrate)
    band = [round(maj + cfg["arena_band_margin"], 4), cfg["arena_band_upper"]]
    out["GATE_ARENA"] = {"majority_rate": round(maj, 4), "band": band,
                         "pass": all(band[0] <= floors[str(s)]["acc"] <= band[1]
                                     for s in seeds)}
    fk = {}
    for s in seeds:
        c = cells[s]
        m = keep & (c["pred"] >= 0)
        p20 = (c["fixk"][topk][m] >= 0).astype(int)
        # every k' on the grid: the in-line truncation vs an independent
        # M.deployed_vote(topk=k') call.  This is the conjunct that actually checks
        # the fixed-k path K-DEG reads; the k'=20 identity alone is near-vacuous.
        indep = {}
        for kk in sorted(c["fixk"]):
            a_, b_ = c["fixk"][kk][m], c["fixk_indep"][kk][m]
            tie = c["fixk_tie"][kk][m]
            fin = np.isfinite(a_) & np.isfinite(b_)
            cmpm = fin & ~tie          # only rows whose top-k' SET is unique
            dmax = float(np.max(np.abs(a_[cmpm] - b_[cmpm]))) if cmpm.any() else None
            peq = bool(np.array_equal((a_[cmpm] >= 0), (b_[cmpm] >= 0)))
            indep[str(kk)] = {
                "max_abs_diff": None if dmax is None else round(dmax, 15),
                "predictions_identical": peq,
                "n_compared": int(cmpm.sum()),
                "n_tie_ambiguous_excluded": int((fin & tie).sum()),
                "ok": bool(fin.all() and peq and int(cmpm.sum()) > 0
                           and dmax is not None and dmax < 1e-9)}
        fk[str(s)] = {"changed": int((p20 != c["pred"][m]).sum()),
                      "d_acc": round(acc(lab[m], p20) - floors[str(s)]["acc"], 12),
                      "independent_recomputation_per_k": indep,
                      "independent_all_ok": bool(all(v["ok"] for v in indep.values()))}
    out["GATE_FIXK20"] = {
        "per_seed": fk,
        "checks": "(a) the k'=20 fixed-k vote reproduces the deployed prediction "
                  "exactly; (b) EVERY k' on the grid matches an independent "
                  "M.deployed_vote(topk=k') call, which is the truncation path K-DEG "
                  "reads and which (a) alone cannot exercise",
        "tie_note":
            "A row whose k'-th and (k'+1)-th neighbour sims are EXACTLY equal has no "
            "unique top-k' set, so its fixed-k vote is not a function of the data and "
            "faiss may return either neighbour depending on the search k. Such rows "
            "are excluded from the equality and counted as n_tie_ambiguous_excluded. "
            "Measured, not assumed: the raw HateMM space contains exact duplicate "
            "keys with OPPOSITE labels (item 537 has two bank neighbours at sim 1.0), "
            "a property of the CLIP features rather than of this code.",
        "tolerance_note":
            "(b) requires the induced PREDICTIONS to be identical AND the votes to "
            "agree to 1e-9. faiss is bit-deterministic in the search k: with fresh "
            "key slices the independent recomputation reproduces the in-line "
            "truncation EXACTLY (max|dvote| = 0) at both float32 and float64. An "
            "earlier draft of this gate reported a ~2.1e-7 discrepancy and ascribed "
            "it to faiss; that reading was WRONG and is retracted. The true cause was "
            "mechfix_ops._norm32 renormalising a reused float32 array IN PLACE, so "
            "each extra deployed_vote call drifted the caller's own keys by ~6e-8. "
            "This is fixed at source (fresh slices per call) and the float64 key "
            "dtype is now asserted in emit_cell, so the tolerance is slack, not a "
            "slackened gate.",
        "pass": all(v["changed"] == 0 and abs(v["d_acc"]) < 1e-12
                    and v["independent_all_ok"] for v in fk.values())}

    # ---- populations, thresholds
    inv = {s: (cells[s]["pred"] != lab) & keep for s in seeds}
    nwrong = {str(s): int(inv[s].sum()) for s in seeds}
    tw = np.sum([inv[s].astype(int) for s in seeds], axis=0)
    P0 = (tw == len(seeds)) & keep
    ok = (tw == 0) & keep
    unstable = keep & (tw > 0) & (tw < len(seeds))
    c_i = np.mean([np.abs(cells[s]["F"]["score"]) for s in seeds], axis=0)
    mean_wrong = float(np.mean(list(nwrong.values())))
    out["populations"] = {
        "per_seed_wrong": nwrong, "mean_wrong": round(mean_wrong, 6),
        "P0": int(P0.sum()), "always_correct": int(ok.sum()),
        "n_unstable": int(unstable.sum())}
    out["GATE_SEED_inversion_sets"] = {str(s): sorted(map(int, np.flatnonzero(inv[s])))
                                       for s in seeds}

    thr = {}
    for f in range(cfg["k_folds"]):
        v = c_i[P0 & (fold_of != f)]
        thr[f] = float(np.median(v)) if len(v) else float("inf")
    out["tau_hi"] = {"per_fold": {str(f): round(thr[f], 10) for f in thr},
                     "spread": round(max(thr.values()) - min(thr.values()), 10)
                     if P0.sum() else None,
                     "full_sample_descriptive": round(float(np.median(c_i[P0])), 10)
                     if P0.sum() else None}
    own_thr = np.array([thr[fold_of[i]] if fold_of[i] >= 0 else np.inf for i in range(n)])
    Ptau = {"tau_0": P0.copy(), "tau_hi": P0 & (c_i >= own_thr)}
    out["populations"]["P_tau"] = {k: int(v.sum()) for k, v in Ptau.items()}
    if P0.sum():
        cs = np.sort(c_i[P0])
        qm = None
        for j in range(len(cs)):
            if (len(cs) - j) / float(nk) >= cfg["reach_bar"]:
                qm = j / float(len(cs))
        out["q_max"] = round(qm, 4) if qm is not None else None
        out["q_max_note"] = ("ACCURACY-LEG ONLY: the largest quantile q of c_i within "
                             "P_0 whose upper tail still contains >= 0.050*n_scored "
                             "items, i.e. the accuracy half of K-REACH's conjunction. "
                             "Descriptive; read by no rule. The macro-F1 leg is not "
                             "a function of |P_tau| alone, so no closed-form quantile "
                             "exists for it.")

    # ---- features (GATE-BLIND)
    XF, XB = build_design(cells, c_i, seeds)
    item = np.tile(np.arange(n), len(seeds))
    cellid = np.repeat(np.arange(len(seeds)), n)
    rfold = np.tile(fold_of, len(seeds))
    rkeep = np.tile(keep, len(seeds))
    aud = banklab.audit
    out["GATE_BLIND"] = {
        "binding_enforcement": "signature + BankLabels audit",
        "feature_phase_signatures": {
            "build_features": str(inspect.signature(build_features)),
            "emit_cell": str(inspect.signature(emit_cell)),
            "build_design": str(inspect.signature(build_design)),
            "design_matrix": str(inspect.signature(design_matrix))},
        "signature_admits_no_label_or_target_array": bool(
            not ({"lab", "labels", "y", "inv", "is_inversion", "P0",
                  "is_stable_inversion", "target"}
                 & set().union(*[set(inspect.signature(f).parameters)
                                 for f in (build_features, emit_cell, build_design,
                                           design_matrix)]))),
        "bank_label_audit": {
            "calls": aud["reads"], "labels_served": aud["labels_served"],
            "holdout_label_requests_refused": aud["holdout_label_requests_refused"],
            "pool_violations_refused": aud["pool_violations_refused"],
            "counter_semantics": "these two count REFUSALS, and a refusal raises, so "
                                 "a nonzero value cannot coexist with a completed run "
                                 "-- they are here so the artifact records that the "
                                 "channel was armed, not as evidence by themselves; "
                                 "the evidence is calls == expected_calls below",
            "per_fold_labels_served": dict(aud["per_fold"]),
            "expected_calls": len(seeds) * cfg["k_folds"],
            "legal_because": "bank labels are what the deployed vote reads "
                             "(mechfix_ops.py:91); the channel refuses any index "
                             "outside the named fold's frozen fitting pool"},
        "guarded_arrays_declared_by_8_1": ["lab_query", "is_inversion[seed]",
                                           "is_stable_inversion"],
        "READ_COUNTER_CLAUSE_DISCHARGED_STRUCTURALLY": (
            "8.1 asks for three read-COUNTING guards over the feature phase and for "
            "all three counts to be 0. That clause is discharged STRUCTURALLY here, "
            "not by counting, and the counters are not emitted as evidence: the "
            "feature phase lives in top-level functions (build_features, emit_cell, "
            "build_design, design_matrix) whose signatures are emitted above and "
            "admit no label or target array, so none of the three arrays is in "
            "scope and a counter over them would read 0 for ANY code whatsoever -- "
            "it would be a tautology, not a check. The binding, MEASURED evidence "
            "is bank_label_audit below: exactly seeds x folds calls, every one "
            "served from the named fold's frozen fitting pool, zero refusals. This "
            "is a DECLARED DEVIATION from 8.1's literal wording, recorded as such "
            "in refine-logs/C09_A0_RECORD.md."),
        "signature_check_is_a_name_blacklist": True,
        "feature_manifest": FEATURE_MANIFEST,
        "n_base": len(BASE_FEATS), "n_full": len(FULL_FEATS),
        "pass": bool(aud["holdout_label_requests_refused"] == 0
                     and aud["pool_violations_refused"] == 0
                     and aud["reads"] == len(seeds) * cfg["k_folds"]
                     and aud["labels_served"] > 0)}
    assert out["GATE_BLIND"]["signature_admits_no_label_or_target_array"], \
        "GATE-BLIND: a feature-phase signature admits a label/target array"
    assert out["GATE_BLIND"]["pass"], "GATE-BLIND: guarded array read during features"

    # ---- FEATURE_DEGENERACY
    fdg = {}
    for si, s in enumerate(seeds):
        m = (cellid == si) & rkeep
        blk = {f: {"sd": round(float(np.nanstd(XF[m, j])), 10),
                   "n_distinct": int(len(np.unique(np.round(XF[m, j], 12))))}
              for j, f in enumerate(FULL_FEATS)}
        uni = keep & (cells[s]["F"]["first_diff_rank"] == topk + 1)
        blk["uniform_neighbourhood_fraction"] = {
            "all": round(float(uni.sum()) / max(1, nk), 4),
            "in_P0": round(float((uni & P0).sum()) / max(1, int(P0.sum())), 4),
            "in_always_correct": round(float((uni & ok).sum()) / max(1, int(ok.sum())), 4)}
        fdg[str(s)] = blk
    out["FEATURE_DEGENERACY"] = fdg

    # ---- strata (row-level per cell) and ITEM-STRATUM
    strat = np.full(len(item), -1, dtype=int)
    edges_used = {}
    for si, s in enumerate(seeds):
        av = np.abs(cells[s]["F"]["score"])
        e = terciles(av[keep])
        strat[cellid == si] = bucket(av, e) * 4 + bucket(cells[s]["F"]["pred_purity"],
                                                         PURITY_EDGES)
        edges_used[str(s)] = [round(float(x), 10) for x in e]
    e_c = terciles(c_i[keep])
    ppm = np.mean([cells[s]["F"]["pred_purity"] for s in seeds], axis=0)
    istrat = bucket(c_i, e_c) * 4 + bucket(ppm, PURITY_EDGES)
    out["strata"] = {"row_score_tercile_edges_per_cell": edges_used,
                     "purity_bucket_edges": PURITY_EDGES,
                     "item_stratum_c_tercile_edges": [round(float(x), 10) for x in e_c]}

    assert rng_children is not None and len(rng_children) == 6
    ch = rng_children
    out["rng"] = {"root_seed": cfg["rng_seed"],
                  "root_created": "once in main(); children passed in (design 5.2)",
                  "children": {"0": "PERM-STRUCT", "1": "PERM-STRUCT-COND",
                               "2": "SHUFFLE-POP", "3": "RANDOM-POP",
                               "4": "item bootstrap (D-FELDMAN, then UNSTABLE-POP: "
                                    "one declared role, one stream, fixed order)",
                               "5": "reserved (UNUSED)"}}

    # ---------- shared machinery -------------------------------------------------
    def memb_and_y(tname, f):
        """A^(f) membership and the fold-f fitting target (§4.3(b), §5.2)."""
        okc = np.ones(n, dtype=bool) if tname == "tau_0" else (c_i >= thr[f])
        pos = P0 & okc & keep
        neg = ok & okc & keep
        return np.tile(pos | neg, len(seeds)), np.tile(pos, len(seeds)).astype(int)

    def oof(tname, Xmat, fitfn, score_all=False, pos_mask=None):
        """OOF scores under the per-scoring-fold analysis set A^(f) (§5.2).

        pos_mask, when given, REPLACES the positive class (a per-item boolean over
        the pool) while leaving A^(f) membership unchanged -- this is what
        SHUFFLE-POP and RANDOM-POP need, and both REFIT (§6.3).
        """
        sc = np.full(len(item), np.nan)
        yy = np.full(len(item), -1, dtype=int)
        inA = np.zeros(len(item), dtype=bool)
        audit = {"per_fold": {}, "rows_checked": 0}
        for f in range(cfg["k_folds"]):
            memb, y_f = memb_and_y(tname, f)
            if pos_mask is not None:
                y_f = (np.tile(pos_mask, len(seeds)) & memb).astype(int)
            tr = memb & (rfold != f)
            te = (rfold == f) & (memb if not score_all else rkeep)
            sel = (rfold == f) & memb
            inA |= sel
            yy[sel] = y_f[sel]
            audit["rows_checked"] += int(sel.sum())
            ran = not (tr.sum() == 0 or te.sum() == 0 or len(np.unique(y_f[tr])) < 2)
            audit["per_fold"][str(f)] = {
                "fit_ran": bool(ran), "n_train_rows": int(tr.sum()),
                "n_scored_rows": int(sel.sum()),
                "n_pos_train": int((y_f[tr] == 1).sum()),
                "n_neg_train": int((y_f[tr] == 0).sum()),
                "n_pos_scored": int((y_f[sel] == 1).sum()),
                "n_neg_scored": int((y_f[sel] == 0).sum()),
                "per_seed": {str(seeds[si]): {
                    "n_pos_scored": int((y_f[sel & (cellid == si)] == 1).sum()),
                    "n_neg_scored": int((y_f[sel & (cellid == si)] == 0).sum()),
                    "n_pos_train": int((y_f[tr & (cellid == si)] == 1).sum()),
                    "n_neg_train": int((y_f[tr & (cellid == si)] == 0).sum())}
                    for si in range(len(seeds))}}
            if not ran:
                continue
            sc[te] = fitfn(Xmat[tr], y_f[tr], Xmat[te], cfg)
        audit["all_folds_fit"] = bool(all(v["fit_ran"] for v in audit["per_fold"].values()))
        audit["items_checked"] = int(len(np.unique(item[inA])))
        return sc, yy, inA, audit

    res_tau = {}
    for tname in ("tau_0", "tau_hi"):
        Pt = Ptau[tname]
        blk = {"n_P_tau": int(Pt.sum()),
               "P_tau_indices": sorted(map(int, np.flatnonzero(Pt)))}
        if flags:
            dd = {}
            for nm, fl_ in flags.items():
                fl_ = np.asarray(fl_)
                base = float(fl_[keep].mean())
                inP = float(fl_[Pt].mean()) if Pt.sum() else None
                dd[nm] = {"n_flagged_in_arena": int(fl_[keep].sum()),
                          "arena_base_rate": round(base, 6),
                          "rate_in_P_tau": None if inP is None else round(inP, 6),
                          "enrichment_P_tau": (None if (inP is None or base == 0)
                                               else round(inP / base, 4))}
            blk["DATA_DEFECT_OVERLAP_population"] = dd

        # ---- O1 (§5.1)
        o1 = {"per_seed": {}}
        for s in seeds:
            c = cells[s]
            p2 = c["pred"].copy()
            p2[Pt] = 1 - p2[Pt]
            m = keep & (c["pred"] >= 0)
            o1["per_seed"][str(s)] = {
                "d_acc": round(acc(lab[m], p2[m]) - floors[str(s)]["acc"], 10),
                "d_mF1": round(mf1(lab[m], p2[m]) - floors[str(s)]["mF1"], 10)}
        o1["d_acc"] = round(float(np.mean([v["d_acc"] for v in o1["per_seed"].values()])), 10)
        o1["d_mF1"] = round(float(np.mean([v["d_mF1"] for v in o1["per_seed"].values()])), 10)
        o1["K_REACH_clears"] = bool(o1["d_acc"] >= cfg["reach_bar"]
                                    and o1["d_mF1"] >= cfg["reach_bar"])
        blk["O1"] = o1

        # ---- D-FELDMAN
        scF, y, inA, aud_nest = oof(tname, XF, fit_lr)
        scB, _, _, aud_B = oof(tname, XB, fit_lr)
        scA, _, _, aud_A = oof(tname, XF, fit_lr, score_all=True)
        scG, _, _, aud_G = oof(tname, XF, fit_gbm)
        scGB, _, _, aud_GB = oof(tname, XB, fit_gbm)
        # independent re-derivation of the analysis set's item count (§5.2), NOT
        # taken from oof's own bookkeeping: item i (in fold f_i) belongs to A^(f_i)
        # iff it is a stable inversion or always-correct, survives `keep`, and
        # clears its own fold's tau.
        okc_i = np.ones(n, dtype=bool) if tname == "tau_0" else (c_i >= own_thr)
        exp_items = int(((P0 | ok) & okc_i & keep).sum())
        blk["GATE_NESTED"] = {
            "items_checked": int(aud_nest["items_checked"]),
            "expected_items_independently_derived": exp_items,
            "rows_checked": int(aud_nest["rows_checked"]),
            "expected_rows": exp_items * len(seeds),
            "per_fold_fit_status": aud_nest["per_fold"],
            "all_folds_fit": bool(aud_nest["all_folds_fit"]),
            # every fitted arm is audited, not only FULL/LR: a BASE fold that failed
            # silently would drop rows from the paired dAUC with no gate firing
            "all_folds_fit_per_arm": {
                "FULL_lr": bool(aud_nest["all_folds_fit"]),
                "BASE_lr": bool(aud_B["all_folds_fit"]),
                "FULL_lr_score_all": bool(aud_A["all_folds_fit"]),
                "FULL_gbm": bool(aud_G["all_folds_fit"]),
                "BASE_gbm": bool(aud_GB["all_folds_fit"])},
            "fit_failure_causes": {
                arm: [("fold" + f, "single_class_train" if v["n_pos_train"] == 0
                       or v["n_neg_train"] == 0 else "empty_train_or_test")
                      for f, v in a["per_fold"].items() if not v["fit_ran"]]
                for arm, a in (("FULL_lr", aud_nest), ("BASE_lr", aud_B),
                               ("FULL_lr_score_all", aud_A), ("FULL_gbm", aud_G),
                               ("BASE_gbm", aud_GB))},
            "note_on_halt_routing":
                "a fold that cannot be fitted fails this gate and therefore HALTs. "
                "That is deliberate and conservative: it can never yield a false "
                "CONTINUE. fit_failure_causes names the cause so a HALT is "
                "immediately diagnosable and a data-caused failure (single-class "
                "training fold) is distinguishable from an instrument failure.",
            "row_loss_note":
                "rows_checked counts A^(f) membership; n_rows_entering_paired_dAUC "
                "counts rows that actually carried finite FULL and BASE scores. A "
                "single NaN would silently drop an item's three rows from dAUC, from "
                "both nulls and from the bootstrap with every other gate still "
                "passing, so the equality is GATED, not merely emitted.",
            "pass": bool(aud_nest["items_checked"] == exp_items
                         and aud_nest["rows_checked"] == exp_items * len(seeds)
                         and aud_nest["all_folds_fit"]
                         and aud_B["all_folds_fit"] and aud_A["all_folds_fit"]
                         and aud_G["all_folds_fit"] and aud_GB["all_folds_fit"])}
        m = inA & np.isfinite(scF) & np.isfinite(scB)
        # the paired-row conjunct, evaluated now that m exists: a single NaN would
        # drop an item's three rows from dAUC, from both nulls and from the bootstrap
        # with every other gate still passing.
        blk["GATE_NESTED"]["n_rows_entering_paired_dAUC"] = int(m.sum())
        blk["GATE_NESTED"]["rows_lost_to_nonfinite_scores"] = \
            int(inA.sum()) - int(m.sum())
        blk["GATE_NESTED"]["pass"] = bool(
            blk["GATE_NESTED"]["pass"]
            and int(m.sum()) == exp_items * len(seeds))
        aF, dropF = pooled_auc(scF[m], y[m], strat[m], cellid[m])
        aB, dropB = pooled_auc(scB[m], y[m], strat[m], cellid[m])
        aG, _ = pooled_auc(scG[m], y[m], strat[m], cellid[m]) if np.isfinite(scG[m]).all() \
            else (None, [])
        aGB, _ = pooled_auc(scGB[m], y[m], strat[m], cellid[m]) \
            if np.isfinite(scGB[m]).all() else (None, [])
        dauc = None if (aF is None or aB is None) else aF - aB

        occ, pw_l, nw_l = [], [], []
        for c in np.unique(cellid[m]):
            mm = m & (cellid == c)
            a, den, nw_, pw_, per = strat_auc_cell(scF[mm], y[mm], strat[mm])
            occ.append({"cell": int(c), "n_weighted_strata": int(nw_),
                        "sum_pos_times_neg": int(den), "p_w": int(pw_),
                        "n_pos": int((y[mm] == 1).sum()), "n_neg": int((y[mm] == 0).sum()),
                        "per_stratum": per})
            pw_l.append(pw_)
            nw_l.append(nw_)
        pw_mean = float(np.mean(pw_l)) if pw_l else 0.0
        nw_mean = float(np.mean(nw_l)) if nw_l else 0.0
        dead = int(Pt.sum()) < cfg["pw_min"]
        under = bool(pw_mean < cfg["pw_min"] or nw_mean < cfg["min_weighted_strata"]
                     or len(dropF) > 0 or dead)
        blk["STRATUM_OCCUPANCY"] = {
            "per_cell": occ, "seed_mean_p_w": round(pw_mean, 4),
            "seed_mean_weighted_strata": round(nw_mean, 4),
            "cells_dropped_all_single_class": dropF,
            "cap_p_w_le_P_tau": int(Pt.sum()),
            "cell_marking": "ARITHMETICALLY_DEAD_AT_THIS_POWER" if dead else "LIVE",
            "IDENTIFIABILITY_UNDERPOWERED": under}
        blk["D_FELDMAN"] = {
            "AUC_strat_FULL": None if aF is None else round(aF, 8),
            "AUC_strat_BASE": None if aB is None else round(aB, 8),
            "dAUC": None if dauc is None else round(dauc, 8),
            "gbm_capacity_check_NO_RULE_READS_IT": {
                "AUC_strat_FULL_gbm": None if aG is None else round(aG, 8),
                "AUC_strat_BASE_gbm": None if aGB is None else round(aGB, 8),
                "dAUC_gbm": (None if (aG is None or aGB is None)
                             else round(aG - aGB, 8)),
                "protocol": "the same two-family shape as the LR arm (F47), so the "
                            "diagnostic the design budgeted is actually readable; "
                            "DET-4 Tier-C -- read by no decision rule"},
            "n_rows_scored": int(m.sum()), "n_pos": int((y[m] == 1).sum()),
            "n_neg": int((y[m] == 0).sum()),
            "per_cell_class_counts": [{"cell": o["cell"], "n_pos": o["n_pos"],
                                       "n_neg": o["n_neg"]} for o in occ]}

        # ---- permutation nulls
        pool = np.zeros(n, dtype=bool)
        for f in range(cfg["k_folds"]):
            mm, _ = memb_and_y(tname, f)
            pool |= mm[:n]
        pool_items = np.flatnonzero(pool)
        slot = np.full(n, -1, dtype=int)
        slot[pool_items] = np.arange(len(pool_items))
        sidx = [FULL_FEATS.index(f) for f in STRUCT_FEATS]
        groups = ([np.arange(len(pool_items))] if len(pool_items) else [])
        cgroups = [np.flatnonzero(istrat[pool_items] == g)
                   for g in np.unique(istrat[pool_items])] if len(pool_items) else []

        def perm_null(D, rng, grps):
            got = []
            if not len(pool_items) or dauc is None:
                return np.asarray(got, dtype="float64")
            for _ in range(D):
                perm = np.arange(len(pool_items))
                for g in grps:
                    if len(g) > 1:
                        perm[g] = g[rng.permutation(len(g))]
                donor = pool_items[perm]
                Xp = XF.copy()
                for si in range(len(seeds)):
                    rows = np.flatnonzero((cellid == si) & np.tile(pool, len(seeds)))
                    src = donor[slot[item[rows]]] + si * n
                    Xp[np.ix_(rows, sidx)] = XF[np.ix_(src, sidx)]
                sp, yp, inAp, _ = oof(tname, Xp, fit_lr)
                mp = inAp & np.isfinite(sp) & np.isfinite(scB)
                ap, _ = pooled_auc(sp[mp], yp[mp], strat[mp], cellid[mp])
                ab, _ = pooled_auc(scB[mp], yp[mp], strat[mp], cellid[mp])
                got.append(np.nan if (ap is None or ab is None) else ap - ab)
            return np.asarray(got, dtype="float64")

        for nm, cid, grps in (("PERM_STRUCT", 0, groups),
                              ("PERM_STRUCT_COND", 1, cgroups)):
            dd = perm_null(cfg["D_perm"], ch[cid], grps)
            fin = dd[np.isfinite(dd)]
            p = ((1.0 + int((fin >= (dauc if dauc is not None else np.inf)).sum()))
                 / (len(fin) + 1.0)) if len(fin) else None
            blk[nm] = {"D_effective": int(len(fin)),
                       "p": None if p is None else round(float(p), 8),
                       "null_mean": round(float(fin.mean()), 8) if len(fin) else None,
                       "null_sd": round(float(fin.std()), 8) if len(fin) else None}

        # ---- item bootstrap (fit-conditional, interval only)
        its = np.unique(item[m])
        rows_by = {int(i): np.flatnonzero(m & (item == i)) for i in its}
        keys = list(map(int, its))
        bs = []
        rng4 = ch[4]
        for _ in range(cfg["B_boot"] if len(keys) else 0):
            pick = rng4.integers(0, len(keys), size=len(keys))
            rows = np.concatenate([rows_by[keys[j]] for j in pick])
            a1, _ = pooled_auc(scF[rows], y[rows], strat[rows], cellid[rows])
            a2, _ = pooled_auc(scB[rows], y[rows], strat[rows], cellid[rows])
            if a1 is not None and a2 is not None:
                bs.append(a1 - a2)
        bs = np.asarray(bs)
        blk["item_bootstrap_fit_conditional"] = {
            "B_effective": int(len(bs)),
            "lower_95_one_sided": round(float(np.quantile(bs, 0.05)), 8) if len(bs) else None,
            "ci95": [round(float(np.quantile(bs, 0.025)), 8),
                     round(float(np.quantile(bs, 0.975)), 8)] if len(bs) else None}

        # ---- SHUFFLE-POP (target permutation over the pool)
        aucs, dl = [], []
        pos_real = (Pt if tname == "tau_hi" else P0)
        for _ in range(cfg["D_shuffle"] if (len(pool_items) and m.sum()) else 0):
            pm = ch[2].permutation(len(pool_items))
            newlab = np.zeros(n, dtype=bool)
            newlab[pool_items] = pos_real[pool_items][pm]
            s1, y1, inA1, _ = oof(tname, XF, fit_lr, pos_mask=newlab)
            s2, _, _, _ = oof(tname, XB, fit_lr, pos_mask=newlab)
            mm = inA1 & np.isfinite(s1) & np.isfinite(s2)
            a1, _ = pooled_auc(s1[mm], y1[mm], strat[mm], cellid[mm])
            a2, _ = pooled_auc(s2[mm], y1[mm], strat[mm], cellid[mm])
            if a1 is not None:
                aucs.append(a1)
            if a1 is not None and a2 is not None:
                dl.append(a1 - a2)
        sm = float(np.mean(aucs)) if aucs else None
        blk["SHUFFLE_POP"] = {
            "D": len(aucs),
            "null_mean_AUC_strat_FULL": None if sm is None else round(sm, 8),
            "band": cfg["shuffle_band"],
            "in_band": bool(sm is not None and cfg["shuffle_band"][0] <= sm
                            <= cfg["shuffle_band"][1]),
            "secondary_joint_null_dAUC_mean": round(float(np.mean(dl)), 8) if dl else None,
            "secondary_joint_null_ASL": (
                None if (not dl or dauc is None)
                else round((1.0 + sum(1 for x in dl if x >= dauc)) / (len(dl) + 1.0), 8)),
            "secondary_joint_null_ASL_definition":
                "(1 + #{draw dAUC >= observed dAUC}) / (D + 1) over the SHUFFLE-POP "
                "draws -- the third reading 5.2 bullet 3 and 6.3 put on the record. "
                "The mean above is a location summary, not an ASL; both are emitted.",
            "n_draws_ge_observed": (None if (not dl or dauc is None)
                                    else int(sum(1 for x in dl if x >= dauc)))}

        # ---- RANDOM-POP (size-matched random target)
        # RANDOM-POP: "a size-matched random sample of query items in place of the
        # stable inversions" (§6.3).  IMPLEMENTATION READING, recorded in the freeze
        # record (refine-logs/C09_A0_RECORD.md): the sample is drawn from the
        # analysis pool P^(tau) = union_f A^(f),
        # which is the only self-consistent domain -- drawing from all n would put
        # unstable errors (in neither class) and always-correct items (already the
        # negatives) into the positive class at once.  Positives = sampled, negatives
        # = pool minus sampled.  Both feature sets REFIT, as the budget line requires.
        rp = []
        npos = int(Pt.sum())
        for _ in range(cfg["D_random"] if (npos and len(pool_items)) else 0):
            sel = ch[3].choice(pool_items, size=min(npos, len(pool_items)), replace=False)
            fake = np.zeros(n, dtype=bool)
            fake[sel] = True
            s1, y1, inA1, _ = oof(tname, XF, fit_lr, pos_mask=fake)
            s2, _, _, _ = oof(tname, XB, fit_lr, pos_mask=fake)
            mm = inA1 & np.isfinite(s1) & np.isfinite(s2)
            a1, _ = pooled_auc(s1[mm], y1[mm], strat[mm], cellid[mm])
            a2, _ = pooled_auc(s2[mm], y1[mm], strat[mm], cellid[mm])
            if a1 is not None and a2 is not None:
                rp.append(a1 - a2)
        blk["RANDOM_POP"] = {
            "D": len(rp),
            "mean_dAUC": round(float(np.mean(rp)), 8) if rp else None,
            "sd_dAUC": round(float(np.std(rp)), 8) if rp else None,
            "sampling_domain": "the analysis pool P^(tau) = union_f A^(f); positives "
                               "= sampled, negatives = pool minus sampled",
            "SCOPE_NARROWING_DECLARED":
                "6.3's prose says 'every reported quantity recomputed against it'. "
                "Only dAUC is recomputed here. O1/NET/K-DEG against a random "
                "population are deterministic functions of that sample's error "
                "content -- d_acc_s = (2|S and wrong_s| - k)/n, which at this "
                "arena's measured per-seed error rate (0.1054-0.1142) is about "
                "-0.78*k/n -- so they price the sample's label content, not "
                "identifiability. NOTE: they are NOT the |P_tau|/n identity of 5.1; "
                "that identity holds only because P_tau is wrong at EVERY seed, "
                "which a random sample is not. What F88-null(3) asks -- whether the "
                "structural signal is specific to the stable-inversion population -- "
                "is carried entirely by dAUC."}

        # ---- NET (§5.3)
        own_A = (P0 | ok) & okc_i & keep      # the item's OWN analysis set A^(fold(i))
        iscore = np.full(n, -np.inf)
        for i in range(n):
            v = scA[[i + si * n for si in range(len(seeds))]]
            v = v[np.isfinite(v)]
            if len(v):
                iscore[i] = float(np.mean(v))
        # GATE-NULL (2): a space that scores fewer items than the frozen n gets its
        # OWN recomputed bar (0.050*743 = 37.15 for the remove-null leg).  A full-n
        # space keeps the frozen bar verbatim -- the frozen 29.0 / 22.3 / 17.4 are
        # rounded from 28.95 / 22.32 / 17.37 and must NOT be silently re-derived.
        bar_frozen = cfg["net_bar"][ds]
        if nk == cfg["n_items"][ds]:
            bar, bar_030 = bar_frozen, cfg["net_bar_030"][ds]
            bar_src = "frozen (full n)"
        else:
            bar = round(cfg["reach_bar"] * nk, 4)
            bar_030 = round(cfg["net_bar_030_rate"] * nk, 4)
            bar_src = "recomputed for n_scored={} (GATE-NULL (2))".format(nk)
        cap_hi = 2.0 * mean_wrong - bar
        ks, seen = [], set()
        for mult, kk in ((1.0, int(Pt.sum())), (1.5, R(1.5 * Pt.sum())),
                         (2.0, R(2.0 * Pt.sum()))):
            if kk > 0 and kk not in seen:
                seen.add(kk)
                ks.append((mult, kk))
        cells_net = []
        order = np.argsort(-iscore, kind="mergesort")
        order = order[keep[order]]
        for mult, kk in ks:
            if kk > len(order):
                continue
            S = np.zeros(n, dtype=bool)
            S[order[:kk]] = True
            per, self_ok = {}, True
            for s in seeds:
                c = cells[s]
                w = int((S & inv[s]).sum())
                net_s = 2 * w - int(S.sum())
                p2 = c["pred"].copy()
                p2[S] = 1 - p2[S]
                mm = keep & (c["pred"] >= 0)
                dacc = acc(lab[mm], p2[mm]) - floors[str(s)]["acc"]
                good = abs(net_s - floors[str(s)]["n"] * dacc) < 1e-6
                self_ok = self_ok and good
                per[str(s)] = {"net": net_s, "d_acc": round(dacc, 12),
                               "d_mF1": round(mf1(lab[mm], p2[mm]) - floors[str(s)]["mF1"], 10),
                               "selftest_net_eq_n_dacc": bool(good)}
            netm = float(np.mean([v["net"] for v in per.values()]))
            mfm = float(np.mean([v["d_mF1"] for v in per.values()]))
            live = bool(bar <= kk <= cap_hi)
            prec = float(np.mean([(v["net"] + kk) / 2.0 / kk for v in per.values()]))
            cells_net.append({
                "k": kk, "multiplier": mult, "live_on_net": live,
                "cell_marking": "LIVE_ON_NET" if live else "ARITHMETICALLY_DEAD_ON_NET",
                "per_seed": per, "mean_net": round(netm, 6),
                "min_net": int(min(v["net"] for v in per.values())),
                "mean_d_mF1": round(mfm, 10),
                "bar_used": bar, "bar_frozen_full_n": bar_frozen, "bar_source": bar_src,
                "pi_star_required_precision": round((1.0 + bar / kk) / 2.0, 6),
                "realised_precision_mean": round(prec, 6),
                "exchange_rate_diagnostic_reads_no_rule": round(
                    float(np.mean([((v["net"] + kk) / 2.0)
                                   / max(1e-9, kk - (v["net"] + kk) / 2.0)
                                   for v in per.values()])), 4),
                "composition_of_S": {
                    "stable_inversion": int((S & P0).sum()),
                    "unstable_error": int((S & unstable).sum()),
                    "always_correct": int((S & ok).sum()),
                    # 5.3 asks for the FRACTION of S outside the support the model was
                    # fitted on, and the support of item i is its OWN A^(fold(i)), not
                    # the union pool (which uses min_f tau_hi^(f) and is a strict
                    # superset at tau_hi).  Both are emitted; the fraction is the one
                    # 5.3 names.
                    "fraction_outside_own_A_f":
                        round(float((S & ~own_A & keep).sum()) / float(kk), 6),
                    "n_outside_own_A_f": int((S & ~own_A & keep).sum()),
                    "n_outside_union_pool_descriptive": int((S & ~pool & keep).sum())},
                "S_indices": sorted(map(int, np.flatnonzero(S))),
                "DATA_DEFECT_OVERLAP_S": ({
                    nm: {"rate_in_S": round(float(np.asarray(fl_)[S].mean()), 6),
                         "enrichment_S": (None if float(np.asarray(fl_)[keep].mean()) == 0
                                          else round(float(np.asarray(fl_)[S].mean())
                                                     / float(np.asarray(fl_)[keep].mean()), 4))}
                    for nm, fl_ in flags.items()} if flags else None),
                "GATE_SELFTEST_pass": bool(self_ok),
                "K_NET_primary_clears": bool(live and netm >= bar
                                             and mfm >= cfg["reach_bar"]),
                "secondary_030": {
                    "bar": bar_030, "frozen_full_n_bar": cfg["net_bar_030"][ds],
                    "mean_net_vs_bar": round(netm - bar_030, 6),
                    "clears": bool(netm >= bar_030 and mfm >= cfg["net_mf1_030"])},
                "K_DEG": k_deg_block(S, kk, cells, seeds, keep, c_i, inv, n, cfg)})
        blk["NET_k_grid_requested"] = [{"multiplier": m_, "k": k_} for m_, k_ in ks]
        blk["NET"] = {"mean_wrong_per_seed": round(mean_wrong, 6),
                      "cap_low": bar, "cap_high": round(cap_hi, 6),
                      "bar_primary": bar, "bar_secondary_030": bar_030,
                      "bar_source": bar_src, "bar_primary_frozen_full_n": bar_frozen,
                      "cells": cells_net}

        # ---- GATE-ZEROOP (empty operator).  Runs the SAME treatment path as a NET
        # cell -- copy, flip S, re-score -- with S = empty and k = 0, so it exercises
        # the operator that GATE-FLOOR does not (design 8.1).
        z = {}
        S0 = np.zeros(n, dtype=bool)
        k0 = int(S0.sum())
        for s in seeds:
            c = cells[s]
            p2 = c["pred"].copy()
            p2[S0] = 1 - p2[S0]
            mm = keep & (c["pred"] >= 0)
            net0 = 2 * int((S0 & inv[s]).sum()) - k0
            z[str(s)] = {"d_acc": round(acc(lab[mm], p2[mm]) - floors[str(s)]["acc"], 12),
                         "d_mF1": round(mf1(lab[mm], p2[mm]) - floors[str(s)]["mF1"], 12),
                         "net": int(net0),
                         "path": "pred.copy(); pred[S]=1-pred[S]; k=0"}
        blk["GATE_ZEROOP"] = {"per_seed": z, "k": k0,
                              "exercises_treatment_path": True,
                              "pass": all(abs(v["d_acc"]) < 1e-12
                                          and abs(v["d_mF1"]) < 1e-12
                                          and v["net"] == 0 for v in z.values())}
        res_tau[tname] = blk

    out["tau"] = res_tau

    # ---- UNSTABLE-POP (tau_0 shape, target = unstable errors)
    if int(unstable.sum()) > 0 and tag != "raw":
        sc = np.full(len(item), np.nan)
        scb = np.full(len(item), np.nan)
        yu = np.tile(unstable, len(seeds)).astype(int)
        memb = np.tile(unstable | ok, len(seeds))
        for f in range(cfg["k_folds"]):
            tr = memb & (rfold != f)
            te = memb & (rfold == f)
            if tr.sum() and te.sum() and len(np.unique(yu[tr])) > 1:
                sc[te] = fit_lr(XF[tr], yu[tr], XF[te], cfg)
                scb[te] = fit_lr(XB[tr], yu[tr], XB[te], cfg)
        mu = memb & np.isfinite(sc) & np.isfinite(scb)
        a1, _ = pooled_auc(sc[mu], yu[mu], strat[mu], cellid[mu])
        a2, _ = pooled_auc(scb[mu], yu[mu], strat[mu], cellid[mu])
        du = None if (a1 is None or a2 is None) else a1 - a2
        its = np.unique(item[mu])
        rb = {int(i): np.flatnonzero(mu & (item == i)) for i in its}
        kk2 = list(map(int, its))
        bb = []
        for _ in range(cfg["B_boot_unstable"] if len(kk2) else 0):
            # child 4 = "item bootstrap" (frozen role).  UNSTABLE-POP's resample IS
            # an item bootstrap, so it belongs to child 4; child 5 stays RESERVED
            # (UNUSED) exactly as 5.2 pins it.  The two bootstraps share one stream
            # and are therefore drawn in a fixed, reproducible order.
            pick = ch[4].integers(0, len(kk2), size=len(kk2))
            rows = np.concatenate([rb[kk2[j]] for j in pick])
            x1, _ = pooled_auc(sc[rows], yu[rows], strat[rows], cellid[rows])
            x2, _ = pooled_auc(scb[rows], yu[rows], strat[rows], cellid[rows])
            if x1 is not None and x2 is not None:
                bb.append(x1 - x2)
        width = (float(np.quantile(bb, 0.975) - np.quantile(bb, 0.025)) if bb else None)
        out["UNSTABLE_POP"] = {
            "n_unstable": int(unstable.sum()),
            "dAUC": None if du is None else round(du, 8),
            "ci_width": None if width is None else round(width, 6),
            "CONTROL_UNDERPOWERED": bool(int(unstable.sum()) < cfg["unstable_min"]
                                         or width is None
                                         or width > cfg["unstable_ci_width_max"]),
            "non_gating": True}
    else:
        out["UNSTABLE_POP"] = {"n_unstable": int(unstable.sum()),
                               "CONTROL_UNDERPOWERED": True,
                               "note": "not computable in the seed-free raw space (§9)"
                                       if tag == "raw" else "empty population",
                               "non_gating": True}

    out["secs"] = round(time.time() - t0, 1)
    log("  [{} {}] done in {:.1f}s  P0={} P_tauhi={}".format(
        tag, ds, out["secs"], int(P0.sum()), int(Ptau["tau_hi"].sum())))
    return out


# ------------------------------------------------------------------------- verdict
def holm2(pa, pb, alpha):
    """Holm over two p-values; returns (reject_a, reject_b).

    An undefined p (its dAUC was not computable -- e.g. every stratum in that tau's
    cells was single-class) is treated as p = 1.0: it cannot reject ITSELF, and it
    must not block its partner.  Substituting 1.0 is exact for the partner, since
    Holm's step-down compares min(p) to alpha/2 and then max(p) to alpha, and a
    p of 1.0 fails the second step on its own account.  Returning (False, False)
    instead would let an uncomputable tau_hi veto a rejecting tau_0.
    """
    pa = 1.0 if pa is None else pa
    pb = 1.0 if pb is None else pb
    lo, hi = (pa, pb) if pa <= pb else (pb, pa)
    r_lo = lo <= alpha / 2.0
    r_hi = r_lo and (hi <= alpha)
    if pa <= pb:
        return bool(r_lo), bool(r_hi)
    return bool(r_hi), bool(r_lo)


def holm_family(tau_blocks, alpha):
    """Holm over the two tau, WITHIN dataset and family (design §5.2/§6.4 text)."""
    out = {}
    for fam in ("PERM_STRUCT", "PERM_STRUCT_COND"):
        r0, rh = holm2(tau_blocks["tau_0"][fam]["p"], tau_blocks["tau_hi"][fam]["p"], alpha)
        out[fam] = {"tau_0": {"p": tau_blocks["tau_0"][fam]["p"], "reject": r0},
                    "tau_hi": {"p": tau_blocks["tau_hi"][fam]["p"], "reject": rh}}
    return out


def feldman_clears(tau_blocks, alpha):
    """K-FELDMAN per tau, as 9(2) defines it: Holm rejection in BOTH families, and
    identifiability neither underpowered nor arithmetically dead.  One definition,
    used by adjudicate() and by GATE-NULL (3) alike, so the two cannot drift."""
    h = holm_family(tau_blocks, alpha)
    cl = {}
    for t in ("tau_0", "tau_hi"):
        occ = tau_blocks[t]["STRATUM_OCCUPANCY"]
        cl[t] = bool(h["PERM_STRUCT"][t]["reject"] and h["PERM_STRUCT_COND"][t]["reject"]
                     and not occ["IDENTIFIABILITY_UNDERPOWERED"]
                     and occ["cell_marking"] == "LIVE")
    return cl, h


def k_rule_fingerprint(results, cfg, space):
    """Every K-rule outcome, extracted from one space, for GATE-NULL (3)."""
    dss = cfg["datasets"]
    fp = {}
    for ds in dss:
        sp = space if space in results[ds] else "head"
        blocks = results[ds][sp]["tau"]
        cl, _h = feldman_clears(blocks, cfg["alpha"])
        for tname in ("tau_0", "tau_hi"):
            t = blocks[tname]
            fp.setdefault(tname, {})[ds] = {
                "K_REACH_clears": bool(t["O1"]["K_REACH_clears"]),
                "K_FELDMAN_clears": bool(cl[tname]),
                "PERM_STRUCT_p": t["PERM_STRUCT"]["p"],
                "PERM_STRUCT_COND_p": t["PERM_STRUCT_COND"]["p"],
                "IDENTIFIABILITY_UNDERPOWERED":
                    bool(t["STRATUM_OCCUPANCY"]["IDENTIFIABILITY_UNDERPOWERED"]),
                "identifiability_cell_marking":
                    t["STRATUM_OCCUPANCY"]["cell_marking"],
                "K_NET_primary_clears_any_cell":
                    bool(any(c["K_NET_primary_clears"] for c in t["NET"]["cells"])),
                "K_DEG_fires_any_cell":
                    bool(any(c["K_DEG"]["K_DEG_fires"] for c in t["NET"]["cells"])),
                # per-multiplier, because 9's CONTINUE needs the SAME multiplier cell
                # to clear on both datasets -- "any cell" does not capture that
                "per_multiplier": {str(c["multiplier"]): {
                    "K_NET_primary_clears": bool(c["K_NET_primary_clears"]),
                    "K_DEG_fires": bool(c["K_DEG"]["K_DEG_fires"]),
                    "live_on_net": bool(c["live_on_net"])}
                    for c in t["NET"]["cells"]}}
    return fp


def gate_null_agreement(results, cfg, V):
    """GATE-NULL (3), design 8.2: the remove-null leg must agree with the primary on
    the VERDICT and on EVERY K-rule outcome.  Any disagreement is published as a
    first-class finding and the verdict is scoped to it."""
    ds0 = cfg["null_row_dataset"]
    if "head_remove_null" not in results.get(ds0, {}):
        return {"ran": False, "agrees": None,
                "note": "remove-null leg absent (no structural-zero row)"}
    prim = k_rule_fingerprint(results, cfg, "head")
    sub = k_rule_fingerprint({ds0: {"head": results[ds0]["head_remove_null"]}},
                             {"datasets": [ds0], "alpha": cfg["alpha"]}, "head")
    # The comparison is on the K-RULE OUTCOMES themselves (9(2)), not on a p <= alpha
    # proxy: Holm over the two tau is min(p) <= alpha/2 then max(p) <= alpha, which is
    # strictly stronger, so two legs can share every "p <= alpha" and still disagree
    # on K-FELDMAN.  Raw p-values are kept as diagnostics only.
    RULES = ("K_REACH_clears", "K_FELDMAN_clears", "IDENTIFIABILITY_UNDERPOWERED",
             "identifiability_cell_marking", "K_NET_primary_clears_any_cell",
             "K_DEG_fires_any_cell")
    disagree = []
    for tname in ("tau_0", "tau_hi"):
        a, b = prim[tname][ds0], sub[tname][ds0]
        for k in RULES:
            if a[k] != b[k]:
                disagree.append({"tau": tname, "rule": k,
                                 "primary": a[k], "remove_null": b[k]})
        for mult in sorted(set(a["per_multiplier"]) | set(b["per_multiplier"])):
            pa = a["per_multiplier"].get(mult)
            pb = b["per_multiplier"].get(mult)
            if pa != pb:
                disagree.append({"tau": tname, "rule": "cell@" + mult,
                                 "primary": pa, "remove_null": pb})
    # ---- and the VERDICT itself (8.2(3) names it first).  adjudicate is re-run with
    # the remove-null leg substituted for the head leg of the null-row dataset; the
    # inner call is told not to recurse.
    sub_results = {d: (dict(results[d]) if not d.startswith("_") else results[d])
                   for d in results}
    sub_results[ds0] = dict(results[ds0])
    # GATE-FLOOR, GATE-PARITY-FOLD and the top-level GATE-NESTED are properties of the
    # 36 mints and the frozen fold partition, not of which items the analysis keeps, so
    # they are inherited verbatim from the primary head; everything else comes from the
    # remove-null leg.  Inheriting them is stated here rather than silently swallowed.
    subhead = dict(results[ds0]["head_remove_null"])
    for g in ("GATE_FLOOR", "GATE_PARITY_FOLD", "GATE_NESTED", "mint_meta_vs_filename"):
        if g in results[ds0]["head"]:
            subhead[g] = results[ds0]["head"][g]
    subhead["INHERITED_FROM_PRIMARY_HEAD"] = [
        "GATE_FLOOR", "GATE_PARITY_FOLD", "GATE_NESTED (top level)"]
    sub_results[ds0]["head"] = subhead
    V_sub = adjudicate(sub_results, cfg, _compare_null=False)
    v_sub = V_sub["verdict"]
    cell_sub = (V_sub.get("continue_scope") or {}).get("cells")
    v_prim = V.get("verdict_provisional")
    halted = (v_sub == "HALT_NO_VERDICT")
    halt_gates = []
    if halted:
        pp = V_sub.get("publication_precondition", {})
        for dsx, gs in (pp.get("per_dataset") or {}).items():
            halt_gates += ["{}/{}".format(dsx, g) for g, ok in gs.items() if not ok]
        for spx, gs in (pp.get("other_spaces_structural_only") or {}).items():
            halt_gates += ["{}/{}".format(spx, g) for g, ok in gs.items() if not ok]
        if not pp.get("GATE_LEDGER", True):
            halt_gates.append("GATE_LEDGER")
    elif v_prim is not None and v_sub != v_prim:
        disagree.append({"rule": "VERDICT", "primary": v_prim, "remove_null": v_sub})
    return {"ran": True, "dataset": ds0, "row_removed": cfg["null_row_index"],
            "rules_compared": list(RULES) + ["per-multiplier NET/DEG cells", "VERDICT"],
            "verdict_primary": v_prim, "verdict_remove_null": v_sub,
            "continue_cells_remove_null": cell_sub,
            "remove_null_leg_halted": bool(halted),
            "remove_null_leg_halt_gates": halt_gates,
            "halt_reading": ("the sensitivity leg HALTed on its own instrument gates; "
                             "per §9 a HALT is evidence neither for nor against C09, "
                             "so it is NOT recorded as a verdict disagreement and does "
                             "NOT scope the primary verdict. The failing gates are "
                             "named above." if halted else None),
            "primary": {t: prim[t][ds0] for t in prim},
            "remove_null": {t: sub[t][ds0] for t in sub},
            "disagreements": disagree, "agrees": bool(not disagree),
            "requirement": "agreement on the verdict and every K-rule outcome (8.2); "
                           "a disagreement is a first-class finding and scopes the "
                           "verdict"}


def adjudicate(results, cfg, _compare_null=True):
    dss = cfg["datasets"]
    V = {"publication_precondition": {}, "per_tau": {}, "verdict": None}
    halts = {}
    for ds in dss:
        h = results[ds]["head"]
        halts[ds] = {
            "GATE_FLOOR": h["GATE_FLOOR"]["pass"],
            "GATE_PARITY_FOLD": h["GATE_PARITY_FOLD"]["pass"],
            "GATE_FIXK20": h["GATE_FIXK20"]["pass"],
            "GATE_BLIND": h["GATE_BLIND"]["pass"],
            "GATE_NESTED": h["GATE_NESTED"]["pass"],
            "GATE_SELFTEST": all(c["GATE_SELFTEST_pass"]
                                 for t in h["tau"].values() for c in t["NET"]["cells"]),
            "GATE_ZEROOP": all(t["GATE_ZEROOP"]["pass"] for t in h["tau"].values()),
            "GATE_ARENA": h["GATE_ARENA"]["pass"],
            "SHUFFLE_POP_band": all(t["SHUFFLE_POP"]["in_band"] for t in h["tau"].values())}
    other_sp = {}
    for ds in dss:
        for sp in ("head_remove_null", "raw"):
            if sp not in results[ds]:
                continue
            h2 = results[ds][sp]
            other_sp["{}/{}".format(ds, sp)] = {
                "GATE_FIXK20": h2["GATE_FIXK20"]["pass"],
                "GATE_BLIND": h2["GATE_BLIND"]["pass"],
                "GATE_NESTED": all(t["GATE_NESTED"]["pass"] for t in h2["tau"].values()),
                "GATE_SELFTEST": all(c["GATE_SELFTEST_pass"] for t in h2["tau"].values()
                                     for c in t["NET"]["cells"]),
                "GATE_ZEROOP": all(t["GATE_ZEROOP"]["pass"] for t in h2["tau"].values())}
    V["publication_precondition"]["per_dataset"] = halts
    V["publication_precondition"]["other_spaces_structural_only"] = other_sp
    V["publication_precondition"]["other_spaces_note"] = (
        "DIAGNOSTIC, NOT A GATE. 8.1 declares its nine gates plus the SHUFFLE-POP "
        "band 'the complete publication precondition of 9', so this file does not "
        "add HALT conditions to that frozen list: these five arithmetic/structural "
        "checks are computed and published for the remove-null and raw legs but do "
        "NOT enter ok_pub. 9 confines the raw leg to KILL corroboration and lets no "
        "raw number reach the decision, and 8.2 makes the remove-null leg a "
        "sensitivity whose disagreement is already published as a first-class "
        "finding by GATE_NULL_AGREEMENT -- so a failure here cannot corrupt a "
        "verdict, while gating on it could void an otherwise clean one for a data "
        "property of a non-decisional leg.")
    V["publication_precondition"]["other_spaces_all_pass"] = bool(
        all(all(v.values()) for v in other_sp.values()))
    V["publication_precondition"]["GATE_LEDGER"] = results["_ledger"]["pass"]
    ok_pub = (all(all(v.values()) for v in halts.values())
              and results["_ledger"]["pass"])
    V["publication_precondition"]["publishes_a_verdict"] = bool(ok_pub)
    if not ok_pub:
        V["verdict_provisional"] = "HALT_NO_VERDICT"
        V["GATE_NULL_AGREEMENT"] = (
            gate_null_agreement(results, cfg, V) if _compare_null
            else {"ran": False, "agrees": None, "note": "inner (remove-null) call"})
        V["verdict"] = "HALT_NO_VERDICT"
        V["halt_note"] = ("A HALT publishes no verdict: engineering result, consumes no "
                          "scientific gate, evidence neither for nor against C09 (§9).")
        return V

    # Holm within dataset and family, over the two tau
    holm, feldc = {}, {}
    for ds in dss:
        feldc[ds], holm[ds] = feldman_clears(results[ds]["head"]["tau"], cfg["alpha"])
    V["holm"] = holm
    V["K_FELDMAN_per_dataset"] = feldc

    cont = None
    for tname in ("tau_0", "tau_hi"):
        tb = {ds: results[ds]["head"]["tau"][tname] for ds in dss}
        reach = all(tb[ds]["O1"]["K_REACH_clears"] for ds in dss)
        feld = all(feldc[ds][tname] for ds in dss)
        marg_only = all(holm[ds]["PERM_STRUCT"][tname]["reject"] for ds in dss) and \
            not all(holm[ds]["PERM_STRUCT_COND"][tname]["reject"] for ds in dss)
        # the (tau, k) cell is paired across datasets on the MULTIPLIER TAG (1x,
        # 1.5x, 2x), never on list position: k itself is a per-dataset function of
        # |P_tau| (§5.3, §9), and a dataset whose grid de-duplicated (possible when
        # |P_tau| is tiny) would otherwise silently shift the pairing.
        common = []
        by_mult = {ds: {c["multiplier"]: c for c in tb[ds]["NET"]["cells"]} for ds in dss}
        mults = sorted(set.intersection(*[set(by_mult[ds]) for ds in dss]))
        for mult in mults:
            per = {ds: by_mult[ds][mult] for ds in dss}
            net_ok = all(per[ds]["K_NET_primary_clears"] for ds in dss)
            deg_fires = any(per[ds]["K_DEG"]["K_DEG_fires"] for ds in dss)
            live = all(per[ds]["live_on_net"] for ds in dss)
            common.append({"multiplier": mult,
                           "k_per_dataset": {ds: per[ds]["k"] for ds in dss},
                           "live_on_net_both": bool(live),
                           "K_NET_clears": bool(net_ok),
                           "K_DEG_fires": bool(deg_fires),
                           "cell_clears": bool(net_ok and not deg_fires)})
        V["per_tau"][tname] = {
            "K_REACH_both": reach, "K_FELDMAN_both": feld,
            "multipliers_paired": mults,
            "multipliers_unpaired": sorted(set().union(
                *[set(by_mult[ds]) for ds in dss]) - set(mults)),
            "MARGINAL_ONLY_NOT_CONDITIONAL": bool(marg_only),
            "net_deg_cells": common,
            "clears": bool(reach and feld and any(c["cell_clears"] for c in common))}
        if V["per_tau"][tname]["clears"] and cont is None:
            cont = tname

    V["verdict_provisional"] = "CONTINUE" if cont is not None else "KILL"
    V["GATE_NULL_AGREEMENT"] = (
        gate_null_agreement(results, cfg, V) if _compare_null
        else {"ran": False, "agrees": None,
              "note": "inner call: this IS the remove-null substitution"})

    if cont is not None:
        lb_ok = all(results[ds]["head"]["tau"][cont]["item_bootstrap_fit_conditional"]
                    ["lower_95_one_sided"] is not None
                    and results[ds]["head"]["tau"][cont]
                    ["item_bootstrap_fit_conditional"]["lower_95_one_sided"] > 0
                    for ds in dss)
        V["verdict"] = "CONTINUE"
        V["continue_scope"] = {
            "tau": cont,
            "GATE_NULL_AGREEMENT": V["GATE_NULL_AGREEMENT"],
            "scoped_to_null_row_disagreement":
                (None if V["GATE_NULL_AGREEMENT"].get("agrees") in (True, None)
                 else "this verdict does NOT survive removal of the structural-zero "
                      "row: see GATE_NULL_AGREEMENT.disagreements"),
            "cells": [c for c in V["per_tau"][cont]["net_deg_cells"] if c["cell_clears"]],
            "robustness": "ROBUST" if lb_ok else "POINT_ESTIMATE_ONLY",
            "PROXY_FIDELITY_FLAG": ("INSTRUMENT_ABSENT"
                                    if results["_devfid"].get("instrument_absent_any")
                                    else results["_devfid"]["stop_rule_triggered_any"]),
            "DATA_DEFECT_OVERLAP": results["_datadefect"],
            "stage1_precondition": "VOID unless a proponent names an operator that is "
                                   "(a) global and symmetric at inference, (b) not one "
                                   "of F75's three named objectives, (c) not "
                                   "hard-example weighting alone, (d) adjudicated "
                                   "afresh against F75, F66, F98 and F99 (§11)"}
    else:
        fired = []
        for tname in ("tau_0", "tau_hi"):
            b = V["per_tau"][tname]
            if not b["K_REACH_both"]:
                fired.append("K-REACH@" + tname)
            elif not b["K_FELDMAN_both"]:
                fired.append("K-FELDMAN@" + tname)
            elif not any(c["cell_clears"] for c in b["net_deg_cells"]):
                fired.append("K-NET/K-DEG@" + tname)
        V["verdict"] = "KILL"
        V["kill_scope"] = {
            "rules_fired": fired,
            "GATE_NULL_AGREEMENT": V["GATE_NULL_AGREEMENT"],
            "scoped_to_null_row_disagreement":
                (None if V["GATE_NULL_AGREEMENT"].get("agrees") in (True, None)
                 else "this KILL does NOT survive removal of the structural-zero "
                      "row: see GATE_NULL_AGREEMENT.disagreements"),
            "closes": ("every tau >= 0 on both metrics by arithmetic (§4.3)"
                       if "K-REACH@tau_0" in fired
                       else "tau in {tau_0, tau_hi} only; neither precision nor AUC is "
                            "monotone in tau (§9)"),
            # per (dataset, tau): the design's own pre-declaration is that tau_hi is
            # the leg most likely to close on power, and a KILL scoped by
            # K-FELDMAN@tau_hi with the tag present only at tau_hi must read "not
            # identifiable AT THIS POWER", not "not identifiable" (§9, §10).
            "identifiability_underpowered": {
                ds: {t: results[ds]["head"]["tau"][t]["STRATUM_OCCUPANCY"]
                     ["IDENTIFIABILITY_UNDERPOWERED"] for t in ("tau_0", "tau_hi")}
                for ds in dss},
            "identifiability_underpowered_any": bool(any(
                results[ds]["head"]["tau"][t]["STRATUM_OCCUPANCY"]
                ["IDENTIFIABILITY_UNDERPOWERED"]
                for ds in dss for t in ("tau_0", "tau_hi"))),
            "secondary_030_cleared_anywhere": any(
                c["secondary_030"]["clears"]
                for ds in dss for t in results[ds]["head"]["tau"].values()
                for c in t["NET"]["cells"]),
            "note_if_secondary_only": "if a cell cleared the +0.030-sized net figure from "
                                      "banned_constraints[10] but not the +0.050-sized "
                                      "figure the C09 registry entry names, the record "
                                      "says so in terms (§5.3)"}
    return V


# ---------------------------------------------------------------------- __main__
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mintdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    det1_assert(str(a.threads))
    for f, sha in FROZEN_SHA.items():
        got = sha256_of(os.path.join(REPO, "scripts/analysis", f))
        assert got == sha, "FROZEN MODULE CHANGED: {} {}".format(f, got)
    torch.set_num_threads(a.threads)
    cfg = json.load(_OPEN(a.config))
    os.makedirs(a.outdir, exist_ok=True)
    logf = _OPEN(os.path.join(a.outdir, "C09_A0.log"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log("[c09] start {}".format(time.strftime("%Y-%m-%dT%H:%M:%S")))
    OUT = {"meta": {"script_sha256": sha256_of(os.path.abspath(__file__)),
                    "config_sha256": sha256_of(a.config), "frozen_sha256": FROZEN_SHA,
                    "config": cfg,
                    "design_of_record": {
                        "freeze_record": "refine-logs/C09_A0_RECORD.md",
                        "design_text": "refine-logs/C09_A0_V17_RECORD.md",
                        "design_review": "GO 0C/0H/0I at round 17",
                        "note": "the design text is the specification; the freeze "
                                "record carries the sha256 table, the measured-claims "
                                "register and every declared implementation reading"},
                    "test_contact": "NONE -- mint npz + train caches + gt text; "
                                    "open()/torch.load guards refuse any test path",
                    "runtime": runtime_block()}}
    # ONE root RNG for the whole job (design 5.2), children handed to every space.
    RNG_ROOT = np.random.default_rng(cfg["rng_seed"])
    RNG_CH = RNG_ROOT.spawn(6)
    OUT["meta"]["rng_root"] = {
        "seed": cfg["rng_seed"], "n_children": 6, "created": "once, in main()",
        "child_seed_sequences": [
            {"index": i,
             "entropy": c.bit_generator.seed_seq.entropy,
             "spawn_key": list(c.bit_generator.seed_seq.spawn_key)}
            for i, c in enumerate(RNG_CH)]}

    banked = {}
    for ds in cfg["datasets"]:
        banked[ds] = [json.load(_OPEN(os.path.join(
            REPO, "scripts/analysis",
            "headspace_arena_{}_s{}_OUT.json".format(ds, s)))) for s in cfg["seeds"]]

    results = {}
    for ds in cfg["datasets"]:
        z0 = np.load(os.path.join(a.mintdir, "mint_{}_s{}_f0.npz".format(
            ds, cfg["seeds"][0])), allow_pickle=True)
        lab = z0["lab"].astype(int)
        n = len(lab)
        assert n == cfg["n_items"][ds], "n mismatch {} {}".format(ds, n)
        skf = StratifiedKFold(n_splits=cfg["k_folds"], shuffle=True,
                              random_state=cfg["fold_seed"])
        splits = list(skf.split(np.zeros((n, 1)), lab))
        X, NRM, mint_meta = {}, {}, {}
        for s in cfg["seeds"]:
            X[s], NRM[s] = [], []
            for f in range(cfg["k_folds"]):
                z = np.load(os.path.join(a.mintdir, "mint_{}_s{}_f{}.npz".format(ds, s, f)),
                            allow_pickle=True)
                assert np.array_equal(z["lab"].astype(int), lab)
                assert np.array_equal(np.sort(z["fit_idx"]), np.sort(splits[f][0])), \
                    "mint fitting pool != frozen fold {} ({} s{})".format(f, ds, s)
                # I-13: the .npz's own meta must agree with the filename it was
                # loaded under, so a stale or mislabelled scratch mint fails loudly
                # here rather than being caught only indirectly by GATE-FLOOR.
                # headspace_mint.py:324 stores meta as a JSON STRING, not a dict.
                mt = json.loads(str(z["meta"].item())) if "meta" in z.files else {}
                for key, want in (("dataset", ds), ("seed", s), ("fold", f)):
                    assert key in mt, "mint meta missing {} ({} s{} f{})".format(
                        key, ds, s, f)
                    assert str(mt[key]) == str(want), (
                        "mint meta {}={} but filename says {} ({} s{} f{})".format(
                            key, mt[key], want, ds, s, f))
                assert str(mt.get("script_sha256")) == FROZEN_SHA["headspace_mint.py"], \
                    "mint was produced by a different headspace_mint.py"
                mint_meta["s{}_f{}".format(s, f)] = {
                    k: mt.get(k) for k in ("dataset", "seed", "fold", "n_train",
                                           "n_dev", "n_fit", "head_dim",
                                           "fold_parity_vs_banked_vsw_ckpt",
                                           "test_contact")}
                K = z["K_train"]
                # C-1: own_norm is the norm BEFORE the l2n that makes the key.
                NRM[s].append(np.linalg.norm(np.asarray(K, dtype="float64"), axis=1))
                X[s].append(P.l2n(K))
        with _OPEN(os.path.join(REPO, "data/gt", cd_ds(ds), "train.jsonl")) as _fh:
            rows = [json.loads(l) for l in _fh]
        assert len(rows) == n, "gt row count != mint n for {}".format(ds)
        # the DATA-DEFECT flags are built positionally, so the gt order must be the
        # cache order or every enrichment in continue_scope/kill_scope is misaligned
        _cd = P.DATASETS[ds]
        _ids, _, _, _ = P.load_cache(_cd["cache_dir"], "train", _cd["model"])
        _gid = [str(r.get("id", r.get("img", ""))) for r in rows]
        gt_order_ok = [str(x) for x in _ids] == _gid
        assert gt_order_ok, ("data/gt/{}/train.jsonl is not in the same order as the "
                            "train cache -- DATA_DEFECT flags would be "
                            "misaligned".format(cd_ds(ds)))
        flags = {"whitespace_only_text":
                 np.array([not str(r.get("text", "")).strip() for r in rows]),
                 "em_keyword_markup":
                 np.array(["<em" in str(r.get("text", "")) for r in rows])}
        head = run_space("head", ds, cfg, lab, splits, X, cfg["seeds"], log, flags=flags,
                         rawnorm_by_seed_fold=NRM, rng_children=RNG_CH)
        head["mint_meta_vs_filename"] = mint_meta

        # ---- GATE-FLOOR / GATE-PARITY-FOLD against the banked arena
        fl, pf = {}, {}
        for i, s in enumerate(cfg["seeds"]):
            r = banked[ds][i]["result"]
            got = head["floors"][str(s)]
            fl[str(s)] = {"banked_acc": r["acc_deployed"], "got_acc": round(got["acc"], 4),
                          "banked_mF1": r["mF1_deployed"], "got_mF1": round(got["mF1"], 4),
                          "acc_ok": round(got["acc"], 4) == r["acc_deployed"],
                          "mF1_ok": round(got["mF1"], 4) == r["mF1_deployed"]}
            _lens_ok = (len(got["fold_acc"]) == cfg["k_folds"]
                        and len(r["fold_acc_deployed"]) == cfg["k_folds"])
            pf[str(s)] = {"banked": r["fold_acc_deployed"], "got": got["fold_acc"],
                          "lengths_ok": bool(_lens_ok),
                          "ok": ([round(x, 4) == round(y, 4) for x, y in
                                  zip(got["fold_acc"], r["fold_acc_deployed"])]
                                 if _lens_ok else [False])}
        bank_rt = banked[ds][0]["meta"]["runtime"]
        drift = {k: [bank_rt["versions"].get(k), OUT["meta"]["runtime"]["versions"].get(k)]
                 for k in bank_rt["versions"]
                 if bank_rt["versions"].get(k) != OUT["meta"]["runtime"]["versions"].get(k)}
        if bank_rt.get("node") != OUT["meta"]["runtime"]["node"]:
            drift["node"] = [bank_rt.get("node"), OUT["meta"]["runtime"]["node"]]
        anch = cfg.get("banked_floors", {}).get(ds, {})
        anch_ok = True
        anch_rows = {}
        for i, s in enumerate(cfg["seeds"]):
            r = banked[ds][i]["result"]
            got = {"acc": (anch.get("acc") or [None] * 3)[i],
                   "mF1": (anch.get("mF1") or [None] * 3)[i]}
            hit = (got["acc"] == r["acc_deployed"] and got["mF1"] == r["mF1_deployed"])
            anch_ok = anch_ok and bool(hit)
            anch_rows[str(s)] = {"config_anchor": got,
                                 "banked_json": {"acc": r["acc_deployed"],
                                                 "mF1": r["mF1_deployed"]}, "ok": bool(hit)}
        head["GATE_FLOOR"] = {"per_seed": fl, "RUNTIME_DRIFT": drift,
                              "frozen_anchor_vs_banked_json":
                                  {"per_seed": anch_rows, "pass": bool(anch_ok)},
                              "pass": all(v["acc_ok"] and v["mF1_ok"] for v in fl.values())}
        assert anch_ok, ("configs/c09/c09_a0.json banked_floors[{}] does not match the "
                         "banked arena JSON it anchors".format(ds))
        head["GATE_PARITY_FOLD"] = {
            "per_seed": pf,
            "pass": bool(all(v["lengths_ok"] and all(v["ok"]) for v in pf.values()))}
        _fold_of_items = np.full(n, -1, dtype=int)
        for _f, (_, _ho) in enumerate(splits):
            _fold_of_items[np.asarray(_ho)] = _f
        head["GATE_NESTED"] = {
            "partition_equals_arena_folds": bool(
                (_fold_of_items >= 0).all()
                and sum(len(h) for _, h in splits) == n
                and len(set().union(*[set(map(int, h)) for _, h in splits])) == n),
            "per_tau": {t: head["tau"][t]["GATE_NESTED"] for t in head["tau"]},
            "assertion": "for every scored item: its arena fold was excluded from the "
                         "model that scored it, all seed-rows excluded together, the "
                         "tau_hi threshold applied was computed with its fold excluded, "
                         "and every row in that fit was assigned to its class by the "
                         "same tau_hi^(f) (A^(f), §5.2)",
            "pass": bool(all(head["tau"][t]["GATE_NESTED"]["pass"] for t in head["tau"]))}
        assert head["GATE_NESTED"]["partition_equals_arena_folds"], \
            "GATE-NESTED: the frozen folds do not partition the {} items".format(ds)
        results[ds] = {"head": head}

        # GATE-NULL (1): the census must be MEASURED and the frozen row confirmed
        # BEFORE the sensitivity leg drops it -- otherwise a changed cache would let
        # the leg remove an arbitrary row while still being labelled "remove the
        # structural zero".
        results[ds]["GATE_NULL"] = gate_null_census(ds, P.DATASETS[ds], lab, cfg)
        _z = results[ds]["GATE_NULL"]
        if ds == cfg["null_row_dataset"]:
            assert cfg["null_row_index"] in _z["zero_img_rows"] \
                and cfg["null_row_index"] in _z["zero_txt_rows"], (
                "GATE-NULL: frozen null_row_index {} is not a structural zero in BOTH "
                "streams of the operative {} train cache (measured img={} txt={})"
                .format(cfg["null_row_index"], ds, _z["zero_img_rows"],
                        _z["zero_txt_rows"]))
        else:
            assert not _z["zero_img_rows"] and not _z["zero_txt_rows"], (
                "GATE-NULL: {} was frozen as having NO structural-zero row, but the "
                "operative cache has img={} txt={}".format(
                    ds, _z["zero_img_rows"], _z["zero_txt_rows"]))
        if ds == cfg["null_row_dataset"]:
            results[ds]["head_remove_null"] = run_space(
                "head_remove_null", ds, cfg, lab, splits, X, cfg["seeds"], log,
                flags=flags, drop=cfg["null_row_index"],
                rawnorm_by_seed_fold=NRM, rng_children=RNG_CH)

        cd = P.DATASETS[ds]
        _, img, txt, lab2 = P.load_cache(cd["cache_dir"], "train", cd["model"])
        assert np.array_equal(np.asarray(lab2).astype(int), lab)
        Xpre = np.concatenate([P.l2n(img), P.l2n(txt)], axis=1)
        Xraw = P.l2n(Xpre)
        nrm_raw = np.linalg.norm(np.asarray(Xpre, dtype="float64"), axis=1)
        raw = run_space("raw", ds, cfg, lab, splits, {0: [Xraw] * cfg["k_folds"]}, [0],
                        log, flags=flags,
                        rawnorm_by_seed_fold={0: [nrm_raw] * cfg["k_folds"]},
                        rng_children=RNG_CH)
        _sdr = float(np.std(nrm_raw))
        _nz = sorted(map(int, np.flatnonzero(nrm_raw == 0.0)))
        raw["own_norm_note"] = {
            "measured_sd": round(_sdr, 12),
            "n_zero_norm_rows": len(_nz), "zero_norm_rows": _nz,
            "reading": (
                "in the raw space the pre-normalisation object is "
                "concat(l2n(img), l2n(txt)); l2n maps a unit-norm row pair to norm "
                "sqrt(2), so own_norm is constant EXCEPT at a structural-zero row, "
                "where l2n(0) = 0. On {} that makes own_norm a one-item indicator "
                "for row(s) {} rather than a constant, and it is NOT absorbed by "
                "fit_lr's sd < 1e-12 guard; where there is no zero row the column is "
                "constant and IS absorbed. Non-decisional either way: the raw leg is "
                "KILL corroboration only (F113)."
            ).format(ds, _nz if _nz else "none")}
        # I-11: the recomputed raw floor must reproduce the banked raw deployed acc
        _brd = banked[ds][0]["membership"]["raw_deployed_acc"]
        raw["banked_raw_deployed_acc"] = _brd
        raw["banked_raw_acc_matches_recomputed"] = {
            "banked": _brd, "recomputed": round(raw["floors"]["0"]["acc"], 4),
            "ok": bool(round(raw["floors"]["0"]["acc"], 4) == round(float(_brd), 4)),
            "non_decisional": True}
        raw["confined_to"] = "KILL corroboration only (unified_pilot_gate.arena; F113)"
        _rawmax = max([abs(c["per_seed"]["0"]["d_acc"])
                       for t in raw["tau"].values() for c in t["NET"]["cells"]] or [0.0])
        raw["CAL_3"] = dict(cfg["cal3_comparators"],
                            raw_leg_max_abs_d_acc_reported=round(float(_rawmax), 6),
                            mandatory_here=bool(_rawmax >= 0.010),
                            arm_escalated="NONE -- see status")
        results[ds]["raw"] = raw

        results[ds]["DATA_DEFECT_FLAG_COUNTS"] = {
            nm: int(np.asarray(v).sum()) for nm, v in flags.items()}
        # (census already measured and asserted above, before the remove-null leg)

    OUT["results"] = results

    # ---- GATE-LEDGER: MEASURED across every process of the job (the 36 mints and
    # the 2 fidelity runs write their own counts at exit; this process's counts are
    # still in memory, so they are added here rather than read back).
    led_dir = os.environ.get("C09_LEDGER_DIR", "")
    other, procs, stale = c09guard.aggregate(led_dir)
    tot = {k: int(other.get(k, 0)) + int(LEDGER.get(k, 0))
           for k in set(other) | set(LEDGER)}
    cov = c09guard.verify_predicate()
    OUT["GATE_LEDGER"] = {
        "predicate_coverage_measured_this_run": {
            "n_repo_files_matched": cov["n_matched"],
            "n_unmatched_paths_containing_test": cov["n_unmatched_containing_test"],
            "sample_unmatched": [x.replace(REPO + "/", "")
                                 for x in cov["sample_unmatched"]],
            "note": "re-derived against the live tree by "
                    "c09guard.verify_predicate(); the unmatched residue is "
                    "inspected in the freeze record and contains no test-SPLIT "
                    "artifact this job can reach"},
        "measured": {
            "test_path_opens": tot["test_path_opens"],
            "dev_path_opens": tot["dev_path_opens"],
            "banked_trainlog_opens": tot["banked_trainlog_opens"]},
        "measured_expectations": {
            "test_path_opens": 0,
            "dev_path_opens": "36 on a fresh run (one dev_seen load per mint); LOWER "
                              "on a resume, since a mint whose .npz already exists is "
                              "skipped without opening anything",
            "banked_trainlog_opens": "6 (GATE-DEVFID reads the banked encoder "
                                     "trainlogs); design 8.1 counts these separately "
                                     "from the 36, for a declared dev-side total of 42",
            "declared_dev_side_total_8_1": 42},
        "evidence": {
            "n_processes_reporting": len(procs),
            "n_processes_expected_fresh_run": 39,
            "n_processes_expected_breakdown":
                "1 version/preflight heredoc + 36 mints + 2 GATE-DEVFID runs "
                "(the arena's own counts are added in memory, not read back)",
            "resume_note": "a resume legitimately reports FEWER: a mint whose .npz "
                           "already exists is skipped and opens nothing. The gate "
                           "requires >= 1 reporting process, not 39.",
            "stale_ledger_files_from_earlier_attempts": stale,
            "why_this_is_gated": "a ledger that reports zero because NO process ever "
                                 "reported is not evidence of a clean run; the gate "
                                 "therefore requires at least one reporting process "
                                 "besides the arena"},
        "by_construction_zero": {
            "test_label_materialisations":
                "no test path can be opened (guard raises), so no test label can be "
                "materialised; this is a consequence of test_path_opens == 0, not an "
                "independent count",
            "dev_label_materialisations_outside_decisions":
                "dev labels are materialised once per mint inside headspace_mint.py "
                "(36 mints) for the deployed-head fit; the arena never loads a dev "
                "label -- it reads only mint .npz key matrices, train caches and "
                "data/gt/*/train.jsonl",
            "dev_or_test_labels_into_decision_quantities":
                "every decision quantity in this file is computed from train-split "
                "labels via BankLabels or the query-side train labels; no dev or "
                "test label array is in scope of any decision path"},
        "per_process": procs,
        "arena_process_counts": dict(LEDGER),
        "ledger_dir": led_dir,
        "guard_predicate": "repo-scoped: basename contains test_seen, or begins "
                           "test. / test_, or a path component is exactly 'test'",
        "pass": bool(tot["test_path_opens"] == 0 and len(procs) >= 1)}

    devfid = {"per_dataset": {}, "stop_rule_triggered_any": False,
              "instrument_absent_any": False,
              "reading": "GATE-DEVFID is REPORTING ONLY (8.2) and does not gate. "
                         "'INSTRUMENT_ABSENT' is a THIRD value, distinct from a "
                         "clean pass: it means the instrument did not run."}
    for ds in cfg["datasets"]:
        pth = os.path.join(a.outdir, "C09_FIDELITY_{}.json".format(ds))
        if os.path.exists(pth):
            with _OPEN(pth) as _fh:
                g = json.load(_fh)["gate"]
            devfid["per_dataset"][ds] = g
            devfid["stop_rule_triggered_any"] |= bool(g.get("STOP_RULE_TRIGGERED"))
        else:
            devfid["per_dataset"][ds] = "INSTRUMENT_ABSENT"
            devfid["instrument_absent_any"] = True
    OUT["GATE_DEVFID"] = devfid

    adj_in = dict(results)
    adj_in["_ledger"] = OUT["GATE_LEDGER"]
    adj_in["_devfid"] = devfid
    adj_in["_datadefect"] = {ds: {
        "flag_counts": results[ds]["DATA_DEFECT_FLAG_COUNTS"],
        "population_enrichment": {t: results[ds]["head"]["tau"][t]
                                  .get("DATA_DEFECT_OVERLAP_population")
                                  for t in results[ds]["head"]["tau"]}}
        for ds in cfg["datasets"]}
    OUT["DECISION"] = adjudicate(adj_in, cfg)
    log("[c09] VERDICT: {}".format(OUT["DECISION"]["verdict"]))

    tmp = os.path.join(a.outdir, "C09_A0_OUT.json.tmp")
    with _OPEN(tmp, "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)
    os.replace(tmp, os.path.join(a.outdir, "C09_A0_OUT.json"))
    tmp2 = os.path.join(a.outdir, "C09_A0_DECISION.json.tmp")
    with _OPEN(tmp2, "w") as fh:
        json.dump({"meta": OUT["meta"], "GATE_LEDGER": OUT["GATE_LEDGER"],
                   "GATE_DEVFID": OUT["GATE_DEVFID"],
                   "DECISION": OUT["DECISION"]}, fh, indent=1, default=str)
    os.replace(tmp2, os.path.join(a.outdir, "C09_A0_DECISION.json"))
    log("[c09] wrote {}".format(os.path.join(a.outdir, "C09_A0_OUT.json")))
    logf.close()


if __name__ == "__main__":
    main()
