#!/usr/bin/env python3
"""W2-A Stage-P' probe — CPU / cloud-eligible, ZERO training, ZERO test touch.

Pre-registration : research-wiki/experiments/exp-w2a-grounded.md  (r1 APPROVED-WITH-AMENDMENTS, cb59a94)
Forensic recon   : refine-logs/W2A_FORENSIC_RECON.md
Prereg review    : refine-logs/W2A_PREREG_REVIEW.md

Memory = TRAIN + VAL only (HateMM 851 incl. 1 zero-guard; MHC-EN 629). LOO: every memory video is a
query against the rest; retrieve top-20 by each arm's pairwise score; call the REAL pipeline vote
(src/utils/metrics.py compute_metrics_retrieval, use_sim=True, arithmetic, topk=20 — NOT reimplemented).

BINDING PERFORMANCE ADJUDICATOR (r1 Amdt 2, option (b) — the SOLE binding perf gate):
  K9  C3-template conditional-info probe of A=grd on top of Z_best = concat(CLIP img[1024], CLIP
      text[768], Qwen img[3584], Qwen text[3584]) = 8960-d (r1 Amdt 1). Un-penalised aux block at
      Z_best's inner-CV-optimal C; 5x5 RepeatedStratifiedKFold; per-video-clustered bootstrap; a
      label-oracle calibration arm (accZA ~ 1.0 or MACHINERY_INVALID) reaching full Fano headroom;
      >=150-permutation null of A across videos AS A DISTRIBUTION. +0.040 TRIPLE rule (C3-verbatim):
      (C1) best decision-k Delta-acc >= +0.040, (C2) per-video bootstrap CI-lower > 0, (C3) real >
      ALL >=150 perm maxima. A grounded key CLIP-redundant against Z_best is DEAD.

KILL-SWITCH (binding, decides whether ANY head GPU is spent):
  K5  oracle-ceiling — per-query gold choice grd-vs-CONCAT (tie->CONCAT); Delta(oracle-CONCAT) acc <
      +0.04 on EVERY dataset -> DEAD, zero head GPU. Conservative (fixed-50/50 CONCAT inflates the gap).

MACHINE-VALIDITY:
  K4  Fano — +/-1 gold-label-agreement key vote acc >= 0.99 both datasets, else VOID.

ADVISORY (reported, NON-gating — r1 Amdt 2b; the fixed-50/50 kNN CONCAT geometry can false-PASS):
  K6  raw HateMM Delta(GROUNDED-CONCAT) acc >= +0.05 AND mF1 >= +0.05, beating CONCAT-PCA + CONCAT-alpha
      in sign; K7 obs Delta > 95th-pct key-shuffle permutation null (seeds 0..99); K7b near-dup-excluded
      advantage survives; K8 bootstrap 5th-pct of paired Delta > 0 (D3). Covered-rows-only HateMM
      secondary (non-empty transcript) for BOTH the binding K9 and the advisory kNN (r1 Amdt 5).

FAIL-CLOSED (N4): never constructs/opens any test_seen file; asserts memory size == 851 / 629. Gold
labels are PROBE-ONLY (Fano / oracle / K9 calibration + targets), train u val only. The executor writes
RAW numbers; `mechanical_gate_check` is pre-registered threshold ARITHMETIC and is NOT the binding
verdict (an independent verdict reviewer rules).
"""
import argparse
import json
import os
import sys
import time
import warnings

# CPU determinism / thread caps (cloud-eligible, features-only).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

REPO = "/data/jehc223/RGCL"
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from utils.metrics import compute_metrics_retrieval  # the REAL vote (do NOT reimplement)

# ---------------- pre-registered constants (prereg §16; hash-frozen) ----------------
TOPK = 20
EXPECTED_MEM = {"HateMM": 851, "MHC": 629}          # train u val (N4 fail-closed size guard)
QWEN = "Qwen2.5-VL-7B-Instruct_HF"
CLIP = "openai_clip-vit-large-patch14-336_HF"
GROUNDED_DIR_TMPL = "grounded_qwen7b_{}f"

ORACLE_BAR = 0.04                                   # §6.4 K5 kill-switch
RAW_BAR = 0.05                                      # §6.5 K6 advisory
FANO_BAR = 0.99                                     # §6.3 K4
NEAR_DUP_THRESH = 0.995                             # §6.6 A3 flag (grd OR concat)
NEAR_DUP_REPORT = [0.98, 0.99, 0.995]
NULL_SEEDS = list(range(100))                       # advisory kNN null seeds 0..99
ALPHA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]             # CONCAT-alpha advisory grid (train u val LOO)
TIEBREAK_EPS = 1e-9
NEG_INF = -1e30

# ---- binding conditional-info (K9) machinery — C3-verbatim (scripts/analysis/c3_fusion_probe.py) ----
CI_N_SPLITS, CI_N_REPEATS = 5, 5
CI_C_GRID = [0.001, 0.01, 0.1, 1.0]
CI_B_BOOT = 5000
CI_SCALE_A = 50.0
CI_MAX_ITER = 2000
CI_BAR = 0.040                                       # +0.040 triple rule (C3-verbatim)
CI_KS_REPORT = [8, 16, 32, 64]
CI_KS_DECISION = [8, 16]
CI_SHUFFLE_SEED = 12345
CI_BOOT_SEED = 20260715
CI_NSEED_PERM = int(os.environ.get("CI_NSEED", "150"))   # >=150 mandated (prereg §16)
CI_PERM_BASE = 70000
CI_EPS = 1e-12


def _log(m):
    print(m, flush=True)


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# Loading (N4: train + dev_seen ONLY; test_seen is NEVER constructed).
# ===========================================================================
def _guard_outname(outname):
    assert outname in ("train", "dev_seen"), \
        "N4 GUARD: probe may only load train/dev_seen, never '{}'".format(outname)


def _grounded_path(dataset, grounded_dir, outname):
    _guard_outname(outname)
    return os.path.join(REPO, "data/CLIP_Embedding", dataset, grounded_dir,
                        "{}_grounded.pt".format(outname))


def _bank_path(dataset, tag, outname):
    _guard_outname(outname)
    return os.path.join(REPO, "data/CLIP_Embedding", dataset, "{}_{}.pt".format(outname, tag))


def _load_bank(dataset, tag, outname, order_ids):
    o = torch.load(_bank_path(dataset, tag, outname), map_location="cpu", weights_only=False)
    idx = {str(i): k for k, i in enumerate(o["ids"][0])}
    order = [idx[i] for i in order_ids]
    return o["img_feats"][order].float(), o["text_feats"][order].float()


def load_memory(dataset, grounded_dir):
    """Assemble the LOO memory from train u val ONLY, all arms id/order-aligned to the grounded ids.
    HARD-fails on any test touch or a size != the exact expected train u val count."""
    ids, grd, grd_pfx, labels, guard, empty = [], [], [], [], [], []
    qimg, qtxt, cimg, ctxt = [], [], [], []
    for outname in ("train", "dev_seen"):           # NEVER test_seen (N4)
        gp = _grounded_path(dataset, grounded_dir, outname)
        if not os.path.exists(gp):
            raise RuntimeError("missing grounded cache: {}".format(gp))
        g = torch.load(gp, map_location="cpu", weights_only=False)
        gids = [str(x) for x in g["ids"][0]]
        ids.extend(gids)
        grd.append(g["grd"].float()); grd_pfx.append(g["grd_pfx"].float())
        labels.append(g["labels"].long())
        guard.append(g["zero_guard"].bool())
        empty.append(g["empty_transcript"].bool())
        qi, qt = _load_bank(dataset, QWEN, outname, gids)
        ci, ct = _load_bank(dataset, CLIP, outname, gids)
        qimg.append(qi); qtxt.append(qt); cimg.append(ci); ctxt.append(ct)

    grd = torch.cat(grd, 0); grd_pfx = torch.cat(grd_pfx, 0)
    labels = torch.cat(labels, 0).numpy().astype(int)
    guard = torch.cat(guard, 0).numpy()
    empty = torch.cat(empty, 0).numpy()
    qimg = torch.cat(qimg, 0); qtxt = torch.cat(qtxt, 0)
    cimg = torch.cat(cimg, 0); ctxt = torch.cat(ctxt, 0)
    N = grd.shape[0]
    exp = EXPECTED_MEM[dataset]
    if N != exp:
        raise RuntimeError("N4 SIZE GUARD: memory N={} != expected train u val {} for {} "
                           "(a stray test row?)".format(N, exp, dataset))
    _log("[{}] memory N={} (train u val) D={} zero-guard={} empty-transcript={}".format(
        dataset, N, grd.shape[1], int(guard.sum()), int(empty.sum())))
    return {"ids": ids, "grd": grd, "grd_pfx": grd_pfx, "labels": labels, "guard": guard,
            "empty": empty, "qimg": qimg, "qtxt": qtxt, "cimg": cimg, "ctxt": ctxt, "N": N}


# ===========================================================================
# kNN score matrices + REAL vote (advisory arms; identical vote path as S2S).
# ===========================================================================
def _l2(x, dim=-1, eps=1e-12):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _cos_mat(V):
    Vh = _l2(V, dim=-1)
    return (Vh @ Vh.t()).numpy().astype(np.float64)


def _concat_halfnorm(a, b, wa=1.0, wb=1.0):
    """[sqrt(wa)*ahat || sqrt(wb)*bhat] then row-normalise -> cosine = wa'*cos_a + wb'*cos_b."""
    ah, bh = _l2(a, dim=-1), _l2(b, dim=-1)
    return torch.cat([ah * (wa ** 0.5), bh * (wb ** 0.5)], dim=1)


def build_matrices(mem):
    grd = mem["grd"]
    S_grd = _cos_mat(grd)
    S_grd_pfx = _cos_mat(mem["grd_pfx"])
    S_pooled_img = _cos_mat(mem["qimg"])
    # CONCAT (Qwen marginals, fixed 50/50) — advisory.
    concat = _concat_halfnorm(mem["qimg"], mem["qtxt"], 1.0, 1.0)
    S_concat = _cos_mat(concat)
    # GROUNDED+TEXT sensitivity = [grd || text_feats] fixed 50/50.
    S_grd_text = _cos_mat(_concat_halfnorm(grd, mem["qtxt"], 1.0, 1.0))
    return {"grd": S_grd, "grd_pfx": S_grd_pfx, "pooled_img": S_pooled_img,
            "concat": S_concat, "grd_text": S_grd_text, "N": mem["N"]}


def concat_pca_matrix(mem):
    """CONCAT-PCA (advisory, dim-matched to grd where the memory rank allows). Memory-only fit
    (leak-free of TEST). n_components is capped at min(3584, N-1, 7168) — PCA rank is bounded by the
    memory sample count, so for N=851/629 the effective dim is <3584 (documented, non-gating)."""
    raw = _concat_halfnorm(mem["qimg"], mem["qtxt"], 1.0, 1.0).numpy().astype(np.float64)  # [N,7168]
    n_comp = int(min(3584, raw.shape[0] - 1, raw.shape[1]))
    P = PCA(n_components=n_comp, random_state=0).fit_transform(raw)
    return _cos_mat(torch.tensor(P)), n_comp


def concat_alpha_matrix(mem, labels):
    """CONCAT-alpha (advisory, weight-tuned): pick alpha on the small grid by train u val LOO acc
    (leak-free of TEST). Returns (best_S, best_alpha, per_alpha_acc)."""
    per = {}
    best_a, best_acc, best_S = None, -1.0, None
    for a in ALPHA_GRID:
        S = _cos_mat(_concat_halfnorm(mem["qimg"], mem["qtxt"], a, 1.0 - a))
        acc, _mf1, _roc, _v = run_vote(S, labels)
        per["alpha_%.2f" % a] = float(acc)
        if acc > best_acc:
            best_acc, best_a, best_S = acc, a, S
    return best_S, best_a, per


def _tiebreak(S):
    N = S.shape[1]
    return S - (np.arange(N, dtype=np.float64)[None, :] * TIEBREAK_EPS)


def run_vote(S, labels, k=TOPK, rank_only=False, exclude=None, subset=None):
    """Top-k LOO retrieval by score matrix S, then the pipeline's real vote. subset (optional bool
    mask [N]) restricts BOTH memory and queries to the covered rows (Amdt-5 covered-only view)."""
    N = S.shape[0]
    St = _tiebreak(S).copy()
    np.fill_diagonal(St, NEG_INF)
    if exclude is not None:
        St[exclude] = NEG_INF
    if subset is not None:
        St[~subset, :] = NEG_INF          # drop non-covered memory columns/queries
        St[:, ~subset] = NEG_INF
    q_idx = np.where(subset)[0] if subset is not None else range(N)
    logging_dict = {}
    for i in q_idx:
        row = St[i]
        topk_idx = np.argpartition(-row, k)[:k]
        topk_idx = topk_idx[np.argsort(-row[topk_idx])]
        topk_idx = topk_idx[row[topk_idx] > (NEG_INF / 2)]   # never let self/excluded enter the vote
        if topk_idx.size == 0:
            raise RuntimeError("degenerate retrieval: no finite neighbours for query {}".format(i))
        sims = (np.ones(topk_idx.size) if rank_only else row[topk_idx].astype(np.float64))
        logging_dict[int(i)] = {"retrieved_label": [int(labels[j]) for j in topk_idx],
                                "retrieved_scores": list(sims)}
    y = labels if subset is None else labels[subset]
    acc, roc, pre, rec, f1, votes, _lab, macro = compute_metrics_retrieval(
        logging_dict, torch.tensor(y), majority_voting="arithmetic", topk=k, use_sim=True)
    return float(acc), float(macro["macro_f1"]), float(roc), np.asarray(votes, dtype=np.float64)


def _preds_from_votes(v):
    return (1.0 / (1.0 + np.exp(-v)) >= 0.5).astype(int)


# ---- oracle-ceiling (K5): per-query gold key choice grd-vs-CONCAT, tie->CONCAT ----
def oracle_ceiling(v_grd, v_cat, labels):
    signed = labels.astype(np.float64) * 2.0 - 1.0
    choose_grd = (signed * v_grd) > (signed * v_cat)      # strict > ; tie -> CONCAT
    v_orc = np.where(choose_grd, v_grd, v_cat)
    pred = _preds_from_votes(v_orc)
    acc = float((pred == labels).mean())
    mf1 = float(f1_score(labels, pred, average="macro", zero_division=0))
    return acc, mf1, float(choose_grd.mean())


# ---- Fano (K4): +/-1 gold-label-agreement key ----
def fano(labels, k=TOPK):
    same = (labels[:, None] == labels[None, :]).astype(np.float64)
    S = same * 2.0 - 1.0
    acc, _mf1, _roc, _v = run_vote(S, labels, k)
    return acc


# ---- near-dup audit (A3) ----
def near_dup_audit(M, guard):
    N = M["N"]
    tri = np.triu(np.ones((N, N), dtype=bool), k=1)
    valid = tri & ~(guard[:, None] | guard[None, :])
    dist = {}
    for th in NEAR_DUP_REPORT:
        dist["grd>=%.3f" % th] = int(((M["grd"] >= th) & valid).sum())
        dist["concat>=%.3f" % th] = int(((M["concat"] >= th) & valid).sum())
    flag_u = ((M["grd"] >= NEAR_DUP_THRESH) | (M["concat"] >= NEAR_DUP_THRESH)) & valid
    flag = flag_u | flag_u.T
    return flag, dist, int(flag_u.sum())


# ---- advisory permutation null (K7): key-shuffle across videos, SAME perm both arms ----
# The rank-only co-diagnostic (sim neutralised to 1.0) gets its OWN null, same per-seed permutation,
# so a GROUNDED advantage can be attributed to rank structure vs sim-scale (S2S A2 pattern).
def permutation_null(M, labels, k=TOPK):
    grd, concat = M["grd"], M["concat"]
    og_a, og_f, _, _ = run_vote(grd, labels, k)
    oc_a, oc_f, _, _ = run_vote(concat, labels, k)
    ogr_a, ogr_f, _, _ = run_vote(grd, labels, k, rank_only=True)
    ocr_a, ocr_f, _, _ = run_vote(concat, labels, k, rank_only=True)
    obs = {"dacc": og_a - oc_a, "df1": og_f - oc_f,
           "dacc_rank": ogr_a - ocr_a, "df1_rank": ogr_f - ocr_f}
    nd = {kk: [] for kk in obs}
    for s in NULL_SEEDS:
        perm = np.random.default_rng(s).permutation(M["N"])
        ix = np.ix_(perm, perm)
        ga, gf, _, _ = run_vote(grd[ix], labels, k)
        ca, cf, _, _ = run_vote(concat[ix], labels, k)
        gar, gfr, _, _ = run_vote(grd[ix], labels, k, rank_only=True)
        car, cfr, _, _ = run_vote(concat[ix], labels, k, rank_only=True)
        nd["dacc"].append(ga - ca); nd["df1"].append(gf - cf)
        nd["dacc_rank"].append(gar - car); nd["df1_rank"].append(gfr - cfr)
    out = {"n_seeds": len(NULL_SEEDS)}
    for kk in obs:
        p95 = float(np.percentile(np.array(nd[kk]), 95))
        out["obs_" + kk] = obs[kk]
        out["null_" + kk + "_p95"] = p95
        out["obs_" + kk + "_gt_p95"] = bool(obs[kk] > p95)
    return out


# ---- bootstrap (D3, K8) ----
def bootstrap_delta(v_grd, v_cat, labels, n_boot, seed=20260715):
    pg, pc = _preds_from_votes(v_grd), _preds_from_votes(v_cat)
    y = labels.astype(int); N = len(y)
    rng = np.random.default_rng(seed)
    dacc, df1 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, N); yy = y[idx]
        dacc[b] = (pg[idx] == yy).mean() - (pc[idx] == yy).mean()
        df1[b] = (f1_score(yy, pg[idx], average="macro", zero_division=0)
                  - f1_score(yy, pc[idx], average="macro", zero_division=0))
    return {"n_boot": n_boot,
            "dacc_p5": float(np.percentile(dacc, 5)), "dacc_p50": float(np.percentile(dacc, 50)),
            "dacc_p95": float(np.percentile(dacc, 95)),
            "df1_p5": float(np.percentile(df1, 5)), "df1_p50": float(np.percentile(df1, 50)),
            "df1_p95": float(np.percentile(df1, 95)),
            "dacc_p5_gt0": bool(np.percentile(dacc, 5) > 0.0),
            "df1_p5_gt0": bool(np.percentile(df1, 5) > 0.0)}


# ===========================================================================
# BINDING K9 — C3-template conditional-info probe of A=grd on Z_best (8960-d).
# Machinery mirrors scripts/analysis/c3_fusion_probe.py VERBATIM.
# ===========================================================================
def build_Zbest(mem):
    """Z_best = concat(CLIP img[1024], CLIP text[768], Qwen img[3584], Qwen text[3584]) = 8960-d."""
    Z = np.concatenate([mem["cimg"].numpy(), mem["ctxt"].numpy(),
                        mem["qimg"].numpy(), mem["qtxt"].numpy()], axis=1).astype(np.float64)
    assert Z.shape[1] == 8960, Z.shape
    return Z


def build_Zqwen(mem):
    """Secondary context baseline Qwen-only concat = 7168-d (reported, non-binding)."""
    return np.concatenate([mem["qimg"].numpy(), mem["qtxt"].numpy()], axis=1).astype(np.float64)


def _ci_pick_C(Z, y):
    best_c, best = CI_C_GRID[0], -1.0
    skf = StratifiedKFold(CI_N_SPLITS, shuffle=True, random_state=0)
    for c in CI_C_GRID:
        a = []
        for tr, te in skf.split(Z, y):
            sc = StandardScaler().fit(Z[tr])
            lr = LogisticRegression(C=c, max_iter=CI_MAX_ITER).fit(sc.transform(Z[tr]), y[tr])
            a.append((lr.predict(sc.transform(Z[te])) == y[te]).mean())
        if np.mean(a) > best:
            best, best_c = float(np.mean(a)), c
    return best_c, best


def _ci_pick_C_combined(Z, A, y):
    best_c, best = CI_C_GRID[0], -1.0
    skf = StratifiedKFold(CI_N_SPLITS, shuffle=True, random_state=0)
    for c in CI_C_GRID:
        a = []
        for tr, te in skf.split(Z, y):
            scz, sca = StandardScaler().fit(Z[tr]), StandardScaler().fit(A[tr])
            Xtr = np.concatenate([scz.transform(Z[tr]), sca.transform(A[tr])], axis=1)
            Xte = np.concatenate([scz.transform(Z[te]), sca.transform(A[te])], axis=1)
            lr = LogisticRegression(C=c, max_iter=CI_MAX_ITER).fit(Xtr, y[tr])
            a.append((lr.predict(Xte) == y[te]).mean())
        if np.mean(a) > best:
            best, best_c = float(np.mean(a)), c
    return best_c, best


def _ci_fit_cor(Xtr, ytr, Xte, yte, C):
    lr = LogisticRegression(C=C, max_iter=CI_MAX_ITER).fit(Xtr, ytr)
    return ((lr.predict_proba(Xte)[:, 1] >= 0.5).astype(int) == yte).astype(float)


def _ci_baseline_cor(Z, y, C_Z):
    n = len(y); cor = np.zeros(n); cnt = np.zeros(n)
    for rep in range(CI_N_REPEATS):
        for tr, te in StratifiedKFold(CI_N_SPLITS, shuffle=True, random_state=1000 + rep).split(Z, y):
            sc = StandardScaler().fit(Z[tr]); cnt[te] += 1
            cor[te] += _ci_fit_cor(sc.transform(Z[tr]), y[tr], sc.transform(Z[te]), y[te], C_Z)
    return cor / cnt


def _ci_oracle_cor(Z, y, C_Z):
    """label-oracle calibration arm: append 2-col one-hot(y) x SCALE_A (raw, unpenalised)."""
    n = len(y); A_lab = np.zeros((n, 2)); A_lab[np.arange(n), y] = 1.0
    cor = np.zeros(n); cnt = np.zeros(n)
    for rep in range(CI_N_REPEATS):
        for tr, te in StratifiedKFold(CI_N_SPLITS, shuffle=True, random_state=1000 + rep).split(Z, y):
            sc = StandardScaler().fit(Z[tr]); Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])
            cnt[te] += 1
            cor[te] += _ci_fit_cor(np.concatenate([Ztr, A_lab[tr] * CI_SCALE_A], 1), y[tr],
                                   np.concatenate([Zte, A_lab[te] * CI_SCALE_A], 1), y[te], C_Z)
    return cor / cnt


def _ci_full_cor(Z, A, y, C_full):
    n = len(y); cor = np.zeros(n); cnt = np.zeros(n)
    for rep in range(CI_N_REPEATS):
        for tr, te in StratifiedKFold(CI_N_SPLITS, shuffle=True, random_state=1000 + rep).split(Z, y):
            scz, sca = StandardScaler().fit(Z[tr]), StandardScaler().fit(A[tr]); cnt[te] += 1
            Xtr = np.concatenate([scz.transform(Z[tr]), sca.transform(A[tr])], axis=1)
            Xte = np.concatenate([scz.transform(Z[te]), sca.transform(A[te])], axis=1)
            cor[te] += _ci_fit_cor(Xtr, y[tr], Xte, y[te], C_full)
    return cor / cnt


def _ci_arm_cor_allk(Z, y, C_Z, src, ks):
    """per-video cor for every k in ks in ONE CV pass (train-fold PCA of A, sliced; leak-free)."""
    n = len(y); cor = {k: np.zeros(n) for k in ks}; cnt = np.zeros(n); kmax = max(ks)
    for rep in range(CI_N_REPEATS):
        for tr, te in StratifiedKFold(CI_N_SPLITS, shuffle=True, random_state=1000 + rep).split(Z, y):
            scz = StandardScaler().fit(Z[tr]); Ztr, Zte = scz.transform(Z[tr]), scz.transform(Z[te])
            cnt[te] += 1
            scs = StandardScaler().fit(src[tr]); Str, Ste = scs.transform(src[tr]), scs.transform(src[te])
            kk = min(kmax, len(tr) - 1, src.shape[1]); pca = PCA(n_components=kk, random_state=0).fit(Str)
            Ptr, Pte = pca.transform(Str), pca.transform(Ste)
            for k in ks:
                j = min(k, kk); scp = StandardScaler().fit(Ptr[:, :j])
                Btr = scp.transform(Ptr[:, :j]) * CI_SCALE_A; Bte = scp.transform(Pte[:, :j]) * CI_SCALE_A
                cor[k][te] += _ci_fit_cor(np.concatenate([Ztr, Btr], 1), y[tr],
                                          np.concatenate([Zte, Bte], 1), y[te], C_Z)
    return {k: cor[k] / cnt for k in ks}


def _ci_boot_ci(cor_arm, cor_base, seed):
    d = cor_arm - cor_base; n = len(d); rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(CI_B_BOOT)])
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _ci_point(Z, A, y):
    """Point arms (baseline, label-oracle calibration, PCA-k arms, full-dim, shuffled). Caches C_Z +
    the per-video baseline-correctness vector so the perm null can resume without recomputing them."""
    n = len(y)
    C_Z, cacc = _ci_pick_C(Z, y)
    C_full, cfacc = _ci_pick_C_combined(Z, A, y)
    base = _ci_baseline_cor(Z, y, C_Z); accZ = float(base.mean())
    orac = _ci_oracle_cor(Z, y, C_Z); accZA_lab = float(orac.mean())
    headroom = 1.0 - accZ
    calib = {"label_accZA": accZA_lab, "headroom_1_minus_accZ": float(headroom),
             "label_dacc": float((orac - base).mean()),
             "headroom_fraction": float((orac - base).mean() / headroom) if headroom > 0 else float("nan"),
             "PASS": bool(accZA_lab >= 0.99)}
    rk = _ci_arm_cor_allk(Z, y, C_Z, A, CI_KS_REPORT)
    full = _ci_full_cor(Z, A, y, C_full)
    A_shuf = A[np.random.default_rng(CI_SHUFFLE_SEED).permutation(n)]
    sk = _ci_arm_cor_allk(Z, y, C_Z, A_shuf, CI_KS_DECISION)
    arms = {}
    for k in CI_KS_REPORT:
        m, ci = _ci_boot_ci(rk[k], base, CI_BOOT_SEED + k)
        arms["pca_k%d" % k] = {"accZA": float(rk[k].mean()), "dacc": m, "ci": ci}
    m, ci = _ci_boot_ci(full, base, CI_BOOT_SEED + 999)
    arms["full_cvC"] = {"accZA": float(full.mean()), "dacc": m, "ci": ci}
    for k in CI_KS_DECISION:
        arms["shuffled_k%d" % k] = {"dacc": float((sk[k] - base).mean())}
    real_maxdec = float(max(arms["pca_k%d" % k]["dacc"] for k in CI_KS_DECISION))
    best_k = max(CI_KS_DECISION, key=lambda k: arms["pca_k%d" % k]["dacc"])
    best = arms["pca_k%d" % best_k]
    return {"n": n, "n_pos": int(y.sum()), "Z_dim": int(Z.shape[1]),
            "C_Z": C_Z, "C_Z_cv_acc": float(cacc), "C_full": C_full, "C_full_cv_acc": float(cfacc),
            "baseline_accZ": accZ, "calibration": calib, "arms": arms,
            "real_max_over_kdec": real_maxdec, "best_decision_k": best_k,
            "best_dacc": best["dacc"], "best_ci": best["ci"],
            "C1_point_ge_bar": bool(best["dacc"] >= CI_BAR), "C2_ci_low_gt_0": bool(best["ci"][0] > 0.0),
            "base": base.tolist()}


def conditional_info_probe(Z, A, y, variant_name, run_perm, cell, save_cb):
    """C3-template conditional-info probe of A on Z. Point arms are cached in `cell`; if run_perm, the
    >=150-perm null of A across videos is RESUMABLE (checkpointed via save_cb every 10 seeds). The
    +0.040 triple-rule verdict is assembled here; run_perm=False (secondary Qwen-only context, per the
    c3_fusion_probe PERM_CELLS precedent) reports point arms only (VERDICT SECONDARY_NO_PERM)."""
    if "point" not in cell:
        cell["point"] = _ci_point(Z, A, y)
        cell["variant"] = variant_name
        save_cb()
    pt = cell["point"]
    C_Z = pt["C_Z"]; base = np.asarray(pt["base"], dtype=np.float64)
    real_maxdec = pt["real_max_over_kdec"]
    perm_null = None
    C3 = None
    if run_perm:
        perm = cell.setdefault("perm", {"maxk": [], "perk": {str(k): [] for k in CI_KS_DECISION}})
        for si in range(len(perm["maxk"]), CI_NSEED_PERM):
            p = np.random.default_rng(CI_PERM_BASE + si).permutation(len(y))
            c = _ci_arm_cor_allk(Z, y, C_Z, A[p], CI_KS_DECISION)
            dk = {k: float((c[k] - base).mean()) for k in CI_KS_DECISION}
            for k in CI_KS_DECISION:
                perm["perk"][str(k)].append(dk[k])
            perm["maxk"].append(float(max(dk.values())))
            if (si + 1) % 10 == 0 or si == CI_NSEED_PERM - 1:
                save_cb()
        am = np.array(perm["maxk"])
        perm_null = {"n_seed": len(am), "maxk_mean": float(am.mean()), "maxk_max": float(am.max()),
                     "maxk_q": [float(np.percentile(am, q)) for q in (2.5, 50, 97.5)],
                     "p_realmax_ge_permmax": float((am >= real_maxdec).mean()),
                     "real_beats_all_permmax": bool(real_maxdec > am.max()),
                     "perk_stats": {k: {"mean": float(np.mean(v)), "max": float(np.max(v))}
                                    for k, v in perm["perk"].items()}}
        C3 = bool(perm_null["real_beats_all_permmax"])
    C1, C2 = pt["C1_point_ge_bar"], pt["C2_ci_low_gt_0"]
    if not pt["calibration"]["PASS"]:
        verdict = "MACHINERY_INVALID"
    elif not run_perm:
        verdict = "SECONDARY_NO_PERM"
    elif C1 and C2 and C3:
        verdict = "CONDINFO_PROCEED"
    else:
        verdict = "GROUNDED_DEAD_AT_ZBEST"
    return {"variant": variant_name, "n": pt["n"], "n_pos": pt["n_pos"], "Z_dim": pt["Z_dim"],
            "C_Z": C_Z, "C_Z_cv_acc": pt["C_Z_cv_acc"], "C_full": pt["C_full"],
            "C_full_cv_acc": pt["C_full_cv_acc"], "baseline_accZ": pt["baseline_accZ"],
            "calibration": pt["calibration"], "arms": pt["arms"],
            "real_max_over_kdec": real_maxdec, "best_decision_k": pt["best_decision_k"],
            "best_dacc": pt["best_dacc"], "best_ci": pt["best_ci"],
            "C1_point_ge_bar": C1, "C2_ci_low_gt_0": C2, "C3_real_beats_all_permmax": C3,
            "perm_null": perm_null, "VERDICT": verdict}


# ===========================================================================
# Per-dataset probe.
# ===========================================================================
def probe_dataset(dataset, grounded_dir, n_boot, ci_ckpt, save_cb):
    mem = load_memory(dataset, grounded_dir)
    y = mem["labels"]
    M = build_matrices(mem)

    arms = {}

    def add(name, S):
        acc, mf1, roc, votes = run_vote(S, y)
        arms[name] = {"acc": acc, "macro_f1": mf1, "roc": roc, "votes": votes}
        return arms[name]

    a_pool = add("POOLED_IMG", M["pooled_img"])
    a_cat = add("CONCAT", M["concat"])
    a_grd = add("GROUNDED", M["grd"])
    add("GROUNDED_TEXT", M["grd_text"])
    add("GROUNDED_PFX", M["grd_pfx"])
    S_pca, pca_dim = concat_pca_matrix(mem)
    a_pca = add("CONCAT_PCA", S_pca)
    S_alpha, best_alpha, alpha_accs = concat_alpha_matrix(mem, y)
    a_alpha = add("CONCAT_ALPHA", S_alpha)

    # advisory raw deltas (GROUNDED - CONCAT), and sign vs CONCAT-PCA / CONCAT-alpha.
    d_acc = a_grd["acc"] - a_cat["acc"]
    d_f1 = a_grd["macro_f1"] - a_cat["macro_f1"]
    beat_pca_sign = bool(a_grd["acc"] > a_pca["acc"])
    beat_alpha_sign = bool(a_grd["acc"] > a_alpha["acc"])
    # rank-only co-diagnostic (sim neutralised to 1.0): does the GROUNDED advantage survive when the
    # similarity SCALE is removed (pure rank structure)? Advisory, mirrors S2S A2.
    gr_acc, gr_f1, _, gr_votes = run_vote(M["grd"], y, rank_only=True)
    cr_acc, cr_f1, _, cr_votes = run_vote(M["concat"], y, rank_only=True)
    d_acc_rank = gr_acc - cr_acc
    d_f1_rank = gr_f1 - cr_f1
    rank_sign_ok = bool(np.sign(d_acc) == np.sign(d_acc_rank) and np.sign(d_f1) == np.sign(d_f1_rank))
    boot_rank = bootstrap_delta(gr_votes, cr_votes, y, n_boot, seed=20260716)

    # K4 Fano, K5 oracle-ceiling (grd vs CONCAT), A3 near-dup, K7 null, K8 bootstrap.
    fano_acc = fano(y)
    orc_acc, orc_f1, orc_frac = oracle_ceiling(a_grd["votes"], a_cat["votes"], y)
    d_orc_acc = orc_acc - a_cat["acc"]
    d_orc_f1 = orc_f1 - a_cat["macro_f1"]

    flag, nd_dist, nd_pairs = near_dup_audit(M, mem["guard"])
    ax_grd = run_vote(M["grd"], y, exclude=flag)
    ax_cat = run_vote(M["concat"], y, exclude=flag)
    d_acc_x = ax_grd[0] - ax_cat[0]
    d_f1_x = ax_grd[1] - ax_cat[1]

    null = permutation_null(M, y)
    boot = bootstrap_delta(a_grd["votes"], a_cat["votes"], y, n_boot)

    # BINDING K9 conditional-info probe: A=grd on Z_best (8960, run_perm=True binding); secondary
    # Qwen-only (7168) reported point-arms-only (run_perm=False, per c3_fusion_probe PERM_CELLS).
    A_grd = mem["grd"].numpy().astype(np.float64)

    def cell(variant):
        return ci_ckpt.setdefault("{}|{}".format(dataset, variant), {})

    ci_zbest = conditional_info_probe(build_Zbest(mem), A_grd, y, "Z_best_8960",
                                      run_perm=True, cell=cell("Z_best_8960"), save_cb=save_cb)
    ci_qwen = conditional_info_probe(build_Zqwen(mem), A_grd, y, "Qwen_only_7168",
                                     run_perm=False, cell=cell("Qwen_only_7168"), save_cb=save_cb)

    # Amdt-5 covered-rows-only HateMM secondary (non-empty transcript). MHC 100% coverage -> None.
    covered = None
    if int(mem["empty"].sum()) > 0:
        cov_mask = (~mem["empty"]) & (~mem["guard"])
        cov_acc_g, cov_f1_g, _, _ = run_vote(M["grd"], y, subset=cov_mask)
        cov_acc_c, cov_f1_c, _, _ = run_vote(M["concat"], y, subset=cov_mask)
        sel = np.where(cov_mask)[0]
        ci_cov = conditional_info_probe(build_Zbest(mem)[sel], A_grd[sel], y[sel], "Z_best_covered",
                                        run_perm=True, cell=cell("Z_best_covered"), save_cb=save_cb)
        covered = {"n_covered": int(cov_mask.sum()),
                   "knn_d_acc": cov_acc_g - cov_acc_c, "knn_d_f1": cov_f1_g - cov_f1_c,
                   "grd_acc": cov_acc_g, "concat_acc": cov_acc_c,
                   "condinfo": _ci_summary(ci_cov)}

    arm_summary = {k: {kk: v[kk] for kk in ("acc", "macro_f1", "roc")} for k, v in arms.items()}
    return {
        "dataset": dataset, "N": mem["N"], "zero_guard": int(mem["guard"].sum()),
        "empty_transcript": int(mem["empty"].sum()),
        "arms": arm_summary,
        "concat_pca_dim": pca_dim, "concat_alpha_best": best_alpha, "concat_alpha_accs": alpha_accs,
        "advisory_primary": {"d_acc": d_acc, "d_f1": d_f1,
                             "beat_concat_pca_sign": beat_pca_sign,
                             "beat_concat_alpha_sign": beat_alpha_sign,
                             "rankonly_d_acc": d_acc_rank, "rankonly_d_f1": d_f1_rank,
                             "rankonly_sign_ok": rank_sign_ok,
                             "rankonly_obs_dacc": null["obs_dacc_rank"],
                             "rankonly_null_p95_acc": null["null_dacc_rank_p95"],
                             "rankonly_obs_gt_p95": null["obs_dacc_rank_gt_p95"],
                             "rankonly_boot_dacc_p5": boot_rank["dacc_p5"]},
        "bootstrap_rankonly": boot_rank,
        "fano_acc": fano_acc,
        "oracle": {"acc": orc_acc, "macro_f1": orc_f1, "d_acc": d_orc_acc, "d_f1": d_orc_f1,
                   "frac_choose_grd": orc_frac},
        "near_dup": {"threshold": NEAR_DUP_THRESH, "flagged_pairs": nd_pairs,
                     "distribution": nd_dist, "excluded_d_acc": d_acc_x, "excluded_d_f1": d_f1_x},
        "permutation_null": null,
        "bootstrap": boot,
        "condinfo_Zbest": _ci_summary(ci_zbest),
        "condinfo_Qwen_secondary": _ci_summary(ci_qwen),
        "covered_rows_secondary": covered,
        "stage_e_gatelog": read_stage_e_gatelog(dataset, grounded_dir),
    }


def _ci_summary(ci):
    """Compact JSON summary of a conditional-info probe run (raw numbers only)."""
    return {"variant": ci["variant"], "Z_dim": ci["Z_dim"], "C_Z": ci["C_Z"],
            "baseline_accZ": ci["baseline_accZ"],
            "calibration": ci["calibration"],
            "best_decision_k": ci["best_decision_k"], "best_dacc": ci["best_dacc"],
            "best_ci": ci["best_ci"], "real_max_over_kdec": ci["real_max_over_kdec"],
            "arms": {k: {"accZA": v.get("accZA"), "dacc": v["dacc"], "ci": v.get("ci")}
                     for k, v in ci["arms"].items()},
            "C1_point_ge_bar": ci["C1_point_ge_bar"], "C2_ci_low_gt_0": ci["C2_ci_low_gt_0"],
            "C3_real_beats_all_permmax": ci["C3_real_beats_all_permmax"],
            "perm_null": ci["perm_null"], "VERDICT": ci["VERDICT"]}


def read_stage_e_gatelog(dataset, grounded_dir):
    out = {}
    for outname in ("train", "dev_seen"):
        p = os.path.join(REPO, "data/CLIP_Embedding", dataset, grounded_dir,
                         "{}_gatelog.json".format(outname))
        if os.path.exists(p):
            out[outname] = json.load(open(p))
    return out


# ===========================================================================
# Mechanical gate check (NON-binding — pre-registered threshold arithmetic ONLY).
# ===========================================================================
def _stage_e_void(r, prefix):
    """Read a Stage-E' VOID flag (prefix 'grounding_void' / 'placebo_void') from the extractor gatelog
    for the probe memory splits (train + dev_seen). Returns True if EITHER split tripped it, False if
    present-and-not-tripped, None if no gatelog carries the flag."""
    ge = r.get("stage_e_gatelog") or {}
    seen = tripped = False
    for sp in ("train", "dev_seen"):
        g = ge.get(sp)
        if not isinstance(g, dict):
            continue
        vk = [k for k in g if k.startswith(prefix)]
        if vk:
            seen = True
            if bool(g[vk[0]]):
                tripped = True
    return tripped if seen else None


def mechanical_gate_check(results):
    by_ds = {r["dataset"]: r for r in results}
    checks = []

    def rec(name, value, thr, op, note=""):
        ok = (value >= thr) if op == ">=" else (value < thr) if op == "<" else (value > thr) \
            if op == ">" else (value == thr)
        checks.append({"gate": name, "value": value, "threshold": thr, "op": op,
                       "result": "ABOVE" if ok else "BELOW", "note": note})

    # K4 Fano machine-validity.
    for ds, r in by_ds.items():
        rec("Fano[%s] (K4)" % ds, r["fano_acc"], FANO_BAR, ">=", "vote machine valid if ABOVE")

    # K5 oracle-ceiling kill-switch (DEAD only if ALL datasets below +0.04).
    oracle_all_below = all(r["oracle"]["d_acc"] < ORACLE_BAR for r in results)
    for ds, r in by_ds.items():
        rec("OracleDacc[%s] (K5)" % ds, r["oracle"]["d_acc"], ORACLE_BAR, ">=",
            "headroom if ABOVE (kill-switch DEAD only if ALL datasets BELOW)")
    checks.append({"gate": "OracleKillSwitch(all-datasets) (K5)", "value": oracle_all_below,
                   "threshold": "all < %.2f" % ORACLE_BAR,
                   "result": "KILL(DEAD)" if oracle_all_below else "SURVIVES",
                   "note": "DEAD -> zero head GPU"})

    # (Fix B) K2/K3 Stage-E' VOID surfacing — a silent no-op grounding (present-set median
    # cos(grd,ungrd_vis)>=0.999) or a content-insensitive key (placebo median>=0.999) NULLIFIES any K9
    # PROCEED. Read the extractor gatelog flags (train + dev_seen, the probe memory splits); a dataset
    # is VOID if EITHER split tripped the flag.
    grounding_void = {ds: _stage_e_void(r, "grounding_void") for ds, r in by_ds.items()}
    placebo_void = {ds: _stage_e_void(r, "placebo_void") for ds, r in by_ds.items()}
    for ds in by_ds:
        gv, pv = grounding_void[ds], placebo_void[ds]
        checks.append({"gate": "GroundingLive[%s] (K2)" % ds,
                       "value": ("VOID" if gv else "LIVE" if gv is False else "N/A"),
                       "threshold": "LIVE", "result": ("VOID" if gv else "LIVE" if gv is False else "N/A"),
                       "note": "present-set median cos(grd,ungrd_vis)>=0.999 -> VOID nullifies K9"})
        checks.append({"gate": "Placebo[%s] (K3)" % ds,
                       "value": ("VOID" if pv else "LIVE" if pv is False else "N/A"),
                       "threshold": "LIVE", "result": ("VOID" if pv else "LIVE" if pv is False else "N/A"),
                       "note": "cross-video mismatched transcript no-op (median>=0.999) -> VOID nullifies K9"})

    # K9 BINDING conditional-info adjudicator (per dataset; sole binding performance gate).
    for ds, r in by_ds.items():
        ci = r["condinfo_Zbest"]
        voided = bool(grounding_void[ds]) or bool(placebo_void[ds])
        checks.append({"gate": "CondInfo Z_best VERDICT[%s] (K9 BINDING)" % ds,
                       "value": ci["VERDICT"], "threshold": "CONDINFO_PROCEED",
                       "result": ("VOID(K2/K3-nullified)" if (ci["VERDICT"] == "CONDINFO_PROCEED" and voided)
                                  else "ABOVE" if ci["VERDICT"] == "CONDINFO_PROCEED"
                                  else "MACHINERY_INVALID" if ci["VERDICT"] == "MACHINERY_INVALID"
                                  else "BELOW"),
                       "note": "calib_accZA={:.4f} bestk={} dacc={:+.4f} CI[{:+.4f},{:+.4f}] "
                               "C1={} C2={} C3={} perm_maxk_max={:+.4f}; grounding_void={} placebo_void={} "
                               "(VOID nullifies K9)".format(
                                   ci["calibration"]["label_accZA"], ci["best_decision_k"],
                                   ci["best_dacc"], ci["best_ci"][0], ci["best_ci"][1],
                                   ci["C1_point_ge_bar"], ci["C2_ci_low_gt_0"],
                                   ci["C3_real_beats_all_permmax"], ci["perm_null"]["maxk_max"],
                                   grounding_void[ds], placebo_void[ds])})
    # SURVIVES requires a CONDINFO_PROCEED on a dataset whose grounding is LIVE AND placebo is LIVE.
    any_ci_pass = any(r["condinfo_Zbest"]["VERDICT"] == "CONDINFO_PROCEED"
                      and not grounding_void[ds] and not placebo_void[ds]
                      for ds, r in by_ds.items())
    checks.append({"gate": "CondInfo BINDING (any LIVE dataset PROCEED) (K9)", "value": any_ci_pass,
                   "threshold": True, "result": "SURVIVES" if any_ci_pass else "BELOW",
                   "note": "sole binding performance gate (r1 Amdt 2b); a K2/K3-VOID dataset cannot count"})

    # ADVISORY (K6/K7/K7b/K8) — HateMM only, non-gating.
    if "HateMM" in by_ds:
        h = by_ds["HateMM"]; ap = h["advisory_primary"]
        rec("RawDacc[HateMM] (K6 adv)", ap["d_acc"], RAW_BAR, ">=", "advisory, non-gating")
        rec("RawDmF1[HateMM] (K6 adv)", ap["d_f1"], RAW_BAR, ">=", "advisory, non-gating")
        checks.append({"gate": "BeatConcatPCA+alpha sign[HateMM] (K6 adv)",
                       "value": bool(ap["beat_concat_pca_sign"] and ap["beat_concat_alpha_sign"]),
                       "threshold": True,
                       "result": "ABOVE" if (ap["beat_concat_pca_sign"] and ap["beat_concat_alpha_sign"])
                       else "BELOW", "note": "advisory"})
        rec("ObsDacc>null95[HateMM] (K7 adv)", h["permutation_null"]["obs_dacc"],
            h["permutation_null"]["null_dacc_p95"], ">", "advisory")
        rec("NearDupExclSurvives[HateMM] (K7b adv)", h["near_dup"]["excluded_d_acc"], 0.0, ">", "advisory")
        rec("Bootstrap5th>0[HateMM] (K8 adv)", h["bootstrap"]["dacc_p5"], 0.0, ">", "advisory")
        # rank-only co-diagnostic corroboration (advisory): sign match AND own null AND own bootstrap.
        rank_corrob = bool(ap["rankonly_sign_ok"] and ap["rankonly_obs_gt_p95"]
                           and ap["rankonly_boot_dacc_p5"] > 0.0)
        checks.append({"gate": "RankOnlyCorroborates[HateMM] (adv)", "value": rank_corrob,
                       "threshold": True, "result": "ABOVE" if rank_corrob else "BELOW",
                       "note": "advisory; sign={} null_gt_p95={} boot5th>0={}".format(
                           ap["rankonly_sign_ok"], ap["rankonly_obs_gt_p95"],
                           ap["rankonly_boot_dacc_p5"] > 0.0)})
    return checks, oracle_all_below, any_ci_pass


# ===========================================================================
# Synthetic end-to-end self-test — drive the REAL vote + logistic machinery on fake data.
# ===========================================================================
def synthetic_self_test():
    _log("[self-test] driving real vote + conditional-info machinery on synthetic data ...")
    rng = np.random.default_rng(11)
    N, D = 120, 64
    y = np.array([0, 1] * (N // 2))
    # GROUNDED planted to beat CONCAT: grd separates the classes; concat is near-noise.
    grd = rng.standard_normal((N, D)) * 0.1
    grd[:, 0] += (y * 2 - 1) * 1.5
    concat = rng.standard_normal((N, D)) * 1.0
    S_grd = _cos_mat(torch.tensor(grd)); S_cat = _cos_mat(torch.tensor(concat))
    ga, gf, _, gv = run_vote(S_grd, y)
    ca, cf, _, cv = run_vote(S_cat, y)
    assert ga > ca, "self-test: planted grd should beat noise concat (acc {:.3f} vs {:.3f})".format(ga, ca)
    # Fano must be ~1.0 on separable labels.
    fa = fano(y)
    assert fa >= 0.99, "self-test: Fano {:.3f} < 0.99 on ideal label key".format(fa)
    # oracle-ceiling >= concat by construction; and >0 headroom when grd helps.
    oa, of, frac = oracle_ceiling(gv, cv, y)
    assert oa + 1e-9 >= ca, "self-test: oracle {:.3f} < concat {:.3f}".format(oa, ca)
    # subset (covered-only) path runs and returns a finite acc.
    mask = np.ones(N, dtype=bool); mask[:10] = False
    sa, sf, _, _ = run_vote(S_grd, y, subset=mask)
    assert 0.0 <= sa <= 1.0
    # rank-only co-diagnostic path runs (sim neutralised to 1.0).
    ra, rf, _, rv = run_vote(S_grd, y, rank_only=True)
    assert 0.0 <= ra <= 1.0 and len(rv) == N
    # oracle KILL branch: when grd carries nothing (grd == concat noise), oracle Delta ~ 0.
    S_noise = _cos_mat(torch.tensor(rng.standard_normal((N, D))))
    na, nf, _, nv = run_vote(S_noise, y)
    oa2, _, _ = oracle_ceiling(nv, cv, y)
    _log("[self-test] noise-grd oracle Delta(acc) vs concat = {:+.4f} (kill branch exercised)".format(
        oa2 - ca))
    # conditional-info machinery: calibration arm must reach ~1.0; a planted-informative A passes C1.
    Z = rng.standard_normal((N, 40))
    A = np.zeros((N, 8)); A[:, 0] = (y * 2 - 1) * 3.0 + rng.standard_normal(N) * 0.1  # A carries the label
    ci = conditional_info_probe(Z, A, y, "selftest", run_perm=True, cell={}, save_cb=lambda: None)
    assert ci["calibration"]["PASS"], "self-test: label-oracle calibration failed to reach 0.99"
    assert ci["best_dacc"] > 0.0, "self-test: informative A produced non-positive conditional dacc"
    assert ci["perm_null"] is not None and ci["C3_real_beats_all_permmax"] is not None
    _log("[self-test] PASS — vote(planted grd>concat), Fano={:.3f}, oracle>=concat, subset path, "
         "CI calib PASS + informative-A dacc={:+.4f}, VERDICT={}.".format(
             fa, ci["best_dacc"], ci["VERDICT"]))


# ===========================================================================
# Output.
# ===========================================================================
def write_markdown(results, checks, out_md):
    L = ["# W2-A Stage-P' Probe — RAW RESULTS (no pass/fail interpretation)\n",
         "_Executor writes RAW numbers only; an independent verdict reviewer renders the binding "
         "ruling (house rule). The BINDING performance gate is the K9 conditional-info probe vs "
         "Z_best (8960-d); the kNN GROUNDED-CONCAT bar is ADVISORY (r1 Amdt 2b). The mechanical gate "
         "arithmetic below is NOT a verdict._\n"]
    for r in results:
        ds = r["dataset"]
        L.append("\n## {}  (memory N={}, zero-guard={}, empty-transcript={})\n".format(
            ds, r["N"], r["zero_guard"], r["empty_transcript"]))
        L.append("| arm | acc | macro_f1 | roc |")
        L.append("|---|---|---|---|")
        for name, a in r["arms"].items():
            L.append("| {} | {:.4f} | {:.4f} | {:.4f} |".format(name, a["acc"], a["macro_f1"], a["roc"]))
        ci = r["condinfo_Zbest"]
        L.append("\n**BINDING K9 — conditional-info of grd on Z_best ({}d):** calib accZA {:.4f} "
                 "(PASS={}); best decision-k={} Δacc {:+.4f} CI[{:+.4f},{:+.4f}]; C1(≥{:+.3f})={} "
                 "C2(CI>0)={} C3(real>all perm)={}; perm-null maxk mean {:+.4f} max {:+.4f} "
                 "(n={}); **VERDICT={}**.".format(
                     ci["Z_dim"], ci["calibration"]["label_accZA"], ci["calibration"]["PASS"],
                     ci["best_decision_k"], ci["best_dacc"], ci["best_ci"][0], ci["best_ci"][1],
                     CI_BAR, ci["C1_point_ge_bar"], ci["C2_ci_low_gt_0"], ci["C3_real_beats_all_permmax"],
                     ci["perm_null"]["maxk_mean"], ci["perm_null"]["maxk_max"],
                     ci["perm_null"]["n_seed"], ci["VERDICT"]))
        cq = r["condinfo_Qwen_secondary"]
        L.append("**Secondary context — conditional-info of grd on Qwen-only ({}d):** best Δacc "
                 "{:+.4f} CI[{:+.4f},{:+.4f}] VERDICT={} (reported, NON-binding).".format(
                     cq["Z_dim"], cq["best_dacc"], cq["best_ci"][0], cq["best_ci"][1], cq["VERDICT"]))
        o = r["oracle"]
        L.append("\n**K5 oracle-ceiling (grd vs CONCAT, tie→CONCAT):** acc {:.4f} "
                 "(Δ vs CONCAT acc {:+.4f}, mF1 {:+.4f}); chose-grd fraction {:.3f}.".format(
                     o["acc"], o["d_acc"], o["d_f1"], o["frac_choose_grd"]))
        L.append("**K4 Fano (±1 gold-label key) acc:** {:.4f}.".format(r["fano_acc"]))
        ap = r["advisory_primary"]
        L.append("\n**ADVISORY kNN Δ(GROUNDED−CONCAT):** acc {:+.4f}, macro_f1 {:+.4f}; beat "
                 "CONCAT-PCA(dim={})-sign={}, beat CONCAT-α(α*={})-sign={}.".format(
                     ap["d_acc"], ap["d_f1"], r["concat_pca_dim"], ap["beat_concat_pca_sign"],
                     r["concat_alpha_best"], ap["beat_concat_alpha_sign"]))
        L.append("**ADVISORY rank-only co-diagnostic (sim→1.0):** Δacc {:+.4f}, ΔmF1 {:+.4f} "
                 "(sign-matches sim-weighted={}); obs Δacc {:+.4f} vs rank-only null-95th {:+.4f} "
                 "(gt={}), rank-only bootstrap-5th {:+.4f}.".format(
                     ap["rankonly_d_acc"], ap["rankonly_d_f1"], ap["rankonly_sign_ok"],
                     ap["rankonly_obs_dacc"], ap["rankonly_null_p95_acc"], ap["rankonly_obs_gt_p95"],
                     ap["rankonly_boot_dacc_p5"]))
        nl = r["permutation_null"]
        L.append("**ADVISORY perm-null ({} seeds):** obs Δacc {:+.4f} vs null-95th {:+.4f} "
                 "(gt={}); obs ΔmF1 {:+.4f} vs null-95th {:+.4f}.".format(
                     nl["n_seeds"], nl["obs_dacc"], nl["null_dacc_p95"], nl["obs_dacc_gt_p95"],
                     nl["obs_df1"], nl["null_df1_p95"]))
        b = r["bootstrap"]
        L.append("**ADVISORY bootstrap ({}):** Δacc [5/50/95]=[{:+.4f}/{:+.4f}/{:+.4f}]; "
                 "ΔmF1 [5/50/95]=[{:+.4f}/{:+.4f}/{:+.4f}].".format(
                     b["n_boot"], b["dacc_p5"], b["dacc_p50"], b["dacc_p95"],
                     b["df1_p5"], b["df1_p50"], b["df1_p95"]))
        nd = r["near_dup"]
        L.append("**A3 near-dup:** flagged pairs (≥{:.3f} grd-OR-concat) = {}; excluded-retrieval "
                 "Δ(GROUNDED−CONCAT) acc {:+.4f}, mF1 {:+.4f}. Distribution: {}.".format(
                     nd["threshold"], nd["flagged_pairs"], nd["excluded_d_acc"], nd["excluded_d_f1"],
                     json.dumps(nd["distribution"])))
        cov = r["covered_rows_secondary"]
        if cov:
            cc = cov["condinfo"]
            L.append("**Amdt-5 covered-rows-only (non-empty transcript, n={}):** advisory kNN "
                     "Δacc {:+.4f} ΔmF1 {:+.4f}; binding K9 best Δacc {:+.4f} CI[{:+.4f},{:+.4f}] "
                     "VERDICT={}.".format(cov["n_covered"], cov["knn_d_acc"], cov["knn_d_f1"],
                                          cc["best_dacc"], cc["best_ci"][0], cc["best_ci"][1],
                                          cc["VERDICT"]))
        ge = r["stage_e_gatelog"]
        if ge:
            frags = []
            for sp, g in ge.items():
                gp = g.get("grounding_present") or {}
                vk = [k for k in g if k.startswith("grounding_void")]
                pk = [k for k in g if k.startswith("placebo_void")]
                frags.append("{}: grecon_cos_min={} grounding_present_median={} {}={} placebo_median={} "
                             "{}={}".format(sp, g.get("grecon_cos_min"), gp.get("median"),
                                            (vk[0] if vk else "grounding_void"),
                                            (g.get(vk[0]) if vk else None),
                                            (g.get("placebo") or {}).get("median"),
                                            (pk[0] if pk else "placebo_void"),
                                            (g.get(pk[0]) if pk else None)))
            L.append("**Stage-E' gates:** " + "; ".join(frags) + ".")
    L.append("\n## Mechanical gate arithmetic (NOT a verdict — see JSON)\n")
    L.append("| gate | value | threshold | op | result |")
    L.append("|---|---|---|---|---|")
    for c in checks:
        L.append("| {} | {} | {} | {} | {} |".format(
            c["gate"], c["value"], c["threshold"], c.get("op", ""), c["result"]))
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")


def _jsonable(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not JSON serializable: {}".format(type(o)))


def main():
    ap = argparse.ArgumentParser(description="W2-A Stage-P' CPU probe (zero training, zero test touch).")
    ap.add_argument("--datasets", type=str, default="HateMM,MHC")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out_md", type=str, default=os.path.join(REPO, "refine-logs/W2A_PROBE_RESULTS.md"))
    ap.add_argument("--out_json", type=str, default=os.path.join(REPO, "refine-logs/w2a_probe_results.json"))
    ap.add_argument("--ci_ckpt", type=str, default=os.path.join(REPO, "refine-logs/w2a_ci_ckpt.json"),
                    help="Resumable checkpoint for the (multi-hour) K9 conditional-info perm-null.")
    ap.add_argument("--self_test_only", action="store_true", help="Run the synthetic self-test and exit.")
    args = ap.parse_args()

    torch.manual_seed(20260715)
    np.random.seed(20260715)

    _log("=" * 78)
    _log("[W2-A probe] datasets={} topk={} null_seeds={} n_boot={} CI_NSEED={} Z_best=8960".format(
        args.datasets, TOPK, len(NULL_SEEDS), args.n_boot, CI_NSEED_PERM))
    _log("[W2-A probe] N4 fail-closed: train+dev_seen ONLY; expected memory {}".format(EXPECTED_MEM))
    _log("=" * 78)

    synthetic_self_test()   # HALT on any machinery bug before touching real caches.
    if args.self_test_only:
        _log("[W2-A probe] self-test only; exiting.")
        return

    grounded_dir = GROUNDED_DIR_TMPL.format(args.num_frames)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # Resumable K9 checkpoint (perm-null is a multi-hour CPU stage). A config signature invalidates a
    # stale checkpoint so a changed Z_best/perm-count/grounded_dir never silently reuses old seeds.
    # (Fix A) The signature ALSO hashes this probe script + the per-dataset grounded caches (the
    # grd/grd_pfx keys A=grd is built from), so a re-extraction into the SAME grounded_dir or any edit
    # to the probe re-derives a fresh checkpoint instead of silently reusing stale point-arms + seeds.
    grd_sha = {ds: "+".join(_sha256_file(_grounded_path(ds, grounded_dir, o))
                            for o in ("train", "dev_seen")) for ds in datasets}
    ci_meta = {"grounded_dir": grounded_dir, "ci_nseed": CI_NSEED_PERM, "Zbest_dim": 8960,
               "probe_sha": _sha256_file(os.path.abspath(__file__)), "grd_sha": grd_sha}
    ci_ckpt = {"_meta": ci_meta}
    if os.path.exists(args.ci_ckpt):
        loaded = json.load(open(args.ci_ckpt))
        if loaded.get("_meta") == ci_meta:
            ci_ckpt = loaded
            _log("[W2-A probe] resuming K9 checkpoint {}".format(args.ci_ckpt))
        else:
            _log("[W2-A probe] K9 checkpoint config mismatch -> starting fresh (old meta={})".format(
                loaded.get("_meta")))

    def save_cb():
        tmp = args.ci_ckpt + ".tmp"
        with open(tmp, "w") as f:
            json.dump(ci_ckpt, f, default=_jsonable)
        os.replace(tmp, args.ci_ckpt)

    t0 = time.time()
    results = []
    for ds in datasets:
        _log("[W2-A probe] --- {} ---".format(ds))
        results.append(probe_dataset(ds, grounded_dir, args.n_boot, ci_ckpt, save_cb))
    _log("[W2-A probe] probing done in {:.1f}s".format(time.time() - t0))

    checks, oracle_dead, any_ci_pass = mechanical_gate_check(results)
    _log("\n[W2-A probe] MECHANICAL gate arithmetic (pre-registered thresholds; NOT the binding "
         "verdict — the independent verdict reviewer rules):")
    for c in checks:
        _log("  GATE {:<44} value={} threshold={} {} -> {}".format(
            c["gate"], c["value"], c["threshold"], c.get("op", ""), c["result"]))

    out = {
        "meta": {"topk": TOPK, "grounded_dir": grounded_dir, "null_seeds": len(NULL_SEEDS),
                 "n_boot": args.n_boot, "ci_nseed_perm": CI_NSEED_PERM, "expected_mem": EXPECTED_MEM,
                 "Zbest_dim": 8960, "ci_bar": CI_BAR, "oracle_bar": ORACLE_BAR, "fano_bar": FANO_BAR,
                 "raw_bar_advisory": RAW_BAR, "near_dup_threshold": NEAR_DUP_THRESH,
                 "alpha_grid": ALPHA_GRID,
                 "note": "Executor writes RAW numbers. mechanical_gate_check is pre-registered "
                         "threshold ARITHMETIC ONLY and is NOT the binding verdict; an independent "
                         "verdict reviewer renders the ruling. BINDING perf gate = K9 (Z_best); the "
                         "kNN GROUNDED-CONCAT bar is ADVISORY (r1 Amdt 2b)."},
        "results": results,
        "mechanical_gate_check": checks,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=_jsonable)
    write_markdown(results, checks, args.out_md)
    _log("[W2-A probe] wrote {} and {}".format(args.out_md, args.out_json))


if __name__ == "__main__":
    main()
