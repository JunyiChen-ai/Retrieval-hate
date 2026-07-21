#!/usr/bin/env python3
"""ISR $0 pre-gate — operator #4 (per-segment-kNN vote-mean) on banked frame-local CLIP + selection-ceiling
arithmetic. CPU-only, ZERO GPU/SLURM/Modal, ZERO test-touch, ZERO training.

Binding skeleton : refine-logs/SEG_REENCODE_FORENSIC_RECON.md (31bcd03) §3 (gate) + §5 (frozen bars).
Pre-declared bars: refine-logs/ISR_PREGATE_RECORD.md §0 (written BEFORE any dev number — forking-path).
Machinery        : reused verbatim from scripts/analysis/w2b_probe.py (load_memory / build_matrices /
                   run_vote → compute_metrics_retrieval / _single_query_vote_margins / oracle_ceiling /
                   fano). NO vote is reimplemented; op#4 == uniform-mean of the frozen per-segment votes.

GATE α (survival-determining PRIMARY = video-level LOO over train∪dev_seen, apples-to-apples with banked
W2-B): op#4 v(Q)=mean_t V_t(Q), V_t = deployed top-20 rank-weighted signed-cosine vote of query-segment t
vs memory videos (s_t(Q,j)=max_m cos(ghat_{Q,t},ghat_{j,m})); predict v(Q)>=0. Baseline = pooled-key one-hop
kNN (run_vote(spool)). CORROBORATING arm = strict dev-query→train-memory (dev items only, no LOO).
Bar: PROMOTE iff mean dev Δacc(op#4−POOLED) >= +0.030 on >=1 dataset (either arm); else KILL.

GATE β: decompose banked W2-B oracle headroom into symmetric slice (legal uniform operator) vs selection
slice (banned per-item selection). Arithmetic on refine-logs/w2b_probe_results.json.

FAIL CLOSED: loader (w2b) hard-asserts split ∈ {train,dev_seen}; we additionally assert no 'test' token in
any path we open and that memory video-counts equal the exact train(∪dev) counts.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # HARD CPU — no GPU may be touched.

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import w2b_probe as W  # noqa: E402  (the frozen machinery)

TOPK = W.TOPK
MODEL_TAG = W.MODEL_TAG
NEG_INF = W.NEG_INF
EXPECTED_PRIMARY = W.EXPECTED_MEM_PRIMARY          # train∪dev: HateMM 851, MHC 629
EXPECTED_TRAIN = W.EXPECTED_MEM_TRAINONLY          # train:     HateMM 744, MHC 549
EXPECTED_DEV = {"HateMM": 107, "MHC": 80}          # dev_seen (851-744 / 629-549); loader asserts it
PROMOTE_BAR = 0.030                                # team-lead pinned gate-α bar (Δacc, >=1 dataset)
RECON_HATEMM_BAR = 0.05                            # recon §5 stricter ladder (reported only)
RECON_MHC_BAR = 0.03
NULL_SEEDS = list(range(100))                      # >=100 perms
N_BOOT = 1000


def _log(m):
    print(m, flush=True)


def _macro_f1(labels, pred):
    return float(f1_score(labels, pred, average="macro", zero_division=0))


def _assert_no_test(*paths):
    for p in paths:
        if "test" in os.path.basename(p).lower():
            raise RuntimeError("TEST-TOUCH GUARD: refusing to open '{}'".format(p))


# ---------------------------------------------------------------------------
# Per-segment vote margins from a list of K score matrices s_t(i,j) [N,Nmem].
# Byte-identical vote to compute_metrics_retrieval (arithmetic, use_sim) == W._single_query_vote_margins.
# ---------------------------------------------------------------------------
def votemargins(St_list, mem_labels, k=TOPK, loo=True):
    N = St_list[0].shape[0]
    K = len(St_list)
    weight = np.arange(1, k + 1, dtype=np.float64)[::-1]
    wsum = weight.sum()
    lab_signed = mem_labels.astype(np.float64) * 2.0 - 1.0
    Vm = np.zeros((N, K), dtype=np.float64)
    for t, St in enumerate(St_list):
        S = W._tiebreak(St).copy()
        if loo:
            np.fill_diagonal(S, NEG_INF)
        for i in range(N):
            row = S[i]
            idx = np.argpartition(-row, k)[:k]
            idx = idx[np.argsort(-row[idx])]
            idx = idx[row[idx] > (NEG_INF / 2)]
            if idx.size == 0:
                raise RuntimeError("degenerate retrieval for query {}".format(i))
            Vm[i, t] = float((lab_signed[idx] * row[idx].astype(np.float64)
                              * weight[:idx.size]).sum() / wsum)
    return Vm


def st_list_from_C(C):
    """s_t(i,j) = max_m cos(subclip (i,t), subclip (j,m))  -> K matrices [N,N] (square LOO arm)."""
    return [C[:, t, :, :].max(axis=2).astype(np.float64) for t in range(C.shape[1])]


# ---------------------------------------------------------------------------
# Cross retrieval (dev query rows -> train memory cols), no LOO. Reuses the same vote formula.
# ---------------------------------------------------------------------------
def _l2(x):
    return W._l2norm(x, dim=-1)


def cross_st_list(gdv, gtr):
    """s_t(q,j) = max_m cos(ghat_dv[q,t], ghat_tr[j,m]) -> K matrices [Vdv,Vtr]."""
    Vdv, K, D = gdv.shape
    Vtr = gtr.shape[0]
    gh_dv = _l2(gdv)                                     # [Vdv,K,D]
    Gtr = _l2(gtr).reshape(Vtr * K, D)                   # [Vtr*K,D]
    out = []
    for t in range(K):
        s = (gh_dv[:, t, :] @ Gtr.t()).reshape(Vdv, Vtr, K).max(dim=2).values
        out.append(s.numpy().astype(np.float64))
    return out


def cross_pooled(gdv, gtr):
    pdv = _l2(gdv.mean(dim=1))
    ptr = _l2(gtr.mean(dim=1))
    return (pdv @ ptr.t()).numpy().astype(np.float64)    # [Vdv,Vtr]


def cross_vote_acc(S, mem_labels, query_labels, k=TOPK):
    """Single-key cross kNN vote (no LOO): per query row, top-k mem cols, rank-weighted signed vote."""
    Vm = votemargins([S], mem_labels, k, loo=False)      # [Vq,1]
    pred = (Vm[:, 0] >= 0.0).astype(int)
    return float((pred == query_labels).mean()), _macro_f1(query_labels, pred), Vm[:, 0]


# ---------------------------------------------------------------------------
# GATE α — primary (LOO) + corroborating (dev->train) + calibration arms.
# ---------------------------------------------------------------------------
def gate_alpha_primary(ds, data_root):
    exp = EXPECTED_PRIMARY[ds]
    mem = W.load_memory(data_root, ds, "subclipK4", ("train", "dev_seen"), exp, want_pooled=True)
    labels = mem["labels"]
    M = W.build_matrices(mem, want_pooled=True)

    # baseline: pooled-key one-hop kNN (== banked W2-B POOLED)
    pool_acc, pool_mf1, pool_roc, pool_votes = W.run_vote(M["spool"], labels, TOPK)

    # operator #4: uniform-mean of frozen per-segment votes
    Vm_ref = W._single_query_vote_margins(M["C"], labels, TOPK)         # frozen machinery
    St_list = st_list_from_C(M["C"])
    Vm = votemargins(St_list, labels, TOPK, loo=True)                   # our vectorized copy
    parity = float(np.max(np.abs(Vm - Vm_ref)))
    if parity > 1e-9:
        raise RuntimeError("op#4 vote parity vs frozen machinery broke: maxabs={}".format(parity))
    votemean = Vm.mean(axis=1)
    op4_pred = (votemean >= 0.0).astype(int)
    op4_acc = float((op4_pred == labels).mean())
    op4_mf1 = _macro_f1(labels, op4_pred)
    d_acc = op4_acc - pool_acc
    d_f1 = op4_mf1 - pool_mf1

    # label-oracle at SAME operator family (per-query segment-vote selection, gold)
    signed = labels.astype(np.float64) * 2.0 - 1.0
    tstar = np.argmax(signed[:, None] * Vm, axis=1)
    sel_vote = Vm[np.arange(len(labels)), tstar]
    orc_fam_pred = (sel_vote >= 0.0).astype(int)
    orc_fam_acc = float((orc_fam_pred == labels).mean())
    d_orc_fam = orc_fam_acc - pool_acc

    # banked-style MaxSim oracle cross-check (should ~match w2b_probe_results.json)
    orcMS_acc, orcMS_f1, _, _ = W.oracle_ceiling(M["C"], labels, TOPK)
    d_orcMS = orcMS_acc - pool_acc

    fano_acc = W.fano(labels, TOPK)

    # permutation null (index perm, seeds 0..99), Delta(op4 - pooled) acc
    obs_d = d_acc
    null_d = []
    for s in NULL_SEEDS:
        perm = np.random.default_rng(s).permutation(mem["V"])
        ix = np.ix_(perm, perm)
        Stp = [St[ix] for St in St_list]
        Vmp = votemargins(Stp, labels, TOPK, loo=True)
        vm_p = Vmp.mean(axis=1)
        pa = float(((vm_p >= 0).astype(int) == labels).mean())
        pp_acc, _, _, _ = W.run_vote(M["spool"][ix], labels, TOPK)
        null_d.append(pa - pp_acc)
    null_p95 = float(np.percentile(null_d, 95))

    # bootstrap Delta(op4 - pooled): op4 votes = votemean, pooled votes = pool_votes
    boot = W.bootstrap_delta(votemean, np.asarray(pool_votes, dtype=np.float64), labels, N_BOOT)

    return {
        "dataset": ds, "arm": "PRIMARY_LOO_train+dev", "V": mem["V"], "K": mem["K"],
        "vote_parity_maxabs": parity,
        "pooled": {"acc": pool_acc, "macro_f1": pool_mf1, "roc": pool_roc},
        "op4": {"acc": op4_acc, "macro_f1": op4_mf1, "d_acc": d_acc, "d_f1": d_f1},
        "oracle_same_family_selection": {"acc": orc_fam_acc, "d_acc": d_orc_fam},
        "oracle_maxsim_banked_style": {"acc": orcMS_acc, "macro_f1": orcMS_f1, "d_acc": d_orcMS},
        "fano_acc": fano_acc,
        "perm_null": {"n_seeds": len(NULL_SEEDS), "obs_d_acc": obs_d, "null_p95": null_p95,
                      "obs_gt_p95": bool(obs_d > null_p95)},
        "bootstrap": {"d_acc_p5": boot["dacc_p5"], "d_acc_p50": boot["dacc_p50"],
                      "d_acc_p95": boot["dacc_p95"], "p5_gt0": boot["dacc_p5_gt0"]},
    }, mem, M


def gate_alpha_dev_vs_train(ds, data_root):
    tr = W.load_memory(data_root, ds, "subclipK4", ("train",), EXPECTED_TRAIN[ds], want_pooled=False)
    dv = W.load_memory(data_root, ds, "subclipK4", ("dev_seen",), EXPECTED_DEV[ds], want_pooled=False)
    gtr, ltr = tr["g"], tr["labels"]
    gdv, ldv = dv["g"], dv["labels"]

    # pooled dev->train
    Sp = cross_pooled(gdv, gtr)
    pool_acc, pool_mf1, _ = cross_vote_acc(Sp, ltr, ldv, TOPK)

    # op#4 dev->train: mean over per-segment cross votes
    St = cross_st_list(gdv, gtr)
    Vm = votemargins(St, ltr, TOPK, loo=False)           # [Vdv,K], memory=train labels
    votemean = Vm.mean(axis=1)
    op4_pred = (votemean >= 0.0).astype(int)
    op4_acc = float((op4_pred == ldv).mean())
    op4_mf1 = _macro_f1(ldv, op4_pred)
    return {"dataset": ds, "arm": "CORROB_dev_query->train_mem",
            "Vdev": int(dv["V"]), "Vtrain": int(tr["V"]), "K": int(tr["K"]),
            "pooled": {"acc": pool_acc, "macro_f1": pool_mf1},
            "op4": {"acc": op4_acc, "macro_f1": op4_mf1,
                    "d_acc": op4_acc - pool_acc, "d_f1": op4_mf1 - pool_mf1}}


def machinery_sanity(mem, M):
    """(A) all-K-segments-identical(=pooled) and (B) K=1 must both make op#4 == POOLED (hard asserts)."""
    labels = mem["labels"]
    _, _, _, pool_votes = W.run_vote(M["spool"], labels, TOPK)
    pool_votes = np.asarray(pool_votes, dtype=np.float64)

    # (A) all identical segments = pooled vector broadcast
    pooled_vec = W._l2norm(mem["g"].mean(dim=1), dim=-1)                 # [V,D]
    g_ident = pooled_vec.unsqueeze(1).repeat(1, mem["K"], 1)            # [V,K,D]
    Ci = W._l2norm(g_ident, dim=-1).reshape(mem["V"] * mem["K"], -1)
    Sffi = (Ci @ Ci.t()).reshape(mem["V"], mem["K"], mem["V"], mem["K"]).numpy().astype(np.float32)
    Vmi = votemargins(st_list_from_C(Sffi), labels, TOPK, loo=True)
    a_ok = float(np.max(np.abs(Vmi.mean(axis=1) - pool_votes)))

    # (B) K=1 (first segment only)
    g1 = mem["g"][:, :1, :]
    C1 = (W._l2norm(g1, dim=-1).reshape(mem["V"], -1) @
          W._l2norm(g1, dim=-1).reshape(mem["V"], -1).t()).numpy().astype(np.float64)
    spool1 = (W._l2norm(g1.mean(dim=1), dim=-1) @ W._l2norm(g1.mean(dim=1), dim=-1).t()).numpy()
    Vm1 = votemargins([C1], labels, TOPK, loo=True)
    _, _, _, pv1 = W.run_vote(spool1, labels, TOPK)
    b_ok = float(np.max(np.abs(Vm1[:, 0] - np.asarray(pv1, dtype=np.float64))))

    assert a_ok < 1e-6, "SANITY-A failed: all-identical op#4 != POOLED (maxabs={})".format(a_ok)
    assert b_ok < 1e-6, "SANITY-B failed: K=1 op#4 != POOLED (maxabs={})".format(b_ok)
    return {"all_identical_maxabs_vs_pooled": a_ok, "K1_maxabs_vs_pooled": b_ok,
            "A_recovers_baseline": True, "B_recovers_baseline": True}


# ---------------------------------------------------------------------------
# GATE β — selection-ceiling arithmetic on banked W2-B oracle.
# ---------------------------------------------------------------------------
def gate_beta(w2b_json, op4_by_ds):
    d = json.load(open(w2b_json))
    out = {"source": w2b_json, "per_dataset": {}}
    for r in d["primaries"]:
        ds = r["dataset"]
        pool = r["arms"]["POOLED"]["acc"]
        set_d = r["arms"]["SET"]["acc"] - pool
        asym_d = r["arms"]["ASYM"]["acc"] - pool
        oracle_d = r["oracle"]["d_acc"]
        op4_d = op4_by_ds.get(ds, float("nan"))            # freshly measured legal uniform operator
        symmetric_slice = float(np.nanmax([set_d, asym_d, op4_d]))
        selection_slice = oracle_d - symmetric_slice
        out["per_dataset"][ds] = {
            "banked_POOLED_acc": pool,
            "banked_SET_d_acc": set_d, "banked_ASYM_d_acc": asym_d,
            "fresh_op4_d_acc": op4_d,
            "oracle_headroom_d_acc": oracle_d,
            "symmetric_slice (legal uniform, best of SET/ASYM/op4)": symmetric_slice,
            "selection_slice (banned per-item selection only)": selection_slice,
            "selection_locked": bool(symmetric_slice < 0.010 and selection_slice > 0.030),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(W._REPO_ROOT, "data"))
    ap.add_argument("--datasets", default="HateMM,MHC")
    ap.add_argument("--w2b_json", default=os.path.join(W._REPO_ROOT, "refine-logs/w2b_probe_results.json"))
    ap.add_argument("--out_json", default=os.path.join(W._REPO_ROOT, "refine-logs/ISR_PREGATE_OUT.json"))
    args = ap.parse_args()

    _assert_no_test(args.w2b_json, args.out_json)
    torch.manual_seed(20260721)
    np.random.seed(20260721)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    _log("=" * 78)
    _log("[ISR $0 pre-gate] CPU-only={} datasets={} topk={} null_seeds={} bar=+{}".format(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "", datasets, TOPK, len(NULL_SEEDS), PROMOTE_BAR))
    _log("[ISR $0 pre-gate] N4 fail-closed: train+dev_seen ONLY; expected mem {}".format(EXPECTED_PRIMARY))
    _log("=" * 78)

    t0 = time.time()
    primary, corrob, sanity = [], [], None
    op4_primary_dacc = {}
    for ds in datasets:
        _log("[ISR] --- {} PRIMARY (LOO train∪dev) ---".format(ds))
        p, mem, M = gate_alpha_primary(ds, args.data_root)
        primary.append(p)
        op4_primary_dacc[ds] = p["op4"]["d_acc"]
        _log("     POOLED acc={:.4f} | op#4 acc={:.4f} Δacc={:+.4f} ΔmF1={:+.4f} | Fano={:.4f} | "
             "oracle(fam) Δ={:+.4f} oracle(MaxSim) Δ={:+.4f}".format(
                 p["pooled"]["acc"], p["op4"]["acc"], p["op4"]["d_acc"], p["op4"]["d_f1"],
                 p["fano_acc"], p["oracle_same_family_selection"]["d_acc"],
                 p["oracle_maxsim_banked_style"]["d_acc"]))
        _log("     perm-null Δacc obs={:+.4f} vs p95={:+.4f} (obs>p95={}) | boot Δacc 5th={:+.4f}".format(
            p["perm_null"]["obs_d_acc"], p["perm_null"]["null_p95"], p["perm_null"]["obs_gt_p95"],
            p["bootstrap"]["d_acc_p5"]))
        if sanity is None:
            sanity = machinery_sanity(mem, M)
            _log("     sanity: all-identical maxabs={:.2e} K=1 maxabs={:.2e} (both recover baseline)".format(
                sanity["all_identical_maxabs_vs_pooled"], sanity["K1_maxabs_vs_pooled"]))
        _log("[ISR] --- {} CORROB (dev→train) ---".format(ds))
        c = gate_alpha_dev_vs_train(ds, args.data_root)
        corrob.append(c)
        _log("     POOLED dev acc={:.4f} | op#4 dev acc={:.4f} Δacc={:+.4f} ΔmF1={:+.4f}".format(
            c["pooled"]["acc"], c["op4"]["acc"], c["op4"]["d_acc"], c["op4"]["d_f1"]))

    beta = gate_beta(args.w2b_json, op4_primary_dacc)
    _log("[ISR] --- GATE β selection-ceiling ---")
    for ds, b in beta["per_dataset"].items():
        _log("     {}: oracle Δ={:+.4f} = symmetric {:+.4f} (legal) + selection {:+.4f} (banned); "
             "selection_locked={}".format(
                 ds, b["oracle_headroom_d_acc"], b["symmetric_slice (legal uniform, best of SET/ASYM/op4)"],
                 b["selection_slice (banned per-item selection only)"], b["selection_locked"]))

    # ---- pre-declared decision logic ----
    def clears(rows):
        return [r["dataset"] for r in rows if r["op4"]["d_acc"] >= PROMOTE_BAR]
    prom_primary = clears(primary)
    prom_corrob = clears(corrob)
    fano_ok = all(p["fano_acc"] >= W.FANO_BAR for p in primary)
    alpha_promotes = bool(prom_primary or prom_corrob)
    beta_selection_locked = all(b["selection_locked"] for b in beta["per_dataset"].values())
    if alpha_promotes:
        verdict = "GO-FOR-QWEN-EXTRACTION"
    elif beta_selection_locked:
        verdict = "NO-GO"
    else:
        verdict = "NO-GO (alpha flat; beta not strictly selection-locked — see arithmetic)"

    decision = {
        "promote_bar_d_acc": PROMOTE_BAR,
        "fano_ok_both": fano_ok,
        "alpha_primary_datasets_clearing_bar": prom_primary,
        "alpha_corrob_datasets_clearing_bar": prom_corrob,
        "alpha_promotes": alpha_promotes,
        "beta_selection_locked_all": beta_selection_locked,
        "VERDICT": verdict,
    }
    _log("\n[ISR] VERDICT = {} (Fano-ok={}, α-promotes={}, β-locked={})".format(
        verdict, fano_ok, alpha_promotes, beta_selection_locked))

    out = {
        "meta": {"date": "2026-07-21", "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                 "topk": TOPK, "null_seeds": len(NULL_SEEDS), "n_boot": N_BOOT,
                 "promote_bar": PROMOTE_BAR, "recon_ladder": {"HateMM": RECON_HATEMM_BAR, "MHC": RECON_MHC_BAR},
                 "binding_skeleton": "refine-logs/SEG_REENCODE_FORENSIC_RECON.md@31bcd03 §3/§5",
                 "expected_mem_primary": EXPECTED_PRIMARY, "expected_train": EXPECTED_TRAIN,
                 "expected_dev": EXPECTED_DEV,
                 "note": "op#4 = uniform-mean of frozen per-segment votes; vote NOT reimplemented "
                         "(parity-checked vs w2b_probe._single_query_vote_margins)."},
        "gate_alpha_primary_LOO": primary,
        "gate_alpha_corrob_dev_vs_train": corrob,
        "machinery_sanity": sanity,
        "gate_beta_selection_ceiling": beta,
        "decision": decision,
    }

    def _j(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=_j)
    _log("[ISR] wrote {} in {:.1f}s".format(args.out_json, time.time() - t0))


if __name__ == "__main__":
    main()
