#!/usr/bin/env python3
"""W2-C CLIP-K4 ORDER pre-check — NON-BINDING PRIOR-MOVER (cloud triage). CPU, ZERO training, ZERO test touch.

WHAT THIS IS. A cheap, zero-GPU prior-mover for wave-2 candidate W2-C (temporal-order / escalation-aware
alignment kernel). The BINDING adjudication of W2-C happens LATER inside the S2S probe's pre-declared
order-kernel arm on the Qwen T=8 frameset; THIS script only moves the prior on the already-banked CLIP
`subclipK4` caches (K=4 contiguous temporal sub-clips over 16 frames, stored IN TEMPORAL ORDER). It emits
RAW numbers only — NO pass/fail gate, NO verdict. Cloud numbers are CLOUD-TRIAGE TIER (~1.4pt cross-hardware
drift) and NEVER enter a local paper table.

Design spec : refine-logs/W2C_FORENSIC_RECON.md  (§A mechanism, §C descriptive findings, §E bar sketch)
Machinery   : REUSES scripts/analysis/w2b_probe.py verbatim for cache-loading (video-level view(V,K,D),
              contiguous-parent + video-count guards), the REAL top-20 rank-weighted signed-cosine LOO
              vote (utils.metrics.compute_metrics_retrieval), and the paired bootstrap. w2b_probe.py is
              IMPORTED, never modified.

ARMS (retrieval-metric swap only; the vote machinery is byte-for-byte the w2b/S2S LOO vote):
  (1) POOLED        cos(mean_k g_k^Q, mean_k g_k^M)                         order-invariant reference
  (2) MEANMAXSIM    mean_q max_m cos(ghat^Q_q, ghat^M_m)                    ORDER-BLIND reference (to beat)
  (3) ORDER-DTW     1 - softDTW_cost/K over C[q,m]=1-cos, monotonic 3-move  soft monotonic alignment kernel
  (4) TRANSITION    cos over L2-normed concat of signed diffs {g_{t+1}-g_t} first-difference (narrative-turn)
  (5) ORDER+SET     OMITTED — recon §A pre-declares only soft-DTW and the transition-set kernel; it does
                    NOT specify a combined ORDER+SET composite, and task arm (5) is conditional on §A
                    specifying one. Recorded as omitted-by-design (see meta.combined_arm).

MECHANISM-SPECIFIC NULL (the sharpest gate). WITHIN-VIDEO ORDER-SHUFFLE: permute each video's K=4 sub-clip
order independently (>=100 seeds), rebuild ONLY the order arms, Delta vs the (order-invariant, hence fixed)
MEANMAXSIM. By construction POOLED and MEANMAXSIM are invariant under this shuffle — asserted as a SELF-CHECK
(if it fails, an order arm is leaking order into an order-blind arm). Report each order arm's observed Delta
vs the shuffle-null 95th quantile.

BOOTSTRAP 1000 on Delta(ORDER-DTW - MEANMAXSIM) and Delta(TRANSITION - MEANMAXSIM), paired on the vote
margins (D3-fragility guard).

K=4 CAVEAT (structural). K=4 => only 3 transitions / a length-4 warp — thin BY CONSTRUCTION. The meaningful
test is the future S2S T=8 (16-frame, 7-transition) arm; this pre-check exists to sharpen the prior early.
CLIP<Qwen caveat (per W2-B §E): a CLIP-null cannot close the Qwen version; a CLIP-positive corroborates.

FAIL CLOSED (inherited from w2b load_memory): never constructs/opens any test_seen file; asserts memory
VIDEO-count == train u dev_seen (851/629). No silent excepts.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# w2b_probe.py sits next to this file (locally in scripts/analysis/, on Modal at /root/scripts/analysis/).
# The directory of the script being executed is on sys.path[0], so a bare import resolves in both places.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import w2b_probe as w2b  # noqa: E402  (reuse: load_memory / run_vote / bootstrap_delta / constants)

# ------- pre-registered constants -------
TOPK = w2b.TOPK                                  # 20 (inherited)
EPS = w2b.EPS
EXPECTED_MEM_PRIMARY = w2b.EXPECTED_MEM_PRIMARY  # {"HateMM": 851, "MHC": 629} (train u dev_seen)
SEED = 20260714
DTW_GAMMA = 0.1                                  # small gamma => softmin ~ hard min => monotonic DTW
NULL_SEEDS = list(range(100))                    # within-video order-shuffle seeds (>=100 required)
N_BOOT_DEFAULT = 1000


def _log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# Retrieval metrics (float; deterministic). POOLED / MEANMAXSIM match w2b_probe exactly.
# ---------------------------------------------------------------------------
def _l2norm(x, dim=-1, eps=EPS):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def pooled_matrix(g):
    """(1) POOLED: cos(mean_k g_k^Q, mean_k g_k^M). Order-invariant (mean over K)."""
    pooled = _l2norm(g.mean(dim=1), dim=-1)                              # [V, D]
    return (pooled @ pooled.t()).numpy().astype(np.float64)             # [V, V]


def meanmaxsim_and_cos(g):
    """(2) MEANMAXSIM: mean_q max_m cos(ghat^Q_q, ghat^M_m). Permutation-invariant on BOTH sides.
    Also returns the full [V,K,V,K] cos tensor so ORDER-DTW can reuse it (cost = 1 - cos)."""
    V, K, D = g.shape
    ghat = _l2norm(g, dim=-1)                                            # zero rows -> ~0
    G = ghat.reshape(V * K, D)
    Sff = (G @ G.t()).reshape(V, K, V, K).numpy().astype(np.float32)     # cos(sub-clip, sub-clip)
    mms = Sff.max(axis=3).mean(axis=1).astype(np.float64)               # [V, V]
    return mms, Sff


def _softmin3(a, b, c, gamma):
    """Stabilised soft-min of three (batched) arrays. exp(-inf)=0 handles DTW border padding cleanly."""
    stack = np.stack([a, b, c], axis=0)
    m = stack.min(axis=0)
    z = np.exp(-(stack - m) / gamma)
    return m - gamma * np.log(z.sum(axis=0))


def _softdtw_cost(C, gamma):
    """Batched monotonic soft-DTW accumulated cost over the last two (query x memory) axes.
    C: [..., K, K] non-negative cost. Classic 3-move recursion (down / right / diagonal) — every move is
    monotonic (indices never decrease). Returns [...] accumulated soft-min-cost path value R[K,K]."""
    Kq, Km = C.shape[-2], C.shape[-1]
    batch = C.shape[:-2]
    R = np.full(batch + (Kq + 1, Km + 1), np.inf, dtype=np.float64)
    R[..., 0, 0] = 0.0
    for i in range(1, Kq + 1):
        for j in range(1, Km + 1):
            r = _softmin3(R[..., i - 1, j], R[..., i, j - 1], R[..., i - 1, j - 1], gamma)
            R[..., i, j] = C[..., i - 1, j - 1] + r
    return R[..., Kq, Km]


def order_dtw_matrix(g, gamma=DTW_GAMMA, Sff=None):
    """(3) ORDER-DTW: monotonic soft-alignment over the K ordered sub-clips.
    Cost C[q,m] = 1 - cos(ghat^Q_q, ghat^M_m); score = 1 - softDTW_cost/K = average cosine along the soft
    monotonic warp (cosine-comparable, higher=more similar, so it drops straight into the sim-weighted vote).
    Order-sensitive: any permutation of the K axis changes the admissible warping paths."""
    V, K, D = g.shape
    if Sff is None:
        ghat = _l2norm(g, dim=-1)
        G = ghat.reshape(V * K, D)
        Sff = (G @ G.t()).reshape(V, K, V, K).numpy().astype(np.float32)
    # Sff axes are [i(query), q, j(memory), m]; DTW wants cost[i, j, q, m].
    C = np.transpose(1.0 - Sff.astype(np.float64), (0, 2, 1, 3))         # [V, V, K, K]
    R = _softdtw_cost(C, gamma)                                          # [V, V]
    return (1.0 - R / K).astype(np.float64)


def transition_matrix(g):
    """(4) TRANSITION: cosine-kNN over the L2-normed CONCAT of signed first-differences {g_{t+1}-g_t}.
    One (K-1)*D vector per video (the ordered narrative 'turns'); sign flips under sequence reversal, so it
    is genuinely order-sensitive (invariant only to a global shift)."""
    V, K, D = g.shape
    diffs = g[:, 1:, :] - g[:, :-1, :]                                   # [V, K-1, D]  signed transitions
    tvec = diffs.reshape(V, (K - 1) * D)                                 # concatenated in temporal order
    that = _l2norm(tvec, dim=-1)                                         # L2-norm the concatenation
    return (that @ that.t()).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Within-video K-axis shuffle (independent per video).
# ---------------------------------------------------------------------------
def shuffle_within_video(g, rng):
    V, K, D = g.shape
    order = np.argsort(rng.random((V, K)), axis=1)                       # [V, K] independent per-video perm
    idx = torch.from_numpy(order).unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(g, 1, idx)


# ---------------------------------------------------------------------------
# Arm runner + packaging.
# ---------------------------------------------------------------------------
def _run(name, S, labels):
    acc, mf1, roc, votes = w2b.run_vote(S, labels, TOPK)
    return {"name": name, "acc": float(acc), "macro_f1": float(mf1), "roc": float(roc),
            "votes": np.asarray(votes, dtype=np.float64)}


# ---------------------------------------------------------------------------
# SELF-CHECK: POOLED & MEANMAXSIM must be invariant under within-video order-shuffle.
# ---------------------------------------------------------------------------
def order_blind_invariance_check(g, labels):
    gs = shuffle_within_video(g, np.random.default_rng(999))
    sp0 = pooled_matrix(g);        sp1 = pooled_matrix(gs)
    mm0, _ = meanmaxsim_and_cos(g); mm1, _ = meanmaxsim_and_cos(gs)
    a0 = _run("POOLED", sp0, labels);     a1 = _run("POOLED_shuf", sp1, labels)
    m0 = _run("MEANMAXSIM", mm0, labels); m1 = _run("MEANMAXSIM_shuf", mm1, labels)
    pooled_mat_diff = float(np.abs(sp0 - sp1).max())
    mms_mat_diff = float(np.abs(mm0 - mm1).max())
    pooled_metric_identical = (a0["acc"] == a1["acc"]) and (a0["macro_f1"] == a1["macro_f1"])
    mms_metric_identical = (m0["acc"] == m1["acc"]) and (m0["macro_f1"] == m1["macro_f1"])
    # Hard asserts: a genuine order-leak bug moves acc/mF1 materially and blows the matrix diff past ULP.
    assert pooled_metric_identical, "SELF-CHECK FAIL: POOLED acc/mF1 changed under within-video shuffle"
    assert mms_metric_identical, "SELF-CHECK FAIL: MEANMAXSIM acc/mF1 changed under within-video shuffle"
    assert pooled_mat_diff < 1e-6, "SELF-CHECK FAIL: POOLED matrix diff {:.2e} >> ULP".format(pooled_mat_diff)
    assert mms_mat_diff < 1e-6, "SELF-CHECK FAIL: MEANMAXSIM matrix diff {:.2e} >> ULP".format(mms_mat_diff)
    return {"pooled_metric_identical": bool(pooled_metric_identical),
            "mms_metric_identical": bool(mms_metric_identical),
            "pooled_matrix_maxabsdiff": pooled_mat_diff,
            "mms_matrix_maxabsdiff": mms_mat_diff,
            "note": "byte-identity = final acc & mF1 exactly equal; matrix diff is float summation-order ULP"}


# ---------------------------------------------------------------------------
# WITHIN-VIDEO ORDER-SHUFFLE null on the order arms (MEANMAXSIM fixed, being order-invariant).
# ---------------------------------------------------------------------------
def order_shuffle_null(g, labels, mms_acc, mms_f1, gamma, n_seeds):
    dtw_dacc, dtw_df1, tr_dacc, tr_df1 = [], [], [], []
    for s in range(n_seeds):
        gs = shuffle_within_video(g, np.random.default_rng(70000 + s))
        d = _run("ORDER_DTW", order_dtw_matrix(gs, gamma), labels)
        t = _run("TRANSITION", transition_matrix(gs), labels)
        dtw_dacc.append(d["acc"] - mms_acc); dtw_df1.append(d["macro_f1"] - mms_f1)
        tr_dacc.append(t["acc"] - mms_acc);  tr_df1.append(t["macro_f1"] - mms_f1)

    def q95(x):
        return float(np.percentile(np.asarray(x), 95))

    def mean(x):
        return float(np.mean(x))

    return {"n_seeds": int(n_seeds),
            "dtw_vs_mms": {"dacc_p95": q95(dtw_dacc), "df1_p95": q95(dtw_df1),
                           "dacc_mean": mean(dtw_dacc), "df1_mean": mean(dtw_df1)},
            "transition_vs_mms": {"dacc_p95": q95(tr_dacc), "df1_p95": q95(tr_df1),
                                  "dacc_mean": mean(tr_dacc), "df1_mean": mean(tr_df1)}}


# ---------------------------------------------------------------------------
# Per-dataset pre-check.
# ---------------------------------------------------------------------------
def probe_dataset(dataset, data_root, n_boot, gamma, null_seeds):
    exp = EXPECTED_MEM_PRIMARY[dataset]
    mem = w2b.load_memory(data_root, dataset, "subclipK4", ("train", "dev_seen"), exp, want_pooled=False)
    g, labels = mem["g"], mem["labels"]

    S_pool = pooled_matrix(g)
    S_mms, Sff = meanmaxsim_and_cos(g)
    S_dtw = order_dtw_matrix(g, gamma, Sff=Sff)
    S_tr = transition_matrix(g)

    a_pool = _run("POOLED", S_pool, labels)
    a_mms = _run("MEANMAXSIM", S_mms, labels)
    a_dtw = _run("ORDER_DTW", S_dtw, labels)
    a_tr = _run("TRANSITION", S_tr, labels)

    def delta(a, b):
        return {"d_acc": a["acc"] - b["acc"], "d_f1": a["macro_f1"] - b["macro_f1"]}

    d_dtw_mms = delta(a_dtw, a_mms)      # PRIMARY contrast 1 (vs order-blind)
    d_tr_mms = delta(a_tr, a_mms)        # PRIMARY contrast 2 (vs order-blind)
    d_dtw_pool = delta(a_dtw, a_pool)    # reference
    d_tr_pool = delta(a_tr, a_pool)      # reference
    d_mms_pool = delta(a_mms, a_pool)    # is the set-matcher even alive here?

    inv = order_blind_invariance_check(g, labels)
    null = order_shuffle_null(g, labels, a_mms["acc"], a_mms["macro_f1"], gamma, len(null_seeds))
    boot_dtw = w2b.bootstrap_delta(a_dtw["votes"], a_mms["votes"], labels, n_boot)
    boot_tr = w2b.bootstrap_delta(a_tr["votes"], a_mms["votes"], labels, n_boot)

    arms = {a["name"]: {"acc": a["acc"], "macro_f1": a["macro_f1"], "roc": a["roc"]}
            for a in (a_pool, a_mms, a_dtw, a_tr)}
    return {"dataset": dataset, "V": mem["V"], "K": mem["K"], "zero_guard": int(mem["guard"].sum()),
            "arms": arms,
            "deltas": {"dtw_vs_meanmaxsim": d_dtw_mms, "transition_vs_meanmaxsim": d_tr_mms,
                       "dtw_vs_pooled": d_dtw_pool, "transition_vs_pooled": d_tr_pool,
                       "meanmaxsim_vs_pooled": d_mms_pool},
            "self_check_order_blind_invariance": inv,
            "order_shuffle_null": null,
            "bootstrap_dtw_vs_mms": boot_dtw,
            "bootstrap_transition_vs_mms": boot_tr}


# ---------------------------------------------------------------------------
# Self-test (synthetic, NO volume). Validates (a) order-blind invariance self-check and (b) that the order
# kernels actually READ order that POOLED/MEANMAXSIM discard.
# ---------------------------------------------------------------------------
def selftest():
    _log("[w2c self-test] planted-order synthetic (label carried by ORDER only; SET & MEAN identical)")
    rng = np.random.default_rng(SEED)
    D, K, V_per = 64, 4, 60
    base = rng.standard_normal((K, D))
    g_list, lab = [], []
    for c in (1, 0):
        order = np.arange(K) if c == 1 else np.arange(K)[::-1]           # same SET, opposite ORDER by label
        for _ in range(V_per):
            g_list.append(base[order] + 0.15 * rng.standard_normal((K, D)))
            lab.append(c)
    g = torch.tensor(np.stack(g_list)).float()
    labels = np.asarray(lab, dtype=np.int64)

    S_pool = pooled_matrix(g)
    S_mms, _ = meanmaxsim_and_cos(g)
    S_dtw = order_dtw_matrix(g)
    S_tr = transition_matrix(g)
    a_pool = _run("POOLED", S_pool, labels)
    a_mms = _run("MEANMAXSIM", S_mms, labels)
    a_dtw = _run("ORDER_DTW", S_dtw, labels)
    a_tr = _run("TRANSITION", S_tr, labels)
    _log("  POOLED acc={:.4f}  MEANMAXSIM acc={:.4f}  ORDER-DTW acc={:.4f}  TRANSITION acc={:.4f}".format(
        a_pool["acc"], a_mms["acc"], a_dtw["acc"], a_tr["acc"]))

    # (a) order-blind invariance self-check must pass on synthetic too.
    inv = order_blind_invariance_check(g, labels)
    assert inv["pooled_metric_identical"] and inv["mms_metric_identical"]
    _log("  [self-check] POOLED/MEANMAXSIM invariant under within-video shuffle: PASS "
         "(pooled matdiff={:.2e}, mms matdiff={:.2e})".format(
             inv["pooled_matrix_maxabsdiff"], inv["mms_matrix_maxabsdiff"]))

    # (b) order-blind arms are ~chance here (same set/mean both classes); order arms must clearly beat them.
    assert a_mms["acc"] < 0.70, "synthetic MEANMAXSIM should be near-chance (same SET both classes)"
    assert a_dtw["acc"] > a_mms["acc"] + 0.15, "ORDER-DTW failed to read planted order"
    assert a_tr["acc"] > a_mms["acc"] + 0.15, "TRANSITION failed to read planted order"

    # (c) order-shuffle null: destroying order collapses the order arms toward the order-blind reference.
    null = order_shuffle_null(g, labels, a_mms["acc"], a_mms["macro_f1"], DTW_GAMMA, 20)
    obs_dtw = a_dtw["acc"] - a_mms["acc"]
    obs_tr = a_tr["acc"] - a_mms["acc"]
    assert obs_dtw > null["dtw_vs_mms"]["dacc_p95"], "ORDER-DTW obs Delta not above its shuffle-null 95th"
    assert obs_tr > null["transition_vs_mms"]["dacc_p95"], "TRANSITION obs Delta not above shuffle-null 95th"
    _log("  [self-check] order arms beat within-video shuffle-null 95th "
         "(DTW obs {:+.4f} > null95 {:+.4f}; TRANS obs {:+.4f} > null95 {:+.4f}): PASS".format(
             obs_dtw, null["dtw_vs_mms"]["dacc_p95"], obs_tr, null["transition_vs_mms"]["dacc_p95"]))
    _log("[w2c self-test] ALL PASS")
    return True


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------
def _fmt_pct(x):
    return "{:+.4f}".format(x)


def write_markdown(results, meta, out_md):
    L = ["# W2-C CLIP-K4 ORDER pre-check — RAW PRIOR-MOVER NUMBERS (no pass/fail, no verdict)\n",
         "_PRIOR-MOVER / NON-BINDING / CLOUD-TRIAGE TIER (~1.4pt cross-hardware drift; these numbers NEVER "
         "enter a local paper table). K=4 => only 3 transitions — thin BY CONSTRUCTION; the binding test is "
         "the future S2S T=8 arm. CLIP<Qwen: a CLIP-null cannot close the Qwen version; a CLIP-positive "
         "corroborates. Arms: POOLED (order-invariant ref), MEANMAXSIM (order-BLIND ref, the one to beat), "
         "ORDER-DTW (monotonic soft-DTW), TRANSITION (concat signed first-differences). Combined ORDER+SET "
         "arm OMITTED — recon §A specifies none._\n",
         "\n**Config:** topk={topk}, null={null} within-video order-shuffle seeds, bootstrap={nboot}, "
         "dtw_gamma={g}, seed={seed}, CUDA_VISIBLE_DEVICES='' (CPU).\n".format(
             topk=meta["topk"], null=meta["null_seeds"], nboot=meta["n_boot"], g=meta["dtw_gamma"],
             seed=meta["seed"])]
    for r in results:
        L.append("\n## {} — K4 primary (memory V={}, K={}, zero-guard={})\n".format(
            r["dataset"], r["V"], r["K"], r["zero_guard"]))
        L.append("| arm | acc | macro_f1 | roc |")
        L.append("|---|---|---|---|")
        for name in ("POOLED", "MEANMAXSIM", "ORDER_DTW", "TRANSITION"):
            a = r["arms"][name]
            L.append("| {} | {:.4f} | {:.4f} | {:.4f} |".format(name, a["acc"], a["macro_f1"], a["roc"]))
        d = r["deltas"]
        L.append("\n**Primary contrasts (order vs order-BLIND MEANMAXSIM):**")
        L.append("- Δ(ORDER-DTW − MEANMAXSIM): acc {}, macro_f1 {}".format(
            _fmt_pct(d["dtw_vs_meanmaxsim"]["d_acc"]), _fmt_pct(d["dtw_vs_meanmaxsim"]["d_f1"])))
        L.append("- Δ(TRANSITION − MEANMAXSIM): acc {}, macro_f1 {}".format(
            _fmt_pct(d["transition_vs_meanmaxsim"]["d_acc"]), _fmt_pct(d["transition_vs_meanmaxsim"]["d_f1"])))
        L.append("\n**Reference contrasts:** Δ(ORDER-DTW − POOLED) acc {} / mF1 {}; "
                 "Δ(TRANSITION − POOLED) acc {} / mF1 {}; Δ(MEANMAXSIM − POOLED) acc {} / mF1 {}.".format(
                     _fmt_pct(d["dtw_vs_pooled"]["d_acc"]), _fmt_pct(d["dtw_vs_pooled"]["d_f1"]),
                     _fmt_pct(d["transition_vs_pooled"]["d_acc"]), _fmt_pct(d["transition_vs_pooled"]["d_f1"]),
                     _fmt_pct(d["meanmaxsim_vs_pooled"]["d_acc"]),
                     _fmt_pct(d["meanmaxsim_vs_pooled"]["d_f1"])))
        inv = r["self_check_order_blind_invariance"]
        L.append("\n**Self-check — POOLED/MEANMAXSIM byte-identity under within-video order-shuffle:** "
                 "POOLED metric-identical={} (matrix max|Δ|={:.2e}); MEANMAXSIM metric-identical={} "
                 "(matrix max|Δ|={:.2e}).".format(
                     inv["pooled_metric_identical"], inv["pooled_matrix_maxabsdiff"],
                     inv["mms_metric_identical"], inv["mms_matrix_maxabsdiff"]))
        n = r["order_shuffle_null"]
        L.append("\n**Within-video ORDER-SHUFFLE null ({} seeds), Δ vs MEANMAXSIM:**".format(n["n_seeds"]))
        L.append("- ORDER-DTW: obs Δacc {} vs null-95th {} (null-mean {}); obs ΔmF1 {} vs null-95th {}.".format(
            _fmt_pct(d["dtw_vs_meanmaxsim"]["d_acc"]), _fmt_pct(n["dtw_vs_mms"]["dacc_p95"]),
            _fmt_pct(n["dtw_vs_mms"]["dacc_mean"]), _fmt_pct(d["dtw_vs_meanmaxsim"]["d_f1"]),
            _fmt_pct(n["dtw_vs_mms"]["df1_p95"])))
        L.append("- TRANSITION: obs Δacc {} vs null-95th {} (null-mean {}); obs ΔmF1 {} vs null-95th {}.".format(
            _fmt_pct(d["transition_vs_meanmaxsim"]["d_acc"]), _fmt_pct(n["transition_vs_mms"]["dacc_p95"]),
            _fmt_pct(n["transition_vs_mms"]["dacc_mean"]), _fmt_pct(d["transition_vs_meanmaxsim"]["d_f1"]),
            _fmt_pct(n["transition_vs_mms"]["df1_p95"])))
        bd, bt = r["bootstrap_dtw_vs_mms"], r["bootstrap_transition_vs_mms"]
        L.append("\n**Bootstrap ({} resamples), paired Δ vs MEANMAXSIM:**".format(bd["n_boot"]))
        L.append("- ORDER-DTW: Δacc [5/50/95]=[{}/{}/{}]; ΔmF1 [5/50/95]=[{}/{}/{}].".format(
            _fmt_pct(bd["dacc_p5"]), _fmt_pct(bd["dacc_p50"]), _fmt_pct(bd["dacc_p95"]),
            _fmt_pct(bd["df1_p5"]), _fmt_pct(bd["df1_p50"]), _fmt_pct(bd["df1_p95"])))
        L.append("- TRANSITION: Δacc [5/50/95]=[{}/{}/{}]; ΔmF1 [5/50/95]=[{}/{}/{}].".format(
            _fmt_pct(bt["dacc_p5"]), _fmt_pct(bt["dacc_p50"]), _fmt_pct(bt["dacc_p95"]),
            _fmt_pct(bt["df1_p5"]), _fmt_pct(bt["df1_p50"]), _fmt_pct(bt["df1_p95"])))
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")


def _strip_votes(o):
    """Recursively drop the big vote arrays before JSON dump (they are intermediate only)."""
    if isinstance(o, dict):
        return {k: _strip_votes(v) for k, v in o.items() if k != "votes"}
    if isinstance(o, list):
        return [_strip_votes(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser(description="W2-C CLIP-K4 order pre-check (prior-mover, non-binding).")
    ap.add_argument("--data_root", type=str,
                    default=os.path.join(w2b._REPO_ROOT, "data"),
                    help="dir holding CLIP_Embedding/ (local repo data/; /root/data on Modal).")
    ap.add_argument("--datasets", type=str, default="HateMM,MHC")
    ap.add_argument("--n_boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--dtw_gamma", type=float, default=DTW_GAMMA)
    ap.add_argument("--selftest", type=int, default=0, help="run synthetic self-test and exit (no volume).")
    ap.add_argument("--out_md", type=str, default="/root/data/W2C_ORDER_PRECHECK_RESULTS.md")
    ap.add_argument("--out_json", type=str, default="/root/data/w2c_order_precheck_results.json")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # hard CPU
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    _log("=" * 82)
    _log("[W2-C order pre-check] PRIOR-MOVER / NON-BINDING / CLOUD-TRIAGE. topk={} null={} n_boot={} "
         "gamma={}".format(TOPK, len(NULL_SEEDS), args.n_boot, args.dtw_gamma))
    _log("=" * 82)

    # Always self-validate first (synthetic; catches env/import/kernel bugs before touching the caches).
    selftest()
    if args.selftest:
        _log("[W2-C order pre-check] --selftest set; exiting after synthetic validation.")
        return

    t0 = time.time()
    results = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        _log("[W2-C order pre-check] --- {} K4 primary ---".format(ds))
        results.append(probe_dataset(ds, args.data_root, args.n_boot, args.dtw_gamma, NULL_SEEDS))
    _log("[W2-C order pre-check] done in {:.1f}s".format(time.time() - t0))

    meta = {"topk": TOPK, "null_seeds": len(NULL_SEEDS), "n_boot": args.n_boot,
            "dtw_gamma": args.dtw_gamma, "seed": SEED,
            "expected_mem_primary": EXPECTED_MEM_PRIMARY,
            "tier": "PRIOR-MOVER / NON-BINDING / CLOUD-TRIAGE (~1.4pt drift; never enters local tables)",
            "k4_caveat": "K=4 => 3 transitions only; thin by construction; binding test = future S2S T=8 arm",
            "clip_qwen_caveat": "frozen-CLIP appearance vectors; CLIP-null cannot close Qwen, CLIP-positive "
                                "corroborates (W2-B §E)",
            "combined_arm": "OMITTED — recon §A pre-declares only soft-DTW and the transition-set kernel; "
                            "no ORDER+SET composite is specified, so task arm (5) is omitted by design",
            "primary_contrast": "Delta(order-kernel - MEANMAXSIM) in acc AND macro_f1 (order beats "
                                "order-blind); gated by the within-video order-shuffle null"}

    out = {"meta": meta, "results": _strip_votes(results)}

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

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=_jsonable)
    write_markdown(results, meta, args.out_md)
    _log("[W2-C order pre-check] wrote {} and {}".format(args.out_md, args.out_json))


if __name__ == "__main__":
    main()
