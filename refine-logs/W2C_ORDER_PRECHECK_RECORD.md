# W2-C CLIP-K4 ORDER pre-check — RAW RECORD

> **TIER: PRIOR-MOVER / NON-BINDING / CLOUD-TRIAGE.** These numbers are Modal-CPU cloud-triage
> (~1.4pt cross-hardware drift, seed-noise level). They **never enter a local paper table** and carry
> **no verdict** — no pass/fail gate is computed. This pre-check only **moves the W2-C prior**; the
> BINDING adjudication of W2-C happens later inside the S2S probe's pre-declared order-kernel arm on the
> Qwen **T=8** frameset.
>
> **K=4 CAVEAT (structural).** K=4 ⇒ only **3 transitions / a length-4 warp** — thin BY CONSTRUCTION. The
> meaningful test lives in the future S2S **T=8 (16-frame, 7-transition)** arm; this pre-check exists to
> sharpen the prior early on the already-banked CLIP `subclipK4` caches.
>
> **CLIP<Qwen CAVEAT (per W2-B §E).** These are frozen-CLIP **appearance** vectors. A CLIP-null **cannot
> close** the Qwen (instruction-conditioned semantic) version; a CLIP-positive would corroborate.

**Provenance.** Script `scripts/analysis/w2c_order_precheck.py` (imports `w2b_probe.py` verbatim for
cache-loading, the real top-20 rank-weighted signed-cosine LOO vote `utils.metrics.compute_metrics_retrieval`,
and the paired bootstrap; `w2b_probe.py` UNMODIFIED). Run on **Modal CPU** (`rgcl-probe`, volume
`rgcl-features`, app `ap-SjUG5dqZK2Vl4FCZ817y8N`), single run, features-only (no video ever left the
cluster; the banked `subclipK4` caches were already on the volume — nothing uploaded). Config:
`topk=20`, `null=100` within-video order-shuffle seeds, `bootstrap=1000`, `dtw_gamma=0.1`, `seed=20260714`,
`CUDA_VISIBLE_DEVICES=''` (CPU). Memory = train ∪ dev_seen, video-level LOO, **no test_seen** (V=851/629
guards tripped clean). Wall time 232.1 s. Repo HEAD at run = `83db137`.

**Arms (retrieval-metric swap only; vote machinery unchanged).** (1) POOLED `cos(mean_k g_k)` order-invariant
reference; (2) MEANMAXSIM `mean_q max_m cos` **order-BLIND** reference (the one to beat); (3) ORDER-DTW
`1 − softDTW_cost/K` over `C[q,m]=1−cos`, monotonic 3-move soft-DTW; (4) TRANSITION cosine over the L2-normed
CONCAT of signed first-differences `{g_{t+1}−g_t}`. **(5) Combined ORDER+SET OMITTED** — recon §A
pre-declares only soft-DTW and the transition-set kernel; it specifies **no** ORDER+SET composite, and task
arm (5) is conditional on §A specifying one → omitted by design.

---

## POOLED / MEANMAXSIM byte-identity self-check (within-video order-shuffle)

By construction POOLED (mean over K) and MEANMAXSIM (permutation-invariant on both sides) MUST be invariant
when each video's K=4 sub-clip order is permuted. **Both PASS on both datasets** — final acc & macro_f1
exactly equal under the shuffle; the residual score-matrix `max|Δ|` is float summation-order ULP only:

| dataset | POOLED metric-identical | POOLED matrix max\|Δ\| | MEANMAXSIM metric-identical | MEANMAXSIM matrix max\|Δ\| |
|---|---|---|---|---|
| HateMM | True | 4.77e-07 | True | 1.19e-07 |
| MHC-EN | True | 4.17e-07 | True | 1.19e-07 |

The kernel implementation is clean: the order arms do **not** leak order into the order-blind arms.

---

## HateMM — K4 primary (memory V=851, K=4, zero-guard=1)

| arm | acc | macro_f1 | roc |
|---|---|---|---|
| POOLED | 0.7568 | 0.7518 | 0.8301 |
| MEANMAXSIM | 0.7521 | 0.7441 | 0.8285 |
| ORDER_DTW | 0.7579 | 0.7492 | 0.8294 |
| TRANSITION | 0.6980 | 0.6943 | 0.7646 |

**Primary contrasts (order vs order-BLIND MEANMAXSIM):**
- Δ(ORDER-DTW − MEANMAXSIM): acc **+0.0059**, macro_f1 **+0.0051**
- Δ(TRANSITION − MEANMAXSIM): acc **−0.0541**, macro_f1 **−0.0498**

**Reference contrasts:** Δ(ORDER-DTW − POOLED) acc +0.0012 / mF1 −0.0026; Δ(TRANSITION − POOLED) acc
−0.0588 / mF1 −0.0575; Δ(MEANMAXSIM − POOLED) acc −0.0047 / mF1 −0.0077.

**Within-video ORDER-SHUFFLE null (100 seeds), Δ vs MEANMAXSIM:**
- ORDER-DTW: obs Δacc **+0.0059** vs null-95th **+0.0059** (null-mean −0.0001); obs ΔmF1 +0.0051 vs null-95th +0.0060.
- TRANSITION: obs Δacc −0.0541 vs null-95th −0.0634 (null-mean −0.0870); obs ΔmF1 −0.0498 vs null-95th −0.0614.

**Bootstrap (1000 resamples), paired Δ vs MEANMAXSIM:**
- ORDER-DTW: Δacc [5/50/95]=[−0.0083/+0.0059/+0.0212]; ΔmF1 [5/50/95]=[−0.0101/+0.0051/+0.0205]. (5th < 0)
- TRANSITION: Δacc [5/50/95]=[−0.0823/−0.0541/−0.0247]; ΔmF1 [5/50/95]=[−0.0780/−0.0497/−0.0208]. (95th < 0)

---

## MHC-EN — K4 primary (memory V=629, K=4, zero-guard=0)

| arm | acc | macro_f1 | roc |
|---|---|---|---|
| POOLED | 0.7186 | 0.6081 | 0.7536 |
| MEANMAXSIM | 0.7202 | 0.6112 | 0.7544 |
| ORDER_DTW | 0.7250 | 0.6256 | 0.7656 |
| TRANSITION | 0.6645 | 0.5434 | 0.5665 |

**Primary contrasts (order vs order-BLIND MEANMAXSIM):**
- Δ(ORDER-DTW − MEANMAXSIM): acc **+0.0048**, macro_f1 **+0.0144**
- Δ(TRANSITION − MEANMAXSIM): acc **−0.0556**, macro_f1 **−0.0678**

**Reference contrasts:** Δ(ORDER-DTW − POOLED) acc +0.0064 / mF1 +0.0175; Δ(TRANSITION − POOLED) acc
−0.0541 / mF1 −0.0647; Δ(MEANMAXSIM − POOLED) acc +0.0016 / mF1 +0.0031.

**Within-video ORDER-SHUFFLE null (100 seeds), Δ vs MEANMAXSIM:**
- ORDER-DTW: obs Δacc **+0.0048** vs null-95th **+0.0127** (null-mean +0.0047); obs ΔmF1 +0.0144 vs null-95th +0.0194.
- TRANSITION: obs Δacc −0.0556 vs null-95th −0.0269 (null-mean −0.0475); obs ΔmF1 −0.0678 vs null-95th −0.0473.

**Bootstrap (1000 resamples), paired Δ vs MEANMAXSIM:**
- ORDER-DTW: Δacc [5/50/95]=[−0.0095/+0.0048/+0.0192]; ΔmF1 [5/50/95]=[−0.0082/+0.0136/+0.0361]. (5th < 0)
- TRANSITION: Δacc [5/50/95]=[−0.0874/−0.0556/−0.0254]; ΔmF1 [5/50/95]=[−0.1123/−0.0688/−0.0266]. (95th < 0)

---

## Honest one-line read (no verdict — this kills nothing)

**Moves the W2-C prior DOWN.** On frozen CLIP-K4, ORDER-DTW's edge over the order-blind MEANMAXSIM is
tiny (+0.006 acc HateMM, +0.005 acc MHC) and **does not exceed the within-video order-shuffle null** — on
HateMM obs Δacc equals the null-95th exactly (a *random* reorder buys the same edge; null-mean ≈ 0), on
MHC obs Δ sits *below* the null-95th on both acc and mF1, and both bootstrap CIs straddle 0 (5th < 0) — so
DTW's advantage is a "richer-kernel" artifact, not order signal; TRANSITION is strictly **worse** than
order-blind (≈ −0.05 acc / −0.05…−0.07 mF1, bootstrap CI entirely below 0) at the thin K−1=3-transition
budget. MEANMAXSIM ≈ POOLED (−0.005…+0.002 acc) re-confirms the W2-B "set-matching adds nothing over
pooling on CLIP" regime. Combined with the recon §C static-hateful-class forensics, this lowers the prior —
**but it is NON-BINDING and cannot close W2-C**: K=4/3-transition thinness plus the CLIP<Qwen asymmetry mean
the semantic-order hypothesis stays untested until the S2S T=8 arm.
