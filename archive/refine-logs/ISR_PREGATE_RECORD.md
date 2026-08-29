# ISR $0 PRE-GATE RECORD — independent-segment re-encode (GAP-3 / F61 cell 5)

**Executor:** isr-pregate subagent (zero-GPU $0 gate). **Date:** 2026-07-21 NZST.
**Binding skeleton:** `refine-logs/SEG_REENCODE_FORENSIC_RECON.md` (commit `31bcd03`), §3 (the $0 pre-gate)
+ §5 (frozen kill-bars). Read end-to-end; this record honors it. Any deviation is logged **loudly** in §0.1.
**Machinery reused verbatim:** `scripts/analysis/w2b_probe.py` — `load_memory` (N4 train/dev-only + video-count
guards), `build_matrices`, `run_vote` → `compute_metrics_retrieval` (the REAL rank-weighted signed-cosine
vote), `_single_query_vote_margins` (per-segment vote), `oracle_ceiling`, `fano`. No vote is reimplemented.

**Discipline honored.** CPU-only (`CUDA_VISIBLE_DEVICES=""`). ZERO GPU / SLURM / Modal / training. NO
test-set read (loader hard-asserts split ∈ {train, dev_seen}; a `test_seen` path raises). No mutation of
`autoresearch/goal_mllm_plus3/state/`. Deliverables: this record, `scripts/analysis/isr_pregate.py`,
`refine-logs/ISR_PREGATE_OUT.json`. Committed on `main`, **not pushed**.

---

## 0. PRE-DECLARED DESIGN + BARS (written BEFORE any dev number is computed — forking-path discipline)

### 0.1 Deviation note (logged loudly)
The team-lead tasking says *"for each dev item … against the train memory"*; the binding recon §3 Gate α
says *"reusing w2b_probe.py's loader + compute_metrics_retrieval vote (**video-level LOO**, no test-touch,
Fano-calibrated)"*. These are two protocols. **Resolution (pre-declared):**
- **PRIMARY / survival-determining arm = video-level LOO over train ∪ dev_seen** (recon §3 verbatim;
  memory V = 851 HateMM / 629 MHC, diagonal self-exclusion). This is **apples-to-apples** with the banked
  W2-B POOLED / SET / oracle in `refine-logs/w2b_probe_results.json` — the whole point of a $0 gate is
  to reuse that banked evidence on the identical memory.
- **CORROBORATING arm = strict dev-query → train-memory** (dev_seen items retrieve only from the train
  bank; metric on dev items only; no LOO needed, disjoint sets). This honors the team-lead's literal
  phrasing. Both arms read **only** train + dev_seen.
- **Verdict uses BOTH:** NO-GO requires **both** arms flat on **both** datasets; GO requires **either**
  arm to clear the bar on **≥1** dataset (conservative toward not-missing-signal on the weaker CLIP encoder).

### 0.2 Gate α — operator #4 (per-segment-kNN vote-mean, fixed uniform), on banked frame-local CLIP
For each query video, for each of its K=4 contiguous segments t, run the **deployed top-20 rank-weighted
signed-cosine vote** (`compute_metrics_retrieval`, arithmetic, use_sim) where segment t scores each memory
**video** j by `s_t(Q,j) = max_m cos( ĝ_{Q,t}, ĝ_{j,m} )` (video-level retrieval, LOO). This yields a
per-segment vote margin `V_t(Q)`. Combine by **UNIFORM MEAN** (NO selection, NO max): `v(Q) = mean_t V_t(Q)`,
predict positive iff `v(Q) ≥ 0`. Baseline = **pooled-key one-hop**: mean the K segment vectors → one key →
top-20 kNN vote (`run_vote(spool)`), byte-identical to the banked W2-B POOLED arm.

**Pre-declared PROMOTION BAR (team-lead pinned):** PROMOTE iff **mean dev Δacc(op#4 − POOLED) ≥ +0.030 on
≥1 dataset** (HateMM or MHC-EN), in either arm. Recon §5 stricter ladder reported alongside for context
(HateMM Δacc ≥ +0.05 AND ΔmF1 ≥ +0.05; MHC-EN Δacc ≥ +0.03 AND ΔmF1 ≥ +0.03) but the binding gate-α bar is
+0.030 Δacc / ≥1 dataset. Else KILL.

**Mandatory calibration arms (must be reported + sane):**
1. **Machine validity (Fano):** ±1 gold-label-agreement key LOO vote acc ≥ 0.99 on both datasets, else VOID.
2. **Label-oracle at same operator family:** per-query segment-vote SELECTION oracle
   `t*(Q)=argmax_t (2y_Q−1)·V_t(Q)` using gold; report Δacc(oracle − POOLED). Also cross-check the banked
   MaxSim oracle (`oracle_ceiling`) against `w2b_probe_results.json`. (This oracle is the selection ceiling
   that BAN B / Law III forecloses — the legal uniform op#4 cannot reach it; feeds Gate β.)
3. **Permutation null (≥100 perms):** index-permutation null of Δ(op#4 − POOLED) acc (seeds 0..99, same
   perm both arms, exactly w2b's `permutation_null` construction); report obs Δ vs null-95th.
4. **Bootstrap:** 1000-resample Δ(op#4 − POOLED); report 5th-pct.
5. **Machinery sanity (hard asserts):** (A) all-K-segments-identical (= pooled vector) ⇒ op#4 votes ≡
   POOLED votes; (B) K=1 (first segment only) ⇒ op#4 votes ≡ POOLED votes. Either recovering the baseline
   proves the operator degenerates correctly.

### 0.3 Gate β — selection-ceiling arithmetic ($0, no run; on banked W2-B oracle)
Decompose the banked oracle headroom Δacc(oracle − POOLED) into:
- **SYMMETRIC / pooled slice** = best *legal uniform* (non-selecting) operator Δ vs POOLED
  (= max of banked SET, ASYM, and freshly-measured op#4), reachable by the surviving operator;
- **SELECTION slice** = oracle_headroom − symmetric_slice = the part reachable **only** by banned
  per-item segment selection (BAN B: Law III alignment > q=0.663 unmeetable in-box; F47 closed all three
  supervision sources; P11 "MIL already carries it").
Cite `refine-logs/w2b_probe_results.json` keys for every banked number. β is **selection-locked** iff the
symmetric slice ≈ 0 while the oracle headroom is large (⇒ the legal operator can access ≈none of it).

### 0.4 Pre-declared VERDICT LOGIC
- **α flat (< +0.030 both datasets, both arms) AND β selection-locked** ⇒ **NO-GO** for the ISR cell
  (bank; Qwen per-segment extraction never happens; 0 GPU-h).
- **α promotes (≥ +0.030 on ≥1 dataset, either arm)** ⇒ **GO-FOR-QWEN-EXTRACTION** — report only; the
  orchestrator escalates (prereg → review → freeze → local SLURM). This executor submits **nothing**.

---

## 1. RESULTS
Source: `refine-logs/ISR_PREGATE_OUT.json` (this run, CPU-only, 35s). op#4 vote **bit-exact** to the frozen
`w2b_probe._single_query_vote_margins` (parity maxabs = 0.0e+00 both datasets — no vote reimplemented).

### 1.1 Gate α — operator #4 (per-segment-kNN vote-mean) vs pooled one-hop
**PRIMARY (video-level LOO, train ∪ dev_seen — apples-to-apples with banked W2-B):**

| dataset | mem V | POOLED acc | op#4 acc | **Δacc** | ΔmF1 | Fano | perm-null 95th | obs>null? | boot Δacc 5th |
|---|---|---|---|---|---|---|---|---|---|
| HateMM | 851 | 0.7568 | 0.7579 | **+0.0012** | −0.0038 | 1.0000 | +0.0282 | No | −0.0153 |
| MHC-EN | 629 | 0.7186 | 0.7218 | **+0.0032** | −0.0012 | 1.0000 | +0.0238 | No | −0.0111 |

**CORROBORATING (strict dev-query → train-memory, dev items only):**

| dataset | V dev / train | POOLED acc | op#4 acc | **Δacc** | ΔmF1 |
|---|---|---|---|---|---|
| HateMM | 107 / 744 | 0.7103 | 0.7196 | **+0.0093** | +0.0054 |
| MHC-EN | 80 / 549 | 0.7250 | 0.7250 | **+0.0000** | +0.0000 |

Every Δacc is far under the +0.030 bar (max = **+0.0093**, one HateMM dev item, n=107). Neither primary Δ
exceeds its permutation-null 95th; both bootstrap 5th-pct are **< 0**. ΔmF1 is negative on both primary
datasets. **α is FLAT.**

**Calibration arms.** Fano = 1.0000 both (machine valid). Cross-checks bit-exact vs banked W2-B
(`w2b_probe_results.json`): POOLED acc 0.7568 / 0.7186; MaxSim-oracle Δ +0.0776 / +0.0700. The
**same-family segment-vote selection oracle** = +0.0776 / +0.0700 (identical to the MaxSim oracle — both
are per-segment SELECTION ceilings). **Machinery sanity (hard asserts PASSED):** all-K-segments-identical
maxabs = 2.2e-07; K=1 maxabs = 0.0 — op#4 recovers the pooled baseline in both degenerate cases.

### 1.2 Gate β — selection-ceiling decomposition (arithmetic on banked W2-B oracle)
`refine-logs/w2b_probe_results.json` → `primaries[].oracle.d_acc`, `arms.{POOLED,SET,ASYM}.acc`; op#4 slice
from §1.1.

| dataset | oracle headroom | symmetric slice (legal uniform = best of SET/ASYM/op#4) | selection slice (banned per-item) | selection-locked |
|---|---|---|---|---|
| HateMM | +0.0776 | **+0.0012** (SET −0.0047, ASYM −0.0059, op#4 +0.0012) | **+0.0764** | **Yes** |
| MHC-EN | +0.0700 | **+0.0064** (SET +0.0016, ASYM +0.0064, op#4 +0.0032) | **+0.0636** | **Yes** |

**One-liner:** ~98% of HateMM and ~91% of EN oracle headroom lives in the SELECTION slice, reachable only
by the banned per-item segment selector (BAN B / Law III: alignment > q=0.663 unmeetable in-box; F47 closed
all three supervision sources; P11 "MIL already carries it"). The one legal operator that survives both
bans — the uniform op#4 — does **no selection**, so it can access only the symmetric slice, which all three
legal operators measure at **≈ 0**. The convertible headroom and the legal operator are **disjoint**.

---

## 2. VERDICT

**NO-GO** for the ISR cell (independent-segment re-encode / GAP-3 / F61 cell 5). **Qwen per-segment
extraction never happens; 0 GPU-h spent.**

Pre-declared logic (§0.4) is satisfied: **α is flat** (op#4 Δacc ≤ +0.0093 < +0.030 on both datasets in
both arms; below the permutation null; bootstrap-5th < 0; ΔmF1 negative on both primary datasets) **AND β is
selection-locked** (symmetric slice ≈ 0; ~90-98% of the oracle headroom is banned-selection-only). Fano =
1.0 confirms the vote machinery is valid, so the flatness is a real negative, not a broken pipe.

This is the expected outcome the recon flagged (§0 BOTTOM LINE, honest prior ~5-10%): the surviving legal
operator is flat on the *weaker* CLIP encoder aimed at a headroom that is the wrong kind (selection, not
aggregation) for it to reach. The lone remaining reed — a frozen CLIP→Qwen encoder swap against a
two-for-two-dead family with Law IV (frozen ⇒ HateMM-only) — does not clear the $0 gate, so spending Qwen
GPU here would repeat the W2-B→S2S mistake. **Cell banked. Orchestrator: submit nothing.**
