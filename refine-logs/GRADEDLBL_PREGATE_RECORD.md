# GRADED 3-CLASS SOFT-LABEL $0 PRE-GATE RECORD — litsweep-5 S2 cand-1

**Executor:** gradedlbl-pregate subagent (zero-GPU $0 gate). **Date:** 2026-07-25 NZST.
**Candidate:** MultiHateClip (EN+ZH) ships a 3-class `Label` {Hateful, Offensive, Normal}; the deployed
pipeline merges Offensive+Hateful→1 (`harmful_vs_normal`). Candidate = give **Offensive** a softer
positive target (τ) in HEAD training only; deployment unchanged (binary kNN vote). Shortlist source:
`refine-logs/LITSWEEP5_HATEMM_EN.md` §3-cand-1 (commit 36d833e), priced **≥+1 either dataset ~15%,
goal-bar ~1–2%**.
**Template honored:** `refine-logs/ISR_PREGATE_RECORD.md` (pre-declared operators + bars BEFORE any
dev number; label-oracle ceiling; F63 positive-perm-null; machinery parity). `refine-logs/B5_PROBE_RECORD.md`
(d_oracle with each arm its OWN dev-optimal threshold).
**Machinery reused VERBATIM (no vote reimplemented):** rank-weighted signed-cosine top-20 vote +
fused-key construction lifted from `scripts/analysis/readout_screen.py` (== `cross_channel_router_gate
.py:73-79` vote / `LP_GATE_RECORD.md` fused key / `ISR_PREGATE_RECORD.md §0.2` two arms).

**Discipline honored.** CPU-only (`CUDA_VISIBLE_DEVICES=""`). ZERO GPU / SLURM / Modal / training / job
submission. **NO test-set read** (loader hard-asserts `split ∈ {train, dev_seen}`; a `test_seen` path
raises). Only own-split train+dev 3-class labels read — **READ-ONLY, no method committed, no head
trained**. No mutation of `autoresearch/goal_mllm_plus3/state/`. Deliverables: this record,
`scripts/analysis/gradedlbl_pregate.py` (sha256 `59cc253da8b35e02a342d5b8af5af1d346ca39c335ac3c7ca3dd3d62e914d77d`),
`refine-logs/GRADEDLBL_PREGATE_OUT.json`. Committed on `main`, **not pushed**. Repo HEAD at execution:
`ec3bec3977aae00dbf891b33e8e20d2514dc365a`.

---

## 0. PRE-DECLARED DESIGN + BARS (written BEFORE any dev number — forking-path discipline)

### 0.1 Objects, datasets, encoders
- **Datasets = the two with a real 3-class Label:** MHC (**EN**) and MHC_zh (**ZH**). HateMM has NO
  Offensive class (binary Hate/Non-Hate) → out of scope for the graded lever.
- **Deployed-encoder fused-key caches** (train+dev only): EN primary = frozen `Qwen2.5-VL-7B-Instruct_HF`
  (litsweep5: EN floor ~0.79–0.81 frozen); ZH primary = `Qwen2.5-VL-7B-Instruct-LoRA_HF` (B3 deployed) —
  this is the **machinery-parity anchor**. EN LoRA_HF added as a SENSITIVITY arm (transparency; never
  survival-determining).
- **Fused key** (identical to deployed $0 gates): per video L2-norm `img_feats`(3584) + `text_feats`(3584),
  concat→7168, L2-renorm. **Vote:** deployed rank-weighted signed-cosine top-20, two arms both reading
  only train+dev: **loo** (memory=train∪dev, diagonal self-exclusion) and **devtrain** (memory=train,
  disjoint dev queries).
- **Signed-target reweighting (the graded proxy).** A retrieved memory neighbour contributes signed
  target × cosine × rank-weight. Binary baseline: Normal −1, **Offensive +1**, Hateful +1. Graded:
  Normal −1, Hateful +1, **Offensive = w_off = 2τ−1**. The **query eval target stays binary** gold.
- **Exactness.** Retrieval on the fused key is LABEL-INDEPENDENT ⇒ the top-20 neighbour set is fixed, so
  the vote is LINEAR in w_off: `vote_q(w_off) = A_q + w_off·B_q` (A_q = Normal/Hateful neighbours, B_q =
  Offensive neighbours). The τ grid and the oracle w_off sweep are therefore **exact, not sampled**.

### 0.2 Operator (i) — graded proxy (pre-declared τ grid)
τ ∈ {0.25, 0.50, 0.75} ⇒ w_off ∈ {−0.50, 0.00, +0.50}. Per dataset/arm report dev Δacc(graded−binary)
and items fixed/broken/net vs the binary baseline (same videos).

### 0.3 Operator (ii) — label-oracle ceiling (gold-cheat upper bound; B5-style own thresholds)
Sweep w_off over the full monotone range [−1,+1] step 0.05 (exact). Two readings, each per dataset/arm:
- **DOP** (deployed operating point, cutoff 0): `max_woff acc(cutoff 0) − binary acc(cutoff 0)`.
- **B5** (each config its OWN dev-optimal cutoff): `max_woff [best-cutoff acc] − binary[best-cutoff acc]`
  (the reweighting-only headroom; the calibration boost cancels in the difference — B5_PROBE_RECORD §(c)).
- **BINDING oracle ceiling** = max(DOP, B5) per arm; per-dataset ceiling = max over arms (most generous
  reading — a PARK under it is airtight). This is a **gold-cheat**: dev labels pick w_off and cutoff, so
  it is a strict UPPER BOUND on any honest monotone Offensive reweighting.

### 0.4 Calibration / validity arms (must be reported + sane)
1. **Machinery parity:** ZH LoRA_HF binary baseline fused-key vote reproduces `READOUT_SCREEN_OUT.json`
   ro_L28 (loo 0.8717948718 / devtrain 0.8589743590) **bit-exact** (|Δ|<1e-12).
2. **Degenerate-recovery assert:** w_off=+1 (τ=1.0) reproduces the binary baseline (Δacc = 0 exactly).
3. **Targeted Offensive-permutation null (F63 warning):** among binary-positive memory items, randomly
   pick |Offensive| of them to receive w_off (best τ), keep the rest +1; recompute Δacc; ≥500 perms.
   Isolates "specifically-Offensive" vs "any equal-size positive subset". Report obs Δ vs null p95.

### 0.5 PRE-DECLARED VERDICT LOGIC
- **PARK (arithmetic, K-D-1 logic)** iff **BINDING oracle ceiling < +0.030 dev acc on BOTH datasets**
  (the gold-cheat can't reach the goal bar ⇒ the vote-side mechanism is arithmetically dead — no GPU).
- **dead-at-proxy** iff the best-τ proxy Δ is flat / ≤ perm-null p95 on both datasets.
- **GO-FOR-CEREMONY** iff oracle ceiling ≥ +0.030 on ≥1 dataset AND the proxy shows real (>null) signal.
- **Honest-wall caveat (declared up front):** the oracle bounds the **VOTE-side** (label-reweighting)
  mechanism the head could exploit; it does NOT bound the head's **representation-reshaping** (raw-key
  proxy limit, `READOUT_FORENSIC_RECON.md §5`). That residual is exactly the F44-capped axis (EN
  label-limited at 5 levels; "rotation not Pareto") and is what the admissibility ruling would gate.

---

## 1. DATA RECON — 3-class counts per split (train+dev ONLY; test never read)

Source: `/data/jehc223/Multihateclip/{English,Chinese}/annotation(new).json` (the `Label` field), mapped
onto the cache `ids` per split. `harmful_vs_normal` binary in every cache is **100% consistent** with
{Offensive,Hateful}→1, Normal→0 (guard `bin_consistent=True` all arms).

| dataset | split | n | Normal | Offensive | Hateful | binary pos | Offensive as %of pos |
|---|---|---|---|---|---|---|---|
| **EN** (MHC) | train | 549 | 381 | **123** | 45 | 168 | **73.2%** |
| EN (MHC) | dev | 80 | 55 | 18 | 7 | 25 | 72.0% |
| **ZH** (MHC_zh) | train | 579 | 399 | **113** | 67 | 180 | **62.8%** |
| ZH (MHC_zh) | dev | 78 | 50 | 18 | 10 | 28 | 64.3% |

**The load-bearing structural fact:** Offensive is the MAJORITY of the positive class on both datasets
(EN 73%, ZH 63%). The graded lever proposes to treat this majority-of-positives as *less* positive.
Because Offensive genuinely IS on the harmful side of the deployed boundary, down-weighting it moves
true-positive mass toward the Normal side — the F44 "the split refines WITHIN-positive structure we
merge, it does not sharpen the harmful-vs-Normal boundary" prediction, made concrete by the counts.

---

## 2. RESULTS (raw; source `refine-logs/GRADEDLBL_PREGATE_OUT.json`)

Machinery parity: ZH LoRA_HF baseline loo **0.8717948718** / devtrain **0.8589743590** = banked ro_L28
**bit-exact** (`parity_check.loo_match / devtrain_match = True`). Degenerate-recovery: w_off=+1 ≡ binary,
Δacc=0 exact, all 6 arms.

### 2.1 Operator (i) — graded proxy Δacc(graded − binary), cutoff 0 (verbatim)

| dataset·arm | binary acc | τ=0.25 (w=−0.50) | τ=0.50 (w=0.00) | τ=0.75 (w=+0.50) |
|---|---|---|---|---|
| EN·loo | 0.7500 | −0.0500 (fix11/brk15/net−4) | −0.0250 (10/12/−2) | −0.0250 (5/7/−2) |
| EN·devtrain | 0.7500 | −0.0500 (10/14/−4) | −0.0125 (9/10/−1) | +0.0000 (4/4/0) |
| ZH·loo | 0.8718 | −0.1538 (7/19/−12) | −0.0513 (6/10/−4) | −0.0128 (4/5/−1) |
| ZH·devtrain | 0.8590 | −0.1410 (7/18/−11) | −0.0385 (7/10/−3) | −0.0128 (3/4/−1) |
| EN·loo (LoRA, sens.) | 0.7250 | −0.0250 (13/15/−2) | −0.0375 (8/11/−3) | +0.0125 (6/5/+1) |
| EN·devtrain (LoRA, sens.) | 0.7625 | −0.0750 (10/16/−6) | −0.0875 (6/13/−7) | −0.0125 (5/6/−1) |

**Every honest proxy Δ is ≤ 0** on both primary datasets and every τ (best = τ=0.75, the closest-to-binary
setting, still 0 or negative). The monotone dose-response is unambiguous: **the more Offensive is
down-weighted, the worse dev accuracy** (ZH·loo τ=0.25 = −0.1538, net −12 of 78). The single positive
cell is one sensitivity arm at +0.0125 (1 item, EN-LoRA·loo τ=0.75) — inside the ±0.014 band, and it dies
under the null (§2.3). The proxy provides **no honest gain**.

### 2.2 Operator (ii) — label-oracle ceiling (gold-cheat upper bound)

| dataset·arm | DOP ceiling (w_off*) | B5 ceiling (w_off*) | **BINDING** |
|---|---|---|---|
| EN·loo | +0.0000 (w=+1.00) | +0.0250 (w=−0.50) | **+0.0250** |
| EN·devtrain | +0.0000 (w=−0.10) | +0.0000 (w=−0.70) | +0.0000 |
| ZH·loo | +0.0256 (w=+0.70) | +0.0000 (w=+0.45) | **+0.0256** |
| ZH·devtrain | +0.0128 (w=+0.30) | +0.0128 (w=−0.20) | +0.0128 |
| EN·loo (LoRA, sens.) | +0.0250 | +0.0125 | +0.0250 |
| EN·devtrain (LoRA, sens.) | +0.0000 | +0.0250 | +0.0250 |

**Per-dataset binding oracle ceiling (max over arms):** EN **+0.0250**, ZH **+0.0256**. **BOTH < +0.030**
even under the fully gold-cheating upper bound (dev labels pick w_off AND cutoff). Note the ceiling is
tiny AND fragile: EN's +0.0250 is a re-thresholded (B5) reading whose DOP twin is +0.0000 (binary is
already optimal at the deployed cutoff); ZH's +0.0256 is +0.0256 = 2 dev items on n=78. The honest proxy
(§2.1) lands far below these ceilings, at the negative end.

### 2.3 Targeted Offensive-permutation null (F63) — best-τ proxy, cutoff 0

| dataset·arm | best τ | obs Δacc | null p95 | null mean | null max | obs > p95? |
|---|---|---|---|---|---|---|
| EN·loo | 0.50 | −0.0250 | −0.0125 | −0.0379 | +0.0000 | **No** |
| EN·devtrain | 0.75 | +0.0000 | +0.0250 | −0.0039 | +0.0375 | **No** |
| ZH·loo | 0.75 | −0.0128 | +0.0256 | +0.0046 | +0.0385 | **No** |
| ZH·devtrain | 0.75 | −0.0128 | +0.0128 | −0.0052 | +0.0385 | **No** |
| EN·loo (LoRA, sens.) | 0.75 | +0.0125 | +0.0375 | +0.0187 | +0.0375 | **No** |
| EN·devtrain (LoRA, sens.) | 0.75 | −0.0125 | −0.0125 | −0.0210 | +0.0000 | **No** |

**The F63 warning is fully realized.** Not a single arm's observed Δ exceeds its permutation-null 95th —
down-weighting the TRUE Offensive set is **no better than down-weighting a random equal-size positive
subset** (obs ≈ or below null mean). Worse: the null MAX (+0.0385 ZH) exceeds the true-Offensive oracle
ceiling — a *random* positive-subset reweighting can chance into more headroom than the Offensive
structure yields. **The "Offensive" identity carries no decision-relevant signal for the deployed
boundary.**

---

## 3. HONEST WALLS

- **F44 arithmetic ceiling (confirmed, mechanistic).** The deployed boundary is harmful-vs-Normal; the
  3-class split refines the WITHIN-positive Hateful-vs-Offensive structure the binary task merges. §1
  quantifies why this cannot help: Offensive is 63–73% of the positive class, so any monotone
  down-weighting drags true positives toward Normal (§2.1 dose-response). The oracle **does** touch the
  boundary (w_off shifts Offensive across the 0 cutoff) — and even so the best gold-cheat it can buy is
  +0.025/+0.026 < +0.030.
- **±0.014 band / small dev.** EN dev n=80, ZH dev n=78 (28 pos). One item = ±0.0125–0.0128. The binding
  ceilings (+0.0250/+0.0256 ≈ 2 items) sit inside 2× the single-item quantum and below the F63 null max.
- **Proxy limit (declared §0.5).** This gate bounds only the VOTE-side (label-reweighting) mechanism.
  The head's representation-reshaping (Hateful/Offensive geometric separation) is a residual the raw-key
  proxy cannot measure — but it is the F44-capped axis (EN label-limited at 5 levels: F44/F50/F55/F58/F65),
  and it is exactly what the pending admissibility ruling would gate before any ~0.3 GPU-h head retrain.

---

## 4. VERDICT

**PARK** (recommendation `PARK`; `oracle_all_below_kill=True`, `any_proxy_alive=False`).

Both pre-declared kill conditions (§0.5) fire together:
1. **Arithmetic PARK:** BINDING oracle ceiling **< +0.030 on BOTH datasets** (EN +0.0250, ZH +0.0256) —
   the gold-cheat upper bound of *any* monotone Offensive reweighting cannot reach the goal bar.
2. **dead-at-proxy:** the honest proxy Δ is **≤ 0 on both primary datasets, every τ**, and **no arm's
   best-τ Δ exceeds its F63 permutation-null p95** — the Offensive identity carries no boundary signal.

Machinery is valid (parity bit-exact; degenerate-recovery exact; Fano-equivalent sanity via the
label-independent linear decomposition), so the flatness is a real negative, not a broken pipe.

**Prior after gate.** litsweep5 priced this ≥+1 either dataset ~15% / goal-bar ~1–2%. The pregate moves
the goal-bar prior **down to <2% (effectively closed for the vote-side mechanism)**: the gold-cheat
oracle misses +0.030 on both datasets, the honest proxy is uniformly negative, and the effect is
null-indistinguishable from random positive-subset reweighting. The only unbounded residual — head
representation-reshaping — is F44-capped and requires the user's admissibility ruling to be spent at all.

**Admissibility ruling (evidence for the user micro-ruling).** The pregate is read-only on the SAME
own-split train annotation at finer granularity — no gold time-span / target-group injection (the F44
veto), no test touch, no method committed. But the ruling is **moot for the performance goal**: there is
no gain to admit (oracle < bar, proxy dead). Recommend PARK regardless of the ruling; revisit only if a
future representation-side probe is separately motivated.

**Orchestrator: submit nothing.** 0 GPU-h spent. Binding close = orchestrator spot-check (ISR/CTF/APX
precedent) of `GRADEDLBL_PREGATE_OUT.json` + parity.

---

## 5. PROVENANCE
- Script: `scripts/analysis/gradedlbl_pregate.py` (sha256 `59cc253d…e914d77d`), CPU-only, seed 20260725.
- Caches (sha256, train / dev): EN frozen `05a9b2de…4940409` / `cd5d4c7d…974cd50c`; ZH LoRA_HF
  `b2e8e78d…a5e01f1d` / `4c07af75…37e4f5d3c`; EN LoRA_HF (sens.) `50293e9a…208e73b0` / `404a3a07…423df8983`.
- 3-class source: `/data/jehc223/Multihateclip/{English,Chinese}/annotation(new).json` (`Label`); binary
  map = `scripts/prep_mhc.py` LABEL_SCHEMES `harmful_vs_normal`.
- Parity anchor: `refine-logs/READOUT_SCREEN_OUT.json` ro_L28 (== deployed no-suffix cache).
- Templates: `refine-logs/ISR_PREGATE_RECORD.md`, `refine-logs/B5_PROBE_RECORD.md`; vote/fused-key
  `scripts/analysis/readout_screen.py` (`cross_channel_router_gate.py:73-79`). Candidate:
  `refine-logs/LITSWEEP5_HATEMM_EN.md` §3-cand-1 (36d833e).
