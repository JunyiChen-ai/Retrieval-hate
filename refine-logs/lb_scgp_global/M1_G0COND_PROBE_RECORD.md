# M1 G0-cond CONDITIONAL-INFORMATION PROBE RECORD — lb_scgp_global M1 sealed cache

Date: 2026-07-13
Author = **Claude Opus 4.8** (`claude-opus-4-8`, 1M ctx), fresh **G0-cond gate executor** for the
`lb_scgp_global_r2` A-line. Distinct from m1-prep (v2 author/freezer), the v2 code/execution
reviewers, the seal executor (`M1_CACHE_V2_EXECUTION_RECORD.md`), and the post-seal
information-content reviewer (`M1_POST_SEAL_INFO_CONTENT_REVIEW.md`).

**Scope / compliance.** Zero GPU, zero SLURM, CPU-only, conda `HateVideo`, login node, runtime ~2 min.
Read-only over the sealed M1 cache + frozen feature `.pt` caches + `data/gt/*/train.jsonl` gold.
**Train gold labels are used only as probe targets** — this is compliant: the cache is SEALED
(`cache_seal_decision.json.labels_enter_after_this_seal_only=true`), and the reflection §4 gate
explicitly permits gold-for-probing ("oracle 上限版(gold 版信号,合规:gold 仅用于 probing)").
No validation/test content or label was read. Write scope = this file only. Not committed (archiver
handles commits).

This document executes the **mandatory zero-cost G0-cond conditional-information gate**
(`research-wiki/REFLECTION_mllm_integration_failures.md` §4) that `M1_POST_SEAL_INFO_CONTENT_REVIEW.md`
§3 called for before any GPU spend, and issues the A-line PROCEED-vs-PAUSE verdict.

---

## 0. VERDICT

**`A_LINE_PAUSE`.** The sealed certificate carries **no positive conditional label information beyond
the frozen encoder features** (real-A codelength gain CI never excludes 0 in the positive direction;
in 2 of 4 representation cells the certificate *increases* codelength), and the **oracle kill-switch
fires**: even a *perfect* certificate at the measured 8.74% / 6.91% coverage projects to a test Δacc
ceiling of **≤ +0.028** (analytic) / **≤ +0.0044** (probe-realized) — an order of magnitude below the
+0.030 + 0.01 = **+0.040** decision bar. The oracle@coverage-1.0 reference arm *does* clear the bar in
some cells (prices a v3 repair), **but the real-A covered-only arm proves the certificate is
noise-quality even where it parsed** (conditional codelength gain ≈ 0, CI includes 0), so a v3 repair
would raise coverage of a zero-information signal — it would **not** approach the gold ceiling.
Verdict is therefore plain **PAUSE**, not `A_LINE_PAUSE_BUT_V3_VIABLE`. Do not spend the 264 M2+M3
GPU-hours; do not build v3. Archive M0 + M1-sealed as the clean stopping point; redirect GPU to C-line.

---

## 1. Which Z the comparator conditions on — resolved to BOTH (ambiguity → run both)

The C2 comparator (`EXPERIMENT_PLAN.machine.json` `comparator_freeze`, run[7]) is the
"**frozen moving strongest same-protocol non-MLLM comparator**" doing
`ordinary_full_video_train_memory_top20_knn` (`immutable_contract.final_inference`). The plan names no
concrete backbone for `Z0` — `FINAL_PROPOSAL.md:128` reads the encoder dim from `Z0.shape[1]` and the
comparator is only qualified as "non-MLLM / strongest same-protocol". The project's floor evidence
(`exp-encoder-3seed.md`, commit 040adb8) shows the two live floors are **frozen-CLIP**
(`openai_clip-vit-large-patch14-336_HF`) and **frozen-Qwen** (`Qwen2.5-VL-7B-Instruct_HF`), with the
strongest floor being dataset-dependent (Qwen wins HateMM; on MHC-EN both sit in the 0.77–0.80 band).
Because the comparator's representation is ambiguous, per the task I ran the probe against **BOTH**
cached representations. Both give the same verdict.

Frozen feature caches (this session, `torch.load`):

| ds | encoder | file | img_feats | text_feats | labels==gold |
|---|---|---|---|---|---|
| MHC | CLIP | `data/CLIP_Embedding/MHC/train_openai_clip-vit-large-patch14-336_HF.pt` | [549,1024] | [549,768] | 0 mismatches |
| MHC | Qwen | `data/CLIP_Embedding/MHC/train_Qwen2.5-VL-7B-Instruct_HF.pt` | [549,3584] | [549,3584] | 0 mismatches |
| MHC_zh | CLIP | `data/CLIP_Embedding/MHC_zh/train_openai_clip-vit-large-patch14-336_HF.pt` | [579,1024] | [579,768] | 0 mismatches |
| MHC_zh | Qwen | `data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct_HF.pt` | [579,3584] | [579,3584] | 0 mismatches |

Alignment verified this session: CLIP `ids` == Qwen `ids`; certificate consensus `video_id` set ==
feature `ids` set (549 / 579); `.pt` `labels` == `parent_video_binary_label` in
`data/gt/{MHC,MHC_zh}/train.jsonl` with **0 mismatches**. `Z = zscore(concat(img_feats, text_feats))`
= CLIP 1792-d / Qwen 7168-d.

**Why raw frozen Z (before the trainable projection) is the fail-closed choice:** the RGCL trainable
projection is a deterministic (label-fit) function of these frozen features, so by the data-processing
inequality it cannot expose *more* label-relevant structure to the certificate than the raw features
already do minus what the projection captures — conditioning on raw Z gives the certificate the
**maximum** benefit of the doubt. If A adds no conditional information even over raw Z, it adds none
over the real floor → a KILL here is maximally robust. (Cross-check: the Z-only probe CV accuracy
0.760–0.813 sits inside the known floor band ≈0.76–0.81 from `exp-encoder-3seed.md`, so Z is not an
artificially weak comparator that would inflate A's apparent gain.)

---

## 2. Probe design (deviations documented)

- **Probe class** = L2-regularized logistic regression on standardized features — the RGCL final
  linear classifier over the projected embedding, capacity-matched (`REFLECTION §4`: "探针容量与实际
  head 匹配"). Same regularization `C` for `g(Z)` and `g'([Z,A])` (the only difference between arms is
  the appended A columns → an honest marginal-information test). `C` chosen per (ds,enc) by inner
  5-fold stratified CV maximizing Z-only accuracy over {1e-3,1e-2,1e-1,1}, then **reused** for every
  arm of that cell. Chosen: MHC CLIP C=0.01, MHC Qwen C=0.01, MHC_zh CLIP C=0.001, MHC_zh Qwen C=0.1.
- **CV** = `RepeatedStratifiedKFold(n_splits=5, n_repeats=5)` = 25 fits/model; `StandardScaler` fit on
  the train fold only (no leakage). Each video is held out 5×; per-video held-out mean bits and mean
  accuracy are averaged over the 5 repeats.
- **Codelength (MDL, preferred over accuracy)** = held-out negative log-likelihood in **bits**:
  `-log2 p_model(y_true|x)` summed over held-out predictions. `ΔL = L(Z) − L([Z,A])` bits saved by A
  (positive ⇒ A reduces codelength ⇒ A carries conditional info). Reported as total bits and
  mean bits/video with a percentile bootstrap CI (`B=5000`, RNG seed 20260713) over the paired
  per-video ΔNLL. This is the decision-rule-(i) statistic.
- **A encoding** = per video, each observable's consensus **state one-hot + mean confidence** (mean
  over the R=4 replicas; identical across replicas for parse-ok videos). **Deviation:** the schema
  (`scgp_global_cert_v2.schema.json`) has **9** observables (8 tri-state {contradicted,supported,
  unresolved} + 1 modality {6 states}), not the "8" in the task text; I encode all 9 (8·(3+1)+1·(6+1)
  = **39 dims**) — more columns can only *help* A, so this is conservative for a KILL. For
  transport-fallback (uncovered) videos A = the single literal constant (all-unresolved, conf 0), i.e.
  A is constant on the 501/539 uncovered videos exactly as the task specifies.
- **Four arms** (all with the same probe/CV/bootstrap machinery):
  1. **real-A full-set** — `[Z, A_real]` vs `Z`, all 549/579 videos (A constant on uncovered).
  2. **real-A covered-only** — restricted to the covered subset (n=48/40); underpowered, flagged.
  3. **oracle-A @ measured coverage** — A = gold label revealed **only on covered videos**
     (3-way one-hot {uncovered, covered-neg, covered-pos}); uncovered = constant. Upper-bounds any
     certificate at the measured coverage.
  4. **oracle-A @ coverage 1.0** — A = one-hot(gold) for **all** videos (2-dim). Prices the v3 repair
     ceiling (what ~100% parse could at MOST deliver if certs were as good as gold).

**Commands (this session):**
- Probe: `conda run -n HateVideo python3 refine-logs/lb_scgp_global/M1_G0COND_PROBE.py
  refine-logs/lb_scgp_global/M1_G0COND_PROBE_OUT.json` (script + full results JSON persisted alongside
  this record for reproducibility).
- Alignment / gold / coverage checks: `torch.load` + `json` over the four `.pt`, the two
  `cache_manifest.json` (`consensus` field), the two `cache.jsonl`, and `data/gt/*/train.jsonl`.

**Coverage / class facts (this session, reproduces `M1_POST_SEAL_INFO_CONTENT_REVIEW.md`):**

| ds | n | covered (non-constant consensus) | coverage c | base rate (pos) |
|---|---|---|---|---|
| MHC | 549 | **48** | **0.0874** | 0.3060 |
| MHC_zh | 579 | **40** | **0.0691** | 0.3109 |

---

## 3. RESULTS — four arms × both representations

`accZ`/`accZA` = Z-only vs [Z,A] CV accuracy; `Δacc` = accZA−accZ with 95% bootstrap CI; `Δbits/vid`
= mean per-video codelength saved by A (bits) with 95% CI; `Δbits_tot` = total bits saved over the set.
**Positive Δbits ⇒ A helps.**

### 3.1 MHC (n=549, coverage 8.74%)

| enc | arm | n | accZ | accZA | Δacc [95% CI] | Δbits/vid [95% CI] | Δbits_tot |
|---|---|---|---|---|---|---|---|
| CLIP | real-A full-set | 549 | 0.7621 | 0.7596 | −0.0026 [−0.0066,+0.0011] | **−0.00844 [−0.01510,−0.00291]** | −4.6 |
| CLIP | real-A covered-only | 48 | 0.5292 | 0.5333 | +0.0042 [−0.0292,+0.0375] | −0.00100 [−0.01908,+0.01547] | −0.0 |
| CLIP | oracle-A @ measured cov | 549 | 0.7621 | 0.7665 | +0.0044 [+0.0000,+0.0087] | +0.01784 [+0.01114,+0.02514] | +9.8 |
| CLIP | oracle-A @ coverage 1.0 | 549 | 0.7621 | 0.8805 | +0.1184 [+0.0991,+0.1377] | +0.29464 [+0.26253,+0.32755] | +161.8 |
| Qwen | real-A full-set | 549 | 0.8047 | 0.8044 | −0.0004 [−0.0033,+0.0026] | **−0.00454 [−0.00899,−0.00083]** | −2.5 |
| Qwen | real-A covered-only | 48 | 0.6208 | 0.6250 | +0.0042 [+0.0000,+0.0125] | +0.00161 [−0.00477,+0.00748] | +0.1 |
| Qwen | oracle-A @ measured cov | 549 | 0.8047 | 0.8087 | +0.0040 [+0.0007,+0.0077] | +0.01095 [+0.00708,+0.01514] | +6.0 |
| Qwen | oracle-A @ coverage 1.0 | 549 | 0.8047 | 0.8444 | +0.0397 [+0.0310,+0.0492] | +0.14737 [+0.12657,+0.16845] | +80.9 |

### 3.2 MHC_zh (n=579, coverage 6.91%)

| enc | arm | n | accZ | accZA | Δacc [95% CI] | Δbits/vid [95% CI] | Δbits_tot |
|---|---|---|---|---|---|---|---|
| CLIP | real-A full-set | 579 | 0.7599 | 0.7603 | +0.0003 [−0.0010,+0.0017] | +0.00038 [−0.00256,+0.00291] | +0.2 |
| CLIP | real-A covered-only | 40 | 0.7250 | 0.7400 | +0.0150 [+0.0000,+0.0400] | −0.00076 [−0.00329,+0.00155] | −0.0 |
| CLIP | oracle-A @ measured cov | 579 | 0.7599 | 0.7606 | +0.0007 [−0.0007,+0.0021] | +0.00402 [+0.00260,+0.00557] | +2.3 |
| CLIP | oracle-A @ coverage 1.0 | 579 | 0.7599 | 0.7941 | +0.0342 [+0.0242,+0.0449] | +0.11054 [+0.10047,+0.12110] | +64.0 |
| Qwen | real-A full-set | 579 | 0.8128 | 0.8086 | −0.0041 [−0.0086,+0.0000] | −0.00969 [−0.02333,+0.00124] | −5.6 |
| Qwen | real-A covered-only | 40 | 0.7100 | 0.7150 | +0.0050 [+0.0000,+0.0150] | −0.00212 [−0.00748,+0.00265] | −0.1 |
| Qwen | oracle-A @ measured cov | 579 | 0.8128 | 0.8142 | +0.0014 [−0.0017,+0.0045] | +0.01157 [+0.00684,+0.01666] | +6.7 |
| Qwen | oracle-A @ coverage 1.0 | 579 | 0.8128 | 0.8639 | +0.0511 [+0.0415,+0.0618] | +0.24132 [+0.20716,+0.27731] | +139.7 |

### 3.3 Projection: oracle @ measured coverage → test kNN Δacc ceiling

Two independent upper bounds on what a **perfect** certificate at the measured coverage could deliver:

**(a) Probe-realized (capacity-matched logistic, above):** oracle@measured Δacc ≤ **+0.0044** (max,
MHC/CLIP), CI upper bound ≤ +0.0087 across all four cells.

**(b) Analytic coverage-scaled ceiling** = `c · (1 − a_cov,Z)` (perfectly relabel every covered video
Z got wrong; a strictly looser bound than the regularized probe):

| ds | enc | c | a_cov,Z | ceiling c·(1−a_cov) | E[covered nbrs] = 20·c |
|---|---|---|---|---|---|
| MHC | CLIP | 0.0874 | 0.7250 | **+0.0240** | 1.75 |
| MHC | Qwen | 0.0874 | 0.6833 | **+0.0277** | 1.75 |
| MHC_zh | CLIP | 0.0691 | 0.7650 | **+0.0162** | 1.38 |
| MHC_zh | Qwen | 0.0691 | 0.7350 | **+0.0183** | 1.38 |

Max analytic ceiling = **+0.0277 < +0.040**. **Concrete top-20 kNN flip argument:** the certificate's
only channel to a test prediction is `ordinary_full_video_train_memory_top20_knn`; it warps the
train-train Gram by a rank-≤8, small-magnitude amount (`FINAL_PROPOSAL.md:189-291`,
`rho_row=0.05·√(N−1)`) touching *only* the ≤48/40 covered memory rows. A test query's label = majority
of its top-20 nearest memory videos; the expected number of covered videos among those 20 is
`20·c ≈ 1.75 (MHC) / 1.38 (MHC_zh)`. Even granting the absurd best case that a perfect certificate
controls the vote of *every* covered neighbor in the decisive direction, the maximum vote swing is
≈1.75/1.38 out of 20, which flips a majority only for the small fraction of test videos whose hate-vote
count already sits within ~2 of the 10/10 boundary — and absent label information (real-A, §3.1/3.2)
those swings are as often wrong-direction as right. So the kNN channel is strictly *weaker* than the
direct logistic probe, and both bounds land at ≈`c`-scale, far under +0.040.

---

## 4. DECISION RULE EVALUATION (pre-declared, fail-closed)

> A-line PROCEEDS only if BOTH (i) conditional codelength gain CI excludes 0 AND (ii) projected test
> Δacc > +0.030 + 0.01 = **+0.040**. The oracle arm failing (ii) kills regardless of (i).

**(i) Conditional codelength gain CI excludes 0 (positive direction) — FAIL.**
Real-A (the actual sealed certificate) `Δbits/vid` 95% CI, all four cells:
- MHC/CLIP **[−0.01510, −0.00291]** — entirely negative (A *increases* codelength).
- MHC/Qwen **[−0.00899, −0.00083]** — entirely negative.
- MHC_zh/CLIP [−0.00256, +0.00291] — straddles 0.
- MHC_zh/Qwen [−0.02333, +0.00124] — straddles 0 (mean negative).
No cell shows a **positive** codelength gain with a CI excluding 0. In half the cells the certificate
is actively *anti-informative* (the 39 A columns add estimation variance with no signal). (i) FAILS.

**(ii) Projected test Δacc > +0.040 — FAIL (kill-switch).**
Oracle-A @ measured coverage — the perfect-certificate ceiling — is **+0.0044 / +0.0040 / +0.0007 /
+0.0014** (probe) with analytic ceiling **≤ +0.0277** (§3.3). Every value is far below +0.040; the
best cell reaches ~70% of the bar only under the looser analytic bound, and the C2 rule additionally
demands +0.030 on **both** datasets × **both** metrics × 3 seeds × Holm — unreachable when even a
single-cell perfect-signal ceiling is < +0.028. (ii) FAILS decisively. **The oracle failing (ii)
kills the entire certificate signal family regardless of (i) — the signal type is coverage-capped.**

**BOTH gates fail ⇒ `A_LINE_PAUSE`.**

**v3-viability sub-check (oracle@1.0 vs real-A covered-only).** The oracle @ coverage-1.0 arm
(prices "what if certs were as good as gold at 100% parse") clears +0.040 in 2 of 4 cells
(MHC/CLIP +0.1184, MHC_zh/Qwen +0.0511) and lands +0.034–0.040 in the other two — i.e. a *gold-quality*
certificate at full coverage could matter. **But that ceiling is unreachable**, because the
**real-A covered-only** arm measures the actual certificate's quality on the videos where it *did*
parse: Δbits/vid CIs are **[−0.019,+0.015] / [−0.005,+0.007] / [−0.003,+0.002] / [−0.007,+0.003]** —
all include 0, means ≈ 0; Δacc within ±0.015 noise on n=48/40. The certificate is **noise-quality even
where it parsed**. A v3 infra-repair would therefore propagate a zero-conditional-information signal to
more videos, not the gold ceiling. Hence **NOT** `A_LINE_PAUSE_BUT_V3_VIABLE` — plain **PAUSE**.

**Adversarial check, both directions:**
- *Not over-killing:* the covered subset is genuinely non-constant, and I gave A every advantage
  (raw pre-projection Z by DPI, all 9 observables + confidences, same C, oracle arms). It still adds
  no conditional information.
- *Not wishfully proceeding:* real-A codelength is ≤ 0 in every cell, the oracle@measured ceiling is
  ~10× under the bar, and real-A covered-only refutes the v3 escape hatch.

---

## 5. Numbers → source index (provenance)

- Feature caches, id/label alignment (§1): `torch.load` over the four `data/CLIP_Embedding/*/train_*.pt`;
  `ids[0]` lists; `labels` vs `data/gt/{MHC,MHC_zh}/train.jsonl` `parent_video_binary_label` (0
  mismatches, this session).
- Coverage/covered set, base rate (§2): `cache_manifest.json.consensus` non-constant test
  (`artifacts/lb_scgp_global/v1/m1/cache/{MHC,MHC_zh}/cache_manifest.json`); 48/40 covered reproduces
  `M1_POST_SEAL_INFO_CONTENT_REVIEW.md` §2b/§2c.
- A encoding (§2): `cache.jsonl` per-replica `observables[*].{state,confidence}` (mean over R=4) +
  `consensus` states; schema states from `schemas/lb_scgp_global_r2/scgp_global_cert_v2.schema.json`
  (8 tri-state {contradicted,supported,unresolved} + modality {multi_modal,single_modal,text_audio,
  unresolved,visual_audio,visual_text}); consensus tri ints {−1,0,1}→{contradicted,unresolved,
  supported} verified this session.
- All arm results (§3): `refine-logs/lb_scgp_global/M1_G0COND_PROBE.py` →
  `refine-logs/lb_scgp_global/M1_G0COND_PROBE_OUT.json`
  (RepeatedStratifiedKFold 5×5, L2 logistic, StandardScaler per fold, bootstrap B=5000 seed 20260713).
- Analytic ceiling / kNN flip (§3.3): `c` and `a_cov,Z` from the probe (`a_covered_Zonly`);
  compile mechanics `FINAL_PROPOSAL.md:189-291` (common basis r_max≤8, proximal caps),
  `immutable_contract.final_inference` = top-20 kNN, `test_n` 161/149.
- Decision bar / C2 rule (§4): `EXPERIMENT_PLAN.machine.json` `statistics_protocol.joint_success_rule`
  (+0.030 both-metrics/both-datasets/3-seeds/Holm), `claim_map.C2`; noise margin +0.01 and the
  +0.030 projection bar from `REFLECTION_mllm_integration_failures.md` §4 + §D3 (±1–2 pt floor).
- Institution: `research-wiki/REFLECTION_mllm_integration_failures.md` §4 (G0-cond spec), §5 (A-line
  disposition); GPU budget 264 GPU-h (M2 48 + M3 216) from `M1_POST_SEAL_INFO_CONTENT_REVIEW.md` §0.

## Required statements
- No performance/accuracy claim on any held-out benchmark; all accuracy/codelength numbers are
  train-only cross-validation on the sealed cache, used solely to measure conditional information.
- Only gold read = `parent_video_binary_label` on the train split, used as probe targets on the SEALED
  cache (compliant, `REFLECTION §4`). No validation/test content or label was opened. No GPU, no
  SLURM, no MLLM/OCR/network. Write scope = this file. Not committed (archiver handles commits).
