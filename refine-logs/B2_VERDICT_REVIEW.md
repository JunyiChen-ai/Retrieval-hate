# B2 Verdict Review — frozen Qwen2.5-VL-32B encoder-scale test (job 13146)

**Reviewer:** fresh independent verdict reviewer (zero prior context; read-only + CPU
re-parse; no GPU, no SLURM, no commits, no test re-touch).
**Date:** 2026-07-14.
**Under review:** experiment B2 — frozen Qwen2.5-VL-32B-Instruct (5120-d) encoder vs
frozen-CLIP (primary) and vs frozen Qwen2.5-VL-7B (secondary), 3 datasets × 3 seeds,
archive OFF, dual protocol, job 13146.
**Sources:** pre-registration `research-wiki/experiments/exp-encoder-32b-b2.md` (rev r1);
execution record `refine-logs/B2_EXECUTION_RECORD.md`; primary trainlogs
`slurm/logs/enc3s_*_Qwen2.5-VL-32B-Instruct_HF_seed*_13146.trainlog`; reference logs
job 12850 (HateMM CLIP/7B; MHC-EN CLIP s0/1/2 + 7B s0), arcbase 12275/12276 (MHC-EN 7B
s1/s2), job 13115 (MHC-ZH CLIP/7B).
**Decision rule applied verbatim** from the prereg / `exp-encoder-3seed.md:73-85`:
per dataset × protocol, paired Δ = (32B − ref) at each seed; **PASS iff meanΔacc ≥ +0.030
AND meanΔmacro-F1 ≥ +0.030 AND 3/3 seeds sign-positive.**

---

## Task 1 — PROVENANCE (independent re-read of every reading)

I re-parsed every trainlog from scratch (regex over full text; val-sel = epoch ≥ warmup 5
maximizing Val_Retrieval acc with roc tie-break; final = epoch 29; macro-F1 read from the
`macroF1:` lines). Line numbers are `grep -n` positions of the `Test_Retrieval` macroF1
line.

### 1a. The 9 new 32B runs (18 readings) vs the execution record's table

**All 18 readings reproduce the execution record EXACTLY** (epoch, F1, acc, roc, and the
cited line numbers all match to 4 dp):

| dataset | seed | val-sel (ep / F1 / acc / roc / line) | final (ep / F1 / acc / roc / line) |
|---|---|---|---|
| HateMM | 0 | e25 / 0.8724 / 0.8791 / 0.9210 / 291 | e29 / 0.8197 / 0.8279 / 0.9195 / 332 |
| HateMM | 1 | e26 / 0.8552 / 0.8605 / 0.9234 / 301 | e29 / 0.8638 / 0.8698 / 0.9197 / 332 |
| HateMM | 2 | e23 / 0.8547 / 0.8605 / 0.9193 / 273 | e29 / 0.8301 / 0.8372 / 0.9151 / 334 |
| MHC-EN | 0 | e14 / 0.5665 / 0.7081 / 0.7566 / 163 | e29 / 0.6674 / 0.7516 / 0.8271 / 299 |
| MHC-EN | 1 | e28 / 0.6972 / 0.7578 / 0.8203 / 288 | e29 / 0.7070 / 0.7640 / 0.8398 / 298 |
| MHC-EN | 2 | e13 / 0.6618 / 0.7391 / 0.7861 / 157 | e29 / 0.6940 / 0.7640 / 0.8302 / 302 |
| MHC-ZH | 0 | e15 / 0.7016 / 0.7584 / 0.8581 / 174 | e29 / 0.7245 / 0.7785 / 0.8498 / 301 |
| MHC-ZH | 1 | e23 / 0.7221 / 0.7718 / 0.8498 / 244 | e29 / 0.7517 / 0.7919 / 0.8626 / 299 |
| MHC-ZH | 2 | e6  / 0.7006 / 0.7785 / 0.8476 / 90  | e29 / 0.7296 / 0.7651 / 0.8598 / 298 |

### 1b. Reference arms re-read from THEIR primary logs (NOT trusting any doc transcription)

Every reference reading the prereg reused matches its primary log to 4 dp:

- **HateMM CLIP (12850)** s0 vs 0.8172/0.8279 fn 0.7997/0.8186 · s1 vs 0.8163/0.8279 fn
  0.7822/0.8047 · s2 vs 0.7920/0.8047 fn 0.7988/0.8140 — **MATCH.**
- **HateMM 7B (12850)** s0 vs 0.8606/0.8698 fn 0.8507/0.8605 · s1 vs 0.8586/0.8651 fn
  0.8514/0.8605 · s2 vs 0.8753/0.8837 fn 0.8753/0.8837 — **MATCH.**
- **MHC-EN CLIP (12850)** s0 vs 0.7113/0.7826 fn 0.7145/0.7640 · s1 vs 0.6034/0.7329 fn
  0.7159/0.7826 · s2 vs 0.6997/0.7702 fn 0.7303/0.7888 — **MATCH.**
- **MHC-EN 7B** s0 (12850) vs 0.7378/0.7888 fn 0.7596/0.8012 · s1 (arcbase 12275) vs
  0.7283/0.7826 fn 0.7203/0.7702 · s2 (arcbase 12276) vs 0.6997/0.7702 fn 0.7475/0.7826 —
  **MATCH** (and the s1/s2 provenance is exactly the Rev-1-corrected arcbase logs, not 12850).
- **MHC-ZH CLIP (13115)** s0 vs 0.7706/0.8054 fn 0.7706/0.8054 · s1 vs 0.7579/0.8054 fn
  0.7542/0.8054 · s2 vs 0.7742/0.8121 fn 0.7913/0.8322 — **MATCH.**
- **MHC-ZH 7B (13115)** s0 vs 0.7412/0.7919 fn 0.7864/0.8188 · s1 vs 0.7871/0.8121 fn
  0.7759/0.8054 · s2 vs 0.7759/0.8054 fn 0.7514/0.7852 — **MATCH.**

**PROVENANCE VERDICT: PASS.** All 18 32B readings and all 24 reference readings verified
against primary logs. No transcription error found in the execution record or the prereg
reference tables.

---

## Task 2 — GATES

### G-repro (config-1 HateMM-s0 sanity — non-degenerate, 5120-d wiring): PASS

`enc3s_HateMM_Qwen2.5-VL-32B-Instruct_HF_seed0_13146.trainlog`:
- 5120-d wiring confirmed at the source: log line 2/3 `Image/Text feature dimension: 5120`;
  head builds `Linear(in_features=5120, out_features=1024)` for both `img_proj`/`text_proj`
  (lines 7/11). No hard-coded 3584/5120 on the path — dim auto-inferred from the loaded `.pt`.
- Trained all 30 epochs with well-formed `Val_/Test_Retrieval` lines.
- Non-degenerate: val-sel Test F1 0.8724 / acc 0.8791, final 0.8197 / 0.8279 — not in the
  0.5 band, no NaN. Gate role satisfied (catch broken wiring / mis-extracted cache, not
  match a prior number — no historical 32B reference exists).

### Namespace-diff gate (32B arm vs references; only model/exp_comment/output_path may differ): PASS (substantive)

Parsed line-1 `Namespace` of the 32B runs against 12850 / 12275-12276 / 13115, per
dataset+seed pair. **Every prereg-named check field matches on every pair:**
`dataset`, `topk=20`, `epochs=30`, `lambda_seg=0.0`, `archive_feats=None` — all identical.
So do all other shared training knobs (lr=1e-4, batch=64, proj/map=1024,
dropout=[0.2,0.4,0.1], loss=triplet, hybrid_loss=True, warmup=5, hard-neg=1, pseudo-gold=1,
fusion=align, metric=cos, group_name).

The three allowed fields (`model`, `exp_comment`, `output_path`) differ as expected.

- **vs same-code-era ZH references (13115):** clean — *only* model/exp_comment/output_path
  differ. Nothing else.
- **vs older references (12850, Jul 11; arcbase 12275/12276, Jul 4):** the 32B `Namespace`
  additionally *contains* argparse fields that are **absent** from those older logs
  (12850: `tarc_*`, `oracle_probe`; arcbase: also `aux_*`, `cf_*`, `mm_*`, `lambda_aux`).
  These are **new-code argparse additions the older runs predate**, and every one is present
  in the 32B run at its **inert/OFF default** (`lambda_tarc=0.0`, `tarc_target_source='off'`,
  `oracle_probe=False`, `lambda_aux=0.0`, `cf_negs=False`, …). No field the reference logs
  carried was dropped (32B is a strict superset), and no *shared* field diverges.

Per kill-rule 2's "**substantive** Namespace field" qualifier, these forward-compatible
inert defaults are **not** substantive divergences — they change no training behaviour, and
the clean 13115 match proves the 32B config is byte-identical to a current-code reference
modulo the three allowed fields. **Gate: PASS.** (This is the same newer-code/inert-flag
situation the parent enc3seed audit accepted.)

### G-dims (from execution record, cross-checked at source): consistent

The trainlogs independently confirm dim 5120 on all 9 runs (head Linear in_features=5120).
Row-count / paired-id / label audit was done at extraction (execution record §Stage-E:
all 9 caches 5120-d, rows 744/107/215, 549/80/161, 579/78/149, ids+labels == 7B arm,
7B-cache mtimes untouched). No contradiction found in the trainlogs.

---

## Task 3 — DECISION RULE (full paired delta tables)

Δ = (32B − reference), per seed, both protocols, acc AND macro-F1.
PASS iff meanΔacc ≥ +0.030 **AND** meanΔF1 ≥ +0.030 **AND** 3/3 seeds both-positive.

### PRIMARY: 32B − CLIP (the goal comparison)

| dataset | protocol | s0 (ΔF1/Δacc) | s1 | s2 | meanΔF1 | meanΔacc | sign (both-pos) | verdict |
|---|---|---|---|---|---|---|---|---|
| HateMM | final   | +0.0200/+0.0093 | +0.0816/+0.0651 | +0.0313/+0.0232 | **+0.0443** | **+0.0325** | 3/3 | **PASS** |
| HateMM | val-sel | +0.0552/+0.0512 | +0.0389/+0.0326 | +0.0627/+0.0558 | **+0.0523** | **+0.0465** | 3/3 | **PASS** |
| MHC-EN | final   | −0.0471/−0.0124 | −0.0089/−0.0186 | −0.0363/−0.0248 | −0.0308 | −0.0186 | 0/3 | **FAIL** |
| MHC-EN | val-sel | −0.1448/−0.0745 | +0.0938/+0.0249 | −0.0379/−0.0311 | −0.0296 | −0.0269 | 1/3 | **FAIL** |
| MHC-ZH | final   | −0.0461/−0.0269 | −0.0025/−0.0135 | −0.0617/−0.0671 | −0.0368 | −0.0358 | 0/3 | **FAIL** |
| MHC-ZH | val-sel | −0.0690/−0.0470 | −0.0358/−0.0336 | −0.0736/−0.0336 | −0.0595 | −0.0381 | 0/3 | **FAIL** |

### SECONDARY: 32B − 7B (pure scale increment)

| dataset | protocol | s0 (ΔF1/Δacc) | s1 | s2 | meanΔF1 | meanΔacc | sign (both-pos) | verdict |
|---|---|---|---|---|---|---|---|---|
| HateMM | final   | −0.0310/−0.0326 | +0.0124/+0.0093 | −0.0452/−0.0465 | −0.0213 | −0.0233 | 1/3 | **FAIL** |
| HateMM | val-sel | +0.0118/+0.0093 | −0.0034/−0.0046 | −0.0206/−0.0232 | −0.0041 | −0.0062 | 1/3 | **FAIL** |
| MHC-EN | final   | −0.0922/−0.0496 | −0.0133/−0.0062 | −0.0535/−0.0186 | −0.0530 | −0.0248 | 0/3 | **FAIL** |
| MHC-EN | val-sel | −0.1713/−0.0807 | −0.0311/−0.0248 | −0.0379/−0.0311 | −0.0801 | −0.0455 | 0/3 | **FAIL** |
| MHC-ZH | final   | −0.0619/−0.0403 | −0.0242/−0.0135 | −0.0218/−0.0201 | −0.0360 | −0.0246 | 0/3 | **FAIL** |
| MHC-ZH | val-sel | −0.0396/−0.0335 | −0.0650/−0.0403 | −0.0753/−0.0269 | −0.0600 | −0.0336 | 0/3 | **FAIL** |

### Per-dataset verdict (fixed format)

- **HateMM (32B vs CLIP):** final-epoch: **PASS**; val-selected: **PASS**.
- **MHC-EN (32B vs CLIP):** final-epoch: **FAIL**; val-selected: **FAIL**.
- **MHC-ZH (32B vs CLIP):** final-epoch: **FAIL**; val-selected: **FAIL**.
- Scale increment (32B vs 7B): **FAIL on all three datasets, both protocols** — 32B adds
  nothing over 7B; on HateMM it is ≈7B (val-sel) to slightly below (final), and on both gap
  datasets it is clearly below 7B.

### GOAL-RELEVANT OUTCOME

The prereg pre-declares: *"Goal-relevant success = a PASS of the 32B-vs-CLIP comparison on
ANY of MHC-EN or MHC-ZH"* (that would supply the 2nd dataset the "MLLM-as-encoder helps on
≥ 2 datasets" rule needs, since HateMM is already banked). **Both MHC-EN and MHC-ZH FAIL
32B-vs-CLIP under both protocols. → GOAL-RELEVANT OUTCOME = FAIL.**

The HateMM 32B-vs-CLIP PASS is **not** a goal pass — per the prereg it merely restates what
the 7B swap already banked; it adds no second goal dataset. **B2 closes the encoder-SCALE
axis as the campaign's 21st negative.** HateMM remains the sole encoder-passing dataset.

---

## Task 4 — CONTEXT (honest characterization of the scale effect)

**Does 32B beat/equal/lose to 7B on HateMM (the anchor)?** It **loses slightly / ties** —
scale did **not** extend the working lever. HateMM final-epoch absolute acc means:
CLIP **0.8124** → 32B **0.8450** → 7B **0.8682**; final-epoch F1: CLIP 0.7936 → 32B 0.8379
→ 7B 0.8591. 32B sits **between CLIP and 7B**: it clears CLIP (hence the primary PASS) but
regresses from 7B (secondary meanΔacc −0.0233 final, −0.0062 val-sel; 1/3 sign both
protocols). The 4× parameter increase bought **no gain over 7B** on the one dataset where
the encoder lever works — the banked positive is a 7B-vs-CLIP effect that 32B reproduces at
a diminished margin, not a scale-monotone trend.

**Are the MHC-EN / MHC-ZH results merely flat or actively below CLIP?** **Actively below**,
not flat. Magnitudes (meanΔ vs CLIP): MHC-EN final −0.0186 acc / −0.0308 F1, val-sel −0.0269
acc / −0.0296 F1; MHC-ZH final −0.0358 acc / −0.0368 F1, val-sel −0.0381 acc / −0.0595 F1.
MHC-ZH is below CLIP by roughly the same 0.03–0.06 magnitude that would count as a *pass* in
the other direction — a clear negative, not noise around zero. The ZH ROC story confirms the
B1 "unconverted ranking signal" diagnosis and shows scale did **not** rescue it: final-epoch
ROC means CLIP **0.8389** < 32B **0.8574** < 7B **0.8888** — 32B carries a *smaller*
unconverted ROC edge over CLIP than 7B did, and it converts to a thresholded acc (**0.7785**)
that is actually **below** CLIP's (0.8143). More frozen capacity widened nothing; if
anything the ZH ranking signal is weaker at 32B than at 7B.

**Anomaly — very early val-selection epochs on the MHC arms.** The 32B val-sel picks land at
MHC-EN e14/e28/e13 and MHC-ZH e15/e23/**e6**. The e6/e13/e14/e15 selections indicate the val
accuracy peaked very early (on the 80-row EN / 78-row ZH dev sets) and the model then overfit
— exactly the val-selection-tax regime the prereg flagged. The most extreme case, MHC-EN
val-sel s0, selects e14 with Test F1 **0.5665** (near-degenerate macro-F1), producing the
outlier −0.1448 vs-CLIP / −0.1713 vs-7B deltas that inflate the val-sel FAIL magnitude. This
is a noise observation consistent with the documented ~2-acc-pt 78-dev selection tax — it
does **not** rescue or overturn the verdict: under the **cleaner final-epoch lens** (which
the prereg names the less-noisy protocol for the small-dev datasets), 32B is **still below
CLIP** on both gap datasets, and both protocols FAIL independently. The rule is applied as
pre-declared; the early-epoch selection merely explains why the val-sel deltas are noisier
than the final-epoch ones, not why the datasets fail.

---

## Summary verdict

| item | result |
|---|---|
| Provenance (18 32B + 24 reference readings vs primary logs) | **PASS** — all match to 4 dp |
| G-repro (HateMM-s0 sanity, 5120-d wiring, non-degenerate) | **PASS** |
| Namespace-diff gate (only model/exp_comment/output_path substantive) | **PASS** |
| G-dims (dim 5120, row counts, paired ids) | consistent, no contradiction |
| HateMM 32B-vs-CLIP | **PASS** both protocols (restates banked 7B win; not a goal pass) |
| MHC-EN 32B-vs-CLIP | **FAIL** both protocols (actively below CLIP) |
| MHC-ZH 32B-vs-CLIP | **FAIL** both protocols (actively below CLIP) |
| 32B-vs-7B (scale increment) | **FAIL** all datasets/protocols (32B ≤ 7B) |
| **GOAL-RELEVANT OUTCOME** | **FAIL** — no MHC-EN/ZH pass vs CLIP |

**B2 closes the encoder-SCALE axis as the 21st campaign negative.** Frozen 32B extends the
HateMM encoder win over CLIP but regresses from 7B there, and is actively below CLIP on both
goal-gap datasets. Scale is not the missing lever. The encoder story remains: HateMM is the
sole dataset where the frozen MLLM-encoder swap passes.
