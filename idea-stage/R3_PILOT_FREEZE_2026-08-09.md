# Round-3 pilot freeze — 2026-08-09

**Status: FROZEN before any implementation line was written.** Rules below were produced by the
cross-model jury (`gpt-5.6-sol`, xhigh, thread `019fe558-4208-79d1-8e25-159c819a2f68`) in the triage
round, after the objective feasibility gate and the arXiv novelty probe were reported to it. Nothing
below may be edited after any result is seen. Any deviation is recorded as a numbered deviation note
appended to this file, never as an edit.

## Shared protocol (all three pilots)

- **Zero test-set contact.** A path guard HALTs on any path containing `test`. Train + validation
  splits only.
- 5 fixed seeds; 5-fold **video-level** out-of-fold estimation; all preprocessing (standardisation,
  PCA, TF-IDF vocabulary, probe fitting) is **fold-local**.
- All scores are **continuous, non-saturating** standardised `decision_function` logits.
  No bounded vote/count scores (P2 forensic rule).
- Every pilot carries a **label-permuted null**: 20 permutations per seed = 100 null replicates.
  `N95` = one-sided 95th percentile of the null statistic. Every observed statistic must clear
  **3× N95** in addition to its absolute threshold.
- Each pilot writes `<name>.json` with every quantity named below, plus SHA256 of every input file.
- Synthetic and label-permuted smoke tests are run before the single real invocation.

---

## Pilot R3-1 — C1 "Target-conditioned attack/defence algebra": algebraic double dissociation

**Data.** MHC-EN and MHC-ZH, train + valid only (801 EN / 800 ZH). Votes from
`data/gt/mhc_votes/mhc_{English,Chinese}_{train,valid}.tsv`. Features = cached MPNet transcript,
CLIP ViT-L/14-336 image+text, and CLAP audio embeddings, concatenated, standardised, then reduced to
**128 PCA dimensions inside each fold**.

**Groups** (disjoint, from raw votes):
- `H` = at least one `Hateful` vote AND no `Counter Narrative` vote
- `C` = at least one `Counter Narrative` vote AND no `Hateful` vote
- `N` = all votes `Normal`

**Procedure.**
1. Fit an `H`-vs-`N` logistic probe **without using `C`**. Its standardised OOF logit is `q`.
2. Remove the probe direction from the features (fold-local projection).
3. Match `H` and `C` by language × `Target_Victim` stratum with inverse-frequency weights, then fit an
   `H`-vs-`C` logistic probe. Its standardised OOF logit is `s`.

**Statistics.**
- `D_content = mean(q | C) − mean(q | N)`
- `G_content = |mean(q | H) − mean(q | C)|`
- `D_stance  = mean(s | H) − mean(s | C)`
- `T_obs = min(D_content, D_stance)`

**Null.** Permute `H`/`C`/`N` assignment within language × target-present strata and rerun the
complete pipeline.

**FROZEN DECISION RULE — GO iff ALL hold, else KILL:**
1. ≥ 40 target-matched `H` items and ≥ 40 target-matched `C` items remain;
2. `D_content ≥ 0.60`;
3. `G_content ≤ 0.35`;
4. `D_stance ≥ 0.60`;
5. `D_content ≥ 0.35` and `D_stance ≥ 0.35` **separately in EN and in ZH**;
6. `T_obs ≥ 3 × N95`.

**7/10 pattern** (what would have to be true for this to become a main-conference candidate):
`D_content ≥ 0.80` in both languages; `G_content ≤ 0.25`; `D_stance ≥ 0.80`; the expected direction in
≥ 80 % of sufficiently populated `Target_Victim` strata; every contrast ≥ 3× its null `N95`; stable
across all 5 seeds and not an EN/ZH pooling artifact.

---

## Pilot R3-2 — C12 "Proposition-mass firewall": formatting mass vs informative-window dilution

**Data.** HateMM train + dev_seen only. `data/OCR/HateMM/ocr_windows_K30.jsonl`
(851 videos × 30 windows; each detection = `{text, conf, bbox(4 points)}`), plus
`data/OCR/HateMM/frame_dims_train.json`.

**Procedure.** Normalise OCR strings with Unicode NFKC + lowercase + whitespace collapse. Fit a
fold-local character 3–5-gram TF-IDF logistic probe on **individual detections**, grouped by video;
the unbounded logit for detection `j` of video `i` is `e_ij`.

Construct per video:
- `r_i` — raw score: `conf × sqrt(box area)`-weighted mean of all `e_ij`
- `c_i` — conserved score: equal-mass mean over **unique normalised strings**
- `g_i` — excess format mass: `log N_i − log U_i` (`N_i` = detection count, `U_i` = unique-string count)
- `k_i` — window-concentration control: mean of the top-3 window logits minus the mean of the
  remaining non-empty window logits
- `z_i = (2 y_i − 1)(c_i − r_i)`

**Statistic 1.** `rho_obs` = one-sided partial Pearson correlation
`pcorr(z, g | k, log U, non-empty-window count, y)`.

**Statistic 2.** Apply the frozen attack suite — token repetition, single-box splitting, box reorder,
box-area scaling, cross-channel duplication — and compute, in clean raw-score standard deviations:
`A_obs = median_i [ max_a |r_i^(a) − r_i| − max_a |c_i^(a) − c_i| ]`.

**Null.** Permute video labels, refit the detection probe, recompute both statistics.

**FROZEN DECISION RULE — GO iff ALL hold, else KILL:**
1. `rho_obs ≥ 0.24`;
2. `A_obs ≥ 0.30`;
3. `rho_obs ≥ 3 × N95(rho)`;
4. `A_obs ≥ 3 × N95(A)`;
5. `corr(c, r) ≥ 0.80` (guards against a trivial information-destroying solution).

**Interpretation clause, frozen in advance:** failure of condition 1 kills the formatting-mass
explanation of the observed OCR sign flip (frozen head +0.0094 vs learned fusion −0.0246) and favours
the competing informative-window-dilution explanation.

---

## Pilot R3-3 — C10 "Cross-channel evasion transduction closure": compositional path dependence

**Data.** HateMM train + dev_seen only. Cached K=30 Whisper ASR
(`data/ASR/HateMM/train_asrK30_whisper-large-v3.jsonl`, `dev_seen_asrK30_...`) and cached OCR strings.

**Frozen transformations.**
- `L_A` — ASR leetspeak map `a/e/i/o/s/t → 4/3/1/0/5/7`
- `L_O` — OCR symbol map `a/e/i/o/s/t → @/€/!/()/$/+`
- `S` — insert periods between the characters of every second token of length ≥ 4
- `M` — move every second eligible ASR token into the OCR channel, preserving token order

**Procedure.** Fit an OOF character 3–5-gram logistic classifier on originals **plus every
single-edge transformation**. Then evaluate semantically equivalent composed paths:
`L_A→M` vs `M→L_O`; `S→M→L_O` vs `L_A→M→S`; and all length-2 and length-3 compositions.

**Statistics** (standardised logits `f`):
- `P_obs = median_i max_{p,q ∈ E_i} |f(p(x_i)) − f(q(x_i))|`, where `E_i` is the set of paths declared
  to share a semantic endpoint
- `A_obs = median_i [ max_{|p|≥2} |f(p(x_i)) − f(x_i)| − max_{|p|=1} |f(p(x_i)) − f(x_i)| ]`

**Null.** Permute labels before training the same single-edge-augmented classifier.

**FROZEN DECISION RULE — GO iff ALL hold, else KILL:**
1. `P_obs ≥ 0.35` clean-logit SD;
2. `A_obs ≥ 0.15` clean-logit SD;
3. `P_obs ≥ 3 × N95(P)`;
4. `A_obs ≥ 3 × N95(A)`;
5. the single-edge-augmented classifier retains ≥ 90 % of the original classifier's mean signed clean
   margin.

---

## Global verdict registered in advance

The jury's pre-pilot scores were: **C1 5.0/10 · C12 4.8/10 · C10 4.7/10**, with the explicit statement
that round 3 currently contains nothing above ~5/10 and that C4 (semantic response-tensor
distillation) is removed from $0 triage because its data premise (a factorial intervention lattice)
does not exist on disk. If all three pilots KILL, the honest conclusion registered here in advance is
that **round 3 produced no viable main-conference methods candidate**, and the pre-existing
evaluation-protocol fallback remains the stronger track.
