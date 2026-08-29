# C3 G0-cond ORACLE conditional-information probe — HateMM train target-semantic ceiling

> **VERDICT OVERTURNED 2026-07-14** by independent review (refine-logs/C3_PROBE_VERDICT_REVIEW.md): shared-L2 regularization crushed auxiliary columns (label-oracle calibration failed at +0.047/+0.014 vs required ~full headroom). Corrected target-oracle ceiling = +0.0487 both encoders (> +0.040 bar) — target content NOT foreclosed at ceiling; real-predictor bet remains marginal. Numbers below are retained for the record but are machinery artifacts.

Date: 2026-07-14
Author = **Claude Opus 4.8** (`claude-opus-4-8`, 1M ctx), fresh **C3 oracle-ceiling probe executor**.
Distinct from the A-line G0-cond executor (`M1_G0COND_PROBE_RECORD.md`) whose machinery this
reuses. Zero GPU, zero SLURM, CPU-only, conda `HateVideo`, login node, runtime ~1 min.

**Scope / compliance.** Read-only over the frozen HateMM feature `.pt` caches + gold target map
+ gold train labels. **Gold target and gold label are used ONLY as probe features/targets** — a
probe-only oracle, never in-method (task-permitted: "gold is probe-only, never in-method"). This
probe bounds the target/community-semantic content that a *future* C3 MLLM dense-text channel
could carry, **before** any GPU is spent on it. No validation/test content or label read (train
split only). Write scope = this file + `scripts/analysis/c3_g0cond_oracle_probe.py` +
`refine-logs/C3_G0COND_ORACLE_PROBE_OUT.json`. Not committed (archiver handles commits).

Executes the mandatory zero-cost **G0-cond oracle kill-switch** for C3
(`research-wiki/LITERATURE_mllm_integration_2026-07-13.md` §1 C3, §3, §4(b)) which requires the
target-content ceiling to clear the decision bar before C3 earns GPU.

---

## 0. VERDICT

**`TARGET_CONTENT_CAPPED`.** A *perfect* target-community channel — gold primary target as a
9-way one-hot (8 named targets + a "none" column) at **coverage 1.0** — carries **genuine but
small** conditional information over the frozen encoder features, and its projected accuracy gain
is **far below the +0.040 bar on both encoders**:

| encoder | oracle-target Δacc [95% CI] | Δbits/vid [95% CI] (excl. 0?) | Fano Δacc proj | vs bar +0.040 |
|---|---|---|---|---|
| CLIP (Z-only 0.8239) | **+0.0145 [+0.0091, +0.0204]** | +0.02344 [+0.01987, +0.02700] ✔ >0 | +0.0090 | **FAIL** (CI upper +0.0204) |
| Qwen (Z-only 0.8384) | **+0.0035 [+0.0013, +0.0059]** | +0.01143 [+0.00923, +0.01366] ✔ >0 | +0.0041 | **FAIL** (CI upper +0.0059) |

The most generous single projection anywhere (CLIP direct-probe Δacc CI upper bound = **+0.0204**)
does not even reach +0.030, let alone +0.040. Unlike the A-line certificate (noise-quality: Δbits
CI straddled/was negative), the target oracle IS real — Δbits CI excludes 0 in both cells — but it
is **magnitude-capped, not coverage- or noise-capped**: gold target has full coverage here, yet
knowing the target community perfectly moves a capacity-matched linear head by only ~1.5 acc pts
(CLIP) / ~0.4 acc pts (Qwen). Target/community reasoning content **cannot** carry C3 to the bar.
This is consistent with TARC's test-flat negative and is now established at the oracle ceiling, so
no predictor-quality improvement can rescue a target-centric C3 prompt. **Do not spend GPU on a
target-community-semantics C3 channel.** C3, if pursued, must pivot its prompt design to
**non-target content** (coded language / dog-whistles / symbols / scene reasoning) whose ceiling
this probe does NOT measure — see §5.

---

## 1. Data facts (this session)

Dataset: **HateMM train = 744 videos** (the only dataset with GOLD target annotations;
`data/gt/HateMM/target_map.json` was built during TARC from `HateMM_annotation.csv`. MHC/MHC_zh have
no gold target — their `train.jsonl` fields are only `{id,text,label}` and their sole target file
`data/gt/MHC/target_pred_qwen7b.json*` is a *Qwen-predicted* proxy, not gold; verified this session).
Both cached frozen representations probed (same caches the A-line probe family uses,
`data/CLIP_Embedding/HateMM/`):

| encoder | file | Z = concat(img, text) dim | Z-only CV acc |
|---|---|---|---|
| CLIP | `data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt` | 1792 (1024+768) | 0.8239 |
| Qwen | `data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct_HF.pt` | 7168 (3584+3584) | 0.8384 |

Alignment verified this session: CLIP `ids[0]` == Qwen `ids[0]` (744, same order); `.pt` `labels`
== stem class (`hate_video_*` ⇔ label 1) with **0 mismatches**; base rate pos = **0.4005**; all 744
train ids present in `target_map.json` (0 missing). Z-only CV accuracy (0.8239 / 0.8384) sits in the
known HateMM floor band, so Z is not an artificially weak comparator inflating A's apparent gain.

**Gold target coverage/format** (`target_map.json._meta`): 8 named targets
`{Blacks:0, Jews:1, Whites:2, Others:3, LGBTQ:4, Muslims:5, Sexits:6, Asian:7}`, `primary` = first
normalized target, `-1` = no target. Train-split primary distribution (n=744):

| primary | -1 none | 0 Blk | 1 Jew | 2 Wht | 3 Oth | 4 LGBTQ | 5 Mus | 6 Sex | 7 Asn |
|---|---|---|---|---|---|---|---|---|---|
| count | 32 | 332 | 72 | 18 | 270 | 6 | 7 | 5 | 2 |

Coverage of a non-none target = **0.957** (32 videos are "none", **all 32 non-hate**). Target is
annotated for both hate and non-hate videos (it is the referenced community, not the label). It is
**genuinely label-relevant but not the label**: target category strongly shifts the hate base rate
(e.g. code 3 "Others" 6 hate / 264 non-hate; code 0 "Blacks" 225 hate / 107 non-hate; every hate
video has a target, none has primary=-1). This is exactly why the oracle target *should* carry some
conditional information — and the probe measures how much.

**Why raw pre-projection Z (DPI, fail-closed):** the RGCL trainable projection is a deterministic
label-fit function of these frozen features, so by the data-processing inequality conditioning on
raw Z gives A the *maximum* benefit of the doubt — a KILL here is maximally robust (identical
rationale to `M1_G0COND_PROBE_RECORD.md` §1).

---

## 2. Probe design (adapted from the A-line G0-cond machinery; deviations documented)

Machinery reused verbatim from `refine-logs/lb_scgp_global/M1_G0COND_PROBE.py`; the new script is
`scripts/analysis/c3_g0cond_oracle_probe.py` (A-line script left unmodified).

- **Probe class** = L2-regularized logistic regression on standardized features (RGCL final linear
  classifier proxy, capacity-matched). **Same `C` for `g(Z)` and every `g'([Z,A])` arm** — the only
  difference between arms is the appended A columns → an honest marginal-information test. `C` chosen
  per-encoder by inner 5-fold stratified CV on Z-only over {1e-3,1e-2,1e-1,1}: both encoders → **C=0.001**.
- **CV** = `RepeatedStratifiedKFold(n_splits=5, n_repeats=5)` = 25 fits/model (**≥ 5 seeds**);
  `StandardScaler` fit on the train fold only (no leakage); per-video held-out mean bits and mean
  accuracy averaged over the 5 repeats.
- **Codelength (MDL, preferred over accuracy)** = held-out `-log2 p_model(y_true|x)` in bits;
  `Δbits = L(Z) − L([Z,A])` (positive ⇒ A carries conditional info). Reported as mean bits/video
  with a **example-clustered (per-video) percentile bootstrap CI** (B=5000, RNG seed 20260714) over
  the paired per-video ΔNLL (each video is one cluster).
- **Fano bits→acc projection** = for binary Y, cross-entropy(bits) ≥ H(Y|X) ≥ H_b(P_e), so
  `P_e ≥ H_b^{-1}(min(bits,1))` and the Fano accuracy ceiling = `1 − H_b^{-1}(mean_bits)`; the Fano
  projected Δacc = ceiling([Z,A]) − ceiling(Z). Reported alongside the direct probe Δacc.

**Deviations from the A-line probe (all justified):**
1. **Oracle = gold TARGET, not gold LABEL.** The A-line's `oracle@coverage-1.0` used one-hot(label)
   — the ceiling of *any* signal. Here the decision arm uses one-hot(**target**) — the ceiling of a
   *target-semantic* channel specifically. The label oracle is retained only as a **diagnostic**
   any-signal reference (§3), never as a decision arm.
2. **Coverage = 1.0**, not the certificate parse coverage (8.7% in the A-line). Gold target exists
   for all 744 videos, so there is no coverage scaling — the direct probe Δacc *is* the ceiling, and
   the A-line's `c·(1−a_cov)` analytic coverage bound does not apply.
3. **Null control arm added** (`shuffled_target`): A rows permuted with a fixed seed to break the
   video↔target alignment while preserving the target marginal — must read ≈0 or the machinery is
   manufacturing signal.
4. Single dataset (HateMM) — it is the only one with gold targets. Both encoders still probed.

Command (this session):
`python3 scripts/analysis/c3_g0cond_oracle_probe.py refine-logs/C3_G0COND_ORACLE_PROBE_OUT.json`
(conda `HateVideo`; script + full results JSON persisted alongside this record).

---

## 3. RESULTS — arms × both encoders

`accZ`/`accZA` = Z-only vs [Z,A] CV accuracy; `Δacc` = accZA−accZ with 95% per-video bootstrap CI;
`Δbits/vid` = mean per-video codelength saved by A (bits) with 95% CI; `Fano Δacc` = Fano
accuracy-ceiling gain. **Positive Δbits ⇒ A helps.** Decision arm = **oracle_target**.

### 3.1 CLIP (Z-only 0.8239, C=0.001)

| arm | A dim | accZ | accZA | Δacc [95% CI] | Δbits/vid [95% CI] | Δbits_tot | Fano Δacc |
|---|---|---|---|---|---|---|---|
| **oracle_target** (9-way, cov 1.0) | 9 | 0.8239 | 0.8384 | **+0.0145 [+0.0091, +0.0204]** | **+0.02344 [+0.01987, +0.02700]** | +17.4 | +0.0090 |
| shuffled_target (null) | 9 | 0.8239 | 0.8247 | +0.0008 [−0.0003, +0.0022] | −0.00017 [−0.00070, +0.00040] | −0.1 | −0.0001 |
| label_oracle (DIAGNOSTIC) | 2 | 0.8239 | 0.8712 | +0.0473 [+0.0363, +0.0591] | +0.09403 [+0.08567, +0.10272] | +70.0 | +0.0339 |

### 3.2 Qwen (Z-only 0.8384, C=0.001)

| arm | A dim | accZ | accZA | Δacc [95% CI] | Δbits/vid [95% CI] | Δbits_tot | Fano Δacc |
|---|---|---|---|---|---|---|---|
| **oracle_target** (9-way, cov 1.0) | 9 | 0.8384 | 0.8419 | **+0.0035 [+0.0013, +0.0059]** | **+0.01143 [+0.00923, +0.01366]** | +8.5 | +0.0041 |
| shuffled_target (null) | 9 | 0.8384 | 0.8379 | −0.0005 [−0.0019, +0.0005] | +0.00006 [−0.00042, +0.00059] | +0.0 | +0.0000 |
| label_oracle (DIAGNOSTIC) | 2 | 0.8384 | 0.8524 | +0.0140 [+0.0097, +0.0188] | +0.05544 [+0.04895, +0.06231] | +41.2 | +0.0190 |

### 3.3 Reading the arms

- **Null control passes**: shuffled_target Δacc ≈ 0 and Δbits CI includes 0 in both cells → the
  machinery is not manufacturing signal; the oracle_target gain is genuinely from video↔target
  alignment.
- **Signal is real but small**: oracle_target Δbits CI **excludes 0** (positive) in both cells — a
  perfect target channel *does* carry conditional information (unlike the A-line certificate, which
  was noise-quality even where it parsed). But the realized accuracy gain is +0.0145 (CLIP) /
  +0.0035 (Qwen), Fano +0.0090 / +0.0041 — an order of magnitude under the bar.
- **Diagnostic label_oracle proves the probe is sensitive**: injecting the gold *label* clears the
  bar on CLIP (+0.0473) — so the machinery CAN detect a +0.040 signal when one exists; the target
  oracle simply is not one. On Qwen the label oracle is only +0.0140, because Qwen's frozen features
  already encode most of the label (Z-only 0.8384), leaving little linear headroom for *any* injected
  feature — the target oracle (+0.0035) is ~25% of even that small label ceiling. Both readings point
  the same way: at the HateMM operating point, target identity is a minor residual over frozen Z.

---

## 4. DECISION RULE EVALUATION (pre-declared, fail-closed)

> C3 target-content channel is a LIVE carrier only if the oracle-target conditional gain projects
> **≥ +0.030 + 0.010 = +0.040** test Δacc. Projection < +0.040 ⇒ `TARGET_CONTENT_CAPPED`.
> (Bar = `REFLECTION_mllm_integration_failures.md` §4 / `LITERATURE_mllm_integration_2026-07-13.md` §3.)

**Oracle-target projected Δacc, every estimator, both encoders:**

| estimator | CLIP | Qwen | ≥ +0.040? |
|---|---|---|---|
| direct probe Δacc (mean) | +0.0145 | +0.0035 | No |
| direct probe Δacc (CI upper) | +0.0204 | +0.0059 | No |
| Fano bits→acc projection | +0.0090 | +0.0041 | No |

The single most generous number anywhere is the CLIP direct-probe Δacc 95%-CI upper bound
**+0.0204** — still below +0.030 and roughly **half** the +0.040 bar. The train-CV logistic probe
is itself a *generous* upper bound on any test-time gain: the deployed channel would be a top-20
kNN vote, strictly weaker than a direct linear head (`M1_G0COND_PROBE_RECORD.md` §3.3), and the
project's C2 rule additionally demands +0.030 on multiple metrics × seeds × Holm. **The oracle
kill-switch fires: `TARGET_CONTENT_CAPPED`.**

**Adversarial check, both directions:**
- *Not over-killing:* I gave the target channel every advantage — raw pre-projection Z (DPI), full
  coverage 1.0, a dedicated "none" column, all 8 named targets, same C, and a sensitive probe that
  demonstrably detects the +0.040 label signal on CLIP. It still lands < half the bar. The null
  control confirms the small gain that exists is real, not an over-kill of a genuine signal.
- *Not wishfully proceeding:* the signal is non-zero (Δbits CI excludes 0), so this is an honest
  "real-but-too-small" verdict, not a "no-signal" one — which makes it *more* robust: the cap is on
  magnitude, and no channel (measured or oracle) can exceed the oracle.

---

## 5. Scope-honesty paragraph (read this before designing C3)

**This probe bounds ONLY the target/community-semantic content of a future C3 text channel — nothing
else.** The `TARGET_CONTENT_CAPPED` verdict means: even a perfect, full-coverage rendering of *which
protected community a video references* — the dominant hypothesized payload of a C3 world-knowledge
prompt — cannot carry C3 to the +0.040 bar (≤ +0.0204 CLIP / ≤ +0.0059 Qwen, both CI upper bounds),
because the frozen encoder features already absorb most of what target identity provides and the
residual is ~1.5 / 0.4 acc pts. This is **consistent with TARC's test-flat negative** and now
strengthens it: TARC showed a *measured* target route is flat; this shows the *oracle ceiling* of
target content is flat, so no improvement in target-predictor quality can rescue a target-centric
C3. **Therefore a C3 prompt must NOT center on target/community reasoning.** If C3 is pursued, its
prompt design must pivot to **non-target content — coded language / dog-whistles, hateful symbols
and their scene context, implicit-meaning / scene reasoning** — whose ceiling **this probe does not
measure** (no gold annotations exist for those channels). Before any GPU, C3-on-non-target-content
needs **either a different oracle** (a gold or high-quality proxy annotation for the coded/symbolic
content, if constructible) **or a small measured pilot** subject to the same G0-cond gate. Do not
read this `CAPPED` verdict as killing C3 wholesale — it kills the *target-semantic hypothesis* for
C3 and redirects the prompt-design search, exactly as the C3 spec's "gate-first, else pivot"
protocol intends.

---

## 6. Numbers → source index (provenance)

- Feature caches, dims, id/label alignment, base rate (§1): `torch.load` over the two
  `data/CLIP_Embedding/HateMM/train_*.pt`; `ids[0]` lists (744, equal across encoders); `labels` vs
  stem class (0 mismatches, this session).
- Gold target coverage/format, primary distribution, none-all-nonhate (§1): `data/gt/HateMM/target_map.json`
  (`_meta.code_dict`, `_meta.primary_rule`; per-video `primary`); reproduced in
  `refine-logs/C3_G0COND_ORACLE_PROBE_OUT.json.target_facts`. Map provenance = TARC exp-tarc-t0 G0
  (`_meta._note`), source CSV `data/gt/HateMM/HateMM_annotation.csv`.
- All arm results (§3): `scripts/analysis/c3_g0cond_oracle_probe.py` →
  `refine-logs/C3_G0COND_ORACLE_PROBE_OUT.json` (RepeatedStratifiedKFold 5×5, L2 logistic C=0.001,
  StandardScaler per fold, example-clustered bootstrap B=5000 seed 20260714, Fano ceiling via H_b^{-1}).
- Machinery lineage (§2): `refine-logs/lb_scgp_global/M1_G0COND_PROBE.py` +
  `M1_G0COND_PROBE_RECORD.md` (design: capacity-matched probes, same-C marginal test, MDL codelength,
  per-video bootstrap, DPI raw-Z rationale, kNN-weaker-than-logistic argument).
- Decision bar / C3 gate (§0, §4, §5): `research-wiki/LITERATURE_mllm_integration_2026-07-13.md`
  §1 C3 (gate-first target vs non-target), §3 (G0-cond recipe incl. oracle kill-switch), §4(b)
  (C3 QUALIFIED SUPPORT, gate-first-or-die); +0.040 bar and ±1–2pt noise band from
  `research-wiki/REFLECTION_mllm_integration_failures.md` §4. TARC test-flat negative:
  MEMORY goal-round2 (TARC = 14th pre-registered negative, target-aware route test-flat both datasets).

## Required statements
- No performance/accuracy claim on any held-out benchmark; all accuracy/codelength numbers are
  train-only cross-validation, used solely to measure conditional information.
- Gold read = `primary` target (`target_map.json`) as appended probe features + `parent_video_binary_label`
  (train stems) as probe targets; both probe-only, never in-method. No validation/test content or
  label opened. No GPU, no SLURM, no MLLM/OCR/network. Write scope = this file +
  `scripts/analysis/c3_g0cond_oracle_probe.py` + `refine-logs/C3_G0COND_ORACLE_PROBE_OUT.json`.
  Not committed (archiver handles commits).
