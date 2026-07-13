# B1 Verdict Review — frozen-Qwen vs frozen-CLIP encoder swap on MHC-ZH (job 13115)

**Reviewer:** fresh independent verdict reviewer (zero prior context; read-only, CPU
verification; no GPU / SLURM / commits). **Date:** 2026-07-14.
**Under review:** experiment B1 (`research-wiki/experiments/exp-encoder-zh-b1.md`,
pre-registration; `refine-logs/B1_EXECUTION_RECORD.md`, raw transcription; SLURM job
13115, COMPLETED / 0:0, elapsed 00:11:21).
**Method:** every number below was re-parsed independently from the six primary raw
trainlogs (`slurm/logs/enc3s_MHC_zh_*_13115.trainlog`) with a fresh parser that splits on
`\r` and `\n`, then applied the pre-registered selection rule (val-sel = epoch ≥ warmup 5
maximizing Val_Retrieval acc, roc tie-break; final = epoch 29) from scratch. The gate
references (old-code Qwen s0 = `rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_1151518.trainlog`;
CLIP floor job 12130 = `exp-consensus-zh-seeds.md:56-64`) were parsed / read directly.

---

## Task 1 — Numeric provenance (independent re-read of all 6 trainlogs)

Namespace audit (raw line 1 of each log): both arms identical except `model=` /
`exp_comment=` / `output_path=`. Common fields: `dataset='MHC_zh'`, `topk=20`,
`epochs=30`, `lambda_seg=0.0`, `archive_feats=None`, `warmup=5`, `force=false`,
`group_name='RAC_video_archive_seeds'`. Dims: CLIP 1024 img / 768 text; Qwen 3584 / 3584.
**Namespace-diff gate (kill rule 2): PASS** — differs only in the manipulated variable.

Independently re-derived Test readings (macroF1 / acc / roc) vs the execution record:

| arm | seed | protocol | selEp | my re-parse (F1 / acc / roc) | exec-record | match |
|---|---|---|---|---|---|---|
| CLIP | 0 | val-sel | 29 | 0.7706 / 0.8054 / 0.8382 | 0.7706 / 0.8054 / 0.8382 | ✅ |
| CLIP | 0 | final | 29 | 0.7706 / 0.8054 / 0.8382 | 0.7706 / 0.8054 / 0.8382 | ✅ |
| CLIP | 1 | val-sel | 28 | 0.7579 / 0.8054 / 0.8346 | 0.7579 / 0.8054 / 0.8346 | ✅ |
| CLIP | 1 | final | 29 | 0.7542 / 0.8054 / 0.8342 | 0.7542 / 0.8054 / 0.8342 | ✅ |
| CLIP | 2 | val-sel | 25 | 0.7742 / 0.8121 / 0.8419 | 0.7742 / 0.8121 / 0.8419 | ✅ |
| CLIP | 2 | final | 29 | 0.7913 / 0.8322 / 0.8444 | 0.7913 / 0.8322 / 0.8444 | ✅ |
| Qwen | 0 | val-sel | 22 | 0.7412 / 0.7919 / 0.8838 | 0.7412 / 0.7919 / 0.8838 | ✅ |
| Qwen | 0 | final | 29 | 0.7864 / 0.8188 / 0.8906 | 0.7864 / 0.8188 / 0.8906 | ✅ |
| Qwen | 1 | val-sel | 25 | 0.7871 / 0.8121 / 0.8874 | 0.7871 / 0.8121 / 0.8874 | ✅ |
| Qwen | 1 | final | 29 | 0.7759 / 0.8054 / 0.8951 | 0.7759 / 0.8054 / 0.8951 | ✅ |
| Qwen | 2 | val-sel | 28 | 0.7759 / 0.8054 / 0.8940 | 0.7759 / 0.8054 / 0.8940 | ✅ |
| Qwen | 2 | final | 29 | 0.7514 / 0.7852 / 0.8806 | 0.7514 / 0.7852 / 0.8806 | ✅ |

**All 12 readings (6 arms × 2 protocols) match to 4 decimals on F1, acc AND roc.**
Selection epochs and roc tie-breaks independently reproduced (e.g. Qwen s0 val-acc 0.8205
tied at e22/e26/e28 → roc tie-break selects e22 roc 0.8693; CLIP s1 val-acc 0.7821 tied at
e18/e27/e28 → e28 roc 0.8836). The old-code "epoch-0 caveat" (raw val-acc peak 0.8333 at
e0) is correctly excluded by warmup ≥ 5.

**One cosmetic, non-value discrepancy (NOT a flag).** The execution record's raw-line
citations (e.g. CLIP s0 ":305", Qwen s2 ":299") exceed the files' actual newline counts
(279, 272). Cause: the record numbers tqdm `\r`-delimited segments as lines while `grep -n`
counts `\n` only. Every cited *value* is exact (spot-checked the raw
`Test_Retrieval Epoch NN macroF1:` lines directly). **Provenance verdict: PASS, zero value
mismatches.**

---

## Task 2 — GATE 1a (HARD): new-code frozen-Qwen s0 vs old-code 1151518

Old-code reference `rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_1151518.trainlog` re-parsed
independently (dataset='MHC_zh', model=Qwen, seed=0, dims 3584/3584):

| protocol | old-code 1151518 (F1 / acc / roc) | new-code s0 13115 (F1 / acc / roc) | 4-dp match |
|---|---|---|---|
| val-sel (e22) | 0.7412 / 0.7919 / 0.8838 | 0.7412 / 0.7919 / 0.8838 | ✅ |
| final (e29) | 0.7864 / 0.8188 / 0.8906 | 0.7864 / 0.8188 / 0.8906 | ✅ |

Both protocols reproduce to 4 printed decimals on F1, acc AND roc; same selection epoch
(e22) and same tie-break structure. **GATE 1a: PASS** — the old-code→new-code confound is
retired for the ZH cell (every `run_rac.py` flag added since the old run is inert at
defaults, confirmed bit-for-bit). **No HALT.**

## Task 3 — GATE 1b (confirmatory cross-runner): new-code frozen-CLIP s0 vs job 12130

Reference (`exp-consensus-zh-seeds.md:60`, job 12130, `train_consensus_seeds` runner):
val-sel = final = 0.7706 F1 / 0.8054 acc at e29.
New-code CLIP s0 (13115, `train_archive_baseline` archive-OFF): val-sel = final = 0.7706 /
0.8054 at e29. **Exact match — divergence 0.0000 ≪ 0.0001 threshold; no audit triggered.**

Stronger than required: B1's fresh CLIP arm reproduces the **entire 3-seed consensus-zh
floor band** (`exp-consensus-zh-seeds.md:60-62`), not just s0 —
s0 0.7706/0.8054·0.7706/0.8054, s1 0.7579/0.8054·0.7542/0.8054, s2 0.7742/0.8121·0.7913/0.8322
(sel·final) all identical across the two different runners. The archive-OFF and seg-λ0
code paths are therefore empirically equivalent on this cell. **GATE 1b: PASS
(confirmatory).**

---

## Task 4 — Decision rule (verbatim parent rule, both protocols)

Rule (transcribed in prereg from `exp-encoder-3seed.md:73-85`): per dataset × protocol,
**PASS iff mean paired Δacc ≥ +0.030 AND mean paired ΔmF1 ≥ +0.030 AND 3/3 seeds sign
positive.** Paired Δ = Qwen − CLIP within seed. Paired-t reported as effect-size descriptor
only (n = 3, no significance claim).

### Protocol (B) final-epoch — the pre-declared reporting-emphasis protocol

| seed | Δacc | ΔmF1 |
|---|---|---|
| 0 | +0.0134 | +0.0158 |
| 1 | +0.0000 | +0.0217 |
| 2 | **−0.0470** | **−0.0399** |
| **mean ± std** | **−0.0112 ± 0.0317** (t = −0.61, **1/3** positive) | **−0.0008 ± 0.0340** (t = −0.04, **2/3** positive) |

→ mean Δacc ≥ +0.030 ✗ · mean ΔmF1 ≥ +0.030 ✗ · sign 3/3 ✗ (acc 1/3, F1 2/3) — **FAIL**

### Protocol (A) val-selected

| seed | Δacc | ΔmF1 |
|---|---|---|
| 0 | −0.0135 | −0.0294 |
| 1 | +0.0067 | +0.0292 |
| 2 | −0.0067 | +0.0017 |
| **mean ± std** | **−0.0045 ± 0.0103** (t = −0.76, **1/3** positive) | **+0.0005 ± 0.0293** (t = +0.03, **2/3** positive) |

→ mean Δacc ≥ +0.030 ✗ · mean ΔmF1 ≥ +0.030 ✗ · sign 3/3 ✗ (acc 1/3, F1 2/3) — **FAIL**

Both protocols fail on **every** clause of the criterion. The means are not merely below
the +0.030 bar — they are **flat-to-negative** on both metrics under both protocols. No
single metric moves in a consistent positive direction (no arm reaches 3/3), so this is a
clean FAIL, not even "FAIL-with-direction" (kill rule 4). Metric-shopping (kill rule 4) and
protocol-shopping (kill rule 3) are moot: neither metric nor protocol passes.

### Prereg outcome-category adjudication (PASS / PARTIAL / FAIL, prereg's own definitions)

- **PASS** (decision rule 4): criterion met under a protocol → **not met** (both FAIL).
- **PARTIAL** (prereg "Honest prior", lines 96-98 / 281-285): the predicted partial =
  *"final-epoch weakly positive … val-selected flat/negative"* — i.e. final-epoch
  directionally positive on both metrics below the bar. **Not met:** the final-epoch mean is
  **negative** on acc (−0.0112) and F1 (−0.0008); it is not "weakly positive." The observed
  result is **weaker than the honest prior expected**.
- **FAIL** (parent's per-dataset category, as applied to MHC-EN "FAIL under both
  protocols"): criterion unmet under both protocols, means flat-to-negative. **This is the
  category satisfied.**

**VERDICT: FAIL under both protocols.** Written in the fixed parent format:
**"final-epoch: fail; val-selected: fail."** The frozen-CLIP → frozen-Qwen encoder swap
does **not** help on MHC-ZH; on the reporting-emphasis final-epoch protocol it trends
slightly *negative*. The prereg's rule (5) "≥ 2 datasets" second-dataset slot is **not**
supplied; HateMM remains the sole formally passing encoder dataset.

---

## Task 5 — Context (no new claims)

**HateMM stays the only encoder-swap PASS.** Across all three tested dataset cells the
frozen-Qwen-for-frozen-CLIP swap now reads: HateMM **PASS** both protocols (+5.3–5.6 acc /
+5.6–6.6 F1, 3/3 seeds; `exp-encoder-3seed.md`), MHC-EN **FAIL** both, MHC-ZH **FAIL** both.
The parent's ">= 2 datasets" headline is still **unmet**; the banked positive remains
HateMM-specific. B1 closes the missing fourth quadrant of the encoder test as a negative.

**The P8c language-match hypothesis is now tested on ZH and answered: not supported for a
frozen encoder.** H1's lead mechanism (Qwen is Chinese-strong; CLIP's English-centric text
tower truncates ~97% of Chinese byte-fragments, so a native-Chinese encoder should recover
ZH where it can't recover EN) predicted ZH might pass where MHC-EN failed. It did not — a
frozen native-Chinese Qwen encoder is **flat-to-slightly-worse** than frozen CLIP on the
RGCL head (final-epoch mean −0.011 acc / −0.001 F1). This confirms the prereg's own honest
prior and `MEMORY.md`: the ZH gain that reached 0.8537 came from **LoRA-SFT of the encoder**
(a different lever), **not** from frozen Qwen hidden states. The language-match truncation
is a real phenomenon, but whatever benefit it confers requires LoRA adaptation to surface;
it does not translate into a frozen-feature accuracy gain. (Observed but verdict-neutral:
the Qwen arm does carry consistently higher Test roc/AUC — 0.88–0.90 vs CLIP 0.83–0.84 on
every seed — i.e. better ranking that never converts to better thresholded acc/F1; the rule
is on acc AND F1, so this does not move the verdict.)

**The s2 final-epoch reversal vs the ZH noise band.** Seed 2 drives the final-epoch mean
negative: Qwen s2 final e29 = 0.7514 F1 / 0.7852 acc vs CLIP 0.7913 / 0.8322 → **−0.0470
acc / −0.0399 F1**. That magnitude is **~2–3× the documented ±1–2 pt ZH noise band** (78-dev
val-selection tax ~2 acc pts, `PAPER_MASTER_TABLES.md:56-57`; 5-seed CLIP-floor final-epoch
std 0.0215 acc / 0.0240 F1, prereg lines 213-214), so it is a genuine reversal, not
attributable to noise alone — a real late-epoch degradation of the Qwen s2 run (its own
val-sel e28 → final e29 fell 0.7759 → 0.7514 F1). But the verdict does not hinge on s2:
dropping it, the s0+s1 final-epoch mean is only +0.0067 acc / +0.0188 F1 — still far below
+0.030 and not 3/3. The FAIL is robust to the single reversing seed.

---

## Summary

| item | result |
|---|---|
| Numeric provenance (12 readings) | **PASS** — all exact to 4 dp; only cosmetic `\r`-vs-`\n` line-number offset, no value error |
| Namespace-diff gate (kill rule 2) | **PASS** — differs only in `model=` |
| GATE 1a (hard, Qwen s0 vs 1151518) | **PASS** — 4-dp both protocols; no HALT |
| GATE 1b (confirmatory, CLIP s0 vs 12130) | **PASS** — exact; whole 3-seed floor band reproduced; no audit |
| Decision rule, final-epoch | **FAIL** — mean −0.0112 acc / −0.0008 F1; sign 1/3 acc, 2/3 F1 |
| Decision rule, val-selected | **FAIL** — mean −0.0045 acc / +0.0005 F1; sign 1/3 acc, 2/3 F1 |
| **VERDICT (prereg categories)** | **FAIL under both protocols** ("final-epoch: fail; val-selected: fail"); weaker than the prereg's predicted `partial` |
| Headline "≥ 2 datasets" | **still NOT met** — HateMM remains the sole passing encoder dataset |
