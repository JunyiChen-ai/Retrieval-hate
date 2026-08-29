# B3 Verdict Review — LoRA-Qwen encoder vs frozen-CLIP on MHC-ZH (job 13150)

**Reviewer:** independent verdict reviewer (zero prior context by design; read-only except this
file + the single §7d erratum). **Date:** 2026-07-14.
**Under review:** the fresh LoRA arm **job 13150** (group `RAC_video_b3_lora`) vs the frozen-CLIP
control **job 13115** on MHC-ZH, per the pre-registration `research-wiki/experiments/exp-lora-zh-b3.md`
and the binding rulings in `refine-logs/B3_PREREG_REVIEW.md`.
**Every number below was re-parsed by me directly from the primary trainlogs** (numeric-provenance
discipline; all 12 fresh readings = 3 seeds × 2 protocols × {acc, macroF1}, plus the CLIP control,
the arcbase anchors, and the frozen-Qwen decomposition — line-numbered provenance in §A).

---

## 0. Verdict summary (headline)

| gate / result | outcome |
|---|---|
| **G-repro (HARD GATE)** | ✅ **PASS** — fresh 13150 reproduces arcbase 12223/12224/12225 to 4 dp on all 6 readings (0 mismatch) |
| **G-namespace** | ✅ **PASS** — all differing fields within the `{model, exp_comment, group_name, output_path}` whitelist; every substantive hyperparameter identical |
| **final-epoch protocol** | **PASS (MARGINAL)** — mean Δacc **+0.0313**, mean ΔmF1 **+0.0453**, sign 3/3 both |
| **val-selected protocol** | **FAIL** — mean Δacc **+0.0246 < +0.030** (AND-rule fails on acc), mean ΔmF1 +0.0339, sign 3/3 both |

**Binding write-up format (per `B3_PREREG_REVIEW.md` §2.2, obeyed verbatim, no upgrade):**
`final-epoch: PASS (MARGINAL); val-selected: FAIL`.

The computed numbers **match the forensic preview exactly** (final-ep +0.0313 acc / +0.0453 F1;
val-sel +0.0246 acc / +0.0339 F1) — because the G-repro gate passes bit-exact, the fresh LoRA arm is
numerically identical to the arcbase anchors the preview used, so there is no preview-vs-result
divergence to reconcile.

---

## 1. G-repro — HARD GATE — **PASS**

Rule (prereg §10.1): each fresh LoRA seed-s **final-epoch (29)** Test acc AND macroF1 must reproduce
its arcbase anchor to 4 printed decimals; any >0.0001 mismatch = HALT, verdict INVALID.

| seed | fresh 13150 acc | anchor acc | fresh 13150 mF1 | anchor mF1 | match |
|---|---|---|---|---|---|
| 0 | 0.8456 | 0.8456 (12223) | 0.8181 | 0.8181 (12223) | ✅ exact |
| 1 | 0.8389 | 0.8389 (12224) | 0.8113 | 0.8113 (12224) | ✅ exact |
| 2 | 0.8523 | 0.8523 (12225) | 0.8226 | 0.8226 (12225) | ✅ exact |

All six readings reproduce to 4 dp. **G-repro PASSES; proceed.** (Evidence: fresh
`enc3s_MHC_zh_...-LoRA_HF_seed{0,1,2}_13150.trainlog:{272,273,268}` vs arcbase
`arcbase_MHC_zh_...-LoRA_HF_seed{0,1,2}_1222{3,4,5}.trainlog:{272,273,268}` — §A.)

## 2. G-namespace — **PASS**

Rule (prereg §10.2): the fresh LoRA Namespace must be identical to the 13115 CLIP Namespace **except**
`{model, exp_comment, group_name, output_path}`, and identical to the 12223-25 arcbase Namespace
**except** `{group_name, output_path}`. I re-read the raw Namespace echoes (each log line 1):

Substantive fields — **identical across all three arms** (LoRA 13150, arcbase 12223-25, CLIP 13115):
`fusion_mode='align', topk=20, metric='cos', loss='triplet', hybrid_loss=True, proj_dim=1024,
map_dim=1024, epochs=30, batch_size=64, lr=0.0001, warmup=5, lambda_seg=0.0, archive_feats=None,
force=False, consensus_topk=10`.

Differing fields, all within the whitelist:
- **vs arcbase 12223-25:** only `group_name` (`RAC_video_b3_lora` vs `RAC_video_archive_seeds`) and
  the derived `output_path`. `model` (`Qwen2.5-VL-7B-Instruct-LoRA_HF`) and
  `exp_comment` (`_Qwen2.5-VL-7B-Instruct-LoRA_HF`) are **identical** → the exact-reproduction
  expectation behind G-repro is structurally sound.
- **vs CLIP 13115:** `model` (`...-LoRA_HF` vs `openai_clip-vit-large-patch14-336_HF`), `exp_comment`
  (`_...-LoRA_HF` vs `_openai_clip-...-336_HF`), `group_name`, `output_path`. All four are whitelisted.

No substantive hyperparameter difference. **G-namespace PASSES.** (Evidence: §A.6.)

## 3. Paired per-seed Δ table — computed by me from primary logs

**LoRA arm** = job 13150 (`RAC_video_b3_lora`). **CLIP arm** = job 13115 frozen-CLIP. Same runner,
same 149 ZH test videos, same `--seed` per pair (head-level paired comparison; see §5 caveats).

### 3a. Final-epoch (protocol B) — epoch 29 both arms

| seed | LoRA acc | CLIP acc | **Δacc** | LoRA mF1 | CLIP mF1 | **ΔmF1** |
|---|---|---|---|---|---|---|
| 0 | 0.8456 | 0.8054 | **+0.0402** | 0.8181 | 0.7706 | **+0.0475** |
| 1 | 0.8389 | 0.8054 | **+0.0335** | 0.8113 | 0.7542 | **+0.0571** |
| 2 | 0.8523 | 0.8322 | **+0.0201** | 0.8226 | 0.7913 | **+0.0313** |
| **mean** | 0.8456 | 0.8143 | **+0.0313** | 0.8173 | 0.7720 | **+0.0453** |

- **Δacc:** mean **+0.0313** (= 0.0938/3), **≥ +0.030**; sign **3/3 positive**.
- **ΔmF1:** mean **+0.0453** (= 0.1359/3), **≥ +0.030**; sign **3/3 positive**.
- Both metrics clear the bar with 3/3 sign ⇒ **final-epoch = PASS (MARGINAL)** (see §4 binding language).

### 3b. Val-selected (protocol A) — warmup≥5, max Val_Retrieval acc, roc tie-break

Val-selected epochs re-derived by me from the per-epoch `Val_Retrieval Epoch NN acc/roc` lines
(argmax independently reconstructed, not taken on trust): LoRA s0→**ep20**, s1→**ep26**, s2→**ep19**;
CLIP s0→**ep29**, s1→**ep28**, s2→**ep25**. (Provenance + argmax audit in §A.3–A.5.)

| seed | LoRA acc | CLIP acc | **Δacc** | LoRA mF1 | CLIP mF1 | **ΔmF1** |
|---|---|---|---|---|---|---|
| 0 | 0.8322 (ep20) | 0.8054 (ep29) | **+0.0268** | 0.8023 | 0.7706 | **+0.0317** |
| 1 | 0.8255 (ep26) | 0.8054 (ep28) | **+0.0201** | 0.7956 | 0.7579 | **+0.0377** |
| 2 | 0.8389 (ep19) | 0.8121 (ep25) | **+0.0268** | 0.8065 | 0.7742 | **+0.0323** |
| **mean** | 0.8322 | 0.8076 | **+0.0246** | 0.8015 | 0.7676 | **+0.0339** |

- **Δacc:** mean **+0.0246** (= 0.0737/3), **< +0.030 — FAILS the acc bar**; sign 3/3 positive.
- **ΔmF1:** mean **+0.0339** (= 0.1017/3), ≥ +0.030; sign 3/3 positive.
- The pass criterion requires mean Δacc ≥ +0.030 **AND** mean ΔmF1 ≥ +0.030 (no metric-shopping,
  prereg §10.4). Acc short ⇒ **val-selected = FAIL**.

## 4. Per-protocol verdicts (binding language)

Judged **independently** per the decision rule (prereg §6; `exp-encoder-3seed.md:73-85`), verbatim:
pass = mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND sign 3/3, per dataset × protocol.

- **final-epoch: `PASS (MARGINAL)`.** Reported exactly per the `B3_PREREG_REVIEW.md` §2.2 ruling —
  NOT plain PASS, NO upgrade. A marginal acc pass on one protocol, with the other protocol FAILing on
  acc, does not become a clean/headline pass.
- **val-selected: `FAIL`.** Acc mean +0.0246 sits below the +0.030 bar; the 78-sample ZH val set's
  documented selection tax bites the LoRA arm on acc despite F1 clearing.

**Fixed format:** `final-epoch: PASS (MARGINAL); val-selected: FAIL`.

### 4a. Mandatory sensitivity note (per §2.2 — all three facts required)

1. **Proximity to the bar.** Mean Δacc **+0.0313 is only +0.0013 above the +0.030 bar** (≈4% of the
   bar). This is the entire margin of the pass.
2. **Uneven carry / per-seed check.** Per-seed Δacc spans **+0.0201 … +0.0402**; **seed 2 (+0.0201)
   is itself below the per-seed +0.030 bar**. The pass rests on seeds 0/1 and on F1 (+0.0453, which
   clears cleanly), not on a uniform per-seed margin.
3. **Margin < between-seed spread.** The +0.0013 acc margin is far **smaller than the between-seed
   Δacc spread** (0.0402 − 0.0201 = 0.0201, ~15× the margin) — i.e. the acc pass is within head-seed
   noise on this measure.

**Marginality is structural, not noise to be argued away.** Because G-repro reproduces the
deterministic arcbase anchors **bit-exact** (cached features + inert-only argv deltas), the fresh
LoRA numbers are **not expected to move** run-to-run; there is no stochastic wobble on *this* run.
The marginality therefore comes from (i) proximity to the bar, (ii) the **single fixed CLIP control
arm** (13115, one draw), and (iii) the **single LoRA encoder draw** with head-only seed variance
(the 3 LoRA seeds share one shared feature cache; they vary only the downstream head). The fresh run
**re-confirms** the same +0.0313 under clean same-code same-seed pairing — it does **not** prove
robustness of the effect.

### 4b. No headline upgrade

Even this confirmed marginal final-epoch pass yields, together with HateMM (which passes BOTH
protocols on the **frozen** encoder swap), only:
- a **FAMILY** claim — "MLLM-encoder family" = frozen-Qwen on HateMM + LoRA-Qwen on ZH — **two
  different levers**, not a single-mechanism "MLLM-as-encoder" pass on ≥2 datasets; and
- a result **pending the user's novelty ruling** (§6 below).

The +0.0013 margin, single-CLIP-draw, single-LoRA-encoder-draw, and head-only-variance caveats
travel with any claim built on this pass.

## 5. Load-bearing decomposition (final-epoch means, all same runner) + erratum

| ZH arm (final-ep, seeds 0/1/2 mean) | acc | Δ vs frozen-CLIP |
|---|---|---|
| frozen-CLIP (13115) | 0.8143 | — (baseline) |
| **frozen-Qwen** encoder swap (13115, B1) | 0.8031 | **−0.0112** (FAILS — B1 20th negative) |
| **LoRA-Qwen** (13150 ≡ arcbase 12223-25) | 0.8456 | **+0.0313** |

frozen-Qwen per-seed acc = 0.8188 / 0.8054 / 0.7852 (13115 ep29; §A.7); mean 0.8031.
frozen-Qwen − frozen-CLIP: per-seed +0.0134 / 0.0000 / −0.0470; mean-of-per-seed = **−0.011200**;
diff-of-means (0.803133 − 0.814333) = **−0.011200**. Both give **−0.0112**, not −0.0113.

⇒ On ZH the **frozen** MLLM-encoder swap *loses* (−0.0112), the **LoRA** fine-tune *wins* (+0.0313):
the entire ZH gain is **LoRA task/language adaptation of the encoder, not MLLM-encoder identity**
(consistent with B1's 20th negative).

### 5a. Erratum (STEP 6 — single authorized edit outside this file) — APPLIED

`research-wiki/experiments/exp-lora-zh-b3.md` §7d stated the frozen-Qwen paired mean as **−0.0113**;
the primary-log truth is **−0.0112** (0.8031 − 0.8143 = −0.01120; the internally-consistent value the
banked B1 verdict already uses at `exp-encoder-zh-b1.md:19`). I corrected that one number in place
(§7d table row: `−0.0113 → −0.0112`). This is a contextual, non-gate cell (it restates the B1 20th
negative; it is NOT any B3 pass/fail number), so the correction does not affect any verdict above.
**Note:** the identical −0.0113 rounding also appears in `B3_FORENSIC_RECON.md:184`; that file is
outside my authorized edit scope and is left untouched — if that decomposition ever migrates to a
paper table, −0.0112 is the correct value.

## 6. What this verdict does NOT decide (explicit)

- **Novelty.** B3 measures the goal's **performance clause only** (+0.03 acc AND +0.03 F1). Whether a
  LoRA-SFT-encoder performance pass counts toward the goal's **"novel"** clause — LoRA being an
  RA-HMD-family "MIXED performance lever, not novelty" (`query_pack.md:44`; `B1_PREREG_REVIEW.md:64`)
  — is an **explicit PENDING USER RULING**, not decided here.
- **Whether the "MLLM-encoder family" (frozen-HateMM + LoRA-ZH) counts as a "both-datasets"
  headline.** The two passes ride on different mechanisms; accepting a family-level framing as a
  ≥2-dataset headline is a user call. If a *single* mechanism is required, neither frozen nor LoRA
  qualifies (frozen: HateMM-only; LoRA: ZH-only).
- **The barred-comparison accounting note** (`PAPER_MASTER_TABLES.md:58`, LoRA-Qwen stack vs
  frozen-CLIP floor "不可直接同格并比"): whether B3's same-runner same-seed pairing overrides it for a
  paper claim is a user decision.
- **Encoder-draw stability of the LoRA lever.** B3 is a **head-seed** paired test on **one** LoRA-SFT
  encoder draw (single shared cache). It does NOT establish LoRA-SFT-training-seed variance; that
  would need ≥3 fresh LoRA-SFT re-trainings + re-extraction (out of B3 scope, pre-declared).

---

## A. Primary-log provenance (every number re-read by me)

**A.1 Fresh LoRA final-epoch (job 13150), `enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed<s>_13150.trainlog`:**
- s0 `:272` `Test_Retrieval Epoch 29 macroF1: 0.8181 … acc: 0.8456 roc: 0.9036`
- s1 `:273` `… Epoch 29 macroF1: 0.8113 … acc: 0.8389 roc: 0.8955`
- s2 `:268` `… Epoch 29 macroF1: 0.8226 … acc: 0.8523 roc: 0.9115`

**A.2 Arcbase anchors (final-epoch), `arcbase_MHC_zh_...-LoRA_HF_seed<s>_1222<j>.trainlog`:**
- s0/12223 `:272` `macroF1: 0.8181 … acc: 0.8456` · s1/12224 `:273` `macroF1: 0.8113 … acc: 0.8389`
  · s2/12225 `:268` `macroF1: 0.8226 … acc: 0.8523`. (⇒ G-repro exact.)

**A.3 Fresh LoRA val-selected (job 13150):** argmax over `Val_Retrieval Epoch NN acc` (≥ep5),
roc tie-break; last `Best model so far, saving...` marker = selected epoch (verified in-context):
- s0 → **ep20**: `Val_Retrieval Epoch 20 acc: 0.8718 roc: 0.9229` (last save marker `:201`);
  Test `:199` `macroF1: 0.8023 … acc: 0.8322`. (ties val-acc 0.8718 at ep9 roc 0.9207 < ep20 0.9229.)
- s1 → **ep26**: `Val 0.8718 roc: 0.9129` (last marker `:250`); Test `:248` `macroF1: 0.7956 … acc: 0.8255`.
- s2 → **ep19**: `Val 0.8718 roc: 0.9086` (last marker `:189`); Test `:187` `macroF1: 0.8065 … acc: 0.8389`.

**A.4 CLIP control final-epoch (job 13115), `enc3s_MHC_zh_openai_clip-...-336_HF_seed<s>_13115.trainlog`:**
- s0 `:275` `macroF1: 0.7706 … acc: 0.8054` · s1 `:274` `macroF1: 0.7542 … acc: 0.8054`
  · s2 `:271` `macroF1: 0.7913 … acc: 0.8322`.

**A.5 CLIP control val-selected (job 13115):** argmax reconstructed from the full per-epoch
`Val_Retrieval Epoch NN acc/roc` ladder (all 30 epochs read):
- s0 → **ep29** (max val acc 0.8077; last marker `:277`); Test `:275` `macroF1: 0.7706 … acc: 0.8054`.
- s1 → **ep28** (val acc 0.7821 tied at ep18/27/28, roc tie-break 0.8836 max; last marker `:267`);
  Test `:265` `macroF1: 0.7579 … acc: 0.8054`.
- s2 → **ep25** (max val acc 0.8205; last marker `:240`); Test `:238` `macroF1: 0.7742 … acc: 0.8121`.

**A.6 Namespace echoes (log line 1):** LoRA 13150 `group_name='RAC_video_b3_lora'`,
`exp_comment='_Qwen2.5-VL-7B-Instruct-LoRA_HF'`, `model='Qwen2.5-VL-7B-Instruct-LoRA_HF'`;
arcbase 12223-25 `group_name='RAC_video_archive_seeds'`, same `exp_comment`, same `model`;
CLIP 13115 `group_name='RAC_video_archive_seeds'`, `exp_comment='_openai_clip-vit-large-patch14-336_HF'`,
`model='openai_clip-vit-large-patch14-336_HF'`. Substantive fields (topk=20, epochs=30, batch_size=64,
lr=0.0001, warmup=5, proj_dim=1024, map_dim=1024, fusion_mode='align', loss='triplet', metric='cos',
hybrid_loss=True, lambda_seg=0.0, archive_feats=None, force=False) identical across all three.

**A.7 Frozen-Qwen decomposition (job 13115), `enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed<s>_13115.trainlog`:**
- s0 `:275` `macroF1: 0.7864 … acc: 0.8188` · s1 `:272` `macroF1: 0.7759 … acc: 0.8054`
  · s2 `:269` `macroF1: 0.7514 … acc: 0.7852`. mean acc 0.8031; mean Δ vs CLIP = −0.0112.

---

## B. Anomalies

**None.** No FAILED/NaN/OOM/NO_PARSE; all three 13150 trainlogs reached epoch 29 with `Last Epoch,
saving...`; the sbatch `RESULT_ROW` lines in `enc3seed_13150.out:{286,568,845}` agree with my
trainlog transcription to 4 dp on all 12 fresh readings; the executor's raw transcription in
`B3_EXECUTION_RECORD.md` matches my independent re-read exactly (no transcription error found).
The only correction is the pre-flagged §7d −0.0113→−0.0112 erratum (§5a), which touches no gate or
verdict number.
