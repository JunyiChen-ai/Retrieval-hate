---
type: experiment
node_id: exp:exp-encoder-zh-b1
title: "B1 — MLLM-as-encoder vs frozen-CLIP on the UNTESTED MHC-ZH cell: 3-seed paired encoder-swap test, dual protocol, archive OFF (PRE-REGISTRATION, DRAFT-REV1-AWAITING-DELTA-CHECK)"
idea_id: ""
status: DRAFT-REV1-AWAITING-DELTA-CHECK
verdict: draft-unreviewed
confidence: n/a
date: "2026-07-14"
hardware: "1x A100 (SLURM), frozen features cached -> ~20-60 s/run; NO extraction needed"
duration: "6 runs (or 4 if seed0 reused as cross-check only), seconds each, one serial sbatch"
provenance: "PRE-REGISTRATION ONLY — NO runs executed yet. Reuses cached features data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_{openai_clip-vit-large-patch14-336_HF,Qwen2.5-VL-7B-Instruct_HF}.pt (Qwen dated 2026-07-02, CLIP dated 2026-07-01). Cross-check refs: rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_1151518.trainlog (frozen-Qwen ZH seed0, old code); exp-consensus-zh-seeds.md:60-64 (frozen-CLIP ZH λ=0 floor, 5 seeds, jobs 12130 s0 + 12297-12300 s1-4). Template: scripts/slurm/train_archive_baseline.sbatch; runner (to adapt): scripts/slurm/enc3seed.sbatch. Parent test: exp-encoder-3seed.md."
added: 2026-07-14T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "frozen-CLIP", "encoder-swap", "multi-seed", "paired-test", "MHC_zh", "MHC-ZH", "pre-registered", "DRAFT-REV1-AWAITING-DELTA-CHECK", "B1"]
---

# B1 — MLLM-as-encoder vs frozen-CLIP on the UNTESTED MHC-ZH cell (PRE-REGISTRATION)

> **B1 FAIL under BOTH protocols — 20th negative (independent verdict review 2026-07-14, refine-logs/B1_VERDICT_REVIEW.md; job 13115).** Gates 1a/1b passed cleanly (frozen-Qwen s0 reproduces 1151518 to 4dp; CLIP s0 reproduces 12130 exactly — no code confound). Paired Qwen−CLIP: final-epoch mean −0.0112 acc / −0.0008 mF1 (1/3 sign), val-sel −0.0045 / +0.0005 (1/3 sign); robust to dropping the anomalous s2 (−0.047). P8c language-match hypothesis REJECTED for the frozen encoder; the ZH 0.8537 gain is LoRA-driven. HateMM remains the sole dataset where the frozen-Qwen encoder swap passes. Note for analysis: Qwen test-ROC beats CLIP on every seed (0.88–0.90 vs 0.83–0.84) without converting to acc/F1.

> **STATUS: `DRAFT-REV1-AWAITING-DELTA-CHECK` — PRE-REGISTRATION ONLY. NO SLURM job
> submitted, NO GPU used, NO test touch spent. Reviewed 2026-07-14
> (`refine-logs/B1_PREREG_REVIEW.md`): APPROVED with 3 minor mandatory revisions;
> Rev-1/2/3 applied below (see Revision history). Awaiting reviewer delta-check +
> conditional authorization BEFORE any run.**

**verdict:** `draft-unreviewed` — this file only *proposes* the test. · **confidence:** n/a

## Purpose (one line)

Apply the project's single banked positive — **swapping the frozen-CLIP encoder for
frozen Qwen2.5-VL-7B hidden states in the otherwise-unchanged RGCL head** — to the
**one dataset cell the original 3-seed encoder test never ran: MHC-ZH (Chinese
MultiHateClip, `dataset=MHC_zh`)**. This is the missing fourth quadrant of
`exp-encoder-3seed.md`, which tested HateMM (PASS both protocols) and MHC-EN (FAIL
both) but **not** ZH.

## Why MHC-ZH was excluded from the original encoder test (recon)

- The parent test's **Design** names its datasets explicitly as *"HateMM and MHC
  (English MultiHateClip; `dataset=MHC`)"* (`exp-encoder-3seed.md:38`). ZH is simply
  **out of scope by declaration** — never listed, never run in that campaign.
- It was **scope economy, not a principled exclusion.** The parent test reused
  pre-existing old-code seed0 + current-code EN-Qwen s1/s2 logs
  (`exp-encoder-3seed.md:11,50-53,87-98`) — i.e. it was built around the
  HateMM/MHC-EN logs that already existed. No ZH encoder-swap logs existed at 3 seeds
  under the archive-OFF `train_archive_baseline` harness, so ZH was left for a
  follow-up rather than deliberately ruled out on noise grounds.
- The **78-sample ZH dev / val-selection-tax** concern is real and documented
  (`PAPER_MASTER_TABLES.md:56-57`: *"78 样本 ZH dev 上 val-acc 选点相对 selection-free
  自损 ~2 acc 点"*; `exp-consensus-zh-seeds.md:127-133`), and the **ZH headline-protocol
  choice is still a pending user decision** (`PAPER_MASTER_TABLES.md:49-52,226,246,254-256`:
  val-selected 0.827 does not cross 0.85; final-epoch 0.8537 does; *"因过线才换口径 =
  rule-shopping 风险"*). That pending decision is a reason to report **BOTH protocols
  independently** here (as the parent test did), NOT a reason to skip ZH.

## Hypothesis (pre-registered)

**H1 (ZH).** Replacing the frozen-CLIP video/text encoder with a frozen-MLLM
(Qwen2.5-VL-7B-Instruct, **no LoRA**) encoder on **`dataset=MHC_zh`** — **every other
component identical** (topk=20, `lambda_seg=0`, archive OFF, same train/dev/test
split, lr=1e-4 / epochs=30 / batch=64 / proj=map=1024 / dropout[0.2,0.4,0.1] /
hard-neg / hybrid-loss / warmup=5) — yields a **>= +0.030 accuracy AND >= +0.030
macro-F1** improvement of the RGCL retrieval head, on **3/3 seeds**, relative to the
frozen-CLIP ZH floor.

The only manipulated variable is `--model`
(`openai_clip-vit-large-patch14-336_HF` -> `Qwen2.5-VL-7B-Instruct_HF`). Both encoders
are frozen; only their pre-extracted feature `.pt` caches feed the identical RGCL head.

**Mechanistic rationale (why ZH might differ from the MHC-EN FAIL).** MHC-EN and MHC-ZH
share the same `MHC` pipeline and the same encoder-swap failed on EN. Three reasons ZH
could still pass where EN did not:
1. **Language match.** Qwen2.5-VL is Chinese-strong; CLIP ViT-L/14-336's text tower is
   English-centric. The documented ZH bottleneck is exactly this: *"ZH 瓶颈 = 冻结
   English-centric CLIP text tower 把中文 byte-fragment 97% 截断"*
   (`PAPER_MASTER_TABLES.md:188,237`; EXP_p8). Swapping to a native-Chinese encoder
   directly removes that truncation — a mechanism that does **not** apply to EN.
2. **High-quality, above-SOTA control.** The frozen-CLIP ZH floor already beats MoRE
   (M-F1 0.7706 vs MoRE 0.7475; `ITERATION_LOG.md:129,140-141`) — the control arm is a
   strong, well-characterized 5-seed baseline, so a pass here would be measured against
   a credible floor, not a weak one. **Deliberately NOT cited as a reason to expect a
   pass: the LoRA-Qwen 0.8537.** That number comes from a *different lever* (LoRA-SFT of
   the encoder, `PAPER_MASTER_TABLES.md:43,49`; review Task A.1/A.3), and the gap
   between LoRA-Qwen (final acc 0.8537) and the frozen-Qwen seed0 (final acc 0.8188) is,
   if anything, **evidence the ZH gain is LoRA-driven, not encoder-driven — a HEADWIND
   for H1, not a tailwind.** (Rev-1, `refine-logs/B1_PREREG_REVIEW.md`.)
3. **Different label quality.** The ZH-validated **consensus-denoising** pillar
   (`exp-consensus-zh-seeds.md`) shows ZH labels are usable enough that de-poisoning
   mechanisms behave sensibly on them — evidence ZH annotation quality is not the
   limiter that SAV diagnosed for MHC-EN.

**HONEST PRIOR — read before believing H1 (this is a `partial`-to-`negative` prior,
NOT a HateMM-style strong-positive prior).** See "Honest prior / expected outcome"
below: the ONE existing frozen-Qwen ZH data point (seed0, old code, job 1151518)
**LOSES to frozen-CLIP under val-selection** (−0.0135 acc / −0.0294 F1), and the ZH
gains that reached 0.8537 came from **LoRA-SFT of the encoder, not frozen hidden
states**. H1 is registered as a genuine open question, and the most probable outcome
is a `partial` verdict (final-epoch weakly positive, val-selected flat/negative),
mirroring MHC-EN more than HateMM.

## Design (pre-registered)

- **Dataset:** MHC-ZH only (`dataset=MHC_zh`, Chinese MultiHateClip; Bilibili).
  Splits: train 579 / val 78 / test 149 (`data/gt/MHC_zh/{train,val,test}.jsonl`;
  `lb_scgp_global_r2_m1_cache_v1_common.py:46` EXPECTED_TRAIN_N["MHC_zh"]=579).
- **Training data = the single dataset's own train split ONLY** (user rule). No gold
  span/attribute annotations used as supervision; no cross-seed ensembles; no OCR.
  The RGCL head trains on the 579-row ZH train split; kNN memory is the ZH train bank.
- **Seeds:** 0 / 1 / 2, paired within seed (CLIP vs Qwen at the same seed).
- **Arms:** `frozen-CLIP` (control) vs `frozen-Qwen` (treatment). Both via the
  archive-OFF path of `scripts/slurm/train_archive_baseline.sbatch`
  (`archive_feats=None` gates all archive/seg behaviour OFF in `src/run_rac.py`;
  `lambda_seg=0`), so the two arms differ **only** in `--model` — identical to
  `exp-encoder-3seed.md:40-43`.
- **Feature inputs already cached — NO extraction required (asset check):**
  - `data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct_HF.pt` (16.6 MB, 2026-07-02)
  - `data/CLIP_Embedding/MHC_zh/dev_seen_Qwen2.5-VL-7B-Instruct_HF.pt` (2.24 MB)
  - `data/CLIP_Embedding/MHC_zh/test_seen_Qwen2.5-VL-7B-Instruct_HF.pt` (4.28 MB)
  - `data/CLIP_Embedding/MHC_zh/train_openai_clip-vit-large-patch14-336_HF.pt` (4.17 MB, 2026-07-01)
  - `data/CLIP_Embedding/MHC_zh/dev_seen_…_HF.pt` / `test_seen_…_HF.pt` (present)
  - Loader path: `src/data_loader/dataset.py:499` lists `MHC_zh` as supported;
    `:587-589` builds `{path}/{dataset}/{split}_{model}.pt`. Qwen ZH is 3584-dim
    image+text (verified in `rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_1151518.trainlog:2-3`),
    CLIP ZH is 1024 img / 768 text. Both feed the same `classifier_hateClipper`.
  - **What is missing to run ZH×Qwen×3seeds: NOTHING (no extraction, no new config).**
    The only artifact to author is a 6-line `CONFIGS` edit to the enc3seed runner
    (or a `DATASET=MHC_zh` loop over `train_archive_baseline.sbatch`).

### Config-match verification (to run FIRST, gates everything)

Mirror `exp-encoder-3seed.md:126-146`: before tabulating, run the **new-code seed0
reproduction cross-check** —
- **frozen-Qwen ZH s0 (new code)** must reproduce the **old-code** frozen-Qwen ZH seed0
  (`rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_1151518.trainlog`, group `RAC_video`) to 4
  printed decimals under both protocols. (The parent test proved every flag added to
  `run_rac.py` since the old runs is inert at defaults for HateMM/MHC-EN; ZH shares the
  same code path, so this is a confirmatory bit-for-bit check, not a new risk.)
- **frozen-CLIP ZH s0 (new code) vs job 12130 = confirmatory CROSS-RUNNER check, NOT a
  hard gate** (Rev-2). The reference floor (`exp-consensus-zh-seeds.md:60`, job 12130:
  val-sel = final = 0.7706 F1 / 0.8054 acc, ep29) was produced by a **different runner**
  (`train_consensus_seeds.sbatch`, `exp_comment=_seg0_full`, λ_seg=0 through the seg
  code path). This is the first time CLIP-ZH runs through `train_archive_baseline`'s
  archive-OFF path, so the two paths' equivalence on this cell is asserted-by-
  transitivity, not previously exercised. On a >0.0001 mismatch: open a code-path
  audit, which is **permitted to conclude** "benign archive-OFF-vs-seg-λ0 runner
  difference; same-runner s0 is the authoritative control" — no automatic campaign
  HALT. The B1 same-runner CLIP s0/1/2 band remains the authoritative control either way.

Namespace diff between the two arms must be **identical except `model=`** (and derived
`exp_comment`/`output_path`), exactly as audited in the parent test.

## Protocols (both reported, judged independently — NO protocol selection)

Transcribed from `exp-encoder-3seed.md:66-71`:

- **(A) val-selected:** pick epoch >= warmup 5 with max Val_Retrieval acc (roc
  tie-break); report that epoch's **Test** macro-F1 / acc / roc.
- **(B) final-epoch:** report **Test** macro-F1 / acc / roc at the last trained epoch
  (29), the standard selection-free protocol.

**Primary protocol for this ZH experiment = (B) final-epoch — REPORTING-EMPHASIS ONLY,
NOT a decision gate** (Rev-3). Rationale (encoder-independent): the 78-sample ZH val set
imposes a documented ~2-acc-point val-selection tax
(`PAPER_MASTER_TABLES.md:56-57`; `exp-consensus-zh-seeds.md:127-133`;
`ABLATION_transcript_vs_archive.md:73-74`), so the selection-free protocol is the less
noisy lens for a 3-seed paired test on this dataset. That is the sole basis for the
designation; no headroom argument enters it. **"Primary" changes which protocol leads
the write-up, and nothing else:** both protocols are judged independently under the
identical rule (4) below, the fixed write-up format "final-epoch: pass/fail;
val-selected: pass/fail" (kill rule 3, transcribed from the parent) governs, and the
parent's ">= 2 datasets" headline still requires ZH to pass under the **same** protocol
HateMM passed (HateMM passed both, so either ZH protocol-pass completes a headline pair
— but only within that protocol's column). The designation is declared **now, before
any run**, to avoid rule-shopping.

## Decision rule (pre-registered, transcribed verbatim from `exp-encoder-3seed.md:73-85`)

> For each dataset x protocol:
> 1. **Per-seed paired difference** delta = (Qwen - CLIP) for acc and macro-F1 at seeds 0/1/2.
> 2. **3-seed mean +/- std** of the paired delta; **sign consistency** (how many of 3 seeds positive).
> 3. n=3 is too small for a formal bootstrap; report the paired-t statistic **as an effect-size
>    descriptor only** alongside the mean/std and sign count — no significance claim is made from n=3.
> 4. **Pass criterion (per dataset x protocol):** mean paired delta_acc >= +0.030 AND
>    mean paired delta_mF1 >= +0.030 AND sign consistency 3/3 positive.
> 5. **Headline claim ("MLLM-as-encoder helps"):** requires the pass criterion met on
>    **>= 2 datasets** under a stated protocol. Each protocol is judged separately; if EN
>    passes only under final-epoch, the verdict is written exactly as
>    "final-epoch: pass; val-selected: fail".

**Application to B1.** Judge ZH under the identical rule (4). A ZH **PASS** under either
protocol supplies the **second dataset** the parent test's rule (5) needs: combined with
HateMM's confirmed PASS, "MLLM-as-encoder helps on >= 2 datasets" would then be met under
that protocol — a clean second-dataset pass of the goal's "+0.03 acc AND +0.03 F1"
target. A ZH **FAIL** leaves HateMM as the only formally passing encoder dataset (status
quo, `MEMORY.md`).

## Reused reference readings (pre-existing — frozen-CLIP ZH floor + frozen-Qwen ZH s0)

Parsed with the same val-selection rule as the sbatch (warmup>=5, val-acc, roc tie-break).

### Frozen-CLIP ZH λ=0 floor (control reference) — `exp-consensus-zh-seeds.md:56-64`, 5 seeds

| arm | seed | selEp | val-sel F1 / acc | final F1 / acc |
|---|---|---|---|---|
| CLIP floor | 0 | 29 | 0.7706 / 0.8054 | 0.7706 / 0.8054 |
| CLIP floor | 1 | 28 | 0.7579 / 0.8054 | 0.7542 / 0.8054 |
| CLIP floor | 2 | 25 | 0.7742 / 0.8121 | 0.7913 / 0.8322 |
| CLIP floor | 3 | 16 | 0.7421 / 0.7785 | 0.7548 / 0.7987 |
| CLIP floor | 4 | 23 | 0.7799 / 0.8121 | 0.7259 / 0.7718 |

- **seeds 0/1/2 mean** (the B1 seeds): val-sel **0.7676 F1 / 0.8076 acc**;
  final **0.7720 F1 / 0.8143 acc**.
- 5-seed mean (`exp-consensus-zh-seeds.md:80,83`; `PAPER_MASTER_TABLES.md:46`):
  val-sel 0.7649 ± 0.0151 F1 / 0.8027 ± 0.0139 acc; final 0.7594 ± 0.0240 F1 /
  0.8027 ± 0.0215 acc.
- Jobs: s0 = 12130 (pre-W5); s1-4 = 12297-12300 (post-W5). Encoder = frozen CLIP
  ViT-L/14-336. **These are the authoritative frozen-CLIP ZH control values**; the B1
  same-runner CLIP s0/1/2 re-runs must match s0 (and land within the s1/s2 band).

### Frozen-Qwen ZH seed0 (treatment reference) — `rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_1151518.trainlog`, old code, 1 seed

| arm | seed | selEp | val-sel F1 / acc | final (e29) F1 / acc |
|---|---|---|---|---|
| Qwen | 0 | 22 | 0.7412 / 0.7919 | 0.7864 / 0.8188 |

- Val-selection under warmup>=5: max Val_Retrieval acc = 0.8205 at epochs {22,26,28};
  roc tie-break selects e22 (roc 0.8693) -> Test F1 0.7412 / acc 0.7919. (The "epoch-0
  caveat" flagged in `DESIGN_iter1.md:83,345` refers to the raw val-acc peak sitting at
  epoch 0 = 0.8333; under the registered warmup>=5 rule that epoch is excluded, and the
  warmup-consistent read is e22 = 0.7412/0.7919, which matches the reported iter1
  number `DESIGN_iter1.md:275`.)
- Encoder = frozen Qwen2.5-VL-7B (no LoRA), 3584-dim, group `RAC_video`. This is exactly
  the seed0 the B1 Qwen arm re-runs under current code as the reproduction cross-check.

### Single available paired seed0 delta (Qwen − CLIP, ZH) — the whole prior we have

| protocol | Δacc (s0) | ΔmF1 (s0) | verdict |
|---|---|---|---|
| (A) val-selected | 0.7919 − 0.8054 = **−0.0135** | 0.7412 − 0.7706 = **−0.0294** | Qwen LOSES |
| (B) final-epoch | 0.8188 − 0.8054 = **+0.0134** | 0.7864 − 0.7706 = **+0.0158** | weak +, **< +0.030 bar** |

Neither protocol shows anything near the HateMM +5 acc / +6 F1 effect at seed0. This is
the honest baseline expectation for the 3-seed test.

## Runs to execute (this campaign) — pending authorization

Runner = `scripts/slurm/enc3seed_zh_b1.sbatch` (authored 2026-07-14: verbatim copy of
`scripts/slurm/enc3seed.sbatch` with ONLY the `CONFIGS` block changed to the 6 ZH rows;
diff + runtime cross-check in `refine-logs/B1_IMPL_NOTES.md`). One serial sbatch,
current code; each run = the exact `train_archive_baseline.sbatch` python command,
differing only in `--dataset MHC_zh` / `--model` / `--seed`:

| # | dataset | model | seed | role |
|---|---|---|---|---|
| 1 | MHC_zh | CLIP | 0 | control + s0 reproduction cross-check vs 12130 |
| 2 | MHC_zh | CLIP | 1 | control |
| 3 | MHC_zh | CLIP | 2 | control |
| 4 | MHC_zh | Qwen | 0 | treatment + s0 reproduction cross-check vs 1151518 |
| 5 | MHC_zh | Qwen | 1 | treatment |
| 6 | MHC_zh | Qwen | 2 | treatment |

(6 runs. Optionally reduce to 4 by treating the two seed0 arms as reproduction-only and
reusing 12130 / 1151518 directly — but a same-runner seed0 is cleaner and costs seconds.
Mirror the parent test, which re-ran seed0 under current code rather than trusting the
old-code logs for the primary table: `exp-encoder-3seed.md:44-49`.)

## Honest prior / expected outcome (declared before running)

- **The banked positive is `frozen-Qwen`, and it is the encoder that ALREADY LOST on
  ZH at seed0** (val-sel −0.029 F1). The ZH result that reaches published-SOTA-level
  0.8537 is **LoRA-Qwen**, a different lever the project classifies as a *"MIXED
  performance lever, not novelty"* (`query_pack.md:44`; `gap_map.md` LoRA note) — and
  even there, archive-kNN contributes exactly 0 at final-epoch
  (`PAPER_MASTER_TABLES.md:49-52`). **Do NOT conflate 0.8537 (LoRA-Qwen) with either arm
  of this frozen-encoder test.** The B1 control is the frozen-CLIP floor ≈ 0.808 acc
  (s0-2) and the B1 treatment must beat it by +0.03 on frozen hidden states alone.
- MHC-EN — which shares the exact `MHC` pipeline — **FAILED the same encoder swap under
  both protocols** (`exp-encoder-3seed.md:212-234`); SAV review diagnosed MHC-EN as
  data/label-limited. ZH's escape hatch is the **language-match mechanism** (P8c: CLIP
  truncates 97% of Chinese byte-fragments), which is real and EN-inapplicable, but it is
  a hypothesis, not a measured effect for the frozen encoder.
- **Most probable outcome: `partial`** — final-epoch weakly positive (seed0 already
  +0.013 acc / +0.016 F1 but below the +0.03 bar), val-selected flat-to-negative. A full
  PASS on 3/3 seeds under either protocol would be a genuinely new positive and the
  cleanest available second-dataset confirmation; it is possible (final-epoch seed0 is
  positive and the 78-dev val tax is removed under protocol B) but should not be assumed.

## Kill rules (pre-registered)

1. **Reproduction gate — split into a hard gate and a confirmatory check (Rev-2).**
   - **1a (HARD gate, same-runner/same-code-lineage).** If new-code frozen-Qwen ZH s0
     does not reproduce `1151518` to 4 printed decimals under both protocols — **HALT**,
     do not tabulate, open a code-path audit. This is the true old-code-vs-new-code
     confound, exactly analogous to the parent test's seed0 reproduction audit
     (`exp-encoder-3seed.md:126-136`).
   - **1b (confirmatory CROSS-RUNNER check, audit-on-mismatch).** If new-code
     frozen-CLIP ZH s0 diverges >0.0001 from the 12130 floor (0.7706/0.8054): open a
     code-path audit **before** tabulating, but the audit is permitted to conclude
     "benign archive-OFF-vs-seg-λ0 runner difference; the same-runner CLIP s0/1/2 band
     is the authoritative control" and let the campaign proceed on that basis. 12130 is
     a cross-runner reference (`train_consensus_seeds.sbatch`), not a same-runner
     reproduction target; a mismatch here does not automatically HALT.
2. **Namespace-diff gate.** If the two arms' `Namespace` differs in any substantive field
   other than `model=`/`exp_comment=`/`output_path=`, HALT.
3. **No protocol-shopping.** Primary = final-epoch, declared above. Both protocols
   reported regardless of which (if either) passes; the write-up format is fixed
   ("final-epoch: pass/fail; val-selected: pass/fail"), transcribed from the parent rule.
4. **No metric-shopping.** Both Δacc AND ΔmF1 must clear +0.03 with 3/3 sign consistency
   for a PASS; an F1-only or acc-only move is reported as FAIL-with-direction, exactly as
   the parent test handled MHC-EN.
5. **Single test touch.** See below; if the budgeted touch is spent, no re-runs with
   tweaked knobs on the ZH test set under this pre-registration.

## Test-touch discipline

- **Did the original encoder test consume any ZH test touch? NO — verified.** The parent
  test's datasets are HateMM and MHC-EN only (`exp-encoder-3seed.md:38`); no ZH run
  appears in its runs table (`:105-118`) or provenance. The `enc3s_MHC_*` trainlogs are
  all `dataset='MHC'` = English (Qwen seed0 log shows 3584-dim EN, and every enc3s
  Namespace reads `dataset='MHC'`). **This encoder-swap comparison has never touched the
  ZH test set.**
- **Budget for B1: ONE test touch.** Note the MHC-ZH *test set itself* is not virgin — it
  has been read by many prior experiments (consensus-zh-seeds, archive-knn-seeds, P3/P4,
  P9, temporal). But the specific pre-registered question "does the frozen-CLIP->frozen-
  Qwen encoder swap yield +0.03 on ZH under the archive-OFF RGCL head at 3 seeds" is new,
  and is allotted exactly one evaluation. No adaptive re-running against ZH test.

## GPU budget

- **Extraction: 0 GPU-hours** — all six input caches already exist (asset check above).
- **Training: 6 runs × ~20-60 s** with cached frozen features. Parent-test runtime
  evidence (job 12850, cached features): MHC-EN CLIP 20-36 s, MHC-EN Qwen 33 s, HateMM
  20-52 s (one I/O-contended outlier 2:55) per run
  (`enc3s_*_12850.trainlog` tqdm final bars). ZH train (579) ≈ EN train (549), so expect
  the same band. **Total wall clock < ~20 min, 1× A100, one serial sbatch job.**
- Fits the per-user cap (16 CPU / 128 GB / 2 GPU); requests 1 GPU / 8 CPU / 64 GB like
  the parent runner. Submit with **no `--time`** (project rule); expect initial
  `PENDING (JobHeldUser)` and wait for auto-release.

## Single-submit ceremony (pre-registered)

1. Freeze this pre-registration (review sign-off DONE; delta-check of Rev-1/2/3 pending).
2. ZH `CONFIGS` runner authored as `scripts/slurm/enc3seed_zh_b1.sbatch` (6 rows);
   diff-verified CONFIGS-only vs parent (`refine-logs/B1_IMPL_NOTES.md` §b) — the python
   command is byte-identical to the parent except `--dataset MHC_zh`.
3. Optional smoke: 1-epoch dry run of one ZH config to confirm both feature caches load
   and dims wire into `classifier_hateClipper` (3584->1024 Qwen / 1024+768->1024 CLIP).
4. One `sbatch` submission of the full 6-run serial job. No mid-run resubmissions.
5. Read back every number from the raw `enc3s_MHC_zh_*` trainlogs (line-numbered
   provenance table), run the reproduction gate FIRST, then tabulate per-seed deltas and
   apply the decision rule verbatim.

## Readiness verdict (what remains before this can be submitted)

1. **Fresh pre-registration review** — DONE 2026-07-14: APPROVED with 3 minor mandatory
   revisions (`refine-logs/B1_PREREG_REVIEW.md`); Rev-1/2/3 applied in this revision.
2. **Reviewer delta-check** of the applied revisions — PENDING.
3. **Implementation check** — runner authored: `scripts/slurm/enc3seed_zh_b1.sbatch`
   (CONFIGS-only copy of `enc3seed.sbatch`; see `refine-logs/B1_IMPL_NOTES.md` for the
   diff, cache/dim cross-check, and log-collision check). Reviewer notes IN-1
   (frozen `Qwen2.5-VL-7B-Instruct_HF`, never `-LoRA_HF`) and IN-2 (`FORCE=False`
   collision re-check at submit time) to be verified at delta-check.
4. **Smoke (optional)** — 1-epoch load check per arm (IN-3).
5. **Conditional authorization** — explicit user/main go (GPUs are shared with the
   user's own loop; `CLAUDE.md` — every GPU task via SLURM, subagents do the work).
6. **Single submit** — one serial sbatch, ~20 min, 1 A100.

**Nothing is blocked on data or compute.** The only gates are delta-check + authorization.

## Connections
- extends -> `exp:exp-encoder-3seed` (the HateMM PASS / MHC-EN FAIL 3-seed encoder test; this adds the missing ZH cell)
- controls-against -> `exp:exp-consensus-zh-seeds` (frozen-CLIP ZH λ=0 floor, 5 seeds)
- cross-checks -> `rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_1151518` (frozen-Qwen ZH seed0, old code)
- contrasts-with -> `exp:exp-lora-sft-encoder` (LoRA-Qwen is the ZH lever that reached 0.8537; this test isolates FROZEN Qwen, no LoRA)
- reviewed-by -> `refine-logs/B1_PREREG_REVIEW.md` (2026-07-14, APPROVED + Rev-1/2/3)
- implemented-by -> `scripts/slurm/enc3seed_zh_b1.sbatch` + `refine-logs/B1_IMPL_NOTES.md`

## Revision history

| rev | date | status | change | authority |
|---|---|---|---|---|
| r0 | 2026-07-14 | DRAFT-UNREVIEWED | Initial pre-registration (recon + draft; no runs). | B1 prep agent |
| r1 | 2026-07-14 | DRAFT-REV1-AWAITING-DELTA-CHECK | Applied the 3 mandatory review revisions from `refine-logs/B1_PREREG_REVIEW.md` (APPROVED): **Rev-1** — mechanistic rationale #2 no longer cites LoRA-Qwen 0.8537 as a frozen-swap tailwind; reframed to the encoder-independent above-SOTA control fact, and the LoRA/frozen gap explicitly marked a HEADWIND for H1. **Rev-2** — reproduction gate split: 1a hard same-lineage gate (frozen-Qwen s0 vs 1151518, HALT on mismatch) vs 1b confirmatory cross-runner check (frozen-CLIP s0 vs 12130, audit-on-mismatch, no auto-HALT; same-runner CLIP band = authoritative control); Config-match section updated to match. **Rev-3** — "primary = final-epoch" explicitly marked reporting-emphasis only (sole rationale = 78-dev val-selection tax), NOT a decision gate; both protocols still judged independently under the verbatim parent rule. Also: runner authored (`scripts/slurm/enc3seed_zh_b1.sbatch`, CONFIGS-only delta) + `refine-logs/B1_IMPL_NOTES.md`; readiness list updated. No floor, decision-rule, seed, or budget change. | `refine-logs/B1_PREREG_REVIEW.md` |
