---
type: experiment
node_id: exp:exp-encoder-3seed
title: "MLLM-as-encoder vs frozen-CLIP: 3-seed paired encoder-swap test (HateMM + MHC-EN), dual protocol, archive OFF"
idea_id: ""
verdict: pending
confidence: ""
date: "2026-07-11"
hardware: "1x A100 (SLURM), frozen features cached -> ~21-25 s/run"
duration: "10 runs, seconds each, one serial sbatch"
provenance: "PRE-REGISTERED before seed1/seed2 (and new-code seed0) runs. reused refs: rgcl_HateMM_openai_clip-vit-large-patch14-336_HF_1035814.trainlog (HateMM CLIP s0, old code), rgcl_HateMM_Qwen2.5-VL-7B-Instruct_HF_1029175.trainlog (HateMM Qwen s0, old code), rgcl_MHC_openai_clip-vit-large-patch14-336_HF_1035813.trainlog (MHC-EN CLIP s0, old code), rgcl_MHC_Qwen2.5-VL-7B-Instruct_HF_1029174.trainlog (MHC-EN Qwen s0, old code; == mllm_train_12113.out bit-for-bit), arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2}_1227{5,6}.trainlog (MHC-EN Qwen s1/s2, current code). template: scripts/slurm/train_archive_baseline.sbatch; runner: scripts/slurm/enc3seed.sbatch"
added: 2026-07-11T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "frozen-CLIP", "encoder-swap", "multi-seed", "paired-test", "HateMM", "MHC", "pre-registered"]
---

# MLLM-as-encoder vs frozen-CLIP: 3-seed paired encoder-swap test

**verdict:** `pending` (pre-registration; results section filled after the runs)

## Hypothesis (pre-registered)

**H1.** Replacing the frozen-CLIP video/text encoder with a frozen-MLLM
(Qwen2.5-VL-7B-Instruct, no LoRA) encoder — **every other component identical**
(topk=20, `lambda_seg=0`, archive OFF, same train/dev/test split, same
lr=1e-4 / epochs=30 / batch=64 / proj=map=1024 / dropout / hard-neg / hybrid-loss /
warmup=5) — yields a **>= +0.030 accuracy AND >= +0.030 macro-F1** improvement
of the RGCL retrieval head on hateful-video detection.

The only manipulated variable is `--model`
(`openai_clip-vit-large-patch14-336_HF` -> `Qwen2.5-VL-7B-Instruct_HF`). Both encoders
are frozen; only their pre-extracted feature `.pt` caches feed the identical RGCL head.

## Design (pre-registered)

- **Datasets:** HateMM and MHC (English MultiHateClip; `dataset=MHC`).
- **Seeds:** 0 / 1 / 2, paired within seed (CLIP vs Qwen at the same seed).
- **Arms:** `frozen-CLIP` (control) vs `frozen-Qwen` (treatment). Both via
  `scripts/slurm/train_archive_baseline.sbatch` (archive-OFF path:
  `archive_feats=None` gates all archive/seg behaviour OFF in `src/run_rac.py`;
  `lambda_seg=0`), so the two arms differ **only** in `--model`.
- **seed0 status (declared):** seed0 for both arms already exists as *old-code* runs
  (`rgcl_*`, `group_name=RAC_video`). This campaign is a **confirmatory seed extension**
  adding s1/s2; it additionally **re-runs seed0 for all four arms under the current
  code** (`train_archive_baseline.sbatch`, `RAC_video_archive_seeds`) so that the
  primary 3-seed table is fully same-code, and keeps the old-code seed0 as a
  reproduction cross-check.
- **MHC-EN Qwen s1/s2 reused:** jobs 12275/12276 (`arcbase_MHC_Qwen..._seed{1,2}`),
  produced by the identical `train_archive_baseline.sbatch` template — bit-for-bit the
  same command as the CLIP arm minus `--model`. Verified against the archive-kNN-seeds
  node ("EN floor / no keys" arm).

### Config-match verification

seed0 old-code `Namespace` diff (CLIP 1035813/1035814 vs Qwen 1029174/1029175) is
**identical except `model=`** — same topk=20, lr=1e-4, ep30, bz64, proj/map=1024,
dropout[0.2,0.4,0.1], hard_negatives, hybrid_loss, `group_name=RAC_video`,
`exp_comment=''`, `force=True`. The current-code arms (s0 re-run + s1/s2) all share the
`train_archive_baseline.sbatch` command (`group_name=RAC_video_archive_seeds`,
`archive_feats=None`, `lambda_seg=0`, warmup=5), again differing only in `--model`.
The archive-OFF/seg-OFF path is documented flag-gated and bit-for-bit reproducible under
src edits (see exp-archive-knn-seeds Addendum 2).

## Protocols (both reported, judged independently — NO protocol selection)

- **(A) val-selected:** pick epoch >= warmup 5 with max Val_Retrieval acc (roc tie-break);
  report that epoch's **Test** macro-F1 / acc / roc.
- **(B) final-epoch:** report **Test** macro-F1 / acc / roc at the last trained epoch (29),
  the standard selection-free protocol.

## Decision rule (pre-registered)

For each dataset x protocol:
1. **Per-seed paired difference** delta = (Qwen - CLIP) for acc and macro-F1 at seeds 0/1/2.
2. **3-seed mean +/- std** of the paired delta; **sign consistency** (how many of 3 seeds positive).
3. n=3 is too small for a formal bootstrap; report the paired-t statistic **as an effect-size
   descriptor only** alongside the mean/std and sign count — no significance claim is made from n=3.
4. **Pass criterion (per dataset x protocol):** mean paired delta_acc >= +0.030 AND
   mean paired delta_mF1 >= +0.030 AND sign consistency 3/3 positive.
5. **Headline claim ("MLLM-as-encoder helps"):** requires the pass criterion met on
   **>= 2 datasets** under a stated protocol. Each protocol is judged separately; if EN
   passes only under final-epoch, the verdict is written exactly as
   "final-epoch: pass; val-selected: fail".

## Reused reference readings (pre-existing, old-code seed0 + current-code EN-Qwen s1/s2)

Parsed with the same val-selection rule as the sbatch (warmup>=5, val-acc, roc tie-break):

| dataset | arm | seed | src | val-sel Test F1 / acc / roc | final-ep Test F1 / acc / roc |
|---|---|---|---|---|---|
| HateMM | CLIP | 0 | 1035814 (old) | 0.8172 / 0.8279 / 0.8903 | 0.7997 / 0.8186 / 0.8857 |
| HateMM | Qwen | 0 | 1029175 (old) | 0.8606 / 0.8698 / 0.9156 | 0.8507 / 0.8605 / 0.9283 |
| MHC-EN | CLIP | 0 | 1035813 (old) | 0.7113 / 0.7826 / 0.8422 | 0.7145 / 0.7640 / 0.8353 |
| MHC-EN | Qwen | 0 | 1029174 (old) | 0.7378 / 0.7888 / 0.8402 | 0.7596 / 0.8012 / 0.8528 |
| MHC-EN | Qwen | 1 | 12275 (cur) | 0.7283 / 0.7826 / 0.8375 | 0.7203 / 0.7702 / 0.8473 |
| MHC-EN | Qwen | 2 | 12276 (cur) | 0.6997 / 0.7702 / 0.8138 | 0.7475 / 0.7826 / 0.8570 |

## Runs to execute (this campaign)

Via `scripts/slurm/enc3seed.sbatch` (one serial sbatch, current code,
`train_archive_baseline.sbatch` command per config):

| # | dataset | model | seed |
|---|---|---|---|
| required 1 | HateMM | CLIP | 1 |
| required 2 | HateMM | CLIP | 2 |
| required 3 | HateMM | Qwen | 1 |
| required 4 | HateMM | Qwen | 2 |
| required 5 | MHC | CLIP | 1 |
| required 6 | MHC | CLIP | 2 |
| confirm 7 | HateMM | CLIP | 0 |
| confirm 8 | HateMM | Qwen | 0 |
| confirm 9 | MHC | CLIP | 0 |
| confirm 10 | MHC | Qwen | 0 |

(MHC-EN Qwen s1/s2 reused from 12275/12276.)

## Results

**PENDING** — filled after the runs complete.
