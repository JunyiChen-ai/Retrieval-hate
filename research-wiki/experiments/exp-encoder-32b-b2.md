---
type: experiment
node_id: exp:exp-encoder-32b-b2
title: "B2 — MLLM-as-encoder SCALE axis: frozen Qwen2.5-VL-32B (5120-d) vs 7B (3584-d) vs frozen-CLIP on all 3 datasets, 3-seed paired, dual protocol, archive OFF (PRE-REGISTRATION, DRAFT-REV1-AWAITING-DELTA-CHECK)"
idea_id: ""
status: DRAFT-REV1-AWAITING-DELTA-CHECK
verdict: draft-unreviewed
confidence: n/a
date: "2026-07-14"
hardware: "Stage-D download = 1x SLURM CPU job (no --gres); Stage-E extraction = 1x A100-80G, existing script AS-IS; Stage-T training = 1x A100 on cached frozen features -> ~20-60 s/run"
duration: "Stage-D ~6 min; Stage-E ~2-2.4 GPU-h/dataset (~6-7 GPU-h all three); Stage-T 9 runs seconds each, one serial sbatch; Stage-C seconds"
provenance: "PRE-REGISTRATION ONLY — NO runs executed, NO download, NO SLURM job, NO test touch. 32B hidden dim = 5120 pinned from HF config.json (Qwen/Qwen2.5-VL-32B-Instruct: text hidden_size=5120, num_hidden_layers=64, torch_dtype=bfloat16; vision out_hidden_size=5120) via WebFetch 2026-07-14. Scale-recon facts cited as 'scale recon 2026-07-14' (given by main). Reference arms REUSED (not re-run): HateMM CLIP & 7B-Qwen (all seeds) + MHC-EN CLIP s0/1/2 & 7B-Qwen s0 from parent enc3s job 12850 (exp-encoder-3seed.md); MHC-EN 7B-Qwen s1/s2 from arcbase jobs 12275/12276 (arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2}_1227{5,6}.trainlog — Rev-1, exactly as the parent reused them, exp-encoder-3seed.md:118,258-259); MHC-ZH CLIP & 7B-Qwen from B1 job 13115 (exp-encoder-zh-b1.md / refine-logs/B1_VERDICT_REVIEW.md). Extraction script: src/utils/generate_VideoMLLM_embedding_HF.py (reuses the 32B-proven bf16/sdpa model-loading path — P10 proved the SCORER on 32B bf16; the extractor's first 32B run is G-repro-gated — Rev-3). Training template: scripts/slurm/train_archive_baseline.sbatch; runners: scripts/slurm/b2_stage_d_download.sbatch + b2_stage_e_extract.sbatch + b2_stage_t_train.sbatch (CONFIGS+QWEN-tag copy of enc3seed.sbatch)."
added: 2026-07-14T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "encoder-scale", "frozen-CLIP", "encoder-swap", "Qwen2.5-VL-32B", "multi-seed", "paired-test", "HateMM", "MHC", "MHC_zh", "pre-registered", "DRAFT-REV1-AWAITING-DELTA-CHECK", "B2"]
---

# B2 — MLLM-as-encoder SCALE axis: frozen Qwen2.5-VL-32B vs 7B vs frozen-CLIP (PRE-REGISTRATION)

> **B2 GOAL-RELEVANT FAIL — 21st negative; SCALE AXIS CLOSED (independent verdict review 2026-07-14, refine-logs/B2_VERDICT_REVIEW.md; job 13146).** 32B-vs-CLIP passes on HateMM only (+0.03/+0.05 both protocols) — restates the banked 7B win while REGRESSING from 7B (final acc CLIP 0.8124 < 32B 0.8450 < 7B 0.8682); MHC-EN and MHC-ZH actively below CLIP (0/3 signs); 32B-vs-7B fails all 3 datasets. Full staged execution (D xet-outage + probe-gated retry, E 1h50m, C weights deleted, T 9m) in refine-logs/B2_EXECUTION_RECORD.md.

> **STATUS: `DRAFT-REV1-AWAITING-DELTA-CHECK` — PRE-REGISTRATION ONLY. NO download, NO
> SLURM job, NO GPU used, NO test touch spent. Reviewed 2026-07-14
> (`refine-logs/B2_PREREG_REVIEW.md`): APPROVED with 4 mandatory revisions + conditional
> execution authorization; Rev-1/2/3/4 applied below (see Revision history). Awaiting
> reviewer delta-check (C2) + explicit user/main go (C3) BEFORE any download or run.**

**verdict:** `draft-unreviewed` — proposal only. · **confidence:** n/a

## Purpose (one line)

Test the **last untried structural lever** in the encoder campaign — **encoder SCALE**.
The project's single banked positive is *swapping frozen-CLIP for frozen Qwen2.5-VL-**7B**
hidden states in the otherwise-unchanged RGCL head*, which PASSES on HateMM (+5 acc / +6 F1,
3/3 seeds) but FAILS on MHC-EN and MHC-ZH. B2 replaces the **7B** encoder with the **32B**
encoder (frozen, no LoRA) — Qwen2.5-VL-32B-Instruct hidden states, **5120-d vs the 7B's
3584-d** — and asks whether the extra representational capacity converts to acc/F1 on the
two **goal-gap** datasets (MHC-EN, MHC-ZH) that 7B could not move, while re-checking the
HateMM anchor. This is the only axis of the only working lever that has never been tried.

## Where B2 sits in the campaign (recon)

- **20 negatives are closed** (`MEMORY.md`; the encoder-swap campaign is settled at 7B).
  Encoder-swap outcome to date: HateMM **PASS** both protocols (`exp-encoder-3seed.md`),
  MHC-EN **FAIL** both (same node), MHC-ZH **FAIL** both (B1, `refine-logs/B1_VERDICT_REVIEW.md`,
  20th negative). HateMM is the sole dataset where the frozen swap passes.
- The two failing datasets were each **diagnosed**: MHC-EN as **data/label-limited** (SAV
  review; `exp-encoder-3seed.md:231-234`), MHC-ZH as **decision-layer non-conversion** —
  the frozen Qwen arm carries *higher* test ROC/AUC than CLIP on every seed (0.88-0.90 vs
  0.83-0.84) that **never converts to thresholded acc/F1** (`B1_VERDICT_REVIEW.md:159-163`),
  and the ZH 0.8537 that crossed 0.85 came from **LoRA-SFT of the encoder, a different
  lever**, not frozen hidden states.
- **Scale is the one axis untried on the one working lever.** Every other manipulation
  (archive, seg, consensus, rerank, pooling, twins, localization-as-training, LoRA, target
  structure) is closed. The honest campaign question B2 answers: *does more encoder capacity,
  frozen, convert the diagnosed gaps — or does it merely re-confirm the 7B verdict at a
  larger dim?*

## Hypothesis (pre-registered)

**H1.** Replacing the frozen Qwen2.5-VL-**7B** encoder with the frozen Qwen2.5-VL-**32B**
encoder — hidden states **5120-d (32B) vs 3584-d (7B)**; **every other component identical**
(topk=20, `lambda_seg=0`, archive OFF, same split, lr=1e-4 / epochs=30 / batch=64 /
proj=map=1024 / dropout[0.2,0.4,0.1] / hard-neg / hybrid-loss / warmup=5) — adds
representation information that yields a **>= +0.030 accuracy AND >= +0.030 macro-F1**
improvement of the RGCL retrieval head, on **3/3 seeds**, **vs the frozen-CLIP control**
(the primary comparison), on at least one goal-gap dataset.

The manipulated variable across the treatment arms is `--model`
(`Qwen2.5-VL-7B-Instruct_HF` -> `Qwen2.5-VL-32B-Instruct_HF`); vs the control it is CLIP ->
32B-Qwen. All encoders are frozen; only their pre-extracted `.pt` feature caches feed the
identical RGCL head.

**32B hidden dim = 5120 (pinned).** Qwen2.5-VL-32B-Instruct `config.json` (WebFetch
2026-07-14): text-side `hidden_size=5120`, `num_hidden_layers=64`, `torch_dtype=bfloat16`;
vision-side `hidden_size=1280` with `out_hidden_size=5120` (vision features are projected
into the 5120-d LM space). The extraction script pools `out.hidden_states[-1]` and sizes the
output as `d = model.config.hidden_size`
(`src/utils/generate_VideoMLLM_embedding_HF.py:332`), so **both `img_feats` and `text_feats`
will be 5120-d** for 32B — exactly the mechanism that produced 3584-d for 7B and needs no
code change (the comment `# 3584 for Qwen2.5-VL-7B` is a label of the 7B value, not a
hard-code).

### Sub-hypotheses (honest priors declared per dataset)

- **H1a — MHC-EN (prior: LOW).** MHC-EN is diagnosed **data/label-limited**; if the ceiling
  is annotation quality, not encoder capacity, scale cannot cross it. A pass here would be
  surprising and would revise the SAV diagnosis.
- **H1b — MHC-ZH (prior: LOW-MEDIUM).** 7B was **flat on acc/F1 but ROC>CLIP unconverted**
  (`B1_VERDICT_REVIEW.md:159-163`). The one non-negative reading is that the ranking signal
  already exists and only the decision layer fails to convert it; more capacity *might*
  sharpen the margin enough to convert. Countervailing: B1 showed the ZH gain that mattered
  was **LoRA**, not frozen hidden states — so frozen scale may just widen the same
  unconverted ROC gap.
- **H1c — HateMM anchor (prior: MEDIUM).** The 7B effect is real and large (+5 acc / +6 F1,
  3/3 seeds, both protocols — the project's most robust positive). Does it **hold or extend**
  at 32B? A HateMM improvement is the anchor that scale *is* adding usable information, but
  **is NOT itself a goal pass** (already banked at 7B — see Decision rule).

## Design (pre-registered)

- **Datasets (all three):** HateMM (`dataset=HateMM`), MHC-EN (`dataset=MHC`), MHC-ZH
  (`dataset=MHC_zh`). Row counts (gt splits, verified 2026-07-14): HateMM 744/107/215,
  MHC 549/80/161, MHC_zh 579/78/149 (train/val/test).
- **Seeds:** 0 / 1 / 2, paired within seed.
- **Arms.** Per dataset, the **32B arm (3 seeds, treatment)** is compared paired-within-seed
  against the SAME-RUNNER CLIP arm (primary) and the SAME-RUNNER 7B arm (secondary):
  - **Primary comparison — 32B vs frozen-CLIP** (the control that defines a goal pass).
  - **Secondary comparison — 32B vs frozen-7B-Qwen** (isolates the pure **scale increment**
    on top of the banked 7B lever), judged under the identical rule and reported alongside.
- **Reference arms are REUSED, not re-run** (only the 32B arm executes):
  - **HateMM:** CLIP and 7B-Qwen arms (all 3 seeds each) come from the **parent enc3s job
    12850** (`exp-encoder-3seed.md:148-159`, `RESULT_ROW`s in
    `slurm/logs/enc3s_HateMM_*_12850.trainlog`).
  - **MHC-EN (Rev-1, provenance-precise):** CLIP s0/1/2 and **7B-Qwen s0** come from job
    12850 (`enc3s_MHC_*_12850.trainlog`); **7B-Qwen s1/s2 are NOT in 12850** — they are the
    reused **arcbase jobs 12275/12276**
    (`arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2}_1227{5,6}.trainlog`), exactly as the
    parent documents and reused them (`exp-encoder-3seed.md:118,258-259`). Still same-runner:
    12275/12276 were produced by the identical `train_archive_baseline.sbatch` command under
    current code (parent's Namespace audit, `exp-encoder-3seed.md:141,145-146`).
  - Same-runner check: the 32B arm will run through the identical `enc3seed`-family runner
    (`scripts/slurm/enc3seed.sbatch`, which IS the `train_archive_baseline.sbatch` python
    command per config), so CLIP/7B references and 32B (B2) share the exact runner and
    command modulo `--model`. **Verify at Stage-T verdict processing** that each reference
    arm's `Namespace` differs from the 32B arm only in `model=`/`exp_comment=`/`output_path=`
    — **cross-check targets: 12850 (HateMM CLIP/7B + MHC-EN CLIP + MHC-EN 7B s0),
    12275/12276 (MHC-EN 7B s1/s2), 13115 (MHC-ZH CLIP/7B)**.
  - **MHC-ZH:** CLIP and 7B-Qwen arms come from **B1 job 13115** (`exp-encoder-zh-b1.md`;
    `B1_VERDICT_REVIEW.md` Task 1, all 12 readings re-verified to 4dp), produced by
    `scripts/slurm/enc3seed_zh_b1.sbatch` (a CONFIGS-only copy of the same runner). Same
    runner family; **verify same-runner at delta-check** exactly as above.
- **Manipulated variable only `--model`.** Both treatment and reference arms use the
  archive-OFF path of `train_archive_baseline.sbatch` (`archive_feats=None` gates all
  archive/seg behaviour OFF in `src/run_rac.py`; `lambda_seg=0`), so arms differ **only** in
  `--model` — identical discipline to `exp-encoder-3seed.md:40-43` and B1.

### Reference readings reused (verbatim from the parent + B1; NOT re-derived here)

Transcribed for the reader; the primary source of truth remains the cited nodes/logs.

**HateMM** (parent `exp-encoder-3seed.md:150-159`, job 12850) — per-seed Test F1 / acc:

| seed | CLIP val-sel | CLIP final | 7B val-sel | 7B final |
|---|---|---|---|---|
| 0 | 0.8172 / 0.8279 | 0.7997 / 0.8186 | 0.8606 / 0.8698 | 0.8507 / 0.8605 |
| 1 | 0.8163 / 0.8279 | 0.7822 / 0.8047 | 0.8586 / 0.8651 | 0.8514 / 0.8605 |
| 2 | 0.7920 / 0.8047 | 0.7988 / 0.8140 | 0.8753 / 0.8837 | 0.8753 / 0.8837 |

**MHC-EN** (parent `exp-encoder-3seed.md:163-170`; CLIP s0/1/2 + 7B s0 = job 12850,
**7B s1/s2 = arcbase jobs 12275/12276** per Rev-1) — per-seed Test F1 / acc:

| seed | CLIP val-sel | CLIP final | 7B val-sel | 7B final |
|---|---|---|---|---|
| 0 | 0.7113 / 0.7826 | 0.7145 / 0.7640 | 0.7378 / 0.7888 | 0.7596 / 0.8012 |
| 1 | 0.6034 / 0.7329 | 0.7159 / 0.7826 | 0.7283 / 0.7826 | 0.7203 / 0.7702 |
| 2 | 0.6997 / 0.7702 | 0.7303 / 0.7888 | 0.6997 / 0.7702 | 0.7475 / 0.7826 |

**MHC-ZH** (B1 `B1_VERDICT_REVIEW.md:29-40`, job 13115) — per-seed Test F1 / acc:

| seed | CLIP val-sel | CLIP final | 7B val-sel | 7B final |
|---|---|---|---|---|
| 0 | 0.7706 / 0.8054 | 0.7706 / 0.8054 | 0.7412 / 0.7919 | 0.7864 / 0.8188 |
| 1 | 0.7579 / 0.8054 | 0.7542 / 0.8054 | 0.7871 / 0.8121 | 0.7759 / 0.8054 |
| 2 | 0.7742 / 0.8121 | 0.7913 / 0.8322 | 0.7759 / 0.8054 | 0.7514 / 0.7852 |

### Asset check (what exists, what must be produced)

- **32B weights — NOT cached; Stage-D download REQUIRED.** `~/.cache/huggingface/hub/`
  holds `models--Qwen--Qwen2.5-VL-7B-Instruct` (+ Qwen3-VL-8B / -235B) but **no
  `models--Qwen--Qwen2.5-VL-32B-Instruct`** (verified 2026-07-14). P10 downloaded and ran
  32B previously (`slurm/logs/dl_qwen25vl_32b.log` = "Fetching 32 files" complete in ~6:05;
  `scripts/slurm/p10_score_ladder.sbatch` ran 32B bf16 on 1×A100-80G), then it was deleted
  under the transient-disk lifecycle. So the weights must be re-downloaded (Stage-D).
- **32B embedding caches — do NOT exist; Stage-E extraction REQUIRED.**
  `find data/CLIP_Embedding -iname "*32B*"` returns nothing. Stage-E produces
  `data/CLIP_Embedding/<ds>/{train,dev_seen,test_seen}_Qwen2.5-VL-32B-Instruct_HF.pt`
  (5120-d), the only new training input.
- **Raw videos — PRESENT (asset check that corrected a false alarm; document so the
  reviewer does not repeat it).** `data/video/<ds>/All/*.mp4` are **symlinks** to raw stores,
  NOT files, so a naive `find -type f` reports zero. Following symlinks
  (`find -L … -type f`): **HateMM 1066/1066, MHC 790/790, MHC_zh 806/806 resolve to real,
  readable mp4** (2026-07-14). Targets:
  `/data/jehc223/HateMM/video/` (1083 mp4, 6.2G) and
  `/data/jehc223/Multihateclip/{English/video_mp4, Chinese/video}/` (27G total). These raw
  stores are **disk_guard-protected** (`_DG_RAW_DATASET_SUBSTR` = `HateMM_raw`,
  `MultiHateClip`, …; `scripts/disk_guard.sh:93-98`) and were never pruned. **No
  video-restore stage is needed.** (Note: the B2 base B2 prefix `b2:junyi-data/RGCL_video`
  has **no** `video/` dir, so `b2_pull.sh` could NOT restore videos — fortunate that they are
  present locally.)
- **What is missing to run B2: the 32B weights (transient download) + the 32B feature
  caches (extraction) + three tiny runner edits.** No new src code, no config change to
  `run_rac.py`/`dataset.py`. The loader builds `{path}/{dataset}/{split}_{model}.pt` and the
  head's input dim is **inferred from the loaded `.pt` tensor shape — CODE-VERIFIED**
  (review §4a, `refine-logs/B2_PREREG_REVIEW.md`): `run_rac.py:1102-1103` reads
  `["image_feats"].shape[1]` / `["text_feats"].shape[1]` from the first train batch →
  passed to `classifier_hateClipper` at `:1117-1120` → sizes only the two first
  `nn.Linear` projections (`classifier.py:76-77`); no hard-coded 3584/5120 anywhere on the
  path. 5120-d auto-wires; **G-repro remains a cache-integrity + first-32B-run sanity
  check**, not a dim-wiring test.

## Config-match verification (to run FIRST, gates everything)

Mirror `exp-encoder-3seed.md:126-146` and B1's gate structure, but note the **key
difference: no historical 32B reference exists**, so the seed0 reproduction gate that the
parent/B1 used (new-code s0 must reproduce an old-code s0 to 4dp) **cannot apply** — there is
nothing to reproduce against. B2's repro gate is therefore **sanity-only** (G-repro below).

- **Namespace-diff gate (HARD).** The 32B arm's `Namespace` must differ from the reused
  CLIP/7B reference arms **only** in `model=`/`exp_comment=`/`output_path=`. Reference-log
  targets (Rev-1): **12850** (HateMM CLIP/7B all seeds; MHC-EN CLIP s0/1/2; MHC-EN 7B s0),
  **12275/12276** (MHC-EN 7B s1/s2 arcbase logs), **13115** (MHC-ZH CLIP/7B all seeds).
  Verify at Stage-T verdict processing by parsing raw line 1 of each 32B trainlog against
  the reference trainlogs. HALT on any other substantive divergence.
- **G-dims gate (HARD).** Every extracted 32B `.pt` must have `img_feats`/`text_feats` of
  width **5120** (== `config.hidden_size`) and row counts == the split sizes above
  (744/107/215, 549/80/161, 579/78/149). The per-arm id lists must equal the CLIP/7B arms'
  id lists (same videos, same order) — the paired-arm id audit B1 ran
  (`B1_IMPL_NOTES.md` §a.1). HALT on dim != 5120 or any row-count/id mismatch.

## Protocols (both reported, judged independently — NO protocol selection)

Transcribed verbatim from `exp-encoder-3seed.md:66-71`:

- **(A) val-selected:** pick epoch >= warmup 5 with max Val_Retrieval acc (roc tie-break);
  report that epoch's **Test** macro-F1 / acc / roc.
- **(B) final-epoch:** report **Test** macro-F1 / acc / roc at the last trained epoch (29),
  the standard selection-free protocol.

Both protocols are judged independently under the identical rule below; the write-up format
is fixed ("final-epoch: pass/fail; val-selected: pass/fail"). No protocol is designated
"primary" for B2 — unlike ZH-only B1, B2 spans all three datasets and HateMM/EN carry no
78-dev val-selection tax, so there is no encoder-independent reason to lead with one
protocol. (For the ZH cell specifically, the documented ~2-acc-pt 78-dev val-selection tax
still makes final-epoch the less noisy lens, but that is a reading note, not a decision gate.)

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

**Application to B2.**

- **Primary comparison = 32B vs frozen-CLIP** — this is the comparison the goal is scored on.
  For each dataset × protocol, paired Δ = (32B − CLIP) on acc AND macro-F1; PASS iff mean
  Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND 3/3 seeds sign-positive.
- **Secondary comparison = 32B vs frozen-7B-Qwen** — same rule, paired Δ = (32B − 7B),
  reported alongside to isolate the **pure scale increment**. This is a diagnostic, not the
  goal gate (the goal is defined against CLIP), but it answers "did scale add anything on top
  of the banked 7B lever, or is 32B ≈ 7B?"
- **Goal-relevant success (pre-declared): a PASS of the 32B-vs-CLIP comparison on ANY of
  MHC-EN or MHC-ZH** — that supplies the second dataset the parent's rule (5) needs (HateMM
  is already banked), completing "MLLM-as-encoder helps on ≥ 2 datasets" under that protocol,
  i.e. the goal's "+0.03 acc AND +0.03 F1 on ≥ 2 datasets".
- **HateMM-only improvement is NOT a goal pass** — a 32B-vs-CLIP HateMM pass merely restates
  what 7B already banked; it does not add a second goal dataset. (A 32B-vs-7B HateMM
  *increment* would be interesting for the scale story but is still not a goal pass.)
- A FAIL on both MHC-EN and MHC-ZH (32B-vs-CLIP) leaves HateMM the sole passing encoder
  dataset (status quo, `MEMORY.md`) and closes the encoder-scale axis as the 21st negative.

## Execution plan (each stage single-submit; pending authorization)

### Stage-D — download 32B weights (1× SLURM CPU job, no GPU)

- SLURM spec: `--partition=slurmpartition`, **no `--gres`** (CPU-only), `--cpus-per-task=8`,
  `--mem=32G`, **no `--time`** (project rule; expect `PENDING (JobHeldUser)` → auto-release).
- Env: **`HF_HUB_OFFLINE=0`** (must reach the hub), `PYTHONUNBUFFERED=1`.
- Command: `hf download Qwen/Qwen2.5-VL-32B-Instruct` (or the equivalent
  `huggingface-cli download` / `snapshot_download`) into the default HF cache
  (`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct`). ~66G bf16 weights
  (~18 safetensors shards among ~32 total repo files, per `dl_qwen25vl_32b.log`), ~6 min at
  ~170 MB/s (scale recon 2026-07-14; matches the prior P10 download log).
- **Do NOT run `disk_guard.sh` with `DISK_GUARD_HF_PURGE=1` while the weights are staged** —
  disk_guard never removes `models--*` (verified `disk_guard.sh:379-380`), so the default is
  safe, but this is called out explicitly.

### Stage-E — extract 32B features (1× A100-80G, existing script AS-IS)

- Runner: `scripts/slurm/b2_stage_e_extract.sbatch`, modelled on
  `scripts/slurm/gen_embed_mllm.sbatch` (which calls
  `src/utils/generate_VideoMLLM_embedding_HF.py` AS-IS), with TWO additions vs the 7B version:
  1. **(Rev-2 — HARD diff-verify item, checked before submit)** pass **BOTH**
     `--model Qwen/Qwen2.5-VL-32B-Instruct` **AND**
     `--out_model_tag Qwen2.5-VL-32B-Instruct_HF`. **Burn risk stated explicitly:** the base
     template passes *neither* flag (`gen_embed_mllm.sbatch` calls the script with only
     `--dataset/--num_frames/--device`), and the script defaults are the **7B** values
     (`--model=Qwen/Qwen2.5-VL-7B-Instruct`, `--out_model_tag=Qwen2.5-VL-7B-Instruct_HF`;
     `generate_VideoMLLM_embedding_HF.py:81,87`). If `--out_model_tag` is omitted, the 32B
     output is written to the **7B cache filename** (`{split}_Qwen2.5-VL-7B-Instruct_HF.pt`)
     — a **silent overwrite of the existing 7B caches**, after which Stage-T would load 32B
     features under a 7B label (and any 7B rerun would load 32B features). The filename
     collision *precedes* the G-dims backstop, so the presence of both flags is a
     **mandatory pre-submit diff-verify item** recorded in `refine-logs/B2_IMPL_NOTES.md`.
  2. export **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (proven necessary for 32B
     bf16 on 80G by the P10 *scorer* runs; `p10_score_ladder.sbatch:22-24`). Keep
     `HF_HUB_OFFLINE=1` (weights now local from Stage-D); bf16 + sdpa + `device_map=None` +
     `model.to(device)` are already the script defaults
     (`generate_VideoMLLM_embedding_HF.py:396-402`).
- **(Rev-3, provenance precision)** The extractor `generate_VideoMLLM_embedding_HF.py` has
  itself only ever run **7B**; what P10 proved on 32B bf16/80G is the *scorer*
  (`p10_score_segments.py` via `p10_score_ladder.sbatch`). The extractor **reuses the same
  32B-proven `Qwen2_5_VLForConditionalGeneration.from_pretrained` bf16/sdpa loading path**,
  and its 8-frame `max_pixels=360*420` forward is *lighter* than P10's K30/M120 windows —
  but its **first 32B run is G-repro-gated**, not "P10-proven".
- Order (per-dataset resumable — each dataset writes its own three `.pt`):
  **HateMM → MHC → MHC_zh.** (HateMM first because it is the anchor whose s0 training run
  later serves as the G-repro sanity readout — Stage-T config #1 per Rev-4; within Stage-E
  itself the per-dataset verification is G-dims on the written `.pt` files.)
- Each video costs **2 frozen forward passes** (img "prefix" span + text "response" span;
  `generate_VideoMLLM_embedding_HF.py:345-359`), 8 frames at `max_pixels=360*420` — a small
  vision sequence that fits comfortably (unlike P10's K30/M120 windows that OOM'd 32B).
- Output: 5120-d caches; each `.pt` is tens of MB (7B HateMM train was 21 MB at 3584-d →
  32B ≈ 30 MB at 5120-d). Embeddings are retained; weights are deleted in Stage-C.

### Stage-T — train the 32B arm (1× A100, cached features, seconds/run)

- Runner: **`scripts/slurm/b2_stage_t_train.sbatch`** (a CONFIGS+`QWEN`-tag copy of
  `scripts/slurm/enc3seed.sbatch`, exactly as B1 authored `enc3seed_zh_b1.sbatch`; the python
  command is byte-identical modulo the config list and the `QWEN=` model-tag line). CONFIGS =
  **9 runs = 3 datasets × 3 seeds, 32B arm only** (CLIP/7B arms reused from 12850, 12275/12276
  and 13115 — Rev-1). **Config #1 = HateMM s0 = the G-repro sanity readout (Rev-4).**

  | # | dataset | model | seed |
  |---|---|---|---|
  | 1-3 | HateMM | Qwen2.5-VL-32B-Instruct_HF | 0/1/2 (config #1 = s0 = G-repro readout) |
  | 4-6 | MHC | Qwen2.5-VL-32B-Instruct_HF | 0/1/2 |
  | 7-9 | MHC_zh | Qwen2.5-VL-32B-Instruct_HF | 0/1/2 |

- Each run = the exact `train_archive_baseline.sbatch` python command differing only in
  `--dataset` / `--model Qwen2.5-VL-32B-Instruct_HF` / `--seed`; `--force False`;
  `GROUP_NAME=RAC_video_archive_seeds`. Cached 5120-d features → ~20-60 s/run (parent-runtime
  band, `exp-encoder-zh-b1.md:333-336`). One serial sbatch, no mid-run resubmission.
- **Log-collision check (do at submit time):** derived logs
  `slurm/logs/enc3s_<ds>_Qwen2.5-VL-32B-Instruct_HF_seed{0,1,2}_<JID>.trainlog` and output
  dirs `logging/Retrieval/<ds>/RAC_video_archive_seeds/..._Qwen2.5-VL-32B-Instruct_HF` are
  new tags — no collision with the 7B/CLIP dirs; the fresh `$SLURM_JOB_ID` double-protects
  (mirror `B1_IMPL_NOTES.md` §a.2).

### Stage-C — cleanup (after extraction verified)

- After Stage-E is verified (G-dims PASS on all 9 caches: dim 5120, correct row counts, id
  match), **DELETE the ~66G weights**
  (`rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct`), documented in
  the execution record. **Embeddings are retained** (they are the training input and are only
  MBs). Stage-T reads only the cached `.pt` (`HF_HUB_OFFLINE=1`, no weights needed), so
  training is unaffected by the deletion.

## Gates (pre-registered)

- **G-repro (sanity-only — no historical 32B reference exists).** Because 32B has
  never run this pipeline, there is nothing to reproduce bit-for-bit; the gate is a
  **sanity** check: the HateMM 32B s0 run must (i) load the 5120-d caches without dim/wiring
  errors, (ii) train 30 epochs and print well-formed `Val_/Test_Retrieval` lines, (iii)
  produce Test acc/F1 in a plausible band (not 0.5-degenerate, not NaN). This is explicitly
  a **first-run** gate — its role is to catch a broken 5120-d wiring or a mis-extracted
  cache, NOT to match a prior number. State this in the record so no reviewer mistakes the
  absence of a repro-match for a skipped gate.
- **G-repro execution mode (Rev-4, pinned).** The HateMM-s0 G-repro sanity is the
  **FIRST-CONFIG READOUT of the Stage-T serial job** — config #1 of the 9-run
  `b2_stage_t_train.sbatch` is `HateMM / Qwen2.5-VL-32B-Instruct_HF / seed 0`, and the
  runner prints its per-run val-sel/final readout serially before the later runs, so the s0
  readout physically precedes everything downstream in the job log. There is **NO separate
  smoke submit** (preserves the single-submit-per-stage discipline) and **NO mid-job
  intervention**: the gate is **applied at verdict processing** — the s0 readout is
  inspected FIRST when reading back the Stage-T log; if it fails the sanity criteria, HALT
  tabulation per the kill rules (the remaining 8 runs' outputs exist but are not tabulated).
- **G-dims (HARD).** As above: dim == 5120, row counts == splits, id lists == CLIP/7B arms.
- **Namespace-diff (HARD).** 32B arm vs reused reference arms differ only in
  `model=`/`exp_comment=`/`output_path=`.

## Kill rules (pre-registered)

1. **Extraction OOM / failure → evidence, no blind resubmit.** If Stage-E OOMs or crashes on
   any dataset, capture the traceback + `nvidia-smi`, record it, and **do not resubmit with
   tweaked knobs** under this pre-registration. (The P10 *scorer* proved 32B bf16 +
   `expandable_segments` runs on 80G with far heavier vision sequences than our 8-frame
   `max_pixels=360*420` inputs, so an OOM would itself be a finding worth a separate
   diagnosis, not a silent retry — Rev-3 wording.)
2. **Namespace-diff gate.** Any substantive `Namespace` field other than
   `model=`/`exp_comment=`/`output_path=` differing between the 32B arm and its reused
   reference arms → HALT, do not tabulate.
3. **G-dims failure.** dim != 5120, wrong row count, or id-list mismatch on any cache → HALT
   that dataset (extraction is wrong); other datasets may continue.
4. **Per-sub-hypothesis independence.** If the 32B arm fails on one dataset, that
   sub-hypothesis (H1a/H1b/H1c) dies; the others continue. A dataset-specific FAIL is
   reported as such (fixed format "final-epoch: pass/fail; val-selected: pass/fail"), exactly
   as the parent handled MHC-EN and B1 handled MHC-ZH.
5. **No protocol-shopping / no metric-shopping.** Both protocols reported regardless of which
   passes; both Δacc AND ΔmF1 must clear +0.03 with 3/3 sign for a PASS; an acc-only or
   F1-only move is reported as FAIL-with-direction.
6. **Test-touch budget spent → no re-runs with tweaked knobs** on any dataset's test set
   under this pre-registration.

## Test-touch discipline

- **Precedent = the parent's protocol** (`exp-encoder-3seed.md:120-124,261-263`): the
  enc3seed runner **reads Test each epoch** (`Test_Retrieval Epoch NN …` printed every epoch)
  and selects/reports via the val-sel (epoch ≥ warmup, val-acc, roc tie-break) and
  final-epoch rules from the per-epoch Test readouts. B2 follows this **exactly** — it is the
  parent's established protocol, and the test-touch accounting follows the parent's
  precedent: the parent counted the whole 3-seed dual-protocol comparison as its pre-declared
  evaluation of the encoder question, not as one-touch-per-epoch. (This differs from the
  strict "one held-out touch" discipline used elsewhere in the project; B2 inherits the
  encoder campaign's precedent rather than inventing a new accounting.)
- **Novelty of the B2 question.** The specific pre-registered question — "does the
  frozen-CLIP → frozen-32B-Qwen swap yield +0.03 on each dataset under the archive-OFF RGCL
  head at 3 seeds" — has **never been evaluated** (no 32B encoder-swap logs exist on any
  dataset). It is allotted exactly this one 9-run evaluation; no adaptive re-running against
  any test set.

## Disk lifecycle + quota-grace risk

- **Transient footprint = the ~66G weights only.** Lifecycle: Stage-D download (~6 min) →
  Stage-E extract (~6-7 GPU-h across three datasets) → Stage-C DELETE. Total time the 66G is
  on disk ≈ the extraction wall-clock, **~a few hours**.
- **Quota state (2026-07-14):** 382G used / **290G soft** (currently in **grace, ~23h43m
  remaining**) / **3000G hard** / **409G filesystem free**. Adding 66G → ~448G used: **fits
  the filesystem free space** (409G > 66G, with headroom because embeddings out are only MBs
  and Stage-C reclaims the 66G), stays **far under the 3000G hard cap**, and the **grace
  window (~24h) comfortably exceeds the ~a-few-hours transient lifecycle** — the soft-quota
  grace will not expire mid-job.
- **disk_guard is effectively blind to the transient weights — no auto-deletion risk
  mid-job.** `scripts/disk_guard.sh` **never removes `models--*`** (hard guard at lines
  379-380: "SKIP (models--* protected)"; only `datasets--*`/`.locks` are ever purgeable, and
  only under the non-default `DISK_GUARD_HF_PURGE=1`). Its reclaim allowlist is
  `logging`, `data/CLIP_Embedding`, rclone-cache, huggingface-cache-datasets only
  (`disk_guard.sh:84-88`); it does not touch `data/video`, the raw stores, or model weights.
  So even if disk_guard fires during Stage-E, it cannot delete the 32B weights out from under
  the running extraction. **The one caveat:** disk_guard *can* prune `data/CLIP_Embedding` if
  the threshold trips — so after Stage-E, the fresh 32B `.pt` should be B2-pushed (the
  extraction runner already ends with a `b2_push` of `data/CLIP_Embedding/<ds>`) so the
  training inputs are recoverable if reclaimed.

## Honest prior / expected outcome (declared before running)

- **20 negatives close the encoder campaign at 7B; B2 is the last structural lever, and the
  base rate says it most likely re-confirms the 7B verdict at a larger dim.** The most
  probable outcome is **HateMM PASS (32B-vs-CLIP), MHC-EN + MHC-ZH FAIL** — i.e. scale
  extends the working lever but does not convert the diagnosed gaps, closing encoder-scale as
  the 21st negative.
- **Why scale might NOT help the gap datasets:**
  - **MHC-EN** is diagnosed **data/label-limited** (SAV); if the ceiling is annotation
    quality, more encoder capacity cannot cross it. The 7B swap already failed both protocols.
  - **MHC-ZH** is a **decision-layer non-conversion**: 7B already produced *higher ROC than
    CLIP* on every seed that never converted to acc/F1, and B1 showed the ZH accuracy gain
    that mattered was **LoRA-driven, not frozen-hidden-state-driven**. Frozen scale may just
    widen the same unconverted ROC gap.
- **The one reason it might help:** the **HateMM 7B effect is real and large** (+5 acc / +6
  F1, 3/3 seeds, both protocols — the project's most robust positive), which is direct
  evidence that frozen Qwen hidden states carry usable hateful-video signal that CLIP lacks;
  **scale is the only untried axis of the only lever that has ever worked.** If more capacity
  sharpens the MHC-ZH ranking margin enough to convert (H1b), that would be the cleanest
  available second-dataset goal pass. Registered as a genuine open question with a LOW-to-LOW-
  MEDIUM prior on the gap datasets and MEDIUM on the HateMM anchor.

## GPU budget & cost estimate

- **Stage-D (download):** 0 GPU-h; 1 CPU job, **~6 min wall** (per `dl_qwen25vl_32b.log`).
- **Stage-E (extraction):** at **7-9 s/video** (scale recon 2026-07-14) × 2 forward passes
  already folded into that per-video figure, applied to the measured video counts
  (HateMM 1066, MHC 790, MHC_zh 806): **~2.1-2.7 GPU-h HateMM, ~1.5-2.0 GPU-h MHC, ~1.6-2.0
  GPU-h MHC_zh → ~5-7 GPU-h total** across the three datasets, 1× A100-80G.
  *(Open question / reconciliation: the scale recon's "~1-1.5 GPU-h/dataset" figure is below
  this count-based estimate — it appears to assume ~600-700 videos/dataset, whereas the
  actual all-splits counts are 790-1066. The honest budget is the count-based ~5-7 GPU-h
  total; the true number depends on 32B's realized s/video, which the HateMM extraction will
  measure first.)*
- **Stage-T (training):** 9 runs × ~20-60 s on cached features = **~3-9 min compute**, one
  serial sbatch, 1× A100.
- **Stage-C:** seconds.
- **Total: ~5-7 GPU-h (extraction-dominated) + ~6 min CPU download + <10 min training.**
  Wall-clock, allowing for `JobHeldUser` auto-release latency between the three single-submit
  stages, is dominated by extraction (~half a day of A100 time if serial, less if HateMM's
  measured rate is faster). Fits the per-user cap (16 CPU / 128 GB / 2 GPU); each GPU stage
  requests 1 GPU. Submit every stage with **no `--time`**.

## Single-submit ceremony (pre-registered)

1. Freeze this pre-registration (review DONE → delta-check → conditional authorization).
2. Author the three runners: `b2_stage_d_download.sbatch`, `b2_stage_e_extract.sbatch`
   (`gen_embed_mllm.sbatch` + BOTH `--model`/`--out_model_tag` (Rev-2 hard diff-verify) +
   `expandable_segments`), `b2_stage_t_train.sbatch` (CONFIGS+`QWEN`-tag copy). Diff-verify
   each is a minimal delta; record sha256 (`refine-logs/B2_IMPL_NOTES.md`).
3. **Stage-D** single submit → verify shard count / cache present; record `df` before/after.
4. **Stage-E** single submit (HateMM → MHC → MHC_zh) → run **G-dims** on all 9 caches
   (dim 5120, row counts, paired-id audit). (G-repro is NOT here — Rev-4.)
5. **Stage-C** delete weights only after G-dims PASS, with `df` before/after evidence
   (embeddings retained + B2-pushed).
6. **Stage-T** one serial sbatch (9 runs; config #1 = HateMM s0). At verdict processing:
   inspect the **HateMM-s0 G-repro sanity readout FIRST** (Rev-4), then run the
   Namespace-diff gate against **12850 / 12275-12276 / 13115** (Rev-1), then read back every
   number from the raw `enc3s_<ds>_Qwen2.5-VL-32B-Instruct_HF_*` trainlogs (line-numbered
   provenance), then tabulate per-seed deltas (32B-vs-CLIP primary, 32B-vs-7B secondary) and
   apply the decision rule verbatim.

## Readiness verdict (what remains before this can be submitted)

1. **Fresh pre-registration review** — DONE 2026-07-14: **APPROVED with 4 mandatory
   revisions + conditional execution authorization** (`refine-logs/B2_PREREG_REVIEW.md`);
   Rev-1/2/3/4 applied in this revision (C0).
2. **Runners authored (C1)** — `scripts/slurm/b2_stage_d_download.sbatch`,
   `b2_stage_e_extract.sbatch`, `b2_stage_t_train.sbatch`; diffs, Rev-2 hard diff-verify,
   cache-collision check and sha256 hashes in `refine-logs/B2_IMPL_NOTES.md`.
3. **Reviewer delta-check (C2)** of C0+C1 — PENDING.
4. **Conditional authorization (C3)** — explicit user/main go — PENDING (GPUs shared with
   the user's own loop; `CLAUDE.md` — every GPU task via SLURM, subagents do the work).
5. **Single submits** — Stage-D, then Stage-E (+ G-dims + Stage-C), then Stage-T.

**Blocked only on data availability? NO** — the one hard prerequisite is the transient 32B
download (Stage-D), which is a ~6-min CPU job; videos and pipeline are all present. The gates
are review + delta-check + authorization.

## Connections
- extends -> `exp:exp-encoder-3seed` (HateMM PASS / MHC-EN FAIL 7B encoder test; B2 adds the SCALE axis)
- extends -> `exp:exp-encoder-zh-b1` (MHC-ZH 7B FAIL; B2 re-tests ZH at 32B)
- controls-against -> parent enc3s job 12850 (HateMM CLIP+7B all seeds; MHC-EN CLIP s0/1/2 + 7B s0), arcbase jobs 12275/12276 (MHC-EN 7B s1/s2 — Rev-1) and B1 job 13115 (MHC-ZH CLIP + 7B arms)
- contrasts-with -> `exp:exp-lora-sft-encoder` (LoRA-Qwen is the ZH lever that reached 0.8537; B2 isolates FROZEN 32B, no LoRA)
- uses -> `src/utils/generate_VideoMLLM_embedding_HF.py` (reuses the 32B-proven bf16/sdpa loading path from the P10 scorer; first 32B extractor run is G-repro-gated — Rev-3; 5120-d output via `config.hidden_size`)
- reviewed-by -> `refine-logs/B2_PREREG_REVIEW.md` (2026-07-14, APPROVED + Rev-1/2/3/4 + conditional authorization)
- implemented-by -> `scripts/slurm/b2_stage_d_download.sbatch` + `scripts/slurm/b2_stage_e_extract.sbatch` + `scripts/slurm/b2_stage_t_train.sbatch` + `refine-logs/B2_IMPL_NOTES.md`

## Revision history

| rev | date | status | change | authority |
|---|---|---|---|---|
| r0 | 2026-07-14 | DRAFT-UNREVIEWED | Initial pre-registration (recon + draft; no download, no runs). 32B hidden dim = 5120 pinned from HF config. Asset check corrected a false "videos pruned" alarm (symlinks resolve; raw stores present). Stage-D/E/T/C plan, gates, kill rules, disk lifecycle, honest priors, cost. | B2 prep agent |
| r1 | 2026-07-14 | DRAFT-REV1-AWAITING-DELTA-CHECK | Applied the 4 mandatory review revisions from `refine-logs/B2_PREREG_REVIEW.md` (APPROVED + conditional authorization): **Rev-1** — MHC-EN 7B-Qwen s1/s2 provenance corrected to arcbase jobs 12275/12276 (NOT 12850, which holds only the 7B s0); fixed in frontmatter provenance, reference-arms bullet, MHC-EN table header, Namespace-gate targets, Stage-T section, ceremony and connections. **Rev-2** — Stage-E "BOTH `--model` AND `--out_model_tag`" elevated to a HARD pre-submit diff-verify item with the burn risk stated explicitly (omitting the tag silently overwrites the 7B cache filenames; collision precedes the G-dims backstop). **Rev-3** — "P10-proven" reworded throughout: the P10 *scorer* proved 32B bf16 on 80G; the extractor reuses that loading path and its FIRST 32B run is G-repro-gated (provenance line, Stage-E note, kill rule 1, connections). **Rev-4** — G-repro execution mode pinned: FIRST-CONFIG READOUT of the 9-run Stage-T serial job (config #1 = HateMM s0), NO separate smoke submit, NO mid-job intervention; gate applied at verdict processing (ceremony steps 4/6 updated). Free strengthening applied: 5120-d auto-wiring now cited as CODE-VERIFIED per review §4a (`run_rac.py:1102-1103` → `:1117-1120` → `classifier.py:76-77`). Runner names pinned to `b2_stage_{d,e,t}_*.sbatch`. No floor, decision-rule, seed, or budget change. | `refine-logs/B2_PREREG_REVIEW.md` |
