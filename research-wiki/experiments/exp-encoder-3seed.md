---
type: experiment
node_id: exp:exp-encoder-3seed
title: "MLLM-as-encoder vs frozen-CLIP: 3-seed paired encoder-swap test (HateMM + MHC-EN), dual protocol, archive OFF"
idea_id: ""
verdict: partial
confidence: high
date: "2026-07-11"
hardware: "1x A100 (SLURM), frozen features cached -> ~21-25 s/run"
duration: "10 runs, seconds each, one serial sbatch"
provenance: "PRE-REGISTERED before seed1/seed2 (and new-code seed0) runs. reused refs: rgcl_HateMM_openai_clip-vit-large-patch14-336_HF_1035814.trainlog (HateMM CLIP s0, old code), rgcl_HateMM_Qwen2.5-VL-7B-Instruct_HF_1029175.trainlog (HateMM Qwen s0, old code), rgcl_MHC_openai_clip-vit-large-patch14-336_HF_1035813.trainlog (MHC-EN CLIP s0, old code), rgcl_MHC_Qwen2.5-VL-7B-Instruct_HF_1029174.trainlog (MHC-EN Qwen s0, old code; == mllm_train_12113.out bit-for-bit), arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2}_1227{5,6}.trainlog (MHC-EN Qwen s1/s2, current code). template: scripts/slurm/train_archive_baseline.sbatch; runner: scripts/slurm/enc3seed.sbatch"
added: 2026-07-11T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "frozen-CLIP", "encoder-swap", "multi-seed", "paired-test", "HateMM", "MHC", "pre-registered"]
---

# MLLM-as-encoder vs frozen-CLIP: 3-seed paired encoder-swap test

**verdict:** `partial` — **HateMM PASSES the pre-registered criterion under BOTH protocols
(3/3 seeds positive, mean +5.3 to +5.6 acc pts, +5.6 to +6.6 F1 pts); MHC-EN FAILS under
BOTH protocols.** The ">= 2 datasets" headline criterion is therefore **NOT met**; the honest
claim is dataset-specific. · **confidence:** `high`

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

## Results (2026-07-11, SLURM job 12850, COMPLETED 19:48, exit 0; all 10 runs)

All numbers below were read back from the raw trainlogs after tabulation
(`slurm/logs/enc3s_<dataset>_<model>_seed<s>_12850.trainlog`; line numbers in the
provenance table at the end of this section). Runner: `scripts/slurm/enc3seed.sbatch`.

### Code-version / reproduction audit (run FIRST, gates everything below)

The current-code seed0 re-runs reproduce the old-code seed0 logs **to all 4 printed
decimals, both protocols, all 4 arms**:

| dataset | arm | old-code s0 (val-sel F1/acc; final F1/acc) | new-code s0 (12850) | match |
|---|---|---|---|---|
| HateMM | CLIP | 0.8172/0.8279; 0.7997/0.8186 (1035814) | identical | MATCH |
| HateMM | Qwen | 0.8606/0.8698; 0.8507/0.8605 (1029175) | identical | MATCH |
| MHC-EN | CLIP | 0.7113/0.7826; 0.7145/0.7640 (1035813) | identical | MATCH |
| MHC-EN | Qwen | 0.7378/0.7888; 0.7596/0.8012 (1029174) | identical | MATCH |

This retires the old-code-vs-new-code confound: every flag added to `run_rac.py` since
the old runs is inert at defaults. Namespace diff between arms (audited on
enc3s MHC CLIP s1 vs reused arcbase 12275): substantive fields differ **only** in
`model=` (+ derived `exp_comment`/`output_path`); the extra fields present only in
current code (`aux_archive_version`, `aux_fields`, `lambda_aux=0.0`, `cf_negs=False`,
`cf_negs_random=False`, `cf_twin_cache='auto'`, `mm_text_weight=0.5`,
`mm_empty_text='parent'`, `mm_subclip_cache='auto'`) are all at inert defaults —
proven inert by the bit-for-bit seed0 reproductions above. The reused MHC-EN Qwen
s1/s2 logs (12275/12276) are therefore fully comparable with the CLIP arm.

### Per-seed absolute readings (Test)

**HateMM**

| seed | arm | val-sel Test F1 / acc / roc (sel ep) | final-ep Test F1 / acc / roc |
|---|---|---|---|
| 0 | CLIP | 0.8172 / 0.8279 / 0.8903 (e24) | 0.7997 / 0.8186 / 0.8857 (e29) |
| 0 | Qwen | 0.8606 / 0.8698 / 0.9156 (e28) | 0.8507 / 0.8605 / 0.9283 (e29) |
| 1 | CLIP | 0.8163 / 0.8279 / 0.8771 (e26) | 0.7822 / 0.8047 / 0.8762 (e29) |
| 1 | Qwen | 0.8586 / 0.8651 / 0.9228 (e22) | 0.8514 / 0.8605 / 0.9283 (e29) |
| 2 | CLIP | 0.7920 / 0.8047 / 0.8734 (e24) | 0.7988 / 0.8140 / 0.8812 (e29) |
| 2 | Qwen | 0.8753 / 0.8837 / 0.9306 (e29) | 0.8753 / 0.8837 / 0.9306 (e29) |

**MHC-EN**

| seed | arm | val-sel Test F1 / acc / roc (sel ep) | final-ep Test F1 / acc / roc |
|---|---|---|---|
| 0 | CLIP | 0.7113 / 0.7826 / 0.8422 (e26) | 0.7145 / 0.7640 / 0.8353 (e29) |
| 0 | Qwen | 0.7378 / 0.7888 / 0.8402 (e28) | 0.7596 / 0.8012 / 0.8528 (e29) |
| 1 | CLIP | 0.6034 / 0.7329 / 0.8048 (e16) | 0.7159 / 0.7826 / 0.8236 (e29) |
| 1 | Qwen | 0.7283 / 0.7826 / 0.8375 (e25) | 0.7203 / 0.7702 / 0.8473 (e29) |
| 2 | CLIP | 0.6997 / 0.7702 / 0.8271 (e27) | 0.7303 / 0.7888 / 0.8233 (e29) |
| 2 | Qwen | 0.6997 / 0.7702 / 0.8138 (e18) | 0.7475 / 0.7826 / 0.8570 (e29) |

(Curiosity, declared: MHC-EN seed2 val-sel picks different epochs per arm — CLIP e27,
Qwen e18 — that happen to yield identical Test macroF1/P/R/acc; only roc differs. Small
test set; treated as an exact-zero paired delta.)

### Paired deltas (Qwen − CLIP), per protocol

**HateMM — protocol (A) val-selected**

| seed | Δacc | ΔmF1 |
|---|---|---|
| 0 | +0.0419 | +0.0434 |
| 1 | +0.0372 | +0.0423 |
| 2 | +0.0790 | +0.0833 |
| **mean±std** | **+0.0527 ± 0.0229** (paired t=+3.99, 3/3 positive) | **+0.0563 ± 0.0234** (t=+4.18, 3/3 positive) |

→ mean Δacc ≥ +0.030 ✓, mean ΔmF1 ≥ +0.030 ✓, sign 3/3 ✓ — **PASS**

**HateMM — protocol (B) final-epoch**

| seed | Δacc | ΔmF1 |
|---|---|---|
| 0 | +0.0419 | +0.0510 |
| 1 | +0.0558 | +0.0692 |
| 2 | +0.0697 | +0.0765 |
| **mean±std** | **+0.0558 ± 0.0139** (t=+6.95, 3/3 positive) | **+0.0656 ± 0.0131** (t=+8.65, 3/3 positive) |

→ **PASS**

**MHC-EN — protocol (A) val-selected**

| seed | Δacc | ΔmF1 |
|---|---|---|
| 0 | +0.0062 | +0.0265 |
| 1 | +0.0497 | +0.1249 |
| 2 | 0.0000 | 0.0000 |
| **mean±std** | **+0.0186 ± 0.0271** (t=+1.19, 2/3 positive) | **+0.0505 ± 0.0658** (t=+1.33, 2/3 positive) |

→ mean Δacc < +0.030 ✗, sign 2/3 ✗ (ΔmF1 mean passes numerically but is carried by one
seed's CLIP-arm selection pathology, s1 e16 → F1 0.6034) — **FAIL**

**MHC-EN — protocol (B) final-epoch**

| seed | Δacc | ΔmF1 |
|---|---|---|
| 0 | +0.0372 | +0.0451 |
| 1 | −0.0124 | +0.0044 |
| 2 | −0.0062 | +0.0172 |
| **mean±std** | **+0.0062 ± 0.0270** (t=+0.40, 1/3 positive) | **+0.0222 ± 0.0208** (t=+1.85, 3/3 positive) |

→ mean Δacc < +0.030 ✗, acc sign 1/3 ✗, mean ΔmF1 < +0.030 ✗ — **FAIL**

(Per pre-registration, paired t at n=3 is reported as an effect-size descriptor only;
no significance claim.)

### Judgment (pre-registered rules, applied verbatim)

- **HateMM:** PASS under **both** protocols. The frozen-Qwen encoder beats frozen-CLIP
  on every seed, both metrics, both protocols; the smallest per-seed acc delta is +0.037.
  This is the most seed- and protocol-robust positive effect measured in this project.
- **MHC-EN:** FAIL under **both** protocols. val-sel: mean Δacc +0.019 (2/3 seeds);
  final-epoch: mean Δacc +0.006 (1/3 seeds). Direction weakly positive for F1 (final-ep
  3/3 but mean +0.022 < +0.030 and driven by small deltas); the honest EN statement
  remains "≈0.77-0.80 regardless of encoder" (consistent with the archive-seeds node).
- **Headline (">= 2 datasets") criterion: NOT MET** — H1 as pre-registered is
  **rejected**. Supported dataset-specific claim: *swapping frozen-CLIP for a frozen
  Qwen2.5-VL-7B encoder in the unchanged RGCL head yields a large, seed-robust,
  protocol-robust improvement on HateMM (≈+5 acc / +6 F1 points) but no reliable
  improvement on MHC-EN.*
- Consistency note: EN Qwen val-sel 3-seed mean acc = 0.7805 vs CLIP 0.7619; final-ep
  0.7847 vs 0.7785 — both arms sit inside the known EN 0.77-0.80 noise band.

### Numeric provenance (raw-log line numbers, all in `slurm/logs/`)

| reading | file:line |
|---|---|
| HateMM CLIP s0 val-sel e24 | `enc3s_HateMM_openai_clip-vit-large-patch14-336_HF_seed0_12850.trainlog:258` |
| HateMM CLIP s0 final e29 | same file `:304` |
| HateMM CLIP s1 val-sel e26 / final e29 | `..._seed1_12850.trainlog:276` / `:304` |
| HateMM CLIP s2 val-sel e24 / final e29 | `..._seed2_12850.trainlog:258` / `:304` |
| HateMM Qwen s0 val-sel e28 / final e29 | `enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog:293` / `:303` |
| HateMM Qwen s1 val-sel e22 / final e29 | `..._seed1_12850.trainlog:235` / `:299` |
| HateMM Qwen s2 val-sel = final e29 | `..._seed2_12850.trainlog:302` |
| MHC-EN CLIP s0 val-sel e26 / final e29 | `enc3s_MHC_openai_clip-vit-large-patch14-336_HF_seed0_12850.trainlog:248` / `:273` |
| MHC-EN CLIP s1 val-sel e16 / final e29 | `..._seed1_12850.trainlog:167` / `:272` |
| MHC-EN CLIP s2 val-sel e27 / final e29 | `..._seed2_12850.trainlog:256` / `:273` |
| MHC-EN Qwen s0 val-sel e28 / final e29 | `enc3s_MHC_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog:263` / `:272` |
| MHC-EN Qwen s1 val-sel e25 / final e29 (reused) | `arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed1_12275.trainlog:240` / `:273` |
| MHC-EN Qwen s2 val-sel e18 / final e29 (reused) | `arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed2_12276.trainlog:185` / `:274` |

Selection rule for "val-sel": epoch >= warmup 5 maximizing Val_Retrieval acc (roc
tie-break), identical to the sbatch template parser; "final" = epoch 29 (all runs
trained the full 30 epochs).

## Cross-reference changelog (does NOT alter any result/verdict above)

- **2026-07-14 — MHC-ZH LoRA arm now formally measured (B3).** This node's frozen-encoder
  swap tested HateMM (PASS both protocols) and MHC-EN (FAIL both) but not ZH. The two ZH
  follow-ups are now closed: the **frozen**-Qwen ZH cell FAILED both protocols (B1, 20th
  negative, `exp-encoder-zh-b1.md`), and the **LoRA**-Qwen ZH cell was measured under the
  identical current-code `enc3seed` runner (**B3, job 13150** vs frozen-CLIP 13115) —
  verdict `final-epoch: PASS (MARGINAL); val-selected: FAIL` (final-ep mean Δacc +0.0313 /
  ΔmF1 +0.0453; val-sel +0.0246 acc < bar / +0.0339 F1; G-repro bit-exact vs arcbase
  12223-25). Pointers: `refine-logs/B3_VERDICT_REVIEW.md`, `research-wiki/experiments/exp-lora-zh-b3.md`,
  `research-wiki/PAPER_MASTER_TABLES.md` PUR-1/PUR-2 (pending-user-ruling addendum). Note the
  ZH pass rides on the **LoRA** lever, a different mechanism from this node's **frozen**-swap
  HateMM pass (family-vs-single-mechanism framing = pending user ruling).
- **2026-07-14 — MHC-EN LoRA cell closed pre-GPU (B4).** The EN-side LoRA-encoder cell (the
  mirror of B3 on `dataset=MHC`) was closed **pre-GPU** as a banked seed0 negative (22nd
  negative): seed0 paired vs frozen-CLIP = val-sel −0.0310 acc / final-ep +0.0062 acc (≪ bar),
  LoRA below both frozen encoders on EN. Pointer: `refine-logs/B4_FORENSIC_RECON.md`
  (recon: `exp-lora-sft-encoder.md:21`). Consistent with this node's MHC-EN frozen-swap FAIL.
