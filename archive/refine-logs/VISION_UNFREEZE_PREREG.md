# VISION-UNFREEZE Pre-Registration — LoRA reaching the Qwen2.5-VL vision tower vs banked generic (LLM-only) LoRA (MHC-EN + HateMM)

**Author:** vision-unfreeze prereg author (CPU-only; no GPU/SLURM/Modal spent; NO job submitted).
**Date:** 2026-07-20 NZST.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced; no job submitted.
**Implements:** `refine-logs/VISION_UNFREEZE_FORENSIC_RECON.md` (commit `fe9639a`, the GO recon) — design ruling
(ONE arm = `freeze_vision_tower: false` + `lora_target: all`, projector frozen), mechanics, resources, and
kill-switch skeleton transcribed and re-verified below. Deviations from the recon are flagged **loudly** in §11.
**House-style precedent:** `refine-logs/LORA_HATEMM_PREREG.md` (binding language, floors, freeze block),
`refine-logs/CAND2_CURRICULUM_PREREG.md` §1.2 (config-diff house style + add-over-generic bar),
`research-wiki/experiments/exp-encoder-3seed.md` (the 12850 encoder-swap protocol + decision rule verbatim),
`refine-logs/HATEMM_LORA_STREAM_DECOMP.md` (F58 image-MOVED machinery).

## Title + claim scope (verbatim)

> This measurement tests **the one lever the whole LoRA-SFT family never pulled — LoRA reaching the
> Qwen2.5-VL vision tower** — on **MHC-EN (the refutation target)** and **HateMM (mechanism-aligned
> hold/upside)**. It is a **PERFORMANCE lever + a refutation of the WORDING of F51's two-object closure /
> `REDTEAM_BAN_SCOPE_AUDIT` GAP-5b** (every banked "encoder adaptation" adapted the LLM only; "EN is closed
> to the entire representation family" was asserted over a vision path that was never adapted). It makes
> **NO novelty claim — D7 (encoder-class novelty boundary) remains the USER's ruling** (same collision as
> generic LoRA, F0.3). This prereg decides the **performance clause only.**

The cell under test is the **encoder-level** LoRA-SFT-adapted Qwen2.5-VL-7B encoder with **LoRA reaching the
ViT blocks** (`freeze_vision_tower: false` + `lora_target: all`; projector `visual.merger` stays frozen),
features fed to the standard archive-OFF RGCL align-fusion head + top-20 kNN (`enc3s`/12850 protocol), paired
3-seed vs the banked **frozen-CLIP** floor (K-V1) **and** vs the banked **generic (LLM-only) LoRA** arm (K-V2,
the decisive ViT-contribution bar), dual-protocol (val-selected AND final-epoch), on **two datasets, each
trained ONLY on its own train split** (hard veto). **ZH is a GO-IF clause only** (§3.7); no ZH config is
authored or submitted here.

---

## 0. Binding facts / honesty clauses (all present; pre-declared)

**F0.1 — Test is NOT virgin (declared).** EN and HateMM test were already read by the frozen-CLIP (12850),
frozen-Qwen (12850), and **generic-LoRA** arms (job 13235, both datasets). This prereg's vis-LoRA head reads
are **re-measurements under the identical protocol**, not first exposures. Each consumes exactly ONE budgeted
**vis-LoRA-encoder** test evaluation (EN-vis + HateMM-vis). Zero test-touch before the independent verdict.
Prior reads of the EN/HateMM test held-out set already spent by this project (every arm listed above +
LoRA-HateMM verdict `LORA_HATEMM_VERDICT_REVIEW.md` + cand-2). The vis arm is a NEW single test-touch/dataset.

**F0.2 — Single-encoder-draw limitation (pre-declared; identical to LoRA-HateMM F0.2 / cand-2 F0.2, and
CRITICAL for K-V2).** The 3 head-seeds read ONE vis-LoRA SFT encoder draw per dataset (head init + data-shuffle
vary; the vis-LoRA encoder is fixed). The reported ±band is **head-seed variance, NOT vis-LoRA-SFT-draw
variance.** The **add-over-generic (K-V2)** comparison is therefore **one vis-LoRA draw vs one generic draw,
both read by 3 head-seeds** — a head-seed-paired test that **cannot separate the vision-reach effect from
SFT-draw luck.** A draw-stability claim would need ≥3 fresh vis-LoRA SFT retrains (~14–17 h) — out of scope,
pre-declared. Symmetric with the single-draw generic and frozen-CLIP controls ⇒ a legitimate head-level paired
test. **This caveat travels with any K-V2 pass.**

**F0.3 — Novelty = D7, SAME collision as generic LoRA, PENDING USER (not decided here).** LoRA-on-ViT is a
2024-25-standard technique; a performance pass is a **performance/ablation row, not a novelty win** (recon
VERDICT + §6). What the cell *does* buy on the method-space ledger is a **refutation of the WORDING** of F51's
two-object closure and `REDTEAM_BAN_SCOPE_AUDIT` GAP-5b / `REDTEAM_UNTESTED_CELLS` C1: the on-disk generic
adapter target-modules = **88 entries, ZERO `visual.*`** (verified this prereg, §1.5), so every banked
"encoder adaptation" adapted the **LLM only** — "EN closed to the entire representation family" was asserted
over a vision path that **was never adapted**. This prereg **measures** it. Whether a pass counts toward the
goal's "novel" clause is the **USER's D7 ruling.** This prereg decides the **performance clause only.**

**F0.4 — Structural ceiling: the EN payoff is the whole ballgame, and it is capped-but-not-closed
(pre-declared, material to any claim).** HateMM/ZH already pass under generic LoRA (`LORA_HATEMM_VERDICT_REVIEW`
F53 / B3); on those legs vision adaptation can only **sharpen an already-passing leg** (adds no dataset).
**EN is the only untested lever aimed at the *upstream* image collapse** (F58/red-team §0: CLIP's healthy EN
image train-LOO AUC 0.7338 collapses to frozen-Qwen 0.5992, and LLM-only LoRA does NOT repair it — the
collapse lives in the frozen vision tower/merger, the exact parameters this arm is the only lever that
reaches, §2.3). Damper: F55's EN oracle ceiling (**+0.025 < +0.030**) is real, but was measured on the
**cross-encoder healthy-CLIP image** glued to Qwen text, NOT a **same-encoder** vision-repaired-then-co-trained
Qwen — so it does not fully subsume this cell. **Realistic best case: a cleaner ≥1-dataset story on datasets
generic LoRA already passes, plus an EN *refutation* (image moves) that may still not clear the conjunct.**

**F0.5 — Single-dataset own-train-split VETO compliance (hard user veto).** The EN vis arm trains on
`mhc_lora_train` = `data/lora_sft/MHC/train.json` (549 records, word-label hateful/normal); the HateMM vis arm
on `hatemm_lora_train` = `data/lora_sft/HateMM/train.json` (743 records). **These are byte-identical to the data
the banked generic-LoRA comparators trained on** (README + `dataset_info.json` verified: both generic adapters
fine-tuned on the same `*_lora_train` word datasets; shas §1.1). NO cross-dataset mixing, NO gold
spans/attributes, NO OCR channel, NO external API, raw videos never leave the machine; LoRA weights stay on
disk. All standing vetoes cleared. (Full vision-tower **full-FT** — ~675M trainable — is NOT this cell and is
overfit-doomed on <750 videos; recon §1.2/§1.5.)

**F0.6 — Clean-superset framing (why K-V2 isolates the vision contribution exactly).** Under `lora_target: all`
+ `freeze_vision_tower: false`, LLaMA-Factory's `find_all_linear_modules` (`misc.py:40-41`) STOPS adding the
ViT keys `["visual.patch_embed","visual.blocks"]` to the forbidden set, so the ViT block Linears become
LoRA-eligible; the projector `visual.merger` stays forbidden **unconditionally** (`misc.py:37-38`), and
`patch_embed` (Conv3d) is in `lora_conflict_keys` (`visual.py:351`) ⇒ never LoRA-able. The LLM coverage under
`all` is **byte-identical to the generic 7-named list** (verified: generic adapter = 88 LLM-only targets,
ZERO `visual.*`). **So vis-LoRA = generic ⊕ ViT-LoRA, a clean superset; the ONLY delta vs the banked generic
arm is the ViT LoRA (attn qkv+proj, mlp gate+up+down × 32 blocks, +11.15M trainable, recon §2.1).** K-V2
(vis − generic) therefore isolates the vision-reach contribution **exactly** — the cell's whole point.

**F0.7 — Pre-declared honest most-likely outcome (recon §6, informative-either-way).** The single most likely
result is **"the EN image stream MOVES but the K-V1 conjunct still FAILS"** (F44 label-limited residual +
F55's EN ceiling). That is **NOT a null result** — it *refutes* "EN is closed to the entire representation
family / no vision lever was ever tried," which is the WORDING this cell exists to test. On HateMM the likely
result is a K-V2 TIE (F58: HateMM's image is already strong/swap-neutral and the pass is text-carried &
frozen-sufficient — vision adaptation on an already-converted dataset likely sharpens a passing leg without
adding a dataset). **Honest priors: EN ~10–15% to clear K-V1; HateMM ~10–15% to clear K-V2. K-V2 TIE on both
is a fully expected, informative closure of the vision-adaptation axis.**

---

## 1. Pipeline spec — fully pinned (3 stages + a pre-declared EN early-kill gate; nothing left to interpretation)

### 1.1 Stage 0 — SFT data build (CPU; idempotent; the SAME data the generic comparator trained on)

The vis arm trains on the **generic word-variant** data — NOT a curriculum, NOT a reweight. STEP 1 of
`lora_sft_vis.sbatch` re-runs `python src/utils/build_lora_sft_data.py --dataset <DS>` (idempotent; frames
cached). The registered word datasets (`mhc_lora_train`/`hatemm_lora_train` → `train.json`,
`mhc_lora_val`/`hatemm_lora_val` → `val.json`) already exist and are **byte-identical to what the banked
generic-LoRA adapters trained on** (READMEs: "fine-tuned … on the `mhc_lora_train` / `hatemm_lora_train`
dataset"). Re-parsed + hashed this prereg (CPU, seconds):

| file | rows | sha256 |
|---|---|---|
| `data/lora_sft/MHC/train.json` | 549 | `7fe4c654b19a30bb48f6a7e6479ea8c009d6ce4df3406c14c241d68b987e1bba` |
| `data/lora_sft/MHC/val.json` | 80 | `575c84f254ebdfa90edc9be572d4cdb592afafeca54330c2b1b266ed24976571` |
| `data/lora_sft/HateMM/train.json` | 743 | `93c6d3d1bffbca22b2dd8beba57a33575a48d8ca61d8d56e3148fecdbb93973a` |
| `data/lora_sft/HateMM/val.json` | 107 | `9e103ed35a014af81eb3aa6af0d51a28707efd66a606c5bf0459db570a9cc9ef` |

The HateMM `train.json` sha `93c6d3d1…` is **identical** to the sha the LoRA-HateMM prereg (§1.1) and cand-2
(§1.1) pinned — the vis arm forks the exact records the 13235 generic HateMM arm trained on. The EN
`train.json` (549) is byte-identical to what the 2026-07-02 `logging/lora/MHC` generic adapter trained on
(unchanged mtime Jul-2). **G-repro data gate (§4.1b):** STEP 1 re-runs the build at submit; the executor
verifies these 4 shas unchanged — any mismatch = STOP (the generic comparator's data provenance would be
broken). **Deviation flag:** `B4_FORENSIC_RECON.md §(ii)` described the EN adapter's data as `train_yn.json`;
the README + `dataset_info.json` show it is the **word** `mhc_lora_train`/`train.json` — see §11 DEV-6.

### 1.2 Stage 1 — vision-unfreeze LoRA-SFT of the encoder (own train split only)

- **Submit:** `sbatch scripts/slurm/lora_sft_vis.sbatch <DS>` (`<DS>` ∈ {MHC, HateMM}).
- **Config (authored this prereg; EXACTLY the two method lines + output_dir changed vs the generic config —
  §5/§6):** `RA-HMD/…/my_configs/hatevideo/<ds>_qwen25vl_lora_vis_sft.yaml`.
  - EN: `mhc_qwen25vl_lora_vis_sft.yaml` = `mhc_qwen25vl_lora_sft.yaml` with L13 `lora_target: q_proj,…,down_proj`
    → `all`; L14 `freeze_vision_tower: true` → `false`; L27 `output_dir: …/lora/MHC` → `…/lora/MHC_vis`.
  - HateMM: `hatemm_qwen25vl_lora_vis_sft.yaml` = `hatemm_qwen25vl_lora_sft.yaml` with the same L13/L14 flips +
    L27 `output_dir: …/lora/HateMM` → `…/lora/HateMM_vis`.
  - **`diff` vs generic = EXACTLY those 3 lines each (verified this prereg, §4.2).**
- **Recipe (BYTE-IDENTICAL to the generic arm except the LoRA reach):** base `Qwen/Qwen2.5-VL-7B-Instruct`;
  `stage: sft` (word-label generative); `lora_rank 16`, `lora_alpha 32`, dropout 0.0; **`lora_target: all`,
  `freeze_vision_tower: false`, `freeze_multi_modal_projector: true`** (LoRA reaches all 28 LLM decoder layers
  — identical to generic — **plus** all 32 ViT blocks; projector frozen); `lr 1.0e-4`, `num_train_epochs 3.0`,
  cosine, `warmup_ratio 0.05`, `per_device_train_batch_size 1`, `gradient_accumulation_steps 8` (eff 8), bf16,
  gradient-checkpointing, 8-frame ShareGPT, `cutoff_len 4096`, `save_strategy epoch`, `eval_strategy epoch`.
  Output adapter → `logging/lora/<DS>_vis/` (does NOT exist; fresh SFT creates it).
- **Trainable footprint (recon §2.1, measured bit-exact from the on-disk generic adapter):** LLM-LoRA
  40,370,176 (unchanged) **+ ViT-LoRA 11,151,360** = **≈51.5M (r16), +27.6% over generic**. VRAM +2–5 GB over
  the generic run (well under 80 G; recon §2.2). **Cost: ~4–5.5 GPU-h SFT/dataset** (generic HateMM SFT
  train_runtime 10,254.7 s ≈ 2.85 h; ViT backward + recompute add ~+30–60%; recon §2.3).

### 1.3 Stage 2 — feature extraction with the vis-LoRA-merged encoder (NO extractor edit)

- **Submit:** `sbatch scripts/slurm/gen_embed_lora.sbatch <DS> logging/lora/<DS>_vis Qwen2.5-VL-7B-Instruct-LoRA-vis_HF`
- The runner (`gen_embed_lora.sbatch` + `generate_VideoMLLM_embedding_lora_HF.py`) is **adapter-generic — no
  edit**: `:419` loads the full base VLM, `:429-441` `PeftModel.from_pretrained` + `merge_and_unload` merges
  **whatever LoRA modules the adapter contains, including `visual.*`** (verified this prereg — the merge is
  generic over the adapter's target set). Takes the out-model-tag as **arg 3**; writes 8-frame dual-stream
  3584-d img/text embeddings for all 3 splits into
  `data/CLIP_Embedding/<DS>/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-vis_HF.pt` (**DISTINCT tag**;
  never clobbers the frozen / `-LoRA_HF` / `-LoRA-curric_HF` caches), then B2-pushes.
- **Deployed encoder input is label-free single-video** (`IMG_INSTRUCTION`/`TEXT_INSTRUCTION`, `:59-66`) — no
  gold enters the deployed path. **Cost:** ~0.4 h GPU/dataset.

### 1.4 Stage 2.5 — EN IMAGE-MOVED early-kill gate ($0 CPU, after EN extraction, BEFORE the head job)

- **Run (CPU, seconds; the ONLY new decision code, §5 artifact E):**
  `python scripts/analysis/vis_image_moved_probe.py --dataset MHC --context`
- Reuses the **committed F58 machinery** (`scripts/analysis/encoder_swap_geometry.py`, imported verbatim — the
  same functions `hatemm_lora_stream_decomp.py` imports). Computes, on banked EN **train + dev_seen** caches
  (zero test-touch): `dAUC_img = AUC_img(vis-LoRA) − AUC_img(generic-LoRA)` on two footings (train-LOO kNN via
  `G.loo_knn`→`G.auc`; held-out dev kNN via `G.knn_vote`→`G.auc`), on the L2-normed image stream.
- **The gate is the branch point of §3.4.** If FLAT/DEGRADED → EN head budget CANCELLED (bank the kill), the
  combined head job runs the **HateMM leg only**; HateMM proceeds regardless. If MOVED → EN head proceeds.

### 1.5 Stage 3 — 3-seed RGCL align-fusion head + kNN (paired vs frozen-CLIP AND generic-LoRA)

- **Submit:** `sbatch scripts/slurm/enc3seed_lora_vis.sbatch "<DSLIST>"` (authored this prereg — §5/§6), where
  `<DSLIST>` = `"HateMM MHC"` if the EN gate MOVED (default), or `"HateMM"` if the EN gate killed EN (§3.4).
- **What it runs:** up to 6 head-only runs (features cached, ~20–25 s each): HateMM-vis seeds 0/1/2 (always)
  **and** MHC-EN-vis seeds 0/1/2 (iff EN gate MOVED), `--model Qwen2.5-VL-7B-Instruct-LoRA-vis_HF`,
  `--group_name RAC_video_lora_vis`, `--force False`.
- **CRITICAL same-code guarantee (verified this prereg — §4.2):** the `run_one`…`PY` block is **BYTE-IDENTICAL**
  to `enc3seed.sbatch` (`diff` empty, 42 lines); the ONLY manipulated variables vs the banked CLIP/generic
  controls are `--model` and `--group_name`. Config: `--batch_size 64 --lr 0.0001 --epochs 30 --topk 20
  --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align --hard_negatives_loss True
  --no_hard_negatives 1 --metric cos --loss triplet --hybrid_loss True --warmup 5 --lambda_seg 0 --archive OFF`.
  Identical to `exp-encoder-3seed.md` H1 / LoRA-HateMM / cand-2.
- **Cost:** ~2 min GPU total.

**Total NEW GPU: ~9–12 A100-h** (2× vis-SFT ~8–11 h dominates; 2× extract ~0.8 h; head ~0.03 h). EN gate is $0 CPU.

---

## 2. Comparison floors + generic arms — INDEPENDENTLY RE-DERIVED from raw trainlogs (numeric-provenance discipline)

Every number below was independently re-parsed **this prereg** from the raw `slurm/logs/enc3s_*_{12850,13235}.trainlog`
(+ EN-Qwen `arcbase_MHC_*_1227{5,6}.trainlog`) with the EXACT `enc3seed.sbatch` embedded parser (val-sel = epoch
≥ warmup 5 max `Val_Retrieval` acc, roc tie-break; final = max epoch). **All means match
`LORA_HATEMM_VERDICT_REVIEW.md` §1–2 and `CAND2_CURRICULUM_PREREG.md` §2.2 to 4dp** — no discrepancy.

### 2.1 HateMM floors + generic arm

| arm | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | 3-seed mean acc/F1 |
|---|---|---|---|---|---|
| **frozen-CLIP** (K-V1 pairs vs this) | val-sel | 0.8279/0.8172 | 0.8279/0.8163 | 0.8047/0.7920 | **0.8202 / 0.8085** |
| **frozen-CLIP** | final-ep | 0.8186/0.7997 | 0.8047/0.7822 | 0.8140/0.7988 | **0.8124 / 0.7936** |
| **frozen-Qwen** (context) | val-sel | 0.8698/0.8606 | 0.8651/0.8586 | 0.8837/0.8753 | **0.8729 / 0.8648** |
| **frozen-Qwen** | final-ep | 0.8605/0.8507 | 0.8605/0.8514 | 0.8837/0.8753 | **0.8682 / 0.8591** |
| **generic-LoRA (13235)** (K-V2 pairs vs this) | val-sel | 0.8605/0.8521 | 0.8698/0.8620 | 0.8558/0.8495 | **0.8620 / 0.8545** |
| **generic-LoRA (13235)** | final-ep | 0.8651/0.8580 | 0.8744/0.8660 | 0.8698/0.8613 | **0.8698 / 0.8618** |

### 2.2 MHC-EN floors + generic arm

| arm | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | 3-seed mean acc/F1 |
|---|---|---|---|---|---|
| **frozen-CLIP** (EN K-V1 pairs vs this) | val-sel | 0.7826/0.7113 | 0.7329/0.6034 | 0.7702/0.6997 | **0.7619 / 0.6715** |
| **frozen-CLIP** | final-ep | 0.7640/0.7145 | 0.7826/0.7159 | 0.7888/0.7303 | **0.7785 / 0.7202** |
| **frozen-Qwen** (EN honesty bar) | val-sel | 0.7888/0.7378 | 0.7826/0.7283 | 0.7702/0.6997 | **0.7805 / 0.7219** |
| **frozen-Qwen** | final-ep | 0.8012/0.7596 | 0.7702/0.7203 | 0.7826/0.7475 | **0.7847 / 0.7425** |
| **generic-LoRA (13235)** (EN K-V2 pairs vs this) | val-sel | 0.7516/0.6916 | 0.7391/0.6920 | 0.7888/0.7506 | **0.7598 / 0.7114** |
| **generic-LoRA (13235)** | final-ep | 0.7702/0.7302 | 0.7764/0.7360 | 0.7888/0.7506 | **0.7785 / 0.7389** |

Provenance (file:line): HateMM/EN CLIP + frozen-Qwen `enc3s_{HateMM,MHC}_{openai_clip-…-336_HF,Qwen2.5-VL-7B-Instruct_HF}_seed{0,1,2}_12850.trainlog`
(EN-Qwen s1/s2 = `arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2}_1227{5,6}.trainlog`); generic-LoRA
`enc3s_{HateMM,MHC}_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13235.trainlog`. Generic EN LoRA = the F53
verdict's bundled B4 arm (`LORA_HATEMM_VERDICT_REVIEW.md §2.2`, FAIL both protocols).

### 2.3 EN image-stream kNN AUC anchor for the image-MOVED gate — RE-DERIVED with the committed F58 operator

Re-derived **this prereg** with `scripts/analysis/encoder_swap_geometry.py` (the operator the pinned diagnostic
runs), image-only stream, EN train (n=549) + dev_seen (n=80):

| encoder (EN, image-only) | train-LOO kNN AUC | dev kNN AUC |
|---|---|---|
| CLIP (healthy) | **0.7338** | 0.7367 |
| frozen-Qwen (collapsed) | **0.5992** | 0.6865 |
| **generic-LoRA (the K-V2/gate anchor)** | **0.6236** | **0.6756** |

- **F44's collapse reproduced** (CLIP 0.7338 → frozen-Qwen 0.5992 train-LOO). **LLM-only LoRA leaves the EN
  image FLAT under the committed operator too:** frozen→generic-LoRA is +0.0245 train-LOO but **−0.0109 dev**
  (dev fails to corroborate ⇒ FLAT under F58's same-sign rule) — machinery-validated this prereg by running
  the diagnostic with generic_tag=frozen-Qwen, vis_tag=generic-LoRA. The collapse lives **upstream of the LLM,
  in the vision tower/merger** — the exact parameters this arm is the only lever that reaches.
- **DEVIATION FLAG (LOUD — §11 DEV-4):** these absolute values differ from the recon's pinned anchor
  (generic-LoRA EN img **0.659/0.695**), which came from the red-team §0 **scratch** probe (uncommitted
  `redteam_stream_topk_probe.py`, a different vote operator, ~0.01–0.04 off). The **gate is unchanged** because
  it is a **same-operator DELTA** (both LoRA arms read by the committed operator) with an **operator-independent
  threshold** (F58's +0.010/+0.005 resolution floor); only the absolute anchor is corrected to the operator the
  diagnostic actually uses.

---

## 3. Decision rule + kill-switches (paired, both protocols judged independently, 3/3 sign, pre-declared)

### 3.1 Decision rule — verbatim from `exp-encoder-3seed.md:73-85`

> For each dataset × protocol: (1) per-seed paired difference δ = (treatment − control) for acc and macro-F1
> at seeds 0/1/2; (2) 3-seed mean ± std + sign consistency (how many of 3 positive); (3) n=3 too small for a
> bootstrap — report the paired-t **as an effect-size descriptor only**, no significance claim; (4) **pass =
> mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive**; (5) headline claim
> requires pass on ≥ 2 datasets under a stated protocol; both protocols judged separately; verdict written
> exactly "final-epoch: pass/fail; val-selected: pass/fail".

Both protocols judged **independently** (no protocol-shopping, no metric-shopping). Treatment = vis-LoRA.

### 3.2 K-V1 — HOUSE PERFORMANCE CONJUNCT (primary; vis-LoRA − frozen-CLIP)

Per dataset × protocol: **mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND sign 3/3**, judged **independently
under EACH protocol**. Floors §2.1/§2.2 (HateMM CLIP val-sel 0.8202/0.8085, final 0.8124/0.7936; EN CLIP
val-sel 0.7619/0.6715, final 0.7785/0.7202). Below the conjunct under a protocol → **NEGATIVE** on that
protocol. **A K-V1 pass that merely equals generic earns nothing (K-V2 is the decisive bar).**

### 3.3 K-V2 — ADD-OVER-BANKED-GENERIC (THE DECISIVE ViT-CONTRIBUTION BAR; vis-LoRA − generic-LoRA, paired by head-seed)

Because vis-LoRA = generic ⊕ ViT-LoRA (F0.6), K-V2 isolates the vision contribution **exactly**. Paired per
head-seed (vis seed s − generic seed s); generic arms banked (job 13235, §2.1/§2.2 — NOT re-run).

- **PASS (per dataset, ≥1 protocol):** mean paired **Δacc ≥ +0.010 AND sign 3/3 positive AND mean ΔmF1 ≥ 0**.
  Justification (identical to cand-2 §3.3/§2.3): the banked generic between-seed acc spread is ≤ 0.014, and a
  3-seed **mean** move ≥ +0.010 with **3/3 concordant sign** is ~2× the per-seed std with every seed improving,
  distinguishable from a within-spread tie (the 3/3-sign requirement is the teeth).
- **TIE = NO ViT CONTRIBUTION (the F0.7 outcome):** mean paired |Δacc| < +0.010 **OR** sign not 3/3 ⇒ "vision
  reach adds nothing over LLM-only LoRA"; **report "generic LoRA with vision reach that did not matter," bank
  the negative, do NOT claim a vision contribution.** **F0.2 caveat travels with any K-V2 PASS** (single draw).

### 3.4 EN IMAGE-MOVED — pre-declared EARLY-KILL gate ($0-after-EN-extract, BEFORE the EN head budget)

`dAUC_img = AUC_img(vis-LoRA) − AUC_img(generic-LoRA)`, committed F58 operator (§1.4/§2.3):
**MOVED iff dAUC_img ≥ +0.010 (train-LOO) AND ≥ +0.005 (dev)** [F58's rule verbatim, `hatemm_lora_stream_decomp.py:86-89`];
DEGRADED iff ≤ −0.010 train-LOO; FLAT otherwise. Anchor = generic-LoRA EN img **0.6236 train / 0.6756 dev**.

- **MOVED → EN head PROCEEDS** (vision LoRA is live on the EN image stream; submit the head job default
  `"HateMM MHC"`).
- **FLAT / DEGRADED → EN head budget CANCELLED**, bank the "**vision LoRA inert on EN image**" kill (independent
  of accuracy — the vision LoRA did not touch the collapsed stream it was the only lever for); submit the head
  job `"HateMM"` only. **HateMM chain proceeds regardless.** This is the recon's "cheap-after-GPU screen"
  (§5): it spends the trivial EN head budget only if the vision reach demonstrably moved the EN image geometry.

### 3.5 EN HONESTY FLAG (vis-LoRA must ALSO beat the frozen-Qwen floor on the claimed protocol)

On EN the vis-LoRA arm must **beat the FROZEN-QWEN floor** on the protocol a claim is made under (EN frozen-Qwen
val-sel **0.7805/0.7219**, final **0.7847/0.7425**; §2.2) — **the bar the generic EN LoRA could NOT clear**
(`LORA_HATEMM_VERDICT_REVIEW.md §2.2`: generic EN LoRA val-sel 0.7598, final 0.7785, both ≤ frozen-Qwen). If
vis-LoRA still cannot clear frozen-Qwen, the vision adaptation **rearranged a dead cell** rather than opening
EN in a decision-relevant way. This flag is **material to any EN claim** and travels with it; it does not by
itself flip K-V1/K-V2 (which are vs CLIP/generic), but a K-V1/K-V2 read on EN that fails this flag is reported
as "moved but did not clear the frozen-Qwen ceiling."

### 3.6 KS-REGRESSION — BELOW-GENERIC KILL

If vis-LoRA − generic-LoRA **mean Δacc ≤ −0.014** on a held leg (below the full banked head-seed spread, §2.3),
the vision reach **degraded** adaptation (ViT-LoRA overfit <750 videos, or destabilised the LLM-LoRA path) →
**KILL**, bank "vision-LoRA hurts."

### 3.7 OVERFIT TRIPWIRES (two prongs, pre-declared)

- **(a) image-stream sanity = the §3.4 gate** (EN image-MOVED). On HateMM the same $0 diagnostic is run for the
  record (`--dataset HateMM`, generic-tag `…-LoRA_HF`, vis-tag `…-LoRA-vis_HF`) but is **not** a kill — HateMM's
  image is already strong/swap-neutral (F58), so a FLAT HateMM image is expected and does not gate the HateMM
  head.
- **(b) eval_loss band:** vis-LoRA `logging/lora/<DS>_vis/all_results.json` eval_loss should land in the
  **~0.10–0.18** band (generic anchors: HateMM 0.1084, MHC 0.1620). A **much lower** vis eval_loss **plus** a
  widening val-sel↘final-epoch gap = overfit warning on <750 videos (report as a caveat; combine with KS-regression).

### 3.8 GO-IF (ZH — clause only, NO config authored/submitted here)

Per the task + recon (§3 datasets table): ZH is off-mechanism (text-borne, F45). **ZH vis-LoRA is proposed ONLY
IF the EN image-MOVED gate PASSES** (i.e. the vision reach demonstrably moves an image stream). If EN moves, a
follow-up prereg may add a ZH vis arm (`mhc_zh_qwen25vl_lora_vis_sft.yaml`, same 3-line diff) as a third leg
(~5–7 GPU-h). **This prereg does not author or submit any ZH artifact.**

### 3.9 Gate order

G-repro-adapted (§4: SFT-loss sanity + data-build sha re-verify + head Namespace-diff) → **EN image-MOVED gate
(§3.4, EN branch point)** → K-V1 → K-V2 → EN honesty flag → KS-regression → overfit read. Single test-touch per
dataset. The verdict is rendered by an **independent 0-context reviewer against this prereg VERBATIM.**

---

## 4. G-repro (adapted — first vis-LoRA draw, no bit-exact anchor) + smoke plan + collision safety

### 4.1 G-repro discipline

- **(a) SFT smoke gate + ViT-LoRA-present check (Stage 1) — LOAD-BEARING.** A tiny SFT smoke (§4.4.1) must show
  loss **finite (no NaN), decreasing, checkpoint written**, AND — the check the whole cell rests on — that the
  smoke adapter **actually contains ViT LoRA tensors**. Exact inspection command (run on the throwaway smoke
  adapter):
  ```
  python - <<'PY'
  from safetensors.torch import load_file
  k = list(load_file("logging/lora/_smoke_vis/adapter_model.safetensors").keys())
  vis = [x for x in k if "visual.blocks" in x]
  print("n_visual_lora_tensors =", len(vis))   # MUST be > 0 (expect ~320 = 32 blocks × 5 Linears × {A,B})
  print(vis[:6])
  PY
  ```
  **If `n_visual_lora_tensors == 0` the config did not reach the ViT ⇒ ABORT** (the arm would be identical to
  generic and K-V2 vacuous). On the full run, eval_loss should land ~0.10–0.18 (§3.7b).
- **(b) Data-build reproducibility gate.** STEP 1 re-runs `build_lora_sft_data.py`; the executor verifies the 4
  data shas §1.1 unchanged (the vis arm must train on the same records as the generic comparator). Mismatch = STOP.
- **(c) Head runs = SAME-CODE as the banked controls.** The Namespace diff between a vis head run and the
  12850/13235 controls MUST be `--model` + derived-inert fields (`exp_comment`, `group_name`, `output_path`)
  only — plus the inert TARC/oracle argparse defaults already blessed by the LoRA-HateMM verdict
  (`LORA_HATEMM_VERDICT_REVIEW.md §4.1b`: provably no-op, B3/F53-precedented). `run_one` byte-identical (§4.2).
- **(d) frozen-CLIP + generic-LoRA controls re-paired from banked logs (§2), not re-run.**

### 4.2 Same-code + syntax + config-diff verification (run this prereg — PASS)

- `run_one`…`PY` block of `enc3seed_lora_vis.sbatch` == `enc3seed.sbatch`: **BYTE-IDENTICAL** (`diff` empty, 42
  lines). Full-file `diff` vs `enc3seed_lora_hatemm.sbatch`: header comment, `LORA` tag
  (`…-LoRA-vis_HF`), `GROUP_NAME` (`RAC_video_lora_vis`), and the CONFIGS block (arg-driven `DATASETS`
  loop instead of a hardcoded 6-row array — §11 DEV-2) only.
- Config `diff` vs generic = **EXACTLY 3 lines** each (`lora_target`, `freeze_vision_tower`, `output_dir`).
- `bash -n` on both new sbatch = **SYNTAX_OK**. `python -m py_compile vis_image_moved_probe.py` = **OK**;
  machinery dry-run reproduces the frozen→generic-LoRA EN image FLAT (dAUC +0.0245 tr / −0.0109 dev).

### 4.3 Collision safety (verified this prereg; re-check at submit)

- `logging/lora/{MHC_vis,HateMM_vis}` — do NOT exist ⇒ fresh SFT (no clobber of generic MHC/HateMM adapters).
- `data/CLIP_Embedding/{MHC,HateMM}/*LoRA-vis*.pt` — do NOT exist ⇒ fresh extraction; frozen + `-LoRA_HF` +
  `-LoRA-curric_HF` caches untouched.
- `logging/Retrieval/{MHC,HateMM}/RAC_video_lora_vis*` — do NOT exist ⇒ fresh group, `force=False` never trips
  `run_rac.py:904-908`; the `-vis` tag differs from CLIP/Qwen/`LoRA`/`LoRA-curric`, so dirs are distinct
  regardless. NO 12850/13235 arm overwritten.
- `slurm/logs/enc3s_{HateMM,MHC}_Qwen2.5-VL-7B-Instruct-LoRA-vis_HF_seed*_*.trainlog` — do NOT exist ⇒ no collision.

### 4.4 Smoke plan (executor runs BEFORE the real submits; leave no artifact that trips §4.3)

1. **SFT smoke + ViT-LoRA-present check:** copy a vis config to a throwaway with `max_steps: 20`,
   `save_steps: 20`, `output_dir: logging/lora/_smoke_vis` — confirm loss finite/decreasing, checkpoint written,
   **and run the §4.1a `safetensors` command → `n_visual_lora_tensors > 0` (LOAD-BEARING)**; then delete the
   smoke dir. (Do NOT smoke-write into `logging/lora/<DS>_vis`.)
2. **1-seed head smoke (optional):** on the existing generic LoRA cache (`data/CLIP_Embedding/<DS>/*LoRA_HF.pt`),
   run ONE `run_rac.py` head with throwaway `--group_name _smoke` to confirm the align-fusion path loads +
   completes 30 epochs; delete the `_smoke` dir. If in doubt, skip — same-code guarantee + cache dims are
   CPU-verified.

---

## 5. Artifacts authored this prereg + hash-freeze block

### 5.1 New artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current) |
|---|---|---|---|
| P | `refine-logs/VISION_UNFREEZE_PREREG.md` | **NEW** — this file | *(reviewer fills at freeze)* |
| A | `RA-HMD/…/my_configs/hatevideo/mhc_qwen25vl_lora_vis_sft.yaml` | **NEW** — 3-line diff vs `mhc_qwen25vl_lora_sft.yaml` (`lora_target: all`, `freeze_vision_tower: false`, `output_dir …/MHC_vis`) | `7d551460239aaf537ecbb62f4c77d859cfeea3403867ccb99b34d31eeeb7fd3f` |
| B | `RA-HMD/…/my_configs/hatevideo/hatemm_qwen25vl_lora_vis_sft.yaml` | **NEW** — 3-line diff vs `hatemm_qwen25vl_lora_sft.yaml` (same flips + `output_dir …/HateMM_vis`) | `634bd0bb02789a1728728be19efdf91b69b36aab27a5f1dd9eab229e3041700b` |
| C | `scripts/slurm/lora_sft_vis.sbatch` | **NEW** — clone of `lora_sft.sbatch`; {MHC, HateMM} cases → vis configs + `<DS>_vis` output; STEP-1 generic build unchanged | `3e895420e308b30d8371c54a7a03ab9cf033ebe4804143a511989e68f3ef7946` |
| D | `scripts/slurm/enc3seed_lora_vis.sbatch` | **NEW** — clone of `enc3seed_lora_hatemm.sbatch`; `run_one` BYTE-IDENTICAL to `enc3seed.sbatch`; `-vis` tag, `RAC_video_lora_vis`, arg-driven `DATASETS` (EN branch) | `ca7749149fd836bd84404cad8436fd868c51c1ff2930c3ed9e91657c6933e2fb` |
| E | `scripts/analysis/vis_image_moved_probe.py` | **NEW** — EN image-MOVED gate; imports `encoder_swap_geometry.py` VERBATIM (F58 machinery); train+dev only, zero test-touch | `719ab1fe837ad4c9f75c750b8e8e5d5853bd64cdcf3c526da35fe0177944c4a6` |

### 5.2 Reused-unchanged machinery (verify sha at submit time; do NOT edit)

| path | role | sha256 |
|---|---|---|
| `scripts/analysis/encoder_swap_geometry.py` | F58 kNN-AUC machinery (imported by E; NO edit) | `974771775e15fd58c31bd07bfd26d6dac43eab304b5fd888235a8449009190f6` |
| `scripts/slurm/gen_embed_lora.sbatch` | extraction (adapter-generic; out-tag arg 3; NO edit) | `c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386` |
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | extractor (merge_and_unload covers `visual.*`; NO edit) | *(unchanged; not a decision gate)* |
| `src/utils/build_lora_sft_data.py` | Stage-0 generic word-variant builder (idempotent; NO edit) | *(unchanged; not a decision gate)* |
| `RA-HMD/…/mhc_qwen25vl_lora_sft.yaml` | EN fork source | `db371c18f306c5a3a00eeef8550964c3ddacf9e20400439324009ef2e69b1b52` |
| `RA-HMD/…/hatemm_qwen25vl_lora_sft.yaml` | HateMM fork source | `d2f415cd93fa6f7b439fd4b4573a536baf48ad42186dc8bd50f3fab20553e36a` |
| `scripts/slurm/enc3seed.sbatch` | same-code anchor for §4.2 | `dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d` |
| `data/lora_sft/MHC/train.json` (549) / `val.json` (80) | EN SFT data (== generic comparator) | `7fe4c654…e1bba` / `575c84f2…76571` |
| `data/lora_sft/HateMM/train.json` (743) / `val.json` (107) | HateMM SFT data (== generic comparator) | `93c6d3d1…73973a` / `9e103ed3…9cc9ef` |

### 5.3 Hash-freeze (to be filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file VISION_UNFREEZE_PREREG.md, after review>
A 7d551460239aaf537ecbb62f4c77d859cfeea3403867ccb99b34d31eeeb7fd3f  mhc_qwen25vl_lora_vis_sft.yaml
B 634bd0bb02789a1728728be19efdf91b69b36aab27a5f1dd9eab229e3041700b  hatemm_qwen25vl_lora_vis_sft.yaml
C 3e895420e308b30d8371c54a7a03ab9cf033ebe4804143a511989e68f3ef7946  lora_sft_vis.sbatch
D ca7749149fd836bd84404cad8436fd868c51c1ff2930c3ed9e91657c6933e2fb  enc3seed_lora_vis.sbatch
E 719ab1fe837ad4c9f75c750b8e8e5d5853bd64cdcf3c526da35fe0177944c4a6  vis_image_moved_probe.py
```
Executor re-runs `sha256sum` on A–E (and this file) at submit time; any mismatch = authorization VOID.

---

## 6. Single-submit / execution plan + resource plan

**Order (EN-first for the early-kill gate; HateMM alongside; one combined head job):**

1. `sbatch scripts/slurm/lora_sft_vis.sbatch MHC` → `logging/lora/MHC_vis/` (~4–5.5 h). Gate: SFT smoke +
   ViT-LoRA-present check (§4.4.1/§4.1a) BEFORE; on COMPLETE apply §4.1a/§4.1b.
2. `sbatch --dependency=afterok:<1> scripts/slurm/gen_embed_lora.sbatch MHC logging/lora/MHC_vis Qwen2.5-VL-7B-Instruct-LoRA-vis_HF`
   → EN vis cache (~0.4 h).
3. **($0 CPU, after 2) EN image-MOVED gate:** `python scripts/analysis/vis_image_moved_probe.py --dataset MHC --context`.
   Record `en_head_proceeds` from the JSON. **BRANCH POINT (§3.4):** MOVED → EN head IN; FLAT/DEGRADED → EN head
   CANCELLED (bank kill).
4. `sbatch scripts/slurm/lora_sft_vis.sbatch HateMM` → `logging/lora/HateMM_vis/` (~4–5.5 h). [may run in
   parallel with 1–2 if the 2-GPU quota allows; else sequential.]
5. `sbatch --dependency=afterok:<4> scripts/slurm/gen_embed_lora.sbatch HateMM logging/lora/HateMM_vis Qwen2.5-VL-7B-Instruct-LoRA-vis_HF`
   → HateMM vis cache (~0.4 h).
6. **Head job** (after 2 AND 5, and the §3 gate read): `sbatch scripts/slurm/enc3seed_lora_vis.sbatch "$DSLIST"`
   where `DSLIST="HateMM MHC"` if the EN gate MOVED, else `"HateMM"`. Produces
   `slurm/logs/enc3s_{HateMM[,MHC]}_Qwen2.5-VL-7B-Instruct-LoRA-vis_HF_seed{0,1,2}_<JID>.trainlog` (~2 min).

Chainable via `--dependency=afterok:`. **Resource plan:** 1×A100/job; `conda activate HateVideo`; `sbatch` with
**NO `--time`**; initial `PENDING (JobHeldUser)` = **WAIT for auto-release, never force** (CLAUDE.md).
`lora_sft_vis.sbatch` sources `conda.sh` directly, sets the offline/import env + `CUDA_HOME` shim, and has a
≥20 G disk guard (inherited verbatim from `lora_sft.sbatch`).

**Cost ledger:** EN + HateMM = **~9–12 GPU-h** (2× vis-SFT dominates); + ZH (GO-IF only, not here) ~14–18.
$0 CPU: the EN image-MOVED gate + all floor/anchor re-derivations.

**Test-touch:** the Stage-3 vis-LoRA head reads are the ONLY budgeted vis-LoRA-encoder test evaluations (EN +
HateMM); zero test-touch before the verdict. **The executor transcribes raw both-protocol per-seed numbers
(line-numbered) and applies NO gates/interpretation** — the verdict (G-repro → ViT-present → Namespace-diff →
EN image-MOVED → K-V1/K-V2 → EN-honesty → KS-regression) is rendered by an **independent 0-context reviewer
against this prereg VERBATIM.**

**No job is submitted by this prereg author.** Submission happens only after the independent review + hash-freeze
(run by the orchestrator).

---

## 7. Outcome table template (filled ONLY from raw trainlogs / gate JSON at verdict time)

### 7.0 EN image-MOVED gate (fill from `scripts/analysis/vis_image_moved_MHC_out.json`)

| footing | generic-LoRA img AUC (§2.3) | vis-LoRA img AUC | dAUC(vis−gen) | MOVED? (≥+0.010 tr / ≥+0.005 dv) |
|---|---|---|---|---|
| train-LOO | 0.6236 | ___ | ___ | ___ |
| dev | 0.6756 | ___ | ___ | ___ |

`EN gate: <MOVED → EN head PROCEEDS | FLAT/DEGRADED → EN head CANCELLED, bank "vision LoRA inert on EN image">.`

### 7.1 HateMM — vis-LoRA vs frozen-CLIP (K-V1) AND vs generic-LoRA (K-V2)

| seed | protocol | vis acc/F1 | CLIP acc/F1 (§2.1) | Δ(vis−CLIP) acc/F1 | generic acc/F1 (§2.1) | Δ(vis−generic) acc/F1 |
|---|---|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8279/0.8172 | ___ | 0.8605/0.8521 | ___ |
| 1 | val-sel | ___ | 0.8279/0.8163 | ___ | 0.8698/0.8620 | ___ |
| 2 | val-sel | ___ | 0.8047/0.7920 | ___ | 0.8558/0.8495 | ___ |
| **mean** | **val-sel** | ___ | **0.8202/0.8085** | **___** | **0.8620/0.8545** | **___** |
| 0 | final-ep | ___ | 0.8186/0.7997 | ___ | 0.8651/0.8580 | ___ |
| 1 | final-ep | ___ | 0.8047/0.7822 | ___ | 0.8744/0.8660 | ___ |
| 2 | final-ep | ___ | 0.8140/0.7988 | ___ | 0.8698/0.8613 | ___ |
| **mean** | **final-ep** | ___ | **0.8124/0.7936** | **___** | **0.8698/0.8618** | **___** |

### 7.2 MHC-EN — vis-LoRA vs frozen-CLIP (K-V1) AND vs generic-LoRA (K-V2) [only if EN gate MOVED]

| seed | protocol | vis acc/F1 | CLIP acc/F1 (§2.2) | Δ(vis−CLIP) acc/F1 | generic acc/F1 (§2.2) | Δ(vis−generic) acc/F1 |
|---|---|---|---|---|---|---|
| 0 | val-sel | ___ | 0.7826/0.7113 | ___ | 0.7516/0.6916 | ___ |
| 1 | val-sel | ___ | 0.7329/0.6034 | ___ | 0.7391/0.6920 | ___ |
| 2 | val-sel | ___ | 0.7702/0.6997 | ___ | 0.7888/0.7506 | ___ |
| **mean** | **val-sel** | ___ | **0.7619/0.6715** | **___** | **0.7598/0.7114** | **___** |
| 0 | final-ep | ___ | 0.7640/0.7145 | ___ | 0.7702/0.7302 | ___ |
| 1 | final-ep | ___ | 0.7826/0.7159 | ___ | 0.7764/0.7360 | ___ |
| 2 | final-ep | ___ | 0.7888/0.7303 | ___ | 0.7888/0.7506 | ___ |
| **mean** | **final-ep** | ___ | **0.7785/0.7202** | **___** | **0.7785/0.7389** | **___** |

EN honesty flag (§3.5): vis-LoRA ≥ frozen-Qwen (val-sel 0.7805/0.7219; final 0.7847/0.7425)? ___ .

### 7.3 Fixed write-up format

`EN gate:  <MOVED/FLAT/DEGRADED>.`
`HateMM:  final-epoch: <pass/fail> (K-V1) · K-V2: <pass/tie> · KS-regression: <ok/kill>.`
`MHC-EN:  <final-epoch/val-selected: pass/fail> (K-V1) · K-V2: <pass/tie> · frozen-Qwen honesty: <clears/does-not> ` [or `CANCELLED by EN gate`].
(+ MARGINAL note if a K-V1 acc pass is within noise, per B3 §2.2 precedent.)

---

## 8. What a PASS / FAIL means for the goal (D7 boundary — this prereg does NOT decide)

- **EN K-V1 PASS + K-V2 PASS + clears frozen-Qwen (recon prior ~10–15%):** vision reach opened EN — the goal's
  performance conjunct would gain the missing 2nd/3rd dataset via a **representation-class lever**. Whether
  LoRA-on-ViT counts as "novel" is the **USER's D7 ruling** (F0.3). Caveat: single vis-LoRA draw (F0.2).
- **EN image MOVES but K-V1 FAILS (recon prior — the MOST likely EN outcome, F0.7):** **still an informative
  refutation** of "EN closed to the entire representation family / no vision lever was tried" (GAP-5b / C1) —
  the honest performance kill for the vision axis, with the *mechanism* (F44 label-limited residual) now
  measured rather than reasoned.
- **EN image FLAT/DEGRADED (§3.4 gate kills EN pre-head):** the vision LoRA is **inert** on the collapsed EN
  image stream even when architecturally reachable — the strongest closure (bank; the EN image collapse is not
  LoRA-repairable at r16 on 549 videos).
- **HateMM K-V2 TIE (recon prior — likely):** vision reach adds nothing over LLM-only LoRA on an
  already-converted, text-carried dataset (F58) — bank; the vision-adaptation axis is exhausted on the passing
  legs. **HateMM K-V2 PASS** would be a bonus ablation row (still D7 = user).
- **KS-regression:** the vision reach hurt — bank "vision-LoRA degrades adaptation."

**Framing sentence (verbatim):** *this measurement tests the one lever the LoRA-SFT family never pulled — LoRA
reaching the Qwen2.5-VL vision tower — on EN (the refutation target) and HateMM (hold/upside); a pass is a
performance/ablation row and a refutation of the WORDING of F51/GAP-5b, NOT a novelty win — D7 remains the
user's.*

---

## 9. Provenance index

- Recon (GO; design ruling, mechanics, resources, kill skeleton): `refine-logs/VISION_UNFREEZE_FORENSIC_RECON.md` (`fe9639a`).
- Evidence chain: `refine-logs/REDTEAM_UNTESTED_CELLS.md` §0 + C1 (the $0 motivation + red-team image AUCs);
  `refine-logs/REDTEAM_BAN_SCOPE_AUDIT.md` GAP-5b; `refine-logs/HATEMM_LORA_STREAM_DECOMP.md` (F58 image-MOVED rule).
- Mechanics (verified this prereg): `RA-HMD/…/src/llamafactory/model/model_utils/misc.py:28-52`
  (`freeze_vision_tower` guard), `.../visual.py:344-352` (qwen2_5_vl composite registration),
  `src/utils/generate_VideoMLLM_embedding_lora_HF.py:419,429-441` (adapter-generic merge_and_unload);
  generic adapter target set `logging/lora/HateMM/adapter_config.json` (88 LLM-only, zero `visual.*`).
- Floors + generic arms (re-derived §2): `LORA_HATEMM_VERDICT_REVIEW.md` §1–2 (F53; HateMM PASS, EN FAIL),
  `CAND2_CURRICULUM_PREREG.md` §2.2; raw `enc3s_*_{12850,13235}.trainlog` + `arcbase_MHC_*_1227{5,6}.trainlog`.
- Protocol + decision rule (verbatim): `research-wiki/experiments/exp-encoder-3seed.md:73-85`.
- Image-MOVED machinery: `scripts/analysis/encoder_swap_geometry.py` (reused verbatim by artifact E).
- Mechanism: `refine-logs/ENCODER_SWAP_DIAGNOSIS.md` (F44 EN image collapse), `refine-logs/HATEMM_LORA_STREAM_DECOMP.md` (F58).

**Required statements:** ZERO GPU/SLURM/Modal spent by this prereg author (only pure-CPU login-node floor
re-parsing + EN image-geometry anchor re-derivation over banked train/dev caches, seconds; no held-out test
metric produced). All floor/generic numbers re-parsed from banked completed-run trainlogs; the EN image anchor
re-derived with the committed F58 operator. No `state/` mutated. No `research-wiki/` mutated. NO job submitted. Not pushed.

---

## 10. (reserved)

---

## 11. DEVIATIONS FROM THE RECON — flagged loudly

1. **DEV-1 (config = 3 changed lines, not "2"). Neutral / clarifying.** The recon says "exactly two changed
   lines" for the *method* (`freeze_vision_tower: false`, `lora_target: all`); the registered config also
   changes `output_dir` (→ `<DS>_vis`) to avoid clobbering the generic adapter. So the `diff` vs generic is
   **exactly 3 lines** {`lora_target`, `freeze_vision_tower`, `output_dir`} — matching the task's "two changed
   lines + output_dir." `dataset`/`eval_dataset` stay the generic word variant (the vis arm trains on the SAME
   data as the generic comparator ⇒ clean superset, F0.6).

2. **DEV-2 (head sbatch: arg-driven `DATASETS` instead of a hardcoded 6-row CONFIGS). Favorable.** The recon's
   chain has an EN early-kill BEFORE the combined head job. A hardcoded 6-row array would force either a
   hash-void edit (to drop the EN rows on a kill) or a wasted EN head run. `enc3seed_lora_vis.sbatch` builds
   CONFIGS from a `DATASETS=${1:-"HateMM MHC"}` arg (`run_one` kept BYTE-IDENTICAL to `enc3seed.sbatch`), so the
   EN drop is a **submit-time argument** (`"HateMM"`), leaving the hash-freeze intact. This makes the
   pre-declared branch executable without mutating a frozen artifact.

3. **DEV-3 (authored a dedicated EN diagnostic `vis_image_moved_probe.py` instead of pointing at the
   HateMM-hardcoded F58 script). Favorable / neutral.** The task names "the F58 script"
   (`hatemm_lora_stream_decomp.py`), but that script is HateMM-hardcoded (`DS_DIR="HateMM"`) and compares
   **LoRA vs frozen-Qwen**; the EN gate needs **vis-LoRA vs generic-LoRA on EN**. The authored probe imports
   `encoder_swap_geometry.py` **VERBATIM** (the exact functions F58's script imports — `load`, `build_modality`,
   `loo_knn`, `knn_vote`, `auc`) and pins F58's MOVED thresholds (+0.010/+0.005) verbatim ⇒ it **is** the F58
   machinery, correctly parameterized. Machinery validated this prereg (reproduces the frozen→generic-LoRA EN
   image FLAT).

4. **DEV-4 (image-MOVED anchor operator corrected — the single most important deviation). LOUD.** The recon
   pinned the anchor "generic-LoRA EN image train-LOO 0.659 / dev 0.695" **from the red-team §0 scratch probe**
   (`scratchpad/redteam_stream_topk_probe.py`, uncommitted, a **different** vote operator ~0.01–0.04 off the
   committed F58 `encoder_swap_geometry.py`; the red-team doc itself notes its probe matches F58 only "to
   ~0.01"). Since the pinned diagnostic runs the **committed** operator, I re-derived the anchor with THAT
   operator: generic-LoRA EN img **0.6236 train / 0.6756 dev** (frozen-Qwen 0.5992/0.6865, CLIP 0.7338/0.7367;
   §2.3). **The gate is unchanged** — it is a **same-operator DELTA** (vis and generic both read by the committed
   operator) with an **operator-independent threshold** (F58's +0.010/+0.005), so only the absolute anchor moves
   to the operator the diagnostic uses. The qualitative story holds bit-for-bit: F44 collapse reproduced
   (0.7338→0.5992), generic LLM-only LoRA leaves the EN image FLAT under the committed operator too
   (+0.0245 train but −0.0109 dev ⇒ dev fails to corroborate ⇒ FLAT).

5. **DEV-5 (kill-bar set follows the task, not the red-team's F55-oracle screen). Documented.** The red-team C1
   §(e) suggested an "F55 oracle-threshold screen (d_oracle ≥ +0.03)" as the GPU kill bar. The task's binding
   kill-bar list is K-V1 / K-V2 / EN-honesty / KS-regression / image-MOVED + eval_loss, which I pin verbatim.
   F55's EN oracle ceiling (+0.025 < +0.030) is carried as a **pre-declared prior damper** (F0.4 / §8), not a
   separate hard gate — consistent with both the task and the recon's honest-prior framing.

6. **DEV-6 (EN generic-adapter data provenance corrected). Favorable / clarifying.** `B4_FORENSIC_RECON.md §(ii)`
   described the 2026-07-02 `logging/lora/MHC` generic adapter's data as `train_yn.json` (549). The adapter
   `README.md` + `dataset_info.json` show it trained on the **word** dataset `mhc_lora_train` → `train.json`
   (549) — the same file the vis EN config's `dataset: mhc_lora_train` points to (unchanged mtime Jul-2). This
   **confirms** the EN vis arm trains on data byte-identical to the banked generic EN comparator (K-V2 clean
   superset holds on EN, not just HateMM). Had it truly been the `_yn` variant, answer-format would have
   confounded K-V2 — it does not.

7. **DEV-7 (ZH = GO-IF clause only, per the task). Documented.** The recon lists ZH as an "optional 3rd
   (off-mechanism)"; the task restricts this prereg to EN + HateMM and makes ZH a GO-IF clause triggered iff EN
   image-stream-MOVED fires (§3.8). No ZH config is authored or submitted here.
