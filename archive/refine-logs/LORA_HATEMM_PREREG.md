# LoRA-HateMM Pre-Registration — encoder-level LoRA-Qwen vs frozen-CLIP on HateMM (+ bundled B4-EN formal closure)

**Author:** LoRA-HateMM prereg author (CPU-only; no GPU/SLURM/Modal spent; NO job submitted).
**Date:** 2026-07-17.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced; no job submitted.
**Implements:** `refine-logs/LORA_HATEMM_FORENSIC_RECON.md` (commit `edeaedc`, the GO recon) — recipe, floors,
and kill-switches transcribed and re-verified verbatim below.
**House-style precedent:** `refine-logs/B3_PREREG_REVIEW.md` (ZH LoRA prereg + verdict), `research-wiki/experiments/exp-encoder-3seed.md`
(the 12850 encoder-swap protocol), `refine-logs/B4_FORENSIC_RECON.md` (EN closure).

## Title + claim scope (verbatim, per orchestrator instruction)

> This measurement **completes the performance evidence for terminus relaxation option (c)**; it makes **NO
> novelty claim — D7 (encoder-class novelty boundary) remains the user's ruling.**

The cell under test is the **encoder-level** LoRA-SFT-adapted Qwen2.5-VL-7B *encoder* on **HateMM**, features
fed to the standard archive-OFF RGCL align-fusion head + top-20 kNN (`enc3s`/12850 protocol), paired 3-seed
vs the banked **frozen-CLIP** floor, dual-protocol (val-selected AND final-epoch). Bundled arm: the **B4-EN**
LoRA-encoder cell (`dataset=MHC`), an expected-FAIL formal closure. This cell is genuinely UNMEASURED (no
HateMM LoRA adapter and no HateMM LoRA feature cache exist on disk); it is NOT a ~2-min formalization — it is
a ~3.5–4 A100-h open run (SFT + extraction + head).

---

## 0. Binding facts / honesty clauses (all present; pre-declared)

**F0.1 — Test is NOT virgin (declared).** HateMM test was already read by the frozen-CLIP and frozen-Qwen
12850 arms; MHC-EN test by the 12850 EN arms. This prereg's LoRA head reads are **re-measurements under the
identical protocol**, not first exposures. Each consumes exactly ONE budgeted LoRA-encoder test evaluation
(HateMM LoRA-encoder + MHC-EN LoRA-encoder). Zero test-touch before the independent verdict.

**F0.2 — Single-encoder-draw limitation (pre-declared, same as B3 §0.2).** The 3 head-seeds read ONE HateMM
LoRA-SFT encoder draw (head init + data-shuffle vary; the encoder is fixed). The reported ±band is
**head-seed variance, NOT LoRA-SFT encoder-draw variance.** An encoder-draw-stability claim would need ≥3
fresh SFT retrains (~9 h) — out of scope, pre-declared. The design is symmetric with the frozen-CLIP control
(also a single draw) ⇒ a legitimate head-level paired test.

**F0.3 — Novelty = D7, PENDING USER RULING (not decided here).** LoRA-SFT encoder adaptation is a
2024-25-standard technique (Axis-B). Whether a performance pass on this cell counts toward the goal's "novel"
clause is the user's D7 ruling. This prereg decides the **performance clause only.**

**F0.4 — Family/mechanism-divergence framing (pre-declared, material to D7).** A HateMM LoRA pass is expected
to be substantially **image-inherited** from the frozen-Qwen Pareto conversion (F44: HateMM image train-LOO
AUC 0.826, the decisive modality), whereas the ZH LoRA pass (B3) was **text-borne and LoRA-specific** (F45).
So "encoder-level LoRA passes on 2 datasets" would rest on **divergent underlying mechanisms** (ZH text /
HateMM image) — this nuance travels with any claim built on the result (KS-2 quantifies it).

**F0.5 — Single-dataset own-train-split VETO compliance (hard user veto).** Stage-1 SFT trains on
`data/lora_sft/HateMM/train.json` = **HateMM own train split ONLY** (743 records, word-label
hateful/normal target). NO cross-dataset mixing, NO gold spans/attributes, NO OCR channel, raw videos never
leave the machine. The bundled EN arm trains on MHC own train split only (its adapter already exists). This
is identical discipline to the MHC/MHC_zh adapters. All standing vetoes cleared.

**F0.6 — Two-regime disambiguation (why P9 does not pre-kill this cell).** P9's banked HateMM negative
(C3-knn −4.7 below floor) is the **decision-level** regime (`sft_classifier`, r128 α256, joint LM+binary-head
SFT, raw-kNN read-out, no trained fusion head). THIS cell is the **encoder-level** regime (`stage: sft`, pure
generative word-label SFT, r16 α32, features → a freshly-trained RGCL align-fusion head + kNN). The two are
proven non-isomorphic by their **opposite ZH behavior** (encoder-level ZH kNN +0.031 vs decision-level ZH
C3-knn −2.2). P9's HateMM datum is a different-regime result; it is folded into the prior as a tempering
yellow flag (KS-3), not a pre-kill.

---

## 1. Pipeline spec — fully pinned (3 stages, nothing left to interpretation)

### 1.1 Stage 0 — SFT data build + registration (CPU; DONE + verified idempotent this prereg)

- **Command (idempotent, pure CPU when frames cached):** `python src/utils/build_lora_sft_data.py --dataset HateMM`
  (default `--answer word` ⇒ hateful/normal target, matching the MHC encoder config). This is ALSO STEP 1 of
  `lora_sft.sbatch HateMM`, so it re-runs deterministically at submit time.
- **Materialized + registered on disk (this prereg re-ran the build on the login node — pure CPU, frames
  cached, 1 unreadable video skipped, seconds; snapshot→build→snapshot sha256 unchanged ⇒ idempotent):**

  | file | rows | sha256 |
  |---|---|---|
  | `data/lora_sft/HateMM/train.json` | 743 (297 hateful / 446 normal) | `93c6d3d1bffbca22b2dd8beba57a33575a48d8ca61d8d56e3148fecdbb93973a` |
  | `data/lora_sft/HateMM/val.json` | 107 (43 / 64) | `9e103ed35a014af81eb3aa6af0d51a28707efd66a606c5bf0459db570a9cc9ef` |
  | `data/lora_sft/HateMM/test.json` | 215 (86 / 129) | `c12ad356aa2917ed80ef17ba93e7854cd36751f770f05a3b19956cfbfdce8462` |
  | `RA-HMD/LLAMA-FACTORY-Ver202512/data/dataset_info.json` | keys `hatemm_lora_{train,val,test}` → the 3 word-variant files | `ebf14b472744b0ca2007695033026b9dde4538aa37ccf019b9482a1ab07681b5` |

  **DEVIATION FROM RECON (flagged loudly, favorable):** recon §2.1/§2.2 states `hatemm_lora_*` registration is
  MISSING. **It is NOT missing** — `dataset_info.json` already carries `hatemm_lora_{train,val,test}` (word
  variant, 743/107/215) AND `hatemm_lora_yn_{train,val,test}` (the decision-level `_yn` variant, unused by this
  cell), and `git status` shows `dataset_info.json` byte-unchanged vs HEAD after re-build. Net effect: Stage 0
  is a no-op confirm, not a build. The encoder config below correctly uses the **word** variant
  `hatemm_lora_train`/`hatemm_lora_val` (the MHC encoder precedent uses `mhc_lora_train`, the word variant;
  the `_yn` variant belongs to the P9 `sft_classifier` regime, NOT this cell).

### 1.2 Stage 1 — LoRA-SFT of the encoder (own HateMM train split only)

- **Submit:** `sbatch scripts/slurm/lora_sft.sbatch HateMM`
- **Config (authored this prereg, verbatim copy of `mhc_qwen25vl_lora_sft.yaml` with EXACTLY 3 changed lines —
  see §6):** `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_sft.yaml`
- **Recipe (pinned by the config, r16/α32/dropout0.0 verified against the MHC adapter):**
  base `Qwen/Qwen2.5-VL-7B-Instruct`; `stage: sft` (pure CAUSAL_LM generative, word-label hateful/normal);
  `finetuning_type: lora`; `lora_rank: 16`, `lora_alpha: 32`, `lora_dropout` 0.0 (LF default);
  `lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`; **`freeze_vision_tower: true`,
  `freeze_multi_modal_projector: true`** (vision + projector frozen — LoRA moves only the LLM backbone; this is
  the mechanism basis for F0.4); `learning_rate: 1.0e-4`, `num_train_epochs: 3.0`, `lr_scheduler_type: cosine`,
  `warmup_ratio: 0.05`, `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 8` (eff 8), `bf16: true`,
  `gradient_checkpointing: true`, 8-frame multi-image ShareGPT, `cutoff_len: 4096`, `save_strategy: epoch`,
  `eval_strategy: epoch`. Output adapter → `logging/lora/HateMM/` (does NOT exist yet; fresh SFT creates it).
- **Cost (anchored to MHC 549→204 steps→8222 s = 2.28 h):** HateMM 743 train, eff-bs 8, 3 ep ⇒ ⌈743/8⌉×3 = **279
  steps** ⇒ ~8222×279/204 ≈ **11,245 s ≈ 3.12 h GPU** (one A100). Wall time longer under `PENDING (JobHeldUser)`.

### 1.3 Stage 2 — feature extraction with the LoRA-merged encoder

- **Submit:** `sbatch scripts/slurm/gen_embed_lora.sbatch HateMM logging/lora/HateMM`
- **Runner is already dataset-generic (NO edit needed; sha `c76bb422…` unchanged):** loads frozen base
  Qwen2.5-VL-7B + `merge_and_unload` of the HateMM adapter, extracts 8-frame dual-stream last-token 3584-d
  img/text embeddings for all 3 splits, writes `{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`
  into `data/CLIP_Embedding/HateMM/` (DISTINCT tag from the frozen cache; never clobbers it), then B2-pushes.
- **Cost:** ~20–30 min GPU (1066 HateMM videos @8fr vs MHC 790 → ~1.35×).
- **Prereq confirmed:** the frozen-Qwen HateMM cache was extracted from `data/video/HateMM/` (implies videos
  present). The LoRA cache `data/CLIP_Embedding/HateMM/*LoRA*.pt` does NOT exist yet (verified) ⇒ no clobber.

### 1.4 Stage 3 — 3-seed RGCL align-fusion head + kNN (paired vs frozen-CLIP floor)

- **Submit:** `sbatch scripts/slurm/enc3seed_lora_hatemm.sbatch` (authored this prereg — see §6)
- **What it runs:** 6 head-only runs (features cached, ~20–25 s each): HateMM-LoRA seeds 0/1/2 **and** MHC-LoRA
  (EN) seeds 0/1/2, `--model Qwen2.5-VL-7B-Instruct-LoRA_HF`, `--group_name RAC_video_lora_hm`, `--force False`.
- **CRITICAL same-code guarantee:** the `run_one` python command in `enc3seed_lora_hatemm.sbatch` is
  **BYTE-IDENTICAL** to the 12850 runner `enc3seed.sbatch` (verified by `diff` this prereg — see §4.2); the
  ONLY manipulated variables vs the 12850 CLIP control are `--model` (CLIP→LoRA) and `--group_name` (fresh).
  Config: `--batch_size 64 --lr 0.0001 --epochs 30 --topk 20 --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1
  --fusion_mode align --hard_negatives_loss True --no_hard_negatives 1 --seed {0,1,2} --metric cos --loss triplet
  --hybrid_loss True --warmup 5 --lambda_seg 0 --archive OFF (archive_feats=None)`. Identical to
  `exp-encoder-3seed.md` H1.
- **Cost:** ~2 min GPU total (6 runs × ~20–25 s).

**Total NEW GPU: ~3.5–4.0 A100-h** (SFT ~3.1 h dominates; extract ~0.4 h; head ~0.03 h). EN arm marginal cost
~1 min (adapter + cache already on disk).

---

## 2. Comparison floors — RE-VERIFIED from primary 12850 trainlogs (numeric-provenance discipline)

**Every number below was independently re-parsed this prereg** from the raw `slurm/logs/enc3s_*_12850.trainlog`
(and reused EN-Qwen `arcbase_..._1227{5,6}.trainlog`) with the EXACT sbatch parser (val-sel = epoch ≥ warmup 5
max Val_Retrieval acc, roc tie-break; final = max epoch). All match the recon §3.4 to 4dp.

### 2.1 HateMM floors

| floor | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | **3-seed mean acc / F1** |
|---|---|---|---|---|---|
| **frozen-CLIP (PRIMARY — KS-1 pairs vs this)** | val-sel | 0.8279/0.8172 | 0.8279/0.8163 | 0.8047/0.7920 | **0.8202 / 0.8085** |
| **frozen-CLIP (PRIMARY)** | final-ep | 0.8186/0.7997 | 0.8047/0.7822 | 0.8140/0.7988 | **0.8124 / 0.7936** |
| **frozen-Qwen (SECONDARY — KS-2 pairs vs this)** | val-sel | 0.8698/0.8606 | 0.8651/0.8586 | 0.8837/0.8753 | **0.8729 / 0.8648** |
| **frozen-Qwen (SECONDARY)** | final-ep | 0.8605/0.8507 | 0.8605/0.8514 | 0.8837/0.8753 | **0.8682 / 0.8591** |

(Context, verified: frozen-Qwen−CLIP paired pass val-sel +0.0527 acc/+0.0563 F1, final +0.0558/+0.0656, 3/3.
The "0.870/0.861" memory shorthand = frozen-Qwen val-sel mean 0.8729/0.8648. CLIP floor per ERRATUM 66012e9.)

### 2.2 MHC-EN floors (for the bundled B4-EN arm)

| floor | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | **3-seed mean acc / F1** |
|---|---|---|---|---|---|
| **frozen-CLIP (EN PRIMARY — EN KS-1 pairs vs this)** | val-sel | 0.7826/0.7113 | 0.7329/0.6034 | 0.7702/0.6997 | **0.7619 / 0.6715** |
| **frozen-CLIP (EN PRIMARY)** | final-ep | 0.7640/0.7145 | 0.7826/0.7159 | 0.7888/0.7303 | **0.7785 / 0.7202** |
| **frozen-Qwen (EN SECONDARY, context)** | val-sel | 0.7888/0.7378 | 0.7826/0.7283 | 0.7702/0.6997 | **0.7805 / 0.7219** |
| **frozen-Qwen (EN SECONDARY, context)** | final-ep | 0.8012/0.7596 | 0.7702/0.7203 | 0.7826/0.7475 | **0.7847 / 0.7425** |

Provenance (file:line, from `exp-encoder-3seed.md` §"Numeric provenance" + re-parsed this prereg):
`enc3s_HateMM_openai_clip-...-336_HF_seed{0,1,2}_12850.trainlog`,
`enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_12850.trainlog`,
`enc3s_MHC_openai_clip-...-336_HF_seed{0,1,2}_12850.trainlog`,
`enc3s_MHC_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog` +
`arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2}_1227{5,6}.trainlog`.

---

## 3. Decision rule (verbatim from `exp-encoder-3seed.md:73-85`) + kill-switches (verbatim from recon)

### 3.1 Decision rule — transcribed verbatim (treatment = LoRA-Qwen; delta = LoRA − CLIP)

> For each dataset × protocol:
> 1. **Per-seed paired difference** delta = (LoRA − CLIP) for acc and macro-F1 at seeds 0/1/2.
> 2. **3-seed mean ± std** of the paired delta; **sign consistency** (how many of 3 seeds positive).
> 3. n=3 is too small for a formal bootstrap; report the paired-t statistic **as an effect-size descriptor
>    only** alongside the mean/std and sign count — no significance claim is made from n=3.
> 4. **Pass criterion (per dataset × protocol):** mean paired delta_acc ≥ +0.030 AND mean paired delta_mF1 ≥
>    +0.030 AND sign consistency 3/3 positive.
> 5. **Headline claim:** requires the pass criterion met on ≥ 2 datasets under a stated protocol. Each protocol
>    is judged separately; the verdict is written exactly as "final-epoch: pass/fail; val-selected: pass/fail".

Both protocols judged **independently** (no protocol-shopping, no metric-shopping). Outcome categories
pre-declared in §8.

### 3.2 KS-1 — PERFORMANCE CONJUNCT (primary kill; verbatim from recon §3.4)

LoRA−CLIP paired: **mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND sign 3/3**, judged **independently under
EACH protocol separately** (val-selected AND final-epoch). HateMM CLIP floors: **val-sel 0.8202/0.8085,
final 0.8124/0.7936**. Below the conjunct under a protocol → **NEGATIVE** on that protocol (encoder-level LoRA
does not generalize to HateMM) — a valid, informative kill outcome.

### 3.3 KS-2 — FAMILY-COHERENCE HONESTY FLAG (NOT a performance kill; verbatim from recon §3.4)

Compare LoRA vs the frozen-Qwen floor: **val-sel 0.8729/0.8648; final 0.8682/0.8591**. If **LoRA < frozen-Qwen
− 0.014** (the seed band), pre-declare: *"on HateMM the best encoder remains frozen-Qwen; the LoRA pass is
image-inherited, not LoRA-driven."* This **weakens (does not break)** the single-lever family narrative and is
material to D7. **LoRA ≥ frozen-Qwen strengthens it** (LoRA = best HateMM encoder). This flag does NOT change
the KS-1 pass/fail.

### 3.4 KS-3 — REGIME SANITY / P9 CROSS-CHECK (verbatim from recon §3.4)

If LoRA-HateMM lands **below the CLIP floor** (echoing P9's decision-level C3-knn −4.7), the encoder-level
regime failed to convert on HateMM despite converting on ZH → bank as the "encoder-level LoRA is ZH-specific
too" negative (a P9-echo).

### 3.5 EN arm (bundled B4-EN) — same rule vs EN CLIP floor

EN treatment = MHC-LoRA-Qwen; delta = LoRA − CLIP vs EN CLIP floors (§2.2): **val-sel 0.7619/0.6715, final
0.7785/0.7202**. **Honest prior: FAIL both protocols** (B4 seed0 anchor: val-sel −0.031 acc, final +0.006 acc;
LoRA below both frozen floors on EN). This formally closes the EN LoRA-encoder cell (the 22nd negative) — it
opens no new ground; it is included only for a fully-formal 3-dataset encoder-level LoRA matrix (ZH pass-marginal
/ EN fail / HateMM new).

---

## 4. G-repro (adapted — no bit-exact anchor; first LoRA draw) + smoke plan

### 4.1 G-repro discipline (adapted from recon §3.4; a first encoder draw has no prior anchor)

- **(a) SFT smoke gate (Stage 1).** A tiny SFT smoke (few steps) must show: **loss finite (no NaN), loss
  decreasing, a checkpoint written**, and the recipe pattern matching the MHC precedent (MHC anchor
  `logging/lora/MHC/all_results.json`: **eval_loss 0.1620, train_loss 0.0964, train_runtime 8222 s, 204 steps,
  epoch 2.96**; a full HateMM SFT should land eval_loss in the ~0.12–0.18 band). A NaN/exploding/flat loss aborts
  before extraction.
- **(b) Head runs = SAME-CODE as 12850.** The Namespace diff between a LoRA head run and the 12850 CLIP control
  MUST be `--model` + derived-inert fields (`exp_comment`, `group_name`, `output_path`) ONLY. The `run_one`
  python block is byte-identical to `enc3seed.sbatch` (§4.2). This retires the code-version confound the same way
  `exp-encoder-3seed.md:126-146` did for the archive-OFF path.
- **(c) frozen-CLIP control re-paired from 12850** (code-stable, verified in §2). Not re-run.

### 4.2 Same-code verification (run this prereg — PASS)

`diff` of the `run_one`-through-`PY` block: `enc3seed_lora_hatemm.sbatch` == `enc3seed.sbatch` **BYTE-IDENTICAL**.
Full-file `diff` vs the B3 precedent `enc3seed_zh_b3.sbatch`: differs ONLY in the header comment, the `CLIP=`
breadcrumb comment, `GROUP_NAME`, and the `CONFIGS` rows. `bash -n` on both `enc3seed_lora_hatemm.sbatch` and the
edited `lora_sft.sbatch` = SYNTAX_OK.

### 4.3 Collision safety (verified this prereg; re-check at submit time)

- `logging/lora/HateMM` — does NOT exist ⇒ fresh SFT creates it (no clobber of MHC/MHC_zh adapters).
- `data/CLIP_Embedding/HateMM/*LoRA*.pt` — does NOT exist ⇒ fresh extraction, frozen HateMM cache untouched.
- `logging/Retrieval/{HateMM,MHC}/RAC_video_lora_hm*` — do NOT exist ⇒ fresh group, `force=False` never trips
  the `run_rac.py:904-908` hard-abort; NO 12850 arm overwritten (`exp_name` is seed+model-derived, and the LoRA
  model tag differs from CLIP/Qwen, so dirs are distinct regardless).
- `slurm/logs/enc3s_{HateMM,MHC}_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed*_*.trainlog` — do NOT exist ⇒ no trainlog
  collision.

### 4.4 Smoke plan (executor runs BEFORE the real submits; leave no artifact that trips §4.3)

1. **SFT smoke:** launch `hatemm_qwen25vl_lora_sft.yaml` with a few steps (e.g. `max_steps: 20`,
   `save_steps: 20`, throwaway `output_dir: logging/lora/_smoke_hatemm`) — confirm loss finite/decreasing, ckpt
   written; then delete the smoke dir. (Do NOT smoke-write into `logging/lora/HateMM`.)
2. **1-seed head smoke:** on the EN LoRA features that already exist (`data/CLIP_Embedding/MHC/*LoRA*.pt`), run
   ONE `run_rac.py` head with a throwaway `--group_name _smoke` to confirm the align-fusion path loads the LoRA
   cache and completes 30 epochs; then delete the `_smoke` dir. (The HateMM LoRA cache does not exist until
   Stage 2, so the head-code smoke uses the existing EN cache.)
   If in doubt, skip the smokes — cache dims and the same-code guarantee are already CPU-verified.

---

## 5. Bundled B4-EN formal closure (recon §4)

Included as 3 extra rows (`MHC $LORA {0,1,2}`) in `enc3seed_lora_hatemm.sbatch` (§6). The EN adapter
(`logging/lora/MHC/`) and EN LoRA feature cache (`data/CLIP_Embedding/MHC/*LoRA*.pt`) already exist ⇒ head-only,
~1 min. Its own CLIP floor is §2.2. Expected-FAIL prior (§3.5); formal closure only.

---

## 6. Artifacts authored this prereg + hash freeze block

### 6.1 New / edited artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current) |
|---|---|---|---|
| A | `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_sft.yaml` | **NEW** — verbatim copy of `mhc_qwen25vl_lora_sft.yaml`, EXACTLY 3 lines changed: L18 `dataset: mhc_lora_train`→`hatemm_lora_train`; L19 `eval_dataset: mhc_lora_val`→`hatemm_lora_val`; L27 `output_dir: …/lora/MHC`→`…/lora/HateMM` | `d2f415cd93fa6f7b439fd4b4573a536baf48ad42186dc8bd50f3fab20553e36a` |
| B | `scripts/slurm/lora_sft.sbatch` | **EDITED** — added `HateMM)` case (CONFIG+OUTDIR), usage comment `{MHC, MHC_zh}`→`{MHC, MHC_zh, HateMM}`, error string `MHC or MHC_zh`→`MHC, MHC_zh or HateMM`. No other lines touched. | `e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f` |
| C | `scripts/slurm/enc3seed_lora_hatemm.sbatch` | **NEW** — copy of `enc3seed_zh_b3.sbatch` (`run_one`/parser/loop/b2-push byte-identical, hence byte-identical to `enc3seed.sbatch` `run_one`); differs only in header comment, `GROUP_NAME=RAC_video_lora_hm`, and CONFIGS (3 HateMM-LoRA + 3 MHC-LoRA rows) | `19c76b177f7dc883a9e03524450ad2e6cb302cdd0a6704d69da68a62188a06fc` |

### 6.2 Reused-unchanged machinery (verify sha at submit time; do NOT edit)

| path | role | sha256 |
|---|---|---|
| `scripts/slurm/gen_embed_lora.sbatch` | extraction (dataset-generic; NO edit needed) | `c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386` |
| `src/utils/build_lora_sft_data.py` | Stage-0 builder (supports `--dataset HateMM`) | *(unchanged; not re-hashed — it is not a decision gate)* |
| `RA-HMD/.../my_configs/hatevideo/mhc_qwen25vl_lora_sft.yaml` | source of the verbatim copy | `db371c18f306c5a3a00eeef8550964c3ddacf9e20400439324009ef2e69b1b52` |
| `scripts/slurm/enc3seed.sbatch` | 12850 runner (same-code anchor for §4.2) | `dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d` |
| `scripts/slurm/enc3seed_zh_b3.sbatch` | B3 head-runner template | `4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad` |

### 6.3 Data artifacts (registered word-variant + dataset_info): see §1.1 table (4 shas).

### 6.4 Hash-freeze (to be filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file LORA_HATEMM_PREREG.md, after review>
A <sha256 hatemm_qwen25vl_lora_sft.yaml>
B <sha256 lora_sft.sbatch>
C <sha256 enc3seed_lora_hatemm.sbatch>
```
Executor re-runs `sha256sum` on A/B/C (and this file) at submit time; any mismatch = authorization VOID.

---

## 7. Single-submit / execution plan + resource plan

**Order (3 sequential SLURM jobs; each after the prior COMPLETED, or chained via `--dependency=afterok:`):**

1. `sbatch scripts/slurm/lora_sft.sbatch HateMM` → produces `logging/lora/HateMM/` adapter (~3.1 h GPU).
   Gate: SFT smoke (§4.4.1) BEFORE this real submit; on COMPLETE, apply the G-repro SFT-loss sanity (§4.1a).
2. `sbatch scripts/slurm/gen_embed_lora.sbatch HateMM logging/lora/HateMM` → produces
   `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` (~20–30 min GPU).
3. `sbatch scripts/slurm/enc3seed_lora_hatemm.sbatch` → 6 head runs (HateMM + EN LoRA), ~2 min GPU. Produces
   `slurm/logs/enc3s_{HateMM,MHC}_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_<JID>.trainlog`.

**Resource plan:** 1×A100; `conda activate HateVideo`; SLURM via `sbatch` with **NO `--time`** flag; initial
`PENDING (JobHeldUser)` = **WAIT for auto-release, never force-release** (CLAUDE.md). Each sbatch already sources
`conda.sh` directly (not `~/.bashrc`) and sets an absolute interpreter + import env (`HF_HUB_OFFLINE`,
`DISABLE_VERSION_CHECK`, `CUDA_HOME` shim) — the PATH-prepend trap is handled. `lora_sft.sbatch` has a ≥20 G
disk guard.

**Test-touch:** the Stage-3 LoRA head reads are the ONLY budgeted LoRA-encoder test evaluations (HateMM + EN);
zero test-touch before the verdict. **The executor transcribes raw both-protocol per-seed numbers (line-numbered)
and applies NO gates/interpretation** — the verdict (G-repro → Namespace-diff → KS-1/2/3 → decision rule under
both protocols) is rendered by an **independent 0-context reviewer against this prereg VERBATIM.**

**No job is submitted by this prereg author.** Submission happens only after the independent review + hash-freeze
(run by the orchestrator).

---

## 8. Outcome table template (all cells filled ONLY from raw trainlogs at verdict time)

### 8.1 HateMM — LoRA-Qwen vs frozen-CLIP (fill from `enc3s_HateMM_...-LoRA_HF_seed{0,1,2}_<JID>.trainlog`)

| seed | protocol | LoRA acc/F1 | CLIP acc/F1 (§2.1) | Δacc | ΔF1 |
|---|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8279/0.8172 | ___ | ___ |
| 1 | val-sel | ___ | 0.8279/0.8163 | ___ | ___ |
| 2 | val-sel | ___ | 0.8047/0.7920 | ___ | ___ |
| **mean** | **val-sel** | ___ | **0.8202/0.8085** | **___** | **___** |
| 0 | final-ep | ___ | 0.8186/0.7997 | ___ | ___ |
| 1 | final-ep | ___ | 0.8047/0.7822 | ___ | ___ |
| 2 | final-ep | ___ | 0.8140/0.7988 | ___ | ___ |
| **mean** | **final-ep** | ___ | **0.8124/0.7936** | **___** | **___** |

KS-1 verdict (per protocol): val-selected: PASS/FAIL · final-epoch: PASS/FAIL (+MARGINAL note if within noise).
KS-2 (vs frozen-Qwen 0.8729/0.8648 val-sel; 0.8682/0.8591 final): LoRA ≥ frozen-Qwen? ___ (family flag).
KS-3 (below CLIP floor?): ___ .

### 8.2 MHC-EN — LoRA-Qwen vs frozen-CLIP (bundled B4 closure; fill from `enc3s_MHC_...-LoRA_HF_seed{0,1,2}`)

| seed | protocol | LoRA acc/F1 | CLIP acc/F1 (§2.2) | Δacc | ΔF1 |
|---|---|---|---|---|---|
| 0 | val-sel | ___ | 0.7826/0.7113 | ___ | ___ |
| 1 | val-sel | ___ | 0.7329/0.6034 | ___ | ___ |
| 2 | val-sel | ___ | 0.7702/0.6997 | ___ | ___ |
| **mean** | **val-sel** | ___ | **0.7619/0.6715** | **___** | **___** |
| 0 | final-ep | ___ | 0.7640/0.7145 | ___ | ___ |
| 1 | final-ep | ___ | 0.7826/0.7159 | ___ | ___ |
| 2 | final-ep | ___ | 0.7888/0.7303 | ___ | ___ |
| **mean** | **final-ep** | ___ | **0.7785/0.7202** | **___** | **___** |

EN verdict: val-selected: PASS/FAIL · final-epoch: PASS/FAIL (expected FAIL both — 22nd negative formal closure).

### 8.3 Fixed write-up format (per §3.1 rule 5)

`HateMM: final-epoch: <pass/fail>; val-selected: <pass/fail>.` · `MHC-EN: final-epoch: <pass/fail>; val-selected: <pass/fail>.`

---

## 9. What a PASS / FAIL means for the goal (D7 boundary — this prereg does NOT decide)

- **PASS (recon prior ~75–85%):** the goal's performance conjunct (+0.03/+0.03 on ≥2 datasets) is met by **one
  lever — encoder-level LoRA** (ZH marginal + HateMM). Honest caveat (KS-2/F0.4): the two passes convert via
  **different modalities** (ZH text-borne LoRA-specific; HateMM image-borne inherited-from-frozen). Whether this
  satisfies the goal's "novel" clause — and whether LoRA-SFT encoder adaptation counts at all — is the user's
  **D7 ruling, not decided here.**
- **FAIL:** banks the 23rd/24th pre-registered negative (encoder-level LoRA is ZH-specific); HateMM's only formal
  encoder pass remains the **frozen** swap.

---

## 10. Provenance index

- Recon (GO, recipe/floors/kill-switches): `refine-logs/LORA_HATEMM_FORENSIC_RECON.md` (`edeaedc`).
- Protocol + 12850 floors: `research-wiki/experiments/exp-encoder-3seed.md` (H1, decision rule :73-85, provenance
  :243-263); ERRATUM `66012e9` (HateMM CLIP floor 0.8279/0.8172).
- House style: `refine-logs/B3_PREREG_REVIEW.md`, `refine-logs/B4_FORENSIC_RECON.md`.
- SFT recipe/anchor: `RA-HMD/.../mhc_qwen25vl_lora_sft.yaml`, `logging/lora/MHC/{adapter_config.json,all_results.json}`.
- Extraction: `scripts/slurm/gen_embed_lora.sbatch`, `src/utils/generate_VideoMLLM_embedding_lora_HF.py`.
- Two-regime / P9: `refine-logs/B4_FORENSIC_RECON.md:127-150`, `research-wiki/EXP_p9_lmm_rgcl_video.md`.
- Mechanism: `refine-logs/ENCODER_SWAP_DIAGNOSIS.md` (F44), `refine-logs/B3_ZH_LORA_DECOMPOSITION.md` (F45).

**Required statements:** ZERO GPU/SLURM/Modal spent by this prereg author (only a pure-CPU login-node data-build
confirm, frames cached, seconds). No held-out test metric produced. All floor numbers re-parsed from banked
completed-run trainlogs (numeric-provenance discipline). No `state/` mutated. NO job submitted. Not pushed.
