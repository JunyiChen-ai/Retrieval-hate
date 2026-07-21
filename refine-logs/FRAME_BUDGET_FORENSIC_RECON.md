# FRAME-BUDGET CELL — FORENSIC RECON (red-team gap #6 / REDTEAM_UNTESTED_CELLS.md cell C2)

**Author:** frame-budget forensic-recon subagent (zero-GPU). **Date:** 2026-07-21 NZST.
**Discipline:** CPU-only. ZERO SLURM / GPU / Modal / job-submission / prereg / test-touch.
No `autoresearch/goal_mllm_plus3/state/` mutation. All token-budget numbers below are computed
from the on-disk Qwen2.5-VL processor math (CPU); all accuracy/timing anchors are copied verbatim
from banked logs/records (provenance §8).

**Mission.** 8 frames is hard-coded everywhere; 16/32-frame or denser temporal sampling is untested.
Produce GO/NO-GO + execution skeleton for the minimal decisive cell. This recon SHARPENS the
red-team's C2 (RANK 3, prior ~8-12%) by finding that the decisive first stage is **~6-8× cheaper and
strictly single-variable** compared to the red-team's full-chain ~6-8 GPU-h estimate — because the
**extractor has no `cutoff_len`** and the token-budget wall lives only in the *training* config.

**VERDICT: GO-IF.** GO on a ~1 GPU-h frozen-Qwen-16f HateMM stage-1 as a clean single-variable
door-closer; the expensive LoRA-16f stage-2 is **NO-GO unless stage-1 moves**. The "-IF" is
priority, not feasibility: prior is LOW-MODEST and the cell is **D7-novelty-dead** (sampling density
is engineering), so it ranks below C1 (vision-unfreeze) and C3 (learned-audio) and is worth its
~1 GPU-h only as a "measured-and-closed" conversion of a currently prose-argued gap.

---

## 1. THE MINIMAL DECISIVE CELL — TWO-STAGE STRUCTURE

The red-team priced C2 as one lumped ~6-8 GPU-h/dataset full chain (re-extract + re-SFT + head).
That conflates two structurally different tests. Split them:

### Stage 1 — frozen-Qwen-16f (CHEAP, DECISIVE, SINGLE-VARIABLE) — the actual cell to run

- **What changes:** exactly one variable, `--num_frames 8 → 16`, in the frozen extractor
  `src/utils/generate_VideoMLLM_embedding_HF.py` (arg already exists, L91-95, default 8). No SFT.
  No code edit. Pooled `img_feats`/`text_feats` object is **byte-for-byte the same operator** —
  mean of last-layer hidden states over the prefix span (img) / response span (text), L2-normed
  (`_encode` L290-322). Same RGCL head, same `enc3seed.sbatch` recipe, same seeds 0/1/2.
- **Why it is the decisive gate:** it isolates the *sampling-density* question from every other
  variable. If denser frames carry no extra label signal through the *frozen* encoder+pool, the
  whole cell is dead AND the expensive LoRA-16f follow-up is auto-dead (you cannot re-SFT your way
  into information the frozen forward proves is not in the denser frames — KS below).
- **Cost:** ~0.5-1.0 GPU-h extraction (HateMM) + ~1.5 min head (3 seeds × ~29 s). See §6.

### Stage 2 — LoRA-16f re-SFT (EXPENSIVE, CONTAMINATED, CONDITIONAL) — only if stage-1 moves

- **What changes:** `NUM_FRAMES 8→16` in `build_lora_sft_data.py:39` **AND** (forced, see §2) a
  `cutoff_len` raise `4096 → ~8192` in the SFT config — **so it is inherently ≥2 changed variables**
  and can never be a clean frame-budget test. Plus a full LoRA re-SFT + LoRA-16f re-extraction.
- **Cost:** ~4-6 GPU-h/dataset (SFT ~3-3.5 h + extraction ~0.5-1 h + head). **Do not spend this
  until stage-1 clears its continue-gate.**

**Bottom line:** the "frame-budget cell" that is worth defining and (if prioritized) running is
**frozen-Qwen-16f on HateMM, extraction + head only.** Everything else is a conditional follow-up.

---

## 2. MECHANICS / TOKEN-BUDGET FINDING (the load-bearing result)

Qwen2.5-VL-7B on-disk config: `patch_size=14`, `spatial_merge_size=2` (⇒ 1 visual token = a
28×28-px block), `temporal_patch_size=2` (⇒ video frames are merged in **pairs**: `grid_t =
num_frames/2`). Verified from
`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/.../config.json` +
`preprocessor_config.json` (`Qwen2VLImageProcessor`). Token counts below are the exact
`smart_resize` math (factor 28), CPU-computed.

**Finding A — the two code paths use DIFFERENT visual-token budgets, and only ONE has a wall.**

| path | how frames enter | per-frame pixel cap | visual tokens @8f | visual tokens @16f | ceiling |
|---|---|---|---|---|---|
| **Extractor** (`generate_VideoMLLM_embedding_HF.py`) | ONE `{"type":"video"}` turn, temporal-merged | `max_pixels=360*420=151200` | ~720–768 | ~1440–1536 | **NONE** — single frozen forward, no `cutoff_len`, no truncation arg; only VRAM |
| **SFT builder** (`build_lora_sft_data.py`) | eight separate `"<image>"*8`, **not** temporal-merged | `image_max_pixels=262144` (train yaml L3) | ~2496–2520 | **~4992–5040** | **`cutoff_len=4096`** (train yaml L21) |

- **Extractor has NO cutoff.** `processor(text=..., videos=[frames], return_tensors="pt")` passes no
  `max_length`/`truncation`; `model(**inputs, output_hidden_states=True)` is a single forward. At
  16f the sequence is only ~1.5k visual tokens + text — trivial for a 7B on the A100 that already
  runs 8f (~0.77k). **Frozen-16f fits with ZERO second changed variable and no VRAM concern.**
  Even 32f (~2880–3072 visual tokens) fits the extractor.
- **The SFT path is where 16 frames break the 4096 wall.** Because the builder emits 8 *separate
  images* (temporal merge does NOT apply to `<image>`), each image is ~312–315 tokens; 8 already
  ≈2.5k (+~500–700 text ⇒ fits under 4096), but **16 images ≈5.0k tokens BEFORE any text ⇒ exceeds
  cutoff_len=4096.** LoRA-16f-SFT therefore forces `cutoff_len ≈8192` (or an `image_max_pixels`
  cut) — a second changed variable that contaminates the comparison. **This is the mechanical
  reason stage-1 must be frozen-encoder-only.**

**Finding B — no code change needed for stage-1.** `--num_frames 16` is even ⇒ `grid_t=8`, clean;
`_sample_frame_indices` = `np.linspace(0, N-1, 16)` works for any count; the in-place
masked-scatter invariant `assert last_hidden.shape[0]==input_ids.numel()` (L283) holds. Videos with
<16 decodable frames yield duplicated indices (graceful; those clips degenerate toward 8f-equiv —
a mild, honest caveat, not a crash).

---

## 3. WHICH DATASET FIRST — HateMM

**HateMM first, decisively; EN deprioritized; ZH only as a stage-1.5 if HateMM moves.**

1. **Coverage argument is strongest where hate is image-borne.** HateMM's hate lives in in-frame
   symbols/gestures/on-screen-text (exactly what `IMG_INSTRUCTION` targets); 8 uniform frames can
   miss a brief hateful shot that 16 would catch. The red-team §0 stream table confirms HateMM has
   the **healthiest image stream** (train-LOO img AUC: CLIP 0.836 / frozen-Qwen 0.820), so denser
   visual coverage has a real conversion surface.
2. **The floor is solid and low-variance.** Frozen-Qwen-8f HateMM is banked at 3 seeds with a
   final-epoch mean **0.8682** (§5) — a clean paired anchor.
3. **EN is the WORST target (red-team §0 fact-1).** MHC-EN's image stream is *collapsed upstream of
   the LLM* (CLIP 0.745 → frozen-Qwen 0.653 train-LOO) and LLM-only LoRA does not repair it. More
   frames cannot heal a collapsed vision tower — feeding a broken encoder 16 frames yields 16 broken
   token-groups. Frame budget is mechanistically mis-aimed at EN.
4. ZH image stream is middling (CLIP 0.724 / frozen-Qwen 0.734); run it only if HateMM stage-1 moves.

---

## 4. $0 PRE-GATE — NONE; and the banked bound on the prior (applied honestly)

**No $0 pre-gate exists.** The frozen-16f `img_feats`/`text_feats` do not exist and cannot be
derived from banked 8f caches — denser sampling requires a new forward. I checked the two banked
things that could conceivably bound it:

- **S2S framesets are 8f only.** On disk: `data/CLIP_Embedding/{HateMM,MHC}/frameset_qwen7b_8f/`
  (T=4 = 8//2). The **16f sensitivity arm was CANCELLED unrun** (F37). And S2S framesets are
  *per-frame-group vectors* (a different pooling object), not the deployed mean-pooled `img_feats`,
  so they could not serve as a 16f proxy even if they existed.
- **P3 evidence-density** scored *segments* for weighted pooling; it varies weighting, not frame
  count — no bound on sampling density.

**Banked bounds that LOWER the prior (no ban, but honest priors):**

- **F37 + F35 (cumulative-causal redundancy).** F37's binding finding: on these cumulative-causal
  Qwen representations, "pooling is effectively lossless … per-segment matching adds no frame-local
  information," and F35 shows group vectors are causal *prefix summaries* (late frames already
  summarize early ones). This does NOT ban denser sampling (F35/F37 concern the *operator over a
  fixed 8f set*, not coverage — see §5 ban-scope check), but it is a real prior-lowering mechanism:
  the pooled mean over a denser prefix grid is partly redundant with the running summary.
- **Mean-pooling dilution tension.** `img_feats` is a MEAN over the whole visual span. A single
  hateful frame among 16 contributes half the weight it does among 8 — so denser sampling *raises*
  the chance of catching a brief event (coverage ↑) but *lowers* its pooled contribution once caught
  (dilution ↓). The two partly cancel; net sign is genuinely unknown, which is why it must be
  measured rather than argued.
- **Red-team §0 fact-1.** The missing-dataset bottleneck (EN) is encoder collapse, not frame count.

Net: no banked evidence RAISES the prior; two lower it. The red-team's **LOW-MODEST ~8-12%** for
≥+1pt on any dataset is honest and stands; I do not revise it up.

---

## 5. KILL-BARS SKELETON

**Comparison anchor (banked, verbatim):** frozen-Qwen-**8f** HateMM, RGCL head, `enc3seed.sbatch`,
seeds 0/1/2, dual protocol (job 12850). Final-epoch (ep29) test acc: seed0 0.8605 / seed1 0.8605 /
seed2 0.8837 ⇒ **mean 0.8682** (bit-matches F53's KS-2 line "frozen-Qwen final 0.8682"). Seed0
val-selected test acc 0.8698 (best-val ep28). Full 3-seed val-sel floor lives in
`exp-encoder-3seed.md` / F53 and is re-read verbatim at prereg time.

**Treatment:** frozen-Qwen-**16f** HateMM, identical head/seeds/protocol; **paired within seed**
(same head-init seed, CLIP-free — this is Qwen-16f vs Qwen-8f, an increment on the already-banked
encoder-swap, NOT vs CLIP).

**Ban-scope check (no collision — verified against F35/F37/F39 wording).** The pooled-embedding
object is unchanged, so none of the temporal-family bans reach it:
- **F37** killed the **retrieval-object / don't-pool / set-matching** family over the 8 frame groups
  (SET-POOLED vs POOLED). Frozen-16f is the *pooled* object, the surviving side of that verdict.
- **F39 (CTF)** killed the **supervised temporal-pool of the `[g_1..g_T]` frame-group tensor as a
  key** — a different object; frozen-16f never forms that tensor.
- **F35** is a mechanism note (causal-prefix), not a kill, and concerns how groups relate, not
  whether denser sampling covers more of the video.
⇒ **Same operator class as the deployed method; no ban collision.** (Matches red-team C2(b).)

**Bars.**
- **KILL / KS (auto-kills stage-2):** if frozen-16f **ties or regresses** the 8f floor —
  mean paired Δacc ≤ 0 OR bootstrap CI straddles 0 on **both** protocols — the cell is KILLED and
  **LoRA-16f is auto-dead** (re-SFT cannot manufacture signal the frozen forward shows denser frames
  do not carry). State this explicitly in the prereg.
- **CONTINUE-to-stage-2 gate (internal, cheap-side):** frozen-16f mean paired Δacc ≥ **+0.010**,
  3/3 sign-consistent, on ≥1 protocol — the minimum that would justify spending ~4-6 GPU-h on
  LoRA-16f. (Not a paper claim; a spend gate.)
- **FORMAL verdict bar (paper-worthy frame-budget effect):** house **+0.030 acc AND +0.030 mF1**
  conjunct, **3/3 seeds positive**, evaluated under **both** protocols vs the banked 8f floor —
  identical to the encoder-swap criterion (`exp-encoder-3seed.md` §pass-criterion).
- **Multiplicity:** pre-declare **16f as the single PRIMARY arm** (2× is the minimal density step).
  32f is a *secondary* arm only, pre-declared and gated on 16f moving, with a multiplicity note —
  never a free sweep (a {16,32} sweep uncorrected = forking path).

---

## 6. COST LEDGER · NAMING · D7

**Timing anchor (banked, S2S full extraction, same model+`max_pixels=151200`+sdpa+bf16, 8f):**
HateMM 3-split (744/107/215 = 1066 vids) = **949.1 s**; MHC 3-split (549/80/161 = 790) = 1519.1 s
(`S2S_EXTRACTION_RECORD.md`). Head = **~29 s / run** (frozen-Qwen trainlog tqdm `30/30 [00:29]`).

| item | scope | est. GPU-h | note |
|---|---|---|---|
| **Stage-1 extraction** frozen-16f | HateMM 3 splits | **~0.4–0.6** | 16f ≈ 1.5–2× the 949 s/8f (frame decode + 2× vision tokens); S2S does extra per-group algebra the plain extractor omits ⇒ upper bound generous |
| **Stage-1 head** | 3 seeds × dual protocol | **~0.03** | ~29 s × 3 ≈ 1.5 min; $0-class after extraction |
| **STAGE-1 TOTAL (HateMM)** | the decisive spend | **~0.5–0.7** | single changed variable, no cutoff, no ban collision |
| +ZH stage-1.5 (only if HateMM moves) | ZH 3 splits + head | ~0.6–0.9 | |
| **Stage-2 LoRA-16f** (CONDITIONAL) | per dataset | **~4–6** | SFT ~3–3.5 h + re-extract ~0.5–1 h + head; **≥2 changed variables (frames + cutoff 8192)**; only if stage-1 clears continue-gate |

**Collision-safe naming (no clobber of banked 8f caches):**
- Stage-1 extraction: `--num_frames 16 --out_model_tag Qwen2.5-VL-7B-Instruct_HF-16f`
  ⇒ `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF-16f.pt`
  (banked 8f is `..._HF.pt` — untouched). Head reads `--model Qwen2.5-VL-7B-Instruct_HF-16f`.
- Stage-2 (if reached): SFT frames `data/lora_frames_16f/<DS>/`, adapter `logging/lora_16f/<DS>/`,
  LoRA cache tag `Qwen2.5-VL-7B-Instruct-LoRA_HF-16f`, config `cutoff_len: 8192`.
- Job/group tags: `fb16_<DS>_...`, group `RAC_video_fb16`.

**D7 status — say it plainly.** Sampling density is an **engineering knob** (how many frames to feed
a fixed encoder+pool), **not** an MLLM-novelty mechanism. **Novelty-nil / D7-DEAD.** Even a formal
PASS is a *performance / ablation* row ("frame-budget: 16f vs 8f"), never a novelty contribution —
same D7 class as C4 (head-eng) and C5 (recipe). This is a door-closer + robustness ablation, not a
goal-reacher.

---

## 7. VERDICT — GO-IF

**GO-IF.**

- **GO** on **frozen-Qwen-16f HateMM, extraction + head only (~0.5–0.7 GPU-h)** as a *clean
  single-variable* door-closer. It is genuinely untested, carries no covering ban (F35/F37/F39 are
  scoped to operators over a fixed 8f set, not to sampling density), is far cheaper and cleaner than
  the red-team's lumped ~6-8 GPU-h estimate (because the extractor has no `cutoff_len`), and is
  aimed at the one dataset (HateMM) whose image stream is healthy enough to convert coverage.
- **The "-IF" is priority, not feasibility.** Prior is LOW-MODEST (~8-12%, honestly bounded by
  F37 cumulative-causal redundancy + mean-pool dilution + §0 EN-collapse) and the cell is
  **D7-novelty-dead**. So it ranks **below C1 (vision-unfreeze, ~12-15%, representation-class) and
  C3 (learned-audio, new channel)** and is worth its ~1 GPU-h only *if* the plan is to convert the
  two remaining "argued-in-prose, never-measured" representation cells into "measured and closed."
- **Stage-2 LoRA-16f is NO-GO** unless stage-1 clears the +0.010/3-sign continue-gate; if frozen-16f
  ties/regresses the 8f floor, the KS auto-kills the entire cell including the LoRA follow-up.

**Sequencing note (no submission made here — recon only, per mission).** If prioritized, this rides
the house queue-on-pass discipline: single-arm prereg (16f primary), 0-context review, hash-freeze,
then SLURM. It is strictly subordinate to C1/C3 in the red-team's top-3.

---

## 8. PROVENANCE

- **Cell source:** `refine-logs/REDTEAM_UNTESTED_CELLS.md` §C2 (RANK 3) + §0 stream table + top-5
  (adb8bc2); F61 (`findings.jsonl`).
- **Extractor / no-cutoff:** `src/utils/generate_VideoMLLM_embedding_HF.py:91-95` (`--num_frames`
  default 8), `:97-101` (`max_pixels=360*420`), `:264-323` (`_encode`, single forward, no
  truncation, in-place scatter assert L283), `:146-152` (`_sample_frame_indices`).
- **SFT cutoff wall:** `src/utils/build_lora_sft_data.py:39-46` (`NUM_FRAMES=8`, `IMG_TOKENS`),
  `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_sft.yaml:3` (`image_max_pixels
  262144`), `:21` (`cutoff_len 4096`).
- **Vision config:** `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/.../config.json`
  (`patch_size 14`, `spatial_merge_size 2`, `temporal_patch_size 2`) + `preprocessor_config.json`.
  Token counts = CPU `smart_resize` math (this recon).
- **Banked floor:** frozen-Qwen-8f HateMM `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_12850.trainlog`
  (final-ep acc 0.8605/0.8605/0.8837 ⇒ mean 0.8682; seed0 val-sel 0.8698); `research-wiki/experiments/exp-encoder-3seed.md`;
  F53 KS-2 line (6b8f634).
- **Ban scopes:** F35/F36 (S2S causal-prefix), F37 (retrieval-object/pooling-lossless, 16f arm
  cancelled), F39 (CTF supervised temporal-pool) — `findings.jsonl`.
- **Timing:** `refine-logs/S2S_EXTRACTION_RECORD.md` (HateMM 949.1 s / MHC 1519.1 s, 8f);
  head ~29 s from trainlog tqdm.
- **Banked 8f framesets (16f absent):** `data/CLIP_Embedding/{HateMM,MHC}/frameset_qwen7b_8f/`.
- CPU-only, zero GPU/Modal/SLURM/test-touch; no `state/` mutation. Committed on `main`, not pushed.
