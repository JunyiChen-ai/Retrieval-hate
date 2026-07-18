# CAND-2 Curriculum LoRA-SFT — DRAW-2 REPLICATION Pre-Registration (HateMM only)

**Author:** cand-2 DRAW-2 replication recon+prereg author (CPU-only; NO GPU/SLURM/Modal spent; NO job submitted).
**Date:** 2026-07-18.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced; no job submitted;
not pushed.
**Replicates:** the ONE live novelty-bearing positive of the project — the **F56 HateMM K-C2-2 add-over-generic
PASS** (`refine-logs/CAND2_VERDICT_REVIEW.md`, commit `546acc5`: HateMM curriculum-LoRA − generic-LoRA **val-sel
mean +0.0155 acc / +0.0166 mF1, sign 3/3 positive ⇒ K-C2-2 PASS**), which carries the frozen draw-1 prereg's own
**F0.2 single-curriculum-draw caveat** (`refine-logs/CAND2_CURRICULUM_PREREG.md` §0.2, frozen sha
`e5a689d9…f939790e`).
**House-style precedent:** `refine-logs/CAND2_CURRICULUM_PREREG.md` (the draw-1 prereg — recipe/floors/freeze
block/decision rule inherited verbatim), `refine-logs/CAND2_VERDICT_REVIEW.md` (the F56 numbers + how draws were
defined), `research-wiki/experiments/exp-encoder-3seed.md:73-85` (the enc3seed protocol + readout parser).

## Title + claim scope (verbatim)

> This measurement is a **replication test of the F56 HateMM val-selected add-over-generic effect ONLY.** It runs
> a **second, independent curriculum-SFT draw** (a different SFT seed over the byte-identical curriculum multiset)
> and asks whether the draw-1 K-C2-2 val-sel PASS **reproduces**. **No new method claim is made; the D7
> memory→adaptation-coupling novelty sub-ruling remains the USER's** (draw-1 prereg F0.3). This prereg decides only
> whether the F56 signal **hardens** (replicates) or **honestly retires** (fails to replicate / reverses).

**Why this is the cheapest decision-relevant GPU work in the box:** the D7 dossier's branch-B condition is
half-met partly *on the F0.2 single-draw caveat*. A second independent draw converts that caveat from a
disclaimer into a measurement — **signal hardens ⇒ the one positive is real; signal fails ⇒ it honestly
retires** — at ~3.5–4 A100-h (one SFT + one extraction + 3 cached head reads on **HateMM only**). **ZH is NOT
re-run** (draw-1 tied K-C2-2 on *both* protocols — F0.7 outcome — so a ZH replication would test a null; §F-R0.5).

---

## 0. Binding facts / honesty clauses (all pre-declared)

**F-R0.1 — What "an independent draw" IS (the load-bearing finding of this recon).** The curriculum **data**
builder `src/utils/build_curriculum_sft_data.py` is **deterministic and RNG-free** (largest-remainder
apportionment; draw-1 prereg DEV-3; verified bit-exact idempotent — `train_curric.json` sha `73307ef2…` reproduces
on every run). So a second *curriculum* draw **cannot** come from re-running the builder (it re-emits identical
bytes). The only knob that produces a genuinely **independent SFT draw of the same curriculum** is the
**LLaMA-Factory SFT seed**, which feeds `transformers.set_seed(training_args.seed)`
(`RA-HMD/LLAMA-FACTORY-Ver202512/src/llamafactory/hparams/parser.py:474`) and thereby governs **(a) the LoRA-A
adapter weight initialization** (kaiming random; B init to zero) **and (b) the training dataloader shuffle order**
over the 3 epochs. Reseeding draws a fresh sample from the *(curriculum-conditioned)* SFT-outcome distribution —
which is **exactly the "SFT-draw luck" that draw-1's F0.2 says the 3 head-seeds cannot separate from the
curriculum effect.** A second SFT seed = a second independent curriculum draw. **This is the correct and only
definition of an independent draw here; the curriculum weights are NOT varied (they are the fixed treatment).**

**F-R0.2 — Draw-1's seed was the implicit HF default 42 (verified).** The frozen draw-1 HateMM curric config
(`hatemm_qwen25vl_lora_curric_sft.yaml`, sha `c12c2b6b…`) pins **no `seed:` line** ⇒ it trained under the HF
`Seq2SeqTrainingArguments` default `seed = 42` (verified by importing the args dataclass under `HateVideo`).
Draw-2 pins an **explicit `seed: 1`** — the single manipulated experimental knob. (Repo precedent: the P9 configs
already use a top-level `seed:` 0/1/2 as their multi-seed knob, confirming this is the project's SFT-seed lever.)

**F-R0.3 — The ONLY changes vs the frozen draw-1 artifacts (enumerated).** The rep2 config
(`hatemm_qwen25vl_lora_curric_sft_rep2.yaml`, sha `d645de31…`) differs from draw-1's config C by exactly:
(1) **`seed: 1` added** — the one experimental variable that changes the trained weights; and (2)
**`output_dir` → `logging/lora/HateMM_curric_rep2`** — an **infrastructure line with no effect on the trained
weights** (collision-avoidance only, so draw-1's banked adapter is never clobbered), exactly parallel to how
draw-1's curric config was declared to differ from the generic arm by `dataset` + `output_dir`. A short comment
block documents the seed. **The training data (`hatemm_lora_curric_train` → `train_curric.json` sha `73307ef2…`),
the recipe (r16/α32, vision+proj frozen, lr1e-4/3ep/cosine/eff-bs8, 8-frame ShareGPT, cutoff 4096), the
extraction path, and the head code are byte-identical to draw-1.** A CLI `--seed` override is NOT an option:
LLaMA-Factory's `read_args` (`parser.py:73`) loads the *entire* config from the yaml via OmegaConf when
`argv[1]` ends in `.yaml` and **ignores extra CLI args** — the seed must be baked into a pinned rep2 yaml.

**F-R0.4 — Claim scope: replication of a performance sub-result, not a new claim.** A draw-2 PASS **hardens** the
F56 HateMM val-sel add-over-generic effect (raises it from single-draw to two-draw); a draw-2 FAIL **retires** it
honestly. **Neither outcome decides D7 novelty or goal satisfaction (the USER's, draw-1 F0.3/§8).** Even a PASS is
still only **2 SFT draws total** — not the ≥3-fresh-retrain curriculum-draw-stability study draw-1 F0.2 named as
out-of-scope; "hardened" means "survived one independent replication," not "proven stable" (F-R0.9).

**F-R0.5 — No ZH re-run (pre-declared rationale).** Draw-1's ZH K-C2-2 was **TIE on both protocols** (val-sel
mean −0.0067 acc sign 1/3; final-ep +0.0067 acc sign 2/3 — `CAND2_VERDICT_REVIEW.md` §3, the F0.7 "generic LoRA
with reshuffled data" outcome on the primary leg) and ZH-robustness was **not strengthened**. There is no ZH
positive to replicate; a ZH draw-2 would re-measure a null at extra GPU cost. **ZH is out of scope. HateMM only.**

**F-R0.6 — Test is NOT virgin (declared).** HateMM test was already read by frozen-CLIP (12850), generic-LoRA
(13235), and the draw-1 curriculum arm (13241). Draw-2's 3 head reads are **re-measurements under the identical
protocol**, consuming exactly ONE budgeted **HateMM-rep2-curriculum-encoder** test evaluation. Zero test-touch
before the independent verdict.

**F-R0.7 — Same protocol split as F56 (val-sel is binding).** F56's K-C2-2 PASS landed on the **val-selected**
protocol; the final-epoch leg was a **+0.0093 near-miss TIE** (below +0.010 by 0.0007). Therefore the **binding**
replication bar (K-REP-1) is on **val-sel**; draw-2's **final-epoch** read is **reported alongside but NON-binding**
(a +0.0093 draw-1 tie is inside noise and is not the effect under replication).

**F-R0.8 — Class-balance shift travels (unchanged from draw-1 F0.8).** The HateMM confusion-weighting shifts SFT
class balance (40.1%→37.7% hateful); identical multiset ⇒ identical shift. Pre-declared, not a new confound.

**F-R0.9 — Two-draws-total limitation (pre-declared).** This design yields **2 SFT draws** (draw-1 seed 42 +
draw-2 seed 1). Even a pooled hardened result is a 2-point estimate of curriculum-draw variance; it does not
license a "curriculum is stable" claim, only "the F56 effect replicated once independently."

---

## 1. Design — HateMM curriculum SFT draw-2 (3 stages; nothing left to interpretation)

### 1.1 Stage 1 — curriculum LoRA-SFT, draw-2 seed (HateMM own train split only)
- **Submit:** `sbatch scripts/slurm/lora_sft_curric_rep2.sbatch` (HateMM hardcoded; no DATASET arg).
- **Config:** `…/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft_rep2.yaml` (sha `d645de31…`) — draw-1's
  config C **+ `seed: 1` + rep2 `output_dir`** (F-R0.3); every other line byte-identical.
- **STEP 1a/1b (idempotent, $0 CPU, in-job):** re-run `build_lora_sft_data.py --dataset HateMM` (generic frames +
  `train.json`, cached) then `build_curriculum_sft_data.py --dataset HateMM --mode softconf`. The RNG-free builder
  **re-emits `train_curric.json` bit-exact** — this is the G-repro proof (§4b) that draw-2 trains the **identical**
  curriculum multiset. (It also re-writes `refine-logs/CAND2_KC20_HateMM.json` bit-identically — idempotent, sha
  `14967d53…`; benign.)
- **STEP 2:** `python src/train.py <rep2 config>` → adapter into `logging/lora/HateMM_curric_rep2/` (fresh dir).
- **Cost:** ~3.1–3.5 h GPU (one A100); identical step count to draw-1 (same N, same 3 epochs).

### 1.2 Stage 2 — feature extraction with the draw-2 encoder (NO extractor edit)
- **Submit:** `sbatch scripts/slurm/gen_embed_lora.sbatch HateMM logging/lora/HateMM_curric_rep2 Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF`
- `gen_embed_lora.sbatch` is dataset-generic and takes the out-model-tag as **arg 3** (line 34, verified) — no edit.
  It merges the draw-2 adapter, extracts 8-frame dual-stream 3584-d img/text embeddings for all 3 splits into
  `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF.pt` (**DISTINCT
  tag** — never clobbers the frozen / generic-LoRA / draw-1-curric caches), then B2-pushes.
- **Leakage:** unchanged from draw-1 (fixed single-video neutral instructions; gold never enters the deployed
  path; `CAND2_CURRICULUM_PREREG.md` §1.3). **CLEAN.** Cost ~0.4 h GPU.

### 1.3 Stage 3 — 3-seed RGCL align-fusion head + kNN (paired vs banked generic-LoRA AND frozen-CLIP)
- **Submit:** `sbatch scripts/slurm/enc3seed_lora_curric_rep2.sbatch` (authored this prereg — §5).
- **What it runs:** 3 head-only runs (features cached, ~20–25 s each): HateMM-curric-rep2 seeds 0/1/2,
  `--model Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF`, `--group_name RAC_video_lora_curric_rep2`, `--force False`.
- **SAME-CODE guarantee (verified this prereg — §4c):** the `run_one`…`PY` block is **BYTE-IDENTICAL** (42 lines,
  `diff` empty) to `enc3seed_lora_curric.sbatch` (draw-1) **and** to the `enc3seed.sbatch` anchor. The ONLY
  manipulated variables vs the banked controls are `--model` and `--group_name`. Config verbatim: `--batch_size 64
  --lr 0.0001 --epochs 30 --topk 20 --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align
  --hard_negatives_loss True --no_hard_negatives 1 --metric cos --loss triplet --hybrid_loss True --warmup 5
  --lambda_seg 0 --archive OFF`.
- **The head seeds 0/1/2 are the head-init/data-shuffle (unchanged); the draw-2 variation lives entirely at the
  SFT-encoder level.** Cost ~1 min GPU total.

**Total NEW GPU: ~3.5–4 A100-h** (SFT ~3.1–3.5 h dominates; extract ~0.4 h; head ~0.02 h). Single GPU (peak
concurrent = 1). Mining is $0 CPU (idempotent re-emit).

---

## 2. Banked comparison arms — HateMM, re-stated from `CAND2_VERDICT_REVIEW.md` (NOT re-run)

The generic-LoRA arm (13235) and frozen-CLIP floor (12850) are **banked** and reused verbatim. Draw-1's curric
arm (13241) is the thing being replicated. All numbers below are from `CAND2_VERDICT_REVIEW.md` §0.1/§1/§2.2.

| arm | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | mean acc/F1 |
|---|---|---|---|---|---|
| frozen-CLIP floor (12850) | val-sel | 0.8279/0.8172 | 0.8279/0.8163 | 0.8047/0.7920 | 0.8202/0.8085 |
| frozen-CLIP floor (12850) | final-ep | 0.8186/0.7997 | 0.8047/0.7822 | 0.8140/0.7988 | 0.8124/0.7936 |
| **generic-LoRA (13235)** — K-REP pairs vs this | val-sel | 0.8605/0.8521 | 0.8698/0.8620 | 0.8558/0.8495 | **0.8620/0.8545** |
| **generic-LoRA (13235)** | final-ep | 0.8651/0.8580 | 0.8744/0.8660 | 0.8698/0.8613 | **0.8698/0.8618** |
| **draw-1 curric (13241)** — the effect under replication | val-sel | 0.8791/0.8730 | 0.8744/0.8678 | 0.8791/0.8724 | **0.8775/0.8711** |
| **draw-1 curric (13241)** | final-ep | 0.8791/0.8730 | 0.8791/0.8724 | 0.8791/0.8724 | **0.8791/0.8726** |

**Draw-1 K-C2-2 (curric−generic), the F56 effect being replicated:**
- **val-sel** per-seed Δacc `[+0.0186, +0.0046, +0.0233]` → **mean +0.0155**, sign **3/3**, ΔmF1 +0.0166 ⇒ **PASS**.
- **final-ep** per-seed Δacc `[+0.0140, +0.0047, +0.0093]` → **mean +0.0093**, sign 3/3, but **< +0.010 ⇒ TIE**.

**Noise band (from `CAND2_CURRICULUM_PREREG.md` §2.3):** HateMM val-sel between-seed acc spread 0.0140; per-seed
acc std ≈ 0.004–0.006. The **±0.014 band** = the largest observed head-seed spread (basis for KS-REP).

---

## 3. Pre-declared bars (decidable, no freedom; judged by the independent verdict reviewer)

Draw-2's curric (seed s) is paired against the **banked** generic-LoRA arm (13235, seed s), same head-seed
pairing and same enc3seed parser as F56. **Protocol split = SAME as F56 (val-sel binding; final-ep non-binding).**

### 3.1 K-REP-1 — PRIMARY, BINDING (the original K-C2-2 bar applied to draw-2, val-sel)
**PASS** ⇔ draw-2 **val-sel** mean Δacc(curric−generic) **≥ +0.010 AND per-seed sign 3/3 positive AND mean ΔmF1 ≥ 0**
(the frozen §3.4 K-C2-2 PASS rule, verbatim, applied to the draw-2 arm). **K-REP-1 PASS ⇒ the F56 val-sel signal
REPLICATES (hardens).** Otherwise K-REP-1 does not pass (see 3.3/3.4 for what that means).

### 3.2 K-REP-2 — SECONDARY, POOLED READ (2 draws × 3 seeds = 6 paired val-sel points)
Report the pooled mean Δacc and the pooled sign count across all 6 curric−generic val-sel points (draw-1's 3 +
draw-2's 3). **Pre-declared "HARDENED"** ⇔ pooled mean Δacc **≥ +0.010 AND ≥ 5/6 sign positive.**
*Justification (arithmetic, from the banked draw-1 spread):* draw-1 contributes `[+0.0186, +0.0046, +0.0233]`
(sum +0.0465, **3/3 positive**). Pooled mean ≥ +0.010 ⇔ draw-2 sum ≥ +0.0135 ⇔ draw-2 mean ≥ ~+0.0045; ≥5/6 sign
⇔ draw-2 contributes ≥ 2/3 positive. So "hardened" ⇔ **draw-2 broadly agrees in direction** (≥2/3 positive) and
the two-draw pooled acc gain clears the per-draw K-C2-2 acc bar — a wash (draw-2 mean ≈ 0, mixed sign) yields
pooled mean ≈ +0.0078 with ≤4/6 sign and is **NOT** hardened. This is the "stop cherry-picking one draw" read.

### 3.3 KS-REP — RETIREMENT KILL (draw-2 reverses beyond the head-seed band)
**FIRES** ⇔ draw-2 **val-sel** mean Δacc(curric−generic) **≤ −0.014** (below the full head-seed spread) ⇒ the F56
effect **reversed on an independent draw** ⇒ **the F56 HateMM add-over-generic is ruled DRAW-NOISE** and the
cand-2 add-over-generic positive is **retired** (banked strong negative). This is the honest hard-retire.

### 3.4 Verdict logic (pre-declared decision tree; the reviewer renders it verbatim)
- **K-REP-1 PASS** → **F56 REPLICATES (HARDENED).** Report K-REP-2 pooled as corroboration. (Still 2 draws — F-R0.9.)
- **K-REP-1 not-pass AND not KS-REP** (draw-2 val-sel mean ∈ (−0.014, +0.010) OR sign not 3/3) → **inconclusive,
  lean-negative.** Adjudicate by K-REP-2: pooled **HARDENED** ⇒ *weakly hardened* (draw-1 carried it, draw-2
  agreed in direction but under-bar); pooled **NOT hardened** ⇒ **F56 honestly DOWNGRADED** to a single-draw
  fluctuation (the one live positive did not replicate).
- **KS-REP FIRES** → **F56 ruled DRAW-NOISE; retire** (§3.3).

### 3.5 Non-binding sanity reads (reported, not decision-bearing)
- **Draw-2 curric − CLIP (K-C2-1-style hold):** does draw-2 still hold the inherited HateMM pass (mean Δacc ≥
  +0.030, 3/3, ≥ generic−0.014)? Reported for regime sanity; **KS-below-floor** = draw-2 mean below the CLIP floor
  (0.8202 val-sel / 0.8124 final) would flag a broken run.
- **Draw-2 final-epoch add-over-generic:** reported alongside val-sel but **NON-binding** (F-R0.7): draw-1's
  final-ep was a +0.0093 tie, not the effect under replication.

---

## 4. G-repro (adapted) + smoke-SKIP declaration + collision safety

**(a) SFT loss sanity.** On the full run, `logging/lora/HateMM_curric_rep2/all_results.json` eval_loss should land
in the recipe band (draw-1 HateMM generic ≈ 0.108; MHC anchor 0.162). NaN/exploding/flat aborts.

**(b) Data-build reproducibility gate (mandatory).** At submit, STEP 1b re-runs `build_curriculum_sft_data.py
--dataset HateMM`; the executor **MUST verify `data/lora_sft/HateMM/train_curric.json` sha256 == `73307ef2e286…
1c91082b`** (the frozen draw-1 curriculum — proves draw-2 trains the identical multiset, so the only difference is
the seed). Any mismatch ⇒ authorization VOID.

**(c) Head runs = SAME-CODE (verified this prereg).** `run_one`…`PY` of `enc3seed_lora_curric_rep2.sbatch` is
**BYTE-IDENTICAL** (42 lines, empty `diff`) to `enc3seed_lora_curric.sbatch` (draw-1) and `enc3seed.sbatch`
(anchor). The Namespace diff between a draw-2 head run and the 13235 generic / 13241 draw-1 controls MUST be
`--model` + `--group_name` + derived-inert fields only. `bash -n` on both new sbatch = **SYNTAX_OK**.

**(d) SMOKE — PRE-DECLARED SKIP (reviewer rules).** No SFT smoke is run for draw-2. Justification: the recipe is
**byte-identical to draw-1 minus the SFT seed and output_dir**, and this exact recipe on this exact HateMM
curriculum data has already been driven to healthy, finite, decreasing loss with checkpoints written **three
times** — the draw-1 SFT smoke **job 13236** (ZH curric, `train_loss 0.2319`, ckpt written), the draw-1 ZH curric
full SFT **job 13237** (COMPLETE), and the draw-1 HateMM curric full SFT **job 13238** (COMPLETE — it produced the
very adapter that yielded F56). A seed change reseeds `transformers.set_seed` **only** (which random init / which
data order) — it **cannot** introduce NaN / shape / OOM pathologies, which are governed by the recipe, data
schema, and memory footprint, all unchanged. A smoke would re-demonstrate the already-thrice-demonstrated and add
no decision-relevant signal. **The reviewer rules on this skip.** (The healthy-start check on the real run — first
SFT RUNNING with a sane first log line + STEP 1b sha re-verify — is retained as the live gate.)

**(e) Collision safety (verified ABSENT this prereg; re-check at submit).**
- `logging/lora/HateMM_curric_rep2` — absent ⇒ fresh SFT (no clobber of draw-1's `HateMM_curric` adapter).
- `data/CLIP_Embedding/HateMM/*LoRA-curric-rep2*.pt` — absent ⇒ fresh extraction; frozen/generic/draw-1 caches untouched.
- `logging/Retrieval/HateMM/RAC_video_lora_curric_rep2*` — absent ⇒ fresh group; `force=False` never trips an overwrite.
- `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF_seed*_*.trainlog` — absent ⇒ no collision.

---

## 5. Artifacts authored this prereg + hash-freeze block

### 5.1 New artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current) |
|---|---|---|---|
| P | `refine-logs/CAND2_REP2_PREREG.md` | **NEW** — this file | *(reviewer fills at freeze)* |
| A | `RA-HMD/…/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft_rep2.yaml` | **NEW** — draw-1 config C + `seed:1` + rep2 `output_dir` (F-R0.3) | `d645de3197739075774b499f335675dad8cd77a3f03b7c6cdc811424506354c6` |
| B | `scripts/slurm/lora_sft_curric_rep2.sbatch` | **NEW** — HateMM-only clone of `lora_sft_curric.sbatch`; rep2 config; STEP 1b sha re-verify | `265f3e736a0e3ae1202cc86bfef562a2e3d830c9d09487eeea9534ab4c763c1e` |
| C | `scripts/slurm/enc3seed_lora_curric_rep2.sbatch` | **NEW** — clone of `enc3seed_lora_curric.sbatch`; `run_one` byte-identical; HateMM×3 only; `-curric-rep2` tag + `RAC_video_lora_curric_rep2` | `a32fd3bbaaa7140d5d5ffdf1dff3d0df7e26e1fb1ba079c5395e11025861baac` |

### 5.2 Reused-unchanged machinery (verify sha at submit time; do NOT edit)

| path | role | sha256 |
|---|---|---|
| `data/lora_sft/HateMM/train_curric.json` | the frozen draw-1 curriculum multiset (draw-2 trains THIS; STEP 1b must re-emit it bit-exact) | `73307ef2e286eddf4fbe12ef13bb3c750f9105d1291494779c7a3a181c91082b` |
| `RA-HMD/…/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft.yaml` | draw-1 config C (fork parent; the 1-knob diff) | `c12c2b6b340151e6c58ed39843aa2cf02a728c17a3296637cae41c2a70b6a4a3` |
| `scripts/slurm/gen_embed_lora.sbatch` | extraction (dataset-generic; out-tag arg 3; NO edit) | `c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386` |
| `scripts/slurm/enc3seed.sbatch` | same-code anchor for §4c | `dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d` |
| `data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct_HF.pt` | (indirect) frozen cache the builder mines | `ba52bc0da3fa14fefa6b93d5d4abcf42e38bcd01261646309ad262a766a6c009` |

### 5.3 Hash-freeze (filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file CAND2_REP2_PREREG.md, after review>
A d645de3197739075774b499f335675dad8cd77a3f03b7c6cdc811424506354c6  hatemm_qwen25vl_lora_curric_sft_rep2.yaml
B 265f3e736a0e3ae1202cc86bfef562a2e3d830c9d09487eeea9534ab4c763c1e  lora_sft_curric_rep2.sbatch
C a32fd3bbaaa7140d5d5ffdf1dff3d0df7e26e1fb1ba079c5395e11025861baac  enc3seed_lora_curric_rep2.sbatch
--- reused (must still match) ---
  73307ef2e286eddf4fbe12ef13bb3c750f9105d1291494779c7a3a181c91082b  train_curric.json (HateMM, draw-2 trains this)
  c12c2b6b340151e6c58ed39843aa2cf02a728c17a3296637cae41c2a70b6a4a3  hatemm_qwen25vl_lora_curric_sft.yaml (fork parent)
```
Executor re-runs `sha256sum` on A–C (and this file) at submit time; any mismatch = authorization VOID. STEP 1b
must re-emit `train_curric.json` bit-exact (§4b).

---

## 6. Execution / resource plan + test-touch

**Order (3 SLURM jobs; single-GPU; afterok-wired):**
1. `sbatch scripts/slurm/lora_sft_curric_rep2.sbatch` → `logging/lora/HateMM_curric_rep2/` (~3.1–3.5 h). Gate:
   §4b STEP 1b sha re-verify; §4a loss sanity on COMPLETE.
2. `sbatch scripts/slurm/gen_embed_lora.sbatch HateMM logging/lora/HateMM_curric_rep2 Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF` (`afterok:J1`; ~0.4 h).
3. `sbatch scripts/slurm/enc3seed_lora_curric_rep2.sbatch` (`afterok:J2`; ~1 min) → 3 head runs
   `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF_seed{0,1,2}_<JID>.trainlog`.

**Resource plan:** 1×A100; `conda activate HateVideo`; `sbatch` with **NO `--time`**; initial `PENDING
(JobHeldUser)` = **WAIT for auto-release, never force** (CLAUDE.md). `lora_sft_curric_rep2.sbatch` sources
`conda.sh` directly, sets the offline/import env + `CUDA_HOME` shim, and has a ≥20 G disk guard.

**Test-touch:** the Stage-3 draw-2 head reads are the ONLY budgeted HateMM-rep2-curriculum-encoder test
evaluations; **HateMM rep2 test touch consumed at the verdict only.** The executor transcribes raw both-protocol
per-seed numbers (line-numbered) and applies **NO gates/interpretation** — the verdict (§3 decision tree) is
rendered by an **independent 0-context reviewer against this prereg VERBATIM.**

**No job is submitted by this prereg author.** Submission happens only after the independent review + hash-freeze
(run by the orchestrator).

---

## 7. Outcome table template (filled ONLY from raw trainlogs at verdict time)

### 7.1 HateMM draw-2 — curriculum-rep2 vs banked generic-LoRA (K-REP) AND vs frozen-CLIP (sanity)

| seed | protocol | curric-rep2 acc/F1 | generic acc/F1 (§2) | Δ(rep2−generic) acc/F1 | CLIP acc/F1 (§2) | Δ(rep2−CLIP) acc/F1 |
|---|---|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8605/0.8521 | ___ | 0.8279/0.8172 | ___ |
| 1 | val-sel | ___ | 0.8698/0.8620 | ___ | 0.8279/0.8163 | ___ |
| 2 | val-sel | ___ | 0.8558/0.8495 | ___ | 0.8047/0.7920 | ___ |
| **mean** | **val-sel** | ___ | **0.8620/0.8545** | **___ (K-REP-1)** | **0.8202/0.8085** | ___ |
| 0 | final-ep | ___ | 0.8651/0.8580 | ___ | 0.8186/0.7997 | ___ |
| 1 | final-ep | ___ | 0.8744/0.8660 | ___ | 0.8047/0.7822 | ___ |
| 2 | final-ep | ___ | 0.8698/0.8613 | ___ | 0.8140/0.7988 | ___ |
| **mean** | **final-ep** | ___ | **0.8698/0.8618** | **___ (non-binding)** | **0.8124/0.7936** | ___ |

### 7.2 Pooled K-REP-2 (val-sel, 6 points)

| draw | s0 Δacc | s1 Δacc | s2 Δacc | draw sign |
|---|---|---|---|---|
| draw-1 (banked) | +0.0186 | +0.0046 | +0.0233 | 3/3 |
| draw-2 (measured) | ___ | ___ | ___ | ___ |
| **pooled (6)** | mean ___ | | | sign ___/6 |

### 7.3 Fixed write-up format
```
HateMM draw-2: K-REP-1 (val-sel add-over-generic): <PASS/not-pass> (mean ___ acc, sign _/3, ΔmF1 ___).
               K-REP-2 (pooled 6-pt): <HARDENED/NOT> (pooled mean ___ acc, sign _/6).
               KS-REP: <fired/not>.  final-ep add-over-generic (non-binding): mean ___ acc, sign _/3.
VERDICT: F56 HateMM val-sel add-over-generic = <REPLICATES/weakly-hardened/downgraded-to-single-draw/ruled-draw-noise>.
(D7 novelty + goal satisfaction remain the USER's — not decided here.)
```

---

## 8. Provenance index

- Draw-1 prereg (recipe/floors/freeze/decision rule inherited): `refine-logs/CAND2_CURRICULUM_PREREG.md`
  (frozen sha `e5a689d9…f939790e`, commit `76ef0e2`); freeze `refine-logs/CAND2_FREEZE.md`; submit
  `refine-logs/CAND2_SUBMIT_RECORD.md` (jobs 13236 smoke / 13237 ZH SFT / 13238 HateMM SFT / 13239-40 extract /
  13241 head).
- The F56 effect being replicated + banked arm numbers: `refine-logs/CAND2_VERDICT_REVIEW.md` (commit `546acc5`).
- Seed knob (definition of an independent draw): `RA-HMD/…/src/llamafactory/hparams/parser.py:474`
  (`transformers.set_seed(training_args.seed)`); `read_args` yaml-only path `parser.py:73`.
- Protocol + readout parser (verbatim): `research-wiki/experiments/exp-encoder-3seed.md:73-85`.
- Curriculum builder (RNG-free, idempotent): `src/utils/build_curriculum_sft_data.py` (sha `085384f5…`).

**Required statements:** ZERO GPU/SLURM/Modal spent by this prereg author (only pure-CPU login-node reads,
sha256, and a dataclass import, seconds). No held-out test metric produced. All banked numbers taken from the
committed draw-1 verdict + prereg (numeric-provenance discipline). No `state/` mutated. NO job submitted. Not pushed.
