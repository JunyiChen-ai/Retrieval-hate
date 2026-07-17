# CAND-2 Pre-Registration — retrieval-confusion CURRICULUM LoRA-SFT vs generic LoRA (ZH + HateMM)

**Author:** cand-2 curriculum-SFT prereg author (CPU-only; no GPU/SLURM/Modal spent; NO job submitted).
**Date:** 2026-07-18.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced; no job submitted.
**Implements:** `refine-logs/CAND2_CURRICULUM_RECON.md` (commit `7087b5a`, the GO-IF recon) — design ruling
(**design (i) only**), leakage audit, C3GEO-distinction, and kill-switch skeleton transcribed and re-verified
below. Deviations from the recon are flagged **loudly** in §11.
**House-style precedent:** `refine-logs/LORA_HATEMM_PREREG.md` (binding language, floors, freeze block),
`refine-logs/B3_PREREG_REVIEW.md` (ZH numbers + marginal-pass ruling), `research-wiki/experiments/exp-encoder-3seed.md`
(the 12850 encoder-swap protocol + decision rule verbatim).

## Title + claim scope (verbatim)

> This measurement tests whether a **retrieval-coupled adaptation curriculum adds over generic encoder LoRA**;
> a PASS strengthens the case for a **D7 sub-ruling** that memory→adaptation coupling is novel-in-field, **but
> the ruling remains the USER's.** This prereg decides the **performance clause only.**

The cell under test is a **confusion-weighted single-video SFT curriculum** (recon design (i-a)): the RGCL
archive/memory — the LOO kNN vote under the **banked frozen-Qwen train features** — assigns each train video a
confusability `c_i`, and that confusion structure **reweights the encoder-adaptation SFT example distribution**.
The SFT records are **byte-identical** to the generic-LoRA arm (same 8 frames, same instruction, same word
target); the **ONLY manipulated variable is how often each record appears**, and the total is capped to
`N_train` so the 3-epoch step count is IDENTICAL to generic (cost-neutral). Features → the standard archive-OFF
RGCL align-fusion head + top-20 kNN (`enc3s`/12850 protocol), 3-seed paired vs banked frozen-CLIP **and** vs the
banked generic-LoRA arm, dual-protocol.

**Two datasets, each trained ONLY on its own train split (hard veto):** ZH (`MHC_zh`) is the **primary** leg
(strengthen B3's marginal / protocol-dependent pass); HateMM is the **hold-the-inherited-pass** leg (generic
LoRA-HateMM PASSED both protocols — `LORA_HATEMM_VERDICT_REVIEW.md`).

**Purpose:** (a) a **novelty-bearing** upgrade over generic LoRA (the D7 sub-ruling remains the USER's — no
novelty is decided here); (b) a **robustness** upgrade of the marginal ZH leg. **Honest pre-declared prior (from
the recon): ~50–60% chance of a K-C2-2 tie collapsing cand-2 to "generic LoRA with reshuffled data" — see F0.7.**

---

## 0. Binding facts / honesty clauses (all present; pre-declared)

**F0.1 — Test is NOT virgin (declared).** ZH and HateMM test were already read by the frozen-CLIP (13115/12850),
frozen-Qwen (12850), and generic-LoRA arms (ZH job 13150; HateMM job 13235). This prereg's curriculum head reads
are **re-measurements under the identical protocol**, not first exposures. Each consumes exactly ONE budgeted
**curriculum-LoRA-encoder** test evaluation (ZH-curric + HateMM-curric). Zero test-touch before the independent verdict.

**F0.2 — Single-encoder-draw limitation (pre-declared; identical to B3 §0.2 / LoRA-HateMM F0.2, and CRITICAL for
K-C2-2).** The 3 head-seeds read ONE curriculum-SFT encoder draw per dataset (head init + data-shuffle vary; the
curriculum encoder is fixed). The reported ±band is **head-seed variance, NOT curriculum-SFT-draw variance.** The
**add-over-generic (K-C2-2)** comparison is therefore **one curriculum draw vs one generic draw, both read by 3
head-seeds** — a head-seed-paired test that **cannot separate the curriculum effect from SFT-draw luck.** A
curriculum-draw-stability claim would need ≥3 fresh curriculum SFT retrains (~9 h) — out of scope, pre-declared.
The design is symmetric with the single-draw generic and frozen-CLIP controls.

**F0.3 — Novelty = D7 SUB-RULING, PENDING USER (not decided here).** Curriculum learning / hard-example mining
for SFT is textbook outside hateful-video; `C3GEO_FORENSIC_RECON.md` already adjudicated the *sibling* idea
(retrieval-mined hard negatives) as **D7-dead**. Cand-2 survives that kill on ONE load-bearing difference (F0.6),
but **whether a memory→adaptation-coupling SFT curriculum counts as distinct from generic LoRA is a narrower,
stronger D7 *sub*-ruling — the USER's, not this experiment's.** This prereg decides the **performance clause only.**

**F0.4 — Structural ceiling: cand-2 opens NO new dataset (pre-declared, material to any claim).** By F44/F45
modality-locus arithmetic, a text/curriculum lever **holds ZH's pass and can add only HateMM or EN**; HateMM is
image-borne (inherited, not converted); EN is label-limited (dead to the representation family). So cand-2's
realistic best case is a **cleaner, coupling-novel, protocol-robust 2-dataset story on the datasets generic LoRA
already passes** — a novelty + robustness upgrade, **not** a new performance route. Performance prior on a *new*
≥2-dataset conjunct generic LoRA doesn't already deliver: **~5%.**

**F0.5 — Single-dataset own-train-split VETO compliance (hard user veto).** ZH-curric trains on
`data/lora_sft/MHC_zh/train_curric.json` (a reweighted multiset of `MHC_zh` own-train records only);
HateMM-curric on `data/lora_sft/HateMM/train_curric.json` (HateMM own-train only). Mining uses **own-train gold
labels** (allowed in training) over the **own-train frozen cache**; **no dev/test video ever enters the mining
index** (enforced by loading only `train_*.pt` with `exclude_self=True`). NO cross-dataset mixing, NO gold
spans/attributes, NO OCR, raw videos never leave the machine. All standing vetoes cleared.

**F0.6 — Non-redundancy vs the head's own mining (the C3GEO distinction — carried verbatim from recon §2.4/§5).**
The downstream RGCL head already mines the hardest opposite-label pairs **per-epoch in its evolving feature
space** (`retrieval.py:347-353,480`; `loss.py:453-455`). But that mining reads **frozen extracted features** — it
can only *exploit* whatever separation the encoder already put on the boundary; **it cannot make the encoder
*allocate* representation capacity there.** Cand-2's curriculum is the only lever that spends the r16/3-epoch
encoder-adaptation budget preferentially on the confusable region — the exact non-redundancy C3-geo lacked
(C3-geo re-sourced hard negatives for a saturated head loss; cand-2 changes *where the encoder spends capacity*).
**This distinction keeps cand-2 legal and non-redundant, but it is thin, and whether it clears D7 is F0.3.**

**F0.7 — Pre-declared honest failure mode (~50–60%, the MOST LIKELY outcome; recon §4).** The head re-mines the
confusable structure from the extracted features regardless of how the encoder was SFT'd. If the curriculum
encoder produces feature geometry ≈ the generic encoder on the confusable boundary (**K-C2-2 ties**), then the
head's own mining has already extracted everything the curriculum tried to inject — the P3/C3-geo "objective
already sees the hard structure, curriculum is redundant" pattern, one level up. This is **not a bug**; it would
be the empirical finding that even encoder-capacity-allocation is redundant with the head's frozen-feature
mining, closing the last adaptation-family cell. **K-C2-2 is designed to detect exactly this and force the honest
"it's just LoRA" verdict.**

**F0.8 — Curriculum induces a class-balance shift (pre-declared property, not a hidden confound).** The
confusable boundary is class-skewed vs the pool, so reweighting shifts the SFT class balance: **ZH 31.1%→41.1%
hateful, HateMM 40.1%→37.7% hateful** (§1.1). This is intrinsic to confusion-weighting (F45: the ZH gain *is* a
minority hate-recall Pareto move, so upweighting confusable hateful videos is mechanism-aligned), not a separate
manipulated variable. It travels with any K-C2-2 interpretation and is stated for full transparency.

---

## 1. Pipeline spec — fully pinned (4 stages; nothing left to interpretation)

### 1.1 Stage 0 — curriculum data build (CPU; DONE + verified idempotent this prereg)

Two CPU steps, both idempotent, run at submit time inside `lora_sft_curric.sbatch` (STEP 1a/1b):

1. `python src/utils/build_lora_sft_data.py --dataset <DS>` — the **generic word-variant** build (frames +
   `train.json`/`val.json` + `<prefix>_lora_{train,val}` registration). Idempotent (frames cached). This is the
   EXACT generic arm the curriculum forks and compares against.
2. `python src/utils/build_curriculum_sft_data.py --dataset <DS> --mode softconf` — **the ONLY new code**
   (§5). Mines confusability over the banked frozen-Qwen train cache and reweights `train.json` into
   `train_curric.json`, registering `<prefix>_lora_curric_train`. **Val is UNCHANGED** (`<prefix>_lora_val`).

**Mining (pinned, ZERO GPU, $0 CPU over the banked cache):** load
`data/CLIP_Embedding/<DS>/train_Qwen2.5-VL-7B-Instruct_HF.pt`; L2-normalise each stream, concat, renormalise (⇒
neighbour similarity = **mean of the img-cosine and text-cosine**); top-20 rank-weighted **signed-cosine** vote
with `exclude_self` LOO (vote machinery lifted verbatim from `cross_channel_router_gate.py:73-78,120-131`);
`c_i = exp(-|vote_i| / τ)` (peaks at the decision boundary). Multiplicity `w_i = 1 + λ·c_i`. **Cost-neutral
multiset:** deterministic **largest-remainder apportionment** of exactly `N_train` slots ∝ `w_i` (floor(quota)
then residual to the largest fractional remainders; ties by larger `w` then lower index). **No RNG on the
registered softconf path** (stronger reproducibility than the recon's "pin the RNG seed" — see §11 dev-3).

**Pinned hyperparameters (frozen; baked into `build_curriculum_sft_data.py`):**
`TOPK=20`, `TAU=0.20`, `LAMBDA=10.0`, `CAP_RATIO=1.0`, fused-vote = concat-of-unit-streams, `mode=softconf`
(design (i-a); `error` (i-b) implemented but NOT the registered arm — running it would be a second bite).

**Materialized + registered on disk (this prereg ran the CPU build twice — bit-exact idempotent):**

| file | rows | unique | maxdup | sha256 |
|---|---|---|---|---|
| `data/lora_sft/MHC_zh/train_curric.json` | 579 (== N) | 386 | 3 | `c8260dd3f5a98394c6ef3d7f08e091dad5810e1d22d58db24ac5654d7029bc0d` |
| `data/lora_sft/HateMM/train_curric.json` | 743 (== N) | 502 | 4 | `73307ef2e286eddf4fbe12ef13bb3c750f9105d1291494779c7a3a181c91082b` |
| `RA-HMD/…/data/dataset_info.json` (curric keys added, additive) | — | — | — | `c2b99d2521b1785a2df8da0fd62b13ea4c0dea086bd783cd724619aec0229fd6` |

Provenance link: the ZH generic fork source `train.json` sha `ecfa663d…31b10d0`; the HateMM fork source
`train.json` sha `93c6d3d1bffbca22b2dd8beba57a33575a48d8ca61d8d56e3148fecdbb93973a` — **identical to the sha the
LoRA-HateMM prereg pinned** (`LORA_HATEMM_PREREG.md §1.1`), i.e. the curriculum forks the exact records the 13235
generic arm trained on.

### 1.2 Stage 1 — curriculum LoRA-SFT (own train split only)

- **Submit:** `sbatch scripts/slurm/lora_sft_curric.sbatch <DS>` (`<DS>` ∈ {MHC_zh, HateMM}).
- **Config (authored this prereg; EXACTLY 2 changed lines vs the generic config — §5):**
  `…/my_configs/hatevideo/<ds>_qwen25vl_lora_curric_sft.yaml`: L18 `dataset: <prefix>_lora_train` →
  `<prefix>_lora_curric_train`; L27 `output_dir: …/lora/<DS>` → `…/lora/<DS>_curric`. `eval_dataset` UNCHANGED.
- **Recipe (BYTE-IDENTICAL to the generic arm otherwise):** base `Qwen/Qwen2.5-VL-7B-Instruct`; `stage: sft`
  (word-label generative); `lora_rank 16`, `lora_alpha 32`, dropout 0.0, `lora_target
  q,k,v,o,gate,up,down_proj`; **`freeze_vision_tower: true`, `freeze_multi_modal_projector: true`**; `lr 1.0e-4`,
  `num_train_epochs 3.0`, `cosine`, `warmup_ratio 0.05`, `per_device_train_batch_size 1`,
  `gradient_accumulation_steps 8` (eff 8), `bf16`, 8-frame ShareGPT, `cutoff_len 4096`, `save_strategy epoch`.
  Output adapter → `logging/lora/<DS>_curric/` (does NOT exist; fresh SFT creates it). **The ONLY difference from
  the generic run is the train multiset.**
- **Cost:** curriculum size == generic N ⇒ identical step count. ZH ~2.8–3.3 h; HateMM ~3.1–3.5 h GPU (one A100).

### 1.3 Stage 2 — feature extraction with the curriculum-merged encoder (NO extractor edit)

- **Submit:** `sbatch scripts/slurm/gen_embed_lora.sbatch <DS> logging/lora/<DS>_curric Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`
- The runner (`gen_embed_lora.sbatch` + `generate_VideoMLLM_embedding_lora_HF.py`) is dataset-generic and takes
  the **out-model-tag as arg 3** — no edit needed. It merges the curriculum adapter, extracts 8-frame dual-stream
  3584-d img/text embeddings for all 3 splits into
  `data/CLIP_Embedding/<DS>/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` (**DISTINCT tag**;
  never clobbers the frozen or generic-LoRA caches), then B2-pushes.
- **Leakage (recon §2.2, decisive):** extraction deploys **FIXED single-video neutral instructions**
  (`IMG_INSTRUCTION`/`TEXT_INSTRUCTION`, `generate_VideoMLLM_embedding_lora_HF.py:59-66`), one video at a time, no
  neighbour, no label. Design (i) changes only *which/how-often* the identical single-video records appear in SFT;
  the deployed encoder input stays label-free and single-video ⇒ **gold never enters the deployed path. CLEAN.**
- **Cost:** ~0.35 h (ZH) / ~0.4 h (HateMM) GPU.

### 1.4 Stage 3 — 3-seed RGCL align-fusion head + kNN (paired vs CLIP floor AND generic-LoRA)

- **Submit:** `sbatch scripts/slurm/enc3seed_lora_curric.sbatch` (authored this prereg — §5).
- **What it runs:** 6 head-only runs (features cached, ~20–25 s each): MHC_zh-curric seeds 0/1/2 **and**
  HateMM-curric seeds 0/1/2, `--model Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`, `--group_name RAC_video_lora_curric`,
  `--force False`.
- **CRITICAL same-code guarantee (verified this prereg — §4.2):** the `run_one` python block is **BYTE-IDENTICAL**
  to `enc3seed.sbatch`/`enc3seed_lora_hatemm.sbatch` (41-line block, `diff` empty); the ONLY manipulated variables
  vs the banked CLIP/generic controls are `--model` and `--group_name`. Config: `--batch_size 64 --lr 0.0001
  --epochs 30 --topk 20 --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align
  --hard_negatives_loss True --no_hard_negatives 1 --metric cos --loss triplet --hybrid_loss True --warmup 5
  --lambda_seg 0 --archive OFF`. Identical to `exp-encoder-3seed.md` H1 / B3 / LoRA-HateMM.
- **Cost:** ~2 min GPU total.

**Total NEW GPU: ~7–8 A100-h** (2× SFT ~6.5 h dominates; 2× extract ~0.75 h; head ~0.03 h). Mining is $0 CPU.

---

## 2. Comparison floors + generic arms — INDEPENDENTLY RE-DERIVED from raw trainlogs (numeric-provenance discipline)

Every number re-parsed this prereg with the EXACT `enc3seed.sbatch` parser (val-sel = epoch ≥ warmup 5 max
`Val_Retrieval` acc, roc tie-break; final = max epoch). All match `B3_PREREG_REVIEW` / `LORA_HATEMM_VERDICT_REVIEW`
to 4dp.

### 2.1 ZH (`MHC_zh`) — CLIP floor (13115) and generic-LoRA arm = B3 (13150, bit-exact vs arcbase 12223-25)

| arm | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | 3-seed mean acc/F1 |
|---|---|---|---|---|---|
| **frozen-CLIP floor** (K-C2-1 pairs vs this) | val-sel | 0.8054/0.7706 | 0.8054/0.7579 | 0.8121/0.7742 | **0.8076 / 0.7676** |
| **frozen-CLIP floor** | final-ep | 0.8054/0.7706 | 0.8054/0.7542 | 0.8322/0.7913 | **0.8143 / 0.7720** |
| **generic-LoRA (B3)** (K-C2-2 pairs vs this) | val-sel | 0.8322/0.8023 | 0.8255/0.7956 | 0.8389/0.8065 | **0.8322 / 0.8015** |
| **generic-LoRA (B3)** | final-ep | 0.8456/0.8181 | 0.8389/0.8113 | 0.8523/0.8226 | **0.8456 / 0.8173** |

Generic-LoRA − CLIP (the B3 verdict, re-derived): **final-ep +0.0313 acc / +0.0453 F1 (3/3 sign; seed2 +0.0201
below the +0.030 per-seed bar ⇒ MARGINAL); val-sel +0.0246 acc / +0.0339 F1 (acc FAIL).** This is the marginal,
protocol-dependent pass cand-2's ZH leg must strengthen.

### 2.2 HateMM — CLIP floor (12850) and generic-LoRA arm (13235; PASSED both protocols)

| arm | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | 3-seed mean acc/F1 |
|---|---|---|---|---|---|
| **frozen-CLIP floor** (K-C2-1 pairs vs this) | val-sel | 0.8279/0.8172 | 0.8279/0.8163 | 0.8047/0.7920 | **0.8202 / 0.8085** |
| **frozen-CLIP floor** | final-ep | 0.8186/0.7997 | 0.8047/0.7822 | 0.8140/0.7988 | **0.8124 / 0.7936** |
| **generic-LoRA (13235)** (K-C2-2 pairs vs this) | val-sel | 0.8605/0.8521 | 0.8698/0.8620 | 0.8558/0.8495 | **0.8620 / 0.8545** |
| **generic-LoRA (13235)** | final-ep | 0.8651/0.8580 | 0.8744/0.8660 | 0.8698/0.8613 | **0.8698 / 0.8618** |

Generic-LoRA − CLIP (the LoRA-HateMM verdict, re-derived): **val-sel +0.0419 acc / +0.0460 F1 (3/3 PASS);
final-ep +0.0573 acc / +0.0682 F1 (3/3 PASS).** HateMM's pass is substantially image-inherited (LoRA ≈ frozen-Qwen,
`LORA_HATEMM_VERDICT_REVIEW.md §3`); the curriculum's job here is to **HOLD** it, not add.

### 2.3 Noise-band derivation (justifies the ±0.014 hold/regression band)

Between-seed acc spread (max−min) in the banked generic arms: HateMM val-sel **0.0140** (0.8698−0.8558), HateMM
final 0.0093, ZH val-sel 0.0134, ZH final 0.0134; per-seed acc std ≈ 0.004–0.006. **The ±0.014 band = the largest
observed head-seed spread.** A 3-seed-mean move beyond ±0.014 is beyond the full head-seed spread; a mean move of
≥ +0.010 **with 3/3 concordant sign** is ~2× the per-seed std with every seed improving (distinguishable from a
within-spread tie) — this is the basis for the K-C2-2 threshold (§3.3).

---

## 3. Decision rule + kill-switches (paired, both protocols judged independently, 3/3 sign, pre-declared)

### 3.1 Decision rule — verbatim from `exp-encoder-3seed.md:73-85`

> For each dataset × protocol: (1) per-seed paired difference; (2) 3-seed mean ± std + sign consistency; (3) n=3
> too small for a bootstrap — report paired-t as an **effect-size descriptor only**, no significance claim; (4)
> **pass = mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive**; (5) headline
> claim requires pass on ≥ 2 datasets under a stated protocol; both protocols judged separately; verdict written
> exactly "final-epoch: pass/fail; val-selected: pass/fail".

### 3.2 K-C2-0 — MINING-VALIDITY ($0 CPU pre-GPU gate; **COMPUTED THIS PREREG — PASS both datasets**)

The curriculum must be a *distinct* method, not generic LoRA in disguise. Computed over the frozen-Qwen train
cache (`refine-logs/CAND2_KC20_<DS>.json`, reproducible by re-running the builder):

| check | criterion | ZH | HateMM |
|---|---|---|---|
| (a) non-degenerate boundary | frozen LOO kNN error ∈ [0.15, 0.35] | **0.2073** ✔ | **0.1935** ✔ |
| (b) concentration | confusion-weight `c` Gini ≥ 0.30 | **0.5634** ✔ | **0.6497** ✔ |
| (c) differs from uniform | curriculum unique coverage < 0.90·N | **0.6667** ✔ | **0.6756** ✔ |
| — descriptor | mass on top-30% confusable vs uniform | **2.11×** | **2.08×** |

**K-C2-0 = PASS both.** The frozen encoder does NOT memorize (error ≈ 0.20, not ≈ 0 ⇒ real boundary mass) and is
NOT noise (not ≈ 0.50); the curriculum concentrates ~2.1× mass on the confusable head at the same budget. **Had
(a) landed ≈ 0 (memorization ⇒ every `c_i ≈ 0` ⇒ curriculum ≡ uniform), this would be an auto-KILL pre-GPU** — it
did not. The reviewer re-verifies by re-running the builder and matching the `train_curric.json` shas (§1.1).

### 3.3 K-C2-1 — PERFORMANCE, primary (must HOLD the inherited passes), curriculum-LoRA − CLIP

Per dataset × protocol: **mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND sign 3/3**, judged independently under
each protocol, **AND ≥ (generic-LoRA arm − 0.014)** (must not regress the pass it inherits). CLIP floors §2.1/§2.2;
generic arms §2.1/§2.2. Below the conjunct → **KILL** on that protocol.

### 3.4 K-C2-2 — ADD-OVER-GENERIC (THE NOVELTY-EARNING BAR), curriculum-LoRA − generic-LoRA, paired by head-seed

The decisive comparison is **against the GENERIC LoRA arm, not just CLIP** (banked: ZH 13150, HateMM 13235 — NOT
re-run). Paired per head-seed (curric seed s − generic seed s), same structure as K-C2-1 pairs curric−CLIP.

- **PASS (per dataset, ≥1 protocol):** mean paired **Δacc ≥ +0.010 AND sign 3/3 positive AND mean ΔmF1 ≥ 0**.
  Justification (§2.3): +0.010 = the recon §3.2 operational target (lift ZH seed2 +0.0201→≥+0.030, or a uniform
  ~+0.007 → mean ≈ +0.040) and ~2× the per-seed std; the **3/3-sign requirement is the teeth** — a within-noise
  tie does not produce three concordant improvements.
- **TIE = NO NOVELTY (the F0.7 outcome):** mean paired |Δacc| < +0.010 **OR** sign not 3/3 ⇒ the curriculum ties
  generic within head-seed noise ⇒ the coupling earns **no** novelty; **report "generic LoRA with reshuffled
  data," bank the negative, do NOT claim the coupling.**

Judged per-dataset; the novelty signal requires K-C2-2 PASS on ≥1 dataset (ZH the a-priori most likely, §3.6).
**F0.2 caveat travels with any K-C2-2 PASS:** it is a single curriculum draw vs a single generic draw.

### 3.5 KS-regression — BELOW-GENERIC KILL

If curriculum-LoRA − generic-LoRA **mean Δacc ≤ −0.014** on a held leg (below the full head-seed spread), the
curriculum **degraded** adaptation (overfit the confusable subset, or the size-cap starved easy-example coverage
— note ~33% of easy videos are dropped, §1.1) → **KILL**, bank "confusion-curriculum hurts."

### 3.6 KS-below-floor — REGIME SANITY

If curriculum-LoRA lands **below the CLIP floor** on ZH — the leg it was built to strengthen — bank the strong
negative (the curriculum broke the mechanism).

### 3.7 ZH-ROBUSTNESS clause — pre-declared "ZH leg strengthened" pattern (the goal-relevant upgrade)

Beyond K-C2-2, the ZH leg is declared **strengthened** iff EITHER (curric − CLIP):
- **(a) val-selected conjunct now PASSES:** val-sel mean Δacc ≥ +0.030 AND ΔmF1 ≥ +0.030, 3/3 (B3 val-sel FAILed
  at +0.0246 acc); **OR**
- **(b) final-epoch becomes NON-marginal:** final-ep mean Δacc ≥ +0.040 AND **3/3 per-seed Δacc ≥ +0.030** (B3
  seed2 was +0.0201, below the per-seed bar).

**Either pattern ⇒ the 2-dataset ZH claim stops depending on protocol choice; both ⇒ fully protocol-robust.** This
is the strongest *performance* case for cand-2 (and it is modest and bounded — F0.4).

### 3.8 Gate order

K-C2-0 ($0, DONE-PASS) → G-repro-adapted (§4: SFT-loss sanity + data-build sha re-verify + head Namespace-diff) →
K-C2-1 → K-C2-2 → KS-regression → KS-below-floor → ZH-robustness read. Single test-touch per dataset.

---

## 4. G-repro (adapted — first curriculum draw, no bit-exact anchor) + smoke plan + collision safety

### 4.1 G-repro discipline

- **(a) SFT smoke gate (Stage 1).** A tiny SFT smoke must show loss **finite (no NaN), decreasing, checkpoint
  written**; on the full run, `logging/lora/<DS>_curric/all_results.json` eval_loss should land in the ~0.12–0.18
  band (MHC anchor 0.1620; generic LoRA-HateMM landed 0.1084 — a tighter fit is benign). NaN/exploding/flat aborts.
- **(b) Data-build reproducibility gate (NEW, cand-2-specific).** At submit time STEP 1b re-runs
  `build_curriculum_sft_data.py`; the executor MUST verify `train_curric.json` sha256 matches the pinned §1.1 shas
  (the registered curriculum == what trains). This prereg verified bit-exact idempotency (two runs, identical sha).
- **(c) Head runs = SAME-CODE as the banked controls.** The Namespace diff between a curriculum head run and the
  12850/13115/13150/13235 controls MUST be `--model` + derived-inert fields only. `run_one` is byte-identical
  (§4.2). This retires the code-version confound the same way `exp-encoder-3seed.md:126-146` did.
- **(d) frozen-CLIP + generic-LoRA controls re-paired from banked logs (§2), not re-run.**

### 4.2 Same-code + syntax verification (run this prereg — PASS)

`run_one`…`PY` block of `enc3seed_lora_curric.sbatch` == `enc3seed_lora_hatemm.sbatch` == `enc3seed.sbatch`:
**BYTE-IDENTICAL** (41 lines). Full-file `diff` vs `enc3seed_lora_hatemm.sbatch`: header comment, `LORA` tag,
`GROUP_NAME`, `CONFIGS` only. `bash -n` on both new sbatch = **SYNTAX_OK**. Config diffs = exactly 2 lines each (§5).

### 4.3 Collision safety (verified this prereg; re-check at submit)

- `logging/lora/{MHC_zh_curric,HateMM_curric}` — do NOT exist ⇒ fresh SFT (no clobber of generic MHC_zh/HateMM adapters).
- `data/CLIP_Embedding/{MHC_zh,HateMM}/*LoRA-curric*.pt` — do NOT exist ⇒ fresh extraction; frozen + generic-LoRA caches untouched.
- `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_lora_curric*` — do NOT exist ⇒ fresh group, `force=False` never
  trips `run_rac.py:904-908`; NO banked arm overwritten (`exp_name` is seed+model-derived; the `-curric` tag
  differs from CLIP/Qwen/LoRA, so dirs are distinct regardless).
- `slurm/logs/enc3s_{MHC_zh,HateMM}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed*_*.trainlog` — do NOT exist ⇒ no collision.

### 4.4 Smoke plan (executor runs BEFORE the real submits; leave no artifact that trips §4.3)

1. **SFT smoke:** launch a curric config with `max_steps: 20`, `save_steps: 20`, throwaway
   `output_dir: logging/lora/_smoke_curric` — confirm loss finite/decreasing, ckpt written; delete the smoke dir.
   (Do NOT smoke-write into `logging/lora/<DS>_curric`.)
2. **1-seed head smoke:** on the existing generic LoRA cache (`data/CLIP_Embedding/<DS>/*LoRA_HF.pt`), run ONE
   `run_rac.py` head with throwaway `--group_name _smoke` to confirm the align-fusion path loads + completes 30
   epochs; delete the `_smoke` dir. If in doubt, skip — the same-code guarantee and cache dims are CPU-verified.

---

## 5. Artifacts authored this prereg + hash-freeze block

### 5.1 New artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current) |
|---|---|---|---|
| P | `refine-logs/CAND2_CURRICULUM_PREREG.md` | **NEW** — this file | *(reviewer fills at freeze)* |
| A | `src/utils/build_curriculum_sft_data.py` | **NEW** — confusion-weighted curriculum builder (273 LOC incl. docstring + K-C2-0 diagnostics), no overwrite of `build_lora_sft_data.py` | `085384f5534ffae9969c95211f7eaefca5cc3d54278734ba76457b84990f66e8` |
| B | `RA-HMD/…/my_configs/hatevideo/mhc_zh_qwen25vl_lora_curric_sft.yaml` | **NEW** — 2-line diff vs `mhc_zh_qwen25vl_lora_sft.yaml` (dataset + output_dir) | `ac1c596293877e827c9db96bec8aefc8f36ebe5e6d3aa95544889be48815fa6d` |
| C | `RA-HMD/…/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft.yaml` | **NEW** — 2-line diff vs `hatemm_qwen25vl_lora_sft.yaml` | `c12c2b6b340151e6c58ed39843aa2cf02a728c17a3296637cae41c2a70b6a4a3` |
| D | `scripts/slurm/lora_sft_curric.sbatch` | **NEW** — clone of `lora_sft.sbatch` + STEP 1b curriculum build + curric configs; {MHC_zh, HateMM} cases | `6a5abb9e7d7427f7e4e9874ee429eaed4ed269e342cff5b6df14d40e59ffd57a` |
| E | `scripts/slurm/enc3seed_lora_curric.sbatch` | **NEW** — clone of `enc3seed_lora_hatemm.sbatch`; `run_one` byte-identical; `-curric` tag, `RAC_video_lora_curric`, 6 rows | `00d9e9956549bdf97c6b8913d42d87811f4a2f150e9f459e8b8b86978b306f02` |
| F | `data/lora_sft/MHC_zh/train_curric.json` | **NEW** — 579 rows (built + idempotent) | `c8260dd3f5a98394c6ef3d7f08e091dad5810e1d22d58db24ac5654d7029bc0d` |
| G | `data/lora_sft/HateMM/train_curric.json` | **NEW** — 743 rows (built + idempotent) | `73307ef2e286eddf4fbe12ef13bb3c750f9105d1291494779c7a3a181c91082b` |
| H | `RA-HMD/…/data/dataset_info.json` | **EDITED (additive)** — `mhc_zh_lora_curric_train` + `hatemm_lora_curric_train` keys | `c2b99d2521b1785a2df8da0fd62b13ea4c0dea086bd783cd724619aec0229fd6` |
| I | `refine-logs/CAND2_KC20_MHC_zh.json` | **NEW** — K-C2-0 diagnostics | `38b21db5909d4affc9f57c3a9286eab0e807b00c6b7a0d7de599d6ca1a0f6f33` |
| J | `refine-logs/CAND2_KC20_HateMM.json` | **NEW** — K-C2-0 diagnostics | `14967d5313e044a556a8caf365ab4ab00178d51b0ce3fd67d7a6263b4048cf6b` |

### 5.2 Reused-unchanged machinery (verify sha at submit time; do NOT edit)

| path | role | sha256 |
|---|---|---|
| `src/utils/build_lora_sft_data.py` | generic word-variant builder (STEP 1a; forked, not edited) | *(unchanged; not a decision gate)* |
| `scripts/slurm/gen_embed_lora.sbatch` | extraction (dataset-generic; out-tag arg 3; NO edit) | `c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386` |
| `data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct_HF.pt` | ZH mining input (frozen cache) | `135a6e243761fa832c712bf4d02478ac34bc49cabaf888a7b5fe465695d3861e` |
| `data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct_HF.pt` | HateMM mining input (frozen cache) | `ba52bc0da3fa14fefa6b93d5d4abcf42e38bcd01261646309ad262a766a6c009` |
| `scripts/slurm/enc3seed.sbatch` | same-code anchor for §4.2 | `dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d` |

### 5.3 Hash-freeze (to be filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file CAND2_CURRICULUM_PREREG.md, after review>
A 085384f5534ffae9969c95211f7eaefca5cc3d54278734ba76457b84990f66e8  src/utils/build_curriculum_sft_data.py
B ac1c596293877e827c9db96bec8aefc8f36ebe5e6d3aa95544889be48815fa6d  mhc_zh_qwen25vl_lora_curric_sft.yaml
C c12c2b6b340151e6c58ed39843aa2cf02a728c17a3296637cae41c2a70b6a4a3  hatemm_qwen25vl_lora_curric_sft.yaml
D 6a5abb9e7d7427f7e4e9874ee429eaed4ed269e342cff5b6df14d40e59ffd57a  lora_sft_curric.sbatch
E 00d9e9956549bdf97c6b8913d42d87811f4a2f150e9f459e8b8b86978b306f02  enc3seed_lora_curric.sbatch
F c8260dd3f5a98394c6ef3d7f08e091dad5810e1d22d58db24ac5654d7029bc0d  train_curric.json (MHC_zh)
G 73307ef2e286eddf4fbe12ef13bb3c750f9105d1291494779c7a3a181c91082b  train_curric.json (HateMM)
H c2b99d2521b1785a2df8da0fd62b13ea4c0dea086bd783cd724619aec0229fd6  dataset_info.json
```
Executor re-runs `sha256sum` on A–H (and this file) at submit time; any mismatch = authorization VOID. STEP 1b
must reproduce F/G bit-exact (§4.1b).

---

## 6. Single-submit / execution plan + resource plan

**Order (5 SLURM jobs; SFT/extract per-dataset, one combined head job):**

1. `sbatch scripts/slurm/lora_sft_curric.sbatch MHC_zh` → `logging/lora/MHC_zh_curric/` (~2.8–3.3 h). Gate: SFT
   smoke (§4.4.1) BEFORE; on COMPLETE apply §4.1a + §4.1b sha re-verify.
2. `sbatch scripts/slurm/lora_sft_curric.sbatch HateMM` → `logging/lora/HateMM_curric/` (~3.1–3.5 h).
3. `sbatch scripts/slurm/gen_embed_lora.sbatch MHC_zh logging/lora/MHC_zh_curric Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` (~0.35 h).
4. `sbatch scripts/slurm/gen_embed_lora.sbatch HateMM logging/lora/HateMM_curric Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` (~0.4 h).
5. `sbatch scripts/slurm/enc3seed_lora_curric.sbatch` → 6 head runs (~2 min). Produces
   `slurm/logs/enc3s_{MHC_zh,HateMM}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_<JID>.trainlog`.

Chainable via `--dependency=afterok:` (SFT→extract per dataset; both extracts → head).

**Resource plan:** 1×A100; `conda activate HateVideo`; `sbatch` with **NO `--time`**; initial `PENDING
(JobHeldUser)` = **WAIT for auto-release, never force** (CLAUDE.md). `lora_sft_curric.sbatch` sources `conda.sh`
directly, sets the offline/import env + `CUDA_HOME` shim, and has a ≥20 G disk guard.

**Test-touch:** the Stage-3 curriculum head reads are the ONLY budgeted curriculum-LoRA-encoder test evaluations
(ZH + HateMM); zero test-touch before the verdict. **The executor transcribes raw both-protocol per-seed numbers
(line-numbered) and applies NO gates/interpretation** — the verdict (G-repro → Namespace-diff → K-C2-0 re-verify
→ K-C2-1/2 → KS → ZH-robustness) is rendered by an **independent 0-context reviewer against this prereg VERBATIM.**

**No job is submitted by this prereg author.** Submission happens only after the independent review + hash-freeze
(run by the orchestrator).

---

## 7. Outcome table template (filled ONLY from raw trainlogs at verdict time)

### 7.1 ZH — curriculum-LoRA vs frozen-CLIP (K-C2-1) AND vs generic-LoRA (K-C2-2)

| seed | protocol | curric acc/F1 | CLIP acc/F1 (§2.1) | Δ(curric−CLIP) acc/F1 | generic acc/F1 (§2.1) | Δ(curric−generic) acc/F1 |
|---|---|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8054/0.7706 | ___ | 0.8322/0.8023 | ___ |
| 1 | val-sel | ___ | 0.8054/0.7579 | ___ | 0.8255/0.7956 | ___ |
| 2 | val-sel | ___ | 0.8121/0.7742 | ___ | 0.8389/0.8065 | ___ |
| **mean** | **val-sel** | ___ | **0.8076/0.7676** | **___** | **0.8322/0.8015** | **___** |
| 0 | final-ep | ___ | 0.8054/0.7706 | ___ | 0.8456/0.8181 | ___ |
| 1 | final-ep | ___ | 0.8054/0.7542 | ___ | 0.8389/0.8113 | ___ |
| 2 | final-ep | ___ | 0.8322/0.7913 | ___ | 0.8523/0.8226 | ___ |
| **mean** | **final-ep** | ___ | **0.8143/0.7720** | **___** | **0.8456/0.8173** | **___** |

### 7.2 HateMM — curriculum-LoRA vs frozen-CLIP (K-C2-1) AND vs generic-LoRA (K-C2-2)

| seed | protocol | curric acc/F1 | CLIP acc/F1 (§2.2) | Δ(curric−CLIP) acc/F1 | generic acc/F1 (§2.2) | Δ(curric−generic) acc/F1 |
|---|---|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8279/0.8172 | ___ | 0.8605/0.8521 | ___ |
| 1 | val-sel | ___ | 0.8279/0.8163 | ___ | 0.8698/0.8620 | ___ |
| 2 | val-sel | ___ | 0.8047/0.7920 | ___ | 0.8558/0.8495 | ___ |
| **mean** | **val-sel** | ___ | **0.8202/0.8085** | **___** | **0.8620/0.8545** | **___** |
| 0 | final-ep | ___ | 0.8186/0.7997 | ___ | 0.8651/0.8580 | ___ |
| 1 | final-ep | ___ | 0.8047/0.7822 | ___ | 0.8744/0.8660 | ___ |
| 2 | final-ep | ___ | 0.8140/0.7988 | ___ | 0.8698/0.8613 | ___ |
| **mean** | **final-ep** | ___ | **0.8124/0.7936** | **___** | **0.8698/0.8618** | **___** |

### 7.3 Fixed write-up format

`ZH:     final-epoch: <pass/fail> (K-C2-1) · K-C2-2: <pass/tie> · ZH-robustness: <strengthened/not>.`
`HateMM: final-epoch: <pass/fail> (K-C2-1, hold) · K-C2-2: <pass/tie>.`
(+ val-selected line each; + MARGINAL note if a K-C2-1 acc pass is within noise, per B3 §2.2 precedent.)

---

## 8. What a PASS / FAIL means for the goal (D7 boundary — this prereg does NOT decide)

- **K-C2-2 PASS on ≥1 dataset (esp. ZH) + ZH-robustness strengthened (recon prior ~40–50%):** the 2-dataset story
  upgrades from "generic encoder LoRA (D7-weak, protocol-dependent on ZH)" to "memory-coupled adaptation
  curriculum (protocol-robust on ZH)" — strengthening the case for a **D7 sub-ruling** that memory→adaptation
  coupling is novel-in-field. **The ruling remains the USER's (F0.3).** Caveat: single curriculum draw (F0.2).
- **K-C2-2 TIE (recon prior ~50–60%, F0.7):** the coupling earns no novelty; cand-2 reduces to "generic LoRA with
  reshuffled data" — the honest empirical closure that even encoder-capacity-allocation is redundant with the
  head's frozen-feature mining. Bank the negative; the adaptation family is exhausted on this axis.
- **KS-regression / KS-below-floor:** the confusion-curriculum hurt — bank "curriculum degrades adaptation."

**Framing sentence (verbatim):** *this measurement tests whether a retrieval-coupled adaptation curriculum adds
over generic encoder LoRA; a PASS strengthens the case for a D7 sub-ruling that memory→adaptation coupling is
novel-in-field, but the ruling remains the user's.*

---

## 9. Provenance index

- Recon (GO-IF; design (i) ruling, leakage audit, kill skeleton): `refine-logs/CAND2_CURRICULUM_RECON.md` (`7087b5a`).
- Load-bearing prior kill (D7 distinction): `refine-logs/C3GEO_FORENSIC_RECON.md`; `state/directions_tried.json` #19 `R3-C3geo`.
- Generic arms + floors (re-derived §2): ZH `refine-logs/B3_PREREG_REVIEW.md` (job 13150 == arcbase 12223-25);
  HateMM `refine-logs/LORA_HATEMM_VERDICT_REVIEW.md` (job 13235) + `refine-logs/LORA_HATEMM_PREREG.md`.
- Protocol + decision rule (verbatim): `research-wiki/experiments/exp-encoder-3seed.md:73-85`.
- Mining machinery ($0 CPU): `scripts/analysis/cross_channel_router_gate.py:73-131`; frozen caches
  `data/CLIP_Embedding/{MHC_zh,HateMM}/train_Qwen2.5-VL-7B-Instruct_HF.pt`.
- Extraction prompt (leakage): `src/utils/generate_VideoMLLM_embedding_lora_HF.py:59-66`.
- Mechanism: `refine-logs/B3_ZH_LORA_DECOMPOSITION.md` (F45, text-stream Pareto), `refine-logs/ENCODER_SWAP_DIAGNOSIS.md` (F44).

**Required statements:** ZERO GPU/SLURM/Modal spent by this prereg author (only pure-CPU login-node mining +
data-build, seconds). No held-out test metric produced. All floor/generic numbers re-parsed from banked
completed-run trainlogs (numeric-provenance discipline). No `state/` mutated. NO job submitted. Not pushed.

---

## 10. (reserved)

---

## 11. DEVIATIONS FROM THE RECON — flagged loudly

1. **DEV-1 (answer variant: `_yn` → WORD). LOAD-BEARING.** The recon skeleton (Appendix H-C2, §1.3) writes
   `train_curric_yn.json` / `<prefix>_lora_curric_yn_train` (the **yes/no** variant). **That is wrong for this
   regime.** The actual generic-LoRA encoder arms — B3 (`mhc_zh_lora_train`) and LoRA-HateMM (`hatemm_lora_train`)
   — both trained on the **word** variant (`train.json`, hateful/normal · 仇恨/正常); the `_yn` variant belongs to
   P9's decision-level `sft_classifier` regime (`LORA_HATEMM_PREREG.md §1.1`). Since K-C2-2's validity requires the
   curriculum arm to be byte-identical to the generic arm **except the example distribution**, the curriculum
   **MUST** use the word variant too — else answer-format becomes a second manipulated variable and confounds the
   add-over-generic comparison. **Resolution: the builder forks the word-variant `train.json`; the pinned generic
   HateMM source sha (`93c6d3d1…`) matches the LoRA-HateMM prereg exactly.** This is the single most important
   deviation and it strengthens the design.

2. **DEV-2 (mining vote fusion pinned). Neutral.** The recon left the frozen LOO signal underspecified ("|LOO
   signed vote|"). I pinned it to a **fused** vote = top-20 rank-weighted signed-cosine over the **concat of
   L2-normalised img+text** (⇒ neighbour sim = mean of the two modality cosines), the closest frozen analogue to
   the align-fusion head's joint space, using the router-gate vote machinery verbatim. Rationale: the head fuses
   both streams, so the curriculum should target the boundary in the fused space (not one modality). K-C2-0
   validates non-degeneracy (LOO error ≈ 0.20 both).

3. **DEV-3 (allocation: deterministic largest-remainder, NO RNG — instead of stochastic resampling).
   Favorable.** The recon said "pin the RNG seed for sampling." I use **deterministic largest-remainder
   apportionment** to exactly `N_train` (no RNG on the registered softconf path), which (a) removes the ~37%
   bootstrap-dropout artifact of with-replacement resampling (that artifact is sampling noise unrelated to the
   curriculum and would weaken the "only distribution changed" claim), and (b) yields **bit-exact idempotent**
   `train_curric.json` (verified: two builds, identical sha) — a stronger reproducibility guarantee than a seeded
   sampler. A nominal `SEED=20260718` is retained only for the (unused) stochastic paths.

4. **DEV-4 (curriculum hyperparameters pinned by a $0 CPU sweep). Neutral.** `TAU=0.20`, `LAMBDA=10.0`,
   `CAP_RATIO=1.0` were selected from a pre-GPU sweep to give a **fair curriculum contrast** (~2.1× mass on the
   top-30% confusable head) at **~67% unique coverage** (dropping ~33% of the easy tail) — balancing "too mild ⇒
   certain tie" against "too aggressive ⇒ coverage-starvation KS-regression." The ~33% easy-drop is pre-declared
   as the KS-regression risk (§3.5).

5. **DEV-5 (new sbatch instead of a case-add to `lora_sft.sbatch`). Favorable.** `lora_sft.sbatch` STEP 1
   hardcodes the generic builder; a case-add there would branch the generic reproduction path and mutate a file
   already hash-pinned by the LoRA-HateMM verdict. A clean new `lora_sft_curric.sbatch` (which runs the generic
   build 1a THEN the curriculum build 1b) is non-invasive and leaves the generic path byte-untouched.

6. **DEV-6 (K-C2-2 threshold made self-consistent). Documented.** The recon states both "beats generic beyond the
   ±0.014 tie-band" and (§3.2) "~+0.007–0.010 over generic" — internally in tension (a +0.010 mean sits *inside*
   ±0.014). Resolution (§3.3): ±0.014 is the between-**seed spread** descriptor; the decisive statistic is the
   3-seed **mean** paired gain, whose noise is ~std/√3 ≈ 0.003, so K-C2-2 = **mean Δacc ≥ +0.010 AND 3/3 sign**
   (the 3/3-sign requirement is the teeth against a within-spread tie), and the ±0.014 band governs KS-regression
   (below-generic kill). The recon's "solidification" target migrates to the explicit **ZH-robustness clause** (§3.7).

7. **DEV-7 (both datasets, per the task; not branch-gated). Documented.** The recon sequenced cand-2 per the
   LoRA-HateMM branch (queue nothing until that verdict). That verdict has landed **PASS both protocols**
   (BRANCH A, `LORA_HATEMM_VERDICT_REVIEW.md`), so this prereg covers **both** ZH (primary) and HateMM (hold) as
   instructed — never as a new-dataset bet (F0.4).
