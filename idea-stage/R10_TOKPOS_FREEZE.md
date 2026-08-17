# R10 freeze — spectral saturation diagnostic (Task A) + token-position readout pilot (Task B)

**Written and committed BEFORE any Task-A or Task-B number exists.** Round 10, budget ¥0 (no API),
local RTX 5090, no SLURM on this machine. Four hard red lines apply: no test-label tuning; decision
rules frozen here before running; no candidate metric computed during design; each frozen run
submitted once.

Date: 2026-08-17. Author: research agent. Light-ceremony (CPU/single-GPU, cheap) — one freeze doc,
no external review, per `CLAUDE.md` §实验流程.

---

## 0. Shared substrate facts (established before this freeze, no labels touched)

- Feature caches: `data/CLIP_Embedding/<DS>/{train,dev_seen,test_seen}_<tag>.pt`,
  dict `{ids, img_feats [N,3584], text_feats [N,3584], labels [N]}`.
- Deployed tags: HateMM `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`; MHC (EN) and MHC_zh
  `Qwen2.5-VL-7B-Instruct-LoRA_HF`; ImpliHateVid `Qwen2.5-VL-7B-Instruct_HF`.
- Split sizes (n / pos / neg): HateMM 744/298/446, 107/43/64, 215/86/129 · MHC 549/168/381,
  80/25/55, 161/49/112 · MHC_zh 579/180/399, 78/28/50, 149/45/104 · ImpliHateVid 1283/649/634,
  325/160/165, 401/200/201.
- **No cache anywhere stores per-token hidden states.** Confirmed by an exhaustive sweep of every
  `.pt/.npy/.npz` under `data/`, `idea-stage/`, `artifacts/`: the only ≥3-D tensors are
  layers×heads (`artifacts/sav_f0`), frame groups, audio segments and synthetic probe cells.
  Task B therefore requires a new extraction pass.
- **What the deployed readout actually is** (`src/utils/generate_VideoMLLM_embedding_HF.py`,
  `..._readout_HF.py`): two *separate* forwards per video, both frozen, causal.
  - `img_feats` = mean of layer-28 hidden states over `[0, last <|im_start|>)` — the whole
    vision+instruction prefix.
  - `text_feats` = mean of layer-28 hidden states over `[last <|im_start|>, end)` — the trailing
    `<|im_start|>assistant\n` header, **3–4 format tokens**. This is a last-token readout.
  - The text-stream prompt is `TEXT_INSTRUCTION + "\nTitle: (none)\nTranscript: " + text`, where
    `text` is title+transcript pre-merged by `scripts/prep_mhc.py`. The prompt also carries the
    same 8 video frames (≈768 vision tokens). Median sequence ≈ 930 positions, ≈82.5 % vision.
  - **Consequence: the title/transcript content positions are pooled by neither stream.** The img
    stream's forward uses a *different* prompt (no transcript at all); the text stream's forward
    contains the transcript but reads only the assistant header. This is exactly the readout
    geometry that arXiv 2605.12726 criticises, and it is untested here under causal attention.
- Prior project work on pooling spans (`refine-logs/MNTP_S1_RECORD.md`, S1/S1b) tried all-position
  and text-position-only mean pooling, but **only under a flipped bidirectional attention mask**,
  which collapses the two streams onto each other (cosine 0.76–0.93) and confounds the readout
  question. Under the deployed **causal** mask no alternative span has ever been measured.

---

## 1. Task A — spectral saturation diagnostic (arXiv 2606.24903)

### 1.1 The test statistic, as published

Gupta, *The Geometry of Saturation: Effective Rank Predicts When Labels Stop Helping in Few-Shot
Classification*, arXiv 2606.24903v2.

For an N-way task with K examples per class:

```
x̄_c        = (1/K) Σ_{i∈c} x_i
Σ̂_c^(K)    = (1/(K-1)) Σ_{i∈c} (x_i - x̄_c)(x_i - x̄_c)ᵀ
Σ̂_W^(K)    = (1/N) Σ_c Σ̂_c^(K)                       # pooled within-class covariance
p_i        = λ_i / tr(Σ̂_W^(K))                        # normalised eigenvalue spectrum
erank      = exp(-Σ_i p_i log p_i)                     # Roy & Vetterli (2007)
S(K)       = erank(Σ̂_W^(K)) / K
```

Decision rule as published (paper §7, "practical two-regime rule", calibrated for unregularised
linear probes):

- **PCA-50 regime** (StandardScaler + 50 principal components, basis fit on the support set only):
  **hard stop when S(K) < τ = 0.02.** This is the regime in which τ was calibrated and in which
  the reported cluster-bootstrap AUC = 0.787 [0.713, 0.860] for stop/continue was measured.
- **Foundation-model regime** (native dim, ℓ2-normalised, no PCA): no hard threshold; the paper
  gives a *monitoring* band, S(K) declining from ≈0.3 to ≈0.05 = diminishing returns.

### 1.2 Our protocol (frozen)

`idea-stage/r10_sat/sat.py`.

- Data: **train + dev_seen only, concatenated. `test_seen` is never opened.** Labels are used only
  to form the class partition, which is what the statistic is defined on.
- Datasets: HateMM, MHC (EN), MHC_zh, ImpliHateVid, deployed tags above.
- Feature views, three per dataset: `concat` = `[img_feats ‖ text_feats]` (7168-d, the pair the head
  actually consumes), `img` (3584-d), `text` (3584-d).
- N = 2 (binary). `K_max` = min class count over train+val. K sweep on the paper's geometric grid
  `{2,4,8,16,32,64,128,256,512,1024,2048,4096}` truncated at `K_max`, plus `K_max` itself as a
  final point.
- 50 trials per (dataset, view, K): sample K items per class without replacement; fit
  StandardScaler + PCA-50 **inside the trial on the support set only** for the PCA-50 arm; report
  mean ± sd of `S(K)` and of `erank`.
- Second arm, no PCA, no standardisation (features are already ℓ2-normed by the extractor):
  native 3584/7168-d, same sampling.
- RNG: `numpy.random.default_rng(20260817 + trial)`, fixed.

### 1.3 Reading rule (frozen)

- The **headline reading is the PCA-50 arm at `K = K_max`**, against τ = 0.02: `S(K_max) < 0.02` →
  the paper says *stop acquiring labels*; `≥ 0.02` → *continue*.
- The native-dim arm is reported against the 0.3 → 0.05 monitoring band, with no hard verdict.
- **This is a diagnostic, not a verdict on any candidate.** No arm is killed or revived by it. It
  cannot pass or fail Task B. It is reported with its scope limits (§1.4) in
  `idea-stage/R10_TOKPOS_RESULT.md`.

### 1.4 Declared scope limits (written before the numbers)

1. The paper validates S(K) against **marginal accuracy gain from doubling the label budget**, on
   image classification with unregularised logistic-regression probes. Our head is a 3-layer
   HateClipper-align MLP with dropout and a triplet+BCE hybrid objective. The mapping from
   "labels stop helping" to "our head saturates" is an extrapolation.
2. τ = 0.02 was calibrated at d = 50. S(K) is invariant to invertible linear maps but **not** to
   dimensionality reduction, so only the PCA-50 arm is on-protocol.
3. S(K) = Θ(1/K) by construction (paper Thm 3.5). Any dataset with large K will read "stop"
   mechanically. The partial correlation controlling for log K in the paper is ρ = 0.324, i.e.
   most of the raw signal *is* the K-dependence. We must therefore compare the four datasets to
   each other and to erank, not read the absolute number as an oracle.
4. It says nothing about whether a *different representation* would help — only about whether more
   *labels* of the current representation would.

---

## 2. Task B — token-position readout pilot

### 2.1 Hypothesis

arXiv 2605.12726 (Doda, ICML 2026 MI workshop): a final-token probe misses probe-visible evidence
carried at earlier prompt positions; naive max-pooling over positions overfires; a position-aware
readout recovers the misses. Our deployed `text_feats` is a final-token readout over a sequence
whose title/transcript content sits at earlier, never-pooled positions.

**H:** under the same frozen causal Qwen2.5-VL-7B forward, a readout that pools the text-content
positions carries discriminative information the assistant-header readout does not.

### 2.2 Extraction (single pass, MHC_zh only)

Fork of `src/utils/generate_VideoMLLM_embedding_readout_HF.py` →
`idea-stage/r10_tokpos/extract_tokpos.py`. Changes, and only these:

- Runs **only the text-stream (baseline-prompt) forward**, one forward per video. `img_feats` for
  every arm are **carried over verbatim** from the banked `-ro_L28` / `-ro_L24` caches, joined by
  `id`. A parity belt asserts the id lists match exactly.
- From that one forward, at layers **28 and 24** (the two already-pinned layers, no layer sweep),
  it pools these spans and stores them all:
  - `A0` = mean over `[last <|im_start|>, end)` — the deployed span, byte-identical target.
  - `TXT` = mean over `[v_end, last <|im_start|>)` where `v_end` = 1 + index of the last
    `<|video_pad|>` token — i.e. the title/transcript/instruction content positions, excluding the
    vision block and the assistant header.
  - `S1..S4` = the same `[v_end, last <|im_start|>)` span cut into 4 contiguous equal segments
    (`numpy.array_split` semantics), each mean-pooled.
  - `ALL` = mean over `[0, seq_len)` — the naive all-position pool, kept as the paper's
    "naive pooling" reference point.
  Every pooled vector is `float()` then L2-normalised, exactly as `_pool_span` does today.
- Splits: `train`, `dev_seen`, `test_seen`. **Test features are extracted; test labels are not read
  by any decision in this freeze.** (Test inputs are unsealed per the 2026-08-09 ruling; test
  labels are read only by the final metric, as in every prior pilot.)
- Output tag prefix `R10TP-`, so no banked cache can be clobbered.

**Parity belt (must pass before any head run):** for every split, cosine between the extracted
`A0` at L28 and the banked `-ro_L28` `text_feats` must be ≥ 0.9999 on ≥ 99 % of rows. If it fails,
the pilot HALTs and the extraction is debugged; no arm result is read.

### 2.3 Arm construction

`idea-stage/r10_tokpos/build_arms.py`, CPU only. Every arm keeps `img_feats` = the banked
`-ro_L28` img stream (3584-d) unchanged; only `text_feats` differs. `n(·)` = row L2-norm.
`R` = the *same* fixed Gaussian (3584,3584) matrix as `idea-stage/r6_readout/build_arms.py`
(`numpy.random.default_rng(20260817)`, entries ~ N(0, 1/√3584)), rebuilt identically and
sha-checked against `idea-stage/r6_readout/build_meta.json`.

| arm | `--model` tag | text_feats | dim | role |
|---|---|---|---|---|
| **A0** | `R10TP-A0` | `n(A0_28)` | 3584 | **control = current deployed readout** |
| **TXT** | `R10TP-TXT` | `n(TXT_28)` | 3584 | same-width readout swap |
| **CAT** | `R10TP-CAT` | `[n(A0_28) ‖ n(TXT_28)]` | 7168 | both readouts |
| **RAND** | `R10TP-RAND` | `[n(A0_28) ‖ n(A0_28 @ R)]` | 7168 | **width control for CAT** |
| *SEG* | `R10TP-SEG` | `[n(A0_28) ‖ n(S1_28) ‖ … ‖ n(S4_28)]` | 17920 | **exploratory only, see §2.5** |

### 2.4 Run protocol (frozen)

Byte-identical to `idea-stage/reaudit/run_grid.sh` (which is byte-identical to
`r6_confirm/run_confirm.sh` → `r6_readout/run_arms.sh`), invoked unchanged:

```
bash idea-stage/reaudit/run_grid.sh <rundir> \
  "A0:R10TP-A0,TXT:R10TP-TXT,CAT:R10TP-CAT,RAND:R10TP-RAND,SEG:R10TP-SEG" \
  MHC_zh "<seeds>" R10TP
```

- Dataset: **MHC_zh** (smallest, largest historical readout effect).
- **Seeds 500–529 (30 seeds).** Disjoint from every consumed range in the project and from the
  concurrently running `REAUD_*` grid (seeds 300–329).
- Read-out: `idea-stage/reaudit/analyze_grid.py` unchanged — P1 = epoch `argmax_{e≥5}` dev
  macro-F1 (primary), P2 = epoch 29 (corroboration), test macro-F1 @ 0.5, paired bootstrap
  B = 20000, seed 20260817, bar 0.005. Run **once**.

### 2.5 Decision rule (frozen — this is the judgement)

Let `Δ(X)` = seed-paired mean of (test macro-F1 of arm X − test macro-F1 of A0) under P1, and
`CI(X)` its paired-bootstrap 95 % interval.

- **GO** iff at least one of the following holds:
  - `Δ(TXT) ≥ +0.005` and `CI(TXT)` excludes 0 and P2 agrees in sign; **or**
  - `Δ(CAT) ≥ +0.005` and `CI(CAT)` excludes 0 and P2 agrees in sign, **and additionally**
    `mean(CAT − RAND) ≥ +0.005` with its own 95 % CI excluding 0. (A wider arm must beat the
    matched-width random-projection control, not just the narrow control.)
- **AMBIGUOUS** iff some arm reaches `Δ ≥ +0.005` but fails the CI or the P2-sign clause or (for
  CAT) the width control.
- **KILL** iff no arm reaches `Δ ≥ +0.005` under P1.

`SEG` is **exploratory and reported without a verdict**. It cannot produce a GO on its own and it
is not part of any contrast above. It is in the grid because it is free (same forward, same seeds).

### 2.6 Conditional legs (only run if §2.5 returns GO)

- **Leg 2 — does it stack on L24⊕L28?** Let `W` ∈ {TXT, CAT} be the arm with the higher **dev**
  macro-F1 under P1 (dev, never test). Build:
  - `C0` = `R6RO-CAT`, already on disk: img `[n(img28) ‖ n(img24)]`, text `[n(A0_28) ‖ n(A0_24)]`.
  - `C1` = img `[n(img28) ‖ n(img24)]`, text = `C0` text with `W`'s L28 and L24 blocks appended.
  Same 30 seeds, same protocol. **Stacks** iff `mean(C1 − C0) ≥ +0.005` with 95 % CI excluding 0
  and P2 sign agreement. Otherwise the token-axis gain does not add to the strongest current
  configuration, and that is reported as such.
- **Leg 3 — second dataset.** HateMM (base tag `-LoRA-curric_HF`), the winning arm vs A0, **15
  seeds (500–514)**, same protocol and same bar. Requires a HateMM extraction pass under §2.2.

If §2.5 returns KILL or AMBIGUOUS, legs 2 and 3 are **not** run.

### 2.7 Budget and fallback

Extraction is 806 MHC_zh videos × 1 forward. If total extraction wall-clock would exceed ~3 h GPU
(e.g. because the shared 5090 has no free memory and the model must be CPU-offloaded), the pilot
falls back to **transcript-text-only extraction** — the same prompt with the video frames removed —
and the deviation is recorded in the result document with the loss of comparability to the banked
img stream stated explicitly. No other reduction is permitted.

### 2.8 What a GO would and would not license

A GO licenses: "on MHC_zh, pooling the transcript-content token positions of the same frozen causal
forward beats the deployed assistant-header readout by ≥ 0.005 test macro-F1". It does **not**
license a claim about other datasets (leg 3 decides that), about layers other than 28/24, or about
any mechanism beyond the readout span. A KILL is a kill of *this readout family on this substrate*,
not of the paper's claim about safety probes.
