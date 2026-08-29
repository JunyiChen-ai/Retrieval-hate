# R12 FREEZE — two pilots, frozen before any pilot code exists

Date 2026-08-18. Round 10 of idea discovery (`IDEA_REPORT.md` §13).
This document is committed **before** `idea-stage/r12_img/` and `idea-stage/r12_anchor/` exist and
before any arm metric on any R12 seed range has been computed. Nothing below may be changed after
the first frozen run is submitted; deviations go in dated `R12_DEVIATION_D*.md` memos filed
**before** any arm metric is read.

API cost of both pilots: **¥0.00** (no paid API call of any kind). Round budget ¥15, cumulative ¥0/¥60.

---

## 0. Why these two, and what was rejected

Twelve candidates were generated from three literature reconnaissance sweeps
(breakage-set mechanisms; image-stream position axis; label-free feature transforms) and scored by
gpt-5.6-sol at xhigh reasoning, instructed to be hostile and given the full constraint map, the
union result and the effective-rank diagnostic. Its ranking, composite 0-10:

| rank | candidate | composite | reviewer's one-line verdict |
|---|---|---|---|
| 1 | **B2 IMGSPLIT** | 4.8 | readout axis genuinely untouched; not a method paper |
| 2 | **B1 IMG2M** | 4.4 | second-order image-token information unmeasured; standard ablation |
| 3 | A5 STABSEL | 3.4 | stability identifies reproducibility, not usefulness |
| 4 | C3 DMD-SEP | 3.1 | a DMD transfer to two blocks of one backbone |
| 5 | **A1 FOCAL-ANCHOR** | 3.0 | mechanism difference, not a new family; violates the literal anti-repeat rule |
| 6 | A4 SPECDEC | 2.9 | off-the-shelf regularisation |
| 7 | B3 IMGSINK | 2.7 | high norm ≠ irrelevant; dangerous free choice of k |
| 8 | A2 ELODI-ENS | 1.7 | distillation of a teacher ensemble that does not beat CAT |
| 9 | C2 DBAT | 1.5 | diversity has no route to accuracy without selection (banned) or averaging (failed) |
| 10 | A3 RECONCILE | 1.0 | the disagreement region holds too few held-out items |
| 11 | B4 IMGFRAME | 0.7 | directly absorbed by four prior temporal/segment failures |
| 12 | C1 PIDU | 0.1 | analysis, out of scope by the method-paper-only rule |

Two pilots are frozen: **R12-IMG** (B1+B2 in one extraction batch, the reviewer's own choice) and
**R12-ANCHOR** (A1, reduced to the identifiable core after the reviewer's algebraic objection).

### 0.1 Standing rulings adopted into this freeze

**(a) The anti-repeat ban and the focal filter.** `research-wiki/TARGET_GATE0_ITER6_LITERATURE.md`
line 103 bans "loss/difficulty bin reweighting" as a renamed robust-learning trick. The reviewer
ruled that `1(reference correct)` is a correctness bin and that weighting a KL term by it is
literally inside that ban. **Ruling for this round:** the ban is scoped to iteration 6 of the RGCL
full-bank campaign (line 92 states it as *"Iteration 6 的 binding anti-repeat"*), whose stated
rationale is that a full-bank vector intervention must not be reconstructible by scalar sample
weighting. That rationale does not bind a head-level distillation experiment on a different
substrate. The ban is therefore **narrowed, explicitly and in advance**, to the campaign it was
written for. To keep the narrowing honest, the reviewer's **shuffled-correctness-mask control is
adopted as a hard clause**: if an arbitrary class- and prevalence-matched bin reproduces the
effect, the candidate is killed regardless of its contrast against CAT.

**(b) The double-source term is dropped as a separate mechanism.** The reviewer showed that
`Σ_s w_s KL(p‖q_s)` is, up to a constant, anchoring to a single weighted geometric-mean
pseudo-teacher with total strength `Σ_s w_s`, and that in binary classification that pseudo-teacher
carries one scalar logit. Accepted. R12-ANCHOR therefore tests only what is identifiable: the
**focal (reference-correctness-gated) filter**, against uniform anchoring, on both a single-source
teacher and the algebraic two-source pseudo-teacher. No arm claims "double source" as a mechanism.

**(c) No method-paper claim is pre-authorised for R12-IMG.** The read-out primitive is occupied
(DINOv2's frozen linear eval is class-token ⊕ mean-pooled patch tokens; `2506.10178` ICLR 2026
benchmarks 13 structured read-outs against global average pooling; in the hate domain HateSieve
`2408.05794` and xDORA `2602.19212` both already read more than one visual position). R12-IMG is
frozen as a **terminal feature-engineering check**: it may bank a stackable feature default and it
closes the last never-varied read-out in the substrate. If it wins it is a feature default and an
ablation row, exactly as CAT and the layer axis were graded. **It may not be written up as a
pooling-method contribution.**

**(d) Already-closed items re-confirmed, not re-run.** The label-free-transform sweep flagged the
macro-F1 operating point (a fixed 0.5 threshold is not macro-F1-optimal) as a possible 1-3 point
bug class. This project already priced it: `R8_DECOMP_MEMO.md` §3 caps every decision-rule /
calibration mechanism at **+0.25 to +1.2 points** with a train+val-fitted global threshold oracle,
and a dev-fitted threshold is **negative on 3 of 4 datasets**. No pilot. Likewise mean-direction
projection, per-dimension standardisation and full/partial whitening are affine or diagonal-linear
maps in front of a **dense** first layer, hence exact reparameterisations of the head's function
class; the two in-house whitening failures (`DRAFT_analysis_chapter.md` §3.13 MECHFIX, `PCD_SPEC.md`)
and the three in-house dev-positive/test-negative artefacts (LL, PC0, MC) are consistent with that.
No pilot.

---

## 1. Shared protocol (both pilots)

- Harness: `idea-stage/reaudit/run_grid.sh` **unchanged**, byte-identical hyperparameters to
  `r6_confirm/run_confirm.sh` (batch 64, lr 1e-4, 30 epochs, topk 20, proj/map 1024,
  dropout 0.2/0.4/0.1, fusion align, triplet+hybrid, warmup 5, `--contrast_mode none`,
  `--final_eval False`, `--Faiss_GPU False`).
- Read-outs: **P1 (primary)** = test macro-F1 at `argmax_{e>=5}` dev macro-F1. **P2** = test
  macro-F1 at the final epoch. Dev-side panel reported for the REAUDIT_NCA selection-rule check.
- Statistics: seed-paired mean, paired bootstrap 95 % CI, B = 20000, bootstrap seed 20260817,
  via `idea-stage/reaudit/analyze_grid.py` (unchanged).
- Seeds: **R12-IMG** MHC_zh 800-829 (30), HateMM 800-814 (15).
  **R12-ANCHOR** MHC_zh 900-929 (30), HateMM 900-914 (15).
  Both ranges are disjoint from every previously consumed range
  (0-119, 30-89, 100-129, 200-229, 300-329, 400-429, 500-529, 600-629, 700-729, 41000-41029,
  50700-50729, 20260900-20260929).
- Datasets: **MHC_zh** (adapter `Qwen2.5-VL-7B-Instruct-LoRA_HF`) and **HateMM** (adapter
  `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`). MHC-EN has no read-out cache and ImpliHateVid has no
  raw video; neither is in either pilot and neither may be added after results are seen.
- **Zero test-label tuning.** Every epoch rule, arm definition, teacher, weight, threshold, split
  and control is fixed by this document or computed from train/dev only.
- **Single submission per pilot.** `idea-stage/r12_img/run_all.sh` and
  `idea-stage/r12_anchor/run_all.sh`, each submitted exactly once. The only pre-run execution
  permitted is a `--smoke` path that prints wall-clock, epoch count and a NaN flag and **no arm
  metric**.
- Analyzer runs exactly once per pilot, on the complete grid. A HALT on any missing run; no subset
  verdicts.
- Belts required before the verdict is read (each must pass or the pilot HALTs):
  (i) macro-F1 recomputed from the dumped per-item logits matches the trainlog to max abs diff 0.0
  on every run; (ii) test id order identical across all arms; (iii) every arm cache's sha256
  recorded in a build-meta JSON; (iv) for R12-ANCHOR, the `λ = 0` arm reproduces the no-anchor code
  path exactly.
- **No absolute number from either pilot is comparable to the project's A100-extracted ledger.**
  Only within-table contrasts are results. This is the standing R10 deviation-D1 rule.

---

## 2. Pilot R12-IMG — the image stream's position axis

### 2.1 What the deployed image read-out is

`src/utils/generate_VideoMLLM_embedding_readout_HF.py:_pool_span(span="prefix")`: the mean of the
layer-28 hidden states over **every position from 0 up to the last `<|im_start|>`** of a forward
whose prompt is `IMG_INSTRUCTION = "Describe the people, symbols, gestures, and on-screen text in
this video."` with 8 sampled frames. That is ~1000 positions — the video block plus a short
instruction — collapsed to one mean. It is the only read-out this stream has ever had.

### 2.2 The premise, measured this round on train splits only

Within-class effective rank (native dimension, exponential of the entropy of the covariance
eigenspectrum), computed on the deployed caches, train split, **no test contact**:

| dataset | img (class 0 / 1) | text (class 0 / 1) | img top-1 PCA variance | text top-1 |
|---|---|---|---|---|
| MHC_zh | 31.5 / 24.7 | 74.6 / 48.4 | 0.283 | 0.133 |
| HateMM | 37.0 / 37.6 | 92.3 / 54.9 | 0.233 | 0.157 |

The image stream carries roughly half the within-class effective rank of the text stream and
concentrates twice as much variance in one direction. This is the symmetric counterpart of the
geometry that produced CAT. **Declared scope limit, in advance:** low effective rank is consistent
with destructive pooling *and* with beneficial denoising. This diagnostic motivates the pilot; it
does not predict its sign, and it will not be used to argue the result either way.

### 2.3 Extraction (one pass per dataset, layer 28 only)

`idea-stage/r12_img/extract_img.py`, a thin fork of the frozen read-out extractor. Everything that
touches the model, the frame sampler (8 frames, `max_pixels = 360*420`), the prompt string, the
LoRA adapter, the cache contract and the pooling math is imported verbatim from
`generate_VideoMLLM_embedding_readout_HF`. This file re-implements only *which position spans are
pooled*, and runs only the **image** forward.

Positions, from the single forward's `input_ids`:
`v_end = 1 + index of the last <|video_pad|>`; `hdr = index of the last <|im_start|>`.
If `v_end >= hdr`, `v_end = 0` (degenerate guard, identical to the R10 fork).

Spans written, all at layer 28, each L2-normalised row-wise after pooling:

| name | definition |
|---|---|
| `PRE` | `mean(h[0:hdr])` — **bit-identical to the deployed `prefix` span** |
| `VIS` | `mean(h[0:v_end])` — the vision block |
| `INS` | `mean(h[v_end:hdr])` — the instruction-text positions |
| `STD` | `std(h[0:hdr], dim=0, unbiased=False)` — elementwise second moment over the same positions |
| `RA` | `mean(h[S])`, `S` = a fixed random subset of `[0,hdr)` of size `v_end` |
| `RB` | `mean(h[[0,hdr) \ S])` — the complement |

`S` is drawn per video from a `numpy.random.default_rng` seeded with
`20261218 + hash-free integer index of the video in the split order`, so it is deterministic,
reproducible and recorded; it is **not** re-drawn and no arm may be rebuilt with a different draw.

Belt B1: on the first 12 videos of each split, `PRE` must equal the frozen
`_pool_span(last_hidden, input_ids, "prefix", im_start_id)` computed on the same forward with
**max abs diff exactly 0.0**. Failure HALTs the pilot.
Belt B2: span statistics (median total length, `v_end`, `hdr`) are printed and recorded.
Zero-vector guard for undecodable videos is carried over verbatim.

Output: `data/CLIP_Embedding/<DS>/{train,dev_seen,test_seen}_<BASE>-ip.pt` with
`{ids, labels, spans: {"28": {name: [N,3584]}}, meta}`. The suffix `ip` has never been used, so no
banked cache can be clobbered.

### 2.4 Arms (`idea-stage/r12_img/build_img.py`, prefix `R12IM`)

**The text stream is `CAT` in every arm** — `[n(A0_28) ‖ n(TXT_28)]`, 7168-d, read from the banked
`-tp` caches, byte-identical across arms, so it cancels in every contrast. Only `img_feats` differ.
`n(·)` = row L2-norm, applied per block before concatenation.

| arm | img_feats | dim | role |
|---|---|---|---|
| **I0** | `n(PRE)` | 3584 | **reference** — the deployed read-out, re-extracted in this pass |
| **ISPLIT** | `[n(VIS) ‖ n(INS)]` | 7168 | **candidate B2** — the image-side analogue of CAT |
| **I2M** | `[n(PRE) ‖ n(STD)]` | 7168 | **candidate B1** — second-moment read-out |
| **IRSPLIT** | `[n(RA) ‖ n(RB)]` | 7168 | **control** — random positional split, same block sizes |
| **IRW** | `[n(PRE) ‖ n(PRE·R)]` | 7168 | **control** — matched width, no new information |
| *IVIS* | `n(VIS)` | 3584 | diagnostic, non-selectable |
| *IINS* | `n(INS)` | 3584 | diagnostic, non-selectable |
| *ISTD* | `n(STD)` | 3584 | diagnostic, non-selectable |

`R` is the **same fixed Gaussian matrix** used by `idea-stage/r6_readout/build_arms.py` and
`r10_tokpos/build_arms.py`; its sha256 is asserted against `r6_readout/build_meta.json` at build
time, exactly as R10 did.

8 arms × (30 + 15) seeds = **360 head-training runs**, ~60 min on the local 5090.

### 2.5 Decision rule — frozen

A candidate `C ∈ {ISPLIT, I2M}` **STANDS** iff **all four** clauses hold **on both datasets**:

1. `C − I0` P1 mean **≥ +0.005** with the paired-bootstrap 95 % CI excluding zero;
2. `C − IRW` P1 mean **≥ +0.005** with the CI excluding zero (the gain is not width);
3. `C − IRSPLIT` P1 mean **≥ +0.005** with the CI excluding zero (the gain is not "any second
   view of the same forward"; this clause applies to **both** candidates, not only to ISPLIT);
4. `C − I0` under **P2** has a 95 % CI whose **lower bound is ≥ −0.005** (P2 may not rescue a
   failed P1, but P2 must not support harm).

If **both** candidates STAND, the pre-committed tie-break is, in order: (i) fewer distinct pooling
operations, (ii) smaller total feature width, (iii) `ISPLIT`, which is named first here. No
selection by test number is permitted.

If **neither** STANDS, the pre-committed conclusion is: *the image stream's flat prefix mean is not
improved by either a semantic positional split or a second-moment read-out at this sample size; the
last never-varied read-out in the substrate is measured and closed.*

**Demotion clause (REAUDIT_NCA check).** Any arm whose P1 test contrast against `I0` is positive
while its **dev** contrast against `I0` is negative with the CI excluding zero is recorded as
selection-rule-bound and **cannot STAND**, whatever clauses 1-4 say. This is the fourth deployment
of this check and it has fired on a control every time it was armed.

`IVIS`, `IINS`, `ISTD` have no verdict power and may not be promoted to candidates.

---

## 3. Pilot R12-ANCHOR — reference-correctness-gated distillation

### 3.1 The target quantity

`R11_UNION_RESULT.md` §3: `CAT` already retains **0.650** (MHC_zh) / **0.822** (HateMM) of the
`CAT ∪ LL` fix pool. The unpurchased headroom is entirely in the **breakage** column — `CAT`
newly breaks **4.07** / **2.40** previously-correct items per seed — and every one of five
mechanisms traded breakage against retention roughly one for one. Uniform out-of-fold soft
anchoring already moved breakage the right way (4.07 → 3.67) and paid for it in retention
(0.650 → 0.489).

The positive-congruent-training literature claims the missing ingredient is that the anchoring term
must be **gated on the reference model's correctness** — anchor hard where the reference is right,
release where it is wrong. Prior art: PC-Training `2011.09161` (CVPR 2021 oral), ELODI `2205.06265`
(TPAMI 2024), MPT `2511.08322`. The single most relevant reported number is MPT Table 4, a
**frozen ViT-B/32 with only the classification layer trained**: negative-flip rate 7.08 → 3.26 and
old-class error 19.44 → 15.16 **simultaneously** — the counterexample to the one-for-one wall.
Occupancy in hate / harm / toxicity / meme detection: zero papers.

**Declared in advance, from the hostile review:** MPT's regime is CIFAR-100, ~50 000 training items
and 100 logits carrying inter-class structure. This pilot is 579/744 items and one binary logit,
where KL is mostly margin matching. The MPT number establishes that the mechanism can work
somewhere; it is **not** taken as a quantitative prior here, and the pilot is justified by the
in-house breakage number, not by MPT's effect size.

### 3.2 Teachers — reused, sha-verified, never refitted

`idea-stage/r11_union/teacher_{MHC_zh,HateMM}_{A0,LL,LBL}.json`, the R11 5-fold out-of-fold
logistic probes (train macro-F1 0.766/0.864 for A0 and 0.810/0.875 for LL; mean |q − y| 0.25-0.31;
under 3 % of items with q within 0.05 of their label). Their sha256 is asserted against
`r11_union/build_meta_*.json` at build time. **No teacher is refitted for this pilot**, so no new
degree of freedom enters.

Derived once, deterministically:

- **Pseudo-teacher** `PT`: `logit_PT = 0.5·(logit_A0 + logit_LL)`, `q_PT = sigmoid(logit_PT)` —
  the algebraic single teacher the reviewer showed a two-source forward-KL sum reduces to.
- **Correctness mask** `m_i = 1[(q_i > 0.5) == y_i]` per teacher, computed from the **out-of-fold**
  probability, never in-sample.
- **Focal weight** `w_i = (α + β·m_i) / mean_train(α + β·m_i)` with **α = 1.0, β = 3.0**, frozen,
  identical on both datasets, no per-dataset choice. The normalisation makes
  `mean_train(w) = 1` exactly, so uniform and focal arms carry **equal expected anchor mass** and
  the contrast cannot be an effective-λ artefact.
- **Shuffled mask** `m̃`: `m` permuted **within each (class, teacher) stratum**, so class prevalence
  and per-class correctness rate are preserved exactly; RNG `numpy.random.default_rng(20261218)`,
  drawn once, written into the build meta, never redrawn.

- **λ = 0.1 for every anchored arm, both datasets, frozen.** R11 dev-selected λ from {0.1, 1.0} and
  its own §2.4 showed dev selection is corrupted by the anchor family (both anchor arms fit dev
  better with the CI excluding zero and scored worse on test). λ is therefore **fixed, not
  selected**, at the value dev chose in 3 of the 4 R11 cells.

### 3.3 Code change — additive, default-off

`src/model/loss.py::compute_anchor_loss` currently returns the **unweighted mean** of the
per-item soft-target BCE. One new optional argument `--anchor_weights <json>` (id → w) makes it the
**w-weighted mean** of the same per-item terms. Absent (the default, and every previously banked
run) → byte-identical behaviour. Belt: at `--lambda_anchor 0` the whole path is untouched, and one
arm at one seed must reproduce an R11 trainlog metric line for line.

### 3.4 Arms (`idea-stage/r12_anchor/build_r12a.py`, prefix `R12AN`)

All arms use the **same `CAT` feature cache** (the R10-COMBO `R10CB-CAT` cache, sha-verified
against `r10_combo/build_meta_*.json`). Only the loss differs.

| arm | teacher | weighting | role |
|---|---|---|---|
| **CAT** | — (λ = 0) | — | **reference** |
| **AU_A0** | A0 | uniform | control, reproduces R11 `ANCA` at fixed λ |
| **AF_A0** | A0 | focal | candidate, single source |
| **AU_PT** | PT | uniform | control, the uniform two-source pseudo-teacher |
| **AF_PT** | PT | focal | **primary candidate** |
| **AF_SHUF** | PT | focal, **shuffled** mask | control — arbitrary matched bin |
| **LBL** | hard labels | uniform | control, R11's best anchor-family arm |

7 arms × (30 + 15) seeds = **315 head-training runs**, ~55 min.

### 3.5 Decision rule — frozen

`AF_PT` **STANDS** iff **all three** clauses hold **on both datasets**:

1. **Gain**: `AF_PT − CAT` P1 mean **≥ +0.005** with the 95 % CI excluding zero;
2. **Filter identification**: `AF_PT − AU_PT` P1 mean **> 0** with the CI excluding zero — the
   focal gate, not the anchoring term, is doing the work;
3. **Semantics identification**: `AF_PT − AF_SHUF` P1 mean **≥ +0.005** with the CI excluding
   zero — reference *correctness*, not an arbitrary class-matched bin.

`AF_A0` is judged by the same three clauses with `AU_A0` and a single-source shuffled comparison
substituted; it is a **secondary** candidate and cannot be promoted over `AF_PT` by a larger test
number. If `AF_A0` stands and `AF_PT` does not, the entry is `AF_A0` and the pseudo-teacher is
recorded as unnecessary.

Outcomes, pre-committed:

- clause 1 fails on either dataset → **KILL**. The recorded conclusion is: *reference-correctness
  gating does not break the retention/breakage exchange rate on this substrate; the
  positive-congruent-training family joins the four already-measured combination families, and the
  round's own recommendation ("find a mechanism that changes which items get broken") is measured
  and negative.*
- clause 1 holds, clause 2 or 3 fails → **BANKABLE CONFIGURATION, MECHANISM NOT IDENTIFIED.** It
  may be recorded as a default; it may not be described as a positive-congruent-training result.
- all three hold → **STANDS**, and the union accounting below becomes the mechanism evidence.

**Demotion clause (REAUDIT_NCA check).** Same as §2.5: any arm that is dev-negative with the CI
excluding zero and test-positive cannot STAND. R11 §2.4 measured both anchor arms as
**dev-positive and test-negative** on MHC_zh, so the mirror check is also armed: an arm that fits
dev better with the CI excluding zero while losing on test is reported as an overfitting signature.

### 3.6 Mandatory secondary read-out (no verdict power)

Union accounting reproduced exactly as `R11_UNION_RESULT.md` §3: per seed, per arm, the fraction of
the `A0`-error pool that `CAT` or `LL` gets right which the arm retains, the count of `A0`-correct
items the arm newly breaks, and the net. This is the quantity the pilot exists to move; it is
reported whatever the verdict, and it is **not** part of the decision rule.

---

## 4. What each outcome licenses, written before the numbers exist

| outcome | what is licensed |
|---|---|
| R12-IMG stands | a feature default on two datasets and an ablation row; **not** a pooling-method claim (§0.1c) |
| R12-IMG fails | the last never-varied read-out in the substrate is measured and closed |
| R12-ANCHOR stands, all clauses | the first identified mechanism in this project that changes which items break; a method-paper candidate, subject to a fresh novelty check |
| R12-ANCHOR stands, clauses 2/3 fail | a bankable configuration only |
| R12-ANCHOR fails | the round's own recommended mechanism class is measured negative; the closure argument in §13 of `IDEA_REPORT.md` is complete |

Neither pilot may be re-run, re-seeded or re-scoped after its analyzer runs. If both fail, the
round's honest output is the closure statement and the list of remaining validation for CAT, not a
third pilot.

---

## 5. Ledger

| item | value |
|---|---|
| API spend | **¥0.00** (round budget ¥15, cumulative ¥0/¥60) |
| GPU | local RTX 5090; ~35 min extraction + ~60 min + ~55 min head grids |
| test-label contact | final metric only; nothing selected on test in either pilot |
| seeds consumed | 800-829 / 800-814 (R12-IMG), 900-929 / 900-914 (R12-ANCHOR) |
