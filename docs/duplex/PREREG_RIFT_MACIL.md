# RIFT-MACIL preregistration

Frozen before implementation or any RIFT validation/test result is inspected.

## Task and starting point

Task: weakly-supervised **frame-level** hateful-video localisation. Training may
use only parent-video labels as human supervision. The model outputs one
continuous hate score per frame/snippet; it does not predict temporal
intervals. The starting point is the official-validation port of MACIL-SD AV
(ACM MM 2022), not a reimplementation of MultiHateLoc.

Primary metric: pooled frame Average Precision (the `Frame PR` column in the
archived table). Secondary metric: pooled frame ROC-AUC. These are the metric
families used by LELA and MultiHateLoc. Test is never used for module choice,
checkpoint selection, threshold selection, or hyperparameter selection.

Current three-seed weak-supervision AP bars are 0.5733 HateMM, 0.4519 MHC-EN,
0.4614 MHC-ZH, and 0.5619 HateClipSeg. A SOTA claim requires strict improvement
over the relevant bar in the frozen evaluator; the training-free one-seed VERA
HateClipSeg result (0.6194) is reported separately and is not relabelled as a
weakly-supervised result.

## Hypothesis

MACIL-SD learns strong between-video separation but weak within-positive-video
ordering. Absolute pseudo frame labels from an MLLM inherit calibration and
class-prior errors. A frozen semantic teacher is more trustworthy about the
*relative* evidentiality of two regions from the same video than about either
region's absolute hate probability. Distilling only confident within-video
orders should suppress the video-identity shortcut while preserving MACIL's
strong audio-visual detector.

## Method modules

Feature extraction, ASR, temporal resampling, padding, and frozen feature
encoding are preprocessing and are explicitly **not** counted as modules.

### Module 1 — Masked Semantic Isolation Teacher (MSIT)

A frozen MLLM reads each ASR-aligned temporal unit under branch-local attention:
each unit can attend to the shared task definition and its own evidence but not
to sibling units. One packed forward produces a hate logit for every unit. The
teacher is used on train/validation media only and is absent at student test
time. Existing packed-vs-sequential equivalence tests must pass before its
outputs are accepted.

MSIT is not direct LELA-style inference: its score is privileged training
information and is never submitted as the final prediction.

### Module 2 — Reliability-Gated Intra-video Rank Distillation (RIRD)

For a positive training video, teacher units separated by at least a confidence
margin define ordered pairs. The MACIL student receives a pairwise ranking loss
on those pairs. Teacher-constant, uncovered, contradictory, and low-margin pairs
receive zero weight. The loss uses ranks/ordering only, never an absolute
teacher threshold or pseudo frame class.

Negative videos retain the ordinary bag loss and contribute no forced temporal
ordering. This prevents fabricated foreground inside a genuinely negative bag.

### Module 3 — Zero-initialised Semantic Residual Adapter (ZSRA)

The MACIL audio-visual snippet representation remains the main path. A small
text-conditioned residual adapter consumes the aligned frozen sentence
embedding and predicts a bounded residual to the MACIL AV logit. Its output
projection and reliability gate are zero-initialised, so the initial model is
exactly MACIL-SD AV rather than a newly random tri-modal fusion network. The
gate may add or withhold a residual but cannot replace the AV score.

## Objective

The training objective is

`L = L_MACIL + lambda_rank * L_RIRD + lambda_res * L_residual_regularisation`.

`L_MACIL` retains the published video-level MIL classification, modality-aware
contrastive terms, and self-distillation. RIRD is the only teacher-driven loss.
The residual regulariser penalises unnecessary departures from the MACIL logit,
especially where text coverage is absent.

## Required ablations

1. MACIL-SD AV reproduced in the same runner.
2. MACIL + ZSRA, without MSIT/RIRD.
3. MACIL + absolute teacher-score regression (expected weaker; distinguishes
   relative distillation from generic pseudo-labelling).
4. MACIL + RIRD, without ZSRA.
5. Full RIFT-MACIL.
6. Full model with teacher order shuffled within each video.

The full method must beat (2), (3), and (6) on validation Frame AP; otherwise
the claimed mechanism is unsupported even if the final score is high.

## Selection and final evaluation

- Official train/validation/test IDs remain frozen.
- At most six validation configurations per corpus, including the baseline.
- Selection criterion: validation pooled Frame AP; ROC-AUC breaks exact ties.
- Seeds: 234, 2025, 3407 for the final reported test aggregate. Test evaluation
  is also permitted for intermediate candidates, provided neither test labels
  nor test metrics influence training, tuning, architecture, checkpoint
  selection, or candidate retention.
- Report mean and sample standard deviation for Frame AP and Frame ROC-AUC.
- Also report within-hateful-video ROC as a diagnostic; it is not substituted
  for the LELA/MultiHateLoc metric pair.
- A claimed gain must have positive paired seed delta in all three seeds.

## Novelty collision boundary

The contribution is **not** generic audio-video-text fusion (CMHKF and
MultiHateLoc), direct per-frame LLM prompting or composition matching (LELA),
adaptive top-K mining, target/attack prompt decomposition, or ordinary
knowledge distillation. The claim is the combination of privileged masked
semantic isolation with confidence-gated **within-video ordinal distillation**
into a zero-initialised residual extension of a weakly-supervised AV locator.

Before a novelty score is accepted, an external reviewer must see this claim,
the closest-work matrix, the actual implementation, and the ablation results.
The required novelty score is at least 6/10; a proposal-only score does not
satisfy the final gate.
