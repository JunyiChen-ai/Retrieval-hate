# Frozen pilot plan — POWA context-quotient span marginal

Date: 2026-08-31. This plan is frozen before formal training. Formal Stage V
requires an independent implementation/evaluation PASS.

## Hypothesis and sole mechanism

POWA's absolute score carries pooled/video evidence, but top-k MIL does not say
that hateful evidence must persist as a temporal span. A trainable residual is
therefore attached to frozen corpus-specific POWA. Every input channel to the
residual is centered within its video, and its output is also exactly zero-mean,
which gives exact invariance to additive video-constant offsets in the supplied
frozen channels. This does not claim to remove every nonlinear video-identity
signal. Positive bags optimize a
normalized marginal over candidate contiguous spans; negative bags provide
dense benign-flattening supervision. At inference the centered residual is added
to the frozen POWA logit.

The residual is defined on POWA's fixed 200-bin whole-video coordinate system.
Training already applies upstream's deterministic uniform sampling/padding to
that grid. At inference, the same transform is applied solely for the residual
head; its output is linearly interpolated back to the native snippet grid and
re-centered there before it is added to the untouched dense POWA anchor. Thus
local kernels and span lengths have the same relative-video scale in training
and inference, while POWA's original dense scores remain exact.

For local logits `l_t = center(logit(POWA_t)) + r_t`, candidate span score is
the mean local logit over
each valid span with duration in `{3,5,9,17,33}` POWA model-grid rows (duration
is clipped to `T`, duplicate durations removed). These lengths are relative
whole-video bins, not literal seconds. Using the nonconstant centered anchor in
the loss gives every matched arm, including singleton, a nonzero learning signal
at the exact-identity initialization. The positive bag logit is
`0.5 * (logsumexp(span_score/0.5) - log(number_of_spans))`. Negative bags use
dense `BCEWithLogits(l_t,0)` and the same bag BCE; because `l` is centered,
this suppresses spurious local ordering rather than claiming an attainable
all-negative absolute logit. A fixed residual L2 weight
`.01` applies to both classes. No train/val/test span label enters training.

## Stage V

Corpora: HateMM and HateClipSeg, independently trained, seed 234, five epochs.
Anchor checkpoints are the same corpus-specific POWA checkpoints used in C2.
Validation checkpoint feasibility requires candidate pooled AP and ROC each at
least POWA minus `.002`; among feasible epochs choose maximum within ROC, then
AP, then earlier epoch.

Matched arms:

1. `span_marginal` core: contiguous variable-duration spans.
2. `singleton`: only duration 1, testing whether ordinary instance marginal is
   sufficient.
3. `shuffled_span`: centered local-logit timestamps are deterministically permuted before
   span enumeration, preserving values and budget but destroying contiguity.
   In code this permutes the complete centered local-logit timestamps, with a
   stable `sha256(seed|epoch|video_id|crop)` seed, so POWA ordering cannot leak
   through the control and every permutation is reproducible.

The core must on both corpora: gain at least `.020` within over POWA; improve at
least 55% of eligible videos; beat each matched control by `.010`; retain
pooled feasibility; and have exact zero-mean residual (`<=1e-6`). HCS high
positive-fraction within must gain at least `.015` and end above `.50`.

Persist evaluator-readable scores for POWA, candidate, residual-only,
singleton/shuffle arms, and fixed chronological/reverse/edge/center assignments
of the learned residual values. A fixed-position control may not explain more
than half the core gain. All metrics come from the shared evaluator.

Stage V failure means `KILL_BEFORE_TEST`; no loss/temperature/duration/epoch
retuning. Only full Stage-V PASS may authorize the runner's one-shot HMM/HCS
test inference; inference verifies the PASS artifact, selected checkpoint hash,
and source hashes before reading test data, then calls the shared evaluator.

## Novelty boundary

Segment-centric WTAL, dynamic programming, action/context separation,
positive-count latent variables, and VLP distillation all exist. Closest primary
work: AutoLoc (ECCV 2018), FTCL and ASM-Loc (CVPR 2022), BPS (ICCV 2023), and
hierarchical latent attention (ICCV 2023). The only potentially defensible claim
is the complete mechanism: a frozen policy-aware hate localizer supplies the
absolute axis, while an exactly additive-context-quotiented residual learns through
normalized variable-span marginal evidence. Component-level novelty is not
claimed; a full independent search is mandatory before formal execution.
