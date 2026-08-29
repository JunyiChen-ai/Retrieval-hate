# POWA-MACIL preregistration — candidate v2

This replaces RIFT-MACIL as the active candidate after an external novelty
review scored RIFT 5.3/10. RIFT remains archived and must not be silently
relabelled as POWA.

## Observation and task-specific hypothesis

The four benchmarks do not implement one universal positive concept:

- HateMM: hate versus non-hate;
- MultiHateClip: the frozen positive class is Hateful OR Offensive;
- HateClipSeg: the frozen positive class is the union of Hateful, Insulting,
  Sexual, Violence, and Self-Harm versus Normal.

Existing WS-VAD ports collapse each policy into a single anonymous anomaly
logit. This makes the learned instance selector chase whichever correlated cue
best separates videos, explaining strong pooled scores and weak within-video
grounding. The hypothesis is that frame localisation improves when the model
must produce a *policy-valid semantic witness* rather than an untyped anomaly.

## Inputs and supervision

Human supervision remains parent-video binary labels only. The moderation
policy is supplied as a fixed natural-language class definition, not as a frame
annotation. A frozen MLLM may provide train-only primitive evidence estimates;
these are privileged machine supervision and are disclosed as such. At test,
the learned student consumes the MACIL I3D/VGGish streams plus aligned frozen
sentence embeddings and emits one score per frame/snippet.

I3D/VGGish/BERT extraction, ASR, padding, and temporal resampling are
preprocessing and are not counted as modules.

## Method modules

### Module 1 — Policy-Conditioned Evidence Factorizer (PEF)

PEF replaces the anonymous one-logit MACIL head with a small shared bank of
typed primitive heads. The fixed vocabulary contains at least: hostile/abusive
predicate, protected-group target, violence, sexual content, self-harm, and
benign/reporting context. A frozen MLLM supplies train-only soft primitive
evidence on ASR-aligned units. The student learns these primitives jointly with
the original binary bag objective; no primitive human labels are used.

Unlike ordinary semantic KD, the teacher does not supply the final hate score.
Its primitives are intentionally insufficient until composed under the policy.

### Module 2 — Asynchronous Witness Binder (AWB)

Targeted hate is represented as a relation, not a class token. AWB constructs a
lag-bounded bipartite binding between hostile-predicate evidence and
protected-target evidence. A differentiable transport/soft-matching operator
allows the target and attack to occur in different modalities and nearby time
units. The bound mass is returned to the participating timestamps to produce a
targeted-hate witness curve. Reporting/quotation context subtracts or gates a
witness but cannot independently create a negative prediction.

Direct harms (e.g. violence or self-harm) bypass relational binding and remain
typed unary witnesses. This distinction prevents the method from reducing to
`target_score * hostility_score` or generic cross-attention.

### Module 3 — Policy-Compiled Witness MIL (PCW-MIL)

The benchmark's published label definition is compiled into a differentiable
OR over admissible witness types. For HateMM the positive witness is targeted
hate. For MHC it is targeted hate OR untargeted offensive abuse. For
HateClipSeg it additionally admits the released offensive categories. The
compiled frame score is the submitted localisation score; top-K pooling of that
same score supplies the video-level MIL loss.

All corpora share parameters, primitive meanings, and the compiler. Only the
published policy expression changes. There is no dataset-specific learned
output head or test-time threshold.

## What is claimed as new

POWA does not claim novelty for MLLM pseudo labels, multi-modal fusion,
cross-attention, target prompting, or logical conjunction separately. The claim
is **policy-conditioned temporal witness construction**: factor weak video
labels into reusable semantic primitives, create relational hate witnesses by
cross-time predicate-target binding, then compile the dataset's moderation
policy into the exact dense score trained by MIL.

This differs from:

- LELA: direct training-free per-frame LLM scoring plus caption composition;
- MultiHateLoc: modality-aware fusion and top-K selection over one hate logit;
- CMHKF: generic heterogeneous knowledge alignment;
- privileged WS-VAD KD: teacher transfers final anomaly knowledge;
- confidence-aware/ranking KD: transfers pseudo-label confidence or order;
- temporal-logic action localisation: no moderation-policy witness ontology or
  relational predicate-target construction.

## Mandatory ablations and falsification

1. MACIL-SD AV in the identical runner.
2. PEF with a flat learned fusion head (tests typed features without policy).
3. PEF + pointwise target×hostility product (tests against naive conjunction).
4. PEF + same-time binding only (tests whether asynchrony matters).
5. PEF + AWB + a learned anonymous output head (tests the compiler).
6. Full POWA-MACIL.
7. Full model with policy expressions permuted across corpora.
8. Full model with primitive teacher channels permuted.

The full method must beat (2), (3), (7), and (8) on validation Frame AP. AWB is
supported only if (6) beats (4), especially on videos where target and predicate
timestamps do not overlap. PCW-MIL is supported only if (6) beats (5).

## Evaluation and gates

Evaluation, splits, SOTA bars, six-configuration validation budget, seeds, and
test discipline are inherited verbatim from `PREREG_RIFT_MACIL.md`. Primary is
pooled Frame AP; secondary is pooled Frame ROC-AUC. Test outputs are dense
scores, never temporal intervals.

An external reviewer must score implemented-method novelty at least 6/10 after
seeing code, closest-work analysis, and ablations. A proposal-only review is a
screening gate, not final acceptance.

## Post-preregistered amendment

This section records deviations discovered during implementation; it is not
backdated preregistration. The final implementation is a method family with
validation-selected corpus regimes rather than a single parameter-shared model:
corpus-specific fine-tuning for HateMM/MHC-EN, frozen-MACIL positive-evidence
integration for MHC-ZH, and joint training with a 48-second AWB window for
HateClipSeg. The positive-evidence readout belongs to PCW-MIL and adds
`-log(1-witness)` to the MACIL logit through a learned non-negative gate.

Intermediate test metrics were visible during development, although no test
frame label entered gradient training. The test is therefore a benchmark
development set, not untouched confirmatory evidence. The exploratory phase
also exceeded the original six-configuration validation budget. All final
results and structural ablations were rerun after fixing AWB to preserve exact
zeros outside its allowed lag/padding support.
