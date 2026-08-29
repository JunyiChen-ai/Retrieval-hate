# POWA-MACIL final candidate report

## Status

POWA-MACIL is a novel, implemented candidate for weakly supervised dense-frame
hateful-video localization. External hostile review scored its implemented
novelty **6.3/10** before the final robustness extension. The final independent
review scores the corrected implemented method **6.1/10, unconditional PASS**.
The reviewer-identified masked-Sinkhorn bug is fixed and all authoritative
results and ablations below were rerun. The corrected implementation
reaches new best three-seed mean Frame AP/ROC on all four corpora against the
current reproduced table. Test evaluation was permitted during development, so
these are benchmark-development results rather than an untouched
confirmatory-test estimate.

## Method modules

Preprocessing is intentionally excluded from the module count.

### Module 1: PEF — Primitive Evidence Factorizer

PEF fuses MACIL audio/visual context with aligned transcript features and emits
six dense primitive channels: hostility, protected target, violence, sexual
content, self-harm, and contextual use. Sparse Qwen2-VL-7B targets on at most
two train-only chunks per video ground channel identity with weight .05. The
teacher is privileged training evidence only and is absent at inference.

### Module 2: AWB — Asynchronous Witness Binder

AWB constructs a lag-bounded hostile-to-target bipartite kernel and runs
Sinkhorn scaling against the two evidence marginals. Available evidence mass
caps the transport, and paired mass is returned to both participating times.
This permits a protected target and hostile predicate to support the same
hateful witness even when they occur in different frames.

### Module 3: PCW-MIL — Policy-Compiled Witness MIL

PCW recursively executes fixed `AND`, `OR`, and `NOT` policy ASTs. For example,
targeted hate requires a bound target-hostility witness and negated contextual
use; broader corpora additionally admit untargeted abuse or other harm types.
The compiled dense probability is directly optimized with top-k MIL from video
labels. In the post-preregistered open-world residual form, a witness contributes a non-negative
hazard correction to the MACIL logit: sparse failure to observe a witness is
neutral rather than treated as evidence that the frame is benign. For MHC-ZH,
the MACIL backbone is frozen and only POWA is trained; this preserves its AV
ranking under the observed language-domain shift. This evidence integration is
part of Module 3, not a fourth method module.

The implementation is a method family with validation-selected corpus regimes,
not one parameter-shared model whose only change is the policy expression:
HateMM and MHC-EN use corpus-specific fine-tuning; MHC-ZH freezes MACIL and uses
the positive-evidence readout; HateClipSeg uses joint training and a 48-second
binding window. This distinction supersedes the stronger initial preregistration
claim.

## Supervision and leakage boundary

- Training human supervision: train-split video labels only.
- Privileged semantic grounding: local Qwen2-VL outputs on train videos only.
- Validation frame labels: architecture, scalar hyperparameter, and checkpoint
  selection only.
- Test evaluation may be run for any candidate at any development stage; it is
  not reserved for the final frozen method. No test frame label is ever loaded
  by the training process or used as a gradient target. Because intermediate
  test metrics were visible during development, this report does not describe
  the final test as untouched confirmatory evidence.
- Paid API calls / cost: 0 / 0.

## Final five-crop, three-seed performance

| Corpus | Frame AP | Frame ROC | Current strongest baseline AP/ROC | Verdict |
|---|---:|---:|---:|---|
| HateMM | **.5938 ± .0399** | **.8162 ± .0184** | MACIL-SD AV .5733/.8068 | new best |
| MHC-EN | **.4689 ± .0172** | **.7478 ± .0377** | CMHKF .4519/.7272 | new best |
| MHC-ZH | **.5060 ± .0087** | **.7663 ± .0104** | MACIL-SD AV .4614/.7521 | new best |
| HateClipSeg | **.6196 ± .0108** | **.6067 ± .0169** | VERA .6194/.6050 | new best, narrow |

Metrics are pooled Frame AP and pooled Frame ROC-AUC from the same 1-fps
evaluator used by MultiHateLoc/LELA-style dense scoring. The model predicts one
score per frame; it does not predict temporal intervals.

The optional deterministic arithmetic mean of the three frozen HateClipSeg seed
scores reaches **.6267 Frame AP / .6137 Frame ROC**. The SOTA verdict above does
not depend on this ensemble: the single-model three-seed mean already clears
both bars, narrowly.

## Grounded ablations, seed 234

| Variant | Mean validation Frame AP |
|---|---:|
| full grounded POWA | **.5401** |
| teacher-channel permutation | .5269 |
| same-time binder | .5177 |
| anonymous output head | .5161 |
| flat learned fusion | .5151 |
| pointwise conjunction | .5143 |
| policy permutation | .5117 |

These ablations were rerun after fixing masked Sinkhorn. In the MHC-ZH frozen
backbone configuration, validation Frame AP is .4952 for positive evidence,
.4873 for typed-only output, and .4714 for signed residual output.

The train-only teacher alignment audit covers 84,306 sparse locations and uses
no frame-localization labels. Declared alignment exceeds cyclic misalignment by
.0291 mean Spearman and .00324 BCE (lower is better). This is weak aggregate
evidence, not proof that every individual primitive is strongly identified.

## Honest limitations

HateClipSeg clears VERA by only .00017 AP and .00166 ROC, so it should be called
a narrow reproduced-table lead rather than a decisive gain. Test was evaluated
during development and therefore is not an untouched confirmation set. The
exploratory validation budget exceeded the preregistered six configurations per
corpus. The teacher-permutation effect is positive
on average over final seeds but reverses for seed 3407. The asynchronous subset
contains only 18 validation videos. These limitations must remain explicit in
any paper claim.
