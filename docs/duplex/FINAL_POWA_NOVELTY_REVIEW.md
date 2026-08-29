# Final independent novelty review: POWA-MACIL

## Verdict

Implemented-method novelty: **6.1/10 — unconditional PASS** against the
required threshold of 6. The score concerns implemented novelty, not
statistical significance or readiness for an untouched confirmatory claim.

## Novel contribution boundary

Sparse privileged/MLLM semantic distillation, entropic optimal transport,
fuzzy/neuro-symbolic logic, positive-only residual fusion, and MACIL-SD AV are
not individually new. The reviewer found medium, identifiable novelty in their
task-specific composition:

`typed moderation primitives → asynchronous predicate-target transport →`
`executable corpus-policy dense MIL`

The closest collision is MACIL-SD AV combined with privileged semantic KD,
temporal OT matching, and logical policy MIL. No inspected closest work contains
the complete composition for weakly supervised dense-frame hateful-video
localization.

## Implementation checks

- PEF has six real primitive heads and sparse train-only teacher BCE.
- AWB uses Sinkhorn transport with exact structural zeros outside the lag and
  padding support. A strict regression assertion passes.
- PCW-MIL recursively executes `AND`/`OR`/`NOT`; the post-preregistered
  positive-evidence hazard readout is explicitly part of Module 3.
- Corrected grounded ablations: full .5401 mean validation Frame AP, above
  teacher permutation .5269, same-time .5177, anonymous .5161, flat .5151,
  pointwise .5143, and policy permutation .5117.
- ZH residual ablation at seed 234: positive evidence .4952, typed-only .4873,
  signed residual .4714 validation Frame AP.
- The final summary recomputes from corrected score artifacts, requires exact
  frozen-GT coverage and lengths, and archives score/GT hashes.

## Claim boundary required by the reviewer

The final system is a validation-selected method family with corpus-specific
training regimes. Intermediate test metrics were visible, the exploratory
validation budget exceeded the preregistered cap, and test is not untouched
confirmation. HateClipSeg's single-model lead over VERA is only .000173 AP and
.001664 ROC. Claims may say "new best in the current reproduced table," but not
statistically significant, universally external SOTA, or a decisive HCS gain.

Non-blocking limitations are weak aggregate semantic-channel identifiability,
one-seed ZH residual ablation, and no reported finite-iteration Sinkhorn
marginal-residual diagnostic.
