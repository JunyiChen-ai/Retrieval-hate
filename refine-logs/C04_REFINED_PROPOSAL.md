# C04 Refined Proposal — Source–Proposition–Stance–Harm Tensor

**Short name:** `SPaSH-Tensor`  
**Status:** `FROZEN / PENDING INDEPENDENT DESIGN REVIEW`  
**Date:** 2026-07-29 (Pacific/Auckland)

The immutable anchor is `refine-logs/C04_PROBLEM_ANCHOR.md`.

## Method thesis

A hateful proposition is not equivalent to the presenter's commitment to that
proposition. C04 represents a video by the *joint binding* of source,
proposition, presenter stance and protected-target harm act, and transfers only
that joint interaction into the native video-memory embedding. It does not add a
factor verdict to the final decision.

## Four noisy factors

For video `v`, the train-only teacher emits one strict JSON object:

1. **Source `S_v`:** the relation of the proposition's origin to the current
   presenter: `current_presenter`, `quoted_or_embedded_source`,
   `performed_or_lyric_source`, `mixed`, or `uncertain`. This is not the
   protected target and need not name a person.
2. **Proposition `P_v`:** one bounded, neutral clause describing the proposition
   being presented. It contains no verdict, rationale or moderation decision.
3. **Stance `T_v`:** the current presenter's commitment relation to that
   proposition: `endorse_or_promote`, `reject_or_counter`,
   `report_or_describe`, `perform_without_clear_commitment`, or `uncertain`.
4. **Harm `H_v`:** a bounded relation with protected-target binding and act in
   `{attack, dehumanize, threaten, exclude, harass, other, none, uncertain}`.
   `none` means the extracted proposition lacks such an act; it is not the
   parent video label.

Every slot has an explicit `uncertain`/missing state. No confidence threshold
drops a sample. The prompt never includes the binary label, predictions,
neighbors, split statistics, error status, target taxonomy from gold, or words
requesting a final hate/non-hate verdict.

## Dense nonseparable tensor

Let a frozen text encoder map each canonical slot rendering to a normalized
vector. Fixed role-specific signed orthogonal projections `R_S,R_P,R_T,R_H`
share output dimension `r=256` and are generated once from the frozen seed
`20260729`.

```text
u_S = normalize(R_S E(S_v))
u_P = normalize(R_P E(P_v))
u_T = normalize(R_T E(T_v))
u_H = normalize(R_H E(H_v))

q4(v) = normalize(u_S ⊙ u_P ⊙ u_T ⊙ u_H)
```

Role-specific maps make the operator ordered rather than a commutative bag of
fields. `q4` becomes zero only through the explicit missing-state encoding, not
through sample deletion. No train label signs, rotates or selects `q4`.

For controls, `q<=3` contains every frozen one-, two- and three-way interaction
projected to the same final dimension and parameter count; `q_add` contains the
capacity-matched additive marginals. Thus C04 cannot attribute a win to fields
unless `q4` beats both.

## Native tensor student

The student retains the normal RGCL full-video representation `z_v` and adds one
capacity-matched tensor branch `g_theta(z_v)`. During training only:

```text
L = L_RGCL + lambda_tensor * (1 - cosine(g_theta(z_v), stopgrad(q4(v))))
z'_v = normalize(concat(z_v, beta * normalize(g_theta(z_v))))
```

`z'_v` is the sole memory/query embedding. At dev/test, `g_theta` reads only the
native video representation; teacher files and `q4` are unavailable. Final
classification is ordinary top-20 kNN over train `z'`, with the existing
evaluator and no router, verifier, score fusion or second index.

`lambda_tensor` and `beta` are frozen from a train-only inner fold in Stage-1
and then fixed for all seeds. The grid, tie rule and compute match must be
pre-registered before Stage-1; no dev/test result selects them.

## Identifiability

The four-way mechanism is identifiable only if all conditions hold:

1. FULL beats a capacity-matched REMOVE branch, tuple SHUFFLE and moment-matched
   NOISE.
2. FULL beats ADDITIVE and the complete LOWER-ORDER (`<=3`) interaction control.
3. Removing each of S/P/T/H reduces the same-direction gain; an individual
   factor cannot carry the result alone.
4. A direct scalar stance/harm control and the historical archive-field
   auxiliary control do not match FULL.
5. The gain appears in the ordinary kNN output, not only in tensor prediction
   cosine or an auxiliary classifier.

If FULL only beats REMOVE but not LOWER-ORDER, the supported statement is
generic structured distillation, not C04. If stance alone wins, the result is a
three-class/scalar policy readout and C04 fails its dedup boundary.

## Leakage and split protocol

- Stage-0 consumes only existing HateMM/MHC-ZH train/dev banks specified in
  `C04_STAGE0_ASSET_AUDIT.md`; outer labels in JSON/PT containers are unread by
  the factor reader.
- Any new Stage-1 teacher reads only each dataset's train video, transcript and
  fixed policy/schema. The cache is sealed by ID allowlist, source/config/prompt/
  model hashes and an access ledger before train labels enter.
- Parent train labels enter only the ordinary RGCL loss, downstream fold
  evaluator and stratified reports after sealing.
- Dev receives native input only; its labels are evaluation-only. No dev factor
  cache, teacher call or factor-quality selection exists.
- Test content and labels are forbidden until a frozen final three-seed lineage.

## Strong controls

Primary:

- `FULL_Q4`
- `REMOVE_TENSOR_CAPACITY_MATCHED`
- `SHUFFLE_Q4_TUPLES` using a frozen label-blind dataset/ID permutation
- `NOISE_Q4_MATCHED` for dimension, norm, covariance, coverage and missingness

Mechanism:

- `ADDITIVE_S_P_T_H`
- `LOWER_ORDER_LE3`
- `REMOVE_S`, `REMOVE_P`, `REMOVE_T`, `REMOVE_H`
- `STANCE_ONLY`, `HARM_ONLY`, `P4_STYLE_INDEPENDENT_AUX`
- `CAPACITY_ONLY_NATIVE_BRANCH`

All arms use the same native input, branch width, optimizer steps, batches,
checkpoint rule, evaluator and compute accounting.

## Non-isomorphism and novelty boundary

C04 is not:

- P4, which predicts independent archive fields with auxiliary heads discarded
  at evaluation;
- SSR, which assigns MLLM relation types to selected hard pairs and changes
  pairwise margins;
- LB-SCGP Global-R2, which compiles coarse full-bank observables into a global
  Gram target and certificate;
- C3-nontarget, which appends an unstructured reasoning-text embedding;
- archive target/mechanism lookup, scalar three-class reweighting, a test-time
  reasoning chain or a per-item stance veto.

The narrow novelty candidate is:

> train-only label-blind four-role binding, represented by a nonseparable dense
> tensor and internalized into the single teacher-free embedding consumed by a
> hateful-video kNN memory classifier.

This is not yet a novelty claim. Prior art already covers:

- quotation identification plus stance filtering for text hate detection
  (`Hypothesis Engineering for Zero-Shot Hate Speech Detection`, 2022);
- target disentanglement for hateful memes (DisMultiHate, ACM MM 2021);
- context-aware hate/counter-speech classification (NAACL 2022);
- objective/hate-assumed/non-hate-assumed reasoning for hateful video (MARS,
  arXiv:2601.15115; and closely related reasoning-aware fusion work);
- literal/pragmatic representation decomposition (Intent Projection,
  arXiv:2606.03604).

Therefore the paper may not claim first quotation/stance reasoning, first
factorized hate representation, first structured MLLM supervision or first
pragmatic projection. Novelty survives only if the exact four-way
privileged-tensor-to-memory mechanism is absent from a final literature review
and LOWER-ORDER/P4 controls empirically fail to explain the result.

## Current feasibility boundary

No explicit matched four-factor bank exists. The existing-bank Stage-0 in
`C04_STAGE0_ASSET_AUDIT.md` is the only legal pre-teacher route currently
defined. Independent review must decide whether it is:

- `GO_STAGE0`: a valid conservative reachability gate;
- `REVISE_PROXY_CANNOT_PASS`: useful only to kill, requiring explicit user
  amendment before a teacher-instantiated Stage-0; or
- `KILL_COLLISION_OR_INFEASIBILITY`.

No implementation or execution is authorized by this proposal.

