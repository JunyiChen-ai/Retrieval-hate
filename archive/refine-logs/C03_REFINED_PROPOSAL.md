# C03 Refined Proposal — Policy-Anchored Native MNTP

**Status:** `FROZEN / KILL_C03_DESIGN_INFEASIBILITY`  
**Date:** 2026-07-29 (Pacific/Auckland)

## Problem Anchor

- **Bottom-line problem:** improve RGCL/RA-HMD hateful-video detection by at least
  `+0.030` accuracy and macro-F1 on at least two datasets under the frozen paired
  protocol.
- **Must-solve bottleneck:** F92 shows readout/mask surgery cannot repair the
  bidirectional text geometry; F93 shows weight adaptation is a real direction but
  an off-weight-point transplant damages the image stream and fusion.
- **Non-goals:** no readout rerun, transplant, external corpus, prompt ensemble,
  teacher/API inference, per-item router, target/stance/span pseudo-gold, or larger
  model.
- **Constraints:** single-dataset train split only; parent-video binary label is the
  only gold; all mask, policy-view, loss-weight and holdout choices are label-blind;
  final inference uses the native video only.
- **Success condition:** FULL beats matched native MNTP-only and the causal baseline
  on two datasets while preserving stream diversity and fusion synergy.

## Prospective method thesis

The smallest defensible C03 method would train MNTP at the deployed task-LoRA weight
point while using a fixed moderation-policy context and a matched image-preservation
anchor, then discard all training-only policy context at inference.

### Prospective FULL, if the gate were available

1. Start from each dataset's deployed task-LoRA lineage and apply the already-frozen
   bidirectional decoder patch.
2. Use only that dataset's train split in the deployed multimodal format.
3. Prepend the same fixed policy block to every policy-present training view:

   > Hateful content attacks, dehumanizes, threatens, or promotes harm toward a
   > protected group. Mention, quotation, documentary reporting, counterspeech,
   > satire, archival reproduction, or lyrics are not violations without
   > endorsement or promotion. Interpret target, harmful proposition, source, and
   > stance together.

4. Policy presence is determined by an ID/epoch hash with fixed probability `0.5`;
   it never uses the video label. The other half is the native no-policy view.
5. Select `20%` of eligible native text positions using a counter-based hash over
   `{dataset, video_id, epoch, token_position}`. Vision, special, padding and policy
   tokens are never prediction targets. Selection and loss weights are label-blind.
6. Apply MNTP next-token CE only at masked native text positions. Add a
   label-blind image-preservation loss against the existing plain-bidirectional
   train image feature on a frozen ID-hash subset. No stance/target/rationale label
   or MLLM relation is generated.
7. Use a fixed two-pass budget and a deterministic outer-train hash holdout only for
   unlabeled MNTP-loss/memorization diagnostics. No dev transcript or dev label
   selects steps, masks, lambda, or checkpoint.

### Native inference

Inference contains no policy block, mask, teacher, pseudo-relation file, or alternate
view. It uses the frozen native bidirectional encoder on the original full video,
the deployed image-prefix/text-tail readouts, the same fusion head, train-memory
top-20 retrieval, and the same checkpoint/evaluator protocol.

## Claim map

- **C1:** native weight-point MNTP can repair text geometry without the image/fusion
  damage caused by the incompatible F93 transplant.
- **C2:** any improvement beyond generic MNTP is caused by semantic policy
  conditioning, not extra tokens, masking, compute, or a changed inference prompt.

Anti-claims: the proposal does not claim that generic MNTP is novel, that F93 is a
native proxy, that policy relations are gold, or that policy is needed at inference.

## Required controls

- `CAUSAL_MATCHED_COMPUTE`
- `NATIVE_MNTP_ONLY`: same native weight point, two-pass budget, masks and image
  anchor as FULL, but no policy block
- `FULL_POLICY_MNTP`
- `POLICY_REMOVE`
- `POLICY_SHUFFLE`: the fixed policy block's non-special tokens are permuted by a
  dataset/split/ID hash while length, token multiset, masks and compute stay matched
- `POLICY_NOISE`: matched length, token-position type, mask count, coverage and
  compute
- `REMOVE_IMAGE_ANCHOR`

FULL must beat `NATIVE_MNTP_ONLY`; otherwise only generic MNTP is supported. It must
also keep mean stream cosine below `0.55`, preserve the plain-bidirectional image
feature at mean cosine at least `0.98`, and keep fusion non-destructive.

`CAUSAL_MATCHED_COMPUTE` would retain causal attention and the deployed task-LoRA
weights while consuming the same number of data-loader/forward accounting slots
through frozen no-update passes; it may not receive MNTP or policy updates.

## Terminal feasibility finding

The prospective definition above constrains label leakage and native inference, but
it is not an execution preregistration. If the registry gate were ever reopened, the
exact compute-matching implementation and control schemas would still need formal
review. The fixed generic policy prefix provides policy conditioning; it does not by
itself identify a per-example policy-relation mechanism.

More importantly, no existing HateMM + MHC-ZH bank isolates even this
policy-conditioned native-MNTP object. Creating one requires prohibited pre-gate GPU
training/extraction.

Accordingly this proposal is retained only as a historical design. C03 is frozen at
`KILL_C03_DESIGN_INFEASIBILITY`; no implementation is authorized.
