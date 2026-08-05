# C04 Independent Design Review — Archive Record

**Candidate:** `C04 Source–Proposition–Stance–Harm Tensor`  
**Date:** 2026-07-29 (Pacific/Auckland)  
**Verdict:** `REVISE_USER_AMENDMENT_REQUIRED`  
**Severity:** `2 Critical / 5 High / 4 Important`  
**Execution authority:** none

This is the local archivist's faithful persistence of the independent reviewer
message. The reviewer remained read-only under the parent task and did not write
this file.

## Reviewed frozen files

The reviewer exact-matched all five submitted SHA256 values:

| File | SHA256 |
|---|---|
| `refine-logs/C04_PROBLEM_ANCHOR.md` | `3625387a84aa1e94ccaf1132f9d09f5ddfb0ec48d532ffa150e6316f38190281` |
| `refine-logs/C04_STAGE0_ASSET_AUDIT.md` | `4e054569d6de19ed4a510212848164e4fbf43563128474fb15fee4a14d392524` |
| `refine-logs/C04_REFINED_PROPOSAL.md` | `bab9ee063515880e140e2c76b1b2769e6296f3eaced40ed51449396a00668f16` |
| `refine-logs/C04_EXPERIMENT_PLAN.md` | `03fefa44a3cc541abfc4c85d4ff47475459bbf01bd7ae1d5e366d91200e71c0f` |
| `refine-logs/C04_EXPERIMENT_TRACKER.md` | `9628dc8c2bdca3e8119bbf97a8687c8b5a71a73c6fa64d40f14667cf7c54c033` |

## What passed

The reviewer accepted the following framework:

- only `parent_video_binary_label` is gold;
- a train-only label-blind teacher cache is sealed before labels enter;
- no dev/test teacher, test-time MLLM, router, score fusion or second index;
- native-only student inference with the ordinary top-20 kNN endpoint;
- the two-dataset, seed-0 and paired seeds `0/1/2` statistical skeleton;
- REMOVE, SHUFFLE and NOISE are required rather than optional;
- the error evidence motivates role binding but is not itself used as a selector.

The verdict is therefore not a KILL of the abstract C04 hypothesis.

## Critical findings

### C1 — the existing proxy cannot PASS or scientifically KILL SPaSH

The proposed existing-bank proxy changes the scientific target:

- P8's summary prompt explicitly preserves **who is targeted** and **what could be
  hateful/offensive**, rather than extracting the neutral bounded proposition in
  C04.
- K4 supplies only a `0..3` evidence-density score. It has no protected-target
  binding or harm act.
- S/T are new bilingual lexical cue states, whereas the final method uses
  teacher-extracted proposition source and presenter commitment.

Consequently DIRECT+STUDENT over P8+K4+cue tensors can establish only that
`summary + density + cue` interactions help or fail. A positive result cannot
PASS a representation-matched SPaSH Stage-0, and a negative result cannot
scientifically KILL the stronger four-factor teacher. Under the current rule
that Stage-0 must pass before any teacher/GPU spend, C04 has no executable
evidence path.

Required disposition: downgrade the existing proxy to a nonbinding diagnostic.
Only the user may authorize a bounded train-only local-teacher pre-gate before
the current Stage-0.

### C2 — reliability and fallback do not yet meet the hard contract

The standing supervision contract requires every MLLM pseudo-signal to have
explicit reliability/confidence and a deterministic missing-signal fallback.
The draft provides `uncertain`/missing states and says two-prompt agreement is
diagnostic, but it does not define the reliability variable, conflict handling,
fallback tensor, coverage report or corruption sensitivity used by the method.

Required revision:

- freeze per-slot reliability states such as
  `stable / single_valid / conflict / missing`;
- define how two prompt forms produce one canonical slot;
- map conflict/missing deterministically to an explicit fallback without
  dropping, selecting or reweighting samples;
- report per-slot and joint coverage, reliability distribution, fallback rate
  and corruption sensitivity;
- state whether reliability enters the representation. If it does, it requires
  its own REMOVE/SHUFFLE/NOISE controls.

## High findings

1. **Insufficient composition controls.** Add a dimension/parameter/compute-
   matched `CONCAT_ALL4_MLP`, a retained independent-four-target control that is
   stronger than historical P4, slotwise shuffle, and role permutation.
   Specify exact `q4`, `q<=3` and capacity matching rather than only naming them.
2. **P4/KD collision not closed.** A P4 control whose auxiliary heads are
   discarded at inference is a weak strawman against a retained tensor branch.
   The revision must compare against a retained independent-factor student and
   address LEAF/C5-style structured or label-free knowledge distillation.
3. **DIRECT/STUDENT arenas are ambiguous.** With dev teacher forbidden, DIRECT
   must be train-only OOF. STUDENT must be measured train-OOF and native-only
   dev. Remove the draft's `80% retention` comparison across unlike arenas.
4. **Novelty review is incomplete.** Add RAMF/TMLR 2026, LEAF/ACL Findings 2026,
   classical TFN/LMF tensor fusion and DR-HM/Intent-style decomposition. Novelty
   cannot rest on the exact conjunction being absent; the paper-level delta must
   survive component and mechanism comparisons.
5. **Fold and seed nesting is not frozen.** Define outer train folds, inner
   hyperparameter folds, OOF tensor targets, native dev evaluation, and paired
   adaptation/head seeds. If adaptation is stochastic, seeds `0/1/2` must each
   cover the full student/adaptation plus downstream-head lineage.

## Important findings

1. Define which prompt form is primary, how semantic disagreement is measured,
   and how conflicts become canonical outputs.
2. Define epsilon-safe normalization, the all-fallback/zero-`q4` result, the
   signed-orthogonal-map generation algorithm and exact map hashes.
3. Resolve the proposed 200-video sample's representativeness and the apparent
   `<=2 GPU-hour` pre-gate versus `4–8 GPU-hour` full-cache estimate.
4. Freeze exact tuple shuffle, slotwise shuffle, role permutation, covariance-
   matched noise and remove-factor replacement semantics.

## Minimal user contract amendment

The smallest defensible extension is **C04-only** and does not waive any final
metric, mechanism, novelty or test rule:

1. Permit a local-open-weight, train-only, label-blind C04 teacher to instantiate
   the *matched* S/P/T/H signal before the current full-bank Stage-0.
2. First tranche: exactly 200 train IDs per dataset, chosen before label access by
   ascending SHA256 of
   `C04-A0T-SMALL-v1 || dataset || video_id || 20260729`; HateMM and MHC-ZH
   remain separate. Two fixed prompt forms, eight fixed frames and capped native
   transcript are allowed. No dev/test content, labels, neighbors, predictions,
   errors, API, OCR or cross-dataset data may enter.
3. The small cache is sealed before train labels enter. It may adjudicate only
   reliability, conditional-information, calibration and permutation gates; it
   is not the full-bank `+0.050/+0.050` Stage-0 and cannot produce a performance
   claim.
4. Hard resource envelope for the first tranche: one GPU at a time,
   `8 CPU / 64 GB`, aggregate cap `2 GPU-hours` across both datasets. All work
   must use reviewed project-local SLURM wrappers under `HateVideo`, no `--time`,
   and must wait for `JobHeldUser` auto-release. If the estimate or execution
   would exceed the cap, halt and return to the user.
5. Only after a small-gate PASS and fresh independent result-to-claim GO may the
   workflow request/consume the remainder of a full train bank. The total
   conditional envelope, including the first tranche, is one GPU at a time and
   at most `8 GPU-hours` across both datasets. It completes exactly the 744
   HateMM train and 579 MHC-ZH train IDs; still no dev/test teacher.
6. The matched full-bank gate is then train-only DIRECT OOF plus native-only
   STUDENT OOF/dev through the actual fold/deployed-head path. The original
   `+0.050` accuracy and macro-F1 two-dataset threshold remains binding.

This amendment is a decision request, not authorization. Even if the user
approves it, the proposal must first be revised for C2 and all High findings,
receive a fresh independent design GO, and pass code/resource review before any
teacher generation or SLURM submission.

## Final boundary

`REVISE_USER_AMENDMENT_REQUIRED`

- Do not implement the cue proxy as a promotable Stage-0.
- Do not generate a teacher bank.
- Do not run Python, tests, GPU work or SLURM.
- Do not access test.
- Keep C04 alive as an abstract candidate pending the user's explicit Stage-0
  contract decision.

