# Independent pre-run review — POWA span-marginal pilot

Date: 2026-08-31  
Reviewer role: independent Rule-9 implementation/evaluation review  
Verdict: **PASS**

This is the formal re-review after the initial BLOCK. It authorizes the six
preregistered **Stage-V validation runs only**. It does not directly authorize
candidate test inference. Test remains fail-closed unless the current source
can reproduce an identical two-corpus Stage-V PASS from all six completed run
artifacts.

The review covered `PILOT_PLAN.md`, `README.md`, `train.py`, `infer.py`,
`complete_run.py`, `complete_infer.py`, `summarize_stage_v.py`,
`run_stage_v.sh`, `test_span_marginal.py`, `src/powa_residual.py`, all imported
POWA/MACIL-SD/data modules, and the shared evaluator. The verdict applies only
to the hashes recorded below. Formal training and candidate test inference
were not started during review.

## CRITICAL

No critical finding remains open.

### C1 resolved — all matched arms learn at exact identity initialization

The loss now operates on
`center(logit(POWA)) + residual`, rather than on a zero residual alone. The
nonconstant frozen anchor supplies a position-dependent gradient while the
trainable residual remains exactly zero-mean. The singleton arm is therefore a
trained ordinary-instance marginal control, not an identity placeholder.

An actual residual-head backward test at zero initialization gives a nonzero
singleton output-weight gradient. The recorded development smoke additionally
shows nonzero gradients in all 5/5 singleton optimizer batches, with mean
gradient norm `0.0336011`. The formal training loop aborts an epoch if the
residual head receives no gradient.

### C2 resolved — residual training and inference use the same coordinate system

The residual mechanism is consistently defined on the training-time 200-bin
whole-video grid. Validation/test apply the same deterministic uniform
sampling or padding to the residual branch, interpolate only the residual back
to the native snippet grid, and re-center it there. The untouched POWA anchor
is still computed and retained on its original dense snippet grid.

Real-data comparisons on long videos verified exact equality of the training
and inference-side coarse visual, audio and text tensors:

| corpus | example | native rows | residual rows | V/A/T exact |
|---|---|---:|---:|---|
| HateMM | `hate_video_104` | 622 | 200 | yes/yes/yes |
| HateClipSeg | `bit_0EHvMSiEHVoc` | 340 | 200 | yes/yes/yes |

The plan now correctly defines `{3,5,9,17,33}` as relative model-grid rows,
not literal seconds. This removes the prior local-kernel/span-scale mismatch.

### C3 resolved — Stage-V and test decisions are artifact-closed

The formal runner creates six corpus/arm runs with fixed anchors, seed and
hyperparameters. Each run receives a complete hashed source snapshot,
configuration, log/PID, checkpoint and validation scores when feasible, the
shared evaluator's `metrics.json`, and an atomic completion record.

The supervisor validates:

- corpus, arm, seed, method, canonical output directory and all frozen
  hyperparameters;
- exact config/metadata agreement;
- canonical anchor path plus current `model.pth` and anchor `train_meta.json`
  hashes;
- corpus-only training, no test use, and frozen POWA metadata;
- selected checkpoint/score/evaluator/completion hashes;
- every internal validation metric against the shared evaluator result;
- identical source inventories across all six runs and every current source
  hash against that inventory;
- every preregistered performance and control gate.

Authorized test inference accepts only the canonical Stage-V summary and
canonical test directory. Before reading test data it reruns the same
supervisor over all six current artifacts and requires byte-identical PASS
output. It then rechecks the selected core, anchor path/model/config hashes,
residual checkpoint hash and current source. Test is restricted to `split=test`;
an exclusive `test_claim.json` prevents concurrent, repeated or alternate-dir
exports. A failure after claiming remains fail-closed and requires explicit
audit rather than automatic retry.

## MAJOR

No major finding remains open.

### M1 resolved — supervision and context-quotient claim are aligned

Optimization reads only video labels for the requested corpus's train split.
No train, validation or test span annotation enters the loss. Validation GT is
used only for validation metrics and checkpoint selection; test data cannot be
read before Stage-V authorization.

Every supplied residual-head channel is centered over valid video rows, and
the output is re-centered after the temporal head. This gives the claimed
invariance to additive video-constant offsets in those supplied channels. The
plan explicitly does not claim removal of all nonlinear video identity.
Negative dense BCE is accurately described as benign/local-order flattening
under the centered axis, not as an attainable all-negative absolute residual.

### M2 resolved — controls are matched and auditable

All arms use the same corpus split, five crops, model initialization, loader
seed, optimizer, epoch budget, selection rule and validation branches.
`shuffled_span` permutes the complete centered local-logit timestamps before
the same variable-span enumeration, so frozen POWA order cannot leak through
the control. Its seed is stable in
`sha256(seed|epoch|video_id|crop)`, independent of batch position, and the rule
is written to training metadata. Singleton changes only the candidate duration
set to `{1}`.

The span marginal implements the frozen normalized log-mean-exp formula,
clips durations to valid length and removes duplicates. The core rewards a
synthetic contiguous pattern over a value-matched scattered pattern, while the
singleton marginal is permutation-invariant in that test.

### M3 resolved — validation, selection and evaluator use one protocol

Checkpoint feasibility constrains both pooled AP and pooled ROC relative to
epoch-0 POWA before maximizing validation within-video ROC, then AP, then the
earlier epoch. Each selected arm persists evaluator-readable POWA, candidate,
residual-only and four fixed-position branches. The runner evaluates all
branches through the repository's single shared evaluator with exact frozen
validation coverage.

The supervisor closes the cross-arm and two-corpus gates: pooled feasibility,
`.020` within gain, 55% improved-video ratio, singleton/shuffle margins,
zero-mean residual, fixed-position attribution, and both HateClipSeg
high-positive-fraction requirements. Any failure produces
`KILL_BEFORE_TEST`.

## Checks rerun on final source

- 6/6 mechanism tests passed: masked centering, zero initialization/output
  mean, span contiguity, deterministic shuffle, singleton optimizer signal,
  and coarse-to-native coordinate consistency.
- The development singleton smoke records five nonzero-gradient batches out
  of five. It is development evidence only and is not a formal result.
- Real train/inference coarse tensors matched exactly on long videos from both
  pilot corpora.
- Train/validation/test cohorts are pairwise disjoint: HateMM `744/109/215`;
  HateClipSeg `251/63/79`. Validation score coverage equals GT exactly at
  `109/109` and `63/63`.
- POWA is loaded from a corpus-specific checkpoint, frozen with
  `requires_grad=False`, held in eval mode, executed under `no_grad`, detached,
  and excluded from the residual optimizer.
- The shared evaluator self-test passed pooled ROC, pooled AP, within-video ROC
  and 1 fps grid checks. No metric implementation is copied into the
  experiment.
- Python compilation, `bash -n run_stage_v.sh`, and `git diff --check` passed.

## MINOR / non-blocking

- The six mechanism tests do not construct a complete synthetic six-run
  authority fixture. The authorization code was inspected directly and is
  additionally exercised by the live supervisor recomputation before test.
  Adding a compact tamper matrix for summary, anchor, source and repeated claim
  failures would improve regression coverage but does not change this review's
  scientific conclusion.
- `complete_infer.py` seals the shared evaluator's test score hash and all test
  artifacts but does not repeat the full internal-versus-evaluator value
  comparison used at Stage V. The frozen runner, source binding,
  `--require-full-coverage`, and common score hash make this non-blocking for
  the pre-run decision.

## Reviewed hashes

| source | SHA256 |
|---|---|
| `PILOT_PLAN.md` | `441373dda32559572e5c49e93c62416b22c5c132abfa0dec96861928442af625` |
| `README.md` | `ac238929af41144d918fde39f603f55711290ff186afc93e6cf15de4f791a274` |
| `train.py` | `8018e9ccd3a1904e1c143e2dd7ceac17046de64afd19970453eec6b583cecdf9` |
| `infer.py` | `c85ad6b9ed954af78b96b44492bef137884d65510b036b71c805bccbf9635449` |
| `complete_run.py` | `bc4605fbe608fff6133eeeb9b77ec8d12bfe339777039c8cbad7cbf8cd300980` |
| `complete_infer.py` | `d0d8ef8ec4c33c3883aeed8c58ce40e0d0d855deaaeff468bc9cdb43fe157719` |
| `summarize_stage_v.py` | `bfd4c9cad39e689ec5a58d031bf1b8f3fd34d66d564c36dadc590e2311d853f9` |
| `run_stage_v.sh` | `c726e77d19aa072de634c7daed77b0ce4a3a385535487c53663b524f6a7b5876` |
| `test_span_marginal.py` | `0d16041dce5d7ae5c262545a3b7b2d04d135eec6ba27bd368e80f288d7a83a15` |
| `src/powa_residual.py` | `ec0631bc86e6b9f51e10dd967d76525d054deca178b72378a64be9c497d208a1` |
| shared `eval_baseline_scores.py` | `2da04398c6e8bb66e275afe5664a06ec7d0f85d0f973ef5f67c01d0252093bdf` |
| shared `frame_eval_common.py` | `8b8c07d483af8ca53138a4a6144e9095781cdd563543308a55a4c2b6a03f801b` |

The formal snapshot additionally includes this PASS review and every imported
Python source under POWA, MACIL-SD and `hate_common`; the supervisor requires
all six inventories to be identical and equal to the current working tree.

## Disposition

The reviewed implementation and evaluation pipeline satisfy Rule 9 for the
frozen pilot. Formal Stage-V validation may start from the hashes above. Do not
run candidate test inference manually: only the unmodified runner may proceed
after the live-recomputed two-corpus summary returns `ADVANCE_TO_STAGE_P`.
