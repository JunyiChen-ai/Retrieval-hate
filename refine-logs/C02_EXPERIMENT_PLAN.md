# C02 Experiment Plan — Evidence-Density Quotient Geometry

**Status:** `FROZEN / KILL_C02_DESIGN_COLLISION_OR_INFEASIBILITY`  
**Candidate:** `C02 Evidence-Density Quotient Geometry`  
**Date:** 2026-07-29 (Pacific/Auckland)  
**Scope:** design and review only. This file does not authorize Python execution, cache
opening, teacher calls, GPU use, test access, or any SLURM submission.

> **Terminal notice:** the design below is retained as a historical proposal.
> Independent review found that its A0 uses an image-pooling proxy for a text-density
> target and that no representation-matched, full-transcript-preserving density-view
> bank exists. Therefore it cannot satisfy the frozen pre-extraction Stage-0 gate and
> must not be implemented or executed. See `refine-logs/C02_DESIGN_REVIEW.md`.

## 1. Decision summary

C02 is worth one **kill-only, zero-new-extraction A0** because three datasets exhibit
evidence-quantity structure with mutually incompatible shapes:

- HateMM: missed hate has median 85 transcript words versus 227 for caught hate, while
  flagged non-hate has 171.5 versus 47 for correct non-hate. Retrieval is length-organized
  (`rho=0.5817`) and a global length correction is measured null
  (`ERRPAT_HateMM_2026-07-26.md:230-296`).
- MHC-EN: errors rise from 9.52% to 22.50% across transcript-length quartiles, but the
  prior SAV and P3 interventions already falsified a generic mean-pooling rescue
  (`ERRPAT_MHC-EN_2026-07-26.md:284-302`;
  `research-wiki/experiments/exp-sav-f0.md:33`).
- MHC-ZH: the effect is non-monotone. The `[31,76)`-character transcript band contains
  11 of 22 stable errors in 37 items (`p=0.0048`), while both shortest and longest
  groups are easier (`ERRPAT_MHC-ZH_2026-07-26.md:281-310`).

Therefore the scientific object cannot be a scalar length threshold, one linear
length direction, P3 evidence pooling, or stream weighting. The only surviving
form is a **nonlinear equivalence relation induced by controlled views of the same
video**: the learned representation should keep semantic identity while contracting
variation caused by evidence redundancy/dilution.

The prior is nevertheless low. P3 already found a real evidence-localization signal
but no downstream conversion; exact 1-D length excision left retrieval-length
organization unchanged (`delta rho <= 0.004` in 9/9 cells); and MHC-EN is largely
label/data limited. C02 must therefore die before new extraction unless the existing
P3 view bank supplies a large two-dataset quotient-oracle margin.

## 2. Frozen claim map

### Primary claim C1

Same-video evidence-density views define a nonlinear nuisance orbit. Contracting that
orbit during RGCL training can reduce density-organized retrieval while preserving
video identity, yielding a deployable native-view representation with substantial
accuracy and macro-F1 gains on HateMM and MHC-ZH.

### Supporting claim C2

Any gain is attributable to the correct within-video orbit relation, rather than
additional capacity, generic consistency regularization, MLLM score pooling, or a
global length correction. `FULL` must beat `REMOVE`, `SHUFFLE`, and `NOISE`.

### Anti-claims

C02 does **not** claim that:

- transcript length is the hate signal or admits a universal threshold;
- one linear nuisance direction is sufficient;
- P3 pooling or summary replacement works;
- MHC-EN is representation-limited;
- P3/MLLM scores are gold labels;
- multiple views, MLLM scores, or per-item routing are needed at inference.

## 3. Method contract: EDQ-Orbit

For video `i`, let `x_i^0` be the native full-video input and let
`x_i^a, a in A_i` be train-only controlled density views. The final encoder/projector
is `q_theta`, and inference uses only `q_theta(x_i^0)`.

Two view families are allowed:

1. **Exact-content quantity view.** Keep frames and transcript token order fixed, and
   repeat the complete transcript once. This changes quantity/redundancy without
   adding a semantic proposition. Items that would truncate under the frozen native
   tokenizer limit are excluded from this view and counted.
2. **Evidence-core dilution view.** Keep the complete frame set fixed. Using the
   already-existing train-only K=4 P3 scores, choose the deterministic max-score
   window as the core (lowest window index breaks ties), then compare the core
   transcript with `core + score-minimum same-video windows`, preserving chronological
   order. P3 scores select a controlled view only; they are never a target, label,
   scalar feature, pooling weight, or inference input.

The train-only loss is:

`L_FULL = L_RGCL(native, y) + lambda_orbit * mean_a[1-cos(q(x^0),q(x^a))]`.

`q_theta` is a capacity-matched residual projector shared by every arm. Collapse is
prevented by the unchanged video-label RGCL objective, not by a free auxiliary label.
Only one pre-registered `lambda_orbit` may be chosen inside outer-train folds; no
test- or outer-held tuning is permitted.

Binding mechanism diagnostics:

- median same-video orbit radius must contract relative to `REMOVE`;
- a nonlinear OOF length probe and the direct retrieval-length statistic must both
  decrease, because prior work proved a 1-D linear probe/excision insufficient;
- OOF label performance must improve, so merely hiding length without preserving
  task distinctions is a failure.

## 4. Baseline and control families

At most three baseline families are allowed.

1. **Paired task floor:** strongest same-protocol native-view RGCL baseline with the
   same split, label space, fold heads, seed, checkpoint rule, retrieval evaluator,
   and parameter count.
2. **Direct nuisance baselines:** the already-recorded P3 pooling result and 1-D
   length-excision result are cited, not rerun. A native-view capacity-matched residual
   projector with `lambda_orbit=0` is the executable direct floor.
3. **Mechanism controls:** `REMOVE` (`lambda_orbit=0`), `SHUFFLE` (ID-hash permutation
   of the non-native view while retaining its marginal distribution; labels are not
   used), and `NOISE` (deterministic ID-hash random tangent, norm-matched to the true
   native-to-view displacement).

No global threshold, length coefficient, stream gate, modality dropout, summary
input, inference-time view search, or P3-weighted pooling may be added.

## 5. Core experiment blocks

### Block A0 — existing-bank quotient reachability, kill-only

- **Purpose:** decide whether C02 merits any new view extraction.
- **Datasets:** HateMM and MHC-ZH only.
- **Inputs:** existing train-only P3 `mean`, `wsoftT1`, and `wmild` cache triplets plus
  their exact IDs/labels. No dev/test path may be opened.
- **Arena:** full-train leave-one-out raw representation arena. It may kill C02 but
  may not promote it.
- **Oracle:** each video's three existing views form a discrete orbit. Use the
  symmetric quotient similarity
  `s_Q(i,j)=max_{a,b in {mean,soft,mild}} cos(z_i^a,z_j^b)` with the same top-20
  weighted vote as the paired `mean` floor. This is explicitly an optimistic
  representation-orbit oracle, not a deployable router.
- **Controls:** mean-only floor, hash-shuffled orbit membership, and norm-matched
  random orbits.
- **Success:** on **both** HateMM and MHC-ZH, quotient-oracle minus mean-only must be
  at least `+0.050 accuracy` and `+0.050 macro-F1`, with enough
  corrected-minus-broken LOO items for a final `+0.030` gain. `FULL` must beat both
  controls in both metrics.
- **Failure:** any dataset fails either `+0.050` bar, controls match/exceed `FULL`,
  duplicate IDs/view misalignment occurs, or any dev/test/test-like path opens.
- **Action:** failure freezes `KILL_C02_EXISTING_ORBIT_UNREACHABLE` and advances the
  serial loop to C03. Passing A0 authorizes only prospective Stage-1 implementation
  and fresh review, not a GPU job.

### Block S1 — actual-view signal gate

- **Purpose:** test whether the exact-content and evidence-core view relations are
  learnable in the strongest native Qwen representation.
- **Data:** train split only, five strict outer folds; view construction and any
  hyperparameter choice occur inside outer-train.
- **Systems:** `FULL`, `REMOVE`, `SHUFFLE`, `NOISE`, all capacity matched.
- **Primary metrics:** strict OOF accuracy and macro-F1; paired example bootstrap,
  10,000 replicates.
- **Mechanism metrics:** median orbit-radius contraction; nonlinear OOF prediction
  of `log1p(transcript quantity)` using a frozen RBF-kernel ridge probe
  (`gamma=1/d`, ridge=1); and
  `rho(query quantity, median quantity of top-20 train neighbors)`.
- **Success:** on both HateMM and MHC-ZH, `FULL-REMOVE >= +0.040` accuracy and
  macro-F1, 95% paired-bootstrap lower bound `>0`; `FULL` beats `SHUFFLE` and
  `NOISE`; median orbit radius falls by at least 25%; and both nonlinear length
  predictability and absolute retrieval-length `rho` fall by at least 20%.
- **Failure:** any performance/mechanism gate fails, view support is below 60% of
  train videos, a view changes the gold label contract, or P3 scores enter the
  classifier as targets/features.

### Block S2 — seed-0 native-view end-to-end pilot

- **Purpose:** confirm conversion under the actual fold-head/deployed-head path.
- **Inference:** native full video only; no teacher, score, view pool, router, or
  per-item branch.
- **Success:** `FULL-REMOVE >= +0.020 accuracy` and `+0.020 macro-F1` on both
  untouched development arenas, with no claimed-dataset harm below `-0.005`;
  `FULL` must also beat `SHUFFLE` and `NOISE`.
- **Failure:** any bar fails. Stop before seeds 1/2.

### Block S3 — three-seed promotion

- **Seeds:** 0/1/2, only after S2 and a fresh execution review.
- **Primary protocol:** final epoch/no selection; validation-selected is
  corroborative.
- **Success:** mean `>=+0.030 accuracy` and `>=+0.030 macro-F1` on at least two
  datasets, all 3/3 paired seed deltas positive, hierarchical paired-bootstrap
  lower bounds above zero after Holm correction, and `FULL` beats
  `REMOVE/SHUFFLE/NOISE`.
- **Failure:** anything weaker is a scientific negative; do not add views, prompts,
  larger teachers, epochs, or model size.

### Block S4 — paper-facing attribution

Run only after S3 success:

- native-versus-repeat and native-versus-core+dilution removal ablations;
- stable-error fix/break table for HateMM speech-poor hate and ZH thin-transcript
  bands;
- plot view agreement versus nonlinear length predictability and final paired gain;
- report failures on MHC-EN as an anti-claim, not a hidden dataset.

## 6. Execution order and resources

1. Independent design/non-isomorphism review.
2. If and only if approved, implement A0 and obtain independent code review.
3. One CPU-only A0 SLURM job (`8 CPU / 32 GB / 0 GPU`, no `--time`).
4. Only an A0 pass can unlock actual-view extraction design.
5. S1 extraction/signal gate, then S2 seed 0, then S3 seeds 0/1/2.

All compute must use SLURM and `conda activate HateVideo`. `JobHeldUser` must
auto-release. No release, retry, array, dependency, chain submission, test access,
or force/overwrite is authorized by this plan.

## 7. Novelty and non-isomorphism risk

The generic ideas “length invariance,” “counterfactual length debiasing,” and
“quotient representation” are not novel:

- Jwalapuram et al., ACL 2022, use a random contiguous positive-document slice to
  introduce length invariance in contrastive coherence learning:
  <https://aclanthology.org/2022.acl-long.418/>.
- CoLD (2025) uses counterfactual verbosity variants, an explicit length penalty,
  a bias estimator, and joint length-invariant reward-model training:
  <https://arxiv.org/abs/2507.15698>.
- OQ-TSAE (2026) explicitly defines nuisance-canonicalized observation quotients and
  quotient-consistent diagnostics:
  <https://arxiv.org/abs/2606.16210>.

The only potentially defensible delta is narrow:

> A train-only, same-video evidence-density orbit defined by exact-content
> repetition plus localized core/background views, used to contract the native
> RGCL memory geometry while the teacher and all alternate views disappear at
> inference, with direct controls against P3 pooling, global length correction,
> shuffled pairing, and norm-matched consistency.

This remains a medium-to-high novelty risk. If an independent reviewer finds that
the method reduces to generic view-consistency regularization, P3 pooling in the
loss, CoLD-style length debiasing, or per-pair routing, C02 must be revised or
killed before implementation.

## 8. Frozen interpretation boundary

- An A0 pass is only reachability evidence.
- S1 is only a train-OOF signal result.
- S2 is only a seed-0 development pilot.
- Only S3 can support the primary performance claim.
- Any scientific KILL is frozen; the serial next action is C03 design, not a C02
  retry or a stronger/larger version.

## 9. Terminal independent-review adjudication

The independent review first returned `REVISE_DESIGN` because the proposed A0 used
P3 image-pooling variants as a proxy for a text-density orbit and because
evidence-core deletion was not guaranteed to preserve meaning. A follow-up
read-only asset audit found no HateMM+MHC-ZH representation bank for legal
full-transcript-preserving repeat/localized-repeat views.

The final verdict is:

`KILL_C02_DESIGN_COLLISION_OR_INFEASIBILITY`

Under the frozen candidate-registry contract, C02 cannot pass the required
existing-bank two-dataset Stage-0 oracle without first extracting the very views
that Stage 0 forbids. The abstract EDQ hypothesis is not experimentally falsified;
this specific candidate is infeasible under the current gate and is frozen before
implementation. The authoritative review record is
`refine-logs/C02_DESIGN_REVIEW.md`.

No A0 code/config/schema/wrapper was created, no cache was loaded, and no Python,
teacher, GPU, test, or SLURM action occurred. The next boundary is C03 design and
independent review only.
