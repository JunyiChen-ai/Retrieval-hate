# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable component of hateful-video RGCL, and do not stop until one frozen method improves the final **ordinary full-video train-memory kNN** accuracy and macro-F1 by at least `+0.030` absolute each over the moving strongest same-protocol non-MLLM comparator on both MHC-EN and MHC-ZH, under paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied verdicts, scores, summaries, auxiliary fields, segment salience, sparse neighbour events, frozen swaps or pseudo-groups. Their semantics were redundant with video labels, orthogonal to vote correctness, absorbed by the fusion head, too sparse, or reducible to sample reweighting/generic gradient surgery. The successor must let label-blind whole-video semantics change the complete final retrieval geometry, including entries outside every old top-64 list, without becoming a teacher key selector, direct rule loss, sample/group weighting or generic pair/triplet metric learning.
- **Non-goals:** No segment, timestamp, span or localization endpoint; no segment weighting/loss; no MLLM verdict, score, free rationale, selected neighbour/key, memory replacement, feature/schema/summary concatenation, score fusion, reranking, veto, router/MoE, native-head claim, direct rule-energy loss, pseudo-group robust optimization, sample reweighting or generic pair/triplet/SupCon method; no test-time MLLM/certificate/target; no scale, data, epoch or ensemble rescue.
- **Constraints:** The **only gold supervision is the parent video's binary label**. No segment-level gold exists or may be assumed. Proposition, stance, quotation, condemnation, reportage and cross-modal-binding states are confidence-bearing train-only weak/privileged MLLM pseudo-signals, never annotations or gold. The teacher is strictly label-blind and receives no label, prediction, margin, error/correctness, neighbour/key/ID, loss or gradient. Teacher records close immutably before a deterministic compiler may read train video labels. Validation/test never load a certificate, target bank or compiler artifact. All future computation is SLURM-only in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no source code and launches no job.
- **Success condition:** Both datasets, seeds 0/1/2, ordinary full-video kNN accuracy and macro-F1 each exceed `max(historical strongest point, paired same-seed REMOVE mean)+0.030`; all paired deltas are positive; hierarchical paired-bootstrap lower bounds exceed zero; four dataset×metric tests survive Holm FWER `0.05`; FULL significantly beats REMOVE and a permutation of certificate identity in both metrics; corruption degrades the gain; final displacement has a verified nonnegative-cone/Farkas separation from ordinary RGCL sample and generic relation gradients. Proposal readiness, solver success, target objective, train leave-one-out accuracy, an auxiliary head or seed-0 evidence cannot close the goal.

## Anchor Check

- **Original bottleneck:** a meaningful, removable train-only MLLM mechanism must produce substantial final ordinary-kNN gains with only video binary gold.
- **Why preserved:** teacher inputs/outputs, immutable cache order, post-cache labels, test-clean inference and final statistical target are unchanged. No certificate quantity is gold and no segment object exists.
- **Drift rejected:** matched direct/moment methods are controls only; FULL remains a stopped full-bank target, not direct rule loss. No learned compiler, sorter, extra module or teacher scale was added.

## Simplicity Check

- **Dominant contribution:** exact-vote-safe proximal realization of one label-blind structural-reflection moment constraint.
- **Deleted/merged:** after the pilot, exactly one common exception family survives by a support-only rule; generic “DARTVAE-style” naming is deleted; the solver is now only product-space Dykstra; bounded search cannot train.
- **Audit status:** Farkas remains mechanism audit, not contribution.
- **Smallness:** eight fixed atoms, one reflection, at most sixteen semantic scalar equations after family selection, zero new trainable/inference modules.

## Changes Made

### 1. Attribution controls and claim

- **Reviewer said:** the target might be conditional moment matching plus KD.
- **Action:** added exact matched `DIRECT-AEXC` and `STATE-MOMENT` controls and made them binding at pilot/seed-0/final. The paper claim is narrowed to the exact-vote-safe proximal realization, not exception algebra generally.
- **Impact:** if either direct control matches FULL, the novelty claim fails even if FULL beats REMOVE.

### 2. One solver

- **Reviewer said:** ADMM/Dykstra was ambiguous.
- **Action:** retained only product-space Dykstra projection over `(G,xi)`, with explicit cyclic updates, set projections, correction storage and stopping rules. Only fully explored adjacent cells yield `LOCAL_STATIONARY_CERTIFIED`; every budget stop maps to REMOVE.
- **Impact:** unique reproducible algorithm and honest termination semantics.

### 3. Remaining audits

- **Reviewer said:** triplet activity is joint, pilot inference must be design-based, and repeated eigenspaces need canonical handling.
- **Action:** use exact joint blocked `(p,n)` scan, machine-readable cone manifest, stratified Rao-Wu replicate bootstrap, per-half ESS gates, seed-0 fixed-continuation protocol, and coordinate-projection bases for repeated eigenspaces/nullspaces.
- **Impact:** closed oracle, estimand and determinism.

## Revised Proposal

# Research Proposal: LB-SCGP — Exact-Vote-Safe Structural-Reflection Gram Projection

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable component of hateful-video RGCL, and do not stop until one frozen method improves the final **ordinary full-video train-memory kNN** accuracy and macro-F1 by at least `+0.030` absolute each over the moving strongest same-protocol non-MLLM comparator on both MHC-EN and MHC-ZH, under paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied verdicts, scores, summaries, auxiliary fields, segment salience, sparse neighbour events, frozen swaps or pseudo-groups. Their semantics were redundant with video labels, orthogonal to vote correctness, absorbed by the fusion head, too sparse, or reducible to sample reweighting/generic gradient surgery. The successor must let label-blind whole-video semantics change the complete final retrieval geometry, including entries outside every old top-64 list, without becoming a teacher key selector, direct rule loss, sample/group weighting or generic pair/triplet metric learning.
- **Non-goals:** No segment, timestamp, span or localization endpoint; no segment weighting/loss; no MLLM verdict, score, free rationale, selected neighbour/key, memory replacement, feature/schema/summary concatenation, score fusion, reranking, veto, router/MoE, native-head claim, direct rule-energy loss, pseudo-group robust optimization, sample reweighting or generic pair/triplet/SupCon method; no test-time MLLM/certificate/target; no scale, data, epoch or ensemble rescue.
- **Constraints:** The **only gold supervision is the parent video's binary label**. No segment-level gold exists or may be assumed. Proposition, stance, quotation, condemnation, reportage and cross-modal-binding states are confidence-bearing train-only weak/privileged MLLM pseudo-signals, never annotations or gold. The teacher is strictly label-blind and receives no label, prediction, margin, error/correctness, neighbour/key/ID, loss or gradient. Teacher records close immutably before a deterministic compiler may read train video labels. Validation/test never load a certificate, target bank or compiler artifact. All future computation is SLURM-only in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no source code and launches no job.
- **Success condition:** Both datasets, seeds 0/1/2, ordinary full-video kNN accuracy and macro-F1 each exceed `max(historical strongest point, paired same-seed REMOVE mean)+0.030`; all paired deltas are positive; hierarchical paired-bootstrap lower bounds exceed zero; four dataset×metric tests survive Holm FWER `0.05`; FULL significantly beats REMOVE and a permutation of certificate identity in both metrics; corruption degrades the gain; final displacement has a verified nonnegative-cone/Farkas separation from ordinary RGCL sample and generic relation gradients. Proposal readiness, solver success, target objective, train leave-one-out accuracy, an auxiliary head or seed-0 evidence cannot close the goal.

## Thesis and Exact Endpoint

For normalized full-bank Gram `G`, stable self-excluded ranking `pi_i(G)`, `c_i=2y_i-1`, and `w_r=21-r`, the exact repository true-class margin is

`m_i(G)=c_i/210 sum_(r=1)^20 w_r c_(pi_i(r)) G[i,pi_i(r)]`.

> **Thesis:** exact-vote-safe proximal realization of a label-blind structural-reflection moment constraint can create a jointly moving full-bank target that the same shared encoder internalizes more effectively than direct semantic moment losses, while validation/test remain unchanged ordinary kNN.

The dominant contribution is the proximal **realization interface**, not certificate extraction, moment alignment, Gram KD or a universal non-metric theorem.

## Complexity and System

- Frozen: exact RGCL encoder, full-video inputs/labels, optimizer/schedule/steps, checkpoint rule, refreshed train bank and top-20 arithmetic similarity vote.
- New trainable/inference components: zero.
- New train-only artifacts: one closed certificate cache, one support-selected reflection operator, stopped `G*/Z*`, and audit logs.

```text
whole train video -> label-blind constrained certificate -> immutable cache
cache CLOSED -> deterministic compiler may read train video labels
 -> one support-valid structural reflection A_exc
eval-mode bank Z0/G0 -> exact rank-cell correlation target G*
 -> factor/Procrustes Z* -> uniform shared-encoder fit
val/test -> rebuilt ordinary full-video train-memory kNN only
```

## Label-Blind Certificate

Teacher input is one whole train video: frozen uniform frames plus complete available ASR/OCR/title. These are ordinary channels, not segment labels. Input contains none of `{label,prediction,probability,margin,error,correctness,loss,gradient,neighbour,key,memory ID,split,seed}`.

Strict JSON has exactly eight atoms, each `state in {supported,contradicted,unresolved}` and `confidence in {0,1,2,3,4}`:

1. group referent present;
2. derogatory/exclusionary/violent surface predicate present;
3. cross-modal target-predicate binding;
4. speaker asserts/endorses surface proposition;
5. quotation negates direct-speaker attribution;
6. condemnation negates direct-speaker attribution;
7. reportage negates direct-speaker attribution;
8. cross-modal speaker-source/stance binding.

No verdict, score, severity, proposition/target text/name, rationale, segment/span/timestamp or ID exists. Four deterministic calls use two prompts × two evidence orders. Modal ties abstain; accepted non-unresolved atoms need agreement `>=3/4`, confidence `>=.5`, all calls parsed and fixed closure: atoms 3–8 require proposition endpoints; direct endorsement excludes supported exceptions; at most one exception; exception/source binding cannot contradict; supported binding cannot have contradicted endpoint.

Any parse/extra-key/closure/agreement/confidence failure rejects the whole record to no semantic membership. Only byte-identical infrastructure retry. Prompt/model revision/generator/input/code hashes, ID allowlist and record hashes close through a Merkle root before labels are readable. No repair/schema change after outcomes.

Label-blind is not label-independent. Post-cache strict cross-fit diagnostics report certificate/atom prediction of label, OOF error and margin without selecting schema. Scalar label/error propensity controls preserve confidence/coverage/missingness. Fine-cell shuffle moves indivisible records within label × OOF prediction × margin quartile × error-propensity bins. Scalar/P4 parity is STOP.

## Compiler and One Structural Reflection

Map states to `+1/-1/0` times confidence. Accepted records obtain `s_i in {D,Q,C,R}` only under the closed direct or exactly-one-exception patterns; all others have no state but stay in the full Gram/vote bank.

For each `(state,label)` define normalized confidence bank column `B[:,(s,c)]` and anchor column `a_(s,c)`. With diagonal masked:

`R(G)=(G-diag(G))B`,  
`rbar_(s,c)(G)=a_(s,c)^T R(G)`.

For exception `e`, fixed orthogonal involution `T_e` swaps D/e profile coordinates independently inside each label and leaves the rest. Candidate equations are

`rbar_(e,c)(G)-T_e rbar_(D,c)(G)=0`, `c in {-1,+1}`.

They are content-invariant aggregate pair-of-pairs moment equations; they never assert same proposition or select a key.

After the 128-video pilot, choose **at most one** common exception family using support only:

`e*=argmax_e min_(dataset,half,state in {D,e},label) ESS`,

requiring the minimum `>=8`; canonical tie order `Q<C<R`. No endpoint/residual chooses the family. If none passes in both datasets/cross-fit halves, STOP. With one family, `A_exc` has exactly 16 L2-normalized scalar rows after duplicate hashing. Products/adjoints use the compressed `a,B,T` algebra; source hashes are machine-readable.

## Gram Target and Rank Cells

At refresh epochs `{5,10,15,20,25}`, build normalized eval-mode `Z0 in R^(N x 1024)`, canonical-ID order, `G0=Z0Z0^T`. `N<600<1024`.

For FULL and LABEL-ONLY, `ell_i=max(m_i(G0),1e-4)`, slack `xi>=0`, and separately per class

`sum xi_i <= .20 sum [ell_i-m_i(G0)]_+`.

This is uniform action strength, not headroom evidence. Target objective in product space is

`min_(G,xi) .5||G-G0||_F^2 + .5||xi||_2^2`

subject to:

- symmetric PSD `G`, `diag(G)=1`, off-diagonal `<=1-1e-4`;
- current-cell `m_i(G)+xi_i>=ell_i`;
- class/global mean exact margins and class-centroid distance no lower than `G0`;
- row trust `.05sqrt(N-1)` and class-mean-row trust `.02sqrt(N)`;
- FULL: `||A_exc vec(G)||<=.90||A_exc vec(G0)||`; LABEL-ONLY omits it.

The global top-20 cell contains all rows' 19 internal order inequalities and twentieth-versus-every-self-excluded-outsider inequalities. Equal values use symbolic canonical-ID order; numeric ties within `1e-7` are ID sorted. All constraints share one symmetric `G`.

At a shared boundary, reduce all active equality constraints to independent orientations. If at most eight, enumerate every globally compatible orientation, take a `1e-6` probe, rerank **all** rows, and solve each feasible adjacent cell. More than eight, 32 total pivots, an unresolved final tie boundary or incomplete orientation enumeration returns `BOUNDED_SEARCH_FEASIBLE`, which is logged but maps to REMOVE and cannot train. `LOCAL_STATIONARY_CERTIFIED` requires every adjacent orientation checked and no feasible improving cell. No feasible current cell is `REMOVE_FALLBACK`. Only `LOCAL_STATIONARY_CERTIFIED` yields `G*`.

## Unique Product-Space Dykstra Solver

Let `x=(G,xi)`, `x0=(G0,0)`, convex sets `C_1..C_M`, corrections `p_s=0`. For each cycle and set in frozen order:

```text
y <- x + p_s
x <- Projection_Cs(y)
p_s <- y - x
```

Stop only after a complete cycle with maximum independently recomputed set violation `<=1e-6` and `||x_new-x_old||/max(1,||x_old||)<=1e-7`; maximum 500 cycles. The fixed set order/projections are:

1. **symmetric/unit diagonal affine:** average `G,G^T`, set diagonal one;
2. **PSD:** float64 symmetric eigendecomposition, clip negative eigenvalues to zero;
3. **off-diagonal box:** scalar clipping;
4. **each sparse rank halfspace:** Euclidean halfspace projection with one stored scalar correction;
5. **each vote/class/global/centroid halfspace:** same closed projection on symmetrized Gram coefficients and `xi` where applicable;
6. **nonnegative class-budget slack:** per-class projection onto `{xi>=0,sum xi<=budget}` by sorted simplex threshold;
7. **each row/class-mean trust ball:** closed radial projection of its symmetric row/block coefficient;
8. **semantic ellipsoid:** project `A g` to its radius by solving the 16-dimensional dual `g'=y-A^T lambda`; diagonalize `AA^T` once and find the unique nonnegative multiplier by safeguarded Newton/bisection.

Sparse halfspace corrections store only coefficient support plus one scalar; dense corrections exist only for PSD/affine/box/trust/semantic sets. Dykstra is the sole solver—no ADMM, interior point or claim that Dykstra corrections are the Farkas witness. An independent float64 verifier recomputes every constraint, exact rankings/margins, primal distance and cell consistency.

Microbenchmark reports per-cycle and PSD-eigen time, number of rank halfspaces, operator/adjoint relative error, cycles/pivots, correction memory, peak GPU memory and verifier residual. Synthetic known-solution plus one sealed real fold must project total ten-fold SCGP-0 cost `<160 GPU-hours`; otherwise STOP before endpoints/teacher.

Abstract target acceptance requires effective rank/per-class variance `>=80%` baseline, centroid distance nondecrease, no duplicate, and trust-bounded nontrivial displacement.

## Factor, Fit and Rollback

Float64 eigendecompose `G*=U Lambda U^T`, reject eigenvalues `<-1e-7`, clip only `[-1e-7,0)`, form `Zr=U sqrt(Lambda) in R^(N x N)`, zero-pad to `N x 1024`, then orthogonal Procrustes to `Z0`.

Repeated eigen/singular subspaces use an explicit coordinate-projector basis: form the subspace projector, project canonical coordinate axes in ID order, deterministic modified Gram-Schmidt, skip norm `<1e-10`, and fix each vector's largest absolute coordinate positive. Fix CPU linear-algebra backend/version/thread count in the manifest. Factor/row errors must be `<=1e-6`.

Each block is exactly the epoch after a refresh. Batches whose canonical index is divisible by four replace base RGCL with uniform

`mean_i ||normalize(f_theta(x_i))-stopgrad(Z_i*)||^2`; others use base RGCL. REMOVE uses base RGCL on all batches. Every arm starts the identical fold checkpoint/model/AdamW/scheduler/scaler/RNG/sampler state and batch order and uses the same checkpoint rule.

After the epoch, eval-mode `Gfit` requires displacement cosine `>=.80`, relative target residual `<=.50`, all exact/collapse guards and no duplicate. Failure restores model, AdamW moments/counters, scheduler, scaler, every RNG, sampler/generator and epoch-start cursor, then replays the complete epoch as REMOVE; replay hash must match a direct clone.

## Farkas Audit with Complete Registered Cones

For abstract and realized displacement, row-sphere-tangent project, vectorize and normalize `d`. Columns are normalized legal **descent displacements**; zero/positive-collinear duplicates are hash removed.

- Example cone: complete singleton-anchor directions of exact detached-bank ordinary RGCL.
- Pair cone: only same-label attraction and opposite-label repulsion; never both directions for a pair.
- Triplet cone: every label-legal currently active margin triplet. The oracle performs exact joint GPU-blocked `(positive,negative)` scan for each anchor, including joint hinge activity, and returns the globally most witness-violating column. A small bank is exhaustively parity checked.
- SupCon cone: one complete-bank primitive per anchor at frozen temperature.

Column generation stops only when no legal primitive violates witness feasibility `H^Tu>=-1e-6`. Emit a machine-checkable manifest of universe definition/count, active-mask hash, enumerated/generated columns, oracle minima and brute-force parity.

NNLS residual, independently optimized Farkas witness feasibility, separation `>=.25` and gap `<=1e-5` must pass for every named cone on both abstract and realized displacement. The claim is limited to these cones. Learned matched controls remain binding for AdamW behavior.

## Binding Direct/Moment Controls

In addition to REMOVE/LABEL-ONLY/scalar/P4/shuffle/noise/pair-triplet-SupCon:

- **DIRECT-AEXC:** at the identical refreshes/fit batches, directly backpropagate `||A_exc vec(G_theta)||^2` through a complete eval-state-equivalent full bank, with no proximal target. Its single coefficient is fixed before outcomes to match FULL's first accepted target-fit batch gradient norm; if exact matching is impossible within 5%, STOP the comparison.
- **STATE-MOMENT:** same D/e*×label cells, confidence, coverage and fit batches; minimize the sum of class-conditional weighted mean and covariance discrepancies between direct and exception embeddings (linear-kernel MMD plus Frobenius covariance difference). One global coefficient is strength-matched by the same rule.

They receive identical steps, refreshes and parameter count. FULL must beat both on actual OOF accuracy/macro-F1, exact-vote repair and realized `A_exc` residual. Matching means proximal realization is an engineering reformulation and the route's dominant novelty fails.

## Staged Validation

### SCGP-0 — zero teacher

Five strict outer folds; target/fitting sees only outer memory inputs/labels, query labels endpoint-only. Synthetic+sealed real-fold numerical microbenchmark first. Both datasets actual ordinary OOF accuracy/macro-F1 must each gain `>=.050`, every fold positive, target/fitting/collapse and abstract/realized Farkas pass. Frozen geometry, LABEL-ONLY and every learned control clone the same fold checkpoint/state/batch order/checkpoint rule. LABEL-ONLY passing raises the moving baseline. Slack feasibility is not headroom evidence.

### SCGP-1 — at most 128 videos/dataset

Before calls, proportional stratified sample by video label × strict-OOF prediction × margin quartile; store inclusion probability. Hash split into A/B halves. Population coverage/support uses Horvitz-Thompson estimates. All uncertainty uses 10,000 stratified Rao-Wu rescaled bootstrap replicates respecting without-replacement unequal-probability sampling, not ordinary paired-ID bootstrap.

For each dataset and half, D×both labels and candidate e×both labels need Kish ESS `>=8`. Choose one common `e*` by the frozen min-ESS rule. When A is evaluated, B alone defines reference `B/a` columns; reverse for B. Unprocessed is not rejected and keeps ordinary Gram/vote participation.

Governance: 100% provenance, zero forbidden access, parse `>=95%`, design-weighted closed coverage `>=80%`, agreement `>=75%`, common-family support. QC only audits whether a whole-video schema state is appropriate; QC is never a training label and creates no segment/span record.

Held-out reflection residual/correction direction must beat label-only, scalar label/error propensity, P4, TextTeacher, fine-cell shuffle, DIRECT-AEXC and STATE-MOMENT with design-based lower bounds above zero. Partial-pilot actual OOF uses seed 0, the exact SCGP-0 fold checkpoints, a fixed one-epoch continuation per registered refresh, no further checkpoint selection, identical cloned states, and must beat every binding control `+.010` accuracy/macro-F1 on both datasets plus exact-vote repair/residual. Otherwise no full calls. Four calls/video: <=512/dataset.

### Seed-0 and final

After pilot authorization, full train-only cache. Seed-0 dev on both datasets: FULL beats REMOVE, moving LABEL-ONLY, CERT-SHUFFLE/NOISE, scalar propensity, P4/TextTeacher, DIRECT-AEXC, STATE-MOMENT and strongest legal relation control by `+.010` in both ordinary-kNN metrics; corruption `{0,.25,.50,.75,1}` monotonically removes gain.

Final seeds 0/1/2: unchanged moving-baseline `+.030` accuracy/macro-F1 on both datasets, all signs positive, paired hierarchical bootstrap lower bounds, Holm FWER and significant FULL-minus-REMOVE/SHUFFLE. Validation/test load no certificate/compiler/target.

## Failure and Novelty Boundary

Any no-common-family, contamination, cache mutation, scalar/P4/direct/moment parity, solver/evaluator/tie/self-exclusion failure, non-certified rank-cell stop, SCGP-0 gate failure, abstract/realized cone failure, target-fit/collapse failure, native-head-only gain or segment/timestamp/span/localization gold use is terminal. No tolerance, schema, teacher, epoch or scale rescue.

Closest work covers rule-guided moments, text anchors, geometry KD, direct latent rules and target-bank correction. The only claimed delta is:

`label-blind structural reflection -> exact-vote-safe proximal full-bank realization -> uniform encoder fit -> ordinary kNN`,

validated specifically against direct use of the same reflection and standard state moments.

## Claim-Driven Handoff

1. **Operator capacity/non-equivalence:** SCGP-0 `+.05/+.05`, numerical certificates, abstract+realized registered-cone Farkas.
2. **MLLM-specific structural value:** design-based 128-video pilot and actual partial OOF beyond scalar/P4/shuffle/DIRECT-AEXC/STATE-MOMENT.
3. **Final causal performance:** seed-0 controls/corruption then two datasets × three seeds final `+.03/+.03` ordinary kNN.

First authorized action after READY is an independent implementation audit and numerical SCGP-0 microbenchmark only. This refinement changes no source and launches no job.
