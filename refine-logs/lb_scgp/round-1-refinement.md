# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable component of hateful-video RGCL, and do not stop until one frozen method improves the final **ordinary full-video train-memory kNN** accuracy and macro-F1 by at least `+0.030` absolute each over the moving strongest same-protocol non-MLLM comparator on both MHC-EN and MHC-ZH, under paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied verdicts, scores, summaries, auxiliary fields, segment salience, sparse neighbour events, frozen swaps or pseudo-groups. Their semantics were redundant with video labels, orthogonal to vote correctness, absorbed by the fusion head, too sparse, or reducible to sample reweighting/generic gradient surgery. The successor must let label-blind whole-video semantics change the complete final retrieval geometry, including entries outside every old top-64 list, without becoming a teacher key selector, direct rule loss, sample/group weighting or generic pair/triplet metric learning.
- **Non-goals:** No segment, timestamp, span or localization endpoint; no segment weighting/loss; no MLLM verdict, score, free rationale, selected neighbour/key, memory replacement, feature/schema/summary concatenation, score fusion, reranking, veto, router/MoE, native-head claim, direct rule-energy loss, pseudo-group robust optimization, sample reweighting or generic pair/triplet/SupCon method; no test-time MLLM/certificate/target; no scale, data, epoch or ensemble rescue.
- **Constraints:** The **only gold supervision is the parent video's binary label**. No segment-level gold exists or may be assumed. Proposition, stance, quotation, condemnation, reportage and cross-modal-binding states are confidence-bearing train-only weak/privileged MLLM pseudo-signals, never annotations or gold. The teacher is strictly label-blind and receives no label, prediction, margin, error/correctness, neighbour/key/ID, loss or gradient. Teacher records close immutably before a deterministic compiler may read train video labels. Validation/test never load a certificate, target bank or compiler artifact. All future computation is SLURM-only in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no source code and launches no job.
- **Success condition:** Both datasets, seeds 0/1/2, ordinary full-video kNN accuracy and macro-F1 each exceed `max(historical strongest point, paired same-seed REMOVE mean)+0.030`; all paired deltas are positive; hierarchical paired-bootstrap lower bounds exceed zero; four dataset×metric tests survive Holm FWER `0.05`; FULL significantly beats REMOVE and a permutation of certificate identity in both metrics; corruption degrades the gain; final displacement has a verified nonnegative-cone/Farkas separation from ordinary RGCL sample and generic relation gradients. Proposal readiness, solver success, target objective, train leave-one-out accuracy, an auxiliary head or seed-0 evidence cannot close the goal.

## Anchor Check

- **Original bottleneck:** obtain substantial ordinary-kNN final gains through a removable MLLM-specific train-only geometry mechanism, with video binary labels as the only gold.
- **Why the revision still addresses it:** the revision keeps the full-bank target path and removes an unsupported content-identity assumption. The certificate supplies only structural exception algebra; labels enter only after cache closure; the solver still acts on the complete PSD Gram bank and the existing encoder.
- **Suggestions rejected as drift:** no proposition free text, target name, semantic embedding or teacher-selected pair was added to recover “identity”; no segment supervision, direct rule loss, differentiable sorter, learned compiler, router or test artifact was added.

## Simplicity Check

- **Dominant contribution after revision:** label-blind structural exception-reflection equivariance on a full-bank row profile, reconciled with exact vote-safe geometry before uniform encoder fitting.
- **Components removed or merged:** all generic “semantic equivalence/same proposition” equations were deleted; eleven atoms were reduced to eight fixed closure atoms; Farkas is now an audit rather than a supporting contribution; the solver uses one global configuration.
- **Suggestions rejected as unnecessary complexity:** content identifiers, closed target ontologies, learned compilers, differentiable sorting and dataset-specific solver grids.
- **Why the remaining mechanism is the smallest adequate route:** one structured cache, one 48-row maximum semantic operator, one correlation-target solver and one uniform fit loss; zero new trainable/inference modules.

## Changes Made

### 1. Compiler semantics

- **Reviewer said:** presence atoms cannot identify the same proposition; generic equivalence would connect unrelated content.
- **Action:** removed proposition identity and all equivalence equations. The sole semantic operator is content-invariant structural exception reflection across direct/quotation/condemnation/reportage states. Exact profile columns and Householder/permutation reflections are enumerated below.
- **Reasoning:** the teacher can certify the role of an exception without emitting content text or choosing pairs. Whether that structural transformation is stable across content is now the explicit falsifiable hypothesis, not an assumed fact.
- **Impact:** tighter novelty and smaller operator; failure of structural equivariance stops the route.

### 2. Solver and SCGP-0

- **Reviewer said:** rank-cell membership and coupled pivots were missing; all-error repair risked collapse; a generic SDP was infeasible.
- **Action:** defined the global top-20 rank-cell polyhedron, simultaneous symmetric boundary events, bounded orientation enumeration, local-stationary convergence claim, uniform class-balanced slack, matrix-free Dykstra/ADMM projections, exact constants and a synthetic-plus-real-fold microbenchmark gate.
- **Reasoning:** within a stated cell, exact vote constraints must truly be linear; all rank changes must be verified as changes of the same symmetric Gram variable.
- **Impact:** the method is implementable and makes no global-nearest claim.

### 3. Farkas, fitting and pilot

- **Reviewer said:** scope the cone claim, inspect realized displacement, close factor/rollback state, and define design-weighted held-out pilot estimation.
- **Action:** cones now contain explicitly registered descent primitives, with a complete pair/triplet column-generation oracle; the audit is applied to both abstract and realized displacement and cannot substitute for learned controls. Factor dimensions, deterministic SVD, collapse guards and rollback state are explicit. Pilot inclusion probabilities, Horvitz-Thompson estimands, cross-fit halves and partial-coverage semantics are frozen.
- **Reasoning:** free-embedding separation is mechanism evidence only for the named cones; realized AdamW behavior must be tested empirically.
- **Impact:** narrower claims and stronger executable evidence.

## Revised Proposal

# Research Proposal: LB-SCGP — Label-Blind Structural Exception Gram Projection

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable component of hateful-video RGCL, and do not stop until one frozen method improves the final **ordinary full-video train-memory kNN** accuracy and macro-F1 by at least `+0.030` absolute each over the moving strongest same-protocol non-MLLM comparator on both MHC-EN and MHC-ZH, under paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied verdicts, scores, summaries, auxiliary fields, segment salience, sparse neighbour events, frozen swaps or pseudo-groups. Their semantics were redundant with video labels, orthogonal to vote correctness, absorbed by the fusion head, too sparse, or reducible to sample reweighting/generic gradient surgery. The successor must let label-blind whole-video semantics change the complete final retrieval geometry, including entries outside every old top-64 list, without becoming a teacher key selector, direct rule loss, sample/group weighting or generic pair/triplet metric learning.
- **Non-goals:** No segment, timestamp, span or localization endpoint; no segment weighting/loss; no MLLM verdict, score, free rationale, selected neighbour/key, memory replacement, feature/schema/summary concatenation, score fusion, reranking, veto, router/MoE, native-head claim, direct rule-energy loss, pseudo-group robust optimization, sample reweighting or generic pair/triplet/SupCon method; no test-time MLLM/certificate/target; no scale, data, epoch or ensemble rescue.
- **Constraints:** The **only gold supervision is the parent video's binary label**. No segment-level gold exists or may be assumed. Proposition, stance, quotation, condemnation, reportage and cross-modal-binding states are confidence-bearing train-only weak/privileged MLLM pseudo-signals, never annotations or gold. The teacher is strictly label-blind and receives no label, prediction, margin, error/correctness, neighbour/key/ID, loss or gradient. Teacher records close immutably before a deterministic compiler may read train video labels. Validation/test never load a certificate, target bank or compiler artifact. All future computation is SLURM-only in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no source code and launches no job.
- **Success condition:** Both datasets, seeds 0/1/2, ordinary full-video kNN accuracy and macro-F1 each exceed `max(historical strongest point, paired same-seed REMOVE mean)+0.030`; all paired deltas are positive; hierarchical paired-bootstrap lower bounds exceed zero; four dataset×metric tests survive Holm FWER `0.05`; FULL significantly beats REMOVE and a permutation of certificate identity in both metrics; corruption degrades the gain; final displacement has a verified nonnegative-cone/Farkas separation from ordinary RGCL sample and generic relation gradients. Proposal readiness, solver success, target objective, train leave-one-out accuracy, an auxiliary head or seed-0 evidence cannot close the goal.

The certificate is a fallible train-only pseudo-signal. “Structural” explicitly means that the method does **not** identify proposition content, target identity or semantically matched videos.

## Technical Gap and Thesis

For normalized full-bank Gram `G`, stable self-excluded ranking `pi_i(G)`, `c_i=2y_i-1`, `K=20` and `w_r=21-r`, the repository true-class vote margin is

`m_i(G)=c_i/210 * sum_{r=1}^{20} w_r c_{pi_i(r)} G[i,pi_i(r)]`.

Previous routes either altered inputs/heads, selected individual neighbours, froze a sparse action universe, or reweighted gradients. None asked a constrained MLLM to express a higher-order whole-video exception transformation and then solved its nearest full-bank realization under the exact final vote.

> **Thesis:** A label-blind exception algebra defines a content-invariant higher-order reflection of full-bank similarity row profiles; solving this reflection as an exact-vote-safe stopped Gram target before uniform shared-encoder fitting can repair ordinary kNN geometry in a way that registered scalar-example and generic relation primitives cannot reproduce.

The single semantic novelty is **exception-reflection row-profile equivariance under exact vote constraints**. No ordinary semantic-equivalence/clustering equation exists.

## Complexity and Data Flow

- Reuse the existing RGCL shared encoder, data, labels, schedule, optimizer, bank, checkpoint rule and ordinary top-20 arithmetic similarity vote.
- Add zero trainable modules and zero inference modules.
- Add one closed teacher cache, one deterministic semantic operator with at most 48 scalar rows, one dense `N<600` correlation target and one uniform target-fit loss.

```text
train whole video (uniform frames + full ASR/OCR/title)
 -> label-blind constrained certificate -> immutable cache closure
 -> post-cache labels + deterministic structural compiler
 -> exception-reflection operator A_exc

eval-mode full train bank Z0 -> correlation G0
 -> global rank-cell exact-vote-safe target G*
 -> PSD factor + Procrustes Z*
 -> uniform fit of the existing encoder

validation/test -> existing encoder -> rebuilt full-video train bank
 -> unchanged ordinary kNN; no certificate/compiler/target
```

## Teacher Certificate and Firewall

The teacher sees one whole training video through fixed uniform frames and complete available ASR/OCR/title. These channels are not segment gold. The payload contains none of `{label,prediction,probability,margin,error,correctness,loss,gradient,neighbour,key,memory ID,split,seed}`.

The strict JSON has exactly eight atoms. Every atom is `{"state": supported|contradicted|unresolved, "confidence": 0|1|2|3|4}`:

1. `proposition.group_referent_present`
2. `proposition.derogatory_exclusionary_or_violent_predicate_present`
3. `cross_modal.target_predicate_bound`
4. `stance.speaker_asserts_or_endorses_surface_proposition`
5. `exception.quotation_negates_direct_speaker_attribution`
6. `exception.condemnation_negates_direct_speaker_attribution`
7. `exception.reportage_negates_direct_speaker_attribution`
8. `cross_modal.speaker_source_stance_bound`

There is no hate/policy verdict, severity, score, target/proposition text, target name, rationale, segment, span, timestamp or ID. Supported exception atoms mean that the named presentation relation defeats the direct-speaker reading; mere quotation/reportage presence is insufficient.

Four deterministic calls use two prompt paraphrases and two evidence orders. Modal ties become unresolved; canonical confidence is the minimum confidence supporting the modal state divided by four. Accept only if all calls parse, every non-unresolved atom has agreement `>=3/4`, record confidence `>=0.5`, and the fixed graph closes:

- atoms 3/4/5/6/7/8 can be supported only if atoms 1 and 2 are supported;
- supported direct endorsement (4) is mutually exclusive with each supported exception (5–7);
- at most one exception atom is supported; multiple or unclear cases abstain;
- a supported exception and contradicted source/stance binding (8) is invalid;
- supported cross-modal binding with a contradicted endpoint is invalid.

Any parse/extra-key/closure/confidence/agreement failure rejects the complete record to exact REMOVE. Only byte-identical infrastructure retry is allowed. Before use, store prompt/model revision/generator/input/code hashes, ID allowlist and record hashes; close with a Merkle root and read-only manifest. No post-outcome append/repair/schema change is allowed. A schema change is a new hypothesis.

The teacher is label-blind, not label-independent. After closure, strict cross-fit probes report atom/full-certificate prediction of label, baseline OOF error and true-class margin but never select atoms or thresholds. Capacity-matched scalar certificate-label and certificate-error propensity controls preserve the original record's confidence/coverage/missingness. Indivisible-record shuffle occurs only post-cache within label × OOF-prediction × margin-quartile × cross-fit error-propensity bins. FULL parity with a scalar propensity or P4-style atom prediction is terminal proxy/redundancy failure.

## Deterministic Structural Exception Compiler

Map states to `+1/-1/0` and multiply by canonical confidence. Rejected/unprocessed records emit no semantic membership; they remain ordinary full-bank rows and are never deleted or weighted. Only after cache closure may the pure versioned compiler read train video labels.

For each accepted record, define one state `s_i in {D,Q,C,R}` only when:

- `D`: atoms 1–4 and 8 supported, atoms 5–7 contradicted;
- `Q/C/R`: atoms 1–3 and 8 supported, atom 4 contradicted, exactly the named exception atom supported and the other exceptions contradicted;
- otherwise no semantic state.

This is a structural state, not proposition identity. Let the eight bank-cell columns be `(state s, label c)` for `s in {D,Q,C,R}`, `c in {-1,+1}`. For each cell, `B[j,(s,c)]` is the accepted record confidence divided by the cell's total confidence when `s_j=s,c_j=c`, else zero. Empty cells invalidate that exception family. Mask the diagonal and define

`R(G)=(G-diag(G))B`, an `N x 8` absolute row-profile matrix.

For anchors in each `(s,c)` cell, let `a_(s,c)` be the analogous normalized confidence membership column. The design-weighted mean profile is

`rbar_(s,c)(G)=a_(s,c)^T R(G)`.

For exception `e in {Q,C,R}`, define the fixed `8x8` orthogonal involution `T_e` that swaps `D` and `e` coordinates separately within each label and leaves the other four coordinates unchanged. Equivalently, each swap block is the Householder reflection across the hyperplane normal to `(unit_D-unit_e)/sqrt(2)`.

The sole semantic equations are

`rbar_(e,c)(G)-T_e rbar_(D,c)(G)=0`, for `c in {-1,+1}` and every support-valid `e`.

Each is eight scalar pair-of-pairs equations, so `A_exc vec(G)=0` has at most `3*2*8=48` rows. It compares aggregates of actual video-pair similarities on both sides but never chooses a neighbour/key or assumes two anchors share content. Duplicate equations are removed by canonical sparse-row hash; every row is L2-normalized; source IDs/cells/weights and compiler hash are logged. Products `A_exc vec(G)` and adjoints are computed from `a`, `B` and matrix multiplies without materializing pair quadruples.

The falsifiable semantic hypothesis is content-invariant structural equivariance: changing a direct-speaker reading to a closed exception should reflect how an anchor relates to direct/exception bank cells, independent of proposition identity. If held-out data do not support this, the method stops; no content identifier or prompt rescue is added.

## Exact-Vote Gram Target

At registered refreshes build `Z0 in R^(N x 1024)` in `model.eval()`, normalize rows, canonical-ID order them and set `G0=Z0 Z0^T`. Because `N=549/579<1024`, every PSD `G` has an exact factor in the existing dimension.

For both LABEL-ONLY and FULL define `ell_i=max(m_i(G0),epsilon_vote)` with `epsilon_vote=1e-4` and uniform slacks `xi_i>=0`. Slacks have no MLLM dependence and satisfy, separately for each class,

`sum_{i:c_i=c} xi_i <= beta * sum_{i:c_i=c} [ell_i-m_i(G0)]_+`, with `beta=0.20`.

Thus the target must remove at least 80% of each class's aggregate frozen vote deficit without forcing every error across zero. FULL and LABEL-ONLY share identical envelopes/slack budgets.

Solve the local minimum-displacement correlation target:

`min 0.5||G-G0||_F^2`

subject to:

- `G=G^T`, `G PSD`, `diag(G)=1`, `G_ij<=1-1e-4` for `i!=j`;
- `m_i(G)>=ell_i-xi_i` under the current exact cell;
- each class/global mean exact margin at least its `G0` mean;
- class-centroid squared distance, a linear block mean of `G`, at least its `G0` value;
- row trust `||G[i,-i]-G0[i,-i]||_2<=0.05 sqrt(N-1)`;
- class-mean-row trust `<=0.02 sqrt(N)`;
- FULL only: `||A_exc vec(G)||_2<=0.90||A_exc vec(G0)||_2`; LABEL-ONLY has no semantic operator.

The constants `{epsilon=1e-4,beta=.20,kappa=.90,tau_row=.05,tau_class=.02}` are one route-wide frozen tuple for EN/ZH and all folds/seeds; no grid exists.

### Global top-20 rank cells

For current stable top-20 set/order `pi_i(1:20)`, the closed global cell is the intersection over all rows of:

- `G[i,pi_i(r)]>=G[i,pi_i(r+1)]` for `r=1..19`;
- `G[i,pi_i(20)]>=G[i,j]` for every self-excluded outsider `j`;
- self index absent from every inequality.

Outsider-internal order is irrelevant to the exact vote. At exact equality, canonical ID ascending supplies the symbolic infinitesimal tie order; numerically values within `1e-7` are treated as tied and independently re-sorted by ID. The repository parity evaluator uses the same rule and must match ID/rank/cosine/prediction.

Within this polyhedron every margin is linear. All row-cell inequalities are solved **simultaneously on the same symmetric `G`**, so one entry's two-row effects are coupled.

### Matrix-free solver and coupled pivot

Use scaled ADMM with cyclic Dykstra projections onto: Frobenius proximal center; individual rank/vote halfspaces and slack budgets; row/class trust balls; semantic second-order cone; affine symmetry/unit diagonal; and the PSD cone by complete symmetric eigendecomposition. Dykstra correction variables recover dual residuals. Operator products use dense `N x N` matrices plus the compressed `A_exc`; no interior-point SDP or exhaustive pair list is used.

Per cell: at most 500 sweeps, primal/dual tolerance `1e-6`, PSD eigen tolerance `1e-7`. Verify with an independent float64 checker. If a solution reaches rank boundaries, collect **all** top20/outside/internal inequalities with slack `<=1e-7` across all rows. They reference the same symmetric entries and form one global event.

- If at most eight binary boundary orientations are independent after equality-graph reduction, enumerate the compatible global orientations in canonical bit order.
- For each orientation, take a `1e-6` probe into the adjacent cell, rerank every row simultaneously, and run 25 feasibility sweeps.
- Admit only orientations satisfying PSD/unit diagonal/trust, all recomputed individual/class/global exact envelopes and abstract collapse guards.
- Continue from the feasible orientation with lowest objective, ties by canonical orientation hash.
- More than eight independent simultaneous orientations, no feasible orientation, 32 total pivots or 12 failed backtracks ends at the best current feasible local target; if none exists, the refresh is REMOVE.

This is a bounded active-cell local solver. The claim is convergence to a numerically certified **locally stationary feasible target among explored adjacent cells**, not the globally nearest point over the nonconvex union. A new outsider can enter through successive pivots; no old top-64 universe is fixed.

Every accepted iterate is reranked from the complete bank by the independent exact evaluator. Post-hoc rank verification never substitutes for cell inequalities.

### Abstract collapse guards

Before factorization, reject targets whose effective eigen-rank falls below 80% of `G0`, whose per-class off-diagonal variance falls below 80% of `G0`, whose class-centroid distance falls, whose maximum off-diagonal exceeds `1-1e-4`, or whose normalized displacement is below `1e-4`/above the trust bound. These are candidate accept/reject guards, never outcome-tuned.

## Factor, Uniform Fit and Rollback

For float64 `G*=U Lambda U^T`, fail if any eigenvalue is below `-1e-7`; clip only values in `[-1e-7,0)` to zero, retain all nonnegative eigenvalues, and set `Zr=U sqrt(Lambda) in R^(N x N)`. Pad with 1024-N zero columns to `Zpad in R^(N x 1024)`. Compute `Zpad^T Z0=P Sigma Q^T` and `O=P Q^T`; set `Z*=Zpad O`. Repeated singular-vector signs/bases use the deterministic LAPACK basis followed by column-sign rule “largest absolute entry positive.” Verify factor/row-norm errors `<=1e-6`.

Target creation and realized-bank checks use eval mode. Training uses ordinary train mode/dropout. Refresh at epochs `{5,10,15,20,25}`. During the next epoch, exactly batches with canonical batch index divisible by four replace the ordinary RGCL step with

`L_fit=mean_i ||normalize(f_theta(x_i))-stopgrad(Z_i*)||^2`;

all other batches run ordinary RGCL. Thus target-fit fraction is exactly 0.25; every row in a fit batch has equal coefficient, regardless of confidence/state. REMOVE runs ordinary RGCL on those same batches. All arms share total steps/epochs/data order/optimizer/scheduler/checkpoint budget.

After a block rebuild `Zfit/Gfit` in eval mode. Require displacement cosine with `G*-G0 >=0.80`, relative target residual `||Gfit-G*||/||G*-G0||<=0.50`, exact individual/class/global guards, effective rank/variance `>=80%` of pre-block, and no duplicate. Otherwise restore model parameters, AdamW moments/step counters, scheduler, AMP scaler, RNG states, dataloader sampler/generator and batch cursor, then replay the block as REMOVE. The replay hash must match a direct REMOVE clone. The target itself is never an inference bank.

## Scoped Cone/Farkas Audit

Let `D` be either abstract `Z*-Z0` or realized `Zfit-Z0`. Project each row onto the tangent of the unit sphere at `Z0`, vectorize and normalize to `d`; zero displacement fails. All dictionary columns are normalized **descent displacement directions**, not gradients. Zero columns are removed and positive-collinear duplicates are canonical-hash deduplicated.

- `H_ex`: the complete registered cone of singleton-anchor descent directions from the exact ordinary RGCL implementation with the refreshed bank detached and all other settings frozen.
- `H_rel`: all active ordered cosine pair-attract/pair-repel primitives, all active margin-triplet primitives over every legal `(anchor,positive,negative)` under video labels, and one complete-bank SupCon primitive per anchor at the registered temperature/margin.

The pair set is enumerated. The triplet cone uses exact column generation: after each NNLS projection, a GPU-blocked oracle finds the legal triplet minimizing witness inner product; because a triplet direction separates into anchor-positive and anchor-negative terms at fixed anchor, the minimum is obtained by exhaustive per-anchor positive/negative scans. Add the most violating column until no column violates `H^T u>=-1e-6`; independently brute-force a small-bank oracle in parity tests. This supports claims only about these **registered primitive cones**, not all possible metric-learning algorithms.

For each cone solve `min_{alpha>=0}||H alpha-d||/||d||`. Require residual `>=0.25`. With projection `p=H alpha`, set/independently optimize `u=(p-d)/||p-d||` and verify `H^T u>=-1e-6`, `d^T u<=-0.25`, primal/separation agreement and duality gap `<=1e-5`. Run on abstract and realized displacement in every SCGP-0/FULL fold. If realized displacement enters either cone, executable non-reweighting fails even if the abstract target passes.

This free-embedding certificate says nothing by itself about AdamW parameter trajectories. Capacity/step-matched learned scalar-example, pair, triplet and SupCon controls remain binding and must not match FULL's constraints or OOF metrics.

## Strict OOF and Staged Validation

### SCGP-0: zero-teacher capacity/fitting screen

Zero teacher/new OCR calls. In each outer fold, train/target/fit uses only `T\F` full-video inputs and labels; outer `F` labels are endpoint-only. Solver constants are fixed above and cannot be selected by outer or dev/test results. Rebuilt `T\F` ordinary kNN predicts outer queries.

Before the ten OOF runs, a synthetic known-PSD/tie/pivot test and one real-fold **numerical microbenchmark without endpoint labels/results** must pass: scalar/vector/evaluator parity, PSD/factor/Procrustes, self exclusion, canonical ties, simultaneous pivot, deterministic repeat, peak memory `<24 GiB`, measured solver wall time supporting ten folds within 160 GPU-hours, and both cone oracles on the label-only target. The real fold may verify feasibility/numerics only; its query endpoint is sealed until configuration freeze.

Binding GO: both MHC and MHC-ZH pooled actual OOF accuracy and macro-F1 each `>=+0.050` over frozen geometry, every fold sign positive, target realization/collapse guards pass, abstract and realized cones/Farkas pass. Failure is terminal with zero teacher calls. LABEL-ONLY becomes a stronger moving non-MLLM comparator if it passes.

### SCGP-1: at-most-128/dataset teacher pilot

Before calls, stratify all train IDs by parent video label × strict-OOF prediction × margin quartile. Use a frozen seed for proportional allocation with a minimum per nonempty stratum, cap 128 unique IDs/dataset, and store each inclusion probability `pi_i=n_h/N_h`. The teacher payload never contains the stratum variables.

All population coverage/support estimates use Horvitz-Thompson weights `1/pi_i` with design-based variance; unprocessed videos are “not sampled,” never “rejected.” Every sampled ID is assigned by hash to cross-fit half A/B before calls. When evaluating half A, bank-cell columns/operator statistics use only accepted half B certificates; A certificates supply only the held-out structural state being scored and never enter their reference profiles. Reverse for B. Self IDs are excluded.

Governance requires 100% provenance, zero forbidden access, parse `>=95%`, design-weighted accepted closure coverage `>=80%`, agreement `>=75%`, and direct plus at least one exception state with both labels and Kish ESS `>=8` in each dataset. At least one **common** exception family must survive support in both datasets; unsupported families are removed only by this frozen support rule, never outcomes.

Held-out structural value requires the fixed reflection to reduce design-weighted row-profile residual and improve exact target correction direction beyond LABEL-ONLY, scalar label/error propensity, P4-AUX, TextTeacher/caption anchor and fine-cell indivisible CERT-SHUFFLE, with paired ID bootstrap lower bounds above zero. In strict OOF partial-pilot fits, only sampled outer-memory certificates define `A_exc`; unsampled/rejected records remain full Gram/vote rows with no semantic membership. FULL must beat every binding control by `+0.010` accuracy and macro-F1 in both datasets and pass abstract/realized Farkas. Failure stops before full-cache calls.

Maximum calls are four per selected video: 512/dataset, 1,024 total. QC evaluates whole-video schema appropriateness only and is not annotation/supervision.

### Seed-0 and final

After pilot authorization only, close the full train-only certificate cache. Seed-0 dev on both datasets must beat REMOVE, moving LABEL-ONLY, CERT-SHUFFLE, CERT-NOISE, scalar propensity, P4-AUX, TextTeacher anchor, direct-rule loss and strongest pair/triplet/SupCon control by `+0.010` in both ordinary-kNN metrics. Mixing indivisible certificate records with their post-cache train marginal at `{0,.25,.50,.75,1}` must degrade gain monotonically; missing maps to no semantic membership.

Final seeds 0/1/2 retain the immutable moving-baseline `+0.030/+0.030`, positive signs, hierarchical paired bootstrap, Holm FWER, FULL-minus-REMOVE/SHUFFLE and actual ordinary-kNN gates. Test is opened only after seed-0 freeze. Validation/test load no teacher/certificate/compiler/target.

## Controls

Staged, capacity/step matched:

- SCGP-0: REMOVE; LABEL-ONLY; scalar-example; complete pair/triplet/SupCon.
- Pilot: LABEL-ONLY; CERT-LABEL-PROPENSITY; CERT-ERROR-PROPENSITY; P4-AUX; TextTeacher/caption anchor; fine-cell CERT-SHUFFLE.
- Seed-0/final: REMOVE; strongest moving LABEL-ONLY/non-MLLM; CERT-SHUFFLE; CERT-NOISE/MISSING; strongest scalar/proxy/prior-art and generic relation controls; DARTVAE-style direct rule loss as a control only.

Every arm shares initialization, data order, steps, refreshes, optimizer, scheduler, checkpoint selection, parameter count and evaluator. FULL never uses sample weights, teacher keys, direct rules or generic relation loss.

## Failure Rules and Novelty Boundary

- No common support-valid exception family -> STOP; never add content identity or enlarge teacher.
- Byte/semantic contamination, cache mutation or scalar/P4 parity -> STOP.
- PSD/rank-cell/evaluator/tie/self-exclusion/pivot parity failure -> STOP; no tolerance relaxation.
- SCGP-0 any dataset/metric `<+.05`, negative fold, collapse, target-fit or cone failure -> zero-teacher STOP.
- Pilot design/power/held-out/OOF failure -> no full calls.
- Abstract-only improvement, realized-cone failure, native-head-only gain or unchanged-kNN failure -> STOP.
- Any segment/timestamp/span/localization gold or endpoint invalidates the route.

Closest work covers gold-grounded explanations, text anchors, rule-guided latent losses, geometry KD, proof sidecars and target-bank correction. The defensible narrow delta is:

`label-blind closed exception certificate -> content-invariant pair-of-pairs row-profile reflection -> exact rank-cell vote-safe Gram target -> uniform shared-encoder fit -> ordinary kNN`.

No claim is made for first semantic supervision, exception reasoning, Gram KD, target fitting or all metric-learning non-equivalence.

## Claim-Driven Validation and Handoff

### Claim 1: executable non-reweightable target operator

- SCGP-0 actual strict OOF `+.05/+.05` both datasets, every fold positive.
- Exact numerical parity, abstract/realized Farkas for registered cones, target realization and collapse guards.

### Claim 2: exception certificate has conditional causal value

- Design-weighted held-out 128-video pilot; common exception support; reflection residual/correction gain beyond scalar/P4/anchor/shuffle.
- Actual partial-pilot OOF `+.01/+.01` beyond all binding controls.

### Claim 3: substantial final ordinary-kNN gain

- Seed-0 all controls and monotonic corruption.
- Two datasets × seeds 0/1/2 final moving baseline `+.03` accuracy and macro-F1 with paired inference.

Highest risks are exception support collapse, infeasible local rank cells, solver cost, target-fit washout and realized displacement entering a generic control cone. The first authorized handoff is an independent implementation audit and SCGP-0 numerical microbenchmark only. No source change, SLURM job or teacher call is authorized by this refinement itself.
