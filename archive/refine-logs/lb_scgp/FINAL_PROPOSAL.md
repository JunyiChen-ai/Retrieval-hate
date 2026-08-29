# Research Proposal: LB-SCGP — Exact-Vote-Safe Structural-Reflection Gram Projection

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable component of hateful-video RGCL, and do not stop until one frozen method improves the final **ordinary full-video train-memory kNN** accuracy and macro-F1 by at least `+0.030` absolute each over the moving strongest same-protocol non-MLLM comparator on both MHC-EN and MHC-ZH, under paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied verdicts, scores, summaries, auxiliary fields, segment salience, sparse neighbour events, frozen swaps or pseudo-groups. Their semantics were redundant with video labels, orthogonal to vote correctness, absorbed by the fusion head, too sparse, or reducible to sample reweighting/generic gradient surgery. The successor must let label-blind whole-video semantics change the complete final retrieval geometry, including entries outside every old top-64 list, without becoming a teacher key selector, direct rule loss, sample/group weighting or generic pair/triplet metric learning.
- **Non-goals:** No segment, timestamp, span or localization endpoint; no segment weighting/loss; no MLLM verdict, score, free rationale, selected neighbour/key, memory replacement, feature/schema/summary concatenation, score fusion, reranking, veto, router/MoE, native-head claim, direct rule-energy loss, pseudo-group robust optimization, sample reweighting or generic pair/triplet/SupCon method; no test-time MLLM/certificate/target; no scale, data, epoch or ensemble rescue.
- **Constraints:** The **only gold supervision is the parent video's binary label**. No segment-level gold exists or may be assumed. G0/G1 subclips are not inputs; inherited parent labels are not segment gold and must not enter a segment-level objective. Proposition, stance, quotation, condemnation, reportage and cross-modal-binding states are confidence-bearing train-only weak/privileged MLLM pseudo-signals, never annotations or gold. The teacher is strictly label-blind and receives no label, prediction, margin, error/correctness, neighbour/key/ID, loss or gradient. Teacher records close immutably before a deterministic compiler may read train video labels. Validation/test never load a certificate, target bank or compiler artifact. All future computation is SLURM-only in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. Round3 repairs are prepared for independent review, not run; no sanitizer/G0/teacher job has launched.
- **Success condition:** Both datasets, seeds 0/1/2, ordinary full-video kNN accuracy and macro-F1 each exceed `max(historical strongest point, paired same-seed REMOVE mean)+0.030`; all paired deltas are positive; hierarchical paired-bootstrap lower bounds exceed zero; four dataset×metric tests survive Holm FWER `0.05`; FULL significantly beats REMOVE and a permutation of certificate identity in both metrics; corruption degrades the gain; final displacement has a verified nonnegative-cone/Farkas separation from ordinary RGCL sample and generic relation gradients. Proposal readiness, solver success, target objective, train leave-one-out accuracy, an auxiliary head or seed-0 evidence cannot close the goal.

## Thesis and Endpoint

For normalized Gram `G`, self-excluded stable ranking `pi_i`, `c_i=2y_i-1`, and `w_r=21-r`, exact true-class vote margin is

`m_i(G)=c_i/210 sum_(r=1)^20 w_r c_(pi_i(r)) G[i,pi_i(r)]`.

> **Thesis:** exact-vote-safe proximal realization of a label-blind structural-reflection moment constraint can produce a jointly moving full-bank target that the shared encoder internalizes more effectively than direct use of the identical semantic moment, while inference remains ordinary kNN.

This realization interface is the only contribution. Certificates, moment alignment, Gram targets, KD and Farkas are not independently claimed new.

## Teacher Certificate and Cache Order

Teacher input is one whole train video: fixed uniform frames plus complete available ASR/OCR/title. These are input channels, never segment annotations. It receives none of `{label,prediction,probability,margin,error,correctness,loss,gradient,neighbour,key,memory ID,split,seed}`.

Strict JSON has eight fixed atoms, each `supported|contradicted|unresolved` plus confidence `0..4`: group referent; derogatory/exclusionary/violent surface predicate; cross-modal target-predicate binding; direct speaker endorsement; quotation/condemnation/reportage negation of direct-speaker attribution; cross-modal speaker-source/stance binding. No verdict, score, severity, target/proposition text/name, rationale, segment/span/timestamp or ID.

Four deterministic calls use two prompts × two evidence orders. All parse; non-unresolved modal agreement `>=3/4`; record confidence `>=.5`; fixed graph closure; at most one exception; direct endorsement and exception mutually exclusive. Failure/extra key gives no semantic membership. Only byte-identical infrastructure retry.

Prompt/model revision/generator/input/code hashes, ID allowlist and record hashes close with a Merkle root **before** the compiler may read train labels. No outcome repair/schema change. Post-cache cross-fit label/error/margin diagnostics do not select schema; scalar propensity controls preserve confidence/coverage/missingness; fine-cell shuffle moves indivisible records. Label/P4/scalar parity is STOP.

## One Structural Reflection Compiler

Closed records obtain structural state `D,Q,C,R` from the fixed atom truth table; other records remain ordinary full-bank rows with no state. For each `(state,label)`, normalized confidence bank column `B` and anchor column `a` define

`R(G)=(G-diag(G))B`, `rbar_(s,c)=a_(s,c)^T R(G)`.

For candidate exception `e`, fixed involution `T_e` swaps D/e profile coordinates inside each label. Equations are

`rbar_(e,c)(G)-T_e rbar_(D,c)(G)=0`, both labels.

They are aggregate pair-of-pairs moment equations, do not identify content and never select a key. After the pilot select at most one common family; its duplicate-hashed L2-normalized `A_exc` has at most 16 rows and matrix/adjoint products use compressed `a,B,T`.

## Exact-Vote Target

At epochs `{5,10,15,20,25}`, eval-mode normalized canonical-ID bank `Z0 in R^(N x 1024)`, `G0=Z0Z0^T`, `N<600`. FULL/LABEL-ONLY share `ell_i=max(m_i(G0),1e-4)`, `xi>=0`, and class-specific `sum xi<=.20 sum[ell_i-m_i(G0)]_+`.

Project `x0=(G0,0)` in product norm

`||(G,xi)||_H^2=||G||_F^2+||xi||_2^2`

onto constraints: symmetric PSD correlation matrix; off-diagonal `<=1-1e-4`; current-cell vote/slack envelopes; nondecreasing class/global mean margins and centroid distance; row trust `.05sqrt(N-1)`; class-mean-row trust `.02sqrt(N)`; FULL semantic radius `||A_exc vecG||<=.90||A_exc vecG0||`.

Global rank cell contains every row's 19 internal top-20 inequalities and twentieth-versus-all-self-excluded-outsider inequalities. Ties within `1e-7` use canonical ID. Shared boundaries are globally orientation-reduced on the same `G`; every compatible orientation is explored only when independent count `<=8`. More than eight, pivot budget 32, unresolved final boundary or incomplete adjacent enumeration is `BOUNDED_SEARCH_FEASIBLE` and maps to REMOVE. Only all-adjacent-checked `LOCAL_STATIONARY_CERTIFIED` targets train.

## Product-Space Dykstra and Exact Projectors

For convex sets `C_s`, correction `p_s=0`, frozen cyclic order:

```text
y=x+p_s;  x=P_Cs(y);  p_s=y-x
```

Ambient matrix space is the complete real `N x N` Frobenius space; symmetry is a separate affine set. Thus a row projector changes only the selected row during that Dykstra step, and later symmetry projection reconciles the intersection.

For any trust set `C(L,r)={G:||L(G-G0)||_2<=r}`, let `Y` be the matrix component entering the step, `q=L(Y-G0)`, and `L*` its Frobenius adjoint.

- If `||q||<=r`, `P_C(Y)=Y`.
- Otherwise KKT gives

  `G(mu)=Y-mu L*(I+mu L L*)^(-1)q`,

  where unique `mu>0` solves `||L(G(mu)-G0)||=r` by safeguarded Newton/bisection.

**Row trust:** `L_iG=G[i,-i]`; `L_iL_i*=I_(N-1)`, so this reduces exactly to row radial clipping, changing row i only.

**Class-mean trust:** `L_cG=(1/n_c)sum_(i:c_i=c)G[i,:]`; `L_cL_c*=(1/n_c)I_N`. Hence

`P(Y)=Y-L_c* n_c(1-r/||q||)q` when violated. This is the exact preimage-ball projection, not an assumed symmetric radial update.

**Semantic set:** vectorize `y=vecY`, `A=A_exc`, radius `r=.90||AvecG0||`. For violation,

`g(mu)=y-mu A^T(I+mu AA^T)^(-1)Ay`,

with unique `mu>0` satisfying `||Ag(mu)||=r`. Eigendecompose the at-most-16-dimensional `AA^T` once; solve the monotone scalar root to relative `1e-10`. If `r=0`, use the Moore-Penrose nullspace projection.

Other projections: average `G,G^T` and set diagonal one; complete float64 PSD eigen-clipping; scalar box; exact sparse halfspace projection for rank/vote/class/global/centroid constraints; per-class nonnegative capped-simplex projection for slack. Dykstra corrections are dense only for affine/PSD/box/trust/semantic sets and sparse scalar-supported for halfspaces.

Stop after full cycle only when independent max set violation `<=1e-6` and relative iterate change `<=1e-7`, max 500. Independent float64 verifier recomputes every set and exact evaluator.

For every nontrivial projector, synthetic parity records primal feasibility, KKT stationarity, complementarity, root residual, idempotence, and the variational inequality `dot(Y-P(Y),Z-P(Y))<=1e-8` for 1,000 random feasible `Z`; finite-difference directional optimality and a small dense reference solve must agree `<=1e-7`. Operator/adjoint dot-product error `<=1e-10`.

Microbenchmark reports cycle/PSD/projector time, rank-halfspace count, cycles/pivots/correction memory, peak memory and verifier residual. Synthetic plus one sealed real fold must extrapolate ten-fold SCGP-0 under 160 GPU-hours or STOP.

## Factor, Fit and Rollback

Float64 PSD factor `N x N`, zero-pad to `N x 1024`, orthogonal Procrustes to `Z0`; reject eigenvalue `<-1e-7`, clip only numerical negatives. Repeated eigenspaces/nullspaces use coordinate-axis projection and deterministic Gram-Schmidt in canonical-ID order; CPU backend/version/thread count frozen. Factor/row errors `<=1e-6`.

One block is the full epoch after a refresh. Scheduled auxiliary steps are canonical batch indices divisible by four. FULL uses uniform target fit; REMOVE uses RGCL. After each block, eval-mode realized bank must have displacement cosine `>=.80`, relative target residual `<=.50`, all exact/collapse guards and no duplicate. Failure restores model, AdamW, scheduler, scaler, every RNG, sampler and epoch cursor, then replays entire epoch as REMOVE; hash must match direct clone.

## Registered-Cone/Farkas Audit

For abstract and realized row-tangent displacement, the registered cone is explicit: singleton columns are every signed anchor-neighbor tangent (same-label attraction and opposite-label repulsion under parent-video labels only); pair columns are all normalized sums of two distinct singleton columns for the same anchor; triplet columns are all normalized sums of one same-label attractive singleton and one opposite-label repulsive singleton; SupCon is the normalized same-label attractive sum per anchor. Full pair/triplet universes are covered by separation oracles rather than by materializing every column. Machine manifests store the registered definition, universe counts, active-set hashes, oracle traces, max witness inner product for every family, residual norm and primal/dual gap.

Producer primal active-set evidence and independent dual/oracle verification must both pass on abstract and realized displacement: relative separation `>=.25`, every registered-family max witness inner product `<=1e-8`, separation not overflowed, and primal/dual gap `<=1e-5`. Claims are limited to these cones; learned controls remain binding for AdamW.

## Direct Attribution Controls

`DIRECT-AEXC` directly optimizes `||A_exc vec(G_theta)||^2`; `STATE-MOMENT` uses the same D/e×label cells/confidence/coverage to align conditional mean plus covariance. Both are controls only; FULL never directly optimizes either loss.

At every scheduled auxiliary step:

1. save current train/eval flag;
2. set `model.eval()` (dropout off, BatchNorm frozen), rebuild the complete differentiable outer-memory bank in canonical gradient-checkpointed chunks;
3. compute one full-bank auxiliary objective and one AdamW step;
4. restore `model.train()` for subsequent ordinary RGCL steps.

FULL auxiliary steps also run in eval mode, but need only the scheduled uniform target minibatch; all arms have the same number/location of optimizer steps, not the same wall time. Wall time and peak memory are reported, never claimed compute matched.

Before any pilot continuation outcome, compute full-outer-memory parameter gradients at the first accepted refresh for FULL target fit and each direct objective on all ten seed-0 fold banks. Freeze one EN/ZH-shared coefficient per control:

`lambda_k=sqrt(sum_f ||g_FULL,f||^2 / sum_f ||g_k,f||^2)`.

No per-fold/dataset/epoch rematching and no endpoint enters. These pilot coefficients are reused unchanged at seed-0 and final. During every epoch report cumulative auxiliary strength `sqrt(sum_t||lambda_k g_k,t||^2)`, FULL analogue and wall time; drift is diagnostic only and cannot tune coefficients.

Seed-0 chooses one global strongest direct control by

`argmax_k min_(dataset,metric) Delta_k_vs_REMOVE`, tie `DIRECT-AEXC` before `STATE-MOMENT`.

That frozen control is run for both datasets and seeds 0/1/2. Final attribution requires FULL-minus-strongest-direct to be positive in all three seeds for every dataset×metric, hierarchical paired-bootstrap 95% lower bound `>0`, and four tests Holm FWER `.05`, in addition to FULL-minus-REMOVE/SHUFFLE and final `+.03/+.03`. Otherwise MLLM may be useful but the proximal contribution is unsupported.

## Pilot Design and Family Selection

Before calls, sample at most 128/dataset proportionally within video-label × strict-OOF-prediction × margin-quartile strata; store inclusion probabilities; hash into A/B halves. Four calls/video maximum. Unprocessed is not rejected.

Main-sample family is the first in fixed priority `Q<C<R` whose D/e×label Kish ESS is `>=8` in both datasets and both halves. Every one of 10,000 stratified Rao-Wu rescaled replicates independently rebuilds:

1. replicate weights and HT cell totals/ESS;
2. the same first-passing family selection;
3. both A/B reference profiles with opposite-half records only;
4. held-out reflection residual and correction-direction statistic.

If no family passes in a replicate, its gain statistics are set to zero (conservative no-effect) and selection failure is counted. Require main-sample selection, replicate selection success `>=95%`, and 95% lower bounds `>0`. Thus uncertainty covers family selection instead of conditioning on it.

Governance remains: provenance 100%, forbidden access zero, parse `>=95%`, design-weighted closure coverage `>=80%`, agreement `>=75%`; QC checks only whole-video schema appropriateness and is never supervision.

Partial-pilot actual OOF uses seed 0, exact SCGP-0 fold checkpoints, fixed one-epoch continuation after each refresh, no further checkpoint selection, cloned states/batch order. FULL must beat every binding control `+.010` accuracy/macro-F1 on both datasets plus exact-vote repair/residual/Farkas; otherwise no full calls.

## Staged Gates and Controls

**SCGP-0:** zero teacher/new OCR; five strict folds; outer query labels endpoint-only. Both datasets actual ordinary OOF acc/mF1 each `+.050`, every fold positive, all numerical/fit/collapse/Farkas gates. LABEL-ONLY becomes moving comparator if stronger.

**Pilot:** <=128/dataset and design-based gates above; FULL beats label-only, scalar label/error propensity, P4, TextTeacher, fine-cell shuffle, DIRECT-AEXC and STATE-MOMENT.

**Seed-0:** full train-only cache; both datasets dev FULL beats REMOVE, moving LABEL-ONLY, shuffle/noise, strongest proxy/relation, both direct controls `+.010` in both metrics; corruption `{0,.25,.50,.75,1}` monotonically removes gain; select strongest direct control.

**Final:** seeds 0/1/2, ordinary kNN; moving baseline `+.030` accuracy/macro-F1 on both datasets, every sign positive, hierarchical paired CI and Holm tests; significant FULL-minus-REMOVE, SHUFFLE and frozen strongest-direct. Validation/test load no certificate/compiler/target.

## Failure and Novelty Boundary

No common family, selection instability, contamination/cache mutation, scalar/P4/direct/moment parity, projector/Dykstra/PSD/rank/tie/self-exclusion failure, non-certified rank target, SCGP-0/pilot/fit/collapse/Farkas failure, native-head-only gain, or any segment/timestamp/span/localization gold use is terminal. No schema/tolerance/teacher/epoch/scale rescue.

The only claim is:

`label-blind structural reflection -> exact-vote-safe proximal full-bank target -> uniform encoder fit -> ordinary kNN`,

and it is valid only if FULL beats direct use of the identical reflection and standard state moments through final paired seeds.

## Handoff

After READY, first authorize an independent implementation audit and SCGP-0 numerical microbenchmark only. This refinement changes no source and launches no job.
