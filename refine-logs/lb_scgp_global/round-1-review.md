# Round 1 Review: LB-SCGP Global

## Parsed Header

| Field | Value |
|---|---:|
| Problem Fidelity | 8.5 |
| Method Specificity | 6.5 |
| Contribution Quality | 7.0 |
| Frontier Leverage | 7.5 |
| Feasibility | 6.0 |
| Validation Focus | 8.0 |
| Venue Readiness | 6.5 |
| Overall | 7.1 |
| Verdict | REVISE |

Drift Warning: NONE

<details open>
<summary>Full raw review</summary>

# Round 1 Raw Review

## Overall Assessment

This is a faithful and unusually disciplined pivot away from the retired local rank-cell route. The proposal preserves the parent-video-only gold boundary, keeps the MLLM train-only and label-blind, avoids test-time teacher/head/rerank paths, caps the paper around one dominant global-geometry mechanism, and correctly refuses to treat inherited v7 evidence as validation of the new global target.

The main weakness is not contribution sprawl. The main weakness is that several mathematical interfaces are still not executable as written. In particular, the target Gram can have rank larger than the encoder dimension, the replica moment operators are dimensionally and coordinate-wise underspecified, the robust interval construction is likely valid but vacuous under the stated row trust scale, and the Dykstra certificate needs an explicit normal-cone/KKT certificate rather than only replayed residuals. These are method-level issues, not experiment-list issues.

Verdict: REVISE. The direction is promising, anchored, and appropriately modern, but it is not READY until the Gram target is made encoder-realizable and the operator/solver certificates are mathematically closed.

## Scores

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 8.5 | The proposal preserves the immutable endpoint, parent-video-only gold, train-only label-blind MLLM cache, ordinary test kNN path, and retirement of local v7/v8. Minor risk remains because the schema contains stance-like atoms and the projection includes robust pairwise rank edges, but these are currently framed as certificates/safety constraints rather than gold or MLLM-selected pairs. |
| Method Specificity | 6.5 | The pipeline is concrete, but important formulas are not yet executable: cross-replica moment dimensions are not well defined, `vech` is misused for non-symmetric cross-basis moments, rank(G*) may exceed encoder dimension d, and the optimality certificate is underspecified. |
| Contribution Quality | 7.0 | One dominant contribution is visible: certificate-to-global-proximal-Gram geometry. It is more focused than prior local rank-cell work. The claim is weakened by the current realizability gap and by the possibility that the MLLM contribution reduces to a deterministic certificate-kernel regularizer unless the direct/scalar controls are decisive. |
| Frontier Leverage | 7.5 | The frozen MLLM certificate role is appropriate and does not need trendier components. The proposal uses the MLLM as a train-only structural sensor rather than as a classifier or rationale teacher. The remaining issue is proving the role is not replaceable by coarse scalar difficulty or direct certificate features. |
| Feasibility | 6.0 | Full N x N PSD projection, Dykstra over PSD/SOC/halfspace/affine sets, rank-d factorization, and uniform encoder fit are plausible only after sharper dimensional gates and solver certificates are specified. Robust top20 coverage may be near zero with the current interval scale. |
| Validation Focus | 8.0 | The three blocks are appropriately claim-driven: G0/replay/isolation, attribution controls, final paired performance. The plan avoids benchmark bloat and includes the required REMOVE/SHUFFLE/NOISE/direct/scalar controls. |
| Venue Readiness | 6.5 | The paper shape is promising, but a top-venue method cannot leave the rank-realizability and operator definitions unresolved. Once fixed, the framing could become a clean method-first contribution. |

OVERALL SCORE = 0.15*8.5 + 0.25*6.5 + 0.25*7.0 + 0.15*7.5 + 0.10*6.0 + 0.05*8.0 + 0.05*6.5 = 7.1.

## Required Fixes for Dimensions Below 7

### Method Specificity - 6.5

Specific weakness: the central mathematical interface is not fully specified. `M_{U,V}(G)=vech(U^T(G-I)V/N)` is only valid as `vech` when the result is square and symmetric. For `U=Q^(r)`, `V=Q^(s)`, ranks may differ and coordinates from separately computed SVD bases are not aligned. Also, the factorization step assumes `rank(G*) <= d` but the convex PSD projection does not enforce that.

Concrete method-level fix: define all replica stability moments in one common coordinate system, preferably the consensus basis `Q`, or use subspace-invariant projection matrices and `vec`, not `vech`, for cross-basis objects. Add an encoder-realizability contract: either require `rank_eps(G*) <= d` as a hard certification gate, or explicitly replace `G*` by a declared rank-d realizable target `G_fit=Z*Z*^T` and make all claims about `G_fit`, not the unconstrained `G*`. If rank truncation is used, it must be part of the method with residual thresholds and no exact-reconstruction claim.

Priority: CRITICAL.

### Feasibility - 6.0

Specific weakness: the proposed robust intervals are likely too conservative to yield non-vacuous top20 safety constraints. With `rho_row = 0.05 sqrt(N-1)`, the coordinate-wise interval `G0_qj +/- rho_row` can be extremely wide for realistic train-bank sizes, often wider than the entire correlation range. In that case, robust edges and vote preservation constraints may vanish. The solver side is also heavy: full-bank PSD projection plus Dykstra corrections over many sets may be slow or numerically delicate without a dual certificate.

Concrete method-level fix: add a coordinate trust bound such as `|G_ij-G0_ij| <= rho_coord` or compute per-edge min/max bounds by solving the projection-derived bound problem before declaring robust edges. Report robust-query coverage as a G0 gate and treat low coverage as "no safety claim", not as failure of the geometry method. For the solver, define an explicit H-metric KKT/normal-cone certificate with dual residuals for affine, box, SOC, PSD, and halfspace constraints, or use an independently replayable conic-QP certificate for smaller fixtures.

Priority: CRITICAL.

### Venue Readiness - 6.5

Specific weakness: the story is focused, but the current manuscript would be attacked on exact realizability and on whether the MLLM is doing anything beyond generating a hand-engineered structural kernel. The novelty claim depends on the target being a valid encoder target and on controls showing that direct certificate features/scalar difficulty do not match it.

Concrete method-level fix: narrow the claim to "closed train-only certificate cache defines a replayable, encoder-realizable global target" and make the rank-d target interface exact. Move robust vote safety to a subordinate diagnostic unless coverage is meaningful. Keep only two claims: executable global geometry and final ordinary-kNN performance/attribution.

Priority: IMPORTANT.

## Formula and Interface Audit

1. Cross-basis moment definition has a mathematical error. `vech` is not valid for `U^T(G-I)V` when `U` and `V` are different bases or have different ranks. Use `vec` for rectangular cross moments, or avoid cross moments by evaluating all replicas in a shared basis.

2. Separately computed SVD bases are not coordinate-aligned. Even with deterministic signs, rotations within close or repeated singular subspaces can change coordinates. Comparing "corresponding" coordinates across `Q^(r)` and `Q^(s)` is not invariant. Fix by anchoring all replica moments in consensus `Q`, using projection matrices `QQ^T`, or adding a deterministic Procrustes alignment between replica bases before moment comparison.

3. The factorization step is dimensionally incomplete. `G*` can have rank up to N, while the encoder target must live in `R^d`. Zero-padding handles `rank(G*) < d`, but there is no valid step for `rank(G*) > d`. This is the most important mathematical blocker.

4. The statement "verify ||Z*Z*^T-G*|| <= 1e-6" is impossible unless the target rank is at most d. If the method keeps the convex projection, add a fail-closed `rank_eps(G*) <= d` gate. If that gate is too strict, define a rank-d approximation target and stop claiming exact reconstruction of `G*`.

5. The projection is convex as a soft-constraint product-space problem, but the wording should distinguish hard constraints from penalized slacks. `A_reg`, `A_eq`, and `A_band` with free slacks are regularized structural preferences, not exact certificate satisfaction unless slack budgets or hard residual caps are added.

6. `K_C = normalize_corr(C C^T + alpha I_missing)` is underspecified. Define `I_missing` exactly, define what happens when rows are all unresolved, and state whether the missingness indicator columns already make `alpha I_missing` redundant.

7. Robust interval construction from row L2 trust is formally conservative but likely vacuous. The proposal should either add coordinate-level trust or explicitly compute tighter per-edge intervals. Ambiguous edges are correctly fail-open for constraints and fail-closed for claims.

8. Dykstra convergence and replay traces are not by themselves an optimality certificate. The independent verifier should certify primal feasibility plus an H-metric normal-cone residual or equivalent KKT/dual certificate. The variational inequality residual must be precisely defined.

9. The product-space norm with lambda weights is acceptable, but every projector must be implemented in the H-metric, not silently in Euclidean coordinates. The solver spec should state this.

10. Vote-margin constraints use parent labels after cache closure. This is allowed by the anchor, but the proposal must keep them as train-only preservation constraints, not as a new label-driven pairwise metric-learning objective.

## Gold-Boundary Audit

The gold boundary is mostly preserved. The only gold label is `parent_video_binary_label`, and the proposal states that labels first enter only after cache sealing, inside the deterministic compiler and final kNN evaluation. Validation/test do not load MLLM records, target banks, teacher outputs, heads, or rerankers.

The main boundary risk is semantic naming. Fields such as `speaker_source_endorsement`, `quotation_condemnation_reportage_exception`, and `satire_reclaimed_or_counter_speech_exception` are stance/mechanism-like certificate atoms. They can remain only if the proposal keeps repeating and enforcing that they are noisy label-blind structural certificates, not gold stance/mechanism/rationale annotations, not pseudo-groups, not evaluation targets, and not selection keys.

The certificate aggregate operators avoid MLLM-selected pairs, sample weights, and pseudo-groups as written. However, robust rank edges are selected pairwise constraints from G0. They are not MLLM semantic pair supervision, but they should remain safety/trust constraints, not the central contribution or an implicit pairwise training route.

## Complexity-Cap Audit

The cap is mostly respected.

- New components: two, if the cache/compiler and convex target/uniform fit are treated as the two components.
- Claims: two, with the G0 executable-geometry claim and final performance/attribution claim.
- Core experiment blocks: three.
- No new head, teacher, reranker, router, segment model, sample weighting, key selection, pair/triplet/SupCon loss, or local v8 stationarity path is introduced.

Complexity risk remains in the solver/constraint system rather than in model architecture. The method may feel overbuilt if `A_reg`, `A_eq`, `A_band`, robust edge constraints, vote constraints, row trust, class trust, factorization gates, and multiple controls are all presented as equally central. The paper should make one thing central: encoder-realizable global proximal certificate geometry.

## Evidence-Status Audit

The proposal correctly states that inherited evidence supports only supervision isolation, endpoint definitions, replay/hash discipline, PSD/unit-diagonal machinery, and the retirement of local rank-cell stationarity. It does not falsely claim that v7 validates the new global pivot.

The new global target, restricted schema, real-fold G0, teacher-cache gate, uniform fit, and final MHC-EN/MHC-ZH performance remain unvalidated. That is acceptable for a proposal, but the text should keep calling them planned gates rather than evidence.

Local rank-cell v7 is retired and no v8/local stationarity path has crept back in. The proposal uses robust top20 intervals only as safety diagnostics/constraints, not as a revival of local stationarity.

## Explicit Audit Checklist

- Restricted label-blind MLLM cache: mostly correct. Parent labels first enter after cache sealing. Add a hard audit that schema atom names cannot be used as stance/mechanism gold.
- Certificate aggregate operators: mostly avoid pseudo-groups, weighting, selected MLLM pairs, and semantic pair supervision. The basis/operator definitions need dimensional repair.
- One closed convex projection: conceptually yes, but the soft-slack formulation and rank-realizability gap must be clarified. The projection is convex; the encoder-realizable target is not guaranteed.
- Dykstra/proximal solver: plausible, not yet sufficient. Needs explicit H-metric KKT/normal-cone certificate.
- Robust intervals: conservative and fail-open/fail-closed logic is correct, but the stated row radius likely makes them vacuous.
- Dimensional validity of G*: not valid until rank(G*) <= d or a rank-d target interface is specified.
- MLLM replacement risk: acknowledged by controls. The claim should remain conditional on beating direct certificate-feature and scalar-propensity controls.
- Caps: <=2 new components, <=2 claims, <=3 core experiment blocks are satisfied.
- Gates: new G0, real-fold gate, teacher-cache gate, and final performance gate are present.
- Final protocol: REMOVE/SHUFFLE/NOISE/direct attribution, strongest same-protocol comparator, MHC-EN/ZH seeds 0/1/2, both metrics +0.030, all deltas positive, hierarchical bootstrap lower >0, Holm are present.
- Novelty boundaries: credible relative to pseudo-groups/reweighting, semantic pair supervision, privileged distillation, and metric learning, subject to fixing the rank/operator issues.
- Inherited evidence: not overclaimed.
- Local v7/v8: v7 retired, no v8 path detected.

## Simplification Opportunities

1. Use one common consensus basis for all structural and replica-stability moments. This deletes cross-basis coordinate matching and removes a real mathematical failure mode.

2. Make robust rank/vote constraints diagnostic or gate-only in the first G0 unless coordinate-bound coverage is non-vacuous. This reduces selected-pair optics and keeps the paper centered on global certificate geometry.

3. Merge `A_eq` and `A_band` into a single cache-stability regularizer or gate if both are not needed. The method should not look like a list of constraint families.

## Modernization Opportunities

NONE. The frozen MLLM-as-certificate-sensor role is already the right modern primitive under the anchor. Adding a reasoning teacher, RL, a learned graph module, or a test-time agent would create drift.

## Drift Warning

NONE.

The proposal currently solves the anchored problem. The only drift risks are future revisions that treat stance/mechanism-like certificate atoms as gold, use robust rank edges as selected-pair metric learning, or revive local rank-cell stationarity under a new name.

## Remaining Action Items Ranked by Priority

1. CRITICAL: Fix encoder realizability. Specify what happens when `rank(G*) > d`, and ensure the final target fitted by the encoder is exactly defined in `R^d`.

2. CRITICAL: Repair the structural moment interface. Replace invalid cross-basis `vech` with invariant/common-basis operators and define replica ranks/alignment.

3. CRITICAL: Make robust intervals non-vacuous or explicitly diagnostic. Add coordinate trust or exact per-edge bound computation, and report robust-query coverage.

4. IMPORTANT: Add an explicit solver optimality certificate: H-metric normal-cone/KKT residuals, not just Dykstra replay traces.

5. IMPORTANT: Clarify soft versus hard structural constraints. Slacked equality/band systems should be described as regularized preferences unless bounded by certified residual caps.

6. IMPORTANT: Tighten the MLLM novelty claim around the direct/scalar controls. State that if DIRECT-CERT-FEATURE or SCALAR-PROPENSITY matches FULL, the MLLM-global-geometry claim fails.

7. MINOR: Rename or annotate stance-like schema fields as noisy structural observables to reduce gold-boundary confusion.

## Verdict

REVISE.

The proposal is anchored, focused, and has the right high-level contribution shape. It is not READY because the core convex-Gram-to-encoder interface is not yet mathematically closed, and because the solver/operator certificates need enough precision that another engineer can implement and independently verify them without interpreting intent.

</details>
