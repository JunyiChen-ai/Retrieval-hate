# Round 1 Independent Method Review — LB-SCGP

<details open>
<summary>Full raw reviewer response</summary>

## Executive verdict

**Weighted overall: 6.74 / 10**  
**Verdict: REVISE**

LB-SCGP preserves the research anchor unusually well and is genuinely different from ECM’s sample-weighted gradient QP. The MLLM role is label-blind, train-only and removable; only parent-video binary labels are gold; final inference remains unchanged ordinary full-video kNN.

However, four blocking method issues remain:

1. The certificate schema records only proposition/stance *presence states*, not proposition identity, so the compiler cannot truthfully infer “same proposition/core.”
2. The proposed rank-cell solver is not yet a mathematically closed algorithm: rank-cell inequalities and globally coupled boundary pivots are missing.
3. The Farkas audit proves separation only from a finite free-embedding gradient dictionary, not automatically from generic metric learning or realized AdamW behavior.
4. The full SDP/rank-pivot implementation and pilot estimand are not yet computationally or statistically specified enough to execute.

No experiment should be implemented until these are repaired. The direction remains worth refining rather than rethinking.

## Anchor and drift audit

### Preserved

- Endpoint: unchanged full-video train-memory cosine top-20 arithmetic vote.
- Target: MHC-EN and MHC-ZH, seeds 0/1/2, accuracy and macro-F1 each moving baseline `+0.030`.
- Teacher: train-only and strictly blind to label, prediction, margin, correctness, neighbour, loss and gradient.
- Gold boundary: only parent-video binary labels.
- No segment/timestamp/span/localization gold, loss or endpoint.
- No teacher key, concat, reranking, score fusion, router, pseudo-group DRO or test artifact.
- Compiler reads labels only after immutable teacher-cache closure.

### Drift warning

**NONE.**

One terminology defect could cause later drift: the compiler currently calls identical atom-presence patterns “same proposition.” Fix this as structural certificate isomorphism; do not repair it by adding unrestricted proposition text, target names or teacher-selected pairs without a new contamination/novelty audit.

The standalone `refine-logs/lb_scgp/PROBLEM_ANCHOR.md` was absent; this review used the complete immutable anchor embedded in the proposal.

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.5 |
| Method Specificity | 5.8 |
| Contribution Quality | 6.2 |
| Frontier Leverage | 8.0 |
| Feasibility | 4.7 |
| Validation Focus | 7.4 |
| Venue Readiness | 5.5 |
| **Weighted overall** | **6.74** |

## 1. Problem Fidelity — 9.5/10

The proposal directly attacks the required final-memory geometry and does not substitute localization, explanation quality, solver success or an auxiliary head for final classification. It correctly distinguishes previous empirical bounds: SSR/EDCM do not bound a jointly moving full bank, while CTE/SQ did not produce performance results.

Minor concern: SCGP-0’s requirement to correct every wrong inner-bank leave-one-out vote is much stronger than the final problem requires and could terminate a potentially useful route for an unrelated feasibility reason.

## 2. Method Specificity — 5.8/10

### Blocking weakness A: compiler semantics are not identified

The eleven atoms say whether a referent, predicate, binding, stance or exception is present. They contain no proposition identity, target identity or content signature. Therefore two videos with identical atom states need not concern the same proposition—or even related content.

Consequently:

- “same supported proposition/binding core” is unsupported;
- exhaustive “equivalence” equations can join unrelated videos;
- exception reflection may enforce false symmetries across unrelated content;
- the resulting operator may mainly encode label-conditioned structural templates.

**Fix — CRITICAL:** choose one defensible interpretation:

1. Preferably redefine constraints as **structural exception equivariance**, explicitly invariant to proposition identity. Never call it semantic equivalence.
2. Specify exactly why the same exception transformation should induce a common row-profile operator across different content.
3. If content identity is indispensable, introduce only a frozen closed ontology or independently generated label-blind identifier and re-audit P4/SSR overlap and contamination. Do not add free rationale or teacher-selected neighbours.

The exact profile columns, conjunctions, `T_e` permutations, confidence normalization, equation normalization, duplicate handling and sparse-operator dimensions must be enumerated.

### Blocking weakness B: rank-cell program is incomplete

Holding `π_i` fixed makes the vote margin linear only if membership in that rank cell is explicitly constrained. The current SDP omits inequalities such as:

`G[i,π_i(r)] >= G[i,π_i(r+1)]`

and top-20-versus-outsider inequalities. Post-hoc reranking cannot retroactively make a solve an optimization inside the claimed cell.

Additionally, one symmetric Gram entry affects rankings of two rows. Rank-boundary events are globally coupled; independently taking the first lexicographic adjacent swap per query is not a complete or clearly feasible pivot rule.

**Fix — CRITICAL:**

- Define the full global rank-cell polyhedron, including top-20 internal order, twentieth-versus-all-outsiders, stable-ID tie semantics and tolerance.
- Define simultaneous boundary events under symmetry.
- Specify how a pivot preserves PSD, all row-cell constraints, trust regions and exact margins.
- State the convergence claim honestly: local stationary feasible target, not globally nearest target.
- Give FULL’s exact `ell_i`; currently it is explicit only for LABEL-ONLY.
- Freeze `epsilon_vote`, `kappa`, trust radii, fit fraction, refresh frequency and termination/backtracking budgets without outer-fold or dev/test outcome selection.

### SCGP-0 feasibility objective

Forcing every inner-bank error over zero can require near class collapse and is unnecessary for demonstrating `+0.05/+0.05` held-out capacity.

**Fix — IMPORTANT:** use uniform slack with a frozen class-balanced slack budget:

`m_i(G) >= ell_i-xi_i, xi_i>=0`,

where the aggregate budget is chosen before endpoint evaluation and implies sufficient train-side repair capacity. No MLLM-dependent slack or sample weight is allowed. Alternatively justify mathematically why all-error repair remains feasible under the frozen trust/collapse guards.

## 3. Contribution Quality — 6.2/10

The full-bank stopped-target interface is meaningfully different from P4 field prediction, SQ triplet ranking, ECM group-risk gradient surgery and teacher-embedding KD.

Still, the current contribution can be read as “LLM rule extraction + linear Gram constraints + geometry KD.” The Farkas audit does not by itself create novelty, particularly if `A_sem` reduces to weighted pairwise similarities.

**Fix — CRITICAL:** make **exception-reflection row-profile equivariance under exact vote constraints** the sole semantic novelty. Treat ordinary equivalence constraints as optional or remove them if they add generic clustering pressure. The paper thesis should be:

> A label-blind exception algebra defines a higher-order full-bank transformation constraint, solved as an exact-vote-safe target before uniform encoder fitting.

Do not claim general non-equivalence to all metric learning unless the dictionary and proof cover that family.

## 4. Frontier Leverage — 8.0/10

The MLLM role is crisp and appropriate: a constrained semantic certifier, not a feature generator or classifier. Immutable provenance, abstention and structured closure are strong foundation-model-era design choices.

No modernization component is needed. Better formalization and fewer semantic equations would improve the method more than adding another modern module.

## 5. Feasibility — 4.7/10

A dense `N≈600` PSD variable is manageable for repeated eigendecomposition, but a generic SDP with roughly `N^2` variables, potentially exhaustive pair-of-pairs constraints, full rank-order inequalities, repeated cell pivots, primal/dual certificates, five OOF folds and many controls is unlikely to fit the provisional 30–80 GPU-hour estimate if implemented with an interior-point solver.

**Fix — CRITICAL:**

- Specify a matrix-free first-order conic solver—such as ADMM/Dykstra with exact PSD projection—and its dual recovery.
- Compress exhaustive equivalent-pair equations into an algebraically identical contrast/operator basis rather than materializing all pairs.
- Give asymptotic and measured estimates for semantic-operator products, PSD projection, ranking, cell constraints and number of pivots.
- Freeze maximum solver calls and fallback policy.
- Require a synthetic-plus-one-real-fold microbenchmark before any full OOF run.

## 6. Validation Focus — 7.4/10

The staged design is strong: zero-teacher capacity before teacher spending; 128-video governance pilot; seed-0 mechanism gate; final paired seeds; label/error-propensity, P4, TextTeacher, direct-rule and generic metric controls.

Two issues require correction:

1. Pilot sampling is stratified by label, OOF prediction and margin. Coverage/support estimates from this sample are not population estimates without inclusion probabilities and weighting.
2. “Held-out pilot identities” and the actual OOF fitting universe are not fully separated.

**Fix — IMPORTANT:**

- Freeze inclusion probabilities and use design-weighted coverage/support estimates.
- Define pilot certificate-fit and pilot semantic-evaluation folds before calls.
- Ensure no identity’s certificate helps construct the operator used to score that identity’s conditional semantic residual.
- State how partial 128-video certificate coverage defines `W`, anchors and bank columns without silently treating all unprocessed videos as rejected records.

## 7. Venue Readiness — 5.5/10

If the compiler and target solver are repaired, the proposal could become a sharp paper. At present, reviewers would likely challenge false “same proposition” identification; whether the target is merely rule-guided metric learning; solver tractability and convergence; whether abstract target non-equivalence survives encoder fitting; and whether label-conditioned row profiles are a sophisticated label proxy.

These are method blockers, not merely missing experiments.

## Farkas/non-reweighting audit

The proposed sign convention can be valid. For projection `p` of normalized `d` onto cone `C={H alpha: alpha>=0}`, a witness based on `u=(p-d)/||p-d||` can satisfy `H^T u>=0` and `d^T u<0`.

However, the current claim is too broad.

### Required fixes

1. Define columns as **descent displacement directions**, not ambiguously as gradients.
2. Normalize the tangent projection and handle zero/duplicate columns deterministically.
3. State that the certificate proves separation only from the complete registered cone represented by `H`.
4. For `H_rel`, either enumerate the entire registered pair/triplet/SupCon primitive family or provide a valid column-generation separation oracle. A budgeted subset cannot support a claim about generic triplet learning.
5. Run the audit on both abstract target displacement `Z*-Z0` and realized post-fit bank displacement `Z_fit-Z0`.
6. Do not claim that the free-embedding cone certificate proves non-equivalence after AdamW. Actual learned matched controls remain binding evidence at parameter/optimizer level.
7. Report primal projection residual, witness feasibility, separation margin and duality gap with independent recomputation.

If realized displacement falls inside the scalar cone, the method’s executable effect is reducible even when its abstract target is not.

## Factorization and fitting audit

The factor/Procrustes path is conceptually correct because `N<1024`. It still needs:

- exact padded matrix and orthogonal-factor dimensions;
- deterministic handling of repeated singular values;
- reject-on-negative-eigenvalue rule—small positive eigenvalues must not be described as rejected;
- exact train/eval/dropout state during target creation and fitting;
- fit-block placement and refresh schedule;
- target-realization thresholds;
- confirmation that rollback restores model, AdamW moments, scheduler and RNG/data-order state.

Collapse guards should also be imposed on the abstract target, not only after fitting; otherwise solver time can be spent on targets guaranteed to roll back.

## Certificate firewall audit

The input firewall and cache-closure protocol are strong. The main contamination risk is semantic, not byte-level leakage: atoms 1–5 approximate the label definition.

Required controls are correctly included, but strengthen them as follows:

- cross-fit scalar propensity controls only after cache closure;
- preserve the complete certificate’s coverage/missingness/confidence in scalar controls;
- shuffle indivisible certificate records within label × OOF prediction × margin × error-propensity strata;
- report atom-wise label/error/margin predictiveness without choosing atoms or thresholds from those results;
- treat any teacher-schema change after pilot as a new hypothesis, not prompt tuning.

No segment-gold issue was found.

## Simplification opportunities

1. Make exception-reflection the single semantic operator; delete generic equivalence equations unless their independent necessity is demonstrated.
2. Reduce certificate atoms to the minimum needed for proposition closure, direct stance and one common exception family. Eleven atoms plus three exception operators may be excessive before support is known.
3. Keep Farkas separation as an audit under Claim 1, not a supporting contribution.
4. Use one solver family and one global trust/fit configuration. Do not create dataset-specific solver grids.

## Modernization opportunities

**NONE.** The proposal already uses the MLLM in an appropriate modern role. Adding learned compilers, differentiable sorters, routers or larger teachers would weaken the thesis.

## Remaining action items

1. Correct “same proposition” to a formally supported structural relation.
2. Freeze exact certificate compiler columns, conjunctions and reflection matrices.
3. Add complete global rank-cell inequalities and coupled pivot semantics.
4. Specify a feasible matrix-free PSD solver and measured microbenchmark gate.
5. Rework or justify the all-inner-error exact-margin envelope.
6. Define every target/solver/fit hyperparameter selection boundary.
7. Scope and complete the Farkas dictionary; audit realized displacement.
8. Close the pilot sampling/held-out estimand.
9. Add abstract-target collapse constraints and exact rollback state.
10. Re-submit the full revised proposal for the same-reviewer Round 2 evaluation.

**Final verdict: REVISE.** The anchor is preserved and the central route is promising, but the compiler semantics, exact-rank solver and non-equivalence proof are not yet READY.

</details>
