# Refinement Report

**Problem:** meaningful and novel MLLM integration with substantial final ordinary-kNN gains.  
**Initial approach:** label-blind certificate -> post-cache label compiler -> full-bank Gram target -> shared encoder.  
**Date:** 2026-07-11  
**Rounds:** 4 / 5  
**Final score:** 9.15 / 10  
**Final verdict:** READY for implementation audit/microbenchmark, not a performance result.

## Problem Anchor

Canonical verbatim anchor: `refine-logs/lb_scgp/PROBLEM_ANCHOR.md`.

## Output Files

- Review summary: `refine-logs/lb_scgp/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/lb_scgp/FINAL_PROPOSAL.md`
- Score history: `refine-logs/lb_scgp/score-history.md`
- Raw reviews: `round-1-review.md` through `round-4-review.md`
- Full anchored revisions: `round-1-refinement.md` through `round-3-refinement.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.5 | 5.8 | 6.2 | 8.0 | 4.7 | 7.4 | 5.5 | 6.74 | REVISE |
| 2 | 9.8 | 8.2 | 6.9 | 8.2 | 6.2 | 8.5 | 6.8 | 7.86 | REVISE |
| 3 | 9.8 | 8.8 | 8.5 | 8.5 | 7.5 | 9.0 | 8.3 | 8.69 | REVISE |
| 4 | 9.8 | 9.3 | 9.0 | 9.0 | 8.3 | 9.5 | 9.0 | 9.15 | READY |

## Method Evolution Highlights

1. Deleted unsupported proposition identity and all generic equivalence equations; retained one support-selected structural reflection.
2. Replaced an underspecified SDP/pivot story with exact fixed-cell Dykstra projections and fail-closed local rank-cell traversal.
3. Added direct identical-moment controls and final three-seed attribution, so proximal realization—not generic MLLM semantics—must cause the gain.
4. Scoped cone separation to registered primitives and required the realized encoder displacement to remain outside them.

## Pushback / Drift Log

| Round | Potential change | Author response | Outcome |
|---:|---|---|---|
| 1 | add proposition/content identity | rejected to avoid free text, key selection and contamination; reframed as content-invariant structure | accepted |
| 2 | add differentiable sorter/learned compiler | rejected as needless complexity | accepted |
| 3 | treat direct controls as seed-0 only | strengthened instead: strongest direct runs final three seeds | accepted |

## Remaining Weaknesses

Specification READY is not experimental success. No LB-SCGP code, SLURM job, teacher cache or new metric exists. The next authorized sequence is independent implementation audit -> sealed synthetic plus one-real-fold numerical microbenchmark -> only if projected total is `<160 GPU-hours`, zero-teacher SCGP-0. Teacher calls remain zero until both-dataset SCGP-0 actual OOF `+0.05 accuracy/+0.05 macro-F1` and all numerical/non-equivalence gates pass.

## Raw Reviewer Responses

The four verbatim responses follow in the referenced round files and are also appended below for checkpoint-local completeness.


---


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


---


<details open>
<summary>Full raw reviewer response</summary>

## Executive verdict

**Weighted overall: 7.86 / 10**  
**Verdict: REVISE**  
**Problem Anchor: preserved**  
**Drift warning: NONE**

Round 1 的主要工程与数学缺口已大幅收敛：

- 删除了不成立的 “same proposition” 假设；
- exception compiler 已成为明确、至多 48 行的高阶全库算子；
- global rank-cell、完整 outsider inequalities、对称耦合 pivot 和 exact evaluator verification 已写明；
- LABEL-ONLY/FULL 共用 class-balanced uniform slack；
- Farkas 结论已正确收窄到注册 cones，并覆盖 abstract/realized displacement；
- factor、rollback、pilot sampling、HT estimand 和 test-clean 路径基本闭合；
- 没有 segment/timestamp/span/localization gold 漂移。

但目前仍不能 READY。核心剩余问题已从“接口不成立”转为两点：**科学区别是否超出 label×certificate-state conditional moment alignment**，以及 **solver/oracle 是否真正可实现并可核验**。

## Scores

| Dimension | Round 2 |
|---|---:|
| Problem Fidelity | 9.8 |
| Method Specificity | 8.2 |
| Contribution Quality | 6.9 |
| Frontier Leverage | 8.2 |
| Feasibility | 6.2 |
| Validation Focus | 8.5 |
| Venue Readiness | 6.8 |
| **Weighted overall** | **7.86** |

## Blocker 1：exception reflection 仍可能只是条件矩匹配

当前算子本质上先形成 `(state D/Q/C/R, video label)` 八个 cells，再要求这些 cells 的平均 Gram row profile满足固定置换：

`rbar_(e,c)=T_e rbar_(D,c)`.

这已经不是 sample reweighting、GroupDRO 或普通 triplet，但最近的等价解释是：

> MLLM 定义离散 pseudo-state，方法对 label×state groups 做 conditional prototype/moment alignment，再通过 target KD 训练 encoder。

Farkas 排除 pair/triplet cone，不能排除这种同状态、同全库统计量的直接 moment objective。

**Required fix — CRITICAL**

加入完全匹配的两个 binding controls：

1. `DIRECT-AEXC`：直接在 refreshed full bank 上优化同一个 `||A_exc vec(G_theta)||²`，不用 proximal target；
2. `STATE-MOMENT/PROTOTYPE`：用相同 D/Q/C/R×label cells、confidence、coverage和步骤预算做标准 conditional centroid/MMD/Gram-moment alignment。

FULL 必须在两库 actual OOF acc/mF1、exact-vote repair、realized semantic residual上显著胜二者。若匹配，则 novelty 只能是工程性的 target reformulation，不足以支撑当前 dominant claim。

同时应把论文表述从宽泛的 “exception algebra” 收紧为：

> exact-vote-safe proximal realization of a label-blind structural-reflection moment constraint.

## Blocker 2：operator-splitting solver尚未唯一闭合

“scaled ADMM with cyclic Dykstra projections”仍混合了两个不同算法描述。当前缺少：

- 完整 primal variables `(G,xi)`；
- 每个 split set 的变量复制；
- consensus/dual update equations；
- slack-budget projection；
- stopping residual定义；
- Dykstra corrections 与最终 dual certificate之间的精确关系。

**Required fix — CRITICAL**

只保留一个可核验算法：product-space Dykstra projection；或 consensus ADMM，每个 convex set 一个明确 prox。

给出逐步伪代码和每个 projection 的闭式/数值实现。微基准必须测：每 sweep 时间；PSD eigendecomposition 时间；rank-halfspace数量；semantic operator/adjoint parity；sweeps/pivots分布；primal feasibility与independent float64复算。

在真实一折未证明 `<160 GPU-hours` 总预算前，feasibility 不能达到 7。

## Important issue 3：triplet separation oracle需修正

当前称 active margin-triplet direction 可按 anchor 的 positive/negative scan 分离。若 hinge activity 依赖 `(a,p,n)` 的联合 margin，该最小化一般不能分别选择 `p` 和 `n`。

**Fix — IMPORTANT**

二选一：

- 将 cone 定义为所有 label-legal algebraic triplet descent directions，不受当前 hinge active mask限制；此时可给出可验证的分离结构；
- 或实现 joint blocked `(p,n)` scan，并在小 bank 上 brute-force parity。

还需明确 pair cone 只能含 label-legal attraction/repulsion方向；若同一 pair 的正反方向都进入 cone，cone可能退化为过大的线性空间。

## Important issue 4：rank-boundary终止语义

全局 cell 与对称 pivot已基本正确，但当独立 boundary orientations 超过 8 而停止时，不能称为邻域局部 stationary，因为尚有未探索的相邻 cells。

**Fix — IMPORTANT**

将状态精确区分：

- `LOCAL_STATIONARY_CERTIFIED`：所有相邻可行 orientations 已检查；
- `BOUNDED_SEARCH_FEASIBLE`：因 orientation/pivot预算停止；
- `REMOVE_FALLBACK`：无可行 target。

只有第一类可用于方法机制 claim；第二类最多进入 feasibility reporting，或者必须预注册其是否允许训练。

最终 target 还必须确认 canonical tie order 与当前 cell一致，不可停在未处理的共享边界上。

## SCGP-0 interpretation

Uniform class-balanced slack修复了“必须纠正所有训练错误”的问题。它现在只保证每类 aggregate frozen vote deficit 至少减少 80%，不保证特定数量的训练错误被纠正。因此：

- `beta=.20` 是 action-strength约束；
- `+0.05/+0.05` strict OOF endpoint 才是 capacity evidence；
- 不得把 slack feasibility写成分类 headroom证明。

还需明确 frozen geometry、LABEL-ONLY和所有 learned controls从相同 fold checkpoint、初始化、batch order和checkpoint-selection rule开始。若 LABEL-ONLY通过，它立即成为 moving non-MLLM comparator，这一点正确。

## Farkas audit

Round 1 的主要问题已关闭：sign convention可成立；columns现在是 descent directions；claim已收窄到 registered cones；abstract与realized displacement都审；不再把 free-embedding separation误写成 AdamW保证；actual learned controls仍binding。

剩余要求：修正 triplet oracle；对 dictionary completeness生成machine-checkable manifest；分别报告 abstract/realized primal residual、dual feasibility、separation和gap；若 target displacement通过但 realized displacement进入任一cone，必须STOP——当前已正确规定。

## Factor、fit与rollback

该部分已基本闭合。PSD factor维度、zero padding、Procrustes、负特征值处理、fit fraction、完整 optimizer/scheduler/RNG rollback和REMOVE replay都明确。

小修：

- “deterministic LAPACK basis”跨平台并不天然确定；应固定CPU backend/version/thread count，并对重复 singular subspace采用显式canonical basis rule。
- abstract target也已有collapse guards，关闭了Round 1 blocker。
- 明确一次“block”是完整的target-refresh后一个epoch，rollback/replay从该epoch起点执行。

## Pilot estimand

HT inclusion probabilities、A/B cross-fit、unsampled≠rejected、partial coverage和outer-memory-only证书使用已基本成立。

仍需：

- 使用与分层不等概率抽样相容的design-based replicate/bootstrap，而不是普通paired-ID bootstrap；
- 冻结每个half最低state×label ESS，否则reflection residual可能不可定义；
- 说明partial-pilot OOF `+0.01` 是哪个seed/checkpoint-selection protocol；
- teacher QC只能验证schema/state appropriateness，不能变成训练标签。当前无segment-gold问题。

## Simplification opportunities

1. Pilot后通过纯support规则最多保留一个两库共同exception family；避免三个reflection并行稀释单一机制。
2. 删除泛化的 `DARTVAE-style` 名称，改成精确的 `DIRECT-AEXC` matched control。
3. 将 Farkas 保持为机制审计，不扩展成第二个理论贡献。
4. Solver只保留一种operator-splitting算法。

## Modernization opportunities

**NONE.** 不需要 differentiable sorter、learned compiler、larger teacher或actual-update projection。

## Remaining action items

1. 增加 `DIRECT-AEXC` 与 matched state-moment/prototype controls。
2. 把 ADMM/Dykstra 二选一并写出完整 solver pseudocode/projections。
3. 修正 triplet column-generation oracle。
4. 区分 certified local stationarity 与 bounded-search stop。
5. 冻结设计型pilot bootstrap/ESS/seed protocol。
6. 固定重复 singular subspace 的真正 deterministic rule。
7. 用同一 reviewer 做 Round 3 复评。

**Final verdict: REVISE.** 方向、监督边界和主体接口均保留；剩余 blocker 是贡献归因与可执行 solver，而不是 anchor drift。

</details>


---


<details open>
<summary>Full raw reviewer response</summary>

## Executive verdict

**Weighted overall: 8.69 / 10**  
**Verdict: REVISE**  
**Anchor status: preserved**  
**Drift warning: NONE**

Round 2 blockers基本关闭。当前方案已成为一个聚焦、可证伪且明显区别于 ECM/SQ/P4 的方法：

- 唯一语义机制收窄为一个 support-selected structural exception reflection；
- `DIRECT-AEXC` 与 `STATE-MOMENT` 正面覆盖了“只是条件矩匹配”的最近解释；
- solver 已统一为 product-space Dykstra；
- 只有 `LOCAL_STATIONARY_CERTIFIED` target 可训练；
- triplet cone、realized displacement、rollback、pilot design inference均显著加强；
- teacher label-blind、cache-before-label、test-clean、no-segment-gold全部保持。

尚余三个窄但真实的 specification blockers，因此不能按 `overall>=9` 规则给 READY。

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.8 |
| Method Specificity | 8.8 |
| Contribution Quality | 8.5 |
| Frontier Leverage | 8.5 |
| Feasibility | 7.5 |
| Validation Focus | 9.0 |
| Venue Readiness | 8.3 |
| **Weighted overall** | **8.69** |

## Blocker 1：两个 trust-set projection 尚未证明是欧氏投影

Dykstra 只有在每个步骤确实计算对应闭凸集合的 Euclidean projection 时才有当前收敛含义。

- 单行 trust ball 若在完整非对称 ambient `G` 上定义，可以作 row radial projection；
- 但最终还与 symmetry 集合相交，而且“symmetric row coefficient”会同时影响 `G_ij/G_ji`，其范数缩放必须明确；
- class-mean-row trust 是线性映射 `L_cG` 的二范数球。除非证明 `L_cL_c^T=alpha I`，其投影不是普通 radial clipping。

Semantic ellipsoid的思路正确，但应把公式完整写出：

`g(mu)=y-mu A^T(I+mu AA^T)^(-1)Ay`,

由标量 `mu>=0` 解 `||Ag(mu)||=r`。当前 “16-dimensional dual + unique multiplier” 尚略含糊。

**Required fix — CRITICAL**

为 row trust、class-mean trust 和 semantic set 分别给出：ambient inner product；linear operator `L`；exact KKT projection；scalar root equation；synthetic finite-difference/projection optimality parity。

若 class-mean set 无简洁闭式，就像 semantic set 一样通过小型 `LL^T` dual solve，不要称为 radial projection。

## Blocker 2：matched direct controls 的执行和 final attribution尚未完全冻结

`DIRECT-AEXC` 是正确且必要的最近控制，但仍需明确：

1. coefficient 是每 fold解析匹配，还是一个 EN/ZH共享全局值；
2. `eval()` full-bank graph、dropout/BatchNorm状态和随后恢复 `train()` 的确切顺序；
3. 每个 fit batch都重建 complete differentiable bank，还是每个 refresh只算一次；
4. first-batch norm matching 后，整个epoch的累计辅助梯度强度可能严重漂移；
5. FULL、DIRECT-AEXC、STATE-MOMENT 的计算量不同，至少应报告而不能写成 compute-matched。

最重要的是，正文开头说 direct controls 在 pilot/seed-0/final 都 binding，但 final gate只明确要求 FULL 显著胜 REMOVE/SHUFFLE。

**Required fix — CRITICAL**

- 冻结一个无 outcome 的解析 coefficient protocol；
- 记录整epoch累计 auxiliary-gradient norm和wall-time，但不得二次调参；
- seed-0选出 `max(DIRECT-AEXC, STATE-MOMENT)`；
- final三 seeds必须运行该最强直接控制；
- FULL-minus-strongest-direct 在两库两指标上也须3/3同号且paired CI lower bound `>0`。否则只证明MLLM有用，未证明 proximal realization 是贡献来源。

## Blocker 3：pilot family selection与design inference需最后闭合

`e*=argmax min ESS` 不使用性能结果，因此不存在普通 outcome tuning；但它仍由同一pilot sample的随机state support选择，而随后又在该样本上估计reflection effect。

**Required fix — IMPORTANT**

二选一：采用固定 `Q<C<R` 中第一个在两库两half均通过ESS门的family；或在每个Rao–Wu replicate内重复support-only family selection，并对selection failure作预注册处理。

同时明确 Rao–Wu replicate同时重建 HT cell totals；selected family；A/B reference profiles；held-out reflection residual；correction-direction statistic。这样 lower bound才覆盖完整pilot procedure，而不是把随机选出的family当固定。

## Dykstra与rank-cell审计

除上述 projector 公式外，本轮已基本闭合：

- primal `x=(G,xi)` 与 projection objective明确；
- symmetry、PSD、box、rank、vote、slack、semantic集合分离；
- cell内部完整19条order及20th-vs-outsider inequalities；
- 同一对称G上的全局boundary event；
- 超过orientation/pivot预算映射REMOVE；
- 只有全部相邻orientation检查完成才称 `LOCAL_STATIONARY_CERTIFIED`；
- unresolved tie不能训练；
- independent float64 exact evaluator仍binding。

该表述诚实地只主张 explored union上的local target，不再暗示global nearest solution。

## Farkas审计

Round 2 blocker已关闭：pair cone只有label-legal单方向；triplet oracle joint扫描 `(p,n)` 并包含hinge activity；small-bank brute-force parity；machine-readable universe/active-mask/column manifest；abstract和realized displacement均审；claim明确限于registered cones；learned AdamW controls仍binding。

Implementation plan仍应验证column-generation终止时的全宇宙minimum witness inner product，而不能只记录最后新增列。除此之外不再构成method blocker。

## Structural exception compiler

当前compiler不再冒充content identity。它明确检验一个更窄的科学假设：direct-speaker reading变成closed exception时，label-conditioned full-bank state profile应发生固定D↔e反射。

这仍是label-conditioned moment structure，但 `DIRECT-AEXC` 和 `STATE-MOMENT` 已把该最近解释变成binding controls。因此当前 novelty边界是可守的，前提是 final也显著胜最强direct control。

信心只进入operator估计，不进入encoder sample loss；这不是ECM式sample/group risk reweighting。

## Factor、fit与rollback

此前 blocker已关闭：factor维度、repeated subspace canonical basis、backend freeze、abstract/realized collapse、block定义、完整rollback及REMOVE replay均明确。

## Supervision与test-clean审计

**PASS。** 唯一gold始终是parent-video binary label；frames/ASR/OCR只是整视频输入；teacher看不到label/prediction/margin/error/neighbour/loss；post-cache compiler才可读train labels；certificate states始终是weak/privileged pseudo-signals；schema/compiler/loss均无segment/timestamp/span/localization对象；validation/test不加载certificate/compiler/target；inference保持ordinary full-video train-memory kNN。

## Simplification opportunities

1. 保持最多一个exception family；不要在后续因support不足合并多个family。
2. final仅携带seed-0最强的一个direct control，而非两个都跑三seed。
3. Farkas继续作为审计，不扩写成通用理论贡献。
4. Dykstra projector统一使用同一种“小型dual root solve”表述，减少特殊情况。

## Modernization opportunities

**NONE。**

## Minimal remaining action items

1. 写出row/class-mean/semantic集合的exact Euclidean projection与parity test。
2. 冻结DIRECT-AEXC/STATE-MOMENT coefficient、mode、累计强度和final strongest-control gate。
3. 让Rao–Wu inference包含support-only family selection。
4. 同一reviewer做最后一轮复评。

**Final verdict: REVISE。** 方法本体已接近READY；剩余问题是三个可局部修正的执行/归因定义，不需要改方向、增加模块或放宽目标。

</details>


---


<details open>
<summary>Full raw reviewer response</summary>

## Executive verdict

**Weighted overall: 9.15 / 10**  
**Verdict: READY**  
**Problem Anchor: preserved**  
**Drift warning: NONE**  
**Unresolved method blocker: NONE**

Round 3 的三个剩余 blocker 均已关闭。LB-SCGP 现在已达到方法 specification READY：这表示可以进入独立实现审计与 SCGP-0 numerical microbenchmark，不表示已有性能提升，更不表示最终目标已经实现。

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.8 |
| Method Specificity | 9.3 |
| Contribution Quality | 9.0 |
| Frontier Leverage | 9.0 |
| Feasibility | 8.3 |
| Validation Focus | 9.5 |
| Venue Readiness | 9.0 |
| **Weighted overall** | **9.15** |

## Anchor / supervision audit

**PASS。**

- 最终 endpoint 始终是 unchanged ordinary full-video train-memory kNN。
- 最终门槛仍是 MHC-EN/MHC-ZH、seeds 0/1/2、accuracy/macro-F1 各 moving baseline `+0.030`。
- 唯一 gold 是 parent-video binary label。
- Teacher 严格 label-blind，不见 prediction、margin、error、correctness、neighbor、loss 或 gradient。
- Teacher cache 先以 Merkle root 关闭，compiler 后读 train labels。
- Certificate atoms始终是 weak/privileged pseudo-signals。
- 不存在 segment/timestamp/span/localization gold、loss、weight或endpoint。
- Validation/test 不加载 certificate、compiler、`G*` 或 `Z*`。
- FULL 无 teacher key、direct rule loss、sample/group reweight、concat、router、rerank或test-time MLLM。

## Exact projector audit

Round 3 blocker已关闭。

在完整实矩阵 Frobenius ambient space中：

- row extraction满足 `L_iL_i*=I`，radial formula确为preimage-ball Euclidean projection；
- class-mean operator满足 `L_cL_c*=(1/n_c)I`，给出的缩放公式正确；
- semantic set使用标准KKT形式 `g(mu)=y-mu A^T(I+mu AA^T)^(-1)Ay` 并通过单调标量root求解；
- `r=0` 使用 Moore–Penrose nullspace projection；
- symmetry作为独立affine set，与row projector在Dykstra交集中协调，定义合法；
- KKT stationarity、complementarity、idempotence、variational inequality、finite difference与dense reference parity均成为binding tests。

Product-space Dykstra现在是唯一solver，且每个步骤是真实闭凸集合的Euclidean projection，因此对固定rank cell的projection/convergence claim成立。Independent float64 verifier仍然binding。

Implementation audit应确认PSD步骤显式对输入取对称部分，但这是代码审计项，不再是method blocker。

## Rank-cell与local target audit

**PASS。** 完整19条top-20内部order约束；20th-vs-all-outsider约束；self exclusion；同一对称 `G` 上的全局coupled boundary；canonical-ID tie；完整orientation enumeration；超过orientation/pivot预算或未处理tie统一映射REMOVE；只有 `LOCAL_STATIONARY_CERTIFIED` target可训练。

方案只声明已探索相邻cells上的numerically certified local target，没有暗示global nearest optimum，表述诚实。

## Direct-control attribution audit

Round 3 blocker已关闭。

- `DIRECT-AEXC`使用与FULL完全相同的 `A_exc`；
- `STATE-MOMENT`覆盖最近的conditional moment/prototype解释；
- 每个aux step的eval-mode、dropout/BatchNorm、complete differentiable outer-memory bank和train-mode恢复顺序明确；
- 一次解析系数由十个seed-0 fold首refresh的full-memory parameter-gradient RMS确定；
- 系数EN/ZH共享、后续不rematch、不读endpoint；
- epoch累计gradient strength仅诊断；
- wall time只报告，不虚称compute matching；
- seed-0按frozen worst-cell rule选一个全局最强direct control；
- final两库三seed实际运行该control；
- FULL-minus-direct要求所有seed/metric/dataset正、hierarchical paired lower bound `>0`、Holm通过。

因此“proximal realization优于直接使用同一semantic moment”已成为最终可证伪claim，而非seed-0印象。

## Farkas audit

**PASS。** Example/pair/triplet/SupCon cones均有明确合法宇宙；pair cone不含同pair双方向；triplet oracle joint扫描 `(p,n)` 并包含hinge activity；small-bank brute-force parity；machine-readable universe、active-mask、generated columns及termination global minimum；abstract与realized displacement均审；conclusion只覆盖registered cones；actual AdamW归因仍依靠learned matched controls。

## Pilot design audit

Round 3 blocker已关闭。

- family选择使用固定 `Q<C<R` first-passing support rule；
- main sample和每个Rao–Wu replicate均重新计算HT totals、ESS、family selection、A/B reference profiles和effect；
- selection failure按零增益处理；
- selection success要求 `>=95%`；
- A/B cross-fit阻止identity自用于reference；
- unprocessed与rejected严格区分；
- uncertainty覆盖family selection，而非条件化于事后family。

该estimand现在完整且无outcome循环。

## Factor、fit与rollback

**PASS。** PSD factor维度与zero padding正确；repeated eigenspace/nullspace使用coordinate-projector canonical basis；backend/version/thread count冻结；target与realized collapse均检查；one block定义清楚；完整恢复model、AdamW、scheduler、scaler、RNG、sampler和epoch cursor；REMOVE replay hash验证；target从不成为inference key。

## Contribution and novelty

当前claim已正确收窄为：

> label-blind structural reflection → exact-vote-safe proximal full-bank target → uniform encoder fit → ordinary kNN。

它不再声称发明certificate、moment alignment、Gram KD、target fitting或通用metric non-equivalence。相对最近解释：vs direct semantic moment用 `DIRECT-AEXC`；vs conditional prototype/moment用 `STATE-MOMENT`；vs P4/schema prediction用P4 control；vs pair/triplet/SupCon用cone audit与learned controls；vs ECM没有pseudo-group risk、sample weighting或gradient surgery；vs teacher geometry KD，target由compiler、video labels和exact vote共同求解，不复制teacher embedding。

该单一组合贡献已经足够集中、可守且可证伪。

## Simplification opportunities

**NONE。** 当前一个exception family、一个solver、一个target-fit path和一个final strongest-direct control已经足够精简。

## Modernization opportunities

**NONE。** 不应加入learned compiler、differentiable sorter、larger teacher、router或actual-update projection。

## Remaining implementation checks

以下是执行审计项，不是方法修订 blocker：

1. 逐projector复算KKT和reference parity。
2. 检查PSD projection显式symmetrization与Dykstra correction存储。
3. 小bank brute-force验证triplet oracle和rank pivots。
4. 验证direct-control pooled coefficient不读取任何endpoint。
5. 验证Rao–Wu replicate完整重选family。
6. 先执行sealed synthetic + one-real-fold microbenchmark；`<160 GPU-hours`失败即STOP。
7. 只有microbenchmark和独立code review通过后才允许SCGP-0；teacher仍保持零调用。

**Final verdict: READY。** 下一步仅是独立实现审计与 SCGP-0 numerical microbenchmark。最终科研目标仍未满足，必须等两库三seed的最终 `+0.030/+0.030` 和完整归因门实际通过。

</details>

## Next Steps

1. Independent implementation/code audit.
2. Sealed synthetic plus one-real-fold numerical microbenchmark; STOP if projected ten-fold cost is not below 160 GPU-hours.
3. If and only if both pass, run zero-teacher SCGP-0. No teacher call is yet authorized.
