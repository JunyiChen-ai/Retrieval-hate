# LB-SCGP Round 2 Re-evaluation

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
