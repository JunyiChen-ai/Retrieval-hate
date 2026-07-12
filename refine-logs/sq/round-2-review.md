# Round 2 Review

**Reviewer agent id:** `/root/sq_reviewer_replacement` (continuous from Round 1)

<details open>
<summary>Full raw reviewer response</summary>

# Round 2 Re-evaluation — SQ-RGCL

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.8 |
| Method Specificity | 7.8 |
| Contribution Quality | 7.1 |
| Frontier Leverage | 8.2 |
| Feasibility | 7.4 |
| Validation Focus | 8.0 |
| Venue Readiness | 6.7 |

**Weighted overall: 7.90/10**  
**Verdict: REVISE**

## Core assessment

- **Problem Anchor:** PRESERVED。直接针对最终普通 full-video train-memory kNN；没有转向 native head、定位或解释任务。
- **Dominant contribution:** 明显变尖锐。现在不是普通 independent environment-weighted SupCon，而是 presentation-crossed positive/negative、vote-exposed 的 triplet ranking。
- **Simplicity:** PASS。一个 posterior、一个 loss、零新 head、零 test-time artifact，方法本身不臃肿。
- **Prior-art differentiation:** 已达到**窄的 method-level difference**，不再只是多加几个 control；但核心仍由标准 metric-learning primitives 组成，创新是 crossed assignment、vote exposure 与 RGCL endpoint 的特定组合，尚不足以支撑宽泛“quotient learning”首创。
- **SQ-0:** 科学上充分作为 zero-new-call learned capacity/cost gate；恢复 actual OOF kNN `+0.05/+0.05` 正确闭合了 Round 1 的关键问题。它不是 MLLM 成功证据。
- **SQ-1:** 方向正确，但当前 128-video pilot 的 pair coverage 尚未闭合。计算 `A_ij` 与真实 top-20 wrong-neighbour enrichment 要求 pair 两端都有 teacher posterior；必须冻结 graph-closed 抽样或 induced-edge universe、每类有效 anchor/edge 数，以及“128 unique videos”对应的实际 prompt×order 总调用上限。
- **No-segment-gold audit:** **PASS**。全部信号属于父视频级 weak/privileged pseudo-signal，没有 segment/span/timestamp supervision、weighting 或 endpoint。
- **CTE interpretation audit:** **PASS**。CTE C0 仅被描述为 numerics-policy STOP，没有被误写成性能上界或 MLLM 负结果。

## Remaining scientific blockers

1. **`q→y` 固定 ceiling 不成立为 nuisance-validity 证明。** Presentation nuisance 本来就可能与标签存在数据集相关性；AUC `>0.70` 不证明语义泄漏，`<=0.70` 也不能排除 stance/harm contamination。应把 `q→y` 改为报告与触发审计的诊断量，不作普适硬门。硬门应是：完整生成 provenance、presentation-only blind audit、禁止语义污染，以及每个 class×environment cell 和两类 crossed relation 的有效 ESS/positivity。

2. **`minority-class mass share ∈[0.20,0.80]` 表述失当。** “minority share”天然不超过 0.5，且绝对比例会错误拒绝真实但偏斜的 nuisance。替换为每个 environment×class 的最小有效质量/ESS，以及每个 anchor 两侧 relation ESS；保留 class-pure cell STOP。

3. **SQ-1 pair universe 必须闭合。** 预注册从 OOF top-20 图选择 anchor，并将所需邻居端点纳入不超过调用预算的闭包；所有 teacher/cheap/base/null 比较必须在完全相同的 observed-edge universe 上完成，并按 anchor bootstrap。

4. **Vote exposure 仍可更直接。** 当前 top-20 内统一 `E=1` 没有使用仓库真实的 similarity-sign/rank weight。建议直接采用冻结的 repository vote-contribution magnitude，并由第20名权重连续定义 tail；这样可删除独立 `eta/kappa` 自由度，同时强化 final-kNN directness。

## Documentation polish

- 明确 `q^arch` 原始 `neutral_summary` 的生成 prompt 是否也完全 label-blind；只审计当前读取 key 不足。
- 精确定义 ENV-SUPCON、Yang-style 与 P4-PREDICT 的 matched loss。
- 明确 full triplet sum 的向量化/采样实现及 pair-budget matching。

## Simplification / modernization

无需增加 HSIC、subspace、router、GroupDRO 或新 head。最有效的简化是用真实 repository rank-weight 替代 `eta/kappa` tail，并把 `q→y` 从伪因果硬门降为诊断。现代化程度已经合适，不需要再加基础模型组件。

当前方案已从“weighted SupCon 换名”推进为可辨识的窄机制，但 SQ-1 pair closure 与 nuisance-validity gate 仍是科学 blocker，因此不能 READY。

</details>
