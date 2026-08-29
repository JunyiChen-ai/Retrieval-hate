# Round 4 Review

**Reviewer agent id:** `/root/sq_reviewer_replacement` (continuous from Round 1)

<details open>
<summary>Full raw reviewer response</summary>

# Round 4 Re-evaluation — SQ-RGCL

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 10.0 |
| Method Specificity | 9.3 |
| Contribution Quality | 8.7 |
| Frontier Leverage | 9.0 |
| Feasibility | 8.9 |
| Validation Focus | 9.2 |
| Venue Readiness | 8.3 |

**Weighted overall: 9.12/10**  
**Verdict: READY**

这里的 READY 仅表示方法规范可以进入 experiment-plan/implementation audit；不表示 SQ 已产生性能结果，更不表示项目终局目标完成。

## Review findings

- **Problem Anchor:** PRESERVED verbatim，无 drift。目标仍是两数据集最终 ordinary full-video train-memory kNN 的 accuracy/macro-F1 substantial improvement。
- **Dominant contribution:** 单一且清晰：train-only MLLM presentation posterior 定义 crossed positive/negative relation，由真实 top-20 harmful-vote exposure 驱动共享最终 embedding。
- **Simplicity:** PASS。FULL 只有一个 posterior 和一个 scalar loss；无新 head、subspace、router、GroupDRO、test teacher 或推理模块。
- **No-segment-gold:** PASS。唯一 gold 是父视频二分类标签；人工 audit 是父视频级信号 QC，不进入训练监督。
- **CTE interpretation:** PASS。CTE C0 仅作为 numerics-policy STOP，没有被当作性能上界。

## Round 3 blocker closure

### Exact top-20 exposure — CLOSED

删除 rank>20 tail 是正确修正。当前曝光严格等于 top-20 rank weight 与 harmful signed cosine contribution，超出 top 20 为零，不再混入无 evaluator 根据的 far-negative prior。

这并未退回 SSR/EDCM 的冻结 action universe：正样本来自 full bank，encoder/query/keys 共同更新，top-20 集合逐 epoch 刷新，因此旧邻域外样本仍可进入最终投票集合。

### Anchor-cluster power — CLOSED

以 anchor 而非 edge 为独立单位、使用 variance upper bound、80% power、Holm-adjusted alpha，并在 closure 超过资源上限时 `STOP_INFEASIBLE`，科学逻辑成立。它不会把大量相关 edges 伪装成重复样本，也不会观察 teacher outcome 后修改样本量。

Cheap-vs-shuffle variance 是合理的 pre-call planning proxy。Experiment plan 应验证所用 upper bound 不低于 bounded-outcome analytic worst case；这是非阻塞执行细节，不需要再改方法。

### Representative closure/IPW — CLOSED

先在 class×OOF-margin strata 内均匀抽 anchor，再添加全部 top-20 endpoints，消除了 hub-overlap 优化带来的主要选择偏差。IPW、anchor bootstrap 与 selected-vs-full balance reporting 定义了明确 estimand。

使用 train binary labels 做抽样分层是合法的监督使用，不是泄漏：标签不提供给 teacher，也不涉及 validation/test。

### Four-call artifact/audit/cap — CLOSED

四 posterior 算术平均、最小 confidence、parse/agreement fail-closed 的定义确定且可复现。`≤256` unique videos 与 `≤1024` invocations 的区分清楚。六个 ontology entries 全部预先 active，audit protocol、raters、adjudication 和 Wilson gate 已冻结。

## Novelty assessment

该方法不再只是 weighted SupCon 的改名。相对 ENV-SUPCON，它使用同一 posterior 构造 coupled crossed triplet，并仅由当前真实 harmful top-20 vote exposure激活；相对 Yang/CDAL，它没有属性去相关、独立子空间或环境预测器；相对 P4，它不预测 posterior。

创新仍应保持窄表述：这是 **MLLM-defined presentation crossing × exact-vote-exposed RGCL ranking** 的机制组合，不是首次 metric learning、首次 invariance 或通用 quotient theory。当前差异已足够精确，可进入实验验证；正式投稿前仍需独立最终查新。

## Non-blocking experiment-plan details

- 固定 power calculation 的完整公式、有限总体修正和 bootstrap 次数。
- 从 evaluator 代码逐项验证 `20..1` rank arithmetic、cosine sign及 tie handling。
- 明确 audit 中“semantic contamination”的标注示例与 adjudication记录格式。
- 在调用前验证 `q^arch` 原始生成 provenance；缺失时按方案直接停止，不得替换或补救。
- 微基准验证64 triplets/anchor、bank refresh与各 control的计算/梯度匹配。

没有剩余需要通过增加模块解决的科学 blocker。下一步应进入独立 experiment plan 和代码审计，先执行 P0/SQ-0；SQ-0 未通过前不得产生新 teacher 调用。

</details>
