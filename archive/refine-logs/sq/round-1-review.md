# Round 1 Review

**Canonical reviewer agent id:** `/root/sq_reviewer_replacement`  
**Recovery note:** The initially spawned reviewer `/root/sq_method_refine/sq_reviewer` repeatedly interrupted before producing any review and was discarded. The replacement is canonical for Round 1 and every later re-evaluation.

<details open>
<summary>Full raw reviewer response</summary>

## Round 1 Review — SQ-RGCL

### 7维评分

1. **Problem Fidelity: 9.5/10**  
严格对齐 final full-video train-memory kNN、双指标 +3pt、可移除 MLLM 和无片段金标注红线，无明显问题漂移。

2. **Method Specificity: 6.4/10**  
后验、关系权重和 loss 已具体，但实际 MLLM pilot 的覆盖、校准、条件增益及停止阈值仍未冻结；full-bank NCA 作用于同一 embedding，却不等于直接优化项目的 top-20 signed rank-weighted vote。

3. **Contribution Quality: 5.8/10**  
“soft quotient”目前算法实质接近 environment-weighted supervised contrastive/NCA；crossed-fiber、软后验及 RGCL endpoint 构成窄组合差异，但尚不足以支撑强机制新颖性。

4. **Frontier Leverage: 7.3/10**  
MLLM 作为 train-only privileged relation teacher 是自然且克制的现代用法，不是装饰性 concat 或 test-time judge。

5. **Feasibility: 7.2/10**  
零新增模块、共享 encoder 和 epoch bank 可实现；主要风险是 full-bank 训练成本、stale keys 与跨数据集共享超参。

6. **Validation Focus: 6.1/10**  
控制较完整，但当前 P0 用 error-AUC/一步梯度 alignment 替代 Gate-0 要求的 learned strict-OOF actual-kNN capacity screen，证据不足。

7. **Venue Readiness: 5.6/10**  
方法叙事集中，但 prior-art 窗口窄，尚不能排除“任务定制的 weighted SupCon”。

**Weighted Overall: 6.88/10**  
**Verdict: REVISE**

### Drift Warning

Problem Anchor 未漂移；但验证协议发生实质弱化：Gate-0 的两库 learned OOF `+0.050 acc/+0.050 mF1` SQ-0 被代理诊断替代。必须恢复，代理指标只能作为附加 fast-fail。

### 阻塞性修正

1. **label-blind 不等于 nuisance-valid。** `neutral_summary` 和视频内容仍可能编码 stance/harm/label proxy。必须冻结 presentation-only prompt/schema，并设置 class×environment overlap、class-pure cell、`q→y` 泄漏和禁止语义字段审计；若 posterior 实质表达 hate semantics，不得靠 class conditioning 补救。

2. **恢复 dense learned SQ-0。** 在两库严格 train-OOF 中实际训练容量匹配 quotient，并以最终 ordinary kNN acc/mF1 过 Gate-0 门槛；一步梯度 alignment 不能替代训练结果。

3. **冻结实际 MLLM SQ-1。** Teacher posterior 必须在覆盖、ESS、conditional wrong-neighbour enrichment 和梯度方向上显著胜 base-cluster、cheap-format、within-class shuffle；不能由 archive proxy 自动解锁。

4. **澄清 final-kNN directness。** 当前 loss 是 full-bank surrogate，不是 exact top-20 vote objective。要么加入与真实 vote/margin一致的最小排序项，要么收窄“exact”表述，并证明邻居投票改善而非仅整体类簇收缩。

5. **正面处理 P2/P4。** 条件模型须预测具体 wrong-class top-20 attraction，而非泛化 error；FULL 必须胜 label-only、field-prediction及标准 environment-weighted SupCon/Yang-style control。

### Simplification / Modernization

保留一个 posterior、一个 crossed-relation loss和一个 inference path；删除“支持贡献”式包装，将 P0只定位为 gate。无需新增 head、HSIC、router或显式子空间。新颖性应收窄为：**train-only MLLM soft presentation assignments驱动 exact RGCL retrieval embedding 的 crossed-fiber supervised NCA**，并诚实承认其强 prior-art 风险。

</details>
