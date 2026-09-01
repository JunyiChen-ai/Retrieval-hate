# Process Review RESET6 — 2026-09-01

独立只读 process review；不审代码、不复跑实验、不提出具体新 candidate。依据 `AGENTS.md`、`RESEARCH_ITERATION_RULES.md`、`research-wiki/STATUS.md`、`research-wiki/FAILURE_EQUIVALENCE_LEDGER.md`、`docs/PROCESS_REVIEW_RESET5_2026-09-01.md`，以及 RESET5 三次正式方法的 README、novelty/technical review 与 `runs/` 权威 summary。

## 裁定

**RESET**。项目继续，但不得原样继续 MultiHateLoc self-derived modality responsibility/fusion 路线。累计连续 performance failure 保持 `11`；落实以下流程修正后，新的 process-review 窗口记为 `0/3`。这不是 performance 清零。

## 1. 当前为何停滞

RESET5 三轮都是真实、有效的正式方法迭代：均通过 novelty，完成一次基础 technical review、逐语料完整 validation hyperparameter/checkpoint selection，以及 HMM/HCS test。停滞不来自 premise 或没有运行方法。

根因是现有诊断证据与候选所需纠错信号之间存在缺口。`runs/20260831_multihateloc_test_error_analysis/main/metrics.json` 的 best-branch oracle gap 与 DMS mismatch 证明“如果知道正确分支，存在 within headroom”，但没有证明 video-label-only 训练下能从当前模型自身的 branch confidence、fused top-K 或 masked forward 推断正确的逐秒模态可信度。三轮方法都从同一自生成 support 派生训练信号：

- Temporal Coalition-Credit 用 fused model 的 top-K 与 masked-coalition credit 监督 local router；HMM aligned within 反而低于 shifted `.000363`。
- Witness-Failure Debiasing 用 fused top-K 与同模型单模态 branches 的 GCE failure 重加权；relative-minus-uniform within 为 HMM `+.000579`、HCS `-.000559`。
- Witness-Preserving TokenFusion 仍用 fused top-K 定义 retain 约束，并在同秒 donor substitution 上学习 gates；aligned-minus-shifted within 仅 HMM `+.000130`、HCS `+.002153`。

这三组结果共同表明：当前 self-derived witness/branch signals 没有提供跨 HMM/HCS 可用的局部纠错方向。更换 credit、loss weighting 或 feature substitution 并未改变这个信息瓶颈。

第二个根因是机制目标与最终 performance 目标错位。三轮都围绕 within ordering 设计，而最终门要求 HMM/HCS 的 pooled AP、pooled ROC、within ROC 全部 SOTA。尤其 HCS same-harness anchor 相对 official seed-234 的 pooled AP/ROC 为 `-.029307/-.046571`，三轮新增机制都只产生千分量级变化；当前路线没有解释或解决这个 pooled gap。RESET5 因而在优化一个局部子问题，却没有一条可信路径覆盖最终六项指标。

## 2. 流程问题判定

### 重复失败链：存在

三轮表面分别是 Shapley valuation、LfF debiasing 与 TokenFusion，但共享同一链条：当前 fused top-K 或当前 branches 产生 latent support/可信度，再反过来训练同一 fused scorer。它们不是代码或公式上的严格同构，却是信息来源上的重复失败。RESET5 的 target lock 防止了 failure-target churn，却同时把整个窗口消耗在一个没有 train-observable correction signal 的 underdetermined target 上。

### Candidate churn：存在，但不是 failure-target churn

没有中途切换 occupancy、position、teacher 或 raw statistic；这一点比 RESET4 改善。churn 发生在同一抽象机制内：连续换跨任务来源和介入位置，却没有先解决“正确局部责任从哪里来”。novelty 三门保证了每个 adaptation 可主张，不保证它们拥有新增任务信息，因此三个窄 novelty GO 不能作为继续该信息链的理由。

### 无效 premise：没有 premise churn；但旧 evidence 被过度解释

RESET5 没有新增 premise，这符合规则。问题是把 test oracle headroom 当成了可学习信号的证据。oracle 只能诊断缺口，不能证明现有 branch/self-witness confidence 能识别正确秒。以后必须在 brief 中明确区分“performance oracle/headroom”与“训练和推理时实际可观测的 correction signal”。这不是恢复 raw-statistic sweep。

### 过早复杂化：存在

在 correction signal 未成立时，依次投入 8-coalition valuation/router、GCE bias experts/relative weights、三路 retain/projection/substitution，并为三轮各跑 14 trials × 2 corpora。完整 validation search 本身合规且必须保留；资源错配发生在进入正式搜索之前选中了信息上无新增约束的机制，而不是 trial 做得太完整。

### 目标偏移：存在

研究任务没有变，但实际优化集中于相对 matched control 的 within 千分量级变化，未要求候选解释如何同时跨越 HMM/HCS 的 pooled 与 within SOTA gap。机制门可以先于最终门，但连续三轮之后仍只针对 within 子问题，已构成 performance 目标偏移。

## 3. 必须落实的流程修正

1. **关闭当前信息链。** 不再从 MultiHateLoc 自身 fused top-K、DMS、masked-coalition credit、普通单模态 branch confidence或其简单变换中生成局部责任，再通过 router、reweighting、replacement、gate、loss 或 head 续命。若未来重开，必须先有独立于当前 fused scorer 自确认的、HMM/HCS 共同的 train/inference-available local correction evidence；不得只换数学工具。
2. **解除“只能继续 modality-selection/fusion”的 target lock。** 下一轮仍必须从已有 test evidence 出发，但 failure target 要由最终六项指标共同决定，不能仅因 best-branch oracle gap 大就继续同一方向。
3. **先做一次 bounded starting-point/goal-gap audit，再生成 candidate。** 只复用现有 authoritative test artifacts，列出 HMM/HCS 各三项的 current anchor、SOTA gap、共同 error subgroup，以及哪些 gap 有 train/inference 时可用的 observation。特别核清 HCS same-harness anchor 相对 official anchor 的 pooled drift来自 selection/config差异还是 scaffold不一致。该审计不是代码 review、不是 premise、不得用 validation 决定方向，也不得扩展成 raw-statistic sweep。若现有 artifacts 不足，只记录缺失证据；不要用术语替代证据。
4. **候选 brief 增加两个必填字段，但仍限一页。** 一是 `observed headroom`，二是 `available correction signal`；后者必须说明训练和 test inference 时实际可见什么、为何不是当前模型的循环自确认，以及该信号怎样进入唯一 raw final score。还须逐项说明预期影响 HMM/HCS 的 pooled AP、pooled ROC 与 within ROC；不要求虚构全部正增益，但必须说明为何有机会跨越主要 gap。
5. **每个新 epoch 首轮保留 clean anchor/control，但不再把 anchor-compatible 等同于只能微调原 fused scorer。** 若候选必须改变 representation 或 backbone，必须在 brief 中把唯一核心机制和其关闭后的 matched control定义清楚；不得同时改多个机制。正式 test 同时报告 matched control、official starting point与固定 SOTA thresholds。
6. **Validation 规则不变。** 含可调超参数的方法必须逐语料进行足够的完整 validation search，联合选择配置与 checkpoint，锁定后立即 test。Trial 数由主 agent按维度和成本决定；不得用 validation performance选择方向、机制或 starting point，也不得设置 test 前 performance gate。
7. **跑前 technical review 只做一次基础审查。** 仅检查会改变实验观察或结论的 bug：机制是否真正进入 final score/gradient、split isolation、padding/alignment、checkpoint/config加载链、test coverage与 canonical evaluator调用。不得审代码风格、做泛化防御性吹毛求疵或反复 review。静态检查和单元测试后直接完整 validation；禁止 smoke。
8. **正式资源继续按 performance iteration 投放。** novelty GO 后立即实现、一次基础 technical review、完整 HMM/HCS validation selection与test。每轮结束必须用 test predictions/GT 做一次聚焦 error analysis，决定关闭或 Rule 18 唯一 corrective；不得在两轮之间插入无界 premise/producer/statistic search。
9. **计数保持诚实。** RESET5 三次均为 performance failure，累计保持 `11`。落实本 RESET 后新窗口 `0/3`；只有通过 Rule 15 双数据集 performance gate才清除连续 failure事实。

## 4. 已有证据与缺失证据

### 已足够

- MultiHateLoc 存在显著 best-branch oracle headroom，DMS 与 oracle branch 匹配差；这足够说明当前融合不理想。
- 三次独立正式 test 足以关闭当前 self-derived modality responsibility/fusion 信息链；差值稳定在零附近且没有共同 load-bearing control 证据。
- Coalition timing、shortcut-failure weighting、aligned donor replacement 的当前版本均无 Rule 18 corrective 依据；不需要再调 alpha、GCE、weight、gate、projection、retain budget、shift 或 fusion strength。
- RESET5 没有 premise churn，novelty 与一次 technical review不是主要瓶颈；无需增加 review 次数。
- 当前候选离 HMM/HCS 全部三指标 SOTA 很远，不能以微小 matched-control within 增益继续包装成接近成功。

### 仍缺失

- 一个同时解释 HMM/HCS pooled 与 within gap、而非仅解释 best-branch within oracle gap的共同 failure decomposition。
- 一个在 train/inference 时真实可用、独立于当前 fused top-K/branch self-confidence、并能跨两语料指向正确局部排序的 correction signal。
- HCS same-harness anchor pooled AP/ROC 明显低于 official seed-234 的确切流程归因；在未核清前，不应把 candidate 的整体 pooled下降归因于新增机制。
- 对下一机制是否有足够幅度跨越固定 SOTA gap的证据。novelty GO 与千分量级 control gain都不能替代这一点。

## 5. 方向裁定

- **CONTINUE**：整体弱监督 hateful video localization 研究；HMM/HCS test-first正式方法迭代；使用现有 test predictions/GT 的聚焦 error analysis；完整 validation hyperparameter/checkpoint selection；一次基础跑前 technical review。
- **RESET**：failure target 选择方式。先从 HMM/HCS 六项 SOTA gap 与可用 correction signal联合确定下一目标，不再由 best-branch oracle 单独指定路线。
- **PAUSE**：任何依赖新 teacher、producer 或全新信息源但尚无 Rule 14 最低证据的方向；需要多核心机制同时改写、无法定义 matched control的方向；在 bounded goal-gap audit完成前的 candidate生成与实现。
- **STOP**：Temporal Coalition-Credit、Witness-Failure Debiasing、Witness-Preserving TokenFusion 当前版本及其 alpha/GCE/weight/support producer/router/gate/projection/retain budget/shift/fusion-strength变体；更广义的“用当前模型自身 latent witness/branch confidence推断局部模态责任”路线，除非出现规则所要求的独立跨语料新证据。

## 恢复条件

主 agent 必须先把本审查的流程修正写入当前状态/规则，完成 bounded starting-point/goal-gap audit并记录权威 test artifact路径，然后才可提出下一 candidate。恢复后执行顺序固定为：已有 test failure与可用 correction signal → 一页 brief → novelty三门 → 最小单机制实现 → 一次基础 technical review → 逐语料完整 validation selection → HMM/HCS test三指标 → 聚焦 test error analysis与关闭/晋级裁定。
