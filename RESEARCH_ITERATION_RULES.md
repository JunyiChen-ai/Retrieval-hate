# Research Iteration Rules

## 本次迭代暴露的问题

- 没有始终锁定弱监督 hateful video localization 这一核心目标。
- 混淆 pooled Frame/video-level 提升与真正的 within-video localization。
- 把 ensemble、calibration 带来的高指标误认为方法创新。
- 未经确认迁移主数据集，破坏了实验连续性。
- 用版本号包装输入、校准、protocol 修补和失败 pilot。
- Review 没有优先检查任务、机制和指标是否对齐。

## 以后必须遵守的规则

1. 主数据集固定为 HateMM、MHC-EN、MHC-ZH、HateClipSeg；每个数据集独立训练、验证和测试。
2. 新数据集只能作为 external validation，加入前必须得到明确同意。
3. 指标优先级：within-video AP/ROC > pooled Frame AP/ROC > video-level AP/ROC。
4. Ensemble 和 calibration 只能作为 baseline 或 upper bound，不能作为论文主方法。
5. 每轮只改变一个核心机制，并保留可归因的 ablation/control。
6. 至少在两个主数据集完成验证，才能称为一次方法迭代。
7. 晋级必须同时满足 performance 和 novelty；视频级或 pooled 指标单独提升不算定位进展。
8. Review 优先阻断影响任务语义、数据泄漏、性能或结论的根本问题。
9. 连续两轮 fundamental gate 失败时，停止当前假设链并向用户汇报；基于失败证据重新定义机制，不在原版本上继续堆补丁。
10. 正式实验前，代码与 evaluation pipeline 必须由独立 subagent review；标准是实现有效、无数据泄漏，且不存在会改变实验观察或结论的 bug，不做无关的防御性吹毛求疵。

## 标准迭代流程

1. **明确研究问题**：固定任务、输入输出、监督条件、评价标准和实际约束。
2. **建立文献地图**：梳理本领域和相邻领域的主流范式、代表方法、性能上限与已知问题。
3. **建立可靠 starting point**：复现最相关的 baseline，检查数据、实现、评测和结果是否可信。
4. **诊断 starting point**：分析它在什么情况下失败、缺少什么能力，以及失败发生在哪个处理环节。
5. **生成候选 idea**：针对失败原因提出机制，也从相邻任务迁移可能适用的新范式。
6. **做初步 novelty check**：由独立 reviewer 搜索是否存在相同问题与核心机制，排除直接重复、简单组件拼接和仅换数据集的方案。
7. **排序候选 idea**：综合考虑 novelty、合理性、可证伪性、实现成本和潜在收益，选择最值得验证的方案。
8. **为候选定义机制假设**：明确它为什么可能有效、应该改善什么、什么结果意味着假设失败。
9. **实现最小 pilot**：只实现验证核心机制所需的最小版本，同时保留 baseline、消融和必要的 control。正式实验前由独立 subagent review 代码和 evaluation pipeline，排除会影响实验观察或结论的问题。
10. **评估 pilot**：每个数据集独立训练，用 validation 选 checkpoint，再在 test 上报告结果；test 标签不得用于训练、选 checkpoint 或调参。同时判断：
    - 是否提高性能；
    - 是否在预期样本上改善；
    - 增益是否来自声称的机制；
    - 是否存在泄漏、错位或评测问题。
11. **根据结果分流**：
    - 性能和机制都成立：晋级；
    - 性能提高但机制不成立：寻找真实增益来源；
    - 机制成立但性能不提高：调整机制的实现方式；
    - 两者都不成立：淘汰当前 idea、记录负结果，并返回第 5 步生成新候选；
    - 实现不可靠：修复后重跑，暂不评价 idea。
12. **重新做深度 novelty check**：由未参与实现和代码 review 的独立 reviewer，根据实验揭示的“真正有效部分”重新查新，并与最接近的方法逐项比较。
13. **精炼方法**：删除无效模块，将复杂原型简化成少数必要、清晰且相互配合的模块。
14. **扩大验证**：跑完整数据、多数据集、多 seed、消融、鲁棒性和统计显著性实验。
15. **完整性审计**：由独立 subagent 检查数据泄漏、score、rank、threshold、alignment、coverage 和 evaluator，确保结果可复现且不存在会改变结论的实现问题。
16. **形成最终结论并向用户汇报**：明确：
    - 方法是否真的 work；
    - 为什么 work；
    - 真正 novel 的部分是什么；
    - 可以 claim 什么；
    - 哪些结论仍不能 claim。

每轮无论成功或失败，都必须归档配置、代码 commit、checkpoint、指标、消融、日志、review 意见、失败归因和与上一版的唯一差异。
