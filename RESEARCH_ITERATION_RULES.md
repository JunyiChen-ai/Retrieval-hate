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
9. 连续两轮 fundamental gate 失败则停止该方向，不继续堆补丁。
10. 正式实验前，代码与 evaluation pipeline 必须由独立 subagent review；标准是实现有效、无数据泄漏，且不存在会改变实验观察或结论的 bug，不做无关的防御性吹毛求疵。

## 标准迭代流程

1. **定义假设**：明确 localization failure 和唯一新增机制。
2. **预注册**：冻结数据、split、指标、baseline、预算和停止条件。
3. **实现与独立 review**：完成最小实现和 synthetic/smoke test，再由 subagent 检查监督边界、split、指标、训练/推理一致性和关键实现；修复所有会影响观察或结论的问题。
4. **小规模 pilot**：先在 HateMM 和 MHC-EN 独立训练，用 validation 选 checkpoint。
5. **机制检查**：比较 backbone、机制移除和破坏性 control，确认增益来源。
6. **晋级判断**：两个 pilot 数据集的 within-video 指标稳定改善，且 Frame 指标不明显退化。
7. **失败处理**：先判断是实现错误、优化失败还是机制假设失败。前两类只允许一次有证据的定向修复并重跑同一预注册；机制失败或修复后仍失败则停止该版本、保留失败结果并回退到上一有效 checkpoint，禁止改名后继续堆补丁。
8. **完整扩展**：通过后再运行 MHC-ZH 和 HateClipSeg，仍然逐数据集独立训练。
9. **Test evaluation**：可以评估中间方法，但 test 标签不得用于训练、选 checkpoint 或调参。
10. **归档**：无论成功或失败，保存配置、代码 commit、checkpoint、指标、ablation、日志、失败归因和与上一版的唯一差异。
11. **论文判断**：跨数据集有效、机制可归因且故事统一后，才进入 SOTA/论文阶段。
