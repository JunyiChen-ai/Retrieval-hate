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

1. **定义假设**：明确 localization failure 和唯一新增机制。
2. **预注册**：冻结数据、split、指标、baseline、预算和停止条件。
3. **实现与代码 review**：完成最小实现和 synthetic/smoke test，再由独立 subagent 检查监督边界、split、指标、训练/推理一致性和关键实现；修复所有会影响观察或结论的问题。
4. **小规模 pilot**：先在 HateMM 和 MHC-EN 独立训练，用 validation 选 checkpoint，再在 test 上报告结果；test 标签不得用于训练、选 checkpoint 或调参。
5. **Performance gate**：两个 pilot 数据集的 within-video 指标稳定改善，且 Frame 指标不明显退化，才进入机制检查。
6. **失败归因与下一轮**：若未通过，区分实现错误、优化失败和机制失败。前两类允许一次有证据的定向修复；机制失败或修复后仍失败，则归档当前版本、回退到上一有效 checkpoint，并基于失败证据提出新的机制假设，重新从第 1 步迭代。
7. **机制检查**：对已 work 的版本比较机制移除和破坏性 control，确认增益来自新增机制；不通过则按第 6 步进入下一轮。
8. **独立 novelty assessment**：由未参与实现和代码 review 的 subagent 检查相关工作、贡献边界和故事完整性。若 novelty 不足，保留当前有效版本作为新 baseline，提出有明确动机的新机制并重新从第 1 步迭代，禁止只改名、堆模块或调 calibration。
9. **完整扩展**：performance、mechanism 和 novelty 均通过后，再运行 MHC-ZH 和 HateClipSeg；每个数据集仍独立训练、validation 选 checkpoint、test 报告结果。
10. **归档**：每轮无论成功或失败，保存配置、代码 commit、checkpoint、指标、ablation、日志、review 意见、失败归因和与上一版的唯一差异。
11. **向用户汇报**：汇总全部数据集结果、机制证据、novelty reviewer 结论、局限和下一步选项，由用户决定继续迭代或推进论文。
