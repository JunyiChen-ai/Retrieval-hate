# POWA test error taxonomy

截至 2026-08-31。只读诊断；不训练、不选 checkpoint、不产生候选 prediction。

按 `RESEARCH_ITERATION_RULES.md` Rule 10，本诊断读取四个 corpus-specific POWA
seed-234 test predictions 与固定 test GT，检查错误是否可由统一的时间偏移、视频位置
shortcut或多模态内容边界解释。权威输出写到
`runs/20260831_powa_test_error_taxonomy/analysis.json`。

由本诊断影响的后续 test 结果均属于 iterative/developmental evidence。test GT 不得
进入梯度、donor/pseudo-label 选择、checkpoint selection 或 corpus-specific inference
规则。

## 结论

权威输出记录可读输入路径与分析配置。固定 lag probe 只用于发现偏移/位置 shortcut；尤其当
最优值落在 `[-30,30]` 搜索边界时，不能解释成可修正的 annotation lag。content-change
boundary probe 同样是只读归因，不是候选后处理。

## 记录的定性 test exposure

为区分 semantic failure 与 boundary artifact，查看了以下已在 quantitative worst-10
中出现的视频帧；没有据此设阈值、时间 prior 或超参：

- `bit_I03hhZDAzOu5`：GT positive `[31,294]`。查看 5/40/100/280 秒帧；5 秒是
  intro logo，之后为持续战争影像。POWA top score 落在 intro，属于真实的
  distinctive-intro shortcut。
- `bit_JUPPGbicIM0r`：GT positive `[10,199]`。查看 5/15/80/180 秒帧；5 秒虽然
  被 GT 标为 benign，却已经包含明确反犹仇恨图像，之后也持续出现仇恨字幕与内容。
  因此其低 within AUC 部分来自 coarse/inconsistent boundary，而非语义 locator 把
  benign 排高。

这项检查否定两个过度结论：不能把 HCS 全部失败归因于模型，也不能把 visual
recurrence/smoothing 的增益直接解释成更准确的 hate semantics。后续设计不得使用这些
具体边界或位置；HCS 仍按固定 GT 评测，但需在最终报告披露该 label-noise evidence。
