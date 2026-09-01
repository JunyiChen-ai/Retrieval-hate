# Validation-Selected Marked Temporal Splat

截至 2026-09-01。RESET3正式方法 2；不运行 premise，不改变 novelty 或 renderer。目的仅是补齐首版 marked splat 缺失的正常 validation hyperparameter/config/checkpoint selection，然后立即 HMM/HCS test。

模型使用`src/marked_temporal_splat.py`，已与归档首版同参数状态做 exact forward equivalence 检查。每个语料四个 train/validation-only配置全部结束后才锁定选择；两语料都选择完毕后才生成任何test prediction。最终仍调用冻结canonical evaluator。若双数据集performance gate失败，RESET3窗口记`2/3`，不追加配置扫描。

权威输出：`runs/20260901_validation_selected_marked_splat/formal_seed234/`。

## Formal result

HMM validation选择`low_regularization`（video AP`.854421`，epoch 21），HCS选择`bag_focus`（`.956770`，epoch 61）。权威汇总为`runs/20260901_validation_selected_marked_splat/formal_seed234/summary.json`。

- HMM AP/pooled ROC/within为`.516086/.756225/.723204`，相对首版为`+.020720/+.015494/-.004504`。
- HCS为`.572285/.550595/.532836`，相对首版为`-.003184/-.003231/-.001245`。

双数据集performance gate失败，RESET3窗口记`2/3`。Validation选择能改善HMM pooled，但缺口远大于配置问题；HCS也不是checkpoint未选好。不追加配置扫描，本轮归档。
