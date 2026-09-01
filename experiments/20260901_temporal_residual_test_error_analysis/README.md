# Temporal Residual Reconcilement post-test error analysis

截至 2026-09-01。使用 validation-selected formal HMM/HCS test predictions 与 test GT，只做失败定位，不训练、不选checkpoint。权威输入为 `runs/20260901_temporal_residual_reconcilement/formal_val_selected_seed234/`，输出为 `runs/20260901_temporal_residual_test_error_analysis/main/metrics.json`。

目标只回答：core 的失败是否在两个语料都呈现同一个、可由一次 corrective iteration 修复的 residual 分配问题。分析 branch metrics、logit scale、core-control per-video ordering change；不运行新 premise，不搜索新 statistic。

## 结果与裁定

HMM core 相对 control 的 fused within 提升 `+.015077`，但仅 `52.9%` eligible videos改善；visual/text absolute-logit mean分别从`1.034/.577`增至`2.694/1.369`，同时其branch pooled ROC降至`.6217/.6482`，表明 residual 放大局部分离的同时破坏了跨视频与分支判别。

HCS core 相对 control within 为`-.009277`，仅`43.3%`视频改善；audio/text branch pooled ROC降至`.4573/.5043`，absolute-logit mean反而从`.813/1.359`降至`.273/.495`。它不是与HMM相同的统一logit过放大，而是弱分支退化/欠利用。

因此不存在一个跨HMM/HCS共同的scale、trust-region或residual-weight correction。继续调幅度、stage、cycle或归一化会重复已完成的validation search，不能由当前test evidence支持。裁定关闭整个 alternating temporal-residual family，不使用Rule 18唯一corrective iteration。
