# Semantic multiplicity test premise

截至 2026-08-31。此诊断按 Rule 10 使用 HateMM/HateClipSeg 的 test predictions 与 test GT，
检验一个窄假设：MIL 是否把跨时间重复出现的相同视觉语义多次计入 bag evidence，从而放大
intro、logo 或重复 benign content 的 false-positive evidence。

固定 density 使用 MultiHateLoc 的 frozen visual ViT-B/16 1 fps feature。每秒的 density 是
与至少相隔 5 秒的其他秒之间的 cosine RBF kernel mass，再加自身质量 1。5 秒内局部邻居被
排除，因为上一轮 test diagnosis 已证明固定 temporal smoothing 在 HMM/HCS 方向相反。

每个 mixed-label positive video 的高分 support 固定为 `ceil(T/4)` 对应的 score superlevel；
cutoff 同分全部纳入，不用时间索引拆同分，并报告实际扩张。FP/TP AUC 把 false-positive 秒
定义为正类，因此 AUC 高于 `.5` 才表示 density 在 FP 上更高。time-reversed control 只在同一
视频内反转 density 顺序，保持其边际分布与视频长度不变。

Premise 必须在两个语料都同时满足：pooled 与 per-video macro 两个层面上，semantic density
对 false positive 的 AUC 都高于 `.5` 且高于 time-reversed control；同时 inverse-density
accounting 必须降低高分 support 内的 false-positive score-mass share。这里的 score mass 是
明确的诊断量，不是对 producer 训练时 bag likelihood 的重建。任一失败即
`STOP_BEFORE_METHOD_IMPLEMENTATION`。

脚本另报告固定 inverse-density rank penalty 的 test diagnostic readout，但它属于 calibration
诊断，不是候选方法、不能晋级、不能作 SOTA claim。若 premise 通过，后续候选只能把
multiplicity invariance 写进训练时 bag likelihood，test 必须直接输出单一网络帧分数。

输入固定为上一轮 HateMM/HateClipSeg 各自 `mobius_nonminimal/score_full` 的正式 test artifact；
脚本核其 source config、test metrics provenance、GT exact coverage、逐视频 shape 与 finite。
项目固定三指标只调用共享 `eval_baseline_scores.evaluate_scores`。逐视频 FP/TP AUC 是 Rule-10
error-analysis 统计，不是第四个项目 performance 指标。test GT 不参与梯度、训练或 checkpoint
选择；本轮输出及由它启发的后续结果均属于 iterative/developmental evidence。

权威输出：`runs/20260831_semantic_multiplicity_test_premise/main/metrics.json`。

## 结果与结论

`premise_pass_both=false`，按冻结规则 `STOP_BEFORE_METHOD_IMPLEMENTATION`。

- HateMM：semantic density 识别 high-score false positive 的 pooled/macro AUC 为
  `.3881/.4521`；inverse-density accounting 把 FP high-score mass share 从 `.3155`
  **提高**到 `.3500`。五个 premise gate 仅过一个。
- HateClipSeg：对应 AUC 为 `.5594/.5272`，FP mass share `.3898→.3832`，五门全过。

该机制只在 HateClipSeg 成立，不能按语料选择。固定 inference-time density penalty 还使两个
语料的三项指标整体下降，因此也没有可保留的 readout。semantic quotient / multiplicity-
invariant MIL 不进入实现，目录归档。
