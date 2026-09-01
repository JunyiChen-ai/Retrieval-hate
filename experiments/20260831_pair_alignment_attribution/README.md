# Fixed-pair lexical alignment attribution

截至 2026-08-31。该实验是 Rule 10 允许的 developmental test error analysis，
不是方法、ensemble、calibration、routing 或 SOTA claim。

## 问题

固定 complementarity diagnostic 已发现：HateMM lexical `.35` + POWA `.65`，以及
HateClipSeg lexical `.05` + VERA `.95`，在 global empirical-CDF 后均同时超过三项
SOTA。这里不再扫描 signal、pair 或权重，只判断增益是否依赖 lexical score 的正确
视频内时间对齐，而不是任意 tie/近平分扰动。

## 固定分析

- 输入、branch、seed 与上一诊断完全相同；只使用上述两个已冻结 pair/weight。
- 每个 signal 在完整 test frame pool 上做 average-tie empirical-CDF normalization。
- aligned arm 使用原始 lexical timing。
- 16 个 phase-shift controls：第 `j` 个 control 对每个视频的 lexical rank array 做
  约 `j/17` 视频长度的非零 circular shift。每个 control 保留该视频 lexical multiset、
  完整全局 lexical rank distribution、视频长度、base score 与固定 blend weight，仅破坏
  lexical 与 GT 的时间对应。
- 所有 arm 均调用共享 evaluator 输出 pooled AP、pooled ROC、within-video macro ROC。
- 另报告 eligible positive video 的 per-video AUC delta、speech coverage、base exact-tie
  fraction、base/blend Kendall rank correlation 与 GT occupancy 分层结果。

## 冻结解释门

只有同时满足以下条件，才允许说“正确 lexical timing 对 fixed-pair gain load-bearing”：

1. aligned arm 在 HMM/HCS 各自仍三项 all-SOTA；
2. 两语料 aligned within 均比 16 个 shifted controls 的 equal-control mean 至少高 `.020`；
3. 两语料 aligned within 均严格高于至少 14/16 个 shifted controls。

通过只支持 lexical temporal alignment 是下一方法可使用的独立局部 observation；不批准
score blend 或 teacher distillation。失败则说明 fixed-pair 互补不足以支持 lexical mechanism，
关闭该 observation，不再围绕它设计方法。

## 正式运行

唯一入口 `run.sh`，只允许新建
`runs/20260831_pair_alignment_attribution/main/`，保存 `run.pid`、`run.log`、
`config.json`、可读 `code_version.txt` 与 `metrics.json`。不执行任何摘要校验操作。

## 结果与结论边界

权威输出 `runs/20260831_pair_alignment_attribution/main/metrics.json`；独立 post-run
逐项重算审计 PASS。

- HMM aligned AP/ROC/within 为 `.647732/.836151/.659823`；aligned within 比 16 个
  shift 的均值高 `.102665`，16/16 shift 更低，best shift 仍低 `.003629`。低/high
  base-tie 两半的 aligned-minus-shift 分别为 `+.121868/+.083004`，贡献不依赖 ties。
- HCS aligned 为 `.629456/.617913/.595685`；aligned within 比 shift 均值高
  `.036685`，14/16 shift 更低，但 best shift 反而高 `.008263`，且 6/16 shifted
  controls 仍 all-SOTA。低 base-tie 一半 aligned-minus-shift 为 `-.000989`，高 tie
  一半为 `+.075501`；收益高度集中于 VERA plateau/tie refinement。

两语料按预注册数值 gate 均 PASS。`metrics.json` 中的
`LEXICAL_TIMING_LOAD_BEARING` 只是“aligned 相对冻结 phase controls 有平均贡献”的
操作性 gate 标签，不表示 all-SOTA 必须依赖正确 lexical timing。允许结论：HMM 有强、
跨 tie strata 的 aligned timing evidence；HCS 是高-tie视频上的有利 aligned
tie-breaking。禁止声称 aligned 是 HCS 最优时序、shift 后不再 SOTA、作用遍及 HCS，
或由此批准 blend/distillation/一般 lexical 机制。
