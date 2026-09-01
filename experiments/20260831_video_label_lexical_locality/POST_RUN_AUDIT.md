# Post-run result-chain audit

截至 2026-08-31。独立 reviewer：`metrics_audit`。权威结果：
`runs/20260831_video_label_lexical_locality/premise/metrics.json`。

Verdict：PASS。只读复算与保存结果逐字段一致。

| corpus | pooled AP | pooled ROC | within ROC | speech within | shift within | gate |
|---|---:|---:|---:|---:|---:|---|
| HateMM | .536109 | .748662 | .632629 | .546803 | .505095 | PASS |
| HateClipSeg | .502332 | .476080 | .522700 | .508136 | .501199 | PASS |

HateMM 为 214 videos / 29,269 frames / 85 both-class positive videos，完成
1,346 个 per-video shift evaluations。HateClipSeg 为 79 / 18,839 / 67，完成
1,072 个。ID coverage、score length、finite、speech 二值性、producer report、
hate IDs、shift 次数和 gate 均核验一致。producer 只使用同语料 train scoped
labels；test GT 数值只在 evaluator 中读取。

结论只允许表述为：冻结的双语料 lexical-locality premise 通过，允许进入
novelty design。它不是方法或 SOTA 证据。HateClipSeg within 只比阈值高
`.00270`、比 speech 高 `.01456`、比 shift 高 `.02150`，且 pooled ROC `.47608`，
因此只能称为弱但符合冻结 gate 的 positive-video 内部 lexical signal。
