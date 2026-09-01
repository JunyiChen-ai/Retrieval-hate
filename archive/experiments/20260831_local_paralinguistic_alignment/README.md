# Local paralinguistic alignment premise

截至日期：2026-08-31。当前阶段：冻结 test diagnostic 已完成；双语料 premise 失败，方向关闭。

## 问题

历史 HateClipSeg matched-head control 显示，Wav2Vec-Emotion 特征在同视频内 circular shift 后性能下降，提示 learned paralinguistic state 可能携带 moment-level information；但该证据使用旧 split/metric，HateMM 也没有同一 probe。这里用当前固定 1fps test evaluator回答一个问题：同一 train-video-label linear readout 的局部 audio state，在 HateMM 与 HateClipSeg positive videos 内是否都与 hate seconds 对齐。

## 冻结设计

- 输入固定为 `dense4fps_w2vemo`，每个整数秒读取索引 `4*t`；score 长度由冻结、无标签的 `clip_b16_1fps` 行数定义，与共享 evaluator 的 audio-clock 网格一致。raw container duration 只作差异审计；末尾不足时重复最后一个有效 feature并报告数量。
- 每个语料只用自身 train video labels。类内每视频等权、两类总权重相等，所有 frame 权重总和固定为 train 视频数；固定 StandardScaler + `C=1` logistic regression，无 validation、无超参数选择。
- producer 不读取 test GT label values；只读取冻结 archive 的 key names 以确定 exact evaluator cohort，并先写完整 test score arrays。
- evaluator随后读取 test GT，调用全仓库共享 evaluator输出 pooled AP、pooled ROC、within-video macro ROC。
- circular-shift control 对每个视频使用最多 32 个均匀、唯一的非零 shift；保持每视频 score multiset不变。三项 pooled/global control 指标报告 32 个均匀调度的联合 control 的均值和范围；gate 的 within shift mean 先在每视频内等权平均其 unique shifts，再对可算视频等权平均，避免短视频重复 shift 改变权重。

## 冻结 premise gate

两语料都必须同时满足：

1. original within-video ROC `> .52`；
2. original within-video ROC 高于 circular-shift mean 至少 `.020`；
3. test prediction coverage、长度、finite 与 evaluator contract 全部通过。

任一语料失败即关闭 paralinguistic alignment 方向，不做 audio encoder swap、feature concat、corpus-specific routing 或超参数扫描。通过只允许进入跨任务时序状态方法的 novelty review，不代表方法、SOTA 或晋级。

## 正式结果与结论

权威输出：`runs/20260831_local_paralinguistic_alignment/premise/metrics.json`。

| Corpus | pooled AP | pooled ROC | within-video ROC | shift mean | within − shift | Gate |
|---|---:|---:|---:|---:|---:|---|
| HateMM | 0.507914 | 0.776237 | 0.542472 | 0.500887 | +0.041585 | PASS |
| HateClipSeg | 0.534425 | 0.495490 | 0.512236 | 0.500201 | +0.012035 | FAIL |

HateMM 同时通过 `.52` 与 `+.020` 两门；HateClipSeg 两门都未通过。联合 verdict 为 `STOP_DIRECTION`。因此旧 HateClipSeg matched-head 证据没有在当前固定 split、1 fps evaluator 与相同 train-video-label readout 下复现，local paralinguistic alignment premise 被双语料 gate 淘汰；按冻结规则不继续该方向。
