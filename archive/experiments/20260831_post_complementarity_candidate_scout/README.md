# 淘汰：post-complementarity candidate scout

截至 2026-08-31。固定 signal complementarity diagnostic 之后，先后审查三个具体候选，
均未进入实现：

1. asymmetric conflict-projected local-order MIL：`STOP 3.3/3.8`；
2. SDR-style heterogeneous blend distillation：`STOP 3.4/3.4`；
3. within-between fixed-effects dispersion MIL：`STOP 3.1/3.6`。

随后两份独立 scout 均返回 `NONE`。cross-video positive recurrence / co-localization
是唯一接近剩余边界的家族，但其核心已被 WTAL temporal co-attention、cross-video
contrast 和 sequence co-localization 占用；常量时间特征仍产生整段 constant score，
低熵匹配只会任意选择时间，不能排除 broadcast。另一项 active-speaker-conditioned
prosodic escalation 仅 `3.9/10`：anger 不是 hate，差分重回 change/equivariance，
absolute affect 又恢复 broadcast，且 HateClipSeg speech coverage/机制不成立。

当前不是缺少另一个 loss，而是观测识别信息不足。只观测 `(x_v1:T, y_v)` 时，正视频
全秒为正与任意单秒为正可以产生相同 bag observation。要区分两者，下一轮必须明确
引入至少一种尚未被失败链否定的局部信息或可检验结构假设；继续改 pooling、attention、
gradient wrapper、centering、distillation 或 generic contrastive loss 不获准。

本目录只记录 scout 结论；未实现、未训练、未生成 prediction。
