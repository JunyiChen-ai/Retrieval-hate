# 仅 upper bound：semantic-neighbor propagation

截至 2026-08-31。validation-only、零训练；未读 test。直接 graph propagation +
Gaussian smoothing 是 VERA/label-propagation 类已知 inference calibration，按 Rule 4
不得成为主方法。本 probe 只检查同一个冻结 rule 是否提供双语料 training-target
feasibility。权威输出：`runs/20260831_semantic_neighbor_probe/analysis.json`。

固定 rule：同语料 train-bag-label concat local probe score，经同视频 CLIP visual
top-15% neighbor、softmax temperature 10 传播，再做 VERA 固定 `sigma=10,radius=7`
smoothing；只用该 ordering 重排 frozen POWA score multiset。

结果：HMM within `.57193→.59515`，HCS `.52707→.55867`，两者均超过 `+.020` 且
pooled feasibility 保持。它仅证明 semantic recurrence + temporal persistence 含有跨
语料可用 ordering signal；不能把这些数字当候选 performance，下一方法也不能直接
提交该 readout、ensemble 或 calibration。
