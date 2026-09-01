# 淘汰：POWA witness-path feasibility probe

截至 2026-08-31。validation-only、零训练的 POWA-native 诊断；不读 test。
`STOP_BEFORE_NOVELTY`：HateMM path-mass transport within `.57193→.62866`，但
fixed center-first 为 `.76550`，不能排除 position prior；HateClipSeg path
`.52707→.49878`。同一机制不具备双语料前提，HMM 归因也不成立。

POWA 的 AWB 学出 hostile↔target transport plan，但当前 dense witness 只把 edge
mass 返还给端点。本 probe 把每条 edge 的 mass 投影到两端之间的离散时间路径，检查
该 path occupancy 是否在 HateMM 与 HateClipSeg 都提供局部排序信号。为避免把
calibration 当方法，只把 frozen POWA score multiset 按各只读信号重排作为 ordering
upper bound；任何结果都不是候选方法性能。

只有同一个 `path_mass` rule 在两个语料都比 frozen POWA within ROC 高至少 `.020`，
且比 endpoint readout 高至少 `.010`，才值得进入独立 novelty review。否则归档。
