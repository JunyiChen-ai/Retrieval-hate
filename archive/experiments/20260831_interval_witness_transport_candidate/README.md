# 淘汰：Interval Witness Transport

截至 2026-08-31。独立查新仅 `conditional GO`、novelty `5.4/10`；未训练。
现有零训练前提已经触发硬否决：HCS path 反向，HMM fixed center-first
`.76550` 高于 path `.62866`，且未归一化 rasterizer 有长边/中心偏置。组合未见直接
先例不足以覆盖 performance/attribution failure，见 `NOVELTY_SCOUT.md`。

## 唯一候选核心

保留 corpus-specific POWA 的 PEF 与 AWB。AWB 的 transport plan `P[i,j]` 表示
timestamp `i` 的 hostile predicate 与 timestamp `j` 的 protected target 构成异步
witness。现有 POWA 只把 `P[i,j]` 返还到两个端点。候选将每条 edge 的 witness mass
可微地 rasterize 到闭区间 `[min(i,j), max(i,j)]`，得到 interval occupancy；该
occupancy 进入 policy compiler 和 dense score，使一个跨时刻成立的关系产生区间值
定位，而不是两个孤立端点。

拟议 novelty claim 仅限：**weak hateful-video localization 中，把 policy-typed
asynchronous optimal-transport witness edges 显式变成 interval-valued dense evidence**。
不 claim Sinkhorn/OT、span pooling、VLM teacher、knowledge distillation、POWA 原模块
或 score transport。

## 已知 feasibility 与缺口

零训练 ordering upper bound：
`runs/20260831_powa_witness_path_probe/analysis.json`。HateMM path mass within
`.57193→.62866`，endpoint `.51929`；HCS path mass `.49878`，低于 POWA `.52707`。
所以现有 HCS AWB 没有可用 path signal，candidate 不能直接训练一个 readout 就宣称
双语料成立。

若 novelty review 允许，HCS 的唯一前置尝试是增加**同语料 train-only、无人工 span
label**的 dense primitive supervision，让 PEF/AWB 学到局部 hostile/target/context
evidence。该 teacher 是 enabling supervision，不是贡献。必须在 HMM/HCS 用完全同一
teacher protocol；四个主数据集仍各自独立训练。

## 最低 controls

- 原 POWA endpoint readout；
- dense primitive supervision + endpoint readout（隔离 teacher 增益）；
- interval occupancy + 原 sparse supervision（隔离 rasterizer）；
- 完整 candidate；
- edge-time shuffle、edge direction shuffle、same-time/diagonal plan；
- fixed center-first / edge-first matched readout（HMM validation 的 center-first
  upper bound `.76550`，高于当前 path `.62866`，必须排除 AWB distance/position prior）；
- path length-normalized readout；
- 相同 POWA anchor、训练预算、evaluator 与 val checkpoint rule。

只有完整 candidate 在 HMM/HCS 都过 within `+.020`、pooled feasibility，并显著胜过
两个单因素 arms，才进入 test；否则归档。不得按语料切换 endpoint/path 或 teacher。
