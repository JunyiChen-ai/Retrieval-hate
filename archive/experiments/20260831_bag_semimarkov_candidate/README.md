# STOP — Bag-constrained explicit-duration semi-Markov localization

截至 2026-08-31。未实现、未训练。

候选拟适配 neural semi-Markov CRF/HSMM：negative video 只允许全-benign path；positive video 对所有至少含一个 hateful segment 的 path 做 forward-DP 精确边缘化；test 输出单模型 hateful-state frame marginal。

四语料 test GT 的相邻同类概率相对同基率随机排列均为正（HMM/EN/ZH/HCS 约 `.232/.161/.241/.333`），所以 span continuity 是真实共同结构。但这不等于 video labels 能识别 duration 或 path。

两路独立审查均 STOP（`3.8/10`、`3.4/10`）。目标任务占用窄门通过，nontrivial mechanism 门失败：positive lattice 仍包含整段 `H^T`；常量正视频 emission 可使 bag likelihood 近完美但 within `.5`；emission 与 duration potential 存在线性重参数化不识别；精确 DP 只正确求和，不提供局部方向。项目既有 exact-partition CRF 与 fixed smoothing 负结果也不支持新增 explicit-duration 成为 load-bearing。不得通过未验证的 max duration/occupancy penalty隐藏假设，不进入 pilot。

