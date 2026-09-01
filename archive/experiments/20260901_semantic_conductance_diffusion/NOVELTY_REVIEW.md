# Novelty review

截至 2026-09-01。Verdict：**STOP，3.7/10**。

允许 adaptation，且未发现 GAD/anisotropic diffusion 用于 hateful-video detection/localization。第三门失败：原 Daudt et al. GAD 已使用内容决定 conductance、多个 guide 取 minimum、边缘阻断传播；当前仅替换时空轴和 guide 类型，仍是 generic content-aware smoothing，与项目已关闭 semantic-neighbor propagation、policy-gated recurrence、fixed smoothing 链同构。禁止实现。
