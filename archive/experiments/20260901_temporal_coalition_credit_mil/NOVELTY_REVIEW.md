# Independent novelty review

截至 2026-09-01。裁定：**GO — 6.8/10**。

三门结论：Wei et al. CVPR 2024 的 sample-level Shapley modality valuation 可 adaptation；在 CMFusion、HVGuard、SAGE 与 MultiHateLoc 等最接近 hateful-video 方法中未检出以 coalition marginal credit 作为训练责任信号；当前从完整分类样本与低贡献模态重采样，改造成 positive top-K witness 秒的精确 coalition credit、并直接监督进入 final fused score 的 time-local responsibility，属于 non-trivial task adaptation。

最窄 claim：在仅有视频级标签的 hateful temporal localization 中，以 positive top-K witness 为条件计算逐秒模态 coalition credit，并用该 credit 监督直接控制最终融合分数的 time-local modality responsibility。

不得声称首次 time×modality routing、首次 dynamic modality selection、首次 adaptive modality fusion、首次在 hateful-video 中使用 Shapley/coalition，或 causal ownership。

主要风险：zero-mask coalition 的分布外效应、自确认 credit、错误 top-K 峰值强化与 signed-credit 映射。按 Rule 12，这些交给正式双 test、`alpha=0` control 与保持训练量一致的 within-video circular-shift control 裁定，不构成 pre-run STOP。

来源：

- https://openaccess.thecvf.com/content/CVPR2024/html/Wei_Enhancing_Multimodal_Cooperation_via_Sample-level_Modality_Valuation_CVPR_2024_paper.html
- https://arxiv.org/abs/2505.12051
- https://aclanthology.org/2025.emnlp-main.456/
- https://aclanthology.org/2026.acl-long.817/
- https://arxiv.org/abs/2512.10408
