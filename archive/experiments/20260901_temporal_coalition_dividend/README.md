# Temporal Coalition-Dividend MIL

截至 2026-09-01。**Novelty STOP 4.8/10，未实现、未训练、未生成 prediction。** 当前正式失败计数保持 `2/3`。

Gate 1/2 PASS：允许 adaptation Harsanyi dividend，且未发现用于 hateful-video detection/localization。Gate 3 FAIL：该方法与已失败的 `mobius_nonminimal` 严格同族——相同 7 个 modality coalitions、shared masked forward、正 interaction 聚合为唯一帧分数和 temporal MIL；这里只把非负 atoms 改写成精确 Möbius inversion 后截断负 dividend，没有新增局部观测或约束。旧正式 HMM/HCS within 已为 `.6338/.5365`，故不能换参数化重开。

## Failure

MultiHateLoc 的 global DMS 与 test-GT best modality 匹配率仅 HMM/HCS `.216/.323`，best branch oracle 相对 fused within 缺口均为 `.106`。Witness-DGM 只改 gradient，Temporal Expert-Choice 强制 modality capacity；前者几乎不改 final ranking，后者把弱模态强制写入 final score。缺失的不是更多 balancing，而是区分“单模态独立证据、真正跨模态协同、加入后反而有害”的局部 final-score责任。

## Source and adaptation

跨任务来源是 Harsanyi dividend / causal-pattern decomposition：对同一输入的全部 feature coalitions 做 Möbius inversion，把完整预测唯一分解为 singleton 与 higher-order interaction effects。来源用于解释 DNN 概念交互，不是 hateful-video detection/localization。

这里把 players 从普通 input variables 改为同一 1fps second 的 visual/audio/text modalities，并把 post-hoc decomposition 改为训练时唯一 localizer：一个共享 coalition scorer 对 7 个非空 modality subsets 输出 frame utility；逐秒精确 Möbius inversion 得到 3 个 singleton、3 个 pair、1 个 triple dividend。Core 的唯一 frame evidence 只累积正向 dividend，负向 dividend不能抵消其他 coalition 已提供的 hate evidence；negative bags 的同一 final MIL 迫使所有伪正向 dividend下降。Test只输出该单模型 raw final score。

这不同于 forced balance：没有 modality quota、router或逐branch正标签；一个 modality 若只会降低 coalition utility，其 dividend 不进入正 evidence。它也不同于普通 fusion：pair/triple 只有在超出全部 subcoalition utility 时才贡献。

## Control and falsification

Matched control 使用完全相同的 7 次共享 coalition forward 与参数，但保留 signed dividends；由 Möbius efficiency 它精确重构 grand-coalition utility，相当于 direct fusion。Core 与 control active computation/参数完全一致，只改变“负 interaction 是否可抵消已存在的局部 hate evidence”这一机制。

正式运行前只做 Rule 12 novelty verdict 与一次 technical review；随后用权威 per-corpus MultiHateLoc validation-selected训练配置分别训练 HMM/HCS，并立即 test 三项固定指标。Core 必须在两语料 within 同时胜 anchor 与 signed control，至少一边 `>=+.020`，否则失败计数到 `3/3` 并立即触发 process review。

来源：Ren et al., *Defining and Quantifying the Emergence of Sparse Concepts in DNNs*, CVPR 2023。
