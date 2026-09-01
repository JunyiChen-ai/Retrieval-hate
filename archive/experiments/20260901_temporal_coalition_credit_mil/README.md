# 淘汰：Temporal Coalition-Credit MIL

截至 2026-09-01。已完成独立 novelty、唯一 technical review、每语料 14 个 validation trials、HateMM/HateClipSeg 正式 test 和一次聚焦 test error analysis；机制门与 SOTA 门失败，family 关闭。

## Failure

依据 `runs/20260831_multihateloc_test_error_analysis/main/metrics.json`：当前 MultiHateLoc reimplementation 的 video-global DMS 在 HateMM/HateClipSeg 分别有 252/255、198/201 个 eligible seed-video 选择 visual，但与 test-GT 最佳单模态匹配率只有 `.216/.323`；best-branch oracle 相对 fused 的 within 缺口均约 `.106`。当前实现没有学习“哪个模态在当前秒提供 hateful witness”的责任。原论文对 DMS 公式未公开完整细节，因此不对所有 MultiHateLoc 实现泛化这一诊断。

## 跨任务来源

Wei et al., *Enhancing Multimodal Cooperation via Sample-level Modality Valuation*, CVPR 2024：用 Shapley coalition value 估计每个样本的模态贡献，并据此针对性训练低贡献模态。来源任务是一般 audio-visual recognition，不是 hateful video detection/localization。来源：https://openaccess.thecvf.com/content/CVPR2024/html/Wei_Enhancing_Multimodal_Cooperation_via_Sample-level_Modality_Valuation_CVPR_2024_paper.html

## Task adaptation delta

不照搬“低贡献模态重采样”。弱监督 localization 中，低贡献模态可能本来就是该秒的 benign carrier，强制均衡会复现已关闭的 capacity/gradient balancing。这里以零 embedding 表示 coalition 中缺失的模态，对三模态的 8 个 coalition 在每个候选 witness 秒精确重算原 fused logit，得到 stop-gradient signed Shapley marginal contribution；target 映射预先固定为 positive-part 后归一化，全非正时回退该视频原 DMS 权重，不扫描截断或温度。只在正视频的原 fused top-K witness 秒监督 responsibility。一个共享 local router 从各模态当前秒 embedding 预测 time×modality responsibility。

## 进入 final score 的路径

保留原 MultiHateLoc branches、DMS、fuse、fuse head、四分支 MIL、smoothness、contrastive 及官方 seed-234 逐语料 schedule（HateMM 50 epochs、HateClipSeg 100 epochs）。令机制强度 `alpha` 同时控制局部责任监督和输入融合：`w_time=(1-alpha)*w_global + alpha*w_local`，再按原方式缩放各模态 embedding，经过原 fuse/fuse-head 输出唯一 raw fused score。`alpha=0` 时不读取 local router，forward、raw score与base loss精确退化同一 harness 的原 MultiHateLoc。

## 可证伪 test 预期与 matched control

固定 prediction：validation 选定配置后，core 相对同 harness `alpha=0` control 必须在 HateMM/HateClipSeg 的 test within-video ROC 都提高，且 pooled AP/ROC 不明显下降；同时局部 router 的 argmax 与原始 aligned coalition-credit argmax agreement 必须高于 shifted-control。Shifted-control 对每个长度大于 1 的正视频把 target 循环平移 `max(1,floor(T/2))` 秒，排除零位移；除此之外模型、容量、训练量、搜索空间和 checkpoint selection 完全相同。两个 arm 的 agreement 都相对同一份未平移 aligned credit 计算。任一语料 within 不升、pooled 明显下降，或 aligned responsibility 不胜 circular control，即否定当前机制。

Validation search 预注册为每语料 14 trials：两个 `alpha=0` anchor trials，加上 `lr in {0.5*official_lr,official_lr}` × `alpha in {.25,.5,.75}` 分别用于 core 与 shifted-control。每个 trial 在 validation 上选择 checkpoint；每个 arm 再联合选择超参数与对应 checkpoint。选择以 validation within-video ROC 为主，要求 validation pooled AP/ROC 相对同学习率 `alpha=0` 不下降超过 `.01`；若无配置满足约束，则先最小化两项最大退化、再最大化 within，仍必须锁定一个配置并立即 test，不把 validation 当 test 前晋级门。Core 与 shifted-control 独立使用完全相同选择规则。锁定后立即进行 HateMM/HateClipSeg test 三指标 evaluation。Validation 不用于修改机制。

独立 novelty review：`NOVELTY_REVIEW.md`，裁定 `GO 6.8/10`。

正式运行前唯一 technical review：`TECHNICAL_REVIEW.md`，裁定 `PASS`。

## 正式结果与去向

权威汇总：`runs/20260901_temporal_coalition_credit_mil/formal_val_selected_seed234/summary.json`。

- HateMM anchor/aligned/shifted 的 AP/pooled ROC/within 为 `.490302/.737340/.632938`、`.490413/.739038/.633645`、`.487623/.744732/.634007`。Aligned 相对 anchor 三项仅 `+.000110/+.001698/+.000707`，within 反而低于 shifted `.000363`。
- HateClipSeg 三臂为 `.523714/.497501/.525970`、`.527508/.500584/.528212`、`.527096/.499777/.527893`。Aligned 相对 anchor 三项 `+.003794/+.003083/+.002242`，相对 shifted within 仅 `+.000320`。
- 两语料都没有达到三指标 SOTA；HMM 没有通过 time-alignment matched control，完整机制门失败。

Developmental test error analysis：`runs/20260901_temporal_coalition_credit_mil/formal_val_selected_seed234/test_error_analysis.json`。高 branch-oracle-gap 视频的 aligned-minus-anchor within 在 HMM/HCS 为 `+.007255/+.004801`，低 gap 为 `-.005689/-.000242`；但高 gap aligned-minus-shifted 在 HMM/HCS 为 `-.003701/+.000125`，不支持正确 coalition 时间对齐是共同增益来源。因此不消耗 Rule 18 corrective；关闭 alpha、credit target、shift schedule、router width 与 disagreement-gating 变体。

正式链路在 HateMM validation 完成后暴露一次纯 inference import/cohort bug：错误地导入 baseline `train.py`，随后 test label loader又要求原 manifest与 evaluator cohort等长。两处均在生成任何候选 test prediction前修复；14个validation trial未重跑。修复后按 canonical 214-video HateMM cohort与79-video HCS cohort完成三臂 test。
