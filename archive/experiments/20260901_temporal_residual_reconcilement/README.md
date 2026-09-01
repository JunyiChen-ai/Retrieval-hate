# 淘汰：Temporal Residual Reconcilement MIL

淘汰原因：完成 HMM/HCS 各 12 个 validation-only hyperparameter trials、validation-selected checkpoint、matched cyclic control 与正式 test 后，core 未通过机制门或 performance 门。权威汇总为 `runs/20260901_temporal_residual_reconcilement/formal_val_selected_seed234/summary.json`。这是 RESET4 的第一次 result-relevant formal failure。

截至 2026-09-01。RESET4 candidate 2。无 premise；现有四语料 test error analysis 直接支持 failure。

## Failure

权威 artifact 为 `runs/20260831_multihateloc_test_error_analysis/main/metrics.json`。MultiHateLoc 同步训练时，DMS 几乎总把最大权重给 visual，但它与 test-GT 最佳单模态的匹配率在 HMM/EN/ZH/HCS 仅 `.216/.333/.375/.323`；fused 胜过全部单模态的比例仅 `.345/.159/.042/.154`，best-branch oracle 相对 fused within 缺口为 `.106/.171/.211/.106`。这支持一个跨语料共同 failure：同步 fused loss 让较快收敛模态控制共享决策，其他模态虽各自存在有用局部排序，却没有被训练成对当前 final predictor 的纠错项。

## Cross-task source

来源为 ReconBoost（Hua et al., ICML 2024），它在普通 sample-level multimodal classification 中交替更新一个 modality learner，并以 reconcilement objective 使当前 learner 纠正历史 modalities 的 residual；每个模态只保留最新 learner，不累积 boosting ensemble。来源实验覆盖 CREMA-D、AVE、ModelNet40、MOSI、MOSEI、SIMS。当前检索未发现 ReconBoost 已用于 hateful-video detection/localization。

来源：[ICML paper](https://arxiv.org/abs/2405.09321)；[official code](https://github.com/huacong/ReconBoost)。

## Non-trivial task adaptation

本方法不是把 sample classifier 直接换成 video classifier。三个 MultiHateLoc modality branches 各自产生逐秒 logit `z_m(t)`，唯一 final score 为 `sigmoid(sum_m z_m(t))`。训练以 modality 为 stage 交替更新；其他 branches 冻结。对当前 modality `m`，先由冻结的 `sum_{j!=m} z_j(t)` 经过同一个 top-K MIL functional 得到当前 bag prediction，再把 bag BCE 对每个时间 logit 的负梯度定义为 **temporal witness residual**：positive bag 的正 residual 只落在冻结模型当前证据不足、但 active branch 能进入 top-K witness competition 的秒；negative bag 则对全部被 active branch 抬高的 witness 产生负 residual。Active branch 直接拟合该 signed temporal residual，同时 final additive score 接正常 bag BCE。

与 ReconBoost 的 task delta 是：普通分类每个 sample 只有一个 residual；这里 bag label 必须通过 non-smooth top-K witness operator 分配到秒，方法把“哪个模态纠错”与“在哪些秒纠错”合成一个交替 functional-gradient step。冻结 branches 的局部 logit 留在最终和式中，因此每次 correction 都是 final score 的 load-bearing 部分，而不是 auxiliary branch、gradient scaling、router、后处理或 teacher distillation。每个语料只用自己的 train video labels；validation 只选 stage checkpoint；test labels 不进训练或选择。

最小实现固定 `visual -> audio -> text` 顺序，每个 stage 一个 epoch，共运行 30 个完整 cycle（90 epochs）；只允许在完整 cycle 末用 validation 选择 checkpoint，不按数据集改变 modality order。为避免旧模型集合变成 ensemble，每个 modality 始终原位更新且只保留一个当前 branch；最终是一个三分支 additive network。非 active branches 不只冻结参数，也切到 eval mode，保证其 dropout 不改变 historical predictor。

## Falsification and matched control

Matched cyclic control 使用完全相同的三个 branches、additive final score、top-K MIL、`visual -> audio -> text` active/frozen schedule、总 optimizer step 数与 validation checkpoint selection，但不拟合 leave-other-modalities-out temporal residual。Core 唯一机制差异是 active modality 是否拟合冻结 peers 的 exact top-K bag-BCE negative functional gradient。普通 MultiHateLoc seed-234 作为外部 anchor，不承担该窄机制归因。

HMM/HCS test 上 core within 必须同时胜 matched cyclic control 与 MultiHateLoc anchor，且至少一边 `>= +.020`；否则机制失败并归档，不改 modality order、stage 数、residual temperature 或 top-K 续命。最终晋级仍要求两语料 pooled AP、pooled ROC、within ROC 全部越过固定 SOTA 门。

## Novelty gates

独立 verdict：`GO 7.1/10`。Gate 1 PASS：ReconBoost 可 adaptation。Gate 2 PASS：未检出 ReconBoost 或等价 additive residual-fitting 已用于 hateful-video detection/localization；TANDEM 已占用宽泛的 alternating multimodal training claim，但没有 additive frame logits、top-K weak MIL 或 signed temporal functional residual。Gate 3 PASS：bag error 经 non-smooth top-K functional 变成 per-second correction，并直接写入 final additive score，不是 sample classifier 的直接时间展开；它也不同于旧 Witness-DGM 的单一 competence scalar 与梯度幅度调制。

最窄 novelty claim：将 ReconBoost 的 sample-level modality reconcilement 改造成弱监督 hateful-video localization 的 alternating temporal functional-gradient mechanism。不能 claim 首次在 hateful video 交替训练 modalities，也不能声称 residual 秒是真实 span owner。

## Validation selection 与正式结果

每个语料独立运行 12 个 validation-only trials，搜索 `learning rate × residual weight × top-K`；validation within-video ROC 同时选择配置与完整 cycle 末 checkpoint。完整 ranking 位于 `runs/20260901_temporal_residual_reconcilement/val_search/{hatemm,hateclipseg}/selection.json`。

- HMM 选择 trial 08：`lr=3.698304456952196e-05`、`lambda_residual=.5`、`K=8`、epoch 30，validation within `.633116`。
- HCS 选择 trial 06：`lr=9.095411152325318e-05`、`lambda_residual=.25`、`K=3`、epoch 24，validation within `.552400`。

锁定各自配置后重训 core 与相同配置的 cyclic control，并立即 test。三项顺序为 pooled AP / pooled ROC / within ROC：

| corpus | cyclic control | temporal residual core | core within vs control | core within vs seed-234 MultiHateLoc |
|---|---|---|---:|---:|
| HateMM | `.502215/.745149/.603935` | `.450790/.706927/.619012` | `+.015077` | `-.009444` |
| HateClipSeg | `.564524/.536387/.537004` | `.523496/.491493/.527727` | `-.009277` | `+.004026` |

Core 在 HMM 改善 matched control 的局部排序，但未胜 MultiHateLoc、效果量不足且 pooled 明显下降；在 HCS 反而输给 matched control。机制门与两语料 all-SOTA performance 门均失败。按预注册不继续调 stage order、cycle 数、loss 或 K；该 family 在一次正式失败后仅允许由新的 post-test error analysis 直接支持的一次 corrective iteration，否则关闭。
