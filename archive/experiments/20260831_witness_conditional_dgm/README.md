# Witness-Conditional Dynamic Gradient Modulation

> **淘汰：已完成有效 HMM/HCS 双 test 方法迭代，但 mechanism 与 performance gate 均失败。** 权威 matched run 为 `runs/20260831_witness_conditional_dgm/pilot_seed234_matched/summary.json`。Core 相对 matched anchor 的 within 增益为 HMM `+.002672`、HCS `+.007208`，两边都胜 source-style DGM，但均未达 `+.020`；两语料全部三项 SOTA 失败。不得调 gamma、改 checkpoint rule或更换 competence 继续续命。

截至 2026-08-31。RESET2 epoch candidate `1`，已通过 novelty `GO 6.7/10`、完成唯一 technical review与正式双语料 test，现已淘汰归档。

## Failure

固定 test error analysis 显示 MultiHateLoc 的 video-global DMS 在 HateMM/HateClipSeg 的 eligible seed-video pairs 中分别 `252/255`、`198/201` 次选 visual，但 DMS 与 test-GT 最佳单模态匹配率仅 `.216/.323`；best-branch oracle 相对 fused within-video ROC 仍有 `.106/.106` 缺口。目标只处理这个已证实的训练期 modality competition / visual domination，不改变 top-K cardinality，不使用 lexical、teacher或test routing。

## Source

跨任务来源为 Fu, Gao, Xu, *Multimodal Imbalance-Aware Gradient Modulation for Weakly-supervised Audio-Visual Video Parsing*（2023）。其 DGM 根据 modality-separated decision unit 的视频/类别预测衡量 audio/visual 优化失衡，并调制各模态 encoder 的反向梯度。初步检索未发现 DGM/MSDU 用于 hateful-video detection/localization。

## Task adaptation delta

直接移植 source 的 video/category confidence 只会平衡 hateful-vs-benign bag classification，仍允许每个 positive video 整段高分。这里把 imbalance statistic 改成 binary hate localization 的 **witness-conditional competence**，但保留 source 的核心“只调制训练梯度、不改变 inference graph”。

MultiHateLoc 已有三个纯 modality branch，直接作为 MSDU；不新增 pseudo head。每个 batch、每个 modality `m` 用当前 branch probability计算一个 stop-gradient competence：

- positive bag：`C_pos,m = topK_mean(p_m) - mean(non_topK(p_m))`，衡量局部 witness 相对同视频其余秒的分离；
- negative bag：`C_neg,m = 1 - topK_mean(p_m)`，衡量最危险局部 false alarm 的抑制；
- `C_m` 是 batch 内两项的等权均值；只有一个类别缺失时使用存在项，不读取 span label。

沿用 source DGM 的 bounded modulation：高于三模态平均 competence 的 branch encoder gradient被平滑衰减，低 competence branch保持原梯度；同一系数同时作用于该 modality branch和其进入 fused head 的输入投影。系数只由当前 train batch与video labels产生并 stop-gradient，不由validation/test选择，不按corpus改变。Gaussian enhancement不纳入 core，避免第二机制。

最终训练 loss、模型容量和 inference graph均与 MultiHateLoc一致；test只输出 raw `score_fused`。DGM statistic、branch score或DMS weight都不参与test routing、ensemble或calibration。

## Final-score path, falsification, control

Gradient modulation直接改变三个 modality encoders，而它们是唯一 fused score输入，因此不存在 auxiliary-head旁路；它仍可能经验上无效，这由test裁定。

可证伪预期：相对 capacity-matched MultiHateLoc，两语料 fused within-video ROC 都提高，至少一边 `>=+.020`，且训练后 visual DMS monopoly下降；完整晋级仍要求固定四语料全部三项SOTA。

唯一 matched control：**source-style video-confidence DGM**，使用相同实现与梯度调制预算，但 competence 只取正确video label的branch bag confidence。若它等于或优于 core，则 temporal witness adaptation不是load-bearing，机制失败。

Novelty通过后立即实现；只做一次 technical review，随后 seed 234 在 HateMM/HateClipSeg 独立训练，validation在各固定arm内部选配置/checkpoint，训练后立即test pooled AP、pooled ROC、within-video ROC。方法test后再检查 gradient coefficients、branch top-K concentration、DMS selection与 source-style control；不新增 premise。

实现后的首次正式运行误用了 baseline 脚本默认超参数，而不是权威 official-validation seed-234 配置：HMM 实际应为 `hidden=512, embed=64, dropout=.05, K=8, lr=1.849152228476098e-05, epochs=50, lambda_smooth=.01420807210603241, lambda_contrast=.18733857665415116, temperature=.07`；HCS 应为 `hidden=512, embed=256, dropout=.05, K=3, lr=.00018190822304650636, epochs=100, lambda_smooth=.10337306075094418, lambda_contrast=.03728675834293724, temperature=.03`。错误配置的 `runs/.../pilot_seed234/` 仅保留为非权威 implementation diagnostic；matched rerun 使用 `runs/.../pilot_seed234_matched/`，三 arm 在每个 corpus 内完全共享对应冻结配置。

来源：[Fu, Gao, Xu, *Multimodal Imbalance-Aware Gradient Modulation for Weakly-supervised Audio-Visual Video Parsing*](https://arxiv.org/abs/2307.02041)。

## 正式结果与失败归因

权威 evaluator 输出位于 `runs/20260831_witness_conditional_dgm/pilot_seed234_matched/{hatemm,hateclipseg}/{anchor,source_dgm,witness_dgm}/metrics.json`，汇总为同目录 `summary.json`。三项顺序为 pooled AP / pooled ROC / within ROC：

| corpus | anchor | source DGM | witness DGM core |
|---|---|---|---|
| HateMM | `.492997/.738259/.628463` | `.494409/.740013/.627138` | `.497713/.742135/.631135` |
| HateClipSeg | `.551339/.542726/.520588` | `.515255/.501630/.525999` | `.517239/.507104/.527796` |

Core在两语料within均胜两个controls，说明 witness-conditional competence 比普通video-confidence DGM方向更对；但最大core-vs-anchor增益只有`.007208`，未过机制效果量门，六项performance均未过SOTA。

Developmental post-test分析为 `runs/20260831_witness_conditional_dgm/pilot_seed234_matched/mechanism_analysis.json`。HMM core-vs-anchor逐视频fused score Spearman均值`.997627`，gradient modulation几乎没有改变最终ranking；HCS为`.539573`且平均绝对差`.125483`，但core合规validation checkpoint为epoch 9、anchor为epoch 64，core pooled AP/ROC大幅下降。两语料都不支持通过调gamma或改变checkpoint选择继续当前机制。

首次 `pilot_seed234/` 误用baseline默认配置，不是权威结果；配置不一致在正式结论前被发现并用matched run完整重跑。最终结论只认matched run。
