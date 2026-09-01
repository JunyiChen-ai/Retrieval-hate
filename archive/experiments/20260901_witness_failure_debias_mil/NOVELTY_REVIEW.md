# Independent novelty review — Witness-Failure Debiasing MIL

**截至 2026-09-01。只读审查；只裁定 novelty 与机制，不审代码。**

## Verdict

**GO，6.5/10。**

Rule 12 三门均通过，但 claim 必须保持很窄：

> 将 LfF 的有标签 sample-level relative-difficulty weighting 改造成弱监督 hateful temporal localization 中仅作用于 latent fused witness 的训练信号：用 GCE 单模态 bag experts 放大易学 shortcut，在正视频 fused top-K support 内按逐秒 unimodal-failure 程度重加权唯一 fused localizer；负视频则利用确定的负标签重加权 hard negatives。Test 只输出 raw fused score。

它不能主张学习了真实 modality ownership，也不能主张首次 modality expert、动态融合、hard-example weighting、GCE、MIL 或 debiasing。

## Gate 1：来源允许 adaptation — PASS

来源是 Nam et al., *Learning from Failure: Training Debiased Classifier from Biased Classifier*, NeurIPS 2020。其 load-bearing core 是同时训练两个分类器：用 GCE 令 bias model 更偏向 easy/bias-aligned samples，再用
`CE(bias,y)/(CE(bias,y)+CE(debiased,y))` 重加权 debiased classifier，使后者聚焦 bias model 难以解释的样本。来源实验是完整标签下的图像/动作分类，不是 hateful-video detection/localization，也不是 temporal MIL。

来源机制可被 adaptation；不要求本项目从零发明，符合 Gate 1。

来源：

- Nam et al. 论文与公式：[arXiv 2007.02561](https://arxiv.org/abs/2007.02561)

## Gate 2：来源是否已进入 hateful-video task — PASS

实际检索了以下精确组合：`Learning from Failure`/`LfF`/`relative difficulty`/`generalized cross entropy` 与 `hateful video`、`hate video`、`HateMM`、`HateClipSeg`。未检出 Nam et al. 的完整 core——GCE bias amplification 加 bias/debiased relative-difficulty weighting——已用于 hateful video detection 或 temporal localization。

同时核对最接近的 hateful-video 方法：

- SAGE 已占用 modality-specific experts、全局 deliberation 与 instance-level tribunal/gating，但没有 GCE bias expert 或 failure-relative training weights。
- HVGuard 已占用 multimodal encoders、MLLM rationale 与 MoE fusion，但训练仍是最终分类交叉熵，不是 biased/debiased pair。
- MM-HSD 已占用独立 modality encoders、cross-modal attention 与 late fusion；未使用 LfF relative-difficulty core。
- RAMF 已占用 local-global fusion、semantic cross-attention 与 contrastive reasoning；未使用 LfF core。
- MultiHateLoc 已占用逐秒三模态 branch、DMS 与 fused localizer，但没有主动训练 shortcut expert 后用其失败重加权 fused witness。

因此不能声称“hateful-video 中首次 experts/debiasing/fusion”，但精确来源机制的目标任务占用门通过。

相邻来源：

- [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.817/)
- [HVGuard, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.456/)
- [MM-HSD, ACM MM 2025](https://publications.idiap.ch/attachments/papers/2025/Cespedes-Sarrias_ACMMM25_2025.pdf)
- [Reasoning-Aware Multimodal Fusion](https://arxiv.org/abs/2512.02743)

检索未命中不是对全部未来或不可索引工作的保证；它足以支持本轮 Rule 12 的窄 occupation 裁定。

## Gate 3：是否为 non-trivial task adaptation — PASS（窄）

这不是把 LfF 的 sample weight 直接放到 video loss。来源中每个 sample 有真实类别，bias 与 debiased model 对同一个有标签 sample 计算 relative difficulty；当前任务的正视频没有秒标签。Brief 做了三个相互依赖的任务改造：

1. 把三个单模态 branch 变成只在 video-level top-K MIL probability 上接受 GCE 的 shortcut experts，而不是假造逐秒正标签。
2. 正视频只在 detached fused top-K latent witness support 内计算 failure weight；不把 video label 广播到全部秒。负视频才使用所有有效秒均为负这一确定监督。
3. Relative-failure weight 不进入 test ensemble或路由，而是直接改变 fused witness loss，唯一 test readout仍是 raw `score_fused`。

这组 delta 对应 STATUS 已证实的具体 failure：fused 很少胜过全部单模态，且现有 global modality choice 与最佳分支严重错配。可证伪机制是：被任一 easy unimodal shortcut 解释的 latent witness 被降权，而 unimodal experts 均难以解释、但 fused learner可学习的 witness 获得更大梯度。其作用对象是“是否有单模态 shortcut 可以解释该 fused witness”，不是“哪一个模态拥有该秒”。

来源所需的真实逐秒标签确实不存在，但这不是本轮的必然 STOP：candidate 没有偷偷广播该标签，而是把监督条件显式改成 fused latent support，并用 uniform-support matched control直接检验 relative failure 是否比普通 latent-witness training有效。错误 top-K、自确认与 bias expert 未真正吸收 shortcut 都是正式 test 风险，而不是 Rule 12 要求的实现前完整可识别性证明。

## 与 failure ledger 的边界

- **Gradient-only modality balancing：不严格同构。** 已关闭 DGM 用每个 modality 的 competence 调整对应 encoder gradient，主要改变模态间优化预算；本候选用逐秒 `max` bias failure 选择 fused witness 的训练强度。它可以改变时间排序，但不会产生显式 modality ownership。故允许运行的前提是 claim 收窄为 witness-level shortcut-conflict weighting，不能把 argmax branch 解释为 ownership，也不能在失败后改成 per-modality gradient coefficient 续命。
- **Capacity-forced ownership：不同。** 没有固定容量、assignment quota 或强制每个模态写入 final score；弱模态不会因负载均衡被硬塞入输出。
- **Self-coalition temporal credit：不同。** 不用 fused scorer 对自身 masked coalitions 的解释作为 router target；bias signal 来自另行接受 GCE bag objective 的单模态 branches。它仍可能自确认，但不是旧 coalition-credit 的代数重命名。
- **Teacher KD：不同。** Bias experts不提供 pair order、soft target或test ensemble；只产生 detached train-time loss weight。若后续蒸馏 branch logits/order，则会落回已关闭 teacher family，当前 GO 不覆盖。
- **Residual family：不同。** 不对 logits/features做 residual subtraction、alternating correction或inference additive reconcilement；最终结构仍是原 fused scorer。

最接近的项目内方法是 **witness-conditional DGM**，最接近的目标任务公开方法是 **SAGE**。前者接近于“branch competence 改训练动态”，后者接近于“modality experts 处理 feature dilution”；本候选区别于两者的唯一可主张部分，是 GCE shortcut experts 的失败程度在 latent temporal witness support 上直接重加权 fused learner。

## 不可变的 load-bearing constraints

GO 只覆盖以下完整机制，任一删除或替换都需要重新做 novelty 裁定：

1. 三个单模态 bias experts 必须以各自 video-level top-K MIL probability接受 GCE，不能退化为普通预训练 branch confidence。
2. Bias probability 与 relative-failure weight必须 detached；bias experts不能通过该权重接受来自 fused learner 的反向梯度。
3. 正视频 failure loss只能作用于 detached fused top-K witness support；禁止向正视频全时段广播正标签。
4. Weight 必须保持 source-faithful relative difficulty，bias项为同秒三个单模态正类 loss中“最容易解释”的等价聚合；不能在看 test 后换成 branch routing、learned gate、teacher order或 residual。
5. 负视频只利用其所有有效秒均为负的确定监督；不得把 test span、其他主数据集 span 或外部 pseudo-span带入训练。
6. 唯一 test 输出必须是原 fused branch 的 raw score；bias experts不得参与 inference ensemble、calibration或routing。
7. `lambda_failure=0` 必须精确退化同 harness MultiHateLoc anchor，包括 forward、raw test score、基础 loss 与 schedule。

## Matched control 与 falsification

唯一 matched mechanism control 应保持完全相同的 bias experts、GCE、参数量、优化预算、fused top-K support、训练 schedule、validation selection 和 test evaluator，只把 support 内的 relative-failure weights替换为 uniform weights。`lambda_failure=0` 另作 exact anchor，不替代 matched control。

机制成立至少要求 HMM/HCS test within-video ROC均胜 uniform control，且至少一个语料达到预注册的 `+.010`；core 的增益还应集中在原 anchor 的 `best unimodal > fused` 视频。若 matched uniform 不输、两语料方向不一致，或增益只来自 pooled scale，当前 family关闭；不得继续扫描 bias aggregation、GCE变体、witness producer、branch routing或 residual。

## 结论

该候选的来源方法可 adaptation，精确 LfF core 未检出已进入 hateful-video detection/localization；从完整标签 sample weighting 到“正袋 latent witness / 负袋确定逐秒负监督”的改变构成可证伪、会进入唯一 final scorer 的 task-specific adaptation。因此按 Rule 12 裁定 **GO，6.5/10**。一般 self-confirmation 风险留给正式 HMM/HCS test 与 uniform matched control，不继续追加 premise 或 novelty review。
