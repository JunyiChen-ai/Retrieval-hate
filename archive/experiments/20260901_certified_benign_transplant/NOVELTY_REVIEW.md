# Certified benign temporal transplant：独立 novelty review

截至 2026-09-01。性质：本候选唯一一次跑前 novelty review；只审机制与先例，不审代码。

## 裁定

**STOP，1.8/10。** 不得实现或训练当前候选。

原因不是 temporal transplant 在外部 hateful-video 文献中已被明确发表，而是当前 brief 与本项目已经在 hateful video localization 上正式训练、test 并淘汰的
`archive/experiments/20260831_powa_benign_insertion_pilot/` **逐项同构**。当前候选没有提出区别于旧方法的新约束、新可用信号或新推理路径；它只是给旧方法换了名称并重新安排相同的 validation search。Rule 12 的 non-trivial adaptation 门和 Rule 21 的已关闭失败链门均失败。

## 三门逐项 verdict

### Gate 1：允许 adaptation 已有方法

**PASS。** 候选可以从 CutPaste、CutMix、Temporal VideoMix、segment shuffling 等跨任务方法出发，不要求算子从零发明。

这项 PASS 只说明“可以 adaptation”，不说明当前 adaptation 新颖。

### Gate 2：来源方法不得已用于 hateful video detection / localization

**FAIL（项目级硬失败；外部公开文献窄口径未检出直接占用）。**

公开检索覆盖了 `hateful video detection/localization + CutPaste / copy-paste / CutMix / temporal transplant / segment mixing / benign insertion`。截至检索日：

- HateClipSeg、LELA、CLARA、HVGuard、SAGE、CRAVE/“Borrowing Eyes”等公开 hateful-video 工作中，未检出“negative-train 连续块移入 positive train video，并仅对移植区施加 benign frame loss”的公开方法。
- CRAVE 的 hateful-video “temporal augmentation”只是改变采样间隔/帧率，不是跨视频片段移植，也不产生局部 benign mask。

但本项目的 `powa_benign_insertion_pilot` 已经在 HateMM/HateClipSeg 上使用了这个完整机制并完成正式 test：negative-train 连续多模态 donor 插入 positive recipient；donor interior 施加 dense benign target；未改 recipient 做 prediction consistency；原 POWA MIL 保留；test 输出原始 POWA frame score。它不是未实现的 scout，而是已经进入目标任务的正式方法。因此按本项目 novelty gate，“该 adaptation 未被 hateful-video task 占用”不成立。

即使把 Gate 2 极窄地只解释为“外部发表文献是否占用”，它最多只能记为窄 PASS；Gate 3 和 Rule 21 仍独立构成 STOP。

### Gate 3：task adaptation 必须 non-trivial，且不得与关闭失败链严格同构

**FAIL。** 当前 brief 与旧正式方法的映射如下：

| 当前 brief | 已淘汰 `powa_benign_insertion_pilot` |
|---|---|
| negative train bag 的所有有效秒是 label-certified benign | 同一 certificate |
| 连续、时间对齐、全模态 donor block | 同一 donor 定义 |
| donor 移入 positive train sequence 的随机位置 | 同一 intervention |
| 只在 donor interior 施加 benign frame loss | 同一局部监督 |
| mask boundary 不参与监督 | 同一边界排除 |
| 未移植区域约束与原路径 score 一致 | 同一 consistency |
| 原始视频继续 POWA bag loss，合成路径不伪造正 span | 同一 MIL/latent 语义 |
| inference 单次原 POWA forward，输出 raw `frame_prob` | 同一 readout |

这不是与旧方法“相邻”，而是方法定义相同。旧正式结果已经给出结构性失败：HateMM within 提高并过门，但 pooled AP/ROC 分别明显下降；HateClipSeg 三项只小幅改善且仍未过门。当前 brief 没有引用新的跨语料 correction signal，也没有提出会改变这条 pooled-versus-within trade-off 的机制差异。把 loss weight 与替换比例做更大的 validation search 属于同一机制调参，不构成 Rule 18 corrective，更不构成新的 novelty。

## 外部相邻先例比较

以下一手来源进一步说明，copy-paste/transplant 算子和“通过合成时序结构获得局部监督”的上位范式均已拥挤：

1. [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/abs/2104.04015)：从 normal 图像剪贴局部 patch，训练模型区分原样本与合成 irregularity。其方向是生成伪异常，且是图像级；不能覆盖本候选的 negative-bag certificate，但占据 CutPaste 基础来源。
2. [VideoMix: Rethinking Data Augmentation for Video Classification](https://arxiv.org/abs/2012.03457)：跨视频替换时空块；Temporal VideoMix 已用于 THUMOS14 的弱监督时序定位。它占据 temporal transplant 算子与 weak localization 应用。
3. [Unsupervised Pre-Training for Temporal Action Localization Tasks](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Unsupervised_Pre-Training_for_Temporal_Action_Localization_Tasks_CVPR_2022_paper.pdf)：把连续 pseudo-action regions 粘贴到 background videos，并利用已知时序变换做定位表征预训练。它占据“连续区域移植到背景上下文、利用合成位置学习 localization”的相邻范式。
4. [A Multi-Head Approach With Shuffled Segments for Weakly-Supervised Video Anomaly Detection](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/AlMarri_A_Multi-Head_Approach_With_Shuffled_Segments_for_Weakly-Supervised_Video_Anomaly_WACVW_2024_paper.html)：在 normal/anomaly 视频间随机重组连续 segments，借已知 virtual-event 结构训练 boundary/center heads。它没有直接做本候选的 transplant-only benign BCE，但已经覆盖 WS-VAD 中 normal/anomaly segment mixing 与合成局部结构监督。
5. [Completeness Modeling and Context Separation for Weakly Supervised Temporal Action Localization](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Completeness_Modeling_and_Context_Separation_for_Weakly_Supervised_Temporal_Action_CVPR_2019_paper.html)：从训练视频挖掘 static clips，组成 background-class hard-negative pseudo videos。它占据利用可靠/启发式 background snippets 补局部负监督的邻近机制。

这些外部工作本身未必逐项否定“negative-bag certificate + positive context”这一窄 adaptation；真正致命的是这项窄 adaptation 已由本项目旧正式方法完整实现。

## 与其他已关闭链的关系

- 与 `counterfactual_positive_transplant_candidate` 不是同方向：该候选把 weak-selected positive window 移入 negative host，因 donor 非 certified positive 而停止。但它的旧查新已经明确指出本项目此前做过当前这个反方向 benign insertion。
- 与 `distributionally_stable_rationale` / `dual_certificate_benign_filled_rationale` 不严格同构：后两者学习 selector/mask，并用 keep/remove replacements 证明 frozen/joint classifier 的 sufficiency 与 necessity；当前方法使用随机已知 transplant mask，不学习 rationale selector。
- 因而本次 STOP 不应错误归因为 rationale collusion；最直接、充分的原因就是与 `powa_benign_insertion_pilot` 完全重复，并重复其已经观察到的 performance trade-off。

## Matched control 是否能证伪机制

当前 positive-donor matched control **能部分证伪机制语义，但不能修复 novelty**：

- 若 negative donor 明显优于相同预算的 positive donor，支持收益来自 negative-bag benign certificate，而不是任意 foreign-window suppression。
- 若两者相当，声称的 certificate 不是 load-bearing，机制失败。

但它不足以排除两个替代解释：普通 splice augmentation/边界 shortcut，以及只是在原 negative videos 上增加更多 dense-negative loss。更重要的是，旧 `powa_benign_insertion_pilot` 的 novelty plan 已经预注册过 positive-donor、splice-only、original-negative dense、boundary 等同类 controls；旧方法因正式双 test performance gate 失败而没有继续运行 controls。现在补做其中一个 control，不会把旧失败方法变成新 candidate，也不满足 Rule 18 所需的 post-test cross-corpus corrective evidence。

## 对 Rule 12 / 14 / 21 的最终解释

- **Rule 12：STOP。** Gate 3 明确失败；项目级 Gate 2 也失败。
- **Rule 14：不需要 premise。** 当前依赖的信息源并不新，且已有该精确 end-to-end 方法的 HMM/HCS test evidence；新 premise 只会重复旧证据。
- **Rule 21：STOP。** Brief 正确区分了 observed headroom 与 train-available negative-bag signal，也解释了三指标路径；但它没有提供区别于已关闭 benign-insertion 方法的新 correction mechanism。validation 扩大调参不能重开失败链。

**最终裁定：STOP。归档为与 `powa_benign_insertion_pilot` 严格同构的重复候选，不进入实现、technical review、validation search 或 test。**

