# Policy-patch excess: narrow occupation check

截至 2026-08-31。范围严格限定为 **hateful video detection/localization**；静态 hateful-meme 工作只算相邻领域，不用于判定来源已进入目标任务。本检查不涉及实现或实验。

## 裁定

- **窄来源占用门：PASS（必须窄写）**。在本次检索覆盖的目标任务论文中，没有发现把 DenseCLIP/RegionCLIP 式的 dense/region text-aligned evidence，或 CLIP patch-token–policy evidence，直接用于 hateful-video 的帧级定位。该结论是截至当前检索范围的“未发现”，不是穷尽性不存在证明。
- **宽来源占用门：FAIL**。不能 claim “首次把空间/物体级证据用于 hateful video detection”：Yadav 与 Singh（WWW Companion 2026）已经从视频帧抽取 object detections 并作显式 evidence attribution；MATCH（TCSVT 2026）也已把方法表述为 spatiotemporal evidence-grounded verification。
- **第三门（non-trivial task adaptation）：FAIL**。当前正式边界只是把一个冻结、手工构造的 patch-score scalar `u_t` 拼入普通 temporal MIL。dense/open-vocabulary spatial scoring、prompt scoring、空间对比和 MIL 均是已有组件；尚无新的可学习约束把“局部 carrier 而非 scene/topic”变成可识别的 hateful-video 机制。
- **总裁定：STOP，4.9/10**。可以把固定 premise 当作 representation feasibility diagnostic，但 premise 即使通过，也不足以使 README 当前描述的正式方法通过 novelty 第三门。

## 目标任务占用

| 工作 | 在 hateful video 中实际使用的证据 | 对本候选的含义 |
|---|---|---|
| [Yadav & Singh, WWW Companion 2026](https://doi.org/10.1145/3774905.3796488) | 从视频帧抽取 object detections，并与 OCR、speech 等共同作显式 evidence attribution | 已占用宽泛的“object/spatial evidence for hateful-video detection”；但不是 dense CLIP patch-text map，也不做弱监督帧定位 |
| [MATCH, TCSVT 2026](https://jianlang.org/papers/MATCH.html) ([manuscript](https://jianlang.org/papers/MATCH.pdf)) | 将采样的完整帧连同邻近 OCR/transcript 组成 spatiotemporal units，再检索线索并由多模态模型验证 | 已占用宽泛的 evidence-grounded hateful-video verification；其视觉单位仍是完整帧，不是 patch/box spatial MIL |
| [MM-HSD](https://github.com/idiap/mm-hsd) | ViT frame embeddings 与其他模态融合 | 仍是整帧表示，不占用窄来源 |
| [RAMF](https://github.com/Multimodal-Intelligence-Lab-MIL/RAMF) | CLIP/ViT frame features 与多模态融合 | 仍是整帧表示，不占用窄来源 |

因此，允许的 source statement 只能是：**“我们未发现 dense/region CLIP policy evidence 被用于 hateful-video temporal localization。”** 不允许扩大为首次使用 region、object、grounded evidence 或 spatial evidence。

## Source fidelity blocker

README 的“冻结 CLIP ViT-B/16，读取 projected patch tokens 后直接与 text embedding 算 cosine”并不等价于所列来源核心：

- [DenseCLIP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Rao_DenseCLIP_Language-Guided_Dense_Prediction_With_Context-Aware_Prompting_CVPR_2022_paper.pdf)把 image-text matching 转为 pixel-text matching，并为 dense prediction 训练/微调上下文提示与网络。
- [RegionCLIP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhong_RegionCLIP_Region-Based_Language-Image_Pretraining_CVPR_2022_paper.html)明确把 vanilla CLIP 的 image-to-region domain shift 作为问题，并进行 region-text pretraining。
- [MaskCLIP](https://mmlab-ntu.github.io/project/maskclip/index.html)使用特定的 dense-label extraction 路径；[Patch Aligned Contrastive Learning](https://ai.meta.com/research/publications/open-vocabulary-semantic-segmentation-with-patch-aligned-contrastive-learning/)则专门训练 patch-text alignment。

这些先例共同表明：普通 CLIP 的内部 patch token 不能未经资格验证就被当作可靠的 text-aligned region embedding。当前候选至多是一个 **naive frozen-patch probe**，不能称为 DenseCLIP/RegionCLIP 的 source-faithful adaptation。若 raw patch 不具备空间文本对齐，后续所有 `u_t` 机制解释均失去前提。

## 数学与 control 问题

1. **现有 permutation control 在公式下不成立。** `u_t` 只依赖 patch/clause score 的 top multiset，以及 `max_c` 后 patch 值的 trimmed mean。若“spatially permuted”只是打乱 patch 位置，两个聚合量严格不变；若只置换 clause 编号，`max_c` 和跨 `(p,c)` top aggregation 也严格不变。该 arm 会与 core 相同，无法检验 policy correspondence。
2. **`u_t` 是局部显著性差值，不是 hate-specific identification。** 小字幕、logo、脸、商品或任意高对比小物体均可产生 top-tail excess。generic prompts 是必要 control，但不足以排除 prompt norm、词频和视觉显著性差异。
3. **benign prototype 与 core 公式不一致。** README 声明 policy clauses 加 benign prototype，`global_policy` 也定义为 policy-vs-benign；但 core `a_{t,p,c}` 与 `u_t` 公式没有 benign log-odds 项。于是 global/core 的方向性定义不匹配。
4. **正式方法仍是 feature concatenation。** 固定 `u_t` 加入既有 localizer 并不会自动证明 temporal ownership 改善来自空间 carrier；capacity-matched scalar control只能排除额外维度，不能建立该机制的可识别性。

## 若只运行 premise diagnostic，必须先修的边界

- 明确 patch tensor 的层、projection、归一化和空间注册，并加入 source-faithful dense extractor 与 naive raw-patch 的对照；两者都失败即停止。
- 将 core 明确定义成 patch-level **policy-vs-benign** log-odds，再作 spatial excess，确保与 global arm 同一方向。
- 删除“shuffle score indices”的伪 control；改为相同数量、长度与 embedding norm 分布的 matched non-policy prompts，或跨帧错配 text embedding。control 必须实际改变 patch-text correspondence。
- 报告 raw spatial saliency/objectness control，确认增益不是任何局部高响应都能获得。
- premise 的 test 分析只能 inform 后续 development；不得表述为未揭盲 confirmatory evidence，也不得把 diagnostic arm 当正式方法或 SOTA arm。

即便完成以上修正，结论仍只是“冻结的局部 policy-aligned observation 是否有增量信息”。要重新通过第三门，需另行提出一个使 spatial carrier/background distinction 对 hateful-video temporal learning **不可替代且可证伪**的训练机制，而不是继续把该 scalar 接入普通 MIL。
