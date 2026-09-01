# Independent Novelty Review — Lexically Anchored DCC

**截至 2026-09-01。只审 novelty；不审代码、训练可行性或实验流程，也不是 process review。**

## Verdict

**GO — 6.4/10。** Rule 12 三门均通过，但只允许一个很窄的 adaptation claim：

> 将 DCC 的模型自生成 foreground/background 区域改造成 **corpus-local、五折 OOF lexical temporal anchors 驱动的 one-sided region admission**：positive bag 只有 lexically supported 连续区域可进入 hateful memory，negative bag 的有效区域可进入 benign memory，而 positive bag 的其余区域保持 unlabeled；该非对称跨视频 region-memory contrast 只在训练期更新同一 POWA shared representation，推理仍输出单一 POWA raw frame score。

不能 claim 首次在 hateful video 使用 contrastive learning、segment-level contrast、memory、pseudo-label、lexical cue、MIL 或 cross-video learning。OOF 本身是防止 in-sample producer 泄漏/自拟合的训练构造，也不能单独作为 novelty。

## 实际检索范围

实际检索了来源论文及目标任务近邻，查询包括 `Denoised Cross-Video Contrast hateful video`、`region-level memory bank hateful video`、`pseudo-label contrastive hateful video temporal`、`lexical region-level contrastive hateful video`、`out-of-fold lexical hateful video`。截至本次检索，未找到 DCC 的“denoised pseudo-region + dataset-wide region memory + action/background region contrast”被用于 hateful video detection 或 localization 的记录。这个结论是“当前检索未发现占用”，不是对所有未索引文献的不存在证明。

- [DCC, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Exploring_Denoised_Cross-Video_Contrast_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2022_paper.html)：来源任务是 weakly supervised temporal action localization。它从模型 activation 产生 snippet pseudo-label，以 clustering confidence 降噪，把 action/background region features 写入全数据集 memory，再做跨视频 region contrast。
- [ImpliHateVid, ACL 2025](https://aclanthology.org/2025.acl-long.842/)：已经在 hateful-video detection 使用两阶段 supervised contrastive learning，先训练 modality encoders，再训练 cross-modal encoders；因此 broad supervised/multimodal contrastive claim 已被占用。它不做 temporal lexical anchors、one-sided region admission 或 cross-video hateful/benign region memory。
- [CLARA, ACM MM 2026](https://arxiv.org/html/2608.15905)：已经在 hateful-video detection 使用 clip-level local-global segment contrast。其正对是同一视频随机 local/global segments，其他视频 segments 是负对，目的是 local-global temporal consistency；它不使用 bag-label-certified benign regions、OOF lexical temporal support 或 class-conditional region memory。因此 segment contrast 的上位 claim 也已被占用，但精确核心不同。
- [MultiHateGNN, BMVC 2025](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_408/paper.pdf)：已明确利用“少量 hateful content 即决定 positive video”的非对称性，以 learned weight graph 强调重要 instances。它占用了“突出稀疏 hateful instance”这一问题陈述，但没有 lexical OOF producer、contrastive region memory 或 positive-unselected-is-unlabeled 的 admission rule。
- [Revealing Temporal Label Noise in Multimodal Hateful Video Classification, MUWS 2025](https://arxiv.org/abs/2508.04900)：已分析 hateful videos 内的 temporal label noise、词汇差异及使用人工 timestamp trimming 的 clean-segment training。它支持任务困难已经被认识，但没有从 video labels cross-fit 出 local anchors，也没有本候选的弱监督 memory mechanism。
- [HateClipSeg](https://arxiv.org/abs/2508.01712)：建立了 segment-level annotation 与 temporal hateful-video localization benchmark；未在其公开方法描述中发现 DCC 或本候选的训练机制。

## Rule 12 三门

### Gate 1 — 跨任务来源可 adaptation：PASS

DCC 是明确、可核查的 WTAL 来源方法，项目规则允许 adaptation 已有方法。候选也没有把 region-memory contrast 或 pseudo-label denoising冒充为从零提出。

### Gate 2 — 精确来源方法/核心机制未见目标任务占用：PASS（窄口径）

目标任务中的 contrastive learning 已明确被 ImpliHateVid 和 CLARA 占用；clip/instance modeling 也已被 CLARA 与 MultiHateGNN 占用。因此若 claim 写成“把 contrastive learning 用于 hateful videos”或“首次做 segment contrast”，本门应当 STOP。

本次通过的对象更窄：检索到的目标任务论文没有把 DCC 的 dataset-wide class-region memory 用于 hateful video，更没有用同语料 train video labels cross-fit 得到的 lexical temporal support，执行 `positive supported -> hateful memory / negative valid -> benign memory / positive unsupported -> unlabeled` 的 one-sided admission。DCC 的精确来源核心尚未见在目标任务被占用，故 Rule 12(b) 通过。

### Gate 3 — non-trivial hateful-localization adaptation：PASS

若只是把 DCC 的 action 名称换成 hate、或把 DCC self score 换成另一个 pseudo-label score，会是 direct port/producer swap，不能通过。本候选可通过的 load-bearing delta 是 **memory supervision topology 的改变**：

1. DCC 用同一 localization model 的 activation 产生 action/background pseudo-regions；本候选用每个语料独立、五折 held-out train videos 产生的 lexical locality，避免用当前 POWA top-K 或 branch confidence重新确认自身局部排序。
2. Weak hateful positive bag 只保证至少一个 hateful second，不能把未被 lexical anchor支持的 positive-video seconds 当可靠 benign。候选因此采用 one-sided admission：只收高支持 positive regions 与 certified negative-bag regions，显式 abstain 于 positive bag 其余时间。这个约束不是 DCC 的对称 action/background pseudo-labeling。
3. Lexical posterior 不直接加到 test score，也不要求 student 拟合其数值或完整排序。它只决定哪些 train regions有资格参与跨视频 class-region contrast；memory中的 target relation还由 bag label、跨视频 region identity和当前 shared embedding共同定义。
4. Contrastive gradient必须进入产生最终 `frame_prob` 的 POWA shared representation；推理移除 lexical producer、memory和projection readout，仍使用同一个 raw policy head。因此其机制目标是改变单学生的 pooled separation 与 positive-video内部排序，不是 inference blend或后处理。
5. Aligned-vs-circular-shifted control保持 anchor数量、连续段结构、模型容量、训练量与 loss不变，只破坏 lexical support的时间对应，能够直接否定“正确 lexical timing通过非对称 region memory改善定位”的故事。

任务机制是明确且可证伪的：hateful videos 中 positive bag 的背景不可信，而 negative bag 的所有有效区域具有可用的 benign certificate；OOF lexical locality只为 positive bag提供稀疏、独立的候选 positive support。通过非对称 admission 做跨视频 region contrast，有机会同时提高跨视频 hate/benign separation（pooled AP/ROC）和同一 positive video 内的局部排序（within ROC）。这不是完全新算法，但超过了只换数据集或直接套 DCC。

## 与 failure-equivalence ledger 的严格同构检查

- **Teacher-order KD：不严格同构，但边界很近。** Lexical posterior没有作为 soft target、margin或全序被 student拟合；它只作 one-sided region admission，positive-unselected regions保持 unlabeled，跨视频对比关系还依赖 certified negative bags 与 region memory。故不是“teacher给 pair order，换一种 distillation loss”。如果实现改为拟合 lexical score、lexical pair order、lexical hard frame labels，或让高低 lexical seconds直接形成完整正负排序，本 GO 立即失效并落回 ordinary teacher/pseudo-label distillation。
- **Self-confirming top-K / branch-confidence chain：不严格同构。** Memory membership来自 cross-fitted lexical model，不读取 POWA fused top-K、DMS、masked self-credit或 branch confidence。当前 scorer只能更新 embedding，不能选择哪些 aligned seconds被宣布为 lexical support。
- **Ensemble / calibration / scale mapping：不严格同构。** Test inference不得读取 lexical posterior或memory score，不做 score blend、CDF、routing或校准；唯一输出是训练后单一 POWA raw `frame_prob`。
- **Direct-head replacement：不严格同构。** 原 POWA policy head保留；region projection只承载训练期 contrastive objective，不能替换 final head或直接成为 test scorer。
- **Auxiliary-head bypass：是风险，不是解析同构。** Projection head可能吸收 contrastive loss而 shared/final ranking几乎不变；这由正式 test 的 aligned-vs-shifted ordering和表示梯度路径检查裁定。只要 loss确实反传到 final scorer共享表示，Rule 12不要求在实现前证明一定 load-bearing。
- **先前 lexical posterior regularization：不是同一约束的重命名。** 旧方法把 OOF lexical evidence作为 aggregate posterior KL/I-projection约束；本候选不约束 score posterior，而改变哪些跨视频 regions可组成 hateful/benign representation pairs。它仍复用同一信息源，所以不能把“换成 contrastive loss”本身 claim 为 novelty；本次唯一可主张部分是 one-sided admission 与 dataset-wide class-region memory的不可拆组合。

## 必须保留的 load-bearing 边界

以下任一项删除或改写，当前 GO 不再覆盖该实现：

1. OOF lexical producer必须按语料独立，仅用该语料 train video labels、train ASR文本与时间戳；每个 train video 的 posterior必须由未见该视频的 fold model产生。
2. Positive bag只有预注册高置信、时间连续的 lexical-supported regions可进入 hateful memory；negative bag的有效 regions可进入 benign memory；positive bag其余 regions不得被写成 benign或形成反向 pseudo-order。
3. Region memory必须跨 training videos形成 class-region contrast；contrastive gradient必须进入与最终 POWA policy head共享的 temporal representation，而不能只训练一个与 final score解耦的 auxiliary embedding。
4. Test inference只能输出同一个 POWA raw `frame_prob`；不得读取 lexical posterior、memory neighbor score、teacher score、校准映射或其他模型分数。
5. 不得额外加入 lexical score BCE/KL、rank distillation、inference blend、teacher routing或 direct lexical head；否则有效部分改变，必须重新审 novelty。

## 唯一必须保留的 matched control

**Matched deterministic half-video circular shift of the OOF lexical anchors。** 对每个 train video，只在其有效时间范围内把完整 OOF lexical posterior循环平移半个视频长度，再用与 core完全相同的 threshold、连续区域构造和 memory admission；padding不得参与 shift。

Core 与 shifted arm 的 POWA初始化、projection、memory capacity与更新、batch sampler、learning rate、contrastive weight、support quantile、epoch、checkpoint selection、随机种子和 evaluator必须一致。该 control保持 lexical score分布、每视频 support质量与区域数量近似不变，只破坏 lexical timing。必须同时报告相同配置的 POWA/no-contrast control，以区分“aligned优于shifted”和“任意额外 contrastive regularization优于anchor”。

预注册机制预期维持 README：core 在 HMM/HCS test 的 within-video ROC 都高于 shifted，并且两个语料各至少一个 pooled指标同向提高；最终晋级仍要求六项全部超过固定 SOTA threshold。若 aligned不胜 shifted，不能把一般 regularization/capacity收益归因于 OOF lexical temporal anchoring。

## 最终裁定

**GO 6.4/10。** DCC 可作为跨任务来源；精确 DCC region-memory核心未见被 hateful-video detection/localization占用，尽管 broad contrastive和segment contrast已经明确被占用。候选的合格 novelty 不是 pseudo-label或contrastive本身，而是针对 weak hateful bags 的 **OOF lexical positive support + certified-negative benign support + positive-unselected abstention + training-only cross-video class-region memory**。它与 teacher-order KD、自确认 top-K、ensemble/calibration和direct-head replacement均不严格同构，但只有在上述边界与 matched circular-shift control完整保留时才允许进入实现。
