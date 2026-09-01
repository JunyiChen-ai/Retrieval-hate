# Independent Novelty Review — Witness-Preserving Temporal TokenFusion

**截至 2026-09-01。只审 novelty，不审代码或实验可行性，也不是 process review。**

## Verdict

**GO — 6.7/10。** Rule 12 三门均通过，但 novelty claim 必须严格限定为：把 TokenFusion 的 aligned token substitution 改造成弱监督 hateful temporal localization 中的逐秒 tri-modal substitution，并用 positive latent-witness retain 约束保证当前 MIL witness 至少保留一个未被替换的原始 carrier。不能声称 token substitution、retain gate 或 multimodal fusion 本身是新方法。

## 实际检索范围

核查了 TokenFusion 原论文/官方实现及 hateful-video 中最接近的 fusion 方法；检索词包括 `TokenFusion hateful video detection localization`、`Multimodal Token Fusion hate video`、`uninformative tokens hateful video`、`hateful video temporal multimodal token substitution`，并在下列目标任务论文/官方页面内查找 `TokenFusion`。截至本次检索，没有找到 TokenFusion 来源方法被用于 hateful video detection 或 localization 的记录。

- [TokenFusion, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Multimodal_Token_Fusion_for_Vision_Transformers_CVPR_2022_paper.html)：动态找出低重要度 token，并以对齐的其他模态投影替换；原论文任务是 image-to-image translation、RGB-depth segmentation 和 image-point-cloud 3D detection。
- [TokenFusion 官方实现](https://github.com/yikaiw/TokenFusion)：公开任务和数据仍是上述视觉生成、分割与 3D detection；未列 hateful-video 任务。
- [MultiHateLoc](https://arxiv.org/abs/2512.10408)：已有逐时刻 dynamic cross-modal fusion、contrastive alignment 与 weakly supervised MIL，但没有低重要度 token 的跨模态替换，也没有 latent-witness retain constraint。
- [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.817/)：同样以 feature dilution 为问题，但采用保留单模态 expert、global deliberation 和 instance-level decision arbitration；它是 video-level detection，不做逐秒 token substitution 或 weak-MIL witness preservation。
- [CMFusion](https://arxiv.org/abs/2505.12051)：采用 temporal cross-attention、channel-wise 与 modality-wise fusion，没有 recipient-token replacement 或 witness retain constraint。
- [MM-HSD 官方实现](https://github.com/idiap/mm-hsd)：使用 late fusion、cross-modal attention feature 或 early CMA，未发现 TokenFusion 式替换。
- [RAMF](https://arxiv.org/abs/2512.02743)：使用 local-global temporal encoding 与 semantic cross-attention，未发现 TokenFusion 式替换。

检索只能支持“未发现占用”，不能证明不存在未索引工作；但当前证据足以通过 Rule 12(b)。

## Rule 12 三门

### Gate 1 — 允许 adaptation：PASS

来源明确、可核查，且规则允许 adaptation 已有方法。候选没有把 TokenFusion 原始 token replacement 冒充从零提出。

### Gate 2 — 来源方法未被目标任务占用：PASS

TokenFusion 原论文及官方实现不包含 hateful video detection/localization。对 MultiHateLoc、SAGE、CMFusion、MM-HSD、RAMF 的核查也没有发现它们使用或引用 TokenFusion 作为 hateful-video fusion 机制。目标任务已有大量 gating、cross-attention、MoE 与 dynamic fusion，但“已有一般 fusion”不等于来源 TokenFusion 已被占用。

### Gate 3 — non-trivial task adaptation：PASS，且为有条件通过

单独把空间 token 换成时间 token、把 transformer block 换成 MLP 前的一次 substitution，会是 direct port，不足以通过。本候选超过 direct port 的部分是弱监督约束的改变：

1. 训练没有秒级标签，不能照搬来源方法以 task loss 判断各空间 token 是否可删；候选把当前 positive-bag fused top-K 作为 detached latent-witness support。
2. 在该 support 上强制 `sum_m retain_m >= 1`，明确禁止一个被 MIL 当作正 witness 的秒把三个原始 carrier 全部替换掉；support 外的 retain budget 则保证 substitution 不是名义模块。
3. 替换后的三个逐秒 embedding 直接进入原 fused MLP/head 和唯一 raw `score_fused`，因此约束改变的是最终定位排序的信息通路，不是 auxiliary loss、训练权重或 inference ensemble。
4. aligned-vs-shifted control 只破坏 donor 的时间对应，可直接证伪“正确时间对齐的 donor substitution 是增益来源”。

这形成了可讲且可否定的任务机制：异步 speech/text/visual carrier 下，低质量 recipient 可从同秒其他模态补充，但弱监督正 witness 必须保留至少一个 native carrier，避免 substitution 把当前局部证据全部覆盖。它不是完全新算法；可主张的新意只在这一 temporal weak-MIL adaptation。

## 与 failure ledger 的严格同构检查

- **DMS / capacity-forced ownership：不严格同构。** 旧方法选择或均衡哪个 branch 负责 final score；本候选改写每个 recipient embedding 的内容，再由原 fused head输出，不设 expert capacity 或均衡责任。
- **Self-coalition temporal credit：不严格同构。** 本候选不计算 coalition/Shapley credit，也不用 frozen fused scorer 的反事实分数决定模态权重。detached fused top-K 仍有 self-confirmation 风险，但这里只定义“哪些正 witness 不可全替换”，不产生模态 competence target。
- **Gradient-only balancing：不严格同构。** substitution 位于唯一 raw fused score 的前向路径，关闭后才退化 anchor，不只是调 branch gradient。
- **Temporal residual：不严格同构。** 没有 alternating optimization、bag-gradient residual 或对 raw temporal score 的加性修正。
- **Latent-witness failure debiasing：不严格同构。** 没有 shortcut expert、GCE 或 sample reweighting；top-K support 上施加的是表示通路可行性约束。

主要风险不是 ledger 的严格重复，而是 retain gate 可能学到任意稳定 routing、top-K support 可能自确认、或 substitution 只充当容量/正则化。按 Rule 12，这些应由正式双 test 和 matched shifted control 裁定，不构成 pre-run STOP。

## 不可变的 load-bearing 约束

以下任一项被删除或改写，当前 GO 不再适用，必须重新做 novelty review：

1. 三个 modality recipient 都必须在**每秒**执行 retain-or-substitute，且 substituted embeddings 必须直接进入原 fused MLP/head 的唯一 raw `score_fused`；不得改成 auxiliary head、branch-score ensemble、calibration 或后处理。
2. Positive support 只能由本语料 positive training bag 的当前 fused top-K 产生并 detach；不得使用 frame/span label、test label、其他主数据集训练样本或 teacher temporal label。
3. 在每个 positive latent-witness 秒必须执行 `sum_m retain_m >= 1`；support 外必须保留预注册的 retain-budget penalty，使模型不能通过“全部 retain、不发生 substitution”绕过机制。
4. `alpha_fusion=0` 必须不构造/调用 substitution 参数，并精确退化为同 harness MultiHateLoc 的 forward、raw score、基础 loss 与 schedule。
5. 只允许本 brief 的一次 fusion-input 机制变化；不得同时加入新 expert routing、branch reweighting、residual score、teacher supervision或额外 temporal regularizer。

## Aligned-vs-shifted matched control

- Aligned arm 的 donor content 只能来自同一有效秒。
- Shifted arm 对每个视频的所有 donor content 作半个有效长度的 circular shift；recipient content、retain gates、donor-mixture logits、projection、参数量、loss 权重、初始化规则、epoch、learning rate、checkpoint selection 和评测流程必须一致。
- Padding 不得参与 shift；每个视频仅在其有效长度内部循环移动。
- 必须同时报告 aligned、其 same-hyperparameter shifted control 和 same-learning-rate `alpha_fusion=0` anchor。
- 机制预期保持 brief 原样：aligned 的 test within-video ROC 必须在 HMM、HCS 都高于 shifted，且至少一个语料差值 `>= .010`。若不满足，应判定正确 donor timing 非 load-bearing，并关闭 gate/projection/budget/shift/fusion-strength family，不以改 gate、换 projection 或调 budget 续命。

## 最终裁定

**GO 6.7/10。** 来源未见目标任务占用；task adaptation 不是靠改名成立，而是靠“逐秒 aligned substitution + positive latent-witness 至少保留一个 native carrier + raw fused score 直达 + matched temporal shift”这一不可拆组合成立。它与已关闭的 DMS、ownership、self-coalition、gradient balancing、temporal residual 和 witness debiasing 均非严格同构。允许进入实现；本审查不对性能或代码正确性作判断。
