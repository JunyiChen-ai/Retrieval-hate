# Cross-modal innovation MIL：独立 novelty / mechanism review

截至 2026-08-31。审查对象是本目录 `README.md` 中的候选定义；没有审查实现，因为候选尚未实现。本轮只做文献与机制审查，没有启动训练或推理。

## 结论

**Verdict：GO，但只批准下文“最小批准边界”中的修订版本。Novelty：5.8/10。**

三项硬门结论如下：

| 硬门 | 结论 | 理由 |
|---|---|---|
| 允许跨任务 adaptation | PASS | 来源是跨模态检索/预训练、multimodal sentiment、missing-modality learning 和 AVVP，不要求从零发明 reconstruction 或 shared/private。 |
| 来源核心没有已经用于 hateful video detection/localization | PASS，检索范围内未发现直接先例 | 截至检索日，hateful-video 原始论文使用 cross-modal attention、MoE、contrastive alignment、VLM rationale、独立模态 expert 或 MIL；未发现“用其余模态逐时预测目标模态，冻结 predictor，把预测残差作为显式通道送入 video-label-only temporal localizer”的方法。 |
| adaptation 必须 non-trivial、任务特定且可证伪 | PASS，但当前公式必须先修 | 将 prediction residual 保留下来、让弱标签只学习其证据方向，并显式区分 missing target 与 observed-but-unpredicted evidence，确实不同于用 reconstruction 改善共同表示或补全缺失模态。但原稿中的可训练目标投影、逐帧 LayerNorm、未说明归一化的 log-sum-exp，以及可忽略 private 的 MIL 会让该机制塌缩或无法归因。只有修正并通过干预 control 后，才不是 Gabeur/MISA 加普通 MIL 的组件串接。 |

这不是对“shared/private disentanglement”“masked modality modeling”“preserving modality-specific hate evidence”的新颖性批准。尤其 [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.817/) 已经在 hateful video detection 中明确以防止 dominant benign modalities 稀释 sparse modality-specific hateful cues 为动机，并采用独立模态 experts 与 arbitration。可保留的窄 claim 只能是：

> 将仅由目标语料 train videos 学到并冻结的 cross-modal conditional-prediction residual，作为一个与 missingness 分离、与 predicted component 同时保留的候选证据通道，适配到仅有 video labels 的 hateful temporal localization；其作用必须由同一 checkpoint 的 residual intervention 证明。

若不能证明 residual 通道对最终逐帧排序是 load-bearing，结论自动转为 **STOP：普通 masked-modality representation 加 MIL**。

## 检索范围与直接占位结论

检索使用的代表性组合包括：

- `hateful video detection/localization + masked modality / modality prediction / reconstruction`；
- `predict one modality from other modalities + residual / modality-specific representation`；
- `weakly supervised audio-visual video parsing + modality-specific / cross-modal fusion`；
- 目标领域方法名 `CLARA / MultiHateLoc / MM-HSD / SAGE` 与 reconstruction、mask、missing modality 的组合。

只把论文、会议开放论文页、ACL Anthology、作者/机构论文页和官方代码作为裁定证据。搜索没有证明不存在任何未索引工作；它支持的是“在截至日期可检索的目标任务 primary literature 中未发现直接先例”。

### 最接近的 representation / missing-modality 工作

1. [Gabeur et al., *Masking Modalities for Cross-Modal Video Retrieval*, WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Gabeur_Masking_Modalities_for_Cross-Modal_Video_Retrieval_WACV_2022_paper.html) 遮掉 appearance、sound、transcribed speech 中的整个 modality，并由另两种模态预测它，用于 How2R、YouCook2、Condensed Movies 的 video retrieval。它占用 whole-modality prediction 与三模态协作预训练；没有保留 prediction error 作为下游证据，也没有 hateful task、temporal hate score 或 weak MIL。

2. [CrossMAE, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_CrossMAE_Cross-Modality_Masked_Autoencoders_for_Region-Aware_Audio-Visual_Pre-Training_CVPR_2024_paper.html) 用 cross-conditioned pixel reconstruction 和 cross-embedding reconstruction 学细粒度 audio-visual 对齐，并评测分类、检索和 audio-visual source localization。它进一步压缩“细粒度 cross-modal reconstruction”这一宽 claim，但 reconstruction residual 不是被保留的下游 MIL evidence channel。

3. [CAV-MAE Sync, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Araujo_CAV-MAE_Sync_Improving_Contrastive_Audio-Visual_Mask_Autoencoders_via_Fine-Grained_Alignment_CVPR_2025_paper.html) 已将 audio-visual masked autoencoding 推到细粒度同步与对齐。因此不能把逐时 reconstruction、fine-grained alignment 或 multimodal masked pretraining 本身写成贡献。

4. [MISA, ACM MM 2020](https://arxiv.org/abs/2005.03545) 显式学习 modality-invariant 与 modality-specific subspaces，并共同融合做 sentiment/humor prediction。[MFM, ICLR 2019](https://iclr.cc/virtual/2019/poster/925) 将表示分成 multimodal discriminative factor 与 modality-specific generative factors，并支持 missing-modality reconstruction。两者已经占用 learned shared/private factorization。候选的差异只在：其 `private` 不是自由 latent subspace，而是相对于一个冻结 conditional predictor 的外显残差；而且下游是 video-label-only temporal MIL。

5. [MissModal, TACL 2023](https://aclanthology.org/2023.tacl-1.94/) 等 missing-modality 工作主要让 incomplete input 接近 complete representation；[Missing Modality Prediction for Unpaired Multimodal Learning, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11614.pdf) 还在 Hateful Memes 上评测了 missing text/image prediction。后者说明 broad harmful-content 邻域已有 modality prediction，但它是静态 meme、缺失模态学习，不是 hateful video detection/localization，也不把 observed target 的 prediction residual 当 temporal witness。

这些工作共同决定：`shared/private`、masked prediction、reconstruction、missing-modality handling 和 residual feature 都不能单独 claim novelty。

### 最接近的 weak audio-visual parsing 工作

1. [JoMoLD, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940424.pdf) 直接处理 video labels 对不同 modalities 造成的 modality-specific label noise，通过跨模态 loss inconsistency 删除噪声 modality labels。

2. [CoLeaF, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01653.pdf) 明确区分 audible-only、visible-only 与 audible-visible events，并指出 cross-modal context 会伤害 unaligned unimodal events；其方法在 embedding space 中只对 aligned events 利用协作信息。

3. [Rethink Cross-Modal Fusion in Weakly-Supervised AVVP, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Xu_Rethink_Cross-Modal_Fusion_in_Weakly-Supervised_Audio-Visual_Video_Parsing_WACV_2024_paper.pdf) 也已指出不完全相关 modalities 的过早融合会伤害单模态事件，并用 cross-audio prediction consistency 抑制不合适的跨模态预测。

因此“弱标签下跨模态共识可能抹掉单模态事件”不是新问题。候选没有被完全占位，是因为它不从 weak labels 估 modality owner，也不删除不一致证据；它先用 label-free conditional prediction 定义一个固定 residual，再让单一 temporal MIL 判断其方向。

### hateful video 领域

- [MM-HSD, ACM MM 2025](https://publications.idiap.ch/publications/show/5688) 把 frames、audio、transcript、OCR 与 cross-modal attention features 做早/晚融合；没有 modality reconstruction residual 或 weak temporal MIL。
- [MultiHateLoc](https://arxiv.org/abs/2512.10408) 使用 modality-aware temporal encoders、dynamic cross-modal fusion、cross-modal contrastive alignment 和 modality-aware MIL；没有冻结 conditional predictor 或 retained residual。它是直接任务 baseline，也是本候选试图修复的 fusion failure 来源。
- [CLARA](https://arxiv.org/abs/2608.15905) 使用 clip-level MoE、local-global contrastive learning 和 VLM-derived rationales；没有 masked-modality reconstruction residual。
- [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.817/) 已在 HateMM/MultiHateClip 上以 disentangle-then-arbitrate 保留 modality-specific semantics，直接占用“融合稀释模态特有 hate cues”的任务故事。它仍是 video-level detection，独立 experts 也不是由其他模态条件预测得到的 shared/residual decomposition。

结论是：**exact operator 尚未在目标任务中找到，但问题陈述和多数部件高度拥挤。** 新意来自窄 adaptation 和严谨归因，不来自部件清单。

## 可识别性与数学 blocker

### 1. 冻结 predictor 只给 operational definition，不给 semantic identification

固定 `F_m` 后，`r_m(t)=z_m(t)-F_m(x_{-m})` 是可重复计算的，所以“相对于这个 predictor 的 innovation”在操作层面成立。但它不识别真实的 modality-private semantic factor：

- `r_m` 同时包含目标模态独有信息、predictor approximation error、训练分布覆盖不足、时间错位、编码器噪声和不可约条件方差；
- 不同容量、优化状态或 context radius 的 predictor 会定义不同 residual；
- 一个强 predictor 可能复制 topic/identity shortcut，一个弱 predictor 则把几乎全部 observed feature 留在 residual。

因此文中只能称 `conditional-prediction residual/innovation`，不能声称识别了 true shared/private semantics、carrier ownership 或 causal evidence。冻结只阻断 label-driven decomposition 漂移，不解决统计不可识别性。

### 2. 当前 `P_m` 存在严格的全零塌缩

原稿写 `z_m=P_m x_m`，再最小化 `F_m(x_{-m})` 对 `z_m` 的 Huber reconstruction。如果 `P_m` 与 predictor 一起自由训练而没有固定目标、方差约束或独立 reconstruction，

`P_m=0, F_m=0`

使每个训练样本的 reconstruction loss 都为零，是全局最优。这不是调参风险，而是定义中的数学退化。

启动实现前必须改为以下一种固定方案，首选第一种：

1. predictor 直接预测 train-split 固定标准化后的 frozen source feature `x_m`，残差也在这个固定 target space 中计算；进入 localizer 后再投影到共同宽度；
2. 或将 `P_m` 预先固定为不塌缩映射，并提供预先规定的方差保持条件。

不能一边学习 target projection 一边只用 predictability objective 约束它。

### 3. 逐帧 LayerNorm 破坏“innovation amount”并可能放大误差

`LayerNorm(z-shared)` 对正尺度基本不敏感。两个 residual 即使一个很小、一个很大，只要方向相近，就会得到近似相同的向量；接近零的 residual 还可能由数值噪声或 predictor 小误差决定方向。于是模型看到的不是“有多少不可预测信息”，而主要是标准化后的 residual direction。

这与“不直接把 residual norm 当 hate score”不是同一件事。可以让 MIL 学 residual 幅度的任务方向，而无需把 norm 硬编码成异常分数。批准版本应使用仅由 train split 固定的逐维标准化/whitening，并保留 residual 的相对幅度；不要对每个 frame 单独 LayerNorm residual。若坚持 LayerNorm，claim 必须改成 `normalized residual direction`，并把无 LayerNorm 的固定方案列为 attribution control，而不能称完整 innovation。

### 4. observed、predicted、residual 没有增加观测信息

在 LayerNorm 前，`residual=observed-predicted`。三者拼接是确定性冗余表示，并没有增加超出原始 modalities 的信息。一个容量足够的 raw model 可以近似学习相同变换。候选提供的是 inductive bias 和两阶段训练，不是新监督或新信息。

因此 `raw_capacity_matched` 是必要 control，但仅匹配参数数目仍不够；还应匹配 depth、temporal receptive field 和 predictor 产生的额外计算。否则提升可能只是额外网络容量或不同优化路径。

### 5. 未归一化 log-sum-exp 是 availability shortcut

若有 `N` 个有效 logits 且它们都等于常数 `c`，普通 log-sum-exp 输出 `c+log N`。目标模态缺失时 mask 会改变 `N`，于是即使证据完全相同，ASR 覆盖或可用 modality 数本身也会系统性改变 frame score。这正好能利用数据中的 transcript coverage shortcut。

融合必须是 masked **log-mean-exp**，即对有效项减去 `log N_valid`，或使用满足“所有有效 logits 相同则输出与可用项数量无关”的等价归一化。实现前应有固定常数输入测试。还必须有 availability-only baseline 和按 ASR coverage 分层的 test diagnostic。

### 6. predictor underfit、时间自相关和 availability 输入可制造伪 residual

局部窗口 `t-k:t+k` 让 predictor 可以主要依赖缓慢变化的场景、重复 transcript 或相邻秒，而不是对应时刻的 cross-modal semantics。availability flags 又可能成为 genre、时长、speech density 的代理。`private_without_crossmodal` 能排除一部分普通时间重建收益，但不能区分：

- 真正 matched cross-modal condition；
- 只用邻近时间的同视频统计；
- availability pattern；
- predictor capacity/underfit。

当前 `shuffled_condition` 也不是充分证据：打乱条件会同时造成 distribution shift 和 reconstruction quality 大幅变化，性能下降不必来自语义 correspondence。它可保留为 diagnostic，但不能单独承担归因。

### 7. MIL 可以完全忽略 residual

共享 evidence head 与 log-sum-exp 不强制 residual logit 对输出有贡献。训练可以把 residual 权重压到零，或让 observed/shared logits始终更高。这样模型即使优于 baseline，也不能证明 carrier innovation 机制；收益可能完全来自 masked prediction pretraining、额外容量或 normalized fusion。

不能通过新增一个不可识别的 latent owner 或强迫固定 residual 权重解决。最小可证伪办法是同一训练完成的 checkpoint 做输入干预：

- 正常 residual；
- residual 置零；
- 在保持 modality、availability 与边际分布的条件下打乱 residual 的 time/video correspondence；
- 用按模态匹配均值与方差的噪声替代 residual。

只有正常 residual 的逐帧排序增益显著高于这些干预，并集中在预注册的 baseline-loses-to-unimodal cases，才说明该通道 load-bearing。查看 test prediction 与 GT 做此 error analysis 符合项目规则，但证据必须标成 iterative/developmental test evidence。

## 是否只是组件串接

**原稿如果直接实现，是高风险组件串接；按最小边界修正并通过干预 gate 后，是可辩护的 non-trivial adaptation。**

统一机制链条应当只有一条：

1. label-free conditional prediction 估计“其他已观测 modalities 在该时刻能解释的 target component”；
2. 固定 residual 保留其他 modalities 不能解释、但 target 确实存在的内容；
3. missing target 不产生 residual，避免把无 ASR 当作 private evidence；
4. video-label MIL 只学习 residual 的 hate direction，而不把 prediction error norm 直接当异常；
5. 单一 frame score 同时允许 shared cross-modal hate 和 carrier-only hate。

如果 predictor 与 MIL 联训、如果 missing 被 impute 后也产生 residual、如果 residual 没有同-checkpoint intervention、或者只是把 MISA-style private branch 接到现有 MIL，那么上述链条断裂，应该判 STOP。

## 与项目既有失败的边界

- flat joint-witness ownership 被淘汰，是因为 time×modality attention/MIL 和 modality-specific noisy-label correction 已有直接先例，而且 attention 不识别 ownership。本候选不把权重命名为 owner，也不从 binary bag label 估 owner；它定义的是 label-free prediction residual。这是实质差异。
- coalition witness 在 HMM/HCS matched controls 下失败，说明“显式结构名称”本身不会带来稳定定位收益。本候选不能引用 coalition 的正确重构来支持新机制，只能靠新的 test pilot。
- deletion-carrier-abstaining ItS2CLR 虽通过 novelty review，但 test 中 core 与 broadcast 的 frame ranking 几乎相同，auxiliary carrier loss没有实质进入最终 localizer。本候选最相似的失败方式就是 MIL 忽略 residual，所以同-checkpoint residual intervention 必须成为硬 gate，而不是补充可视化。
- MultiHateLoc 的 DMS mismatch 和 best-branch oracle gap只证明当前 fusion 有改善空间，不证明 conditional residual 是正确解。oracle 不能用于训练、路由或选择模态。

## 最小批准实现边界

GO 只批准以下一次固定 pilot；任何扩展都要等结果：

1. 每个主语料独立训练；predictor 只读该语料 train split，不读 video label、video ID、validation/test 输入或 target modality。
2. predictor target 使用固定标准化的 frozen source feature；不得联合学习一个仅受 prediction loss 约束的 target projection。
3. predictor 训练后冻结。用固定 train statistics 标准化 residual，保留相对幅度；不做逐帧 residual LayerNorm。
4. target modality 不可用时，该 modality 的 observed、predicted 和 residual evidence 全部 mask；不得用 hallucinated target residual 替代 missing modality。
5. 所有 evidence 融合使用 masked log-mean-exp 或满足同一不变性的归一化形式。固定常数输入在不同 availability pattern 下必须给相同输出。
6. 一个共享容量的 temporal localizer 输出一个原始 frame score；无 branch selection、routing、ensemble、calibration 或 test-time smoothing。
7. 先只跑固定 HateMM/HateClipSeg、seed 234。Validation 仅在每个 arm 内选 checkpoint；选定后立即跑固定 test 三指标。不得先用 validation 比较 arms 或调整机制。

这一路径可称 `frozen cross-modal conditional-innovation MIL`。不批准 `identified shared/private ownership`、`causal carrier decomposition` 或 `first modality-specific hateful-video model` 等表述。

## 必须 controls 与晋级 gate

### 模型完整性 gate，训练正式 localizer 前完成

1. 固定 target 表示不存在全零塌缩路径。
2. 在 train-only held-out folds 上，cross-modal predictor 的 aggregate reconstruction 必须在 HMM/HCS 两边都优于逐模态 unconditional mean predictor；把其他视频、相同 availability pattern 的条件输入打乱后，误差必须回升。否则没有证据表明 residual 来自 matched cross-modal predictability，而不是“原特征减一个均值”。这是一次固定的 premise 检查，不用于选 predictor 配置。
3. constant-logit availability invariance 测试通过。
4. missing target 的 residual/evidence 确实不进入任何 pooling；全 missing text 秒不会因有效项数量变化获得固定 score shift。

以上任一项失败，均为 **premise kill**：不启动正式 localizer pilot，不通过改 context radius、projection width 或 predictor capacity 追结果。

### 最小 arms

1. `core_fixed_innovation`：observed + predicted + fixed-scale residual；
2. `standard_masked_shared`：observed + predicted，无 residual；
3. `raw_capacity_receptive_field_matched`：匹配参数、depth、时间 receptive field 与预算；
4. `same_modal_temporal_residual`：区分 cross-modal condition 与一般 reconstruction residual；
5. `availability_only`：只读 modality availability、ASR coverage 与长度；
6. `core_residual_zero`、`core_residual_permuted`、`core_residual_matched_noise`：对同一个 core checkpoint 的 inference intervention，不重新训练。

原稿的 `shuffled_condition` 可作为额外 diagnostic，但不能替代第 6 组，因为它同时改变 predictor quality 与输入分布。

### 机制通过条件

1. core 相对 `standard_masked_shared` 与 matched raw 在 HMM/HCS 的 test within-video ROC 同向提高，且至少一边达到预注册的 `0.020`；
2. core 优于 same-modal residual，证明不是普通时间重建；
3. 在同一 checkpoint 上，zero/permuted/matched-noise 三种 residual intervention 中至少两种，必须在 HMM/HCS 两边各自消除至少一半 `core - standard_masked_shared` 的 within-video ROC 增益，且至少一个语料的绝对下降达到 `0.010`；同时必须实质改变 eligible videos 的 within-video frame ranking。只看 branch 权重或 residual norm 不算；
4. 增益不能由 availability-only 模型、ASR coverage、视频长度、residual norm 或 predictor reconstruction error单独解释；
5. 改善应集中在预注册的 MultiHateLoc fused 败给至少一个 unimodal branch 的视频，但不能用 test GT 选择推理路径；
6. 两语料六个固定 SOTA 单元全部严格过门才扩 MHC-EN/ZH。若机制 gate 失败，不围绕 context radius、projection width、fusion temperature 或 reconstruction loss 做 test-driven sweep。

### 明确的 mechanism kill

出现以下任一情况即 **STOP 并归档**，不把结果解释成“方向正确但需要调参”：

1. core 未同时优于 shared-only 与 matched raw，或 HMM/HCS 的 within-video ROC 方向不一致；
2. same-modal temporal residual 匹配或超过 core，说明 cross-modal condition 不是必要机制；
3. residual interventions 未达到上一节第 3 条，或干预前后逐视频排序仍近乎不变，说明 MIL 忽略 residual；
4. availability-only、ASR coverage 或有效 evidence 数即可重现主要增益，说明模型使用 missingness shortcut；
5. 增益只存在于 residual norm / reconstruction-error 高的样本，而 learned evidence direction 没有超越 norm-only control，说明方法退化为 generic reconstruction anomaly score；
6. mechanism gate 通过但两语料六个固定指标未全部 SOTA：机制可以记录为有效负结果，但按项目晋级规则仍不扩另外两语料、不作为当前主方法。

## 最终裁定

来源方法 gate 通过，exact target-task mechanism 尚未发现直接占位；adaptation 也不只是把 reconstruction error 当 anomaly score，任务故事成立。但领域邻近度很高，SAGE 已占 modality-specific hate preservation，MISA/MFM 已占 shared/private，Gabeur/CrossMAE/CAV-MAE 已占 cross-modal prediction。因此 novelty 只能给 **5.8/10**。

**GO 的含义是批准一次经上述三项公式修复后的最小 pilot，不是批准当前 README 原公式直接实现。** 若保留可塌缩 `P_m`、逐帧 residual LayerNorm 或未归一化 log-sum-exp，则在训练前改判 STOP；若 MIL 最终可忽略 residual 且 intervention 不伤排序，则运行后改判 STOP，并归档为 masked-modality representation + MIL 的负结果。
