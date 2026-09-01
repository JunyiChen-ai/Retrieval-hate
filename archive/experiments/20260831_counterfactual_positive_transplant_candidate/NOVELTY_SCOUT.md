# Counterfactual positive-window transplant：独立 novelty review

截至日期：2026-08-31（Pacific/Auckland）  
检索依据代码 commit：`13257e004fc1d306e2dfbadb4e93317062513f83`  
性质：focused prior-art search；只评 novelty 与可证伪性，不实现方法、不报告性能。

## 结论先行

**Verdict：STOP，novelty 3/10。** 不建议把该候选作为下一轮主方法进入正式训练；最多可作为一个低成本机制诊断。检索到期日前未发现与“同语料 weak-positive selector → 全部对齐模态移植进 negative host → 原负例/合成视频配对 → edit-mask ordinal student → teacher-free inference”逐项完全相同的一手论文，但这组差异主要是已有组件的特定拼接，不是清楚的新学习原理。

最直接的碰撞是：

1. **Temporal VideoMix 已经在 WTAL 上把一个视频的连续时间块替换进另一个视频，并直接在 temporal feature axis 操作；**所以跨视频 temporal transplant 及其弱定位用途不新。
2. **AlMarri et al. 2024 已经在 WS-VAD 中随机交织 normal/anomaly videos 的连续 segments，生成 virtual events，并用已知合成结构训练 boundary/center heads；**所以“normal/anomaly segment mixing 提供合成的局部结构监督”也不新。
3. **PivoTAL、RefineLoc、RSKP 等已经用弱模型挖掘高置信 action snippets / snippet pseudo labels，再训练或修正定位头；**所以 frozen evidence selector 到 student 的信息流属于常规 pseudo-label distillation。
4. **Aich 2023、Rai 2024、PA-VAD 2025 等已经从 normal data 构造带局部或视频级伪异常监督，并明确讨论 synthetic artifact / domain-gap shortcut；**所以“原 normal 与 synthetic abnormal 配对训练、推理不需要生成器”这个上位范式也已拥挤。

本候选的根本语义问题不是有没有准确的 splice mask，而是：**splice mask 只精确标出哪里发生了编辑，不证明那里是 hateful。** donor 来自 positive bag，但窗口是 frozen selector 推断出来的，因而仍是未校准的 pseudo-positive。若直接要求 mask 内 residual/order 高于 mask 外，训练标签只是把 selector 的判断换一个 host 重放；若 gains 不超过“在原正例窗口上直接蒸馏同一 selector”，transplant 没有新增监督信息。

因此不能把该方法称为“dense positive supervision”“certified positive transplant”或可识别的“causal effect of hate”。最窄、勉强可辩护但必须由实验支持的表述只有：

> **Selector-conditioned cross-video transplantation constructs matched negative-host pairs that convert uncertain positive-window hypotheses into within-host ordinal constraints.**

这仍然是一个 conditional empirical claim，不是当前已成立的 novelty claim。POWA frozen anchor、residual head、score transport/readout 均不计入本文 novelty。

## 形式化边界：已知的是 change，不是 positive label

对某个语料独立定义：negative host 为 `x^-`，positive bag 为 `x^+`，冻结 selector 为 `q`，选择窗口 `w=q(x^+)`。把 `w` 的全部对齐模态放入 `x^-` 的随机位置 `r`，得到

\[
\tilde{x}=\operatorname{splice}(x^-,w,r),\qquad
m_t=\mathbf{1}[t\in r:r+|w|).
\]

由训练标签和编辑过程确知的只有：

- 在数据集的 negative-bag 假设下，原 host 的各秒是 label-certified negative；
- `m` 是精确的 **change/edit mask**；
- `x^-` 与 `\tilde{x}` 除移植区及其边界效应外形成 matched pair。

不能由此推出：

- `w` 中任一秒、或全部秒，满足数据集 hateful policy；
- 合成视频的 bag label 必为 positive；
- `m` 是 dense hateful mask；
- `f(\tilde{x})-f(x^-)` 识别了 hate 的因果效应。

正例 bag 只保证某处存在正实例，不保证 selector 选中它，也不保证整个连续窗口同质。更重要的是，hate 往往依赖话语对象、前后文、说话人立场与视听组合；把窗口移入另一 host 会改变这些变量。全部模态一起移植减少模态错配，却同时引入同步的身份、场景、音色、ASR 分段和边界变化。因此该操作是 **synthetic intervention / counterfactual augmentation**，不是满足可识别条件的 causal intervention。

若排序损失写成

\[
\mathcal{L}_{\mathrm{mask}}=
\ell\!\left(\operatorname{Agg}_{t:m_t=1} h_t(\tilde{x}),
\operatorname{Agg}_{t:m_t=0} h_t(\tilde{x})\right),
\]

其正向语义必须显式条件于“`q` 选对 donor”。若不估计 selector precision/coverage，也不设 abstention 或 noise robustness，这只是 hard pseudo-labeling。paired loss 可以减少 host-level nuisance，但不能凭空提高 donor label 的可信度。

## 最接近的一手先例

| 接近度 | 一手来源 | 已有机制 | 与本候选的剩余差异 | Novelty 含义 |
|---|---|---|---|---|
| **最高** | [AlMarri et al., *A Multi-Head Approach With Shuffled Segments for Weakly-Supervised Video Anomaly Detection*, WACV Workshops 2024](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/AlMarri_A_Multi-Head_Approach_With_Shuffled_Segments_for_Weakly-Supervised_Video_Anomaly_WACVW_2024_paper.html) | 从 normal/anomaly video pair 连续抽段并随机交织成 virtual events；共享 encoder 上学习 anomaly、boundary、center heads，利用已知合成边界/中心结构。 | 本候选不是随机交织多个段，而是 selector 选 weak-positive window，插入一个 certified-negative host；拟用原/合成 matched pair 做 ordinal residual。 | 直接占据“WS-VAD 的 normal/anomaly segment mixing + synthetic local structure supervision”。本候选只能声称更窄的 matched-host ordinal 组织。 |
| **最高** | [Yun et al., *VideoMix: Rethinking Data Augmentation for Video Classification*, 2020](https://arxiv.org/abs/2012.03457)（[作者论文 PDF](https://sangdooyun.github.io/data/yun2020videomix.pdf)） | 从一个视频剪下 cuboid 插入另一个视频；Temporal VideoMix 沿 temporal feature axis 替换连续段并混合标签；论文直接在 THUMOS14 WTAL 上验证。 | donor 随机、标签按替换比例混合，没有 weak-positive selection、原 host paired ordinal loss。 | temporal transplant、feature-level replacement 与 WTAL 应用均不新；剩余差异是 selector 与 loss，不是算子。 |
| 很高 | [Rizve et al., *PivoTAL: Prior-Driven Supervision for Weakly-Supervised Temporal Action Localization*, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Rizve_PivoTAL_Prior-Driven_Supervision_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html) | Base WTAL head 产生 confidence-aware pseudo-action snippets，再用这些片段训练 prior-driven localization head。 | 不把片段移到 negative host，也没有 matched edit pair；其 pseudo snippets 仍留在原上下文。 | 占据“高置信 weak snippets → 第二定位头”的核心信息流。必须证明 transplant 比直接 pseudo-label student 多了信息。 |
| 高 | [Pardo et al., *RefineLoc: Iterative Refinement for Weakly-Supervised Action Localization*, WACV 2021](https://openaccess.thecvf.com/content/WACV2021/html/Pardo_RefineLoc_Iterative_Refinement_for_Weakly-Supervised_Action_Localization_WACV_2021_paper.html) | 用弱模型生成 snippet-level pseudo foreground/background，再以伪标签监督后续 attention/localization model。 | 没有跨视频编辑和 negative-host pairing。 | 进一步说明 selector/teacher-free student 不是新概念；donor mask 若按正标签训练，本质仍是 pseudo-label self-training。 |
| 高 | [Huang et al., *Weakly-Supervised Temporal Action Localization via Representative Snippet Knowledge Propagation*, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Weakly-Supervised_Temporal_Action_Localization_via_Representative_Snippet_Knowledge_Propagation_CVPR_2022_paper.html) | 挖掘高置信 representative snippets，经 intra/inter-video knowledge propagation 生成更好的 snippet pseudo labels并修正主分支。 | 传播而非 copy-paste；没有 matched negative host。 | “跨视频运输高置信局部知识”已经存在；只有显式 transplant pair 的归纳偏置尚可区分。 |
| 高 | [Aich et al., *Cross-Domain Video Anomaly Detection Without Target Domain Adaptation*, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Aich_Cross-Domain_Video_Anomaly_Detection_Without_Target_Domain_Adaptation_WACV_2023_paper.html) | 向 normal frames 加 foreign objects，生成 pseudo-abnormal examples，并以 real normal / pseudo abnormal 的相对差异训练 normalcy classifier。 | 空间 foreign-object synthesis、normal-only/cross-domain setting；不是从真实 positive bag 选连续多模态窗口。 | 占据“normal host + synthetic positive region + relative classifier”的上位范式；真实 donor 和 temporal/all-modal 只是实现差异。 |
| 中高 | [Hashimoto et al., *PA-VAD: Diffusion-Based Pseudo-Only Video Anomaly Detection via Domain-Aligned Memory Updates*, 2025](https://arxiv.org/abs/2512.06845) | 用 CLIP/VLM 辅助选择条件并生成 pseudo-abnormal videos，与 real normal videos训练；专门发现并缓解 synthetic anomaly 的 feature-magnitude/artifact bias。 | diffusion 生成整段伪异常，不使用真实 positive donor 或 edit mask。 | 原 normal / synthetic abnormal 配对训练已知；更重要的是，它实证 synthetic artifact 会被 MIL 放大为 shortcut，正中本候选最大风险。 |
| 中高 | [Rai et al., *Video Anomaly Detection via Spatio-Temporal Pseudo-Anomaly Generation: A Unified Approach*, CVPR Workshops 2024](https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/K._Video_Anomaly_Detection_via_Spatio-Temporal_Pseudo-Anomaly_Generation__A_Unified_CVPRW_2024_paper.html) | 在 normal videos 上通过 diffusion inpainting 与 optical-flow mixup 制造空间/时间伪异常，并用其训练重建和语义判别。 | one-class VAD；伪异常是扰动，不来自 positive videos。 | “带已知被修改区域的 spatio-temporal pseudo-anomaly supervision”不是新概念。 |
| 中 | [Hu et al., *CITAL: Counterfactual Intervention for Temporal Action Localization With Point-Level Annotation*, Neurocomputing 2025](https://doi.org/10.1016/j.neucom.2025.130006) | 生成保留 contextual clues、移除 action 的 counterfactual inputs，比较原/反事实响应来抑制 background；使用 point supervision。 | 干预方向相反，且监督更强；没有跨视频 positive transplant。 | “counterfactual intervention for temporal localization”这一标题级 claim 已被占用；本候选不能只靠 counterfactual 命名取得 novelty。 |
| 基础算子 | [Yun et al., *CutMix*, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html) | 跨样本剪贴区域并按面积混合标签，展示弱监督定位收益。 | 图像空间而非视频时间；没有 MIL/paired ordering。 | copy-paste 与“局部修改促进弱定位”均是基础先例，不能计入贡献。 |

截至检索日，公开检索还覆盖了 2025–2026 的 synthetic/pseudo anomaly generation、WTAL pseudo-label/self-training、causal/counterfactual temporal localization 与 hateful-video localization。未找到完整逐项等价实现不等于该组合具有强 novelty；这里的主要问题是已有组件是否形成新的、非显然的监督量，当前答案是否定的。

## 项目内重复与方向反转风险

该候选还与项目内两条已淘汰路线高度相邻：

- `archive/experiments/20260831_powa_benign_insertion_pilot/` 已做过同语料连续窗口 transplant 的反方向版本：negative donor → positive recipient。其 novelty scout 已把 VideoMix 与 AlMarri 2024 定为最接近先例；正式 pilot 虽提高 HateMM within-video ROC，但 pooled AP/ROC 分别下降约 `.0529/.0543`，最终被固定 gate 淘汰。
- V26 Counterfactual Temporal Witnesses 已用原输入/替换输入的 exact replacement effect 做局部 witness；视频判别强，但替换效应未对齐真实 span，且 learned negative reference 不是必要条件。

正向移植不是这两者的数学同义词，但仅把 donor/host 方向互换，再把监督从“known benign”改成“selector-chosen positive”，反而失去了上一候选最清楚的 label-certified 语义。若没有证明 paired negative host 能系统性去除 selector 的 context bias，而不只是放大 donor 或 splice artifact，这更像 rejected augmentation 的变体，不构成新 core。

## 能否 defend 的最小 claim

只有在下述所有归因证据成立后，才可把候选从 STOP 升为 **conditional GO（最高约 5/10）**：

> 在 video-level weak supervision 下，将同语料 selector-chosen positive hypotheses 移入 matched certified-negative hosts，并对原/合成 pair 施加 edit-conditioned ordinal loss，比直接 pseudo-label distillation 更能隔离 host-level nuisance，从而改善原始未编辑视频的 within-video ordering。

这个 claim 的边界必须同时写清：

- donor 是 **selector-chosen / hypothesized positive**，不是 certified positive；
- mask 是 **edit mask**，不是 ground-truth hateful mask；
- 机制主张是 matched-host nuisance control / ordinal constraint，不是 Copy-Paste、pseudo anomaly generation、teacher-student、POWA anchor 或 score transport；
- “counterfactual”只能作为合成 paired example 的描述，不能声称可识别的 causal effect；
- 每个主数据集独立、train-only 选 donor 和 host，不得跨语料或使用 val/test span 选择 selector/threshold。

即使 controls 全过，该 claim 仍需和 AlMarri 2024、VideoMix、PivoTAL 做逐项表格对照；若新增收益只来自 teacher、splice 或 edit mask任一常规组件，则不能保留组合式 claim。

## 致命 anti-pattern

出现任一项即可直接否决：

1. **把 edit mask 当 dense positive GT。** 这是逻辑错误；它只准确标记编辑位置。
2. **把 paired difference 写成 hate 的 causal effect。** donor selection 非随机，intervention 同时改变语义、身份、场景、边界和上下文，不满足可识别条件。
3. **没有 direct teacher-distillation control。** 若不比较“同一 selector、同一 coverage、直接在原 positive windows 蒸馏”，无法证明 transplant 新增任何信息。
4. **模型主要识别 splice。** 全模态同步跳变、音频相位/音色变化、ASR chunk reset、位置编码、padding 或 video identity 都可能比 hate 更容易。
5. **只在 synthetic mask accuracy 上成功。** 训练 mask 本来就容易被编辑痕迹预测；必须改善原始 test videos 的 within-video ROC/AP，并保持 pooled AP/ROC。
6. **selector 与 student 循环确认。** 若 donor 由 POWA/VERA 高分选出，student 又被迫复现其排序，而没有 independent precision/coverage audit，这只是 teacher sharpening。
7. **用 span GT 调 selector coverage、窗口长度或阈值。** 任何 val/test span 进入 donor 选择、checkpoint 或超参决定都会使弱监督 claim 失效。
8. **依赖单一语料或特定编辑层。** 只在一个数据集、一个模态或一个 splice 实现有效，更像 dataset/edit artifact；按项目门槛至少需两个主数据集方向一致。
9. **事后用 calibration/ensemble 补 pooled 指标。** 这会掩盖 residual 对 frozen anchor 的破坏，不能用于救候选。

## 若仍做 conditional diagnostic：最小可证伪 controls

不建议直接做完整四语料训练。若因实现成本极低而保留一个 pilot，先在两个主数据集各自 train-only 运行以下同预算矩阵；所有组使用同一个 frozen selector、相同 donor coverage、同一个 residual head 与相同 POWA anchor：

| 组别 | 作用 | 必须回答的问题 |
|---|---|---|
| A. POWA + residual，no extra local loss | 容量/训练基线 | 新 head 本身是否改变指标？ |
| B. direct pseudo-label distillation | 在 donor 原视频原位置用同一 selector 伪标签训练，不 transplant | 若 B≈完整候选，transplant 没有新增监督信息，STOP。 |
| C. transplant，no paired/mask loss | 完全相同编辑，仅保留原 MIL | 若 C≈完整候选，收益只是普通 augmentation。 |
| D. Temporal VideoMix/CutMix matched baseline | 同 donor/host/长度预算，使用标准混合 bag label 或无 edit-conditioned ordinal | 完整候选是否优于最直接的已知算子？ |
| E. 完整候选 | selected positive donor → negative host + paired edit-conditioned ordinal | 仅 E 稳定优于 A–D，才可能支持最窄 claim。 |

在这个矩阵之外，四个 control 不可省略：

1. **Negative-donor splice control**：同语料 negative donor、相同长度/host/位置/模态/边界。若也产生相同 gain，模型利用的是 foreign-window 或 splice，而非 positive hypothesis。
2. **Random-positive / low-confidence donor control**：从 positive videos 随机取窗或取 selector 低分窗，coverage 完全匹配。高置信 donor 必须明显更好，且增益随 selector 质量单调，而不是随剪辑强度变化。
3. **Shuffled-mask control**：保留相同合成视频，随机平移/置换监督 mask。若不显著退化，edit-conditioned supervision 没有 load-bearing。
4. **Artifact-matched neutral splice + boundary audit**：从另一 negative video 移入窗口，并使用同样转场、归一化、padding、ASR/audio 处理；固定排除边界 buffer，分别报告 boundary 与 interior。另训练一个不看语义标签的 mask probe；若它能在 neutral splices 上高精度找出 mask，或候选收益集中于边界，立即 STOP。

建议再加一个 **all-modal vs one-modal transplant** 诊断，但不能据 test 结果挑版本：视觉、音频、文本分别移植与全模态移植的差异可判断 student 在学语义 conjunction，还是某一模态的编辑/身份 cue。

### 预注册的通过条件

只有同时满足以下条件才值得重新查新：

- 至少两个主数据集上，原始未编辑 test videos 的 within-video ROC 与 within-video AP 均优于 A–D，方向一致；
- pooled frame AP/ROC 相对 frozen POWA 不出现需要后验修补的实质下降；
- E 明显优于 direct teacher distillation、Temporal VideoMix、splice-only、random/negative donor；
- shuffled mask 消除主要收益，且收益存在于远离 splice boundary 的 interior；
- selector precision/coverage 只用 train 机制确定，并在可用的 held-out span 上仅作冻结后的审计；高置信 donor 的质量确实高于 random/low-confidence donor；
- neutral-splice artifact probe 不能解释 E 的排序收益，且 residual 在原始视频而非 synthetic samples 上保持非零、稳定的 within-video variance。

若这些条件通过，下一步不是立即扩大训练，而是围绕真正 load-bearing 的“matched-host ordinal nuisance control”重新做第 12 步深查新。若 B、C、D 或任何 artifact control 与 E 等效，本候选应按 STOP 归档，不再增加 selector、teacher ensemble、policy routing或额外 loss 修补。

## 检索范围与判断置信度

检索仅使用论文主页、CVF/ECVA/OpenReview/期刊官方页面、作者论文页与 arXiv 原文等一手来源。关键词覆盖：`temporal VideoMix/CutMix`、`WS-VAD shuffled segments`、`positive/anomaly snippet mixing`、`pseudo anomaly generation`、`WTAL pseudo-action snippet distillation`、`weakly supervised localization augmentation`、`counterfactual/causal temporal localization`、`synthetic anomaly artifact/domain gap`、`hateful video localization`，并补查 2025–2026 工作。

判断置信度：**中高**。没有发现完整逐项同构论文，但最高风险并非漏掉一篇完全同构工作，而是候选的每个有效信息源都已有直接先例，且 edit mask 与 semantic label 被混淆。除非最小 controls 证明 matched negative host 提供了 direct pseudo-labeling 无法获得的可迁移 ordering signal，否则没有足够理由把它作为新方法核心。
