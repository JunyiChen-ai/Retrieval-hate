# Monotone-warp rank equivariance：独立 novelty 与可识别性审查

截至 2026-08-31。审查对象为本目录 `README.md`。本轮只检索一手来源并做数学/机制审查；未实现、未训练、未生成 prediction。

## 裁定

**STOP**  
**novelty：4.6 / 10**

按最新三项硬门逐项裁定：

1. **允许 adaptation 外部方法：PASS。** 使用 Gong et al. 的 temporal-transform equivariance 本身合规。
2. **source/core 尚未进入 hateful-video detection/localization：PASS（在本次检索范围内）。** 检查 MultiHateLoc、HateClipSeg、LELA 及相关 hateful-video temporal-localization 结果，未发现 Gong 2021、monotone-warp equivariance或 ordinal pullback consistency 被用于该目标任务。
3. **必须是 non-trivial task adaptation：FAIL。** Gong 2021 已在只有 video labels 的 WTAL 中同时占用 temporal resize/window warp/attention-guided warp、原/变换 CAS equivariance、双分支 video classification，以及 adversarial transform selection。当前候选把其 pointwise/distribution consistency 换成 pairwise sigmoid-JS，并把动机从 proposal boundary 换成 within-video ROC；但所写 loss 在数学上并非纯 ordinal，完整 pair graph 下等价于 pullback scores 相差一个视频级常数。其余 adversarial finite-bank selection是来源方法 adversarial policy的简化 hard-example mining。它没有为 hateful localization 注入新的监督信息，也没有解决 MultiHateLoc 的 time×modality ownership错配。

因此该方案最多是 **Gong-style temporal equivariance + pairwise relative-score consistency** 的目标指标适配，不能作为新方法进入正式实现。即使 test 有增益，也只能说明 temporal augmentation/consistency regularization 有用，不能支持“学习了正确 hate ordering”或“修复了 fragmentation shortcut”。

## 一手先例与占位边界

### 1. Gong et al. IJCAI 2021：几乎完整占用外部 core

Gong et al. 在 weakly supervised temporal action localization 中只用 video-level action labels，采用原视频与 temporal-transform 视频的 Siamese localization network；变换包括 resize、window warp与 attention-piloted time warp。它把原 CAS 经过同一 temporal transform 后与变换视频 CAS 对齐，并用双向 KL 做 equivariant consistency；另有 policy network选择使 classification/consistency loss最大的 transform：

- Gong et al., *Self-Supervised Video Action Localization with Adversarial Temporal Transforms*, IJCAI 2021，[官方论文页](https://www.ijcai.org/proceedings/2021/96)，[官方 PDF](https://www.ijcai.org/proceedings/2021/0096.pdf)。

来源论文还明确以局部变速改变 duration/boundary、密采 action、稀采 background，并用 attention-complemented warp挖被当前 attention遗漏的短 action。这不只是普通全局 speed augmentation；它已经讲了候选当前声称的“改变局部 duration/采样密度、攻击 discriminative-fragment shortcut、要求 pullback localization稳定”的核心故事。

候选与其真实差异仅有：

- 变换限制为固定、严格单调、分段线性 warp；
- 用 pairwise sigmoid-JS 替代 temporal-softmax CAS 的双向 KL；
- 用有限 bank 的最大 disagreement替代 learned policy；
- test 输出原视频单次 frame score，而非 threshold proposals。

最后一点是本项目输出协议，不是学习机制 novelty；第三点比来源更简单；第一点只是变换族收窄。因此 novelty完全依赖第二点，而该点存在下述等价退化。

### 2. WTAL 中 relative/rank consistency 已有

P-MIL 已在 video-label WTAL 中提出 instance-level rank consistency：在重叠 proposal cluster 内，把 RGB 与 FLOW proposal scores经 softmax归一化，再用双向 KL 对齐相对排序/相对分数，以提高 NMS 后检测稳定性：

- Ren et al., *Proposal-Based Multiple Instance Learning for Weakly-Supervised Temporal Action Localization*, CVPR 2023，[CVF 官方页](https://openaccess.thecvf.com/content/CVPR2023/html/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html)，[arXiv:2305.17861](https://arxiv.org/abs/2305.17861)。

P-MIL 对齐的是 modality views 与 proposal clusters，不是 clean/warped seconds，所以没有逐项占用本候选。但它已占用“让弱监督时序候选的相对 score/rank跨 view一致”这一宽 claim。候选不能 claim 首次 WTAL rank consistency。

Bi-SCC 等 WTAL 方法也已使用 temporal context augmentation与跨 view prediction consistency以抑制 co-scene shortcut：

- Li et al., *Weakly-Supervised Temporal Action Localization with Bidirectional Semantic Consistency Constraint*, 2023，[arXiv:2304.12616](https://arxiv.org/abs/2304.12616)。

更一般的 temporal-equivariant representation/pretraining亦已有直接 TAL 先例，例如：

- Zhang et al., *Unsupervised Pre-training for Temporal Action Localization Tasks*, CVPR 2022，[作者 PDF](https://tianyu-yang.com/resources/up-tal.pdf)；
- Jenni & Jin, *Time-Equivariant Contrastive Video Representation Learning*, ICCV 2021，[CVF 官方页](https://openaccess.thecvf.com/content/ICCV2021/html/Jenni_Time-Equivariant_Contrastive_Video_Representation_Learning_ICCV_2021_paper.html)。

这些并不等同于 hateful-frame ordinal pullback，但说明 temporal equivariance、pairwise relation和 weak localization consistency均为成熟组件。

### 3. Hateful-video 目标领域检索

代表性目标工作包括：

- MultiHateLoc，[arXiv:2512.10408](https://arxiv.org/abs/2512.10408)：modality-aware temporal encoders、dynamic fusion、cross-modal contrast与 MIL；
- HateClipSeg，[arXiv:2508.01712](https://arxiv.org/abs/2508.01712)：segment-level benchmark与 temporal localization；
- LELA，[arXiv:2602.09637](https://arxiv.org/abs/2602.09637)：training-free multimodal/LLM frame scoring。

本次没有检到这些工作使用 Gong temporal transforms、warp-pullback equivariance或 rank-equivariance training。因此硬门 2 通过；STOP 不是因为来源已在 hateful task 被使用，而是 adaptation 没有形成新的可识别机制。

## 数学审查

### 1. 所写 sigmoid-JS 并非纯 ordinal loss

候选定义

`q_ij = sigmoid((s_i - s_j) / tau)`，

并令 clean `q_ij` stop-gradient，最小化其与 pullback `q^w_ij` 的 Bernoulli JS divergence。若 loss 为零，则 JS 的严格性给出 `q_ij = q^w_ij`。sigmoid 在有限值上单射，因此：

`s_i - s_j = s^w_i - s^w_j`。

若采样 pair graph连通，对任意参考节点 `r` 都有：

`s^w_i - s_i = s^w_r - s_r = c_video`。

即 pullback curve必须等于 clean curve加一个视频级常数。这是 **pointwise consistency modulo additive offset**，不是只保留排序的 ordinal consistency。它确实不约束整体平移/calibration，但仍约束所有 pairwise差值和尺度。

这与 within ROC 的不变性并不对齐：within ROC 对任意严格单调变换 `g(s)` 都不变，而该 loss 对 `g(s)=a s`（`a != 1`）通常不为零。`tau` 因而成为隐式 score-scale超参数。README 所称“只要求排序概率一致、不要求 calibration一致”只对 additive shift成立，对尺度和非线性 monotone calibration不成立。

若把 `tau` 设得很小，让 sigmoid饱和以近似只比较符号，则又产生数值退化：同号但幅度任意的 pairs loss近零、梯度消失，真正接近 boundary的 pairs才有梯度。这个 regime 与稳定的纯 rank objective也不等价。

### 2. 若改成真正 ordinal，错误排序会被原样保存

即使把 loss改为 sign/Kendall-only，warp equivariance只能说明“同一内容在两种采样下给出同一排序”，不能说明哪一秒应高于哪一秒。clean target来自当前模型自身，且没有 temporal GT：

- clean 初始 false-positive若在 warp后稳定，loss认为它正确；
- 不稳定但正确的细短 hate cue可能被 adversary与插值反复压制；
- positive bag中 hate–benign、hate–hate、benign–benign pair均未知，uniform pair sampling不给 cross-class pair方向；
- negative bags不存在有意义的 hate ordering，却仍会被迫保持任意 benign内部排序。

stop-gradient只防止单个 step 中 clean/warp两端同时朝中点移动，不会把 clean side变成可靠 teacher；共享参数更新后，下一步 clean target本身仍漂移。该约束提供的是几何自洽，不是新的 label information。

因此它可能作为 regularizer改善 sampling robustness，但无法从 video labels识别“正确的 within-video hate ranking”。尤其 MultiHateLoc 的固定 top-third MIL只监督少数高分证据；rank consistency可能把其早期错误高分顺序保护起来，而不是纠正它。

### 3. 常数、广播与尺度退化

当每视频所有 seconds具有相同 logit时，所有 `q_ij = 0.5`，rank loss严格为零。video MIL仍可通过视频级/全局 context使 positive videos整段高、negative videos整段低；这正是 pooled指标可高而 within ROC为 `.5` 的 broadcast shortcut。README 的 temporal-total-variation diagnostic能在实验后发现 collapse，但训练 objective没有排除它。

另一个近似退化是把所有非零差值放大到 sigmoid饱和区：只要 warp不翻转符号，loss与梯度都很小。反之，把差值全部缩向零也会降低许多 disagreement，同时 bag score可由视频级 additive offset维持。故仅报告 total variation“不整体塌缩”不够，还必须报告 per-video score standard deviation、pair margin分布、tie rate及 eligible-positive/negative分别统计。

### 4. 离散 monotone warp不是精确可逆的数据变换

连续严格单调 `w` 保持顺序；但 1 fps 离散序列经 stretch/compress、插值和重新采样后，通常发生 frame duplication、丢失与低通混合。`w^{-1}`只能拉回坐标，不能恢复被 downsampling丢掉的内容。于是 clean/pullback disagreement混合了：

- 模型真正的 duration/position shortcut；
- interpolation与anti-alias实现；
- temporal encoder对重复帧/丢帧的正常响应；
- 边界处一对多或多对一 correspondence误差。

最大 disagreement selector会优先选数值 aliasing最严重的 warp，并不必然选最有任务意义的 shortcut。若无 identity-on-bandlimited synthetic test、round-trip error和有效 correspondence mask，所谓 adversarial机制可能只是“对重采样 artifact最坏情况训练”。

### 5. Adversarial warp 没有新增方向监督

在固定 bank 中取当前 disagreement最大者，是 hard-example selection。Gong 2021 已用 learned policy最大化 transformed classification与consistency loss，并比较 random versus adversarial selection。候选的 selector只改变“在哪个已有 equivariance violation上施加同一个自洽约束”，不会判断 clean排序是否正确，也不针对 multimodal ownership。

因此“adversarial”不解决 identifiability；它最多提高 regularization强度。若候选最终优于 random warp，也只说明 worst-case augmentation有用，不能说明攻击到了 hate-specific duration shortcut，除非同时证明被选 warp与预注册 fragmentation/duration failure相关、且不由 resampling error解释。

## 与项目既有证据的关系

### POWA orbit probe

既有 cyclic-origin probe表明 POWA 对 absolute temporal origin已近似 equivariant，且 cyclic view不优于 matched arbitrary permutation。它只否定 origin-shift bottleneck，**不直接否定 local monotone speed warp**。所以当前候选没有被该 probe形式上重复，但必须把 claim限制为 local sampling-density robustness，不能重新包装为一般 temporal-position equivariance。

### POWA rank transport

既有 rank-transport pilot固定 POWA score multiset、只学习 temporal assignment；它在双语料未过冻结机制门，并被 center-first/shifted-mask/direct-additive controls削弱归因。它与本候选不同：前者改时间归属，后者要求同一模型跨 warp自洽。然而它提供了相关负证据：当前语料中“改变/保持 ordering”不自动转化为足够 within增益，且简单位置先验可胜过 learned ordering。

### Pairwise ordinal / distillation 既有路线

项目中 span-transfer、consensus distillation及其他候选已多次使用 within-video pairwise ordinal loss；该 loss family不能再计 novelty。本候选唯一可能的新组合是“monotone warp对应关系作为 self-pair来源”，而这又被 Gong temporal equivariance直接压窄。

## 当前 controls 的评价

`source_pointwise_eq`、`warp_bce_only`、`no_warp`是合理最低 controls，三指标立即 test与不使用 test-time ensemble也合规。但它们不足以支持机制归因：

1. 缺 **random-select same-bank** arm，不能证明 adversarial选择超出普通 temporal augmentation；这是 Gong 论文自己的关键 control。
2. 缺 **same-warp true ordinal/sign-only** 与当前 sigmoid-difference arm，不能证明增益来自 ROC-aligned ordering而非 relative-logit magnitude matching。
3. 缺 **broadcast/constant-score capacity control**，TV gate只是事后诊断，不排除 video-offset shortcut。
4. 缺 **same-view duplicate-forward consistency**，用于量化 dropout/optimizer噪声造成的表观 disagreement。
5. 缺 **resampling-null control**：对无 temporal encoder的 framewise scorer、bandlimited synthetic curve和 identity/round-trip interpolation测量 pullback误差。
6. 缺 **positive-only versus all-bag rank consistency**。negative bag的内部顺序没有 label语义，可能贡献大量无关梯度。
7. 缺 **pair graph coverage**：每视频连通性、pair temporal-distance分布、有效 correspondence coverage、tie/saturation比例。
8. `source_pointwise_eq` 必须明确是相同固定 warp bank下的 matched pointwise loss；若同时复现 Gong self-refine或 learned policy，就不再是单变量 control。

README 的 non-monotone permutation diagnostic可以发现完全忽略内容/常数输出，但不是充分 anti-collapse gate。一个按局部内容但排序错误的模型会对 permutation敏感，仍可通过该 diagnostic。

## 允许与禁止的 claim

当前 STOP，不应形成方法 claim。作为 diagnostic/baseline，最多允许：

> We test whether replacing score-level temporal-warp consistency with relative-logit consistency improves sampling robustness of a weakly supervised hateful-video localizer.

不得 claim：

- 首次 adversarial temporal transform、首次 temporal equivariance或首次 WTAL warp consistency；
- 首次 rank consistency / pairwise ordering用于 weak temporal localization；
- loss只保留 ordinal information或严格 metric-aligned，除非改掉 sigmoid-difference等价；
- 该约束能从 video labels恢复正确 hate ordering；
- disagreement下降证明 fragmentation、duration shortcut或 boundary completeness被修复；
- source method已被实质重新发明为 hate-specific mechanism。

## 若仅作低成本诊断，最小边界

本裁定不建议正式实现为 novelty candidate。若主线仍要把它作为不晋级的来源 baseline/diagnostic，边界必须是：

1. 只做 `no_warp / warp_bce / random pointwise / adversarial pointwise / random relative / adversarial relative` 的 matched matrix；
2. 固定相同 warp bank、forward预算、训练 steps与模型容量；
3. 先做 train-only synthetic pullback与constant/broadcast tests，再训练；
4. 正式训练后立即在 HMM/HCS test报告三指标，validation只选各固定 arm checkpoint；
5. 同时报告 clean/pullback disagreement、TV、score std、tie/saturation、round-trip error及 warp selection分布；
6. 只称 source-method adaptation baseline，不进入 novelty claim。

要重新提交为 GO，不能只补 controls。必须新增一个由同语料 train video labels可识别、能够给不稳定 pair提供**方向性纠错**而非仅要求自洽的 task mechanism，并证明它针对 time×modality ownership或已证实 objective bottleneck；否则仍是 Gong 2021 的 ordinal loss substitution。

## 最终理由

候选的 failure observation真实，local monotone warp也不同于已经失败的 cyclic-origin probe；但 novelty审查对象是学习机制，不是动机。Gong 2021 已在相同监督结构的 WTAL 中完成 adversarial temporal warp + pullback localization consistency，P-MIL又占用了 relative/rank consistency。候选的 sigmoid pair loss在连通 pair graph上等价于 score curve相差常数，并非宣称的纯 ordinal约束；若改成真正 ordinal，又只能保存当前排序，包括错误排序与常数/broadcast解。adversarial selector不提供方向监督。故第三项硬门失败，裁定 **STOP，4.6/10**。
