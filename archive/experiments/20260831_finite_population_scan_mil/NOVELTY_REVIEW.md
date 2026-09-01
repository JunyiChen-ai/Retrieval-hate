# Finite-population scan MIL：独立 novelty / identifiability review

截至 2026-08-31。审查对象是本目录 `README.md` 中尚未实现的候选；本轮没有实现、训练或推理。

## 结论

**Verdict：STOP。Novelty：3.4/10。不要进入正式 pilot。**

三项硬门：

| 硬门 | 结论 | 理由 |
|---|---|---|
| 允许跨任务 adaptation | PASS | epidemic change / multiscale scan 可以适配到 weak temporal localization。 |
| 来源核心尚未用于 hateful video detection/localization | PASS，检索范围内未发现直接先例 | 检索到的 hateful-video 方法没有使用所写的 interval-versus-full-complement standardized scan MIL。 |
| adaptation 必须 non-trivial、任务特定且可证伪 | **FAIL** | 所写统计量严格允许 single spike 与 `T-1` 区间达到同一个全局上界；hard selection 时 video variance 是对所有候选相同的乘数，完全不改变 interval ranking。实际机制只剩“连续区间、长度加权的 top-instance selector”，而 interval/context contrast 与 proposal MIL 已被 AutoLoc、P-MIL 等 WTAL 工作覆盖。 |

STOP 的主要原因不是“找到了一篇公式逐字相同的 hateful-video 论文”，而是候选已经在数学上触发 README 自己的停止条件：**弱标签可以用单点 spike 或极小 complement 无代价达到最大 scan score；所谓 variance normalization 在 hard selector 中又不参与候选排序。**

## Primary-literature occupation search

检索组合覆盖 `WTAL / WS-VAD + interval complement / inside outside / variance normalized / scan statistic / change point / proposal MIL`，以及 `hateful video localization + scan / CUSUM / change point`。只用论文、会议开放页面、期刊页面和官方项目页作裁定依据。

### 直接目标领域

- [MultiHateLoc](https://arxiv.org/abs/2512.10408) 使用 modality-aware temporal encoders、dynamic fusion、cross-modal contrastive alignment 与 modality-aware MIL，没有 interval-complement scan。
- [LELA](https://arxiv.org/abs/2602.09637) 是 training-free LLM-based hate localization，没有该训练算子。
- [HateClipSeg](https://arxiv.org/abs/2508.01712) 提供 segment annotations 和 temporal localization benchmark，不是该 weak scan MIL。

本次检索未发现 scan-statistic 核心已用于 hateful video detection/localization。因此第二门本身通过。

### WTAL 中的最近邻

1. [AutoLoc, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Zheng_Shou_AutoLoc_Weakly-supervised_Temporal_ECCV_2018_paper.html) 已用 Outer-Inner-Contrastive loss：最小化 proposal 外部平均 activation 与内部平均 activation 的差，直接从 video labels 学 temporal boundary。它的 outer 是邻近扩展区而不是整段 complement，也没有本候选的 global variance denominator；但“interval 相对 context 的 mean contrast 作为弱 proposal supervision”已经被占用。

2. [Proposal-Based MIL, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.pdf) 在训练和测试直接分类 proposals，并含 Surrounding Contrastive Feature Extraction、proposal completeness 与 proposal ranking。它没有所写的 full-complement standardized statistic，但已经占用 proposal-level MIL 与 surrounding contrast 的任务框架。

3. [PseudoFormer, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Bridge_the_Gap_From_Weak_to_Full_Supervision_for_Temporal_CVPR_2025_paper.pdf) 仍把 OIC score 用于 weak proposal/pseudo-label pipeline，说明 inside-versus-outside scoring 不是历史上孤立的组件。

4. [AHLM, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Weakly-Supervised_Action_Localization_by_Hierarchically-Structured_Latent_Attention_Modeling_ICCV_2023_paper.pdf) 已把 unsupervised change-point detection 引入 weakly supervised action localization。它的 generative change-point model不等于本公式，但“从 change-point detection 适配到 WTAL”这一宽 claim 也已被占用。

因此没有找到数学完全相同的 `interval vs entire complement + global variance denominator`，但可主张差异只剩 full complement 与特定 self-normalization，不能 claim 首次 contrastive interval selection、首次 proposal MIL 或首次 change-point adaptation。

### WS-VAD 中的最近邻

- [RTFM, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.html) 用 top-k feature magnitude 选择 positive/negative snippets 再训练 classifier，是候选必须超过的 selector-only anti-pattern。
- [Event-driven weakly supervised VAD, 2024](https://doi.org/10.1016/j.imavis.2024.105169) 使用多尺度 sliding-window event proposals 与 MIL event scoring；没有 full-complement standardized contrast，但说明 contiguous multiscale event proposals 已进入 WS-VAD。
- [Prompt-Enhanced MIL, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html) 继续在 frame-level weak localization 中使用 MIL 与 temporal context，但没有该 scan statistic。

结论：**未找到 exact WTAL/WS-VAD 公式占位；然而 hard-selector 版本的实际可表达机制只是已有 MIL selector 家族中的 contiguous standardized selector。**

### 统计来源

[Yao, *Tests for change-points with epidemic alternatives*, Biometrika 1993](https://academic.oup.com/biomet/article-abstract/80/1/179/228230) 已研究正态序列未知位置、未知持续时间的 epidemic mean change。[Sharpnack, *Learning Patterns for Detection with Multiscale Scan Statistics*, COLT 2018](https://proceedings.mlr.press/v75/sharpnack18a.html) 讨论跨位置与尺度取最大值的 multiscale scan 以及 scale correction。故 standardized interval scan 本身是来源方法，不是候选原创。

更重要的是，经典 scan 的检测意义依赖明确 null、噪声模型、scale-dependent threshold/penalty 或独立/稳健的 null variance estimate。候选把正在被网络优化、且包含候选 signal 本身的整段 logit variance 当 denominator，却没有统计显著性阈值；因此它只是一个 self-normalized selector，不能继承 scan test 的 false-positive 或 multiscale optimality 解释。

## 严格数学分析

令 `n=|I|`、`m=T-n`，并先令 `eps=0`。设整段使用 biased population variance

`v = (1/T) sum_t (e_t - mean(e))^2`。

候选统计量为

`Z(I) = (mean_I(e) - mean_C(e)) * sqrt(nm / (T v))`。

### 1. 它就是 point-biserial correlation

令 `q_t=1[t in I]`。直接展开 covariance 可得

`Corr(e, q) = (mean_I(e) - mean_C(e)) * sqrt(nm) / (T sqrt(v))`，

所以

`Z(I) = sqrt(T) * Corr(e, 1_I)`。

于是 `|Z(I)| <= sqrt(T)`。若使用 unbiased sample variance，上界只改为 `sqrt(T-1)`，下述退化不变。

这揭示了候选不是在估计一个独立的“区间异常强度”，而是在找与当前 logit 序列最相关的 binary interval template。

### 2. single spike 严格达到全局上界

取一个 frame 为 `A>0`、其余为零，并令 `I` 就是该单帧。此时

`delta=A`，`v=A^2(T-1)/T^2`，

代入得到

`Z(I)=sqrt(T)`。

这已经是理论最大值，与 `A` 的大小无关。也就是说 positive bag 只制造一个任意 frame spike，就能得到最强 scan witness。negative-bag loss只压 negative videos，不会禁止 positive video 使用这条捷径。

这与候选声称避免普通 top-k 的动机相反：允许 `|I|=1` 时，最坏情形正是 top-1。

### 3. tiny complement / `T-1` 区间也严格达到上界

令 `I` 内全部为 `A`，唯一 complement frame 为零。相同计算得到

`Z(I)=sqrt(T)`。

所以排除 whole-video proposal 没有排除 whole-video shortcut，只把它变成“整段高分、任选一个 frame 低分”。对 prefix/suffix interval，该极小 complement 仍是合法连续 proposal。README 要求“不能继续集中于最长允许区间”，但公式本身不给任何严格阻力。

更一般地，只要 logits 在 interval 内是一个常数、complement 内是另一个常数，任意长度都达到同一个上界。长度标准化没有偏好真实 span，而是奖励任意二平台分割。

### 4. affine invariance 不是无条件优点

- 平移 `e'=e+b`：`delta` 不变、variance 不变，严格 invariant。
- 正缩放 `e'=a e, a>0`：`eps=0` 时 `Z` 严格不变；负缩放只翻转符号。
- 固定 `eps>0` 时，scale invariance 被破坏：网络可以放大 logits 使 `eps` 变得可忽略，并让 `Z` 接近饱和上界。

scan selector 因此不能校准 absolute hate direction 或 margin。README 另加 raw positive MIL 与 negative loss 来定方向，这也说明 scan 本身只是选择规则，而不是 bag likelihood。

### 5. global variance 对 hard selector 完全无效

对同一个视频，`v+eps` 与 interval `I` 无关。因此

`argmax_I Z(I) = argmax_I delta(I) / sqrt(1/n + 1/m)`。

全局 variance denominator只是所有候选共享的正乘数。若候选是 hard select，它不会改变 interval identity，也不会提供所谓 variance-normalized proposal comparison。当前 controls 缺少真正数学等价的 `length_standardized_mean_difference`；`mean_difference_only` 删除了长度项，不能隔离 variance 的贡献。

若用 `softmax(Z)` 加权 proposals，global variance只成为每个视频一个可学习的 temperature：增大 variance会让权重变平，减小 variance会让权重变尖。它仍不改变候选排序，而且网络能通过 logit distribution 操纵 selector entropy。

### 6. variance inflation 的真实作用

简单增加 interval 内或 complement 内的零均值噪声会增加 denominator 而不增加 `delta`，所以不能提高某个 `Z`；它会降低所有候选的绝对 scan score。相反，objective鼓励压低 within-group variance、形成二平台 logits，并通过增大 between-group contrast接近饱和上界。

因此“variance inflation 无代价过门”不是最准确的 blocker；更严重的 blocker 是：

- hard selector 中 variance根本不参与排名；
- soft selector 中它只是可操纵 temperature；
- total variance含 signal，使 effect self-normalize并饱和；
- 没有独立 null scale，不能给统计显著性解释。

### 7. 高正例率与极小 complement

当真实 positive fraction接近 1 时，complement mean由极少数 frame 决定，具有高方差且容易被边界、padding、片头片尾或一个低分点控制。公式中的 `1/m` 表面上惩罚小 complement，但二平台构造时 numerator 与 total variance同步变化，惩罚完全抵消，仍达到上界。

此外 complement 未必 benign。高正例率视频中它可能仍含 hate；强迫 interval 高于 complement只创造相对排序，不保证任何 frame 的绝对标签。raw MIL虽然提供方向，却无法判断被压低的 complement 是否真为 negative。

## Training-only selector 是否超越 top-k

**没有形成足够的机制差异。** 它相对普通 top-k 增加了两个结构偏置：候选必须连续，长度由 `sqrt(nm/T)` 权重调整。但它没有引入新的监督、独立 background reference 或可识别 span constraint。

若 hard argmax 后只对所选 interval 的 raw logits做正 bag MIL，scan 的梯度不能穿过离散 selector；训练就是反复选择一个当前最优的 contiguous subset，再把其 raw score推高。这是 structured hard-instance mining。single-frame interval退化为 top-1，`T-1` interval退化为近全视频 label broadcast。

若做 soft weighting，必须明确 proposal set、temperature 与 bag score；但这会变成 differentiable proposal attention，且 global variance只是 attention temperature。AutoLoc/P-MIL 已有 interval/context scoring，普通 attention MIL 已有 soft instance weighting。仅把权重写成 correlation scan 不能通过 non-trivial adaptation 门。

最后 test 只读 raw `e_t`，scan 不参与 readout。于是 scan 是否真正改变最终 ranking完全依赖训练路径，而候选没有提供能排除“selector换了但 same raw ranking”的额外识别条件。项目 deletion-carrier 方向已经出现 auxiliary mechanism几乎不进入最终 frame ranking的失败；本候选风险相同。

## Controls 中的缺口

即使忽略 STOP，原 controls 也不能正确归因：

1. 必须加入 `length_standardized_mean_difference`，其 selector与 core hard argmax严格相同；没有它就不能声称 global variance normalization有效。
2. 必须分别固定 `|I|=1`、`|I|=T-1`，证明这两个解析退化不会匹配 core；当前笼统 length penalty不足以隔离两端。
3. complement 循环错位不是干净 control：它同时破坏 contiguity relation、边界结构与 interval statistics。即使性能下降，也不能证明 full complement 是 benign reference。
4. 同 checkpoint 的 scan-disabled inference没有意义，因为 test本来就不用 scan；应比较训练后的 raw rankings与 matched selector arms。但这只能测训练影响，不能使 latent span可识别。
5. 必须监控 selected length histogram、single-frame rate、`T-1` rate、piecewise-constant saturation、`Z/sqrt(T)` 分布及梯度实际落到多少 frames。这里只是诊断，不能修复公式。

## 最小 premise 能否挽救

当前核心没有可批准的“冻结最小 premise”，因为 single spike 与 tiny complement是解析反例，不需要数据即可证伪。以下修改会形成另一个候选，而不是本候选的小修：

- 排除靠近 1 和 `T-1` 的长度并加入预设 occupancy bounds；这重新引入候选声称避免的事件比例假设。
- 用从 negative train videos 独立估计的 robust null scale；这会转成 normal-reference scan，而不是 finite-video self-normalization。
- 使用 multiscale null critical values/penalties控制不同 interval lengths；这会转成真正 statistical scan likelihood，但仍需解释时间相关性与 hate semantics。
- 给 complement latent contamination model，而不是默认 relative background；这会新增不可识别结构。

这些都不能作为实现阶段临时 patch。若未来提出新候选，应重新做 novelty review。

## 最终裁定

文献层面没有发现 exact formula 已进入 hateful video localization，WTAL/WS-VAD 中也未发现逐字相同的 full-complement global-variance scan MIL。但 AutoLoc、P-MIL、PseudoFormer 已占 interval/context proposal supervision，AHLM 已占 change-point-to-WTAL adaptation，RTFM 与 event-level VAD 已占 structured instance/proposal selection。

数学层面更为决定性：`Z=sqrt(T) Corr(e,1_I)`，single spike、`T-1` interval及任意二平台分割都达到上界；hard selection 时 global variance不改变 interval ranking。候选因此既不能排除已观察到的 whole-video shortcut，也不能排除 top-1 shortcut，且所谓 variance-normalized机制在 selector中不可归因。

**STOP，3.4/10。不要实现或运行。**
