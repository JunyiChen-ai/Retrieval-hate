# Bag-Constrained Dawid–Skene Temporal Occupancy：独立查新与机制审查

截至 2026-08-31。审查对象：本目录 `README.md`。本轮未实现、未训练、未生成 prediction。

## 裁定

**STOP**  
**novelty：5.8 / 10**

截至本次检索，没有发现 occupancy–detection、Dawid–Skene（DS）或等价的 modality-confusion latent model
已用于 hateful-video detection/localization；按当前标准，外部 source 可以迁移，这一点本身通过。

停止原因不是目标任务已被直接占用，而是当前 adaptation 的核心不能支持它声称的机制：

1. continuous weak-source generative aggregation、MIL 中“negative instances certified + positive bag 至少一个
   positive”、constrained pseudo-label self-training，以及 teacher-to-student 已分别有直接先例；
2. exact positive-bag OR conditioning **不会改变同一视频内的 posterior 排序**，只能给所有秒乘同一个
   video-level scale；
3. 三个 reports 是用同一 bag label 训练出的模型输出，不是独立 repeated detections。OOF 只避免本视频标签被
   producer 见过，不能建立 conditional independence；
4. `pi_t` 又由产生 reports 的同源 fused features 学习，使 occupancy prior 与 positive emission 可以互相解释
   observations。当前约束不足以识别“模态漏检率”或 temporal occupancy。

因此它目前更接近 **continuous label model + standard MIL constraint + pseudo-label student 的组件组合**，而不是
一个已被识别、能直接纠正 time×modality ownership 的新机制。即使 test 指标改善，也无法由现有 controls 排除
普通 score stacking、per-video target reweighting 或 self-training regularization。

## 直接来源与适用边界

### Occupancy–detection

MacKenzie et al. 从 repeated surveys 的 detection histories 同时估计 site occupancy 与小于 1 的 detection
probability。关键数据结构是：同一 site 在 closure period 下有多次实际 survey，每次有 observed detection / 
nondetection：

- MacKenzie et al., *Estimating Site Occupancy Rates When Detection Probabilities Are Less Than One*, Ecology
  2002，[作者公开 PDF](https://www.sfu.ca/~lmgonigl/materials-qm/papers/mackenzie-2002-2248.pdf)，
  [DOI](https://doi.org/10.1890/0012-9658(2002)083%5B2248:ESORWD%5D2.0.CO;2)。

本候选的 visual/audio/text reports 是同步的 learned predictions，不是重复开展且结果独立的 survey。生态学
source 可以提供建模动机，但不能据此把估计出的 Beta parameters 称为真实 detection probabilities。

### Dawid–Skene 与 latent-class identifiability

Dawid–Skene 以多个 observer 对同一 item 的 observed categorical responses 和 observer-specific error matrices
推断 latent response：

- Dawid and Skene, *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm*, JRSS-C 1979，
  [DOI](https://doi.org/10.2307/2346806)。

latent product models 的可识别性依赖足够多、在 latent class 条件下独立且具有区分性的 observed variables；
这不是“有三个输出”就自动满足：

- Allman, Matias and Rhodes, *Identifiability of Parameters in Latent Structure Models with Many Observed
  Variables*, Annals of Statistics 2009，[arXiv:0809.5032](https://arxiv.org/abs/0809.5032)，
  [DOI](https://doi.org/10.1214/09-AOS689)。

本候选把模型 predictions 类比 annotators 是允许的跨任务 adaptation，但不能继承 DS 的 observer-error 语义或
identifiability 结论。

## 最近邻占位

### Continuous weak-source aggregation：核心大半已有

CAGE 已把 discrete labeling functions 扩展到 `(0,1)` continuous scores，并用 generative label model 汇总为
latent labels；论文还专门讨论无标注 likelihood 对初始化/训练设置不稳定，需要 quality guides 稳定：

- Chatterjee, Ramakrishnan and Sarawagi, *Data Programming Using Continuous and Quality-Guided Labeling
  Functions*, AAAI 2020，[论文页](https://ojs.aaai.org/index.php/AAAI/article/view/5742)。

Data Programming 也已明确允许在 label model 中表示 weak-source dependencies，而不是默认独立 votes：

- Ratner et al., *Data Programming: Creating Large Training Sets, Quickly*, NeurIPS 2016 / author manuscript，
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5985238/)。

差异：本候选的 reports 来自 OOF modality MIL models，另加 video OR constraint 和 certified-negative instances；
这是一项具体组合，但“continuous reports + source-specific reliability + latent soft labels”不是新 core。

### MIL / WSVAD / WTAL：constraint-to-pseudo-label student 已拥挤

- MIST 已在 WSVAD 中由 video-level labels 生成 clip pseudo labels并 self-train 最终 feature encoder：
  [CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/html/Feng_MIST_Multiple_Instance_Self-Training_Framework_for_Video_Anomaly_Detection_CVPR_2021_paper.html)。
- Ma et al. 已把 MIL 直接重写为 instance-level semi-supervised self-training：negative-bag instances 是已知
  negatives，positive-bag instances 未知；再用来自 positive bag 的 global/local constraints 防止全负退化，
  训练 instance classifier：
  [arXiv:2408.04813](https://arxiv.org/abs/2408.04813)。
- WS-TAL 已普遍用 multimodal branches、uncertainty-weighted pseudo labels 和 direct snippet supervision；例如
  M2PT 的 RGB/flow cross-modal modeling 与 uncertainty-weighted pseudo-label loss：
  [CVPRW 2024 paper](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/html/Hu_Weakly-Supervised_Temporal_Action_Localization_with_Multi-Modal_Plateau_Transformers_CVPRW_2024_paper.html)。
- WSVAD 中针对 pseudo-label completeness / uncertainty 再训练 frame model也已有：
  [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Exploiting_Completeness_and_Uncertainty_of_Pseudo_Labels_for_Weakly-Supervised_CVPR_2023_paper.html)。

差异：这些工作未见 per-modality Beta emission 的 DS/occupancy interpretation，也未见本候选的 exact OR posterior；
但 negative certification、positive-bag constraint、pseudo-label EM/self-training 和 direct student 都不能作为新意。

### Multimodal reliability / noisy modality

动态 modality weighting、uncertainty fusion、noisy-or fusion与模态退化鲁棒性已有大量工作。一个尤其接近的
概率先例是 UNO：估计各 modality uncertainty，再做 probabilistic noisy-or fusion：

- Prakash et al., *UNO: Uncertainty-aware Noisy-Or Multimodal Fusion for Unanticipated Input Degradation*,
  [arXiv:1911.05611](https://arxiv.org/abs/1911.05611)。

它不是 temporal MIL，也不估 DS confusion emission；但已经占用“modality-specific uncertainty + noisy OR
probabilistic fusion”的宽 claim。

### Hateful video

检查 MultiHateLoc、LELA、TANDEM、CLARA、MM-HSD 与 HateClipSeg，未发现 occupancy–detection / DS、continuous
weak-source label model或 per-modality Beta confusion posterior用于 hateful-video temporal localization。代表性
直接邻居：

- MultiHateLoc，[arXiv:2512.10408](https://arxiv.org/abs/2512.10408)；
- LELA，[arXiv:2602.09637](https://arxiv.org/abs/2602.09637)；
- TANDEM，[arXiv:2601.11178](https://arxiv.org/abs/2601.11178)；
- CLARA，[arXiv:2608.15905](https://arxiv.org/abs/2608.15905)；
- MM-HSD，[DOI](https://doi.org/10.1145/3746027.3754558)；
- HateClipSeg，[arXiv:2508.01712](https://arxiv.org/abs/2508.01712)。

所以 STOP 不是因为 hateful-video 直接查重失败。

## 机制审查

### 1. Exact OR conditioning 对 within-video 排序没有作用

设加入三个 emissions 后、尚未看 bag label 的逐秒独立 posterior 为
`s_t = P(z_t=1 | r_t, x_t)`，positive bag 事件为 `A = OR_t z_t=1`。则：

`P(z_t=1 | r, A) = s_t / (1 - product_j(1-s_j))`。

分母对同一视频的所有 `t` 完全相同。因此：

- OR-conditioned `q_t` 与 unconditioned `s_t` 的帧内排序和 within-video ROC 完全相同；
- 它只改变各视频的 target scale / positive mass，可能影响 pooled 指标及 student 的跨视频优化权重；
- full 与 “去 OR” student 若最终 within 不同，差异来自 student 对 video-level rescaling 的训练响应，不是 OR
  posterior 找到了不同 witness。

当前 README 把 OR conditioning列为核心 temporal allocation机制并要求 full 在 within 上胜出，机制归因不成立。
必须增加一个 **per-video scale-matched unconditioned target** control；若它追平 full，则所谓 OR 增益只是 weighting。

### 2. OOF 不产生 independent annotators

OOF 是正确的数据卫生措施：held video 的 report producer 没见过该 video label。但三个 modality branches：

- 仍由同一组 video labels、同一 corpus bias和同一事件时间训练；
- reports 同时由同一视频内容产生，hate 的共同原因、场景和剪辑会造成 residual dependence；
- branch error 可因 video-level shortcut 同步，而不是像 repeated surveys 那样独立漏检。

product-of-Betas 会把相关 agreement 当成多份独立证据，造成过窄 posterior。尤其现有 DMS 几乎恒选 visual 与
HCS near-constant branch 证据已提示 branches 不具 DS 式互补性。OOF 不能修复这一点。

### 3. `pi_t` 与 positive emission 未按当前规格识别

negative bags 能直接估计 `f_m0(r)`，这是模型最可靠的部分。但 positive bags只给一个 OR bit。与此同时：

- `f_m1` 的 Beta shapes没有 positive frame anchor；
- `pi_t` 是同源 OOF fused feature 的 learned function，而 reports 也是这些 features 的 learned functions；
- 于是同一 score pattern可以被解释为高 occupancy + 弱 positive emission，或低 occupancy + 尖锐 positive
  emission；linear head限制容量，但没有消除这种 trade-off；
- 把 `x` 同时放进 `P(z|x)`，再把由 `x` 确定的 reports 当作 `P(r|z)` observations，构成 information double-use，
  不能直接套用 latent product-mixture 的识别结论。

`mean(Beta_1) > mean(Beta_0)`只解决 label ordering，不证明区分性，也不等于 sensitivity > .5；两个 Beta densities
可以交叉多次。对 continuous reports 使用“sensitivity 退化到 .5”也没有定义，除非预先固定 threshold，而 README
明确不阈值化。

因此目前最多能称估计一个 regularized score-fusion latent decomposition，不能称已估计 modality sensitivity、
specificity、false-negative rate或真实 occupancy。

### 4. Direct student 是合理输出约束，但不是新机制

只输出一个 student frame probability、禁止 inference ensemble/routing 是合规且清楚的工程选择。可是从 teacher
soft targets 训练 direct student 是标准 pseudo-label/self-training结构。它能确保 posterior 进入最终排序，但不能
反向证明 teacher posterior 的概率语义或 modality ownership。

## 当前 controls/gates 的问题

1. **rate-preserving modality permutation control 无效/含糊。** 若“固定打乱 modality identity”指对所有样本
   使用同一个 permutation，模型只会同步重命名三组 Beta parameters，理论上应与 core 等价；它不是 negative
   control。若指每个 `(video,time)` 独立随机 permutation，必须明确并固定随机规则，才会破坏 modality identity。
2. shared-confusion 只检验 modality-specific parameter capacity，不能检验 conditional dependence或概率语义。
3. simple mean control 没有匹配 Beta label model的非线性容量；一个 train-only logistic stacker / small MLP
   stacker可能用更少假设实现相同 report fusion。
4. “去 OR”没有匹配 OR 带来的 per-video target scale，无法归因。
5. Spearman `<.95`只说明 targets 改了，不说明改得对；一个任意单调性破坏也能过门。
6. emission direction / finite / entropy不构成 identifiability test；EM 多起点可到不同参数但相似 likelihood。
7. 没有测试同一 latent class 下的 branch residual dependence，也没有 correction 与 correlated-report control。
8. 固定一次 3-fold split可能把 fold producer calibration差异误当 modality confusion；需要 cross-fit partition
   stability，不能只看 seed 234 的 downstream指标。

## 若要重提，必须先修改

这不是建议直接实现当前模型。只有先完成下列 train-only / synthetic falsification，再能提交一个新候选：

1. **承认 OR rank invariance。** 把 OR 从“定位分配机制”降为 bag-consistency / per-video mass constraint，并加入
   scale-matched unconditioned target control。若研究目标仍是修正 within ranking，必须提出另一个真正改变相对
   `q_t` 的 task mechanism。
2. **移除同源 learned occupancy prior。** 首轮 identifiability probe 固定一个跨语料一致的 `pi`，或让 prior 只用
   与 reports 分离且预注册的 covariates；不能让 prior head读取产生 reports 的同源 fused features。
3. **synthetic parameter recovery。** 在已知 Beta emissions、prevalence和 bag lengths 下验证恢复，再逐级加入
   measured report correlations；报告多初值的 parameter/posterior spread。只恢复 likelihood不算通过。
4. **train-only dependence audit。** negative train seconds有 certified `z=0`，可直接计算三 branch 在条件于
   fold/video covariates后的 residual dependence。若强相关，product-Beta core前提失败；不能依靠 test 指标掩盖。
5. **capacity controls。** 至少加入 mean、train-only logistic stacking、同参数量 nonlinear stacking、CAGE式
   continuous label model、standard constrained MIL self-training，以及 per-video-scale-matched no-OR。
6. **正确 permutation。** modality identity要对每个 train item独立置换，同时保持各 report marginal与 time
   autocorrelation的预注册方案；全局固定 permutation只能作为 equivariance单元测试。
7. **连续 emission gate。** 用 train-only held-fold log likelihood、likelihood-ratio separation、posterior
   stability 和 Beta density diagnostics；删除未定义的 `.5 sensitivity` gate。
8. **student attribution。** 同时评估 frozen teacher posterior与 student；若 teacher没有改善排序而 student改善，
   结论只能归于 self-training，不归于 occupancy posterior。
9. 所有方法仍须四主语料独立训练，validation只选 checkpoint，训练后立即 test三指标；test error analysis可
   inform development，但结果标 developmental。禁止跨主数据集训练、ensemble、calibration或routing。

这些修改中第 1、2 项会改变当前 mechanism story。若没有新的、能改变 within-video relative evidence 的约束，
不建议仅把 controls 补齐后重新实现同一候选。

## 允许与禁止的 claim

当前 STOP 状态不允许作为新方法 claim。若以后解决识别问题且严格 controls 通过，最多可写：

> To our knowledge, we adapt a continuous weak-source label model to weakly supervised hateful-video temporal
> localization, using out-of-fold modality reports, certified negative-bag instances, and a video-level MIL
> constraint to generate soft targets for a single frame student.

必须称 `weak-source reports` 或 `model reports`，不能无条件称 annotators、independent detections或 repeated
surveys。除非有额外识别证据，不得 claim 学到了真实 modality sensitivity/specificity、causal modality
ownership、calibrated occupancy probability或 independent evidence。也不得 claim 首次 EM pseudo labels、首次
multi-view MIL、首次 modality reliability、首次 positive-bag constraint或首次 teacher-to-student localization。

## 最终理由

“source 未用于 hate localization”满足最新 novelty 标准的前两层，但第三层要求 non-trivial 且有成立的任务机制。
当前新组合里，唯一特殊的 exact OR conditioning对 within ranking是严格不变的；其余排序变化来自已有的
continuous label aggregation，而该 aggregation又建立在明显不满足、也未被 controls识别的 independent-report
假设上。故诚实裁定为 **STOP，5.8/10**。
