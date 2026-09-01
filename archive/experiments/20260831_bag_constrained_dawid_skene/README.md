# 已淘汰：Bag-Constrained Dawid–Skene Temporal Occupancy

淘汰原因：独立novelty/identifiability review裁定`STOP 5.8/10`。exact positive-bag OR conditioning只把同一
视频所有unconstrained posterior乘共同归一因子，无法改变within-video排序；OOF branches也不建立Dawid–Skene
所需的conditional independence，learned prior与positive Beta emission可互相补偿。未实现、未训练、未生成
新prediction。

截至 2026-08-31。状态：**独立review `STOP 5.8/10`，已在实现前淘汰。**

## 研究问题与test-informed依据

目标仍是四个主语料各自独立训练的弱监督hateful video temporal localization，最终同一frame score必须在
pooled AP、pooled ROC和within-video ROC三项都超过固定SOTA。

当前两条直接证据：

1. MultiHateLoc test诊断中，DMS最高权重与test-GT最佳单模态匹配率在HMM/EN/ZH/HCS仅
   `.216/.333/.375/.323`，且selector几乎总选visual；best-branch oracle相对fused within仍有
   `.106/.171/.211/.106`缺口。说明“哪个模态可靠”不是全视频固定，也不能由当前gate正确判断。
2. deletion-carrier ItS2CLR的modality abstention只作为auxiliary SupCon relation，core-vs-broadcast frame
   Spearman在HMM/HCS高达`.97568/.99723`，没有进入最终排序。后续机制必须直接定义frame posterior。

所用developmental test artifacts：

- `runs/20260831_multihateloc_test_error_analysis/main/metrics.json`
- `runs/20260831_owner_abstaining_its2clr/pilot_seed234/test_error_analysis.json`

test GT只inform本轮机制选择，不参与gradient或checkpoint selection。

## 跨任务来源

来源一是生态学occupancy–detection model：MacKenzie et al. (Ecology 2002)把“物种真实占据site”与“重复survey
是否检测到”分开，避免把imperfect detection造成的false absence当成真实absence。来源二是crowdsourcing的
Dawid–Skene latent-class model：每个annotator有自己的sensitivity/specificity，通过EM从多个noisy reports
推断latent true label。

本候选把每个1fps秒视为site、visual/audio/text三个独立OOF branch视为不完美survey/annotator、latent
`z_t`视为该秒是否hateful。negative train video提供`z_t=0`的certified seconds；positive video只提供
`OR_t z_t=1`，不把video label广播给全部秒或全部模态。

正式查新必须确认occupancy–detection、Dawid–Skene或等价的annotator-confusion latent model没有用于
hateful video detection/localization，并检查它是否已等价进入multimodal MIL、WS-TAL、WSVAD或modality-noise
learning。只要核心已被占用，直接停止。

## 单一核心机制：bag-constrained imperfect-detection posterior

### 1. OOF modality reports

只在目标语料train split做固定三折cross-fitting。每个fold producer使用另外两折video labels训练三个
modality-specific MIL branch，再对held fold生成每秒连续report `r_tm in (0,1)`；任何train video的report都
不能来自见过该video label的producer。最终模型训练前不读取validation/test。

不阈值化report。每个modality学习两个Beta emission：

`r_tm | z_t=0 ~ Beta(alpha_m0,beta_m0)`，

`r_tm | z_t=1 ~ Beta(alpha_m1,beta_m1)`。

negative train seconds固定属于`z=0`并识别false-positive emission；positive bags中的`z`未知。

### 2. Bag-constrained E-step

给定固定occupancy prior `pi_t`与三个emission likelihood，先算unconstrained posterior odds；negative bag强制
所有`q_t=0`，positive bag则对独立Bernoulli posterior精确条件化在`OR_t z_t=1`。因此某一modality低report只按
其learned false-negative率提供有限反证，不会自动把该秒变成background；多个可靠modality的一致report会提高
`q_t`。

为避免模型仅靠视频位置先验，`pi_t`只由OOF fused feature的共享linear head产生，不输入归一化时间、视频ID、
长度或语料特定规则。E/M固定迭代次数，不由validation/test选择。Beta均加固定弱先验并强制每个modality
positive-emission mean高于negative-emission mean；若约束只能靠label swapping满足或任一sensitivity退化到
`.5`附近，mechanism premise失败。

### 3. Direct student

最终单一temporal student在目标语料train上以`q_t`作为soft frame target训练，同时保留原bag-label MIL loss。
inference只输出该student的一个frame probability；不保留三个teacher branch、不做ensemble、test-time
routing或post-hoc calibration。与ItS2CLR的关键差异是latent posterior直接监督最终frame scorer，而不是只改变
projection-space positives。

## 为什么可能是non-trivial adaptation

经典Dawid–Skene有同一item的observed annotator labels；经典occupancy model有observed repeated detection
history和site-level covariates。本任务只有learned continuous modality reports与video-levelOR constraint。
adaptation需要同时解决三件事：train-only OOF reports避免自标注；negative bag识别modality-specific false
positive emission；positive bag通过exact OR-conditioned posterior区分latent occupancy与imperfect detection。
最终posterior再作为单student的直接temporal supervision。它不是简单modality weighting，也不是把低置信度
改成abstain。

但可识别性风险很高：三个branch由同一个video label训练，reports可能条件相关且没有真正独立的重复survey；
HCS branch可能近常数；positive-emission Beta与occupancy prior也可能互相补偿。若独立review判断这些因素使
mechanism不可证伪或只是普通EM pseudo-label MIL，应在实现前停止。

## 最小pilot与硬门

只在HateMM/HateClipSeg做seed 234最小pilot，语料完全独立。formal arms：

1. 当前MultiHateLoc anchor；
2. capacity-matched direct self-training：同一OOF reports但简单三模态均值生成soft target，不估confusion；
3. Dawid–Skene但去掉positive-bag OR conditioning；
4. full bag-constrained imperfect-detection core；
5. rate-preserving modality permutation control：在同video/time内固定打乱modality identity再估confusion；
6. shared-confusion control：三个modality强制同一emission，检验增益是否来自modality-specific detection。

所有arm训练定义先冻结，validation只在arm内部选checkpoint，之后立即test三项指标。core机制门：

- HMM/HCS within都高于simple self-training和unconditioned control，至少一边`>=+.020`；
- OOF train posterior相对simple mean必须实质改变排序（per-video Spearman均值`<.95`），避免再出现auxiliary
  relation没有进入输出；
- modality permutation与shared-confusion均不能追平core；
- 三个emission在两语料都满足预注册的方向、finite与非退化门，并报告posterior entropy/positive mass。

performance gate仍是两语料各自pooled AP、pooled ROC、within ROC全部严格超过固定SOTA；失败不扩MHC，
不按语料删modality、不调Beta family/iteration/OR temperature、不把teacher posterior作为test-time output。

## 当前claim上限

即使全部成立，也只能claim：把imperfect-detection/crowd-aggregation latent model改造成带video-level OR
constraint和OOF continuous modality reports的弱监督temporal posterior，并用它直接训练单一hateful-video
localizer。不能claim首次modality reliability、首次EM pseudo labels、首次multi-view MIL、真实/因果modality
ownership或conditional independence成立。
