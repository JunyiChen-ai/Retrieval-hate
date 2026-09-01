# 已淘汰：Evidence-Program Graph CTC candidate

> 淘汰原因：两份独立 novelty/identifiability review 均为 STOP（`3.6/10`、`5.5/10`）。CTC/ECTC/graph-WFST 来源未检出进入 hateful-video task，第二门窄 PASS；但无 timestamp 的 video-specific program 没有提供新的时间观测，global-video broadcast + position-only alignment可同时满足 graph likelihood、partial order 与 inverse recovery而 within仍为`.5`，第三门失败。未运行 generator premise、未实现、未训练、未生成 prediction。

截至 2026-08-31。状态：双独立 review 后 `STOP_BEFORE_PREMISE`；未实现、未训练、未生成 prediction。

## 研究问题与当前证据

目标仍是四个主语料各自独立训练，只使用该语料 train video labels，最终在完整 test 上同时超过 pooled AP、pooled ROC、within-video macro ROC 三项固定 SOTA。

现有 developmental test evidence 表明四个异构局部信号存在稳定但不能直接当方法的共同 upper bound：

- `runs/20260831_universal_teacher_simplex_diagnostic/main/metrics.json` 找到 7 个 HMM/HCS 共同 all-SOTA tuple；
- `runs/20260831_teacher_scale_transfer_diagnostic/main/metrics.json` 中这 7 个 tuple 在 fivefold video-heldout ECDF 下仍 7/7 双语料 all-SOTA，但 per-video ECDF 与 raw identity 的共同通过数均为 0。

因此缺口不是再发明一个视频内 ranking loss，而是获得一种能跨视频共享语义尺度、同时把 video-specific hateful claim 对齐到具体时间的局部方向。普通 multi-teacher KD/knowledge amalgamation 已因目标任务占用和机制平凡而停止，本候选不使用那四条 score 作为训练 target。

## 跨任务来源

主要来源族是：

1. Graves et al. 的 Connectionist Temporal Classification：在只有有序符号序列、没有 frame alignment 时，通过 blank、重复折叠与 forward-backward 边缘化全部合法路径；
2. Huang et al. ECCV 2016 的 Extended CTC（ECTC）：把 CTC 适配到弱监督视频动作标注，用无 frame timestamp 的 action transcript 学习时序对齐；
3. differentiable dynamic time warping / weak action segmentation 中对 latent temporal alignment 的可微优化。

独立 review 必须检索 CTC、ECTC、graph/WFST-CTC、weak transcript alignment 或等价 latent automaton 是否已经用于 hateful-video detection/localization。只要真正 core 已进入目标任务，按 novelty 第二门停止。

## 非 trivial adaptation：video-specific typed evidence program → partial-order automaton

### 1. Label-blind evidence program producer

对每个 train/test video，用冻结的同一 multimodal generator读取原视频、ASR、OCR与稀疏帧，prompt **不提供 video label、不要求 timestamp、不要求直接输出 hateful/benign verdict**，只输出视频中实际出现的、可由观测支持的 proposition：

- accountable source；
- predicate/utterance/action；
- mentioned or depicted target；
- stance/context（endorsed、quoted、reported、condemned、ambiguous）；
- carrier（speech、screen text、visual action/depiction）。

每个 proposition必须引用短的 observed lexical/visual evidence string，但没有时间。Producer 在打开任何 GT 前固定，parse failure保留显式 null program；不得用 video label重试、过滤或改 prompt。这样 program 是冻结外部观测，不是把 train label广播成 pseudo span。

### 2. Hate-policy graph，而不是固定 sequence

把每个 video-specific proposition 与固定 policy rule编译成小型有向 automaton。状态不是 generic `hate/non-hate`，而是有 identity 的 `(source, predicate, target, stance, carrier)` evidence nodes：

- blank/self-loop允许无关 seconds；
- source、predicate、target可因视听异步按若干合法偏序到达，不强迫单一展示次序；
- accepting path必须完成一个 policy-valid relation；predicate-only、target-only、quoted/reported-without-endorsement走 non-accepting path；
- 多个 proposition形成 alternatives，不在 inference 选 branch或加权 ensemble，统一在一个 automaton partition中边缘化。

MultiHateLoc 的共享 frame embedding与每个 video-specific node text/role embedding产生 emission。Graph-CTC forward algorithm边缘化全部合法 time×state paths。Positive train bag最大化至少一个 accepting relation的概率；negative train bag压低全部 accepting relation，但仍允许观测到 target、hostile quotation等 non-accepting nodes。

### 3. 最终 frame score与训练目标同源

唯一 frame score是同一个 automaton posterior中该秒属于任一 accepting relation evidence state的边缘概率。禁止另接普通 fused MIL head、用 program/branch routing、NMS、test calibration或多模型平均。若完整 video-specific program不可用，模型必须通过同一个显式 null path给出可审计失败，而不是回退另一个方法。

为防止 node identity只是装饰，训练加入 inverse token recovery：每个 node的posterior-weighted frame embedding必须区分本 video node与同 role、同 target/topic的其他 train-video donor node。该项只能读取train programs；test不做 donor检索。若删除 recovery或置乱 node identity不改变最终 ranking，则 graph semantics不是 load-bearing，机制失败。

## 为什么它可能解决当前失败模式

二元 bag MIL允许 whole-video topic broadcast；固定 policy bank又允许相同 query在所有帧恒定激活。Video-specific evidence program提供每个视频不同的 observed claim identities，CTC blank竞争要求模型把这些 identities解释到一部分 frames，policy automaton再区分完整攻击关系与 target-only/quotation。跨视频共享的是同一个 emission geometry、role semantics与policy transition，不是每视频独立 rank；这与 scale-transfer diagnostic 显示的“需要可迁移 corpus-level geometry”一致。

这项机制可被明确否定：若 program与时间无关、global video feature足以重构全部 nodes、CTC任意对齐或 policy states只是固定 query别名，那么 time shuffle、cross-video program swap、fixed-policy和无 inverse-recovery controls会追平 core，候选必须停止。

## 初步 novelty/identifiability 硬门

两名独立 reviewer 必须分别回答：

1. 来源 CTC/ECTC/graph transcript alignment是否未被 hateful-video detection/localization占用；
2. 相比 LELA、TANDEM、CLARA、MultiHateLoc、POWA、项目旧 inverse-compositional/fixed-policy链，真正新增部分是否是完整的 `label-blind video-specific proposition → policy partial-order automaton → accepting-state frame posterior`，而非 VLM caption、固定 query或普通 grounding的简单拼接；
3. binary bag label、无 timestamp program与 automaton是否仍存在 whole-video broadcast/任意 alignment反例，当前 controls能否实际证伪；
4. test时需要 frozen program producer是否构成允许的统一系统，而不是 ensemble、routing或 calibration；
5. HCS silent/static/meme videos能否产生有身份的 OCR/visual proposition；若机制天然只覆盖speech，不得按语料 fallback。

任一 novelty 硬门失败就归档，不运行 generator premise或训练。

## 独立审查结论

两名 reviewer 均确认：CTC、ECTC 或 WFST graph objective 的精确 core 未检出用于 hateful-video detection/localization，允许 adaptation 与来源未占用两门可以窄通过；但本候选的 non-trivial/load-bearing 门失败。

决定性反例令每秒表示复制同一个 video-global feature `h_v`。模型可用 `h_v` 判断该视频与全部 video-specific nodes，再用固定 position code把 nodes放进最早合法 slots；positive提高 accepting emissions、negative压低即可优化 bag graph likelihood。CTC只对合法路径求和，不会产生真实 evidence time。此解同时满足 partial order、accept/reject、cross-video donor recovery和 inverse token recovery，但 accepting posterior除边界/path-count效应外近似时间常数，within ROC仍可为`.5`。

Video-specific token recovery还会奖励video identity：任意帧都包含global signature时，posterior-weighted frame可以恢复该视频program。相对 LELA/TANDEM/CLARA 与项目旧 source-scoped proposition graph/inverse-compositional链，VLM proposition、role/policy semantics和semantic grounding已是已知部分；新增 graph-CTC 只是把无时间戳 pseudo-program与标准 structured alignment拼接，没有新增可识别监督量。HCS silent/static meme还可能产生整段恒定OCR/visual proposition或被稀疏producer漏掉。

另外，正式test readout若condition在accepting paths上会等价使用positive-label oracle；改为accept/reject/null全partition的无条件posterior才合规，但不解决上述反例。裁定不运行12-video premise，不靠补control继续同一机制。

## 若 review 通过：最小 premise 与 pilot

先对 HateMM/HateClipSeg 各固定 12 个 test positive videos运行同一 label-blind producer，覆盖 speech、OCR、visual-only、static meme与parse failure；选择仅按预先固定的 carrier strata，不看 span GT。随后统一评估：

- program node与GT positive seconds的 frozen emission alignment；
- within-video program time-shuffle；
- same-role/topic-matched cross-video program swap；
- fixed global policy program；
- sequential CTC 对照与 partial-order graph CTC。

只有两语料 core frozen ordering都胜三种破坏性 controls至少 `.020` within ROC，且 program exact coverage/parse gate通过，才实现 train localizer。正式 seed-234 pilot必须各语料独立训练；validation只在固定 arm内部选 checkpoint，选定后立即在完整 test跑全部三指标。Core必须在 HMM/HCS 六项全部严格超过 SOTA，并胜 no-recovery、sequence-only、program-shuffle、policy-edge-shuffle与 capacity-matched MultiHateLoc；失败即归档，不扩MHC、不改 prompt、不按语料增删 state。
