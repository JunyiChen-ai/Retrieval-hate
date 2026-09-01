# Witness-Failure Debiasing MIL

**截至 2026-09-01。候选 brief；尚未实现或运行。**

## Failure

MultiHateLoc 的现有四语料 test error analysis 显示，DMS 与逐视频最佳单模态的匹配率只有
`.216/.333/.375/.323`，fused 胜过全部单模态的比例只有
`.345/.159/.042/.154`。RESET5 candidate 1 又表明 fused 模型自身的 coalition credit
在 HMM 上不胜时间错位 control，因此不能把同一个 fused scorer 的自解释继续当作局部模态可信度。

## Cross-task source

Nam et al., *Learning from Failure: Training Debiased Classifier from Biased
Classifier*, NeurIPS 2020。来源方法用 GCE 训练一个主动放大 easy shortcut 的 bias model，
再以 bias/core 的相对困难度加权 debiased model。来源是有显式样本标签的分类，不是 hateful
video detection/localization，也不是 temporal MIL。

## Task adaptation delta

本任务没有正例秒标签，因此不把 video label 广播到所有正例秒。三条约束共同构成 adaptation：

1. MultiHateLoc 的三个单模态 branch 是 bias experts，只在各自的 video-level top-K MIL
   probability 上用 GCE；它们显式吸收容易的单模态/video-global shortcut。
2. 最终 learner 仍是原 fused branch。正视频只在 detached fused top-K latent witness set 内
   计算逐秒 positive BCE，并用 LfF 相对困难度
   `CE(bias_t,+)/(CE(bias_t,+)+CE(fused_t,+))` 归一化后加权；`bias_t` 是三个 detached
   单模态 probability 中最容易解释该秒的最大值。负视频所有有效秒均有确定负标签，保留原
   fused top-K negative MIL，并增加同式 hard-negative weighting，不制造正秒标签。
3. 加权项强度 `lambda_failure=0` 时，模型、forward、raw fused test score、基础 loss 和
   schedule 精确退化为同 harness 的 MultiHateLoc anchor。正式输出永远只有单个
   `score_fused`，bias experts 不参与 inference ensemble、routing 或 calibration。

这不是把 LfF 的 sample weight 原样搬到 video：核心改造是把相对失败信号限制在 MIL 当前
选择的 latent witness support，并区分“负视频所有秒可监督”和“正视频只有 latent witness
可训练”，使 shortcut failure 直接改变 time-local fused gradients。

## Final-score path

`modality frame probabilities -> detached max bias probability -> relative-failure weight on fused
top-K witness loss -> fused branch parameters -> raw score_fused`。

## Falsifiable test expectation and control

- HMM 与 HCS 的 validation 分别选择超参数/checkpoint后立即 test。机制最低预期：core 的
  within-video ROC 在两个语料都高于 matched uniform-witness control，且至少一个语料提高
  `>= .010`；最终目标仍是两个语料三个固定 test 指标全部 SOTA。
- Matched control 使用完全相同的 bias experts、GCE、参数量和训练 schedule，但把 fused
  top-K 内 relative-failure weights 替换为 uniform weights。另报告 `lambda_failure=0`
  exact anchor。
- 机制诊断：core 相对 control 的逐视频 within 增益应集中在原 anchor 的
  `best unimodal > fused` 视频；若两语料不一致，或 matched control 不输，则关闭该 family，
  不扫描 bias aggregation、GCE 变体或 witness producer。

## Validation selection (pre-registered)

每语料完整训练两个 learning rate × 三个 `lambda_failure`（建议 `.25/.5/1.0`）的 core，
并为每个 learning rate 训练 matched uniform control；每个 trial 在完整官方 epoch budget 内
逐 epoch 选择 checkpoint。选择以 validation within ROC 为主，相对同 learning-rate control
的 pooled AP 与 pooled ROC 各自最多下降 `.01`；若无可行配置，先最小化最大违反量，再最大化
within。两个语料独立选择，锁定后立即生成 test prediction。禁止 smoke、缩数据或缩 epoch。

## Sources checked so far

- Nam et al. NeurIPS 2020 official paper/abstract:
  https://arxiv.org/abs/2007.02561
- 初步精确检索未发现 LfF、GCE bias-amplification 或 relative-difficulty weighting 已用于
  hateful video detection/localization；最终占用裁定由独立 novelty reviewer 给出。

## Formal result and disposition

独立 novelty review 为 `GO 6.5/10`，唯一跑前基础 technical review 为 `PASS`。HMM/HCS
各自完成 14 个完整 validation-only trial（2 anchor、6 uniform、6 relative），联合选择
超参数与 checkpoint 后立即生成 test prediction。权威汇总：
`runs/20260901_witness_failure_debias_mil/formal_val_selected_seed234/summary.json`。

- HateMM anchor/uniform/relative 的 AP、pooled ROC、within ROC 分别为
  `.490302/.737340/.632938`、`.487314/.736276/.631975`、
  `.487075/.736710/.632554`。Relative 相对 uniform 为
  `-.000239/+.000435/+.000579`。
- HateClipSeg 三臂分别为 `.523714/.497501/.525970`、
  `.538995/.516846/.530318`、`.538333/.516499/.529759`。Relative 相对
  uniform 为 `-.000661/-.000347/-.000559`。

机制门失败：两语料方向不一致，且没有任何语料达到预注册的 relative-vs-uniform within
`+.010`。双语料 all-SOTA 门同样失败。没有共同 post-test corrective evidence，关闭
GCE aggregation、relative weight、support producer及其强度变体；本实验淘汰归档。
