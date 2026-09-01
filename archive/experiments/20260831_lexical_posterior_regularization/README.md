# 淘汰：双语料 Stage A test performance 全败，HCS mechanism gate 亦失败；不运行 Stage B、不调参。

# Lexical posterior regularization

截至 2026-08-31。候选 novelty review：`CONDITIONAL GO 5.9/10`。来源机制为
Generalized Expectation / Posterior Regularization；独立 review 未检出该核心已用于
hateful-video detection/localization。只有下述 latent posterior projection 可以作为
候选；lexical concat、BCE pseudo-label、hinge/rank distillation 均不构成该方法。

## 机制假设

标准 video-label MIL 可以靠整段 topic broadcast 完成 bag classification。上一轮
冻结 premise 证明，同语料 train video labels 学到的 char-ngram direction 在 HMM 与
HCS 都携带真实、虽在 HCS 很弱的时间位置信号。本方法不把 lexical seconds 当 GT，
而把它们变成 latent frame posterior 的集合期望约束：

`q* = argmin_q KL(q || p_theta)`。

- 每个 train video 的 lexical evidence 必须由不含该视频的 5-fold OOF classifier
  产生 OOF local-window lexical logits，避免 rare-ngram 自记忆；本实现不声称
  做 token-level contribution transport。
- 所有 support 首先要求至少 10 个 speech-supported 秒。positive video 使用
  speech evidence 的 20/80 percentile thresholds，阈值上的 ties 全部纳入；
  high/low 各至少 2 秒且 evidence gap 非零时，约束
  `mean(q_high)-mean(q_low) >= .20`；否则 abstain。
- negative video：使用 speech evidence 的 80 percentile superlevel，ties 全纳入，
  并要求 evidence `>0`、集合至少 2 秒；约束
  `mean(q_high) <= .10`。negative bag 的全部秒是 exact benign，这一项校正
  corpus-specific lexical false positives。
- projection 只作用于 bounded fused probabilities；普通 Bernoulli product KL
  的 dual shift 严格按集合大小缩放（high `+lambda/n_high`、low
  `-lambda/n_low`，negative `-lambda/n_high`）。每个 batch 解析固定约束的
  KL projection、核验 primal/KKT residual，并训练 raw student `p_theta` 逼近
  detached `q*`。
- inference 完全丢弃 lexical evidence，只评 raw fused student。

## Stage A 冻结设置

backbone、MIL、smoothness、contrastive、optimizer、100 epoch 与 MultiHateLoc
faithful anchor 相同；validation 只在每个固定 arm 内选择 video-AP checkpoint，
不参与方法比较或方向判断。固定 `lambda_PR=1`，不扫 constraint、quantile、margin、
epsilon 或 loss weight。

Stage A 在 HateMM/HateClipSeg 各自独立训练 anchor 与 core，并立即在完整 test
cohort 评测 pooled AP、pooled ROC、within-video ROC。只有 core 同时满足：

1. 两语料三项指标全部达到当前 SOTA 门；
2. 两语料 within 均高于 matched anchor，至少一个语料提升 `>=.020`；
3. train support 的正、负视频数都非零，所有诊断 finite，且 raw student 的正、
   负 expectation violation mean 各自相对初始化至少下降 10%；

才运行 Stage B 的 timestamp-shuffle、direct-rank、pointwise pseudo-label、
negative-only、positive-only、posthoc-concat controls。Stage A 任一硬失败即归档，
不调参数、不按语料 routing。

首次 formal run 在 HMM core 训练中因 float32 projection residual
`2.136e-4 > 2e-4` fail-closed 中止，未生成 core test prediction，不评价方法。
修复仅把 dual projection 改为 float64；重跑输出目录冻结为
`runs/20260831_lexical_posterior_regularization/stage_a_fix1/`。fix1 随后在 HMM
core 训练中再次 fail-closed：接近 0/1 的 target 被旧 `1e-6` clamp 后反推
logit，产生伪 KKT residual `1.226`；仍未生成 core test prediction。fix2 统一
projection/KL 的 float64 数值域，以 `xlogy` 计算 Bernoulli KL，并直接在生成
target 的 dual-logit 域审计 stationarity，不从饱和 probability 反推 logit。
fix2 权威目录冻结为
`runs/20260831_lexical_posterior_regularization/stage_a_fix2/`。

## 最终结果

独立 post-run result-chain audit：PASS。权威 verdict：
`runs/20260831_lexical_posterior_regularization/stage_a_fix2/stage_a_summary.json`。

| corpus | arm | pooled AP | pooled ROC | within ROC |
|---|---|---:|---:|---:|
| HateMM | anchor | .481054 | .742432 | .605824 |
| HateMM | core | .480173 | .719155 | .603966 |
| HateClipSeg | anchor | .506771 | .491895 | .507167 |
| HateClipSeg | core | .504344 | .488873 | .506184 |

Core-anchor delta 为 HMM `-.000881/-.023277/-.001858`，HCS
`-.002428/-.003022/-.000983`。两语料三项 SOTA、within 胜 anchor、任一边
`>=+.020` 全失败。HMM train positive/negative violation 从
`.200081/.409038` 降到 `.104652/.010860`，机制门通过但未转成 test ranking；
HCS 为 `.200215/.408531` 到 `.199891/.426758`，机制门失败。

裁定 `STOP_AND_ARCHIVE`。这淘汰固定的 OOF lexical posterior-regularization
实现，不否定上一轮 lexical-locality premise。不得运行 Stage B controls、扫描
constraint strength/loss weight/support threshold，或按语料 routing。
