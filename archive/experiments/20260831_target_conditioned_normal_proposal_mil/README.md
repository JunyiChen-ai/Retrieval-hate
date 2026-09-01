# Target-conditioned normal proposal MIL

> 淘汰原因：冻结两语料 premise gate 失败；HateClipSeg topic support 仅 `.28869`，且 conditional
> normal energy 的 within ROC 在 HateMM/HateClipSeg 均低于 matched unconditional。按预定规则
> `STOP_BEFORE_FORMAL_METHOD`，未实现或训练正式 flow/MIL 方法。

截至 2026-08-31；状态：独立 novelty review 窄范围 PASS（6.2/10），premise result-chain 独立审计
PASS，机制前提 FAIL，已淘汰。

## 最终 premise 结果

权威 verdict：`runs/20260831_target_conditioned_normal_proposal_mil/premise/verdict.json`；逐 arm 原生
共享 evaluator 输出位于相同 run root 的 `{hatemm,hateclipseg}/metrics.json`。本轮是允许驱动后续开发
的 developmental test evidence，不是正式方法或 SOTA prediction。

| corpus | arm | pooled AP | pooled ROC | within ROC | topic support |
|---|---|---:|---:|---:|---:|
| HateMM | conditional | .19172 | .40397 | .29331 | .86960 |
| HateMM | unconditional | .19296 | .40846 | .29633 | .86960 |
| HateClipSeg | conditional | .49010 | .45998 | .38494 | .28869 |
| HateClipSeg | unconditional | .49067 | .46078 | .39528 | .28869 |

HateMM/HateClipSeg 的 conditional-minus-unconditional within 分别为 `-.00302/-.01034`；HCS support
也未过 `.80`。这同时否定了“同 topic benign support 足够”和“topic conditioning 改善 hate interval
排序”两个必要前提。不得事后调 topic PCA、support bandwidth、proposal 尺度或 flow；下一候选不得
继续以 conditional one-class anomaly energy 为核心。

## 可主张的跨任务 adaptation

来源方法是 proposal-based MIL（P-MIL，CVPR 2023）与 conditional normalizing-flow anomaly
localization（CFLOW-AD，WACV 2022 / ContextFlow++，UAI 2024）。这些方法尚未用于 hateful-video
detection/localization，但 proposal MIL、inside/surrounding contrast、conditional density、normal-reference
anomaly detection 和 weak video MIL 都是已知方法，不能分别 claim novelty。

唯一允许的窄 claim 是：**hate 是相对于相同 protected-target/topic 的 benign use 才成立的关系性
residual；因此每个 temporal proposal 的 instance energy 必须由同 topic 的 conditional normal model
定义，再由 positive video bag 选择低-normal-likelihood interval。** 检索未发现这套具体 adaptation 已
用于 hateful-video detection/localization。完整查新与限制见 `NOVELTY_REVIEW.md`。

## 固定数学定义

对长度 `T` 的 1fps 视频枚举所有非空 interval proposals `I=[s,e)`，不按视频长度设事件比例、不用
top-K。proposal descriptor 使用 prefix sums 计算：

1. `z_I` 是 frozen text sentence features 在 interval 内的均值，先投影到 policy harm prototype
   span（hostile/violence/sexual/self-harm/context）的正交补，再以仅 negative-train proposals 拟合的
   PCA 压到固定维度。`z` 全程 stop-gradient，只定义 protected-target/topic context；不得输入 video
   ID、video pooled label feature 或 test information。
2. `r_I` 是可训练 A/V/T proposal-local encoder 的 inside mean 与左右同尺度 surrounding mean 的
   residual；没有可用一侧时只用另一侧，whole-video proposal 的 surrounding 置零并作为 shortcut
   control 单独报告。`r` 固定归一化后进入 conditional flow。
3. conditional normal flow `p0_phi(r_I | z_I)` 只在目标语料 negative-train proposals 上最大化
   likelihood。proposal energy 为维度归一化 NLL：`a_I=-log p0_phi(r_I|z_I)/dim(r)`，不是
   `p_hate/p_normal` ratio，也不声称真实或 causal hostility density。
4. 每个视频对 standardized proposal energies 做 sparsemax 得到 `w_I`。bag logit 是
   `b + scale * sum_I w_I a_I`；video BCE 使用 train video label。positive/negative bag BCE 都可更新
   proposal-local `r` encoder和两个全局 affine 参数；flow 参数只接收 negative proposal MLE，`z`
   永不更新。这样 positive weak labels能 orient proposal representation，但不能直接把 positive samples
   写进 normal density。
5. test 不用 video label。frame `t` 的 score 为覆盖该秒且 `w_I>0` proposals 的归一化加权概率：
   `sum_{I contains t} w_I sigmoid(b+scale*a_I) / sum_{I contains t} w_I`；无 active proposal覆盖时为
   0。constant proposal logits 必须产生时间上平坦的 score，避免 noisy-union coverage-count bias。

## 为什么针对已观察失败

factorial CRF 对全部 non-empty paths 求平均后，三个 modality posterior 都约 `.46–.52`，形成高熵
弥散 attribution；learned transition 相对 zero-transition 没有跨语料增益。这里 latent unit 是显式
interval proposal，sparsemax 自适应选择可变数量 proposals，不平均指数级 paths。

普通 normal-reference UOT 又因全视频 frames 竞争共享 normal capacity，在长/high-positive 视频压平
排序。本候选不分配共享 transport mass；每个 proposal 独立询问“在相同 topic 下，这个 relational
residual 对 benign normal model 是否异常”。这也直接控制 protected identity/topic 本身被当成 hate
shortcut 的问题。

## 冻结 premise gate（正式实现前）

先在 HateMM/HateClipSeg 用一个无 flow、无 positive-label optimization 的 fixed conditional Gaussian
probe 检查两个必要前提：

1. negative-train topic support：两语料 test proposals 都必须有足够的同 topic negative support；
2. target conditioning 的方向：conditional normal energy 必须在两语料 test within ROC 都严格超过
   matched unconditional normal energy，并至少一边达到 `+.020`。

probe 只作为 developmental test premise，不是候选方法或 SOTA 数字。固定设置为：1fps 上枚举每秒
起点的 `1/2/4/8/16/32/64/128s` proposals，并加入 whole-video proposal；negative-train frames 拟合
topic PCA-16 与 A/V/T residual PCA-32；conditional Gaussian 用 ridge `1.0` 从 topic 预测 residual，
unconditional Gaussian 使用同一 residual；两种 energy 都按各自在 negative-train proposals 上的
均值/标准差标准化。frame readout 是覆盖该秒 proposals 的平均 standardized energy，用 difference
array 实现，constant energy 必须严格平坦。negative proposal topic PCA 上拟合 64-centroid k-means，
negative 内部到最近 centroid 距离的 95th percentile 为 support 门；每个 test corpus 至少 80%
proposals 必须在门内。任一 support/direction 前提失败就 `STOP_BEFORE_FORMAL_METHOD`，不得为语料选择
不同 topic bandwidth、flow 或 feature。

## 正式 controls 与晋级 gate

若 premise 通过，所有 arms 在每个语料分别重训，validation 只选本次固定训练 checkpoint，随后立即
test 三项指标：

1. `core`：target-conditioned normal flow + positive-oriented proposal encoder。
2. faithful `plain_pmil`：相同 proposals/SCFE/encoder/aggregator/参数量，无 normal model。
3. `unconditional_flow`：相同 normal flow与容量，仅移除 `z`。
4. `target_shuffled`：negative proposals 中打乱 `z↔r`，保持样本数和计算量。
5. `proposal_ngmil`：normal prototypes + similarity refinement，不按 topic 条件化。
6. `retrieval_augmentation`：相同 context neighbors 只拼到 P-MIL classifier，不形成 conditional energy。
7. `target_only`、`residual_only`、matched kNN/Gaussian、frozen one-class，以及 constant-logit/coverage
   readout diagnostics。

首轮 seed 234，HateMM/HateClipSeg。core 必须在两语料全部 pooled AP、pooled ROC、within ROC 严格
超过 SOTA；within 还必须严格超过 trainable controls。任一语料失败即淘汰，不扩 MHC-EN/ZH，不按
corpus 路由。

必须报告 active proposal count/entropy/length/overlap、whole-video proposal mass、topic support、flow
base-density 与 log-Jacobian贡献、rare-benign error、temporal shuffle、inside/surrounding swap。若
conditional flow不超过 unconditional、target-shuffled、matched kNN 和 faithful P-MIL，novelty 自动
转为 FAIL。

## 最窄 claim

> A target-conditioned one-class proposal MIL adaptation for hateful-video localization, where a
> conditional normal model learned from benign training proposals scores relational residuals relative to
> the same protected-target/topic context, and positive video labels orient at least one low-normal-
> likelihood temporal proposal.
