# 淘汰：Policy-complete proposal MIL

> 淘汰原因：独立 novelty review `STOP`（4.1/10）。P-MIL 未见用于 hateful task，但当前设计只是
> P-MIL completeness head 与 POWA-style frozen policy score/pseudo-label gate 的组件拼接；没有 joint
> role binding，HCS unary clauses又退化为 ordinary prototype saliency。未实现、未训练、未生成新
> test prediction；完整裁定见 `NOVELTY_REVIEW.md`。

截至 2026-08-31；状态：查新阶段淘汰。

## 唯一核心 adaptation

来源方法是 P-MIL（Ren et al., CVPR 2023）：它把 weakly-supervised temporal localization 的训练与
推理单位从独立 snippets 改为 interval proposals，并用 surrounding contrast、proposal completeness
与跨 view rank consistency 抑制只覆盖最显著 action part 的短 proposal。P-MIL 尚未见用于 hateful
video detection/localization。本候选不 claim proposal MIL、SCFE、pseudo-label refinement、rank
consistency、moderation policy primitives 或 MIL 本身。

要审查的唯一 claim 是：**把 P-MIL 的 action completeness 改成 policy-clause completeness；一个 hate
proposal 只有在 interval 内包含足以使目标语料 label policy 的至少一个 admissible clause 成立的全部
证据，并且该 clause 在 matched surrounding 中不成立时，才是 complete positive proposal。** 这不是
“偏离 benign density”：positive-bag BCE 直接学习 proposal ordering，negative bags 直接约束所有
proposals。它也不要求每个 hateful corpus 都必须出现 protected target；HateMM 的 clause 是 targeted
hate，MHC 允许 targeted hate/untargeted abuse，HateClipSeg 还允许 violence/sexual/self-harm。语料间不
共享 train data、模型参数或伪标签；policy clause 只编码各数据集已经公开的 label definition。

Primary source:
https://openaccess.thecvf.com/content/CVPR2023/html/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html

## 针对当前 test 失败的机制故事

MultiHateLoc 的 test error analysis 显示 DMS 几乎总选 visual，选中 test-GT 最佳单模态的比例只有
`.216/.333/.375/.323`；frame/snippet MIL 把视频标签广播后没有可靠决定“哪一段、哪些模态共同完成了
label”。Factorial witness CRF 又把大量 path 平均成高熵 attribution；conditional normal proposal
probe 则在 HMM/HCS 都产生低于 `.5` 的 within ROC，证明“异常于 benign”不是正确排序方向。

P-MIL 解决训练时 snippet、测试时 interval 的单位错位；本 adaptation 进一步规定 hateful proposal 的
完整性不是 generic foregroundness 或 duration，而是 policy clause 是否在该 interval 内完成。例如，
target mention 与 hostile predicate 分处相邻秒时，只含其中一端的短 proposal 即使显著也应被压低；
但 HCS 的 violence clause 不被强迫寻找 target。这样机制同时允许短单模态 offense 和跨时刻/跨模态
targeted hate，而不通过 corpus-specific top-K、时长或后处理路由实现。

## 固定最小实现

首轮 seed 234，仅 HateMM 与 HateClipSeg；两语料使用完全相同代码、proposal grid、容量、优化器与
epoch budget，各自独立训练。Validation 只按固定 video-level BCE 选择本次训练 checkpoint；选定后
立即在 test 评测 pooled AP、pooled ROC 与 within-video macro ROC。test predictions/GT 只在评测后做
developmental error analysis，绝不进梯度或 checkpoint selection。

1. 输入为现有 1fps 对齐、row-normalized A/V/T local features。共享 temporal encoder 产生 frame
   embedding；不加载 POWA/MultiHateLoc prediction 或 checkpoint。
2. proposals 固定为每秒起点的 `1/2/4/8/16/32/64/128s` intervals 加 whole video；训练时若显存不足，
   必须在运行前冻结一个与 label/score 无关的均匀 proposal subsampling rule，core/controls 共用。
3. faithful P-MIL descriptor 是 inside pooled embedding、左右同尺度 surrounding pooled embedding 与
   inside-minus-surrounding residual。相同 proposal head 输出 base proposal logit；positive bag 用
   smooth max，negative bag同时使用 smooth max 与全-proposal negative loss。
4. 六个 frozen semantic prototypes只产生每秒 role evidence：hostile、target、violence、sexual、
   self-harm、protected context。role evidence 不单独预测 hate，不接受 test normalization；英文用于
   HMM/HCS，后续 MHC-ZH 固定用中文 prototype。
5. 对每个 proposal，在其内部对每条 admissible policy clause 做 differentiable role coverage；targeted
   hate 为 hostile 与 target coverage 的 soft-min，再乘 `(1-context)`，untargeted abuse 与 HCS 单角色
   offense clauses沿用公开 policy AST。matched surrounding 用同一 AST 得到 clause coverage。
6. `clause_completeness = max_clause(inside_clause - surrounding_clause)`。它只作为 proposal
   completeness pseudo-label 的一个必要因子：stop-gradient 的 base confidence 与 normalized positive
   clause completeness 同时过冻结阈值才产生 positive pseudo proposal；negative bags 不产生 positive
   pseudo labels。proposal completeness head用该 pseudo target训练。
7. 最终 proposal logit是 base classification logit加全局标量乘 completeness log-odds；bag BCE仍是
   唯一 hate label supervision。test frame score为覆盖该秒 proposals 的 normalized positive weights，
   constant proposal logits 必须产生平坦 temporal score。

不得加入 conditional density、flow、CRF、generic smoothing、POWA score ensemble、test-fitted
calibration、按语料 top-K/threshold 或其他第二核心机制。

## 必须同跑的归因 controls

1. `faithful_pmil`：完全相同 encoder、proposal grid、SCFE、base head、训练预算与 readout，只使用
   P-MIL 原 completeness pseudo target，不输入 policy clauses。
2. `core`：faithful P-MIL + policy-clause completeness，唯一变化。
3. `role_time_shuffle`：每个 train video 内共同打乱 role evidence 的时间位置，保持每个 role 的边际、
   proposal数和网络容量。
4. `flat_role_max`：相同六个 role coverage直接 max，不执行 clause AST，区分 typed completeness 与
   generic semantic saliency。
5. `inside_only_clause`：去掉 matched surrounding clause，检验 surrounding contrast 是否 load-bearing。
6. `base_only_readout` 与 `completeness_only_readout`：定位实际增益来源。
7. proposal length/center、whole-video mass、active proposal count/entropy、constant-logit flatness、
   clause type mass、role deletion、inside/surrounding swap。

所有 controls 共享同一 validation checkpoint rule，并在 checkpoint 选定后立即跑 test 三指标；不得先看
validation arm 排名再决定哪些 arms 进入 test。

## 冻结晋级与淘汰门

首轮是一次正式两语料方法 pilot，不再设 test 前 performance gate。`core` 必须同时满足：

1. HateMM 与 HateClipSeg 的 pooled AP、pooled ROC、within ROC 全部严格超过当前 SOTA；
2. 两语料 within ROC 都严格超过 `faithful_pmil`、`role_time_shuffle` 与 `flat_role_max`；
3. 至少一语料 core-minus-faithful within `>= .020`，另一语料不得下降；
4. role deletion 对被选 proposal 的 logit 降幅显著高于 shuffled deletion，inside/surrounding swap 使
   completeness下降；否则 policy-completeness attribution FAIL；
5. pooled 增益不能只来自视频级尺度；必须报告每视频 rank-only within、positive-rate strata 与 score
   multiset诊断。

任一语料三指标 SOTA 失败或机制 controls 失败即淘汰，不调 policy、threshold、proposal length 或
readout，不扩 MHC-EN/ZH。两语料全部通过才扩四语料、多 seed与深度 novelty review。

## Reviewer 必须优先阻断的问题

- 完整 adaptation 是否已被 hateful localization/detection 使用；
- policy-clause completeness 是否只是把 P-MIL pseudo-label 输入换成 POWA semantic score，属于 trivial
  feature substitution；
- frozen role prototype 可靠性是否已被本项目的 primitive negative results直接否定，使机制不可证伪；
- HCS 单角色 clauses 是否使“clause completeness”退化为 ordinary saliency；
- positive pseudo proposal 是否形成 label broadcasting/circular self-training，controls 能否隔离；
- normalized proposal-to-frame readout 是否仍有 coverage/length shortcut。
