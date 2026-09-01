# 淘汰：Context-residual segment refinement

淘汰原因：独立 novelty review 给出 `STOP, 4.2/10`。虽然未发现 ASM-Loc 已用于
hateful video detection/localization，但 context separation + completeness modeling 与
action/context 双子空间已有直接跨任务先例；更关键的是，只有同一 video label 时无法识别
持续 topic context 与局部 hostility。gated residual 可通过关闭 gate 或 prototype projection
退化为普通特征，global video probability 不改变 within-video 排序，ASM-Loc 多轮 pseudo
segment refinement 又会确认自身错误。因此没有批准实现边界，未实现、未训练、未生成
prediction。完整依据见 `NOVELTY_REVIEW.md`。

截至 2026-08-31。当前状态：已在 novelty/identifiability 阶段淘汰，未实现、未训练、
未产生 prediction。

## 研究问题与直接证据

当前跨语料 trainable starting architecture 是 MultiHateLoc。它对每秒独立分类，使用
固定 top-third MIL；文本特征又把每个 Whisper fragment 的独立 BERT CLS 向量复制到
其覆盖的所有秒。HateMM/HateClipSeg 全数据的文本覆盖率均约 82%，每视频 ASR fragment
中位数分别为 7/11。已有 test error analysis 还显示：HCS 的 fused branch 超过全部
单模态 branch 的视频比例只有 `.154`，proposal oracle within ROC 为 `.63450`，但已有
P-MIL proposal scoring 只有 `.47661`。因此当前共同问题不是没有候选时间区域，而是：

1. video-level topic/context 被当成 temporal hate evidence；
2. 独立秒 MIL 只抓最显著秒，不能稳定地把局部证据变成完整 segment；
3. 多模态融合把无证据 modality 当成同等 teacher，HCS 尤其严重。

上述数字来源：

- `runs/20260831_multihateloc_test_error_analysis/main/metrics.json`
- `runs/20260831_multimodal_pmil_baseline/pilot_seed234/hateclipseg/metrics.json`
- `/home/jehc223/Hate-follow-up/results/reproduction/features/bert_sentence_1fps/{hatemm,hateclipseg}/index.json`

## 跨任务来源方法

来源方法是 ASM-Loc（He et al., CVPR 2022, *Action-Aware Segment Modeling for
Weakly-Supervised Temporal Action Localization*）。官方源码只用于审读，位于
`third_party/ASM-Loc/`。其核心包括 dynamic segment sampling、segment 内/segment 间
attention、pseudo instance supervision 和多轮 proposal refinement；训练只用 video-level
action labels。

ASM-Loc 的 action-aware background loss 明确把同一视频 background 也训练成能够预测
action category：动作数据里的台球桌等背景对类别有用。这个假设不能直接带入 hateful
localization。仇恨视频中，整段视频可能持续讨论同一 target/topic，但只有局部 statement
表达贬损、威胁或伤害；让 background 预测 hate 会制造整段 topic shortcut。

## 非 trivial adaptation

候选暂称 context-residual segment refinement。每个主数据集仍完全独立训练，只使用该
数据集 train video labels；validation 只选 checkpoint；test labels 不参与训练或选择。

### 1. 两个语义变量而非一个 foreground attention

模型显式分开：

- `g_t`：可跨整段持续的 video topic/context representation，用于 video-level 判别；
- `r_t`：相对同视频背景 context 新增的局部 hostile-event residual，用于最终 frame
  posterior 与 segment refinement。

初始背景权重来自当前 local head 的低分位秒，但仅在 train forward 内计算。对每个
modality 独立形成 stop-gradient background prototype `b_m`，再用 learned gated residual
`r_m(t) = P_m x_m(t) - gate_m(t) P_m b_m`。gate 由当前秒与背景 prototype 的相似性决定，
避免无条件相减删除真正的 hate evidence。三个 residual 经 missing-evidence mask 融合；
没有 ASR 的秒不得把零文本向量当成 negative text evidence。

### 2. ASM-Loc segment refinement 只作用于 residual localizer

第一阶段以 video labels 训练 global head 与 residual local head。随后只从 train
predictions 生成 pseudo segments；ASM-Loc 的 dynamic short-segment sampling、segment 内/
segment 间 attention 和多轮 pseudo instance refinement作用于 `r_t`，不作用于 `g_t`。
这使 topic branch 不能把整段高分直接写回 pseudo localization labels。

### 3. 单一概率模型，不做 post-hoc ensemble/routing

最终 frame posterior 由同一模型直接输出为
`p(y_t=1|V)=p(video positive|V) * p(t is local hostile event|V, video positive)`。
前一项对同视频是常数，所以不能伪造 within-video 排序；后一项承担全部 localization
机制。不得在 test 后选择 branch、阈值、平滑或语料特定组合。

## 可证伪机制假设

如果机制成立：

1. core 相对 `no-residual ASM-Loc` 在 HateMM/HCS 两边 within ROC 同向提升，且至少一边
   `>= .020`；
2. `no-stop-gradient` control 应更容易被 topic shortcut 污染，core 应优于它；
3. `unconditional subtraction` control 会误删局部证据，core 应优于它；
4. 在 test error analysis 中，core 的增益应集中于 positive occupancy 较低、同视频
   topic feature 持续但 GT 发生切换的视频；若增益只来自 video multiplier 提高 pooled
   指标而 within 不变，则机制失败；
5. 最终必须在 HMM/HCS 同时超过三项固定 SOTA gate，才扩展 EN/ZH。任何一项不是
   SOTA 都不得晋级为完成方法。

## 固定 pilot 与 anti-pattern guard

- Pilot corpora：HateMM、HateClipSeg，各自独立训练。
- Starting features 与 MultiHateLoc 相同：1 fps visual/audio/text；不读取任何 train span。
- Validation：只按固定方法的 video AP 选择 checkpoint，不用于比较 arms 或改方法。
- Evaluation：checkpoint 选定后立即在 test 上跑 pooled AP、pooled ROC、within-video
  macro ROC；test prediction/GT 可用于记录 developmental error analysis。
- 三个 controls 与 core 使用同一训练预算、seed、evaluator 和输入。
- 不允许固定 smoothing、test routing、score ensemble、test threshold search、按语料换
  机制、或从 test oracle 生成训练 pseudo label。
- 正式运行前必须由独立 reviewer 完成 novelty/source/identifiability review；实现后再由
  另一独立 reviewer 检查 train/validation/test isolation、pseudo-label 来源和 evaluator。

## Novelty review 必须回答的三门

1. 允许基于 ASM-Loc adaptation，这一条本身不是问题。
2. ASM-Loc 或相同核心是否已经用于 hateful video detection/localization？必须实际检索。
3. `topic-context residualization + residual-only segment refinement` 是否构成针对本任务
   topic shortcut 的 non-trivial adaptation，还是只是 ASM-Loc、prototype subtraction 与
   two-head MIL 的组件拼接？是否存在数学退化、不可识别性或更近的已占用方法？
