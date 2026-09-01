# CLIP policy-direction reopening premise

截至 2026-08-31。本轮不是 method candidate，不作 novelty claim，不替代
MultiHateLoc，也不增加 RESET epoch 候选计数。它只检验一个预先固定、覆盖每个
1fps 画面的 native statistic 能否通过 Rule 14。

**淘汰原因（2026-08-31）**：正式双语料 test controls 已完成，权威输出为
`runs/20260831_clip_policy_reopening/main/metrics.json`。HMM/HCS raw
AP/pooled ROC/within 分别为 `.297188/.573920/.526238` 与
`.540172/.526173/.517859`。HMM 的 aggregate time-shift、mean-repeated、
position-only 通过，但 ASR-low、OCR-high、visual-low strata 失败；HCS 的三个
aggregate controls 和 carrier-strata 联合门全部失败。裁定
`KEEP_CANDIDATE_FREEZE`；不换 prompts、模型、聚合或 threshold。

## 冻结 statistic

对现有 `openai/clip-vit-base-patch16` 1fps image embedding，使用同一模型编码四条
公开 label-definition 导出的 policy prompts 与四条 matched benign prompts。各组先
平均再归一化，唯一 raw score 为 image embedding 对
`policy_centroid - benign_centroid` 的 cosine projection。HMM/HCS 使用完全相同的
模型、prompts、公式和参数；不训练、不校准、不融合其他 score、不按语料 routing。

这个 statistic 与已失败的 ASR lexical source 不同：每个 grid second 都有原生视觉
embedding，因而在无 speech/OCR 秒不会解析退化为统一空向量。它也不是曾被拒绝的
patch spatial-excess candidate：这里只验证 global frame observation，不提出 spatial
机制或方法 claim。

## 冻结 controls 与 gate

Producer 只读 frozen features、split membership、本语料 train video labels 与
label-blind ASR/OCR availability；不读 test labels 或 temporal GT。Evaluator 才读取
test GT。

1. 每个 eligible positive test video 最多 16 个均匀非零 circular shifts；raw-minus-
   shift 在 HMM/HCS 均须 `>=.020`。
2. 每视频 raw mean 重复到全时间；raw-minus-mean-repeated 均须 `>=.020`。
3. 只用本语料 positive-train raw scores 构造 20-bin relative-position template；
   raw-minus-position-only 均须 `>=.020`。
4. ASR coverage、OCR coverage、visual change 三项都只用 positive-train median 冻结成
   low/high 两档。六档各须至少 5 个 both-class positive test videos，且每档
   raw-minus-shift `>=.020`。
5. Exact cohort、shape、finite、feature/ASR/OCR coverage 全部 fail closed；完整 score
   maps 只调用共享 canonical evaluator。

任一失败即 `KEEP_CANDIDATE_FREEZE`，不换 prompts、模型、聚合或 threshold。全部通过
也只记 `REOPENING_EVIDENCE_PASS_ONLY`，之后才允许生成并独立审查真正的 cross-task
method candidate。
