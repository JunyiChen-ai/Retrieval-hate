# Source-scoped proposition graph MIL

**淘汰原因：旧 LB-SCGP 已在 hateful-video detection 中占用 source/endorsement/quotation/stance binding；当前时间化 complete-path 版本仍是 endpoint/granularity 变化，novelty Gate 2/3 失败。**

截至日期：2026-08-31。当前阶段：独立 novelty review 后停止；未运行 premise，未训练正式方法，未生成 prediction。

## 研究问题

MultiHateLoc 与后续失败候选能识别局部 hostile/target evidence，却不能区分该命题由谁负责、是当前说话者认同，还是引用、报道、否定或谴责。若数据集把这些语用作用域区分为 hateful/benign，这会使相同的 offensive words 在 positive video 内被错误地整段广播。

## 跨任务来源与非平凡 adaptation

来源方法族是 quote attribution、speaker/addressee attribution 与 target-conditioned conversational stance，不是 hateful-video detection/localization 方法。adaptation 不是把 stance score 拼到现有特征：

1. 每个有时间戳的 ASR/OCR/caption proposition 保留 typed nodes：accountable source、quoted/current scope、hostile predicate、protected target、stance。
2. 只有完整的 `source -> endorses -> hostile(predicate, target)` 路径能产生 positive temporal instance；reported/quoted/denied/condemned 路径保留为不同结构，不能与 endorse 合并成一个 hostile scalar。
3. frame score 是覆盖该秒的完整路径 posterior；正式训练时该 posterior 只能由 frozen external parser 和目标语料自身 train video labels形成 auxiliary instance responsibility，不能使用任何 frame/span label。
4. 机制 control 必须删除或置乱 source/scope edge，同时保留相同 hostile words、target mentions、proposal geometry 与参数容量；若 core 不胜这些 controls，命题归责不是 load-bearing mechanism。

必须额外与项目旧 LB-SCGP 区分：`archive/refine-logs/lb_scgp/FINAL_PROPOSAL.md` 已把 whole-video direct-speaker endorsement、quotation/condemnation/reportage exception 和 speaker-source/stance binding 用于 hateful-video detection 候选。因而本候选不能 claim 首次把这些语义用于 hateful video。唯一可能保留的新增量是 **有时间戳、保留 proposition 与 accountable-source identity 的 complete-path latent instance**，以及由该结构直接定义 temporal responsibility posterior。若独立 reviewer 认为这只是把旧 whole-video certificate 切成时间片，第二、第三 novelty 门失败，候选在 premise 前停止。

## Frozen test premise（允许 inform development）

在 HateMM 与 HateClipSeg 上先完成同一个 frozen diagnostic；读取 test predictions 与 test GT，因此全部结果标记为 developmental test evidence。

1. **Policy-alignment gate**：在含 hostile proposition 的 eligible seconds/videos 中，GT 必须对 `endorse` 与 `quote/report/deny/condemn` 呈一致的方向差异。若数据集把被引用的表面 hate 也标 positive，机制与任务定义冲突，立即停止。
2. **Coverage gate**：两语料都必须有足够比例的 positive videos 形成完整 source-scope-predicate-target path；不得因 HCS 缺 ASR 后改成 corpus-specific unary visual branch。
3. **Ordering gate**：固定完整路径 posterior 在两语料的 test within-video ROC 都必须胜过删除 source/scope edge 的 hostile-target scalar control；至少一边提升 `>= .020`，另一边方向为正。三项 pooled/within 固定指标同时报告，但该 gate 不替代最终 SOTA gate。

任一 gate 失败：归档候选，不训练 student，不改 parser prompt，不按语料选择 branch。

## 若 premise 通过后的最小正式方法

以 MultiHateLoc 为共享 starting architecture，四语料各自独立训练。只在 HMM/HCS 先做一轮：validation 仅在固定 arm 内选择 checkpoint，随后立即在 test 上输出 pooled AP、pooled ROC、within-video macro ROC。core 与 source/scope-deleted、edge-shuffled、capacity-matched scalar controls 使用相同特征、proposal grid、训练预算和 evaluator。

最终晋级要求两语料全部三个指标超过当前 SOTA gate，且 source/scope mechanism control 成立；否则淘汰。
