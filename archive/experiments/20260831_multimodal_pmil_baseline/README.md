# Multimodal P-MIL baseline port

> 归档原因：正式两语料 baseline pilot 完成且结果链独立审计 PASS，但 performance/SOTA FAIL。HateMM
> 三指标接近起点但均未过 SOTA；HateClipSeg 三指标明显失败。该 port 只作为相邻任务 baseline 与后续
> developmental error evidence，不是 novelty candidate。

截至 2026-08-31；状态：formal baseline pilot 与 test error analysis 均完成并经独立审计，已归档。

## 正式 test 结果与结论

权威 evaluator 输出：
`runs/20260831_multimodal_pmil_baseline/pilot_seed234/{hatemm,hateclipseg}/metrics.json`。

| corpus | pooled AP | pooled ROC | within ROC | SOTA gate |
|---|---:|---:|---:|---|
| HateMM | .589256 | .814221 | .589898 | 三项均 FAIL |
| HateClipSeg | .460507 | .423070 | .476610 | 三项均 FAIL |

HateMM 选 epoch 15，HCS 选 epoch 5；selection只用 official validation video AP。正式 scores 精确覆盖
HMM 214 videos/29,269 seconds、HCS 79/18,839，逐视频长度、finite、`[0,1]` 与共享 evaluator复算均
通过。完整性审查见 `POST_RUN_REVIEW.md`。

冻结 checkpoint 的 developmental test error analysis：
`runs/20260831_multimodal_pmil_baseline/pilot_seed234/test_error_analysis.json`。其删组件、单模态与
proposal oracle arms 都不是重训方法或 SOTA entries，只用于定位失败：

- HMM proposal oracle within `.73952`，但 full P-MIL `.58990`；同 checkpoint去掉 completeness 的
  diagnostic为 `.64966`，CAS-only为 `.66263`。PCE completeness提高部分 pooled尺度，却破坏视频内排序。
- HCS proposal oracle仍有 `.63450`，所以 proposal set不是硬上限；full仅 `.47661`，且 whole-video是
  top proposal的比例 `48/79=.60759`，top proposal中位长度 213 秒。
- HCS eligible videos中，visual-text/audio-text frame-rank Spearman分别有 `53/67` 个因常数序列无定义；
  all-pair IRC/PCE把无局部证据的 modality当 teacher，形成常数/whole-video shortcut。HMM对应 whole-top
  仅 `3/214=.01402`。
- GT-event proposal max-IoU≥.5 recall为 HMM `.76243`、HCS `.45385`；HCS boundary quality较弱，但 oracle
  within仍过当前 SOTA，主要失败仍是 proposal scoring而非完全没有候选。

后续不得按语料选择 HMM audio、CAS-only或移除 completeness；这些没有独立重训且是看过 test 后的
diagnostic。可用于新设计的共同结论只有：无证据 modality不应被强制做全 pair teacher，missing boundary
context也不能通过 zero padding把 whole-video proposal变成高 completeness shortcut。

这是相邻任务 starting-point baseline，不是 novelty candidate。目的只有一个：确认 P-MIL（CVPR 2023）
的 proposal-level train/test consistency 在弱监督 hateful localization 上是否有可用的双语料 test 基础。
任何正结果都只能称 P-MIL 的跨任务 baseline performance；在完成 test error analysis并提出额外、
non-trivial、尚未用于 hateful task 的机制前，不构成论文方法。

## 与官方 P-MIL 对齐的 load-bearing 结构

官方源码与论文：
https://openaccess.thecvf.com/content/CVPR2023/html/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html

1. 先用 frozen S-MIL 产生 train/validation/test candidate proposals。本 port 使用各目标语料独立训练的
   MultiHateLoc seed-234 checkpoint；它只做 blind frame scoring，不读取当前 split 的 temporal GT。
   从 fused score 的 9 个相对阈值连通域与 top-16 peaks 的固定邻域产生最多 256 proposals，并保证
   whole-video proposal用于 dense frame coverage。
2. 对每个 proposal 向两侧各扩 25%，RoIAlign 到 12 bins；每个 modality独立计算
   `[inside-left, inside, inside-right]` surrounding-contrast descriptor。
3. Visual/audio/text 三个 proposal branches各自输出 2-class CAS、attention与completeness。官方的两 view
   RGB/flow 扩为本任务的三个原生 modalities；不是把三模态先拼成一个 view。
4. 原 proposal MIL 与 attention-suppressed proposal MIL均保留。Positive bag的原 CAS目标含 foreground与
   background，suppressed CAS只预测 foreground；negative bag两者都预测 background。
5. PCE 保留：从其他 modalities 的平均 attention以 NMS产生 pseudo instances，所有 proposals 到 pseudo
   instances的最大 temporal IoU监督 positive-bag completeness head；真实 negative bag 没有 foreground
   pseudo instance，其 completeness 直接回归 0。沿用官方 `gamma=.8`、`lambda_comp=20`。
6. IRC 保留：positive bags内，每个有重叠 proposal neighborhood的 hate-CAS rank在所有有序 modality
   pairs间做 teacher-student KL；teacher stop-gradient，沿用官方 `lambda_IRC=2`。
7. test proposal score是三模态 `hate probability × attention × completeness` 的平均。1fps frame score
   是覆盖该秒 proposals 的最大 score；whole-video proposal保证 full coverage，constant proposal score
   必须产生严格平坦的 frame score。

## 有意的任务适配与不能 claim 的内容

- action classes改为 binary hate/background；加入真实 normal bags，因此 negative bags需显式训练
  background，官方 action-only训练集没有这一监督结构。
- RGB/flow两 views改为 visual/audio/text三 views，IRC/PCE做全 pair/leave-one-modality-out扩展。
- 官方 proposals 来自外部 S-MIL；这里固定为当前最相关的 corpus-specific MultiHateLoc。
- 官方输出 action detections；这里用同一 proposal confidence生成 dense 1fps score并交给全仓库唯一
  evaluator。

因此它是 **multimodal binary P-MIL port**，不是官方结果的逐字复现，也不 claim 新 proposal MIL、
completeness、IRC、SCFE、三模态学习或 hateful-localization novelty。

## 冻结 pilot

- corpora：HateMM、HateClipSeg；seed 234；各自完全独立训练。
- optimizer沿用官方 Adam `5e-5` 与 bag batch size 10；不同长度 proposal bags以顺序
  forward/backward累积平均梯度实现同一 optimizer batch，不把视频裁成等长。
- 完整 official train manifest 用于训练；冻结 official validation manifest只以 video AP 选择 15
  epochs中的 checkpoint。所有 epochs均不读取 validation localization GT。
- checkpoint 选定后立即 blind infer完整 evaluator test cohort，再调用唯一共享 evaluator输出 pooled AP、
  pooled ROC与 within-video macro ROC。
- producer只调用 scoped train/validation video labels；test labels/temporal GT只由共享 evaluator在
  scores关闭后读取。
- 输出：`runs/20260831_multimodal_pmil_baseline/pilot_seed234/{corpus}/`；包含 config、proposal diagnostics、
  checkpoint、train log、blind scores、evaluator metrics。root含 detached `run.log` 与 `run.pid`。

这个 baseline 不设 validation performance gate。两语料都必须完成 test 三指标才算 baseline port完成。
完成后用 test predictions+GT做 developmental error analysis，重点检查：proposal recall、whole-video top
proposal比例、PCE/IRC是否改善真实 within rank、三模态 rank consistency是否错误压制 unimodal hate。

若 baseline本身在任一语料出现全部三指标明显低于 MultiHateLoc，仍记录并归档；不得通过换 proposal
threshold、训练 epoch、frame readout或按语料选 view来制造后验提升。
