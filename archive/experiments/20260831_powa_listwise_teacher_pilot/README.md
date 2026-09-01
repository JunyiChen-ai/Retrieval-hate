# REJECTED — POWA within-video listwise MLLM teacher pilot

截至 2026-08-31。状态：HateMM validation smoke 触发 kill gate；未跑 HCS、
未进入完整 teacher、蒸馏或 candidate test。唯一 student starting point 是
corpus-specific POWA。

## 单一机制变化

上一轮 Qwen2.5-VL 对每个 16 秒 window 独立给 0–10 分，HateMM/MHC-EN
within 都未过 `.60`。本轮不把模型升级本身作为方法，而改为同视频 listwise
preference elicitation：Qwen3-VL-8B 每次同时看最多四个 window，直接按 hateful
evidence 排序；重放相反展示顺序并平均 Borda score，控制位置偏差。teacher 只
提供视频内顺序，绝对 score 不蒸馏；最终 student 仍是便宜的 POWA localizer。

## Smoke 与 gate

先取 HateMM/HateClipSeg validation 按 video id 排序的前 8 个正例视频。相同模型、
相同 16 秒/8 秒 stride、相同 frames+ASR 同时跑：

1. pointwise 0–10 control；
2. listwise ranking，连续四窗、步长三窗，每组按正序/逆序各询问一次。

Smoke 只筛查明显无信号实现，不做最终性能结论。若任一 corpus 的 listwise within
不高于 pointwise，或解析/顺序一致性明显失败，直接淘汰。若两者均为正，再冻结
完整 validation gate、做独立 novelty review，之后才扩展。

## Novelty 边界

VLP→WTAL distillation、pair/listwise ranking distillation、MLLM temporal evidence
以及 LELA/TANDEM 都已有先例。不能 claim ranking、distillation 或 MLLM teacher
本身。当前初查未发现“同视频多个 hate windows 的 comparative elicitation →
order-only POWA distillation”这一完整机制；但这是窄差异，必须实证证明 listwise
优于同模型 pointwise，且最终优于无 teacher/shuffled teacher controls。

Closest primary sources:

- Ju et al., CVPR 2023, VLP collaboration for WTAL:
  https://openaccess.thecvf.com/content/CVPR2023/html/Ju_Distilling_Vision-Language_Pre-Training_To_Collaborate_With_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html
- Liang et al., CVPR 2024, pair/listwise ranking distillation for VideoQA:
  https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Ranking_Distillation_for_Open-Ended_Video_Question_Answering_with_Insufficient_Labels_CVPR_2024_paper.html
- LELA, arXiv:2602.09637; TANDEM, arXiv:2601.11178.

输出：`runs/20260831_powa_listwise_teacher_pilot/smoke/`。

## 结果与裁定

首次 2-video smoke 的 ranking prompt 错把 `A>B>C>D` 写成格式示例，导致模型
复读展示顺序；该实现结果作废。去除顺序泄漏后，在同两个 HateMM validation
正例视频上，Qwen3 pointwise within ROC 为 `.848968`，listwise 为 `.438291`。
12/12 ranking calls 均可解析，但同一四窗集合正序/逆序呈现时判断不一致；Borda
平均后内容顺序信号接近消失。

按 smoke gate，listwise 没有高于 pointwise，机制淘汰，不再扩大到 HCS。可保留
的独立观察是 Qwen3 pointwise 在这两个视频上很强；它只能触发对原 dense-teacher
signal 的完整 qualification，不能把 backbone upgrade 包装成 listwise 方法。
