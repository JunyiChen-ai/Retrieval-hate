# 三模块方法计划（2026-09-03 立，用户定义，冻结）

依据：`experiments/20260902_verdict_boundary_contrast_mil/` 修订 4 已在 HateMM / HateClipSeg 两语料通过规则 8 三 seed 确认（README 第 8 节；STATUS 截至 2026-09-02 22:58）。用户裁定：性能已到 SOTA，但方法叙述不成范式，需按三个模块分别做出 novelty。

## 用户对三个模块的定义（原文归档）

> 我大概的构思是：第一个模块是 VLM 模块，第二个是骨干网络模块，第三个是融合模块。最好这三个模块都要做出它的 Novelty。
>
> 1. 第一个模块：现在的样子要不要继续改进？Novelty 是可选的。
> 2. 第二个模块：必选。要在骨干网络上进一步做出 Novelty。不一定非要用 Addition 的方法，也可以去改它当前实现中不好的地方，以此做出 Novelty。
> 3. 第三个模块：必选。要找到一个更 fancy 的理论去包装这种融合方式。
>
> 目标：性能提升多少不重要，但要有可观察的提升，且是由于我们的设计所导致的。
> 最后的过门：方法有性能提升，并且做了方法里所有必要的消融。当然 module 1 可以进一步的优化是最好的。

用户对现状的三点判断（同日）：(1) 很难 claim 是 novel paradigm，论文必须 claim 一个 paradigm；(2) 每个模块都要与其来源模块有区别，骨干直接复用不行；(3) 裁定先验本质是 VLM 与骨干分数的融合，目前像 engineering trick，没有理论或算法包装。

## 各模块现状（修订 4，作为改动起点）

| 模块 | 现状 | 来源 | 已有证据 |
|---|---|---|---|
| 1 VLM 裁定 | 冻结 Qwen2.5-VL-7B-Instruct，零样本；视频等分 K=30 与 K=4 窗，每窗 4 帧 + 该窗 Whisper ASR，孤立打 0–3 | LELA 类 training-free 打分 | 裁定本身 test：HateMM K30/K4/均值 AP .397/.457/.500、ROC .683/.782/.801；HateClipSeg .610/.576/.630、.616/.585/.633（`runs/20260902_verdict_boundary_contrast_mil/verdict_only_gran/`） |
| 2 骨干 | MACIL-SD 原样（I3D 五 crop + VGGish；BERT 句向量与裁定/位置拼入音频流） | MACIL-SD ACM MM 2022 | 骨干只受视频级 BCE；裁定只拼输入时骨干不用它（修订 1 相关 .003）；位置单独 within HateMM .725 / HateClipSeg .618 而模型 .640 / .549 |
| 3 融合 | z̃_t = z_t + w·S_t + b，初始化等于 α·(两粒度平均等级/3 − ½)，α 搜索；训练后 w、b 不离初始化 | Tip-Adapter / AMU-Tuning 的 logit bias | 去掉整个裁定通道 AP HateMM −.105、HateClipSeg −.078；先验相对只拼输入 HateMM +.012（噪声内）、HateClipSeg +.056 |

不进主张、建议删除：SniCo 边界对比（最优 checkpoint 在其开启前）、"可学习"先验措辞、位置通道（训练后权重 .02–.04；HateMM 上因 within 下限被剪掉，HateClipSeg 上 +.01 在噪声内）。

## 流程约束

- 规则 9 的三轮修改已用完，以上改动构成新候选：规则 4 一次独立 novelty 复核 → 实现 → 规则 6 一次 code review → 两语料 seed 234 各 20 trial 搜索 → 规则 8 三 seed 确认 → 规则 14 清单。
- 每个进主张的模块必须有 seed 234 消融显示去掉后 pooled 下降（规则 14(g)）。
- 规则 13：两语料同一架构、损失、流程，只允许标量超参数不同。

## 计划（按验证成本从低到高）

1. 模块 3：先离线验证理论是否与已有搜索结果一致（不训练）；成立后替换搜索出的 α。
2. 模块 2：改 MACIL-SD 在本任务上明确失效的部分，每一处改动对应一条已有证据，单独消融。
3. 模块 1：可选；若做，先只看裁定本身在 test 的三项是否高于现值，再进训练。

各模块的具体设计、结果与去向记在新实验目录 `experiments/20260903_<slug>/README.md`，本文件不再改写。
