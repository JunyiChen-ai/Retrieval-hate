# Ambiguous-Point Completeness MIL

> **淘汰（2026-08-31）**：独立 novelty verdict `STOP 5.4/10`。Gate 1
> PASS、Gate 2窄PASS、Gate 3 FAIL。LACP的关键识别条件是每个action instance内
> 有可靠人工point；OOF lexical plateau不保证位于hate span、覆盖每个事件或覆盖
> silent/static hate。Soft marginalization随score scale会塌缩到最容易的错误seed；
> 为同时包含错误seed与真实hate又会扩张成近whole-video interval。`s_t=c_v+epsilon*a_t`
> 可由video topic完成bag分类、微小错误lexical峰满足outer-inner contrast，而within
> 仍可`.5`。当前方案退化为SAR-PU soft selector + lexical seed propagation + 标准
> WTAL completeness。未实现、未训练、未生成prediction；不修补该候选。

截至 2026-08-31。Process epoch candidate `2/3`。先做 Rule 12 novelty 三门；未全过
则不实现、不训练。

## 跨任务来源

拟适配 Lee and Byun, *Learning Action Completeness From Points for Weakly-Supervised
Temporal Action Localization*（ICCV 2021）。来源方法用人工单帧 action point与
pseudo-background point搜索完整 action sequence，并以action/background score和feature
contrast学习 completeness。初步检索未发现该 point-completeness核心用于
hateful-video detection/localization；独立 reviewer负责正式 occupation verdict。

## 为什么不是直接套用

本任务没有人工 point，且 hate span可由speech、OCR、visual/context异步组成；把lexical
最大秒当clean point会重复已失败的SAR-PU/pseudo-label链。现有最低证据只说明OOF lexical
timing在HMM/HCS均有弱的同向局部信息，不保证任何单秒为真阳性。

本 adaptation 把每个positive-train video中speech-supported、OOF lexical高分的连续
plateaus保留为**ambiguous candidate-point sets**，不选top-1、不作frame target。Student
是MultiHateLoc local scorer。对每个candidate set与所有包含它、但不覆盖整段视频的
interval，使用来源方法的outer-inner score/feature completeness；以soft latent
marginalization联合选择“哪个候选点可信、哪个interval完整”。Negative videos的全部秒是
exact background；positive video的lexical-low plateaus只作为ambiguous background sets，
不当clean negative。最终训练仍有video-label MIL，test只输出一个student raw frame
score；lexical point sets只在训练使用。

## Hate-specific机制故事

标准MIL只取少数峰值，面对HMM长段hate与HCS跨模态/静态段会学到局部token或整段topic。
Lexical plateaus提供弱时间锚，但它们往往只覆盖一句话，不能代表完整hate interval。
Ambiguous-point marginalization避免把错误top-1硬编码成GT；outer-inner completeness要求
模型从候选锚向相邻visual/audio/OCR context扩展，同时排除与外侧background不可分的
fragment或whole-video proposal。

Constant/broadcast score使任意interval的inner与outer score/feature contrast为零，不能满足
completeness；固定position spike也不能系统性跟随每个视频不同的candidate set。与lexical
posterior regularization/rank KD不同，本方法没有dense lexical target或pair order，lexical
只定义一个被边缘化的partial point-supervision support。

## 可证伪实验

Novelty三门通过后实现最小HMM/HCS end-to-end方法；各自独立训练，validation只选择固定
方法配置/checkpoint，一次technical review后立即test三个指标。Core within须两语料都胜
matched MultiHateLoc、至少一边`>=+.020`；最终晋级仍须四语料全部三指标SOTA。

方法test后做：candidate-set circular shift、hard top-1 point、无outer-inner completeness、
允许whole-video interval四个controls；carrier strata只诊断。若shifted set或无completeness
不劣于core，机制失败。
