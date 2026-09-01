# Lexical-Gated Messenger Bottleneck

> **淘汰：独立 novelty verdict `STOP 5.6/10`。** Gate 1 PASS、Gate 2 窄 PASS、Gate 3 FAIL。模型可令 messenger 恒零并用 unimodal video 常量完成 bag 分类；时间去均值也保留 `video identity × zero-mean position basis`，再乘 lexical gate 后与已失败的 `global indicator × local lexical pattern` 同构。未实现、未训练、未生成 prediction。

截至 2026-08-31。Process epoch candidate `3/3`。状态：novelty 门停止。

## 已有最低证据

不新增 premise。直接使用已完成的同语料 lexical-locality test evidence：raw lexical 相对 matched circular time-shuffle 的 within-video ROC margin 在 HateMM/HateClipSeg 分别为 `+.127533/+.021501`。它只证明局部 lexical 变化存在、不是纯 position/broadcast 且两语料同向；不把 raw score当 frame teacher。

## 跨任务来源与 occupation 边界

来源为 Xu et al., *Rethink Cross-Modal Fusion in Weakly-Supervised Audio-Visual Video Parsing*（WACV 2024）的 messenger-guided mid-fusion：用低容量 messenger 代替 full cross-modal context，减少 modality-agnostic video label造成的跨模态噪声传播。初步检索未发现该方法用于 hateful-video detection/localization。Yang et al., *Revealing Temporal Label Noise in Multimodal Hateful Video Classification*（2025）分析了 coarse video label的 temporal noise，但没有提出本候选的 messenger fusion。

允许的 claim 不是 messenger、cross-attention、lexical feature或MIL本身新，只能是下面面向 hate localization 的受限传播机制。

## 任务机制

MultiHateLoc的 video-level positive label会同时监督 audio/visual/text，并允许一条 video-global transcript/topic shortcut经 full cross-modal interaction广播到所有秒。Hateful evidence常由局部攻击词、target提及或转述边界触发；因此本候选不预测 modality owner，而限制“什么内容可以跨模态传播、传播到何时”。

每个语料独立训练。Lexical observation由本语料 train video labels拟合：训练视频使用固定五折 OOF whole-transcript TF-IDF classifier，validation/test producer只用完整 train；窗口和超参数沿用已冻结 premise。它是输入观测，不是 span label或dense teacher。

对每个 modality 的 temporal state `h_tm`，先由 source-style low-rank messenger压缩跨模态上下文。与直接移植不同，text发送到 audio/visual 的 message必须同时满足：

1. message只由局部 residual `h_text,t - mean_valid(h_text)`生成，video-constant text component在代数上为零；
2. message乘以局部 lexical innovation `g_t = speech_t * (l_t - mean_speech(l))`，并经有界奇函数，正负变化都保留；
3. audio/visual到text以及audio↔visual仍使用相同容量的 residual messenger，不由 lexical score选择输出 branch。

原始 full cross-modal path删除。最终只有一个 fused frame head与一个video MIL loss；没有owner pseudo-label、deletion effect、teacher distillation、test routing、score blend或post-hoc calibration。Lexical值不能直接加到frame logit，只能调制跨模态 residual message。

该改造的解析性质是：若 text表示为整段常量 topic `h_text,t=c_v`，则其跨模态 message严格为零；若 lexical score只提供video identity或常数，`g_t=0`。因此最常见的 text-to-all-seconds broadcast不能通过该路径实现。局部 residual与lexical timing一致时才允许改变其他模态的同秒表示。

## 与旧失败链的边界

- 不重复 deletion-carrier/owner-abstention：不构造carrier/background/abstain state，不替换模态，不做pseudo ownership，也没有辅助SupCon可被最终head旁路。
- 不重复 counterfactual carrier-alignment：没有aligned-vs-shifted bag loss，也不声称同步本身证明hate；lexical只限制message通道。
- 不重复 context-quotient span marginal：不在frozen POWA score上加zero-mean residual、不枚举span、不使用anchor score。约束对象是端到端模型中的跨模态通信，而非最终logit的后处理残差。
- 不重复 ordinary lexical concat：lexical没有到输出的直接路径；gate-off后模型仍是capacity-matched residual messenger network。

剩余可证伪风险：模型仍可能在各 unimodal branch 内学习video-global shortcut，或完全忽略messenger。故本候选不声称解析解决全部weak-label identifiability，只声称阻断一个具体的跨模态广播路径；是否改善最终ranking必须由方法test与load-bearing controls决定。

## novelty gate 与最小正式方法运行

独立 reviewer 必须裁定：

1. source mechanism是否确未进入 hateful-video task；
2. constant-annihilating、lexical-innovation-gated messenger是否是non-trivial hate-specific adaptation，而非WACV方法加一个普通attention gate；
3. 是否与项目内context quotient、lexical alignment或modality ownership失败链机制等价。

只有三门全过才实现。实现后只做一次 technical review，随后 seed 234 独立训练 HateMM/HateClipSeg；validation只在各固定方法内部选择checkpoint，完成后立即在test跑 pooled AP、pooled ROC、within-video ROC。

固定 arms：MultiHateLoc anchor、capacity-matched ungated residual messenger、lexical-gated messenger core。方法test后才运行 time-shuffled lexical、gate-off与mean-repeated inference controls。Core须在两语料within均胜两个anchor，且至少一边 `>=+.020`；完整晋级仍要求固定四语料全部三指标严格SOTA。

来源：

- Xu, Hu, Lee, WACV 2024, *Rethink Cross-Modal Fusion in Weakly-Supervised Audio-Visual Video Parsing*.
- Yang et al., 2025, *Revealing Temporal Label Noise in Multimodal Hateful Video Classification*.
