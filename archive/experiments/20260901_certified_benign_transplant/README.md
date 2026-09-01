# Certified benign temporal transplant MIL

> 已淘汰（2026-09-01）：novelty `STOP 1.8/10`；与本项目已完成双 test 的 `powa_benign_insertion_pilot` 在核心机制上逐项同构，未实现、未训练。

截至 2026-09-01。RESET6 提案；在 novelty 门停止，未实现或训练，不计正式 performance failure。

## Failure 与可用 correction signal

HMM 当前 POWA matched anchor 的 pooled AP/ROC/within 为
`.584460/.804897/.596995`，距离固定门仍缺 `.009372/.011287/.034537`；HCS 为
`.575832/.545819/.516630`，仍缺 `.043539/.059203/.045278`。上一轮把 lexical-supported
positive regions写入跨视频memory后，HMM负帧分数升幅反而大于正帧且视频内波动被压缩；HCS虽
小幅改善，方向不跨语料一致。该family已关闭。

本候选不使用 test GT、当前模型 top-K/confidence、lexical posterior或teacher。实际可用信号是
同语料 negative train bag 的所有有效秒均为 certified benign。弱监督的关键缺口是 positive bag
内哪些秒是 benign 未知；negative donor片段可在不伪造 hateful pseudo-label 的前提下提供精确
局部负监督。

## 跨任务来源与 task adaptation

候选来源为 CutPaste / copy-paste self-supervision：通过局部内容移植构造带精确空间mask的合成
训练样本。本 adaptation 反转其“粘贴异常”语义：从 negative train video取一个连续且时间对齐的
audio/visual/text feature block，移植到 positive train video 的随机位置，构成已知 benign temporal
counterfactual。原始未改视频继续走完整 POWA bag loss；增强副本不继承新的bag标签，只在移植mask
施加 benign frame loss，并在未移植区约束 raw frame score 与原始路径一致。所有模态共同替换，
mask边界不计监督，避免单模态缺失或拼接边界成为标签捷径。

这不是把 positive bag 全部当正例，也不是普通 mixup/ensemble/KD。新增机制是利用 negative-bag
certificate在 positive context 中生成 exact local benign supervision，直接训练同一个 POWA
frame head压低可迁移的 benign false positives。

## 指标路径、control 与可证伪预期

- pooled AP/ROC：跨视频可迁移的 benign donor被同一 frame head压低，目标是降低 positive bag
  背景及 negative frame false positives。
- within ROC：positive context内的 benign counterfactual获得局部负监督，而未移植区域保持原
  score与MIL竞争，扩大 hateful候选与背景的排序间隔。
- inference只运行一次原POWA forward并输出raw `frame_prob`；无移植、teacher、calibration、
  routing或多模型分数组合。

Matched control使用相同模型、随机mask、替换比例、双forward、一致性项和损失预算，但 donor来自
另一 positive train bag；它没有 certified-benign语义。Core必须在HMM/HCS test三项相对matched
anchor方向一致，且两语料within均胜positive-donor control；最终仍要求六项全部超过固定SOTA门。

若novelty通过，每个语料独立跑2个learning rate `{1e-4,2e-4}` × 3个transplant loss weight
`{.1,.3,.5}` × 2个替换比例 `{.10,.25}`，共12个core trial，另跑2个matched POWA anchor。
每个trial使用official POWA完整5 epochs。Validation在同learning-rate anchor pooled AP/ROC
不低于`-.005`的配置中，以within、AP、ROC联合选择配置与checkpoint；两个语料都锁定后，才训练
selected-config positive-donor control并立即跑HMM/HCS test。无smoke。

## Novelty 边界

必须由独立reviewer检索CutPaste/copy-paste或等价的certified benign temporal transplantation是否
已用于 hateful-video detection/localization，并判断“negative-bag certificate + positive-context
temporal transplant + transplant-only benign supervision + untouched consistency”是否构成non-trivial
task adaptation。若只是目标任务已有copy-paste或普通组件拼接，则STOP。
