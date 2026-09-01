# Counterfactual Carrier-Alignment MIL

> **淘汰（2026-08-31）**：独立 novelty verdict `STOP 5.7/10`。Gate 1
> PASS、Gate 2 窄 PASS、Gate 3 FAIL。解析反例可令 `b_t=c_v`、
> `r_t=g_v*a_t`：全局 topic 产生正视频指示 `g_v`，ASR/text 或 speech presence
> 产生 lexical 位置 `a_t`；正视频 aligned 胜 shifted、负视频因 `g_v=0` 保持
> shift-invariant，但真实 within 仍可为 `.5`。只 shift lexical、不 shift speech
> mask 还会改变有效 score multiset，产生纯 speech-presence shortcut。当前方案只是
> lexical temporal shuffle + correspondence loss，未识别 hate-specific multimodal
> alignment。未实现、未训练、未生成 prediction；不按建议继续修补同一候选。

截至 2026-08-31。Process epoch candidate `1/3`。本目录先做 Rule 12 novelty 三门，
三门未全过则不实现、不训练。

## 已有最低证据

同语料 train-video-label lexical locality 在 developmental test 上相对 matched
circular time-shuffle 的 within ROC margin 为 HMM `+.127533`、HCS `+.021501`。
因此两个语料都存在方向一致的 local lexical timing signal，且它不是纯 video
broadcast 或固定 position。HCS 信号较弱、无 speech 秒为常数，这些是覆盖诊断，
不再被错误设为 raw-statistic 前置硬门。

## 拟适配的跨任务来源

核心来源是 Zhang et al., *Action Shuffling for Weakly Supervised Temporal
Localization*（arXiv:2105.04208）：在只有 video label 的 temporal action
localization 中用 intra/inter-action shuffling 构造自监督时序约束。初步检索未发现
ActShufNet 或其 action-shuffling objective 被用于 hateful-video detection/localization；
正式 occupation verdict 由独立 novelty reviewer 给出。

## Hate-specific adaptation

Starting architecture 固定为 MultiHateLoc。每个语料独立训练。训练集 lexical score
由五折 OOF whole-transcript classifier产生；test lexical classifier只用完整 train。
Lexical score与speech mask是输入观测，不是frame label或teacher prediction。

模型在 MultiHateLoc fused local logit `b_t` 之外学习 content-dependent reliability
`r_t`，最终唯一 frame logit为：

`s_t = b_t + r_t * speech_t * (l_t - mean_speech(l))`。

对每个视频固定生成若干非零 circular shifts `pi(l)`，只移动 lexical 时间轴，保持
视频、label、全部词与其分数multiset、视觉/音频特征、长度和position grid不变。

- positive bag：要求 aligned bag evidence 高于各 shifted counterfactual 的平均值；
- negative bag：要求 aligned 与 shifted evidence 相同，因为任何 lexical/frame
  coincidence 都不应构成 hate；
- 同时保留原 video-label MIL，test只输出 aligned raw `s_t`，不做ensemble、CDF、
  calibration或routing。

这不是普通 lexical concat：若 `b_t` broadcast 且 `r_t` 为常数，permutation保持
`{l_t}` multiset，aligned与shifted的bag evidence严格相同，positive contrast无法满足；
若 `r_t=0` 同样无法满足。模型必须学习“hate-directed lexical cue 与同秒多模态内容
共同出现”这一交互。负视频invariance阻止generic profanity/topic coincidence被当作
充分证据。

## 与既有失败链的边界

- 不提供 lexical pair target，不做rank/KD/pseudo-label；
- 不把 lexical score blend 到已有test score；它进入单一模型并由matched temporal
  intervention决定是否load-bearing；
- 不用graph、Hodge、reference、CDF或teacher ensemble；
- 与失败的 lexical posterior regularization 不同：不投影latent posterior、不强迫
  high/low lexical集合成为目标，而是比较同一观测在正确/错误时间对应下对bag evidence
  的因果差值；
- 与generic AV-sync不同：被移动的是已由同语料train labels确定方向的hate-directed
  lexical carrier，不是任意声画同步事件。

## 可证伪预期与最小方法实验

Novelty 三门全过后才实现。一次 technical review 后，HMM/HCS 各自独立训练，validation
只在固定方法内部选配置/checkpoint，随后立即在test报告pooled AP、pooled ROC、within
ROC。Core须在两语料within均胜matched MultiHateLoc，至少一边`>=+.020`；最终晋级仍
要求四语料全部三指标SOTA。

方法test后再做机制controls：训练时用fixed shifted lexical替代aligned；inference
gate-off；position-template lexical；按ASR/OCR/visual carrier strata只作诊断。若core
增益在shifted或gate-off仍存在，alignment mechanism失败。
