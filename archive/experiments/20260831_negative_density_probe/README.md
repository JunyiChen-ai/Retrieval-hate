# 淘汰：Same-corpus negative-density content probe

淘汰原因：HateMM audio 有内容排序信号，但 HateClipSeg 最好 visual transport 只比
POWA within 高 `.01093`，未过预设 `+.020`，不能形成跨两语料机制。

截至 2026-08-31。validation-only diagnostic，非候选方法，未读 test、未选
checkpoint。权威输出：`runs/20260831_negative_density_probe/{hatemm,hateclipseg}/analysis.json`。

仅使用各目标语料自己的 train video labels。对齐后的 audio / visual / transcript
frame features 先逐行归一化；负视频帧作为 reliable benign，正视频帧作为 noisy
positive mixture，训练一个 class/video-balanced linear density-ratio probe。验证同时报告
probe 自身排序，以及按照 probe 排序重新分配同一视频原 POWA score multiset 的
transport；后者必须逐视频保持 score multiset 和跨视频绝对尺度。它不保证 pooled
frame AP/ROC 不变，因为分数在视频内换位仍会改变 frame-label pairing，因此 pooled
feasibility 仍需实测。

这个 probe 本身不具备 novelty，只用于决定下一候选是否值得做
negative-certified local evidence。HateMM 与 HateClipSeg 任一语料 within 不超过
POWA `+.020`，或收益被 fixed-position control 解释，就停止该方向。

结果：HateMM audio transport 为 AP/ROC/within `.76689/.87740/.62774`
（POWA `.75766/.87442/.57193`）；HCS visual transport 为
`.51318/.60709/.53800`（POWA `.50639/.59854/.52707`）。逐视频 score multiset
误差均为 `0`。HMM center-position control `.76550`，因此即使 HMM 内容 probe 有效，
本数据上的位置捷径仍远强于它；HCS 则没有足够内容信号。按冻结规则终止。
