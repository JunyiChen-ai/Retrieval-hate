# 淘汰：negative-null conformal discovery probe

截至 2026-08-31。validation-only、零训练诊断；未读 test，未进入 novelty review。
权威输出：`runs/20260831_negative_conformal_probe/analysis.json`。

## 问题

能否把同语料 validation negative videos 的全部局部分数当 empirical null，对每个
positive video 内的 frame 做单侧 p-value，并用固定 BH `q=.10` 选出高置信 hateful
frames？这里只验证前提；正式方法若存在，null 必须改为 train-only cross-fitting。

## 裁定

`STOP_BEFORE_NOVELTY`。HateMM audio discoveries 的 macro precision `.905`，高于
eligible positive-video base rate `.697`，但 HCS 的同一 concat rule 只覆盖 2 个视频，
precision `.438`，低于 base rate `.450`；HCS audio/text 也不成立。visual 虽在 2 个
HCS 视频上 precision `.712`，coverage 仅 2/49，且按语料/validation GT 选 view 属
branch-selection anti-pattern。没有一个冻结 single-view rule 同时具备双语料 coverage
和 precision，因此不训练、不查新。
