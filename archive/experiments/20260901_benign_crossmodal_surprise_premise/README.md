# Benign cross-modal surprise premise

截至 2026-09-01。**最低证据失败，停止并归档；不是方法迭代。** 权威结果：`runs/20260901_benign_crossmodal_surprise_premise/main/metrics.json`。

HMM aligned/shifted within ROC 为 `.500748/.514832`，HCS 为 `.500843/.524704`。两语料 aligned 都近 chance 且低于时间错位 control，joint gate 失败。该信息源不进入 novelty 或方法训练；Rule 13 正式方法失败计数保持 `2/3`。

## Question

Temporal Expert-Choice 的正式 test 表明固定 modality capacity 只保证负载，不能从 bag label 识别局部 competence。这里检查一个独立信息源：只用同语料 train negative seconds 学习正常情况下 visual/audio/text 的同步关系；test 每秒的跨模态 disagreement 是否在 HMM/HCS 都具有局部变化、是否与 hateful span 同向、以及是否优于破坏同步关系的 circular-shift control。

Producer 不读取 test GT。Test GT 只由统一 evaluator 和本 premise gate 读取。若两语料任一不满足 aligned within ROC `>.5`、aligned 高于 shifted control、且 eligible positive videos 的 median within-video score std `>0`，立即停止该信息源；不更换 projection、loss 或 anomaly score。

## Run

```bash
bash experiments/20260901_benign_crossmodal_surprise_premise/run.sh
```
