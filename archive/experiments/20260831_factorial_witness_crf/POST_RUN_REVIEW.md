# Factorial witness CRF pilot：独立 post-run audit

日期：2026-08-31  
范围：`runs/20260831_factorial_witness_crf/pilot_seed234/` 下 HateMM/HateClipSeg ×
core/zero-transition/collapsed 六个 run、`test_error_analysis.json`，以及冻结的 train/predict/evaluate
结果链。  
结果链结论：**PASS。** 六个正式 run 的训练选择、test coverage、score alignment、共享评测输出和
error analysis 均可复算。Pilot 结论为 **FAIL**：三指标 SOTA gate 与双语料机制 gate 均未通过，
不得扩 MHC-EN/ZH。

## Config、checkpoint selection 与 test isolation

- 六个 config 都严格对应 frozen seed-234 pilot：40 epochs、batch 16、hidden 128、dropout `.1`、
  LR `3e-4`、weight decay `1e-4`，三个 arms 分别独立训练；corpus、arm 与输出路径均正确。
- 六份 train history 都包含连续 epoch 1–40，loss 与 validation video AP 全部 finite。所选 epoch
  分别为 HMM core/zero/collapsed `11/12/18`，HCS `39/30/21`；每个都是本 arm history 中最高
  validation video AP，且与 `train_log.json` 和 checkpoint metadata 一致。
- 训练只从同语料 train/validation scoped video labels 取监督。Validation video AP 只选择本次
  arm checkpoint；checkpoint 固定后，`predict.py` 才读取 test membership 与 features，并为 test
  loader 使用全零 placeholder labels。
- `predict.py` 不 import temporal GT 或 test labels；只有之后的 evaluator/error analysis 读取 test
  GT。六份 train logs 均声明 test 未参与梯度或 checkpoint selection，与实际调用链一致。

## Exact coverage、posterior contract 与共享 evaluator

- 每个 HMM arm 精确覆盖 frozen evaluator-test cohort 的 214 个唯一视频、29,269 帧；每个 HCS
  arm 覆盖 79 个唯一视频、18,839 帧。所有 JSONL 顺序与 manifest 一致，无 missing、extra、
  duplicate。
- 每行包含 `score_core`、`active_posterior` 和 `bit_posterior`。全部数组 finite、位于 `[0,1]`，
  前两项长度逐视频等于 GT；typed arms 的 bit shape 为 `[T,3]`，collapsed 为 `[T,1]`。Typed bit
  marginal 均不超过 union active posterior。
- 数值上每个视频都满足 `score_core = active_posterior × video_scale`，全六 arm 最大重建误差低于
  `5.6e-8`。因此同一视频内的 ROC 排序由 active posterior 决定，而 pooled 指标还受 video-level
  scale 影响。
- 对六份 `score_core` 在内存中逐一调用仓库唯一
  `eval_baseline_scores.evaluate_scores`；完整 reports 与六个 `metrics.json` 逐字段一致，coverage
  均为零缺失/零额外：

| corpus | arm | pooled AP | pooled ROC-AUC | within-video ROC-AUC | within n |
|---|---|---:|---:|---:|---:|
| HateMM | core | 0.4335202245925353 | 0.7144150833886491 | 0.6343663186652669 | 85 |
| HateMM | zero transition | 0.4742254195146277 | 0.7133427535606409 | 0.6345957858077447 | 85 |
| HateMM | collapsed | 0.4193404679174154 | 0.6902215252480425 | 0.6317192875698975 | 85 |
| HateClipSeg | core | 0.5744873410301343 | 0.5465529780408402 | 0.5210532098785206 | 67 |
| HateClipSeg | zero transition | 0.5712823490269529 | 0.5377350188313383 | 0.5201167057900489 | 67 |
| HateClipSeg | collapsed | 0.5808107725659467 | 0.5444739711693509 | 0.5177546678004639 | 67 |

## `test_error_analysis.json` 独立验证

从原始 scores/GT 独立重算 active metrics、每个 eligible positive video 的 AUC、core-control delta、
positive fraction、GT transition rate、bit statistics 和 video scale；重建 payload 与正式 JSON
逐字段一致。

- Active posterior 的 within ROC 与对应 `score_core` 完全相同，符合逐视频正标量不改变排序；
  HMM core/zero/collapsed 为 `.6343663/.6345958/.6317193`，HCS 为
  `.5210532/.5201167/.5177547`。去掉 video scale 后，HMM core pooled AP/ROC 为
  `.41403/.70322`，HCS 为 `.57414/.54531`。
- HMM core-minus-zero 的 mean within delta `-.0002295`，core 更好的视频比例 `.5647`；
  core-minus-collapsed mean `+.0026470`，但 core 更好比例仅 `.4941`。HCS 对应 mean delta 为
  `+.0009365/+.0032985`，core 更好比例 `.4925/.4478`。中位数也都接近零或为负，没有稳定的
  per-video mechanism gain。
- Delta 与 GT transition rate 的 Spearman 相关在 HMM 为 `-.0438/-.0257`，HCS 为
  `-.0786/-.0705`；没有证据表明 learned transition 特别改善真实边界更频繁的视频。与 positive
  fraction 的相关绝对值最高约 `.247`，也不足以建立稳定归因。
- Core bit posterior 的 frame-weighted audio/visual/text means：HMM
  `.4615/.4595/.4868`，HCS `.5117/.5095/.5197`；argmax fractions 分别为
  `.2416/.2664/.4920` 和 `.3436/.3062/.3502`。Top-bit exact ties 仅 HMM 3 帧、HCS 4 帧，
  deterministic argmax 不改变总体观察。Pairwise correlations 大多接近零（最大绝对值 HMM
  audio-text `.166`）；这些只是 latent marginals，不证明真实 modality ownership。
- Core video scale 对 test video label 的 ROC 为 HMM `.77155`、HCS `.63043`；positive/negative
  mean scale 分别为 HMM `.6852/.1921`、HCS `.9855/.8000`。这验证 HMM pooled score 明显携带
  video-level separation，而 HCS scale 更饱和；不能把 pooled 变化全归因于 temporal posterior。

## Frozen gates 与最终裁定

- HateMM core within `.6343663` 严格超过 within SOTA `.6315317`，但 pooled AP/ROC 均低于
  `.5938316/.8161838`，所以三指标 SOTA gate FAIL。Core within 高于 collapsed，却低于
  zero-transition `.6345958`，所以该语料机制 gate FAIL。
- HateClipSeg core AP/ROC/within 均低于固定 SOTA
  `.6193711/.6050225/.5619079`，所以 SOTA gate FAIL。Core within 同时高于 zero-transition 与
  collapsed，因此该语料机制 gate PASS，但冻结机制 gate 要求两个语料都通过。
- 综合结果：双语料三指标 SOTA gate **FAIL**；双语料 `core > both controls` within gate
  **FAIL**。HMM 的小幅 within-SOTA 不能覆盖 pooled 失败，也不能在 zero-transition 更高时证明
  learned transition load-bearing。
- `test_error_analysis.json` 是允许的 iterative/developmental test error analysis，只能用于解释
  本轮失败；test GT 没有回流到已完成训练或 checkpoint selection。

**最终 verdict：PASS（result-chain integrity）；pilot：FAIL。** 按冻结计划淘汰当前 factorial
witness CRF pilot，不扩 MHC-EN/ZH，也不围绕这些 test 数字调参。本审查未修改任何正式
prediction 或 metrics。
