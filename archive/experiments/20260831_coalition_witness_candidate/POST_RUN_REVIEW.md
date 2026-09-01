# Coalition witness pilot：独立 post-run 审计

日期：2026-08-31  
权威汇总：`runs/20260831_coalition_witness_candidate/pilot_seed234/metrics.json`  
审计结论：**PASS（结果链可信）；方法结论为 FAIL，不得扩到四语料。**

## 完整性

- HateMM 与 HateClipSeg 的 `no_infonce`、`all_subset_mil`、`synib`、
  `mobius_nonminimal`、`coalition_witness` 共 10 个新 arm 全部通过 formal arm validator。
- 每个 arm 的 checkpoint、scores、训练记录、配置、日志和 evaluator 输出均存在且可解析；
  corpus、arm、seed、modality order、冻结超参数与 pre-run plan 一致。
- HateMM 五个新 arm 均 exact cover 214 个 test 视频、29,269 个 1 fps 帧；HateClipSeg
  均 exact cover 79 个 test 视频、18,839 帧。没有 missing/extra video，所有 score shape 与 GT
  一致且为 finite。
- 对 10 个新 arm 的逐帧 scores 重新调用唯一共享 `evaluate_scores`；pooled AP、pooled ROC、
  within-video macro ROC、逐视频 AUC、视频数和帧数均与各 arm `metrics.json` 及最终 summary
  一致。两个既有 MultiHateLoc starting-point source 也独立复算一致。

## 训练、validation 与 test isolation

- 两个 corpus 使用各自 official train/validation/test manifests，独立 model、optimizer、
  checkpoint 和输出目录；没有跨语料训练。
- 每个新 arm 的 history 长度与冻结 epoch budget 一致。记录的 selected epoch 均是该 arm
  official validation video AP 的首个最大值。
- history 只含训练 loss 和本 arm checkpoint-selection AP；没有 test performance。test score
  在 checkpoint 选定后才写出，随后立即由共享 evaluator 在 `split=test` 上评测。
- `no_infonce` 的 producer 参数与各 corpus official starting-point 配置一致，唯一机制变化是
  contrastive weight 为零。所有新 arm 配置均明确记录 test labels 未用于梯度或 checkpoint
  selection。

## Reconstruction 与 posterior

- HateMM/HateClipSeg 的 `mobius_nonminimal` 和 `coalition_witness` 均记录 full-score
  reconstruction max absolute residual 为 `0.0`，通过冻结 `1e-5` 门。
- Candidate posterior diagnostics exact cover 全部 test cohort：HateMM 214、HateClipSeg 79。
  每个视频均有七项 finite、非负 posterior mass，和在容差内为 1；MAP subset/time 合法，
  atom summaries 为 finite。
- Test prediction 只有 `score_full`，来自同一 atoms 的 reconstruction；没有独立 fused branch、
  posterior routing、ensemble、calibration 或 transport。

## 冻结 gate 复算

| corpus | candidate within ROC | 对 controls 的结论 | mechanism gate | 三项 SOTA gate |
|---|---:|---|---|---|
| HateMM | 0.62708 | 高于 all-subset，但低于 SynIB、Möbius non-minimal 和 no-InfoNCE | FAIL | AP、ROC、within 全部 FAIL |
| HateClipSeg | 0.52346 | 低于 all-subset、SynIB、Möbius non-minimal 和 no-InfoNCE | FAIL | AP、ROC、within 全部 FAIL |

独立按冻结严格不等式重算后：

- `mechanism_pass_by_corpus = false/false`；
- `mechanism_pass_both = false`；
- `sota_pass_by_corpus = false/false`；
- `sota_pass_both = false`；
- `continue_to_four_corpora = false`。

因此结果不仅未过项目 SOTA 门，核心 `(time, coalition)` latent witness 也未胜过 matched
`mobius_nonminimal`：HateMM 为 `0.62708 < 0.63377`，HateClipSeg 为
`0.52346 < 0.53652`。现有 evidence 不支持把收益归因给单一 latent coalition witness；该方向
应按冻结规则停止，不得挑选 HMM 的某个 control 或修改 gate 后继续。

## 非阻断记录

`research-wiki/STATUS.md` 在本审计时仍写“正在后台运行”。结果文件已完整，指标和 verdict
不受影响；本轮收尾时应把 STATUS 更新为“pilot 完成且双门失败”。

