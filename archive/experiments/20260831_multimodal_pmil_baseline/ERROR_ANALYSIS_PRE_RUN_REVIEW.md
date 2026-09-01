# Frozen test error analysis pre-run review

截至 2026-08-31。审查对象：`analyze_test_errors.py`、新增 synthetic
`test_error_analysis.py`，以及其直接调用的 frozen P-MIL run、proposal producer和共享 evaluator。

## 裁定

**PASS，可以在两个正式 baseline runs训练、blind prediction和共享 evaluation全部成功后，运行 frozen
post-training test error analysis。** 本次只执行 synthetic tests和静态检查，没有运行正式 test
analysis，也没有读取 test GT。

本脚本生成的是 iterative/developmental test evidence。只有 `pmil_full` 是已训练 baseline 的重算；
删组件 arms 是同 checkpoint inference diagnostics，`proposal_oracle` 是显式使用 GT 的 proposal-set
upper bound。它们均不是独立训练的方法，不进入 SOTA 表，不能作为 routing、calibration或新方法结果。

## Review 中修复的结论级问题

1. **缺少 frozen prediction 一致性门。** 原脚本虽加载 `model.pt`，但没有证明重算 `pmil_full` 就是正式
   `scores.jsonl` 所用模型和 proposal path。现要求 config/train log/model/scores/evaluator metrics全部存在，
   检查 selected epoch、corpus/test scope和 exact cohort，并逐视频要求重算 full frame score与 frozen
   `score_pmil` 在 `1e-6` 内一致。不一致立即停止分析。
2. **top proposal length受 tie顺序影响。** 原实现用 `argmax`，并列最高时由 lexicographic proposal顺序
   任意决定长度。现把 `1e-12` 内并列最高的 proposals视为同一 top set；whole-top按是否属于该集合定义，
   每视频 top length取 tied lengths的中位数，并另外报告发生多重 top tie的视频比例。
3. **Spearman静默丢弃 undefined videos。** 原实现只保留 finite值，不报告 denominator，constant modality
   score可能使相关性均值产生选择偏差。现每个 modality pair同时报告 finite n与undefined n，并写明
   Spearman是在同时含正负 GT frames的视频上，对两个 per-modality full frame-score vectors计算。
4. **proposal recall聚合单位不明确。** 现输出名与定义明确为 event-macro：每个 contiguous positive GT
   interval先取 frozen proposals中的最大 temporal IoU，再让所有 GT intervals等权计算mean/recall。
5. **diagnostic arms身份不明确。** 输出新增 `arm_policy`，逐项标记 source diagnostic、same-checkpoint
   diagnostic与 GT-informed oracle，并明确它们不是 retrained method或 SOTA entry。

## 1. Frozen checkpoint 与 test information boundary

- 分析入口固定读取正式 run root下每语料的 `config.json`、`train_log.json`、`model.pt`、
  `scores.jsonl`和`metrics.json`。缺任何一个即拒绝；因此不能在训练未完成、checkpoint未选择或共享
  evaluator未完成时启动分析。
- P-MIL只加载 `run.py`保存的 selected `model.pt`。source proposal model路径必须与 config及固定
  corpus/seed checkpoint解析结果一致，结构仍由原 inference path加载。
- 所有模型设为 `eval()`，整个视频循环处于 `torch.no_grad()`。脚本没有 optimizer、loss、backward、
  checkpoint写入或 selection分支。
- test GT只由 `hdata.gt_arrays(corpus, "test")` 在完整 frozen run检查后加载，用于 diagnostic evaluator、
  proposal/GT overlap和确定 within-eligible videos。它不影响 source proposals、model parameters、frozen
  full scores或 checkpoint selection。
- 每视频在使用 GT前先检查 feature length与 GT length一致；analysis cohort必须与 evaluator-test GT
  keys完全一致。

结论：test GT的用途严格限定为用户允许的 post-training error analysis与共享 metric复算。

## 2. Score arms

- `source_smil`：frozen corpus-specific MultiHateLoc fused per-frame probability，正是 candidate proposal
  producer输入；不是重新训练的 source arm。
- `pmil_full`：每 modality独立计算
  `softmax(CAS)[hate] × sigmoid(attention) × sigmoid(completeness)`，先对三模态平均 proposal score，再以
  covering-proposal max投影到 frames。它与正式 `model.scores`/`proposal_to_frames`定义一致，并必须逐视频
  重现 frozen `score_pmil`。
- `pmil_without_completeness`：同 checkpoint，将每模态 score改为 `hate × attention`后平均；只诊断
  completeness在 inference product中的影响，不是 PCE训练消融。
- `pmil_hate_cas_only`：同 checkpoint，仅平均三模态 hate probability；不是 independently retrained
  CAS-only baseline。
- `pmil_{visual,audio,text}_full`：各 modality自身的 `hate × attention × completeness` frame max readout，
  用于检查三模态贡献与rank consistency，不进行 test routing或best-modality selection。
- `proposal_oracle`：每个 frozen proposal赋值为其对所有 contiguous GT intervals的最大 temporal IoU，
  再用相同 proposal-to-frame max readout。它是 proposal support的 GT-informed upper bound，不是模型输出。

所有非 source arms共享同一 frozen proposal集合、同一 selected checkpoint与同一 frame readout，因此差异
不会混入 proposal generator或checkpoint变化。共享 `evaluate_scores`对所有 arms统一输出三项固定指标；
这些 diagnostic metrics只用于error analysis。

## 3. Proposal/GT、whole/top length

- GT intervals通过在 1fps binary array两端补零后取 transitions生成，采用 `[start,end)`；proposal bounds
  同样是该半开区间约定。synthetic test验证partial overlap、相邻不重叠和完全相等的 IoU。
- proposal oracle与event recall都使用标准
  `intersection / (len(P)+len(G)-intersection)`。无 positive GT interval的视频oracle全零；正 GT events
  逐个记录best proposal IoU。若整个 corpus没有正 events则拒绝输出非有限summary。
- whole proposal必须恰好一个 `[0,T)`，否则停止。`whole_video_is_top_proposal_fraction`包括与其他proposal
  并列最高的情况；tie policy已显式记录。
- top length是每视频 top-score tied proposals长度中位数，再跨所有 test videos报告median/mean；同时报告
  top tie视频比例，避免把并列结果解释成唯一边界选择。

## 4. Pairwise modality Spearman

- 输入是每 modality full proposal score经相同 max readout后的 **frame score vector**，不是 CAS logits、
  proposal index或GT排序。
- 只在 GT同时含0/1的 test视频计算，与 within-video ranking的eligible cohort一致。每个 pair分别调用
  Spearman；无 ties时等价于rank Pearson，有 ties时由 SciPy使用average ranks。
- constant vector或其他 undefined情况不被写成0；其值不进入finite mean，但被计入
  `n_undefined`。每 pair的finite denominator单独输出，因此三组均值可审计。
- 该量衡量最终per-modality frame rank agreement，只能辅助解释IRC，不能单独证明IRC有效；严格IRC训练
  归因仍需要独立重训的无IRC control。

## 5. 共享 evaluator 与输出

- 脚本直接import仓库登记的唯一共享 `eval_baseline_scores.evaluate_scores`，没有复制 pooled AP、pooled
  ROC或within ROC实现。所有 arm score maps覆盖同一 test cohort；shape与finite检查由
  `proposal_to_frames`和共享 evaluator共同执行。
- `hate_ids`由frame GT中是否含positive构造。对within metric，只有同时含两类frames的视频实际进入AUC，
  与共享 evaluator的eligible semantics一致；pooled指标不使用该集合。
- 唯一输出写到正式 run root的 `test_error_analysis.json`，并记录 frozen model/prediction文件绝对路径、
  GT用途、developmental status与arm policy。没有写入 `data/`或实验源码目录。

## 6. 已执行检查

- 3项 synthetic tests全部通过：半开区间IoU、proposal-oracle/per-event best IoU、top-tie order invariance。
- `analyze_test_errors.py`与`test_error_analysis.py` compile通过。
- 静态检查确认分析脚本没有 training、optimizer、backward、validation selection或test-label loader。
- 未运行 `analyze_corpus`或`main`，因此本次review没有加载test GT或生成正式analysis artifact。

最终裁定：**PASS FOR FROZEN POST-TRAINING TEST ERROR ANALYSIS**。
