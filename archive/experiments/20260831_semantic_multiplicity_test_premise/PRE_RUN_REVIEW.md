# Semantic multiplicity test premise：独立 pre-run review

日期：2026-08-31  
状态：正式 `main()` 运行前  
结论：**PASS，可以运行正式 diagnosis；只有 `premise_pass_both=true` 才能进入方法实现。**

## 输入、split 与共享 evaluation

- `mdata` 已改为显式 `from multihateloc import data as mdata`。运行时核实其实际文件是
  `scripts/reproduction_baselines/multihateloc/data.py`，不会因顶层 `data` 同名模块或
  `sys.path` 顺序误导入其他实现。
- 两个 score source 固定为上一轮 corpus-specific `mobius_nonminimal/score_full` 的正式
  test artifact。脚本现同时核 source config 的 corpus/arm/test-isolation 声明，以及 companion
  metrics 的 `split=test`、branch、score 路径、视频数和帧数。
- Score 必须与共享 evaluator 的 test GT exact cover，逐视频 score/GT shape 相同且 finite，GT
  只能是二值。视觉 feature 还必须逐视频严格为 `(T, 768)` 且 finite。
- Dry analysis 覆盖 HateMM 214 个 gold test 视频、29,269 秒和 HateClipSeg 79 个视频、18,839
  秒。HateMM manifest 有 215 个 ID，其中 `hate_video_427` 不在冻结 gold/evaluator cohort；脚本
  显式记录这一排除，并要求 GT 不得出现 manifest 外 ID。两语料 mixed-label positive cohort
  分别为 85 和 67 个视频。
- 项目固定三指标只调用唯一共享 `eval_baseline_scores.evaluate_scores`。本脚本的 FP/TP
  per-video AUC 仅是 error-analysis 统计，不是新增项目 performance 指标。

## 发现并修复的结论级问题

1. **只读 feature 被原地归一化，原实现首视频即失败。** `mmap_mode="r"` 返回只读数组，初版
   的 `/=` 会抛异常。修订版先创建私有 float32 copy，再做归一化；不修改共享 feature cache。
2. **初版 top-quarter 定义会因 score ties 排除整个 cutoff plateau。** 它先取 average rank，
   再用 `rank >= .75`；真实数据中 HateMM 甚至存在选中 0 秒的视频。修订版直接取
   `ceil(T/4)` 对应 score cutoff，并 tie-inclusive 纳入完整 plateau，同时记录 intended count、
   actual fraction 与 expansion。Dry analysis 识别出 HateMM 6 个、HateClipSeg 37 个 cutoff-tie
   expansion 视频。
3. **初版 gate 只看 pooled FP-density AUC。** Raw density 是非局部 kernel mass，其尺度可能受
   视频长度和视频间构成影响；只用 pooled 结果可把跨视频混淆当作帧内机制。修订版同时要求
   pooled 与 per-video macro AUC 高于 `.5` 且分别胜过 reverse control。
4. **初版 evidence share 与高分 support 不一致。** 初版在全视频上用任意
   `exp(5 * rank)` 权重，却把结果命名为 MIL evidence share。修订版改为同一 high-score
   support 内的 model score-mass share，并明确它只是 diagnosis，不是 producer 训练时 bag
   likelihood 的重建。

## 统计定义与方向

- Semantic density 使用 frozen MultiHateLoc visual ViT-B/16 1 fps feature。每行 L2
  normalization 后，累计与时间距离至少 5 秒的行之间
  `exp((cosine_similarity - 1) / 0.05)`，最后加 self mass 1。局部邻居、自身和 padding 均不会
  进入 nonlocal sum。
- FP/TP AUC 的 label 是 `1 - GT`，且只在 raw score 的 per-video high-score support 上计算；
  因而 AUC `> .5` 的方向确实表示 false-positive 秒具有更高 semantic density。
- Reverse control 只在同一个视频内反转 density 的时间顺序，保留 density 边际分布、视频
  长度和 raw high-score support。它是 position-association control，不单独证明因果性。
- Per-video macro 只平均同时含 FP 与 TP high-score 秒的视频；脚本记录其定义视频数。Pooled
  与 macro 必须同时过门，不能挑一个层级解释为 premise 成立。
- Inverse-density share 使用 high-score support 内的 raw score 作为质量，再除以 density；GT
  只用于读出其中 FP share。该数值是 test oracle diagnosis，不能变成 inference-time selector。

## 固定 gates 与 dry verdict

修订后每个 corpus 都必须通过五项硬门：pooled AUC 高于 `.5`、pooled 胜 reverse、macro AUC
高于 `.5`、macro 胜 reverse、inverse density 降低 FP high-score mass share。两个 corpus 的
全部门都通过，`premise_pass_both` 才为真；任一失败即
`STOP_BEFORE_METHOD_IMPLEMENTATION`。

不写正式输出的全量 dry analysis 得到：HateMM pooled/macro FP-density AUC 分别约为
`.3881/.4521`，两个高于 `.5` 的门均失败，且 inverse-density share 方向也失败；HateClipSeg
对应约为 `.5594/.5272`，本身通过。故当前输入下双语料 premise 将是 **FAIL**。该结果只能
支持停止此 premise，不能挑选 HateClipSeg 单语料继续。

## Test-label 边界

脚本没有训练、梯度、checkpoint 写入或 checkpoint selection 路径。test GT 只用于 Rule-10
允许的 error analysis：FP/TP density、score-mass share、fixed calibration readout 和共享
evaluation。Fixed inverse-density penalty、reverse control 及其读出均不得作为候选方法、
部署 branch、SOTA claim 或后续 test selector；受本轮结果影响的后续 evidence 均属于
iterative/developmental。

## 验证

- 显式模块解析、只读 feature normalization、cutoff-tie synthetic case 均通过。
- 两语料完整 `analyze_corpus()` dry analysis 通过；raw 三指标与 source evaluator artifact
  一致，coverage、alignment、finiteness 和 eligible cohort 全部通过。
- `py_compile` 与 diff check 通过。
- 评审结束时没有创建正式 run 目录或 metrics 输出。
- 修订后未发现其余会改变本轮观察或结论的问题，也没有引入被项目禁止的内容校验机制。

**最终 verdict：PASS。** 可以运行正式 diagnosis；按当前 dry evidence，正式结论应为
`premise_pass_both=false`，随后停止方法实现。
