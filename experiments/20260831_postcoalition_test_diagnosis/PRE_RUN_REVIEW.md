# Post-coalition test diagnosis：独立 pre-run review

日期：2026-08-31  
状态：正式 `main()` 运行前  
结论：**PASS，可以运行。**

## 输入与 evaluation

- 两语料 structured source 现统一为上一轮 `mobius_nonminimal` 的 `score_full` test
  predictions；HateMM 与 HateClipSeg 分别来自各自独立训练的正式 pilot run。
- 两语料第二个输入均为各自 corpus-specific POWA seed-234 `score_powa` test anchor。
- Structured source 是根据已经发生的 prior test pilot 选出的两语料共同 within-best control；
  该选择已在 README 和输出 provenance 中明确标作 test-informed error-analysis source，不是
  可部署 branch selection。
- 所有项目固定三指标（raw 与每个 fixed smoothing window）均调用仓库唯一共享
  `evaluate_scores`；没有在实验目录复制这三项评测逻辑。逐视频 `safe_auc` 只用来生成
  mixed-label positive-video error-analysis 统计，不作为项目固定三指标或 performance claim。
- Dry analysis 验证 HateMM 214 个 test 视频、29,269 帧，HateClipSeg 79 个视频、18,839
  帧。两个 source 在各 corpus 都与 test GT exact coverage，逐视频 shape 完全一致且全部
  finite。within-video cohort 是同时含正负秒的 positive videos：HateMM 85、HateClipSeg 67。

## 发现并修复的问题

1. **Rank mean 初版用 stable argsort 拆同分。** 这会把时间索引变成隐式 secondary key。
   实际 eligible 视频中，structured score 存在 ties 的比例并不小。修订版改为 average rank；
   相同 score 必得相同 rank，组合结果不再依赖原数组顺序。
2. **GT-occupancy mask 初版也用时间索引拆 cutoff tie。** 修订版改为 cutoff-score 的
   tie-inclusive superlevel set；同时记录 intended count、实际 count/fraction、cutoff score
   和 boundary expansion。Dry analysis 实际识别出 HateMM 7 个、HateClipSeg 17 个 cutoff
   plateau 视频，证明该修复会实质改变 fragmentation 观察。
3. **初版按 corpus 使用不同 structured control，且“最佳”标准不清。** HateClipSeg 原指向
   SynIB；修订版在两个 corpus 都固定使用 `mobius_nonminimal`。它是 prior pilot 中两语料
   within-video ROC 都最高的 coalition structured control，从而避免按语料切换机制；其
   test-informed 来源仍只允许用于本轮 diagnosis。

## 统计定义

- Fixed smoothing 只报告预先列出的 `1/3/7/15/31` 秒窗口；每个视频独立处理，边界使用
  `nearest`，不跨视频。脚本不选一个窗口写回 prediction 或训练。
- `best_window_tie_inclusive_video_counts` 是 test-GT oracle 诊断，允许多个并列窗口同时计数，
  不应把 counts 相加解释为视频数。
- Fragmentation 用 GT positive count 定义 score cutoff，但 boundary ties 全部纳入并报告实际
  occupancy；它是明确的 test oracle，不是 inference rule。
- Transition count 是相邻二值秒发生变化的次数；inflation ratio 逐视频计算后取 median，视频
  等权。
- Per-video AUC 通过 sklearn 只在 mixed-label positive videos 上计算，是 diagnosis 内部统计；
  项目的 pooled ROC、within-video macro ROC 和 pooled AP 仍只认共享 evaluator 输出。
  `fraction_*` 和 mean oracle 都以视频为单位，不把帧或视频长度当额外权重。
- `mean_best_of_two_test_oracle_auc` 使用 test GT 按视频取较好现有模型；`rank_mean` 是两个既有
  模型的 ensemble；smoothing 是 calibration。三者均只用于定位失败来源，不能作为候选方法、
  SOTA 结果、部署规则或后续 test branch selector。

## Test-label 边界

脚本不训练模型、不更新 checkpoint、不生成供训练使用的 pseudo-label，也没有任何梯度或
checkpoint-selection 路径。test GT 只用于允许的 Rule-10 error analysis：计算 evaluator
指标、occupancy/transition、per-video oracle 和 fixed-window诊断。输出明确标为
`iterative/developmental`；受此分析影响的后续 test 结果不得描述为未揭盲 confirmatory
evidence。

## 验证

- 修订后的 rank tie 与 occupancy cutoff tie 合成测试通过。
- 两语料完整 `analyze_corpus()` dry analysis 通过，未写正式 run 输出。
- `py_compile` 通过。
- 未发现会改变剩余观察或结论的实现问题，也未发现被禁止的内容校验依赖。
- `runs/20260831_postcoalition_test_diagnosis/` 在评审结束时仍无正式输出。

**最终 verdict：PASS。** 可以按 README 命令运行；运行后必须继续把 smoothing、occupancy、
oracle 和 rank mean 只解释为 developmental diagnosis。
