# Frozen test error analysis post-run audit

截至 2026-08-31。审计对象：
`runs/20260831_multimodal_pmil_baseline/pilot_seed234/test_error_analysis.json`，对应两个 corpus的
frozen model、正式 `scores.jsonl` / `metrics.json`，以及已通过 pre-run review的
`analyze_test_errors.py`。

## 裁定

**PASS。** artifact由两个正式 baseline runs和共享 test evaluation完成后生成；`pmil_full`与正式
prediction/evaluator数字完全一致，cohort、proposal-event denominator、whole/top tie计数和Spearman
denominator均闭合。没有修改任何结果，也没有重新运行模型或启动实验。

该 JSON 只能作为 iterative/developmental test error analysis使用。除 `pmil_full` 外的删组件、单模态和
oracle arms均是 frozen same-checkpoint diagnostics，不是独立训练方法，不进入 SOTA 表，也不授权
per-corpus routing或后验选择。

## Frozen run 与 full-score gate

- HateMM formal run先完成 `model.pt`、`scores.jsonl`和共享 `metrics.json`，随后才生成 analysis JSON；
  selected epoch为15。HateClipSeg同样先完成正式 artifacts，selected epoch为5。
- 两 corpus正式 score files均 exact-cover evaluator cohort，无 missing/extra videos：HateMM 214，
  HateClipSeg 79。
- 使用共享 `evaluate_scores`从正式 `scores.jsonl`只读复算 `score_pmil`，并与 formal metrics及analysis
  `pmil_full`逐值比较：

| corpus | pooled AP | pooled ROC | within ROC | within n |
|---|---:|---:|---:|---:|
| HateMM | 0.5892557205 | 0.8142208576 | 0.5898977996 | 85 |
| HateClipSeg | 0.4605069285 | 0.4230704517 | 0.4766100594 | 67 |

三份记录完全一致。这也与脚本的强制门相符：analysis逐视频重算 `pmil_full`，若与 frozen
`score_pmil`差异超过 `1e-6`就不会写出最终 JSON。

## Diagnostic metrics 审计

- 所有8个 arms均覆盖相同 cohort，三项指标有限且位于 `[0,1]`；所有 arms的within n分别固定为
  HateMM 85、HateClipSeg 67。
- `source_smil`、`pmil_without_completeness`、`pmil_hate_cas_only`和三条 per-modality full arms的定义与
  当前代码一致；`pmil_full`使用三模态各自 `hate × attention × completeness` 后求平均，而不是跨模态
  分量相乘。
- `proposal_oracle`按每个 frozen proposal到所有 contiguous GT intervals的最大 temporal IoU赋分，再走
  同一 max proposal-to-frame readout。artifact已明确标为 GT-informed upper bound；其高分不构成方法或
  SOTA结果。
- GT event数只读复算为 HateMM 181、HateClipSeg 260。artifact的event-macro recall恰好对应：
  HateMM IoU `.10/.30/.50` 为 `176/160/138` events；HateClipSeg为 `223/153/118` events。三个threshold的
  recall单调，mean maximum IoU均在合法范围内。
- whole-video top fraction精确还原为 HateMM `3/214`、HateClipSeg `48/79`。两 corpus
  `top_score_tie_video_fraction=0`，因此当前 top-length median/mean没有并列聚合歧义；数值分别为
  HateMM `8.0/24.1869s`、HateClipSeg `213.0/199.2532s`。
- 每个 pair的Spearman finite与undefined数量之和均等于within eligible cohort：HateMM
  `79+6`, `75+10`, `75+10`；HateClipSeg `64+3`, `14+53`, `14+53`。这与代码“mixed-GT视频上的
  per-modality full frame-vector Spearman；constant vector记undefined”的定义一致。HateClipSeg涉及text的
  两组只有14个finite videos，不能忽略 denominator直接解释均值。

## 可复算性与使用边界

- 正式 full arm可直接由已保存 `scores.jsonl`和共享 evaluator复算，本审计已完成。
- 其余 diagnostic scores没有另存逐视频 arrays；数值复算需要使用 frozen selected model、frozen source
  checkpoint、固定 proposals、test features/GT和当前 analysis脚本重新执行。所有输入路径与定义均已
  固定，脚本还会先验证 full arm与正式 prediction一致，因此具备确定的复算路径；但仅凭 summary JSON
  本身不能独立重建这些逐帧 diagnostic arrays。这是artifact粒度限制，不改变本次结果含义。
- JSON顶层和每 corpus均明确标记 developmental test evidence、GT用途及
  `diagnostic_arms_are_not_retrained_methods_or_sota_entries=true`。后续设计可以使用这些error findings，
  但之后的test结果必须继续表述为 iterative/developmental evidence。

最终裁定：**PASS FOR DEVELOPMENTAL USE**。
