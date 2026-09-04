# Pipeline Summary

**Problem**：弱监督 hateful video localization（HateMM + HateClipSeg，一套架构），为骨干与融合各找一个可 claim 的设计。
**Final Method Thesis**：逐秒分数 = 视频级仇恨比例项（由 coarse-first 条件标注模型监督）+ 视频内排序项；细裁定可靠性以粗裁定为条件。
**Final Verdict**：REVISE（Codex 第 2 轮 5/10；方法可实现，中心命题依赖 CPU 门 A1/A2 的结果）。
**Date**：2026-09-05。

## Final Deliverables
- Proposal: `docs/idea-stage-20260905/refine-logs/FINAL_PROPOSAL.md`
- Review summary: `docs/idea-stage-20260905/refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `docs/idea-stage-20260905/refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `docs/idea-stage-20260905/refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `docs/idea-stage-20260905/refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- Dominant contribution：模块 3 coarse-first 条件标注模型（细裁定发射以粗裁定与前一细裁定为条件；负例识别误报侧）。
- Supporting contribution：模块 2 scale–rank 分解头（学习的视频级比例标量 + 中心化排序项 + 最终概率比例损失），文本 between/within 路由作子开关。
- Explicitly rejected complexity：MIL 精调发射、内容条件化发射、块级加性项、精度加权 PoE、块内受限注意力/register、query 门与 sink、HSMM/秒级 IOHMM。

## Must-Prove Claims
- K1 条件后验 > K4-only 后验（posterior-alone 与端到端）。
- K2 q_v 是可用的比例估计。
- K3 学习的 s_v 超过精确重参数化对照，两语料 pooled 提升，推理时 within 不变。

## First Runs to Launch
1. Block A CPU 门（A1、A2）——0 GPU。
2. Block B 精确对照 B0/B1（bit-match）。
3. Block C 111 搜索 seed 234 两语料。

## Main Risks
- 增量落在 ±.02 噪声内：按预注册标准删开关，不硬撑。
- HateMM val/test 对 K30 可靠性方向相反：A1 同时报 val/test。
- 顶会 novelty 为应用级：按 FINAL_PROPOSAL 第 2、3 节定位并正面引用 CHMM/FABLE/Dugong/MSL/ARMS/2026 审计。

## Next Action
- 用户裁定后：写 `experiments/<date>_vcde/README.md`（规则 4 proposal review）→ 实现 → 规则 6 code review → Block A → `/run-experiment`。
