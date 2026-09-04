# Research Contract: 裁定条件化密度估计（Verdict-Conditioned Density Estimation）

> 当前选定 idea 的工作契约。来源 `docs/idea-stage-20260905/IDEA_REPORT.md` 推荐候选 R1；方法细节 `refine-logs/FINAL_PROPOSAL.md`；实验 `refine-logs/EXPERIMENT_PLAN.md`。截至 2026-09-05，未实现、未运行。

## Selected Idea
- **Description**：模块 3 把两级 HMM 的细裁定发射改为以粗裁定与前一细裁定为条件的 logistic 发射（coarse-first 有向模型；θ_0 由负例视频 logistic 回归识别，其余正例 EM；训练期后验 OOF）。模块 2 把逐秒分数分解为视频级比例标量 s_v（裁定统计 + detach 内容池化 + BERT 视频均值）与视频内中心化排序项 r_t（候选 1 完整分数去均值），比例损失作用于最终逐秒概率均值 → HMM 期望仇恨比例；文本只以逐秒偏差进内容流。
- **Source**：IDEA_REPORT.md，簇 C7 + C1（C13 子开关）。
- **Selection rationale**：唯一同时具备项目内测得杠杆（K4-only 后验 +.05 AP；视频间方差 92%/79%；no_text ROC +.027）、按构造的推理期 within 保护、0 GPU 可证伪的 CPU 门、以及交叉模型分诊第 1 的骨干 + 融合组合；novelty 复核无 ABANDON。

## Core Claims
1. K1：coarse-first 条件标注模型的后验优于全局发射 HMM 与 K4-only 后验（posterior-alone 与端到端），且增益不只来自自回归项（no_b4 / no_bprev）。
2. K2：q_v 是可用的每视频比例估计（MAE / 偏差 / 校准优于 global 与 K4-only）。
3. K3：学习的 s_v + 中心化 r_t + 比例损失在两语料 pooled 超过候选 1 与精确重参数化对照；推理时 within 不变。
4. K4（子）：文本 between/within 路由在 within ≥ 候选 1 − .005 的前提下取得 no_text 的 ROC 增益。
5. 范围声明：不从密度项 claim localization 改善；within 并列报告。

## Method Summary
见 `refine-logs/FINAL_PROPOSAL.md` 第 2、3 节。骨干其余（投影、共享跨模态层、CMAL、top-k bag、块级 MIL 结构）与候选 1 相同；EMA 伙伴删除。

## Experiment Design
- Datasets：HateMM、HateClipSeg，各自 train/val/test；一套架构。
- Baselines：候选 1 三 seed 记录（HateMM .657/.842/.646；HCS .699/.681/.553）；规则 8 表（MACIL-SD、Fed-WSVAD、DSANet、MultiHateLoc）；K4-only 先验；4-cell lookup；精确重参数化；固定 logit q_v；MSL 乘性门。
- Metrics：pooled AP / ROC（主）、within-video macro ROC（下限）。
- Key hyperparameters：候选 1 的 9 个 − EMA + λ_prop + s_v dropout；两语料同一空间。
- Compute：≈ 16–18 GPU-h；CPU 门 0 GPU。

## Baselines
| Method | Dataset | AP / ROC / within | Source |
|---|---|---|---|
| 候选 1（3 seed） | HateMM | .657 / .842 / .646 | `runs/20260903_hier_evidence_mil/hatemm/seed*/study_summary.json` |
| 候选 1（3 seed） | HateClipSeg | .699 / .681 / .553 | `runs/20260903_hier_evidence_mil/hateclipseg/seed*/study_summary.json` |
| HMM 后验单独 | HateMM / HCS | .541/.818 ; .698/.661 | `runs/20260903_hier_evidence_mil/verdict_hmm_only/` |
| K4-only 后验 | HateMM | .591 / .851 | `runs/20260903_hier_evidence_mil/verdict_hmm_only_wfine/hatemm/test/` |

## Current Results
（空；未运行。）

## Key Decisions
- 条件发射只读裁定上下文，不读内容：候选 2 的内容驱动门卡死；负袋可识别性只在裁定上下文成立。
- s_v 加性而非乘性：推理期顺序保持；MSL 乘性门作对照臂。
- 比例损失作用于最终概率均值：sigmoid(s_v) 版本不约束逐秒比例（评审）。
- 先 CPU 门再实现网络：三个开关预期增量都可能在噪声内，先证伪最便宜的部分。
- 已知风险：HateMM val/test 对 K30 可靠性方向相反；顶会 novelty 为应用级；test 驱动搜索协议为项目裁定。

## Status
- [x] Idea selected
- [ ] Baseline reproduced（B0 bit-match）
- [ ] Main method implemented
- [ ] Block A CPU gate
- [ ] Full dataset results
- [ ] Ablation studies
- [ ] Paper draft
