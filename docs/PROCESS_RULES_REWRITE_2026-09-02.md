# 流程规则重写记录 — 2026-09-02

用户裁定，主 agent 执行。本文件冻结，只记录重写原因与新旧对照；现行规则以 `RESEARCH_ITERATION_RULES.md` 为准。

## 重写原因（数据来源均为 `runs/` 评测器输出与 `docs/duplex/official_val_results.json`）

1. **旧晋级门不可达。** 旧门要求 HateMM/HateClipSeg 六项指标全部超过"各列最高值"，单 seed 234。
   - HateClipSeg within 门 .562 取自 VERA（0/1 分数经固定后处理；去后处理为 .521，见 `runs/20260831_dense_vera_test_diagnostic/hateclipseg.json`），高于同特征 train-span 全监督线性分类器的 .560（`runs/20260831_imagebind_feature_ceiling/analysis.json`）。
   - HateMM within 门 .6315 等于 starting point MultiHateLoc 自身 3 seed 均值（.633/.628/.633）。
   - 七个已发表 baseline 没有一个能过这套门。14 个正式方法全部"失败"；marked temporal splat HateMM within .728 仍被淘汰。
2. **单 seed 判定，差异在噪声内。** baseline within 的 seed 标准差 .003–.05；多数"机制不成立"结论基于 .0001–.005 的差异。
3. **对照每次重训。** 各候选自行重训"机制关闭"版 MultiHateLoc，HateMM within 从 .555 到 .633 不等；HateClipSeg 对照 pooled 比官方低 .029/.047（学习率与 checkpoint 选法不同，见 `research-wiki/RESET6_GOAL_GAP_AUDIT.md`）。
4. **实现前理论否决无预测力。** 约 40 个候选被 novelty 审稿以"可能退化/shortcut"推理 STOP，未验证；通过审稿（GO 6.6–7.1）的候选也全部失败。
5. **流程自我膨胀。** 两天 7 次 process reset，规则编号项 25 → 38，RESET5 强制 anchor-compatible、RESET7 撤销。
6. **信息源被锁死。** 旧规则 16/21/22 要求新输入先有双语料证据，实际把需要新输入的迁移方法全部挡住；两天 105 个候选都在同一套冻结特征上换 MIL 头，效应 ±.01。

## 新旧对照

| 项目 | 旧 | 新 |
|---|---|---|
| 晋级门 | 两语料六项全超各列最高，单 seed | 两语料 within 超最强训练 baseline 3 seed 均值，pooled 不低于 MultiHateLoc 均值 −.02；单 seed 筛选，3 seed 确认 |
| VERA | HateClipSeg 三项门 | 汇报不作门 |
| 对照 | 每候选重训 matched control，先过机制门 | 只与固定 baseline 表比；消融放在 SOTA 之后 |
| Novelty 审核 | 三门 + 可识别性推理 + failure ledger 阻断 | 只挡四类：来源已用于本任务、纯 ensemble、纯 calibration/后处理、纯工程技巧 |
| 新输入/特征 | 需先有双语料 premise 证据 | 方法需要即抽取，不单独过审 |
| 失败处理 | 一次失败关闭 family；3 次触发 process review/RESET | 有 .01 以上提升则继续改（最多 3 轮）；无提升归档；无自动停机 |
| Smoke / 单元测试 | 禁 smoke，要求静态检查+单元测试 | 禁 smoke，不要求单元测试 |
| Code review | 一次基础 review | 不变 |
| Validation 用途 | 方法内选超参/ckpt | 不变 |

## 同步修改的文件

- `RESEARCH_ITERATION_RULES.md`：整体重写。
- `AGENTS.md`：删除 Test-first / Novelty / 三候选停机三节，改为指向规则文件。
- `research-wiki/STATUS.md`：顶部新增"流程重写"块，声明旧裁定作废，给出新门数值与运行中任务状态。
- `research-wiki/FAILURE_EQUIVALENCE_LEDGER.md`：降为参考信息。

## 同日第二次修订（用户裁定）

1. 晋级主门改为文献通用的 pooled frame AP / ROC（MultiHateLoc、LELA、SafeLens 均只报这两项；within-video macro ROC 在 hateful video 文献中无先例，在 VAD 文献中只有 Georgescu TPAMI 2021 / UBnormal CVPR 2022 一支少数派使用）。within 改为下限约束（不低于 MultiHateLoc 3 seed 均值），用途是排除整段视频打高分的 shortcut。
2. 新增规则第 13 条"方法必须统一"：按语料切换 backbone 冻结、手写 policy、读出形式、模块开关、联合训练、后处理或分支的做法视为反模式；POWA 现状即此反模式。
3. 新增规则第 14 条"做完了的声明门槛"九项核对，防止以非科研方法或开发期数字 claim 完成。


## 同日第三次修订

- 主数据集缩为 HateMM 和 HateClipSeg。MHC-EN、MHC-ZH 不再跑、不作门、不进论文主表；删除规则第 8 条"确认后补跑 MHC-EN/ZH"一项，第 13 条与第 14 条 (i) 的"四语料"改为两语料。`CLAUDE.md` 与 `research-wiki/STATUS.md` 同步修改。


## 同日第四次修订

- 规则第 7 条改为固定 Optuna 搜索：每个 seed、每个语料一个 study，TPE sampler；单 trial 不超过 1 小时的方法 20 trial，超过 1 小时统一 5 trial（含 MLLM 与否不再区分）；trial 内 train 训练、validation 选 checkpoint、test 出三项指标；Optuna 目标值为标量 (test pooled AP + test pooled ROC)/2，仅用于排序 trial，过门仍按两指标各自对照；within 破下限的 trial 记失败。最优 trial 的 test 数字为该 seed 的有效检验值（开发期上限测量，与论文报告无关）。第 8 条筛选/确认、第 10 条、第 14 条 (a)(d)、标准流程第 5–6 步同步改写。
