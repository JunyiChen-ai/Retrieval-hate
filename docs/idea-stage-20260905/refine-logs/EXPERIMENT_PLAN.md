# EXPERIMENT PLAN — 裁定条件化密度估计（claim-driven；纸面计划，未执行）

依据 `FINAL_PROPOSAL.md`。协议按 `RESEARCH_ITERATION_RULES.md` 第 7/8/9/14 条（Optuna 20 trial/(语料, seed)，目标 test (AP+ROC)/2，validation 选 ckpt，within 破下限剪枝；seed 234 筛选，2025/3407 确认）。所有臂三 seed × 两语料，臂用该 seed best-trial 超参，每臂固定 3 条随机数流取均值。运行主机与输出目录按 CLAUDE.md（`runs/<exp_id>/<corpus>/seed<seed>/...`，远程结果回传后才更新 STATUS）。

## 0. Claims
- **K1（主）**：coarse-first 条件标注模型的后验优于全局发射 HMM 与 K4-only 后验（posterior-alone 与端到端）。
- **K2**：其每视频期望仇恨比例 q_v 是可用的比例估计（MAE / 偏差 / 校准优于 global 与 K4-only）。
- **K3**：scale–rank 分解头（学习的 s_v + 中心化 r_t + 最终概率比例损失）在两语料提升 pooled，且超过零参数的精确重参数化对照；推理时 within 不变。
- **K4**（子）：文本 between/within 路由在 within 不降的前提下取得 no_text 臂的 pooled ROC 增益。
- **F1**（发现，不作贡献）：保留 s_v 时删除注意力层 pooled 几乎不降，即注意力的作用是跨视频密度。

## Block A — CPU 门（0 GPU-h，先于一切；脚本放 `experiments/<date>_vcde/label_model_gate.py`，只读 `data/`，输出 `runs/<exp_id>/label_model_gate/`）
| 行 | 内容 | 判定 |
|---|---|---|
| A1 | global（候选 1 HMM）、K4-only、K30-only、4-cell lookup、conditional（full）、`no_b4`、`no_bprev` 的 posterior-alone test/val AP/ROC/within，两语料 | K1 门：conditional HateMM AP 与 ROC 都 ≥ K4-only + .005 且 ≥ lookup；HCS ≥ .698/.661；`no_b4` 与 `no_bprev` 都低于 full，否则不能 claim 粗条件化 |
| A2 | 每视频比例：q_v（OOF 后验）对 GT 比例的 MAE、偏差、Pearson、可靠性图，五种模型 | K2 门：conditional 三项都优于 global 与 K4-only；否则删除模块 2 的比例损失 thesis（保留 s_v 但 λ_prop = 0） |
| A3 | 合成数据多起点恢复：θ_0/θ_1/q4/r4/A/p0 | 报告，不作门 |
| A4 | OOF 训练后验生成（5 折）与 in-sample 的差异 | 报告；网络一律用 OOF |
A1 或 A2 不过 → 模块 3 归档；若 A1 过、A2 不过 → 模块 3 保留、模块 2 只做 s_v（无比例损失）。

## Block B — 精确对照（≈ 1.8 GPU-h，候选 1 seed best-trial 超参，无搜索）
| 臂 | 内容 | 预期 |
|---|---|---|
| B0 | 候选 1 在新代码路径下逐位复现（分数 bit-match） | 必须相同，否则先修代码 |
| B1 | 精确重参数化：s_v = mean a_t，r_t = a_t − mean，无新参数 | 与 B0 相同分数 |
| B2 | 固定 s_v = logit q_v（无学习） | 报告 |
| B3 | no-s_v + 不中心化（= B0）、s_v 只读裁定统计 | 报告 |
| B4 | block-relocation-only（块级 MIL 搬到总分、不加 s_v） | 隔离块损失搬迁 |
| B5 | no_text（候选 1 去 BERT） | HateMM ROC ≈ .869 的参考上限 |

## Block C — 完整方法搜索（≈ 9 GPU-h）
配置 111（C+D+T）按规则 7 在两语料各 seed 20 trial 搜索；搜索空间 = 候选 1 的 9 个超参 − EMA 相关 + λ_prop（log[.05, 2]）+ s_v MLP dropout；两语料同一空间。规则 8 筛选与确认。

## Block D — 机制消融（2^3 八格 + 必做对照；≈ 4.5 GPU-h 含 3 流）
| 臂 | 回答 |
|---|---|
| 000 (=B0) / 100 / 010 / 001 / 110 / 101 / 011 / 111 | 三个开关的 leave-one-out 与交互 |
| 111 − λ_prop（λ_prop = 0） | 比例损失是否有贡献（K3） |
| 111 with s_v from content only | 裁定统计是否必要 |
| 111 uncentred r_t | 中心化是否必要（K3 可识别性） |
| 111 keep s_v, delete attention | F1 |
| 111 text decomposition applied to visual+audio | 对照：应变差（K4 特异性） |
| 111 with MSL-style multiplicative video gate instead of additive s_v | 竞争头（novelty 定位） |
| 111 with K4-only prior / 4-cell offset instead of conditional | 模块 3 的简单竞争者 |
| 候选 1 原有臂 no_block / no_prior / no_cmal / no_verdict / mean_prior | 论文全表 |
逐开关 claim 标准（论文）：HateMM objective ≥ +.020、HCS ≥ +.012，两项 pooled 不低于 −.005，within ≥ 候选 1 − .005，seed 级配对 CI 排除 0。

## Block E — 追加 seed（≈ 1.8 GPU-h，可选）
对 111、000、两个最关键 leave-one-out 臂预先追加 3 个配对 seed（评审：检出 HateMM +.020 约需 8 seed）。

## 顺序与决策点
1. Block A（CPU，半天）→ 决定模块 3 存废与 K2。
2. Block B（1.8 h）→ B0/B1 不 bit-match 则停。
3. Block C（9 h）→ 规则 8 seed 234 筛选；不过则规则 9 分流。
4. Block D（4.5 h）→ 规则 14(g) 与论文 claim 标准。
5. Block E 视 D 的 CI 而定。
总计 ≈ 16–18 GPU-h（HateMM 约 2/3）。按 CLAUDE.md 选机；HateMM 与 HCS 尽量同机。

## 结果到 claim 的映射
| 结果 | 允许的 claim |
|---|---|
| A1 过、A2 过、C 过规则 8、D 中 C 与 D 开关都过标准 | 完整方法：条件标注模型 + 裁定条件化密度估计；T 视其自身标准 |
| A1 过、A2 不过 | 只 claim 条件标注模型（模块 3）；模块 2 退为 s_v 无比例损失的辅助分析 |
| A1 不过 | 模块 3 归档；记录"K4-only 调温已是上限"；模块 2 单独走 Block B–D |
| B1 ≈ 111（学习 head 不超过精确重参数化） | 不能 claim scale–rank 头；只报告分解作分析 |
| D 中 F1 成立（删注意力保 s_v 不降） | 作为经验发现写入分析节 |
| 任何开关 within 低于候选 1 − .005 | 该开关删除 |
