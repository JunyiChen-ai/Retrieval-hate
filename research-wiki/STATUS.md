# 当前研究状态

截至 **2026-09-12 03:10 NZST**。依据：模块一第 2 轮实验（`experiments/20260910_online_query_within/`）第 3 次迭代（证据分解）两语料三 seed 完成并回传（`runs/20260910_online_query_within_it3/`；第 1、2 次迭代在 `_it1`、`_it2`）：within 两语料首次超过最强 baseline，HCS pooled 也升，HateMM AP 差 P1 floor .011（实验 README 第 8 节）；候选 3 修订 4（`experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/`，`runs/20260908_c3_rev4_rev2_backbone_interval_hmm/`）P2 通过；模块一自适应 VLM 查询第 1 轮（`experiments/20260908_adaptive_vlm_query/`，commit 4f99109 起）两语料三 seed 搜索、曲线、七臂消融完成并 rsync 回本机（`runs/20260908_adaptive_vlm_query_it1/`；第 0 轮在 `runs/20260908_adaptive_vlm_query/`）。

## 当前目标与结论

**候选 3 修订 4（修订 2 骨干 + 修订 3 区间 HMM 融合）两语料三 seed 过规则 8 并通过预注册 P2：HateMM .6630/.8486、HCS .7024/.6972，两语料都不低于修订 3 − std，HateMM 回到修订 2 一个 std 内。修订 4 是候选 3 的新起点（规则 9 的 3 次修改已用完）。** 修订 3 的结论保留：17 组消融里 13 组两语料成立；区间证据 HMM 融合两语料确认有效；query 门控与 c 只加 logit 在 HateMM 有害，修订 4 已去掉这两处。目标不变：两语料 pooled AP/ROC 过固定 baseline 表（三 seed 确认），三模块各有可主张贡献（规则 14(g)），方法统一、方法级超参少（修订 3：α、λ_block 两个，w_fine 已删）。within 只报告。

## 当前方法：候选 3（证据引导注意力）— 修订 4 为论文当前版本

修订 4：[experiments/20260908_c3_rev4_rev2_backbone_interval_hmm](../experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/README.md) = 修订 2 骨干（key 偏置 g ≡ 1、视频级校准 c 加进两路表示）+ 修订 3 的区间证据 HMM 融合（归一化时间、正例约束、w_fine 固定 1），方法级标量 α、λ_block 两个；用户裁定不做消融，论文消融沿用修订 3 臂表。修订 2：[experiments/20260904_evidence_guided_attention](../experiments/20260904_evidence_guided_attention/README.md)。修订 3：[experiments/20260907_c3_rev3_interval_evidence](../experiments/20260907_c3_rev3_interval_evidence/README.md)（第 0 节审稿批评对应改动，第 1 节方法，第 9 节规则 14 清单）。修订 3 = 修订 2 骨干 + 三处改动（视频级校准 c 只加 logit、query 门控逐头偏置、删 w_fine）+ 融合换成区间证据 HMM（30/4 真实区间合并成 32 段、连续时间转移、正例约束、归一化时间）。评测器未改。

## 最新权威结果（test，1 fps；三 seed 234/2025/3407，每 seed 20-trial 搜索最优 trial；来源 `runs/<exp>/<corpus>/seed<seed>/trial<best>/metrics.json`）

| 方法 | HateMM AP / ROC / within | HCS AP / ROC / within |
|---|---|---|
| 规则 8 门（最强训练 baseline） | .573 ± .033 / .807 ± .019 / .632 | .562 / .528 / .524 |
| 候选 1（rev4） | .657 / .842 / .646 | — |
| 候选 3 修订 2（`runs/20260904_evidence_guided_attention_rev2_noprune`，HCS `runs/20260904_evidence_guided_attention`） | .6678 ± .0097 / .8504 ± .0049 / .6233 ± .0150 | .6976 ± .0076 / .6843 ± .0095 / .5488 ± .0111 |
| 候选 3 修订 3（`runs/20260907_c3_rev3_interval_evidence`） | .6409 ± .0174 / .8421 ± .0080 / .6310 ± .0101 | .7045 ± .0053 / .6924 ± .0089 / .5678 ± .0040 |
| **候选 3 修订 4**（`runs/20260908_c3_rev4_rev2_backbone_interval_hmm`，best trial 11/6/13 与 11/19/19） | **.6630 ± .0173 / .8486 ± .0097 / .6229 ± .0168** | **.7024 ± .0080 / .6972 ± .0129 / .5630 ± .0023** |
| 模块一第 2 轮实验第 3 次迭代：证据分解（细窗 HMM + 文本对数几率 + 视频级粗块项）+ 单次训练在线查询（`runs/20260910_online_query_within_it3`，best trial 18/8/7 与 17/17/11；测试 8 次、训练 8 次） | .6438 ± .0061 / .8511 ± .0047 / **within .6419 ± .0152**；停止点 4.84 次 .6388 / .8465 / .6401 | .6953 ± .0069 / .6820 ± .0027 / **within .5650 ± .0093**；停止点 7.30 次 .6963 / .6806 / .5663 |
| 模块一第 2 轮实验第 1 次迭代：单次训练在线查询 + 窗裁定预测（`runs/20260910_online_query_within_it1`，seed 234） | .6569 / .8389 / .6190（trial 8） | .6822 / .6716 / .5344（trial 14） |
| 模块一自适应查询 + 修订 4 骨干（`runs/20260908_adaptive_vlm_query_it1`，best trial 11/11/11 与 12/17/15；测试期每视频 12 次 VLM 裁定，训练期 15.3 次，全量 34） | .6674 ± .0130 / .8487 / within .6089（seed 234） | .6903 ± .0043 / .6873；validation 停止点 11.5 次 .6983 / .6871 |

修订 3 消融（三 seed 均值 full − arm，AP / ROC；两语料 ≥ .01 才可主张）：两语料成立 13 组：index_hmm（HateMM .037/.020，HCS .007/.024）、no_constraint、seconds_time、no_cell、no_bias、shared_bias、no_context、mean_prior、mean_prior_all、no_block、no_prior、no_cmal、no_verdict。不成立：no_qk_enc（HCS ≈ 0）、avce（HateMM .005/.009）、key_bias（HateMM −.010）、ctx_in_rep（HateMM −.023）。机制检验：打乱证据时间对应 HateMM 掉 .012/.013/within .023，HCS 不掉；c 推断时置零两语料变化 ±.009 内。自适应查询回放（0 次新 VLM 调用）见实验 README 第 8 节。

模块一（自适应 VLM 查询，实验 README 第 6 节）：先问 4 个粗块，骨干按 EOC 决定再问哪些细窗；0 次新 VLM 调用（全部从 34 条缓存裁定模拟）。预注册判定：E1（平均调用 ≤ 12）两语料过；E2（三 seed 不低于修订 4 均值 − std）按 validation 选的停止点两语料过（HateMM 9.3 次 .6666 / .8489；HCS 11.5 次 .6983 / .6871），按固定 12 次点 HateMM 过、HCS AP 差 .004；E3（比 uniform 高 ≥ .01 AP）HCS 过（+.010）、HateMM +.009。七臂消融三 seed（AP 两语料 ≥ .01）：no_dropout、train34、round0_only、coarse4_train 成立；uniform / conflict / no_missing_state 单侧差 .001–.003 不可主张。第 0 轮（训练期策略从 seed 窗起跑）：HateMM .6560 / .8521、HCS .6884 / .6853。

## 运行任务与监控

截至 2026-09-09 10:40：无运行中任务。uoa-lab1 / uoa-lab3 GPU 空闲；本机 GPU 被他人占用。会话内 heartbeat 已关。

## 下一步

0. **模块一第 2 轮实验（2026-09-12 更新）**：第 3 次迭代（证据分解）三 seed：W1、E1 两语料过，P1 HCS 过（ROC −.0003 噪声内），HateMM AP −.011 未过；第 1、2 次迭代（窗裁定预测；文本进 HMM 当观测族）失败并记录。待用户裁定：规则 3（文本分类器的分数经 α 进最终分数）与 HCS ROC −.0003 是否按噪声处理。下一步：训练期预算自适应分配（README 第 9 节）。原 2026-09-10 阻塞记录：预注册 P1（pooled 不低于第 1 轮 12 次点 − std）、W1（within ≥ max(起点 + .01, 最强 baseline)：HateMM .638、HCS .562）在 8 次调用下两语料都不过。诊断（实验 README 第 6 节）：within 由裁定证据决定，内容流视频内排序为随机水平（HCS 4 粗块 .485）；HCS 8 次各策略 within ≤ .54，12–16 次的 entropy / localization 策略才到 .57–.59；HateMM 训练期细窗从 11.3 降到 4 代价 AP .03–.04。需用户裁定：放宽调用（12–16）、放宽 within 门（起点 + .01）或换视频内监督的来源。
1. 用户裁定模块一主操作点：停止点（两语料 E1+E2 过）或固定 12 次点（HCS AP 差 .004 → 第 2 轮，候选见实验 README 第 6 节"去向"）。
2. 论文表述：机制主张"证据决定从哪聚合"限定 HateMM；HCS 写 limitation。B.2 引 WavLM 门控相对位置偏置，融合引 Dugong / CHMM / CT-HMM / EM-MIL（修订 3 `REVIEW_RULE4.md`）；模块一引 Covert 2023 / DIME / EDDI / A2MT / VideoAgent / VADTree / Holmes-VAU / LELA / CLARA（`experiments/20260908_adaptive_vlm_query/REVIEW_RULE4.md` 第 5 节），"没问过"状态按沿用 DFS 掩码输入惯例写，EOC 按 DIME 目标量的非摊销变体写。
3. 搜索目标继续按 test（用户裁定，不再讨论）。

## 资料与历史

[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)。负结果细节：[C9](../archive/experiments/20260906_interval_evidence_transport/README.md)、[C6](../archive/experiments/20260905_latent_evidence_sequence/README.md)、[C7用户取消](../archive/experiments/20260906_context_witness/README.md)、[C4](../archive/experiments/20260904_null_token_cma/README.md)。[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。冻结Hate-follow-up引用不动；缓存出处见各data子目录PROVENANCE.md。
