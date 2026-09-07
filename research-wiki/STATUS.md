# 当前研究状态

截至 **2026-09-08 06:15 NZST**。依据：候选 3 修订 4（`experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/`）两语料三 seed 搜索完成并 rsync 回本机（`runs/20260908_c3_rev4_rev2_backbone_interval_hmm/`），P2 判定通过；修订 3 的消融、机制检验结果不变（`runs/20260907_c3_rev3_interval_evidence/`）。

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

修订 3 消融（三 seed 均值 full − arm，AP / ROC；两语料 ≥ .01 才可主张）：两语料成立 13 组：index_hmm（HateMM .037/.020，HCS .007/.024）、no_constraint、seconds_time、no_cell、no_bias、shared_bias、no_context、mean_prior、mean_prior_all、no_block、no_prior、no_cmal、no_verdict。不成立：no_qk_enc（HCS ≈ 0）、avce（HateMM .005/.009）、key_bias（HateMM −.010）、ctx_in_rep（HateMM −.023）。机制检验：打乱证据时间对应 HateMM 掉 .012/.013/within .023，HCS 不掉；c 推断时置零两语料变化 ±.009 内。自适应查询回放（0 次新 VLM 调用）见实验 README 第 8 节。

## 运行任务与监控

截至 2026-09-08 06:15。
- 模块一自适应 VLM 查询（`experiments/20260908_adaptive_vlm_query/`，commit 3a3a00b 起）：调研（`docs/20260908_adaptive_query_survey.md`）、代码、规则 4 复核（PASS，`REVIEW_RULE4.md`）、规则 6 复核（PASS，`REVIEW_RULE6.md`）完成；骨干钉为修订 4（key/rep）；seed 234 两语料搜索启动中（HateMM uoa-lab1，HCS uoa-lab3；`runs/20260908_adaptive_vlm_query/<corpus>/seed234/search.log`）。预注册见实验 README 第 2 节（E1 操作点 12 次/视频；E2 三 seed 不低于修订 4 − std；E3 比 uniform 高 ≥ .01；τ 按 validation 规则选）。
- 会话内 heartbeat 每 3 小时一次。本机 GPU 被他人占用。

## 下一步

1. 模块一 seed 234 两语料 → 规则 8 → E1/E2 单 seed 预判；不满足按 README 2.4 / 第 5 节迭代表迭代到满足为止（用户指令）；满足则 seed 2025/3407、曲线与对照、消融（第 3 节臂表，三 seed 两语料，不做 bootstrap）。
2. 论文表述：机制主张"证据决定从哪聚合"限定 HateMM；HCS 写 limitation。B.2 引 WavLM 门控相对位置偏置，融合引 Dugong / CHMM / CT-HMM / EM-MIL（修订 3 `REVIEW_RULE4.md`）；模块一引 Covert 2023 / DIME / EDDI / A2MT / VideoAgent / VADTree / Holmes-VAU / LELA / CLARA（`experiments/20260908_adaptive_vlm_query/REVIEW_RULE4.md` 第 5 节）。
3. 搜索目标继续按 test（用户裁定，不再讨论）。

## 资料与历史

[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)。负结果细节：[C9](../archive/experiments/20260906_interval_evidence_transport/README.md)、[C6](../archive/experiments/20260905_latent_evidence_sequence/README.md)、[C7用户取消](../archive/experiments/20260906_context_witness/README.md)、[C4](../archive/experiments/20260904_null_token_cma/README.md)。[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。冻结Hate-follow-up引用不动；缓存出处见各data子目录PROVENANCE.md。
