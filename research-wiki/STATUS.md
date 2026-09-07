# 当前研究状态

截至 **2026-09-07 13:30 NZST**。依据：候选 3 修订 3（`experiments/20260907_c3_rev3_interval_evidence/`）两语料三 seed 搜索、17 组 × 3 seed × 2 语料消融、配对 bootstrap、证据打乱与 c 置零检验全部完成并 rsync 回本机；权威数字均引用本机 `runs/20260907_c3_rev3_interval_evidence/**/metrics.json`。

## 当前目标与结论

**候选 3 修订 3 两语料三 seed 过规则 8 确认；HCS 三项全面高于修订 2，HateMM AP 比修订 2 低 .027。17 组消融里 13 组两语料成立；区间证据 HMM 融合（第三批）两语料确认有效，骨干的两处新改动（query 门控、校准 c 只加 logit）在 HateMM 有害、HCS 有益，不能主张。** 候选 3 规则 9 的修改次数已用 2 次，剩 1 次；下一步交用户裁定。目标不变：两语料 pooled AP/ROC 过固定 baseline 表（三 seed 确认），三模块各有可主张贡献（规则 14(g)），方法统一、方法级超参少（修订 3：α、λ_block 两个，w_fine 已删）。within 只报告。

## 当前方法：候选 3（证据引导注意力）— 修订 2 为论文当前版本，修订 3 结果如下

修订 2：[experiments/20260904_evidence_guided_attention](../experiments/20260904_evidence_guided_attention/README.md)。修订 3：[experiments/20260907_c3_rev3_interval_evidence](../experiments/20260907_c3_rev3_interval_evidence/README.md)（第 0 节审稿批评对应改动，第 1 节方法，第 9 节规则 14 清单）。修订 3 = 修订 2 骨干 + 三处改动（视频级校准 c 只加 logit、query 门控逐头偏置、删 w_fine）+ 融合换成区间证据 HMM（30/4 真实区间合并成 32 段、连续时间转移、正例约束、归一化时间）。评测器未改。

## 最新权威结果（test，1 fps；三 seed 234/2025/3407，每 seed 20-trial 搜索最优 trial；来源 `runs/<exp>/<corpus>/seed<seed>/trial<best>/metrics.json`）

| 方法 | HateMM AP / ROC / within | HCS AP / ROC / within |
|---|---|---|
| 规则 8 门（最强训练 baseline） | .573 ± .033 / .807 ± .019 / .632 | .562 / .528 / .524 |
| 候选 1（rev4） | .657 / .842 / .646 | — |
| 候选 3 修订 2（`runs/20260904_evidence_guided_attention_rev2_noprune`，HCS `runs/20260904_evidence_guided_attention`） | .6678 ± .0097 / .8504 ± .0049 / .6233 ± .0150 | .6976 ± .0076 / .6843 ± .0095 / .5488 ± .0111 |
| **候选 3 修订 3**（`runs/20260907_c3_rev3_interval_evidence`） | .6409 ± .0174 / .8421 ± .0080 / .6310 ± .0101 | **.7045 ± .0053 / .6924 ± .0089 / .5678 ± .0040** |

修订 3 消融（三 seed 均值 full − arm，AP / ROC；两语料 ≥ .01 才可主张）：两语料成立 13 组：index_hmm（HateMM .037/.020，HCS .007/.024）、no_constraint、seconds_time、no_cell、no_bias、shared_bias、no_context、mean_prior、mean_prior_all、no_block、no_prior、no_cmal、no_verdict。不成立：no_qk_enc（HCS ≈ 0）、avce（HateMM .005/.009）、key_bias（HateMM −.010）、ctx_in_rep（HateMM −.023）。机制检验：打乱证据时间对应 HateMM 掉 .012/.013/within .023，HCS 不掉；c 推断时置零两语料变化 ±.009 内。自适应查询回放（0 次新 VLM 调用）见实验 README 第 8 节。

## 运行任务与监控

截至 2026-09-08 01:30。
- 候选 3 修订 4（修订 2 骨干 + 区间 HMM 融合，`experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/`，commit 0476eca，规则 6 复核 PASS）三 seed 搜索运行中：HateMM 三 seed 并行在 uoa-lab1，HCS 三 seed 并行在 uoa-lab3；`runs/20260908_c3_rev4_rev2_backbone_interval_hmm/<corpus>/seed<seed>/search.log`。用户裁定不做消融，只看总性能；判定规则见该实验 README 第 2 节 P2。
- 模块一自适应 VLM 查询（`experiments/20260908_adaptive_vlm_query/`）：文献调研完成（`docs/20260908_adaptive_query_survey.md`），代码写完，规则 4 与规则 6 复核进行中；等修订 4 出结果定骨干后开跑。
- 会话内 heartbeat 每 3 小时一次。本机 GPU 被他人占用。

## 下一步

1. 修订 4 三 seed 完成后按 P2 判定新起点（过则修订 4，否则回退修订 3）。
2. 模块一自适应查询：在起点骨干上跑 seed 234 两语料，按 README 2.3 的 E1（test 平均调用 ≤ 12）/ E2（pooled 不低于 34 次起点 − 一个 std）判定，不满足就按 2.4 迭代到满足为止（用户指令）。
3. 论文表述：机制主张"证据决定从哪聚合"限定 HateMM；HCS 写 limitation。B.2 引 WavLM 门控相对位置偏置，融合引 Dugong / CHMM / CT-HMM / EM-MIL（`REVIEW_RULE4.md`）。
4. 搜索目标继续按 test（用户裁定，不再讨论）。

## 资料与历史

[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)。负结果细节：[C9](../archive/experiments/20260906_interval_evidence_transport/README.md)、[C6](../archive/experiments/20260905_latent_evidence_sequence/README.md)、[C7用户取消](../archive/experiments/20260906_context_witness/README.md)、[C4](../archive/experiments/20260904_null_token_cma/README.md)。[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。冻结Hate-follow-up引用不动；缓存出处见各data子目录PROVENANCE.md。
