# 当前研究状态

截至 **2026-09-07 10:40 NZST**。依据：候选 3 修订 3（76ef6f0）HCS 三 seed 搜索与 HateMM seed 234 搜索完成并回传，消融与 HateMM 其余 seed 运行中；候选 3 外部审稿完成（4/10，`experiments/20260904_evidence_guided_attention/REVIEW_NOVELTY_GPT6ASTRA.md`）；候选 3（证据引导注意力，修订 2 模型）HateMM 三 seed 在不剪 within 的规则下重跑完成并回传，12 臂消融三 seed 完成并回传；HCS 沿用修订 2 三 seed。权威数字均引用本机 runs 原评测。

## 当前目标与结论

**候选 3 两语料三 seed 过规则 8 确认，12 臂消融中 11 臂两语料成立（唯一不成立：证据进 q/k 编码，HCS ≈ 0）；VLM / 骨干 / 融合三模块各有可主张的部件。候选 3 为当前方法。** 目标不变：两语料 pooled AP/ROC 过固定 baseline 表（三 seed 确认），三模块各有可主张贡献（规则 14(g)：三 seed 均值降 ≥ .01、两语料；不要求每 seed 都降），方法统一、方法级超参少（现在 α、w_fine、λ_block 三个）。within 只报告，不剪枝、不作门。

候选 3 第 7.2 节的"HateMM 不过 within 下限、归档"判定撤销：不剪枝后 HateMM 三 seed 均值 .668 / .850，是所有候选里 pooled 最高的（候选 1 .657 / .842，精简版 v2 .632 / .835），within .623 比候选 1 低 .023（只报告）。

## 当前方法：候选 3（证据引导注意力，修订 2；修订 3 搜索中）

[experiments/20260904_evidence_guided_attention](../experiments/20260904_evidence_guided_attention/README.md)（方法第 1、7 节，结果第 8.1 节）。候选 1 的 VLM 裁定 + HMM 后验 + 块级 MIL + 先验不变；骨干改为证据只进跨模态注意力的 query/key（四格嵌入 + 两列线性）、逐头 key 偏置、视频级证据上下文，内容表示保持纯内容；去掉 EMA。搜索 6 标量（lr、max_seqlen、λ_cma、α、w_fine、λ_block）。评测器未改。

## 最新权威结果

顺序 **AP / ROC / within**，test，按 test 选 trial 的开发期证据。

| 候选/语料 | 结果 | 来源 |
|---|---|---|
| **候选 3 HateMM 三 seed（不剪枝）** | **.6678±.0097 / .8504±.0049 / .6233±.0150**（best trial 17/8/13） | [搜索](../runs/20260904_evidence_guided_attention_rev2_noprune/hatemm/)，[两语料汇总](../runs/20260904_evidence_guided_attention_rev2_noprune/ablations/three_seed_summary_both_corpora.json) |
| **候选 3 HCS 三 seed** | **.6976±.0076 / .6843±.0095 / .5488±.0111**（best trial 13/12/16） | [搜索](../runs/20260904_evidence_guided_attention_rev2/hateclipseg/) |
| 候选 3 消融（12 臂 × 3 seed × 2 语料） | 去掉整个证据引导注意力：HMM −.044/−.019，HCS −.009/−.025；去 VLM 裁定：−.159/−.090，−.100/−.118；q/k 编码 HCS −.001 | 同上汇总 |
| 精简版 v2（候选 1 精简）三 seed | HMM .632/.835/.625；HCS .706/.690/.565；外部审稿 3/10 | [目录](../runs/20260906_hier_evidence_clean_v2/) |
| C1 三 seed（within 剪枝下搜索） | HMM .657±.013/.842±.005/.646±.004；HCS .699±.006/.681±.016/.553±.007 | [搜索](../runs/20260903_hier_evidence_mil/) |
| 规则 8 门 | HMM .573/.807；HCS .562/.528；within 参考 .632/.524 | `docs/duplex/OFFICIAL_VAL_RESULTS.md` |

## 运行任务与监控

截至 2026-09-07 10:40。候选 3 修订 3（`experiments/20260907_c3_rev3_interval_evidence/`，commit 76ef6f0）：
- HCS 三 seed 搜索完成并回传：AP .7045 / ROC .6924 / within .5678（三 seed 均值；单 seed 见实验 README 第 6 节），高于修订 2 的 .6976 / .6843 / .5488。HCS 三 seed 17 组消融完成并回传（`runs/20260907_c3_rev3_interval_evidence/ablations/hateclipseg/seed<seed>/`，配对 bootstrap 同目录 `paired_bootstrap.json`）：17 组里 16 组三 seed 均值 AP 或 ROC 下降 ≥ .01，只有 no_qk_enc 不达。
- HateMM seed 234 完成并回传：AP .6209 / ROC .8346 / within .6195，过规则 8 筛选但低于修订 2 的 .668 / .850；seed 2025 搜索与 seed 234 消融在 lab1、seed 3407 搜索在 lab3 运行中（`.../hatemm/seed<seed>/search.log`，`.../ablations/hatemm/seed234/`）。
- HCS 证据打乱检验完成（本机 CPU）：打乱证据时间对应不掉分（实验 README 第 7 节）。
查看：`ssh uoa-lab1 tail -f ~/Retrieval-hate/runs/20260907_c3_rev3_interval_evidence/hatemm/seed2025/search.log`。本机 GPU 被他人占用。会话内 heartbeat 每 3 小时检查一次。

## 下一步

1. 修订 3 = 骨干改动（视频级校准只加 logit、query 门控逐头偏置、删 w_fine）+ 区间证据 HMM 融合，一次搜索流程：seed 234 规则 8 筛选 → seed 2025/3407 → 17 臂消融 × 3 seed × 2 语料 → 配对 bootstrap、证据打乱检验 → 规则 14 清单 → 汇报。预注册预期与臂表见实验 README 第 2、3 节。
2. 待用户裁定（不阻塞）：减少 VLM 调用预算（4 / 22 次）下训练一次完整模型确认；视频随机效应的 M 步（回放里 σ 发散，已弃）。
3. 搜索目标继续按 test（用户裁定，不再讨论）。

## 资料与历史

[研究规则](../RESEARCH_ITERATION_RULES.md)、[固定baseline表](../docs/duplex/OFFICIAL_VAL_RESULTS.md)、[评测协议](../docs/duplex/FRAME_EVAL_PROTOCOL.md)。负结果细节：[C9](../archive/experiments/20260906_interval_evidence_transport/README.md)、[C6](../archive/experiments/20260905_latent_evidence_sequence/README.md)、[C7用户取消](../archive/experiments/20260906_context_witness/README.md)、[C4](../archive/experiments/20260904_null_token_cma/README.md)。[旧状态索引](../archive/research-wiki/STATUS_20260905_before_cleanup.md)。冻结Hate-follow-up引用不动；缓存出处见各data子目录PROVENANCE.md。
