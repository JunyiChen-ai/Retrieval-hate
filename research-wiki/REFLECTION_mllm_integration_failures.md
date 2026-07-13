# REFLECTION: 14 条 MLLM 集成失败路线的共同点(2026-07-13)

触发:用户 2026-07-13 指令「这些失败的有什么共同点么 及时的reflection啊 如果一条线一直失败就换思路 灵感枯竭的时候多看文献 不同领域的顶会文献」。
本文档 = 主对话的结构性 reflection;配套文献综述见 `LITERATURE_mllm_integration_2026-07-13.md`。

## 1. 共同 schema:低带宽、决策侧的旁路注入

全部 14 条失败路线(P1–P5、P9b、P10 族、P11、TARC V1/V2/V3、自动记忆修复)是**同一个 schema 的变体**:把 MLLM 输出压缩成每视频几个 bit 到几十个 bit 的旁路信号(先验标量 / 邻居排序 / 段权重 / schema 字段 / target 类别 / 段分数),注入一个「冻结表征 + 小 head + ~1k 样本」的管线的**决策侧**。

| 路线 | 注入信号 | 带宽/视频 | 注入位置 |
|---|---|---|---|
| P1 先验重校准 | 1 标量 | ~bits | 决策(head 校准) |
| P2/P2b 邻居重排(含 32B) | 1 排序 | ~bits | 决策(kNN 排序) |
| P3 证据密度池化 | ~8 段权重 | ~几十 bits | 决策侧池化 |
| P4 schema 蒸馏 | 数个结构化字段 | ~几十 bits | 特征拼接(低维) |
| P5 反事实孪生 | 合成样本对 | 低 | 训练对构造 |
| P9b head↔memory 耦合 | 分数再分配 | 低 | 决策耦合 |
| P10/P10b/P10c logit 融合 | 1 logit | ~bits | 决策(融合) |
| P11 弱监督定位标签 | 每段 1 分数 | ~几十 bits | 训练标签 |
| TARC V1/V2/V3 | 1 target 类别 | ~3 bits | 检索图条件化 |
| 自动记忆两票修复 | 删除票 | ~bits | 记忆库编辑 |

## 2. 三条可量化的共性诊断

**D1 — 杀死它们的是冗余,不是质量。** 所有 gate 测的都是信号的*边际*质量(parse 率、对 GT 的 macro-F1、probe AUC),全部通过;从未有 gate 测*条件*信息量(信号在冻结特征之外还能否预测标签)。「probe 过了、训练打平」至少重复 4 次(P3、P11、TARC G2→G3、P4)——同一句话:信号很准,但特征里早就有了。

**D2 — 带宽错配。** 项目唯一稳过 +3 acc 的杠杆是 encoder swap(CLIP→Qwen2.5-VL-7B 冻结特征,HateMM +5.3~5.6 acc,3/3 seed 双协议;exp-encoder-3seed.md,commit 040adb8)——MLLM 以*表征本体*进场。增益来自 representation,从不来自 decision-level 装饰。跨领域文献独立复证同一规律(见文献综述 §3:2024–2026 HateMM 诚实 SOTA 全部由加表征通道驱动,LMM-as-reasoner 分类器反而落后监督融合 frontier;RecSys 的 LLM-as-feature vs LLM-as-ranker 结论同构)。

**D3 — 小测试集让弱效应不可测量。** test 150–300 样本、78-dev val 选择 → 噪声地板 ±1–2 acc 点。TARC 的 +0.0347 假阳性 cell、archive-as-key 撤稿都由此产生。预期效应 < 3 pt 的决策侧 trick 在本 regime **结构性**低于可测量线。

## 3. 推论:仍然活着的路线形态只有三种

1. **表征级**:修复 encoder swap 在 MHC-EN 的失败(= 直接满足「≥2 数据集」);LoRA/QLoRA 端到端微调 MLLM encoder(RA-HMD 直接先例,EMNLP 2025);稀疏注意力头特征挖掘(SAV,ICCV 2025)。
2. **新输入通道(高带宽)**:管线目前只有 16 帧视觉 + title + ASR。文献中 HateMM 最大的已验证单通道增益是 OCR 通道(MM-HSD +2.6 M-F1),**但用户 2026-07-13 裁定:不加 OCR(判定无用)——此路线关闭,仅留作 SOTA 校准证据**。剩余可行形态:MLLM 世界知识/推理密集文本(实体、暗语、隐含 target 推理,非转写复述、非 OCR),必须先过 §4 条件信息 gate(P4 冗余陷阱的直接暴露面)。
3. **数据级**:MLLM 伪标注无标注外部视频扩充*表征训练*(不是直接进 kNN 投票池——那是决策侧,文献与我们自己的负结果都反对)。

## 4. 新制度:条件信息 Gate(G0-cond,强制)

任何未来辅助信号路线,在花 GPU 之前必须过零成本条件探针(完整配方见文献综述 §5):
- 条件 V-information / MDL 探针:比较 g(Z) vs g'([Z,A]) 的 codelength(不是 accuracy),探针容量与实际 head 匹配;
- bits→acc 换算后投影增益必须 > +3 acc + 噪声带,多 seed bootstrap CI 排除 0;
- 先跑 oracle 上限版(gold 版信号,合规:gold 仅用于 probing):oracle 条件增益 < +3 → 整个信号族直接毙掉。
此 gate 若 6 个月前存在,P1–P5、TARC、P11 一张 GPU 卡都不用烧。

**校准强制项(2026-07-14 增补,源于 C3 探针判决被推翻的教训,refine-logs/C3_PROBE_VERDICT_REVIEW.md):** 任何 G0-cond 探针必须内置 label-oracle 校准 arm(把 gold 标签本身当 A 喂入),且该 arm 必须达到 ~100% 的 Fano headroom;达不到即判定探针机器无效(常见病因:对 [Z,A] 全列共用重 L2 会把辅助列系数压死——修法:Z 按其最优正则处理,A 以未惩罚/弱惩罚的原始编码进入)。凡校准 arm 未过,任何"信号被 cap"的负判决一律不得接受。

## 5. A 线(lb_scgp_global)处置

按 D2 标准,A 线证书 = 每视频 8 个 observables,带宽同样偏低;与 P1–P5 的区别是作用点为全局 Gram 几何(表征侧),不同构但先验被本 reflection 压低。处置:M1 缓存已完成(jobs 13012/13013 COMPLETED),M2/M3 成本小且有预注册干净判决 → **给 A 线一次 M3 判决机会,C 线(文献候选)并行排队;M3 若败零空转切换**。不再出现单线多轮消耗。

## 6. 数字出处(供归档审计)

encoder swap +5.3~5.6 / 双协议 3/3 seed:`research-wiki/experiments/exp-encoder-3seed.md`(commit 040adb8)。HateMM CLIP floor 0.8279/0.8172:erratum commit 66012e9。TARC 全部数字:`research-wiki/experiments/exp-tarc-t0.md` §10–§12。P1–P11 结论:`research-wiki/CAMPAIGN_mllm_method_role.md` 及 TERMINUS 文档。ZH floor 0.8537±0.012、EN 0.79–0.81:novelty-scope 既有记录。
