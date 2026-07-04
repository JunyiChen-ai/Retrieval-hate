# DEMO: kNN 记忆编辑 — 档案字段让记忆库可定向编辑(capability demo)

> **诚实条款**:test 切片样本量很小(LGBTQ+ 切片 EN n=24、ZH n=12);本文只报告绝对数,不做显著性声明。定位是**能力演示**——证明档案字段使记忆库支持按语义划片的定向编辑——而非性能 claim。

## 场景与操作

- 场景: 平台需要 (a) "移除针对某群体的整类记忆"(演示组: LGBTQ+,档案中最高频目标族)或 (b) "下架被误标条目"(W2 发现的 `XScP1AiMkNM` 牛油果酱、`QvPp8Q7QhWE` 数钱,以及同 mechanism 的低置信条目)。
- 记忆库 = 获胜配置(archive-kNN α=0.25)的 train 键;编辑 = 直接从 faiss 索引删除对应train 条目,**纯 CPU,零训练**。查询协议与训练日志逐位一致(topk=20、arithmetic 加权、相似度符号投票)。
- 条目归属完全由**档案字段**决定(target_groups / neutral_summary / modality_cues 的关键词匹配;低置信 = label=harmful 但档案报 无 target + explicitness=none + mechanism⊆{coded_language},该规则恰好同时命中两条 W2 噪声样本)。gt 标签与原始文本不参与编辑决策。
- 对照: (c) 每次编辑配等量随机删除(5 seeds,报 mean [min,max])。

## MHC (EN)

- ckpt: job 12210 val-selected epoch 24 (`best_model_24_0.7875.pt`)
- 记忆库 N=549,test N=161;LGBTQ+ 记忆条目 91 条,LGBTQ+ test 切片 24 条,低置信(噪声样)记忆条目 14 条。
- 复现门:baseline acc 0.8075 / macro-F1 0.7626,训练日志记录 0.8075 / 0.7626。

| 编辑 | 删除数 | 整体 acc | 整体 macro-F1 | LGBTQ+切片 acc (n=24) | 其余切片 acc (n=137) | 翻转(切片/其余) |
|---|---|---|---|---|---|---|
| baseline (no edit) | 0 | 0.8075 | 0.7626 | 0.8333 | 0.8029 | — |
| (a) remove LGBTQ+-targeting memory | 91 | 0.7764 | 0.7261 | 0.7917 | 0.7737 | 3 / 8 |
| (c) random control for (a), 5 seeds | 91 | 0.7950 [0.7764,0.8075] | 0.7408 [0.7144,0.7559] | 0.8250 [0.7917,0.8333] | 0.7898 [0.7737,0.8029] | 0.2 [0,1] / 5.8 [4,8] |
| (b) remove W2 noisy ids + same-mechanism low-confidence | 14 | 0.8199 | 0.7748 | 0.8333 | 0.8175 | 0 / 2 |
| (b') remove ONLY the 2 W2 noisy ids | 2 | 0.8199 | 0.7748 | 0.8333 | 0.8175 | 0 / 2 |
| (c) random control for (b), 5 seeds | 14 | 0.8050 [0.7950,0.8137] | 0.7587 [0.7401,0.7717] | 0.8333 [0.8333,0.8333] | 0.8000 [0.7883,0.8102] | 0.0 [0,0] / 2.0 [0,4] |

- (b) 删除条目: `YNf2tZgh4WM`, `TRFp4a4lD0o`, `My5PVJLP6Bg`, `QvPp8Q7QhWE`, `8Pim0TnLQDQ`, `2ytDPK74q28`, `aeOm9oT0_qk`, `hKwgFaE7fbQ`, `6hFEc1MLZC0`, `lNCfDw80YSQ`, `dcrX2-oto8Y`, `EU-dip0ITa4`, `XScP1AiMkNM`, `Z2Cs5Oqm9iU`
- (a) 删除条目的 gt 标签构成: {0: 71, 1: 20}

## MHC_zh (ZH)

- ckpt: job 12207 val-selected epoch 18 (`best_model_18_0.8717948717948718.pt`)
- 记忆库 N=579,test N=149;LGBTQ+ 记忆条目 23 条,LGBTQ+ test 切片 12 条,低置信(噪声样)记忆条目 9 条。
- 复现门:baseline acc 0.8523 / macro-F1 0.8270,训练日志记录 0.8523 / 0.8270。

| 编辑 | 删除数 | 整体 acc | 整体 macro-F1 | LGBTQ+切片 acc (n=12) | 其余切片 acc (n=137) | 翻转(切片/其余) |
|---|---|---|---|---|---|---|
| baseline (no edit) | 0 | 0.8523 | 0.8270 | 0.9167 | 0.8467 | — |
| (a) remove LGBTQ+-targeting memory | 23 | 0.8389 | 0.8090 | 0.9167 | 0.8321 | 0 / 2 |
| (c) random control for (a), 5 seeds | 23 | 0.8510 [0.8389,0.8591] | 0.8252 [0.8090,0.8359] | 0.9000 [0.8333,0.9167] | 0.8467 [0.8321,0.8540] | 0.2 [0,1] / 0.8 [0,2] |
| (b) rule-only low-confidence takedown (no W2-flagged ids in this dataset) | 9 | 0.8389 | 0.8090 | 0.8333 | 0.8394 | 1 / 1 |
| (c) random control for (b), 5 seeds | 9 | 0.8456 [0.8255,0.8523] | 0.8179 [0.7904,0.8270] | 0.9000 [0.8333,0.9167] | 0.8409 [0.8248,0.8467] | 0.2 [0,1] / 0.8 [0,3] |

- (b) 删除条目: `BV1go4y1J7MX`, `BV1de41127of`, `BV1T44y1R7fk`, `BV1oK41127n4`, `BV1ou4y1n7vU`, `BV1i34y1p7jU`, `BV1Bw411z71n`, `BV1wX4y1R7WZ`, `BV158411i7rQ`
- (a) 删除条目的 gt 标签构成: {1: 18, 0: 5}

## 结论(定向性)

1. **复现门通过(逐位一致)**:两个数据集的 baseline 与训练日志完全一致 (EN 0.8075/0.7626, ZH 0.8523/0.8270),说明编辑实验操作的确实是获胜配置本身的推理路径,而非近似复刻。
2. **EN 组级删除是定向的(以翻转率计)**:删除 91 条 LGBTQ+ 记忆后,LGBTQ+ 切片翻转 3/24 (12.5%),其余切片 8/137 (5.8%);等量随机删除的切片翻转仅 0.2/24 (0.8%),其余 5.8/137 (4.2%)。即:**定向删除把翻转集中到目标切片(切片翻转率约为随机对照的 15 倍),随机删除的扰动则几乎全部落在切片外**。切片 acc 0.8333→0.7917、其余 0.8029→0.7737——其余切片的下降幅度落在随机对照包络内 (rest acc 随机 min=0.7737),而切片下降超出随机包络均值。删 16.5% 的记忆整体掉约 3 个点,属预期的容量损失。
3. **ZH 组级删除无定向效果(诚实负结果)**:LGBTQ+ 切片 acc 编辑前后不变 (0.9167→0.9167, 0 翻转)。原因与审计发现一致 (AUDIT #5):ZH 档案几乎不填 target 字段(train 583 条中仅 6 条非空),关键词匹配只召回 23 条记忆——**编辑的定向性受档案质量上限约束**,这是"档案让记忆可审计"主张的边界条件,不是反例:字段缺失导致的是"编辑不到",而非"错编辑"。
4. **误标下架:删 2 条 W2 噪声条目,整体不降反升**:EN 删除 14 条低置信条目后 acc 0.8075→0.8199、macro-F1 0.7626→0.7748,超过全部 5 个随机对照种子 (acc max 0.8137);且 (b') 只删 `XScP1AiMkNM`+`QvPp8Q7QhWE` 两条即可复现全部增益 (0.8199/0.7748),其余 12 条规则命中条目增删无影响。**两条被人工标记的疑似噪声条目确实是有害记忆,删除即修复,零训练。**
5. **但"低置信规则"单独使用并不安全**:ZH 无人工种子、纯规则删 9 条反而 0.8523→0.8389 (低于随机对照均值 0.8456);且 EN 规则命中的 `TRFp4a4lD0o` 经审计确认标题确有毒性 (AUDIT: whitewash)。**规则适合做"隔离候选队列"供人工复核,不能当自动删除策略。**
6. **对可审计性 claim 的支撑**:本演示证明了 (i) 记忆条目可被档案字段语义寻址;(ii) 删除是外科手术式的(EN 定向、2 条噪声条目精确摘除);(iii) 全程 CPU、秒级、零训练——这是 trained-MoE 头结构上做不到的操作。未证明的:切片级效应的统计显著性(n 太小)、ZH 的组级编辑(受档案 target 召回率限制)。

(数字由 scripts/analysis/memory_editing_demo.py 生成;原始 JSON 已存 research-wiki/DEMO_memory_editing_results.json;结论正文为人工撰写,引用的审计编号见 research-wiki/AUDIT_archive_faithfulness.md。)
