# DEMO v2: ZH 记忆编辑定向性复测(archive prompt v2)

> v1 诚实负结果的复测:ZH 组级删除 0 翻转,归因于 v1 档案 target 字段召回过低 (train 583 条仅 6 条非空)。本文用 **冻结的 v1 获胜 ckpt + v2 档案键**重跑同一协议 (topk=20, arithmetic, α=0.25):多 seed post-mortem + sha1 审计已确认 kNN 键通道不参与训练,换键无需重训,且隔离了重训混淆。切片样本量仍然很小,只报绝对数,不做显著性声明。

- ckpt: v1 获胜 job 12210 val-selected epoch 24 (`best_model_24_0.7875.pt`),档案键 = prompt v2。
- 记忆库 N=549,test N=161;复现门 = v1 键 baseline 应等于训练日志 0.8075/0.7626 (acc/macro-F1)。

## 切片定义: LGBTQ+ by target_groups 字段(claim 本体;v1 该切片 0 翻转)

- LGBTQ+ 记忆条目 74 条,LGBTQ+ test 切片 14 条。

| 行 | 删除数 | 整体 acc | 整体 macro-F1 | 切片 acc (n=14) | 其余 acc (n=147) | 翻转(切片/其余, vs v2键基线) |
|---|---|---|---|---|---|---|
| baseline v1 keys (reproduction gate) | 0 | 0.8075 | 0.7626 | 0.7857 | 0.8095 | — |
| baseline v2 keys (no edit) | 0 | 0.8012 | 0.7462 | 0.8571 | 0.7959 | — |
| (a) remove LGBTQ+-targeting memory [lgbt_target_field] | 74 | 0.8012 | 0.7565 | 0.8571 | 0.7959 | 2 / 4 |
| (c) random control, 5 seeds | 74 | 0.7988 [0.7950,0.8075] | 0.7429 [0.7322,0.7522] | 0.8571 [0.8571,0.8571] | 0.7932 [0.7891,0.8027] | 0.0 [0,0] / 3.2 [1,5] |

- (a) 删除条目的 gt 标签构成: {0: 44, 1: 30}

## 切片定义: 女性 by target_groups 字段(ZH 最大目标族,记忆条目量级≈EN v1 演示)

- 女性 记忆条目 45 条,女性 test 切片 12 条。

| 行 | 删除数 | 整体 acc | 整体 macro-F1 | 切片 acc (n=12) | 其余 acc (n=149) | 翻转(切片/其余, vs v2键基线) |
|---|---|---|---|---|---|---|
| baseline v1 keys (reproduction gate) | 0 | 0.8075 | 0.7626 | 0.3333 | 0.8456 | — |
| baseline v2 keys (no edit) | 0 | 0.8012 | 0.7462 | 0.5000 | 0.8255 | — |
| (a) remove women-targeting memory [women_target_field] | 45 | 0.8012 | 0.7462 | 0.5000 | 0.8255 | 0 / 0 |
| (c) random control, 5 seeds | 45 | 0.7963 [0.7888,0.8075] | 0.7398 [0.7303,0.7484] | 0.5000 [0.5000,0.5000] | 0.8201 [0.8121,0.8322] | 0.0 [0,0] / 2.0 [0,3] |

- (a) 删除条目的 gt 标签构成: {0: 17, 1: 28}

## 切片定义: LGBTQ+ by 全档案文本关键词(v1 同款,可比性)

- LGBTQ+ 记忆条目 100 条,LGBTQ+ test 切片 27 条。

| 行 | 删除数 | 整体 acc | 整体 macro-F1 | 切片 acc (n=27) | 其余 acc (n=134) | 翻转(切片/其余, vs v2键基线) |
|---|---|---|---|---|---|---|
| baseline v1 keys (reproduction gate) | 0 | 0.8075 | 0.7626 | 0.8148 | 0.8060 | — |
| baseline v2 keys (no edit) | 0 | 0.8012 | 0.7462 | 0.8519 | 0.7910 | — |
| (a) remove LGBTQ+-targeting memory [lgbt_fulltext_v1style] | 100 | 0.7888 | 0.7446 | 0.8519 | 0.7761 | 2 / 6 |
| (c) random control, 5 seeds | 100 | 0.8000 [0.7888,0.8075] | 0.7442 [0.7303,0.7522] | 0.8519 [0.8519,0.8519] | 0.7896 [0.7761,0.7985] | 0.0 [0,0] / 2.2 [1,4] |

- (a) 删除条目的 gt 标签构成: {0: 69, 1: 31}

(数字由 scripts/analysis/memory_editing_demo_v2.py 生成;结论正文由报告撰写时人工核对。)
