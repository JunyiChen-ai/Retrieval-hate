# DEMO v2: ZH 记忆编辑定向性复测(archive prompt v2)

> v1 诚实负结果的复测:ZH 组级删除 0 翻转,归因于 v1 档案 target 字段召回过低 (train 583 条仅 6 条非空)。本文用 **冻结的 v1 获胜 ckpt + v2 档案键**重跑同一协议 (topk=20, arithmetic, α=0.25):多 seed post-mortem + sha1 审计已确认 kNN 键通道不参与训练,换键无需重训,且隔离了重训混淆。切片样本量仍然很小,只报绝对数,不做显著性声明。

- ckpt: v1 获胜 job 12207 val-selected epoch 18 (`best_model_18_0.8717948717948718.pt`),档案键 = prompt v2。
- 记忆库 N=579,test N=149;复现门 = v1 键 baseline 应等于训练日志 0.8523/0.8270 (acc/macro-F1)。

## 切片定义: LGBTQ+ by target_groups 字段(claim 本体;v1 该切片 0 翻转)

- LGBTQ+ 记忆条目 20 条,LGBTQ+ test 切片 11 条。

| 行 | 删除数 | 整体 acc | 整体 macro-F1 | 切片 acc (n=11) | 其余 acc (n=138) | 翻转(切片/其余, vs v2键基线) |
|---|---|---|---|---|---|---|
| baseline v1 keys (reproduction gate) | 0 | 0.8523 | 0.8270 | 0.9091 | 0.8478 | — |
| baseline v2 keys (no edit) | 0 | 0.8255 | 0.7875 | 0.8182 | 0.8261 | — |
| (a) remove LGBTQ+-targeting memory [lgbt_target_field] | 20 | 0.8121 | 0.7679 | 0.8182 | 0.8116 | 0 / 2 |
| (c) random control, 5 seeds | 20 | 0.8228 [0.8188,0.8255] | 0.7842 [0.7778,0.7875] | 0.8182 [0.8182,0.8182] | 0.8232 [0.8188,0.8261] | 0.0 [0,0] / 0.4 [0,1] |

- (a) 删除条目的 gt 标签构成: {1: 17, 0: 3}

## 切片定义: 女性 by target_groups 字段(ZH 最大目标族,记忆条目量级≈EN v1 演示)

- 女性 记忆条目 63 条,女性 test 切片 10 条。

| 行 | 删除数 | 整体 acc | 整体 macro-F1 | 切片 acc (n=10) | 其余 acc (n=139) | 翻转(切片/其余, vs v2键基线) |
|---|---|---|---|---|---|---|
| baseline v1 keys (reproduction gate) | 0 | 0.8523 | 0.8270 | 0.7000 | 0.8633 | — |
| baseline v2 keys (no edit) | 0 | 0.8255 | 0.7875 | 0.7000 | 0.8345 | — |
| (a) remove women-targeting memory [women_target_field] | 63 | 0.7987 | 0.7437 | 0.6000 | 0.8129 | 1 / 5 |
| (c) random control, 5 seeds | 63 | 0.8242 [0.8121,0.8322] | 0.7867 [0.7679,0.7971] | 0.6800 [0.6000,0.7000] | 0.8345 [0.8273,0.8417] | 0.2 [0,1] / 2.4 [1,4] |

- (a) 删除条目的 gt 标签构成: {1: 44, 0: 19}

## 切片定义: LGBTQ+ by 全档案文本关键词(v1 同款,可比性)

- LGBTQ+ 记忆条目 16 条,LGBTQ+ test 切片 12 条。

| 行 | 删除数 | 整体 acc | 整体 macro-F1 | 切片 acc (n=12) | 其余 acc (n=137) | 翻转(切片/其余, vs v2键基线) |
|---|---|---|---|---|---|---|
| baseline v1 keys (reproduction gate) | 0 | 0.8523 | 0.8270 | 0.9167 | 0.8467 | — |
| baseline v2 keys (no edit) | 0 | 0.8255 | 0.7875 | 0.8333 | 0.8248 | — |
| (a) remove LGBTQ+-targeting memory [lgbt_fulltext_v1style] | 16 | 0.8188 | 0.7778 | 0.8333 | 0.8175 | 0 / 1 |
| (c) random control, 5 seeds | 16 | 0.8228 [0.8188,0.8255] | 0.7842 [0.7778,0.7875] | 0.8333 [0.8333,0.8333] | 0.8219 [0.8175,0.8248] | 0.0 [0,0] / 0.4 [0,1] |

- (a) 删除条目的 gt 标签构成: {1: 12, 0: 4}

(数字由 scripts/analysis/memory_editing_demo_v2.py 生成;结论正文由报告撰写时人工核对。)
