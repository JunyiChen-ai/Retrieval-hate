# Test accuracy addendum(裸头 vs trick 三编码器 ensemble)

2026-08-12。**零训练、纯离线**,只读已冻结产物;未触碰任何训练/推理。
项目历来只报 macro-F1,本文补上同口径的 **test accuracy**。

## 数据来源与口径

| 配置 | 分数来源 | 阈值口径 |
|---|---|---|
| **trick(三编码器 ensemble)** | `idea-stage/r4_pilot1.json` → `datasets[<ds>]['per_seed'][i]['scores'][<method>]`,3 seeds | R4-1 在 **val** 上选的阈值未存盘;按 `idea-stage/r5_phase_a.py` 的 `recover()` **反演已存的 test macro-F1** 恢复阈值,accuracy 在同一阈值下计算 |
| **裸头(单编码器 L1 BCE + I1 头输出)** | `logging/runs/rgcl_ablation/logs/<ENC>_<DS>_L1_s{0,1,2}.trainlog` | 无需恢复:trainlog **同一行**同时打印该 epoch 的 `acc` 与 `macroF1`,同一决策规则(head 默认 0.5);epoch 由 `scripts/rgcl_ablation_analyze.py` 的冻结规则(warmup=5,argmax dev acc→roc)选出,并在脚本内 assert 与 `parse_run()` 的选择逐位一致 |

每数据集的方法/编码器选取(均为各自表内 F1 最优,选取在本次计算之前已固定):

- trick:HateMM=`mlp`、MHC(EN)=`mean_logit`、MHC_zh=`logistic`、ImpliHateVid=`logistic`(§8.7)
- 裸头:HateMM=LoRA、MHC-EN=Qwen、MHC-ZH=LoRA、ImpliHateVid=CLIP(§2 消融;ImpliHateVid 的 CLIP 与 Qwen 并列 0.9118,取 CLIP)

标签取自 `data/CLIP_Embedding/<ds>/test_seen_openai_clip-vit-large-patch14-336_HF.pt`,
并 assert 与 `r4_pilot1.json` 里存的 `y` 完全一致。

## 主表:test accuracy(mean ± std over seeds 0/1/2)

| 数据集 | n_test | 裸头 acc | trick acc | Δ(trick−裸头) | HVGuard 论文 acc(自切 split,**不可直接比**) |
|---|---|---|---|---|---|
| HateMM | 215 | **0.8837 ± 0.0047** | 0.8791 ± 0.0093 | −0.0047 | 0.8563 |
| MHC (EN) | 161 | 0.7681 ± 0.0072 | **0.8012 ± 0.0108** | +0.0331 | 0.8539 |
| MHC_zh | 149 | 0.8255 ± 0.0134 | **0.8479 ± 0.0103** | +0.0224 | 0.8603 |
| ImpliHateVid | 401 | 0.9119 ± 0.0038 | **0.9277 ± 0.0025** | +0.0158 | — |

> **HVGuard 数字为该论文自行切分的 split 上的 accuracy**,与本项目的 split / 标签二值化 /
> 协议均不同,**不构成同表可比**,仅作量级参照列出。

逐 seed accuracy:

| 数据集 | 裸头 seed0/1/2 | trick seed0/1/2 |
|---|---|---|
| HateMM | 0.8884 / 0.8837 / 0.8791 | 0.8698 / 0.8791 / 0.8884 |
| MHC (EN) | 0.7640 / 0.7764 / 0.7640 | 0.8075 / 0.8075 / 0.7888 |
| MHC_zh | 0.8389 / 0.8121 / 0.8255 | 0.8389 / 0.8456 / 0.8591 |
| ImpliHateVid | 0.9077 / 0.9127 / 0.9152 | 0.9252 / 0.9302 / 0.9277 |

## 核对项:恢复的阈值 / 选出的 epoch 下,macro-F1 必须复现已汇报数字

**8/8 全部复现到 4 位小数。**

| 数据集 | 裸头 F1(重算) | 裸头 F1(已汇报) | ✔ | trick F1(重算) | trick F1(已汇报) | ✔ |
|---|---|---|---|---|---|---|
| HateMM | 0.8774 | 0.8774 | ✅ | 0.8732 | 0.8732 | ✅ |
| MHC (EN) | 0.7331 | 0.7331 | ✅ | 0.7776 | 0.7776 | ✅ |
| MHC_zh | 0.7821 | 0.7821 | ✅ | 0.8183 | 0.8183 | ✅ |
| ImpliHateVid | 0.9118 | 0.9118 | ✅ | 0.9276 | 0.9276 | ✅ |

`recover()` 内部另有硬断言:反演阈值网格上必须存在使 macro-F1 与存盘值相差 <1e-9 的点,
否则直接 assert 失败;四个数据集 × 3 seeds 共 12 次恢复全部通过。
裸头一侧另有断言:重解析选出的 epoch 与 `parse_run()` 冻结选择逐位一致(12/12 通过)。

## 读数

- **accuracy 与 macro-F1 的排序在三个数据集上一致**,只有 HateMM 反号:裸头 F1 0.8774 > trick 0.8732,accuracy 也是裸头 0.8837 > trick 0.8791 —— 即"HateMM 上 ensemble 并不比最好的单编码器裸头强"这一结论在 accuracy 口径下同样成立。
- 两个 MHC 上 accuracy 的**绝对值**显著高于 macro-F1(MHC-EN trick:acc 0.8012 vs F1 0.7776;MHC-ZH trick:0.8479 vs 0.8183),类别不平衡下 accuracy 会系统性抬高读数;同时 trick 相对裸头的**增益在 accuracy 口径下变小**(MHC-EN:F1 +0.0445 → acc +0.0331;MHC-ZH:F1 +0.0362 → acc +0.0224)。**对外汇报仍应以 macro-F1 为主列**,accuracy 只作补充列。
- ImpliHateVid 上 accuracy ≈ macro-F1(0.9277 vs 0.9276),类别近乎平衡。
