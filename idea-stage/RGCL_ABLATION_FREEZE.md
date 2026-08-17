# RGCL 检索管线部件归因消融 — 冻结判决书

Frozen: 2026-08-09, **before any grid run was launched**.
Author: subagent (approved by user 2026-08-09).
Deliverable: `idea-stage/RGCL_ABLATION_RESULT.md`.

问题:检索管线到底比裸分类头准多少、增量来自哪个部件。

---

## 0. 协议变更记录(用户指令)

**2026-08-09,任务执行中**,用户裁定评估协议由"只报 val"改为**标准三段式**:

> train 上训练、val 上选 epoch/超参、test 上报告最终数字 —— 与项目已发表数字
> (HateMM 0.870 等)同协议同框可比。

随之生效的约束(全部写入本冻结书,先于任何网格运行):

1. 本节即协议变更的补记(用户指令,时间戳 2026-08-09)。
2. 整张网格**一次性提交、所有格子全报**;绝不做"看了 test 再改设计"的迭代。
   本文件冻结之后,设计、开关、判决阈值、读数规则一律不得修改。
3. 结果表 **test 为主列**,val 同表附报。
4. 部件贡献判决(kNN 读出 / 检索引导 / 对比正则,阈值 +0.005、seed 同向)
   **在 test 数字上执行**,val 作稳健性对照。
5. HateMM 带文本通道的格子附脚注:**test 空转录本密度 12.1% vs train 5.2%**
   (`refine-logs/TRAINTEST_AUDIT_2026-08-09.md` §307),
   即 HateMM 文本通道的 test 数字有**已知的虚高方向**。

在此之前的设计(3×2 因子、编码器盘点、3 seeds、后台运行纪律)不变。
本次运行因此**有意接触 test 集**,这是用户显式裁定的协议,不是纪律失守;
接触方式仅限"在 val 选定的 epoch 上读一次 test 指标",无任何 test 驱动的选择。

---

## 1. 因子设计(3 × 2)

### 训练损失(3 档),同头结构、同预算、同 BCE 梯度尺度

总损失装配式在三档中**完全相同**:
`total = contrastive * (1 - ce_weight) + BCE * ce_weight`,`ce_weight = 0.5`(默认)。

| 档 | `--contrast_mode` | contrastive 项 |
|---|---|---|
| **L1** | `none` | 恒为 0 → 只剩 `0.5 * BCE` |
| **L2** | `random` | `relu(in_batch_neg − pos_rand + neg_rand + margin)`,配对**batch 内均匀随机**(随机同标签正例 + 随机异标签负例) |
| **L3** | `retrieval` | 现行 RGCL:FAISS 挖最近邻伪金正例 + 最像的异标签难负例(**默认路径,逐位不变**) |

L2 与 L3 的唯一差别是**选择规则**(uniform random vs FAISS-nearest):
同 triplet、同 margin(0.1)、同 in-batch 负例项、同每锚点配对数
(`no_pseudo_gold_positives=1`, `no_hard_negatives=1`)、伙伴向量同样 **detach**
(L3 的挖掘结果本来就取自每 epoch 一次的 detached train bank)、找不到伙伴时同样
留零向量(与挖掘路径的 "not found" 约定一致)。
shuffle 过的 batch 本身是 train 的均匀随机子集,故"batch 内均匀随机"在分布上等于
"从 FAISS 索引所覆盖的同一 train 池里均匀随机" —— 池不变,只有"最近"变成"随机"。

L1 保持 `ce_weight=0.5` 而非改成 1.0,是为了让 BCE 的梯度尺度在三档完全一致;
代价是 L1 的绝对 BCE 学习率是"纯 BCE 基线"的一半,这是**有意的控制**,记录在此。

实现:`src/model/loss.py`(`--contrast_mode none` 早返回;`_random_inbatch_pairs()`),
`src/run_rac.py`(CLI + exp_name 加 `_cm-<mode>` 后缀防目录冲突)。
`--contrast_mode retrieval` 是默认值且不进入任何新分支 ⇒ 存量运行逐位不变。

### 推理(2 档),同一次训练里同时读出,不额外训练

| 档 | 读出 | 日志行 |
|---|---|---|
| **I1** | 分类头 sigmoid 输出 | `dev  Epoch N ... \| macroF1:` / `test Epoch N ... \| macroF1:` |
| **I2** | 学到空间的 kNN 读出(topk=20,相似度加权 arithmetic 投票,现行默认) | `Val_Retrieval Epoch N macroF1:` / `Test_Retrieval Epoch N macroF1:` |

两个读出来自**同一个 checkpoint 的同一次 forward**,因此 I2−I1 是纯读出差,
没有任何训练差异混入。

---

## 2. 网格

- **数据集**:HateMM、MHC(EN)、MHC_zh、ImpliHateVid。各自官方 train/dev_seen/test_seen。
- **编码器**(以本地实际存在的特征缓存为准,盘点表进结果文件;缺格标缺,不现抽):
  - `CLIP` = `openai_clip-vit-large-patch14-336_HF` — 4 个数据集全有
  - `QWEN` = `Qwen2.5-VL-7B-Instruct_HF`(frozen) — 4 个数据集全有
  - `LORA` — HateMM / MHC_zh = `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`(与
    `scripts/slurm/enc3seed_lora_curric.sbatch` 一致);MHC(EN)= 本地只有
    `Qwen2.5-VL-7B-Instruct-LoRA_HF`;**ImpliHateVid 无 LoRA 缓存 → 该格标缺**
- **格子数**:4×3 − 1 = **11** 个 (encoder, dataset) 格
- **运行数**:11 × 3 损失档 × 3 seeds (0,1,2) = **99** 次头级训练,单次提交

超参逐字沿用现行调用(`scripts/slurm/enc3seed*.sbatch`):
`--batch_size 64 --lr 1e-4 --epochs 30 --topk 20 --proj_dim 1024 --map_dim 1024
--dropout 0.2 0.4 0.1 --fusion_mode align --metric cos --loss triplet
--batch_norm False --hybrid_loss True --warmup 5 --majority_voting arithmetic
--no_pseudo_gold_positives 1 --hard_negatives_loss True --no_hard_negatives 1
--lambda_seg 0 --Faiss_GPU False --final_eval False`

---

## 3. 读数规则(冻结,先于结果)

每次运行、每个读出档,独立按**该读出自己的 val 指标**选 epoch,再在该 epoch 上读 test:

- **I2**:选 epoch = `argmax_{epoch ≥ 5} (Val_Retrieval acc, tie-break Val_Retrieval roc)`
  —— 即 `enc3seed*.sbatch` 里既有的、产出已发表数字的那条规则,逐字沿用。
- **I1**:选 epoch = `argmax_{epoch ≥ 5} (dev head acc, tie-break dev head roc)`
  —— I2 规则的对称类比。
- 主数字 = 选定 epoch 的 **test macro-F1**;同表附报同一 epoch 的 **val macro-F1**。
- 每格 3 seeds,报 **mean ± std**(std = 样本标准差 ddof=1)。
- 若某 seed 的运行崩溃/无解析,该格标 FAIL,不用其余 seed 顶替。

warmup=5 与既有 banked 运行一致。除以上之外**不做任何 epoch 级挑选**。

---

## 4. 部件贡献与判决(冻结,test 数字上执行)

定义(每格独立计算,配对到同一 seed):

- **kNN 读出贡献** = `I2 − I1`(固定同一 loss 档) → 格 = (encoder, dataset, loss),共 **33** 格
- **检索引导贡献** = `L3 − L2`(固定同一 inference 档) → 格 = (encoder, dataset, inference),共 **22** 格
- **对比正则贡献** = `L2 − L1`(固定同一 inference 档) → 格 = (encoder, dataset, inference),共 **22** 格

**格支持(cell supports)** 的判定:该格 3 个 seed 的**配对差**(同 seed 相减)
① 均值 ≥ **+0.005** macro-F1,**且** ② ≥ 2/3 个 seed 的差为正(方向一致)。

**部件判决**(无 AMBIGUOUS):

- **活着 (ALIVE)**:支持格数 ≥ ⌈总格数/2⌉(kNN: ≥17/33;检索引导、对比正则: ≥11/22)
- **装饰 (DECORATIVE)**:否则(均值 < +0.005,或方向混乱)

val 数字按同样规则再算一遍,作**稳健性对照**;test 与 val 判决不一致时,
以 test 为准并在结果文件里明写不一致。

**交互项(特别报告)**:
比较 `(L3,I2) − (L3,I1)` 与 `(L1,I2) − (L1,I1)`(每格、配对 seed)。
- RGCL 故事(对比学习整理空间,使 kNN 读出真正生效)**成立** ⟺
  差之差 `[(L3,I2)−(L3,I1)] − [(L1,I2)−(L1,I1)] ≥ +0.005` 在 ≥ 半数格子(≥6/11)上成立
  且该格 ≥2/3 seed 同向。
- 否则判**不成立**(kNN 读出的好处与是否做了检索引导对比无关)。

---

## 5. 已知偏倚脚注(必须出现在结果文件)

**HateMM 且使用文本通道的全部格子**:官方 test 切分的空转录本密度 **12.1%**,
train 仅 **5.2%**(`refine-logs/TRAINTEST_AUDIT_2026-08-09.md`)。
⇒ HateMM 的 test 数字在文本通道上有**已知的虚高方向**,跨数据集比较时须打折看。
(本实验全部编码器都吃 `text_feats`,故 HateMM 三个编码器格子全部适用此脚注。)

---

## 6. 纪律

- 冻结在前、运行在后:本文件写完并落盘之后才提交网格。
- 实现期烟测使用**合成随机特征 + 随机标签**的假缓存(`--model SMOKESYNTH`),
  不在真实数据上计算候选指标 → 设计/实现期盲性保持。
- 正式网格**单次提交**,`setsid nohup` 后台,
  日志 `logging/runs/rgcl_ablation/run.log`,PID `logging/runs/rgcl_ablation/run.pid`,
  进度行格式 `PROGRESS enc=X ds=Y loss=Z inf=W seed=S`。
- 缺格(ImpliHateVid × LoRA)如实标注,不硬凑。
