# Train-only premise 独立运行前审查

日期：2026-08-31

## Verdict

**PASS — 允许启动当前源码与冻结参数下的 HMM + HateClipSeg formal premise。**

此 PASS 只授权 train-only premise；不代表 premise 已通过，也不授权正式 localizer。只有 formal
`analysis.json` 对两个语料均满足冻结 gate，且 availability 与 shuffle contract tests 同时通过，
才可进入下一阶段。已有 CPU smoke 明确输出 `SMOKE_ONLY_NO_DECISION`，不能用作性能证据。

## 审查范围

- `README.md`
- `NOVELTY_REVIEW.md`
- `premise.py`
- premise 实际调用的共享算子 `components.py`
- HMM/HateClipSeg 当前 train feature 与 split 的只读 contract 检查

未运行 formal premise，未读取 validation/test split、frame GT 或 video labels。

## 结论级检查

### 1. Train-only 与 OOF isolation：PASS

`load_train` 唯一 split 调用是 `hdata.load_split(corpus, "train")`，随后只读取三模态 feature；
代码没有读取 label、validation、test、GT 或 evaluator。`mdata` 的导入解析到
`scripts/reproduction_baselines/multihateloc/data.py`，三模态顺序为
`visual, audio, text`。

HMM 744 个 train videos 被固定分成 `248/248/248`；HateClipSeg 251 个 train videos 被分成
`84/84/83`。每折 PCA/whitening 只在另外两折 `train_ids` 的真实可用 rows 上拟合；held video
只在该折 PCA 下 transform 和 evaluate。predictor、unconditional mean 与优化数据也只来自
该折 `train_ids`。因此 held video 没有进入该折任何拟合统计或参数更新。

### 2. Fixed target 与 error aggregation：PASS

PCA target 固定且不参与 predictor 优化；每折、每模态均检查 explained variance finite 且严格
大于阈值，阻断可学习 target 与 predictor 一起全零塌缩。Huber loss 只覆盖 target modality
真实可用 rows。

每个 fold×target 记录 matched/mean/shuffled 的 `loss_sums` 与 `n_elements`。正式硬门按所有 held
elements 的总 loss 除以总 element 数进行 micro aggregation；不等长视频、不同模态与不同折
不会获得相同权重。fold×modality macro 仅作为 diagnostic，不参与 verdict。

### 3. Shuffled control：PASS

每个 held fold 内独立建立 donor assignment，donor 必须与 recipient 不同，且两个 conditioning
modalities 在 donor 全时长均真实可用。两个 conditioning modalities 共用同一个单调 uniform
donor time map；之后才施加 recipient 原本的 availability masks。因此该 control 保留：

- recipient 的逐行 availability flags；
- donor 内两个 conditioning modalities 的同一时间对应；
- donor 时间顺序与局部连续性；
- recipient 与 donor 的视频身份分离。

eligible donors 采用确定性均匀循环，不会退化成几乎全部 recipient 使用同一个 donor。按 formal
三折 split 对实际 availability masks 穷举：HMM visual/audio 每折分别有 41、45、52 个 eligible
donors，最大单 donor 负载分别为 7、6、5；HateClipSeg 分别有 9、8、10 个，最大负载为
11、11、9。所有 eligible donors 都被使用，三个 target 的 self-assignment 均为 0。text target
的 conditioning modalities 全覆盖，因此每折每个 recipient 使用不同 donor。

内置 time-coded contract test 得到 pair error 0、donor time step 非负、recipient flags exact。
此前分别重采样 modalities、按 availability pattern 交织 donor times、以及字典序 fallback 导致
donor 集中三类 confound 均已在正式运行前移除。

### 4. Availability/missingness：PASS

缺失 row 由 finite 且 norm 非零共同定义；PCA 只拟合 observed rows，project 后缺失 row 为零并
保留独立 boolean mask。当前 train features 的三模态长度逐视频一致且无 non-finite rows；
visual/audio 全覆盖，text 缺失确实存在，因此该路径在真实数据上被执行，而不是空测试。

`masked_logmeanexp` 是后续模型预定复用的实际算子。9-channel 测试按 modality-major 的
observed/predicted/private 三通道设置极端 absent logits，并确认每个 missing modality 的三个
通道全部被 mask；四种真实相关 availability pattern 的 constant-logit 输出最大差异低于
`1e-6`。算子还拒绝无任何有效 evidence 的输入。

### 5. 冻结 recipe 与 verdict：PASS

非 smoke 正式入口精确锁定：两个语料按 `hatemm, hateclipseg` 顺序运行，seed 234、PCA width 64、
hidden 128、radius 2、3 folds、6 epochs、batch 16、learning rate `3e-4`、每模态每折最多
30,000 个 PCA fit rows、CUDA。任一非默认设置在创建输出目录前失败；只有显式 `--smoke`
允许缩小设置，且无论 smoke 数值如何都只能输出 `SMOKE_ONLY_NO_DECISION`。

formal `PROCEED_TO_FORMAL_LOCALIZER` 需要同时满足：

1. HMM micro matched error < micro unconditional-mean error；
2. HMM micro shuffled error > micro matched error；
3. HateClipSeg 同时满足以上两项；
4. availability invariance contract PASS；
5. shuffle pair/time/alignment contract PASS。

任何一项失败均输出 `STOP_BEFORE_FORMAL_LOCALIZER`。不存在单语料、macro diagnostic、smoke 或
非冻结参数可产生正式 proceed 的路径。

## 实际检查记录

- `premise.py` 与 `components.py` compile PASS；工作树 whitespace/diff check PASS。
- availability invariant：PASS。
- shuffle pair、monotone time 与 exact flags：PASS。
- formal 非默认 corpus/device 调用：在创建输出前按预期失败。
- HMM/HateClipSeg 完整 train availability、fold coverage、donor feasibility 与 donor balance
  只读穷举：PASS，无不可分配 recipient。
- 双语料 CPU smoke 完整结束并写出 `SMOKE_ONLY_NO_DECISION`；其缩小 PCA、fold、epoch 与 CPU
  设置只验证执行链，不参与 premise 裁定。

## 非阻断边界

Shuffled condition 仍只是一项 train-only correspondence diagnostic；跨视频替换天然会改变内容
边缘分布，不能单独证明语义或因果 shared/private identification。`NOVELTY_REVIEW.md` 已正确要求
后续正式 localizer 由 same-checkpoint residual zero/permutation/noise interventions 与 matched
controls 承担机制归因。本次 PASS 不放宽该边界，也不允许 premise 失败后扫描 radius、PCA width、
capacity 或 loss 追门。
