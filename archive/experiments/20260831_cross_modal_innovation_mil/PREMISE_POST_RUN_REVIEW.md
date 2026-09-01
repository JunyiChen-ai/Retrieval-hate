# Train-only premise 独立 post-run 审计

日期：2026-08-31

## Verdict

**Result-chain integrity：PASS。**

**冻结实验裁定：`STOP_BEFORE_FORMAL_LOCALIZER`，复算成立。** 两个语料的 shuffled control 都比
matched 更差，但 matched predictor 在两个语料都没有优于 unconditional mean。按照预注册的
双语料合取 gate，本方向必须在正式 localizer 之前停止；本报告不改变 gate，也不提出围绕该结果
调参或重跑。

权威输入：
`runs/20260831_cross_modal_innovation_mil/premise_seed234/analysis.json`。

## 运行与产物完整性

- final `analysis.json` 可完整解析，所有数值 finite；包含冻结 args、两个 corpus、每 corpus 三折×
  三 target 的 9 个结果 row、PCA diagnostics、两个 contract tests 与最终 verdict。
- args 精确为 HMM+HateClipSeg、seed 234、PCA width 64、hidden 128、radius 2、3 folds、6 epochs、
  batch 16、learning rate `3e-4`、PCA sample rows 30,000、CUDA、`smoke=false`，输出路径为当前
  formal run 目录。
- `analysis.partial.json` 已包含两个完整 corpus；其中所有字段与 final 中对应字段逐值一致。final
  只在其基础上增加最终 `verdict`，不存在只完成一个语料后误判的情况。
- `run.log` 正常结束并打印与 final 一致的 STOP verdict；未见 traceback、exception、被杀、显存
  失败或非有限数值提示。`run.pid` 所指进程已退出。
- HMM 有 744 个 train videos，三折各 248；HateClipSeg 有 251 个，三折为 84/84/83。每个
  fold×target row 的 `n_eval_videos` 均与该折一致，9 个组合无缺失或重复。
- 每折三模态 PCA width 均为 64，最小/最大 explained variance 均 finite，最小值严格大于冻结
  退化阈值。
- donor diagnostics 完整：每 row 的 recipient 数与 fold size 一致，eligible donors 全部使用，
  self-assignment 全为 0，最大负载与记录的 fraction 可逐项复算。

run 目录没有另存一份可读的 code-version note 或独立 config 文件；完整配置已嵌入 final
`analysis.json`，且正式运行发生在独立 pre-run PASS 之后、当前受审源码未再改变。此项是输出格式
记录不足，不改变本次保守 STOP 结论。

## 独立 micro 复算

复算方法严格使用每个 corpus 的 9 个 fold×target rows：对同一 branch 求
`sum(loss_sums) / sum(n_elements)`。没有使用 row macro，也没有重新训练或读取其他 split。

| Corpus | Held elements | Matched loss sum / micro | Mean loss sum / micro | Shuffled loss sum / micro |
|---|---:|---:|---:|---:|
| HMM | 20,312,576 | 7,631,707.5390625 / 0.37571342694607024 | 7,629,578.865234375 / 0.37560863108816800 | 7,986,434.7578125 / 0.39317685545213465 |
| HateClipSeg | 10,751,488 | 3,928,868.9965820312 / 0.36542560402634794 | 3,895,405.2377929688 / 0.36231312705673570 | 4,002,739.8046875 / 0.37229635606601620 |

上述重算值与 final `aggregate_micro_huber_per_element` 逐项一致。每个 row 内的
`loss_sum / n_elements` 也与所记录 `errors` 一致。

### HMM gate

- `matched < mean`：**FAIL**。matched 比 mean 高 `0.00010479585790224`。
- `shuffled > matched`：**PASS**。shuffled 比 matched 高 `0.01746342850606442`。
- corpus pass：**false**。

### HateClipSeg gate

- `matched < mean`：**FAIL**。matched 比 mean 高 `0.00311247696961225`。
- `shuffled > matched`：**PASS**。shuffled 比 matched 高 `0.00687075203966825`。
- corpus pass：**false**。

即使 HMM 的 first-gate 差距较小，冻结 gate 是严格方向判断；结果仍是 FAIL，不能事后加入容差、
改聚合或只保留 shuffle gate。

## Contract tests 复核

### Availability invariant

- 四个 constant-logit availability patterns 的最大输出差为
  `1.1920928955078125e-07 < 1e-6`。
- missing modality 的 observed/predicted/private 三个通道均被 mask。
- 记录值与当前冻结判断式一致：**PASS**。

### Shuffle pair/time/alignment

- conditioning pair error：`0.0 < 1e-7`；
- 最小 encoded donor time step：`1.0 >= 0`；
- recipient availability flags exact：true；
- 记录值与当前冻结判断式一致：**PASS**。

## 最终结果链

两 corpus pass 的合取为 false；availability contract 与 shuffle contract 虽均为 true，也不能覆盖
corpus premise failure。独立重算得到：

`all_corpora_pass=false` → `STOP_BEFORE_FORMAL_LOCALIZER`。

该结果与 `analysis.json` 和 `run.log` 完全一致。因此可用结论仅为：固定 cross-modal conditional
predictor 没有在 HMM/HateClipSeg 两边同时证明优于 unconditional mean，冻结 premise 被淘汰；
不得启动该候选的正式 localizer，也不得为追过 premise 扫描 PCA width、context radius、capacity、
epoch、loss 或改变聚合/gate。
