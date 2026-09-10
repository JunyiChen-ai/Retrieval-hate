# 规则 6 代码复核：机制 D（证据分解，模块一第 3 轮）

复核对象：`git diff --cached`（`src/interval_evidence_hmm.py`、`src/hier_evidence_common.py`、`experiments/20260910_online_query_within/train.py`、README 第 8 节；`model.py`、`acquire.py` 本次无改动）。复核日期 2026-09-10，只读，CPU 检查（临时脚本在 scratchpad：`check_decomp.py`，`old_hc.py`/`old_ieh.py` = HEAD 版），未写 `runs/`、`data/`，未启动训练。

## 结论：**PASS，无必修项**；两条建议修（都是臂组合的保护，不影响默认配置的观察）

## 已确认无问题的项目

### 1. `any_hate_logodds`（`src/interval_evidence_hmm.py:610-620`）
- 正确性：k=4/j=2 小网格上对 8^G 条增广状态路径穷举，`normalized_time` 开/关各 30 组随机裁定（含细裁定、MISSING、随机时长），`logit(1 − Z0/Z)` 与函数输出最大差 3e-15。`_fb` 的 log Z 与 `_log_all_zero` 用同一个 `e`（`:616-617`），初始项都取 `Tm[0,0,·]`，与 `_posterior_video` 的 constrain 分支（`:424-426`）同式。
- 数值：`p0 = clip(exp(lz0 − lz), 1e-6, 1−1e-6)`，v ∈ ±13.8；`e[g,0]` 是 r/(1−r) 的幂、`Tm[g,0,0]` 是 CTMC 概率，不会取 log 0。
- 调用方式：`decomposed_logodds`（`hier_evidence_common.py:536`）传细裁定全 MISSING + 粗裁定，`_emissions` 对 MISSING 不乘因子（`:348,353`），所以 v 只由 4 个粗裁定决定，且在同一视频所有 mask 下恒定（实测）。已拟合 HCS HMM（it1 trial5）：bc = 0000 / 1000 / 1100 / 1111 → v = .20 / 2.49 / 4.62 / 8.94。
- `duration`：走 `_transitions(duration)` → `_segment_dt`，`normalized_time=True`（DEFAULTS `train.py:113`）时按比例，调用方传 `n_seconds`，与 `posterior` 同一路径。`R == 1` 断言：DEFAULTS `regimes=1`；`positive_constraint=True` 只影响拟合，`posterior` 与 v 都是无约束后验，口径一致。

### 2. `decomposed_logodds` / `scaffold_rows_interval(decomposed=True)`（`hier_evidence_common.py:529-538, 516-519`）
- `posterior(b_fine, b_coarse, n, w_coarse=0.0)`：签名接受 `w_coarse`（`:622`），`_emissions:351` 的 `w_coarse > 0` 为假时跳过粗因子；实测 50 组随机 mask，与"粗裁定全 MISSING"的 p_s / p_h 差 0。
- 文本项在行级加：`text_llr_rows` 把 (T,) 秒级 x_t 用 `resample_intervals` 重采到 snippet 行，再 `clip(ell_rows + tl, ±ELL_SCALE)`，P(s) 列 = sigmoid(截断后的 E)（`:517-518`）；传入的段级 `ps_seg` 在 decomp 下被覆盖，无副作用。
- 6 个 HCS test 视频 × 3 种 mask（无细裁定 / 每 7 窗一个 / 全 30）实测：ell 列 = `clip(rows(ell_fine + v) + x_rows)`、P(s) = sigmoid(ell)、COL_TEXT = x_rows、第 2–5 列（b_fine、b_coarse、P(h)、block）与 `evidence="hmm"` 逐位相同（P(h) 仍来自含粗因子的完整后验 `:553`），`make_scaffold_fn(evidence="decomp")` 与 `masked_fn(全裁定)` 逐位相同。
- 反事实：`acquire.py:190-194` 对每个候选窗 `cache.build(vid, bf2)` → `masked_fn` → `decomposed_logodds(bf2, bc)`：ell_fine 用新增裁定重算，v 用全 MISSING + bc 不变。训练（`TrainDataset:224`）、验证/测试（`EvalDataset:257`，`train.py:405-407` 全部传 masks）、获取都走 `masked_fn`，默认 scaffold 只在 `ScaffoldCache` 初始化时算一次、不进输入（与第 1 轮相同）。
- `text_centre`（`:462-476`）：只遍历传入的 train ids，读 `p_asr/p_ocr`，不读标签；实测 HCS −5.543、HateMM −5.717，与 README 第 8 节一致。`text_logit_seconds`（`:479-491`）：按 `isfinite` 过滤，两族取 max，无文本秒 0；6 视频无 NaN。

### 3. `train.py` 接线
- DEFAULTS（`:119-124`）：`window_target="verdict"`、`text=False`、`text_input=False`、`evidence="decomp"`；`text_term`（`:184`）= decomp 且不在 `no_text_term`/`no_text`。`evidence_hmm` 臂 → `evidence="hmm"`、`text_term=False` → 第 1 轮融合、文本列全 0；`no_text_term` 臂 → E = ell_fine + v。
- `text_x`（`:209-210`）按 `all_ids` 用 train centre 建一次；`refit`（`:239-240`）每次把同一个 `text_x` 传给新的 `masked_fn`，`ScaffoldCache` 只建一次、`masked_fn` 每次换（`:249`），与第 1/2 轮模式相同。
- 文本只经 E 进模型：`model.py:201-203` 的证据编码器读 `COL_ELL/ELL_SCALE`、`COL_PS`，先验项 `:233` 用同一 ell 列；`COL_TEXT` 只在 `text_input=True` 时进 `enc.lin`（`model.py:57`），默认 False → 无重复计入。`hmm_params_<tag>.json` 与最终 `hmm_params.json` 照旧保存（`:236,393`）；`results["evidence"]` 记录 mode / text_term / centre（`:401-402`），x_t 可由 centre 复现。

### 4. 泄漏
- centre 只用 train ids；`gt_arrays` 只用于 val/test id 过滤与评测（与前几轮相同）；test 文本只在 `masked_fn` 推理时进入。HMM 拟合 `fit_hmm(..., text_obs={})`（`use_text=False`）：只用裁定与 train 视频标签，与第 1 轮完全相同。

### 5. 其它实验
- `evidence` 为关键字参数、默认 `"hmm"`，`decomposed=False` 时 `scaffold_rows_interval` 与旧代码同一算式：HEAD 版与暂存版在 `make_scaffold_fn` / `make_masked_scaffold_fn` 两路上实测逐位相同（差 0）。其它调用方（20260904/05/06/07/08 各 train、`evidence_shuffle_test.py`、`adaptive_query_replay.py`）都用位置参数或不传新参数，不受影响。

## 建议修（非必修）

1. **`text_input` 与 `text_term` 同时为真会重复计入文本**：`train.py:239-240` 把 `text_llr` 换成 `text_x` 后，若 `cfg.text=True, text_input=True, evidence="decomp"`，COL_TEXT 列（x_t/5）经 `enc.lin` 进模型，同时 x_t 又在 E 里。默认关闭，但没有断言。建议 `train.py:184` 后加 `assert not (text_input and text_term)`。
2. **`regimes3` 臂与 decomp 不兼容**：`any_hate_logodds:614` 断言 R == 1，`regimes3`（`TRAIN_ARMS`）下第一次建 scaffold 即崩。诊断臂，建议在 `train.py:186` 后加 `assert not (regimes > 1 and evidence == "decomp")` 或让该臂自动退回 `evidence="hmm"`。

## 备注

1. **E 的截断饱和比例高**：x_t 最大约 +12（p 上限 1−1e-4、centre −5.5），v 最大 8.9，两者相加常超过 ELL_SCALE 13.8。6 个 HCS test 视频实测：粗裁定 ≥ 2 个为 1 的视频，24%–71% 的行 |E| 恰为 13.8（证据编码器输入 = 1.0、P(s) = 1.0、先验项 = α），其余视频 0–3%。这些行之间的排序只剩内容流。T2 门（README 第 8 节）用的是同一代码路径的 ell 列，因此门值已含这一效应，不改变已记录的观察；但先验项幅度从第 1 轮的 ell/13.8 ∈ [−.3, .1] 变到 ±1，若后续要改 x_t 的尺度或 centre，T2 须重算。
2. `hmm_weight` 臂与 `localization`/`entropy` 控制策略仍用完整后验（`acquire.py:127-128`，含粗因子）估计 VLM 应答分布 `pred_fine`：这是对裁定的预测而非逐秒证据，与 README "获取不变"一致。
3. `no_text` 臂在本轮等价于 `no_text_term`（`use_text` 已默认 False）。

## 判定
PASS。建议 1、2 是臂组合保护，默认配置与两条预注册臂（`evidence_hmm`、`no_text_term`）不受影响；可直接开搜索。
