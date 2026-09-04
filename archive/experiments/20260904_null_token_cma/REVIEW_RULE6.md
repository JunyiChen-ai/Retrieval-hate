# 候选 4 规则 6 code review（2026-09-04）

审查对象：`model.py`、`train.py`、`search.py`、`README.md`（commit `1bad5d2`）；共享代码 `src/hier_evidence_common.py`（当日已按逐字升入复核，本次不重验）、`src/verdict_hmm.py`、`src/vlm_verdict.py`、`scripts/reproduction_baselines/macilsd/{avce_network,Transformer,CMA_MIL,train,align,utils}.py`。这些共享文件自候选 3 review（commit `b2bdca2`）以来无提交、工作区干净，候选 3 review 第 2-3 节对训练脚本、泄漏、对齐、评测器的核对结论对本候选同样成立；本次重点是 `model.py` 与 `train.py` 相对候选 3 的差异。只查规则 6 五项。未跑训练、未用 GPU；数值核对用 CPU 小张量（B=3、T=12、seq_len=[12,7,3]、prior_scale=2.7，脚本在会话 scratchpad `verify_c4.py` / `verify_c4_train.py`，33 项检查全部通过）。

## 结论：PASS

BLOCKER：无。未改动任何候选 4 文件。

## 1. `train.py` 相对候选 3 的差异

`diff` 只有四处：docstring、`from model import NTCA`、`TRAIN_ARMS` 增加 `no_input`、模型构造多传 `hide_input=(ablation == "no_input")`。其余（数据、HMM 拟合、scaffold cache、loss、CMAL 预热、checkpoint 选择、评测器调用、summary）与候选 3 逐字相同，候选 3 review 已核。`search.py` 与候选 3 只差 docstring 里的 out-root 路径（`runs/20260904_null_token_cma`，与 README 第 5 节一致，不会接续候选 3 的 study）。

`no_input` 语义与候选 1 一致：候选 1 `Candidate(hide_input=True)` 把 `f_a_in[..., SCAF_OFFSET:]` 置零、prior_scale 与 lambda_block 不动；本候选 `train.py:97-98` 对 `no_input` 不清零 prior_scale / lambda_block，`arm="full"`，`NTCA(hide_input=True)` 把 evid（输入路径四列）与 c 置零、先验照加（数值核对 D3：fc_a 输入的裁定列全 0、c 全 0、content logit 对 scaffold 列随机替换不变、`av_log - content == prior_scale·ell/ELL_SCALE`）。block loss 仍从原始 f_a 读 p_h / block 列。

## 2. 机制是否进入 forward / loss / 最终分数（`model.py`）

**`no_token_unmasked` = 候选 1 前向（主对照臂）。** 建候选 1 `Candidate`（`AVCE_Model`，v_feature_size=1024、a_feature_size=902、topk_div=16），参数加随机扰动后逐层拷进 `NTCA(arm="no_token_unmasked")`：fc_v；fc_a 取候选 1 权重前 900 列（候选 1 第 900-901 列对应被置零的 p_h/block）；`linears[0..3]` → `lin_q/k/v/o`；`sublayer[0].norm`/`sublayer[1].norm` → `norm_attn`/`norm_ff`；`feed_forward.w_1/w_2` → `ff[0]/ff[3]`；`att_mmil.fc` → `fc`。带 padding 的 batch、prior_scale>0、eval 模式：六个返回值最大差 mmil 0 / sigmoid a,v 1.5e-8 / av_log 0 / v_out 0 / a_out 0，`last_content_logit` 差 0。train 模式（dropout 生效、同一 RNG 种子）六个输出差 0：三处 dropout（注意力概率 .1、FFN 内 .1、残差 cfg.dropout=.2）位置、顺序、概率与 MACIL-SD `TransformerLayer` 相同。该臂 padding key 有非零注意力质量（随机输入 .337），即候选 1 的偶然空 token 行为被原样保留。bag = sigmoid(top ceil(t/16) 均值)，与候选 1 `Candidate.bag` 一致（不是 `Att_MMIL.clas` 的 t//16+1，候选 1 本来就覆盖了它）。

**`no_token_masked` 只差 key mask。** 无 token 参数，state_dict 键与 unmasked 臂相同；无 padding 时两臂 av_log 差 < 1e-6；有 padding 时 padding key 注意力恰为 0，无 padding 的样本输出相同、有 padding 的样本有效行不同（3.6e-1）。

**`full`。** 用 forward hook（含 kwargs）抓两次注意力调用：
- 方向 v←a：key 序列长 T+1，key[0] = `base[1]` + `cond(c)`（音频 base），key[1:] = fc_a 输出；value 序列与 key 序列相同。
- 方向 a←v：key[0] = `base[0]` + `cond(c)`（视觉 base），key[1:] = fc_v 输出。两方向 token 不同。
- key mask：第 0 位恒 True，其后 = `arange(T) < seq_len`；padding key 注意力 0，token 注意力 > 0（随机输入下均值 .128），每行注意力和为 1。
- c = 有效行上 [ell/ELL_SCALE, p_s, b_fine, b_coarse] 的均值（差 < 1e-6），与含 padding 行的均值不同（padding 确实被排除）；ell 只除一次 ELL_SCALE。
- 输出长度保持 T（token 只作 key/value，不作 query）。
- 参数量 full 347,137 / no_token 346,241，与 README 第 1 节一致。
- 梯度：BCE 反传后 `base` 两行与 `cond.weight` 梯度范数 7.6e-3 / 7.7e-3 / 1.6e-2，均非零、未 detach。

**`const_token`**：`cond is None`，state_dict 无 `cma.cond.*`，base (2, hid)。**`shared_token`**：base (1, hid)，两方向 key[0] 相同且 = base + cond(c)。

**先验 / no_verdict / 评测路径。** `av_log - last_content_logit == prior_scale · ell_raw / ELL_SCALE`（ell 取自 f_a 未缩放列，差 < 1e-5）；prior_scale=0 时差恰为 0；`last_content_logit` 不含先验（块级 MIL 读它）。`no_verdict`：c=0、fc_a 裁定列 0、scaffold 全列随机替换后 av_log 差 0、无先验。p_h/block 两列加 100 后 av_log 差 0；forward 用 `.clone()`，f_a 未被原地修改。`seq_len=None`（`score_split` 五 crop 路径）：mask 全 True，av_log 与显式满长度 seq_len 路径差 < 1e-6，bag = sigmoid(mean av_log)，元组第 3 项 = 带先验的 av_log；`score_split` 正是解包第 3 项。

## 3. 泄漏、对齐、加载链、评测器、data/ 写入

- 与候选 3 逐字相同的部分（HMM 只在 train_ids 上拟合；val_gt 只用于每 epoch 选 checkpoint；test_gt 只用于过滤 test_ids 与评测；labels 只进 TrainDataset / block_bag_loss / within 正例集合；TrainDataset 三路同一 `process_feat` 保持行对齐；`_seq_len_of` 数非零行、padding 在尾部）候选 3 review 已核，代码未变。本候选新增的 mask（`arange(T) < seq_len`）与 c 的有效行均值建立在同一 seq_len 上。
- 加载链：`search.py` 每 trial 写 `hparams.json`（lr/max_seqlen/lamda_cma/prior_scale/w_fine/lambda_block，六个键都在 `train.py DEFAULTS` 中）并以 `--config` 传给 `train.py`；`train.py` 用 `DEFAULTS.update(config)` 建模。checkpoint 按 validation (AP+ROC)/2 深拷贝 state_dict，`load_state_dict` 后再评 val/test。Optuna 目标 = test (AP+ROC)/2，within 破下限剪枝，README 第 4 节与 docstring 写明（规则 7/10）。test 标签不进梯度、不进 checkpoint 选择。
- 评测器：`hc.run_evaluator` 子进程调用 `scripts/reproduction_baselines/eval_baseline_scores.py`（路径存在），写 `metrics_val.json` / `metrics.json`；epoch 内选 checkpoint 用 `scripts/duplex/frame_eval_common.evaluate`（同一核心）。无第二份评测逻辑。
- `data/` 写入：grep 候选 4 三个文件与 `hier_evidence_common.py`、`verdict_hmm.py`、`vlm_verdict.py`、`macilsd/align.py`、`utils.py`、`hate_common/*` 的所有 `open(..,"w")`/`save`/`makedirs`，目标全部在 `out_dir` / `root`（runs/）下；`verdict_hmm.py:62` 是 `save(path)`，只被 `train.py:109` 以 `out_dir/hmm_params.json` 调用。无写入 `data/`。
- `python -c "import train, search"` 通过。

## 4. 非阻塞备注

1. c 在训练时是 `process_feat` 等距抽样（≤ max_seqlen）后有效行的均值，测试时是全序列（不截断）的均值；等距抽样下两者是同一量的近似，属设计本身，不改变观察。
2. `full` 臂下 `no_input` / `no_verdict` 的空 token仍存在（= base + cond.bias，常量），README 第 3 节"c 也置零"与此一致。
3. `search.py --ablation` 若传非 `full`，与 `full` 搜索共用同一 root 与 study（候选 3 review 备注 2 同样适用）；消融按 README 用 `train.py --config <best hparams.json>` 单独跑。
4. `no_token_unmasked` 复现的是候选 1 的前向与 loss 形式，不含候选 1 的 EMA/伙伴网络（README 已声明，候选 1 no_ema 两语料 ≈ 0）；README 第 4 节把它作为同协议主对照是正确的，与候选 1 记录数字（EMA 训练）的比较只作参考。
