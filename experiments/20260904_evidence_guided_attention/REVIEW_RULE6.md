# 候选 3 规则 6 code review（2026-09-04）

审查对象：`model.py`、`train.py`、`search.py`、`README.md`；共享代码 `src/hier_evidence_common.py`（当日从候选 1 升入）、`src/verdict_hmm.py`、`src/vlm_verdict.py`、`scripts/reproduction_baselines/macilsd/`、`scripts/reproduction_baselines/hate_common/`。只查规则 6 五项：机制是否进 forward/loss 与最终分数、train/val/test 泄漏、特征/时间/标签/split 对齐、超参数与 checkpoint 加载链、是否调用统一评测器；另确认无脚本写入 `data/`。未跑训练、未用 GPU；数值核对用 CPU 小张量（脚本在会话 scratchpad `verify.py`，输出摘录于下）。

## 结论：PASS

BLOCKER：无。对候选 3 文件做了一处非阻塞修正（见第 4 节）。

## 1. 机制是否进入 forward / loss / 最终分数（README 第 1、3 节 vs `model.py`）

各臂开关（实例化后检查部件存在性，参数量）：

| 臂 | concat 裁定进音频流 | 证据编码 enc | 格子嵌入 | key 偏置 β | 上下文 C | fc_a 输入维 | 参数量 |
|---|---|---|---|---|---|---|---|
| full | 否 | 有 | 有 | 有 | 有 | 896 | 363,653 |
| avce | 是 | 无 | — | 无 | 无 | 900 | 346,241 |
| no_enc | 否 | 有（只喂 β 与 C，不加进两流） | 有 | 有 | 有 | 896 | 363,653 |
| evid_audio_only | 否 | 有（只加进音频流） | 有 | 有 | 有 | 896 | 363,653 |
| no_cell | 否 | 有（四列线性，`enc.lin.in_features=4`） | 无 | 有 | 有 | 896 | 363,397 |
| no_bias | 否 | 有 | 有 | **无（`cma.bias is None`，state_dict 无该参数）** | 有 | 896 | 363,137 |
| no_context | 否 | 有 | 有 | 有 | 无 | 896 | 347,141 |

参数量与 README 第 1 节数字（363,653 / 346,241）一致。用 forward hook 读 `cma` 的两个输入流，确认 e_t 的路由：full/no_cell/no_bias/no_context 两流都加；no_enc 两流都不加；evid_audio_only 只加音频流。每臂只切换描述的那一个部件。

`avce` 臂 = 候选 1 数据通路：把候选 1 `Candidate`（`AVCE_Model`）的权重逐层拷进 `EGCA(arm="avce")`（`fc_a` 取候选 1 前 900 列，候选 1 的第 900-901 列对应被置零的 p_h/block 输入），无 padding 时六个返回值与 `last_content_logit` 最大差 ≤ 6e-8（sigmoid 精度）；即裁定四列拼接位置、ell/ELL_SCALE、共享单层 pre-norm 跨模态注意力、共享 fc 头、ceil(T/16) top-k 均值均与候选 1 一致。有 padding 时有效行 av_log 差 4e-2，来源是候选 3 屏蔽了 padding 的 key（候选 3 attention 落在 padded key 上的概率质量 = 0），README 已声明该差异；无 EMA/伙伴网络同样已声明。

p_h / block 两列不可见：对 `avce` 臂与 `full` 臂把 f_a 的第 900-901 列加 100 后 av_log 差 = 0。`block_bag_loss` 仍从原始 f_a 读这两列，forward 用 `.clone()`，f_a 未被原地修改（`torch.equal` 为真）。

`BiasedMultiHeadAttention`：把 query 投影置零后，p[b,h,i,:] 对每个 query i 都等于 softmax(key_bias[b,h,:])（差 0），说明偏置加在 key 轴且对所有 query 相同；masked key 概率质量 0。`no_bias` 臂偏置模块不存在，不是零初始化。

零初始化模块可学：`enc.cell.weight`（零初始化）、`cma.bias.weight`（零初始化）、`ctx.weight`、`enc.lin.weight` 在一次 BCE 反传后梯度范数分别 3.7e-1、5.0e-2、5.6e-1、1.6e-1，均 `requires_grad=True`，未 detach。`cma.bias.bias` 梯度 2e-9：该项对所有 key 相同、softmax 中抵消，本来就不起作用，不影响观察。

证据编码输入 `evid`：hook 抓到的 enc 输入第 0 列 = 原始 ell / ELL_SCALE（差 0，只除一次），第 1 列 = p_s，第 2-3 列与原始 b_fine/b_coarse 逐元素相等且为 0/1，cell 索引 = 2·b_fine + b_coarse（四种取值全出现）。真实 ell 的界 = log((1-1e-6)/1e-6) = ELL_SCALE，故 evid 第 0 列在 [-1, 1]。

先验：`av_log - last_content_logit == prior_scale · ell_raw / ELL_SCALE`（差 1e-7），用的是 f_a 里未除的 ell；`prior_scale=0` 时差为 0；候选 1 同式核对通过。`last_content_logit` 不含先验，块级 MIL 读它。`no_verdict`：evid 置零、先验关、块损失关，把全部 scaffold 列替换成随机数后 av_log 差 = 0（此时 e_t 为常数向量，β 常数在 softmax 里抵消，C 为常数偏置，不携带裁定信息）。

`seq_len=None`（`score_split` 路径）：forward 正常，bag = sigmoid(mean av_log)（差 0），与显式给满长度 seq_len 的 av_log 差 0；返回元组第 3 项 = 带先验的 av_log。

## 2. 训练脚本语义与候选 1 对齐（`train.py`）

- `no_prior` → prior_scale=0；`no_block` → lambda_block=0；`no_verdict` → 两者为 0 且 `EGCA(no_verdict=True)`；`no_cmal` → lamda_cma=0，`lam>0` 为假跳过 CMAL；`mean_prior` → 传给 `hc.make_scaffold_fn`，与候选 1 同一函数同一分支。均与候选 1 `train.py` 定义一致。
- CMAL 预热：`lam = min(lamda_cma, lamda_cof·epoch)` 乘 (c1+c2+c3+c4)；候选 1 是 `min(lamda_a2b, cof·epoch)·(c1+c3) + min(lamda_a2n, cof·epoch)·(c2+c4)`，在 a2b=a2n 时逐 epoch 数值相同（0/.05/.5/1.3 核对）。CMAL 收到的 a_log/v_log 为 sigmoid 概率，与候选 1 相同。
- `fix_rep_swap` 默认 False，`audio_rep, visual_rep = v_out, a_out`，与候选 1 及上游一致。
- Dropout：残差 dropout = cfg.dropout(.2)，注意力概率与 FFN dropout .1，与 MACIL-SD `TransformerLayer` 相同。

## 3. 泄漏、对齐、加载链、评测器、data/ 写入

- HMM 只在 `train_ids` 上拟合（`hc.fit_hmm`）；val/test 的 scaffold 只由冻结 VLM 裁定 + 该 HMM 算出，不用 val/test 标签。`val_gt` 只用于每 epoch 选 checkpoint，`test_gt` 只用于过滤 test_ids 与评测；`labels` 只进 `TrainDataset(train_ids)`、`block_bag_loss`（train batch）与 within 宏平均的正例集合。test 标签不进梯度、不进选 checkpoint。
- 对齐：`TrainDataset` 对 f_v、f_a、w 用同一 `process_feat(max_seqlen, is_random=False)`（长则等距抽样、短则尾部补零），行对齐不变；`_seq_len_of(f_v)` 数非零行，padding 在尾部，`mask = arange(T) < seq_len` 正确。`EvalDataset` 五 crop 堆叠、`index_map = snippet_index_for_seconds` 把 snippet 行映射到秒，`score_split` 对五 crop 的 sigmoid 取均值后按 index_map 取秒，行数 ≠ n_seconds 时抛错。以上全部为候选 1 原码（见下）。
- 加载链：`search.py` 每 trial 写 `hparams.json`（六个键 lr/max_seqlen/lamda_cma/prior_scale/w_fine/lambda_block）并以 `--config` 传给 `train.py`；`train.py` 用 `DEFAULTS` 更新后的 cfg 建模，六个键都在 `DEFAULTS` 中。checkpoint 按 validation (AP+ROC)/2 深拷贝 state_dict，`load_state_dict` 后再评 val/test。Optuna 目标 = test (AP+ROC)/2，within 低于下限剪枝，README 第 4 节与 docstring 均写明（规则 7/10）。
- 评测器：`hc.run_evaluator` 子进程调用 `scripts/reproduction_baselines/eval_baseline_scores.py`（`--split val|test`），写 `metrics_val.json` / `metrics.json`；epoch 内选 checkpoint 用 `scripts/duplex/frame_eval_common.evaluate`（同一评测器核心）。无第二份评测逻辑。
- `data/` 写入：grep 三个候选文件、`hier_evidence_common.py`、`verdict_hmm.py`、`vlm_verdict.py`、`macilsd/align.py`、`utils.py`、`hate_common/*` 的所有 `open(..,"w")`/`save`/`makedirs`，目标全部在 `out_dir`（runs/）下；`ScaffoldCache` 只在内存。无写入 `data/`。

## 4. 升入 `src/hier_evidence_common.py` 是否逐字

用 AST 逐定义比较 `git show HEAD~1:experiments/20260903_hier_evidence_mil/{dataset.py,train.py}` 与 `src/hier_evidence_common.py`：`TEXT_ROOT/TEXT_DIM/SCAF_DIM/COL_*/N_INPUT_SCAF/A_EXT_DIM/SCAF_OFFSET/EVALUATOR/K_FINE/J_COARSE/ELL_SCALE`、`text_path/load_text_rows/scaffold_rows/ScaffoldCache/TrainDataset/EvalDataset/_scalar/_git_describe/usable/score_split/frame_metrics/write_scores/run_evaluator/fit_hmm` 全部 SAME；仅三处差异：`REPO_ROOT` 相对路径从 `../..` 改为 `..`（文件位置变了，解析到同一绝对路径），`block_bag_loss` 与 `make_scaffold_fn` 去掉 `ds.` 前缀（同模块内引用）。行为不变。候选 1 `dataset.py` 改为再导出，`train.py` 从 `hier_evidence_common` 导入这十个函数，`import train` 检查通过（`ELL_SCALE`、`SCAF_OFFSET` 与升入前相同）；`src/verdict_hmm.py`、`src/vlm_verdict.py`、`macilsd/`、`hate_common/`、评测器自候选 1 出数以来无提交、工作区干净，候选 1 记录的数字仍可复现。

## 5. 非阻塞修正与备注

1. **已修**（`search.py:4`）：docstring 示例的 `--out-root runs/20260903_hier_evidence_mil` 是候选 1 的目录。候选 3 的 study_name 与候选 1 相同（`<corpus>_seed<seed>`）且 `load_if_exists=True`，照抄该命令会接续候选 1 已完成的 study（budget.json 已存在 → 不跑新 trial）并覆盖候选 1 的 `study_summary.json`。改为 `runs/20260904_evidence_guided_attention`，与 README 第 5 节一致。代码逻辑无改动。
2. `search.py --ablation` 若传非 `full`，输出与 `full` 搜索共用同一 root 与 study；README 规定消融只用 `train.py --config <best hparams.json>` 单独跑，按 README 走即可，不要用 search.py 跑消融。
3. `cma.bias` 的 `bias` 向量对所有 key 相同、softmax 中抵消，是无效参数（不影响任何数字，只是 README 参数量里的 4 个）。
