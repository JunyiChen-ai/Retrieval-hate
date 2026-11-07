# 规则 6 代码审查：review round 1 的新代码（2026-11-08）

审查对象：工作树中未提交的改动（`git diff` + `git status`）：`src/qtl/data.py`、`src/qtl/train.py`、`src/qtl/replay.py`、`experiments/20261101_r6_length_table/launch/sbatch_arms.sh`、`experiments/20261101_r5_completion/{sim_monotone,final_curves}.py`、`experiments/20261108_astra_review/{policy_replay,e3_level_matched,e4_budget_audit,baseline_infer,e1_fusion}.py` 与 `launch/run_round1_cpu.sh`。只查会改变观察或结论的问题；不审风格。所有检查在本机 CPU 上几秒内完成，没有改动仓库文件（本文件除外）。

## 结论：FAIL（1 个会改变 E1 结论的问题，1 个 E1 的 baseline 分支与官方表不一致；其余 5 项通过）

必须处理后才能开跑 E1：F1。应当修正或至少在 README 明说：F2。E2、E3、E4、共享 replay、policy_replay 未发现问题。

---

## F1（E1，改变结论）：fusion head 在训练集上看到的 baseline 分数是样本内分数，head 会学到过度信任 baseline 的权重

位置：`experiments/20261108_astra_review/e1_fusion.py:158-160`（`z["train"]` 取自 `baseline_scores/<method>/<corpus>/seed_<s>/train.jsonl`）、`:185-194`（head 用这些 `z` 和视频标签训练）。

问题：冻结的 baseline 是在这些训练视频上训练出来的，它在训练集上的逐秒分数比在 validation / test 上分得开得多。本机实测 MACIL-SD HateMM（`runs/20261108_astra_review/baseline_scores/macilsd/hatemm/seed_*/{train,val,test}.jsonl`，视频级 AP 用每视频最大分）：

| seed | train | val | test | 正例视频平均分 train / val / test | 负例视频平均分 train / val / test |
|---|---|---|---|---|---|
| 234 | .978 | .817 | .722 | .661 / .471 / .428 | .028 / .076 / .094 |
| 2025 | .975 | .796 | .770 | .705 / .553 / .522 | .070 / .133 / .120 |
| 3407 | .985 | .804 | .708 | .617 / .469 / .430 | .069 / .125 / .132 |

HCS 的 Fed-WSVAD / DSANet 和 DeHate 的 Fed-WSVAD 由 `baseline_infer.py` 用同一个冻结 checkpoint 给训练视频打分，性质相同（baseline 的 validation 选 checkpoint 在 `train_meta.json`，训练集就是 `train_ids`）。

后果：head 的输入 `z_t` 在训练时几乎单独就能预测视频标签，top-k MIL 损失会把权重放在 `z_t` 上，对 8 个答案的 token 路径学得很少；到 test 上 `z_t` 变弱，fusion 约等于 baseline 本身。E1 要回答的是"baseline 拿到全部 8 个答案并学会融合后差距是否消失"，这个设置系统性地压低对照组，方向是对本方法有利的。外部评审会把 E1 的任何"仍然领先"归因于这一点，所以必须先处理。

最小修复（baseline 仍冻结，不重训 baseline）：把 `z_t` 从 head 的训练输入里拿出来，改为 `f_t = a · z_t + h_θ(tokens, ctx[, feat])`：`h_θ` 照现在的方式在训练视频上用视频标签训练（输入不含 `z_t`），标量 `a`（连同 lr、epoch）在 validation 上按 (AP + ROC) / 2 网格选取（validation 标签不进梯度，符合 2026-09-29 裁定；与现在选 epoch 的做法同级）。这样 baseline 的融合权重在样本外估计，答案侧仍然用全部 8 个答案并且是学出来的。若不改设计，至少在 README 写明该偏差及其方向（对照组被压低），并给出上表数字。

## F2（E1 步骤 1，baseline 分支）：DSANet 读的是 `score_align`，官方表用的是 `score_mlp`

位置：`experiments/20261108_astra_review/baseline_infer.py:6-8`（docstring 称 score_align 是"官方表的分支"）、`:70-72`（`logits2 = o[2]`，`1 - softmax(logits2)[:, 0]`）、`:110`（与 `--ref` 的 `score_av` 否则 `score_align` 比较）。

事实：`scripts/reproduction_baselines/aggregate_official_val.py:19` 规定 `"dsanet": ("score_mlp", ...)`。官方三 seed `frame_eval.json`（`~/Hate-follow-up/runs/legacy_lab2_results_pre20261004/reproduction/official_val/final/dsanet/hateclipseg/seed_*/`）：score_mlp AP/ROC = .538/.524、.580/.553、.543/.507，三 seed 均值 .5535/.5280 = `docs/duplex/OFFICIAL_VAL_RESULTS.md` 的 HCS DSANet 行；score_align = .576/.546、.567/.541、.541/.498，均值 .5614/.5283。`baseline_infer.py` 复算的是后者，`check.json` 里的 `test_vs_official_max_abs_diff` 也是与后者比，所以这个检查不会发现分支不对。

影响：E1 给 DSANet 的 `z_t` 不是规则 8 对照表里的那条分支（AP 高 .008，ROC 基本相同）。方向上对对照组有利、对本方法不利，幅度小，但 README 里"冻结的官方 baseline"的数字会和官方表对不上。同一问题早已存在于规则 14(f) 对照 `experiments/20261101_r5_completion/baseline_plus_vlm.py:39`（`score_av` 缺失时退回 `score_align`）：r6 README 14(f) 表里"HCS（DSANet）baseline 本身 .561 / .528 / .555"正是 score_align 三 seed 均值，而官方表是 .554 / .528 / .545。该问题不在本次 diff 内，这里只记一行。

最小修复：`baseline_infer.py` 的 DSANet 打分改为 `logits1 = o[1].reshape(-1, o[1].shape[2]); score = sigmoid(logits1[0:n].squeeze(-1) / args.temp)`（与 `dsanet/infer.py` 的 `score_mlp` 一致，`temp` 来自 `frozen_config.json`，seed 234 为 2.0），`--ref` 比较改读 `score_mlp`，docstring 改为 score_mlp；或者保留 score_align 但在 docstring 与 README 明写"不是官方表的分支，官方表为 score_mlp"。

---

## 通过的检查（含依据）

### 1. E2：`soft_fixed_edges` 只改答案分档

- `src/qtl/data.py:107-127`：`configure_source` 第 4 个参数缺省 `None`，每次调用先把模块变量 `SOFT_FIXED` 重置，所以 `policy_replay.py:40`、`final_curves.py:52`、`sim_monotone.py:203` 等仍用 3 参数调用的地方得到 `None`，行为与改动前相同（本机验证：`configure_source("soft_both", 2, None, [0.5])` 后 N_CAT/N_LEV = 1/2，p = 0/.49 → 0，.5/.51/1 → 1；再调 `configure_source("soft_both", 8)` 后 N_LEV = 8、`SOFT_FIXED = None`）。
- `data.py:175-182`：只在 `load_answers` 里把 `soft_edges` 换成固定边；`hard2q`（`soft_levels` 2、无固定边）走 `soft_edges(P, split, 2)` = 训练集 p 的中位数。
- 分档后的 `answers` 是全链路唯一入口：答案表拟合 `qtree.fit_anchored_shared`（`train.py:247`，读 `ta` = `answers`，表形状用全局 `qtree.N_LEV`）、训练观测 `train_obs = qdata.observed(answers)`（`train.py:214`）、EIG 与后验里的 `policy.Asker`（`outcome_table` / `answer_index` 用 `qtree.N_LEV`）、`cpolicy.run_batch` 读 `answers[v]`。grep 确认训练路径上没有第二处重新分档（`soft_edges(` 其它调用只在诊断脚本里）。
- `train.py:85` 把新键加进 `DEFAULTS`，`main()` 的未知键检查通过；`sbatch_arms.sh:36-37` 的合并顺序 extra → best params → arm，`hard2` 只覆盖 `soft_levels` 与 `soft_fixed_edges`，`answer_model`/`length_term`/`node_prior` 与完整方法相同。旧 trial 的 `summary["cfg"]` 没有该键，`load_trial` 合并 DEFAULTS 后为 `None`。

### 2. 共享 replay：旧行为不变；`level` 与训练期 `query_level` 同一限制

- `src/qtl/replay.py:10-28` 与被删掉的 `sim_monotone.run_source`（diff）逐项相同：同样按 T 排序、按 `eval_chunk` 分块、`order="eig"`、`copy_mode="both"`、补 `voi`；`level=0` 时 `allowed=None`，与旧代码不传 `allowed` 等价；`truncate` 原样搬移。`final_curves.py` / `sim_monotone.py` 只改 import。
- `level > 0`：`replay.py:20` 用 `qtree.level_nodes(store.T[v], level)` 建 `allowed`，与 `train.py:216-217` 相同函数、相同 T，经 `Evaluator.run`（`train.py:153-154`）同样传给 `cpolicy.run_batch` 的 `allowed`，在 `cpolicy.py:136` 以同一方式限制候选；`allowed` 列表顺序与 `chunk` 顺序一致。

### 3. `policy_replay.py`

- `load_trial` 与 `final_curves.py` 相同；`copy_fn=None`，与报告曲线一致（`runs/20261101_r6_length_table/final_curves/{test,val}_selected/<corpus>/summary.json` 的 `copy` 均为 False，`cap` 8）。
- `--max-calls 8` 与报告曲线的 32 次后 `truncate(8)` 等价：策略确定性、每个 split 单独排序分块（与 `final_curves` 相同），前 8 次提问、`scores[:9]`、`eig_sum/n_cand[:8]` 一致。
- 小提醒（不影响本轮）：`:46` 只断言 `soft_levels == 8`，没有断言 `soft_fixed_edges is None`；现在两个 hard 臂都是 2 档，能被挡住，建议加一句 `assert cfg.get("soft_fixed_edges") is None`。

### 4. E3：逐视频配对与分数下标

- `e3_level_matched.py:53`：`k_v = min(8, len(lev[v]["eig"]))`，`len(eig)` = 单层 run 实际提问数 = min(max_calls, 该层可问节点数)，即题设的 k_v。
- `cpolicy.run_batch:147-150`：`scores[0]` 在任何提问前记录，之后每步在读入答案后记录，所以 `scores[k]` = k 次提问后的后验；`scores` 长度 ≥ 提问数 + 1（按全树可问节点数记录），`lev[v]["scores"][k_v]` 与 `full[v]["scores"][min(k_v, len(full eig))]` 下标都合法；全树可问节点 ⊇ 单层节点，所以 `len(full eig) ≥ k_v`，方法侧总能读到恰好 k_v 次。
- 两边同一 checkpoint、同一答案表（同一 `load_trial`）、同一缓存答案；test 指标走 `hc.run_evaluator`。

### 5. E4：确定性部分复现报告曲线；随机化只用 validation 调用数

- 确定性部分（`e4_budget_audit.py:58-72`）与 `final_curves.py:80-87` 同一串调用：`truncate(cap)` → `stop_rules.rule_values` → `add_quantile_rules` → `policy.calibrate` → `policy.stop_calls` → `scores[calls]`，输入是同一 trial 的同一前 8 次 run（见第 3 项），所以应逐数复现 `curve.json["rule"]`。
- 随机化：`cands` 与 `calibrate` 内部候选集相同（同一组 validation 规则值，降序）；`i_lo` = 最大的满足 `vmean <= B` 的下标。本机用 300 组随机 run × 7 个预算验证：`cands[i_lo]` 与 `calibrate` 返回的阈值、`vmean[i_lo]` 与其 validation 均值全部一致，`vmean` 沿候选单调，`vmean[i_lo+1] > B`，0 处不符。`q = (B - m_lo) / (m_hi - m_lo)` 只由 validation 调用数决定，期望 validation 均值恰为 B；test 只用来执行停止规则与打分，没有标签、没有 test 调用数进入阈值或 q。`vmean[i_lo] == B` 时不随机化，正确。

### 6. `baseline_infer.py`：模型重建与官方一致（除 F2 的分支外）

- Fed-WSVAD：`Model(512, vl, prompt_prefix, prompt_postfix, 512, visual_layers, 1, attn_window, device)` 与 `fed_wsvad_adapter.py:168-169` 的训练构造完全相同，参数直接读该 checkpoint 的 `train_meta.json`；打分函数就是写官方 `scores.jsonl` 的 `fa.score_ids`（`score_align`），`load_state_dict` 默认 strict。HCS 三 seed 与 DeHate 三 seed 的 `train_meta` 均为 layers 2 / prompt 20,5 / 256 / 64。
- DSANet：用 `frozen_config.json["params"]` 组 argv 再经 `option.resolve`，与三个 seed `train_meta.json["args"]` 逐键比较，只有 `device/lr/batch_size/max_epoch/seed/out_dir` 不同（纯训练期参数），架构参数（embed/width/head/layers 1/prompt 10,10/decoder_depth 8/num_prototypes 16/DNP_use/attn_window 32/visual_length 256）一致；strict 加载；forward 调用 `model(visual, None, PROMPT_TEXT, lengths, DNP_use)` 与 `dsanet/infer.py` 相同；`ds[i]` 加 batch 维后 `prepare_test_item` 的 reshape 路径与 DataLoader(batch 1) 相同。
- MACIL-SD HateMM 的转换文件：`test.jsonl` 与官方 `scores.jsonl` 的 `score_av` 214 个视频逐值相同（最大绝对差 0）；`train_infer/evidence_manifest.json` 记录其 checkpoint 为同一 `model.pth`，`val_or_test_scores_used_as_train: false`；train 744 / val 109 个视频。

### 7. E1：其它检查项

- 无 test 泄漏：每个 (seed, variant) 在 lr × epoch 上按 validation (AP + ROC) / 2 选一个 state（`e1_fusion.py:173-206`，`best` 跨两个 lr 保留），test 只在 `:212-219` 打分一次，走 `hc.run_evaluator`；head 只在 `ids["train"]` 上做梯度。
- validation / test 的节点 = `run["asked"][:8]`（`asked_nodes`，`:116-124`），来自 test-selected trial 的 L0 replay，与 `ours_fixed8` 的 trial 由 `launch/run_round1_cpu.sh` 按 seed 顺序对齐；缓存里无答案的节点被跳过，与方法处理"调用但无观测"一致。建议加一句 `assert rp["trial"] == ours[name]["trial"]` 防止传错目录。
- 损失与读出：`:191-194` 为 BCE(视频标签, 前 k 个逐秒 logit 的均值)，k = T // 16 + 1；读出 `sigmoid(f_t)` 逐秒，与 docstring 一致。
- 对齐：树用 `qtree.tree(store.T[v])`，与 replay 时相同的 T；token 覆盖 `[a, b) ⊂ [0, T)`，`index_add` 下标合法；baseline 分数按 `fit_len` 截断 / 末值填充到 T（与 14(f) 控制相同）；`feat` 变体用同一 `Store` 的 crop 0 I3D + VGGish + BERT 行，长度 T。

## 其它说明（非规则 6 范围，记录备查）

- `tandem.html` 出现在仓库根目录（`git status` 未跟踪），根目录白名单不含它，与本次审查无关。

---

## 修复确认（2026-11-08，commit ec64449）：PASS

- **F1 已修**（`e1_fusion.py:75-98, 196-222`）：`Head.forward` 的输入只剩 token 和 / 最大值、覆盖数、视频上下文（和可选特征），`out` 层输入维 68 / 100，不含 `z_t`；本机用随机输入验证两组不同 `z` 下 h 输出逐元素相同。h 只在 `ids["train"]` 上做梯度（top-k MIL BCE，k = T // 16 + 1）；`alpha ∈ {0, .25, .5, .75, 1, 1.5, 2}` 与 lr、epoch 一起按 validation (AP + ROC) / 2 选（`:205-214`，`hc.frame_metrics`，无梯度）；test 用选中的 state 与 alpha 只打分一次（`:220-225`，统一评测器）。样本内 baseline 分数不再进入任何学习到的权重。
- **F2 已修**（`baseline_infer.py:70-72, 109-111`）：DSANet 改读 `o[1]`，`sigmoid(logits1[0:n].squeeze(-1))`，与 `dsanet/infer.py` 的 `score_mlp` 完全一致（该分支不除 temp）；`--ref` 比较对 dsanet 读 `score_mlp`，其它方法读 `score_align`；docstring 已改并注明 14(f) 控制用的是 score_align。
- 两条建议的断言已加：`policy_replay.py:47`（`soft_fixed_edges is None`）、`e1_fusion.py:171`（replay 的 trial 与 `ours` 的 trial 同一路径）。
