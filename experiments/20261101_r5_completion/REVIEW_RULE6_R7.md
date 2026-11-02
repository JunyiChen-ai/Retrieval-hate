# 规则 6 code review（第 7 版：第 6 版 + 先验的视频标签项 `prior_label`）

**判定：FAIL。** 一项阻断：新加的项 `BCE(g, y)` 与第 6 版损失里已有的 `loss_g` 完全是同一个量（零膨胀链下 `loss_g` 本来就是不看回答的 −log P(G = y | x) = BCE(g, y)，回答似然项对 g 的梯度为 0），所以第 7 版代码实际做的是"把视频标签项的权重从 1 改成 2"，而不是 README 3.4 说的"在只经全部回答的后验学习 g 的目标上补一个不看回答的先验项"。README 3.4 的原因假设（答案表让后验对 g 依赖变小、g 梯度变小）与代码不符。数字本身不会错，但实验检验的机制与预注册写的不是一回事，必须先改 README（或改代码）再开搜索。另外三个消融 arm 会被新断言挡住。

- 审稿人：独立 agent（Fable 5.1），2026-11-02，一次性。审阅 `git show 3cdd319`（`src/qtl/train.py`、`experiments/20261102_r7_prior_label/*`）、`src/qtl/{ctree,model,data,search,checkpoint,backbones,inside_outside}.py`、`experiments/20261101_r5_completion/{final_curves,sim_monotone}.py`、`REVIEW_RULE6_R6.md`、r5 / r6 trial 的 `config.json` 与 `study_summary.json`。未跑训练；只在 CPU 上用三条合成视频（T = 5 / 9 / 12）跑了一次 `ctree.up` / `ctree.log_evidence` 的数值检查（脚本在本会话 scratchpad，不入库）。未改本文件以外的任何文件。

## 1. 阻断项

**1. 新项是 `loss_g` 的重复，机制没有以 README 描述的形式进入损失。** `src/qtl/train.py:386–388, 399–401`。

- 代码路径（r7 配置：`prior chain`、`chain_form zero_inflated`、`objective tree`、`node_prior true`、`g_head true`，与 r5 / r6 trial 的 `config.json` 一致）：`ctree.log_evidence`（`ctree.py:210–217`）返回 `l1 = w1n − logZ`、`l0 = w0n − logZ`，其中 `w1n, w0n = up(..., A3 = 0, ...)` 是**不带回答**的量；零膨胀链下 `up` 返回 `g + v1 − v1_prior` 与 `lp0`（`ctree.py:140–142`），A3 = 0 时 `v1 == v1_prior`、`lp0 == 0`，所以 `w1n = g`、`w0n = 0`，`l1 = log σ(g)`、`l0 = log(1 − σ(g))`。于是 `train.py:388` 的 `loss_g = −mean(where(y, l1, l0))` 就是 `BCE(g, y)`。
- 回答项 `lo1 = w1a − w1n = v1(A3) − v1_prior`、`lo0 = w0a − w0n = lp0(A3)`，g 在两处都消掉：`loss_ans` 对 g 的梯度恒为 0（回答只训练逐秒 logit s、节点势 phi 和链）。
- 合成数值检查（closed / free 边界 × 有无 phi 四种组合）：`|P(G = 1 | x, 无回答) − σ(g)|` ≤ 4.5e−8；`|loss_g − BCE(g, y)|` ≤ 1.2e−7；`∂loss_ans/∂g = [0, 0, 0]`；`∂total/∂g` 第 7 版 = 第 6 版 × 2.000，且等于 (σ(g) − y)/B。
- 结论：第 7 版的 `total = 2·BCE(g, y) + loss_ans (+ λ·CMAL)`。这与 README 3.4 的三句话冲突：(a) "训练时 g 只经 −log P(G = y | x, 全部训练回答) 学习"——代码里 g 从来只经不看回答的 BCE 学习，r5 / r6 都是如此；(b) "带长度的表……后验对 g 的依赖变小，g 得到的梯度变小"——g 的梯度与答案表无关，r5 与 r6 的 0 次差异不可能来自这条路径（只能来自 s / phi / 链 / 搜索选到的超参或 trial 噪声）；(c) "训练目标覆盖'全部回答'和'没有回答'两端、不加新超参"——实际只有"没有回答"一端，改动是一个未搜索的损失权重 1 → 2。
- 实际效果：`model.vid`（视频头）只被 BCE 项训练，Adam 下梯度翻倍几乎无影响；有影响的是共享编码器（`cma`、`fc_v`、`fc_a`、`att`）里标签项相对回答项与 CMAL 的权重变为 2 倍。预注册"机制检查 1"（0 次 AP / ROC 高于第 6 版）仍是"权重 2 对权重 1"的合法经验检验，但不是对原因假设的检验。
- 失败场景：搜索按现状跑完，若 Q1–Q3 过，归档与论文会把涨点归因于"补了先验标签项、覆盖两端"，而实际是损失权重改动；若不过，会错误地排除一个从未被实现的机制。
- 修法（主 agent 二选一，都要先改 README 3.4 再开搜索）：(i) 如实改写 3.4：第 7 版 = 第 6 版、视频标签项权重 2（删掉原因假设 (a)(b)，承认是损失权重超参）；(ii) 若要的真是"后验 + 先验两端"，需把 `loss_g` 改成带回答的后验 −log P(G = y | x, o)（即 `objective posterior` 那条路径的 `up(fo, s, g, A3)` 归一化形式）再加先验项——这是方法改动，按规则 4 重新提案。

**2. 新断言挡掉三个消融 arm。** `src/qtl/train.py:316–318` 要求 `prior == chain and chain_form == zero_inflated and objective == tree`；`experiments/20261102_r7_prior_label/launch/sbatch_arms.sh:45` 把 `study_summary.json` 的 `extra`（含 `prior_label: true`）与 arm 改动合并，下列 arm 在 `train.py` 启动即 AssertionError，记 FAILED，不产生数字（不会出错数，但 arm 表缺项）：

| arm（`sbatch_arms.sh:22–35`） | 触发条件 | 应改为 |
|---|---|---|
| `c_label` `{"objective": "label", "node_prior": false}` | objective ≠ tree | 加 `"prior_label": false`。该 objective 下 `loss_g` 已是 BCE(g, y)（`train.py:345–362`，A3 = 0），再加一次同样只是权重 2。 |
| `e_independent` `{"prior": "independent", "node_prior": false}` | prior ≠ chain | 加 `"prior_label": false`。该路径 `train.py:397` 的 `loss_g` 本来就是 `F.binary_cross_entropy_with_logits(g, label)`。 |
| `g_coupled` `{"chain_form": "coupled", "node_prior": false}` | chain_form ≠ zero_inflated | 加 `"prior_label": false`。耦合链下 `l1 = g + v1 − logZ` 不等于 log σ(g)，BCE(g, y) 在那里是另一个量，断言挡住是对的。 |

其余 11 个 arm（`no_node`、`no_length`、`per_state_len`、`words`、`b_flat`、`d_bfs`、`f_joint`、`lvl4/8/16`、`noback`）都满足断言；`noback` 的 `ConstPrior` 返回 g 为 (B,) float32（`model.py:141`），BCE 形状匹配。

## 2. 已确认无误的项（对应任务清单 1–5）

1. **同一个 g**：`train.py:334` 的 `g` 直接作为 `g_tree`（`objective tree` 时 `direct = True`，`train.py:336, 340`，不 detach）进入 `ctree.log_evidence(fo, s_tree, g_tree, A3, chain, phi)`（`train.py:386`），`up` 里 `g + v1 − v1_prior`（`ctree.py:142`）就是零膨胀链 P(G = 1 | x) 的 logit；`train.py:401` 的 BCE 用的是同一个张量。有 phi（`node_prior true`）时 phi 同时进入 `v1` 与 `v1_prior`（`ctree.py:139, 141`），P(G = 1 | x, 无回答) 仍精确等于 σ(g)（合成检查 ≤ 4.5e−8）。形状 / dtype：`g` (B,) float32（`model.py:84`），`label` (B,) float32（`data.py:233`）。每个 batch 加一次（`train.py:400–401`），在 `total.backward()`（`train.py:416`）之前；只涉及 g 与 label，不碰 `am`（`anchored_shared` 时 `am.requires_grad_(False)`，`train.py:261–262`，且不在优化器里）、不碰 `Evaluator` / `cpolicy` / `opolicy` / `stop_rules` / `evaluate`（diff 未动这些文件）。
2. **`prior_label false` 下 r5 / r6 不变**：diff 对 `src/` 只有三处——`DEFAULTS` 新键（`train.py:85`）、受 `cfg["prior_label"]` 保护的断言（316–318）与加项（400–401）。`checkpoint.load_trial`（`checkpoint.py:15–16`）对旧 `summary["cfg"]`（无 `prior_label`）用默认 False；`train.main` 的未知键检查（`train.py:568–569`）对旧 `hparams.json` 不受影响；r6 目录的 `search.py` / `sbatch_arms.sh` 的 `extra` 不含该键。
3. **r7 主配置过断言**：`experiments/20261102_r7_prior_label/search.py:10–11` 的 `EXTRA_DEFAULTS` + `DEFAULTS`（`prior chain`、`chain_form zero_inflated`、`objective tree`）满足 `train.py:316–318`；`src/qtl/search.py:67–68` 的 `known` 含 `prior_label`；`sbatch_search.sh:25` 调 r7 `search.py` → `train_entry` = r7 `train.py` → `qtl.train.main()`，配置经 `hparams.json` 传入（含 `prior_label true`）。arm 见阻断项 2。
4. **落位与标记**：四个启动脚本的 `O` / `S` 都是 `runs/20261102_r7_prior_label`；`search.py` 写 `<out-root>/<corpus>/seed<s>/`；arms 写 `$S/arms/<corpus>/seed<s>/<arm>/`；`final_curves.py` 写 `runs/20261102_r7_prior_label/final_curves/<tag>/<corpus>/`（`final_curves.py:55`，trial 名取 `trial.parents[2].name` = `20261102_r7_prior_label`）；`sim_monotone.py` 写 `runs/20261102_r7_prior_label/sim_monotone/<corpus>/<trial>/`。r7 目录内 `grep 20261101_r6_length_table` 只命中 README 对第 6 版结果的引用。DONE / FAILED：`sbatch_search.sh:30` 按各 seed 的 `search.FAILED` 文件；`sbatch_arms.sh:38, 56–59` 同时要求 `search.DONE` 与 `study_summary.json`（r6 review 发现 2 已修），FAILED 数 = arm `.FAILED` 文件数 + 缺 `search.DONE` 的 seed 数；`run_final_curves.sh:16–28` 任一步失败置 `fail=1`；`run_sim.sh:23` 按日志中的 `Traceback`。
5. **重载链**：`prior_label` 不改保存的 `model` / `am` / `chain`（`train.py:442–445`）；`load_trial` 用 `make_model` → `backbones.py:8` → `inside_outside.make_model`，`macil` 返回 `LegacyPriorNet` = `qtl.model.PriorNet`（`inside_outside.py:12, 136`），`strict=True`；`AnswerModel(theta, len_mu, len_sd, length_term=True, 2, [0])` 后 strict 加载 theta 与 omega（`anchored_shared`）；`final_curves.py` / `sim_monotone.py` 只读 `cfg["categories"]`、`cfg["eval_chunk"]`、`cfg["answer_source"]`，不读 `prior_label`。与 r6 trial 完全同一条路径。

## 3. 非阻断

1. `sbatch_arms.sh` 没有"去掉第 7 版新项"的 arm（`{"prior_label": false}`，即第 6 版在第 7 版超参下）。规则 14(g) 要求每个部件有消融，若第 7 版被采用此 arm 必须有；按阻断项 1，它等价于"权重 1 对权重 2"。
2. `train.py:418` 的 `tot[0]` 只累计 `loss_g`，不含第 7 版加的那一项，`run.log` 与 `history["loss_g"]` 里显示的 g 损失是实际的一半。只影响日志读数。
3. `run_final_curves.sh:6–7`、`run_sim.sh:6–7` 头部的启动命令把 stdout 重定向到 `$O/final_curves/` 与 `$O/sim_monotone/` 下，而 `mkdir -p` 在脚本内才执行；目录不存在时重定向先失败。与 r6 相同（r6 已跑通，说明当时目录已存在）。运维问题。
4. `experiments/20261102_r7_prior_label/README.md:5` 与 `train.py:87–89` 的注释同样写着"加到 −log P(G = y | x, 全部训练回答) 上"，随阻断项 1 一并改。
