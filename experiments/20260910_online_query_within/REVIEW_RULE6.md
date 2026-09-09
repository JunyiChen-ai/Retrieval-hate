# 规则 6 代码复核：online query + within（模块一第 2 轮，第 0 版）

复核对象：未提交工作树（`git diff` 的 `src/hier_evidence_common.py`、`src/interval_evidence_hmm.py`；新目录 `experiments/20260910_online_query_within/` 的 `train.py`、`acquire.py`、`search.py`、`screen.py`、`hmm_eval.py`、`launch/run_search.sh`、README 第 1–2 节）。复核日期 2026-09-10，只读，CPU 检查（Python 3.11.8，torch 2.7.1+cu128，multiprocessing 默认 `fork`）；临时脚本在 scratchpad（`check_hmm.py`、`check_r3.py`、`check_loss_fork.py`、`old_ieh.py` = HEAD 版 HMM），未写 `runs/`、`data/`。当前无 search.py / train.py 进程在跑；`runs/20260910_online_query_within/` 只有 `hmm_only/`。

## 结论：**必须修一处**（修完即 PASS，不需重开 review）

### 必修 1：`IntervalEvidenceHMM.fit` 在没有任何细窗观测时把 `r_f` 设成 0.001，而不是按注释"保留参数"

- `src/interval_evidence_hmm.py` `fit()` 循环前保留了旧代码的负例矩估计初始化：`if neg_videos: self.r_f_z[0] = (neg_f + 1e-3) / max(neg_nf, 1)`（`r_c` 同式）。`neg_nf == 0` 时得到 `1e-3 / 1 = 0.001`。循环内新加的"该族无观测则跳过 M 步"分支随后原样保留这个 0.001。
- 本轮**每个 online trial 的 `refit("init")` 都走这条路**：`allowed = {}`，`bin_train` 把正例、负例的全部细窗都遮成 MISSING，`neg_nf = 0`。实测（合成数据，CPU）：新代码 `q_f .8000 / r_f .0010`（旧代码是 1.0 / 1.0，更差，但本轮不应以旧为准）；代码注释与 README 第 1 节 D 的说法是"a family with no observed verdict at all keeps its parameters"，即应为初始值 (.8, .1)。
- 后果（都在默认臂、每个 trial）：
  (a) epoch 1–5 的 validation scaffold（`val_masks` 揭开 bit-reversal 前 4 窗）用 `r_f = .001` 算后验，影响这几个 epoch 的 checkpoint 判据值；
  (b) epoch 5 的第一次获取事件：eoc 的两个反事实 scaffold 用这组参数。实测同一窗 P(s)：`r_f=.001` 下"假设答 1"→ 1.000、"答 0"→ .547；`r_f=.1` 下 → .980 / .572。所有候选共用同一尺度，排序变化应当很小，但它决定每个训练视频的第 1 个窗，进入后续全部 allowed 集合。
  epoch 5 之后每视频至少 1 个细窗观测，M 步正常，不再受影响。
- 修法：循环前两行各加观测数条件——`if neg_videos and neg_nf > 0:` 更新 `r_f_z[0]`，`if neg_videos and neg_nc > 0:` 更新 `r_c_z[0]`。`regimes=1` 有观测时数值不变（该路径与旧代码逐位一致，见下"已确认" 6）。影响面只有 init 拟合，第 1 轮实验的 `coarse4_train` 臂已归档不重跑。

## 已确认无问题的项目

### 1. 泄漏
- test 帧标签只进 `hc.run_evaluator`；`labels`（视频级）只在 `window_targets`（train 视频）与 `fit_hmm`（train 视频）里按 train id 索引；`hate_ids` 只用于 validation 的 within。
- test 裁定：`Acquirer.run_video` 从 `initial=None`（只有 4 粗块）起，`bf[w] = bf_true[w]` 只在选中之后执行（`acquire.py:205`）；eoc 打分只用 `bf2[c] = 0/1` 两个假设（`:189-193`），权重 `pred` 来自当前状态的 content logit（`window_bags(clog[0], ...)`）或 HMM 预测，都不读候选的真实裁定；`localization`/`conflict`/`entropy` 同样只用 HMM 后验或模型输出。训练期获取事件用同一 `run_video`（`initial={v: sorted(allowed[v])}`），同样成立。
- checkpoint：`val_criterion` = validation 固定 uniform 掩码（`uni[:4]` = bit-reversal 前 4 窗）的 (AP+ROC+within)/3（`train.py:109,172,297-308`），与 README 第 1 节 E 一致；停止点 `tau_val` 只从 validation 网格选（`:382-398`）。
- `ScaffoldCache` 为全部视频建了 34 裁定全开的默认 scaffold，但 train 走 `mask_sampler`、val/test 走 `EvalDataset(masks=...)`（所有 id 都在 masks 里）、`Acquirer.video_inputs` 不用 `f_a`，默认 scaffold 从未被输入。

### 2. DataLoader worker 看到获取事件后的状态
- `train_loader`、`val_loader` 都是非 persistent、`fork`；每个 epoch 的 `for batch in train_loader` 重新 fork worker，获取事件发生在 epoch 循环体之后，下一 epoch 的 worker 复制的是更新后的 `allowed`、`policy_order`、`state["hmm"]`、`cache.masked_fn`（`refit` 原地换属性，`cache` 对象不换）。
- 实测（闭包引用 dict + 对象属性、`num_workers=2`、非 persistent）：三个 epoch 依次读到 `([], h0, v0)`、`([0], h1, v1)`、`([0,1], h2, v2)`。
- `Acquirer` 在每次事件时新建（`train.py:312`），拿到当前 `state["hmm"]` 与 `cache`；`self.Tm` 在构造时按当前 HMM 算。

### 3. 窗损失目标与 `w_rows` 对齐
- `window_targets`（`train.py:214-235`）：负例全部 30 窗目标 0；正例只对 `allowed[vid]` 中本次输入为 MISSING 的窗给目标（裁定或 `hmm.infer(E ∪ b_w)["p_hf"][w]`），其余 NaN；`allowed` 为空时全 NaN。目标窗的裁定不在输入 `b_input` 里。与 README 第 1 节 C 一致。
- `hc.window_bag_loss`：只取非 NaN 目标窗，`k = ⌈n_w/16⌉` top-k 均值 + BCE-with-logits，先每条目标窗均值、再跨有目标的条目均值，无目标返回 0。实测 toy（含 NaN 项、负例、无行窗、无目标条目）与手算一致到 1e-6；梯度到达 `content_logit`，无目标条目梯度为 0。
- `w_rows`：`TrainDataset` 里 `window_rows[:, None]` 与 `f_v`、`f_a` 走同一个 `process_feat(..., is_random=False)`，超长序列用 `uniform_extract`（按索引抽行，不平均），窗号保持整数且与行一一对应；不足则 0 填充，训练里按 `seq_len`（`_seq_len_of(f_v)`）截到有效行。`w_rows[:, :keep]` 与 `f_a[:, :keep]` 同截断。
- `eoc` 权重的 `window_bags` 与损失用同一个 `topk_div`、同一 content logit（五 crop 均值），窗行来自未截断的 `cache.window_rows`。

### 4. 调用记账与工作点
- 训练：`acq.run_split(train_ids, "eoc", 1, initial=allowed)`，`unobserved ≥ 26` 恒非空，每视频每次事件恰揭 1 个 allowed 之外的窗；`acq_epochs=[5,10,15,20]` → `train_total_per_video = 8`，`train_fine_observed_per_video` 记录实际均值（应恒 = 4）。`fixed_uniform_train` 臂 `allowed = uni[:4]`，同为 8。负例视频同样计入。
- 测试：`op_key = cap4_tau0`，`b_max=4 ∈ b_caps=[2,4,8]`，`eval_max_picks=18 ≥ 8`；`stop_index` 在 tau=0、gain ≥ 0 下恒返回 cap（实测 4），`mean_calls = 8`。停止规则在 cap = b_max 上按 validation 选最大 tau（AP、ROC 都 ≥ tau=0 值 − .005），与 README 第 2 节 E1 一致；曲线 budgets [0,2,4,8,12,18,30] = 4/6/8/12/16/22/34 次。

### 5. `Acquirer` 的 eval/train 与内存
- `Acquirer.forward` 每次 `model.eval()` + `no_grad`；训练期事件后 `model.train()`（`train.py:322`），每 epoch 开头也 `model.train()`；`best_state` 在事件前 deepcopy，不受影响。
- 分块参数与第 1 轮相同：`CHUNK=12`、`SEQ_T2_BUDGET=1e7`、每视频 `empty_cache()`；训练期每次事件每视频只跑 1 步（≤ 60 个反事实序列），比第 1 轮的 8 步少。

### 6. HMM：`infer`、regime EM、旧调用方
- `regimes=1` 与 HEAD 版逐位一致：合成数据、`positive_constraint` 开/关各拟合 8 轮，`lam01/lam10/q/r/p0`、`posterior`、`predictive_fine` 与 `infer()["pred_fine"]` 差 0.00e+00。
- `infer()["pred_fine"]` 正确：对 R=1 与 R=3 都验证 `pred_fine[w] = exp(logmarg(E ∪ {b_w=1}) − logmarg(E))`，R=1 误差 9e-16（R=3 在参数未退化的迭代同样成立，退化情形见建议 1）。
- EM：责任度 `rho` 由分量后验权按 regime 求和，`gamma_z` 按 regime 内节点权归一；M 步计数 `rho_z · P(h | z, data)`；负例闭式责任度与暴力乘积一致（未退化时）；`pi = (Σ rho_pos + Σ rho_neg) / N`；`p0`、转移率用混合后的 `gamma`/`xi`（共享参数）。合成数据 R=3 前 4 轮对数似然单调上升（−417.3 → −371.6）；全观测合成数据 15 轮单调、无 NaN。
- 所有策略统一走 `hmm.infer`（`Acquirer.state`）；`predictive_fine` 在新代码里无调用，且断言 R=1。
- 旧调用方：`experiments/20260908_adaptive_vlm_query/acquire.py` 用 `_emissions(bf, bc)`（默认 `z=0`）、`_fb`、`predictive_fine`（R=1）→ 可用；`20260907_c3_rev3_interval_evidence/interval_hmm_eval.py:61` 读 `itv.q_f` 等（property 返回 float）赋给 `verdict_hmm` 对象 → 可用；`adaptive_query_replay.py:124` 读 `hmm.q_f`；`hc.make_masked_scaffold_fn`/`make_scaffold_fn` 走 `posterior()`（已改为取 `["gamma"]`）；`_posterior_video` 无仓库内其它直接调用（grep）。`from_params` 读旧参数文件（无 `regimes`）→ R=1、`q_f=.8`；R=3 参数 round-trip 一致；R>1 时 `hmm.q_f = x` 断言（实测）。

### 7. 配置链路
- `DEFAULTS` 完整：`train.py` 的全部 `a.<key>` 与 `model.py` 的 `cfg.<key>`/`getattr(cfg, ...)` 都在 `DEFAULTS` 里（机械核对，缺 0 个；`fusion`、`w_fine` 两个键无人读，无害）。`main()` 拒绝未知键。
- `search.py`：`--extra-config` 键必须 ⊆ `train.DEFAULTS`，且在 `objective` 内与实际采样键 `sampled` 求交断言（第 1 轮建议已落实）。搜索空间 5 个标量与第 1 轮 `sample()` 逐行相同（仅注释不同）。`--no-within-prune` 默认 True → 不剪枝。目标 (AP+ROC+within)/3 在 `summary["test"]`（= cap4_tau0）上，与 README 第 2 节一致。
- `run_search.sh` `bash -n` 通过；`EXTRA_CONFIG` 数组展开与第 1 轮相同。
- 全部新模块 import 成功（CPU）。

## 建议（非必修）

1. **`regimes3` 臂（本阶段不跑）在 M 步给出恰为 1.0 的发射参数时 EM 会变 NaN**：`_neg_loglik_z` 算 `n0c · log1p(−r_c_z)`，`r_c_z = 1.0` 时对没有 0 裁定的负例视频得到 `0 · (−inf) = NaN`，随后 `pi`、`rho`、全部参数 NaN。恰为 1.0 是可达的（`q = cf/nf`、`r = (cf0+neg)/(nf0+negn)` 在某档全为 1 裁定时正好等于 1；README 第 4 节 HateMM 的过度触发档已拟出 `q_f = 1.0`）；实测合成数据 R=3 在第 5 轮左右 NaN。R=1 不受影响（有观测时 r、q 不会恰好 1；无观测时跳过）。若之后要跑 `regimes3`，先把 M 步后的 `q_*_z`、`r_*_z` 裁到 `[1e-4, 1−1e-4]`（或 `_neg_loglik_z` 用 `scipy.special.xlogy`），并核对 `hmm_only` 的参数文件没有 NaN。
2. `TrainDataset._rng` 在每个 epoch 重新 fork 的 worker 里都从 `seed + worker_id` 重启（实测三个 epoch 首个随机数相同）；掩码随机流每 epoch 一样，只因 shuffle 换了条目才不同。第 1 轮同样如此，不影响本轮与第 1 轮的可比性，记录在案。
3. 规则 7 原文的搜索目标是 (AP+ROC)/2、checkpoint 判据是 (AP+ROC)/2；本轮 README 第 2 节预注册改为 /3 含 within（计划已批准）。代码与 README 一致；STATUS 汇报时注明这是本轮的预注册偏离，避免与第 1 轮的 trial 目标值直接比较。
4. 若选中 epoch ≤ 20，测试用的 HMM（epoch 20 事件后重拟合）与该 checkpoint 训练时的 HMM 不同（发射参数随观测数略变）；与第 1 轮"round-1 HMM 配 round-1 模型"同类，`selected_epoch` 与 `hmm_params_epoch*.json` 都有记录，可事后核对。
5. `search.py` 的 `--no-within-prune` 是 `store_true` 且默认 True，不可能开启剪枝；`WITHIN_FLOOR` 与 `objective_unconstrained` 分支是死代码，无害。

## 判定
- 必修 1 修复后 PASS（`src/interval_evidence_hmm.py` 两行加观测数条件；`regimes=1` 有观测路径不变）。按规则 6 只确认修复，不重开 review；建议 1 在跑 `regimes3` 臂之前处理。
