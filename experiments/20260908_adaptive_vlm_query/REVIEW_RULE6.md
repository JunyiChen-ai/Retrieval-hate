# 规则 6 代码审查：自适应 VLM 查询模块（证据 dropout + EOC 获取）

审查日期 2026-09-08。范围只限规则 6：机制是否真的进入 forward / loss / 最终分数、train/validation/test 泄漏、特征/时间/标签/split 对齐、超参与 checkpoint 加载链、是否调用统一评测器。不审风格、健壮性、理论。

审查对象（工作树，未提交）：`experiments/20260908_adaptive_vlm_query/{README.md,model.py,acquire.py,train.py,search.py,launch/run_search.sh}`，以及 `src/hier_evidence_common.py`（`ScaffoldCache.build`、`TrainDataset.mask_sampler`、`EvalDataset.masks`、`make_masked_scaffold_fn`，commit 0476eca 相对 76ef6f0 的全部改动）和 `src/interval_evidence_hmm.py`（`posterior_gamma`、`summarize_gamma`、`predictive_fine`）。对照：`experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/train.py`（修订 4，其 REVIEW_RULE6.md 已 PASS）。

数值检查脚本（只读、CPU、未训练模型、不跑训练）：
- `/tmp/claude-135258174/-home-jehc223-Retrieval-hate/85e4ce56-9930-4e7b-937a-61b295613068/scratchpad/check_adaptive.py`（(b)(c)(d)(e)(g) 与初步计时）
- `/tmp/claude-135258174/-home-jehc223-Retrieval-hate/85e4ce56-9930-4e7b-937a-61b295613068/scratchpad/check_adaptive2.py`（(a) 修正版、conflict 内容 logit、长视频计时、评测器计时）
- `/tmp/claude-135258174/-home-jehc223-Retrieval-hate/85e4ce56-9930-4e7b-937a-61b295613068/scratchpad/check_adaptive3.py`（审查中途 `train.py`/`acquire.py` 被改后的复核：`run_video(initial=...)`、τ_val 选择逻辑）

审查中途（00:54）`train.py` 与 `acquire.py` 被改动：`Acquirer.run_video/run_split` 加 `initial` 参数（训练期策略从 4 个种子窗起跑，picks 全是新调用）；`train.py` 加 validation 上的 (cap, τ) 网格与预注册的 τ_val 选择（`summary["stop_rule"]`）。本报告以改后的版本为准，改动部分已重新数值复核（第 6、7 节）。`src/`、`model.py`、`search.py`、`launch/run_search.sh` 与 HEAD 一致（`diff` 无输出）。
- 旧版 `src/hier_evidence_common.py` 取自 `git show 76ef6f0`，存为同目录 `hc_old.py`（修正其 `REPO_ROOT/TEXT_ROOT` 指回仓库后比较）。

输入：HateMM，6 个视频（train 正例 hate_video_100/103、train 负例 non_hate_video_1、val hate_video_105、test hate_video_1/10），HMM 参数取 `runs/20260907_c3_rev3_interval_evidence/adaptive_replay/hatemm/hmm_params.json`（interval、normalized_time、positive_constraint），模型 = `ERCA(DEFAULTS, prior_scale 2)` 随机初始化后所有参数加 N(0, .05²) 噪声（打破 β / cell 的零初始化，使证据列真的影响输出），`eval()` 模式。

## 结论

**PASS，无必修项。** 损失、checkpoint 规则、评测链与修订 4 相同（差异只有遮蔽）；`src` 改动不动默认路径（逐元素差 0）；遮蔽语义、采样器、HMM 拟合、获取策略、调用计数、评测器调用全部按 README 实现；无 val/test 泄漏；τ_val 只看 validation。下面第 11 节列 4 条不阻塞的注意事项（第 3 条是抄消融表时的读数位置，README 第 3 节已写明）。

## 1. 与修订 4 训练器的差异

把两边 `for epoch in range(a.max_epoch): ... selected epoch` 的循环去缩进后 `diff`（`scratchpad/loop_rev4.txt` vs `loop_new.txt`）：

| hunk | 内容 | 是否改变机制 |
|---|---|---|
| `lam = min(lamda_cma, ...)` → `min(a.lamda_cma, ...)`；`lambda_block` → `a.lambda_block`（两处） | 修订 4 里这些变量是"消融臂覆盖后的值"，新版没有 `no_block/no_cmal` 臂，直接读 config | 否（full 臂两边同值） |
| `val_scores = ...; vm = frame_metrics(val_scores...)` 合成一行；日志加 `tag` 前缀 | 格式 | 否 |

其余逐字相同：`BCELoss(mmil, label)`、`CMAL(mmil, a_log, v_log, seq_len, v_out, a_out)`（`fix_rep_swap=False`）、`block_bag_loss(model.last_content_logit, f_a, seq_len, label, topk_div)`、`crit = (val AP + val ROC)/2`、`if crit > best` 存 `best_state`、结束 `load_state_dict(best_state)`。优化器 Adam + CosineAnnealingLR(T_max=sched_tmax) 相同。

`model.py` 相对修订 4 `model.py` 的 `diff`：docstring；`STRUCT_ARMS` 缩为 `("full", "no_missing_state")`；`EvidenceEncoder` 改为 `Embedding(6 或 4, hid)` + `Linear(2, hid)`（零初始化 cell 同修订 4）；`ERCA.__init__` 里 `concat=False`、`bias_mode/ctx_mode` 从 config 读（默认 `key`/`rep` = 修订 2/4 骨干）。`BiasedMultiHeadAttention`、`EvidenceRoutedCMA`、`ERCA.forward`、`bag` 无改动。

## 2. (a) `src` 改动不动默认路径

| 比较（6 视频） | 最大绝对差 |
|---|---|
| `ScaffoldCache` 旧版 vs 新版（不传 `masked_fn`）：`items[vid][0]` 全 f_a_ext（VGGish+BERT+6 列 scaffold） | **0.0** |
| 旧版 vs 新版（传 `masked_fn`）：`items` | **0.0**；`n_seconds`、`snip` 相同；`window_rows` 差 0 |
| 新版不传 `masked_fn` 时 `.base` 为空 | True |
| `build(vid, 全 30 条裁定)` vs 旧版存储的默认 f_a | **0.0**（遮蔽路径在"全观测"时退化为原路径） |
| `TrainDataset` 旧 vs 新（无 `mask_sampler`）5 个 item 的 f_v/f_a/w/label | **0.0** |
| `EvalDataset` 旧 vs 新（无 `masks`）6 个 item | **0.0** |
| `EvalDataset` 传 `masks` 但不含该视频 | 与旧版差 0.0（只对给了 mask 的视频重建） |

`git diff 76ef6f0 HEAD -- src/` 确认：`make_scaffold_fn`、`scaffold_rows_interval`、`block_bag_loss`、`score_split`、`fit_hmm`、`run_evaluator` 无改动；`interval_evidence_hmm.py` 只新增三个方法，`fit/posterior/_emissions/_fb` 无改动。修订 3/4 的代码路径不受影响。

## 3. (b) 遮蔽语义

| 项 | 证据 |
|---|---|
| `masked(bf, allowed)`：未问窗 = −1，已问窗 = 真裁定 | True（allowed = {0,15,7,22}） |
| `make_masked_scaffold_fn`：b_fine 列在未问窗的行 = −1，已问窗的行 = 真裁定 | True / True |
| 粗块列 = 完整粗裁定（永远观测） | True；块索引列不变 True；VGGish+BERT 块差 0.0 |
| ell / P(s) 用 MISSING 发射重算 | 遮蔽后 P(s) 与全观测 P(s) 最大差 .453（确实重算）；未问窗的发射行 `_emissions` 全为 1（不发射）True；ell 列 = logit(P(s)) 差 9.6e-8；p_h 列 = 遮蔽后块后验，差 1.8e-8 |
| 全缺失时后验不依赖真细裁定 | 差 0.0（只由粗块决定） |
| `EvidenceEncoder`（6 格）索引：(bf,bc) = (−1,0),(−1,1),(0,0),(0,1),(1,0),(1,1) | [0, 3, 1, 4, 2, 5]，即 3·bc + (bf+1)，与 README 一致；嵌入表 (6, hid)；e(−1,bc) ≠ e(0,bc) True |
| `no_missing_state`（4 格） | 同序索引 [0,1,0,1,2,3]：−1 落到 0 的格；e(−1,bc) − e(0,bc) = 0.0 |

## 4. (c) 训练期证据 dropout 采样器

采样器是 `train.py` 里 `train()` 内的闭包（第 240–248 行），脚本按原文逐行转录后测试（4000 次抽样）：

| 项 | 证据 |
|---|---|
| Round 0（mix = 0，allowed = 4 个种子窗）：暴露 allowed 之外窗的次数 | **0** |
| Round 0 保留数分布 0..4 | [843, 805, 820, 787, 745]（均匀，`randint(0, |S|+1)`） |
| 暴露的值 = 真裁定 | True |
| Round 1（mix = .5，allowed = 种子 4 + 策略 4）：allowed 之外 | **0** |
| Round 1 是策略顺序前缀的比例 | .637（= .5 前缀分支 + 随机子集恰为前缀的份额：m=0、m=8 必为前缀各 1/9，合计约 .61–.64，符合） |
| Round 1 保留数 0..8 | [450, 405, 430, 486, 461, 452, 438, 429, 449] |
| `no_dropout`：`lambda vid, rng: binary[vid][0]` | 等于全 30 条 True |
| `coarse4_train`：allowed = ∅ → `randint(0,1)` = 0 | 全 −1 True |
| 每 worker RNG：`RandomState(seed + worker_id)`，worker 0–3 首 5 个整数互不相同 | True；同 worker 重建后流可复现 True |

`train.py` 第 238 行：`mix = prefix_mix if r > 0 else 0.0`，round 0 只有随机子集。`policy_order[vid] = seed_w + picks(去重)`，`order` 再按 `allowed` 过滤，所以前缀只含 allowed 内的窗。

## 5. (d) HMM 拟合与 checkpoint 选择不泄漏

静态（`train.py`）：
- 第 226 行 `bin_train = {v: (masked(binary[v][0], allowed[v]), binary[v][1]) for v in train_ids}`，第 227 行 `hc.fit_hmm(corpus, train_ids, labels, bin_train, ...)`；`fit_hmm` 内 `pos_ids/neg_ids` 只取 `train_ids` 且 `v in binary`（= bin_train）。val/test 视频不在 `bin_train`，其裁定不进 `fit`。每轮重拟合（round 1 用 `allowed |= picks` 后的集合）。
- 第 219–220 行 `val_masks = masked(bf, uni[:b_max])`，`uni = bit_reversal_order(30)` = [0,15,7,22,3,18,...]，确定性；`train_one_model` 的 `val_loader = EvalDataset(val_ids, cache, masks=val_masks)`，每个 epoch 用它选 checkpoint。策略在 val 上只在训练全部结束后跑一次（第 316 行 `vruns`，12 步网格），写 `summary["val"]` 作参考，不参与选点。
- test 裁定只经 `Acquirer.run_video` 的 `bf[w] = bf_true[w]`（每步一个窗）暴露；`fixed34` 是显式的"全 34 次"对照。默认 cache 项虽含 val/test 全裁定（`make_scaffold_fn(..., "full")`），但 `EvalDataset` 对每个 val/test 视频都给了 mask（字典覆盖全部 id），`Acquirer.video_inputs` 只取 `n_seconds/snip`，`run_video` 全走 `cache.build`；默认项在 val/test 上不进模型。

数值：`fit`（5 轮 EM，12 正 + 12 负 train 视频）在遮蔽裁定上，把被遮蔽位置的真值随机打乱再拟合，7 个参数最大差 **0.0**（`fit` 确实跳过 MISSING）；遮蔽 vs 全观测的 q_fine = .9178 vs .9634（遮蔽确实改变拟合，非空比较）。

## 6. (e) `acquire.py`

| 项 | 证据 |
|---|---|
| (i) `forward` 批处理配对：`f_a.repeat_interleave(5)` × `f_v5.repeat(n)` 后第 i·5+c 行 = (f_a[i], f_v[c]) | True（逐行 `torch.equal`）；`view(n,5,-1).mean(1)` 因此是同一候选的 5 个 crop 均值 |
| (i) 3 个候选一次 `forward` vs 各自 `EvalDataset(masks)` + `hc.score_split` | 每个候选最大差 **[0.0, 0.0, 0.0]**；候选之间确有差异（.041） |
| `Acquirer.gamma`（`_fb(Tm(1.0), e)`）vs `hmm.posterior_gamma(bf, bc, n_seconds)` | 差 0.0（normalized_time 下与时长无关）；`summarize_gamma` 的 P(s) = `hmm.posterior` 的 P(s)，差 0.0 |
| (ii) `predictive_fine` = q_f·P(h_w=1) + r_f·(1−P(h_w=1)) | 差 0.0；已观测窗 w=0（b=1）P(h)=.929，pred=.888（对已观测窗无意义，代码只对 `cands`＝未观测窗取值） |
| (ii) EOC 公式：从粗块态手算 30 个窗的 Σ_b p(b_w=b) · mean_t \|σ(z(E+b)) − σ(z(E))\| | argmax = `picks[0]` True；max 与 `gains[0]` 差 3e-10；EOC 值 max/中位/min = .00797/.00727/.00357 |
| (iii) `stop_index`（gains=[.05,.02,.004,.03,.001]）：τ=.01 → 2；τ=0 cap 5 → 5；cap 3 → 3；τ=.1 → 0；cap 8 → 5（受 picks 数封顶） | [2, 5, 3, 0, 5] |
| (iv) `scores_at(runs, k)`：k=0/2/9 → scores[0]/[2]/[末] | [0, 2, 5] |
| (v) 起点 = 4 粗块、30 细窗全 −1：`scores[0]` vs `score_split(全 −1 mask)` | 差 0.0；`len(scores)=n_steps+1, len(picks)=len(gains)=n_steps` = (4,3,3) |
| 暴露的是真裁定：3 步后 `scores[3]` vs `score_split(mask=这 3 个窗)` | 差 0.0；picks 互不重复 True |
| (vi) `entropy / localization / uniform` 换一个随机骨干后 picks 相同 | True / True / True（只用 HMM）；`eoc / conflict` 换骨干后 picks 不同（[15,21,8,4,25,12] vs [18,11,24,5,15,21]；[7,8,9,…] vs [29,28,27,…]） |
| (vi) `conflict` 用 `last_content_logit`：`av_log − last_content_logit = prior_scale·ell/ELL_SCALE` | 差 1.2e-7（不含先验；`ctx_mode=rep` 下 c 在表示里，同修订 2/4 定义）；`csig[0]` = 五 crop 均值 σ(content logit)，eval 模式下差 0.0 |
| `uniform` 顺序 = `bit_reversal_order(30)` = [0,15,7,22,3,18,…]，是 0..29 的置换 | True |
| `random` 由 `run_split` 的 `RandomState(seed)` 驱动，seed 0/1 不同 | [[2,28,13,10],[17,21,10,19]] |
| **`initial=seed_w`（改后新增）**：`scores[0]` = `score_split(mask=种子 4 窗)` | 差 0.0；8 步 picks 不含种子窗 True，picks/gains 各 8 个，新 picks 数 = b_max = 8；`scores[8]` = `score_split(mask=种子 4 窗 + 8 picks)` 差 0.0 |
| `initial=None`（测试期默认）行为不变 | `scores[0]` = 粗块态差 0.0；前 3 个 picks [15,21,8] 与改前一致；`run_split(initial=seed_w)` 两个视频都是 8 个种子窗外的新 picks True |

补充：`hc.score_split` 结束时 `model.train()`；`Acquirer.forward` 开头 `self.model.eval()`，所以 `train.py` 里 fixed34/coarse4 评测后再跑策略不会带 dropout（实测 True）。

## 7. (f) 调用计数与 summary

`train.py`（改后版本）：
- 训练期：`calls["train_round0_per_video"]` = 4 + len(seed_windows) = 8（full / no_missing_state / round0_only），34（no_dropout / train34），4（coarse4_train）。两轮臂：`run_split(train_ids, "eoc", b_max, initial=seed_w)`，`picks = len(set(picks) − set(seed_w))`（因 `initial` 已把种子窗移出 `unobserved`，实测恒 = b_max），`train_total_per_video = 4 + 4 + mean(new picks)` = 16（τ = 0）。与 README 1.3 "8 + 平均新增 pick、τ = 0 时常数 16" 一致。
- 测试：`eoc_grid[cap_tau]["mean_calls"] = 4 + mean(stops)`，`calls_hist = bincount(stops, minlength=cap+1)`；曲线点 `calls = 4 + k`。
- validation 网格：`vruns = run_split(val_ids, "eoc", max(b_caps)=12)`，同一 (cap, τ) 网格，写 `metrics_val_eoc_*.json` 与 `results["val_eoc_grid"]`；`summary["val"] = val_grid["cap8_tau0"]`。
- τ_val（README 第 2 节）：`for tau in sorted(grid): if val AP ≥ AP(τ=0) − .005 and val ROC ≥ ROC(τ=0) − .005: tau_val = tau` → 取通过条件的**最大** τ（允许中间有不通过的 τ；脚本用合成网格验证：单调下降例 → .01，末位回升例 → .08）。`results["stop_rule"] = {tau_val, key, val: val_grid[key], test: eoc_grid[key]}`——τ 只由 validation 决定，test 只是在该 key 上读数。键格式 `cap%d_tau%g` 两边一致（`cap8_tau0`、`cap8_tau0.005`、`cap8_tau0.08`）。
- `summary["test"] = results["eoc_grid"]["cap%d_tau0" % b_max]`（默认 `cap8_tau0`，`b_caps` 含 8），不受 τ_val 影响。
- `search.py` 第 105–111 行读 `s["test"]["pooled_ap"/"pooled_roc"/"within_roc"]` 与 `s["val"][...]`，目标 `(test AP + test ROC)/2`；`best = max(COMPLETE, key=value)`；`--no-within-prune` 默认 True → `floor=None`，不剪 within。搜索空间 5 个标量同修订 3。

## 8. (g) 评测器

所有 `metrics_*.json` 都由 `evaluate_scores()` → `hc.write_scores` → `hc.run_evaluator` → `scripts/reproduction_baselines/eval_baseline_scores.py` 子进程产生（fixed34、coarse4、6 策略 × 预算、eoc 网格 21 个、val 操作点），没有第二份评测逻辑。`score_split` 与 `run_video` 都用 `snippet_index_for_seconds` 映射到 1 fps：test 视频 hate_video_1 / hate_video_10 的分数长度 (95, 95)、(39, 39) 与 GT 相同；评测器一次调用 n_videos 214、n_frames 29269，长度不符会 `ValueError`。

## 9. (h) 锁定消融启动器

`scripts/run_locked_ablations.sh` 的解析逻辑原样执行：id 通过正则，`trainer=experiments/20260908_adaptive_vlm_query/train.py`（实验目录直接命中，无需去后缀）。`train.py --ablation` 选项 = {full, no_missing_state, no_dropout, train34, round0_only, coarse4_train}，README 第 3 节的 4 个训练臂都在；`uniform_at_matched / no_stop / conflict` 是同一 run 的策略/网格读数，不是训练臂。

## 10. (i) 耗时（CPU 实测，4 线程）

| 项 | 实测 |
|---|---|
| eoc 8 步：57 行（39 s 视频）/ 193 行（130 s 视频） | 0.60 s / 3.27 s（≈ 17 ms/行） |
| eoc 12 / 18 步：193 行 | 4.11 s / 5.58 s（≈ 21 / 29 ms/行） |
| conflict / entropy / localization / uniform / random 各 30 步：193 行 | .16 / .15 / .40 / .14 / .14 s（HMM 部分是 numpy，不吃 GPU） |
| 一次 `hmm.posterior` / 一次 `cache.build` | .32 ms / .35 ms |
| 一次评测器子进程（214 test 视频） | 1.04 s |
| 平均时长：HateMM train/val/test 152/124/137 s（test 最长 1000 s，48 个 > 200 s）；HateClipSeg train/test 238/238 s（251/79 个） | |

外推（按 1 s ≈ 1.5 行，GPU = CPU × 20）：
- HateMM test：eoc 18 步 214 × 205 行 × 29 ms ≈ 21 min CPU → ≈ 1 min GPU；5 个对照 ≈ 214 × 1.0 s ≈ 3.5 min（大部分是 HMM，GPU 帮不上）；评测器 test 2 + 6 + 35 + 21 = 64 次 ≈ 1.1 min，val 21 次（109 视频）≈ .4 min；val eoc 12 步 109 × 186 行 × 21 ms ≈ 7 min CPU → .4 min GPU。**合计 ≈ 6.5 min（20×）**；若小 batch 只有 5× 加速也约 10 min，仍在 15 min 以内。HateClipSeg test 更小（79 个）：eoc ≈ 14 min CPU → < 1 min GPU，对照 ≈ 2.5 min。
- 训练期策略（每 trial 一次，8 步、从种子窗起）：HateMM 744 × 228 行 × 17 ms ≈ 48 min CPU → ≈ 2.4 min GPU（5× 时 ≈ 10 min）；HateClipSeg 251 × 358 行 × 17 ms ≈ 25 min CPU → ≈ 1.3 min GPU。
- 结论：评测阶段不会超过 ~15 min；整个 trial 是 2 轮 × 50 epoch 训练 + 上面两项，规则 7 的 1 h 门要以第一个 trial 实测为准（README 已按此写）。

## 11. 不阻塞的注意事项

1. **`Acquirer.Tm = hmm._transitions(1.0)` 只在 `normalized_time=True` 下正确**（DEFAULTS 固定 True、不在搜索空间，`make_masked_scaffold_fn` 用的是 `hmm.posterior(..., n_seconds)`，两者当前一致，实测差 0）。若以后把 `normalized_time` 改 False，获取端会静默用错时长；建议在 `Acquirer.__init__` 加 `assert hmm.normalized_time`。
2. **每 worker RNG 每 epoch 重新播种**：DataLoader 默认 `persistent_workers=False`，每个 epoch 重建 worker，`_rng` 重新 `RandomState(seed + worker_id)`，同一 worker 每 epoch 的"保留数序列"相同；因 `shuffle=True` 视频–mask 配对逐 epoch 变化，50 个 epoch 下每个视频仍看到不同子集。不改变结论，记录在案。
3. **`coarse4_train` 臂的数怎么读**：该臂的 `summary["test"]` 仍是 eoc cap8 τ0（测试时向从未见过细裁定的模型暴露 8 个细窗），"训练与测试都只用 4 个粗块"的数在 `results["coarse4"]` / `metrics_test_coarse4.json`；README 第 3 节已按此写明，抄表时照做。
4. 修订 4 审查里提到的 `hate_video_427` 排除、cohort 检查等不变；README 引用的 `REVIEW_RULE4.md`、`docs/20260908_adaptive_query_survey.md` 都在。

## 12. 必修项

无。
