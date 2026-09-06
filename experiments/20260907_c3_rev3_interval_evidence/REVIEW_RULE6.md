# 规则 6 代码审查：候选 3 修订 3（证据路由跨模态注意力 + 区间证据 HMM）

审查日期 2026-09-07。审查范围只限规则 6：机制是否真的进入 forward / loss / 最终分数、train/validation/test 泄漏、特征/时间/标签/split 对齐、超参与 checkpoint 加载链、是否调用统一评测器。不审风格、健壮性、理论。

审查对象（未提交的工作树版本）：`experiments/20260907_c3_rev3_interval_evidence/{model.py,train.py,search.py,launch/run_search.sh,interval_hmm_eval.py}`、`src/interval_evidence_hmm.py`、`src/hier_evidence_common.py`（改动部分）、`scripts/run_locked_ablations.sh`。对照上一版 `experiments/20260904_evidence_guided_attention/{model.py,train.py}`。

数值检查脚本（只读，未训练）：`/tmp/claude-135258174/-home-jehc223-Retrieval-hate/2df02dd1-cd60-4dfd-b172-74b0a62d3d46/scratchpad/{check_hmm.py,check_model.py}`，用 `~/miniconda3/envs/HateVideo/bin/python` 运行；下文引用的数字都来自这两次运行和一段真实数据检查。

## 结论

**FAIL：1 处必须修，在消融启动链，不在模型/训练/融合代码里。** 修好这一处后即 PASS。搜索本身（`launch/run_search.sh` → `search.py` → `train.py`）不经过这处，可以先开跑。

### 必修 bug

**B1. `scripts/run_locked_ablations.sh` 第 206–207 行的后缀剥离把本实验目录名剥坏，找不到 trainer，18 个消融臂一个都跑不起来。**
```
exp_dir="${experiment%_v[0-9]*}"
exp_dir="${exp_dir%_rev[0-9]*}"
trainer="experiments/${exp_dir}/train.py"
```
bash 的 `%_rev[0-9]*` 是 glob，`*` 匹配任意后缀，所以 `20260907_c3_rev3_interval_evidence` 被剥成 `20260907_c3`。实际验证：
```
$ e=20260907_c3_rev3_interval_evidence; x="${e%_v[0-9]*}"; x="${x%_rev[0-9]*}"; echo $x
20260907_c3
$ test -f experiments/20260907_c3/train.py   -> 不存在；archive/ 下也不存在 -> 第 212 行 test -f 失败，脚本退出
```
以前的实验 id 里 `_rev2` 都在末尾（`20260904_evidence_guided_attention_rev2_noprune` 也能碰巧剥成正确目录），这次 `_rev3` 在中间，第一次踩到。影响：README 第 3 节全部消融观察无法产生；不是静默出错，是直接启动失败。修法任选其一：把剥离模式限定为末尾（例如只剥 `_rev[0-9]` / `_v[0-9]` 这种确切后缀，或用 `[[ $experiment =~ ^(.*)_(rev|v)[0-9]+(_noprune)?$ ]]` 之类只匹配尾部），或者允许通过环境变量/第 4 参数直接指定 trainer 路径。修完用上面那条 bash 命令复核输出等于 `20260907_c3_rev3_interval_evidence`。

## 逐项检查

### 1. `model.py`

| 项 | 结果 | 证据 |
|---|---|---|
| (a) 校准 c 只加在 av_log；`last_content_logit` 与 a_out/v_out 不含 c | 通过 | 代码：`model.py:241-247`，`last_content_logit = a_log + v_log` 在加 c 之前赋值，`ctx_mode=="logit"` 时 v_out/a_out 不改。数值：9 个结构臂全部 `last_content_logit == fc(a_out)+fc(v_out)`（差 0）；`av_log − last_content_logit − prior` 沿时间轴常数（差 ≤ 7e-7），full/no_qk_enc/no_cell/no_bias/key_bias/shared_bias 该常数非零（c 生效），avce/no_context/ctx_in_rep 为 0（c 不在 logit 上）。 |
| (b) 路由项形状 (B,H,Tq,Tk)，gate 在 query 轴、bias 在 key 轴 | 通过 | `model.py:141-152`：`g (B,H,Tq)[:,:,:,None] * kb (B,H,Tk)[:,:,None,:]`。数值：形状 (3,4,20,20)；与手算 `g[i,h]·β[j,h]` 差 0；扰动 e[5] 后第 5 列（作 key）和第 5 行（作 query）都变、且改变量都由该秒决定。 |
| (c) 初始化时 full == key_bias（g ≡ 1）；shared_bias 各头共用一个 β | 通过 | `model.py:136-139` gate 权重与偏置零初始化 → `2σ(0)=1`。数值：把 full 的权重拷进 key_bias，两模型 av_log 差 0。`shared_bias` 的 β 为 `Linear(128→1)`，routing 输出各头完全相同（差 0），无 gate。 |
| (d) 各臂开关与 docstring 一致 | 通过 | 实例化检查：avce = fc_a 输入 900 维（内容 896 + 4 列裁定）、无 enc/ctx/β、qk_enc False；no_qk_enc = qk_enc False 但 gated β 与 logit c 保留；no_cell = Embedding 去掉、线性层输入 4 列；no_bias = routing 返回 None；key_bias = β(hid→4) 无 gate；shared_bias = β(hid→1) 无 gate；no_context = ctx None；ctx_in_rep = ctx(hid→hid) 加进两流、logit 上无 c。 |
| A：e_t 只进 q/k，value 与残差不含 | 通过 | `model.py:155-159`：`attn(q_in, k_in, y, ...)`，value 输入是 y。数值：把裁定列清零，`lin_v` 的输入完全不变（差 0）。 |
| (e) key padding 屏蔽 | 通过 | `model.py:102-103, 216-218, 231`。数值：随机改写 padding 行的 f_a/f_v 后，full / ctx_in_rep / avce 三臂有效行的 av_log、v_out、a_out 与 mmil 差全为 0。 |
| 骨干只读 scaffold 列 0–3，块级 MIL 用列 4–5 | 通过 | `model.py:219`、`hier_evidence_common.py:193-194`。数值：改写 p_h / block 两列，av_log 不变（差 0）。 |

### 2. `train.py`

| 项 | 结果 | 证据 |
|---|---|---|
| 融合臂到达 `fit_hmm` 的选项 | 通过 | `train.py:116-121`：`index_hmm` → `model="index"`、无选项；`seconds_time` → `normalized_time=False, positive_constraint=True`；`no_constraint` → `normalized_time=True, positive_constraint=False`；其它臂 → 默认 `interval / True / True`。这三臂结构为 `full`（第 108 行）。 |
| `mean_prior_all` 到达 `make_scaffold_fn` | 通过 | `train.py:127-128` 把 `ablation` 原样传入；`hier_evidence_common.py:377-388` 对 `mean_prior_all` 同时换先验/输入列和块标签。真实数据（hatemm 一个训练视频）：`mean_prior_all` 的 p_s 列 = 平均等级、p_h 列 = `bc[block]`；`mean_prior` 的 p_h 列仍 = HMM 的 `p_h[block]`。 |
| w_fine 固定 1 | 通过 | `DEFAULTS["w_fine"]=1.0`（第 63 行）；`search.py` 不采样 w_fine；消融用的 `hparams.json` 只含 5 个采样标量，所以 `cfg.update` 不会覆盖它。 |
| `--config` 覆盖 DEFAULTS | 通过 | `train.py:261-264`。 |
| checkpoint 只按 validation 选 | 通过 | `train.py:201-214`：`crit = (val AP + val ROC)/2`，`best_state` 只在此更新；test 只在训练结束后（第 225 行起）打分。 |
| test 标签不进训练/选点 | 通过 | `test_gt` 只用于筛 `test_ids`（第 100 行）；HMM 只用 `train_ids` 的视频标签拟合（`fit_hmm`，`hier_evidence_common.py:318-319`）；scaffold 对 val/test 只用 VLM 裁定（`posterior` 不接触标签）。 |
| 调用统一评测器 | 通过 | `hc.run_evaluator` → `scripts/reproduction_baselines/eval_baseline_scores.py`（`hier_evidence_common.py:53-54, 298-302`），`metrics.json` 由它写出；`summary.json` 只是转录。 |
| 与修订 2 的训练差异 | 通过 | `diff` 显示 train.py 只改了：融合选项、臂表、模型类名、日志文本。损失、CMAL 的 rep 传法（`fix_rep_swap=False`）、调度器、epoch 数都未变。 |

### 3. `src/interval_evidence_hmm.py`

| 项 | 结果 | 证据 |
|---|---|---|
| (a) k=30, j=4 网格 | 通过 | G=32；段 7 = [0.2333,0.25)，fine 7 / coarse 0，`fine_end=False, coarse_end=True`；段 8 = [0.25,0.2667)，fine 7 / coarse 1，`fine_end=True, coarse_new=True`。每个细窗恰一个 end（30 个）、每个粗块恰一个 end（4 个）；`fine_new[g] == fine_end[g-1]`、粗同。 |
| (b) 增广转移掩码 | 通过 | `_build_masks`（第 158-178 行）：新区间处 h 置 0 再 OR，`h_new = s_new OR h_prev`。 |
| (c) OR 因子每条裁定恰发射一次、缺失不发射 | 通过 | `_emissions` 第 208-218 行只在 `fine_end / coarse_end` 段发射，`b == -1` 跳过。 |
| (d) 前向后向缩放与 xi | 通过 | 穷举验证（见下）。 |
| (e) 正视频约束 | 通过 | 穷举验证；`posterior()`（第 374-380 行）调用 `_posterior_video` 时 `constrain` 取默认 False，只有 `fit` 传 `constrain=self.positive_constraint`。 |
| (f) EM M 步计数 | 通过 | `fit` 第 350-354 行：q 只用正视频 P(h=1) 加权计数，r 用正视频 P(h=0) 计数 + 负视频（负视频只进 r 的分子分母，第 305-311 行按 `!= MISSING` 计观测数）。合成数据（30/4，40 正 40 负）上 EM 对数似然逐轮上升到收敛（约束开/关各 8 轮），参数回到真值附近。 |
| (g) `posterior()` 不用标签 | 通过 | 签名只有 (b_fine, b_coarse, duration, w)。 |
| (h) `rows_from_segments` 与 `verdict_rows` 的时长约定 | 通过 | 两者都按行中点 / duration；`scaffold_rows_interval` 与 `ScaffoldCache` 都传同一个 `n_seconds`。 |

**穷举验证**：在 (k,j) = (5,2)、(6,2)、(7,3)、(5,2 归一化时间) 四种小网格上，对 15 组随机裁定（含缺失 -1）× 约束开/关，把 2^G 条路径逐条枚举求后验，与 `_posterior_video` 的 γ（P(s_g)、P(h_j)）、xi（按 s 合并）、logZ 比较：最大绝对差 1.8e-15。嵌套网格（k=32, j=4，等长段，用区间模型的 exp(QΔt) 作索引模型的 A）与 `src/verdict_hmm.py` 的后验最大差 3.3e-16。

### 4. `src/hier_evidence_common.py`

| 项 | 结果 | 证据 |
|---|---|---|
| `scaffold_rows_interval` 列序 = (ell, p_s, b_fine, b_coarse, p_h, block) | 通过 | 第 344-347 行与 `COL_*` 常量（第 65 行）一致；真实视频检查：`b_coarse 列 == bc[block 列]`、`p_h 列 == posterior p_h[block 列]`、`ell 列 == logit(p_s 列)` 全部成立。 |
| 区间模型的 COL_BLOCK 来自行中点所在粗区间 | 通过 | 第 343 行 `hmm.coarse_of_rows(snip, n_seconds)` → `rows_from_segments(grid["coarse_of"], ...)`。 |
| `video_duration` == `ScaffoldCache` 的 n_seconds | 通过 | 两者都是 `align.load_audio(...).shape[0]`（`align.py:283-291`）。真实数据：HateMM、HCS 各 60 个视频 0 处不等；且与 test GT 长度也一致（各 30 个视频 0 处不等）。 |
| `fit_hmm` 的 index 分支不受影响 | 通过 | `HierEvidenceHMM.params()` 没有 `model` 键 → `make_scaffold_fn` 走原索引路径。 |
| 用训练集重拟合采用配置 | 通过 | hatemm 重拟合的 7 个参数与门 A1 的 `hmm_only/hatemm/interval_norm_constraint_params.json` 完全一致（298 正 / 446 负）。 |

### 5. `search.py` / `launch/run_search.sh`

| 项 | 结果 | 证据 |
|---|---|---|
| 恰好采样 README 第 2 节的 5 个标量 | 通过 | `sample()` 第 40-46 行：lr log[1e-4,1e-3]、max_seqlen {150,200,300}、λ_cma [0.5,2]、prior_scale log[0.5,8]、λ_block log[0.05,2]。 |
| `study_summary.json` 的 best = COMPLETE trial 中目标最高 | 通过 | 第 149-154 行。目标 = test (AP+ROC)/2（第 114 行，README 已声明）；within 不剪枝（`floor=None`）。 |
| 启动脚本 | 通过 | `run_search.sh` 直接调 `search.py`，输出到 `runs/20260907_c3_rev3_interval_evidence/<corpus>/seed<seed>/`，不经过 B1 那段代码。 |

### 6. `interval_hmm_eval.py`（门 A1，已跑）

只用训练集视频标签拟合、只用裁定打分、走统一评测器；`index` 变体按细窗中点、`interval` 变体按段中点映射到秒，两者都不用标签。无问题。

## 非阻断记录（不改变观察，不要求修）

1. 行落段（`rows_from_segments`，按 `frac = mid/D` 与 `start` 比较）与 `verdict_rows`（`floor(mid·k/D)`）在中点恰落在区间边界的行上可能因浮点差一格：5–400 秒所有时长、1 秒与 0.667 秒两种网格共 399,148 行里 134 行（0.03%）不一致。这些行的 ell/p_s 取相邻段、b_fine 取相邻窗，两个都是合法邻居，对指标无可见影响。
2. `search.py` 第 37 行注释写"README section 4"，实际搜索空间在 README 第 2 节；只是注释。
3. `search.py` 预算规则仍是"首 trial ≤ 1 h 则 20，否则 5"，README 写死 20；与修订 2 相同，属既有流程。

## 处理记录（2026-09-07，主 agent）

- B1 已修：`scripts/run_locked_ablations.sh` 改为先找 `experiments/<完整 id>/train.py`，找不到时只从末尾剥 `_(v|rev)<n>(_<词>)?` 后缀再找（`experiments/`、`archive/experiments/` 各试一次）。复核：`20260907_c3_rev3_interval_evidence` 解析到本目录 `train.py`；`20260904_evidence_guided_attention_rev2_noprune`、`..._rev2`、`20260903_hier_evidence_mil_v2` 仍解析到原目录。
- 非阻断 2：`search.py` 注释改为"README section 2"。
- 非阻断 1、3 不改，保持既有流程。
