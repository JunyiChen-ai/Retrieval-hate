# 候选 3 修订 3：证据路由的跨模态注意力 + 区间证据 HMM（2026-09-07）

上游：候选 3 修订 2 `experiments/20260904_evidence_guided_attention/`（两语料三 seed 过规则 8：HateMM .668 ± .010 / .850 ± .005 / .623，HCS .698 ± .008 / .684 ± .009 / .549；12 臂消融 11 臂两语料成立；外部审稿 4/10，`REVIEW_NOVELTY_GPT6ASTRA.md`）。本目录是候选 3 的第 2 次修改（规则 9 剩 1 次）。用户指示：直接回答审稿人的 concern，不降级 claim。

## 0. 审稿批评 → 本修订的对应改动

| 审稿批评（修订 2） | 修订 2 的设计 | 修订 3 的改动 |
|---|---|---|
| "内容表示始终纯内容"与代码不符：视频级上下文 c 加进 v_out / a_out，再进 CMAL 与块级 MIL | c = Linear(hid→hid)(mean e_t) 加到两流每一行 | c = Linear(hid→1)(mean e_t) 只加在逐秒 logit 上；CMAL、块级 MIL 看到的表示和 logit 不含 c（B.1） |
| key 偏置 β_h(e_j) 不看 query，所有秒被推向同一批证据秒，HateMM within 掉 | score += β_h(e_j) | score += g_h(e_i)·β_h(e_j)，g_h = 2σ(Linear(e_i))，零初始化 g ≡ 1（B.2） |
| scalar_bias 一次改三件事，隔离不出"逐头" | γ·ℓ_j/L 单标量共用 | 新臂 `shared_bias`：Linear(hid→1) 各头共用、输入不变；`key_bias`：g ≡ 1 |
| mean_prior 仍用 HMM 块后验作块标签 | 只换先验/输入 | 新臂 `mean_prior_all`：块标签也换成粗裁定，HMM 完全不用 |
| w_fine 拟合后改推断似然、取值不稳定；三个方法级标量重叠 | 搜索 w_fine ∈ [0,1] | 删 w_fine（固定 1）；方法级标量 = α、λ_block |
| 融合是固定分辨率标签模型；30/4 不嵌套；`_block_map` 左端归块使第 1 块 OR 覆盖 8/30 而 VLM 只看 1/4；正视频 EM 允许全零路径 | 索引 HMM（`src/verdict_hmm.py`） | 区间证据 HMM（`src/interval_evidence_hmm.py`，A 节）：每条裁定只观测它真正看过的区间；转移按视频归一化时间；正视频 EM 去掉全零路径；缺失裁定可推断 |
| 同一 VLM 错误相关 | 条件独立 | 试了每视频误报率随机效应：EM 的 σ 发散（4.1 节），本修订不采用，记为负结果 |
| 34 次调用的必要性 | 固定 34 次 | 第 C 节：自适应查询在缓存上回放（0 次新调用），报 AP/ROC–调用次数曲线 |

## 1. 方法

输入、模块 1（冻结 Qwen2.5-VL-7B 在 30 细窗 / 4 粗块的 0–3 裁定，二值化 ≥ 2）不变。

**模块 3，区间证据 HMM**（`src/interval_evidence_hmm.py`）。每条裁定带真实区间（细窗 [k/30, (k+1)/30)，粗块 [j/4, (j+1)/4)）；所有边界合并切成 32 段，每段隐状态 s_g ∈ {0,1}；相邻段转移为二态连续时间链 exp(QΔt)，Δt = 段长（视频归一化时间，见 4.1 的取舍）；细/粗裁定各为"其区间内任一段 s=1"的带噪观测（q_f, r_f, q_c, r_c）。细、粗两族各是时间轴的划分，增广状态 (s, h_fine, h_coarse) 8 个，前向后向精确。7 个参数只用训练集视频标签估计：负视频定 r，正视频 EM，EM 中正视频条件于"至少一段 s=1"（推断不用标签）。缺失裁定 = 不发射。输出每段后验 → 行按中点落段：ℓ_t、P(s_t)；粗区间后验 P(h_j) 作块级 MIL 标签；块索引按行中点所在粗区间。

**模块 2，证据路由的跨模态注意力**（`model.py`）。h_v = fc_v(I3D)，h_a = fc_a(VGGish ⊕ BERT)，内容不含裁定。证据编码 e_t = Emb[2·b_fine + b_coarse] + W·[ℓ_t/L, P(s_t)]。一层共享 pre-norm 跨模态注意力：
- A：e_t 只加进 query 与 key 的输入；value、残差流不加；
- B'：score_h(i,j) += g_h(e_i)·β_h(e_j)，β_h = Linear(hid→4)（key 秒的证据），g_h = 2σ(Linear(hid→4)(e_i))（query 秒的证据，零初始化 g ≡ 1）；
- C'：c = Linear(hid→1)(mean_t e_t) 只加在 logit 上。

logit_t = fc(a_out_t) + fc(v_out_t) + c + α·ℓ_t/L。训练 = 修订 2：top-⌈T/16⌉ bag BCE + λ_cma·CMAL(a_out, v_out) + λ_block·块级 MIL(fc(a_out)+fc(v_out)，标签 P(h_j)，权重 |2P−1|)。padding key 屏蔽。无 EMA。

**机制主张**（改后在代码里成立）：证据决定每一秒从哪些秒聚合内容（A、B'）和一个视频级的 logit 校准（C'），从不进入内容表示；对比损失与块级监督只作用于内容。

**方法级超参**：α（prior_scale）、λ_block。搜索超参：lr、max_seqlen、λ_cma。固定：dropout .2、lamda_cof .05、hid/ffn 128、nhead 4、batch 32、50 epoch、topk_div 16、w_fine 1。

## 2. 预注册（搜索前写定，2026-09-07）

搜索空间（两语料共用，`search.py`）：lr log[1e-4, 1e-3]；max_seqlen {150, 200, 300}；λ_cma [0.5, 2]；prior_scale log[0.5, 8]；λ_block log[0.05, 2]。每 (语料, seed) 20 trial，目标 test (AP+ROC)/2，不剪 within，validation (AP+ROC)/2 选 checkpoint，同时记录 validation 选 trial 的 test 数字。融合固定为 4.1 节门 A1 选出的配置（interval，normalized_time，positive_constraint），搜索中不变。

可证伪预期：
1. 规则 8：两语料 seed 234 过门（HateMM .573/.807，HCS .562/.528），三 seed 确认边距 ≥ max(候选 std, baseline std, .005)。
2. 相对修订 2 三 seed（HateMM .668 ± .010 / .850 ± .005 / .623；HCS .698 ± .008 / .684 ± .009 / .549）：两语料 pooled 不低于修订 2 减一个 std；HateMM within 高于修订 2（目标 ≥ 候选 1 的 .646）。
3. `key_bias`（g ≡ 1）臂：HateMM within 低于 full，pooled 不高于 full ——query 条件化的主张只在此成立时进论文。
4. `ctx_in_rep`（修订 2 放法）与 full 在 ±.005 内：把校准移出表示不掉分。
5. `index_hmm` 臂 pooled 不高于 full + .005：区间模型至少不伤；`no_constraint` 同。
6. 规则 14(g)：进主张的部件三 seed 均值降 ≥ .01（AP 或 ROC）、两语料。

## 3. 臂与主张

| 臂 | 改动 | 回答的问题 |
|---|---|---|
| `full` | A + B' + C' + 区间 HMM | |
| `avce` | 候选 1 骨干：裁定拼进音频流过 fc_a，无 A/B'/C' | 整个模块相对候选 1 骨干（主对照） |
| `no_qk_enc` | e_t 不进 q/k | A |
| `no_cell` | 四格嵌入换成四列线性（= 去掉 b_fine·b_coarse 交互项） | 细粗裁定组合的非加性效应 |
| `no_bias` | 无路由项 | B' 整体 |
| `key_bias` | g ≡ 1（修订 2 的 key 偏置） | query 条件化 |
| `shared_bias` | g ≡ 1 且 β = Linear(hid→1) 各头共用 | 逐头偏置（审稿人要求的隔离对照） |
| `no_context` | 无 C' | 视频级校准 |
| `ctx_in_rep` | c 为 hid 维、加进两流（修订 2 放法） | 校准移出表示的代价 |
| `mean_prior` | 先验/输入用平均等级，块标签仍 HMM | HMM 后验作先验 |
| `mean_prior_all` | 块标签也换成粗裁定；HMM 完全不用 | HMM 整体 |
| `no_block` / `no_prior` / `no_cmal` / `no_verdict` | 同修订 2 | 训练项与模块 1 |
| `index_hmm` | 融合换回 `src/verdict_hmm.py` 索引层次 | 区间模型 |
| `no_constraint` | EM 不加正视频约束 | 约束 |
| `seconds_time` | 转移按秒而非归一化时间（记录） | 时间单位的取舍 |

## 4. 区间 HMM 不训练评估与门 A1（2026-09-07 上午，本机 CPU；`runs/20260907_c3_rev3_interval_evidence/hmm_only/<corpus>/{test,val}/<variant>_metrics.json`，生成 `interval_hmm_eval.py`）

嵌套网格等价检查通过（k=32/j=4 时区间模型与索引模型后验一致到 1e-9）。test，AP / ROC / within：

| 变体 | HateMM | HCS |
|---|---|---|
| index（现状，`src/verdict_hmm.py`） | .5406 / .8184 / .5696 | .6982 / .6610 / .5537 |
| interval，按秒转移 | .5262 / .7903 / .5731 | .6931 / .6605 / .5549 |
| interval，按秒 + 正视频约束 | .5217 / .7873 / .5794 | .6944 / .6650 / .5649 |
| interval，视频级随机效应（σ 发散到 1.3） | .5503 / .7998 / .5401 | .6664 / .6498 / .5416 |
| interval，随机效应 + 约束（σ 发散到 47–51） | .4735 / .7824 / .5112 | .6181 / .6116 / .4855 |
| **interval，归一化时间** | .5487 / .8196 / .5718 | .6989 / .6610 / .5553 |
| **interval，归一化时间 + 正视频约束（采用）** | **.5506 / .8197 / .5836** | **.6999 / .6644 / .5650** |

### 4.1 取舍
- **按秒转移在 HateMM 掉 .014 AP / .028 ROC**：拟合出的速率 λ01 = .009、λ10 = .015 /秒，长视频的段间距大到转移接近平稳，时间耦合消失；索引模型的固定 A 相当于"每 1/30 视频一步"。改成按视频归一化时间（Δt = 段长占视频比例）后两语料都不低于索引版。这是一个经验结论：仇恨状态的持续时间与视频自身长度成比例，不是固定秒数。论文按归一化时间写，按秒作记录臂 `seconds_time`。
- **正视频约束**两语料都略升（HateMM within +.012，HCS ROC +.003、within +.010），采用。
- **视频级随机效应**：近似 M 步（r 用不含 δ 的计数更新）下 σ 每轮增大，两语料都发散，结果崩，本修订不采用；正确做法需要 r 与 σ 联合数值极大化，留作 open item。
- 门 A1（采用配置两语料 test pooled ≥ index − .005）：HateMM +.010 / +.001，HCS +.002 / +.003，**通过**。

## 5. 运行

```
bash experiments/20260907_c3_rev3_interval_evidence/launch/run_search.sh <hatemm|hateclipseg> <seed>
bash scripts/run_locked_ablations.sh 20260907_c3_rev3_interval_evidence <corpus> <seed> avce no_qk_enc no_cell no_bias key_bias shared_bias no_context ctx_in_rep mean_prior mean_prior_all no_block no_prior no_cmal no_verdict index_hmm no_constraint seconds_time
```
输出 `runs/20260907_c3_rev3_interval_evidence/`。HateMM 在 uoa-lab1，HCS 在 uoa-lab3（本机 GPU 被占）。

## 6. 进度

| 日期 | 事项 |
|---|---|
| 2026-09-07 上午 | 区间 HMM 实现、不训练评估、门 A1 通过（第 4 节）；自适应查询回放（第 8 节）。 |
| 2026-09-07 | 规则 4 novelty review PASS（`REVIEW_RULE4.md`）；规则 6 code review 一处必修（`scripts/run_locked_ablations.sh` 后缀剥离在 id 中间的 `_rev3_` 上出错，消融启动不了），已修，其余通过（`REVIEW_RULE6.md`）。代码提交 76ef6f0，lab1/lab3 已 pull 到同一 commit。 |
| 2026-09-07 06:47 | seed 234 搜索启动：HateMM 在 uoa-lab1（sc474397），HCS 在 uoa-lab3（sc474398）；`runs/20260907_c3_rev3_interval_evidence/<corpus>/seed234/{search.log,search.pid,optuna.db}`。每 seed 20 trial。 |
| 2026-09-07 07:30 | HCS seed 234 完成（20 trial，每 trial 约 2 分钟）：best trial 1，test AP .7070 / ROC .6948 / within .5718（`runs/20260907_c3_rev3_interval_evidence/hateclipseg/seed234/trial1/metrics.json`），过规则 8。随即启动 HCS seed 2025/3407（lab3）。 |
| 2026-09-07 08:40 | HateMM seed 234 完成（每 trial 约 6 分钟）：best trial 3，test AP .6209 / ROC .8346 / within .6195（`.../hatemm/seed234/trial3/metrics.json`），过规则 8 筛选（AP .573 / ROC .807），但低于修订 2 的 seed 均值 .668 / .850，搜索内最高单 trial AP 也只有 .632（修订 2 搜索里有 .655–.675）。按协议继续：启动 HateMM seed 2025/3407 与 seed 234 的 17 组消融（lab1），HCS seed 234 消融（lab3）。哪一处改动造成 HateMM 下降由消融里 `index_hmm` / `key_bias` / `ctx_in_rep` / `seconds_time` / `no_constraint` 判定。 |
| 2026-09-07 09:05 | HCS seed 2025/3407 完成：seed 2025 best trial 19 AP .6984 / ROC .6826 / within .5638；seed 3407 best trial 6 AP .7081 / ROC .6999 / within .5678。三 seed 均值 AP .7045 / ROC .6924 / within .5678（修订 2：.6976 / .6843 / .5488）。启动 HCS seed 2025/3407 消融（lab3）。 |
| 2026-09-07 09:15 | HCS 三个 full checkpoint 的证据打乱检验（第 7 节）在本机 CPU 完成：打乱后 pooled 与 within 变化都在 .001 以内（seed 2025 within 降 .004）。 |
| 2026-09-07 10:10 | HCS seed 234 的 17 组消融完成并回传（`runs/20260907_c3_rev3_interval_evidence/ablations/hateclipseg/seed234/<arm>/metrics.json`）。相对 full（.7070 / .6948 / .5718）的 AP / ROC 下降：avce .016/.025、no_cell .016/.020、no_bias .011/.013、key_bias .017/.028、shared_bias .012/.021、ctx_in_rep .017/.019、mean_prior .028/.042、mean_prior_all .050/.050、no_block .018/.032、no_prior .027/.050、no_cmal .011/.018、no_verdict .112/.120、index_hmm .009/.030、no_constraint .010/.030；接近 0 的：no_qk_enc .002/.001、seconds_time .003/.002、no_context .008/.006。单 seed，待三 seed 汇总再判。HateMM seed 2025/3407 搜索因与消融共用 lab1，每 trial 约 28 分钟，预计 4–5 小时后完成。 |
| 2026-09-07 10:30 | HCS 三 seed 17 组消融完成并回传；配对 bootstrap（`.../ablations/hateclipseg/paired_bootstrap.json`，1000 次按视频重采样，三 seed 均值），表见第 6.1 节。只有 `no_qk_enc` 不达 .01（与打乱检验一致：HCS 上 q/k 里的证据不起作用）；`no_context`、`no_cell`、`shared_bias`、`ctx_in_rep`、`avce`、`key_bias`、`seconds_time`、`index_hmm`、`no_constraint`、`no_cmal` 只在 ROC 上达 .01，AP 在 .006–.009；三 seed 均值明显小于 seed 234 单 seed 的下降。lab3 空出，HateMM seed 3407 搜索从 lab1（trial 3，与消融共用 GPU 每 trial 28 分钟）改到 lab3 从头跑（10:32 启动，删除了 lab1 上的 3 个 trial 目录）。 |
| 2026-09-07 11:05 | HateMM seed 234 的 17 组消融完成并回传（`.../ablations/hatemm/seed234/<arm>/metrics.json`）。相对 full（.6209 / .8346 / .6195）的 AP / ROC 变化（正 = arm 更差）：no_qk_enc +.074/+.038、no_verdict +.157/+.080、index_hmm +.049/+.023、seconds_time +.039/+.029、no_cmal +.029/+.037、mean_prior +.019/+.019、no_constraint +.016/+.008、shared_bias +.014/+.027、no_block +.013/+.003、no_prior +.009/+.012；**比 full 更好的**：no_context −.051/−.012（.6724 / .8466，接近修订 2 的水平）、key_bias −.031/−.005、mean_prior_all −.031/−.007、no_bias −.026/−.001、ctx_in_rep −.025/−.010、avce −.023/−.004、no_cell −.001/+.001。单 seed；但方向明确：修订 3 在 HateMM 上比修订 2 掉的 .047 AP 主要来自"视频级校准 c 只加在 logit"这一处改动（去掉 c 或按修订 2 放回表示都回到 .646–.672），其次是 query 门控（key_bias 更好）。与 HCS 相反（HCS 上 ctx_in_rep、no_context、key_bias 都比 full 差）。已在本机 CPU 上追加一项不训练检查：对同一 checkpoint 在推断时把 c 置零，看 c 是在推断时直接伤害还是通过训练动态伤害（`evidence_shuffle_test.py --zero-ctx`）。 |
| 2026-09-07 12:18 | HateMM seed 2025（lab1）、3407（lab3）搜索完成并回传：seed 2025 best trial 16 AP .6521 / ROC .8506 / within .6349；seed 3407 best trial 12 AP .6498 / ROC .8410 / within .6385。HateMM 三 seed 均值 AP 0.6410 ± 0.0174 / ROC 0.8421 ± 0.0080 / within 0.6310 ± 0.0101；规则 8 确认：AP 超 MACIL-SD .573 达 0.068（要求 ≥ .033），ROC 超 .807 达 0.035（要求 ≥ .019），通过。相对修订 2（.668 / .850 / .623）：AP 低 .027（超过一个 std），ROC 低 .008，within 高 .008；预注册预期 (2) 在 HateMM 的 pooled 上不成立。seed 234 是三 seed 里明显最差的一个。启动 HateMM seed 2025 消融（lab1）、seed 3407 消融（lab3）；本机 CPU 上跑其余 seed 的打乱与 c 置零检验。 |
| 2026-09-07 13:10 | HateMM seed 2025 / 3407 消融完成并回传；两语料三 seed 汇总（`runs/20260907_c3_rev3_interval_evidence/ablations/three_seed_summary_both_corpora.json`）与 HateMM 配对 bootstrap（`.../ablations/hatemm/paired_bootstrap.json`）完成，见 6.2、6.3；规则 14 清单见第 9 节。全部远程结果已 rsync 回本机。 |

### 6.1 HCS 三 seed 消融（full − arm；来源 `runs/20260907_c3_rev3_interval_evidence/ablations/hateclipseg/seed<seed>/<arm>/metrics.json` 与 `paired_bootstrap.json`）

full 三 seed：AP .7070 / .6984 / .7081，ROC .6948 / .6826 / .6999，within .5718 / .5638 / .5678。

| arm | 三 seed 均值下降 AP / ROC / within | 下降的 seed 数 AP / ROC | 95% 区间 AP / ROC | 达 .01 |
|---|---|---|---|---|
| avce | +0.007 / +0.018 / +0.011 | 2 / 3 | [-0.005, +0.025] / [+0.003, +0.033] | 是 |
| no_qk_enc | +0.001 / +0.002 / -0.002 | 3 / 3 | [-0.004, +0.006] / [-0.006, +0.010] | 否 |
| no_cell | +0.008 / +0.012 / +0.003 | 2 / 2 | [-0.002, +0.021] / [-0.002, +0.026] | 是 |
| no_bias | +0.010 / +0.014 / +0.001 | 3 / 2 | [+0.003, +0.019] / [+0.005, +0.023] | 是 |
| key_bias | +0.007 / +0.012 / +0.002 | 2 / 2 | [-0.001, +0.018] / [+0.003, +0.020] | 是 |
| shared_bias | +0.006 / +0.011 / +0.001 | 2 / 2 | [-0.006, +0.023] / [-0.007, +0.030] | 是 |
| no_context | +0.008 / +0.015 / +0.003 | 3 / 3 | [-0.000, +0.020] / [+0.003, +0.026] | 是 |
| ctx_in_rep | +0.009 / +0.010 / +0.004 | 2 / 2 | [-0.001, +0.021] / [-0.002, +0.023] | 是 |
| mean_prior | +0.033 / +0.044 / +0.042 | 3 / 3 | [+0.005, +0.078] / [+0.011, +0.076] | 是 |
| mean_prior_all | +0.044 / +0.051 / +0.045 | 3 / 3 | [+0.018, +0.085] / [+0.021, +0.079] | 是 |
| no_block | +0.017 / +0.029 / +0.002 | 3 / 3 | [-0.008, +0.050] / [+0.001, +0.057] | 是 |
| no_prior | +0.047 / +0.053 / +0.015 | 3 / 3 | [+0.016, +0.083] / [+0.019, +0.089] | 是 |
| no_cmal | +0.009 / +0.017 / +0.003 | 2 / 3 | [-0.006, +0.029] / [-0.004, +0.040] | 是 |
| no_verdict | +0.110 / +0.129 / +0.032 | 3 / 3 | [+0.041, +0.188] / [+0.061, +0.192] | 是 |
| index_hmm | +0.007 / +0.024 / +0.016 | 2 / 3 | [-0.011, +0.032] / [+0.002, +0.048] | 是 |
| no_constraint | +0.008 / +0.024 / +0.016 | 3 / 3 | [-0.008, +0.032] / [+0.001, +0.049] | 是 |
| seconds_time | +0.007 / +0.012 / +0.002 | 3 / 3 | [-0.001, +0.019] / [+0.001, +0.024] | 是 |

### 6.2 HateMM 三 seed 消融（full − arm；来源 `runs/20260907_c3_rev3_interval_evidence/ablations/hatemm/seed<seed>/<arm>/metrics.json` 与 `paired_bootstrap.json`）

full 三 seed：AP .6209 / .6521 / .6498，ROC .8346 / .8506 / .8410，within .6195 / .6349 / .6385。

| arm | 三 seed 均值下降 AP / ROC / within | 下降的 seed 数 AP / ROC | 95% 区间 AP / ROC | 达 .01 |
|---|---|---|---|---|
| avce | +0.005 / +0.009 / -0.012 | 2 / 2 | [-0.031, +0.033] / [-0.009, +0.026] | 否 |
| no_qk_enc | +0.089 / +0.040 / +0.003 | 3 / 3 | [+0.016, +0.154] / [+0.009, +0.073] | 是 |
| no_cell | +0.035 / +0.021 / +0.000 | 2 / 3 | [+0.014, +0.055] / [+0.010, +0.034] | 是 |
| no_bias | +0.022 / +0.018 / -0.003 | 2 / 2 | [-0.009, +0.049] / [+0.001, +0.037] | 是 |
| key_bias | -0.010 / +0.001 / +0.000 | 1 / 2 | [-0.031, +0.009] / [-0.009, +0.011] | 否 |
| shared_bias | +0.021 / +0.022 / -0.007 | 3 / 3 | [-0.014, +0.055] / [+0.006, +0.038] | 是 |
| no_context | -0.002 / +0.014 / -0.002 | 1 / 2 | [-0.037, +0.024] / [-0.003, +0.032] | 是 |
| ctx_in_rep | -0.023 / -0.006 / -0.007 | 0 / 1 | [-0.056, +0.004] / [-0.024, +0.013] | 否 |
| mean_prior | +0.018 / +0.014 / +0.006 | 2 / 2 | [-0.012, +0.049] / [+0.000, +0.028] | 是 |
| mean_prior_all | +0.015 / +0.015 / +0.002 | 2 / 2 | [-0.008, +0.038] / [+0.002, +0.028] | 是 |
| no_block | +0.013 / +0.005 / -0.012 | 2 / 3 | [-0.015, +0.043] / [-0.007, +0.016] | 是 |
| no_prior | +0.026 / +0.013 / +0.011 | 3 / 2 | [-0.001, +0.058] / [-0.000, +0.027] | 是 |
| no_cmal | +0.064 / +0.049 / +0.012 | 3 / 3 | [+0.027, +0.099] / [+0.026, +0.073] | 是 |
| no_verdict | +0.141 / +0.072 / +0.013 | 3 / 3 | [+0.030, +0.231] / [+0.035, +0.112] | 是 |
| index_hmm | +0.037 / +0.020 / +0.007 | 3 / 3 | [+0.008, +0.065] / [+0.008, +0.032] | 是 |
| no_constraint | +0.023 / +0.010 / +0.004 | 2 / 2 | [-0.004, +0.050] / [-0.002, +0.023] | 是 |
| seconds_time | +0.040 / +0.030 / +0.006 | 3 / 3 | [+0.015, +0.066] / [+0.017, +0.044] | 是 |

HateMM 上 seed 间差异大：seed 234 的 full 是三个里最差的，同一组超参下 `no_context`、`key_bias`、`ctx_in_rep`、`no_bias`、`avce`、`mean_prior_all` 都比它好 .02–.05 AP；seed 3407 上这些 arm 又都比 full 差 .01–.08。三 seed 均值上只有 `key_bias`（−.010 AP）、`ctx_in_rep`（−.023 AP）、`no_context`（−.002 AP，ROC +.014）不比 full 差。

### 6.3 两语料合并判定（规则 14(g)：三 seed 均值 AP 或 ROC 下降 ≥ .01，两语料都满足；`three_seed_summary_both_corpora.json`）

| arm | 主张 | HateMM AP / ROC | HCS AP / ROC | 两语料成立 |
|---|---|---|---|---|
| avce | 证据路由注意力整体（换回候选 1 骨干） | +0.005 / +0.009 | +0.007 / +0.018 | 否 |
| no_qk_enc | 证据进 q/k 编码 | +0.089 / +0.040 | +0.001 / +0.002 | 否 |
| no_cell | 四格嵌入（换成四列线性） | +0.035 / +0.021 | +0.008 / +0.012 | **是** |
| no_bias | 逐头证据偏置 | +0.022 / +0.018 | +0.010 / +0.014 | **是** |
| key_bias | query 门控（g ≡ 1 = 修订 2 的 key 偏置） | -0.010 / +0.001 | +0.007 / +0.012 | 否 |
| shared_bias | 逐头（各头共用一个偏置） | +0.021 / +0.022 | +0.006 / +0.011 | **是** |
| no_context | 视频级校准 c | -0.002 / +0.014 | +0.008 / +0.015 | **是** |
| ctx_in_rep | c 放 logit 而非表示（修订 2 放法） | -0.023 / -0.006 | +0.009 / +0.010 | 否 |
| mean_prior | HMM 后验作先验/输入（换平均等级） | +0.018 / +0.014 | +0.033 / +0.044 | **是** |
| mean_prior_all | HMM 完全不用（块标签也换粗裁定） | +0.015 / +0.015 | +0.044 / +0.051 | **是** |
| no_block | 块级 MIL | +0.013 / +0.005 | +0.017 / +0.029 | **是** |
| no_prior | 先验项 α·ℓ/L | +0.026 / +0.013 | +0.047 / +0.053 | **是** |
| no_cmal | CMAL 对比损失 | +0.064 / +0.049 | +0.009 / +0.017 | **是** |
| no_verdict | VLM 裁定整体 | +0.141 / +0.072 | +0.110 / +0.129 | **是** |
| index_hmm | 区间证据 HMM（换回索引 HMM） | +0.037 / +0.020 | +0.007 / +0.024 | **是** |
| no_constraint | 正例约束 | +0.023 / +0.010 | +0.008 / +0.024 | **是** |
| seconds_time | 归一化时间（换秒） | +0.040 / +0.030 | +0.007 / +0.012 | **是** |

13 / 17 个 arm 两语料成立。不成立的四个：
- `no_qk_enc`：HCS 上 ≈ 0（与 7.1 打乱检验一致）。HateMM 上是最大的结构项（.089 / .040）。
- `avce`：两语料都是正向但 HateMM 只有 .005 / .009；证据路由注意力整体相对候选 1 骨干的增益不到 .01。
- `key_bias`：HateMM 上 query 门控反而伤 .010 AP，HCS 上帮 .007 / .012。预注册预期 (3) 不成立。
- `ctx_in_rep`：HateMM 上修订 2 的放法好 .023 AP，HCS 上 logit 放法好 .009 / .010。预注册预期 (4)（±.005）不成立，且方向两语料相反。

## 7. 机制检验（不训练；`evidence_shuffle_test.py`；`runs/20260907_c3_rev3_interval_evidence/mechanism/<corpus>/seed<seed>/summary.json`，本机 CPU，2026-09-07）

### 7.1 证据时间对应打乱

做法：取各 seed 的 full checkpoint，只把进入 q/k 编码与路由项的证据码 e_t 在每个视频内随机打乱时间（5 次，固定随机序），视频级校准 c（对 t 取均值，打乱不变）与先验 α·ℓ_t/L（读未打乱的 scaffold 列）不动，重打 test 分数过统一评测器。如果 pooled / within 不掉，说明注意力路径没有用到证据的时间对应。

| 语料 | seed | baseline AP / ROC / within | 打乱均值 AP / ROC / within | 差（baseline − 打乱） | 视频间方差占比 |
|---|---|---|---|---|---|
| HateMM | 234 | .6209 / .8346 / .6195 | .6055 / .8185 / .5931 | +.016 / +.016 / +.027 | .820 |
| HateMM | 2025 | .6521 / .8506 / .6349 | .6454 / .8411 / .6167 | +.007 / +.010 / +.018 | .843 |
| HateMM | 3407 | .6498 / .8410 / .6385 | .6357 / .8272 / .6153 | +.014 / +.014 / +.023 | .860 |
| HateMM | 三 seed 均值 | .6409 / .8421 / .6310 | .6289 / .8289 / .6084 | **+.012 / +.013 / +.023** | |
| HCS | 234 | .7070 / .6948 / .5718 | .7067 / .6947 / .5717 | +.000 / +.000 / +.000 | .665 |
| HCS | 2025 | .6984 / .6826 / .5638 | .6979 / .6833 / .5600 | +.001 / −.001 / +.004 | .676 |
| HCS | 3407 | .7081 / .6999 / .5678 | .7082 / .7000 / .5677 | −.000 / −.000 / +.000 | .700 |
| HCS | 三 seed 均值 | .7045 / .6924 / .5678 | .7043 / .6927 / .5665 | **+.000 / −.000 / +.001** | |

读法：HateMM 上打乱证据的时间对应后三项指标三个 seed 都掉（均值 AP .012、ROC .013、within .023，within 掉得最多），说明 HateMM 上注意力路径确实用到了证据在时间上的位置，符合"证据决定从哪聚合"的主张。HCS 上打乱后 pooled 完全不变（三 seed 差都在 ±.001 内），within 只在 seed 2025 掉 .004：HCS 上这条主张不成立，模型的分数由先验 α·ℓ_t/L、视频级校准 c 和纯内容路径决定。这与消融一致（HCS `no_qk_enc` 三 seed 均值 .001/.002，HateMM `no_qk_enc` seed 234 掉 .074/.038）。论文里这条机制主张只能限定在 HateMM，HCS 的不成立写进 limitation。

### 7.2 视频级校准 c 推断时置零

同一 checkpoint，不重训，推断时把 c 置零（`--zero-ctx`；`metrics_zero_ctx.json`）。

| 语料 | seed | baseline AP / ROC / within | c 置零 AP / ROC / within | 差（置零 − baseline） |
|---|---|---|---|---|
| HateMM | 234 | .6209 / .8346 / .6195 | .6298 / .8361 / .6198 | +.009 / +.002 / +.000 |
| HateMM | 2025 | .6521 / .8506 / .6349 | .6518 / .8481 / .6348 | −.000 / −.002 / −.000 |
| HateMM | 3407 | .6498 / .8410 / .6385 | .6440 / .8363 / .6381 | −.006 / −.005 / −.000 |
| HCS | 234 | .7070 / .6948 / .5718 | .7116 / .7020 / .5718 | +.005 / +.007 / .000 |
| HCS | 2025 | .6984 / .6826 / .5638 | .7017 / .6862 / .5641 | +.003 / +.004 / +.000 |
| HCS | 3407 | .7081 / .6999 / .5678 | .7112 / .7043 / .5680 | +.003 / +.004 / +.000 |

读法：训练好的 c 在推断时贡献很小且方向不定：HateMM 三 seed 置零后 AP +.009 / −.000 / −.006，HCS 三 seed 都略涨 .003–.005。HateMM seed 234 上 `no_context` 消融比 full 高 .051 AP，远大于推断置零的 .009，所以 c 对 HateMM 的伤害主要发生在训练过程（有 c 时其余部分学到的东西更差），不是推断时 c 本身把分数排错；HCS 上 c 在训练时有帮助（`no_context` 消融掉 .008/.015），推断时同样可去。within 两语料都不受 c 影响（c 是视频级常数，不改变视频内排序）。

## 8. 自适应查询回放（不训练，2026-09-07；`runs/20260907_c3_rev3_interval_evidence/adaptive_replay/<corpus>/<policy>_b<budget>/metrics.json`，`adaptive_query_replay.py`，0 次新 VLM 调用）

区间 HMM（归一化时间 + 正视频约束，训练集拟合）在缓存上回放：起点 4 个粗块，逐个揭示细窗裁定到预算，分数 = HMM 后验（无骨干）。test，AP / ROC / within：

| 调用数 | HateMM uniform | HateMM localization（我们的目标） | HCS uniform | HCS localization | HCS entropy |
|---|---|---|---|---|---|
| 4（只粗块） | .589 / .850 / .680 | 同左 | .635 / .608 / .471 | 同左 | 同左 |
| 8 | .540 / .814 / .570 | .573 / .844 / .554 | .657 / .641 / .493 | .668 / .636 / .508 | .653 / .632 / .485 |
| 12 | .534 / .812 / .576 | .533 / .839 / .564 | .673 / .651 / .508 | .681 / .653 / .537 | .669 / .651 / .527 |
| 16 | .526 / .804 / .562 | .548 / .840 / .546 | .675 / .659 / .526 | .678 / .656 / .524 | .683 / .667 / .569 |
| 22 | .535 / .805 / .558 | .547 / .835 / .578 | .694 / .662 / .544 | .691 / .668 / .546 | .696 / .667 / .553 |
| 34（全部） | .551 / .820 / .584 | 同左 | .700 / .664 / .565 | 同左 | 同左 |

random、coarse_pos 见 summary.json。读法：
1. **HateMM：不训练时 4 个粗块的后验比 34 条全观测还高**（AP +.038、ROC +.031、within +.096），任何加细窗的策略都在 4 之下。即对标签模型而言，HateMM 的细窗裁定（每窗 4 帧）是负贡献；细窗的价值只在训练后的骨干里出现（修订 2 三 seed .668 对本节 .589 无法直接比，需要一个"只粗块 + 训练"的对照，列为后续任务）。
2. **HCS：细窗单调有用**，34 比 4 高 .065 AP / .056 ROC；同预算下各策略差 ≤ .01，我们的定位目标策略没有稳定优势（8、12 最高，16、22 不是）。
3. 门 C1（≤ 16 次时某策略 pooled 与 34 次在 ±.01 内）：HateMM 平凡通过（4 次即超过），HCS 不通过（16 次最好 AP −.017；22 次 uniform −.006）。**不能作效率主张**；本节只回答"34 次买到了什么"：HateMM 上不训练时什么都没买到，HCS 上买到 .065 AP，且需要接近全部细窗。后续任务：训练后的模型在预算 4 / 22 下各跑一次完整训练确认（需要用户裁定）。

## 9. 规则 14 清单、预注册预期核对与去向（2026-09-07）

### 9.1 预注册预期（第 2 节）

| # | 预期 | 结果 |
|---|---|---|
| 1 | 两语料 seed 234 过规则 8 | 成立（HateMM .6209 / .8346，HCS .7070 / .6948）。 |
| 2 | 三 seed pooled 不低于修订 2 减一个 std；HateMM within 高于修订 2（目标 ≥ 候选 1 的 .646） | HCS 成立（.7045 / .6924 / .5678 vs .6976 / .6843 / .5488，三项都高）。HateMM 不成立：AP .6409 ± .0174 比修订 2 的 .6678 低 .027（超过一个 std），ROC .8421 ± .0080 低 .008；within .6310 高 .008 但未到 .646。 |
| 3 | `key_bias` within 低于 full 且 pooled 不高于 full | HCS 成立（within 低 .005 / .000 / .001，pooled 低 .007 / .012）；HateMM 不成立（AP 高 .010）。 |
| 4 | `ctx_in_rep` 与 full 在 ±.005 内 | 两语料都不成立，方向相反（HateMM ctx_in_rep 高 .023 AP；HCS 低 .009 / .010）。 |
| 5 | 门 A1 过时 `index_hmm` pooled 低于 full ≥ .005 或持平 | 成立：HateMM .037 / .020，HCS .007 / .024；`no_constraint`（.023 / .010，.008 / .024）与 `seconds_time`（.040 / .030，.007 / .012）同样两语料成立。 |
| 6 | 打乱证据时间对应后 pooled 下降 | HateMM 成立（.012 / .013 / within .023）；HCS 不成立（0）。 |

### 9.2 规则 14 (a)–(i)

| 项 | 状态 |
|---|---|
| (a) 三 seed 规则 8 确认，每 seed 数字来自该 seed 完整 20-trial 搜索的最优 trial | 满足。HateMM best trial 3 / 16 / 12，HCS 1 / 19 / 6；HateMM 领先 MACIL-SD .068 AP / .035 ROC（要求 ≥ .033 / .019），HCS 领先 Fed-WSVAD .142 AP、DSANet .164 ROC。 |
| (b) 报 seed 标准差 | HateMM AP ± .0174 / ROC ± .0080 / within ± .0101；HCS ± .0053 / ± .0089 / ± .0040。 |
| (c) 方法统一 | 同一架构、损失、训练、推理；语料间只有 5 个搜索标量不同。 |
| (d) validation 选 checkpoint；搜索空间 / 20 trial / 目标 test (AP+ROC)/2 在第 2 节搜索前写定；validation 选 trial 的 test 数字 | 满足；validation 选 trial 的 test 数字（各 `study_summary.json` 的 `validation_selected`）：HateMM trial 7 / 15 / 12 → test AP .6194 / .6603 / .6498、ROC .8055 / .8422 / .8410（均值 .6432 / .8296）；HCS trial 1 / 19 / 13 → .7070 / .6984 / .6943、.6948 / .6826 / .6765（均值 .6999 / .6846）。HateMM seed 3407、HCS seed 234 / 2025 与 test 选出的 trial 相同。 |
| (e) 无后处理、无 ensemble、无按语料分支；train-only teacher（Qwen VLM 裁定）已写明，`no_verdict` 消融两语料掉 .141 / .072 与 .110 / .129 | 满足。 |
| (f) 特征与 baseline 相同（I3D / VGGish / BERT），VLM 裁定是额外输入，已按 (e) 报去掉的数字 | 满足；"最强 baseline + VLM 裁定输入"的对照属第一批（用户跳过），未做。 |
| (g) 核心机制消融两语料 ≥ .01 | 13 / 17 成立（6.3）；不成立的四项不能作主张。 |
| (h) 评测器、split、GT、1 fps 未改 | 满足。 |
| (i) 两语料三项指标全报 | 满足（6.2、6.1、第 6 节进度表）。 |

### 9.3 结论与去向

- 修订 3 两语料三 seed 都过规则 8 确认；HCS 三项全面高于修订 2，HateMM AP 低 .027（主要原因：视频级校准 c 只加 logit 与 query 门控这两处骨干改动在 HateMM 上有害，在 HCS 上有益）。
- 可主张的部件（两语料 ≥ .01）：区间证据 HMM 及其两个设计（归一化时间、正例约束）、HMM 后验作先验与块标签、块级 MIL、先验项、CMAL、四格嵌入、逐头偏置本身（`no_bias`、`shared_bias`）、视频级校准 c（`no_context`，HateMM 只有 ROC 达标）。
- 不能主张：query 门控（B.2）、c 移到 logit（C′）、证据进 q/k（HCS 无效）、证据路由注意力整体相对候选 1 骨干（< .01）。
- 机制主张"证据决定从哪聚合"只在 HateMM 成立（7.1），HCS 上写 limitation。
- 规则 9：候选 3 已用 2 次修改，剩 1 次。下一步的选择（是否用掉最后一次修改：保留区间 HMM 融合，把 query 门控与 c 的放法退回修订 2 或去掉 c；w_fine 去留）交用户裁定；自适应查询回放（第 8 节）与 4 / 22 次预算下训练确认同样待裁定。
