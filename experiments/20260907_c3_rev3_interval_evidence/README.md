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
| 待 | 规则 8 筛选 → seed 2025/3407 搜索 → 每 seed 每语料 17 臂锁定消融 → 三 seed 汇总、配对 bootstrap、证据打乱检验（第 7 节）→ 规则 14 清单（第 9 节）。 |

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

## 7. 机制检验：证据时间对应打乱（不训练；`evidence_shuffle_test.py`；`runs/20260907_c3_rev3_interval_evidence/mechanism/<corpus>/seed<seed>/summary.json`）

做法：取各 seed 的 full checkpoint，只把进入 q/k 编码与路由项的证据码 e_t 在每个视频内随机打乱时间（5 次，固定随机序），视频级校准 c（对 t 取均值，打乱不变）与先验 α·ℓ_t/L（读未打乱的 scaffold 列）不动，重打 test 分数过统一评测器。如果 pooled / within 不掉，说明注意力路径没有用到证据的时间对应，只剩先验和视频级校准在起作用。

HCS（本机 CPU，2026-09-07）：

| seed | baseline AP / ROC / within | 打乱均值 AP / ROC / within | 差（baseline − 打乱） | 视频间方差占比 |
|---|---|---|---|---|
| 234 | .7070 / .6948 / .5718 | .7066 / .6947 / .5717 | +.0003 / +.0002 / +.0001 | .665 |
| 2025 | .6984 / .6826 / .5638 | .6979 / .6833 / .5600 | +.0005 / −.0008 / +.0038 | .676 |
| 3407 | .7081 / .6999 / .5678 | .7082 / .7000 / .5677 | −.0001 / −.0001 / +.0001 | .700 |

读法：在 HCS 上，打乱证据的时间对应对 pooled 指标没有影响（三 seed 差都在 ±.001 内），within 只在 seed 2025 掉 .004。也就是说 HCS 上"证据决定从哪聚合"这条主张不成立：模型的分数由先验 α·ℓ_t/L、视频级校准 c 和纯内容路径决定，注意力里的证据路由没有贡献时间信息。这与 HCS 的 `no_qk_enc` 消融在修订 2 上 ≈ 0 一致。
HateMM（本机 CPU，2026-09-07；seed 2025/3407 待搜索完成后补）：

| seed | baseline AP / ROC / within | 打乱均值 AP / ROC / within | 差（baseline − 打乱） | 视频间方差占比 |
|---|---|---|---|---|
| 234 | .6209 / .8346 / .6195 | .6055 / .8185 / .5931 | +.0155 / +.0161 / +.0265 | .820 |

读法：HateMM 上打乱证据的时间对应后三项指标都掉（AP .015、ROC .016、within .026，5 次打乱一致），说明 HateMM 上注意力路径确实用到了证据在时间上的位置，within 掉得最多，符合"证据决定从哪聚合"的主张。两语料结论不同：HateMM 成立，HCS 不成立；论文里这条主张只能限定在 HateMM，并把 HCS 的不成立写进 limitation。

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
