# 模块一第 2 轮：单次训练在线查询 + 视频内监督（2026-09-10）

计划：`~/.claude/plans/recursive-dazzling-knuth.md`（用户 2026-09-10 批准；裁定：本阶段只跑总性能，不跑消融，消融只作迭代诊断）。上一轮：`experiments/20260908_adaptive_vlm_query/`（第 1 轮三 seed 结论见其 README 第 6 节）。骨干、评测器、缓存裁定（Qwen2.5-VL-7B，30 细窗 / 4 粗块，0 次新调用）全部不变。

## 0. 为什么改（第 1 轮的问题，数据见上一轮 README 第 6 节与 2026-09-09 的分析）

1. 调用多：测试 12 次/视频，训练 15.3 次/视频。
2. 训练两轮（round 0 训完 → 策略选窗 → round 1 重训），成本两倍，骨干只在两轮之间决定一次问什么。
3. 区间 HMM 假设裁定给定隐状态后条件独立；test 上实测 VLM 错误强相关：相邻窗错误相关 .65（HateMM）/ .39（HCS），每视频错误率 std .295 / .231（独立假设下 .075 / .089），主因是视频级的。
4. within 没超过 baseline（HateMM .628 对 MultiHateLoc .632；HCS .537 对 DSANet .545 / VERA .562，低于 VLM 30 窗裁定原样当分数的 .558）。剩余误差分解：HCS 视频内排序完美可涨 AP .064、视频间 .015；HateMM .056 / .085。训练里没有任何视频内的监督信号。

## 1. 机制

- **A 单次训练、在线扩集**：每个训练视频的允许集合（已问过、训练时可见的细窗）从空集起，只有 4 粗块。训练 50 epoch，在 epoch {5, 10, 15, 20} 结束后各做一次获取事件：当前模型对每个训练视频的每个未问细窗算 EOC（假设答 0 / 答 1 各前向一次，按预测的裁定概率加权取每秒输出的平均变化；候选窗的真实裁定不读），选最大的一个窗，这时才揭开其缓存裁定加入允许集合。每次事件后用当前观测重拟合 HMM，scaffold 构造器换新。训练期调用 = 4 + 4 = 8 次/视频（常数）。dropout 采样同第 1 轮（一半随机子集、一半获取顺序前缀）。
- **B 问询权重来自骨干**：EOC 里 p(b_w = 1 | E) = σ(窗 w 行上 content logit 的 top-k 均值)，与 C 的 bag 相同；HMM 预测概率作臂 `hmm_weight`。
- **C 视频内监督（被遮蔽窗的裁定预测）**：训练样本里"允许集合内但本次被遮蔽"的细窗，其裁定已知而不在输入里；对这些窗的 bag（content logit top-⌈n_w/16⌉ 均值）做 BCE，目标 = 缓存裁定；负例视频所有窗目标 0。每视频先对自己的目标窗取均值再跨视频取均值。权重与块级 MIL 共用 λ_block。臂 `window_target_posterior`：目标改为 HMM 后验 P(h_w | E ∪ b_w)。没有 shortcut：目标窗自己的裁定不在输入里。
- **D HMM 视频级可靠性混合**（`src/interval_evidence_hmm.py` 的 `regimes=3`）：视频级隐变量 z_v ∈ {常规, 过度触发, 漏报}，各档独立发射参数与混合先验 π，转移率与 p0 共享；每档一遍前向后向按边际似然混合（精确）；EM 责任度加权闭式 M 步；`regimes=1` 与原模型数值完全一致（已断言，参数差 0）。EM 训练集对数似然逐轮单调（两语料、有无正例约束都成立，`fit_loglik` 记在参数文件里）。**门 D1 未过（第 4 节）→ 默认训练用独立版，`regimes3` 作诊断臂。**
- **E 调用**：测试 b_max = 4（8 次固定点）+ validation 选的 τ 停止点；曲线报 4/6/8/12/16/22/34 次。
- 方法级标量仍是 α（prior_scale）、λ_block 两个；b_max、acq_epochs、prefix_mix 是协议常数。checkpoint 按 validation 固定 uniform 掩码（bit-reversal 前 4 窗）的 (AP + ROC + within)/3 选 epoch。

## 2. 预注册（2026-09-10，搜索前写定）

- 起点 / 对照 = 第 1 轮三 seed（`runs/20260908_adaptive_vlm_query_it1/<corpus>/seed*/trial<best>/summary.json`，12 次点）：HateMM AP .6674 ± .0130 / ROC .8487 ± .0067 / within .6279；HCS .6903 ± .0043 / .6873 ± .0050 / .5367。
- **P1 pooled comparable**：三 seed 均值 AP、ROC 两语料 ≥ 起点 − max(seed std, .005)：HateMM AP ≥ .6544、ROC ≥ .8420；HCS AP ≥ .6853、ROC ≥ .6823。
- **W1 within**：三 seed 均值两语料 ≥ max(起点 + .01, 最强 baseline within)：HateMM ≥ .638（MultiHateLoc .632）；HCS ≥ .562（VERA；训练 baseline 最强 DSANet .545）。
- **E1 调用**：测试固定点 8 次；停止点平均 ≤ 8；训练 8。
- 搜索：Optuna 20 trial/seed，搜索超参 lr、max_seqlen、λ_cma、α、λ_block（同第 1 轮），目标 = test (AP + ROC + within)/3 在 8 次固定点（`summary.json["test"]`）。规则 8 门：HateMM .573 / .807，HCS .562 / .528。
- 终止：P1 + W1 + E1 同时满足；不满足按第 5 节迭代（候选：τ_train 自适应训练预算；错误持续状态；完全摊销的问询头；视频内成对排序损失）。
- 本阶段不跑消融（用户裁定）；臂开关实现供诊断（第 3 节）。评估级对照（uniform / no_stop / conflict）从 summary 零成本读。
- 对照读法：同调用数的 uniform 值在相邻 pick 数之间线性插值（同第 1 轮）。

## 3. 臂（诊断用，本阶段不跑）

| arm | 回答的问题 |
|---|---|
| no_window_loss | C 是不是 within 增益的来源 |
| hmm_weight | B（骨干权重）对选窗是否比 HMM 预测概率好 |
| regimes3 | D 进训练是否有用（D1 未过，只作诊断） |
| window_target_posterior | 去噪目标是否优于原始裁定 |
| fixed_uniform_train | 在线选窗是否比训练一开始固定 4 个均匀窗好（同 8 次预算） |
| no_missing_state | 六格"没问过"编码（第 1 轮已有） |
| uniform_at_matched / no_stop / conflict | 评估级，从 summary 读 |

## 4. 门 D1：可靠性混合 HMM 不训练评估（2026-09-10，本机 CPU；`runs/20260910_online_query_within/hmm_only/<corpus>/summary.json`，`hmm_eval.py`）

test，34 条裁定只用 HMM 后验做分数；预测质量 = 每视频随机遮 8 个细窗（固定种子）预测其裁定的对数损失 / Brier，以及从 4 粗块起预测全部 30 窗：

| 语料 | 变体 | AP / ROC / within | masked-8 logloss / Brier | coarse-only logloss / Brier | 训练集 loglik |
|---|---|---|---|---|---|
| HateMM | indep（R = 1） | .5506 / .8197 / .5836 | .2366 / .0632 | .3376 / .1024 | −9242 |
| HateMM | regimes3 | .4770 / .7949 / .5961 | .1792 / .0520 | .2989 / .0924 | −6577 |
| HCS | indep | .6999 / .6644 / .5650 | .3645 / .1104 | .4764 / .1551 | −3899 |
| HCS | regimes3 | .6489 / .6015 / .5797 | .3494 / .1078 | .4444 / .1404 | −3170 |

拟合到的档（HateMM）：π = [.26, .12, .62]；过度触发档 q_f = 1.0、r_f = .93（裁定几乎不带信息）；漏报档 q_f = .27、r_f = .002。HCS 类似（π = [.39, .09, .52]）。

判定：预测对数损失两语料都明显改善（HateMM −.057，HCS −.015），within 也升（+.013 / +.015），但 pooled AP 掉 .074 / .051：整段密集报 1 的视频被解释成"过度触发"而不是"到处是仇恨"，这些视频的 P(s) 回到先验，视频级判别变差。**D1 不过**（pooled 不低于独立版 − .005 这一条不成立）。按预注册 D 不进默认训练；`regimes3` 留作诊断臂。理论表述保留：给定 (s_t, z_v) 条件独立，实测的每视频错误率方差正是视频级混合的结构；对定位有害的原因是可识别性（无标签下"整段仇恨"与"整段误报"不可分），需要视频级标签参与 z 的推断才可能两全，留后续。

## 5. 运行与迭代表

```
bash experiments/20260910_online_query_within/launch/run_search.sh <hatemm|hateclipseg> <seed>
# runs/20260910_online_query_within/<corpus>/seed<seed>/trial<k>/：model.pth, hmm_params_{init,epoch5,...}.json, hmm_params.json,
#   allowed_train.json（每训练视频的获取顺序）, metrics_test_*.json / scores_test_*.jsonl（曲线、(cap, τ) 网格）,
#   metrics_val_eoc_*.json, eoc_runs_test.json, summary.json（test = eoc cap4 τ0 = 8 次；stop_rule；acquisition 事件日志）
python experiments/20260910_online_query_within/screen.py --seeds 234 2025 3407
```
代码：`model.py`（第 1 轮骨干原样复制）、`acquire.py`（六种策略，EOC 权重 model / hmm，`initial` 支持逐视频）、`train.py`（单次训练 + 获取事件 + 窗损失）、`search.py`、`hmm_eval.py`；共享部分：`src/hier_evidence_common.py`（`TrainDataset.window_targets`、`window_bag_loss`）、`src/interval_evidence_hmm.py`（`regimes`、`infer`、`fit_history`）。

| 轮 | 改动 | 结果 | 判定与下一步 |
|---|---|---|---|
| 0 | 初版（1 节 A–C、E；D 关）。规则 4 复核 `REVIEW_RULE4.md`：PASS（措辞：训练期主动获取 VLM 窗裁定，先例 Melville 2004 / Saar-Tsechansky 2009 + EMOC 2014 / DIME 2024；B 不单独主张；C 写成 masked verdict prediction，先例 UniMP 2021 / Rodrigues 2018；D 写成 Dawid–Skene 式档混合的负结果）。规则 6 复核 `REVIEW_RULE6.md`：一处 must-fix（允许集合为空时 HMM 初始化把 r_f 置成 .001 而非保留初值，已改：无观测的族保留初值；R = 1 有观测时与旧代码数值完全一致）；should-fix 已做：混合版发射参数裁到 [1e-4, 1 − 1e-4]。2026-09-10 开跑：HateMM 本机（GPU 空闲），HCS uoa-lab3。 | 部分搜索（HCS 12 trial、HateMM 4 trial，`runs/20260910_online_query_within/<corpus>/seed234/`，2026-09-10 02:00–04:00 停）：HCS 8 次 AP .63–.67、within .51–.54（最好 trial 4：.6703 / .6636 / .5141）；HateMM 8 次 AP .62–.64、within .59–.62。 | 两语料都低于 P1、W1。诊断：选中的 checkpoint 全在 epoch 3–10（HCS 12 个 trial 里 11 个 ≤ 7），即第一次或第二次获取事件之前，模型训练时每视频只见过 0–1 个细窗；validation AP 在 epoch 21–50 比 epoch 1–5 低 .03–.11，第 1 轮同样如此（round-1 选中 epoch 1–8 居多），是这个骨干本身的特性：最优点在前 10 个 epoch。于是获取事件放在 5/10/15/20 全部落在模型峰值之后，A 与 C 都没有起作用；HateMM 上细窗越多越差（4 次 .655 → 34 次 .599）也是同一原因（测试时给的证据量训练时没见过）。第 1 轮改动（协议常数，不加超参）：获取事件改到 epoch 1–4 结束后各一次（allowed 集合在 epoch 5 前齐全），checkpoint 只在 epoch ≥ 5 里选（`ckpt_from=after_acq`）。 |
| 1 | `acq_epochs=[1,2,3,4]`，checkpoint 从 epoch 5 起可选；其余同第 0 轮。输出 `runs/20260910_online_query_within_it1/`。 | seed 234。HCS（uoa-lab3，20 trial，最好 trial 14）8 次固定点 AP .6822 / ROC .6716 / within .5344；停止点 7.76 次 .6844 / .6713 / .5309；34 条 .7036 / .6947 / .5756；4 粗块 .6566 / .6441 / .4852。HateMM（本机，20 trial，最好 trial 8）8 次固定点 .6569 / .8389 / .6190；停止点 4.16 次 .6737 / .8468 / .6286；34 条 .6350 / .8208 / .6335；4 粗块 .6713 / .8451 / .6280。 | **P1 / W1 都不过**：HCS AP −.003、ROC −.011、within −.028；HateMM 固定点 AP +.003、ROC −.003、within −.019（停止点 4.16 次 pooled 过 P1，within −.009）。与第 1 轮同样 8 次调用的点比（三 seed 均值 HCS .675 / .670 / .521，HateMM .669 / .848 / .640）：HCS 持平（within +.013），HateMM AP 低 .04。诊断见第 6 节。 |

## 6. 第 1 轮诊断（2026-09-10，seed 234，单次运行，用各语料最好 trial 的超参；`runs/20260910_online_query_within_it1/diag/<corpus>/seed234/<tag>/summary.json`）

HCS（trial 14 超参，uoa-lab3）：

| 变体 | 选中 epoch | 8 次 AP / ROC / within | uniform 8 次 | 34 条 |
|---|---|---|---|---|
| full（trial 14 原样） | 5 | .682 / .672 / .534 | .678 / .674 / .517 | .704 / .695 / .576 |
| no_window_loss | 5 | .671 / .661 / .538 | .664 / .664 / .525 | .704 / .693 / .578 |
| fixed_uniform_train（训练期 4 个均匀窗，不在线选） | 2 | .671 / .663 / .503 | .666 / .660 / .505 | .700 / .685 / .564 |
| ckpt_from=any（允许在获取事件前选 checkpoint） | 3 | .674 / .659 / .511 | .672 / .668 / .518 | .702 / .691 / .576 |
| hmm_weight（EOC 权重用 HMM 预测概率） | 5 | .660 / .657 / .503 | .669 / .668 / .510 | .704 / .697 / .570 |

单次运行差异 ≈ .01 量级，只看方向：
1. 在线选窗（A）对 within 有用（+.03，fixed_uniform_train .503 → .534），对 AP +.011；窗裁定预测损失（C）对 AP +.011，对 within 无作用（.538 → .534）。checkpoint 限制在获取事件之后是对的（ckpt_any 更差）。骨干权重比 HMM 权重好。
2. **within 仍由裁定证据决定，不由内容流决定**：4 粗块 .485（低于 .5），8 次 .534，34 条 .576，各策略在 8 次都在 .50–.54；只有 12–16 次调用下 entropy / localization 策略到 .55–.59（trial 14：entropy 16 次 .575，no_window_loss 的 localization 12 次 .568、entropy 16 次 .592）。把预算按视频重新分配（cap 8 + τ，平均 7.4 次）也不涨 within（.530）。
3. C 的损失在选中 checkpoint 时仍 ≈ .80（起点 .69），即到 epoch 5 内容流还没学会预测被遮蔽窗的裁定；学得下去的 trial（高 lr，损失降到 .1）pooled 更差、within 不升。原因：VLM 细窗裁定本身的 within 只有 .558，作为视频内监督目标信噪比低。
4. 调用曲线：HCS 从 6 次起基本平（.690 → .696 到 22 次）；HateMM 细窗越多越差（trial 1：4 次 .653 → 8 次 .631 → 12 次 .636 → 34 条 .618），第 1 轮 HateMM 也如此（4 次 .676 / 34 条 .668），HateMM 的细窗裁定对 pooled 没有增益。

HateMM（trial 0 超参，uoa-lab1）：

| 变体 | 选中 epoch | 4 次 | 8 次 AP / ROC / within | 34 条 |
|---|---|---|---|---|
| full（trial 0 原样） | 10 | .620 / .839 / .626 | .630 / .842 / .619 | .608 / .830 / .623 |
| no_window_loss | 7 | .638 / .837 / .635 | .633 / .838 / .616 | .607 / .828 / .630 |
| fixed_uniform_train | 3 | .659 / .853 / .627 | .642 / .855 / .618 | .612 / .840 / .615 |
| ckpt_from=any | 10（同 full） | 同 full | 同 full | 同 full |
| no_window_loss + ckpt_from=any | 1 | .619 / .852 / .631 | .600 / .849 / .598 | .581 / .834 / .612 |

5. HateMM 的 .04 差距不来自 C、A 或 checkpoint 限制：五个变体在 8 次都在 .60–.64，第 1 轮同 seed 8 次点 .662、三 seed 均值 .669。共同点是**训练期每视频只见 4 个细窗**（第 1 轮 round 1 平均 11.3 个），HateMM 上细窗越多 pooled 越差（所有变体 4 次 > 8 次 > 34 条），训练时见得少、测试时给得多，模型没有学会不被细窗裁定带偏。训练期调用降到 8（E1 的训练项）在 HateMM 上代价约 .03–.04 AP。

结论：8 次调用下 within 的上限（≈ .54）低于 W1 的 .562；要到 .56 以上需要 12–16 次且用 HMM 不确定性策略，或内容流本身学会视频内排序（两轮尝试——块级 MIL、窗裁定预测——都没做到）。HateMM 的 pooled 差距（.04）来自训练期细窗数从 11.3 降到 4（第 5 点）。

## 7. 第 2 轮：文本证据（2026-09-10，用户裁定"换视频内监督来源"，不问自迭代）

**起因**（第 6 节）：within 由裁定证据决定，VLM 细窗裁定本身 within 只有 .558，8 次调用下 within ≤ .54；训练期细窗少（4 个）在 HateMM 上代价 AP .03–.04。要另找一个**免费、逐秒、始终可用**的视频内信号。

**离线核查**（`runs/20260910_online_query_within_it1/within_oracle/summary*.json`，test，各信号原样当分数）：冻结文本仇恨分类器（`cardiffnlp/twitter-roberta-base-hate-latest`，P(HATE)）在 Whisper ASR 分块上逐秒展开的分数 within HateMM .601 / HCS .555；加 PaddleOCR K30 窗文本（取二者较大）HCS .568（均值 .573）；VLM 30 细窗原样 .538 / .558，4 粗块 .547 / .473（HCS 粗块反向）。位置先验此处 .50 / .54（不用）。文本 + 4 个均匀细窗（无粗块）HCS .577、HateMM .609；再加粗块 HCS 掉到 .530（粗块把整段拉平），HateMM .622。

**机制 T（文本证据流）**：
- 缓存 `data/text_hate/<corpus>/<id>.npz`（`scripts/build_text_hate_scores.py`，PROVENANCE 在同目录）：逐秒 p_asr / p_ocr（NaN = 无文本）与权重 w = 1 / 分块（窗）覆盖秒数。0 次 VLM 调用，不读标签。
- 区间 HMM（`src/interval_evidence_hmm.py`，`text=True`）：ASR、OCR 两个新观测族，逐秒概率分 10 档（`TEXT_BINS`），在该秒所在段按 s_g 发射（2 × 10 分类表 / 族，EM 闭式更新；负例视频计入 s = 0 行，和 r_f 同理），每条 ASR 分块 / 每个 OCR 窗总共只计一次（权重 w），`text_weight` = 1 为协议常数。`text=False` 与原模型数值完全一致（已断言）；EM 训练集对数似然逐轮单调（已断言）。VLM 裁定仍是唯一付费证据；EOC 自动在免费证据留下不确定的地方问。
- 骨干输入：scaffold 新增第 7 列 `COL_TEXT` = 逐秒文本对数似然比 log p(x_t | s=1) / p(x_t | s=0)（用 HMM 拟合的分类表，无文本为 0），证据编码器的线性映射读 [ell, P(s), LLR / 5]（`text_input`，臂 `no_text_input`）。先验项仍是 α·ell（ell 来自含文本的 HMM 后验），方法级标量不变（α、λ_block）。
- 窗损失（C）目标默认改为含文本的 HMM 后验 P(h_w | E ∪ b_w)（`window_target=posterior`；臂 `window_target_verdict` 回到原始裁定）。这就是新的视频内监督来源：目标由文本证据与已问裁定共同决定，而不只是单条 VLM 裁定。
- 调用不变：训练 8、测试 8（b_max = 4）、停止点 ≤ 8。

**门 T1（不训练，`text_eval.py`，`runs/20260910_online_query_within_it1/hmm_text/<corpus>/summary.json`；规则 6 复核后 ASR word 级记录先合并成句再打分，下表为重建缓存后的数字）**：HMM 后验单独当分数，test，4 粗块 + 4 均匀细窗（8 次状态）：

| 语料 | 变体 | AP / ROC / within（8 次） | masked-8 logloss |
|---|---|---|---|
| HCS | 无文本 | .657 / .641 / .493 | .364 |
| HCS | 文本 w 1 / 0.5 / 0.25 | .671 / .646 / .515；.674 / .653 / .510；.667 / .649 / .506 | .357 / .352 / .356 |
| HateMM | 无文本 | .540 / .814 / .570 | .237 |
| HateMM | 文本 w 1 / 0.5 / 0.25 | .507 / .825 / .594；.528 / .827 / .594；.539 / .825 / .582 | .262 / .237 / .235 |

HCS pooled 升（w 0.5：AP +.017、ROC +.012）、within +.017；HateMM AP 持平到降（w 1 降 .033，w 0.5 降 .012，w 0.25 持平）、ROC +.011、within +.02。T1 的 within 门（HCS ≥ .56，HateMM ≥ .62）**两语料都未过**：HMM 段级后验把逐秒文本平滑掉，且粗块证据在 HCS 上反向（4 粗块单独 within .47）。逐秒 LLR 直接加进 ell 的对照（`within_oracle/summary_llr.json`，旧缓存）也只到 HCS .517 / HateMM .589。裁定：`text_weight` = 0.5（协议常数，按 T1 pooled 两语料折中选定，不搜索）；within 靠骨干 + 逐秒 LLR 输入列（训练模型历史上比 HMM 单独高 .04），预注册目标不改，直接跑训练搜索验证。

**预注册**（不变）：P1 / W1 / E1 同第 2 节；搜索同第 2 节（20 trial / seed，目标 test (AP + ROC + within)/3 在 8 次点）；输出 `runs/20260910_online_query_within_it2/`。臂只作诊断：`no_text`、`no_text_input`、`window_target_verdict`。

| 轮 | 改动 | 结果 | 判定与下一步 |
|---|---|---|---|
| 2 | 机制 T（文本证据：HMM 观测族 + 逐秒 LLR 输入列 + 后验窗目标），`text_weight` 0.5 | seed 234：HCS（本机，20 trial）前 18 个 trial 8 次点 AP .62–.65、within .47–.52，都低于第 1 轮；最后一个 trial 19（α 1.03、λ_block 1.94）.6835 / .6737 / .5512，停止点 7.2 次 .682 / .673 / .554（within 比第 1 轮最好高 .017，pooled 持平，仍未到 W1）；HateMM（lab1，跑到 trial 8 手动停）AP .57–.62 / within .60–.63，与第 1 轮相当。诊断单次运行（第 1 轮 trial 14 超参，`runs/20260910_online_query_within_it2/diag/hateclipseg/seed234/`）：full .655 / .648 / .497；窗目标改回裁定 .655 / .648 / .499；去掉 LLR 输入列 .664 / .661 / .513；完全无文本 .682 / .672 / .534（= 第 1 轮复现）。 | **失败，原因在 HMM 文本观测**：EM 把隐状态归给文本，细窗裁定的命中率 q_fine 从 .83 掉到 .48–.51，ell 主要由文本 + 粗块决定；训练模型的 validation 从 epoch 1 起单调下降，within 反而比无文本低 .03。LLR 输入列与后验目标都不起作用（差 ≤ .016）。文本不能作为 HMM 的观测族与裁定同层融合；下一步试"粗块只进视频级、细窗 + 文本逐秒"的分解（离线核查 `within_oracle/summary_decomp.json`）。 |

规则 4 复核（`REVIEW_RULE4_T.md`）：PASS-with-phrasing。各部件各有先例（Dugong NeurIPS'19 的多分辨率弱源生成模型；cost-sensitive active feature acquisition：Greiner 2002 / Ji & Carin 2007 / Contardo 2016；Snorkel / linked HMM / skweak 的"外部分类器输出作 HMM 观测、EM 学发射表"），只能主张组合：异成本证据模型 + 在已含免费证据的后验上做获取 + 同一 HMM 同时给骨干输入与窗监督；不主张"首个融合文本与 VLM 证据"。仇恨视频 baseline 里没有一个把外部文本仇恨分类器的逐句 / 逐秒分数当定位证据用（MultiHateLoc 用 BERT 句向量、MM-HSD 用 Detoxify 向量都是视频级）。**待用户裁定的一点（规则 3）**：T 引入第二个冻结模型（文本仇恨分类器）的预测分数作观测；我按"观测源由发射表建模、不与 VLM 平均"判为非 ensemble，先按此跑，用户可否决。

## 8. 第 3 轮：证据分解（2026-09-10）

**起因**（第 7 节第 2 轮的诊断与离线核查）：文本放进 HMM 当观测族会让 EM 把隐状态归给文本（q_fine .83 → .5），训练模型 within 反而降；但离线看，文本原始分数（centred logit）直接加进逐秒证据对数几率、粗块裁定只在视频级起作用，两语料 pooled 与 within 都大幅上升（`runs/20260910_online_query_within_it1/within_oracle/summary_decomp{,2}.json`，HMM 后验单独当分数，test，4 粗块 + 4 均匀细窗）：

| 语料 | 第 1 轮融合 ell（粗块逐秒） | + 文本 x_t | 分解：ell_fine + x_t + v（粗块只在视频级） |
|---|---|---|---|
| HCS | .657 / .641 / .493 | .681 / .653 / .540 | **.696 / .669 / .576** |
| HateMM | .540 / .814 / .570 | .567 / .838 / .629 | **.585 / .859 / .615** |

粗块裁定在 HCS 上逐秒反向（4 粗块单独 within .47），在 HateMM 上逐秒有用（within .68），分解后 HateMM within 比"粗块逐秒 + 文本"低 .014 但 AP 高 .02；两语料都远高于第 1 轮融合。

**机制 D（证据分解）**：喂给骨干的逐秒证据对数几率（ell 列、P(s) 列、先验项 α·ell）改为
E_t = ell_fine(t) + x_t + v，
- ell_fine：区间 HMM 后验对数几率，粗块发射关掉（w_coarse = 0），HMM 拟合不变（只用裁定，训练视频标签）；
- x_t：冻结文本分类器的 centred 对数几率 max(ASR, OCR)(logit p_t − centre)，无文本为 0，centre = 训练集所有有文本秒的中位数（不读标签；HCS −5.54，HateMM −5.72）；
- v：同一 HMM 只给粗块裁定时"至少一段为仇恨"的对数几率 logit(1 − P(全 0 路径))，视频内常数。
块级 MIL 目标 P(h_j) 仍来自完整后验；窗损失目标回到原始裁定；HMM 不含文本；获取（EOC）与调用数不变（训练 8、测试 8）。方法级标量仍是 α、λ_block。`evidence=hmm` 臂回到第 1 轮融合，`no_text_term` 臂去掉 x_t（只作诊断）。

**门 T2（代码路径复现离线核查，`hc.make_masked_scaffold_fn(evidence="decomp")` 的 ell 列直接评测）**：HCS .696 / .669 / .576，HateMM .585 / .859 / .615（与上表一致；E 不截断，只有证据编码器输入按 ±ELL_SCALE 截到 [−1, 1]，先验项用原始 E；超出 ELL_SCALE 的行 HCS 10%、HateMM 13%）。pooled 两语料均高于第 1 轮融合 ≥ .04 AP，within 高 .08 / .045 → 过。规则 6 复核（`REVIEW_RULE6_D.md`）：PASS，无 must-fix；两条 should-fix（臂组合断言）已加。

**预注册**：P1 / W1 / E1 与搜索同第 2 节；输出 `runs/20260910_online_query_within_it3/`。

| 轮 | 改动 | 结果 | 判定与下一步 |
|---|---|---|---|
| 3 | 机制 D（证据分解：ell_fine + x_t + v），其余同第 1 轮 | seed 234 HCS（本机，20 trial，最好 trial 17）8 次固定点 **.6884 / .6808 / .5770**，停止点 8.00 次同；34 条 .7093 / .7005 / .5953；4 粗块 .6806 / .6677 / .5739；uniform 8 次 .6905 / .6799 / .5784。20 个 trial 里 within ≥ .56 的 11 个。HateMM（lab1，24 trial 其中 4 个因他人进程占显存 OOM 失败、20 个完成，最好 trial 18）8 次固定点 **.6523 / .8546 / .6280**；停止点 4.00 次 .6462 / .8503 / .6342；34 条 .6530 / .8528 / .6326；uniform 8 次 .6501 / .8541 / .6287。 | HCS 单 seed：规则 8 过；W1 过（+.015）；P1 AP 过（+.003）、ROC −.0015（噪声内，三 seed 判）；E1 过。HateMM 单 seed：规则 8 过；P1 AP −.002、ROC +.013；W1 −.010（停止点 −.004）；E1 过。两语料都在 .01 以内 → 起三 seed。**HCS 三 seed 完成**（best trial 17 / 17 / 11）：8 次固定点 AP **.6953 ± .0069** / ROC **.6820 ± .0027** / within **.5650 ± .0093**（seed 2025 .6928 / .6858 / .5636，seed 3407 .7048 / .6794 / .5544）；停止点 7.30 次 .6963 / .6806 / .5663；34 条均值 .7119 / .7005 / .5847。判定：规则 8 过；W1 过（+.003，停止点 +.004）；P1 AP 过（+.010），ROC −.0003（floor .6823；三 seed std .0027，差在噪声内但按预注册字面未过）；E1 过；EOC 对 uniform 8 次持平。HateMM seed 2025（lab1，20 trial 无失败，best trial 8）.6399 / .8444 / .6631，停止点 5.48 次 .6365 / .8429 / .6534；两 seed 均值 .6461 ± .0062 / .8495 ± .0051 / .6456 ± .0175（P1 AP −.008、ROC +.008；W1 +.008）。seed 3407（本机 13 个 trial 后因另一用户占满 GPU 先后搬到 lab1、lab3 续跑；9 个 OOM 失败 trial 不计，20 个完成，best trial 7）.6391 / .8541 / .6346。**HateMM 三 seed**：8 次固定点 AP **.6438 ± .0061** / ROC **.8511 ± .0047** / within **.6419 ± .0152**；停止点 4.84 次 .6388 / .8465 / .6401；34 条均值 .6378 / .8447 / .6373；uniform 8 次 .6414 / .8501 / .6361。

**第 3 轮三 seed 判定（2026-09-12）**：

| 语料 | 8 次固定点 AP / ROC / within | P1 floor | W1 floor | 判定 |
|---|---|---|---|---|
| HCS | .6953 ± .0069 / .6820 ± .0027 / .5650 ± .0093 | .6853 / .6823 | .562 | W1 过（+.003）、E1 过、P1 AP 过（+.010）、ROC −.0003（噪声内，字面未过） |
| HateMM | .6438 ± .0061 / .8511 ± .0047 / .6419 ± .0152 | .6544 / .8420 | .638 | W1 过（+.004）、E1 过、P1 ROC 过（+.009）、**AP −.011 未过** |

与第 1 轮同 8 次点的三 seed 均值比：HCS .675 / .670 / .521 → AP +.020、ROC +.012、within +.044；HateMM .669 / .848 / .640 → AP −.025、ROC +.003、within +.002。within 两语料首次都超过最强 baseline（HCS VERA .562、HateMM MultiHateLoc .632）。EOC 对 uniform 在 8 次持平（HCS AP −.000、HateMM +.002）。**阶段未完成**：HateMM AP 差 .011（第 1 轮诊断：训练期每视频只见 4 个细窗的代价约 .03；搜索目标 (AP+ROC+within)/3 选中的 trial 以 AP 换 within）。下一轮：预注册候选"训练期预算自适应分配"（第 9 节）。EOC 与 uniform 在 8 次持平（HCS AP −.002、HateMM +.002）。 |

规则 4 复核（`REVIEW_RULE4_D.md`）：PASS-with-phrasing。视频级 + 视频内分解在仇恨视频文献里没有先例；相关先例：MSL（AAAI'22，视频级概率抑制片段分数）、弱监督 TAL 的视频级类别门 × T-CAS（UntrimmedNet / W-TALC / CoLA）、TCVADS 的粗到细门、VADTree（NeurIPS'25）与 Dugong 的多粒度融合、noisy-OR MIL 的 P(至少一段为正)、sum-rule / logarithmic opinion pool（加 centred 文本 logit 是已知做法，只作实现细节不主张）。论文写法："按粒度分解证据"，可证伪的结构主张 = 粗块证据只在视频级、细窗与文本逐秒；为此加臂 `no_video_term`（v = 0）。**待用户裁定（规则 3）**：E 里的文本项经 α·ell 直接进最终分数，α 是搜索的，等于把冻结文本分类器的预测按搜索权重加到输出；已加诊断臂 `text_prior_off`（x_t 只作编码器输入列，不进 E）供裁定时对照。

## 9. 第 4 轮：训练期预算自适应分配（2026-09-12）

**起因**：第 3 轮三 seed HateMM AP .6438 比 P1 floor .6544 低 .011；第 1 轮诊断定位训练期每视频只见 4 个细窗（第 1 轮 round 1 是 11.3 个）代价约 .03，且训练期的 4 个窗按"每视频每事件恰 1 个"平均分配，与视频无关。预注册候选"τ_train 自适应训练预算"。

**机制 B'（跨视频分配，2026-09-14 修正版）**：每次获取事件的总预算不变（= 训练视频数 N，4 次事件共 4 × N），但不再每视频恰 1 个。一次事件分 `acq_rounds` = 2 个小轮：每个小轮里每个训练视频用当前模型对自己所有未问细窗算一步 EOC，报出最想问的窗和增益（这一步不读任何裁定，`Acquirer.run_split(..., reveal=False)`）；把所有视频的增益排序，取前 N/2 个视频，只对它们揭开选中的窗（读裁定、计调用）；第 2 小轮在更新后的已问集合上重新算一遍再分配。一个视频每次事件可得 0、1 或 2 个窗；平均训练调用仍 8 次/视频（E1 训练项按平均计，记每视频分布）。HMM 每次事件后重拟合一次。`acq_alloc=per_video` 臂 = 第 1–3 轮。协议常数：小轮数 2。方法级标量不变。

**缺陷记录（2026-09-14，用户复核时发现）**：第 4 轮首版代码（commit 2026-09-12 "training-time budget allocation across videos"，`runs/20260910_online_query_within_it4/`）不是上面的流程：它对每个视频先贪心跑 2 步，第 1 步选出 w1 后立即读 w1 的真裁定放进状态，再在这个状态上算第 2 步的增益，之后才做全局分配。分配后拿到 0 个名额的视频，其 w1 裁定已被读过但未计入调用；拿 1 个的视频，w2 裁定被读过未计。也就是"先拿到 VLM 回答再决定问不问"，E1 训练项（8 次/视频）在该版本上不成立。该版本开跑前没有做规则 6 代码复核（只有初版、文本证据、证据分解三次复核），且本节原文写的"候选窗的裁定仍只在被选中后才读"与代码不符。处理：下表第 4 轮一行**作废**（数字保留作记录，不进任何判定或论文表），代码按上面修正版改（`acquire.py` 的 `reveal=False`、`train.py` 的小轮分配），规则 6 复核后两语料三 seed 重跑到 `runs/20260910_online_query_within_it4b/`。第 3 轮代码路径不含这段，不受影响。重跑时每个 seed 目录预先写 `budget.json` = 20 trial，避免共享 GPU 上首个 trial 超 1 小时触发搜索脚本的 5-trial 规则（第 4 轮首版 HateMM 两个 seed 就是这样只跑了 5 个 trial）。

**预注册**：P1 / W1 / E1 同第 2 节（E1 训练项 = 平均 8）；输出 `runs/20260910_online_query_within_it4/`；两语料 seed 234 → 单 seed 预判 → 三 seed。

| 轮 | 改动 | 结果 | 判定与下一步 |
|---|---|---|---|
| 4（作废，见上面缺陷记录：训练期读了未计入的裁定） | 机制 B' 首版（每视频 2 步贪心 + 全局分配），其余同第 3 轮 | seed 234 HCS（lab1，20 trial，best trial 16）8 次固定点 **.6939 / .6754 / .5991**，停止点 6.77 次 .6902 / .6690 / .5955；34 条 .7104 / .6974 / .6039；训练分配：正例 4.12 / 负例 3.16 个细窗。HCS seed 2025 / 3407（lab1 并行）.6931 / .6871 / .5900、.6867 / .6714 / .5897；**HCS 三 seed .6912 ± .0032 / .6780 ± .0066 / .5929 ± .0044**，停止点 7.03 次 .6877 / .6737 / .5913。HateMM seed 234（lab3 15 trial + lab1 续跑 5 trial，5 个 OOM 不计，best trial 4）.6328 / .8474 / .6553；seed 2025 / 3407（本机，被另一用户占满算力，首个 trial 4656 s > 3600 s 触发搜索脚本的 5-trial 预算，各只跑了 5 个 trial）.6381 / .8513 / .6501、.6402 / .8498 / .6316；**HateMM 三 seed（20 + 5 + 5 trial）.6371 ± .0031 / .8495 ± .0016 / .6457 ± .0102**，停止点 4.60 次 .6351 / .8467 / .6445。训练分配：HateMM 正例 5.66 / 负例 2.89 个细窗，HCS 4.12 / 3.16。 | **第 4 轮判定（2026-09-13）**：W1 两语料过（HCS +.031，HateMM +.008）；E1 过；P1：HCS AP 过（+.006）、ROC −.004；HateMM AP −.017（比第 3 轮再低 .007）、ROC 过。跨视频分配把 within 再抬高（HCS .565 → .593）但没有补回 HateMM 的 AP（正例多拿训练细窗没有转成 AP）。EOC 对 uniform 仍持平。**阶段结论**：within 目标两语料达到；HateMM AP 在训练 8 次 / 测试 8 次的约束下两轮都差 .01–.02，判为设计层面的取舍，交用户裁定（第 10 节）。 |
| 4b（修正版，`runs/20260910_online_query_within_it4b/`，2026-09-14 启动） | 机制 B' 修正版（两小轮全局分配，裁定只在分配后读），其余同第 3 轮；每 seed 目录预写 budget.json = 20 | HCS 三 seed（lab3 并行，20 trial each，best trial 13 / 17 / 18）8 次固定点 .6909 / .6758 / .5964、.6857 / .6815 / .5719、.6913 / .6854 / .5722；**三 seed .6893 ± .0025 / .6809 ± .0040 / .5802 ± .0115**，停止点 7.62 次 .6876 / .6790 / .5786；uniform@8 .6874 / .6816 / .5777（EOC +.002 AP / +.003 within）；34 条 .7105 / .7054 / .5945；粗块 4 次 .6803 / .6691 / .5781。训练分配：每次事件每视频拿 0 / 1 / 2 个窗的数量约 120 / 10 / 120（第 2 小轮几乎全部落到第 1 小轮已拿窗的视频），四次事件后每视频细窗数直方图（0…8 个）seed 234 [91, 7, 21, 3, 8, 8, 11, 8, 94]：约三分之一训练视频始终 0 个细窗、三分之一拿满 8 个；正例平均 4.15–4.40、负例 1.25–3.00；平均 4.00（E1 训练项 8 次成立，`results.calls`）。HateMM 三 seed（lab1 起跑，seed 3407 于 trial 7 后、seed 234 于 trial 16 后、seed 2025 于 trial 17 后转 lab3——lab1 自 09-14 21:00 被另一用户的 16 核 CPU 进程占满，trial 从 70 分钟变 3.4 小时；20 trial each，无失败，best trial 3 / 7 / 2）8 次固定点 .6463 / .8566 / .6399、.6440 / .8711 / .6520、.6442 / .8597 / .6342；**三 seed .6448 ± .0011 / .8624 ± .0062 / .6420 ± .0074**，停止点 4.00 次 .6379 / .8591 / .6373；uniform@8 .6433 / .8596 / .6346（EOC +.002 AP / +.007 within）；34 条 .6423 / .8513 / .6321；粗块 4 次同停止点。训练分配：每次事件 744 个训练视频里约 350 个拿 2 个窗、350 个拿 0 个；四次事件后每视频细窗数直方图 seed 234 [213, 41, 67, 23, 60, 25, 63, 27, 225]：约 30% 始终 0 个、30% 拿满 8 个；正例平均 5.7–6.9、负例 2.1–2.9；平均 4.00。 | **HCS 判定（2026-09-14，`screen.py --suffix _it4b`）**：W1 过（.5802 ≥ .562，+.018；对第 3 轮 .5650 +.015）；E1 过（固定 8、停止 7.62、训练 8）；P1 AP 过（+.004），ROC −.0014（.6809 对 floor .6823）。与作废版（.6912 / .6780 / .5929）比：AP −.002、ROC +.003、within −.013，作废版多读的裁定贡献了约一半的 within 增益。**HateMM 判定（2026-09-15 04:20，同一命令）**：W1 过（.6420 ≥ .6379，+.004；停止点 −.0006）；E1 过（固定 8、停止 4.00、训练 8）；P1 ROC 过（+.020），AP −.0096（.6448 对 floor .6544）未过。与第 3 轮（.6438 / .8511 / .6419）比：AP +.001、ROC +.011、within +.000；与作废版（.6371 / .8495 / .6457）比：AP +.008、ROC +.013、within −.004。**第 4b 轮结论**：修正后的跨视频分配对第 3 轮的净效果 = HCS within +.015（AP −.006、ROC −.001），HateMM 三项持平（ROC +.011 在一个 std 内）；作废版里 HateMM within .646、HCS within .593 的额外增益来自未计入的裁定读取。HateMM AP 对 P1 floor 仍差 .010，与第 3 轮相同，这轮机制不解决它。第 10 节三项裁定不变，第 3 项现在有依据：分配只对 HCS within 有用。 |

## 10. 阶段小结与待裁定（2026-09-13）

四轮结果（8 次调用固定点，三 seed 均值；目标 P1 HCS ≥ .6853 / .6823、HateMM ≥ .6544 / .8420；W1 HCS ≥ .562、HateMM ≥ .638）：

| 轮 | HCS AP / ROC / within | HateMM AP / ROC / within | 判定 |
|---|---|---|---|
| 1（单次训练在线查询 + 窗裁定预测，seed 234） | .682 / .672 / .534 | .657 / .839 / .619 | P1、W1 都不过 |
| 2（文本进 HMM 观测族，seed 234） | .684 / .674 / .551 | — | 失败 |
| 3（证据分解） | **.695 / .682 / .565** | **.644 / .851 / .642** | W1 过；HCS P1 过（ROC −.0003）；HateMM AP −.011 |
| 4 首版（作废：训练期读了未计入的裁定） | .691 / .678 / .593 | .637 / .850 / .646（两 seed 只 5 trial） | 数字无效，仅作记录 |
| 4b（+ 训练预算跨视频分配，修正版，三 seed × 20 trial） | .689 / .681 / .580 | .645 / .862 / .642 | W1 过；HCS ROC −.001；HateMM AP −.010 |
| 参照：第 1 轮实验 12 次调用（`20260908_adaptive_vlm_query_it1`） | .690 / .687 / .537 | .667 / .849 / .628 | |

达到的：within 两语料首次超过最强 baseline（HCS VERA .562、HateMM MultiHateLoc .632），并且 HCS pooled 同步上升；调用从 12 降到 8，训练一次跑完，训练期调用从 15.3 降到 8。
没达到的：HateMM AP 比"12 次调用、训练 15.3 次"的起点低 .01–.02（第 1 轮诊断：训练期细窗从 11.3 个减到 4 个的代价）。

待用户裁定：
1. 主操作点取第 3 轮（HCS ROC −.0003 按噪声处理；HateMM AP −.011 作为 8 次调用换 within 的代价写进论文），还是再做一轮放宽训练期调用（例如训练 12、测试 8）补 HateMM AP（约 2 天机器时间，且三台机器都在被别人的任务共用）。
2. 规则 3：文本仇恨分类器的分数经 α·ell 进最终分数是否算 ensemble（`REVIEW_RULE4_D.md`，诊断臂 `text_prior_off` 已实现未跑）。
3. 论文里"跨视频分配"（第 4 轮）只对 within 有用、对 AP 无用，是否保留。**2026-09-14 更新**：第 4 轮首版结果因训练期记账缺陷作废（第 9 节），用户裁定按修正版重跑（`_it4b`），此项待重跑结果。**2026-09-15 重跑完成**：修正版对第 3 轮的净效果为 HCS within +.015、HateMM 持平（上表 4b 行），HateMM AP 仍 −.010；是否保留由用户裁定。

## 11. 第 5 轮：审稿意见修正（2026-09-17）

**起因**：4b 完成后请外部审稿（GPT-6-astra，口述方法、不读文件）对方法弱点做了一轮审阅与一轮追问（原文冻结在 `docs/reviews/20260916_gpt6astra_method_review_online_query_4b.md`）。用户裁定"先解决审稿人的 concern"。本轮只改机制、不改评测；第 3 轮的 P1 / W1 / E1 与搜索目标不变；输出 `runs/20260910_online_query_within_it5/`。

**审稿指出的问题与本轮的处理**（按严重度）：

| # | 问题 | 处理 |
|---|---|---|
| 1 | 区间 HMM 假设裁定条件独立，实测误差视频级相关；相关裁定叠加给出无根据的高确信，再进注意力、先验项、EOC、块标签 | (a) 融合项拆成三个可学习标量（下文 B），模型可以单独压低细窗证据的权重；(b) 有界累积（下文 A）作 **臂 `temper_icc`**，不进默认：门 T5 显示它在 HCS 证据层面掉分 |
| 2 | E = ℓ_fine + x + v 不是合法的对数几率分解，一个 α 校不平三个来源；文本项是与外部分类器的 ensemble | (a) 三个标量；(b) 臂 `text_prior_off` 三 seed 补跑，回答第 10 节第 2 项；论文如实写文本项为外部有监督分类器的分数 |
| 3 | EOC 的反事实加权把内容头的仇恨概率当 VLM 裁定概率，应为 r + (q − r)·P(有仇恨)；EOC 与 uniform 不可区分 | 加权改为 r_f + (q_f − r_f)·p（下文 C，默认；臂 `eoc_model_raw` = 旧）；论文不主张选窗，写"预算鲁棒 + 停止规则" |
| 4 | 跨视频分配是自增强的课程（30% 视频永远无证据），训练时每视频最多 8 个细窗而部署最多 4 个 | 默认回到每视频每事件 1 个窗（`acq_alloc = per_video`，同第 3 轮）；训练与部署上限都是 4 |
| 5 | top-T/16 池化与弥散仇恨冲突 | 本轮不动，记为 limitation |
| 6 | "证据不进内容表示"说法不成立 | 论文改为"证据不进 value 投影和初始内容嵌入，但决定聚合、校准与最终预测" |
| 7 | 时间分辨率、每窗 4 帧、孤立打分 | 本轮不动，记为 limitation |

**机制改动**（代码：`src/hier_evidence_common.py`、`model.py`、`acquire.py`、`train.py`）：

- **A 细窗证据有界累积（臂 `temper_icc`，默认关）**：视频里已揭开 n 个细窗时，每个细窗发射的对数似然乘 κ_n = n_eff / n，n_eff = n / (1 + (n − 1)ρ)（相关观测的有效样本数，Kish design effect）。ρ = 训练集负例视频**已揭开**细窗裁定的组内相关 ICC(1)（负例上所有为 1 的裁定都是误报，只读视频标签；不平衡设计；每次 HMM 重拟合时重估，只用已计入训练预算的裁定，规则 6 复核指出首版读了负例全部 30 条裁定，已改；`hc.fine_verdict_icc`）：每视频 4 个已揭开窗时的估计与全 30 条一致（HateMM .577 对 .568，HCS .739 对 .683）。κ 同时作用于 ℓ_fine、块标签 P(h_j) 和 EOC 用的 HMM 状态；EM 拟合不变（边际发射率不受相关影响）。
- **B 三个融合标量（默认）**：先验项 α·E_t 改为 (a_f·ℓ_fine + a_x·x_t + a_v·v) / ELL_SCALE，a_f、a_x、a_v 是可学习参数，都初始化为搜索到的 α（`prior_mode = split`，臂 `single_prior` = 旧）。初始化时与旧项数值一致（差 1e-7，已验证）。scaffold 新增第 8 列 `COL_V`（v，视频内常数），`COL_TEXT` 仍是 x_t。方法级超参仍是 α、λ_block；三个标量是参数不是超参。
- **C EOC 裁定概率（默认）**：P(VLM 答 1) = r_f + (q_f − r_f)·p_w，p_w = 内容头在窗 w 的 top-k sigmoid，q_f、r_f 取当前 HMM 的细窗发射参数（`eoc_weight = model_cal`）。
- **D 分配**：`acq_alloc = per_video`（第 3 轮方式）。

**门 T5（不训练，离线分析，用缓存里负例全部 30 条裁定估 ρ，不作训练预算主张；`temper_eval.py`，`runs/20260910_online_query_within_it5/hmm_only/<corpus>/summary.json`）**：分解证据 E_t 直接当分数，test，4 粗块 + {0, 4 均匀, 8 均匀, 30} 细窗，ρ = 0 对 ρ = ICC：

| 语料 | 细窗 | ρ = 0 AP / ROC / within | ρ = ICC AP / ROC / within |
|---|---|---|---|
| HateMM | 0 | .5907 / .8578 / .6237 | 同 |
| HateMM | 4 | .5849 / .8585 / .6149 | .5881 / .8600 / .6282 |
| HateMM | 8 | .5971 / .8590 / .6253 | .5940 / .8597 / .6271 |
| HateMM | 30 | .5968 / .8583 / .6237 | .5925 / .8599 / .6272 |
| HCS | 0 | .6904 / .6573 / .5595 | 同 |
| HCS | 4 | .6958 / .6690 / .5762 | .6940 / .6630 / .5649 |
| HCS | 8 | .6981 / .6770 / .5789 | .6928 / .6628 / .5622 |
| HCS | 30 | .7146 / .6918 / .6015 | .6944 / .6629 / .5810 |

读法：HateMM 上有界累积在 4 个细窗处 within +.013、AP +.003，30 个处 AP −.004；HCS 上 4 个细窗处 ROC −.006 / within −.011，30 个处 AP −.020 / ROC −.029：HCS 的正例细窗裁定带真实的定位信息，可交换相关假设（从 32 个负例估的 ρ = .68）在它上面不成立。另外 HMM 层面 HateMM 并没有"细窗越多越差"（30 个 ≥ 8 个 ≥ 0 个），4b 训练模型在 12 次以后掉分不是证据叠加造成的，更像训练分布（最多 4 个细窗）之外的输入。**判定：A 不进默认，作臂**；审稿人的"最可能补回 HateMM AP"的建议在证据层面不成立，如实记录。

**预注册（2026-09-17，搜索前写定）**：
- 搜索同第 2 节，每 seed 首个 trial 用 4b 同 seed best trial 的五个超参热启动（`search.py --enqueue-json`），预算 HateMM 12 trial / HCS 15 trial（`budget.json` 预写；依据：第 3、4b 轮 HateMM best trial 在前 8 个以内，HCS 在前 18 个以内且单 trial 只要 15 分钟；用户要求减少 trial）。
- 判定 P1 / W1 / E1 同第 2 节。相对 4b（HateMM .6448 / .8624 / .6420，HCS .6893 / .6809 / .5802）：预期 HateMM pooled 不低于 4b − std，within 可能回到第 3 轮水平（.642，跨视频分配去掉）；HCS within 预期回落到第 3 轮的 .565 附近（分配去掉），pooled 不低于 4b − std。
- 可证伪：(1) 三个标量学出来 a_f < a_v（模型压低细窗、保留视频级项），否则"独立加权"没有起作用；(2) 34 次点不低于 8 次点 − .005 两语料（4b HateMM 34 次 .6423 < 8 次 .6448）。
- 主搜索完成后三 seed 补跑臂（各 seed best trial 超参）：`temper_icc`、`text_prior_off`、`single_prior`。判定按规则 14(g)（两语料 ≥ .01）。
- 2026-09-17 19:40 补充（三 seed 主搜索出来后加，目的是把 it5 对 4b 的差归到具体改动）：再跑 `eoc_model_raw`（EOC 权重回到内容头概率，其余同默认）和 `global_alloc`（`acq_alloc = global`，即 4b 的跨视频分配，其余同默认；每视频训练上限随之回到 8）。两者同样用各 seed best trial 超参，与前三个臂同表报告。
- 2026-09-17 19:30 运行记录：本机 GPU 被其他用户进程占用（27.6 GB），HateMM seed 234 的 trial 10、11 因显存不足失败；该 study 目录同步到 uoa-lab3，`budget.json` 改 14（失败 trial 计入预算）续跑 2 个有效 trial，之后在同机跑它的五个臂（`launch/run_it5_chain.sh`）。HateMM seed 2025/3407 的两个补充臂在 uoa-lab3，HCS 三 seed 的补充臂在 uoa-lab1（`launch/run_it5_arms.sh`）。
- 机器：本机 HateMM seed 234 + HCS 三 seed；uoa-lab1 HateMM seed 2025；uoa-lab3 HateMM seed 3407。规则 6 代码复核见 `REVIEW_RULE6_5.md`。
