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
| 0 | 初版（1 节 A–C、E；D 关）。规则 4 复核 `REVIEW_RULE4.md`：PASS（措辞：训练期主动获取 VLM 窗裁定，先例 Melville 2004 / Saar-Tsechansky 2009 + EMOC 2014 / DIME 2024；B 不单独主张；C 写成 masked verdict prediction，先例 UniMP 2021 / Rodrigues 2018；D 写成 Dawid–Skene 式档混合的负结果）。规则 6 复核 `REVIEW_RULE6.md`：一处 must-fix（允许集合为空时 HMM 初始化把 r_f 置成 .001 而非保留初值，已改：无观测的族保留初值；R = 1 有观测时与旧代码数值完全一致）；should-fix 已做：混合版发射参数裁到 [1e-4, 1 − 1e-4]。2026-09-10 开跑：HateMM 本机（GPU 空闲），HCS uoa-lab3。 | 待 seed 234 两语料 | |
