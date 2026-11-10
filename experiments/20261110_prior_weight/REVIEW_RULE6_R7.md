**结论:PASS。没有发现必须修复的 bug,可以按 README 第 8 节直接开跑完整搜索。** 有 4 条不阻断的备注(第 8 节),其中备注 1(DeHate 单 trial 耗时离规则 7 的 1 小时线很近)需要主 agent 在第 1 个 DeHate trial 结束后核对。

# 第 7 版(revision 7)规则 6 代码审查

- 审查日期:2026-11-10。审查人:独立 agent,一次。
- 审查的代码版本:提交 "all-trials stage A: validation table with the anchored no-hate state"(2026-11-10 20:37 +1300),工作区无未提交修改。该提交相对 "revision 7 candidate: README (stage A evidence, decision, proposal, search protocol), search entry and launcher, per-budget table"(20:35)只改了 `stage_a_all_trials.py`,`src/qtl` 与搜索入口没有变化。
- 对比基线:提交 "answer likelihood: stage A rejects 11 answer tables; the 1-call gap is the over-confident video prior (README); STATUS"(2026-11-10 17:41)。本轮 `src/qtl` 改动 4 个文件:`train.py`、`cpolicy.py`、`prior_weight.py`(新增)、`checkpoint.py`。
- 范围:只查会改变实验观察或结论的 bug(规则 6)。没有跑训练、缩短训练或 smoke(规则 7)。本机 `hier_evidence_common.load_fixed_cohort` 不能执行,依赖数据的代码靠阅读;纯函数用 CPU 上的小规模合成输入核对(第 9 节)。没有计算或记录哈希。

## 1. c 的拟合是否不用标签、split 是否干净

是。

- 每个 epoch 结束后:`src/qtl/train.py:446` `pweight.collect(model, store, ids["val"], device)`,`train.py:447` `pweight.fit(pri, store, ids["val"], qam, chain, answers, device, mode="loc")`。传进去的只有 validation 视频 id、这些视频的先验网络输出、缓存回答、答案模型和 chain。`fit` 的参数表里没有 `labels`、`gt`(`src/qtl/prior_weight.py:147`)。
- `_chunks`(`prior_weight.py:59-96`)只读 `pri[v]`、`store.T[v]`、`answers[v]`,v 只来自传入的 `vids`(= `ids["val"]`)。`_nll_all`(`prior_weight.py:123-131`)只用这些张量。
- 答案模型 `qam` 就是 `am`(`train.py:287`;`answer_offset` 为 False,`eval_table` 为 None):`anchored_shared` 表在训练视频的回答和训练视频标签上拟合一次后固定(`train.py:257-263`、`276-277`)。chain 只在训练 loss 里更新;拟合 c 时经 `ctree._DoubleChain` 取 detach 后的值(`prior_weight.py:157`,`src/qtl/ctree.py:285-289`)。合成核对确认拟合后 chain 和答案模型上没有梯度。
- `prior_weight.label_reference`(用标签的诊断,`prior_weight.py:235-244`)和 `answer_calib.fit_validation_answer_model`(用 validation 逐秒标注,`train.py:293-297`)在搜索配置下都不会被调用:前者 `train.py` 里没有引用,后者只在 `eval_table == "val3"` 时进入。
- test:`train()` 里 test 视频在最终 `evaluate` 之前只用于建特征仓库和两条完整性断言(`train.py:220-225`),这部分是第 6 版原有代码。c、checkpoint 选择、阈值校准(`train.py:568`,validation)都不读 test。Optuna 目标用 test 指标是规则 7/10 规定的,README 第 8 节已写明。
- soft 回答的分档边界只用训练 split 的 p 值(`src/qtl/data.py:149-154`),第 5 版原有代码,本轮未改。

## 2. 被最大化的是否是模型自己的回答边际似然;A3 是否与策略一致

是。

- `_nll_all`(`prior_weight.py:127-130`):`ctree.log_evidence(fo, c·S, a·G+b, A3, chain_d, c·PHI)` 返回 `log P(G=1|x)`、`log P(G=0|x)`、`log P(o|G=1,x)`、`log P(o|G=0,x)`(`ctree.py:210-217`),目标是 `-Σ_video logaddexp(l1+lo1, l0+lo0)`,即 `-Σ log[P(G=1)P(o|G=1) + P(G=0)P(o|G=0)]`。
- zero-inflated chain:`ctree.up` 在 `chain.zero_inflated` 下返回 `g + V1(回答) − V1(无回答)` 和状态 0 下的回答似然(`ctree.py:140-142`);两个 V1 用同一份缩放后的 s 和 phi,所以无回答的归一化项也随 c 变。节点势与逐秒 logit 用同一个 c(`prior_weight.py:127-128`)。`_DoubleChain` 带着 `zero_inflated`、`closed`(`ctree.py:278-283`)。
- 推理端用的是同一个模型:`cpolicy.run_batch` 把 S、PHI 乘 c(`src/qtl/cpolicy.py:287-290`)后交给 `ctree.marginals`(`cpolicy.py:329`),`marginals` 内部同样调 `up` / `_up`(`ctree.py:229-230`)。
- A3 与策略一致:
  - 节点编号:`qdata.observed` 返回 `qtree.tree(T)` 编号(`data.py:196-209`);`_chunks` 写到 `nid + fo.offs[b]`(`prior_weight.py:88`);策略写到 `fo.offs[b] + node`(`cpolicy.py:501`)。两边的 `ctree.Forest` 都由 `qtree.tree(T)` 逐视频拼接(`ctree.py:86-88`)。
  - 节点长度:`observed` 给 `b − a`;策略的 `Asker` 用 `vt.length[q] = tr["b"] − tr["a"]`(`cpolicy.py:22-23`,`src/qtl/policy.py:52`)。两边都经 `AnswerModel.level_logp` 的同一个长度标准化。
  - 答案模型与状态展开:`am.loglik`(`src/qtl/qtree.py:121-129`,两状态时状态 1 取状态 0 的行)对应 `Asker` 的 `lp[:, :, [0, 0, 1]]` 加 `outcome_table`(`policy.py:53-61`)。
  - 没有回答的节点行为 0,与 `ctree.up` 的约定一致。
- 合成核对 A:`_nll_all` 与逐条枚举 y 的暴力计算(闭合 zero-inflated chain、节点势、两状态带长度项的答案表)在 c = 1 / .37 / 2.2 下相差 < 1e-13。合成核对 B:`_chunks` 的 A3 行与 `Asker.loglik` 逐行相等(最大差 0)。

## 3. 拟合出的 c 是否进入分数、checkpoint 选择、存盘和 Optuna 目标

是。

- 每 epoch 的 validation:`train.py:448-449` `ev.run(..., prior_weight=pweight.as_tuple(pw), priors=pri)` → `Evaluator.run` 的 chain 分支(`train.py:162-167`)→ `cpolicy.run_batch(prior_weight=, priors=)`。选 checkpoint 的 `crit` 由这次运行算出(`train.py:452-453`)。
- 选中的 epoch 的 c:`best` 只在 `crit` 变高时更新,同时存下当轮的 `pw`(`train.py:459-462`)。最终评测用 `best["prior_weight"]`(`train.py:479-480`),不是最后一个 epoch 的。`pw` 每轮新建,没有被后续轮次改写的引用。
- 存盘:`saved["prior_weight"] = best["prior_weight"]`(`train.py:472-473`),`summary["prior_weight"]`(`train.py:481`),每 epoch 的值在 `history`(`train.py:454-455`)。
- 最终评测:`evaluate` 的 validation 和 test 两次运行都带 `prior_weight`(`train.py:534-537`);固定预算 0-32 都取自这两条轨迹(`train.py:550-562`);阈值规则在 validation 轨迹上校准、在 test 轨迹上取分(`train.py:567-581`);`summary["test"]` 和 `metrics.json` 取固定 8 次的 test(`train.py:582`、`595-599`)。`src/qtl/search.py:91-100` 读 `summary["test"]` 和 `summary["fixed"]["8"]["val"]`,目标 = (test AP + test ROC) / 2。
- `priors=`:`collect` 调的就是 `run_batch` 自己会调的 `policy.video_prior_nodes`(五个 crop 的均值,`policy.py:26-40`),先切 eval 模式、结束后恢复 train 模式(`prior_weight.py:48-56`);`run_batch` 在 `priors` 给出时直接取 `priors[v]`(`cpolicy.py:278`)。最终评测不传 `priors`,由 `Evaluator.run` 在 eval 模式下重新前向(`train.py:146`),此时模型已载入选中 epoch 的权重(`train.py:463-466`),与拟合 c 时的输出相同(GPU 浮点误差之内)。合成核对:`Evaluator.run(prior_weight, priors)` 与"自己前向 + 同一权重"的 `run_batch` 结果逐元素相同;`collect` 内部为 eval 模式、结束后为 train 模式。
- 训练 loss 未改(c 只在 validation / test 用),与 README 第 8 节一致。

## 4. `cpolicy.run_batch` 里权重是否只乘一次、乘在该乘的地方

是。

- `prior_scale` 默认 1.0,`Evaluator.run` 不传,`cpolicy.py:285-286` 不执行;权重只在 `cpolicy.py:287-290` 乘一次。
- 模式 `loc`:`fit` 里 a、b 不参加优化,保持 1 和 0(`prior_weight.py:158-159`、`175-177`),输出 `"a": 1.0, "b": 0.0`;`run_batch` 里 `G = 1·G + 0`,不变;S、PHI 乘 c。
- 缩放在第一次推理之前完成,之后所有用到先验的地方都读同一份 S、G、PHI:后验 `cpolicy.py:329`;EIG 的状态权重 `cpolicy.py:387-388`、`415-417`;停止规则的 voi `cpolicy.py:443-461`;记录的 `p_G`、`scores`。`policy_prior_scale` 为 None。
- 合成核对 C:`run_batch(prior_weight=(1, 0, .37), priors=)` 在每一步的分数和 `p_G` 与"s、phi 乘 .37、g 不变"的暴力后验相差 < 1e-14;`prior_weight=(1, 0, 1)` 与 `None` 逐元素相同。

## 5. test 是否仍走统一评测器;第 6 版路径是否不变

是。

- test:`evaluate.test_eval`(`train.py:541-546`)→ `hc.write_scores` → `hc.run_evaluator`(`src/hier_evidence_common.py:467-473`,调 `scripts/reproduction_baselines/eval_baseline_scores.py`)。validation 用 `hc.frame_metrics`(`hier_evidence_common.py:422-428`,同一核心 `frame_eval_common`)。本轮没有改动 `scripts/`、`src/hier_evidence_common.py`、cohort 与 GT 加载。
- 第 6 版路径(`prior_weight: None, plan: None, eval_table: None`):
  - `train.py`:三个新键默认 None(`train.py:88`);`train.py:290`、`293`、`445`、`472`、`475` 的分支都不进入;每 epoch 走 `train.py:451`,调用与改动前相同;`evaluate` 里 `planned` 为 False,validation / test 各跑一次 `max_calls`,`primary_runs` 与 `test_runs` 是同一对象(`train.py:555-556`、`583`)。新增的只有 `history` / `summary` 里值为 None 的 `prior_weight` 键和一条 `primary_budget ∈ fixed_budgets` 的断言。
  - `cpolicy.run_batch`:`plan is None` 时 `planned` 恒为 False(`cpolicy.py:380`),`cpolicy.py:427-430` 不执行,`cpolicy.py:465` 只在 `no_nested` 时进入且行为与原来相同。
  - 合成核对 D:现在的 `run_batch`(无权重、无 plan)与基线提交的 `run_batch` 在 150 个合成视频、32 次调用、带 voi 的设置下 `asked`、`eig`、`voi`、`p_G`、`scores` 逐元素相同;`Evaluator.run` 用不含三个新键的旧 config 也相同。
- 搜索不用的新增部分确实不生效:
  - `plan`:搜索配置不设(`experiments/20261110_prior_weight/search.py:10-11`),见上。`_plan_disjoint` 只在 `cpolicy.py:429` 被调用。
  - `eval_table`:`train.py:293`、`475` 的条件为假,`qam` 仍是 `am`。
  - trust 标量 / `TrustTable`:`train.py:288` 的断言只允许 `None / loc / joint / root_all`;`fit` 只在模式以 `trust` 开头时进入该分支(`prior_weight.py:165`);`_trust_rows` 在 `trust is None` 时原样返回 A3(`prior_weight.py:103-104`);`TrustTable`、`table` 在 `train.py`、`cpolicy.py` 中没有引用。

## 6. `checkpoint.load_prior_weight`

正确。`src/qtl/checkpoint.py:47-52` 读 `model.pth` 的 `prior_weight` 字典(`train.py:473` 存的就是 `fit` 的返回值),返回 `(a, b, c)` 浮点三元组,没有该键时返回 None;`run_batch` 的 `prior_weight` 参数接受的正是这两种(`cpolicy.py:287-290`)。目前仓库里还没有调用方。

## 7. 其它会悄悄改变搜索数字的地方

没有发现。

- 搜索入口:`experiments/20261110_prior_weight/search.py:10-11` = 第 6 版的四个固定设置加 `prior_weight: "loc"`;`train.py` 入口直接调 `qtl.train.main()`,五个键都在 `DEFAULTS` 内,不会触发未知键断言。`src/qtl/search.py` 本轮未改(四个标量、TPE、sampler seed = 训练 seed、目标定义)。
- 启动脚本与第 6 版只差输出目录和入口路径(`launch/sbatch_search.sh:25`、`27`)。
- 拟合 c 不消耗随机数(L-BFGS;`collect` 在 eval 模式下前向),所以同一 seed、同一超参数下网络的训练过程与第 6 版相同(GPU 非确定性除外);差别只来自带 c 的推理和 checkpoint 选择。TPE 前 10 个 trial 是按 sampler seed 随机采样(第 6 版三个语料同一 seed 的 trial 0-9 超参数完全相同,`runs/20261101_r6_length_table/<corpus>/seed*/study_summary.json`),第 7 版每个 seed 的 trial 0-9 的超参数因此与第 6 版同号 trial 相同,可以直接做配对比较;前提是搜索在前 10 个 trial 内没有被中断重启。
- import 检查通过:`qtl.train`、`qtl.checkpoint`、`qtl.prior_weight`、`qtl.cpolicy`、`qtl.search` 和 `experiments/20261110_prior_weight/search.py`。

## 8. 备注(不阻断)

1. **DeHate 单 trial 耗时接近规则 7 的 1 小时线。** 第 6 版 DeHate 每个 trial 3308-3407 秒,均值 3336 秒(三个 seed 同卡并行;`runs/20261101_r6_length_table/dehate/seed*/study_summary.json` 的 `user_attrs.seconds`),每 epoch 59-65 秒(`runs/20261101_r6_length_table/dehate/seed234/trial0/run.log`)。阶段 A 在 559 个 DeHate validation 视频上拟合 c 一次约 2.4-3.7 秒(`runs/20261110_prior_weight/stage_a_dehate.log`:`root_all` 2.73-4.04 秒减 `root` .33-1.07 秒,3 个 trial)。第 7 版每个 trial 多 50 次拟合,约多 120-185 秒,预计 3430-3590 秒,离 3600 秒只有几十到一百多秒;三进程并行时可能更慢。`launch/sbatch_search.sh:25` 预写了 20 个 trial,搜索不会自己量第 1 个 trial。规则 7 规定超过 1 小时的方法每个 seed 跑 5 个 trial。建议:第 1 个 DeHate trial 结束后读 `study` 里的 `seconds` 写进 README;若超过 3600 秒,按规则 7 向用户说明,不要自行决定。HateMM(727-861 秒)和 HateClipSeg(234-301 秒)不受影响。
2. **c 没有取值约束。** 阶段 A 全部 trial 的 `loc` 记录(`runs/20261110_prior_weight/all_trials/<corpus>/seed*/trial*.json`;同一 trial 的多个 variant 可能重复记录同一次拟合):HateMM 最小 .006、中位 .166、最大 1.636;HateClipSeg −.002 / .190 / 2.313;DeHate .005 / .243 / 4.802。c 接近 0 时逐秒先验和节点势基本不起作用,0 次调用的分数在视频内近似常数。这是方法本身的性质,不是 bug;README 第 8 节已计划逐 trial 报告 c,报告时建议同时给出最小值和 c < .05 的 trial 数。
3. **"不用标签"的范围。** 似然里没有任何标签。但拟合用的 validation 视频集合是评测 cohort:DeHate 的 `ids["val"]` 去掉了"仇恨但没有可用 span"的视频(`hier_evidence_common.py:373-380`),这个筛选依赖标注。论文里的表述应限定为"估计 c 时不使用视频标签和逐秒标注",不要写成 validation 集合本身与标注无关。
4. **第 1 个 trial 跑完后可做的零成本核对(读真实搜索的输出,不是试跑)。** `summary.json` 里 `fixed["8"]["val"]` 应与 `history[best_epoch − 1]["val"]` 一致(差 < 1e-4),`summary["prior_weight"]` 应等于 `history[best_epoch − 1]["prior_weight"]`,且 `a = 1`、`b = 0`;`run.log` 有一行 "prior weight (validation answers, no labels)"。这四项成立即说明选中 epoch 的 c 确实进了最终评测。

## 9. 合成核对记录

脚本在会话临时目录,未进仓库;CPU,合成的 s、g、phi、回答,模型配置同搜索(`soft_both` 一个类别 8 档、两状态带长度项的答案表、闭合 zero-inflated chain、节点势)。

| 核对 | 结果 |
|---|---|
| A:`_nll_all` 对暴力枚举的边际似然(T ≤ 12,c = 1 / .37 / 2.2) | 差 < 1e-13 |
| B:`_chunks` 的 A3 行对 `Asker.loglik` | 最大差 0 |
| C:`run_batch(prior_weight=(1, 0, .37), priors=)` 每一步的分数、`p_G` 对暴力后验 | 差 < 1e-14 |
| C:`prior_weight=(1, 0, 1)` 对 `None` | 逐元素相同 |
| D:现 `run_batch`(无权重、无 plan)对基线提交的 `run_batch`(150 视频、32 次、voi) | 逐元素相同 |
| D:`priors=` 路径对自己前向路径 | 逐元素相同 |
| E:`fit(mode="loc")` 的 c 对 `_nll_all` 的网格最小点(步长 .01) | 差 < .02;a = 1,b = 0 |
| E:拟合后 chain、答案模型上的梯度 | 无 |
| `Evaluator.run(prior_weight, priors)` 对"自己前向 + 同一权重" | 逐元素相同 |
| `collect` 的模式 | 内部 eval,结束后恢复 train |
