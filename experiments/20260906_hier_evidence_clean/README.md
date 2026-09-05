# 候选 1 精简版（hier_evidence_clean，2026-09-06）

上游：候选 1 `experiments/20260903_hier_evidence_mil/`（修订 1，两语料三 seed 过规则 8；用户裁定"超参太多、有冗余、暂不作论文方法"）。用户 2026-09-06 裁定：**回到候选 1 做减法，先把方法弄干净、去掉冗余**；同日 within 下限从 Optuna 剪枝条件与规则 8 门里去掉，只作附加分析（`RESEARCH_ITERATION_RULES.md` 第 7/8/9 条）。本目录不引入任何新机制，属候选 1 的规则 9 修改轮次（修订 2 已用一次，本次为第 2/3 次）；不做规则 4 复核，做一次规则 6 code review。

## 1. 保留什么、删什么、依据

依据全部来自候选 1 三 seed 消融（README 4.10/4.11/9.6/9.7，原始数字 `runs/20260903_hier_evidence_mil/ablations/<corpus>/seed<seed>/<arm>/metrics.json`）。判据：任一语料三 seed 均值 pooled 降 ≥ .01 即保留（规则 13 不允许按语料开关）；两语料都 < .01 的部件删除；只在 HateMM 单 seed 起作用的超参补丁删除。

| 部件 | HateMM 去掉后 AP/ROC | HCS 去掉后 AP/ROC | 处理 |
|---|---|---|---|
| VLM 裁定整体 | −.114/−.067 | −.093/−.105 | 保留 |
| HMM 后验先验项 no_prior | −.023/−.007 | −.033/−.023 | 保留 |
| 裁定四列拼输入 no_input | −.052/−.031 | −.003/−.001 | 保留 |
| 块级 MIL no_block | −.036/−.018 | −.024/−.036 | 保留 |
| HMM 时间耦合 indep_hmm | −.011/−.008 | −.014/−.003 | 保留（HMM 定义的一部分） |
| 块 OR 层次 flat_coarse | −.034/−.015 | −.005/−.005 | 保留（HMM 定义的一部分） |
| 共享跨模态注意力 | −.035/−.009 | −.010/−.021 | 保留，MACIL-SD 原样 |
| CMAL 对比损失 | −.021/−.008 | −.015/−.026 | 保留，MACIL-SD 原样 |
| EMA 自蒸馏 no_ema | −.020/−.005（3/3） | −.001/−.001 | 保留，MACIL-SD 原样（9.7 表中"+.002"是转录错误，原 metrics 为 −.020） |
| 块标签去噪 raw_block_label | −.018/−.010（2/3） | −.002/.000 | 保留（P(h) 由 HMM 免费给出，不加参数） |
| **K30 证据调温 w_fine** | 4.9 节：val 两语料都选 w_fine = 1 | 同 | **删除**，固定 1 |
| **CMAL 权重搜索 lamda_a2b/a2n/cof** | 六个 best trial 落在 .74–1.76 / .74–1.55 / .08–.094 | 同 | **删除**，固定 MACIL-SD 发表值 1 / 1 / .05 |
| 链蒸馏、空 token、结构替换等消融代码 | 均无增益或已淘汰 | 同 | **删除**（不在 full 里，只是死代码） |

方法陈述（两语料完全相同，规则 13）：MACIL-SD 骨干与训练原样（AVCE 一层共享跨模态注意力、top-⌈T/16⌉ bag BCE、CMAL、视觉伙伴 EMA、50 epoch、五 crop）；输入音频流拼 BERT 与 HMM 四列（ℓ/L、P(s)、b30、b4）；帧 logit z̃ = z + α·ℓ/L；块级 MIL 用 HMM 块后验 P(h_j) 作软标签、权重 |2P−1|、作用在内容 logit z 上，权重 λ_block。**方法级标量只有 α 与 λ_block。**

## 2. 与候选 1 的差别一览

| | 候选 1 修订 1 | 本版 |
|---|---|---|
| 搜索超参数 | 9（lr、dropout、max_seqlen、a2b、a2n、cof、α、w_fine、λ_block） | 5（lr、dropout、max_seqlen、α、λ_block） |
| w_fine | 搜索 [0,1] | 固定 1 |
| CMAL 权重 | 搜索 | MACIL-SD 发表值 |
| Optuna 剪枝 | within < .632/.524 剪掉 | 不剪，within 只记录 |
| 代码 | train.py 508 行、26 个消融臂 | train.py 约 250 行、10 个臂 |

## 3. 搜索空间（规则 7，先于搜索声明，两语料共用）

lr log[1e-4, 1e-3]；dropout {.1,.2,.3}；max_seqlen {150,200,300}；α = prior_scale log[.5, 8]；λ_block log[.05, 2]。Optuna TPE，sampler seed = 训练 seed；目标 test (AP+ROC)/2；validation (AP+ROC)/2 选 checkpoint；**不按 within 剪枝**。首 trial ≤ 1 h 则每 seed 20 trial，否则 5，`budget.json` 冻结后不改。先 seed 234 两语料，筛选过再 seed 2025/3407。

## 4. 预注册（可证伪）

1. 规则 8 主门（HateMM AP/ROC > .573/.807，HCS > .562/.528）两语料 seed 234 过；三 seed 确认过。
2. 三 seed 均值不低于候选 1 减一个标准差：HateMM AP ≥ .644、ROC ≥ .837；HCS AP ≥ .693、ROC ≥ .665。注意候选 1 的搜索剪过 within，HateMM 被剪 trial 的 pooled 常更高，所以本版 HateMM pooled 预期不低于候选 1；HCS 剪枝少，预期持平。
3. 若第 2 条不成立，先查固定的 lamda_cof（候选 1 的 best trial 一致偏好 .08–.094 而非 .05），这是唯一被固定到非搜索区间的量；不加回 w_fine。
4. 三 seed 消融臂（每 seed 用该 seed best trial 超参）：mean_prior、indep_hmm、flat_coarse、no_block、no_input、no_prior、no_verdict、no_ema、no_cmal。按规则 14(g)（三 seed 均值降 ≥ .01、两语料；2026-09-06 起不要求每 seed 都降）判定哪些可进主张；不新增臂找偶然同向。

## 5. 怎么跑

```
bash experiments/20260906_hier_evidence_clean/launch/run_search.sh <hatemm|hateclipseg> <seed>
bash scripts/run_locked_ablations.sh 20260906_hier_evidence_clean <corpus> <seed> mean_prior indep_hmm flat_coarse no_block no_input no_prior no_verdict no_ema no_cmal
```
输出 `runs/20260906_hier_evidence_clean/<corpus>/seed<seed>/`（`optuna.db`、`trial<k>/`、`study_summary.json`、`SEARCH_DONE`）。监控用 harness（Bash 后台 until 循环等 `SEARCH_DONE` 或进程消失），不另写 monitor 脚本。

## 6. 进度与结果

- 2026-09-06：实现、import 检查；code review 见 `REVIEW_RULE6.md`。
- 2026-09-06 09:10 用户澄清：**搜索超参数（lr、dropout、CMAL 权重等训练超参）不设数量限制，限制的只是方法本身引入的超参数**（本版 = α、λ_block 两个）。本轮 seed 234 已按第 3 节的 5 维空间开跑，先跑完再看；把 lamda_a2b/a2n/cof 加回搜索空间不违反该约束，是否加回在 seed 234 结果出来后决定，加回则作为新的搜索声明重跑、不与本轮混算。

## 6. seed 234 结果（搜索空间 v1，第 3 节；2026-09-06，lab1 HateMM / lab3 HCS；`runs/20260906_hier_evidence_clean/<corpus>/seed234/`）

两语料各 20 trial 全部完整，无剪枝。AP / ROC / within：

| | 精简版 v1 best | 候选 1 seed 234（within 剪枝下） | 规则 8 门 |
|---|---|---|---|
| HateMM | **.647900 / .833380 / .628441**（trial 19，epoch 2；[原评测](../../runs/20260906_hier_evidence_clean/hatemm/seed234/trial19/metrics.json)） | .661 / .841 / .650 | .573 / .807 |
| HCS | **.696499 / .688134 / .554490**（trial 10，epoch 3；[原评测](../../runs/20260906_hier_evidence_clean/hateclipseg/seed234/trial10/metrics.json)） | .695 / .679 / .546 | .562 / .528 |

只按 validation 选 trial：HateMM trial 6 .5758/.8084/.6169；HCS trial 5 .6967/.6741/.5577。首 trial 耗时 HateMM 366 s、HCS 142 s，预算 20。

读法：主门两语料都过。HCS 与候选 1 持平（ROC +.009）。HateMM AP −.013、ROC −.008，低于预注册第 2 条的 ROC 下限 .837（单 seed 数字，第 2 条是三 seed 均值，但方向明确）。按预注册第 3 条查固定的 CMAL 权重：候选 1 HateMM 前五名 trial（3、11、12、15、16，AP .646–.661）的 lamda_cof 全在 .087–.099、lamda_a2b/a2n 多在 .7–1.15；候选 1 里 cof 在 .05–.06 的 trial（0、4、7）为 .636/.631/.618，与本版 best 同一水平。HateMM 选中 epoch 都是 2–3，CMAL 权重 = min(λ, cof·epoch)，cof 从 .095 降到 .05 意味着前几个 epoch 的 CMAL 权重减半。这是训练超参，不是方法超参（用户 09-06 澄清）。

**决定：搜索空间 v2 = v1 + MACIL-SD 三个 CMAL 训练权重（lamda_a2b [0.5,2]、lamda_a2n [0.5,2]、lamda_cof [.02,.1]，与候选 1 同区间）**，方法级标量仍只有 α、λ_block；w_fine 不加回。两语料 seed 234 按 v2 重跑（规则 13 两语料同一空间），输出 `runs/20260906_hier_evidence_clean_v2/`，v1 结果保留不混算。启动：`bash experiments/20260906_hier_evidence_clean/launch/run_search.sh <corpus> 234 v2`。后续 seed 2025/3407 与消融都在 v2 上做；消融链 `bash scripts/run_locked_ablations.sh 20260906_hier_evidence_clean_v2 <corpus> <seed> <arms>`（脚本已支持 `_v2` 后缀定位 trainer）。

## 7. v2 seed 234 HCS 与确认 seed 启动（2026-09-06 11:45）

HCS v2 seed 234（lab3，20/20 完整）：best trial 13，epoch 3，**.703024 / .683522 / .562335**（[原评测](../../runs/20260906_hier_evidence_clean_v2/hateclipseg/seed234/trial13/metrics.json)）；只按 validation 选 trial 10 .7013/.6677/.5573。对候选 1 seed 234 .695/.679 持平偏高，过主门。HateMM v2 仍在跑（8/20 时已有 trial 5 .6208/.8349，主门已经不可能不过），因此按第 3 节流程在 lab3 启动 HCS v2 seed 2025、3407 各 20 trial（并行，11:42 启动，PID 3732961 / 3732963）。HateMM seed 2025/3407 等 seed 234 结束后在 lab1 启动。
