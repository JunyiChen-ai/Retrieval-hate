# Backbone 候选：时间树上的区间内外表示

截至 2026-10-01；状态：实现完成，[一次独立 proposal review：GO](../../docs/reviews/20261001_inside_outside_backbone_proposal.md)，[一次独立 code review：PASS](../../docs/reviews/20261001_inside_outside_backbone_code.md)。用户最新授权自动迭代：优先做 novel 且涨点的 backbone；充分尝试仍无提升时优先可验证的新意，完成后汇报。当前旧骨干实现已升入 `src/qtl/model.py`，旧实验文件保留兼容入口。within 按 AGENTS.md 最新裁定与 pooled AP / ROC 并列为主指标；不改写研究规则文件。

## 1. 需要解决什么

现有内容编码器是 MACIL-SD 的跨时间、跨模态 attention；第 4 版节点势仍然只是对这批逐秒特征做区间 attention pooling。问题树影响了训练与输出，但尚未成为表征学习的结构。

第 4 版默认超参、三 seed 对照（固定 8 次，AP / ROC / within）：

| 数据集 | 完整模型 | 无内容骨干 | 无骨干的 within 变化 |
|---|---|---|---|
| HateMM | .6322 / .8783 / .6887 | .6008 / .8617 / .7155 | +.0268 |
| HateClipSeg | .6382 / .6483 / .5495 | .6846 / .6425 / .6075 | +.0580 |
| DeHate（外部验证） | .1820 / .7398 / .6230 | .1621 / .7303 / .6676 | +.0446 |

来源：`runs/20260927_query_paradigm_r4/diag/<corpus>/seed{234,2025,3407}/{abl_full,abl_noback}/metrics_test_fixed8.json`，从 `results.score_av` 汇总。它们是旧版、默认超参的诊断证据，不是第 5 版结论，也不能与搜索最优数字混比；移除骨干还同时移除节点势，不能据此单独归因于 attention。它们支持“应检查内容表示是否损害视频内区分”的假设，尚不证明新结构有效。

## 2. 具体机制

将内容 backbone 改为当前二分时间树上的 inside–outside encoder。它计算表示；已有 OR 树/链计算标签后验，两者职责不同。

1. **叶子**：同一份 I3D 与 VGGish+BERT 缓存，投影成视觉与音频文本两路逐秒向量 `i_t^m = P_m x_t^m`。叶子输入不先经过跨时间 attention，以保证下面的 outside 不读取目标区间的缓存行。
2. **自下而上（inside）**：每路在父节点用共享的、有左右顺序的门控组合器 `i_n^m = F(i_L^m, i_R^m)` 学习区间内容。不同尺度共享参数；并非复用原节点头的加权平均。
3. **自上而下（outside）**：根的 outside 为可学习常量；`o_L^m = G_L(o_n^m, i_R^m)`、`o_R^m = G_R(o_n^m, i_L^m)`。因此某区间的 outside 仅从区间外的缓存行计算，不把包含自己的整段向量重复广播回来。这里的“区间外”不等于“无害背景”，其中可以同样有仇恨。排除只保证缓存行索引上的计算依赖：I3D 感受野、BERT 行本身可能跨区间，不能声称原始媒体级隔离，也不要求两种表示统计独立。
4. **局部判断**：共享融合器读取本区间 inside、outside、二者差/乘积及另一模态对应向量，生成两路 `r_n^m`。从叶子表示输出逐秒先验 `s_t`，从内部节点表示输出 `phi_n`，根 inside 输出视频头 `g`。两路叶子表示继续供已有 CMAL 使用，避免把删损失混进新结构。
5. **训练与推断接口**：仍用视频标签和已有 VLM 答案的树边际似然；骨干不读已问/未问答案，VLM 观测仍只经原答案模型进入最终后验。先在第 5 版 8 档、固定 8 次预算上隔离骨干效果，再检查既有停止设置；不在同一比较里同时挑新停止规则。

假设：模型能结合“这一小段是什么”和“同一视频其他部分是什么”形成可区分的表示，降低整段语义对局部判断的干扰。是否改善 pooled 与 within 必须由实验检验；不承诺三个指标都会升。

## 3. Novelty 的准确范围与来源

- inside–outside 表示来自 [Le & Zuidema, EMNLP 2014](https://aclanthology.org/D14-1081/)，与 [DIORA, NAACL 2019](https://aclanthology.org/N19-1116/) 的内外表示有联系；有序组合还需对照 [Tree-LSTM, ACL 2015](https://aclanthology.org/P15-1150/)。不声称首次发明树网络或 inside–outside。
- 本候选主张范围是：在 hateful video localization 中，以多尺度问题树组织内容表示，让区间内外比较产生秒级/节点级先验，并由同一带噪问题树似然训练。是否已有 hateful-video 方法用过该来源机制，由独立 reviewer 实际检索。
- 独立检索未找到该来源机制的 hateful-video 直接应用，按规则 4 放行；这不证明世界首次。[RAMF](https://arxiv.org/html/2512.02743v2)、[MultiHateGNN](https://arxiv.org/html/2509.13515v1)、[CLARA](https://arxiv.org/html/2608.15905v1) 已有局部/全局融合或结构化表示，故这些宽泛描述不能作为本候选 novelty；具体方法差异见独立评审。
- 与当前节点势的区别在 encoder 的计算图：旧版先全局 attention、再池化；候选先区间组合，再从不含自己的外部上下文形成局部表示。与答案传播的区别：传播内容表示，不沿骨干相似度扩散 VLM 答案；此前答案传播负结果不能当作本候选已失败或已成功的证据。

## 4. 如何检验主张

依规则做一次独立 code review，再做两主数据完整训练；搜索开始前固定协议见第 6 节。需要回答的比较：

- 第 5 版原骨干 vs 新骨干，使用相同输入、答案缓存、融合、预算和训练目标；所有三个主指标并列报告。
- 新骨干移除 outside，检验区间外上下文是否有贡献。
- 学习的有序组合替换为池化，检验结构是否只相当于多尺度 pooling。
- 保留相同投影/融合容量的逐秒编码器对照，避免仅把参数量差异归为结构贡献。
- 同时报 0 次与 8 次调用：0 次检查内容先验自身，8 次检查完整方法。最终结构贡献不能只靠视频级分类改善，within 必须纳入主要结论。

## 5. 成本与执行状态

新增 VLM 调用 **0 次**，新增特征/转录抽取 **0 次**；已有缓存全部复用。处理新视频时仍按原策略调用 VLM，不增加一次提问的视图数。新成本仅是可训练骨干前向/反向；固定二叉树只有约 `2T` 个节点，组合部分约 `O(T h²)`，但 CUDA 调度和常数未测，不能据此宣称比现有 attention 更快。

参考耗时来自第 5 版 seed234 的 `study_summary.json`：HateMM 单 trial 约 739 秒，HCS 约 354 秒，均含完整训练与评测。当作占卡墙钟预算粗估，两语料各 20 trial、单 seed 顺序累计约 **6.1 GPU 小时**，三 seed 约 **18.2 GPU 小时**；原运行有并发资源争用，这不是独占 GPU 的实测耗时或新架构速度承诺。新架构耗时须以首个完整 trial 实测修正。新增 VLM 抽取 GPU 时间为 0。

proposal review 只决定候选是否值得实现，不等于 novelty 贡献或性能已经成立。运行主机为 **uoa-lab3 / sc474398**；2026-10-01 20:57 NZDT 已启动两主数据 seed234 完整搜索，独立并行，满足筛选后再安排其余 seed。lab1 当前他人 VLLM 约占 28 GiB，共享服务器仅约 2.5 GiB 空闲；不干扰其任务。本机沿用此前不跑训练的安排。资源变化时重新调度。

## 6. 第一轮训练与搜索协议（开跑前固定，2026-10-01）

- 输入与监督：第 5 版 8 档单问软答案；I3D / VGGish / BERT；视频标签与全部训练节点缓存答案的树边际似然。无逐秒标签进梯度，train/val/test 不混合。共享逻辑为 `src/qtl/`，统一评测器未修改。
- 唯一结构改动：`backbone.py` 的 inside–outside 编码器替换原 CMA 编码器及对应先验头；零膨胀链、答案模型、CMAL 损失、EIG 与固定 8 次主比较点沿用第 5 版。inside 与 outside 的组合器、局部融合器跨深度与模态共享；左右 outside 组合器各有参数。`io_outside=true`、`io_merge=gated`，消融才关闭或换均值。
- 两语料同一空间：Optuna TPE，sampler seed=训练 seed。`lr` log-uniform [1e-4,1e-3]，`lamda_cma` uniform [.5,2]，`dropout` uniform [.1,.5]，`lr_answer` log-uniform [1e-4,.1]。其它固定：hidden=128，batch=32，50 epoch，crop_repeat=5，long_T=512，沿用原长度分批规则。
- 每个 trial 完整训练；validation AP / ROC 均值选 checkpoint，立即 test；Optuna 继续沿用规则 7 的 `(test AP + test ROC)/2` 选择 trial，以便和第 5 版比较，同时记录 validation 选 trial 的三项 test 数字。**within 作为主指标参与方向判断和结论**，但本次未擅自更改既定搜索标量或恢复 within 剪枝。
- trial 数由首个完整 trial 实测决定：≤1h 为20，>1h为5，写 `budget.json` 后锁定；不做 smoke，不预写20规避实测。完整预算全部结束后才判断该 seed；实现错误先修复，不评价方法。
- 搜索目录：`runs/20261001_inside_outside_backbone/<corpus>/seed<seed>/`。每个 trial 有 config、可读代码版本说明、日志、checkpoint、统一评测器 metrics；run owner 另写进程身份与 `completion.json`，只有完整预算和输出核验通过才记 success。
- 性能优先：对第 5 版相同预算的三 seed 结果同时看 AP / ROC / within 的变化和 seed 方差；不以任意加权总分掩盖某个主指标退化。筛选/确认仍按研究规则执行，随后用误差分析选择修改；新颖但无提升的配置不能仅凭 proposal GO 宣告完成，仍需完成结构贡献验证并如实报告代价。
- 计划消融：原骨干、去 outside、有序组合改递归均值、保留同样叶子融合的逐秒/池化对照；用锁定配置三 seed，并报 0/8 次。最终按规则 14(g) 核对可主张贡献，同时将 within 作为主要证据。
- 启动入口：`launch/run_{hatemm,hateclipseg}_uoa-lab3.sh <seed>`，必须用 setsid/nohup。共享 `scripts/monitor_run.py` 独立后台运行，每120秒核对主进程身份、同进程组子进程、主机和目录；完成/确认异常后 `codex queue` 通知当前会话，SSH失败只重试。

## 7. 当前运行记录

- 初版实现已通过一次独立 proposal / code review，启动时本机与 lab3 代码一致、跟踪文件干净；不在运行中修改代码。运行日志记录可读代码版本说明与主机。
- 输入检查全量通过；日志在本机 `runs/20261001_inside_outside_backbone/setup/inputs_{hatemm,hateclipseg}.log`，覆盖 HateMM 1067 个、HCS 393 个视频，各自 split 不重叠。
- 搜索目录为 `runs/20261001_inside_outside_backbone/{hatemm,hateclipseg}/seed234/`。首 trial 实测 HateMM 559.52 秒、HCS 235.49 秒，各自自动锁定 **20 trial**，依据对应 `budget.json`。HCS 已完成全部 20 个，HateMM 继续原定搜索。
- 每个搜索已配置独立后台完成通知，当前会话另复用每 3 小时 heartbeat；Pursuing goal 保持暂停。具体运行 PID 与 monitor 路径只在 STATUS 维护。

## 8. HateClipSeg seed234 完整结果与结构诊断（2026-10-01）

22:14 NZDT 在 sc474398 完成 20/20 COMPLETE。22:16 完成通知唤醒后核验真实主/子进程均已结束，将全部输出回传本机；逐 trial 检查完整 50 epoch 配置、全部固定预算评测可解析、79 个 test 视频 / 18839 秒覆盖及三项指标有限。所有数字仍以本机统一评测器文件为准。

| 配置与选择 | AP | ROC | within |
|---|---:|---:|---:|
| 新骨干 seed234，test 选 trial11 | .668767 | .635762 | .567188 |
| 第 5 版原骨干 seed234，test 选 trial13 | .688467 | .674845 | .556131 |
| 差值 | −.019700 | −.039083 | +.011057 |
| 新骨干 seed234，仅 validation 选 trial6 | .648418 | .616496 | .571958 |

来源：`runs/20261001_inside_outside_backbone/hateclipseg/seed234/trial{11,6}/metrics_test_fixed8.json`、`runs/20260929_query_paradigm_r5/hateclipseg/seed234/trial13/metrics_test_fixed8.json`；trial 选择依据各自 `study_summary.json`。新骨干尚未涨点；虽 HCS pooled 超过固定 baseline 门，HateMM 未完成，不能宣告两语料筛选、三 seed 确认或 novelty 验证通过。

误差分析读取上述最优 trial 的固定 0 / 8 / 32 次统一评测输出，以及两方法前 10 个相同超参的 trial 评测。0 次新骨干 .668739 / .644563 / .529660，旧骨干 .669585 / .664387 / .542849；0 次 mean-score 视频 AUC 为 .723188 vs .846377。退化已出现在内容先验及视频间区分，不能只归因于提问策略；根组合压缩、outside 干扰均只是待检验解释。前 10 个相同超参 trial 的 AP / ROC 差值全部为负，平均 −.025793 / −.043585；within 平均 −.008364。最优 trial 的 within 改善并未普遍出现。原始逐视频 within 有 32 个改善、35 个下降，不能把单 seed 均值改善写成稳定收益。派生分析记录 `runs/20261001_inside_outside_backbone/analysis/hateclipseg_seed234_comparison.json`，字段保留原始来源；这属于已查看 test 的开发证据。

**诊断协议，新增训练前固定：**沿用第 4 / 6 节预设消融，在 HCS 完整搜索已锁定的 trial11 配置上做 `full`、`nooutside`（只设 `io_outside=false`）、`mean`（只设 `io_merge=mean`）。三者各 seed234 / 2025 / 3407，完整 50 epoch，validation 选 checkpoint，立即经统一评测器 test，报告 0 / 8 / 32 次与 seed 方差；不重搜、不改其它超参。full seed234 直接复用现成 trial11，其余 **8 次完整训练**。配置预存 `configs/diagnostic_hateclipseg_seed234.json`，启动入口 `launch/run_diag_hateclipseg_uoa-lab3.sh <arm>`，输出 `runs/20261001_inside_outside_backbone/diagnostics/hateclipseg/<arm>/seed<seed>/`。每个 arm 独立后台 owner + monitor。

这些是固定配置的结构诊断，不是规则 8 的确认级 Optuna；不把它们当成三 seed 搜索最优，也不据此宣告两语料机制贡献。只检验已实现且已评审的开关，HateMM 在跑代码保持不变。新增 VLM 调用 0；按当前 HCS 单 trial 235 秒估算额外训练累计约 0.52 小时（并发实际耗时另记），用于判断下一轮保留或修正哪一结构。
