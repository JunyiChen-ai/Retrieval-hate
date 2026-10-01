# 可加统计量 inside–outside backbone：一次独立 code review

日期：2026-10-01。结论：**PASS，可以进入两主数据完整训练。没有未修复的阻断问题。**

按 `RESEARCH_ITERATION_RULES.md` 第 6 条，只核对会影响观察与结论的实现错误。范围为 `experiments/20261001_associative_io_backbone/{backbone,train,search}.py`、该实验新启动入口、新增共享 `src/qtl/{content,run_owner}.py`，并追踪已有 trainer、树推断与统一评测的调用链。这是本候选唯一一次独立 code review；未重做 proposal review，未改训练代码、评测器或规则，未运行训练、smoke 或缩短实验。Pursuing goal 状态未变更。

## 1. 机制与索引

- **统计量没有重复计数。** 点式投影及每行每 head 正权重形成 `[w*x,w]`；父 inside 严格等于左右子 inside 之和；子 outside 严格等于父 outside 加兄弟 inside。传播不再经过非线性组合器，归一化只发生在 readout。对长度 `[9,1,5,2,17,3,8,7,16]` 的混合 batch，逐一核对全部 127 个节点与直接区间、直接补集的加权和及 mass：float64 最大绝对误差分别为 `3.55e-15`、`1.07e-14`，读出最大差 `9.99e-16`。这是浮点误差范围的等价，不支持“不同求和顺序带来额外表达力”的主张。
- **outside 排除自身缓存行。** 对 9 行视频的 `[0,4)` 节点，在 ContextFusion 之前读取 outside，并对原始两路输入求导：自身 4 行的最大梯度严格为 `0`；其余行的两路梯度均非零。权重没有全视频 softmax，叶子的 LayerNorm 也只沿 hidden 维。排除仅针对输入缓存行，不涉及上游 I3D/BERT 原始感受野；融合后的节点表示本来就同时读取 inside 和 outside。
- **奇数长度、单叶根与 batch 顺序对齐。** `ContentLayouts` 沿用 `ctree.Forest` 的 leaf-first 层布局，叶子恢复用 `leaf_row`，节点势恢复用全局 node id，视频根用 `pos_root`。上述变长 float32 batch 与逐视频前向比较，六项输出和节点势最大差 `4.77e-7`。把所有 padding 输入改为大数后，有效输出与节点势逐元素不变。trainer 生成前缀连续 mask，符合布局使用 `mask.sum(1)` 的约定。
- **数值与消融开关正确。** 主配置 float32 下 `exp(clamp(score,-10,10))` 正且有限；非空区间 mass 不会触及 `1e-12` 下限。将 attention 参数扩大至产生饱和权重后，所有统计量和读出仍有限。根 outside 的统计量与读出严格为零；关闭 outside 后各层 outside 都为零；关闭 attention 后统计量等于普通逐行求和与区间长度。四 head 均属于同一网络。

## 2. 实际训练与最终分数

- 新入口把 `EXTRA_DEFAULTS` 和 `make_model` 注入共享 trainer；搜索入口显式指向本实验 `train.py`，没有误用旧 MACIL 或初版骨干。超参 JSON 在既有 defaults 上合并，未知字段仍拒绝；主配置为 `associative_io`、8 档 `soft_both`、`node_prior=true`，原有 50 epoch 与四标量空间保留。
- 主目标 `objective=tree` 下，`s/g/phi` 均直接进入 `ctree.log_evidence`，没有 detach 内容编码器。用默认零初始化节点头、随机输入和 queryable 节点答案似然，构造与 trainer 相同的视频标签项及按答案数归一化的似然项：23 个骨干参数张量全部获得有限非零梯度。没有执行优化器更新。
- 节点势进入有答案和无答案的归一化计算；推断经 `policy.video_prior_nodes → cpolicy.run_batch → ctree.marginals` 使用。以非零节点头作计算图探针，分别去掉 `phi`、将 `s` 或 `g` 置零，最终逐秒后验最大改变约 `.0560`、`.6408`、`.1882`。这些支路实际影响最终分数。
- 重新调用工厂实例化、序列化并严格加载 `state_dict` 后，六项输出和 `phi` 逐元素一致。共享 trainer 仍按 validation AP/ROC 均值选择并重载 checkpoint，再立即 test；test 三项经 `hc.run_evaluator` 调用原统一评测器。没有新增评测实现或按语料分支。
- 骨干 forward 只接收特征和 mask，不接收答案或标签。原共享数据链的 train/validation/test 划分、时间轴覆盖与对齐不变；训练答案、答案模型锚定及软答案分位数均沿用 train 限定。test 仍仅进入评测及规则明确允许的开发期 Optuna 排序；within 按最新用户裁定参与主要结论，此次没有自行改动 checkpoint/search 标量。

## 3. 新启动入口与现有活动任务

- 两条 lab3 shell 均指向新实验的 `launch/run_search.py`；该入口向共享 owner 传入新实验 `search.py` 和独立默认输出根 `runs/20261001_associative_io_backbone`。共享 owner 的 repository 根经 `parents[2]` 正确解析；后续 trial 调用本实验 trainer，输出层级为 `<corpus>/seed<seed>/trial<k>`。
- owner 继续记录 host、PID、进程组、session、启动 ticks 和绝对 run 路径；已有 identity 不自动重启。只有子进程成功、COMPLETE trial 数等于预算、每个 fixed8 原始评测文件存在且三项指标有限、没有缺视频时才写 success；共享 search 对预算未完成同样报错。完成通知仍不能替代接续时核验完整 epoch 历史和其它预算评测。
- `ContextFusion` 与 owner 原逻辑已升入共享目录，新候选使用共享定义；初版正在运行的文件保留冻结定义，等其搜索结束再切换兼容入口。这是有意暂存，不能为去重而中途替换活动实验代码。本次未改旧原型、共享 trainer 或推断实现。实际后台 monitor 的启动、首次成功观察和结果回传仍由主 agent 在正式运行时确认。
- 三个 CLI 的 `--help` 与两条 shell 的语法检查通过；没有启动搜索或训练。

## 验证边界

数值核验在 `sc474399` 的 `HateVideo` CPU 上完成，随机 seed 713、hidden 8、dropout 0，未读取训练数据、没有优化器步骤，也没有 GPU 训练。原始数值记录：`runs/20261001_associative_io_backbone/code_review/numeric_checks.json`。这些检查证明所审计算图、索引和加载链与声明一致，不证明性能、吞吐、seed 稳定性或有效 novelty；后者仍需完整研究实验。
