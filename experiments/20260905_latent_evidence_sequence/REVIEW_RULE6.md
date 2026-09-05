# 候选6独立 code review（规则6）

截至 2026-09-05；评审人：独立 agent `code_review_c6`。依据本目录当前 `README.md`、`REVIEW_RULE4.md`、`model.py`、`train.py`、`search.py`、`launch/run_search.sh`，以及 `src/fixed_training_protocol.py`、`src/interventional_observations.py`、`src/fixed_optuna_protocol.py`、`src/hier_evidence_common.py` 和其既有时间映射实现。未运行正式实验、缩短训练或小数据试训。

## 结论：GO

本次未发现会改变科学观察或结论的实现 bug，可按既定规则启动两语料完整 seed234 搜索。本结论只针对所审实现，不代表性能、三模块 novelty 或整体范式已成立；不增加额外前置测试门或评审轮次。

## 已核对的科学关键路径

- **M1 密度正确进入训练。** K30/K4 的四路 v2 原始 logits 按既有窗口中点映射拼成8维；均值、标准差及状态均值初始化仅读取 train 输入/视频标签。Cholesky 正对角、二次型、完整 logdet 与高斯常数项齐全；似然定义在固定 train 标准化坐标中。其固定 Jacobian 不影响同臂参数优化。均值、完整协方差通过事件项及观测项反传，不是冻结 teacher。
- **M2 转移实际进入路径权重。** 内容只使用视觉及音频/文本列，不读取保留的 scaffold 标签列。两个 sigmoid 定义合法的2×2转移；首状态有单独的内容条件概率。三状态自动机正确区分从未命中背景、当前正状态和已命中背景，正状态返回背景后不会忘记事件，也允许再次进入正状态。首 token 为正时正确计入正视频事件。
- **M3 分区函数、后验及损失一致。** 首矩阵重复初始行后只读取一行，不产生额外初始质量；并行前缀归一化的累计 scalar 被正确恢复到 `log Z`。正事件用两种已命中终态的 logsumexp，不作近等量相减。后缀方向正确，后缀归一化丢弃的仅是每时间点状态共享常数，因此不会改变局部后验。实际损失为事件 NLL 加 `-log Z/(D*T)`；没有 test 标签条件化。训练主臂不计算用不到的后验，但其事件/观测损失与测试后验来自同一个状态模型，不存在另一推理模型或后处理。
- **Padding 不进入有效路径或长度。** 无效位置在内容投影/时间卷积后置零，转移矩阵替换为单位矩阵；累计配分函数及有效 token 后验不变，观测项以有效长度归一化。沿用既有训练长度恢复及确定性共同下采样，音频、视觉与VLM观测使用相同时间索引；评估保留完整序列。
- **三个主消融隔离对应贡献。** `diagonal_emission` 只去8维协方差的非对角项；`static_transition` 只改时间共享的转移，保留内容条件初始概率；`event_to_topk` 只将视频事件项替换为局部后验 top-k BCE，保留相同观测 NLL 和最终后验。Top-k divisor 固定16，仅用于该臂训练。辅助 `independent_state` 的两行转移确实相同。`full_input_emission` 使用两粒度 av 共2维，损失分母自动使用该臂维数，不把零填充当8维输入。
- **Split、checkpoint、评测与搜索链正确。** 完整固定 split 与唯一旧 GT 排除项有显式检查；验证/测试 GT 不进入梯度或初始化。每 trial 完整50 epoch，validation pooled AP/ROC 均值选 checkpoint，恢复后分别调用同一评测器输出 validation/test 三指标。测试使用既有五 crop 平均及1 fps 映射。Optuna 使用 test(AP+ROC)/2 排序，within 下限只决定 prune，validation 选 trial 的结果仅另存参考。首 trial 实测决定固定20/5预算，确认 seed 继承预算；无提前缩短或 test 选 epoch。

## 有界数学核对（只读诊断，非新增实验门）

在 CPU float64 上直接核对数学函数，不读取训练数据、不优化模型、不写 checkpoint：

- 并行前缀/后缀与逐步 log-semiring 乘积比较，长度1、2、3、5、17、257；恢复累计 scalar 后最大绝对误差约 `1.36e-12`，对应梯度最大差约 `5.76e-13`。
- 对实际模型参数生成的长度3/5状态序列枚举全部二值路径，核对 full、diagonal、static、independent、event_to_topk：局部后验最大误差低于 `1.5e-16`，事件 log 概率低于 `1.2e-14`，观测 NLL 低于 `2.3e-16`；损失反传未见非有限梯度，均值/协方差获得非零梯度。
- 非对角 Cholesky 的发射 log 密度与标准多元高斯实现比较，最大误差约 `1.78e-15`。
- 同一有效4 token 序列与补齐到7 token 的输入比较：事件概率及观测 NLL 无差异，有效后验最大差约 `1.57e-17`。

这些数值仅确认所审递推与公式，没有替代或预判正式训练结果。

## C5 共享代码迁移

核对 C5 `train.py` 原循环与迁入 `src/fixed_training_protocol.py` 的调用：Adam、50 epoch cosine、batch 顺序、长度裁剪、C5 视频 BCE/可选 block loss、validation criterion、最佳 state 恢复、五 crop 分数及统一评测器均保持原语义。覆盖检查由 `len(ids[split])` 改成同一 IDs 构造的 dataset 长度，二者等价。

v2 解析迁入 `src/interventional_observations.py` 保持版本、ID、顺序、窗口索引、shape 与特征转换。唯一检查收紧是立即拒绝原始 logits/entropy 的非有限值（旧实现只在臂转换后检查）；这不改变有效有限 v2 缓存的数值或 C5 已完成结果，不需要重跑旧实验。

## 解释边界

长序列事件饱和、重复窗口观测的条件独立假设、初始化偏置、生成项影响及三模块是否有效仍是完整实验要回答的问题，不以理论猜测阻断。共享最终分数格式保留既有概率精度限制；不修改评测器。后续仍需两语料完整筛选/确认、规定消融以及最强 baseline + 同输入，才可能作最终研究声明。
