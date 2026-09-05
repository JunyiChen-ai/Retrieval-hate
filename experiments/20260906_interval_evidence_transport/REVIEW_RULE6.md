# 候选9独立 code review（规则6，一次）

日期：2026-09-06。审阅者：独立 agent `code_review_c9`。依据：完整阅读 AGENTS.md、RESEARCH_ITERATION_RULES.md、本目录 README 和已 GO 的 REVIEW_RULE4.md；检查本目录 model.py、train.py、search.py、launch/run_search.sh，src/interval_observation_data.py，以及其实际调用的 temporal_measure、vlm_verdict、hier_evidence_common、fixed_training_protocol、fixed_optuna_protocol 和原 MACIL-SD 时间/采样函数。

## 结论：GO

未发现会改变实验观察或结论的实现 bug。允许按既定预算开始两语料 seed234 完整训练；本结论不认证性能、三模块有效性或 novelty 完成。没有运行模型 smoke、缩小训练或任何训练；只做静态审阅、独立公式的小张量计算及实际输入只读解析。

## 机制与评测链

- **M1**：内容 prior 仅来自真实交叠加权的两路内容，不读取 grade。观察 NLL 使用 `logsumexp(log prior + log E(g|z))`，不是 posterior 重构输入。发射矩阵最后一维是观测 g，gather 轴正确；posterior 进入 state embedding、query、分配和最终预测。`hard_observation` 保留同一 prior/NLL，仅将 posterior 固定 one-hot；`categorical_noise` 初始化为同一序数发射；`no_vlm` 不加载任何 split 的 grade，且观察 loss 为零。
- **M2**：区间外位置为零概率；A 在位置维归一化。同一 aa/av 用于内容聚合、跨模态回送以及最终 R。两路更新使用同一个参数集合，并同时取更新前的对方区间表示。`uniform_assignment` 同时替换这些共享路径，不是只改变展示量。seq_len 在与 GPU arange 比较前显式移动至设备。
- **M3**：内容 `[a,v,a*v]` 投影与区间投影之和精确等于 512→128 拼接第一仿射层；偏置只加一次。后续共享非线性给出条件概率，R 按区间维归一化后边缘化。`additive_readout` 的 B×34×T 与 B×34×1 广播正确，输出 B×T。训练 top-ceil(valid T/16) 使用同一最终概率，返回其 logit 后经共享 sigmoid 恢复概率；数值 floor 是已声明固定参数。
- **时间与输入**：训练 visual/audio 采用原 `uniform_extract` 的 uint16 linspace 索引；integration_cells 使用相同索引，覆盖完整归一化时间。五 crop 与原始评测 snippet→1fps lookup 不变。观察 grade 是冻结 VLM 输入而非 test GT；所有 split 可读 grade 是本方法部署定义，只有 train 进行观察 loss 优化。归一化只统计 train/crop0。split 去重、固定 GT 覆盖及 VLM 严格覆盖检查存在。
- **训练/选择**：50 epoch、Adam/cosine、batch32，validation pooled(AP+ROC)/2 选并恢复完整 state_dict；随后唯一评测器输出 val/test。TPE 搜索空间和 README 相符；本次 review 内确认原三参数 sample 原样升入 `src/content_search_space.py` 后 import 正确、范围未改。test pooled 均值排序，within 不合格 trial 完整输出后 PRUNED；首 trial 耗时冻结 20/5，确认 seed 继承预算。没有新评测实现、ensemble 或 inference 后处理。
- **复杂度**：没有 T×T 注意力。M3 第一仿射的内容投影只执行一次；34 个区间循环剩余非线性，推断不常驻 B×T×34×H。训练 autograd 仍保存各区间计算所需激活，不能据此声称训练内存不随 34 增长；实际耗时由完整首 trial 测量。

## 实际输入边界核查（本机，只读）

用固定 cohort、原 feature 路径的 mmap 和时间元数据逐视频检查；未读取 test 标签内容来训练或改参数。训练五 crop 全部真实行检查是否全零，三个搜索 max_seqlen 对应的完整时间单元检查每个 K30/K4 窗口是否有正交叠；val/test 检查原秒映射是否落到零宽单元。

| 语料 | train/val/test 视频 | train 五 crop 真实行数 | 全零真实 I3D 行 | eval 零宽单元 | 秒映射引用零宽单元 | 全 mask 窗口 |
| --- | --- | --- | --- | --- | --- | --- |
| HateMM | 744/109/214 | 844820 | 0 | 0 | 0 | 0 |
| HateClipSeg | 251/63/79 | 446905 | 0 | 16 | 0 | 0 |

HCS 的16个零宽单元来自已有尾部时间范围；模型屏蔽它们，原评测秒映射不引用它们，故不影响当前分数覆盖。当前输入不存在使共享 `_seq_len_of` 把真实全零 I3D 行误当 padding 的情况。检查范围中最长原始序列 HateMM 8712、HCS 524；eval 保持完整序列，没有为了成本截断。

## 独立小张量公式核查

CPU float64、B=2/J=34/T=7/H=128，仅计算公式，不实例化或训练候选模型：M3 拼接/因式分解第一仿射最大绝对差 1.28e-13；M1 概率域求和与 logsumexp 边缘对数差 4.44e-16；A/R 在有效维和为1，零宽位置边缘概率为0，内容/区间/分配梯度均有限；additive 广播输出形状 2×7。

本轮未发现需要修复的阻断项，不要求额外 review 或前置训练。后续运行若出现异常，先诊断实际日志与输出，不据此预判机制有效或失败。
