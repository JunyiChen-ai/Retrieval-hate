# 区间内外残差 backbone：独立 code review

日期：2026-10-02。审稿人：独立 Codex agent。依据：当前新实验实现、共享 QTL 接入代码及 `docs/reviews/20261002_residual_io_backbone_proposal.md`。已完整阅读实验 README；本记录是规则 6 的一次独立 code review，只审影响实验观察与结论的 bug。

**结论：PASS。未发现需要阻断完整搜索的实现 bug，无修复要求。** 不代表性能或机制贡献已经成立。没有训练、smoke、缩减 epoch、优化器更新或新增测试套件；没有修改模型、共享 trainer、研究规则、AGENTS 或其它实验。

## 1. 阅读范围与计算链

- 全部新代码：`experiments/20261002_residual_io_backbone/{backbone.py,train.py,search.py}`、三个 launch 文件，以及 `src/qtl/interval_stats.py`。
- 共享接入：`src/qtl/{model,content,train,search,run_owner}.py`，并追踪 `data.py` 的批次和 padding、`policy.py` 的五 crop 节点先验、`cpolicy.py` 的最终后验，以及 `ctree.py` 的树似然和节点势。
- 评测调用：`src/hier_evidence_common.py` 的 `run_evaluator` 调用现有 `scripts/reproduction_baselines/eval_baseline_scores.py` 统一入口；没有新评测实现。r5 的旧 model 入口也是共享 `qtl.model` 的兼容入口。

核对结果：

1. **区间/补集统计正确。** 同一次 `fc_v/fc_a` 的点式投影同时进入 CMA 和统计分支。统计叶行由真实 `leaf_row` 选取；count 为有效缓存行数，padding 不进入 sum 或 count。inside 子树求和、outside 父补集加兄弟子树覆盖单叶、奇数长度、不平衡树及混合长度 batch。根 outside 为 0。每个节点读出按真实行数除法，不是对子均值不加权平均。
2. **outside 的排除边界正确。** 补集统计对本区间投影缓存行无依赖，`io_outside=false` 把补集统计置 0。投影是逐行线性层，统计发生在 CMA 前；CMA 自身仍允许跨全视频信息，不能把该排除解释为整个表示或原始媒体感受野隔离。
3. **残差接入和 r5 等价性正确。** context 最后 Linear 的 weight/bias 全零；相同 base 参数和同次调用的 dropout 随机状态下，六项 forward 输出和节点势都严格回到共享 PriorNet。视频头只读取 `base_v+base_a`，非零 context 权重不会直接改变 g。联合训练仍会改变共享投影/CMA/视频头参数；新增 dropout 也会改变后续随机流，不保证训练轨迹等于 r5。
4. **节点与训练链有效。** 叶残差进入共享秒级头和原 CMAL 表示。内部节点的原 attention 池化以当前秒级表示为输入，再加两模态节点残差；`super().node_logits + F.linear(D, node.weight)` 与一次 `node(H+D)` 等价，只计一次 bias，叶 phi 保持 0。trainer 把同一 forward 的 s/g/phi 传入 `ctree.log_evidence`，并把所有 model 参数交给 Adam；推理由 `video_prior_nodes → cpolicy.run_batch → ctree.marginals` 使用同一 phi，不存在计算了新机制却未进入最终分数的分支。
5. **layout 与 checkpoint 接入正确。** layout cache 仅保存由长度、padding 宽度及 device 决定的拓扑；训练使用 prefix mask、同一视频顺序及同一 padded width 构建 Forest，五 crop 推理同样对齐。每次 forward 清除旧输出引用，`node_logits` 核对当前输出引用、视频长度和节点数；参数均进入 state_dict，重载不依赖旧计算图。关闭 `io_residual` 直接恢复 base forward/node 路径。
6. **搜索与输出协议未改变。** 两语料入口只改变 corpus，均使用同一 backbone 和四标量搜索空间。默认完整 50 epoch，无提前停止/within 剪枝；validation AP/ROC 均值选择 checkpoint，加载该 checkpoint 后统一 test。每 seed 独立 TPE，sampler seed 等于训练 seed；首 trial 实测后锁定 20 或 5，完整预算不足时 search/owner 明确失败。开发期 test 目标与 validation 选 trial 均独立记录；within 照常评测，按最新用户裁定报告为主指标。
7. **新目录与监控 owner 接入正确。** launcher 默认写 `runs/20261002_residual_io_backbone/<corpus>/seed<seed>/`，不复用前两版目录；既有 identity 会阻止误启动。共享 owner 写真实 host/PID/PGID/SID/start_ticks、日志和 completion，兼容现有独立 monitor 的真实进程核验。review 没有启动实验或 monitor；主 agent 启动时仍需按已有要求绑定 monitor 并确认首次 RUNNING。completion 本身不能替代结束后的全部 epoch/评测审计。

## 2. CPU 数值核验

环境：`HateVideo` Python，torch `2.7.1+cu128`，仅 CPU 单线程。输入为合成张量，不读取训练数据、不训练；检查 forward/backward 算子、索引、缓存和 state_dict。原始数值记录：`runs/20261002_residual_io_backbone/code_review/cpu_operator_review.json`。

| 检查 | 实际结果 |
|---|---|
| 长度 1–65、padded width 70 的 4,225 节点，sum/count 与逐节点直接区间/补集求和比较（float64） | 最大绝对误差 `7.11e-15`；关闭 outside 全零 |
| 一个非根内部节点的 outside 对全部输入行的梯度支持 | 本视频补集严格为 1；本区间、padding、其它视频严格为 0 |
| 相同 base 参数，零 context，eval 与 train 模式下六项 forward 输出及 phi | 全部严格相等，最大差 `0`；节点头另外设为非零以避免零 phi 的空验证 |
| 非零 context 时 g 与 base 比较 | 严格相等；s 实际发生变化 |
| 非零 node bias 下，节点实现与直接 `node(H+D)` 比较 | 最大误差 `2.98e-7`；叶 phi 严格为 0 |
| 真实 `ctree.log_evidence` 反向传播 | 两投影、node head 和全部 context 参数梯度均有限且非零；零初始化时末层 weight 梯度范数 `2.3322`，可以开始学习 |
| 非零残差的最终树后验 | 相对 base 最大变化 `.06436`；去 phi 最大变化 `.23556`，确认两条接入均影响后验 |
| 已填充缓存后 state_dict 保存/重载 | 六项输出与 phi 严格相等 |
| 非零 context 权重但关闭 `io_residual` | 六项输出与 phi 严格回到 base |
| 混合长度 `[1,3,8,17]`、padding 20，与逐视频无 padding 比较 | s/phi 最大误差 `7.45e-7`；缓存不含带梯度张量 |

这些数值仅是代码审查证据，不能充当完整 trial、性能结果或机制消融。可按已通过提案中的既定协议直接启动两主数据 seed234 完整搜索。
