# Inside–outside backbone：一次独立 code review

日期：2026-10-01。结论：**PASS，可以进入两主数据完整训练。**

范围按 `RESEARCH_ITERATION_RULES.md` 第 6 条：只查可能改变实验观察或结论的实现错误。本次检查的是 `experiments/20261001_inside_outside_backbone/` 初版、共享基础设施迁入 `src/qtl/` 后的工作树，以及配套启动器与 `scripts/monitor_run.py`。未重新做 proposal review，未改训练实现、评测器或研究规则；未启动训练、smoke 或缩短实验。

## 核验结果

1. **内容依赖与节点顺序正确。** `backbone.py` 的 inside 依次读取有序左右子节点；outside 读取父 outside 与兄弟 inside；叶子恢复用 `forest.levels[*].leaf_row`，内部节点势用全局 node id，视频头按 `forest.pos_root` 恢复原 batch 顺序。用长度 `[1,2,3,5,8,9]` 的变长 batch 与逐视频运行比较，六项输出及节点势最大差 `4.77e-7`。覆盖了根本身是叶子、奇数长度和不同树深。
2. **outside 确实排除本区间缓存行。** 对 9 秒树的 `[0,4)` 节点，在融合器之前取 outside，向两路输入求导：该区间输入梯度严格为 `0`，外部输入梯度非零。这里核验的是缓存行的计算依赖，不代表 I3D/BERT 原始感受野与该区间隔离。
3. **结构进入实际训练目标和最终分数。** 按共享 trainer 的 `ctree.log_evidence` 构造视频标签与答案似然损失；全部启用的新骨干参数均有有限非零梯度，包括 inside 组合、左右 outside 组合、根 outside、融合器、逐秒头、视频头及节点头。刻意未启用的备用 `g0` 不计。`phi` 没有在训练路径中 detach，亦进入先验归一化；推断中经 `policy.video_prior_nodes → cpolicy.run_batch → ctree.marginals` 使用。将非零节点头置入同一结构后，去掉 `phi` 使最终秒级后验最大变化约 `0.0929`，不是只训练但不读出的支路。
4. **工厂、参数与 checkpoint 一致。** 新 CLI 显式将 `make_model` 与 `EXTRA_DEFAULTS` 注入共享 trainer；搜索设置经 hparams/config 进入同一工厂。验证选择后重载同一实例的参数，保存 model/answer model/chain。CPU 上重新建模并 `load_state_dict(..., strict=True)` 后，六项输出与 `phi` 逐元素一致。
5. **迁移未改变旧模型数学行为与共享 global。** 逐文件比较迁移前内容与 `src/qtl/`：树推断、答案模型、数据处理和旧模型除包相对导入外保持原逻辑，trainer 的实质增加仅为工厂/default 注入。旧 `data`、`qtree`、`train` 薄入口与共享模块是同一 module 对象，`configure_source` 的软答案类别数/档数修改在双方可见。迁移前/后 MACIL 模型在相同随机种子下，参数名称、初始化值、六项前向输出和节点势逐元素一致。旧 search CLI 第 8 行已显式传同目录 trainer，新 search CLI 同样显式传自己的 trainer；两者均不直接执行含相对导入的共享 train 文件。
6. **无新增 split/答案泄漏，统一评测链未改。** 数据与缓存对齐沿用共享 `load_fixed_cohort` 的 split overlap、覆盖率检查及 `store.T == T_ans` 检查；软答案分位数与锚定答案模型只用 train，梯度目标只取 train 视频。骨干 forward 不接收答案或标签；validation 选 checkpoint；test 只用于评测和现行开发期 Optuna 排序。validation 指标仍经共享 `frame_metrics`，test 经共享 `run_evaluator`，没有复制评测逻辑。AP、ROC、within 三项均保留；本轮未自行改动明确约定的 checkpoint/search 标量。
7. **结束标记没有代替结果完整性。** shared search 在 COMPLETE 数不等于固定 budget 时抛错；run owner 又核对 study、每个 trial 的原始 fixed8 评测文件、缺失视频数与三项指标有限性，之后才写 success。失败不能以正常 DONE 退出。
8. **monitor 绑定与状态处理符合本轮启动链。** owner 记录 host/run/pid/start_ticks/pgid/sid；monitor 固定该 identity，核对主进程及同会话包含仓库路径的相关进程，排除 zombie。SSH 失败、超时、暂不可读输出只重试；无 completion 的停止需连续三次观察；通知锁、已发送标记和失败重试均存在。训练/搜索实际子命令使用绝对仓库路径，可被现有筛选匹配。monitor 不重启或改动训练。

## 验证边界

数值核验在本机 `HateVideo` CPU 上进行，使用随机张量、隐藏宽度 8、dropout 0，没有训练数据、优化器更新或 GPU 训练。它们只验证计算图、顺序和加载正确性，不证明性能、吞吐或 novelty 已成立。首次正式启动后仍由主 agent 按 AGENTS.md 核对脱离终端运行、monitor 存活与首次成功观察；通知接口的实际可达性不由这次静态检查替代。

审查中曾怀疑旧 search CLI 没有传 trainer；读取其实际第 8 行后确认已有传参，撤回该问题，不列为缺陷。没有需要阻断本轮完整训练的未修复问题。
