# 候选5：独立 code review（规则6）

日期：2026-09-05。审阅者：独立 agent `code_review_c5`。依据：本目录 README、REVIEW_RULE4.md 与当前实现；只做静态代码及实际模型配置检查，未运行 smoke、训练或单元测试，未修改实现代码。

## 裁定：必须修复，暂不启动训练

完成下列三项后只确认修复，不重开泛化 review。提案 GO 不受本次实现问题影响；当前输入与实现尚不能用于方法结论。

### 1. VLM读取的是经过惩罚的 generation scores，不是原始下一token分布

位置：`scripts/analysis/extract_interventional_evidence.py` 的 `measure()`。

`model.generate(... output_scores=True)` 后读取 `out.scores[0]`。实际 Qwen2.5-VL-7B-Instruct 的 `generation_config.json` 设置 `repetition_penalty: 1.05`，抽取没有覆盖它；本机安装的 transformers `generation/utils.py` 会添加 `RepetitionPenaltyLogitsProcessor`，`scores` 是处理后分数，`logits` 才是未经处理分数。提示中含 Yes/No，所以目标token也可能受惩罚。`do_sample=False` 不会取消 repetition penalty。

影响：四干预的连续log-odds和熵不是预注册的原始条件分布，并可能受提示token出现方式影响；不能仅将其记录为raw logits继续实验。

修复：读取 `output_logits=True` 返回的 `out.logits[0]`（安装版本支持），或直接前向末token logits；明确保持同一前缀的两个单token读出。递增抽取可读版本，核验远端实际配置；已生成的受影响输出保留供诊断但不得与修复后输入混用，不能仅改缓存版本字段。训练入口也须验证所需版本。

### 2. 缺失远端特征会静默改变训练/评测队列

位置：`train.py` 的 `ids = ... common.usable(...)`、随后GT交集过滤，以及最终 `common.run_evaluator()` 结果读取。

`usable` 按文件存在性删样本；val/test随后再取GT交集。共享评测器默认允许缺少GT样本，并把缺失计数写到结果；此训练实现没有检查这些计数，搜索只读取有限数值。因此远端准备遗漏特征时，可能在缩小的train/val/test上照常完成并进入Optuna排序，改变固定评测结论。

修复：以冻结split及原有固定GT协议为准，训练前拒绝缺失必需特征；val/test最终名单必须等于相应冻结GT名单，不允许未知样本被交集自动剔除。已知HateMM test split为215、GT为214，`hate_video_427`无GT是既有baseline协议，须显式记录并只允许这一既定排除，不改GT集合。最终确认评测器的missing/extra计数均为零或调用已有full-coverage开关。无需改动评测器数学。

### 3. 补seed重新决定5/20预算，未继承首轮冻结预算

位置：`src/fixed_optuna_protocol.py` 的 `budget_path = root / 'budget.json'` 和首次trial后写预算逻辑。

每个seed各自根据本机首trial耗时重新决定5或20；跨机器负载变化可能使seed2025/3407与seed234预算不同。规则8明确补seed走同样trial数，此实现没有保持该约束。

修复：首轮seed234按完整trial耗时冻结预算；同语料确认seed读取并验证该预算，缺少来源预算则停止，不自行重测改预算。各seed仍可记录自身实测耗时。README记录冻结值；跨机复制这项可读预算及来源，不计算哈希。

## 已核对、未发现阻断bug的部分

- 四路顺序为av/v/a/empty，差分、熵索引正确；`no_interaction` 同时移除交互和全输入熵，`full_input_only` 与 `four_logits` 不因数组置零别名而丢数据。
- 两粒度先用同一snippet中点映射，再与content共同做确定性行采样；key/value具有同一行索引。padding key被屏蔽，bag/block损失限有效长度。继承既有均匀帧分窗/秒映射约定，不声称新增精确帧时间戳。
- Yager的belief、冲突和unknown公式正确，概率归一；最终bag loss保留到两个证据头、关联骨干的梯度，无detach或推理专用融合。Dempster替换使用非冲突归一化；additive替换、ordinary_attention确实改变对应模块。
- 模型仅读取音频/BERT内容与新16通道，不读取旧HMM scaffold。HMM拟合与posterior输入仅train，val/test scaffold为零；块损失只在train使用。加载test GT没有进入梯度或checkpoint选择。
- 50epoch完整训练、validation AP/ROC均值选state副本、加载该state后test；调用原有统一评测器，未复制指标算法。
- Optuna使用每seed固定TPE、test AP/ROC目标、within剪枝并保留输出；validation选trial参考包含有完整输出的pruned trial。失败/遗留RUNNING会拒绝自动resume，正常resume保留sampler状态和已消耗trial数，无自动重复旧trial。

最强baseline+同输入尚未实现，三seed模块替换结果尚无；这是完整目标缺口，不是当前全方法训练的额外门。修复本记录的问题也不等于三模块novelty或整体paradigm已成立。
