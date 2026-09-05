# 候选7独立 code review（规则6，唯一一次）

日期：2026-09-06。审查 agent：`code_review_c7`。依据：本目录首版实现、修复后的 `Candidate.forward(..., seq_len=None)`，以及下列共享源码；不使用内容哈希。

结论：**GO（发现的接口 bug 已修复并核对）**。这只表示未发现尚未解决、会改变科研观察/结论的实现 bug，不表示性能或三个模块 novelty 已成立。没有运行训练、缩短实验或 smoke；仅阅读源码及进行 CPU 上的确定性张量/梯度数学检查。

## 范围

- 本目录 `README.md`、`model.py`、`train.py`、`search.py`、`launch/run_extract.sh`、`launch/run_search.sh`。
- `src/context_witness.py`、`src/measurement_inputs.py`、`scripts/analysis/extract_context_witness.py`。
- `src/hier_evidence_common.py` 的 cohort/cache/dataset/score 接口、`src/fixed_training_protocol.py`、`src/fixed_optuna_protocol.py`；相关 `vlm_verdict.verdict_rows`、共享视频采样器、MACIL-SD 时间对齐/长度恢复。
- uoa-lab1 实际环境 transformers 4.49 的 Qwen2.5-VL processor、`make_batched_videos` 与 generation `_sample` 源码。未启动 VLM 推理。

## 唯一发现的阻断 bug 与修复

初版 `Candidate.forward(audio, visual, lengths)` 不接受共享 `score_split` 的 `seq_len=None` 关键字调用，首个完整 trial 的 validation 必然报 `TypeError`；即使只重命名参数，None 也不能直接转长度 tensor。

主 agent 已将签名改为 `forward(audio, visual, seq_len=None)`，None 时为每个 batch 元素填入当前完整序列长度，训练第三个位置参数仍有效。本次审查内核对修复：同一确定性输入下，`seq_len=None` 与显式完整长度的输出最大绝对差为 **0**，形状 `[2,5,1]`。没有重开泛化 review。

## 影响结论的关键核对

1. **M1 实际输入与条件回答**：四路同一冻结模型，before/target/after 帧顺序为2/4/2，target/context 删除时图像置黑且对应转录不供给，边界邻窗显式缺失，不跨视频。K30 ASR、120帧等距采样及相对窗口映射一致沿共享约定；没有片段 GT 或视频标签参与抽取。六项回答为11个受格式约束的生成 token，记录偶数位置 No/Yes raw log-odds 与二项 entropy；它们条件于此前答案，不是六次独立边际推断。
2. **transformers 4.49 兼容性**：真实安装源码支持四个“PIL帧列表”构成的视频 batch，并按 prompt 顺序展开 video token。`output_logits` 保存 grammar 前的 logits；自定义 grammar 创建新 mask tensor、不原地改写传入 logits，故不会把受掩码 scores 冒充原始观测。固定 `repetition_penalty=1.0`、非采样生成；强制答案与输出长度均检查。
3. **M1 同信息对照**：full 前24维由四路 log-odds 可逆重参数化，最后6维为 full entropy；raw_four 同样保留四路 log-odds 与 full entropy。确定性随机向量重建原四路最大误差 `2.98e-7`（float32）。target_only 仅保留 target logits/entropy，其他组零填充。不能把 raw_four 差异归为增加输入信息。
4. **M2 留一位置和 padding**：双向 GRU 分方向移位后，位置 t 的重建只读 forward(t−1)/backward(t+1)，packed sequence 保证短序列右边界为零状态。重建目标是冻结1920维输入，归一化统计仅来自 train/crop0。确定性检查中，位置2的重建对当前位置视觉输入 Jacobian 最大值为 **0**，对其他位置梯度绝对和 `103.4363`；短序列单独计算和在 padding batch 中计算的有效输出最大差 `2.98e-8`。重建损失有效位置归一化，不计 padding。
5. **M3 梯度与最终读出**：全/保留/删除分类共用同一次 dropout 后 token、同一 classifier，权重均值按有效权重总和归一化。确定性检查的 kept/erased 分支对 selector 最后线性层的梯度绝对和分别为 `.17473/.19175`，非零；完整损失对 GRU 输入权重梯度绝对和 `1.18073`，有限。共享评分器使用第四返回值，C7该值为 selector logit z；最终只有 sigmoid(z) 的既有五crop平均与1fps映射，没有分类概率乘法、VLM分数相加或平滑。
6. **消融**：no_residual 同时删除 residual 输入与 reconstruction loss，故只支持二者组合的作用；visible_reconstruction 取消移位；no_deletion/no_sparsity 只删对应 loss；其他 arms 保持一致架构/训练/读出。三模块是否达效应门仍待完整 test 消融，不从梯度存在推断有效性。
7. **数据与训练链**：共享 cohort 校验 train/val/test 无重复或交叉、baseline 特征全覆盖、标签为二元，固定保留 HateMM test 的既有 `hate_video_427` 无GT排除。梯度训练只迭代 train IDs；train 均值/方差不会读取 val/test 统计。完整50epoch，以 validation pooled AP/ROC 均值选 state_dict，随后统一评测器输出 val/test。搜索以 test pooled AP/ROC 均值排序，within 下限剪枝，首完整trial计时冻结20/5预算；validation选trial仅附记，不用于搜索排序。

## 非阻断但必须保留的解释边界

- README 已补明：processor 默认2fps只给采样8帧编码归一化时间，不代表真实原视频秒时长；不得声称模型收到真实秒时间。prompt 是 frame positions，缓存是相对窗口，最终仍使用既有1fps映射。
- 加权归一化、共享分类器和非零梯度不能排除 selector/classifier 共同利用捷径；删除目标0也是方法假设。这些由完整结果判断，不作实现前否决。
- 以上检查不验证 GPU 显存峰值或全部原始文件的可解码性；正式抽取的真实解析、覆盖率和结束审计仍必须执行。不得把完成标记或本记录当作缓存完整性证明。

## 同次审查的 OOM 修复确认

正式四模式 batch 抽取在两台32GB GPU 的 vision SDPA 处 OOM，未产出视频 JSON；主 agent 将四模式改为按 `ORDER` 顺序、每次 batch size 1 调用同一冻结模型，并使用新运行目录保留失败日志。这里只确认该修复，不重开泛化 review。

已检查修改后的 `extract_context_witness.py`：四份 prompt、帧、ASR、缺失模式和问题构建未变；每个模式独立设置自身 prefix，仍生成六个自回归 Yes/No（11 token），并取位置0/2/4/6/8/10的原始 No/Yes logits。单模式 `pairs` 为 `[1,6,2]`，按原顺序沿 batch 轴拼接为 `[4,6,2]`，最终 log-odds/entropy 仍为 `[4,6]`、answers 为 `[4,6]`，每视频30窗缓存协议不变。`del output, inputs` 后仅保留小型答案和抽取后的二元 logits，不累积四模式完整模型输出。config 已记录 `mode_batch_size=1`，provenance 已改成 sequential。

结论仍为 **GO**：这是保持逐模式测量定义的内存调度修复，不改变方法、属性或下游形状。不同 batch/padding 下底层浮点实现可能有微小数值差异，不能据此声称逐位等同；此前无成品缓存，因此不存在混入旧 batch 结果的问题。是否足以消除实际 OOM，仍以正式运行结果为准；本确认未启动 VLM 或训练。
