# 候选8独立 Code Review：GO

日期：2026-09-06。审查者：独立 agent `code_review_c8`。范围为现行研究规则6，仅检查影响实验观察/科研结论的实现问题；这是本候选唯一一次 code review，不是性能或 novelty 确认。

结论：**GO，未发现需要阻止正式完整训练的结论级 bug。** 未运行训练、smoke、缩短 epoch 或数据子集试跑；执行的只有只读代码/缓存元数据检查与 CPU 合成张量数学、梯度检查。

## 已核对的执行链

- 已完整阅读本候选 README、REVIEW_RULE4、model.py、dataset.py、train.py、search.py、launch/run_search.sh，及共享 temporal_measure、vlm_verdict、hier_evidence_common、fixed_training_protocol、fixed_optuna_protocol；追查原 MACIL-SD 的 uniform_extract/process_feat、序列长度恢复、时间对齐和唯一评测器调用。
- M1：严格加载只返回 train ID 的原 Qwen K30/K4 裁定，完整覆盖 HateMM 744 与 HateClipSeg 251 个训练视频；值域、长度、有限性及重复记录一致性检查通过。阈值≥2与方案相同；误报率初始化只用 train 负视频，Beta(1,1)平滑，q=r+(1-r)*sigmoid(gap)保证单调通道。held-out 缓存文件可能因统一 JSONL 扫描而被解析，但非 train 记录立即跳过，不进入输出、统计、梯度或推断。
- 教师标签只在附加列 A_EXT_DIM+2 后，forward 仅使用内容1920维与前两列时间区间。EvalDataset 提供零标签占位，不调用 VLM 缓存。CPU 检查将全部教师列替换为99，forward 输出逐元素相同；部署无 VLM 调用。
- M2：train/crop0内容统计作为 checkpoint buffer，无 held-out 统计。factorized 的 total 为正总强度，softmax(local+log(width))生成区间质量比例；rate=mass/width 是相对时间密度。unfactorized 使用同一内容骨干的逐位置 softplus 密度。卷积 padding 逐层遮挡，seq_len=None 使用完整序列，短序列可执行。
- 时间积分与原 uniform_extract 使用完全相同的 uint16 linspace 索引。选中中心的 Voronoi 区间覆盖[0,1]，不删除未抽中的时间；两尺度窗交叠总量逐 token 与逐 window 均守恒。密度的单位是“每单位归一化视频时间的事件率”；mass=rate*d 才是区间积分强度。REVIEW_RULE4 中将 lambda_t 称为质量的那句不能沿用；以当前 README 与实现的密度/质量明确区分为准。
- M3：视频 Bernoulli 事件 NLL 用 expm1 稳定表达；窗口质量经1-exp(-mass)进入同一可学习噪声通道，三项均按视频/窗口均值求和。topk_event 只替换窗口事件计算，保留视频事件监督与噪声通道；有效交叠成员保证短窗口至少一个 token，ceil(N/16)与代码一致。hard_observation、no_vlm、fine_only 与描述对应。
- 推断返回第四项 log(rate)，共用 score_split 的 sigmoid、五crop平均与既有1fps映射；未增加 calibration、平滑或 ensemble，唯一评测器未改。积分时间按现有基础特征/评测的 n_seconds 定义，不声称重新精确解码原视频时间。
- 完整50epoch、batch32、Adam/cosine、预先声明的三项搜索空间无方法内分支。validation AP/ROC均值选择并加载 checkpoint 后调用统一 evaluator；test AP/ROC均值给 TPE 选 trial，within 下限只 prune 已完整评测的 trial；首 trial 耗时固定20/5预算。保存并恢复 normalization/noise/model 全部参数，不存在只保存最后 epoch 的问题。

## 限定检查记录

1. 合成时间网格：(原长度,选中长度)=(1,1),(2,2),(7,7),(503,150),(503,300)。原采样索引一致；K4/K30窗宽及 token 宽度的交叠和均在数值容差内守恒。
2. 六个 arm，各检查有效长度1、3、41及两个padding token：forward、loss、参数梯度有限；内容投影有梯度；使用学习通道的 arm 有通道梯度；裁定列变化不影响 forward；移除padding并使用 seq_len=None 的有效输出与带padding输出相同。
3. 视频事件 NLL 在总强度1e-8、1e-4、1、10、100、1e4及两类标签下，loss/梯度均有限。sigmoid(log(rate))与rate/(1+rate)一致。
4. 两语料全部 split 的现有时间元数据及150/200/300训练选点均可构建区间。个别视觉尾部超出音频/评测时域产生零宽区间，由 valid 遮挡；进一步检查两语料所有 val/test 的实际1fps lookup，均未引用零宽区间。

上述检查只保证实现与当前方案相符。因子化是否定位不足、噪声通道是否趋于低信息、以及三个模块是否达到三seed两语料消融门，必须由正式实验回答；不在本次 code review 中作先验否决或成功声明。
