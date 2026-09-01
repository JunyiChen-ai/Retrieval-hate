# Shared visual–speech temporal-address VLM localization：独立 novelty review

截至 2026-09-01。本审查只评估 `README.md` 中已固定候选的 Rule 12 三项 novelty 硬门；未审查代码、未提出替代 candidate、未运行训练或推理。

## 裁定

**Verdict：STOP。Novelty：4.8/10。不得实现或进入正式 HMM/HCS test。**

| Rule 12 硬门 | Verdict | 决定性依据 |
|---|---|---|
| 1. 允许跨任务 adaptation | **PASS** | NumPro 可以作为跨任务来源，不要求本项目从零发明可见帧号。 |
| 2. 来源方法未被 hateful-video detection/localization 占用 | **PASS（窄口径）** | 检索到的 NumPro/Number-Prompt 原始与后续工作均属于自然语言 video temporal grounding / moment retrieval / highlight detection；未发现 NumPro、把可见唯一数字覆写到视频帧这一来源 core，已经用于 hateful-video detection/localization。 |
| 3. adaptation non-trivial、task-specific 且 load-bearing | **FAIL** | 目标领域已经存在逐帧对齐的 speech/image/OCR multimodal hate prompting，以及 timestamp-aligned text-to-frame localization。当前唯一新增 delta 是把 NumPro 的同一帧号复制为 ASR/OCR 行的文本前缀，并要求一次输出 16 个分数；这是输入序列化/提示格式扩展，不是新的 hateful-localization 学习约束、表示机制或推理原理。 |

任一硬门失败即停止，因此总裁定为 **STOP**。

## 来源核查

NumPro 原论文的机制边界很明确：在每个视觉帧上覆写唯一数字，利用 Vid-LLM 的 OCR 能力把视觉内容与帧号关联，再把帧号写入语言答案。训练-free 版本只额外给出一句说明数字含义的 instruction；NumPro-FT 则用带编号视频和自然事件 temporal-grounding 问答进行微调。它评测 Charades-STA、ActivityNet 与 QVHighlights，任务是 moment retrieval / highlight detection，不是 hateful-video detection/localization。

Primary source：

- Wu et al., CVPR 2025, [*Number it: Temporal Grounding Videos like Flipping Manga*](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Number_it_Temporal_Grounding_Videos_like_Flipping_Manga_CVPR_2025_paper.html)。原文将 NumPro 定义为给视频帧覆写唯一数字，把 temporal grounding 转为视觉对齐；训练-free 与 fine-tuned 两种设置都只覆盖自然事件 VTG。
- [NumPro supplementary material](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Wu_Number_it_Temporal_CVPR_2025_supplemental.pdf)。报告 1 FPS/0.5 FPS 的自然事件 grounding 与 highlight-detection 配置，没有 hateful-video 实验。

目标任务检索覆盖 `NumPro / Number-Prompt / numbered frame / temporal indicator` 与 `hateful video detection / hate localization / HateMM / MultiHateClip / HateClipSeg` 的组合；截至审查日没有检出直接占用。因此第二门不能因 LELA 或 MultiHateLoc 的一般时间对齐而误判失败：它们没有使用 NumPro 的可见帧号 source core。

## 第三门为什么失败

### 1. 相对 NumPro：主要是直接换 query，再把编号复制到文本行

NumPro 已经完成候选最关键的操作：让模型读取覆写在视觉帧上的数字，并用数字把视觉内容映射到时间位置。当前候选保留相同操作，把自然事件 query 换成 hate policy query，并将 `[i]` 再写到同秒的 `SPEECH`、`SCREEN` 行前。

这个复制操作没有定义新的训练目标、结构化约束、cross-modal matching function 或可学习表示；数字与内容的对应关系由输入构造器直接给定。一次生成 16 个分数也只是把来源的 start/end frame answer 改成 dense score vector。因而相对来源，candidate 的完整方法可以准确描述为：

> NumPro visual frame numbering + timestamp-prefixed ASR/OCR serialization + dense hate-scoring prompt。

这是简单组件串接和任务 query/output-format 替换，落入 Rule 12 明确禁止的 trivial adaptation。

### 2. 相对 LELA：目标任务中的逐帧多 carrier composition 已被占用

LELA 已经在 hateful-video localization 中：

- 为 image、speech、OCR、music、video context 生成 frame-aligned captions；
- 在每个 frame `j` 显式把 `speech_j` 与其他 modality caption `C_j^m` 拼接；
- 用 hate-policy multi-stage prompting 得到逐帧、逐模态分数，再形成 frame-level hate profile。

Primary source：[Sun et al., *Towards Training-free Multimodal Hate Localisation with Large Language Models*](https://arxiv.org/abs/2602.09637)。其 composition matching 公式明确是同一 frame index 上的 speech 与其他 modality caption 拼接，而不是无时间归属的整段文本。

因此“视觉、speech、OCR 在同一个时间地址参与 hate 判断”不是本候选新增的任务机制。候选仅把 LELA 的隐式/数据结构 frame index 改写成模型可见的 `[i]` 字符，并把逐帧调用改成 16 帧一次调用。可见索引可能是有用的工程优化，但它没有改变目标任务中已存在的 evidence composition 原理。

### 3. 相对 MultiHateLoc：时间 join 已由 timestamp-to-frame expansion 明确实现

MultiHateLoc 已使用 Whisper sentence start/end timestamps，将 sentence embedding 扩展到相应 frame interval，再与 1 秒 audio 和 visual sequence 做 temporal modeling、dynamic fusion 与 weak MIL。Primary source：[Sun et al., *MultiHateLoc: Towards Temporal Localisation of Multimodal Hate Content in Online Videos*](https://arxiv.org/abs/2512.10408)。

当前 `[i]` 并没有估计未知对齐或纠正异步 carrier；它只是把上游已有 timestamp-to-second assignment 以文本形式再次呈现给 VLM。换言之，join relation 是预处理已知事实，candidate 不学习或推断它。

### 4. 相对旧 Qwen pointwise diagnostic：模型、局部块、timestamped ASR 与 hate query 都已存在

项目归档 `archive/experiments/20260831_qwen3_pointwise_test_diagnostic/README.md` 已固定同一 Qwen3-VL-8B、16 秒局部窗口、timestamped ASR 与局部 hate-policy scoring。它的 HMM/HCS within ROC 为 `.561760/.539628`，证明旧实现的窗口级单分数时间归属不足。

当前候选确实把输出粒度从“每窗一个 score”改成“每秒一个 score”，但新增实现仍是 NumPro 的标准 frame-address readout；`[i] SPEECH`/`[i] SCREEN` 只是将已有 timestamped carrier 放入相同编号模板。它没有引入区别于 pointwise VLM 的新监督或模型机制。

### 5. 与近期关闭链的关系

- **timestamp 链**：SafeLens 已在目标任务做 segment-level speech/text/visual fusion与 policy-LLM structured prediction；MultiHateLoc 已把 ASR timestamps 展开到 frame。项目的 ASR token-rationale transport 也因目标领域已有 timestamped segment hate reasoning 而关闭。候选没有新的 timestamp inference，只是显示已有秒号。
- **prompt 链**：LELA 已占用 frame-level multimodal hate prompting；旧 Qwen diagnostic 已占用 16 秒局部 Qwen hate scoring。把输出模板改为 16 个数字不形成新的 reasoning mechanism。
- **cross-modal alignment 链**：已有方法学习或显式构造同秒跨模态 alignment。候选的 join key 是人工给定的重复 identifier；它不解决 alignment estimation，只把已知 alignment 重新编码给模型。
- **program 链**：候选没有 Evidence-Program Graph 那类新的 structured state/path；固定的 16 项 JSON/列表输出只是一种 parse schema，不能作为结构化推理 novelty。

本候选不必与上述每个失败方法在数学上严格同构才会失败。Rule 12 允许的 pre-run STOP 条件之一就是 adaptation 明确只是 direct application / simple composition；当前候选满足这一条件。

## Controls 为什么不能挽救 novelty

`visual_numpro` 与 `address_permuted` 是合理的效果归因 controls：若 core 胜出，可以说明模型实际利用了文本中的编号，而不是只靠视觉帧号或自然顺序。但它们最多证明 **visible duplicated index 对当前 prompt 的性能是 load-bearing**。

它们不能证明 task adaptation non-trivial，因为：

1. core、control 都读取完全相同的已对齐 carrier 内容；区别只是是否显示或破坏一个由预处理直接给定的索引；
2. `visual_numpro` 中语言行仍按自然时间顺序排列，序列位置已经提供同一 join relation；数字是该关系的冗余显式编码；
3. `address_permuted` 制造的是自相矛盾的 metadata，而不是删除一个学到的 hateful-localization constraint；
4. 即使 core 显著更好，最窄结论仍是“Qwen 对显式编号的输入格式更稳”，不是新的 hateful temporal localization 方法机制。

机制 control 可以证实一个 trivial prompting delta 有效，但不能把该 delta 转换为 Rule 12 所要求的 non-trivial novelty。

## 最窄可保留 claim

若未来仅作为 baseline 或工程系统单独评测，可表述为：

> A NumPro-augmented, shared-index serialization baseline for dense multimodal hate scoring with a frozen VLM.

不能 claim：

- novel shared temporal-address mechanism for hateful-video localization；
- first cross-carrier temporal binding for hateful videos；
- first frame-addressed multimodal hate reasoning；
- 对 NumPro 的 non-trivial task adaptation。

## 最终决定

**STOP 4.8/10。Gate 1 PASS；Gate 2 narrow PASS；Gate 3 FAIL。**

来源未被目标任务直接占用，但新增部分只是把可见帧号复制到已 timestamp-aligned 的 ASR/OCR 行，并把 NumPro 接到 LELA/旧 Qwen 已有的逐帧 multimodal hate prompt。它可以是有用的 baseline/input-format experiment，但不满足项目的新 novelty 标准，不得作为 novel candidate 实现或晋级。
