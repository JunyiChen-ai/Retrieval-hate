# Shared visual–speech temporal-address VLM localization

> 淘汰原因（2026-09-01）：独立 novelty review 裁定 `STOP 4.8/10`。NumPro 来源未检出已进入 hateful-video task，但把可见帧号复制到已对齐的 ASR/OCR 行并输出 dense score vector，只是 NumPro 与 LELA/Qwen hate prompting 的输入序列化和简单组件串接，不构成 non-trivial task adaptation。未实现、未运行，不计正式 performance failure。

截至 2026-09-01。RESET6 第三个正式候选 brief；不做 premise。通过一次独立 novelty 三门后直接实现并正式评测。

## Failure 与来源

已有完整 test 证据显示，Qwen3-VL 对独立 16 秒窗口输出一个 hate score 时，HMM/HCS within ROC 仅
`.561760/.539628`：模型能读取局部内容，但一个窗口级结论不能稳定回答“具体哪一秒”。与此同时，HMM/HCS
六项缺口要求新方法既保留跨视频语义判断，又改善视频内时间归属。本候选不使用任何已有 localizer score、
teacher、test GT、ensemble 或 calibration。

跨任务来源是 Wu et al., CVPR 2025 *Number it: Temporal Grounding Videos like Flipping Manga* 的
Number-Prompt：把唯一数字直接写到每张视频帧上，使 Vid-LLM 把 timestamp prediction 转为读取视觉地址。
来源任务是 natural-language video temporal grounding，不是 hateful-video detection/localization。来源：
<https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Number_it_Temporal_Grounding_Videos_like_Flipping_Manga_CVPR_2025_paper.html>。

## Task adaptation 与唯一输出路径

普通 NumPro 只有视觉帧地址，而 hateful evidence 经常来自 speech、screen text 与画面的组合。对每个固定、
不重叠的 16 秒块，本方法建立一个共享地址表：第 `i` 秒的帧左上角写入高对比度 `[i]`，同秒 ASR 与 OCR
也分别写成 `[i] SPEECH: ...`、`[i] SCREEN: ...`。Qwen3-VL-8B-Instruct 一次读取这 16 张编号帧及对应地址表，
按固定 hate policy 定义输出恰好 16 个 `0..10` 分数；每个分数直接映回同一地址的 1 fps 秒。最后不足 16 秒
只输出真实秒，parse/inference failure 保留零分，不删除视频或重试 prompt。test 的唯一 frame score是
`score/10`，不做重叠平均、平滑、阈值、分支选择、score fusion 或后处理。

与来源相比，load-bearing adaptation 是**同一个可见数字同时成为视觉面板与语言 carrier 的外显 join key**：
模型不必依靠输入序列位置猜测一句 speech/OCR 属于哪张图，也不能只给整块一个判断。它针对本任务中
speech-only slur、screen-text meme、以及视觉 target 与语言 predicate 同秒或邻秒出现的局部归属问题；固定
policy prompt只定义 hate，不生成 proposition graph、prompt bank、counter-evidence branch 或 latent state。
块内模型仍可利用上下文，因此不声称数字本身识别 hate，只 claim shared temporal address 改善多 carrier
证据到秒的绑定。

## Formal evaluation、control 与否证

本方法没有训练参数、validation hyperparameter 或 checkpoint；16 秒块、字号、位置、模型、解码和 prompt
在打开本轮 test 输出前固定。Novelty GO 后一次基础 technical review，只检查地址映射、split/GT isolation、
长度、parse 与统一 evaluator。随后对 HMM/HCS 完整 test cohort 运行：

1. `shared_address` core：帧、ASR、OCR共用可见数字；
2. `visual_numpro` matched control：帧仍编号，但 ASR/OCR 只按自然顺序给出、不携带数字；
3. `address_permuted` mechanism control：保持帧、文本、数字集合和计算完全相同，只固定置乱语言行的地址。

三个 arm 使用同一 Qwen、prompt、块、解码预算与 evaluator。Core 必须在 HMM/HCS within 都胜
`visual_numpro` 与 `address_permuted`，至少一边 `>=+.020`；否则 shared cross-modal address 不 load-bearing。
正式 performance gate 是同一个 core 在 HMM/HCS 的 pooled AP、pooled ROC、within ROC 六项全部严格超过
固定 SOTA。失败记 RESET6 正式 performance failure `3/3` 并立即触发独立 process review；不换字号、块长、
prompt、Qwen版本或地址格式续命。

Novelty reviewer 必须重点核查：(1) NumPro/shared temporal indicators 是否已进入 hateful-video task；
(2) 相对 LELA 的五模态逐帧 prompting/composition matching 与旧 Qwen pointwise diagnostic，共享可见 join key
是否构成 non-trivial、load-bearing task adaptation，还是仅把 NumPro 直接换 query；(3) 是否与已关闭的
timestamp/program/prompt/cross-modal alignment 链严格同构。任一硬门失败即归档，不实现。
