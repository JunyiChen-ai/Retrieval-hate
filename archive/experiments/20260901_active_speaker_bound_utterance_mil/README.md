# Active-speaker-bound utterance MIL

截至 2026-09-01。RESET6 第三个正式候选 brief；不做 premise，先过一次独立 novelty 三门。

> **淘汰原因（2026-09-01）**：完整 validation 选参和 HMM/HCS 双 test 已完成。正确 physical source assignment 没有在两语料共同胜同容量 face-permutation control，六项固定 SOTA 全部失败；关闭该机制族，不调整 ASD 模型、阈值、track、face encoder、adapter 或 gate 续命。权威汇总：`runs/20260901_active_speaker_bound_utterance_mil/formal_seed234/summary.json`。

## 正式结果与结论

每个语料完成 2 个 matched anchor 和 6 个 core validation trial；HMM/HCS 均选择 `lr=1e-4, relation_weight=1.0`，随后以相同配置独立训练 permuted control 并立即评测 test。权威 test AP / pooled ROC / within ROC 为：

- HMM anchor=`.579832/.794955/.586322`，permuted=`.569762/.803181/.609872`，core=`.569911/.802566/.612938`。Core 相对 permuted 为 `+.000149/-.000615/+.003067`。
- HCS anchor=`.541681/.515035/.507398`，permuted=`.579362/.549037/.515496`，core=`.579061/.548190/.515489`。Core 相对 permuted 为 `-.000301/-.000847/-.000008`。

HMM 只有很小的 within ordering 差异，HCS 正确 assignment 三项均不胜 permutation；预注册的双语料 mechanism gate 失败。Core 对固定 SOTA 的 HMM 阈值 `.593832/.816184/.631532` 和 HCS 阈值 `.619371/.605022/.561908` 六项全败，performance gate 失败。本轮记 RESET6 第三个正式方法失败，窗口达到 `3/3`，触发独立 process review。

## Failure、来源与实际可用信号

HMM主要缺 pooled AP/ROC，HCS三项都缺；POWA在HMM提供强跨视频判别但within弱，在HCS也没有形成可靠局部排序。
现有 text-to-frame alignment 只回答一句话何时出现，不回答画面中的谁正在说话。新闻、访谈、reaction和多人画面中，
同秒全帧视觉与ASR直接融合会把被报道者、主持人、旁观者与真正发言者混成一个 evidence unit，可能把转述/报道误当
当前人物的endorsed hate。候选不使用test GT、已有localizer score、teacher、ensemble或calibration。

跨任务来源是 active speaker detection / audiovisual assignment：TalkNet用audio/visual temporal encoders、cross-attention
与long-term self-attention判断某条face track是否正在发声；MAAS把多个人脸与一个speech event显式建模为assignment问题。
来源任务是“谁在说话”，不是hateful-video detection/localization。Primary sources：
<https://arxiv.org/abs/2107.06592>；
<https://openaccess.thecvf.com/content/ICCV2021/html/Alcazar_MAAS_Multi-Modal_Assignation_for_Active_Speaker_Detection_ICCV_2021_paper.html>。

## Non-trivial task adaptation 与 final-score path

使用冻结TalkNet producer对每个目标语料train/validation/test视频生成face tracks及逐秒active-speaker posterior；producer不读
任何项目label。每个timestamped ASR utterance只形成一个source-bound token：若某face track在该utterance重叠区间具有唯一
最高且超过固定producer阈值的active posterior，token由该track的face-crop CLIP表示与utterance BERT表示组成；否则使用显式
`offscreen_or_ambiguous` source token与同一utterance表示，不删除speech。每秒可以同时保留独立的full-frame visual/audio/OCR
evidence，但**utterance不得再与整帧或非active face直接做cross-modal interaction**。

在corpus-specific POWA starting architecture中加入一个共享relation adapter，只读取`source-bound face × utterance`交互和
offscreen状态，将其写入同秒shared temporal representation，再由原POWA policy head、原bag loss与原`frame_prob`产生唯一test
score。原visual/audio/text输入、POWA forward、loss和schedule保留；adapter为零时精确退化为同harness POWA anchor。每个语料
只用自身train video labels训练，validation选超参数/checkpoint；TalkNet冻结且不接收hate gradient。

这不是给MultiHateLoc再加一个modality或ASD confidence。来源的speech-to-face assignment在本任务中被改造成一个**排他性的
evidence-binding constraint**：语言predicate只能与物理上归属的visible speaker face形成relation，不能与同帧其他人物或
video-global scene形成关系；offscreen speech则明确保持无face source。机制应同时减少跨视频新闻/报道false positive以改善
pooled separation，并把真实发言区间的relation token限制在局部秒以改善within ordering。不能claim active speaker detection、
source attribution或POWA本身新，也不声称active speaker自动等于hate speaker。

## Validation、formal test 与否证

Novelty GO后实现完整producer与模型，一次基础technical review后直接完整运行，不做smoke。每语料validation搜索6个core配置：
learning rate `{5e-5,1e-4}` × relation weight `{.25,.5,1.0}`，在官方完整epoch budget内以within为主、pooled AP/ROC相对同learning-rate
anchor各不低于`-.005`选择配置和checkpoint。锁定后训练同配置matched control并立即跑HMM/HCS完整test三指标。

Matched control保留完全相同的TalkNet outputs、face crops、utterance token、adapter、参数量和训练预算，但在每个视频内按固定
cyclic permutation把utterance分配给另一条candidate face track；single-face/offscreen utterance保持相同状态，只破坏可判定的
speech-to-face identity assignment。另报告adapter-off同harness anchor。Core必须在HMM/HCS within都胜permuted control，并在两语料
至少各改善一个pooled指标；至少一边within `>=+.020`。最终performance gate仍是core在HMM/HCS六项全部严格超过固定SOTA。
若失败，记RESET6正式failure `3/3`并立即触发process review，不换ASD模型、阈值、track算法、face encoder、adapter或gate续命。

## 实现与运行

实现使用官方冻结 TalkNet TalkSet checkpoint 和冻结 CLIP ViT-B/16 image projection。派生缓存由
`scripts/build_active_speaker_bound_cache.py` 写入 `data/active_speaker_bound/`；训练代码只读缓存。
face detector 固定按 25fps 网格每 5 帧运行一次，track box 插值回完整 25fps 后再运行 TalkNet；`minTrack=2`
保持原实现约 0.4 秒的最短 track 门槛。该 producer 设置预先锁定，不进入 validation search。
`method.py` 中 relation branch 只读取 assigned face、同秒 utterance 和 source state；matched control 只换成同视频
cyclic face identity。`relation_weight=0` 是同 harness adapter-off anchor，最终只导出原模型的 raw `frame_prob`。

完整运行入口是 `bash experiments/20260901_active_speaker_bound_utterance_mil/launch_formal.sh`。它依次完整生成
HMM/HCS cache，完成每语料 2 个 matched anchor 与 6 个 validation core trial，锁定配置后训练 permuted control，
然后立即在两语料 test 上调用共享 evaluator。没有 smoke 或缩量运行。

Novelty reviewer必须核查：(1) TalkNet/MAAS或speech-to-visible-face assignment是否已用于hateful-video task；(2) 排他性
source-bound evidence unit是否比“ASD feature + ordinary fusion”更强、且相对LELA/SafeLens/MultiHateLoc/MM-HSD与项目旧
source/stance graph形成独立、load-bearing机制；(3) offscreen/null路径或原POWA path是否使adapter完全旁路，从而只剩简单组件拼接；
(4) matched permutation是否真正隔离physical source assignment。任一硬门失败即归档，不实现。
