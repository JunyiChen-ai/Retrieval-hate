# Word-aligned lexical premise

截至 2026-08-31；这是输入信号 premise，不是候选方法，也不构成 novelty claim。

## 观察与假设

现有 HMM/HCS ASR cache 的 segment timestamp 有极长尾：长 chunk 会把一句话扩散到几十乃至数百秒，可能直接破坏 lexical score 的时间定位。固定假设是：Whisper word-level timestamp 能在不使用任何标签的情况下，产生严格有效、明显短于旧 chunk 的时间区间；随后把同语料 train video-label lexical model 的分数映射到这些词区间，应改善 test localization，尤其是 HCS。

## 冻结顺序

1. 严格 smoke：HCS 固定视频 `bit_Y4NcS9xwARDO` 的前 90 秒；不得从 word timestamp 降级为 chunk timestamp。
2. smoke 通过条件：至少 10 个非空 word；所有端点有限、`0 <= start <= end <= duration`；start 非递减；正时长 word 的中位长度小于 2 秒。
3. 只有通过后才生成所需语料的 test word-timestamp cache。ASR 不读取 GT 或标签；split 只决定处理哪些视频。当前 classifier 的 whole-transcript train 输入不变，因此无需重转录 train。
4. 新 ASR 只提供时间证据。通过单调 sequence alignment，把时间映射给旧 cache 中原 baseline 实际使用的有效 chunk tokens；送入 lexical classifier 的可用文本内容锁定不变（旧 loader 已丢弃的内容不会被重新引入）。无法 exact match 的旧 token 在相邻 matched anchors 间插值，并报告 exact-token match fraction。若旧有效文本存在但新 ASR 无词，则该视频使用旧 chunks + 旧 timing 的 identity fallback，不删除旧文本，也不给新 timing 贡献增益。
5. lexical classifier 仍只用该语料 train video labels，配置与现有 whole-transcript char n-gram probe固定一致。完成后立即在 test 上报告 pooled AP、pooled ROC、within-video macro ROC，并与旧 chunk lexical 对照。

正式 gate（每个语料分别满足）：test exact coverage、word-aligned within ROC 至少比旧 chunk lexical 高 `.020`、pooled AP/ROC 各自下降不超过 `.020`、仅允许 `OK/NO_AUDIO/EMPTY_SPEECH` 状态、loader 不丢弃任何异常 chunk、平均 exact-token match fraction 至少 `.70`。smoke 的至少 10 词门只用于固定有语音视频；正式集允许真实无音轨/无语音视频 fail-closed 为空。先跑 HCS；HCS 失败则停止，不消耗 HMM 全量计算。

本轮不按 test 选择 ASR 模型、语言、对齐粒度或 corpus-specific 分支。HMM/HCS 固定使用 cached `openai/whisper-large-v3`、英语、word timestamp。实现使用 Whisper 原生 long-form generation（`chunk_length_s=0`）；显式 30 秒 pipeline 滑窗在首次 smoke 中产生 60–69 秒重复且倒序的 overlap 文本，故被拒绝，不用排序掩盖。

## 运行

```bash
bash experiments/20260831_word_aligned_lexical_premise/run_smoke.sh
bash experiments/20260831_word_aligned_lexical_premise/run_hcs_test.sh
```

输出：`runs/20260831_word_aligned_lexical_premise/smoke/` 和 `runs/20260831_word_aligned_lexical_premise/hateclipseg_test/`。

## 结论

**STOP_AND_ARCHIVE。** 严格 smoke 在单个固定视频通过，但 HCS formal producer 前 8/79 个视频中只有 3 个 `OK`，5 个未通过 finite/positive/bounded/monotonic 合并的 strict word-timestamp validation。失败行未保留 raw chunks，因此不把五例进一步归为某一个子条件。冻结 gate 要求零 ASR error，因此在第一个 error 出现后已不可能晋级；为保留重复失败证据，运行至 8 个后终止。没有排序清洗、chunk fallback、test metric evaluation 或 HMM 扩展。权威 producer verdict：`runs/20260831_word_aligned_lexical_premise/hateclipseg_test/metrics.json`。
