# REJECTED PRE-RUN — ASR timestamp-fallback test diagnosis

截至 2026-08-31。只读 developmental test error analysis；不训练、不选 checkpoint、
不生成新 prediction。

## 动机

定性检查发现 HCS worst cases 中存在 100 秒以上的 ASR chunk，而 ASR producer 在 word
timestamp 推理失败时会退回 chunk timestamps。Qwen3 dense teacher 按 16 秒窗口选择重叠
ASR；超长 chunk 因而可能把同一段语义广播到大量窗口。该诊断只判断这一 alignment failure
是否值得重新生成统一 word alignment，不把 ASR mode 当作模型 routing 信号。

## 固定分析

- HMM/HCS 使用已完成的 Qwen3 test teacher predictions 与固定 test GT。
- 只保留同时含两类秒的视频，按共享 1fps grid densify window scores，再算逐视频 ROC-AUC。
- 从既有 ASR row 读取 producer 记录的 `timestamps` mode，并统计最长非空 chunk 时长。
- 报告 word/chunk 两组 macro AUC、差值，以及逐视频 AUC 与最长 chunk 时长的 Spearman。
- 进入统一 word-alignment premise 的固定 gate：两个 corpus 都要求
  `mean_auc(word)-mean_auc(chunk) >= .03` 且 Spearman `rho < 0`。任一失败即不把 timestamp
  fallback 作为共同性能瓶颈；不按 corpus 使用不同 ASR pipeline。

原计划输出为 `runs/20260831_asr_timestamp_failure_diagnosis/main/analysis.json`；因 pre-run
review 阻断，该文件未生成。

## Pre-run verdict

独立审查 `STOP`，未运行分析、未产生 `analysis.json`：

1. 候选脚本读取 `data/ASR/...test_seen_asrK4...jsonl`，但 Qwen producer 实际读取
   `results/reproduction/asr/{hatemm_all,hateclipseg_all}/timestamped_chunks.jsonl`。
   两套输入在 eligible cohort 中只有 HMM 19/85、HCS 20/67 的完整 chunk list 相同；
   前者的 word/chunk fallback mode 不能解释后者的 Qwen prediction。
2. 原 coverage 从 prediction 反推 cohort，检查恒真；逐视频 AUC 也没有调用共享 evaluator。
3. 即使改读实际输入，chunk length 与音频质量、视频长度、语音密度和 GT 结构混杂，只能形成
   descriptive association，不能证明或否定共同瓶颈。

因此不通过“修脚本后继续跑”恢复该诊断。只有预先定义、同一 frozen teacher 的 controlled
chunk-ASR vs word-alignment prediction 对照才可能回答因果问题；但该路线还必须先通过 novelty
来源门，不能把 alignment repair 本身称为方法。
