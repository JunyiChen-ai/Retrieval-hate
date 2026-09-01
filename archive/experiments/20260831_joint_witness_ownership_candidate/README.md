> **淘汰原因（2026-08-31）：独立查新 STOP，novelty 2.5/10；核心被 HAN、JoMoLD、PoiBin 与 MACIL-SD 占位，本质是已知 MMIL loss correction。未实现、未训练、未运行新 test prediction。**

# Joint witness ownership candidate

候选试图把 MultiHateLoc 对每个正例 modality branch 重复施加 video label 的训练语义，
改成 time×modality 联合 witness MIL。四语料 test error analysis 支持其问题动机，但独立
查新证明机制本身不新。详细证据与若要重开的必要差异见 `NOVELTY_REVIEW.md`。

