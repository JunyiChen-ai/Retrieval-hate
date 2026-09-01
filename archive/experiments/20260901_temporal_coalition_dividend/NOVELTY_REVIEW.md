# Novelty review

截至 2026-09-01。Verdict：**STOP，4.8/10**。

- Gate 1 PASS：Harsanyi dividend 可以 adaptation。
- Gate 2 PASS：未发现 Harsanyi dividend 用于 hateful-video detection/localization；hateful-meme 的 Shapley 是静态 post-hoc attribution，不构成该任务占用。
- Gate 3 FAIL：当前 core 与归档 `20260831_coalition_witness_candidate` 的 `mobius_nonminimal` 严格同族。旧方法已经使用 7 个 modality coalitions、shared masked forward、正 interaction 聚合成唯一 frame score，再接 temporal MIL。新候选只是改写 interaction 参数化，没有新增 hateful-localization observation 或 constraint。

决定性历史 test：旧 `mobius_nonminimal` HMM/HCS within 为 `.6338/.5365`，未解决双语料 failure。禁止实现或训练该候选。
