# Idea 候选池（2026-09-05，精简版；全文 `IDEA_REPORT.md`）

| # | Idea | Pilot | Novelty | Reviewer | 状态 |
|---|---|---|---|---|---|
| 1 | 裁定条件化密度估计 = C7 coarse-first 条件标注模型 + C1 scale–rank 分解头（C13 文本路由作子开关） | 未测（needs manual pilot；先 0 GPU CPU 门） | C1 6/10、C7 4/10（模块 6/10）、C13 5/10 | 3/10 → 修订后 5/10（Weak Reject） | RECOMMENDED |
| 2 | C9-T6 秒级 IOHMM（内容变化驱动转移，裁定作 OR 发射） | 未测（有 0 GPU 预检） | 未单独复核（先例 IOHMM / CHMM / HSLA） | 分诊融合第 2 | BACKUP |
| 3 | C4 只收不发的可靠性注意力 | 未测 | 未单独复核（DDG-Net 几乎相同） | 分诊骨干第 3 | BACKUP |
| 4 | C10-X10 基数 CRF；C8 精度加权 PoE | 未测 | — | 分诊降级 | BACKUP（记录） |
| — | C2 query 门控 CMA、C5 NOP sink、C11 证据 HMM、C12 双记忆、C14 密度协变量先验、C3-R5、C6-T3、C10-R6 | — | — | 分诊淘汰 | ELIMINATED |

## 当前 Idea
**进行中**：#1 裁定条件化密度估计 → `docs/idea-stage-20260905/docs/research_contract.md`
- 假设：把两粒度可靠性校正搬进标注模型（细裁定以粗裁定为条件）并把视频级密度做成可识别的加性标量，能在 pooled 上超过候选 1，且推理时不改变视频内排序。
- 关键证据（项目内测得）：K4-only 后验 .591/.851 对全证据 .541/.818；92%/79% 分数方差在视频间；注意力只作用 pooled；no_text ROC +.027 / within −.036。
- 下一步：Block A CPU 门 → 规则 4 复核 → 实现。

## 已淘汰（一行原因）
C2：候选 4 的 `gated_cma` 臂已在 HCS 变差。C5：`zero_value_sink` 臂零结果，padding 工程。C11：mass 不可标定，被 C7/C8 覆盖。C12：UR-DMU 直接迁移 + 循环伪标签。C14：裁定计数双重计入，视频级校准。C3-R5：候选 3 的块尺度广播。C6-T3：HVGuard 已用 MoE。C10-R6：只改 pooling 配置。
