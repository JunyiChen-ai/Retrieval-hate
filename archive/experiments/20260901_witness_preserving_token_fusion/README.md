# Witness-Preserving Temporal TokenFusion

**截至 2026-09-01。RESET5 candidate 3 brief；尚未实现或运行。**

## Failure

MultiHateLoc 四语料 test evidence 显示，逐视频 global DMS 与最佳单模态匹配率只有
`.216/.333/.375/.323`，而 fused 胜过全部单模态的比例只有
`.345/.159/.042/.154`。RESET5 前两轮进一步排除 fused self-coalition credit 与
GCE shortcut-expert failure weighting：问题不能继续只靠全局选择、branch loss或训练权重修复。
本轮锁定 fused branch 在每一秒接收的三模态 token 内容。

## Cross-task source

Wang et al., *Multimodal Token Fusion for Vision Transformers*, CVPR 2022
(https://arxiv.org/abs/2204.08721)。来源方法检测不重要 token，并用对齐的其他模态投影
替换它们，以避免直接拼接稀释单模态内部信息；来源任务是图像转换、RGB-depth segmentation
和 image-point-cloud 3D detection，不是 hateful video detection/localization 或 temporal MIL。

## Task adaptation delta

MultiHateLoc 没有 transformer blocks，且正视频没有秒标签。本 adaptation 将来源机制变成
单层、逐秒、弱监督 witness-preserving substitution：

1. 每个原 MultiHateLoc modality embedding 产生逐秒 retain gate。对每个 recipient modality，
   其低-retain 部分由同一秒另外两个 modality 经独立 projection 后的 soft donor mixture替换；
   三个 substituted embeddings才进入原 fused MLP与唯一 raw `score_fused`。单模态 branch
   probability、原 MIL、smoothness、contrastive loss 与 training schedule保留。
2. 来源的 token sparsity 改为 latent-witness 约束：在 detached fused top-K positive witness
   support 上，`sum_m retain_m >= 1`，防止所有原始 token同时被替换而失去本秒的直接证据；
   support外用固定 retain-budget `.5` 的平方 penalty促使替换确实发生；coverage与budget
   的总权重固定为`.1`。它不产生正秒标签，只约束当前MIL witness的信息通路。
3. Aligned arm 只允许同秒 donor；matched shifted arm 在每个视频内把所有 donor circularly
   shift半个有效长度，但保留完全相同的 gates、projection、参数量、loss与训练预算。
   `alpha_fusion=0` 时不构造或调用 substitution参数，forward、raw score、基础loss和schedule
   精确退化同 harness MultiHateLoc anchor。

Task-specific load-bearing部分不是普通 cross-attention，而是“低-retain token可被替换，但
positive latent witness 每秒至少保留一个原始 carrier”的约束；它直接针对异步 speech/text/
visual carrier 下全局DMS无法逐秒修复融合输入的问题。

## Final-score path

`modality embedding -> per-second retain gate -> aligned other-modality projected substitution ->
original fused MLP/head -> raw score_fused`。

## Falsifiable test expectation and control

- Validation仅在固定方法内部联合选择超参数与checkpoint，随后立即HMM/HCS test。机制要求
  aligned 的 within-video ROC 在两语料都胜 matched shifted，且至少一边 `>= .010`；最终目标
  是两语料 AP、pooled ROC、within ROC 全部 SOTA。
- Matched shifted control只破坏 donor 的正确时间对齐，其他完全相同；另报告
  `alpha_fusion=0` exact anchor。
- 若 aligned 不在两语料共同胜 shifted，或增益只来自 shift/control本身的正则效应，关闭
  gate、projection、budget、shift与fusion-strength family，不做Rule18修补。

## Validation selection (pre-registered)

每语料训练两个 learning rate × 三个 `alpha_fusion` (`.25/.5/.75`) 的 aligned 与 shifted，
外加每个learning rate一个exact anchor，共14个完整trial。每个trial跑完整官方epoch预算并逐
epoch选择checkpoint。Aligned以validation within为主，相对同learning-rate、同alpha shifted
的pooled AP/ROC各最多下降`.01`；若均不可行，先最小化最大违反量，再最大化within。锁定
aligned及其matched shifted和same-lr anchor后立即test。禁止smoke、缩数据或缩epoch。

## Formal result and disposition

独立 novelty review 为 `GO 6.7/10`，唯一跑前基础 technical review 为 `PASS`。HMM/HCS
各完成14个完整validation-only trial，锁定matched chain后立即test。权威汇总为
`runs/20260901_witness_preserving_token_fusion/formal_val_selected_seed234/summary.json`。

- HateMM anchor/aligned/shifted AP、pooled ROC、within ROC 为
  `.490302/.737340/.632938`、`.491716/.735082/.632594`、
  `.492314/.737547/.632464`；aligned-minus-shifted 为
  `-.000597/-.002466/+.000130`。
- HateClipSeg 三臂为 `.523714/.497501/.525970`、
  `.523741/.499416/.527481`、`.522390/.497081/.525328`；
  aligned-minus-shifted 为 `+.001351/+.002335/+.002153`。

正确 donor timing 在两语料within方向都为正，但均远低于预注册的至少一边`+.010`，机制门
失败；两语料all-SOTA门也失败。Retain coverage penalty在两语料test witness上均为零，平均
retain gate约`.53/.54`，说明机制确实执行但性能作用不足。不调整gate、projection、budget、
shift或fusion strength；本family关闭并归档。RESET5达到`3/3`，必须先独立process review。
