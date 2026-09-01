# REJECTED — POWA within-video modality-desynchronization effect pilot

截至 2026-08-31。状态：validation-only、zero-training probe 未过冻结前提门，
直接淘汰；没有实施训练、没有读取 test。起点为 corpus-specific POWA。

## 机制假设

对同一视频做循环 temporal shift，只改变模态间对齐，不改变任一模态的内容集合
或 video label。若 POWA 在某秒的 policy evidence 真正依赖 hateful multimodal
co-occurrence，则原始同步输入相对多个错位输入的 logit effect 应在 hateful seconds
更高。这个 effect 不可使用 video-constant context，因 shift 前后该内容完全相同。

先在 HateMM 与 HateClipSeg validation 测四种 intervention：只移 text、只移
audio、只移 visual，以及 audio+text 同步移动（speech 相对 visual）。固定 shift
为序列长度约 `1/4` 与 `1/2` 的正负循环位移。probe 不训练、不读取 test、不选
checkpoint。

晋级前提：至少一个相同 intervention 在两个 corpus 上都达到 within ROC `>.53`，
且 original-plus-effect 不把 validation pooled AP/ROC 各降低超过 `.01`。否则机制
前提证伪，直接归档，不实施训练。

## 初步 novelty 边界

跨模态对应、audio-visual temporal localization、modality swapping 和 temporal
contrast 均已有充分先例。最近的直接先例包括 Wu & Yang, CVPR 2021
“Exploring Heterogeneous Clues for Weakly-Supervised Audio-Visual Video Parsing”，
其跨视频 audio/visual swapping 生成 modality-specific labels，并做视频内 temporal
contrast；Xia & Zhao, CVPR 2022 做 cross-modal background suppression；TANDEM
2026 也强调 hate video 的 cross-modal consistency。

因此不能 claim 上述通用概念。只有在实验证明必要时，潜在窄 claim 是：在
weakly-supervised hateful-video localization 中，用同视频、内容边际完全保持的
modality desynchronization 来估计 POWA policy-witness temporal effect，并把该
effect 作为训练时局部证据通道。若普通 shift augmentation 或任一单模态分数取得
同样结果，novelty/归因失败。

Primary sources:

- https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Exploring_Heterogeneous_Clues_for_Weakly-Supervised_Audio-Visual_Video_Parsing_CVPR_2021_paper.html
- https://openaccess.thecvf.com/content/CVPR2022/html/Xia_Cross-Modal_Background_Suppression_for_Audio-Visual_Event_Localization_CVPR_2022_paper.html
- https://openaccess.thecvf.com/content/ECCV2018/html/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.html

## 输出

权威 probe 输出：`runs/20260831_powa_sync_effect_pilot/val_probe.json`。

## 结果与裁定

HateMM POWA validation pooled AP/ROC/within 为
`.762920/.879167/.575085`。audio desynchronization effect 的 within 为
`.593225`，但 `POWA + effect` 把 pooled AP/ROC 降到 `.661169/.853895`；visual
effect within 只有 `.467572`。

HateClipSeg POWA 为 `.505909/.597609/.529684`。visual effect within
`.552146`，但相同 visual intervention 在 HateMM 反向；HCS audio effect within
只有 `.511792`。

没有同一 intervention 在两个 corpus 同时超过 `.53` within，也没有满足
`POWA + effect` 的 pooled 容差。机制前提证伪：hateful seconds 并不稳定对应某种
跨语料一致的 modality-synchronization effect。按预注册不进入训练实现与正式
review，候选淘汰。
