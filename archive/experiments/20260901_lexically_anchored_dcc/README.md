# Lexically anchored denoised cross-video contrast

> 已淘汰（2026-09-01）：完整 validation search 后的 HMM/HCS 双 test 未通过机制门或六项 SOTA 门；正确 lexical timing 只稳定胜错位 control，不能稳定胜 matched POWA anchor，且没有跨语料一致的 corrective 证据。

截至 2026-09-01。已完成实现、一次限定范围的跑前 technical review、完整 validation search、checkpoint selection 与 HMM/HCS test。

## Failure 与现有证据

Official MultiHateLoc 相对固定 SOTA threshold 的缺口为：HMM pooled AP/ROC/within
`+.100831/+.077925/+.003076`，HCS 为 `+.066350/+.060950/+.038207`。POWA 在
HMM 提供强 pooled carrier，但 within 弱；HCS 的 POWA 三项也未过门。独立的
train-video-label lexical locality 在 test 上有 HMM/HCS within `.632629/.522700`；固定
test complementarity 中 lexical+POWA 在 HMM、多个 signal+VERA 在 HCS 可同时过三项。
这只证明 headroom，不授权 inference blend。

实际可用的 correction signal 是每个语料独立生成的五折 OOF train lexical posterior：
它只由该语料 train video labels、ASR 文本和时间戳产生，不来自 POWA 的 top-K、branch
confidence 或 test GT。已有生成逻辑位于归档的 lexical posterior regularization 实验，可在
本 run 内重新生成并保留完整 provenance。

## 跨任务来源与 task adaptation

来源为 Li et al., CVPR 2022, *Exploring Denoised Cross-Video Contrast for
Weakly-Supervised Temporal Action Localization*。来源方法用 pseudo-label denoising、
region-level memory bank 和 cross-video contrast 改善弱监督 action/background 表示。

本 adaptation 不复制来源的 self-prediction clustering。对 positive train video，用 OOF
lexical posterior 的高置信、时间连续区域写入 hateful-region memory；对 negative train video，
全部有效区域写入 benign-region memory。POWA 的 AVT shared temporal representation 直接
归一化形成 region embedding，不设置可旁路 final scorer 的独立 projection head；采用跨视频
supervised contrastive objective：aligned
positive regions靠近其他 positive-video hateful regions并远离 negative-video benign regions；
其余 positive-video区域不被伪装成可靠 benign。该 loss 直接更新 POWA shared representation，
最终 `frame_prob` 仍由同一个 POWA policy head产生。

与来源相比，新增机制是用跨拟合的本任务 lexical locality 取代模型自生成 pseudo-cluster，
并采用 asymmetric memory：negative bags提供可信 benign regions，positive bags只提供
lexically supported hateful regions。这针对 hateful event 稀疏、positive bag内背景不可当负例、
以及当前 self-confidence 无法识别 local responsibility 的共同失败。

## 进入六项指标与 final score 的路径

- pooled AP/ROC：跨视频 hateful-region 与 negative-video benign-region separation 直接改变
  shared representation及同一 frame head，目标是提高跨视频正负秒分离。
- within ROC：只有真实时间对齐的 OOF lexical regions进入 hateful memory，促使 positive
  video 内对应局部表示高于未支持区域。
- inference 只输出训练后 POWA 的 raw `frame_prob`；不读取 lexical posterior、memory、
  teacher score、CDF、calibration、routing 或其他模型分数，不构成 inference ensemble。

## 可证伪预期、control 与训练选择

Matched control 使用完全相同的 POWA、representation normalization、memory budget、训练量与 loss，只把每个
train video 的 OOF lexical posterior作确定性 half-video circular shift。Core 必须在 HMM/HCS
test 的 within 都高于 shifted control，且两语料至少各有一个 pooled 指标同向提高；最终晋级仍
要求 HMM/HCS 六项全部超过固定 SOTA threshold。若 core 不满足双语料 performance gate，计
RESET6 第一次正式 failure，不通过调 anchor quantile、memory 或 contrastive loss 续命，除非
唯一 post-test error analysis支持 Rule 18 corrective。

若 novelty 通过，每个语料独立做完整 validation search：两个 learning rate
`{1e-4,2e-4}` × 三个 contrastive weight `{.05,.2,.5}` × 两个 lexical support quantile
`{.70,.85}`，共 12 个 core trial，另对两个 learning rate 各跑一个 matched POWA anchor。
每个 trial 跑 official POWA 的完整 5 epochs，并以 validation within、pooled AP、pooled ROC
的顺序联合选择 checkpoint。跨 trial 优先选择相对同 learning-rate anchor 的 validation pooled
AP/ROC delta 均不低于 `-.005` 的配置，再最大化 within；若没有配置满足约束，则最大化两项 pooled
delta 的较小者，再比较 within。两个语料全部锁定后，才训练各自 selected-config shifted
control，随后立即跑 HMM/HCS test。

## 参考与边界

- 来源论文：<https://openaccess.thecvf.com/content/CVPR2022/html/Li_Exploring_Denoised_Cross-Video_Contrast_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2022_paper.html>
- 目标任务占用检索需由独立 novelty reviewer裁定。不得 claim 首次在 hateful-video 中使用
  contrastive learning；最窄 claim 只能是 OOF lexical-anchor asymmetric region-memory adaptation。

## 正式结果与结论

权威结果为 `runs/20260901_lexically_anchored_dcc/formal_seed234/summary.json`。每个语料完成
2 个 matched anchor 与 12 个 aligned validation trials；配置和 checkpoint 锁定后训练同配置
shifted control，再立即评测 test。没有 smoke。

- HMM anchor/shifted/aligned 的 pooled AP、pooled ROC、within ROC 分别为
  `.584460/.804897/.596995`、`.589560/.803775/.573599`、
  `.581460/.800954/.583727`。Aligned 相对 anchor 为
  `-.003000/-.003943/-.013268`，相对 shifted 为
  `-.008100/-.002821/+.010128`。
- HCS anchor/shifted/aligned 分别为 `.575832/.545819/.516630`、
  `.509599/.466380/.506930`、`.581133/.559766/.526224`。Aligned 相对 anchor 为
  `+.005301/+.013947/+.009594`，相对 shifted 为
  `+.071535/+.093386/+.019294`。
- 两个语料的 aligned 都未超过各自三个固定 SOTA threshold；整体
  `mechanism_gate=false`、`hmm_hcs_all_sota=false`。

唯一一次 post-test error analysis 读取三个 test prediction artifact 与 test GT。按正例占比四分位
统计 per-video within delta 后，HMM aligned 相对 anchor 在前三个四分位均下降
`-.008643/-.007124/-.046421`，只在最高四分位上升 `+.008897`；HCS 四分位变化为
`-.033777/+.033234/-.004106/+.045114`，不是共同 correction pattern。HMM 平均视频内分数
标准差从 anchor `.049769` 降至 aligned `.037909`，且负帧均值增幅大于正帧，解释了 pooled 与
within 同时下降。正确 timing 胜 shifted 证明 lexical alignment 被模型使用，但相对原 POWA 的
方向不跨语料一致，因此不足以支持 Rule 18 corrective。

决定：关闭本轮 asymmetric lexical region-memory / shared-representation cross-video contrast
family；不继续扫描 loss、memory、support quantile、projection 或 contrast weight。本方法计
RESET6 第一次正式 performance failure。
