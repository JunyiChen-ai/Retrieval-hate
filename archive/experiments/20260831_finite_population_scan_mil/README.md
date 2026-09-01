# Finite-population scan MIL — 淘汰：解析退化为相关性区间 selector

截至 2026-08-31。独立 review `STOP 3.4/10`；未实现、未训练、未生成新 prediction。

决定性反例：忽略 `eps` 时 `Z(I)=sqrt(T)*Corr(e,1_I)`；单帧 spike 与长度 `T-1` 的近整段
proposal都严格达到全局上界。hard selection时video variance又是所有候选共享的正乘数，不改变
interval argmax。实际机制只剩length-standardized contiguous hard-instance mining，未通过第三道
non-trivial adaptation硬门。

## 直接失败证据

multimodal P-MIL 在 HMM/HCS 的 proposal-oracle within ROC 为 `.73952/.63450`，但正式 frame
within 只有 `.58990/.47661`。HCS `48/79` 个视频最高分 proposal 是 whole-video，top proposal
长度中位数 213 秒；全 pair PCE/IRC 又让无证据 view成为 teacher。说明候选 interval 覆盖并非
主要瓶颈，弱 bag label 下的 proposal scoring 被 video topic、proposal长度和 whole-video shortcut
占据。

依据：`runs/20260831_multimodal_pmil_baseline/pilot_seed234/test_error_analysis.json`。该 test 证据
按现行规则属于 developmental evidence。

## 跨任务来源

候选拟从 epidemic change-point / multiscale scan statistics 适配“有限总体、区间对补集的标准化
excess”算子，而不是从 action completeness、outer-inner contrast 或普通 top-k MIL改名。来源
核心是否已进入 hateful video detection/localization，以及它与 WTAL completeness/context modeling
的实质边界，必须由独立 reviewer检索裁定。

## 核心机制

模型先产生单一原始逐秒 logit `e_t`。对视频长度 `T` 的候选区间 `I`，`1 <= |I| <= T-1`：

`delta(I) = mean_{t in I}(e_t) - mean_{t not in I}(e_t)`

`Z(I) = delta(I) / sqrt((1/|I| + 1/(T-|I|)) * (var_video(e) + eps))`。

它对应从同一个有限视频总体抽取区间与补集的 standardized mean contrast：

- video-wide topic/identity offset严格相消；
- 区间长度通过有限总体方差项比较，而不是由proposal元素个数或raw sum获益；
- whole-video proposal因补集为空在定义域外，不能成为正bag的退化witness；
- 候选相对的是同视频补集，不需要跨视频假设“另一个视频是语义匹配背景”。

该算子只在训练中选择/加权 positive-bag witness proposal。被选 interval 的 raw `e_t`仍需满足正
bag MIL，negative bags 对全部 raw frame logits施加标准 top-k negative loss，从而保留绝对 hate
方向；最终 test 输出就是单次 forward 的连续 `sigmoid(e_t)`，不在test做scan、proposal rasterize、
calibration、smoothing、routing或ensemble。

任务机制是：高正例率 hateful videos 中，video label容易被整段topic复制；标准化 interval-vs-
complement scan只决定“哪段比本视频其余部分更像正witness”，negative bags再决定绝对方向。
这不是声称 complement一定 benign，也不声称识别完整 hate span。

## 固定 controls 与 kill gate

Pilot仍为HMM/HCS分别训练、seed234，validation只在固定arm内选checkpoint，随后立即test三个固定
指标。最小arms匹配backbone、参数和预算：

1. `finite_population_scan`；
2. `raw_interval_mean`：去掉补集与方差标准化；
3. `mean_difference_only`：有补集但无长度方差项；
4. `length_penalty`：普通learned/fixed proposal length penalty，排除只因禁whole-video获益；
5. `topk_frame_mil`：无proposal scan的capacity-matched起点。

机制必须满足：core在HMM/HCS test within相对controls同向提高、至少一边 `>=.020`；top proposal
不能继续集中于最长允许区间，增益不能被固定center/length rule解释；同checkpoint把proposal的
complement对应关系在视频内循环错位应消除主要增益。两语料六个SOTA单元全部严格过门才扩
MHC-EN/ZH。

以下任一项即STOP：有限总体标准化在数学上等价已有outer-inner/completeness loss；弱标签可通过
人为增大video variance、单点spike或固定边界位置无代价过门；positive fraction接近1时补集噪声
使两语料方向不一致；raw/mean-difference/length control匹配或超过core；最终frame ranking几乎
不受scan训练影响。

## 独立 review 必答

1. scan-statistic核心是否已用于 hateful video detection/localization？
2. WTAL/WS-VAD是否已有数学等价的 interval-vs-complement variance-normalized proposal MIL？
3. `Z(I)`对 affine scale、单点spike、方差膨胀、极小补集和高正例率有何严格退化？
4. training-only proposal selector是否真正能改变最终 `e_t` ranking，还是普通top-k MIL换selector？
5. 什么最小premise/control可以在正式训练前证伪，而不把test scan readout包装成方法？
