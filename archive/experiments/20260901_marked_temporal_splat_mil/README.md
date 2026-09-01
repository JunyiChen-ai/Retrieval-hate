# Marked Temporal Splat MIL

截至 2026-09-01。第三次正式方法迭代已完成 HMM/HCS 独立训练与 test evaluation；机制门通过，但双数据集 performance gate 失败。连续失败方法计数因此达到 `3/3`，本实验停止并触发独立 process review。未运行 premise。

## Failure

MultiHateLoc/top-K MIL只把少数最高秒推成正证据；已有 HMM/HCS test 显示 structured scorer 的高分区相对真实 transition 中位膨胀 `4.5×/5.5×`，即大量孤立峰。固定 smoothing 在两语料方向相反，说明不能事后平滑完整 score；需要训练时把每个局部 evidence center 的支持范围显式写入最终 frame score。

## Source and adaptation

跨任务来源是 D'Amicantonio et al. ICCV 2025 的 Temporal Gaussian Splatting (TGS)：从 WSVAD score peaks 构造 Gaussian dense pseudo-label，缓解 top-K 只监督少量异常 snippets。检索未发现 TGS/temporal Gaussian splatting 用于 hateful-video detection/localization。

本候选不复制其 peak→pseudo-label self-training。Visual/audio/text 三个 modality encoders 在每秒直接输出 evidence amplitude 与固定多尺度 Gaussian bank 的 scale mixture；每个 `(time, modality)` 是一个 marked splat center。所有 centers 被可微渲染到 1fps 网格，三模态证据通过 noisy-OR 组成唯一 frame probability。于是 speech slur、visual symbol、OCR/text 可产生不同中心和持续范围；没有全视频 router、modality quota、per-branch positive MIL、teacher或 post-hoc smoothing。Negative bags 通过同一个 final MIL 压低所有伪 splats；positive bags只要求联合 splat field 存在 hate evidence。

相对来源的 non-trivial delta 是：从预测后生成自训练 target，改成单模型内可微的 `time × modality × duration` marked evidence representation；从 anomaly-class experts 改成异步 evidence carriers；从相加 pseudo-label改成概率 noisy-OR final localizer。Test只输出该单模型 raw splat field。

## Control and falsification

Matched control使用同一三模态 encoders、amplitude heads和参数预算，但每个 center 只作用于当前秒（delta kernel），即 ordinary point-evidence MIL；core唯一变化是 learnable fixed-bank duration mixture。Gaussian widths固定为 `1/2/4/8` 秒并对两语料一致，不做搜索。

Core必须在 HMM/HCS test within 同时胜 matched point control 与 MultiHateLoc anchor，至少一边 `>=+.020`，并保持 pooled SOTA可行性；否则记第三次失败方法迭代，立即触发 process review，不调 kernel bank、noisy-OR temperature或top-K续命。

来源：[D'Amicantonio et al., *Mixture of Experts Guided by Gaussian Splatters Matters*, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Amicantonio_Mixture_of_Experts_Guided_by_Gaussian_Splatters_Matters_A_new_ICCV_2025_paper.pdf)。

## Formal test result

权威汇总：`runs/20260901_marked_temporal_splat_mil/pilot_seed234/summary.json`。全部数字由冻结的 canonical evaluator 在 test split 输出。

- HateMM anchor / point control / splat 的 AP、pooled ROC、within ROC 分别为 `.492997/.738259/.628463`、`.471589/.726494/.601614`、`.495366/.740731/.727709`。Splat 相对 anchor/control 的 within 增益为 `+.099246/+.126095`。
- HateClipSeg 对应结果为 `.551339/.542726/.520588`、`.543328/.518907/.524428`、`.575469/.553826/.534081`。Splat 相对 anchor/control 的 within 增益为 `+.013493/+.009653`。
- 机制门通过：core 在两个语料都胜 anchor 与 matched point control，且 HateMM 增益超过 `+.020`。Duration field 确实进入最终 frame ranking，不是无效组件。
- Performance gate 失败：两个语料都没有同时越过三个固定 SOTA 门；HateMM 的主要改善集中在 within ordering，跨视频 pooled discrimination 仍不足，HCS 三项虽均胜 anchor，但增益不足以达到 SOTA。

## Decision

`ARCHIVE_THIS_RUN_KEEP_DURATION_MECHANISM`。不扫描 kernel bank、top-K、noisy-OR 或按语料选择配置；不把有效机制门误报成方法晋级。独立 process reviewer 的 `RESET` 裁定见`docs/PROCESS_REVIEW_RESET3_2026-09-01.md`：当前具体 run 归档，但 duration-field 记录为“机制成立、performance 未完成”，后续先做现有 test prediction error analysis，再允许一个由该证据直接支持的方法内迭代。
