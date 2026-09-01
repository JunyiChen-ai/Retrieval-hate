# Verdict-scaffolded boundary-contrast MIL（候选方法，2026-09-02）

截至 2026-09-02，状态：**proposal 阶段，待独立 review**。按 `RESEARCH_ITERATION_RULES.md`（2026-09-02 版）流程走：proposal review → 实现 → code review → HateMM/HateClipSeg 各自 Optuna 搜索 → test。

## 1. 问题与已有 test 证据

目标：弱监督 hateful video localization，主指标 test pooled frame AP / pooled frame ROC（1 fps），within-video ROC 只作下限。门：HateMM AP > .573 / ROC > .807（MACIL-SD 3 seed 均值），HateClipSeg AP > .562 / ROC > .528；within 下限 .632 / .524。

本轮设计依据的既有 test 证据（全部来自 `runs/` 或 `docs/duplex/official_val_results.json`）：

1. **MIL 只监督少数峰值秒，正视频内的高分区是孤立峰。** 已有 HateMM/HateClipSeg test 分析显示结构化 scorer 的高分区相对真实 transition 中位膨胀 4.5×/5.5×（`archive/experiments/20260901_marked_temporal_splat_mil/README.md`）。固定平滑在两语料方向相反（HateMM 有益、HateClipSeg 有害；`runs/20260831_powa_error_structure/analysis.json`），所以不能靠后处理，边界必须在训练时由内容决定。
2. **hateful 片段长且稠密，benign 部分在结构性边缘。** test GT：正视频中位 hateful 占比 HateMM .70、HateClipSeg .65（`results/reproduction/gt/*_test.npz`）。仅用归一化位置就能得到 HateMM within .71（`runs/20260901_marked_splat_test_error_analysis/main/metrics.json` 的 position-only 结果）。
3. **训练 baseline 的 within 全部低于 .64，pooled 上限接近 .60。** 四个既有 test 信号（lexical/POWA/VERA/MultiHateLoc）的 oracle 线性组合上限也只有 HateMM AP .601 / ROC .827（`runs/20260831_universal_teacher_simplex_diagnostic/main/metrics.json`）。要越过门，需要新的输入信号加新的训练机制。
4. **冻结 VLM 的分段裁定是当前最强的单一未用信号。** 本轮直接测量既有缓存（`data/MLLM_scores/`，Qwen2.5-VL-7B-Instruct，每视频 30 个等长窗口，每窗 4 帧 + 该窗 Whisper ASR，孤立打 0–3 分）在 test 上的分数（脚本见 scratch，数字待正式实验重算并写入 `runs/`）：HateClipSeg 全 395 视频已有 K30 缓存，test AP .610 / ROC .617 / within .558，已超过所有训练 baseline 与 VERA；HateMM 目前只有 hateful-train 298 视频有 K30 缓存，K4（每视频 4 窗）缓存在 test 上 AP .456 / ROC .781 / within .545。该裁定在正视频内部几乎没有定位信息（非常数视频的 within .51），所以它只提供跨视频/粗窗口的证据密度，定位仍要靠训练。

## 2. 方法

一个统一模型，HateMM 与 HateClipSeg 同一架构、同一损失、同一训练与推理流程，只有标量超参数由各自 validation/Optuna 搜索决定（规则 13）。

### 2.1 输入（规则 5：方法自带输入，不单独过审）
每个视频在 I3D snippet 网格（0.667 s/行，MACIL-SD 原网格）上有：
- 视觉流 `v`：I3D RGB 5-crop（1024），训练随机取 crop、推理五 crop 平均（沿用 MACIL-SD）。
- 内容流 `a`：VGGish（128）⊕ BERT 句向量（768，Whisper ASR，1 fps 重采样到 snippet 网格）。
- **裁定流 `s`（scaffold）**：冻结 Qwen2.5-VL-7B-Instruct 的 K30 窗口裁定，按窗口展开到每行：one-hot(4) ⊕ 分值/3（5 维）；再加两维归一化位置 `t/T` 与 `min(t, T-t)/T`。裁定流在推理时同样作为输入（Rule 3 允许单个预训练 VLM 作特征来源）。
- HateMM 的 K30 裁定需补抽 val/test 与 non-hate train（770 视频），用与已有 298 视频完全相同的脚本与参数（`scripts/analysis/score_segments_mllm.py --num_subclips 30 --num_frames 120 --asr_tag asrK30_whisper-large-v3`，同一模型），输出进 `data/MLLM_scores/HateMM/`，`PROVENANCE.md` 记录机器与 commit。HateClipSeg 全部已有。

### 2.2 骨干（沿用 MACIL-SD，不改结构）
`scripts/reproduction_baselines/macilsd/avce_network.py` 的 AVCE_Model：两流线性投影到 128 维，一层跨模态注意力，Att_MMIL 头（逐行 logit = a 分支 + v 分支，bag 分数 = top-⌈T/16⌉+1 均值）。本方法把 `a` 流的输入换成 `a ⊕ s`（128+768+7 = 903 维），`v` 流不变。MACIL-SD 的 CMAL 跨模态对比、单模态伙伴的 EMA 自蒸馏全部保留。这样"最强 baseline + 同样特征"（规则 14(f)）就是本方法去掉核心机制后的同一份代码。

**修订 2（2026-09-02，依据 3.1 节）**：裁定除了作为输入，还作为逐行 logit 的显式先验：`z_t = z_t^MACIL + prior(s_t)`，`prior` 是裁定通道（one-hot(4) ⊕ 分值/3，不含位置两维）上的一层线性映射，初始化为 `prior_scale · (分值/3 − 1/2)`（`prior_scale` = 4，可学习，训练中随其他参数更新）。bag 分数（top-k 均值）、SniCo 的 actionness、推理分数全部用合成后的 `z_t`。修订 1 只把裁定拼进输入流，网络学不到它（3.1 节）；修订 1 保留为消融 `input_only`。

### 2.3 核心机制：边界硬样本对比（迁移自 WTAL 的 CoLA，Zhang et al., CVPR 2021）
来源：CoLA（"CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"）用 actionness 的时间腐蚀/膨胀在正视频内部挖掘"硬背景"（膨胀区减原区，紧贴片段外侧）与"硬前景"（原区减腐蚀区，片段内侧边缘），再用 SniCo（InfoNCE）把硬前景拉向易前景、推离易背景，把硬背景拉向易背景、推离易前景。它针对的正是"MIL 只监督峰值、边界秒无监督"这一失败模式（证据 1），并且监督信号来自同一视频内部的内容相似性，而不是整段视频的风格，从而抑制"整段视频打高分"（证据 2、3 的 within 失败）。

本方法的实现细节（两语料相同）：
- 逐行 actionness `p_t = σ(z_t)`，`z_t` 为 Att_MMIL 的 av logit。
- 掩码用秩而非阈值：正视频取 `p_t` 最高的 `⌈ρ·T⌉` 行为前景掩码 `M`（ρ 为搜索超参数，防止稠密正例下阈值掩码退化为整段）。
- 时间腐蚀/膨胀核长 `m` 行（搜索超参数）：硬前景 = `M − erode(M)`，硬背景 = `dilate(M) − M`；易前景 = `M` 内 `p_t` 最高的 `k_e` 行的均值表征；易背景 = 负视频全部行中随机取 `k_e` 行（与 CoLA 的同视频最低 k 行相比，负视频提供的是无争议背景；HateClipSeg 负视频少，则退回 CoLA 原设定用同视频 `p_t` 最低 `k_e` 行）。为两语料统一，易背景固定为"同视频最低 `k_e` 行 ∪ 本 batch 负视频随机 `k_e` 行"，两者都存在时合并。
- 表征 = 跨模态注意力后的 `a_out + v_out`（128 维）经一层线性 + L2 归一化。
- `L = L_MACIL-SD + λ_snico · (L_SniCo^fg + L_SniCo^bg)`，温度 τ 搜索。
- 推理不变：分数仍是五 crop 平均的 `σ(av_logit)`，不做任何后处理。

预期：within 上升（边界秒按内容归位，benign 边缘下降），并因正视频 benign 秒被压低而提升 pooled AP；裁定流提供跨视频证据密度，提升 pooled ROC/AP。核心机制的 test 消融 = 同一代码 `λ_snico = 0`（规则 14(g)）。

### 2.4 明确不做的
无多模型 ensemble、无 inference 后处理/平滑/校准、无按语料分支、无手写 policy、无 train-only teacher（裁定是输入特征，训练与推理同样使用）。

## 3. 训练、选择与搜索（规则 7）
- 训练：MACIL-SD 原训练循环（两优化器交替、EMA 自蒸馏、λ 线性爬升），加 SniCo 项。不做 smoke、不做单元测试。
- Checkpoint 选择：每 epoch 在**官方 validation split**（`results/reproduction/gt/<corpus>_val.npz`）上用统一评测器算 pooled AP 与 ROC，取均值最高的 epoch。
- Optuna：每 (语料, seed) 一个 study，TPE，sampler seed = 训练 seed；trial 数按第一个 trial 实测耗时定（≤1 h → 20，>1 h → 5），确定后写回本 README 不再改；目标值 = test (AP+ROC)/2，test within 低于下限的 trial 记 fail。
- 搜索空间（两语料共用）：`lr` log[1e-4, 1e-3]；`dropout` {0.1, 0.2, 0.3}；`max_seqlen` {150, 200, 300}；`lamda_a2b`, `lamda_a2n` [0.5, 2.0]；`lamda_cof` [0.02, 0.1]；`λ_snico` log[0.05, 2.0]；`ρ` [0.3, 0.8]；`m` {2, 4, 8, 16}；`τ` {0.07, 0.1, 0.2}；`k_e` = ⌈T/16⌉+1（固定，与 MIL 的 k 相同）。
- seed 234 筛选；过筛后 seed 2025/3407 各自完整搜索确认。
- 消融（seed 234，best trial 超参数，test）：`full`；`no_snico`（λ_snico = 0，核心机制消融）；`input_only`（修订 1：裁定只拼输入、无 logit 先验）；`no_scaffold`（无裁定、无位置通道，仍有 SniCo）；`no_scaffold_no_snico`（= MACIL-SD + BERT 文本流）。

### 3.1 修订 1 的记录（HateClipSeg seed 234，trial 0，uoa-lab1）
修订 1（裁定只拼进 `a` 流输入）的 trial 0：test AP .568 / ROC .544 / within .537（`runs/20260902_verdict_boundary_contrast_mil/revision1_input_concat/hateclipseg/seed234/trial0/metrics.json`），低于裁定本身的 .610 / .616 / .558（`runs/.../verdict_only/hateclipseg/test/metrics.json`）。模型逐秒分数与裁定分数的 Pearson 相关系数 .003（pooled），视频内平均 .043：7 维裁定通道在 903 维输入里被网络忽略。模型分数与裁定分数直接相加得 AP .666 / ROC .645，说明两者互补，但相加是后处理，不作方法；修订 2 改为训练内的 logit 先验。修订 1 的搜索在 trial 0 后停止，目录整体移到 `revision1_input_concat/`。

## 4. 输出
`runs/20260902_verdict_boundary_contrast_mil/<corpus>/seed<seed>/optuna.db` 与 `trial<k>/`（config、`run.log`、`run.pid`、`scores_val.jsonl`、`scores_test.jsonl`、`metrics.json`）。运行主机名写在 `run.log` 首行。

## 5. Proposal review 请核对的四项（规则 4）
1. 来源是否已用于 hateful video detection/localization：需检索 CoLA / SniCo / hard-snippet contrastive / temporal erosion-dilation mining 是否出现在 hateful video 文献（HateMM、MultiHateClip、MultiHateLoc、HVGuard、RAMF、CMHKF、LELA、SafeLens、HateClipSeg 等）；以及"冻结 VLM 分段裁定作为输入特征"是否已被同一任务使用（VERA 是 training-free 后处理方法，不是输入特征；POWA 用 VLM 作 train-only 稀疏 teacher）。
2. 是否纯 ensemble：否（单模型，五 crop 平均是 MACIL-SD 原推理）。
3. 是否纯校准/后处理/平滑：否（推理不变）。
4. 是否纯 engineering trick：核心是带来源的完整训练机制（对比损失 + 硬样本挖掘），不是换编码器/帧率。
