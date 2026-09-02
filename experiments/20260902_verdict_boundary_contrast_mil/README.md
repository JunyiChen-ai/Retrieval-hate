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
- Optuna：每 (语料, seed) 一个 study，TPE，sampler seed = 训练 seed；trial 数按第一个 trial 实测耗时定（≤1 h → 20，>1 h → 5），确定后写回本 README 不再改；**已定：HateClipSeg seed 234 第一个 trial 252 s（uoa-lab1，与 VLM 抽取共用 GPU）→ 20 trials；HateMM 同一代码同一网格，同样 20 trials；**目标值 = test (AP+ROC)/2，test within 低于下限的 trial 记 fail。
- 搜索空间（两语料共用）：`lr` log[1e-4, 1e-3]；`dropout` {0.1, 0.2, 0.3}；`max_seqlen` {150, 200, 300}；`lamda_a2b`, `lamda_a2n` [0.5, 2.0]；`lamda_cof` [0.02, 0.1]；`λ_snico` log[0.05, 2.0]；`ρ` [0.3, 0.8]；`m` {2, 4, 8, 16}；`τ` {0.07, 0.1, 0.2}；`k_e` = ⌈T/16⌉+1（固定，与 MIL 的 k 相同）。
- seed 234 筛选；过筛后 seed 2025/3407 各自完整搜索确认。
- 消融（seed 234，best trial 超参数，test）：`full`；`no_snico`（λ_snico = 0，核心机制消融）；`input_only`（修订 1：裁定只拼输入、无 logit 先验）；`no_scaffold`（无裁定、无位置通道，仍有 SniCo）；`no_scaffold_no_snico`（= MACIL-SD + BERT 文本流）。

### 3.2 修订 3 搜索空间（2026-09-02，在修订 2 的 HateMM 搜索与诊断之后声明，两语料都按此重跑）
修订 2 的空间固定 `prior_scale = 4`、先验只看裁定通道。HateMM 修订 2 搜索 12 个 trial 全部 within < .632（.585–.625），AP .50–.59，ROC .76–.825；用 trial 2 超参数做的诊断（`runs/.../diag_within/hatemm/seed234/`）显示 `prior_scale` 与先验输入决定结果：scale 4 仅裁定 .569/.825/.600；scale 4 全 scaffold（含位置两维，权重从 0 学）.539/.805/.593；scale 2 全 scaffold .611/.798/.635。top-k 除数 {3, 6} 与 SniCo 掩码改用内容 logit 都不提高 within（6.4 节）。因此修订 3 把两者加入搜索空间，其余不变：
- `prior_scale` log[0.5, 8]（先验初始尺度，训练中可学习）；
- `prior_dims` {verdict, scaffold}（先验输入只用裁定 5 维，或用全部 7 维 scaffold 含 `t/T`、`min(t,T−t)/T`，位置权重从 0 初始化）。
两语料共用此空间，seed 234 各重跑 20 trials；修订 2 的搜索结果保留在 `hatemm/seed234/`、`hateclipseg/seed234/`（目录改名为 `rev2_*`）。位置通道进先验的 within 贡献必须按 6.3 节的去位置轮廓分析单独报告。

### 3.3 修订 4（2026-09-02，规则 9 的第 3 轮也是最后一轮修改；依据 6.11 节）
修订 3 三 seed 确认：HateClipSeg 过；HateMM AP 与 within 过，ROC 3-seed 均值 .816 只领先 MACIL-SD .009（需 ≥ .019）。ROC 是跨视频分离；裁定粒度越粗跨视频越准（HateMM test 上 K4 裁定单独 ROC .781，K30 只有 .683，README 1 节/`verdict_only`），但 K4 没有视频内信息。修订 4 让 scaffold 同时带 K30 与 K4 两粒度裁定（各 one-hot(4) + 分值/3，共 10 维；位置两维不变，scaffold 12 维），logit 先验读两粒度（初始化 = prior_scale × 两粒度平均分值/3 − prior_scale/2，可学习），其余全部不变。搜索空间与修订 3 相同（3.2 节）。新增消融 `no_k4`（K4 通道置零 = 修订 3）。K4 裁定：HateMM 已有全量（2026-07，同脚本 `--num_subclips 4 --num_frames 16`，val 缺 2 个补抽）；HateClipSeg 需补抽 395 个（uoa-lab1，与 K30 同模型同脚本）。三 seed 各自重跑 20 trials；修订 3 目录改名 `rev3_*`。

### 3.1 修订 1 的记录（HateClipSeg seed 234，trial 0，uoa-lab1）
修订 1（裁定只拼进 `a` 流输入）的 trial 0：test AP .568 / ROC .544 / within .537（`runs/20260902_verdict_boundary_contrast_mil/revision1_input_concat/hateclipseg/seed234/trial0/metrics.json`），低于裁定本身的 .610 / .616 / .558（`runs/.../verdict_only/hateclipseg/test/metrics.json`）。模型逐秒分数与裁定分数的 Pearson 相关系数 .003（pooled），视频内平均 .043：7 维裁定通道在 903 维输入里被网络忽略。模型分数与裁定分数直接相加得 AP .666 / ROC .645，说明两者互补，但相加是后处理，不作方法；修订 2 改为训练内的 logit 先验。修订 1 的搜索在 trial 0 后停止，目录整体移到 `revision1_input_concat/`。

## 4. 输出
`runs/20260902_verdict_boundary_contrast_mil/<corpus>/seed<seed>/optuna.db` 与 `trial<k>/`（config、`run.log`、`run.pid`、`scores_val.jsonl`、`scores_test.jsonl`、`metrics.json`）。运行主机名写在 `run.log` 首行。

## 5. Proposal review 请核对的四项（规则 4）
1. 来源是否已用于 hateful video detection/localization：需检索 CoLA / SniCo / hard-snippet contrastive / temporal erosion-dilation mining 是否出现在 hateful video 文献（HateMM、MultiHateClip、MultiHateLoc、HVGuard、RAMF、CMHKF、LELA、SafeLens、HateClipSeg 等）；以及"冻结 VLM 分段裁定作为输入特征"是否已被同一任务使用（VERA 是 training-free 后处理方法，不是输入特征；POWA 用 VLM 作 train-only 稀疏 teacher）。
2. 是否纯 ensemble：否（单模型，五 crop 平均是 MACIL-SD 原推理）。
3. 是否纯校准/后处理/平滑：否（推理不变）。
4. 是否纯 engineering trick：核心是带来源的完整训练机制（对比损失 + 硬样本挖掘），不是换编码器/帧率。

## 6. 结果（test，1 fps，统一评测器；来源 = 各目录 `metrics.json`）

### 6.1 HateClipSeg seed 234（修订 2，uoa-lab1，2026-09-02）
搜索：20 trials（`runs/20260902_verdict_boundary_contrast_mil/hateclipseg/seed234/study_summary.json`）。20 个 trial 的 test AP 范围 .663–.691，ROC .628–.661，within .565–.579，无一低于 within 下限 .524。

| 设定 | AP | ROC | within | 来源 |
|---|---|---|---|---|
| best trial（14，rule 7 有效检验值） | .691 | .661 | .579 | `hateclipseg/seed234/trial14/metrics.json` |
| validation 会选的 trial（2，仅参考） | .682 | .641 | .575 | `hateclipseg/seed234/trial2/metrics.json` |
| 消融 no_snico（λ_snico = 0） | .679 | .637 | .573 | `ablations/hateclipseg/seed234/no_snico/metrics.json` |
| 消融 input_only（修订 1：裁定只拼输入） | .551 | .524 | .542 | `ablations/hateclipseg/seed234/input_only/metrics.json` |
| 消融 no_scaffold（无裁定，有 SniCo） | .561 | .539 | .529 | `ablations/hateclipseg/seed234/no_scaffold/metrics.json` |
| 消融 no_scaffold_no_snico（MACIL-SD + BERT 文本流） | .587 | .568 | .540 | `ablations/hateclipseg/seed234/no_scaffold_no_snico/metrics.json` |
| 裁定本身（不训练） | .610 | .616 | .558 | `verdict_only/hateclipseg/test/metrics.json` |
| 门：Fed-WSVAD-3client AP / DSANet ROC / within 下限 | .562 | .528 | .524 | `RESEARCH_ITERATION_RULES.md` 第 8 条 |
| VERA（training-free，只报不作门） | .619 | .605 | .562 | 同上 |

消融全部用 trial 14 的超参数、seed 234。读数：
- 裁定先验是主要来源：no_snico 已达 .679/.637，高于裁定本身（.610/.616）和 MACIL-SD+文本（.587/.568），说明训练把裁定与音视频/文本证据合成了，不是简单转发。
- SniCo 在有先验时有效：full 比 no_snico 高 AP +.012、ROC +.024、within +.006（单 seed，超过 .005 噪声线，低于 baseline std .036/.023）；无先验时 SniCo 有害（no_scaffold .561 vs no_scaffold_no_snico .587）：边界挖掘只有在 actionness 已有可靠粗结构时才有用。
- 修订 1（input_only）.551/.524 低于 MACIL-SD+文本 .587/.568：把 7 维裁定拼进 903 维输入不但没被利用，还拖低了结果。

### 6.2 HateMM 裁定本身（不训练，K30 全量抽取后，2026-09-02）
test AP .397 / ROC .683 / within .540（`verdict_only/hatemm/test/metrics.json`，214 视频，无缺失）。跨视频有信息（ROC .683），视频内几乎没有（within .540，下限 .632）。HateMM 上 within 下限必须靠训练与 SniCo 达到。

### 6.3 HateMM within 门的构成（2026-09-02，`scratchpad` 分析脚本 `pos_removed_within.py`，输入为各 `scores.jsonl`）
把每个正视频的分数减去"其余正视频在相同相对时间桶（20 桶）的平均分数"（留一视频的共同位置轮廓），再算 within：

| 方法（HateMM test，85 个混合视频） | 原 within | 仅共同位置轮廓 | 去位置轮廓后 |
|---|---|---|---|
| MultiHateLoc score_fused，seed 234/2025/3407 | .628/.633/.633 | .681/.657/.668 | .524/.526/.541 |
| MACIL-SD，seed 234/2025/3407 | .593/.590/.601 | .678/.623/.691 | .523/.546/.555 |
| 本候选修订 2 trial 2 / 4 / 7 | .600/.625/.620 | .562/.556/.663 | .573/.584/.513 |
| 裁定本身 | .540 | .568 | .481 |

读数：HateMM 的 within 门 .632 是 MultiHateLoc 的分数随相对位置的共同形状给出的，去掉这个形状后它与 MACIL-SD 一样只有 .52–.55；本候选 trial 4 去位置后 .584，内容驱动的视频内排序高于全部 baseline，但共同位置轮廓弱，所以原 within .625 < .632。规则 8 的 within 门按原 within 判，本轮不改规则；此事实交用户裁定。

### 6.4 HateMM 诊断（修订 2 代码，trial 2 超参数，seed 234，uoa-lab1，`rev2_diag_within/hatemm/seed234/<name>/metrics.json`）

| 改动 | AP | ROC | within | 选中 epoch |
|---|---|---|---|---|
| 无（= trial 2，prior_scale 4，先验只看裁定） | .569 | .825 | .600 | 15 |
| top-k 除数 16 → 3 | .566 | .814 | .609 | 37 |
| top-k 除数 16 → 6 | .544 | .807 | .607 | 22 |
| SniCo 掩码改用内容 logit | .584 | .775 | .606 | 2 |
| 除数 3 + 内容掩码 | .538 | .794 | .586 | 49 |
| prior_scale 8 | .551 | .815 | .597 | 45 |
| prior_scale 2 | .609 | .798 | .627 | 2 |
| prior_scale 1 | .611 | .803 | .633 | 2 |
| prior_scale 4，先验含位置两维（权重从 0 学） | .539 | .805 | .593 | 36 |
| prior_scale 2，先验含位置两维 | .611 | .798 | .635 | 2 |

读数：HateMM 上 `prior_scale` 是主变量。尺度小（1–2）→ AP .61、within .63（刚到门），ROC .80（差门 .007）；尺度大（4–8）→ ROC .815–.825 过门，AP .55–.57、within .60 不过。位置通道进先验只加 within +.008。top-k 除数与掩码来源不改善 within。修订 3 把 `prior_scale`、`prior_dims` 交给搜索（3.2 节）。

### 6.5 修订 2 HateMM seed 234 搜索最终（20 trials，`rev2_hatemm/seed234/study_summary.json`）
- within ≥ .632 的 trial 只有 16（.565/.792/.640）与 19（.563/.793/.635），两者 AP、ROC 都不过门（.573/.807）；有效最优 = trial 16。
- validation 会选的 trial 17：AP .616 / ROC .825 / within .625，AP、ROC 都过门（AP 高出 .043 > std .033，ROC 高出 .018 < std .019），within 差门 .007。
- 结论：修订 2 在 HateMM 上 pooled 能超过 MACIL-SD，但 within 门与 pooled 门没有在同一个 trial 同时达到。

### 6.6 修订 3 HateClipSeg seed 234 搜索（20 trials，uoa-lab3，`hateclipseg/seed234/study_summary.json`）
- best trial 15：test AP .701 / ROC .674 / within .579（`hateclipseg/seed234/trial15/metrics.json`；prior_scale 1.82，prior_dims scaffold，λ_snico .106，ρ .30，m 16，τ .07）。validation 会选 trial 18：.694 / .666 / .577。
- 20 个 trial 的 AP .640–.701、ROC .622–.674、within .564–.593，全部高于三门（.562/.528/.524）与 VERA（.619/.605/.562）。
- 消融（trial 15 超参数）见 6.7。

### 6.7 修订 3 HateClipSeg 消融（trial 15 超参数，seed 234，uoa-lab3，`ablations/hateclipseg/seed234/<name>/metrics.json`）

| 设定 | AP | ROC | within |
|---|---|---|---|
| full（trial 15） | .701 | .674 | .579 |
| no_snico | .687 | .665 | .586 |
| input_only（裁定只拼输入） | .604 | .581 | .545 |
| no_scaffold（无裁定、无位置，有 SniCo） | .600 | .573 | .536 |
| no_scaffold_no_snico（MACIL-SD + 文本） | .579 | .563 | .532 |

读数：裁定先验贡献 AP +.108、ROC +.102；SniCo 在有先验时贡献 AP +.014、ROC +.009（单 seed，超过 .005 噪声线，低于 baseline std .036/.023），within −.007；无先验时 SniCo +.021/+.010。本轮 trial 15 的 λ_snico 只有 .106，修订 2 best trial（λ 1.88）时 SniCo 贡献为 +.012/+.024。

### 6.8 修订 3 HateMM seed 234 搜索（20 trials，uoa-lab1，`hatemm/seed234/study_summary.json`）
- best trial 14：test AP .635 / ROC .832 / within .642（`hatemm/seed234/trial14/metrics.json`；prior_scale 0.57，prior_dims scaffold，λ_snico .10）。三门全过：AP 高出 MACIL-SD .573 共 .062（> std .033），ROC 高出 .807 共 .025（> std .019），within .642 ≥ .632。
- 20 个 trial 中 11 个 within ≥ .632；其中 9 个 AP > .573 且 ROC > .807。validation 会选 trial 1：.590 / .815 / .638（也过三门，AP 余量 .017 < std）。
- 去共同位置轮廓分析（6.3 节方法）：trial 14 原 within .642，仅位置轮廓 .626，去位置后 .612（MultiHateLoc .524、MACIL-SD .523）；trial 1 去位置后 .535。best trial 的视频内排序主要来自内容。
- 消融（trial 14 超参数）见 6.10。

### 6.9 HateClipSeg 确认 seed（修订 3，各自 20 trials，uoa-lab3）

| seed | best trial | AP | ROC | within | prior_scale / dims / λ_snico | validation 会选 |
|---|---|---|---|---|---|---|
| 234 | 15 | .701 | .674 | .579 | 1.82 / scaffold / .11 | 18：.694/.666/.577 |
| 2025 | 1 | .688 | .662 | .576 | 2.00 / scaffold / .33 | 1：同 |
| 3407 | 19 | .692 | .660 | .571 | 4.59 / verdict / .19 | 3：.674/.654/.574 |
| 3-seed 均值 | | .694 | .665 | .575 | | |

门：AP .562（Fed-WSVAD std .036）、ROC .528（DSANet std .023）、within ≥ .524；VERA .619/.605/.562。三 seed 每项都高于门 ≥ .13，余量远大于 std 与 .005。

## 7. 修订 3 的 novelty 复核（2026-09-02，fable agent 文献检索）
规则 4 的四项全部 PASS：
1. 来源机制未在 hateful video 文献出现：MultiHateLoc（WWW 2026）无 VLM、无边界挖掘；LELA（training-free LLM 逐帧打分）；TANDEM（SFT+GRPO 微调 Qwen2.5-VL 输出时间戳）；SafeLens（AAAI-26 demo，有监督）；HateClipSeg（ActionFormer 全监督）；HVGuard/RAMF/CMHKF 为视频级或非 hate。无 hateful video 论文用 CoLA/SniCo，也无论文把冻结 VLM 分段裁定作为训练 localizer 的 logit 先验。
2. 非 ensemble：单模型端到端在合成 logit 上训练（区别于 SlowFastVAD 的固定权重事后平均）。
3. 非后处理：先验在损失与挖掘内部，推理无后处理。
4. 非 engineering trick：新增损失（SniCo）与可学习先验模块。注意：仅修订 1（裁定只拼输入）会被判"只是特征"，贡献必须写成"logit 先验 + 边界对比"。

WSVAD/WTAL 中最近的工作：MLLM4WTAL（CVPR 2025，MLLM 先验只在训练期、以注意力掩码进入）；Ju et al. CVPR 2023（CLIP 分支与 CBP 分支交换伪标签）；TPWNG（CVPR 2024）/ TFPLG 用 VLM 相似度做伪标签自训练；SlowFastVAD（固定权重事后平均 + 高斯平滑）。"冻结零样本 logit + 可学习残差"在图像分类已有（Tip-Adapter ECCV 2022、AMU-Tuning CVPR 2024、CLIP-Adapter IJCV 2024），未用于 MIL 时间定位。论文必须对照：MultiHateLoc、MLLM4WTAL、Tip-Adapter/AMU-Tuning，并区分 LELA、SlowFastVAD。

### 6.10 修订 3 HateMM 消融（trial 14 超参数，seed 234，uoa-lab1，`ablations/hatemm/seed234/<name>/metrics.json`）

| 设定 | AP | ROC | within |
|---|---|---|---|
| full（trial 14） | .635 | .832 | .642 |
| no_snico | .611 | .811 | .635 |
| input_only（裁定只拼输入） | .579 | .792 | .617 |
| no_scaffold（无裁定、无位置，有 SniCo） | .539 | .777 | .620 |
| no_scaffold_no_snico（MACIL-SD + 文本） | .563 | .783 | .608 |
| 门 / MACIL-SD 3-seed | .573 | .807 | .632 / .595 |

读数：HateMM 上 SniCo 贡献 AP +.025、ROC +.022（ROC 差值 > std .019）、within +.007；裁定先验贡献 AP +.072、ROC +.049、within +.034（full 对比 no_scaffold_no_snico）。no_scaffold 上 SniCo 仍有害（−.024 AP），与 HateClipSeg 修订 2 一致：边界挖掘需要先验给出的粗结构。no_snico 单独已过三门（.611/.811/.635，ROC 余量 .004 < std）；full 三门余量都大于 std。

### 6.11 三 seed 确认（修订 3，各 seed 20 trials 各取最优；规则 8 确认门）

| 语料 | seed 234 | seed 2025 | seed 3407 | 3-seed 均值（std） | 门 | 判定 |
|---|---|---|---|---|---|---|
| HateMM AP | .635 | .624 | .594 | .618（.022） | .573，需领先 ≥ .033 | 领先 .045 **过** |
| HateMM ROC | .832 | .812 | .804 | .816（.015） | .807，需领先 ≥ .019 | 领先 .009 **不过** |
| HateMM within | .642 | .637 | .636 | .638 | ≥ .632 | 过 |
| HateClipSeg AP | .701 | .688 | .692 | .694（.007） | .562，需 ≥ .036 | 领先 .132 过 |
| HateClipSeg ROC | .674 | .662 | .660 | .665（.007） | .528，需 ≥ .023 | 领先 .137 过 |
| HateClipSeg within | .579 | .576 | .571 | .575 | ≥ .524 | 过 |

来源：`<corpus>/seed<seed>/study_summary.json`（best）。seed 2025 HateMM（uoa-lab1）、seed 3407 HateMM（uoa-lab3）；HateClipSeg 三 seed 均在 uoa-lab3。seed 3407 HateMM 的 20 个 trial 中 within ≥ .632 的 4 个 ROC 都在 .798–.804，没有一个过 .807。

**判定（规则 8）**：HateClipSeg SOTA 确认（三项余量远大于 std）。HateMM：AP 与 within 确认，ROC 3-seed 均值 .816 只领先 MACIL-SD .009，低于要求的 .019，未确认。按规则 9，方法保留，已用 2 轮修改（修订 2、3），还剩 1 轮。

### 6.12 修订 4 HateClipSeg seed 234（20 trials，uoa-lab3，`hateclipseg/seed234/`；消融 `ablations/hateclipseg/seed234/`）
- best trial 3：test AP .684 / ROC .656 / within .539（prior_scale 6.73，scaffold，λ_snico .25，选中 epoch 1，即训练一个 epoch、SniCo 尚未开启）；validation 会选同一 trial。20 个 trial AP .640–.684、ROC .629–.665、within .524–.555。仍全部高于门（.562/.528/.524）与 VERA，但低于修订 3（.701/.674/.579）：K4 粗窗口把 within 拉低 .04。
- 消融（trial 3 超参数）：no_k4（= 修订 3 结构）.680/.641/.568；no_snico .684/.656/.539（与 full 相同：选中 epoch 1 在 SniCo 预热之前，本 trial 的 SniCo 没有参与）；input_only .628/.618/.523；no_scaffold = no_scaffold_no_snico .606/.589/.535（同样 epoch ≤ 2）。
- 读数：HateClipSeg 不需要 K4；K30 已够细，K4 只加噪声。修订 4 在 HateClipSeg 上是否保留由两语料统一原则（规则 13）和 HateMM 确认结果决定。

### 6.13 修订 4 HateMM seed 234 搜索（20 trials，uoa-lab1，`hatemm/seed234/study_summary.json`）
- best trial 19：test AP .667 / ROC .839 / within .647（prior_scale 1.09，verdict，λ_snico .15，ρ .73，m 16，选中 epoch 2）；validation 会选 trial 11（.633 / .840 / .648）。三门全过：AP 比 MACIL-SD 高 .094，ROC 高 .032（≥ 基线 std .019），within 高于下限 .632。
- 修订 3 同 seed 最优是 .646 / .830 / .642（6.8）；修订 4 AP +.021、ROC +.009。20 个 trial 里 11 个 within 通过；within 通过的 trial 全部 prior_scale ≤ 3.1 且大多选 verdict 列（不含位置通道），与 6.8 的读数一致。
- 位置剖面去除（6.3 方法）：trial 19 去除公共位置剖面后 within 仍 .647（去除前 .647；仅位置剖面 .417），即这一 trial 的 within 不来自位置先验。修订 3 trial 14 同分析是 .642 → .612。

### 6.14 修订 4 HateMM 消融（trial 19 超参数，seed 234，uoa-lab1，`ablations/hatemm/seed234/<name>/metrics.json`）

| 设定 | AP | ROC | within | 选中 epoch |
|---|---|---|---|---|
| full（trial 19） | .667 | .839 | .647 | 2 |
| no_k4（= 修订 3 结构） | .617 | .808 | .638 | 2 |
| no_snico | .606 | .826 | .640 | 17 |
| input_only（K30+K4 只拼输入，有 SniCo） | .655 | .834 | .643 | 2 |
| no_scaffold = no_scaffold_no_snico | .562 | .783 | .621 | 2 |
| 门 / MACIL-SD 3-seed | .573 | .807 | .632 / .595 | |

读数：
- K4 第二粒度在 HateMM 上有效：同一训练轨迹（都选 epoch 2）AP +.050、ROC +.031。与 HateClipSeg（6.12，K4 使 within −.04）相反。
- **SniCo 在这一 trial 的选中模型里没有参与**：`snico_warmup_epochs` = 2,epoch 1–2 的 λ_snico = 0,full 与 no_snico 的 epoch 1–2 训练记录逐位相同（`summary.json` 的 `history`）。full 选 epoch 2 是因为 SniCo 开启后 val 准则下降（epoch 3 .768,epoch 17 .751);no_snico 里 val 在 epoch 17 升到 .804,但 test 反而 .606。所以 full 的 .667 = 一个训练 2 个 epoch、无 SniCo 的模型。6.12 HateClipSeg 修订 4 trial 3 选 epoch 1，同样在 SniCo 开启前。修订 3 的最优 trial 选 epoch 6（HateMM）/ 10（HateClipSeg），SniCo 有参与。这条要进规则 14 清单的组件贡献项：修订 4 seed 234 的最优 checkpoint 在两个语料上都不含 SniCo 贡献。
- 逻辑先验对比只拼输入：+.012 AP / +.005 ROC（input_only .655/.834）。K30+K4 拼输入比修订 3 的 K30 拼输入（6.10 的 .579/.792，trial 14 超参数，不是同一组超参数）高，倾向于修订 1 失败的主因是 K30 单粒度信息太粗，但两组超参数不同，不能定论。

### 6.15 修订 4 HateClipSeg 三 seed 确认（各 seed 20 trials 各取最优，uoa-lab3，`hateclipseg/seed<seed>/study_summary.json`）

| seed | best trial | AP | ROC | within | 选中 epoch | validation 选 trial 的 test |
|---|---|---|---|---|---|---|
| 234 | 3 | .684 | .656 | .539 | 1 | 同一 trial |
| 2025 | 6 | .700 | .678 | .556 | 14 | trial 2：.684 / .665 / .548 |
| 3407 | 15 | .683 | .676 | .552 | 7 | trial 4：.678 / .651 / .535 |
| 均值 ± std | | .689 ± .010 | .670 ± .012 | .549 ± .008 | | |

门：AP .562（Fed-WSVAD，std .036）、ROC .528（DSANet，std .023）、within ≥ .524。AP 余量 .127 ≥ .036，ROC 余量 .142 ≥ .023，within 全部 seed 通过。修订 4 在 HateClipSeg 上确认通过。与修订 3（6.11：.694 / .665 / .575）相比 AP −.005、ROC +.005、within −.026，差异在 std 内，只有 within 明显下降。seed 2025 / 3407 的最优 trial 选中 epoch 14 / 7，SniCo 有参与；seed 234 没有（6.14）。

### 6.16 修订 4 HateMM 三 seed 确认（各 seed 20 trials 各取最优，uoa-lab1，`hatemm/seed<seed>/study_summary.json`）

| seed | best trial | AP | ROC | within | 选中 epoch | within 过下限的 trial 数 | validation 选 trial 的 test |
|---|---|---|---|---|---|---|---|
| 234 | 19 | .667 | .839 | .647 | 2 | 10/20 | trial 11：.633 / .840 / .648 |
| 2025 | 19 | .663 | .834 | .638 | 2 | 8/20 | trial 3：.620 / .819 / .620（within 低于下限） |
| 3407 | 0 | .637 | .835 | .634 | 2 | 4/20 | 同一 trial |
| 均值 ± std | | .656 ± .016 | .836 ± .003 | .640 ± .007 | | | |

门：AP .573（MACIL-SD，std .033）、ROC .807（std .019）、within ≥ .632。AP 余量 .083 ≥ .033，ROC 余量 .029 ≥ .019，within 三 seed 都过下限。**修订 4 在 HateMM 上确认通过**（修订 3 的 ROC 余量 .009 未过，见 6.11）。三个最优 trial 都选 epoch 2、prior_dims 都是 verdict、prior_scale 1.1–3.2。位置剖面去除（6.3 方法）后 within：seed 234 .647、seed 2025 .650、seed 3407 .623，三 seed 均值 .640 与去除前相同，within 不来自位置先验（MultiHateLoc 去除后 .52–.54）。

## 8. 修订 4 两语料确认汇总与规则 14 清单（2026-09-02）

| 语料 | AP | ROC | within | 门（最强训练 baseline） |
|---|---|---|---|---|
| HateMM（3 seed） | .656 ± .016 | .836 ± .003 | .640 ± .007 | .573 / .807 / ≥ .632 |
| HateClipSeg（3 seed） | .689 ± .010 | .670 ± .012 | .549 ± .008 | .562 / .528 / ≥ .524 |

- (a) 两语料三 seed 确认全过，每个数字来自该 seed 完整 20-trial 搜索的最优 trial（6.15、6.16）。
- (b) std 已报；两语料 pooled 余量都大于 std。
- (c) 规则 13：两语料同一架构、同一损失、同一训练与推理流程；不同的只有搜索选出的超参数。其中 `prior_dims`（先验读 10 列裁定或 12 列裁定+位置）是搜索空间里的类别项，HateMM 三 seed 选 verdict、HateClipSeg 三 seed 选 scaffold。它由同一份声明的搜索空间自动选出，但不是标量，是否算"按语料换结构"要用户裁定；若按最严格读法，可把 HateClipSeg 限定为 verdict 重跑（6.7 修订 3 的 HateClipSeg 最优 trial 也选 scaffold）。
- (d) checkpoint 由 validation (AP+ROC)/2 选；搜索空间、trial 数、目标在 3.2/3.3 先于搜索写明；目标为 test，validation 选 trial 的 test 数字同表给出。HateMM validation 选 trial 的 test 均值 .630 / .831 / .634，HateClipSeg .682 / .657 / .541，也都过 pooled 门。
- (e) 无推理后处理、无 ensemble、无按语料分支。冻结 Qwen2.5-VL-7B 分段裁定（K30 与 K4）在训练与推理都作为输入，不是 train-only teacher；去掉它的数字 = no_scaffold_no_snico（HateMM .562/.783/.621；HateClipSeg .606/.589/.535）。
- (f) 最强 baseline + 同样输入：MACIL-SD + BERT 文本 + 裁定拼输入 = input_only（HateMM .655/.834/.643；HateClipSeg .628/.618/.523，seed 234）。HateMM 上 logit 先验只比拼输入高 .012 AP / .005 ROC；HateClipSeg 高 .056 / .038。
- (g) 核心机制去除：裁定先验去除后 pooled AP 在 HateMM −.105、HateClipSeg −.078（seed 234）。**SniCo 边界对比不满足 (g)**：修订 4 seed 234 两语料最优 checkpoint 都在 SniCo 开启前选出（6.12、6.14），HateMM 三 seed 都是如此；no_snico 消融在 HateClipSeg 与 full 相同，在 HateMM 更低只是因为 validation 换选了更晚的 epoch。SniCo 不能作为 novelty 主张；方法主张应写为"两粒度冻结 VLM 裁定的可学习 logit 先验"。SniCo 留在训练里的作用是让 validation 选早期 checkpoint，这是选择副作用，不是机制。
- (h) 评测器、split、GT、1 fps 协议未改动。
- (i) 两语料三项指标全报。

**结论**：规则 8 确认级两语料全过，规则 14 除 (c) 的 `prior_dims` 读法与 (g) 的 SniCo 主张外全部满足。SOTA 可以汇报；novelty 主张只剩裁定 logit 先验一项（第 7 节的 novelty 复核是按"先验 + 边界对比"过的，先验单独是否够，需要用户或重新复核裁定）。规则 9 的 3 轮修改（修订 2、3、4）已用完，本候选不再修改。

## 9. 收窄主张（只剩裁定 logit 先验）的 novelty 复核（2026-09-02，fable agent 文献检索）

主张：MACIL-SD 骨干上，冻结 Qwen2.5-VL-7B 两粒度（K30、K4）分段裁定作为逐帧 logit 的可学习线性先验，进入 top-k bag 选择、损失与推理；裁定同时拼入 a 流输入。规则 4 四项：
1. hateful video 文献无先例（PASS）。检索：MultiHateLoc（无 LLM/VLM）、LELA（training-free 逐帧打分，无训练定位器）、TANDEM（SFT+GRPO 微调 MLLM 直接输出时间戳，全监督）、SafeLens、HateClipSeg（ActionFormer 全监督）、HVGuard（MLLM 输出作视频级特征）、RAMF、CMHKF、ImpliHateVid、MM-HSD、LEAF、MARS 等均为视频级或无 MIL。
2. 非 ensemble（PASS）：单模型，先验参数与骨干在同一损失下联合更新，top-k 选择用合成 logit；对照 SlowFastVAD 的事后融合。
3. 非后处理（PASS）：先验在训练内部，推理无额外操作。
4. 非纯工程技巧（PASS，但是边缘）：公式本身是 Tip-Adapter / CLIP-Adapter / AMU-Tuning 的 logit-bias 范式迁移到 MIL 时间定位；新在先验来源（MLLM 离散裁定）、进入 MIL 选择规则（改变哪些实例接收正梯度）、两粒度联合。WSVAD/WTAL 逐一核对无同类：MLLM4WTAL（文本嵌入注意力先验，推理不用 MLLM）、Ju et al. CVPR 2023 / TFPLG / TPWNG / CPL-VAD（伪标签路线）、VadCLIP、DSANet、TbVAD / TEVAD / π-VAD（文本特征拼接 = 本方法 input_only）、Holmes-VAU / ECVT（多粒度是标注或描述层级）、GlanceVAD（人工 glance 先验进 MIL，思路最近但先验来源不同）、SteerVAD、LAVAD / VERA / AnomalyRuler（training-free）。多粒度 MLLM 裁定联合学习未找到先例。

必须如实写：HateMM 上 logit 先验相对拼输入（input_only）的增量 .012 AP / .005 ROC 在 seed std 内，只有 HateClipSeg（.056 / .038）显著；主张应定为"裁定条件化的 MIL 定位"而不是"新的融合公式"，两语料 input_only 对比都要报。论文对照：MultiHateLoc、LELA、TANDEM、MLLM4WTAL、TFPLG / Ju et al.、Tip-Adapter / AMU-Tuning、SlowFastVAD、TEVAD / HVGuard。

不改模型即可加强主张的分析（待做）：(1) top-k 选中集合在骨干 logit 与合成 logit 两种排序下的重叠率，以及被先验换进/换出的秒的 GT 阳性率；(2) K30 单独、K4 单独、联合的 test 三指标与按视频长度分桶；(3) 训练后先验权重相对初始化的偏离与 prior_scale–within 关系。

### 9.1 第 9 节三项分析的结果（2026-09-02，本机 CPU；脚本 `analysis_topk_prior.py`、`verdict_only_eval.py --k 30 4`）

**分析 2：裁定本身按粒度（不训练，test，`verdict_only_gran/<corpus>/test/metrics.json`）**

| 语料 | K30 | K4 | K30+K4 均值 |
|---|---|---|---|
| HateMM | .397 / .683 / .540 | .457 / .782 / .549 | .500 / .801 / .572 |
| HateClipSeg | .610 / .616 / .559 | .576 / .585 / .474 | .630 / .633 / .528 |

HateMM 上 K4 单独就比 K30 强（ROC .782 对 .683），两者平均再到 .801；HateClipSeg 上 K4 的 within .474 低于 .5，平均后 within 从 .559 降到 .528。这与训练后的消融同向（6.12、6.14）：K4 的价值是数据集性质，HateMM 视频短、跨视频差异大，粗窗口把视频级判断带进逐帧分；HateClipSeg 视频内需要细粒度。

**分析 1：先验是否改变 MIL 的 top-k 选择（train 正例视频，crop 0，k = ⌈T/16⌉，`analysis/topk_prior_<corpus>_<seed>.txt`）**

| 语料 / seed | Jaccard 均值 | 中位数 |
|---|---|---|
| HateMM 234 / 2025 / 3407 | .808 / .614 / .784 | 1.00 / .60 / 1.00 |
| HateClipSeg 234 / 2025 / 3407 | .222 / .400 / .527 | .05 / .31 / .48 |

HateClipSeg 上先验大幅改变哪些实例进 bag（seed 234 中位数只有 .05 重叠），HateMM 上只在少数视频改变。与 (f) 的观察一致：HateClipSeg 上 logit 先验相对拼输入 +.056 AP，HateMM 上只 +.012。被先验换进/换出的秒的 GT 阳性率没算：train 集没有评测器用的 GT 数组。

**分析 3：训练后先验权重相对初始化**

六个最优 trial 的 `prior.weight`：两个 level 列各 ≈ prior_scale/2（与初始化差 < .03），四个 one-hot 列和位置列 |w| ≤ .04，bias 与初始化差 < .04。也就是说训练几乎没有改动先验，先验实际上等于固定的 `prior_scale · (K30 级别 + K4 级别)/6 − prior_scale/2`，"可学习"三个字在实验里没有内容；起作用的是搜索出的 `prior_scale` 与两粒度。论文写法要如实：先验是尺度由 validation/搜索决定的固定线性先验，进入 bag 选择与损失。
