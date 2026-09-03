# 候选 3：证据引导的跨模态注意力骨干（Evidence-Guided Cross-Modal Attention，2026-09-04 提案）

上游：候选 1 `experiments/20260903_hier_evidence_mil/`（修订 1；模块 3 分层证据 HMM + 模块 2 裁定块级 MIL，两语料规则 8 确认通过，用户裁定"暂不作论文方法：超参太多、骨干架构没改"）。本目录只改**模块 2 的骨干结构**，模块 3（HMM 先验）、块级 MIL、CMAL 对比损失、搜索协议全部沿用候选 1。候选 2（证据链网络）已淘汰（`archive/experiments/20260904_evidence_chain_net/`）。

## 0. 出发点：候选 1 的骨干靠什么 work，哪里做得不好（全部来自候选 1 README 第 9 节，test 作 developmental evidence）

候选 1 骨干 = MACIL-SD AVCE：`fc_v`、`fc_a` 两个线性投影 → **一层共享的跨模态注意力**（同一层权重，视频查音频、音频查视频）→ 共享 `fc` 头，逐行 logit = fc(a_out) + fc(v_out)。VLM 裁定的四列（ℓ_t、P(s_t)、b_fine、b_coarse）**拼在音频+文本的 896 维向量后面一起过 `fc_a`**，是骨干里裁定唯一的入口。

已确认的事实：
1. 骨干靠三件事 work（9.5）：视频级仇恨密度估计（z 方差 92%/79% 在视频之间，最强预测变量是"视频内触发块比例"）；两粒度可靠性校正（网络学到 K30 单独触发在 HateMM 不可信，HMM 固定噪声率学不到，9.2）；视频内排序主要来自 HMM 后验。
2. AVCE 结构里唯一在两语料都确认的部件是**共享的跨模态注意力**（9.7：去掉注意力 HateMM −.054 AP / −.037 ROC 三 seed 全降，HateClipSeg −.004/−.012；换自注意力比不要注意力还差；不共享 −.038 AP）。注意力的贡献全在 pooled，within 不动 → 它做的是跨视频的分数尺度校准（视频级密度），不是视频内排序。
3. 没用的部件：EMA 自蒸馏（no_ema ≈ 0，两语料）、链后验蒸馏（两臂 HateClipSeg 都不升）。
4. 做得不好的地方：(a) 四列裁定混进 896 维内容向量过一个线性层，密度估计、可靠性校正都要靠注意力隐式学；(b) 裁定只进音频流，视频流的 query 看不到自己所在秒的裁定；(c) 注意力对哪些秒加权完全由内容相似度决定，与证据无关；(d) 训练里带着无效的 EMA 和视觉伙伴网络，多出 m、ema_epochs、single_lr_scale 三个超参；(e) padding 行没有在注意力里屏蔽（MACIL-SD 原样）。

## 1. 方法：把三个机制做成骨干的显式部件

记 hid = 128，投影 h_v = fc_v(I3D)、h_a = fc_a(VGGish ⊕ BERT)（内容，不含裁定）。裁定四列 x_t = [ℓ_t/L, P(s_t), b_fine_t, b_coarse_t]，L ≈ 13.8。

- **A. 模态共享的证据编码**：e_t = Emb[cell_t] + W·[ℓ_t/L, P(s_t)]，cell_t = 2·b_fine_t + b_coarse_t ∈ {0,1,2,3}，即两粒度裁定的一致格子（9.2 的 2×2 表）各一个可学习向量（零初始化），加上后验两列的线性映射。e_t **同时加到两个模态流**：h_v ← h_v + e_t，h_a ← h_a + e_t（同位置编码的用法：证据是这一秒的属性，不属于哪个模态）。格子嵌入 = 可靠性校正的显式形式。
- **B. 证据偏置的共享跨模态注意力**：仍是一层共享的 pre-norm transformer 层，两个方向共用；注意力分数加一项按 key 秒的证据算出的逐头偏置：score(i, j) = q_i·k_j/√d + β_h(e_j)，β_h = Linear(hid → nhead)（零初始化，起点等于候选 1 的共享 CMA）。两个模态都被引导去看证据所在的秒。padding 的 key 屏蔽。
- **C. 视频级证据上下文**：c = Linear(mean_{t 有效} e_t)，加到两个流的每一行再进头。这是 9.5 的"视频级密度"项的显式形式（每个视频一个向量，只由裁定分布决定）。
- 头不变：logit_t = fc(a_out_t) + fc(v_out_t)；z̃_t = logit_t + prior_scale·ℓ_t/L；bag = top-⌈T/16⌉ 均值。

训练 = 候选 1 减去 EMA/伙伴网络：L = BCE(bag(z̃), y) + λ_cma·CMAL + λ_block·L_block(z)。CMAL 的两个权重合并为一个 λ_cma（候选 1 搜索里 a2b/a2n 分开，无理由）。

**超参数**：搜索 6 个（lr、max_seqlen、λ_cma、prior_scale、w_fine、λ_block）；固定 dropout .2、lamda_cof .05（MACIL-SD 值）、hid/ffn 128、nhead 4、batch 32、50 epoch、topk_div 16。候选 1 搜 9 个 + 3 个 EMA 固定超参。

**新增参数量**：A 4×128 + 2×128、B 128×4、C 128×128；总 363,653 对 avce 臂 346,241（+5%）。

## 2. 为什么预期有可观察的提升，提升由什么设计导致

- HateMM：候选 1 上注意力层贡献 .054 AP 且全是视频级校准，说明骨干最需要的是"从裁定分布算出视频级密度并校准分数"。A+C 把这条路径做成显式的、参数专用的、两流共享的；B 让内容表示的聚合按证据加权（"裁定说仇恨的秒，内容长什么样"）。预期 pooled 上升，within 不变或微升。
- HateClipSeg：骨干本来接近"投影 + 头"，注意力贡献 ≤ .012 ROC；预期持平到小升（A 的格子嵌入在 HateClipSeg 上 K30 单独触发也可信，网络原本把它压低了一点，9.2）。
- 提升若出现而 avce 臂（同训练、同搜索超参）没有，就是由 A/B/C 设计导致；每个部件各有一臂。

## 3. 臂与主张（规则 14(g)：每条主张对应一个 pooled 下降的消融；三 seed，每 seed 用该 seed best trial 超参）

| 臂 | 改动 | 回答的问题 |
|---|---|---|
| `full` | A + B + C | |
| `avce` | 候选 1 骨干：裁定拼进音频流过 fc_a，无 A/B/C；训练与本候选完全相同 | 整个模块相对候选 1 骨干是否有增益（**主对照**） |
| `no_enc` | e_t 不加进两个流（只经 B 偏置和 C 上下文起作用） | 共享证据编码是否必要 |
| `evid_audio_only` | e_t 只加进音频流 | "模态共享"是否必要（对应候选 1 只进音频流） |
| `no_cell` | 格子嵌入换成四列线性映射 | 显式可靠性格子是否有贡献 |
| `no_bias` | β_h ≡ 0 | 证据偏置注意力是否有贡献 |
| `no_context` | 无 C | 显式视频级密度项是否有贡献 |
| `no_block` / `no_prior` / `no_cmal` / `mean_prior` / `no_verdict` | 同候选 1 定义 | 训练项与模块 3 的贡献（论文全表） |

## 4. 预注册（搜索前写定，2026-09-04）

搜索空间（两语料共用）：lr log[1e-4, 1e-3]；max_seqlen {150, 200, 300}；λ_cma [0.5, 2]；prior_scale log[0.5, 8]；w_fine [0, 1]；λ_block log[0.05, 2]。每 (语料, seed) 20 trial（单 trial < 1 小时），目标 test (AP+ROC)/2，within 破下限剪枝，validation (AP+ROC)/2 选 checkpoint。同时记录 validation 选 trial 的 test 数字。

可证伪预期：
1. 规则 8 筛选（seed 234）与确认（三 seed）两语料全过（门 HateMM .573/.807/≥.632，HateClipSeg .562/.528/≥.524）。
2. **相对候选 1 修订 1 三 seed**（HateMM .657 ± .013 / .842 ± .005 / .646；HateClipSeg .699 ± .006 / .681 ± .016 / .553）：HateMM pooled AP 或 ROC 至少一项高 ≥ .005 且另一项不低于候选 1 减一个 std；HateClipSeg 两项都不低于候选 1 减一个 std；within 两语料不低于下限。
3. **相对 avce 臂三 seed**（同训练同超参）：HateMM pooled AP 或 ROC 至少一项高 ≥ .005 且另一项不低；HateClipSeg 不低。这一条不成立则"提升不是设计导致"，本候选按规则 9 修改或归档。
4. 主张只落在三 seed 两语料 pooled 均值都下降的部件上；一个部件都不成立则本骨干不能作 novelty 主张。
5. 若 (1) 不过：按规则 9，某语料比最强训练 baseline 高 .01 以上且 within 未破可修改（≤ 3 轮），否则归档。

## 5. 运行
```
python experiments/20260904_evidence_guided_attention/search.py --corpus hateclipseg --seed 234 --out-root runs/20260904_evidence_guided_attention
python experiments/20260904_evidence_guided_attention/train.py --corpus hatemm --seed 234 --config <best hparams.json> --ablation avce --out-dir runs/20260904_evidence_guided_attention/ablations/hatemm/seed234/avce
```
本机 GPU 被占；HateClipSeg 在 uoa-lab3，HateMM 在 uoa-lab1（候选 1 链蒸馏臂结束后）。输出 `runs/20260904_evidence_guided_attention/<corpus>/seed<seed>/`、`ablations/<corpus>/seed<seed>/<arm>/`。

共享代码：数据集、scaffold、块级 MIL、评分与评测调用已从候选 1 升入 `src/hier_evidence_common.py`（逐字搬迁，候选 1 的 `dataset.py` 改为再导出）。

## 6. 进度
- 2026-09-04 10:20：提案、`model.py`/`train.py`/`search.py` 写完，七个结构臂前向/反向形状检查通过；等规则 4、规则 6 复核。
