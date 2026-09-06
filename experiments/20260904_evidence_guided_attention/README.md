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
| `scalar_bias` | β_h(e_j) 换成单个可学习标量 γ·ℓ_j/L，各头共用（规则 4 复核要求） | B 的增益是否来自"按证据编码学出的逐头偏置"，还是任何按后验的重加权都行；B 只在 full 两语料三 seed pooled 都高于本臂时才作主张 |
| `no_context` | 无 C | 显式视频级密度项是否有贡献 |
| `no_block` / `no_prior` / `no_cmal` / `mean_prior` / `no_verdict` | 同候选 1 定义 | 训练项与模块 3 的贡献（论文全表） |

## 4. 预注册（搜索前写定，2026-09-04）

搜索空间（两语料共用）：lr log[1e-4, 1e-3]；max_seqlen {150, 200, 300}；λ_cma [0.5, 2]；prior_scale log[0.5, 8]；w_fine [0, 1]；λ_block log[0.05, 2]。每 (语料, seed) 20 trial（单 trial < 1 小时），目标 test (AP+ROC)/2，within 破下限剪枝，validation (AP+ROC)/2 选 checkpoint。同时记录 validation 选 trial 的 test 数字。

可证伪预期：
1. 规则 8 筛选（seed 234）与确认（三 seed）两语料全过（门 HateMM .573/.807/≥.632，HateClipSeg .562/.528/≥.524）。
2. **相对候选 1 修订 1 三 seed**（HateMM .657 ± .013 / .842 ± .005 / .646；HateClipSeg .699 ± .006 / .681 ± .016 / .553）：HateMM pooled AP 或 ROC 至少一项高 ≥ .005 且另一项不低于候选 1 减一个 std；HateClipSeg 两项都不低于候选 1 减一个 std；within 两语料不低于下限。
3. **相对 avce 臂三 seed**（同训练同超参）：HateMM pooled AP 或 ROC 至少一项高 ≥ .005 且另一项不低；HateClipSeg 不低。这一条不成立则"提升不是设计导致"，本候选按规则 9 修改或归档。
4. 主张只落在三 seed 两语料 pooled 均值都下降的部件上；一个部件都不成立则本骨干不能作 novelty 主张。B 额外要求 full 高于 `scalar_bias` 臂（规则 4 复核）。C 在论文里写作"视频级证据偏移"（头是线性的，C 等价于每视频一个由 mean e_t 线性映射的 logit 偏移），对照 GIG-VAD、PEL4VAD 的视频级上下文。
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
- 2026-09-04 10:45：规则 6 code review PASS 无 BLOCKER（`REVIEW_RULE6.md`：avce 臂与候选 1 前向数值一致 ≤ 6e-8，无泄漏，共享评测器）。规则 4 复核 GO 6/10（`REVIEW_RULE4.md`）：四类都不触发；最近先例 A+B 合起来是 Graphormer（离散属性嵌入 + 逐头加性注意力偏置，这里把图结构换成 VLM 裁定），B 的谱系 ALiBi / T5 偏置 / Yang 2018 / MLLM4WTAL，C 最近 GIG-VAD / PEL4VAD；无先例用另一个模型的逐秒裁定作跨模态 key 偏置。必须项：加 `scalar_bias` 臂（已加）；novelty 表述为"裁定来源 + 2×2 粒度一致格子 + 一份共享证据编码经三个入口进弱监督跨模态 MIL 骨干"，不是"注意力偏置"本身；引用并对照 Graphormer、ALiBi、T5、Yang 2018、Lin 综述、MLLM4WTAL、GIG-VAD、PEL4VAD；相关工作对照 VLM 分段分数进 WSVAD/WTAL 的三种方式（伪标签损失 TPWNG/TFPLG/LAVAD、注意力调制 MLLM4WTAL、输入编码=本文）；主表保留 avce 臂、HMM 后验单独行、MultiHateLoc。搜索启动。

### 6.1 修订 1 结果（2026-09-04；`runs/20260904_evidence_guided_attention/`）

**HateClipSeg**（uoa-lab3，三 seed 各 20 trial，5/6/7 个被 within 下限剪掉；best trial 都在第 1 个 epoch 被选出）：

| | AP / ROC / within | 对候选 1（.699 ± .006 / .681 ± .016 / .553） |
|---|---|---|
| seed 234 best（trial 19） | .693 / .659 / .549 | |
| seed 2025 best（trial 12） | .689 / .660 / .555 | |
| seed 3407 best（trial 12） | .700 / .660 / .554 | |
| 三 seed 均值 | .694 ± .006 / .660 ± .000 / .553 | AP −.005、**ROC −.021（超过一个 std）** |
| validation 选 trial 的 test | .681/.653/.538、.671/.630/.447、.692/.656/.536 | |

规则 8 门过（.562/.528/.524），预注册第 2 条（不低于候选 1 减一个 std：ROC ≥ .665）**不成立**。

三 seed 消融（每 seed 用该 seed best trial 超参；`ablations/hateclipseg/seed<seed>/<arm>/`）：

| 臂 | 三 seed 均值 | 对 full（.694/.660/.553） | 选中 epoch |
|---|---|---|---|
| avce（候选 1 骨干，同训练） | .696 / .667 / .555 | +.002 / +.007 / +.003 | 3, 4, 1 |
| no_enc（e_t 不进残差流） | .694 / .666 / .548 | .000 / +.006 / −.005 | 4, 1, 1 |
| evid_audio_only | .692 / .665 / .550 | −.002 / +.005 / −.002 | 1, 7, 1 |
| no_cell | .690 / .656 / .554 | −.004 / −.004 / +.001 | 1, 1, 1 |
| no_bias | .696 / .661 / .556 | +.003 / +.001 / +.004 | 1, 1, 1 |
| scalar_bias | .696 / .661 / .556 | +.003 / +.001 / +.004 | 1, 1, 1 |
| no_context | .695 / .660 / .557 | +.002 / .000 / +.004 | 1, 1, 1 |
| no_block | .674 / .652 / .559 | −.020 / −.007 / +.006 | |
| no_prior | .673 / .654 / .540 | −.020 / −.006 / −.013 | |
| no_cmal | .694 / .660 / .553 | .000 / .000 / .000（第 1 个 epoch CMAL 权重为 0，与 full 同一模型） | 1, 1, 1 |
| mean_prior | .653 / .624 / .530 | −.040 / −.035 / −.023 | |
| no_verdict | .584 / .556 / .529 | −.110 / −.104 / −.024 | |

**HateMM**（uoa-lab1，三 seed 各 20 trial）：56/60 个 trial within 低于 .632 下限（seed 234、3407 全部 20 个被剪），没有有效 best；不看约束的最好 seed 234 .656/.840/.595（trial 8）、2025 .685/.862/.608（trial 5）、3407 .660/.843/.595（trial 17），within 最高 .629/.642/.612（候选 1 .646）。消融链因 best 为空未运行。

**读法（修订 1 淘汰）**：修订 1 的骨干比候选 1 骨干（avce 臂）没有任何增益，所有结构臂差异在 ±.005 内；HateMM 上 pooled 持平但 within 掉 .05，破下限。原因看 validation 轨迹（三 seed 一致）：凡是 e_t 加进残差流的臂（full、no_bias、no_context、evid_audio_only、no_cell、scalar_bias）val ROC 从第 1 个 epoch 的 .74–.76 三个 epoch 内掉到 .66–.68，视频级 bag 损失同时下降一倍快（.52→.27 对 avce 的 .52→.43）；不进残差流的 avce、no_enc 保持 .75–.77（no_enc 3407 例外）。即：证据编码直接进内容表示后，网络一两个 epoch 内就靠证据把视频级 bag 拟合掉，跨视频的帧分数排序（pooled ROC）和视频内排序（HateMM within）随后退化；候选 1 之所以没这个问题，是裁定四列只占 fc_a 输入 902 维中的 4 维，初始化下贡献很小，相当于隐式的弱化。

## 7. 修订 2（规则 9 第 1/3 次修改，2026-09-04 18:40）：证据只进注意力的 query/key，不进内容表示

**改动（一处）**：`model.py` `EvidenceGuidedCMA.one`：q_in = LayerNorm(x) + e，k_in = y + e，**value 仍是 y**，残差流不加 e。证据编码 e_t 决定"内容从哪些秒聚合"（q/k 相似度里多出 content·e、e·content、e·e 三项），加上 B 的 key 偏置和 C 的视频级上下文；内容表示（CMAL 对比的对象、头的输入）保持纯内容。B、C、D、头、损失、搜索空间不变。

**臂**：`full`（q/k 编码 + B + C）、`avce`（候选 1 骨干）、`stream_enc`（= 修订 1 full，只作记录）、`no_qk_enc`（= 修订 1 no_enc：证据只经 B、C）、`no_cell`、`no_bias`、`scalar_bias`、`no_context`、训练臂同前。`evid_audio_only` 删除（残差流不再加 e）。

**预注册（可证伪）**：
1. 规则 8 两语料 seed 234 过门；HateMM within 恢复到候选 1 水平（≥ .632，多数 trial 不被剪）。
2. 三 seed：HateMM pooled AP 或 ROC 相对 avce 臂高 ≥ .005 且另一项不低；HateClipSeg 不低于 avce 臂；full 不低于候选 1 减一个 std（HateMM AP ≥ .644、ROC ≥ .837；HateClipSeg AP ≥ .693、ROC ≥ .665）。
3. 若 (2) 在两语料都不成立：修订 2 淘汰，剩两次修改机会；若结构臂仍全部在 ±.005 内，本方向（把证据显式接进 AVCE 注意力）归档，换候选。

输出 `runs/20260904_evidence_guided_attention_rev2/`；HateClipSeg 在 uoa-lab3，HateMM 在本机（GPU 空闲）。

### 7.1 修订 2 结果（2026-09-04 19:50；`runs/20260904_evidence_guided_attention_rev2/`）

**HateClipSeg seed 234**（uoa-lab3，20 trial，5 个被剪）：best trial 13（epoch 7）**.691 / .676 / .552**；validation 选 trial 15 .690/.674/.545。对候选 1 seed 234（.695/.679/.546）差 .004/.003，在噪声内；规则 8 门过。训练塌陷已修（val ROC 不再在第 1 个 epoch 后掉）。

| 臂（trial 13 超参） | AP / ROC / within | 对 full | 选中 epoch |
|---|---|---|---|
| full | .691 / .676 / .552 | | 7 |
| avce（候选 1 骨干） | .694 / .665 / .549 | +.003 / **−.011** / −.003 | 3 |
| stream_enc（修订 1） | .688 / .658 / .543 | −.003 / −.017 / −.009 | 1 |
| no_qk_enc | .691 / .674 / .552 | .000 / −.001 / .000 | 7 |
| no_bias | .694 / .675 / .555 | +.003 / −.001 / +.003 | 6 |
| scalar_bias | .693 / .669 / .557 | +.002 / −.007 / +.005 | 12 |
| no_cell | .694 / .675 / .555 | +.004 / −.001 / +.003 | 2 |
| no_context | .692 / .672 / .555 | +.001 / −.003 / +.003 | 8 |
| no_block | .692 / .663 / .561 | +.001 / −.013 / +.009 | 1 |

读法：整个模块相对候选 1 骨干 ROC +.011（单 seed），但 q/k 编码、偏置、上下文、格子嵌入四个部件**单独去掉都不掉**（≤ .003），三条证据入口互相冗余；规则 14(g) 下没有一个部件能作主张。三 seed 与 HateMM 消融链继续跑只作记录。

**HateMM seed 234**（本机，搜索仍在跑）：前 4 个 trial within .610/.626/.625/.582，全部低于 .632 下限，与修订 1 相同。诊断（`diag/hatemm/seed234/`，都用候选 1 seed 234 best trial 3 的超参；`key_mask` 为本次加的诊断开关）：

| 对照 | AP / ROC / within |
|---|---|
| 候选 1 no_ema 臂（同超参，无 key 屏蔽，README 9.6） | .661 / .841 / .650 |
| avce，屏蔽 padding key（修订 1/2 的设置） | .625 / .825 / .643 |
| avce，不屏蔽（MACIL-SD 原样） | .636 / .842 / .654 |
| full（修订 2），屏蔽 | .616 / .825 / .616 |
| full（修订 2），不屏蔽 | .644 / .836 / .622 |
| avce / no_context / no_bias，trial 0 超参，屏蔽 | .609/.832/.628、.607/.821/.625、.619/.835/.618 |

两个结论：
1. **屏蔽 padding key 在 HateMM 上有害**（avce：ROC −.017、within −.011；full：AP −.028）。MACIL-SD 训练时 padding 行（零向量经 fc 后是同一个偏置向量）作为 key 参与注意力，实测训练好的模型把平均 .254 的注意力放在 padding key 上（744 个训练视频，中位 162 行、max_seqlen 200，padding 占 .324）；测试时序列不截断、没有 padding，这部分注意力落回真实秒。即候选 1 的跨模态注意力依赖一个"训练时有、测试时没有"的偶然空 token；去掉它反而更差，说明"允许一秒不看另一模态"是有用的，但现在的实现形式是训练/测试不一致的。
2. **证据引导注意力在 HateMM 上损害视频内排序**：同样不屏蔽，full within .622 对 avce .654（−.032），ROC −.006，AP +.008。机制：偏置与 q/k 证据编码让视频内所有秒都去看同一批"证据所在的秒"，每行加进的跨模态上下文趋同，行与行的内容差异被抹平；pooled 靠视频级校准补回，within 补不回。这是修订 1、2 HateMM 全部 trial within 破下限的原因。

**判定（预注册第 3 条）**：修订 2 在 HateClipSeg 结构臂全部在 ±.005 内、HateMM 相对 avce 更差，**候选 3 方向（证据显式接进跨模态注意力）归档**，不用剩余修改。保留的发现：(i) 证据进内容表示会让训练塌陷，(ii) 证据引导注意力抹平视频内排序，(iii) padding key 是候选 1 注意力里起作用的、训练测试不一致的空 token。(iii) 直接引出下一个候选（`experiments/20260904_null_token_cma/`）。

### 7.2 修订 2 三 seed 记录（2026-09-04 22:10；HateClipSeg 完整，HateMM 只有 seed 234 的 9 个 trial）——**修正 7.1 的 HateClipSeg 读法**

**HateClipSeg 三 seed**（uoa-lab3；seed 2025 best trial 12 .706/.694/.558、seed 3407 best trial 16 .696/.683/.537，被剪 5/8 个）：full **.698 ± .008 / .684 ± .009 / .549 ± .011**，对候选 1（.699 ± .006 / .681 ± .016 / .553）持平（ROC +.003）。三 seed 消融（每 seed 用该 seed best trial 超参）：

| 臂 | 三 seed 均值 | 对 full | AP / ROC 下降 seed 数 |
|---|---|---|---|
| avce（候选 1 骨干） | .688 / .659 / .545 | −.009 / **−.025** / −.003 | 2/3、3/3 |
| stream_enc（修订 1） | .669 / .645 / .546 | −.028 / −.040 / −.003 | 3/3、3/3 |
| no_qk_enc | .697 / .683 / .548 | −.001 / −.001 / −.001 | 3/3、3/3（幅度 ≈ 0） |
| no_cell | .683 / .657 / .545 | −.015 / −.027 / −.004 | 2/3、3/3 |
| no_bias | .686 / .666 / .547 | −.011 / −.018 / −.001 | 2/3、3/3 |
| scalar_bias | .686 / .664 / .548 | −.012 / −.020 / −.001 | 2/3、3/3 |
| no_context | .678 / .655 / .541 | −.020 / −.030 / −.008 | 2/3、3/3 |
| no_block | .651 / .627 / .545 | −.047 / −.058 / −.004 | 2/3、3/3 |
| no_prior | .648 / .632 / .527 | −.050 / −.052 / −.022 | 3/3、3/3 |
| no_cmal | .680 / .655 / .537 | −.017 / −.029 / −.012 | 2/3、3/3 |
| mean_prior | .673 / .659 / .528 | −.025 / −.025 / −.021 | 3/3、3/3 |
| no_verdict | .597 / .566 / .531 | −.100 / −.118 / −.018 | 3/3、3/3 |

**修正**：7.1 只看 seed 234 时所有结构臂都在 ±.005 内；seed 2025/3407 上差距大得多，三 seed 均值下 avce（−.025 ROC，3/3）、格子嵌入（−.027）、证据偏置（−.018；换标量偏置 −.020）、视频级上下文（−.030）都成立，只有 q/k 编码 ≈ 0。**在 HateClipSeg 上，证据引导注意力是可确认的机制**：视频级证据上下文 + 按证据编码的逐头 key 偏置 + 一致格子嵌入，合起来对候选 1 骨干 +.025 ROC，把候选 1 的骨干在这个语料上从"投影 + 头"变成有可确认贡献的部件（候选 1 结构消融里注意力本身只值 ≤ .012 ROC）。seed 234 单独不足以判定结构臂，以后结构臂判定一律等三 seed。

**HateMM**（本机记录搜索，9/20 trial 后停止让出 GPU）：9 个 trial within .582–.627 全部低于 .632 下限；不看约束最好 .646/.846/.626（trial 7）。与 7.1 诊断一致：同一机制在 HateMM 抹平视频内排序。

**判定不变但理由修正**：候选 3 修订 2 在 HateClipSeg 通过预注册第 2、3 条（相对 avce +.009/+.025，四个部件中三个三 seed 都降），在 HateMM 不过规则 8 的 within 下限（机制本身损害视频内排序，7.1 诊断）。规则 13 要求一法两语料，所以本候选不能作方法；归档。可复用的结论：视频级证据上下文与证据偏置在注意力本身贡献小的语料（HateClipSeg）有 +.02–.03 ROC 的增益，在注意力贡献大、within 靠内容的语料（HateMM）损害 within；候选 4 的空 token 是把视频级证据上下文改成"每行按自己 query 决定拿多少"的形式，正是为了保留前者、避免后者。

## 8. 现行规则下重跑 HateMM（2026-09-06 晚，用户裁定）

背景：修订 2 在 HateMM 只跑了 seed 234 的 9 个 trial，且当时按 within < .632 剪枝；09-06 起 within 不再剪枝、不作门，规则 14(g) 只看三 seed 均值。用户裁定：按现行流程重跑 HateMM 三 seed 并补消融。

- **模型不变**：修订 2（证据只进 q/k + key 偏置 + 视频级上下文，第 7 节）。
- **搜索空间不变**：第 4 节的 6 个标量（lr、max_seqlen、λ_cma、α、w_fine、λ_block），与 HCS 已有的三 seed 研究保持同一空间（规则 13）。w_fine 暂保留，是否删除等 HateMM 结果出来再定，删则两语料一起重跑。
- **只改剪枝**：`search.py --no-within-prune`，within 照常记录。输出 `runs/20260904_evidence_guided_attention_rev2_noprune/hatemm/seed<seed>/`。
- **HCS 沿用修订 2 已有结果**：按新规则（不剪枝、best = 全部 trial 中目标值最高者）从已有 study 重算，三个 seed 的 best trial 与旧规则相同（seed 234 trial 13、2025 trial 12、3407 trial 16，`runs/20260904_evidence_guided_attention_rev2/hateclipseg/seed<seed>/study_summary.json`，被剪 trial 的 `objective_unconstrained` 均低于 best），所以 HCS 三 seed 数字与消融不需要重跑。
- 流程：HateMM seed 234/2025/3407 各 20 trial → 每 seed 用 best trial 超参跑 12 个消融臂（avce、stream_enc、no_qk_enc、no_cell、no_bias、scalar_bias、no_context、mean_prior、no_block、no_prior、no_cmal、no_verdict，`scripts/run_locked_ablations.sh 20260904_evidence_guided_attention_rev2_noprune hatemm <seed> ...`）→ 两语料合并按 14(g) 判定 → 外部审稿。
