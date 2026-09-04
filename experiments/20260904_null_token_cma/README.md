# 候选 4：空 token 跨模态注意力骨干（Null-Token Cross-Modal Attention，2026-09-04 20:10 提案）

上游：候选 1 `experiments/20260903_hier_evidence_mil/`（修订 1，两语料规则 8 确认；用户裁定"骨干架构没改，不能作论文方法"）。候选 3 `experiments/20260904_evidence_guided_attention/`（证据引导注意力）两次修订后归档，本候选来自它的诊断（其 README 7.1）。模块 3（HMM 先验）、块级 MIL、CMAL、裁定四列拼入音频流的输入路径、搜索协议全部沿用；EMA 与伙伴网络删除（候选 1 no_ema 两语料 ≈ 0）；搜索超参 6 个。

## 0. 出发点：候选 1 的跨模态注意力靠一个训练时有、测试时没有的空 token（test 作 developmental evidence）

1. 候选 1 结构消融（其 README 9.7）：共享跨模态注意力是 AVCE 里唯一两语料都确认的部件（去掉：HateMM −.054 AP / −.037 ROC，HateClipSeg −.004/−.012）；它的贡献全在 pooled，within 不动。
2. MACIL-SD（候选 1 照搬）训练时把 padding 行当 key 参与注意力，不屏蔽。padding 行是零向量，经 fc 后全是同一个偏置向量。实测（候选 3 README 7.1；候选 1 骨干、候选 1 seed 234 最优超参、不屏蔽、训练好的模型，744 个 HateMM 训练视频，中位 162 行、max_seqlen 200，padding 占 key 的 .324）：**有效 query 平均把 .254 的注意力放在 padding key 上**。测试时序列不截断、没有 padding，这部分注意力落回真实秒。
3. 把 padding key 屏蔽掉（候选 3 修订 1/2 的设置）在 HateMM 明显更差（同超参：ROC .842 → .825、within .654 → .643；候选 3 full AP −.028）。即"允许一秒不看另一模态"是有用的：并不是每一秒都有对应的另一模态证据（只有画面的仇恨、只有语音的仇恨），强迫它在真实秒里分配全部注意力反而把无关的跨模态上下文塞进来。
4. 候选 3 的教训：让所有秒都去看"证据所在的秒"会抹平视频内排序（HateMM within −.032）；证据直接进内容表示会让 bag 损失过快塌陷。视频级的证据上下文应当以"每行自己决定要不要拿"的形式进入，而不是加到每一行。

## 1. 方法

记投影 h_v = fc_v(I3D)、h_a = fc_a(VGGish ⊕ BERT ⊕ 裁定四列)（候选 1 输入路径不变）。每个 key 模态 m ∈ {视觉, 音频} 增加一个**空 token**：

n_m = b_m + W_m · c，c = 有效行上裁定四列 [ℓ_t/L, P(s_t), b_fine_t, b_coarse_t] 的均值（视频级证据摘要，4 维）。

共享的跨模态层（MACIL-SD 的 pre-norm transformer 层，两个方向共用权重）里，query 模态 x 看 key 模态 y 时，key/value 序列 = [n_y, y_1..y_T]，padding 行屏蔽。一秒在另一模态里找不到相关内容时可以把注意力放到 n_y 上；n_y 带着这个视频的裁定摘要，所以"什么都不看"等于"拿到视频级证据上下文"。它替代 MACIL-SD 里偶然的 padding 空 token，训练与测试一致；它是每行按自己的 query 决定权重的（不是候选 3 那种加到每一行的常量或偏置）。头、先验、bag、损失不变。

新增参数：2 × 128（b_m）+ 2 × 4 × 128（W_m）= 1,280；总 347,137 对候选 1 骨干 346,241。

## 2. 为什么预期有提升，由什么导致

- HateMM：候选 1 的注意力已经依赖空 token（去掉 −.017 ROC），但测试时没有它，模型在测试时把 ~25% 的注意力重新分配给真实秒，跨模态上下文比训练时多。显式空 token 消除这一不一致；预期 pooled 上升，within 不降（每行自己决定，不抹平）。
- HateClipSeg：注意力本身贡献小（≤ .012 ROC），预期持平；视频级证据摘要经空 token 进入可能给 ROC 小幅增益（候选 3 avce 对 full 的 −.011 ROC 显示视频级证据入口在 HateClipSeg 有用）。
- 由设计导致的判据：full 相对 `no_token_unmasked`（候选 1 骨干，同训练同超参）与 `no_token_masked` 都高。

## 3. 臂与主张（三 seed，每 seed 用该 seed best trial 超参；规则 14(g)）

| 臂 | 改动 | 回答的问题 |
|---|---|---|
| `full` | 证据条件化的空 token，每个 key 模态一个，padding 屏蔽 | |
| `no_token_unmasked` | 候选 1 / MACIL-SD 原样（padding 当偶然空 token，测试时无） | 显式空 token 是否优于偶然空 token（**主对照**） |
| `no_token_masked` | 屏蔽 padding、无 token | 空 token 是否必要（预期两语料都降，HateMM 明显） |
| `const_token` | n_m = b_m（无证据摘要） | 证据摘要经空 token 进入是否有贡献 |
| `shared_token` | 两个 key 模态共用一个 token | 按模态分是否必要 |
| `zero_value_sink` | 可学习空 key，value 固定为 0（纯"分母"空 token，gpt-oss 形式；规则 4 复核要求） | "什么都不看"是否只需要一个吸收注意力的位置；主张链 no_token_masked < zero_value_sink ≤ const_token < full |
| `gated_cma` | 无 token、屏蔽 padding、注意力输出乘以按 query 算的逐行 sigmoid 门（最小形式的 Leaky Gated Cross-Attention，Lee et al. WACV 2022；规则 4 复核要求） | 空 key 形式是否优于"逐行门控"这个同任务先例；full 不低于它才能主张空 key 形式 |
| `no_input` | 裁定四列不拼入输入（c 也置零，先验保留） | 输入路径（候选 1 结论：HateMM 必要） |
| `no_block` / `no_prior` / `no_cmal` / `mean_prior` / `no_verdict` | 同候选 1 | 论文全表 |

## 4. 预注册（搜索前写定，2026-09-04 20:10）

搜索空间（两语料共用，与候选 3 相同）：lr log[1e-4, 1e-3]；max_seqlen {150, 200, 300}；λ_cma [0.5, 2]；prior_scale log[0.5, 8]；w_fine [0, 1]；λ_block log[0.05, 2]。固定 dropout .2、lamda_cof .05、hid/ffn 128、nhead 4、batch 32、50 epoch、topk_div 16。每 (语料, seed) 20 trial，目标 test (AP+ROC)/2，within 破下限剪枝，validation (AP+ROC)/2 选 checkpoint；同时记录 validation 选 trial 的 test 数字。

可证伪预期：
1. 规则 8 两语料 seed 234 过门；HateMM within 不像候选 3 那样大面积破下限。
2. 三 seed：HateMM 相对 `no_token_unmasked` 臂 pooled AP 或 ROC 高 ≥ .005 且另一项不低；HateClipSeg 不低；full 不低于候选 1 减一个 std（HateMM AP ≥ .644、ROC ≥ .837；HateClipSeg AP ≥ .693、ROC ≥ .665）。
3. 机制：`no_token_masked` 两语料三 seed pooled 都低于 full（空 token 必要）；`const_token` 低于 full 才能主张"证据摘要经空 token 进入"，否则只主张空 token 本身。
4. (2) 两语料都不成立则按规则 9 修改（≤ 3 轮）或归档；若 `no_token_masked` 也不低于 full，说明空 token 不起作用，方向归档。
5. 规则 4 复核追加：full 不低于 `gated_cma`（否则不能主张空 key 形式优于逐行门控）；`zero_value_sink` 与 `const_token` 之间的差决定"空 token 内容"是否可主张，`const_token` 与 full 之间的差决定"证据条件化"是否可主张。

## 5. 运行
```
python experiments/20260904_null_token_cma/search.py --corpus hatemm --seed 234 --out-root runs/20260904_null_token_cma
python experiments/20260904_null_token_cma/train.py --corpus hatemm --seed 234 --config <best hparams.json> --ablation no_token_unmasked --out-dir runs/20260904_null_token_cma/ablations/hatemm/seed234/no_token_unmasked
```
HateMM 在 uoa-lab1（空闲），HateClipSeg 在 uoa-lab3（候选 3 修订 2 记录链结束后）；本机在跑候选 3 修订 2 HateMM 的记录搜索。共享代码 `src/hier_evidence_common.py`。

## 6. 进度
- 2026-09-04 20:45：规则 6 code review PASS 无 BLOCKER（`REVIEW_RULE6.md`：no_token_unmasked 与候选 1 前向逐位一致，含 dropout 状态）。规则 4 复核 GO 5/10（`REVIEW_RULE4.md`）：四类不触发；最近先例 Mask-Align 的 leaky attention（ACL 2021，可学习 k_NULL/v_NULL 拼在交叉注意力 K/V 前；`const_token` 臂即此形式）、Sukhbaatar 2019 persistent memory、PyTorch add_bias_kv、sink/register 谱系（Xiao 2023、Darcet 2023、gpt-oss、Bondarenko、Miller）、条件化形式同 CoCoOp；同任务最近先例 Leaky Gated Cross-Attention（Lee et al. WACV 2022，弱监督多模态 TAL 的逐帧门控）。novelty 只能表述为"诊断（padding 是训练有测试无的偶然空 token、屏蔽有害）+ 裁定条件化的空 token 来源 + 在弱监督跨模态 MIL 骨干上的验证"，不是"空 token"本身。必须项：加 `gated_cma`、`zero_value_sink` 两臂（已加）；引用并对照上述先例；`no_token_unmasked` 仍是主对照；padding 屏蔽本身不作主张。搜索启动。
- 2026-09-04 20:10：提案、`model.py`（`NTCA`、`NullTokenCMA`）、`train.py`、`search.py`；五个结构臂前向/反向检查：token 拼在 key 位置 0，padding 屏蔽后 padding 注意力为 0，`no_token_unmasked` 与 MACIL-SD 相同（padding 注意力 .247 于随机输入）。等规则 4、规则 6 复核。

### 6.1 HateMM seed 234 诊断与确定性检查（2026-09-04 21:30–22:05，本机；`runs/20260904_null_token_cma/diag/hatemm/seed234/`）

用候选 1 seed 234 最优超参（trial 3，λ_cma 取 a2b/a2n 均值 .793）跑四个臂各一次（四个并行）：

| 臂 | AP / ROC / within |
|---|---|
| full | .655 / .841 / .634 |
| const_token | .654 / .843 / .646 |
| no_token_unmasked（候选 1 骨干） | .636 / .842 / .654 |
| no_token_masked | .625 / .825 / .643 |
| 参照：候选 1 no_ema 臂（同超参，README 9.6） | .659 / .844 / .646 |

同一设置串行重跑 3 次（GPU 上无其它任务）：no_token_unmasked 三次都是 .6358/.8416/.6536，full 三次都是 .6550/.8405/.6340，与并行跑的那次逐位相同。**训练是确定性的**，同一臂的数字不含单次随机波动；候选 1 骨干在本候选训练设置下比候选 1 自己记录的 no_ema 低 .023 AP，是设置差异（去掉伙伴网络改变了随机数消耗顺序、CMAL 两权重合并）造成的确定性偏移，不是噪声。因此 `no_token_unmasked` 是本候选内唯一有效的对照，候选 1 的记录数字只作参照；臂之间的比较仍要按协议看三 seed（每 seed 各自最优超参）的均值，因为超参点与 seed 的变化才是这里的主要方差来源（候选 1 三 seed AP std .013）。

单点读数：空 token 相对屏蔽 padding +.030 AP / +.016 ROC，相对候选 1 骨干 +.019 AP / ROC 持平 / within −.020（const_token −.008）；证据条件化相对常量 token 无增益。

### 6.2 HateMM seed 234 搜索（uoa-lab1，2026-09-04 20:07–21:49；`runs/20260904_null_token_cma/hatemm/seed234/`）

20 trial，12 个被 within 下限剪掉。best = trial 13（epoch 14；lr 1e-3、max_seqlen 150、λ_cma .93、prior_scale 7.95、w_fine .03、λ_block .11）：**.646 / .860 / .639**；validation 选中的也是 trial 13。规则 8 门过。对候选 1 seed 234（.661/.841/.650）：AP −.015、ROC +.019。消融（trial 13 超参）陆续出：no_token_unmasked .660/.854/.641（full 对它 −.014 / +.006 / −.002）。

### 6.3 HateMM seed 234 全部臂（uoa-lab1，2026-09-04 21:50–22:41；`runs/20260904_null_token_cma/ablations/hatemm/seed234/`）

全部用 trial 13 超参。数字来自各臂 `metrics.json`。

| 臂 | AP / ROC / within | 对 full |
|---|---|---|
| full | .646 / .860 / .639 | — |
| no_token_unmasked（候选 1 骨干，主对照） | .660 / .854 / .641 | +.014 / −.006 / +.002 |
| no_token_masked | .605 / .842 / .636 | −.041 / −.018 / −.003 |
| const_token | .623 / .839 / .646 | −.023 / −.021 / +.007 |
| shared_token | .662 / .851 / .643 | +.016 / −.009 / +.004 |
| zero_value_sink | .631 / .846 / .647 | −.015 / −.014 / +.008 |
| gated_cma | .635 / .838 / .645 | −.011 / −.022 / +.006 |
| no_input | .603 / .814 / .638 | −.043 / −.046 / −.001 |
| no_block | .662 / .846 / .646 | +.016 / −.014 / +.007 |
| no_prior | .592 / .837 / .616 | −.054 / −.023 / −.023 |
| mean_prior | .618 / .824 / .606 | −.028 / −.036 / −.033 |
| no_cmal | .645 / .840 / .643 | −.001 / −.020 / +.004 |
| no_verdict | .524 / .773 / .587 | −.122 / −.087 / −.052 |

单 seed 读数（判定等三 seed）：
- 屏蔽 padding 且不给空 token 最差（−.041 AP），与诊断一致：候选 1 骨干靠 padding 当空 token。
- full 对候选 1 骨干 AP 低 .014、ROC 高 .006；预注册第 2 条（pooled 一项高 ≥ .005 且另一项不低）在此 seed 不成立。
- 机制链 `no_token_masked` < `zero_value_sink` ≤ `const_token` < full 不成立：zero_value_sink (.631) 高于 const_token (.623)，shared_token (.662) 高于 full。按 seed 234，空 token 有用但"证据条件化"和"每模态独立 token"都没有增益，最简单的 shared_token 与候选 1 骨干持平。
- gated_cma 低于 full（−.011 AP / −.022 ROC），逐行门控在此 seed 不如空 key。
- 输入路径各臂（no_prior、mean_prior、no_verdict、no_input）方向与候选 1 一致。

### 6.4 HateClipSeg seed 234 搜索与全部臂（uoa-lab3，2026-09-04 22:05–23:14；`runs/20260904_null_token_cma/hateclipseg/seed234/`、`ablations/hateclipseg/seed234/`）

20 trial，2 个被 within 下限剪掉。best = trial 15（epoch 3；lr 2.3e-4、max_seqlen 300、λ_cma 1.70、prior_scale .96、w_fine .52、λ_block .57）：**.702 / .687 / .550**。规则 8 门过。对候选 1 seed 234（.695 / .679 / .546，`runs/20260903_hier_evidence_mil/hateclipseg/seed234/study_summary.json`）：AP +.007、ROC +.008。

全部臂用 trial 15 超参：

| 臂 | AP / ROC / within | 对 full |
|---|---|---|
| full | .702 / .687 / .550 | — |
| no_token_unmasked（候选 1 骨干，主对照） | .672 / .660 / .548 | −.030 / −.027 / −.002 |
| no_token_masked | .673 / .661 / .548 | −.029 / −.026 / −.002 |
| const_token | .680 / .668 / .551 | −.022 / −.019 / +.001 |
| shared_token | .702 / .687 / .551 | .000 / .000 / +.001 |
| zero_value_sink | .673 / .657 / .522 | −.029 / −.030 / −.028 |
| gated_cma | .683 / .676 / .531 | −.019 / −.011 / −.019 |
| no_input | .664 / .651 / .543 | −.038 / −.036 / −.007 |
| no_block | .606 / .569 / .535 | −.096 / −.118 / −.015 |
| no_prior | .667 / .660 / .535 | −.035 / −.027 / −.015 |
| mean_prior | .683 / .675 / .536 | −.019 / −.012 / −.014 |
| no_cmal | .682 / .666 / .533 | −.020 / −.021 / −.017 |
| no_verdict | .582 / .567 / .527 | −.120 / −.120 / −.023 |

单 seed 读数：
- HateClipSeg 上 padding 屏蔽与否无差别（no_token_masked ≈ no_token_unmasked），与 HateMM 不同：HateClipSeg 片段短、max_seqlen 300 时 padding 占比小。
- full 对候选 1 骨干 +.030 AP / +.027 ROC，within 持平；预注册第 2 条 HateClipSeg 一侧成立。
- 机制链 no_token_masked (.673) < zero_value_sink (.673) ≤ const_token (.680) < full (.702)：成立（前两者持平）。纯吸收位置无增益，常量 token +.008，证据条件化再 +.022。
- shared_token 与 full 逐位相同数字（.702/.687/.551）：两语料 seed 234 都显示按模态分两个 token 没有必要，起作用的是"证据条件化的空 token"。
- gated_cma 低于 full（−.019 AP），且 within −.019。
- 输入路径各臂方向与候选 1 一致。

两语料 seed 234 合并读数（判定仍等三 seed）：证据条件化空 token 在 HateClipSeg 上 +.030/+.027，在 HateMM 上 −.014/+.006；const_token 两语料都低于 full 约 .02 AP；shared_token 两语料都不低于 full。若三 seed 维持此模式，方法主张应改为"单个证据条件化空 token"（shared 形式），full 的按模态分不作主张。

### 6.5 HateClipSeg seed 2025 搜索与全部臂（uoa-lab3，2026-09-04 23:14–2026-09-05 00:15；`runs/20260904_null_token_cma/hateclipseg/seed2025/`、`ablations/hateclipseg/seed2025/`）

20 trial，6 个被 within 下限剪掉。best = trial 4（epoch 3；lr 3.3e-4、max_seqlen 150、λ_cma 1.20、prior_scale 4.13、w_fine .92、λ_block .21）：**.707 / .685 / .550**。规则 8 门过。对候选 1 seed 2025（.706 / .698 / .560，`runs/20260903_hier_evidence_mil/hateclipseg/seed2025/study_summary.json` trial 19）：AP +.001、ROC −.013。

全部臂用 trial 4 超参：

| 臂 | AP / ROC / within | 对 full |
|---|---|---|
| full | .707 / .685 / .550 | — |
| no_token_unmasked（候选 1 骨干，主对照） | .695 / .666 / .554 | −.012 / −.020 / +.004 |
| no_token_masked | .695 / .666 / .554 | −.012 / −.020 / +.004 |
| const_token | .695 / .666 / .554 | −.012 / −.020 / +.003 |
| shared_token | .706 / .684 / .551 | −.001 / −.001 / .000 |
| zero_value_sink | .691 / .657 / .552 | −.016 / −.029 / +.002 |
| gated_cma | .692 / .659 / .551 | −.015 / −.026 / +.001 |
| no_input | .706 / .685 / .559 | −.001 / .000 / +.008 |
| no_block | .694 / .662 / .553 | −.013 / −.024 / +.003 |
| no_prior | .649 / .647 / .550 | −.058 / −.038 / .000 |
| mean_prior | .668 / .664 / .523 | −.039 / −.021 / −.028 |
| no_cmal | .693 / .663 / .551 | −.013 / −.022 / .000 |
| no_verdict | .597 / .570 / .535 | −.109 / −.116 / −.016 |

单 seed 读数：
- max_seqlen 150 下 HateClipSeg 片段无 padding，no_token_masked 与 no_token_unmasked 逐位相同（.69481/.6655）。const_token 与它们只差第五位小数（.69482/.66553）：常量 token 在此设置下学成了没有作用的位置。
- full 对候选 1 骨干 +.012 AP / +.020 ROC；shared_token 与 full 相同（−.001）。
- **与 seed 234 不一致的一点**：no_input（裁定四列不拼入输入、c 置零，即空 token 只剩常量）.706/.685，与 full 相同。也就是说 seed 2025 这个超参点上（w_fine .92，先验路径占主导），去掉输入路径不掉分，而 const_token（保留输入路径、去掉条件化）掉 .012/.020。这两个臂都没有证据条件化，一个与 full 持平、一个低 .012，说明此 seed 上 .01–.02 的差异不能单独归因于证据条件化；"证据条件化是增益来源"要看三 seed 均值是否稳定高于 const_token。
- zero_value_sink、gated_cma 都低于 full（−.016 / −.015 AP）。
- no_prior、mean_prior、no_verdict 方向与候选 1 一致；no_block、no_cmal 各 −.013。
