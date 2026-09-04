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

### 6.6 HateClipSeg 三 seed 汇总（uoa-lab3，2026-09-04 22:05–2026-09-05 01:17；`runs/20260904_null_token_cma/hateclipseg/seed<seed>/study_summary.json`、`ablations/hateclipseg/seed<seed>/<arm>/metrics.json`；汇总脚本输出按此转录）

seed 3407：20 trial，5 个剪掉，best = trial 12（epoch 5；lr 1.6e-4、max_seqlen 150、λ_cma .51、prior_scale 2.11、w_fine .98、λ_block .21）：.710 / .699 / .568。

| 臂 | seed 234 | seed 2025 | seed 3407 | 三 seed 均值 AP / ROC / within | 对 full |
|---|---|---|---|---|---|
| full | .702/.687/.550 | .707/.685/.550 | .710/.699/.568 | **.706 / .690 / .556**（std .004/.008/.010） | — |
| no_token_unmasked（候选 1 骨干，主对照） | .672/.660/.548 | .695/.666/.554 | .691/.672/.555 | .686 / .666 / .552 | −.020 / −.024 / −.004 |
| no_token_masked | .673/.661/.548 | .695/.666/.554 | .691/.672/.555 | .686 / .666 / .552 | −.020 / −.024 / −.004 |
| const_token | .680/.668/.551 | .695/.666/.554 | .689/.671/.557 | .688 / .668 / .554 | −.018 / −.022 / −.002 |
| shared_token | .702/.687/.551 | .706/.684/.551 | .711/.700/.568 | .706 / .690 / .557 | .000 / .000 / .000 |
| zero_value_sink | .673/.657/.522 | .691/.657/.552 | .690/.657/.555 | .685 / .657 / .543 | −.022 / −.033 / −.013 |
| gated_cma | .683/.676/.531 | .692/.659/.551 | .695/.662/.551 | .690 / .666 / .545 | −.016 / −.025 / −.012 |
| no_input | .664/.651/.543 | .706/.685/.559 | .691/.666/.550 | .687 / .667 / .550 | −.020 / −.023 / −.006 |
| no_block | .606/.569/.535 | .694/.662/.553 | .686/.660/.551 | .662 / .630 / .546 | −.045 / −.060 / −.010 |
| no_prior | .667/.660/.535 | .649/.647/.550 | .652/.654/.538 | .656 / .654 / .541 | −.051 / −.037 / −.015 |
| mean_prior | .683/.675/.536 | .668/.664/.523 | .688/.677/.538 | .680 / .672 / .532 | −.027 / −.018 / −.024 |
| no_cmal | .682/.666/.533 | .693/.663/.551 | .691/.666/.550 | .689 / .665 / .545 | −.018 / −.025 / −.011 |
| no_verdict | .582/.567/.527 | .597/.570/.535 | .598/.571/.530 | .592 / .569 / .531 | −.114 / −.121 / −.025 |

候选 1 记录（`runs/20260903_hier_evidence_mil/hateclipseg/seed<seed>/study_summary.json`）：seed 234 .695/.679/.546、2025 .706/.698/.560、3407 .696/.666/.553，均值 .699 / .681 / .553。

HateClipSeg 三 seed 判定（README 第 4 节预注册）：
- 规则 8：三 seed 全部过门（AP > .562、ROC > .528、within ≥ .524）。
- 第 2 条：对主对照 no_token_unmasked +.020 AP / +.024 ROC，三 seed 每个都高（+.030/+.027、+.012/+.020、+.019/+.027）；对候选 1 记录 +.007 AP / +.009 ROC；full ≥ 候选 1 − 1 std（.693 / .665）成立。
- 第 3 条：no_token_masked 三 seed 都低于 full（−.029、−.012、−.019 AP），空 token 必要；const_token 三 seed 都低于 full（−.022、−.012、−.021 AP），证据条件化可主张。
- 第 5 条：full 高于 gated_cma（+.016 AP / +.025 ROC）；机制链 no_token_masked (.686) ≈ zero_value_sink (.685) ≤ const_token (.688) < full (.706) 成立：纯吸收位置无增益，常量 token +.002，证据条件化 +.018。
- shared_token 三 seed 与 full 相同（均值差 .000）：按模态分两个 token 不必要。
- HateClipSeg 上 padding 屏蔽无作用（no_token_masked = no_token_unmasked），与 HateMM 不同。
- 单 seed 例外：seed 2025 的 no_input 与 full 持平（6.5 节），三 seed 均值 no_input −.020。

### 6.7 HateMM seed 2025 全部臂（uoa-lab1，2026-09-05 00:25–01:23；`runs/20260904_null_token_cma/hatemm/seed2025/`、`ablations/hatemm/seed2025/`）

20 trial，14 个剪掉。best = trial 11（epoch 4；lr 1e-3、max_seqlen 200、λ_cma .57、prior_scale 6.23、w_fine .01、λ_block .05）：**.645 / .844 / .644**。规则 8 门过。候选 1 seed 2025 记录 .643/.838/.644。

| 臂 | AP / ROC / within | 对 full |
|---|---|---|
| full | .645 / .844 / .644 | — |
| no_token_unmasked（主对照） | .617 / .836 / .633 | −.028 / −.008 / −.011 |
| no_token_masked | .618 / .846 / .644 | −.027 / +.002 / .000 |
| const_token | .621 / .836 / .644 | −.024 / −.008 / .000 |
| shared_token | .630 / .841 / .630 | −.015 / −.003 / −.014 |
| zero_value_sink | .601 / .831 / .631 | −.044 / −.013 / −.013 |
| gated_cma | .653 / .836 / .630 | +.008 / −.008 / −.014 |
| no_input | .625 / .822 / .647 | −.020 / −.022 / +.003 |
| no_block | .631 / .841 / .647 | −.014 / −.003 / +.003 |
| no_prior | .573 / .787 / .624 | −.072 / −.057 / −.020 |
| mean_prior | .627 / .838 / .643 | −.018 / −.006 / −.001 |
| no_cmal | .616 / .825 / .641 | −.029 / −.019 / −.003 |
| no_verdict | .496 / .773 / .624 | −.149 / −.071 / −.020 |

两 seed 均值（234 + 2025）：full .646/.852，no_token_unmasked .639/.845（−.007/−.007），const_token .622/.837，shared_token .646/.846，gated_cma .644/.837。等 seed 3407（搜索 + 消融，uoa-lab1 进行中）后判定。

### 6.8 HateMM 三 seed 汇总（uoa-lab1，2026-09-04 20:07–2026-09-05 04:12；`runs/20260904_null_token_cma/hatemm/seed<seed>/study_summary.json`、`ablations/hatemm/seed<seed>/<arm>/metrics.json`）

seed 3407：20 trial，17 个剪掉（前 14 个全部 within < .632），best = trial 14（epoch 2；lr 1.6e-4、max_seqlen 300、λ_cma 1.09、prior_scale .84、w_fine 1.00、λ_block .30）：.612 / .825 / .635。候选 1 的 seed 3407 搜索同样 16/20 剪掉。

| 臂 | seed 234 | seed 2025 | seed 3407 | 三 seed 均值 AP / ROC / within | 对 full |
|---|---|---|---|---|---|
| full | .646/.860/.639 | .645/.844/.644 | .612/.825/.635 | **.634 / .843 / .639**（std .019/.018/.004） | — |
| no_token_unmasked（候选 1 骨干，主对照） | .660/.854/.641 | .617/.836/.633 | .610/.822/.639 | .629 / .837 / .638 | −.005 / −.006 / −.002 |
| no_token_masked | .605/.842/.636 | .618/.846/.644 | .641/.838/.649 | .621 / .842 / .643 | −.013 / −.001 / +.004 |
| const_token | .623/.839/.646 | .621/.836/.644 | .608/.828/.630 | .617 / .834 / .640 | −.017 / −.009 / +.001 |
| shared_token | .662/.851/.643 | .630/.841/.630 | .610/.826/.634 | .634 / .839 / .636 | −.001 / −.004 / −.004 |
| zero_value_sink | .631/.846/.647 | .601/.831/.631 | .636/.833/.641 | .623 / .837 / .639 | −.012 / −.006 / .000 |
| gated_cma | .635/.838/.645 | .653/.836/.630 | .628/.829/.632 | .638 / .835 / .636 | +.004 / −.009 / −.004 |
| no_input | .603/.814/.638 | .625/.822/.647 | .588/.796/.633 | .605 / .810 / .639 | −.029 / −.033 / .000 |
| no_block | .662/.846/.646 | .631/.841/.647 | .613/.822/.637 | .635 / .836 / .644 | +.001 / −.007 / +.004 |
| no_prior | .592/.837/.616 | .573/.787/.624 | .616/.830/.647 | .593 / .818 / .629 | −.041 / −.025 / −.010 |
| mean_prior | .618/.824/.606 | .627/.838/.643 | .623/.833/.626 | .623 / .832 / .625 | −.012 / −.011 / −.014 |
| no_cmal | .645/.840/.643 | .616/.825/.641 | .595/.805/.631 | .619 / .823 / .638 | −.016 / −.020 / −.001 |
| no_verdict | .524/.773/.587 | .496/.773/.624 | .482/.753/.612 | .501 / .766 / .608 | −.134 / −.077 / −.032 |

候选 1 记录（`runs/20260903_hier_evidence_mil/hatemm/seed<seed>/study_summary.json`）：seed 234 .661/.841/.650、2025 .643/.838/.644、3407 .668/.848/.646，均值 .657 / .842 / .646。

## 7. 三 seed 判定（2026-09-05 04:30，按第 4 节预注册逐条）

| 条 | HateClipSeg | HateMM |
|---|---|---|
| 规则 8 门 | 三 seed 全过 | 三 seed 全过（3407 .612/.825/.635 对门 .573/.807/.632） |
| 第 2 条 对主对照 pooled 高 ≥ .005 且另一项不低 | **成立**：+.020 AP / +.024 ROC，三 seed 每个都高 | 勉强：+.0050 AP / +.0058 ROC（三 seed −.014 / +.028 / +.002） |
| 第 2 条 full ≥ 候选 1 − 1 std | 成立：.706 ≥ .693、.690 ≥ .665；且对候选 1 +.007 / +.009 | **不成立**：AP .634 < .644（对候选 1 −.023）；ROC .843 ≥ .837 |
| 第 3 条 no_token_masked 三 seed 都低于 full | 成立（−.029 / −.012 / −.019） | **不成立**：seed 3407 反高 .029（.641 对 .612）；均值 −.013 |
| 第 3 条 const_token 三 seed 都低于 full | 成立（−.022 / −.012 / −.021） | 成立但弱（−.023 / −.024 / −.004）；均值 −.017 |
| 第 5 条 full ≥ gated_cma | 成立（+.016 / +.025） | **AP 不成立**（−.004），ROC 成立（+.009） |
| 第 5 条 链 masked < zero_value ≤ const < full | 成立（.686 ≈ .685 ≤ .688 < .706） | 不成立：zero_value .623 > const .617 |
| shared_token 对 full | 相同（.000） | 相同（−.001 AP） |

结论：
1. HateClipSeg 上方法成立且机制消融齐全：证据条件化空 token 对候选 1 骨干 +.020 AP / +.024 ROC，对候选 1 记录 +.007 / +.009，空 token 必要、证据条件化必要、纯吸收位置无用、优于逐帧门控，单个共用 token 即可。
2. HateMM 上对主对照只有 +.005 / +.006，机制消融不齐（seed 3407 的 no_token_masked 反高、gated_cma AP 持平），且绝对数字低于候选 1 .023 AP。规则 13（一个方法两语料）不满足。
3. HateMM 绝对数字低的直接原因不是空 token：主对照（候选 1 骨干、本候选训练设置、无伙伴网络 EMA）三 seed .629 / .837，本身就比候选 1 记录低 .028 AP。候选 1 的 no_ema 消融只在 seed 234 做过（.659 对 .661），本候选据此去掉了 EMA 伙伴网络；三 seed 下这个设置差异（去伙伴网络、CMAL 两权重合并、随机数消耗顺序）值 .028 AP，在 seed 2025（.617 对 .643）与 3407（.610 对 .668）尤其大。因此"候选 4 对候选 1"不是同设置比较。
4. 规则 9：对最强训练 baseline（候选 1 记录）没有任一语料 ≥ .01 的 pooled 提升（HateClipSeg +.007 / +.009），按字面应归档；但预注册主对照下 HateClipSeg +.020 / +.024 由设计导致且消融齐全，HateMM 的缺口可归因于训练设置差异而非空 token。按用户 2026-09-03 裁定（骨干模块涨点可小、须由设计导致且消融齐全）进入规则 9 修改第 1 轮，修改内容只有一项：**把空 token 放回候选 1 的原训练设置**（保留伙伴网络 EMA、CMAL 两权重、候选 1 搜索空间），用 shared 形式（单个证据条件化 token，两语料三 seed 都与 full 相同、参数更少），主对照改为候选 1 记录本身，其它全同。这样候选 1 与修订 1 只差空 token 一项，HateMM 的比较才成立。修订 1 见第 8 节。
