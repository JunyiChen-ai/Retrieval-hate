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
