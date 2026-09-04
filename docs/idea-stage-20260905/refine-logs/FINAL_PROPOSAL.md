# FINAL PROPOSAL — 裁定条件化密度估计（Verdict-Conditioned Density Estimation）：coarse-first 条件标注模型 + scale–rank 分解头

日期 2026-09-05。来源：`docs/idea-stage-20260905/IDEA_REPORT.md` 推荐候选（簇 C7 + C1，C13 作 C1 子开关）。经两轮 Codex（gpt-5.6-sol，ultra）评审修订；评审记录 `REVIEW_SUMMARY.md`，修订记录 `REFINEMENT_REPORT.md`。**只做了纸面验证，未跑任何实验（用户约束：GPU 全部被预注册链占用）。**

## 0. Problem Anchor（冻结）
弱监督 hateful video localization，HateMM 与 HateClipSeg 一套架构；冻结特征（I3D / VGGish / BERT）与冻结 Qwen2.5-VL-7B 两粒度二值裁定（K30、K4）；比较指标 = test pooled AP / ROC（规则 8），within-video macro ROC 只作下限（HateMM ≥ .632、HCS ≥ .524）。论文并列报告 within，**不**从密度项 claim localization 改善（评审 §1.22 记录在案）。

## 1. 方法命题（一句话）
MIL 定位器的逐秒分数分解为"视频级仇恨比例项"与"视频内排序项"；比例项由一个对两粒度 VLM 裁定的多分辨率标注模型监督，该标注模型采用 coarse-first 的有向发射结构，使细粒度裁定的可靠性以粗粒度裁定为条件。

## 2. 主贡献：模块 3 = coarse-first 条件标注模型（替换候选 1 的四个全局发射率）
生成模型（每视频 K=30 细窗、J=4 粗块，隐状态 s_t ∈ {0,1} 马尔可夫链 (A, p0)，h_j = OR_{t∈j} s_t，增广状态 (s_t, h_t) 精确前向后向，与 `src/verdict_hmm.py` 相同）：
- 粗裁定：P(b4_j | h_j) = Bernoulli(q4 若 h_j = 1，否则 r4)（全局，不变）。
- 细裁定（**新**）：P(b30_t = 1 | s_t = k, b4_block(t), b30_{t−1}) = sigmoid(θ_k · [1, b4_block(t), b30_{t−1}])，k ∈ {0,1}，t = 1 时 b30_0 := 0。
- 联合似然 P(b4, b30 | s, h) = Π_j P(b4_j | h_j) · Π_t P(b30_t | s_t, b4_block(t), b30_{t−1})：所有条件变量都是观测量，是合法有向模型（评审第 1 轮 §1.6 修正：去掉"块内其它细裁定计数""b30_{t+1}""粗发射含细计数"这三个造成循环的协变量）。前向后向仍精确。
- 参数估计（两阶段，不称全参数 ML-EM，评审第 2 轮 §3）：θ_0（s=0 侧）只在**负例视频**上用 logistic regression 拟合（负例全程 s = h = 0，规则允许的唯一标签信息），只对负例中出现过的上下文 (b4, b30_{t−1}) 组合作可识别性陈述；固定 θ_0 后，θ_1、q4、r4、A、p0 在正例上做标准 EM（M 步 = 后验加权 logistic regression）。可识别性只 claim r 侧；q 侧、p0、A 用 CPU 合成数据多起点恢复实验报告。
- 输出 ℓ_t、P(s_t)、P(h_j) 进网络的方式与候选 1 完全相同（先验项 α·ℓ_t/L、输入列、块级 MIL 软标签）。训练期网络输入的后验必须**label-free、inference-mode、out-of-fold**（对训练视频用 5 折外拟合的参数生成；评审第 2 轮 §2）。
- 版本 1 不含"经可微前向后向由 MIL 损失精调发射参数"阶段（评审：提升最低、校准风险最高）。
- 理论包装：IOHMM 的输入依赖发射（Bengio & Frasconi 1995）/ 实例依赖标签噪声（Xia et al. NeurIPS 2020）在多分辨率弱源标注模型（Dugong NeurIPS 2019）上的实例化；差异 = 条件变量是另一分辨率的裁定与前一细裁定（源间依赖显式化），误报侧由负袋识别。最近先例 CHMM（ACL 2021）、FABLE（AISTATS 2023）、Hyper Label Model（ICLR 2023）、Linked HMM（AAAI 2020）必须正面引用。

**CPU 门（任何 GPU 运行之前，只重拟合 HMM，用统一评测器评 posterior-alone）**：
1. HateMM：条件后验单独 AP 与 ROC 都高于 K4-only 后验（.591 / .851）至少 .005，且高于 4-cell lookup 表（对 ℓ_t 的每格偏移）；HCS 不低于 .698 / .661。
2. 比例有效性（评审第 2 轮唯一最能提分项）：在有时间 GT 的 test 上，条件后验的每视频比例估计 q_v = mean_t P(s_t) 的 MAE、偏差、与 GT 比例的相关、可靠性图，都优于 global 与 K4-only；否则 q_v 只能称"HMM 派生标量目标"，模块 2 的比例损失 thesis 删除。
3. 机制隔离：`no_b4`（发射只条件 b30_{t−1}）与 `no_bprev`（只条件 b4）两臂在同一表；若增益全来自自回归项，则不能 claim"粗裁定条件化细裁定可靠性"。
不过门 → 模块 3 归档，只记录 K4-only 调温结果。

## 3. 支撑贡献：模块 2 = scale–rank 分解头（替换候选 1 的单一 logit + top-k bag；骨干其余不变，EMA 伙伴删除）
- a_t = z_t + α·ℓ_t/L（候选 1 的完整分数，含先验；评审 §1.15：先加先验再中心化）
- r_t = a_t − mean_{t 有效} a_t（视频内零均值排序项）
- s_v = MLP(φ_v) + w_μ·μ_v + b；φ_v = [细窗触发比例、块触发比例、4 格一致直方图、最长 run]（**不含** mean P(s_t)，评审 §1.13 去掉 target-copy）⊕ **detach** 后的 fc_v / fc_a 时间均值与最大值（s_v 不向共享投影回传梯度）；μ_v = 该视频 BERT 逐秒均值（`text_decompose` 开时，BERT 只以偏差 d_t = x_t − μ_v 进入 fc_a）。
- score_t = s_v + r_t。推理时 s_v 是视频常数，**不能改变视频内排序**；训练期 ∂L/∂r_t 仍依赖 s_v（评审第 2 轮 §6），所以只 claim"推理时顺序保持"，within 在每个臂监控。
- 损失：(i) bag BCE：sigmoid(s_v + mean top-⌈T/16⌉ r_t) 对视频标签；(ii) 比例损失作用在**最终逐秒概率**上：(1/T) Σ_t sigmoid(score_t) → q_v（正例 q_v = 模块 3 的期望仇恨比例，负例 0；权重 λ_prop 进搜索）；(iii) 块级 MIL 作用在块内的总分 s_v + r_t（绝对量，评审 §1.17）；(iv) CMAL 不变。
- 理论包装：learning from label proportions / cardinality potential——pooled 分数 = 视频间比例项 + 视频内排序项，对应测得的方差分解（92% / 79% 在视频间）。最近先例 MSL（AAAI 2022，视频级概率乘性抑制片段分数）、ARMS（TMM 2024，自估比例的 ratio 损失）、3C-Net（ICCV 2019）必须正面引用；差异三轴：比例目标来源（外部标注模型 vs 二值标签/自估）、作用位置（加性视频标量 vs 片段分数门）、可识别性（中心化排序项 + 精确重参数化对照）。
- 两篇 2026 审计（arXiv 2608.21854、2608.11985：pooled frame AUC 主要度量跨视频排序）作为"诊断"引用，本方法是把该诊断变成设计。

**必做对照**：(i) 精确重参数化臂 s_v = mean a_t、r_t = a_t − mean（零参数，前向分数与候选 1 逐位相同）——学习的 s_v 必须胜过它；(ii) 固定 s_v = logit q_v；(iii) 候选 1（无 s_v、不中心化）；(iv) s_v 只读裁定统计；(v) λ_prop = 0（同一 head）；(vi) 只搬块级 MIL 到总分、不加 s_v（relocation-only）；(vii) 保留 s_v、删除注意力层（作为"注意力 = 密度路径"的经验发现报告，不作贡献）；(viii) 改编 MSL 乘性视频门（同骨干）。

`text_decompose`：D 开时 μ_v 进 s_v、d_t 进 fc_a；D 关时定义为独立标量文本头 score_t = a_t + w·μ_v + b（d_t 进 fc_a），因此 T 是独立开关、做完整 2^3 八格（评审第 2 轮 §8–9）；T = 0 指原始 BERT 拼接（候选 1）。成功须在 within ≥ 候选 1 − .005 的前提下 HateMM ROC 不低于 no_text 臂（.869）。

## 4. 决策协议（项目规则 7/8/9/14 为正式门；评审的逐开关标准作论文 claim 标准）
- 正式门：规则 8（两语料 pooled 全过 baseline 表 + within 下限；seed 234 筛选，2025/3407 确认，领先幅度 ≥ std）；规则 9（对候选 1 记录任一语料 pooled ≥ .01 才保留改进）；规则 14(g)（去核心机制两语料 pooled 明显下降）。
- 论文 claim 标准（每个开关，从 111 leave-one-out，三 seed 配对均值，seed 级 CI）：HateMM (AP+ROC)/2 ≥ +.020 且两项 pooled 都不低于 −.005；HCS (AP+ROC)/2 ≥ +.012 且两项不低于 −.005；两语料 within ≥ 候选 1 − .005。不满足的开关从 claim 中删除。
- 噪声控制（预先固定，不按结果追加）：每臂在该 seed best-trial 超参下跑 3 条随机数流，报均值。
- **诚实的预期（评审第 2 轮 §10）**：按作者自己的预期（模块 3 端到端 HateMM +.00~+.02 AP、HCS ≈ 0；模块 2 HateMM +.00~+.02、HCS ROC +.005~+.010；文本 HateMM ROC ≤ +.027），三个开关都可能达不到论文 claim 标准。因此顺序是：CPU 门先证伪模块 3 与 q_v；模块 2 只有在精确重参数化对照被学习 head 明显超过时才继续。

## 5. 主动放弃的复杂度
MIL 精调发射参数；内容条件化发射（候选 2 失败路径）；块级加性项（D6）；逐秒精度加权 PoE 融合（C8，校准暴露）；块内受限注意力 / 块级 register（候选 3 的块尺度版本）；query 门与 sink（候选 4 的证据）；HSMM / 秒级 IOHMM（C9，within 风险高，作备选）。

## 6. 剩余风险
1. 增量落在 ±.02 噪声内（主风险）。2. HateMM val/test 对 K30 可靠性方向相反，train 拟合的 θ 未必移动 test 后验。3. HCS 上模块 2 几乎无空间。4. 顶会评审会把整体归为"已有部件在新任务上的经验组合"（application novelty）；项目规则 4 不阻断，但论文需按第 2、3 节的定位写。5. test 驱动的搜索协议是项目裁定，评审视为确认性失效，论文须如实写明。
