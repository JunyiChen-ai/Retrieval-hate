# TERMINUS — Round-2 MLLM-integration campaign(goal: ≥+3 acc,2026-07-13 → 07-14)

触发:dali_autoresearch 协议 stale_count=4(连续 4 个结构性方向以预注册负结果关闭)→ flag-for-human。goal 钩子(MLLM meaningfully+novelly integrated,≥+3 test acc)在当前约束空间内已不可通过继续搜索满足;剩余选项全部需要用户裁决。

## 1. 本轮关闭的 7 条路线(第 15–21 条;全部三层验证:执行核验 + 独立判决复核 + 归档)

| # | 路线 | 死因 | GPU 成本 | 关键文档 |
|---|---|---|---|---|
| 15 | A 线 lb_scgp_global(证书→Gram) | G0-cond 探针:缓存 91–93% 常数,oracle@覆盖率低于线一个量级,v3 不可行(parse-ok 部分即噪声) | 0(M2/M3 264 GPU-h 拦下) | A_LINE_PAUSE_DECISION.md |
| 16 | C1 RA-HMD 式 QLoRA 顺序两阶段 | 锚论文消融定价 +0.7 + 自家 P9/P9b;实测 DEV kNN −0.02(13039) | ~40s | C1_KILL_REVIEW.md, C1_SETTLE_DEV_RECORD.md |
| 17 | C3-target(真预测器) | 校准探针:Qwen-7B 预测 target 条件信息 ≈0,MHC 反信息 | 0 | C3_REAL_PREDICTOR_PROBE.md |
| 18 | C2 SAV 稀疏注意力头 | F-G1 KILL,修正机器后更强(机器缺陷曾抬高 SAV);稀释假说证伪(MHC-EN 数据受限) | ~1.5h | SAV_F1_VERDICT_REVIEW.md |
| 19 | C3-nontarget 密集推理文本 | 融合关卡三条规则全败;CLIP-only +0.053 = 编码器冗余(信息已在 Qwen 通路) | ~1.4h(生成) | C3_FUSION_PROBE_RECORD.md |
| 20 | B1 frozen-Qwen encoder × ZH | 双协议 FAIL(配对均值 −0.011/−0.000,1/3 同号);P8c 语言匹配假说否定;ZH 0.8537 系 LoRA 杠杆 | ~11min | B1_VERDICT_REVIEW.md |
| 21 | B2 Qwen-32B encoder(scale 轴) | goal FAIL:HateMM 32B 介于 CLIP 与 7B 之间(scale 退步);MHC-EN/ZH 低于 CLIP;32B-vs-7B 全败 | ~2h | B2_VERDICT_REVIEW.md |

累计:**21 条预注册负结果**(前 14 条见 CAMPAIGN_mllm_method_role.md / exp-tarc-t0.md)。

## 2. 科学诊断(至此五次独立命中,视为坚实)

**D1 冗余律:** 低带宽决策侧 MLLM 信号对强冻结表征条件冗余("probe 过、训练平")。本轮新增证据:C3-target oracle→真实坍缩、C3-nontarget 融合坍缩(3.3×)、A 线证书零条件信息。
**D2 表征律:** 唯一 +3 级杠杆 = MLLM-as-encoder,但**仅 HateMM**;EN 是数据/标签受限(SAV 复核),ZH 是决策层不兑现(B1/B2 的 ROC>acc 反差),**scale 不是缺失变量**(B2:32B 在锚数据集上比 7B 退步,单调性反证 72B)。
**D3 测量律:** ±1–2pt 噪声地板 + 小 val 使一切 <3pt 决策侧效应不可测。
**新增方法学制度(经实战校准):** G0-cond 条件信息 gate(强制 label-oracle 校准 arm 达满 headroom;permutation null 必须测分布 ≥100 seeds);判决复核制度本轮 5 次触发、抓出 2 个方向相反的探针机器 bug + 1 个假阳性 null 假象。

## 3. 仍然成立的资产(论文可用)

- HateMM encoder 效应:7B frozen +5.3–5.6 acc 双协议 3/3(项目最强正效应);32B 复现方向但弱于 7B(scale 消融现成)。
- MLLM earned roles:encoder(HateMM)+ localizer(P6/P10-b)+ guard-rail/audit;21 条负结果的完整归因链 = 论文的 negative-results 主体叙事。
- 全链方法学:G0-cond gate、校准强制项、permutation-null 分布化、判决复核制度——可写成 methods/appendix 的 rigor 卖点。

## 4. 需要用户裁决的剩余选项

(a) **72B-AWQ encoder**:唯一未跑的 scale 点,但 B2 的单调退步使先验 ≈0;需抽取脚本加 AWQ 路径 + autoawq 安装 + delta-check + 41G 下载。**不建议**(dead-axis grinding),除非你另有机制假设。
(b) **LoRA 族杠杆**(ZH LoRA-Qwen 0.8537 vs CLIP 0.8027 的未配对差距):按你的框架 LoRA-SFT 属非 novelty 杠杆且 P9 家族已关;若你愿意重新定位其角色(如作为 encoder 适配的正式消融而非 novelty 主张),这是数据上最接近 +3 的未验证配对。**需要你重新划 novelty 边界才可动。** → 已实测,见 §6。
(c) **goal 重议**:以 HateMM-only encoder +5 作为"MLLM meaningful integration"的论文主张,把 21 条负结果做成 negative-results/rigor 贡献(现有 TERMINUS/OPTION_KITS + 本文档已备料)。
(d) 基建裁决:~120 个未推送 commit;disk_guard quota 解析 bug(当前全盲,建议修);lora_p9 83G / Retrieval 41G 未备份未裁决;A 线 M-A/realbank is_science 遗留。

## 5. 协议状态

dali_autoresearch 转入监控模式(心跳照常,不再对已关闭轴烧 GPU);全部状态在 autoresearch/goal_mllm_plus3/state/;本轮资源账:GPU ≈ 5.5h 实验 + 0 浪费在被 gate 拦截的路线上,单提纪律零违规,1 次探针门控基础设施重试(上游 CDN 瞬断)。

## 6. B3 补遗(2026-07-14,判决后)

选项 (b) 的"未验证配对"已实测。B3 迷你仪式(预注册 → 复核 APPROVED(绑定 marginal 语言)→ 单提 job 13150(约 2.5h JobHeldUser 后 2m46s COMPLETED)→ 原始转录 → 独立判决复核,`refine-logs/B3_VERDICT_REVIEW.md`)在 current-code `enc3seed` 同码同种子协议下,把 arcbase 12223-25 的 LoRA-vs-CLIP 预览转成正式配对判决。

**实测配对结果(MHC-ZH,LoRA-Qwen vs frozen-CLIP,3 种子,job 13150 vs 13115):**
- final-epoch:均值 Δacc **+0.0313** / ΔmF1 **+0.0453**,两指标均 **3/3 同号**。
- val-selected:均值 Δacc **+0.0246**(< +0.030,AND 规则在 acc 上失败)/ ΔmF1 **+0.0339**(3/3 同号)。
- G-repro 硬门 **bit-exact** 复现 arcbase 12223-25(6/6 读数 4dp,零失配)。
- **绑定语言(`B3_PREREG_REVIEW.md` §2.2,逐字,不得升级):`final-epoch: PASS (MARGINAL); val-selected: FAIL`。**

**三条强制敏感度事实(§2.2 要求全列):**
1. **贴边:** 均值 Δacc +0.0313 仅高出 +0.030 门 **+0.0013(≈门的 4%)**——这就是整个 pass 的全部余量。
2. **逐种子不均:** seed2 的 Δacc = **+0.0201,本身低于逐种子 +0.030 门**;pass 靠 seed0/1 与 F1(+0.0453 干净过线),而非均匀的逐种子余量。
3. **余量 ≪ 种子间散布:** +0.0013 的 acc 余量远小于种子间 Δacc 散布 **0.0201**(0.0402 − 0.0201,≈15× 余量)——即 acc pass 落在 head-seed 噪声内。

**分解(final-epoch 均值,同 runner):** frozen-Qwen 编码器交换 **−0.0112**(B1 第 20 条负结果,FAIL)vs LoRA-Qwen **+0.0313** ⇒ **ZH 的增益全部来自 LoRA 的任务/语言适配,而非 MLLM-encoder 身份本身**(与 B1 一致)。

**单编码器抽样警告(scope):** B3 是 **head-seed** 配对测试,仅一次 LoRA-SFT 编码器抽样(3 种子共享单一特征缓存,只变下游 head)。它**不**建立 LoRA-SFT 训练种子方差;那需 ≥3 次全新 LoRA-SFT 重训 + 重抽取(B3 范围外,预注册声明)。

**对选项账目的影响:** 本补遗把 §4(b) 从**"未验证"**转为**"已实测,等待用户的 novelty 裁决 + `PAPER_MASTER_TABLES.md:58` 的'不可直接同格并比'覆盖决定"**。B3 只判 goal 的**性能子句**(+0.03 acc AND +0.03 F1);是否算 novelty、"MLLM-encoder family"(HateMM frozen-swap + ZH LoRA,两种不同机制)是否算"双数据集"headline、以及 B3 的同 runner 同种子配对是否覆盖 PMT:58 的记账注,全部仍是用户裁决。这是 round-2 的**首个实测(部分)正结果**;其余全部搜索轴仍关闭(21 条负结果)。

## 7. B4 补遗(2026-07-14):EN-LoRA 单元预-GPU 关闭(第 22 条)

B3 把 ZH 的增益归给 LoRA 适配后,剩下的唯一非同构表征级候选是 **EN 侧的同一 LoRA 单元**(LoRA-Qwen 编码器 on MHC-EN,3-种子配对 vs frozen-CLIP)。取证侦察(`refine-logs/B4_FORENSIC_RECON.md`,零 GPU/零提交)证明**此单元并非未验证**:同一 adapter(`logging/lora/MHC`)+ 同一特征缓存 + 同一 RGCL+kNN head 已在 **seed0 双协议**测过并入账为负结果(`exp-lora-sft-encoder.md:21`,verdict partial,2026-07-02;主日志 `rgcl_MHC_...LoRA_2723309.trainlog:250/:275`)。seed0 配对 vs enc3s EN CLIP 控制(12850):**val-selected −0.0310 acc / −0.0197 F1(回归)**,final-epoch **+0.0062 acc / +0.0157 F1(约为 +0.030 门的 1/5)**;EN 上 LoRA **低于两个 frozen 编码器**(CLIP 与 frozen-Qwen)。P9 的决策级 EN LoRA 也独立失败(−2.7 vs floor,`EXP_p9:212`),且 P9 本身就把此单元当作"已关闭"来跳过 EN(`:62-63`)。诚实先验:**双协议 FAIL,证伪概率 <5%**。

**LoRA 族三数据集地图现已由证据补全:HateMM(P9 fail)/ EN(banked fail)/ ZH(B3 marginal-pass)⇒ LoRA 是 ZH 特定的语言适配杠杆,不是普适的 MLLM 集成机制**(ZH 上 LoRA 缓解了 English-CLIP 文本塔处理中文的记录在案的劣势;EN 上 549 样本的 LoRA-SFT 反而退化编码器——这正是 ZH/EN 符号翻转的既定解释,不是谜团)。

**判决(orchestrator,level=decision):** 依"不在已关闭轴上烧 GPU"规则,B4 作为**第 22 条预注册负结果类条目预-GPU 关闭,零 GPU 成本**。

**用户选项(veto-clean,按需可跑):** 因 adapter 与特征缓存均已在盘,把 seed0 锚定的入账负结果升级为正式 3-种子配对判决**仅需约 2 分钟 GPU**(缓存特征 → 每 run ~20-25 s;`scripts/slurm/enc3seed.sbatch` 加三行 `"MHC Qwen2.5-VL-7B-Instruct-LoRA_HF {0,1,2}"`)。此单元清过全部三条现行 veto(单数据集自有 train split / 无 OCR / 无 gold aux),可作论文表格的一行**正式闭合**——但只会形式化一个已知负结果,不开新地。**现在不跑;留作用户请求项。**
