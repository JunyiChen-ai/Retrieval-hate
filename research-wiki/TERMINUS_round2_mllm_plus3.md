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
(b) **LoRA 族杠杆**(ZH LoRA-Qwen 0.8537 vs CLIP 0.8027 的未配对差距):按你的框架 LoRA-SFT 属非 novelty 杠杆且 P9 家族已关;若你愿意重新定位其角色(如作为 encoder 适配的正式消融而非 novelty 主张),这是数据上最接近 +3 的未验证配对。**需要你重新划 novelty 边界才可动。**
(c) **goal 重议**:以 HateMM-only encoder +5 作为"MLLM meaningful integration"的论文主张,把 21 条负结果做成 negative-results/rigor 贡献(现有 TERMINUS/OPTION_KITS + 本文档已备料)。
(d) 基建裁决:~120 个未推送 commit;disk_guard quota 解析 bug(当前全盲,建议修);lora_p9 83G / Retrieval 41G 未备份未裁决;A 线 M-A/realbank is_science 遗留。

## 5. 协议状态

dali_autoresearch 转入监控模式(心跳照常,不再对已关闭轴烧 GPU);全部状态在 autoresearch/goal_mllm_plus3/state/;本轮资源账:GPU ≈ 5.5h 实验 + 0 浪费在被 gate 拦截的路线上,单提纪律零违规,1 次探针门控基础设施重试(上游 CDN 瞬断)。
