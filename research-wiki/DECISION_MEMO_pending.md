# DECISION MEMO — 悬置用户决策合并单(一页清空版)

> **性质声明(读前必看)。** 本备忘录**只汇总当前所有悬置的用户决策**,**不含任何已批准行动**;
> 每项的「我方推荐」仅是建议,**一切裁决以用户答复为准**。全部理由引用**已 commit** 的证据(文档 + commit)。
> 汇总自:`TERMINUS_mllm_campaign_DRAFT.md`(FINAL)、`OPTION_KITS_terminus.md`、`MORNING_REPORT.md §7/§9`、
> `experiments/exp-archive-knn-seeds.md`、`experiments/exp-consensus-zh-seeds.md`、`PAPER_MASTER_TABLES.md`。
> 生成:2026-07-09 · 基线 HEAD `ea83142`。**六项一次拍完即全部清空。**

---

## D1 — MLLM campaign 战略终局(TERMINUS 三选项主裁决)

- **问题:** 13 条预注册路线全结题后,论文的 MLLM 故事走 **(a) 定稿 / (b) 闭源 API 续攻定位(需数据外发批准)/ (c) 换方法族**?
- **选项与代价:** (a) 近零算力、放弃「substantial 主表增益」强 claim;(b) 中成本、须批准把仇恨内容送第三方商业 API 且接受闭源不可复现;(c) 周期以周计、放弃现有四支柱资产。
- **我方推荐:(a) 定稿**(若用户接受数据外发 + 闭源代价,可 **(a)+(b) 并行**,(b) 仅作定位放大的可选增量,不阻塞定稿)。
- **理由:** 13 路线全为诚实 kill / within-noise;开源可行域**三面墙全闭**——重聚合 0.5932 / 规模梯 72B 0.5913 / 代际 Qwen3-VL-32B 0.5866,**均 < 0.616 校准线**→ substantial 定位开源不可达(`TERMINUS §2/§3/§6`,commit `03880f2`/`74f0eac`/`0b3cf40`)。(b) 的两点校准→test 映射(0.5387→0.5435、0.5913→0.5755,transfer 仅 ~60%)诚实外推**闭源翻案期望低**且仅动定位、非主表(`OPTION_KITS Kit-B B.2/B.5`);(c) 上限高但 P9/P9b 已警示决策级 LMM 只匹配现有 LoRA(`EXP_p9_lmm_rgcl_video`,`4d28655`)。
- **影响文档/章节:** 方法章 MLLM 一节整体定位;`TERMINUS §5`;`OPTION_KITS Kit-A/B/C`;`PAPER_MASTER_TABLES` T2.1/T2.2。

## D2 — Headline 协议(ZH 主表口径)

- **问题:** ZH 主表用 **val-选点(0.827,预注册,不过 0.85)** 还是 **final-epoch(0.8537±0.012,过 0.85)**?
- **选项与代价:** 改用 final-epoch 过线好看,但「**因过线才换口径 = rule-shopping**,rebuttal 必死」;且 final-epoch 下 archive 通道 ZH 贡献**恰为零**、EN 为负。
- **我方推荐:两口径并排** —— 主表沿用预注册 val-选点,附录放五规则鲁棒性全表 + 已成文说明段;若改 final-epoch 须以「未来预注册」名义全线统一并自曝时序。
- **理由:** 同一 seed 权重逐位相同、val-选点在 78 样本 dev 上自损 ~2 acc 点(`exp-archive-knn-seeds` Addendum 2,sha1 审计);selection-robustness 段落已成文可直接引用(`MORNING_REPORT §6.1/§7.1`)。
- **影响文档/章节:** `MORNING_REPORT §1` 记分板;主表 T1;方法学附录 selection-robustness 段。

## D3 — EN 近天花板定位 + 摘要措辞

- **问题:** MHC-EN ≈0.79–0.80(所有杠杆穷尽),摘要用「**同场决定性胜出 + 近天花板归因**」如实报数,还是「near-SOTA-at-fraction-of-params」的强措辞?
- **选项与代价:** 强措辞更抢眼但 CRAVE 发表 79.81 F1 在**全量 split**、与我方 clean 子集**不可直接比**,直报 near-SOTA 会被审稿人抓可比性。
- **我方推荐:直报双口径 EN 数 + 同场胜出框架 + 近天花板归因**(不主张绝对 SOTA)。
- **理由:** 同场 MoRE clean 仅 0.69–0.72,我方 +8.7 acc / +22.9 F1(`BASELINE_MoRE_rerun`,`MORNING_REPORT §2`);EN 章主体 = §4③ 归因链 + oracle 复活条件(role-3 门控留 0.857–0.888,`EVAL_role3_selective_reasoning`);0.85 作为该 split 未达公开目标如实报(`HEADTOHEAD_FEASIBILITY §3`,`OPTION_KITS A.4 旧决策②`)。
- **影响文档/章节:** 摘要;`MORNING_REPORT §1/§2`;EN 章 §4③。

## D4 — 三件套 / 四支柱措辞(MLLM 角色定位)

- **问题:** 方法章 MLLM 一节按 campaign 终局的**「三个挣得的角色 + 一条明确的非角色」**重写,并把定位对照改写为 memory 对照?
- **选项与代价:** 不改则停在 campaign 前的视觉-only 记忆键口径(wv-AUC 0.526,仅能力演示),与终局的 P6→P10-b **统计稳固 modest-plus 定位器**(0.5755)不一致。
- **我方推荐:采用 Kit-A A.2 四角色框架** —— encoder(HateMM +4.2 F1 跨 0.85)+ 定位打分器(P6 0.5435 → P10-b 72B A-fuse **0.5755**,对 memory +0.0615 配对显著)+ guard-rail/审计 + **明确非角色**(11+2 路线 ruled-out map);**MORNING_REPORT §3/§4④/§5 三处定位表述同向更新**到 P6→P10-b。
- **理由:** 定位对照写 **memory 而非 MIL**——A-fuse−memory **+0.0996** 显著、A-fuse−MIL n.s.(`TERMINUS P11 §4`,`0b3cf40`);MIL 需目标域视频标签、memory-swap 不需,非同一能力口径。数字与框架见 `TERMINUS §4/§6.2`、`PAPER_MASTER_TABLES`(`d9731e8`)。
- **影响文档/章节:** 方法章 MLLM 节;`MORNING_REPORT §3/§4④/§5`(第 13 撤回行缩范围);第 5 章定位节。

## D5 — 杂项归档(入库 / 保留说明)

- **问题:** ①`disk_guard.log`(现 ~55 万行守护流水)入不入库?②`HateVideoVLM` conda 环境 与 ③`data/gt/HateClipSeg/p11_split.json`(冻结未消费)如何留档?
- **选项与代价:** raw 守护日志入库会污染仓库且非研究产物;环境/split 若无说明,未来会被误删或误当已消费资产。
- **我方推荐:** ①`disk_guard.log` **不入库**(gitignore / 仅留本地)——其 load-bearing 的 sha1 / B2 对账证据已在 `MORNING_REPORT §6.3` 与 `EXP_mm_segment_keys §1` 成文;②`HateVideoVLM` 环境**保留** + 一行溯源(P10-c Qwen3-VL 打分环境,复现 `EXP_p10_loc_amplify` P10-c 节,`74f0eac`);③`p11_split.json` **保持冻结** + 一行说明(为任一前提不同的 HateClipSeg 弱监督训练路线预留,test-touch 从未消费,`TERMINUS P11`,`0b3cf40`)。
- **影响文档/章节:** `.gitignore`;`ITERATION_LOG` 归档节;`EXP_p10_loc_amplify` / `EXP_p11_weaksup_localization` 资产说明。

## D6 — 投稿目标(venue + 截稿)

- **问题:** 目标 venue 与截稿定哪个(候选:WWW / ICWSM / ACL-ARR,尚未定)?
- **选项与代价:** 未定则无法按页数/附录政策裁剪正文-附录分配;素材形态(主表 + 四支柱 + 归因章 + 方法学附录 + 负结果 ruled-out map)本身 **venue-agnostic**,不阻塞其余五项。
- **我方推荐:尽快锁 venue+截稿以解冻 Kit-A A.3 的正文/附录分配**;鉴于方法学章含大量强负结果 + ruled-out map,倾向**欢迎负结果/方法学贡献**的 venue(ICWSM / ACL-ARR 线)。
- **理由:** Kit-A A.3 已给出正文/附录分配草案(同场 MoRE 主表 + 四支柱进正文,11 路线记分板 + 双口径鲁棒性进附录),只待 venue 页数政策定稿(`OPTION_KITS A.3`,`MORNING_REPORT §7.3`)。
- **影响文档/章节:** 全文裁剪;`OPTION_KITS A.3` 表位分配。

---

*(本备忘录随 D1–D6 任一被裁决而逐项失效;裁决后由主会话把结论落地到对应文档,并从本单移除已清项。)*

---

## Round-2 追加裁决项(2026-07-14,B3/B4 之后)

> 本节承接 D1–D6,追加 **round-2 MLLM-integration campaign** 收尾后新增的悬置裁决项:B3(LoRA-Qwen
> 编码器 vs frozen-CLIP on MHC-ZH,job 13150,`final-epoch: PASS (MARGINAL); val-selected: FAIL`)是本轮
> **首个实测(部分)正结果**,B4(EN 侧同一 LoRA 单元)是**第 22 条预注册负结果类条目预-GPU 关闭**。数字与
> 判决语言逐字转录自命名源:`research-wiki/TERMINUS_round2_mllm_plus3.md`(§4/§6/§7)、
> `refine-logs/B3_VERDICT_REVIEW.md`(§4b/§6)、`refine-logs/B4_FORENSIC_RECON.md`、
> `research-wiki/PAPER_MASTER_TABLES.md`(PUR addendum:PUR-1/PUR-2/PUR-banner)。**同样只汇总悬置决策,
> 不含已批准行动;每项「我方推荐」仅是建议。** 编号续 D1–D6。

## D7 — LoRA / RA-HMD-family 编码器杠杆是否计入 goal 的「novel」子句

- **问题:** B3 把 LoRA-Qwen 编码器 vs frozen-CLIP 在 MHC-ZH 定为 `final-epoch: PASS (MARGINAL)`(mean
  Δacc **+0.0313**)——这是全项目**最接近** goal「+3 acc AND +3 F1」的实测配对结果(其余 21 条搜索轴全为
  负)。但 LoRA / RA-HMD-family 一直被本项目分类为**「MIXED performance lever, not novelty」**
  (`query_pack.md:44`;`B1_PREREG_REVIEW.md:64`)。**一个 LoRA-encoder 的性能 pass 是否计入 goal 的
  「novel」子句?**
- **我方推荐:不主张 novelty。** 把 B3 作为 encoder-adaptation 的**正式消融/性能行**入表,novelty 叙事仍
  挂四支柱(retrieval-contrastive+kNN 核 + 可更新记忆 + 共识去噪 + 可审计档案);LoRA 只提供 ZH 上表征级
  增益的证据。若用户愿按 `TERMINUS §6(b)` 重划边界(把 LoRA 正式化为 encoder 适配消融而非 novelty 主张),
  可据此升级表述——**须用户先划 novelty 边界才可动。**
- **来源:** `TERMINUS §6`(选项 b)· `B3_VERDICT_REVIEW.md §6`(Novelty bullet,显式 PENDING)·
  `PAPER_MASTER_TABLES.md` PUR-banner (i)·性能数 = PUR-1。

## D8 — 「MLLM-encoder family」能否作 ≥2-数据集 headline

- **问题:** 两个编码器级 pass 骑在**不同机制**上——HateMM 是 **frozen-Qwen** swap **双协议 PASS**,ZH 是
  **LoRA-Qwen** 微调 **final-epoch marginal PASS**。可否以「MLLM-encoder family」作为跨 ≥2 数据集的
  headline?**若 goal 要求单一机制跨 ≥2 库过线,则 frozen(仅 HateMM)与 LoRA(仅 ZH)均不单独满足。**
- **我方推荐:不打「单一机制双库」headline。** 如实报「MLLM-encoder family」= 两条不同杠杆各在一库,并在
  正文明确二者机制不同(frozen-swap vs LoRA-adaptation);headline 主张仍走 HateMM 单库最强正效应
  (+5.3–5.6 acc 双协议 3/3)+ 同场 MoRE 胜出。
- **来源:** `B3_VERDICT_REVIEW.md §4b`(No headline upgrade)+ §6(第二 bullet)· `TERMINUS §6`·
  `PAPER_MASTER_TABLES.md` PUR-banner (ii)。

## D9 — `PAPER_MASTER_TABLES.md:58`「不可直接同格并比」注是否被 B3 配对覆盖

- **问题:** 主表脚注(`PAPER_MASTER_TABLES.md:58`)记「LoRA-Qwen 主栈与 frozen-CLIP floor 不同编码器,
  **不可直接同格并比**」。B3 用**同 runner、同 `--seed`、同 149 ZH test videos** 的 head-level 配对
  (job 13150 vs 13115),是现存最干净的配对读数并已记于 PUR-1。**这一同 runner 同种子配对是否覆盖 :58
  记账注,以支撑一个论文主张?**
- **我方推荐:覆盖仅限「配对 Δ」这一受控读数**(可在附录以 B3 同 runner 配对表 + 三条敏感度事实呈现),
  **不**升级为主表 headline 并比;主表 T1.1 保留 :58 注。理由:B3 仍是**单一 CLIP 抽样 + 单一 LoRA 编码器
  抽样**(3 种子共享单缓存,只变下游 head),不建立训练种子方差。
- **来源:** `PAPER_MASTER_TABLES.md:58` + PUR-1/PUR-banner (iii)· `B3_VERDICT_REVIEW.md §6`(第三 bullet)·
  `TERMINUS §6`。

## D10 — 可选:EN-LoRA 正式闭合跑(约 2 分钟 GPU,veto-clean)

- **问题:** B4 取证证明 EN 侧同一 LoRA 单元**并非未测**:同 adapter + 同特征缓存 + 同 RGCL+kNN head 已在
  **seed0 双协议**入账为负结果(val-sel **−0.0310 acc** / final-ep +0.0062 acc,`exp-lora-sft-encoder.md:21`)。
  因 adapter 与缓存均在盘,把 seed0 锚定负结果**升级为正式 3-种子配对闭合行仅需约 2 分钟 GPU**,且清过全部
  三条现行 veto(单数据集自有 train split / 无 OCR / 无 gold aux)。**跑不跑?**
- **我方推荐:现在不跑**(遵「不在已关闭轴上烧 GPU」)。它只会把**已知负结果**形式化为论文的一行正式闭合,
  不开新地;诚实先验 = **双协议 FAIL,证伪概率 <5%**。留作**用户请求项**——若用户要一行形式化的 3-种子闭合行
  以补全 LoRA-encoder 三数据集地图(PUR-2),可按 `enc3seed.sbatch` 加三行运行。
- **来源:** `B4_FORENSIC_RECON.md §(iii)/(v)`(成本表 + 诚实先验)· `TERMINUS §7`(用户选项,veto-clean)·
  数字 = PUR-2 MHC-EN 行。

## D11 — 72B-AWQ scale 点(为完整性列出)

- **问题:** 72B-AWQ 编码器是唯一未跑的 scale 点。是否值得跑?
- **我方推荐:不建议(dead-axis grinding)。** B2 已实测 32B 在 HateMM 锚数据集上**介于 CLIP 与 7B 之间**
  (scale **单调退步**,CLIP<32B<7B),对 72B 的先验 ≈0;真要跑需抽取脚本加 AWQ 路径 + autoawq 安装 +
  delta-check + 41G 下载。除非用户另有机制假设,否则不动。
- **来源:** `TERMINUS §4(a)` · scale 单调退步实测 = `B2_VERDICT_REVIEW.md`(第 21 条负结果,`TERMINUS §1`)。

## D12 — 基建裁决(备份 / 配额 / 遗留 flag)

- **问题:** 四项运维待裁:① main 上 **125 个未推送 commit**(实核 `git rev-list --count origin/main..HEAD`
  = 125,备份风险);② **disk_guard quota 解析 bug** —— 当前**设计上全盲**(不自动裁剪),修复可能在**超阈值时
  重新启用自动 pruning**,风险不对称,须用户明确放行才动;③ **lora_p9 83G + Retrieval 41G 未备份**未裁决;
  ④ A 线 **M-A/realbank `is_science` 遗留 flag**。
- **我方推荐:** ① 尽快 push 或另做冷备(纯文档 commit,无 GPU);② disk_guard 修复**须用户明确 go**(在此
  之前维持全盲,避免超阈值自动删);③④ 留待用户逐项裁决,不擅动。
- **来源:** `TERMINUS §4(d)`。

---

*(D7–D12 与 D1–D6 同规:任一被裁决即逐项失效,裁决后由主会话落地并从本单移除已清项。)*
