# PAPER_MASTER_TABLES — 论文级主表汇编(纯转录,不含新实验)

> **汇编日期:2026-07-09。** 本文件是全项目**已结题、已 commit** 结果的**纯汇编**:所有数字均为
> 已 commit wiki 文档的**转录**,每表带「来源:文档名 + commit」列或脚注。**本文档不跑任何实验、
> 不提交 SLURM、不重算任何数、不引入任何新测量。** 它是三个终局选项(定稿 (a) / 闭源续攻 (b) /
> 换方法族 (c))的公共底座。若个别数字在不同文档间互相矛盾,以**更晚 commit** 的为准并加脚注;所有
> 已知张力汇总在文末「数字矛盾/张力清单」。
>
> **协议口径(全局)。** 除非注明,分类数字为预注册口径:warmup≥5、val-selected(max
> `Val_Retrieval acc`,roc tie-break);另并排 final-epoch(selection-free,epoch 29)。测试集为
> ~150 量级:MHClip test EN n=161 / ZH n=149;HateMM clean n=215(P9 匹配口径另注)。噪声地板约定:
> 1 acc 点 ≈ 1.6 视频;sub-1pt 效应记为 within-noise。**全程无 cross-seed ensemble。**
>
> **源文档与 commit(document-level):** BASELINE_MoRE_rerun.md `ebc1988` · MORNING_REPORT.md
> `78ab700` · exp-archive-knn-seeds.md `ebc1988` · exp-consensus-zh-seeds.md `ebc1988` ·
> exp-cross-dataset-transfer.md `becfd91` · exp-baseline-reproduction.md `becfd91` ·
> EVAL_temporal_memory_W4.md `8a746ad` · EXP_auto_memory_repair.md `d4e58aa` ·
> EVAL_localization_hateclipseg.md `ebc1988` · EXP_p6_mllm_localization.md `c9e3bd8` ·
> EXP_p9_lmm_rgcl_video.md `4d28655` · EXP_p10_loc_amplify.md `74f0eac` ·
> CAMPAIGN_mllm_method_role.md `78ab700` · TERMINUS_mllm_campaign_DRAFT.md `2781349` ·
> OPTION_KITS_terminus.md `a6a7a59`。表内「commit」列引各前沿结果的**结果级 commit**(最精确出处)。

---

## T1 — 主表(分类)

四数据集 × 关键配置,两协议并排(val-selected / final-epoch),mean±std 与 seed 数。MoRE 同场重跑
对比在 T1.2。**读法:** EN/ZH 的「当前最优栈」在多 seed 审计下与 floor 不可分(见脚注与 T4/机制结论);
HateMM/ImpliHateVid 早已达标 acc≥0.85。

### T1.1 我方各配置(clean test 子集)

| 数据集 (n) | 配置 | 编码器 | val-sel acc | val-sel macro-F1 | final-ep acc | final-ep macro-F1 | seeds | 来源 · commit |
|---|---|---|---|---|---|---|---|---|
| **HateMM** (215) | frozen-CLIP RGCL floor | CLIP ViT-L/14-336 | 0.8279 | 0.8172 | — | — | 1 | `1035814.trainlog:257-259`(val-sel ep24) · exp-baseline-reproduction |
| **HateMM** (215) | **frozen-Qwen RGCL(最优栈)** | Qwen2.5-VL-7B(冻结) | **0.870** | **0.861** | — | — | 1 | exp-baseline-reproduction / MoRE §3.2 · `ebc1988` |
| **HateMM** (P9 匹配) | trained-RGCL floor(P9 口径) | frozen-Qwen | 0.870 | — | 0.8605 | — | 3 | EXP_p9 · `4d28655` |
| **HateMM** (P9 匹配) | raw-kNN floor(P9 口径) | frozen-Qwen | — | — | 0.786 | — | 3 | EXP_p9 · `4d28655` |
| **ImpliHateVid** | frozen-CLIP floor | CLIP | 0.910 | — | — | — | 1 | exp-baseline-reproduction · `becfd91` |
| **ImpliHateVid** | frozen-Qwen floor | frozen-Qwen | 0.900 (~0.91) | — | — | — | 1 | exp-baseline-reproduction / MORNING §1 · `78ab700` |
| **MHC-EN** (161) | frozen-Qwen floor(无键) | frozen-Qwen | 0.7702 ± 0.0221 | 0.7010 ± 0.0448 | 0.7888 ± 0.0152 | 0.7488 ± 0.0208 | 4 | exp-archive-knn-seeds Add.3 · `ebc1988` |
| **MHC-EN** (161) | + archive-kNN α0.25(最优栈) | frozen-Qwen | 0.7935 ± 0.0205 | 0.7497 ± 0.0250 | 0.7826 ± 0.0134 | 0.7430 ± 0.0196 | 4 | exp-archive-knn-seeds Add.3 · `ebc1988` |
| **MHC-ZH** (149) | LoRA-only floor(无键) | Qwen2.5-VL-7B-LoRA | 0.8282 ± 0.0139 | 0.7962 ± 0.0167 | **0.8537 ± 0.0120** | 0.8259 ± 0.0124 | 5 | exp-archive-knn-seeds Add.2 · `ebc1988` |
| **MHC-ZH** (149) | + archive-kNN α0.25(最优栈) | Qwen2.5-VL-7B-LoRA | 0.8268 ± 0.0266 | 0.7915 ± 0.0397 | **0.8537 ± 0.0120** | 0.8259 ± 0.0124 | 5 | exp-archive-knn-seeds · `ebc1988` |
| **MHC-ZH** (149) | + consensus 去噪(机制/robustness)† | frozen-CLIP | 0.8107 ± 0.0347 | 0.7764 ± 0.0406 | 0.8175 ± 0.0129 | 0.7841 ± 0.0204 | 5 | exp-consensus-zh-seeds · `ebc1988` |
| **MHC-ZH** (149) | consensus 对应 λ=0 floor† | frozen-CLIP | 0.8027 ± 0.0139 | 0.7649 ± 0.0151 | 0.8027 ± 0.0215 | 0.7594 ± 0.0240 | 5 | exp-consensus-zh-seeds · `ebc1988` |

**关键脚注(载重):**
- **ZH final-epoch = 唯一跨 0.85 的口径**(0.8537±0.0120,seeds 3/4 达 0.8658);val-selected 两臂均
  ~0.827 不过线。**archive-kNN 键在 final-epoch 对 ZH 贡献恰好 0**:同 seed ckpt sha1 字节相同,
  α=0.25 键在 ep29 翻转 0 票(exp-archive-knn-seeds Add.2 权重身份审计),故 archive 臂与 floor 臂
  final-epoch 逐位相同。**「archive-kNN 带来 accuracy」主张已多 seed 撤回**(配对 dAcc −0.0014±0.0313)。
- **EN 镜像同一教训**:val-sel 下 archive 看似 +2.3pt(0.7935 vs 0.7702),但大半来自 floor 一个 seed
  的病态选点(s3 选 epoch6→0.7391);final-epoch 下 Δ=−0.0062±0.0051(0/4 seed 正)。**EN 故事 =
  「≈0.78–0.80,任何键增强都不分离」**,非排序主张。
- **val-selection 税**:78 样本 ZH dev 上 val-acc 选点相对 selection-free 自损 ~2 acc 点(两臂
  val-sel ~0.827 vs final 0.8537)。
- **† consensus 是 CLIP-base 独立子实验**(不同编码器,与 archive-kNN 的 LoRA-Qwen 主栈不可直接
  同格并比);列于此仅为完整。作为 ZH 主表**机制/robustness 行**,非 headline accuracy 行。
- ImpliHateVid / HateMM 早达标后未再动;seed 数为 baseline-reproduction 单次口径,无 MoRE 同场轨道
  (MoRE 复跑不含 ImpliHateVid)。

### T1.2 同场 MoRE 对比(clean test 子集;ours vs MoRE rerun vs MoRE reported)

MoRE(WWW 2025)官方代码全量复跑,**同 split(逐行 diff 一致)、同 clean test、双 variant
(as-released / bugfix)+ 5-seed 敏感性**;缺失件(caption/tsv/OCR)本地复原并脚注。

| 数据集 (clean n) | 我方最优配置(MoRE 同场点估)‡ | MoRE as-released | MoRE bugfix | MoRE 5-seed 均值(as-rel) | MoRE reported(full data) | **Δ(我方 − MoRE 较优 variant)** |
|---|---|---|---|---|---|---|
| **HateMM** (215) | frozen-Qwen **0.870 / 0.861** | 0.8140 / 0.7988 | 0.8047 / 0.7899 | 0.792±0.035 / 0.781±0.038 | 0.8341 / 0.8235 | **+5.6 acc / +6.2 F1** |
| **MHC-EN** (161) | frozen-Qwen **0.7888 / 0.7378** | 0.6894 / 0.4438 | 0.7019 / 0.5084 | 0.722±0.031 / 0.530±0.111 | 0.7750 / 0.7519 | **+8.7 acc / +22.9 F1** |
| **MHC-ZH** (149) | LoRA-SFT **0.8322 / 0.8023** | 0.7651 / 0.6882 | 0.7584 / 0.7058 | 0.717±0.035 / 0.661±0.023 | — / 0.7475 | **+6.7 acc / +9.7 F1** |

来源:BASELINE_MoRE_rerun §3.1/§3.2 · `ebc1988`;MORNING_REPORT §2 · `78ab700`。

**MoRE 对比脚注:**
- **三库全部同场胜出**,取 MoRE seed 均值上界或较优 variant 均不翻转;名义训练标签量还是 MoRE 略占优
  (EN 618 / ZH 633 vs 我方 clean 550 / 579)。
- **sanity:** HateMM(唯一数据完备库)复跑落发表值 −2~3pt(单 seed 方差内)= 复现成功。MHClip 两库
  低于发表值,主因**数据缺失**(EN 标签 890/视频 792、ZH 897/814;发表基于全量 1000);EN reported→
  rerun 塌 −23pt F1 因 val 早停(91 样本 val,epoch5 定格,F1 尚在爬升即被掐)。发表值仅列 "reported
  (full data)",**不与 clean 子集数字直接比**。
- **caption 为我方复原**(Qwen2.5-VL-7B,原文未文档化生成方式);**ZH OCR 为 easyocr 替换**(paddle GPU
  cudnn 不可装 / CPU SIGILL);均只进检索记忆库。释出代码 7 项缺陷(einops 漏列、merge audio 循环 bug、
  形状不配、config 键错位、O(n²) 写盘等)均文档化处置,处置全程留痕。
- **‡ 张力提示:** T1.2 的「我方最优点估」引 ITERATION_LOG 的 warmup-consistent **val-selected 单配置**
  (EN=frozen-Qwen、ZH=LoRA-SFT 最优);与 T1.1 的多 seed 审计口径不同源。ZH 的 **0.8322 是单 seed 点**
  (已在 MORNING §5 kill #6 标注),多 seed 现实为 val-sel 0.8268±0.0266 / final 0.8537±0.0120;EN 的
  0.7888/0.7378 亦为单配置 val-sel(多 seed 见 T1.1)。**论文主表建议以 T1.1 多 seed 为 headline,
  MoRE 对比 Δ 附单配置点估脚注**,以免被审稿人以「seed cherry-pick」质疑。详见文末矛盾清单 #1。

---

## T2 — 定位表

### T2.1 HateClipSeg 定位链(within-video mean-AUC 主指标;395 视频,wv 在 329 两类俱全视频上)

| 配置 | within-video AUC | 95% CI | 显著性 | paired vs memory | paired vs P6-7B | 来源 · commit |
|---|---|---|---|---|---|---|
| random(seed 0) | 0.5088 | — | — | — | — | EVAL_localization_hateclipseg / EXP_p6 · `c9e3bd8` |
| memory `knn_hatemm_subclip` K=30 | 0.5140 | [0.4955, 0.5323] | sign-p 0.11 (n.s.) | — | — | EVAL_localization_hateclipseg · `ebc1988` |
| *(memory 最强 cell:K=4 subclip)* | *0.5259* | *[0.5048, 0.5468]* | *sign-p 0.0066(唯一显著 cell)* | — | — | EVAL_localization_hateclipseg · `ebc1988` |
| **P6 — MLLM 逐窗打分器(7B)** | **0.5435** | [0.5330, 0.5544] | sign-p 5.4e-8 | Δ+0.0296 CI[+.0088,+.0504] p=0.0071 | — | EXP_p6 · `c9e3bd8` |
| **P10-b — 72B A-fuse(promoted)** | **0.5755** | [0.5581, 0.5933] | sign-p 1.4e-9 (n=329) | Δ+0.0615 CI[+.0359,+.0869] p=4.9e-5 | Δ+0.0319 CI[+.0170,+.0474] p=0.0024 | EXP_p10(P10-b test) · `74f0eac` |

**判定:** 三档 bar(≥0.60 substantial / 0.56–0.60 且 CI 排除 P6 0.5435 = modest / <0.56 = P6 站住)。
P10-b **0.5755 ∈ [0.56, 0.60) 且 CI 下界 0.5581 > 0.5435 → MODEST amplification**(未达 substantial)。
定位角色由 modest(7B)升为 modest-plus(72B A-fuse),earned-roles 性质不变、程度加强。**唯一一次
HateClipSeg test 触碰已花掉。** 诚实警告:MLLM 主导能力仍是 video-level density(broadcast AP 0.62),
within-window 是更小但统计稳固、随 scorer 规模单调增长的增量。

### T2.2 HateMM 校准 leaderboard(三轮 14 比较 vs 冻结 7B anchor 0.5387;n=266 both-class)

bar(round-1/2):paired Δ ≥ +0.04 且 CI 排除 0(等价 wv-AUC ≥ 0.5787);round-3 收紧为 wv-AUC ≥ 0.616。

| round | variant | scorer | HateMM wv-AUC | paired Δ vs anchor | paired Δ 95% CI | 过 bar |
|---|---|---|---|---|---|---|
| — | anchor(raw K30) | Qwen2.5-VL-7B | 0.5387 | — | CI[0.5244,0.5534] · sign-p 5.6e-11 | — |
| 1 | A-gate | 7B | 0.5314 | −0.0074 | [−0.0195, +0.0045] | no |
| 1 | K60 | 7B | 0.5319 | −0.0068 | [−0.0156, +0.0019] | no |
| 1 | fewshot | 7B | 0.5359 | −0.0028 | [−0.0090, +0.0034] | no |
| 1 | A-lex | 7B | 0.5450 | +0.0062 | [−0.0000, +0.0123] | no |
| 1 | **A-fuse (K4×K30)** | 7B | 0.5693 | +0.0305 | [+0.0175, +0.0437] | no (Δ<+0.04) |
| 2 | R2-5 · A-fuse×A-lex (CPU) | 7B | 0.5752 | +0.0365 | [+0.0223, +0.0506] | no |
| 2 | R2-1 · anchor-agg | 32B | 0.5512 | +0.0125 | [−0.0006, +0.0257] | no |
| 2 | R2-2 · A-fuse | 32B | 0.5825 | +0.0437 | [+0.0240, +0.0631] | **yes** |
| 2 | R2-3 · anchor-agg | 72B | 0.5593 | +0.0206 | [+0.0065, +0.0347] | no |
| 2 | **R2-4 · A-fuse(晋级,test 0.5755)** | 72B | **0.5913** | **+0.0526** | **[+0.0333, +0.0721]** | **yes — 最高 Δ,晋级** |
| — | *(EXPLORATORY 天花板)72B fuse×lex* | 7B/32B/72B 重聚合 | *0.5932* | *+0.0544 [+0.0348,+0.0742]* | 重聚合封顶 < 0.616 | *n/a* |
| 3 | C2a · anchor-agg | Qwen3-VL-30B-A3B | 0.5469 | +0.0082 | [−0.0058, +0.0222] | no |
| 3 | C1a · anchor-agg | Qwen3-VL-32B | 0.5594 | +0.0207 | [+0.0077, +0.0339] | no |
| 3 | C2b · A-fuse | Qwen3-VL-30B-A3B | 0.5821 | +0.0433 | [+0.0227, +0.0644] | no |
| 3 | **C1b · A-fuse(round-3 best)** | Qwen3-VL-32B | 0.5866 | +0.0479 | [+0.0287, +0.0677] | no (<0.616) |

来源:EXP_p10_loc_amplify(P10 round-1 `7194ee2`、P10-b `03880f2`、EXPLORATORY `93e82fa`、P10-c
`74f0eac`)· 汇总 `74f0eac`。

**两条干净梯度 + 代际定论:**
- **raw scorer 规模单独走不过线:** anchor-agg 单调 7B 0.5387 → 32B 0.5512 → 72B 0.5593,但 72B 的 Δ
  (+0.0206)仍只有 gate 的一半。
- **A-fuse × 规模是唯一杠杆:** coarse×fine 融合增益随 scorer 增长 7B +0.0305 → 32B +0.0437 → 72B
  +0.0526;A-fuse 显著性在 **5 个 scorer**(7B/32B/72B/Qwen3-32B/Qwen3-30B-A3B)复现。
- **换代 ≠ 换规模(P10-c):** Qwen3-VL-32B A-fuse 0.5866 落在两代前 Qwen2.5-VL-32B 0.5825 噪声内,
  低于 72B 冠军 0.5913;30B-A3B(3B active)最弱 → 定位能力由**激活参数量**主导。
- **开源域三面墙全闭:** 重聚合(0.5932)/ 规模梯(72B 0.5913)/ 代际同档(Qwen3-32B 0.5866)均
  < 0.616 目标线 → substantial(校准≈0.616 → test≥0.60)在本集群开源域**不可达**。P10-b 0.5755 为
  最终定位数。

---

## T3 — 能力表(非精度贡献:更新性 / 可控性 / 定位能力)

| 能力 | 关键数字 | 机制 / 判定 | 来源 · commit |
|---|---|---|---|
| **跨数据集 memory swap** | 6 个 informative cross cell 中 **5/6 above-majority**;跨库落后 in-domain **~0.04–0.09 macro-F1**;**测试时换库零重训** | 换库=换一个配置项,trained-MoE 头**结构上不具备**此能力;headline novelty vs MoRE | exp-cross-dataset-transfer · `becfd91` |
| **temporal threshold recal(时间协议)** | EN temporal split 掉 −0.084 F1(0.7113→**0.6273**);**k=20 新期标注样本阈值再校准 → 0.7336**(≥random floor 0.7113,全额收复);oracle 天花板 0.7646 / acc 0.8199;ZH 无漂移(负对照) | 掉点主成分=**校准漂移非可分性损失**(temporal ROC 0.8484 > random-split 0.7175);检索架构把 operating point 暴露为一等 O(1) 可逆旋钮,trained-MoE 藏在权重里。原始「加样本进记忆」机制 flat-to-negative(k=20 memory-aug 只 0.6180) | EVAL_temporal_memory_W4 · `8a746ad` |
| **human-in-the-loop memory edit** | 删 2 条人工标记噪声记忆:EN test acc **0.8075 → 0.8199**(macro-F1 0.7626→0.7748),seed 0,**零重训**;超全部 5 随机 seed floor。**⚠ F88 多 seed 更正(载重):该正结果为 SINGLE-SEED** —— 精确多 seed replay 给出 seed 0 +0.0124 而 **seed 1/2/3 各 0 次翻转**,4-seed 均值 **+0.0031**;14-id 规则表更强(+0.0093 acc / +0.0089 mF1,3/4 seed,6 修 **0 坏**)但仍 3× 低于门、落 ±0.014 带内、已 test-consumed | 语义寻址 + 外科删除,纯 CPU 秒级;**口径固定:human-in-the-loop capability demonstration, single-seed; not an accuracy claim**(不再写「EN 全项目最高单点」) | DEMO_memory_editing / EXP_auto_memory_repair(复现门 PASS)· `d4e58aa`;更正 ERRPAT_MHC-EN_2026-07-26.md §6.5 · `ad56a62`(见 T6.5) |
| **guard-rail(auto-repair 语义否决)** | 两票 AND 规则 **C−A = +0.0000**(0/4 EN,不复现手工增益);但 **C−D = +0.47pt EN / +0.40pt ZH**(语义票否决 embedding-only 过删) | 语义票**否决** Cleanlab 式 embedding-only 对「真仇恨但 embedding-hard」记忆(虐待证词/性侵报道/含 slur)的过删,是 C>D 唯一来源;可审计(标签盲重找到人审 2 个噪声 id 且理由正确)。付费点=完整性/可控性,**非 raw acc** | EXP_auto_memory_repair · `d4e58aa` |
| **span-free 定位(记忆键口径)** | 最强 cell(HateMM 子片段记忆,K=4)full AP 0.545 / AUC 0.588,对 random +0.088/+0.100;within-video wv-AUC 0.526(仅 1/4 cell 显著) | 视觉-only 键对 speech-carried 仇恨盲;池化指标主体=毒性密度视频间排序(broadcast 追平)→ 只作能力演示。**MLLM 打分器口径见 T2**(挣得的可移除角色) | EVAL_localization_hateclipseg · `ebc1988` |
| **定位打分器(MLLM,挣得的可移除角色)** | 见 T2:P6-7B wv-AUC 0.5435 → P10-b 72B A-fuse 0.5755(对 memory/random/P6 均配对显著) | 唯一 scale 能移针的赛道;modest-plus,未达 substantial 0.60 | EXP_p6 / EXP_p10 · `c9e3bd8` / `74f0eac` |

---

## T4 — 反结果表(方法学章素材:MLLM 方法角色 campaign 全 13 路线,已结题)

判定口径:1 acc 点 ≈ 1.6 视频;sub-1pt = within-noise。每条均有复现 / bit-for-bit / probe 护栏背书
(非 harness 假象)。**行 7(P6)与 P10-b 是正例**,置于此表以完整呈现 campaign。

**计数口径(与 DRAFT_analysis_chapter.md §1 对齐,避免 11 与 13 混淆):** 本表 **13 行 = campaign
全部 13 条预注册路线**(route-family 粒度;CAMPAIGN_mllm_method_role.md「13 条预注册路线全部结题」)。
其中 **定位赛道 3 行**——行 7 **P6**(scorer,正例)、行 11 **P10/P10-b/P10-c**(amplify,P10-b MODEST
正例、P10-c 落线)、行 12 **P11**(weak-sup training,probe-fail);**其余 10 行为主表 accuracy 路线,
全数证伪**。DRAFT §1 把主表路线以更细粒度记为「**十一条**」(P9/P9b 分列计数),与本表 10 主表行**同指
一组结果、仅计数粒度不同**;两处终态一致 = **13 全结题,主表 accuracy 角色被证伪,定位赛道
= P6 + P10-b/P10-c/P11**。

| # | 路线(MLLM 方法职责) | 关键数字 | kill / 判定依据 | 文档 · commit |
|---|---|---|---|---|
| 0 | **auto-repair**:两票 AND 规则自动删噪记忆,复现手工 2-entry 增益 | C−A **+0.0000**(0/4 EN);手工删的 2 id embedding 反对率 0.50/0.60 < 0.80 阈值;C−D +0.47 EN/+0.40 ZH。**F88 更正:被复现的「手工 2-entry 增益」本身是 single-seed**(seed 0 +0.0124,seed 1/2/3 零翻转,4-seed 均值 +0.0031)⇒ 本行的 kill 判定不变,但「未复现的目标」量级须按 single-seed 读 | **FAIL**:AND 规则结构性删不到「语义矛盾但非 embedding-outlier」的记忆;幸存=guard-rail(见 T3) | EXP_auto_memory_repair · `d4e58aa`;更正 ERRPAT_MHC-EN_2026-07-26.md §6.5 · `ad56a62` |
| 1 | **P1** 零标注先验重校准:读档案→估先验 p̂→重设漂移门控阈值 | p̂ 误差 **0.22 EN / 0.18 ZH**(criterion ≤0.07);corrected recal 0.48 < static 0.63(EN);ZH forced −0.055 | **FAIL**:判据 FPR 在时间边界漂移(EN .372→.238),train 校正失真;机制成立(oracle 先验补回 EN 80% 缺口) | EXP_p1_zerolabel_recal · `2a69246` |
| 2 | **P2** 7B 邻居重排:按可比性删 INCOMPARABLE 邻居再投票 | B−A **−0.002 EN / −0.020 ZH**;过判 INCOMPARABLE **83% EN / 70% ZH**;selectivity lift +1.1% / −3.2% | **FAIL**:删除与投票正确性无关,过删稀释;ZH 净伤 4/5 seed | EXP_p2_neighbor_rerank · `bc689e1` |
| 3 | **P2b/P2c** 强判据 + train 端校准:7B/32B/72B × 证据 × prompt selectivity 榜 | 最佳 EN lift **+2.7pt**(bar +10);ZH 全 8 配置为负;drop-rate 随 scale 收敛 7B 72.5%→32B 64.6%→72B 30.9% | **FAIL(train 端即死)**:**comparability ⊥ vote-correctness at every open-source scale**;calibration 涨、selectivity 不涨 | EXP_p2b_stronger_judge · `cc4ca6e`,`aae1efe` |
| 4 | **P3** 证据密度池化(EN/ZH/HateMM):MLLM 段级 0–3 → softmax 重加权池化嵌入 | EN probe **−0.0055**;ZH val −0.0074/final +0.0088;HateMM 最干净 probe **+0.0108** 却训练 val −0.0041/final +0.0004(均 <1pt) | **FAIL(三库)**:**probe 必要非充分**——习得 align-fusion(img×text)头吸收输入端重加权;信号真实(段内 var 1.11/0.40 EN)但作定位资产(P6) | EXP_p3_evidence_pooling · `c2ba59f`,`15f5f08`,`22fe62a` |
| 5 | **P4** schema 字段蒸馏:辅助头蒸馏档案字段,λ=0.1,eval 丢弃 | probe **PASS**(字段可解码 AUC .62–.93、预测标签 .74–.78);train EN −0.001 / ZH +0.008(sub-threshold) | **within-noise**:字段真实但**与已直接监督的标签冗余** | EXP_p4_schema_distill · `6f1f0da`,`00816aa` |
| 6 | **P5** 反事实孪生负样本:MLLM 洗白转写 → 同视觉+洗白文本 hard-neg | 质量门 **CLOSED**:self-verdict flip **0.503 EN / 0.337 ZH**(≪0.80);诊断训练伤 EN −0.027;cfrand≈cf | **FAIL(前提不成立)**:MLLM 洗不干净;干净孪生**共享 anchor 视觉过近**(cos 0.73)反伤正样本簇 | EXP_p5_counterfactual_negs · `fc25cac`,`66d3103` |
| 7 | **P6** MLLM 定位打分器:逐窗证据分做无 span 时序定位 ✅ | wv-AUC **0.5435** vs memory 0.5140 / random 0.5088;配对 b>a Δ+0.0296 CI[+.009,+.050] **p=0.007**;对空 p=5.4e-8 | **PASS(唯一正例)**:可移除的定位角色,幅度 modest、统计稳固 | EXP_p6_mllm_localization · `c9e3bd8` |
| 8 | **P7** 分数级融合:视觉 kNN 票 × MLLM 语义通道,两条冻结规则 | corr(channel, vote share) **+0.21…+0.51(正相关)**;8 组 rule×channel net **−0.10…−0.38**;通道 AUC 0.54–0.69 < floor 0.81–0.86 | **FAIL(train 端 KILL)**:**证伪「decorrelated error channels」**——通道与决策变量冗余且是更弱分类器 | EXP_p7_score_fusion · `8f920e5` |
| 9 | **P8/P8b/P8c** 语义压缩 speech 通道:MLLM ≤60 词摘要作文本通道 | EN probe 开(B 0.7523>A 0.7359>朴素截断 C 0.7067)却训练 B −0.023/−0.079 劣于 C;ZH/HateMM probe 关;P8c 中文摘要 0.7168 最差 | **FAIL(全库)**:**campaign 最强 probe 却训练不过**(probe 必要非充分最尖锐);ZH 瓶颈=冻结 English-centric CLIP text tower 把中文 byte-fragment 97% 截断 | EXP_p8_semantic_compression / EXP_p8b_vision_summary · `e63d8fe`,`703f4fd` |
| 10 | **P9/P9b** 决策级 LMM-SFT:LoRA-SFT 整个 Qwen2.5-VL + 自带头(C3);P9b 加 rgcl-ON 臂(D3) | C3-mlp EN 0.7909(+0.6 noise)/ ZH 0.8635(**+1.0 vs 协议匹配 LoRA floor 0.8537**,noise);**C3-knn EN −2.7 / ZH −2.2 / HateMM −4.7 BELOW floor**;P9b D3-knn ZH 0.8389(−1.5)/ EN 0.7743(−1.0),0/12 cell 超 floor;D3−C3′ = head↔memory ±1.8pt 再分配 | **FAIL(最后架构 locus 关闭)**:LMM 头 displaces 而非 enhances memory;rgcl 项只再分配、非净增益。幸存=首次成功 port RA-HMD(released rgcl-OFF)stage-2 到 video(5 fork 修复) | EXP_p9_lmm_rgcl_video · `455e666`,`4d28655` |
| 11 | **P10/P10-b/P10-c** 放大定位角色:HateMM span 标定 scorer,单次测 HateClipSeg | round-1 A-fuse 7B **+0.0305 < +0.04**;P10-b 72B A-fuse 校准 0.5913 → **test 0.5755 MODEST**;P10-c Qwen3-VL-32B A-fuse **0.5866 < 0.616**(test 未触) | **round1/P10-c FAIL(no promotion);P10-b = MODEST amplification**(见 T2);把定位从 modest 升 modest-plus,不改主表终局 | EXP_p10_loc_amplify · `7194ee2`,`03880f2`,`74f0eac` |
| 12 | **P11** MLLM 段级密度作**弱监督训练信号**:蒸馏 72B A-fuse 段级密度 → 训练段级 head,比 (A) video-label MIL 与 (C) memory-kNN 弱标注器 | matched 同算子门 **A-fuse−MIL A-fuse +0.0359** CI[−0.0009,+0.0730] sign-p **0.13 n.s.**(差 0.0009);raw-vs-raw 两 K 亦 n.s.(+0.0058 K4 / +0.0143 K30);**教师优势 = coarse×fine 聚合技巧**非更好段级 labeller,5-fold 线性 MIL 已 ~0.55 wv-AUC(video labels 已含大部分教师信号);**memory 教师 0.4917 ≈ random** | **PROBE FAIL → 保守 kill**:committed letter gate(A-fuse−MIL K4 **+0.0386** CI excl 0)pass 但 granularity/operator-confounded,binding **matched** gate 不显著即 kill(反 bar-shopping,保守方向);**零训练成本**(仅一个 1h K30 特征提取,cache 可复用);**HateClipSeg test split 冻结未消费**;§3 成功线(B−A/B−C ≥ +0.05 & B abs ≥ 0.65)provably unreachable | EXP_p11_weaksup_localization · `eaf72db`,`0b3cf40` |

**两条方法学定论(横切,贯穿 T4):**
1. **过 no-head probe 是必要非充分。** P3-HateMM(三库最干净 probe +0.0108,证据最密)与 P8-EN
   (全 campaign 最强 probe,同时压过 floor +1.6 与朴素截断 +4.6)**均训练 within-noise / 劣于截断**——
   习得的 align-fusion 头吸收输入端优势。须双口径(val-sel + final-epoch)双看。
2. **语义能力 ⊥ 决策变量(或与之冗余)。** comparability ⊥ vote-correctness(P2/P2b 全 8 配置,7B–72B
   规模梯);localized-visual-evidence ⊥ 冻结 CLIP 可分性(P3);verdict-rate 在其估计的先验上漂移(P1);
   schema 字段 ⊂ 已监督标签(P4);语义通道与视觉票正相关且更弱(P7)。语义「关于什么」≠「在仇恨边界
   哪一侧」,而后者正是检索头**已直接监督**的量。

---

## T5 — 轮次 2–3 预注册负结果扩展(novelty-first;与 T1–T4 严格隔离,纯转录)

> **本节 append 于 2026-07-17,遵循与文首相同的纯转录纪律**(不跑实验、不提交 SLURM、不重算任何数)。
> T4 的 **13 条 campaign 路线计数保持不变**;本节是其后三轮(round-2 终结 2026-07-14、round-3 终结
> 2026-07-16/17、round-4 2026-07-17)在 **D7-收紧 novelty 门**下的预注册负结果**扩展**,不改 T1–T4 任何数、
> 不并入主表。计数纪律(见文末张力清单 #7):**13 = campaign 路线粒度**(T4);本节 #15–22(round-2)+
> round-3 六向 + round-4(T5.3)是 **负结果账**上的续接,两个计数轴**不可混淆**。**round-4 的 ordinal 张力
> 见文末张力清单 #9**(findings.jsonl 把 F47/F50 记为「22nd/23rd pre-registered negative」,该 ordinal 从
> round-2 *终结* 计数续接、与本节 #22=B4 的冲刺编号不对齐——本汇编不据此铸造有争议的总数,只逐轮记账)。用户 D7
> 裁决(encoder-class 杠杆不满足 novelty)见 `research-wiki/TERMINUS_round2_mllm_plus3.md` §8。每行数字转录自
> 命名的判决/记录文档,commit 内联。

### T5.1 Round-2 负结果(冲刺 7 条 #15–21 + B4 预-GPU 关闭 #22)

来源:`research-wiki/TERMINUS_round2_mllm_plus3.md` §1/§7。

| # | 路线 | 死因(一行) | 判决 · 记录 |
|---|---|---|---|
| 15 | A 线 `lb_scgp_global`(标签盲证书 → 全局 Gram) | G0-cond 探针预-GPU 关闭:缓存 91–93% 单一常数,oracle@覆盖率低于 +0.040 线一个量级,v3 否决(parse-ok 部分即噪声);省 264 GPU-h | A_LINE_PAUSE_DECISION.md |
| 16 | C1 RA-HMD 两阶段顺序 QLoRA | 锚论文消融把未测格定价仅 +0.7;实测 DEV kNN ≈ −0.02 vs 冻结 floor(job 13039) | C1_KILL_REVIEW.md |
| 17 | C3-target(真 Qwen-7B target 预测器作条件通道) | oracle 天花板 +0.0487 marginal,真预测器 ≈0(最佳 +0.0094 < +0.040),MHC 反信息;校准机器 | C3_REAL_PREDICTOR_PROBE.md |
| 18 | C2-SAV 稀疏注意力头挖掘(784 图像流头,冻结 7B) | F-G1 KILL 于修正机器下确认;MHC 格=压塌基线假象,HateMM 伤害真实;**稀释假说证伪**(MHC-EN 数据/标签受限) | SAV_F1_VERDICT_REVIEW.md |
| 19 | C3-nontarget 密集推理文本通道(最优配置融合) | DEAD_AT_FUSION:三条预声明融合规则在校准+置换-null 仪器上全败;CLIP-only 增益=编码器冗余(信息已在 Qwen 通路) | C3_FUSION_PROBE_RECORD.md |
| 20 | B1 frozen-Qwen 编码器 × MHC-ZH(3 seed 配对) | 双协议 FAIL(final-epoch 均值 −0.0112 acc,1/3 seed 同号;gates 干净);ZH 0.8537 系 LoRA 杠杆非冻结编码器 | B1_VERDICT_REVIEW.md |
| 21 | B2 Qwen2.5-VL-32B 冻结编码器(scale 轴) | goal FAIL:HateMM 上 32B 介于 CLIP 与 7B(**scale 退步**),MHC-EN/ZH 低于 CLIP,32B-vs-7B 全败——scale 非转换杠杆 | B2_VERDICT_REVIEW.md |
| 22 | B4 LoRA-Qwen 编码器 × MHC-EN(3 seed 配对) | **现已随 LoRA-HateMM 正式测量**(job 13235,F53):双协议 FAIL —— val-sel mean Δacc **−0.0021**(acc 2/3)/ final-ep **+0.0000**(acc 1/3),均 ≪ +0.030 门;seed0 anchor 逐位复现预-GPU 值(val-sel −0.0310 acc vs CLIP,低于两个冻结 floor);EN LoRA-encoder 单元关闭 | B4_FORENSIC_RECON.md → LORA_HATEMM_VERDICT_REVIEW.md |

**B3(唯一 marginal 正例,pending novelty 裁决,不并入主表):** LoRA-Qwen vs frozen-CLIP,MHC-ZH,3-seed
配对——**final-epoch +0.0313 acc / +0.0453 mF1,3/3 同号 → PASS(MARGINAL)**;**val-selected +0.0246 acc
FAIL** +0.030 AND 门。绑定语言逐字:`final-epoch: PASS (MARGINAL); val-selected: FAIL`。三条敏感度:门上
余量仅 +0.0013(≈门 4%);seed2 +0.0201 低于逐种子门;+0.0013 余量 ≈ 种子间散布 0.0201 的 1/15。分解:ZH
增益全部来自 LoRA 适配而非编码器身份(frozen-Qwen ZH −0.0112)⇒ 无单一 MLLM-encoder 机制跨 ≥2 库过线。
详见本文件 **PUR-1 / PUR-2 与 PUR-banner**(pending 用户 novelty / family-headline / :58 覆盖裁决)。
来源:`refine-logs/B3_VERDICT_REVIEW.md`(job 13150)· PUR-1。

### T5.2 Round-3 负结果(novelty-first;每轴以绑定判决或校准-零 $0 gate 关闭)

来源:`refine-logs/TERMINUS_round3_mllm_plus3.md` §0/§1;findings F37/F39/F41/F42/F43。**核心科学产出 = 两条
结构律**(见 `DRAFT_analysis_chapter.md` §3.6–3.7):law-I「better-signal-without-conversion」四实例
(P3 / S2S / W2-A / 编码器交换,F44 机制为收束);law-II「累积因果三层闭合」(F35 结构 / F37 无监督 / F39 监督)。

| 路线 | 死因(一行) | 判决 · commit |
|---|---|---|
| **S2S** — Qwen 帧组集合匹配(检索对象 / don't-pool) | KILL 双库:HateMM SET−POOLED +0.0035 acc / +0.0003 mF1 六条子条件全败于 +0.05 门,MHC-EN −0.0397;gold oracle 头空间 +0.0917 / +0.1399 而 MeanMaxSim 兑不出;跨双编码器关闭检索对象族 | S2S_PROBE_VERDICT_REVIEW.md · `2c96ab6` |
| **CTF** — 因果前缀帧组张量的监督时序池化 / arc 增量 | $0 条件信息 gate,四格全 kill,校准有效:[g_1…g_T] over 池化键 +0.0000(HateMM)/ −0.0029(MHC),arc −0.0049 / −0.0010;累积因果闭合的监督腿(§3.7) | CTF_GATE_RECORD.md · `0eb6d33` |
| **APX** — 整段古典韵律(eGeMAPS 88-d)辅助通道 | $0 gate 双条件全触发,校准有效:最佳臂 −0.0038,最严 raw-88-d 臂 +0.0005 = over Z_best 恰好零条件信息;ASR 转写已银行化口播仇恨内容,古典韵律条件冗余 | APX_GATE_RECORD.md · `9c54faf` |
| **AVC** — 韵律 × 视觉片段对应 | 未启动:门控于 APX 之后,随之而死;音频轴 parked | (门控于 APX;APX_GATE_RECORD.md) |
| **W2-A** — 转写优先 grounded 视觉键 | 双库 DEAD 于绑定条件信息 gate K9:Δacc −0.0000(HateMM)/ −0.0038(MHC)over 8960-d Z_best;咨询 kNN grounded 键劣于 concat(−0.0259 / −0.0509);"clean CLIP-redundancy null"(§3.6),第三个 oracle-exists-but-unconvertible | W2A_PROBE_VERDICT_REVIEW.md · `7228373` |
| **GIR** — 孤立 grounded-incongruity 残差(grd − ungrd) | $0 gate 五格全 kill:r_cache +0.0012(HateMM)/ −0.0051(MHC),r_field +0.0000 / −0.0064;残差为基线的**精确线性子集**(残差范数 0),W2-A K9 零在数学上包含它;池中最后一个候选 | GIR_GATE_RECORD.md · `b64a85b` |

**Round-3 recon/triage 关联关闭(6 项,喂给上表轴闭合):** W2-B(frozen-CLIP 子片段集合匹配,cloud-triage
判决 (d),`0f43bdd`)、W2-E(原型记忆,预仪式关闭)随 S2S 关闭检索对象 / 记忆重组族;W2-C(时序 order-kernel)在
S2S 死亡时熄灭(唯一授权载体);C5(7B 关系 CRD)、R3-C3geo(frozen-Qwen 几何硬负挖掘)为 D7 下 encoder-class /
冻结重组预仪式 no-go;**B5**(逐编码器阈值校准)证明 frozen-Qwen ZH 排序边在任何操作点(含 label-oracle 切点)
不可兑现(`50f01b9`)——性能/诊断线,回答 B1 之谜并支撑 §3.6 的 rotation-not-Pareto 读法。至 GIR,wave-3 池空、
冻结约束盒内每个注入点均由绑定判决或校准-零 gate 关闭;余下皆用户裁决,非继续搜索
(`TERMINUS_round3_mllm_plus3.md` §1/§4)。

### T5.3 Round-4 负结果(novelty-first;wave-4 选择/融合杠杆;$0 gate)

来源:`refine-logs/ROUTER_GATE_RECORD.md`(F47)· `refine-logs/MJ_FORENSIC_RECON.md`(F49)·
`refine-logs/FA_GATE_RECORD.md`(F50)· `refine-logs/WAVE4_CANDIDATES.md`(F48 correction, `6032d32`)·
`refine-logs/WAVE5_CANDIDATES.md`(F51, `7166232`)· `refine-logs/LORA_HATEMM_VERDICT_REVIEW.md`(F53, `6b8f634`,line-A
正例)。**核心科学产出 = law-I 增至第五实例(FA AUC 0.898 = 全 campaign 最尖锐的 better-signal-no-conversion)+ 新增
law-III「per-item selection 三监督源全闭」+ line-A(F53)确认 law-IV「convertibility 经由适配而非编码器身份」**
(见 `DRAFT_analysis_chapter.md` §3.6 / §3.8 / §3.9)。

| 路线 | 死因(一行) | 判决 · commit |
|---|---|---|
| **Router** — 逐项跨通道路由(CLIP 臂 vs Qwen 臂),决策级 meta-feature | $0 gate,可部署读与可实现天花板双 KILL:oracle 头空间真实(**+0.1083 MHC-EN / +0.0498 HateMM**),但 train→dev router 每 seed **+0.0000**(CLIP 头记忆化 train,LOO 0.998 vs Qwen 0.800 ⇒ 路由目标退化,「Qwen-correct」0/109·0/102·0/92 = dev 基率 0.55–0.65 的逆);dev-CV 天花板 **−0.0458** CI[−0.0875,0] 低于 perm-null p95 +0.0042(p=0.97);per-item 通道选择于**三监督源全闭**(§3.8);机器 12/12 逐位一致,oracle-calib accZA 1.000 | ROUTER_GATE_RECORD.md · `30d0ee1` |
| **MJ** — MLLM 模态可靠性判断作**新** router 输入(F47 明留的 carve-out) | 纯算术 pre-GPU NO-GO:清 +0.020 门需 which-arm-wins 精度 **q ≥ 0.663**,而模态-locus 对齐天花板 **≤ 0.588**(F44/F47 实测 ≈0.50–0.41)⇒ **完美判据也失败**(增益 ≈ 0 到 −0.046);判断已 **banked**(archive `modality_cues`,`d0f9e7b`,dev 全覆盖)⇒ 无需生成;$0 closure probe 依「ceiling-below-bar = kill 已完成」先例(A-line/G0-cond)**declined** | MJ_FORENSIC_RECON.md · `d57d05d` |
| **FA** — 模态重加权 / 跨编码器融合:F44 被抵消的 Qwen-text 增益在 MHC-EN 可兑现否? | $0 gate,KILL:within-Qwen 重加权在每个权重都是**纯 rotation**(w=0.5 时 F44-exact +0.040 hate / −0.036 non-hate);跨编码器 `CLIP-imĝ ⊕ Qwen-text̂` 把 MHC-EN dev AUC 抬到 **0.898 = 全 campaign 最高**却不可兑现:唯一点-Pareto 配置(Δacc +0.050)败于 bootstrap CI([−0.0625,+0.150])与 selection-null(p=0.766),label-oracle 阈值边仅 **+0.025 < +0.03**(移植 B5 kill-switch 触发);校准有效(HateMM 正对照 +0.0467 过)。**第五个 better-signal/no-conversion 实例**;修正 F44 concat→align(Hadamard)勘误(F44 数字经 sign-faithful 代理仍立) | FA_GATE_RECORD.md · `e0877c9` |

**Round-4 recon 关联关闭 + line-A 已落地:** **wave-4 候选枚举**(`6032d32`)判定冻结池空于 goal-hitting 候选,
并提出使 FA 格可测的 F44 concat→align 勘误;**wave-5 适配族 recon**(`7166232`)确立两-adapted-object 闭合——
适配只能触及**编码器**(通用 LoRA,D7-encoder-class)或**联合 encoder+decision**(检索损失进 LoRA = 已杀的 P9b
对象),无第三 adapted object,故唯一新成员(检索挖掘 hard-negative SFT curriculum)不开新数据集、门控于用户 D7
子裁决。**round-4 line-A**(通用 encoder-level LoRA-HateMM 3-seed 编码器跑,F53)**已完成并判决**(job 链
13233→13234→13235,verdict `6b8f634`):**HateMM PASS 双协议**(final +0.0573/+0.0682、val-sel +0.0419/+0.0460,
3/3),bundled B4-EN FAIL 双协议 —— 见 **PUR-3**(性能数)与 PUR-2(3-库地图)。它是**性能正例**(非负结果表
一行),无论 novelty 如何均为 encoder-class 杠杆(D7),**不并入主表 T1–T4**,不改 13 路线 campaign 账目或 novelty
结论;确认 `DRAFT_analysis_chapter.md` §3.9 的 adaptation law(convertibility 经由适配而非编码器身份)。

### T5.4 Round-4 收尾(closing:cand-2 curriculum coupling probe + premise-(d) EN composition gate)

来源:`refine-logs/CAND2_VERDICT_REVIEW.md`(F56,`546acc5`,job 13241,独立 0-context 判决,hash-verified vs
frozen prereg `76ef0e2`)· `refine-logs/PREMISE_D_GATE_RECORD.md`(F55,`6e6061b`,$0 CPU gate)·
`refine-logs/TIE_BRANCH_RECON.md`(F54,`6b9985a`,TIE-branch recon,premise 修正 + premise-(d) 识别)·
`research-wiki/experiments/exp-cand2-curriculum.md` · `research-wiki/experiments/exp-premise-d.md`。**这两项均**
**不并入主表 T1–T4、不改 13 路线 campaign 账目;cand-2 = tie(held pending D7 sub-ruling),premise-(d) = 第 6 个**
**better-signal/no-conversion 的 $0-gate 负结果。**

**(a) cand-2 curriculum —— confusion-weighted 单视频 SFT curriculum(唯一 manipulated variable = 样本重数,cost-neutral)**
**vs 通用 LoRA(K-C2-2)与 frozen-CLIP(K-C2-1),ZH + HateMM,3 head-seed 配对,两协议:**

| 数据集 | 协议 | curric mean acc/mF1 | Δ vs CLIP acc/mF1(K-C2-1) | Δacc vs generic(sign,K-C2-2) | K-C2-2 |
|---|---|---|---|---|---|
| **MHC-ZH** | val-sel | 0.8255/0.7947 | +0.0179/+0.0271 | −0.0067(1/3) | **tie** |
| **MHC-ZH** | final-ep | 0.8523/0.8249 | +0.0380/+0.0529 | +0.0067(2/3) | **tie** |
| **HateMM** | val-sel | 0.8775/0.8711 | +0.0573/+0.0626 | **+0.0155(3/3)** | **pass** |
| **HateMM** | final-ep | 0.8791/0.8726 | +0.0667/+0.0790 | +0.0093(3/3) | **tie** |

**判决(绑定,`CAND2_VERDICT_REVIEW.md` §5,frozen prereg §7.3 逐字):**

```
ZH:     final-epoch: PASS (K-C2-1, MARGINAL) · K-C2-2: tie · ZH-robustness: not strengthened.
        val-selected: FAIL (K-C2-1)          · K-C2-2: tie.
HateMM: final-epoch: PASS (K-C2-1, hold)     · K-C2-2: tie.
        val-selected: PASS (K-C2-1, hold)     · K-C2-2: pass (single-draw caveat, F0.2).
```

K-C2-1:ZH final-ep PASS(marginal)/ val-sel FAIL;HateMM 双协议 PASS(held)。K-C2-2:**ZH = tie 双协议**
(NO novelty on ZH,预声明 F0.7「generic LoRA with reshuffled data」),**HateMM = pass 仅 val-sel**(+0.0155 acc /
+0.0166 mF1,3/3;draw-1 单 SFT draw F0.2,rep2 后为 **pooled weakly-hardened across two draws(5/6 sign),
per-draw 3/3 gate not met**,见 (a-rep2);final-ep tie,+0.0093 < +0.010 门 0.0007)。
**ZH-robustness = NOT strengthened**(§3.7(a) val-sel 不过 + (b) final-ep 未变 non-marginal,ZH final +0.0380 <
+0.040、seed2 +0.0134 < 逐种子门;≈ B3 现状)。**KS-regression / KS-below-floor 均未触发,无 kill;合规干净**
(same-code 76/80 fields,单次 test-touch/库,F0.8 class-balance shift 预声明)。**载重读法:memory→adaptation
coupling 相对通用 LoRA 的可测效应是 dataset- 与 protocol-local**——仅 HateMM val-sel(rep2 后 pooled
weakly-hardened across two draws,5/6 sign;per-draw 3/3 gate not met,见 (a-rep2))有,主 ZH 腿无;
cand-2 **不开新数据集**(F0.4)、**不并入主表**;是否足够支撑 D7 memory-coupling 子裁决 = 用户裁决(见 PUR-4 /
PUR-banner + `refine-logs/D7_RULING_DOSSIER.md` `def6ce3`)。

**(a-rep2) cand-2 draw-2 replication(F59,`aa48275`,job 13246,独立 0-context 判决 vs frozen rep2 prereg**
**`2d15ffb`,banked arms 逐位复现)—— HateMM add-over-generic 复现读:**

draw-1 的 HateMM K-C2-2 val-sel PASS(+0.0155,3/3)只有单 curriculum SFT draw(F0.2),故预注册跑**恰一**独立
第二 draw(seed=1 为唯一 manipulated variable;draw-1 = HF default 42;curriculum multiset 与 draw-1 逐位一致,
sha `73307ef2…82b`)**仅 HateMM**。draw-2 val-sel add-over-generic 逐种子 **[+0.0139,−0.0047,+0.0233]**,mean
**+0.0108** acc(ΔmF1 +0.0120):点 bar(≥ +0.010)过,但 **3/3 sign gate 败**(seed1 −0.0047 → 2/3)⇒
**K-REP-1(主/绑定)NOT-PASS**;**KS-REP**(退休 kill,mean Δacc ≤ −0.014 才触发)**未触发**。pooled 两-draw
(K-REP-2:draw-1 [+0.0186,+0.0046,+0.0233] + draw-2 [+0.0139,−0.0047,+0.0233])mean **+0.01317**、sign
**5/6** ⇒ **HARDENED**。非绑定 final-ep add-over-generic:mean **+0.0140**,3/3。**判决(verbatim,
`CAND2_REP2_VERDICT_REVIEW.md` §6):F56 HateMM val-sel add-over-generic = WEAKLY-HARDENED** —— 未完全复现
(seed1 翻负 −0.0047、per-draw 3/3 gate 未过)、未反转、pooled 方向一致,仍是 **2-draw** 估计;单次 draw-2 attempt
**绑定且已消耗**(无更多 draw;`training_args.bin` seed=1 已核)。rep2 仅测 HateMM,**ZH-robustness 仍 NOT**
**strengthened**。故 `D7_RULING_DOSSIER.md` §5 (B) 分支第一半(HateMM add-over-generic)从「single-draw caveat」
升为「pooled weakly-hardened across two draws(5/6 sign),per-draw 3/3 gate not met」,ZH-robustness 半条仍未达。
Novelty 仍 = 用户 D7 sub-ruling,**不并入主表**。provenance:`2d15ffb`→`e2aee03`→`6c11988`→`d06ad07`→`aa48275`。

**(b) premise-(d) gate —— CLIP-img ⊕ LoRA-EN-Qwen-text(F50 的 adaptation carve-out):MHC-EN 转换否?**

| 路线 | 死因(一行) | 判决 · commit |
|---|---|---|
| **premise-(d)** — F50 ban 明留的 carve-out「conversion requires adaptation」:把 frozen Qwen-text 换成 **LoRA-EN-adapted** Qwen-text,保留 healthy CLIP-img | $0 CPU gate,KILL:FA 机器逐位复现 FA-A2(max\|diff\| **0.000000**,peak AUC 0.8982);LoRA-text swap **不闭合** +0.005 oracle 缺口(grid 上 max `d_oracle` 仍 **+0.0250 < +0.03**,移植 B5/K-D-1 kill-switch 触发),且 adapted text **恶化**合成:peak dev AUC **0.8982 → 0.8698(−0.0284)**(ZH 的镜像,F45 是 0.847→0.925 升);唯一点-Δacc 配置(+0.050)非 Pareto(Δnon-hate −0.0545),败 bootstrap CI([−0.0503,+0.1625])与 selection-null(p=0.7532);HateMM 正对照 +0.0467 过 ⇒ 校准。**第 6 个 better-signal/no-conversion**;EN 于 frozen(F50)/ collapsed-adapted(B4/F53)/ healthy-img⊕adapted-text(premise-(d))**三个 composition level 同时关闭** | PREMISE_D_GATE_RECORD.md · `6e6061b` |

**premise 修正(F54,`6b9985a`,honest scoping,记录于此):** F45「LoRA 只动 text 流、image 流 flat」是**经验**(SFT
target 属性)而**非架构**壁垒——vision tower / projector 冻结,但 LLM backbone(`lora_target: all`)re-contextualize
vision-pad tokens,banked `img_feats` 池化自穿过该 adapted backbone 的 forward,故 image 流**架构上可被** vision-obligatory
SFT target 移动;此处保持 flat 仅因 campaign 所有 SFT target 都是 transcript-present 的 text-decodable yes/no。相位图不变
(F50/premise-(d) 已把 EN 的 healthy image 流定价于 oracle 门下),但「text-only」应读作「text-only *for these targets*」。
详见 `DRAFT_analysis_chapter.md` §3.9 scoping note。

---

## T6 — 轮次 5–6 后终结审计(post-terminus robustness audit;与 T1–T4 严格隔离,纯转录)

> **本节 append 于 2026-07-25,遵循与文首相同的纯转录纪律**(不跑实验、不提交 SLURM、不重算任何数)。
> round-4 清空冻结约束盒后,用户指令下继续审计:**round-5 三-agent 红队**在**枚举**层面反驳穷尽 claim
> (6 个 cell 曾以 prose 论证但从未实测,`REDTEAM_UNTESTED_CELLS.md` `adb8bc2` / `REDTEAM_EXTERNAL_FAMILIES.md`
> `d0f91a5` / `REDTEAM_BAN_SCOPE_AUDIT.md` `5dd23e4`)并逐一实测关闭;**round-6 两波文献扫**提出若干可借操作并
> 逐一实测关闭 / parked。**这两轮不增 T4 的 13 路线 campaign 计数、不改 T1–T4 任何数**;项目最优数(HateMM
> cand-2 0.8775/0.8791)在 ~16(round-5)+ ~3.5(round-6)GPU-h 后**不变**。科学产出为确认性:四条结构律各经
> 一次直接攻击后存活,three mechanism sharpenings 折入分析章(`DRAFT_analysis_chapter.md` §3.6 arithmetic-Law-I /
> §3.7 causal-mask attack / §3.10 small-head 优化注)。**这两轮的 cell 记为 findings F61–F74,刻意不入
> campaign-route 计数、不占负结果 ordinal(逐轮框架,同文末张力清单 #9)。** 每行数字转录自命名的判决/记录文档,
> commit 内联(numeric-provenance discipline)。

### T6.1 Round-5 红队审计(6 个 prose-argued gap,全数实测关闭)

| 路线 | 死因(一行) | 判决 · commit |
|---|---|---|
| **LP** — kNN 记忆图上的标签传播 / 图扩散(决策 topology) | $0 gate,三库 KILL:多跳 LLGC 在同一冻结键上随扩散强度单调变负(HateMM 最佳 −0.0187,ZH −0.0385 / α=0.9 塌方 −0.19/−0.22,EN +0.0125 = net +1 item 落 perm-null p95 +0.063 内、null 中心为正);one-hop 已在 1-hop-separable 天花板;ZH oracle 头空间 +0.1026 未兑现(law-I 第 7 实例) | LP_GATE_RECORD.md · `7be6e3f` |
| **SWA** — per-epoch head ckpt 单轨迹权重平均(攻 F45 dev 选点税) | $0 probe,双库 KILL:HateMM SWA 在有真选点 gap 的两 seed 落 val-sel max 下 0.9–6.6 dev-acc 点(mid-peak dev 曲线,平均收不回);ZH regen(job 13294,G-repro bit-exact)= dev-underpowered KILL(cond_A 0/3;78-item dev jitter = 效应量级)。治理:单轨迹权重平均需用户 micro-ruling vs cross-seed-ensemble 否决,方可入 claims 表 | SWA_PROBE_RECORD.md · `5a40bb1`/`17db531` |
| **Learned-audio** — Whisper-large-v3 encoder 隐状态流(从未筛的 MHC-EN 音频空 cell) | $0 gate,三库双 Z-arm KILL:mean⊕max 2560-d 视频向量对部署表征加零条件信息(HateMM +0.0014,EN +0.0041 deployed / −0.0013 strict,ZH −0.0052/−0.0082;CI 全跨 0,accZA=1.0);ASR 转写已银行化口播仇恨 ⇒ **无 oracle 盈余——信号本身缺失,非 law-I**。关闭 EN 音频空 cell;仅 Whisper realization(AST/BEATs 仍 download-gated) | LAUD_GATE_RECORD.md · `3573f82` |
| **Vision-unfreeze LoRA** — LoRA-SFT 内解冻 ViT tower + projector(未枚举的表征 cell) | 3-seed 判决,~15 GPU-h:EN image 流 **MOVED**(+0.0320 train-LOO / +0.0065 dev,reviewer bit-for-bit——首个移动它的杠杆,反驳 F51/GAP-5b 措辞)但 K-V2 = **TIE** 双库双协议(HateMM val-sel −0.0016 acc 0/3,final +0.0000 1/3;EN val-sel +0.0269 acc sign 2/3,final −0.0062 1/3)——image 动、head 兑零(law-I 第 8 实例) | VISION_UNFREEZE_VERDICT_REVIEW.md · `09d02f8` |
| **ISR** — 独立 per-segment 重编码 + uniform per-segment-kNN vote-mean(最后一个聚合对象) | $0 pre-gate,NO-GO:合法 uniform 算子 flat(HateMM +0.0012 / EN +0.0032,低于 perm-null,boot-5th < 0,vote bit-exact Fano 1.0);决定性 β-分解证明 oracle 头空间 **selection-locked**——HateMM +0.0776 = +0.0012 legal + +0.0764 banned,EN +0.0700 = +0.0064 + +0.0636(91–98% 仅 banned-selection)⇒ **law-I 现为算术命题**;Qwen per-segment 提取从不发生,0 GPU-h | ISR_PREGATE_RECORD.md · `a6e41f8` |
| **Frame-16** — 视觉采样 8→16 帧(冻结编码器) | 3-seed 判决 vs banked 8f floor,~0.6 GPU-h:val-sel mean −0.0077 acc(0/3),final +0.0015(1/3);KS-16f-dead 双协议 KILLED ⇒ cell 关闭,昂贵 LoRA-16f stage-2 **AUTO-DEAD**(预声明 spend 判决);8 帧非瓶颈,池化表征在 8f 已饱和 | FRAME16_VERDICT_REVIEW.md · `32c2e6f` |

### T6.2 Round-6 文献扫审计(可借操作,全数实测关闭 / parked)

| 路线 | 死因(一行) | 判决 · commit |
|---|---|---|
| **Grad-norm selection** — 最小 head-gradient-norm 的 validation-free 选点(arXiv 2601.16874;攻 F45 ZH 选点税) | $0 probe,MECHANISM REFUTED:论文前提(Spearman(‖g‖, acc) ≈ −0.85…−0.98)在我们微型 head 上**反号**(+0.61/+0.72/+0.62,3/3 seed);scale-normalized grad 随 accuracy 单调升,argmin 落最差 epoch;F68-P2 于 $0 killed(可晋级 ZH/HateMM-curric ckpt 已 disk-prune 到 B2,鉴于反号 restore 无意义) | GRADNORM_SELECT_PROBE_RECORD.md · `ada5849` |
| **Readout axis** — intermediate-layer / one-word-prompt / last-token 提取变体(MLLM-embedding 范式内唯一未枚举轴) | $0 CPU screen,KS-readout-dead(~2 GPU-h 仅提取):ZH 最佳 +0.0128 dev-query,HateMM 最佳 +0.0093 LOO——均落 perm-null 带内(p95 +0.0769 / +0.0939,boot-5th < 0);one-word readout 令 HateMM 退步(−0.056/−0.065);部署 final-layer mean-pool 已在局部最优,无 head,零 test-touch | READOUT_SUBMIT_RECORD.md · `a60f6cf` |
| **MCR** — modality-competition rebalancing / data-remixing SFT schedule(逼 collapsed EN image 流在适配中承载) | forensic recon,PARKED(无 GPU):honest transplant 存在(EN-only masked-SFT schedule,~4–6 GPU-h)但 F65 已 nulled 同轴(image 动、零兑现)且 F55 把 EN 流-rebalance oracle 封在 +0.025 < +0.030 门 ⇒ 算术封顶,prior ~5–8%;作 user-gated paper-closure null 保留,非性能 bet | MCR_FORENSIC_RECON.md · `6d0495b` |
| **Bidir mask-flip** — training-free causal→bidirectional attention(LLM2Vec / NV-Embed;最高 novelty) | 3-seed 判决,~1.2 GPU-h:**DEGRADE 双库**——ZH mean −0.1163(val)/ −0.1409(final)acc,HateMM −0.1210 / −0.1256,0/12 逐种子 delta 正,至 −0.28 macro-F1;"Llama-pattern" 塌方(≈7–10× −0.014 线)直接确认部署表征依赖 causal prefix(§3.7);Stage-2 MNTP 路由用户 funding 决定,非 auto-defund | BIDIR_STAGE1_VERDICT_REVIEW.md · `f733bbe` |
| **Head-recipe** — align head 上 SAM flat-minima 优化器 + modality-dropout | 3-seed 判决,< 0.15 GPU-h:4 个 arm×dataset cell 全 KS-arm-dead、FORMAL-FAIL 双协议——SAM×ZH −0.0246/−0.0424(伤),SAM×HateMM +0.0047/+0.0046(within-noise,非 3/3),mod×ZH 0.0000/−0.0313,mod×HateMM −0.0201/−0.0062;两 disclosed headwind(F69 反号 SAM、F45/F58 text-carried mod-dropout)兑现 | HEADRECIPE_VERDICT_REVIEW.md · `8e60f42` |

**within-noise 观察(非主张,不入任何主表):** SAM×HateMM 在 val-sel 把均值抬 **+0.0047 acc**(≈+0.5 点),
其**单 seed 最高 0.8884 val-sel = 全项目 HateMM 单值最高**,但均值远低于 +0.030 门、acc sign 非 3/3
(KS-arm-dead)⇒ 记为 **within-noise,永不作主张**。

**外部验证(round-6 lit-sweep,triage-only,不与本地 G-repro 数混表):** 已发表 HateMM 榜首 **MM-HSD 0.878
macro-F1** 仅经**我方否决的 OCR 通道**超过 cand-2——去 OCR 后 **0.845**,落在我方 0.8775/0.8791 带内;SOTA
视频-embedding 工作(VLM2Vec-V2 / VidVec)**不用任何时序算子**,独立佐证 F35/F37/F67 时序闭合。约束盒被外部
佐证,而非被击破 [DOC:LITSWEEP2_FRESH_2026.md;DOC:LITSURVEY_MLLM_EMBEDDING.md]。

**来源(T6.1+T6.2):** round-5 = `refine-logs/{LP_GATE_RECORD, SWA_PROBE_RECORD, LAUD_GATE_RECORD,
VISION_UNFREEZE_VERDICT_REVIEW, ISR_PREGATE_RECORD, FRAME16_VERDICT_REVIEW}.md`;round-6 =
`refine-logs/{GRADNORM_SELECT_PROBE_RECORD, READOUT_SUBMIT_RECORD, MCR_FORENSIC_RECON,
BIDIR_STAGE1_VERDICT_REVIEW, HEADRECIPE_VERDICT_REVIEW}.md` + 综述 `LITSURVEY_{RETRIEVAL_MEMORY,
MLLM_EMBEDDING, NOVEL_MECHANISMS}.md` / `LITSWEEP2_{HEAD_OBJECTIVES, INPUT_FIDELITY, FRESH_2026}.md`。
findings F61–F74(`state/findings.jsonl`)。**不入负结果 ordinal 账、不铸造总数(逐轮框架,同文末张力清单 #9)。**

### T6.3 Round-6 波次 3–5 审计(litsweep2 batch-3 / litsweep-3 batch-4 / litsweep-5;实测 + $0 recon-park,全 null/park)

> **续 append 2026-07-25**(同纯转录纪律)。波次 3–4 = F75–F80,波次 5 = F81–F82。measured-dead:NCA(F75)、
> ZH-prompt(F80);$0 recon-PARK:resolution(F76)、curation(F78)、ELR(F79)、graded-label(F82);lit-sweep
> companion:litsweep-3(F77,3 agent)、litsweep-5(F81,3 agent)。**不改 T1–T4 任何数、不增 13 路线计数、不占
> 负结果 ordinal**;项目最优数(HateMM 0.8775/0.8791;ZH final 0.8456/0.8173)不变。**premise 修正三则(全部
> provenance-noted):** (1)litsweep2「~6.5× 下采样」= 捏造 720p 前提 → ffprobe 实测 **HateMM 2.71×(源 480p)/
> EN 10.55× / ZH 13.71×(源 1080p)**;(2)「ZH transcript median ~4 words / 退化」= whitespace-split 伪影(中文
> 无词间空格)→ 部署 ZH 文本 = median **~106 汉字**(train 106/val 108.5/test 105)Bilibili 描述元数据(非
> Whisper ASR);(3)test-set size **EN=161 / ZH=149 / HateMM clean=215** 与 wiki 既有一致(litsweep-5 纠的是
> task-prompt 的 ~2k/500/430,非 wiki)。

**T6.3-a 波次 3–4(F75–F80)**

| 路线 | 死因(一行) | 判决 · commit |
|---|---|---|
| **NCA / soft-kNN head-loss 族** — head 的 triplet+BCE 换成 vote-consistent(NCA τ0.1/0.2)/ contrastive(SupCon)/ mixup-BCE(4 臂,最直接优化部署 kNN vote 的损失) | 1-bite 3-seed 判决,~0.33 GPU-h(job 13482):**0/8 FORMAL、7/8 KS-arm-dead**;族内最大 A3-mixup ZH final +0.0134(2/3),唯一 KS 存活 NCA τ0.1×ZH val-sel +0.0112/+0.0113(3/3 sign)落 ±0.014 带下 = within-noise hardening、D7-dead;**首个「trained-reshaping 兑现 oracle 头空间」的实测负例** ⇒ law-I 对 trained 算子亦成立(§3.10);codex gate 捕获 + re-freeze 修 A3-only dropout-mode confound(预 spend) | NCA_VERDICT_REVIEW.md · `f03cae0`(+REFREEZE `8f08e9f`/`467a6f4`) |
| **Spatial resolution** — 抬 per-frame `max_pixels`(151200px)向源生分辨率(最后一个 virgin input-fidelity 轴) | $0 forensic recon,PARK:litsweep2「~6.5×」前提 = 捏造 720p → ffprobe 实测 **HateMM 2.71×(480p)/ EN 10.55× / ZH 13.71×(1080p)**;headroom 与 conversion **反相关**(唯一转换的 HateMM 近生;有 headroom 的 EN 塌陷/ZH marginal),mean-pool 衰减,F65 law-I + F70 readout-null 双约束,提取 raw-video-bound ⇒ **无 Modal-triage 路径**;≥+1 HateMM ~5–10%,≥+3-on-2 <3%;~1 GPU-h HateMM@410k door-closer 已 spec、未跑 | RESOLUTION_FORENSIC_RECON.md · `5c6075b` |
| **Memory-bank curation** — train-label-only 剪枝 / prototype-select / class-balance 部署 kNN bank(LOO-influence / Data-OOB;自动化 human-2-entry-EN 删除) | $0 forensic recon,PARK:「$0 on banked keys」前提 **FALSE**——部署 vote 索引 *trained head embedding*,6 个 floor head ckpt(13150/13241×3 seed)已 disk-删,faithful multi-seed pregate 需 ~0.3 GPU-h re-mint;唯一 $0 对象(raw fused key)seed-independent → single-draw = 已撤回的 archive-as-key 失败类;F63(1-hop-separable、正 perm-null)+ W2-E(prototype 已死)+ Wall-C 封 prior;≥+1 ~5–8%,+3 ~1% | CURATION_FORENSIC_RECON.md · `7025391` |
| **ELR / noise-robust head** — additive early-learning 正则(lead)+ co-teaching(contrast)于 FAISS-mined pairs(在 F75 ban letter 外) | $0 forensic recon,PARK:mined pairs 由 *gold label 过滤* ⇒「mined-pair noise」≡ gold-label noise(pillar-3 之对象);ELR 挂 BCE 腿而部署 kNN vote 不读它(二阶);noise proxy 13–17% raw-space 上界、boundary-hardness 主导;**Wall-C 量化**(HateMM test 峰 ep18/21/24,+4/+7/+14 于 dev 饱和后;ZH final−valsel = +0.0134 ×3 seed)令 early-target pull 反向;≥+1 ~5–8%,+3 ~1–2%;0.16 GPU-h probe 未跑 | ELR_FORENSIC_RECON.md · `9e41447` |
| **ZH 中文指令重提取** — 把部署英文提取 instruction/scaffolding 译成中文(ZH-path 唯一未变轴;测 SFT 训练/推理 language-mismatch 假设) | 3-seed 判决,~1.1 GPU-h(job 13487,KS-parity bit-exact):**双臂双协议 KS-dead**——LoRA −0.0358 val-sel / −0.0112 final acc,frozen −0.0336 / −0.0045;两 val-sel 腿过 −0.014(中文 prompt *伤*);mismatch 假设 **REFUTED**(双臂近等幅回退)⇒ extraction-instruction-language 轴 CLOSED、D7-dead | ZHPROMPT_VERDICT_REVIEW.md · `1a8c5fe` |

**T6.3-b 波次 5(F81–F82)**

| 路线 | 死因(一行) | 判决 · commit |
|---|---|---|
| **Graded 3-class soft-label** — 给合并进正类的 Offensive 一个更软的正 target(MHC 3-class {Normal,Offensive,Hateful},部署合 Offensive+Hateful→1;EN-revival longshot) | $0 pre-gate,PARK:label-independent 检索 ⇒ vote 对 Offensive 权重线性、τ 栅格与 oracle 精确;Offensive 是正类多数(EN 73%/ZH 63%),下压把真阳拖向 Normal——honest proxy 双库每 τ 单调负(ZH loo τ0.25 −0.1538),完全 gold-cheat oracle 顶棚仅 **EN +0.0250 / ZH +0.0256(均 < +0.030)**,无臂过 F63 perm-null(真 Offensive 下压 ≯ 随机等量正子集)⇒ F44 within-positive 墙的 label 轴算术化;机器 parity bit-exact,0 GPU-h | GRADEDLBL_PREGATE_RECORD.md · `c4333ce` |

**外部前沿综合(litsweep-5 S2,triage-only,不与本地 G-repro 混表):** 2023–2026 HateMM 已发表前沿——合法通道
CMFusion 0.823/0.860、Koushik(CLAP)0.854/0.848、RAMF(32B)0.856/0.851、Xiong 0.849/0.840、Wang 0.820——**全
≤ HOUSE 0.879/0.873**;唯一超我方的 MM-HSD 0.878/0.874 靠**否决的 OCR**(ablation 掉任一模态 → mF1 0.815–0.845,
OCR load-bearing)⇒ **无合法已发表路径到 HateMM > 0.88**。MHC-EN 前沿(RAMF 0.740/0.717、coarse-video
0.684/0.644、GPT-4V 0.63 mF1)全 ≤ 我方 ~0.79–0.81 ⇒ EN 是 field-ceiling label-limited、非 method-limited。TANDEM
(2601.11178)HateMM 0.78/0.79、MHC 0.67/0.38,在我方之下且多重越界(gold+RL),佐证时序闭合。HOUSE 0.879/0.873 =
curric-LoRA final-epoch 0.8791/0.8726 的外部论文 3dp 口径,**不改 T1–T4**。[DOC:LITSWEEP5_HATEMM_EN.md `36d833e`;
LITSWEEP5_TEMPORAL.md `ad81ffb`]

**per-annotator 票不存在(limitations 硬约束):** MultiHateClip 仅释出聚合 majority-vote 3-class label(2 标注者→
分歧第 3→专家升级),**无 per-annotator 票/票数**(不同于 HateXplain),in-repo 与公开版皆无 ⇒ LeWiDi /
annotator-distribution soft-label 谱系在**数据层**被封,annotator-level 建模不可行。[DOC:LITSWEEP5_HATEMM_EN.md §1;
arXiv 2408.03468]

**来源(T6.3):** 波次 3–4 = `refine-logs/{NCA_VERDICT_REVIEW(+NCA_REFREEZE_FIX/REVIEW), RESOLUTION_FORENSIC_RECON,
CURATION_FORENSIC_RECON, ELR_FORENSIC_RECON, ZHPROMPT_VERDICT_REVIEW}.md` + lit-sweep
`LITSWEEP3_{SELECTOR_CONVERSION, ZH_SPECIFIC, DATA_CENTRIC}.md`;波次 5 = `LITSWEEP5_{TEMPORAL, HATEMM_EN,
COMPLETENESS}.md` + `GRADEDLBL_PREGATE_RECORD.md`(+`OUT.json`)。findings F75–F82(`state/findings.jsonl`)。
**不入负结果 ordinal、不铸总数(逐轮框架,同文末张力清单 #9)。**

### T6.4 Round-7 融合门闩 + 代码复现启发战役(F83–F87;实测 null + $0 survey/gate/park)

> **续 append 2026-07-26**(同纯转录纪律)。measured-dead:fusion-concat(F85,~0.1 GPU-h)、MokA-ZH(F87,
> 5.573 GPU-h,measured-not-promoted);$0:repro-survey(F84)、LSMI PID gate(F86)、SynIB port(F86 kill-switch
> 触发 → PARK)。**不改 T1–T4 任何数、不增 13 路线计数、不占负结果 ordinal**;项目最优数(HateMM 0.8775/0.8791;
> ZH final 0.8456/0.8173)不变。**law-I 实例计数仍为 8 ——F87 的第 9 例明确未获认证**(见 T6.4-b)。
> **⚠ 后续更新(round-8):F87 的「未认证」判定不变,但第 9 例已由 F91(Molmo2 编码器交换)另行认证,现行计数 = 9;
> 全文口径见 T6.5 末「law-I 计数对账」。**

**T6.4-a 融合门闩 + 复现调研(F83–F85)**

| 路线 | 死因(一行) | 判决 · commit |
|---|---|---|
| **Trained fusion operator** — 把部署的 `align`/Hadamard 换成 head 内**训练过的** `concat`+MLP(唯一一个从未在视频上跑过的 first-class `fusion_mode` 分支;recon 证实 F50 只禁 *fixed* composition、F75 只禁 *loss* swap,双 ban letter 均 over-reach) | 1-bite 3-seed × 2-dataset 判决,~0.1 GPU-h,**零代码 diff**(job 13514;branch-assert 6/6、3 个独立 parser 48/48 一致、prereg re-hash 匹配):**两个 cell 全 KS-arm-dead**——ZH val-sel +0.0067(2/3)但 final −0.0045(1/3);HateMM −0.0031 / −0.0031(每条腿 0/3);4 条腿无一接近 FORMAL +0.030/+0.030;无 KS-regression(最差 −0.0045 > −0.014)。HateMM 全部效应 ≤ **2 个翻转样本**/seed(n=215;6 个 Δacc 中 **3 个**恰为 0.0000,其余为 −1/−1/−2 样本——**更正**源记录 NB-3 的「4 个」,该数与其自身 D5(a) 表及本次重读的原始 `RESULT_ROW` 均不符);ZH val-sel 均值 = **+1 样本/seed**(n=149)⇒ **fusion-operator 轴 CLOSED,为实测 null**;该 null 属于 **concat + 2.0× first-Linear 捆绑**(2,098,176 vs 1,049,600 参数),**不得**升格为「加 head 容量没用」 | FUSIONCAT_VERDICT_REVIEW.md · `129fe2e`(recon FUSIONSWAP_FORENSIC_RECON.md · `934bc9a`) |
| **Repro-survey** — 邻域(**非**仇恨视频)2025-H2→2026 **带可运行代码**工作,按 file-tree + `py_compile` 而非 README 分诊 | $0(无 GPU / SLURM / 权重下载):clone 8 个 repo(91 MB,gitignored);按启发价值排序为 **SynIB(arXiv 2606.09853)> LSMI(ICML 2025)> MokA(NeurIPS 2025 Oral)**,但**执行顺序**以 **LSMI 优先**(它是 gate 住 SynIB port 的 $0 诊断);UniME-V2 于源头排除(MLLM-as-a-judge 软标签 = 被禁的 P11/P2 监督源),VLM2Vec 无我方没有的 pooling(仅 `last\|mean\|cls`),**VidVec `main` 分支为空**(无代码),**RASR 被作者撤稿**(v2,2026-06-30);HateMM 数据集论文自带 baseline 代码**按发布状态无法编译**(`Codes/1.FastTextEmb_and_LASEREmbExtraction.py:45` SyntaxError),而 LSMI/SynIB/UniME-v2/BalanceBenchmark/MokA/VLM2Vec 分别 4/4、99/99、95/95、55/55、214/214、271/271 编译干净。**覆盖度自陈:** WebSearch 配额 200/200 用尽 ⇒ 这是 **8 个仓库的深度分诊,不是穷举枚举** | REPRO_SURVEY_2025.md · `9367338`(+ ERRATUM `81e2eaf`) |

**SynIB 刻画勘误(F84,进入论文框架):** survey §4.1/§6 把 SynIB 目标写成「intact 与 masked 预测之间的
symmetric KL」——**源码层为假**,一律以 port recon 对 `synib_mask_model.py` 的读法为准:intact 预测**从不**进入
任何 KL;live 变体为 Gaussian KL→N(0, I) 无信息先验、Dirichlet KL、以及对 detached 单模态 anchor 的 **forward**
KL(Hateful-Memes 配置);唯一的 symmetric logit-KL helper 在仓库里是**注释掉的**。同一读法另得三条上游事实:
`zeros | noise | ema` 三种 mask fill 是死代码(live 路径为 batch-permutation fill)、config `p` 键失效
(`p_min = 0.30` 生效)、HM anchor head **无梯度路径**(未训练随机初始化)。[DOC:REPRO_SURVEY_2025.md ERRATUM
`81e2eaf`;SYNIB_PORT_FORENSIC_RECON.md `9e638ea`]

**T6.4-b 复现启发战役 —— shortlist 自上而下执行(F86–F87)**

| 条目(survey §5 执行序) | 结果(一行) | 成本 | 判决 · commit |
|---|---|---|---|
| **LSMI**(#1)—— 对部署双流在 banked train+dev 缓存上做 sample-level PID(`R`/`U1`/`U2`/`S`),三条 lineage 全测 | **先认证机器再信数**:released `d'=64` 配方把**确定性** XOR 读成随机(joint out-of-fold 0.513/0.530/0.508)⇒ 该层 `LSMI_MEASUREMENT_INVALID`;认证维 **`d*=16`**(`d'=8` 复现:最大 synergy 0.6931 被还原为 0.7077/0.7321/0.7105,误差 ≈2%);duplicate-stream 真值-0 控制在认证维 **恰为 0.0000**,而在 `d'=64` 为 +0.0838/+0.1516/+0.2240。认证维结果:`S` = **−0.0747(ZH)/ −0.0802(HateMM)/ −0.0000(EN)**,6 个 cell 中 5 个 ≤ 0,perm-null q95 = 0,dev 复现(−0.0004/−0.0575/−0.1041);`U2`(text)为 5/6 cell 最大 atom(0.076–0.237),**`U1`(image)在 5/6 cell 恰为 0.0000**;`I12` 0.149–0.359 nats。机械标签 INDETERMINATE(ZH/HateMM)+ FUSION_CAPPED(EN)——**synergy 半边处处触发**,dominance 半边因该对是 **uniqueness-dominated(text 侧)** 而非 redundancy-dominated 而不触发 ⇒ 支撑句:**fusion 只能重组 `R`/`U1`/`U2`,没有 `S` 可捕获**(F50 rotation + F44 EN image collapse 的机制层解释,与 F85 concat null 一致但**不**互相推导) | **0 GPU-h**(纯 CPU,零 test-touch) | LSMI_GATE_RECORD.md · `a8905ac`(预声明链 `d4b06f0`→`362a60e`;撤回记 `915a60d`) |
| **SynIB**(#2)—— masked-branch 信息瓶颈项,**加**在 triplet+BCE 之上(非替换) | **PARK,未跑。** port recon 预声明「LSMI 读数」为 kill-switch,触发的是分支 (a)(`s ≈ 0` 全数据集)⇒ 一个为把 head 推向 *synergistic* 结构而设计的目标,在这里没有结构可推(recon 对 goal 门的 prior 1–2%)。条件项 BalanceBenchmark(survey #4)因「有 synergy 才需平衡」而**不解锁** | **0 GPU-h** | SYNIB_PORT_FORENSIC_RECON.md · `9e638ea`(park 随 gate 记录于 `a8905ac`) |
| **MokA**(#3)—— modality-routed LoRA(per-modality 下投影 `A`、共享 `B`、`r_v=r_t=16`)进部署 ZH encoder-SFT;**本战役从未变过的 PEFT-adapter-structure 轴** | **MEASURED — NOT PROMOTED。** `final-epoch: fail; val-selected: fail`(对两个 floor 皆然);对 banked merged floor 13150 **双协议 +0.0000 acc**。drift 门 6/6 触发(最差 mean per-item cos **0.99954879** < 0.9999)⇒ same-path **unmerged** floor 强制且 binding;对之 arm 读 **+0.0268 val-sel(3/3)**——D7 裁定**不可归因于 routing**:routing **完全缺席**时 unmerged 路径自身丢 **−0.0268 / −0.0340(0/3)**,三比较为同一恒等式 **+0.0000 = +0.0268 + (−0.0268)**;主导 seed −0.0604 = **−9/149 样本**,其 val-selection 塌到 ep5(两 epoch 并列 Val 0.8718,roc tie-break 取早者),而**无选点**的 final-epoch 协议同一操作只 −0.0067(1 样本)。`KS-MOKA-3` = **NULL-OP**:text **FLAT**(Δ train-LOO −0.0007/+0.0018)⇒ prereg 自己的 text-side 赌注**被证伪**;image **AMBIGUOUS 非 MOVED** ⇒ **第 9 个 law-I 实例未获认证**(计数仍为 8);visual-modality-protection 叙事**被禁**。经济解释 = F0.6 regime 反转:我方 SFT **94.6% vision token**(median 2,688 vision + 153 text)vs MokA 自家 **98.4% TEXT**(16,128 vs 256)⇒ routing 让 `A_t` 不被稀释却**被饿**(占位 100%→≈5.4%,token-gradient ≈18× 少),`A_v` 梯度范数比 `A_t` **低 25–40×**。test-touch 6 花 / 6 预算 | **5.573 GPU-h**(job 13537/13551/13552/13566/13573 = 0.003/0.532/3.414/1.212/0.413;cap 4.70 ⇒ +0.87,**+18.6% 已披露**,逐项映射到预注册条目) | MOKA_VERDICT_REVIEW.md · `91f64a6`(submit `ed609eb`,re-freeze `72a947b`) |

**methods-note(F86,进论文方法/附录):** released LSMI 的 entropy-estimator 循环**从不调用 `optimizer.zero_grad()`**
(`main_lsmi.py:174-187`),KNIFE 梯度全程累积;该缺陷在作者 2 维 demo 中 4dp 不可见,在我方维度下**改变结论**——
判别器逐位相同而熵估计移动(ZH `H1` 1130.19 → 710.62)、**`S` 符号在 3 个数据集中 2 个翻转**(ZH +0.2345 → −0.0672;
HateMM −0.1517 → +0.1152)。更关键:**released in-sample 读法在我方 n 下饱和**(三判别器 acc 0.99–1.00,三 pointwise
MI ≈ log 2),对**真实对**与**真值不同的两个控制**(duplicate-stream、split-half)**给出同一句**「redundancy-dominated,
`S ≈ 0.02`」⇒ 按发布状态跑会产出一个干净、可引用且**错误**的论文结论;cross-fitted 读法(在合成控制触发后、任何
RGCL cache cell 之前声明)是抓到它的原因。

**D7 裁定(F87,写作纪律,binding):** 把 +0.0268 报成 routing 增益属于 `0.8732` 纪律要针对的同类 numeric-provenance
违规;正确的一句是「modality-routed LoRA 把 ZH encoder 落在 shared-`A` floor 上」,**不是**「+0.027 val-sel 增益」。
同一 cell 产出两条可迁移方法学:(i) **同路径 floor 是任何 adapter-structure 比较的默认成本**,非 contingency;
(ii) merged/unmerged 的 bf16 漂移**不对称**——text 流比 image 流远 ≈3×(均值 ≈0.99955 vs ≈0.99985),而两个被测协议
都骑在 text 流上。F0.2 同时 binding:**仅一次 SFT draw**,`--seed` 只变 head ⇒ encoder-draw 噪声与 routing 效应
**不可分离**(limitations)。

**来源(T6.4):** `refine-logs/{FUSIONSWAP_FORENSIC_RECON, FUSIONCAT_VERDICT_REVIEW, REPRO_SURVEY_2025(+ERRATUM),
SYNIB_PORT_FORENSIC_RECON, LSMI_GATE_RECORD(+LSMI_GATE_OUT.json), MOKA_VERDICT_REVIEW, MOKA_SUBMIT_RECORD,
MOKA_REFREEZE_FIX}.md`。findings F83–F87(`state/findings.jsonl`)。round-7 总 GPU ≈ **5.7 GPU-h**
(MokA 5.573 + fusion-concat ~0.1;survey / PID gate / SynIB park 皆 $0)。**不入负结果 ordinal、不铸总数
(逐轮框架,同文末张力清单 #9);law-I 实例计数保持 8**(round-8 更新为 9,见 T6.5 末对账)。

---

### T6.5 Round-8 误差取证 + $0 关闭链 + 最后两个 gated 通道(F88–F98)

> **续 append 2026-07-28**(同纯转录纪律:不跑实验、不提交 SLURM、不重算任何数)。本轮四条线并行:
> ①**三库误差取证**(F88,$0)——首次拿到 per-item 的失败结构;②**$0 in-box 关闭链**(F89 eval-time vote
> operator / F94 top-k / F96·F97·F98 三条 LITSWEEP-6 pregate)——把「误差结构本身建议的修法」逐条实测掉;
> ③**两个 user-gated 通道兑现**(F90 CLAP general-audio、F91 Molmo2-8B 编码器交换);④**MNTP 三臂**
> (F92 S1/S1b readout、F93 S2a published-adapter transplant)。**不改 T1–T4 任何数、不增 13 路线计数、不占
> 负结果 ordinal**;项目最优数(**HateMM 0.8775/0.8711 val-sel、0.8791/0.8726 final;ZH final 0.8456/0.8173**)
> **不变**。round-8 GPU ≈ **3.0 GPU-h**(MNTP S1+S1b 1.691 + S2a 1.006 + Molmo2 抽取 ~18 min ≈ 0.3;F88/F89/
> F94–F98 与 F90 的 gate 全为 $0 CPU)。**两条载重更正**:law-I 实例计数 **8 → 9**(F91);pillar-④ 的 EN
> 2-entry 记忆编辑正结果为 **single-seed**(F88)。二者口径见本节末。

**T6.5-a 误差取证与 $0 in-box 关闭(F88 / F89 / F94)**

| 路线 | 死因(一行) | 判决 · commit |
|---|---|---|
| **ERRPAT** — 三库 per-item 误差取证(HateMM/MHC-EN/MHC-ZH),$0 GPU,CPU proxy 逐 cell 4dp 验证 | **不是 kill,是本轮的结构性发现 + 6 条新实测 null。** 误差集 **~90% seed-invariant**(HateMM 24–25 of 26–28 错在 3/3 seed;ZH 25 项并集中 22 项 3/3、无一项恰为 2/3;EN 4 seed 共识 22 项 + 20 项 seed-flip 噪声带 + 119 项从不错),且**每个错误都是 confident neighbourhood inversion**:HateMM 中位 top-20 真标签 rank-weighted purity **0.1667**、median \|vote\| **0.7267** vs 恒对项 0.9873、top-1 邻居带真标签仅 **7.4%**(恒对项 95.2%);ZH 中位 purity 0.15 / core **0.1167**(22 项 stable core **无一**邻域多数正确)、median \|vote\| 0.7137 vs 0.9999;EN 共识误差正确类占比 **0.2205** vs seed-flip 0.4781 vs 恒对 0.8738。**不是覆盖问题**:ZH raw fused 空间里首个同 gold-class train 邻居的**中位 rank = 1.5**(11/22 在 rank 1,22/22 在 rank 14 内)——正确类比项在场、排前,只是**被投票压倒**。6 条新 null 全为 door-closer:HateMM 全局阈值重校准(dev-fit acc **+0.0000**/+0.0016,train-LOO logistic −0.0016;test-fitted ORACLE 仅 +0.0078)、length de-bias(train-LOO −0.0016,dev-fit 系数**反号**)、LOO bank curation(+0.0016,**不敌**同量随机删除 +0.0031/+0.0000);ZH test-fitted 阈值 ORACLE 均值 **+0.0201**(低于门的 gold-cheat 上界)、Whisper-ASR 换文本通道 $0 天花板 **+0.0134**;EN dev-selected 阈值 **0/6 arm 改善**(Qwen −0.0083 acc / CLIP −0.0104)⇒ **三库 in-box $0 开放集 EMPTY** | ERRPAT_{HateMM,MHC-EN,MHC-ZH}_2026-07-26.md · `ad56a62` |
| **MECHFIX** — 5 个 eval-time vote 算子替换部署 top-20 rank-weighted signed-cosine 投票(T1 class-balanced quota / T2a CSLS hubness / T2b Ledoit-Wolf whitening / T3 精确 1-D length 方向剔除 / T4 whiten+balanced),三库配对同 head | $0 pregate(零 GPU/SLURM/Modal/训练),15/15 test + 15/15 dev floor-parity 4dp PASS,**0/5 可晋级**:全局最好 = T4×MHC-ZH **+0.0067 acc / +0.0052 mF1**(4.5× 低于 +0.030 门、落 ±0.014 带内、mF1 非 3/3)。机制层三个 door-closer + 一条新结构事实:**T1 退化**(与部署投票在 HateMM 215/215、ZH 149/149 上预测完全相同,独立 float64 numpy 对照复核)⇒ **local class prior 在 cone-collapsed 空间里与检索信号是同一个统计量**;**T2a 惰性**(hubness r(x) IQR ~1e-4,无动态范围);**T3 惰性且信息量大**(1-D 剔除精确到残差 ≤8.6e-9,却令检索的 length 组织 **Δρ ≤ 0.004 于 9/9 cell**、9/9 cell 零预测变化)⇒ **length 组织不由任何单一线性方向承载**;**T2b 负**(把 cone 从 0.9999 打开到 0.5220,却把 length nuisance 轴 ρ 从 ~0.52 **抬到 ~0.87**,因 d>n 时 LW 收缩仅 0.00041–0.0027)。T2b/T4 是**首批真的够到 stable-core 错误的算子**(每 cell 修 1–5 个)却**至少同量地打坏原本正确项**(18 个 T2b/T4 cell 中 12 个净 ≤0)= F47/F66 的同一条 selection-lock 算术 ⇒ **eval-time vote-operator 轴 CLOSED as measured** | MECHFIX_PREGATE_2026-07-27.md · `110dff8`(ops sha256 `635c1312…c83fc8d`) |
| **KSWEEP** — 部署 kNN 投票的 top-k 全扫(用户提问:有没有试过**减小** k 来削邻域噪声) | $0 forensic(只重放已银行化、已 test-consumed 的 per-item 邻居表;零 GPU/SLURM/重训/新 test 推理,~40 s CPU),19/19 cell 4dp parity + EN ARM-V k=20 投票 bit-exact 复现 banked floor:**k=20 已在平台上或之上(6/6 arm),平台自 k≈10–15 起**;**小 k 有害且不是更锐的投票——它就是 1-NN**(k∈{1,2,3} 的预测向量在 **19/19 cell** 与 top-1 标签向量逐元素相同,有闭式证明:cos 降序下 3·s₀ ≥ 2·s₁+1·s₂ 恒成立),代价 −0.0157…−0.0388 acc;**用户前提在 HateMM 上结构性为假**:rank 11–20 已被 rank 权重压成惰性(k=10 时 5/6 cell 预测变化数 **0/215**,第 6 个 cell 仅 1 项;k=15 时 6/6 为 0)⇒ 无尾部噪声可削,ERRPAT 说的噪声在 **rank 1–5**、且是**标签本身错**。deployment-legal 读数(dev 选 k)为**无用到有害**(HateMM final −0.0140,ZH final −0.0157,pooled ZH −0.0179/−0.0233;dev 一旦离开 k=20 通常跳到 k=3);**per-seed oracle-k 上界最大仅 +0.0145**(不到门的一半)⇒ **轴 CLOSED 双向** | KSWEEP_RECORD.md · `d5d78ad`(`scripts/analysis/ksweep_OUT.json`) |

**T6.5-b user-gated 通道 + MNTP 三臂(F90–F93)**

| 路线 | 死因(一行) | 成本 | 判决 · commit |
|---|---|---|---|
| **CLAP general-audio**(F88 排名第 1 的 gated 天花板:HateMM FN1 speech-poor 视觉仇恨 +0.0326,唯一「由信号缺席定义」的通道) | **G0-cond gate KILL**(spec 冻结于任何 CLAP 权重下载**之前**)。binding best-of{k8,k16} Δacc = **−0.0009**(deployed_7168)/ **−0.0038**(strict_8960),4 cell × {k8,k16} 全局最大 **+0.0009** ⇒ ~44× 低于 +0.040 门,且**低于 F64/LAUD 自己的全局最大 +0.0041**;全部决策 CI 跨 0;context arm 随 k 单调退化(k64 到 −0.0193)= 纯冗余稀释。K-CLAP-1 四 cell 全 VALID(label-oracle accZA = 1.0000)⇒ 是真 null 非机器伪影。FN1 层读数 INCONCLUSIVE_NARROW(预声明按 KILL 行动):CLAP-alone 0.8411 CI[0.7640,0.9073] **输给**已被杀的 Whisper 块 0.8482 CI[0.7844,0.9053](C2 = −0.0071 vs 要求 ≥+0.05),对 Z_deployed(0.8937)的条件 ΔAUC = +0.0113 CI[−0.0283,+0.0533] 跨 0。机制:音频信号**是真的存在但已被 Z 携带**——ρ(CLAP 分, n_words) = **+0.4430**(p=3.2e-42),即 ERRPAT §4.3 认定的**产生 FN1 的那条 length 偏置**,不是解药。**音频轴至此在三个表征层级全闭**:F41 经典韵律(eGeMAPSv02 88-d,−0.0038 strict)/ F64 学习语音-ASR(Whisper-large-v3 2560-d,+0.0014/+0.0014)/ F90 学习通用音频语义(1024-d,−0.0009/−0.0038)。诚实裂缝(预声明为 underpowered-context-only,不渲染判决):≤1 词层(n=87、8 正)CLAP 胜 Whisper +0.1266,但 CI[−0.1169,+0.4300] | 0.78 GB 下载 + ~1.7 h **CPU**(job 13647,无 gres)+ 681 s gate = **0 GPU-h** | CLAP_GATE_RECORD.md · `eee862c`(spec `6c8929d`) |
| **Molmo2-8B 编码器交换**(allenai/Molmo2-8B = Qwen3-8B LLM + SigLIP2-so400m-patch14-384;2025 代、video-native,选在编码器身份**唯一转换过**的 HateMM 上) | **KILL,且方向信息量最大。** 对最强同路径 floor(LoRA-curric)**双协议双指标皆低**:val-sel **−0.0217 acc / −0.0249 mF1**(per-seed 符号 − − −),final **−0.0124 / −0.0151**(+ − −),对预声明门(≥+0.0200 双指标 3/3 双协议)**大幅未达且符号相反**;对 like-for-like frozen-Qwen 对照是 **TIE**(\|Δ\| ≤ 0.0068 ≈ 1–2 个 test 样本)⇒ **更好的 video-native 编码器 ≠ 对本任务更好的编码器**。几何解离才是结论:raw image kNN **0.7814/0.7689** vs floor 0.7256/0.7112、frozen-Qwen 0.7163/0.7014(**+0.0558 / +0.0651**,HateMM 有史以来最强 image 流),而 cone collapse **更糟**(top-1 cos 0.9881–0.9999 vs Qwen 0.9439–0.9686)、length nuisance 轴**原封不动**(ρ +0.9052 vs +0.9432/+0.9530)、raw Hadamard **退化**(acc 0.5628,PR 3.069)。**⇒ 第 9 个获认证 law-I 实例**(见本节末对账);编码器交换轴在**视觉侧 PARK**,若再开应在**文本侧**(Molmo2 把 raw text kNN 从 0.8233/0.8186 **降到** 0.8000)| 抽取 job 13648 **~18 min GPU ≈ 0.3 GPU-h**;probe job 13653 为 CPU-only($0) | MOLMO2_PROBE_RECORD.md · `3298e8e`(recon `c1d450c`/`997b227`,门在抽取前定死) |
| **MNTP S1 + S1b** — bidir readout 路线(S1 = LLM2Vec 全非 padding 位置 mean pool;S1b = 仅文本位置 pool,**按 token id 选**而非 span 算术) | **ZERO-training 关闭该路线。** S1:HateMM text **0.7477** = 恢复 **−0.1999**(比 F72 crater 本身 0.7570 还低),ZH 0.7051 = +0.3529 partial ⇒ 无数据集达 50% 门**且符号相反**,sign-consistency 条款触发。S1 的真正发现是 **stream collapse**:arm 内 mean per-item cos(text, img) = **0.9273–0.9404**(HateMM)/ 0.9316–0.9320(ZH),对照 causal 0.3027–0.3523——「text」向量是 img 向量的 ~0.93 近拷贝,因为 S1 跨度 ~82.5% 是视觉 token ⇒ ZH 的「部分恢复」是 **stream substitution 而非 readout 修复**。S1b 随后在**预声明的 collapse belt 上自我证伪**(bar < 0.60,实测 0.7566/0.7624 HateMM、0.7565/0.7538 ZH)——**尽管这次 accuracy 门单独会说 CONTINUE**(HateMM +0.2003 / ZH +0.2941 且同号);冒烟在 HateMM 上 text 行与 img 行 acc/mF1 **数值完全相同**(0.7664/0.7540)= substitution 的直证。机制:双向注意下每个文本 token 都注意全部 ~720 视觉 token ⇒ **排除视觉 position ≠ 排除视觉 information**,readout 无法撤销由拓扑造成的信息混合;三种 pooling 跨度(F72 EOS-tail / S1 全位置 / S1b 仅文本)全部远低于 causal floor,collapse 随跨度**单调**(0.31–0.35 → 0.76 → 0.93)。H1 强形式**第三次被反驳**(img 流在 bare mask flip 下反而略好 +0.0093/+0.0128,三次独立抽取 cosine 1.000000)| **1.691 GPU-h**(budget ~2.0) | MNTP_S1_RECORD.md · `4a87836`/`f15dabc`/`12e2f18` |
| **MNTP S2a** — 把**已发表的 McGill MNTP adapter** 移植到我方 merged Qwen2.5-VL trunk(零训练、零语料裁决、零 test-touch) | **STOP,但这是整场战役第一个真实的 bidir 信号。** HateMM text **0.7850** vs F72 bidir 0.7570 = **+0.0280 = +0.6006 crater recovery**,**首个越过冻结 50% 门(bar50 0.7804)的臂**;ZH 0.6923 vs 0.6282 = +0.0641 = +0.2941 partial,**两侧同号** ⇒ 权重适配做到了三种 readout 做不到的事,**支持 MNTP 方向**并佐证 S1b 的诊断(病灶在权重/拓扑而非 readout)。**STOP 由四条独立理由 overdetermined**(无单一门载重):(1) 预声明 collapse belt 触发(within-arm cos(text,img) 0.6494/0.6550 HateMM、0.6386/0.6433 ZH,均 ≥0.60 且规则是**无论 accuracy 如何**自我证伪);(2) **融合由相加转为相消**——causal 下 concat 比最好单流 **+0.0467**(HateMM)/+0.0128(ZH),S2a 下变成 **−0.0467 / −0.0256**,而部署系统**就是**一个融合 head;(3) 每个 S2a 数都低于自己的 causal floor(text −0.0187/−0.1538,img −0.0280/−0.0128,concat −0.1121/−0.1538);(4) KS-MNTP-3 不可能满足。机制:adapter 是拟合在 **Qwen2.5-7B-Instruct 权重点**上的低秩 delta,而 VL trunk 已漂移 ⇒ 在新权重点上是**大而钝的扰动**(mean per-item cos(S2a, plain-bidir) 仅 0.3639/0.3076)。**被证伪的是零训练移植捷径,不是 MNTP 假说**;唯一存活的活假说 = **S2b 在我方权重点自训 MNTP,卡在用户语料裁决**。外部 codex gate 再次救臂:PEFT 键深一层会**静默加载 0 权重**(S2a 会伪装成 F72 重跑),以及 suffix 匹配会绑上 292 模块含 96 个视觉塔模块(修后精确 196 = 28×7,零视觉) | **1.006 GPU-h**(budget ~1.0;S1+S1b+S2a 合计 2.697) | MNTP_S1_RECORD.md §6d/§6e · `0663ab7`(修正案先于分叉提交)/ `b328dc9` |

**T6.5-c 关系型 / membank pregate 链(F95–F98)—— LITSWEEP-6 的 accuracy 菜单 $0 下 0-for-3**

> 计数口径:**0-for-3 指 LITSWEEP-6 自己的菜单**(F96 restrans / F97 VGA·VNQ / F98 aggnet);**F95 是这条链的
> 前置 cell**(它冻结的 `mechnov_pairverify.py` 被后三条 sha256-assert 复用),列在此处是为了让链条完整,
> **不计入那个 3**。

> 三条 lane 的文献扫记录已 commit:`LITSWEEP6_MEMBANK.md`(`62efd82`,5 个排序候选,**全部 $0/CPU pregate、完整版 0 GPU-h ——本战役首次**)、`LITSWEEP6_PARADIGM.md`(`49e15ec`)、`LITSWEEP6_RELGEN.md`(`f62e777`)。以下四个 cell 全部:CPU ≤8 线程、**零 GPU / SLURM / Modal / 任何部署臂的训练**、**test-split 接触 NONE**(仅 train split;dev_seen/test_seen 从未被任何脚本打开),且逐条 sha256-assert 复用 F89 冻结的 `mechfix_ops.py` 与 F95 冻结的 `mechnov_pairverify.py`。

| 路线 | 死因(一行) | 判决 · commit |
|---|---|---|
| **MECHNOV pair-verify** — 用**训练过的 pair verifier** 取代部署 kNN **投票**(检索从「判决」降级为「提名」;n 个 item 标签 → ~n² 个 pair 标签) | **KILL,且是 SPLIT VERDICT ——两半都载重。CONTROL-1 以 4.3–8.8× 通过**(18/18 cell、5/5 fold 符号):fused pair-AUC HateMM cosine 0.5843 → MLP **0.7753**(+0.1910)、ZH 0.5123 → **0.7748**(+0.2625)、EN 0.5057 → **0.7009**(+0.1952)⇒ 关系型 n→n² 监督**确实**买到更好的关系打分器,**不是 cosine 的再推导**(且顺带给部署度量定价:ZH/EN 上部署检索 cosine 自身的 pair-AUC **距随机不到 0.02**)。**CONTROL-2 端到端 36 个 cell 无一通过(0/36)**:primary HateMM 0.8441→0.8401(−0.0040)、ZH 0.8480→0.8014(−0.0466)、EN 0.7796→0.7650(−0.0146);36 个 cell 中仅 3 个 5-fold 均值为正(最大 +0.0094),全在次要空间/聚合且全部未过 +0.010 门。**两条实测死因**:(i) **被丢掉的聚合本来在干活**——control-2b 用 cosine 跑**同一形状**规则,形状本身先付 −0.0417/−0.0293/−0.0437,verifier 再赚回 +0.0377/−0.0173/+0.0291,**赚回的少于形状毁掉的**;(ii) **更好的关系 ≠ 更好的决策**——verification 确实**够到**了 ERRPAT 诊断为不可达的 36.7–54.6% 错误(F89 各臂只够到 0–5 个),但 fix/break 兑换率 0.9474/0.5345/0.8596,全 battery 上限 1.1667,**无一 cell 到 1.2**;够到量 **10×** 增长而兑换率纹丝不动 | MECHNOV_PAIRVERIFY_PREGATE.md · `0261b82`(冻结臂 sha256 `77b0defd…b7240d`) |
| **RESTRANS(membank-C1)** — 去偏**投票所搬运的标签场**(而非几何):保持检索、k=20、权重 [20..1]、阈值、键空间**完全不变**,只把 s_i = 2·lab_i−1 换成残差 r_i = s_i − (2·p̂_i−1),p̂_i = P(hate \| bank 项转写量),仅用 fitting fold 拟合 | **KILL,且 kill 是机制性的——退化对照(bar 3,预声明为 KILL 非 caveat)直接触发。** 21/21 cell 全负:primary(fused)HateMM B-a **−0.0188**(兑换率 0.4167,10 修 24 坏)、ZH **−0.0863**(0.3243)、EN **−0.1002**(0.4860);全 battery 最好 C1 数 = **−0.0013**(HateMM×img)。决定性一步:把 p̂_i 换成它自己的 **bank 均值**(= 纯全局阈值移动,一条三库皆已实测死的杠杆),两者预测在 **95.03% / 97.75% / 99.45%** 的 item 上一致 ⇒ **C1 是穿着 item-level 外衣的阈值移动**;闭式解释:cone-collapsed 空间里 cosine 近似均匀时 v_res = v_dep − (2·p̂−1) **精确成立**,而 item 项的离散度比常数项小 20–200×。C1 **够到了正确的人群**(改变的决策 34/34、98/98、158/159 落在「最近同类 bank 项 rank ≤5」的病理群)却**按 2.1–3.1× 打坏它** ⇒ LITSWEEP6 定律 (iii) 的第十个数据点:**够到病理不是难点**。机制层唯一值得留下的一句:**CP1(长度-条件类先验)是 HateMM 专属事实**——ρ(转写量, gold) = **+0.2842**(p=2.74e-15,HateMM)/ **−0.1152**(p=0.00553,ZH,**符号相反**)/ **−0.0050**(p=0.906,EN,**无**)⇒ 任何 C2 prereg 都不得再用 p̂ 做放置判据 | RESTRANS_PREGATE_RECORD.md · `bf6d03b`(冻结脚本 sha256 `99a770cd…2531`) |
| **VGA / VNQ(relgen-C1/C2)** — C1:用 verifier profile 在「部署投票 ↔ F95 裁决**不一致**」的 item 上做 per-item **裁决门**(在一致项上按构造是 no-op,故 F95 的形状代价被结构性归零);C2:把同一 profile 读成**选择性预测**风险排序(AUGRC) | **两个都 KILL,6 条冻结门中 5 条判死,决定性的是 K-VGA-3。** K-VGA-1 FAIL 0/3(primary verifier 全局最好 **+0.0108**,差 3×);K-VGA-2 FAIL(p = **0.8706 / 0.5174 / 0.9751**)。**K-VGA-3(new-signal 对照,mandatory)FIRES**:只用 **F47 族特征**(投票 margin、purity、子投票,**无 verifier**)的门在**三库全部**打败 verifier 门 —— **+0.0269**(HateMM,p=0.0050,fold 符号 +++++)/ **+0.0104**(ZH,p=0.0050)/ **+0.0182**(EN,p=0.0100),分别高出 +0.0161/+0.0104/+0.0164,且**最悬殊处正是 verifier 统计学上已死的两个库** ⇒ 「genuinely new information source」这一**预注册可证伪主张被证伪**,该轴按测量关闭而非「未打开」。C2/VNQ 输给**最廉价的基线**:AUGRC(越低越好)HateMM 0.0458 vs kNN-UE 0.0429 vs **免费的 vote margin 0.0465**;ZH 0.0417 / 0.0393 / **0.0384**;EN 0.0810 / 0.0758 / **0.0696** ⇒ K-VNQ-1 0/3、K-VNQ-2 1/3 且那 1 个未过 fold 门。**唯一新正数据点(记录但明确不晋级)**:disagreement set 上**确实存在**置换验证过的门信号,但它由 F47 特征承载、三库全部低于 +0.030 门、且门的是一个**未门控时三库皆净负**的裁决器(族 oracle 上限仅 +0.0726/+0.0535/+0.0893)⇒ **analysis datum, not a lever**。**关系型资产至此结算为 analysis-grade only**(三次转换尝试 F95 替换投票 / C1 门控替换 / C2 读作风险,**全负**),LITSWEEP6_RELGEN §5 的预承诺兑现 | VGA_PREGATE_RECORD.md · `db2eae8`(冻结脚本 sha256 `a3a41ae7…7ce56` / `ea37c57b…4f4e34`) |
| **AGGNET(membank-C3)** — **学习型聚合 profile 网络**:检索、键空间、k=20、候选集、阈值、标签场**全部不变**,只把固定 rank 权重 [20..1] 换成 per-query 的 g_θ(邻域 profile)(1316 参数,**部署锚定初始化**使 epoch 0 与 floor 逐位相同,λ 由内层 CV 选、λ→∞ 精确退回部署规则) | **KILL:决定性门差 2× 以上,且两条 mandatory 退化对照在唯一为正的库上同时触发 ⇒ conditional-aggregation 族 CLOSED。** 本 cell 的载重数字是**覆盖率**:非负权重能翻转 sign(v) 的前提是 top-20 类混合,占 0.8683/0.8290/0.9709,**可达的部署错误 111/116、88/88、120/121 = 96–100%**,**族 oracle Δacc = +0.1492/+0.1520/+0.2186**(是 F95/VGA 裁决门 oracle 的 2–4×、F94 per-seed oracle-k 的 10–15×)——**C3 带着本族史上最大的天花板入场**。实测:primary **+0.0134**(HateMM,fold 符号 −0+++,34 项改变中 22 修 12 坏,兑换率 1.8333)/ **−0.0069**(ZH)/ **+0.0000**(EN);45 个 cell **最大 +0.0134,0 个到 +0.030**。退化对照:**DEG-A**(与裸全局阈值移动的一致率)**0.9570**(HateMM)/ 0.9508(EN);**DEG-B**(与 F94 网格里单个固定 k 的一致率)**0.9610**(k=15,HateMM)/ **0.9964**(k=20,EN,即 C3 在那里**就是**部署规则);而 **THRESH_best 单独就有 +0.0188**(比 C3 还高,不用网络不用 profile),**DIRECT_logit**(同 profile 的无约束 logistic)**恰为 +0.0134** @ 0.9516 一致率 ⇒ 聚合**形式**没贡献任何两个退化孪生没给的东西。**因此 kill 是机制性而非预算性的**:标准替代解释「算子够不到错误」被 C3 消除,函数类近乎 profile 分类器全类(自检臂 B),而它仍收敛到**两条已关闭的杠杆**;**族内 delivery 与 ceiling 无关**(F94 上限 +0.0145→交付 −0.0140…+0.0041;F95/VGA 上限 +0.0726/+0.0535/+0.0893→交付 +0.0269/+0.0104/+0.0182;C3 上限 +0.1492/+0.1520/+0.2186→交付 +0.0134/−0.0069/+0.0000)。**F96 的勘误级锐化(须随该数一起走)**:F96 的「死亲戚」D1 在 HateMM 上给出本战役首批过 1.2 兑换率(+0.0215/1.8889 fused、+0.0282/2.2353 text),F96 归因于**长度协变量**;本记录里把协变量**拿掉**的同一算子(THRESH_best)测得 +0.0188/1.5833 与 +0.0242/1.7200 ⇒ 协变量只值 +0.0027/+0.0040(2–3 个样本),**D1 的正数 ~87% 是裸阈值移动**,该杠杆在部署 head 空间的 test 上仍**实测死**(+0.0000/+0.0016) | AGGNET_PREGATE_RECORD.md · `fa1e3b3`(冻结脚本 sha256 `8e95c2fc…e8a9`) |

**读表须知 —— proxy floor 与 T1/T5 锚数的对账(避免被误读成矛盾):** 本节多处引用的 HateMM「同路径 floor」
**0.8775 / 0.8715(val-sel)与 0.8760 / 0.8699(final)** 是 **errpat CPU proxy 头**的读数(6 个 floor head
ckpt 已按 F78 disk-删,proxy 是同一 `run_rac.py` 命令在同一 banked 特征缓存上的重建);T1/T5 的 GPU 锚数是
**0.8775 / 0.8711(val-sel)与 0.8791 / 0.8726(final)**。两者的差**恰是 F88 记录的 proxy-vs-floor 偏移**:
val-sel **+0.0000 / +0.0004**(4dp 上精确)、final **−0.0031 / −0.0027**(0.67 个 test 样本/seed,残差 =
CUDA-vs-CPU dropout RNG)。**所有 round-8 的 Δ 都是同路径配对量**(proxy 臂 vs proxy floor),偏移在 Δ 中抵消;
**T1/T5 的锚数未被任何 round-8 cell 触碰**。绑定纪律(F87 → F88 → F91 一脉):**CPU 训练的臂只能配 CPU 训练的
floor**,F91 因此在同一 job 里重跑 arm B/C 而不是引用银行化数字,并在 4/4 cell 上复现 proxy 到 4dp。

**组织性事实(round-8 的科学产出,四次独立测量收敛):RANKING QUALITY ≫ DECISION QUALITY。**
(i) **F95** —— verifier 在关系层比 cosine 高 **+0.13 到 +0.27 pair-AUC**(18/18 cell,4.3–8.8× 过门),端到端 **0/36**;
(ii) **W4 temporal**(`EVAL_temporal_memory_W4.md`)—— EN 时间切分 **ROC 0.8484** 高于随机切分参照 0.7175,macro-F1 却
**掉 −0.084**;(iii) **F88 ERRPAT** —— 正确类比项在**中位 rank ~1.5** 却被压倒,误差是 ~90% seed-invariant 的
**confident inversion**;(iv) **F50/F48** —— dev AUC **0.898** 而「unconvertible」。**我们的系统排序远好于它决策**;
而每一条已死方向都在试图用「对同一批分数的更好的**统一决策规则**」去合上这个缺口——vote 算子(F89)、k(F94)、
阈值(F88)、损失(F75)、verifier(F95)、门(F97)——**该轴现已从六个方向关闭**。LITSWEEP6-PARADIGM 因此不再推荐
第七条规则,而推荐**换输出对象**(三路认证输出 + 策略化 operating point),而这**先需要用户对交付物的裁决**(见
`DECISION_MEMO_pending.md` 现行裁决单 S1)。

**law-I 计数对账(一次说清,全文按此口径):**
- **F63 = 第 7 例**(LP,ZH oracle 头空间 +0.1026 未兑现)、**F65 = 第 8 例**(vision-unfreeze,EN image 流 MOVED、head 兑零)。
- **F87 的候选第 9 例(MokA image 流 AMBIGUOUS)明确未获认证**,该判定**不变**(T6.4-b 原文保留)。
- **F91(Molmo2)= 第 9 个获认证实例**,且是**迄今最干净**的一例:此前的 law-I 数据是「image 流移动而兑现为零」,
  这里是 **raw image 流真的变好**(+0.0558 / +0.0651,HateMM 有史以来最强 image 流)**而兑现为负**(−0.0217 val-sel /
  −0.0124 final)。**现行计数 = 9。**
- **F95 自述为「迄今最锐利的 law-I 实例」——本汇编不把它计入编号实例**(它是 train-split raw-space 的 $0 事后诊断,
  不促成任何 arm、不动任何部署读数),而记为**该定律最锐利的一次测量**:law-I 首次在**决策所消费的那个量本身**上
  双侧测得(关系分 +0.13–0.27 pooled / +0.16–0.23 within-query pair-AUC,端到端 0/36),中间隔着 F66 的 selection-lock。
  **后续文档不得据此把计数写成 10。**

**pillar-④ 更正(F88 §6.5,`ad56a62`,载重,已传播到全部引用点):** EN「删 2 条人工标记噪声记忆」的正结果是
**single-seed**。banked top-60 邻居表上的精确多 seed replay(未编辑重放逐 seed 复现 floor 到 <1e-12):seed 0
**+0.0124**(0.8074534161490683 → 0.8198757763975155,macro-F1 0.7625707625707625 → 0.7748468920287408,与
`DEMO_memory_editing.md:52` 逐位一致),**seed 1/2/3 各 0 次投票翻转**,**4-seed 均值 +0.0031**;seed 0 翻的两项首次被
点名(`cYQyH7hbNnw`、`xqilG4oMvvI`),**均为 C8 噪声带的低 margin FP,都不是硬错误**。14-id 规则表严格更强
(**+0.0093 acc / +0.0089 mF1,3/4 seed 正,6 修 0 坏**),但 3× 低于 +0.030 门、落 ±0.014 带内、且已 test-consumed;
合法的 dev 侧 pregate **算术上不可能**(dev n=80 ⇒ 1 项 = 0.0125,无法解析 +0.009 效应)。**统一措辞(所有引用点):
"human-in-the-loop capability demonstration, single-seed; not an accuracy claim."** 附带两条账本更正:F78 的
「$0 curation 前提为假」**对 EN 的纯删除编辑不成立**(4 seed 的 top-60 邻居表已银行化,支持精确 $0 多 seed 重放;
F78 的限制只对 bank **新增**、键空间变更与重训成立),以及 HateMM align head 在 8 CPU 上 **52 s** 端到端训完 ⇒
head 侧诊断/消融/校准变体现在都是 CPU-分钟级、可全 3-seed 纪律执行(**绑定 caveat:CPU-trained 臂只能配 CPU-trained
floor**,final-epoch 上 −0.0031 的路径差不可忽略)。

**来源(T6.5):** `refine-logs/{ERRPAT_HateMM_2026-07-26, ERRPAT_MHC-EN_2026-07-26, ERRPAT_MHC-ZH_2026-07-26,
MECHFIX_PREGATE_2026-07-27, KSWEEP_RECORD, MECHNOV_PAIRVERIFY_PREGATE, CLAP_GATE_RECORD, MOLMO2_PROBE_RECORD,
MOLMO2_FORENSIC_RECON, MNTP_S1_RECORD, MNTP_FORENSIC_RECON, RESTRANS_PREGATE_RECORD, VGA_PREGATE_RECORD,
AGGNET_PREGATE_RECORD, LITSWEEP6_{MEMBANK,PARADIGM,RELGEN}}.md` + 机器可读输出 `scripts/analysis/{errpat_*,
mechfix_*, ksweep_OUT, mechnov_pairverify_*, restrans_pregate_*, vga_pregate_*, aggnet_*}.json`。findings
**F88–F98**(`state/findings.jsonl`)。**不入负结果 ordinal、不铸总数(逐轮框架,同文末张力清单 #9);law-I 实例
计数 = 9。**

---

## 骨架段 — 一页纸论文章节骨架

每节标注三终局选项 **(a) 接受现状定稿 / (b) 闭源 API 攻定位 / (c) 换方法族** 下需要改什么。

### Intro(定位)
- **问题:** hateful **video** detection;把 RGCL/RA-HMD(原 hateful-meme)适配到视频。
- **贡献定位(四支柱 + MLLM 三角色):** ①检索对比 + kNN 记忆核心 ②可更新记忆 + 时间协议
  ③共识去噪(修复机制,ZH-scoped)④可审计/可编辑档案记忆;MLLM 挣得 **encoder + 定位打分器 +
  guard-rail/审计** 三角色,**主表 accuracy 角色被 13 路线全 campaign 中的 11 条主表路线证伪**
  (定位赛道 P6/P10-b/P10-c/P11 另计;方法学负结果贡献)。
- **(a):** 强调「四支柱能力 + 定位 modest-plus + 强负结果链」,**不主张 substantial main-table**。
  **(b):** Intro 保留定位子目标为「开放问题:闭源能否 0.5755→0.60+」。**(c):** 重写 Intro 把 headline
  从主表 accuracy 转到「唯一 scale 起作用的定位赛道」或换编码器族的 ZH 主表。

### 方法(四支柱 + MLLM 三角色 + 一条明确非角色)
- 四支柱骨架 + MLLM 三角色框架(OPTION_KITS A.2);明确非角色 = 13 路线 ruled-out map + 两条方法学定论。
- **(a):** 照 OPTION_KITS A.2 直接落地,零改动。**(b):** 方法章加一小节 P10-c 闭源定位放大(须先获数据外发
  批准)。**(c):** 方法章重构——可训练-MLLM + in-context 检索,或换 CN 文本塔检索族;放弃部分现有四支柱资产。

### 实验(= T1 主表 + T2 定位 + T3 能力)
- 正文:T1.2 同场 MoRE 三库全胜 + T1.1 多 seed 主表;T2 定位(P6→P10-b);T3 能力(swap/temporal/edit)。
- 附录:T2.2 校准 leaderboard(14 比较)+ EXPLORATORY 天花板;selection-robustness 五规则 + sha1 双口径。
- **(a):** T1–T3 全部照录,ZH 口径取「val-选点主表 + final-epoch 附录并排」(须先锁 headline 协议,
  见待拍板)。**(b):** T2.1 增一行闭源候选 test(仅晋级后),T2.2 增 round-4 闭源比较列。
  **(c):** T1 主表可能整体重做(新方法族基座),T3 swap/temporal 若资产可迁移则保留。

### 分析(= T4 + 机制结论五条 + 局限)
- T4 反结果表 + 两条方法学定论;下列五条机制结论作分析主线。
- **机制结论五条(TERMINUS §3 + 各前沿):**
  1. **语义能力 ⊥ 决策变量**(最统一失败形状)。
  2. **calibration 随 scale 涨、selectivity 不涨**(P2b/P2c 规模梯);唯一例外 = 定位赛道(scale 起作用)。
  3. **过 no-head probe 是必要非充分**(P3-HateMM / P8-EN)。
  4. **P9b head↔memory 再分配非净增益**(rgcl 把精度在 LMM 头与 kNN 读出间 ±1.8pt 对调)。
  5. **ZH 独立瓶颈 = 冻结 English-centric 编码器,非 MLLM**(P8c:中文 byte-fragment 97% 截断)。
- **(a):** 五条全用,局限=定位仍 modest(<0.60)、EN 未达 0.85(近天花板归因)、~150 样本 seed 噪声支配。
  **(b):** 结论 2 的「定位例外」升级为主线,局限增「闭源不可复现、不可写进开源 pipeline、数据外发合规」。
  **(c):** 结论 5(换编码器)或结论 4(容量争夺)升为立项动机;局限=换族上限高但周期/不确定性最大,
  且 P9/P9b 警示决策级 LMM 也只匹配不超现有 LoRA。

### 局限(全选项通用)
- ~150 样本 test + 78 样本 dev:seed/选点噪声支配 ≤2 点增益(n=5 配对 MDE ~0.04–0.05 F1)。
- EN ≈0.78–0.80 未达 0.85(近天花板:同场 MoRE 仅 0.69–0.72;CRAVE 发表 79.81 F1 为该 split 场上最高)。
- ZH 双口径悬置(val-选点 0.827 不过 / final-epoch 0.8537 过;因过线才换口径 = rule-shopping 风险)。
- 定位 modest-plus(0.5755 < 0.60);定位 baseline(MultiHateLoc/LELA/TANDEM)codeless,红线内无法同场。
- 措辞红线:只说 **span-free**,不说 first/annotation-free/dense-supervision-free。

---

## 数字矛盾 / 张力清单(汇编时发现)

1. **【张力,非硬矛盾】ZH「我方最优」口径不一致。** T1.2 MoRE 对比引 **LoRA-SFT 0.8322/0.8023**(ITERATION_LOG
   单配置 val-selected);多 seed 审计(exp-archive-knn-seeds `ebc1988`,更晚)为 **val-sel 0.8268±0.0266 /
   final-epoch 0.8537±0.0120**,且 0.8322 被 MORNING §5 kill #6 明标为「单 seed 口径,已撤回」。**以更晚
   commit(多 seed 审计)为准**;T1.2 的 0.8322 保留为「MoRE 同场对比所用的单配置点估」,已加脚注 ‡。
2. **【张力】EN「我方最优」同源问题。** T1.2 EN frozen-Qwen **0.7888/0.7378**(单配置 val-sel)与 T1.1 多 seed
   (val-sel floor 0.7702 / archive 0.7935;final 0.7888/0.7826)不同源;F1 0.7378 与 final-ep F1 0.7488 亦有
   ~1pt 差(不同配置/选点)。**以 T1.1 多 seed 为 headline**,MoRE Δ 附点估脚注。
3. **【已按更晚 commit 消解】P10-b 状态。** CAMPAIGN_mllm_method_role.md(`78ab700`,较早)与 MORNING §9 仍把
   P10-b 记为「IN FLIGHT,CPU 行 7B fuse×lex 0.5752」;TERMINUS(`2781349`)/ EXP_p10(`74f0eac`,更晚)已落定
   **P10-b 72B A-fuse test 0.5755 MODEST**。**本汇编一律采更晚 commit 的 0.5755**(T2)。
4. **【非矛盾,granularity 澄清】HateClipSeg memory baseline 两个值。** T2.1 主链用 **K=30 knn_hatemm_subclip
   0.5140**(与 P6/P10-b 同粒度 head-to-head 的配对基线);EVAL 文档另有 **K=4 subclip 0.5259**(唯一显著的
   memory cell,存在性证明)。两者不同 window 粒度,非冲突;主链取 0.5140,0.5259 作 memory 最强 cell 附注。
5. **【口径澄清】HateMM 「0.870」出现在两处不同 test-n。** T1.1「frozen-Qwen RGCL 0.870/0.861」为 clean n=215
   (MoRE 同场口径);P9「trained-RGCL val-sel 0.870 / final 0.8605」为 P9 匹配 test 口径。数值 0.870 一致
   (val-sel),final-epoch 侧 P9 报 0.8605;已分行标注,非矛盾。
6. **【口径提醒】consensus ZH 与 archive-kNN ZH 编码器不同。** consensus 行(0.8107/0.8175)是 **frozen-CLIP**
   base;archive-kNN ZH 主栈(0.8268/0.8537)是 **LoRA-Qwen**。T1.1 已用 † 标注不可直接同格并比;consensus 仅
   作机制/robustness 行。
7. **【计数粒度澄清,非矛盾】「11 条主表」vs「13 条总数」。** 两个数**同时正确、指不同粒度**,勿混淆:
   **13 = campaign 全部预注册路线**(T4 本表 13 行 / CAMPAIGN_mllm_method_role.md「13 条全结题」/ DRAFT §1
   "thirteen-route"),其中定位赛道 = **P6 + P10-b/P10-c/P11**(3 行),其余 10 行瞄准主表 accuracy。
   **11 = DRAFT §1 记的主表路线细粒度计数**(把 P9/P9b 分列),与 T4 的 10 主表行**同指一组结果**。两处终态一致:
   主表 accuracy 角色全数证伪。**易错点:早期文本「主表被 12/13 路线证伪」暗示全部路线都瞄准主表——错;
   定位赛道(P6/P10/P11)不属主表路线**,已在 T4 表头与骨架 Intro/方法段更正。
8. **【硬勘误,已修正 2026-07-11】HateMM frozen-CLIP RGCL floor 数字错误。** 原记 **0.8732 acc / 0.8686 mF1**,
   并错引「job 12132, ep24」。溯源(2026-07-11):**0.8732 实为该日志的 Val_Retrieval ROC-AUC**
   (`rgcl_HateMM_openai_clip-vit-large-patch14-336_HF_1035814.trainlog` ep9/ep20,line 117/217),被误当成
   test acc 抄入;**0.8686 在 1035814 与 1029175 两份日志中均无对应读数(幽灵数)**;错引的 **job 12132 实为一个
   MHC seg 训练作业**(`slurm/logs/mhc_train_seg_12132.out`),与 HateMM CLIP 无关。**正确读数 = val-selected ep24
   的 test acc 0.8279 / macro-F1 0.8172**(n=215,`1035814.trainlog:257-259`)。**传播链:**
   EVAL_localization_hatemm.md:105 →(本表 T1.1 L35 + DRAFT_experiments_chapter.md L104 表/L151 正文 +
   MORNING_REPORT.md L13)→ 用户 target-loop 注册表 `TARGET_STATE.json`(exact_baselines.HateMM,L62-63)。
   **修正:上述我方 4 处文档已全部改为 0.8279/0.8172 并更正出处(见本次勘误 commit);TARGET_STATE.json 属用户 loop
   状态,未改,需用户自行同步 L62-63 及其 hard_target/audit_note。** 附:frozen-Qwen 0.870/0.861 经同轮交叉核对
   (`1029175` ep28 val-selected test 0.8698/0.8606 ≈ 0.870/0.861)确认无误,保留不动。
9. **【张力,非硬矛盾;round-4 集成时发现】负结果 ordinal 两套计数不对齐,不铸造总数。** 两个 committed loop-state
   文件对「第 N 个 pre-registered negative」用**不同**计数轴:(i)`state/directions_tried.json` 的 epitaph ordinal
   把 round-2 冲刺记为 15th–22nd,**终于 B4=22nd**(A-line 15 / C1 16 / C3-target 17 / SAV 18 / C3-nontarget 19 /
   B1 20 / B2 21 / B4 22),round-3/4 条目**无 ordinal**;(ii)`state/findings.jsonl` 把 round-4 的 **F47(router)记为
   「22nd」、F50(FA)记为「23rd」pre-registered negative**——该 ordinal 从 round-2 *终结* 计数(memory「21 pre-reg
   negatives」,B2=21st)+1/+2 续接,**未计 B4、亦未计 round-3 五个实测负结果(S2S/CTF/APX/W2-A/GIR)**,故与 (i) 在
   「22nd」处**冲突且低计**。`directions_tried.dead` 数组实有 **34 条**(含 campaign P-路线 + TARC + auto-repair +
   round-2/3/4 + recon/triage companion),wave-5 provenance 的「23 dead」是**近似口径**、非数组长度。**处置(本汇编):**
   载重计数轴仍是 **T4 = 13 campaign 路线(不变)**;round-2/3/4 走**逐轮记账**(T5.1 #15–22 / T5.2 六向 / T5.3
   router+MJ+FA),**不把 findings 的 22nd/23rd ordinal 传入论文**(会与 B4=#22 冲突并低计 round-3),亦**不铸造有争议的
   累计总数**。论文 §7 与本节均采逐轮框架;此 ordinal 差异记录于此,不静默修改任一 loop-state 文件。
   **【2026-07-18 补充,F53 集成时】** round-4 line-A 的 **B4-EN 正式测量关闭**在 `state/findings.jsonl` F53 记录里被记为
   **「24th pre-registered negative」**(该 ordinal 沿 findings 轴 +1 续接,自认 "ordinal tension noted");但 `LORA_HATEMM_VERDICT_REVIEW.md`
   §3/§5 与本汇编一致把 EN 闭合记为 **B4 = 第 22 条**(round-by-round 轴,现从预-GPU 升级为实测,同一 cell 同一 ordinal)。
   **论文不采 findings 的「24th」**(会与 B4=#22 冲突);LoRA-HateMM 的 **HateMM cell 是性能正例、不入负结果账**(不占 ordinal)。
   处置同上:逐轮框架,不铸造总数,不改 loop-state 文件。
   **【2026-07-18 补充2,F55/F56 round-4 closing 集成时】** round-4 收尾两项同样**不铸造总数**:(i)**premise-(d)**
   (`PREMISE_D_GATE_RECORD.md` F55,`6e6061b`)是 **$0-gate 负结果**,记录 §5 明记为 **non-binding executor label**
   (binding close = orchestrator's),自认「第 6 个 better-signal/no-conversion 实例」;它是 FA(F50)carve-out 的 $0
   follow-on,与 GIR/CTF/APX(round-3 $0 gates)同类,**采逐轮框架(T5.4(b))、不分配 grand-total ordinal**。(ii)**cand-2**
   (`CAND2_VERDICT_REVIEW.md` F56,`546acc5`)是 **tie / coupling probe**,非 clean negative 亦非 clean positive:ZH
   K-C2-2 tie 双协议、HateMM K-C2-2 pass 仅 val-sel(rep2 后 pooled weakly-hardened,5/6 sign,per-draw 3/3 gate not met)、no kill fired——**held pending D7 sub-ruling
   (PUR-4),不入负结果账、不占 ordinal、不并入主表**。两项均记于 T5.4,遵循逐轮框架,不改任一 loop-state 文件。

---

*(本文件为纯汇编,不含新实验;所有数字均为已 commit 结果的转录,出处见各表 commit 列与文首 document-level
commit 清单。三终局选项的具体第一周动作草案见 `OPTION_KITS_terminus.md`。)*

---

## PENDING-USER-RULING addendum(2026-07-14,扩展 2026-07-18):LoRA-encoder family(B3/B4/LoRA-HateMM)

> **本节 append 于 2026-07-14、round-4 line-A 后扩展于 2026-07-18,与主表 T1–T4 严格隔离,遵循与文首相同的
> 纯转录纪律**(不跑实验、不提交 SLURM、不重算任何数)。**本节每一行都 pending 用户裁决**(见节末 banner),
> **不进主表、不改主表任何数、不移动任何行到 T1–T4**。数字均转录自命名源文档(numeric-provenance discipline):
> B3 实测数 = `refine-logs/B3_VERDICT_REVIEW.md`(job 13150,独立判决复核)+ `research-wiki/experiments/exp-lora-zh-b3.md`
> §7a/§7b/r1;**LoRA-HateMM 实测数(F53)= `refine-logs/LORA_HATEMM_VERDICT_REVIEW.md`(commit `6b8f634`,job 13235,
> 独立 0-context 判决复核)+ `research-wiki/experiments/exp-lora-hatemm.md`**;B4 EN 现已随 LoRA-HateMM 正式测量
> (原 `refine-logs/B4_FORENSIC_RECON.md` 预-GPU 关闭,now `LORA_HATEMM_VERDICT_REVIEW.md` §2.2 实测);
> 选项账目 = `research-wiki/TERMINUS_round2_mllm_plus3.md` §6–§7。

### PUR-1 — B3 配对表:LoRA-Qwen 编码器 vs frozen-CLIP,MHC-ZH(3 head-seed,job 13150 vs 13115,两协议并排)

同 runner、同 `--seed`、同 149 ZH test videos 的 head-level 配对读数。**LoRA 臂 = job 13150**(fresh,G-repro
bit-exact 复现 arcbase 12223-25,6/6 读数 4dp 零失配);**CLIP 臂 = B1 job 13115**(frozen-CLIP,既有日志,未重跑)。

**协议 (B) final-epoch(epoch 29 两臂):**

| seed | LoRA acc | CLIP acc | **Δacc** | LoRA mF1 | CLIP mF1 | **ΔmF1** |
|---|---|---|---|---|---|---|
| 0 | 0.8456 | 0.8054 | **+0.0402** | 0.8181 | 0.7706 | **+0.0475** |
| 1 | 0.8389 | 0.8054 | **+0.0335** | 0.8113 | 0.7542 | **+0.0571** |
| 2 | 0.8523 | 0.8322 | **+0.0201** | 0.8226 | 0.7913 | **+0.0313** |
| **mean** | 0.8456 | 0.8143 | **+0.0313** | 0.8173 | 0.7720 | **+0.0453** |

**协议 (A) val-selected(warmup≥5,max Val_Retrieval acc,roc tie-break):**

| seed | LoRA acc | CLIP acc | **Δacc** | LoRA mF1 | CLIP mF1 | **ΔmF1** |
|---|---|---|---|---|---|---|
| 0 | 0.8322 (ep20) | 0.8054 (ep29) | **+0.0268** | 0.8023 | 0.7706 | **+0.0317** |
| 1 | 0.8255 (ep26) | 0.8054 (ep28) | **+0.0201** | 0.7956 | 0.7579 | **+0.0377** |
| 2 | 0.8389 (ep19) | 0.8121 (ep25) | **+0.0268** | 0.8065 | 0.7742 | **+0.0323** |
| **mean** | 0.8322 | 0.8076 | **+0.0246** | 0.8015 | 0.7676 | **+0.0339** |

**判决(绑定语言,逐字 per `refine-logs/B3_PREREG_REVIEW.md` §2.2 — 不得升级):**
`final-epoch: PASS (MARGINAL); val-selected: FAIL`。
(final-epoch:mean Δacc +0.0313 ≥ +0.030 AND mean ΔmF1 +0.0453 ≥ +0.030 AND sign 3/3 ⇒ PASS,标注 MARGINAL;
val-selected:mean Δacc **+0.0246 < +0.030**,AND 规则在 acc 上失败 ⇒ FAIL,尽管 mean ΔmF1 +0.0339 与 sign 3/3
达标。)来源:`B3_VERDICT_REVIEW.md` §0/§3/§4 · job 13150。

**三条强制敏感度事实(脚注,`B3_VERDICT_REVIEW.md` §4a 要求全列):**
- **[SF1] 贴边。** mean Δacc **+0.0313 仅高出 +0.030 门 +0.0013(≈门的 4%)**——这就是整个 pass 的全部余量。
- **[SF2] 逐种子不均。** 逐种子 Δacc 跨度 **+0.0201 … +0.0402**;**seed2(+0.0201)本身低于逐种子 +0.030 门**;
  pass 靠 seed0/1 与 F1(+0.0453 干净过线),而非均匀的逐种子余量。
- **[SF3] 余量 ≪ 种子间散布。** +0.0013 的 acc 余量远小于种子间 Δacc 散布(0.0402 − 0.0201 = **0.0201**,
  ≈15× 余量)——即 acc pass 落在 head-seed 噪声内。**结构性 marginal**(非可辩掉的随机噪声):G-repro bit-exact
  ⇒ fresh 数不随 run 抖动,marginality 来自贴边 + 单一 CLIP 控制抽样(13115 一次)+ 单一 LoRA 编码器抽样
  (3 种子共享单缓存,只变下游 head),不建立 LoRA-SFT 训练种子方差。

### PUR-2 — LoRA-encoder 三数据集地图 + 解释行

| 数据集 | encoder-level LoRA 单元结果 | 关键数字 | 来源 |
|---|---|---|---|
| **HateMM** | **encoder-level PASS(双协议,solid)** — F53 | 3-种子配对 vs frozen-CLIP:**val-sel mean Δacc +0.0419 / ΔmF1 +0.0460**(3/3),**final-ep +0.0573 / +0.0682**(3/3);均远超 +0.030 门(≈9× B3 余量)。**注:P9 的 HateMM C3-knn −4.7 below floor 是 decision-level 另一 regime(r128/α256,joint-SFT+raw-kNN),与此 encoder-level 单元非同构**(两 regime 由相反 ZH 行为证伪:encoder-level ZH +0.031 vs decision-level ZH −2.2) | `LORA_HATEMM_VERDICT_REVIEW.md`(job 13235)· 见 PUR-3;decision-level = `EXP_p9_lmm_rgcl_video.md`(T4 行 10) |
| **MHC-EN** | **3-seed 实测 FAIL 双协议**(B4 随 LoRA-HateMM 正式测量,第 22 条) | mean vs frozen-CLIP:**val-sel Δacc −0.0021**(acc 2/3)/ **final-ep +0.0000**(acc 1/3),均 ≪ +0.030 门;seed0 anchor 逐位复现预-GPU 值(val-sel −0.0310 acc / −0.0197 F1);EN 上 LoRA 低于两个 frozen 编码器 | `LORA_HATEMM_VERDICT_REVIEW.md` §2.2 · `B4_FORENSIC_RECON.md` §(i)/(v) · `exp-lora-sft-encoder.md:21` |
| **MHC-ZH** | **B3 marginal pass** | `final-epoch: PASS (MARGINAL)`(mean Δacc +0.0313)/ `val-selected: FAIL`(mean Δacc +0.0246);见 PUR-1 | `exp-lora-zh-b3.md` r1 · `B3_VERDICT_REVIEW.md` |

**解释行(载重,round-4 line-A 后更新):** **encoder-level LoRA = 唯一跨 ≥2 库过线的单一杠杆,但受协议限定,且两次
过线经由不同模态。** 性能账(带协议限定词):**final-epoch 协议下,一个杠杆(encoder-level LoRA)在两库过 +0.03/+0.03
门 —— HateMM(+0.0573/+0.0682,solid)+ MHC-ZH(B3 +0.0313/+0.0453,marginal);val-selected 协议下同一杠杆仅
HateMM 过线**(ZH val-sel FAIL,78-dev 选点税)。这是 campaign 首个单一编码器杠杆跨 ≥2 库过线——但是**一个杠杆两种
机制,非单一机制**:分解(`B3_ZH_LORA_DECOMPOSITION.md` F45 / `LORA_HATEMM_VERDICT_REVIEW.md` §3.3 F53):
- **ZH:** frozen-Qwen 交换 **−0.0112**(B1 第 20 条,FAIL)vs LoRA **+0.0313** ⇒ ZH 增益全部来自 LoRA 任务/语言适配
  (text-borne,LoRA-specific);缓解 English-centric CLIP 文本塔处理中文 byte-fragment 的记录劣势(PMT:188,237)。
- **HateMM:** KS-2 honesty flag **未触发** —— final-ep LoRA 0.8698 ≥ frozen-Qwen 0.8682(+0.0015 acc),val-sel 落在
  0.014 种子带内 —— 即 **LoRA ≈ frozen-Qwen**;LoRA 只动语言 backbone(vision tower/projector 冻结)。零-GPU 逐流分解
  (`HATEMM_LORA_STREAM_DECOMP.md`,`51eb95b`,F58)实测:HateMM 的**决定性单流是 text**(text-only kNN AUC ≥ image-only,
  CLIP/frozen/LoRA 三编码器双 footing),image 流 strong 但 **swap-neutral**(LoRA 仅 +0.0045 train / +0.0062 dev,flat,
  未塌陷,不同于 MHC-EN 的 0.599);LoRA 把 text 流 sharpen(train-LOO 0.888→0.920)但**加 ≈0**(final +0.0015 acc /
  val-sel −0.0108),因为 **frozen swap 已把 HateMM 的 text 信号转成 Pareto**(frozen−CLIP +0.0558 acc)—— 无 boundary 可再动。
  ⇒ HateMM 过线是 **text-carried / frozen-swap-sufficient / LoRA-inherited**,非 LoRA-specific。
- **MHC-EN:** 549 样本 LoRA-SFT 退化编码器,落到两个 frozen floor 之下(label-limited + image 塌陷,F44)⇒ 任何编码器
  移动都不转换,fail 双协议。

**⇒ 无单一机制(单一模态)跨 ≥2 库过线;但单一杠杆(encoder-level LoRA)在 final-epoch 协议下跨 2 库过线,
经由不同模态,marginal on ZH。novelty 是否成立 = 用户 D7 裁决(见 banner);无论如何是 encoder-class 杠杆,不并入主表。**

### PUR-3 — LoRA-HateMM 配对表:encoder-level LoRA-Qwen vs frozen-CLIP,HateMM(3 head-seed,job 13235,两协议并排)

同 runner、同 `--seed`、同 215 HateMM test videos 的 head-level 配对读数。**LoRA 臂 = job 13235**(fresh SFT
13233 → 提取 13234 → head 13235;hash-freeze 提交时逐位复核 match);**CLIP 臂 = 既有 12850 frozen-CLIP 日志,
未重跑**。floors 由 12850 trainlog 独立重解析(与 PUR-1/T1.1 同一 parser)。

**协议 (A) val-selected(warmup≥5,max Val_Retrieval acc,roc tie-break):**

| seed | LoRA acc | CLIP acc | **Δacc** | LoRA mF1 | CLIP mF1 | **ΔmF1** |
|---|---|---|---|---|---|---|
| 0 | 0.8605 (ep19) | 0.8279 | **+0.0326** | 0.8521 | 0.8172 | **+0.0349** |
| 1 | 0.8698 (ep14) | 0.8279 | **+0.0419** | 0.8620 | 0.8163 | **+0.0457** |
| 2 | 0.8558 (ep22) | 0.8047 | **+0.0511** | 0.8495 | 0.7920 | **+0.0575** |
| **mean** | 0.8620 | 0.8202 | **+0.0419** | 0.8545 | 0.8085 | **+0.0460** |

**协议 (B) final-epoch(epoch 29):**

| seed | LoRA acc | CLIP acc | **Δacc** | LoRA mF1 | CLIP mF1 | **ΔmF1** |
|---|---|---|---|---|---|---|
| 0 | 0.8651 | 0.8186 | **+0.0465** | 0.8580 | 0.7997 | **+0.0583** |
| 1 | 0.8744 | 0.8047 | **+0.0697** | 0.8660 | 0.7822 | **+0.0838** |
| 2 | 0.8698 | 0.8140 | **+0.0558** | 0.8613 | 0.7988 | **+0.0625** |
| **mean** | 0.8698 | 0.8124 | **+0.0573** | 0.8618 | 0.7936 | **+0.0682** |

**判决(绑定,`LORA_HATEMM_VERDICT_REVIEW.md` §5 逐字):** `HateMM: final-epoch: PASS; val-selected: PASS.`
两协议 KS-1 双 conjunct(mean Δacc AND mean ΔmF1 ≥ +0.030)+ sign 3/3 均达标,余量 val-sel +0.0119 acc / +0.0160 mF1、
final-ep +0.0273 / +0.0382 —— **非 marginal**(val-sel acc 余量 ≈9× B3 的 +0.0013)。

**KS 诚实旗(均未触发):**
- **KS-2(family-coherence,非性能 kill):** final-ep LoRA 0.8698/0.8618 vs frozen-Qwen 0.8682/0.8591 ⇒ LoRA − Qwen
  **+0.0015 acc / +0.0026 mF1**(LoRA ≥ frozen-Qwen ⇒ **未触发,STRENGTHENS** 单杠杆叙事);val-sel LoRA 0.8620 vs
  frozen-Qwen 0.8729,阈值 0.8729−0.014=0.8589,LoRA 0.8620 ≥ 0.8589 ⇒ 落在 0.014 种子带内,**未触发**。数据表明
  **LoRA ≈ frozen-Qwen**(over-CLIP 增益主要是 frozen-Qwen 转换的继承);逐流分解(`HATEMM_LORA_STREAM_DECOMP.md`,
  `51eb95b`,F58)校正了预声明 F0.4 的「image-inheritance」措辞 —— HateMM 决定性单流是 **text**(text-only AUC ≥ image-only,
  三编码器双 footing),过线是 **text-carried / frozen-swap-sufficient**(frozen swap 已转换,LoRA text-sharpen 加 ≈0),此 nuance
  travels to D7(见 PUR-2 解释行、`DRAFT_analysis_chapter.md` §3.9)。
- **KS-3(P9-echo):** LoRA(val-sel 0.8620 / final 0.8698)远高于 CLIP floor(0.8202 / 0.8124),**未触发** ⇒ encoder-level
  regime 在 HateMM 转换(与 decision-level P9 C3-knn −4.7 相反),重申两-regime 区分。

**frozen-Qwen 次级 floor(KS-2 配对用,PMT/exp-encoder-3seed 一致):** val-sel 0.8729/0.8648;final-ep 0.8682/0.8591。

**合规(`LORA_HATEMM_VERDICT_REVIEW.md` §4):** hash-freeze 提交时逐位 match;head runner `run_rac.py` argv 与 12850
CLIP 控制逐字一致(仅 `--model` + fresh group);每库单次 test-touch;single-encoder-draw 预声明(±band = head-seed 方差,
非 SFT-draw 方差)。**一处非载重偏离**(诚实标注):LoRA head 跑于较新 `run_rac.py`,带 7 个 12850 无的 TARC/oracle
argparse 字段,全为 inert OFF 值(可证 no-op,且与已接受的 B3 判决同一条件);另一 benign SFT-loss 注(eval_loss 0.1084
略紧于 MHC anchor 0.1620)—— 均不影响任何 KS 判决。

来源:`refine-logs/LORA_HATEMM_VERDICT_REVIEW.md`(commit `6b8f634`,job 13235,独立 0-context 判决复核)·
`research-wiki/experiments/exp-lora-hatemm.md`。

### PUR-4 — cand-2 curriculum coupling probe:memory→adaptation coupling 是否升级 LoRA 腿(更窄的 D7 sub-ruling)

cand-2(confusion-weighted 单视频 SFT curriculum,唯一 manipulated variable = 样本重数,cost-neutral)预注册测试
「memory→adaptation coupling 是否 add-over-generic LoRA」以支撑一个比 D7 更窄、更强的子裁决。**结果(见 T5.4(a) /
`research-wiki/experiments/exp-cand2-curriculum.md`,job 13241,`546acc5`):ZH K-C2-2 tie 双协议**(NO novelty on
a-priori-most-likely 主腿,预声明 F0.7),**HateMM K-C2-2 pass 仅 val-sel**(+0.0155 acc / +0.0166 mF1,3/3,
rep2 后 pooled weakly-hardened across two draws,5/6 sign,per-draw 3/3 gate not met——见 T5.4(a-rep2);
final-ep tie +0.0093),**ZH-robustness NOT strengthened**。故 `D7_RULING_DOSSIER.md`
§5 的 **(B) 分支**条件(「K-C2-2 PASS ≥1 dataset **AND** ZH-robustness strengthened」——prereg §8 要求 BOTH)**只满足**
**一半**:add-over-generic 在一个 dataset(HateMM,val-sel;rep2 后 pooled weakly-hardened,5/6 sign,per-draw 3/3 gate not met)成立,ZH-robustness 半条未达。coupling 的可测
效应 **dataset/protocol-local**,**不开新数据集**(F0.4)。是否据此把 LoRA 腿从「generic encoder-class」升为
「memory-coupled adaptation curriculum」并主张 coupling novel-in-field = 用户 D7 sub-ruling,本节不判、**不并入主表**。
来源:`refine-logs/CAND2_VERDICT_REVIEW.md`(`546acc5`,job 13241)· `research-wiki/experiments/exp-cand2-curriculum.md`
· `refine-logs/D7_RULING_DOSSIER.md`(`def6ce3`,evidence-only,零 advocacy)。

### PUR-banner — 本节全部行 PENDING 以下用户裁决(逐条,未在此解决)

> **EVERY ROW IN THIS SECTION IS PENDING USER RULINGS:**
> **(i) novelty 边界。** LoRA / RA-HMD-family 杠杆被项目分类为 *"MIXED performance lever, not novelty"*
> (`query_pack.md:44`;`B1_PREREG_REVIEW.md:64`)。一个 LoRA-encoder 性能 pass 是否计入 goal 的 "novel" 子句,
> 是 pending 用户裁决,本节不判。
> **(ii) 单杠杆-两机制 headline。** round-4 line-A(F53)后,**单一杠杆 encoder-level LoRA 在 final-epoch 协议下跨
> 2 库过线**(HateMM solid + ZH marginal),不再需要 frozen+LoRA 的 "family" 拼接。逐流分解
> (`HATEMM_LORA_STREAM_DECOMP.md`,`51eb95b`,F58)实测:两次过线的**决定性模态相同(均 text-carried)**,区别在**适配杠杆**——ZH
> 过线是 text-borne / **LoRA-specific**(frozen-Qwen 在 ZH FAIL,故适配是必需杠杆),HateMM 过线是 text-carried /
> **frozen-swap-sufficient**(frozen swap 已转换,LoRA ≈ frozen-Qwen 只是继承;KS-2 未触发)。是否接受"一个杠杆、同一决定性
> 模态(text)、两种适配角色"作 ≥2-数据集 headline —— 抑或要求**单一杠杆-单一角色**跨 ≥2 库(此时 LoRA-specific 转换仅
> ZH、frozen-sufficient 继承仅 HateMM,EN 又 fail)—— 是用户裁决;且 val-selected 协议下同一杠杆仅 HateMM 过线(协议依赖)。
> **(iii) :58 barred-comparison 注**未在此解决。B3 的同 runner 同种子配对是现存最干净的配对读数并已在 PUR-1
> 记录,但它是否**覆盖** `PAPER_MASTER_TABLES.md:58` 的"不可直接同格并比"记账注以支撑一个论文主张,是**用户的
> override 决定**——本节**不编辑、不重解释 :58 本身**,亦不据此把任何行并入主表 T1–T4。
> **(iv) memory→adaptation coupling 子裁决(cand-2,round-4 closing,PUR-4)。** cand-2 是否把 LoRA 腿从
> "generic encoder-class" 升为 "memory-coupled" 并计入 novelty = 用户 D7 sub-ruling。measured 结果:D7 dossier
> (B) 分支只满足一半(HateMM val-sel add,rep2 后 pooled weakly-hardened、5/6 sign、per-draw 3/3 gate not met;ZH-robustness NOT strengthened;ZH K-C2-2 tie 双协议),
> coupling 效应 dataset/protocol-local、不开新数据集——本节不判、不并入主表。
