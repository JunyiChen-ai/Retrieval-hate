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
| **HateMM** (215) | frozen-CLIP RGCL floor | CLIP ViT-L/14-336 | 0.8732 | — | — | — | 1 | exp-baseline-reproduction · `becfd91` |
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
| **human-in-the-loop memory edit** | 删 2 条人工标记噪声记忆:EN test acc **0.8075 → 0.8199**(macro-F1 0.7626→0.7748),seed 0,**零重训**;超全部 5 随机 seed floor | 语义寻址 + 外科删除,纯 CPU 秒级;EN 全项目最高单点(能力演示,非主表) | DEMO_memory_editing / EXP_auto_memory_repair(复现门 PASS)· `d4e58aa` |
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
| 0 | **auto-repair**:两票 AND 规则自动删噪记忆,复现手工 2-entry 增益 | C−A **+0.0000**(0/4 EN);手工删的 2 id embedding 反对率 0.50/0.60 < 0.80 阈值;C−D +0.47 EN/+0.40 ZH | **FAIL**:AND 规则结构性删不到「语义矛盾但非 embedding-outlier」的记忆;幸存=guard-rail(见 T3) | EXP_auto_memory_repair · `d4e58aa` |
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

---

*(本文件为纯汇编,不含新实验;所有数字均为已 commit 结果的转录,出处见各表 commit 列与文首 document-level
commit 清单。三终局选项的具体第一周动作草案见 `OPTION_KITS_terminus.md`。)*
