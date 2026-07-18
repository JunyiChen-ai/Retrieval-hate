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
+0.0166 mF1,3/3,**single-curriculum-draw caveat F0.2**;final-ep tie,+0.0093 < +0.010 门 0.0007)。
**ZH-robustness = NOT strengthened**(§3.7(a) val-sel 不过 + (b) final-ep 未变 non-marginal,ZH final +0.0380 <
+0.040、seed2 +0.0134 < 逐种子门;≈ B3 现状)。**KS-regression / KS-below-floor 均未触发,无 kill;合规干净**
(same-code 76/80 fields,单次 test-touch/库,F0.8 class-balance shift 预声明)。**载重读法:memory→adaptation
coupling 相对通用 LoRA 的可测效应是 dataset- 与 protocol-local**——仅 HateMM val-sel(single-draw)有,主 ZH 腿无;
cand-2 **不开新数据集**(F0.4)、**不并入主表**;是否足够支撑 D7 memory-coupling 子裁决 = 用户裁决(见 PUR-4 /
PUR-banner + `refine-logs/D7_RULING_DOSSIER.md` `def6ce3`)。

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
   K-C2-2 tie 双协议、HateMM K-C2-2 pass 仅 val-sel(single-draw)、no kill fired——**held pending D7 sub-ruling
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
single-curriculum-draw caveat F0.2;final-ep tie +0.0093),**ZH-robustness NOT strengthened**。故 `D7_RULING_DOSSIER.md`
§5 的 **(B) 分支**条件(「K-C2-2 PASS ≥1 dataset **AND** ZH-robustness strengthened」——prereg §8 要求 BOTH)**只满足**
**一半**:add-over-generic 在一个 dataset(HateMM,val-sel,single-draw)成立,ZH-robustness 半条未达。coupling 的可测
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
> (B) 分支只满足一半(HateMM val-sel single-draw add;ZH-robustness NOT strengthened;ZH K-C2-2 tie 双协议),
> coupling 效应 dataset/protocol-local、不开新数据集——本节不判、不并入主表。
