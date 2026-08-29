# FINAL REPORT — 项目终局收卷(原 MORNING_REPORT,文件名保留)

**FINAL - 2026-07-05**

_面向项目负责人验收的终版报告:全部实验已收敛,无在飞作业,无未判读结果。每条结论后括号内为 research-wiki 内的出处文件;所有数字可追溯到 job id 与日志行。协议如未特别注明,均为预注册口径:warmup≥5、val-selected(max Val_Retrieval acc,roc tie-break),150 量级测试集,MHClip test EN n=161 / ZH n=149。本文件此前为 2026-07-03/04 晨报,本版为终局固化版;历史版本见 git。_

---

## 1. 终版记分板(目标 acc ≥ 0.85)

| 数据集 | 状态 | 终版数字 | 说明 |
|---|---|---|---|
| **HateMM** | **✓ 达标** | frozen-Qwen RGCL **0.870** / F1 0.861;frozen-CLIP 0.8279 / F1 0.8172 | 早已达标,后续未动(`experiments/exp-baseline-reproduction.md`) |
| **ImpliHateVid** | **✓ 达标** | **~0.91**(frozen-CLIP 0.910 / frozen-Qwen 0.900) | 早已达标(`experiments/exp-baseline-reproduction.md`、`DESIGN_iter1.md`) |
| **MHClip-ZH** | △ 双口径,协议选择=用户拍板项 | **val-选点 ~0.827**(archive 臂 0.8268±0.0266 / LoRA-only floor 0.8282±0.0139,5 seeds,均不过 0.85);**final-epoch(selection-free)floor 0.8537±0.0120 —— 过 0.85**(seeds 3/4 达 0.8658;archive 臂与 floor 每 seed 逐位相同) | 预注册口径不过线、标准 selection-free 口径过线;因过线才换口径 = rule-shopping,风险与两口径并排方案见 §7.1(`experiments/exp-archive-knn-seeds.md` Addendum 1/2) |
| **MHClip-EN** | ✗ 未达,**近天花板定位**(所有杠杆穷尽) | **≈0.78–0.80 双口径**:val-选点 floor 0.7702±0.0221 / archive 0.7935±0.0205(n=4);final-epoch floor 0.7888±0.0152 / archive 0.7826±0.0134;全配置挤在 0.77–0.79,无任何配置分离(`experiments/exp-archive-knn-seeds.md` Addendum 3) | **已测杠杆全部噪声级、有害或未过 val 门**(全名单见 §5):LoRA-SFT、consensus-clip/archive/blend/mm、transcript 键、archive-kNN α 网格、mode=both、double key、role-3 三代仲裁器。**同场定位**:MoRE 同场复跑 EN clean 仅 **0.69–0.72 acc**(as-released 0.6894 / bugfix 0.7019 / 5-seed 均值 0.722);**CRAVE 发表 M-F1 79.81 / ACC 82.50 为该 split 场上最高发表数字**(全量 split,与 clean 子集不可直接比,`HEADTOHEAD_FEASIBILITY.md` §3)。我们 0.79/0.74(acc/F1,frozen-Qwen)已在发表最强者的量级上、大幅高于同场可复跑者 —— 支持"近天花板"定位,叙事转归因分析(§4③) |

参考点(非 headline):EN 记忆编辑(role-2,删 2 条人工标记噪声记忆)后 test acc 0.8199 —— 全项目 EN 最高单点,属能力演示不属主表(`DEMO_memory_editing.md`)。

---

## 2. 同场 MoRE 三库全胜表(论文主对比)

MoRE(WWW 2025)官方代码全量复跑,同 split(逐行 diff 一致)、同 clean test 子集、双 variant(as-released / bugfix)+ 5-seed 敏感性;缺失件(caption/tsv/OCR)全部本地复原并脚注(`BASELINE_MoRE_rerun.md`)。

| 数据集(clean test n) | MoRE as-released | MoRE bugfix | MoRE 5-seed 均值 | 我们最好配置 | **Δ(我们 − MoRE 较优 variant)** |
|---|---|---|---|---|---|
| HateMM(215) | 0.8140 / 0.7988 | 0.8047 / 0.7899 | 0.792±0.035 / 0.781±0.038 | frozen-Qwen **0.870 / 0.861** | **+5.6 acc / +6.2 F1** |
| MHClip-EN(161) | 0.6894 / 0.4438 | 0.7019 / 0.5084 | 0.722±0.031 / 0.530±0.111 | frozen-Qwen **0.7888 / 0.7378** | **+8.7 acc / +22.9 F1** |
| MHClip-ZH(149) | 0.7651 / 0.6882 | 0.7584 / 0.7058 | 0.717±0.035 / 0.661±0.023 | LoRA-SFT **0.8322 / 0.8023** | **+6.7 acc / +9.7 F1** |

- **三库全部同场胜出,取 MoRE seed 均值上界或较优 variant 均不翻转**;名义训练标签量还是 MoRE 略占优(EN 618/ZH 633 vs 我们 clean 550/579)。
- sanity:HateMM(唯一数据完备库)复跑落在其发表值 −2~3pt(单 seed 方差内)= 复现成功;MHClip 低于发表值的归因(数据缺失为主、EN val 早停塌缩)已按证据强度排序入档,发表值另列 "reported (full data)" 行不与 clean 数字直接比。
- 引用口径:主表用 as-released(seed2024)+ 脚注(seed 均值±std、bugfix、caption/OCR 复原、EN 早停塌缩);CRAVE 列 in-dataset SOTA 发表数字行并划界(方法族不同、无同场轨道)。

---

## 3. 定位评测(第 5 章素材,能力演示口径)

- **HateMM 金标定位台**:model-score full mAP 0.589/AUC 0.781,但 video-broadcast 对照 0.578/0.774 → 段内分辨贡献小,hateonly AUC 0.577;视觉-only 键对 speech-carried 仇恨是盲区,如实写(`EVAL_localization_hatemm.md`)。
- **HateClipSeg 零训练跨库定位**(395 视频/10,572 段,90.8% 存活子集):最好配置(HateMM 子片段记忆,K=4)full AP 0.545/AUC 0.588,对 random +0.088/+0.100;**within-video 时序信号统计显著但幅度小(wv-AUC 0.526,仅 1/4 cell 过 Bonferroni)**;池化指标主体是"毒性密度"的视频间排序;K=30 密度匹配为负结果;**换记忆(HateMM↔MHC)零重训可预期地改变行为** = 可换记忆支柱的双向证据(`EVAL_localization_hateclipseg.md`)。
- 措辞红线:只说 **span-free**,不说 first/annotation-free/dense-supervision-free(MultiHateLoc/LELA/TANDEM 占位,`DESIGN_iter3.md`)。

---

## 4. 四支柱终版(claim 措辞 + 证据文件)

### ① 检索对比学习 + kNN 记忆(核心骨架)

**Claim**:同一 RGCL/RA-HMD 检索引导对比 + kNN 投票骨架承载 4 个 hateful-video 数据集,HateMM/ImpliHateVid 达标,并在严格同场(同 split、同 clean test、复跑而非引数)下三库全胜 MoRE(+5.6~+8.7 acc);跨数据集 kNN 记忆换库 5/6 有效跨格 above-majority、零重训 —— trained-MoE 头结构上不具备。
**证据**:`experiments/exp-baseline-reproduction.md`、`BASELINE_MoRE_rerun.md`、`experiments/exp-cross-dataset-transfer.md`、`HEADTOHEAD_FEASIBILITY.md`。

### ② 可更新记忆 + 校准适应(时间演化协议)

**Claim**:"hate evolves" 在 MHClip-EN 窗口内可测(temporal split −0.084 macro-F1),其主成分是**校准漂移而非可分性损失**(temporal ROC 0.8484 > 随机 split 参考 0.7175);正确的 k-shot 轻量适应是**阈值再校准:k=20 个新期标注样本零重训全额收复漂移**(0.7336 ≥ 随机 floor 0.7113),检索架构把 operating point 暴露为一等、O(1)、可逆旋钮,trained-MoE/分类头把它藏在权重里。原始"加样本进记忆"机制 flat-to-negative,如实报废;ZH 无漂移 = 负对照,无漂移信号时小 k 校准纯噪声,部署应由漂移监测门控。
**证据**:`EVAL_temporal_memory_W4.md`、`ideas/evolving-memory-protocol.md`(validated-as-calibration)。

### ③ 共识去噪 = 修复机制(ZH-scoped)+ 完整 EN 归因链

**Claim(修复,ZH)**:继承视频级标签的子片段监督毒化 ZH(−0.066 F1,单 seed 大效应);检索共识重标注**消除该毒化并落在 floor 之上或持平**(5 seeds、双口径,均值两口径皆最高:val-选点 0.7764±0.0406 / final 0.7841±0.0204),**但"反超 floor"不成立**(val-选点 +0.0115±0.0418,p≈0.57;final +0.0247±0.0272,p≈0.11)——论文措辞必须是 "consensus de-poisons sub-clip supervision (−0.066 → ≈ floor / weakly above)",不是 accuracy win(`experiments/exp-consensus-zh-seeds.md`)。
**Claim(归因,EN,三段闭环)**:(i) 视觉 clip 键共识毒化训练(−0.117 F1),投票实为视频级(within-video vote std 0.048)、严重度反相关、正监督供给崩塌 56% all-pruned;(ii) 换投票空间(archive/blend)救不回(双语全灭)⇒ 投票空间不是病灶;(iii) **证据匹配的片段语音键(窗级 Whisper ASR + CLIP-text 双通道)把 annotator 全面修好**(供给 56%→19%、投票变片段级 wv-std 0.048→0.12、严重度反相关消除、灾难性 clip-consensus 被完全救回 +0.10~0.13 F1)**但训练端仍 ≤ floor**(final-ep 3/3 seeds −0.0116±0.0087)⇒ **病灶钉死在片段监督通道本身对语音承载仇恨无增益**。该链堵死"你们只是键选得差"的审稿质疑;附 ZH 反例(mm 探针死:窗文本率 48.5% + CLIP-zh 弱 → ASR 通道对 ZH 是噪声)⇒ 方法学副产品 = "evidence-matched segment keys" + probe-before-train。
**证据**:`experiments/exp-consensus-kill-ablation.md`、`EXP_mm_segment_keys.md`、ITERATION_LOG W5 节(jobs 12243–12246)。

### ④ 可审计 / 可编辑的档案记忆

**Claim**:MLLM 结构化档案记忆是**可审计**(60 条分层抽审 faithful 77%;失败模式三类定型:字段级虚报 15%、标题-only 洗白 5%、1 例内容级虚构)与**可编辑**(语义寻址 + 外科删除,纯 CPU、秒级、零训练)的:EN 定向删除切片翻转率 ≈ 随机对照 15×;**删 2 条人工标记噪声记忆即修复 EN 0.8075→0.8199**(超全部 5 随机 seed);审计驱动的 prompt v2 把 ZH 有害类 target 召回 1.6%→49.0%(EN 11.6%→54.5%),修复 ZH 可寻址性(0→20/63 条),EN 方向性效应在 target 字段切片下复现且更干净(2/14 翻转 vs 随机 5-seed 全 0,整体 acc 不掉)。**定位 = 能力演示,不做切片级显著性主张**(n 太小);v2 键无 accuracy 收益(ZH −2.7 acc)——档案的付费点在审计/编辑,不在检测。
**证据**:`AUDIT_archive_faithfulness.md`、`DEMO_memory_editing.md`、`DEMO_memory_editing_v2_{zh,en}.md`、`ARCHIVE_V2_ITERATION.md`。

---

## 5. 全部被杀 / 撤回主张清单(终版)

| # | 被杀主张 | 撤回依据 | 出处 |
|---|---|---|---|
| 1 | **archive-kNN 键带来 accuracy 提升**(seed-0 ZH +0.020 / EN +0.019) | 5-seed 配对 dAcc −0.0014±0.0313;same-seed ckpt sha1 字节相同、α=0.25 键 ep29 0 票翻转 → 增益全是 78 样本 dev 选点运气 | `experiments/exp-archive-knn-seeds.md`(Addendum 2 sha1 审计) |
| 2 | **"archive > transcript = 结构化蒸馏"** 排序 | 多 seed ΔF1 +0.0001±0.0388(ZH)/+0.0013±0.0141(EN);truncation-repair 假设 ZH 也死 | `ABLATION_transcript_vs_archive.md` |
| 3 | **机制统一**(共识投票搬进档案/混合空间) | W5 双语言双配置全部低于原空间与 floor | ITERATION_LOG §W5(jobs 12243–12246) |
| 4 | **任务首次类措辞**(first / annotation-free / dense-supervision-free) | MultiHateLoc / LELA / TANDEM 占位;红线=只说 span-free | `DESIGN_iter3.md`、`NOVELTY_CHECK_dirA.md` |
| 5 | **cross-seed ensemble** | 用户政策明令禁止;零作业零脚本零数字 | `experiments/exp-archive-knn-seeds.md` Addendum 2 |
| 6 | **"ZH best-ever 0.8322" 单 seed 口径** | floor 本身多 seed 0.8282±0.0139(val-选点)/ 0.8537±0.0120(final);0.8322 只是 seed-0 一个点 | `ABLATION_transcript_vs_archive.md` |
| 7 | **MultiHateLoc 复现** | 官方仓库空(仅 LICENSE);用户政策"无代码不复现";起步代码标 ABANDONED 留档 | `EVAL_localization_hatemm.md`、`baselines/multihateloc_reimpl/` |
| 8 | **multi-granularity 段级检索**作为 headline | 语言符号翻转、噪声 MIL 伪正样本;降级为诚实消融 | `experiments/exp-seg-mode-ablation.md` |
| 9 | **mm 片段键主表主张**(EN consensus-mm ≥ floor,共识升级双语) | 预注册判定 FAIL:final-ep 3/3 seeds 低于 floor(−0.0116±0.0087,同向一致);val-选点 +0.0245 由单 seed 运气驱动(±0.0881);ZH mm 探针死未训练(预注册纪律)。**保留价值 = 归因链第三段 + probe-before-train 方法学**(annotator 修复层全部成立) | `EXP_mm_segment_keys.md` |
| 10 | **role-3 选择性推理**(kNN margin 门控 → MLLM 仲裁)作为 EN 破 0.85 杠杆 | 三代仲裁器(v1 通用 prompt / v2 口径校准 / v3 任务 LoRA)全部未过 val 门(EN 最好 0.7750<0.7875;ZH 最好 0.8590<0.8718),val 选定配置=不仲裁;v3 EN deferred-acc 0.615 < 0.667 打平线 << 0.846 跨线。**门控本身有效**(EN 24% 样本拿住 42% 错误;oracle 0.857–0.888)= 复活条件已量化,留给 ≥72B/API 级仲裁器 | `EVAL_role3_selective_reasoning.md` |
| 11 | **ZH 共识"反超 floor"**(seed-0 +0.0158) | 5-seed val-选点 +0.0115±0.0418(3/5 胜,p≈0.57)= 掷硬币;final +0.0247±0.0272(4/5,p≈0.11)仍不显著;**修复毒化主张幸存**(任何 seed/口径都不复现 −0.066 洞) | `experiments/exp-consensus-zh-seeds.md` |
| 12 | **v2 档案键作为 accuracy 手段** | 冻结获胜头换键:ZH 0.8523→0.8255(−2.7 acc)、EN 0.8075→0.8012;v2 只在审计/编辑维度付费 | `ARCHIVE_V2_ITERATION.md` §4 |
| 13 | **"定位能力强"类主张**(HateClipSeg/HateMM) | 池化指标主体=毒性密度的视频间排序(broadcast 对照几乎追平);within-video 信号仅 1/4 cell 显著(wv-AUC 0.526);K=30 密度匹配负结果 → 只作"span-free 能力演示 + 模态盲区归因" | `EVAL_localization_hateclipseg.md`、`EVAL_localization_hatemm.md` |

---

## 6. 方法学章素材(值一节附录)

**主命题:150 量级测试集 + 78 样本 dev 上,seed 噪声与选点噪声支配一切 ≤2 点的"增益";本项目的对策全部可复用。**

1. **选点噪声定量**:val-acc 选 epoch 相对 selection-free 协议自损 ~2 acc 点(ZH 两臂 val-选点 ~0.827-0.828 vs last5 ~0.846-0.848 vs final-epoch 0.8537);五规则网格(val-acc/val-ROC/top3/last5/final)下 ZH 配对档案效应在 −0.013~+0.008 摆动 —— **选点规则挪动估计值的幅度超过待测效应**;无规则跨臂一致占优 → 预注册规则不改。n=5 配对 MDE ~0.04-0.05 F1,真 +0.01-0.02 效应按设计不可测。selection-robustness 论文段落已成文可直接引用(`experiments/exp-archive-knn-seeds.md` Addendum 1)。
2. **probe-before-train**:零训练探针(与训练 E-step 共用同一键构造实现)+ 预注册双闸门(严重度相关性、正监督供给),先探针后训练;ZH mm 探针死 → 不硬跑(省 GPU 且免事后择臂);EN 探针过闸、臂间排序被训练端兑现(PRIMARY>SECONDARY),但探针过闸≠下游增益 —— 探针是必要非充分门(`EXP_mm_segment_keys.md` §3.2/3.4)。
3. **sha1 / bit-for-bit 审计纪律**:harness 确定性(同代码两遍 12/12 ckpt sha1 同)→ 改动无侵入(flag 默认关时逐位同)→ same-seed 跨臂 ckpt 字节相同(证明 kNN 键不触训练)→ disk_guard B2 sha1 对账。一切"增益"主张先过身份审计再过统计(`EXP_mm_segment_keys.md` §1、`experiments/exp-archive-knn-seeds.md` Addendum 2)。
4. **双口径并排报告**:val-选点(预注册)+ final-epoch(selection-free)全表并报,分歧本身入文(EN mm:val-选点假阳性 vs final 3/3 同向负;EN archive:val-选点 +2.3 假增益 vs final 0/4 正)。
5. **复跑取证协议**(MoRE):释出代码 7 项缺陷全部文档化处置、bug 保留(as-released)+ bugfix 双轨、seed 敏感性、缺失件本地复原并脚注、每视频预测落盘可审计(`BASELINE_MoRE_rerun.md`)。
6. **预注册纪律的负例价值**:role-3 的 val 门决策"不仲裁"(ZH v3 test 侧 +0.02 增益如实报告但按协议不选);mm PRIMARY/SECONDARY 提交前预注册,未做训练后挑臂。

---

## 7. 用户待拍板

1. **Headline 口径:val-选点 vs final-epoch(ZH 0.827 vs 0.8537)。** 事实与风险不变:预注册口径不过 0.85,final-epoch 过线但因过线才换=rule-shopping,rebuttal 必死;且 final-epoch 下 archive 通道 ZH 贡献恰好为零、EN 为负。**建议案维持:两口径并排**——主表沿用预注册口径,附录放五规则鲁棒性全表+已成文说明段;若改用 final-epoch,须以"未来预注册"名义全线统一并自曝决策时序。
2. **EN 近天花板定位确认。** 全部杠杆已穷尽(§1/§5),EN ≈0.78-0.80 双口径;同场 MoRE 仅 0.69-0.72,CRAVE 发表 79.81 F1 为场上最高(全量 split)。请确认接受"近天花板 + 归因分析"叙事:0.85 作为该 split 上未被本方法族达到的公开目标如实报告,EN 章节主体为 §4③ 归因链 + oracle 复活条件(role-3 门控留出 0.857-0.888 空间)。
3. **投稿目标。** 素材形态:主表(同场 MoRE 三库全胜)+ 四支柱 + 归因章 + 方法学附录;请拍板目标 venue 与截稿,以便按其页数/附录政策裁剪(候选讨论中曾出现 WWW / ICWSM / ACL-ARR 线,未定)。

---

## 8. 遗留 TODO 清单(收卷后未尽事项,均不阻塞定稿)

| # | 事项 | 说明 / 现状 |
|---|---|---|
| 1 | **ZH transcript 多 seed 的 final-epoch 合并表** | `experiments/exp-archive-knn-seeds.md` Addendum 3 的 EN 主表已含 floor/archive 双口径;transcript 臂(12260-12266)的 final-epoch 口径尚未并入同一张表(val-选点口径已在 `ABLATION_transcript_vs_archive.md`)。纯日志重解析,零 GPU |
| 2 | **HateClipSeg 用 mm 片段键重打定位** | 定位评测(`EVAL_localization_hateclipseg.md`)是视觉-only 键,结论明示改进方向=语音模态键;mm 片段 ASR 键基建(`generate_segment_asr_HF.py`/`generate_subclip_mm_embedding_HF.py`)已就绪,需对 HateClipSeg 抽 ASR 后重打 within-video 表。若做,先修 word-ts 降级(EN 41%,建议 whisperX) |
| 3 | **v3 档案方向** | `ARCHIVE_V2_ITERATION.md` §6 已列:few-shot 对比示例(医疗科普 vs 攻击)、target 区分"话题涉及/被攻击"、标题-only 毒性单独字段;v2 残留缺陷(benign mechanism 幻觉 59%/51%、标题洗白未修)是靶点 |
| 4 | **更强仲裁器** | role-3 复活条件已量化:EN deferred@30% ≥0.667 打平、≥0.846 跨 0.85;7B 线终结,留 ≥72B/API 级;ZH v3 test 侧未选中正增益提示"任务校准>prompt 工程"方向 |
| 5 | ZH full-mode 毒化洞(−0.066)补 seed | 修复 claim 的洞本身仍是单 seed(`exp-consensus-zh-seeds.md` caveat);如审稿要求可补 λ=0.5 full 臂 seeds 1-4 |
| 6 | 疑似 gt 漏标 `BV1MU4y1D7Ks` 人工终审 | 审计反向发现,模型初判、待人工确认(`AUDIT_archive_faithfulness.md`) |
| 7 | ASR 资产复用前修 word-ts | transformers word-ts DTW bug 致 EN 41% 降级 sentence-level;复用 `data/ASR/` 前先修或换 whisperX(`EXP_mm_segment_keys.md` 偏离条款) |

---

_附:`research-wiki/ITERATION_LOG.md` 已追加终局记录(2026-07-05);ideas 节点收口:`mm-segment-keys` 新建(outcome=attribution-closed)、`role3-selective-reasoning` 新建(closed)、`retrieval-consensus-denoising` 终态刷新(repair-yes / beat-floor-no / attribution-closed);全部 open 问题已移入本报告 §8 TODO。三条用户政策(禁 cross-seed ensemble / 无代码不复现 / 不发邮件缺件自补)全程执行,记录在 ITERATION_LOG §10。_

---

## 9. MLLM 方法角色攻关(2026-07-06,收卷后新增波次)

> 本节为终版报告固化(2026-07-05)之后新增的独立攻关波次。**不改动 §1–§8 任何结论,也不改动 §4 定位/三件套(仍待用户确认)。** 完整八行记分板与证据见 `research-wiki/CAMPAIGN_mllm_method_role.md`。

**问题(用户命题):** MLLM 除做冻结 encoder 外,能否挣得一个可被消融的**方法角色**——移除它会可测量地掉点(超过这些 ~150 样本测试集 ~1.6 视频 ≈ 1 acc 点的噪声地板)?六条独立集成路线各自预注册、各带"移除 MLLM"消融。

**结论:在 MHClip 上 MLLM 未挣得可移除的方法角色。** 七个已结前沿全部为诚实 kill 或 within-noise,且**每条都有复现 / bit-for-bit / probe 护栏背书**(非 harness 假象)。

| 前沿 | MLLM 的方法职责 | 结果 | 一句话死因 | doc·commit |
|---|---|---|---|---|
| P1 零标注先验重校准 | 读档案→无标注 HARMFUL/BENIGN→adjusted classify-count 估先验→分位重设漂移门控阈值 | FAIL(p̂ 误差 0.22 EN/0.18 ZH) | 判据 FPR 在时间边界漂移(.372→.238),train 端校正失真;机制本身成立(oracle 先验补回 EN 80% 缺口) | p1·`2a69246` |
| P2 7B 邻居重排 | 边界样本按可比性删 INCOMPARABLE 邻居再投票(不出标签) | FAIL(B−A −0.002/−0.020) | 过判 INCOMPARABLE(83%/70%),删除与投票正确性无关(selectivity +1.1%/−3.2%) | p2·`bc689e1` |
| P2b 强判据+train 端校准 | 7B/32B×证据×prompt train 端选择性榜,过 +10pt 才碰 test | FAIL(train 端即死,最佳 EN lift +2.7,ZH 全负) | **可比性 ⊥ 投票正确性**,32B 亦不选择性;重排线彻底关闭 | p2b·`cc4ca6e` |
| P3-EN 证据密度池化 | MLLM 打分段级证据密度 0–3→softmax 重加权池化视频嵌入 | FAIL(probe kill −0.0055@k20) | 信号真(hate/benign 段内 var 1.11/0.40)但**干预不迁移**:冻结 CLIP 中集中视觉信号并不比均值更可分 | p3·`c2ba59f` |
| P3-ZH 同上 | 同 | within-noise 无 claim(val −0.007/final +0.009,均 <1pt) | ZH 证据 ASR 稀疏(var 0.33/0.12),thin probe 早已预示 | p3·`15f5f08` |
| P4 schema 蒸馏 | 辅助线性头蒸馏档案字段(explicit/modality/mechanism/target,λ=0.1),eval 丢弃 | within-noise(EN −0.001/ZH +0.008 sub-threshold) | 字段可解码(AUC .62–.93)且预测标签(.74–.78),但**与直接标签监督冗余** | p4·`6f1f0da`,`00816aa` |
| P5 反事实孪生负样本 | MLLM 洗白转写→同视觉+洗白文本负样本(每 anchor 一个额外 hard-neg) | FAIL(质量门关闭 flip 0.503/0.337;诊断训练伤 EN −0.027) | MLLM 无法可靠洗白;干净孪生因**共享 anchor 视觉过近**(cos 0.73)反伤,pairing 不胜随机 | p5·`fc25cac`,`66d3103` |

**跨前沿定性(统一失败形状):** MLLM 语义能力真实(会读档案、能定位证据、字段可解码),但该能力**与决策变量正交或冗余**——语义"关于什么"不等于"在仇恨/冒犯/良性边界的哪一侧",而后者正是检索头已直接监督的量。

**存活价值(独立于上述 kill,可入论文):**(a) 可编辑记忆的**否决/守门**角色(auto-repair 定向删噪改善 EN)与人审记忆卫生——移除代价体现在完整性/可控性而非 raw acc;(b) P3 段级证据密度分是**无标注定位显著图**,是该信号的正确归宿(cross-ref `EVAL_localization_hateclipseg.md` / `EVAL_localization_hatemm.md`);(c) P2 oracle 头部空间 **+7.5/+10.6(均跨 0.85)** 作为量化天花板 + 已被 P2b 排除的"更强判据"路线,为未来"成员性信号"工作定标。

**两条前沿已落地(2026-07-07 更新):**
- **[已结·negative] P3-HateMM 训练** — 三者中唯一 k-consistent probe 正例(+0.0108@k20,证据最密 var 1.28/0.71),但训练仍 within-noise(wsoftT1 vs floor:val-sel ΔF1 −0.0041 / final +0.0004,双口径均 <1pt;floor 复现 published 0.828)。**决定性教训:过 no-head probe 是必要非充分——习得的 align-fusion(img×text)头吸收了输入端重加权。** 证据密度池化在 EN/ZH/HateMM 均未挣得方法角色。commit `22fe62a`/`783b751`。
- **[已结·POSITIVE] P6 HateClipSeg 定位(p2-rerank 承接)** — MLLM 逐窗证据分(帧+ASR)做无 span 时序定位:within-video AUC **0.5435** > memory 0.5140 > random 0.5088;配对 b>a Δ+0.0296 CI[+.009,+.050] p=0.007,对空 p=5.4e-8。**MLLM 挣得一个可移除的定位角色(幅度温和、统计稳固)。** commit `c9e3bd8`。

**Campaign 答案:** MLLM 在本项目挣得恰好**两个可移除方法角色**——**encoder**(HateMM +4.2 F1 跨 0.85)与**定位打分器**(P6)。**主表 accuracy 角色被八条预注册路线(P1/P2/P2b/P3-EN,ZH,HateMM/P4/P5,7B–32B 规模)彻底证伪**:无任何 MLLM 组件把静态测试 acc 抬过 ~1.6 视频噪声地板,且每条均护栏背书(复现 / bit-for-bit / probe)。两条方法论定论:**(i) 过 no-head probe 是必要非充分**(P3-HateMM 是最干净 probe 却训练 within-noise);**(ii) 语义能力与决策变量正交或冗余**(P1/P2/P2b/P4/P5)——"关于什么"不等于"在仇恨/冒犯/良性边界的哪一侧",而后者才是主表提升需移动、且已被直接监督的量。

**07-08 endgame(P9b / P10 收尾 + 终局草稿就位):** 收卷后又跑完两条路线,均 FAIL,主表 accuracy 结论不动。
- **P9b(rgcl-ON 12-run 波次)FAIL —— 再分配机制,非净增益。** 打开我方检索对比(rgcl)损失训练 LMM 嵌入空间(D3):D3-knn test **ZH 0.8389±0.005**(−1.5pt vs floor 0.8537,0/3 seeds)、**EN 0.7743±0.008**(−1.0pt vs 0.7847,0/3);判据 2(D3-knn ≥ D3-mlp−1pt)PASS,但判据 1 败,**0/12 cell 超 floor**。机制:rgcl 项把精度从 LMM 自带头**搬到**我方 kNN 读出(D3−C3′ knn +1.8pt ZH / +0.2pt EN),而自带头镜像下降(−1.8 / −1.2 mlp)——**head↔memory 精度再分配,非净增益**。这关闭了「决策级+rgcl」这最后一个架构 locus。详见 `CAMPAIGN_mllm_method_role.md` P9b 行 / `EXP_p9_lmm_rgcl_video.md`「P9b WAVE RESULTS」,commit `4d28655`。
- **P10 一轮(HateMM-span 标定 leaderboard)FAIL —— A-fuse 显著但未达标。** 在 HateMM span 上自由标定 P6 scorer、单次测 HateClipSeg,想把 P6 的 modest 定位放大到 substantial;无变体过 **+0.04** 晋级线。**A-fuse(K4×K30 coarse×fine)+0.0305 CI[+0.0175,+0.0437] p=7e-7 显著但 < +0.04 bar**,HateClipSeg test 未触、**P6 as-is 站住(wv-AUC 0.5435)**。commit `7194ee2`。
- **P10-b 在飞。** 32B/72B × A-fuse 二轮(预注册 `3d641f4`)由另一代理执行中,CPU 行 7B fuse×lex 0.5752(差 0.0035 到 0.5787 bar);只影响**定位角色能否从 modest 放大到 substantial**,不影响主表结论。
- **终局草稿就位。** `research-wiki/TERMINUS_mllm_campaign_DRAFT.md` 已以 DRAFT 状态汇总 11 路线判定 + 5 条横切机制定论 + 4 幸存角色 + 3 决策选项(§6 留 P10-b 占位符),待 P10-b 落地后由主会话决定是否定稿。

**07-09 终局(P10-b/P10-c 落地 + campaign 全数结题):** 收卷波次全部落地,主表 accuracy 结论不动;**定位角色止步 MODEST,substantial 线开源不可达。**
- **P10-b 终 —— 72B A-fuse 唯一晋级,test = MODEST。** 二轮标定中仅 **72B A-fuse** 过晋级线(HateMM-span 校准 **0.5913**),单次触 HateClipSeg test = **0.5755 = MODEST**(< 0.60 substantial 线;对 memory 基线 +0.0615、对 P6 as-is +0.0319,均显著)。commit `03880f2`。
- **重聚合天花板 0.5932。** 对已算 scorer 输出做全变体重聚合的上界为 **0.5932**(仍 < 0.60),关闭「换聚合方式即可到 substantial」的希望。commit `93e82fa`。
- **P10-c 开源代际跳跃 FAIL。** 押注「换代 > 换规模」:**Qwen3-VL-32B A-fuse 校准仅 0.5866**,< **0.616** 晋级门槛,且低于 72B 冠军 —— **换代 ≠ 换规模,激活参数量主导**;HateClipSeg test 未触碰。commit `74f0eac`。
- **结论:开源可行域三面墙闭合。** 规模(≤72B)、代际(Qwen3-VL)、聚合(重聚合天花板)三向均已撞墙,**substantial 定位线开源不可达**;MLLM campaign **12 条路线全部结题**。
- **决策悬置(等用户裁决)。** 三选项:**(a) 定稿**(MODEST 定位 + encoder/guard-rail 角色收官)/ **(b) 闭源 API**(需数据外发批准)/ **(c) 换方法族**。执行包 `research-wiki/OPTION_KITS_terminus.md`、终局报告 `TERMINUS_mllm_campaign_DRAFT.md`(已转 **FINAL**)、论文主表 `PAPER_MASTER_TABLES.md`(commit `d9731e8`)均就绪。
