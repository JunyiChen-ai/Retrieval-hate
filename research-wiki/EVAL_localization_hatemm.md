# HateMM 时序定位评测台 — span 验证 + 定位协议 + 我们方法的定位数字

Date: 2026-07-03. Scope: 方向 A(共识去噪)的时序定位评测台。
**范围决定(用户,2026-07-03)**:MultiHateLoc(arXiv 2512.10408v3, WWW 2026)官方仓库为空、无代码
→ **不做复现**,只在 related work 中讨论其发表数字(见 §4 协议对照表)。
已起草的复现代码停留在 `/data/jehc223/RGCL/baselines/multihateloc_reimpl/`(STATUS: ABANDONED,不再投入)。
诚实条款:所有自定口径逐条写明;协议不明的外部数字一律标注"不可直接比较"。未 git commit。

---

## 1. HateMM 金标 span 可用性验证 — 判定:**可行**

来源:官方 Zenodo 标注(record 7799469)`HateMM_annotation.csv`(4 列
`video_file_name,label,hate_snippet,target`),已存
`/data/jehc223/RGCL/data/gt/HateMM/HateMM_annotation.csv`。

**格式**:hate_snippet 为 `HH:MM:SS` 时间区间列表(如 `[['00:00:34','00:01:34']]`,可多段)。
**427/427 个 Hate 视频全部非空且可解析,0 条格式失败**;656 个 Non-Hate 行全空。

**异常(共 3 条,已处理并记入金标 note 字段)**:

- hate_video_86 第 2 段、hate_video_275 第 3 段:零长点标注(start==end)→ 丢弃该段;
- hate_video_412 第 2 段 `00:02:57→00:02:53` 起止颠倒(疑手误)→ 交换端点。

**统计**(全部 427 个 hateful 视频,共 671 段):

| 维度 | 数值 |
|---|---|
| 段数分布 | 1 段 268 / 2 段 102 / 3 段 40 / ≥4 段 17(最多 7) |
| 段时长 | 中位 32s(P25 15s,P75 77s;最短 1s,最长 634s) |
| span 覆盖视频时长比例 | 中位 0.459(P25 0.238,P75 0.741) |
| ≥95% 全片覆盖 | 62/427(14.5%);≥90%:83(19.4%) |
| span 越界(end > 视频时长) | 仅 6 条,超出 0.04–2.0s(秒级舍入),截断处理(`clipped:true`) |

**与视频时长对齐性**:ffprobe 全部 1083 个 mp4 成功;随机抽 10 条(seed 42)人工核对
(hate_video_23/388/105/291/44/7/233/356/129/401),无一条结构性错位;全片覆盖的均为短片
(如 61s、33s 全程仇恨),长片短 span(如 hate_video_129:542s 片中 [305,398])定位价值高。

**判定理由**:span 全覆盖、对齐良好、且只覆盖 hateful 视频时长的中位 ~46% → frame-level
定位评测在 HateMM 上**可行且非平凡**(不退化为视频级分类)。

**splits 交集**(我们的 splits,`data/gt/HateMM/{train,val,test}.jsonl`):
train 300 hateful(全带 span)/ val 42 / test 85;test 另含 128 non-hate → test 共 213 视频。

**产出文件**:

- `/data/jehc223/RGCL/data/gt/HateMM/hate_spans.json` — 机器可读金标(1083 条全量,
  `{video_id: {duration, spans:[[s,e],…], clipped, note}}`,non-hate spans=[])
- `/data/jehc223/RGCL/data/gt/HateMM/video_durations.jsonl` — ffprobe 时长缓存(断点续跑)
- `/data/jehc223/RGCL/scripts/analysis/hatemm_spans.py` — 解析/统计/金标生成脚本(可重跑)

## 2. 定位协议定义(自定口径,逐条)

参照文献只报 HateMM frame-level mAP/AUC 而**不写明口径**(帧定义、是否含 non-hate 视频、
池化方式均未说明)。以下为我们自定并全项目统一的口径:

1. **帧定义**:1 fps;第 t 秒帧代表 [t,t+1),label=1 ⇔ 秒中点 t+0.5 落在任一 gold span 内。
   视频末不足 1s 的尾巴丢弃;duration 以 hate_spans.json(ffprobe container duration)为准。
   理由:秒级粒度与金标 HH:MM:SS 精度一致;秒中点规则消除边界二义。
2. **protocol-full(主口径)**:test 集**全部 213 个视频**(85 hate + 128 non-hate)的所有秒
   池化为一个二分类集合,报 AP(sklearn `average_precision_score`,文献语境称 mAP)+ ROC-AUC。
   理由:与 video-level 检测同一 test 全集;负帧同时含"非仇恨视频的帧"与"仇恨视频的
   非仇恨秒",最接近部署语义。
3. **protocol-hateonly(附加口径)**:只池化 85 个 hateful 视频内部的秒,报 AP + AUC。
   理由:剥离"视频级可分性"对定位分数的泄漏,测纯段内定位能力;两口径并报防口径挑选。
   注意 hateonly 正例 base rate 很高(59.3% 的秒为正)→ **mAP 数值虚高,AUC 更诚实**。
4. **不做 per-video AP 平均**:62/427 全片覆盖视频无负帧、per-video AP 无定义,会被迫丢样本。
5. **不做 segment-level AP@IoU 主指标**:金标为分钟级粗 span,我们方法只有 K=4 等分粗段,
   IoU 匹配对两边都严重量化,无信息量;只在需要时作附录。

## 3. 我们方法的定位数字(HateMM test,213 视频,协议同上)

基于已有资产:subclipK4 缓存(job 12187)+ consensus 片段级投票(共识票 = 片段仇恨分),
未改 src/,纯 CPU 分析脚本。

| 配置 | full mAP | full AUC | hateonly mAP | hateonly AUC |
|---|---|---|---|---|
| consensus-vote(training-free kNN 共识票)| 0.4975 | 0.7062 | 0.6161 | 0.5673 |
| model-score(RAC_video_CLIP ckpt 头前向)| **0.5892** | **0.7813** | **0.6244** | **0.5771** |
| video-broadcast(视频级概率广播,诚实对照)| 0.5776 | 0.7735 | 0.5936 | 0.5000 |
| random(seed 0,sanity 下界)| 0.2601 | 0.4980 | 0.5928 | 0.4977 |

**配置说明**:consensus 超参 topk=10、τ=0.2、EM=2、drift 保留、conflict=ignore
(与 exp-consensus-kill-ablation 一致);kNN 库 = train 集 3132 个子片段(783 视频 × K=4,
视频级标签),票 = 邻居标签的 τ-加权相似度软平均,EM 轮间更新库标签权重;test 子片段 852 个。
subclip→时间映射:`generate_subclip_embedding_HF.py` 对已抽帧序列 `np.array_split` 等分 K=4,
HateMM 帧目录为均匀抽帧(抽样核对 12 个视频)→ 第 k 段映射 [k·D/4,(k+1)·D/4)。
无任何插值/平滑后处理。

**诚实解读**:

- **protocol-full 的大头来自视频级可分性**:video-broadcast 已达 0.578/0.774,
  model-score(0.589/0.781)只高 +0.012/+0.008 → K=4 段的段内分辨贡献很小。
- **hateonly 口径下仅 0.57–0.58 AUC**(高于随机 0.50 但弱):K=4 粗段(平均 ~47s/段,
  大于金标段中位 32s)+ CLIP 视觉-only 键,对 HateMM 大量 speech-carried 仇恨是盲区。
  consensus-vote 与 model-score 的段内区分能力相当(0.567 vs 0.577)。
- 我们的方法**并非为定位设计**(K=4 是检测用的粗分段);此表的作用是给共识票的
  "片段仇恨分"一个定位语义下的定量参照,并指出改进方向(更细分段、语音模态键)。
- consensus 超参直接搬 MHC 配置,未在 HateMM 上调过。

**Video-level 参考行(同一 test split,主表可比)**:

| 方法 | test acc | test macro-F1 | 来源 |
|---|---|---|---|
| frozen-CLIP RGCL(RAC_video_CLIP)| 0.8279 | 0.8172 | job 1035814,val-selected ep24(n=215),`rgcl_HateMM_openai_clip-vit-large-patch14-336_HF_1035814.trainlog:257-259` |
| frozen-Qwen RGCL | 0.870 | — | exp-baseline-reproduction.md |

**产出**:`/data/jehc223/RGCL/scripts/analysis/eval_localization_ours.py`
(--config {consensus,model,video,random,all},断点安全);
中间产物 `/data/jehc223/RGCL/scripts/analysis/out_localization/`
(secs_cache.npz + results_ours_loc.json)。

## 4. MultiHateLoc 发表数字协议对照表(仅供 related work 讨论)

> **不可直接比较声明**:MultiHateLoc 官方仓库为空(无代码、无预测文件、无特征);
> 论文未写明 frame-level 协议的关键口径(帧定义/采样率、是否纳入 non-hate 视频的帧、
> 池化方式、splits)。下表仅用于 related work 中的定位讨论,**不得**与 §3 我们的数字
> 做同表同口径比较。用户决定(2026-07-03):无代码的工作不复现。

| 项 | MultiHateLoc(发表,HateMM frame-level)| 我们(§3,自定协议) |
|---|---|---|
| 数字 | mAP 0.645 / AUC 0.799(full V+A+T) | model-score:full 0.589/0.781;hateonly 0.624/0.577 |
| 帧定义 | 未写明 | 1 fps,秒中点规则(§2.1) |
| 负帧范围 | 未写明(是否含 non-hate 视频未知) | full=含 213 全集;hateonly=仅 85 hate 视频(两口径并报) |
| 池化方式 | 未写明(池化 AP 还是 per-video 平均未知) | 全集池化 AP;不做 per-video 平均(§2.4) |
| splits | 未写明与我们是否一致 | 我们的 train/val/test(300+42+85 hate) |
| 监督/特征 | 弱监督 MIL Top-K(K=3),ViT/VGGish/BERT(Whisper) | 视频级标签,CLIP ViT-L/14-336 视觉,K=4 粗段 |
| 可复核性 | 无代码、无预测文件 | 脚本+金标+中间产物全部在库(§1/§3 路径) |

related work 定位话术(供论文用):MultiHateLoc 是 hateful video 时序定位方向的首个
弱监督工作,报 HateMM frame-level mAP 0.645/AUC 0.799,但官方无代码且协议欠明,数字
不可复核、不可直接对比;我们在公开金标(Zenodo hate_snippet)上定义了完全可复核的
frame-level 协议(§2)并给出我们方法与诚实对照的数字(§3)。

## 5. 清理状态

- `/data/jehc223/RGCL/baselines/multihateloc_reimpl/` — 复现起步代码(仅骨架,未产出特征
  /未训练),**STATUS: ABANDONED**,留在原地不删、不再投入;该工作线从未提交过 SLURM
  作业(squeue/sacct 已核实,队列中 arc*/trs*/hcs_download 均属其他工作线)。
- 本文档相关工作未做任何 git commit。
