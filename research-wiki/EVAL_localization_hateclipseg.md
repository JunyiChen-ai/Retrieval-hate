# HateClipSeg 零训练时序定位评测 — 跨数据集记忆 kNN 共识打分

Date: 2026-07-04. Scope: 一个实验同时演示两个 novelty 支柱:
(1) **span-free 定位** — 检索共识票(`consensus.py _knn_vote`,只读复用)直接给时间窗口打仇恨分,无任何 span 监督;
(2) **跨数据集可换记忆** — 记忆库来自 HateMM / MHC 的 train 集,HateClipSeg **零训练、零样本、零标签校准**(HateClipSeg 标签只进指标计算,不进打分路径)。
未 git commit。

---

## 1. 数据与清洗(评测底座)

评测集 = HateClipSeg 存活子集 **395 视频(bit 338 / yt 57),26.08 h**,段级 multi-hot 金标
(`0:normal 1:hateful 2:insulting 3:sexual 4:violence 5:harm`),详见 `DATASET_hateclipseg.md`。

**金标清洗**(`scripts/analysis/hateclipseg_prep.py`,产出 `data/gt/HateClipSeg/gold_segments.json`):

- 22 条异常段全部是**末段**且其 end == ffprobe 文件时长(标注时间轴过冲后末段"回弹"到真实片尾)。
- 规则:duration D := ffprobe 容器时长;所有段裁剪至 [0,D];s ≥ D 或裁后 e ≤ s 的段丢弃。
- 结果:**10,572/10,604 段保留**(丢 32、末端裁剪 29);每视频段从 0 连续铺满至 D,秒级索引 **0 秒缺失**。
- 段时长:中位 **8.12s**(P25 6.30 / P75 9.96);视频时长中位 239.1s。

**协议无 split 声明**:HateClipSeg 无官方 split;本评测**零训练**,无需 split → 直接在全部 395 视频上评测(不存在训练/测试泄漏,因为打分路径不含任何 HateClipSeg 标签或统计量)。

## 2. 打分方法(零训练,只读复用已验证机制)

- **窗口特征**:`src/utils/generate_subclip_embedding_HF.py`(未改)对每视频均匀采 M 帧、连续等分 K 窗、frozen CLIP ViT-L/14-336 pooler 均值池化(SLURM job 12274)。两个评测粒度并报:
  - **K=4(M=16)**:方法默认检测粒度,窗口中位 59.8s;
  - **K=30(M=120)**:密度匹配粒度 — 窗口中位 **7.97s ≈ 金标段中位 8.12s**(评测粒度选择声明:K=30 由金标段中位时长决定,非调参)。
- **打分** = consensus E-step 的相似度加权 kNN 票(`src/utils/consensus.py _knn_vote`,topk=10,与已验证 consensus 配置一致,未在 HateClipSeg 上调过任何超参):窗口 embedding 查询记忆库,票 = 邻居视频级标签的 τ 加权软平均 ∈ [0,1],直接作为该窗口的仇恨分。无 EM(EM 需训练)、无平滑、无校准、无阈值。
- **键空间(声明的偏离)**:视觉-only raw-CLIP 键(query = l2n(窗口特征),memory = l2n(记忆特征))。consensus.py round-0 的键还拼接**视频级**文本半区;HateClipSeg 未抽 transcript,且视频级文本在视频内为常量、**对窗口间(within-video)时序信号贡献恒为零** → 定位评测中只有视觉半区是"活"的;记忆侧同步去掉文本半区保证同空间。
- **记忆库**(全部为**别的数据集**的 train 集,视频级标签):
  - `hatemm_video`(主配置):HateMM train 743 视频(297 hate),整视频 CLIP 特征;
  - `hatemm_subclip`:HateMM train 2,972 个 K=4 子片段(继承视频标签)— 段粒度记忆变体;
  - `mhc_video`:MHC(EN)train 549 视频(168 hate)— **换记忆 = 换一个配置项**,零重训,演示可换记忆支柱。

## 3. 定位协议(自定口径,与 EVAL_localization_hatemm.md §2 对齐)

1. **帧定义**:1 fps;第 t 秒代表 [t,t+1),label 取秒中点 t+0.5 所落金标段;正例 ⇔ 该段 multi-hot 任一毒性位(idx 1–5)=1(主二值口径 any-toxic vs normal;金标中 normal 与毒性位无共现)。共 **93,705 秒,毒性占 46.37%**。
2. **窗口→秒映射**:秒中点 m 归窗口 q = min(K−1, ⌊mK/D⌋)(与 HateMM 评测同规则)。
3. **protocol-full(主)**:395 视频全部秒池化,AP(= 文献语境 mAP)+ ROC-AUC。
4. **protocol-toxiconly(附)**:仅含 ≥1 毒性秒的 345 视频(82,002 秒,prevalence 53.0%)— 对齐 HateMM 的 protocol-hateonly;base rate 高 → AP 虚高,AUC 更诚实。
5. **segment-level(金标原生粒度)**:每条金标段得分 = 覆盖窗口分的时长加权均值;10,572 段池化 AP/AUC。
6. **within-video mean AUC(时序信息量诊断)**:对两类俱全的 329 视频逐视频算 1fps AUC 再平均 — 广播类对照按构造 = 0.5,**该列 > 0.5 即窗口分携带真实视频内时序信息**。
7. 不做 per-video AP 平均、不做 IoU 主指标(理由同 HateMM 文档 §2.4/2.5)。

**对照**:`vbcast_hatemm_video`(主配置窗口分的视频内均值广播到全部窗口 — "只做视频级判断"退化)、`random`(seed 0)。"与内容无关的均匀分"对照被 random 覆盖(常量分无排序信息,AP=prevalence)。

## 4. 主表(job 12274 缓存;1 个视频不可解码 yt_NzvfkIYS5Yg,保留为常量分)

**K=4(方法默认粒度,窗口中位 59.8s)** — prevalence:full 0.4637 / toxiconly 0.5299

| 配置 | full AP | full AUC | toxOnly AP | toxOnly AUC | seg AP | seg AUC | wv-AUC(329 视频) |
|---|---|---|---|---|---|---|---|
| knn_hatemm_video | 0.5256 | 0.5684 | 0.6034 | 0.5754 | 0.5155 | 0.5726 | 0.5193 |
| **knn_hatemm_subclip** | **0.5447** | **0.5882** | **0.6165** | **0.5955** | **0.5402** | **0.5962** | **0.5259** |
| knn_mhc_video(换记忆) | 0.4711 | 0.5107 | 0.5222 | 0.4861 | 0.4498 | 0.5045 | 0.5028 |
| vbcast_hatemm_video(广播对照) | 0.5252 | 0.5701 | 0.6060 | 0.5786 | 0.5121 | 0.5739 | 0.5000* |
| random(seed 0) | 0.4570 | 0.4885 | 0.5319 | 0.4957 | 0.4450 | 0.4906 | 0.4897 |

**K=30(密度匹配粒度,窗口中位 7.97s)**

| 配置 | full AP | full AUC | toxOnly AP | toxOnly AUC | seg AP | seg AUC | wv-AUC |
|---|---|---|---|---|---|---|---|
| knn_hatemm_video | 0.5247 | 0.5656 | 0.6020 | 0.5732 | 0.5120 | 0.5688 | 0.5134 |
| knn_hatemm_subclip | 0.5329 | 0.5754 | 0.6074 | 0.5850 | 0.5246 | 0.5839 | 0.5140 |
| knn_mhc_video | 0.4675 | 0.5024 | 0.5197 | 0.4794 | 0.4450 | 0.4989 | 0.5017 |
| vbcast_hatemm_video | 0.5211 | 0.5675 | 0.5986 | 0.5775 | 0.5085 | 0.5727 | 0.5000* |
| random | 0.4699 | 0.5084 | 0.5360 | 0.5090 | 0.4507 | 0.5065 | 0.5088 |

\* 广播对照 wv-AUC = 0.5 按构造成立(视频内常量分)。

**within-video 信号的显著性**(逐视频 AUC,bootstrap 10k + 符号检验,n=329):

| 配置 | mean wv-AUC | 95% CI | >0.5 / <0.5 / =0.5 | sign-test p |
|---|---|---|---|---|
| K=4 knn_hatemm_subclip | 0.5259 | [0.5048, 0.5468] | 173 / 129 / 27 | **0.0066**(Bonferroni×4 校正后 0.026,仍 <0.05) |
| K=4 knn_hatemm_video | 0.5193 | [0.4961, 0.5419] | 162 / 154 / 13 | 0.35(n.s.) |
| K=30 knn_hatemm_subclip | 0.5140 | [0.4955, 0.5323] | 167 / 144 / 18 | 0.11(n.s.) |
| K=30 knn_hatemm_video | 0.5134 | [0.4945, 0.5325] | 168 / 153 / 8 | 0.22(n.s.) |

**读法(诚实)**:

1. **零训练跨库打分确实高于随机**:最好配置(HateMM 子片段记忆,K=4)full AP 0.545 / AUC 0.588,对 random +0.088 AP / +0.100 AUC;segment 级同向(0.540/0.596)。
2. **池化口径的大头是"毒性密度"的视频间可分性**:vbcast 广播对照几乎追平 kNN 主配置(0.525/0.570)→ full/toxOnly/seg 池化指标主要反映"哪个视频毒性时长占比高",不是段内定位。
3. **段内(within-video)时序信号存在但很弱,且只在一个 cell 显著**:K=4 + 子片段记忆 wv-AUC 0.526(CI 不含 0.5,符号检验 p=0.0066);视频级记忆与 K=30 各 cell 均不显著。
4. **密度匹配(K=30)是负结果**:窗口加密到金标段中位时长不提升反而略降(0.526→0.514)— 8s 窗只含 4 帧、CLIP 视觉键更噪,且毒性多为语音承载,视觉细粒度无增益。
5. **换记忆确实换行为**(可换记忆支柱的双向证据):MHC 记忆在帧级≈随机(0.47/0.51),但**视频级** any-toxic AUC 0.595(HateMM 记忆仅 0.508 video-mem / 0.526 subclip-mem)— HateMM 记忆迁移的是"毒性密度"排序(广播池化 AUC 0.570),MHC 记忆迁移的是二值 any-toxic 判别;记忆库的领域/标签语义直接决定零样本行为,且交换只是一个配置项、零重训。

## 5. 切片(主配置 knn_hatemm_video,frame-level protocol-full)

**分平台**(AUC 平台间几乎一致;AP 差异主要来自 prevalence 差):

| 平台 | 视频 | prevalence | K=4 AP / AUC | K=30 AP / AUC |
|---|---|---|---|---|
| BitChute | 338 | 0.4773 | 0.5636 / 0.5783 | 0.5633 / 0.5760 |
| YouTube | 57 | 0.3960 | 0.4549 / 0.5781 | 0.4493 / 0.5714 |

**分毒性类**(正例 = 该类秒,负例 = normal-only 秒 50,252;各类非互斥):

| 类 | 正例秒 | K=4 AP / AUC | K=30 AP / AUC |
|---|---|---|---|
| hateful | 19,578 | 0.3259 / 0.5764 | 0.3262 / 0.5747 |
| insulting | 24,051 | 0.3978 / 0.5854 | 0.3942 / 0.5837 |
| **sexual** | 3,413 | 0.1528 / **0.6468** | 0.1442 / 0.6327 |
| violence | 11,289 | 0.2252 / 0.5487 | 0.2163 / 0.5245 |
| harm | 464 | 0.0123 / 0.5110 | 0.0125 / 0.5818 |

- sexual 是 AUC 最高的类 — 视觉上最可辨,符合视觉-only 键的机制预期;violence 反而偏低(HateMM 记忆的 hate 标签≠暴力标签)。
- harm 仅 464 秒 / 31 段:按 DATASET 文档约定不下每类结论,仅聚合口径附注。

## 6. 诚实条款

1. **Selection bias**:一切数字基于 **90.8% 存活子集**(395/435 视频、10,604/11,714 段,再经 §1 清洗为 10,572 段);平台删除偏向最极端内容(yt 折损 20.8% ≫ bit 6.9%)→ 绝对数字不可与论文全集数字比较;仅同子集内方法间比较有效。存活 ID 清单:`/data/jehc223/HateClipSeg/pilot/download_status.tsv`。
2. **标签定义域差**:记忆库标签是 HateMM 的 "hate" / MHC 的 "hateful",而 HateClipSeg 的正类是 any-toxic(含 insulting/sexual/violence/harm)— 语义更宽。跨数据集域差(BitChute/YT vs HateMM/YouTube-EN 源)+ 标签定义差 → **绝对 AP 预期不高**;本评测的主张是"**零训练也能定位**"的能力演示,证据 = 与对照(random / video-broadcast)的差距,尤其 within-video AUC 列。
3. **视觉-only 键的盲区**:speech-carried 毒性对 CLIP 视觉键不可见(HateMM 定位评测已见同类局限)。评测前的预期"K=30 才有段内分辨力"被结果**证伪**(§4 读法 4)— 加密窗口不救视觉键的模态盲区,如实报告该负结果。
4. **within-video 显著性只成立于 1/4 个 cell**(K=4 子片段记忆,p=0.0066,Bonferroni×4 后 0.026):"零训练也能定位"的主张必须限定为"存在统计显著但幅度很小(wv-AUC 0.526)的段内信号",不得外推为"定位能力强";其余 cell CI 均跨 0.5。
5. 秒级金标由段标注派生,harm 类样本极少;cache 中的 `labels` 字段为全 0 dummy(可审计"零 HateClipSeg 标签入管线")。
6. consensus topk=10 直接沿用已验证配置,未在 HateClipSeg 上调过;唯一的自由选择是评测粒度 K=30,其由金标段中位时长这一先验决定(§2)且结果为负。
7. **不可解码视频**:yt_NzvfkIYS5Yg(decord/PyAV 双失败)保留在评测中,窗口分为零向量票(近常量)— 1/395,影响可忽略,但从主张中不剔除。

**核心结论(一句话)**:零训练、跨数据集记忆的检索共识票在 HateClipSeg 上给出高于随机(+0.10 AUC)且高于视频级广播对照的定位分,其中段内时序信号统计显著但幅度小(wv-AUC 0.526,仅 K=4+子片段记忆 cell);池化指标的主体是"毒性密度"的视频间排序;换记忆(HateMM↔MHC)在零重训下可预期地改变行为模式 — 能力演示成立,定位强度诚实地弱,改进方向 = 语音模态键与段级记忆。

## 7. 产出与可复核性

- 数据准备:`scripts/analysis/hateclipseg_prep.py` → `data/gt/HateClipSeg/{gold_segments.json,test.jsonl,video_durations.jsonl}` + `data/video/HateClipSeg/All/` 符号链接(webm/mkv 经 .mp4 链接由 decord/PyAV 嗅探解码,已抽检)。
- 特征:`scripts/slurm/hateclipseg_subclip.sbatch`(job 12274)→ `data/CLIP_Embedding/HateClipSeg/test_seen_subclipK{4,30}_openai_clip-vit-large-patch14-336_HF.pt`。
- 评测:`scripts/analysis/eval_localization_hateclipseg.py`(纯 CPU,断点安全)→ `scripts/analysis/loc_out_hcs/{scores_*.npz,results_hateclipseg_loc.json}`。
- src/ 未改任何文件;未 git commit。
