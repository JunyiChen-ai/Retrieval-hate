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

## 4. 主表(frame-level AP / AUC;segment-level;within-video)

TBD(job 12274 完成后由 `scripts/analysis/eval_localization_hateclipseg.py` 填入)

## 5. 切片(主配置 knn_hatemm_video)

TBD:分平台(bit/yt)、分毒性类(正例 = 该类秒,负例 = normal-only 秒;harm 仅 464 秒/31 段 → 只作聚合口径附注,不下每类结论)。

## 6. 诚实条款

1. **Selection bias**:一切数字基于 **90.8% 存活子集**(395/435 视频、10,604/11,714 段,再经 §1 清洗为 10,572 段);平台删除偏向最极端内容(yt 折损 20.8% ≫ bit 6.9%)→ 绝对数字不可与论文全集数字比较;仅同子集内方法间比较有效。存活 ID 清单:`/data/jehc223/HateClipSeg/pilot/download_status.tsv`。
2. **标签定义域差**:记忆库标签是 HateMM 的 "hate" / MHC 的 "hateful",而 HateClipSeg 的正类是 any-toxic(含 insulting/sexual/violence/harm)— 语义更宽。跨数据集域差(BitChute/YT vs HateMM/YouTube-EN 源)+ 标签定义差 → **绝对 AP 预期不高**;本评测的主张是"**零训练也能定位**"的能力演示,证据 = 与对照(random / video-broadcast)的差距,尤其 within-video AUC 列。
3. **视觉-only 键的盲区**:speech-carried 毒性对 CLIP 视觉键不可见(HateMM 定位评测已见同类局限);K=4 粗粒度窗口(≈60s)大于金标段中位(8.1s)8 倍,预期 K=30 才有段内分辨力。
4. 秒级金标由段标注派生,harm 类样本极少;cache 中的 `labels` 字段为全 0 dummy(可审计"零 HateClipSeg 标签入管线")。
5. consensus topk=10 直接沿用已验证配置,未在 HateClipSeg 上调过;唯一的自由选择是评测粒度 K=30,其由金标段中位时长这一先验决定(§2)。

## 7. 产出与可复核性

- 数据准备:`scripts/analysis/hateclipseg_prep.py` → `data/gt/HateClipSeg/{gold_segments.json,test.jsonl,video_durations.jsonl}` + `data/video/HateClipSeg/All/` 符号链接(webm/mkv 经 .mp4 链接由 decord/PyAV 嗅探解码,已抽检)。
- 特征:`scripts/slurm/hateclipseg_subclip.sbatch`(job 12274)→ `data/CLIP_Embedding/HateClipSeg/test_seen_subclipK{4,30}_openai_clip-vit-large-patch14-336_HF.pt`。
- 评测:`scripts/analysis/eval_localization_hateclipseg.py`(纯 CPU,断点安全)→ `scripts/analysis/loc_out_hcs/{scores_*.npz,results_hateclipseg_loc.json}`。
- src/ 未改任何文件;未 git commit。
