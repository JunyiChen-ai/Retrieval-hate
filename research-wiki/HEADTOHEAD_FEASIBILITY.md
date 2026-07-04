# 头对头比较可行性侦察报告(MoRE / MultiHateLoc / CRAVE / HateClipSeg)

_日期:2026-07-03。范围:只侦察不跑训练。方法:GitHub API 文件树 + raw 文件核读 + ICCV PDF 全文(pdftotext)+ arXiv v3 协议核对 + 本地 split 文件逐行 diff。_

---

## 0. 核心发现(先说结论)

1. **切分已天然对齐,同场比较的最大障碍不存在。** 本地 `/data/jehc223/HateMM/splits/` 与 `/data/jehc223/Multihateclip/{English,Chinese}/splits/` 的 `train/valid/test.csv` 与 MoRE 官方仓库 `data/*/vids/*.csv` **逐行 diff 完全一致**(HateMM 757/109/217;MHClip-EN 701/100/200;MHClip-ZH 699/101/200)。CRAVE 仓库的 HateMM / MHClipEN split 也与 MoRE **逐字节相同**(同组复用)。即 MoRE、CRAVE、我们三家用的是同一套 70/10/20 切分。
2. **唯一偏差是我们的 `_clean` 子集**(剔除本地缺失视频,`prep_mhc.py` / `prep_video_dataset.py` 均取 `*_clean.csv`):
   - HateMM:train 744/757、test 215/217 —— 差 <1%,基本无害;
   - **MHClip-EN:test 161/200(缺 19.5%)、train 550/701(缺 21.5%)**;
   - **MHClip-ZH:test 149/200(缺 25.5%)、train 579/699(缺 17.2%)**。
   → 在 MHClip 上,直接拿我们的数字对 MoRE/CRAVE 的**发表数字**有可比性硬伤(test 集缺 1/5~1/4)。这是"必须亲手复跑 MoRE(在 clean 子集上)"的最强论据。
3. **MoRE 代码完整可复跑(天级);MultiHateLoc 官方仓库是空占位符(仅 LICENSE);CRAVE 代码完整但它才是 HateMM/MHClip-EN 的最高发表数字持有者;HateClipSeg 只发标注不发代码。**

---

## 1. MoRE(WWW 2025,头号比较对象)

### 代码状态
- 官方仓库 [Jian-Lang/MoRE](https://github.com/Jian-Lang/MoRE),17 commits,最近活动 2026-05-25(改标题),2025-12-17 有实质更新("Update code for making retrieval embedding")。issues 仅 1 个已关闭(加 bib)——无复现投诉,但也说明**尚无人公开复现**。
- **齐备**:训练/评测入口 `src/main.py` + Hydra 配置(`HateMM_MoRE.yaml` / `MHClipEN_MoRE.yaml` / `MHClipZH_MoRE.yaml`);模型 `src/model/MoRE/model/MoRE.py` + `submodule.py`;三数据集 dataloader;预处理全套脚本(16 帧抽取、PaddleOCR 中英、Whisper-v3 ASR、`google/vit-base-patch16-224` 帧特征、BERT 文本特征、librosa MFCC);检索全套(`retrieve/make_retrieval_result.py`:三模态 cosine 相似度直接相加、按 label 各取 top-100、ignore_self——与论文描述一致)。
- **不提供**:预提特征、checkpoint、原始视频(版权,给 ID)。依赖轻(torch/transformers/hydra 等,py312)。
- 瑕疵:README run 命令写的是 "Run **ExMRD** for the HateMM dataset"(同组另一项目的复制残留);`extract_text_caption.py` 只做 BERT 编码,**`caption.jsonl` 本身如何生成仓库未文档化**(复跑风险点 #1)。旁证:本地 `annotation(new).json` 的 `Frames_description/Text_description/Mix_description` 字段与该管线格式吻合,本地已有的这份文件很可能就覆盖了 caption 输入,需开工时核对字段映射。
- README 声称提供 "temporal and five-fold splits",但仓库实际只有单一 70/10/20 切分(措辞残留,不影响使用)。

### 复跑路径
1. 视频本地已有(缺失视频恰好 = clean 差集);MHC transcript 官方自带,HateMM transcript 本地已有 → Whisper 步骤可大部分跳过。
2. 特征:16 帧 × ~2,900 视频的 ViT-base 前向 + BERT 文本 + MFCC —— 单卡小时级;PaddleOCR 中英是环境安装上最麻烦的一步(风险点 #2,集群装 paddle 常踩坑)。
3. 检索构建:CPU 分钟级。
4. 训练:模型很小(FFN experts + 双极注意力 + soft router),作者单卡即可;三数据集 × 多 seed 一晚可完成。SLURM 化改造量小(纯 Hydra 单机脚本)。

### 预计成本与风险
- **定级:天级,可行。乐观 1–2 天(caption 字段能对上、OCR 顺利),现实 2–4 天。**
- 风险:(a) caption.jsonl 来源未文档化——可用本地 annotation(new).json 字段替代或发 issue 问作者;(b) PaddleOCR 环境;(c) 论文报 p<0.01 应为多 seed,但 yaml 里 seed 协议未核实,复跑数字与发表数字可能有 ±1–2 pt 浮动;(d) 我们在 clean 子集上复跑出的 MoRE 数字会与其全量发表数字不同 → 论文里应**两套并列**(发表数字为参考 + 同 clean 子集复跑数字为主对比)。
- 额外收益:一旦复跑通,可把 MoRE 塞进我们的 temporal split 与跨数据集矩阵(它的 memory bank 结构天然支持换库),让"updatable kNN memory"的对比在 MoRE 身上也同场——这是发表数字永远给不了的。

---

## 2. MultiHateLoc(WWW 2026,方向 A 的定位 baseline)

### 代码状态
- 官方仓库 [Multimodal-Intelligence-Lab-MIL/MultiHateLoc](https://github.com/Multimodal-Intelligence-Lab-MIL/MultiHateLoc):**空占位符**。创建于 2026-01-28,仅 1 commit、只有 LICENSE 文件,创建当天后再无 push(核查于 2026-07-03)。论文正文写 "Code is available at ..." 但实际未放码。无权重、无特征、无标注处理脚本。
- 同组(Exeter, Zeyu Fu)的前作 HatefulVideoLabelNoise(ACM MMWS 2025)有真实放码记录 → 后续放码有希望,值得邮件催问(顺带要 splits 和 mAP 计算脚本)。

### 评测协议可复现性(arXiv 2512.10408v3 全文核对)
- 特征端**已明确**:ViT-B/16 帧特征(768d)、VGGish 1s 粒度(128d,插值对齐)、Whisper 转写 + BERT 句级编码 —— 均为标准组件,可自行复刻。
- **关键口径论文未写明**:train/val/test 切分未公布;帧粒度/fps 未写;frame-level mAP/AUC 的计算范围(只在 hateful test 视频上算,还是含 non-hateful;逐帧展平还是逐视频平均)未写;baseline("同一特征与训练管线重实现" VAD-CLIP/Early/Late/CMFusion)细节未给。
- GT 来源:HateMM 帧级 GT = 原生 hate-snippet 时间戳(在 Zenodo 的 `HateMM_annotation.csv`;**本地只有 annotation(new).json,无 span 字段,需从 Zenodo 补一个小 CSV**,我们已有数据访问权);MHC 的 segment 标签 = MultiHateClip 官方标注,本地同样未存,需从官方渠道补。
- 我方基建:W2 的 HateMM subclip 基建已就绪(`/data/jehc223/HateMM/quad/` 每视频 4 subclip + `src/utils/generate_subclip_embedding_HF.py`),支撑**我们自己口径**的定位评测没问题;但对齐他们 1s-frame 粒度需要更细的帧级特征(重新抽帧,量不大)。

### 复跑路径与成本
- (a) **等码/邮件作者**:成本不可控,回复概率中等;(b) **自实现其协议**:2–3 天(补 span GT、1s 粒度特征、mAP 口径做 2–3 种敏感性),但口径是猜的,审稿人可挑"协议不一致";(c) **发表数字 + 协议对照表**:0.5 天,当前最稳。
- **定级:代码路线目前不可行(无码);协议自实现天级但带不可比风险。**

---

## 3. CRAVE(ICCV 2025,MoRE 同组)

### 代码状态
- 官方仓库 [ronpay/CRAVE](https://github.com/ronpay/CRAVE):1 commit(squash 发布)但**结构完整**:`preprocess/`(32 帧→聚类 10 关键帧、16 帧特征、OCR、transcript)、`retrieve/`(CLIP 跨域检索)、`src/model/CRAVE/`(模型 + 4 数据集 yaml + loss)、`run/*.sh`、requirements.txt。无特征、无 checkpoint、原始数据给 ID。
- 编码器:`clip-vit-large-patch14`;单卡 RTX 4090 可训。外部图文语料:**FHM(hate 轨道)/ Fakeddit(rumor 轨道)**,需额外下载。

### 协议与可比性(ICCV PDF 全文核对)
- 指标 ACC / M-F1 / M-P / M-R,binary,与 MoRE 同口径;**HateMM、MHClipEN 切分与 MoRE 逐字节相同(已 diff 验证)= 与我们同切分**。
- **发表数字:HateMM ACC 87.09 / M-F1 86.51;MHClipEN ACC 82.50 / M-F1 79.81** —— 全面高于 MoRE(83.41/82.35;77.50/75.19)。**CRAVE 才是这两个切分上当前最高发表数字**,我们的对比表必须把它列进 in-dataset SOTA 行,不能只对 MoRE。
- 与我们跨数据集矩阵的可比性:**没有同场轨道**。CRAVE 的 "cross-domain" 是 image-text meme 语料 → video 的**训练期增强**(检索库训练时固定),不做 train-on-A-test-on-B 的 video→video 泛化;其 low-resource 实验是 5%/10%/20% 训练子集缩减,与我们的 updatable-memory / evolving 协议不同轴。它只在 in-dataset binary 轨道上与我们相遇。
- 未评测 MHClip-ZH —— ZH 轨道上 MoRE 仍是唯一检索系对比对象。

### 复跑成本与风险
- 复跑量级与 MoRE 相当偏高(多出 FHM+Fakeddit 下载与 CLIP-L 特征、跨域检索对):**2–4 天**。
- 策略风险:它比 MoRE 强 ~4 pt M-F1。若我们最终数字过不了 CRAVE,需在正文明确划界(方法族不同:外部图文语料增强 vs in-domain 检索对比 + kNN + 可更新记忆;CRAVE 无 kNN 推断、无库更新、无跨数据集)。**建议先引用发表数字,复跑预算等 MoRE 头对头出结果后再决定。**

---

## 4. HateClipSeg(ACM MM 2025,弱监督赛道外部验证)

### 代码状态
- 仓库 [Social-AI-Studio/HateClipSeg](https://github.com/Social-AI-Studio/HateClipSeg) 是**纯数据发布**:`Dataset/video_level_annotation.csv` + `Dataset/segment_level_annotation.csv`(multi-hot 6 类 × 精确到 0.01s 的 [start,end] 时间戳)+ `lexicons.json` + README。**无评测脚本、无 baseline 代码、无官方 splits、无视频/特征下载渠道**。
- 视频 ID 带平台前缀(`yt_` YouTube / `bit_` BitChute),需自爬;BitChute 内容灭失风险高(数据集 ~435 视频:380 offensive + 55 normal,11,714 segments)。

### 复跑路径与成本
- 论文三任务(trimmed 分类 / 时序定位 / online 分类)协议只能从论文抄,定位评测脚本(frame-mAP / mAP@tIoU)自写半天级——反而是全链路里最便宜的部分。
- 主要成本 = 拿视频:yt-dlp 爬取 1–2 天 + 灭失率不可控。**建议先做 50 视频小试点统计存活率**,再决定投入。
- 定位:它是数据集论文,"头对头"对象是其全监督 baseline 数字;我们弱监督赛道先到先得(与 NOVELTY_CHECK_dirA §4.2-3 一致)。

---

## 5. 优先级建议表

| 对象 | 头对头方式 | 建议 | 预计成本 | 一句话理由 |
|---|---|---|---|---|
| **MoRE** | **复跑官方代码 on 我们的 clean 子集**,再扩展进 temporal split / 跨数据集矩阵 | **值得做,第一优先** | 2–4 天 | 代码全、切分与我们同源已 diff 验证;MHClip test 缺 20–25% 使"直接引发表数字"有硬伤,只有复跑才有干净同场;复跑通后还能白得 evolving 协议下的 MoRE 对照 |
| **CRAVE** | 发表数字 + 协议对照表(同切分同指标,可直接引);复跑缓办 | 引数字为主,复跑视 MoRE 结果再定 | 引用 0.5 天 / 复跑 2–4 天 | 与我们同切分可直接引;但它是 HateMM/MHClip-EN 最高发表数字(M-F1 86.5/79.8),先确认我们的落点;跨数据集矩阵无同场轨道 |
| **MultiHateLoc** | 发表数字 + 协议对照表;并行邮件作者要码/splits/评测脚本 | 暂用对照表,不要现在自复现 | 对照表 0.5 天;自实现协议 2–3 天(带不可比风险);等码不可控 | 官方 repo 自 2026-01-28 起只有 LICENSE;mAP 口径/切分论文未写明,自复现会被挑"协议不一致" |
| **HateClipSeg** | 自写评测脚本 + 自爬视频,作外部验证金标(非头对头) | 做 50 视频存活率小试点 | 爬取 1–2 天 + 脚本 0.5 天 | 无官方评测代码,协议从论文抄即可;segment 标注质量高(α=0.817),是方向 A 唯一的弱监督外部金标 |

### 建议执行顺序
1. **MoRE 复跑**(先核对本地 annotation(new).json 与其 caption 管线字段映射,PaddleOCR 环境提前踩);
2. 同步发两封邮件:MultiHateLoc 要码,MoRE 问 caption.jsonl 生成方式(如字段对不上);
3. CRAVE / MultiHateLoc 先进协议对照表(附我们 clean 子集与全量切分差异声明);
4. HateClipSeg 存活率试点,活率 >70% 再立项爬全量。

---

## 附:证据与本地路径
- Split 一致性验证:MoRE/CRAVE CSV 下载于 scratchpad(`more_splits/`, `crave_*.csv`),与 `/data/jehc223/HateMM/splits/`、`/data/jehc223/Multihateclip/{English,Chinese}/splits/` sort 后 diff 为空。
- 我们的 gt 实际规模:`/data/jehc223/RGCL/data/gt/` — HateMM 744/107/215,MHC 549/80/161,MHC_zh 579/78/149(`_clean` 子集)。
- CRAVE 论文全文:ICCV open-access PDF(scratchpad `crave.txt`);MultiHateLoc 协议:arXiv 2512.10408v3。
- 相关既有笔记:`research-wiki/papers/lang2025_biting_off_more.md`、`sun2025_multihateloc_towards_temporal.md`、`wang2025_hateclipseg_segmentlevel_annotated.md`、`research-wiki/NOVELTY_CHECK_dirA.md`(CRAVE §2.8)。
- 我方已有对照基线:`ITERATION_LOG.md` 已按 MoRE 协议对齐过一版(frozen-CLIP floor:HateMM M-F1 0.8172 vs MoRE 0.8235;MHC-EN 0.6219 vs 0.7519;MHC-ZH 0.7706 vs 0.7475)——注意这版是 clean 子集数字,正是本报告指出需要复跑 MoRE 才能转正的对比。
