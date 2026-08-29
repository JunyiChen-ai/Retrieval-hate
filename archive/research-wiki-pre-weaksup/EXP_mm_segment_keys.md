# EXP_mm_segment_keys — 多模态片段键修复 EN 共识去噪

**Status:** **DONE — 判定:EN consensus-mm 未超 floor;共识去噪不升级为双语主张;
机制修复(探针层)成立且入归因链** · **Started:** 2026-07-04 · **Finished:** 2026-07-05 ·
**Owner:** subagent (方法开发, src/ 编辑授权)

**一句话结论:** 片段级 ASR 多模态键把 EN 共识 E-step 的 annotator 修好了(正监督供给
56%→19%、投票从视频级变片段级、严重度反相关消除、灾难性 clip-consensus −0.117 F1 被完全
救回 +0.10~0.13),**但训练端仍不超"不用片段监督"的 floor**(final-epoch 3/3 seed 低
−0.012 F1;val-选点 F1 +0.024 由单 seed 运气驱动、±0.088)。EN 的病灶因此被钉死在
**"片段监督通道本身对语音承载仇恨无增益"**,不是投票键/投票空间——归因链闭环,claim
维持 ZH-scoped。ZH mm 探针死,未训练(预注册纪律)。

## 0. 动机与靶点

W2 归因(`scripts/analysis/consensus_forensics.py`,本实验开工时在 MHC-EN 上复算确认):

- EN 仇恨证据以语音/屏幕文字承载为主:hateful train videos 中 speech/text-only 110/168
  (65.5%),+mixed 151/168(89.9%);纯 visual-only 仅 15/168。
- 共识投票实为视频级:mean within-video vote std = **0.0477**(between-video 0.2281)——
  片段键的文本通道是父视频级 text(title+transcript 77-token 截断),四个窗共享,只有视觉通道随窗变化,而视觉通道对语音承载的仇恨是盲区。
- 严重度反相关:真 **Hateful 视频 mean-vote 0.495 < Offensive 0.542**(vote 本应随严重度升)。
- 正监督供给崩塌:**94/168 = 56.0%** hateful 视频全部 4 窗零 ROLE_POS(all-pruned)。

对症方法:给每个 K=4 窗一个**自己的** ASR 转写(Whisper word-level 时间戳对齐到窗),
CLIP text 编码后与帧 CLIP 拼成双通道片段键,共识投票相似度 = (1-w)·cos_img + w·cos_segtext。
w=0.5 时记忆侧键与 clip 空间 round-0 记忆键**完全相同**,唯一变化 = 查询文本通道从"父视频文本"换成"本窗语音"——干净的单变量归因。

## 1. 铁律与验收(已过)

所有新行为在默认关闭 flag 之后(`--consensus_space mm`,默认 `clip`)。

**Bit-for-bit 验收(2026-07-04,PASS):**
- 协议:CPU dry-run,MHC,epochs=2,seed 0,两配置(λ=0 floor;λ=0.5 consensus/clip EM=2),
  production 命令逐参照抄(train_consensus_v2.sbatch)。
- 先证明 harness 本身确定:同代码跑两遍,12/12 ckpt sha1 及全部关键日志行相同。
- 后证明改动无侵入:改动前后(consensus.py 重构出 `build_vote_keys` + mm 分支;run_rac.py
  新增 `--consensus_space mm / --mm_text_weight / --mm_empty_text / --mm_subclip_cache`),
  12/12 ckpt sha1 相同、keylines(roles/Val/Test/flip-rate)相同。
- 现场:scratchpad `dryrun_{before,before2,after}/`(ckpt.sha1 / keylines.txt)。

## 2. 实施

### Phase 1 — 片段级 ASR(Whisper)
- `src/utils/generate_segment_asr_HF.py`:PyAV 解 mp4 音轨→16k mono(免 ffmpeg 二进制);
  transformers pipeline + **openai/whisper-large-v3**(登录节点已预下载,作业内
  HF_HUB_OFFLINE=1);**word-level 时间戳**,词中点落窗分配;
  窗时界 = 与 `generate_subclip_embedding_HF.py` 同一帧采样契约推导
  (M=16 均匀帧,窗 k=帧 [4k..4k+3],窗界=相邻帧时刻中点,即 duration·{3.5,7.5,11.5}/15)。
  word-ts DTW 偶发 crash → 自动降级 sentence-level(记录 `timestamps` 字段)。
  语言强制:MHC→en,MHC_zh→zh。逐视频 resume。
- 产出:`data/ASR/<DS>/{train,dev_seen,test_seen}_asrK4_whisper-large-v3.jsonl`
  (id/duration/audio_ok/timestamps/chunks/window_bounds/window_text)。
- `scripts/slurm/gen_segment_asr.sbatch`;smoke = LIMIT=20(job 12302),全量随后。
- 屏幕文字 OCR:按任务书为加分项,不在本期(E0b 档案 modality_cues 不分段,不用)。

### Phase 2 — 多模态片段键缓存
- `src/utils/generate_subclip_mm_embedding_HF.py`:**不重算视觉**(逐字节复制现有
  subclip 缓存张量),新增 `subclip_txt_feats` [S,768](窗转写 CLIP text pooler,
  与全视频 text 流同一 `encode_text` 分块均值池化;空窗=零向量)+
  `subclip_txt_has_text` [S] bool + `asr_source`。
- 产出:`data/CLIP_Embedding/<DS>/{split}_subclipK4_mm_openai_clip-vit-large-patch14-336_HF.pt`
  (新文件,不覆盖旧缓存)。`scripts/slurm/gen_subclip_mm.sbatch`。

### Phase 3 — 共识投票升级(flag 后)
- `src/utils/consensus.py`:键构造重构为 `build_vote_keys(...)`(E-step 与探针共用同一实现,
  探针 round-0 = 训练 round-0 恒成立);新增 `mm` 分支:
  - query = l2n([√(1-w)·l2n(帧CLIP) | √w·l2n(窗ASR CLIP text)]),投票相似度 =(1-w)cos_img + w·cos_segtext;
  - memory = l2n([√(1-w)·l2n(vid_img) | √w·l2n(vid_txt)])(已含转写;w=0.5 时与 clip round-0 记忆键相同);
  - 空窗文本通道:`--mm_empty_text parent`(默认,回退父视频文本)/ `zero`(纯视觉键);
  - **EM 轮不变**(与 archive 空间同先例;mm 键针对 round-0 监督供给,且训练头没有片段文本输入流可重编码)——与 clip 模式后续轮用 fused-head 空间不同,已如实记录为设计偏离。
- `src/run_rac.py`:flag 定义 + mm 缓存加载(仅 mm 时),载入时对 subclip_parent 与
  subclip_img_feats 与视觉缓存**逐位一致性硬断言**。
- 探针:`scripts/analysis/consensus_probe_mm.py`(不训练,round-0 投票质量,clip vs mm×w 网格)。

**预注册探针 gate(EN,过则进 Phase 4,不过则调 w 或如实报死):**
1. 严重度反相关修复:真 Hateful 视频 mean-vote ≥ Offensive;
2. 正监督供给不崩:hateful 视频 all-pruned(零 ROLE_POS)占比 ≤ clip 空间的 56.0%(94/168)。

### Phase 4 — 训练验证(探针过 gate 才启动)
- `scripts/slurm/train_consensus_mm.sbatch`:GROUP=`RAC_video_consensus_mm`,FORCE=False,
  MHC-EN + MHC_zh × seed {0,1,2},λ=0.5,w=0.5(或探针选出的 w),协议与 kill-ablation 全同
  (warmup≥5 val-selected 主口径,final-epoch 口径一并报)。
- 判定:EN consensus-mm ≥ floor(双口径、3-seed 均值)?ZH 保住/加强?
  对照:floor(12128/12130 及多 seed 版)、consensus-clip(12176/12179)。
- **对照缺口(如实记录)**:CLIP 栈 λ=0 floor 与 consensus-clip 目前均只有 seed-0
  (12128/12130/12176/12179;现存多 seed floor 是 frozen-Qwen 栈的,不可混用)。
  Phase 4 将同时补 CLIP floor seed {1,2} × 双语(同 sbatch,LAMBDA_SEG=0),
  使"mm ≥ floor"是同栈同协议 3-seed 对 3-seed;consensus-clip 多 seed 视 GPU 预算,
  优先级低于 floor(EN clip 已单 seed 硬失败,ZH clip 单 seed 为既有记录)。

### 附加自检(2026-07-04,PASS)
- **归约不变量**:`build_vote_keys` 的 mm(w=0.5, 全空窗, parent 回退)与 clip round-0 键
  逐元素相等(max|Δ| = 6e-8,float32 量级)——w=0.5 时与 clip 的一切差异只来自"有转写的窗"。
- zero 模式 query 范数 ∈ [0.99999976, 1.00000024]。
- 本地 whisper-base CPU 冒烟:word-level 对齐正确;1/2 视频触发 transformers word-ts DTW
  IndexError → 自动 sentence-level 降级路径验证通过。

## 3. 结果

### 3.1 ASR smoke(20 视频 × 双语)— **PASS**(job 12302,3m27s)
- 覆盖:audio_ok 100%/100%(EN/ZH);全部视频有转写;窗文本率 EN 67.3%、ZH 43.8%
  (ZH 低是 Bilibili 音乐/BGM 视频多,13/20 视频仅 1 窗有语音——真实现象非 bug)。
- 时间戳:越界 chunk 0;窗对齐人工抽查(EN 歌词逐窗推进、ZH 医学视频语义连贯)OK。
- 语言:CJK 字符占比 EN 0.000 / ZH 0.937 —— 语言强制生效;ZH 转写质量可用
  (同音字错误如 阴静/阴茎,对 CLIP-text 编码语义影响有限,如实记录)。
- 幻觉:重复 5-gram 标志 0/26、0/20。
- **word-ts 降级率:EN 8/26(30.8%)、ZH 4/20(20.0%)**(transformers word-ts DTW
  IndexError → sentence-level midpoint,落窗更粗;jsonl `timestamps` 字段可审计)。
- 吞吐:~46 视频/3.5min(A100)→ 全量 ~1596 视频估 ~2h。
- QC 工具:`scripts/analysis/asr_qc.py`(已挂入 sbatch 尾部)。
- 全量+QC+mm 缓存 job **12303** 已接续运行(resume 跳过 smoke 已做部分)。

### 3.1b ASR 全量 + mm 缓存(job 12303,COMPLETED)
- ASR 全量:EN 549 train 视频(窗文本率 71.1%,word-ts 降级 41.0%,重复标志 3.1%);
  ZH 579 train(窗文本率 48.5%,降级 26.3%,重复标志 1.9%,CJK 0.906);val/test 同步产出。
- mm 缓存已生成(train/dev_seen/test_seen × EN/ZH),视觉张量与原缓存逐位相同(探针内
  torch.equal 断言过);EN train 1562/2196 窗有文本、ZH 1123/2316。

### 3.2 探针(round-0,零训练)— **EN 过闸(带限定),ZH 不过闸**
探针 = `scripts/analysis/consensus_probe_mm.py`(键构造与训练 E-step 共用 `build_vote_keys`);
JSON:`scripts/analysis/probe_out/consensus_probe_mm_{MHC,MHC_zh}.json` + `_MHC_hi_w.json`。

**MHC-EN**(clip 基线:H=0.495 < O=0.542;all-pruned 94/168=56.0%;wv-std 0.048):

| space | Hate | Off | H≥O | all-pruned | supply≤56% | subclip AUC | wv-std |
|---|---|---|---|---|---|---|---|
| clip | 0.495 | 0.542 | ✗ | 56.0% | ✓ | 0.765 | 0.048 |
| mm w0.5 parent | 0.551 | 0.576 | ✗(−0.025) | **23.8%** | ✓ | 0.762 | 0.101 |
| mm w0.7 parent | 0.566 | 0.587 | ✗(−0.021) | 19.6% | ✓ | 0.737 | 0.114 |
| **mm w0.7 zero** | 0.562 | 0.560 | **✓(+0.002)** | **19.0%** | ✓ | 0.720 | 0.120 |
| mm w0.8 zero | 0.559 | 0.556 | ✓(+0.003) | 19.0% | ✓ | — | 0.121 |
| mm w0.9 zero | 0.547 | 0.546 | ✓(+0.001) | 19.6% | ✓ | — | 0.124 |

- **供给闸门:全配置大幅 PASS**——all-pruned 56.0%→19-24%;speech/text-only 恶意视频
  pos% 35.7→48-49%、drift% 23.2→13-15%:对症机制正中靶点。
- **严重度闸门:仅 zero 模式 w≥0.7 翻正,且是毫米级(+0.001~0.003)**;parent 模式把
  反相关差距从 −0.047 收窄到 −0.021 但不翻。如实定性:反相关"消除"成立、"强反转"不成立。
- 代价:w 增大 → subclip AUC 降(0.765→0.72)、benign 侧 conflict 增(159→318)。
- 票不再是视频级:within-video std 0.048→0.10-0.12(片段键真的在分段投票)。

**MHC_zh:mm 全配置不过供给闸门 —— 如实报死,ZH 不进 mm 训练**:

| space | Hate | Off | H≥O | all-pruned | supply≤55.6% |
|---|---|---|---|---|---|
| clip | 0.527 | 0.431 | ✓ | 55.6% | ✓(基线) |
| mm w0.3-0.7 parent | 0.46-0.48 | 0.38-0.41 | ✓ | 58.3-62.2% | ✗ |
| mm w0.3-0.7 zero | 0.41-0.42 | 0.37-0.38 | ✓ | 68.9-70.6% | ✗ |

- ZH clip 键本就严重度正相关(H 0.527 ≫ O 0.431),mm 反而稀释:窗文本率仅 48.5% +
  CLIP 中文 text 编码弱 → ASR 通道对 ZH 是噪声,drift 率 41.7%→52.8%、正监督塌到 14.9%。
- 结论:**键的语言/证据适配性本身是发现**——EN 证据在语音,片段语音键修复 EN 供给;
  ZH 证据在视觉/屏幕文字(且 CLIP-zh 弱),视觉+标题键已对症。方法主张候选升级为
  "consensus denoising with **evidence-matched segment keys**"(EN=mm, ZH=clip),
  而非单一 mm 空间双语通吃。

### 3.3 训练(Phase 4)— 预注册(提交作业前写定)
- **PRIMARY:MHC-EN consensus-mm,w=0.7,empty=zero,seeds {0,1,2}**
  (唯一双闸门通过的配置族取最小 w;λ=0.5,其余超参与 kill-ablation 全同)。
- **SECONDARY(探索臂,单独报告,不与 PRIMARY 混选):MHC-EN consensus-mm,w=0.5,
  empty=parent,seeds {0,1,2}**——最强 annotator(AUC 0.762、rho 0.471、供给 23.8%)但
  严重度差 −0.025;检验"毫米级 H≥O 翻正"是否是训练收益的必要条件。
- **对照:MHC-EN CLIP floor(λ=0)seeds {1,2}**(seed0=job 12128 已有)→ floor 3-seed 均值。
- ZH:**不提交 mm 训练**(探针死,per 预注册"不硬跑");ZH 主张维持 clip 空间既有记录(12179)。
- 判定标准(不变):EN mm(PRIMARY)3-seed 均值 vs floor 3-seed 均值,val-选点主口径 +
  final-epoch 口径并报;EN consensus-clip(12176,−0.117 F1)为历史对照。
- **已提交(2026-07-04)**:PRIMARY 12310/12311/12312(seed 0/1/2);
  SECONDARY 12313/12314/12315(seed 0/1/2);EN floor λ=0 12316/12317(seed 1/2)。
  GROUP=`RAC_video_consensus_mm`,FORCE=False,sbatch=`scripts/slurm/train_consensus_mm.sbatch`。

### 3.4 训练结果与终判(2026-07-05,全部 8 作业 COMPLETED,0 Traceback)

**逐 seed(Test macro-F1 / acc;val-选点 warmup≥5 主口径,final-epoch 并报):**

| arm | seed | job | selEp | val-sel F1/acc | final-ep F1/acc |
|---|---|---|---|---|---|
| floor λ=0 | 0 | 12128(既有) | 26 | 0.7113 / 0.7826 | 0.7145 / 0.7640 |
| floor λ=0 | 1 | 12316 | 16 | 0.6034 / 0.7329 | 0.7159 / 0.7826 |
| floor λ=0 | 2 | 12317 | 27 | 0.6997 / 0.7702 | 0.7303 / 0.7888 |
| PRIMARY mm w0.7/zero | 0 | 12310 | 27 | 0.6997 / 0.7702 | 0.7086 / 0.7702 |
| PRIMARY mm w0.7/zero | 1 | 12311 | 28 | 0.7283 / 0.7826 | 0.7086 / 0.7702 |
| PRIMARY mm w0.7/zero | 2 | 12312 | 14 | 0.6598 / 0.7205 | 0.7086 / 0.7702 |
| SECONDARY mm w0.5/parent | 0 | 12313 | 22 | 0.6454 / 0.7329 | 0.6783 / 0.7578 |
| SECONDARY mm w0.5/parent | 1 | 12314 | 27 | 0.6997 / 0.7702 | 0.6929 / 0.7578 |
| SECONDARY mm w0.5/parent | 2 | 12315 | 14 | 0.6845 / 0.7391 | 0.7144 / 0.7764 |

**3-seed mean±std 与同 seed 配对 Δ(vs floor):**

| lens | arm | F1 | acc | 配对 ΔF1(逐 seed) | 配对 Δacc |
|---|---|---|---|---|---|
| val-选点 | floor | 0.6715±0.0592 | 0.7619±0.0259 | — | — |
| val-选点 | PRIMARY | 0.6959±0.0344 | 0.7578±0.0329 | **+0.0245±0.0881**(−0.012/+0.125/−0.040) | −0.0041±0.0502 |
| val-选点 | SECONDARY | 0.6765±0.0280 | 0.7474±0.0200 | +0.0051±0.0830 | −0.0145±0.0458 |
| final-ep | floor | 0.7202±0.0087 | 0.7785±0.0129 | — | — |
| final-ep | PRIMARY | 0.7086±0.0000 | 0.7702±0.0000 | **−0.0116±0.0087(3/3 为负)** | −0.0083±0.0129(2/3 负) |
| final-ep | SECONDARY | 0.6952±0.0182 | 0.7640±0.0107 | −0.0250±0.0103(3/3 负) | −0.0145±0.0095(3/3 负) |

**判定(预注册标准:EN consensus-mm ≥ floor,双口径 3-seed 均值):FAIL。**
- final-epoch(selection-free):PRIMARY 3/3 seed 低于 floor(ΔF1 −0.0116±0.0087)——
  同方向一致的小负效应,不是噪声摆动。
- val-选点:ΔF1 +0.0245 完全由 seed1 一格驱动(floor seed1 val-选点崩到 0.6034,
  PRIMARY 该格 +0.125),配对 std ±0.088 远大于效应;acc 双臂均 ≤ floor。
  又一次证实本仓库方法学结论:78 样本 dev 选点噪声支配 ≤2 点的"增益"。
- SECONDARY < PRIMARY(双口径)——与探针排序一致(严重度闸门有筛选力,探针作为
  配置选择器这次预测对了训练端的臂间排序)。

**如实记录的正面发现(进归因链,不进主表 claim):**
1. **灾难救回**:consensus-clip EN(12176)0.5948/0.7329 → consensus-mm ≈0.70/0.77
   (val-选点,+0.10~0.13 F1)。键换对了,共识不再毒化训练。
2. **机制层修复全部按探针预测兑现**:训练 round-0 role 表与探针逐位一致
   (668/778/305/127/318);EM flip-rate round≥1 恒 0(设计使然)。
3. **PRIMARY final-epoch 三 seed 混淆矩阵完全相同**(F1/acc/P/R 四指标逐位同,仅 ROC
   异:0.8271/0.8353/0.8304)——kNN 多数票离散化把 seed 方差压到 0,是检索头
   稳定性的有趣佐证(如实记录,未深挖)。
4. **EN 病灶定位收紧**:键修好后片段监督仍 ≤ floor ⇒ 失败根因不是投票空间
   (W5 已排除档案空间)也不是键模态(本实验排除),而是**片段监督通道本身**
   在语音承载仇恨上无增益——与 kill-ablation 里 selfscore 同败的伏笔闭环。
   这直接堵死"你们只是键选得差"的审稿人质疑,EN scoping 论证更硬。

## 5. 最终定位(回答任务书判定问题)

- **"共识去噪能否升级为双语方法主张?"——不能。** EN 训练端未超 floor(双口径),
  ZH mm 探针死未训练。共识去噪 claim 维持 **ZH/视觉承载仇恨 scoped**(12179 记录不变)。
- 本实验的论文价值在**归因章节**:三段式证据链
  (视觉键投票=视频级噪声 → 档案/混合空间救不回 → 证据匹配的片段语音键把 annotator
  全面修好但下游仍无增益)+ "evidence-matched segment keys" 的探针方法学
  (probe-before-train,含预注册双闸门与 ZH 反例)。
- 主表:EN 侧无翻盘;MHClip-EN 维持"近天花板 + 归因分析"叙事(MORNING_REPORT §6.2)。

## 4. 偏离与诚实条款记录

- mm 空间 EM 轮不变(clip 模式后续轮为 fused-head 空间)——设计选择,见 Phase 3;
  archive 空间已有同先例。flip-rate 在 round≥1 恒 0,EM=2 实为"同伪标签重训一次"。
- 窗时界按帧采样契约推导(非均匀四等分,差异微小:0.233/0.5/0.767 vs 0.25/0.5/0.75)。
- ASR 对齐粒度:whisper word-level;个别视频降级 sentence-level(midpoint 落窗,长句会整句
  落单窗)——jsonl 有 `timestamps` 字段可审计占比。
- Whisper 对纯音乐/无语音片段可能幻觉转写;未做幻觉过滤(探针/训练若受害,将在结果节如实报)。
- ZH 窗转写用 CLIP text 编码:CLIP 对中文弱,但与现有 ZH 全视频文本通道**一致**(同弱),
  不引入新的不对称。
- EN floor seed{1,2}(12316/12317)经 `train_consensus_mm.sbatch` 以 LAMBDA_SEG=0 运行:
  λ=0 时 segment/consensus 全路径 inert(Phase-0 bit-for-bit 已验证该保证),与 12128 同协议。
- ASR 全量 word-ts 降级率(EN 41.0%)高于 smoke(30.8%);降级视频窗对齐粒度更粗,
  未做逐视频剔除或修复(结果端 mm 训练已判 FAIL,不影响结论方向;若日后复用 ASR 资产
  应先修 transformers word-ts DTW bug 或换 whisperX 对齐)。
- 探针 w 网格在 EN 扩到 {0.3,0.5,0.7,0.8,0.9}×{parent,zero} 共 10 配置后选 PRIMARY——
  属预注册允许的"调 text_weight";训练臂在提交前已在 §3.3 预注册为 PRIMARY/SECONDARY
  两支,未做训练后挑臂。
