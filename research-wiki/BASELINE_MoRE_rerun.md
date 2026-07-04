# MoRE(WWW 2025)同场复跑日志

_目标:复跑 [Jian-Lang/MoRE](https://github.com/Jian-Lang/MoRE) 官方代码,产出 (a) 官方 split 完整 test 的 sanity 数字,(b) 限制到我们 `_clean` 测试子集(HateMM 215 / MHClip-EN 161 / MHClip-ZH 149)的严格同场数字。上游侦察:`HEADTOHEAD_FEASIBILITY.md`。不发邮件、不等作者;缺失件一律本地复原并如实标注。_

---

## 阶段 1:环境 + 数据脚手架 + 缺失件核验(2026-07-03)

### 1.1 隔离环境 ✅
- 代码:克隆到 `/data/jehc223/baselines/MoRE`(RGCL 仓库外,不动 HateVideo env)。
- **`MoRE_env`**(conda,python 3.12,按其 README):torch 2.12.1+cu130、**transformers 4.49.0、pandas 2.2.3(主动降级** ——pip 默认装到 transformers 5.12/pandas 3.0,与其 2024/25 年代 API 不兼容:`pd.set_option('future.no_silent_downcasting')` 在 pandas 3 已删,会让其 `main.py` 启动即崩)+ hydra-core/torchmetrics/loguru/librosa/easyocr/scikit-image/jieba/Levenshtein/autocorrect/einops 等。**einops 是其模型的实际依赖但 README 漏列**。ffmpeg 6.1.2 装进 env(conda-forge ffmpeg 8 有 libjxl 断链,降到 6 可用)。
- **`MoRE_paddle`**(conda,python 3.10):paddlepaddle-gpu 2.6.2 + paddleocr 2.7.3,仅供 ZH OCR。踩坑记录:paddle 与 Cython 冲突(zlib 解压错误)→ **卸载 Cython 后可导入**;numpy 必须 <2(ABI)。CPU 冒烟通过;GPU 路径留在 SLURM 作业内验证,失败则自动回退 easyocr ch_sim(`rerun/slurm/g2_ocr_zh.sbatch` 内建 fallback,若触发将在此文档标注)。
- HF 模型预下载:`google/vit-base-patch16-224`、`google-bert/bert-base-chinese`、`briaai/RMBG-1.4`(bert-base-uncased、Qwen2.5-VL-7B-Instruct 本地已有)。easyocr en/ch_sim 权重已缓存。

### 1.2 caption.jsonl 核验:侦察猜想不成立,启用我方复原 ⚠️
- **实测**:三个数据集本地 `annotation(new).json` 的 `Frames_description` / `Text_description` / `Mix_description` 字段**全部为空**(0/1066、0/891、0/897)。侦察报告"本地字段可能就是其 caption 输入"的猜想被否定。
- MoRE 仓库对 caption.jsonl 的生成方式**零文档**(唯一线索是 OCR 脚本里注释掉的 `# caption = image_to_caption(video_frames)`);caption 仅用于**检索记忆库 vision 分支**(BERT 编码后与 3 组 ViT 特征拼接),不进主模型输入。
- **处置:caption 来源为我方复原** —— 本地 Qwen2.5-VL-7B-Instruct 对同一套 16 帧生成 2-3 句视觉描述(EN 数据集英文、ZH 数据集中文,与其 bert-base-chinese 编码器匹配),贪心解码。脚本 `rerun/gen_caption_qwen.py`。**最终报告所有数字都带此脚注。**

### 1.3 其余缺失件盘点与来源
| 缺失件 | 状态 | 来源/处置 |
|---|---|---|
| `HateMM_annotation.csv`(官方) | ✅ 本地已有 | `/data/jehc223/RGCL/data/gt/HateMM/HateMM_annotation.csv`(431 Hate/652 Non Hate,1083 全);与文件名前缀推导标签 1083/1083 一致后采用 |
| MHClip `annotation/{split}.tsv` | ✅ 重建 | 由 `annotation(new).json` 的 Label 字段生成(Video_ID+Majority_Voting);EN 覆盖 890/1000、ZH 897/1000,缺标签视频自动落出(与其 `isin` 过滤行为一致) |
| `speech.jsonl` / `title.jsonl` | ✅ 重建 | annotation(new).json 的 Transcript/Title 字段(HateMM 无标题→空串;缺失转写→空串) |
| `label.jsonl`(检索库标签) | ✅ 重建 | 同上,binary 映射 Hateful/Offensive→1, Normal→0(其 MHClipEN_base.py 的官方映射) |
| `ocr.jsonl` | 🔄 生成中 | EN 数据集用 easyocr(**其官方 EN 协议本来就是 easyocr**,只有 ZH 用 PaddleOCR——侦察报告"PaddleOCR 中英"有误);ZH 用 paddle,fallback easyocr |
| `caption.jsonl` | 🔄 生成中 | 我方复原(见 1.2) |
| 视频/帧/音频特征 | 🔄 生成中 | 按其 preprocess 脚本(16 帧 ffmpeg、ViT-base CLS、MFCC-128、RMBG-1.4 前后景) |

### 1.4 数据覆盖率(决定两套评测的口径上限)
| | 官方 split 总量 | 有标签 | 有视频 | test 有标签 | test 有视频 | our clean test |
|---|---|---|---|---|---|---|
| HateMM | 757/109/217 | **1083/1083** | **1083/1083** | 217/217 | 217/217 | 215 |
| MHClip-EN | 701/100/200 | 890 | 792 | 182/200 | 162/200 | 161 |
| MHClip-ZH | 699/101/200 | 897 | 814 | 176/200 | 157/200 | 149 |

→ **sanity (a) 在 HateMM 上完全成立**(全量视频+官方标签)。MHClip 的 (a) 实际是"官方 test ∩ 有标签"(EN 182、ZH 176),其中无视频者按 MoRE 自己的缺失协议(黑帧/零音频/空文本)处理——与其发表值比较时要带此保留。(b) 干净同场数字不受影响。

### 1.5 释出代码问题清单(复跑期间发现,均已文档化处置)
1. **README 漏依赖**:einops;requirements 无文件,版本全靠猜(我们锁 transformers 4.49/pandas 2.2.3)。
2. **`retrieve/merge_feature.py` audio 循环 bug**:循环体内从不读取当前模态特征(缺 `fea = fea_dir[vid]`),导致"audio 记忆库"实际是 caption 嵌入的两份拷贝。**主跑严格按释出代码(bug 保留,variant=asreleased);另跑 bugfix variant 做敏感性**(仅此一行差异,variant=bugfix)。
3. **`merge_feature.py` 的 torch.save 在 vid 循环体内**(O(n²) 写盘,数小时纯浪费)→ 复跑副本把 save 提到循环外,**最终文件内容不变**(纯性能)。
4. **释出特征脚本与释出模型代码形状不一致**:extract 脚本存 text (768,)/mfcc (128,) 1-D,但模型 forward 要求主 text/audio 特征为 (1,d)(expert attention 把它当长度 1 序列,router `squeeze(1)`;其 submodule 里未被调用的 `check_shape` 也印证 (1,d) 约定)。1-D 直接喂会在 einops rearrange 崩掉——**其释出预处理脚本原样跑不通其释出训练代码**。处置:`rerun/fix_fea_shapes.py` 把主特征 unsqueeze 到 (1,d)(检索 merge 输出不受影响,幂等)。
5. `extract_frames.py` 的 main() 写 32 帧,但全管线(dataloader/特征名)均消费 `frames_16` → 按 16 帧跑。
6. README 的 run 命令写 "Run ExMRD"(同组项目复制残留),实际入口正确。
7. hydra 行为实测:hydra 1.3 + `version_base=None` 不 chdir,从仓库根启动即可,相对路径成立。

### 1.6 复跑管线(全部 SLURM,已提交,依赖链挂好)
| Job | 内容 | 依赖 | JobID |
|---|---|---|---|
| P1 `more_p1_frames` | 16 帧抽取 ×3 + wav→MFCC ×3(逐数据集,MFCC 后立删 wav,磁盘纪律) | – | 12235 |
| G1 `more_g1_ocr_en` | easyocr EN OCR(HateMM+MHClip-EN),断点续跑 | – | 12236 |
| G2 `more_g2_ocr_zh` | PaddleOCR ZH(GPU→CPU→easyocr 三级 fallback) | – | 12237 |
| G3 `more_g3_caption` | Qwen2.5-VL-7B caption 复原 ×3 | P1 | 12238 |
| G4 `more_g4_vit` | RMBG-1.4 前后景 + ViT 特征 ×9(前后景帧用完即删) | P1 | 12239 |
| G5 `more_g5_textfea` | 全部 BERT 文本特征(跑其原始脚本) | G1,G2,G3 | 12240 |
| C2 `more_c2_retrieval` | 形状调和 + merge + 检索(asreleased/bugfix 两 variant) | P1,G4,G5 | 12241 |
| G6 `more_g6_train` | 官方配置训练 ×3 数据集 ×2 variant(seed=2024 其默认,不调参)+ 双轨评测 | C2 | 12242 |

- 训练超参:全按其官方 yaml(HateMM num_pos/neg=30/90 α=0.8;EN 10/30 α=0.7;ZH 10/30 α=0.8;AdamW 5e-4/5e-5,bs=128,50 epoch,patience=5 早停 on val acc,seed=2024 单种子——其发表 p<0.01 的多 seed 协议仓库未释出)。
- 磁盘:开工时 quota 252/290G;峰值增量约 +20G(wav/前后景帧均用完即删)。视频不复制(symlink 到本地数据)。
- 评测脚本 `rerun/eval_subset.py`(我方新增,不改其代码):同一 checkpoint 上算 (a) 官方 test 全量(其 torchmetrics 口径:ACC/M-F1/M-P/M-R)与 (b) test∩clean 子集,并落每视频预测供审计。

### 下一步
- 等 SLURM 队列(前面有既有作业,held→自动放行)。逐阶段核对产物后追加进度。
- 待产出:阶段 2(特征完成度)、阶段 3(训练+两套数字 vs 发表值 HateMM ACC 0.8341/M-F1 0.8235;MHClip-EN 0.7750/0.7519;MHClip-ZH 0.7475 M-F1 等)、阶段 4(与我们主表同场对比)。

---

## 阶段 2 进行中:预处理执行记录(2026-07-04 滚动更新)

### 2.1 G1(easyocr EN OCR)COMPLETED(6.5h)
- HateMM 1083/1083、MHClip-EN 792/792(有视频者全覆盖)。抽查内容有效(屏幕仇恨文本被正确提取)。
- 个别 AV1 编码视频硬解失败(cv2 软解仍成功,不影响)。

### 2.2 G2(PaddleOCR ZH)事故与处置
- 事故:首跑(12237)paddle GPU 报 `Cannot load cudnn shared library`(集群无 cudnn8 系统库);脚本的 per-video try/except 把系统性失败当普通坏视频吞掉,开始写空 OCR 行(污染 7 行)。
- 止损:scancel + 删污染的 ocr.jsonl;三层修复:
  1. `MoRE_paddle` env 补 `nvidia-cudnn-cu11==8.9.6.50` + `nvidia-cublas-cu11`(pip wheel),sbatch 注入 `LD_LIBRARY_PATH`;
  2. paddle/easyocr ZH 脚本加 5 连败快速失败(exit 3,不写空行),让 sbatch 三级 fallback(paddle GPU→paddle CPU→easyocr ch_sim)真正可触发;
  3. 重提 G2=12254。
- **终局(12254,修正先前判断)**:paddle 双路皆不可用——(i) GPU:pip cudnn8 + LD_LIBRARY_PATH 仍不被 paddle dynload 认,5 连败→快速失败 exit 3;(ii) CPU:`Illegal instruction`(SIGILL, exit 132),paddle 2.6.2 CPU 内核指令集与本节点 CPU 不兼容(登录节点小图冒烟未触发该内核路径)。三级 fallback 走到底:**ZH OCR 最终由 easyocr(ch_sim+en, GPU)完成,814/814 行、745 非空,耗时 2h27m —— 属任务书预授权的文档化替换**(EN 数据集本就是官方 easyocr 协议,故三库 OCR 引擎实为统一的 easyocr;此为与原文的已知偏差点之一,写入最终报告)。无污染行(GPU 失败期间不写行,easyocr 轮全量重做)。
- 教训入档:吞异常的断点续跑脚本必须区分"单视频坏"与"系统性坏",否则 fallback 形同虚设。

### 2.3 依赖链重排(held 审批期作业禁止 scontrol 改依赖)
- 取消重提:G5=12255(afterok G2'=12254, G3=12238)、C2=12256(afterok P1,G4,G5)、G6=12257(afterok C2)。
- 现行链:P1=12235(R)→ G3=12238/G4=12239;G2=12254;G5=12255 → C2=12256 → G6=12257。
- **二次重排(应 coordinator 提速要求,2026-07-04)**:P1 实际进度快于预估(HateMM+EN 已全部完成,ZH 帧 57%)。评估:MFCC 不在关键路径(仅 C2 消费)不拆;**G3/G4 改 per-dataset 依赖**——HateMM+EN 的 caption/RMBG 立即可跑,不等 P1 的 ZH 尾巴,省 ~2-3h 关键路径。现行链 v3:P1=12235(R)| G2'=12254 | G3A=12267(cap HM+EN,即刻)| G4A=12268(RMBG+ViT HM+EN,即刻)| G3B=12269、G4B=12270(ZH,afterok P1)| G5=12271(afterok G2',G3A,G3B)| C2=12272 | G6=12273。
- G5 新增前置步:`rerun/pad_jsonl.py` —— EN/ZH 的 vids.csv 含 1000 id 但 OCR/caption 只覆盖有视频者(792/814),其 fea_extract 脚本 `.values[0]` 硬索引遇缺行必崩;按其缺失协议补空文本行(幂等)。


---

## 阶段 2 完成:特征与检索全产出(2026-07-04)

| 产物 | HateMM | MHC-EN | MHC-ZH | 备注 |
|---|---|---|---|---|
| frames_16 | 1083(4 坏视频黑帧) | 792 | 814(1 不完整) | 官方 16 帧协议 |
| MFCC-128 | 1083(1068 非零) | 1000(792) | 1000(814) | 缺音频→零向量(其协议) |
| OCR | 1083 | 792(+208 空补) | 814(+186 空补) | 统一 easyocr(ZH 为文档化替换,paddle GPU cudnn/CPU SIGILL 双卒) |
| caption(我方复原) | 1083(1081 非空) | 1000(792) | 1000(814) | Qwen2.5-VL-7B,16 帧,EN 英文/ZH 中文 |
| ViT×3(plain/front/back) | 1083 | 1000 | 1000 | RMBG-1.4 前后景,帧用完即删 |
| BERT 文本特征 | 4 组 | 5 组 | 6 组 | 其原始脚本原样跑 |
| 检索 all_modal | 2 variant | 2 variant | 2 variant | asreleased(bug 保留)/ bugfix;base=train+valid(866/801/800) |

- 形状调和按计划执行(6 个主特征 → (1,d)),merge/检索不受影响。
- 磁盘峰值受控,前后景帧与 wav 均已清理。
- 剩余:G6=12273(6 组训练:3 数据集 × 2 检索 variant,官方 yaml,seed=2024)+ 双轨评测,等 GPU 槽位。