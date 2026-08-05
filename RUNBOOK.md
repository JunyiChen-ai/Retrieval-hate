# RUNBOOK — RGCL hateful-video detection

给**另一台全新机器上的 coding agent** 的操作手册:如何把 environment、data、credentials 从零装起来,
以及必须继承的 conventions。**本文不含"该跑什么实验"的任务指令。**

事实基准:2026-08-06 于 `/data/jehc223/RGCL`(SLURM cluster,user `jehc223`)实测。
下文凡标 **[本 cluster 专属]** 的段落在新机器上不适用,只作为 invocation precedent 参考。

---

## 1. 阅读顺序(先读文档,再动手)

1. `CLAUDE.md` — **唯一权威**:GPU/SLURM 政策、Modal 云端政策、实验流程与四条硬红线、权责声明。(`AGENTS.md` 是 14 行的 agent 侧简版。)
2. `TARGET_STATE.json` — 6308 行,campaign 的结构化状态(iteration、active hypothesis、gate 记录)。
3. `TARGET_FINDINGS.md` — 941 行,**按时间正序**追加的 findings 台账(F-编号);**从文件尾部倒读**,最新结论在最后。`TARGET_LOOP.md` 是配套的逐次决策记录。
4. `refine-logs/` — 544 个文件,单条 lineage 的 prereg / review / record 全文。文件名即 lineage(如 `C02_A0_V9_*`、`MECHFIX_PREGATE_2026-07-27.md`、`DISK_BACKUP_RECORD_2026-07-14.md`)。只在需要某条具体 lineage 细节时进去查。

---

## 2. Environment

### 2.1 版本事实(实测)

| 项 | 值 |
|---|---|
| conda env 名 | `HateVideo`(本机路径 `/data/jehc223/miniconda3/envs/HateVideo`) |
| Python | 3.11.8 |
| torch / torchvision | 2.6.0+cu124(CUDA 12.4)/ 0.21.0 |
| transformers | 4.49.0 |
| faiss-cpu | 1.13.2(**pip 装的 CPU 版;项目里没有 faiss-gpu**) |
| numpy | 1.26.4(< 2.0,勿升) |
| scipy / scikit-learn / torchmetrics | 1.17.1 / 1.5.2 / 1.9.0 |
| peft / accelerate / bitsandbytes / qwen-vl-utils | 0.14.0 / 1.5.2 / 0.49.2 / 0.0.14 |
| decord 0.6.0 / sentence-transformers 5.6.0 | decord = video 解码主路径(fallback PyAV) |
| modal / python-socks / aiohttp_socks | 1.5.2 / 2.8.2 / 0.11.0,见 §2.3 |

### 2.2 两个 export 文件

本次已生成并放在 repo 根:

- **`environment_HateVideo.yml`**(250 行)= `conda env export -n HateVideo`。**这是应当使用的那一份。**
  含 31 个 conda 包(基本只有 python/openssl/libgcc 等 runtime 底座)+ 213 个 pip 包。
- **`environment_HateVideo_minimal.yml`**(6 行)= `conda env export -n HateVideo --from-history`。

**重要发现:`--from-history` 变体实际上只有 `python=3.11.8` 一行依赖** —— 因为这个 env 几乎完全是用 pip 装的,
conda 只负责 Python 本身。所以 minimal 变体**不能作为重建规格**,仅可用来确认"conda 层唯一被显式请求的是 Python 3.11.8"。
重建请一律用完整的 `environment_HateVideo.yml`。

重建命令:

```bash
conda env create -n HateVideo -f environment_HateVideo.yml
conda activate HateVideo
```

完整 export 里的 `prefix:` 指向本机路径,在新机器上 conda 会忽略它;若报错就删掉那一行。
如果目标机 CUDA 版本不同,先把 pip 段里的 `torch==2.6.0+cu124` / `torchvision==0.21.0` 换成对应 wheel,其余照装。

补充参考:`requirement.txt` 是上游 RGCL 的粗粒度清单(无版本 pin),**不要用它重建**;
`scripts/cloud/requirements-cloud.txt` 是 Modal client 侧的精确 pin。

### 2.3 Modal client 依赖(代理环境下必装)

本 cluster 只能经 squid CONNECT proxy(`http(s)_proxy=http://squid.auckland.ac.nz:3128`)访问 `api.modal.com`。
两个包缺一不可,且走的是**两条不同的传输路径**:

- `python-socks[asyncio]==2.8.2` — gRPC 控制面。缺它时 modal 抛的是误导性的 "Could not connect to the Modal server."
- `aiohttp-socks==0.11.0` — volume / blob 批量上传走 aiohttp,**另一条** proxy 代码路径。缺它时 `::sync` 直接 ImportError。

根因记录见 `refine-logs/MODAL_CONNECTIVITY_DEBUG_2026-07-14.md`。新机器若不经 proxy 出网,这两个包无害但非必需。

### 2.4 faiss 说明

装的是 `faiss-cpu`。所有 retrieval / kNN vote 路径(`src/utils/retrieval.py`、
`scripts/analysis/mechfix_ops.py`)都用 `faiss.IndexFlatIP` + `faiss.normalize_L2` 在 float32 上做 exact search,
**不依赖 GPU faiss**。新机器直接 `pip install faiss-cpu==1.13.2` 即可,不要替换成 faiss-gpu(会改变数值路径)。

---

## 3. 硬件与调度假设

### 3.1 本质需求(与调度器无关,任何机器都适用)

- **Head 训练 / arena / probe = 纯 CPU**。一次 head retrain ≈ **52 秒**;典型 batch 用 8 threads / 32 GB RAM。
  所有 `scripts/analysis/*.py` harness 都是普通 Python 脚本,`python xxx.py <args>` 直接可跑。
- **Encoder 级抽取与 LoRA 训练 = 1 张 GPU**。本 cluster 上 109 个 sbatch 用 `--gres=gpu:a100:1`,
  显存按 **40–80 GB** 规划(Qwen2.5-VL-7B bf16 + 8 帧 video pack)。配 8 CPU / 64 GB RAM。
- 磁盘:raw videos ~44 GB + repo `data/` ~5 GB + `artifacts/` ~5.8 GB + HF model cache ~31 GB。
  本机 quota 287G/290G,**几乎打满**——新机器请预留 ≥ 100 GB。

### 3.2 **[本 cluster 专属]** SLURM 约定

新机器若没有 SLURM,以下整节跳过,直接 `python` 跑脚本即可。保留在此仅作 invocation precedent。

- 提交:`sbatch scripts/slurm/<name>.sbatch`。partition = `slurmpartition`。
- **绝不设 `--time`**(所有 sbatch 头部都显式注明 "intentionally NO --time")。
- 每用户上限:16 CPU / 128 GB / 2 GPU(**submit-time aggregate cap**:不要同时跑两个 16-CPU 作业)。
- 作业初始常为 `PENDING (JobHeldUser)` → **等自动放行**,不要强行 release。
- **登录节点 = 计算节点**:非 SLURM 的长进程会被回收。新机器上如果进程不被回收,这条限制自动消失。
- 日志固定落 `slurm/logs/%x_%j.{out,err}`。
- `scripts/disk_guard.sh` 是 quota 看门狗(可 source 进 sbatch),默认 dry-run;**只在本 cluster 有意义**。

---

## 4. Code

### 4.1 获取

`git clone https://github.com/JunyiChen-ai/Retrieval-hate.git` —— 这是本项目的 remote,所有成果推这里。

**remote 布局(两个,别搞混)**:

| remote 名 | URL | 用途 |
|---|---|---|
| `project` | `https://github.com/JunyiChen-ai/Retrieval-hate.git` | **本项目的 remote,push 只推这里** |
| `origin` | `https://github.com/JingbiaoMei/RGCL` | 上游原 hateful-meme 项目,**只读,永远不要往它推**(我们对它无写权限,推也会 403) |

新机器上从 `project` clone 之后,`origin` 就是本项目 remote;若要保留上游做对比,再
`git remote add upstream https://github.com/JingbiaoMei/RGCL`。

Branch `main`。当前 HEAD 以 `git log --oneline -1` 为准(本文件写作时的最后一条研究提交是
`a1d1013 MECH-STAGE-B: all four new-mechanism candidates KILLED at first contact (F121)`,
其后是本次成果归档的若干提交)。

### 4.2 入口点清单(只列位置,不解释用法)

- `src/run_rac.py` — 主训练入口(RAC / RGCL head),90 个 CLI 参数。用法先看 `scripts/slurm/enc3seed*.sbatch` 里的实际调用行。
- `src/utils/generate_*_embedding_HF.py` — 各 encoder 的 feature 抽取器(CLIP / VideoMLLM / subclip / archive / ALIGN / AltCLIP / Molmo2 / bidir 变体)。
- `src/utils/extract_CLIP_features.py`、`src/utils/retrieval.py`、`src/utils/metrics.py` — 特征、faiss 检索、评测指标。
- `scripts/analysis/` — 399 个文件,arena / probe / pregate harness 主战场。代表:
  `mechfix_ops.py`(frozen 决策算子契约)、`mechfix_run.py`、`mech_stage_b.py`、
  `headspace_arena.py` / `headspace_mint.py` / `headspace_fidelity.py`、
  `c02_a0_arena_v9.py`、`c06_falsifier_arena.py`、`c09_a0_arena.py`。
- `scripts/slurm/*.sbatch` — 175 个,**invocation precedent 的权威来源**:任何脚本怎么调、环境变量怎么设,都能在这里找到实跑过的原文。
- `scripts/wrappers/*.sh` — sbatch 与 driver 之间的薄包装。
- `scripts/cloud/modal_probe_runner.py` — Modal 云端 probe runner(profile `jehc223`)。
- `scripts/b2_push.sh` / `scripts/b2_pull.sh` — B2 上传/下载(base prefix `b2:junyi-data/RGCL_video`)。
- `configs/` — 冻结的实验配置(`c01/ c02/ c04/ c09/ lb_scgp/` …),findings 用 sha256 引用它们。
- `artifacts/` — 5.8 GB,各 lineage 的 banked 产物(见 §6 验证用到的 `artifacts/c06_falsifier/mints`)。
- `RA-HMD/LLAMA-FACTORY-Ver202512/` — vendored LLaMA-Factory 分支(LoRA SFT 用),**当前未被 git 跟踪**。

---

## 5. Data

### 5.1 Raw datasets(repo 外,`/data/jehc223/` 下)

| 路径 | 内容 | 大小 |
|---|---|---|
| `/data/jehc223/HateMM` | `video/` 6.2G、`frames/` 2.4G、`quad/` 1.1G、`splits/`、`annotation(new).json` | **9.6 G** |
| `/data/jehc223/Multihateclip` | `English/`(video 4.9G + video_mp4 6.9G + quad 2.6G + audios 852M)、`Chinese/`(video 8.0G + quad 3.0G + audios 784M) | **27 G** |
| `/data/jehc223/HateClipSeg` | `videos/` 4.2G、`Dataset/`、`Images/`、`lexicons.json`、`pilot/` | **4.2 G** |
| `/data/jehc223/ImpliHateVid` | `splits/`、`annotation(new).json`(**video/frames 目录为空 — 本机只留了 metadata**) | **3.7 M** |

数据集代号映射:`MHC` = MultiHateClip English,`MHC_zh` = MultiHateClip Chinese,`HateMM`,`ImpliHateVid`,`HateClipSeg`。

**新机器如何获得**:raw videos 来自各数据集原始发布方,受各自 DUA 约束。**不在 B2 备份范围内,也永远不上云**(§7)。
新机器若只做 head-level 工作,**不需要 raw videos** —— feature caches 就够(§5.3)。

### 5.2 Splits / labels(repo 内)

`data/gt/`(7.6 M,已入 git):

```
data/gt/{HateMM,MHC,MHC_zh,ImpliHateVid,HateClipSeg,MHCsmoke,MHC_temporal,MHC_zh_temporal}/
    train.jsonl  val.jsonl  test.jsonl   (+ target_map.json, hate_spans.json, target_pred_qwen7b*.json 等)
```

每行 JSON:`{"id": "...", "text": "<transcript>", "label": 0|1, ...}`。实测行数:

| dataset | train | val | test |
|---|---|---|---|
| HateMM | 744 | 107 | 215 |
| MHC (EN) | 549 | 80 | 161 |
| MHC_zh | 579 | 78 | 149 |
| ImpliHateVid | 1283 | 325 | 401 |
| HateClipSeg | — | — | 395(**只有 test.jsonl,无 train/val**) |

随 clone 一起到位,无需额外下载。

### 5.3 Feature caches(repo 内,**日常工作的真正输入**)

| 路径 | 内容 | 大小 |
|---|---|---|
| `data/CLIP_Embedding/{HateMM,MHC,MHC_zh,ImpliHateVid,HateClipSeg}/` | `{train,dev_seen,test_seen}_<encoder-tag>.pt` | **2.2 G** |
| `data/audio/{HateMM,MHC,MHC_zh}/` | CLAP audio embedding cache | **247 M** |
| `data/lora_frames/` | LoRA SFT 用的抽好的帧 | 2.6 G |
| `data/{ASR,lora_sft,Archive,MLLM_scores,Summaries*}/` | 派生文本 / 分数 | 27M / 13M / 3.8M / 2.8M / 小 |

`.pt` 结构(实测 `data/CLIP_Embedding/HateMM/dev_seen_Qwen2.5-VL-7B-Instruct_HF.pt`):

```python
{"ids": list[str], "img_feats": Tensor[N, 3584] float32,
 "text_feats": Tensor[N, 3584] float32, "labels": Tensor[N] int64}
```

CLIP-L/14-336 的 tag 维度不同(`openai_clip-vit-large-patch14-336_HF`,768 维)。

**新机器如何获得**:
1. **优先从 B2 restore**(见 §5.4)。这些是浮点派生特征,可以传输、也允许上 Modal。
2. **可重新抽取**(需 GPU + raw videos)。用 `src/utils/generate_VideoMLLM_embedding_HF.py`
   / `generate_CLIP_embedding_HF.py`。**parity 常数必须钉死**,否则新特征与 banked 数字不可比:
   - `num_frames = 8`(uniform sampling)
   - `max_pixels = 360*420 = 151200`,且必须在 `AutoProcessor.from_pretrained(...)` **构造时**传入
     (transformers 4.49 下 `__call__` 会忽略它 —— 见 `generate_VideoMLLM_embedding_HF.py:449-450`)
   - `bf16`,`attn_implementation="sdpa"`
   - `none_placeholder = "(none)"`(空 title/transcript 的占位符,identity/parity guard)
   同一组常数在 `scripts/analysis/sav_f0_common.py:65` 与 `w2a_extract.py:82` 里被复制为 banked parity 常量。

### 5.4 Backblaze B2 备份 / restore

remote 名 `b2`,bucket `junyi-data`,base prefix **`b2:junyi-data/RGCL_video`**。两类前缀:

- `RGCL_video/{logs,embeddings,adapters,archives,...}` — `disk_guard.sh` 的自动镜像。
- `RGCL_video/manual_backup_<date>/...` — 人工整目录备份。已知:
  - **`manual_backup_2026-07-14/{lora_p9,Retrieval}`** — 详见 `refine-logs/DISK_BACKUP_RECORD_2026-07-14.md`
    (含完整 restore 流程、pre-manifest 文件数 978 / 2277、`rclone check --one-way` 验证证据)。
  - **可能存在 2026-08-06 的新备份记录** —— 由并行 agent 撰写中。到达新机器时请先
    `ls refine-logs/ | grep BACKUP` 看有没有更新的那一份,**以最新记录为准**。

restore 用法:

```bash
bash scripts/b2_pull.sh <b2_subpath_under_RGCL_video> <local_path>
# e.g. bash scripts/b2_pull.sh manual_backup_2026-07-14/Retrieval /path/to/logging/Retrieval
```

内部就是 `rclone copy --transfers 8 --progress`。rclone 二进制在本机是
`/data/jehc223/home/.local/bin/rclone`(v1.70.3),新机器自行安装。
拉完务必 `rclone check <local> <remote> --one-way` 复核。

---

## 6. 验证 checklist(装完后自检,不碰任何敏感数据)

按顺序跑,全绿即可认为环境可用。**没有任何一步会读 test 标签用于决策**。

```bash
conda activate HateVideo
cd <repo>

# 1) 核心 import + 版本
python -c "import torch, faiss, transformers, numpy, sklearn; \
print(torch.__version__, torch.version.cuda); print(faiss.__version__); print(transformers.__version__)"
# 期望: 2.6.0+cu124 12.4 / 1.13.2 / 4.49.0

# 2) faiss 数值路径可用
python -c "
import numpy as np, faiss
X=np.ascontiguousarray(np.random.randn(64,32).astype('float32')); faiss.normalize_L2(X)
ix=faiss.IndexFlatIP(32); ix.add(X); D,I=ix.search(X,5); print(D.shape, I[0][0]==0)"

# 3) feature cache 能加载,shape/dtype 对
python -c "
import torch
d=torch.load('data/CLIP_Embedding/HateMM/dev_seen_Qwen2.5-VL-7B-Instruct_HF.pt', map_location='cpu')
print(list(d), d['img_feats'].shape, d['img_feats'].dtype, len(d['ids']))"
# 期望: ['ids','img_feats','text_feats','labels'] torch.Size([107, 3584]) torch.float32 107

# 4) git HEAD
git log --oneline -1     # 期望 a1d1013 (或更新)

# 5) fold-arena replay 对 banked floors —— 最强的端到端 instrument check
CUDA_VISIBLE_DEVICES="" python scripts/analysis/mech_stage_b.py instrument \
  --outroot artifacts/mech_stage_b --scratch artifacts/mech_stage_b/scratch
```

第 5 步说明:它从 `artifacts/c06_falsifier/mints/`(66 个 `.npz`,1.1 G)重放 fold arena,
要求复现 banked floors,**容差 `FLOOR_TOL = 5e-5`,任何一行超差即 HALT 整个 battery,无 fallback**
(`mech_stage_b.py:73,169-204`)。输出写 `artifacts/mech_stage_b/instrument.json`。
本机已有的通过结果(2026-08-05)可作对照:HateMM seed0 `replay_acc 0.888441` vs `banked 0.8884`(`abs_diff 4.086e-05`),
banked floors 全表为 `hatemm acc [0.8884, 0.8858, 0.8858]` / `zh acc [0.8929, 0.8895, 0.8946]`。
**注意**:该 stage 会覆写 `instrument.json`,若那份文件属于某条冻结记录,请先改 `--outroot` 到临时目录。
决策算子的 frozen 契约本体在 `scripts/analysis/mechfix_ops.py`(TOPK=20、rank weights `[20..1]`、
`sigmoid(v)>=0.5` ⇔ `v>=0`;顶部注释即契约全文)。

---

## 7. 不可协商的规则(新机器必须原样继承)

**权威文本是 `CLAUDE.md`,下面是摘要,冲突时以 `CLAUDE.md` 为准。**

四条硬红线(任何仪式简化都不砍):

1. **零 test-set 接触** —— test 标签不得进入任何选择/调参/判决前的读取路径。
2. **判决规则在看到结果之前冻结**(prereg + 冻结哈希)。
3. **盲性** —— 设计/实现期间不得计算候选指标。
4. **正式运行单次提交** —— formal run 只提交一次,不重跑挑好的那次。

其余长期禁令与政策:

- **训练数据只用单数据集自己的 train split**,禁止跨数据集混合(用户裁定 2026-07-14)。
- **无用户裁定不得开 OCR channel**(用户 veto)。
- **禁止 cross-seed ensembles**(用户规则)。
- **raw videos 永不上云 / 永不进任何 API 服务**;只有派生的 `.pt` 浮点特征与 label JSON 可以上 Modal
  (`modal_probe_runner.py` 里的硬拦截**不得移除**)。
- **云端正式验证须"同表同硬件"**:候选与配对 baseline 的全部 seeds 同一 GPU 型号 + 同一镜像,
  prereg 里钉死型号/镜像;**云端与本地数字永不混进同一张对比表**(跨硬件漂移实测 ~1.4pt)。
- **proportional ceremony**:仪式与"跑废成本"成正比。CPU 级便宜实验(≤ ~1h)**最多一轮 review**,
  只有"会产出错误判决 / 触碰 test set"的缺陷才能拦路;修完不复审;禁止文档自洽性迭代单开评审轮。
- **numeric provenance**:任何数字必须重读原始 log 后才能转写,禁止凭记忆转录、禁止编造 companion metric。
- **汇报语言**:标准技术词汇,不发明黑话/比喻/外号;先说跑了什么、出了什么数、对决策意味着什么。
- **权责**:主对话只做讨论/决策/汇报,一切杂活交 subagent;主对话调 subagent 只用 Opus 5。

---

## 8. Secrets / credentials(只列"需要什么",不含任何密钥值)

新机器需要自备下列凭据,**本文件与 repo 中都不存放任何 secret**:

| 凭据 | 位置 / 形式 | 用途 | 备注 |
|---|---|---|---|
| **B2 credentials** | `~/.config/rclone/rclone.conf` 中名为 `b2` 的 remote(bucket `junyi-data`) | `scripts/b2_{push,pull}.sh`、`disk_guard.sh` | 需 rclone ≥ 1.70 二进制 |
| **Modal token** | `~/.modal.toml`,profile section `[jehc223]` | `scripts/cloud/modal_probe_runner.py` | 本机该文件仅 4 行 |
| **HTTP proxy**(如在受限网络) | `http_proxy` / `https_proxy` 环境变量 | Modal / HF 下载 | 配合 §2.3 两个 socks 包 |
| **HuggingFace cache** | 本机 `/data/jehc223/home/.cache/huggingface`(**31 G**),无 `HF_HOME`/`HF_HUB_CACHE` 环境变量,靠默认 `~/.cache` 解析 | 模型权重 | 新机器建议显式设 `HF_HOME` 指向大盘 |
| **HF token** | 若目标模型 gated 则需要 | 权重下载 | 本项目当前用到的模型均非 gated |

### 本地已有的模型权重(新机器需重新下载,按需取)

| 模型 | 本地大小 | 何时需要 |
|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct` | **16 G** | VideoMLLM feature 抽取 / LoRA 训练的主 encoder |
| `openai/clip-vit-large-patch14-336` | 3.2 G | CLIP baseline feature |
| `openai/whisper-large-v3` | 2.9 G | ASR transcript |
| `openai/whisper-base` | (小) | ASR 轻量档 |
| `laion/larger_clap_general` | 744 M | audio CLAP cache |
| `google-bert/bert-base-chinese` / `bert-base-uncased` | 1.6 G / 421 M | 文本 baseline |
| `sentence-transformers/{all-mpnet-base-v2, all-MiniLM-L6-v2, paraphrase-multilingual-*}` | 各 ~0.4–1.1 G | archive / transcript embedding |

cache 中还存有若干**与本项目无关**的大模型(InternVL3-78B、Qwen3-VL-235B、LLaVA-NeXT-Video-34B 等,来自同机其他项目),
**不要照搬下载**。

**只做 head-level / arena 工作的话,以上权重一个都不需要** —— feature caches(§5.3)已经是 encoder 的输出。

---

## 9. 新机器最小可用路径(总结)

1. `conda env create -f environment_HateVideo.yml` → `conda activate HateVideo`。
2. `git clone https://github.com/JunyiChen-ai/Retrieval-hate.git` → 确认 `git log --oneline -1`。
3. 配 rclone `b2` remote → `scripts/b2_pull.sh` 拉回需要的 feature caches / artifacts;
   `data/gt/` 与 `configs/` 随 clone 已到位。
4. 跑 §6 五步 checklist,重点是第 5 步 fold-arena replay(5e-5 容差)。
5. 读 §1 的文档顺序,把 §7 的规则当作硬约束。
6. raw videos 与 GPU encoder 抽取只在确实需要重建特征时才配置;届时严守 §5.3 的 parity 常数。
