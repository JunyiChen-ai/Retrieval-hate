# CLAUDE.md

## 项目
弱监督 hateful video localization。主数据集:HateMM、HateClipSeg(2026-09-02 裁定;MHC-EN/MHC-ZH 已停用,不跑、不作门、不进论文主表;新数据集只能做 external validation)。研究迭代流程与晋级标准见 `RESEARCH_ITERATION_RULES.md`。

**当前状态唯一入口:`research-wiki/STATUS.md`**(最新代码在哪、权威数字在哪、做到哪了)。每轮迭代结束必须更新它。

## 环境
- 单机 RTX 5090,无 SLURM,直接在终端跑;conda 环境 `HateVideo`。
- 原始视频在 `~/data/`(HateMM、Multihateclip、HateClipSeg),仓库内 `data/` 存派生缓存。
- 长任务必须与 SSH 会话解耦:`nohup`/`setsid` 后台运行,日志与 PID 写进该 run 的输出目录(见下),随时可 `tail -f` 查进度。

## 多机运行（2026-09-02 立；机器状态为当日核查，变动时更新本表）

| 别名（`~/.ssh/config`） | 主机 | 状态（2026-09-02） |
|---|---|---|
| 本机 = `uoa-lab2` | sc474399, 130.216.119.32 | **主机器**。代码最新（本地领先 GitHub 29 个 commit）、派生缓存最全（`data/` 71G，含 `CLIP_Embedding` 50G）、原始视频 `~/data` 34G、conda `HateVideo`、rclone `b2` 已配。RTX 5090 32G。 |
| `uoa-lab1` | sc474397 | RTX 5090 空闲。`~/Retrieval-hate` 是旧布局（停在 GitHub HEAD，落后本机 29 commit），仓库 `data/` 仅 813M；`~/data` 有 HateMM 12G、Multihateclip 22G、ImpliHateVid，**HateClipSeg 视频缺失**；无 conda、无 torch；rclone `b2` 已配；空盘 1.2T。 |
| `uoa-lab3` | sc474398 | RTX 5090 空闲。空机：无仓库、无数据、无 conda/torch、无 rclone；空盘 1.7T。 |
| `lab-server` | sc448960，账号 `junyi`（用户本人账号） | RTX 5090，当前被用户 `ling` 的进程占 19G/88%，需先看空闲。有 miniconda（HVGuard 等旧环境，无 `HateVideo`）、rclone `b2` 与 `gdrive` 已配；无本项目仓库与数据；空盘 941G。 |

### 选机规则
1. 本机 GPU 可用（`nvidia-smi` 显示占用 < 50% 且空闲显存 ≥ 16G）时一律在本机跑。
2. 本机被占时，按"代码与数据最完整"排序选远程机：`uoa-lab1` > `uoa-lab3` > `lab-server`；`lab-server` 与他人共用 GPU，放最后，开跑前同样按第 1 条的占用标准判断。远程机首次使用前必须完成下面"远程机准备"，缺一项不得开跑。
3. 同一实验的 HateMM/HateClipSeg 两个语料尽量在同一台机器上跑，避免环境差异。

### 远程机准备（一次性，按顺序）
1. **代码**：本机 `git add -A && git commit && git push origin main`，远程 `git clone https://github.com/JunyiChen-ai/Retrieval-hate.git` 或 `git pull`。远程机上不改代码；如必须改，改完立即 commit 并 push，回本机 pull。
2. **环境**：远程装 miniconda 后 `conda env create -f environment_HateVideo.yml -n HateVideo`；本机 torch 为 cu128 版本，远程按本机 `pip freeze` 对齐。
3. **派生缓存**：从本机 rsync 实验需要的 `data/<类型>/` 子目录（不整包拷 71G）：`rsync -a --info=progress2 ~/Retrieval-hate/data/<类型>/ <别名>:~/Retrieval-hate/data/<类型>/`。同步 `PROVENANCE.md`。
4. **原始视频**：优先从本机 rsync `~/data/<数据集>`；本机也缺时从 b2 取：`rclone copy b2:junyi-data/RGCL_video/raw/<HateMM|Multihateclip|HateClipSeg> ~/data/<数据集>/ --transfers 8`。另有 `b2:junyi-data/hate-followup/` 下的 `hatemm_processed.tar`、`mhclip_en_processed.tar`、`mhclip_zh_processed.tar` 为旧仓库处理包。
5. **rclone**：远程没有 `b2` remote 时，`scp ~/.config/rclone/rclone.conf <别名>:~/.config/rclone/`（含密钥，不写进任何文档或 commit），装 rclone 到 `~/.local/bin`。

### 结果回传（强制）
- `runs/` 不进 git。远程实验结束后立即回传：`rsync -a --info=progress2 <别名>:~/Retrieval-hate/runs/<exp_id>/ ~/Retrieval-hate/runs/<exp_id>/`，回传完成后才允许更新 `research-wiki/STATUS.md`；STATUS 只引用本机路径。
- 远程新生成的派生缓存（新特征等）同样 rsync 回本机 `data/<类型>/`，并在 `PROVENANCE.md` 注明生成机器。
- 每个 run 的 `run.log` 首行与实验 README 写明运行主机名。
- 远程长任务同样 `nohup`/`setsid`，PID 与日志写进 run 目录；本机用 `ssh <别名> tail -f` 看进度。

## Agent 调用
- 所有通过 Agent 工具 spawn 的子 agent（proposal review、code review、general-purpose、Explore 等）一律指定 `model: fable`（Claude Fable 5.1），不得降级到 sonnet/haiku/opus。
- 用户可能要求单独 spawn 一个 agent 并直接交代任务；主 agent 先 spawn 待命，再用 SendMessage 把用户的任务原文转给它。

## 目录规范(2026-08-30 立;新文件必须遵守,存量逐步迁移)

| 目录 | 放什么 | git |
|---|---|---|
| `src/` | 稳定共享基础设施:数据加载、特征抽取、评测、指标、通用工具 | 提交 |
| `experiments/` | 迭代原型代码,每轮一个目录 | 提交 |
| `scripts/` | 数据准备、一次性工具、baseline 复现入口 | 提交 |
| `configs/` | 配置文件 | 提交 |
| `docs/` | 冻结文档:协议、预注册、最终报告、评审记录 | 提交 |
| `research-wiki/` | 活文档:当前状态、权威结果表、方向索引 | 提交 |
| `data/` | 输入与派生缓存:特征、转录、OCR、GT 数组 | 忽略 |
| `runs/` | 全部实验输出:checkpoint、分数、日志、指标 | 忽略 |
| `archive/` | 淘汰的实验、过时文档、历史根目录文件 | 提交(仅文本) |
| `third_party/` | 外部代码原样克隆,commit 钉死 | 忽略(除 actionformer) |

### 评测指标(裁定 2026-09-01,查证记录 `experiments/20260830_spantransfer_pilot/METRIC_CONVENTIONS.md`)
本项目 localization 评测固定用三个指标,全部 1fps 帧网格、test 集:
1. **Frame-level (pooled) ROC-AUC**——全部 test 视频的秒拼一池算一个 AUC。标准指标(Sultani CVPR'18 谱系)。
2. **Within-video macro ROC-AUC**——对每个同时含两类秒的正例视频单独算 AUC 再平均。先例 = Georgescu TPAMI'21 的 macro-averaged AUC + UBnormal CVPR'22(官方指标);"仅正例视频"限制引 UR-DMU AUC_sub 谱系,理由:单类视频 AUC 无定义。**2026-09-02 裁定:SOTA 比较用指标 1 和 3(文献通用);within 作为 shortcut 下限约束与附加分析指标,不作比较主指标**(pooled 在高正例率数据集上近似视频级指标,实证:整段广播的 Vad-R1 拿 pooled 第一、within 恰 .500;hateful video 文献无人报 within,VAD 文献仅 Georgescu TPAMI'21/UBnormal CVPR'22 一支使用)。晋级门数值见 `RESEARCH_ITERATION_RULES.md` 第 8 条。
3. **Frame-level (pooled) AP**——同池算 average precision。文献惯例即 pooled(XD-Violence 官方实现);macro AP 无先例,如报告必须标注为扩展指标。

### 代码
- **评测器全仓库只有一份**(`src/` 内的 frame/video 评测),所有实验、baseline 调用同一份;任何目录不得复制或重写评测逻辑。改评测器 = 全表数字失效,必须显式裁定。
- 每轮迭代一个目录:`experiments/<YYYYMMDD>_<slug>/`,内含 `README.md`(机制假设、怎么跑、结论与去向)、训练/推理代码、config。实验目录之间不得互相 import;共享逻辑必须先升入 `src/`。
- 同一段代码被第二个实验需要时就升入 `src/`,不做第三份拷贝。
- 实验淘汰后整目录移入 `archive/experiments/`,README 顶部补一行淘汰原因;不留在 `experiments/` 里腐烂。

### 数据
- `data/` 对实验代码只读:任何训练/推理脚本不得向 `data/` 写文件。
- 新建派生缓存放 `data/<类型>/`,同目录放 `PROVENANCE.md`:生成脚本路径、代码 commit、日期、上游输入。没有出处的缓存视为不可信。
- 大文件(视频、特征、checkpoint)永不进 git。

### 输出
- 每次运行写 `runs/<exp_id>/<run_name>/`:config 快照、代码 commit 哈希、`run.log`、`run.pid`、`metrics.json`(评测器直接输出)。
- **权威数字只认 `runs/` 里的评测器输出文件**;markdown 表格一律是转录,引用时注明来源文件路径。

### 文档
- 根目录白名单:`CLAUDE.md`、`Readme.md`、`RESEARCH_ITERATION_RULES.md`、`LICENSE`、环境文件、`.gitignore` 等配置。**其余任何 markdown/JSON/txt 不得新增到根目录**;报告进 `docs/`,状态进 `research-wiki/`。
- 三层文档,各司其职:
  1. `research-wiki/` = 现状,可原地更新,每份写明"截至日期 + 依据的 commit/结果文件";
  2. `docs/` = 冻结记录(预注册、协议、最终报告),只新增不改写;
  3. `experiments/<id>/README.md` = 单轮实验的自述。
- 同一事实不写第三份。发现两份文档数字冲突,以 `runs/` 原始输出为准,当场修正并注明。

### 存量迁移
- 旧文件按"碰到才迁"处理:改到哪个文件,顺手迁到规范位置;不做一次性大搬迁。
- **冻结溯源路径不动**:POWA teacher 等模块显式指向 `Hate-follow-up` 归档产物的引用保持原样(见 `MIGRATION_ARCHIVE.md`)。
- 根目录存量杂项(TARGET_*.md、MORNING_REPORT.md、sout2.txt 等)迁往 `archive/root-2026-08/`,在 `research-wiki/` 留一行指针。

## 汇报语言
标准技术词汇直说,不发明黑话、不打比喻、不起外号;先说跑了什么、出了什么数、对决策意味着什么。
