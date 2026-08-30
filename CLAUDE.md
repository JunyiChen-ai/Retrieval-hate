# CLAUDE.md

## 项目
弱监督 hateful video localization。主数据集:HateMM、MHC-EN、MHC-ZH、HateClipSeg(裁定固定,新数据集只能做 external validation)。研究迭代流程与晋级标准见 `RESEARCH_ITERATION_RULES.md`。

**当前状态唯一入口:`research-wiki/STATUS.md`**(最新代码在哪、权威数字在哪、做到哪了)。每轮迭代结束必须更新它。

## 环境
- 单机 RTX 5090,无 SLURM,直接在终端跑;conda 环境 `HateVideo`。
- 原始视频在 `~/data/`(HateMM、Multihateclip、HateClipSeg),仓库内 `data/` 存派生缓存。
- 长任务必须与 SSH 会话解耦:`nohup`/`setsid` 后台运行,日志与 PID 写进该 run 的输出目录(见下),随时可 `tail -f` 查进度。

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
2. **Within-video macro ROC-AUC**——对每个同时含两类秒的正例视频单独算 AUC 再平均。先例 = Georgescu TPAMI'21 的 macro-averaged AUC + UBnormal CVPR'22(官方指标);"仅正例视频"限制引 UR-DMU AUC_sub 谱系,理由:单类视频 AUC 无定义。**这是定位主指标**(pooled 在高正例率数据集上近似视频级指标,实证:整段广播的 Vad-R1 拿 pooled 第一、within 恰 .500)。
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
