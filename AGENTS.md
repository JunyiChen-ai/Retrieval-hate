## 项目
弱监督 hateful video localization。主数据集:HateMM、HateClipSeg(2026-09-02 裁定;MHC-EN/MHC-ZH 已停用,不跑、不作门、不进论文主表;新数据集只能做 external validation)。研究迭代流程与晋级标准见 `RESEARCH_ITERATION_RULES.md`。

**当前状态唯一入口:`research-wiki/STATUS.md`**(最新代码在哪、权威数字在哪、做到哪了)。每轮迭代结束必须更新它。
- STATUS 只保留当前目标与结论、三模块实现与缺口、最新权威结果及来源、运行任务与 monitor、下一步。更新时替换对应条目，不逐轮追加时间线；详细过程放实验 README/评审记录，旧状态归档后只留链接，不复制整份研究规则。

## 禁止哈希
- **唯一例外（用户裁定 2026-09-05）**：沿用 CLAUDE.md 的多机代码同步规则，允许读取、记录、比较 Git commit 标识，并通过 Git 提交/推送/拉取同步代码；同时检查未提交修改与未跟踪文件，不能仅凭 commit 相同认定代码一致。此例外仅用于多机代码同步及其检查记录，不扩展到一般 run/结果溯源、数据、缓存、模型或其它文件校验。
- 从现在起，项目研究、数据和运行流程中禁止计算、记录、比较或依赖任何哈希、checksum 或 digest，包括但不限于 SHA、MD5，以及文件、媒体、特征、cache、模型、配置、代码、文档和结果的内容哈希。
- 不得把哈希校验作为 cache 复用、训练、推理、评测或审计的前置门槛；现有代码中的哈希门槛在碰到时必须删除，不得继续扩展。
- 溯源只记录可读的输入/输出路径、配置、模型名称与版本、代码版本说明、日期和生成命令。文件是否可用通过实际解析、覆盖率、shape、split isolation 和任务级测试确认。

## 方法计算成本（用户裁定 2026-09-06）
- **允许使用VLM**，包括最初模块1一类的使用方式；本要求不是禁止VLM。避免没有明确必要性、却大幅增加预处理或新视频推理成本的方法，尤其每窗口多次重复VLM调用。候选7的每窗4次、每视频120次方案已被用户因成本不合理叫停，不恢复。
- 提案时同时考虑性能改进假设与计算代价，说明哪些缓存可复用、新增VLM调用次数和预计GPU时间。不能用“冻结”“离线预处理”“跨trial复用”掩盖处理新视频仍需付出的成本；优先已有缓存与单次/少量必要观测，不以堆调用次数代替机制设计。
- 若方案相对现有方法显著增耗，先向用户说明必要性、廉价替代方案及成本，再决定是否执行大规模抽取。没有证据时不得声称高成本必然带来substantial improvement；不自行设定VLM禁令或任意绝对次数门槛。

## 环境

- 单机 RTX 5090,无 SLURM,直接在终端跑;conda 环境 `HateVideo`。
- 原始视频在 `~/data/`(HateMM、Multihateclip、HateClipSeg),仓库内 `data/` 存派生缓存。
- 长任务必须与 SSH 会话解耦:`nohup`/`setsid` 后台运行,日志与 PID 写进该 run 的输出目录(见下),随时可 `tail -f` 查进度。

### 长实验自动监控（用户裁定 2026-09-05）
- 启动长时间训练、搜索、抽取或实验链时，**必须同时自动配置独立后台 monitor**，无需用户另行提醒。已有对应 monitor 时复用，不重复创建。
- 优先使用完成事件；无事件接口时由轻量脚本定时检查（默认每 120 秒），不调用模型轮询，不因等待而反复开启推理回合或发送相同进度。
- 监控绑定本次运行的主机、进程身份、输出目录及当前会话；核对主进程和相关子进程，区分正常结束、异常停止与暂时失联。SSH 失败或观察超时只重试，不视为实验结束，不自动重启实验。DONE 标记只表示链结束，不能替代结果完整性检查。
- 正常结束或确认异常停止后，通过可用的会话通知接口自动唤醒当前会话；本机已验证 `codex queue --thread <当前会话ID> --message <通知>` 可达。通知包含运行位置、结束状态和接续任务，使用锁及已发送标记避免重复通知；通知失败保留日志并重试。接口不可用时明确说明，不能声称已经设置自动唤醒。
- monitor 同样通过 `nohup`/`setsid` 与终端解耦；脚本放 `experiments/<id>/launch/`（复用逻辑按目录规则升入共享目录），日志、PID、锁及通知标记放 `runs/<exp_id>/<run_name>/monitor/`。启动后确认监控进程存活、首次检查成功，并在 STATUS 记录监控位置。
- 接到通知后，先核验真实进程与原始输出、回传远端结果，再依研究规则继续已授权任务；监控本身不得改动、停止或重复启动训练。
- Goal 自动续轮开关由用户界面控制；若它导致等待期间反复唤醒，提示用户暂停自动续轮，保留后台 monitor 与完整目标，不将等待标记为完成或 blocked。

## 迭代规则（2026-09-02 重写）
全部流程规则只在 `RESEARCH_ITERATION_RULES.md`，此处不复述。要点：一名独立 agent 做 proposal review（只挡已用于 hateful video 的来源、纯 ensemble、纯 calibration/后处理、纯工程技巧）；一名独立 agent 做一次 code review（只查影响结论的 bug）；不做 smoke、不要求单元测试；HateMM/HateClipSeg 各自完整训练、validation 选 checkpoint、立即 test，开发期超参搜索按规则第 7 条；SOTA 定义与分流见规则第 8、9 条。禁止 multi-model ensemble（训练阶段同样禁止）与 inference 后处理。取消失败计数、process review、RESET、premise 门、matched control 门。

## 多机运行（2026-09-05 从 CLAUDE.md 适配）

| SSH 别名 | 主机 | 用途 |
|---|---|---|
| 本机 `uoa-lab2` | sc474399 | 主仓库、代码编辑与结果汇总 |
| `uoa-lab1` | sc474397 | 远程实验机 |
| `uoa-lab3` | sc474398 | 远程实验机 |
| `lab-server` | sc448960，用户账号 `junyi` | 共享实验机，按实际可用资源参与并行调度 |

机器是否空闲、环境与数据是否齐全必须实时检查；不沿用 CLAUDE.md 的 2026-09-02 状态快照。

### 选机与准备
1. **GPU 并行调度（用户裁定 2026-09-05）**：尽量让所有可用 GPU 满载；实时检查各机 GPU 利用率、空闲显存与任务需求，将结果互不依赖、不会相互阻塞的实验并行运行。取消本机优先、固定远端顺序和两语料尽量同机的限制；跨机须对齐代码与环境并记录主机。已满足启动条件的不同语料、独立 seed 搜索、已锁定配置的消融及独立抽取任务可并行；依赖上游结果或尚未满足评审/筛选条件的任务仍按研究规则等待。同 GPU 资源充足时可并发，以实际吞吐提升为准，避免显存溢出、资源争用和输出覆盖；不干扰他人任务，不为占满 GPU 重复跑无必要实验。每项长任务自动配 monitor，完成后及时调度下一项就绪任务。本条是最新用户调度裁定，优先于其它文件中的旧选机偏好，不改变科学评测与晋级要求。
2. 本机维护代码，提交并推送后远端 clone/pull 同步；用 Git commit 标识与工作树状态检查各机代码一致性（上文唯一哈希例外）。提交、推送仅限当前任务已授权的改动，不用 `git add -A` 混入用户其它工作。必要远端修复须提交、推送并回传主仓库，不能留独立代码副本。其它溯源仍采用可读说明、日期、命令。
3. 首次运行前准备 miniconda 与 `HateVideo` 环境，按 `environment_HateVideo.yml` 创建，并按本机实际依赖版本对齐（包括 torch/CUDA）；不能假定远端已经可用。
4. 只同步所需 `data/<类型>/` 子目录及 `PROVENANCE.md`，不整包复制缓存。原始视频优先从本机 `~/data/<数据集>/` 同步；缺失时使用已配置的 `b2:junyi-data/RGCL_video/raw/<数据集>`。文件可用性以实际解析、覆盖率、shape、split isolation 验证；传输工具不得启用 checksum 比较或校验。
5. 需要远端 rclone 时安装到 `~/.local/bin`，仅在任务需要时安全配置所需 remote；配置含密钥，不输出到日志、文档或提交。

### 结果回传（强制）
- 远程实验完成后，将 `runs/<exp_id>/` 对应输出回传本机，再更新权威结果；`research-wiki/STATUS.md` 的结果引用只用本机路径。同步不用删除目标文件的选项，不覆盖不相关运行。
- 远端生成的派生缓存同样回传本机 `data/<类型>/`，在 `PROVENANCE.md` 注明生成机器。
- 每个 run 的 `run.log` 首行与实验 README 写明运行主机名。远程长任务使用 `nohup`/`setsid`，PID、日志放 run 目录，本机通过 SSH 查看进度。

### 文件落位（所有机器一致）
- 项目文件不得散落家目录或仓库外；家目录中的项目主目录为 `~/data`（原始视频）、`~/miniconda3`、`~/Retrieval-hate`，系统与工具自身配置按其规范保留。
- 启动/链脚本进 git，放 `experiments/<id>/launch/run_<corpus>_<机器>.sh`，各机使用同一份同步代码，不运行家目录或 scratchpad 中的独立副本。链日志、启动输出、`search.pid`、DONE 标记放 `runs/<exp_id>/`。
- 一次性抽取/打分输出、GT 文件放 `data/<类型>/` 并写出处，或放 `runs/<exp_id>/`；不得使用 `--out_dir ~/xxx`。环境安装日志放 `runs/_setup_<机器>/`。
- 第二份 checkout 命名为 `~/Retrieval-hate-<分支名>`，仅用于分支代码；输出仍写主仓库 `runs/`、派生缓存写主仓库 `data/`。分支合入后，确认没有未保存工作再清理对应 checkout/worktree。
- 开跑前和汇报前运行 `bash scripts/check_layout.sh`，核对三台机器的 Git commit、脏文件数、未跟踪文件及家目录散落文件；有改动时进一步查看具体文件。发现 STRAY、commit 不一致或影响运行的未提交/未跟踪代码时，先查清归属并处理，不擅自清理他人文件，不在活动实验中途替换代码。

## Agent 调用（适配 Codex）
- Proposal/code review 按研究规则使用独立 agent；其它委派须有用户或适用规则要求。默认继承主 agent 模型，不套用 Claude 专属 `fable` 模型名或不可用的工具接口。
- 用户指定子 agent 任务时，完整传递任务原文与必要上下文；支持时直接在 spawn 的任务描述中交代，不要求先创建无任务 agent。

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
| `third_party/` | 外部代码原样克隆,版本固定 | 忽略(除 actionformer) |

### 代码
- **评测器全仓库只有一份**(`src/` 内的 frame/video 评测),所有实验、baseline 调用同一份;任何目录不得复制或重写评测逻辑。改评测器 = 全表数字失效,必须显式裁定。
- 每轮迭代一个目录:`experiments/<YYYYMMDD>_<slug>/`,内含 `README.md`(机制假设、怎么跑、结论与去向)、训练/推理代码、config。实验目录之间不得互相 import;共享逻辑必须先升入 `src/`。
- 同一段代码被第二个实验需要时就升入 `src/`,不做第三份拷贝。
- 实验淘汰后整目录移入 `archive/experiments/`,README 顶部补一行淘汰原因;不留在 `experiments/` 里腐烂。

### 数据
- `data/` 对实验代码只读:任何训练/推理脚本不得向 `data/` 写文件。
- 新建派生缓存放 `data/<类型>/`,同目录放 `PROVENANCE.md`:生成脚本路径、代码版本说明、日期、上游输入。没有出处的缓存视为不可信。
- 大文件(视频、特征、checkpoint)永不进 git。

### 输出
- 每次运行写 `runs/<exp_id>/<run_name>/`:config 快照、代码版本说明、`run.log`、`run.pid`、`metrics.json`(评测器直接输出)。
- **权威数字只认 `runs/` 里的评测器输出文件**;markdown 表格一律是转录,引用时注明来源文件路径。

### 文档
- 根目录白名单:`CLAUDE.md`、`Readme.md`、`RESEARCH_ITERATION_RULES.md`、`LICENSE`、环境文件、`.gitignore` 等配置。**其余任何 markdown/JSON/txt 不得新增到根目录**;报告进 `docs/`,状态进 `research-wiki/`。
- 三层文档,各司其职:
  1. `research-wiki/` = 现状,可原地更新,每份写明"截至日期 + 依据的代码版本说明/结果文件";
  2. `docs/` = 冻结记录(预注册、协议、最终报告),只新增不改写;
  3. `experiments/<id>/README.md` = 单轮实验的自述。
- 同一事实不写第三份。发现两份文档数字冲突,以 `runs/` 原始输出为准,当场修正并注明。

### 评测指标(裁定 2026-09-01,查证记录 `experiments/20260830_spantransfer_pilot/METRIC_CONVENTIONS.md`)
本项目 localization 评测固定用三个指标,全部 1fps 帧网格、test 集:
1. **Frame-level (pooled) ROC-AUC**——全部 test 视频的秒拼一池算一个 AUC。标准指标(Sultani CVPR'18 谱系)。
2. **Within-video macro ROC-AUC**——对每个同时含两类秒的正例视频单独算 AUC 再平均。Macro AUC 有 Georgescu TPAMI'21/UBnormal CVPR'22 先例；UR-DMU 的异常视频子集 AUC 不等同于逐视频 macro AUC。**按现行 2026-09-02 裁定，SOTA 比较用指标 1 和 3；within 作下限约束与附加分析，不作比较主指标**，数值与搜索用途以 `RESEARCH_ITERATION_RULES.md` 为准。2026-09-05 对话已讨论取消硬门的建议，尚未收到修改规则的明确指令。
3. **Frame-level (pooled) AP**——同池算 average precision。文献惯例即 pooled(XD-Violence 官方实现);macro AP 无先例,如报告必须标注为扩展指标。

### 存量迁移
- 旧文件按"碰到才迁"处理:改到哪个文件,顺手迁到规范位置;不做一次性大搬迁。
- **冻结溯源路径不动**:POWA teacher 等模块显式指向 `Hate-follow-up` 归档产物的引用保持原样(见 `MIGRATION_ARCHIVE.md`)。
- 根目录存量杂项(TARGET_*.md、MORNING_REPORT.md、sout2.txt 等)迁往 `archive/root-2026-08/`,在 `research-wiki/` 留一行指针。

## 汇报语言
标准技术词汇直说,不发明黑话、不打比喻、不起外号;先说跑了什么、出了什么数、对决策意味着什么。
