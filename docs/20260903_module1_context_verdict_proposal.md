# 模块 1 提案：上下文条件化的 K30 裁定引出（Context-conditioned verdict elicitation）

日期 2026-09-03。状态：**提案，未开跑**（等用户裁定；本文件冻结，结果写回 `experiments/20260903_hier_evidence_mil/README.md`）。基线代码：commit 2026-09-03 17:45 "hier_evidence_mil rev 2: HateClipSeg ablations recorded"。分支 `worktree-module1-context-verdict`。

## 1. 要解决的失败模式（test 上的 developmental evidence，README 8.2）

| test | K30 触发、K4 未触发 | K4 触发、K30 未触发 | 两者都触发 |
|---|---|---|---|
| HateMM 秒级 GT 仇恨率 | **.158**（n=603） | .417（n=5845） | .597（n=5649） |
| HateClipSeg | .732（n=873） | .501（n=5250） | .762（n=3842） |

完整 2×2 格子（`experiments/20260903_hier_evidence_mil/module1_cells.py --fine-tag qwen`，输出 `runs/20260903_hier_evidence_mil/module1_elicitation/<corpus>/cells_qwen.json`）：

| 秒级 | HateMM val（n / GT 率） | HateMM test | HateClipSeg val | HateClipSeg test |
|---|---|---|---|---|
| K30 否、K4 否 | 7662 / .147 | 17172 / .068 | 7412 / .258 | 8874 / .417 |
| K30 否、K4 是 | 2906 / .500 | 5845 / .417 | 4405 / .538 | 5250 / .501 |
| K30 是、K4 否 | 266 / **.767** | 603 / **.158** | 498 / .651 | 873 / .732 |
| K30 是、K4 是 | 2699 / .488 | 5649 / .597 | 2525 / .779 | 3842 / .762 |
| K30 召回 / 精度 | .371 / .513 | .489 / .554 | .349 / .758 | .360 / .756 |
| K4 召回 / 精度 | .675 / .494 | .821 / .505 | .660 / .626 | .561 / .611 |
| K30 触发窗中 K4 也触发的比例（全语料窗级） | .910 | | .826 | |

两点必须如实写：(1) HateMM 的"K30 是、K4 否"格子 val 与 test 相反（val .767，test .158），与 README 4.9 里 val/test 对 K30 可靠性判断相反是同一件事；本提案的失败模式证据只来自 test（developmental evidence），val 不支持。(2) 该格子只占 HateMM test 秒数的 2%（603/29269）；即使把其中全部误报消掉，K30 精度也只从 .554 升到约 .60，对 pooled AP 的直接上限很小。上下文引出的主要期待不是修这个格子，而是让 K30 在 K4 触发的块内（"K30 否、K4 是"5845 秒 GT .417 与"都是"5649 秒 GT .597）落点更准，即块内召回与精度同时改善。

现有 K30 裁定是"每个窗口单独看 4 帧 + 该窗口转录"打分（`scripts/analysis/score_segments_mllm.py` 的 SYSTEM_PROMPT 明说"不要猜视频其他部分"）。HateMM 上短窗口单独触发的裁定 84% 是误报：短窗口里只有一个模糊线索（一个词、一个符号、一个手势），没有上下文判断不了是否针对受保护群体。HateClipSeg 的短窗口触发是可靠的（.732），说明这是语料的问题（HateMM 视频长、话题散），不是 VLM 一律不可靠。证据模型（模块 3）的 EM 只用视频标签，在 HateMM 上把 K30 可靠性估成 q_fine .955，无法纠正（README 4.9：test 上越不信 K30 越好，val/EM 都说该信）。

## 2. 机制

引出 K30 裁定时，提示里加上**所在 K4 块的转录**作为上下文（`--context block_asr`，块划分与 `src/verdict_hmm._block_map` 相同的 (k·4)//30），并明确指令：仍只评本窗口的帧与本窗口转录里的仇恨证据，上下文只用于解释本窗口里模糊或隐晦的线索——上下文表明是良性的就不算，上下文表明针对受保护群体的就算。其余不变：同一模型 Qwen2.5-VL-7B-Instruct、同样 4 帧/窗口、同样 0–3 评分、贪心解码、K4 裁定不变。输出写到新 tag `*_segscoreK30_qwenctx.jsonl`，原文件不动。

定位与来源（规则 4 复核，第 7 节）：模块 1 是"粗到细证据流水线"的引出阶段。来源家族 = 局部打分带时间上下文的 training-free 打分：LLMVS（arXiv 2504.11199，滑窗邻帧 caption 一起喂 LLM、只评中心帧）、LAVAD（CVPR 2024，帧中心时间窗 caption 摘要后给异常分）、CoReVAD（arXiv 2605.23116，粗看长窗建立上下文再细评片段）、文本仇恨检测的上下文提示（arXiv 2510.15685，父帖/前文作上下文）。hateful video 领域：LELA（arXiv 2602.09637）是逐帧独立打分、无邻域上下文；TANDEM（arXiv 2601.11178）的 context baseline 是视频级输入。差别：本机制的上下文是同视频、同语音源的所属粗块转录（不是邻帧 caption 或推理结论），带"只评本窗、上下文只用于解释线索"的范围指令，输出进证据模型与弱监督训练。模块 1 只有在第 3 节消融通过时才进论文主张，否则只在实现细节里提。

与模块 3 的关系：模块 3 把 K4 当块级观测、K30 当窗口级观测，两者条件独立；模块 1 让窗口级观测本身带块上下文，等于把"层次"前移到证据引出阶段。论文里三个模块的统一主线是"粗到细的层次证据"。

## 3. 预注册预期（可证伪，按顺序判）

1. **引出层面（不训练，先看；`module1_cells.py --fine-tag qwenctx`，与 cells_qwen.json 配对报完整 2×2 表、召回/精度、窗级一致率）**：HateMM test 上 K30 精度从 .554 上升且召回（现 .489）下降不超过 .10；"K30 是、K4 否"格子 GT 率从 .158 上升到 ≥ .30；HateClipSeg test K30 精度不低于 .756 − .03、召回不低于 .360 − .05。**控制**：K30ctx 触发窗中 K4 也触发的比例不得升到 ≥ .97（HateMM 现 .910、HateClipSeg .826）——升到接近 1 说明上下文只是把 K30 变成 K4 的复制，即使其它数字通过也淘汰。val 的同一表同时报，val/test 方向相反时如实记录。不满足 1 则模块 1 淘汰，不进训练。
2. **训练无关对照行**：`verdict_hmm_eval.py --fine-tag qwenctx`（输出自动写到 `verdict_hmm_only_qwenctx/`，不覆盖修订 1 的对照行），HateMM val AP > .486（现 w_fine=1 行），test 高于 .541/.818；HateClipSeg val 不低于 .727 − .01。**控制**："只用 K30ctx"行（`score_hmm_fine_only`）两语料 test 都要高于现"只用 K30"行（HateMM .514/.788，HateClipSeg .697/.662）；否则说明提升来自 K4 而非上下文 K30，淘汰。
3. **完整方法**（修订 1 代码，只换 `--fine-tag qwenctx`，规则 7 两语料 seed 234 搜索 → 规则 8 → seed 2025/3407 → 消融）：HateMM 三 seed AP 高于修订 1 的 .657 至少一个标准差（≥ .670）；HateClipSeg 不低于 .699 − .006。模块 1 的消融 = 原 K30 裁定（修订 1 全部数字）对上下文 K30 裁定，seed 234 与三 seed 均报。
4. 不满足 3：模块 1 淘汰，论文回到修订 1；本轮消耗规则 9 修改轮次 2/3。

## 4. 执行规格（一次写全）

- 机器：uoa-lab1（两语料原始视频 `~/data/HateMM/video`、`~/data/HateClipSeg/videos`，`data/ASR/*` K30 与 K4 转录，Qwen2.5-VL-7B 模型缓存均在）。先等或停掉 lab1 上修订 2 的记录链（只作记录，可停）。
- 命令（每语料一条，`nohup`，日志与 PID 写 `runs/20260903_hier_evidence_mil/module1_elicitation/<corpus>/`；脚本内部 `--resume true` 幂等，可断点续跑）：
  - HateMM：`python scripts/analysis/score_segments_mllm.py --dataset HateMM --splits val,test,train --gt_dir ~/scorer_gt --video_dir ./data/video --asr_dir ./data/ASR --asr_tag asrK30_whisper-large-v3 --num_subclips 30 --num_frames 120 --context block_asr --context_subclips 4 --context_asr_tag asrK4_whisper-large-v3 --out_tag qwenctx`
  - HateClipSeg：同上，`--dataset HateClipSeg --splits test`（该语料一个 manifest 含全部 393 视频，与原 K30 文件一致）。
- 用时估计：原 K30 抽取约 7.5 s/视频；上下文使提示变长，估 9 s/视频：HateMM 1068 视频约 2.7 h，HateClipSeg 393 视频约 1 h，串行约 4 h。
- 回传：`rsync` 新 jsonl 到本机 `data/MLLM_scores/<Corpus>/`，`PROVENANCE.md` 追加条目（机器、命令、代码 commit 日期、视频/ASR 输入、覆盖数、video_ok 计数）。
- 之后：第 3 节第 1 步用 `experiments/20260903_hier_evidence_mil/module1_cells.py --fine-tag qwenctx` 判定（权威数字只认其输出 json）；过则第 2 步；过则第 3 步（lab1 HateMM、lab3 HateClipSeg，链脚本同修订 1/2，`search.py --fine-tag qwenctx`，输出 `runs/20260903_hier_evidence_mil_rev3/`）。
- 不做 smoke、不做 `--limit` 试跑。

## 5. 代码改动（本分支，已通过语法检查，未跑）

- `scripts/analysis/score_segments_mllm.py`：`--context {none,block_asr}`、`--context_subclips`、`--context_asr_tag`、`--out_tag`；`CONTEXT_PROMPT`；context 模式下系统提示改为 `SYSTEM_PROMPT_CONTEXT`（把"不要猜视频其他部分"一句换成"上下文只用于解释本段线索、不评上下文本身"，避免与用户提示矛盾）；`build_messages(frames, asr, context)`；上下文 ASR 缺失/为空/窗口数≠K_c 时硬失败；默认参数下行为与原来完全一致。块归属按窗起点规则 (k·4)//30（与 `_block_map` 相同），30 不被 4 整除，窗 7、15、22 跨两块、只拿起点块的转录，其中窗 7 约 90% 时长在块 1 而被归块 0；保持与模块 3 一致，不改。
- `experiments/20260903_hier_evidence_mil/module1_cells.py`：第 3 节第 1 步的判定脚本（2×2 格子、召回/精度、窗级一致率）。
- HateClipSeg 视频数：K30 原文件 395 条（含 2 个不在官方 split 的视频），K4 文件 393 条；本次按原 K30 manifest 跑，加载端按 split 取用，不影响。
- `experiments/20260903_hier_evidence_mil/train.py`：`cfg["fine_tag"]`/`cfg["coarse_tag"]`（默认 qwen）选择裁定缓存。
- `experiments/20260903_hier_evidence_mil/search.py`：`--fine-tag`，写入每个 trial 的 hparams.json（消融复用同一 hparams，自动带 tag）。
- `experiments/20260903_hier_evidence_mil/verdict_hmm_eval.py`：`--fine-tag`。

## 6. 规则 6 code review（2026-09-03，独立 fable agent）
无 BLOCKER。逐项：默认参数下抽取脚本字节级不变（提示、输出路径、记录字段）；块索引 (k·4)//30 与 `_block_map` 逐元素相同；K4/K30 ASR 文件全部恰好 4/30 个窗口，无越界；CONTEXT_PROMPT 转义正确（渲染文本已核）；train/search/verdict_hmm_eval 的 tag 传递正确，消融复用 hparams.json 自动带 tag；训练代码不写 data/；断点续跑读的是新 tag 文件。
要求修改并已改：(1) `verdict_hmm_eval.py` 在 tag ≠ qwen 时输出目录自动加后缀（否则会覆盖修订 1 的对照行文件）；(2) 抽取脚本在 `block_asr` 下上下文 ASR 文件缺失/为空/窗口数不等于 K_c 时硬失败（原来只 WARN 后全用"无上下文"照跑）；(3) 注释"midpoint rule"改为"start-of-window rule"；(4) train.py 用 K_FINE/J_COARSE 作键。
既有偏差记录（不改，与模块 3 一致）：K30 窗口 7 按起点规则归块 0，但其 90% 时长在块 1 内，该窗口的"上下文"块不含它自己的大部分转录；窗口 15、22 无此问题。回传时 PROVENANCE 必须写明 context 模式与命令。

## 7. 规则 4 novelty 复核（2026-09-03，独立 fable agent，9 次检索 + 6 篇原文核对）

裁定 **GO，信心 7/10**。四种 STOP 情形均不适用（来源方法未用于 hateful video；非 ensemble；非纯后处理；不属于"只调超参/换特征/加增强/改训练配置"的工程情形，但作为独立 novelty 很薄，只能作为粗到细流水线的引出阶段元素）。三条件：允许 adaptation 成立；来源未被本任务占用（LELA 逐帧独立打分、TANDEM context baseline 视频级、MultiHateLoc/HateClipSeg 无 VLM 打分）；条件 3 勉强成立，需补控制。无标签或 test 信息泄漏（上下文只是同视频 Whisper K4 转录）。
复核指出并已改：(1) 补"只用 K30ctx 对只用 K30"与"K30ctx 与 K4 一致率"两项控制（第 3 节）；(2) 第 1 步报完整 2×2 表与召回基线，并量化该格子只占 2%（第 1 节）；(3) 判定脚本落库为 `module1_cells.py`；(4) context 模式系统提示改为一致，块归属写为起点规则并注明跨块窗；(5) 第 2 节写明定位与来源家族。
复核未提但本次核算发现：HateMM 的失败模式格子 val 与 test 方向相反（val .767 / test .158），已写入第 1 节；这使模块 1 的依据只来自 test，属 developmental evidence。
