# STATUS — 当前研究状态入口

**截至 2026-08-30。本文件是全仓库唯一的状态入口:任何时候想知道"最新代码在哪、最新数字在哪、现在做到哪了",从这里开始。每轮迭代结束必须更新本文件。**

## 研究方向

弱监督 hateful video localization:只用 video 级标签训练,输出帧级仇恨分数,目标是 within-video 定位(不只是 pooled 指标)。主数据集固定:HateMM、MHC-EN、MHC-ZH、HateClipSeg。迭代流程与晋级标准:`RESEARCH_ITERATION_RULES.md`。

2026-08-30 从 `/home/jehc223/Hate-follow-up` 迁入本仓库(迁移清单 `docs/MIGRATION_ARCHIVE.md`);label-free 方向留在原仓库。本仓库更早期的方向(RGCL 适配、MLLM 前置、TERA、OCR 等)已全部归档,见下文"归档"。

## 当前进展(按时间)

1. **Baseline 复现完成(至 08-26)**:7 个弱监督方法 × 4 数据集,3 seed 官方验证 —— VadCLIP、DSANet、MACIL-SD(AV/audio/visual)、MultiHateLoc(论文重实现)、CMHKF、Fed-WSVAD(1/3-client)、VERA(training-free,1 seed)。
   **权威表:`docs/duplex/OFFICIAL_VAL_RESULTS.md`**(代码 commit 0e15378)。
2. **POWA-MACIL(08-28)**:自研候选(PEF 语义通道 + Sinkhorn 异步绑定 + 策略树 MIL)。3 seed pooled Frame AP/ROC 在 4 个数据集全部超过复现表最强 baseline;独立评审 6.1/10 PASS。**保留问题:开发期看过 test,数字不是干净的 confirmatory 结果;HateClipSeg 仅超 VERA .00017 AP。** 报告:`docs/duplex/FINAL_POWA_REPORT.md`。
3. **RELATION V2→V26(08-28 → 08-30,已停止)**:以 within-video 定位为目标的迭代链。V19/V20 链 pooled 数字好但真实局部增益只在 MHC-EN;V21–V26 逐个失败,V26(counterfactual temporal witnesses)video AP .888 很强但 within-video 定位门全部未过,判定为负结果并停止迭代。
   **结论:瓶颈不是 video 判别,是弱标签下时序定位的可识别性;下一轮必须以 within-video AP/ROC + shuffle 对照为首要判据。**
   归档:`docs/V20_V26_FINAL_ITERATION_ARCHIVE.md`;教训已写入 `RESEARCH_ITERATION_RULES.md`。

**进行中的迭代(2026-08-30 起,目标 = within-video 定位 SOTA + novel):**

- 协议裁定(用户 2026-08-30):test 任何阶段可用于评估,汇报 performance 一律 = test 结果。
- 第 4 步诊断完成(`experiments/20260830_powa_within_diagnosis/`):POWA within-video 不领先;失败集中在高正例占比视频(MHC-ZH 片头片尾被排最高,反转);监督上限(train 帧标签训练,test 评)HateMM .7495 / MHC-EN .7692 / ZH .6217 / HCS .5989,同架构弱 MIL 对照只有 .578/.459/.413/.523 ⇒ HateMM/EN/ZH 主要矛盾是目标函数,HCS 是特征。
- 第 5–7 步完成(`NOVELTY_SCOUT.md`):选定候选 C1 = 稠密 VLM 窗口打分 + 视频内排序蒸馏(open-with-differentiation);必须新增 baseline:LELA(2602.09637)、TANDEM(2601.11178)。
- 第 9 步 pilot 进行中(`experiments/20260830_vlm_order_pilot/`,预案+kill 门已冻结,独立评审 PASS):Stage T = Qwen2.5-VL-7B 给 test hate 视频每 16s 窗口打分,teacher 自身 within-ROC 在 HateMM 和 MHC-EN 双双 < .60 则杀;过门进 Stage D 蒸馏。

## 最新代码在哪

- **共享评测器(全仓库唯一):`scripts/reproduction_baselines/eval_baseline_scores.py`**(经 `scripts/duplex/frame_eval_common.py`);所有方法和 baseline 的数字都必须出自它。
- 共享数据/特征层:`scripts/reproduction_baselines/hate_common/`(splits、gold、features)。
- 各方法:`scripts/reproduction_baselines/{vadclip,dsanet,macilsd,multihateloc,powa_macil,relation_v2..v26}/` + `cmhkf_adapter.py`、`fed_wsvad_adapter.py`、`vera_adapter.py`。
- 官方验证入口:`scripts/reproduction_baselines/run_official_val_confirmation.sh`。
- 新迭代按 `CLAUDE.md` 规范放 `experiments/<YYYYMMDD>_<slug>/`,输出进 `runs/`。

## 最新文档在哪

| 要查什么 | 文件 |
|---|---|
| baseline 权威数字(验证集) | `docs/duplex/OFFICIAL_VAL_RESULTS.md` / `official_val_results.json` |
| POWA-MACIL 结果与保留问题 | `docs/duplex/FINAL_POWA_REPORT.md`、`FINAL_POWA_NOVELTY_REVIEW.md` |
| V20–V26 迭代总账与负结果 | `docs/V20_V26_FINAL_ITERATION_ARCHIVE.md` |
| 评测协议(1 fps 帧网格) | `docs/duplex/FRAME_EVAL_PROTOCOL.md` |
| 预注册/修正案 | `docs/duplex/PREREG_*.md`、`AMENDMENT_*.md` |
| 迭代规则 | `RESEARCH_ITERATION_RULES.md` |

## 数据与外部依赖

- 原始视频:`~/data/{HateMM,Multihateclip,HateClipSeg}`(不在仓库)。
- 仓库 `data/`:派生缓存(特征、转录、OCR、GT),gitignored。
- **V26 签名产物与 steward_private(加密 test 标签等)仍在 `/home/jehc223/Hate-follow-up/results/steward_private/`,未迁移**;`V20_V26_FINAL_ITERATION_ARCHIVE.md` 里的 `results/...` 相对路径指向该仓库。
- POWA teacher 溯源同样指向 Hate-follow-up 归档(`docs/duplex/POWA_TEACHER_PROVENANCE.json`)。

## 归档(只读历史,不再更新)

- `archive/root-2026-08/`:旧根目录文件(TARGET_*、MORNING_REPORT、RESEARCH_BRIEF、RUNBOOK 等,SLURM 集群时代)。
- `archive/idea-stage/`:MLLM 前置 / OCR / TERA / label-free 探针时代的全部实验目录。
- `archive/refine-logs/`:同时代的预注册/评审/执行记录(557 个文件)。
- `archive/research-wiki-pre-weaksup/`:旧 wiki(RGCL/MLLM 时代的草稿、评审、结果表)。
- `archive/slurm/`:旧集群作业与日志。
- `research-wiki/WEAKSUP_FRAMELEVEL_BASELINES_2026-08-23.md`:过时快照,仅 4 方法,已加过时标注。
- 磁盘上的 `logging/`(83G)与 `artifacts/`(6.7G)是旧时代运行输出,gitignored,未整理未删除。
