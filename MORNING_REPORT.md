# MORNING REPORT — 2026-07-03 夜 → 2026-07-04(两日实验收官总报告)

_面向项目负责人验收。每条结论后括号内为 research-wiki 内的出处文件;所有数字可追溯到 job id 与日志行。协议如未特别注明,均为预注册口径:warmup≥5、val-selected(max Val_Retrieval acc,roc tie-break),150 量级测试集,MHClip test EN n=161 / ZH n=149。_

---

## 1. 目标记分板(acc ≥ 0.85)

| 数据集 | 状态 | 当前数字 | 说明 |
|---|---|---|---|
| **HateMM** | **✓ 达标** | frozen-Qwen RGCL 0.870;frozen-CLIP RGCL 0.8732 | 早已达标,后续未动(`experiments/exp-baseline-reproduction.md`;`EVAL_localization_hatemm.md` §3 参考行) |
| **ImpliHateVid** | **✓ 达标** | ~0.91(frozen-CLIP 0.910 / frozen-Qwen 0.900) | 早已达标(`experiments/exp-baseline-reproduction.md`、`DESIGN_iter1.md`) |
| **MHClip-EN** | ✗ 未达,近天花板 | val-选点 floor 均值 **~0.78**(seeds 0–2:0.7888 / 0.7826 / 0.7702,n=3 均值 0.7805;s3 在跑,快照 0.7391,若确认则 n=4 均值降至 ~0.770) | **所有已测杠杆全部噪声级或有害**:LoRA-SFT 0.7516(有害)、consensus −0.117 F1(有害)、consensus 档案空间/blend(更差,见 §2③)、transcript 长上下文键(acc 噪声级)、archive-kNN α 网格 0.15–0.35 单 seed 0.7888–0.8137(噪声级,α=0.25 多 seed 0.7935±0.0205)、archive mode=both 0.7702(有害)、double key 0.7764(有害)。**唯一未决杠杆 = 角色 3 仲裁(kNN 低置信 → Qwen 裁决),job 12279 在飞**(`experiments/exp-archive-knn-seeds.md`、`ABLATION_transcript_vs_archive.md`、`experiments/exp-consensus-kill-ablation.md`) |
| **MHClip-ZH** | ✗/△ 取决于口径 | val-选点 **~0.827**(archive 臂 0.8268±0.0266、LoRA-only 0.8282±0.0139,均值不过 0.85);**final-epoch 无选点口径 floor 0.8537±0.0120 —— 过 0.85**(5 seeds,seeds 3/4 达 0.8658;archive 臂与 floor 逐位相同) | **协议选择权留给用户**。final-epoch 是合法、标准、selection-free 的协议,但**因为它过线才采纳 = 事后 rule-shopping,风险必须写明**;推荐方案是两口径并排:主表沿用预注册 val-选点,附录给五规则 selection-robustness 全表(附录段落已成文)(`experiments/exp-archive-knn-seeds.md` Addendum 1/2) |

---

## 2. 四支柱终版故事

### ① 检索对比学习 + kNN 记忆(核心骨架)

- 4 数据集全部由同一 RGCL/RA-HMD 骨架(检索引导对比 + kNN 投票头)承载;HateMM/ImpliHateVid 达标,MHClip 见记分板(`experiments/exp-baseline-reproduction.md`)。
- 跨数据集 kNN 记忆换库:5/6 有效跨格 above-majority、零重训 —— MoRE 的 trained-MoE 头结构上不具备该能力(`experiments/exp-cross-dataset-transfer.md`)。
- 与 MoRE 的同场对比:同 split 已逐行核实一致,复跑管线特征/检索全产出,最终训练 G6 在飞(`HEADTOHEAD_FEASIBILITY.md`、`BASELINE_MoRE_rerun.md`)。

### ② 可更新记忆 + 时间演化协议(W4)

- MHClip temporal split 上:**EN 时间漂移真实存在**(macro-F1 0.7113→0.6273,−0.084);ZH 无漂移(+0.014,负对照)(`EVAL_temporal_memory_W4.md` §1)。
- **关键发现:演化 = 校准漂移,不是可分性损失。** EN temporal ROC 0.8484 反高于随机 split 参考 0.7175;只有 8.7% 的 test 分数过 0.5 阈值 vs 真实正例率 24.2%(`EVAL_temporal_memory_W4.md` §1)。
- 原始"往记忆里加新期样本"机制:**全曲线 flat-to-negative,不成立**(所有 k≤80、三种选样策略、双语言)(§2)。
- 成立的替代主张:**k=20 个新期标注样本只做阈值再校准,零重训、O(1)、可逆,全额收复 EN 漂移**(0.7336 ≥ 随机 split floor 0.7113;阈值天花板 0.7646)。检索架构把 operating point 暴露为一等公民旋钮;**trained-MoE/分类头把它藏在权重里,适配必须微调 —— 结构性做不到**。ZH 负对照同时表明:无漂移信号时小样本再校准纯属噪声,应由漂移监测门控(§3、Honest verdict)。

### ③ 共识去噪(retrieval-consensus segment denoising)

- **ZH 成立**:consensus 0.7864 F1 / 0.8188 acc,赢下 kill-ablation(修复 full-mode 洞 0.7050→0.7864 并反超 floor 0.7706/0.8054);机制诊断:ZH 大量剔除"毒正样本"子片段(41.7%)而获益(`experiments/exp-consensus-kill-ablation.md`)。
- **EN 双重证伪**:(i) 视觉 CLIP 键空间硬失败(0.5948/0.7329,−0.117 F1,E1);(ii) **W5 把投票空间换成 MLLM 档案空间/混合空间仍然失败**——EN archive-space 0.5663/0.7205、blend 0.6453/0.7143,双双低于 consensus-visual 且远低于 floor;ZH 换空间同样变差(archive 0.7221/0.7718、blend 0.7232/0.7651 vs 0.7864/0.8188)(jobs 12243–12246 trainlog,本次收录进 `ITERATION_LOG.md`;实现开关 `--consensus_space` 见 `src/run_rac.py`,default=clip 时与 pre-W5 逐位一致)。
- 终版定位:**完整归因链作为分析章节**——EN 仇恨偏语音承载 → 视觉子片段键投票是噪声 → 档案语义键也救不回(投票空间不是病灶)→ claim 严格 scoped 到 ZH/视觉承载仇恨;不作为双语方法主张。

### ④ 可审计 / 可编辑的档案记忆

- **忠实度 77%**:60 条分层抽审(EN/ZH 各 30),faithful 46/60;幻觉 15% 几乎全是"字段级虚报"(benign 内容被安 spurious mechanism,ZH-Normal 最重),仅 1 例内容级虚构;洗白 5% 全部是"毒性只在标题"模式;并反向发现 1 例疑似 gt 漏标(`AUDIT_archive_faithfulness.md`,模型初判、待人工终审的定位已写明)。
- **定向删除 15×**:EN 删 91 条 LGBTQ+ 记忆,目标切片翻转率 12.5% ≈ 随机对照的 15 倍,扰动集中于目标切片;**删 2 条人工标记噪声条目即修复 EN**(0.8075→0.8199 acc,超全部 5 个随机 seed,零训练、秒级、纯 CPU);ZH v1 组级删除 0 翻转 —— 诚实负结果,归因 = v1 档案 target 字段召回过低(`DEMO_memory_editing.md`)。
- **v2 档案闭环**:按审计缺陷改 prompt(target 必填规则 + mechanism 须有可引证据),ZH 有害类 target 召回 **1.6% → 49.4%**(3/182 → 89/180,train);v2 键复测:women 切片(63 条)删除产生切片效应(0.70→0.60,超随机包络),LGBTQ+ 字段切片仍 0 翻转,定向性改善有限且 v2 键下整体基线 0.8523→0.8255(`DEMO_memory_editing_v2_zh.md`、`scripts/slurm/gen_archive_v2.sbatch` 头注)。
- 定位:**能力演示(记忆可语义寻址、外科手术式删除、零训练),不做切片级显著性主张**(n 太小,文中已声明)。

---

## 3. 已撤回 / 被杀的主张清单

| # | 被杀主张 | 撤回依据 | 出处 |
|---|---|---|---|
| 1 | **archive-kNN 键带来 accuracy 提升**(ZH +0.020 / EN +0.019,seed-0) | 多 seed 配对 dAcc = −0.0014±0.0313(t=−0.10);final-epoch 权重 sha1 逐位相同、α=0.25 键 0 票翻转 → 全部"增益"是 78 样本 dev 上的选点运气 | `experiments/exp-archive-knn-seeds.md`(含 Addendum 2 sha1 审计) |
| 2 | **"archive > transcript = 结构化蒸馏"** 的排序主张 | seed-0 差距是 favorable draw;多 seed ΔF1 = +0.0001±0.0388(ZH)/ +0.0013±0.0141(EN);truncation-repair 假设在 ZH 也死(transcript ≤ floor 4-5/5 seeds)。仅存的可写信号:ZH val-选点 ROC 4/5 seeds +0.009,只作分析段 | `ABLATION_transcript_vs_archive.md` |
| 3 | **机制统一**(把共识投票搬进档案空间,一套记忆空间统一检测+去噪) | W5 双语言、双配置(archive/blend)全部低于原空间与 floor | jobs 12243–12246(§2③;`ITERATION_LOG.md` 本次追加) |
| 4 | **任务首次类措辞**("首个 hateful-video 时序定位"、"annotation-free"、"dense-supervision-free") | MultiHateLoc / LELA / TANDEM 分别占位;措辞红线定为只说 "span-free" | `DESIGN_iter3.md` 措辞红线、`NOVELTY_CHECK_dirA.md` |
| 5 | **cross-seed ensemble** 线 | 用户政策明令禁止;计划即撤,零作业、零脚本、零数字 | `experiments/exp-archive-knn-seeds.md` Addendum 2 |
| 6 | **"ZH best-ever 0.8322"** 单 seed 口径(MEMORY 记录) | LoRA-only floor 本身多 seed:val-选点 0.8282±0.0139;final-epoch seeds 3/4 达 0.8658 —— floor 比记录更强,0.8322 只是 seed-0 一个点 | `ABLATION_transcript_vs_archive.md` Paper wording 节 |
| 7 | **MultiHateLoc 复现** | 官方仓库为空(仅 LICENSE),用户政策"无代码不复现";已起步代码标注 ABANDONED 留档 provenance,从未提交过 SLURM 作业 | `EVAL_localization_hatemm.md` §范围决定/§5;`baselines/multihateloc_reimpl/` |
| 8 | (存量,列入以齐全)**multi-granularity 段级检索**作为 headline | 语言符号翻转、噪声 MIL 伪正样本;降级为诚实消融,最高价值 = anti-repeat | `experiments/exp-seg-mode-ablation.md`、`ideas/multigranularity-temporal-retrieval.md` |

---

## 4. 方法学发现(本身值一节论文附录)

**150 量级测试集上,seed 噪声 + 选点噪声支配一切 ≤2 个点的"增益"。**

- 78 样本 dev 上按 val-acc 选 epoch,相对 selection-free 协议(last5-mean / final-epoch)**自损约 2 个 acc 点**(ZH 两臂 val-选点 ~0.827-0.828 vs last5 ~0.846-0.848 vs final-epoch 0.8537)。
- **五规则网格**(val-acc / val-ROC / top3-mean / last5-mean / final-epoch)重打分全部臂:ZH 配对档案效应在 −0.013 ~ +0.008 间摆动 —— **选点规则挪动估计值的幅度超过待测效应本身**;无任何规则跨臂一致占优 → 不改预注册规则,headline 数字不动。
- **sha1 审计**:same-seed 的 archive 臂与 LoRA-only 臂 epoch-29 checkpoint 字节相同(`6d6551e4…`,与 disk_guard 推 B2 时记录的 sha1 吻合)——kNN 键通道确实不触训练,全部差异只在 eval-time 检索键;这也反向解锁了"换键无需重训"的 v2 复测设计。
- 论文用 selection-robustness 附录段落已成文,可直接引用。
- (出处全部:`experiments/exp-archive-knn-seeds.md` Addendum 1/2;分析脚本 `scripts/analysis/selection_rule_robustness.py`)

---

## 5. 在飞未决(截至本报告写作时刻)

| 事项 | job | 状态 | 判读点 |
|---|---|---|---|
| **W7 角色 3 仲裁**(kNN 低置信 margin 门控 → Qwen2.5-VL 裁决) | 12279 | PENDING | EN 最后一个未测杠杆;gate/margin 基线已产出(`scripts/role3/out/gate_*.json`,deferral 集已冻结) |
| **EN floor seed-3** | 12277 | RUNNING(训练已至 ep29,收尾中) | 快照:val-选点 0.7391 / final-epoch 0.8012 —— 若确认,val-选点 floor n=4 均值降至 ~0.770,进一步坐实选点噪声结论(`ABLATION_transcript_vs_archive.md` 的 EN floor 行将由 n=2 升 n=4) |
| **v2 档案 EN 全量生成** | 12280 | PENDING(smoke 12259 已过:30 条 target 非空率 30%) | 补齐 v2 双语;EN 侧 v2 键复测视结果决定 |
| **MoRE 复跑 G6 最终训练** | 12273 | PENDING(阶段 1/2 全部完成:环境、缺件复原、特征、检索 ×2 variant) | 产出 (a) 官方 split sanity 数字 vs 发表值、(b) clean 子集严格同场数字(`BASELINE_MoRE_rerun.md`) |
| **HateClipSeg 定位评测** | 12274 | RUNNING(K4/K30 窗口特征抽取) | 数据已落库(395/435=90.8% 存活,金标清洗完毕),评测脚本就绪,主表 TBD(`DATASET_hateclipseg.md`、`EVAL_localization_hateclipseg.md` §4) |

---

## 6. 用户待拍板

1. **Headline 口径:val-选点 vs final-epoch。**
   - 事实:预注册 val-选点下 ZH 均值 0.827,不过 0.85;final-epoch(标准、selection-free)下 floor 0.8537±0.0120,过 0.85。
   - 风险:因为过线才换口径 = 事后 rule-shopping,rebuttal 必死;且 final-epoch 口径下 archive-kNN 通道在 ZH 贡献恰好为零(逐位相同)、在 EN 低于单 seed floor。
   - 建议案:**两口径并排**——主表沿用预注册口径,附录放五规则鲁棒性全表 + 已成文的说明段;若决定改用 final-epoch,须以"未来预注册"名义全线统一,并在文中自曝该决策时序。
2. **EN 0.85 的处置。** 若 W7 角色 3 也不中(其余杠杆已全部证伪):接受 **"近天花板定位"** —— EN floor ~0.78(val-选点)/ ~0.80(final-epoch),叙事转为归因分析(语音承载仇恨 + 视觉/档案键盲区),0.85 作为该 split 上未被本方法族达到的公开目标如实报告。

---

_附:本报告与 `research-wiki/ITERATION_LOG.md`(07-03 夜 → 07-04 追加节)、各终报文档同步入库;ideas 节点状态已刷新(archive-as-key → refuted;consensus-denoising → ZH-validated / EN-refuted;evolving-memory → validated-as-calibration)。_
