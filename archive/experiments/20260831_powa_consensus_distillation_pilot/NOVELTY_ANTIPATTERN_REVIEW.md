# POWA anchored consensus-order distillation：novelty / anti-pattern review

**评审日期：2026-08-31**  
**评审对象：** `README.md`、`PILOT_PLAN.md`  
**仓库依据：** 当时的工作树源码快照  
**性质：** proposal-stage 历史独立评审；其运行状态与 validation-first 内容已被现行
test-first 规则、修订后的 plan 和 `IMPLEMENTATION_REVIEW.md` 的正式 PASS 覆盖。本文仍只作为
novelty/anti-pattern 边界记录，不作为当前运行授权。

## 1. Verdict

### **历史 verdict：GO_AFTER_FIX；所列修正已完成，当前授权见 implementation review**

这条路线在方法身份上可以诚实成立，但必须先补齐第 7 节列出的 attribution controls，再进行
正式 validation training。

核心判断分成两层：

1. **论文整体仍可称为 novel method。** POWA 的已实现核心已经过独立评审：
   `typed moderation primitives → asynchronous predicate-target transport → executable
   corpus-policy dense MIL`，novelty `6.1/10`、unconditional PASS。一个 novel architecture
   使用已知的 train-only auxiliary supervision，并不会因为辅助项本身不新就失去全部 novelty。
2. **本轮 audio+VERA percentile ordinal distillation 没有新的 novelty。** 它是明确的
   multi-teacher score aggregation、within-video rank calibration 和 pairwise ranking KD。
   这三项均有直接先例。它只能作为 POWA 的训练辅助完整披露，不能成为 contribution、方法名
   中被暗示为新算法，或在 ablation 缺失时承担最终性能故事。

当前 `core / shuffled_teacher / anchor_only` 三臂不足以回答两个关键问题：

- gain 是否只是 residual head + MIL/anchor training，而不是 teacher order；
- 最终主指标 gain 是否在普通 MACIL/base student 上同样成立，从而使 POWA novel core 对结果
  非 load-bearing。

因此当前文档不应直接触发正式训练。完成必要修正后，可按同一冻结 recipe 做最小 pilot；若通过，
仍只说明“已知 auxiliary 能改善 novel POWA”，不产生新的 KD novelty。

## 2. 方法身份：什么时候仍能诚实称为 novel

### 2.1 可以成立的写法

最终方法应称为 **POWA**，贡献边界沿用既有独立评审：

- PEF：policy-typed moderation primitives；
- AWB：hostile predicate 与 protected target 的 asynchronous transport；
- executable policy AST 下的 dense MIL。

audio+VERA teacher 应放在“training supervision / implementation details”中，使用类似表述：

> We train POWA with an additional train-only ordinal distillation loss derived from an
> audio probe and a fixed VLM teacher. This auxiliary is a standard multi-teacher ranking
> distillation device and is not claimed as an algorithmic contribution. Neither teacher is
> used at inference.

论文 contribution list、标题和 abstract 不应把 `consensus-order distillation`、percentile
aggregation、pairwise ranking 或 teacher-free inference 列为新贡献。可以报告它们提高了训练效果，
但必须称为 auxiliary protocol。

### 2.2 会使整体 novelty 故事失效的情况

即使 POWA 历史 novelty review 已通过，以下结果仍会使“当前最终方法”的机制故事不成立：

- `MACIL/base + 相同 ordinal KD` 达到与 `POWA + ordinal KD` 相同的 within-video gain；
- POWA 的 PEF/AWB/policy ablations 在加入 KD 后不再有差异；
- 最终结果主要来自 audio-only 或 VERA-only teacher，但按语料选择不同 teacher；
- 论文标题、方法图或贡献列表实际把 known KD 当成新 core；
- 只用原 POWA 的旧 pooled ablations 证明 novel core，却不检查加入新 auxiliary 后 POWA 在当前
  主指标 within-video ROC 上是否仍 load-bearing。

在第一种情况下，论文仍可以说“POWA 是一个 novel architecture，且我们使用标准 KD 训练它”，
但不能把最终 localization improvement 归因于 POWA，也不能满足本项目“performance 与 novelty
同时晋级”的要求。

## 3. Prior art 与 claim 边界

本轮无需为 generic KD 寻找新名字；下列直接先例足以冻结边界：

| 一手来源 | 已占据内容 | 对本 proposal 的影响 |
|---|---|---|
| [RankDistil, AISTATS 2021](https://proceedings.mlr.press/v130/reddi21a.html) | 用 teacher item order 和 ranking objective 蒸馏 student | pairwise/ordinal order distillation 不新 |
| [Distilling Vision-Language Pre-Training to Collaborate with WTAL, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ju_Distilling_Vision-Language_Pre-Training_To_Collaborate_With_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html) | 在 WTAL 中把 VLP 的 foreground knowledge 与另一分支的 background knowledge做双向 pseudo-label distillation | VLM temporal guidance、互补 teachers 和 weak localization distillation 已有直接先例 |
| [MLLM4WTAL, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Weakly_Supervised_Temporal_Action_Localization_via_Dual-Prior_Collaborative_Learning_Guided_CVPR_2025_paper.html) | MLLM 提供 temporal key semantics / complete semantics，并通过 interactive distillation 提升传统 WTAL | expensive MLLM train guidance、student-style部署故事不新 |
| [DAKD, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Dalvi_Distilling_Aggregated_Knowledge_for_Weakly-Supervised_Video_Anomaly_Detection_WACV_2025_paper.html) | 聚合 I3D/S3D/CLIP multi-backbone teacher，再蒸馏到单 backbone WSVAD student | multi-view teacher aggregation → single test-time student 已在最近邻任务直接出现 |
| [TSCN, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510035.pdf) | two-stream consensus 与迭代 temporal pseudo targets | temporal consensus supervision 不新 |
| [CO2-Net, ACM MM 2021](https://arxiv.org/abs/2107.12589) | appearance/motion cross-modal consensus，互相生成 pseudo target | audio/VLM 跨 view 互补不是开放的上位 claim |
| [Multiview Semi-supervised Learning for Ranking, ECML-PKDD 2011](https://nrc-publications.canada.ca/eng/view/object/?id=cbacfa57-40f8-42ad-bf51-5e1417b443bf) | 多 view rankers 对样本对产生一致 ordinal pseudo labels | multi-view rank aggregation 后训练单 ranker 的基本思想很早已有 |

项目内 [view-stable ordinal review](../../archive/experiments/20260831_view_stable_ordinal_candidate/NOVELTY_SCOUT.md)
已据这些先例把 multi-view ordinal consensus 判为 `STOP` novelty core；本轮不能通过“只在 train
使用”或“推理时只有 student”推翻该裁定。不同点仅是本轮不再 claim 它，而把既有 POWA 保留为
唯一 novel core。这一方法身份是可接受的。

## 4. Ensemble / calibration anti-pattern

### 4.1 它客观上是什么

teacher recipe：

```text
q(t) = 0.5 * percentile(audio_oof(t)) + 0.5 * percentile(VERA(t))
```

因此它客观上同时是：

- 两个 teachers/views 的固定等权 ensemble；
- 每视频 percentile rank calibration；
- 从聚合分数产生 pairwise ordinal pseudo-label；
- 将 ensemble order 蒸馏到一个 student。

“inference 不加载 teachers”只说明它不是 test-time ensemble，不能把训练 teacher aggregation
重新定义成非 ensemble。

### 4.2 为什么不是当前致命违规

项目 Rule 4 禁止把 ensemble/calibration 当论文主方法。当前 README 已明确：

- direct transport 只作为 validation upper bound；
- inference 只有 direct-additive single student；
- KD/consensus/calibration 不作 novelty claim。

只要这一边界在代码、实验表和论文中保持一致，train-only ensemble teacher 可以作为辅助监督，
不是自动否决项。

### 4.3 何时立即变成违规

- 将目录名中的 `consensus-order` 升格为论文方法名或 contribution；
- 把 validation transport 数字 `.62076/.56402` 与 student 数字混在同一方法行；
- inference、checkpoint selection 或 score export 读取 audio-probe/VERA cache；
- 根据 corpus 选择 audio-only、VERA-only、audio+VERA、不同权重或不同 percentile rule；
- test 后再选择 pair threshold、margin、teacher weights 或 additive/transport readout；
- 用 “teacher-free” 暗示训练 supervision 不依赖多 teacher。

## 5. Corpus、split 与 leakage 审查

### 5.1 没有发现的违规

- HMM 与 HCS 分开训练，没有跨主数据集混合 train set，符合 Rule 1。
- audio probe 使用 5-fold OOF：每个 train video 的 audio teacher prediction 来自未见该 video
  的 fold model，避免最直接的 self-label shortcut。
- VERA 使用固定 checkpoint/prompt，train score 不读取 span GT。
- validation GT 仅用于 checkpoint/gate，属于允许的 validation selection。
- proposal 明确 test 不参与本 pilot；当前没有新 test leakage。
- HMM/HCS 使用同一 teacher、student、loss 和 checkpoint rule，没有新增 corpus-specific branch。

### 5.2 必须披露的适应性

`audio+VERA` 不是原 multiview probe 唯一预注册的 primary arm。权威
`runs/20260831_multiview_consensus_probe/analysis.json` 同时评估了 audio、visual、text、concat、
VERA、all-view、density-view、audio+VERA、visual+VERA 和 concat+VERA。当前 recipe 是在看到
这些 validation within-video results 后选出的同一双语料可行组合。

这不是 test leakage，也不禁止继续开发，但必须表述为 **validation-adaptive design choice**，不能
写成预先指定或无选择偏差的 teacher。尤其当前数据本身显示：

- HMM audio upper bound `.62774` 高于 audio+VERA `.62076`；
- HCS VERA `.55738`，audio+VERA `.56402`；concat+VERA 更高到 `.57052`。

这正是 single-teacher controls 和禁止 corpus routing 必不可少的原因。

此外，仓库已记录 POWA 开发期存在既往 test exposure。即使本 pilot 不读取 test，未来 test
仍必须按 Rule 10 标为 iterative/developmental evidence，不得称 untouched confirmatory result。
若设计还受到任何 test error analysis 的具体影响，应在 provenance 中逐项记录。

### 5.3 Cache 与 eligibility 边界

HMM 完整 K16 cache 未生成前不能训练。正式 code review 还应确认：

- manifest 记录 checkpoint revision、prompt、window/index rule、video ID、timestamp 和 decode
  failure；cache 通过实际解析、coverage、时间索引和 score 范围检查；
- 现有 30/744 cache 不复用；
- HCS 排除的 13 个 undecodable videos 只依据 label-free media audit；
- `core`、所有 controls、audio OOF folds 与 student training 使用完全相同的 eligible video set，
  避免 coverage 成为 arm difference；
- 报告排除数及 label counts，不把 `238/238 decodable` 写成完整原始 train coverage。

## 6. 当前 frozen gate 的优点与不足

### 6.1 已经合理的部分

- zero-init identity `max error <= 1e-6` 可验证 residual 实现没有静默改变 anchor；
- within-video `+.020` 与 pooled AP/ROC `-.010` 同时约束，符合当前 localization 主指标；
- core 必须比 shuffled teacher 多 `.010`，是最基本的 teacher-signal attribution；
- 同一 HMM/HCS recipe，任一失败即 kill，阻止按语料选 branch；
- inference graph 不读取 teacher artifacts；
- transport 只作为 upper bound，不进入 checkpoint/test；
- first pooled feasibility、then within selection 的规则已经冻结。

### 6.2 当前 controls 回答不了的问题

`anchor_only` 是零 residual 的原 POWA，不是一个经过相同 optimizer、epoch、MIL 和
anchor loss 的 trainable residual control。因此：

- `core > anchor_only` 可能只是 residual head + extra MIL optimization；
- `core > shuffled` 只能说明真实 order 比随机 order 好，不能说明两个 teacher 的 consensus
  比单 teacher 好；
- 没有 non-POWA student，不能证明最终 gain 与 POWA novel representation 有关；
- shuffled arm 若在 threshold 前打乱，可能改变 accepted-pair coverage/margin，不能隔离 order；
- 一个 seed 足够做 kill pilot，不足以形成论文结论。

所以现有 frozen gate 对 performance feasibility 尚可，对 final method attribution 不足。

## 7. 运行前必须修正

### 7.1 最低新增训练 arms

所有 arms 必须使用同一 eligible train set、residual head、optimizer、epochs、MIL/anchor loss、
checkpoint rule 与 evaluator。

| Arm | 定义 | 必须回答的问题 |
|---|---|---|
| `powa_anchor` | 原 frozen POWA，零 residual，不训练 | evaluator/input identity |
| `powa_residual_no_teacher` | residual head 正常训练，保留 anchor + original MIL，pair loss 关闭 | residual capacity/extra MIL 是否已解释 gain |
| `powa_shuffled_teacher` | 与 core 完全相同的 pair 数、pair endpoints、margin/coverage，只随机翻转/置换方向 | 非随机 teacher order 是否 load-bearing |
| `powa_audio_only` | 只用 OOF audio percentile 构造同 coverage pairs | HMM 是否实际只是 audio KD |
| `powa_vera_only` | 只用 VERA percentile 构造同 coverage pairs | HCS 是否实际只是 VERA KD |
| `powa_audio_vera` | 当前固定等权 teacher | 完整辅助是否胜过各单源 |

`shuffled_teacher` 必须在 accepted pair set 冻结后做 direction permutation，或另行做严格
matched-coverage/matched-margin sampling；不能先 shuffle score 再重新 threshold，导致 pair 数和
难度变化。

**Pilot attribution gate 必须增加：** `core` 在 HMM/HCS 都高于
`powa_residual_no_teacher`；否则 teacher mechanism 失败。若论文使用“互补 teacher”描述，core 还须
胜过 audio-only 与 VERA-only；否则只能如实称单-teacher KD，且不得按 corpus 选择不同 arm。

### 7.2 POWA core load-bearing 的 2×2 control

在进入 test 或论文 claim 前必须比较：

| | 无 ordinal KD | 同一 audio+VERA KD |
|---|---|---|
| MACIL/base matched student | A | B |
| POWA matched student | C | D |

非 POWA arm 使用匹配容量的 temporal residual head、相同输入可用性和训练预算；不得故意移除
audio modality或减少参数。需要检查：

- `D - C`：auxiliary 在 POWA 上的增益；
- `B - A`：auxiliary 在普通 backbone 上的增益；
- `D - B`：加入同一 auxiliary 后 POWA core 的剩余贡献；
- interaction `(D-C) - (B-A)`，只作机制解释，不强制必须为正。

如果 `D ≈ B`，最终性能来自通用 KD，而非 POWA。可以保留工程结果，但不能用它支持 POWA
机制的 localization claim。

### 7.3 POWA 内部 ablation 的复核

现有 POWA novelty review 的 PEF/AWB/policy ablations主要依据旧 validation pooled Frame AP。
若 full pilot 通过，扩大验证前至少需在 HMM/HCS 当前 within-video ROC 上，用同一 KD recipe 复核：

- full POWA；
- same-time/pointwise binder；
- flat or anonymous head；
- policy/teacher channel permutation。

不要求最小 kill pilot 一开始跑完全部，但在声称最终 novel method 或进入正式 test 表之前必须完成。

### 7.4 计划文字必须补充

正式运行前应把下列内容写入 frozen plan，而不是只留在本 review：

1. audio+VERA 是 validation-adaptive、known multi-teacher ordinal KD；
2. `powa_residual_no_teacher`、audio-only、VERA-only 的定义与 gates；
3. shuffled control 如何严格保持 accepted pair coverage/endpoints；
4. HMM/HCS decode eligibility 对所有 arms 完全相同；
5. future test 为 Rule-10 developmental evidence；
6. 单 seed 只做 pilot，论文数字需多 seed 与 paired uncertainty/statistical analysis；
7. zero-mean residual 不等于 sigmoid 后逐视频 score mass 严格守恒；只能 claim anchor regularization
   与 pooled empirical feasibility，不能 claim exact calibration preservation。

修改 `PILOT_PLAN.md` 后应由独立 code/evaluation reviewer 检查实现；本报告本身不构成代码
PASS。

## 8. Teacher-at-inference 审查要求

当前设计在概念上没有 teacher-at-inference：student 输入是 POWA 本来使用的多模态输入和 evidence
representation；“不加载 audio probe”不等于“不使用 audio modality”，两者应明确区分。

代码 review 必须验证：

- inference CLI 不接收 teacher cache、VERA model/path、fold models 或 teacher pair files；
- checkpoint 只含 frozen POWA + residual parameters，不序列化 teacher scores；
- 删除/重命名所有 teacher artifacts 后 inference property test 仍输出相同 score；
- score export 直接来自 `sigmoid(logit(POWA)+residual)`；
- validation checkpoint selection不查询 transport upper bound；
- production/test inference 不动态生成 percentile、pair ranks 或 per-video teacher calibration。

这些条件通过后，可以诚实称 single-student inference。仍应在训练成本表中披露 InternVL2-8B
teacher calls/cache 与五个 OOF audio fold models。

## 9. Frozen go/kill decision after revision

修正后的最小 pilot 可按以下顺序裁定：

1. cache/provenance、property tests、独立 code/evaluation review PASS；
2. HMM/HCS 分别训练 `powa_residual_no_teacher`、`powa_shuffled_teacher`、
   `powa_audio_only`、`powa_vera_only`、`powa_audio_vera`，validation 只选各自 checkpoint；
3. 每个训练臂选定 checkpoint 后立即跑 test 全指标，不设 test 前 performance gate；
4. 只依据 test 判断 core 是否胜过 anchor、no-teacher、matched shuffled 和 single teachers；
5. 不允许根据 corpus 退回不同 single teacher；
6. test 支持继续后完成 MACIL×KD 2×2 与 POWA internal ablations。

若 test 上 core 不胜两个 single teachers，只能得出“某个已知 single-teacher KD 有效”；不得保留
`consensus` 故事，也不得按语料路由。若 MACIL+KD 与 POWA+KD 等效，则 KD 可以作为
baseline/engineering result，但最终论文不能把该增益归因于 POWA。

## 10. 最终 claim boundary

全部必要 controls 通过后，可说：

> POWA is the proposed novel weakly supervised hateful-video localization architecture. We use a
> fully disclosed, train-only audio/VLM ordinal distillation auxiliary to improve local ordering;
> this auxiliary is removed at inference and is not claimed as novel. Matched controls show that
> POWA remains load-bearing under the same auxiliary supervision.

不能说：

- consensus-order distillation 是新算法；
- teacher recipe 不是 ensemble/calibration；
- audio+VERA 是未看 validation 前预先指定；
- teacher-free inference 使 multi-teacher KD 本身变新；
- zero-mean logit residual 严格保持 score multiset、mean、quantiles 或 pooled metrics；
- future test 是 untouched confirmation；
- 一个共享统一模型只靠 policy expression 适配所有 corpora。现有 POWA 是 validation-selected、
  corpus-specific training-regime family，这一点必须继续披露。

## 11. Historical recommendation and current disposition

**历史 proposal verdict：GO_AFTER_FIX。所列 controls 已写入 frozen plan，当前实现审查为 PASS，
正在按 test-first 协议运行。**

这不是因为 auxiliary 缺 novelty——它本来就不需要承担 novelty——而是因为当前 controls 还不能
证明最终方法中 POWA novel core 与 audio+VERA auxiliary 各自贡献了什么。当前最小 pilot 已包含
no-teacher、两个 single-teacher 与 matched shuffled；只有 test 支持继续后才运行 MACIL×KD 2×2
和 POWA internal ablations。若最终不补完整归因，最可能的 reviewer 结论会是：用 validation
挑出的 audio/VLM rank ensemble 蒸馏了一个 residual student，再把历史 POWA novelty 贴到最终
数字上；当前三臂无法反驳这一判断。
