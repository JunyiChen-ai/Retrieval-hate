# Process Review RESET7 — 2026-09-01

独立只读 process review；未审代码、未修改文件、未复跑实验，也未提出或实现具体新 candidate。

依据：`research-wiki/STATUS.md`、`RESEARCH_ITERATION_RULES.md`、`docs/PROCESS_REVIEW_RESET6_2026-09-01.md`、`research-wiki/RESET6_GOAL_GAP_AUDIT.md`，以及 RESET6 三个正式方法的归档 README、novelty review 和 `runs/` summary：Lexically Anchored DCC、Policy-Constrained Cluster Transport、Active-speaker-bound utterance MIL。

## 唯一总裁定

**RESET。**

项目整体不停止，但不得按 RESET6 的候选生成和 admission 方式直接继续。累计连续正式 performance failure 保持 `14`；落实本审查修正后，可建立新的 process-review 触发窗口 `0/3`，但不表示 performance failure 已清零。

## 1. 当前探索为何停滞

RESET6 三次都是有效正式方法迭代：均通过 novelty、逐语料完成完整 validation 超参数与 checkpoint 选择、锁定后立即 HMM/HCS test、没有 smoke，且跑前只做一次限定范围 technical review。因此停滞不来自 validation 使用不合规、review 太少或正式实验不足。

根因是 RESET6 关闭 self-derived responsibility 后，没有真正建立一个由 HMM/HCS 已观察证据支持的 correction-signal admission gate。三个方法来源不同，却共享同一上位失败链：把语义上合理的外部信号作为 auxiliary constraint/adapter 写入基本保留的 POWA raw scorer，正确语义 control 没有在 HMM/HCS 共同形成 load-bearing 增益。

- DCC：正确 timing 在两语料胜 shifted，但 HMM 相对 matched anchor 三项全降，HCS只有小幅提高。
- Policy transport：policy core 在两语料 within 都输 binary/unconstrained control。
- Active-speaker：HMM 相对 face permutation within 仅 `+.003067`，HCS为 `-.000008`，其他差异近零或为负。

三方法在 HMM/HCS 共十八个候选×语料×指标位置上没有一项超过固定 SOTA。问题不是某个权重没调好，而是有效幅度和 final-score 控制力不足。

## 2. 流程诊断

- **重复失败链存在**：三种公式不严格同构，但都属于“POWA主路径 + 语义辅助约束”。Rule 21 允许改变 representation/backbone，实际候选选择没有落实。
- **candidate churn明显**：三个正式方法之间又有 certified transplant、privileged slack、D2、prompt competition、refusal geometry、shared address 六个 novelty STOP。Novelty gate在工作，问题是送审前候选生成质量低，跨来源方法名切换没有被 goal-gap evidence约束。
- **没有 premise churn，但存在 evidence bypass**：signal可计算被当成与当前六项缺口相关。Active-speaker更从跨任务语义故事直接进入昂贵全量 producer，缺少 HMM/HCS 已观察的覆盖/幅度证据。
- **过早复杂化存在**：完整 validation search合规且必须保留；资源错配发生在正式搜索前的 candidate admission，不应通过削弱 validation 或做 smoke解决。
- **目标偏移仍存在**：候选选择仍偏向机制故事与来源 novelty，而不是所需增益量级。固定缺口为 HMM AP/ROC/within `+.100831/+.077925/+.003076`，HCS `+.066350/+.060950/+.038207`；千分级 auxiliary effect与目标缺口不匹配。

## 3. 必须落实的流程修正

1. 用已有 core/permuted/anchor test prediction与GT完成 Active-speaker 一次聚焦 error analysis，至少记录 multi-face eligible覆盖、eligible组 core-minus-permuted变化，以及收益是否只存在于极小子集。只用于关闭原因，不得修补或重开该family。
2. 建立 RESET7 cross-candidate failure matrix，只复用三个正式 summary和已有 test artifacts；记录 observed headroom、实际可用signal、matched-control delta、相对official starting point/SOTA缺口、raw final score控制力和共同失败组。
3. 将 correction-signal 从brief描述升级为候选 admission硬条件：下一候选必须引用权威artifact，证明核心observation在HMM/HCS都存在、train/test inference可得、不依赖test GT oracle或当前scorer自确认，并直接关联至少一个主要 pooled/within缺口。不得要求raw scalar独立完成定位或预先承担全部方法control。
4. 真正落实architecture freedom：下一epoch不得默认“POWA不变 + auxiliary loss/adapter”。可改变representation/backbone；POWA可作official/matched comparator，但不是唯一骨架。
5. Brief增加数值gain budget，列出当前起点到SOTA的实际缺口，并说明已有证据为何支持相应量级；只有千分量级证据而目标缺口为`.04–.10`时不得进入正式训练。
6. 主agent先依据failure ledger、目标任务占用和关闭family预筛，只把一个最强且证据完整的brief交给novelty reviewer。连续两个brief在Gate 2/3 STOP后，停止换来源术语，回到failure matrix修正生成标准。
7. 全新producer若无既有HMM/HCS证据，不得仅凭跨任务成功或语义故事投入昂贵全量cache。Rule 14确需时，只允许一次无训练、无参数扫描、覆盖完整预定cohort的bounded observation；不得做子集smoke或producer/statistic sweep。
8. Validation与test流程不变：逐语料做足够的完整validation search并联合选择checkpoint，锁定后立即test全部固定指标。
9. 跑前仍只做一次基础technical review，process reviewer不审代码，不增加重复review。
10. 每次正式双test后立即完成一次聚焦error analysis，再按Rule 18关闭family或最多一次artifact直接支持的corrective。

## 4. 已足够的证据

- 三次selection/test链完整合规，无需重跑确认失败。
- DCC region-memory、policy cluster transport、active-speaker/source-bound face assignment及各自参数/producer变体均可关闭。
- 当前POWA additive semantic-auxiliary上位family不足以接近六项SOTA，不得仅更换外部来源与辅助loss续命。
- Novelty gate、一次technical review和validation search不是主要瓶颈；不得加重review或把validation压缩成smoke。
- 既有complementarity/simplex diagnostic仍表明任务存在数值headroom，因此整体研究不应STOP。

## 5. 仍缺的关键证据

- HMM/HCS共同成立、训练和推理真实可用，并同时触及pooled separation与within ordering的correction signal。
- 能解释“HMM主要缺pooled、HCS三项都缺”的共同failure decomposition。
- 不依赖ensemble/calibration、普通KD或当前scorer自确认，却能把互补signal转成单一raw score的机制依据。
- 下一starting scaffold具备所需增益幅度的证据；POWA擅长HMM pooled不足以证明其也是HCS最合理骨架。
- Active-speaker eligible subgroup的test归因记录；这只影响关闭解释，不影响关闭决定。

## 6. 方向处置与恢复顺序

继续：整体任务、HMM/HCS test-first迭代、validation选参/ckpt、一次基础technical review、test artifacts error analysis。暂停：完成本次流程修正与两项分析前的新candidate、无双语料证据的新teacher/producer、默认POWA不可变骨架。彻底关闭：RESET6三个正式family、六个novelty STOP proposal，以及当前epoch内“保留POWA主raw scorer、只换语义auxiliary signal/loss/adapter”的同类方案，除非先出现新的独立双语料权威证据。

恢复顺序：冻结本裁定 → Active-speaker聚焦error analysis → RESET7 failure matrix → 依据六项gap和observed correction signal选择failure target/starting scaffold → 一页brief+gain budget → novelty三门 → 最小单机制实现 → 一次technical review → 完整validation → 立即双test → 一次聚焦error analysis。
