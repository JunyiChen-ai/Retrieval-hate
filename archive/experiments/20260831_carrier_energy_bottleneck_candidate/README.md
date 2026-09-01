# Carrier-Energy Bottleneck candidate

> **淘汰：双独立 novelty review 均为 STOP（4.0/10、4.6/10）。** Gate 1 PASS、Gate 2 窄 PASS、Gate 3 FAIL；direct signed prototype energy 等价于 normalized binary linear head，属于旧 pseudo-state 的直接 head replacement，且 smooth-max 仍允许恒高 abstain modality 支配全视频、within-video ROC 保持 `.5`。正式旧 pilot 的逐行 OOF cache 也未保留，因此 README 原称“复用 frozen cache”不可执行。未实现、未训练、未生成 prediction。

截至 2026-08-31。状态：双独立 novelty/mechanism review 均已 STOP；未实现、未训练、未生成新 prediction。

## 唯一改动与直接依据

Starting architecture仍为MultiHateLoc；四主语料各自独立训练，只使用本语料train video labels。候选不是重跑旧deletion-carrier ItS2CLR，而只修复其正式test揭示的一个明确failure：carrier relation停留在auxiliary projection space，几乎没有进入最终fused frame ranking。

权威依据：`runs/20260831_owner_abstaining_its2clr/pilot_seed234/test_error_analysis.json`。

- HMM core/broadcast逐视频frame-score Spearman均值`.97568`；
- HateClipSeg为`.99723`，pooled absolute score change均值仅`.000372`；
- core相对broadcast within只`+.00313/+.00105`，两语料三项SOTA全败。

旧projection head可以独自满足selective SupCon，而最终frame score仍来自另一个fused classifier。下一轮禁止调整carrier margin、replacement、self-paced schedule或按语料选择modality；唯一变化是让同一carrier relation直接参数化唯一frame score。

## 来源与 novelty 边界

保留的跨任务来源是ItS2CLR（CVPR 2023 medical MIL）：以bag label与OOF/self-paced instance states迭代训练instance representation。旧候选独立novelty review曾对以下窄adaptation给`GO 6.3/10`：每秒×modality的train-only deletion-sensitive carrier、background、abstain三态改变SupCon relation，未获支持的modality abstain而不是被写成background。

新增的跨任务实现来源是proxy/prototypical metric classification：每类由learned prototype表示，sample logit由到正/负prototype的相对相似度直接定义。独立review必须检索proxy/prototype energy、ItS2CLR direct metric readout或等价carrier-energy bottleneck是否已用于hateful-video detection/localization。

允许的最终claim只能是：**把train-only cross-fitted deletion-sensitive modality-instance relation从可旁路的auxiliary SupCon graph改造成最终temporal scorer的abstaining signed-energy bottleneck。** 不能claim prototype learning、metric learning、MIL、modality fusion、deletion attribution或contrastive learning本身新。

## 单一核心机制

沿用旧pilot已经冻结、只由对应corpus train构造的三折OOF state producer。对每个有效 `(video,time,modality)`，state仍为：

- `carrier`：positive bag高置信秒中，两种预定义replacement的local deletion effect均为正；
- `background`：真实negative train bag seconds，以及旧冻结规则指定的positive low-tail background；
- `abstain`：positive bag内既未被证明carrier、也未被指定为高置信background的modality-instance。

不改变这些定义，不重新看test调state rate。

### Direct signed prototype energy

每个modality encoder输出`h_tm`，经一个共享维度但不共享参数的线性映射后L2-normalize为`z_tm`。每个modality只有两个learned unit prototypes `p_m+`、`p_m-`：

`e_tm = (cos(z_tm,p_m+) - cos(z_tm,p_m-)) / tau`。

`carrier/background`直接对`e_tm`做binary logistic loss；`abstain`完全不进入instance loss。不存在额外projection-only空间、独立fused classifier或teacher score regression。

唯一frame logit是所有available modalities的固定smooth maximum：

`s_t = tau_m * logsumexp_m(e_tm / tau_m)`。

availability由现有feature loader的真实channel coverage产生；missing channel从partition移除，不当作零证据。`tau`与`tau_m`固定，不按validation/test扫描。唯一video score由同一`s_t`做与MultiHateLoc一致的top-K MIL pooling；video BCE与instance prototype loss共同更新同一encoder、同一prototype和同一`s_t`。Test只输出`sigmoid(s_t)`，不保留OOF producer、state、teacher、branch selector、ECDF、ensemble或calibration。

这个bottleneck解析排除旧的projection-decoupling解：若carrier/background instance loss改变`e_tm`，它必然改变唯一frame/video logit。它不自动证明pseudo state正确；若state语义错误，错误会直接伤害ranking并由controls暴露。

## 必须解析检查的退化

1. **Video-global broadcast**：如果所有`e_tm`在同视频恒定，positive bag仍可分类。Carrier与background states若在同一positive video都存在，direct instance loss会产生冲突并排除此解；但若某视频只有carrier或只有abstain，broadcast仍可能存在。必须报告每个positive train video是否同时有至少一个carrier与background state，以及mean-repeated test control。
2. **Prototype label shortcut**：encoder可能用video identity把所有秒推向同一prototype。Same-video carrier-vs-background margin、video-mean repeated和within-video state classification必须共同核验；跨视频accuracy不能代替。
3. **Smooth-max branch dominance**：固定smooth max可能长期由一个modality主导。必须报告per-video/per-time winning-modality rate、energy gap与available-count strata；不能把它解释成true owner或做test routing。
4. **State producer mismatch**：首轮复用旧正式pilot的frozen train-only OOF cache，只用于判断direct coupling能否让已有relation进入ranking。它不是新的test target。若pilot通过，扩大验证前必须把producer和energy student整合为从固定seed开始的完整可复现train-only流程。
5. **Pooled/within tradeoff**：per-video ECDF control已证明只保留within会毁掉pooled；因此prototype energy必须在train corpus上共享同一尺度，不能每视频normalize或center。

## 双独立 novelty/mechanism gate

Reviewer必须回答：

1. ItS2CLR + modality deletion/abstention + direct prototype energy的真正有效合取是否已在hateful-video task占用；
2. 把auxiliary relation改为唯一score是否只是普通head replacement，还是使已通过novelty的task-specific relation成为load-bearing bottleneck；
3. reuse旧OOF cache作为首轮固定train supervision是否合规、是否需要匹配producer controls；
4. smooth max是否构成routing/ensemble，或只是单模型固定可微OR aggregation；
5. carrier/background同视频覆盖是否足以解析排除broadcast，最小premise应怎样fail closed。

任一novelty硬门失败则归档，不实现。

## 若review通过：先做train-only coverage premise

不读取validation/test，只审计旧OOF cache：两语料positive train videos中同时含carrier与background的比例、各modality state rate、三轮state变化与同视频可比较pair数。固定gate：HMM/HCS都至少80% positive train videos具有同视频carrier/background pair；三种modality各自carrier覆盖至少10% positive videos，且没有单一modality占全部carrier的90%以上。任一失败即`STOP_BEFORE_STUDENT`，不改阈值。

## 正式最小pilot

若coverage通过，seed 234独立训练HMM/HCS，validation只在每个固定arm内部选checkpoint，选定后立即完整test三指标：

1. `bag_energy`：同architecture/prototypes，只用video MIL，不读states；
2. `broadcast_energy`：把positive high-tail state广播到全部available modalities；
3. `carrier_energy_core`：deletion-carrier/background/abstain direct energy；
4. `shuffled_carrier_energy`：同video、同state rate、同confidence层打乱time×modality carrier assignment；
5. `projection_decoupled`：旧carrier relation只进入auxiliary projection，frame score用capacity-matched独立head；
6. `state_sign_flip`：carrier/background互换，检查普通额外监督量解释；
7. `mean_repeated`：同checkpoint inference，把每视频各modality feature沿时间替换为自身mean，within必须约`.5`。

Mechanism gate：core在两语料within都胜`bag_energy`与`broadcast_energy`，至少一边`>=+.020`；core与projection-decoupled逐视频score Spearman均值都`<.95`；shuffled与sign-flip不得追平；mean-repeated within绝对偏离`.5`不超过`.01`。Performance gate：HMM/HCS各自pooled AP、pooled ROC、within ROC六项全部严格超过固定SOTA。任一失败即归档，不调temperature/state rate、不改aggregation、不扩MHC。

Primary source：[ItS2CLR, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Multiple_Instance_Learning_via_Iterative_Self-Paced_Supervised_Contrastive_Learning_CVPR_2023_paper.html)。
