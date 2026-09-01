# 已淘汰：Deletion-Carrier-Abstaining ItS2CLR for weakly supervised hateful video localization

淘汰原因：双语料全部三项SOTA门失败，core相对capacity-matched broadcast的within增益仅
HMM `+.00313`、HCS `+.00105`；冻结test error analysis与独立post-run audit确认core几乎没有改变最终
frame ranking，carrier-dependent gain弱且不跨语料稳定。禁止围绕本轮调margin、replacement或schedule。

截至 2026-08-31。状态：独立 novelty review `GO`（6.3/10），pre-run code/evaluation review最终
`PASS FOR FORMAL PILOT`；HateMM/HateClipSeg 的 anchor、capacity-matched broadcast与core均已完成正式
test。**core双语料 performance与机制效果量均失败，`STOP`，不扩MHC-EN/ZH。** 权威结果：
`runs/20260831_owner_abstaining_its2clr/pilot_seed234/verdict.json`。首次正式顺序run在HMM core硬失败后
停止未完成controls；随后只补齐HCS三主arms。未完成controls不作任何归因claim。

## 正式两语料 test 结果与结论

| corpus | arm | pooled AP | pooled ROC | within ROC |
|---|---|---:|---:|---:|
| HateMM | anchor | .48590 | .74127 | .62467 |
| HateMM | broadcast ItS2CLR | .49903 | .75887 | .61524 |
| HateMM | deletion-carrier core | .48233 | .73533 | .61837 |
| HateClipSeg | anchor | .52957 | .51547 | .51355 |
| HateClipSeg | broadcast ItS2CLR | .52090 | .51033 | .51078 |
| HateClipSeg | deletion-carrier core | .52137 | .51114 | .51183 |

Core相对capacity-matched broadcast的within增益只有HMM `+.00313`、HCS `+.00105`，虽同方向但远低于
冻结的至少一语料`+.020`门；两语料core的三项指标又全部低于SOTA。因此不能声称方法work，也不能仅因
方向一致继续调carrier margin、replacement或self-paced比例。三轮OOF刷新确实改变fused score与carrier
mask（HMM每轮mask变化约`.021–.024`，HCS约`.027–.038`），排除“迭代根本没有执行”；失败是该监督
relation没有产生足够reranking效果。

## Developmental test error analysis

权威输出：`runs/20260831_owner_abstaining_its2clr/pilot_seed234/test_error_analysis.json`。分析只读取冻结的
anchor/broadcast/core test predictions、test GT与core checkpoint；没有训练、没有重新选checkpoint，也没有
改变正式prediction。独立pre-run review与post-run result-chain audit均为`PASS`；后者见
`ERROR_ANALYSIS_POST_RUN_REVIEW.md`。

- HMM core与broadcast逐视频frame-score Spearman均值/中位数为`.97568/.98116`，pooled绝对分数变化均值
  `.05199`；HCS为`.99723/.99752`，绝对变化均值仅`.000372`。core基本保留了broadcast的排序。
- HMM visual/audio/text carrier rate与per-video AUC delta的Spearman分别为`-.136/.139/.240`；HCS为
  `.087/.178/.016`，没有稳定、强的carrier-dependent gain关系。
- HMM在GT正例占比三个分层的mean delta为`+.00357/-.00775/+.00676`；HCS为
  `+.00176/+.00246/-.00032`。失败也不能归结为一个跨语料一致的occupancy区间。

因此本轮不是“机制已显著重排、只差调强度”，而是modality-specific辅助SupCon几乎没有进入最终定位排序；
不得围绕margin、replacement、self-paced比例或单语料carrier rate继续调参。下一候选必须让跨任务机制直接
作用于训练时的temporal ranking/readout，并先证明同一机制在HMM与HCS都改变错误排序。

## 研究问题与直接证据

四个主数据集仍各自独立训练，只使用本语料 train video labels；validation 只在一次固定训练内选
checkpoint；选定后立即在 test 上报告 pooled AP、pooled ROC 与 within-video macro ROC。

本候选只针对一个已由 developmental test error analysis 共同支持的错误：正视频中的 hate evidence
不一定同时存在于 visual/audio/text，而现有目标把 video label广播给每个 modality。MultiHateLoc 的 DMS
最高权重与 test-GT 最佳单模态匹配率在 HMM/EN/ZH/HCS 仅为 `.216/.333/.375/.323`，且 fused 超过全部
单模态的 eligible-video 比例仅 `.345/.159/.042/.154`；P-MIL 在 HCS 又出现大量 modality rank常数，
全 pair IRC/PCE仍强迫无局部证据的 modality充当 teacher。依据分别为：

- `runs/20260831_multihateloc_test_error_analysis/main/metrics.json`
- `runs/20260831_multimodal_pmil_baseline/pilot_seed234/test_error_analysis.json`

这些 test artifacts 允许 inform 本轮设计；此后的 test 结果明确属于 iterative/developmental evidence。

## 跨任务来源方法

来源方法是 Liu et al., CVPR 2023 的 **ItS2CLR**（Iterative Self-paced Supervised Contrastive Learning
for MIL Representations）：从 bag label与当前 MIL instance prediction产生 self-paced instance pseudo labels，
用 supervised contrastive loss迭代改善 instance representation。原方法用于 whole-slide image、乳腺超声和
肺癌突变 MIL，并非 hateful video detection/localization。

Primary source:
https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Multiple_Instance_Learning_via_Iterative_Self-Paced_Supervised_Contrastive_Learning_CVPR_2023_paper.html

普通 ItS2CLR 直接套到 fused frame embedding 只作为 adaptation baseline，不是本项目 novelty。普通时序
pseudo-label contrastive learning在 WTAL/WSVAD 已有大量先例；hateful video detection 也已有 multimodal
contrastive learning。因此 claim 不能是 self-paced、pseudo label、SupCon、MIL 或 contrastive learning本身。

## 单一核心机制：deletion-carrier-abstaining pseudo-instance relation

实例是目标语料自身的 1 fps 秒，不引入跨语料 span。设 seed MIL 对秒 `t` 输出 fused logit `z_t`，并在
训练时原生支持 modality dropout。对每个 modality `m`，只遮蔽该秒的 `m` 后重算 fused logit
`z_t^{-m}`，定义训练期局部删除贡献：

`d_{t,m} = z_t - z_t^{-m}`。

每个 `(t,m)` 的 pseudo state不是二元标签，而是 `{deletion-sensitive hate carrier,
background, abstain}`。这里的 carrier只表示 seed model 对预定义 replacement的局部 deletion
sensitivity，不表示真实或因果 modality ownership：

1. 真实 negative train bag的所有有效 `(t,m)` 是 `background`。
2. Positive train bag中，只有同时进入当前 self-paced 高置信秒集合，而且在两种固定 train-only
   replacement（negative-train modality centroid与同视频局部邻秒插值）下 `d_{t,m}` 都为正的 modality，
   才是 `deletion-sensitive hate carrier`；这条双干预符号一致规则在训练前冻结，不按结果调 margin。
3. ItS2CLR的双侧 self-paced relation保留：positive bag中，OOF fused score最高尾部提供候选carrier秒，
   最低尾部提供高置信 background秒；两端之间的低置信秒保持`abstain`。在最高尾部内，未获稳定 deletion
   support的 modality仍一律`abstain`，不能因为同秒其他modality有证据就被改成background。
4. Self-paced正/负两侧比例按预先固定 schedule从各自最可信实例扩展，不由validation localization或
   test选择。把最低尾部之外、仅因 all deletion effects非正的秒额外改成background，只作为危险假设的
   独立 control，不属于core。

对每个 modality使用独立 projection head计算 supervised contrastive loss：同一 modality内
`carrier↔carrier` 与 `background↔background` 为 positives，异类为 negatives，`abstain`完全不进入
该损失。**不存在跨 modality positive pair**，因此不会把 audio-only slur、text-only subtitle或
visual-only symbol强制对齐。MIL、smoothness和最终 fused scorer仍在全部秒上训练；owner labels只改变
instance representation的训练关系，不作为 inference rule、threshold、ensemble或 calibration。

为防止同一个模型用自己的拟合噪声教自己，pseudo states由 train-only iterative cross-fitting产生：固定
三折，先各自训练seed MIL并为held fold生成OOF relation；随后每5 epochs刷新一次，共15个representation
refinement epochs。每个fold model更新时，它使用的fit-video relation来自没有训练过对应视频的另一个fold
model；更新后再为自己的held fold重生成fused ranking、branch ranking与两种deletion effects。三轮刷新后
才训练最终单模型。由此保留ItS2CLR的 aggregator/pseudo-relation/representation迭代闭环，同时任何视频
始终不能给自己产生训练relation。Cross-fitting不是独立claim，也不读取validation/test。

## 为什么不是 trivial adaptation

ItS2CLR 原始二元 instance pseudo label默认每个 instance只有一个观测视图与一个潜在类别；直接用于本任务
会继续把 positive秒的 bag证据广播到三个 modality。这里改变的是 supervised-contrastive relation本身：
一个时间实例被展开成三组带 abstention 的 modality-owner relation；“没有证明该 modality拥有 hate
evidence”与“该 modality提供 background evidence”在 loss中具有不同代数作用。该不对称关系专门对应
hateful video中 evidence可单模态出现、异步出现或依赖上下文，而非一般 missing-modality robustness。

机制故事可证伪：如果错误 modality supervision确是共同瓶颈，owner-abstaining core应同时提高 HMM与HCS
的 within-video ranking，并降低高分错误秒的 non-owner branch贡献；若只提高 video/pooled separation、
只在单一语料有效，或把 abstain改成 negative后结果不劣，则机制故事失败。

## 固定最小 pilot与 controls（独立 novelty review已 GO；实现后仍须 pre-run code review）

- corpora：HateMM、HateClipSeg；seed 234；两者完全独立。
- backbone：当前 MultiHateLoc reimplementation；不使用 P-MIL proposal readout，不按语料选择 branch。
- 固定 arms：
  1. 原 MultiHateLoc性能 anchor；
  2. capacity-matched per-modality ItS2CLR（同样三个 projection heads，把 fused高置信秒广播给三模态）；
  3. deletion-carrier-abstaining core；
  4. branch-selector（按各 branch confidence选相同数量carrier，不用 deletion）；
  5. rate-preserving shuffled-carrier（同 bag、同 confidence层内打乱 carrier time assignment）；
  6. `abstain→negative`；
  7. positive-bag low-confidence/all-nonpositive `→background`；
  8. projection-only coupling control（相同pseudo states，但SupCon不向共享encoder回传梯度）。
- 所有 arms使用同一 optimizer budget和 validation video-AP checkpoint selection；validation不比较方法性能，
  各 arm训练定义冻结后都直接 test。
- core mechanism gate：两语料 within-video ROC均高于 capacity-matched per-modality ItS2CLR，且至少一语料
  提升 `>=.020`；branch-selector、shuffled-carrier与`abstain→negative`均不得等于或优于core；危险的
  `all-nonpositive→background`若更好，则删除当前保守解释并重新审查，不能事后并入core。
- performance gate：两语料各自的 pooled AP、pooled ROC、within-video ROC全部严格超过固定 SOTA：
  HMM `.5938315566/.8161837922/.6315317180`；HCS `.6193710950/.6050224699/.5619078936`。
- 不允许按 corpus改变 self-paced schedule、删除 modality、选择输出 branch、混合 baseline score或做
  post-hoc calibration。失败即记录 test结果和 error analysis，再生成新机制，不围绕该次 test调阈值。

## 初步最近邻与待审查边界

- ItS2CLR：来源框架；占用普通 self-paced MIL contrastive claim。
- DCC（CVPR 2022）与 DELU/Delta pseudo labels（CVPR 2023）：占用 WTAL 的 denoised temporal pseudo-label
  contrast/self-correction，故本项目不能 claim temporal pseudo-label denoising。
- ImpliHateVid（ACL 2025）与 CLARA（2026）：已占 hateful video的 multimodal/segment contrastive learning，
  故本项目不能 claim multimodal contrastive或 local-global contrastive。
- SAGE（ACL 2026）已在 hateful video detection提出 modality-specific experts与 instance-level evidentiary
  arbitration，并把 dominant benign modality压制 sparse hateful cue称为 feature dilution；因此本候选也不能
  claim“保留单模态证据”或“按证据选择 modality”。唯一待审边界只能是弱监督 temporal instance上，使用
  train-only deletion effect构造带 abstention 的 owner pseudo-relation来改变 self-paced SupCon监督。
- MultiHateLoc：已有 dynamic modality selection与cross-modal contrast；本候选不能 claim modality-aware
  fusion。
- JoMoLD/MoRE 等 hateful multimodal方法可能已涉及 modality-specific noisy labels或 expert contribution；
  独立 reviewer必须重点判断“train-only local deletion ownership + abstention改变 SupCon relation”是否仍只是
  这些方法与ItS2CLR的简单拼接。若是，直接 `STOP_BEFORE_IMPLEMENTATION`。

## 独立 novelty review结论

完整报告：`NOVELTY_REVIEW.md`。裁定 `GO`、novelty `6.3/10`，只允许最窄 claim：train-only
cross-fitted seed的 per-time/per-modality deletion-sensitive carrier selection，unsupported modality在
SupCon中 abstain。禁止使用真实/因果 owner、首次 modality contribution、evidence arbitration等宽 claim。
报告要求的 capacity、selector、shuffled、abstention、intervention robustness与coupling controls已纳入上述
pilot。下一步先实现并经独立代码/evaluation review，通过后才可启动正式 GPU run。
