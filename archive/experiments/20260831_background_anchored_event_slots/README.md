# 已淘汰：Background-Anchored Event-Slot MIL

淘汰原因：独立review裁定`STOP 6.0/10`。binary noisy-OR与activation penalty在一个slot足以分类后系统性偏向
单slot，无法识别多事件分工；bag probability不由最终frame marginal聚合，仍有topic/whole-video shortcut；
negative EMA只是普通normal prototype，且SlotSPE已占用weak-bag latent event slots。未实现、未训练、未生成
prediction。

截至 2026-08-31。状态：**独立review `STOP 6.0/10`，已在实现前淘汰。**

## 直接证据与研究问题

MultiHateLoc三seed developmental test诊断显示，fused within-video AUC与GT transition count在HMM为
`rho=-.291`，HCS为`-.205`；HCS最差视频常有4–12次transition。P-MIL candidate-set oracle在HMM/HCS达到
`.73952/.63450`，但single proposal scoring形成whole-video/constant shortcut。factorial CRF又显示连续路径
prior不是答案：zero-transition control不差于core，generic smoothing在HMM/HCS方向相反。

因此当前问题不是再加duration/smoothness，而是正视频可能包含多个语义相似但不连续的hate事件，单一top-k、
单一proposal witness或单一路径会把它们压成一个shortcut。需要一个set-valued latent representation，同时
最终仍输出单一frame score。

所查看的test artifacts：

- `runs/20260831_multihateloc_test_error_analysis/main/metrics.json`
- `runs/20260831_multimodal_pmil_baseline/pilot_seed234/test_error_analysis.json`
- `runs/20260831_factorial_witness_crf/pilot_seed234/test_error_analysis.json`

这些只用于developmental method design；test GT不进入gradient或checkpoint selection。

## 跨任务来源

主要来源是Locatello et al. NeurIPS 2020的Slot Attention：多个exchangeable slots通过对input tokens的竞争式
迭代attention学习set-valued decomposition。更接近的跨任务source是ICLR 2026 SlotSPE，它把slot attention
用于histopathology/genomics MIL，将slots解释为多个latent prognostic events并做selective slot activation。
二者均不是hateful video detection/localization方法。

最邻近的时序方法包括SloTTAr的slot-based temporal abstraction、PRSA-Net的temporal action proposal slot、
WS-TAL的Action Unit Memory与普通multi-head attention。正式查新必须判断本候选是否已被这些方法占用，或只是
把SlotSPE的patch换成seconds。如果核心等价，直接停止。

## 单一核心机制：negative-anchored background slot + emptyable event slots

输入仍是目标语料自己的1fps visual/audio/text feature，经共享temporal encoder得到每秒token `x_t`；不输入
归一化时间位置，避免slot按片头/片尾分工。模型含一个background slot `b`和`K=4`个exchangeable event slots
`e_k`：

1. `b`不是普通learned token。它由当前mini-batch的negative-video seconds通过EMA更新，只接收目标语料train
   negative bags，形成可识别的background anchor；positive bag不能更新anchor。
2. event slots用Slot Attention的三轮competitive updates读取同视频tokens。竞争归一化包含background slot；
   每秒可交给background，不强迫所有seconds进入event slots。
3. 每个event slot另有activation `a_k`和一个显式null key。`a_k`通过hard-concrete gate允许slot完全为空；不设
   “每个positive video必须恰好K个事件”。positive bag只要求`noisy_or_k a_k h_k`为正，negative bag要求所有
   event slots不激活。
4. negative anchor与event slot之间加train-only margin；event slots之间不用简单orthogonality，而用Slot
   Attention原生竞争和总activation penalty。避免为了diversity把同一hate span拆成多个假事件。
5. 唯一frame prediction是同一个模型的slot marginal：

   `s_t = sum_k assignment(t,k) * a_k * sigmoid(h_k)`，只对event slots求和。

   不做NMS、ensemble、per-corpus branch routing或post-hoc calibration。assignment直接参与bag loss与frame
   output，保证机制能改变within ranking。

为防止slots仅按场景topic重构输入，core不使用全feature reconstruction；slot更新只接受bag classification、
negative-anchor contrast与activation sparsity。其可证伪机制是：若多个noncontiguous event slots确实缓解
single-witness shortcut，core应在多transition positive videos上优于single-event control，同时不依赖连续
smoothing；若slots塌缩、按固定位置分工、或K=1等效，机制失败。

## 为什么可能是non-trivial adaptation

Slot Attention原任务分解同一scene的objects，SlotSPE把WSI patches压成prognostic event slots；二者没有
video-level OR supervision、certified negative temporal bags或一个必须解释全部benign seconds的background
anchor。本adaptation改变slot语义与训练可识别性：negative bags只更新background anchor；positive bags允许
数量未知、可为空、可非连续的event slots；最终frame posterior由slot assignment与event activation的同一
marginal产生，而不是只做bag classification。

但风险明确：

- SlotSPE已经占用“slots as latent events in MIL”和selective activation；如果background anchor/null slot只是
  常规background token与sparse gate拼接，novelty不够。
- 只有bag label时，多个exchangeable event slots可能不可识别并完全塌缩；hard-concrete只控制数量，不能证明
  slots对应不同events。
- negative anchor可能退化成ordinary prototype anomaly detection；该路线此前在HCS失败。

独立review必须先判断这三点，不因source未用于hate就自动放行。

## 最小两语料pilot与controls

只在HateMM/HateClipSeg、seed 234做最小pilot，各自独立训练。固定`K=4`来自HCS test中常见多段事件仅作为
developmental design，不扫描K。validation只在固定arm内部选checkpoint，随后立即test三项指标。

arms：

1. current MultiHateLoc anchor；
2. capacity-matched transformer MIL（相同参数量，没有slots）；
3. `K=1` single-event slot；
4. `K=4`但background为普通learned token，不使用negative-only anchor；
5. `K=4`但没有null/activation gate，所有slot必须吸收tokens；
6. full background-anchored emptyable event-slot core；
7. within-video temporal permutation control：训练时固定打乱seconds后恢复原index读出，破坏event assignment与
   content的局部对应但保留bag和token multiset。

机制门：core在HMM/HCS within都高于capacity-matched、K=1与无anchor arms，至少一边`>=+.020`；core的
train-only slot assignment相对capacity control必须实质改变最终frame ordering；有效event-slot数不全为1或4；
temporal permutation不得追平core。另按test GT做developmental分层：`>=4 transitions`视频的core-minus-K1
delta应高于`<=2 transitions`，但该分层不能用于checkpoint或改变方法。

performance gate：两语料core的pooled AP、pooled ROC、within ROC全部严格超过固定SOTA。失败不扩MHC、
不扫描K/temperature/gate penalty、不加generic smoothing、不把不同slot当test ensemble输出。

## 当前claim上限

若全部成立，最多claim：把cross-task latent event slots改造成由negative temporal bags锚定background、允许
未知数量非连续event slots并直接边缘化为frame score的弱监督hateful-video localizer。不能claim首次Slot
Attention、首次slot MIL、首次multi-event modeling、真实event discovery或slot可解释性。
