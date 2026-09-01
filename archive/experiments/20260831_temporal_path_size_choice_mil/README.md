# 已淘汰：Temporal Path-Size Choice MIL

淘汰原因：固定`beta=1`的双语料frozen test premise使HMM/HCS within-video ROC均下降，同时增加
long-proposal posterior mass并降低near/thin扰动下的frame-ranking stability。独立post-run result-chain
audit为`PASS`，方法裁定`STOP_BEFORE_FORMAL_METHOD`；禁止扫描beta或改PS公式/readout/cutoff挽救。

截至 2026-08-31。状态：独立novelty review为`GO 6.6/10`，但只批准冻结premise；该premise已在双语料
test上失败，裁定 **`STOP_BEFORE_FORMAL_METHOD`**。没有训练新模型、没有扫描beta、没有修改PS公式。
权威输出：`runs/20260831_temporal_path_size_choice_mil/premise/analysis.json`。

## Frozen developmental test premise结果

| corpus | beta | pooled AP | pooled ROC | within ROC |
|---|---:|---:|---:|---:|
| HateMM | 0 | .34520 | .64844 | .70074 |
| HateMM | 1 PSL | .34411 | .66317 | .68279 |
| HateClipSeg | 0 | .49737 | .45002 | .47761 |
| HateClipSeg | 1 PSL | .52495 | .51726 | .47006 |

两语料within均下降，已经足够否决formal training。机制风险也实际发生：HMM exact-whole top fraction
`.0140→.0888`，HMM/HCS long-proposal posterior mass分别`.0333→.0500`与`.0313→.1248`。HCS虽然
whole/near-whole top fraction下降，但可纠正错误top的`log PS - best-IoU log PS`均值为`+.84385`，与
“错误proposal应因重叠获得更小path size”的必要方向相反。

`beta=1`确实让全候选等倍复制的bag evidence/frame readout数值级不变，也显著减小near-duplicate和thinning
下的bag log-evidence变化；但它在HMM/HCS都让对应frame ranking Spearman低于`beta=0`。因此PSL稳定的是
bag级总质量，不是本项目需要的秒级排序。该结果不是“值得调beta”的信号，而是task mechanism被证伪；不实现
outside-option formal MIL，不跑MHC-EN/ZH。

## 直接证据与研究问题

当前 MultiHateLoc 的 test branch oracle 说明 modality readout 存在大缺口；multimodal P-MIL 的 frozen test
error analysis进一步显示 proposal candidate set仍有定位上限（HMM/HCS proposal-oracle within
`.73952/.63450`），但full scoring把HCS `48/79`个视频的最高分给whole-video proposal，并大量产生
constant modality ranking。刚淘汰的deletion-carrier ItS2CLR又显示auxiliary representation loss几乎不改变
最终排序：core-vs-broadcast per-video score Spearman在HMM/HCS为`.97568/.99723`。

因此下一候选只处理一个readout问题：大量重叠proposal不是独立证据，但普通proposal softmax/top-k MIL把
每个重叠interval当成独立alternative；同一局部cue被枚举成多个近重复proposal后，其总选择质量随枚举密度
膨胀，弱video label的梯度会奖励proposal multiplicity而非时间定位质量。

所查看的developmental test artifacts：

- `runs/20260831_multimodal_pmil_baseline/pilot_seed234/test_error_analysis.json`
- `runs/20260831_owner_abstaining_its2clr/pilot_seed234/test_error_analysis.json`

这些test证据只用于生成机制；test label不进入gradient或checkpoint selection。

## 跨任务来源

来源方法是交通route-choice中的 **Path-Size Logit (PSL)**。普通multinomial logit在两条route共享大量link时
违反独立无关选项假设：重复/高度重叠route会虚增该方向的总choice probability。PSL把每条route相对choice
set的非重复path size写入utility，按共享link数量折减重叠alternative。

初步来源与边界：

- Ben-Akiva / Ramming 系谱的PSL用于route-choice overlap correction，不是video或hate方法。
- P-MIL用于WTAL并直接训练proposal classifier，但不建模候选proposal因时间重叠产生的choice-set
  multiplicity；它是本项目baseline，不是本候选的novel来源。
- temporal proposal graph、NMS、IoU loss与soft-NMS会利用overlap，但通常做message passing、去重或
  inference suppression；本候选必须证明不是把固定overlap penalty换名。

正式查新必须确认PSL或等价的choice-set size correction没有用于hateful video detection/localization，并
检查它在MIL、WS-TAL、temporal grounding和proposal detection中是否已有等价实现。

## 候选核心：把秒当link、proposal当path

给同一视频的候选interval集合 `P`，proposal `p`的原始learned utility为`u_p`。时间秒`t`相当于route link，
`n_t`是包含`t`的候选proposal数。标准离散版本的path size为：

`PS_p = sum_{t in p} (1 / |p|) * (1 / n_t)`。

完全独立proposal的`PS=1`；当choice set只由`J`个完全重复proposal组成时，各自`PS=1/J`。更一般的严格
性质是把**整个候选集**等倍复制`J`次：每个`t`的`n_t`乘`J`，每个proposal的`PS`除以`J`，因此不会仅凭
全局重复枚举获得`J`倍总choice mass。这里只声明并测试整个候选集等倍复制的不变性；只重复其中一个
subgroup时，其他overlap会改变occupancy，不声称严格不变。候选把它写进训练时proposal utility：

`v_p = u_p + log(PS_p)`。core固定`beta=1`；只有此时整个候选集等倍复制后的总choice mass才严格不随`J`
变化。learned-beta只能是未来额外arm，不能替代core或宣称上述严格duplication invariance。

每个bag另有一个learned outside/background option `v_0`。positive video最大化“选择任一hate proposal”的
概率，negative video最大化outside option概率：

`P(y=1|V) = sum_p exp(v_p) / (exp(v_0) + sum_p exp(v_p))`。

推理frame score不是post-hoc NMS或calibration，而是训练时同一个choice model的proposal posterior对覆盖秒
的marginal；proposal utility、path-size correction与outside option共同接受bag-label梯度。四个主语料仍
分别训练，validation只在固定arm内选checkpoint，随后立即test全部三项指标。

## 为什么可能是non-trivial adaptation

PSL原任务有observed route choice；本任务只有video是否含hate，没有observed proposal choice。adaptation把
proposal identity设为latent choice，并增加negative-bag可识别的outside option，把route重叠校正嵌入MIL
bag likelihood及frame marginal，而非在prediction后做overlap suppression。任务机制是可证伪的：若
proposal duplication确实在弱监督下放大同一局部cue，PSL core应对候选集的重复/细分保持frame ranking稳定，
并在HMM和HCS同时降低whole-video/高重叠错误proposal的posterior mass。

但当前机制仍有两个重大风险，review任一确认即停止：

1. `PS_p`可能偏爱含有稀有边缘秒的whole-video proposal，反而加重已观察到的length shortcut。
2. 若MIL/WS-TAL已有等价的overlap-normalized log-sum-exp、set-size correction或coverage marginal，本候选只是
   已知regularizer重写，不满足novelty标准。

## 只允许的最小前提与pilot

先在frozen P-MIL候选集做不训练的test premise analysis。看结果前固定如下定义，不选per-corpus参数：

1. frozen utility固定取三模态P-MIL `hate_softmax * sigmoid(attention) * sigmoid(completeness)`的均值
   `s_p`，再令`u_p=logit(clamp(s_p,1e-6,1-1e-6))`；不得换成CAS-only或pre-sigmoid arm。
2. proposal posterior固定为`softmax_p(u_p + beta log PS_p)`；frame readout固定为覆盖该秒的posterior之和。
   premise不拟合outside logit，因为outside不改变proposal-conditional ranking。
3. 仅在positive test video且proposal oracle IoU `>=.5`时判断可纠正性；frozen top proposal的GT IoU `<.3`
   定义错误top，并比较它与该视频best-IoU proposal的`log PS`。两语料都必须至少有一个该类case，且
   `mean(log PS_error - log PS_best) < 0`，否则PSL没有按预期更强惩罚可纠正的错误top，前提失败。
4. HMM/HCS都必须满足：`beta=1`相对`beta=0`的within ROC不下降；exact whole-video top fraction下降；
   top duration ratio中位数不升；`duration>=2T/3`的top fraction下降；long-proposal posterior mass不升，且
   near-whole/length四项至少一项严格改善。pooled AP/ROC完整报告但不能代替within门。
5. 对同一frozen utility固定做三种candidate-set perturbation：复制全部proposal一次；加入每个proposal向右
   平移1秒的合法near duplicate；按lexicographic interval顺序保留偶数位并强制保留whole proposal的grid
   thinning。near-duplicate与thinning上，`beta=1`相对`beta=0`必须同时降低bag log-evidence绝对变化，并
   提高原集与扰动集frame ranking Spearman；两项均比较同一语料的video-level mean。Spearman只允许在
   `beta=0/1`都finite的逐视频配对cohort上比较，且该cohort必须等于完整test cohort；任一undefined即
   fail-closed。复制全部proposal时，`beta=1`的bag evidence与frame score必须在**每个视频**数值级不变，
   以全cohort maximum absolute error `<=1e-10`为门；`beta=0`的frame posterior也会因全体等倍复制而不变，
   所以不要求这一项严格优于beta=0。

任一语料、任一硬门不过即`STOP_BEFORE_FORMAL_METHOD`，不能扫描beta、改path-size、改utility/readout、改
IoU或length cutoff，也不能按语料选择。前提通过后，formal
pilot才训练同一P-MIL backbone的三臂：普通latent-choice MIL、只加outside option的capacity control、完整
PSL core；另做candidate duplication invariance control。core必须在HMM/HCS都优于capacity control，至少一边
within提升`>=.020`，并且两语料全部三项test指标超过固定SOTA，才可晋级。

## 当前claim上限

若后续全部成立，只能claim：把route-choice的path-size correction改造成带latent proposal与outside option的
弱监督temporal choice likelihood，消除重叠proposal枚举导致的evidence-multiplicity bias。不能claim首次
proposal overlap modeling、首次MIL proposal selection、首次outside option或首次temporal marginalization。

实际premise没有成立，所以本轮不能提出上述方法claim；只保留“PSL能稳定bag choice mass但损害双语料
within-video frame ranking”的负结果。
