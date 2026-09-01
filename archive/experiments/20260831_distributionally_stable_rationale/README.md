# Distributionally stable rationale — 淘汰：只识别frozen classifier最小证书

截至 2026-08-31。独立novelty review `STOP 4.8/10`，独立解析审查`STOP`；未实现、未训练、
未生成新prediction。多replacement只增强teacher-faithfulness：max-pool单峰严格退化top-1，
topic/broadcast产生近全或全视频mask，不能建立真实hate span方向性。

## 直接问题

当前弱监督模型能做video discrimination，却不能给positive video内部proposal稳定定向：V26 real
counterfactual模型video AP达到`.88824`，temporal within ROC只有`.55994`；deletion-carrier方法最终
frame ranking又与broadcast control几乎相同。单帧deletion effect和auxiliary attribution都不能保证
机制进入最终定位输出。

## 跨任务来源

候选拟适配Sufficient Input Subsets、selective rationalization、sufficiency/comprehensiveness rationale
learning这类解释方法：寻找一个最小输入子集，使原预测在只保留该子集时仍成立，而删除该子集后预测
消失。来源核心是否已用于hateful video detection/localization，以及video rationale/WTAL中是否已有数学
等价方案，必须由独立reviewer检索。minimal rationale、deletion attribution、mask learning、MIL本身都
不能单独claim novelty。

## Task-specific adaptation

每个语料先用train video labels训练一个OOF video classifier `F`；对产生rationale的视频，`F`永远来自
没有训练过该视频的fold并冻结。对原始三模态序列`x`和mask `m_t in [0,1]`，从该语料negative-train
videos构造一个在运行前冻结的benign replacement集合`R(x)`。同一个temporal mask同时作用于三个模态，
missing modality仍由原availability mask控制。

对每个replacement `r`定义：

- kept-only input：`x_keep(m,r)=m*x+(1-m)*r`；
- removed input：`x_remove(m,r)=(1-m)*x+m*r`。

selector不最小化一个可任意权衡的加权和，而求lexicographic constrained problem：

1. 对所有`r in R(x)`，kept-only的frozen positive margin不低于原prediction的固定fraction
   （distributional sufficiency）；
2. 对所有`r`，removed input降到固定normal margin以下（distributional necessity）；
3. 在满足1–2的可行mask中先最小化`sum_t m_t`，再以固定tie-break最小化replacement之间的mask
   disagreement；不使用GT、固定事件长度或test-selectedsparsity。

最终rationale score是原始输入下每秒的最小可行membership/stability，不是`F`的attention、单秒
deletion effect或auxiliary loss。正式方法若获准，使用train OOF rationales监督一个单一temporal student；
student的frame posterior就是test输出，因此rationale机制不能被另一个fused readout忽略。

任务故事：whole-video topic可以支持bag prediction，但若真正local hate witness存在，则同一小段在多种
benign替换下应同时足以维持positive decision、并在被移除后使decision消失。只做necessity会选择任意
敏感点，只做sufficiency会保留topic；二者与跨replacement一致性共同定义候选witness。该定义仍不声称
rationale等于causal hate span，必须由test ranking和controls证伪。

## 最小冻结premise

先对现有MultiHateLoc OOF/test classifier做HMM/HCS一次固定selector premise；selector不读test GT，
完成prediction后立即用shared evaluator报告test pooled AP/ROC/within ROC。固定controls：

1. full stable sufficient+necessary core；
2. single-frame deletion effect（V26式）；
3. sufficiency-only；
4. necessity-only；
5. single replacement而非distributionally stable；
6. same-mass random与frozen attention/top-k；
7. replacement identity在视频间均衡打乱的control。

core必须在HMM/HCS within相对原MultiHateLoc和全部controls同向提高，至少一边`>=.020`；pooled AP/ROC
两语料均不得相对起点下降超过`.010`。rationale mass不得退化为单点、接近全视频、固定片头/片尾，且
kept/removed constraints在原feature空间重新forward必须真实成立。失败即STOP，不扫描margin、replacement
数、优化步数、mask temperature或按语料改规则。

## Formal gate

只有premise与独立result-chain审计均通过才实现student。HMM/HCS各自独立train、validation只在固定arm
内选checkpoint、立即test三指标；六个SOTA单元全部严格过门才扩MHC-EN/ZH。test无branch selection、
score ensemble、calibration、threshold search或post-hoc smoothing。

## 独立review必答

1. sufficient/necessary rationale selection是否已经用于hateful-video localization？
2. 与V26 temporal intervention、deletion-carrier、INVASE/SIS/HardKuma及video rationale方法的精确边界？
3. replacement-robust sufficiency+necessity能否排除classifier-selector collusion、topic-frame单点与mask
   artifact，还是仍不可识别？
4. lexicographic最小mask是否系统性鼓励top-1而不是完整span？
5. 什么解析反例会在实现前直接判STOP；若没有，最小可运行premise应如何锁定？
