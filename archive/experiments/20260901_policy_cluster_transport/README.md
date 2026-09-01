# Policy-constrained cluster transport MIL

> 已淘汰（2026-09-01）：完整 validation search 与 HMM/HCS 双 test 后，policy core 在两语料 within 都输 binary/unconstrained CASE control，六项 SOTA 全败；有效部分是 generic clustering/background separation，不是 policy-constrained adaptation。

截至 2026-09-01。RESET6 candidate 2；已完成实现、唯一跑前基础审查、完整 validation search、两个 matched controls 与 HMM/HCS test。

## Failure 与结构假设

当前 POWA 在HMM matched anchor为AP/pooled ROC/within
`.584460/.804897/.596995`，HCS为`.575832/.545819/.516630`，仍未过六项门。Lexical
region-memory对HMM压缩视频内排序、对HCS仅小升；该signal family已关闭。本候选不使用lexical、
test GT、teacher ensemble、当前top-K责任或inference calibration。

实际可用结构是 hateful positive bag并不是单一同质“action class”：HMM要求targeted hostility，
HCS还允许untargeted abuse、violence、sexual与self-harm；negative train bag则给出certified
background。POWA已有train-only sparse primitive supervision和固定policy compiler，但普通framewise
primitive BCE/MIL没有显式阻止不同policy primitive与background在表示空间混成同一簇。

## 跨任务来源与 task adaptation

来源为Liu et al., ICCV 2023 CASE：通过snippet clustering、cluster classification与optimal-
transport self-labeling，在video-level supervision下分离foreground/background。

本adaptation保留其load-bearing clustering + OT self-labeling，但不把hateful video直接压成一个
generic foreground：

- 在POWA `shared_rep`上学习`background + six policy states`七个cluster prototypes（targeted
  hate、untargeted abuse、violence、sexual、self-harm、protected context）；
- negative bags的有效snippets只可向background transport，形成同语料certified background anchor；
- positive bags允许background与其policy可用primitive states，并保留null/abstain mass；
- OT的row mass来自有效snippet，column上限按policy clause分组而非固定单一foreground比例；
  policy-invalid primitive在对应语料不可接收mass；
- transported soft assignments监督POWA既有primitive logits，原始POWA bag/base/teacher losses全部保留；
- final `frame_prob`仍由同一primitive logits、AWB和固定policy compiler产生，cluster prototypes与OT
  在test完全删除。

相对CASE，non-trivial delta是把binary F/B self-labeling改为“negative-anchored background +
policy-valid latent primitive transport + clause-level mass feasibility”，以处理同一个positive video
可能由不同逻辑组合成立、未出现primitive必须abstain而不能被伪标的hateful-video结构。

## 六项路径、control与可证伪预期

- pooled AP/ROC：negative bags的background cluster直接抑制跨视频benign false positives；不同
  harmful primitive不会被一个binary foreground centroid互相抵消。
- within ROC：positive bag只把与policy-valid primitive prototype一致的局部mass写回primitive head，
  background与未出现primitive保留abstention，避免整段binary broadcast。
- inference只输出一个raw POWA `frame_prob`，无cluster、OT、teacher blend或postprocess。

正式 control 有两个，均保持同一七cluster、transport预算、loss权重与训练量：

- `binary` 将positive transport退化为CASE式generic foreground/background，不使用policy state
  feasibility；这是隔离task adaptation是否load-bearing的主matched control。
- `permuted` 循环置换六个policy state到primitive targets的对应关系，background不动；这是语义
  对应关系control，不能单独承担机制证明。

Core必须在HMM/HCS test within都胜`binary`，且每个语料至少一个pooled指标相对matched POWA
anchor提高；最终仍要求六项全部超过固定SOTA threshold。

若novelty通过，每语料独立跑2个learning rate `{1e-4,2e-4}` × 3个OT loss weight
`{.05,.2,.5}` × 2个transport temperature `{.05,.2}`，共12个core trial，另跑2个matched
POWA anchor。每trial用official POWA完整5 epochs。Validation在同learning-rate anchor pooled
AP/ROC不低于`-.005`的配置中按within、AP、ROC联合选配置与checkpoint；两语料锁定后训练selected
binary与permuted controls并立即双test。无smoke。

## Novelty边界

独立review必须实际检索CASE/OT clustering/policy-constrained latent clustering是否已用于
hateful-video detection/localization，并与项目已失败的P-MIL、policy-simplex uncertainty、
event-slot、background prototype与普通pseudo-label链比较。若只是CASE直接换类别名、普通组件拼接，
或policy constraints不能产生区别于POWA自身primitive BCE的load-bearing训练信号，则STOP。

## 正式结果、error analysis 与决定

权威结果为 `runs/20260901_policy_cluster_transport/formal_seed234/summary.json`。每个语料完成
2个anchor与12个policy完整validation trials，锁定配置/checkpoint后训练binary和permuted controls，
再立即test；没有smoke。首次正式运行在首个policy trial完成前触发projection tensor类型错误；只做
一行修复与无训练回归确认后重启，最终正式run完整结束。

- HMM anchor/binary/permuted/policy 的AP、pooled ROC、within分别为
  `.597684/.801849/.605852`、`.579737/.801058/.601712`、
  `.585211/.796831/.580744`、`.587251/.797688/.579366`。Policy相对binary为
  `+.007514/-.003370/-.022346`，相对anchor为`-.010434/-.004162/-.026486`。
- HCS四arm分别为`.573072/.538007/.515481`、`.609860/.574532/.525104`、
  `.559091/.526197/.508485`、`.584762/.557686/.507993`。Policy相对binary为
  `-.025098/-.016846/-.017111`，相对anchor为`+.011690/+.019679/-.007487`。
- 两语料policy三项均未超过固定SOTA threshold；`mechanism_gate=false`、
  `hmm_hcs_all_sota=false`。

唯一post-test error analysis读取四arm prediction与test GT。HMM policy-minus-binary within在正例
占比四分位全部为负：`-.014109/-.019176/-.028306/-.028184`；HCS为
`+.021381/-.015821/-.007027/-.070092`。Selected policy训练末HMM harmful/abstain mass为
`.0541/.0420`，HCS为`.8741/近0`；HCS的宽policy可行域几乎不约束assignment，而HMM窄约束在所有
占比组都伤害排序。Binary control在HCS AP达到`.609860`但仍非SOTA，且它不支持policy novelty。

决定：关闭policy-constrained cluster transport family；不扫描column budget、state数、abstain、
temperature或prototype变体。本方法计RESET6第二次正式performance failure。
