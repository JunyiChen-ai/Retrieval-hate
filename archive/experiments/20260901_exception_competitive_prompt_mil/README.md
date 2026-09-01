# Exception-Competitive Prompt MIL

> 淘汰：独立 novelty review 裁定 `STOP 4.7/10`。PE-MIL 来源未检出被目标任务占用，但 harmful-use/exception-use signed competition 与 MARS counter-evidence、POWA contextual negation、LB-SCGP quotation/condemnation/reportage exception 在机制上实质重合；局部 prompt、减法与 exclusivity loss 只是实现变化。未实现、未训练、未生成 prediction。

截至 2026-09-01。RESET6 novelty-only proposal；已停止，正式 performance failure 窗口保持 `2/3`。

## 已有 test failure 与可用信息

HMM 的 POWA starting point 已接近 pooled threshold，但 within 仍低于门；HCS 的 VERA 说明
semantic normal/abnormal prompting 有强 pooled 与 within 基础。两语料共同的困难不是缺少另一个
self-derived router，而是同一局部 token/frame 在不同语用中含义相反：直接攻击、威胁或贬损应为
hateful，引用、谴责、报道或反驳同样的词/符号通常不应为 hateful。POWA 的单一 `context`
probability由 binary bag loss间接学习，policy cluster transport又证明把宽泛 policy state当可行列会在
HCS吸收几乎全部 harmful mass；因此本候选不再学习 policy cluster，也不使用现有模型 prediction、
teacher、ensemble、calibration或 test label训练。

Observed headroom 与 train-available correction signal 分开：现有 test blend 只证明数值 headroom，
不进入方法；实际训练可用的是每个语料自身 train video label、对齐的 visual/audio/ASR feature，
以及固定文字定义的 harmful-use 与 exception-use prompt。HMM/HCS独立训练。

## 跨任务来源与 non-trivial task adaptation

跨任务来源是 Chen et al., CVPR 2024 的 Prompt-Enhanced MIL（PE-MIL）：保留其 load-bearing
结构，即 abnormal-aware semantic prompts 动态增强局部表示、normal-context prompt拉开异常与正常
context，并用 video-level MIL训练 dense anomaly score。来源任务是 weakly supervised generic video
anomaly detection，不是 hateful-video detection/localization。

本任务 adaptation 不是把 `anomaly` 改名为 `hate`。Generic abnormal/normal 二分在仇恨语境中不成立，
因为相同显式内容可因 communicative use 相反而翻转标签。模型使用两个共享语义子空间：

- harmful-use prompts：针对受保护群体的贬损、去人化、排斥、威胁，以及数据集允许的其他 harmful
  conduct；
- exception-use prompts：引用/转述、谴责/反驳、新闻或教育性报道、受害者叙述。

每秒 multimodal hidden state分别与两组 prompt做cross-attention；得到 `h_t` 与 `e_t` 后，唯一 raw
frame logit为 `z_t = base_t + alpha * (h_t - e_t)`。Exception不是额外类别或后处理阈值，而是对
同一局部 harmful evidence 的 signed competing explanation。Negative train bag约束所有时间的
`z_t`为低；positive train bag只对top-k `z_t`做MIL，不把未选时间伪标为exception。另加
prompt-exclusivity loss，惩罚同一时间同时由 harmful 与 exception 高置信解释；不加frame伪标签。
推理只输出这一条单模型 `sigmoid(z_t)`。

与 source-faithful PE-MIL 的 matched control保持相同 backbone、prompt数量、参数量、训练量、
validation search和最终公式，但把 exception prompts替换为普通可学习 normal-context prompts，
并移除 signed exception语义约束。这样control检验增益来自 hateful-specific exception competition，
而非 prompt参数量或直接迁移 PE-MIL。

## 六指标路径、可证伪预期与执行

HMM pooled AP/ROC由强 base bag separation与negative-bag全时间约束保持；within由局部 exception
竞争压低正视频中的引用/谴责段。HCS pooled由视觉 harmful-use prompt保留 VERA 式 semantic prior，
within由同一 signed competition区分静态符号的 harmful use与报道/反驳 context。若 HMM/HCS core
within不能同时胜 matched PE-MIL control，或任一语料两个 pooled指标均不胜 matched control，机制
即失败；最终晋级仍要求两个语料全部六项超过固定 SOTA。

Novelty通过后，每语料独立完整搜索 12 个 core 配置：learning rate `{5e-5, 1e-4}` × prompt
strength `{.25,.5,1.0}` × exclusivity weight `{.05,.2}`；另跑同 learning-rate 的 matched PE-MIL
controls。每个 trial 使用完整 official epochs，validation联合选择超参数和checkpoint，锁定后立即
HMM/HCS test三个固定指标。无 smoke。正式训练前只做一次会影响实验观察的基础 technical review。

## Novelty review 必查边界

独立 reviewer 必须实际检索 PE-MIL、hateful-video prompt learning、LELA、MARS、SafeLens/CLARA/
LEAF 及本项目 POWA/LB-SCGP，回答：(1) PE-MIL或其 abnormal-aware + normal-context prompt core是否
已用于 hateful-video detection/localization；(2) local learnable harmful-vs-exception signed
competition是否只是 MARS counter-evidence、LELA multi-stage prompt或 POWA `NOT context` 的实现替换；
(3) matched control是否能隔离 non-trivial adaptation。任一 novelty硬门失败即STOP，不实现。
