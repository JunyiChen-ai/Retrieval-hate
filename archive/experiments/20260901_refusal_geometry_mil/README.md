# Negative-Recentered Refusal Geometry MIL

> 淘汰原因（2026-09-01）：独立 novelty review 裁定 `STOP 4.3/10`。语料级常数 recenter 在标准化 refusal coordinate 中被严格消除，对 orthogonal content head 也只是可由 bias 吸收的常数；因此核心 adaptation 不 load-bearing，剩余方法退化为 frozen activation scalar + 普通 temporal MIL。未实现、未训练，不计正式 performance failure。

截至 2026-09-01。RESET6 第三个正式候选 brief；尚未实现或训练，先过一次独立 novelty 三门。

## Failure、available signal 与目标

HMM当前主要缺pooled AP/ROC，HCS三项都缺；现有test complementarity只证明headroom，不进入本方法。
既有POWA/lexical/VERA/MultiHateLoc prediction、test GT、teacher score、ensemble或calibration都不作为输入。
本候选引入的是此前未使用的 frozen MLLM internal activation：安全对齐模型在拒绝有害请求时形成的
refusal/safety direction。它不是模型生成的hate分数，也不是pseudo label。HMM/HCS各自只用本语料train
video label训练localizer，并独立用validation选超参数/checkpoint。

任务监督结构提供一个与普通multimodal safety不同的锚：negative train video的每个有效局部window都由
bag semantics认证为non-hateful；positive video只保证至少一个hateful window。这个锚可用于估计同域视频输入
相对text-only safety geometry的modality drift，而不把positive未选时间伪标为负。

## 跨任务来源与 task adaptation

来源一是Arditi et al., NeurIPS 2024的refusal direction：chat model residual stream中存在可通过
activation addition/erasure因果操纵refusal的低维方向。来源二是2026年的multimodal refusal-direction工作
MARS/ReGap：textual refusal direction跨模态可迁移，但image/video输入会产生modality-induced drift，需
re-centering或drift correction恢复safety separability。这些来源研究MLLM安全控制/越狱防御，不是
hateful-video detection/localization。

本adaptation不把“拒绝强度”直接当最终分数。对冻结Qwen2.5-VL的每个4秒window（1秒stride），在固定neutral
content-moderation prompt的assistant first-token位置读取预先指定层的hidden activation `a_vt`。每层refusal
direction `d`只由一组固定、成对的harmful-request/harmless-request文本模板在同一冻结模型上形成mean
difference；模板不含任何项目视频、dataset label或span。随后：

1. 只用该语料negative-train windows估计multimodal drift `c = mean(a_neg) - mean(a_text-neutral)`；
2. 局部safety coordinate为 `r_vt = <a_vt-c, d>`，orthogonal content为
   `u_vt=(a_vt-c)-r_vt*d`；所有centering/scale统计在train锁定；
3. 一个小temporal head从`u_vt`产生content logit `q_vt`，唯一frame logit为
   `z_vt = q_vt + softplus(beta)*standardize(r_vt)`；`beta`必须非负，使refusal coordinate只能按来源
   方向增加harm evidence，不能被训练翻转后当任意feature；
4. negative bag对全部valid `z_vt`做background BCE；positive bag只对top-k `z_vt`做MIL。test只输出该
   单模型 `sigmoid(z_vt)`，4秒window以固定triangular overlap-add还原1fps，不做平滑、阈值或模型融合。

这把“用于控制模型是否拒绝”的global behavior direction改造成由same-corpus certified-negative视频几何
重新居中的local temporal coordinate，并让同一个coordinate同时参与bag训练与raw frame readout。它针对
HMM/HCS的机制故事是：absolute safety coordinate承担跨视频pooled separation；逐window activation和orthogonal
content head承担within ordering；negative recentering避免视频模态本身的drift被误当harm。不能claim refusal
direction、activation steering或multimodal drift correction本身新，也不声称该方向等于ground-truth hate。

## Matched controls、可证伪预期与正式执行

三个matched arms共享完全相同的Qwen windows、temporal head、参数量、训练量、validation选择与1fps readout：

- `core`：negative-recentered true refusal direction；
- `uncentered`：同一true direction但`c=0`，检验same-domain multimodal recentering；
- `random_direction`：固定norm-matched、与`d`正交的deterministic random direction，并执行相同negative
  recentering，检验是否只是任意额外scalar/normalization；
- `content_only`：把`beta=0`，保留同一orthogonal content head与容量。

Core必须在HMM/HCS test within都胜`content_only`和`random_direction`，并在两语料至少各胜一个matched
control的pooled AP与ROC；否则refusal geometry不load-bearing。最终晋级仍要求HMM/HCS六项全部超过固定SOTA。

Novelty通过后先完整生成两个语料train/validation/test的4秒、1秒stride local activation cache；生成脚本位于
`scripts/`，cache写`data/refusal_geometry/`并附`PROVENANCE.md`，实验训练代码不写`data/`。不做smoke。
每语料完整搜索12个core配置：layer `{20,24,28}` × learning rate `{1e-4,3e-4}` × initial safety
weight `{.25,1.0}`，每trial完整5 epochs；validation联合选择配置和checkpoint，锁定后训练matched controls并
立即跑HMM/HCS test固定三指标。正式训练前只做一次仅限result-affecting bug的technical review。

## Novelty review 必查边界

Reviewer必须检索refusal-direction/activation-steering、多模态safety geometry、hateful-video detection与
localization、LELA/MARS(hate reasoning)/SafeLens/CLARA/LEAF，以及本项目global-local、negative-density、
prompt/counter-evidence链。重点判断：(1) refusal geometry是否已用于hateful-video task；(2) negative-video
recentered local refusal coordinate + monotone MIL是否是non-trivial task adaptation，还是普通frozen embedding
加scalar feature、prompt scoring或global/local分解的包装；(3) controls能否隔离true direction与recenter效果。
任一硬门失败即STOP，不实现、不换层/模板/名称修补。
