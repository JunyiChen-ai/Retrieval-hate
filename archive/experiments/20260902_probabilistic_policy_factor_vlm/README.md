# Single-VLM Probabilistic Policy-Factor Localization

> 淘汰原因（2026-09-02）：独立 novelty review 裁定 `STOP 4.6/10`。PVLR 来源未检出被 hateful-video task 占用，但候选实际只保留 generic Gaussian projection/text matching，再组合目标领域已有 policy-role/counter-evidence 与项目已有 global-prior/centered-local decomposition；role uncertainty没有新的监督约束，atomic control也不能隔离正确 policy semantics。未实现、未训练、未生成 prediction。

截至日期：2026-09-02。RESET7 第一个 Gate 3 STOP brief；formal performance窗口保持`0/3`，累计连续failure保持`14`。

## 失败、现有证据与 gain budget

目标是同一个方法在 HateMM（HMM）和 HateClipSeg（HCS）各自独立训练，并在 test 同时严格超过固定 AP、pooled ROC、within-video ROC 门：HMM `.593832/.816184/.631532`，HCS `.619371/.605022/.561908`。相对 official MultiHateLoc seed-234 starting point，六项缺口分别是 HMM `+.100831/+.077925/+.003076`，HCS `+.066350/+.060950/+.038207`，依据 `research-wiki/RESET6_GOAL_GAP_AUDIT.md`。

本候选不使用 lexical、POWA、VERA、MultiHateLoc 或任何其他模型的 prediction、feature、posterior、pseudo-label。现有 correction evidence 只用于 admission，不进入训练：固定单一 `Qwen/Qwen2.5-VL-7B-Instruct` 对完整 HMM/HCS positive test cohort 做相同 16 秒窗口判断，within ROC 为 `.578023/.550232`，依据 `runs/20260830_vlm_order_pilot/stage_t_eval.json`；两边均有局部语义方向，距离 fixed within 门为 `.053509/.011676`。同语料 train-span-supervised temporal model的 test within ceiling为 HMM `.7495`、HCS `.5989`，依据 `experiments/20260830_powa_within_diagnosis/README.md`；它只证明原始输入中存在足够局部信息，不进入弱监督训练。需要同时解决的 pooled 缺口由同一模型的 video-global prior承担，不能靠另一个 classifier、score blend或校准补齐。

## 跨任务来源

唯一主要来源是 Lim et al., ACM MM 2024 的 Probabilistic Vision-Language Representation for Weakly Supervised Temporal Action Localization（PVLR）。PVLR把 snippet 与 atomic action-category text编码为概率分布，并用分布内、分布间对比学习缓解 deterministic vision-language representation 对细粒度动作的不确定性。来源任务是 THUMOS/ActivityNet 的 weakly supervised action localization，不是 hateful-video detection/localization。

独立 review 必须检索 PVLR 的完整 probabilistic vision-language core 是否已经用于 hateful-video detection/localization，并比较 MultiHateLoc、LELA、MARS、SafeLens、CLARA、LEAF、POWA 以及本项目已经关闭的 evidential/policy-cluster/prompt/scalar families。

## 单一 task adaptation 与最终 score 路径

Atomic action label可以直接作为一个 text distribution；binary `hate` 不行。Hatefulness是一个政策条件事件：HMM 至少需要 protected target 与 hostile conduct 在局部共同成立，并排除 quotation/condemnation/reporting 等 exception use；HCS 还允许其标注政策中的 target-free harmful categories。直接把类别名从 `action` 换成 `hate` 是不合格 control，而不是本方法。

方法只加载一个 `Qwen/Qwen2.5-VL-7B-Instruct` checkpoint。对连续视频块一次 forward 得到带真实时间地址的 frame/ASR token states；不读取任何其他 encoder、teacher 或 localizer artifact。一个共享 probabilistic projection把每秒 state映射为 diagonal Gaussian `q_t`，同一个 Qwen text stack对固定政策短句产生 target、conduct、exception 与 target-free-harm factor distributions。每个 factor与`q_t`的可学习 overlap给出 log evidence及其 uncertainty。

HMM 的局部 log likelihood 是 `target + conduct - exception`；HCS 是 targeted clause与 policy允许的 target-free harmful clauses的 differentiable union，再减 exception。高方差 factor的precision低，不能像确定性 prompt similarity一样由一个缺失或含混 role任意支配。所有 policy factor、projection与 temporal context都属于同一 checkpoint/pipeline并联合训练，不是多个独立模型或多教师。

唯一 frame logit使用同一模型的 Bayesian prior/likelihood分解：

`z_vt = g_v + ell_vt - mean_valid(ell_v)`。

`g_v`由同一组 Qwen states的masked mean经一个共享 global projection得到；`ell_vt`是上面的 policy-factor likelihood。严格 centering使 `g_v`负责跨视频 pooled separation而不改变视频内排序，`ell_vt`负责 within ordering而不能用整段常数冒充局部 likelihood。训练 bag probability也只从同一组`z_vt`聚合；没有独立 bag-classifier bypass。Negative train video对全部有效`z_vt`施加 benign likelihood，positive video用固定 smooth existential likelihood。Test只输出`sigmoid(z_vt)`这一条1fps raw score，不做模型融合、CDF、threshold tuning、routing或 post-hoc calibration。

相对 PVLR 的 non-trivial adaptation 是把一个已知 atomic category distribution改造成由 moderation-label semantics决定的 factorized event likelihood，并把 uncertainty变成 role evidence的precision：target、conduct与exception不是多个类别投票，只有满足语料政策公式的联合证据才能进入 local likelihood。再以同一 posterior中的 global prior与centered local likelihood分别承担 pooled 与 within；不能从另一个模型补分。

## 可证伪预期、matched control 与执行

主 matched control 是 source-faithful atomic PVLR：同一 Qwen、同一 probabilistic维度、参数量、temporal context、global/local公式、MIL、训练量与 validation选择，但所有 policy factors替换成等数量的可学习 atomic `hateful content` text distributions，并按普通 log-sum-exp形成`ell_t`。它隔离增益是否来自 policy-factor likelihood，而不是 probabilistic representation或容量本身。

机制预期：core相对 atomic control在 HMM/HCS test 的 AP、pooled ROC、within ROC均同向非负，并在每个语料至少一个主要缺口上达到 `+.02`；正确 policy factors相对 fixed within-video circular time permutation的 likelihood在两语料都应更有用。若 core不在两语料共同胜 atomic control、主要增益低于预算，或最终 score主要由`g_v`形成整段常数，则关闭该 family，不扫描 factor同义词、公式、分布族或 Qwen layer续命。最终晋级仍只认六项全部严格超过固定 SOTA。

Novelty通过后才实现。正式配置固定使用一个 Qwen checkpoint，不新增 producer sweep。每个语料独立进行完整 validation search：learning rate `{3e-5, 1e-4}` × probabilistic dimension `{64,128}` × factor regularization `{.05,.2, .5}`，共12个 core trials，每个均跑完整 official epoch budget；validation在预注册 pooled non-inferiority约束内联合选择最佳超参数和对应checkpoint。锁定后训练同配置 atomic control，立即在 HMM/HCS test运行共享 evaluator的三个固定指标。无 smoke；正式启动前只做一次影响实验观察的基础 technical review。

## Claim boundary

若 novelty 与 performance 均成立，最多 claim：将 atomic-category probabilistic vision-language WTAL改造成单一 VLM 内的 policy-factor event likelihood，使 role uncertainty、video-global prior与centered temporal likelihood共同产生一个 weakly supervised hateful-localization posterior。不能 claim 首次 probabilistic embedding、首次 prompt learning、首次 policy reasoning、首次 multimodal hate detection或严格可识别的真实 role discovery。
