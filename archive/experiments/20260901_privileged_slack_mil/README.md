# Privileged-slack temporal MIL

> 已淘汰（2026-09-01）：novelty `STOP 4.1/10`；已有 MIL-PI 覆盖 train-only textual privileged information + correcting slack，当前 temporal MIL delta 只剩标准 latent witness 与 loss weighting。未实现、未训练。

截至 2026-09-01。RESET6 提案；在 novelty 门停止，未实现或训练，不计正式 performance failure。

## Failure、headroom 与实际可用 signal

RESET6 candidate 1 的 lexical region-memory在HCS相对POWA三项小升、HMM三项下降；说明把
弱 lexical support硬写成positive region representation不是共同机制。另一方面，既有固定 test
diagnostic 已证明 lexical/POWA/VERA/MultiHateLoc 的共享 tuple 在HMM/HCS都存在六项过门
headroom；这只是 upper bound，不能作为ensemble或teacher target。

训练时实际可用且独立于当前 POWA confidence 的信号是：(1) 每个语料自身 train video labels
产生的五折 OOF lexical locality；(2) frozen VERA 的 train-only local evidence。两者均不读取
test GT。候选不把它们当概率标签、排序目标或 inference input，而只作为 privileged variables。

## 跨任务来源与 task adaptation

来源为 Learning Using Privileged Information / SVM+：训练时额外信息不进入部署 predictor，
而由 correcting function参数化样本约束的 slack。这里把该原理 adaptation 到 temporal MIL：

- 主模型仍是单个 POWA，原始 bag loss 与 raw `frame_prob` 保留。
- negative train bag 的有效帧具有确定 benign 标签；其 frame margin slack 由 privileged corrector
  预测，避免 frozen auxiliary evidence不可靠时强迫主模型拟合。
- positive train bag 不产生 dense positive pseudo-label。它只要求至少一个 latent witness满足
  positive margin；每个候选 witness 的 violation 加上由 privileged corrector预测的非负 slack cost，
  通过 soft-min 形成 bag constraint。因而 privileged information只改变“违反弱监督约束的代价”，
  不给 frame指定 teacher score或 pairwise order。
- corrector只读每帧 lexical 与 VERA evidence，并带 video内中心化输入；其参数和输出在 inference
  完全删除。最终 evaluator只看到主 POWA raw score。

这针对 hateful localization 的具体机制是：positive bag 内没有 frame labels，普通 top-K由当前
student自确认；privileged slack允许独立局部证据只调节latent witness约束的可信度，同时保留
bag label作为唯一正负方向，避免上一轮把 lexical高分直接当hateful region。

## 六项路径、control 与可证伪预期

- pooled AP/ROC：全部 certified-negative frames 的 margin仍由主student承担；privileged slack
  只对确有辅助歧义的负帧容错，减少错误强约束，同时positive bag仍须产生raw positive witness。
- within ROC：positive latent witness的约束代价随同一时间的独立 privileged evidence变化，正确
  timing应使真实局部witness比video-global topic更便宜。
- inference不运行corrector，不读lexical/VERA，不融合、校准、route或蒸馏任何score。

Matched control逐视频对两个 privileged 时间序列作固定 half-video circular shift，其网络、参数量、
slack预算、训练量和主模型完全相同。Core必须在HMM/HCS test within都胜shifted，并在两语料至少
各有一个 pooled指标相对matched anchor提高；最终晋级仍要求六项全部超过固定SOTA threshold。

若novelty通过，每语料独立做2个learning rate `{1e-4,2e-4}` × 3个slack weight
`{.05,.2,.5}` × 2个slack temperature `{.1,.3}`，共12个core trial，另跑2个matched POWA
anchor。每个trial是official POWA完整5 epochs。Validation在同learning-rate anchor pooled
AP/ROC不低于`-.005`的配置中，按within、AP、ROC联合选择配置与checkpoint；两语料锁定后训练
selected shifted control并立即双test。无smoke。

## Novelty 边界

独立review必须检索LUPI/SVM+、privileged slack、teacher-guided MIL或数学等价方法是否已用于
hateful-video detection/localization，并判断它是否只是 ordinary KD、pseudo-label、loss weighting
或已失败 lexical selector的改名。最窄claim只能是“train-only privileged correcting function
parameterizes asymmetric negative-frame and latent-positive MIL slack，部署时删除corrector”。若该
delta不构成non-trivial task adaptation，或与已关闭teacher/lexical链同构，则STOP。
