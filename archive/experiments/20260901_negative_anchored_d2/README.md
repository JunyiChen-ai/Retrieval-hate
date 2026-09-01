# Negative-anchored D2 denoising MIL

> 淘汰：独立 novelty review 裁定 `STOP 4.8/10`；D2 来源结构可用且未检出被目标任务占用，但新增的 negative-bag 全帧 BCE 属于标准 dense-negative supervision，重复既有失败链，不构成 non-trivial task adaptation。未实现、未训练、未生成 prediction。

截至 2026-09-01。RESET6 novelty-only proposal；已停止，正式 performance failure 窗口保持 `2/3`。

## Failure与可用结构

Policy-cluster transport的novel policy core在HMM/HCS within都输binary CASE；HCS binary clustering
提高pooled与within但未过SOTA，HMM则未改善。该family已关闭。本候选不使用lexical、teacher
ensemble、test GT、self-derived modality responsibility或inference calibration。

实际可用结构是每个语料自身train video label：negative video的所有有效帧是certified background；
positive video只保证至少一个hateful witness，内部帧不做正/负伪标签。当前POWA只有top-down raw
frame activation，缺少独立bottom-up foreground process来去噪其foreground/background混淆。

## 跨任务来源与task adaptation

来源为Narayan et al., ICCV 2021 D2-Net。保留其两个load-bearing部分：(1) top-down activation
加权的foreground/background embedding discriminability；(2) bottom-up foreground attention与
top-down activation的snippet-levelDMI，以及bag prediction与video label的batch-levelDMI。

Task adaptation为asymmetric certified-negative anchor：

- 在POWA `shared_rep`上增加一个bottom-up foreground head `g_t`；
- negative train bag对全部valid `g_t`施加exact background BCE；
- positive train bag不产生dense target，仍只由原POWA bag loss、D2 discriminative loss与DMI约束；
- snippet DMI在每个视频内对`g_t`和原POWA `frame_prob`的二类joint matrix做determinant MI；
  video DMI在batch内对最终bag prediction与真实video label做同样约束；
- final single-model frame score按D2 source path为 `frame_prob * g_t`，不融合其他模型或手工校准。

与action-only WTAL setting相比，hateful数据含显式negative videos；把其所有帧用作bottom-up
background certificate改变了D2 attention的监督条件，同时严格保留positive-unselected abstention。

## 六项路径、control与可证伪预期

- pooled AP/ROC：negative frames直接压低bottom-up foreground，DMI同时保持bag label separation。
- within ROC：top-down与bottom-up必须在positive video内部形成共同foreground/background variation，
  embedding discriminability扩大局部间隔；negative anchor不把positive background伪标为负。
- inference只运行一个POWA+D2 head，输出单一乘积score。

Matched `source_d2` control保留同一bottom-up head、D2 discriminative/DMI、参数量、训练量、final
score与validation选择，只把negative-anchor BCE权重设为0。Core必须在HMM/HCS test within都胜
source_d2，且每个语料至少一个pooled指标胜matched POWA anchor；最终仍要求六项全部超过固定SOTA。

若novelty通过，每语料独立跑2个learning rate `{1e-4,2e-4}` × 3个D2 loss weight
`{.05,.2,.5}` × 2个negative-anchor weight `{.1,.5}`，共12个core trial，另跑2个matched
POWA anchor。每trial为official POWA完整5 epochs。Validation在同learning-rate anchor pooled
AP/ROC不低于`-.005`的配置中按within、AP、ROC联合选择配置/checkpoint；两语料锁定后训练selected
source_d2 control并立即双test。无smoke。

## Novelty边界

独立review必须实际检索D2-Net、DMI denoising、negative-video anchored bottom-up attention是否已用于
hateful-video detection/localization，并与项目benign insertion、binary CASE、background prototype、
contrast/DMI或ordinary dense-negative loss比较。若只是D2直接换数据集后加BCE、或exact negative
anchor不能构成non-trivial hateful weak-supervision adaptation，则STOP。
