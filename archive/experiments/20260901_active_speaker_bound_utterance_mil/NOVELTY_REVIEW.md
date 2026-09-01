# Independent novelty review: Active-speaker-bound utterance MIL

截至 2026-09-01。本审查只评价 `README.md` 已固定候选的 Rule 12 三项 novelty 硬门；未审代码、未实现、未运行实验，也未提出新 candidate。

## Verdict

**GO — 6.3/10（最窄 claim 下）。**

- Gate 1（允许 adaptation 已有方法）：**PASS**。
- Gate 2（来源方法不得已用于 hateful-video detection/localization）：**PASS（截至本次 primary-literature 检索的窄口径）**。
- Gate 3（必须是 non-trivial task adaptation）：**PASS，但 claim 必须限定为 relation branch 内的 physical-source-exclusive evidence binding；不能 claim 全模型不存在 POWA 旁路，也不能 claim source attribution、stance/endorsement reasoning 或 offscreen-null 的独立贡献已经成立。**

三门均通过，可以进入实现。最终 novelty 仍取决于实际实现保留这里审查的排他 assignment、正式 HMM/HCS test 中正确 assignment 胜 matched permutation，且 adapter-off 证明新增路径实际进入 final score；若实现退化成把 TalkNet posterior/face feature拼到 POWA，则本次 GO 自动失效。

## 一手资料与占用检索

本次检索了 `TalkNet / MAAS / active speaker detection / speech-face assignment / speaker-face assignment` 与 `hateful video / hate video / multimodal hate / hate localization` 的组合，并逐页核对最接近的 hateful-video 方法。没有找到 TalkNet、MAAS 或 speech-to-visible-face assignment 已用于 hateful-video detection/localization 的一手论文。

### 跨任务来源

- Tao et al., [*Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection*，ACM MM 2021](https://arxiv.org/abs/2107.06592)，以及[作者官方实现](https://github.com/TaoRuijie/TalkNet-ASD)。TalkNet 用 audio/visual temporal encoders、audio-visual cross-attention 与 long-term self-attention，输出某条可见 face track 是否正在发言；来源任务是 active speaker detection。
- Alcázar et al., [*MAAS: Multi-Modal Assignation for Active Speaker Detection*，ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Alcazar_MAAS_Multi-Modal_Assignation_for_Active_Speaker_Detection_ICCV_2021_paper.html)。MAAS 将同一 speech event 与画面中的多个 candidate faces 显式建成 assignment problem，并允许 “none” 意义上的无可见说话者情形；来源任务仍是 active speaker detection。

这两项来源都提供 frame/track-level 的物理“谁在说话”判断，不提供 hate label、hate score、protected target、endorsement/quotation/stance 或 hateful span supervision。候选使用冻结 producer，不以目标语料 hate label 训练 producer，符合跨任务 adaptation 的定义。

### 目标任务最近邻逐项核对

- Sun et al., [MultiHateLoc，WWW 2026](https://arxiv.org/html/2512.10408) 使用 modality-aware temporal encoders、dynamic cross-modal fusion、cross-modal contrastive alignment 与 modality-aware top-K MIL。论文全文未出现 active speaker / speaker-face assignment；其 visual feature 是 frame-level scene representation，speech/text 与画面中具体说话者没有排他绑定。
- Sun et al., [LELA，2026](https://arxiv.org/html/2602.09637) 将每秒 speech caption 分别与 image/OCR/music/video captions连接、逐组合提示并取最大分数。它恰好体现候选要区别的 broadcast composition：同秒 speech 与整帧/scene description组合，没有 visible-speaker assignment，全文也未出现 active speaker。
- Wang et al., [SafeLens，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/42390) 在 segment 内提取 timestamped Whisper speech、OCR 和周期性 full-frame descriptions，再交给 policy LLM。它没有 face tracks、active-speaker posterior 或 speech-to-face exclusivity。
- Céspedes-Sarrias et al., [MM-HSD，ACM MM 2025](https://arxiv.org/abs/2508.20546) 融合 video frames、audio、whole-video transcript 与 OCR，并研究 cross-modal attention query/key 组合；没有 active-speaker assignment，也不输出 temporal localization。
- Mathew et al., [TANDEM，2026](https://arxiv.org/html/2601.11178) 以 30-second audio/video chunks、独立 VL/AL structured predictions、cross-modal context 与 tandem RL 输出 classification/timestamps/targets；其 target 是受攻击群体，不是可见 speaker identity。全文未出现 active speaker / face-speaker assignment。
- 当前 POWA-MACIL 使用 `typed moderation primitives → asynchronous predicate-target transport → executable policy dense MIL`。它将 aligned transcript 与 MACIL audio/visual context融合，但不解析 speech 属于哪条 face track；其 target-predicate binding 也不是 audio event 到可见说话者的物理 assignment。

因此，Gate 2 的可辩护范围是：**把 active-speaker / speech-to-visible-face assignment 用作 hateful-video temporal-localization 的 evidence construction 来源，未在上述一手目标领域方法中检出。**不能扩大成“首次进行 speaker attribution”或“首次将 source semantics 用于 hateful video”。

## Gate 1：允许 adaptation 已有方法

**PASS。**

TalkNet/MAAS 是明确的跨任务 producer。项目规则允许使用已有模块；冻结、预训练的 ASD 模型是否由本项目从零训练不是 novelty 门。候选也没有把 TalkNet 本身写成贡献。

## Gate 2：来源是否已被 hateful-video task 占用

**PASS（窄口径）。**

检索到的目标领域方法都停留在 full-frame / clip-level visual evidence、audio、speech transcript、OCR 的融合或 structured temporal reasoning，没有把一个 speech event 排他分配给同秒 candidate face track，再以 `(assigned active face, utterance)` 作为 hateful relation 的基本 evidence unit。TalkNet/MAAS 的来源核心尚未被目标任务占用。

项目旧 `LB-SCGP` 和 `source-scoped proposition graph` 不推翻这个结论，但严格限制 claim：

- LB-SCGP 已在 hateful-video detection 中占用了 direct-speaker endorsement、quotation/condemnation/reportage exception、speaker-source/stance binding 的**语义判断原则**。
- 当前候选不估计 endorsement、quotation 或 stance；active speaker 只回答“声音由哪张可见脸产生”，不能推出该人物认同所说内容，也不能区分主播报道、引用和谴责。
- 因而二者不是严格同构：旧链是 semantic accountability/stance，当前窄机制是 physical audiovisual assignment。反过来，当前候选也绝不能以“解决 endorsement/reportage”作为 novelty 或性能解释；新闻主播仍会被正确识别为 active speaker，即使其语用立场是报道。

## Gate 3：是否为 non-trivial task adaptation

**PASS，理由是排他 evidence unit，而不是 ASD feature 本身。**

若候选只把 TalkNet posterior、active-face embedding 或 speaking/not-speaking scalar 与 POWA feature concatenation，它就是简单组件拼接，应当 STOP。当前 brief 比这一步更强：

1. 一个 timestamped utterance 只能与唯一通过 producer 判定的 active face 形成 relation token；同秒其他人物和 full-frame scene不能进入该 utterance 的 relation interaction。
2. 无唯一可靠 visible speaker 时，utterance仍保留，但走显式 `offscreen_or_ambiguous` source state，而不是任意挑一张脸或删除 speech。
3. 最终 adapter 读的是 `(assigned face, utterance, offscreen state)` 的关系，而不是独立 ASD confidence；它针对“同秒 broadcast fusion 把发言者、被报道者、旁观者混成一个 evidence unit”的具体 hateful-localization failure。
4. 机制预期可被直接否定：如果真实 face assignment 不胜同视频内的 matched face permutation，physical source binding没有提供声称的信息；如果 adapter-off 与 core相当，新增路径没有进入 final score。

这属于把 ASD 的输出语义从“谁在说话”改造成 hateful temporal MIL 中的**关系可达性约束**，而不是只换 backbone 或增加一种 modality。它与 LELA/SafeLens/MM-HSD/MultiHateLoc 的整帧或整段 fusion有明确结构差异，也与 TANDEM 的 victim-group target identification不同。

## POWA 旁路与 load-bearing 判定

原 POWA visual/audio/text forward 和 score path仍保留，所以整个网络并不满足“所有 hate decisions 都必须经过 active-speaker binding”。原路径可以独立完成预测，relation adapter也可能在训练中被忽略。这是实质限制，但按 Rule 12，它属于需要 end-to-end test 与 matched control裁定的 shortcut/load-bearing 风险，不能仅凭“存在 baseline residual path”在实现前 STOP；否则任何 Rule 19 anchor-compatible additive method都会被同一理由否决。

因此本次 GO 只承认：**relation branch 内部的 utterance-face interaction是排他的，且该分支被加入同一 final-score computation。**以下更强说法当前不成立：

- “全模型彻底禁止 utterance 与 full-frame/non-active-face evidence共同影响 score”；
- “物理 active speaker 等价于 accountable/endorsing hate speaker”；
- “原 POWA path 已被结构上强制依赖 assignment”；
- “offscreen/null 本身已经被证明有独立贡献”。

正式结果中，adapter-off 是 load-bearing 的必要证据；若 core不胜 adapter-off，不能保留 novelty claim。即使性能提高，如果新增 adapter实际只读 utterance或ASD confidence而不依赖 assigned face identity，也应在 implemented-method novelty reassessment 中判为简单拼接。

## Permuted-face control 是否足以归因

**对“正确 physical face assignment 是否有用”基本足够；对全部更强机制 claim 不足。**

该 control 在同一视频内保留 TalkNet outputs、candidate face crops、utterance tokens、adapter容量、训练预算和 offscreen/single-face状态，只把可判定多脸 utterance cyclically分给另一张 candidate face。它保留了 producer可用性、face数量、局部时间、文本和参数量，同时破坏 speech-to-face identity，因此能直接检验 assigned-face identity，而不只是“多了一种 face feature”。双语料 within均胜该 control 是合理的最小机制门。

它不能单独证明：

- offscreen/null state有独立价值，因为 single-face/offscreen样本在两臂保持相同；
- 全模型的 cross-modal interaction都是排他的，因为原 POWA 仍是旁路；
- active speaker具备 endorsement/stance语义；
- 改善不是只集中在很小的 multi-face eligible subset。

因此，permuted-face control加 adapter-off足以支撑本次**最窄 relation-assignment claim**，但不能支撑上述更强说法。若实现后真正有效的部分变成 offscreen flag、generic face crop、额外参数或 unchanged utterance shortcut，必须重新做 implemented-method novelty review，不能沿用本次 GO。

## 最窄可主张贡献

> 将冻结 active-speaker assignment 适配为弱监督 hateful-video temporal localization 中的 source-bound relation unit：每个 utterance只允许与其物理上匹配的 visible-speaker face形成新增 relation evidence；无可靠 visible speaker时保留显式 null-source state；该 relation写入唯一 dense MIL score，并以同视频 face-identity permutation检验正确 assignment是否 load-bearing。

不能 claim TalkNet/MAAS、active speaker detection、speaker attribution、face tracking、POWA、MIL、offscreen detection、stance/endorsement reasoning本身新；也不能把“报道/引用 false positive 被解决”作为未经实验支持的既成结论。

## Final decision

**GO to implementation under the fixed brief.** 来源未被目标任务占用；排他 `(assigned active face, utterance)` relation unit相对 full-frame speech fusion和旧 semantic source/stance链具有独立、non-trivial 的物理 assignment机制。原 POWA旁路不构成 pre-run STOP，但把 claim限制在新增 relation branch，并把 adapter-off与 matched face permutation变成 load-bearing证据。实现若只是 ASD feature concatenation，或正式 test中正确 assignment不胜 permutation，本次 novelty结论不再成立。
