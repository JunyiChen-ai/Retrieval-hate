# Policy-patch excess：独立 novelty / source / identifiability review

截至 2026-08-31。审查对象是本目录 `README.md` 中尚未实现的候选。本轮没有抽取特征、训练模型或生成 prediction。

## 结论

**Verdict：STOP。Novelty：3.8/10。不要按当前定义运行 premise。**

三项硬门：

| 硬门 | 结论 | 理由 |
|---|---|---|
| 允许跨任务 adaptation | PASS | DenseCLIP、MaskCLIP、RegionCLIP 等 dense/region vision-language 方法可以适配到 hate localization。 |
| 来源核心尚未用于 hateful video detection/localization | PASS，窄范围内未发现 exact 先例 | 检索未发现“冻结 dense CLIP policy patch logits，加同帧 spatial excess，再作为 weak hateful-video temporal MIL 输入”的直接目标任务方法。 |
| adaptation 必须 non-trivial、任务特定且可证伪 | **FAIL** | 当前方法是固定 policy prompts、off-the-shelf patch/text cosine、手写 top-q-minus-trimmed-mean scalar，再接普通 temporal MIL。它没有学习 spatial-policy relation，也没有 region supervision；标准 CLIP patch-token接口本身又没有可靠 text alignment保证。即使 premise有预测力，也只能证明一个手工 zero-shot feature有效，不能证明新的任务机制。 |

第二门的 PASS 不能挽救第三门。局部/region hate evidence 的宽故事已经进入 hateful memes，dense CLIP 又已解决 generic region-text prediction；当前差异只是一条手写聚合公式与一组 prompts。

## 检索范围

检索组合包括：

- `hateful video detection/localization + patch-level / region-level / dense CLIP / visual grounding`；
- `hateful meme + object region / patch / CLIP / visual grounding`；
- `DenseCLIP / MaskCLIP / RegionCLIP + patch token text alignment`；
- 2025–2026 hateful-video 方法 `MultiHateLoc / CLARA / LELA / MM-HSD / SafeLens / MARS / Cross-Modal Transfer`。

只使用论文、会议/期刊页面与官方代码作为占位证据。没有检索到直接先例只支持窄结论，不证明所有未索引工作均不存在。

## Dense / region CLIP 已占用的核心

1. [CLIP, ICML 2021](https://proceedings.mlr.press/v139/radford21a.html) 的训练目标是整图与整段文本的 contrastive matching。官方 [OpenAI CLIP implementation](https://github.com/openai/CLIP/blob/main/clip/model.py) 的标准 ViT image representation读出 class token并经 projection；原始训练没有对每个 patch 与文本分别施加监督。

2. [RegionCLIP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhong_RegionCLIP_Region-Based_Language-Image_Pretraining_CVPR_2022_paper.html) 明确指出，把全局 CLIP 直接用于 image regions 会因整图训练与 region-text 任务之间的 domain shift而表现不佳；它通过 region-text pseudo pairs专门预训练 region representations。该论文已经占用 region-level language-image alignment，也直接否定“标准 global CLIP自然保证 region-text alignment”的隐含前提。

3. [DenseCLIP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Rao_DenseCLIP_Language-Guided_Dense_Prediction_With_Context-Aware_Prompting_CVPR_2022_paper.pdf) 把 image-text matching改成 pixel-text score maps，并在 dense prediction datasets 上 fine-tune。它不是简单读取冻结标准 patch tokens。

4. [MaskCLIP, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880687.pdf) 提供 annotation-free dense prediction，但其关键不是“取 ViT 输出的 14×14 tokens然后乘原 projection”。它重组最后 attention，使用 value features并移除/改变 query-key readout；论文还报告 naive dense feature baseline显著较差。MaskCLIP 的证据只能支持其指定 readout contract，不能自动支持 README 所写的普通 projected patch tokens。

因此 `DenseCLIP/MaskCLIP/RegionCLIP` 不能被统称为当前 feature extractor 的来源验证。若使用标准 projected patch tokens，必须把它当未验证的 heuristic；若改为 MaskCLIP 或 RegionCLIP，则方法更接近已有 zero-shot dense classifier，novelty进一步只剩 spatial excess scalar。

## Hate-content 邻域已有的 region evidence

### Hateful memes

- [The Hateful Memes Challenge, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/1b84c4cee2b8b3d823b30e2d604b1878-Abstract.html) 的多模态 baselines 已使用 Faster R-CNN image-region features。region/object representation进入 hate classification并不是新方向。
- [Hateful Memes Detection via Complementary Visual and Linguistic Networks](https://arxiv.org/abs/2012.04977) 同时使用 contextual-level 与 sensitive object-level information，并用 object detector产生 RoIs。
- [Hate-CLIPper, 2022](https://aclanthology.org/2022.nlp4pi-1.20/) 已把 CLIP image/text representations用于 hateful meme classification，虽不是 patch-level scoring。
- [TRACE, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/41220) 使用 RAM++、GroundingDINO 与 grounded caption augmentation处理 hateful memes，并明确把 visual grounding用于减少 benign confounders。它不是 patch-policy excess，但已占用“grounded local visual evidence改善 hate detection”的宽任务故事。

### Hateful videos

- [Cross-Modal Transfer from Memes to Videos, WWW 2025](https://arxiv.org/abs/2501.15438) 已把 re-annotated hateful memes迁移到 HateMM/MultiHateClip video detection。它不做 dense patches，但缩短了 hateful meme region evidence 与 hateful video之间的跨域距离。
- [MM-HSD, ACM MM 2025](https://publications.idiap.ch/publications/show/5688)、[MultiHateLoc](https://arxiv.org/abs/2512.10408)、[CLARA](https://arxiv.org/abs/2608.15905)、[LELA](https://arxiv.org/abs/2602.09637)、[SafeLens, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/42390) 均使用 visual frames或视觉语义，但本次检索未发现它们使用 fixed patch-policy spatial excess。

结论是：exact target application未被占位，但 global-to-local visual evidence、region grounding、CLIP prompting与 meme-to-video transfer都已有直接先例。不能以“首次发现小区域 hateful evidence”作为 claim。

## 标准 CLIP patch tokens 不具有所需的可靠接口

README 指定“冻结 CLIP ViT-B/16，读取 14×14 projected patch tokens”。这里至少有四个问题：

1. **训练监督不匹配。** CLIP只监督最终整图 embedding 与文本匹配，没有 patch-text loss。把最后一层每个 spatial token套用 class-token projection，不等于得到经过验证的 dense CLIP classifier。
2. **source implementation不忠实。** MaskCLIP使用最后 attention的 value-path dense readout并专门处理 query/key；RegionCLIP重新做 region-text pretraining；DenseCLIP在 dense task上训练。当前三者都没有被忠实采用。
3. **patch不严格局部。** ViT transformer后的 patch token已经通过 self-attention读取全图。14×14网格是 token位置，不意味着每个 token只描述一个 16×16独立局部区域；“同帧多数 patch代表 scene null、少数 patch代表 carrier”的解释没有结构保证。
4. **复杂 policy sentence不等于 visual concept。** CLIP dense transfer最可靠的是具象 object/category phrase。诸如贬损 protected group、incitement、讽刺或 predicate-target关系常需要 OCR、人物身份与全局组合；一个 patch与整句 policy clause的 cosine不能识别这种关系。

因此在抽取前，候选尚未满足 source contract。用 premise test发现 raw patch token恰好相关，最多是经验 feature discovery，不会反向证明 patch-text semantic alignment。

## `u_t` 的可识别性

定义

`u_t = topq_logmeanexp_{p,c}(a_{t,p,c}) - trimmed_mean_p(max_c a_{t,p,c})`。

### 它能保证什么

若所有 patch-clause logits都加同一个常数，两个项都会加该常数，`u_t` 可消掉理想化的 additive frame-wide offset。这是明确的代数性质。

### 它不能保证什么

1. **不能把 topic 与 carrier分开。** 任意局部显著物体、字幕、logo、脸、边框或 compression artifact都可产生高 patch-text similarity。减去全帧 trimmed mean只说明某些 cells高于同帧其余 cells，不说明它们是 hate witness。
2. **会删除 spatially broad evidence。** 若 hateful scene或大字字幕覆盖大部分画面，第一项与背景项会一起升高，excess反而接近零。
3. **没有 policy composition。** 对 `(p,c)` 全局 top-q，再对 clauses取 `max`，会在输入 localizer前丢掉 clause identity、protected target、predicate以及 clause之间关系。最终只有一个 scalar；它表达的是“某个 patch像某个 prompt”，不是 policy satisfaction。
4. **multiple-comparison effect。** clause数越多，top-q与 `max_c` 越容易出现高值。固定 clause数可让一次运行可复现，但不能赋予 score 概率语义。
5. **scale/temperature敏感。** additive shift相消不代表 cosine scale、CLIP logit temperature、top-q fraction或 trim fraction不影响排序。`q` 与 trim在 source literature中没有针对 hate carrier的固定依据。
6. **spatial samples相关。** 相邻 ViT tokens共享大范围 receptive field；trimmed patch distribution不是独立 spatial null，也没有 false-positive控制。

所以 `spatial excess` 是一个手工 local-saliency contrast，不是 identifiable localized-hate variable。

## Fixed policy clauses 是方向性观测还是 prompt engineering

固定 clauses确实比完全由 bag label自举的 visual head多了一个外部语义方向。但当前方法把它们仅作为 zero-shot classifier prompts，且在聚合前没有 learned/structured policy binding。因此其方法学类别仍是 prompt-engineered feature extraction。

要称为 task mechanism，至少需要证明：

- 高分 patch对应 clause所描述的视觉实体/行为，而不是任意 salient region；
- hate clauses相对 visualness、具体性、词频和长度匹配的 non-hate clauses有额外信息；
- spatial contrast而非仅 prompt ensemble 或 patch max 是 load-bearing；
- scalar进入 temporal localizer后实质改变 frame ranking。

现有 frame-level GT没有 spatial region labels，无法验证第一条。它只能判断某一帧是否 hateful，不能判断选中的 patch是否正确，也不能区分真正 local carrier与同帧共现 confounder。

## Premise controls 不足

### 1. `spatially_permuted_clause` 可能是严格不变或语义不清

若只是重排 clause indices，`topq_{p,c}` 与 `max_c` 都对 clause permutation不变，`u_t` 严格不变。这个 control无法运行出不同结果。

若是对每个 clause独立打乱 patch positions：第一项仍因保留全部 `(p,c)` multiset而不变，只有第二项可能因不同 clauses的空间共定位被破坏。这测试的是 clause-score co-location，不是 policy-token semantic correspondence。

若想测试 prompt semantics，必须替换 text embeddings为预先匹配的 decoy clauses并重新计算 similarities；不能把 index permutation称为 correspondence test。

### 2. `generic_object_prompts` 匹配条件不够

仅匹配数量与字符串长度不能匹配 CLIP visual detectability、concreteness、词频、图像先验或 prompt-template sensitivity。`person holding a weapon` 与抽象 generic clause的差异可能来自 visualness而非 hate policy。至少要有：

- equally concrete、同语法的 benign action/object clauses；
- policy nouns保留但 polarity/关系改变的 hard decoys；
- 与常见视频 topic匹配的 political/news/logo clauses。

这些 controls仍只能证明 prompt direction有预测价值，不能把手写 excess提升为新训练机制。

### 3. global control 与 core语义不一致

README 的 `global_policy` 是 policy-vs-benign score，但 core公式只使用 policy logits与 spatial trimmed baseline，未出现 benign prototype。于是 global-vs-core同时改变空间粒度和文本对比定义，不是单变量 control。

必须分别有：

- global policy-only；
- global policy-vs-benign；
- patch policy-only；
- patch policy-vs-benign；
- 每一版本的 raw与spatial-excess readout。

但补齐这些只是严谨 feature ablation，不会改变 STOP verdict。

### 4. test premise能支持的结论很窄

按项目规则，可以在完整 test cohort做 developmental error analysis并让结果 inform method development；必须明确不是未揭盲 confirmatory evidence。若 core胜 global，它只支持“这个冻结 scalar对 test frame ranking更有用”。它不支持：

- selected patch是正确 hate region；
- spatial baseline去除了 topic而不是其他信号；
- policy clause发生了组合推理；
- 把 scalar接入 trainable MIL 后仍会被使用。

正式模型还可把 `u_t` 权重学成零。若曾允许实现，至少需要同 checkpoint把 `u_t` 置零/时间置换的干预；当前 formal arms只有分别训练的 feature controls，不能证明 core scalar load-bearing。

## 与项目既有 policy/primitive 负结果

项目已有证据进一步削弱 premise，而不是证明它必然失败：

- dense typed primitive teacher在完整 HMM qualification未过固定门；
- policy recurrence中 shuffled primitive不差，HCS加 typed primitive还下降；
- policy-complete proposal MIL因固定 role/AST scalar与 P-MIL拼接、HCS unary退化而在 novelty阶段停止；
- policy-AST intervention的部分方向性由固定结构自动满足，不能证明 semantic mechanism。

本候选换成 spatial patch后可能获得新的 visual observation，但仍把固定 policy similarities先压成一个 scalar再接现有 localizer，与上述“primitive可能不 load-bearing”的失败模式相同。必须有新的 region-level evidence或可识别训练约束才能越过，而不是再换一个 handcrafted reduction。

## 是否存在可批准的最小修复

没有。以下调整都不足以把当前 proposal变成 GO：

- 把 raw CLIP patch tokens换成 MaskCLIP value features：修复 source contract，但剩下的是 faithful MaskCLIP zero-shot outputs、手工 spatial contrast与普通 MIL。
- 增加更多 prompts、换 CLIP版本、扫描 `q/trim`：只是扩大 prompt/readout search，并增加 developmental test overfitting。
- 添加 generic clause controls：改进归因，但不形成 non-trivial task adaptation。
- 用 RegionCLIP/DenseCLIP：仍是将已有 dense classifier直接作为额外 scalar feature。

若以后重提，必须改变核心而非 feature extractor。例如需要同语料 train labels可学习、又能由区域级 intervention或独立 spatial annotation证伪的 policy relation；并且它必须保留 clause/region结构到 temporal objective，而不是先压成一个固定 scalar。那将是新候选，需要重新查新。

## 最终裁定

检索没有发现 exact policy-patch excess已用于 hateful video detection/localization，故来源占用门本身通过。但 DenseCLIP、MaskCLIP、RegionCLIP 已占 dense/region text alignment；hateful meme工作已使用 region/object evidence，TRACE更已用 visual grounding处理 benign confounders；meme-to-video transfer也已有先例。

当前 proposal未忠实采用已有 dense CLIP readout，标准 patch-text alignment前提不成立；`u_t`只保证消除理想 additive frame offset，不能识别 localized hate；premise controls既有 permutation不变性问题，也没有 spatial GT。即使 test premise数值通过，方法仍是 **zero-shot patch prompting + handcrafted contrast + ordinary MIL**。

**STOP，3.8/10。不要抽取或运行该 premise。**
