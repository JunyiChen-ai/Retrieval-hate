# 区间 inside–outside backbone：规则 4 独立 proposal review

日期：2026-10-01。审稿人：独立 Codex agent。一次性提案评审；未实现、未训练。依据：`experiments/20261001_inside_outside_backbone/README.md`、当前 `experiments/20260925_query_paradigm/model.py`、`RESEARCH_ITERATION_RULES.md` 第 4 条。旧 `REVIEW_RULE4_R4.md` 仅作背景，本次重新联网检索。within 按最新 AGENTS.md 与 pooled AP/ROC 并列主指标；未改写研究规则。

**判定：GO。** 在本次实际检索和原论文方法核对范围内，未找到把 inside–outside 递归表示用于 hateful video detection/localization 的直接先例；其余三种 STOP 条件也不成立。这是值得实现检验的迁移候选，不是已证明的创新贡献或世界首次。

## 1. 本次评审的准确对象

候选以固定二分时间树替换原 MACIL-SD 内容 encoder：逐秒缓存经模态投影形成叶子；有左右顺序的可学习组合器自下而上计算区间 inside；父 outside 与兄弟 inside 自上而下计算子区间 outside；局部内外表示形成逐秒、节点和视频先验，由既有问题树似然训练。VLM 答案仍只经原答案模型进入后验。

当前 `model.py` 的 `CrossModalLayer` 先作全时间跨模态 attention；`PriorNet.node_logits` 再以视频头相同 attention 权重，通过 log-sum-exp 合并得到区间加权平均。候选改变**内容编码计算图**，并非给已有节点池化换名字，也不是标签树的和积推断本身。左右有序非线性组合与排除本区间缓存行的 outside 均应真实进入后续实现。

## 2. 实际检索记录

工具：联网 search/open/find。以下为实际执行的主要查询；无相关结果仅表示未检索到，不证明不存在。

| 查询（原文） | 用途与结果 |
|---|---|
| `"inside-outside" "hateful" video`；`"inside-outside" "HateMM"`；`"inside-outside" "hateful video detection"`；`"hateful video" "inside–outside"` | 未找到来源机制的 hateful-video 直接应用 |
| `"Tree-LSTM" "hateful" video`；`"Tree-LSTM" "hateful video"`；`"Tree-LSTM" "HateMM"`；`"hate video" "Tree-LSTM"`；`"hateful video" "TreeLSTM"` | 未找到 hateful-video 直接应用；出现文本 hate 分类的 Tree-LSTM 标题，不能据此泛称 Tree-LSTM 从未用于 hate speech |
| `"recursive neural" "hateful video"`；`"hateful video" "recursive"`；`"hateful video" "binary tree"`；`"hateful video" "tree" neural` | 未找到本候选结构的直接应用；扩大检查到 MultiHateGNN |
| `"MultiHateLoc" paper` | 打开作者论文 v3 并核对 §3 |
| `RAMF hateful video reasoning augmented multimodal fusion paper` | 打开作者论文 v2 与官方仓库，核对 local/global 计算 |
| `"TANDEM" "hate" "video" localization arxiv` | 定位并打开作者论文，核对 §3 |
| `HateMM dataset multimodal hate video detection model LSTM paper` | 打开 ICWSM 原论文，核对模态模型与融合 |
| `"Multimodal Hate Detection Using Dual-Stream Graph Neural Networks" paper` | 打开作者论文，核对实例图与权重图 |
| `"inside-outside" "video" neural network localization`；`"inside-outside recursive" "video"` | 检查跨任务先例；出现视频 grounding 文献对 DIORA 的引用，不支持“首次用于视频”的大范围主张 |
| `"Hate Classifier for Social Media Platform Using Tree LSTM" itmconf`；`site.itm-conferences.org "Hate Classifier" "Tree"` | 出版者页面抓取失败，未将二手摘要作为判定依据 |

直接打开并阅读的来源论文还包括 IORNN、DIORA、Tree-LSTM，以及新近 CLARA 的作者论文。以下结论依据原论文/官方代码，不依赖检索中的自动摘要站点。

## 3. 最近先例与区别

| 原始来源 | 核对内容及对 novelty 的含义 |
|---|---|
| [Le & Zuidema，IORNN，EMNLP 2014](https://aclanthology.org/D14-1081.pdf)，§4.2 | 已明确用子节点计算 inside，用父 outside 和兄弟 inside 计算 outside，根 outside 可学习。本候选的这套递归是迁移来源，不能当作新发明 |
| [Drozdov et al.，DIORA，NAACL 2019](https://aclanthology.org/N19-1116.pdf)，§2 | chart 上组合所有候选树，outside 用于重构叶子；包含 Tree-LSTM/MLP 组合器。本候选用固定时间树及问题答案似然，无潜在句法树诱导或词重构 |
| [Tai et al.，Tree-LSTM，ACL 2015](https://aclanthology.org/P15-1150.pdf)，§3.2 | N-ary Tree-LSTM 已提供有序子节点的门控组合。左右有序的树组合本身不是新意 |
| [MultiHateLoc，作者论文 v3](https://arxiv.org/html/2512.10408v3)，§3 | 模态专属 Transformer 时间编码、动态跨模态融合及 top-K MIL；没有本候选的区间 inside/outside 递归 |
| [RAMF，作者论文 v2](https://arxiv.org/html/2512.02743v2)，§3.5–3.6；[官方仓库](https://github.com/Multimodal-Intelligence-Lab-MIL/RAMF) | local 分支为 Conv1D 与最大池化，global 分支为全局均值池化，门控融合后作语义跨注意力。全局量包含局部输入，不是排除本区间的互补 outside。不能把“融合局部与全局”作为候选的新意 |
| [TANDEM，作者论文](https://arxiv.org/html/2601.11178v1)，§3 | 30 秒块、视觉/音频语言模型生成时间戳、跨模态上下文与 RL；并非区间树递归内容 encoder |
| [HateMM，ICWSM 原论文](https://ojs.aaai.org/index.php/ICWSM/article/download/22209/21988/26272)，模型部分 | 视觉帧特征经顺序 LSTM，模态特征再融合分类；顺序 LSTM 不等同于本候选的有序二叉树及 outside |
| [MultiHateGNN，作者论文](https://arxiv.org/html/2509.13515v1)，§3 | 局部实例子图提取特征，全局权重图产生实例权重，随后加权聚合分类；不是多尺度二分树，也无父 outside + 兄弟 inside 递归。已证明结构化局部/全局建模不是空白方向 |
| [CLARA，作者论文](https://arxiv.org/html/2608.15905v1)，§3 | utterance-aligned clip、MoE 融合、local/global segment contrastive 与 rationale-gated Transformer；与本候选共享局部/上下文动机，但没有本候选的区间互补 outside 计算图 |
| [WINNER，CVPR 2023 原论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_WINNER_Weakly-Supervised_hIerarchical_decompositioN_and_aligNment_for_Spatio-tEmporal_Video_gRounding_CVPR_2023_paper.pdf)，§2.2、§3 | 视频 grounding：自下而上建立语言分解树，结构注意力引入视频 tube 特征，自上而下回传及层级对比学习对齐视频与语言结构。DIORA 作为参考文献 [2] 出现在 §2.2 的弱监督工作综述；该引用本身**不等于采用 DIORA**，论文未据此声明使用 DIORA encoder。它是视频多模态树表示的相关先例，不能当作本候选固定时间树与互补 outside 的直接应用 |

WINNER 的 Web open 抓取返回错误，随后从同一 CVF 原始 PDF 地址直接读取并用 `pdftotext` 核对上述段落及引用上下文，未落地额外项目文件。

## 4. 四项 STOP 判定

| 条件 | 判定 | 理由 |
|---|---|---|
| 来源已用于 hateful video detection/localization | 未触发 | 上述来源与 hate-video 方法的实际检索/方法核对未发现直接应用；结论受检索范围限制 |
| 纯 ensemble | 未触发 | 两模态分支属于一个共同训练的 prior network；没有组合多个独立模型的预测或新增 teacher 聚合 |
| 纯 calibration/后处理/平滑 | 未触发 | 改动发生在可训练内容 encoder 内，影响训练先验和查询推断输入；不是最终分数的事后变换 |
| 纯工程技巧 | 未触发 | 改变多尺度表示的递归结构和上下文可访问范围，构成完整可检验的表征假设；不限于换特征、调超参或训练配置 |

## 5. 非阻断限制及建议主张

- **主张范围**：将区间 inside–outside 表示迁移到 hateful video localization，使问题树同时组织内容先验与带噪问题答案的学习。不能主张发明 inside–outside、Tree-LSTM、local/global 融合或首次结构化 hateful-video 表征。
- **排除范围**：outside 只能保证其计算图不读取本区间的 1-fps 缓存行。已有 I3D 重叠感受野、BERT K30 上游编码可能跨越区间，不能写成原始媒体级隔离或无信息泄漏。outside 也不意味着无害背景。
- **证据边界**：第 4 版无骨干对照同时去掉节点势，不足以把 within 差异归因为 attention，更不是第 5 版新骨干的有效性证据。三项主指标的实际结果决定后续结论。
- **区分贡献**：原骨干、新骨干、去 outside、有序组合换池化以及相近容量逐秒 encoder 的比较可区分新结构与已有 pooling。保持同一答案缓存、训练目标、预算和停止设置；这不是新增晋级门。
- **成本**：提案复用全部缓存，新增 VLM 调用和特征抽取均为 0；新视频仍负担原提问策略成本。树组合约 O(T h²) 不代表 GPU 实测更快；README 的 GPU 时间仅是旧实现参考，待首个完整 trial 确认。

GO 只表示规则 4 不阻断此提案。本轮未启动训练，不把性能、效率或 novelty 的实验贡献写成已成立。
