# MLLM_USAGE_LANDSCAPE — 领域内 LLM/MLLM 用法对照报告

_日期:2026-07-02。范围:hateful / harmful video detection 领域内所有"用 LLM/MLLM 做检测"的竞品方法的精确机制,外加 meme 侧 caption 线(引用定位)与上游 RA-HMD。机制描述全部从论文原文 / 官方代码核实(arXiv HTML、ACL Anthology PDF、GitHub 源码),非印象。_

_目的:判定我们计划中 MLLM 为检索记忆库服务的三个候选角色有没有被占:_
- **角色 1**:MLLM 生成结构化描述条目(攻击目标 / 攻击机制 / 载体模态 / 显隐性 schema)作为**检索键**,可解释可审计;
- **角色 2**:训练时给弱监督片段当**裁决者**(标签共识分歧时才调用);
- **角色 3**:推理时 **kNN 不确定才唤醒** MLLM、对检索到的记忆条目做 **in-context 推理**。

**一句话结论先行:视频领域现有全部 reasoning-VLM 方法都是 always-on、无门控、无检索/记忆、生成文本即用即弃;MoRE 有记忆库但存不可读特征、零 LLM 参与;RA-HMD 三个判决头之间无任何动态门控。角色 1 OPEN,角色 2 PARTIAL(视频级标注被占、片段级分歧裁决没被占),角色 3 领域内 OPEN(工业界有相似度路由级联需引用划界)。**

---

## 第一部分:视频侧 reasoning-VLM 线(逐一精扒)

### 1. MARS(arXiv:2601.15115,ICASSP 2026)— training-free 多阶段对抗推理

**机制流水线**:每个视频均匀采样 16 帧 + Whisper Medium 转录,喂给**同一个 MLLM 做 4 次顺序调用**:① 客观描述(中性事实描述 D_obj);② "假设含恨"推理(证据 E_hate + 推理 R_hate + 置信度);③ "假设不含恨"对称推理;④ 元分析综合(meta-prompt 权衡双方,**直接输出最终判决对象** {y_pred, conf, key factors, rationale})。backbone 测了 Qwen2.5-VL-32B / Llama4-17B / GPT5-mini / Gemini2.5-Flash,无任何下游可训练分类器。HateMM 最好 Acc 78.4 / M-F1 77.8(LLaMA4);MHC-ZH Acc 75.9 / M-F1 71.3(Qwen2.5)。

**关键定性**:
- **always-on**:每样本必走完整 4 阶段。各阶段虽产生 confidence,但原文明确 "The confidence score conf_final is intended solely for interpretability rather than thresholding" —— **置信度不做任何阈值/路由**。
- 输出含固定字段(标签 + 标量置信 + key factors),但**无语义 schema**(无攻击目标/机制/显隐性字段);证据与推理为自由文本。
- 生成文本**一次性用完即弃**,不存不复用,无 retrieval / memory / kNN。

**与三角色重叠**:角色 1 ✗(自由文本、不入库);角色 2 ✗;角色 3 ✗(always-on、MLLM 就是主判决者而非被唤醒的辅助)。

### 2. HVGuard(EMNLP 2025 Main,pp.8993–9006)— GPT-4o 三步 CoT rationale → MoE 融合

**机制流水线**:视频 → FunASR 转录 + 语音情绪标签、采样 32 帧、标题;→ **GPT-4o 三次串行调用**(角色设定 "content moderation specialist"):① 视觉理解(描述帧内容、忽略字幕);② 文本理解(标题 + 转录,显式提示注意谐音梗/双关);③ 融合理解(合并全部信号,"only answer implicit meanings and whether this video expresses hateful content")→ 自由文本 rationale。rationale 经文本编码器变成 embedding,与 XLM 文本 / ViT 视觉 / Wav2Vec 音频 embedding 拼接,过 **MoE(8 个 FFN expert + gating)** 用 CE 训练分类。**最终判决权在 MoE**,GPT-4o 的自带判断只是文本特征的一部分。HateMM 0.8563 / 0.8597;MHC-EN 二分类 0.8539 / 0.7714;MHC-ZH 二分类 0.8603 / 0.8219。消融:去 CoT 后 MHC-EN 三分类 M-F1 0.6646→0.4715;换 Qwen-VL 最好仅 0.6276("MLLM 能力是最大因素")。

**关键定性**:
- **always-on**,每样本必调 3 次 GPT-4o,无门控/级联/abstention。
- rationale 为**自由文本**,无结构化字段。
- 唯一"存储"是工程性 **embedding cache**(§3.5,预计算一次、训练中复用)——**不是检索键、不进记忆库**。
- 无 retrieval / kNN;标签全部人工,GPT-4o 不做标注。

**与三角色重叠**:角色 1 ✗(自由文本 + 工程缓存 ≠ 可检索可审计条目);角色 2 ✗;角色 3 ✗。

### 3. IARE(arXiv:2606.11953,SIGIR 2026)— SFT + DPO 的 rationale 生成式判决

**机制流水线**:先造带解释标注的 Ex-HateMM / Ex-ImpliHateVid(caption + 分模态 harmful elements + gold rationale;**人工主导两名硕士级标注 + 博士级合并,GPT-4o-mini 仅为提效辅助工具**)。训练 Qwen2.5-VL-7B(LoRA)两阶段:**Stage 1 SFT**——输入帧 + ASR/OCR + caption + harmful elements,监督 gold rationale + 标签,输出模板 `Prediction / Rationale`;**Stage 2 DPO**——正例 gold rationale,**负例由 Qwen2.5-VL-72B 为故意错配的标签编造解释**(显式捕获 spurious correlation),再人工过滤。推理时双 MLLM 流水线:GPT-4o-mini 生成 caption + harmful elements(替代 oracle 标注),微调后的 Qwen2.5-VL-7B **端到端直接输出判决 + rationale**。Ex-HateMM 90.37 / 90.14;Ex-ImpliHateVid 91.75 / 91.75。

**关键定性**:
- **always-on**(每样本 GPT-4o-mini + Qwen2.5-VL 各至少一次),无门控。
- 输出为自由文本 rationale(仅 Prediction/Rationale 两段);**输入侧** harmful elements 有固定类别体系(文本按侮辱词表;视觉四类:violence / hate 符号 / pornography / illegal activities)——**半结构化但只是推理中间物,不入库、不做检索键**。
- 无 retrieval / memory / kNN。
- **弱监督成分(注意)**:(a) GPT-4o-mini 辅助数据集标注(人验);(b) Qwen2.5-VL-72B 生成 DPO 负例(人过滤);(c) 测试期辅助信号由 GPT-4o-mini 自动生成。

**与三角色重叠**:角色 1 PARTIAL 参照(有"分模态 harmful elements"半结构化字段,但没有 target/机制/显隐性 schema,且不持久化不检索);角色 2 PARTIAL 参照(LLM 辅助标注 + LLM 造负例,但都是数据集构建期、人工兜底、非片段级裁决);角色 3 ✗。

### 4. TANDEM(arXiv:2601.11178)— tandem RL 结构化输出(唯一有输出 schema 的视频方法)

**机制流水线**:Qwen2.5-VL-7B(视觉)+ Qwen2-Audio-7B(音频)各挂 LoRA。① SFT 冷启动:**Qwen3-Omni-30B-A3B-Thinking 对 100 条视频生成 silver 结构化标注**,按 gold 分类标签严格过滤后使用;② tandem RL(GRPO/GSPO):每 10 步交替更新一方、冻结另一方,冻结方经 SCCR(zero-shot 推理产生结构化上下文)注入训练方 prompt。推理时视频切 30s chunk,两模型对每 chunk 独立输出 **XML 结构化预测**,前一 chunk 的 `<summary>` 传给下一 chunk。Reward = 分类正确 + 时间戳 IoU + target 集合 F1 + summary 长度 + XML 格式合规 五项加权。HateMM 二分类 Acc 0.78 / M-F1 0.78 / **Target F1 0.73**(卖点),时间戳 IoU 仅 0.18;MHC 三分类 M-F1 0.32。

**关键定性**:
- **有输出 schema**:`<reasoning> <classification> <timestamps> <targets>(从 prompt 给定的固定 taxonomy 选)<summary>` 五个 XML 字段。**但没有攻击机制 / 载体模态 / 显隐性字段**,且结构化输出**不进任何库、不做检索键**——只有两处瞬态复用(SCCR 跨模态注入、summary 跨 chunk 传递),不落盘。
- **always-on**(VL/AL 对每 chunk 必跑),无门控。
- 无 retrieval / memory / kNN。
- **弱监督成分**:Qwen3-Omni 生成 SFT silver 标注(gold 过滤)——MLLM 当标注器的领域内实锤之一,但是视频级冷启动、非片段级裁决。

**与三角色重叠**:角色 1 PARTIAL 参照(结构化输出最接近的先例,但字段不同且无检索/记忆链条——**写论文时必须显式对比**);角色 2 PARTIAL 参照(silver 标注);角色 3 ✗。

### 5. RAMF(arXiv:2512.02743,TMLR)— 冻结 VLM 三视角推理文本作"第四模态"

**机制流水线**:离线对每样本(16 帧 + Whisper 转录)调用**冻结 Qwen2.5-VL-32B** 生成三段文本:T_O 客观描述、T_H 假设含恨推理、T_N 假设不含恨推理(prompt 与 MARS 同款,同一实验室)。训练侧:原始三模态编码(BERT/HateXplain、MFCC+CLAP、ViT+CLIP),三段推理文本同样编码后分两步融合(LGCF + SCA 得 Y₁,再与 T_H/T_N 过 SCA 得 Y₂),送**可训练分类头**(CE)。VLM 不做判决,只产生额外文本模态。HateMM Acc 84.3 / M-F1 83.7(超 MoRE);MHC-ZH 72.4 / 69.3;MHC-EN 68.5 / 64.1。

**关键定性**:Algorithm 1 逐样本无条件生成 → **always-on**,无门控;**纯自由文本** rationale,无 schema;文本按样本离线生成一次、编码即融合,**不跨样本复用、不做检索键**;无 retrieval / memory / kNN。

**与三角色重叠**:全部 ✗。RAMF 是"MLLM 文本当特征"路线在视频域的最强代表(HateMM 当前最强训练法之一),我们引用时把它归入"生成即弃的 rationale-as-feature"一类。

### 6. LELA(arXiv:2602.09637)— training-free 帧级定位,LLM 逐帧打分

**机制流水线**:专用 captioner 把视频拆成 5 路模态文本(BLIP-2 图像 caption、EasyOCR、Whisper 语音、LP-Music-Caps 音乐、PDVC 视频上下文);composition matching:每帧把 speech caption 与各模态 caption 拼接后用 LLM 摘要;然后对**每帧 × 每模态**做三阶段 prompting(角色设定 + 仇恨定义 → 自由文本 rationale → 输出 0–1 仇恨分数),帧级分数取模态 max,阈值 τ=0.5 得逐帧定位。主用 GPT-4o-mini,LLM 不看像素(视觉均先文本化)。HateMM 帧级 PR-AUC 72.64;MHC 72.27。低于全监督(MM-HSD 87.8 Acc)。

**关键定性**:**always-on 且调用量极大**(每帧每模态多次调用),无门控、无成本讨论;rationale 自由文本 + 结构化标量分数,无语义 schema;caption/rationale 逐帧即弃,无存储无检索。

**与三角色重叠**:全部 ✗。

---

## 第二部分:MoRE(WWW 2025)— 检索记忆库的形态确认

**机制流水线**(笔记完整 + 本次复核无更新):Memory Bank 由 **train+val 的 (audio, text, vision) 特征三元组**构成(Whisper→BERT 音频、BERT 标题+描述文本、ViT 关键帧视觉);**冻结的 weighted-cosine 视频到视频检索器**取 top-K=50 hateful + top-L=50 non-hateful 双极邻居;邻居特征经 BHAN(双极交叉注意,"inspired by contrastive learning" 但实现是 attention)注入各模态 expert,MSR 软路由融合;**全部监督为 BCE**,无 InfoNCE,检索器不学习。HateMM M-F1 0.8235;MHClip-Y 0.7519;MHClip-B 0.7475;去检索器 HateMM 掉到 0.7355。

**两点确认(任务问的)**:
1. **记忆条目形态 = 原始模态特征向量三元组**(embedding),人类不可读、不可编辑、不可审计——不是描述、不是结构化条目。
2. **零 LLM/MLLM 参与**:管线内只有 Whisper(ASR)+ BERT/ViT 编码器,无任何生成式 LLM;LLaVA-OneVision / Qwen2-VL 只出现在 baseline 对比里。

**与三角色重叠**:角色 1 的"记忆库"半边被占(有库),但"条目 = MLLM 生成的可读结构化描述"半边完全没占——这正是我们与 MoRE 的可审计性对比点。角色 2、3 ✗(无 LLM、推理时全量检索无门控)。

---

## 第三部分:meme 侧 caption/description 线(引用定位,相关领域)

_结论先行:"LLM 生成描述/caption/rationale → 再分类"在 meme 域已经非常成熟,以下表述已被占,我们必须避开把它们当卖点:_

| 已占表述 | 占位论文 | 机制一句话 |
|---|---|---|
| **probing/VQA 式生成 caption → 文本分类器** | **Pro-Cap**(Cao et al., ACM MM 2023, arXiv:2308.08088) | 用一组 hate 相关 probing 问题问 frozen BLIP-2,答案拼成 caption 喂 PromptHate/BERT。**probing 维度已覆盖:通用内容、race、gender、religion、nationality、disability、animal** —— 即"按目标维度提问生成描述"本身在 meme 已被做过 |
| **caption + prompt 预训练 LM + demonstration** | **PromptHate**(Cao et al., EMNLP 2022) | ClipCap caption + Web 实体 + FairFace 人口学字段 → RoBERTa MLM 预测 good/bad;demonstration 固定非检索 |
| **LLM rationale 蒸馏给轻量学生** | **Mr.Harm**(Lin et al., EMNLP 2023 Findings) | ChatGPT 拿 gold 标签做 abductive reasoning 生成 rationale;Flan-T5 学生两阶段(先学 rationale 再学标签);推理时只出标签 |
| **多 LLM 正反辩论 + 小模型 judge** | **ExplainHM**(Lin et al., WWW 2024) | harmless/harmful 两方各生成 rationale,frozen LLM 排序,Flan-T5 judge 微调判类;同立场 rationale 兼作解释 |
| **LMM 生成解读文本再独立编码融合** | **IntMeme**(ICWSM 2025, arXiv:2502.11073) | LMM 解读文本与 meme 各自编码后融合分类 |
| **prompt 级检索标注样例 + LMM in-context** | **LMM Agents 低资源 meme**(EMNLP 2024 Findings, arXiv:2411.05383) | LMM agent 检索相似**带标注 meme** 进 prompt 做 few-shot 判类 + 自反思修正。**meme 线里唯一带检索成分的**——检索结果进 LLM 上下文而非 embedding 判决空间。**与我们角色 3 的"对检索条目做 in-context 推理"表面最像,必引必划界**(它 always-on、无不确定性门控、检索的是原始样本非结构化条目、meme 域) |
| **VLM 解读 + LLM 生成干预文本** | **MemeGuard**(Jha et al., ACL 2024) | 做 moderation intervention 生成,非检测 |

**meme 线未占的空位**(= RA-HMD 与我们的位置):在 **LMM embedding 空间**做检索引导对比学习 + kNN 记忆判决;以及以上一切在**视频域均无对应物**。

**写作红线**:凡是我们的 MLLM 角色 1,不能表述成 "generate caption/description then classify"(Pro-Cap 占)、不能是 "LLM rationale as feature/distillation"(HVGuard/RAMF/Mr.Harm 占)、不能是 "LLM debate/judge"(ExplainHM/MARS 占)。必须锚定在:**固定语义 schema 的条目化生成 + 持久化入检索记忆 + 作为检索键与审计证据**,这三件事的合取没有任何 meme 或 video 工作做过。

---

## 第四部分:RA-HMD / LMM-RGCL(上游,arXiv:2502.13061v4,EMNLP 2025 Oral)

**LMM 的精确用法**(原文 + 本地官方代码 `RA-HMD/LLAMA-FACTORY-Ver202512/src/llamafactory/custom/infer_utils.py` 交叉验证):

- **双重身份,判决不靠生成**:LMM(Qwen2-VL / LLaVA)既是 embedding encoder 也保留语言生成,框架含三个判决头:**LMH**(原 LM head)、**LRC**(logistic 回归分类头)、**RKC**(retrieval kNN classifier)。
- **Stage 1**:LoRA 微调 LMM,联合损失 L_LM + L_LR(LM loss 学生成 "hateful"/"benign" token,LR loss 训分类头)。
- **Stage 2**:**冻结 LMM,只训 MLP 投影 + LRC**,损失 L_CL + L_LR,其中 L_CL 即 RGCL 式检索引导对比(FAISS 挖 pseudo-gold positives 同标签近邻 + hard negatives 异标签近邻)。
- **embedding 取处**:最后一层、序列最后一个 token(`outputs.hidden_states[-1][:, -1, :]`),经 2 层 MLP 投影到 1024 维。
- **推理判决 —— 无任何门控(重点)**:in-domain 用 LRC,cross-dataset / 低资源用 RKC(top-K=20 相似度加权多数投票),**用哪个头是按实验协议人为指定的,全文不存在置信度阈值、LM-kNN 一致性检查、动态路由或 fallback**。→ 我们的角色 3(kNN 不确定才唤醒 MLLM)**没有被上游占用**;同时引用 RA-HMD 时不可声称它有动态切换。
- **rationale 生成**:LMM 保留生成能力,但生成文本**完全不进分类管线**,仅 §4.9 做解释质量评测(LLM-as-Judge 5.4 分)。
- **retrieval/memory**:FAISS 库存**投影后的 embedding + 标签**(非原始样本、非可读条目);cross-dataset 协议 = 冻结权重、把目标域训练集编码入库、纯 RKC 投票零梯度 —— **"换库即适配"的 training-free 更新机制事实存在,但论文没有把 updatable memory 当显式卖点写进摘要/引言**。我们在视频域做 updatable cross-dataset kNN memory 时需引用并区分(视频域、跨语言库、增量/持续更新、可审计条目)。

---

## 第五部分:2024–2026 四点扫描(每点单独结论)

_扫描范围:hateful/harmful/toxic/offensive VIDEO detection 及内容审核系统;约 15 篇关键论文逐篇核实。_

### 扫描点 1 — 结构化 schema 标注生成(作为检索键):**OPEN**

没有任何 hateful/harmful video 工作让 MLLM 生成固定字段结构化条目并**存入检索库作检索键/可复用证据**。最接近的先例(全部"结构化但不入库"):
- **SafeLens**(AAAI-26 Demo,SUTD/Roy Lee 组):微调 Llama3-8B 对每 segment 输出结构化 JSON(label、置信度、一句话理由、harm categories、**模态归因**),宣称 "auditable via reproducible JSON logs" —— 但 JSON 是**一次性推理产物/审计日志**,不进检索库、无复用。**话语上与我们角色 1 最像,必须显式对比"输出即弃的审计日志 vs 持久化可检索记忆"**。
- **TANDEM**:XML 输出 schema(classification/timestamps/targets/summary),无库无检索。
- **IARE**:分模态 harmful elements(半结构化),是标注/推理中间物。
- 相邻非视频:LLM-based Semantic Augmentation(arXiv:2504.15548,text/meme,Explanation+Triggers 字段拼进分类器输入,不持久化)。

### 扫描点 2 — MLLM 当弱监督标注器/裁决者:**PARTIAL(最拥挤,措辞要小心)**

视频级"MLLM 当标注器"已被占;**片段级弱标签 + 分歧触发裁决 + 训练学术 hateful-video 模型**的组合没人做:
- **IPS(In-Prompt Process Supervision)**(TikTok, arXiv:2412.15251, v3 2026-05)——最接近占用:"replacing human-annotated ancillary labels with MLLM-generated ones results in only marginal performance degradation",MLLM 生成的辅助过程标签替代人工训练下游审核模型,任务含 Hate Speech Detection,工业部署。差距:标注的是辅助 QA 过程标签非主任务 hate 标签、视频级、无冲突仲裁。**投稿前需复查其演化。**
- **MetaHarm**(ICWSM 2025, arXiv:2504.16304)——GPT-4-Turbo 是三个标注主体之一(专家/众包/GPT-4),标签入多数投票 ground truth;姊妹篇 arXiv:2411.05854 结论 GPT-4-Turbo ≈ 银标准非金标准。差距:视频级、平级标注者非裁决者、不训模型。
- **HateClipSeg**(ACM MM 2025)——微调 LLaMA-3.2-11B 做**预筛**(4,745→435),segment 标签纯人工、冲突由人工多数投票仲裁(非 LLM)。
- **TANDEM**(Qwen3-Omni silver 标注,gold 过滤)、**IARE**(GPT-4o-mini 辅助标注 + Qwen2.5-VL-72B 造 DPO 负例,均人工兜底)。
- 外围:HarmVideoBench(arXiv:2606.27187)、TikTok livestream 蒸馏(arXiv:2512.03553)、Youth safety audit(arXiv:2509.05838)。

### 扫描点 3 — 置信度门控的选择性 MLLM 调用:**PARTIAL(领域内学术基准上 OPEN)**

学术 hateful-video 基准(HateMM/MultiHateClip/ImpliHateVid)上**没有任何论文做过"轻量模型不确定才唤醒 MLLM"**;领域内全部 reasoning-VLM 方法(MARS/HVGuard/IARE/TANDEM/RAMF/LELA)均 always-on,RA-HMD 判决头切换是协议固定非动态。必须引用划界的相邻先例:
- **Filter-And-Refine**(TikTok, arXiv:2507.17204)——工业视频审核 MLLM 级联:router 用与人工挑选的高危 seed 视频库的 **embedding 相似度**砍掉 97.5% 流量,2.5% 进 MLLM ranker(单 token Yes/No + softmax)。差距:(a) **相似度路由,不是主分类器置信度/不确定性触发的 deferral**;(b) 12 类未公开 violation 的通用审核,非 hate 专项、非学术基准;(c) 未对检索条目做 in-context 推理。
- **Supporting Human Raters with LLMs**(Google, arXiv:2406.12800)——LLM 预筛清晰样本、边界样本升级给**人**(LLM→human 而非 classifier→MLLM),非视频专项。
- meme 域 **LMM Agents**(arXiv:2411.05383)占了"对检索到的标注样本做 in-context 推理"的表面形态,但 always-on、meme 域、检索原始样本。

### 扫描点 4 — 可审计/可编辑记忆:**OPEN(领域内),相邻领域有强先例**

hateful video 领域内没有任何检索记忆是人类可读、可人工增删改的:
- **MoRE**:特征三元组库,不可读不可编辑(PARTIAL 基线)。
- **Filter-And-Refine**:seed 库支持人工挑选入库(粗粒度可编辑),但条目是视频 embedding,用于路由非证据。
- **TikTok livestream**(arXiv:2512.03553):按 violation 类别分库的 HNSW embedding 索引,无人工编辑接口描述。
- **SafeLens**:审计日志非检索记忆。

**相邻领域两个强先例(reviewer 大概率提,必须主动引用划界)**:
1. **Class-RAG**(Meta GenAI, arXiv:2410.14881)——内容审核的可动态更新检索库("semantic hotfixing",库存正/负样例),免重训快速修补;对象是**文生图 prompt 审核**,非视频、条目是样例非结构化描述。
2. **Contextual Policy Engine**(arXiv:2508.06204)——检索库存**人类可读的 hate speech 政策文档**,可直接增删受保护群体免重训(F1 保持 0.97+);**纯文本域**。

→ 我们的叙述定位:把 Class-RAG/CPE 式可编辑库思想**首次带入多模态 hateful video**,条目形态从"样例/政策文档"升级为"MLLM 生成的逐视频结构化描述条目",并与 MoRE 的不可读特征库正面对比可审计性。

---

## 总表:三个候选角色 × 占位判定

| 候选角色 | 判定 | 依据(领域内) | 必须引用划界的最近先例 |
|---|---|---|---|
| **角色 1:MLLM 生成结构化描述条目(target/机制/载体/显隐性 schema)作为检索键,可审计** | **OPEN** | 视频域无一例"schema 条目 + 入库 + 检索键"合取:MARS/HVGuard/RAMF/LELA 全是自由文本即弃;TANDEM 有 XML 输出 schema 但无库;IARE harmful elements 半结构化但是推理中间物;MoRE 有库但条目是不可读特征且零 LLM | SafeLens(结构化 JSON 审计日志,即弃)、TANDEM(输出 schema,字段不同)、Pro-Cap(meme,probing 维度含 target,但产物是 caption 喂分类器非检索键)、Class-RAG/CPE(可更新/可读库,非视频、条目非 MLLM 生成描述) |
| **角色 2:训练时给弱监督片段当裁决者(共识分歧才调用)** | **PARTIAL** | 视频级 MLLM 标注已被占(IPS 用 MLLM 标签替人工训下游、含 HSD;MetaHarm GPT-4 标签入 ground truth;TANDEM silver 标注;HateClipSeg LLM 预筛;IARE LLM 辅助标注)。**但"segment 级弱标签 + 标签分歧才触发 MLLM 仲裁 + 训练学术基准模型"三要素合取无人做**;HateClipSeg 的 segment 冲突仲裁恰恰是纯人工 | IPS(arXiv:2412.15251,最接近,投稿前复查 v3+)、MetaHarm、HateClipSeg;写作时强调:片段级、**分歧触发**(选择性而非全量标注)、裁决者而非平级标注者 |
| **角色 3:推理时 kNN 不确定才唤醒 MLLM、对检索条目做 in-context 推理** | **领域内 OPEN(计入工业/meme 相邻先例则 PARTIAL)** | 领域内全部 reasoning-VLM 方法 always-on 无门控(MARS 置信度明文"只作解释不做阈值");RA-HMD 三头切换为协议固定、全文无置信度门控/一致性检查;MoRE 无 LLM | Filter-And-Refine(TikTok,**相似度路由**级联非置信度 deferral、非 hate 基准)、Google 2406.12800(LLM→人 escalation)、meme LMM Agents(arXiv:2411.05383,检索标注样本进 prompt 但 always-on、meme 域)。我们的差异:**kNN 邻居标签熵/间隔做门控 + 被检索的是结构化条目(与角色 1 闭环)+ 学术基准可复现** |

### 附:各方法机制速查表

| 方法 | venue | MLLM 角色 | 调用模式 | 输出形态 | 存储/复用 | retrieval/memory | 弱监督标注 |
|---|---|---|---|---|---|---|---|
| MARS | ICASSP 2026 | 直接判决(4 阶段对抗推理) | always-on | 标签+置信+自由文本 | 无 | 无 | 无 |
| HVGuard | EMNLP 2025 | rationale 当第四模态特征 | always-on | 自由文本 | 工程缓存 | 无 | 无 |
| IARE | SIGIR 2026 | 微调 MLLM 直接判决+rationale | always-on(双 MLLM) | 自由文本(输入侧半结构化) | 无 | 无 | LLM 辅助标注/造负例(人兜底) |
| TANDEM | arXiv 2026 | RL 微调 MLLM 结构化判决 | always-on(双模型×每 chunk) | **XML schema**(label/timestamps/targets/summary) | 瞬态(SCCR、跨 chunk) | 无 | Qwen3-Omni silver 标注 |
| RAMF | TMLR | 冻结 VLM 三视角文本当特征 | always-on | 自由文本 | 无 | 无 | 无 |
| LELA | arXiv 2026 | LLM 逐帧打分定位 | always-on(每帧×每模态) | 自由文本+标量分 | 无 | 无 | 无 |
| MoRE | WWW 2025 | **无 LLM** | — | — | — | **特征三元组记忆库+冻结检索器** | 无 |
| RA-HMD | EMNLP 2025 | LMM=encoder(+保留生成仅作解释) | 判决头协议固定,无门控 | 分类 logit / kNN 票 | FAISS embedding 库 | **有(嵌入级,换库即适配)** | 无 |

---

## 写作行动要点

1. **角色 1 的表述锚点**:"structured, human-auditable memory entries generated by an MLLM under a fixed schema (target / mechanism / modality carrier / implicitness), persisted as retrieval keys" —— 三件事(schema 化生成、持久化入库、作检索键)的**合取**是空位;任何单件都有 PARTIAL 先例(TANDEM/SafeLens/IARE/MoRE),related work 必须逐一点名划界。
2. **角色 2 的表述锚点**:selective / disagreement-triggered adjudication at **segment level** —— 与 IPS(全量视频级辅助标签)和 MetaHarm(平级标注者)的区别就是"片段级 + 只在共识分歧时调用 + 裁决者"。
3. **角色 3 的表述锚点**:**confidence-gated(kNN 邻居标签熵/相似度间隔)** deferral,被唤醒后对**检索到的结构化条目**(非原始样本)做 in-context 推理 —— 与 Filter-And-Refine 的相似度路由、与 meme LMM-Agents 的 always-on 样本检索划清界限;并注明上游 RA-HMD 无任何动态门控(角色 3 与上游正交)。
4. **避开的表述**(meme 线已占):"generate caption/description then classify"、"LLM rationale as feature / distillation"、"LLM debate/judge"、"probing questions for captioning"。
5. **风险跟踪**:IPS v3(2026-05)持续演化;Filter-And-Refine 若中 ACL industry track 可见度上升;SafeLens 与 MultiHateClip/HateClipSeg 同属 Roy Lee 组生态,"auditable" 话语撞车风险最高,须在论文中显式对比。

---

## 证据来源

- MARS: arXiv:2601.15115(HTML 全文)+ GitHub Multimodal-Intelligence-Lab-MIL/MARS
- RAMF: arXiv:2512.02743(HTML 全文)+ GitHub Multimodal-Intelligence-Lab-MIL/RAMF
- LELA: arXiv:2602.09637(HTML 全文)
- HVGuard: aclanthology.org/2025.emnlp-main.456(官方 PDF 全文)+ GitHub yihengjingWHU/HVGuard(CoT.py 源码)
- IARE: arXiv:2606.11953(HTML 全文)
- TANDEM: arXiv:2601.11178(HTML 全文)
- MoRE: research-wiki/papers/lang2025_biting_off_more.md(深读笔记)+ GitHub Jian-Lang/MoRE
- RA-HMD: arXiv:2502.13061v4(HTML 全文)+ 本地代码 `RA-HMD/LLAMA-FACTORY-Ver202512/src/llamafactory/custom/infer_utils.py`、`RA-HMD/Stage2/src/`
- meme 线: Pro-Cap arXiv:2308.08088;PromptHate aclanthology 2022.emnlp-main.22;Mr.Harm aclanthology 2023.findings-emnlp.611;ExplainHM arXiv:2401.13298;MemeGuard aclanthology 2024.acl-long.439;LMM Agents arXiv:2411.05383;IntMeme arXiv:2502.11073
- 四点扫描: SafeLens(AAAI-26 Demo, ojs.aaai.org 42390);IPS arXiv:2412.15251;MetaHarm ICWSM 2025 arXiv:2504.16304;MLLMs as Alternative Annotators arXiv:2411.05854;HateClipSeg arXiv:2508.01712;Filter-And-Refine arXiv:2507.17204;Supporting Human Raters arXiv:2406.12800;Class-RAG arXiv:2410.14881;Contextual Policy Engine arXiv:2508.06204;TikTok livestream arXiv:2512.03553;HarmVideoBench arXiv:2606.27187;Youth safety audit arXiv:2509.05838;LLM Semantic Augmentation arXiv:2504.15548
