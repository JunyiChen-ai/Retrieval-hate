# 查新报告:方向 A — Retrieval-Consensus Segment Denoising / 无 span 监督的时序仇恨定位

_日期:2026-07-02。查新口径:novelty 严格限定在 hateful/harmful VIDEO detection 领域内判断;其它领域(meme、通用视频理解、通用 MIL/WSVAD)只算 [相关领域,需引用]。_

_检索方法:多轮 WebSearch + Semantic Scholar citation API(HateClipSeg arXiv:2508.01712 的全部引用、MoRE DOI:10.1145/3696410.3714560 的全部 18 篇引用、MultiHateLoc arXiv:2512.10408 的全部引用)+ Social-AI-Studio 官网 + CRAVE GitHub + MultiHateLoc v3 全文 related work 交叉验证 + 本库 26 篇已读论文笔记(papers/)与 gap_map.md。_

---

## 1. 结论(一句话)

**机制层面 SAFE(置信度:中高,~75–80%);任务层面 THREATENED——"弱监督时序仇恨定位"这个任务已被 MultiHateLoc(WWW 2026)抢注,"training-free 定位"已被 LELA 抢注,方向 A 的卖点必须严格定位在"retrieval-consensus 去噪机制"上,不能定位在"任务第一"上。**

展开:在 hateful-video 领域内,**没有任何工作**用"检索记忆库 kNN 邻居标签投票 vs 自身视频级标签的一致性"给 sub-clip 定伪标签、去除 MIL 噪声正样本、再 EM 式自训练——这个机制(检索当标注器 / retrieval-as-annotator)在领域内完全无人认领,且没有任何工作把 retrieval 和 temporal localization 连接起来(MoRE/CRAVE/HCG-MPB 的 retrieval 全是 video-level 分类;MultiHateLoc/LELA/TANDEM 的定位全都不用 retrieval)。但"video-level 标签 → segment 定位"这个**任务**在 2025.08–2026.02 短短半年内已挤进 3 个玩家(MultiHateLoc、LELA、TANDEM),Exeter 组(Zeyu Fu)沿"temporal label noise → 弱监督定位"这条轨迹推进得非常快,**他们的下一步显而易见就是自动去噪**——时间风险真实存在,建议尽快做。

---

## 2. 领域内最近邻工作(hateful/harmful video,逐篇精确对比)

### 2.1 MultiHateLoc(Sun et al., arXiv:2512.10408, 2025.12,标注 WWW 2026)— **最大威胁,任务撞车**
- **它做了什么**:自称"第一个弱监督多模态仇恨定位框架"。只用 video-level 标签,输出 frame-level 仇恨分数。机制 = modality-aware temporal encoders + 动态跨模态融合(per-timestep gating)+ 跨模态对比对齐(同视频同时间戳跨模态为正样本)+ modality-aware MIL(Top-K, K=3)。HateMM frame-level mAP 0.645 / AUC 0.799;MHC mAP 0.445。
- **与方向 A 的精确差别**:已核对其 v3 全文——**不用伪标签、不用 retrieval/kNN、不做标签去噪**;它是纯 MIL-ranking + attention 路线(自述"inspired by WSVAD")。它的对比学习是"跨模态同时间戳对齐",不是检索标注的实例级对比。方向 A 的邻居投票伪标签、噪声正样本剔除、EM 自训练、可更新记忆库,它一样都没有。
- **后果**:"first weakly-supervised hate localization"这句话已经死了,不能再说。但它同时给了我们现成的 frame-level mAP/AUC 评测协议和必比 baseline。**必须正面对比。**

### 2.2 LELA(Sun et al., arXiv:2602.09637, 2026.02,Exeter 组)— 任务撞车(training-free 侧)
- **它做了什么**:第一个 training-free 的 LLM 仇恨定位框架:视频分解为 image/speech/OCR/music/context 五路 caption,多阶段 prompting 给每帧打仇恨分,composition matching 增强跨模态推理。HateMM/MHC 上超所有 training-free baseline。
- **与方向 A 的精确差别**:零训练、零标签、纯 prompting;没有 embedding 学习、没有记忆库、没有伪标签、没有迭代。方向 A 是"有 video-level 标签的训练方法",监督级别和机制都不同。**注意措辞冲突**:LELA 已占用"无需标注即可定位"的话语空间——方向 A 应自称 **"span-free supervision / 无片段标注的弱监督"**,避免用"annotation-free"(我们毕竟用了 video-level 标签)。

### 2.3 TANDEM(Koushik et al., arXiv:2601.11178, 2026.01)— 任务部分撞车(MLLM-RL 侧)
- **它做了什么**:把仇恨检测重构为结构化推理任务,VLM+audio-LM 串联 RL(GRPO/GSPO)互相优化,输出时间戳和目标群体,**明确声称"不需要 dense frame-level supervision"**。HateMM target-ID 0.73 F1。
- **与方向 A 的精确差别**:生成式 MLLM + RL 路线,时间戳是推理产物;无 embedding 空间、无检索记忆、无伪标签、无去噪概念。但它进一步坐实了"无 span 监督出时间戳"不是空白任务。

### 2.4 Yang et al. 温度标签噪声(arXiv:2508.04900, MUWS@ACM MM 2025,Exeter 组)— **最近的概念邻居 + 最好的动机引文,也是最大速度风险来源**
- **它做了什么**:诊断性论文:用 HateMM/MHC 的**金标时间戳**裁剪出纯仇恨片段,证明 video-level 标签是系统性时序标签噪声,裁剪后分类大幅提升(gap_map 记录 +19–30% headroom),呼吁 temporally-aware 模型。
- **与方向 A 的精确差别**:它只诊断、不治疗——去噪靠金标 timestamp,**没有提出任何自动去噪方法**。方向 A 恰好是"这篇论文的自动化版本"(retrieval consensus 替代金标 span)。完美的 setup 引文。**风险**:该组(Zeyu Fu:本篇 + MultiHateLoc + LELA + RAMF)正沿这条线连发,自动去噪是他们显而易见的下一步。

### 2.5 HateClipSeg(Wang, Wang, Lee, arXiv:2508.01712, ACM MM 2025, SUTD Social-AI-Studio)— 评测场,非威胁
- **它做了什么**:435 视频 / 11,714 segment 的 segment-level 标注数据集(Normal + 5 类 offensive + 目标群体),三个 benchmark 任务:trimmed 分类、时序定位、online 分类。定位 baseline 是**全监督**的(ActionFormer/LSTR 类,tIoU0.3 约 59.4)。
- **与方向 A 的精确差别**:纯数据集+全监督 baseline,论文本身不做弱监督、不做伪标签、不做 retrieval。**已逐篇核查其全部引用(截至本日仅 4 篇)**:Vietnamese audio toxic span(ICASSP 2026,监督式音频 span)、细粒度性别歧视视频数据集(De Grazia 2026)、StreamSense(见 2.6)、RAMF——**没有任何人在 HateClipSeg 上做弱监督定位**。它作为方向 A 的"免费金标评测场"仍然完全空置。

### 2.6 StreamSense(Han Wang et al., arXiv:2601.22738, WWW 2026, SUTD 组的后续)— 已核查,非威胁
- **它做了什么**:流式社交任务检测(含仇恨内容审核):轻量流式编码器 + 疑难样本选择性升级到 VLM + 决策延迟机制;训练用跨模态对比项 + **IoU 加权损失(用到 segment 边界标注,即监督式)**。
- **与方向 A 的精确差别**:这是 SUTD 组基于 HateClipSeg 的 online/efficiency 方向,不是弱监督方向;有 segment 监督、无 retrieval、无伪标签。说明该组的后续走了"流式部署"而非"segment 弱监督"——方向 A 的空窗仍在,但他们手握数据集,随时可以掉头。

### 2.7 MoRE(Lang et al., WWW 2025)— 领域内 retrieval 原点,已知,非威胁
- **它做了什么**:唯一已发表的 retrieval-augmented hateful-video 方法:冻结加权余弦检索 top-50 hateful + top-50 non-hateful,喂给三模态专家 + BHAN 双极注意力 + MoE 路由,全 BCE 监督。
- **与方向 A 的精确差别**:retrieval-for-experts,**检索结果只当上下文特征用,从不当"标注器"用**;video-level 分类,无 segment、无伪标签、无对比损失、无 EM 迭代、检索器不学习。**已逐篇核查其 18 篇引用**(见 2.8–2.10 及相关领域清单),没有一篇把 retrieval 用于伪标签/去噪/定位。

### 2.8 CRAVE(Hong et al., ICCV 2025,MoRE 同组 UESTC)— 领域内 retrieval 工作 #2,需引用,非威胁
- **它做了什么**:跨域检索增强(从资源丰富的 image-text 域 FHM/Fakeddit 检索,增强 malicious video 检测),评测含 **HateMM、MHClip-EN**、FakeTT、FVC。
- **与方向 A 的精确差别**:已核查其 GitHub——video-level 分类;检索的作用是跨域知识迁移(补数据稀缺),不是标注器;无 segment、无伪标签、无时序。

### 2.9 HCG-MPB(Hongxia Sun et al., ICMR 2026)— **唯一未完全核实的条目,投稿前需补查全文**
- **它做了什么**(仅从引用元数据获知):"Hierarchical Complementary Gating Mechanism with Multimodal Pattern Bank for Hateful Video Detection"——层级互补门控 + 多模态"模式库"做仇恨视频**分类**。
- **风险评估**:"Pattern Bank"是记忆库味道的组件,大概率是存储típ模式做注意力/检索增强分类(MoRE 变体),video-level;标题无任何 segment/弱监督/定位字样。多轮检索(S2 两次被限流、OpenAlex 503、Google 无正文)未获全文。**判断:对方向 A 威胁概率低,但这是本报告唯一没读到摘要的领域内 retrieval 系工作,投稿前必须补查。**

### 2.10 其余领域内工作(均为 video-level 分类或 TTA,一句话排除)
- **SCANNER**(Li et al., AAAI 2026):source-free TTA,momentum K-means 质心对齐;它把"伪标签自训练"当作被超越的 TTA baseline 提及,自己不用检索记忆、不做 segment。
- **ImpliHateVid/TCL**(ACL 2025)、**IARE**(SIGIR 2026)、**RAMF**、**MARS**(ICASSP 2026)、**HVGuard**、**MM-HSD**(ACM MM 2025)、**CMFusion**、**MultiHateGNN**(BMVC 2025)、**Ma et al. 2025**(WWW,few-shot 多模态表示):全部 video-level 分类,无 retrieval-as-annotator、无 segment 伪标签。
- **DeHate**(ACM MM 2025)、**MHC**(ACM MM 2024)、**HateMM**(ICWSM 2023):数据集自带 segment 时间戳/帧 span 标注 → 可做方向 A 的**额外**弱监督评测场(训练只用 video-level 标签,时间戳只用于评测),这一点没有任何方法类论文利用过。

---

## 3. [相关领域,需引用](不构成 prior art,但 reviewer 一定会问)

### 3.1 弱监督暴力/异常检测(WSVAD:UCF-Crime / XD-Violence 线)— 必须诚实正面处理的一条线
任务口径核实结论:**"他们通常是 MIL-ranking/attention,不是 retrieval-consensus"这个说法基本属实,但需要加一个诚实的限定——该领域确实存在"伪标签+自训练+标签噪声清洗"子线,只是没有任何工作用"带标签跨视频检索记忆库的 kNN 邻居投票"来定伪标签。** 具体:
- **Sultani et al.(CVPR 2018,UCF-Crime)**:MIL-ranking 开山,video-level 标签 → snippet 异常分。
- **Wu et al.(ECCV 2020,XD-Violence)**:多模态(音视频)弱监督暴力检测基准 + HL-Net。
- **Zhong et al.(CVPR 2019)"GCN label noise cleaner"**:⚠️ **精神上最接近的一篇**——把 WSVAD 重构为"噪声标签学习",用特征相似度图 + 时间一致性图做 GCN 传播来清洗 snippet 噪声标签,交替迭代训练(EM 味)。**精确差别**:清洗信号来自**同视频/批内特征图传播**,不是跨视频**带标签范例记忆库的 kNN 投票**;无检索、无"邻居视频级标签 vs 自身标签一致性"这个 consensus 判据;分类器是普通动作分类器,不是对比 embedding。related work 必引并明确划界。
- **MIST(Feng et al., CVPR 2021)**:multiple-instance 伪标签生成器 + 自引导注意力编码器,两阶段自训练。差别:伪标签来自模型自身 MIL 打分(self-labeled),不是检索外部带标签范例(retrieval-labeled);无记忆库。
- **Zhang et al.(CVPR 2023)"Exploiting Completeness and Uncertainty of Pseudo Labels"**、**Full-Stage Pseudo Label Quality Enhancement(arXiv:2407.08971)**、**Zhou et al.(CVPR 2023,train-test gap)**:伪标签质量增强线,均为模型自打分+过滤,非检索共识。
- **C2FPL**:无监督聚类生成 coarse-to-fine 伪标签——聚类共识与 kNN 共识有亲缘,但无标签记忆库、无"与 bag 标签一致性"判据。
- **RTFM(ICCV 2021)**、**UR-DMU(AAAI 2023,记忆单元)**、**HyperVD/双空间(2024)**、**多尺度 bottleneck transformer(2024)**:MIL/attention/记忆原型线,记忆是**学出来的原型**不是**带标签的检索范例**。
- **CKNN(2024,cleansed kNN 无监督 VAD)**:kNN 直接当异常打分器(测试时),不是训练时给 sub-clip 定伪标签。
- **VadCLIP / 时空 prompt 检索(ACM MM 2024)**、**SlowFastVAD(RAG-VLM,2025)**、**Geometry-Aware training-free VAD(2026)**:检索/RAG 出现在 VLM-prompting 侧,检索对象是概念文本/知识库,不是带标签视频范例,更不做伪标签共识。
- **EM-MIL(Luo et al., ECCV 2020)**:EM 式 MIL 弱监督动作定位——"EM 迭代"这个词的通用出处,必引。

### 3.2 弱监督时序动作定位(WS-TAL)伪标签线
video-level 标签 → segment 伪标签 → 回归学生模型,是 WS-TAL 的成熟范式(如 Zhou CVPR 2023、arXiv:2407.08971、PVLR 等)。差别同上:伪标签源是模型自打分/CAM,非带标签检索记忆。

### 3.3 音视频视频解析(AVVP)segment 伪标签线
- **Zhou et al.(arXiv:2406.00919)**:用 CLIP/CLAP 给 segment 打伪标签(video-level → segment-level);**Rachavarapu et al.(CVPR 2024)**:prototype-based pseudo-labeling。"弱标签 → segment 伪标签"在 AVVP 已成熟;差别:伪标签源是预训练模型 zero-shot 打分或原型,不是**领域内带标签范例的 kNN 投票**,且无"与自身 bag 标签一致性"判据。

### 3.4 通用噪声标签学习的 kNN/邻居共识线(机制的通用祖先,必引划界)
- **Bahri et al.(ICML 2020)Deep k-NN for Noisy Labels**:kNN 邻居标签不一致 → 判为噪声样本剔除——**方向 A 判据的最直接通用祖先**。
- **Iscen et al.(CVPR 2022)Neighbor Consistency Regularization**;**Northcutt et al. Confident Learning(JAIR 2021)**;**TopoFilter**;**Litrico et al.(CVPR 2023)**(SFDA 中用邻居聚合精炼伪标签);**Citation-kNN(Wang & Zucker 2000)**(MIL 的 kNN 古典方法,bag 级)。
- 划界话术:这些都是**实例级、单模态、有独立实例标签假设**;方向 A 的新颖组合在于(a)实例(sub-clip)本身**没有**标签,标签是从 bag(video)继承的 MIL 噪声标签;(b)共识来自**多模态检索记忆库中带 video-level 标签的邻居**;(c)判据是"邻居投票 vs 继承标签的一致性",服务于 MIL 正包去噪 + 对比自训练;(d)副产品是时序定位。

### 3.5 Meme / 图文域(inspiration)
- **RGCL(Mei et al., ACL 2024)/ RA-HMD(2025)**:我们自己的机制来源——FAISS 伪金正样本 + 硬负例 + kNN 多数投票分类头。从未用于视频、从未用于 sub-instance 伪标签/去噪。
- **ALARM(Lang et al., KDD 2025)**:label-free harmful meme 检测,LMM agent 自提升(伪标签味道,meme 域)。

### 3.6 相邻 harmful-video 应用(需引,均非 prior art)
- **RAVEN / RAVEN++(EMNLP 2025 Industry)**:广告违规视频的时序定位,RL 推理路线(有监督+RL,非弱监督检索)。
- **KDD 2026 直播审核(arXiv:2512.03553)**:工业系统,"reference-based similarity matching"——把进来的内容与**带标签违规参考样例**做相似度匹配当分类器。⚠️ 这是 harmful-content 工业界"检索当分类器"的实例(与 RGCL kNN-vote 头同族),但是流级/视频级判决、无 segment 伪标签、无去噪训练、非 hateful-video 学术基准。引用并划界即可。
- **Vy Huynh et al.(ICASSP 2026)**:越南语音频 toxic span 检测(全监督、纯音频)。

---

## 4. 威胁分析与规避/重定位建议

### 4.1 什么已经死了(不能再写的话)
1. ~~"我们是第一个只用 video-level 标签做时序仇恨定位的工作"~~ — MultiHateLoc 已抢注(且 LELA、TANDEM 分占 training-free 与 MLLM-RL 侧翼)。
2. ~~"annotation-free hate localization" 作为主标语~~ — 与 LELA 的 training-free 话语空间冲突,且我们其实用了 video-level 标签。改用 **"span-free / 无片段标注监督"**。
3. ~~"仇恨视频里没人研究时序标签噪声"~~ — Yang et al. 2025 已系统诊断(但只诊断,这是我们的机会而非障碍)。

### 4.2 什么仍然安全(可主张的 claim,按防御力排序)
1. **机制第一性**:hateful-video 领域内第一个"检索共识伪标签"(retrieval-as-annotator):用带标签跨视频记忆库对 sub-clip 做 kNN,以邻居 video-level 标签投票与继承标签的一致性来剔除 MIL 噪声正样本 —— 领域内零重合;且在通用 WSVAD/WS-TAL/噪声标签文献里也找不到完全相同的组合(最近的是 Zhong 2019 图传播清洗与 Bahri 2020 kNN 判据,均可清晰划界)。
2. **第一个把 retrieval 和 temporal localization 连接的 hateful-video 工作**:MoRE/CRAVE/HCG-MPB 的检索全在 video-level 分类;MultiHateLoc/LELA/TANDEM 的定位全都无检索。交叉点是空的。
3. **第一个在 HateClipSeg 上做弱监督定位评测的方法**(该数据集的定位任务目前只有全监督 baseline,引用者中无人做弱监督)——先到先得,但窗口在收窄。
4. **对我们自己 G4 负结果的闭环**:gap_map 已记录 auto sub-clip 的 MIL 噪声正样本导致 seg-mode 语言间符号翻转——方向 A 正是对这个已确诊病因的对症治疗,叙事上自洽("我们先证明了噪声存在且有害,再给出去噪机制")。

### 4.3 重定位建议(具体)
1. **主 claim 定位为"去噪机制"而非"定位任务"**:标题/摘要主张 retrieval-consensus denoising 提升(a)video-level 分类(联动 Yang 2025 的 headroom 诊断)和(b)现有 RGCL 框架的 seg-mode 稳定性;时序定位降格为"free byproduct + 在 HateClipSeg 金标上的外部验证"。这样 MultiHateLoc 变成 baseline 而非竞争 claim。
2. **必比 baseline 矩阵**(reviewer 一定会要):MultiHateLoc(frame mAP/AUC 协议)、LELA(training-free 上界参考)、至少一个 WSVAD 移植(MIST 式自打分伪标签、RTFM)、以及"Zhong 2019 式图传播清洗 vs 我们的检索共识清洗"的机制消融——最后这个消融是把 [相关领域] 威胁转化为卖点的关键实验。
3. **消融必须回答**:检索共识伪标签 vs 模型自打分伪标签(MIST 式)谁更好?这是方向 A 相对整个 WSVAD 伪标签线的一票定生死实验。若赢,novelty 从"组合新"升级为"机制优";若输,方向 A 只剩组合新颖性。
4. **措辞**:全文用 "span-free supervision"、"retrieval-consensus pseudo-labeling"、"retrieval-as-annotator";避免 "annotation-free"、"first weakly-supervised localization"。
5. **投稿前 TODO**:(a) 补查 HCG-MPB(ICMR 2026)全文确认 pattern bank 无伪标签成分;(b) 重扫 HateClipSeg 新增引用(现在只有 4 篇,增长很快);(c) 盯 Exeter 组(Zeyu Fu)和 SUTD 组(Roy Ka-Wei Lee)的 arXiv 新帖——两组都离这个 idea 一步之遥。**速度是本方向最大的非技术风险。**

### 4.4 残余不确定性(诚实声明)
- HCG-MPB(ICMR 2026)未读到摘要正文(S2/OpenAlex 限流),按标题与引用上下文判断为 video-level 分类,威胁概率低但未闭环。
- 2026 年 5–7 月最新 arXiv 预印本索引可能不全(WWW 2026 / ACL 2026 camera-ready 潮);建议投稿前一个月再跑一次本查新。
- 置信度综合评为 **~75–80%**:领域内三条引用链(HateClipSeg/MoRE/MultiHateLoc)已穷尽式核查,主要不确定性来自未索引的最新预印本。

---

## 附:检索证据源(节选)
- HateClipSeg: [arXiv:2508.01712](https://arxiv.org/abs/2508.01712), [ACM MM 2025](https://doi.org/10.1145/3746027.3758289), [GitHub](https://github.com/Social-AI-Studio/HateClipSeg);引用列表经 Semantic Scholar API 全量核查(4 篇)。
- MultiHateLoc: [arXiv:2512.10408](https://arxiv.org/abs/2512.10408)(v3 全文核查:无伪标签/无检索);引用 2 篇(TANDEM、RAMF)。
- MoRE: [WWW 2025](https://dl.acm.org/doi/10.1145/3696410.3714560), [GitHub](https://github.com/Jian-Lang/MoRE);引用 18 篇经 S2 API 全量核查。
- CRAVE: [ICCV 2025 论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Hong_Borrowing_Eyes_for_the_Blind_Spot_Overcoming_Data_Scarcity_in_ICCV_2025_paper.pdf), [GitHub](https://github.com/ronpay/CRAVE)(数据集/机制核查)。
- LELA: [arXiv:2602.09637](https://arxiv.org/pdf/2602.09637);TANDEM: [arXiv:2601.11178](https://arxiv.org/abs/2601.11178);StreamSense: [arXiv:2601.22738](https://arxiv.org/abs/2601.22738), [GitHub](https://github.com/Social-AI-Studio/StreamSense);Yang 标签噪声: [arXiv:2508.04900](https://arxiv.org/abs/2508.04900)。
- WSVAD 线:[MIST CVPR 2021](https://arxiv.org/abs/2104.01633)(内含 Zhong 2019 GCN cleaner 机制对比)、[Zhang CVPR 2023](https://arxiv.org/abs/2212.04090)、[CKNN](https://arxiv.org/pdf/2408.03014)、[SlowFastVAD](https://arxiv.org/pdf/2504.10320)、[EM-MIL](https://arxiv.org/pdf/2004.00163)、[XD-Violence 相关检索结果](https://arxiv.org/pdf/2101.10030)。
- 直播审核 KDD 2026: [arXiv:2512.03553](https://arxiv.org/html/2512.03553v1);RAVEN++: [arXiv:2511.19168](https://arxiv.org/abs/2511.19168)。
- RGCL(inspiration, meme): [arXiv:2311.08110](https://arxiv.org/abs/2311.08110)。
