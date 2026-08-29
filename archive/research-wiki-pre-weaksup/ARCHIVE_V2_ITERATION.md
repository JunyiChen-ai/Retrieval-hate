# 档案 v2 迭代:target 召回修复 + mechanism 幻觉约束(终报)

> 动机:AUDIT_archive_faithfulness.md 两大缺陷 —— (1) target_groups 召回过低(抽样 ZH 30 条仅 1 条非空、EN 6/30;全量 ZH 1.4%/EN 6.5%),直接导致记忆编辑 ZH 半边 0 翻转;(2) 9/60 幻觉多为给 benign 内容安 mechanism(全量代理:gt-Normal 上 mechanism 非空 ZH 87.7%/EN 79.2%)。
> 本轮:prompt v2 一次修两个缺陷,版本隔离生成 + 全量统计 + ZH/EN 记忆编辑复测。
> **计划变更**:原定 4 个键增强对照训练(GROUP=RAC_video_archive_v2)在多 seed post-mortem(键增强无稳健 accuracy 效应、seed-0 增益为选点伪影、sha1 审计确认 knn 键不参与训练)后被取消;键对比改为冻结 v1 获胜头下的**推理时换键对比**(零训练,且无重训混淆)。

## 1. 诊断与 prompt v2 改动

v1 prompt 对 target_groups 只说 "use [] if no group is targeted",从不**要求**在有攻击时点名群体;对"辱骂词自指目标群体"(娘炮→女性化男性、泼妇/母夜叉→女性)无指引;仅标题携带的攻击常被忽略。mechanism 侧无证据门槛,模型习惯性填非空(v1 全量 84-91% 非空,而数据 ~69% 是 Normal)。

v2(`src/utils/generate_video_archive_HF.py`,`--prompt_version v2`,v1 默认且字节不变):
- **target 必填规则**:攻击任何群体(race/ethnicity/nationality-region/religion/gender/sexual orientation & gender expression/disability/occupation/…)时 MUST 点名;slur 自指目标群体(给出 effeminate men / women / nationality 例);仅标题的攻击同样计入;[] 仅限无群体目标(纯个人攻击/benign)。
- **mechanism 证据约束**:只有能"引用"到具体证据才可填;涉性医疗科普/测评/祝福/儿歌/新闻等敏感话题的 benign 内容不是攻击;"Never fill in a mechanism without sufficient evidence"。
- **一致性规则**:mechanism 非空且攻击指向群体 ⇒ target 必须点名;explicitness=none ⇒ mechanism 与 target 必须双空。
- 保留 v1 的中性取证 system prompt、严格 JSON、固定 mechanism 词表、全英文 pivot。

**版本隔离**:v2 JSONL → `data/Archive/{MHC_zh,MHC}/v2/`,v2 .pt → `data/CLIP_Embedding/{MHC_zh,MHC}/v2/`;v1 文件与路径未动(多个在跑实验依赖 v1);新增 `scripts/slurm/gen_archive_v2.sbatch`,v1 sbatch 未动。

## 2. Smoke 门(各 30 条 train 前缀,同条目 v1/v2 配对)

| 指标 | ZH v1 | ZH v2 | EN v1 | EN v2 |
|---|---|---|---|---|
| target 非空(harmful 子集) | 1/10 (10%) | **7/10 (70%)** | 2/8 (25%) | **5/8 (62.5%)** |
| target 非空(整体) | 3.3% | 26.7% | 10% | 30% |
| mechanism 非空(gt-Normal,幻觉代理) | 75% | **45%** | 81.8% | **54.5%** |
| parse_ok | 30/30 | 30/30 | 29/30 | 30/30 |

人工核对非空 target:ZH 8 条中 6 条有据(women×3、effeminate men、men 弱、1 条新闻虚报、1 条个人误判为群体);EN 9 条中约 6 条有据(FAGGY FF→effeminate men、"Everyone's a whore"→women 等),1 条明确虚报(亲 LGBT 历史科普)。两门均判达标后才提全量。

## 3. 全量 v1 vs v2(双语对照表;生成 job:ZH 12258 / EN 12280,均 parse_ok 100%)

| 指标 | ZH v1 (n=806) | ZH v2 | EN v1 (n=790) | EN v2 |
|---|---|---|---|---|
| target 非空 · harmful (n=253/242) | 1.6% | **49.0%** | 11.6% | **54.5%** |
| target 非空 · 整体 | 1.4% | 22.5% | 6.5% | 30.5% |
| target 非空 · Normal(虚报风险面) | 1.3% | 10.3% | 4.2% | 19.9% |
| mechanism 非空 · Normal(幻觉代理) | 87.7% | **59.0%** | 79.2% | **50.9%** |
| mechanism 非空 · harmful | 96.8% | 83.4% | 93.4% | 78.9% |
| explicitness=none 却有 mechanism(结构违规) | 14.6% | **1.0%** | 10.3% | **0.1%** |
| parse_ok / schema_ok | 99.5/98.4% | 100/99.0% | 99.7/98.7% | 100/99.9% |

ZH v2 target 分布(按频次):women 81(54 harmful)、men 35、**effeminate men 32(30 harmful,= 娘炮家族的英文 pivot)**、gay people 6、chinese/japanese/taiwanese 各 4、…——语义结构与数据集内容相符。

### 审计失败案例定点复查(paired)

- **target 漏报(AUDIT #5)**:ZH 8 条点名案例 v2 填了 6(娘炮×3 → 'effeminate men' 全中、泼妇 2/3 → 'women'、母夜叉 → 'women';小日子一条填成 'women' 属方向错误);EN 3 条填 1(**旗舰案例反同布道 l3eUapefQog → 'gay people' 修复**;Chelsea rent boys 反而退化为全空 = 新增洗白;GTA hooker 未修)。
- **benign 幻觉(AUDIT #1)**:ZH 6 条清了 2(儿歌、舰长直播);**涉性医疗 stereotyping 模式未根除**(阳痿偏方/阴茎异常勃起/阴超 3 条仍在,其一还从 stereotyping 升级为 slur+explicit);EN 3 条清 1(包皮环切广告),亲 LGBT 历史科普仍被 target+stereotyping(该类"LGBT 词面≠仇恨"失败模式 v2 未修)。
- **标题洗白(AUDIT #2)**:未修。Floozy 空乘与牛油果酱标题(转写空、画面 benign)在 v2 仍全空;破鞋一条从 stereotyping/implicit 提为 insult/explicit(方向性改善但 target 仍空)。标题-only 毒性 + 空转写场景下,"标题计入内容"的指令不足以扭转画面主导的判断。

## 4. 键对比(冻结 v1 获胜头,推理时换键;复现门逐位通过)

| | v1 键(=训练日志) | v2 键 |
|---|---|---|
| ZH (job 12207, ep18) | acc 0.8523 / F1 0.8270 | acc 0.8255 / F1 0.7875 |
| EN (job 12210, ep24) | acc 0.8075 / F1 0.7626 | acc 0.8012 / F1 0.7462 |

v2 键在两种语言上都**不带来 accuracy 收益**(ZH −2.7 acc 点,EN −0.6),与 post-mortem 结论一致:档案键增强不是 accuracy 手段;v2 的付费点在可审计性/可编辑性,不在准确率。(原定重训对照已取消,此表即键差异的无混淆版本。)

## 5. 记忆编辑复测(v2 target 字段定向删除;协议同 v1 demo:topk20/arithmetic/α0.25,5-seed 随机对照)

切片按 **target_groups 字段**匹配;LGBT 家族正则补了英文 pivot(effeminate 等)——这是 v1 正则中 娘炮/人妖 的语义等价翻译(v1 输出中文词面、v2 输出英文 pivot),非新家族。

**ZH(复测主体;v1 时 target 字段只能寻址 ~3 条记忆,0 翻转)**

| 切片(记忆条目/test 切片) | 编辑翻转 切片/其余 | 随机对照 切片/其余 | 切片 acc 编辑前→后(随机均值) |
|---|---|---|---|
| LGBTQ+ by target 字段(20/11) | 0 / 2 | 0.0 [0,0] / 0.4 | 0.8182→0.8182 (0.8182) |
| women by target 字段(63/10) | **1 / 5** | 0.2 [0,1] / 2.4 | 0.70→0.60 (0.68) |
| LGBTQ+ 全文关键词 v1 同款(16/12) | 0 / 1 | 0.0 [0,0] / 0.4 | 0.8333→0.8333 |

**EN(同协议对照)**

| 切片(记忆条目/test 切片) | 编辑翻转 切片/其余 | 随机对照 切片/其余 | 备注 |
|---|---|---|---|
| LGBTQ+ by target 字段(74/14) | **2 / 4** | 0.0 [0,0] / 3.2 | 切片翻转只在定向删除下发生(5 个随机 seed 均为 0);整体 acc 不降(0.8012→0.8012) |
| women by target 字段(45/12) | 0 / 0 | 0.0 / 2.0 | |
| LGBTQ+ 全文关键词(100/27) | 2 / 6 | 0.0 / 2.2 | |

**结论(诚实版)**
1. **ZH 的 blocking 缺陷已修:可寻址性恢复。** v1 中 target 字段在 ZH 记忆库中只能寻址 ~3 条(train 非空 6/583),"编辑不到";v2 可按字段寻址 LGBT 家族 20 条(17 harmful/3 normal)与 women 家族 63 条。删除操作本身仍是纯 CPU 索引编辑、秒级、零训练。
2. **但 ZH 切片级行为效应仍未证得。** LGBT 切片(n=11)编辑后 0 翻转;women 切片有弱方向信号(切片翻转 1 vs 随机 0.2,切片 acc 降幅超随机均值)但 n=10 不足以下 claim。可审计性 claim 的 ZH 半边:**操作层面(寻址+删除)补上了,行为效应层面证据不足**——受小切片 n 与 α=0.25 键权重(fused 通道主导)限制,不宜宣称超过"可寻址、可删除"。
3. **EN 方向性在 target 字段切片下复现且更干净**:切片翻转 2/14 只在定向删除下出现(随机 5 seed 全 0),整体 acc 不掉(v1 demo 删 91 条掉 3 点;v2 字段切片删 74 条不掉)。
4. **新 caveat**:v2 在 Normal 内容上的 target 非空率升至 10-20%,EN LGBT 字段切片删除的 74 条里 44 条是 gt-Normal——**按 target 字段划片会捎带 benign 的 LGBT 话题内容**(亲 LGBT 科普被打 target 的失败模式未修),平台场景下需叠加 explicitness/mechanism 过滤或人工复核队列。

## 6. v2 是否真的更好(总判定)

**是,但只在它声称的维度上,且有残留缺陷:**
- 修好了:target 召回(harmful 子集 ZH×30、EN×5 的量级提升;审计点名案例大部分命中)、结构一致性(违规 14.6%→1.0%)、JSON 纪律无损、ZH 编辑可寻址性(0→20/63 条)。
- 改善但未根除:benign mechanism 幻觉(59%/51% 仍高;涉性医疗模式顽固)。
- 未修:标题-only 毒性洗白(2 条旗舰案例依旧);"LGBT 词面=目标"的 benign 误伤转移到了 target 字段;EN 出现 1 例新增洗白(Chelsea rent boys)。
- 代价:v2 键无 accuracy 收益(ZH −2.7 点)——键通道请继续用 v1 或视为纯审计工件。
- 建议的 v3 方向(未执行):few-shot 对比示例(医疗科普 vs 攻击)、target 字段区分"话题涉及"与"被攻击"、标题-only 毒性单独出线索字段。

## 7. 产物与协议

- 代码:`src/utils/generate_video_archive_HF.py`(+`--prompt_version`,v1 字节不变;selftest 通过)、`scripts/slurm/gen_archive_v2.sbatch`、`scripts/analysis/memory_editing_demo_v2.py`(协议函数 import 自 v1 demo)。
- 数据:`data/Archive/{MHC_zh,MHC}/v2/*.jsonl`(806+790 条,parse_ok 100%)、`data/CLIP_Embedding/{MHC_zh,MHC}/v2/*.pt`(契约校验:ids 单子表=gt 顺序、Dt=768、zero-vec=0、shape==v1)。
- 结果:`research-wiki/DEMO_memory_editing_v2_{zh,en}.md` + `_results.json`;smoke/全量统计脚本在 scratch(一次性)。
- B2:`archives/{MHC_zh,MHC}/v2/`、`embeddings/{MHC_zh,MHC}/v2/`;本报告与 demo 结果推送至 `research-wiki/`。
- SLURM:12234(ZH smoke)、12258(ZH 全量+编码)、12259(EN smoke)、12280(EN 全量+编码);对照训练按计划变更取消,未提交。
- 复现门:两语言 v1 键 baseline 与训练日志逐位一致(ZH 0.8523/0.8270,EN 0.8075/0.7626)。

(统计与表格由脚本生成后人工核对;审计编号引用 research-wiki/AUDIT_archive_faithfulness.md。)
