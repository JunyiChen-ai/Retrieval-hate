# EVAL: Role 3 — 置信门控的选择性推理(kNN margin gate + Qwen2.5-VL 仲裁)

> **诚实条款**:deferred 子集很小(test 每工作点 15–42 条),只报绝对数,不做显著性声明。门槛全部在 val 上选(deferral 率 ≈10/20/30% 三个工作点),工作点本身也由 val 仲裁后 acc 选出;test 不参与任何调参。MLLM 解析失败回退 kNN 判决(计数在表中)。

## 协议

- 基座:获胜 archive-kNN α=0.25 seed0(EN frozen-Qwen job 12210 ckpt epoch24;ZH LoRA job 12207 ckpt epoch18),零训练;复现门 bit-identical (EN 0.8075/0.7626, ZH 0.8523/0.8270,见 gate_MHC_base.json / gate_MHC_zh_base.json)。
- 门控:margin = |similarity-signed arithmetic vote|(与训练日志逐位一致的投票);val margin 分位数取门槛,defer ⇔ margin < t(三工作点嵌套)。
- 仲裁:frozen Qwen2.5-VL-7B-Instruct,输入 = 16 帧 + title/transcript + 该视频自己的档案 + top-5 检索邻居的档案+gt 标签(证据卡);输出严格 JSON {verdict: hateful/offensive/normal, key_evidence, cited_neighbor};hateful/offensive→1, normal→0 后仅替换 deferred 样本的判决。
- 变体:base;memory-clean(先删 DEMO_memory_editing 的 2 条 W2 噪声记忆,合法:源于训练侧取证);text-only 仲裁对照(不给帧)。
- 仲裁 prompt 两版:v1(通用平台安全口径)与 v2(按数据集标注口径重校准 + 邻居给三分类细标签;动因:v1 smoke 显示系统性 over-flagging)。**v1/v2 与工作点一起只在 val 上选**,test 两版都报告。
- **v3 = 任务校准 LoRA 仲裁器**:base Qwen + logging/lora/<DS> adapter(MHClip train 上一词答 SFT,与 ZH 获胜编码器同源),prompt 同 v2、JSON-first 合同不变,新增一词裸答 fallback(单独计数);同一 deferred 队列,与 v1/v2 同池在 val 上选配置。

## MHC (EN)

**val 选定配置:不启用仲裁(保持 kNN,deferral 0%)**——所有 (prompt, rate) 候选在 val 上的 after-acc 都不超过 before 0.7875;门控的诚实决策是不花推理预算。

### val,prompt v1(配置选择依据;N=80)

| rate | defer n | before acc/F1 | after acc/F1 | 仲裁正确率(MLLM) | kNN在deferred上 | flips(好/坏) | 回退 |
|---|---|---|---|---|---|---|---|
| 0.10 | 8 (10.0%) | 0.7875 / 0.7554 | 0.7750 / 0.7484 | 0.5000 (n=8) | 0.6250 | 3 (1/2) | 0 |
| 0.20 | 16 (20.0%) | 0.7875 / 0.7554 | 0.7375 / 0.7181 | 0.3750 (n=16) | 0.6250 | 8 (2/6) | 0 |
| 0.30 | 24 (30.0%) | 0.7875 / 0.7554 | 0.7250 / 0.7067 | 0.4167 (n=24) | 0.6250 | 9 (2/7) | 0 |

### val,prompt v2(配置选择依据;N=80)

| rate | defer n | before acc/F1 | after acc/F1 | 仲裁正确率(MLLM) | kNN在deferred上 | flips(好/坏) | 回退 |
|---|---|---|---|---|---|---|---|
| 0.10 | 8 (10.0%) | 0.7875 / 0.7554 | 0.7750 / 0.7484 | 0.5000 (n=8) | 0.6250 | 3 (1/2) | 0 |
| 0.20 | 16 (20.0%) | 0.7875 / 0.7554 | 0.7375 / 0.7181 | 0.3750 (n=16) | 0.6250 | 8 (2/6) | 0 |
| 0.30 | 24 (30.0%) | 0.7875 / 0.7554 | 0.7250 / 0.7067 | 0.4167 (n=24) | 0.6250 | 9 (2/7) | 0 |

### val,prompt v3(配置选择依据;N=80)

| rate | defer n | before acc/F1 | after acc/F1 | 仲裁正确率(MLLM) | kNN在deferred上 | flips(好/坏) | 回退 |
|---|---|---|---|---|---|---|---|
| 0.10 | 8 (10.0%) | 0.7875 / 0.7554 | 0.7750 / 0.7436 | 0.5000 (n=8) | 0.6250 | 3 (1/2) | 0 |
| 0.20 | 16 (20.0%) | 0.7875 / 0.7554 | 0.7500 / 0.7205 | 0.4375 (n=16) | 0.6250 | 7 (2/5) | 0 |
| 0.30 | 24 (30.0%) | 0.7875 / 0.7554 | 0.7375 / 0.7091 | 0.4583 (n=24) | 0.6250 | 8 (2/6) | 0 |

### test(before/after,每变体×prompt 版)

| 变体 | rate | defer n (率) | MLLM calls | before acc/F1 | after acc/F1 | Δacc | 仲裁正确率 | kNN@deferred | flips(好/坏) | 回退 |
|---|---|---|---|---|---|---|---|---|---|---|
| base+frames+v1 | 0.10 | 16 (9.9%) | 16 | 0.8075 / 0.7626 | 0.7888 / 0.7506 | -0.0186 | 0.5000 (n=16) | 0.6875 | 7 (2/5) | 0 |
| base+frames+v1 | 0.20 | 27 (16.8%) | 27 | 0.8075 / 0.7626 | 0.7640 / 0.7274 | -0.0435 | 0.4444 (n=27) | 0.7037 | 11 (2/9) | 0 |
| base+frames+v1 | 0.30 | 39 (24.2%) | 39 | 0.8075 / 0.7626 | 0.7578 / 0.7244 | -0.0497 | 0.4615 (n=39) | 0.6667 | 14 (3/11) | 0 |
| base+frames+v2 | 0.10 | 16 (9.9%) | 16 | 0.8075 / 0.7626 | 0.7888 / 0.7506 | -0.0186 | 0.5000 (n=16) | 0.6875 | 7 (2/5) | 0 |
| base+frames+v2 | 0.20 | 27 (16.8%) | 27 | 0.8075 / 0.7626 | 0.7640 / 0.7274 | -0.0435 | 0.4444 (n=27) | 0.7037 | 11 (2/9) | 0 |
| base+frames+v2 | 0.30 | 39 (24.2%) | 39 | 0.8075 / 0.7626 | 0.7640 / 0.7302 | -0.0435 | 0.4872 (n=39) | 0.6667 | 13 (3/10) | 0 |
| base+frames+v3 | 0.10 | 16 (9.9%) | 16 | 0.8075 / 0.7626 | 0.8012 / 0.7596 | -0.0062 | 0.6250 (n=16) | 0.6875 | 7 (3/4) | 0 |
| base+frames+v3 | 0.20 | 27 (16.8%) | 27 | 0.8075 / 0.7626 | 0.7888 / 0.7477 | -0.0186 | 0.5926 (n=27) | 0.7037 | 9 (3/6) | 0 |
| base+frames+v3 | 0.30 | 39 (24.2%) | 39 | 0.8075 / 0.7626 | 0.7950 / 0.7566 | -0.0124 | 0.6154 (n=39) | 0.6667 | 10 (4/6) | 0 |
| clean+frames+v1 | 0.10 | 18 (11.2%) | 18 | 0.8199 / 0.7748 | 0.7888 / 0.7506 | -0.0311 | 0.5556 (n=18) | 0.8333 | 9 (2/7) | 0 |
| clean+frames+v1 | 0.20 | 26 (16.1%) | 26 | 0.8199 / 0.7748 | 0.7702 / 0.7331 | -0.0497 | 0.4615 (n=26) | 0.7692 | 12 (2/10) | 0 |
| clean+frames+v1 | 0.30 | 39 (24.2%) | 39 | 0.8199 / 0.7748 | 0.7578 / 0.7244 | -0.0621 | 0.4615 (n=39) | 0.7179 | 16 (3/13) | 0 |
| clean+frames+v2 | 0.10 | 18 (11.2%) | 18 | 0.8199 / 0.7748 | 0.7888 / 0.7506 | -0.0311 | 0.5556 (n=18) | 0.8333 | 9 (2/7) | 0 |
| clean+frames+v2 | 0.20 | 26 (16.1%) | 26 | 0.8199 / 0.7748 | 0.7702 / 0.7331 | -0.0497 | 0.4615 (n=26) | 0.7692 | 12 (2/10) | 0 |
| clean+frames+v2 | 0.30 | 39 (24.2%) | 39 | 0.8199 / 0.7748 | 0.7640 / 0.7302 | -0.0559 | 0.4872 (n=39) | 0.7179 | 15 (3/12) | 0 |
| clean+frames+v3 | 0.10 | 18 (11.2%) | 0 | 0.8199 / 0.7748 | 0.8199 / 0.7748 | +0.0000 | — (n=0) | 0.8333 | 0 (0/0) | 0 |
| clean+frames+v3 | 0.20 | 26 (16.1%) | 0 | 0.8199 / 0.7748 | 0.8199 / 0.7748 | +0.0000 | — (n=0) | 0.7692 | 0 (0/0) | 0 |
| clean+frames+v3 | 0.30 | 39 (24.2%) | 0 | 0.8199 / 0.7748 | 0.8199 / 0.7748 | +0.0000 | — (n=0) | 0.7179 | 0 (0/0) | 0 |
| base+textonly+v1 | 0.10 | 16 (9.9%) | 16 | 0.8075 / 0.7626 | 0.7888 / 0.7506 | -0.0186 | 0.5000 (n=16) | 0.6875 | 7 (2/5) | 0 |
| base+textonly+v1 | 0.20 | 27 (16.8%) | 27 | 0.8075 / 0.7626 | 0.7640 / 0.7274 | -0.0435 | 0.4444 (n=27) | 0.7037 | 11 (2/9) | 0 |
| base+textonly+v1 | 0.30 | 39 (24.2%) | 39 | 0.8075 / 0.7626 | 0.7578 / 0.7244 | -0.0497 | 0.4615 (n=39) | 0.6667 | 14 (3/11) | 0 |
| base+textonly+v2 | 0.10 | 16 (9.9%) | 16 | 0.8075 / 0.7626 | 0.7888 / 0.7506 | -0.0186 | 0.5000 (n=16) | 0.6875 | 7 (2/5) | 0 |
| base+textonly+v2 | 0.20 | 27 (16.8%) | 27 | 0.8075 / 0.7626 | 0.7702 / 0.7331 | -0.0373 | 0.4815 (n=27) | 0.7037 | 10 (2/8) | 0 |
| base+textonly+v2 | 0.30 | 39 (24.2%) | 39 | 0.8075 / 0.7626 | 0.7702 / 0.7359 | -0.0373 | 0.5128 (n=39) | 0.6667 | 12 (3/9) | 0 |
| base+textonly+v3 | 0.10 | 16 (9.9%) | 0 | 0.8075 / 0.7626 | 0.8075 / 0.7626 | +0.0000 | — (n=0) | 0.6875 | 0 (0/0) | 0 |
| base+textonly+v3 | 0.20 | 27 (16.8%) | 0 | 0.8075 / 0.7626 | 0.8075 / 0.7626 | +0.0000 | — (n=0) | 0.7037 | 0 (0/0) | 0 |
| base+textonly+v3 | 0.30 | 39 (24.2%) | 0 | 0.8075 / 0.7626 | 0.8075 / 0.7626 | +0.0000 | — (n=0) | 0.6667 | 0 (0/0) | 0 |

### deferred 切片的三分类构成(rate 0.30,test;原始 MultiHateClip 标注)

- deferred: {'Normal': 21, 'Hateful': 6, 'Offensive': 12}
- non-deferred: {'Normal': 91, 'Offensive': 24, 'Hateful': 7}

### 翻转案例(人工读样例;val 选定为不仲裁,以下取最佳非空配置 @ rate 0.30 仅供定性;共 10 个翻转)

- `cYQyH7hbNnw` gt=0 kNN=1 → MLLM **normal**(✔ 纠正,margin 0.009);证据:The video content does not express or endorse hatred, discrimination, demeaning stereotypes, or violence towards any group or its members. (cited: none);标题:This is what you call a Hand job....lol #handjob #shorts #Thailand #construction . 🎼.
- `lzKJ_AWegCc` gt=1 kNN=0 → MLLM **hateful**(✔ 纠正,margin 0.066);证据:The video depicts a stereotypical question about gender identity, which is harmful. (cited: none);标题:Boy or girl, maybe ladyboys? Pattaya, Walking street #shorts #thailand
- `OMSByZ-o3Ww` gt=0 kNN=0 → MLLM **hateful**(✘ 改错,margin 0.065);证据:The video contains explicit discussions about sexual abuse and inappropriate behaviors towards children. (cited: pzXX4LUKO_U);标题:07262022 signs your child might have/experienced , sexually abuse . I. this little girl just fucked around and stuck her finger up the statues as saying and tha

## MHC_zh (ZH)

**val 选定配置:不启用仲裁(保持 kNN,deferral 0%)**——所有 (prompt, rate) 候选在 val 上的 after-acc 都不超过 before 0.8718;门控的诚实决策是不花推理预算。

### val,prompt v1(配置选择依据;N=78)

| rate | defer n | before acc/F1 | after acc/F1 | 仲裁正确率(MLLM) | kNN在deferred上 | flips(好/坏) | 回退 |
|---|---|---|---|---|---|---|---|
| 0.10 | 8 (10.3%) | 0.8718 / 0.8558 | 0.8462 / 0.8329 | 0.3750 (n=8) | 0.6250 | 4 (1/3) | 0 |
| 0.20 | 16 (20.5%) | 0.8718 / 0.8558 | 0.8590 / 0.8480 | 0.6875 (n=16) | 0.7500 | 5 (2/3) | 0 |
| 0.30 | 23 (29.5%) | 0.8718 / 0.8558 | 0.8590 / 0.8501 | 0.6957 (n=23) | 0.7391 | 7 (3/4) | 0 |

### val,prompt v2(配置选择依据;N=78)

| rate | defer n | before acc/F1 | after acc/F1 | 仲裁正确率(MLLM) | kNN在deferred上 | flips(好/坏) | 回退 |
|---|---|---|---|---|---|---|---|
| 0.10 | 8 (10.3%) | 0.8718 / 0.8558 | 0.8462 / 0.8329 | 0.3750 (n=8) | 0.6250 | 4 (1/3) | 0 |
| 0.20 | 16 (20.5%) | 0.8718 / 0.8558 | 0.8590 / 0.8480 | 0.6875 (n=16) | 0.7500 | 5 (2/3) | 0 |
| 0.30 | 23 (29.5%) | 0.8718 / 0.8558 | 0.8590 / 0.8501 | 0.6957 (n=23) | 0.7391 | 7 (3/4) | 0 |

### val,prompt v3(配置选择依据;N=78)

| rate | defer n | before acc/F1 | after acc/F1 | 仲裁正确率(MLLM) | kNN在deferred上 | flips(好/坏) | 回退 |
|---|---|---|---|---|---|---|---|
| 0.10 | 8 (10.3%) | 0.8718 / 0.8558 | 0.8462 / 0.8329 | 0.3750 (n=8) | 0.6250 | 4 (1/3) | 0 |
| 0.20 | 16 (20.5%) | 0.8718 / 0.8558 | 0.8590 / 0.8480 | 0.6875 (n=16) | 0.7500 | 5 (2/3) | 0 |
| 0.30 | 23 (29.5%) | 0.8718 / 0.8558 | 0.8590 / 0.8501 | 0.6957 (n=23) | 0.7391 | 7 (3/4) | 0 |

### test(before/after,每变体×prompt 版)

| 变体 | rate | defer n (率) | MLLM calls | before acc/F1 | after acc/F1 | Δacc | 仲裁正确率 | kNN@deferred | flips(好/坏) | 回退 |
|---|---|---|---|---|---|---|---|---|---|---|
| base+frames+v1 | 0.10 | 15 (10.1%) | 15 | 0.8523 / 0.8270 | 0.8523 / 0.8310 | +0.0000 | 0.7333 (n=15) | 0.7333 | 4 (2/2) | 0 |
| base+frames+v1 | 0.20 | 28 (18.8%) | 28 | 0.8523 / 0.8270 | 0.8456 / 0.8278 | -0.0067 | 0.6429 (n=28) | 0.6786 | 9 (4/5) | 0 |
| base+frames+v1 | 0.30 | 42 (28.2%) | 42 | 0.8523 / 0.8270 | 0.8255 / 0.8080 | -0.0268 | 0.5750 (n=40) | 0.6905 | 12 (4/8) | 2 |
| base+frames+v2 | 0.10 | 15 (10.1%) | 15 | 0.8523 / 0.8270 | 0.8591 / 0.8377 | +0.0067 | 0.8000 (n=15) | 0.7333 | 3 (2/1) | 0 |
| base+frames+v2 | 0.20 | 28 (18.8%) | 28 | 0.8523 / 0.8270 | 0.8523 / 0.8344 | +0.0000 | 0.6786 (n=28) | 0.6786 | 8 (4/4) | 0 |
| base+frames+v2 | 0.30 | 42 (28.2%) | 42 | 0.8523 / 0.8270 | 0.8322 / 0.8145 | -0.0201 | 0.6000 (n=40) | 0.6905 | 11 (4/7) | 2 |
| base+frames+v3 | 0.10 | 15 (10.1%) | 15 | 0.8523 / 0.8270 | 0.8658 / 0.8446 | +0.0134 | 0.8667 (n=15) | 0.7333 | 2 (2/0) | 0 |
| base+frames+v3 | 0.20 | 28 (18.8%) | 28 | 0.8523 / 0.8270 | 0.8725 / 0.8548 | +0.0201 | 0.7857 (n=28) | 0.6786 | 5 (4/1) | 0 |
| base+frames+v3 | 0.30 | 42 (28.2%) | 42 | 0.8523 / 0.8270 | 0.8523 / 0.8344 | +0.0000 | 0.6905 (n=42) | 0.6905 | 8 (4/4) | 0 |
| base+textonly+v1 | 0.10 | 15 (10.1%) | 15 | 0.8523 / 0.8270 | 0.8523 / 0.8310 | +0.0000 | 0.7333 (n=15) | 0.7333 | 4 (2/2) | 0 |
| base+textonly+v1 | 0.20 | 28 (18.8%) | 28 | 0.8523 / 0.8270 | 0.8456 / 0.8278 | -0.0067 | 0.6296 (n=27) | 0.6786 | 9 (4/5) | 1 |
| base+textonly+v1 | 0.30 | 42 (28.2%) | 42 | 0.8523 / 0.8270 | 0.8255 / 0.8080 | -0.0268 | 0.5641 (n=39) | 0.6905 | 12 (4/8) | 3 |
| base+textonly+v2 | 0.10 | 15 (10.1%) | 15 | 0.8523 / 0.8270 | 0.8523 / 0.8310 | +0.0000 | 0.7333 (n=15) | 0.7333 | 4 (2/2) | 0 |
| base+textonly+v2 | 0.20 | 28 (18.8%) | 28 | 0.8523 / 0.8270 | 0.8523 / 0.8344 | +0.0000 | 0.6786 (n=28) | 0.6786 | 8 (4/4) | 0 |
| base+textonly+v2 | 0.30 | 42 (28.2%) | 42 | 0.8523 / 0.8270 | 0.8322 / 0.8145 | -0.0201 | 0.6000 (n=40) | 0.6905 | 11 (4/7) | 2 |
| base+textonly+v3 | 0.10 | 15 (10.1%) | 0 | 0.8523 / 0.8270 | 0.8523 / 0.8270 | +0.0000 | — (n=0) | 0.7333 | 0 (0/0) | 0 |
| base+textonly+v3 | 0.20 | 28 (18.8%) | 0 | 0.8523 / 0.8270 | 0.8523 / 0.8270 | +0.0000 | — (n=0) | 0.6786 | 0 (0/0) | 0 |
| base+textonly+v3 | 0.30 | 42 (28.2%) | 0 | 0.8523 / 0.8270 | 0.8523 / 0.8270 | +0.0000 | — (n=0) | 0.6905 | 0 (0/0) | 0 |

### deferred 切片的三分类构成(rate 0.30,test;原始 MultiHateClip 标注)

- deferred: {'Hateful': 9, 'Normal': 17, 'Offensive': 16}
- non-deferred: {'Normal': 87, 'Hateful': 8, 'Offensive': 12}

### 翻转案例(人工读样例;val 选定为不仲裁,以下取最佳非空配置 @ rate 0.30 仅供定性;共 8 个翻转)

- `BV1ch411L7VP` gt=1 kNN=0 → MLLM **hateful**(✔ 纠正,margin 0.028);证据:The video features derogatory terms towards Audrey Hepburn, including the phrase "我是天下第一淫妇！" which translates to "I am the world's number one prostitute!" (cited: none);标题:奥黛丽赫本：“我是天下第一淫妇！” . 🎼我就是天下第一夫。道光。🎼把每个黑暗的地方全部都照亮，庞大时光像男儿的胸膛。😊用无穷的力量如此的坚强。
- `BV1ia411m7Yy` gt=1 kNN=0 → MLLM **hateful**(✔ 纠正,margin 0.094);证据:The video features animated characters expressing disgust and frustration through exaggerated facial expressions and text overlays, including phrases like '恶心到怀疑人生' (so disgusted I question life itself). (cited: BV1ca4y1e7ud);标题:珍爱生命，远离公主病 . 🎼一想到你，我就我心心激心心到变我心到怀疑人生，我心到匆楚天际，我心到匆楚天际。😊
- `BV1Qk4y1g7PM` gt=0 kNN=0 → MLLM **hateful**(✘ 改错,margin 0.206);证据:The video uses derogatory terms like '娇妻' (weak wife) and '女司机' (female driver) towards women. (cited: BV1wj411h7ZN);标题:一些无意识的厌女词汇 . 🎼大哥大哥，你帮帮我好不好？哎呦，你可真是个大爷啊。大姐，你没事吧，阿妈，你有病吧。哦，你好有男友力好man啊，这一看就是女司机吧。知道你这种日系双容的精髓代表什么吗？娇妻能不能不要再服美艺了？大姐，你打扮的这么邋遢，你还是别出来丢人现眼了吧。我跟你说，我看她长相知道她不怎么样。😡🎼嗯，真是爱

## 成本核算素材(MLLM 调用)

| run | calls | 平均 wall s/call | 总 wall s |
|---|---|---|---|
| MHC:base:val:frames:v1 | 24 | 4.6 | 111 |
| MHC:base:test:frames:v1 | 39 | 4.4 | 172 |
| MHC:clean:test:frames:v1 | 39 | 4.4 | 172 |
| MHC:base:test:textonly:v1 | 39 | 1.7 | 66 |
| MHC_zh:base:val:frames:v1 | 23 | 4.4 | 100 |
| MHC_zh:base:test:frames:v1 | 42 | 4.5 | 189 |
| MHC_zh:base:test:textonly:v1 | 42 | 1.9 | 81 |
| MHC:base:val:frames:v2 | 24 | 4.7 | 112 |
| MHC:base:test:frames:v2 | 39 | 4.4 | 172 |
| MHC:clean:test:frames:v2 | 39 | 4.4 | 172 |
| MHC:base:test:textonly:v2 | 39 | 1.6 | 64 |
| MHC_zh:base:val:frames:v2 | 23 | 4.3 | 99 |
| MHC_zh:base:test:frames:v2 | 42 | 4.4 | 185 |
| MHC_zh:base:test:textonly:v2 | 42 | 1.9 | 80 |
| MHC:base:val:frames:v3 | 24 | 4.4 | 105 |
| MHC:base:test:frames:v3 | 39 | 4.1 | 162 |
| MHC_zh:base:val:frames:v3 | 23 | 4.3 | 98 |
| MHC_zh:base:test:frames:v3 | 42 | 4.3 | 181 |

## 结论

1. **门控本身是有效的(role-3 的正结果)**。margin 门把错误集中到 deferred 切片:EN test 在 30% 工作点上,24% 的样本(39/161)拿住了 42% 的 kNN 错误(13/31;切片内错误率 33% vs 切片外 15%);deferred 切片的三分类构成(Hateful 6 / Offensive 12 / Normal 21)相对 non-deferred(7/24/91)显著偏向 hate/offensive 边界——门控确实在"该花推理预算的地方"报警。**oracle 仲裁器**(deferred 全对)在 20%/30% 工作点即达 EN test acc 0.8571/0.8882,即门控留出了跨 0.85 的空间。
2. **主判定:三代 7B 仲裁器全部未过 val 门,该线终结**。val 上全部 (v1/v2/v3 × rate 10/20/30%) 候选的 after-acc 都低于 before(EN 最好 0.7750 < 0.7875;ZH 最好 0.8590 < 0.8718),按协议**两个数据集的 val 选定配置都是"不启用仲裁"**:EN 最终 test acc 维持 **0.8075,离 0.85 差 4.25 分**(memory-clean 记忆下 0.8199、差 3.01 分,系 role-2 编辑增益);ZH 维持 0.8523。
3. **v3(任务校准 LoRA 仲裁器)对两道量化线的答卷:不过线**。EN test@30% 的 deferred-acc 阶梯:v1 0.462 → v2 0.487 → **v3 0.615(24/39)< 0.667 打平线 << 0.846 跨 0.85 线**;10%/20% 同样低于同点 kNN(0.625 vs 0.688、0.593 vs 0.704)。仲裁器质量随"通用 prompt → 口径校准 prompt → 任务 SFT"单调上升,但 7B 量级的天花板落在打平线之下。**如实终结:7B 级仲裁器(含任务校准 LoRA)在该边界切片上不可用,选择性推理的 oracle 空间(0.857–0.888)留待更强模型。**
4. **失败模式与 v3 的部分修复**。v1/v2 是单向棘轮:487 个判决中 1→0 翻转 0 次、0→1 翻转 162 次,"normal" 判决 v1 仅 1/243、v2 仅 7/244——通用安全先验把边界队列全部推向 harmful。v3 打破了饱和(128 个判决中 normal 18 个、出现 4 次 1→0 翻转;EN test 修复 4 个错误、只引入 6 个新错),方向对了,幅度不够。
5. **诚实脚注:ZH v3 在 test 上有未被选中的正增益**。ZH test@10%/20% after 0.8658/0.8725(+0.0135/+0.0202),仲裁器在 deferred 上 0.867/0.786 > kNN 0.733/0.679,翻转 2好/0坏、4好/1坏——**但 ZH val 在全部三个点上都是负的**(0.8462–0.8590 < 0.8718),按协议不选、不作 claim。val(defer n=8–23)与 test(n=15–28)方向相反,正是诚实条款警示的小样本波动;它同时提示"任务校准 > prompt 工程"的方向在更强仲裁器上值得重试。
6. **帧对仲裁的贡献 ≈ 0(v1/v2 测得)**。EN v1 帧版与纯文本版 39/39 二值判决完全一致,v2 纯文本反而略好(test@30% 0.7702 vs 0.7640);纯文本 1.7s/call vs 帧版 4.4s/call(2.6×)。当前结论下两者都不部署;若未来部署,纯文本是更便宜且不更差的起点。
7. **成本核算素材(独立于仲裁器质量成立的 delta)**。选择性推理调用量 = 16/27/39 次(EN test 10/20/30%)vs always-on(MARS/HVGuard 式)161 次全量,省 76–90% MLLM 调用;门控纯 CPU(faiss,毫秒级)。接口可靠性:全部 18 个 run、624 条记录,可用判决率 98.6%(615/624;严格 JSON 612,一词裸答 fallback 仅 3,全在 ZH v3 test),解析失败一律回退 kNN。
8. **对总盘子的含义**:role-3 不能把 MHClip-EN 推过 0.85,当前最接近的仍是 role-2 记忆编辑后的 0.8199。已量化的复活条件:仲裁器在 EN deferred@30% 上做到 ≥0.667 才打平、≥0.846 才跨 0.85——建议留给 ≥72B 级或 API 级模型,而非继续在 7B 上做 prompt/LoRA 迭代。

(数字由 scripts/role3/analyze_role3.py 生成,结论人工撰写并逐条对表核对。原始 JSON:scripts/role3/out/role3_results.json;门控计划 scripts/role3/out/gate_*.json;仲裁输出 scripts/role3/out/arb_*.jsonl。SLURM job 12279 (v1, 19m56s) / 12288 (v2, 33m59s) / 12305 (v3, LoRA 仲裁器);两个 kNN ckpt 曾被 disk_guard 清扫、已从 B2 恢复。注意:重跑 analyze_role3.py 会重置本结论节。)
