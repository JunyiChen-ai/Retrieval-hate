# REVIEW SUMMARY — Codex gpt-5.6-sol（ultra）两轮评审（trace：`.aris/traces/research-review/20260905_run01/`）

## 第 1 轮（原始组合提案 C7 + C1 + C13）：3/10 Reject，置信 4/5
决定性问题：(a) 条件发射把邻窗细裁定、块内计数、粗裁定同时放进两级发射，是循环的伪似然，不是合法生成式 IOHMM，不能称 EM 单调；(b) s_v 的比例损失作用在 sigmoid(s_v) 而非最终逐秒概率均值，且 mean P(s_t) 既是 s_v 输入又是目标（target-copy）；(c) 先中心化再加先验使 r_t 不再零均值；(d) "s_v 不破坏 within"只对固定排序器的推理成立；(e) 块级 MIL 监督中心化后的相对 logit 语义冲突；(f) 缺精确重参数化对照（s_v = mean a_t）；(g) 三开关不独立（T 嵌套 D，D=0/T=1 未定义）；(h) 文本开关门槛低于 no_text 臂（.869）；(i) 预注册门槛非对称、低于噪声（HateMM 差值 SEM ≈ .0155，+.020 只有 1.3 SEM；HCS +.006 等于已测选择偏差）；(j) CPU 预检只要求复制 K4-only；(k) 可识别性 claim 过宽；(l) test 驱动的 Optuna/剪枝/预检使确认性评测失效（项目裁定，记录不改）；(m) 预期效应被高估，端到端增量可能在噪声内；(n) novelty 为应用级。
要求的最小实验包（按 acceptance lift / GPU-h）：CPU 标注模型门 → 精确候选 1 对照与精确重参数化对照 → `no-s+uncentred` / direct-logit q / 删 q 输入 / no-text 四臂 → K4-only 网络先验与 4-cell 偏移 → 只对存活 full 做一次搜索 → 六个可定义配置 → 追加 3 个配对 seed → MSL/LLP/count 竞争头 → MIL 精调最后。约 16–24 GPU-h。

## 第 2 轮（修订提案）：5/10 Weak Reject，置信 4/5
接受：coarse-first 有向分解合法；先验后中心化；最终概率比例损失；去掉 q_v 输入；总分块级 MIL；精确重参数化对照。方法已可实现。
未解决：(1) q_v 未被证明是比例估计——须用 OOF、inference-mode 后验在有时间 GT 的数据上比较条件 / global / K4-only 的每视频比例 MAE、偏差、校准，并与同 head 的 λ_prop = 0 比较（**最能提分的单一改变**，通过可到 6/10，不通过则删除 scale/proportion thesis）；(2) 训练期网络输入的后验须 OOF、label-free；(3) "ordinary EM"应写成"固定 θ_0 后对其余参数 EM"；(4) 加 `no_b4` / `no_bprev` 隔离机制；(5) CPU 门须有预定最小优势且同时胜过 4-cell lookup；(6) detach 只切断特征分支梯度，∂L/∂r_t 仍依赖 s_v，只能 claim 推理时顺序保持；(7) 模块 2 捆绑三处改动，加 λ_prop = 0 与 block-relocation-only 臂；(8) 配置表缺 101，须做完整 2^3；(9) 预期效应与采用门槛算术冲突：按作者预期三个开关都可能被删除。
以上 (1)–(8) 已全部写入 `FINAL_PROPOSAL.md`；(9) 作为主风险如实记录。
