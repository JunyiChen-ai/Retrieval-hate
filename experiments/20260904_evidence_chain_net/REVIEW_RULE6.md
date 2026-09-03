# 规则 6 code review:evidence_chain_net(证据链骨干候选)

日期 2026-09-04。审查对象:main 快照(`/home/jehc223/.claude/jobs/cc29fdc9/tmp/main_snapshot/`,= main commit 2026-09-04 01:39:11,其中 `experiments/20260904_evidence_chain_net/train.py` 与 `ref/shape_check.py` 更新到 commit 2026-09-04 01:55:56 "evidence_chain_net: log positive/negative video-loss means separately; random-tensor shape/gradient check script")。审查者:fork 会话(fable)本体;独立审查 agent 五次因 API 529 中断未产出,改由会话本体逐项审查并写入本文件。范围:`src/evidence_chain.py`、`experiments/20260904_evidence_chain_net/{dataset,model,train,search}.py`、README 第 1/1.5/2 节、REVIEW_RULE4.md "必须修改" 九条。允许执行:import、自检、参考对拍、随机张量前向反向、各消融臂前向、损失函数各调一次;未做任何训练。

## 结论

**PASS(无 BLOCKER)**。两条非阻塞修改建议在开搜索前改掉更好(都只影响消融臂,不影响 full 与主表):(1) `topk_head` 与 `flat_coarse` 两个臂在训练截断(T > max_seqlen)时裁定势能尺度与评测全长不一致;(2) `no_vlm` 臂并未完全去掉 VLM 信息(密度 d_v 的裁定分布输入与块级 MIL 的软目标仍来自裁定),README 里该臂的名字与"no_verdict"对照要写清。

## BLOCKER

无。

## 非阻塞

1. **训练截断下两个消融臂的势能尺度**(`dataset.py:127–138` `fit_length`:`phi_f` 分块求和、其余行量取块首)。
   - `topk_head`(`model.py:172`):`score = u + gf·phi_f·n_w + ...`。全长时 `phi_f·n_w` = 窗 LLR;截断后 `phi_f` 是分块求和值而 `n_w` 仍是原窗行数,乘积 = 窗 LLR × (T/max_seqlen)。实测 T=300、max_seqlen=150:逐行裁定项均值 2.40 → 4.80(2 倍)。训练(长视频截断)与评测(全长)下 u 与裁定项的相对权重不同。修法:`fit_length` 里对 `n_w` 也按块求和(或在 `topk_head` 用 `phi_f · n_w_chunked`),或直接用未除以 n_w 的窗 LLR 列。
   - `flat_coarse`(`model.py:164`):`phi_c / n_j` 按行平铺,`n_j` 取块首(原块行数),截断后块内行数减半,块级证据总和 1.40 → 0.70。修法:`fit_length` 对 `n_j` 重算为截断后块内行数,或平铺时用 `1/块内有效行数` 由 mask 现算。
   - 链本身(full 及其它臂)不受影响:`phi_f` 分块求和保持窗证据总量(实测 24.000 → 24.000),`phi_c` 只在块末发射一次。
2. **`no_vlm` 臂的定义**(`model.py:160–161`):只把 φ_f、φ_c 置零;`d_v` 仍读裁定分布(`profile`),门仍读 bf/bc 列(乘 0 无效),块级 MIL 的软目标 `ph` 仍来自固定证据模型,`contrast_vlm_thresh` 不相关。README 第 2 节把它与修订 1 的 `no_verdict` 并列("两语料大幅下降 | no_verdict"),但两者不等价:本臂是"无 VLM 势能",不是"无 VLM"。建议 README 明写"φ ≡ 0,密度先验与块级 MIL 目标仍用裁定",或再加一个真正无裁定的臂(profile 置常数、block_weight 0)作对照。
3. `macilsd_encoder` 臂(`model.py:113`)调用 `AVCE_Model(f_a, f_v, None)` 不传 `valid_mask`,padding 行参与注意力(该臂以外的编码器用 `src_key_padding_mask=~mask`)。只影响该对照臂;写明即可。
4. `search.py:125` 预算规则"首 trial ≤ 1 h 则 20,否则 5"与 README"每(语料, seed)20 trial"不完全一致(与修订 1 的 search.py 相同)。README 补一句该回退规则。
5. 验证集选 checkpoint 用 `frame_eval_common.evaluate`(`train.py:218–222`),最终 val/test 数字用共享评测脚本(`train.py:232–238`,分支名 `score_chain`)。两者同一底层函数,与修订 1 做法相同;记录以备核对。
6. `model.py:148` 门读的是 `d_v.detach()`:门不能通过梯度改变密度头。与 README"门读裁定上下文 + d_v"一致,只是设计上 d_v 对门是常数;写进 README 常数表更清楚。
7. `model.py:97–98` `u_head` 是两层 MLP,README 1 节写"由一层 MLP 给出";数字不重要,措辞统一。

## REVIEW_RULE4 "必须修改" 九条核对

| # | 要求 | 落实 | 证据 |
|---|---|---|---|
| 1 | 正例视频不消失的片段级梯度项;"仅 1 − Z0/Z"降为消融臂 | 落实 | `train.py:112–130` `block_mil_loss` 作用于 `out["u"]`,软目标 `ph`(固定证据模型块后验,负例 0),权重 `abs(2p−1)`,`block_weight` 常数 1(`DEFAULTS`);`train.py:333` `no_block_mil` 臂把权重置 0;README 1 节训练目标第 2 项引用 `runs/20260904_evidence_chain_net/analysis/rho_and_density_profile.json`(本机存在,2026-09-04 01:24) |
| 2 | 切换率换算到片段步;断言 a ≤ 1 | 落实 | `model.py:122–124` `a_step = 1 − (1 − a)^{K/L}`,L = mask 有效步数;`src/evidence_chain.py:58` `assert (a > 0) & (a ≤ 1)`;`model.py:82` `assert 0 < a_window ≤ 1`;`dataset.py:61` a = A01 + A10 |
| 3 | 对比保留 CMAL 跨模态配对、只换选段器;权重爬升写明;空集处理 | 落实 | `train.py:147–179`:视觉 top-k 均值作 query、音文 top-k 正键、音文 bottom-k + 负例视频行为负键,再反向;选段器 `posterior` / `self_topk` / `vlm_thresh`;`train.py:345` `lam_c = min(1, epoch/10)`;k = max(1, ⌈t/16⌉) 保证非空,批内无正例时返回 0(`train.py:177`) |
| 4 | d_v 只读裁定分布或限制范围;test 诊断 | 落实 | `model.py:99–101,137–140` 密度头输入 `profile`(11 维裁定分布),`density_content` 臂才拼内容;`D_LO, D_HI = .01, .99`;`train.py:241–260` 报门均值按格子、d_v 与 GT 密度相关、饱和比例(只在 test 评测后) |
| 5 | 输出分数对数域 log-odds | 落实 | `src/evidence_chain.py:145` `logodds = log_post[1] − logaddexp(log_post[0], log_post[2])`;`model.py:189` `score = logodds_s1`;`train.py:205` 五 crop 平均后重采样 |
| 6 | 消融表补 no_block_mil / topk_head 含块级 MIL / no_text 两语料 / u ≡ 0 报 within / P2 输入拆分 | 落实 | `model.py:38–42` 17 个臂;`topk_head` 臂训练仍带 `block_mil_loss`(`train.py:333` 只在 `no_block_mil` 置 0);`no_text` 在模型内置零文本列(`model.py:110–111`),两语料同一开关;within 由评测器对所有臂统一输出 |
| 7 | 数字落盘并引用路径 | 落实 | README 0 节引用 `runs/20260904_evidence_chain_net/analysis/rho_and_density_profile.json`(本机存在);HateMM no_cmal/no_text/no_ema 数字已写入 0 节 D 行 |
| 8 | 理论近似第三条(1/n_w 分摊) | 落实 | README 2 节末"三处理论近似"第 (3) 条;`dataset.py:116` `phi_f = llr_f[w] / n_w[w]` |
| 9 | 超参节按项列固定常数与来源 | 落实 | README 1 节常数表(13 行,含来源);搜索空间只剩 lr / dropout / max_seqlen(`search.py:33–38`) |

## 运行的命令与输出摘要(全部在快照根目录,随机张量,无训练)

1. `python -c "import evidence_chain, dataset, model, train, search"`:通过;`ABLATIONS` 17 个;`frame_eval_common` 从 `scripts/duplex/` 导入成功。
2. `python src/evidence_chain.py`:`evidence_chain self-check: True`(20 组穷举 + padding 不变性)。
3. `python experiments/20260904_evidence_chain_net/ref/check_ref.py --trials 200 --seed 3`(numpy 参考对穷举):失败 0,最大误差 4e-15,PASS。此前另有 torch 对 numpy 参考的对拍:小规模 400 组误差 9e-15,真实规模 T 150–300、J 4 误差 1e-13,float32 可用(`docs/20260904_evidence_chain_numeric_check.md`)。
4. `python experiments/20260904_evidence_chain_net/ref/shape_check.py`(B=3,T=40,padding 行 w=K、j=J,标签 [1,0,1]):13 个模型臂前向 + `video_loss + block_mil_loss + contrast_loss` 反向,梯度全部有限;`indep` 臂 a_step = 1;`self_topk` 5.27、`vlm_thresh` 3.44;全负 batch 对比损失 0、块损失有值;每视频只 1 个有效行时三种损失有限;`fit_length` 长序列 phi_f 求和保持(46.6679 → 46.6679)、j 单调、短序列 padding j=4、w=30。
5. `review_extra_checks.py`(本会话):(a) 截断尺度:`topk_head` 裁定项 2.40 → 4.80,链 `phi_f` 总和 24.000 → 24.000;`flat_coarse` 每块粗证据 1.40 → 0.70;(b) `eval_batch` 展开键 = {bc, bf, bfn, bfp, f_a, f_v, j, mask, n_j, n_w, ph, phi_c, phi_f, profile, w},13 个模型臂在 B=5 crop、T=37 的评测 batch 上前向全部有限;(c) 单行块的 `block_mil_loss` 有限;对比损失在有效行数 t = 1..5 下有限。

## 焦点项 (a)–(j)

| 项 | 结论 | 证据 |
|---|---|---|
| (a) 每个 `torch.topk` 的 k ≤ 有效行数 | PASS | `train.py:119–123` 块内 n 由 `sel = (j==b) & m` 计数,k = max(1, ⌈n/16⌉) ≤ n(n ≥ 1);`train.py:170–172` k = max(1, ⌈t/16⌉) ≤ t,`_select_topk` 用 ±1e9 掩蔽 padding,k ≤ t 保证只取有效行;`model.py:176–178` `score[i, :t]` 依赖 padding 在末尾(`fit_length` 保证)。实测 t=1..5、单行块均有限 |
| (b) padding 行被排除 | PASS | 链:`evidence_chain.py:84` 恒等转移、`:96` 零发射、`:124` log_Z0 掩蔽,自检 padding 不变性通过;编码器 `model.py:119` `src_key_padding_mask=~mask`;门 `group_mean` 用 mask 计数(`model.py:53–63`,padding 组返回 0);块损失/对比只用 `mask` 行;`masked_mean` 用 mask。例外:`macilsd_encoder` 臂不传 valid_mask(非阻塞 3) |
| (c) `eval_batch` 展开全部键 | PASS | `train.py:183–191` 展开 f_a、mask、profile、ROW_KEYS(9)、IDX_KEYS(2),f_v 本身按 crop 成 batch;模型读取的键集合 ⊆ 展开集合,实测 13 臂前向通过 |
| (d) a_step 训练/评测换算 | PASS | 同一函数 `model.py:122–124`,L = 有效步数:训练截断时 L = max_seqlen,评测全长时 L = T;两者都满足 (1−a_step)^L = (1−a)^K(期望切换数守恒),与 README 1 节定义一致;`indep` 臂 a_step = 1 即平稳独立 |
| (e) 无 test/val 帧标签进训练或选择 | PASS | 训练只用视频标签(`batch["label"]`);证据模型 EM 只用 train ids(`train.py:92–95,299`);val 帧 GT 只用于 checkpoint 选择(`train.py:368–370`,规则允许);test GT 只在训练结束后评测与诊断(`train.py:392–396`);缺裁定即中止(`train.py:314–317`) |
| (f) 不向 data/ 写 | PASS | `train.py` 所有写入在 `out_dir`(runs/ 下);`search.py` 在 `out_root`;`dataset.py` 无写;`model.py` 无写 |
| (g) 共享评测器、分支前缀、输出目录 | PASS | `train.py:232–238` 调 `scripts/reproduction_baselines/eval_baseline_scores.py`,分支 `score_chain`;val 选择用同一底层 `frame_eval_common.evaluate`(非阻塞 5);输出 `runs/20260904_evidence_chain_net/<corpus>/seed<seed>/trial<n>/` |
| (h) 窗/块映射一致 | PASS | `dataset.py:101–102` w 由 `vlm_verdict.verdict_rows`(时间中点)给出,j = `verdict_hmm._block_map(K,J)[w]`;`evidence_chain.block_layout` 只依赖 j 非递减(中点映射单调,分块取块首后仍单调,实测 `j monotone True`);padding j = J 使最后有效行成为块末(粗发射位置正确) |
| (i) 梯度有限、分数为对数域后验 | PASS | shape_check 13 臂梯度全部有限;`model.py:189` score = `logodds_s1`(对数域),`train.py:205` 五 crop 平均 |
| (j) run.log 首行主机名、config 快照、代码行 | PASS | `train.py:277–283` 首行 `host ... | code: <git log -1>`,`config.json` 含 hparams/host/code;`summary.json` 含 host/code;`search.py:62` 搜索日志首行主机名 |

## 小结

代码与 README 第 1/1.5/2 节一致,REVIEW_RULE4 九条"必须修改"逐条有对应实现;链头与独立 numpy 参考在穷举和真实规模下一致,13 个模型臂与 4 个损失臂在随机张量上前向反向均有限,padding、top-k、评测展开、标签使用、写入位置、评测器调用均符合规则。无 BLOCKER。两条建议在开搜索前顺手改(只影响 `topk_head`/`flat_coarse` 两个消融臂的训练截断尺度,以及 `no_vlm` 臂在 README 里的定义措辞),其余五条为记录性质。规则 6 的一次 code review 到此完成。
