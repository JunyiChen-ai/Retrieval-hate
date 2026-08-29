# A-LINE PAUSE DECISION — lb_scgp_global 存档于 M0+M1-sealed(2026-07-13)

决策人:主循环(goal 指令授权自主推进);证据链全部独立产生,可复核。

## 判决

**PAUSE_A_LINE:不执行 M2/M3(节省 ~264 GPU-h),不建 v3 修复 lineage。** lb_scgp_global 存档于干净停点:M0(synth-KKT v4 PASS + realbank v2 GO)+ M1(缓存 v2 双数据集 CACHE_SEALED,seal job 13035 GO)。

## 证据链(三层独立)

1. **执行核验**(M1_CACHE_V2_EXECUTION_RECORD.md):封印程序完全合法,20/20 严格检查过,Merkle 双根独立复算一致。程序无瑕疵——这不是执行失败。
2. **信息含量复核**(M1_POST_SEAL_INFO_CONTENT_REVIEW.md,fresh 零上下文):91.3%/93.1% 的行为同一字面常数;覆盖 48/549 (8.74%) / 40/579 (6.91%);R=4 副本逐字节相同(复制零增益);有效 0.065/0.040 bit/视频;parse 失败机理 = 裸 except 吞掉的长输入 OOM(输入中位 94 vs 429 字符),属可修 infra bug 而非 prompt 问题。
3. **G0-cond 探针**(M1_G0COND_PROBE_RECORD.md,预注册判决规则,CLIP+Qwen 双表征):
   - 规则 (i) real-A 条件 codelength 增益 CI 排除 0(正向)——**四 cell 全 FAIL**(MHC 两 cell 整体为负:CLIP −0.00844 [−0.01510,−0.00291],Qwen −0.00454 [−0.00899,−0.00083]);
   - 规则 (ii) 投影 test Δacc > +0.040——**oracle 杀开关触发**:oracle@实测覆盖 ≤ +0.0044(探针)/ ≤ +0.0277(解析上界),低于线一个数量级;top-20 kNN 通道更弱(期望覆盖邻居 1.75/1.38 个);
   - v3 定价 arm:oracle@100% 覆盖 2/4 cell 过线,但 real-A covered-only 显示**已 parse 的证书本身即噪声质量**(全部 Δbits CI 含 0)→ 修覆盖率只传播零信息,v3 不可行。

## 解读

这是 D1 冗余诊断(REFLECTION_mllm_integration_failures.md)对 A 线自身的直接命中:证书 = 低带宽旁路信号,即使在其设计的表征侧作用点(全局 Gram 几何),条件信息也为零。lb_scgp_global 成为第 15 条负结果路线,但与前 14 条不同:它在 M2/M3 花费前被零 GPU gate 拦截——G0-cond 制度的第一次实战即回本。

## 后续

- GPU 全部转向 C 线(当前:SAV = C 线头号,预注册修订中;C1 QLoRA 已 KILL_CONFIRMED,settling DEV 读出进行中)。
- 科学遗留(不阻塞,待用户裁决):M-A synthetic-G0 语义、realbank b_struct is_science=false 覆盖、以及本 PAUSE 本身(用户可推翻;推翻需先反驳探针的 oracle 覆盖上界论证)。
- 论文叙事:lb_scgp_global 作为"预注册 + 零成本条件信息 gate 拦截"的方法学案例,证书缓存与全部 ceremony 文档已封存可审计。
