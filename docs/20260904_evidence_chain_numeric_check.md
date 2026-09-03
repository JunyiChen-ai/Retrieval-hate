# 证据链输出头(src/evidence_chain.py)真实规模数值检查

日期 2026-09-04。被检代码:main 上 commit 2026-09-04 01:31:38 "evidence_chain_net: review-1 fixes ..." 的 `src/evidence_chain.py`(接口 `chain(u, phi_f, gf, phi_c_rows, gc_rows, d, a, block_of_t, mask)`,返回 log_Z、log_Z0、log_rho、log_p_video、post、post_s1、logodds_s1)。参考:`experiments/20260904_evidence_chain_net/ref/evidence_chain_ref.py`(numpy float64,已与穷举一致,误差 1e-14 量级)。脚本在本会话 job tmp(`stability_chain.py`),未入库;随机种子 0,100 组用例。

## 1. 用例规格(主会话给定)
T ∈ [150, 300] 均匀随机,J = 4(块 = (t·4)//T),K = 30 细窗(窗 = (t·30)//T);phi_f 每片段 = ±2.4 / n_w(n_w = T/30,每窗一个符号);phi_c = ±1.4 每块;u ~ N(0, 3);gamma_f、gamma_c ~ U[0,1];d ~ U[.01, .99];a ~ U[.005, .2];mask 全 True;batch 1。

## 2. 结果(torch 对 numpy 参考的最大绝对误差,100 组)

| 量 | float64 | float32 |
|---|---|---|
| log_Z | 4.3e-14 | 5.7e-4 |
| log_Z0 | 2.1e-13 | 1.0e-5 |
| log_rho = log_Z0 − log_Z | 2.3e-13 | 5.8e-4 |
| log_p_video = log(1 − exp(log_rho)) | 4.9e-17 | 4.9e-20 |
| p_video | 0 | 0 |
| 后验三列 post[T,3] | 7.5e-13 | 1.6e-5 |
| logodds_s1(绝对 / 相对) | 1.3e-5 / 6.9e-7 | 8.3e-5 / 5.8e-5 |
| nan/inf 出现次数 | 0 | 0 |

- logodds_s1 在 float64 下的 1.3e-5 差异来自**参考侧**:参考用 log p − log1p(−p) 由后验反算,p 接近 1 时 1 − p 丢失精度;被检代码的对数域实现更准。相对误差 6.9e-7,视为一致。
- `a` 为 python float 与 `[B]` 张量两种传法输出逐位相同(20 组,差 0)。
- float32 可用:后验 1.6e-5、log-odds 相对 5.8e-5、log_Z 5.7e-4(T 300 累加)。

## 3. 一个建模层面的观察(不是 bug)

在上述势能量级下,P(y=1) 在 float64 里精确等于 1 的比例:u~N(0,3) 时 96/100(float32 100/100)。即便 u ≡ 0(无内容证据),仅靠 VLM 势与先验,log P(y=0) = log_rho 的中位数也是 −12.7(最小 −56.6,最大 −0.77);u~N(0,3) 时中位数 −92、最小 −201。含义:
- 正视频损失 −log P(y=1) 几乎恒为 0(1e-39 量级),负视频损失 −log_rho 为 10–200 nat,训练信号几乎全来自负视频与 d、a;u 的尺度必须从小开始(u~N(0,0.3) 时 P(y=1)=1 的比例降到 2/40),否则正视频对 u 无梯度。
- 随机 ± 符号让一半细窗触发,比真实负视频(K30 误报率约 .09)苛刻;真实数据下 log_rho 会小得多,但"负视频项主导"的方向不变。建议在预注册里写明 u 的初始化尺度与正负视频损失的相对量级,并在 seed 234 首个 trial 的日志里记录两类损失的均值。

## 4. 结论
review-1 版本的 src/evidence_chain.py 在真实规模、给定势能量级下与独立参考一致,float32/float64 均无 nan/inf,log 域输出正确。可进入下一步(规则 6 code review 由主会话自行派)。
