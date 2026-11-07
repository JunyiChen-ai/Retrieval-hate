## 1. **Score**

**5/10 — not ready。**

核对了 180 个 trial、126 个消融结果及评测器输出。C1 有支持：三语料固定 8 次的 AP/ROC 分别为 **.702/.889、.691/.680、.278/.770**。主要缺口是 C3/C4 的机制归因，以及 C2/C5 的过强表述。

## 2. **Weaknesses**

### 1. baseline+VLM 对照不足以支撑 C4

**(a)** [baseline_plus_vlm.py:75](/home/jehc223/Retrieval-hate/experiments/20261101_r5_completion/baseline_plus_vlm.py:75) 每秒只保留覆盖它的**最短已问节点**，丢弃其他覆盖答案，再做单参数 logit 融合。r6 则联合使用全部答案，并在训练中利用全树答案监督。因此这里只证明优于一种信息压缩后的融合方式。

而且 [DeHate 原始结果](/home/jehc223/Retrieval-hate/runs/20261101_r6_length_table/baseline_plus_vlm/dehate_r6_k8.json) 的 baseline+VLM **ROC=.776760，高于 r6 的 .769821，三个 seed 都更高**。

**(b)** 损害 C4。  
**(c)** **EXPERIMENT**。  
**(d)** 最小实验：冻结官方 baseline，新增一个训练期学习的轻量 MIL fusion head；输入保留全部 8 个答案、区间位置/长度及 baseline features，不做测试后分数混合。训练时从全树缓存抽取节点，使用视频标签；test 使用 r6 实际问到的同一组答案。HateMM/MACIL-SD、HCS/Fed-WSVAD 与 DSANet、DeHate/Fed-WSVAD，各 seed **234/2025/3407**。若差距消失，撤回广义 C4；若优势保留，按指标报告，不能再声称 DeHate ROC 也领先。

### 2. “14 个臂证明每个部件必要”包含复合改动和错误归因

**(a)** [消融配置:25](/home/jehc223/Retrieval-hate/experiments/20261101_r6_length_table/launch/sbatch_arms.sh:25) 中，`c_label` 同时去掉答案训练和 node prior；`f_joint` 同时改变联合学习和 length term。`words` 更不是单独去掉 soft answer：它换成另一套 prompt、五类别、四档生成答案，**仍然输入视频帧和词级转录**。

已有结果也说明不能直接归因：HCS 在共同关闭 node prior 后，`no_node − c_label` 的 AP/ROC 为 **−.000664/−.004081**，没有显示答案训练的独立 pooled 收益。另据 [arms/table.json](/home/jehc223/Retrieval-hate/runs/20261101_r6_length_table/arms/table.json)，HCS 去 backbone 后 within 从 **.5630 升到 .6266**，“必要”必须说明针对哪个指标。

**(b)** 损害 C3。  
**(c)** **EXPERIMENT**。  
**(d)** 最小实验：用同一 `soft_both` 缓存，将 `p_yes` 按 `.5` 二值化；保持 prompt、输入、节点、训练流程及 shared-length table 结构一致，只比较 **hard 2-level 与 soft 8-level**。三语料、上述三 seed，完整训练并按原协议选 checkpoint。其余复合臂先用现有 `no_node`、`no_length` 作对应参照。只有隔离后的差值仍满足既定 ≥.01 标准，才能把收益归给该单独部件；否则报告联合贡献或交互作用。

### 3. 单层窗口对照混入了训练信息量和调用次数差异

**(a)** [train.py:212](/home/jehc223/Retrieval-hate/src/qtl/train.py:212) 的 `query_level` 同时过滤训练答案和测试候选。原始 `lvl16` 结果中，HateMM/DeHate 实际平均调用仅 **5.014/4.004**，完整方法为 **7.626/7.264**。因此下降可能来自较少监督或较少观测，不能直接证明跨尺度提问必要。此外，这些臂仍保留 tree posterior 和 node potentials。

**(b)** 损害 C3 的 tree/multiscale 主张。  
**(c)** **EXPERIMENT**。  
**(d)** 最小实验：固定三语料、三 seed 的 r6 checkpoint，分别限制候选为 `L=4/8/16` 的单层；对每个视频令两种 policy 都问  
`k_v=min(8, 该层可问节点数)`。全部使用现有缓存，比较跨层 EIG 与单层 EIG 的三项指标及配对差值。训练信息与实际调用完全相同时仍有稳定优势，才支持跨尺度查询贡献；优势消失则原消融不能承担该归因。

### 4. qmixE1 不能兑现部分设定的平均预算

**(a)** [policy.calibrate:130](/home/jehc223/Retrieval-hate/src/qtl/policy.py:130) 只搜索离散阈值，使 validation 平均调用 **≤B**，并不保证等于 B。[HateMM 原始曲线](/home/jehc223/Retrieval-hate/runs/20261101_r6_length_table/final_curves/val_selected/hatemm/summary.json) 中，seed3407 设置 **B=7**，validation/test 实际为 **4.716/4.841**；seed234 设置 **B=2**，test 为 **2.523**。前者连校准集都明显未达到目标，不能仅用分布变化解释。

**(b)** 损害 C5，也影响按名义预算比较效率。  
**(c)** **EXPERIMENT**。  
**(d)** 最小实验：三语料、三 seed、B=1–7，缓存重放并报告逐 seed 的目标预算、validation/test 实际预算。对离散跳变，用同一模型的相邻 stopping thresholds 按视频随机选择，概率仅由 validation 调用数确定；不混合预测。检查 validation 期望调用能否达到 B，并报告 test 偏差和区间。若仍明显偏离，只能称“validation 校准的目标预算”，曲线横轴必须用实际调用数。

### 5. C2 的逐点单调性已经被现有结果否定

**(a)** [HCS 曲线](/home/jehc223/Retrieval-hate/runs/20261101_r6_length_table/final_curves/val_selected/hateclipseg/summary.json) 固定 **3→4 次**，AP **.668802→.660177**、ROC **.659046→.649940**；HateMM 固定 **24→32 次**，ROC **.880303→.874840**。预算曲线也有回落，例如 DeHate **B=6→7**，AP **.227740→.225247**。

这些数字不证明总体增益不存在，但足以否定“更多调用总会更好”。

**(b)** 损害 C2。  
**(c)** **WRITING**。改为“总体改善、存在局部回落和收益饱和”，完整展示曲线；无需追加搜索来维持绝对单调表述。

## 3. **Experiment list**

1. **E1：保留全部答案的 frozen-baseline + learned MIL fusion 对照**，解决 C4。
2. **E2：同 prompt、同缓存的 hard-2 vs soft-8 消融**，并重算已有复合臂的对应差值。
3. **E3：同 checkpoint、逐视频同调用数的跨层 vs 单层查询比较**。
4. **E4：预算实现误差审计，以及离散阈值随机化重放**。

全部复用缓存，**新增 VLM 调用为 0**。E3/E4 仅需 CPU 重放；E2 按给定耗时约 **2 GPU-hours**，E1 只训练新增 fusion head，官方 baseline 保持冻结。