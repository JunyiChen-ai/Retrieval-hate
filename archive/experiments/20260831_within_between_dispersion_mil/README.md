# 淘汰：within-between fixed-effects dispersion MIL

截至 2026-08-31。本候选在双独立 novelty/identifiability review 阶段淘汰，评分
`3.1/10` 与 `3.6/10`；未实现、未训练、未生成 prediction，也未获准 premise。

## 候选

拟从 panel-data within estimator / Mundlak decomposition 适配：每模态 frame feature
严格拆成视频均值 `b_v` 与偏差 `w_vt=x_vt-b_v`；between head 预测 video label，
within head 输出再逐视频中心化的 `d_vt`。local bag evidence 原拟为
`q_v=logsumexp(d_v)-log T`，正包鼓励 concentrated excess、负包鼓励 flat；5-fold
OOF lexical pairs 只用于确定 `d` 的方向。最终单模型 raw logit 为
`s_vt=a_v+rho*d_vt`。

## 裁定

精确 within-between/Mundlak 公式未检出用于 hateful-video localization，窄来源门
通过；第三道 non-trivial mechanism 门失败：

- Mundlak identification 依赖逐期 outcome/treatment 与线性加性模型；这里只有一个
  video bag label。feature/logit centering 只去掉加性常数，不识别哪一秒 hateful。
- 当 `mean(d)=0` 时，Jensen 不等式给出 `q>=0`，小扰动下 `q` 近似
  `Var(d)/2`。它监督的只是 temporal dispersion，不是 signed hate evidence。
- 正视频可在任意固定片头、scene cut 或最大 feature-norm 秒制造单点 spike，使 `q`
  无界增大；`[c,-c]` 与 `[-c,c]` 具有同一 evidence，方向仍不识别。
- centered feature 仍保留视频尺度、协方差、剪辑节奏、缺失模式与 identity；模型可先
  识别 positive video，再在无关位置输出 spike。
- `a_v` 对同视频是常数，只改善 pooled；任意 `rho>0` 不改变 within 排序，只调跨视频
  尺度，因而是 pooled calibration。项目 V20 与 context-quotient span marginal 也已
  占用更接近的 global constant + zero-mean residual 结构并失败。

因此不批准 premise。若 `logmeanexp` 已经包含归一化后再减 `log T`，还会额外引入
直接 duration shortcut；正式定义必须避免这种歧义，但修正它不能恢复机制门。

## 来源

- Mundlak, *On the Pooling of Time Series and Cross Section Data*, Econometrica 1978.
- Tian et al., [RTFM](https://openaccess.thecvf.com/content/ICCV2021/html/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.html), ICCV 2021.
- Shou et al., [AutoLoc](https://openaccess.thecvf.com/content_ECCV_2018/html/Zheng_Shou_AutoLoc_Weakly-supervised_Temporal_ECCV_2018_paper.html), ECCV 2018.
- Ren et al., [P-MIL](https://openaccess.thecvf.com/content/CVPR2023/papers/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.pdf), CVPR 2023.
