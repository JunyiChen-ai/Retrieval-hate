# Reopening Evidence Audit

截至 2026-08-31。依据下表所列`runs/` artifacts与`docs/PROCESS_REVIEW_2026-08-31.md`。本表只审计已有证据，不提出method candidate。

历史表按旧 reopening gate 产生；2026-08-31 流程纠正后不再把全部 controls 与 carrier strata 作为 raw statistic 的前置硬门。当前最低 premise 只要求局部变化存在、非纯 position/broadcast、HMM/HCS 同向；本表中的旧 `FAIL` 只保留当时结果，不自动否定可学习的 end-to-end mechanism。

| Statistic/source | HMM/HCS已有证据 | Gate状态 | 裁定 |
|---|---|---|---|
| Qwen3-VL pointwise windows | `runs/20260831_qwen3_test_teacher_diagnostic/formal/metrics.json`：within `.561760/.539628`，均低于SOTA，零生成故障 | 性能前提失败；不是新local observation | 关闭 |
| OmniVTG frozen interval | `runs/20260831_omnivtg_grounder_diagnostic/formal/metrics.json`：raw within HMM/HCS `.626380/.539010`；`archive/experiments/20260831_omnivtg_reopening_controls/PRE_RUN_REVIEW.md` | target-preserving 8秒block control在HMM `hate_video_329`只有1 block，而raw成功，预注册integrity gate解析失败；未运行GPU | 当前statistic无法完成统一position control，旧teacher STOP保持 |
| Word-aligned ASR lexical timing | `runs/20260831_word_aligned_lexical_premise/hateclipseg_test/metrics.json`：HCS formal producer前8视频仅3个通过strict timestamp validation | HCS coverage失败 | 关闭当前producer |
| Raw video-label lexical locality | `runs/20260831_video_label_lexical_locality/premise/metrics.json`：raw lexical-minus-shift HMM/HCS `+.127533/+.021501` | raw statistic满足旧time-shuffle门；但`archive/experiments/20260831_lexical_reopening_controls/PRE_RUN_REVIEW.md`解析证明ASR-carrier-absent秒的TF-IDF向量恒为零、score恒为intercept、within必为`.500` | Rule 14 carrier coverage失败，关闭当前lexical statistic |
| Lexical timing加入strong base | `runs/20260831_pair_alignment_attribution/main/metrics.json`：aligned-minus-shift mean within HMM/HCS `+.102665/+.036685` | 只证明corpus-specific base+weight下timing有用；HCS部分shift仍all-SOTA，且blend不能修复raw statistic的carrier-absence失败 | 不足以重开 |
| Content change/boundary | `runs/20260831_powa_test_error_taxonomy/analysis.json`：GT boundary macro AUC HMM/HCS `.486/.528` | 跨语料方向/强度不成立 | 关闭generic change |
| Train-label instance density | `runs/20260831_instance_density_test_probe/{hatemm,hateclipseg}/metrics.json`：HMM concat within`.60843`，HCS concat`.49753`、best visual`.51816` | HCS失败 | 关闭 |
| ImageBind multimodal ceiling | `runs/20260831_imagebind_feature_ceiling/analysis.json`：HMM提升，HCS三指标全降 | 双语料统一性失败 | 关闭当前feature family |
| Semantic recurrence/persistence | `runs/20260831_semantic_neighbor_probe/analysis.json` upper bound双语料有提升，但直接graph propagation/smoothing/transport属于calibration；`runs/20260831_policy_recurrence_probe/analysis.json`中typed gate不load-bearing | anti-pattern且机制control失败 | 关闭方法化 |
| DSANet semantic-alignment raw branch | `runs/20260831_dsanet_alignment_reopening/main/metrics.json`：HMM/HCS raw within `.573931/.568621`，time-shuffle margin `+.071238/+.064966`，mean-repeated margin `+.073931/+.068621` | 两语料carrier-strata均失败；HMM reversal Spearman失败；HCS position-only margin `-.032212` | `KEEP_CANDIDATE_FREEZE`，关闭该statistic |
| Frozen CLIP policy direction | `runs/20260831_clip_policy_reopening/main/metrics.json`：HMM/HCS raw within `.526238/.517859`，time-shift margin `+.028386/+.015649` | HMM carrier-strata失败；HCS time-shift、mean、position与carrier均失败 | `KEEP_CANDIDATE_FREEZE`，关闭global frame prompt statistic |
| Native AV sync/TCC/event | `archive/experiments/20260831_post_scale_candidate_scout/`未找到统一覆盖HMM/HCS的statistic；TCC允许identity+position解 | 没有formal双语料time-shuffle+三controls证据 | 暂停 |

## 结论

旧完整 reopening gate 已撤销：它把 observation evidence 与 method-level attribution 混在一起，并产生不可达 carrier 门槛与 premise churn。Frozen CLIP 与 DSANet 结果继续约束同一信息源的直接重跑；raw lexical 的 carrier-absence 常数反例继续成立，但这些证据不再要求任意新 raw scalar 先独立承担完整定位。

下一步不是第三个 raw-statistic sweep，而是先从现有 test error artifacts固定 HMM/HCS 共同 failure mode；最低 premise 只检验局部变化、非 position/broadcast 与双语料同向。通过后立即 novelty 三门与最小 end-to-end 方法；方法 test 后才做完整 time-shuffle、position、carrier attribution controls。
