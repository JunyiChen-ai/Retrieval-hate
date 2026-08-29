# NOVELTY_SCOUT — Weakly-supervised hateful video localization, 4 candidate mechanisms
Date: 2026-08-30. Scope: arXiv / CVPR / AAAI / ACL / ACM MM, 2023–2026. Stance: adversarial (looking for reasons candidates are NOT novel).
Task frame: train with video-level labels only; output per-second hate score; metric = within-video ROC/AP on hateful videos.

---

## C1. Dense MLLM ordering distillation (Qwen2-VL scores every 16–32s window; student trained with MIL + within-video pairwise ranking distillation; VLM absent at inference)

### Closest papers
1. **LAVAD — "Harnessing Large Language Models for Training-free Video Anomaly Detection"**, CVPR 2024, arXiv:2404.01014. Captioner + LLM produce per-frame anomaly scores training-free; no student is ever trained; LLM needed at inference.
2. **MLLM4WTAL — "Weakly Supervised Temporal Action Localization via Dual-Prior Collaborative Learning Guided by Multimodal Large Language Models"**, arXiv:2411.08466 (v2 2025-06, under review). MLLM priors (key-semantic matching + masked reconstruction) guide a WTAL head; **MLLM is training-only, absent at inference** — same deployment pattern as C1. But the MLLM does NOT densely score windows; guidance is via text-description similarity matrices, and losses are BCE/reconstruction/MSE, not ranking distillation.
3. **Ju et al. — "Distilling Vision-Language Pre-training to Collaborate with Weakly-Supervised Temporal Action Localization"**, CVPR 2023, arXiv:2212.09335. CLIP (not MLLM) distilled into a WTAL model via alternating foreground/background pseudo-labels; teacher-free inference. Establishes "distill VLP into weak temporal localizer" as a known pattern.
4. **DAKD — "Distilling Aggregated Knowledge for Weakly-Supervised Video Anomaly Detection"**, arXiv:2406.02831; and **"Weakly Supervised Multimodal Video Anomaly Detection Based on Knowledge Distillation"** (Springer LNCS 2025, 10.1007/978-3-032-03215-7_7). Teacher→student distillation inside WSVAD, with soft-label / logits distillation. Teachers are aggregated visual backbones, not an MLLM; losses are score/representation matching, not order-only.
5. **Holmes-VAU**, CVPR 2025 Highlight, arXiv:2412.06171 (and Holmes-VAD arXiv:2406.12235). LLM-assisted dense multi-granularity annotation (HIVAU-70k) then fine-tunes an MLLM; MLLM present at inference. **VERA**, CVPR 2025, arXiv:2412.01095: verbalized learning of guiding questions; VLM present at inference.
6. **Training-Free VLM-Based Pseudo Label Generation for VAD**, IEEE (TCSVT-class) 2025, doi 11015429: CLIP-alignment pseudo-labels sharpen frame-level WSVAD labels — pseudo-label idea taken for CLIP-level teachers.
7. **PRD — "Harnessing Pairwise Ranking Prompting Through Sample-Efficient Ranking Distillation"**, arXiv:2507.04820 (text retrieval): distilling pairwise LLM ranking judgments into a pointwise student — exactly the loss family of C1, but in IR, not video.
8. Hate-domain teacher already exists: **LELA — "Towards Training-free Multimodal Hate Localisation with Large Language Models"** (Sun, Yang, Jiao, Fu), arXiv:2602.09637, Feb 2026. Five modality captions (BLIP-2, EasyOCR, Whisper, music captions, PDVC) + multi-stage LLM prompting → per-frame hate scores on HateMM & MultiHateClip-EN, frame-level AUC/AP. **No student, no distillation, LLM at inference.**

### Overlap analysis
Every ingredient exists separately: MLLM-as-training-only-teacher for weak temporal localization (MLLM4WTAL), VLP→student distillation for WTAL (Ju CVPR'23), teacher→student KD inside WSVAD (DAKD, Springer'25), LLM dense frame scoring (LAVAD, LELA), pairwise ranking distillation from LLM judgments (PRD). A hostile reviewer will assemble exactly this list.

### What survives
(a) **Dense per-window MLLM scoring of every training video** as the teacher signal (existing works use text-similarity priors or CLIP alignment, not dense generative-VLM window scores); (b) **within-video ORDER-only distillation** (pairwise ranking, explicitly discarding absolute scores for miscalibration robustness) — not found anywhere in VAD/WTAL; the closest (PRD) is text retrieval; (c) the hate-localization application — no one has distilled a LELA-style teacher into a cheap inference-time-free student; (d) direct fit to the within-video ROC/AP metric: a within-video ranking loss optimizes the actual evaluation quantity, which none of the above do. Also a clean contrast with in-house POWA sparse teacher (≤2 chunks) → dense ordering is a real internal delta.
Scoop risk: LELA authors' natural next paper is "distill LELA". TANDEM (arXiv:2601.11178) already RL-tunes Qwen2.5-VL for hate temporal grounding but keeps the 7B model at inference and needs SFT — C1's cheap-student story is distinct.

### Verdict: **open-with-differentiation** (mechanically crowded; the within-video order-distillation loss + teacher-free hate localizer is the defensible core; must cite and empirically beat/compare LELA as teacher-only baseline).

---

## C2. Per-second OCR text channel as fourth modality in weak MIL

### Closest papers
1. **LELA**, arXiv:2602.09637 (above): **already uses OCR (EasyOCR) as an explicit modality for frame-level hate localization** on HateMM/MHC, taking per-frame max over modality scores. The "OCR modality for hate localization" flag is planted.
2. **SafeLens: Segment-Level Hate Speech Detection in Online Videos**, AAAI-26 demo (Wang, Raharja, Hu, Lee, SUTD; AAAI proceedings p.41712). Per-segment EasyOCR every 3–5s + Whisper + Qwen2.5-VL frames, fused by LoRA-tuned Llama3-8B trained on **HateClipSeg segment labels**; outputs per-segment hate labels with OCR modality attribution. Segment-supervised, not weak — but OCR-for-segment-level-hate is published.
3. **MM-HSD: Multi-Modal Hate Speech Detection in Videos**, arXiv:2508.20546. Uses on-screen text (OCR) as the query modality in cross-modal attention on HateMM; shows OCR materially improves video-level detection (M-F1 0.874).
4. **HateMM** (arXiv:2305.03915) ecosystem & "Enhanced Multimodal Hate Video Detection via Channel-wise..." arXiv:2505.12051 — multimodal hate classification lines that C2 would extend.
5. Our own lab evidence (Gate-C reanalysis: on_screen_text OR 2.29) motivates it, but motivation ≠ novelty.

### Overlap analysis
"OCR helps hateful video detection" is now published three ways: training-free localization (LELA), segment-supervised system (SafeLens), video-level weak classification (MM-HSD). What is literally unpublished: OCR embeddings as a per-second input stream to a **trained weak-MIL localizer**. That residue is an engineering delta a reviewer will call incremental ("add a modality to MultiHateLoc").

### Verdict: **crowded**. Not viable as a headline contribution; keep it as an input-channel ablation inside C1/C3 (where it can still earn a table row and echoes MM-HSD/LELA evidence).

---

## C3. Two-sided MIL: mine pseudo-benign seconds inside hateful videos via cross-video kNN to benign-video frames, supervise as negatives (fix rank inversion in high-positive-fraction videos)

### Closest papers
1. **UR-DMU**, AAAI 2023, arXiv:2302.05160. Normal + abnormal memory banks; models normality inside the same architecture; margin between representations. Normality modeling ≠ explicit per-second negative supervision inside positive bags, but same spirit.
2. **BN-WVAD**, arXiv:2311.15367. Divergence-from-batch-mean statistic selects likely-abnormal snippets (and implicitly likely-normal ones) inside abnormal videos — self-statistic-based two-sided evidence.
3. **MIST**, CVPR 2021, arXiv:2104.01633, and **"Exploiting Completeness and Uncertainty of Pseudo Labels for WSVAD"**, CVPR 2023, arXiv:2212.04090. Self-training pipelines that assign snippet-level pseudo labels in abnormal videos — including pseudo-NORMAL snippets — from the model's own scores. This is the direct prior for "supervise seconds inside hateful videos as negatives"; the generic idea is taken.
4. **TPWNG — "Text Prompt with Normality Guidance for WSVAD"**, CVPR 2024. Normality text prompts guide pseudo-label generation, i.e., a normality-side signal for in-video labels.
5. WTAL background-modeling line: **ProCL** (arXiv:2206.11011, complementary learning), **ACM-BANets** (MM 2020), Local-Global Background Modeling (arXiv:2106.11811) — supervising background (negative) frames inside positive videos is a mature theme in action localization.
6. **GlanceVAD**, ICME 2025 Oral, arXiv:2403.06154 — cheap extra supervision (one glance frame) with Gaussian splatting; adjacent labeling-paradigm competitor.

### Overlap analysis
"Also supervise some seconds in positive videos as negatives" = known (MIST-family self-training, background modeling). "Model normality" = known (UR-DMU, TPWNG, BN-WVAD). What I could NOT find anywhere: (a) negative mining evidence taken from **cross-video nearest neighbors into the benign corpus** (all existing selection uses the model's own scores, batch statistics, or text prompts — circular under exactly the failure mode C3 targets, since a rank-inverted model self-labels the hateful middle as normal); (b) explicit motivation from the **high-positive-fraction regime** (hate videos are often mostly-hateful, violating the sparsity assumption baked into MIL top-k; surveys note anomalies ~20% of frames in VAD benchmarks, and long-anomaly failure is acknowledged but unsolved); (c) the within-video rank-inversion diagnosis (intros/outros scoring highest) as the stated target. Note the diagnosis-to-fix chain is non-circular here: the evidence for a second being benign (feature-space proximity to benign videos) is independent of the MIL model's own score.

### Verdict: **open-with-differentiation** (narrow but real: cross-video-evidence pseudo-negatives + high-positive-fraction framing; must position hard against MIST/CU-pseudo-labels/BN-WVAD/UR-DMU or reviewers will collapse it into them).

---

## C4. Transductive test-time kNN propagation of MIL scores over all test-video seconds

### Closest papers
1. **LAVAD**, CVPR 2024 (above): its third stage refines frame scores by **aggregating scores of semantically similar frames via kNN in video-text space** — the same mechanism, applied training-free and (essentially) within-video. Closest single overlap; a reviewer will call C4 "LAVAD's score refinement applied to a trained MIL model, cross-video."
2. **ECALP — "Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model"**, arXiv:2412.18303. Transductive label propagation over a graph of test samples — identical machinery, image classification domain.
3. **CKNN**, arXiv:2408.03014 — kNN as the anomaly scorer (unsupervised VAD); establishes kNN-in-feature-space credibility for VAD but is not propagation of a trained model's scores.
4. **Adaptive Graph Convolutional Networks for WSVAD**, arXiv:2202.06503 — graph over snippets (similarity + temporal) but learned at train time, not transductive test-time.
5. Test-time adaptation VAD exists mostly outside video (graph/tabular/time-series TTA: AdaGraph-T3 arXiv:2502.14293, TA-GGAD arXiv:2603.09349); no cross-video transductive score propagation for weakly-supervised temporal localization found.

### Overlap analysis
Label propagation is 20-year-old machinery; novelty weight of the mechanism alone is low, and LAVAD already refined VAD scores by neighbor aggregation. Nothing found doing **cross-video transductive propagation over the entire test corpus for per-second hate/anomaly localization**, and our lab's test-input-usable protocol (user ruling 2026-08-09) makes it legitimate here where standard VAD papers avoid it.

### What survives
The transductive setting itself for this task + cross-video propagation (hateful evidence in one test video sharpening ordering in another). Risks: (i) reviewers tag it a post-processing trick, not a method; (ii) transductive protocol breaks comparability with inductive baselines — needs both-protocol reporting; (iii) gains may duplicate C3 (both exploit cross-video feature neighborhoods).

### Verdict: **open** as a mechanism-in-this-setting, but low ceiling as a standalone contribution; best as an add-on module with an inductive/transductive ablation.

---

## Broad sweep: weakly supervised hateful video localization 2025–2026 (beyond MultiHateLoc & HateClipSeg)

Must-know new entrants:
1. **TANDEM — "Temporal-Aware Neural Detection for Multimodal Hate Speech"**, arXiv:2601.11178 (v3 2026-07). Qwen2.5-VL-7B + Qwen2-Audio-7B, LoRA SFT + GRPO/GSPO RL; joint video-level classification + **temporal segment localization** (IoU/Acc@0.5) + target identification on HateMM, MultiHateClip, ImpliHateVid. Not weakly supervised and 7B at inference, but it is the strongest new temporal-hate competitor; any new method must argue against it on supervision cost and inference cost.
2. **LELA**, arXiv:2602.09637 — training-free frame-level hate localization (HateMM, MHC-EN), frame AUC/AP protocol = our protocol. Mandatory baseline going forward (alongside VERA).
3. **SafeLens**, AAAI-26 demonstration (SUTD, Roy Ka-Wei Lee group) — segment-level moderation system on HateClipSeg (Whisper + EasyOCR + Qwen2.5-VL + LoRA-Llama3-8B). Signals the HateClipSeg group is actively building segment-level systems; expect a full-paper follow-up.
4. **ImpliHateVid** (Rehman et al., ACL 2025): implicit-hate video benchmark + two-stage contrastive framework — video-level, but a new dataset our task could extend to.
5. **MM-HSD**, arXiv:2508.20546 — OCR-as-query cross-modal attention, video-level SOTA-ish on HateMM.
6. **"Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning"**, arXiv:2601.15115 (Jan 2026, same Exeter group as LELA) — video-level only.
7. Peripheral: "Enhanced Multimodal Hate Video Detection via Channel-wise..." (arXiv:2505.12051), "Cross-Modal Transfer from Memes to Videos" (arXiv:2501.15438), GuardReasoner-Omni (arXiv:2602.03328, multimodal guardrail incl. video).

No other weakly-supervised hate-localization trainer beyond MultiHateLoc surfaced; the weak-supervision niche itself is still thin — but the Durham/Exeter (LELA) and SUTD (HateClipSeg/SafeLens) groups are both converging on temporal hate, so the window is months, not years.

---

## Ranked recommendation (novelty × feasibility)

1. **C1 (dense MLLM ordering distillation)** — best position. Defensible core = within-video pairwise ORDER distillation from dense VLM window scores into a VLM-free student, which directly optimizes the within-video ROC/AP metric; no paper found combining these, and it cleanly upgrades our own POWA sparse teacher. Cost: dense Qwen2-VL passes over the training set (bounded, train-once). Mandatory: LELA/VERA as teacher-only baselines; cite MLLM4WTAL + Ju'23 + DAKD upfront and differentiate on loss and density.
2. **C3 (two-sided MIL via cross-video pseudo-negatives)** — targets our diagnosed failure (rank inversion in high-positive-fraction videos) with a selection signal that is non-circular (feature evidence, not self-scores). Novelty is narrow; survives only with hard positioning against MIST/CU/BN-WVAD/UR-DMU/TPWNG. Cheap (CPU-level on cached features) → good second arm, and combinable with C1.
3. **C4 (transductive propagation)** — genuinely unoccupied for this task but low ceiling and protocol friction; run as a cheap add-on ablation (inductive vs transductive both reported), not a headline.
4. **C2 (OCR fourth modality)** — crowded (LELA, SafeLens, MM-HSD all planted OCR flags in hate video 2025–26). Demote to an input-channel ablation inside C1/C3; do not claim as contribution.

Best composite paper shape: C1 as the method, C3 as the second loss term fixing the high-positive-fraction regime, C2 as an input channel, C4 as an ablation — versus MultiHateLoc / CMHKF / VadCLIP-family / LELA / VERA baselines already reproduced.

## Key references (ids)
LAVAD 2404.01014 · VERA 2412.01095 · Holmes-VAU 2412.06171 · Holmes-VAD 2406.12235 · MLLM4WTAL 2411.08466 · Ju et al. 2212.09335 · DAKD 2406.02831 · PRD 2507.04820 · LELA 2602.09637 · TANDEM 2601.11178 · SafeLens AAAI-26 demo (p.41712) · HateClipSeg 2508.01712 (MM'25) · MM-HSD 2508.20546 · ImpliHateVid ACL'25 · UR-DMU 2302.05160 · BN-WVAD 2311.15367 · MIST 2104.01633 · CU-pseudo 2212.04090 · TPWNG CVPR'24 · ProCL 2206.11011 · GlanceVAD 2403.06154 · CKNN 2408.03014 · ECALP 2412.18303 · AGCN-WSVAD 2202.06503 · TF-PLG IEEE 11015429 · 2601.15115 (training-free classification)
