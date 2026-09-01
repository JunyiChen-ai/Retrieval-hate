# Joint Witness Ownership: independent novelty and anti-pattern review

**Verdict: STOP**  
**Novelty score: 2.5/10 as currently specified**  
**Review date: 2026-08-31**  
**Scope:** primary papers and authors' official code repositories only. This review did not inspect or use validation performance, did not implement the model, and did not participate in its design or implementation.

## Candidate reviewed

The proposed change replaces MultiHateLoc's positive-video supervision that copies the bag label into each modality branch, together with unconditional cross-modal InfoNCE, by a joint time-by-modality MIL objective:

- every cell in a negative bag is constrained to be benign;
- a positive bag needs only a finite set of positive cells in the joint temporal-modality lattice;
- the selected cells are described as latent “witness ownership”;
- inference uses one fused temporal localizer, without ensemble, routing, or calibration.

The motivating diagnosis is credible: MultiHateLoc's video-global modality selection rarely agrees with the test-GT best modality, fused scores are often worse than a unimodal branch, and the best-branch test oracle gaps are large. The proposed label semantics are also better aligned with multimodal hate than forcing every positive video label onto every modality. The novelty problem is that both the semantics and the joint-lattice aggregation already have direct prior art.

## Decisive prior art

### 1. Classical MIL already defines the proposed positive/negative semantics

Maron and Lozano-Pérez, *A Framework for Multiple-Instance Learning*, NeurIPS 1997, defines a positive bag as one containing at least one positive instance and a negative bag as one whose instances are all negative. Treating each `(time, modality)` cell as an instance is a direct application of this definition, not a new learning principle.

[Ilse et al., *Attention-based Deep Multiple Instance Learning*, ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html) learns a Bernoulli bag label with a neural attention aggregation and interprets the learned weights as instance contributions. Renaming such weights “ownership” or “responsibility” does not create a distinct mechanism unless the candidate specifies a new probabilistic model or an identifiable constraint with behavior that attention MIL cannot express.

Generic noisy-OR is even closer to the stated logic: it marginalizes the event that at least one cell is positive, while a negative bag requires all cells to be negative. [Wang et al., *Comparing the Max and Noisy-Or Pooling Functions in Multiple Instance Learning for Weakly Supervised Sequence Learning Tasks*, 2018](https://arxiv.org/abs/1804.01146) directly studies this for weakly labelled temporal localization and warns that noisy-OR can fail to localize even when its bag semantics are correct. Consequently, “joint noisy-OR over time and modality” is a baseline, not a novelty claim.

### 2. HAN is a direct time-by-modality MMIL predecessor

[Tian et al., *Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing*, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/513_ECCV_2020_paper.php) formulates weakly supervised audio-visual temporal parsing as Multimodal Multiple Instance Learning. Its attentive MMIL pooling predicts a probability for every `(time, modality, class)` cell, learns attention along the temporal and modality axes, and sums the weighted joint lattice to obtain one video-level probability. The paper's equation (7) is explicitly a double sum over time and modality. The [authors' official repository](https://github.com/YapengTian/AVVP-ECCV20) is the implementation reference.

Functionally, this already supplies:

- a joint time-by-modality instance lattice;
- latent, input-dependent allocation of bag evidence across cells;
- one modality-agnostic video label rather than a required positive label for every modality;
- temporal predictions obtained from the same model at inference.

Factorizing the weights over the two axes versus normalizing one flat `T×M` vector is an aggregation parameterization, not a new core mechanism. Extending two modalities to visual/audio/text is also routine.

[Rachavarapu et al., *Weakly-Supervised Audio-Visual Video Parsing with Prototype-based Pseudo-Labeling*, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Rachavarapu_Weakly-Supervised_Audio-Visual_Video_Parsing_with_Prototype-based_Pseudo-Labeling_CVPR_2024_paper.html) treats this joint MMIL model as its starting baseline: its equations (1)--(4) generate cell probabilities over time and modalities, jointly pool them to one video probability, and apply one video-level cross-entropy. It then adds genuinely distinct prototype and pseudo-label machinery. This is strong evidence that joint time-modality MIL is established infrastructure rather than an open mechanism.

### 3. The exact modality-label-noise motivation is established

[Cheng et al., *Joint-Modal Label Denoising for Weakly-Supervised Audio-Visual Video Parsing*, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940424.pdf) states the same semantic defect: a video-level event can appear in only one modality, so assigning it to the other modality creates modality-specific label noise. Its key assumption is that a positive event must appear in at least one modality, not in every modality. JoMoLD dynamically removes modality-specific noisy labels by comparing losses across modalities. The [official code](https://github.com/MCG-NJU/JoMoLD) makes this an especially important baseline.

The candidate differs algorithmically from JoMoLD—latent joint pooling rather than dynamic loss-based label removal—but cannot claim discovery of the label-semantics problem or the “at least one modality owns the event” principle.

[Wu et al., *Exploring Heterogeneous Clues for Weakly-Supervised Audio-Visual Video Parsing*, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Exploring_Heterogeneous_Clues_for_Weakly-Supervised_Audio-Visual_Video_Parsing_CVPR_2021_paper.pdf) also targets modality-specific weak-label noise and uses modality exchanges to refine modality labels. Lin et al., *Exploring Cross-Video and Cross-Modality Signals for Weakly-Supervised Audio-Visual Video Parsing*, NeurIPS 2021, exploits common and diverse modality semantics and event co-occurrence to exclude irrelevant temporal segments.

### 4. “Finite witnesses” and latent positive cardinality are occupied

Sparse witness selection is standard in weak temporal localization. [Nguyen et al., *Weakly Supervised Action Localization by Sparse Temporal Pooling Network*, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Nguyen_Weakly_Supervised_Action_CVPR_2018_paper.html) learns a sparse subset of key temporal segments under video-level supervision.

More directly, [Rachavarapu et al., *Boosting Positive Segments for Weakly-Supervised Audio-Visual Video Parsing*, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Rachavarapu_Boosting_Positive_Segments_for_Weakly-Supervised_Audio-Visual_Video_Parsing_ICCV_2023_paper.pdf) models the number of positive segments as a latent variable with a Poisson-binomial formulation. The paper notes that fixing the count to one recovers the ordinary “at least one positive segment” weak-supervision constraint. Therefore a fixed top-k, sparsemax, capped noisy-OR, or cardinality-constrained witness set is not novel by itself.

### 5. The violence/anomaly literature already covers asynchronous modality MIL

[Yu et al., *Modality-Aware Contrastive Instance Learning with Self-Distillation for Weakly-Supervised Audio-Visual Violence Detection*, ACM MM 2022](https://arxiv.org/abs/2207.05500) directly diagnoses modality asynchrony and the noise caused by treating a synchronized audio-visual pair as one integral MIL instance. It constructs separate unimodal bags and modality-aware semi-bags before contrastive learning; its [official code](https://github.com/JustinYuu/MACIL_SD) is available. The candidate's removal of unconditional alignment is sensible, but the asynchrony diagnosis and modality-specific instance allocation are already occupied.

[Tian et al., *Weakly-Supervised Video Anomaly Detection With Robust Temporal Feature Magnitude Learning*, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.html) and [Ren et al., *Proposal-Based Multiple Instance Learning for Weakly-Supervised Temporal Action Localization*, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html) cover sparse positive temporal instances and proposal-level MIL, respectively. They do not use a modality lattice, but they establish that changing the instance domain and pooling granularity is ordinarily treated as a MIL design choice, not sufficient novelty on its own.

### 6. Gating, modality imbalance, and missing-modality robustness do not rescue novelty

[Fu et al., *Multimodal Imbalance-Aware Gradient Modulation for Weakly-supervised Audio-Visual Video Parsing*, 2023](https://arxiv.org/abs/2307.02041) studies unequal modality learning under a joint objective and adds a modality-separated decision unit plus gradient modulation. [Xu et al., *Rethink Cross-Modal Fusion in Weakly-Supervised Audio-Visual Video Parsing*, 2023](https://arxiv.org/abs/2311.08151) argues that early fusion entangles incompletely correlated modalities and harms single-modality events. These are close problem statements.

Adaptive modality gating is itself generic. For example, [Li et al., *SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality*, ICCV 2025](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.pdf) uses modality experts and an input-dependent gate. Missing-modality work addresses availability rather than weak temporal labels, so it is not a direct anticipation of joint witness MIL; it does show that a learned modality gate or modality dropout cannot be the novelty claim.

## Comparison with the current MultiHateLoc starting point

[MultiHateLoc](https://arxiv.org/abs/2512.10408) already contains modality-specific temporal branches, dynamic modality selection, a fused temporal branch, modality-specific top-k selection, and a union of fused and unimodal selected frames. It additionally applies cross-modal contrastive alignment. The candidate correctly challenges two questionable semantics in that formulation, but as currently described it is a **loss swap from one known multimodal MIL design to another known MMIL design**:

| Element | MultiHateLoc | Candidate | Prior occupancy |
|---|---|---|---|
| Positive supervision | modality/fused top-k construction | positive evidence somewhere in joint lattice | classical MIL; HAN |
| Negative supervision | bag-level objective | all cells benign | classical MIL semantics; standard weak anomaly MIL |
| Modality allocation | DMS plus selected branch scores | latent cell ownership | HAN attentive MMIL; attention MIL |
| Sparse/finite positives | adaptive top-k | finite witnesses | STPN; PoiBin |
| Cross-modal relation | unconditional InfoNCE | removed | prior AVVP/asynchrony critiques already motivate this |
| Test output | fused score | one fused score | ordinary single-model inference |

The change may improve performance and may be a strong corrective baseline, but neither usefulness nor task fit establishes novelty.

## Anti-pattern and identifiability review

1. **Attention is not ownership.** A normalized weight over cells is not an identifiable latent cause. Multiple weight/score combinations can yield the same bag probability. “Responsibility” is only defensible if derived from a specified probabilistic latent-variable model and validated beyond visualizing weights.
2. **Easiest-modality collapse remains likely.** Joint MIL can allocate nearly all positive mass to the globally easiest modality, reproducing DMS's failure under a different loss. A low-entropy ownership map is not evidence that the selected modality is correct.
3. **Sparse MIL commonly finds only the most discriminative snippet.** This is precisely the failure emphasized by the CVPR 2024 prototype-pseudo-label paper. A finite-witness constraint may improve video discrimination while worsening full temporal extent and within-video ROC.
4. **Negative bags are useful but insufficient for positive-video ranking.** All-cell negative supervision can improve pooled separation without teaching which benign seconds inside a positive video are negative. The project has repeatedly observed this anti-pattern.
5. **Train/test mismatch is possible.** If the loss is applied to modality-cell logits but inference reads a separate fused localizer, the fused output is not identified by the proposed bag semantics. If the fused output is simply the ownership-weighted cell sum, the design becomes even closer to HAN.
6. **Removing InfoNCE confounds the claim.** A gain may come entirely from deleting a harmful alignment loss rather than from joint witness ownership. This must be isolated.
7. **Noisy-OR is especially risky for long videos.** Its bag probability grows with the number of cells and can spread small positive probabilities across many cells. Length normalization or cardinality control is necessary, but those fixes are known pooling variants and do not supply novelty.
8. **A name cannot carry the claim.** “Joint witness ownership” must not be used to rebrand flat MMIL attention, top-k pooling, sparsemax, or noisy-OR.

## What would be required to reopen the direction

Reopening would require a mathematically distinct mechanism, not a renamed aggregator. At minimum, the proposal would need all of the following before another novelty review:

1. A precise latent-variable model stating the random variables, likelihood, marginalization or variational bound, and how fused frame scores are identified from cell responsibilities.
2. A property that HAN attention, flat top-k, noisy-OR, and latent-count PoiBin cannot express—for example, a falsifiable constraint coupling modality ownership across time to the single fused localizer without copying bag labels or requiring synchronized modalities.
3. A proof or direct numerical equivalence test showing it does not reduce to attentive weighted pooling under a reparameterization.
4. A mechanism-specific prediction beyond higher aggregate metrics, such as calibrated recovery of which modality carries each hateful interval under controlled modality deletion, while preserving temporal extent.

Without these additions, the correct characterization is “a semantically corrected MMIL baseline for MultiHateLoc,” not a novel method.

## Minimum falsifiable pilot if retained as a corrective baseline

This pilot is not authorization to call the current idea novel. It is the cheapest test of whether the semantic correction is technically useful.

### Fixed setup

- Train HateMM and HateClipSeg independently; do not mix main-dataset train sets.
- Freeze MultiHateLoc features, temporal encoders, fused localizer, optimization budget, and checkpoint-selection procedure.
- Use validation only to select a checkpoint within each fixed run. Immediately evaluate the selected checkpoint on test using the shared evaluator and all three fixed metrics.
- No ensemble, branch routing, score calibration, or post-hoc transport.

### Required arms

1. **Original MultiHateLoc loss.**
2. **Deletion control:** original model with unconditional InfoNCE removed, leaving the original MIL semantics unchanged.
3. **Known-MMIL control:** flatten the `T×M` cells and use a standard top-k/max or HAN-style attentive MMIL bag loss with matched parameters.
4. **Candidate:** the proposed ownership objective.

The candidate is falsified as a mechanism if it does not beat both arm 2 and arm 3 on test within-video ROC on both datasets while retaining pooled AP and ROC. Beating only arm 1 means the result may be caused by removing InfoNCE. Matching arm 3 means the result is generic MMIL.

### Required mechanism checks

- Report modality mass, temporal support size, and modality entropy separately for positive and negative test videos.
- On positive test videos, compare ownership against controlled single-modality deletion effects at the same timestamps. This is attribution analysis, not a training target.
- Measure whether predicted support covers the full positive temporal extent rather than only the peak snippet.
- Check whether negative-video all-cell loss merely lowers every score or actually improves benign-second ranking inside positive videos.
- Report results stratified by the already exposed categories: fused-better-than-all-unimodal versus best-unimodal-better-than-fused.

## Mandatory attribution controls for any later paper claim

1. **Flattened MMIL:** same cell logits and parameter count, no named ownership variable.
2. **HAN attentive pooling:** factorized temporal and modality attention.
3. **Temporal-only MIL:** apply the same positive/negative semantics to fused temporal scores without a modality axis.
4. **Uniform ownership:** replace learned modality responsibilities with equal weights.
5. **Shuffled ownership:** within each video, independently permute responsibilities across time and across modalities while preserving their marginal distributions.
6. **Stop-gradient ownership:** test whether gains require responsibility learning or only score reweighting.
7. **InfoNCE deletion factorial:** cross the old/new MIL objective with InfoNCE present/absent.
8. **Cardinality control:** fixed top-1/top-k and latent-count or soft support-size variants, because “finite witnesses” is otherwise un-attributed.
9. **Capacity control:** add a parameter-matched auxiliary head to the standard MMIL baseline.
10. **Fused-output linkage:** remove any auxiliary modality-cell loss at inference and verify that the single fused localizer itself carries the gain.
11. **Modality-collapse audit:** compare learned modality mass to per-timestamp deletion effect; do not equate peaked weights with correct ownership.
12. **Temporal-extent audit:** peak accuracy and video classification are insufficient; the claimed mechanism must improve within-video ranking and positive-span coverage.

## Claim boundary

With the current specification, the following claims are **not supportable**:

- first joint time-modality MIL for weak temporal localization;
- novel latent witness ownership;
- novel positive-at-least-one / negative-all semantics;
- novel sparse or finite witness learning;
- novel solution to modality-specific weak-label noise or modality asynchrony.

A defensible engineering statement would be: “We replace MultiHateLoc's branch-copied supervision and unconditional alignment with a standard joint MMIL objective better matched to modality-agnostic video labels.” That may be valuable as a baseline or ablation, but it does not meet this project's novelty gate.
