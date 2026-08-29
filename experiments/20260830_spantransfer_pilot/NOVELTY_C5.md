# Adversarial Novelty Check — Candidate C5

**Candidate:** Cross-corpus span-supervised pretraining + rank-preserving weak adaptation.
Leave-one-corpus-out (LOCO) over HateMM / MultiHateClip-EN / MultiHateClip-ZH / HateClipSeg: temporal conv head on frozen CLIP+VGGish+BERT 1-fps features, frame-supervised pretraining on the other corpora's train spans, then MIL adaptation on the target with only video-level labels under a pairwise self-distillation constraint that preserves the pretrained model's within-video score ordering.

**Date:** 2026-08-30. **Method:** WebSearch/WebFetch over arXiv + ECCV/ICCV/CVPR/AAAI/ACM MM proceedings, 2018–2026. Stance: adversarial (looking for reasons it is NOT novel).

---

## 1. Closest prior art (ranked by threat level)

| # | Paper | Venue/Year | ID | What they do (one line) |
|---|-------|-----------|----|--------------------------|
| 1 | **AherNet: Learning to Localize Actions from Moments** (Long et al.) | ECCV 2020 | arXiv:2008.13705 | Transfers temporal-localization ability learned from a small fully span-annotated action set to a large set of categories that have only trimmed moments / video-level supervision, via a weight-transfer function + adversarial context synthesis. |
| 2 | **Cross-Domain Learning for Video Anomaly Detection with Limited Supervision** (CDL) | ECCV 2024 | arXiv:2408.05191 | Weakly-supervised (video-level MIL) VAD that adds *unlabeled* external-domain data during training, weighting it by estimated prediction bias / uncertainty, for cross-domain generalization on UCF-Crime / XD-Violence. |
| 3 | **Temporal Action Detection with Multi-level Supervision** (Shi et al.; SSAD/OSAD) | ICCV 2021 | arXiv:2011.11893 | Trains one temporal detector from a mix of fully-labeled + weakly-labeled + unlabeled videos (omni-supervision), i.e., "frame spans here + video labels there" — but all within one dataset/taxonomy. |
| 4 | **Anomaly Crossing** | arXiv 2021 (WACV-line) | arXiv:2112.06320 | Formulates VAD as cross-domain few-shot learning: representation learned on a large source video corpus transferred to a target domain with a few labeled clips. |
| 5 | **RefineVAD** | arXiv 2025 | arXiv:2511.13204 | Weakly-supervised VAD with semantic recalibration; reports zero-shot cross-dataset transfer (UCF-Crime→XD-Violence, 77.6 AP) — cross-corpus eval exists in weak VAD, but with no target adaptation phase. |
| 6 | **Cross-Modal Transfer from Memes to Videos** | arXiv 2025 | arXiv:2501.15438 | Augments hateful *video* training (HateMM, MultiHateClip) with hateful *meme* datasets — cross-dataset transfer inside the hate domain, but purely video-level classification, no temporal localization. |
| 7 | **Revealing Temporal Label Noise in Multimodal Hateful Video Classification** | arXiv 2025 | arXiv:2508.04900 | Uses the HateMM / MultiHateClip-EN span timestamps to show video-level labels are temporally noisy; motivates segment granularity but proposes no transfer or localization method. |
| 8 | **HateClipSeg** | ACM MM 2025 | arXiv:2508.01712 | Segment-level annotated hate video dataset; defines "Temporal Hateful Video Localization" as a benchmark task with (fully-supervised) baselines — the task definition C5 targets, no cross-corpus or weak-adaptation protocol. |

Component-level precedent (not head-to-head competitors, but kills "new mechanism" claims for the ingredients):
- **Ranking Distillation** (Tang & Wang, KDD 2018) and **RankDistil** (Reddi et al., AISTATS 2021): distilling to preserve top-k / pairwise orderings is an established objective family.
- **Self-Distillation Fine-Tuning** literature (incl. LwF-style anti-forgetting): using the pre-adaptation model as its own teacher to prevent finetuning from destroying prior knowledge is standard.
- **WENO** (NeurIPS 2022, arXiv:2210.03664) and Distill-to-Label (arXiv:1907.12926): distillation between bag-level MIL teachers and instance-level students — MIL + distillation coupling is known.
- **Synthetic Video Generation for Weakly Supervised Cross-Domain VAD** (Springer, ~2024): another cross-domain weak-VAD data-side approach.
- **SafeLens** (AAAI 2026 demo) and **TANDEM** (arXiv:2601.11178): segment-level hate detection systems / temporal-aware hate models evaluated across HateMM/MultiHateClip/ImpliHateVid — the hate-video temporal space is getting populated, but by LLM-pipeline or single-corpus-supervised approaches.
- Text hate speech: leave-one-dataset-out cross-dataset generalization is a well-worn protocol (e.g., arXiv:2208.10598 and the generalisable-hate-speech review arXiv:2102.08886) — LODO itself is not a novel protocol idea, only its instantiation at the temporal level in video.

## 2. Exact overlap vs. what remains novel

**Already taken (do not claim):**
1. *"Pretrain with full temporal supervision on auxiliary data, adapt to targets that have only video-level labels."* AherNet (2020) is precisely this recipe in TAL (cross-category rather than cross-corpus, but a reviewer will not care about that distinction unless forced to). OSAD (ICCV 2021) covers the mixed span+video-label supervision regime in-domain.
2. *"Cross-dataset transfer / generalization for weakly-supervised anomaly-style temporal scoring."* CDL (ECCV 2024), RefineVAD's zero-shot transfer, Anomaly Crossing, and the synthetic-data cross-domain VAD paper occupy this space in VAD.
3. *"Rank/order-preserving distillation"* and *"self-distillation to prevent forgetting during finetuning"* are both mature component techniques (KDD'18, AISTATS'21, SDFT line). The pairwise self-distillation loss itself is not a new mechanism.
4. *"Cross-dataset transfer within the hate domain"* at the **video level**: memes→videos (2501.15438) and multi-corpus video-level evaluations (TANDEM) exist.
5. *Leave-one-dataset-out* as a protocol: standard in text hate speech.

**Still open (defensible):**
1. **No work does corpus-to-corpus *temporal* transfer among hateful-video datasets.** All existing hate-video cross-dataset work is video-level; all existing segment-level hate work (HateClipSeg baselines, SafeLens, TANDEM) is single-corpus-supervised or zero-shot-LLM. A four-corpus LOCO benchmark for temporal hate localization does not exist.
2. **The specific coupling is unclaimed:** span-supervised cross-corpus pretraining *protected during MIL adaptation by a rank-preservation constraint*, with the explicit decomposition "auxiliary corpora supply within-video ordering (which video labels information-theoretically cannot), target video labels supply corpus-specific calibration/prior." Neither AherNet nor CDL makes or ablates a claim about MIL finetuning destroying transferred ordering; CDL's external data is *unlabeled*, AherNet has no target-side adaptation loss protecting ordering.
3. **Definition shift as the transfer obstacle.** In VAD/TAL the domain gap is visual (surveillance vs. movies; categories). Across hate corpora the gap is *annotation-definition and platform* shift (what counts as hate on YouTube vs. Bilibili vs. BitChute), mirroring the text-side generalization literature — this framing has no video-side occupant.

## 3. Adversarial attacks a reviewer will make

1. **"This is AherNet + Ranking Distillation applied to a new domain."** The strongest attack. Both halves have citable precedent; the composition must be defended by *evidence* (ablation showing naive MIL adaptation catastrophically degrades transferred ordering and rank-preservation recovers it), not by claimed mechanism novelty.
2. **"Calling this weakly-supervised is misleading."** The method consumes frame-level spans — just not the target's. Expect a demand to rename to "cross-corpus transfer" / "omni-supervised" and to compare against (a) zero-shot span-pretrained model with no adaptation, (b) in-corpus fully-supervised ceiling, (c) pure MIL floor. If the adapted model does not clearly beat (a), the whole adaptation story collapses to "supervised transfer + calibration," a much weaker paper.
3. **"Four tiny corpora."** HateMM ~1k videos; MultiHateClip 2k; HateClipSeg ~11.7k segments. Cross-corpus conclusions from datasets this small invite noise objections; multi-seed + CI mandatory.
4. **"HateClipSeg already benchmarks temporal hateful localization; you are a method entry on their task plus a protocol tweak."** Mitigated by being the *first transfer/LOCO* entry, but the task itself cannot be claimed.
5. **"CDL shows uncertainty-weighted external data beats naive transfer"** — CDL-style bias-corrected joint training is a required baseline, or a reviewer will ask why rank-distillation beats simply mixing losses.

## 4. Verdict

**open-with-differentiation.**
- Component mechanisms: all taken.
- Generic recipe (full-span source → weak target): taken in TAL/VAD (AherNet, OSAD, CDL family) — crowded.
- Cross-corpus temporal transfer *among hate-video corpora* with a rank-preservation adaptation mechanism and a LOCO protocol: no occupant found (2021–2026 sweep). Novelty survives only if framed as domain + protocol + mechanistic-evidence contribution, not as a new learning mechanism.

## 5. Framing choices that maximize differentiation

1. **Lead with the problem + protocol, not the recipe.** "First cross-corpus study of temporal hate localization: four corpora, leave-one-corpus-out, target supervision strictly video-level." This makes AherNet/CDL related work, not competitors — none of them can be run on this claim without your protocol.
2. **Make rank-preservation an *evidenced finding*, not an invented loss.** Center the paper on the measurable phenomenon: naive MIL adaptation destroys the transferred within-video ordering (report the ordering metric before/during/after adaptation); pairwise self-distillation is the minimal fix. Cite Ranking Distillation / SDFT openly as the tool; claim the diagnosis, not the tool.
3. **Frame the supervision asymmetry as definition-shift calibration.** Auxiliary spans give ordering that video labels cannot express; target video labels re-calibrate to the target corpus's hate definition (YouTube vs. Bilibili vs. BitChute annotation policies) — connect explicitly to the text-side LODO generalization literature (arXiv:2102.08886) that has no video-side counterpart. Avoid the bare word "weakly-supervised" in the title claim; use "target-span-free" or "video-label-only adaptation."

## 6. Mandatory baselines implied by prior art
- Zero-shot: span-pretrained LOCO model, no adaptation (RefineVAD-style transfer eval).
- Pure MIL on target only (floor).
- Joint multi-task: spans on auxiliaries + MIL on target in one training run, no distillation (OSAD-style), and a CDL-style uncertainty-weighted variant.
- Naive finetune (pretrain → MIL, no rank constraint) — the ablation that carries the paper.
- In-corpus fully-supervised ceiling (HateClipSeg baseline numbers where available).

## Sources
- https://arxiv.org/abs/2008.13705 (AherNet, ECCV 2020)
- https://arxiv.org/abs/2408.05191 (CDL, ECCV 2024)
- https://arxiv.org/abs/2011.11893 (Multi-level supervision TAD, ICCV 2021)
- https://arxiv.org/abs/2112.06320 (Anomaly Crossing)
- https://arxiv.org/abs/2511.13204 (RefineVAD)
- https://arxiv.org/abs/2501.15438 (Memes→Videos transfer)
- https://arxiv.org/abs/2508.04900 (Temporal label noise in hateful video classification)
- https://arxiv.org/abs/2508.01712 (HateClipSeg, ACM MM 2025)
- https://proceedings.mlr.press/v130/reddi21a/reddi21a.pdf (RankDistil, AISTATS 2021)
- https://arxiv.org/abs/2210.03664 (WENO, NeurIPS 2022)
- https://arxiv.org/abs/2208.10598 (LODO hate speech, text)
- https://arxiv.org/abs/2102.08886 (Generalisable hate speech review)
- https://ojs.aaai.org/index.php/AAAI/article/download/42390/46351 (SafeLens, AAAI 2026 demo)
- https://arxiv.org/abs/2601.11178 (TANDEM)
- https://link.springer.com/chapter/10.1007/978-3-031-78354-8_24 (Synthetic cross-domain weak VAD)
