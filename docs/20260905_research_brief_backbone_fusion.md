# Research Brief — weakly supervised hateful video localization: backbone (module 2) and fusion (module 3) novelty

Date 2026-09-05. Written for the idea-discovery pipeline after candidates 2, 3 and 4 of the backbone module failed. Authoritative numbers live in `runs/`; this brief transcribes them.

## Problem Statement

Weakly supervised temporal localization of hateful content in videos: training uses only video-level labels (hateful / not), inference must score every second (1 fps grid). Two corpora: HateMM (744 train / 215 test videos, long videos, median test length 175 s) and HateClipSeg (393 videos, short clips, median 355 s at 1 fps grid after alignment). Metrics on test, fixed: pooled frame-level AP and pooled frame-level ROC-AUC (main); within-video macro ROC-AUC over positive videos that contain both classes (a floor constraint, HateMM ≥ .632, HateClipSeg ≥ .524, not a comparison metric).

We need a **method paper**: a backbone module and a fusion module that are each a claimable design, each with full ablations showing the pooled gain is caused by the design, on BOTH corpora (one method, no per-corpus variants). Gains can be small (+.005 to +.02 pooled) but must be real under the noise floor described below.

## Background

Current strongest method ("candidate 1", `experiments/20260903_hier_evidence_mil/`, confirmed 3 seeds):
- Frozen VLM (Qwen2.5-VL-7B) gives per-window binary hate verdicts at two granularities (K=30 fine windows and J=4 coarse blocks per video); these are the only "evidence" signal beyond raw features.
- Module 3 (fusion, existing): a two-level hidden Markov model over the verdicts (fine verdicts = noisy observations of a Markov hate state, coarse verdicts = noisy observations of a block-level OR), parameters fitted by EM on train video labels; its per-second posterior log-odds enter the frame logit as a fixed-scale prior. Verdict columns are also concatenated to the audio input.
- Module 2 (backbone, existing = MACIL-SD AVCE): linear projections of visual (I3D-like) and audio+text features into 128-d, ONE transformer layer shared by both cross-modal directions (audio queries video, video queries audio), per-frame logit = fc(a_out) + fc(v_out), top-k MIL bag over frames, cross-modal alignment contrastive loss (CMAL), an EMA self-distillation partner network (visual-only). Plus a verdict-block MIL loss (one bag per coarse block, soft label from the HMM posterior).
- Numbers (3 seeds, mean): HateMM AP .657 / ROC .842 / within .646; HateClipSeg .699 / .681 / .553. HMM posterior alone (no training): HateMM .541 / .818, HateClipSeg .698 / .661 — the backbone adds a lot on HateMM, almost nothing on HateClipSeg AP.
- What the backbone provably uses (candidate 1 README 9.5–9.7): video-level hate density estimated from the verdict distribution plus content; correcting verdict reliability across the two granularities; within-video ordering comes mostly from the HMM temporal posterior, content adds ~.03 AP on HateMM, text adds nothing within-video. The shared cross-modal attention is the only AVCE part that is confirmed on both corpora (removing it: HateMM −.054/−.037, HateClipSeg −.004/−.012); it acts on pooled metrics only, not within-video.

Failed backbone candidates (all archived, details in their READMEs):
- Candidate 2, evidence-chain network (`archive/experiments/20260904_evidence_chain_net/`): explicit density encoder + granularity reliability gate + differentiable three-state chain head trained by marginal likelihood. Both corpora failed rule 8; the only effective piece was distilling the label-conditioned chain posterior into the network, and that gave no gain when added to candidate 1.
- Candidate 3, evidence-guided cross-modal attention (`experiments/20260904_evidence_guided_attention/`): evidence encoding into query/key, per-head key bias from evidence, video-level evidence context vector. On HateClipSeg it helps ROC (+.025 over same-training control, 3 seeds) but on HateMM every variant breaks the within-video floor: pushing every second toward the same evidence seconds erases within-video ordering.
- Candidate 4, null-token cross-modal attention (`experiments/20260904_null_token_cma/`): a learnable "attend to nothing" key/value token conditioned on the video-level verdict summary, padding masked. Diagnosis that motivated it: MACIL-SD never masks padded rows, so training uses zero rows as an accidental attention sink (.25 of attention mass on HateMM) that is absent at test time. Result: HateClipSeg +.014 AP over the same-training control after removing stream noise, HateMM +.005; zero effect when put inside candidate 1's training with the EMA partner; at test time the token receives only 1/T of the attention on HateClipSeg (used on HateMM: audio queries put .13 on it) — the mechanism is not what was claimed.
- Module 1 (VLM elicitation) variant "context-conditioned verdicts" was eliminated: adding block transcript context makes the VLM trigger more with lower precision.

## Constraints

- Compute: three RTX 5090 (32 GB) machines; one training trial ≈ 5 min on HateMM, ≈ 2 min on HateClipSeg; protocol is 20 Optuna trials per (corpus, seed), 3 seeds, objective = test (AP+ROC)/2, val-selected checkpoint, within floor prunes trials. A candidate costs ~6 GPU-hours HateMM + ~2.5 GPU-hours HateClipSeg for search + ablations.
- Features are frozen and cached (visual, audio, text/ASR embeddings, VLM verdicts). No fine-tuning of the VLM. Re-eliciting VLM verdicts with new prompts costs ~2 GPU-hours per corpus and is allowed but is module 1, not the target here.
- **Noise floor (measured 2026-09-05, candidate 4 README 8.2)**: re-running the same arm with the same hyperparameters but a different random stream moves pooled AP by std .006–.009 (HateClipSeg); the searched best trial is +.006 above its own stream mean (test-selected search). Single-run arm differences within ±.02 are not interpretable; HateMM is noisier (3-seed std .019). Any idea must plausibly produce > .02 pooled gain on at least one corpus, or come with a plan for a fair (multi-stream) ablation.
- HateMM within-video floor kills anything that makes all seconds of a video share one video-level signal in the content stream (candidate 3). Video-level signals may only enter through the prior or through per-second gating.
- One method for both corpora; no per-corpus architecture switches. Datasets MHC-EN/ZH are retired; any new dataset can only be external validation.
- Existing code conventions: evaluator is a single shared script; experiments live in `experiments/<date>_<slug>/`; shared logic goes to `src/`.

## What I'm Looking For

1. A backbone (module 2) design that changes the MACIL-SD AVCE architecture in a way that (a) is a claimable module with a mechanism story, (b) targets what the backbone actually does (video-level density estimation from verdict distribution + content; granularity reliability correction; within-video ordering from content on HateMM), (c) does not inject a shared video-level vector into every second's content representation, (d) has ablation arms that isolate the mechanism, and (e) is plausibly worth > .02 pooled AP or ROC on at least one corpus.
2. A fusion (module 3) design with a theoretical wrapper: the current HMM-posterior prior works; candidates could make the verdict-to-frame fusion learned but structured (e.g., calibrated evidence combination, reliability-weighted temporal models, uncertainty-aware fusion of VLM verdict and content logit) — but must beat the fixed-scale HMM prior on both corpora.
3. For each idea: which existing component it replaces, expected mechanism, the two-line ablation plan, the expected failure mode on HateMM within-video ordering, and closest prior work (weakly supervised violence/anomaly detection: MACIL-SD, UR-DMU, MGFN, PEL4VAD, CLIP-TSA, VadCLIP, LAVAD/LAP; temporal action localization; multimodal hate detection: HateMM, MultiHateClip, HateClipSeg papers).

## Domain Knowledge

- Verdict evidence dominates: removing verdicts from everywhere costs .13 AP on both corpora; the prior path (HMM posterior) is worth .04–.06 AP; the verdict-block MIL loss .02–.04.
- Cross-modal attention is confirmed useful only on pooled metrics; within-video ordering on HateMM is fragile (most search trials break the .632 floor for any attention modification).
- Padding in MACIL-SD's attention is unmasked and acts as a train-only sink on HateMM; masking without replacement hurts HateMM (−.04 AP); an explicit null token restores it but adds little.
- Deterministic training: same setting reruns are bit-identical; differences between settings at one hyperparameter point are deterministic offsets, not noise in the usual sense, so multi-stream or multi-seed averaging is required for arm comparisons.

## Non-Goals

- Benchmark, audit, metric or protocol papers. Only a method paper.
- New datasets as main results; new VLMs; VLM fine-tuning.
- Tricks without a mechanism claim (ensembles, pairwise post-processing) — recorded but not claimable.
- Engineering variations of candidate 1's losses/inputs without an architectural claim (the user rejected "revision 1" as paper method for that reason).

## Existing Results (if any)

- `research-wiki/STATUS.md` (status entry), `experiments/20260903_hier_evidence_mil/README.md` sections 4, 7, 9 (candidate 1 numbers and mechanism analysis), `experiments/20260904_null_token_cma/README.md` sections 6–8 (candidate 4 and the noise-floor measurement), `experiments/20260904_evidence_guided_attention/README.md` section 7.2 (candidate 3, HateClipSeg-only evidence-guided attention finding), `docs/20260904_evidence_chain_backbone_novelty_precheck.md` (literature pre-check for the evidence-chain direction), `RESEARCH_ITERATION_RULES.md` (gates and protocol).
