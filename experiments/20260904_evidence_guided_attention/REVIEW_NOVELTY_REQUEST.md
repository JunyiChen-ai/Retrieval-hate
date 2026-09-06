# Review brief: candidate 3, evidence-guided cross-modal attention (2026-09-07)

Repository root: /home/jehc223/Retrieval-hate. All paths relative to it. Everything is on disk; verify numbers against the cited files rather than trusting this brief. Do not modify any file.

## Task and setting

Weakly supervised temporal localization of hateful content in videos. Training uses only video-level labels; evaluation is frame-level at 1 fps on the test split: pooled frame AP, pooled frame ROC-AUC (the standard metrics of weakly supervised video anomaly detection, Sultani CVPR'18 / Wu ECCV'20), plus a within-video macro ROC over positive videos reported only as an additional analysis (project ruling 2026-09-06: it is not a gate and not a comparison metric). Two corpora: HateMM (744/109/214 train/val/test) and HateClipSeg (251/63/79). Evaluator: `scripts/reproduction_baselines/eval_baseline_scores.py`. Baseline table with 3-seed means: `docs/duplex/OFFICIAL_VAL_RESULTS.md`.

## Context: the previous review

On 2026-09-06 you (same model, thread 01a07587-d980-79f0-ad5d-4afbe77abf74) reviewed the *cleaned candidate 1* (`experiments/20260906_hier_evidence_clean/`, brief `REVIEW_NOVELTY_REQUEST.md`, review `REVIEW_NOVELTY_GPT6ASTRA.md`): score 3/10, verdict "engineering note, not a paper"; the main criticisms were (i) backbone is MACIL-SD unchanged, (ii) the label model over VLM verdicts is Snorkel/CHMM-style with a fixed arbitrary 30/4 grid, (iii) on HateClipSeg training adds little over the training-free posterior, (iv) test-objective hyperparameter search. The owner then chose candidate 3 as the current method because it is the one candidate whose backbone changes have three-seed structural ablations on both corpora. Please read the previous review before writing this one, and say explicitly which of its criticisms candidate 3 answers and which remain.

The owner has ruled that the hyperparameter search objective stays test-based (developmental upper-bound measurement); do not spend the review on that point beyond one sentence.

## The method (`experiments/20260904_evidence_guided_attention/README.md` sections 1 and 7; code `model.py`, `train.py`; shared code `src/hier_evidence_common.py`, `src/verdict_hmm.py`)

Inputs per 0.667 s row: frozen I3D (video), VGGish (audio), BERT sentence vectors from Whisper transcripts.

1. **Module 1, frozen VLM verdicts** (unchanged from candidate 1). Qwen2.5-VL-7B-Instruct, zero-shot, rates each video at two fixed granularities: 30 equal windows and 4 equal blocks, each shown 4 frames plus its transcript, output a 0-3 level binarised at >= 2. 34 VLM calls per video, cached in `data/MLLM_scores`. The 30 and 4 are historical (caches existed), not swept; they are not nested.
2. **Module 3, fusion: hierarchical evidence HMM** (unchanged). A two-level generative label model over the verdicts: latent hate state per fine window as a 2-state Markov chain; fine verdict = noisy observation of the state; coarse verdict = noisy observation of "any hate in the block" (OR). Six parameters from TRAIN video labels only (EM). Forward-backward gives per-row posterior log-odds ell_t, P(s_t), per-block P(h_j). Frame logit z~_t = z_t + alpha * ell_t / L. A searched weight w_fine in [0,1] scales the fine-verdict log-likelihood in the HMM.
3. **Module 2, backbone (this candidate's change).** MACIL-SD's shared cross-modal attention layer is kept, but the verdict columns are no longer concatenated to the content input. Instead, an evidence code per row e_t = Embedding(4,128)[2*b30_t + b4_t] + Linear(2->128)([ell_t/L, P(s_t)]) enters the attention in three places: (A) added to the query and key inputs only (value and residual stream stay pure content), (B) a per-head additive key bias beta_h(e_j) = Linear(128->4), (C) a video-level context c = Linear(mean_t e_t) added to both modality streams before the linear head. Padding keys are masked. EMA/visual-partner self-distillation of MACIL-SD is removed. Head, top-ceil(T/16) bag BCE, CMAL contrastive loss (two weights merged into one lambda_cma), and the verdict-block MIL on the content logit (block soft label P(h_j), weight |2P-1|, lambda_block) are as in candidate 1.

Method-level scalars: alpha (prior scale), w_fine, lambda_block. Searched training scalars: lr, max_seqlen, lambda_cma. Optuna 20 trials per (corpus, seed), objective test (AP+ROC)/2, checkpoint by validation. No post-processing, no ensemble, one method for both corpora.

## Results (3 seeds = 234/2025/3407, each the best trial of its own 20-trial search)

Raw files: HateMM `runs/20260904_evidence_guided_attention_rev2_noprune/hatemm/seed<seed>/trial<k>/metrics.json` (best trials 17/8/13; searched without within pruning), HateClipSeg `runs/20260904_evidence_guided_attention_rev2/hateclipseg/seed<seed>/trial<k>/metrics.json` (best trials 13/12/16; that search still had within pruning, but recomputation under the no-prune rule selects the same trials). Summary with every source path: `runs/20260904_evidence_guided_attention_rev2_noprune/ablations/three_seed_summary_both_corpora.json`.

| | AP | ROC | within | strongest trained baseline (3-seed) | candidate 1 (3-seed) |
|---|---|---|---|---|---|
| HateMM | .668 +- .010 | .850 +- .005 | .623 +- .015 | MACIL-SD .573 +- .033 / .807 +- .019 | .657 / .842 / .646 |
| HateClipSeg | .698 +- .008 | .684 +- .009 | .549 +- .011 | Fed-WSVAD .562 (AP), DSANet .528 (ROC); VERA (training-free VLM) .619 / .605 / .562 | .699 / .681 / .553 |

Three-seed ablations, both corpora, 12 arms, each seed with its own best-trial hyperparameters (per-arm metrics under `runs/20260904_evidence_guided_attention_rev2_noprune/ablations/hatemm/seed<seed>/<arm>/metrics.json` and `runs/20260904_evidence_guided_attention_rev2/ablations/hateclipseg/seed<seed>/<arm>/metrics.json`). Numbers are mean drops in AP / ROC relative to full. Project criterion for "a part is useful": three-seed mean drop >= .01 in AP or ROC on each corpus (no per-seed requirement).

| removed | HateMM | HateClipSeg | holds on both |
|---|---|---|---|
| whole evidence-guided attention, back to candidate 1 backbone (avce) | -.044 / -.019 | -.009 / -.025 | yes |
| evidence added to the content stream instead (stream_enc, revision 1) | -.047 / -.021 | -.028 / -.040 | yes (record only) |
| evidence in q/k (no_qk_enc; bias and context kept) | -.039 / -.022 | -.001 / -.001 | **no** |
| 2x2 cell embedding -> linear on two columns (no_cell) | -.020 / -.014 | -.015 / -.027 | yes |
| per-head key bias (no_bias) | -.020 / -.014 | -.011 / -.018 | yes |
| per-head bias -> single scalar (scalar_bias) | -.025 / -.015 | -.012 / -.020 | yes |
| video-level context (no_context) | -.040 / -.027 | -.020 / -.030 | yes |
| HMM posterior -> plain mean level (mean_prior) | -.052 / -.035 | -.025 / -.025 | yes |
| verdict-block MIL (no_block) | -.070 / -.040 | -.047 / -.058 | yes |
| prior term (no_prior) | -.050 / -.020 | -.050 / -.052 | yes |
| CMAL (no_cmal) | -.053 / -.029 | -.017 / -.029 | yes |
| all VLM verdicts (no_verdict) | -.159 / -.090 | -.100 / -.118 | yes |

Per-seed detail is in the summary JSON: on HateMM seed 3407 no_bias/scalar_bias do not drop (-.003 / .000); on HateClipSeg seed 234 eight arms do not drop in AP while the other two seeds drop >= .01.

Known facts from candidate 1's analysis (`experiments/20260903_hier_evidence_mil/README.md` section 9), still relevant: 92% (HateMM) / 79% (HateClipSeg) of score variance is between videos; the backbone mainly estimates per-video hate density; within-video ordering comes mostly from the HMM posterior. Earlier diagnosis (README section 7.1) found that evidence-guided attention lowers within-video ROC on HateMM relative to the plain backbone; the new three-seed run confirms within is .023 below candidate 1 while pooled is above it.

## What has already been tried and failed (archived under `archive/experiments/`)

Between 2026-09-03 and 09-06: evidence-chain network, null-token cross-modal attention, verdict-conditioned density estimation, interventional evidence (4 masked VLM passes per window), latent evidence sequence model, context-witness (4 context-conditioned VLM passes, stopped for cost), censored evidence process, interval evidence transport. None beat candidate 1 on either corpus. Pattern: pushing verdict evidence into the content representation collapses training or hurts within-video ordering; replacing top-k MIL with a sequence-model objective is worse.

## Constraints from the project owner

- Method paper for a top venue; performance passes the SOTA gate, novelty is the problem. Gains may be small but must be caused by design and shown by ablation (three-seed mean drop >= .01 on both corpora).
- No ensembles, no inference post-processing, one identical method for both corpora, few method-level hyperparameters (currently three: alpha, w_fine, lambda_block; the owner may drop w_fine).
- Three modules (VLM verdict / backbone / fusion); backbone and fusion must each carry a claimable novelty.
- VLM cost matters: 34 calls per video is the budget; proposals multiplying calls per window were stopped. The owner is interested in adaptive coarse-to-fine granularity (only refine blocks the coarse pass flags, or skip the VLM where the backbone is confident) as a way to reduce cost, and in whether the fixed 30/4 grid can be replaced by something principled.
- Compute: three RTX 5090s; one 20-trial search per corpus-seed takes 1-3 hours; a three-seed confirmation plus 12-arm ablations for one variant takes about half a day.

## What we want from you

1. Which of the previous review's criticisms does candidate 3 answer, and which remain? Be specific: does "evidence enters attention only through q/k, key bias and a video-level context, with value and residual kept pure content" count as a backbone contribution distinct from evidence-conditioned attention in prior work (e.g. VadCLIP / prompt-conditioned attention, GIG-VAD / PEL4VAD video-level context, relative position or ALiBi-style key biases, Snorkel/CHMM label models feeding a network)? Name the closest prior work for each of A, B, C and say what is and is not new.
2. Can the method claim an underlying mechanism or paradigm (for example: "evidence decides where content is aggregated from, but never what the content representation is")? Is that claim supported by the ablations on disk, in particular by the asymmetry that q/k encoding matters on HateMM but not on HateClipSeg, and by within-video ROC falling while pooled rises? Say what additional analysis (no new training, or at most one cheap run) would make the claim credible or refute it.
3. Which parts of the evidence would a reviewer attack? Include: the per-seed inconsistencies above, the HateClipSeg baselines near chance, the training-free posterior nearly matching the trained model on HateClipSeg, the fixed 30/4 grid, the three method-level scalars, and the within-video ROC drop on HateMM.
4. Propose 3 to 5 concrete areas for improvement ranked by (novelty or defensibility gained) / (cost), each with the exact mechanism, why it differs from the closest prior work, the ablation that proves it, the risk of repeating a failed candidate, and compute cost in the units above. Distinguish analysis that needs no new search from mechanisms that need one. Include your assessment of the owner's adaptive-granularity / VLM-cost direction.
5. A mock review (summary, strengths, weaknesses, questions, score 1-10, confidence) of candidate 3 if submitted today with no changes.

Write in plain language; no metaphors. Be adversarial and specific; cite file paths you actually opened. Chinese output preferred.
