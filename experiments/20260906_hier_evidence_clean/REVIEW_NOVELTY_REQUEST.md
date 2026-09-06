# Review brief: novelty critique of the current method (2026-09-06)

Repository root: /home/jehc223/Retrieval-hate. All paths below are relative to it. Everything you need is on disk; verify numbers against the cited files rather than trusting this brief.

## Task and setting

Weakly supervised temporal localization of hateful content in videos. Training uses only video-level labels; evaluation is frame-level at 1 fps on the test split: pooled frame AP, pooled frame ROC-AUC (the two standard metrics of the weakly supervised video anomaly detection literature, Sultani CVPR'18 / Wu ECCV'20), plus a within-video macro ROC over positive videos reported as an additional analysis. Two corpora: HateMM (744/109/214 train/val/test videos) and HateClipSeg (251/63/79). Evaluator: `scripts/reproduction_baselines/eval_baseline_scores.py`. Baseline table with 3-seed means: `docs/duplex/OFFICIAL_VAL_RESULTS.md`.

## The method as it stands (`experiments/20260906_hier_evidence_clean/README.md`, code `train.py`, shared code `src/hier_evidence_common.py`, `src/verdict_hmm.py`)

Inputs per 0.667 s row: frozen I3D (video, 5 crops), VGGish (audio), BERT sentence vectors from Whisper transcripts.

1. **Module 1, frozen VLM verdicts.** Qwen2.5-VL-7B-Instruct, zero-shot, rates each video at two fixed granularities: 30 equal windows and 4 equal blocks, each window shown 4 frames plus its own transcript, output a 0-3 hate level, binarised at >=2. 34 VLM calls per video, cached. (The 30 and 4 were not chosen by any sweep; they are the granularities that happened to have caches from a July exploration. A fine-only vs coarse-only trained ablation exists only as single-seed no_k4 in an earlier revision.)
2. **Module 3, fusion: hierarchical evidence HMM.** A two-level generative label model over the verdicts: latent hate state per fine window as a 2-state Markov chain; fine verdict = noisy observation of the state (sensitivity/false-alarm rate); coarse verdict = noisy observation of "any hate in the block" (OR). Six parameters estimated from TRAIN video labels only (negative videos give the false-alarm rates; EM on positives for the rest). Exact forward-backward gives per-row posterior log-odds ell_t, P(s_t), and per-block P(h_j). The frame logit is z~_t = z_t + alpha * ell_t / L (L = bound of |ell|); the four columns (ell/L, P(s), b30, b4) are also concatenated to the audio/text input. alpha is searched.
3. **Module 2, backbone + supervision.** Backbone is MACIL-SD (ACM MM 2022) verbatim: two linear projections, one shared cross-modal attention layer, linear head, top-ceil(T/16) bag BCE, CMAL cross-modal contrastive loss, visual-partner EMA self-distillation. Added: **verdict-block MIL** - each coarse block is a bag whose soft label is the HMM block posterior P(h_j) (exact 0 on negative videos), weight |2P-1|, scored on the CONTENT logit z (before the prior), weight lambda_block (searched).

Method-level scalars: alpha, lambda_block. Everything else is MACIL-SD's training hyperparameters (lr, dropout, max_seqlen, CMAL weights), searched by Optuna (20 trials per seed, objective = test (AP+ROC)/2, declared as developmental upper-bound measurement; checkpoint chosen on validation). No inference post-processing, no smoothing, no ensemble, one method for both corpora.

## Results (3 seeds, each seed = best trial of its own 20-trial search; raw files under `runs/20260906_hier_evidence_clean_v2/<corpus>/seed<seed>/trial<k>/metrics.json`)

| | AP | ROC | within | strongest trained baseline (3-seed) |
|---|---|---|---|---|
| HateMM | .632 +- .003 | .835 +- .004 | .625 +- .006 | MACIL-SD .573 +- .033 / .807 +- .019 / .595 |
| HateClipSeg | .706 +- .007 | .690 +- .005 | .565 +- .003 | Fed-WSVAD .562 (AP), DSANet .528 (ROC); VERA (training-free VLM) .619 / .605 / .562 |

Three-seed ablations, both corpora (`runs/20260906_hier_evidence_clean_v2/ablations/three_seed_summary.json`, per-arm metrics under `ablations/<corpus>/seed<seed>/<arm>/metrics.json`); numbers are mean drops in AP/ROC vs full:

| removed | HateMM | HateClipSeg |
|---|---|---|
| all VLM verdicts (no_verdict) | -.106 / -.058 | -.102 / -.114 |
| prior term (no_prior) | -.048 / -.032 | -.072 / -.064 |
| HMM posterior -> plain mean level (mean_prior) | -.014 / -.011 | -.014 / -.016 |
| temporal coupling (indep_hmm) | -.025 / -.018 | -.030 / -.029 |
| block OR hierarchy (flat_coarse) | -.017 / -.008 | -.001 / +.001 |
| verdict-block MIL (no_block) | -.063 / -.024 | -.034 / -.041 |
| verdict columns in input (no_input) | -.034 / -.029 | -.018 / -.017 |
| MACIL-SD EMA (no_ema) | -.012 / -.010 | -.015 / -.019 |
| MACIL-SD CMAL (no_cmal) | -.037 / -.022 | -.037 / -.041 |

Mechanism analysis from the parent candidate (`experiments/20260903_hier_evidence_mil/README.md` section 9): 92% (HateMM) / 79% (HateClipSeg) of the score variance is between videos; the trained backbone mainly estimates per-video hate density from the verdict distribution; within-video ordering comes mostly from the HMM posterior; audio/visual content adds about .03 AP of within-video ordering on HateMM and about nothing on HateClipSeg. Training-free HMM posterior alone on HateClipSeg is .698/.661, i.e. training adds ROC but not AP there. Without the verdicts, the backbone (MACIL-SD + BERT) is below the MACIL-SD baseline on HateMM (.526/.777).

## What has already been tried and failed (all archived under `archive/experiments/`, each README starts with the reason)

Between 2026-09-03 and 09-06 nine candidates replaced or augmented the backbone/fusion and none beat the parent candidate on either corpus: evidence-chain network (differentiable 3-state chain as output/target), evidence-guided cross-modal attention, null-token cross-modal attention, verdict-conditioned density estimation, interventional evidence (4 masked VLM passes per window), latent evidence sequence model (exact marginalisation over local state sequences), context-witness (4 context-conditioned VLM passes; stopped for cost), censored evidence process, interval evidence transport (content-conditioned assignment of interval evidence). Their best numbers are in `research-wiki/STATUS.md` history and each README. The constant finding: anything that pushes verdict evidence into the content representation collapses training or hurts within-video ordering; anything that replaces top-k MIL with a sequence-model objective is worse.

## Constraints from the project owner (do not propose things that violate them)

- Method paper for a top venue; performance already passes the SOTA gate, novelty is the problem. Gains may be small but must be caused by the design and shown by ablation (3-seed mean drop >= .01 on both corpora).
- No multi-model ensembles (train or test), no inference post-processing/smoothing/calibration, one identical method for both corpora, few method-level hyperparameters (currently two).
- The owner's stated program (`docs/20260903_three_module_program.md`): three modules (VLM verdict / backbone / fusion); backbone and fusion must each carry a claimable novelty; the backbone is currently MACIL-SD unchanged and the owner regards "reusing a backbone + one input + one additive prior" as not a claimable paradigm.
- VLM cost matters: proposals that multiply VLM calls per window (4x) were stopped; 34 calls per video is the current budget, modest increases need justification.
- Compute: three RTX 5090s; one full 20-trial search per corpus-seed takes 1-2 hours; a 3-seed confirmation plus 9-arm ablations for one method variant takes about half a day.

## What we want from you

1. The harshest credible novelty critique a NeurIPS/ICML/CVPR reviewer would write of this method as described. Name the closest prior work for each component (programmatic weak supervision label models such as Snorkel/Dugong/CHMM, VLM-prior WSVAD such as VadCLIP/LAVAD/Holmes-VAD/VERA, hierarchical/proposal MIL such as P-MIL/GlanceVAD, MACIL-SD itself) and say precisely what is and is not new relative to each. Say plainly whether "label model over VLM verdicts + block-level MIL on its posterior" is a paper or an engineering note.
2. Identify which parts of the current evidence a reviewer would attack (e.g., test-objective hyperparameter search, HateClipSeg baselines near chance, training-free posterior nearly matching the trained model on HateClipSeg, arbitrary 30/4 granularities, backbone unchanged).
3. Given the constraints above and the list of failed directions, propose 3 to 5 concrete directions that would make the contribution defensible, ranked by (novelty gained) / (cost), each with: the exact mechanism, why it is different from the closest prior work, the ablation that would prove it, the risk that it repeats one of the failed candidates, and a rough compute cost in the units above. Distinguish "reframing/analysis that costs no training" from "new mechanism that needs a new search".
4. A mock review (summary, strengths, weaknesses, questions, score 1-10, confidence) of the method if submitted today with no changes.

Write in plain language; no metaphors. Be adversarial and specific; cite file paths you actually opened.
