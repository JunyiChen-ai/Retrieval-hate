---
type: experiment
node_id: exp:exp-consensus-zh-seeds
title: "Multi-seed verification of the ZH consensus win (NEGATIVE on beat-floor, POSITIVE on repair)"
idea_id: "idea:retrieval-consensus-denoising"
verdict: partial
confidence: high
date: "2026-07-04"
hardware: "1x A100 per job, ~2-18 min each"
duration: "12 jobs (12289-12300)"
provenance: "slurm/logs/consseed_MHC_zh_*_{12289..12300}.trainlog; seed-0 carried from mhc_train_cons_12179/12180.out + mhc_train_seg_12130.out; ckpt group logging/Retrieval/MHC_zh/RAC_video_consensus_seeds/"
added: 2026-07-04T00:00:00Z
tags: ["hateful-video", "consensus-denoising", "multi-seed", "MHC_zh", "replication", "NEGATIVE-beat-floor", "POSITIVE-repair", "wave-1"]
---

# Multi-seed verification of the ZH consensus win (NEGATIVE on beat-floor, POSITIVE on repair)

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:retrieval-consensus-denoising`
(replication of `exp:exp-consensus-kill-ablation`'s ZH cell)

## TL;DR

**共识修复毒化成立、反超 floor 不成立。** Across 5 seeds the seed-0 headline
(consensus +0.0158 F1 over the λ=0 CLIP floor) becomes **+0.0115 ± 0.0418 F1,
3/5 wins, paired t p=0.57** under the registered val-selection protocol — a
coin flip. Direction is preserved in every aggregate (consensus has the best
mean F1 of the three arms under BOTH criteria), and the final-epoch view is
stronger (+0.0247 ± 0.0272, 4/5 wins, p=0.11) but still not separable from
seed noise at n=5. The claim that survives is the *repair* claim: sub-clip
supervision with consensus labels never reproduces the Phase-3 −0.066
inherited-label poisoning, under any seed or criterion. Do NOT write
"consensus beats the ZH floor" in the paper; write "consensus de-poisons
sub-clip supervision (−0.066 → ≈ floor / weakly above)".

## Protocol

Exact replica of the job-12179 command (Namespace in
`slurm/logs/mhc_train_cons_12179.out`) with **only `--seed` varied** — frozen
CLIP ViT-L/14-336, MHC_zh, RGCL head, 30 epochs, λ_seg=0.5, K=4, consensus
topk=10 / τ=0.2 / EM=2 / drift kept / conflict=ignore, `--Faiss_GPU False`,
FORCE=False, fresh GROUP `RAC_video_consensus_seeds`. No `--consensus_space`
flag passed: the default `clip` is the verified pre-W5 bit-for-bit control
path in post-W5 `src/utils/consensus.py` (code-diff checked before launch).
Floor arm = same command with λ_seg=0 (bit-for-bit whole-video baseline,
verified in the kill ablation). Selection = registered protocol: warmup-floored
(epoch ≥ 5) max `Val_Retrieval acc`, tie-break roc, last-EM-round line per
epoch; second criterion = final epoch (29). Seed-0 numbers carried from jobs
12179 (consensus), 12180 (selfscore), 12130 (floor); parser reproduces the
registered seed-0 numbers exactly.

Jobs: consensus s1-4 = 12289-12292 · selfscore s1-4 = 12293-12296 ·
floor s1-4 = 12297-12300. Runner: `scripts/slurm/train_consensus_seeds.sbatch`.
(The LoRA-base `RAC_video_archive_seeds` floor could NOT be reused — consensus
is CLIP-base; CLIP floor existed only at seed 0, hence the third arm.)

## Per-seed results (Test macroF1 / acc)

| arm | seed | selEp | sel F1 | sel acc | fin F1 | fin acc |
|---|---|---|---|---|---|---|
| floor | 0 | 29 | 0.7706 | 0.8054 | 0.7706 | 0.8054 |
| floor | 1 | 28 | 0.7579 | 0.8054 | 0.7542 | 0.8054 |
| floor | 2 | 25 | 0.7742 | 0.8121 | 0.7913 | 0.8322 |
| floor | 3 | 16 | 0.7421 | 0.7785 | 0.7548 | 0.7987 |
| floor | 4 | 23 | 0.7799 | 0.8121 | 0.7259 | 0.7718 |
| selfscore | 0 | 29 | 0.7746 | 0.8188 | 0.7746 | 0.8188 |
| selfscore | 1 | 26 | 0.7608 | 0.8121 | 0.7845 | 0.8255 |
| selfscore | 2 | 27 | 0.7675 | 0.8188 | 0.7351 | 0.7987 |
| selfscore | 3 | 29 | 0.7845 | 0.8255 | 0.7845 | 0.8255 |
| selfscore | 4 | 12 | 0.7073 | 0.7517 | 0.7904 | 0.8255 |
| consensus | 0 | 23 | 0.7864 | 0.8188 | 0.7890 | 0.8188 |
| consensus | 1 | 10 | 0.7073 | 0.7517 | 0.7513 | 0.7987 |
| consensus | 2 | 26 | 0.8090 | 0.8389 | 0.8046 | 0.8322 |
| consensus | 3 | 26 | 0.8023 | 0.8322 | 0.7799 | 0.8121 |
| consensus | 4 | 22 | 0.7771 | 0.8121 | 0.7956 | 0.8255 |

## Aggregates (n=5 seeds)

| criterion | arm | Test F1 (mean±std) | Test acc (mean±std) |
|---|---|---|---|
| val-selected | floor | 0.7649 ± 0.0151 | 0.8027 ± 0.0139 |
| val-selected | selfscore | 0.7589 ± 0.0302 | 0.8054 ± 0.0304 |
| val-selected | **consensus** | **0.7764 ± 0.0406** | **0.8107 ± 0.0347** |
| final-epoch | floor | 0.7594 ± 0.0240 | 0.8027 ± 0.0215 |
| final-epoch | selfscore | 0.7738 ± 0.0224 | 0.8188 ± 0.0116 |
| final-epoch | **consensus** | **0.7841 ± 0.0204** | 0.8175 ± 0.0129 |

## Same-seed paired deltas

| comparison | criterion | metric | Δ mean±std | wins | paired-t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| consensus − floor | val-sel | F1 | **+0.0115 ± 0.0418** | 3/5 | 0.572 | 0.625 |
| consensus − floor | val-sel | acc | +0.0080 ± 0.0398 | 3/5 (1 tie) | 0.675 | 0.750 |
| consensus − floor | final | F1 | **+0.0247 ± 0.0272** | 4/5 | 0.112 | 0.125 |
| consensus − floor | final | acc | +0.0148 ± 0.0234 | 3/5 (1 tie) | 0.232 | 0.250 |
| consensus − selfscore | val-sel | F1 | +0.0175 ± 0.0458 | 4/5 | 0.441 | 0.438 |
| consensus − selfscore | final | F1 | +0.0103 ± 0.0376 | 3/5 | 0.575 | 0.625 |
| selfscore − floor | val-sel | F1 | −0.0060 ± 0.0417 | 3/5 | 0.764 | 1.000 |
| selfscore − floor | final | F1 | +0.0145 ± 0.0450 | 4/5 | 0.512 | 0.438 |

Per-seed consensus−floor F1 (val-sel): s0 +0.0158, s1 **−0.0506**, s2 +0.0348,
s3 +0.0602, s4 −0.0028. (final): s0 +0.0184, s1 −0.0029, s2 +0.0133,
s3 +0.0251, s4 +0.0697.

## Judgments (pre-registered)

1. **Consensus vs floor ("反超 floor"): NOT ESTABLISHED.** Val-selected
   (registered protocol): +0.0115 ± 0.0418, 3/5 wins, p≈0.57 — indistinguishable
   from zero; one seed (s1) flips hard negative (−0.051). Final-epoch is more
   consistent (+0.0247 ± 0.0272, 4/5, p≈0.11) but still short of significance
   at n=5. The seed-0 +0.0158 was within seed noise, as feared.
2. **Consensus vs selfscore: NOT ESTABLISHED** (val-sel +0.0175, 4/5, p≈0.44;
   final +0.0103, 3/5, p≈0.58; final acc actually −0.0013). Consistent with the
   kill-ablation caveat that the consensus-vs-selfscore ranking was inside the
   noise band. The robust statement remains "both denoisers ≈ floor, ranking
   unresolved" — NOT "consensus > selfscore".
3. **Repair of the Phase-3 hole ("修复毒化"): HOLDS.** full(inherit-labels,
   λ=0.5, seed 0) = 0.7050 F1 = −0.0656 vs floor. Nothing in the consensus arm
   ever approaches that regime: worst paired delta anywhere is s1 val-sel
   −0.0506 (a val-selection artifact — the same run is −0.0029 at final epoch),
   and the consensus arm mean ≥ floor mean under both criteria and both
   metrics. Sub-clip supervision with consensus pseudo-labels is safe where
   inherited labels poison. (Caveat: the hole itself is still single-seed —
   the full-λ0.5 arm was not re-seeded here.)

## Mechanism note

The dominant noise source is **val-selection on the 78-clip ZH val set**, not
training: consensus s1 selects epoch 10 (test F1 0.7073) while its final epoch
is 0.7513; floor s4 selects 0.7799 but ends at 0.7259. Final-epoch paired
deltas are one-signed for consensus−floor in 4/5 seeds. If a robustified
selection rule (e.g., val-F1 smoothing or last-k averaging) were registered
*before* looking at test, the comparison could tighten — that would be a new
pre-registration, not a rescue of this one.

## Caveats

- n=5 seeds, 149-clip test set: the MDE of a paired test here is ~0.04-0.05 F1;
  a true +0.01-0.02 effect is undetectable by design. Absence of significance
  is not evidence of absence — but it IS absence of a paper-grade claim.
- Seed-0 arms came from pre-W5 code (jobs 12179/12180/12130), seeds 1-4 from
  post-W5 HEAD (26dc558) with `consensus_space=clip` default; the control path
  was code-diff-verified identical, and floor λ=0 was previously verified
  bit-for-bit through the seg code path.
- Floor arm exp_comment differs from the Phase-3 floor (`_seg0_full` vs
  `_seg0`) — naming only, command-identical.
- selfscore s4 val-sel (epoch 12, 0.7073) is the same selection pathology as
  consensus s1; it inflates the consensus−selfscore val-sel delta (+0.0698 on
  that seed). Do not quote that comparison in isolation.

## Consequences for the paper

- The ONLY method-level ZH claims this experiment licenses:
  (a) inherited-label sub-clip supervision poisons ZH (single-seed, large:
  −0.066); (b) retrieval-consensus relabeling removes that poisoning and lands
  at-or-slightly-above the whole-video floor across 5 seeds (never below it on
  mean). Frame consensus as a *denoising/repair* mechanism with the drift-role
  diagnostic (41.7% toxic-positive demotion), not as an accuracy win.
- The consolidated paper delta (archive-kNN +2/4-seed wins, updatable memory)
  is unaffected and remains the headline; consensus drops from "唯一方法级涨点"
  to a supporting robustness/mechanism section.

## Connections
- replicates → `exp:exp-consensus-kill-ablation` (ZH cell only; EN hard fail
  already established there)
- informs → `idea:retrieval-consensus-denoising` (verdict downgraded:
  repair-yes / beat-floor-no)
