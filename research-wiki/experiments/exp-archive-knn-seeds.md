---
type: experiment
node_id: exp:exp-archive-knn-seeds
title: "Multi-seed robustness check of the two winning archive-kNN configs (ZH LoRA + EN frozen, alpha=0.25) + paired LoRA-only control"
idea_id: ""
verdict: no
confidence: high
date: "2026-07-04"
hardware: "1x A100 per job (SLURM)"
duration: "~25 min/run x 12 runs"
provenance: "slurm/logs/arcSeed_zh_s{1..4}_{12215..12218}.out; slurm/logs/arcSeed_en_s{1..3}_{12219..12221}.out; slurm/logs/arcBase_zh_s{0..4}_{12223..12227}.out; seed-0 refs: arc_MHC_zh_knn0.25_12207.out, arc_MHC_knnB_0.25_12210.out, mllm_train_12149.out; EN-floor refs: mllm_train_12113.out + slurm/logs/arcBase_en_s{1..3}_{12275..12277}.out; scripts/slurm/train_archive.sbatch; scripts/slurm/train_archive_baseline.sbatch; scripts/analysis/selection_rule_robustness.py"
added: 2026-07-04T00:00:00Z
tags: ["hateful-video", "MLLM-archive", "knn-memory", "multi-seed", "robustness", "MHC_zh", "MHC", "negative-result", "iteration-3"]
---

# Multi-seed robustness check: archive-kNN alpha=0.25 winners (ZH + EN) + paired LoRA-only control

**verdict:** `no` (neither headline claim survives multi-seed) · **confidence:** `high`

## Setup

Exact replicas of the winning command lines (verified against the full `Namespace` in the
seed-0 logs), changing ONLY `--seed`. Protocol unchanged and pre-registered: val-selected
epoch >= warmup 5 by Val_Retrieval acc, roc tie-break. `GROUP=RAC_video_archive_seeds`
(fresh dir), `FORCE=False`. All runs `--epochs 30 --lr 1e-4 --batch_size 64`.

- **ZH archive arm**: `MHC_zh` + `Qwen2.5-VL-7B-Instruct-LoRA_HF` + `--archive_feats auto
  --archive_mode knn --archive_alpha 0.25` (= job 12207). Seeds 1-4 = jobs 12215-12218.
- **EN archive arm**: `MHC` + `Qwen2.5-VL-7B-Instruct_HF` (frozen) + same archive flags
  (= job 12210). Seeds 1-3 = jobs 12219-12221.
- **ZH LoRA-only control** (coordinator-requested): identical command with the archive flag
  line removed entirely (`archive_feats=None` = bit-for-bit OFF path in `src/run_rac.py`;
  no string off-value exists). New template `scripts/slurm/train_archive_baseline.sbatch`.
  Seeds 0-4 = jobs 12223-12227.
- **Control validity check**: baseline seed-0 rerun (12223) reproduced job 12149
  **bit-for-bit** (epoch 20; 0.8023/0.8322/0.8825) despite intervening src edits by W5 —
  confirms all new W5 behavior is flag-gated OFF and the paired comparison is same-code.

## Per-seed results (val-selected Test)

### MHC_zh — LoRA-Qwen + archive-kNN alpha 0.25

| seed | job | sel. epoch | macroF1 | acc | roc |
|---|---|---|---|---|---|
| 0 | 12207 | 18 | 0.8270 | 0.8523 | 0.9107 |
| 1 | 12215 | 23 | 0.8158 | 0.8456 | 0.9130 |
| 2 | 12216 | 14 | 0.8046 | 0.8322 | 0.9041 |
| 3 | 12217 | 17 | 0.7837 | 0.8188 | 0.8904 |
| 4 | 12218 | 12 | 0.7266 | 0.7852 | 0.9130 |
| **mean±std** | | | **0.7915 ± 0.0397** | **0.8268 ± 0.0266** | 0.9062 ± 0.0096 |

### MHC_zh — LoRA-Qwen, archive OFF (LoRA-only control)

| seed | job | sel. epoch | macroF1 | acc | roc |
|---|---|---|---|---|---|
| 0 | 12223 (=12149) | 20 | 0.8023 | 0.8322 | 0.8825 |
| 1 | 12224 | 26 | 0.7956 | 0.8255 | 0.9004 |
| 2 | 12225 | 19 | 0.8065 | 0.8389 | 0.8838 |
| 3 | 12226 | 17 | 0.7677 | 0.8054 | 0.9068 |
| 4 | 12227 | 22 | 0.8090 | 0.8389 | 0.9100 |
| **mean±std** | | | **0.7962 ± 0.0167** | **0.8282 ± 0.0139** | 0.8967 ± 0.0129 |

### MHC (EN) — frozen-Qwen + archive-kNN alpha 0.25

| seed | job | sel. epoch | macroF1 | acc | roc |
|---|---|---|---|---|---|
| 0 | 12210 | 24 | 0.7626 | 0.8075 | 0.8489 |
| 1 | 12219 | 29 | 0.7145 | 0.7640 | 0.8251 |
| 2 | 12220 | 21 | 0.7505 | 0.7950 | 0.8437 |
| 3 | 12221 | 27 | 0.7713 | 0.8075 | 0.8358 |
| **mean±std** | | | **0.7497 ± 0.0250** | **0.7935 ± 0.0205** | 0.8384 ± 0.0104 |

## Paired same-seed archive gain (ZH, archive − LoRA-only)

| seed | dAcc | dMacroF1 | dROC |
|---|---|---|---|
| 0 | +0.0201 | +0.0247 | +0.0282 |
| 1 | +0.0201 | +0.0202 | +0.0126 |
| 2 | −0.0067 | −0.0019 | +0.0203 |
| 3 | +0.0134 | +0.0160 | −0.0164 |
| 4 | **−0.0537** | **−0.0824** | +0.0030 |
| **mean±std** | **−0.0014 ± 0.0313** (t=−0.10) | **−0.0047 ± 0.0446** (t=−0.23) | **+0.0095 ± 0.0172** (t=+1.24) |

Positive on 3/5 seeds for acc/F1, 4/5 for roc; paired t (n=5) nowhere near significance
for any metric.

## Verdict (honest)

1. **ZH >=0.85 claim: FAILS.** 0.8523 was a single-seed lucky high — the best of 5 seeds.
   Mean acc = **0.8268 ± 0.0266**; only seed 0 crosses 0.85; worst seed = 0.7852 (< 0.83
   criterion also fails). MHClip-ZH 0.85 remains **OPEN**.
2. **ZH archive-kNN gain over LoRA-only: NOT ESTABLISHED.** Mean paired dAcc =
   **−0.0014 ± 0.0313** (archive mean 0.8268 vs LoRA-only 0.8282 — archive is nominally
   *lower* and 2x noisier). The +0.02 seen at seeds 0/1 is real at those seeds but is wiped
   out by the seed-4 collapse (archive run val-selected epoch 12, acc 0.7852 vs baseline
   0.8389). At n=5 the honest statement is: **no reliable accuracy gain; direction unknown**.
3. **Secondary observation**: paired dROC = +0.0095, positive on 4/5 seeds — weak evidence
   the archive-kNN keys improve *ranking* while val-selected thresholding/epoch-selection
   noise erases the accuracy benefit. Worth one analysis paragraph, not a claim.
4. **EN 0.8075: NOT a stable point estimate.** Mean acc = **0.7935 ± 0.0205**; 0.8075 is
   the max, hit by 2/4 seeds (min 0.7640). Report EN as 0.794 ± 0.021.

## Addendum: selection-rule robustness re-analysis (zero GPU, 2026-07-04)

Motivation: the suspected lesion is epoch selection by val-acc on the 78-sample ZH dev set.
All arms re-scored from the existing trainlogs under four selection rules
(`scripts/analysis/selection_rule_robustness.py`; warmup >= 5 throughout):

- **(a) val-acc** (roc tie-break) — pre-registered rule, control;
- **(b) val-ROC** max (acc tie-break);
- **(c) top3-mean** — mean Test over the 3 best epochs by val-acc (smoothed selection);
- **(d) last5-mean** — mean Test over the final 5 epochs (no-selection reference).

Extra reference arm: EN frozen-Qwen floor (no archive), single seed, job 12113.

### Test acc, mean±std per rule

| arm | (a) val-acc | (b) val-ROC | (c) top3-mean | (d) last5-mean |
|---|---|---|---|---|
| ZH archive (n=5) | 0.8268 ± 0.0266 | 0.8282 ± 0.0295 | 0.8327 ± 0.0130 | 0.8459 ± 0.0105 |
| ZH LoRA-only (n=5) | 0.8282 ± 0.0139 | 0.8416 ± 0.0102 | 0.8246 ± 0.0072 | **0.8475 ± 0.0122** |
| EN archive (n=4) | 0.7935 ± 0.0205 | 0.7919 ± 0.0212 | 0.7919 ± 0.0202 | 0.7870 ± 0.0175 |
| EN floor (n=1) | 0.7888 | 0.8075 | 0.7598 | 0.8087 |
| **ZH paired dAcc** | −0.0014 ± 0.0313 (+3/5) | −0.0134 ± 0.0277 (+2/5) | **+0.0080 ± 0.0121 (+4/5, t=+1.48)** | −0.0016 ± 0.0042 (+2/5) |

(macro-F1 tracks acc in every cell; see script output. ZH paired dF1 mean per rule:
a −0.0047, b −0.0238, c +0.0062, d −0.0020.)

### Answers

1. **No rule makes the archive gain direction-consistent.** The friendliest is (c) top3-mean
   (+4/5 seeds, +0.0080 ± 0.0121, t=+1.48, n.s.); the pre-registered (a) is ~0; (b) val-ROC
   makes it *more negative* (−0.0134); the no-selection reference (d) is the tightest
   estimate and is centered at zero (−0.0016 ± 0.0042). Under the least-noisy estimator the
   ZH archive effect is indistinguishable from zero.
2. **Highest ZH mean: 0.8475 ± 0.0122 — the LoRA-only baseline under (d) last5-mean**
   (archive arm: 0.8459 ± 0.0105 under the same rule). No arm reaches a 0.85 *mean* under
   any rule; 0.85 stays open regardless of selection procedure.
3. **What the flips mean.** Across defensible rules the ZH paired effect spans −0.013 to
   +0.008 — the choice of selection rule moves the estimate *more than the treatment
   itself*. Moreover both ZH arms sit at ~0.846-0.848 under last5-mean vs ~0.827-0.828
   under val-acc selection: val-acc selection on a 78-sample dev *costs* ~2 acc points
   relative to simply averaging late epochs. The apparent seed-0 archive gain (+0.020) was
   therefore a val-selection artifact, not a model-quality difference — the two arms'
   final-model quality is equal to within ±0.004. Suggestive pattern (likely noise-fitting,
   flagged as such): rules that help the archive arm (c) hurt the baseline, and vice versa
   (b).
4. **Pre-registration revision candidate: NOT proposed.** val-ROC has the right prior
   (ROC more stable than acc on small dev sets) and does reduce baseline variance
   (±0.0102) while lifting ZH baseline to 0.8416, but it is not uniformly dominant: it
   degrades both archive arms (more variance ZH, lower mean EN) and flips signs elsewhere.
   Per the pre-agreed criterion (consistent dominance across all arms), no rule qualifies
   as a headline replacement. If anything is pre-registered *for future runs*, the
   candidates are val-ROC or top3-style smoothing — decision left to the user; nothing in
   this node's headline numbers changes (rule (a) remains the reported protocol).

### Addendum 2: final-epoch (no-selection) protocol + weight-identity audit (2026-07-04)

**(e) final-epoch rule** — Test at the last trained epoch (29), the most standard
selection-free protocol (zero GPU, reparsed from the same trainlogs):

| arm | final-epoch acc | final-epoch macroF1 | per-seed acc |
|---|---|---|---|
| ZH archive (n=5) | **0.8537 ± 0.0120** | 0.8259 ± 0.0124 | 0.8456 / 0.8389 / 0.8523 / 0.8658 / 0.8658 |
| ZH LoRA-only (n=5) | **0.8537 ± 0.0120** | 0.8259 ± 0.0124 | *identical to archive arm, every seed* |
| EN archive (n=4) | 0.7826 ± 0.0134 | 0.7430 ± 0.0196 | 0.7888 / 0.7640 / 0.7826 / 0.7950 |
| EN floor (n=1) | 0.8012 | 0.7596 | — |

**Weight-identity audit (why the ZH rows are identical).** `archive_mode=knn` by design
does not touch training ("Training is IDENTICAL to baseline", run_rac.py help; the bank is
used only inside `evaluate_rac.retrieve_evaluate_RAC_`). Verified empirically: the local
ZH LoRA-only seed-4 epoch-29 checkpoint's sha1 (`6d6551e4593770223df985f3a9aa6bbc995a11c3`)
exactly matches the sha1 disk_guard recorded when it pushed the (since-pruned) ZH *archive*
seed-4 epoch-29 checkpoint to B2 — same-seed runs produce **byte-identical weights**. The
whole archive-kNN effect therefore lives in the eval-time retrieval keys, and at epoch 29
the alpha=0.25 key augmentation (6% similarity weight) flips **zero** test votes on all
5 ZH seeds (acc/F1 exactly equal). It only perturbs scores: final-epoch paired
dROC = −0.0050 ± 0.0102, positive on 1/5 seeds — the "+dROC on 4/5 seeds" seen under
val-acc selection does NOT survive the no-selection protocol; even the ranking benefit is
selection-epoch-dependent, not a property of the final model.

**Final-epoch observations (honest framing):**
- ZH final-epoch mean is **0.8537 ± 0.0120 — the only protocol whose ZH mean crosses
  0.85** (seeds 3/4 reach 0.8658). It also beats every val-selection variant, confirming
  that val selection on the 78-video dev is net harmful (~2 acc points).
- CAVEAT: adopting final-epoch as headline *because* it crosses 0.85 would be post-hoc
  rule-shopping. It is a legitimate, standard, selection-free protocol and a reasonable
  candidate for *future pre-registration* — decision left to the user. Under it the
  archive-kNN channel contributes exactly nothing on ZH (identical numbers) and sits below
  the single-seed floor on EN (0.7826 ± 0.0134 vs 0.8012).
- A 5-seed cross-seed ensemble line was briefly planned and then **withdrawn per user rule
  (no cross-seed ensembles in the method)**: no ensemble jobs were run, no ensemble script
  exists, no ensemble numbers were produced.

### Addendum 3: EN main-table draft — floor vs archive, both protocols (2026-07-04)

EN frozen-Qwen floor extended to 4 seeds (jobs 12275/12276/12277 = seeds 1/2/3 via
`train_archive_baseline.sbatch`, GROUP=RAC_video_archive_seeds; seed 0 = job 12113,
justified by the bit-for-bit archive-OFF reproduction shown in Addendum 2). Transcript arm
is owned by another agent and will be merged into this table separately.

**MHC (EN), frozen Qwen2.5-VL-7B embedding — Test acc / macroF1, mean±std over 4 seeds:**

| arm | (a) val-acc 选点 acc | (a) F1 | (e) final-epoch acc | (e) F1 |
|---|---|---|---|---|
| floor (no keys) | 0.7702 ± 0.0221 | 0.7010 ± 0.0448 | 0.7888 ± 0.0152 | 0.7488 ± 0.0208 |
| + archive-kNN α0.25 | **0.7935 ± 0.0205** | **0.7497 ± 0.0250** | 0.7826 ± 0.0134 | 0.7430 ± 0.0196 |
| + transcript (other agent) | *tbd* | *tbd* | *tbd* | *tbd* |
| **paired Δ (arc − floor)** | +0.0233 ± 0.0357 (+3/4, t=+1.31) | +0.0487 ± 0.0622 (+3/4) | **−0.0062 ± 0.0051 (0/4, t=−2.45)** | −0.0059 ± 0.0042 (0/4) |

Per-seed val-acc acc (floor | archive): s0 0.7888@e28 | 0.8075@e24 · s1 0.7826@e25 |
0.7640@e29 · s2 0.7702@e18 | 0.7950@e21 · s3 **0.7391@e6** | 0.8075@e27.
Per-seed final-epoch acc: floor 0.8012/0.7702/0.7826/0.8012 · archive 0.7888/0.7640/0.7826/0.7950.

**Reading (honest):**
- EN mirrors the ZH lesson. Under the pre-registered val-acc rule the archive arm looks
  +2.3 acc points better, but a large share of that gap comes from one floor seed's
  pathological selection (s3 picked epoch 6 → 0.7391, while its final-epoch is 0.8012).
  Under the selection-free final-epoch protocol — where same-seed weights are byte-identical
  and only the retrieval keys differ — the archive keys flip 0-2 test votes per seed and
  never upward: Δ = −0.0062 ± 0.0051, positive on 0/4 seeds (t=−2.45, p≈0.09 at n=4). Weak
  but directionally consistent evidence that the α=0.25 keys are zero-to-slightly-negative
  for EN final-model accuracy; the val-selected "gain" is a selection interaction, not
  model quality.
- No EN configuration separates from the pack: all cells sit in 0.77-0.79. The best cells
  (archive@val-acc 0.7935 vs floor@final-epoch 0.7888) differ by half a std. The EN story
  for the paper should be "≈0.79 regardless of key augmentation", not a ranking claim.
- Floor val-acc variance (±0.0221 acc, ±0.0448 F1) is again dominated by selection noise:
  its final-epoch variance is smaller and its mean higher — the same 1-2 point
  val-selection tax as ZH.

### Paper appendix paragraph (selection robustness)

> **Selection robustness.** Because our dev split is small (78 videos for MHClip-ZH), we
> audited the sensitivity of all conclusions to the epoch-selection rule. We re-scored
> every run under five rules: the pre-registered val-accuracy selection, val-AUROC
> selection, the mean test performance of the three best epochs by val-accuracy, the
> unselected mean over the last five epochs, and the final epoch. The paired
> archive-vs-baseline difference on MHClip-ZH ranges from −1.3 to +0.8 accuracy points
> across rules — i.e., the selection procedure shifts the estimate by more than the
> candidate effect. Under the two selection-free protocols the difference is −0.2 ± 0.4
> points (last-5 average) and exactly zero (final epoch: because the kNN-key augmentation
> leaves training untouched, same-seed checkpoints are byte-identical, and at the final
> epoch the augmented keys flip no test decisions). We therefore do not claim an accuracy
> improvement from the archive-kNN channel on MHClip-ZH, and we note that val-based epoch
> selection itself costs roughly two accuracy points relative to selection-free protocols
> on dev sets of this size. All headline tables use the pre-registered rule; no post-hoc
> rule was adopted.

## Implications

- Any paper table must report mean±std over these seeds, not seed-0 numbers.
- The consolidated-delta story ("archive-kNN beats LoRA-only floor") cannot cite
  0.8523-vs-0.8322: same-seed pairing shows the gain is within noise. If archive-kNN stays
  in the delta, it needs either (a) a variance-reduction fix (e.g., more stable val
  selection — seed 4 selected epoch 12; selected-epoch spread 12-23) or (b) a metric where
  the effect is consistent (roc) with an honest framing.
- LoRA-only ZH is itself remarkably stable (std 0.014) — the instability is introduced by
  the archive-kNN channel, not the encoder.
