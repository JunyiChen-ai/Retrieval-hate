# Experiments

*Draft chapter — protocol presentation per DECISION_MEMO D2 recommendation (dual-protocol
side-by-side); revisable upon user ruling. All quantities are transcribed from committed project
documents; where a number could differ between documents, `research-wiki/PAPER_MASTER_TABLES.md`
(commits `d9731e8`, `24343e6`) is authoritative and any residual tension is footnoted. Internal
provenance is given as [DOC:file] and external methods / datasets as `\cite{}` placeholders. No
number in this chapter is new — each is traceable to a committed result. Terminology follows the
method and analysis chapters: the retrieval objective is the **triplet-margin** contrastive
instance, the MLLM is framed by its **three earned roles and one explicit non-role**, and the
localization contrast is written against the **retrieval memory**, not a MIL head.*

---

## 1. Experimental setup

### 1.1 Datasets and protocol alignment

We evaluate on four public hateful-video benchmarks, all reduced to a single **binary
harmful-vs-normal** decision so that a common protocol, memory bank, and metric apply across
languages and label schemes. HateMM \cite{das2023hatemm} (1,083 EN videos, ~43h, binary
hate/non-hate with frame-span rationales) and ImpliHateVid \cite{rehman2025implihatevid} (2,009 EN
videos: 509 implicit / 500 explicit / 1,000 non) are English; MultiHateClip \cite{multihateclip}
(MHC) supplies a matched EN/ZH pair (~1,000 videos each) whose native three-way
hateful/offensive/normal label we binarise to hateful∪offensive vs normal, following the field
convention we also apply to the MoRE re-run (§3). HateClipSeg \cite{hateclipseg} is used only for
localization (§4), where its segment-level multi-hot gold spans are held out as evaluation labels
and never enter any scoring path.

**Binary declaration.** Every classification number in this chapter is the binary
harmful-vs-normal task. For MHC we map `hateful,offensive→1`, `normal→0` (the same mapping the MoRE
released code uses), so the two-way head is comparable across EN, ZH, and HateMM; ImpliHateVid's
implicit/explicit split is collapsed into the positive class.

**Leakage discipline and our split declaration.** Cross-paper baseline numbers were untrusted, so
all "floors" in this chapter are re-established on **our own clean splits** rather than lifted from
prior papers [DOC:experiments/exp-baseline-reproduction.md]. Two leakage controls travel with every result.
First, on the classification benchmarks the memory bank is built from the **training** split only;
the test split never enters the bank, and the kNN index is rebuilt per epoch over training
embeddings alone. Second, for localization (§4) HateClipSeg has no official split; we declare our
split to be **all 395 alive-subset videos scored zero-training**, so no HateClipSeg label or
statistic touches the scorer — the gold spans are validation-only [DOC:EVAL_localization_hateclipseg.md].
For the temporal protocol (§5) the temporal-val period lies strictly between the train and test
periods, and the temporal-test split never enters the memory [DOC:EVAL_temporal_memory_W4.md].

**Clean-test coverage.** Because a fraction of MHClip videos have missing labels or dead download
links, our strict same-arena test subsets are **HateMM n=215, MHClip-EN n=161, MHClip-ZH n=149**;
the corresponding clean training banks hold 743 (297 hateful) HateMM videos, 549 (168 hateful) EN,
and 579 ZH videos [DOC:BASELINE_MoRE_rerun.md §1.4, DOC:experiments/exp-cross-dataset-transfer.md]. These
counts are fixed across all our configurations and across the MoRE re-run, so every reported delta
is on identical test videos.

### 1.2 Metrics

Classification is reported as **accuracy, macro-F1, and macro precision / recall**; macro-F1 is the
headline because the positive rate is minority (≈24–34% depending on split) and accuracy alone is
inflated by the majority class. Localization is reported as **within-video mean AUC (wv-AUC)** — the
per-video AUC of window hate-scores against the frame-level gold, averaged over the videos that
contain both classes — as the primary, threshold-free metric, with 10k-bootstrap 95% CIs and a
one-sided sign-test versus 0.5; pooled frame/segment AP and AUC are reported as secondary evidence
because, as we show in §4, pooled AP is dominated by video-level toxicity *density* rather than
within-video localization [DOC:EXP_p6_mllm_localization.md].

### 1.3 Two calibrations, side by side

Every classification number is reported under **two selection protocols in parallel** — this is the
D2 dual-protocol presentation, and it is a substantive methodological choice, not a formatting
preference. The first is our **pre-registered protocol**: warmup-floored best epoch (epoch ≥ 5) by
`Val_Retrieval` accuracy, ROC tie-break. The second is a **selection-free final-epoch** protocol
(epoch 29). The reason to show both is that our dev sets are tiny (78 videos for MHClip-ZH), and a
selection-rule robustness audit shows that **validation selection on a 78-sample dev itself costs
≈ 2 accuracy points** relative to selection-free late-epoch averaging, while shifting the estimate
of a candidate treatment effect by *more* than the treatment itself [DOC:experiments/exp-archive-knn-seeds.md
Addendum 1–2]. Reporting a single protocol would therefore let the calibration choice, rather than
the model, decide the headline. We keep the pre-registered protocol as the primary headline and
place the final-epoch view immediately beside it; the ZH ≥ 0.85 crossing exists only under the
final-epoch protocol, and adopting it *because* it crosses would be post-hoc rule-shopping, so that
decision is explicitly left open (D2).

### 1.4 Seed protocol and noise floor

There is **no cross-seed ensembling anywhere** in this chapter (a user rule); every gain claim
first passes a sha1 / bit-for-bit weight-identity audit and only then a statistical test. On our
~150-video test sets the noise floor is explicit: **1 accuracy point ≈ 1.6 videos**, so any
sub-point, single-protocol "gain" is recorded as *within-noise, no claim*, and the reported signal
is the paired-delta sign pattern across seeds rather than a p-value mined from one run. At n=5 seeds
on a 149-video test the minimum detectable paired effect is ≈ 0.04–0.05 F1, so a true +0.01–0.02
effect is undetectable by design — absence of significance is reported as absence of a paper-grade
claim, not as evidence of no effect [DOC:experiments/exp-consensus-zh-seeds.md].

---

## 2. Main classification results (T1)

Table 1 transcribes the main table: four datasets × key configurations, both calibrations side by
side, with mean ± std and seed count. The headline reading is that **HateMM and ImpliHateVid already
clear the field's acc ≥ 0.85 bar, while MHClip-EN and MHClip-ZH sit near a documented ceiling** whose
attribution is developed in the analysis chapter.

**Table 1. Main classification table (clean test subset; val-selected / final-epoch).**

| Dataset (n) | Config | Encoder | val-sel acc | val-sel F1 | final-ep acc | final-ep F1 | seeds |
|---|---|---|---|---|---|---|---|
| HateMM (215) | frozen-CLIP RGCL floor | CLIP ViT-L/14-336 | 0.8279 | 0.8172 | — | — | 1 |
| **HateMM (215)** | **frozen-Qwen RGCL (best)** | Qwen2.5-VL-7B (frozen) | **0.870** | **0.861** | — | — | 1 |
| HateMM (P9-matched) | trained-RGCL floor | frozen-Qwen | 0.870 | — | 0.8605 | — | 3 |
| HateMM (P9-matched) | raw-kNN floor | frozen-Qwen | — | — | 0.786 | — | 3 |
| ImpliHateVid | frozen-CLIP floor | CLIP | 0.910 | — | — | — | 1 |
| ImpliHateVid | frozen-Qwen floor | frozen-Qwen | 0.900 (~0.91) | — | — | — | 1 |
| MHC-EN (161) | frozen-Qwen floor (no keys) | frozen-Qwen | 0.7702 ± 0.0221 | 0.7010 ± 0.0448 | 0.7888 ± 0.0152 | 0.7488 ± 0.0208 | 4 |
| MHC-EN (161) | + archive-kNN α0.25 (best) | frozen-Qwen | 0.7935 ± 0.0205 | 0.7497 ± 0.0250 | 0.7826 ± 0.0134 | 0.7430 ± 0.0196 | 4 |
| MHC-ZH (149) | LoRA-only floor (no keys) | Qwen2.5-VL-7B-LoRA | 0.8282 ± 0.0139 | 0.7962 ± 0.0167 | **0.8537 ± 0.0120** | 0.8259 ± 0.0124 | 5 |
| MHC-ZH (149) | + archive-kNN α0.25 (best) | Qwen2.5-VL-7B-LoRA | 0.8268 ± 0.0266 | 0.7915 ± 0.0397 | **0.8537 ± 0.0120** | 0.8259 ± 0.0124 | 5 |
| MHC-ZH (149) | + consensus denoising † | frozen-CLIP | 0.8107 ± 0.0347 | 0.7764 ± 0.0406 | 0.8175 ± 0.0129 | 0.7841 ± 0.0204 | 5 |
| MHC-ZH (149) | consensus λ=0 floor † | frozen-CLIP | 0.8027 ± 0.0139 | 0.7649 ± 0.0151 | 0.8027 ± 0.0215 | 0.7594 ± 0.0240 | 5 |

Source: T1.1 [DOC:PAPER_MASTER_TABLES.md], from exp-archive-knn-seeds, exp-consensus-zh-seeds,
exp-baseline-reproduction, EXP_p9.

**HateMM and ImpliHateVid are solved on our split.** The frozen-Qwen stack reaches HateMM
**0.870 / 0.861**, crossing the field's acc ≥ 0.85 bar; ImpliHateVid floors sit at ≈ 0.90–0.91.
These two datasets were established once and not iterated further, so they carry single-seed
reference numbers and no MoRE same-arena track (the MoRE re-run does not include ImpliHateVid).

**The MHClip near-ceiling, read honestly.** MHClip-EN sits at ≈ 0.78–0.80 across every
configuration and calibration, and MHClip-ZH at ≈ 0.827 (val-selected) / 0.854 (final-epoch). Three
load-bearing facts control how these rows may be cited:

1. **The ZH final-epoch protocol is the only calibration that crosses 0.85** (0.8537 ± 0.0120;
   seeds 3/4 individually reach 0.8658). Both arms are ≈ 0.827 val-selected and do not cross.
2. **The archive-kNN key contributes exactly zero accuracy on ZH at final-epoch.** `archive_mode=knn`
   never touches training, so same-seed checkpoints are byte-identical (verified by sha1 audit); at
   epoch 29 the α = 0.25 key (≈ 6% similarity weight) flips **zero** test votes on all 5 ZH seeds, so
   the archive and floor arms are bit-for-bit identical. The multi-seed paired accuracy delta is
   −0.0014 ± 0.0313 — the earlier "archive-kNN gives accuracy" claim is **withdrawn**
   [DOC:experiments/exp-archive-knn-seeds.md].
3. **EN mirrors this lesson.** The val-selected archive arm looks +2.3 acc points over the floor
   (0.7935 vs 0.7702), but a large share of that gap is one floor seed's pathological selection
   (s3 selected epoch 6 → 0.7391); under the selection-free final-epoch protocol the delta is
   −0.0062 ± 0.0051 (positive on 0/4 seeds). The EN story is therefore **"≈ 0.78–0.80, no key
   augmentation separates,"** not a ranking claim.

The consensus rows (†) are a **CLIP-base mechanism / robustness sub-experiment** (a different
encoder from the LoRA-Qwen main ZH stack) and are not directly commensurable with the archive-kNN
rows; they are listed for completeness and analysed as a de-poisoning mechanism in §5, not as a
headline accuracy row.

**Encoder-role ablation (CLIP vs frozen-Qwen vs LoRA-Qwen).** The encoder is the one component
whose swap moves the main table, and it moves it as a *frozen-encoder identity*, not as a new method
role. On HateMM, frozen Qwen2.5-VL features beat frozen CLIP by ≈ +4.2 macro-F1-equivalent and are
what carries the 0.85 crossing (0.870 / 0.861 vs the 0.8279-accuracy / 0.8172-macro-F1 CLIP floor). On ZH, LoRA
adaptation of Qwen2.5-VL is the best available front-end and is used for the main ZH stack; on EN,
frozen Qwen is best and LoRA regresses. The three front-ends are read by the same head with no
head-code change (feature dims are read from cache), which is exactly what makes the encoder a
swappable, run-once front-end rather than part of the trainable recipe. This is the MLLM's
**encoder role** — earned and removable (reverting to CLIP loses the HateMM crossing) — and we label
it as the encoder identity rather than the accuracy-bearing method role the campaign's mandate
sought (§4, and the analysis chapter's non-role result) [DOC:PAPER_MASTER_TABLES.md T1.1]. Why the
swap carries HateMM but not MHClip is now **mechanistically characterised rather than left open**: the
Qwen representation upgrade is uniform across all three datasets (top-20 neighbourhood purity +0.021–
0.023, text-stream AUC +0.041–0.054), but it converts to accuracy only where hate is visually grounded
and the residual errors are representation-limited. On MHC-EN the Qwen image stream collapses to
near-chance (train-LOO AUC 0.734 → 0.599) and the deployed align head's element-wise (Hadamard) fusion
(`fusion_mode='align'`, `src/model/classifier.py:110–122`) corrupts the fused key multiplicatively,
cancelling the text gain (net dev −0.012); the round-4 FA gate measures this cell and confirms no
modality reweight converts (analysis §3.6 erratum) — which is also why 32B scale regresses rather than
rescues (image AUC still 0.608) [DOC:ENCODER_SWAP_DIAGNOSIS.md, commit `8a48938`;
DOC:FA_GATE_RECORD.md, commit `e0877c9`; analysis chapter §3.6].

---

## 3. Same-arena baseline comparison (T1.2)

MoRE (WWW 2025) \cite{more} is the only published retrieval-augmented hateful-video method, so it is
the correct same-arena baseline. We re-ran its **official released code** on the **identical split**
(line-for-line diff verified), the identical clean test subset, in both an as-released and a
bug-fixed variant, plus a 5-seed sensitivity sweep. Table 2 places our best configuration beside
both MoRE variants, the MoRE 5-seed mean, and MoRE's own reported (full-data) numbers.

**Table 2. Same-arena MoRE comparison (clean test subset).**

| Dataset (clean n) | Ours (best, val-sel) ‡ | MoRE as-released | MoRE bugfix | MoRE 5-seed (as-rel) | MoRE reported (full) | **Δ (ours − better MoRE)** |
|---|---|---|---|---|---|---|
| **HateMM (215)** | frozen-Qwen **0.870 / 0.861** | 0.8140 / 0.7988 | 0.8047 / 0.7899 | 0.792±0.035 / 0.781±0.038 | 0.8341 / 0.8235 | **+5.6 acc / +6.2 F1** |
| **MHC-EN (161)** | frozen-Qwen **0.7888 / 0.7378** | 0.6894 / 0.4438 | 0.7019 / 0.5084 | 0.722±0.031 / 0.530±0.111 | 0.7750 / 0.7519 | **+8.7 acc / +22.9 F1** |
| **MHC-ZH (149)** | LoRA-SFT **0.8322 / 0.8023** | 0.7651 / 0.6882 | 0.7584 / 0.7058 | 0.717±0.035 / 0.661±0.023 | — / 0.7475 | **+6.7 acc / +9.7 F1** |

Source: T1.2 [DOC:PAPER_MASTER_TABLES.md], from BASELINE_MoRE_rerun §3.1–3.2.

**We win on all three shared benchmarks** — +5.6 / +8.7 / +6.7 accuracy and +6.2 / +22.9 / +9.7
macro-F1 over the stronger MoRE variant — and taking MoRE's 5-seed mean as an upper bound, or its
better variant, does not reverse any of the three. Nominal training-label count still slightly
favours MoRE (EN 618 / ZH 633 labelled videos vs our clean 550 / 579), so the win is not bought with
more supervision.

**Honest re-run declaration.** Reproduction integrity is the point of this section, so the caveats
are stated plainly rather than buried:

- **Sanity.** On HateMM — the only dataset whose data we hold complete — the re-run lands 2–3 points
  below the published value (as-released 0.8111 / 0.7954 vs reported 0.8341 / 0.8235), inside
  single-seed variance (seed 2023 dips to 0.713 F1). This is a **successful reproduction**. The
  MHClip re-runs fall further below the published values, driven mainly by **local data loss**: we
  hold EN labels for 890/1000 and video for 792, ZH 897/814, against the paper's full 1,000, and the
  degradation is monotone in coverage (HateMM 100% → −2.8 F1; ZH 86% → −6.2; EN 79% → −23). The EN
  F1 collapse is a validation-early-stopping artefact — a 91-sample val set freezes the checkpoint at
  epoch 5 while F1 is still climbing. MoRE's published values are therefore shown as
  "reported (full data)" and are **not** compared directly against the clean subset.
- **Locally reconstructed inputs.** The `caption.jsonl` generation is undocumented upstream, so we
  reconstructed captions with Qwen2.5-VL-7B; ZH OCR was produced with easyocr after PaddleOCR failed
  on this cluster (GPU cudnn unavailable, CPU SIGILL) — a task-authorised documented substitution.
  Both feed only the retrieval memory bank, and HateMM's clean reproduction bounds their impact.
- **Five released-code defects, all documented and handled.** (1) a missing `einops` dependency;
  (2) an audio-branch loop bug in `merge_feature.py` (the "audio memory" is silently two copies of
  the caption embedding) — kept in the as-released variant, corrected in the bugfix variant, whose
  ±1–2pt effect is smaller than seed variance and direction-inconsistent, so it is **not** the
  reproduction gap driver; (3) an O(n²) in-loop `torch.save` (performance-only, corrected off-loop
  with identical file contents); (4) a preprocessing↔model shape mismatch that makes the released
  preprocessing script fail against the released training code (handled by an idempotent unsqueeze);
  (5) a `task: binary` config key placed where `main.py` cannot read it (handled by a zero-file-edit
  CLI override). Every handling left an audit trail [DOC:BASELINE_MoRE_rerun.md §1.5, §3.0].

‡ **Tension note (per T1.2 footnote and tension #1–2).** The "ours" point estimates in Table 2 are
the ITERATION_LOG warmup-consistent **single-config val-selected** cells (EN frozen-Qwen; ZH
LoRA-SFT best). These are not the same source as the multi-seed audit in Table 1: the ZH 0.8322 is a
single-seed value (killed as such and superseded by val-sel 0.8268 ± 0.0266 / final 0.8537 ± 0.0120),
and EN 0.7888 / 0.7378 is likewise a single-config val-selected cell. **The paper's headline is
Table 1's multi-seed audit; the MoRE Δ is the single-config point estimate, footnoted, to pre-empt a
"seed cherry-pick" objection.** The MoRE deltas remain qualitatively unchanged if the multi-seed
means are substituted (ZH multi-seed final 0.854 vs MoRE 0.765 is still +8.9 acc).

---

## 4. Localization (T2)

Localization is evaluated as a **zero-shot chain of increasing scorer strength** on HateClipSeg,
all measured by within-video mean AUC on the same 329 both-class videos with the same estimator.
The chain isolates exactly how much of the ranking comes from the scorer versus the harness.

**Table 3. HateClipSeg localization chain (within-video mean AUC).**

| Config | wv-AUC | 95% CI | significance | paired vs memory | paired vs P6-7B |
|---|---|---|---|---|---|
| random (seed 0) | 0.5088 | — | — | — | — |
| memory `knn_hatemm_subclip` K=30 | 0.5140 | [0.4955, 0.5323] | sign-p 0.11 (n.s.) | — | — |
| *(memory strongest cell: K=4 subclip)* | *0.5259* | *[0.5048, 0.5468]* | *sign-p 0.0066* | — | — |
| **P6 — MLLM per-window scorer (7B)** | **0.5435** | [0.5330, 0.5544] | sign-p 5.4e-8 | Δ+0.0296, CI[+.0088,+.0504], p=0.0071 | — |
| **P10-b — 72B A-fuse (promoted)** | **0.5755** | [0.5581, 0.5933] | sign-p 1.4e-9 (n=329) | Δ+0.0615, CI[+.0359,+.0869], p=4.9e-5 | Δ+0.0319, CI[+.0170,+.0474], p=0.0024 |

Source: T2.1 [DOC:PAPER_MASTER_TABLES.md], from EVAL_localization_hateclipseg, EXP_p6, EXP_p10.

**The chain.** A random scorer sits at 0.5088. The zero-training cross-dataset memory scorer — a
consensus-kNN vote of a *different dataset's* video labels over CLIP-visual windows — reaches only
0.5140 at the density-matched K=30 granularity (its single significant cell is K=4 subclip memory at
0.5259, sign-p 0.0066). The **P6 MLLM per-window scorer** reads M = 120 frames binned into K = 30
windows (≤ 4 frames/window) plus that window's Whisper \cite{whisper} ASR and emits an integer 0–3
hate-evidence-density; it reaches **wv-AUC 0.5435**, CI excluding 0.5 (sign-p 5.4e-8, three orders
tighter than the memory cell) and paired over memory Δ+0.0296 (p = 0.007). The promoted **72B
A-fuse** amplifier — a coarse×fine rank aggregation `0.5·(K=30 fine) + 0.5·(K=4 coarse)` computed
from the *same* scorer's outputs — reaches **wv-AUC 0.5755** on the single permitted HateClipSeg test
touch, paired-significant over both memory (Δ+0.0615) and P6-7B (Δ+0.0319). The mechanism matches the
diagnosis: the memory scorer's CLIP-visual keys are blind to speech-borne hate, and the MLLM's window
ASR is what closes part of that gap.

**Verdict: modest, not substantial.** Against a pre-registered three-tier bar (≥ 0.60 substantial /
0.56–0.60 with CI excluding 0.5435 = modest / < 0.56 = P6 stands), 0.5755 ∈ [0.56, 0.60) with CI
lower bound 0.5581 > 0.5435 → **modest amplification**. The localization role is upgraded from modest
(7B) to modest-plus (72B A-fuse); its earned-and-removable character is unchanged, only its
magnitude grows. The honest caveat travels with it: the MLLM's *dominant* competence here is
video-level toxicity density (broadcast AP 0.62), and the within-window increment is the smaller —
though statistically stable and monotone in scorer scale — signal.

**Calibration leaderboard and the three walls (summary; full 14-comparison table in the appendix).**
On a HateMM calibration leaderboard versus a frozen-7B anchor (wv-AUC 0.5387), raw scorer scale alone
does not clear the bar (anchor-agg 0.5387 → 0.5512 (32B) → 0.5593 (72B)), whereas the A-fuse gain
grows monotonically with scorer size (+0.0305 7B → +0.0437 32B → +0.0526 72B) and its significance
replicates across five scorers — the one lane where scale converts. **Three walls close the
open-source ceiling in one sentence:** re-aggregation tops out at 0.5932, the 72B scale champion at
0.5913, and the next-generation Qwen3-VL-32B A-fuse at 0.5866, all below the 0.616 calibration line
that would extrapolate to a substantial (≥ 0.60) test result — so substantial localization is
unreachable in the open-source domain on this cluster, and 0.5755 is the final localization number
[DOC:PAPER_MASTER_TABLES.md T2.2].

**Two narrative boundaries.** First, we claim only **span-free** localization — never *first*,
*annotation-free*, or *dense-supervision-free*. Second, the paper's contrast for the localization
role is the **retrieval memory, not a MIL head**: A-fuse − memory = +0.0996 (significant), whereas
A-fuse minus a same-operator video-label MIL proxy is *not* significant (P11: +0.0359, CI
[−0.0009, +0.0730], sign-p 0.13), because a 5-fold linear MIL head already reaches ≈ 0.55 wv-AUC —
video labels alone contain most of what the 72B weak label would teach. We contrast against memory,
not MIL, because a MIL head needs target-domain video labels that the **zero-shot memory swap does
not**; the two are not the same capability, and reporting the memory contrast keeps the swap
capability (§5) intact [DOC:EXP_p11_weaksup_localization.md]. Published localization baselines
(MultiHateLoc / LELA / TANDEM) are codeless with under-specified protocols and cannot be brought into
a same-arena table; they are discussed in related work only [DOC:EVAL_localization_hatemm.md §4].

---

## 5. Capability experiments (T3)

Beyond accuracy, the retrieval-memory read-out supports operations that a trained MoE / classifier
head is **structurally incapable** of. Each result below is a capability demonstration reported at
its true strength, and each is paired with the operation the baseline head cannot perform.

**Cross-dataset memory swap.** A head trained on source A classifies target T by **swapping in T's
own labelled memory with no gradient step**. The learned space transfers: it beats the target
majority baseline on **5 of 6** informative cross cells, lagging in-domain by only ≈ 0.04–0.09
macro-F1 on working cells [DOC:experiments/exp-cross-dataset-transfer.md]. A trained MoE router bakes its
decision into weights and **cannot be re-pointed at a new support set** — this is the headline
capability delta versus MoRE, reported as a capability (cross never beats in-domain), not an accuracy
win.

**Temporal recalibration.** On an MHClip-EN temporal split, macro-F1 drops −0.084 (0.7113 → 0.6273).
We attribute the drop to **calibration drift, not lost separability**: the temporal-split ROC
(0.8484) *exceeds* the random-split reference (0.7175), so the ranking survives the shift and only
the operating point moved (8.7% of test scores clear the 0.5 threshold against a 24.2% true positive
rate). The correct lightweight fix is therefore **threshold recalibration, not memory growth**:
labelling k = 20 new-period examples and re-calibrating the vote threshold recovers the drift in full
(**0.6273 → 0.7336** ≥ the random-split floor 0.7113, zero retraining), against an oracle ceiling of
0.7646 (acc 0.8199). Two honesty controls travel with the claim: the naive "add the k = 20 samples to
the memory" mechanism is flat-to-negative (0.6180), so the win is recalibration *specifically*; and
MHClip-ZH shows **no drift** (a negative control, +0.014), where small-k recalibration is pure noise
and should be gated behind a drift monitor. The retrieval architecture exposes the operating point as
a **first-class, O(1), reversible knob**; a trained MoE head hides it inside the weights and requires
fine-tuning to move it [DOC:EVAL_temporal_memory_W4.md].

**Human-in-the-loop memory edit — a capability demonstration, single-seed, not an accuracy claim.**
Semantic addressing plus surgical deletion is pure-CPU and seconds-fast. Deleting **two** human-flagged
noisy memory entries lifts MHClip-EN test accuracy **0.8075 → 0.8199** (macro-F1 0.7626 → 0.7748) **at
seed 0** with **zero retraining**, exceeding all five random-seed floors and all five same-size
random-deletion controls (max 0.8137) [DOC:DEMO_memory_editing.md]. **Round-8 multi-seed correction (F88),
which binds every citation of this result:** an exact replay of the same deletion on the banked top-60
neighbour lists of all four deployed seeds — the un-edited replay reproducing each seed's floor to
< 1e-12 — gives **+0.0124 on seed 0 and exactly zero vote flips on seeds 1, 2 and 3**, a four-seed mean of
**+0.0031**, and the two items seed 0 flips (`cYQyH7hbNnw`, `xqilG4oMvvI`) are both low-margin false
positives from the seed-flip noise band rather than hard errors. The 14-id rule-hit list is strictly
stronger — **+0.0093 acc / +0.0089 mF1, 3 of 4 seeds positive, 6 items fixed and 0 broken on any seed** —
but remains 3× under the +0.030 bar, inside the ±0.014 seed band, and is now test-consumed; a legitimate
dev-side pregate is arithmetically impossible, since at dev n = 80 one item is 0.0125 and cannot resolve a
+0.009 effect [DOC:ERRPAT_MHC-EN_2026-07-26.md §6.5, commit `ad56a62`]. The phrase "the project's best
single EN point" is therefore **withdrawn**; the claim this cell supports is that the memory admits
targeted, retraining-free, semantically addressed surgery at all — the same operation on a trained head
does not exist. A second F88 correction narrows F78 in our favour: for EN *deletion-only* edits the "$0 on
banked keys" premise **does hold** (the four seeds' top-60 lists are banked in the deployed key space), so
EN curation, if ever revisited, is a CPU job rather than a GPU re-mint; F78's limitation applies to bank
*additions*, key-space changes and retraining.

**Guard-rail / semantic veto.** A two-vote AND rule for *automatic* repair does **not** reproduce the
human 2-entry gain (C − A = +0.0000, 0/4 EN seeds; the target it fails to reproduce is itself the
single-seed effect corrected above): it structurally cannot reach memories that are
semantically contradictory yet not embedding outliers. Its surviving value is a **semantic veto** — it
blocks an embedding-only (Cleanlab-style) rule from over-deleting genuinely-hateful-but-embedding-hard
entries (abuse testimony, assault reporting, slur-bearing text), worth C − D = **+0.47pt EN / +0.40pt
ZH** [DOC:EXP_auto_memory_repair.md]. As with the edit, the label-blind archive audit independently
re-discovers the human-flagged noisy ids with correct reasons. The archive's payoff is **integrity and
controllability, not raw accuracy** — a framing defensible precisely because it is not dressed as an
accuracy claim; a weight-baked head offers no auditable, editable surface to veto over at all.

---

## 6. Ablations and robustness

**Dual-calibration robustness.** Table 1 is itself the primary robustness artefact: every
classification cell is shown under both the pre-registered val-selected and the selection-free
final-epoch protocol, so no conclusion in §2 rests on a single calibration. The two protocols agree
on the qualitative reading (EN ≈ 0.78–0.80 with no key separation; ZH ≈ 0.827/0.854 with the crossing
only at final-epoch) and disagree only on the ZH ≥ 0.85 headline, which is exactly the open D2
decision.

**λ = 0 bit-for-bit checks.** Two claims are anchored on weight-identity audits rather than
statistics. (i) The archive-kNN channel leaves training untouched (`archive_mode=knn`), so same-seed
checkpoints are byte-identical (sha1 match against the disk-guard-recorded hash), and at final-epoch
the α = 0.25 keys flip zero ZH votes — the archive and floor arms are literally the same numbers, so
no accuracy is claimed for the channel [DOC:experiments/exp-archive-knn-seeds.md Addendum 2]. (ii) The
consensus floor arm is the λ_seg = 0 whole-video baseline, verified bit-for-bit through the segment
code path, so the consensus delta is measured against a control that is provably the same recipe with
the segment term switched off [DOC:experiments/exp-consensus-zh-seeds.md]. The consensus claim that
survives is the **repair** claim: inherited-label sub-clip supervision poisons ZH by −0.066 macro-F1,
and consensus relabeling removes that hole and lands at-or-weakly-above the floor across 5 seeds
(val-selected +0.0115, p ≈ 0.57; final-epoch +0.0247, p ≈ 0.11) — a de-poisoning mechanism, *not* an
accuracy win.

**Selection-rule sensitivity (appendix pointer).** Because the ZH dev set is 78 videos, all
conclusions were re-scored under five epoch-selection rules — pre-registered val-accuracy, val-AUROC,
top-3-mean, last-5-mean, and final-epoch. The paired archive-vs-baseline difference on MHClip-ZH
ranges from **−1.3 to +0.8 accuracy points across rules** (i.e. the selection procedure shifts the
estimate by more than the candidate effect), collapsing to −0.2 ± 0.4 points under last-5-mean and
exactly 0 under final-epoch. Both ZH arms sit at ≈ 0.846–0.848 under last-5-mean versus ≈ 0.827–0.828
under val-accuracy, quantifying the ≈ 2-point **val-selection tax**. All headline tables use the
pre-registered rule; no post-hoc rule was adopted. The full five-rule table and the written
selection-robustness paragraph are deferred to the appendix, together with the T2.2 calibration
leaderboard (14 comparisons) and the exploratory re-aggregation ceiling
[DOC:experiments/exp-archive-knn-seeds.md Addendum 1–3].

---

## 7. Rounds 2–3: the novelty-first extension (constraint-space closure)

The thirteen-route campaign of §4 and T4 answered the original mandate; a user ruling then **tightened
the goal to require a *novel* MLLM mechanism** (D7: an encoder-class lever, however well it performs,
does not by itself satisfy novelty) and the search was re-run under that stricter bar across three
further sprints. These sprints add **pre-registered negatives** to the ledger (and one round-4 performance
positive, LoRA-HateMM), plus a round-4 **closing** pair — a memory→adaptation-coupling curriculum probe
(cand-2, which ties generic LoRA on the primary ZH leg and adds over generic on HateMM val-selected only)
and the premise-(d) EN-composition gate (which closes MHC-EN at the last untested composition level);
they do **not** revise the campaign's 13-route accounting (T4), and are reported here as a
clearly-labelled extension. The four structural laws that crystallised from them are analysed in the
analysis chapter §3.6–3.9; this section is the results ledger.

**Count discipline.** T4's thirteen routes are unchanged: at route-family granularity, ten main-table
accuracy rows plus three localization rows (P6 / P10 / P11); the analysis chapter's finer count instead
splits P9/P9b to report eleven main-table routes — the same thirteen results under two granularities,
as the master-table tension list #7 documents. **This 13-route campaign count is the load-bearing
accounting and is untouched by every sprint below.** On the separate novelty-first negative ledger:
round 2 adds seven sprint negatives (#15–21) plus the B4 EN-LoRA-encoder cell (#22) — closed pre-GPU
then, now formally measured as a FAIL under round 4's line-A run (F53) — and one *marginal positive*
held pending a user novelty ruling (B3); round 3 adds six directions, every one closed at a binding
verdict or a calibrated-zero conditional-info gate; round 4 adds two further pre-registered
negatives — the per-item cross-channel router (F47) and the fusion/composition FA gate (F50) — plus a
pre-GPU arithmetic kill (MJ, F49), a wave-5 adaptation-family structural closure (F51), and the line-A
LoRA-HateMM measurement (F53): an encoder-level LoRA **HateMM PASS under both protocols** (a *second*
performance positive alongside B3, held pending the same user D7 novelty ruling) that also formally
measures the bundled B4 EN cell (#22) as a FAIL; and a round-4 **closing** pair — the cand-2 curriculum
coupling probe (F56, a tie on ZH / one-cell add on HateMM, no kill fired, held pending the same D7
sub-ruling, opens no new dataset) and the premise-(d) EN-composition gate (F55, a $0 CPU KILL that is the
sixth "better-signal / no-conversion" datum). None of these five round-4 items is a main-table-accuracy
route, and none revises the 13-route campaign accounting (T4); the two closing items add one extension
negative (premise-(d)) and one pending-ruling coupling probe (cand-2). With round 4
no surviving candidate remains in the frozen constraint box [DOC:TERMINUS_round2_mllm_plus3.md,
DOC:TERMINUS_round3_mllm_plus3.md, DOC:ROUTER_GATE_RECORD.md, DOC:FA_GATE_RECORD.md,
DOC:CAND2_VERDICT_REVIEW.md, DOC:PREMISE_D_GATE_RECORD.md]. (A ledger-ordinal
note: the round-4 records label F47 / F50 the "22nd / 23rd pre-registered negative," an ordinal
continued from the round-2 *terminus* count that does not line up with this section's sprint numbering,
where #22 already denotes B4; the paper uses the round-by-round framing above and the master-table
tension list #9 records the discrepancy rather than minting a contested grand total.) No number below is
new: each is transcribed from its committed verdict/record with the commit cited inline.

**Table 4. Round-2 negatives (novelty-first sprint; #15–22).**

| # | Direction | Epitaph (one line) | Verdict · record |
|---|---|---|---|
| 15 | A-line `lb_scgp_global` (label-blind certificates → global Gram) | killed pre-GPU by the G0-cond probe: cache 91–93% one literal constant, oracle@coverage an order of magnitude under the +0.040 bar, v3 rejected (parsed certs are noise-quality); 264 GPU-h saved | A_LINE_PAUSE_DECISION.md |
| 16 | C1 RA-HMD two-stage sequential QLoRA | anchor-paper ablation prices the untested cell at only +0.7; measured DEV kNN ≈ −0.02 vs the frozen floor (job 13039) | C1_KILL_REVIEW.md |
| 17 | C3-target (real Qwen-7B target predictor as conditional channel) | oracle ceiling +0.0487 marginal, real predictor ≈ 0 (best +0.0094 < +0.040), MHC anti-informative; calibrated machinery | C3_REAL_PREDICTOR_PROBE.md |
| 18 | C2-SAV sparse attention-head mining (784 image-stream heads, frozen 7B) | F-G1 KILL confirmed under corrected machinery; the MHC cell was a crushed-baseline artefact, the HateMM harm real; **the dilution hypothesis is falsified** (MHC-EN is data/label-limited) | SAV_F1_VERDICT_REVIEW.md |
| 19 | C3-nontarget dense reasoning-text channel (late fusion on best config) | DEAD_AT_FUSION: all three pre-declared fusion rules fail on a calibrated + permutation-null instrument; the CLIP-only gain is encoder redundancy (info already banked in the Qwen pathway) | C3_FUSION_PROBE_RECORD.md |
| 20 | B1 frozen-Qwen encoder on MHC-ZH (3-seed paired) | FAIL both protocols (final-epoch mean −0.0112 acc, 1/3 seeds same-sign; gates clean); the ZH 0.8537 is a LoRA lever, not a frozen-encoder one | B1_VERDICT_REVIEW.md |
| 21 | B2 Qwen2.5-VL-32B frozen encoder (scale axis) | goal FAIL: on HateMM 32B sits *between* CLIP and 7B (**scale regresses**), below CLIP on MHC-EN/ZH, 32B-vs-7B fails everywhere — scale is not the conversion lever | B2_VERDICT_REVIEW.md |
| 22 | B4 LoRA-Qwen encoder on MHC-EN (3-seed paired) | **now formally measured** (job 13235, bundled with the LoRA-HateMM run, F53): FAIL both protocols — val-sel mean Δacc −0.0021 (acc 2/3 seeds), final-ep +0.0000 (acc 1/3 seeds), each ≪ the +0.030 bar; the seed-0 anchor reproduces the pre-GPU forensic value exactly (val-sel −0.0310 acc vs CLIP, below both frozen floors); the EN LoRA-encoder cell is closed | B4_FORENSIC_RECON.md → LORA_HATEMM_VERDICT_REVIEW.md |

**B3 — the one marginal positive, held pending a novelty ruling.** LoRA-adapting Qwen on MHC-ZH is the
only rounds-2/3 result that clears the performance clause on any protocol: 3-seed paired vs frozen-CLIP,
**final-epoch +0.0313 acc / +0.0453 macro-F1, 3/3 same-sign → PASS (MARGINAL)**, while **val-selected
+0.0246 acc FAILS** the +0.030 AND-rule (binding language verbatim: `final-epoch: PASS (MARGINAL);
val-selected: FAIL`) [DOC:B3_VERDICT_REVIEW.md (job 13150); DOC:PAPER_MASTER_TABLES.md PUR-1]. Three
mandatory sensitivity facts travel with it: the +0.0313 mean clears the bar by only +0.0013 (≈ 4% of
the bar); seed-2 alone is +0.0201, below the per-seed bar; and that +0.0013 margin is ≈ 15× smaller
than the +0.0201 across-seed spread. A zero-GPU decomposition locates the entire +0.0313: it lives in
the **text stream** (train-LOO AUC 0.802 → 0.847 → 0.925 for CLIP → frozen-Qwen → LoRA, image stream
untouched), and on test it converts as a genuine **Pareto** minority-recall gain (hate-recall +0.1111
at −0.0032 non-hate — the HateMM encoder-swap signature) rather than the frozen-Qwen **rotation**
(+0.0741 hate bought with −0.0481 non-hate, net −0.0112); the val-selected FAIL is a 78-sample-dev
selection-noise artefact (LoRA's dev-acc plateaus by ~epoch 19 while test keeps climbing to 29), not
mechanism fragility [DOC:B3_ZH_LORA_DECOMPOSITION.md, commit `d76e407`]. This answers the *performance*
half of the novelty question yes-but-thin; the gain is still LoRA *adaptation*, not encoder identity
(frozen-Qwen is −0.0112 on ZH). B3 alone is single-dataset, but the round-4 LoRA-HateMM run (below)
closes the gap it left open: the *same* encoder-level LoRA lever also passes HateMM, so under the
final-epoch protocol one lever now clears the +0.03/+0.03 conjunct on two datasets — with the honest
caveat that the two passes convert via the **same decisive modality (text) but different levers** (ZH
text-borne and LoRA-specific; HateMM text-carried on a swap-neutral image base and inherited from the
frozen swap, analysis §3.9). Whether either LoRA pass counts
toward the goal's *novel* clause is an explicit user ruling; neither is folded into any main table
[DOC:PAPER_MASTER_TABLES.md PUR-banner].

**Table 5. Round-3 negatives (novelty-first; every axis closed at a binding verdict or a $0 gate).**

| Direction | Epitaph (one line) | Verdict · commit |
|---|---|---|
| **S2S** — Qwen frame-group set-matching (retrieval-object / don't-pool) | KILL both datasets: HateMM SET−POOLED +0.0035 acc / +0.0003 mF1 fails the +0.05 bar on all six sub-conditions, MHC-EN −0.0397; a gold oracle shows +0.0917 / +0.1399 headroom that MeanMaxSim cannot convert (§3.6); closes the retrieval-object family across both encoders | S2S_PROBE_VERDICT_REVIEW.md · `2c96ab6` |
| **CTF** — supervised temporal-pool / arc-increment of the causal-prefix frameset | $0 conditional-info gate, kill-side on all four cells with valid calibration: [g_1…g_T] adds +0.0000 (HateMM) / −0.0029 (MHC), arc −0.0049 / −0.0010 over the pooled key; the supervised leg of the cumulative-causal closure (§3.7) | CTF_GATE_RECORD.md · `0eb6d33` |
| **APX** — whole-video classical prosody (eGeMAPS 88-d) auxiliary channel | $0 gate, both conditions fire, calibration valid: best arm −0.0038, strictest raw-88-d arm +0.0005 = exactly zero conditional info over Z_best; the ASR transcript already banks the spoken-hate content, so classical prosody is conditionally redundant | APX_GATE_RECORD.md · `9c54faf` |
| **AVC** — prosody × visual-segment correspondence | never started: gated behind APX, dies with it; the audio axis is parked | (gated behind APX; APX_GATE_RECORD.md) |
| **W2-A** — transcript-first grounded vision key | DEAD both datasets at the binding conditional-info gate K9: Δacc −0.0000 (HateMM) / −0.0038 (MHC) over the 8960-d best rep; the advisory kNN grounded key is *worse* than concat (−0.0259 / −0.0509); "a clean CLIP-redundancy null" (§3.6), the third oracle-exists-but-unconvertible instance | W2A_PROBE_VERDICT_REVIEW.md · `7228373` |
| **GIR** — isolated grounded-incongruity residual (grd − ungrd) | $0 gate, kill-side on all five cells: r_cache +0.0012 (HateMM) / −0.0051 (MHC), r_field +0.0000 / −0.0064; the residual is an **exact linear subset** of the baseline (verified residual-norm 0), so the W2-A K9 null mathematically subsumes it — the last candidate in the pool | GIR_GATE_RECORD.md · `b64a85b` |

Round 3 also retired six recon-/triage-stage companions that fed the axis closures: **W2-B** (frozen-CLIP
subclip set-matching, cloud-triage verdict (d), commit `0f43bdd`) and **W2-E** (prototype memory, killed
pre-ceremony) closed the retrieval-object and memory-reorganisation families alongside S2S; **W2-C**
(temporal order-kernel) was extinguished when S2S died — its sole authorised vehicle — atop a
prior-lowering CLIP-K4 pre-check; **C5** (7B relational CRD) and **R3-C3geo** (frozen-Qwen geometry
hard-negative mining) were pre-ceremony no-gos as encoder-class / frozen-reorganisation levers under
D7; and **B5** (per-encoder threshold calibration) proved the frozen-Qwen ZH ranking edge unconvertible
at any operating point, including the label-oracle cut [DOC:B5_VERDICT_REVIEW.md, commit `50f01b9`] — a
performance/diagnosis line that answers the B1 mystery and underpins §3.6's rotation-not-Pareto reading.
With GIR the wave-3 pool is empty and every injection point in the frozen constraint box is closed by a
binding verdict or a calibrated-zero gate; the remaining moves are user rulings, not further search
[DOC:TERMINUS_round3_mllm_plus3.md §1, §4].

**Table 6. Round-4 negatives (novelty-first; wave-4 selection/fusion levers).**

| Direction | Epitaph (one line) | Verdict · commit |
|---|---|---|
| **Router** — per-item cross-channel routing (CLIP-arm vs Qwen-arm) over decision-level meta-features | $0 gate, KILL at the deployable read AND the realizable ceiling: oracle headroom is real (**+0.1083 MHC-EN / +0.0498 HateMM**) but the train→dev router yields **+0.0000** on every seed (the CLIP head memorises train — LOO 0.998 vs Qwen 0.800 — degenerating the routing target, "Qwen-correct" 0/109·0/102·0/92), and the dev-CV ceiling is **−0.0458** (CI [−0.0875, 0]) below the perm-null p95 +0.0042; per-item channel-selection now closed at all three supervision sources (analysis §3.8), machinery 12/12 bit-exact, oracle-calib accZA 1.000 | ROUTER_GATE_RECORD.md · `30d0ee1` |
| **MJ** — MLLM modality-reliability judgment as a *new* router input (the carve-out F47 left ajar) | NO-GO pre-GPU on arithmetic alone: clearing the +0.020 gate needs which-arm-wins accuracy **q ≥ 0.663**, but the modality-locus alignment ceiling is **≤ 0.588** (≈0.50–0.41 as F44/F47 measured), so **even a perfect judge fails** (gain ≈ 0 to −0.046); the judgment is already **banked** (archive `modality_cues`, `d0f9e7b`, full dev coverage), so no generation is owed and the $0 closure probe was **declined** per the ceiling-below-bar precedent | MJ_FORENSIC_RECON.md · `d57d05d` |
| **FA** — modality-reweighted / cross-encoder fusion: does the F44-cancelled Qwen-text gain convert on MHC-EN? | $0 gate, KILL: within-Qwen reweight is a **pure rotation** at every weight (F44-exact +0.040 hate / −0.036 non-hate at 50/50); cross-encoder `CLIP-imĝ ⊕ Qwen-text̂` lifts the MHC-EN dev AUC to **0.898 — the highest measured in the campaign** — yet unconvertible: the sole point-Pareto config (Δacc +0.050) fails the bootstrap CI ([−0.0625, +0.150]) and the selection-null (p=0.766), and its **label-oracle-threshold** edge is **+0.025 < +0.03** (ported B5 kill-switch fires); calibrated (HateMM positive control +0.0467 passes). Fifth "better-signal / no-conversion" datum; corrects the F44 concat→align(Hadamard) erratum (F44 numbers stand via the sign-faithful proxy) | FA_GATE_RECORD.md · `e0877c9` |

The round-4 recon that framed these gates is itself part of the closure: the **wave-4 candidate
enumeration** found the frozen pool empty of goal-hitting candidates and surfaced the F44 concat→align
erratum that made the FA cell measurable (`6032d32`), and the **wave-5 adaptation-family recon**
established a two-object closure — an adaptation touches either the encoder (generic LoRA, D7-encoder-
class) or the joint encoder+decision (the retrieval loss into the LoRA, which is exactly the killed
P9b object), with no third adapted object, so the one fresh member (a retrieval-mined hard-negative SFT
curriculum) opens no new dataset and is held behind a user D7 sub-ruling (`7166232`)
[DOC:TERMINUS_round3_mllm_plus3.md]. The **round-4 line-A** measurement — a generic encoder-level
LoRA-HateMM 3-seed encoder run — has now **completed** (job chain 13233→13234→13235; verdict `6b8f634`);
it is an encoder-class lever regardless of outcome (D7), so it does not alter the 13-route campaign
accounting or the novelty verdict, but its performance result is material and is reported next.

**LoRA-HateMM (round-4 line-A) — encoder-level LoRA passes on HateMM under both protocols.** The line-A
cell replaces the frozen-CLIP front-end with an encoder-level LoRA-SFT-adapted Qwen2.5-VL-7B encoder on
HateMM (r16/α32, generative word-label SFT on HateMM's own 743-video train split, vision tower and
projector frozen so only the language backbone moves), features read by the unchanged archive-OFF RGCL
align-fusion head + top-20 kNN, 3 head-seeds paired vs the banked frozen-CLIP floor — the same protocol
as B3. It was measured (job chain 13233→13234→13235) and **PASSES the +0.03/+0.03 conjunct under both
protocols with 3/3 sign** [DOC:LORA_HATEMM_VERDICT_REVIEW.md, commit `6b8f634`; provenance chain
`edeaedc`→`3ebd880`→`2e41332`→`8de0991`→`56a732a`→`6b8f634`].

**Table 7. LoRA-HateMM 3-seed paired (LoRA-Qwen − frozen-CLIP; both protocols).**

| seed | protocol | LoRA acc / mF1 | CLIP floor acc / mF1 | Δacc | ΔmF1 |
|---|---|---|---|---|---|
| 0 | val-sel | 0.8605 / 0.8521 (e19) | 0.8279 / 0.8172 | +0.0326 | +0.0349 |
| 1 | val-sel | 0.8698 / 0.8620 (e14) | 0.8279 / 0.8163 | +0.0419 | +0.0457 |
| 2 | val-sel | 0.8558 / 0.8495 (e22) | 0.8047 / 0.7920 | +0.0511 | +0.0575 |
| **mean** | **val-sel** | **0.8620 / 0.8545** | **0.8202 / 0.8085** | **+0.0419** | **+0.0460** |
| 0 | final-ep | 0.8651 / 0.8580 | 0.8186 / 0.7997 | +0.0465 | +0.0583 |
| 1 | final-ep | 0.8744 / 0.8660 | 0.8047 / 0.7822 | +0.0697 | +0.0838 |
| 2 | final-ep | 0.8698 / 0.8613 | 0.8140 / 0.7988 | +0.0558 | +0.0625 |
| **mean** | **final-ep** | **0.8698 / 0.8618** | **0.8124 / 0.7936** | **+0.0573** | **+0.0682** |

Source: LORA_HATEMM_VERDICT_REVIEW.md §1–§2 (job 13235 LoRA head trainlogs; floors re-parsed from the
banked 12850 frozen-CLIP logs), commit `6b8f634`.

**Kill-switch rulings and compliance.** Both KS-1 conjuncts (mean Δacc AND mean ΔmF1 ≥ +0.030) clear on
each protocol independently with 3/3 sign — val-selected +0.0419 acc / +0.0460 mF1 (cushion +0.0119 /
+0.0160 over the bar), final-epoch +0.0573 / +0.0682 (cushion +0.0273 / +0.0382) — so, unlike B3, this
is **not** a marginal pass (the val-sel acc cushion is ≈ 9× B3's +0.0013). The two honesty flags do not
fire: **KS-2** (family-coherence) is **not tripped** — at final-epoch LoRA 0.8698 ≥ frozen-Qwen 0.8682
(+0.0015 acc / +0.0026 mF1), and at val-selected LoRA 0.8620 sits inside the 0.014 seed band below
frozen-Qwen 0.8729 — so the LoRA pass is not a below-frozen degradation; **KS-3** (P9 regime echo) does
**not** fire, LoRA landing far above the CLIP floor rather than below it, confirming the encoder-level
regime converts on HateMM where P9's decision-level C3-knn regressed −4.7. Compliance is clean:
hash-freeze matched byte-for-byte at submit time; the head runner's `run_rac.py` argv is byte-identical
to the 12850 CLIP control (only `--model` and a fresh group changed); one budgeted test-touch per
dataset; single encoder draw as pre-declared (the ±band is head-seed variance, not SFT-draw variance,
symmetric with the single-draw CLIP control). One non-material deviation is flagged honestly: the LoRA
head ran under a newer `run_rac.py` carrying seven additional TARC/oracle argparse fields absent in the
12850 code, all set to their inert OFF values (provably no-op, and the identical condition under which
the already-accepted B3 verdict was rendered), plus one benign SFT-loss note (eval_loss 0.1084, a
slightly tighter generative fit than the MHC anchor's 0.1620); neither affects any kill-switch
[DOC:LORA_HATEMM_VERDICT_REVIEW.md §3–§4].

**Consequence for the goal (protocol-qualified, D7 still open).** The measurement's reach is bounded and
stated with its protocol qualifier: **under the final-epoch protocol, one lever — encoder-level LoRA —
now clears the +0.03/+0.03 conjunct on two datasets (HateMM +0.0573 / +0.0682 solid; MHC-ZH B3 +0.0313 /
+0.0453 marginal); under the val-selected protocol the same lever clears HateMM only (ZH val-selected
FAILs)**. This is the first time a *single* encoder lever passes ≥ 2 datasets — B3 alone was ZH-only, and
HateMM's *frozen*-Qwen pass is a different lever — but the two passes convert via the **same decisive
modality (text) through different levers**, so the conjunction is one lever with two mechanisms, not one
mechanism (analysis §3.9): on ZH the gain is text-borne and LoRA-specific (frozen-Qwen −0.0112), while on
HateMM the KS-2 result shows LoRA ≈ frozen-Qwen, so the gain over CLIP is substantially the frozen-Qwen
conversion that LoRA inherits and preserves — a per-stream decomposition measures that conversion as
text-carried on a swap-neutral image base, the frozen swap already converting HateMM's text signal to a
Pareto so LoRA's further text-sharpening adds ≈ 0 [DOC:HATEMM_LORA_STREAM_DECOMP.md, commit `51eb95b`]. The bundled **B4-EN closure arm** (same job 13235) formally measures
the EN LoRA-encoder cell and **FAILs both protocols** — val-selected mean Δacc −0.0021, final-epoch
+0.0000, each ≪ the bar, its seed-0 anchor reproducing the pre-GPU forensic value exactly — upgrading the
round-2 #22 negative from a pre-GPU forensic close to a measured one and leaving EN's only formal encoder
pass the frozen swap. Whether the encoder-level LoRA performance conjunct counts toward the goal's
*novel* clause is the pending user D7 ruling; the LoRA-HateMM cell is an encoder-class lever regardless
of outcome and is **not** folded into any main table
[DOC:LORA_HATEMM_VERDICT_REVIEW.md; DOC:PAPER_MASTER_TABLES.md PUR-3, PUR-banner].

**Curriculum coupling probe (cand-2) — a memory-mined SFT curriculum ties generic LoRA on ZH and adds
over generic on HateMM val-selected only.** The round-4 closing probe asks whether *coupling the
retrieval memory into the adaptation objective* upgrades the generic LoRA leg from "encoder-class" to
"memory-coupled." The cell is a **confusion-weighted single-video SFT curriculum**: the RGCL memory's
leave-one-out kNN vote over the banked frozen-Qwen train features assigns each train video a
confusability weight, and that weight **reweights how often each SFT record appears** — the SFT records
are byte-identical to the generic-LoRA arm (same 8 frames, instruction, and word target), the **single
manipulated variable is example multiplicity**, and the reweighted multiset is capped to N_train so the
3-epoch step count is identical to generic (cost-neutral). Trained on each dataset's own train split only
(ZH the primary leg — strengthen B3's marginal pass; HateMM the hold leg — inherit its both-protocol
pass), the features feed the unchanged archive-OFF RGCL head + top-20 kNN, 3 head-seeds paired vs both
the frozen-CLIP floor (K-C2-1) and the generic-LoRA arm (K-C2-2), dual protocol
[DOC:CAND2_CURRICULUM_PREREG.md, commit `76ef0e2`; DOC:experiments/exp-cand2-curriculum.md].

Table 8 gives the outcome. Against the frozen-CLIP floor the curriculum **holds** every inherited pass —
ZH final-epoch (marginal), HateMM both protocols — but the add-over-generic bar (K-C2-2: mean Δacc ≥
+0.010 AND sign 3/3 AND ΔmF1 ≥ 0) is met on **exactly one cell**: HateMM val-selected (+0.0155 acc /
+0.0166 mF1, 3/3). That pass rested on a single curriculum SFT draw (pre-declared F0.2); a pre-registered
second draw (rep2, F59 — see the draw-2 paragraph below) now makes the HateMM val-selected add-over-generic
**pooled weakly-hardened across two draws (5/6 sign), per-draw 3/3 gate not met** — and a protocol-split
caveat travels with it (HateMM final-epoch **ties** at +0.0093 acc, 0.0007 below the +0.010 bar), and it
lands off the a-priori-favoured leg. ZH **ties** generic on both protocols — the prereg's own
pre-declared most-likely outcome (F0.7: "generic LoRA with reshuffled data"). No kill-switch fired
(KS-regression and KS-below-floor both untriggered); compliance was clean (same-code pairing
76/80 Namespace fields identical, single test-touch per dataset, the confusion-weighting class-balance
shift pre-declared F0.8).

**Table 8. cand-2 curriculum (curric 3-seed mean; Δ vs frozen-CLIP = K-C2-1; Δacc vs generic-LoRA =
K-C2-2; both protocols).**

| Dataset | protocol | curric acc / mF1 | Δ vs CLIP acc / mF1 | Δacc vs generic (sign) | K-C2-2 |
|---|---|---|---|---|---|
| MHC-ZH | val-sel | 0.8255 / 0.7947 | +0.0179 / +0.0271 | −0.0067 (1/3) | tie |
| MHC-ZH | final-ep | 0.8523 / 0.8249 | +0.0380 / +0.0529 | +0.0067 (2/3) | tie |
| HateMM | val-sel | 0.8775 / 0.8711 | +0.0573 / +0.0626 | +0.0155 (3/3) | **pass** |
| HateMM | final-ep | 0.8791 / 0.8726 | +0.0667 / +0.0790 | +0.0093 (3/3) | tie |

Source: T5.4 [DOC:PAPER_MASTER_TABLES.md], from CAND2_VERDICT_REVIEW.md (job 13241), commit `546acc5`.
Comparison floors/generic arms re-derived to 4dp against the prereg (ZH generic-LoRA job 13150, HateMM
generic-LoRA job 13235, frozen-CLIP 13115/12850).

**Verdict (binding, per the frozen prereg §7.3, verbatim).**

```
ZH:     final-epoch: PASS (K-C2-1, MARGINAL) · K-C2-2: tie · ZH-robustness: not strengthened.
        val-selected: FAIL (K-C2-1)          · K-C2-2: tie.
HateMM: final-epoch: PASS (K-C2-1, hold)     · K-C2-2: tie.
        val-selected: PASS (K-C2-1, hold)     · K-C2-2: pass (single-draw caveat, F0.2).
```

The pre-declared **ZH-robustness** clause — the primary declared purpose of cand-2, to strengthen the
marginal ZH leg — is **not** met: neither does the val-selected conjunct pass, nor does final-epoch
become non-marginal (ZH final-epoch curric +0.0380 acc is below the +0.040 non-marginal bar, seed-2
+0.0134 below the per-seed bar — essentially B3's status). The memory→adaptation coupling's measurable
effect over generic LoRA is therefore **dataset- and protocol-local**: present only on HateMM
val-selected (pooled weakly-hardened across two draws, 5/6 sign; per-draw 3/3 gate not met — see the
draw-2 paragraph below), absent on the primary ZH leg. cand-2 opens **no new dataset** (pre-declared
F0.4) and is **not folded into any main table**; whether its one-cell add-over-generic suffices for the
D7 memory→adaptation-coupling novelty sub-ruling is the user's decision, not this experiment's
[DOC:D7_RULING_DOSSIER.md, commit `def6ce3`; DOC:PAPER_MASTER_TABLES.md PUR-4, PUR-banner].

**Curriculum coupling probe — draw-2 replication (rep2, F59): the HateMM add-over-generic is pooled
weakly-hardened, not a clean replication.** Because the draw-1 K-C2-2 HateMM val-selected pass rested on a
single curriculum SFT draw (pre-declared F0.2), a pre-registered second draw (rep2) was run on **HateMM
only** — seed = 1 the single manipulated variable (draw-1 was the HF default 42), curriculum multiset
bit-exact to draw-1 — to test whether it replicates. Draw-2 val-selected add-over-generic is per-seed
[+0.0139, −0.0047, +0.0233], mean **+0.0108** acc (ΔmF1 +0.0120): the point bar (mean Δacc ≥ +0.010) is
cleared but the **3/3 sign gate fails** (seed1 −0.0047 → 2/3), so the binding primary bar (K-REP-1) does
**not** pass, and the retirement kill (KS-REP; fires iff mean Δacc ≤ −0.014) does **not** fire. The pooled
two-draw read (K-REP-2: draw-1 [+0.0186, +0.0046, +0.0233] + draw-2 [+0.0139, −0.0047, +0.0233]) is mean
**+0.01317** acc at **5/6** positive sign → **HARDENED**. The binding verdict is therefore *F56 HateMM
val-selected add-over-generic = **WEAKLY-HARDENED*** — it did not fully replicate (per-draw 3/3 gate missed
on seed1), did not reverse, agreed in direction pooled, and remains a 2-draw estimate; the single draw-2
attempt is binding and consumed (no further draws). Non-binding final-epoch add-over-generic replicated
cleanly (mean +0.0140 acc, 3/3). rep2 measured HateMM only, so the ZH leg is unchanged and
**ZH-robustness remains not strengthened**; novelty stays the pending user D7 sub-ruling, not folded into
any main table [DOC:CAND2_REP2_VERDICT_REVIEW.md, commit `aa48275`, job 13246; frozen rep2 prereg
`2d15ffb`; provenance `2d15ffb`→`e2aee03`→`6c11988`→`d06ad07`→`aa48275`].

```
HateMM draw-2: K-REP-1 (val-sel add-over-generic): NOT-PASS (mean +0.0108 acc, sign 2/3, ΔmF1 +0.0120).
               K-REP-2 (pooled 6-pt): HARDENED (pooled mean +0.01317 acc, sign 5/6).
               KS-REP: NOT fired.  final-ep add-over-generic (non-binding): mean +0.0140 acc, sign 3/3.
VERDICT: F56 HateMM val-sel add-over-generic = WEAKLY-HARDENED.
```

**Premise-(d) gate — even an *adapted* text stream does not convert MHC-EN.** The FA gate (F50) closed
EN's *frozen* cross-encoder composition but carved out one untested cell in its own ban language —
"conversion requires adaptation" — namely CLIP's healthy image stream composed with the **LoRA-adapted**
(not frozen) Qwen text stream (`CLIP-imĝ ⊕ LoRA-Qwen-text̂`). A $0 CPU gate measured it, reusing the FA
oracle machinery verbatim: the frozen-text control arm reproduces FA-A2 **bit-exact** (max absolute
difference 0.000000, peak AUC 0.8982). Swapping the frozen Qwen-text block for the LoRA-EN-adapted block
does **not** close the +0.005 oracle gap — the maximum label-oracle `d_oracle` anywhere on the grid
stays pinned at **+0.0250** (identical to frozen, below the +0.03 bar, so the ported B5 kill-switch
fires) — and the adapted text stream actively **degrades** the composition: peak dev AUC drops **0.8982 →
0.8698 (−0.0284)**, the mirror image of ZH, where the same LoRA lifts the text stream (F45: 0.847 →
0.925). The single point-Δacc config (+0.050) is non-Pareto (a −0.0545 non-hate cost), fails the
bootstrap CI ([−0.0503, +0.1625]) and the selection-null (p = 0.7532); the identical test passes on
HateMM's genuine win (+0.0467), so the kill is calibrated. This closes MHC-EN at **all three composition
levels — frozen (F50), collapsed-adapted (B4/F53), and healthy-image ⊕ adapted-text (premise-(d)) — and
is the campaign's sixth "better-signal / no-conversion" datum** [DOC:PREMISE_D_GATE_RECORD.md, commit
`6e6061b`; DOC:experiments/exp-premise-d.md; analysis §3.6]. It spends no GPU and no test-touch (train +
dev features/labels only) and does not revise the 13-route campaign accounting (T4).

---

*Consistency note: all numbers in §1–§6 are transcribed from `PAPER_MASTER_TABLES.md` (T1–T3) and its
source documents; §7 is transcribed from the round-2/3/4 terminus maps and the individual verdict/record
files cited inline (numeric-provenance discipline). No discrepancy against the master tables was found
during drafting. The three known tensions carried forward — the ZH/EN "our best" single-config vs
multi-seed sourcing in Table 2 (‡), the ZH ≥ 0.85 dual-calibration headline (D2), and the round-4
ledger-ordinal discrepancy (master-table tension list #9) — are surfaced in-text rather than silently
resolved, per the master-table tension list #1–2, #9. The rounds-2/3/4 extension adds pre-registered
negatives and one round-4 performance positive (LoRA-HateMM, F53) without revising the 13-route campaign
count (§7 count-discipline note); the LoRA-HateMM per-seed numbers and floors are transcribed from
`refine-logs/LORA_HATEMM_VERDICT_REVIEW.md` (`6b8f634`) and re-checked against the job 13235 trainlogs.
The round-4 closing pair — the cand-2 curriculum verdict (Table 8) and the premise-(d) EN-composition
gate — are transcribed from `refine-logs/CAND2_VERDICT_REVIEW.md` (`546acc5`, job 13241, hash-verified
vs the frozen prereg `76ef0e2`) and `refine-logs/PREMISE_D_GATE_RECORD.md` (`6e6061b`) respectively;
cand-2 is a coupling probe held pending the D7 sub-ruling (opens no new dataset), and premise-(d) is a
$0-gate negative (sixth "better-signal / no-conversion" datum), neither folded into any main table.*

---

## 8. Rounds 5–6: post-terminus robustness audit (constraint-box unchanged)

After round 4 emptied the frozen constraint box (§7), two further **post-terminus** rounds re-opened every
gap that the four structural laws had closed by *prose* rather than by measurement, on a standing user
directive to keep auditing. Round 5 was a three-agent red-team that refuted the exhaustion claim at the
**enumeration** level — six cells had been argued-down but never measured [DOC:REDTEAM_UNTESTED_CELLS.md,
commit `adb8bc2`; DOC:REDTEAM_EXTERNAL_FAMILIES.md, commit `d0f91a5`; DOC:REDTEAM_BAN_SCOPE_AUDIT.md, commit
`5dd23e4`] — and then measured all six dead. Round 6 was a two-wave literature sweep that surfaced a handful
of borrowable operators and measured each one dead or parked. **These rounds add no route to the 13-route
campaign count (T4) and revise no number in T1–T4**: they are labelled post-terminus audit/robustness rounds,
not new campaign routes, and the project's best numbers (HateMM cand-2 0.8775 / 0.8791, §7) are **unchanged**
after ~16 GPU-h (round 5) + ~3.5 GPU-h (round 6). The scientific yield is confirmatory — every law survived a
direct attack — plus three mechanism sharpenings folded into the analysis chapter: the ISR β-decomposition
that makes Law I arithmetic (analysis §3.6), the bidirectional-attention crater that directly confirms the
causal-prefix closure (analysis §3.7), and two small-head optimization-landscape notes (analysis §3.10).

**Count discipline (carried).** T4's thirteen routes are the load-bearing campaign accounting and are
untouched here; the round-5/6 cells below are **audit measurements of prose-argued gaps**, banked as findings
F61–F74, and are deliberately kept off the campaign-route count and off the negative-result ordinal (per the
master-table tension list #9 round-by-round framing). No number below is new — each is transcribed from its
committed verdict/record with the commit cited inline and re-checked against the primary logs.

**Table 9. Round-5 red-team audit (six prose-argued gaps, all measured dead).**

| Direction | Epitaph (one line) | Verdict · record |
|---|---|---|
| **LP** — label propagation / graph diffusion over the kNN memory graph (decision *topology*) | $0 gate, KILL all 3 datasets: multi-hop LLGC over the same frozen keys is monotone-negative in diffusion strength (HateMM best −0.0187, MHC-ZH −0.0385 / α=0.9 catastrophic −0.19/−0.22, MHC-EN +0.0125 = net +1 item inside the perm-null p95 +0.063 whose centre is *positive*); one-hop already at the 1-hop-separable ceiling; MHC-ZH oracle headroom +0.1026 unconverted (Law I 7th) | LP_GATE_RECORD.md · `7be6e3f` |
| **SWA** — single-trajectory stochastic weight-averaging of per-epoch head checkpoints (attacks the F45 dev-selection tax) | $0 probe, KILL both datasets: HateMM SWA lands 0.9–6.6 dev-acc pts below the val-sel max on the two seeds with a real selection gap (mid-peak dev curve, averaging cannot recover it); MHC-ZH regen (job 13294, G-repro bit-exact) is a *dev-underpowered* KILL (cond_A 0/3; 78-item dev jitter = the effect size). Governance: single-trajectory weight-avg needs a user micro-ruling vs the cross-seed-ensemble veto before any claims-table entry | SWA_PROBE_RECORD.md · `5a40bb1`/`17db531` |
| **Learned audio** — Whisper-large-v3 encoder hidden-state stream (the never-screened MHC-EN audio blank cell) | $0 gate, KILL all 3 datasets both Z-arms: mean⊕max 2560-d video vectors add zero conditional info (HateMM +0.0014; MHC-EN +0.0041 deployed / −0.0013 strict; MHC-ZH −0.0052/−0.0082; all CIs straddle 0, calibration accZA=1.0); the ASR transcript already banks the spoken-hate content, so **no oracle surplus — signal itself absent, not a Law I datum**. Closes the EN audio blank cell; Whisper realization only (AST/BEATs stay download-gated) | LAUD_GATE_RECORD.md · `3573f82` |
| **Vision-unfreeze LoRA** — unfreeze the ViT tower + projector inside LoRA-SFT (the un-enumerated representation cell) | 3-seed verdict, ~15 GPU-h: EN image stream **MOVED** (+0.0320 train-LOO / +0.0065 dev, reviewer bit-for-bit — first lever to move it, refutes the F51/GAP-5b "no vision lever" wording) but K-V2 = **TIE** both datasets both protocols (HateMM val-sel −0.0016 acc 0/3, final +0.0000 1/3; MHC-EN val-sel +0.0269 acc sign 2/3, final −0.0062 1/3) — image moved, head converted zero (Law I 8th) | VISION_UNFREEZE_VERDICT_REVIEW.md · `09d02f8` |
| **ISR** — independent per-segment re-encode read by a uniform per-segment-kNN vote-mean (last aggregation object) | $0 pre-gate, NO-GO: legal uniform operator flat (HateMM +0.0012 / MHC-EN +0.0032, under perm-null, boot-5th < 0, vote bit-exact Fano 1.0); decisive β-decomposition proves the oracle headroom **selection-locked** — HateMM +0.0776 = +0.0012 legal + +0.0764 banned, MHC-EN +0.0700 = +0.0064 + +0.0636 (91–98% banned-selection-only) ⇒ Law I is now arithmetic; Qwen per-segment extraction never happens, 0 GPU-h | ISR_PREGATE_RECORD.md · `a6e41f8` |
| **Frame-16** — double visual sampling 8→16 frames through the frozen encoder | 3-seed verdict vs banked 8f floor, ~0.6 GPU-h: val-sel mean −0.0077 acc (0/3), final +0.0015 (1/3); KS-16f-dead KILLED both protocols ⇒ cell CLOSED and the expensive LoRA-16f stage-2 AUTO-DEAD (pre-declared spend verdict); 8 frames is not the bottleneck — the pooled representation saturates at 8f | FRAME16_VERDICT_REVIEW.md · `32c2e6f` |

**Table 10. Round-6 literature-sweep audit (borrowable operators, all measured dead or parked).**

| Direction | Epitaph (one line) | Verdict · record |
|---|---|---|
| **Grad-norm selection** — validation-free checkpoint selection by minimum head-gradient norm (arXiv 2601.16874; attacks the F45 ZH selection tax) | $0 probe, MECHANISM REFUTED: the paper's premise (Spearman(‖g‖, acc) ≈ −0.85…−0.98) **inverts** on our tiny head (+0.61/+0.72/+0.62, 3/3 seeds); scale-normalised grad rises monotonically *with* accuracy, argmin lands at the worst epoch; F68-P2 killed at $0 (the promotable ZH/HateMM-curric ckpts were disk-pruned to B2, and a restore is pointless given the sign flip) | GRADNORM_SELECT_PROBE_RECORD.md · `ada5849` |
| **Readout axis** — intermediate-layer / one-word-prompt / last-token extraction variants (the one un-enumerated axis inside the MLLM-embedding paradigm) | $0 CPU screen, KS-readout-dead (~2 GPU-h extraction only): MHC-ZH best +0.0128 dev-query, HateMM best +0.0093 LOO — both inside the perm-null band (p95 +0.0769 / +0.0939, boot-5th < 0); one-word readout actively regresses HateMM (−0.056/−0.065); the deployed final-layer mean-pool is already at the local optimum, no head, zero test-touch | READOUT_SUBMIT_RECORD.md · `a60f6cf` |
| **MCR** — modality-competition rebalancing / data-remixing SFT schedule (force the collapsed EN image stream to carry load during adaptation) | forensic recon, PARKED (no GPU): an honest transplant exists (EN-only masked-SFT schedule, ~4–6 GPU-h) but F65 already nulled the same axis (image moved, zero conversion) and F55 caps the EN stream-rebalancing oracle at +0.025 < the +0.030 bar ⇒ arithmetic-capped, prior ~5–8%; available as a user-gated paper-closure null, not a performance bet | MCR_FORENSIC_RECON.md · `6d0495b` |
| **Bidir mask-flip** — training-free causal→bidirectional attention on the LoRA-Qwen decoder (LLM2Vec / NV-Embed recipe; highest-novelty candidate) | 3-seed verdict, ~1.2 GPU-h: **DEGRADE both datasets** — MHC-ZH mean −0.1163 (val) / −0.1409 (final) acc, HateMM −0.1210 / −0.1256, 0/12 per-seed deltas positive, up to −0.28 macro-F1; the "Llama-pattern" crater (≈7–10× the −0.014 line) directly confirms the deployed reps exploit the causal prefix (analysis §3.7); Stage-2 MNTP routed to a user funding decision, not auto-defunded | BIDIR_STAGE1_VERDICT_REVIEW.md · `f733bbe` |
| **Head-recipe** — SAM flat-minima optimiser + modality-dropout on the align head | 3-seed verdict, < 0.15 GPU-h: all 4 arm×dataset cells KS-arm-dead, FORMAL-FAIL both protocols — SAM×ZH −0.0246/−0.0424 (hurts), SAM×HateMM +0.0047/+0.0046 (within-noise, not 3/3), mod×ZH 0.0000/−0.0313, mod×HateMM −0.0201/−0.0062; both disclosed headwinds (F69 wrong-sign SAM, F45/F58 text-carried mod-dropout) borne out | HEADRECIPE_VERDICT_REVIEW.md · `8e60f42` |

Two round-6 recon-stage companions round out the ledger at zero GPU. The merged three-agent lit-survey (F68)
and the round-2 sweep (F74) both concluded, independently across six agents, that **no borrowable operator
carries a strong prior on a new dataset** and that the binding walls are the MHClip label limit, the F66
arithmetic, and the HateMM ceiling — what could clear the goal is a user ruling (ZH protocol, goal
renegotiation, or a model download), not an operator [DOC:LITSURVEY_NOVEL_MECHANISMS.md;
DOC:LITSWEEP2_FRESH_2026.md]. A within-noise observation to record (never a claim): the SAM×HateMM cell nudges
the val-sel mean up +0.0047 acc and its best single seed reaches **0.8884** val-sel — the highest single
HateMM value seen anywhere — but the mean is far below the bar and not 3/3-signed, so it is not folded into any
table. External validation from the same sweep is corroborative rather than competitive: the published HateMM
leader **MM-HSD** \cite{mmhsd} reaches 0.878 macro-F1 but only through the OCR channel we veto — 0.845 without
it, inside the band of our 0.8775 / 0.8791 — so its lead is entirely the vetoed channel, and the SOTA
video-embedding works (VLM2Vec-V2, VidVec) use **no temporal operator at all**, independently corroborating the
F35 / F37 / F67 temporal closure [DOC:LITSWEEP2_FRESH_2026.md; DOC:LITSURVEY_MLLM_EMBEDDING.md].

*Consistency note (§8): every number is transcribed from the named round-5/6 verdict/record files
(numeric-provenance discipline) and re-checked against the primary logs; none revises T1–T4, and the 13-route
campaign count is untouched — these are post-terminus audit rounds, not new campaign routes, banked as findings
F61–F74 and deliberately kept off the negative-result ordinal (master-table tension list #9).*

The audit then ran three further round-6 waves (litsweep2 batch-3, litsweep-3 batch-4, litsweep-5) as findings
F75–F82, on the same discipline. Every cell below is again a null, a $0-recon PARK, or a $0 pre-gate PARK; none
revises T1–T4, none adds to the 13-route count, and the project's best numbers (HateMM 0.8775 / 0.8791, ZH final
0.8456 / 0.8173) are unchanged.

**Table 11. Round-6 waves 3–4 audit (F75–F80: measured cells + $0 recon-parks, all null or park).**

| Direction | Epitaph (one line) | Verdict · record |
|---|---|---|
| **NCA / soft-kNN head-loss family** — swap the head's triplet+BCE toward a vote-consistent (NCA τ0.1/0.2), contrastive (SupCon), or mixup-BCE objective (4 arms — the losses that most directly optimise the deployed kNN vote) | 1-bite 3-seed verdict, ~0.33 GPU-h (job 13482): **0/8 FORMAL, 7/8 KS-arm-dead**; family-max A3-mixup ZH final +0.0134 (2/3), sole KS survivor NCA τ0.1×ZH val-sel +0.0112/+0.0113 (3/3 sign) *below* the ±0.014 band = within-noise hardening, D7-dead; **first measured negative for trained-reshaping-unlocks-oracle-headroom** ⇒ Law I holds against a trained operator (analysis §3.10). Codex gate caught + re-freeze fixed an A3-only dropout-mode confound pre-spend | NCA_VERDICT_REVIEW.md · `f03cae0` (+ REFREEZE `8f08e9f`/`467a6f4`) |
| **Spatial resolution** — raise the per-frame `max_pixels` cap (151200 px) toward native source resolution (the last virgin input-fidelity axis) | $0 forensic recon, PARK: the litsweep2 "~6.5×" downscale premise is a fabricated 720p figure — ffprobe on raw sources gives **HateMM 2.71× (480p) / MHC-EN 10.55× / MHC-ZH 13.71× (1080p)**; headroom is *anti-correlated with conversion* (HateMM, the only image-converting dataset, is nearly native; EN-collapsed / ZH-marginal have the pixels), mean-pool attenuates, F65 law-I + F70 readout-null both bind, and extraction is raw-video-bound ⇒ **no Modal-triage path**; ≥+1 HateMM ~5–10%, ≥+3-on-2 <3%; a ~1 GPU-h HateMM@410k door-closer is spec'd, unrun | RESOLUTION_FORENSIC_RECON.md · `5c6075b` |
| **Memory-bank curation** — train-label-only prune / prototype-select / class-balance of the deployed kNN bank (LOO-influence / Data-OOB; automates the banked human-2-entry-EN deletion) | $0 forensic recon, PARK: the "$0 on banked keys" premise is **false** — the deployed vote indexes the *trained head embedding* and all 6 floor head ckpts (13150/13241 × 3 seeds) are disk-deleted, so a faithful multi-seed pregate needs a ~0.3 GPU-h re-mint; the only $0 object (raw fused key) is seed-independent → single-draw = the withdrawn archive-as-key failure class; F63 (1-hop-separable, positive perm-null) + W2-E (prototype dead) + Wall-C cap the prior; ≥+1 ~5–8%, +3 ~1% | CURATION_FORENSIC_RECON.md · `7025391` |
| **ELR / noise-robust head** — additive early-learning regulariser (lead) + co-teaching (contrast) on the FAISS-mined pairs (outside the F75 ban letter) | $0 forensic recon, PARK: mined pairs are *gold-label-filtered* ⇒ "mined-pair noise" ≡ gold-label noise (pillar-3's object); ELR attaches to the BCE leg the kNN vote doesn't read (second-order); noise proxy 13–17% raw-space upper bound is boundary-hardness-dominated; **Wall-C quantified** (HateMM test peaks ep18/21/24, +4/+7/+14 after dev saturates; ZH final−valsel = +0.0134 ×3 seeds) makes early-target pulls anti-aligned; ≥+1 ~5–8%, +3 ~1–2%; 0.16 GPU-h probe unrun | ELR_FORENSIC_RECON.md · `9e41447` |
| **ZH Chinese-instruction re-extraction** — translate the deployed English extraction instruction/scaffolding to Chinese (the un-varied ZH-path axis; tests the SFT train/inference language-mismatch hypothesis) | 3-seed verdict, ~1.1 GPU-h (job 13487, KS-parity bit-exact): **both arms KS-dead both protocols** — LoRA −0.0358 val-sel / −0.0112 final acc, frozen −0.0336 / −0.0045; both val-sel legs past −0.014 (Chinese prompt *hurts*); mismatch hypothesis **refuted** (arms regress near-identically) ⇒ extraction-instruction-language axis CLOSED, D7-dead | ZHPROMPT_VERDICT_REVIEW.md · `1a8c5fe` |

Two zero-GPU lit-sweeps frame these cells. The three-agent litsweep-3 (F77) enumerated the selector-conversion,
ZH-specific, and training-data-centric lenses and converged — a fourth independent time after F68/F74 — on no
in-box operator carrying a defensible ≥+3-on-2-datasets prior, the binding walls being F66's arithmetic, the EN
label-limit, and the memorised-bank data-generating-process obstacle (CLIP LOO 0.998 makes every train-supervised
selector/reshaper target degenerate) [DOC:LITSWEEP3_SELECTOR_CONVERSION.md, commit `e103d54`;
DOC:LITSWEEP3_ZH_SPECIFIC.md, commit `d4af64b`; DOC:LITSWEEP3_DATA_CENTRIC.md, commit `8629188`]. The same sweep
**corrected two inherited premises** that bear on the paper's framing (both provenance-noted). (i) The ledger's
"ZH transcripts median ~4 words / degenerate" figure is a **whitespace-split artefact** — Chinese has no
inter-word spaces, so `text.split()` is meaningless; the deployed ZH text stream is a median **~106 Chinese
characters** (train 106 / val 108.5 / test 105) of Bilibili *description* metadata (not the near-empty Whisper
ASR), with the search-keyword slur often surfaced un-obfuscated in `<em>` markup — so the ZH stream is
content-rich, and its wall is 78-dev selection noise plus representation saturation (LoRA-Qwen ZH text-AUC 0.925),
not a degenerate transcript. (ii) The litsweep2 resolution "~6.5×" downscale figure is superseded everywhere by
the ffprobe-measured **HateMM 2.71× / EN 10.55× / ZH 13.71×** multipliers in Table 11
[DOC:LITSWEEP3_ZH_SPECIFIC.md, commit `d4af64b`; DOC:RESOLUTION_FORENSIC_RECON.md, commit `5c6075b`].

**Wave 5 — the datasets themselves at the published frontier (F81).** The final literature sweep treated the
datasets, not the pipeline, as the object and produced a load-bearing external calibration. On *legal* channels
we already sit at the 2025–2026 published HateMM frontier and at/above it on MHClip-EN. Table 12 lists each
method on its own protocol (5-fold CV or a custom split) — **ordering and channel-legality only, never a paired
4dp comparison** with our n = 215 clean test.

**Table 12. Published HateMM frontier vs the house configuration (each method's own protocol; ordering and
channel-legality only).**

| Method (year) | Channels | HateMM acc / mF1 | In our box? |
|---|---|---|---|
| Das 2023 (dataset paper) | text + audio(MFCC) + video | 0.798 / 0.790 | audio download-gated |
| Wang 2025b (Vid+RM-FT) | text + audio + video | 0.820 / 0.820 | audio download-gated |
| CMFusion \cite{cmfusion} | Whisper-text + MFCC + ViT, gated add-fusion | 0.823 / 0.860 | no OCR; MFCC audio = our F41-killed eGeMAPS class |
| Xiong 2024 (TCE-DBF) | text + audio + video | 0.849 / 0.840 | audio download-gated |
| Koushik HCC1 \cite{koushik} | HateXplain-text + CLIP + CLAP-audio, late concat | 0.854 / 0.848 | legal iff CLAP download user-gated; no OCR; ablation +2.9 mF1 from a base below us |
| RAMF \cite{ramf} (Dec-2025) | text + audio + video + Qwen2.5-VL-32B counterfactual reasoning | 0.856 / 0.851 | 32B local = B2-dead, method = our P5/P10-dead; no OCR |
| MM-HSD \cite{mmhsd} (ACMMM-25) | transcript + audio(wav2vec2-xlsr) + video + **OCR** + cross-modal attn | 0.878 / 0.874 | **OUT — uses OCR** (ablation: drop any modality → mF1 0.815–0.845 ⇒ OCR load-bearing) |
| MultiHateGNN (BMVC-25) | multimodal GNN | – / 0.771 | below field |
| **HOUSE (curric-LoRA, ours)** | LoRA-Qwen dual-stream + RGCL + kNN, no OCR / no audio | **0.879 / 0.873** | in-box |

Every published method that is legal-in-box (Koushik 0.848, CMFusion 0.860, RAMF 0.851, Xiong 0.840, Wang 0.820
macro-F1) sits at or below our 0.873; the single method above us — MM-HSD 0.874 — buys its edge with the vetoed
OCR channel, and its own ablation proves that channel load-bearing (dropping any modality costs 0.03–0.06 mF1).
There is thus **no legal, published route to HateMM > 0.88**: HateMM is near-ceiling for our channel set, and the
OCR veto — not method weakness — is the 0.874 gap. The same reading holds on MHClip-EN, where the published
frontier (RAMF 0.740 / 0.717, coarse video-level 0.684 / 0.644 per yang et al. \cite{yang2025}, GPT-4V multiclass
0.63 mF1) sits *below* our ≈ 0.79–0.81 — confirming EN is label-limited at the *field* ceiling, not
method-limited [DOC:LITSWEEP5_HATEMM_EN.md, commit `36d833e`]. Concurrent temporal work corroborates the closure:
TANDEM \cite{tandem} (arXiv 2601.11178, May-2026) adds scene-change keyframe selection + RL over optional gold
temporal/target-identity annotations and reaches HateMM 0.78 / 0.79 and MHClip 0.67 / 0.38 — below us and
out-of-box on several constraints [DOC:LITSWEEP5_TEMPORAL.md, commit `ad81ffb`]. (HOUSE 0.879 / 0.873 are the
curric-LoRA final-epoch numbers 0.8791 / 0.8726 at the external-paper 3dp convention; they revise nothing in
T1–T4.)

Two wave-5 companions round out the ledger at zero GPU. **F82 (graded 3-class soft-label)**, the sweep's
top EN-revival longshot, is measured PARK by a $0 pre-gate (§3.10 of the analysis): the fully gold-cheating
oracle for any monotone Offensive reweighting reaches only EN +0.0250 / ZH +0.0256, both below +0.030, the
honest proxy is monotone-negative at every τ, and the true-Offensive effect does not beat its F63 permutation
null — the F44 within-positive wall on the label axis, at 0 GPU-h with bit-exact machinery parity
[DOC:GRADEDLBL_PREGATE_RECORD.md, commit `c4333ce`]. **F81's adversarial completeness audit** (the F61 job redone)
finds enumeration is *not literally* exhausted — ≥ 7 never-/under-measured in-box cells plus two ban-scope
letter-overreaches survive an adversarial read — but every surviving cell is D7-novelty-dead, F66-arithmetic-capped,
or park-priced < 3 %, and the goal's remaining upside lives only behind user gates (ZH val-selection retirement,
which would make ZH a second passing dataset under a single final-epoch protocol, ranked far above any operator or
download) [DOC:LITSWEEP5_COMPLETENESS.md, commit `4e3b09a`]. A data-level limitation surfaced by the same sweep:
MultiHateClip releases only the aggregated majority-vote label with **no per-annotator votes** (unlike HateXplain),
so the learning-with-disagreement lineage is foreclosed at the data level (limitations §3).

*Consistency note (waves 3–5): every number above is transcribed from the named litsweep2/-3/-5 verdict/record
files (numeric-provenance discipline) and re-checked against the primary logs; none revises T1–T4, and the
13-route campaign count is untouched — these remain post-terminus audit rounds, banked as findings F75–F82 and
kept off the negative-result ordinal (master-table tension list #9). Three inherited premises are corrected with
provenance (median-4-words → ~106 Chinese chars; 6.5× → 2.71×/10.55×/13.71×; test sizes EN 161 / ZH 149 / HateMM
215 verified against the wiki, which was already correct — the litsweep-5 correction was to the task prompt, not
these tables). External frontier numbers are each paper's own protocol and are kept strictly out of any 4dp
house comparison.*

**Round 7 — the last un-run fusion flag, and a code-first reproduction survey (F83–F85).** The audit's final
round changed instrument twice over. It spent the campaign's cheapest remaining GPU cell on the one first-class
head flag that had never been run on video, and it replaced paper-reading with **cloning and execution-triaging
the code** of adjacent-field 2025-H2 → 2026 work (the previous five sweeps opened no repository). Both cells
below follow the F61–F82 discipline: neither revises T1–T4, neither adds to the 13-route count, and the
project's best numbers (HateMM 0.8775 / 0.8791, ZH final 0.8456 / 0.8173) are unchanged.

**Table 13. Round-7 fusion door-closer and reproduction survey (F83–F85).**

| Direction | Epitaph (one line) | Verdict · record |
|---|---|---|
| **Trained fusion operator** — swap the deployed `align`/Hadamard fusion for a trained `concat`+MLP in the head (the only first-class `fusion_mode` branch never run on video; outside the F50 *fixed*-composition and F75 *loss*-swap ban letters, both over-reaches confirmed by recon) | 1-bite 3-seed × 2-dataset verdict, ~0.1 GPU-h, **zero code diff** (job 13514; branch-assert 6/6, 48/48 values agreed by three independent parsers, prereg re-hash match): **both cells KS-arm-dead** — ZH +0.0067 val-sel acc (2/3 sign) but −0.0045 final (1/3); HateMM −0.0031 / −0.0031 (0/3 on every leg); 0/4 legs near the FORMAL +0.030/+0.030 conjunct; no KS-regression (worst −0.0045 > −0.014). HateMM's whole effect ≤ **2 flipped items** on any seed (n = 215; 3 of 6 Δacc exactly 0.0000, the rest −1/−1/−2 items — correcting the source record's NB-3 count of four, which is inconsistent with its own D5(a) table and with the raw `RESULT_ROW` lines re-read here); ZH val-sel mean = **+1 item/seed** (n = 149) ⇒ fusion-operator axis **CLOSED as a measured null**; the null is about the **concat + 2.0× first-Linear bundle** (2,098,176 vs 1,049,600 params), never the operator alone, and never "capacity cannot help" | FUSIONCAT_VERDICT_REVIEW.md · `129fe2e` (recon FUSIONSWAP_FORENSIC_RECON.md · `934bc9a`) |
| **Reproduction survey** — adjacent-field (deliberately *not* hateful-video) 2025-H2 → 2026 work **with runnable code**, triaged by file tree + `py_compile` rather than by README | $0 (no GPU, no SLURM, no weight downloads): 8 repos cloned (91 MB, gitignored); shortlisted by inspiration value as **SynIB (arXiv 2606.09853) > LSMI (ICML 2025) > MokA (NeurIPS 2025 Oral)** but ordered *for execution* with **LSMI first**, as the $0 diagnostic that gates the SynIB port; UniME-V2 excluded at source (its MLLM-as-a-judge soft labels are exactly the banned P11/P2 supervision), VLM2Vec offers no pooling we lack (`last\|mean\|cls` only), **VidVec's `main` branch is empty** (no code exists) and **RASR was withdrawn by its authors** (v2, 2026-06-30); the HateMM dataset paper's own baseline code **does not compile as shipped** (`Codes/1.FastTextEmb_and_LASEREmbExtraction.py:45`, `SyntaxError`), while LSMI/SynIB/UniME-v2/BalanceBenchmark/MokA/VLM2Vec compile 4/4, 99/99, 95/95, 55/55, 214/214, 271/271. Coverage caveat stated in-record: the session's WebSearch quota exhausted at 200/200, so this is a **deep triage of 8 repositories, not an exhaustive enumeration** | REPRO_SURVEY_2025.md · `9367338` (+ ERRATUM `81e2eaf`) |

*Erratum carried into the paper's framing (F84).* The survey's §4.1/§6 description of SynIB's objective as a
"symmetric KL between the intact-input and the masked-input prediction" is **wrong at source level** and is
superseded everywhere by the port recon's reading of `synib_mask_model.py`: the intact prediction never enters
any KL; the live variants are a Gaussian KL to an uninformative N(0, I) prior, a Dirichlet KL, and a **forward**
KL to detached unimodal anchors (the Hateful-Memes configuration), and the only symmetric logit-KL helper in the
repository is commented out. The same reading finds three further upstream facts that any transplant claim must
respect — the `zeros | noise | ema` mask fills are dead code (batch-permutation fill is the live path), the
config `p` key is dead (`p_min = 0.30` binds), and the Hateful-Memes anchor heads carry **no gradient path**
(untrained random init). No sentence about SynIB in this paper may use the survey's original characterization
[DOC:REPRO_SURVEY_2025.md, ERRATUM commit `81e2eaf`; DOC:SYNIB_PORT_FORENSIC_RECON.md, commit `9e638ea`].

The survey's shortlist was then executed **top-down**, cheapest first, each item gated by the one before it
(Table 14). The ordering was itself the finding: the $0 diagnostic at rank 1 discharged its own question *and*
killed rank 2 by the port's own pre-declared kill-switch, leaving exactly one GPU cell to run.

**Table 14. Reproduction-inspiration campaign — the survey shortlist executed in order (F86–F87).**

| Item (survey §5 execution rank) | Outcome | Cost | Verdict · record |
|---|---|---|---|
| **LSMI** (#1) — sample-level PID (`R` / `U1` / `U2` / `S`) of the two deployed streams on the banked train + dev caches, all three lineages | **Gate returned, machinery certified first**: the released `d' = 64` recipe reads a *deterministic* XOR at chance (joint out-of-fold 0.513 / 0.530 / 0.508) ⇒ that layer measurement-invalid; certified `d* = 16` (replicated `d' = 8`, where a maximal synergy 0.6931 is recovered as 0.7077 / 0.7321 / 0.7105). At the certified dims: `S` = −0.0747 / −0.0802 / −0.0000 (ZH / HateMM / EN), ≤ 0 on 5 of 6 cells, perm-null q95 = 0, dev replicates; `U2` (text) largest atom on 5/6 cells (0.076–0.237); **`U1` (image) exactly 0.0000 on 5/6**; `I12` 0.149–0.359 nats. Mechanical label INDETERMINATE (ZH / HateMM) + FUSION_CAPPED (EN) — the synergy half fired everywhere, the *dominance* half failed because the pair is uniqueness-dominated, not redundancy-dominated (analysis §3.11) | **0 GPU-h** (CPU-only, no test touch) | LSMI_GATE_RECORD.md · `a8905ac` (chain `d4b06f0` → `362a60e`; withdrawal note `915a60d`) |
| **SynIB** (#2) — masked-branch information-bottleneck term added to (not replacing) the triplet+BCE hybrid | **PARK, unrun.** The port recon pre-declared "the LSMI reading" as its kill-switch; branch (a) — `s ≈ 0` on all datasets — fired, and an objective built to push a head onto *synergistic* structure has none to push onto (recon prior for the goal bar: 1–2 %). The conditional BalanceBenchmark screen (survey #4) never unlocks, being conditional on synergy existing to balance | **0 GPU-h** | SYNIB_PORT_FORENSIC_RECON.md · `9e638ea` (park recorded with the gate, `a8905ac`) |
| **MokA** (#3) — modality-routed LoRA (per-modality down-projection `A`, shared `B`, `r_v = r_t = 16`) inside the deployed ZH encoder-SFT; the PEFT-adapter-structure axis no banked adapter ever varied | **MEASURED — NOT PROMOTED.** `final-epoch: fail; val-selected: fail` against both floors; **+0.0000 acc on both protocols** vs the banked merged floor 13150. Drift gate fired 6/6 (worst mean per-item cosine 0.99954879 < 0.9999) ⇒ same-path unmerged floor mandatory and binding; against it the arm reads +0.0268 val-sel (3/3) — adjudicated **not attributable to routing**, being exactly cancelled by the unmerged path's own −0.0268 with routing absent (**+0.0000 = +0.0268 + (−0.0268)**, analysis §3.10). Stream decomposition = **null-op** (text FLAT ⇒ the prereg's own text-side bet refuted; image AMBIGUOUS ⇒ **the 9th law-I instance is NOT certified**; visual-protection narration barred). Test-touch 6 spent / 6 budgeted | **5.573 GPU-h** (jobs 13537 / 13551 / 13552 / 13566 / 13573 = 0.003 / 0.532 / 3.414 / 1.212 / 0.413; cap 4.70 ⇒ +0.87, **+18.6 % disclosed**, all mapped to pre-registered items) | MOKA_VERDICT_REVIEW.md · `91f64a6` (submit `ed609eb`, re-freeze `72a947b`) |

Two campaign-level notes. **Process.** The mandatory external code gate blocked the MokA family *before any GPU
spend* on two P1 defects — a modality-mask hook registered on `get_base_model()` that never fires on the
production `PeftModelForCausalLM` (that call chain reaches the base model by a direct `.forward()`, and
`nn.Module` hooks fire only in `__call__`, so `hook_calls = 0`), and a `median` computed as `vals[len(vals)//2]`
over 196 layers, an upper-neighbour order statistic — the second and third such catch after wave 3's NCA
dropout-mode confound; after the fix and re-freeze, routing was runtime-verified live (`hook_calls` 314,
`routed_calls` 77,224, `fallback_calls` **0**), so the null is functional rather than mechanical, and the durable
output of the cell is identity-controlled routed-LoRA machinery with zero vendored-tree edits and a bit-exact
default-equals-identity guarantee [DOC:MOKA_REFREEZE_FIX.md, commit `72a947b`]. **Accounting.** Round 7 cost
≈ **5.7 GPU-h** in total (5.573 MokA + ~0.1 fusion-concat), with the survey, the PID gate and the SynIB park at
zero; no cell revises T1–T4, no cell enters the 13-route count, and the project's best numbers are unchanged.

*Consistency note (round 7): every number above is transcribed from the named refine-logs verdict/record files
(numeric-provenance discipline) and re-checked against the primary logs and job outputs; findings F83–F87 are
banked as post-terminus audit cells and kept off the negative-result ordinal (master-table tension list #9). The
law-I instance count stays at **eight** — F87's candidate ninth is explicitly **not certified** (analysis §3.6,
§3.10). **Round-8 update: that non-certification stands, and a ninth instance was certified separately by the
Molmo2 encoder-swap probe (F91, §9), so the current count is nine.**

## 9. Round 8: reading the residual, then closing what it suggested (F88–F98)

Round 8 did what the previous rounds had not: it read the residual **per item** on all three datasets, and
then measured every repair that reading suggested. The organising economics are worth stating first, because
they are unusual for this project — of the eleven cells below, **eight cost nothing at all** (CPU-only, no
SLURM, no Modal, no training of any deployed arm, no new test inference), one gated channel cost zero
GPU-hours after a CPU extraction, and only the encoder swap and the three MNTP arms consumed GPU, for a round
total of **≈ 3.0 GPU-h**. As with rounds 5–7, **no cell revises T1–T4, none enters the 13-route campaign
count, and the project's best numbers (HateMM 0.8775 / 0.8711 val-selected and 0.8791 / 0.8726 final-epoch;
MHC-ZH final 0.8456 / 0.8173) are unchanged**. Findings F88–F98 are banked as post-terminus audit cells and
kept off the negative-result ordinal.

**Table 15. Round-8 forensics and the $0 in-box closures (F88, F89, F94).**

| Direction | Epitaph (one line) | Verdict · record |
|---|---|---|
| **ERRPAT** — per-item error forensics on all three datasets, $0, CPU proxies validated cell-by-cell to 4 dp | **Not a kill — the structural read of the residual, plus six measured nulls.** Error sets are **~90 % seed-invariant** (HateMM 24–25 of 26–28 wrong in 3/3 seeds; MHC-ZH 22 of a 25-item union, nothing at exactly 2/3; MHC-EN 22 consensus errors over 4 seeds against a 20-item flip band and 119 never wrong), and every error is a **confident neighbourhood inversion** (HateMM median top-20 true-label purity 0.1667, median \|vote\| 0.7267 vs 0.9873 for always-correct, top-1 neighbour correct on 7.4 % of errors vs 95.2 %; MHC-ZH core purity 0.1167 with 0 of 22 majority-correct; MHC-EN consensus correct-class fraction 0.2205 vs 0.4781 vs 0.8738). **Not a coverage failure**: MHC-ZH's first same-gold-class neighbour sits at **median rank 1.5** (11/22 at rank 1). Six new nulls, all door-closers: HateMM threshold recalibration (dev-fit +0.0000 / +0.0016; test-fitted oracle only +0.0078), length de-bias (−0.0016 train-LOO, sign-flipped on dev), LOO bank curation (+0.0016, **losing to random deletion of the same size** at +0.0031 / +0.0000); MHC-ZH test-fitted threshold oracle mean **+0.0201**; MHC-ZH Whisper-ASR text re-channel capped at **+0.0134** at $0; MHC-EN dev-selected threshold **0 of 6 arms** improving (−0.0083 Qwen / −0.0104 CLIP) ⇒ **the in-box $0 open set is empty on all three datasets** (analysis §3.12) | ERRPAT_{HateMM,MHC-EN,MHC-ZH}_2026-07-26.md · `ad56a62` |
| **MECHFIX** — five frozen eval-time vote operators vs the deployed top-20 rank-weighted signed-cosine vote (class-balanced quota / CSLS / Ledoit–Wolf whitening / exact 1-D length excision / whiten+balance), paired same-head on all 3 datasets | $0 pregate, 15/15 test + 15/15 dev floor-parity at 4 dp, **0 of 5 promotable** (best anywhere +0.0067 acc / +0.0052 mF1, T4 × MHC-ZH, 4.5× under bar, inside the seed band). Mechanism is the product: class balancing is **degenerate** (predictions identical to the deployed vote on 215/215 HateMM and 149/149 ZH, independent float64 control) ⇒ the local class prior is not separable from the retrieval signal; CSLS is **inert** (hubness IQR ~1e-4); the 1-D length excision is **inert and informative** (exact removal, residual ≤ 8.6e-9, yet Δρ ≤ 0.004 in 9/9 cells and zero prediction changes ⇒ **length organisation is not carried by any single linear direction**); whitening is **negative with a cause** (cone 0.9999 → 0.5220 but the length axis rises ρ 0.52 → 0.87, because LW shrinkage ≈ 0 at d > n). Whitening/balance are the first operators to reach documented stable-core errors (1–5 per cell) and still break at least as many ⇒ the F47/F66 arithmetic again. **Eval-time vote-operator axis closed as measured** | MECHFIX_PREGATE_2026-07-27.md · `110dff8` |
| **KSWEEP** — full top-k sweep of the deployed vote (answering "has anyone tried *reducing* k?") | $0 forensic replay of banked, already-test-consumed neighbour lists (~40 s CPU, 19/19 parity cells at 4 dp; the EN ARM-V k = 20 vote reproduces the banked floor bit-exactly). **k = 20 is at or above the plateau on all six arms**, plateau starting at k ≈ 10–15. Small k is **1-NN, not a sharper vote** — with weights [k…1] and descending cosines the sign at k ≤ 3 is the top-1 label, proved and then verified element-wise in **19 of 19 cells** — costing −0.0157 to −0.0388. The premise is structurally false on HateMM: **ranks 11–20 are already inert** (0 of 215 predictions change at k = 10 in five of six cells, none at k = 15 in all six). Deployment-legal dev selection is worthless-to-harmful (HateMM final −0.0140, ZH final −0.0157, pooled ZH −0.0179 / −0.0233); **per-seed oracle k tops out at +0.0145**. Axis closed in both directions | KSWEEP_RECORD.md · `d5d78ad` |

**Table 16. Round-8 gated channels and the MNTP arms (F90–F93).**

| Direction | Epitaph (one line) | Cost | Verdict · record |
|---|---|---|---|
| **CLAP general-audio** — F88's top-ranked gated ceiling (HateMM speech-poor visual hate, +0.0326), the only channel defined by the *absence* of the signal every existing channel carries | **G0-conditional gate KILL**, spec frozen before any CLAP weight was downloaded. Binding best-of-{k8,k16} Δacc **−0.0009** (deployed 7168-d) / **−0.0038** (strict 8960-d); global maximum over all four cells **+0.0009**, ~44× under the +0.040 bar and below the already-killed Whisper channel's own maximum (+0.0041); every decision CI straddles zero; context arms degrade monotonically with k. Machinery valid in all four cells (label-oracle accZA = 1.0000) ⇒ genuine nulls. On the speech-poor stratum the audio signal is **real but redundant**: CLAP 0.8411 and Whisper 0.8482 both beat the word-count baseline 0.6610, but Z alone already scores 0.8937 and adding CLAP moves it +0.0113 with a CI spanning zero, while ρ(CLAP score, word count) = **+0.4430** (p = 3.2e-42) — the length prior that *produces* those errors, not a cure. **The audio axis is now closed at all three representational levels**: classical prosody (F41, −0.0038 strict), learned speech-ASR (F64, +0.0014 / +0.0014), learned general-audio semantics (F90). Honest crack, pre-declared as context-only and rendering no verdict: on the ≤ 1-word stratum (n = 87, 8 positives) CLAP beats Whisper by +0.1266 with CI [−0.1169, +0.4300] | 0.78 GB download + ~1.7 h **CPU** (job 13647, no gres) + 681 s gate = **0 GPU-h** | CLAP_GATE_RECORD.md · `eee862c` (spec `6c8929d`) |
| **Molmo2-8B encoder swap** — allenai/Molmo2-8B (Qwen3-8B LLM + SigLIP2-so400m-patch14-384), a 2025-generation video-native encoder, on the one dataset where encoder identity ever converted | **KILL, in the informative direction.** Below the strongest same-path floor on **both protocols and both metrics** — val-selected **−0.0217 acc / −0.0249 mF1** (per-seed signs − − −), final-epoch **−0.0124 / −0.0151** — against a pre-declared bar of ≥ +0.0200 on both metrics, 3/3 sign-consistent, both protocols; against the like-for-like frozen-Qwen control it is a **tie** (\|Δ\| ≤ 0.0068 ≈ 1–2 test items). Yet the vision tower genuinely improved: raw image kNN **0.7814** vs 0.7256 (floor) and 0.7163 (frozen Qwen), the best image stream ever measured on HateMM ⇒ **the ninth certified law-I instance, and the cleanest** (earlier instances showed a stream moving with zero conversion; here conversion is *negative*). Geometry explains the park: cone collapse worsened (top-1 cosine 0.9881–0.9999 vs 0.9439–0.9686), the length nuisance axis is untouched (ρ +0.9052 vs +0.9432 / +0.9530), raw Hadamard degenerates (0.5628, PR 3.069). Same-path floor validated first — arm B reproduces the banked CPU proxy to 4 dp in all four cells (0.8775 / 0.8715 val-sel, 0.8760 / 0.8699 final) | extraction job 13648 **~18 min GPU ≈ 0.3 GPU-h**; probe job 13653 CPU-only | MOLMO2_PROBE_RECORD.md · `3298e8e` (recon `c1d450c` / `997b227`) |
| **MNTP S1 + S1b** — the bidirectional **readout** route (all-position mean pool; then text-positions-only pool selected by token id) | **Route closed at zero training.** S1: HateMM text 0.7477 = recovery **−0.1999**, *below the F72 crater itself* (0.7570), against MHC-ZH 0.7051 = +0.3529 — opposite signs, sign-consistency clause fires. The finding is **stream collapse**: within-arm cos(text, img) 0.9273–0.9404 (HateMM) / 0.9316–0.9320 (ZH) against causal 0.3027–0.3523, because ~82.5 % of the pooled span is vision tokens ⇒ ZH's "recovery" is **stream substitution**. S1b then self-refuted on its pre-declared collapse belt (bar < 0.60; measured 0.7566 / 0.7624 and 0.7565 / 0.7538) **even though the accuracy gate alone would have said continue** — the belt overrode the gate, which is why it was declared before the arm was built — with the smoking gun that S1b's HateMM text row is *numerically identical* to its image row (0.7664 / 0.7540). Mechanism: under bidirectional attention every text token attends to all ~720 vision tokens, so excluding vision **positions** does not exclude vision **information**; collapse is monotone in pooled span across the three spans now measured | **1.691 GPU-h** (budget ~2.0) | MNTP_S1_RECORD.md · `4a87836` / `f15dabc` / `12e2f18` |
| **MNTP S2a** — transplant of the **published McGill MNTP adapter** onto our merged Qwen2.5-VL trunk (zero training, zero corpus ruling, zero test touch) | **STOP — and the campaign's first real bidirectional signal.** HateMM text **0.7850** vs the F72 bidirectional 0.7570 = **+0.0280 = +0.6006 crater recovery**, the first arm ever to clear the frozen 50 % bar; MHC-ZH +0.0641 = +0.2941 partial; both signs positive ⇒ weight adaptation moves what three readouts could not. The stop is **overdetermined on four independent grounds**, so no single gate is load-bearing: (1) the pre-declared collapse belt fires on both datasets (0.6494 / 0.6550 and 0.6386 / 0.6433 against a 0.60 bar, self-refuting *regardless of accuracy*); (2) **fusion inverts from additive to destructive** — causal concat beats the best single stream by +0.0467 / +0.0128, S2a concat is worse by −0.0467 / −0.0256, and the deployed system *is* a fusion head; (3) every S2a number sits below its causal floor (text −0.0187 / −0.1538, image −0.0280 / −0.0128, concat −0.1121 / −0.1538); (4) the escalation gate cannot be met. Mechanism: a low-rank delta fitted at the Qwen2.5-7B-Instruct weight point acts as a large blunt perturbation on a drifted VL trunk (cos(S2a, plain-bidir) 0.3639 / 0.3076). **Refuted: the zero-training transplant shortcut. Not refuted: the MNTP hypothesis**, whose only live form is training at our own weight point behind the corpus ruling. Process: the pre-submission code gate caught a defect that would have loaded **zero** adapter weights (PEFT keys one `.model` deeper on the outer VL wrapper, loaded non-strictly), silently duplicating F72 while looking like it ran, plus an undeclared vision-tower binding (292 modules matched, 96 in the frozen tower; the fix binds exactly 196 = 28 × 7 with zero vision) | **1.006 GPU-h** (S1 + S1b + S2a = 2.697) | MNTP_S1_RECORD.md §6d/§6e · `0663ab7` (amendment committed before the fork was built) / `b328dc9` |

**Table 17. The relational / memory-bank pregate chain (F95–F98).** LITSWEEP-6's own accuracy menu is
**0 for 3** at $0 (F96 restrans, F97 VGA/VNQ, F98 aggnet); the F95 pair-verification pregate precedes that lane
and supplies the frozen machinery all three reuse, so it is listed here as the chain's first cell rather than as
a fourth menu item. All four cells are CPU-only at
≤ 8 threads with **zero GPU / SLURM / Modal**, no training of any deployed arm, and **no test-split contact**
(train split only; `dev_seen` / `test_seen` never opened by any script), each sha256-asserting the frozen
modules of the cell before it. Sweep records: LITSWEEP6_MEMBANK.md `62efd82`, LITSWEEP6_PARADIGM.md `49e15ec`,
LITSWEEP6_RELGEN.md `f62e777`.

| Direction | Epitaph (one line) | Verdict · record |
|---|---|---|
| **MECHNOV pair-verify** — replace the deployed kNN **vote** with a trained **pair verifier** (retrieval nominates, a relation scorer adjudicates; n item labels → ~n² pair labels) | **KILL, split verdict — both halves load-bearing.** Control 1 passes by **4.3–8.8×** (18/18 cells, 5/5 fold signs): fused pair-AUC 0.5843 → **0.7753** (HateMM), 0.5123 → **0.7748** (ZH), 0.5057 → **0.7009** (EN) ⇒ relational supervision genuinely buys a better relation scorer and it is **not** the cosine re-derived; it also prices the deployed metric (on ZH and EN the retrieval cosine's own pair-AUC is within 0.02 of chance). Control 2 is cleared by **0 of 36** end-to-end cells (primary −0.0040 / −0.0466 / −0.0146). Two measured reasons: (i) **the discarded aggregation was doing the work** — the same rule *shape* scored by cosine already costs −0.0417 / −0.0293 / −0.0437, and the verifier recovers less than the shape destroyed; (ii) **better relations ≠ better decisions** — verification reaches 36.7–54.6 % of exactly the errors ERRPAT called unreachable (F89's operators reached 0–5) at exchange rates 0.9474 / 0.5345 / 0.8596, ceiling 1.1667, **no cell at 1.2**. Sharpest law-I measurement recorded: within-query pair-AUC 0.6067 / 0.5363 / 0.5228 → 0.7639 / 0.7665 / 0.7013, and the query × bank interaction share of score variance rises from **26.6–37.7 %** (the deployed similarity is mostly *not* a relation) to **77–93 %**, with zero conversion | MECHNOV_PAIRVERIFY_PREGATE.md · `0261b82` |
| **RESTRANS (membank C1)** — de-bias the **label field** the vote transports, not the geometry (retrieval, k, weights, threshold, key space all identical; transported summand becomes a residual against a length-conditional prior) | **KILL, mechanistically: the degeneracy control fires.** 21 of 21 cells negative (primary fused −0.0188 / −0.0863 / −0.1002, exchange rates 0.4167 / 0.3243 / 0.4860; best anywhere −0.0013). Replacing the per-item prior with its own bank mean — a pure global threshold shift, already dead on all three datasets — agrees with C1 on **95.03 % / 97.75 % / 99.45 %** of items ⇒ **C1 is a threshold move in an item-level costume**, and the closed form shows why (under a cone-collapsed cosine profile the residual vote reduces exactly to the deployed vote minus a constant; the item-level dispersion is 20–200× too small). It **reached the right population** (34/34, 98/98, 158/159 changed decisions in the rank ≤ 5 pathology band) and broke it 2.1–3.1× faster than it fixed it. Durable by-product: the length prior is a **HateMM-specific fact** — ρ(volume, gold) = +0.2842 (p = 2.74e-15) / **−0.1152** (p = 0.0055, sign-inverted) / −0.0050 (p = 0.906) — so no future candidate may target on it | RESTRANS_PREGATE_RECORD.md · `bf6d03b` |
| **VGA / VNQ (relgen C1 / C2)** — gate the F95 adjudication per item on the disagreement set (structurally zeroing F95's shape cost), and read the verifier profile as a selective-prediction risk ordering | **KILL both, on 5 of 6 frozen bars, and the decisive one is a refutation rather than a miss.** Net gain fails 0/3 (best primary +0.0108, a third of bar) and the permutation null fails on all three (p = 0.8706 / 0.5174 / 0.9751). The mandatory new-signal control **fires**: a gate using **F47 features only** (vote margin, purity, sub-votes; no verifier) *beats* the verifier gate on 3/3 datasets with significance — **+0.0269** (HateMM, p = 0.0050, signs +++++), **+0.0104** (ZH, p = 0.0050), **+0.0182** (EN, p = 0.0100) — and the inversion is sharpest where the verifier gate is statistically dead, so the "genuinely new information source" claim that was this candidate's whole licence to revisit F47 is **refuted by measurement**. Selective prediction loses to the cheapest baseline: AUGRC (lower better) 0.0458 / 0.0417 / 0.0810 for the verifier profile against the **free vote margin** at 0.0465 / **0.0384** / **0.0696** and a fitted kNN-uncertainty baseline at 0.0429 / 0.0393 / 0.0758. **The relational asset is settled as analysis-grade only** — three conversion attempts (replace the vote, gate the replacement, read it as risk) all negative | VGA_PREGATE_RECORD.md · `db2eae8` |
| **AGGNET (membank C3)** — learn the **weighting** per query (retrieval, key space, k = 20, candidate set, threshold and label field identical; fixed rank weights [20…1] replaced by g_θ(neighbourhood profile), deployed-anchored init, shrinkage selected by nested CV) | **KILL, and it closes the conditional-aggregation family.** C3 enters with by far the largest ceiling the family has had — 96–100 % of every deployed error is reachable (111/116, 88/88, 120/121) and the family oracle is **+0.1492 / +0.1520 / +0.2186**, 2–4× the adjudication-gate oracle and 10–15× F94's oracle-k — and delivers **+0.0134 / −0.0069 / +0.0000**, with a 45-cell maximum of +0.0134 and **0 cells at +0.030**. The decisive bar is missed by more than 2× *and* it lands 0.0135 below the cheap F47 gate it had to exceed. Both mandatory degeneracy controls fire on the only dataset where it is positive: 0.9570 agreement with a bare global threshold shift, 0.9610 with a single fixed k = 15 — while **a bare threshold alone scores +0.0188**, more than the 1316-parameter network, with no profile at all, and an unconstrained logistic on the identical profile scores exactly +0.0134. **Delivery is uncorrelated with ceiling inside this family** (F94 +0.0145 → −0.0140…+0.0041; F95/VGA +0.0726/+0.0535/+0.0893 → +0.0269/+0.0104/+0.0182; C3 +0.1492/+0.1520/+0.2186 → +0.0134/−0.0069/+0.0000): what binds is neither reach nor capacity but that the local configuration carries no learnable signal about which neighbours to trust at n = 549–744. This is the **second** operator to collapse into a threshold-shift-in-costume after C1 — two different halves of the vote, edited two different ways, degenerate to the same place | AGGNET_PREGATE_RECORD.md · `fa1e3b3` |

**Reading the floors (a reconciliation, not a discrepancy).** Several round-8 cells quote a HateMM same-path
floor of **0.8775 / 0.8715** (val-selected) and **0.8760 / 0.8699** (final-epoch). Those are **CPU-proxy head**
readings — the six floor checkpoints were disk-deleted under F78, so the proxy re-runs the byte-identical
training command on the same banked feature caches — while the T1/§7 anchors are the GPU numbers **0.8775 /
0.8711** and **0.8791 / 0.8726**. The difference is exactly the proxy-versus-floor offset F88 measured and
recorded: **+0.0000 / +0.0004** val-selected (exact at 4 dp) and **−0.0031 / −0.0027** final-epoch (0.67 test
items per seed, the residual being CUDA-versus-CPU dropout RNG). Every round-8 delta is a **paired same-path**
quantity in which that offset cancels, and no round-8 cell touches the anchors. The binding discipline running
from F87 through F88 to F91 is that a CPU-trained arm may only be paired against a CPU-trained floor, which is
why the Molmo2 probe re-ran its control arms in the same job rather than quoting banked numbers, reproducing
the proxy to 4 dp in all four cells.

**Accounting and count discipline (round 8).** Total GPU ≈ **3.0 GPU-h** — MNTP S1 + S1b 1.691, S2a 1.006,
Molmo2 extraction ≈ 0.3 — with the three forensic reports, the five $0 pregates and the whole CLAP gate at
zero. No cell revises T1–T4, none enters the 13-route count, and the best numbers are unchanged. Two ledger
movements carry forward and are stated once so they stay consistent everywhere: the **law-I instance count is
now nine** (F63 seventh, F65 eighth, F91 ninth; F87's candidate remains explicitly uncertified, and F95 is
recorded as the law's sharpest *measurement* rather than a tenth instance — analysis §3.6, §3.13), and the
**pillar-4 EN memory-editing positive is single-seed** (+0.0124 on seed 0, zero vote flips on seeds 1–3,
four-seed mean +0.0031; the 14-id rule list is +0.0093 with 3 of 4 seeds positive and 0 items broken, still
3× under bar and now test-consumed), so every citation of it reads *human-in-the-loop capability
demonstration, single-seed; not an accuracy claim* (§5, analysis §4).

*Consistency note (round 8): every number above is transcribed from the named refine-logs record at write
time and spot-checked against primary artefacts (numeric-provenance discipline) — the pair-verify and aggnet
pooled deltas were recomputed from `scripts/analysis/mechnov_pairverify_hatemm_OUT.json` and
`aggnet_main_hatemm_OUT.json`, the F47-gate numbers re-read from `vga_pregate_OUT.json`, the CLAP binding
points from `CLAP_G0COND_GATE_OUT.json`, the MHC-ZH final-epoch anchor (0.8456 / 0.8173) re-derived from the
three job-13150 trainlogs, and the Molmo2 arm means re-derived from the job-13653 per-seed reads. Findings
F88–F98 are banked as post-terminus audit cells and kept off the negative-result ordinal (master-table
tension list #9).*
