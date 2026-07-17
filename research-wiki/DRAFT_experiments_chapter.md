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

**Human-in-the-loop memory edit.** Semantic addressing plus surgical deletion is pure-CPU and
seconds-fast. Deleting **two** human-flagged noisy memory entries lifts MHClip-EN test accuracy
**0.8075 → 0.8199** (macro-F1 0.7626 → 0.7748) at seed 0 with **zero retraining**, exceeding all five
random-seed floors and all five same-size random-deletion controls (max 0.8137) — the project's best
single EN point, reported as a controllability demonstration, not a main-table row
[DOC:DEMO_memory_editing.md]. The same operation on a trained head does not exist.

**Guard-rail / semantic veto.** A two-vote AND rule for *automatic* repair does **not** reproduce the
human 2-entry gain (C − A = +0.0000, 0/4 EN seeds): it structurally cannot reach memories that are
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
positive, LoRA-HateMM); they do **not** revise the
campaign's 13-route accounting (T4), and are reported here as a clearly-labelled extension. The four
structural laws that crystallised from them are analysed in the analysis chapter §3.6–3.9; this section
is the results ledger.

**Count discipline.** T4's thirteen routes are unchanged: at route-family granularity, ten main-table
accuracy rows plus three localization rows (P6 / P10 / P11); the analysis chapter's finer count instead
splits P9/P9b to report eleven main-table routes — the same thirteen results under two granularities,
as the master-table tension list #7 documents. **This 13-route campaign count is the load-bearing
accounting and is untouched by every sprint below.** On the separate novelty-first negative ledger:
round 2 adds seven sprint negatives (#15–21) plus the B4 EN-LoRA-encoder cell (#22) — closed pre-GPU
then, now formally measured as a FAIL under round 4's line-A run (F53) — and one *marginal positive*
held pending a user novelty ruling (B3); round 3 adds six directions, every one closed at a binding
verdict or a calibrated-zero conditional-info gate; and round 4 adds two further pre-registered
negatives — the per-item cross-channel router (F47) and the fusion/composition FA gate (F50) — plus a
pre-GPU arithmetic kill (MJ, F49), a wave-5 adaptation-family structural closure (F51), and the line-A
LoRA-HateMM measurement (F53): an encoder-level LoRA **HateMM PASS under both protocols** (a *second*
performance positive alongside B3, held pending the same user D7 novelty ruling) that also formally
measures the bundled B4 EN cell (#22) as a FAIL. With round 4
no surviving candidate remains in the frozen constraint box [DOC:TERMINUS_round2_mllm_plus3.md,
DOC:TERMINUS_round3_mllm_plus3.md, DOC:ROUTER_GATE_RECORD.md, DOC:FA_GATE_RECORD.md]. (A ledger-ordinal
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
caveat that the two passes convert via **different modalities** (ZH text-borne and LoRA-specific;
HateMM image-borne and inherited from the frozen swap, analysis §3.9). Whether either LoRA pass counts
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
HateMM's *frozen*-Qwen pass is a different lever — but the two passes convert via **different
modalities**, so the conjunction is one lever with two mechanisms, not one mechanism (analysis §3.9): on
ZH the gain is text-borne and LoRA-specific (frozen-Qwen −0.0112), while on HateMM the KS-2 result shows
LoRA ≈ frozen-Qwen, so the gain over CLIP is substantially the frozen-Qwen image-modality conversion
that LoRA inherits and preserves. The bundled **B4-EN closure arm** (same job 13235) formally measures
the EN LoRA-encoder cell and **FAILs both protocols** — val-selected mean Δacc −0.0021, final-epoch
+0.0000, each ≪ the bar, its seed-0 anchor reproducing the pre-GPU forensic value exactly — upgrading the
round-2 #22 negative from a pre-GPU forensic close to a measured one and leaving EN's only formal encoder
pass the frozen swap. Whether the encoder-level LoRA performance conjunct counts toward the goal's
*novel* clause is the pending user D7 ruling; the LoRA-HateMM cell is an encoder-class lever regardless
of outcome and is **not** folded into any main table
[DOC:LORA_HATEMM_VERDICT_REVIEW.md; DOC:PAPER_MASTER_TABLES.md PUR-3, PUR-banner].

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
`refine-logs/LORA_HATEMM_VERDICT_REVIEW.md` (`6b8f634`) and re-checked against the job 13235 trainlogs.*
