# Introduction, Related Work, and Limitations

*Draft sections — English academic prose, ~3 pages of body density. All quantities are transcribed
from committed project documents; internal provenance is given as [DOC:file] and external methods /
datasets as `\cite{}` placeholders. No number or claim below is new — each is traceable to a
committed result, and terminology follows the method / experiments / analysis chapters (the
retrieval objective is the **triplet-margin** contrastive instance; the MLLM is framed by its
**three earned roles and one explicit non-role**; the localization contrast is written against the
**retrieval memory**, not a MIL head).*

> **Abstract-level positioning is pending user decisions and is NOT fixed here.** The near-ceiling
> MHClip-EN wording follows the DECISION_MEMO **D3** recommendation (report dual-protocol EN numbers +
> same-arena win framing + near-ceiling attribution, *not* an absolute-SOTA claim), and the
> efficiency co-headline weight / negative-results emphasis / body–appendix split follow the pending
> venue decision **D6**. Both are marked inline as `[TODO-D3]` and `[TODO-D6]`; the abstract itself is
> deferred until D3 and D6 are ruled [DOC:DECISION_MEMO_pending.md D3/D6].

---

## 1. Introduction

Hateful and harmful video is now one of the fastest-growing surfaces of online abuse, and it is a
uniquely hard one: a single clip fuses speech, on-screen text, imagery, and prosody, and the same
target can be attacked explicitly with a slur or implicitly through coded language, memes, and
irony \cite{das2023hatemm,multihateclip,rehman2025implihatevid}. Two properties of the problem shape
what a *deployable* detector must offer. First, **hate evolves**: new slurs, reclaimed terms, and
platform-specific in-jokes appear continuously, so a detector whose decision is frozen into trained
weights is stale the day it ships and expensive to refresh. Second, **moderation is accountable**:
platform trust-and-safety workflows increasingly require that a flag be *inspected*, *corrected*, and
*kept current* — properties that a monolithic classifier, which absorbs its decision into
un-addressable parameters, does not provide.

Two method families dominate the field, and neither meets both requirements at once. The first is a
family of **reasoning VLMs** — MARS, HVGuard, IARE, and RAMF \cite{mars,hvguard,iare,ramf} — that
prompt a large multimodal model to reason over each clip, always-on and per-video. This family is
strong, especially on implicit hate, but the reasoning is heavy, runs at inference for every clip,
and produces a verdict rather than a persistent, inspectable memory that a moderator can edit. The
second family is **retrieval-augmented**: here the field state is a single method, **MoRE**
(WWW 2025) \cite{more}, the only published retrieval-augmented hateful-video detector. MoRE *does*
carry a memory bank, but its retriever is a **frozen** weighted-cosine heuristic, its supervision is
BCE over attention experts, and its decision is baked into a trained mixture-of-experts router — so
the memory is neither learned, nor updatable at test time, nor readable. Retrieval, in other words,
**already exists in hateful video**; what does not exist is the intersection our method targets — a
*learned* retrieval-guided **contrastive** embedding whose inference is a **kNN vote** over a memory
that can be swapped, grown, audited, and edited without a gradient step [DOC:gap_map.md G1/G2].

**Our approach.** We detect hateful video with a **retrieval-memory** detector rather than a trained
classification head, porting the retrieval-guided-contrastive core of RGCL / RA-HMD (developed for
hateful *memes*) \cite{rgcl,rahmd} to video and building four video-specific memory capabilities on
top of it. Each clip is encoded once, offline, by a **frozen or LoRA-adapted** MLLM/CLIP encoder
(8-frame visual pooling + a title/transcript text tower) into a two-stream feature cache; a
lightweight **alignment-fusion head** (a few million trainable parameters) is trained with a
**retrieval-guided contrastive (triplet-margin) objective** over FAISS-mined pseudo-gold positives
and hard negatives, where the hard negative is exactly the hateful-vs-offensive / benign-confounder
near-duplicate these benchmarks document as hardest; and, at inference, the label is decided by a
**kNN vote over a labelled memory bank** in the learned fused space — a non-parametric read-out, not
the classifier logit [DOC:DRAFT_method_chapter.md §1-2]. This single design choice — reading the
decision off an inspectable memory — is what makes the memory *updatable*, *swappable*, *auditable*,
and *editable*, and it is also what keeps the detector lightweight relative to always-on reasoning
VLMs: a run-once encoder plus a few-million-parameter head plus a CPU-side kNN read-out
`[TODO-D6: promote efficiency to a co-headline only if the venue rewards it]`.

**Contributions.** Scoped strictly to hateful **video**, we make four capability contributions, frame
the MLLM's roles, and — as a first-class methodological contribution — report a large body of
pre-registered negative results honestly rather than hiding them.

- **(C1) Retrieval-guided-contrastive + kNN core, ported to hateful video and tested head-to-head
  against MoRE.** On the identical binary protocol (same split verified by line-for-line diff, same
  clean test subset, MoRE's own released code re-run as-released and bug-fixed with a 5-seed sweep),
  our best configuration wins on **all three shared benchmarks**: **+5.6 / +8.7 / +6.7 accuracy** and
  **+6.2 / +22.9 / +9.7 macro-F1** (HateMM / MHClip-EN / MHClip-ZH) over the stronger MoRE variant,
  with nominal training-label count still favouring MoRE [DOC:BASELINE_MoRE_rerun.md,
  DOC:PAPER_MASTER_TABLES.md T1.2]. The precise mechanism delta vs MoRE (learned vs frozen retriever;
  contrastive vs BCE-over-experts; kNN vote vs MoE router) is developed in the method chapter — we
  make a *mechanism* claim, never a "we bring retrieval to hateful video" claim.
- **(C2) A zero-retrain updatable memory with an O(1) temporal-recalibration protocol.** A head
  trained on source *A* classifies target *T* by swapping in *T*'s labelled memory — no retraining —
  beating the target majority on **5 of 6** informative cross cells, a capability a trained MoE router
  is *structurally incapable* of. Under temporal shift (MHClip-EN, −0.084 macro-F1) we show the loss
  is calibration drift, not lost separability, and recover it in full by re-calibrating the operating
  point from **k = 20** new-period labels (0.6273 → 0.7336, zero retraining), with a ZH no-drift
  negative control [DOC:experiments/exp-cross-dataset-transfer.md, DOC:EVAL_temporal_memory_W4.md].
- **(C3) Consensus denoising of label-inherited segment supervision.** Sub-clips inherit the video
  label but are mostly benign, which poisons a naive segment-contrastive term; a retrieval-consensus
  E-step de-poisons it. This is **validated on Chinese as a repair** (removes a −0.066 macro-F1 hole),
  with an honest, complete cross-lingual boundary: the repair does not transfer to English, and a full
  attribution chain pins the residual failure on the segment-supervision channel itself having no gain
  for speech-carried hate — not on a bad key choice [DOC:experiments/exp-consensus-zh-seeds.md,
  DOC:EXP_mm_segment_keys.md].
- **(C4) An auditable and human-editable archive memory.** MLLM-produced structured archive records
  make the memory auditable (a stratified audit finds it faithful on 77% of records; the label-blind
  audit re-discovers human-flagged noisy ids with correct reasons) and editable: deleting **two**
  human-flagged noisy entries lifts MHClip-EN accuracy 0.8075 → 0.8199 with zero retraining **at seed 0
  — a human-in-the-loop capability demonstration, single-seed, not an accuracy claim** (the round-8
  multi-seed replay finds zero vote flips on seeds 1–3 and a four-seed mean of +0.0031; the 14-id rule
  list reads +0.0093 acc / +0.0089 mF1, 3 of 4 seeds, 0 items broken, still sub-bar and dev-unresolvable)
  [DOC:ERRPAT_MHC-EN_2026-07-26.md §6.5]. A bounded automatic *guard-rail* survives as a semantic veto
  against embedding-only over-deletion; the archive's payoff is **integrity and controllability, not raw
  accuracy** — a framing the correction strengthens rather than weakens, since the property being claimed
  is the existence of a retraining-free, semantically addressed edit surface
  [DOC:AUDIT_archive_faithfulness.md, DOC:DEMO_memory_editing.md, DOC:EXP_auto_memory_repair.md].
- **(C5) The MLLM's three earned roles and one explicit non-role.** Fixed as a pre-registered
  mandate, the MLLM earns exactly three *removable* roles — **encoder** (frozen Qwen2.5-VL beats CLIP
  by +4.2 macro-F1 on HateMM and crosses 0.85), **span-free localization scorer** (below), and
  **guard-rail / audit** — and, at 7B–72B open-weights scale, **no main-table-accuracy role** in this
  retrieval-memory pipeline [DOC:CAMPAIGN_mllm_method_role.md, DOC:DRAFT_analysis_chapter.md §4].
- **(C6) Negative results as a methodological contribution.** The non-role in (C5) is not an absence
  of evidence but a **thirteen-route pre-registered campaign** whose eleven main-table-accuracy routes
  are all honest kills or within-noise — each guard-backed by a reproduction, bit-for-bit, or probe
  check — yielding two transferable mechanistic conclusions (semantic competence is orthogonal to the
  decision variable; a passing no-head probe is necessary but not sufficient). We present these as a
  contribution, not a hidden appendix [DOC:PAPER_MASTER_TABLES.md T4, DOC:DRAFT_analysis_chapter.md].

**Result highlights.** HateMM and ImpliHateVid clear the field's acc ≥ 0.85 bar on our split
(frozen-Qwen **0.870 / 0.861** macro-F1 on HateMM; ImpliHateVid ≈ 0.90–0.91), and we win the
same-arena MoRE comparison on all three shared benchmarks (**+5.6 / +8.7 / +6.7** accuracy). For
localization, a span-free zero-shot per-window MLLM scorer ranks HateClipSeg hate windows at
within-video AUC **0.5435** (7B), amplified to **0.5755** by a 72B coarse×fine aggregation — paired-
significant over the retrieval memory read-out (0.5140; Δ +0.0615, p = 4.9e-5) though **modest**,
below the 0.60 substantial bar [DOC:PAPER_MASTER_TABLES.md T1.1/T2.1]. On the two MHClip splits our
detector sits **near a documented ceiling**: MHClip-EN is ≈ 0.78–0.80 under both selection protocols
— decisively above the strongest same-arena MoRE re-run (0.69–0.72) and in the magnitude band of the
highest published number on that split (CRAVE, 79.81 macro-F1, reported on the *full* split and not
directly comparable to our clean subset), so we report the EN result as a **same-arena win with a
near-ceiling attribution rather than an absolute-SOTA claim** `[TODO-D3: finalize EN abstract wording
once ruled]` [DOC:MORNING_REPORT.md §1, DOC:HEADTOHEAD_FEASIBILITY.md §3]. MHClip-ZH crosses 0.85
only under the selection-free final-epoch protocol (0.854 vs 0.827 validation-selected), a headline
that we deliberately leave to a separate protocol decision rather than adopt *because* it crosses
[DOC:DECISION_MEMO_pending.md D2].

---

## 2. Related Work

We position our contribution by its **clean difference from the nearest competitors**, not by any
first-in-the-world claim; the closest-SOTA set is fixed as MoRE, RA-HMD / LMM-RGCL, RGCL, MM-HSD,
ImpliHateVid / TCL, MARS, HVGuard, IARE, and MultiHateLoc [DOC:novelty-scope-and-plan.md].

**(a) Hateful video detection: heavy reasoning VLMs vs light heads.** The field's strongest recent
detectors are **reasoning VLMs** that prompt a large multimodal model to reason over each clip,
always-on: MARS (dual-hypothesis reasoning), HVGuard (chain-of-thought for puns and homophones), IARE
(rationale supervision), and RAMF (multi-perspective reasoning) \cite{mars,hvguard,iare,ramf}, with
MM-HSD \cite{mmhsd} the video-level accuracy leader on HateMM. These are accurate but computationally
heavy and produce a per-clip verdict rather than a reusable, editable memory. MM-HSD's published 0.878
macro-F1 doubles as an **external calibration of our constraint box**: its lead rests on an on-screen-text
OCR channel we veto, and *without* OCR it falls to 0.845 — inside the band of our best HateMM configuration
(0.8775–0.8791, experiments §7), so the field's SOTA over our detector on HateMM is the vetoed channel, not
a stronger core [DOC:LITSWEEP2_FRESH_2026.md]. This calibration holds across the full 2023–2026 HateMM
frontier: every published method on *legal* channels — CMFusion's gated MFCC-audio fusion \cite{cmfusion},
Koushik's CLAP general-audio late-concat \cite{koushik}, and RAMF's Qwen2.5-VL-32B counterfactual reasoning
\cite{ramf} — sits at or below our macro-F1, and only OCR-using MM-HSD edges us (experiments §8)
[DOC:LITSWEEP5_HATEMM_EN.md]. At the other end sit
light multimodal fusion baselines (HateMM's own fusion, CMFusion, MultiHateGNN)
\cite{das2023hatemm}. *Our delta:* we occupy a distinct point in this space — a **run-once frozen /
LoRA encoder + a few-million-parameter head + a kNN memory read-out** — matching or beating heavier
systems on the shared benchmarks while keeping the decision inspectable, positioning efficiency as a
co-headline `[TODO-D6]`.

**(b) Retrieval-augmented hateful video.** **MoRE** \cite{more} is the *only* published
retrieval-augmented hateful-video method, and it is our correct same-arena baseline. It builds a
video-to-video memory bank, but its retriever is a frozen weighted-cosine heuristic, its supervision
is BCE over attention experts, and its decision is a trained MoE router; the memory is therefore not
learned, not test-time-updatable, and not readable. Adjacent to it, **CRAVE** (ICCV 2025) \cite{crave}
is retrieval-augmented **cross-domain training** — retrieval used to *train* a more transferable
model — which is a different capability boundary from ours: our memory swap is a **zero-retrain**
test-time operation, whereas CRAVE retrains. *Our delta* (the precise learned-retriever /
contrastive-objective / kNN-vote mechanism table lives in the method chapter): we do **not** "bring
retrieval to hateful video" — retrieval is already MoRE's; we contribute the missing intersection of
**learned retrieval-guided contrastive embedding + kNN-vote inference over an updatable memory**
[DOC:DRAFT_method_chapter.md §2, DOC:gap_map.md G1].

**(c) Contrastive learning in hateful video.** Contrastive objectives are established in the field,
but the positives and negatives are chosen by **class label or timestamp**, never mined by retrieval:
ImpliHateVid / TCL uses supervised contrastive learning by class label \cite{rehman2025implihatevid},
MultiHateLoc uses frame-level cross-modal contrastive alignment with same-video / same-timestamp
positives \cite{multihateloc}, and IARE uses DPO preference contrast over correct-vs-wrong rationale
paths \cite{iare}. *Our delta:* our contrastive signal is **retrieval-mined** — the pseudo-gold
positive is the nearest same-label exemplar and the hard negative is the nearest *opposite*-label
near-duplicate in the *learned* space — a distinct and unclaimed contrastive signal in video that
directly targets the confusable hateful-vs-offensive boundary [DOC:gap_map.md G2].

**(d) Temporal localization of hate.** Weakly-supervised temporal hate localization is a task-first
line owned by **MultiHateLoc** (MIL Top-K, no retrieval or denoising) \cite{multihateloc}, alongside
**LELA** (training-free frame localization) \cite{lela} and **TANDEM** (timestamps via RL)
\cite{tandem}; **HateClipSeg** \cite{hateclipseg} supplies the segment-level evaluation data. We
acknowledge this line and cite it as related work — we do **not** enter a same-arena localization
table with it, because these methods are codeless with under-specified protocols
[DOC:EVAL_localization_hatemm.md §4]. *Our delta, stated within the field's wording conventions:* our
contribution is **span-free**, zero-shot existence-scoring of hate windows out of the retrieval
memory, amplified by an MLLM per-window scorer — we claim **only** "span-free" and never *first*,
*annotation-free* (occupied by LELA), or *dense-supervision-free* (occupied by TANDEM)
[DOC:DRAFT_experiments_chapter.md §4, DOC:MORNING_REPORT.md §3].

**(e) Editable / auditable memory and structured records.** Structured MLLM records appear in
adjacent work but are treated as **throwaway**: TANDEM produces schema fields that are *not stored*,
and SafeLens produces structured evidence that is *discarded* after the decision \cite{tandem,safelens}.
*Our delta, with an explicit wording boundary:* our archive is a **persistent, addressable memory**
that is audited and human-edited, not a transient reasoning log — so where SafeLens uses "auditable"
for its (discarded) evidence trace, our audit object is the **retained memory bank itself**, and we
scope our wording accordingly to avoid the clash [DOC:novelty-scope-and-plan.md,
DOC:AUDIT_archive_faithfulness.md]. We further position the archive within the **model-editing** lineage
— SERAC's external edit-cache with a scope classifier, GRACE's key-value adaptors, and WISE's side-memory
for lifelong edits \cite{serac,grace,wise} — which supplies a principled reliability / generality /
**locality** vocabulary for what our human-in-the-loop deletions do (fix the target queries without moving
unrelated ones); our archive is a *discriminative* instance of that program, edited at inference with no
weight change [DOC:LITSURVEY_RETRIEVAL_MEMORY.md]. To our knowledge no hateful-video method offers a memory
that is simultaneously swappable, temporally recalibratable, auditable, and surgically editable at inference
with zero retraining. *Wording boundary, binding on this paragraph:* the editing evidence is a **single-seed
capability demonstration** (round-8 correction F88 — zero vote flips on three of four deployed seeds, four-seed
mean +0.0031), so the editing lineage is invoked for **reliability / generality / locality vocabulary**, never
to claim an accuracy improvement from editing [DOC:ERRPAT_MHC-EN_2026-07-26.md §6.5].

**(f) Usable information, annotator disagreement, and modality imbalance — positioning the negative-results
and mechanism contributions.** Three adjacent literatures name phenomena our campaign measures. First,
**V-usable information** (Xu et al. \cite{xu2020vinfo}; Ethayarajh et al. \cite{ethayarajh2022vinfo}, whose
pointwise-V-information quantifies per-instance difficulty) is the exact formalization of our Law I — an
oracle proves the convertible headroom is present, yet no operator in the model family recovers it, i.e. a
gap between information *present* and information *usable* (analysis §3.6). Second, our Chinese consensus-
denoising pillar (C3) is an instance of the **learning-with-disagreement** program crystallised in the
LeWiDi-2025 shared task \cite{lewidi2025}, where soft-label / distributional supervision beats collapsed
majority-vote gold on subjective phenomena including toxicity — a citation lineage the pillar previously
lacked. Third, our MHClip-EN image-collapse analysis (a Qwen image stream that drops to near-chance under
Hadamard fusion, §3.6 of the analysis) is the documented **modality-imbalance / text-dominance** signature
\cite{balancebench}, which lets us frame F65's "image moved, converted nothing" as the signature of a
*label-limited* rather than *imbalance-curable* collapse. Relatedly, the current SOTA video-MLLM-embedding
works (VLM2Vec-V2, VidVec) use **no special temporal operator** — they read a single pooled embedding — which
independently corroborates our F35 / F37 / F67 finding that temporal-pooling and frame density are not the
lever [DOC:LITSURVEY_NOVEL_MECHANISMS.md, DOC:LITSURVEY_MLLM_EMBEDDING.md].

**(g) Three adjacent-field methods enter this paper as credited transplant sources, not as baselines.** Our final
audit round mined *neighbouring* fields for techniques with runnable code rather than mining hateful-video work,
on the explicit ground that being first to bring an adjacent-field technique into hateful-video detection is the
defensible contribution while lifting from a direct competitor is not [DOC:REPRO_SURVEY_2025.md]. Three sources
survived triage, and each is credited where it is used. **LSMI** \cite{lsmi} (ICML 2025) supplies the
sample-level partial-information decomposition — redundancy / per-stream uniqueness / synergy — that we
re-implement *for measurement only* and run on our own banked features; the resulting information-structure
result (no image×text synergy on any dataset, image uniqueness pinned at zero, text-uniqueness dominant) is the
mechanism under our fusion nulls (analysis §3.11), and we additionally report a defect in the released estimator
that would have produced the opposite-flavoured, false conclusion. **MokA** \cite{moka} (NeurIPS 2025 Oral)
supplies modality-routed LoRA — a per-modality down-projection with a shared up-projection — which we transplant
into our encoder-SFT and measure (experiments §8); any claim from that cell is bounded, by pre-registration, to
*first application of modality-routed PEFT to hateful-video encoders*, with MokA credited, and no phrasing may
imply we invented modality-routed adaptation. **SynIB** \cite{synib} (arXiv 2606.09853) supplies a masked-branch
information-bottleneck objective that we priced, pre-registered a kill-switch for, and **parked at zero cost**
once the PID measurement returned no synergy for it to exploit. None of the three is a baseline: two are
instruments, one is an unrun transplant, and the honest overall read of the survey is that the neighbouring-field
hunt surfaced no technique with a defensible prior at our goal bar.

---

## 3. Limitations

We state the limits of the evidence plainly; several are load-bearing for how the results may be
cited.

**Small-sample noise dominates sub-two-point effects.** Our strict same-arena test subsets are ~150
videos (MHClip-EN n = 161, MHClip-ZH n = 149, HateMM clean n = 215) with a 78-video ZH dev set, so
**1 accuracy point ≈ 1.6 videos** and the n = 5 paired minimum detectable effect is ≈ 0.04–0.05
macro-F1. Validation selection on the 78-sample dev itself costs ≈ 2 accuracy points, larger than
many candidate treatment effects. Every sub-point, single-protocol "gain" is therefore reported as
*within-noise, no claim*, and we withdrew several earlier ranking claims (e.g. an archive-kNN
accuracy gain) on multi-seed and sha1 re-audit [DOC:PAPER_MASTER_TABLES.md T1.1,
DOC:experiments/exp-archive-knn-seeds.md].

**MHClip is not solved to the field bar, and the ZH headline is protocol-contingent.** MHClip-EN sits
at ≈ 0.78–0.80 and does **not** cross acc ≥ 0.85 under any configuration or protocol; we report it as
a near-ceiling result (same-arena MoRE only 0.69–0.72; CRAVE's published 79.81 macro-F1 is the highest
on that split but on the full split, not directly comparable) rather than as an absolute-SOTA claim
`[TODO-D3]` [DOC:MORNING_REPORT.md §1]. This ceiling is now mechanistically attributed rather than
merely observed — a collapsed Qwen image stream under equal-weight fusion plus a label-limited error
core, so no encoder upgrade converts there (§3.6 of the analysis)
[DOC:ENCODER_SWAP_DIAGNOSIS.md, commit `8a48938`, DOC:DRAFT_analysis_chapter.md §3.6]. MHClip-ZH crosses 0.85 only under the selection-free
final-epoch protocol (0.854 vs 0.827 validation-selected); adopting that calibration *because* it
crosses would be post-hoc rule-shopping, so the ZH headline is left to an explicit protocol decision
[DOC:DECISION_MEMO_pending.md D2].

**Localization is modest, and the open-source ceiling is empirically closed.** The promoted 72B
span-free localizer reaches wv-AUC **0.5755 < 0.60** (the substantial bar). Three walls close the
open-source ceiling: re-aggregation of existing scorer outputs tops out at 0.5932, the 72B scale
champion at 0.5913, and the next-generation Qwen3-VL-32B at 0.5866 — all below the 0.616 calibration
line that would extrapolate to a substantial test result [DOC:PAPER_MASTER_TABLES.md T2.2]. The
dominant pooled signal is video-level toxicity *density*, not within-window ranking, and — per (d)
above — codeless published baselines preclude a same-arena localization table.

**Consensus denoising is language-conditioned.** The consensus repair is validated on Chinese and
does **not** transfer to English; the complete attribution chain shows the residual EN failure is the
segment-supervision channel itself having no gain for speech-carried hate, not a fixable key choice
[DOC:EXP_mm_segment_keys.md]. The mechanism is therefore reported as a de-poisoning *repair* under a
language-conditioned boundary, not a general accuracy lever.

**Temporal-protocol survivorship bias.** The evolving-hate evaluation is feasible only on MHClip
(HateMM / ImpliHateVid are anonymized at release and stay a static cross-dataset matrix), and the
datable subset is skewed: dead download links disproportionately remove hateful ZH videos, and
undatable samples are pinned to the training period, so the temporal split carries a survivor bias we
report rather than correct [DOC:TEMPORAL_SPLIT_FEASIBILITY.md, DOC:novelty-scope-and-plan.md]. The
k = 20 recalibration win is likewise gated behind a drift monitor, since under the ZH no-drift regime
small-k recalibration is pure noise.

**Open-weights conclusions do not extrapolate to closed weights.** The MLLM non-role and the "scale
improves calibration, not selectivity" mechanism are established across 7B–72B *open-weights* models
on one cluster; they do not license a claim about closed, frontier-scale models. Whether the
localization lane can be pushed from modest (0.5755) to substantial (≥ 0.60) with a closed-weights
scorer — without violating reproducibility or data-egress constraints — is the one open question this
work leaves precisely posed `[TODO-D6: whether to keep this as an open question or open a closed-API
localization sub-thread depends on the venue and data-egress decision]`
[DOC:DRAFT_analysis_chapter.md §5, DOC:DECISION_MEMO_pending.md D1/D6].

**Dataset lifespan and link rot.** Because a fraction of MHClip videos have dead links or missing
labels, our clean subsets are smaller than the released datasets and will shrink further over time as
links decay; all "floors" are re-established on our own clean splits rather than lifted from prior
papers, and every reported delta is on identical test videos, but the absolute counts are a moving
target that future re-runs on freshly-downloaded data may not reproduce exactly
[DOC:DRAFT_experiments_chapter.md §1.1, DOC:experiments/exp-baseline-reproduction.md].

**Per-annotator labels are unavailable.** MultiHateClip releases only the aggregated majority-vote 3-class
label — two annotators, a third on disagreement, expert escalation — with no per-annotator votes or vote counts
(unlike HateXplain), neither in-repo nor in the public release [DOC:LITSWEEP5_HATEMM_EN.md §1, arXiv 2408.03468].
This forecloses at the *data* level the learning-with-disagreement / soft-label-from-annotator-distribution
lineage of §2(f): our consensus-denoising pillar operates on model-vote consensus, not on a released
human-disagreement distribution, and no annotator-level modelling is possible on these datasets. The one legal
finer granularity that *is* released — the 3-class {Normal, Offensive, Hateful} label the deployed task merges —
carries no boundary-sharpening signal under any monotone reweighting, gold-cheat oracle included
(EN +0.0250 / ZH +0.0256, both < +0.030; analysis §3.10, F82).

**Every encoder-adaptation result rests on a single SFT draw.** In each LoRA-SFT cell in this paper — the ZH and
HateMM adaptations, the curriculum variant, the bidirectional stage-1 arm, and the round-7 modality-routed
adapter — **one** encoder is trained and `--seed` varies only the head (initialisation and data shuffling), with
pairing done per head seed. Encoder-draw variance is therefore never estimated, and in any comparison whose
treatment arm carries its *own* SFT draw it is **confounded with the manipulated variable and not separable**;
disentangling would need an SFT seed sweep (≈ 9 GPU-h per cell) that no cell's budget held. The consequence is
asymmetric and worth stating plainly: it does not threaten the *negative* verdicts (an effect that fails a bar
fails it with the confound included), but it means no sub-bar adaptation-side delta may be causally attributed to
the adaptation change — the explicit ground on which the round-7 routed-LoRA cell's residual +0.0268 was refused
(analysis §3.10) [DOC:MOKA_VERDICT_REVIEW.md §D8.3].

**The merged and unmerged adapter paths are not numerically interchangeable.** Folding a LoRA adapter into the
base weights (`merge_and_unload()`, a single `W+BA` matmul) and running it unmerged (`Wx + B(Ax)`) are the same
model in method space and differ only in bf16 accumulation order — measured on the banked ZH cache at mean
per-item cosine **0.99955–0.99987** across all six (split × stream) cells, with the **text** stream drifting
≈ 3× further than the image. Yet through the 78-item ZH dev wall that numerically-null difference moved the
**val-selected** test readout by **−0.0268 acc / −0.0340 macro-F1 (0/3 seeds)** — larger than our ±0.014
head-seed band — while the no-selection **final-epoch** protocol moved by only **−0.0067**, a single test item.
Two limits follow: any comparison pairing a merged-path floor with an unmerged-path arm needs a **same-path
floor** as a default cost rather than a contingency, and val-selected deltas on these splits must be read as
protocol-conditioned quantities whose sensitivity to method-null perturbations can exceed the effects they are
asked to certify [DOC:MOKA_VERDICT_REVIEW.md §D7, §D8.4].

**The memory-editing evidence is a single-seed capability demonstration.** The pillar-4 human-in-the-loop edit
(deleting two human-flagged noisy entries, MHClip-EN 0.8075 → 0.8199, zero retraining) holds **on seed 0 only**:
an exact multi-seed replay on the four deployed seeds' banked neighbour lists finds **zero vote flips on seeds
1, 2 and 3**, a four-seed mean of **+0.0031**, and the two items seed 0 flips are low-margin false positives from
the seed-flip noise band rather than hard errors; a stronger 14-id rule list (+0.0093 acc / +0.0089 mF1, 3 of 4
seeds, 0 items broken) is still 3× under bar, inside the ±0.014 band, test-consumed, and cannot be pregated
because at dev n = 80 one item is 0.0125 [DOC:ERRPAT_MHC-EN_2026-07-26.md §6.5]. Every citation of this cell in
this paper reads *human-in-the-loop capability demonstration, single-seed; not an accuracy claim*; the property
claimed is the existence of a retraining-free, semantically addressed edit surface, which a weight-baked head
does not have at all.

**The residual is selection-locked at the item level, and we can now say so from per-item measurements.**
Round-8 forensics show the errors are ~90 % seed-invariant across all three datasets and are *confident
neighbourhood inversions* rather than boundary cases — the correct analogue is typically present at raw rank ~1.5
and simply out-voted [DOC:ERRPAT_{HateMM,MHC-EN,MHC-ZH}_2026-07-26.md]. This limits the paper in a specific way:
every uniform decision-rule repair we could construct over the same scores is now measured dead (vote operators,
neighbourhood depth, thresholds, training losses, trained pair verifiers, and per-item gates — analysis §3.13),
so we do **not** claim the residual is irreducible in principle, only that it is unreachable by any uniform rule
over these representations at these sample sizes; reaching it would require either a different representation or
a different output object, and the latter is a deliverable decision rather than an experimental one
[DOC:LITSWEEP6_PARADIGM.md, DOC:DECISION_MEMO_pending.md S1].

**The encoder-swap axis is parked on the vision side by measurement, not by exhaustion.** A 2025-generation,
video-native encoder (Molmo2-8B, SigLIP2 tower) produces the best raw image stream ever measured on HateMM
(+0.0558 over the deployed floor's) and still lands below that floor on both protocols and both metrics
(−0.0217 / −0.0249 val-selected; −0.0124 / −0.0151 final-epoch), tying the like-for-like frozen control
[DOC:MOLMO2_PROBE_RECORD.md]. We therefore state that *a better video-native encoder is not a better encoder for
this task* on this dataset — not that no encoder could ever help; the untested direction the evidence points at
is the **text** side, and the one live text-side hypothesis (masked-next-token adaptation trained at our own
weight point) is blocked on a user corpus ruling rather than on evidence [DOC:MNTP_S1_RECORD.md,
DOC:MNTP_FORENSIC_RECON.md §3].

---

*Provenance note: all numbers are transcribed from `PAPER_MASTER_TABLES.md` (T1–T4) and the committed
source documents cited inline; no discrepancy against the master tables was introduced. Citation keys
reuse the method / experiments / analysis chapters where they already exist (`more`, `rgcl`, `rahmd`,
`das2023hatemm`, `multihateclip`, `rehman2025implihatevid`, `hateclipseg`, `qwen25vl`, `clip`,
`whisper`) and extend consistently for related-work-only methods (`mars`, `hvguard`, `iare`, `ramf`,
`mmhsd`, `multihateloc`, `crave`, `safelens`, `tandem`, `lela`) and for the round-5/6 audit's
mechanism/positioning citations (`xu2020vinfo` = Xu et al. ICLR 2020; `ethayarajh2022vinfo` = Ethayarajh
et al. ICML 2022; `lewidi2025` = LeWiDi-2025 shared task; `serac`, `grace`, `wise` = model-editing lineage;
`balancebench` = modality-imbalance / text-dominance diagnosis; `llm2vec` = LLM2Vec for the
bidirectional-attention baseline; and, for the wave-5 HateMM/EN frontier positioning, `cmfusion` = CMFusion
(arXiv 2505.12051), `koushik` = Koushik HCC1 CLAP-audio (arXiv 2502.07138), `yang2025` = temporal-label-noise
/ segment-contamination (arXiv 2508.04900), plus arXiv IDs registered for the already-used keys `tandem`
(2601.11178) and `ramf` (2512.02743); and, for the round-7 transplant sources of §2(g), `lsmi` = LSMI
sample-level multimodal-interaction estimator (ICML 2025, `GeWu-Lab/LSMI_Estimator` — **venue verified, arXiv ID
not established in-record, so none is registered here**), `moka` = MokA modality-routed LoRA (NeurIPS 2025 Oral,
arXiv 2506.05191), `synib` = SynIB synergy information bottleneck (arXiv 2606.09853)), all verified against the
litsurvey, litsweep-5 and repro-survey PAPER-VALUE lists
[DOC:LITSURVEY_NOVEL_MECHANISMS.md, DOC:LITSURVEY_RETRIEVAL_MEMORY.md, DOC:LITSURVEY_MLLM_EMBEDDING.md,
DOC:LITSWEEP5_HATEMM_EN.md, DOC:LITSWEEP5_TEMPORAL.md, DOC:REPRO_SURVEY_2025.md]; the shared bibliography is a `\bibliography` placeholder
pending assembly.*
