# R15 — candidate slate and hostile scoring (2026-08-18)

Round 13 of idea discovery. Sub-direction: **hateful video temporal localization at the proposal
level** (F1@tIoU on HateClipSeg) — the same arena round 12 entered and closed on three factors.

Reviewer: **gpt-5.6-sol at xhigh reasoning**, conversation only, no tool use, given the full
constraint map, all twelve closures, and every measurement in `R14_WVD_RESULT.md` and
`R11_SEG_PILOT_RESULT.md`. Bundle: `idea-stage/codex_brainstorm_bundle_r15_2026-08-18.md`.

---

## 1. The slate

Twelve candidates. D1-D3 are the three families the round-12 reviewer named but that were never
tested (`R14_CANDIDATES.md` §4, items 2-4); D4-D12 are new, and include the cross-domain transplants
the round brief asked for (change-point detection, speaker diarization, topic segmentation).

| # | candidate | mechanism | score | load-bearing reason |
|---|---|---|---|---|
| D1 | **TRANSD** toxic-state transition discrimination | head on `(x_i, x_{i−1}, x_i − x_{i−1})` predicting *change* in toxicity, not level; intervals are the regions between predicted change points | **0** | at K=30, **73.4% of train windows are boundary-positive**, so the partition target is near-vacuous at that resolution; and boundary classification is the core of BSN `1806.02964`, BMN `1907.09702` and ActionFormer `2202.07925`. Renaming the boundary "toxic-state transition" changes the label, not the mechanism |
| D2 | **TGATT** target/attack state factorization | slowly-varying "who is targeted" state × fast "an attack is happening" state, learned interaction; the 6-way per-segment category labels supervise the target axis | **1** | not strictly killed by K2 *if* the target genuinely varies within a video, but a constant target branch reduces to a per-video intercept and is then RUBi `1906.10169` / Learned-Mixin `1909.03683`. No verified hateful-video occupant. Would have to beat a same-capacity MLP **and** an audio-only control **and** show the target state is not constant |
| D3 | **STRIDESPAN** output stride ≠ evidence span | predict every 2 s from overlapping 2 / 8 / 24 s contexts | **1** | baseline engineering, not a contribution — decoupled stride, multi-scale context and temporal pyramids are what competent detectors already do. The 87.6 grid ceiling against a current 23.8 shows resolution is not the binding constraint |
| D4 | **SEGPOOL** homogeneous-region pooling as the prediction unit | label-free temporally-constrained segmentation of each video's feature sequence, then score the region | **1** | run the oracle ceiling only; write no clustering code until it passes. TW-FINCH / CTE / ABD already occupy unsupervised temporal segmentation, so this is a first application. Oracle gold boundaries also leak target structure, so even the ceiling is optimistic |
| D5 | **CAPACITY** in-sample fit diagnostic | is 0.588 an information ceiling or an estimation gap? | **2** | cheap routing diagnostic, not a method — and **its stated inference is wrong**: one head's failure to fit does not prove no mechanism can extract the information, and a successful fit may be memorisation |
| D6 | **XSEG** cross-corpus segment-level supervision | HateMM / MHC / ImpliHateVid spans as segment supervision, never used that way; high-coverage corpora teach *what*, low-coverage teach *where* | **1** | could move the number, but ordinary cross-corpus pre-training is standard engineering, and the high-coverage sources supply almost no within-video negative contrast. The coverage-asymmetric objective is the only non-trivial part and nothing yet distinguishes it from source weighting |
| D7 | **SPKUNIT** speaker-turn units | pyannote diarization → turn-level prediction unit + speaker-identity state | **0** | speaker boundaries are not toxicity boundaries; monologues collapse to one unit; diarization is mature (EEND `1909.06247`) and using its output as a grid is not a hateful-video mechanism |
| D8 | **RELGATE** reliability-conditioned fusion | gate cross-modal interaction on ASR confidence, OCR occupancy, audio SNR | **1** | the direction is interesting but the **gate conditions on the wrong variable**: the demonstrated problem is temporal informativeness / negative transfer, not reliability. A visually clear window can still carry only video identity. Generic occupant is GMU `1702.01992`, not StreamSense |
| D9 | **CATAUX** category timeline as auxiliary supervision | 6-way multi-hot target alongside the binary one | **0** | generic multi-task learning plus a direct internal null (`B2_DENSE`, −0.33). Changing the read-out does not rehabilitate a null auxiliary head |
| D10 | **OFFSET** evidence–label misalignment | is the score curve systematically lagged relative to the labels? | **2** | worth a minutes-long diagnostic; a winning lag would establish misalignment, not a method — shifting curves is K1 decoding, asymmetric context is standard temporal modelling |
| D11 | **MEMSEG** retrieval over a labelled segment memory | score a window by kNN against 10 572 labelled train segments | **0** | directly dead: the project's own zero-training kNN measured wv-AUC **0.5259** against a 0.5252 broadcast control. "Retrieval lineage" is not evidence |
| D12 | **UNITLATTICE** annotation-free adaptive unit grid | union of Whisper sentence boundaries, OCR text-change points, PySceneDetect cuts | **0** | occupied by SafeLens (transcript/scene-change segmentation), Vid2Seq `2205.14315`, DuVOG `2208.11307`; and M3 already shows semantic/editing units do not match toxicity boundaries (32% recall / 27% precision) |

**No candidate scored 3.** The reviewer's words: *"There is no score of 3 because no slate item is
presently a paper candidate. D5 and D10 are diagnostics only."*

---

## 2. The one family the reviewer added, and why it is not any of the twelve

Asked directly whether a legal family remained, the reviewer answered **yes, exactly one**, and
derived it from a contradiction inside this project's own numbers rather than from the slate:

- audio-only within-video AUC **0.623** vs the four-channel concat **0.5878** — a gap of +0.035 in
  favour of dropping three quarters of the input;
- the round-11 circular-shift control: shuffling audio within a video costs **−3.30** macro-F1 (CI
  excluding zero), shuffling CLIP visual costs **−0.28** (CI containing zero).

Read together: the frozen substrate *does* contain moment-level information, and the concatenated
model is **diluting it with channels that carry only video identity**. The family is therefore
**temporal-informativeness-aware fusion / within-video nuisance suppression** — per-modality
within-video feature residualization, or fusion gated on whether a modality's temporal alignment
carries predictive information.

Three things the reviewer attached to it, all of which are in the pre-registration:

1. **K11 does not kill it.** K11 says a positive affine map *of the score* preserves within-video
   ranks; residualizing *features* before a non-linear head is a different object.
2. **The bar is audio-only 0.623, not fused 0.5878.** Simply selecting audio is a corrected baseline,
   not a paper.
3. **The premise may be inadmissible.** The 0.623 comes from M7, a **val-split, epoch-selected**
   reconnaissance reading; the 0.5878 is 5-fold CV in train with no selection. The same
   reconnaissance run reported 0.671 for all four channels, which becomes 0.588 under the pilot
   protocol. Verbatim: *"If the 0.623 result was not produced under exactly the same
   five-fold/no-selection protocol as 0.5878, then it is not admissible evidence. Under that
   alternative, the honest answer becomes: no legal family is left."*

That conditional is why round 13's pilot is a **falsification probe on our own motivating number**
rather than a mechanism confirmation. It is pre-registered in `idea-stage/R15_NT_FREEZE.md`
(commit `e3740dc`, committed before `scripts/r15_nt/run_nt.py` existed).

---

## 3. Families the slate missed, with occupancy

The reviewer's own sweep, `[M]` = from its memory and not independently verified here.

| family | slate coverage | occupancy and verdict |
|---|---|---|
| temporal-informativeness / nuisance suppression | **missed**; D8 only adjacent | GMU `1702.01992`, RevIN `2105.15078`, AVTS `1808.06246` `[M]`. **The only empirically live family; novelty unproved** |
| full temporal detector: pyramid + label assignment + boundary regression | **missed** | ActionFormer `2202.07925`, BSN `1806.02964`, BMN `1907.09702` `[M]`. Occupied — a mandatory baseline, never a contribution |
| change-point detection | D1 / D4 | KL-CPD `1706.01042` `[M]`, classical PELT / BOCPD. On learned scores it becomes K1; on raw features it does not create toxic evidence |
| speaker diarization | D7 | EEND `1909.06247`, pyannote `[M]`. Occupied unitization, wrong boundary semantics |
| topic segmentation | D12-adjacent | TextTiling (*Computational Linguistics* 1997) and embedding-based descendants `[M]`. Topic changes need not mark attack acts |
| sound event detection | not substantively covered | `1906.06909`, SEBB `2406.04212`, nSEBB `2505.11889`. The relevant score→event machinery is already K1 |
| video anomaly localization | not on slate | RTFM `2101.10030` `[M]`, the nSEBB video-anomaly port `2604.09327`. Occupied MIL / decoding family |
| HMM / HSMM / regime switching / semi-Markov CRF | **missed** | classical occupied family; another temporal prior over insufficient emissions, bracketed by K3 |
| CUSUM / Page–Hinkley / sequential detection | **missed** | classical occupied family. On scores → K1; on features → change-point detection |
| survival / hazard boundary modelling | **missed** | generic event-time family, no verified hateful-video occupant. Supplies duration priors, not missing moment evidence |
| weakly-supervised MIL localization | neighbourhood | MultiHateLoc, W-TALC `1807.10418` `[M]`. Occupied and already weak at high coverage |

**Every "missed" family that is not the nuisance-suppression one is either occupied outright or is a
temporal prior over emissions that round 11 already showed are the binding constraint.** That is the
structural reason the slate is not merely unlucky.

---

## 3b. Occupancy sweep on the one live family (zero-GPU agent, run in parallel with the pilot)

Rating for the composite *"per-modality within-video residualization + audio-led fusion for
per-segment harmful-content prediction"*: **b** — near neighbours exist, difference is clean. The
sweep ran while the pilot ran; the pilot then killed the family empirically, so this is recorded as
the field map, not as a live assessment. Three of its findings outlive the family.

| finding | detail |
|---|---|
| **the operator is old, one field over** | "Subtract the per-video mean feature before per-segment classification" is not published as a named mechanism in temporal localization, video anomaly detection or audio-visual video parsing — but it is exactly **cepstral mean normalization** (speech, 1970s-), **RevIN** (ICLR 2022, per-instance temporal mean/std) and per-trial baseline subtraction in EEG. Any novelty would have lived in the diagnosis, never the operation |
| **the same diagnosis is already published with a different remedy** | **UMIL** (CVPR 2023, `2303.12369`) diagnoses that *video-level context bias corrupts snippet-level prediction* in video anomaly detection, and fixes it with invariance across confident/ambiguous groups. Strongest same-goal occupant |
| **the measure-then-modulate template is crowded** | OGM-GE (CVPR 2022) and the modality-imbalance family own "measure a per-modality quantity, then modulate fusion"; the quantity is optimisation dynamics, never temporal informativeness. **MultiHateLoc** (`2512.10408`, WWW 2026 — id resolved this round) already ships *learned* per-modality importance for weakly-supervised hate localization |
| **a per-modality shuffle diagnostic exists** | `2606.00959` shuffles video/audio streams per modality and reports asymmetric sensitivity — but the shuffle is *across samples*, not within-video temporal order, and it drives data-level reweighting, not a fusion mechanism |
| **Q5 came back blank** | The specific decomposition — a modality can be helpful for *across-video* discrimination and simultaneously harmful for *within-video* discrimination because its variance is video-level — was not found stated in any domain. It was the most defensible claim in the package. **The pilot then showed it is also false on this substrate** |

**Two published contradictions the sweep surfaced, and the pilot agrees with both.** HateClipSeg's
own table uses `wav2vec2-emotion` as its audio encoder and still reports visual **30.99** > audio
**18.83** > text **11.89** F1 at tIoU 0.7, and states that late fusion fails to improve; and
`2508.04900` (MMUW'25 @ ACM MM 2025) concludes that *"hate speech does not possess uniquely
distinguishable acoustic signatures when isolated by temporal annotations."* An audio-led framing
would have had to be argued **against** two existing numbers rather than into a gap. R15-NT's G0
(audio-only 0.5507 vs fused 0.5901) lands on the same side as both.

**Coverage gaps in the sweep, recorded rather than buried.** The arXiv Atom API's phrase search
broke mid-session and `id_list` was broken throughout, so **no arXiv id in that sweep was verified by
direct lookup**; Semantic Scholar rate-limited after two calls, so there is **no citation-graph
cross-check** (a RevIN citing-paper scan is the named highest-value gap); ACM DL, IEEE Xplore and
OpenReview were reached only through search snippets; and the Q1/Q5 zero-hit results were partly
collected during the API outage, so those absences are weaker than the rest.

## 4. The ActionFormer gap — the reviewer's fourth answer, recorded in full

Our pipeline sits at **F1@tIoU0.5 = 23.8**; the dataset paper publishes **ActionFormer 52.65** on the
same corpus with roughly the same features. The reviewer's judgment:

- **No candidate on the slate addresses the reason.** D3 copies output stride and multi-scale
  context, D1 predicts boundaries; neither supplies the detector machinery — multi-level feature
  pyramid, positive/negative point assignment, localization losses, boundary-distance regression,
  detector-specific suppression.
- The comparison is **not clean** either: non-random attrition in our 395-video subset, and
  "roughly the same features" is not the same feature sequence, split or protocol. That uncertainty
  makes reproduction *more* necessary, not less.
- **A competent ActionFormer-style baseline on the frozen local split is a precondition for any
  method claim in this arena.** Otherwise any apparent gain may merely be recovering functionality
  that is missing from a thresholded-score baseline.
- And, stated plainly: *"Reproducing it is a legitimate internal baseline-readiness round. It is
  **not** a method-paper round under constraint 1. It is an admission that this direction is not
  ready for method ideation."*

This is an **open item for the user**, not something round 13 resolves. It is the second of the two
scope questions this round hands back.

---

## 5. What went to pilot

The reviewer's two prescribed sub-hour falsification experiments, adopted with the arm set extended
by three descriptive single-channel arms so that the M7 contradiction can be read directly:

- **R15-NT** — a seven-arm matched channel-composition panel (`ALL`, `AUD`, `VIS`, `TXT`, `ALLCENT`,
  `AUDCENT`, `AUDVIS0`), 5-fold video-grouped CV inside the 237 train videos, 5 seeds 4300-4304,
  identical head / optimiser / epochs / grid in every arm, primary endpoint video-macro wv-AUC,
  δ = +0.010.
- **R15-FS** — a fixed-score panel on the dumped out-of-fold scores: label-offset shifts (falsifies
  D10) and oracle region-pooling ceilings (falsifies D4), no fitting.

Both are pre-registered in `idea-stage/R15_NT_FREEZE.md`, which commits **in advance** the sentence a
null forces: that no legal mechanism family remains under the current constraints, and that the goal
is escalated to the user as a scope question rather than quietly re-attempted.
