# R11 novelty check — temporal action segmentation for hateful video, and the high-coverage regime argument

Date: 2026-08-18. Subject: the two-part direction produced by round 11 (`idea-stage/IDEA_REPORT.md`
§14, `research-wiki/TEMPORAL_SPAN_LANDSCAPE_2026-08-18.md` §10).

- **Part A (analysis / motivation):** at gold-span coverage 0.8–1.0 the mainstream temporal
  localization mechanism families (softmax-over-time, MIL top-K, intra-video saliency InfoNCE
  negatives, background classes, tIoU metrics — 12 families in landscape §10.2) are *structurally
  invalid*, and every published hate-localization method is drawn from that list.
- **Part B (method):** import the temporal action segmentation (TAS) framework — designed for 100%
  coverage, no background class, GTEA/Breakfast data scale — into per-moment hateful-video
  classification, with hate-domain adaptations (multi-hot over ~6 toxicity categories; multimodal
  audio + ASR + OCR + visual input, where the TAS literature is near-uniformly vision-only).

**Method.** arXiv API (~30 field-scoped boolean queries, title+abstract index, 2026-08 snapshot),
HuggingFace papers search (semantic, no rate limit), DBLP publication API, Semantic Scholar graph
API (citation walk on HateClipSeg — the one call that succeeded before 429), WebSearch, and one
PDF read. OpenAlex was **unavailable** (daily budget exhausted: `"Insufficient budget … Resets at
midnight UTC"`), so there is **no OpenAlex citation cross-check** in this document. One independent
cross-check by GPT-5.6-Sol at xhigh reasoning, conversation only, no tool use.

Verification tags: `[resolved]` = the arXiv id was fetched in this session and its abstract read;
`[read-pdf]` = the full text was read; `[dblp]` = the record exists in DBLP; `[s2]` = returned by
the Semantic Scholar citations endpoint; `[model-claim]` = named by the cross-check model and not
independently resolved here.

---

## 1. Question 1 — temporal action segmentation × hate / harmful / toxic content

**Result: blank at index level, and the blank is real.**

| query (arXiv `all:` field, title+abstract+comments) | hits |
|---|---|
| `"temporal action segmentation" AND "hate"` | **0** |
| `"action segmentation" AND ("harmful" OR "toxic" OR "offensive")` | **0** |
| `"action segmentation" AND "hate speech"` | **0** |
| `"temporal segmentation" AND "hateful"` | **0** |
| `"audio-visual video parsing" AND "hate"` | **0** |
| `"video parsing" AND ("toxic" OR "harmful")` | **0** |
| DBLP `action segmentation hate` | **0** |

WebSearch for `"MS-TCN" OR "ASFormer" OR "DiffAct"` × violence / content moderation / hate returned
no application paper in any of the three cases. The cross-check model independently reported that
it knows of no work running MS-TCN, ASFormer, DiffAct or another procedural-TAS model on
hateful/toxic video.

**But the blank is narrower than "nobody does dense per-moment harmful-content prediction."** Three
things already occupy the *task*, if not the *architecture family*:

1. **HateClipSeg** `2508.01712` (ACM MM 2025) `[resolved]` defines the online per-timestamp task
   itself. Renaming that task "temporal action segmentation" is a reframing, not a new task.
   The cross-check model made this point unprompted and it is correct.
2. **StreamSense** `2601.22738` (WWW 2026) `[resolved]` `[s2]` — a streaming per-timestamp detector
   for social-signal tasks including hate moderation. Verbatim: *"handles most timestamps with the
   lightweight streaming encoder, escalates hard/ambiguous cases to the VLM, and defers decisions
   when context is insufficient"*, trained with *"an IoU-weighted loss that down-weights poorly
   overlapping target segments, mitigating label interference across segment boundaries."* It has
   already noticed and patched the boundary-label problem.
3. **SafeLens** (AAAI-26, Wang, Raharja, Hu, Lee, SUTD; AAAI proceedings pp. 41712–41714)
   `[read-pdf]` — **the closest neighbour found anywhere in this check, and it is absent from the
   landscape report.** It is a per-segment multimodal hate moderation system: Whisper word-level
   speech, **EasyOCR on-screen text sampled every 3–5 s with confidence filtering**, and
   Qwen2.5-VL frame descriptions, fused into a prompt for a **LoRA-tuned Llama3-8B fine-tuned on
   HateClipSeg**, emitting per-segment label + confidence + categories + rationale. That is
   Part B's exact modality set, exact corpus, and exact output granularity.

   **What SafeLens is not:** it has **no temporal model**. Segments are produced by a
   transcript/scene-change heuristic and then scored **independently** — no sequence decoder, no
   transition prior, no duration or coverage constraint, no cross-segment smoothing. It is an
   AAAI *demonstration* paper (3 pages), reports no benchmark table in the paper itself, and points
   at a GitHub page for evaluation. So it does not close the slot, but it does remove
   "first multimodal per-segment hate classifier with OCR" from the contribution list.

**Adjacent, resolved:** ViToSA (Interspeech 2025) `[dblp]` audio-based toxic *span* detection on
Vietnamese speech, and its follow-up "A Multi-Task Approach Towards Robust Vietnamese Audio-Based
Toxic Span Detection" (ICASSP 2026, DOI `10.1109/icassp55912.2026.11460540`) `[s2]`. Speech-only,
word-span, not video, not TAS — but it is the second independent line doing sub-video harmful-span
prediction.

---

## 2. Question 2 — has anyone stated the high-coverage degradation argument?

**Result: no paper states it as a cross-family structural claim. Several state pieces of it, and
one adjacent subfield exists precisely because of it.**

### 2.1 What is genuinely unstated

Field-scoped arXiv queries returned nothing for `"foreground ratio"` × temporal action
localization, `"anomaly ratio"` × video anomaly detection, `"sparsity assumption"` × video anomaly
detection, `"action density"`, `"event density"` × sound event detection. WebSearch on the argument
in both TAL and VAD phrasings surfaced only method papers on foreground/background *separation*
(e.g. `2312.14138`, `2106.11811` `[model-claim]`/search-listed), never an analysis of what happens
as coverage → 1. The cross-check model reported the same: *"I do not know a paper that makes your
exact cross-family argument."*

So the **synthesis is open**, and so is the hate-domain measurement that motivates it (the
zero-temporal-resolution oracle at frame-AP 0.675 vs. MultiHateLoc's 0.645; MultiHateLoc's own
top-K ablation).

### 2.2 What is already stated, piecewise — and this matters

- **The sparsity prior is explicit in its source.** Sultani et al., CVPR 2018 (UCF-Crime) justify
  their sparsity loss by assuming anomalous events occupy a small part of an anomalous video
  `[model-claim, well established]`. SDST `2507.07744` `[resolved]` says it plainly in its own
  abstract: prior side-tuners are criticised for *"overlooking the inherent sparse nature of MR."*
  The assumption is not hidden; it is advertised.
- **The MIL quantity has a name.** "Witness rate" — the fraction of positive instances in a
  positive bag — is standard vocabulary in computational-pathology and immune-repertoire MIL
  (`2604.07722`, `2511.14639`, `2007.13505`, `2307.15934`, all `[resolved]` by title/abstract).
  arXiv returns **zero** witness-rate papers in video temporal localization, so the transfer of
  the term is open, but a reviewer from that community will supply it.
- **Pooling-operator occupancy dependence is established in audio.** AutoPool (McFee et al.,
  TASLP 2018) and Wang et al. ICASSP 2019 compare max/average/attention/linear-softmax pooling
  precisely as different instance-occupancy assumptions `[model-claim]`.
- **The "dense" subfield already exists.** MultiTHUMOS / Charades dense multi-label action
  detection was created because sparse-foreground TAL assumptions do not fit real video:
  "Every Moment Counts" `1507.05738` `[resolved]` (*"labeling every frame … placing multiple labels
  densely"*), MS-TCT `2112.03902` `[resolved]`, PAT `2308.05051` `[resolved]`, RefDense
  `2501.18509` `[resolved]`. This is a fourth non-invalid family the landscape's §10.3 does not
  name, and it is a closer fit to the proposed output shape (per-frame multi-label sigmoid, high
  co-occurrence, no background) than GTEA-line TAS is.
- **The metric observation is a known genre, not a known result.** Otani et al. BMVC 2020,
  "Uncovering Hidden Challenges in Query-Based Video Moment Retrieval" `[model-claim]`, is the
  canonical "the benchmark is degenerate, trivial baselines are strong" paper for temporal
  grounding. Its content is annotation prior, not coverage. Our tIoU point is the same *genre* of
  finding on a different cause.

### 2.3 Where Part A is technically wrong as written — the load-bearing objection

The cross-check model, given the full §10.2 list, accepted five items and rejected six. Its
rejections are correct and must be conceded before this is written up:

| §10.2 item | verdict |
|---|---|
| explicit foreground-sparsity penalties | **valid** — forces < cT positives by construction |
| low-saliency intra-video InfoNCE negatives (UniVTG `2307.16715`, R²-Tuning `2404.00801`, SDST `2507.07744`) | **valid** — at coverage 1.0 the negative set is literally true foreground; straightforward label contradiction |
| relative CAS thresholding / background pseudo-label mining | **valid** — guaranteed to manufacture background inside every positive video |
| duration priors that cap foreground below true coverage | **valid** |
| top-K MIL pooling | **overstated** — biased and wasteful, not incapable; it is inconsistent only when the unselected T−k snippets are *taught* to be background. Our own MultiHateLoc ablation evidence shows selection is *unnecessary* here, not that top-K cannot learn |
| **softmax-over-time pooling** | **wrong as stated — the weakest claim in the list.** Attention weights are mixture weights, not per-instant foreground posteriors. Setting a_t = 1/T for all t and reading foreground off a separate per-instant sigmoid represents 100% coverage fine. It fails only where a paper treats normalized attention *as* an absolute foreground mask |
| ActionFormer / TriDet with focal-loss negatives | **wrong** — both can emit a segment spanning the whole video; focal loss is mis-tuned under a reversed class ratio, not invalid |
| DETR-style no-object class | **wrong — category error.** No-object applies to unmatched *query slots*, not to timestamps. One query predicts the whole-video interval, the rest are no-object |
| auxiliary background class / background suppression | **overstated** — at coverage 0.8, 20% of the positive video is still background, and entirely non-hateful videos exist |
| outer–inner completeness scoring | **weakened, not invalid** — uninformative when there is no exterior context |
| tIoU metric degeneracy | **valid but must be renamed.** Correct for one contiguous gold interval (IoU = c, so c = 0.8 clears 0.3/0.5/0.7). For *multi-span* videos, standard TAL matches per gold interval, so a whole-video prediction scores each interval's own duration / T. Must be reported split by single-span vs multi-span — which we can do, since HateMM single-block is 72.8% |

The defensible restatement, which survives all of the above:

> Mechanisms that impose low foreground density, or that manufacture negatives from relative
> within-video scores, become statistically inconsistent as foreground coverage approaches 1.

That is roughly a third of the current list, not twelve families.

One further methodological concession: the frame-AP 0.675 oracle uses a **perfect** video-level
classifier and is therefore not a fair model comparison against a learned WSTAL system — it shows
that frame-AP leaks video-level separability and prevalence. The honest version is a **learned**
video-classifier-broadcast baseline with declared tie handling, plus coverage-stratified reporting.

---

## 3. Question 3 — who has used HateClipSeg since 2025-08?

Semantic Scholar citation walk on `arXiv:2508.01712`, retrieved 2026-08-18, **4 citers**:

| citer | id | what it does with HateClipSeg |
|---|---|---|
| StreamSense: Streaming Social Task Detection with Selective VLM Routing (WWW 2026) | `2601.22738` `[resolved]` | **attacks the online per-timestamp task**; streaming encoder + VLM escalation + deferral, IoU-weighted segment loss |
| A Multi-Task Approach Towards Robust Vietnamese Audio-Based Toxic Span Detection (ICASSP 2026) | DOI `10.1109/icassp55912.2026.11460540` `[s2]` | related-work citation; audio-only span task |
| Beyond Binary Classification: Detecting Fine-Grained Sexism in Social Media Videos | `2602.15757` `[resolved]` | related-work only; contributes FineMuSe, a Spanish video sexism dataset, LLM evaluation, no temporal task |
| Reasoning-Aware Multimodal Fusion for Hateful Video Detection (RAMF, TMLR 2025) | `2512.02743` `[resolved]` | video-level classification; cites HateClipSeg as related work |

**Plus one citer Semantic Scholar missed: SafeLens (AAAI-26)** `[read-pdf]`, §1 above — it
fine-tunes on HateClipSeg and produces per-segment predictions. Found via WebSearch, not via the
citation graph. Treat the S2 count as a lower bound; without OpenAlex there is no second graph.

**Answer to the sub-question.** The per-timestamp task has been attacked by (a) HateClipSeg's own
baselines, (b) StreamSense, (c) SafeLens. **Nobody has attacked it with a sequence-segmentation
model.** That specific slot is empty.

---

## 4. Question 4 — how multimodal is the TAS literature?

**Thin, but not empty, and getting less empty in 2026.**

- arXiv `ti:"audio-visual" AND ti:"action segmentation"` → **0**.
- arXiv `abs:"action segmentation" AND abs:"speech"` → 2 hits, neither relevant.
- DBLP `temporal action segmentation audio` → **0**.
- DBLP `multimodal temporal action segmentation` → **3**: Ego-METAS (2026), a CoDIT 2024
  manufacturing paper, a 2020 Sensors paper.

What exists:

- **Ego-METAS** `2606.02246` `[resolved]` (2026-05) — *"the first Egocentric online Multimodal
  Energy-efficient Temporal Action Segmentation benchmark"*, 5 modalities (RGB, audio, gaze, IMU,
  monochrome), **online** TAS formulation, 100+ hours. This is the strongest evidence that
  multimodal *online* TAS is now a named setting; it is egocentric/embodied, not content-moderation.
- **M2R2** `2504.18662` `[resolved]` — multimodal (proprioceptive + exteroceptive) TAS feature
  extractor for robotics, JIGSAWS/REASSEMBLE.
- **Multi-modal TAS for manufacturing scenarios** (Eng. Appl. AI 2025, ScienceDirect) — surfaced by
  WebSearch, `[unverified]`.
- **AVVP** `2007.10558` (ECCV 2020, already in landscape §10.4) — per-second **multi-label**
  audio-visual event parsing. The cross-check model flagged this as the structural twin of the
  proposed multi-hot multimodal timeline output, and that is fair: it is the same output shape with
  a different label vocabulary. Also AVEL (Tian et al. ECCV 2018) `[model-claim]`.

So the honest claim is: **modern procedural TAS is evaluated with visual features only, and the
multimodal variants that exist are robotics/egocentric; hate-specific audio + ASR + OCR + visual
fusion in a TAS-shaped model is unoccupied.** The claim "dense multimodal temporal labeling is new"
is false and must not be made — AVVP owns it.

---

## 5. Question 5 — the video-anomaly-detection analogue

**No analogue found, and there is a conceptual reason.**

- arXiv: `"video anomaly detection" AND "long anomalies"` → 0; `"anomaly ratio"` × VAD → 0;
  `"sparsity assumption"` × VAD × video → 0.
- WebSearch surfaced only incidental remarks (e.g. Subway Entrance events being longer than the
  100-frame prediction window) — nothing with the failure regime as its subject.
- The cross-check model: *"I do not know a verified VAD paper centered on systematic performance as
  the anomaly ratio approaches 100%."*

Its structural objection is worth recording: a "no-background anomaly detector" is close to
incoherent, because anomaly is defined relative to normality — even if every frame of one positive
video is anomalous, the normal videos still supply the negative class, and under a multi-label
sigmoid "all categories off" simply *is* the background state, re-encoded. Removing the explicit
background class changes the parameterisation, not the problem. The same objection applies to
Part B's "TAS has no background class, therefore it fits" argument: at HateMM's coverage the
non-hate remainder still exists, and at HateClipSeg's 0.544 the normal class is nearly half the
timeline. **The no-background property is the weakest of the three reasons given for importing TAS**;
the strong ones are the data scale (GTEA 28 / 50Salads 50) and the absence of intra-video
contrastive negatives.

Adjacent and already correctly excluded by the landscape: UCF-Crime / RTFM / MIST / MGFN /
XD-Violence are all normal-vs-abnormal MIL ranking or magnitude contrast `[model-claim, standard]`.

---

## 6. Ratings

Scale: **a** = open slot plus a real mechanism insight; **b** = near neighbours exist, difference is
clean and defensible; **c** = crowded; **d** = occupied.

| | rating | reason |
|---|---|---|
| **Part A alone** | **b** | The cross-family synthesis and the hate-domain measurement are genuinely unstated. But the pieces are individually published (sparsity priors advertised in their own abstracts; witness rate named in MIL; pooling-occupancy in SED; the whole dense-action-detection subfield exists because sparse TAL does not fit dense video), and **six of the twelve listed families do not actually break** — softmax attention, ActionFormer/TriDet, DETR no-object, focal loss, background class at 0.8, top-K pooling. As written it is a false universal; narrowed to "objectives that force low density or manufacture within-video negatives," it holds. Also: **Part A alone is an analysis contribution and is closed by the method-paper-only rule regardless of its rating.** |
| **Part B alone** | **c** | Three occupants of the task (HateClipSeg, StreamSense, SafeLens), one of them — SafeLens, AAAI-26 — with the identical modality set and corpus. The high-coverage per-frame multi-label output shape is owned by dense action detection (MultiTHUMOS/Charades line) and by AVVP. What remains open is *architecture transfer*: no sequence-segmentation model has been run on hate data. That is a first-application, not a mechanism. |
| **A + B as one method paper, as specified** | **c** | The analysis is stronger than the method, and the method does not operationalise the analysis: nothing in "run ASFormer/DiffAct with multimodal features" is coverage-aware. The reviewer reads the two halves as unconnected. |
| **A + B with an actual coverage-aware mechanism** | **b** | Reachable, not reached. See §7. |

### The single strongest reviewer objection

> The structural-invalidity thesis conflates a training prior with representational impossibility.
> Normalized attention, focal-loss detectors, ActionFormer and DETR can all represent whole-video
> foreground, so TAS is neither uniquely necessary nor shown to fix the stated problem.

Second strongest: **there is no new method.** Third, and specific: HateClipSeg's per-timestamp task
is **causal/online**, while MS-TCN, ASFormer, DiffAct and Viterbi/CAD decoding are **acausal**.
Running them on that task without a causal conversion is an evaluation-protocol violation, and a
separate offline protocol forfeits comparison with StreamSense's 72.06.

---

## 7. Three nearest neighbours, and the difference that would have to be defended

1. **SafeLens** — AAAI-26, Wang / Raharja / Hu / Lee (SUTD), pp. 41712–41714, `[read-pdf]`.
   Per-segment multimodal hate moderation on HateClipSeg: Whisper speech + EasyOCR on-screen text +
   Qwen2.5-VL descriptions → LoRA Llama3-8B → per-segment label, confidence, categories, rationale.
   **Difference:** segments are scored *independently*; there is no sequence model, no transition or
   duration prior, no coverage constraint, and no cross-segment smoothing. Everything our direction
   would add is temporal structure over exactly this input. **It is also a demo paper with no
   benchmark table in the PDF**, so it is a citation risk, not a numeric baseline. Same lab as
   HateClipSeg — they will be the reviewers.

2. **StreamSense** `2601.22738`, WWW 2026, `[resolved]`. Occupies the online per-timestamp task with
   a streaming encoder + selective VLM escalation + deferral, and already has a loss that
   down-weights poorly overlapping target segments at boundaries. **Difference:** routing/latency
   architecture rather than sequence segmentation, and no coverage or duration prior. **Warning:**
   its escalation/deferral primitive is this project's already-dead uncertainty-gated deferral
   (−0.0135, 0/3 seeds), so beating it must not route through that mechanism.

3. **The dense action detection line** — "Every Moment Counts" `1507.05738`, MS-TCT `2112.03902`,
   PAT `2308.05051`, RefDense `2501.18509`, all `[resolved]`. Per-frame multi-label sigmoid, heavy
   co-occurrence, no background class, on MultiTHUMOS/Charades. **Difference:** vision-only, action
   vocabulary, no hate. **Why it hurts:** it is a fourth family that is not on the invalid list and
   is a *better* fit to a 6-way multi-hot toxicity timeline than the GTEA-line TAS the proposal
   names. A reviewer will ask why MS-TCT/PAT are not the baselines, and "TAS is the only surviving
   family" is not defensible with this line on the table.

Honourable mentions: **AVVP** `2007.10558` (per-second multi-label audio-visual parsing — the output
shape); **Ego-METAS** `2606.02246` (the multimodal *online* TAS setting already named, 2026);
**CAD** `2605.10149` `[resolved]` — verified real, *"transition confidence, action boundary sets,
and per-class duration"* folded into a modified Viterbi, training-free, code at
`github.com/LUNAProject22/CAD`. CAD remains the single most directly liftable coverage-prior
mechanism found in either sweep.

---

## 8. Direct judgment — can this be pushed as a method paper?

**Not as currently specified. It is one mechanism short.**

The blank is real (no TAS on hate, anywhere, by four independent indexes), and the diagnosis is
real. But the direction as written is *analysis + architecture transfer*, and the project's
method-paper-only rule bans the analysis half from being the contribution while the transfer half
is a first-application on a task that three papers already occupy — one of them, SafeLens, with the
identical modality set on the identical corpus.

**What would convert it to a defensible `b`,** all of which is cheap on the existing HateClipSeg
cache (per §14.5 the K=30 grid is already ~8.0 s vs a median gold segment of 8.12 s):

1. **A mechanism that consumes coverage.** The obvious one is CAD-style constrained decoding with
   a **per-video** duration/coverage prior conditioned on the video-level hate score — i.e. the
   video-level classifier the project already has sets the decode's foreground budget. That is a
   real coupling between Part A and Part B, it is training-free, and it is not in any of the three
   occupants.
2. **A training objective that provably never manufactures within-video negatives** — plain
   per-instant BCE, no intra-video contrastive term — stated as the falsifiable consequence of the
   narrowed Part A claim, and tested against a UniVTG-style intra-video-negative arm on the same
   features. That is a controlled experiment on the surviving third of the invalid list, not a
   rhetorical claim about twelve families.
3. **Causal-vs-offline discipline.** Report the online task causally (or do not report it), and put
   the acausal segmentation numbers in a separately labelled offline protocol.
4. **Coverage-stratified evaluation with a learned broadcast control**, split single-span vs
   multi-span, replacing the perfect-oracle framing.
5. **Drop the "no background class" argument** and the six wrong entries in §10.2 before anything
   is written down.

Absent (1) and (2), the honest label is **c**, and a reviewer's "this is ASFormer with extra
features on a dataset whose authors already shipped a per-segment system" would stand.

---

## 9. Corrections to the round-11 documents that this check produced

- **`TEMPORAL_SPAN_LANDSCAPE_2026-08-18.md` is missing SafeLens (AAAI-26).** It is the nearest
  neighbour in the hate domain and it is from the HateClipSeg authors. Add to §2.1 / §6.1.
- **§10.2's list is wrong on six of twelve entries** (§2.3 above). It should not be carried into a
  pre-registration or a paper in its present form.
- **§10.3 omits the dense action detection family** (MultiTHUMOS/Charades: `1507.05738`,
  `2112.03902`, `2308.05051`), which is a closer architectural fit than GTEA-line TAS and is also
  not on the invalid list.
- **RefDense `2501.18509` is mischaracterised.** The landscape calls it "per-frame multi-label
  sigmoid, no background class, coverage-agnostic by construction." The abstract's actual
  contribution is decomposition of ambiguous dense action classes into unambiguous sub-concepts
  handled by separate sub-networks, plus a language-guided contrastive loss to replace independent
  BCE — on Charades and MultiTHUMOS, i.e. dense action *detection*, not GTEA-line TAS. Directionally
  right, specifically wrong.
- **Coverage figure mismatch:** landscape §10.5 quotes HateClipSeg at 44.6% coverage;
  IDEA_REPORT §14.5 measures median toxic coverage **0.544** from the local copy. Reconcile.
- **CAD `2605.10149` and SDST `2507.07744` are both confirmed real** with the quoted mechanisms;
  no correction needed.

## 10. Coverage gaps in this check

- **OpenAlex was unavailable** (budget exhausted until 00:00 UTC), so there is exactly one citation
  graph behind §3, and it already demonstrably missed a paper (SafeLens). Re-run the HateClipSeg
  citation walk on OpenAlex before any pre-registration.
- **Semantic Scholar returned 429** on the second call; only the citations endpoint succeeded.
- arXiv `all:` indexes title, abstract and comments only, so all "0 hits" results above are
  **absence from titles and abstracts**, not absence from full texts.
- HuggingFace papers search is semantic, not boolean, and returns a fixed 120-result ceiling; it was
  used for recall, never for an absence claim. It has **no record of HateClipSeg at all** (0 hits),
  which is itself a reason not to rely on it for negatives.
- SafeLens's evaluation numbers live on GitHub, not in the paper; they were not retrieved.
