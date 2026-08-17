# Idea-generation bundle — hateful video detection, mechanism-level novelty (2026-08-09)

You are a senior ML researcher brainstorming research ideas. Read everything below, then produce
the deliverable in the final section. Reason hard; this is a high-stakes selection.

---

## 0. The research direction

Hateful / harmful **video** detection (short social-media clips; binary harmful-vs-normal).
The team's base system is a **retrieval-guided contrastive** pipeline ported from RGCL
(hateful-meme detection, ACL 2024):

- FROZEN CLIP ViT-L/14-336 (and a frozen Qwen2.5-VL variant) → precomputed embeddings
  (8 uniformly sampled frames mean-pooled for video; title + ASR transcript for text)
- lightweight ~5 M-param MLP head (HateClipper-style element-wise "align" fusion)
- FAISS retrieval-guided contrastive (triplet) + BCE hybrid loss
- inference by retrieval kNN read-out over a memory of training items

**Target**: a *methods* paper at NeurIPS / ICML / ICLR / CVPR / ACL **main conference**.
Novelty must be **mechanism-level**. Accuracy alone is no longer a viable claim (see §2).

---

## 1. Assets actually in hand (this is what makes ideas cheap or expensive)

**Data (local, this workstation: 1× RTX 5090 32 GB shared, 16 CPU, 60 GB RAM, no SLURM):**
- HateMM (EN, BitChute): 1083 videos total; train split 744 (298 hateful) / dev_seen 107 / test 215.
  Official **gold hate-span** annotations exist.
- MultiHateClip EN (n=161 test) and ZH (n=149 test) — 3-class source labels
  (Hateful / Offensive / Normal), collapsed by this project to binary (Offensive folded INTO hateful).
  Also carries segment timestamps, target group, contributing-modality fields.
- HateClipSeg (segment-level annotations, 395 videos cached), ImpliHateVid.
- **Precomputed CLIP embeddings for all four datasets** (`data/CLIP_Embedding/`, 2.2 GB) — FAISS
  pipeline is ready; whole-video and per-segment (K=30 windows) visual vectors exist.
- **OCR cache (project-unique)**: PaddleOCR PP-OCRv6 over K=30 mid-point windows,
  1246 videos (HateMM 851 → 25,530 windows; HateClipSeg 395 → 11,850 windows), per-detection
  bounding boxes + confidences, SHA-256 manifest. HateMM: 80.85 % of videos have filtered text;
  57.85 % of windows have text; median 390 chars/video.
- Audio caches for HateMM / MHC-EN / MHC-ZH (247 MB).
- **Gate-C blinded human adjudication census**: 73 false negatives + 30 true positives, each coded
  for `required_modalities` (visual / on_screen_text / transcript / audio) and, on 99 videos,
  a **minimal sufficient evidence interval** (a much sharper localization target than the official
  span). Coverage: official spans cover mean 0.717 / median 0.829 of the video; coder minimal
  intervals cover median 0.131 — a **2.0×** ratio on 99 paired videos.
- MoRE (WWW 2025, the only published retrieval-augmented hateful-video method) **official code
  fully re-run** on identical splits with 7 documented code defects handled. Cross-dataset memory
  swap matrix (6 informative cross cells, 5/6 above-majority, zero retraining).
- Head-level retraining costs ~52 s. Frozen-feature CPU experiments cost seconds to minutes.
- Claude API may read raw video frames/clips for **annotation/audit** purposes (general exemption).
  **Hard rule: Claude-produced or Gate-C annotations must NEVER enter any deployable training or
  inference path — audit/analysis only.**

**Performance level today** (binary, test): HateMM 0.870 acc / 0.861 macro-F1 (single seed);
ImpliHateVid ~0.90/0.90; MHC-EN 0.789/0.738; MHC-ZH 0.832/0.802.

---

## 2. The landscape, and the three things that just closed down

Recent (2025-12 → 2026-08) sweep of the field. Verification level in brackets:
[A] = confirmed via arXiv API, [B] = confirmed via Crossref/S2/OpenAlex DOI metadata, [C] = unverified.

### 2.1 Three framing killers — any new idea must route around these

1. **SAGE** (ACL 2026 Main Long, `10.18653/v1/2026.acl-long.817`) [B]: decision-level expert
   arbitration + instance-level "tribunal" against *feature dilution*. HateMM **0.8710 acc /
   0.8628 macro-F1** — statistically indistinguishable from this project's own 0.870/0.861.
   ⇒ **The HateMM accuracy race is closed.** Everyone is fighting in the third decimal.
   A pure accuracy claim is not publishable at a main conference any more.
2. **HCG-MPB** (ICMR 2026, `10.1145/3805622.3810724`) [B]: replaces per-instance retrieval with an
   LLM-distilled **prototype bank**, and *explicitly argues in its motivation that instance-based
   retrieval is a flawed design* (semantic ambiguity + storage/latency cost).
   ⇒ Any RGCL-family hateful-video paper must now rebut this in related work.
3. **`2607.23304` Context-Adaptive Inference** [A]: proves that under **squared loss + linear
   prediction head + fixed features**, explicit parameter adaptation and implicit routing are BOTH
   equivalent to kernel ridge regression on joint (input, context) features.
   ⇒ **"Our retrieval module is a form of (test-time) adaptation" is formally absorbed.**
   Related: **ERM `2602.05152`** [A] proves query expansion ≡ key expansion under standard
   similarity ⇒ "we improved the query/key construction" is no longer an independent claim.

### 2.2 Occupancy map by mechanism slot

**Retrieval / memory**: MoRE (WWW 2025, frozen weighted-cosine retriever, BCE-only supervision);
HCG-MPB (prototype bank); CRAVE (ICCV 2025, cross-domain retrieval-augmented *training*);
Class-RAG (Meta, hot-swappable library + "semantic hotfixing"); *Now You See the Hate*
(`2607.19061`, retrieve-and-calibrate over a complementary view bank, image domain).
Out-of-domain key design is heavily filled in 2026H1: LaPR (CVPR 2026, image-label joint key +
query-adaptive MoE routing), CIRCLES (CVPR 2026, attribute-decomposed keys), ERM.
**Neighbourhood-consensus denoising is CLOSED as a mechanism claim** — AAAI-26 `2512.24064`,
ICML-26 IN2R `2606.04061`, CVPR-26 ConeSep `2604.20358` all landed within 8 months. BUT: all three
are **noisy correspondence** (a wrong pairing exists, a unique correct answer exists, neighbourhood
geometry can recover it). **Subjective annotator disagreement has no recoverable correct answer —
their mechanism premise does not transfer.**

**Temporal / localization**: MultiHateLoc (WWW 2026, modality-aware hard top-K MIL, HateMM frame
mAP 0.645 / AUC 0.799); LELA (`2602.09637`, training-free 5-modality per-frame LLM prompting);
TANDEM (AAAI-ICWSM 2027); HateClipSeg (ACM MM 2025 baseline, localization degrades hard with tIoU,
and V+T+A is *worse* than visual-only at every tIoU); MultiHateGNN (BMVC 2025, soft attention
segment aggregation). **Essentially no room left on the modelling side.**

**Modality fusion**: SAGE; HCG-MPB; **TIHD/QGC-Net** (ICMR 2026, names the "Alignment Trap" —
symmetric consistency-seeking fusion erases cross-modal *incongruity*, which is exactly the signal
for implicit hate; occupies "cross-modal contradiction = evidence"); MM-HSD (`2508.20546`, OCR as
cross-modal-attention query, self-reported HateMM 0.878/0.874 under 5-fold CV — a DIFFERENT
protocol, do not mix); UniSafe (WWW 2026 Companion, frozen encoders + shared-safety-space
projection, order-invariant aggregation ⇒ any non-empty modality subset without retraining;
ablation says **modality dropout** is the main robustness driver).

**Supervision signal**: LEAF (ACL Findings 2026, LMM-teacher explanation distillation);
DeHate (ACM MM 2025, human segment-level contributing-modality labels); IARE (SIGIR 2026,
fine-grained harmful elements + gold rationales); SenBen (CVPRW 2026, sensitive scene graphs);
IPS (ACL 2026 Industry, in-prompt process supervision); **Beyond Hate `2603.22985`** [A/C]
(re-labels 2030 Hateful Memes along two separable axes — **incivility (tone) vs intolerance
(content)** — joint coarse+fine training halves the FNR−FPR gap: 0.74→0.42 LLaVA-1.6,
0.54→0.28 Qwen2.5-VL). **Out-of-domain learning-with-disagreement is EXTREMELY crowded**
(LeWiDi-2025, DiADEM `2604.08425`, EDO `2607.08493`, soft-label `2511.14117`, Socio-Contrastive
`2604.18069`, RGPO `2607.20515`, STABLEVAL `2605.02122`, NEC `2605.03135`) — **but every single
one is text; ZERO are video, and ZERO connect disagreement to retrieval.**

**Inference strategy**: SCANNER (AAAI 2026, source-free TTA); MARS / HVGuard / LELA / MATCH
(training-free / CoT / multi-agent verification — saturated); **cost-aware acquisition / "should we
retrieve at all" is the single most crowded area of the sweep** — ICLR-26 `2601.22570`
(retrieval-based selective prediction), ACL-26 `2605.13277` (information-gain evidence utility,
with proofs), VOILA `2602.03007`, `2607.05438`, `2606.29959`, `2606.11907` (ECML 2026), 12+ papers.
**All of them define cost as compute / latency / tokens and benefit as accuracy against a fixed
gold label.** ResponseGuard `2607.21401` is a notable reverse result: a 2B non-CoT guard beats a 3B
reasoning guard at ~150× lower latency, and attributes the residual gap to the **frozen visual
encoder**, not to missing reasoning.

**Evaluation protocol**: the ONE slot the sweep confirmed **entirely empty** for hateful video.
Adjacent occupants are all out-of-domain: NExT-GQA (Acc@GQA) / EG-VQA (`2606.24797`, joint
correct-and-grounded metric shape); NEC `2605.03135` (per-item cost weighting from annotator vote
margin — and it honestly reports that cost-sensitive **training** gains are unstable, its
contribution is on the evaluation side); PaSBench-Video `2606.02443` (frame-level risk-onset +
explicit false-positive control). Also **AAAI 2026 `10.1609/aaai.v40i42.40841`** [B]: five SOTA
VideoLLMs have **>90 % miss rate** on harmful content, attributed to sparse uniform frame sampling
+ aggressive visual-token downsampling — this is a **validity threat to our own 8-frame / K=30
uniform sampling**.

Other constraints: **`2606.11198` The Structural Attention Tax** [A/C] — the *format* of retrieved
content distorts attention independently of relevance (demonstration attention compressed up to
42 %), so any causal claim of the form "retrieval helps because retrieved content is relevant"
must control for it. **`2604.17375` When Text Hijacks Vision** and **`2608.04244` SIGNPOST-Bench**
[A] — when on-screen text conflicts with the image, MLLMs systematically hallucinate toward the
overlaid text's semantics.

### 2.3 The "never claim novelty for this" list

1. Growing/replacing a datastore for zero-gradient domain adaptation (kNN-LM ICLR'20, kNN-MT ICLR'21).
2. Inserting the model's own test-time predictions back into memory (AdaNPC ICML'23, TDA CVPR'24).
3. Age/staleness-scored memory eviction (RoTTA CVPR'23, Lu et al. AIJ 2016).
4. Confidence/entropy-gated cache admission (CRG / ACE / DOTA / SCA, 2025).
5. Wave-wise memory insertion + evaluation on later time slices (Mireshghallah et al. EMNLP 2023,
   `2209.05706` — frozen encoder + kNN memory + future-slice testing + beating retrained baselines,
   +64 %; one paper does all of it).
6. "Non-parametric continual learning" as a phrase (HippoRAG 2, ICML 2025).
7. "Zero-gradient vs gradient are mechanistically different" — refuted by `2305.13034` (EMNLP'23).
8. Cross-lingual retrieval transferring hate-detection supervision from high- to low-resource
   (Ghorbanpour et al., EMNLP 2025 Main); merged multilingual datastore (Stap & Monz EMNLP'23,
   CORA NeurIPS'21); PARC / XRICL / XAMPLER for demonstration retrieval.
9. "Our retrieval module is test-time adaptation" (`2607.23304`).
10. "We improved retrieval query/key construction" as a standalone mechanism claim (`2602.05152`).
11. "Neighbourhood-consensus denoising" as a mechanism novelty (three top-tier papers in 8 months).
12. "Per-sample decision of whether retrieval/escalation/abstention is worth it" (ICLR-26
    `2601.22570` + ACL-26 `2605.13277` + VOILA, with proofs).
13. "Cross-modal inconsistency/contradiction signals implicit hate" (TIHD ICMR 2026 in-domain).
14. "We are the first to bring annotator disagreement to hate detection" (text side is packed —
    can only be claimed **on video**, and must cite all of them).
15. "We are the first to use on-screen text for hateful video" (MM-HSD / LELA / WWW26 agentic).

---

## 3. DEAD DIRECTIONS — do not re-skin these, they are empirically closed by this project

Each was killed by this project's own frozen-verdict experiments. Re-proposing a cosmetic variant
is the single worst failure mode here.

| dead direction | how it died |
|---|---|
| **Multi-segment complementarity** | TERA Gate-0: NO-GO-C, only 6/73 = 8.2 % of failures need multiple segments; 83.6 % are short-window local/cross-modal. |
| **Single-segment selection** | Same gate; SAGE independently solved the same "feature dilution" framing with decision-level arbitration instead of segment selection. |
| **OCR − ASR residual as a key** | External review: `r = o − a` is a fixed linear projection `[I, −I]` (2/10); TIHD occupies "contradiction = evidence" in-domain. |
| **CVoI / cost-of-information acquisition** | Killed; and §2.2 shows the slot filled by 12+ ICLR/ACL-tier papers in 2026H1. |
| **Segment-level retrieval keys** | Sign-flips by language; and the 2026-08-09 late-interaction pilot (below) refuted it directly. |
| **Visual-purity segment selection** | P2 forensic: within-video AUROC **0.511** [0.488, 0.533] — the statistic carries NO within-video localization signal. |
| **Type-hard-partitioned memory** | Dead. |
| **Streaming / continual memory** | W4: memory insertion flat-to-negative in every cell; the surviving fix (threshold recalibration, 0.6273 → 0.7336 with k=20) is classic label-shift adaptation (BBSE / Saerens-EM). |
| **Cross-lingual EN memory rescuing ZH** | Premise is backwards — our ZH (0.802 macro-F1) beats our EN (0.743); measured EN→ZH transfer −0.138 macro-F1. |

## 3.1 Today's fresh negative results (2026-08-09) — read these carefully, they are informative

**(a) OCR three-stream fusion pilot → AMBIGUOUS (+0.0094, sub-threshold).** Mean-pooled CLIP-text
embedding of OCR windows concatenated to a *linear head* input: +0.0094 macro-F1, positive on 3/3
seeds but below the +0.015 GO bar. Dose curve strongly concave: 3 of 30 windows recover 61 % of the
gain. 20.2 % of videos have no usable OCR anywhere.

**(b) Late-interaction segment retrieval pilot → NO-GO.** MaxSim over 30 retained segment keys is
WORSE than the single whole-video key: neighbour purity@10 lift +0.144 vs +0.179 baseline
(paired Δ −0.019, CI excludes 0); kNN macro-F1 −0.043. Adding OCR into the segment key makes it
worse again (−0.018). Attribution: dropping the transcript costs −0.029 — **the transcript block
carries most of the baseline's retrieval advantage**. Hypothesised (not measured) cause: `max_j` is
an extreme-order statistic over ~17,850 memory segments, so it is dominated by generic hub segments
(title cards, black frames, talking-head crops; watermarks / channel handles / UI chrome on the OCR
side).

**(c) A0 ± OCR end-to-end → NO-GO (−0.0246 macro-F1, 3/3 seeds).** The SAME OCR vector that gave
+0.0094 through a frozen linear head gives **−0.0246** when routed through the learned fusion MLP +
contrastive loss (a sign flip, ratio −2.6×). Confounded with capacity (+1.84 M params, +36.8 %, on
744 training videos; no parameter-matched control was pre-registered).
**Anomaly worth noting: at the selected epoch, retrieval ROC went the OTHER way — arm A 0.8821 vs
arm B 0.9008, i.e. +0.019 for the OCR arm. The OCR stream made the learned-space kNN similarity
*ranking* better while the thresholded kNN *vote* got worse.** (Selection-confounded: epochs were
chosen by accuracy, not ROC, so the arms are read at different epochs. Recorded as an observation,
not a result.)

**(d) P2 forensic (diagnosis, not method).** The famous "segment-keyed retrieval lands below chance
(0.544 vs chance 0.762)" number was an `np.argmax` tie-break artifact interacting with a positional
prior in HateMM gold spans (median 2 distinct values across 30 segments; 51.3 % of hateful videos
select k=0). Under a random tie-break: 0.768 vs chance 0.762 — exactly at chance. The real,
surviving findings:
  - a frozen-CLIP visual segment key's neighbourhood-purity statistic is a strong **video-level**
    label/style/channel detector (AUROC 0.782) and a **coin flip within a video** (AUROC 0.511).
  - **Transferable design rule**: any selection score that is a bounded vote/count over a memory
    this size is degenerate by construction (K=20 ⇒ 21 levels; 52 % of hateful videos saturate at
    1.0 across half their segments) — selection collapses to whatever the tie-break is. Continuous,
    non-saturating scores (margins, similarity-weighted quantities, calibrated probabilities) do
    not have this failure mode.
  - Localization metrics on HateMM official spans are near-unusable: chance is 0.762 and the
    positional prior alone spans 0.34 → 0.86.

---

## 4. The seven open slots the survey identified, with their honest ceilings

1. **Chance-corrected grounding/localization evaluation for hateful video.** Per-video attainable
   hit probability is never corrected for; on HateMM it is 0.762. Also: `2508.04900` reports
   "trim to gold span → +19.34/+30.45 macro-F1", but this project's *controlled* decomposition with
   a length-matched random-window control gives full 0.8196 / random-window 0.8155 / gold-span
   0.8203 ⇒ generic trim −0.41 pt, oracle alignment **+0.48 pt**, bootstrap CI [−0.79, +1.76].
   Honest ceiling: **ACL Main (Resources & Evaluation) / NeurIPS D&B / ACM MM. Not a mechanism paper.**
2. **Retrieval over contested labels** (neighbours whose labels have no correct answer).
   Shape is right for a mechanism paper. **Gated on a data-availability check (see §5).**
3. **Buying ANNOTATION rather than compute as the cost axis of value-of-information.**
   Empty intersection. Gated on the same check + needs a measurable "disagreement reduction" payoff.
4. **Prosody / paralanguage as a retrieval KEY (not as a fourth modality).** Slot verified empty by
   two independent searches. But our own evidence opposes it: 65.5 % of MHC-EN hate evidence is
   carried by speech/on-screen-text, visual-only is 15/168 = 8.9 % — hate is in *what is said*,
   not *how it is said*. Cheapest idea to implement (audio caches exist).
5. **Per-instance memory vs distilled prototypes** — the controversy HCG-MPB just created. Nobody
   has run a fair head-to-head. We have the only full MoRE re-run + a zero-retraining cross-dataset
   memory-swap matrix. Honest ceiling: empirical study, ICMR/ACM MM; D&B only with a benchmark
   contribution. **Blocked: HCG-MPB PDF is paywalled.**
6. **Near-duplicate contamination + "naturally occurring minimal pairs"** (same footage, opposite
   label: hateful original vs news-report / counter-speech repost). Nobody has audited near-dup
   leakage on these benchmarks. $0 / CPU-minutes with the cached CLIP embeddings + FAISS.
   Verdict depends entirely on what the measurement finds.
7. **Provenance typing of on-screen text**: uploader-overlaid text (subtitles, title bars, meme
   text = the uploader's own speech act, attributable intent) vs in-scene text (signs, clothing,
   book pages = filmed content, possibly evidence ABOUT hate rather than hate itself). Falls exactly
   on the field's worst failure mode (news reports / counter-speech / quotation judged hateful).
   Our OCR cache carries per-detection boxes ⇒ cross-window box-position stability is computable for
   $0, and overlaid text is precisely the "fixed position, persists across frames" class.
   Honest ceiling: **not enough as a standalone paper; possibly a component.**

---

## 5. THE DATA GATE — RESOLVED, AND IT IS A **GO** (audit completed 2026-08-09)

Slots #2/#3 hinged on: **do our video datasets carry an annotator-level disagreement signal?**
A full audit (local files + the actual bytes of the upstream releases) has now answered it.

### 5.1 The headline: MultiHateClip DOES release raw per-annotator votes

**This overturns the project's own written record.** `research-wiki/PAPER_MASTER_TABLES.md:446`
states "per-annotator votes do not exist (limitations hard constraint) ... neither in-repo nor in
the public release ⇒ the LeWiDi / annotator-distribution soft-label family is sealed off at the
data layer." **That is false.** The project was working from a *reduced* derived copy
(`annotation(new).json`, HVGuard-derived: `Video_ID, Title, Transcript, Label` only). The
**official** Social-AI-Studio release carries much more. Verified by downloading the actual files
from `github.com/Social-AI-Studio/MultiHateClip`, `{English,Chinese}_data/annotation/{train,valid,test}.tsv`:

```
Video_ID    Majority_Voting    Label    Target_Victim    Component    Duration
3v7239cr4z0 Hateful  ['Hateful','Counter Narrative','Hateful']  ['...']  ['Vision component','Transcript','Audio','Metadata']  [(0, 46)]
pQbEa24u-HM Normal   ['Normal','Offensive','Normal']            ['Woman'] ['Transcript','Metadata','Vision component']         [(2, 10)]
```

`Label` is the **list of individual annotators' raw labels**; `Majority_Voting` is the aggregate.
Measured directly (2001 videos, EN 1001 / ZH 1000):

| quantity | MHC-EN | MHC-ZH |
|---|---|---|
| vote-list length 2 / 3 / 4 | 824 / 174 / 3 | 752 / 237 / 11 |
| **non-unanimous items** | **213 (21.3 %)** | **299 (29.9 %)** |
| **items split across OUR binary boundary** (harmful vs normal) | **123 (12.3 %)** | **162 (16.2 %)** |
| items with a `Counter Narrative` vote | 63 | 76 |
| items with `Component` (contributing modality) | 380 | 409 |
| items with `Duration` (segment timestamps) | 331 | 327 |

**Joinability: perfect.** Every one of our local IDs is present upstream (MHC 790/790,
MHC_zh 806/806). Per split, without touching test:

| split | n | non-unanimous | binary-split | ≥3 votes (escalated) |
|---|---|---|---|---|
| MHC train | 549 | 113 (20.6 %) | 54 (9.8 %) | 88 |
| MHC val | 80 | 20 (25.0 %) | 9 (11.2 %) | 15 |
| MHC test | 161 | 34 (21.1 %) | 26 (16.1 %) | 32 |
| MHC_zh train | 579 | 189 (32.6 %) | 105 (18.1 %) | 157 |
| MHC_zh val | 78 | 20 (25.6 %) | 10 (12.8 %) | 15 |
| MHC_zh test | 149 | 35 (23.5 %) | 20 (13.4 %) | 30 |

Three structural facts that matter for mechanism design:
1. **Vote-list length is itself an escalation flag.** The protocol is "≥2 annotators, add more if
   consensus is not reached", so length ≥3 means the first two disagreed. This is a *per-item,
   protocol-generated contestedness label*, independent of the label values.
2. **There is a vote class that does not exist in the aggregate label set: `Counter Narrative`**
   (63 EN + 76 ZH items). It never appears in `Majority_Voting` — aggregation destroys it.
   Top disagreement patterns include `(Counter Narrative, Normal)` 36 EN / 53 ZH,
   `(Counter Narrative, Offensive)` 19 EN / 4 ZH, `(Counter Narrative, Hateful)` 7 EN / 19 ZH.
   **This lands exactly on the field's worst failure mode** — counter-speech / reportage / quotation
   being judged hateful. One annotator saw hate; another saw someone *pushing back against* hate.
3. **`Component` and `Duration` are also recovered** — contributing modality and segment timestamps
   for ~380/409 and ~331/327 items. `research-wiki/ITERATION_LOG.md:255+` recorded these as "gated
   out of the release we hold". They are not; they were only missing from the reduced copy.

**Limitation, stated honestly**: there are **no annotator IDs**. So annotator-*identity* modelling
(LeWiDi-style annotator embeddings, demographic residuals, per-rater reliability) is NOT available.
What IS available is per-item **vote multisets**, soft label distributions, escalation counts, and
the Counter-Narrative dissent class.

### 5.2 The other three datasets: aggregate-only

- **HateMM** — official Zenodo `HateMM_annotation.csv` is exactly
  `video_file_name, label, hate_snippet, target` (431 Hate / 652 Non Hate). Correction to a common
  belief: HateMM used **two** annotators, not three, with an expert tie-break; κ = 0.625 is
  corpus-level. The tie-break events exist in the authors' records but **are not released**.
  The GitHub repo carries no data files at all.
- **HateClipSeg** — repo tracks exactly 5 files. `video_level_annotation.csv`
  (`Video Id, Video-Level Label, Target Victim`) and `segment_level_annotation.csv`
  (`Video Id, Segment-Level Label, Segment Timestamp`), post-adjudication consensus only.
  Aggregate α per task, before → after the discussion round: video-level 0.791 → 0.817,
  **segment-level 0.715 → 0.757 (lowest of the four tasks)**, category 0.840 → 0.899,
  target 0.716 → 0.721. **The discussion stage actively destroyed the disagreement before freezing
  labels**, and the pre-discussion labels are not released.
  *A usable structural proxy does exist locally*: the segment label is a multi-hot 6-vector
  (`0:normal 1:hateful 2:insulting 3:sexual 4:violence 5:harm`) over 11,714 segments —
  hateful 1,259, insulting 1,720, **hateful+insulting co-occurring 663**, and **1,511 segments
  (12.9 %) carry ≥2 offensive labels; 246 of 435 videos are multi-label**. This is
  label *co-occurrence*, NOT coder disagreement — do not conflate them — but it is untouched by
  this project.
- **ImpliHateVid** — DUA-gated; could not be verified. The paper reports **no inter-annotator
  agreement statistic at all** and never states how many annotators labelled each video. Locally we
  hold only a derived split; the 3-way class is recoverable from the ID prefix
  (`EX_`/`IM_`/`NH_`), which is an explicit-vs-implicit **difficulty** axis, not a contested band.
- **HateXplain** (text, same group as HateMM) releases full per-annotator records *with* annotator
  IDs and per-annotator rationales — proof that the convention is per-dataset, not per-group.

### 5.3 What this licenses

**Slot #2 (retrieval over contested labels) is UNBLOCKED with real data**, on MHC-EN + MHC-ZH,
2001 videos, ~21 %/30 % non-unanimous, ~12 %/16 % contested across the exact binary boundary our
protocol uses. Slot #3 (annotation as the VoI cost axis) is *partially* unblocked — we can measure
disagreement reduction, but there are no annotator IDs and no way to buy *new* human votes.

Two cautions you must respect:
- A prior $0 pre-gate in this project tested **graded 3-class soft labels as a label-weighting
  lever** and it came back NEGATIVE (oracle ceiling EN +0.0250 / ZH +0.0256, both under the +0.030
  bar). So "just use soft labels" is already known to be sub-threshold **as an accuracy lever**.
  Any proposal here must do something structurally different from label re-weighting, or must
  claim a beyond-accuracy capability (which the pre-gate did not measure).
- MHC-EN/ZH are the project's **smallest and noisiest** sets (test n=161 / 149; ~1 accuracy point
  ≈ 1.6 videos). Anything whose payoff is a small accuracy delta on these sets is not credible.

**Also audit-only**: a project-generated Gate-C census — 133 unique HateMM-train videos coded by
**Claude agents** (22 double-coded, 5 adjudicated), per-item `confidence` (high 67 / medium 84 /
low 14), `primary_cause`, `required_modalities`, `minimal_sufficient_intervals`, inter-coder
κ = 0.733. **LLM coders, not humans — may NEVER enter a deployable training/inference path.**

## 6. Hard constraints on acceptable candidates

- **Compute**: pilots must be CPU-level or head-level, ≤ 2 h each, at most 3 of them. The GPU is a
  single shared RTX 5090. Raw video may only be processed locally. Full experiments later may use
  the GPU but must stay modest.
- **Claim structure**: because SAGE closed the accuracy race, a viable claim is
  **mechanism + genuine (multi-seed, same-sign) improvement + a quantifiable capability BEYOND
  accuracy** (abstention, calibration, disagreement prediction, auditability, cost curves...).
  Pure leaderboard-climbing is not viable. The improvement need NOT beat SOTA — it must be real.
- **Red lines**: zero test-set contact; decision rules frozen before results are seen; blind design
  (no candidate metric computed during design/implementation); a single submission for the real run.
- **Gate-C / Claude-produced annotations are audit-only** and may never enter a deployable path.
- Do NOT propose anything in §3 (dead) or §2.3 (never-claim) in re-skinned form. If a candidate
  touches one, you must state explicitly what makes it mechanistically different, not just renamed.
- The team has a strong preference for ideas whose result is informative **whichever way it goes**.

---

## 7. Deliverable

Generate **8–12 concrete research ideas**. Spread them across analytic lenses — do not put them all
in one slot. Deliberately include at least a few that are NOT slot #2/#3, since those are gated.
Useful lenses: *method-transfer* (works in domain A, untried in B), *contradiction* (conflicting
findings to resolve), *untested-assumption* (everyone assumes it, nobody tested it),
*scaling-regime*, *diagnostic*, and *reversal* (take one of our own negative results and ask what
its existence implies).

For each idea give exactly:
1. **Title** (short).
2. **Method in plain language** — 2-4 concrete steps: what we actually build / train / run.
   No jargon, no claim IDs. This comes FIRST.
3. **Core hypothesis** — what you expect to find and why.
4. **Minimum viable experiment** — the cheapest thing that produces a positive OR negative signal,
   with an explicit cost estimate in CPU-minutes / head-retrains, and what assets it uses.
5. **Contribution type**: empirical finding / new method / theoretical result / diagnostic.
6. **Risk**: LOW (likely works) / MEDIUM (50-50) / HIGH (speculative).
7. **Claim structure** — the "mechanism + real gain + beyond-accuracy capability" triple, spelled out.
8. **Nearest prior work and the precise difference** — name the paper, say what it owns, say what is
   left. If you cannot name a nearest neighbour, say so honestly rather than inventing one.
9. **Gate dependency** if any (G-full / G-proxy / G-none / none).
10. **The strongest objection a NeurIPS/CVPR reviewer would raise**, and whether it is answerable.

Then finish with:
- **Your own ranking** of all ideas for a top-venue methods submission, with one line of reasoning each.
- **The 2-3 you would actually work on**, and why.
- **An honest statement**: if you believe NONE of these clears a main-conference mechanism bar, say
  so plainly and say what the highest-value fallback is. Do not manufacture optimism. A well-argued
  "all of these fall short, and here is the one honest paper that is actually available" is a more
  valuable answer than twelve upbeat proposals.

Do not invent arXiv IDs, DOIs, venues, or numbers. Where you are unsure of a citation, write
"[unverified]".
