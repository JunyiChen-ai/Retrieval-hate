# Idea-triage bundle — hateful video detection (ARIS idea-creator Phase 4)

You already have the full landscape context from the earlier message in this thread (assets, Gate-0
evidence, the FN-vs-TP forensic table, the literature red lines MultiHateGNN / MM-HSD / MoRE, the
gaps G-A/G-B/G-C/G-D/G-F, the hard constraints, the banned near-duplicates). Do not re-derive it.

Below is the **full annotated candidate set**: 25 raw candidates from six independent generators
(five lens shards + your own brainstorm), mechanically deduplicated into **14 candidates**.
Nothing has been eliminated on quality — that is your job.

## Two facts discovered after the brainstorm (they change some cost estimates)

1. **No OCR cache exists anywhere in the project.** Building one means running PaddleOCR/EasyOCR
   over ~89k HateMM-train frames (744 videos x 120 frames): roughly 1-3 GPU-hours, feasible but
   NOT free, and definitely not inside a <=2 GPU-h pilot. Any candidate whose *pilot* needs OCR is
   therefore more expensive than its author thought, unless it has an OCR-free surrogate.
   Surrogates that do exist: the `required_modalities` flags in the 133-video Gate-C audit, and
   the ASR word counts.
2. **Gold spans are richer than assumed**: `data/gt/HateMM/hate_spans.json` covers 1083 videos with
   `{duration, spans, label}` — i.e. every hateful video has explicit second-level intervals, so
   selection quality is directly measurable on the train split at zero cost.

---

# The 14 candidates

### I1 — Trim-Gain Decomposition (+ paired counterfactual invariance regularizer)
- **Mechanism**: build two views of the SAME frozen segment features — `V_trim` = mean of only the
  gold-span segments, `V_full` = mean of all 30 — train the identical head on each, and decompose
  any gain into a *dilution* term and a *label-alignment* term. The method deliverable is a paired
  (full, trimmed) hard logit-agreement penalty that penalizes a model that only works when trimmed.
- **MVE**: CPU-minutes on `train_subclipK30_...pt` + `hate_spans.json` + the frozen 5 folds.
  delta = OOF macro-F1(V_trim) - OOF macro-F1(V_full), paired bootstrap.
  GO if delta >= +2.0 pts (dilution real, G-A revived); GO-AS-NEGATIVE if delta <= +0.5 pts
  (trimming does not transfer to frozen features; the decomposition + regularizer is the paper);
  ambiguous in between = drop.
- gaps G-A, G-D | risk LOW | effort days | contribution diagnostic
- `prior_work`: 2508.04900 claims the trimming gain; nobody separated dilution from label
  alignment; Gate-0 falsified multi-segment aggregation but never tested single-span trimming.
- `so_what`: one CPU-minutes experiment that revives or kills the whole temporal-selection family,
  and explains why our census and the literature disagree. **This is also the only candidate that
  directly measures the "selection headroom" our sealed Gate-A numbers no longer let us quote.**

### I2 — Within-video normality deviation (is hate an intra-video anomaly at all?)
- **Mechanism**: nonparametric — per-video median prototype over the 30 segment vectors, top-1
  cosine deviation as the "selected" segment. No learned weights at all, so structurally disjoint
  from MultiHateGNN.
- **MVE**: CPU-minutes. Top-1 hit rate against gold spans on 298 hateful train videos vs the
  per-video duration-prior chance rate. GO if >= 0.30 with bootstrap CI above the prior.
- gaps G-A | risk MEDIUM | effort days | contribution empirical
- `prior_work`: SESAD 2607.10298 and the weakly-supervised VAD line import this assumption and
  report gains; nobody has tested the assumption itself on hateful video with gold spans.
- `so_what`: a negative closes an entire transfer family (VAD -> hateful video) with a citable
  structural reason and justifies redirecting to modality coverage.

### I3 — Two-sided hard evidence margin (incriminating minus exculpating, with a null slot)
- **Mechanism**: per-segment incriminating score a_i and exculpating score b_i; video logit =
  max_i a_i - max_j b_j + g(whole-video), straight-through argmax or smoothed-DP top-1 on both
  sides, plus a null/no-locatable-evidence state. Soft attention is structurally incapable of
  subtractive selection.
- **MVE**: ~20 GPU-min. Apply the existing A0 head per segment. METRIC 1: AUC of in-span vs
  out-of-span segments (is evidence locatable at all?) — GO if >= 0.60. METRIC 2: fraction of
  NOT-hateful train videos whose max segment score exceeds the median max of hateful videos —
  GO if >= 0.30 (max-pooling provably cannot separate, so a subtractive branch is necessary),
  NO-GO if < 0.15.
- gaps G-A, G-F | risk MEDIUM | effort weeks | contribution method
- `prior_work`: all MIL/VAD pooling is positive-only; MultiHateGNN is additive soft attention;
  quotation/counter-stance is a named Gate-C category (6 items) with no method anywhere.
- `so_what`: falsifies the field's implicit additive-evidence assumption and gives the mechanism
  that fixes it; the margin decomposes into two named intervals, giving G-F for free.

### I4 — Segment-keyed retrieval-purity closed loop (segment-keyed RGCL)
- **Mechanism**: a hard top-1 segment (smoothed-DP top-k, 2601.21775) is the ONLY retrieval key
  into a bank of *selected* segments; the selector's gradient is the label-purity of the
  neighbourhood its key retrieves. Selection and retrieval supervise each other with no span gold.
  Optional variant: the retrieved neighbours update a dual-prototype decision boundary.
- **MVE**: ~10 CPU-minutes, zero training. For each query video's 30 segments, top-20 cosine
  neighbours over the fold's train-side segments only (same-parent excluded); purity p_j; take
  j* = argmax p_j. METRIC: gold-span hit rate of j* vs uniform-random-segment rate.
  GO if >= 2x random AND >= 0.55 absolute; NO-GO below 1.3x. Secondary 60 s: head on
  [whole-video || segment j*] vs A0, target >= +0.02 OOF macro-F1.
- gaps G-A x G-B | risk MEDIUM | effort weeks | contribution method
- `prior_work`: MoRE retrieves whole videos with a frozen retriever; RGCL is whole-meme; nobody
  keys retrieval on a discretely selected segment or supervises the selector with the retrieval
  outcome. Distinct from our banned failure (that was multi-granularity auto-sub-clip FAISS with
  MIL drifting hard negatives; this is fixed K=30, a bank of selected segments only, purity
  objective, no hard-negative mining).
- `so_what`: converts the field's n=1 retrieval line from whole-video to evidence-keyed retrieval.

### I5 — Unsaid-text retrieval: OCR-minus-ASR residual as the key (gated by a redundancy test)
- **Mechanism**: hard-select the segment with the largest learned OCR-ASR semantic residual and
  use *that residual* as the retrieval key, explicitly indexing on-screen claims that were never
  spoken. Gated by a prior cheap question: **is retrieval already doing OCR's job?**
- **MVE (OCR-free, CPU-minutes)**: build a fold-internal kNN memory over whole-video `img_feats`;
  join to the Gate-C audit flags. (1) kNN@10 neighbour label purity conditioned on
  `on_screen_text` required True vs False (Fisher); (2) recovery rate = fraction of the 22
  OCR-required-no-speech census FNs that the kNN vote calls hateful while A0 calls not-hateful.
  GO-REDUNDANT if recovery >= 0.30 and purity gap >= 0.10 (retrieval substitutes for OCR — the
  redundancy result itself is the paper); GO-COMPLEMENTARY if recovery >= 0.30 and purity gap
  <= 0.02 (then spend ~1-3 GPU-h building the OCR cache); NO-GO if recovery < 0.15.
- gaps G-A x G-B | risk MEDIUM | effort weeks | contribution method + empirical
- `prior_work`: MM-HSD (OCR, no retrieval) and MoRE (retrieval, no OCR) have never been run
  against each other; nobody has asked whether the two mechanisms recover the same videos.
- `so_what`: if redundant, we get MM-HSD-class coverage of the OCR failure population with no OCR
  model at inference.

### I6 — Deterministic silence route + cross-tower text imputation
- **Mechanism**: a label-free deterministic gate (ASR word count from `train_asrK30_...jsonl`)
  hard-partitions videos into speech-bearing and speech-absent; the speech-absent route does not
  get a degenerate text vector, it gets one *imputed* by a ridge map img_feats -> text_feats fitted
  only on speech-bearing videos. Hard partition + regression; no gating network, no attention.
- **MVE**: stage 1 (zero GPU): macro-F1(low-ASR) vs macro-F1(high-ASR) straight from
  `oof_predictions.jsonl`; GO if gap >= 0.08 with n(low-ASR) >= 60. Stage 2 (~5 GPU-min): OOF
  median cosine(Wx, true text_feat) >= 0.40 and >= 3x a row-shuffled control. Only then refit.
- gaps modality-coverage / G-A-adjacent | risk LOW | effort days | contribution method
- `prior_work`: missing-modality imputation (SMIL, ActionMAE) is general multimodal; never applied
  to hateful video, where modality absence is *correlated with the hateful mechanism* (text-on-
  image memetic video). MM-HSD assumes all four modalities always present.
- `so_what`: repairs the exact 30% of failures that carry on-screen text without adding any OCR
  model; gives a stratified error account nobody in this field publishes.

### I7 — Discrete modality-interval bottleneck / budgeted conditional OCR acquisition
- **Mechanism**: a differentiable-discrete knapsack jointly picks one temporal segment AND a
  minimal subset of modalities (visual / ASR / OCR) under a true cost budget, with a null-evidence
  option; equivalently, a hard policy chooses at most 3 of 30 segments on which to *pay for* OCR.
  Cardinality/cost feasibility is exact, not a temperature.
- **MVE**: needs the OCR cache first (1-3 GPU-h). Then compare zero-OCR / 3-segment-OCR / all-30-OCR
  heads in one frozen-fold OOF readout. GO if all-30 OCR gives >= +0.015 macro-F1 AND the
  3-segment policy recovers >= 90% of that gain with <= 10% of the OCR calls.
- gaps G-A | risk MEDIUM | effort 1-3 weeks + OCR build | contribution method + empirical
- `prior_work`: differentiable knapsack (2601.21775) has no vision-language application; MM-HSD is
  always-on tetra-modal; no cost-constrained modality acquisition exists in this field.
- `so_what`: the exact "mechanism + efficiency" claim shape the project wants.

### I8 — Typed-evidence distillation for hard kNN routing
- **Mechanism**: at TRAINING time only, Claude reads frames and emits, per segment, an evidence
  **type** (on-screen-text / visual-symbol / speech) — a type, not a verdict. A 5M head distils
  type + segment selection from frozen CLIP features. At inference the selected segment plus its
  predicted discrete type becomes a **typed key** that hard-partitions the cross-dataset kNN
  memory, so retrieval is restricted to same-type entries. No MLLM at inference.
- **MVE (spends no MLLM budget)**: use the audit's `required_modalities` flags as surrogate type
  labels on 133 videos. (a) linear probe from segment+whole-video feats to the binary
  `on_screen_text required` flag, 5-fold OOF; (b) top-20 neighbour purity under unrestricted vs
  same-predicted-type retrieval. GO if OOF AUROC >= 0.65 AND typed purity exceeds unrestricted by
  >= 0.05. NO-GO if AUROC <= 0.60 — the Claude budget is never spent.
- gaps G-C x G-A x our kNN-memory novelty | risk HIGH | effort weeks | contribution method
- `prior_work`: HFS 2512.11534 distils an MLLM teacher for generic video-QA frame selection with
  Gumbel-Softmax set weights; nobody distils an evidence *type* as a retrieval routing variable.
  Distinct from our banned "MLLM archive as retrieval key" (that fed MLLM text into the key; here
  the MLLM output is a 3-way discrete partition of the index and the key stays a CLIP vector).
- `so_what`: targets the only statistically enriched failure property we have (modality, not
  timing) and removes the topic confound from our headline cross-dataset memory claim.
- **Variants folded in**: an *interventional* teacher ("which evidence removal would reverse the
  judgment?") and a *disagreement-triggered* variant (teacher-vs-dataset disagreement becomes an
  abstention signal rather than a pseudo-label replacement).

### I9 — Contested-label routing (3-way {hateful, not, CONTESTED} + selective risk)
- **Mechanism**: a 3-output head where CONTESTED is not a human label but a train-internal
  label-provenance variable — videos whose label the model's own bootstrap ensemble persistently
  disputes (top-decile predictive variance union small margin), or the flagged set from a
  confident-learning confident-joint. Hard route, not a soft weight; contested videos are deferred.
- **MVE**: GPU-minutes, no new features. Two frozen numbers: (1) AURC for selective macro-F1 must
  improve >= 10% relative to a max-softmax-probability baseline on the same A0 head;
  (2) precision of the CONTESTED route against the adjudicated annotation-noise items in
  `gate_c_audit.jsonl` must be >= 0.50 (chance ~0.20 on the audited 133, ~0.09 over 165 rows).
- gaps G-D | risk MEDIUM | effort days | contribution method
- `prior_work`: selective prediction / SelectiveNet / deep abstaining classifier / confident
  learning are all outside this field; G-D is diagnosed (2508.04900, 2606.28772) with no method;
  MoRE / MM-HSD / MultiHateGNN all train plain binary heads with no abstention.
- `so_what`: first abstention-capable hateful-video classifier; a route-to-human class is the
  operationally correct output for moderation; can beat MM-HSD's 0.874 *on the clean
  subpopulation* without a bare SOTA claim.
- **Variant folded in**: a minority-preserving version that fits modality-conditioned confusion
  matrices so high-specificity minority evidence is not erased (per 2606.28772).

### I10 — Selection-margin certified abstention
- **Mechanism**: a video's label is trusted only if *some single segment* can carry it. The
  top1-minus-top2 segment-evidence margin is a hard binary certificate; uncertified videos leave
  the classification loss entirely (hard gate) and route to an abstention head.
- **MVE**: ~3 GPU-minutes. Train a 5M segment-scoring head 5-fold OOF with video-level labels only
  (video logit = max over segments); margin m_i = top1 - top2. METRIC: AUROC of -m_i for predicting
  the coder noise flag over the 133 audited videos. GO if AUROC >= 0.70 with bootstrap CI lower
  bound > 0.55; NO-GO if <= 0.60.
- gaps G-A x G-D | risk MEDIUM | effort weeks | contribution method
- `prior_work`: abstention is universally driven by *classifier confidence*; keying it on a
  *localization* margin is new. Loss-magnitude noise-robust training (co-teaching) confounds noise
  with hardness, and our hard cases are exactly the ones we must keep.
- `so_what`: a diffuse selection margin is a label-independent signature of "no localizable
  hateful interval exists", which is precisely the annotation-noise mode.

### I11 — Noise-evicted kNN memory (retrieval memory hygiene)
- **Mechanism**: each memory entry carries a hard trust bit from OOF label-prediction disagreement;
  untrusted entries are **evicted** (not down-weighted) and the retrieved trusted set votes under a
  discrete majority-with-abstention rule. Eviction + abstention is a discrete set operation.
- **MVE**: CPU-minutes. suspect = |p - label| > 0.7 (tau frozen before any downstream number). Two
  fold-internal kNN memories, full vs evicted; concatenate the neighbour-vote feature to the A0
  head. METRIC 1: OOF macro-F1(evicted) - macro-F1(full) >= +1.5 pts. METRIC 2: precision of the
  suspect set against the census noise flags >= 0.40. Both required for GO; < +0.5 pts = NO-GO but
  still reportable (retrieval's sign flip is not noise-driven).
- gaps G-B x G-D | risk MEDIUM | effort weeks | contribution method
- `prior_work`: MoRE assumes a clean memory. Our own segment-retrieval negative had EN/ZH sign
  flips that nobody explained.
- `so_what`: explains our tested-negative retrieval result and directly protects the cross-dataset
  updatable kNN memory — the component most exposed to imported label noise.

### I12 — Chance-corrected evidence faithfulness (CCEF) + joint evidence-decision score
- **Mechanism**: an evaluation instrument, two numbers. (a) CCEF: top-1 selected-segment hit rate
  against gold spans, kappa-normalized against the per-video chance rate implied by that video's
  own span coverage — necessary because median span/duration is 0.145, so an uninformative
  selector already "hits" ~15% of the time. (b) Faithfulness Gap: macro-F1 minus evidence-faithful
  macro-F1 (a TP counts only if its argmax segment has tIoU >= 0.5 with a gold span). A stronger
  variant defines a proper joint score -log sum_{s in G_y} p(y,s) over 30 segments plus null and
  optimizes its differentiable surrogate.
- **MVE**: zero GPU, minutes, on `folds/*/segment_scores.jsonl` + `hate_spans.json`. Publish-worthy
  if the bootstrap CI lower bound of CCEF excludes 0 AND the Faithfulness Gap >= 10 points; a CCEF
  indistinguishable from 0 is equally reportable (current selection is chance-level). Ordering
  check: oracle > learned > shifted > random in all 5 folds.
- gaps G-F | risk LOW | effort days | contribution diagnostic + theory
- `prior_work`: tIoU/R@1 and AUC are reported in isolation; ERASER-style faithfulness is
  perturbation-based, NLP/image, not gold-span-based. G-F is empty.
- `so_what`: cheapest possible citable contribution — applies to any method emitting a per-segment
  scalar, including MultiHateGNN's attention weights and MoRE's retrieval scores, so others can
  adopt it without reimplementing anything. Also gives our own selection work an honest yardstick
  now that the Gate-A oracle numbers are sealed.

### I13 — The HateMM annotation-noise ceiling
- **Mechanism**: a fresh *uniform-random* blinded census of 120 HateMM-train videos (the existing
  165 audited rows are stratified on A0 error type and cannot give a population rate), two coders
  plus adjudication, to estimate class-asymmetric flip rates and invert the Natarajan noisy-label
  risk into a macro-F1 **ceiling band**.
- **MVE**: zero GPU; cost is API calls plus adjudication. PROCEED if disagreement rate >= 0.05
  with 95% CI lower bound >= 0.03; ABANDON if CI upper bound < 0.03.
- gaps G-D | risk MEDIUM | effort days | contribution empirical
- `prior_work`: Northcutt "Pervasive Label Errors in Test Sets" did this for ImageNet/CIFAR;
  nothing equivalent exists for any hateful-video benchmark.
- `so_what`: if the noise is real, the MoRE(0.8235) -> MM-HSD(0.874) gap may not be distinguishable
  from annotation noise, which reframes the leaderboard and licenses the abstention framing.
  **Caveat you should weigh**: it audits the TRAIN split only (test contact is forbidden), so it
  can only argue about the ceiling by extrapolation.

### I14 — Knapsack encoder routing under a FLOP budget
- **Mechanism**: a cheap CLIP-only gate makes a hard, exactly-feasible discrete choice of which
  frozen encoders to pay for on each video (CLIP / Qwen2.5-VL-7B / Qwen2.5-VL-32B / Molmo2-8B),
  with knapsack weights equal to measured per-video encode cost. The gate is blind to the expensive
  features.
- **MVE**: head-only, well under 1 GPU-h — all four whole-video caches already exist for train.
  OOF macro-F1 at 25% of all-encoder cost. GO if routed >= (all-encoder - 0.01) AND >=
  (CLIP-only + 0.02).
- gaps efficiency, none of G-A..G-F directly | risk MEDIUM | effort weeks | contribution method
- `prior_work`: MoE / early-exit routing uses soft gates or confidence thresholds; differentiable
  knapsack has no vision-language application; no routing exists in this field.
- `so_what`: if hardness is not predictable from cheap features, this kills every cascade/early-exit
  idea for this task in one cheap shot.

---

# What I need from you

Play devil's advocate on the full set, then rank. Specifically:

1. For **each** of I1-I14: the strongest objection a top-venue reviewer would raise, and the most
   likely failure mode. Be brutal and specific; one or two sentences each is enough.
2. Which `prior_work` notes are **real novelty problems** and which are differentiable? Name any
   candidate you believe is already published somewhere I have not listed.
3. **Rank all 14** for a top-venue submission (CVPR / ACL / EMNLP / WWW / ACM MM class), taking
   seriously that the project needs a **novel mechanism AND strong performance**, not only a
   diagnostic. A pure-diagnostic idea can still rank high only if it is the load-bearing gate for
   a strong method.
4. Which **2-3** would you actually work on, and why? Note explicitly which of them are
   *sequentially dependent* (i.e. one is the gate that decides whether another is worth building).
5. Given a pilot budget of **at most 3 pilots, <=2 GPU-h each, <=8 GPU-h total, HateMM-train only,
   zero test contact**, which 3 MVEs give the most information per GPU-minute? If you would change
   a stated go/no-go threshold because it is too loose or too tight, say the new number.
6. Name any candidate you would **kill outright** and why.

Answer in that order. Be concise per item but do not skip any of the 14.
