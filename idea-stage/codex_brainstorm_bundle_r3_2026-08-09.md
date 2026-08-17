# Round-3 idea generation bundle — hateful video detection, mechanism-level novelty

You are a senior ML researcher brainstorming research ideas for a top-venue **methods** paper.
Read this whole file, then produce the output specified in the last section.

**Reasoning effort: maximum. This is the third round. Two prior rounds produced 27 candidates and
all 27 are dead. Do not produce round-3 variants of round-1/round-2 corpses — the death list below
is exhaustive and binding.**

---

## 0. The task in one paragraph

Project: hateful **video** detection, originally an adaptation of RGCL / RA-HMD (retrieval-guided
contrastive learning for hateful *memes*) to video. Two rounds of idea generation have closed the
retrieval/memory line, the temporal/segment line, the OCR line, the cross-lingual line, and the
annotator-disagreement line. We need **structurally new radii**. The deliverable is a methods paper
for NeurIPS / ICML / ICLR / CVPR / ACL main conference. Nothing else counts.

---

## 1. Hard constraints (non-negotiable)

1. **Methods paper, main conference only.** Not a workshop, not a D&B/Resources track, not a
   findings track. (An evaluation-protocol paper already exists as this project's fallback; it is
   NOT what this round is for.)
2. **Mechanism-level novelty required.** "Apply X to Y", "add modality Z", "swap the encoder",
   "tune the loss weight" are automatic rejects. The claim must be a *mechanism* that did not exist.
3. **Pure accuracy racing is closed.** SAGE (ACL 2026 Main Long, `10.18653/v1/2026.acl-long.817`)
   reports HateMM **0.8710 acc / 0.8628 F1**, statistically indistinguishable from this project's
   own 0.870 / 0.861. MM-HSD (`2508.20546`) reports M-F1 0.874. ⇒ Any viable claim structure is
   **mechanism + a real (honest, replicated) gain + a quantifiable additional capability that is
   not an accuracy metric.** Gains can be modest but must be real and multi-seed.
4. **Data boundary.** Raw video is readable **locally only** (never uploaded to third-party compute).
   Video frames / clips **may** be sent to the Claude API (a standing user-granted exemption unique
   to this project — treat this as an operational degree of freedom competitors may not have, but
   *note*: "we called an API" is not a mechanism).
5. **One prior human-annotation campaign ("Gate-C") is forbidden from any deployment path** — it may
   not be a model input or a training target in a shipped method.
6. Compute: **1× RTX 5090 (32 GB), 16 CPU, 60 GB RAM, no SLURM.** All feature caches already exist
   (see §3). A head-level retrain is ~52 s. GPU budget for the whole project is small; a candidate
   requiring >1 week of 5090 time is infeasible.
7. **Four red lines**: zero test-set contact; decision rules frozen before results are seen;
   blindness (no candidate metric computed during design); a single confirmatory run.

---

## 2. Three framing killers — any new mechanism must route around all three

1. **SAGE** (ACL 2026) — decision-level expert arbitration + instance-level tribunal against feature
   dilution. Closes the accuracy race on HateMM.
2. **HCG-MPB** (ICMR 2026, `10.1145/3805622.3810724`) — replaces per-instance retrieval with an
   LLM-distilled prototype bank and **argues in its motivation that instance-based retrieval is a
   flawed design**. Every RGCL-family video paper must now rebut it.
3. **`2607.23304` Context-Adaptive Inference** — under squared loss + linear head + fixed features,
   explicit parameter adaptation and implicit context routing are *both* kernel ridge regression on
   joint (input, context) features. Combined with **ERM `2602.05152`** (query expansion ≡ key
   expansion), this **formally absorbs** both "our retrieval module is a form of test-time
   adaptation" and "we improved the query/key construction" as independent claims.

---

## 3. Asset inventory (what we actually have on disk, verified 2026-08-09)

### 3.1 Datasets and labels
| dataset | items | notes |
|---|---|---|
| HateMM | 1,083 (train/dev/test split banked) | BitChute; binary hate labels; gold hate **spans** exist |
| MultiHateClip-EN (MHC) | 1,000 (701 train / 100 val / 200 test) | YouTube |
| MultiHateClip-ZH (MHC_zh) | 1,000 (699/101/200) | Bilibili |
| HateClipSeg | 395 cached | segment-level annotations |
| ImpliHateVid | cached features only | ACL 2025 |

### 3.2 The MultiHateClip official release fields (`data/gt/mhc_votes/*.tsv`) — **rich and mostly unused**
Columns: `Video_ID, Majority_Voting, Label, Target_Victim, Component, Duration`.

- **`Label`** = the *raw per-annotator vote list* over {Hateful, Offensive, Normal, Counter Narrative}.
  Votes per item: EN 580×2, 120×3, 1×4. (ZH similar.)
  - EN train+val (801): Normal 529 / Offensive 206 / Hateful 66.
  - ZH train+val (800): Normal 541 / Offensive 156 / Hateful 103.
  - **139 items total (63 EN + 76 ZH) carry at least one `Counter Narrative` vote** that majority
    aggregation destroys.
- **`Target_Victim`** = the annotated **target group** of the attack (e.g. `['LGBTQ']`, `['Couple']`).
  Non-empty for **248 EN + 278 ZH** items. **This field has never been used by any candidate in
  either prior round.** There is also a model-predicted version on disk
  (`data/gt/MHC/target_pred_qwen7b.jsonl`) covering more items.
- **`Component`** = the human-annotated **contributing modality set** per item, from
  {Vision component, Transcript, Audio, Metadata}. EN counts: Transcript 222, Metadata 197,
  Audio 150, Vision 143. ZH: Metadata 294, Vision 214, Transcript 172, Audio 142.
  **Note `Metadata` is the single largest contributor in ZH and #2 in EN** — i.e. title/description
  text, a channel no model in this project currently consumes.
- **`Duration`** = annotated hateful **time spans**; non-empty for 260 EN / 262 ZH items
  (11 EN / 5 ZH have multiple spans).

### 3.3 Feature caches (all precomputed; CPU-only experiments are free)
- CLIP ViT-L/14-336 frame embeddings, mean-pooled and K-segment variants, for all datasets (2.2 GB).
- Qwen2.5-VL 7B/32B embeddings; several LoRA-adapted variants; MPNet transcript embeddings.
- **CLAP audio embeddings** (`larger_clap_general`) for HateMM / MHC / MHC_zh — per-video `.npz`.
- Whisper-large-v3 ASR at K=4 / K=30 / K=60 windows, all datasets.
- **OCR cache** (PaddleOCR, K=30 windows, HateMM 851 + HateClipSeg 395), with per-frame box geometry
  and frame dims.
- **MLLM per-segment harmfulness scores** (Qwen2.5-VL / Qwen3-VL, K=30 and K=4, several prompts and
  fusion variants) for HateMM / MHC / MHC_zh / HateClipSeg.
- LoRA training frames (jpg) for HateMM / MHC / MHC_zh (2.6 GB, 8 frames per video).
- Raw audio + raw video available locally.
- `data/Counterfactual/*/train_twins.jsonl`: 168 EN + 180 ZH generated counterfactual text twins.
- Upload dates for MHC EN/ZH (`MHC_*_upload_dates.jsonl`) and temporal splits.

---

## 4. THE DEATH LIST — 27 dead candidates. Re-skinning any of these is the documented failure mode.

### 4.1 Round 1 (archived report, 2026-08-08) — 14 candidates
| id | idea | how it died |
|---|---|---|
| I1 | Trim-gain decomposition as a method | pilot AMBIGUOUS, null point estimate; regularizer half = known crop-consistency loss |
| I2 | Within-video normality deviation | standard video anomaly detection as a method |
| I3 | Two-sided hard evidence margin | weak-supervision identifiability unresolved; MARS/RAMF own the framing in-domain |
| I4 | Segment-keyed retrieval-purity closed loop | **KILLED by pilot** — selector only reaches chance; within-video purity AUROC 0.511 |
| I5 | Unsaid-text OCR−ASR residual retrieval key | novelty 5/10, then killed: r = o − a is a fixed linear projection of [o‖a] |
| I6 | Silence route + cross-tower text imputation | ridge map img→text adds no information; SMIL / ActionMAE |
| I7 | Budgeted discrete modality acquisition | merged into I1-line, then died with it |
| I8b | Typed hard partition of the kNN memory | **KILLED by pilot** — 0/5 folds reached the purity bar |
| I9 | Contested-label routing / selective risk | SelectiveNet / Deep Abstaining Classifier / confident learning |
| I10 | Selection-margin "certified" abstention | top1−top2 is not a certificate |
| I11 | Noise-evicted kNN memory | Wilson's Edited NN; evicting disputed entries is harmful in minority-value hate |
| I12 | CCEF chance-corrected evidence faithfulness | 5/10; part (b) anticipated by NExT-GQA / EG-VQA. Survives only as an evaluation contribution |
| I13 | HateMM annotation-noise ceiling census | train-only audit cannot establish a test ceiling |
| I14 | Knapsack encoder routing under a FLOP budget | MSDNet / BlockDrop / SkipNet / early-exit |

Round-1 headline (**Pay-for-Evidence**, distilled evidence-type gate driving budgeted modality
acquisition) reached external review and scored **2/10** — ceiling 0.862 < SOTA 0.874. Archived.

### 4.2 Round 2 (current report §2–§6, 2026-08-09) — 13 candidates
| # | idea | jury / pilot verdict |
|---|---|---|
| 1 | Human-Agreement Retrieval (vote-distribution memory + agreement-defined contrastive topology) | **CLOSED.** Frozen gate fired (P-A-v2 KILL, EN −0.0174 / ZH −0.0105 AUROC vs trained baseline); deep novelty 3/10 ABANDON (GenSCL `2206.00384` owns the objective template; Crowd-Calibrator `2408.14141` and `2411.04090` own the distributional read-out; UAKNN `2504.01508` owns kNN-over-label-distributions); external review NeurIPS 2/6 ICML 2/6 ICLR 3/10 ACL 2/5 **Reject** |
| 2 | Dissent-Preserving Prototype Bank | = confidence/entropy-gated cache admission (never-claim 4) |
| 3 | Counter-Narrative Matched Retrieval | **P-B killed the data premise**: near-duplicates are label-*concordant* (5 conflicting pairs observed vs 24.1 expected); 1 genuine minimal pair exists |
| 4 | Duplicate-Conflict Memory | phenomenon absent (P-B) |
| 5 | Provenance-typed OCR fusion (overlay vs scene text) | **P-C**: typed loses to untyped on 3/3 seeds (−0.0020); label-permuted null produced 90% of the "effect". `2211.11350` already does overlay-vs-scene at 0.95 F1 |
| 6 | Sampling-phase robust retrieval | generic consistency training + TTA; partial re-skin of the dead temporal line |
| 7 | Rank–Vote decoupling (continuous class-density score) | similarity-weighted kNN + calibration, standard |
| 8 | Retrieval placebo suite | mandatory supporting science, not a headline |
| 9 | Chance-corrected temporal grounding | metric correction — Eval track, not methods |
| 10 | Component-sufficiency training (leave-one-modality-out supervised by MHC `Component`) | sits between DeHate (component supervision) and UniSafe (modality dropout); `Component` lists are not minimal/causal evidence sets |
| 11 | Contested-item abstention | never-claim 12 (per-sample abstention/escalation routing) |
| 12 | Annotation-escalation prediction | never-claim 12 again; observational records cannot establish annotator-hours saved |
| 13 | Modality-attributed retrieval decomposition | an ablation table is not a contribution |

### 4.3 Directions closed by this project's own frozen-verdict experiments
Multi-segment complementarity (TERA Gate-0 **NO-GO-C**: only 6/73 = 8.2% of baseline misses need
multi-segment evidence, vs a 15% kill line) · single-segment selection · OCR−ASR residual ·
CVoI acquisition · segment-level retrieval keys · visual-purity segment selection ·
type-hard-partitioned memory · streaming/continual memory (W4: per-sample adaptation gain ≤ 0 in
**every** cell of k∈{5,10,20,50,80} × 3 selection strategies × 2 languages) ·
cross-lingual EN-rescues-ZH (Phase-3b measured **EN→ZH −0.138 macro-F1** — the motivation is
backwards).

### 4.4 Inherited negative results that bound what "adding a channel" can buy
- OCR three-stream fusion through a frozen head: **+0.0094**, below the +0.015 bar; dose curve
  concave (3 of 30 windows give 61% of the gain).
- The *same* OCR vector through a learned fusion MLP end-to-end: **−0.0246**, 3/3 seeds (sign flip).
- Late-interaction segment retrieval (MaxSim over 30 segment keys): **−0.043 macro-F1** vs the
  whole-video key. **Dropping the transcript costs −0.029 — the transcript carries most of the
  retrieval advantage.**
- A frozen-CLIP visual segment key is a video-level *style* detector (AUROC 0.782) and a coin flip
  *within* a video (AUROC 0.511).
- **Transferable rule discovered by forensic analysis: a bounded vote/count selection score is
  degenerate by construction (argmax tie-breaking artifacts). Use continuous non-saturating scores.**

### 4.5 The 11-item "never claim novelty for this" list (each has a verified occupant)
1. Growing/replacing a datastore for gradient-free domain adaptation — kNN-LM (ICLR 2020), kNN-MT (ICLR 2021)
2. Inserting the model's own test-time predictions back into memory — AdaNPC (ICML 2023), TDA (CVPR 2024)
3. Age/staleness-scored memory eviction — RoTTA (CVPR 2023), Lu et al. (AIJ 2016)
4. Confidence/entropy-gated cache admission — CRG / ACE / DOTA / SCA (2025)
5. Observing that pseudo-label errors accumulate ("cache noise") — same
6. Wave-style memory insertion + evaluation on later time slices (incl. bucket transfer matrix and
   update-vs-not ablation) — **Mireshghallah et al. EMNLP 2023 (`2209.05706`) does all of it**
7. The phrase "non-parametric continual learning" — HippoRAG 2 (ICML 2025)
8. "Gradient-free vs gradient are fundamentally different mechanisms" — **disproved** by `2305.13034` (EMNLP 2023)
9. Adapting to new classes/tasks by swapping memory contents — Memory-Modular Classification (TMLR)
10. "Text is the better retrieval key" / retrieval-library adversarial poisoning — PoisonedRAG (USENIX Sec 2025), AgentPoison (NeurIPS 2024)
11. Datastore compression/pruning — Efficient kNN-LM (EMNLP 2021), Cluster-based kNN-MT (ACL 2022)

*(and, from the round-2 jury, an operative 12th:)* **12. per-sample abstention / escalation /
routing** — SelectiveNet, Deep Abstaining Classifier, and the 12+ ICLR/ACL-tier 2026H1 cost-aware
acquisition papers.

### 4.6 The three most likely reviewer rejection reasons, from the round-2 external review
1. "This is [known text-domain mechanism] transferred to video with a new dataset."
2. "The gain is within seed noise and the comparator is not the right one."
3. "The additional capability you advertise is an evaluation artifact, not a capability."

### 4.7 Slot occupancy (compressed)
- **Retrieval/memory** — MoRE (WWW 2025), HCG-MPB (ICMR 2026), CRAVE (ICCV 2025), Class-RAG,
  *Now You See the Hate*; key design filled in 2026H1 (LaPR, CIRCLES, ERM). Neighbourhood-consensus
  denoising closed by three papers in eight months.
- **Temporal / localization** — MultiHateLoc, LELA, TANDEM, HateClipSeg baseline, MultiHateGNN.
  Essentially no modelling room.
- **Fusion** — SAGE, HCG-MPB, TIHD/QGC-Net (owns "cross-modal contradiction = evidence"), MM-HSD, UniSafe.
- **Supervision** — LEAF (ACL Findings 2026, explanation distillation), DeHate (ACM MM 2025,
  human contributing-modality labels), IARE, SenBen, IPS, Beyond Hate. Out-of-domain
  learning-with-disagreement is extremely crowded — **all text, zero video**.
- **Inference strategy** — cost-aware acquisition is the single most crowded area (12+ papers in 2026H1).
- **Evaluation protocol** — the one slot confirmed empty for hateful video (but it is not a methods paper).
- **Robustness** — AAAI 2026 finds five SOTA VideoLLMs miss >90% of harmful content, attributed to
  sparse uniform frame sampling (this is also a validity threat to our own 8-frame / K=30 sampling).
  `2606.11198` (Structural Attention Tax): retrieved-content *format* distorts attention independently
  of relevance. `2604.17375` / `2608.04244`: when on-screen text conflicts with the image, MLLMs
  hallucinate toward the overlaid text.

---

## 5. What this round must produce

Generate **12–16 candidates**. **At least 2 in each of families 1–4, and at least 3 in family 5.**

**Family 1 — training signal.** Uses of the official MultiHateClip per-annotator votes as a *method
component* other than (a) contrastive pair topology and (b) abstention — both dead. Uses of the
`Counter Narrative` vote class (139 items) as hard examples. Mechanism-level uses beyond label
smoothing / reweighting. **Also consider `Target_Victim` (248 EN + 278 ZH) and `Component`
(esp. `Metadata`, the largest ZH contributor) — `Target_Victim` has never been touched by any of the
27 dead candidates.**

**Family 2 — distillation / generation.** Narrow forms of a video-LMM-as-teacher that **LEAF does
not occupy** (LEAF owns stage-wise *explanation* distillation into an LMM-free detector). Uses of
the Claude-API-readable frames that are a mechanism rather than an API call. Synthetic hard-example
generation (note: 348 counterfactual text twins already exist on disk).

**Family 3 — structure / representation.** Representation learning that abandons CLIP mean pooling.
**Careful: "segment-level keys / MaxSim" is dead — but "how the representation is constructed" is
not the same slot as "how the retrieval key is constructed".** Audio waveform / prosody: listed in
the original project brief but **never actually attempted**; Phase 1 judged the prior weak — find the
angle on which that prior judgement was wrong (CLAP embeddings are already cached; raw audio is local).

**Family 4 — robustness / deployment.** Adversarial evasion (hateful content actively disguising
itself: voice modification, character obfuscation / leetspeak in overlay text, metaphor and dogwhistle
substitution), cross-platform generalization. **Check occupancy before committing** — SCANNER
(AAAI 2026) already owns source-free TTA across platform/language for hateful video, and the
cross-platform A→B setting on exactly our three datasets.

**Family 5 — free.** ≥3 candidates with no constraint except the death list. Be genuinely inventive.

---

## 6. Required output format

Return **one markdown table row per candidate plus a detail block**, in this schema:

```
### C<N> — <title>
- family: <1-5>
- summary: one sentence
- mechanism: the precise new mechanism in 2-3 sentences — what is computed that was not computed before
- why not a re-skin: name the closest dead candidate / never-claim item / occupant AND state the load-bearing difference
- hypothesis: what you expect and why
- MVE ($0 or near-$0 preferred): the cheapest experiment that would produce a kill-or-go signal,
  stated as a *frozen* decision rule with a numeric threshold, using ONLY the cached assets in §3
  and ONLY train/val splits
- claim structure: mechanism + expected real gain + the non-accuracy capability it buys
- prior_work: what you believe exists nearby (flag anything you are not sure about — do not invent
  arXiv IDs; write "UNVERIFIED" rather than guessing an ID)
- risk: LOW / MEDIUM / HIGH
- effort: days / weeks / months
- single most likely reason it dies
```

Then finish with:

1. **Your own ranking of all candidates for a top-venue methods submission**, with the top 3 named
   and justified.
2. For each of your top 3, **the strongest objection a NeurIPS/ICML/CVPR/ACL reviewer would raise**,
   and whether it is survivable.
3. **An honest global verdict**: if you believe none of your own candidates clears a main-conference
   mechanism bar, say so plainly and say which is closest and by how much. A truthful "all of these
   are 4/10" is worth far more to us than an inflated ranking — two rounds of inflated rankings are
   exactly why we are on round 3.
