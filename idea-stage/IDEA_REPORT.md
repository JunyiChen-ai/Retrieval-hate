# Research Idea Report — hateful video detection, mechanism-level novelty

**Direction**: hateful / harmful video detection; retrieval-guided contrastive pipeline (RGCL port)
as the base system, large deviations allowed. Target = **methods paper, NeurIPS / ICML / ICLR /
CVPR / ACL main conference**, mechanism-level novelty required.
**Generated**: 2026-08-09 (idea-discovery Phase 2, composed mode).
**Extended**: 2026-08-09 with **§5 (Phase-3 deep novelty check)** and **§6 (Phase-4 external
review)** on the Rank-1 mechanism.
**Supersedes**: `idea-stage/IDEA_REPORT_2026-08-08_archived.md` (previous round, archived not deleted).
**Extended again**: 2026-08-09 with **§7 (Round 3 — forced structural pivot)**. Round-3 funnel:
**14 candidates generated → 2 killed by the objective feasibility gate → 3 piloted under rules frozen
in advance → 0 survivors.** Cumulative across three rounds: **41 candidates, 0 live mechanism
candidates.** See §7.8 for the bottom line and §7.6 for the one candidate that is *unfunded* rather
than disproven.

**Funnel**: 13 candidates generated (10 cross-model + 3 executor lens) → 13 through the objective
feasibility gate (none was budget-infeasible) → cross-model triage ranked them → **3 pilots run**
→ recommendations in §4 → **P-A-v2 retest killed the Rank-1 gate (§3.1b)** → **Phase-3 novelty (§5)
retracted its key novelty verdict** → **Phase-4 review (§6) returned Reject / no-prereg**.
**Net result of this round: zero live main-conference mechanism candidates; the evaluation-validity
track is the surviving direction.**

> **Stance: honesty first.** Phase 1 concluded that no ready-to-write, mechanism-level,
> main-conference slot had been found. This round changes that picture in exactly one place — the
> data gate (§0) opened, and it opened wider than the survey assumed. It does not change the
> picture anywhere else. The cross-model jury's own bottom line is reproduced verbatim in §4.
> **Update after §5/§6: the one candidate this round promoted has since been closed. Phase 1's
> conclusion stands unchanged.**

---

## §0 — The data availability gate (P0, $0): **PASS, and the project's own record was wrong**

Phase 1's only mechanism-level open slots (#2 disagreement-aware retrieval, #3 annotation-as-VoI)
were both blocked on one unanswered question: **do our video datasets carry an annotator-level
disagreement signal?** Two independent audits were run — a full local file audit, and a check of
the actual bytes of each upstream release.

### 0.1 Headline: MultiHateClip **does** release raw per-annotator votes

This **overturns the project's own written record.** `research-wiki/PAPER_MASTER_TABLES.md:446`
states that per-annotator votes "do not exist (limitations hard constraint) … neither in-repo nor
in the public release", concluding that the LeWiDi / annotator-distribution family was sealed off
at the data layer. That is **false**. The project had been working from a *reduced derived copy*
(`data/.../annotation(new).json`, HVGuard-derived: `Video_ID, Title, Transcript, Label` only).

The official Social-AI-Studio release
(`{English,Chinese}_data/annotation/{train,valid,test}.tsv`) carries:

```
Video_ID | Majority_Voting | Label | Target_Victim | Component | Duration
3v7239cr4z0 | Hateful | ['Hateful','Counter Narrative','Hateful'] | [...] | ['Vision component','Transcript','Audio','Metadata'] | [(0,46)]
pQbEa24u-HM | Normal  | ['Normal','Offensive','Normal']          | ['Woman'] | ['Transcript','Metadata','Vision component'] | [(2,10)]
```

`Label` is the **list of the individual annotators' raw labels**; `Majority_Voting` is the
aggregate. Verified by download; staged with SHA-256 at `data/gt/mhc_votes/`.

Measured directly (2001 videos, EN 1001 / ZH 1000):

| quantity | MHC-EN | MHC-ZH |
|---|---|---|
| vote-list length 2 / 3 / 4 | 824 / 174 / 3 | 752 / 237 / 11 |
| **non-unanimous items** | **213 (21.3 %)** | **299 (29.9 %)** |
| **items split across our binary boundary** | **123 (12.3 %)** | **162 (16.2 %)** |
| items carrying a `Counter Narrative` vote | 63 | 76 |
| items with `Component` (contributing modality) | 380 | 409 |
| items with `Duration` (segment timestamps) | 331 | 327 |

**Joinability is perfect**: every local id is present upstream (MHC 790/790, MHC_zh 806/806).
Per split, without touching test:

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
   consensus is not reached", so length ≥3 means the first two disagreed — a per-item,
   protocol-generated contestedness label independent of the label values.
2. **There is a vote class absent from the aggregate label set: `Counter Narrative`**
   (63 EN + 76 ZH). It never appears in `Majority_Voting` — **aggregation destroys it.** Top
   disagreement patterns: `(Normal, Offensive)` 73 EN / 96 ZH; `(Hateful, Offensive)` 54 / 84;
   **`(Counter Narrative, Normal)` 36 / 53; `(Counter Narrative, Offensive)` 19 / 4;
   `(Counter Narrative, Hateful)` 7 / 19.** This lands exactly on the field's worst failure mode —
   counter-speech / reportage / quotation judged hateful. One annotator saw hate; another saw
   someone pushing *back against* hate.
3. **`Component` and `Duration` are recovered too** — contributing modality and segment
   timestamps, which `research-wiki/ITERATION_LOG.md:255+` had recorded as gated out of the
   release we hold. They were only missing from the reduced copy.

**Limitation, stated plainly: there are no annotator IDs.** Annotator-*identity* modelling
(LeWiDi-style annotator embeddings, demographic residuals, per-rater reliability) is **not**
available. What is available: per-item vote multisets, soft label distributions, escalation
counts, and the Counter-Narrative dissent class.

### 0.2 The other three datasets: aggregate-only

| dataset | verdict | evidence |
|---|---|---|
| **HateMM** | **aggregate-only** | Official Zenodo CSV is exactly `video_file_name, label, hate_snippet, target` (431 Hate / 652 Non Hate). **Correction: HateMM used two annotators, not three**, with an expert tie-break; κ = 0.625 is corpus-level. Tie-break events exist in the authors' records but are **not released**. The GitHub repo carries no data files at all. |
| **HateClipSeg** | **aggregate-only** | Repo tracks exactly 5 files; labels are post-adjudication consensus. Aggregate α per task, before → after a discussion round: video-level 0.791→0.817, **segment-level 0.715→0.757 (lowest of the four tasks)**, category 0.840→0.899, target 0.716→0.721. **The discussion stage actively destroyed the disagreement before labels were frozen.** |
| **ImpliHateVid** | **unknown (DUA-gated)** | Official repo has code only. The paper reports **no inter-annotator agreement statistic at all** and never states how many annotators labelled each video. Local copy is a derived split; class is recoverable from the id prefix (`EX_`/`IM_`/`NH_`), an explicit-vs-implicit **difficulty** axis, not a contested band. |
| *(reference)* **HateXplain** | releases per-annotator records **with annotator IDs and per-annotator rationales** | Same group as HateMM, two years earlier. Proof the convention is per-dataset, not per-group. |

A structural proxy also exists locally in HateClipSeg: segment labels are multi-hot 6-vectors
(`normal / hateful / insulting / sexual / violence / harm`) over 11,714 segments — hateful 1,259,
insulting 1,720, **hateful+insulting co-occurring 663**, **1,511 segments (12.9 %) carrying ≥2
offensive labels, 246 of 435 videos multi-label**. This is label **co-occurrence, not coder
disagreement** — the two must not be conflated — but it is untouched by this project.

### 0.3 Verdict on Phase 1 slots #2 and #3

- **Slot #2 (retrieval over contested labels): UNBLOCKED with real data.** MHC-EN + MHC-ZH,
  2001 videos, ~21 % / 30 % non-unanimous, ~12 % / 16 % contested across the exact binary boundary
  our protocol uses.
- **Slot #3 (annotation as the VoI cost axis): only half-unblocked, and the surviving half was
  killed in triage.** We can *measure* where annotation budget went, but there is no way to *buy*
  new human votes, so the counterfactual "annotator-hours saved at fixed label quality" cannot be
  established from observational data. The jury ruled this a direct re-skin of per-sample
  escalation routing (never-claim item 12). **Closed.**

Two cautions carried forward: (a) a prior $0 pre-gate in this project tested **graded 3-class soft
labels as a label-weighting lever** and it came back **negative** (oracle ceiling EN +0.0250 /
ZH +0.0256, both under the +0.030 bar) — "just use soft labels" is already known sub-threshold as
an accuracy lever; (b) MHC-EN/ZH are the project's **smallest and noisiest** sets (test n = 161 /
149; ~1 accuracy point ≈ 1.6 videos), so any payoff that is a small accuracy delta there is not
credible on its own.

**Audit-only asset**: the Gate-C census — 133 HateMM-train videos coded by **Claude agents**
(22 double-, 5 triple-coded), per-item confidence, `primary_cause`, `required_modalities`,
`minimal_sufficient_intervals`, inter-coder κ = 0.733. **LLM coders, not humans; may never enter a
deployable training or inference path.**

---

## §1 — Landscape (folded from `idea-stage/phase1_landscape_update.md`)

Full working notes, with per-item verification levels and stated blind spots, remain in that file.
What matters for this round:

### 1.1 Three framing killers — any new mechanism must route around these

1. **SAGE** (ACL 2026 Main Long, `10.18653/v1/2026.acl-long.817`) — decision-level expert
   arbitration + instance-level tribunal against *feature dilution*. HateMM **0.8710 / 0.8628**,
   statistically indistinguishable from this project's own 0.870 / 0.861.
   ⇒ **The HateMM accuracy race is closed.** A pure accuracy claim is no longer publishable.
2. **HCG-MPB** (ICMR 2026, `10.1145/3805622.3810724`) — replaces per-instance retrieval with an
   LLM-distilled prototype bank and **explicitly argues in its motivation that instance-based
   retrieval is a flawed design**. Every RGCL-family hateful-video paper must now rebut it.
   *(Blocked: PDF paywalled — no design decision may rest on its numbers.)*
3. **`2607.23304` Context-Adaptive Inference** — under squared loss + linear head + fixed features,
   explicit parameter adaptation and implicit routing are both kernel ridge regression on joint
   (input, context) features. ⇒ **"our retrieval module is a form of test-time adaptation" is
   formally absorbed.** With **ERM `2602.05152`** (query expansion ≡ key expansion), "we improved
   the query/key construction" also ceases to be an independent claim.

### 1.2 Slot occupancy, compressed

- **Retrieval / memory** — MoRE (WWW 2025), HCG-MPB, CRAVE (ICCV 2025), Class-RAG,
  *Now You See the Hate*. Key design heavily filled in 2026H1 (LaPR, CIRCLES, ERM).
  **Neighbourhood-consensus denoising is closed as a mechanism claim** (AAAI-26, ICML-26, CVPR-26
  in eight months) — **but all three are noisy correspondence**, where a unique correct answer
  exists and neighbourhood geometry recovers it. **Subjective disagreement has no recoverable
  correct answer; their premise does not transfer.** This is the seam §0 reopened.
- **Temporal / localization** — MultiHateLoc, LELA, TANDEM, HateClipSeg baseline, MultiHateGNN.
  Essentially no modelling room left.
- **Fusion** — SAGE, HCG-MPB, **TIHD/QGC-Net** (owns "cross-modal contradiction = evidence"),
  MM-HSD, UniSafe.
- **Supervision** — LEAF, DeHate, IARE, SenBen, IPS, **Beyond Hate** (splits labels into
  incivility/tone vs intolerance/content and halves the FNR−FPR gap). **Out-of-domain
  learning-with-disagreement is extremely crowded — and every single paper is text; zero video,
  zero connecting disagreement to retrieval.**
- **Inference strategy** — cost-aware acquisition / "should we retrieve at all" is the single most
  crowded area (12+ ICLR/ACL-tier papers in 2026H1, all defining cost as compute/latency/tokens).
- **Evaluation protocol** — the one slot confirmed **entirely empty** for hateful video.
  Adjacent occupants are all out-of-domain (NExT-GQA/EG-VQA, NEC, PaSBench-Video).

### 1.3 Standing constraints

- **AAAI 2026**: five SOTA VideoLLMs miss >90 % of harmful content, attributed to sparse uniform
  frame sampling — a **validity threat to our own 8-frame / K=30 uniform sampling**.
- **`2606.11198` Structural Attention Tax**: retrieved-content *format* distorts attention
  independently of relevance.
- **`2604.17375` / `2608.04244`**: when on-screen text conflicts with the image, MLLMs hallucinate
  toward the overlaid text.
- The 15-item "never claim novelty for this" list in `research-wiki/NOVELTY_RECON_2026-08-09.md`
  §5 is in force and was used as a re-skin filter in triage.

### 1.4 Dead directions (re-skinning these is the documented failure mode)

Multi-segment complementarity · single-segment selection · OCR−ASR residual · CVoI acquisition ·
segment-level retrieval keys · visual-purity segment selection · type-hard-partitioned memory ·
streaming/continual memory · cross-lingual EN-rescues-ZH. Each was closed by this project's own
frozen-verdict experiments.

### 1.5 This round's inherited negative results (2026-08-09, all pre-existing)

| result | verdict | what it bounds |
|---|---|---|
| OCR three-stream fusion (`OCR_FUSION_PILOT_RESULT.md`) | **AMBIGUOUS +0.0094**, 3/3 seeds, below the +0.015 bar | mean-pooled untyped OCR into a *linear head*; dose curve concave (3 of 30 windows = 61 % of the gain) |
| Late-interaction segment retrieval (`LI_RETRIEVAL_PILOT_RESULT.md`) | **NO-GO**, −0.043 macro-F1, −0.051 purity | MaxSim over 30 segment keys loses to the whole-video key; **dropping the transcript costs −0.029 — the transcript carries most of the retrieval advantage** |
| A0 ± OCR end-to-end (`A0_OCR_E2E_RESULT.md`) | **NO-GO −0.0246**, 3/3 seeds | the *same* OCR vector that gave +0.0094 through a frozen head gives −0.0246 through the learned fusion MLP — a sign flip. **Anomaly: retrieval ROC moved the other way (+0.019 for the OCR arm)** — ranking improved while the thresholded vote worsened (selection-confounded; recorded as observation) |
| P2 forensic (`P2_FORENSIC_MEMO.md`) | diagnosis | "below chance" was an argmax tie-break artifact; the surviving finding is that a frozen-CLIP visual segment key is a **video-level style detector (AUROC 0.782)** and a **coin flip within a video (AUROC 0.511)**. **Transferable rule: a bounded vote/count selection score is degenerate by construction; use continuous non-saturating scores.** |

---

## §2 — Candidate table (all 13, with the jury's kill reasons)

Generated by cross-model brainstorm (`gpt-5.6-sol`, xhigh reasoning; bundle at
`idea-stage/codex_brainstorm_bundle_2026-08-09.md`) plus three executor-generated lens candidates.
No candidate was eliminated by the executor on quality grounds; **the cross-model jury did all the
narrowing**, per the acceptance-gate protocol.

| # | candidate | lens | gate dep. | jury verdict | why |
|---|---|---|---|---|---|
| 1 | **Human-Agreement Retrieval** — memory entries store the vote *distribution*; contrastive pair topology defined by expected inter-annotator agreement; kNN read-out returns a distribution | untested-assumption | needs votes | **survivable, rank 1** | changes supervision topology *and* the memory's returned object, not merely label weighting. Degenerates into a soft-label re-skin if reduced to weighted pairs without distributional inference |
| 2 | **Dissent-Preserving Prototype Bank** — compress consensus regions into prototypes, retain disputed items as instances | contradiction (vs HCG-MPB) | needs votes | **fatal standalone; keep as ablation of #1** | "uncomfortably close to confidence/entropy-gated cache admission" (never-claim 4); human-label entropy is a different signal but the admission mechanism is the same |
| 3 | **Counter-Narrative Matched Retrieval** — content-matched pairs differing in Counter-Narrative dissent, to separate "hate is present" from "the uploader endorses it" | method-transfer | needs votes | **rank 3, very high risk** | not a contradiction re-skin, but one dissent vote is not adjudicated stance; generic hard-pair training is not itself novel |
| 4 | **Duplicate-Conflict Memory** — near-duplicate clusters as the unit of voting, preserving within-cluster label conflict | untested-assumption | none | **fatal as methods headline unless P-B finds scale; alive as audit** | cluster reweighting is conventional; deciding two uploads are duplicates can erase the very context that sets the label |
| 5 | **Provenance-Typed OCR Fusion** — separate screen-fixed overlay text from scene-attached text before fusion | method-transfer | none | **rank 2** | not an OCR−ASR-residual or TIHD re-skin — provenance is source attribution, not contradiction. Lives or dies on whether box geometry is a valid provenance proxy |
| 6 | **Sampling-Phase Robust Retrieval** — train/infer over shifted uniform-sampling phases | scaling-regime | none | **fatal** | "generic consistency training + TTA"; and **partial re-skin of the dead temporal directions** — averaging phases still bets on multi-window complementarity, which Gate-0 closed |
| 7 | **Rank–Vote Decoupling** — replace the discrete neighbour count with a continuous class-density score, parameter-matched, to resolve the OCR ROC/F1 sign reversal | diagnostic | none | **fatal standalone; correct forensic test** | similarity-weighted kNN and calibration are standard; no mechanism novelty |
| 8 | **Retrieval Placebo Suite** — memory interventions (prior-matched random, similarity-bin permutation, duplicate-flattened) separating relevance from prior/multiplicity/style | diagnostic | none | **fatal as headline; mandatory supporting science** | should merge with #13 |
| 9 | **Chance-Corrected Temporal Grounding** — normalise grounding by each video's attainable random-window hit probability | diagnostic | none | **fatal for methods track; strong for Eval/D&B** | it is a metric correction, not a model — must be presented honestly as such |
| 10 | **Component-Sufficiency Training** — directional leave-one-modality-out constraints supervised by MHC `Component` sets | method-transfer | none | **probably fatal** | sits directly between DeHate (component supervision) and UniSafe (modality dropout); `Component` lists are not minimal or causal evidence sets |
| 11 | **Contested-Item Abstention** — abstain on predicted human contestedness rather than model confidence | untested-assumption | needs votes | **fatal standalone; keep as #1's evaluation** | despite the different *target*, it instantiates never-claim 12 (per-sample abstention/escalation routing). The target distinction is an evaluation-target distinction, not a new mechanism |
| 12 | **Annotation-Escalation Prediction** — predict which items will consume a 3rd annotator; route the annotation budget | reversal | needs votes | **fatal — direct re-skin** | never-claim 12 again; and observational escalation records **cannot** establish annotator-hours saved at fixed quality. If every predicted item still needs the third vote, the policy changes ordering, not cost |
| 13 | **Modality-Attributed Retrieval Decomposition** — decompose retrieval advantage per modality with random-memory and permuted-label controls | reversal on our own negative | none | **fatal standalone; valuable merged with #8** | "an ablation table is not a contribution"; and if it becomes "text is the better retrieval key" it hits never-claim 10 immediately |

**Executor note on candidates 11–13 (mine).** The jury killed all three, two of them as re-skins of
the never-claim list. That judgement is recorded as-is rather than argued with — the project has a
documented failure mode of re-skinning dead ideas, which is exactly what the cross-model gate
exists to catch.

---

## §3 — Pilot results

All three decision rules were frozen in `idea-stage/PILOT_FREEZE_2026-08-09.md` **before any
implementation**. Each pilot was a single submission after synthetic + label-permuted smoke tests.
Zero test-set contact in all three (id/path guards armed and logged). Total cost: CPU-minutes.

| pilot | what it gates | verdict |
|---|---|---|
| **P-A** disagreement retrievability | every vote-based candidate (#1, #2, #3, #11) | **GO**, **overturned by P-A-v2 → KILL** (§3.1b) |
| **P-A-v2** the same gate against a *trained* baseline | same | **KILL** |
| **P-B** near-duplicate & label-conflict census | #4, and the "natural minimal pairs" route into #3 | **DEAD** |
| **P-C** OCR provenance typing | #5 | **AMBIGUOUS** (and see the sharper contrast below) |

### 3.1 P-A — disagreement retrievability: **GO** (superseded — read §3.1b)

Frozen question: is an item's *contestedness* predictable from its retrieval neighbourhood at
all, and does the vote signal add anything over what label geometry already implies? MHC-EN
train+val (629) and MHC-ZH train+val (657), leave-one-out, k=20, **similarity-weighted mean**
(continuous, per the P2 forensic rule against bounded counts). Join 100 %; `Majority_Voting`
agrees with our cached binary labels 1.000 EN / 0.9985 ZH.

| quantity | MHC-EN | MHC-ZH |
|---|---|---|
| base rate T1 (non-unanimous) | 0.211 | 0.317 |
| **E1 — AUROC(neighbourhood score → contestedness)** | **0.6855** | **0.7089** |
| E1 bootstrap 95 % CI | [0.6357, 0.7314] | [0.6671, 0.7487] |
| E2 — AUROC on binary-boundary split (T2) | 0.6556 | 0.6849 |
| **E3 — AUROC of the label-only hardness baseline** | 0.6272 | 0.6180 |
| **Δ = E1 − E3 (paired)** | **+0.0583** | **+0.0909** |
| Δ bootstrap 95 % CI (paired draws) | **[−0.0096, +0.1232]** | **[+0.0416, +0.1400]** |
| Δ fraction of resamples positive | 0.957 | 0.9995 |
| permutation null AUROC (seed 20260909) | 0.4933 | 0.5413 |

**Verdict against the frozen rule: GO.** Both languages clear `AUROC ≥ 0.60` with bootstrap
LB > 0.55, and both clear `Δ ≥ +0.03` on the point estimate.

**Honest qualification the rule did not require.** The frozen rule was written on point estimates.
On the *interval*, the discriminator is solid in ZH (paired Δ CI excludes zero, 99.95 % of
resamples positive) but **suggestive-only in EN (paired Δ CI includes zero, 95.7 % positive)**.
So: contestedness is genuinely retrievable in both languages, and in ZH it demonstrably carries
information beyond label hardness; in EN that increment is positive but not separated from zero at
n = 629. The ZH null (0.5413) also sits near the upper edge of the accepted [0.45, 0.55] band.

**Three further caveats from the implementer, which bear directly on §4.**
- **The hardness baseline may be unfairly weak.** `h` is a 20-NN weighted label fraction. A system
  with a *trained classifier* could produce a much better-calibrated hardness estimate, and against
  that baseline Δ could shrink or vanish. The pilot rules out only the crude label-geometry
  baseline it specified — which is exactly the failure mode the freeze's own "Reading" section
  named.
- **T1 is measured at very low annotator resolution.** 526/629 EN and 485/657 ZH items have only
  two raw votes, so for ~80 % of EN and ~74 % of ZH, "non-unanimous" means simply "the two
  annotators differed". T1 is therefore partly a proxy for single-annotator noise rather than
  genuine item ambiguity.
- **Retrievability ≠ usability.** AUROC ≈ 0.70 against a 21 %/32 % base rate is a weak ranker in
  absolute terms, and leave-one-out over a pooled train+val set overstates neighbourhood density
  relative to deployment.

**What this licenses.** The necessary condition for the disagreement family holds: a retrieval
neighbourhood does carry disagreement structure, and it is not merely re-encoding "this item is
hard" *as measured by label geometry*. **It does not license any accuracy claim** — P-A measures
retrievability of contestedness, nothing else.

### 3.1b P-A-v2 — the same gate against a *trained* baseline: **KILL**

P-A's own result section named the risk: `E3` was a 20-NN weighted label fraction with no trained
parameters, and "against a trained classifier Δ could shrink or vanish". P-A-v2 ran that test.
Rules frozen in `idea-stage/PILOT_FREEZE_2026-08-09.md` §"P-A-v2" **before implementation**;
synthetic + label-permuted smokes first; single submission; 87 s CPU; zero test-set contact
(same armed path guard). Targets, features, similarity, vote parsing and the AUROC estimator are
imported unchanged from the P-A code, so T1/T2 are byte-identical (base rates reproduce exactly:
0.2114 EN / 0.3166 ZH).

**Protocol.** Stratified 5-fold on T1, 3 fold seeds (20260910/11/12), all arms on byte-identical
folds. The neighbour pool is **fold-restricted** — a held-out item's neighbours come only from the
training folds — so no arm ever sees a held-out item's votes. Every arm is one frozen logistic
regression recipe (L2, `class_weight='balanced'`, inverse strength chosen per outer fold by inner
5-fold CV over `{0.003 … 10}`), so no arm is advantaged.

| arm | input |
|---|---|
| **B1** | trained baseline on the frozen 1792-d CLIP key `[l2(img) ‖ l2(txt)]` |
| **B2** | B1 + the model's own OOF label uncertainty `[entropy, margin]` (labels only, no votes) |
| **C** | the candidate: 8 retrieval-neighbourhood **vote** features, trained |
| **D** | `B1 ‖ C` features |
| *C0* | P-A's raw scalar `s` used directly as a score, under v2 folds (descriptive) |

**T1 (primary, gating), OOF AUROC averaged over 3 seeds.**

| quantity | MHC-EN | MHC-ZH |
|---|---|---|
| base rate T1 | 0.2114 | 0.3166 |
| **B1 — trained baseline, frozen features** | **0.6838** | **0.7150** |
| B2 — B1 + model uncertainty | 0.6842 | 0.7149 |
| **C — retrieval-neighbourhood vote signal** | **0.6664** | **0.7045** |
| D — B1 ‖ C | 0.6836 | 0.7152 |
| C0 — P-A's scalar under v2 folds | 0.6514 | 0.7056 |
| **C − B1 (paired)** | **−0.0174** | **−0.0105** |
| C − B1 bootstrap 95 % CI / frac. positive | [−0.0567, +0.0205] / 0.184 | [−0.0429, +0.0215] / 0.269 |
| D − B1 | −0.0001 | +0.0002 |
| D − B1 bootstrap 95 % CI | [−0.0011, +0.0007] | [−0.0005, +0.0009] |
| B2 − B1 | +0.0005 | −0.0001 |
| permutation null, arm C (seed 20260909) | 0.4751 | 0.5139 |

Per-seed spread is small (EN C: 0.6714/0.6593/0.6685; ZH C: 0.6938/0.7204/0.6994) and both nulls
sit inside the frozen [0.45, 0.55] band — the ZH null (0.5139) is in fact better behaved than
P-A's (0.5413).

**Verdict against the frozen rule: KILL.** The gating clause is
`AUROC(C) ≥ AUROC(B1)` and it fails in **both** languages. The looser literal reading in the
commissioning brief ("condition holds in at least one language") also comes out negative
(`literal_at_least_one_flag = false`), so the two readings agree and there is nothing to
re-adjudicate.

**What actually happened to P-A's Δ.** P-A reported Δ = +0.058 EN / +0.091 ZH of the neighbourhood
score over its label-only baseline `E3` (0.627 / 0.618). A trained logistic regression on the
*same frozen features* reaches **0.684 / 0.715** — i.e. the trained baseline gains +0.057 EN /
+0.097 ZH over `E3`, which is, to within noise, **exactly the increment P-A credited to the
votes**. P-A's Δ was a measurement of how weak `E3` was, not of what the neighbourhood knows.

**Secondary target T2 (binary boundary split), non-gating.** Same direction, no rescue:
EN B1 0.6720 vs C 0.6528 (C−B1 = −0.0192); ZH B1 0.6997 vs C 0.6919 (−0.0078).

**Three honest qualifications, stated with the result.**
- **The D arm is diluted, and its near-zero increment should not be read as a complementarity
  test.** Inner CV selected the strongest available penalty (`C = 0.003`, the **grid edge**) on
  15/15 outer folds for B1, B2 and D. Under one shared L2 penalty across 1800 standardised
  dimensions, 8 extra features cannot move the decision function — hence `D ≈ B1` to four decimal
  places and an implausibly tight CI. A properly powered complementarity test needs block-wise
  penalties or a residualised design. That is a **different pre-registration**, not a re-run of
  this one; per the frozen red lines this submission is not re-tuned after seeing results.
  **The KILL does not rest on the D arm** — the `C ≥ B1` clause fails on its own.
- **Grid-edge saturation** also means B1 is, if anything, *under*-regularised relative to what the
  inner CV wanted; a wider grid could make the baseline stronger still, which can only deepen the
  KILL.
- **C is not at chance.** AUROC chance is **0.5** (not the class prevalence — an earlier draft of
  this bullet compared 0.666 / 0.705 to the base rates 0.21 / 0.32, which is metrically wrong;
  corrected 2026-08-09 after external review §6.7 item 10). The correct evidence is the permutation
  nulls, 0.4751 EN / 0.5139 ZH, both inside the frozen [0.45, 0.55] band; a single null value
  without its full distribution is weak support for calling the null "well-behaved". The finding is
  not "neighbourhoods know nothing about contestedness" — it is "**they know nothing a trained model
  on the same frozen features does not already know**". P-A's inherited caveat about T1's low
  annotator resolution (2 votes for ~80 % EN / ~74 % ZH items) applies unchanged here.
- **The `C ≥ B1` clause tests replacement, not conditional information** (external review §6.7
  item 9). It asks whether 8 neighbourhood vote features can *stand in for* the 1792-d
  representation, not whether they add anything *given* it — and the `D` arm that would have tested
  that was regularised away. The frozen gate legitimately fires; the stronger reading "the
  neighbourhood contains no additional information" is **not** established by this run.

**What this closes.** The necessary condition that P-A was written to test — that the vote-based
family has a *mechanism* the feature geometry does not already supply — now fails against a fair
baseline. Human-Agreement Retrieval (§4 Rank 1) loses the empirical support it was ranked on.
Raw numbers: `idea-stage/pilot_a_v2.json`; code `idea-stage/pilot_a_v2_strong_baseline.py`;
log `logging/runs/pa_v2_retest/run.log`.

### 3.2 P-B — near-duplicate & label-conflict census: **DEAD** (with a useful by-product)

Train splits only, all datasets with cached embeddings. Frozen gate on the conservative count
(`c_img ≥ 0.90` **and** transcript Jaccard ≥ 0.5).

| threshold | pairs (N1) | within / cross dataset | conflicting (N2) | Jaccard ≥ 0.5 | **conservative conflicting** |
|---|---|---|---|---|---|
| `c_img ≥ 0.90` | 645 | 630 / 15 | 53 | 49 | **5** |
| `c_img ≥ 0.95` | 195 | 191 / 4 | 39 | 34 | **5** |

**Verdict against the frozen rule: DEAD** (conservative count 5 < 10).

Pool = 3,155 train items (HateMM 744, MHC 549, MHC_zh 579, ImpliHateVid 1,283; **HateClipSeg
excluded — only `test_seen` caches exist locally, so it has no train assets**), 4,975,435
unordered pairs. **N3 = 0**: no surviving conflicting MHC pair has a Counter Narrative vote on
either side.

**The by-product is the interesting part.** Under label permutation (200 permutations, seed
20260909) the expected conservative conflicting count is **24.1, 95 % range [15, 30]**. The
observed value is **5** — far *below* chance. Near-duplicate videos in these corpora are
**strongly label-concordant**: reposts overwhelmingly carry the same label as their source.

**Post-hoc forensic (run after the verdict was fixed; it only reduces the count).** Four of the
five surviving pairs are **artifacts, not reposts**: HateMM contains an 11-video group with
*byte-identical* mean-pooled image features spanning both labels, all carrying the placeholder
transcript `"🎼  🎼  Yeah."` — i.e. failed video decode / failed ASR. The single genuine minimal
pair is MHC `0q1PET_IDGc` (label 0) vs `KV49pENhk4c` (label 1). **Verified genuine count = 1.**
The frozen number stays 5; the verdict is DEAD either way.

Three consequences:
1. **The "naturally occurring minimal pair" route is closed.** There is no free corpus of
   same-footage / opposite-label pairs to mine — there is exactly one pair. This kills the
   near-duplicate path into candidate #3 and removes the empirical basis for candidate #4.
2. **The duplication worry is bounded and negligible.** 49 verified near-duplicate pairs across
   4.98 M pairs, voting concordantly. A caveat to state, not a contamination finding.
   **The train↔test leakage audit was deliberately not run** — it requires test contact and needs
   its own authorisation.
3. **A genuine data-quality flag for the project, found incidentally.** Groups of **bit-identical
   mean-pooled CLIP image features** exist in the train caches: HateMM 3 groups / 16 items
   (largest group = **11 videos sharing one vector, spanning both labels**:
   `hate_video_{76,109,127,298,308}`, `non_hate_video_{25,90,110,308,395,470}`),
   MHC_zh 1 group / 2 items, ImpliHateVid 4 groups / 8 items. The HateMM ones all carry the same
   placeholder transcript `"🎼  🎼  Yeah."` — failed decode / failed ASR producing a degenerate
   feature vector. **These items are silently present in every experiment this project has run on
   those splits**, and should be audited independently of any idea in this report.

### 3.3 P-C — OCR provenance typing: **AMBIGUOUS**, and the sharper contrast is negative

HateMM train only (744), reusing the OCR fusion pilot's folds, head, seeds and filter.
Unsupervised typing rule (box-centre stability + persistence), no label access.

**Harness validation — exact reproduction.** Arm 0 per-seed `[0.8077, 0.8143, 0.8092]` and
arm 1 per-seed `[0.8155, 0.8205, 0.8235]` are **identical, seed for seed**, to the prior OCR
fusion pilot's baseline and OCR-30 arms; `arm1 − arm0 = +0.0094`, reproducing the prior headline
exactly. The two pilots are provably measuring the same object.

Provenance coverage (descriptive, label-free) — **not underpowered**: of 744 videos, 401 have
overlay text, 448 have scene text, 255 have both, 150 have neither; 146 overlay-only, 193
scene-only. Detections split 42,105 overlay / 47,812 scene (46.8 % overlay); characters 40.9 %
overlay. 1,751 of 27,965 tracks are overlay-like.

| arm | input | seed-mean OOF macro-F1 |
|---|---|---|
| 0 | baseline, 1792-d | 0.8104 |
| 1 | + untyped OCR, 2560-d | 0.8198 |
| 1c | + untyped OCR duplicated (**parameter-matched control**), 3328-d | 0.8134 |
| 2 | + **typed** overlay ‖ scene, 3328-d | 0.8178 |

| contrast | per-seed | mean | positive seeds |
|---|---|---|---|
| **arm2 − arm1c (gating)** | +0.0013, +0.0024, +0.0095 | **+0.0044** | 3/3 |
| **arm2 − arm1 (typing vs untyped)** | −0.0017, −0.0013, −0.0030 | **−0.0020** | **0/3** |
| arm1c − arm1 (pure dimensionality penalty) | −0.0030, −0.0037, −0.0126 | −0.0064 | 0/3 |
| arm1 − arm0 (reproduction) | +0.0077, +0.0061, +0.0144 | +0.0094 | 3/3 |

**Verdict against the frozen rule: AMBIGUOUS** (`+0.003 … +0.010`).

**Honest reading, which is substantially worse than the verdict label suggests.** Three things,
in increasing order of severity:

1. The gating contrast is positive only because arm 1c pays a **−0.0064 dimensionality penalty**
   that arm 2 partly recovers. Against the contrast that actually matters — **typed versus plain
   untyped OCR — typing is negative on 3/3 seeds (−0.0020)**.
2. **My pre-registered control was flawed.** Arm 1c duplicates the OCR block, and under weight
   decay a duplicated block is *not* a neutral capacity control — it is actively handicapped. So
   the gating contrast was measured against a comparator that was too weak. This is a design error
   in the freeze, not an implementation error, and it is mine.
3. **The decisive caveat: the label-permuted null run produced `arm2 − arm1c = −0.0039`, whose
   magnitude is 90 % of the real run's +0.0044.** The gating effect is the same size as what the
   harness generates from pure noise. The honest conclusion is **not "weak positive" but "no
   evidence of an effect"**.

Descriptively, the presence of overlay text and of scene text are each essentially uninformative
about the video label on their own (AUROC 0.493 and 0.457).

**Two limitations that cut in the candidate's favour and must be stated too.**
- **Dilution**: 150/744 videos have no OCR at all, 343 have an all-zero overlay block and 296 an
  all-zero scene block, so **only 255 videos (34.3 %) exercise both typed blocks simultaneously**.
  Even a real typing effect would be attenuated by averaging over a population where two thirds of
  items exercise at most one block.
- **The typing rule was never validated against ground truth.** Nobody checked whether the tracks
  labelled "overlay" really are burned-in captions/watermarks. It is a geometric heuristic frozen
  from a description, and **a wrong split would look exactly like a null result.** The rule also
  degenerates on low-text videos (a single-window track trivially satisfies persistence with
  centre-std 0), though the affected text mass is tiny (0.24 % of overlay detections).

So the honest scope of this negative is: *this* fusion, with *this* unvalidated typing rule, on
*this* population, does not pay. It is not a refutation of the provenance concept.

**What this licenses.** Slot #7 does not open on this evidence. The provenance categories are
extractable and well-populated — the typing rule works as a *measurement*, and the OCR cache
schema does carry the geometry needed (4-point polygons in absolute source pixels; true frame
dimensions read from video headers for 744/744) — but the split does not pay through this fusion,
and the effect it did produce is indistinguishable from the harness's own noise. A future attempt
would need a different use of the typing than a second mean-pooled block, a genuine
parameter-matched control (not a duplicated block), and would have to beat plain untyped OCR.

### 3.4 Deviations and process notes

All deviations were logged by the implementers in the per-pilot result files rather than hidden.
The ones that affect how the numbers should be read:

- **P-A**: the freeze specified only the null seed, so the bootstrap seed (20260908) was fixed in
  code pre-run. The permutation-null acceptance window `[0.45, 0.55]` is **weak** — at this n the
  null SD is ~0.02–0.03, so the window is only about ±1.7 SE, and ZH landed at 0.5413, near the
  ceiling.
- **P-A**: `logging/runs/pilot_a/run.pid` was not written (the launcher backgrounded the whole
  setup chain, so `echo $!` raced). The run completed once, exit 0, log intact; a `NOTE.md` records it.
- **P-B**: HateClipSeg excluded — no train-split assets exist locally, only `test_seen`. Not
  fabricated, just absent. The Jaccard tokenizer was unspecified in the freeze and was fixed
  pre-run (lowercase alphanumeric plus per-character CJK/Kana/Hangul; both-empty → 0).
- **P-C**: the freeze left several typing-rule details open; they were fixed in code before any
  number was computed. The null-run acceptance bars were also declared in the script before the
  permuted run. Arms 0 and 1 reuse the comparator's seed scoping so the reproduction is bit-exact;
  the new arms get new seed tags, so this cannot touch the gating contrast.

**Lesson for the next freeze**: two of the three freezes under-specified something the
implementation had to decide (bootstrap seed, tokenizer, typing-rule thresholds), and one
specified a control (a duplicated feature block) that is not neutral under weight decay. Freezes
should name the control's *mechanism of neutrality*, not just its dimensionality.

---

## §4 — Ranking and recommendations

**Post-pilot funnel**: 13 generated → 13 feasible → cross-model jury kept 3 as live methods bets
(#1, #5, #3) → pilots + novelty check moved them to **1 recommended, 1 conditional (as a component
of the first), 1 closed** → plus one honest fallback that is stronger than any single mechanism
candidate here.

### Rank 1 — **Human-Agreement Retrieval** (~~RECOMMENDED~~ → **PERMANENTLY CLOSED**; see §3.1b, §5, §6, §6.9)

> **Final disposition 2026-08-09: closed as a main-conference mechanism candidate.** Three
> independent instruments agree: the frozen pre-registered gate fired (**P-A-v2 KILL**, §3.1b), the
> Phase-3 deep novelty check **retracted the leg-(ii) "OPEN" verdict and found direct occupants for
> leg (iii)** (§5, cross-model novelty score **3/10, ABANDON**), and the Phase-4 external review
> scored it **NeurIPS 2/6 · ICML 2/6 · ICLR 3/10 · ACL 2/5, Reject**, answering **NO** to the
> pre-registration question (§6). Everything below is the record of what was proposed and why.

> **Standing overturned 2026-08-09 by P-A-v2 (§3.1b).** This ranking was justified by P-A's GO,
> whose Δ has since been shown to be an artefact of an untrained baseline: against a trained
> logistic regression on the same frozen features, the retrieval-neighbourhood vote signal is
> **behind** the baseline in both languages (EN −0.0174, ZH −0.0105 AUROC). The gate's frozen
> KILL clause fired. Everything below is retained as the record of what was proposed and why,
> **not** as a live recommendation; reviving it requires a new mechanism argument that does not
> depend on neighbourhood-carried contestedness, plus a fresh pre-registration.

**Method (what we actually build).**
1. Convert each training item's raw annotator votes into a smoothed distribution over
   {Hateful, Offensive, Normal, Counter Narrative} — the votes are on disk (§0) and cost nothing.
2. Train the existing ~5 M-param head so that two items count as "similar" in proportion to the
   probability that two independently sampled annotators would label them the same way, keeping
   the ordinary binary loss alongside it. This changes **which pairs are positives** in the
   contrastive objective — currently decided by majority-label identity.
3. Store the vote distribution (not the majority label) as each memory entry's payload, and make
   the kNN read-out return an averaged **distribution**.
4. Sum Hateful+Offensive for the binary decision; keep contestedness and Counter-Narrative mass as
   separate outputs.

**Mechanism in one sentence.** Expected inter-annotator agreement, rather than majority-label
identity, defines both the contrastive geometry and the object the memory returns.

**Novelty boundary.** Not "we are first to use annotator disagreement" (the text side is packed —
LeWiDi, DiADEM, EDO, soft-label training, Socio-Contrastive, RGPO, STABLEVAL, NEC; all must be
cited). Not "soft labels help" (already known, and this project's own pre-gate found soft labels
sub-threshold **as a label-weighting lever**: oracle ceiling EN +0.0250 / ZH +0.0256). The claim is
narrower: **on video, and connecting disagreement to a retrieval memory** — a combination for which
two independent sweeps returned zero occupants.

**Novelty check verdict (Phase-2 targeted sweep, 2026-08-09): OPEN for the composite, ADJACENT on
two of its three legs.** ~45 arXiv API sweeps plus OpenAlex; twelve distinct query formulations for
"annotator disagreement × retrieval/memory/datastore/kNN" returned **essentially nothing**,
replicating Phase 1's zero. Decomposing the candidate:

| leg | Phase-2 status | nearest occupant |
|---|---|---|
| (i) memory entries carry vote distributions | **ADJACENT — pre-empted in weak form** | **UAKNN `2504.01508`** [A] — kNN inside label-distribution learning with uncertainty-aware neighbour weighting. Literally "kNN + label distributions". |
| (ii) **contrastive pair topology defined by expected inter-annotator agreement** | **OPEN — zero papers found, in any field** | none |
| (iii) distributional kNN read-out | **ADJACENT — pre-empted in weak form** | **Opt-ICL `2510.07105`** [A], LeWiDi-2025 overall winner — retrieves *rater examples* in-context; ablation says they are the single most important component. |

> ⚠️ **This table is superseded. The Phase-3 deep sweep (§5) retracts the leg-(ii) "OPEN" verdict
> and finds direct occupants for leg (iii) that Phase 2 missed.** Corrected statuses:
> **(i) ADJACENT (two occupants, not one) · (ii) ADJACENT — the mechanism template is owned by
> GenSCL `2206.00384` · (iii) NOT OPEN — occupied in the same domain by Crowd-Calibrator
> `2408.14141` and `2411.04090`.** The sentence below ("leg (ii) is the actual novelty") no longer
> holds as written: what survives is a *kernel choice* inside a published objective template.

**Leg (ii) is the actual novelty. Legs (i) and (iii) are not, and the paper must not lean on them.**
*(Retained as the Phase-2 record; retracted by §5.)*

**Nearest neighbours and the precise difference.**
- **UAKNN `2504.01508`** — the single most dangerous citation. Difference: its distributions are
  generic LDL targets, not annotator votes; no subjectivity framing, no contrastive objective,
  no moderation domain.
- **Opt-ICL `2510.07105`** and **DeMeVa `2509.09524`** [A] (compares example-sampling strategies
  for ICL over perspectivist annotations, aggregating per-annotator predictions into soft labels)
  — structurally "select neighbours → read out a vote distribution", but selection is prompt-time
  heuristic sampling in a meta-trained LLM: no learned retriever, no embedding space, no
  agreement-defined pair structure.
- **QuMAB `2507.17653`** [A] and **LPI-RIT `2508.08163`** [A] own annotator-*behaviour* modelling —
  parametric, and both need annotator identity, which we do not have (§0.1).
- *RGCL* (ACL 2024, `2311.08110`) owns retrieval-guided contrastive learning with **label-identity**
  positives for hateful memes.
- *IN2R* (ICML 2026), *ConeSep* (CVPR 2026), AAAI-26 `2512.24064` own neighbourhood-consensus
  denoising — but all three are **noisy correspondence**, where a unique correct answer exists and
  geometry recovers it. Subjective disagreement has no recoverable answer, so their premise does
  not transfer. This is the seam.
- *ICLR-26 `2601.22570`* owns retrieval-based selective prediction — different target.
- **Do not confuse with `2508.04900`** ("Revealing Temporal Label Noise in Multimodal Hateful Video
  Classification") — that is *temporal* label noise from video-level annotation, **not** annotator
  variation.

**Confirmed clear**: no hateful/harmful **video** work uses annotator vote distributions at all.
HateMM, MultiHateClip, HateClipSeg and ImpliHateVid all aggregate to hard labels; the
disagreement-analysis literature in hate is text-only.

**The citations a reviewer will reach for to reject it**: `2504.01508` ("kNN over label
distributions exists"), `2510.07105` ("retrieving rater examples already wins the shared task"),
`2311.08110` ("retrieval-guided contrastive learning in hateful content is done").

**Claim structure** (mechanism + real gain + beyond-accuracy capability):
agreement-defined retrieval geometry with distribution-valued memory → reproducible multi-seed
binary gains in **both** EN and ZH → **plus** prediction of vote distributions, item contestedness,
and Counter-Narrative dissent, none of which is an accuracy metric.

**Pilot evidence.** P-A **GO**: contestedness is retrievable (AUROC 0.686 EN / 0.709 ZH) and beats
the label-only hardness baseline (Δ +0.058 / +0.091). Qualification: the paired increment's CI
excludes zero in ZH but **includes zero in EN**.

**Single most likely reason it dies** (jury's words, and I agree): after controlling for ordinary
label hardness, 2–3 votes per item may not contain enough predictable structure to improve binary
decisions on two small, noisy datasets. Three signals already point that way — the EN increment's
CI includes zero; ~80 % of EN items carry only two votes, so "contested" often just means "two
annotators differed"; and P-A's hardness baseline was a crude 20-NN label fraction, so a properly
trained hardness estimator could absorb the increment entirely. **The very first thing the
follow-up must do is re-run the Δ against a trained-classifier hardness baseline.** If Δ vanishes
there, close the direction.

**Reviewer's strongest objection.** "This is soft-label supervised contrastive learning transferred
from text, with distributions estimated from 2–3 annotations." Survivable **only** with an ablation
that separates three things — soft BCE, distributional *pair topology*, and distribution-valued
*read-out* — and only if binary gains replicate in both languages. If it collapses to weighted
pairs without distributional inference, it is a soft-label re-skin and should be abandoned.

**Next step.** Pre-register the three-way ablation on MHC-EN + MHC-ZH train/val, multi-seed,
with the soft-BCE arm as the control that the pre-gate already bounded. Freeze rules first.

### Rank 2 — **Counter-Narrative dissent as an auxiliary target** (CONDITIONAL, restated)

The original candidate #3 planned to mine content-matched opposite-stance pairs. **P-B kills that
route**: near-duplicates are strongly label-*concordant* (5 conflicting pairs observed vs 24.1
expected under permutation), so there is no free minimal-pair corpus. What survives is the direct
signal: **139 items (63 EN + 76 ZH) carry a `Counter Narrative` vote that majority aggregation
destroys**, concentrated on the field's worst failure mode (counter-speech / reportage judged
hateful).

Restated method: predict Counter-Narrative *dissent probability* as an auxiliary head alongside the
binary decision, supervised by the raw votes. **This is a component of Rank 1, not a separate
paper** — it is where Rank 1's beyond-accuracy capability becomes concrete and socially meaningful.

**Boundary.** Must be described as *probability of annotator dissent*, never as objective stance
truth — one dissenting vote is not an adjudicated discourse-role label. Do not reframe it as
"cross-modal contradiction is evidence" (TIHD, ICMR 2026, owns that in-domain).

**Novelty check verdict: ADJACENT — the problem is owned in text, the video setting is startlingly
empty.** `"counter speech" AND (video OR visual OR meme)` on arXiv returns **0**; OpenAlex returns
7 works, all communications/humanities, **no detection models**. Counterspeech itself is crowded
but almost entirely text + *generation*; counterspeech **detection** is a small, old, Twitter-scale
literature.
- **`2404.01651`** [A] (NAACL 2024), *"NLP Systems That Can't Tell Use from Mention Censor
  Counterspeech"*, **owns the exact problem statement** — counterspeech mentions harmful language
  without using it, LMs conflate the two, and this censors counterspeech. **This is the citation
  that rejects a poorly-framed version of this idea.** Difference: text-only, prompt-level
  mitigation, no contrastive training, no paired data, no video.
- **FC-CONAN `2601.01350`** [A] owns exhaustively paired hate↔counterspeech data — but the pairs
  are *post ↔ response*, not same-content/opposite-stance items, and they are evaluation data.
- **ImpliHateVid `2508.06570`** [A] (ACL 2025) already does two-stage contrastive learning on
  hateful video with **label-defined** pairs.
- Related live area: slur *reclamation* (`2604.16654`, `2602.12818`, the MULTIPRIDE 2026 shared
  task) — the same use/mention failure, heating up.

What is left: use/mention in **video/multimodal** (genuinely unoccupied) and dissent-supervised
factoring of "content present" from "uploader endorses". The novelty checker flagged **feasibility,
not novelty**, as the risk — and **P-B has now answered that: content-matched opposite-stance pairs
do not exist at usable density (1 verified pair)**. That is why this is a component of Rank 1
rather than a paper.

### Rank 3 — **Provenance-typed OCR fusion** (CLOSED for now by its own pilot)

P-C returned AMBIGUOUS on the frozen gating contrast, but three things say the honest reading is
"no evidence": **typed OCR loses to plain untyped OCR on 3/3 seeds (−0.0020)**; the positive gating
number only reflects recovery of a dimensionality penalty the typing introduced against a control
that was not neutral under weight decay; and **the label-permuted null produced a contrast 90 % as
large (−0.0039 vs +0.0044)**. Coverage was adequate (401 overlay / 448 scene / 255 both), so this
is not a power failure.

**Novelty check verdict: ADJACENT, and the discriminator is already built.**
- **`2211.11350` "Rooms with Text"** [A] provides an annotated dataset and a baseline for **binary
  overlay-vs-scene text classification at 0.95 F1**. It is static e-commerce images with no
  temporal features and no moderation framing — but it means "we can tell overlay from scene text"
  is not a contribution.
- **MM-HSD `2508.20546`** [A] is the paper this must beat: on-screen text as a first-class modality
  in video hate detection, with **OCR-as-cross-modal-attention-query** its best configuration
  (M-F1 0.874), treating OCR as one undifferentiated channel.
- The classical graphic-vs-scene taxonomy is **near-dormant**: `"graphic text" AND "scene text"` on
  arXiv returns **0 hits ever**; OpenAlex returns 5 works total (2002–2025).
- Genuinely unoccupied: the *attributability* framing (overlaid text = the uploader's speech act)
  applied to moderation, and **bbox persistence across frames as the type discriminator in video**
  (both returned 0). Closest framing neighbour is a human-subjects study on superimposed text in
  short-video misinformation, not a detector.
- **Coverage caveat**: ACM DL / IEEE Xplore / ICDAR were not searched directly, and this
  discriminator is exactly the kind of thing that lives at ACM MM or ICDAR without reaching arXiv.
  **Treat this "no occupant" finding as weaker than the other two.**

Keep the typing rule as a **measurement asset** — the geometry is in the cache and it is
project-unique. Do not promote it to a registered candidate. But note the scope honestly: the
typing rule was **never validated against ground truth**, and only 255 of 744 videos exercise both
typed blocks, so a wrong or diluted split is indistinguishable from a genuine null here. Any
revival needs (a) a small blinded human check that the "overlay" tracks really are burned-in text,
(b) evaluation on the subpopulation that actually has both text types, (c) a use of the typing
other than a second mean-pooled block, (d) a genuinely neutral parameter-matched control, and
(e) a margin that clears the harness's own null. That is a lot of prerequisites for a channel
whose untyped ceiling is +0.0094 — which is why this is ranked third and not pursued now.

### Everything else — closed or absorbed

| candidate | disposition |
|---|---|
| #2 Dissent-Preserving Prototype Bank | absorbed into Rank 1 as a memory-compression ablation; too close to entropy-gated cache admission to headline |
| #4 Duplicate-Conflict Memory | **closed** by P-B — the phenomenon is not there |
| #6 Sampling-Phase Robust Retrieval | closed — generic TTA/consistency, and a partial re-skin of the dead temporal directions |
| #7 Rank–Vote Decoupling | absorbed as a component / forensic test; no standalone novelty |
| #8 Retrieval Placebo Suite | absorbed as mandatory validity evidence inside any retrieval submission |
| #9 Chance-Corrected Temporal Grounding | retained **as the fallback headline** (below), not as a methods claim |
| #10 Component-Sufficiency Training | closed — sits between DeHate and UniSafe |
| #11 Contested-Item Abstention | absorbed as Rank 1's evaluation protocol; standalone it is never-claim item 12 |
| #12 Annotation-Escalation Prediction | **closed** — direct re-skin, and observational escalation records cannot establish annotator-hours saved |
| #13 Modality-Attributed Retrieval Decomposition | absorbed into the fallback with #8 |

### The honest bottom line

> **Revised 2026-08-09 after P-A-v2 (§3.1b).** The paragraph immediately below was written when
> P-A's GO still stood. It no longer does: the strong-baseline retest returned KILL in both
> languages, so **zero** mechanism candidates now carry empirical support, and the "fallback"
> evaluation-validity paper described further down is no longer a fallback — it is the only
> route in this report with evidence behind it. The original text is kept unedited for the record.

**One idea now has a mechanism with empirical support behind it, and that is a real change from
Phase 1 — but it is not yet a main-conference paper.** The cross-model jury's verdict, recorded
before the pilots ran, was that the pool "does not yet clear a main-conference mechanism bar";
the pilots moved exactly one candidate forward and closed two. Rank 1 is a credible bet, not a
sure one, and its own gate (P-A) is decisively positive in ZH and only suggestive in EN.

**The fallback is stronger than any single mechanism candidate here, and should be built in
parallel rather than held in reserve.** An evaluation-validity paper on hateful video, whose
components are already measured or cheaply measurable:
- chance-corrected grounding (HateMM top-1 chance is **0.762**; gold spans cover median 0.829 of
  the video) plus the length-matched random-window control that reduces an external paper's
  reported "+19.34/+30.45 from trimming to gold spans" to **+0.48 pt, CI [−0.79, +1.76]**;
- the positional-prior and argmax tie-break artifacts documented in the P2 forensic;
- P-B's near-duplicate census, including the label-concordance finding;
- the memory-placebo and modality-attribution decomposition (#8 + #13);
- **and the §0 correction itself** — that MultiHateClip's per-annotator votes, contributing-modality
  and segment-timestamp fields are publicly released but absent from the derived copy the community
  has been circulating, and that binarising the labels destroys a Counter-Narrative class.

Target for that: ACL Main (Resources & Evaluation), NeurIPS D&B, or ACM MM. It must be presented
honestly as an evaluation contribution, not dressed as a mechanism paper.

### Immediate next steps

- [x] **First**: re-run P-A's Δ against a *trained-classifier* hardness baseline, not the 20-NN
      label fraction. This is cheap and it is the sharpest available kill-test for Rank 1 — if the
      vote signal adds nothing over a properly calibrated hardness estimate, close the direction
      before spending anything else.
      **Done 2026-08-09 — P-A-v2 (§3.1b) returned KILL in both languages. Rank 1 is closed.**
- [ ] ~~Then pre-register the Rank-1 three-way ablation (soft BCE / pair topology / distributional
      read-out), multi-seed, MHC-EN + MHC-ZH train+val, rules frozen before implementation.~~
      **Cancelled by P-A-v2**, and independently by §5 (novelty 3/10, ABANDON) and §6 (Reject,
      "NO" to pre-registration). Promote the evaluation-validity paper (see "the honest bottom
      line") from fallback to primary track instead.
- [ ] Before drafting the evaluation-validity paper, discharge the reviewer's six conditions in
      §6.6: one thesis + one reusable protocol (not a grab bag); **produce actual numbers for the
      memory-placebo and modality-attribution decompositions** (currently no evidence); show that
      the 16 degenerate items (~0.5 %) change a ranking or a conclusion, or drop the claim;
      quantify how widely the reduced derived MHC copy is used and what it cost the field; and
      pre-register the confirmatory replications with the full post-hoc selection history disclosed.
      The near-duplicate census (5 pairs, 1 genuine) cannot be a headline.
- [x] ~~If leg (ii) is ever tested: declare it a **new, adaptively selected hypothesis**, freeze the
      primary comparator as **GenSCL / direct LDL** (not hard-label RGCL), include majority- and
      entropy-preserving shuffled-`q` placebos, freeze smoothing / temperature / pair normalisation
      / loss weight / tuning budget, add a futility rule, and treat any positive MHC result as
      **exploratory**. The original gate stays failed regardless of the outcome (§6.5).~~
      **Done 2026-08-09 under exactly those conditions — LEG2-KILL, verdict FAMILY-CLOSED
      (`C − B = −0.0051`, 0/3 seeds; `C − D = −0.0037`, 1/3 seeds). §6.9.**
- [ ] Correct `research-wiki/PAPER_MASTER_TABLES.md:446` and `ITERATION_LOG.md:255+` — both record
      as unavailable data that is in fact publicly released (§0.1).
- [ ] Re-derive the MHC feature caches from the **official** release so `Component`, `Duration`,
      `Target_Victim` and the raw votes are available downstream.
- [ ] **Audit the degenerate CLIP image features** found incidentally by P-B — HateMM 16 items
      (one group of 11 spanning both labels), MHC_zh 2, ImpliHateVid 8, all bit-identical
      mean-pooled vectors, the HateMM ones with placeholder ASR. Present in every experiment on
      those splits; independent of any idea in this report.
- [ ] Obtain the HCG-MPB PDF before any instance-vs-prototype claim is written.
- [ ] Do **not** reopen: near-duplicate minimal pairs (P-B), OCR provenance as a fusion block
      (P-C), annotation-escalation routing (jury).

### Novelty-check coverage and blind spots (stated so the verdicts are not over-read)

The targeted sweep ran ~45 arXiv Atom API queries (zero 429s, all `[A]` items have title + date +
abstract pulled directly) plus 14 OpenAlex `title_and_abstract.search` queries (`[B]` items:
title/date/venue seen, **abstracts not read**). Nothing is asserted at `[C]`; no IDs were invented.

**What failed or was not covered — this bounds every "OPEN" verdict above:**
- **Semantic Scholar was almost entirely rate-limited** (~14 attempts, one partial return). S2 is
  the best index for ACL/EMNLP/NAACL/ACM/IEEE venue papers, so a hidden occupant — a LeWiDi-adjacent
  workshop paper for Rank 1, or an ACM MM / ICWSM paper for Rank 2 — would most plausibly live
  exactly there. The one thing S2 did surface that nothing else did (ImpliHateVid at ACL 2025) is
  evidence this gap costs real coverage.
- **ACL Anthology search failed entirely** (JS-rendered search page); mitigated only indirectly via
  OpenAlex's ACL DOI coverage.
- **ACM DL and IEEE Xplore were not searched directly**; CVPR/ICCV/ECCV/ACM MM were reached only
  via OpenAlex/arXiv. This matters most for Rank 3.
- **WebSearch was not used at all** (session quota exhausted).
- Not covered: ICWSM / CSCW / web-science venues, non-English literature, patents, and industry
  platform technical reports — the last being a plausible unpublished home for Rank 3.

---

## §5 — Phase-3 deep novelty check on Human-Agreement Retrieval (2026-08-09)

Scope: the Rank-1 mechanism only. Method: 30 successful arXiv Atom API queries in three batches
plus 5 `id_list` requests returning **30 abstracts read in full**, then cross-model verification
(`gpt-5.6-sol`, xhigh). Trace: `.aris/traces/novelty-check/2026-08-09_run01/`
(dossier, reviewer response, raw JSON per query).

**Citation discipline.** `[A]` = title + submission date + abstract fetched from the arXiv API in
this session. `[A-prior]` = verified in an earlier session, not re-fetched. **No ID was invented**;
no claim rests on an unverified ID. Venue attributions not independently confirmed are not asserted.

### 5.1 Headline: two of the three Phase-2 leg verdicts were wrong

| leg | Phase-2 | **Phase-3 (corrected)** | why it changed |
|---|---|---|---|
| (i) memory entries carry vote distributions | ADJACENT (1 occupant) | **ADJACENT — 3 occupants, one closer than the one Phase 2 named** | Phase 2 named only UAKNN. `2503.04869` (Dual-kNN) and `2307.10189` (CrowdOpinion) are equally or more on-point. |
| (ii) contrastive pair topology defined by expected inter-annotator agreement | **OPEN — "zero papers found, in any field"** | **ADJACENT — the mechanism template is published and explicitly general. Verdict RETRACTED.** | GenSCL `2206.00384` is exactly "replace one-hot positives with a continuous label-*distribution* similarity". Four further soft-contrastive papers reinforce it. |
| (iii) distributional read-out + disagreement-driven abstention | ADJACENT | **NOT OPEN — directly occupied, and in the same domain** | Crowd-Calibrator `2408.14141` does crowd-agreement-driven **abstention under selective prediction, evaluated on hate speech**. `2411.04090` does disagreement-as-auxiliary-task + disagreement-triggered human review in content moderation. Phase 2 cited neither. |

**What is left unoccupied**, stated precisely: the similarity kernel
`s_ij = q_i^T q_j = P(Y_i = Y_j)` computed from **annotator vote multisets**, and the **hateful-video
setting**. That is a kernel choice inside a published objective, plus a domain transfer.

### 5.2 Leg (ii) — nearest neighbours and the exact difference

| paper | date | what it owns | difference from us |
|---|---|---|---|
| **GenSCL `2206.00384`** [A] | 2022-06-01 | "a **generalized supervised contrastive loss, which measures cross-entropy between label similarity and latent similarity** … fully utilizing the **label distribution**" | **None at the mechanism level.** Its label distributions come from CutMix / knowledge distillation, not human votes. Ours is one choice of the label-similarity function GenSCL already parameterises. **This is the desk-reject citation for leg (ii).** |
| **MaskCon `2303.12756`** [A] | 2023-03-22 | "**soft labels based on sample distances, that are masked by the coarse labels**"; claims to "obtain as **special cases many existing state-of-the-art works**" and gives generalization bounds | Soft targets derive from feature distance masked by coarse labels, not from an annotation distribution. Weaker occupant than GenSCL but a strong "this family is already generalised" citation. |
| **SCE `2111.14585`** [A] | 2021-11-29 | soft contrastive learning: "estimate a **continuous distribution to push or pull instances based on their semantic similarities**" | Self-supervised; similarity is model-estimated, not label-derived. |
| **SoftCon `2405.20462`** [A] | 2024-05-30 | "**soft contrastive learning that optimizes cross-scene soft similarity based on multi-label supervision**, naturally solving … **too strict positive matching**" | Land-cover multi-label kernel, Earth observation. |
| `2501.19145` [A] | 2025-01-31 | multi-label contrastive weighted by recovered **label distributions** | multi-label, not annotation variation. |
| **ConLE `2305.09500`** [A] | 2023-05-16 | contrastive learning **inside** label-distribution learning (label enhancement) | recovers distributions; does not use them to define pair topology for a downstream task. |
| **`2112.15411`** [A] | 2021-12-31 | contrastive **ranking** loss over multi-annotator data + gradient reversal for **annotator invariance**; motivated by subjective tasks | Uses annotator **identity** to build the topology and tries to *remove* annotator variance. We have no annotator IDs, and we want to *preserve* the variance. This is the closest annotator-aware contrastive paper and the sharpest contrast to draw. |

Negative searches for leg (ii), executed this session on the arXiv `all:` field:
`"probability" AND "annotators agree"` → **0 hits**;
`"contrastive regression" AND "continuous labels"` → 0;
`"crowdsourced" AND "contrastive" AND "soft label"` → 1 unrelated;
`"supervised contrastive" AND "label similarity"` → 2 (GenSCL + an unrelated CAD-retrieval paper).
So the *vote-derived* kernel itself is still unnamed in the literature — but the slot it fits into
is not.

### 5.3 Leg (i) — retrieval memory carrying label/vote distributions

| paper | date | what it owns |
|---|---|---|
| **UAKNN `2504.01508`** [A] | 2025-04-02 | LDL performed **as** a kNN predictor with uncertainty-aware neighbour handling, 12 benchmarks; pitched for online deployment. |
| **Dual-kNN `2503.04869`** [A] | 2025-03-06 | **two kNN modules retrieve neighbours from the training set to "augment the distribution of labels"**, plus an LDL module that learns label similarity; explicitly motivated by noisy / similar-label datasets. **This is arguably a closer occupant than UAKNN and Phase 2 missed it.** |
| **CrowdOpinion `2307.10189`** [A] | 2023-07-07 | pools **similar items' label distributions** (language features + label distributions) to predict label distributions, on social-media hate/abuse data. Domain-matched. |
| **Opt-ICL `2510.07105`** [A] | 2025-10-08 | LeWiDi-2025 winner; meta-learned ICL over **rater examples**, ablation says they are the most important component. |
| **DeMeVa `2509.09524`** [A] | 2025-09-11 | compares **example-sampling strategies** for ICL over perspectivist annotations; aggregates per-annotator predictions into soft labels. |
| **`2506.06113`** [A] | 2025-06-06 | few-shot **demonstration selection by textual similarity (BM25/PLM) and by annotation-disagreement entropy**, on **hate speech / offensive language**. Answers the commissioning question directly: **yes, the 2025–26 LLM-annotator-disagreement wave has a retrieval side — but it is prompt-time example selection, not a learned dense retriever.** |

Negative searches: `"human label variation" AND "retrieval"` → **0**;
`"nearest neighbours" AND "predictive distribution" AND "annotators"` → 0;
`"perspectivist" AND "retrieval"` → 1 unrelated. The *learned-retriever* framing is genuinely
unoccupied; the *retrieve-then-read-a-distribution* function is not.

### 5.4 Leg (iii) — disagreement-driven abstention / selective prediction

| paper | date | what it owns | severity |
|---|---|---|---|
| **Crowd-Calibrator `2408.14141`** [A] | 2024-08-26 | "we **calibrate models for subjective tasks based on crowd worker agreement** … models the distance between the distribution of crowd worker labels and the model's own distribution over labels **to inform whether the model should abstain** … **selective prediction** setting … on two highly subjective tasks, **hate speech detection** and NLI … outperforms or achieves competitive performance with existing selective prediction baselines" | **Kills leg (iii) as a contribution. Same mechanism, same domain, two years earlier. Phase 2 never found it.** |
| **`2411.04090`** [A] | 2024-11-06 | content moderation: **toxicity primary task + annotation disagreement as an auxiliary task**, conformal prediction over both, moderator-tunable disagreement thresholds that **trigger review** | **Also owns the "auxiliary dissent head" design** that Rank 1 proposed as its beyond-accuracy capability. |
| **Ghost Annotator `2606.02911`** [A] | 2026-06-01 | conformal prediction + annotator representations for human label variation across **four content-moderation datasets** | 6-month-window parallel work. |
| `2605.24773` [A] | 2026-05-23 | soft-label objective on the empirical annotator distribution, reporting **selective-prediction AURC/AUROC jointly with JSD-to-annotator-distribution** | occupies the exact "joint accuracy + distribution-fidelity + selective-risk" reporting protocol Rank 1 planned. |
| `2606.22725` [A] | 2026-06-21 | aleatoric/epistemic decomposition validated against **annotator disagreement**, inference-time **routing**; notes a strong **LDL baseline already recovers disagreement comparably** | supplies the baseline that would have to be beaten. |
| `2605.02122` STABLEVAL [A], `2604.08425` DiADEM [A], `2508.08163` DisCo [A], `2510.08460` LeWiDi-2025 overview [A], `2601.09065` survey [A] | 2025-08 → 2026-05 | the disagreement-aware modelling/evaluation ecosystem | DiADEM and DisCo both need **annotator metadata/identity**, which MHC does not release — a genuine differentiator, but one that limits us rather than helping. |

**Three findings that cut against the idea's premise, not just its novelty:**
- `2606.28772` [A] (2026-06-27, HateXplain): 42.6 % of annotator disagreement sits at the
  hate/offensive boundary; **hard-label and soft-label models both lose ~22 points on disagreement
  items**; **three downstream interventions all fail**; the authors conclude the fix "must be
  **upstream in annotation design**".
- `2509.06704` [A]: "**combining contrastive loss with binary cross-entropy loss does not improve
  performance**" for subjectivity flagging — the exact loss combination Rank 1 proposed.
- `2605.01168` [A] and `2509.06704` [A] both already do **direct prediction of item contestedness**
  in hate/offensive data, without retrieval.

### 5.5 Recent-window (6 month) parallel-work scan

`all:"annotator disagreement"` sorted by submission date, 30 most recent: the wave is real
(≈ one relevant paper per week through 2026) but is concentrated on **annotator-identity /
demographic modelling** (`2604.08425`, `2605.27313`, `2604.18069`, `2605.28802`), **uncertainty and
calibration** (`2605.24773`, `2605.24722`, `2606.22725`, `2604.24170`), and **evaluation**
(`2605.02122`, `2606.28772`, `2601.09065`). **No 2026 paper puts disagreement on the retrieval
side.** `all:"hateful video"`, 14 most recent: `2606.11953`, `2602.09637`, `2602.00132`,
`2601.15115`, `2512.02743`, `2509.13515`, `2508.06570`, `2508.04900`, `2508.01712`, `2505.12051`,
`2501.15438`, `2408.03468`, `2305.03915`, `2207.00111` — **none uses annotator vote distributions.**
The video setting is genuinely empty; that part of the Phase-2 finding replicates cleanly.

### 5.6 Cross-model novelty verdict (gpt-5.6-sol, xhigh)

Confirms the retraction. Verbatim conclusions:
- "GenSCL alone is enough to invalidate 'zero papers in any field'."
- What remains is "a potentially new **similarity kernel and interpretation**, but not a new
  contrastive-learning mechanism … the contribution is *a human-vote-derived instantiation of
  generalized supervised contrastive learning*."
- Terminology objection worth recording: with no annotator IDs, `q_i^T q_j` is **cross-item
  label-match probability, not inter-annotator agreement**; and the kernel **conflates similarity
  with certainty** (two identical 50/50 items score 0.5, two identical unanimous items score 1.0).
- Composite: "an **application/combination contribution**, not presently a main-conference-level
  mechanism contribution."
- **Novelty score 3/10 · recommendation ABANDON** for the mechanism framing, allowing one bounded
  leg-(ii) kill test.
- Unsearched prior-art leads it flagged (**not verified, must not be cited until checked**): fuzzy
  kNN (Keller/Gray/Givens), Geng's LDL, Peterson et al. human-uncertainty soft labels, SemEval-2023
  Task 11 system papers, HateXplain, Raykar et al. learning-from-crowds / Dawid–Skene, SelectiveNet
  and learning-to-defer; loosely recalled: evidential/Dempster–Shafer kNN, CrowdTruth.

### 5.7 Coverage and blind spots for this Phase-3 sweep

Bounds every "unoccupied" statement above:
- **WebSearch: 0 queries** — the session's 200-call quota was already exhausted before Phase 3.
- **OpenAlex: 0 usable results** — every query returned `Rate limit exceeded` after one probe.
- **Semantic Scholar: not attempted.** ACL Anthology, ACM DL, IEEE Xplore, ICWSM, CSCW: not searched.
- Consequently **workshop and venue-only literature is systematically under-covered**, and the
  LeWiDi / NLPerspectives proceedings are the single most likely home of an unfound leg-(i)/(iii)
  occupant. The external reviewer made the same point independently: "Forty-eight arXiv queries and
  30 abstracts, with no ACL Anthology, ACM DL, IEEE Xplore, ICWSM, Semantic Scholar, or workshop
  search, cannot support 'unoccupied'."
- Two earlier query batches failed outright (HTTP 400 from a URL-encoding bug, then HTTP 429 from
  parallel bursts) and are retained in the trace as `r1.json`/`r3.json` for audit.

---

## §6 — Phase-4 external review of Human-Agreement Retrieval (2026-08-09)

Reviewer: Codex MCP, `gpt-5.6-sol`, reasoning effort **ultra**, single round, adversarial framing.
Full text: `.aris/traces/research-review/2026-08-09_run01/001-round-1-review.response.md`;
brief: `RESEARCH_REVIEW_REQUEST.md` in the same directory (thread `019fe359-46fd-7612-ae6b-d1d4e95ef3c6`).

**Material handed over — nothing withheld.** The mechanism; P-A's numbers *and* the EN CI that
includes zero; **P-A-v2's KILL in both languages with the full arm table**; the explicit statement
that legs (i)/(iii) lost their empirical support while leg (ii) was never directly tested; the
dataset-coverage constraint (votes only in MHC EN/ZH, n = 790/806, test 161/149, ~80 %/74 % of items
with only two votes, no annotator IDs); the prior soft-label pre-gate negative; P-B and P-C;
the §5 novelty retraction; and the §5.7 search-coverage failures.

### 6.1 Score

| venue | score | confidence | call |
|---|---|---|---|
| NeurIPS | **2 / 6** | 5/5 | Reject |
| ICML | **2 / 6** | 5/5 | Reject |
| ICLR | **3 / 10** | 5/5 | Reject |
| ACL main | **2 / 5** | 5/5 | Reject |

Reviewer's framing: "These are charitable **idea-level** scores. A literal submission containing no
trained model and no end-to-end results would receive the floor score." And: "acceptance does not
require proof of harm; it requires evidence of contribution. **There is none.**"

### 6.2 The three most likely rejection reasons (reviewer's own wording)

1. "The central empirical claim is unsupported: the preregistered follow-up explains the original
   positive result as an artefact of a weak baseline and **fails its gate in both languages**, while
   the only surviving training-objective hypothesis has not been evaluated."
2. "The technical novelty is insufficient for a main-conference contribution: generalized
   label-distribution contrastive learning, distribution-valued kNN, and disagreement-driven
   abstention are **all occupied**, leaving `q_i^T q_j` as a narrow kernel choice within an existing
   template."
3. "The evaluation cannot sustain the proposed claims: it relies on one dataset's small EN/ZH
   subsets, tests of only **161/149 videos**, mostly **two anonymous votes** per item, and only
   **63/76 total items** with any Counter-Narrative vote."

### 6.3 Minimum viable strengthening

**"No small experiment set on the existing data would move this to a defensible main-conference
accept at a cost proportional to the expected payoff."**

The bounded kill test the reviewer *would* accept (~1–2 engineering weeks, ~10² GPU-h) requires the
primary comparison to be against **the strongest existing distributional contrastive baseline
(GenSCL / LDL), not against hard-label RGCL**, with majority-preserving and entropy-preserving
shuffled-`q` placebos, identical tuning budgets, and dataset-resampling inference (multi-seed does
not repair 149–161-item sampling uncertainty). Even a positive result "would remain adaptively
selected, single-dataset evidence for a low-novelty instantiation".

A genuine accept package would need a new objective (not a kernel swap), a purpose-built
multi-rater video corpus with rater IDs, and powered beyond-accuracy evaluation — the reviewer's own
rough estimate is 25k–100k safety-sensitive judgments, 3–6+ person-months, hundreds to low-thousands
of GPU-hours. **Not proportional.**

### 6.4 Verdict on the pre-registration question

> **NO.** … "The votes are useful. They are **not currently evidence for a retrieval mechanism**."

Recommended disposition of the MultiHateClip vote data instead: a canonical derived release
restoring votes + `Component` + `Duration`; the parsing / aggregation-loss audit; disagreement-
stratified performance with **strong non-retrieval LDL and calibration baselines**; quantification
of what majority aggregation erases (especially Counter Narrative); and archiving the
pre-registered negative.

### 6.5 Process ruling: is "test leg (ii) alone" legitimate?

**"As continuation of this candidate, it is motivated continuation."** Leg (ii) was not logically
falsified, so testing it is scientifically coherent — but the candidate was promoted *because* P-A
appeared to show a neighbourhood-vote advantage, the frozen follow-up gate then failed in both
languages, and "selecting the only leg the gate did not touch is **adaptive survival after
failure**". A new pre-registration "can constrain future flexibility; it cannot erase the adaptive
reason the hypothesis was selected". If run at all, it must be declared a **new, adaptively selected
hypothesis**, with the primary comparison frozen against GenSCL/LDL, all loss/temperature/smoothing
degrees of freedom frozen, a futility rule, and any positive MHC result treated as **exploratory
until independently confirmed**. **The original gate stays failed regardless of the outcome.**

### 6.6 On the evaluation-validity fallback

"**Yes.** It has higher expected acceptance and higher expected scientific value" — but "it is not
yet a paper". The reviewer's conditions, which are now the binding to-do list for that track:
- It needs **one thesis and one reusable protocol**, not a grab bag of post-failure audits.
- Memory-placebo and modality-attribution results **have no numbers yet** → currently no evidence.
- 16 degenerate items / 3,155 ≈ 0.5 %; **impact on rankings has not been shown**.
- 5 conservative conflicting pairs (1 genuine) **cannot carry a paper**.
- The prevalence and consequences of the community's derived-copy usage are **unquantified**.
- One external temporal-grounding reanalysis **does not establish a field-wide validity problem**.
- New confirmatory replications should be pre-registered with the full post-hoc selection history
  disclosed.
Its strongest asset stands: "an apparent +19.34/+30.45 gain collapses to +0.48 points with
CI [−0.79, +1.76] under the appropriate random-window control."

### 6.7 Design defects the reviewer found that we had not listed

These are new; several are fatal to the mechanism as specified and must be answered before any
revival.

1. **The kernel conflates similarity with certainty.** Two identical `[0.5,0.5,0,0]` items get
   `s = 0.5`; two identical unanimous items get `s = 1.0`. **Contested items are systematically
   weaker positives to each other — precisely the items the method claims to serve.**
2. **The 4-class geometry fights the binary task.** A unanimous *Hateful* and a unanimous
   *Offensive* item have `s_ij = 0` while both are positives under Hateful+Offensive BCE; same for
   Normal vs Counter Narrative on the negative side. No gradient-conflict analysis was planned.
3. **The "new topology" may be nothing but marginalised hard-label training.** `q_i^T q_j` is the
   expected hard positive indicator under independently sampled labels — depending on loss
   normalisation this *is* ordinary SupCon averaged over label uncertainty.
4. **Contestedness is confounded by vote count** — more votes ⇒ more chances to be non-unanimous.
   Neither P-A nor P-A-v2 adjusted for it.
5. **`q_i` is not a population distribution.** With two anonymous votes it is a noisy histogram;
   smoothing injects an unspecified prior; "probability two annotators agree" presumes exchangeable
   sampling from a stable item-specific population, which is unevidenced.
6. **Quadratic pairs are not quadratic evidence** — ~550–580 items per language, seeds are
   optimisation replicates, and EN/ZH from one collection are not demonstrated independent
   replications.
7. **A positive result would not identify the mechanism** — generic smoothing, confidence weighting
   or regularisation would produce it. Placebo arms are mandatory.
8. **The work is not video-specific**: a modality-agnostic loss on frozen pooled CLIP features "does
   not become a video-method contribution merely because the examples happen to be videos".
9. **We over-read P-A-v2 too.** `C` vs `B1` asks whether 8 neighbourhood features can *replace* the
   1792-d representation, not whether they add *conditional* information; the `D` arm that should
   have tested complementarity was regularised away. The frozen gate legitimately says stop, **but
   "the neighbourhood contains no additional information" is not established** — §3.1b should be
   read with that limit.
10. **A metric error in §3.1b, now corrected.** "C is not at chance (0.666/0.705 against base rates
    0.21/0.32)" compares AUROC to class prevalence; **AUROC chance is 0.5**. The permutation nulls
    (0.4751/0.5139) are the correct evidence, and a single null value without its distribution does
    not license calling the null "well-behaved".

### 6.8 Consolidated disposition

**Human-Agreement Retrieval is closed as a main-conference mechanism candidate.** Both independent
model passes (novelty xhigh, review ultra) reached ABANDON / Reject / NO-prereg without prompting,
on top of a frozen pre-registered gate that already fired. The residual actions are:
- Do **not** open a full pre-registration for the composite.
- If leg (ii) is tested at all, it is a **new, adaptively selected, capped kill test** whose primary
  comparator is GenSCL/LDL, declared as such, with a futility rule — and a positive result buys
  "exploratory", not "recommended".
  → **Executed 2026-08-09 under exactly those conditions. Result: FAMILY-CLOSED. See §6.9.**
- Reassign the MultiHateClip vote recovery to the **resource / evaluation-validity track**, where it
  is a genuine contribution (the votes, `Component`, `Duration` and the destroyed Counter-Narrative
  class are publicly released but absent from the copy the community circulates).
- Fold §6.7 items 1–8 into the design record so any future revival starts from them rather than
  rediscovering them.

---

### 6.9 LEG2-KILL — the capped kill test §6.5/§6.8 permitted, executed: **FAMILY-CLOSED**

*(This is the §6 follow-on requested as "§6.1"; that number is already taken by the score table, so
the result is filed here as §6.9 to keep the section numbering unambiguous.)*

**Declaration, reproduced from the freeze and from the result file.** This experiment was an
**adaptively selected** hypothesis, chosen *after* the P-A-v2 gate failed, because leg (ii) is the
only leg that gate did not touch. It **inherited no prior GO**; **the original gate stays failed
regardless of this outcome**; a positive result would have granted only the label "**exploratory**",
never "recommended"; a negative result **permanently closes the entire Human-Agreement family**.

Rules frozen in `idea-stage/PILOT_FREEZE_2026-08-09.md` §LEG2-KILL **before implementation**
(sha256 `14c803d1…a2f902`), synthetic + label-permuted smokes only during implementation, single
submission, 753 s CPU, zero test-set contact (guard armed; 12 paths touched, none containing
`test`). Full write-up `idea-stage/LEG2_KILL_RESULT.md`; raw `idea-stage/leg2_kill.json`; code
`idea-stage/leg2_kill.py`; log `logging/runs/leg2_kill/run.log`.

**Design.** MHC EN (629) + ZH (657), pooled train+val, OOF, stratified 5-fold × 3 fold seeds,
byte-identical folds across arms. One head (`1792→128→{64 proj, 1 logit}`), one optimiser, one
frozen recipe. Objective = `BCE + λ · L_gen` with `L_gen` the **GenSCL Eq. 2 loss**
(arXiv `2206.00384`), `τ = 0.1`, full-batch anchors, no re-normalisation of the kernel. `λ` chosen
per arm per outer fold by inner 3-fold CV over the frozen grid `{0.1, 0.3, 1, 3}` — identical budget
for every contrastive arm. `q_i` = raw 4-class vote histogram, **no smoothing**.

| arm | label similarity `simY(q_i,q_j)` | role |
|---|---|---|
| A | — (`λ=0`) | hard-label BCE baseline |
| **B** | `q_i·q_j / (‖q_i‖‖q_j‖)` | **the primary comparator the reviewer specified**: GenSCL's published cosine label similarity |
| **C** | `q_i·q_j = P(Y_i=Y_j)` | the candidate mechanism |
| **D** | `q_π(i)·q_π(j)` | shuffled-`q` placebo, marginals preserved, labels untouched |

Removing the cosine normalisation leaves exactly the certainty factors `‖q_i‖‖q_j‖` — so `C − B`
**is** §6.7 item 1, isolated. `C − A` was never part of the gate.

**Primary endpoint — OOF binary macro-F1, `M` = mean of EN and ZH.**

| arm | EN | ZH | **M** | M − A |
|---|---|---|---|---|
| A — BCE only | 0.7016 | 0.7243 | 0.7130 | — |
| **B — GenSCL cosine** | **0.7142** | **0.7335** | **0.7239** | **+0.0109** |
| **C — `q_i^T q_j`** | 0.7104 | 0.7272 | 0.7188 | +0.0058 |
| **D — shuffled-`q` placebo** | 0.7130 | 0.7319 | 0.7225 | +0.0095 |

| gate quantity | seed 1 | seed 2 | seed 3 | mean | required | met |
|---|---|---|---|---|---|---|
| `C − B` | −0.00002 | −0.01231 | −0.00286 | **−0.00506** | ≥ +0.005, 3/3 same sign | **no (0/3)** |
| `C − D` | +0.00136 | −0.00911 | −0.00328 | **−0.00368** | ≥ +0.005, 3/3 same sign | **no (1/3)** |

**Verdict: FAMILY-CLOSED.** No AMBIGUOUS branch existed (futility rule per §6.5). Secondary
distribution metrics do not rescue it: KL to the vote-derived soft target is 0.5916 EN / 0.6439 ZH
for C against 0.5838 / 0.6456 for B, and macro soft-F1 is flat at ≈ 0.688 for all four arms.

**The finding that outruns the verdict.** The **shuffled-vote placebo captures 87 % of the
contrastive gain** (`D − A = +0.0095` vs `B − A = +0.0109`), while the candidate captures
`+0.0058`. The contrastive term is doing something for this head, but it is not carrying
annotator-agreement information — votes randomly reassigned to other videos do essentially as well.
That is §6.7 item 7 running in reverse: there was no mechanism-specific effect to identify.

**Honest limits.** `λ` saturated at the grid edge (`3.0` most often, 10/15 folds for EN arm B): a
wider grid would most plausibly strengthen **B**, which only deepens the kill — structurally the same
caveat as P-A-v2's `C = 0.003` edge, pointing the same way. The gate quantities (−0.005, −0.004) are
small against the across-seed spread of `M` (0.704–0.738), and **no bootstrap was pre-registered**,
so the established claim is "**no advantage at or above the pre-registered effect size, with no seed
agreeing in sign**", not "C is significantly worse". §6.7 items 1–5 were deliberately **not** fixed —
this tested the original mechanism as specified. §6.7 items 6 and 8 stand unchanged.

**Disposition.** Human-Agreement Retrieval is now closed on all three legs, by a pre-registered
gate that the family itself asked for. No revival branch on this data. The MultiHateClip vote
recovery remains a live contribution on the **resource / evaluation-validity track** (§6.4, §6.6);
that assignment is untouched by this result.

---

### Reproducibility index

| artifact | path |
|---|---|
| this report | `idea-stage/IDEA_REPORT.md` |
| previous round (archived, not deleted) | `idea-stage/IDEA_REPORT_2026-08-08_archived.md` |
| frozen pilot rules (written before implementation) | `idea-stage/PILOT_FREEZE_2026-08-09.md` |
| cross-model brainstorm bundle | `idea-stage/codex_brainstorm_bundle_2026-08-09.md` |
| pilot scripts | `idea-stage/pilot_a_disagreement_retrievability.py`, `idea-stage/pilot_b_dup_conflict_census.py`, `idea-stage/pilot_c_ocr_provenance.py` |
| pilot raw results | `idea-stage/pilot_{a,b,c}.json` |
| **P-A-v2 strong-baseline retest** | script `idea-stage/pilot_a_v2_strong_baseline.py`, raw `idea-stage/pilot_a_v2.json`, log `logging/runs/pa_v2_retest/run.log`, rules `idea-stage/PILOT_FREEZE_2026-08-09.md` §P-A-v2 |
| **LEG2-KILL** (§6.9) — adaptively selected capped kill test of leg (ii) | write-up `idea-stage/LEG2_KILL_RESULT.md`, script `idea-stage/leg2_kill.py`, raw `idea-stage/leg2_kill.json`, log `logging/runs/leg2_kill/run.log`, rules `idea-stage/PILOT_FREEZE_2026-08-09.md` §LEG2-KILL |
| pilot write-ups | `idea-stage/PILOT_{A,B,C}_RESULT.md` |
| pilot logs | `logging/runs/pilot_{a,b,c}/` |
| **newly recovered data** (official MHC release, SHA-256'd) | `data/gt/mhc_votes/` |
| Phase 1 landscape working notes | `idea-stage/phase1_landscape_update.md` |
| novelty recon / never-claim list | `research-wiki/NOVELTY_RECON_2026-08-09.md` |
| **Phase-3 deep novelty check** (§5) — dossier, reviewer response, 30 raw arXiv query dumps, 30 fetched abstracts | `.aris/traces/novelty-check/2026-08-09_run01/` |
| **Phase-4 external review** (§6) — brief + verbatim reviewer response + run metadata | `.aris/traces/research-review/2026-08-09_run01/` |

---

## §7 — Round 3 (2026-08-09): forced structural pivot

**Mandate.** After 27 dead candidates across two rounds (14 archived + 13 in §2), this round was
run under a **structural-turn constraint**: no new variant may be generated from the
retrieval / memory / OCR / temporal / cross-lingual / disagreement families, all of which are
documented dead in §1.4, §2, §4.3 of the archived report, and `research-wiki/NOVELTY_RECON_2026-08-09.md`.
Five new radii were mandated instead: (1) training signal, (2) distillation / generation,
(3) structure / representation, (4) robustness / deployment, (5) free.

**Funnel**: 14 candidates generated → 14 through the objective feasibility gate as *ideas*, but
**2 lost their MVE to a missing asset** (C4, C5) → cross-model triage re-ranked → **3 $0 CPU pilots
run under rules frozen in advance** → see §7.5 for survivors.

**Method.** Generation and triage by `gpt-5.6-sol` at `model_reasoning_effort: xhigh`
(thread `019fe558-4208-79d1-8e25-159c819a2f68`); bundle
`idea-stage/codex_brainstorm_bundle_r3_2026-08-09.md`. The executor eliminated nothing on quality
grounds; the objective feasibility gate reported *facts about disk state* back to the jury, and the
jury did all the narrowing. Pilot rules frozen in `idea-stage/R3_PILOT_FREEZE_2026-08-09.md` before
any implementation line was written.

### 7.0 An asset the first two rounds never touched

The feasibility survey found that the official MultiHateClip release carries **three** structured
fields beyond the votes §0 recovered, and that only one of them had ever been considered:

| field | coverage (train+val) | used by any of the 27 dead candidates? |
|---|---|---|
| `Label` (raw per-annotator votes) | EN 801 / ZH 800; 580 items ×2 votes, 120 ×3, 1 ×4 | yes — round 2 #1, #2, #3, #11, #12 (all dead) |
| `Component` (contributing modality set) | EN: Transcript 222 / Metadata 197 / Audio 150 / Vision 143; ZH: Metadata 294 / Vision 214 / Transcript 172 / Audio 142 | yes — round 2 #10 (dead) |
| **`Target_Victim`** (annotated target group) | **248 EN + 278 ZH** | **no — never touched** |
| `Duration` (annotated hateful spans) | 260 EN / 262 ZH (11 EN / 5 ZH multi-span) | indirectly, via the dead temporal line |

Two observations that shaped this round:
- **`Target_Victim` is the only genuinely unspent structured label in the project.**
- **`Metadata` (title/description text) is the single largest contributing modality in ZH (294) and
  the second largest in EN (197) — and no model in this project consumes it at all.** That is a
  documented, human-annotated evidence channel sitting unread. It is recorded here as a standing
  observation, not as a candidate: "add the metadata channel" is exactly the kind of
  add-a-modality move §4.4's inherited negatives bound, and it is not a mechanism.

### 7.1 Candidate table — all 14

| # | candidate | family | mechanism in one line | disposition |
|---|---|---|---|---|
| C1 | **Target-conditioned attack/defence algebra** | 1 | latent (target-binding × derogatory-content × stance); Hateful votes constrain the signed product > 0, Counter-Narrative votes constrain content > 0 with stance < 0 — hate and counter-speech become algebraic inverses | **piloted (R3-1)** |
| C2 | Victim-marginalized attack energy | 1 | video score = temperature-controlled marginal of an open-vocabulary energy `E(x, target)`; `Target_Victim` supervises which term dominates | held — jury 4.8/10, "may be just a multi-task target classifier" |
| C3 | Vote-constrained semantic polytope | 1 | each raw vote maps to linear inequalities over latent harm axes; training minimises distance to the feasible set instead of matching a vote histogram | held — jury 3.8/10, hand-authored geometry |
| C4 | Semantic response-tensor distillation | 2 | distil the teacher's *finite-difference Jacobian/Hessian* over named semantic interventions, not its logits or its explanations | **feasibility KILL for $0 — data premise absent (§7.2)** |
| C5 | Möbius interaction distillation | 2 | Möbius-invert teacher scores over modality coalitions into an interaction spectrum; student predicts the coefficients | **self-KILL on its own precondition (§7.2)** |
| C6 | Executable agency-graph distillation | 2 | typed graph over speakers/uploader/quoted-sources/propositions/victims with `asserts / quotes / endorses / condemns` edges; hate counted only when an endorsement path reaches an accountable agent | held — needs generation, 1–2 months, "explanation distillation in disguise" risk |
| C7 | Noncommutative rhetorical pooling | 3 | video = ordered *product* of near-identity matrices `exp(A(z_t))`; commutator terms represent rhetorical reversal | held — jury 5.0/10; labels probably cannot identify it (Gate-0's 8.2 % multi-segment finding) |
| C8 | **Prosody-as-operator binding** | 3 | audio may only parameterise a low-rank operator applied to transcript states; it is forbidden from emitting a hate logit directly | **piloted round 4 (2026-08-09) → KILL, family closed (§7.10)** |
| C9 | Rank-copula multistream pooling | 3 | within-video soft empirical ranks → differentiable cross-modal copula tensor instead of mean/max pooling | held — jury 4.5/10; and §7.2 shows its stability arm is single-dataset |
| C10 | **Cross-channel evasion transduction closure** | 4 | evasion as a typed transduction graph across overlay text / speech / metadata; worst-path loss + **path closure** (two attack sequences with the same semantic endpoint must land in the same latent state) | **piloted (R3-3)** |
| C11 | Platform-invariant semantic derivatives | 4 | align the conditional distribution of `D_g f(x) = f(gx) − f(x)` across platforms, not features or risks | held — jury 4.1/10, "IRM in finite-difference notation" |
| C12 | **Proposition-mass firewall** | 5 | OCR/ASR tokens soft-assigned to proposition units; each proposition gets one *conserved* unit of evidence mass, doubly-stochastically allocated across its duplicates, boxes and channels | **piloted (R3-2)** |
| C13 | Counterexample-guided harm-circuit compiler | 5 | synthesise an editable monotone logic circuit over atomic harm facts by CEGIS against declared policy invariants | held — jury 4.7/10, atom errors dominate |
| C14 | Relational quotient induction (latent dogwhistles) | 5 | represent a phrase by the *operator it induces on a target-conditioned context state*; quotient by relational action, flag low raw-similarity members as codewords | held — jury 3.5/10, **and text-domain occupancy is heavy (§7.3)** |

**Nothing was eliminated by the executor on quality grounds.** C4 and C5 were dropped from the pilot
set by the *objective* feasibility gate (a fact about disk contents, per the acceptance-gate protocol);
the rest of the narrowing was the cross-model jury's.

### 7.2 Objective feasibility gate — what the disk actually contains

This is where the round's most useful negative information came from. Two candidates lost their
minimum viable experiment to assets that do not exist, and two more lost scope.

**C4 — the intervention lattice does not exist.** The candidate's MVE assumed the 348 cached
"counterfactual twins" provide a factorial lattice of semantic interventions. They do not.
`data/Counterfactual/{MHC,MHC_zh}/train_twins.jsonl` has schema
`{id, label, orig_text, sanitized_text, orig_verdict, san_verdict, regen_used, flipped}` and:
- **every record has `label = 1`** — harmful videos only, no negative-class twins;
- there is exactly **one** intervention type, a toxicity-sanitising rewrite of the transcript. No
  target substitution, no endorsement/condemnation reversal, no surface obfuscation, no modality
  removal. Mixed second-order derivatives therefore have no data at all;
- verdict transitions: EN {HARMFUL→BENIGN 75, HARMFUL→HARMFUL 74, BENIGN→BENIGN 18, BENIGN→HARMFUL 1};
  ZH {HARMFUL→HARMFUL 112, HARMFUL→BENIGN 57, BENIGN→BENIGN 9, BENIGN→HARMFUL 2}. Only **132 of 348
  pairs actually flip**.

Building the lattice is possible (the Claude-API frame/text exemption makes it cheap in hours, not
GPU-days) but it is **not a $0 cached-asset pilot**, and the generator would sit in the same model
family as the teacher — which is precisely the "amplifies generator bias" objection the jury raised
against C4 in the first place. **Disposition: C4 is not dead, it is *unfunded*. It is the one
candidate whose revival has a concrete, bounded prerequisite** (build and human-verify a real
intervention lattice with ≥ 2 intervention axes and both classes) rather than a novelty or
mechanism problem.

**C5 — self-kill on its own frozen precondition.** Its MVE required cached teacher scores over the
modality coalitions ∅, V, T, A, VT, VA, TA, VTA. `data/MLLM_scores/*` contains only *per-segment
harmfulness scores under different prompts and model sizes* (Qwen2.5-VL 7B/32B/72B-bnb4,
Qwen3-VL-30B/32B, plus fuse / lex / gate / fewshot prompt variants) — never a coalition sweep.
The candidate's own rule says "absence is an immediate KILL". Recorded as such.

**C9 — scope loss.** CLIP subclip caches exist at K=4 **and** K=30 for HateMM, but at **K=4 only**
for MHC and MHC_zh; K=60 exists for ASR text only, never for CLIP. Its "sampling-density stability"
arm is therefore single-dataset, and its frozen clause "no dataset exceeds 0.05 median drift" is
unfalsifiable as written.

**Fully runnable at $0 CPU on cached assets, train/val only**: C1, C2, C3, C8, C10, C11, C12, C13;
C7 on HateMM only. C6 and C14 need generation or GPU.

**Test-set integrity check performed during the gate**: `data/OCR/HateMM/ocr_video.jsonl` covers
exactly 851 videos = train (744) + val (107), with **zero** of the 215 test ids present. The OCR
cache is structurally incapable of leaking test data.

### 7.3 Novelty probe — thin instrument, honest bounds

**WebSearch was unavailable: the session's 200-call budget was already exhausted** (the same
limitation §4's coverage note records). The probe therefore ran on the **arXiv Atom API and OpenAlex
only**. Semantic Scholar, ACL Anthology, ACM DL and IEEE Xplore were **not** queried. Treat every
"no occupant" statement below as **weaker than a Phase-3 sweep** and as `[UNVERIFIED]` unless an ID
is given with a confirmed title.

| probe | result |
|---|---|
| `abs:"counterfactual" AND abs:"distillation"` | **DISCO `2212.10534`** (Distilling Counterfactuals with LLMs), **`2510.21631`** (Few-Shot KD of LLMs with Counterfactual Explanations). Phrase `"counterfactual distillation"` → 0 hits |
| `abs:"Jacobian matching" AND abs:distillation` | **`1803.00443`** Knowledge Transfer with Jacobian Matching — the closest thing to C4's mechanism, and it predates the LLM era |
| `abs:"metamorphic" AND abs:"distillation"` | `2511.05476` (metamorphic-testing view of KD for code LMs) |
| duplicate/repetition-invariant multimodal aggregation; Sinkhorn evidence fusion (C12) | **0 hits on three formulations** |
| `abs:"counter speech" AND abs:"stance" AND abs:"hate"` | 1 hit (`2403.15449`, persuasion modes) |
| `abs:"target" AND abs:"stance" AND abs:"hate speech" AND abs:"compositional"` | 0 |
| noncommutative / matrix-product sequence representation (C7) | pure mathematics; nothing ML-relevant |
| `abs:"prosody" AND abs:"hate speech" AND abs:"video"` | **0** |
| `abs:"obfuscation" AND abs:"hate" AND abs:"multimodal" AND abs:"robust"` | 0 |
| dogwhistle / euphemism (C14) | **heavily occupied in text**: FETCH! `2412.12072`, Silent Signals `2406.06840`, Dogwhistles to Bullhorns `2305.17174`, Chinese cant `2104.02704`, plus a euphemism-detection shared-task line |

The only probe that returned a *damaging* occupant is C14's. C4's nearest neighbour
(`1803.00443` Jacobian matching) is a mechanism template, not an occupant of the semantic-intervention
framing — but combined with DISCO it means C4 could never claim "distilling counterfactual behaviour"
as new, only the specific *named-intervention response tensor*.

### 7.4 The audio question, revisited (family 3)

The brief asked for the angle on which Phase 1's "audio prior is weak" judgement might have been
wrong. The jury's answer is worth recording even though C8 was not piloted: **Phase 1 almost
certainly measured the wrong estimand.** A weak *marginal* audio signal (audio-only accuracy, or the
gain from concatenating an audio vector) is fully compatible with a strong *conditional interaction*
— threat, mockery, sarcasm and slogan-chanting change what a transcript means without being
detectable from the audio alone. C8's proposed discipline follows from that: constrain the audio
branch so it can only *transform* transcript states and is structurally forbidden from emitting a
hate logit, then require that ≥ 70 % of any gain disappears when audio is shuffled across examples
within label × language strata.

It carries a clean falsification and is among the cheapest untested ideas left. It was not piloted
this round because the jury ranked it 4.3/10 on prior-art risk (FiLM, hypernetworks, bilinear
fusion). **Recorded as the first thing to run if a fourth round happens** — with one asset
correction attached, below.

> **Superseded 2026-08-09 by §7.10.** C8 was run as the first round-4 pilot and **killed**. The
> estimand argument above was correct that Phase 1 had only measured the marginal — but when the
> conditional estimand is measured, it is **negative**, and it is *most* negative exactly inside the
> text-boundary band where this paragraph predicted it would be largest. The paragraph is left
> unedited because it was the reasoning that justified the experiment; §7.10 carries the answer.

> **Asset correction discovered during pilot R3-1, recorded here because it changes C8's cost.**
> The feasibility survey stated that CLAP audio embeddings are cached for HateMM / MHC / MHC_zh.
> That is **wrong**. `data/audio/HateMM/` holds `clap_larger_clap_general*`; `data/audio/MHC/` and
> `data/audio/MHC_zh/` hold **Whisper-large-v3 encoder** caches only
> (`whisper_whisper-large-v3*`). So CLAP exists for **HateMM only**. A prosody experiment on MHC
> must either use the Whisper audio-encoder features already cached, or extract CLAP from the raw
> audio (`data/audio/` totals 247 MB; raw audio is local, so this is cheap but non-zero). Pilot R3-1
> hit this directly and refused to substitute silently — its gating features were MPNet + CLIP
> image + CLIP text only, with a labelled non-gating Whisper-audio variant reported separately.

### 7.5 Pilot results

Three $0 CPU pilots, decision rules frozen in `idea-stage/R3_PILOT_FREEZE_2026-08-09.md` **before any
implementation line was written**. Each was a single submission after synthetic + label-permuted smoke
tests. Zero test-set contact in all three (path guards armed; touched-path lists recorded in each JSON).
Total cost: CPU-minutes.

| pilot | candidate | what it gates | verdict |
|---|---|---|---|
| **R3-1** | C1 target-conditioned attack/defence algebra | the algebraic double dissociation (counter-speech carries hate-like *content* yet separates on an independent *stance* direction) | **KILL** — 4 of 6 conditions fail |
| **R3-2** | C12 proposition-mass firewall | whether formatting mass, rather than informative-window dilution, explains the OCR sign flip | **KILL** — see below |
| **R3-3** | C10 cross-channel evasion transduction closure | whether composed evasion paths carry inconsistency single-edge augmentation does not already cover | **KILL**, and inverted |

#### 7.5.1 R3-1 — C1 stance algebra: **KILL**

Code `idea-stage/r3_pilot1_stance_algebra.py`, raw `idea-stage/r3_pilot1.json`, log
`logging/runs/r3_pilot1/run.log`. 396 s CPU, 5 seeds × 5-fold OOF, 100 null replicates.

| quantity | value | frozen rule | result |
|---|---|---|---|
| matched `H` / matched `C` | **34 / 46** | ≥ 40 each | **FAIL** |
| `D_content` | **0.5857** | ≥ 0.60 | **FAIL** (narrowly) |
| `G_content` | **0.4615** | ≤ 0.35 | **FAIL** |
| `D_stance` | **1.4640** | ≥ 0.60 | PASS |
| per-language | EN 0.588 / 0.453 · ZH 0.574 / 1.285 | both ≥ 0.35 | PASS (but see below) |
| `T_obs` vs `3×N95` | **0.5857** vs **0.8398** (`N95` 0.2799) | `T ≥ 3×N95` | **FAIL** |

Per-seed `T`: 0.5611 / 0.5943 / 0.5882 / 0.5554 / 0.6293 — stable; no seed reaches the bar.
Groups over 1002 analysable items (284 of 1286 fall outside `H`/`C`/`N`, e.g. Offensive-only or
Hateful+Counter-Narrative mixtures): pooled `H` 189, `C` 83, `N` 730.

**The one impressive number is an artifact, and this is the finding.** `D_stance = 1.46` does not
survive either check:
1. **The matched population is almost perfectly confounded with language.** Matched `H` = 32 EN + 2 ZH;
   matched `C` = 6 EN + 40 ZH. The `H`-vs-`C` contrast is largely EN-versus-ZH. Condition 5 "passes"
   only because EN's `D_stance` (0.453) rests on **6** matched `C` items and ZH's (1.285) on **2**
   matched `H` items.
2. **`D_stance` does not clear its own null.** Its label-permutation `N95` is **1.2886** against an
   observed 1.4640 — nowhere near 3×. The tiny matched population makes large stance-like separations
   easy to obtain by chance. Only `D_content` has a well-behaved null (`N95` 0.280), and it fails both
   its absolute bar and 3×N95.

So both arms of the double dissociation fail: content signal is real but weak, and the stance arm has
no separation beyond what relabelling produces. **This is the same failure shape as round 2's P-A/P-A-v2**
— a signal that looks strong until it is measured against a properly matched comparator — and it is why
the 3×-null clause was written into the freeze.

Two non-gating sensitivities confirm the choices did not drive the outcome: the permissive stratum
keying (first target label) lifts matched counts to 60/47 and passes conditions 1 and 5, but leaves
`D_content` 0.5857, `G_content` 0.4615, `T` 0.5857 vs 3×N95 0.840 → **KILL**; adding Whisper audio →
`D_content` 0.567, `G_content` 0.458, `D_stance` 1.467 → **KILL**.

#### 7.5.2 R3-2 — C12 proposition-mass firewall: **KILL**, and the sign is the finding

Code `idea-stage/r3_pilot2_proposition_mass.py`, raw `idea-stage/r3_pilot2.json`, log
`logging/runs/r3_pilot2/run.log`. 1292 s CPU. 851 videos in the OCR cache → **688 retained**
(163 have zero surviving detections after the project's standard `conf ≥ 0.5`, `len ≥ 2` filter);
108,291 of 121,462 detections kept; mean `N_i` 157.4 vs mean `U_i` 43.1 (median 60 vs 14) — so there
**is** abundant formatting redundancy to detect, mean excess format mass `g` = 1.277 nats.

| frozen condition | value | bar | result |
|---|---|---|---|
| 1. `rho_obs ≥ 0.24` | **−0.0345** | 0.24 | **FAIL** |
| 2. `A_obs ≥ 0.30` | **0.0380** | 0.30 | **FAIL** |
| 3. `rho_obs ≥ 3×N95(rho)` | −0.0345 vs **0.2062** | — | **FAIL** |
| 4. `A_obs ≥ 3×N95(A)` | 0.0380 vs **0.1586** | — | **FAIL** |
| 5. `corr(c, r) ≥ 0.80` | **0.9345** | 0.80 | PASS |

Per-seed `rho`: −0.0551 / −0.0352 / −0.0107 / −0.0547 / −0.0168 — **negative in all five seeds**.
Per-seed `A`: 0.0387 / 0.0350 / 0.0377 / 0.0426 / 0.0361, against a **null `A` mean of 0.0409** — the
observed protection advantage is *below the average of the label-permuted null*. The verdict is
identical under the stricter min-over-seeds reading.

**The estimator is not dead — a planted effect is recovered.** The synthetic smoke carries a
deliberately planted format-mass effect and the same estimator returns `rho = 0.447` against
`N95 = 0.167`. So the near-zero real-data value is a measurement, not a broken instrument.

**The frozen interpretation clause fires.** The freeze registered in advance that "failure of
condition 1 kills the formatting-mass explanation of the observed OCR sign flip and favours the
competing informative-window-dilution explanation." Condition 1 did not merely fail, it came out
**negative and consistently so**: videos with *more* excess format mass are, if anything, videos where
conserving mass helps *less*. **The +0.0094 → −0.0246 OCR sign flip is not caused by duplicated or
fragmented overlay text.** Combined with the known concave dose curve (3 of 30 windows give 61 % of the
gain), the surviving explanation is dilution by uninformative windows. The pilot also located *where* the
apparent effect goes: `N_i`/`U_i` inflation is heavily entangled with **how many windows carry text at
all**, so what looks like formatting mass in the raw marginal is absorbed by the window-count and
concentration controls. In the synthetic control the same entanglement collapsed a raw `corr(z, g)`
of 0.55 to ≈ 0 once repetition also inflated the window count. And `corr(c, r) = 0.93` bounds the
upside independently: mass conservation barely reorders anything, so even a perfect firewall has ~7 %
of variance to work with — not enough to account for a −0.0246 macro-F1 swing.

**Why the conservation guarantee is narrower than it looks — a design finding worth keeping.** The
per-attack breakdown shows the conserved score `c` is *exactly* invariant (displacement 0.000) to
box reorder, box-area scaling, and cross-channel duplication, exactly as the mechanism promises. But
it is **not** invariant to the two attacks that actually dominate the maximum: token repetition
(`|Δc|` 0.19 vs `|Δr|` 0.24) and single-box splitting (`|Δc|` 0.22 vs `|Δr|` 0.25). The reason is
structural: both attacks change the *normalised string itself*, so the unique-string key changes and
"one conserved unit per proposition" never engages. Conservation keyed on exact string identity
therefore protects only against the attacks nobody would use. The raw score's exposure to the three
pure-formatting attacks is real but **small** (0.07–0.09 SD), while both text-rewriting attacks move
**both** scores by 0.20–0.26 SD; since `A_obs` takes max-over-attacks separately for `r` and `c`, both
maxima are set by the text attacks and the residual gap is only 0.038 SD, inside the null band. This is precisely the objection the jury
raised pre-pilot ("the formal guarantee covers exact duplication only, while real attacks use
semantically similar paraphrases"), now measured rather than argued.

*(A recorded literal-rule note: two of the null 95th percentiles could in principle be negative, which
would make `3 × N95` a weaker bar than `N95`. The frozen rule was applied literally and unchanged; the
stricter `3 × max(N95, 0)` variant is also reported and both fail, so the verdict cannot turn on it.)*

#### 7.5.3 R3-3 — C10 transduction closure: **KILL, and the sign is inverted**

Code `idea-stage/r3_pilot3_transduction_closure.py`, raw `idea-stage/r3_pilot3.json`, log
`logging/runs/r3_pilot3/run.log`. 152 s CPU, 851 videos (744 train + 107 val), 341 positive,
41 evaluated states per video (identity + 4 single edges + 12 length-2 + 24 length-3).

| frozen condition | value | bar | result |
|---|---|---|---|
| 1. `P_obs ≥ 0.35` | **0.2856** | 0.35 | **FAIL** |
| 2. `A_obs ≥ 0.15` | **0.1984** | 0.15 | PASS |
| 3. `P_obs ≥ 3×N95(P)` | 0.2856 vs **1.7448** | — | **FAIL** |
| 4. `A_obs ≥ 3×N95(A)` | 0.1984 vs **1.0919** | — | **FAIL** |
| 5. clean-margin retention ≥ 0.90 | **1.0039** | 0.90 | PASS |

`N95(P) = 0.5816`, `N95(A) = 0.3640`. Per-seed P: 0.2881 / 0.2866 / 0.2844 / 0.2877 / 0.2811.

**The decisive number is that the null is *larger* than the observation.** Null `P` over 100 replicates
spans 0.447–0.596; the observed value from every one of the five seeds falls **below the entire null
distribution**. A permuted-label model has no stable clean direction, so its logit SD is noise and path
perturbations move it freely; the real model has a genuine direction that composed evasion paths fail
to disturb proportionally. That is the **opposite** of C10's premise.

Condition 5 is not merely passed but saturated at **1.004** — training on originals plus all four
single-edge attacks costs nothing in clean signed margin and marginally helps. **The augmentation is
free, and it already covers composition.** Path-closure regularisation would be constraining a
quantity that is already at or below noise.

*Honesty note, recorded because it cuts the other way.* A wider **non-gating** equivalence grouping
(all length-2/3 compositions grouped by multiset of abstract ops, with `L_A`/`L_O` both mapping to a
generic obfuscation op — a rule that reproduces both of the freeze's declared classes) gives
`P = 0.4161`, which *would* clear condition 1's 0.35 bar. It changes nothing: conditions 3/4 still fail
by 4–6× under any grouping, because the wider grouping raises the null identically. The gating number
remains the freeze's literal one. `L+L = 0.000` exactly (the two obfuscation maps commute, as they
must), which is a sanity signal that the transforms behave as specified.

### 7.6 Ranking and survivors

**Survivors: zero.** All three piloted candidates were killed by rules frozen before implementation.
The pre-pilot cross-model ranking was C1 5.0/10 · C12 4.8/10 · C10 4.7/10, with the jury's own
statement — recorded in the freeze before any result existed — that round 3 contained nothing above
~5/10. The pilots did not rescue anything; they lowered all three.

| rank | candidate | pre-pilot | post-pilot | why |
|---|---|---|---|---|
| — | C1 stance algebra | 5.0/10 | **CLOSED** | both arms of the double dissociation fail; the one large number is language-confounded and inside its own null |
| — | C12 proposition-mass firewall | 4.8/10 | **CLOSED** | the mechanism's premise is false (negative `rho`), and its invariance guarantee provably misses the two attacks that dominate |
| — | C10 transduction closure | 4.7/10 | **CLOSED** | observed path inconsistency lies *below the entire null*; single-edge augmentation is free and already sufficient |
| 4 | **C4 response-tensor distillation** | 6.0/10 | **UNFUNDED, not disproven** | the only candidate whose blocker is a *missing asset* with a bounded fix, not a failed mechanism or an occupant |
| — | C8 prosody-as-operator | 4.3/10 | **CLOSED (§7.10)** | piloted 2026-08-09: the boundary-band interaction is **−0.044 / −0.039 AUC**, 0/3 seeds positive in both prosody representations; §7.4's dilution prediction comes out inverted |
| 6+ | C2, C3, C6, C7, C9, C11, C13 | 3.8–5.0/10 | held | none was rescued by anything this round produced |
| — | C5, C14 | 4.0 / 3.5 | **CLOSED** | C5 self-killed on its own asset precondition; C14 is heavily occupied in text |

**The closest thing to a survivor is C4, and it is closest by a specific, nameable gap.** Its jury
score (6.0/10) was the highest of the round against a practical 7–7.5/10 bar, and — uniquely — it was
not killed by a mechanism failure, an occupant, or a null. It was killed by the discovery that
`train_twins.jsonl` contains one intervention type on one class. **Its revival condition is concrete
and bounded**: build an intervention lattice with ≥ 2 axes (at minimum target substitution and
endorsement/condemnation reversal) over **both** classes, human-verify a sample of it, and only then
run the response-tensor gate. The Claude-API frame/text exemption makes that hours of work, not
GPU-days. That is the single most defensible next expenditure this round identified — but it is a
*data-building* bet, and its prior-art position (`1803.00443` Jacobian matching + DISCO `2212.10534`)
means it can only ever claim the named-intervention response tensor, never "distilling counterfactual
behaviour".

### 7.7 What this round actually bought

Three kills is the headline, but the transferable findings are worth more than the candidate list:

1. **The OCR sign flip has a cause, and it is not formatting.** R3-2's negative `rho` retires the
   duplication/fragmentation hypothesis and leaves informative-window dilution as the surviving
   explanation. That constrains every future OCR design in this project: the lever is *which windows*,
   not *how the evidence is weighted*.
2. **Composed evasion needs no special machinery on this data.** R3-3 shows single-edge augmentation
   is free (retention 1.004) and already absorbs length-2 and length-3 compositions. Any future
   robustness claim in this project must clear that baseline, which is cheap and now measured.
3. **String-identity conservation is the wrong key.** Exact invariance held for reorder, area scaling
   and cross-channel duplication and failed for repetition and splitting, for a structural reason.
   Any revival needs a proposition key that survives paraphrase — a much harder object.
4. **`Target_Victim` is spent as a stratifier and remains unspent as a mechanism.** R3-1 used it for
   matching and immediately hit the binding constraint: strict stratum matching leaves 34 `H` / 46 `C`
   and near-total language confounding (matched `H` = 32 EN + 2 ZH; matched `C` = 6 EN + 40 ZH).
   Any future target-conditioned design must solve that confound first, or it will reproduce R3-1's
   artifact.
5. **A standing asset correction** (§7.4): CLAP is cached for **HateMM only**; MHC/MHC_zh carry
   Whisper-encoder audio. This had been recorded incorrectly and was caught only because R3-1 refused
   to substitute silently.
6. **An unread evidence channel is documented**: `Metadata` is the largest human-annotated
   contributing modality in ZH (294) and second in EN (197), and no model here consumes it. Not a
   candidate — "add a modality" is not a mechanism — but it bounds how much of the human-annotated
   evidence the current system can even in principle see.

### 7.8 Honest bottom line for round 3

> **Round 3 produced no viable main-conference methods candidate.** This is the conclusion that was
> registered *in advance* in `idea-stage/R3_PILOT_FREEZE_2026-08-09.md` for exactly this outcome, and
> it is reported unchanged.

Cumulative across three rounds: **41 candidates generated, 41 dead or held below the bar, 0 live
mechanism candidates.** The forced structural pivot did what it was supposed to do — it produced
genuinely new radii rather than re-skins, and the cross-model jury flagged no candidate as a death-list
re-skin — but the new radii died on evidence rather than on novelty, which is the more informative way
to fail.

The standing recommendation from §4 is unchanged and now better supported: **the evaluation-validity
track is the only direction in this project with evidence behind it.** Round 3 adds two publishable
negative results to that track's inventory (the OCR-sign-flip forensic in §7.5.2 and the
composition-is-free result in §7.5.3), both with frozen rules, proper nulls and zero test contact.

**If a round 4 is authorised, the ordered queue is:**
1. ~~**C8 prosody-as-operator**~~ — **executed 2026-08-09, KILL (§7.10).** Queue position 1 is spent.
2. **C4 response-tensor distillation** — highest jury score, blocked by a bounded data-build. Do the
   lattice only if someone will commit to the human verification, since without it the round-2
   rejection reason "the capability you advertise is an evaluation artifact" applies immediately.
   **With C8 dead this is the only remaining item on the queue.**
3. Nothing else on the list justifies its cost as currently formulated.

### 7.9 Coverage and process notes (so the verdicts are not over-read)

- **The novelty probe is weak** (§7.3). WebSearch was fully exhausted; only arXiv Atom API and
  OpenAlex were queried. Semantic Scholar, ACL Anthology, ACM DL and IEEE Xplore were not. **No KILL
  in this round rests on a novelty verdict** — all three rest on frozen empirical gates, which is the
  one respect in which this round's verdicts are more robust than round 2's.
- **Generation and triage were cross-model** (`gpt-5.6-sol`, xhigh, thread
  `019fe558-4208-79d1-8e25-159c819a2f68`); the executor eliminated nothing on quality grounds. The
  feasibility gate reported only objective disk facts back to the jury.
- **Zero test contact in all three pilots**, guards armed, touched-path lists and input SHA256s
  recorded in each JSON. R3-1 touched 18 files, R3-3 touched 5, none containing `test`.
- **One process deviation (R3-1)**: the first launch backgrounded a compound `mkdir && nohup`, starting
  two identical instances. Both were killed, partial output deleted, and a single clean run relaunched;
  the code is deterministic and reproduced seed-1 numbers exactly. The aborted log is preserved at
  `logging/runs/r3_pilot1/run.aborted-duplicate-launch.log`. No threshold, statistic or rule was
  changed at any point in any pilot.
- **Two non-gating sensitivities were run and reported** (R3-1 permissive stratum and Whisper-audio
  variants; R3-3 wider equivalence grouping). All are labelled non-gating; none changes any verdict.
  The R3-3 one is recorded specifically because it would have *passed* one condition — it is kept
  visible rather than dropped.

**Reproducibility index for §7**

| artifact | path |
|---|---|
| generation bundle | `idea-stage/codex_brainstorm_bundle_r3_2026-08-09.md` |
| frozen pilot rules | `idea-stage/R3_PILOT_FREEZE_2026-08-09.md` |
| R3-1 code / raw / log | `idea-stage/r3_pilot1_stance_algebra.py` · `idea-stage/r3_pilot1.json` · `logging/runs/r3_pilot1/run.log` |
| R3-2 code / raw / log | `idea-stage/r3_pilot2_proposition_mass.py` · `idea-stage/r3_pilot2.json` · `logging/runs/r3_pilot2/run.log` |
| R3-3 code / raw / log | `idea-stage/r3_pilot3_transduction_closure.py` · `idea-stage/r3_pilot3.json` · `logging/runs/r3_pilot3/run.log` |
| **C8 frozen rules (round 4)** | `idea-stage/PILOT_FREEZE_2026-08-09.md` §C8 |
| **C8 record / code / raw / log** | `idea-stage/C8_PROSODY_RESULT.md` · `idea-stage/c8_prosody_operator.py` · `idea-stage/c8_prosody.json` · `logging/runs/c8_prosody/run.log` |

---

## §7.10 — Round 4, pilot 1 (2026-08-09): **C8 prosody-as-operator → KILL**

Queue position 1 of §7.8, executed. Rules frozen in `idea-stage/PILOT_FREEZE_2026-08-09.md` §C8
before any implementation line was written; full record `idea-stage/C8_PROSODY_RESULT.md`, raw
`idea-stage/c8_prosody.json`, code `idea-stage/c8_prosody_operator.py`, log
`logging/runs/c8_prosody/run.log`. Single submission, CPU, **874 s**, zero test contact (guard armed,
**4** touched paths, none containing `test`).

**The question.** §7.4 argued Phase 1 measured the wrong estimand: a weak *marginal* audio signal is
compatible with a strong *conditional* one, because prosody modulates what a transcript means rather
than adding a fourth evidence stream — and if so the effect lives only where the text is genuinely
ambiguous, so a global average dilutes it away. This pilot measures that conditional estimand
directly.

**Design.** HateMM **train only**, 744 rows minus the **39** whitespace-transcript rows
(`refine-logs/EMPTY_TEXT_AUDIT_2026-08-09.md`, exclusion frozen in advance) → N = 705, base rate
0.4184. Three heads, 5-fold stratified OOF, 3 seeds: **M0** = text-only, **M1** = text ⊕ prosody
(the Phase-1 marginal), **M2** = M1 ⊕ an 8×8 bilinear text×prosody block. M2 ⊃ M1, so
`Δ_int = AUC_band(M2) − AUC_band(M1)` isolates the interaction. **Boundary band** = middle 30 % by
rank of M0's OOF probability (n = 211), computed from the text-only model so it is identical across
arms and placebos. **Placebo** = prosody permuted *within label strata*, whole pipeline re-run,
3 × 10 = 30 replicates.

**Two prosody representations, both pre-registered as gating, both named** (per the §7.4/§7.7-5 asset
correction — CLAP is cached for HateMM only): **arm P** = openSMILE eGeMAPSv02 (88-d, the literal
sense of "prosody"; same cache SHA as `APX_GATE_RECORD.md`), **arm C** = CLAP
`laion/larger_clap_general` `proj` (1024-d, the asset §7.8 budgeted). Whisper-large-v3 encoder
features were pre-registered as **excluded**: they carry lexical content, so a text×Whisper
interaction is partly text×text.

| gate quantity | arm P (eGeMAPS) | arm C (CLAP) | required | met |
|---|---|---|---|---|
| Δ_int per seed | −0.0532 / −0.0416 / −0.0360 | −0.0399 / −0.0439 / −0.0338 | | |
| mean Δ_int (band) | **−0.0436** | **−0.0392** | ≥ +0.010 | **no** |
| seeds positive | **0/3** | **0/3** | 3/3 | **no** |
| placebo P95 (n=30) | −0.0110 (mean −0.0577) | −0.0115 (mean −0.0518) | mean > P95 | **no** |

**KILL.** Per the freeze this closes the **audio-operator family** — prosody-as-operator, FiLM /
gating / bilinear audio conditioning, any successor whose mechanism is "audio modulates text" — on
this project's data. No re-run, no re-specification.

**Three things this bought beyond the kill.**

1. **§7.4's dilution prediction comes out inverted.** The interaction is *more* damaging inside the
   boundary band than outside it (P: −0.0436 vs −0.0284; C: −0.0392 vs −0.0104). The band is where
   the text signal is weakest, so it is where 64 extra parameters cost most and return least. The
   estimand argument was real and testable, and the data answered it the other way. This is a
   stronger result than "no effect": the region the hypothesis nominated is the region where the
   mechanism is worst.
2. **The audio channel's failure mode is redundancy, not weakness.** Read off the placebo runs (a
   descriptive comparison, not a gated endpoint): a *within-label-permuted* prosody vector adds
   **more** to a text head than the real one does — Δ_marg observed **+0.0031 / +0.0122** vs placebo
   mean **+0.0294 / +0.0448**. Within-label permutation preserves `P(prosody | label)` and destroys
   only the coupling to text, so a randomised vector carries the same label-marginal information
   while being conditionally independent of the transcript. Real prosody's label-relevant content is
   largely already in the transcript. This is a more specific statement than the standing "the audio
   prior is weak" and it explains why the marginal and the conditional framings fail together.
   Hedged: 3 seeds, one dataset, no null of its own.
3. **The pre-registered empty-transcript exclusion was load-bearing.** The labelled non-gating re-run
   on all 744 rows moves arm P from −0.0436 to **+0.0084 with 2/3 seeds positive** — most of the way
   to the bar, produced by 39 rows whose CLIP text vector is one constant point that is 92.3 %
   non-hate. It still would not have passed (+0.010, 3/3 required), so no verdict was at stake, but
   the artifact the freeze named in advance is visible in the numbers. Fourth confirmation of the
   audit's §2d finding.

**Machinery validation.** The marginal arm reproduces the project's existing audio record — eGeMAPS
+0.0031, CLAP +0.0122, against `APX_GATE_RECORD.md` best-k −0.0038 and `CLAP_GATE_RECORD.md` `proj`
best-k −0.0009 — so the setup is wired correctly. The synthetic positive control (planted
text×prosody interaction) returned **+0.3203**, 32× the bar; the label-permuted negative control
returned **−0.0013**. Both smokes ran before submission.

**Cumulative:** the audio axis is now **0-for-4** on HateMM — eGeMAPS marginal (F41/APX, 2026-07-16),
LAUD (F64), CLAP marginal + FN1 stratum (2026-07-27), and now the conditional/operator estimand. The
first three killed the marginal; §7.4 correctly identified that none of them had tested the
conditional. C8 tested it. Cross-round total: **41 candidates, 0 live mechanism candidates**, and the
one remaining queue item is C4, still blocked on its data-build.

---

## §8 — Round 4 (2026-08-10): new foundation, unsealed test set, ImpliHateVid

**Mandate.** Three things changed under the project between round 3 and round 4, and this round
exists to exploit them: (i) the 99-cell frozen ablation (`idea-stage/RGCL_ABLATION_RESULT.md`)
replaced the retrieval pipeline with **frozen features + a bare BCE head** as the baseline;
(ii) the user unsealed the **test set**, unlocking the transductive/TTA mechanism family that the
old red line had killed before rounds 1–3 could write it down; (iii) **ImpliHateVid** (2009 videos,
never attacked in three rounds) came into scope with an unspent implicit/explicit stratification.

**Funnel**: incremental landscape update → 4 fresh forensic recon measurements on the (now unsealed)
test sets → **14 candidates** generated cross-model → objective feasibility gate on disk state →
cross-model triage cut 14 → 8 runnable → **2 pilots authorised** (the jury explicitly declined the
third slot) → 1 deviation raised and ruled on before any primary result → see §8.6.

**Method.** Generation and triage by `gpt-5.6-sol` at `model_reasoning_effort: xhigh`, thread
`019fe784-eefa-7fd1-b53b-67753e528bc0`. Bundles: `idea-stage/codex_brainstorm_bundle_r4_2026-08-10.md`
(generation) and `idea-stage/codex_triage_bundle_r4_2026-08-10.md` (triage). Full candidate set with
all eight required fields: `idea-stage/section9_round4_candidates_2026-08-10.md`; triage verdict:
`idea-stage/triage_r4_verdict_2026-08-10.md`. **The executor eliminated nothing on quality grounds**;
the feasibility gate reported only facts about disk contents back to the jury, and the jury did all
the narrowing. Pilot rules frozen in `idea-stage/R4_PILOT_FREEZE_2026-08-10.md` **before any
implementation line was written**.

### 8.1 Landscape increment

Full notes with per-item verification levels: `idea-stage/phase1_landscape_r4.md` (33 web queries).
Only the four axes that changed were surveyed; prior recon was reused for everything else.

**(a) The transductive "set-structure" branch is NOT open — but its regime is.** Every sub-branch
has a top-venue owner: **TransCLIP** (2406.01837, **NeurIPS 2024 Spotlight**) is the canonical
"treat the whole unlabelled test set as a set" method; **ZLaP** (2404.04072, CVPR 2024) and ECALP
(2412.18303) own graph/label-propagation transduction; **SAT** (2411.17002) owns Sinkhorn/OT
assignment over the test batch; BBSE (1802.03916) owns test-prior estimation; 2509.04631 /
2605.01452 own transductive conformal. **But all of them assume many-class, text-anchored, roughly
balanced pools** — they import a class anchor from a text encoder, which a *binary policy label*
does not provide. Domain application remains essentially empty (only SCANNER, 2602.00132 AAAI 2026,
and it is centroid alignment).

**(b) The counter-literature is strong and must be answered by any TTA-flavoured candidate.**
*On Pitfalls of Test-Time Adaptation* (2306.03536, ICML 2023) — batch dependency wrecks model
selection, no method wins everywhere. **StatA / Realistic TTA** (2501.03729, **CVPR 2025**) — current
transductive methods "systematically compromise initial zero-shot robustness", and **TransCLIP
degrades when the number of effective classes drops**, which is exactly our 2-class regime. Plus
*The Illusion of Progress?* (2506.24000, NeurIPS 2025 D&B). Their convergent demand — **prove your
adaptation cannot damage the un-adapted model** — is currently unstated in the binary/moderation
regime.

**(c) ImpliHateVid source facts, verified.** arXiv **2508.06570**, **ACL 2025 Main Long**
(`2025.acl-long.842`), Rehman et al. 2,009 videos = 1000 NH / 509 IM / 500 EX; splits 1283/325/401,
matching our 92 EX + 108 IM + 201 NH exactly. Binary headline **87.53 acc / 87.73 F1** — **our bare
head's 0.9118 is above it**. They *do* report the 3-class breakdown, and it inverts the naive story:
3-class macro-F1 69.18 = **NH 84.48, IM 66.05, EX 57.02**; the live confusion is **EX↔IM**, not
hate↔non-hate. Implicit-hate mechanisms are text-only and crowded (DuPL WWW 2026; **HatePrototypes**
2511.06391, whose finding — prototypes are *interchangeable* between IM and EX — pre-empts "implicit
hate needs its own representation"; FiADD 2309.11896). **New to our map and the likeliest direct
competitor: DeHate, ACM MM 2025 (`10.1145/3746027.3758272`), explicitly on "explicit and implicit"
hateful video** — paywalled, so no design decision rests on its numbers.

**(d) "Simple beats complex" is already published, in-domain, on HateMM.** Koushik, Kanojia &
Treharne, arXiv **2502.07138** (MM4SG @ WebConf 2025): *"simple embedding fusion achieves
state-of-the-art performance on video content (HateMM) with a 9.9 pt F1 improvement."*
⇒ **the 99-cell ablation is a stronger version of an existing workshop finding, not a methods paper.**

**(e) What actually beats a frozen-feature linear probe at 10²–10³ examples**: LP++ (2404.02285,
CVPR 2024), CLAP (CVPR 2024), GDA (2402.04087), Tip-Adapter (2111.03930), LDA (2604.03928).
**Two patterns hold across all of them**: they are closed-form or hyper-parameter-search-free, and
they borrow a text class anchor. Neither transfers free to a binary policy label over heterogeneous
multimodal features.

**(f) No new SOTA 2026-06 → 08.** The sweep found benchmarks/audits/jailbreak work only. Frontier
unchanged: SAGE 0.8628 macro-F1 on HateMM. **Our bare head is at or past the published frontier on
two of four benchmarks** — a benchmark-validity fact, and §3's methods-only constraint forbids the
paper that would be made of it.

### 8.2 Fresh forensic recon (2026-08-10, on test, disclosed)

Run specifically to inform generation, using `idea-stage/r4_harness.py` — validated to reproduce the
ablation's bare-head cells (HateMM/CLIP 0.8013 vs 0.7993; ImpliHateVid/CLIP 0.9068 vs 0.9118;
HateMM/Qwen 0.8588 vs 0.8640). **These are measurements on the test set, disclosed as such; any
candidate built on them is confirmatory-by-construction and says so.**

**(1) The implicit axis is not where ImpliHateVid fails — the non-hate side is.** Per-subtype test
accuracy of the bare head, 3-seed mean:

| encoder | EX (n=92) | IM (n=108) | NH (n=201) | AUROC EX-vs-NH | AUROC IM-vs-NH |
|---|---|---|---|---|---|
| CLIP | 0.978 | 0.907 | 0.871 | 0.985 | 0.957 |
| Qwen | 1.000 | 0.917 | 0.876 | 0.980 | 0.950 |

Implicit hate is nearly as easy as explicit hate **in the binary task**. Of ~38 errors, ~25 are
**false positives on non-hate** and ~12 are missed hate. The hypothesis "accuracy is carried by the
explicit subset" is **false here**. (Not in tension with §8.1c: the 3-class EX↔IM confusion is a
different and harder problem, and is not our binary target.)

**(2) Decision-rule headroom is small; the remaining error is genuine ranking error.**

| cell | base @0.5 | val-tuned thr | train-prior quantile (transductive) | **oracle thr (upper bound)** |
|---|---|---|---|---|
| HateMM / LoRA | 0.8651 | 0.8632 | 0.8611 | 0.8804 (**+0.015**) |
| MHC-EN / Qwen | 0.7338 | 0.7347 | 0.7311 | 0.7727 (**+0.039**) |
| MHC-ZH / LoRA | 0.7864 | 0.8101 | 0.8218 | 0.8326 (**+0.046**) |
| ImpliHateVid / CLIP | 0.9068 | 0.9077 | 0.9160 | 0.9185 (**+0.012**) |

**Even a test-label oracle threshold buys only +1.2 to +4.6 points.** Threshold/calibration
mechanisms are capped there, and that framing is BBSE/Saerens — never-claim item 13.

**(3) Encoder errors are strongly complementary, and the complementarity is being wasted.**

| dataset | best single (F1/ROC) | mean-prob ensemble (F1/ROC) | error disjointness |
|---|---|---|---|
| HateMM | 0.8658 / 0.9315 | 0.8586 / **0.9333** | CLIP&Qwen both-wrong 16, only-CLIP 25, only-Qwen 12 |
| MHC-EN | 0.7302 / 0.8571 | 0.6891 / **0.8768 (+2.0 ROC)** | Qwen&LoRA both-wrong 28, only-Qwen 10, only-LoRA 11 |
| MHC-ZH | 0.8039 / 0.8983 | 0.7704 / **0.9175 (+1.9 ROC)** | CLIP&Qwen both-wrong 13, only-CLIP 14, only-Qwen 17 |
| ImpliHateVid | 0.9151 / 0.9699 | **0.9276** / **0.9745** | CLIP&Qwen both-wrong 20, only-CLIP 18, only-Qwen 14 |

**This was the sharpest live signal in the recon and it is what pilot R4-1 was built to test.**
Naive averaging converts encoder disjointness into **+1.5 to +2.0 ROC** on every dataset, but
macro-F1 *falls* on three of four because the averaged score's threshold is wrong — the **same
pathology as ablation §6** (ranking is learned, the decision rule throws it away). Ensemble +
prior-matched threshold beats the best single encoder on MHC-EN (+2.0) and ImpliHateVid (+1.8) and
loses on HateMM (−0.2) and MHC-ZH (−0.8): mean ≈ +0.7, inconsistent — **and it is a trivial
baseline, not a mechanism**.

### 8.3 Candidate table — all 14

Generated cross-model against the five mandated groups. Each candidate carried a **death-list check**
and an **absorption check** (why `2607.23304` kernel-ridge equivalence does not absorb it); a
candidate without those lines was to be discarded. Scores are the jury's pre-pilot judgements.

| # | candidate | group | mechanism in one line | jury (gen → triage) | disposition |
|---|---|---|---|---|---|
| F1 | **Monotone Disagreement Lattice (MDL)** | free | monotone lattice over per-encoder OOF logits, pinned to the val-best encoder where encoders agree, free to learn non-additive corrections only where they disagree | 7.4 → **6.8** | **PILOTED (R4-1)** |
| R1 | **Balanced Semantic Response-Tensor Distillation (B-SRTD)** — C4 revived | revival | distil the teacher's finite-difference Jacobian + mixed partial over *named* semantic interventions, not logits | 7.0 → n/a | **NOT PILOTABLE — asset build** |
| T1 | **Pool-Relative Evidence Sparsification (PRES)** | TTA×asset | use the unlabelled pool only to estimate a background over the 30 OCR windows; pool from the few windows with highest conditional surprisal | 6.6 → **2.0** | **REMOVED by feasibility** |
| R2 | Executable Accountability-Path Distillation (EAPD) — C6 revived | revival | typed agency graph; hate only when an attack path reaches an accountable *endorsing* agent | 6.4 | held — 330-video annotation build |
| F2 | Safe Covariate-Shift Rank Adaptation (SCRA) | free | worst-case target-weighted pairwise AUC subject to a certificate that it cannot underperform the deployed head | 6.1 | held — weeks of theory |
| I1 | Incomparable-Positive Partial-Order Head (IPPO) | ImpliHateVid | EX and IM as two *incomparable* positive strata; optimise only `NH<EX` and `NH<IM`, never EX-vs-IM | 5.8 → **4.2** | **REMOVED — one-dataset diagnostic** |
| B1 | **Jackknife Lower-Bound Rank Head (JLR)** | bare-head | pairwise objective on the leave-one-block-out *lower confidence bound* of each margin, discounting orderings supported by few items | 5.4 → **5.2** | **PILOTED (R4-2)** |
| T3 | Jury-Robust Global Safe Adaptation (JRSA) | TTA×asset | MHC vote fractions → a label-ambiguity polytope; adapt only when the candidate weakly dominates the bare head for every labelling in it | 5.3 | held — theory, may be vacuous |
| T2 | Test-Specified Metadata Nuisance Nulling (TMN) | TTA×asset | null the metadata-predictable feature directions whose prevalence shifts at deployment, preserving the class direction | 5.1 | held — needs metadata embeddings |
| B2 | Policy-Cone Discriminant Head (PCD) | bare-head | represent the binary policy as a *cone* of paired violation/safe-use clause directions rather than one prompt anchor | 5.0 → **4.8** | **best reserve** — must be pinned to the deployed head first |
| I2 | Shared-Hate Cone Head (SHC) | ImpliHateVid | one max-margin direction constrained to separate NH from *both* EX and IM | 4.7 → 3.4 | dead — one-dataset + hostile comparator |
| I3 | Cross-Fitted Non-Hate Veto (CNV) | ImpliHateVid | one-sided residual that may only *subtract*, trained on NH items the base head over-fires on | 4.4 → 4.0 | dead — reads as residual boosting |
| B3 | Negative-Tail CVaR Rank Head (NTC) | bare-head | CVaR over the worst hate/non-hate margins, targeting the high-scoring negative tail | 4.2 → 3.8 | dead — occupied by pAUC/CVaR work |
| R3 | Sparse Rank-Copula Pooling (SRCP) — C9 re-gated | revival | sparsify first, then soft within-video ranks → cross-stream copula instead of mean/max pooling | 3.8 | dead — K=30 caches HateMM-only |

**Revival verdict on the round-3 holds** (jury): **C4 → 7.0** (the only candidate whose death was
purely an asset failure; the bare head makes the student cheap and the Claude exemption makes the
lattice buildable). **C6 → 6.4** (§8.2's false-positive budget supplies the matching failure mode).
**C9 → below 4** (its K=30 arm is HateMM-only). **C7 stays dead** (order-sensitive parameters
unidentified in an 8.2 % multi-segment regime). **C2, C3, C11, C13, C14 stay dead** — the new
foundation reduces engineering cost but supplies none of their missing causal signal or novelty
defence.

### 8.4 Objective feasibility gate — what the disk actually contains

This is where the round's most decisive information came from: **two of the jury's own top three were
removed by disk facts, not by opinion.**

- **F1 MDL — feasible on all four datasets**, and it is the *only* top candidate that is. Encoder
  caches: CLIP + Qwen + LoRA on HateMM/MHC-EN/MHC-ZH; **CLIP + Qwen only on ImpliHateVid (no LoRA
  cache)**. ~180 head trainings ≈ 1.5–3 GPU-h at ~5–20 s per run.
- **T1 PRES — structurally single-dataset, and that dataset is the contaminated one.** OCR window
  vectors exist for **HateMM only** (`pilot_ocr_window_vecs.npz` 6565×768 train+val;
  `test_ocr_window_vecs.npz` 2111×768 test). **HateClipSeg cannot supply a second dataset**: it has
  windows (11,850 rows) but **no train/test split at all** — all 395 items are one partition — and
  its transcript channel is 395/395 constant. MHC-EN/MHC-ZH/ImpliHateVid have **no OCR cache**. Its
  decisive test-background-vs-train-background comparison therefore cannot exist twice. Jury:
  *"a diagnostic wearing a method's clothes"* — **removed, not demoted** (6.6 → 2.0).
- **I1 IPPO — one-dataset by construction.** ImpliHateVid is the only corpus in the project with a
  hate-subtype annotation. Combined with the fact that both its error target and its functional form
  came from §8.2's test recon, the jury ruled it *"a clean local diagnostic"*, not a methods-pilot
  (5.8 → 4.2).
- **R1 B-SRTD — not pilotable.** Re-verified today: `data/Counterfactual/MHC/train_twins.jsonl` =
  **168 records, every one `label=1`**; `MHC_zh` = **180 records, every one `label=1`**; one
  intervention axis (toxicity-sanitising transcript rewrite). The MVE's precondition — ≥200 train +
  80 val balanced 2×2 lattices over both classes and ≥2 intervention axes — **requires building an
  asset that does not exist**. Hours of Claude-API work plus human verification; a real and bounded
  expenditure, but not a pilot.
- **R3 SRCP**: K=30 CLIP subclip caches exist for HateMM only (MHC/MHC-ZH are K=4), so its
  cross-dataset arm is not buildable. **T2 TMN** needs a metadata embedding cache that does not exist.

### 8.5 The jury declined the third pilot

Verbatim: *"Pilot two candidates, not three: F1 MDL first, then B1 JLR. Do not spend the third slot
merely because it exists."* Reason: T1 and I1 cannot produce multi-dataset methods evidence; B3 and
I3 are recognisable occupied baselines; I2 has both the one-dataset and weak-comparator problems; and
B2 needs a mathematical design decision (attaching the policy cone to the deployed nonlinear
Hadamard-fusion head) before it can be frozen at all — *"the next legitimate action for B2 is a
paper-and-pencil specification plus novelty check, not a pilot."*

The `MAX_PILOT_IDEAS = 3` budget was therefore **under-spent by design**, on the jury's instruction,
and this is recorded rather than quietly back-filled.

### 8.6 Deviation D1 — the frozen R4-1 null was a false-KILL generator, caught in smoke

Record `idea-stage/R4_DEVIATION_D1_2026-08-10.md`; ruling `idea-stage/R4_DEVIATION_D1_RULING.md`;
amendment appended to the freeze. **Raised before any primary result existed**, by the mandatory
smoke test.

**Defect.** The frozen null said: *"within each hard-label stratum, independently permute every
non-reference encoder's logit rows … this preserves each encoder's class-conditional score
distribution and ROC while destroying item-level complementarity."* The first half is true; the
second is **the opposite of true**. Within-label permutation makes the permuted encoder's errors
*conditionally independent of the reference encoder's given the label* — the **best** case for score
combination, not the worst. It manufactures idealised complementarity.

| synthetic demo (n = 2000, correlated errors) | encoder A ROC | encoder B ROC | mean-of-two ROC |
|---|---|---|---|
| real | 0.7981 | 0.7885 | **0.8266** |
| B permuted within label | 0.7981 | 0.7885 (unchanged, as the freeze predicted) | **0.8785** |

Real-data confirmation, 2-rep smoke on MHC-ZH: null `DeltaROC` **+0.0549, +0.0692**. Since clause 2
demanded `MeanDeltaROC ≥ 3 × Null95`, the rule as written required a ~15-point ROC gain and would
have returned **KILL for every possible input** — the one defect class the project's process rules
say must block, because it produces a wrong verdict by construction.

**Jury ruling** (same thread; executor did not re-specify): (1) *"a false-KILL generator and a
blocking defect — do not apply it literally"*; (2) **no non-arbitrary permutation** can
simultaneously hold every encoder's class-conditional distribution fixed, destroy item-level
complementarity, and define a canonical "no-complementarity" distribution — complementarity lives in
the joint dependence structure, and independence / comonotonicity / rank-correlation preservation /
parametric copulas each impose a different substantive assumption. The jury **declined to replace one
arbitrary copula with another** and substituted a **paired stratified joint-row bootstrap**: 10,000
reps, `rng(20260810)`, dataset order fixed, positives before negatives, the *same* sampled joint rows
applied to MDL, the comparator and every seed; `LCB95 = quantile(boot, 0.05, "linear")`, no
truncation, no ×3. **Amended clause 2: `MeanDeltaROC ≥ +0.010` and `LCB95 > 0`.** Clauses 1/3/4 and
everything else unchanged.

**Partial unblinding, disclosed.** The smoke printed MHC-ZH seed-0 primary numbers (1 of 12 cells)
before the defect was identified; values are recorded verbatim in the deviation file. Jury: *"the
round survives with disclosure; no restart or scope change is required"* — the replacement is a
standard paired uncertainty calculation, is not tuned to that cell, and relaxes no clause. Required
handling was adopted: those predictions carried forward unchanged, remaining cells run exactly once,
and **per-cell test output suppressed in the log until every prediction and comparator choice was
saved** (only validation ROC was logged live).

### 8.7 Pilot R4-1 — F1 MDL (Monotone Disagreement Lattice): **KILL, 0 of 4 clauses**

Code `idea-stage/r4_pilot1_mdl.py`, raw `idea-stage/r4_pilot1.json`, log
`logging/runs/r4_pilot1_mdl/run.log`. Single submission, ~4 min GPU + 71 s bootstrap.
Smokes before submission: **positive control** (planted non-additive `min(a,b)` interaction) lattice
test ROC **0.9922** vs mean-logit 0.9519 vs best-single 0.8514 — the instrument recovers a real
interaction; **negative control** (permuted labels) **0.5085**; **monotonicity assertion** passes on
both axes.

Full test table, 3 seeds, all comparators reported (frozen comparator per dataset in bold context):

| dataset (encoders) | single | mean-logit | mean-prob | weighted | logistic | MLP | **MDL** |
|---|---|---|---|---|---|---|---|
| **HateMM** (C+Q+L) ROC | 0.9303 | 0.9359 | *0.9338* | 0.9364 | 0.9275 | 0.9326 | **0.9350** |
| macro-F1 | 0.8598 | 0.8722 | *0.8600* | 0.8672 | 0.8654 | 0.8732 | **0.8738** |
| **MHC-EN** (C+Q+L) ROC | 0.8540 | 0.8820 | *0.8771* | 0.8818 | 0.8706 | 0.8646 | **0.8782** |
| macro-F1 | 0.7358 | 0.7776 | *0.7702* | 0.7434 | 0.7334 | 0.7272 | **0.7453** |
| **MHC-ZH** (C+Q+L) ROC | 0.8837 | 0.9175 | 0.9172 | *0.9175* | 0.9200 | 0.9236 | **0.9158** |
| macro-F1 | 0.7655 | 0.7971 | 0.8120 | *0.8009* | 0.8183 | 0.8013 | **0.8059** |
| **ImpliHateVid** (C+Q) ROC | 0.9697 | *0.9753* | 0.9740 | 0.9752 | 0.9743 | 0.9743 | **0.9746** |
| macro-F1 | 0.9093 | *0.9226* | 0.9226 | 0.9226 | 0.9276 | 0.9276 | **0.9259** |

*(italic = the frozen comparator for that dataset, selected on mean validation ROC before any test
metric was read: mean-prob / mean-prob / weighted / mean-logit.)*

| dataset | DeltaROC | DeltaF1 |
|---|---|---|
| HateMM | +0.0012 | +0.0138 |
| MHC-EN | +0.0011 | **−0.0248** |
| MHC-ZH | −0.0017 | +0.0050 |
| ImpliHateVid | −0.0007 | +0.0033 |
| **mean** | **−0.0000** | **−0.0007** |

Paired stratified joint-row bootstrap: mean −0.00001, **LCB95 = −0.00253**.

| clause | required | observed | met |
|---|---|---|---|
| 1 | `DeltaROC_MHC-EN ≥ +0.010` | +0.0011 | **no** |
| 2 | `MeanDeltaROC ≥ +0.010` and `LCB95 > 0` | −0.0000, LCB95 −0.00253 | **no** |
| 3 | ≥3 of 4 `DeltaROC_d` positive, none < −0.005 | 2 of 4 positive | **no** |
| 4 | `MeanDeltaF1 ≥ +0.010`, none < −0.005 | −0.0007; MHC-EN −0.0248 | **no** |

**KILL — 0 of 4 clauses.** Per the freeze this closes the disagreement-lattice mechanism.

**What the kill actually establishes, and it is the more useful half.** The lattice landed on top of
the trained comparators to within ±0.002 ROC on every dataset, while the *instrument* demonstrably
recovers a planted non-additive interaction (+0.041 ROC over mean-logit in the positive control).
The most economical reading, which is also the failure mode the jury predicted pre-pilot:
**the cross-encoder complementarity measured in §8.2(3) is real but essentially additive.** A
monotone non-additive surface over the encoder logits finds nothing that a plain average has not
already extracted. The remaining macro-F1 gap is therefore not an unexploited disagreement geometry.

**The descriptive finding survives the kill, and is stronger than expected.** Every ensemble
comparator beat the validation-best single encoder by a wide margin on test:

| dataset | best single | best ensemble comparator | ROC gain | macro-F1 gain |
|---|---|---|---|---|
| HateMM | 0.9303 / 0.8598 | 0.9364 (weighted) / 0.8732 (MLP) | +0.0061 | **+0.0134** |
| MHC-EN | 0.8540 / 0.7358 | 0.8820 (mean-logit) / 0.7776 (mean-logit) | **+0.0280** | **+0.0418** |
| MHC-ZH | 0.8837 / 0.7655 | 0.9236 (MLP) / 0.8183 (logistic) | **+0.0399** | **+0.0528** |
| ImpliHateVid | 0.9697 / 0.9093 | 0.9753 (mean-logit) / 0.9276 (logistic/MLP) | +0.0056 | **+0.0183** |

This is a **trivial baseline, not a mechanism**, and §3's constraint forbids the paper — but it is a
real, multi-seed, test-reported effect that any future candidate in this project now has to clear.
It also revises §8.2(3)'s pessimism: the macro-F1 loss seen there came from averaging *at a fixed
0.5 threshold*; once each method picks its threshold on validation (as the frozen protocol requires),
ensembling converts its ROC advantage into macro-F1 on all four datasets.

### 8.8 Pilot R4-2 — B1 JLR (Jackknife Lower-Bound Rank Head): **KILL, 1 of 4 clauses**

Code `idea-stage/r4_pilot2_jlr.py`, raw `idea-stage/r4_pilot2.json`, log
`logging/runs/r4_pilot2_jlr/run.log`. Single submission, chained to start automatically after R4-1
so the fold machinery is shared, ~20 min GPU. Four pre-declared cells, 3 seeds, five heads, sd
coefficient 1.0, BCE weight 0.1 — all fixed, no grid.

Full test table, all comparators reported (*italic* = frozen comparator, selected on mean validation
ROC before any test metric was read):

| cell | **JLR** | ens_pair_sd0 | ens_bce | single_pair | single_bce |
|---|---|---|---|---|---|
| **HateMM/LoRA** ROC | 0.9310 | *0.9322* | 0.9226 | **0.9328** | 0.9248 |
| macro-F1 | **0.8785** | *0.8727* | 0.8513 | 0.8650 | 0.8724 |
| **MHC-EN/Qwen** ROC | 0.8692 | **0.8717** | *0.8560* | 0.8707 | 0.8540 |
| macro-F1 | 0.7396 | 0.7377 | *0.7399* | 0.7368 | 0.7358 |
| **MHC-ZH/LoRA** ROC | 0.9177 | **0.9203** | 0.9123 | *0.9211* | 0.9096 |
| macro-F1 | **0.7975** | 0.7823 | 0.7778 | *0.7607* | 0.7466 |
| **ImpliHateVid/CLIP** ROC | 0.9711 | *0.9716* | 0.9708 | **0.9717** | 0.9697 |
| macro-F1 | 0.9085 | *0.9102* | 0.9093 | 0.9093 | 0.9093 |

| cell | DeltaROC | DeltaF1 | frozen comparator |
|---|---|---|---|
| HateMM | −0.0011 | +0.0058 | ens_pair_sd0 |
| MHC-EN | **+0.0131** | −0.0003 | ens_bce |
| MHC-ZH | −0.0034 | +0.0367 | single_pair |
| ImpliHateVid | −0.0005 | −0.0017 | ens_pair_sd0 |
| **mean** | **+0.0020** | **+0.0101** | |

| clause | required | observed | met |
|---|---|---|---|
| 1 | `DeltaROC_MHC-EN ≥ +0.010` **and** `≥ 3 × Null95` | +0.0131 ✓ but Null95 = 0.0272 → bar +0.0816 | **no** |
| 2 | `MeanDeltaROC ≥ +0.010` | +0.0020 | **no** |
| 3 | ≥3 of 4 `DeltaROC_d` positive, none < −0.005 | **1 of 4** positive | **no** |
| 4 | `MeanDeltaF1 ≥ +0.005`, none losing > 0.005 | +0.0101, min −0.0017 | **yes** |

**KILL — 1 of 4 clauses.**

**The decisive number is the sd-coefficient ablation, and it is unambiguous.** `ens_pair_sd0` is the
*identical* five-head pairwise ensemble with the jackknife standard-deviation coefficient set to
zero — i.e. the mechanism's defining component removed. On test ROC:

| cell | JLR (sd = 1.0) | ens_pair_sd0 (sd = 0) | JLR − sd0 |
|---|---|---|---|
| HateMM | 0.9310 | 0.9322 | **−0.0012** |
| MHC-EN | 0.8692 | 0.8717 | **−0.0025** |
| MHC-ZH | 0.9177 | 0.9203 | **−0.0026** |
| ImpliHateVid | 0.9711 | 0.9716 | **−0.0005** |

**The lower-confidence-bound term is a small, consistent drag in 4 of 4 cells.** Every point JLR
gains over a single BCE head comes from the pairwise objective and the five-head ensemble, not from
the stability discount. This is exactly the pre-declared explicit KILL condition ("a gain over a
single BCE head that vanishes against either five-head comparator is a KILL") and exactly the failure
mode the jury predicted pre-pilot: *"the standard-deviation term will either collapse diversity among
the five heads or add nothing beyond an identically sized BCE ensemble."* It also answers the
underlying hypothesis directly and negatively: **unstable train-pair ordering is not what limits the
frozen-feature head — ordinary model variance is.**

**Null caveat, recorded, verdict does not turn on it.** R4-2's null is the same within-hard-label
permutation whose defect deviation D1 established; the D1 ruling was explicitly scoped to R4-1, so
the R4-2 rule was applied **literally and unchanged**, and it inflated identically (`Null95` = 0.0272,
i.e. all-positive null deltas, running mean +0.017). **This changes nothing**: clause 1's *first*
conjunct passes on its own (+0.0131 ≥ +0.010) and clauses 2 and 3 fail with no null involved
(`MeanDeltaROC` +0.0020 < +0.010; 1 of 4 positive). The R3-2 precedent therefore applies — the
literal rule stands because the verdict cannot turn on it. **Any future use of this null class in the
project is closed by D1 regardless.**

**By-product worth keeping: a pairwise objective beats BCE on ranking in 4 of 4 cells.** Independent
of the killed mechanism, `single_pair` (one head, pairwise-AUC + 0.1 BCE) beats `single_bce` (the
project's current baseline head) on test ROC everywhere:

| cell | single_bce ROC | single_pair ROC | gain |
|---|---|---|---|
| HateMM/LoRA | 0.9248 | 0.9328 | **+0.0080** |
| MHC-EN/Qwen | 0.8540 | 0.8707 | **+0.0167** |
| MHC-ZH/LoRA | 0.9096 | 0.9211 | **+0.0115** |
| ImpliHateVid/CLIP | 0.9697 | 0.9717 | **+0.0020** |

Consistent in sign and non-trivial in size on the two hardest datasets. Like the ensemble finding in
§8.7 this is **a baseline upgrade, not a mechanism** (pairwise/AUC losses are standard machinery, and
the project's never-claim discipline applies), but it re-bases future comparisons the same way.

### 8.9 Ranking and survivors

| rank | candidate | gen → triage → post-pilot | status |
|---|---|---|---|
| 1 | **R1 B-SRTD** (C4 revived) | 7.0 → n/a (not pilotable) | **UNFUNDED, still not disproven** — the only candidate across four rounds never killed by a mechanism failure, an occupant or a null. Blocker is a bounded data build. |
| 2 | **R2 EAPD** (C6 revived) | 6.4 | **held** — needs a 330-video annotation build; newly motivated by §8.2(1)'s false-positive budget |
| — | **F2 SCRA** | 6.1 → **CLOSED** | **VACUOUS + OCCUPIED** (`idea-stage/SCRA_THEORY_MEMO.md`, 2026-08-10). Certificate slack 0.32–1.45 AUC vs a 0.008–0.017 prize; no measurable train/test covariate shift on any of the four datasets (domain-classifier AUC 0.42–0.56, MMD p 0.17–0.96); mechanism occupied by TCPR (Kouw & Loog) and UMVP (Li/Zha/Zhou AAAI'16, worst-case AUC gain over baseline). Frozen rules R2 3/4, R3 4/4. |
| — | **F1 MDL** | 7.4 → 6.8 → **CLOSED** | piloted R4-1: 0 of 4 clauses; the complementarity is additive |
| — | **B1 JLR** | 5.4 → 5.2 → **CLOSED** | piloted R4-2: 1 of 4 clauses; the sd term loses to its own sd=0 ablation in 4/4 |
| 6 | **B2 PCD** | 5.0 → 4.8 | **best reserve**, but needs a paper-and-pencil spec + novelty check before it can be frozen |
| 7 | T3 JRSA · T2 TMN | 5.3 · 5.1 | held; theory / missing embedding cache |
| — | **T1 PRES** | 6.6 → **2.0** | **removed by disk facts** — structurally one-dataset, on the contaminated dataset |
| — | **I1 IPPO** | 5.8 → **4.2** | removed — one-dataset diagnostic |
| — | I2 SHC · I3 CNV · B3 NTC · R3 SRCP | 4.7→3.4 · 4.4→4.0 · 4.2→3.8 · 3.8 | dead |

**Survivors of the pilots: zero.** Both piloted candidates were killed by rules frozen before
implementation. **Cumulative across four rounds: 55 candidates generated, 0 live mechanism
candidates.**

### 8.10 What this round actually bought

Beyond the two kills, the transferable findings:

1. **The cross-encoder complementarity is real and it is additive.** §8.2(3) measured +1.5–2.0 ROC
   of disjoint-error information; R4-1 showed a monotone non-additive surface extracts *nothing*
   beyond a plain average (±0.002 ROC on every dataset), with a validated instrument. Any future
   candidate proposing to "exploit encoder disagreement" in this project now has a measured negative
   to clear.
2. **The trivial ensemble baseline is much stronger than the project's single-encoder baselines** —
   +1.3 to +5.3 macro-F1 over the validation-best single encoder on all four datasets, multi-seed,
   test-reported. It is not a mechanism and §3 forbids the paper, but it **re-bases every future
   comparison**: a candidate that beats the bare head but not a three-encoder average has shown
   nothing.
2b. **A pairwise/AUC objective beats BCE on ranking in 4 of 4 cells** (+0.008 / +0.017 / +0.012 /
   +0.002 test ROC over the project's current single-BCE head; §8.8). Standard machinery, so no
   novelty is claimed — but combined with (2) it means the project's *baseline* is now
   "three-encoder ensemble of pairwise-trained heads", not "one BCE head", and every future
   candidate is measured against that.

3. **The implicit-hate framing for ImpliHateVid is empirically false in the binary task.** IM-vs-NH
   AUROC is 0.95–0.96 and implicit recall is 0.91; ~2/3 of the error budget is **false positives on
   hate-adjacent non-hate**. Any "we solve implicit hate" story on this corpus is unsupported. The
   source paper's EX↔IM confusion (57.0 vs 66.1 3-class F1) is a *different, harder* task that is not
   our binary target.
4. **The decision-rule ceiling is measured**: a test-label *oracle* threshold buys only +1.2 to +4.6
   macro-F1 points. Calibration mechanisms are capped there — and that ceiling was reached in
   practice, since once each method picks its threshold on validation, ensembling already converts
   its ROC advantage into macro-F1 on all four datasets.
5. **A null can be wrong in the conservative direction and still be fatal.** D1 is the round's most
   transferable process finding: within-hard-label permutation, the obvious "destroy the coupling"
   null, actually *manufactures* the conditionally-independent ideal case. Any future permutation
   null in this project must be checked against a planted-signal control **and** a
   no-signal control before it gates anything — the R3-2 precedent (apply the literal rule and note
   the ambiguity) only holds when the verdict cannot turn on it.
6. **PRES's removal is a standing asset fact, not a one-round judgement**: there is exactly one
   dataset in this project with split-level OCR windows, and HateClipSeg can never be the second one
   because it has no train/test split at all. Any future OCR-window mechanism inherits this.

### 8.11 Honest bottom line for round 4

> **Round 4 produced no viable main-conference methods candidate.** Both authorised pilots were
> killed by rules frozen before implementation, and the two highest-scoring ideas of the round
> (B-SRTD 7.0, EAPD 6.4) were never pilotable because the assets they need do not exist.

The round did what it was asked to do: the three changed conditions were genuinely exploited (the
bare-head foundation produced 3 native candidates, the unsealed test set produced 3 transductive
ones, ImpliHateVid produced 3), the jury flagged no candidate as a death-list re-skin, and the two
things that died, died on evidence rather than on novelty. But the two most promising candidates
across rounds 3 and 4 are the *same* candidate — C4/B-SRTD — and it has now been blocked on the same
missing asset twice.

**The single most defensible next expenditure remains the B-SRTD intervention lattice**: build a
balanced two-axis (target substitution × endorsement/condemnation reversal) counterfactual lattice
over **both** classes, ≥200 train + 80 val, human-verify a sample, and only then run the
response-tensor gate. The Claude-API frame/text exemption makes this hours, not GPU-days. It is a
*data-building* bet, and it should only be funded if someone will commit to the human verification —
without it, round 2's rejection reason ("the capability you advertise is an evaluation artefact")
applies immediately.

### 8.12 Coverage and process notes (so the verdicts are not over-read)

- **Confirmatory-by-construction.** Both piloted candidates were motivated by measurements taken on
  these same test sets earlier the same day (§8.2). The jury ruled this *survivable for a disclosed
  pilot* but explicitly **not** sufficient for a paper claim. Since both came out KILL, the bias
  direction was favourable to the candidates and they still failed — which makes the kills *stronger*
  than they would otherwise be, not weaker.
- **Partial unblinding, disclosed** (§8.6): MHC-ZH seed 0 of R4-1 (1 of 12 primary cells) was seen
  during the smoke test before deviation D1 was identified. Verbatim values are in
  `idea-stage/R4_DEVIATION_D1_2026-08-10.md` §4. No threshold, bar, comparator rule or mechanism
  definition was changed as a result; the null replacement was specified by the cross-model jury, not
  by the executor.
- **Test-set protocol.** Per the user's 2026-08-09 ruling, test inputs and labels were used for
  one-shot pre-registered evaluation and for disclosed diagnostics. Both pilots were single
  submissions with all cells reported. No "look at test → change design → look again" loop was run
  on either pilot; the recon in §8.2 preceded candidate generation and is labelled as such.
- **Generation and triage were cross-model** (`gpt-5.6-sol`, xhigh, thread
  `019fe784-eefa-7fd1-b53b-67753e528bc0`). **The executor eliminated nothing on quality grounds** —
  the feasibility gate reported only objective disk facts back to the jury, and the jury did all the
  narrowing, declined the third pilot slot, and ruled on the deviation.
- **The novelty probe is inherited, not fresh for every candidate.** The landscape update spent 33
  web queries on four axes; per-candidate novelty checks were *not* run, because the jury killed both
  pilots on empirical gates rather than on novelty. **No KILL in this round rests on a novelty
  verdict.** Stated blind spots from the surveyor: no non-English search; DeHate / HCG-MPB / TIHD /
  DuPL PDFs paywalled (**DeHate, ACM MM 2025, is the priority fetch** — it is the likeliest direct
  competitor on ImpliHateVid); ACM MM 2026 / EMNLP 2026 / NeurIPS 2026 accept-lists not indexed; ~20
  landscape items are marked `[C]` (id + title seen in a result page, not opened) and must be
  verified before appearing in a paper.
- **Launcher defect, recorded** (third occurrence of this class in the project): the R4-1 launch
  chained `mkdir && … && setsid nohup … & echo $! > run.pid`, so the pid write raced the backgrounded
  chain and failed. **Only one instance ever ran** (verified by `pgrep`); the pid file was written
  post-hoc from `pgrep`. No output was affected, but the pattern that produced the R3-1 duplicate
  launch is still in use and should be replaced with a launcher script.
- **Harness fidelity.** `idea-stage/r4_harness.py` selects the epoch on **validation macro-F1 of the
  head**, whereas `src/run_rac.py` selects on the *retrieval* metric. This is declared rather than
  hidden: absolute numbers are near, not byte-identical, to the ablation table (HateMM/CLIP 0.8013 vs
  0.7993; ImpliHateVid/CLIP 0.9068 vs 0.9118; HateMM/Qwen 0.8588 vs 0.8640), and every pilot verdict
  is a **seed-paired delta computed inside the harness**, so the comparison is internally same-frame.

**Reproducibility index for §8**

| artifact | path |
|---|---|
| landscape increment | `idea-stage/phase1_landscape_r4.md` |
| generation bundle | `idea-stage/codex_brainstorm_bundle_r4_2026-08-10.md` |
| all 14 candidates, 8 fields each | `idea-stage/section9_round4_candidates_2026-08-10.md` |
| triage bundle / verdict | `idea-stage/codex_triage_bundle_r4_2026-08-10.md` · `idea-stage/triage_r4_verdict_2026-08-10.md` |
| **frozen pilot rules (+ D1 amendment)** | `idea-stage/R4_PILOT_FREEZE_2026-08-10.md` |
| deviation D1 record / ruling | `idea-stage/R4_DEVIATION_D1_2026-08-10.md` · `idea-stage/R4_DEVIATION_D1_RULING.md` |
| shared pilot harness | `idea-stage/r4_harness.py` |
| R4-1 code / raw / log | `idea-stage/r4_pilot1_mdl.py` · `idea-stage/r4_pilot1.json` · `logging/runs/r4_pilot1_mdl/run.log` |
| R4-2 code / raw / log | `idea-stage/r4_pilot2_jlr.py` · `idea-stage/r4_pilot2.json` · `logging/runs/r4_pilot2_jlr/run.log` |

### 8.13 Addendum (2026-08-10, later same day) — B2 PCD closed at the specification stage

§8.5 recorded the jury's instruction that *"the next legitimate action for B2 is a paper-and-pencil
specification plus novelty check, not a pilot."* That action was executed. Record:
**`idea-stage/PCD_SPEC.md`**. **Verdict: DEAD. No pilot was run; no test file was opened.**

**The jury's mathematical question is answered** (§1 of the spec): the cone **cannot** live inside
the deployed head — `x = i ⊙ t` and the 3-layer MLP are *learned* spaces in which a policy sentence
has no image, and any map into them would have to be fitted, destroying the "direction fixed before
training" property that defines a policy clause. The cone must live in the frozen CLIP joint space
and compose with the head **at the logit** (additive policy logit). A **new standing asset fact**
falls out: applying CLIP's frozen `visual_projection`/`text_projection` to the stored
pre-projection poolers recovers the 768-d joint space at zero extraction cost (generic hate anchor,
no training: **val ROC 0.845 on ImpliHateVid**) — but **Qwen/LoRA `text_feats` is the mean of the
last 4 tokens of the assistant generation-prompt tail inside a multimodal prompt, not a
sentence-embedding space**, so *no* text-anchored mechanism can be built on three of the four
val-best cells without a new extraction pass.

**Cause 1 — occupied.** LaBo (CVPR 2023) is a trained head whose weights are a non-negative
combination of frozen text-clause directions over frozen VLM features, clause set fixed in advance,
random-concept ablation included. **Zero-Shot Image Moderation in Google Ads (WSDM 2025,
2412.16215)** turns written policy into violating *and* matched non-violating clause embeddings in
a frozen multimodal space and decides on the violating-minus-safe margin — PCD's exemption
arithmetic, deployed. RoboShot (ICLR 2024) has paired-prompt difference directions plus an explicit
suppression operator; Hypothesis Engineering (COLING 2022) has hand-written exemption clauses over
a frozen model overriding the hate score on exactly this failure mode; Rule By Example (ACL 2023)
encodes hand-written moderation rules as vectors for hate classification. Polyhedral Conic
Classifiers (CVPR 2017/TPAMI 2020) own the geometry and the name; Concept Cones (ICML 2025) owns
the framing sentence. Two of the candidate's stated mathematical claims are also wrong: non-negative
weights alone give a **halfspace**, not a cone, and with the per-clause ReLU the model is a
one-hidden-layer ReLU net with a frozen first layer and non-negative output layer.

**Cause 2 — the premise is empirically false, measured on train/val only.** Matched violation and
exemption clauses embed at cosine **0.920** (CLIP joint space) and **0.833 / 0.869** (multilingual
mpnet, EN / ZH); the `K` pair-difference directions are mutually near-orthogonal (**0.035–0.067**),
so there is no shared "violation vs safe use" axis. Consequently, with a trained readout on train
and ROC read on val, the clause directions **lose to dimension-matched random directions on 3 of 4
datasets** (0.804/0.648/0.722/0.581 vs random 0.841/0.756/0.645/0.697 over 5 draws; **mean −0.046**),
and the exemption term is worth **≈ +0.005 with a sign flip** across six measurements. Both hold in
a genuine sentence-embedding space, so this is not a CLIP artefact. This is the pre-declared KILL
condition ("if the gain survives randomisation the clause semantics contribute nothing") met with
the gain not merely surviving but **negative**.

**Transferable, beyond this candidate:** (i) `2K` *random* projections of the frozen joint feature
are a strong baseline (val ROC up to 0.88 on ImpliHateVid, 0.82 on HateMM) with wide across-draw
spread — any future "K interpretable directions" candidate must beat a random-direction arm
**averaged over several draws**; (ii) the frozen text encoders in this project do not represent
policy exemptions at all, so separating hate from condemnation/quotation/reclaimed use must come
from a model that reasons, not from an embedding direction.

**Round-4 survivor count is unchanged at zero.** B2 was the last reserve in §8.9; ranks 1–3
(R1 B-SRTD, R2 EAPD, F2 SCRA) remain the only unfalsified candidates, all three blocked on builds
rather than on evidence.

---

## §9 — Round 5 (2026-08-10): training-level mechanisms — **HALT, zero GPU spent**

**Mandate (user, fixed and non-negotiable).** Four existing datasets only, video-level binary
classification only, and **the mechanism must move training or representation** — LoRA objective,
LoRA target, how the MLLM is used, encoder-level input integration, generative decision. Head-level
and fusion-level mechanisms are forbidden this round (round 4 exhausted that space). Hard exclusion:
anything of the form *"learn something else on top of frozen features"*.

**Outcome.** Phase A found a large prize pool and a single dominant error mode. Phase B generated
10 training-level candidates. **The cross-model jury scored all ten DEAD and instructed: pilot
nothing, spend zero GPU-hours.** Phase C was therefore not executed. The round's deliverable is
the prize-pool map, the error taxonomy, four new measurements, a frozen conditional
pre-registration, and **one decision escalated to the user**.

Triage bundle: `idea-stage/codex_triage_bundle_r5_2026-08-10.md`. Jury: `gpt-5.6-sol`,
`model_reasoning_effort: xhigh`, thread `019feb84-40e2-7253-826a-de98f6fbe98a`.

### 9.1 Phase A1 — the prize pool is real and annotation noise is not the binding constraint

MultiHateClip ships **per-annotator vote lists** (`data/gt/mhc_votes/`), never used before in this
project. 2.18 (EN) / 2.26 (ZH) annotators per item; binary positive = Hateful ∪ Offensive. The
majority of those votes reproduces the project's binary label on **100 %** of test ids in both
languages, so the vote file and the training labels are the same object — the ceiling below is
computed on the exact labels we report against.

| | pairwise raw agreement | Krippendorff α (binary) | split-vote rate | 1-annotator-vs-rest macro-F1 | **panel-resample ceiling** (test split) | base (§8.10) | **prize** |
|---|---|---|---|---|---|---|---|
| MHC-EN | 0.817 | 0.803 | 12.3 % | 0.928 | **0.9276** (p05 0.896) | 0.7776 | **+15.0** |
| MHC-ZH | 0.781 | 0.772 | 16.1 % | 0.908 | **0.9387** (p05 0.909) | 0.8183 | **+12.0** |

The panel-resample ceiling is the macro-F1 that a *perfect* predictor of one annotator panel scores
against an independently resampled panel — the strongest defensible upper bound available from 2–4
votes. HateMM and ImpliHateVid ship no per-annotator votes; their headroom to a perfect predictor
is +12.7 and +7.2.

> **Every dataset has ≥ 7 macro-F1 points of genuinely purchasable headroom.** The Phase-A stop
> rule ("if the prize pool is < 2 points, report and stop") was cleared by a factor of 3.5 on the
> tightest dataset. Annotation noise is *not* what limits this project.

Code `idea-stage/r5_phase_a.py`, `r5_phase_a3.py`; raw `r5_phase_a.json`, `r5_phase_a3.json`.

### 9.2 Phase A2 — the error taxonomy: one bucket dominates

All **108** test errors of the round-4 best ensemble comparator per dataset (HateMM/MLP,
MHC-EN/mean-logit, MHC-ZH/logistic, ImpliHateVid/logistic) were reconstructed exactly — the
per-method validation-selected threshold was recovered by inverting the stored `test_macro_f1`
over the test-score grid, and the recovered macro-F1 reproduces §8.7 to 4 decimal places on all
four datasets (0.8732 / 0.7776 / 0.8183 / 0.9276). Every error was then read individually (full
transcript, plus the video-level OCR cache on HateMM) and coded **before any count was taken**.

Buckets: **S** stance / use-vs-mention · **O** decisive evidence is burned-in on-screen text ·
**M** transcript empty or music-only · **A** annotators split or label conflicts with the material ·
**D** item named in this project's own train↔test audit as duplicate/degenerate · **X** ordinary
ranking error. Coding in `idea-stage/r5_buckets.json`; evidence dump `r5_error_dump.json`.

| dataset | n err | S | O | M | A | D | X | **oracle-fix S** | oracle-fix S+O+M |
|---|---|---|---|---|---|---|---|---|---|
| HateMM | 26 | 8 | 5 | 1 | 1 | 2 | 9 | **+3.47** | +6.35 |
| MHC-EN | 31 | 16 | 0 | 2 | 5 | 0 | 8 | **+10.48** | +11.99 |
| MHC-ZH | 24 | 12 | 0 | 2 | 1 | 0 | 9 | **+8.90** | +10.31 |
| ImpliHateVid | 27 | 13 | 0 | 0 | 2 | 1 | 11 | **+3.00** | +3.00 |
| **total** | 108 | **49 (45.4 %)** | 5 (4.6 %) | 5 (4.6 %) | 9 (8.3 %) | 3 (2.8 %) | 37 (34.3 %) | **mean +6.46** | mean +7.91 |

**The stance bucket is 45 % of all errors and is worth a mean +6.5 macro-F1** — five times the
combined value of the on-screen-text and silent-video buckets, and larger than the entire
decision-rule ceiling measured in §8.2(2). Representative members, so the coding can be judged:

- HateMM false positives: a John Lennon protest song containing a slur; 1950s archival footage of
  a segregationist committee; a video exposing a named neo-Nazi (counter-speech); the Ellen show
  performing a satirical song.
- ImpliHateVid false positives: a monologue *arguing against* using racial slurs (and therefore
  mentioning one); a satirical sketch about sexism; commentary about Palestine and gay rights.
- False negatives: a CTV news *report* about international students, labelled implicit hate;
  a news report about a child told "why didn't you stay in Mexico", labelled hate.

This is the same failure the project already characterised from the other direction in §8.13:
*"separating hate from condemnation/quotation/reclaimed use must come from a model that reasons,
not from an embedding direction."* Phase A now puts a price on it.

### 9.3 Phase A3 — MHC errors concentrate 5–6× on items the annotators themselves split

| | split-vote rate, test set | split-vote rate, error set | odds ratio |
|---|---|---|---|
| MHC-EN | 26/161 = 16.1 % | 12/31 = 38.7 % | **5.23** |
| MHC-ZH | 20/149 = 13.4 % | 9/24 = 37.5 % | **6.22** |

Oracle-fixing only the split-vote errors buys +8.0 (EN) / +7.0 (ZH); fixing only the unanimous
errors buys +13.7 / +11.3. So the residual error is *not* mostly irreducible ambiguity — but the
enrichment is real and any candidate should expect its remaining errors to sit there.

### 9.4 Phase A4 — new measurement: **OCR is redundant with ASR on MultiHateClip**

PaddleOCR K=30 was run on both MHC test splits for the first time (161 + 149 videos, 15 min wall,
`logging/runs/r5_ocr_mhc/run.log`; caches `data/OCR/MHC_test/`, `data/OCR/MHC_zh_test/`;
`scripts/ocr_cache/extract_ocr_windows.py` extended with MHC splits). 95 % (EN) / 99 % (ZH) of
videos carry ≥ 20 characters of on-screen text — but on inspection of **every** error item the text
is burned-in captioning of the speech already in the transcript, plus uploader watermarks.
**Not one MHC error is decidable from on-screen text the transcript lacks.** On HateMM the opposite
holds: MEMRI-TV translated subtitles, Britain First title cards, and a slur that appears only in the
burned-in text of a video whose transcript is empty.

> **Standing asset fact (extends §8.10 item 6):** the on-screen-text channel is *complementary* on
> HateMM and *redundant* on MultiHateClip. Any future OCR mechanism in this project is a
> one-dataset mechanism on evidentiary grounds, not merely on cache-availability grounds.

### 9.5 Phase B — feasibility constraints that killed candidates before scoring

- **K1. ImpliHateVid has no raw video on this machine** (cloud backup only; `data/video/ImpliHateVid/`
  contains an id→path TSV and nothing else). **Every training-level or encoder-level candidate is
  at most 3 of 4 datasets.**
- **K2. The GPU is not available.** One RTX 5090 (32 GB), shared; another user's job held 21–28 GB
  at 97 % utilisation throughout this session. ~5 GB free — a 7B MLLM does not fit even in 4-bit.
- **K3. The LoRA training stack is not on disk.** `RA-HMD/LLAMA-FACTORY-Ver202512` is an
  uninitialised submodule; the `my_configs/hatevideo/*.yaml` were authored locally and are gone;
  `llamafactory` is not pip-installed; paths point at `/data/jehc223/RGCL`, absent here; **no trained
  LoRA adapter exists on disk.** Present: Qwen2.5-VL-7B weights (16 GB), pre-extracted 8-frame JPGs
  for all 2,661 videos (2.6 GB), ShareGPT SFT JSONs (with a stale absolute path prefix).
- **K4. Cost of a properly powered design.** Recorded on A100-80GB: 2.28 h (MHC) / 2.40 h (MHC-ZH) /
  2.85 h (HateMM) per LoRA-SFT arm, plus 0.54 GPU-h per dataset for feature re-extraction.
  3 datasets × 2 arms × 3 seeds = **54.9 A100 GPU-h**; a single-LoRA-seed variant is 18.3 A100 GPU-h.
  No 5090 measurement exists, so the wall-clock on a contended card is unknown.
  *(The bundle originally said "~100 GPU-h"; the jury caught the arithmetic error and it is
  corrected here — the true figure is 54.9.)*
- **K5. The generative-decision null is already measured in-house.**
  `research-wiki/EXP_p9_lmm_rgcl_video.md`: LoRA-SFT'ing the MLLM and reading its own decision head
  lands at the protocol-matched floor (EN +0.6, ZH +1.0, HateMM +0.9, all inside seed noise), and
  reading it through the retrieval memory is 2.2–4.7 pts **below** floor.
- **K6. No leakage-free stance supervision exists on disk.** MultiHateClip's `Target_Victim` is
  ~95 % determined by the label (EN: 631/662 negatives empty, 278/339 positives filled) and cannot
  be an auxiliary target. The one clean signal is the **`Counter Narrative` annotator vote**, which
  never survives majority aggregation (63 EN + 76 ZH corpus-wide; rate 7.7 % of positives vs 5.6 %
  of negatives in EN, 7.1 % vs 7.8 % in ZH — genuinely non-leaking). **But only 1 of the 55 MHC test
  errors carries a CN vote: the asset does not mark the bucket it would be used to target.**

### 9.6 Novelty recon — six training-level families

Independent recon this session; every arXiv id fetched and title-matched, with a separate
verification pass over the eight load-bearing 2026 citations.

| family | verdict | strongest occupant | rejection citation |
|---|---|---|---|
| **F1 rationale-then-verdict SFT / RLVR** | **OCCUPIED** | IARE `2606.11953` — CoT-SFT + DPO on hateful **video**, Ex-HateMM 85.86→90.14 at **n=749**, Ex-ImpliHateVid 89.50→91.75 at n=1205; LEAF `2026.findings-acl.604`; ExPO-HM `2510.08630` (ICLR 2026) | ExPO-HM on Qwen2.5-VL-7B: Direct-SFT **75.0** F1 > CoT-SFT 74.5 > GRPO 74.5 — naive explain-then-detect *loses*; plus `2409.12183` |
| **F2 generative MLLM as classifier** | ADJACENT | RA-HMD `2502.13061` (EMNLP 2025 oral); `2501.15438` (WWW 2025); HateClipSeg `2508.01712` | RA-HMD App. G: label-token 90.2 vs head 91.1 AUC in-domain; `2603.02546` (ICLR 2026) discriminative +2.5 % on video; **small-n is a loss** — MHC-EN n=1000, generative 0.78 vs head 0.79. Compounded by K5 |
| **F3 stance / use-vs-mention as SUPERVISION** | **OPEN** | `2404.01651` (NAACL 2024) is **prompting-only** and its Limitations explicitly leave fine-tuning unexplored; TANDEM `2601.11178` supervises *target*, not stance; ImpSH `2606.18852` contrasts *implied statement* | `2404.01651` kills any inference-time-prompt framing (82.6 % FPR reduction already banked); `2307.03377` kills bare auxiliary-head MTL via negative transfer |
| **F4 annotator votes as training target** | **DROP** | AI Wizards EXIST 2026 `2607.04410` (multimodal port already done) | On HS-Brexit — 1120 items, **six** annotators — the flagship multi-annotator architecture ranked 19th and last, both worse than majority-class. We have **two** |
| **F5a/b OCR integration (encoder or prompt)** | **OCCUPIED** | **MM-HSD `2508.20546` (ACM MM 2025): PaddleOCR at 1 fps as the cross-modal attention query, macro-F1 0.874 on HateMM — equal to our ensemble's 0.8732**; `2602.09637` puts OCR in an LLM prompt on HateMM + MultiHateClip | MM-HSD + `2602.09637`, reinforced by our own §9.4 measurement |
| **F5c text-bearing frame selection for moderation** | OPEN | SFA `2511.20190`; AKS `2502.21271`, Q-Frame `2506.22139` (query-relevance, not text) | `2508.10974` (AAAI 2026): relevance sampling still misses > 90 % of harmful content |
| **F6 missing-modality / silent-video training** | ADJACENT | `2602.01101` (WWW 2026, memes); IMOL `2025.acl-long.1494` (fake-news video) | Dai et al. CVPR 2024 `2403.04245`: plain modality dropout buys robustness by *inducing modality bias*, costing accuracy on complete data — fatal when 88 % of the split has speech |

**Citation hygiene.** No arXiv id exists for LEAF, HVGuard, Fornaciari et al., Uma et al., MMIN,
MissModal, IMOL, TCE-DBF — cite Anthology/DOI, do not invent ids. `2605.20642` is an **ICML
workshop** paper and `2607.04410` a **shared-task system** paper; neither should be leaned on.
Three venue fields in `research-wiki/papers/` are stale (RA-HMD → EMNLP 2025 Main oral, ExPO-HM →
ICLR 2026, RAMF → TMLR).

### 9.7 The 10 candidates and the jury's verdict — **0 survivors**

| # | candidate | mechanism in one line | bucket | jury | verdict |
|---|---|---|---|---|---|
| C1 | Conditional-Mask Stance Auxiliary LoRA | verdict loss everywhere + a stance head supervised only on the 139 Counter-Narrative-vote items, masked elsewhere | S | **1.5** | DEAD — sparse proxy, marks 1/55 of the errors it targets: no fuel |
| C2 | Stance-Contrast LoRA | same asset as the contrast axis of a supervised-contrastive term inside the LoRA | S | **1.0** | DEAD — 139 anchors stretched into unsupported geometry |
| C3 | Stance-Conditioned Extraction Prompt | require the extractor to state endorse/condemn/report/quote before summarising; pooled over the assistant generation | S | **2.5** | DEAD — it is prompting, and §8.13's exemption geometry predicts failure |
| C4 | Hate-Relevant-Evidence Frame Selection | pick the 8 frames by predicted hate-relevant on-screen-text yield instead of `linspace` | O | **2.0** | DEAD — entire addressable prize is 5 HateMM errors |
| C5 | Bias-Controlled Transcript Dropout | stochastic transcript blanking + dominant-modality correction | M | **1.0** | DEAD — 5 errors cannot justify damaging the 88 % complete-modality majority |
| C6 | Naturally-Silent-Subset Training | train and evaluate on the natural 12.1 % silent subset rather than i.i.d. masking | M | **0.5** | DEAD — slice analysis dressed as a mechanism |
| C7 | Pairwise-Ranking LoRA Objective | move §8.10's head-level pairwise gain down into the LoRA | all | **3.0** | DEAD — the comparator already contains pairwise-trained heads |
| C8 | Self-generated rationale SFT (STaR) | rejection-sampled self-rationales, then SFT | S | **0.0** | DEAD — F1 occupied; CoT-SFT measured *below* direct SFT |
| C9 | Calibrated Generative Verdict | matched head-vs-logprob-vs-verbalised comparison with logprob repairs | all | **0.0** | DEAD — K5 is an in-house protocol-matched null; repeating it is not research |
| C10 | Vote-Fraction Soft-Target SFT | SFT the annotator vote fraction instead of a hard label | A | **0.0** | DEAD — two annotators; published evidence favours hard labels |

**The jury declined to name any pilot and declined to add a candidate of its own**, verbatim:
*"None. Spend zero GPU-hours on this slate."* and *"Every plausible addition without new
supervision reduces to prompting, pseudo-label distillation, ordinary label-contrastive training,
or learning on frozen features. The missing ingredient is data that identifies stance — not another
objective wrapped around labels that do not."*

### 9.8 The one real decision, escalated to the user

Phase A and the novelty recon converge on the same point from opposite directions: **the largest
purchasable error bucket (S, 45 %, mean +6.5 macro-F1) is also the only OPEN novelty family (F3),
and the only thing missing is stance labels.** Everything else in the training-level space is
occupied, already nulled in-house, or worth under a point.

The project *can* produce those labels without a new dataset — the standing user ruling permits raw
frames and text of all four datasets into the Claude API, and that route already produced the
133-item Gate-C annotation set with an adjudication pass. A bounded build would cover
~1,100 MHC-EN + MHC-ZH training items plus 743 HateMM, on a five-way taxonomy
(endorses / condemns / reports / quotes-mentions / depicts-without-comment).

Two facts argue against, and they are the reason this is a user decision and not an executor one:
(i) this project has been blocked on a data build **three** times (C4/B-SRTD twice, C6/EAPD once)
and the build was never funded; (ii) round 2's rejection reason — *"the capability you advertise is
an evaluation artefact"* — applies directly to machine labels validated by machines.

**The jury's ruling on this is unambiguous and is the operative constraint:**

> *"As proposed — Claude labels checked by another model — it is the same trap. It becomes the
> correct next expenditure only if human validation is funded before generation begins; model–model
> agreement is not independent validation."*

Minimum admissible audit, as specified by the jury (all four thresholds must pass, or the machine
labels may not be used as supervision):

1. **375 items** — 25 randomly sampled per (corpus × predicted-stance class), 3 corpora × 5 classes.
2. **Two independent label-blind human annotators per item**, native-language where applicable,
   plus third-human adjudication. **≈ 750 human judgements + adjudication.**
3. Human–human nominal **Krippendorff α ≥ 0.80 overall and ≥ 0.67 within every corpus**.
4. Machine labels vs adjudicated human gold: **macro-F1 ≥ 0.80 within every corpus, no stance-class
   F1 below 0.65**.

> **If the project will not fund ~750 human judgements, the jury's instruction is to close the
> stance direction outright.** That is the question for the user; the executor does not have the
> standing to answer it, because it is a resource commitment, not an evidence question.

### 9.9 Frozen conditional pre-registration

Frozen now, before any implementation line exists. Written so a later round cannot quietly relax it.

> **Trigger.** No pilot may begin until every proposed training item carries a five-way stance
> label **and** the frozen 375-item human audit of §9.8 passes all four thresholds.
> **Second-model agreement cannot satisfy this trigger.**
>
> **Mechanism unlocked.** Qwen2.5-VL-7B LoRA under the existing 8-frame protocol, joint loss
> `L = L_verdict + 0.5 · L_stance`, where `L_stance` predicts the five audited categories.
> No taxonomy, prompt, loss-weight or sampling change is permitted after unlocking.
>
> **Comparator.** The frozen round-4 three-encoder pairwise-head ensemble
> (HateMM 0.8732, MHC-EN 0.7776, MHC-ZH 0.8183), **plus an otherwise identical verdict-only LoRA
> ablation** — without that ablation the result is uninterpretable.
>
> **Decision.** GO only if the three-seed candidate improves mean macro-F1 over the frozen ensemble
> by **≥ +1.0 absolute point**, loses no dataset by more than 0.5 point, and its paired-bootstrap
> 95 % CI against the verdict-only LoRA excludes zero. Failure of any condition is **DEAD** — no
> fourth seed, no subset rescue, no prompt revision, no threshold relaxation.
>
> **Budget at unlock.** 54.9 A100 GPU-h for 3 datasets × 2 arms × 3 seeds (18.3 for a
> single-LoRA-seed variant). ImpliHateVid is out of scope by K1 and must be declared as such in any
> resulting table.

### 9.10 What round 5 bought

1. **The prize pool is measured for the first time, from the datasets' own annotators.** MHC-EN
   +15.0 and MHC-ZH +12.0 macro-F1 to the panel-resample ceiling; ≥ +7 on all four datasets.
   The recurring worry that the project is chasing label noise is **false**.
2. **The error budget has a dominant, priced mode.** Stance / use-vs-mention is 45 % of all 108
   errors and worth mean **+6.5 macro-F1** — larger than the on-screen-text, silent-video,
   annotation-ambiguity and data-defect buckets combined (17.3 %, mean +1.9), and larger than the
   entire decision-rule ceiling of §8.2(2). Any future round in this project should target it or
   justify why not.
3. **The on-screen-text channel is redundant on MultiHateClip and complementary only on HateMM**
   (§9.4) — measured, not assumed, and it upgrades §8.10 item 6 from a cache fact to an
   evidentiary one. It also makes MM-HSD (ACM MM 2025, macro-F1 0.874 on HateMM with OCR) the
   correct external comparator for anything in that family.
4. **The training-level space is mapped and mostly closed**: F1 occupied on our exact dataset
   lineage within months; F2 nulled in-house *and* a published small-n loss; F4 contraindicated at
   two annotators; F5a/b occupied. **F3 is the only OPEN family, and its supervision does not
   exist.**
5. **A hard resource fact the project did not have**: the LoRA stack is not on this machine, no
   adapter survived the move, ImpliHateVid's raw video is gone, and a properly powered
   training-level pilot is 54.9 A100 GPU-h on a GPU that is currently 84 % owned by another user.
   Any future training-level round must budget the stack restoration first.

### 9.11 Honest bottom line for round 5

> **Round 5 produced no viable candidate and, on the jury's instruction, spent zero GPU-hours.**
> Ten training-level candidates were generated against a measured error taxonomy; all ten were
> scored DEAD, seven of them on evidence that existed before any code could be written.
> **Cumulative across five rounds: 65 candidates generated, 0 live mechanism candidates.**

The round is nonetheless the most informative since round 1, because it converted a diffuse question
("what should we train differently?") into a single, priced, falsifiable one:

> **Can speaker stance be labelled well enough to supervise, and does supervising it buy the +6.5
> macro-F1 that the error analysis says it is worth?**

That question is now blocked on exactly one thing — **~750 human annotation judgements** — and on
nothing technical. The jury's position, which the executor adopts: fund the human audit, or close
the stance direction. No third option preserves the paper.

**Reproducibility index for §9**

| artifact | path |
|---|---|
| Phase A1/A2 code, raw | `idea-stage/r5_phase_a.py` · `r5_phase_a.json` |
| prize-pool / split-vote code, raw | `idea-stage/r5_phase_a3.py` · `r5_phase_a3.json` |
| per-error evidence dump | `idea-stage/r5_error_dump.py` · `r5_error_dump.json` |
| bucket coding (frozen before counting) | `idea-stage/r5_buckets.json` |
| bucket repair values | `idea-stage/r5_bucket_value.py` · `r5_bucket_value.json` |
| triage bundle (10 candidates, all constraints) | `idea-stage/codex_triage_bundle_r5_2026-08-10.md` |
| MHC OCR caches (new) | `data/OCR/MHC_test/` · `data/OCR/MHC_zh_test/` · log `logging/runs/r5_ocr_mhc/run.log` |
| OCR extractor, extended to MHC splits | `scripts/ocr_cache/extract_ocr_windows.py` |

**Process notes.** (a) Test labels were read for the Phase-A diagnostics under the user's 2026-08-09
protocol ruling; this is a disclosed diagnostic, no threshold or design was tuned on them, and no
candidate metric was ever computed. (b) The bucket coding was written to
`idea-stage/r5_buckets.json` **before** any bucket count or repair value was computed. (c) The
coding used transcripts and (HateMM/MHC) OCR text only — **no video frames were viewed**, so the
S/X boundary on visually-carried items is the coding's weakest edge and the S share may be
under- rather than over-stated. (d) The executor eliminated nothing on quality grounds; all ten
candidates went to the jury and the jury killed all ten. (e) The jury caught one factual error in
the bundle (the GPU-hour arithmetic, "~100" for a true 54.9) — corrected in §9.5 rather than
silently fixed.

---

## §10 — Round 6 (2026-08-17): the substrate is exhausted, and the measuring instrument was not fit for purpose

**Cost: ¥0.00 of a ¥60 API budget. ~2.5 GPU-hours on the local RTX 5090, all of it on cached
features. 708 head-training runs, 0 failures.**

### 10.0 Headline

Round 6 generated 12 candidates, piloted 2, and produced **one confirmed component-level gain that
is not a method contribution**. Cumulative across six rounds: **77 candidates, 0 method candidates.**

The round's substantive output is not a candidate. It is this: **the 3-seed / +0.005 decision
protocol that rendered the verdicts of the last several rounds cannot resolve the effects it was
judging.** Measured over 30 seeds, it fires GO 12.9 % of the time on a HateMM effect whose true
value is +0.0019, and it misses a genuinely above-bar MHC-ZH effect 56.5 % of the time. It needs 7
to 71 seeds and uses 3.

### 10.1 Landscape increment (independent sweep, 2026-08-17)

- **The niche has gone quiet.** An arXiv full-index sweep on `"hateful video"` returns **14 papers
  ever**; the newest is 2026-06-10. No new hateful-video method paper in roughly two months.
- **The contrast line is at or above the published frontier on three of four datasets.** HateMM:
  MM-HSD 0.874 is a statistical tie with our 0.8774 and is the same architectural class (frozen
  encoders + cross-modal attention + PaddleOCR); SAGE 0.8710; RAMF 0.851; MoRE 0.8235; MARS 0.758;
  LELA 0.7043. ImpliHateVid: TCL 0.8773 vs our 0.9118. **The one published cell that clearly beats
  us is HVGuard at 0.822 on MHC-ZH** against our 0.7821.
- **IARE's 0.9014 / 0.9175 are on Ex-HateMM / Ex-ImpliHateVid**, re-annotated variants with added
  rationale labels, and require LoRA-SFT + DPO. Not comparable to a frozen-feature head.
- The literature sweep named four transferable families as open. **Three were pre-empted by this
  project's own F-registry**: noise-robust objectives by F79 (boundary-dominated error is a 13-17 %
  upper bound) and F75 (head-loss swaps 0/8 FORMAL, 7/8 arm-dead); intermediate-layer features by
  F70; transductive inference by F63 (label propagation killed on all three datasets,
  monotone-negative in the diffusion coefficient). The fourth, joint multi-dataset training, is
  closed by a standing user veto — see §10.9.

### 10.2 The constraint map was rewritten before anything else

`RESEARCH_BRIEF.md` was replaced (commit `c1d04f8`). The previous version described the SLURM era, a
met accuracy goal and a novelty story that rounds 2-5 dismantled. The new one carries the dataset
table and test sizes, the bare-head contrast line, the measured 11 s/run head-training cost, the
asset inventory, the budget and red lines, a nine-section map of closed families, the S/O/M/A/D/X
error taxonomy with X sealed, and the seven open spaces the project's own documents still admit.

### 10.3 Candidate table — 12 generated, 0 recommended by the jury

The jury (gpt-5.6-sol, xhigh) was given the full constraint map including a section listing the
F-registry findings that pre-empt the "open" families, plus Law I and the operator pincer. It
declined to admit four families outright as isomorphic (noise-robust objectives, transduction, bare
loss swaps, single-intermediate-layer probing) and generated 12 others.

| # | candidate | P | N | G | C | composite | closest closed entry |
|---|---|---|---|---|---|---|---|
| 1 | Taxonomy-preserving hierarchical head | 5 | 1 | 2 | 10 | **3.50** | F75 / F82 |
| 2 | Metadata privileged-information residual | 3 | 3 | 2 | 8 | 3.00 | Just KIDDIN' `2411.12174`, Law I, F95 |
| 4 | Conjunctive content × endorsement | 2 | 4 | 2 | 7 | 2.75 | §4.8 zero-supervision stance |
| 5 | Sparse multi-layer residual mixer | 2 | 4 | 2 | 3 | 2.55 | F70 |
| 9 | Named-intervention derivative distillation | 1 | 5 | 2 | 5 | 2.45 | §4.8 template dominance (3.8×) |
| 7 | Cross-modal orthogonal-residual fusion | 2 | 3 | 1 | 10 | 2.40 | F95, additive complementarity |
| 11 | Typed accountability-role graph | 1 | 4 | 2 | 6 | 2.25 | synthetic attribution (10/99 markers) |
| 3 | Verified counterfactual slur invariance | 2 | 2 | 2 | 6 | 2.20 | CAD |
| 10 | Prediction-invariant entity canonicalization | 2 | 2 | 1 | 6 | 1.95 | CAD |
| 6 | Same-item bilingual consistency (ZH) | 1 | 2 | 2 | 5 | 1.70 | EN→ZH −0.138 |
| 8 | Class-conditional transcript density ratio | 1 | 3 | 1 | 4 | 1.65 | F2, likelihood read-out |
| 12 | Fixed ASR n-best marginalization | 1 | 2 | 1 | 3 | 1.35 | audio closure |

Jury verdict, verbatim: *"Number worth a method pilot: zero. … Every admitted candidate ultimately
sits on the fixed/global side of the operator pincer; none exhibits a genuine third kind."*

The executor proceeded past that verdict anyway, on the ground that a head trains in 11 seconds, so
"worth a pilot" has an unusually low bar here. **Two pilots were selected not for expected gain but
because each closes a gap the project's own records left formally open.** Both were frozen before
any code was written (`idea-stage/R6_PILOT_FREEZE_2026-08-17.md`, commit `753bb08`).

### 10.4 Pilot R6-1 — multi-layer readout fusion — **KILL** under its frozen 3-seed rule

Why it was not already closed: `refine-logs/READOUT_SUBMIT_RECORD.md` declared the readout axis
`KS-readout-dead` on a dev-split retrieval-arena kNN screen that (a) **never trained a head** — its
own record says *"NO verdict GPU, ZERO test-touch, NO head job"*; (b) had a permutation null with
**p95 = +0.0769 (ZH) / +0.0939 (HateMM)** against observed winners of +0.0128 and +0.0093, i.e. a
null band 8-9 points wide; (c) tested each layer **alone**, never in combination; and (d) ran on an
arena **F111 later declared unvalidated as a predictor** (pooled Spearman −0.3039).

Arms, on caches where layers 24 and 28 were harvested in the **same forward pass** so they are
bit-matched on prompt, frame sampling and pooling span: A0 = L28; L24; CAT = concat(l2norm L28,
l2norm L24); RANDCAT = concat(l2norm L28, l2norm(L28·R)) with one sha-pinned Gaussian R. 3 seeds,
2 datasets, 24 runs, 203 s.

| dataset | A0 | L24 | CAT | RANDCAT |
|---|---|---|---|---|
| HateMM | **0.8774** ± 0.0041 | 0.8628 ± 0.0013 | 0.8759 ± 0.0016 | 0.8712 ± 0.0023 |
| MHC_zh | 0.7603 ± 0.0531 | 0.7798 ± 0.0235 | 0.7873 ± 0.0291 | 0.7854 ± 0.0069 |

HateMM CAT−A0 = −0.0016 (1/3); MHC_zh CAT−A0 = +0.0270 (1/3) with RANDCAT−A0 = +0.0252 on the same
dataset. **0 of 2 datasets pass → KILL.** HateMM A0 reproduces the banked contrast line to four
decimals, so the instrument is the one the 2026-08-13/14 series used.

Two things were flagged at the time. MHC_zh's A0 arm had a **seed std of 0.0531** against a +0.005
GO bar. And `hate_video_95` is an all-zero row in both source `ro_` caches — the 2026-08-09
degenerate-feature repair was never propagated to the `ro_` family.

### 10.5 Pilot R6-2 — transductive pool refinement — **AMBIGUOUS**, on a defective instrument

TransCLIP / UNEM / StatA ported to a trained binary head: a two-component class-conditional Gaussian
with shared spherical covariance fitted over the **unlabelled test pool** by block-MM EM, initialised
from the inductive probabilities, with a KL anchor (λ) and a class-balance term (ρ) selected on the
dev pool. Legal under the 2026-08-09 test-input unsealing. Distinguished from F63 mechanically: it
estimates global class centroids from pool density rather than moving label mass along kNN edges.

| dataset | IND | TRANS | SHUF | T−IND | 3/3 | passes |
|---|---|---|---|---|---|---|
| HateMM | 0.8541 | 0.8828 | 0.5178 | +0.0287 | 2/3 | no |
| MHC-EN | 0.7273 | 0.7235 | 0.4808 | −0.0038 | 0/3 | no |
| MHC-ZH | 0.7776 | 0.8110 | 0.4528 | +0.0334 | 3/3 | **yes** |
| ImpliHateVid | 0.9118 | 0.9109 | 0.5079 | −0.0009 | 0/3 | no |

Exactly one dataset passes → **AMBIGUOUS**. Two defects were declared rather than patched around:

1. **λ and ρ are numerically inert.** At d = 1024 the Gaussian log-odds term has median magnitude
   ~3164 while the anchor `λ·log(p₁/p₀)` has median 1.3 and p90 4.2, at most ~17 at λ=4. Posteriors
   saturate to hard 0/1 after one E-step (**0 of all test items land in (0.01, 0.99)**) and **all 15
   grid cells return bit-identical dev macro-F1 in all 12 runs**. What ran is hard spherical 2-means
   over the pool, not the KL-anchored operator specified.
2. **SHUF is degenerate.** Because λ is inert, destroying the geometry leaves clustering unanchored,
   so SHUF collapses to chance and the `TRANS − SHUF` clause is free everywhere (+0.24 to +0.40).
   Only two of three clauses ever bound.

**The corrected experiment, run on dev only.** Rescaling the anchor by S ∈ {1 … 10000} sweeps the
whole continuum from pure clustering to pure inductive. Best-over-grid dev delta vs IND at **every**
S on **every** dataset: HateMM −0.0140 → −0.0074; MHC-EN −0.0040 flat; MHC-ZH −0.0088 flat;
ImpliHateVid −0.0051 → −0.0010. All negative or zero, approaching zero from below. **The mechanism
has no operating point.**

MHC-ZH's pass rests on 11 flipped items across 3 seeds on n=149, and its dev delta on the same
selected configuration was 0.000 / −0.0263 / 0.000. HateMM's near-pass is a threshold artifact: the
val-selected threshold came out at 0.207 and 0.178 on two seeds, costing IND ~0.034 on test, which
TRANS's saturated posteriors simply do not pay; held at 0.5 the deltas are +0.0088 / −0.0005 /
+0.0093, still not 3/3. TRANS agrees with the plain inductive prediction on 95.3-100 % of test items.

**Disposition: the frozen AMBIGUOUS stands, unaltered, and no corrected test-side re-run is
warranted** — it would spend a test read on a mechanism whose legal-split evidence is negative at
every operating point. Two mechanically different transductive operators have now failed on this
substrate; the likely common cause is on record, namely that there is **no measurable train/test
covariate shift on any of the four datasets** (domain-classifier AUC 0.42-0.56, MMD p 0.17-0.96), so
there is nothing for a pool-level correction to correct.

### 10.6 The measurement-protocol audit — **the round's real finding**

R6-1's MHC_zh seed std of 0.0531 against a +0.005 bar prompted a pre-registered variance diagnostic:
2 datasets × 3 arms × **30 seeds** = 180 runs, 27 minutes, both read-out protocols computed from the
same runs. P1 = epoch selected on val by validation macro-F1, threshold 0.5. P2 = final epoch,
threshold 0.5. 0 failures, 0 collapses.

| dataset | prot | pair | 30-seed mean | std | n* for MC SE ≤ 0.0025 | **P(3-seed rule fires GO)** |
|---|---|---|---|---|---|---|
| HateMM | P1 | CAT−A0 | +0.0019 | 0.0067 | 7.1 | **0.129** |
| HateMM | P1 | CAT−RANDCAT | +0.0071 | 0.0093 | 13.9 | 0.368 |
| HateMM | P1 | RANDCAT−A0 | −0.0052 | 0.0130 | 27.0 | 0.035 |
| HateMM | P2 | CAT−A0 | +0.0072 | 0.0156 | 38.7 | 0.325 |
| MHC_zh | P1 | CAT−A0 | **+0.0145** (t=3.78, p=0.0007) | 0.0211 | 71.0 | **0.435** |
| MHC_zh | P1 | CAT−RANDCAT | +0.0167 | 0.0189 | 57.0 | 0.499 |
| MHC_zh | P1 | RANDCAT−A0 | −0.0022 | 0.0142 | 32.4 | 0.085 |
| MHC_zh | P2 | CAT−A0 | +0.0108 | 0.0156 | 39.0 | 0.378 |
| MHC_zh | P1b | CAT−A0 | +0.0093 | 0.0354 | **200.9** | 0.201 |

GO rates are exact, enumerated over all C(30,3) = 4060 three-seed subsets.

**Findings.**
1. **The 3-seed / +0.005 protocol is not fit for purpose on either dataset.** On HateMM it declares
   GO 12.9 % of the time on an arm pair whose 30-seed truth is +0.0019 — a pure false-GO rate. On
   MHC_zh it misses a genuinely above-bar +0.0145 effect **56.5 % of the time**. The random control
   RANDCAT−A0, whose true effect is ≤ +0.0004, still returns GO 3.5-9.4 % of the time.
2. **A protocol deviation was found.** The freeze specified epoch selection by validation macro-F1;
   `idea-stage/r6_readout/analyze.py` inherited `scripts/rgcl_ablation_analyze.py::parse_run`, which
   selects on `(dev acc, dev roc)`. That key — recorded here as P1b — is what produced the 0.0531
   seed std, driving selected epochs down to 15-16 with as few as **22 predicted positives against a
   true 45**. Under it, n* rises to 201-357.
3. **Fixing the epoch schedule does not help.** In 5 of 6 dataset × pair cells P2's paired-delta
   variance is equal to or larger than P1's; HateMM CAT−A0 gets **5.5× worse** under P2. Validation-
   based epoch selection is on balance a stabiliser here, not the noise source. The dominant
   controllable term is the optimisation seed.
4. **Test-item sampling dominates the absolute number.** Item-resampling std is **0.0234 (HateMM)
   and 0.0363 (MHC_zh)** — 7-20× larger than any training-seed variance. It does not enter paired
   deltas, since every arm is scored on the same fixed test set, but it means no single macro-F1 on
   these splits is known to better than about ±0.023 / ±0.034. **The +0.005 bar is roughly one fifth
   of the irreducible width of the instrument.**

**Consequence: every recent 3-seed MHC-ZH verdict in this project — GO or KILL — is uncalibrated at
+0.005 resolution, and HateMM is only marginally better.** This does not retroactively overturn
verdicts whose effects were large (the −0.037, −0.051 and −0.021 kills of the MLLM series are far
outside this noise), but it does mean **no KILL in the ±0.01 band should be treated as settled.**

### 10.7 R6-1C — powered confirmation — **CONFIRMED-1DS**

Because the audit showed the instrument could not see the effect it had judged, a **separate,
independently pre-registered, properly powered confirmation** was frozen
(`idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md`, commit `90e9c5a`) and run. The R6-1 KILL was **not**
withdrawn; it stands as the frozen verdict of an underpowered instrument.

Guards: **seeds 30-89, disjoint from the audit's 0-29**; **two independent** random-control matrices
R_A and R_B (sha-pinned, empirical correlation 0.001) instead of one, adopting the reviewer's
objection that a single matrix estimates the redundant-view null from a sample of one; both read-out
protocols; paired bootstrap 95 % CIs over 20 000 resamples; a VOID clause if the two random arms
differ. 2 datasets × 4 arms × 60 seeds = **480 runs, 80 minutes, 0 failures**.

Absolute test macro-F1 (60 seeds):

| dataset | prot | A0 (L28) | **CAT (L28‖L24)** | RANDA | RANDB |
|---|---|---|---|---|---|
| HateMM | P1 | 0.8747 | 0.8731 | 0.8699 | 0.8705 |
| HateMM | P2 | 0.8675 | 0.8696 | 0.8657 | 0.8637 |
| MHC_zh | P1 | 0.8014 | **0.8199** | 0.8063 | 0.8035 |
| MHC_zh | P2 | 0.8080 | **0.8194** | 0.7978 | 0.7989 |

Paired deltas with bootstrap 95 % CIs:

| dataset | prot | pair | mean | MC SE | 95 % CI | pos/60 |
|---|---|---|---|---|---|---|
| HateMM | P1 | CAT−A0 | −0.0016 | 0.0014 | [−0.0043, +0.0012] | 24/60 |
| HateMM | P1 | CAT−RAND | +0.0028 | 0.0013 | [+0.0004, +0.0053] | 36/60 |
| HateMM | P1 | RANDA−A0 | −0.0048 | 0.0017 | [−0.0081, −0.0014] | 20/60 |
| HateMM | P2 | CAT−A0 | +0.0021 | 0.0022 | [−0.0023, +0.0062] | 37/60 |
| **MHC_zh** | **P1** | **CAT−A0** | **+0.0185** | 0.0033 | **[+0.0120, +0.0251]** | **46/60** |
| **MHC_zh** | **P1** | **CAT−RAND** | **+0.0150** | 0.0030 | **[+0.0089, +0.0207]** | **48/60** |
| MHC_zh | P2 | CAT−A0 | +0.0115 | 0.0027 | [+0.0056, +0.0162] | 50/60 |
| MHC_zh | P2 | CAT−RAND | +0.0211 | 0.0029 | [+0.0150, +0.0266] | 57/60 |

VOID sanity check passed: |mean(RANDA − RANDB)| = 0.0006 / 0.0020 (HateMM) and 0.0028 / 0.0010
(MHC_zh), all below the 0.005 void threshold.

**Verdict: CONFIRMED-1DS.** MHC_zh passes under P1 with both CIs excluding zero, P2 agrees in sign,
and HateMM shows no material harm (−0.0016 ≥ −0.002).

**What is now established.** On MultiHateClip-Chinese, concatenating the frozen decoder's **layer-24**
readout with its **layer-28** readout buys **+0.0185 macro-F1** over the final layer alone (60 seeds,
CI [+0.0120, +0.0251]), replicates under a second read-out protocol at +0.0115, and beats two
independent dimension-matched random projections by **+0.0150**. On HateMM the effect is absent
(−0.0016, CI straddling zero) and harmless. Under P2 the random arms actively hurt MHC_zh (−0.0101,
−0.0091) while CAT still helps — so this is not "any second view of the same forward pass helps".

**An unexplained discrepancy, recorded not claimed.** The `ro_L28` baseline on MHC_zh scores
0.8014-0.8080, which is **above the recorded MHC-ZH contrast line of 0.7821** before any of this
round's changes. The `ro_` family differs from the deployed cache in encoder tag and readout span,
so this is not a like-for-like comparison and **no gain over the contrast line is claimed here**. It
does mean a matched re-check of the deployed MHC-ZH extraction is owed.

### 10.8 Novelty verdict on the confirmed mechanism — **not a contribution**

An adversarial novelty check was run against the confirmed mechanism. Every arXiv id below was
verified against its abstract page.

- **Closest paper: `2512.21863` v2 (5 May 2026), "Frozen LVLMs for Micro-Video Recommendation: A
  Systematic Study of Feature Extraction and Fusion".** Frozen video-LLM as a black-box extractor
  **over video**; compares caption features to **intermediate decoder hidden states**; finds
  intermediate states beat captions, **different layers are complementary**, middle layers are best
  individually, and **multi-layer fusion further boosts performance**. Every empirical claim this
  mechanism would make is already stated there, on video, with a frozen video-LLM.
- **Domain claim also occupied: `2604.18519` SIREN, ACL 2026 Long** — per-layer linear probes then
  **weighted multi-layer concatenation** into a lightweight classifier on a frozen backbone, for
  **harmful content detection**, explicitly reporting that middle layers outperform the terminal
  layer. Text-only; that is the only daylight.
- **"Middle beats final" is established canon** for decoder-only LLMs (`2502.02013`, ICML 2025, up
  to 16 % on MTEB; `2412.09563`), for frozen MLLMs on video (`2507.17394`, HiProbe-VAD, ACM MM
  2025), for ViTs (`2601.09322`), and for audio encoders (`2605.10494`, ICASSP 2026). At least seven
  independent concurrent works fall in the Feb-Aug 2026 window.
- **Every serious occupant does something strictly more general**: learned attention over all layers,
  validation-weighted concatenation across all layers, dynamic per-item saliency, adaptive fusion.
  A fixed hand-picked pair with L2-norm and concatenation reads as a simplified ablation of prior
  work.

**Verdict: (d) too incremental to be a contribution, with (c) known-mechanism/known-domain as the
fallback.** The effect is real, replicated and controlled — and it is a **free implementation detail
worth banking as a better feature default and an ablation row**, not a direction. Under the user's
standing "incremental gains are acceptable" ruling it should be kept; under the "method paper only"
constraint it cannot be the paper.

The one salvage the novelty check identified — that the useful layer might depend on the *hate
modality* (implicit vs explicit, screen-text-carried vs speech-carried), which would be a
hate-specific mechanism rather than a re-derivation — **collides with Law III / F47**, because a
modality-dependent layer choice at inference is per-item selection.

### 10.9 Four closures established by measurement this round, with no pilot needed

1. **The "unused Metadata / title channel" does not exist.** Title is **already concatenated into
   the text the encoder sees** on MHC-EN (790/790 rows, mean 54.2 chars) and MHC-ZH (806/806, mean
   26.4 chars), byte-verified by re-deriving `Title + " . " + Transcript` from
   `annotation(new).json` and comparing to the deployed gt text — **790/790, 806/806, 1066/1066
   exact match, 0 mismatches**. HateMM has **0/1066** titles and ImpliHateVid **0/2009**. Open item
   #7 in the brief is closed: there is no unused metadata channel on any dataset. (Incidental: the
   Qwen extractor's `Title:` prompt slot is always literally `(none)`, because `read_gt` pulls a
   `title` key that the gt schema never contains.)
2. **The larger teacher loses to the student.** Qwen2.5-VL-32B caches exist for three datasets, and
   the 32B encoder is measurably **worse** than the 7B on every dataset and both protocols (HateMM
   final-epoch 0.8379 vs 0.8591; MHC-EN 0.6895 vs 0.7425; MHC-ZH 0.7353 vs 0.7713). Any
   "distil from a larger teacher" candidate starts with a teacher that loses to its student.
3. **Joint multi-dataset training is closed by a user veto, not by evidence.** It is the one clearly
   unoccupied slot in this benchmark family per the literature sweep — no hateful-video paper trains
   jointly across datasets. `directions_tried.json::banned_constraints[8]` records a **2026-07-14
   user veto**: *"TRAINING DATA = single-dataset train split ONLY … no cross-dataset split mixing
   (trivial trick, not a contribution)"*. The stated ground is a novelty judgement — the same
   category the user assigned to the three-encoder ensemble — so lifting it would buy numbers that
   cannot be the paper.
4. **ImpliHateVid is not permanently out of reach.** Its raw video is absent locally but present on
   Backblaze B2 (**2012 objects, 50.1 GiB**, verified reachable). The "3 of 4 datasets" ceiling on
   every encoder-level candidate is a 50 GB fetch, not a structural limit.

### 10.10 External review

The round's pilots, dispositions and proposed conclusion were sent to an external reviewer
(gpt-5.6-sol, xhigh) with instructions to be hostile.

- **R6-1 KILL: sound.** CAT−RANDCAT is *"weak diagnostic evidence"* only, because *"CAT beating
  RANDCAT means 'L24 is less harmful than the random redundant view', not 'L24 improves
  deployment'"*, and 3 head seeds are not 3 independent random-matrix draws. **The two-matrix
  design in R6-1C was adopted directly from this objection.**
- **R6-2 disposition: defensible.** *"The implementation tested seeded hard spherical two-means …
  the preregistered three-clause instrument did not operate as designed."* A corrected test re-run
  after inspecting both the original test behaviour and the corrected dev sweep *"would be fishing,
  not confirmation"*. The one legal continuation named is train-only nested resampling or a new
  evaluation set — not a corrected test re-run.
- **The seed variance is a serious protocol red flag**, and its identification of a
  measurement-protocol certification audit as the single highest-expected-value next action is what
  produced §10.6 and §10.7.
- **Grades: honesty 8/10, usefulness 9/10.** Overstatements it named, and which are corrected in
  §10.12 below: *"the substrate does not contain a reachable method contribution"* exceeds the
  evidence; the 91-98 % disjointness result is formal for a defined operator class, not for every
  possible learning mechanism; *"F3 is the only open family"* is too absolute; and the user **can**
  technically lift the selection ban — the accurate claim is that doing so has low expected value
  because every available selector has failed, not that the ruling is immovable.

### 10.11 What round 6 bought

1. **The project's decision instrument is now characterised.** For the first time there are exact
   false-GO and miss rates for the 3-seed / +0.005 rule (0.129 false GO on HateMM; 0.565 miss on
   MHC-ZH), the seed counts it actually needs (7-71, and 201-357 under the `(dev acc, dev roc)` key
   that was silently in use), and a decomposition separating optimisation-seed, epoch-selection and
   test-item variance. **No pilot bar in this project should be set again without this table.**
2. **A silent protocol deviation was caught**: the analysis path used `(dev acc, dev roc)` epoch
   selection while freezes specified validation macro-F1. That single key is responsible for the
   0.0531 seed std and a 3-30× inflation of the seeds required.
3. **The readout axis is closed on the arena that counts** — the deployed head path, per F113 —
   rather than on the raw retrieval arena F111 later invalidated. And the closure is *qualified*:
   the two-layer concatenation is worth **+0.0185 on MHC-ZH** and nothing on HateMM.
4. **A real, replicated, controlled component gain exists and is banked**, together with an honest
   novelty verdict that it is not a paper.
5. **Four open items in the brief were closed by measurement** at zero cost (§10.9), including one —
   the metadata channel — that had been carried as open since round 3.
6. **The transductive family is now closed from two mechanically independent directions** (F63 edge
   propagation, R6-2 pool-density estimation), with the shared cause named: no covariate shift.
7. **A data-integrity defect was found**: the 2026-08-09 degenerate-feature repair was never
   propagated to the `ro_` cache family, where `hate_video_95` remains an all-zero row.

### 10.12 Honest bottom line for round 6

> **Round 6 produced no method candidate. Cumulative across six rounds: 77 candidates, 0 method
> candidates.** It produced one confirmed component-level gain (+0.0185 macro-F1 on MHC-ZH from a
> two-layer frozen readout) whose mechanism is occupied by at least two 2026 papers, one of them on
> video with a frozen video-LLM and one of them on harmful content at ACL — so it is an
> implementation default, not a contribution.

The accurate strong claim, stated as the reviewer required rather than as originally drafted:
**no identified candidate under the present scope has cleared a valid pre-registered gate**, and the
instrument used to render most recent gates has now been shown unable to resolve effects in the
±0.01 band.

The binding constraints, stated precisely:

- **(a) The prize is stance, and it is blocked on money, not on ideas.** Stance / use-vs-mention is
  49 of 108 test errors (45.4 %), worth mean **+6.46 macro-F1**, and F3 is the only training-level
  family the literature leaves open. Six independent zero-supervision routes have now failed, and
  the sixth — the synthetic-attribution probe — explains why: **only 10 of 99 real transcripts
  contain any attribution marker at all**. The cue is not in the speech. Unblocking costs roughly
  **750 human judgements** (375 items × 2 blind annotators + adjudication, α ≥ 0.80, machine-vs-human
  macro-F1 ≥ 0.80) or a paid corpus licence (**LDC BeSt `LDC2023T13`**, label `ROB`, 10,777 EN
  instances — institutional membership still unchecked).
- **(b) Most measured headroom is unreachable, and that is evidence, not policy.** The F66
  decomposition puts **91-98 % of convertible headroom inside per-item selection** (HateMM +0.0776 =
  +0.0012 legal-symmetric + +0.0764 banned-selection). Law III closed selection at all three
  supervision sources by measurement — unsupervised probes find no signal, train-supervised
  selectors degenerate, dev-supervised routing is negative. **A user ruling cannot open this**, and
  the disjointness result is formal for the defined operator class rather than for every conceivable
  learning mechanism.
- **(c) The remaining policy-only blockers buy numbers, not papers.** Cross-dataset mixing and
  graded auxiliary targets (F82) are blocked on novelty grounds. Lifting them is cheap and would
  probably gain +1 to +3 on the small datasets; under the method-paper constraint neither can be the
  contribution.
- **(d) The measuring instrument is itself now a binding constraint.** A +0.005 bar is one fifth of
  the test-item resampling width on these splits, and the 3-seed rule cannot resolve it.

**Recommended next actions, in order.**
1. **Adopt the audit's numbers as protocol.** No new pilot bar without a stated n*; minimum 30 seeds
   on MHC-ZH and 15 on HateMM for any effect claimed in the ±0.02 band; fix the epoch-selection key
   to validation macro-F1 everywhere; report paired-bootstrap CIs, not seed counts.
2. **Bank the two-layer readout as the default feature construction** on MHC-ZH and re-check the
   deployed MHC-ZH extraction, whose contrast line (0.7821) sits below the `ro_L28` baseline
   (0.8014-0.8080) for reasons not yet explained.
3. **Escalate the one real decision, unchanged from §9.8**: fund the stance supervision, or close the
   stance direction. Round 6 adds two facts to that decision — the zero-cost routes are now
   exhausted six times over, and the instrument that would evaluate the funded result needs fixing
   first, or the supervision will be evaluated through an uncalibrated gate.
4. **Propagate the `-degenfix1` repair to the `ro_` cache family.**

### 10.13 Reproducibility index and process notes

| artifact | path |
|---|---|
| rewritten constraint map | `RESEARCH_BRIEF.md` (commit `c1d04f8`) |
| brainstorm bundle given to the jury | `idea-stage/codex_brainstorm_bundle_r6_2026-08-17.md` (`3d1bbaf`) |
| pilot freeze (R6-1, R6-2) | `idea-stage/R6_PILOT_FREEZE_2026-08-17.md` (`753bb08`) |
| pilot results | `idea-stage/R6_PILOT_RESULT_2026-08-17.md` (`9dbb510`) |
| R6-1 code + raw | `idea-stage/r6_readout/{build_arms.py,run_arms.sh,analyze.py,build_meta.json,results.json}` |
| R6-2 code + raw | `idea-stage/r6_trans/{run_heads.sh,dump_r6.py,em_r6.py,results.json,dumps/}` |
| protocol audit code + raw | `idea-stage/r6_audit/{run_audit.sh,analyze_audit.py,results.json}` |
| confirmation freeze | `idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md` (`90e9c5a`) |
| confirmation code + raw | `idea-stage/r6_confirm/{build_randab.py,run_confirm.sh,analyze_confirm.py,build_meta.json,results.json}` |
| logs | `logging/runs/{r6_readout,r6_trans,r6_audit,r6_confirm}/` |

**Process notes.** (a) Every decision rule was frozen and committed before the corresponding code
was written; the R6-1C freeze was committed before any seed in 30-89 was executed. (b) Test labels
were read for the §10.6 audit as a disclosed diagnostic under the 2026-08-09 protocol ruling; no
threshold, epoch rule or design was tuned on them, and the arms were already killed. (c) The
R6-1 KILL is recorded as-is and was not withdrawn when the audit showed the instrument was
underpowered; the confirmation is a separate run on disjoint seeds. (d) Both R6-2 defects were
declared by the executor rather than patched around, and the frozen AMBIGUOUS was not upgraded.
(e) The external reviewer's objection to a single random-control matrix was adopted before the
confirmation ran, not after. (f) Total API spend for the round: **¥0.00 of ¥60**.
