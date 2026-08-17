# Round-4 idea-generation bundle — hateful video detection, mechanism-level novelty

You are a senior ML researcher brainstorming research ideas for a **methods paper aimed at a top
venue** (NeurIPS / ICML / ICLR / CVPR / ACL Main / ACM MM). Read everything below, then produce
the deliverable in §9.

**This project has generated 41 candidates over three prior rounds and 0 survived.** Read §5 (the
death list) carefully — a re-skin of anything on it is an automatic waste of this round. The
ground under the project changed on 2026-08-09/10 in three ways (§1), which is why a fourth round
is being run at all.

---

## §1 — What changed since round 3 (the whole reason this round exists)

### 1.1 The foundation was replaced: the retrieval pipeline is dead, the bare head is the baseline

A 99-cell frozen ablation (`idea-stage/RGCL_ABLATION_RESULT.md`, 11 encoder×dataset cells × 3 loss
rungs × 3 seeds, single submission, test-reported) took apart the project's own deployed
RGCL/RA-HMD retrieval pipeline:

| component | verdict | evidence |
|---|---|---|
| retrieval-guided contrastive pairing (FAISS kNN vs in-batch random) | **DECORATIVE** | 9/22 supporting cells (needed ≥11), cross-cell mean +0.039 |
| contrastive regulariser itself (vs BCE only) | **DECORATIVE, and the sign is negative** | 1/22 cells, cross-cell mean **−0.170** |
| kNN vote read-out (vs linear head) | nominally ALIVE (20/33) but **diagnosed as an artefact** | the +0.176 gain exists only in the 22 cells where the contrastive term collapsed the head's *threshold*. In the 11 cells where the head is healthy, kNN−head = **−0.0022**, 2/11 cells |
| full deployed pipeline vs bare BCE head | pipeline **loses** | cross-cell mean **−0.0017**; pipeline wins ≥+0.005 in 2/11 cells, loses ≤−0.005 in 7/11 |

**The most important diagnostic in that report (§6): adding a contrastive term does not damage the
head's *ranking* — test ROC is unchanged to ±0.01 — it destroys the head's *threshold*, pinning all
logits on one side of 0.5 so macro-F1 collapses to the all-negative degenerate value.** The entire
apparent "kNN read-out is better" effect is kNN supplying a usable decision rule where the sigmoid
head no longer has one.

**Consequence: the new baseline, and the thing any new mechanism must beat, is frozen features +
a bare BCE head.** Its test macro-F1 (train→train, val→epoch selection, test→report):

| dataset | best encoder | **test macro-F1** | n test | published comparator |
|---|---|---|---|---|
| HateMM | LoRA-Qwen | **0.8774** | 215 | MM-HSD 0.874, SAGE 0.871 (ACL 2026) — *we are already above published SOTA* |
| MultiHateClip-EN | frozen Qwen2.5-VL-7B | **0.7331** | 161 | — |
| MultiHateClip-ZH | LoRA-Qwen | **0.7821** | 149 | — |
| ImpliHateVid | CLIP or Qwen | **0.9118** | 401 | — |

A new mechanism has to beat *these four numbers*, and it no longer has to live inside a retrieval
pipeline — the pipeline is gone. **A real, reproducible +1 to +2 macro-F1 points is enough for a
story** (the user has explicitly ruled that incremental-but-real gains are acceptable and that
"one-step +5 points" is NOT the kill line). It must be a *mechanism*, not an engineering stack.

### 1.2 The test set is unsealed — transductive / TTA mechanisms are unlocked

The project previously ran under a hard "zero test-set contact" red line, which killed every
transductive idea in rounds 1–3 before it was written down. On 2026-08-09 the user lifted it:

- **Allowed**: unlabelled test *inputs* (features / pixels / transcripts / OCR) at inference time;
  transductive and test-time-adaptation mechanisms, declared honestly in the paper; one-shot
  pre-registered grids evaluated on test and reported in full.
- **Not allowed**: iterating (look at test → change design → look again) and then reporting the
  result as held-out.
- A full unlabelled train↔test audit has already been done (`refine-logs/TRAINTEST_AUDIT_2026-08-09.md`)
  — see §4.4 for its load-bearing findings.

This is a genuinely new degree of freedom for this project. **But the TTA literature is extremely
crowded** and the project's own recon already mapped the occupancy (§5.2). What we need is the
intersection of this new freedom with an asset or a measured failure mode nobody else has.

### 1.3 ImpliHateVid — 2009 videos, never attacked in three rounds

Three rounds spent themselves on HateMM and MultiHateClip. ImpliHateVid (1283/325/401) was never
the target of a single candidate. Its item ids partition the corpus into **EX** (explicit hate),
**IM** (implicit hate), **NH** (non-hate) — test = 92 EX / 108 IM / 201 NH — an *implicitness*
stratification no prior round used. §4.1 reports what happened when we finally measured it.

---

## §2 — Exclusive assets (things only this project has)

1. **MultiHateClip official per-annotator votes** (`data/gt/mhc_votes/*.tsv`, EN 801 + ZH 800 items;
   580 items ×2 votes, 120 ×3, 1 ×4). Nobody in the field uses these. **Direct mechanisation is
   dead** — see §5.3, the Human-Agreement-Retrieval family was killed three separate ways. Other
   uses (evaluation target, auxiliary signal) are not exhausted, but treat this asset as *heavily
   burned* and require a strong argument.
2. **Full OCR cache** for HateMM + HateClipSeg, train+val+**test**, K=30 windows and video-level
   (`data/OCR/`). Every existing way of feeding it to a model has failed (§5.4).
3. **HateClipSeg Whisper transcripts** (394/395) — cached, never wired into any model.
4. **Claude-API exemption for raw video frames** — a general standing ruling: frames/clips may go to
   the Claude API for any dataset and any task, no per-case approval. Generation/annotation is
   therefore cheap in *hours*, not GPU-days.
5. **ImpliHateVid implicit/explicit strata** (§1.3) — unspent until this round.
6. MultiHateClip `Target_Victim` (248 EN + 278 ZH) — spent as a *stratifier* in round 3 (R3-1) and
   found to be near-totally confounded with language; unspent as a mechanism.
7. MultiHateClip `Metadata` (title/description) — the single largest human-annotated contributing
   modality in ZH (294) and second in EN (197), and **no model in this project reads it**. Recorded
   as a fact, not a candidate: "add a modality" is not a mechanism.

**Compute**: one RTX 5090, no cluster. A bare-head training run is ~30–60 s. Pilots must be
CPU-minutes to low-GPU-minutes. Anything needing a week of GPU is out of budget by definition.

---

## §3 — Hard constraints (non-negotiable)

- **Methods paper only.** The user has permanently forbidden benchmark / audit / evaluation-metric /
  resource papers. "This would be a great D&B or Findings paper" = the candidate is dead, not
  redirected. Three prior rounds kept producing evaluation-track material and it is all unusable.
- **Top venue only.** Workshop / Findings / short-paper fallbacks do not count as survival.
- **Mechanism-level novelty.** "Apply X to Y", "add modality Z", "swap encoder", "an ablation table",
  "a metric correction" are all pre-rejected.
- Data boundary: raw video stays local; frames may go to the Claude API.
- Evaluation protocol for everything below: **train on train, select epoch on val, report TEST**,
  ≥3 seeds, decision rule frozen before the run, single submission, all cells reported.

---

## §4 — Fresh forensic recon run for this round (2026-08-10, on test, disclosed)

These are new measurements taken today specifically to inform this brainstorm. They were run with a
harness (`idea-stage/r4_harness.py`) validated to reproduce the ablation table's bare-head cells
(HateMM/CLIP 0.8013 vs published 0.7993; ImpliHateVid/CLIP 0.9068 vs 0.9118; HateMM/Qwen 0.8588 vs
0.8640). **These are recon on the test set, disclosed as such.** Any candidate built directly on
them is confirmatory-by-construction and must say so.

### 4.1 The implicit axis is NOT where ImpliHateVid fails — the non-hate side is

Per-subtype test accuracy of the bare head (mean over 3 seeds):

| encoder | EX (explicit, n=92) | IM (implicit, n=108) | NH (non-hate, n=201) | AUROC EX-vs-NH | AUROC IM-vs-NH |
|---|---|---|---|---|---|
| CLIP | 0.978 | 0.907 | 0.871 | 0.985 | 0.957 |
| Qwen | 1.000 | 0.917 | 0.876 | 0.980 | 0.950 |

**Implicit hate is nearly as easy as explicit hate on this corpus (IM-vs-NH AUROC 0.95).** The
hypothesis "our accuracy is carried by the explicit subset and implicit is where everything fails"
is **false here** and should not be proposed. The error budget is inverted: of ~38 errors, ~25 are
**false positives on non-hate** and only ~12 are missed hate. The model over-fires on hate-adjacent
non-hate content. Whatever mechanism is proposed on this dataset must target *false positives on
non-hate*, not implicit-hate recall.

### 4.2 The decision-rule headroom is small — remaining error is genuine ranking error

For each dataset, the bare head at fixed 0.5, versus threshold variants (3-seed means):

| cell | base @0.5 | val-tuned threshold | train-prior quantile (transductive, no test labels) | **oracle threshold (upper bound)** |
|---|---|---|---|---|
| HateMM / LoRA | 0.8651 | 0.8632 (−0.002) | 0.8611 (−0.004) | 0.8804 (**+0.015**) |
| MHC-EN / Qwen | 0.7338 | 0.7347 (+0.001) | 0.7311 (−0.003) | 0.7727 (**+0.039**) |
| MHC-ZH / LoRA | 0.7864 | 0.8101 (+0.024) | 0.8218 (+0.035) | 0.8326 (**+0.046**) |
| ImpliHateVid / CLIP | 0.9068 | 0.9077 (+0.001) | 0.9160 (+0.009) | 0.9185 (**+0.012**) |

**Reading: even a test-label oracle threshold buys only +1.2 to +4.6 points.** Threshold/calibration
mechanisms are capped there, and "threshold recalibration under prior shift" is BBSE/Saerens-EM,
a 20-year-old solved problem already on the project's never-claim list. **Any candidate must move
ROC, not just the decision rule.**

### 4.3 Encoder errors are strongly complementary, and the complementarity is being wasted

Three encoders (CLIP ViT-L/336, frozen Qwen2.5-VL-7B, LoRA-Qwen), 3-seed mean probabilities:

| dataset | best single (macro-F1 / ROC) | mean-prob ensemble (macro-F1 / ROC) | disjoint errors |
|---|---|---|---|
| HateMM | 0.8658 / 0.9315 | 0.8586 / **0.9333** | CLIP&Qwen both-wrong 16, only-CLIP 25, only-Qwen 12 |
| MHC-EN | 0.7302 / 0.8571 | 0.6891 / **0.8768 (+2.0 ROC)** | Qwen&LoRA both-wrong 28, only-Qwen 10, only-LoRA 11 |
| MHC-ZH | 0.8039 / 0.8983 | 0.7704 / **0.9175 (+1.9 ROC)** | CLIP&Qwen both-wrong 13, only-CLIP 14, only-Qwen 17 |
| ImpliHateVid | 0.9151 / 0.9699 | **0.9276** / **0.9745** | CLIP&Qwen both-wrong 20, only-CLIP 18, only-Qwen 14 |

**This is the sharpest live signal in the whole recon.** Different encoders fail on largely
*disjoint* items (on MHC-EN, only 28 of 38+39 errors are shared). Naive probability averaging
converts that into a consistent **+1.5 to +2.0 ROC** gain on every dataset — but macro-F1 goes
*down* on three of four, because the averaged score's threshold is wrong. This is the **same
pathology as ablation §6**: ranking information exists and the decision rule throws it away.

Combining both fixes (ensemble + train-prior quantile threshold) beats the best single encoder on
MHC-EN (+2.0) and ImpliHateVid (+1.8) but loses on HateMM (−0.2) and MHC-ZH (−0.8): mean ≈ +0.7,
inconsistent. **And "ensemble + prior-matched threshold" is a trivial baseline, not a mechanism** —
a top venue rejects it on sight. The *gap* it exposes is the interesting object: there is
+1.5–2.0 ROC of real, cross-encoder complementary information that no current decision procedure
converts into macro-F1. A mechanism that converts it — for a principled reason, not by tuning —
would be worth a paper. Note also that this is a natural place for the newly-unlocked transductive
freedom: the conversion problem is about the *unlabelled test pool as a set*.

### 4.4 Data-hygiene facts that bound what any result can claim

From `refine-logs/TRAINTEST_AUDIT_2026-08-09.md` (zero labels read):

- **HateMM's official test split carries 12.1 % whitespace-only transcripts vs 5.2 % in train
  (Fisher OR 2.49, p = 0.001)**, and that stratum is one constant CLIP-text vector sitting in a
  ~93 %-one-class region. **Every HateMM test number with a CLIP text channel is inflated by an
  unquantified amount.** This has now bitten four separate experiments.
- HateMM: 3 md5-identical files span split boundaries (one **val↔test** under opposite class-prefix
  ids); 7 content clusters put near-identical video in both train/val and test = 3.3 % of test.
- MHC-EN is **CLEAN** (zero cross-split near-duplicates, max cross-split cosine 0.899, zero
  degenerate rows). It is the only one of the four that survives a hostile duplicate audit.
- MHC-ZH 2.7 % of test in cross-split clusters; ImpliHateVid 1.5 % (feature-level evidence only —
  raw media not on this machine).
- **HateMM and ImpliHateVid ids encode the label in the id string.** Never expose ids to a model.

---

## §5 — The death list. A re-skin of any of these is an automatic fail.

### 5.1 Killed directions (each closed by this project's own frozen-verdict experiments)

Multi-segment complementarity · single-segment selection · OCR−ASR residual · CVoI cost-aware
acquisition · segment-level retrieval keys · visual-purity segment selection · type-hard-partitioned
memory · streaming/continual memory · cross-lingual "EN memory rescues ZH" · **the retrieval
pipeline itself** (new, §1.1) · **the audio-operator family** (prosody-as-operator, FiLM/gating/
bilinear audio conditioning, any "audio modulates text" successor — closed by the C8 pilot, and the
audio axis is now 0-for-4 on HateMM).

### 5.2 Never-claim novelty for these (occupied, with citations on file)

1. Growing/swapping a datastore for zero-gradient domain adaptation — kNN-LM (ICLR 2020), kNN-MT (ICLR 2021).
2. Inserting the model's own test-time predictions back into memory — AdaNPC (ICML 2023), TDA (CVPR 2024).
3. Age/staleness-scored memory eviction — RoTTA (CVPR 2023), Lu et al. (AIJ 2016).
4. Confidence/entropy-gated cache admission — CRG / ACE / DOTA / SCA (2025).
5. Observing that pseudo-label errors accumulate ("cache noise") — same family.
6. Wave-style memory insertion + evaluation on later time slices — Mireshghallah et al. (EMNLP 2023) does the whole thing.
7. The phrase "non-parametric continual learning" — HippoRAG 2 (ICML 2025).
8. "Zero-gradient insertion is mechanistically different from gradient updates" — disproved by 2305.13034 (EMNLP 2023).
9. Adapting to new classes by swapping memory contents — Memory-Modular Classification (TMLR).
10. Adversarial poisoning of a retrieval store — PoisonedRAG (USENIX Sec 2025), AgentPoison (NeurIPS 2024).
11. Datastore compression/pruning — Efficient kNN-LM (EMNLP 2021), Cluster-Based kNN-MT (ACL 2022).
12. Per-sample abstention / escalation routing (killed twice as a re-skin in round 2).
13. Threshold/prior recalibration under label shift — BBSE (ICML 2018) / Saerens-EM (2002).

### 5.3 Framing killers — a new mechanism must route around all three

1. **SAGE** (ACL 2026 Main) hits HateMM 0.8710/0.8628, statistically indistinguishable from this
   project's own numbers ⇒ **a pure accuracy claim on HateMM is not publishable**.
2. **HCG-MPB** (ICMR 2026) replaces per-instance retrieval with an LLM-distilled prototype bank and
   explicitly argues in its motivation that instance-based retrieval is a flawed design.
3. **`2607.23304` Context-Adaptive Inference**: under squared loss + linear head + fixed features,
   explicit parameter adaptation and implicit routing are *both* kernel ridge regression on joint
   (input, context) features ⇒ **"our module is a form of test-time adaptation" is formally
   absorbed**. With **ERM `2602.05152`** (query expansion ≡ key expansion), "we improved the
   query/key construction" also ceases to be an independent claim. **Any TTA-flavoured candidate
   must state why it is not absorbed by 2607.23304.**
4. **The Illusion of Progress?** (NeurIPS 2025 D&B) — TTA for VLMs shows limited gains over the
   earliest work and buys accuracy at the cost of trustworthiness. Any "our cache adapts better"
   claim collides with it head-on.

### 5.4 The 41 dead candidates, compressed (do not regenerate these)

**Round 2 (13)**: Human-Agreement Retrieval (vote-distribution memory + agreement-defined contrastive
topology) · dissent-preserving prototype bank · counter-narrative matched retrieval ·
duplicate-conflict memory · provenance-typed OCR fusion · sampling-phase robust retrieval ·
rank–vote decoupling · retrieval placebo suite · chance-corrected temporal grounding ·
component-sufficiency training · contested-item abstention · annotation-escalation prediction ·
modality-attributed retrieval decomposition.

**Round 3 (14)**: C1 target-conditioned attack/defence stance algebra (**piloted, KILL** — both arms
of the double dissociation failed, the one large number was language-confounded and inside its own
null) · C2 victim-marginalised attack energy · C3 vote-constrained semantic polytope · **C4 semantic
response-tensor distillation (jury 6.0/10 — the round's highest, killed only by a MISSING ASSET, not
by a mechanism failure or an occupant; see §6)** · C5 Möbius interaction distillation (self-killed on
its own asset precondition) · C6 executable agency-graph distillation · C7 noncommutative rhetorical
pooling · **C8 prosody-as-operator (piloted, KILL, family closed)** · C9 rank-copula multistream
pooling · **C10 cross-channel evasion transduction closure (piloted, KILL, and the observed
inconsistency fell *below the entire null* — the premise is inverted)** · C11 platform-invariant
semantic derivatives · **C12 proposition-mass firewall (piloted, KILL — negative rho; the OCR sign
flip is NOT caused by formatting mass)** · C13 counterexample-guided harm-circuit compiler · C14
relational quotient induction for latent dogwhistles.

### 5.5 Transferable findings from the graveyard (these constrain new designs)

- **The OCR sign flip has a cause and it is *which windows*, not how evidence is weighted.** The same
  OCR vector gives +0.0094 through a frozen head and **−0.0246** through a learned fusion MLP. The
  dose curve is strongly concave: 3 of 30 windows deliver 61 % of the gain. Formatting-mass
  explanations are dead (R3-2). Surviving explanation: dilution by uninformative windows.
- **Composed evasion needs no special machinery**: single-edge augmentation is free (clean-margin
  retention 1.004) and already absorbs length-2 and length-3 compositions.
- **A bounded vote/count selection score is degenerate by construction** — use continuous
  non-saturating scores. (A "below chance" result turned out to be an argmax tie-break artefact.)
- **A frozen-CLIP visual segment key is a video-level style detector (AUROC 0.782) and a coin flip
  within a video (AUROC 0.511)** — there is no within-video localisation signal in it.
- **The audio channel's failure mode is redundancy, not weakness**: a within-label-permuted prosody
  vector adds *more* to a text head than the real one does. Prosody's label-relevant content is
  already in the transcript.
- **Signals that look strong against a weak comparator die against a trained one.** This killed
  round 2's rank-1 candidate (P-A → P-A-v2) and round 3's C1. Every pilot must include a properly
  trained comparator and a label-permuted null, and the bar should be ≥3× the null's 95th
  percentile.

---

## §6 — Old candidates eligible for revival (the bare-head foundation lowered their cost)

The new foundation (§1.1) removes the retrieval pipeline these were designed to sit inside, which
makes several of them *cheaper*, not necessarily better. Re-rank them honestly; say if they should
stay dead.

- **C4 semantic response-tensor distillation** (jury 6.0/10, the highest score of round 3). Distil a
  teacher's *finite-difference Jacobian/Hessian over named semantic interventions*, not its logits or
  explanations. **Blocker is purely a data build**: `data/Counterfactual/*/train_twins.jsonl` has 348
  pairs but **every record is label=1** (harmful only) and there is exactly **one** intervention type
  (a toxicity-sanitising transcript rewrite); only 132 of 348 flip. A real lattice needs ≥2
  intervention axes (at minimum target substitution and endorsement/condemnation reversal) over
  **both** classes. The Claude-API frame/text exemption makes that hours of work. Prior art bounds
  the claim to *the named-intervention response tensor*: `1803.00443` (Jacobian matching) and DISCO
  `2212.10534` (distilling counterfactuals) exist; "distilling counterfactual behaviour" is taken.
- **C7 noncommutative rhetorical pooling** (5.0/10) — video as an ordered *product* of near-identity
  matrices `exp(A(z_t))`, commutator terms representing rhetorical reversal. Concern on file: the
  labels probably cannot identify it (Gate-0 found only 8.2 % multi-segment structure).
- **C9 rank-copula multistream pooling** (4.5/10) — within-video soft empirical ranks → differentiable
  cross-modal copula tensor instead of mean/max pooling. Note §5.5: pooling *is* the measured culprit
  in more than one place.
- **C6 executable agency-graph distillation** (unscored, 1–2 months) — typed graph over speakers /
  uploader / quoted sources / propositions / victims with `asserts / quotes / endorses / condemns`
  edges; hate counted only when an endorsement path reaches an accountable agent. Risk on file:
  "explanation distillation in disguise". **Note this one against §4.1**: ImpliHateVid's error budget
  is dominated by false positives on non-hate, and use-vs-mention / quoting / reporting is exactly
  the confusion that produces those.
- **C2** victim-marginalised attack energy (4.8), **C13** harm-circuit compiler (4.7), **C11**
  platform-invariant semantic derivatives (4.1), **C3** vote-constrained polytope (3.8), **C14**
  relational quotient induction (3.5).

---

## §7 — Landscape update for this round

Full working notes with per-item verification levels: **`idea-stage/phase1_landscape_r4.md`**
(read it — it has the arXiv ids, and it marks unverified items `[C]`). Condensed:

### 7.1 Transduction: the set-structure branch is NOT open — but its *regime* is

Every sub-branch we hoped was free has a top-venue owner:

| branch | owner |
|---|---|
| "treat the whole unlabelled test set as a set" (GMM/MLE + KL anchor, block-MM) | **TransCLIP**, arXiv 2406.01837, **NeurIPS 2024 Spotlight** — the canonical method |
| graph / label propagation transduction | **ZLaP** (2404.04072, CVPR 2024), **ECALP** (2412.18303) |
| Sinkhorn / optimal-transport assignment over the test batch | **SAT** (2411.17002) |
| test-prior estimation | BBSE (1802.03916), 2411.15204, 2511.18615 |
| transductive conformal | 2509.04631, 2605.01452 |

**But**: all of them are built for **many-class, text-anchored, roughly balanced** pools. They import
a class anchor from a text encoder ("a photo of a {class}"), which a **binary policy label** —
"is this hateful" — does not provide. **Domain application is still essentially empty**: the only
occupant in hateful video is SCANNER (2602.00132, AAAI 2026), which is centroid alignment, not
prior estimation or set-structure transduction.

### 7.2 The counter-literature is strong, specific, and must be answered

- **On Pitfalls of Test-Time Adaptation** (2306.03536, ICML 2023) — batch dependency wrecks model
  selection; no TTA method wins everywhere.
- **StatA / Realistic Test-Time Adaptation** (2501.03729, **CVPR 2025**) — current transductive
  methods "systematically compromise initial zero-shot robustness" and gain only under favourable
  test-distribution assumptions; **TransCLIP degrades when the number of effective classes drops** —
  which is precisely our regime (2 classes).
- **The Illusion of Progress?** (2506.24000, NeurIPS 2025 D&B).

**The convergent demand of all three is: prove your adaptation cannot damage the un-adapted model.**
That demand is currently unstated in the binary / moderation regime, where a false-positive-side
regression is a deployment failure rather than an accuracy dip.

### 7.3 ImpliHateVid — verified source facts, and they complicate §4.1

arXiv **2508.06570**, **ACL 2025 Main Long** (`2025.acl-long.842`), Rehman et al. 2,009 videos =
1000 NH / 509 IM / 500 EX, splits 1283/325/401 (matches our 92 EX + 108 IM + 201 NH exactly).
Encoders: ImageBind + VADER/NRCLex/OFA captions.

- Their **binary** headline: **87.53 acc / 87.73 F1**. Our bare head's 0.9118 is above it.
- **They do report the 3-class breakdown and it inverts the naive story**: 3-class macro-F1 69.18 =
  **NH 84.48, IM 66.05, EX 57.02**. In the 3-class task the live confusion is **EX↔IM**, not
  hate↔non-hate. Note this is *not* in tension with §4.1 — §4.1 is the binary task, where both EX and
  IM separate easily from NH; telling EX from IM is a different and much harder problem, and it is
  **not** our binary target. Any candidate proposing to exploit it must explain why the 3-class task
  is the right one to optimise.
- **Implicit-hate mechanisms are text-only and crowded**: DuPL (WWW 2026), **HatePrototypes**
  (2511.06391) — whose finding is that prototypes are *interchangeable* between implicit and
  explicit, which pre-empts "implicit hate needs its own representation"; FiADD (2309.11896);
  2406.07886; 2606.18852.
- **New to our map and the likeliest direct competitor: DeHate, ACM MM 2025
  (`10.1145/3746027.3758272`), explicitly on "explicit and implicit" hateful video.** Paywalled —
  no design decision may rest on its numbers.

### 7.4 "Simple beats complex" is already published, in-domain, on HateMM

Koushik, Kanojia & Treharne, arXiv **2502.07138** (MM4SG @ WebConf 2025): *"simple embedding fusion
achieves state-of-the-art performance on video content (HateMM) with a 9.9 pt F1 improvement"*.
⇒ **The 99-cell ablation of §1.1 is a stronger version of an existing workshop finding. It is not a
methods paper and must not be proposed as one.**

### 7.5 What actually beats a frozen-feature linear probe at 10²–10³ examples

**LP++** (2404.02285, CVPR 2024), **CLAP** (CVPR 2024), **GDA** (2402.04087), **Tip-Adapter**
(2111.03930), frozen-feature augmentation (2403.10519), LDA (2604.03928). **Two patterns hold across
all of them**: (i) they are **closed-form or hyper-parameter-search-free**; (ii) they **borrow a text
class anchor**. Neither transfers for free to a binary policy label over heterogeneous multimodal
features — which is exactly the unoccupied version.

### 7.6 No new SOTA 2026-06 → 2026-08

The arXiv sweep found only benchmarks / audits / jailbreak work (2608.05210, 2607.15442, 2607.11597,
2607.21151, 2606.18852, 2606.09700). Frontier unchanged: SAGE 0.8628 macro-F1 on HateMM. **Our bare
head is at or past the published frontier on two of four benchmarks — a benchmark-validity fact, not
a contribution** (and §3 forbids the paper that would make of it).

### 7.7 The surveyor's six structural gaps, verbatim

1. Transduction is built for many-class, text-anchored, balanced pools — **nobody owns transduction
   for a binary, prior-shifted, few-hundred-item moderation pool whose class anchor is a policy
   definition rather than a text embedding**.
2. This project's own W4 experiment says drift failure is **calibration, not separability** — and
   **no one has estimated the test-pool label prior transductively for hateful video**.
3. Implicit/explicit is universally treated as a **dataset partition**, never as a **latent
   coordinate that the decision rule is a function of**.
4. Every implicit-hate mechanism is text-only; **nobody exploits the fact that in video the surface
   cue and the inference it requires sit in different modalities**.
5. What beats a linear probe at this scale is closed-form / validation-free + text-anchored; **the
   binary-policy-label, heterogeneous-multimodal-feature version is unoccupied**.
6. The counter-literature's convergent demand is **"prove your adaptation cannot damage the
   un-adapted model"** — unstated in the binary/moderation regime.

*(Surveyor's stated blind spots: no non-English search; DeHate / HCG-MPB / TIHD / DuPL PDFs
paywalled; ACM MM 2026 / EMNLP 2026 / NeurIPS 2026 accept-lists not indexed; ~20 items marked `[C]`
= id+title seen in a result page but not opened. Treat "unoccupied" claims as weaker than a full
Phase-3 sweep.)*

---

## §8 — What a good round-4 candidate looks like

Given everything above, the shape of a candidate worth this round:

- Its mechanism is defined on **frozen features + a trainable head** (or on the inference procedure),
  because that is now the whole system. It does not need a retrieval pipeline and should not
  reintroduce one.
- It has a reason to move **ROC / ranking**, not just the decision rule (§4.2 caps decision-rule
  mechanisms at +1.2 to +4.6 points, and that framing is already occupied).
- It either exploits a genuinely exclusive asset (§2) or a measured failure mode of this project
  that outsiders have not measured (§4, §5.5).
- If it touches TTA/transduction, it must survive §5.3-3 (kernel-ridge absorption) and §5.3-4
  (Illusion of Progress) explicitly.
- Its minimum viable experiment is CPU-minutes to GPU-minutes on cached features, with a
  label-permuted null and a properly trained comparator.
- It is a mechanism a reviewer can name in one sentence, and the answer is interesting whichever way
  it comes out.

---

## §9 — Deliverable

Generate **12–16 concrete candidates**. Coverage is mandatory across these five groups (roughly 3
each; the "free" group may be larger if you have something better):

1. **Mechanisms native to the bare-head foundation** — things that only make sense now that the
   pipeline is gone, e.g. what a head can be made to do with frozen features that a BCE objective
   does not.
2. **Transduction/TTA × an exclusive asset** — the newly unlocked freedom crossed with something only
   this project has. Generic TTA is dead on arrival (§5.2, §5.3).
3. **ImpliHateVid / the implicitness axis** — but read §4.1 first: the naive framing is empirically
   false. The live target on that corpus is false positives on hate-adjacent non-hate.
4. **Revival re-ranking** — of §6's held candidates, which (if any) deserve to come back on the new
   foundation, and what specifically changed for them. Saying "none" is an acceptable answer if
   argued.
5. **Free** — anything, including attacking the §4.3 complementarity gap, provided it is not a re-skin.

For each candidate give exactly:
1. **One-sentence summary.**
2. **Core hypothesis** — what you expect to find and why.
3. **Minimum viable experiment** — the cheapest thing that produces a GO/KILL signal, stated
   concretely enough to implement against cached `.pt` features, with the decision quantity named.
4. **Contribution type** — empirical finding / new method / theoretical result / diagnostic.
5. **Risk** — LOW / MEDIUM / HIGH.
6. **Effort** — days / weeks / months.
7. **Death-list check** — name the closest entry in §5 and state in one sentence why this is not that.
   *A candidate without this line will be discarded.*
8. **Absorption check** — if it involves adaptation, routing, or context, state why §5.3-3 does not
   formally absorb it.

Then rank all candidates and name the **top 3 you would actually run**, with the strongest objection
a reviewer would raise against each.

**Be honest about the base rate.** 41 candidates have died here. If your read is that a given group
has nothing above ~5/10, say so plainly rather than manufacturing filler — a well-argued "this group
is empty" is more useful to us than a padded list. The project has a documented failure mode of
re-skinning dead ideas, and you are the gate that catches it.
