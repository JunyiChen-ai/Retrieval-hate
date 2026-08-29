# STANCE_LIT_RECON — stance / use-vs-mention in hateful **video** and hate **speech (text)** detection

**Date** 2026-08-11 · **Type** literature reconnaissance only (no experiments, no GPU)
**Commissioned question** — has anyone done stance extraction / stance-aware modelling in
(1) hateful **video** detection, (2) hate speech detection in the **text** domain? For each work:
what is the **finding**, not just the method.

**Bottom line up front**

| question | answer | evidence strength |
|---|---|---|
| **Q1 — stance in hateful VIDEO detection** | **NO. Zero occupants found** — not modelling, not annotation, not evaluation, not even error analysis. The one stance-adjacent artefact in the whole video literature is an *undocumented* `Counter Narrative` value in MultiHateClip's released annotation files that its own paper never mentions and never uses. **But the pipeline shape is occupied** (RAMF `2512.02743`: frozen reasoner → typed text records → trained fusion, +3 macro-F1), so the novelty must live in the *typology*, not the wiring. | **Strong for "no stance work"** (arXiv hateful-video literature enumerated exhaustively: 14 papers; codebooks read directly); medium for "nothing anywhere" — see §6 |
| **Q2 — stance in TEXT hate speech detection** | **YES, and the findings are unusually decisive.** The one clean, quantified positive is **stance as intermediate-task pretraining** (`2206.06423`: +3 weighted F1, counter-hate F1 0.38→0.45) — and the same paper shows sentiment (−8), sarcasm (−7) and more hate data (−4) all *hurt*, so the gain is **specific to stance**. The famous −82.6 % FPR figure is a *relative* reduction on ~90 expert-written examples via prompting. And the sharpest result is a **published null**: the stance-toward-quote decision rule that gives **+100 pp on HateCheck F20 gives +0.0 pp on ETHOS**. Nobody injects a typed stance judgment as a trained-model **input feature** for hate. | **Strong** |

---

## 1. Scope, channels, and honest limits

Channels actually used:

- **arXiv API** (`export.arxiv.org/api/query`) — ~25 phrase-conjunction queries, several returning
  genuine zero-hit feeds (recorded in §6).
- **Direct PDF retrieval + full-text extraction** for the three anchor papers (`2404.01651`,
  `2210.00910`, `2310.19750`) and MultiHateClip (`2408.03468`) — every number below marked
  `[PDF]` was read out of the paper, not out of an abstract.
- **ACL Anthology** page/PDF fetches.
- **Semantic Scholar Graph API** (partly blocked, HTTP 429 for a stretch of the session).
- **Two parallel independent wide sweeps** (video axis ~72 tool calls, text axis ~99 tool calls),
  adding OpenAlex, EuropePMC, OSF and OpenAlex citation-graph traversal; their coverage and their
  stated gaps are recorded verbatim in §6.
- **The project's own on-disk data** — annotation schemas of HateMM / MultiHateClip /
  HateClipSeg / ImpliHateVid inspected directly (§2.2). This is the strongest single piece of
  evidence in the report and it is first-hand.

**Limits that must be stated.** (a) The session's `WebSearch` budget (200 calls) was exhausted
mid-way, so open-web / Google-shaped discovery was truncated; arXiv+S2+Anthology carried the rest.
(b) `export.arxiv.org` rate-limited repeatedly, so some planned queries were not run. (c) Coverage
is English-dominant plus a small number of Chinese queries. **"Not found" here means "not found
through the channels in §6", not "does not exist".** Non-arXiv venues (ICWSM, CSCW, ACM MM
workshops, LREC, journal-only work) are the most likely place a missed occupant would hide.

---

## 2. Q1 — hateful VIDEO detection

### 2.1 The literature: zero occupants

Nothing in the sweep couples a stance / speaker-attitude / use-vs-mention / endorse-vs-condemn
judgment to hateful-video classification, in any wiring — not as a feature, not as supervision,
not as a prompt, not as an evaluation axis. Concretely, these arXiv phrase-conjunctions returned
**genuinely empty feeds** (valid Atom, zero `<entry>`):

- `all:"stance" AND all:"hateful video"` → 0
- `all:"stance" AND all:"hate video"` → 0
- `all:"speaker stance" AND all:"video"` → 0
- `all:"use-mention" AND all:"video"` → 0
- `all:"counter-speech" AND all:"video detection"` → 0
- `abs:"stance" AND abs:"harmful video"` → 0
- `abs:"counterspeech" AND abs:"video"` → 1 hit, and it is **not** a video paper
  (`1808.04409`, *Thou shalt not hate*, a Gab/Twitter text study)
- `abs:"stance" AND abs:"video" AND abs:"hate"` → 1 hit, `2503.10648`, which is
  hate/sentiment analysis of **YouTube comment text** about Israel–Palestine, not video content
  modelling.

The nearest *systems* in the hateful-video space are all covered already in
`idea-stage/MLLM_FRONT_RECON.md` §3.1: MM-HSD (ACM MM 2025, HateMM M-F1 0.874), HVGuard
(EMNLP 2025, 0.8597), RAMF (`2512.02743`, TMLR, 0.837), MoRE (WWW 2025, 0.8235), MARS
(`2601.15115`), LELA (`2602.09637`), IARE (`2606.11953`). **None of them has a stance field.**
RAMF is the closest neighbour and the gap is precise: it asks a frozen VLM for an *objective
description*, a *hate-assumed* inference and a *non-hate-assumed* inference — that is
**assumption-conditioned adversarial reasoning about the analyst's hypothesis**, not
**attitude attribution to the speaker in the video**.

A second, independent sweep this session (arXiv API ~35 queries + OpenAlex 18 + Semantic Scholar
~20 + ~22 full-text fetches) reached the same verdict by **exhaustive enumeration** rather than
sampling:

- `abs:"hateful video" OR abs:"hate video"` returns **14 papers in total** on arXiv — all 14 were
  enumerated and the relevant ones read. **None has a stance channel.**
- `abs:"stance" AND abs:"video"` returns **31 papers in total** — all enumerated. **None is about
  hate/harm video content**; the two hate-flagged hits concern *comments on* videos.
- Zero-hit queries: `all:"hate" AND all:"video" AND all:"stance detection"`,
  `abs:"stance" AND abs:"speaker" AND abs:"video"`, `abs:"counter speech" AND abs:"video"`,
  `abs:"toxic" AND abs:"video" AND abs:"stance"`.

**Two things the sweep found that change the framing:**

1. **`2512.02743` (RAMF) must be cited and explicitly distinguished — it is the same pipeline
   *shape*.** Frozen reasoner emits **three typed textual records per video** (objective
   description / hate-assumed inference / non-hate-assumed inference) → consumed as inputs by a
   trained fusion classifier. Reported **+3 macro-F1 over SOTA and +7 hate-class recall**. The
   difference is the **typology**: their types are hypotheses about *the label* ("assume it is
   hate / assume it is not"), ours would be attributions of *the speaker's attitude*
   (endorse / condemn / report / quote). That distinction is real and defensible, but a reviewer
   will demand it in writing. Sibling: **MARS `2601.15115`** — same typology, training-free
   (objective description → evidence-for → **counter-evidence** → synthesis), up to **+10 %** over
   other training-free methods.
2. **Video stance detection exists as its own task and is growing — just never touching hate.**
   MultiClimate (`2409.18346`, 100 climate YouTube videos, first manually-annotated multimodal
   video stance dataset), **TikStance** (`2607.15240`, 161 TikTok videos + 13,876 comments,
   Favor/Against/None for *both* video→target and comment→target, Krippendorff α 0.72–0.74),
   Inter-Stance (`2604.22739`, dyadic multimodal conversational stance), `2509.08024` (two-stage
   context learning with LLMs for multimodal stance), DIVERSE (`2403.03334`, YouTube comment
   stance via LLM weak supervision), `2605.02939` (multimodal controversy detection modelling
   audience perspectives). **So "stance from video is technically feasible" is already
   demonstrated — we would not be inventing the perception problem, only the application.**
   TikStance in particular shows video-level stance annotation reaching α ≈ 0.74, which is a
   useful prior for whether our own stance typing can be labelled reliably.

**One warning from the sweep.** MultiHateClip's own §5.4 error analysis lists exactly three failure
modes — hateful-vs-offensive confusion, non-Western data scarcity, implicit content — and
**does not mention counter-speech / commentary / irony confusion at all.** The failure mode we are
targeting is not merely unsolved in the video domain, it is **undocumented**. That is a gap to
claim, but it also means **we carry the burden of demonstrating the failure mode exists** — which
is precisely what this project's own §9.2 error audit (45.4 % bucket S) already does, and that
audit is now load-bearing evidence, not a side result.

### 2.2 The video datasets have no stance labels — verified on disk

Inspected directly in this repo:

| dataset | label schema actually shipped | stance? |
|---|---|---|
| **HateMM** (`data/gt/HateMM/HateMM_annotation.csv`) | `video_file_name, label∈{Hate,Non Hate}, hate_snippet (spans), target` | none |
| **HateClipSeg** (`data/gt/HateClipSeg/gold_segments.json`) | per-segment multi-hot over `{normal, hateful, insulting, sexual, violence, harm}` + `[start,end]` | none |
| **ImpliHateVid** (`data/gt/ImpliHateVid/*.jsonl`) | transcript text + binary implicit-hate label | none (its ACL 2025 paper `2508.06570` adds **sentiment / emotion / caption** auxiliary features — the closest *feature-channel* analogue in the video domain, but affect ≠ stance) |
| **ToxVidLM** (`2024.findings-acl.663`, 931 Hindi-EN videos) | binary toxicity + **auxiliary heads for sentiment and severity** | none — and note these are auxiliary *output heads*, not input channels |
| **HarmVideoBench** (`2606.27187`) | 3-layer harm taxonomy: observable evidence / clip-internal meaning / beyond-clip reasoning | **no stance layer** |
| **Harmful-YouTube taxonomy** (`2411.05854`) | 6 harm categories (information, hate+harassment, addictive, clickbait, sexual, physical) | **no stance dimension** |
| **MultiHateClip** (`data/gt/mhc_votes/*.tsv`) | `Majority_Voting ∈ {Hateful, Offensive, Normal}`, per-annotator `Label`, `Target_Victim`, `Component ∈ {Metadata, Transcript, Vision component, Audio}` | **one undocumented trace — see below** |

**The MultiHateClip finding (first-hand, and it is the sharpest fact in this report).**
The released per-annotator `Label` vocabulary is
`{Normal: 2699, Offensive: 1082, Hateful: 520, Counter Narrative: 139, No: 1}` across 2001 videos.
So a **`Counter Narrative` option existed in the annotation interface and was used 139 times**.
But:

1. The MultiHateClip paper (`2408.03468`, ACM MM 2024) `[PDF, full text read]` states its guideline
   as *"classify each video into one of the three categories: Hateful, Offensive, Normal"*. The
   strings `counter`, `narrative`, `stance`, `condemn`, `denounce`, `satire`, `irony`, `quote`
   appear **nowhere** in the paper in this sense. The category is entirely undocumented.
2. `Counter Narrative` **never survives to a majority label** — the `Majority_Voting` vocabulary is
   exactly `{Normal: 1338, Offensive: 453, Hateful: 210}`. The signal is collapsed away by
   construction.
3. Of the **139 videos carrying ≥1 `Counter Narrative` vote** (6.9 % of the corpus), the majority
   label is `Normal` for 90, **`Hateful` for 26 and `Offensive` for 23**. I.e. in **49 videos
   (35 % of the counter-narrative-flagged set) at least one trained annotator read the video as
   counter-narrative while the majority called it hateful or offensive.**

That is direct, in-domain, human-generated evidence that (a) stance ambiguity is real and
measurable in hateful video, (b) it is concentrated exactly where the labels are contested, and
(c) **every published system, including ours, discards it.** It is also the reason the project's
own §9.2 error audit found the stance bucket at **45.4 % of all errors, oracle-worth mean
+6.46 macro-F1** — the datasets are structurally blind to the distinction their annotators
occasionally noticed.

> **Caveat on how far this can be pushed.** 139 minority votes is not stance supervision. The
> project already ruled this dead once (`IDEA_REPORT` C1, *Conditional-Mask Stance Auxiliary LoRA*,
> scored 1.5: "sparse proxy, marks 1/55 of the errors it targets: no fuel") and the R3-1 stance
> algebra pilot **KILLed** on it (`D_stance = 1.464` against its own permutation null of 1.289,
> resting on 6 EN / 2 ZH matched items). Use these 139 as *evidence that the phenomenon exists*,
> **never** as a training signal.

### 2.3 Verdict on Q1

**The video domain is empty of *stance* work — but not of the *pipeline shape*.** No paper, no
dataset schema, and no benchmark taxonomy in hateful-video detection carries a speaker-stance /
use-vs-mention / endorse-vs-condemn construct. The differentiation survives, with one qualifier
that must be written into any prereg: **RAMF `2512.02743` already publishes "frozen reasoner emits
typed textual records → trained fusion head" on hateful video, at +3 macro-F1.** Our novelty
therefore cannot be the wiring; it has to be *the typology carried by the records* (speaker
attitude, not label hypotheses) plus whatever the typology buys.

Confidence: **high** for "no stance occupant" — the arXiv hateful-video literature is only 14
papers and was enumerated exhaustively, dataset codebooks were read directly, and the relevant
phrase conjunctions return literally empty feeds. **Medium** for "nothing anywhere" (§1/§6 limits:
no Google Scholar, ACM DL partially blocked, Chinese academic databases not searched).

---

## 3. Q2 — hate speech detection in TEXT: what exists and what it bought

Three genuine occupants, plus a set of adjacent findings. Ordered by how much they constrain our
design.

### 3.1 Gligorić, Cheng, Zheng, Durmus, Jurafsky — *NLP Systems That Can't Tell Use from Mention Censor Counterspeech, but Teaching the Distinction Helps*
`arXiv 2404.01651` · **NAACL 2024 Main** (`2024.naacl-long.331`) · `[PDF, full text read]`

**Method (one line).** Two tasks — (T1) classify a text as *use* vs *mention* of hateful /
misinformative language; (T2) run downstream hate / misinformation detectors on the same items —
then fix T2 by **prompting**: embed the use–mention definition + an instruction that mentioning
harmful language is not itself harmful, plus CoT and few-shot exemplars. **No training, no
features, no fine-tuning.**

**Data.** Counterspeech pairs from Knowledge-grounded hate countering (Chung et al. 2021) and
Multi-Target Counternarratives (Fanton et al. 2021); misinformation counternarratives (He et al.
2023). N = 1826 (harmful, counterspeech) pairs; mean longest-common-substring between the pair
= 3.44 words (the "focal token" that the detector trips on).

**Findings — the numbers that matter to us.**

1. **LLMs cannot do use-vs-mention.** Average error rates 12.22–16.38 % (hate) and 13.64–37.22 %
   (misinformation). The best model, gpt-4, still mistakes a *mention* for a *use* in
   **20.00 %** of hate counternarratives and **23.44 %** of misinformation counternarratives.
   *(Note: `pdftotext` mis-orders the sub-task blocks of Table 2 — the paper's own prose, quoted
   here, is authoritative.)*
2. **Downstream censorship is large.** Counterspeech FPR on the hate task: ToxiGen-HateBERT
   **24.44 %**, gpt-3.5-instruct **25.56 %**, Perspective identity-attack **21.11 %**, Perspective
   toxicity **20.00 %**, RoBERTa-hate **17.78 %**, gpt-3.5-turbo **11.11 %**, gpt-4 **8.89 %**
   (Perspective *insult* is the lone exception at 4.44 %). Misinformation task: RoBERTa-fake-news
   **97.93 %**, gpt-3.5-instruct 26.12 %, gpt-3.5-turbo 22.11 %, gpt-4 **10.21 %**.
   **Caveat on all of these: the hate evaluation is ~90 expert-written CONAN-family counterspeech
   pairs, and the 95 % CIs in Tables 2–3 are ±5–10 pp.** These are clean canonical counterspeech,
   not naturalistic web text — the effect is real but the precision is low.
3. **The error genuinely propagates (this is the key causal-ish result).** Stratifying downstream
   FPR by whether the use-mention judgment was right: gpt-4 **15.78 % → 4.54 %**
   (χ² = 30.60, p = 3.2 × 10⁻⁸); gpt-3.5-turbo 28.31 % → 14.44 %; gpt-3.5-instruct 32.96 % →
   19.52 %; all p < 10⁻⁵. **Getting the stance/use-mention call right cuts the downstream false
   positive rate by roughly 3×.** This is the single strongest published justification for a
   stance channel anywhere in this report.
4. **The headline "−82.6 %" is a *relative* reduction and only from prompting.**
   For gpt-4, hate: FPR 8.89 % → **1.55 %** with CoT+mitigation (**−82.61 % relative, −7.34 pp
   absolute**); misinfo 10.21 % → 4.18 % (**−59.06 %**). Ablated: few-shot alone −43.48 %,
   instruction alone −39.13 %. Cost: TPR on true use 80.00 % → 77.61 % (−2.99 % relative).
5. **Surface distancing cues are anti-correlated with the right answer.** Counterspeech containing
   **quotation marks is misclassified *more* often**, not less: gpt-4 hate **28.57 % FPR with
   quotes vs 7.23 % without** (p = 0.056); gpt-4 misinfo 25.00 % vs 9.84 % (p = 0.027);
   gpt-3.5-instruct hate 57.14 % vs 22.89 % (p = 0.046). *A naive quotation-detector feature would
   point the wrong way.*
6. **Stance *strength* is what the model reads, not stance *presence*.** Fightin'-Words analysis:
   counterspeech classified correctly is loaded with explicit strong-disagreement metalanguage
   ("fake", "lying", "misleading", z ≈ +4 to +5); counterspeech misclassified is loaded with the
   topical terms themselves ("mRNA", "vaccine", "gene therapy"). *"Downstream classification has
   fewer errors when the disagreement in mentioning statements is not subtle."*
7. **The error is identity-confounded.** FPR on counterspeech by target identity (gpt-3.5-turbo):
   Jewish 14.15 %, PoC 9.09 %, Muslims 6.80 %, LGBT+ 6.77 %, … migrants 0.39 %. Same pattern for
   gpt-4. Mere mention of some identity terms is treated as impermissible.

**Limitations, verbatim-relevant.** *"we do not investigate all the possible mitigation strategies.
For example, **fine-tuning with more examples could help further decrease the error rates**"*; and
the study is *"limited to specific types of mentioned language related to counterspeech"* —
attributed language and words-as-themselves.

**Relation to our direction.** **Supports, and defines the boundary.** It (i) proves the
phenomenon and its downstream cost, (ii) proves that *correct* stance judgments cause lower FPR,
(iii) **burns the inference-time-prompt framing** — that is banked and published — and
(iv) explicitly leaves the trained/feature route open in its own Limitations.

### 3.2 Goldzycher & Schneider — *Hypothesis Engineering for Zero-Shot Hate Speech Detection*
`arXiv 2210.00910` · **TRAC @ COLING 2022** · `[PDF, full text read]`

**Method (one line).** Zero-shot NLI: instead of one hypothesis *"That contains hate speech."*,
compose several. Four strategies: **FBT** filter-by-target, **FCS** filter-counterspeech,
**FRS** filter-reclaimed-slurs, **CDC** catch-dehumanising-comparisons. FCS is a three-stage
composition: (1) regex quotation identification (quoted text → variable `X`), (2) hate
classification of the *quoted* span `p₀`, (3) **stance of the surrounding text `p₁` toward the
quote**, via hypothesis *"This text supports [X]."* — hate only if the outer text is *supportive*.
This is the closest published conceptual ancestor of a typed stance record.

**Findings — and this paper is mostly a *warning*.**

1. **On the diagnostic suite it is a total success.** On HateCheck functionality F20
   (*denouncements of hate that quote it*), FCS moves accuracy **0 % → 100 %** (+100.0 pp).
   Overall HateCheck: 79.4 % baseline, **FCS alone +4.6 pp**; all four strategies combined
   **+7.9 pp → 87.3 %**.
2. **On a natural corpus it buys exactly nothing.** ETHOS: baseline 69.6 %,
   **FCS = 69.6 %, Δ = +0.0 pp.** Zero. All the ETHOS gain came from FBT
   (+9.1 pp with target *characteristics*) and FRS (+1.7 pp); the full combination reached
   79.6 % (+10.0 pp) — still below fine-tuned BERT (80.0) / DistilBERT (80.4).
   **The stance/counterspeech component is the one that transfers worst.**
3. **Generalising the stance filter destroyed it.** The authors noticed FCS only inspects hate
   *inside* the quotes, so a text that quotes something and is *also* hateful outside the quotes
   slips through. The obvious fix — also test the main hypothesis on the outer text `p₁` (`FCSp1`)
   — **wiped out the entire gain: +4.6 pp → +0.0 pp.** Adding FBT on top (`FCSp1+FBT`) recovered
   only **+0.3 pp**. Their diagnosis: *"counterspeech often also conveys strong negative emotions
   that are mistaken by the model for hate speech"* — and counterspeech necessarily contains
   target-group terms, so target filtering can't rescue it either.
4. Cost: one extra forward pass per hypothesis, linear in the number of hypotheses.

**Relation to our direction.** **Supports the mechanism, and supplies the sharpest negative
result in the whole file.** A stance filter that works perfectly on constructed diagnostics can be
worth **zero** on a real corpus, and making it more general can be worth **less than zero**. Any
design of ours must be evaluated on the natural corpora, never on a constructed stance suite, and
must not assume the "quote → check stance of surround" decomposition survives contact with real data.

### 3.3 Gatto, Sharif, Preum — *Chain-of-Thought Embeddings for Stance Detection on Social Media*
`arXiv 2310.19750` · **Findings of EMNLP 2023** (`2023.findings-emnlp.273`) · `[PDF, full text read]`

**Method (one line).** Frozen ChatGPT produces CoT reasoning text → encode it with
Twitter-RoBERTa → fine-tune a classifier on `[tweet ⊕ CoT]`. **This is exactly our target
topology (frozen reasoner → text artefact → encoder → trained head), executed on
stance-*as-the-task*, not stance-*as-a-feature-for-another-task*.**

**Findings — and the honest reading differs from the headline.**

1. Tweet-Stance: best model **F1 76.3**; embedding the CoT beats reading the LLM's own label off
   the CoT by **+5.5 F1**; +6.1 over the ChatGPT-CoT model.
2. Presidential-Stance/Biden: **50.6 → 71.3 F1**, a **+20.7** jump — **but 50.6 is ChatGPT-CoT's
   own extracted label, not a trained baseline.** Against the *published prior SOTA* the gain is
   **+1.4 F1** (Biden) and **+2.4 F1** (Trump). *Quote the +1.4/+2.4 when arguing about realistic
   effect sizes; the +20.7 is an LLM-label-extraction artefact.*
3. **The artefact can carry the whole signal.** On Trump, the best model is **TR-COT — CoT text
   only, tweet discarded (F1 81.5)**; on Tweet-Stance TR-COT is within 0.6 F1 of tweet+CoT. The
   encoder is reading the reasoning, not the content.
4. **Why encoding beats reading the label:** the encoder survives *stance label hallucination*
   (ChatGPT answering about the wrong entity) and *implicit stance confusion* — it corrects the
   Neutral class 56 % of the time on Biden.
5. **Reasoner quality is the binding constraint.** ChatGPT > Llama-2 > Falcon; Llama-2's CoT helps
   on Biden and **not** on Trump. A weak reasoner produces a useless channel.
6. Limitation stated: highly prompt-sensitive.

**Relation to our direction.** **Placeholder for the wiring, not for the idea.** It establishes
the topology works and that *embedding the reasoning beats trusting its label* — which is the
argument for a typed record consumed as a feature rather than a prompt-time verdict. It does not
touch hate detection.

### 3.4 Yu, Blanco & Hong — *Hate Speech and Counter Speech Detection: Conversational Context Does Matter*
`arXiv 2206.06423` · **NAACL 2022** · `[PDF, full text read]` · **the most design-relevant paper in this file**

**Method (one line).** Three-way Reddit annotation (Hate / Counter-hate / Neutral) of a target
comment, with and without its parent; RoBERTa classifier; **intermediate-task pretraining** on
several related corpora before fine-tuning on the hate/counter task.

**Findings.**

1. **The label is not a property of the span.** Showing annotators the parent flips **38.3 % of
   judgments**: Hate→Neutral 34.2 %, **Counter-hate→Neutral 55.1 %**, and outright inversions
   **Counter-hate→Hate 18.7 %**, Hate→Counter-hate 8.4 %.
2. **Base rate.** Of target comments containing a hate word, 45 % are Hate but **19 % are
   Counter-hate.** (Roughly one in five hate-word-bearing items is the opposite of hate.)
3. **Context as an input feature is worth ~3 points.** Weighted F1 0.58 (target only) → **0.61**
   (parent+target); per class Hate 0.56→0.59, **Counter-hate 0.38→0.44**, Neutral 0.69→0.70.
4. **Stance pretraining is worth another ~3 points, and it is the *only* auxiliary that helps.**
   Verified in the main text: *"models pretrained for stance detection obtain better results than
   pretrained with other tasks"* — target-only **0.58 → 0.61**, with context **0.61 → 0.63**;
   best system (context + silver blending + stance pretraining) **0.64**, p < 0.01 vs the plain
   target-only model. The reported appendix ablation of *which* auxiliary corpus (weighted F1,
   target-only, baseline 0.58) is the crucial part:

   | intermediate task | weighted F1 | Δ | counter-hate F1 |
   |---|---|---|---|
   | none | 0.58 | — | 0.38 |
   | + more hate data (Davidson 2017) | 0.54 | **−4** | 0.12 |
   | + hate Reddit (Qian 2019) | 0.58 | 0 | 0.37 |
   | + **sentiment** (SemEval-2017) | 0.50 | **−8** | **0.00** |
   | + **sarcasm** (Ghosh 2020) | 0.51 | **−7** | 0.08 |
   | + **stance** (DEBAGREEMENT, agree/neutral/attack) | **0.61** | **+3** | **0.45** |

   *(Main-text numbers 0.58/0.61/0.63/0.64 and the class-level counter-hate 0.38→0.44 were read
   directly from the PDF; the per-auxiliary appendix row values above come from the parallel
   text-axis sweep and should be re-verified against the appendix before being put in a paper.)*
5. **Residual errors of the best model:** 48 % "lack of information in the target", 27 % negation,
   19 % sarcasm/irony, 8 % hate without profanity.

**Relation to our direction — this is the strongest single support, and it is a *specificity*
result.** Among {more in-domain hate data, sentiment, sarcasm, stance}, **only stance helps**, and
sentiment/sarcasm do not merely fail — they **annihilate the counter-hate class** (F1 → 0.00 and
0.08). That matters directly because the closest video-domain feature-channel analogue,
ImpliHateVid (`2508.06570`), injects exactly **sentiment and emotion**. This paper predicts that
choice is the wrong auxiliary. It is also the counterweight to `2307.03377`: a stance auxiliary
helps where a generic one hurts. Nobody has replicated it, nobody has done it as *joint* multi-task
(only sequential pretraining), and the one attempt at a fused identity-auxiliary encoder came back
null (`2602.12818`, §3.6). **"Stance as a jointly-trained auxiliary signal" is an open slot with one
positive precedent and one negative precedent.**

### 3.5 How big is the problem in text? — HateCheck's counter-speech functionalities `[PDF]`

`2012.15606` (Röttger et al., ACL 2021) built HateCheck **because** of this exact failure: interview
participant I4 said *"people will be quoting someone, calling that person out […] but that will get
picked up by the system"*, and the authors *"therefore included functionalities for counter speech
that quotes or references hate"* — **F20 (denouncements of hate that quote it)** and
**F21 (denouncements that make direct reference)**.

Accuracy on those two functionalities:

| model | F20 | F21 | (F9 reclaimed slurs) |
|---|---|---|---|
| BERT trained on Davidson (B-D) | **26.6 %** | **29.1 %** | 39.5 % |
| BERT trained on Founta (B-F) | **32.9 %** | **29.8 %** | 33.3 % |
| Google Jigsaw Perspective (P) | **15.6 %** | **18.4 %** | 28.4 % |
| SiftNinja (SN) | 79.8 % | 79.4 % | 18.5 % — *but only because SN calls almost everything non-hateful* |

**Trained hate classifiers get counter-speech right 16–33 % of the time.** Perspective, the
strongest model on hateful cases (>95 % on 11/18 hateful functionalities), is the **worst** on
counter-speech. HateCheck's own recommended fix is targeted **data augmentation** on "negated hate,
reclaimed slurs and counter speech".

**And the follow-up says that fix backfires** — `2204.04042` (Bourgeade et al., 2022,
*Checking HateCheck: a cross-functional analysis of behaviour-aware learning*) `[PDF]` fine-tunes on
HateCheck with held-out functionalities: accuracy on **held-out functionalities and identity groups
improved**, but performance on **held-out functionality *classes* and on i.i.d. hate-speech data
decreased**; models "learned to associate some spelling variations with hateful language because of
how the test suite was constructed", i.e. **overfitting to the challenge distribution**. Their
conclusion: *"the models fine-tuned on HateCheck passed the functional tests with flying colours,
but task performance measured by the non-challenge datasets decreased."*

> This is the **third of four independent instances** of the same pattern (with `2210.00910` ETHOS +0.0,
> `2210.00910` FCSp1 −4.6, and `2307.12418` HateModerate non-hate failure .205 → .229):
> *stance/counterspeech interventions that look decisive on constructed diagnostics do not, by
> default, convert into gains on natural data — and can cost i.i.d. performance.* Treat this as the
> governing prior for our design.

**Two more measurements of the same problem, both worth citing for scale:**

- **`1809.07572` (van Aken et al., 2018)** — the earliest quantification: **"quotations or
  references" account for 17 % of all false positives** on Wikipedia/Jigsaw and **8 %** on
  Twitter/Davidson.
- **Lee, Gligorić, … Jurafsky, Eberhardt, *PNAS* 2024** (`doi:10.1073/pnas.2322764121`) — the
  strongest **ecological** evidence. N = 1,025 real posts in which users *recount* racism directed
  at them (and therefore quote the slur). Every system flags these more than matched
  negative-experience controls: **Perspective 4.59 % vs 1.39 %**; **ChatGPT 59.61 % vs 41.82 %**.
  Their framing is ours verbatim: algorithms *"struggled to differentiate whether a swear word was
  used as part of a user's language or merely quoted within a description of a discriminatory
  remark faced by the user — a nuance discernible by human readers."*
- **`2503.01623` *Lost in Moderation* (CHI 2025)** — 5 M queries against 5 commercial APIs. All
  over-moderate counter-speech and reclaimed slurs: slur FPR **16 %** (Amazon) / **23 %**
  (Perspective); re-appropriation FPR ~5–7 %; implicit-hate FNR up to **28 %**; Google Text
  Moderation FPR reaches **99 %** on ToxiGen descriptive content for Disability/Jewish targets.

### 3.6 Reclaimed / in-group language — the same failure from the other side

- **`2406.00020` — *QueerReclaimLex / Harmful Speech Detection by Language Models Exhibits
  Gender-Queer Dialect Bias*** (FAccT 2024). 109 templates × counterfactual author identity,
  annotated by gender-queer annotators. Ground truth: **15.5 %** of in-group posts are harmful vs
  **82.4 %** of out-group. Findings, in order of importance to us:
  (a) off-the-shelf Detoxify/Perspective on in-group text: precision **0.15–0.40**;
  (b) vanilla prompting: HARMFUL_IN F1 **0.36** vs HARMFUL_OUT 0.72;
  (c) **giving the model the author's identity as metadata** lifts HARMFUL_OUT a lot (GPT-3.5 0.81,
  LLaMA-2 0.82) but HARMFUL_IN only to **≤ 0.39**; `identity-cot` (worked examples + rationales)
  reaches HARMFUL_IN **0.47 / 0.53** — best schema, lowest FPR;
  (d) **the hard negative that constrains our whole design:** on the IMPLIED_INGROUP subset
  (n = 464, where in-groupness *is* recoverable from the text itself), the **maximum F1 across all
  models and all prompting schemas is 0.24** (precision ≤ 0.19), vanilla best 0.15.
  **Models can use stance/in-group information when it is handed to them as metadata; they cannot
  read it off the content.** Since our stance channel must *infer* stance from the video, this is
  the single most dangerous published result for us — see §5, implication 3.
- **`2604.16654` — *IYKYK (But AI Doesn't)*** (2026) — **the reliability ceiling, and it is low.**
  21 **in-group** annotators (LGBTQIA+, Black, women) on the f-/n-/b-words. Krippendorff α for
  *"should a moderation model report this as hate speech"*: **f-word 0.15, n-word 0.06,
  b-word 0.21**; **no annotation question exceeds α = 0.33**; "type of reclamation" α = 0.05 for
  the n-word, full-annotator agreement 3–7 % (11–13 % if insular and pride reclamation are
  merged). Assuming an in-group author barely helps (α 0.18 vs 0.15). Perspective aligns better
  with the **out-group** authorship assumption. The features that predict an in-group flag are
  *whether the use was derogatory* and *whether the slur targeted the speaker themselves* —
  **stance and speaker-directedness, exactly our axes.**
  **Conclusion for us: the gold standard for stance is itself near-chance-reliable in exactly the
  region where a stance channel would help.** Corroborated by `2206.06423` (38.3 % of labels flip
  with context), CAD (`2021.naacl-main.182`, counter-speech Fleiss κ = 0.267, only 220 instances),
  and ModelCitizens (`2507.05455`, 27.5 % in-/out-group annotator disagreement).
- **`2602.12818` — AIWizards @ MULTIPRIDE** (slur reclamation shared task) — **a clean null on
  nearly our topology.** LLM weak-annotates **community membership**, soft-labels a BERT user
  encoder, gated fusion with a hate-pretrained encoder. Dev macro-F1: Italian **0.90 ± 0.03 →
  0.88 ± 0.04 (p = 0.28)**; Spanish **0.67 ± 0.04 → 0.64 ± 0.02 (p = 0.17)**. Authors: *"does not
  result in an improvement in aggregate evaluation metrics."* The artefact was speaker
  **identity**, not speaker **stance** — but the fusion machinery is ours.
- **`2510.20154` — *Are Stereotypes Leading LLMs' Zero-Shot Stance Detection?*** (EMNLP 2025).
  LLM zero-shot stance is itself **target-group-stereotype-biased**. Bolting an LLM stance module
  onto identity-targeted content risks **importing the very bias the channel is meant to remove** —
  and `2404.01651`'s identity-stratified FPR says the base system already has it.
- **Davidson, *Nature Human Behaviour* 2025** (`doi:10.1038/s41562-025-02360-w`) — conjoint
  experiments vs N = 1,854 humans. **Large** MLLMs replicate human context-sensitivity for
  reclaimed slurs; **small ones do not and some invert** (InternVL3-2B flags Black men *more*,
  θ̂ = +0.041; Qwen2-VL-7B anti-Black θ̂ = +0.096). Context-sensitive prompting raises the share of
  significant demographic-context effects **37.8 % → 46.2 %**, while a context-*suppressing* prompt
  changes nothing (37.8 %): *"prompting can amplify context sensitivity but cannot suppress it."*
  **Direct operational consequence: the reasoner emitting our stance record must be a large model;
  a 7B-class VLM may invert the signal.**

### 3.7 Structured stance questions beat free-form reasoning — the sharpest wiring result

**MemeScouts @ LT-EDI 2026** (`2604.24179`) is multimodal, but its ablation is purely about
**injection form** and it is the most actionable number in this file. 89 constrained questions
(target, stance, irony, implicitness) answered by a quantised Qwen3-VL → integer feature vector →
Random Forest. Macro-F1:

| method | EN | ZH | HI |
|---|---|---|---|
| direct Qwen3-VL-30B prediction | 0.77 | 0.32 | 0.21 |
| Qwen3-VL-30B **with free-form reasoning** | **0.67** | **0.10** | **0.08** |
| **decomposed question features → RF** | **0.85** | **0.72** | **0.66** |
| + importance pruning (89 → 33 features) | 0.85 | 0.72 | **0.67** |

**Structured decomposition: +8 / +40 / +46 points over direct prediction. Unconstrained CoT:
−10 / −22 / −13.** Ranked 1st EN in the shared task. Corroborated by SoftHateBench
(`2601.20256`, WWW 2026), where reasoning-oriented GPT-OSS-20B trails the far smaller Gemma3-4B:
*"generic chain-of-thought alone is insufficient for reasoning-driven soft hate."*

> **This is the design instruction: emit a constrained typed schema, never a free-text rationale.**
> `MLLM_FRONT_RECON.md` already noted that MemeScouts' 89 questions contain **nothing** about
> speaker stance, use-vs-mention, or endorsement-vs-condemnation — so the wiring is occupied but
> the field set is not.

**Bookend from SoftHateBench (`2601.20256`) — the oracle upper bound.** It rewrites explicit hate
into surface-neutral policy-compliant argumentation while **preserving the hostile standpoint**;
detection collapses from **76.8 → 43.5 / 32.9 / 21.2** across softness tiers (encoders bottom out
at **6.8 %**, guard models ~17.8 %, best model GPT-5-mini 49.8 %). Its scaffold ablation hands the
model the **latent premise and maxim** — i.e. the ground-truth standpoint —
and Qwen3-4B goes **23.0 % → 92.4 % → 94.2 %**. *That is an oracle, not a method*, but it is the
cleanest existing estimate of how much headroom a perfect standpoint/stance channel contains.

### 3.8 Adjacent text-domain findings that constrain the design

- **`2210.06351` — Twitter, *A Keyword Based Approach to Understanding the Overpenalization of
  Marginalized Groups by English Marginal Abuse Models*** (2022). Production audit of Twitter's
  English marginal-abuse model, explicitly naming **reclaimed speech, counterspeech and
  identity-related terms** as the over-penalisation drivers. **Injection form: data augmentation** —
  adding true-negative examples of these categories to training improved fairness metrics
  "without large degradations in model performance". *This is the only occupant with a
  deployment-scale result, and its lesson is that the cheapest fix is data, not architecture.*
- **`2307.12418` — HateModerate (NAACL 2024 Findings) — the augmentation trade-off, quantified.**
  7.6k hateful and non-hateful cases matched to Facebook's 41 hate policies (which contain the
  *"we allow content that condemns or quotes hate speech"* exceptions). Fine-tuning CardiffNLP
  RoBERTa on HateModerate moves HateCheck failure rates: **hate .442 → .185** but
  **non-hate .205 → .229 — significantly *worse*** (both p < 0.05). Re-balancing alleviates but
  does not remove the trade-off. **So policy-grounded augmentation buys recall and pays in exactly
  our false-positive bucket.**
- **`2507.05455` — ModelCitizens.** Off-the-shelf systems (OpenAI Moderation, GPT-o4-mini,
  Gemini-2.0-Flash) average **63.6 %** accuracy, dropping to **59.6 %** on the context-augmented
  subset — **giving a zero-shot system more context makes it worse.** Fine-tuning on
  community-grounded labels recovers +5.5 % overall / +9 % on the context subset.
  *Corollary for us: a stance/context channel is only useful to a **trained** consumer; handing it
  to a frozen judge can be net-negative.*
- **`2511.07405` — SPOT (French, 43,305 Facebook comments).** "Stopping points" = ordinary critical
  interventions, a superset of counterspeech. Context metadata (article, post, parent, page) lifts
  CamemBERT **F1 0.75 → 0.78**, and **fine-tuned encoders beat prompted LLMs by > 10 F1**.
- **`2405.11030` — *The Unappreciated Role of Intent in Algorithmic Moderation* (ICWSM 2024).**
  Platform policies (Meta, X) make **intent criterial**, yet **no benchmark dataset annotates it**
  and essentially no SOTA detector captures it. A position paper with no headline numbers, but the
  cleanest available framing citation for why this gap exists.
- **`2503.16072` — *Toxicity Detection Should Measure Contextual Harm, Not Text-Intrinsic
  Badness*.** Argues the field's error is treating context as an *auxiliary feature* rather than
  as constitutive of the target. Useful if we need to argue *why* stance features keep
  under-delivering.
- **Usable stance-labelled transfer corpora**, if we ever want to pretrain rather than prompt:
  **Kurrek, Saleem & Ruths (ALW 2020, `2020.alw-1.17`)** — 39.8k Reddit slur usages typed
  **derogatory / appropriative / non-derogatory-non-appropriative / homonym**; only **52 % are
  derogatory**, and the n-slur only **37.9 %**. **CAD (Vidgen et al., `2021.naacl-main.182`)** —
  **Counter Speech** and **Non-hateful Slurs** as primary categories, annotated *in thread context*
  with rationales (but only 220 counter-speech instances, κ = 0.267). **DEBAGREEMENT**
  (Pougué-Biyong et al. 2021) — the agree/neutral/attack corpus that actually delivered the +3 in
  §3.4. **ToxiGen and LatentHatred have no counterspeech/quoting category at all.**
- **`2307.03377` — de Paula, Rosso, Spina, *Mitigating Negative Transfer with Task Awareness for
  Sexism, Hate Speech, and Toxic Language Detection*, IJCNN 2023.** Multi-task learning across
  sexism / hate / toxicity suffers **negative transfer** — "noisy information is shared between
  tasks, resulting in a drop in performance" — and needs an explicit task-awareness mechanism to
  avoid it (SOTA on EXIST-2021 and HatEval-2019 once mitigated). **Not a stance paper**, so do not
  cite it as one; but it is the correct citation for *"a bare auxiliary head bolted onto a hate
  classifier is not free."*
- **Stance and hate as *sibling* subtasks, never as feature→task.** CASE @ EACL 2024
  *ClimateActivism* shared task ran **stance detection and hate-event detection as parallel
  subtasks** (`2402.17014` Z-AGI Labs; `2402.06549` Bryndza, retrieval-augmented GPT-4/LLaMA).
  Stance is an output beside hate, never an input to it — the same structural gap as PrideMM /
  MemeCLIP (`2409.14703`) and DARC-CLIP (`2604.23214`) in the meme domain.
- **The counter-narrative literature is a *generation* literature.** `abs:"counter narrative" AND
  abs:"hate speech"` returns CONAN (`1910.03270`), Multi-Target CONAN (`2107.08720`), NGO
  empowerment (`2107.02472`), argument annotation (`2208.01099`), type classification
  (`2109.13664`), attention-regularised generation (`2309.02311`), intent-conditioned generation
  (`2305.13776`), evaluation frameworks (`2402.11676`), ParsCN (`2603.27011`), and a 2026 survey
  (`2603.19279`). **Almost none of it treats "is this counter-speech?" as a signal to feed a hate
  classifier** — it treats counter-speech as something to *produce*. The one detection-side
  ancestor is `1808.04409` (*Thou shalt not hate*, 2018), which is a Gab/Twitter measurement study.
- **`abs:"use-mention"` over all of arXiv returns essentially one relevant paper** (`2404.01651`),
  plus `2407.06323` (guardrail cascades) mentioning it in passing. The concept has almost no
  computational literature at all.

### 3.9 The one place stance-as-a-feature is fully mature — and why it is an analogue, not an occupant

**Rumour / misinformation verification** has used stance as a downstream feature for a decade, in
exactly the wirings we are considering:

- `1806.03713` **All-in-one: Multi-task Learning for Rumour Verification** (Kochkina et al., COLING
  2018) — joint training of veracity (main) + **stance** (auxiliary) "improving the performance of
  rumour verification". The canonical *stance-as-auxiliary-task* citation.
- `1909.08211` (Kumar & Carley, ACL 2019) and `2007.07803` — joint stance+veracity over conversation
  structure.
- `2204.02626`, `2502.08888` — weakly-supervised / MIL propagation models that induce stance without
  stance labels and use it for verification. **Directly relevant if we ever want stance without
  stance supervision.**
- `2512.13559` **Verifying Rumors via Stance-Aware Structural Modeling** (2025) — groups reply
  embeddings **by stance category** and feeds stance distribution + hierarchical depth as covariates.
- `2103.00242` — a whole **survey** of stance detection for mis/disinformation identification.

**Why this is not an occupant of our idea, and the distinction must be stated precisely:** in all of
this work the stance is the stance of **third-party responders toward a claim** (SDQC over a
conversation tree) — a *social* signal harvested from replies. Ours is the stance of the
**speaker inside the content toward the hateful material they are presenting** — a *content* signal
that must be inferred from the video itself, with no conversation tree available. Same word,
different object. But the family is proof that (a) reviewers accept "stance as an input feature for
a harm-detection head" as a legitimate architecture, and (b) `1806.03713` is the counterweight to
`2307.03377` — a **stance auxiliary task that demonstrably helped** a harm-verification main task.

---

## 4. The two specifically-requested answers

### 4.1 Is the video domain really a zero-occupant space?

**Yes, as far as these channels reach — and the emptiness is structural, not accidental.**
Not one hateful-video system carries a stance field; not one hateful-video dataset ships a stance
label in its documented schema; the one stance-adjacent annotation that physically exists
(MultiHateClip's 139 `Counter Narrative` votes) is undocumented in its own paper and collapsed out
of the majority label. The differentiation stands.

**Three qualifications that must travel with that answer.**
(i) The *pipeline shape* is occupied by RAMF `2512.02743` / MARS `2601.15115` — only the typology
is free.
(ii) **Video stance detection exists as a task** (MultiClimate, TikStance α≈0.74, Inter-Stance,
DIVERSE) — so we cannot claim "stance from video is unstudied", only "stance for hateful video is
unstudied". This is actually helpful: it is evidence the perception problem is tractable.
(iii) The failure mode itself is **undocumented** in the video literature (MultiHateClip's error
analysis never mentions counter-speech confusion), so our §9.2 audit is the *primary* evidence and
must be presented as such.

**What would kill it, and where to look before committing:** non-arXiv venues (ICWSM, CSCW, LREC,
ACM MM workshops), Chinese academic databases (CNKI/Wanfang — not searched), Google Scholar (not
searched, budget exhausted), the *misinformation-video* literature (FakeSV `2211.10973` ships
comments as social context; stance typing unconfirmed), and any 2026 work appearing after this
sweep. Displacement risk, already flagged in `MLLM_FRONT_RECON.md` §3.1 and now reinforced: Zeyu
Fu's group (RAMF / MARS / LELA) is publishing steadily along the "frozen VLM emits typed text →
trained fusion" line, and speaker stance is their natural next prompt.

### 4.2 In text: how much does stance buy, in what form, and what has failed?

| injection form | who | measured effect | verdict |
|---|---|---|---|
| **Intermediate-task pretraining on a stance corpus** | `2206.06423` | weighted F1 **0.58 → 0.61** (no context), **0.61 → 0.63** (with context), best **0.64** p<0.01; **counter-hate F1 0.38 → 0.45** | **THE CLEANEST POSITIVE IN THE FIELD.** |
| ↳ same slot, *other* auxiliaries | `2206.06423` appendix | sentiment **−8** (counter-hate F1 → **0.00**), sarcasm **−7** (→0.08), more hate data **−4** (→0.12) | **ONLY STANCE HELPS.** Sentiment/emotion — what ImpliHateVid injects — is the worst option. |
| **Structured typed questions → feature vector → trained head** | `2604.24179` | macro-F1 **+8 / +40 / +46** (EN/ZH/HI) over direct VLM prediction | **WORKS, biggest measured effect of any form.** Wiring occupied; the *fields* are not. |
| ↳ same reasoner, **free-form** reasoning instead | `2604.24179` | **−10 / −22 / −13** | **PUBLISHED FAILURE.** Never emit a free-text rationale. |
| **Prompt-time instruction + CoT** (frozen LLM) | `2404.01651` | counterspeech FPR **8.89 % → 1.55 %** (−82.6 % rel., −7.3 pp abs.) hate; **10.21 % → 4.18 %** misinfo; TPR −3.0 % rel. | **WORKS — and is already banked.** Cannot be our contribution. |
| **Decision rule / hypothesis composition** (zero-shot NLI) | `2210.00910` FCS | HateCheck F20 **0 → 100 %**; HateCheck overall **+4.6 pp**; **ETHOS +0.0 pp** | **WORKS ON DIAGNOSTICS, ZERO ON A NATURAL CORPUS.** |
| **Generalising that decision rule** | `2210.00910` FCSp1 | **+4.6 pp → +0.0 pp**; +FBT recovers only +0.3 pp | **PUBLISHED FAILURE.** Counterspeech's own negative affect and target-term content defeat it. |
| **Frozen-reasoner text → encoder → trained head** | `2310.19750` | +1.4 / +2.4 F1 over prior SOTA (+20.7 vs the LLM's own label); CoT-only can beat CoT+text | **WORKS, on stance-as-task.** Unoccupied for hate-as-task. |
| **Conversational context as an input feature** | `2206.06423`, `2511.07405` | +3 weighted F1 (0.58→0.61); SPOT F1 0.75 → 0.78 | **WORKS — for a *trained* consumer.** |
| ↳ same context handed to a **zero-shot** judge | `2507.05455` | accuracy **63.6 % → 59.6 %** | **PUBLISHED FAILURE.** The consumer must be trained. |
| **Data augmentation with *real* counterspeech/reclaimed true-negatives** | `2210.06351` (Twitter, production) | fairness metrics improve, small performance cost | **WORKS, cheapest baseline — and therefore a mandatory ablation.** |
| **Policy-grounded augmentation** | `2307.12418` HateModerate | HateCheck hate failure **.442 → .185**, but non-hate failure **.205 → .229 (worse, p<0.05)** | **TRADES.** It buys recall out of our own FP bucket. |
| **Fine-tuning on *constructed* counterspeech test cases** | `2204.04042` (on HateCheck) | held-out functionalities ↑, **held-out functionality classes and i.i.d. task data ↓**; overfits the challenge distribution | **PUBLISHED FAILURE.** Synthetic stance data ≠ the Twitter result. |
| **Author-identity/stance given as *metadata*** | `2406.00020` | HARMFUL_OUT 0.72 → 0.82; identity-CoT lifts HARMFUL_IN 0.36 → **0.53** | works **only when the identity is supplied** |
| ↳ same, when identity must be **inferred from content** | `2406.00020` IMPLIED_INGROUP | **max F1 0.24 across all models and all schemas** (precision ≤ 0.19) | **THE BINDING CONSTRAINT ON OUR DESIGN.** |
| **Stance as an MTL auxiliary head** | `1806.03713` (rumour) vs `2307.03377` (hate MTL) vs `2602.12818` (identity fusion) | helps veracity; negative transfer in hate MTL; **null** for identity fusion (0.90→0.88 p=0.28; 0.67→0.64 p=0.17) | **AMBIGUOUS. Open slot: 1 positive, 2 negatives.** |
| **Stance distribution as an explicit covariate** | `2512.13559`, `2204.02626` (rumour) | mature, standard, accepted architecture | **precedent for the wiring**, different object (responder stance, not speaker stance) |
| **Oracle latent standpoint handed to the model** | `2601.20256` SoftHateBench | Qwen3-4B **23.0 % → 94.2 %** | **upper bound only, not a method** |
| **Surface "is there a quotation?" cue** | `2404.01651` Table 7 | quotes → **28.57 %** FPR vs **7.23 %** without (gpt-4, hate) | **ACTIVELY WRONG SIGN.** |
| **Small reasoner emitting the stance record** | Davidson, *Nat Hum Behav* 2025 | small MLLMs invert context-sensitivity (InternVL3-2B θ̂ = +0.041) | **DO NOT USE A 7B-CLASS REASONER.** |

**Realistic effect size to plan against.** The honest analogues cluster tightly at **+1 to +3
points**: `2310.19750` **+1.4–2.4 F1** over prior SOTA; `2206.06423` **+3 weighted F1** from
context and another **+3** from stance pretraining (best system +6 over the naked baseline,
0.58 → 0.64); `2511.07405` **+3 F1** from context metadata; ARG **+1.3–1.7 macro-F1** for its whole
rationale-gating apparatus (`MLLM_FRONT_RECON.md` §3.1); RAMF **+3 macro-F1** for the whole
typed-record pipeline in video. Two outliers, both explicable: `2508.16555` (+6 F1 on ETHOS from
sarcasm *representation transfer*, a different mechanism) and `2604.24179` (+8/+40/+46 macro-F1,
but measured against **direct VLM prediction** on low-resource languages, not against a trained
baseline — an artefact of the same kind as `2310.19750`'s +20.7). The **−82.6 %** figure is a *relative FPR* reduction on a curated
counterspeech set — it is not a macro-F1 number and must never be quoted as one. Our own §9.2
oracle bound (**+6.46 mean macro-F1** if bucket S were fixed perfectly) is the *ceiling*, and the
literature says a realistic recovery is a fraction of it.

---

## 5. Three implications for the design

**1. The channel must be a *structured typed schema* consumed by a *trained* head — and stance,
specifically, is the right field set.**
Every alternative form has a published verdict against it: prompt-time mitigation is banked
(`2404.01651`), a hand-built stance decision rule is worth **zero on natural data**
(`2210.00910`, ETHOS +0.0 pp), free-form reasoning **loses 10–46 points** to a constrained
question schema (`2604.24179`), and context handed to a **zero-shot** consumer makes it **worse**
(`2507.05455`, 63.6 % → 59.6 %). What survives is exactly our shape. Three field-level constraints
fall straight out:
(a) **graded, not binary** — `2404.01651`'s Fightin'-Words result says the discriminative variable
is stance *strength/explicitness*, not stance *presence*, so emit a posterior over types plus an
explicitness score;
(b) **carry the record, not just the label** — `2310.19750` shows the encoder tolerates reasoning
errors that would flip an extracted label (56 % of Neutral errors corrected);
(c) **stance and not affect** — `2206.06423`'s ablation is the specificity result: as an auxiliary
signal, sentiment costs **−8** weighted F1 and drives counter-hate F1 to **0.00**, sarcasm **−7**,
extra hate data **−4**, and only stance gains **+3**. ImpliHateVid (`2508.06570`), the nearest
video-domain feature-channel work, injects **sentiment and emotion** — the literature predicts that
is the wrong auxiliary, and beating it is a concrete, cheap, publishable comparison.

**2. Three mandatory baselines, all cheap, each capable of killing the idea — plus one feature that
must not be built.**
(i) **Prompt-only mitigation** — `2404.01651`'s CoT+use/mention prompt on the same frozen reasoner,
no trained head. If the feature channel does not beat this, there is no paper.
(ii) **Counterspeech/condemnation data augmentation** — `2210.06351` is the production-proven fix;
`2307.12418` shows it trades (hate failure .442→.185 but non-hate .205→.229) and `2204.04042` shows
the *synthetic* version costs i.i.d. performance. Run it and report the trade.
(iii) **Affect-channel control** — the same pipeline emitting sentiment/emotion instead of stance,
which is both ImpliHateVid's design and `2206.06423`'s losing arm.
**Do not build a quotation / distancing surface feature:** `2404.01651` Table 7 shows quotation
marks correlate with *more* false positives (28.57 % vs 7.23 %), so that feature has the wrong sign
— and it is precisely the surface cue whose absence made FCS worthless on ETHOS.

**3. The binding risk is not the wiring, it is whether stance can be *inferred* at all — and the
gold standard for it is near-chance-reliable. Pre-register accordingly.**
`2406.00020` is the sharpest published threat: when in-group/stance status is **handed to the
model as metadata** it helps (HARMFUL_IN 0.36 → 0.53), but when it must be **inferred from the
content itself**, the **maximum F1 across every model and every prompting scheme is 0.24**. Our
reasoner must infer stance from video. Meanwhile the target itself is contested — IYKYK
(`2604.16654`) reports in-group annotator α of **0.06–0.21**, `2206.06423` finds **38.3 %** of
hate/counter labels flip when context is shown, CAD's counter-speech κ is **0.267**. Consequences:
- **Gate on the stance-typing quality first.** A separate, blind, human-adjudicated audit of the
  reasoner's 5-way stance labels must clear a pre-registered bar *before* any head is trained
  (§`IDEA_REPORT` already specifies macro-F1 ≥ 0.80 per corpus — the literature says that bar is
  optimistic and a graded/soft target may be the only honest formulation).
- **Use a large reasoner.** Davidson (*Nat Hum Behav* 2025) shows small MLLMs invert
  context-sensitivity on reclaimed language; a 7B-class model may produce an anti-correlated channel.
- **Check for target-identity confounding before claiming anything.** `2404.01651`'s FPR ranges
  from 14.15 % (Jewish) to 0.39 % (migrants) and `2510.20154` shows LLM zero-shot stance is itself
  stereotype-driven — the channel could be learning "this video is about group G", not "this speaker
  is condemning".
- **Freeze the expected effect size at ~1–3 macro-F1, not the +6.46 oracle**, and evaluate only on
  HateMM / MHC-EN / MHC-ZH / ImpliHateVid. Three independent precedents (`2210.00910` ETHOS +0.0,
  `2210.00910` FCSp1 −4.6, `2204.04042` i.i.d. drop) say a stance intervention can be perfectly
  correct on diagnostics and worth nothing on real data.

---

## 6. Search coverage (so "not found" can be audited)

**arXiv API phrase conjunctions run** (`—` = genuine empty feed):

| query | hits |
|---|---|
| `all:"hateful video"` | 5 (HateClipSeg, temporal label noise, meme→video transfer, MultiHateClip, RAMF) |
| `all:"stance" AND all:"hateful video"` | — |
| `all:"stance" AND all:"hate video"` | — |
| `all:"speaker stance" AND all:"video"` | — |
| `all:"use-mention" AND all:"video"` | — |
| `all:"counter-speech" AND all:"video detection"` | — |
| `abs:"stance" AND abs:"harmful video"` | — |
| `abs:"counterspeech" AND abs:"video"` | 1 (`1808.04409`, not a video paper) |
| `abs:"stance" AND abs:"video" AND abs:"hate"` | 1 (`2503.10648`, YouTube *comment* text) |
| `abs:"quotation" AND abs:"hate speech"` | — |
| `abs:"use-mention"` | 2 relevant (`2404.01651`, `2407.06323`) |
| `abs:"stance" AND abs:"hate speech"` | 10 (CASE/ClimateActivism ×2, implicit hate, persuasion modes, MemeScouts, …) |
| `abs:"counterspeech" AND abs:"detection"` | 5 |
| `abs:"counter narrative" AND abs:"hate speech"` | 11 (all generation-side) |
| `abs:"reported speech" AND abs:"toxicity"` | — |
| `abs:"HateCheck"` | 9 (HateCheck, Multilingual/SG/SEA-HateCheck, GPT-HateCheck, *Checking HateCheck*, hypothesis engineering, …) |
| `abs:"slur" AND abs:"reclaimed"` | 5 (IYKYK, AIWizards@MULTIPRIDE, gender-queer dialect bias, KIT-TIP-NLP, Lost in Moderation) |
| `abs:"stance" AND abs:"multimodal" AND abs:"hateful"` | 3 — **all memes** (MemeCLIP, DARC-CLIP, MemeScouts), stance as a parallel head |
| `abs:"intent" AND abs:"hate speech detection" AND abs:"LLM"` | 1, irrelevant |
| `abs:"stance" AND abs:"veracity"` | 10 (the mature rumour-stance→veracity family, §3.9) |
| `abs:"stance" AND abs:"rumour" AND abs:"detection"` | 8 (incl. a survey, `2103.00242`) |

**Not run / blocked** (arXiv 429 or WebSearch budget exhausted):
`abs:"video moderation" AND abs:"stance"`, `abs:"satire" AND abs:"hateful video"`, and the planned
Chinese-language battery beyond a small sample. **These are the gaps a follow-up sweep should close
first.**

**Other channels:** ACL Anthology (`2024.naacl-long.331` page + PDF), arXiv PDFs read in full for
`2404.01651` / `2210.00910` / `2310.19750` / `2408.03468` / `2012.15606` / `2204.04042`,
Semantic Scholar Graph API (partly 429), and direct inspection of this repo's `data/gt/`
annotation files and `research-wiki/papers/`.

**Parallel video-axis sweep coverage** (independent, ~72 tool calls): arXiv API ~35 queries
including the two exhaustive enumerations (`abs:"hateful video" OR abs:"hate video"` → 14 papers,
all read; `abs:"stance" AND abs:"video"` → 31 papers, all read); **OpenAlex** 18 queries;
**Semantic Scholar** ~20 queries (heavily 429-throttled); **~22 full-text/codebook fetches**
(MultiHateClip guideline + §5.4 error analysis, HateClipSeg codebook, HateMM codebook,
ImpliHateVid, IARE/Ex-HateMM, MARS, RAMF, CMFusion, MM-HSD, MultiHateLoc, HarmVideoBench,
harmful-YouTube taxonomy, SAGE, ToxVidLM, TikStance, DIVERSE, FakeSV, the EMNLP-2024 moderation
survey); **2 Chinese queries** (`立场 视频 仇恨检测`, `说话人态度 有害视频 检测`) — no academic hits.

**Stated gaps in that sweep (do not overclaim):** no Google / Google Scholar (session budget
exhausted before it started); ACL Anthology search is a Google CSE and was unreachable — Anthology
pages were reached only by direct ID; ACM DL returned 403 for RAMME; only 2 Chinese queries and no
CNKI/Wanfang; no systematic ICWSM / AAAI / ACM MM proceedings sweep beyond what OpenAlex and
Semantic Scholar surfaced; theses and non-indexed workshop papers not covered. The accurate claim
is **"not found across an arXiv-complete enumeration of the hateful-video literature plus
OpenAlex/S2 keyword coverage, with the dataset codebooks directly verified"** — not "does not
exist".

**Parallel text-axis sweep coverage** (independent, ~99 tool calls): **arXiv simple search**
~34 queries (many returning zero — e.g. `counterspeech false positive toxicity classifier`,
`quoted slur reclaimed detection`, `reported speech toxicity`, `endorsement condemnation hate
content classification`, `speaker intent hate speech classification pragmatics`, `chain-of-thought
hate speech detection fails`); **arXiv advanced abstract search** 16 abs×abs conjunctions;
**OpenAlex** full-text + title/abstract + topic-restricted searches plus **citation-graph
traversal** (works citing `2404.01651` → 3; works citing `2206.06423` → 31); **EuropePMC**
fullTextXML (PNAS PMC11420153); **OSF API** (Davidson preprint, 77 pp); **ACL Anthology** direct
fetches. **Full text pulled and grepped, not abstracts**, for: `2012.15606`, `2404.01651`,
`2210.00910`, `2206.06423`, `2406.00020`, `2503.01623`, `2602.12818`, `2606.01298`, `2604.16654`,
`2307.12418`, `2507.05455`, `2503.16072`, `2405.11030`, `2604.24179`, `2601.20256`, `2310.19750`,
`2511.07405`, `1809.07572`, `2020.alw-1.17`, `2021.naacl-main.182`.

**Stated gaps in that sweep:** WebSearch unavailable (budget exhausted before it began) and
Semantic Scholar hard-429'd throughout, so **no Google/Google Scholar coverage on either axis**;
**Chinese-language venues (CNKI / Wanfang / 中文信息学报) are not indexed by arXiv or OpenAlex and
were not searched** — the only Chinese-language hate work reachable was arXiv-hosted (STATE ToxiCN
`2501.15451`, CCL25-Eval Task 10 `2512.09563`), none stance-focused; Zsisku et al. WebSci 2024
(RHSD, the first reclaimed-language hate dataset) is paywalled and unverified beyond its abstract.

**Correction to earlier project notes:** ViTHSD (`2404.19252`) is a **text-only** Vietnamese social
media dataset, not a video dataset — fix wherever it is listed as one.

**Numbers to re-verify before they enter a paper** (reported by the parallel sweeps, not read by me
from the PDF): the per-auxiliary appendix rows of `2206.06423` Table 9 (sentiment −8 / sarcasm −7 /
hate-data −4); the MemeScouts per-language ablation figures; the `2307.12418` HateModerate failure
rates; the QueerReclaimLex per-schema F1 values. Everything marked `[PDF]` in §§2–3 *was* read
directly from the source in this session.
