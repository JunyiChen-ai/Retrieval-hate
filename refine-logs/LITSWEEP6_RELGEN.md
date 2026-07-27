# LITSWEEP-6 — RELATIONAL + GENERATIVE MECHANISMS

**Date:** 2026-07-27 · **Agent:** litsweep-relgen · **GPU spent:** 0 · **Test contact:** NONE
**Lens:** convert the campaign's two *proven-but-unconverted* assets — (1) relational n² supervision
works at the relation level (F95 control 1, 18/18 cells) and (2) the memory/audit capability story —
into something claimable.
**Inputs read before searching:** `autoresearch/goal_mllm_plus3/state/directions_tried.json`
(63 dead entries, 9 banned constraints), `refine-logs/MECHNOV_PAIRVERIFY_PREGATE.md` (F95 in full),
plus F47/F63/F66/F89/F94 ban scopes.

---

## §0. TWO PREMISE CORRECTIONS BEFORE ANYTHING ELSE

**(a) The pair-score matrices are NOT banked.** The tasking assumed "our banked pair-score matrices
from F95 are reusable". They are not: `scripts/analysis/mechnov_pairverify.py` contains **zero**
`np.save`/`torch.save` calls; only per-cell summary JSON is written (`mechnov_parts/*.json`,
`mechnov_pairverify_{hatemm,zh,en}_OUT.json`). The `S[query, bank]` matrices and all per-item
fixed/broken identities were computed in memory and discarded.

This is *not* a blocker — it is a ~10-minute CPU cost. Timing from `.mechnov_drive.log`: each
dataset×space cell is 5 folds × 11-16 s ≈ **60 s**, so all 9 cells regenerate in **~9 minutes on
≤8 threads**. Every pregate below therefore begins with the same regeneration step. Ceremony note:
the frozen arms module (sha256 `77b0defd…7b240d`) must **not** be edited to add persistence — write
a *new* emitter script that imports it, exactly as `mechnov_pairverify_runner.py` already does, and
freeze that separately before any real-data number.

**(b) Arena caveat inherited from F95, and it is severe.** All numbers below live in
**raw banked key space, on the train split, under leave-one-fold-out**, not in deployed head space
and not on test. F95 §6 states this limitation itself. Everything in this document is therefore a
*pregate quantity*, and the campaign's history (F47, F66, F89) is that raw-space oracles do not
survive the trip to the deployed head. Stated once here, assumed throughout.

---

## §1. THE ARITHMETIC EVERY CANDIDATE MUST DEFEAT

Recomputed directly from `mechnov_pairverify_*_OUT.json` (fused × MLP × max), not transcribed:

| dataset | n | deployed acc | shape-only (2b) | **shape cost** | deployed errors | pathology pop | fixed | broken | net | exchange rate |
|---|---|---|---|---|---|---|---|---|---|---|
| HateMM | 744 | 0.8441 | 0.8024 | **−0.0417** | 116 | 88 | 54 | 57 | −3 | 0.9474 |
| MHC-ZH | 579 | 0.8480 | 0.8187 | **−0.0293** | 88 | 79 | 31 | 58 | −27 | 0.5345 |
| MHC-EN | 549 | 0.7796 | 0.7359 | **−0.0437** | 121 | 109 | 49 | 57 | −8 | 0.8596 |

Four walls, in the order they will kill things:

1. **Shape cost (F95 control 2b).** Discarding the rank-weighted top-20 average costs −0.029 to
   −0.044 acc *before any verifier runs*. Any candidate that changes the aggregation shape starts
   in a hole roughly the size of the target.
2. **Exchange-rate law (F95 §4).** Best exchange rate anywhere in a 36-cell battery: **1.1667**.
   Fixes rose 10× over F89 and the rate did not move. Symmetric operators pay par or worse.
3. **Selection lock (F66, F47, F94).** 91-98 % of oracle headroom is formally selection-locked;
   decision-level meta-features carry no per-item routing signal at any of three supervision
   sources; k∈[1,60] is closed both directions and F94's ban text explicitly generalises to
   "no truncation **or re-weighting** of the retrieved list reaches it".
4. **Graph route pre-charged (F63).** Multi-hop label propagation over the frozen kNN graph is
   monotone-negative in α, at/below one-hop on all three datasets. F63's ban scope does explicitly
   *not* price **learned graph weights** — that is the only crack, and it is narrow.

**The one number that is genuinely new and genuinely encouraging.** F95 measured only *ungated*
adjudication. Gate it perfectly and the same machinery gives, per dataset, `fixed × (1/n)`:

| dataset | oracle-gate ceiling | items needed for +0.030 | gate set N (=fixed+broken) | base rate of "fix" | **break-even precision** |
|---|---|---|---|---|---|
| HateMM | **+0.0726** | 22.3 | 111 | 0.4865 | **0.6005** |
| MHC-ZH | **+0.0535** | 17.4 | 89 | 0.3483 | **0.5976** |
| MHC-EN | **+0.0893** | 16.5 | 106 | 0.4623 | **0.5777** |

**The oracle is above the +0.030 bar on all three datasets** — the first time in several rounds that
a $0 pregate has produced an above-bar oracle on 3/3 (contrast F55 MCR +0.025, graded EN +0.0250 /
ZH +0.0256, both *below* bar). And the required precision is only **0.578-0.601** against base rates
**0.348-0.487**, i.e. a precision lift of +0.09 to +0.25. That is a demanding but not absurd ask,
and it is measurable at zero GPU. Everything in §2 is ranked around this fact.

---

## §2. CANDIDATES (ranked by survival × novelty)

### C1 — VGA: Verifier-Gated Adjudication  ★ rank 1
**CP:** i (aggregation without max-fragility) **and** ii (inversion detection), jointly.

**Verified anchor.** *When Reranking Hurts: Uncertainty-Based Gating for Few-Shot Reranking*,
Orian Dabod, Amir DN Cohen, Gabriel Stanovsky, arXiv:2606.31087 (30 Jun 2026, rev. 1 Jul 2026).
Fetched and confirmed. Headline: across 8 LLMs, 7 NLU datasets and 9 MT domain-language pairs,
*selectively* reranking only high-uncertainty instances **beats always-reranking**, cutting cost
15-80 % while improving average performance by up to 2 %. Secondary anchor, fetched:
*CAR: Query-Guided Confidence-Aware Reranking for RAG*, Song et al., arXiv:2605.04495 (6 May 2026) —
its **query-level gate** leaves the baseline ordering untouched when the query is already confident,
which is the identical design primitive.

**Mechanism (3 sentences, with data flow).** For each query, compute both decisions we already
know how to compute — the deployed rank-weighted top-20 kNN vote, and the F95 nomination +
pair-verifier adjudication — and additionally extract a *gate feature vector* from the verifier's
within-query score profile over the shortlist. A gate `g(·)`, fit on train folds only, predicts
whether adjudication would **fix** or **break** this item; the system emits the adjudicated label
iff the gate fires and the deployed vote otherwise. Data flow: `keys → cosine top-N shortlist →
verifier scores S[q, ·] → {deployed vote, adjudicated label, gate features} → switch → label`.

**Distinctness vs the dead list — the four arguments, stated explicitly.**
- *vs F95 ban clause (c)* ("verifier-as-reranker-inside-the-vote **without first pricing control
  2b**"): the shape cost is priced **to exactly zero on non-gated items by construction** — those
  items receive the deployed vote bit-identically. On gated items the 2b cost is already fully
  contained in the measured fixed/broken counts. The pricing requirement is discharged
  structurally, not argued away.
- *vs F95 ban clauses (a) and (b)*: no head-space verification, no new scorer architecture. The
  frozen MLP arm is reused unchanged. F95's finding that scorer quality is not the binding
  constraint is *accepted*, not contested — this candidate attacks the exchange rate, which F95
  names as the binding term.
- *vs F47* ("do not re-propose per-item selectors over frozen channels … unless the selector input
  is a genuinely NEW information source not derivable from banked features/votes"): F47's features
  were vote margins, purity, sub-votes, confidence differential, transcript stats — all
  *unsupervised functions of the cosine ordering*. The gate's inputs are the **out-of-fold trained
  relation profile**, which F95 control 1 proves carries ordering information the cosine does not
  (within-query AUC +0.1572 / +0.2302 / +0.1785, 5/5 fold signs, 18/18 cells). F47's
  train-supervised leg died of memorisation (CLIP LOO 0.998); the verifier's scores are
  out-of-fold by construction, so that failure mode is structurally excluded. **This argument is
  the load-bearing one and it is pre-registered as falsifiable** — see control G2 below.
- *vs F94*: this is not a change of k and not a re-weighting of the retrieved list; k stays 20 and
  the rank weights are untouched. It is a per-item switch between two complete decision rules.

**Transplant sketch.** (i) New emitter `mechnov_gate_emit.py` imports the frozen arms module and
dumps, per fold and item: gold, deployed pred, `mlp_max` pred, `mlp_mean3` pred, and the gate
features. (ii) Gate features, all test-time-computable, none using labels: max verifier score;
top-3 mean; **gap between best pos-class and best neg-class verifier score**; Spearman ρ between
the cosine ordering and the verifier ordering over the shortlist; rank in the verifier ordering of
the cosine top-1; verifier score dispersion. (iii) Gate = logistic + shallow GBM, fit **nested**
(inner folds only) so it never sees the fold it scores. (iv) Operating point chosen on inner folds,
never on the evaluated fold.

**Frozen kill bar (declare before any real-data number).**
- **K-VGA-1 (primary):** nested-CV net ≥ **+0.030 acc on ≥2 of 3 datasets**, fold-sign ≥4/5 on
  those datasets. Miss ⇒ KILL, axis closed.
- **K-VGA-2 (permutation null, mandatory):** the gate must beat a label-shuffled null at the same
  fitting budget. With N = 89-111 gated items, this is the dominant overfitting risk and the null
  is not optional.
- **K-VGA-3 (new-signal control, mandatory):** an arm whose gate uses **F47 features only**
  (vote margin, purity, sub-votes). If it matches the verifier-feature gate, the "new information
  source" argument that unlocks F47 is **refuted** and the direction dies *regardless of net*.
- **K-VGA-4 (class balance):** positive rate within 0.10 of the bank rate, per F95 control 4 —
  the logistic arm collapsed to 0.0237-0.0604 there and its nulls were uninterpretable.

**Honest kill risk — high.** Four named ways this dies. (1) **n is tiny**: fitting a gate on 89-111
items with ~6 features is exactly the regime where CV optimism eats a +0.10-0.25 precision lift;
K-VGA-2 exists because I expect this to be the killer. (2) **F47's precedent is bad**: three
supervision sources, zero signal, and the escape clause I am invoking is an argument, not a
measurement. (3) **Fix/break may be intrinsically unpredictable** — if whether adjudication helps
is governed by the *bank item's* label noise rather than the query's relational profile, no
query-side feature can see it, and ERRPAT's "errors are confident neighbourhood inversions" reading
weakly suggests this. (4) **F66's selection lock**: even a clean train-LOO pass in raw space must
then survive head space and test, which is the step that killed F47/F89. My estimate:
P(clear K-VGA-1 on ≥2 datasets) ≈ **8-12 %**; P(survive to a promoted test number) ≈ **5 %**.
It is ranked first not because it is likely but because it is the only live route with an
**above-bar oracle on 3/3 datasets** at **zero GPU**, and because it produces a decisive result
either way in one afternoon of CPU.

---

### C2 — VNQ: Verifier Neighbourhood-Quality as a selective-prediction signal  ★ rank 2
**CP:** ii, in its pure detector form. **Shares C1's pregate entirely** — same emitter, same
features, different read-out. Run them as one job.

**Verified anchors.** *Efficient Nearest Neighbor based Uncertainty Estimation for NLP Tasks*
(kNN-UE), Wataru Hashimoto, Hidetaka Kamigaito, Taro Watanabe, **Findings of NAACL 2025**,
arXiv:2407.02138 — fetched; kNN-UE uses exactly **distances from neighbours + the ratio of labels
among neighbours**, i.e. it *is* the F47 feature family, which makes it the correct pre-registered
baseline to beat. *Overcoming Common Flaws in the Evaluation of Selective Classification Systems*,
Traub, Bungert, Lüth, Baumgartner, Maier-Hein, Maier-Hein, Jaeger, arXiv:2407.01032 — fetched;
introduces **AUGRC** (area under the generalised risk-coverage curve), interpretable as the average
risk of undetected failures, and shows metric choice flips rankings on 5 of 6 datasets. Third,
fetched: *Can LLM Rerankers Predict Their Own Ranking Performance?*, Ni, Bi, Guo, Wu, Han, Cheng,
arXiv:2606.03535 — establishes that a ranker's **own score profile** predicts its per-query quality
with good calibration, which is the general form of the claim here.

**Mechanism.** The verifier's within-query score profile is read as a *neighbourhood-quality
estimate*: how cleanly does a learned relation model separate this query's shortlist into
same-class and different-class? That scalar orders items by predicted-error risk, and the system
abstains (or defers to human review) above a threshold. Data flow: `S[q, ·] → profile statistics →
risk score → risk-coverage curve`.

**Distinctness.** Selective prediction / abstention is a **genuinely untouched axis** in this
campaign — I grepped the full state file: `abstain`=0, `abstention`=0, `selective predict`=0,
`risk-coverage`=0 hits, and `findings.jsonl` has none either. It is also not an accuracy claim,
so it does **not** fight the exchange-rate law at all — the wall that kills C1 does not exist here.
Against kNN-UE, the distinctness is precisely that kNN-UE's inputs are distance + label ratio
(dead as a routing signal in F47) whereas ours is a trained relational profile.

**Frozen kill bar.** **K-VNQ-1:** AUGRC improvement over the kNN-UE baseline (distance + neighbour
label ratio) on **≥2 of 3 datasets**, fold-sign ≥4/5. **K-VNQ-2:** must also beat the plain vote
margin. Both are computed from the same per-item table as C1, so marginal cost is minutes.

**Honest kill risk — moderate, but the *claim value* is the risk.** Technically this probably
passes: the verifier's within-query AUC advantage is +0.16 to +0.23 with 5/5 fold signs on all
three datasets, and it would be surprising if none of that showed up in a risk ordering.
The real risk is that **it does not answer the user's goal** — it is a capability/deployment claim,
not the +3-on-2-datasets performance conjunct, and it must never be dressed as one. Second risk,
and it should be stated in the paper rather than hidden: *Unequal Uncertainty*
(Sargeant, Jorgensen, Shah, Goring, Weller, Bhatt, arXiv:2508.07872, 11 Aug 2025, rev. 7 Jul 2026)
shows uncertainty-based abstention **exacerbates disparities** because under-represented groups
disproportionately receive uncertain predictions, and recommends **"selective friction"** — showing
the prediction together with a salient uncertainty warning — over withholding it. For a *hate-video*
system this is not a footnote; it is the correct design and it routes straight into C3.

---

### C3 — VEA: Verified-Evidence Audit  ★ rank 3
**CP:** iii. This is the mechanism the auditability pillar currently lacks.

**Verified anchors.** *Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking*,
Mubashara Akhtar, Michael Schlichtkrull, Andreas Vlachos, **TACL** (arXiv:2411.05375, Nov 2024,
rev. Jul 2025) — fetched; proposes evidence evaluation combining **reference alignment** with
**verdict-level proxy scoring**, explicitly moving past binary relevance and exact-match, with
stronger human correlation and adversarial robustness than reference-based, proxy-reference and
reference-less baselines. *Attribution, Citation, and Quotation: A Survey of Evidence-based Text
Generation with LLMs*, Tobias Schreieder, Tim Schopf, Michael Färber, arXiv:2508.15396,
**accepted at ACL 2026** — fetched; 134 papers, 300 metrics across seven dimensions, unified
taxonomy. (Caveat: the abstract does not confirm whether quality-*ranking* is separated from binary
attribution in the taxonomy; the full PDF must be read before the survey is cited for that specific
point.)

**Mechanism.** Pillar 4 currently offers a capability list (zero-retrain swap, O(1) recalibration,
editable entries); VEA replaces the list with a *measured* mechanism — the pair verifier re-ranks
the evidence the system cites, and we show that verifier-ranked evidence surfaces the correct
analogue at rank 1 far more often **precisely on the items the system gets wrong**. Data flow:
`decision → cited top-m neighbours → verifier re-rank → audit view (evidence + quality score)`.
The headline numbers already exist in F95 and need no new measurement: the median rank of the first
same-class analogue is **1.0 over all items but 2.0-3.0 over the deployed vote's errors**, and
**72-92 %** of all deployed errors are in the pathology population (HateMM 88/116, ZH 79/88,
EN 109/121) — so on the error set, cosine-ordered evidence shows the human the *wrong* analogue
first, and the verifier does not.

**Distinctness.** F95's ban scope names this as the one thing that **remains legal and unmeasured**:
"the pair verifier as an EVIDENCE RANKER for the auditability pillar — relation scoring and
evidence presentation only, NEVER an accuracy claim." Against the literature, the entire
attribution/citation line (Ev2R, the ACL 2026 survey, CiteGuard, "Correctness is not Faithfulness")
evaluates **evidence for LLM-generated text**; nobody in that line is ranking **labelled memory
entries cited by a classifier**. That gap is real and it is where our contribution sits.

**Frozen kill bar.** **K-VEA-1:** verifier-ranked evidence must beat cosine-ranked evidence on
*first-correct-analogue rank* restricted to the deployed-error set, on ≥2 of 3 datasets, fold-sign
≥4/5. Note this is close to already-measured, so the bar should be set on a *fresh* read-out
(e.g. precision@1 and MRR of the correct-class analogue on the error set) rather than on the
median-rank statistic F95 already reported, to avoid re-scoring a known number as a new result.

**Honest kill risk — low technically, moderate rhetorically.** It will almost certainly measure
positive. The risks are (1) a reviewer reading it as a dressed-up accuracy claim — the record must
carry F95's "NEVER an accuracy claim" binding verbatim; (2) it is a *measurement*, not a novel
mechanism, so on its own it is a strong analysis section and not a paper; (3) it needs a
human-facing evaluation to be fully convincing, and we have no annotator budget. Composed with C2
under the "selective friction" framing (flag the item, show the verifier-ranked evidence, let the
human decide) it becomes a coherent deployable mechanism rather than two loose observations —
**that composition is the recommended paper framing regardless of whether C1 lives.**

---

### C4 — VSW: Verifier soft re-weighting of the vote, λ-interpolated  ★ rank 4 (epistemic value only)
**CP:** i.

**Verified anchor.** *Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order
Information*, Rui Ai, Yuqi Pan, David Simchi-Levi, Milind Tambe, Haifeng Xu, arXiv:2510.01499
(1 Oct 2025, rev. 19 May 2026), **ICML 2026** — fetched; introduces **Optimal Weight (OW)** and
**Inverse Surprising Popularity (ISP)**, which use second-order information (cross-voter correlation
and heterogeneity) and **provably** mitigate majority voting's limitations under mild assumptions.
Related graph-side anchor, fetched: *Don't Forget to Connect! Improving RAG with Graph-based
Reranking* (G-RAG), Jialin Dong, Bahare Fatemi, Bryan Perozzi, Lin F. Yang, Anton Tsitsulin,
arXiv:2405.18414 (28 May 2024) — a GNN reranker over an AMR-derived document graph that beats
PaLM 2 as a reranker at a smaller footprint.

**Mechanism.** Keep the deployed rank-weighted sum over the top-20 exactly as it is, and multiply
each neighbour's rank weight by a monotone function of its verifier score, with an interpolation
coefficient λ such that **λ=0 reproduces the deployed vote bit-exactly**. The deployed vote is
first-order (labels × fixed rank weights); the verifier supplies second-order information (how
genuinely each retrieved item *relates* to this query), which is structurally the OW/ISP move.
Data flow: `S[q, ·] → per-neighbour multiplier → λ-blend with rank weights → same sum → label`.

**Distinctness — and where it is weakest.** It does not change the aggregation shape, so the
−0.029/−0.044 control-2b cost is zero at λ=0 and grows continuously. It is not a change of k, so
F94's measured content does not touch it. **But F94's ban text says in terms: "no truncation *or
re-weighting* of the retrieved list reaches it."** That clause was generalised from F49/F66/F86,
not measured for *learned* weights — so this candidate does not violate a measurement, but it does
contradict a stated generalisation, and that must be flagged to the reviewer up front rather than
finessed. F63 is a second charge: diffusion over the frozen graph is monotone-negative in α, and
while F63 explicitly does not price learned edge weights, a one-hop verifier-weighted vote is close
enough that the burden of proof is ours.

**Frozen kill bar.** **K-VSW-1:** net ≥ +0.030 on ≥2 of 3 datasets at a λ selected on inner folds.
Given the exchange-rate law I expect this to fail. **The reason to run it anyway is K-VSW-2, a
diagnostic that cannot fail:** sweep λ from 0 to 1 and record the exchange rate as a function of
aggregation sharpness. F95 measured the exchange rate at exactly **two** points (max and mean-top-3).
A full curve either finds a sharpness regime where the rate exceeds 1.2 — which no cell in a 36-cell
battery reached — or it shows the rate is bounded below 1 across the entire continuum, which
**closes the aggregation axis arithmetically** and is a materially stronger law-I datum than the
two-point read F95 currently carries into the paper.

**Honest kill risk — near-certain death as a performance bet.** P(clear K-VSW-1 on ≥2 datasets)
≈ **2 %**. Marginal cost over C1's pregate is ~1 hour of CPU because the emitter and the folds are
shared. Recommendation: run as a rider on C1, budget it as analysis, never as a lever.

---

### C5 — Spectral / structured aggregation over the pair-score matrix  ★ NO-GO, recorded for closure
**CP:** i. Swept because tasked; recommending against, with reasons.

The family (Rank Centrality, Negahban–Oh–Shah, arXiv:1209.1688; Spectral MLE, Chen & Suh,
arXiv:1504.07218; energy/CRF-style joint assignment; GCN-LPA-style learned edge weights,
arXiv:2002.06755) is mathematically attractive and mostly **pre-closed here** for three independent
reasons, any one sufficient:

1. **Spectral ranking returns an item-level score.** Rank Centrality's output is a stationary
   distribution over items — structurally a *main effect*, exactly the 62-73 % item-level hubness
   offset component that F95 §4.1 measured and that F89's CSLS arm (T2a) already found **inert** on
   this system. The verifier has *already* inverted the query/interaction split to 77-93 %
   interaction; a spectral pass would push back toward the component we know does not decide.
2. **Joint/structured assignment over test items is transductive**, and F63's ban scope rules
   transductive test-graph variants "out-of-box regardless".
3. **Prototype construction from verified relations** is pre-closed twice over: W2-E prototype
   memory is dead and the W2-E prototype-select ban stands; memory-bank curation is parked at recon
   (F78, P(+3) ≈ 1 %) with a door-closer the user has not invoked.

The only fragment worth keeping is the OW/ISP *second-order weighting* idea, which is already
carried by C4 in the one form that pays zero shape cost.

---

## §3. NOVELTY CHECK (tasked duty)

**Question:** does "learned pair-verification over retrieval memory for content moderation" — the
F95 machinery — exist 2024-2026, and is it claimable as novel-in-field?

**Method:** ~14 targeted searches across the RelationNet lineage, retrieval-augmented moderation,
RAG reranking/verification, and hateful-meme/video detection; every load-bearing paper fetched.

**Answer, in three parts.**

1. **The scorer is not novel and must not be claimed.** The relation module on
   `[ |z−z'| , z⊙z' ]` is textbook: *Learning to Compare: Relation Network for Few-Shot Learning*,
   Sung, Yang, Zhang, Xiang, Torr, Hospedales, **CVPR 2018**, arXiv:1711.06025 (fetched — confirmed
   title/authors/venue; the abstract does not state the concatenation form, so cite it for the
   *learned relation scorer* concept only, not for our specific feature map). Any claim phrased as
   "we propose a pairwise verifier" is dead on arrival.

2. **The application to *this* setting appears genuinely unoccupied.** Our own lineage —
   RGCL (ACL 2024) and RA-HMD (EMNLP 2025, arXiv:2502.13061) — scores retrieval by
   cosine/contrastive similarity with a kNN classifier; I found no learned pairwise verifier over
   retrieved memory entries in either. The nearest 2025-2026 neighbours each miss on a different
   axis: **CPE** (*Classification is a RAG problem: A case study on hate speech detection*,
   Willats, Pennington, Mohan, Vidgen, arXiv:2508.06204, 8 Aug 2025 — fetched) reframes moderation
   as retrieval over *policy text* with an agentic LLM, no learned relation scorer over a labelled
   memory; **G-RAG** (arXiv:2405.18414) learns relational structure over retrieved documents but
   for RAG passage reranking with AMR edges, not moderation and not a labelled classification
   memory; **CAR** (arXiv:2605.04495) and **When Reranking Hurts** (arXiv:2606.31087) gate or
   reweight retrieval by *generator/model confidence*, not by a trained pair relation.
   **Caveat, stated plainly:** this is absence of evidence across ~14 searches, not a systematic
   review, and it should be re-run with the `novelty-check` skill before any submission.

3. **What is actually claimable is the *result*, not the machinery.** The defensible novel-in-field
   contribution is the **measurement**: in a retrieval-memory moderation classifier, a trained
   relation scorer beats cosine by +0.13-0.27 pooled pair-AUC and +0.16-0.23 within-query AUC
   (18/18 cells, 5/5 fold signs), inverts the score-variance decomposition from 27-38 % to 77-93 %
   query×bank interaction, reaches 36.7-54.6 % of the diagnosed error population — **and the
   deployed decision does not improve**, because the exchange rate never exceeds 1.17. That is a
   negative result measured on *both sides* of the retrieval chain with the selection lock in
   between, and it is the strongest law-I datum the campaign has. It is publishable as analysis
   in a way "we built a verifier" is not.

---

## §4. PRE-CLOSED — DO NOT RE-PROPOSE (recorded so the next sweep does not re-derive these)

| proposal | closed by |
|---|---|
| head-space pair verification as a rescue | F95 ban (a) |
| bilinear / cross-attention / siamese-with-margin pair scorers | F95 ban (b) — scorer quality is not the binding term |
| verifier-as-reranker inside the vote, unpriced | F95 ban (c) — C1 discharges this, nothing else does |
| any change to k, either direction | F94 |
| CSLS / hubness correction, LW whitening, class-balanced quota, 1-D excision | F89, 0/5 promotable |
| multi-hop label propagation over the frozen cosine graph | F63 |
| transductive test-graph inference | F63 ban scope, out-of-box |
| per-item selectors over frozen channels on cosine-derived features | F47 |
| prototype memory / prototype-select | W2-E, dead |
| memory-bank curation | F78, parked at recon, needs user door-closer |
| MLLM scores as training signal; OCR; cross-dataset training; ensembles; closed APIs | standing bans |

---

## §5. RECOMMENDATION

**Run C1+C2+C4 as a single $0 CPU pregate** (~1 afternoon, ≤8 threads, zero GPU/SLURM/Modal, train
split only, test untouched). They share one emitter and one set of folds; the marginal cost of C2
and C4 over C1 is minutes and hours respectively. Freeze all bars — K-VGA-1/2/3/4, K-VNQ-1/2,
K-VSW-1/2 — in a prereg **before** the emitter is pointed at real data, per standing ceremony.

**Write C3 regardless of the outcome.** It is the one route F95 explicitly declared legal and
unmeasured, its headline numbers already exist, and composed with C2 under *selective friction*
(arXiv:2508.07872) it gives pillar 4 a mechanism instead of a capability list.

**Expected outcome, stated honestly so nobody is surprised:** C1 most likely dies at K-VGA-2
(permutation null) or K-VGA-3 (the F47-feature control matching the verifier-feature gate), and if
it survives those it faces F66's selection lock on the way to test. The realistic value of this
sweep is (a) one genuine ~8-12 % shot with an above-bar oracle on 3/3 datasets at zero GPU, (b) a
closed aggregation axis with a full exchange-rate curve instead of two points, and (c) a mechanism
for the audit pillar. If C1 fails, the relational asset is settled as **analysis-grade only**, and
the campaign should stop trying to convert it.

---

## §6. SOURCES

Every paper used to justify a candidate in §2 C1-C4 and the novelty check in §3 was **fetched and
its title/authors/venue/date confirmed against the arXiv record**. The three C5 entries are listed
only to document *why that family is closed*; they were surfaced by search but **not fetched**, and
must be verified before any of them is cited in writing.

- Dabod, Cohen, Stanovsky. *When Reranking Hurts: Uncertainty-Based Gating for Few-Shot Reranking*. arXiv:2606.31087, 30 Jun 2026. https://arxiv.org/abs/2606.31087
- Song, Zhou, Kong, Jiao, Ye, Gao, Shi, Zhou, Qi. *CAR: Query-Guided Confidence-Aware Reranking for RAG*. arXiv:2605.04495, 6 May 2026. https://arxiv.org/abs/2605.04495
- Hashimoto, Kamigaito, Watanabe. *Efficient Nearest Neighbor based Uncertainty Estimation for NLP Tasks*. Findings of NAACL 2025, arXiv:2407.02138. https://arxiv.org/abs/2407.02138
- Traub, Bungert, Lüth, Baumgartner, Maier-Hein, Maier-Hein, Jaeger. *Overcoming Common Flaws in the Evaluation of Selective Classification Systems* (AUGRC). arXiv:2407.01032. https://arxiv.org/abs/2407.01032
- Ni, Bi, Guo, Wu, Han, Cheng. *Can LLM Rerankers Predict Their Own Ranking Performance?* arXiv:2606.03535, 2 Jun 2026. https://arxiv.org/abs/2606.03535
- Akhtar, Schlichtkrull, Vlachos. *Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking*. TACL, arXiv:2411.05375. https://arxiv.org/abs/2411.05375
- Schreieder, Schopf, Färber. *Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with LLMs*. ACL 2026, arXiv:2508.15396. https://arxiv.org/abs/2508.15396 — *taxonomy detail unverified, abstract only*
- Ai, Pan, Simchi-Levi, Tambe, Xu. *Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information*. ICML 2026, arXiv:2510.01499. https://arxiv.org/abs/2510.01499
- Dong, Fatemi, Perozzi, Yang, Tsitsulin. *Don't Forget to Connect! Improving RAG with Graph-based Reranking* (G-RAG). arXiv:2405.18414, 28 May 2024. https://arxiv.org/abs/2405.18414
- Sargeant, Jorgensen, Shah, Goring, Weller, Bhatt. *Unequal Uncertainty: Rethinking Algorithmic Interventions for Mitigating Discrimination from AI*. arXiv:2508.07872, 11 Aug 2025. https://arxiv.org/abs/2508.07872
- Willats, Pennington, Mohan, Vidgen. *Classification is a RAG problem: A case study on hate speech detection*. arXiv:2508.06204, 8 Aug 2025. https://arxiv.org/abs/2508.06204
- Sung, Yang, Zhang, Xiang, Torr, Hospedales. *Learning to Compare: Relation Network for Few-Shot Learning*. CVPR 2018, arXiv:1711.06025. https://arxiv.org/abs/1711.06025
- Negahban, Oh, Shah. *Rank Centrality: Ranking from Pairwise Comparisons*. arXiv:1209.1688 / Operations Research 65(1). https://arxiv.org/abs/1209.1688 — *NOT FETCHED; listed for §2 C5 closure only*
- Chen, Suh. *Spectral MLE: Top-K Rank Aggregation from Pairwise Comparisons*. ICML 2015, arXiv:1504.07218. https://arxiv.org/abs/1504.07218 — *NOT FETCHED; listed for §2 C5 closure only*
- Wang, Leskovec. *Unifying Graph Convolutional Neural Networks and Label Propagation* (GCN-LPA). arXiv:2002.06755. https://arxiv.org/abs/2002.06755 — *NOT FETCHED; listed for §2 C5 closure only*
