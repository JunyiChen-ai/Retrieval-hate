# B2 PCD — Policy-Cone Discriminant Head: paper-and-pencil specification, novelty check, and verdict

**Date:** 2026-08-10 · **Status:** **DEAD — killed at the specification stage, before any pilot.**
**Two independent causes, either sufficient:** (1) the core construction is occupied by published
work; (2) the mechanism's central premise is empirically false at the encoder level, measured here
on **train/val only**.

**Test-set contact: none.** Every number in this document comes from `train` and `dev_seen`
features. The probe scripts assert on the split name before any `torch.load`.

Mandate: `idea-stage/IDEA_REPORT.md` §8.5 — the jury declined a pilot slot for B2 and ruled that
*"the next legitimate action for B2 is a paper-and-pencil specification plus novelty check, not a
pilot,"* because the candidate needed a mathematical decision about how a cone of policy-clause
directions attaches to the deployed nonlinear Hadamard-fusion head. §1 below answers that question;
§2–§4 specify the construction that answers it; §5 is the novelty check; §6 is the feasibility
screen; §7 is the verdict.

---

## §1 — The mathematical question the jury named, and its answer

### 1.1 What the deployed head actually is

`idea-stage/r4_harness.py::Head` mirrors `src/model/classifier.py::classifier_hateClipper` with the
deployed ablation hyper-parameters. For a cached item with frozen image feature `a` and frozen text
feature `b`:

```
i = L2norm(W_i a)              W_i : R^{Da -> 1024}
t = L2norm(W_t b)              W_t : R^{Db -> 1024}
x = i ⊙ t                      (align fusion, Hadamard)
z = MLP_3(x)                   3 x [Linear(1024,1024), ReLU, Dropout]
s = w^T z + c                  logit
```

`W_i`, `W_t`, `MLP_3`, `w` are all **trained**. `x`, `z` therefore live in **learned** spaces with
no text-encoder semantics: a policy sentence has no image in them, and any map from sentence space
into them would itself have to be learned — which destroys the one property that makes a policy
clause a policy clause, namely that its direction is **fixed before training and not fitted to the
labels**.

### 1.2 Consequence — the cone cannot live inside the head

This is the decisive structural fact and it forecloses the "obvious" attachment points:

- **Not in `x` or `z`.** Learned spaces; see above.
- **Not by constraining `W_t`.** One could freeze `K` rows of `W_t` to clause directions
  (construction **PCD-C**, §3.3). But then the clause scores are immediately re-mixed by the
  3-layer MLP, the non-negativity/disjunction structure survives nowhere, and no exemption term is
  expressible. It is "structured first-layer initialisation", which is neither the candidate's
  hypothesis nor defensible as a mechanism.
- **The cone must live in the frozen encoder's own joint space**, the only space where a written
  policy clause is literally embeddable, and it must compose with the head **at the logit** — the
  only layer at which two scores computed in different spaces can be added without inventing a
  learned bridge. Log-odds addition is also the semantically correct composition: "this item
  violates clause *k* unless exempted" is naturally an additive evidence term in log-odds.

**Answer to the jury's question:** the cone attaches as an **additive policy logit** in the frozen
CLIP joint space (construction **PCD-B**, §3.1), not inside the fusion head.

### 1.3 Which frozen space is available — a hard asset constraint

| encoder | what the cache stores | can a standalone policy clause be embedded in the same space? |
|---|---|---|
| **CLIP-L/14-336** | `img_feats` = frame-mean `CLIPVisionModel.pooler_output` (1024-d, **pre-projection**); `text_feats` = 77-token-chunk-mean `CLIPTextModel.pooler_output` (768-d, pre-projection) | **Yes.** Applying the frozen `visual_projection` (1024→768) and `text_projection` (768→768) puts both channels in the 768-d CLIP joint space; a clause is embedded by the identical text path. Chunk-mean commutes with the linear projection, so this is exact. |
| **Qwen2.5-VL-7B** (and the LoRA variants) | `img_feats` = mean over all prompt tokens except the assistant tail; `text_feats` = mean over the **last 4 tokens of the assistant generation-prompt tail**, inside a multimodal prompt that already contains 8 frames + title + transcript | **No.** This is a decoder read-out position, not a sentence-embedding space. A standalone clause has no frames and no read-out position; there is no encoder call that places a clause in this space. Building one would require a new extraction pass *and* a decision about what frames to pair a clause with — i.e. a new asset, not a pilot. |

`src/utils/generate_VideoCLIP_embedding_HF.py` (lines 221–275) and
`src/utils/generate_VideoMLLM_embedding_HF.py` (lines 282–360) are the sources for the table.

**Immediate consequence for the protocol.** Round 4's val-best cells are HateMM/LoRA-Qwen,
MHC-EN/Qwen, MHC-ZH/LoRA-Qwen, ImpliHateVid/CLIP (§8.8). **Three of the four cannot host the cone
at all.** The CLIP cache does exist for all three splits of all four datasets, so a clean
same-frame CLIP-cell pilot is buildable; but the mandate's "each dataset at its val-best encoder"
is unsatisfiable for this mechanism by construction. This was recorded before the feasibility
screen was run and is not a post-hoc excuse.

---

## §2 — Clause set (frozen before any measurement)

`K = 6` matched pairs for the screen (the candidate's prereg said 12; the screen is
dimension-matched against its own controls, so `K` does not bias the comparison — see §6.3).
Each pair is (violation clause, matched **safe-use / exemption** clause) describing the *same
surface content* in a non-violating use. The exemption structure is not invented here: it is
written into the deployed policies themselves (YouTube's EDSA — Educational, Documentary,
Scientific, Artistic — exception; Meta's condemnation / awareness-raising / self-referential /
quotation carve-outs).

| k | violation clause | matched exemption clause |
|---|---|---|
| 1 | attacks a protected group with a slur | quotes a slur in order to condemn it |
| 2 | dehumanises an ethnic group as animals or vermin | gives a neutral factual description of an ethnic group |
| 3 | calls for violence against a religious group | condemns violence against a religious group |
| 4 | mocks people for their disability | documents discrimination against disabled people |
| 5 | praises a racist ideology | reports on a racist ideology as news |
| 6 | says women are inferior and should be controlled | criticises sexism |

Each is wrapped in the fixed template `"a video that {clause}"`. One template, no ensembling —
prompt ensembling is itself occupied (§5) and would confound the mechanism with a known baseline.
Chinese clauses (`pcd_space_probe4.py::ZH_PAIRS`) are literal translations, written once, not tuned.

---

## §3 — Candidate constructions

Notation: `u_v(x) = L2norm(W_V a(x))`, `u_t(x) = L2norm(W_T b(x))` are the item's two channels in
the 768-d CLIP joint space; `v̂_k`, `ê_k` are the L2-normalised clause embeddings; per-channel
whitening uses train-only `μ`, shrinkage covariance `Σ_λ = (1-λ)Σ + λ·tr(Σ)/d·I`, `λ = 0.1` fixed.

### 3.1 PCD-B — additive policy logit (**selected main construction**)

Per channel `c ∈ {v, t}` and pair `k`:

```
r_ck(x) = <ũ_c(x),  â_ck>          violation projection   (â  = whitened, normalised v̂_k)
m_ck(x) = <ũ_c(x),  b̂_ck>          exemption projection   (b̂  = whitened, normalised ê_k)
z_ck(x) = relu( r_ck(x) - β_ck·m_ck(x) - τ_ck )        β_ck = softplus(·) ≥ 0
g(x)    = Σ_{c,k} α_ck · z_ck(x)                      α_ck = softplus(·) ≥ 0
s_total(x) = f_θ(a(x), b(x)) + g_φ(x)
```

`f_θ` is the **untouched** deployed head; the two are trained jointly under the same BCE objective,
schedule, and validation epoch selection.

- **Semantics.** `relu(r − βm − τ)` reads "clause *k* fires unless its matched exemption fires";
  `Σ_k α_k z_k` (or `logsumexp_k`) is the disjunction "hate iff **some** clause fires unexempted".
  This is the natural logical form of a written policy and is *not* what prompt ensembling computes
  (a mean of clause embeddings is an AND-flavoured centroid).
- **Trainable parameters of the cone:** `α, β, τ ∈ R^{2K}` each, plus scale and bias
  → `6K + 2 = 38` at `K = 6` (74 at `K = 12`), against ~5.3 M in the head. Negligible.

### 3.2 PCD-A — clause activations as an extra input block (rejected as main)

Feed the `2K` activations into `Head(..., extra_dim=2K)` — natively supported by the harness. The
3-layer MLP then re-mixes them freely, so the non-negative, disjunctive, exemption-gated structure
that *is* the mechanism does not survive to the logit. The ablations would then test "extra CLIP
prompt-similarity features", not the cone. Retained only as a declared comparator, because it is
what a reviewer will say the method really is.

### 3.3 PCD-C — clause directions as frozen rows of `W_t` (rejected)

See §1.2. Structured initialisation; no exemption term expressible.

### 3.4 Required degeneracies (all clean, all pre-specified)

| ablation | how | what it isolates |
|---|---|---|
| **no-exemption** | `β ≡ 0` → `z = relu(r − τ)` | the paired exemption/veto term |
| **random directions** | replace `{â_k, b̂_k}` by `2K` i.i.d. unit vectors, everything else identical, **identical parameter count** | whether *clause semantics* contribute anything |
| **single anchor** | `K = 1`, generic hate/benign prompt pair | whether a heterogeneous policy needs more than one centroid |
| **mean aggregation** | `α_k ≡ 1/K`, `β = τ = 0` | reduces to prompt ensembling |
| **free directions** | same `2K`-unit ReLU layer, directions *learned* (upper bound, not parameter-matched) | whether a tiny extra layer explains any gain |

### 3.5 Parameter alignment with the bare head

Strict alignment is via the **random-direction arm**: byte-identical architecture, byte-identical
parameter count, only the frozen directions differ. That is the fair comparison and it is also the
pre-declared KILL test (task mandate: if the gain survives randomisation, the clause semantics
contribute nothing).

---

## §4 — What the pilot would have been (specified, then not run)

Primary metric test ROC-AUC, secondary macro-F1 at a validation-selected threshold; four datasets
on their **CLIP** cell (the only cell that can host the cone); ≥3 seeds; single submission; frozen
comparator selected on mean validation ROC before any test metric is read (R4-1/R4-2 practice);
uncertainty by the paired stratified joint-row bootstrap the D1 ruling substituted for permutation
nulls (`idea-stage/R4_DEVIATION_D1_RULING.md`), **not** a permutation null; synthetic planted-signal
and no-signal smokes before submission.

`idea-stage/PCD_FREEZE.md` was **not written and no pilot was run**, because §5 and §6 both return
DEAD and the project's rule is that a candidate whose core construction is occupied does not
proceed to a pilot.

---

## §5 — Novelty check

Method: three independent cross-model literature sweeps over the six mandated axes (CLIP zero-shot
multi-prompt / prompt ensembling; negative-prompt and difference-of-prompt directions;
policy-as-prompts moderation; concept bottlenecks; polyhedral/cone classifiers; rule-based and
clause-based hate-speech heads), targeting NeurIPS/ICML/ICLR/CVPR/ICCV/ECCV/ACL/EMNLP/NAACL/AAAI/
ACM MM/WWW. Items marked `[C]` were seen as a title or snippet only and were **not** opened; no
verdict below rests on a `[C]` item alone.

### 5.1 Occupancy map, component by component

| PCD component | strongest occupant | what it takes |
|---|---|---|
| **projections of frozen VLM features onto frozen text-clause directions → trained head** | **LaBo**, Yang et al., **CVPR 2023** (2211.11158) | The whole outer scaffolding, *plus* non-negativity: LaBo's head is `g(x,E_C)·σ(W)^T` with softmax along the concept axis, so its per-class weight is literally a **non-negative combination of frozen text-clause directions**. It also fixes the clause set in advance and ships the **random-concept ablation**. |
| same, general formulation | **Post-hoc CBM**, ICLR 2023 (2205.15480); **Label-Free CBM**, ICLR 2023 | "score = f(projections of a frozen backbone onto a fixed set of text/concept directions) + sparse head" as a named family. PCBM's bottleneck is literally `⟨f, c_i⟩` over a bank of CLIP text-embedding concept vectors. |
| **non-negativity over text directions, with the antonym question already settled** | **SpLiCE**, NeurIPS 2024 (2402.10376) `[C]` | Strictly non-negative sparse decomposition of CLIP embeddings over a text concept dictionary — **and it adds antonyms as separate dictionary atoms** rather than using signed/difference directions. The design choice PCD treats as its contribution is already a settled, published alternative. |
| **direction = enc(positive clause) − enc(negative clause)** | **TCAV**, ICML 2018 (concept vector = separating direction vs. a contrast set); standard CLIP-CBM concept-presence scoring; **Chuang et al. 2023** (2302.00070) hand-written prompt pairs → difference directions → projected *out* of the classifier | The paired-difference direction is close to folklore in CLIP concept scoring; it is also the diff-in-means "refusal/harmfulness direction" line (Arditi et al., NeurIPS 2024). |
| **paired difference + explicit suppression, inside a classifier** | **RoboShot**, Adila et al., **ICLR 2024** | Paired LLM-authored contrasts → difference direction → **vector rejection (suppression) + addition** over K concepts, for zero-shot classification. Has pairs, differences, and a suppression operator together. Lacks only the trained head. |
| **positive/negative prompt pair on frozen CLIP, trained, for content safety** | **Q16**, Schramowski et al., 2022 (2202.06675) | PCD's `K = 1` degenerate case, end-to-end, in the safety domain. |
| **learned positive+negative prompt pair per class** | **DualCoOp**, NeurIPS 2022 / TPAMI 2024; **NegPrompt**, CVPR 2024 | Negative prompts as a joint/vetoing term. |
| **cone / polyhedral decision geometry** | **Polyhedral Conic Classifiers**, Cevikalp & Triggs, **CVPR 2017** / **TPAMI 2020**; **Convex Polytope Machine**, NIPS 2014 | The name *and* the geometry *and* the motivation (asymmetric binary problem, positive class tightly circumscribed). |
| **"a safety concept is a cone of directions, not one direction"** | **The Geometry of Refusal in LLMs: Concept Cones and Representational Independence**, Wollschläger et al., **ICML 2025** (2502.17420) | The exact framing sentence, already published, in safety. |
| **hand-written moderation rules encoded as vectors, for hate classification** | **Rule By Example**, Clarke et al., **ACL 2023** | Rule embeddings aligned with matching text; rule-grounded hate prediction; beats prior SOTA on three hate datasets. This is the domain-side occupant. |
| **multiple clauses with an explicit counter-speech / quotation veto** | **Hypothesis Engineering for Zero-Shot Hate Speech Detection**, Goldzycher & Schneider, TRAC @ COLING 2022 (2210.00910) | Multiple hand-written hypotheses; quotation identification and stance toward the quote explicitly override the hate hypothesis. The veto structure, in NLI form. |
| **paired allow/forbid policy clauses driving a safety classifier** | **Constitutional Classifiers**, 2501.18837; **Llama Guard** per-category "can / should not"; **Class-RAG** (2410.14881) paired safe/unsafe evidence *to suppress false positives* | The policy-exemption pairing as a deployed design. |
| **policy clauses as the unit of a moderation decision** | **Policy-as-Prompt**, FAccT 2025 (2502.18695); **Classification is a RAG problem / Contextual Policy Engine** (2508.06204) | The framing and the phrase. |
| **the clause list itself** | **HateCheck**, ACL 2021 | Counter-speech, quotation/use-mention, reclaimed slurs, non-hateful group mentions — a hand-written paired hateful/non-hateful contrast taxonomy. §2's table substantially re-derives it. |
| **the random-direction ablation** | **WaffleCLIP**, ICCV 2023 | Published finding that random descriptors frequently *match* semantic ones — so a null there reads as confirming WaffleCLIP, not as PCD's control passing. |
| **written policy → violating *and* matched non-violating clause embeddings in a frozen multimodal space → decide by violating-minus-safe margin** | **Zero-Shot Image Moderation in Google Ads with Domain-Adapted Vision-Language Models**, **WSDM 2025** (2412.16215) | **The closest work on semantics.** Policy is turned into curated **"in-scope" (violating) and "out-of-scope" (non-violating carve-out)** textual descriptions, all embedded in a frozen image–text co-embedding space; an item is flagged when it matches the in-scope pool more than the out-of-scope pool by a margin. This is PCD's paired-exemption arithmetic, in a frozen multimodal moderation system, already deployed. It differs only in being a thresholded match-count over unpaired pools with no trained coefficients. |
| **frozen CLIP + small head whose class weights are initialised from encoded text, for hateful memes** | **MemeCLIP**, **EMNLP 2024** (2409.14703) | The backbone-and-head skeleton inside the hateful-multimodal domain: frozen CLIP ViT-L/14, linear projections, adapters, cosine classifier with *Semantic-Aware Initialisation* from CLIP text embeddings of the label names. |
| **K text-concept anchors + frozen text encoder + tiny trained module, trained on matched safe/unsafe prompt pairs** | **Latent Guard**, **ECCV 2024** (2404.08031) | Concept anchors + a trained mapping over a frozen encoder for safety detection, with matched safe/unsafe pairs as the training signal. |
| **community rule documents → rule vectors → relevance-weighted trained moderation classifiers** | **CRCM**, 2408.12035 | "Written rules become text vectors that weight a trained classifier" for moderation. |
| **paired positive/negative prompt sets whose difference is the decision direction** | **WinCLIP**, CVPR 2023 `[C]`; **DualCoOp**, NeurIPS 2022 (`p = σ(⟨v,t⁺⟩/τ − ⟨v,t⁻⟩/τ)`, monotone in `v·(t⁺−t⁻)`) | The difference-direction score itself, hand-written (WinCLIP) and learned (DualCoOp). |
| **text-anchored few-shot heads over frozen VLM features** | **CLAP**, CVPR 2024 (class-wise Lagrangian proximity to frozen text prototypes); **LP++**, CVPR 2024 (`f^T(w_k + α_k t_k)`) | The "constrain/anchor a trained head to frozen text directions" family, as the proper few-shot baselines PCD would have had to beat. |

### 5.2 Two mathematical corrections the sweep forced

1. **Non-negative weights do not produce a cone-shaped decision region.** With
   `score = Σ_k α_k <d_k, x> − b`, `α_k ≥ 0` and no per-clause nonlinearity, the model is still
   **linear**; the region is a halfspace. All non-negativity buys is `w ∈ cone{d_1..d_K}` — a
   conic-hull-constrained linear probe, i.e. NNLS on a fixed dictionary. The candidate's
   "polyhedral cone rather than a halfspace" language (`section9_round4_candidates_2026-08-10.md`
   §B2) is, as written, **false**.
2. **With the per-clause ReLU, PCD is exactly a one-hidden-layer ReLU network with a frozen first
   layer and a non-negative output layer** — the convex-neural-network form (Bengio et al., NIPS
   2005) over a fixed dictionary (Rahimi & Recht, NIPS 2007), with the random dictionary swapped
   for text-clause differences. Its score is convex in `x`, so the region that is a cone is the
   **negative** one (the polar cone), not the positive one. The `max_k` variant is the Convex
   Polytope Machine (NIPS 2014).

### 5.3 Novelty verdict

**The four-part conjunction — hand-written *paired* policy clauses → difference directions →
non-negative cone score with an explicit exemption-suppression term → over frozen multimodal
features — was not found published as a single unit.** That is the honest upper bound on what
survives. But:

- **Machinery:** **LaBo (CVPR 2023)** occupies two of the three structural components
  simultaneously, in nearly the exact algebraic form — `logit_y = x·(Σ_c W_{y,c} e_c)` with `W`
  made non-negative by a softmax along the concept axis is *literally* a trained head restricted to
  the non-negative span of `K` frozen text directions, with the clause set fixed in advance and a
  random-concept ablation shipped.
- **Semantics:** **Zero-Shot Image Moderation in Google Ads (WSDM 2025, 2412.16215)** occupies the
  paired-exemption arithmetic itself: written policy → violating **and** matched non-violating
  clause embeddings in a frozen image–text co-embedding space → flag on the violating-minus-safe
  margin. In a deployed moderation system.
- **Exemption concept:** **Hypothesis Engineering (TRAC @ COLING 2022)** occupies "hand-written
  safe-use clauses, evaluated by a frozen text model, explicitly override the hate score", on
  exactly PCD's target failure modes (quotation, self-reference, no protected group), with
  +7.9 HateCheck / +10.0 ETHOS.
- **Geometry + suppression in a classifier:** **RoboShot (ICLR 2024)**.
- **Domain instantiation:** **Rule By Example (ACL 2023)** — hand-written moderation rules as
  embeddings, rule-grounded hate prediction, with FP-aware contrastive correction; **MemeCLIP
  (EMNLP 2024)** — frozen CLIP + small head with class weights initialised from encoded text, for
  hateful memes; **Latent Guard (ECCV 2024)** — concept anchors + trained mapping over a frozen
  encoder, trained on matched safe/unsafe pairs.
- The name and geometry are occupied by **Cevikalp & Triggs (CVPR 2017/TPAMI 2020)**; the framing
  sentence "a safety concept is a cone of directions, not one direction" is occupied by
  **Concept Cones (ICML 2025)**, whose basis directions are themselves harmful-minus-harmless
  difference vectors with non-negative coefficients.
- The `K = 1` case is **Q16 (2022)** and, without the nonlinearity, is plain two-prompt CLIP
  zero-shot with a learned scale.

**One nearby work is *not* an occupant, recorded so it is not miscited later.** **HatePrototypes**
(2511.06391), flagged in §8.1c as the nearest-sounding competitor, builds class prototypes as means
of LM hidden states over ~50 **labeled examples** per class. It has no policy text, no clauses, no
pairs and no exemption. It occupies "class vector + frozen LM + cheap head" and nothing else.

The residue is exactly three joints, all of them recombination: (i) clauses paired **one-to-one**
(topic held fixed, only the policy verdict varying) rather than as two unpaired pools; (ii) the pair
collapsed into a **continuous** difference-projection feature rather than a match count (Google
Ads), a logical override (Goldzycher), or an argmax (Q16); (iii) a **trained** head with a learned
suppression coefficient, on multimodal video features. That is a difference in implementation layer
between four published mechanisms.

**The algebraic hazard, stated explicitly.** If the suppression term collapses to the plain
difference `⟨f, v̂_k − ê_k⟩` — i.e. if `β_k` is unconstrained and the ReLU is inactive — then the
whole method is DualCoOp/WinCLIP/Q16/Chuang's difference direction with a trained scale, and axis 1
occupies it completely. The suppression therefore *has* to be non-linear
(`relu(r_k) − β_k·relu(m_k)`, or a multiplicative gate) **and** has to beat a plain-difference
ablation. §6.4 measures that the exemption side is worth ≈ +0.005 with a sign flip, so the
non-linear form has nothing to work with.

**Baselines any surviving version would have owed.** CLAP and LP++ (CVPR 2024, text-anchored
few-shot heads), LaBo and PCBM (conic text-concept heads), MemeCLIP (in-domain frozen-CLIP head
with text-initialised class weights), a Google-Ads-style clause-count decision, a Goldzycher-style
rule override, and a plain frozen-feature linear probe. Note the last one alone reaches val ROC
0.77–0.97 in §6.3 against the cone's 0.53–0.80.

**Verdict: DEAD on novelty.** The residue — "the exemption term is a *learned geometric veto*
rather than a prompt-level, retrieval-level or data-generation-level one" — is a difference in
implementation layer between four published mechanisms, not a mechanism. A reviewer who knows LaBo
and RoboShot will say so, and per `research-wiki/NOVELTY_RECON_2026-08-09.md`'s standing discipline
(re-skinning an occupied mechanism is an automatic kill) this does not clear a main-conference
methods bar. Per the mandate, **stage 3 (pilot) is skipped.**

---

## §6 — Feasibility screen (run anyway, train/val only, zero test contact)

The screen was written before the novelty reports returned, to answer §1.3 (does the CLIP joint
space even work) and to give the kill an empirical leg. It is cheap (CPU/GPU seconds) and it turned
out to be the more decisive of the two causes.

Code: `idea-stage/pcd_space_probe.py`, `pcd_space_probe2.py`, `pcd_space_probe3.py`,
`pcd_space_probe4.py`. Log: `logging/runs/pcd_spec_probe/run.log`. Launcher:
`idea-stage/pcd_probe_launch.sh`.

### 6.1 The joint-space construction works — that part is confirmed

Zero-shot, no training: projecting the cached pre-projection poolers through CLIP's
`visual_projection` / `text_projection` and scoring against a generic hate-minus-benign anchor
gives **val ROC 0.845 on ImpliHateVid** (0.831 text channel alone, 0.741 image channel alone). The
space is real and the construction in §1.3 is correct. Everything below is therefore a statement
about the *mechanism*, not about a broken pipeline.

### 6.2 The clause geometry falsifies the central premise

Mean cosine between the matched violation and exemption clause of the same pair, and mean
off-diagonal cosine among the `K` difference directions:

| encoder | cos(violation_k, exemption_k) | cos among violations | cos among **differences** | ‖v−e‖ (unit vectors) |
|---|---|---|---|---|
| CLIP-L/14-336 joint space | **0.920** | 0.829 | **0.039** | 0.389 |
| mpnet multilingual (EN clauses) | **0.833** | 0.624 | **0.067** | 0.561 |
| mpnet multilingual (ZH clauses) | **0.869** | 0.642 | **0.035** | 0.495 |

Two readings, both fatal:

1. **The matched pairs are nearly the same vector.** "attacks a protected group with a slur" and
   "quotes a slur in order to condemn it" sit at cosine 0.92 in CLIP and 0.83 in a genuine
   sentence-embedding space. The difference `v − e` is a short, low-energy residual. The whole
   candidate rests on the exemption axis being a *distinguishable direction*; it is not, in any
   encoder tested. This is consistent with the published finding that CLIP-family text encoders do
   not represent negation or stance.
2. **The `K` difference directions are mutually near-orthogonal (0.04–0.07).** If "violation
   vs. safe use" were a shared semantic axis, the pairwise differences would be strongly
   *positively* correlated. They are not — so there is no common exemption axis, only `K`
   independent noise residuals.

### 6.3 The clause directions lose to matched-count random directions

The screen's main measurement. Directions are used exactly as §3 specifies (cosine projections onto
`2K` directions), a logistic readout is fit on **train**, and ROC is read on **val**. `random_2K`
is dimension- and parameter-matched (`2K` i.i.d. Gaussian directions), reported as the mean over
**5 draws** with the observed range.

**Un-whitened CLIP joint space (`pcd_space_probe3.py`), val ROC:**

| dataset | clause_2K (vio+exm) | vio only (dim-matched) | anchor pair (K=1) | **random_2K (5 draws)** | clause − random |
|---|---|---|---|---|---|
| ImpliHateVid | 0.8042 | 0.8102 | 0.7371 | **0.8409** [0.8195, 0.8847] | **−0.037** |
| HateMM | 0.6479 | 0.6428 | 0.5389 | **0.7560** [0.7093, 0.8201] | **−0.108** |
| MHC-EN | 0.7222 | 0.7062 | 0.7018 | **0.6454** [0.5905, 0.6785] | +0.077 |
| MHC-ZH | 0.5807 | 0.5764 | 0.5529 | **0.6971** [0.6250, 0.8021] | **−0.116** |
| **mean** | 0.6888 | 0.6839 | 0.6327 | **0.7099** | **−0.046** |

**Clause directions lose to random directions on 3 of 4 datasets and by −0.046 ROC on average.**
This is precisely the pre-declared KILL condition in the mandate ("增益若不消失 = 条款语义无贡献 =
KILL") — except that the gain is not merely non-vanishing under randomisation, it is *negative*.

**Whitened variant (`pcd_space_probe2.py`, the candidate's own prereg wording asked for train-covariance
whitening), val ROC:** clause_2K 0.537 / 0.580 / 0.530 / 0.539 on the four datasets, against a full
logistic probe on the same frozen joint-space features at **0.968 / 0.867 / 0.771 / 0.851**.
Whitening removes the anisotropic mean structure in which almost all of the zero-shot anchor signal
lived, and collapses every clause arm to near chance. So the construction is bad both with and
without the prereg's whitening step; the two variants fail for different reasons.

**Multilingual sentence encoder (`pcd_space_probe4.py`), val ROC on the cached
`paraphrase-multilingual-mpnet-base-v2` transcript features** — this exists to foreclose "CLIP is
just a bad text encoder for policy clauses":

| dataset | clause_2K | vio only | random_2K (5 draws) | full LR on same features |
|---|---|---|---|---|
| MHC-EN | 0.6560 | 0.6502 | 0.6429 [0.5702, 0.7185] | **0.7484** |
| MHC-ZH | 0.7536 | 0.7493 | 0.7097 [0.5793, 0.7871] | **0.7900** |

In a proper sentence-embedding space the clause arm beats random by +0.013 and +0.044 — both
comfortably **inside the random draws' own range**, i.e. not distinguishable from a lucky draw. And
both are far below an unconstrained probe on the identical features. The failure is not
CLIP-specific.

### 6.4 The exemption term — the one component with any novelty residue — contributes nothing

`clause_2K` (violation + exemption projections) versus `vio only` (violation projections duplicated
to the same dimension), val ROC delta: **−0.006** (ImpliHateVid), **+0.005** (HateMM), **+0.016**
(MHC-EN), **+0.004** (MHC-ZH) in CLIP space; **+0.006** and **+0.004** in mpnet space. Mean ≈
**+0.005 across six measurements, with a sign flip** — noise, and far below the mandate's
`Δ ≥ +0.005` bar for a *real* effect, let alone for the component that carries the entire novelty
argument. This is the direct empirical consequence of the 0.83–0.92 pair cosine in §6.2.

### 6.5 Honest limits of the screen

- `K = 6`, not the pre-registered 12. Every comparison is dimension-matched against its own
  controls, so `K` cannot bias the clause-vs-random contrast; but a larger clause set is untested.
- Clause wording was written once and **not tuned** (tuning on val would be design iteration).
  A reviewer could argue better wording exists. The counter is §6.2: the 0.83–0.92 pair cosine and
  the 0.04–0.07 inter-difference cosine are properties of how these encoders represent stance and
  negation, not of this particular wording — and they replicate across two unrelated encoder
  families and two languages.
- The screen tests the cone **alone**, not `head + cone`. It therefore does not prove
  `head + cone ≤ head`. What it does prove is that the cone's `2K` numbers contain no clause-
  *semantic* content to add: they are outperformed by `2K` random projections of the same features.
- The screen used a logistic readout rather than the exact `relu(r − βm − τ)` parameterisation.
  The ReLU/veto form is strictly *less* expressive than the free logistic readout over the same
  projections, so it cannot rescue an arm the logistic readout could not lift.

---

## §7 — Verdict

**B2 PCD is DEAD.** Two independent, individually sufficient causes:

1. **Occupied (§5).** LaBo (CVPR 2023) already builds a trained head whose weights are a
   non-negative combination of frozen text-clause directions over frozen VLM features, with the
   clause set fixed in advance and a random-direction ablation. **Google Ads / WSDM 2025 already
   builds the paired violation/exemption policy-clause decision in a frozen multimodal embedding
   space, in production.** RoboShot (ICLR 2024) already builds paired-prompt difference directions
   with an explicit suppression operator inside a classifier. Hypothesis Engineering (COLING 2022)
   already has hand-written exemption clauses over a frozen model overriding the hate score on
   exactly this failure mode. Rule By Example (ACL 2023) already encodes hand-written moderation
   rules as vectors for hate classification. Polyhedral Conic Classifiers (CVPR 2017 / TPAMI 2020)
   own the geometry and the name; Concept Cones (ICML 2025) owns "a safety concept is a cone of
   difference directions with non-negative coefficients". Two of the candidate's stated
   mathematical claims are additionally wrong (§5.2).
2. **Premise empirically false (§6).** The matched violation/exemption clause pair has cosine
   0.83–0.92 in every encoder tested and the `K` difference directions are mutually near-orthogonal
   (0.04–0.07): there is no exemption axis to exploit. Consequently the clause directions **lose to
   dimension-matched random directions on 3 of 4 datasets (mean −0.046 val ROC)** and the exemption
   term is worth ≈ +0.005 with a sign flip. This is the mandate's explicit KILL condition, met
   before the test set was touched.

Both the structural constraint in §1.3 (three of four val-best cells cannot host the cone at all)
and the ceiling in §6.3 (an unconstrained probe on the same features reaches 0.77–0.97 where the
cone reaches 0.53–0.80) are recorded as standing project facts, independent of this candidate.

**No pilot was run.** `idea-stage/PCD_FREEZE.md` and `PCD_RESULT.md` do not exist, by design.

### 7.1 Transferable facts for future rounds

1. **The CLIP joint space is recoverable from the existing caches** by applying the frozen
   `visual_projection`/`text_projection` to the stored pre-projection poolers, and it carries real
   zero-shot signal (val ROC 0.845 on ImpliHateVid from a generic hate anchor, no training). Any
   future zero-shot / text-anchored candidate can use this at zero extraction cost.
   `idea-stage/pcd_space_probe.py` is the reference implementation.
2. **Qwen/LoRA `text_feats` is not a sentence-embedding space.** It is the mean of the last 4
   tokens of the assistant generation-prompt tail inside a multimodal prompt. **No text-anchored
   mechanism can ever be built on the project's three strongest cells without a new extraction
   pass.** This is a standing asset fact, like PRES's OCR-window fact (§8.10.6).
3. **`2K` random projections of the frozen joint feature are a strong baseline** (val ROC up to
   0.88 on ImpliHateVid, 0.82 on HateMM, with wide spread across draws). Any future
   "K interpretable directions" candidate in this project must beat a matched random-direction arm
   *averaged over several draws* — a single draw is uninformative at this spread.
4. **Text encoders in this project do not represent policy exemptions.** Violation and matched
   safe-use clauses embed at cosine 0.83–0.92 in both CLIP and a multilingual sentence encoder.
   Any future mechanism that needs to separate hate from condemnation/quotation/reclaimed use
   *inside a frozen text embedding* inherits this measured negative; the separation has to come
   from a generative model that reasons, not from an embedding direction.
5. **The random-direction ablation is itself published** (WaffleCLIP, ICCV 2023, showed random
   descriptors frequently match semantic ones). It remains the right control, but a null result
   there is a confirmation of known work, not a novel finding.

### 7.2 Reproducibility index

| artifact | path |
|---|---|
| this specification + novelty check + screen | `idea-stage/PCD_SPEC.md` |
| zero-shot joint-space probe | `idea-stage/pcd_space_probe.py` |
| whitened trained-readout probe | `idea-stage/pcd_space_probe2.py` |
| un-whitened trained-readout probe, 5 random draws | `idea-stage/pcd_space_probe3.py` |
| clause geometry + multilingual sentence-encoder probe | `idea-stage/pcd_space_probe4.py` |
| launcher (script, not an `&&` chain — §8.12 defect class) | `idea-stage/pcd_probe_launch.sh` |
| run log | `logging/runs/pcd_spec_probe/run.log` |
