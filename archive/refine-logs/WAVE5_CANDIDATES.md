# WAVE-5 CANDIDATE RECON — is there a NOVELTY-BEARING adaptation-family method?

**Agent:** wave-5 ideation/recon. **Date:** 2026-07-17. **ZERO GPU / ZERO test-touch / ZERO Modal spend / ZERO user
interaction.** Reading + forensic reasoning + prereg-shaped design only. Deliverable = this committed doc.

**Frame.** F45 established the campaign's convertibility law: *the only lever that CONVERTS (re-decides, Pareto) rather than
re-ranks (rotates) is ADAPTATION of the representation* — generic encoder LoRA turned ZH's frozen rotation into a +0.0313
Pareto pass (B3), where frozen-identity swaps only rotate. The generic-LoRA-HateMM performance measurement is already in
ceremony (line A / task #55; ~75–85% PASS prior). **This wave asks the narrow follow-on:** is there a *novelty-bearing*
adaptation — one whose OBJECTIVE or COUPLING is specific to the retrieval-contrastive hateful-video architecture, not a
generic encoder-class lever — that is (i) not banned, (ii) mechanistically ≥ generic LoRA, (iii) defensibly novel-in-field?

**Governing docs read this recon (verbatim):** `state/directions_tried.json` (23 dead + 9 bans + positives bank),
`findings.jsonl` F44–F50, `LORA_HATEMM_FORENSIC_RECON.md` (the P9-regime disambiguation §1, load-bearing),
`B3_ZH_LORA_DECOMPOSITION.md` (F45), `ENCODER_SWAP_DIAGNOSIS.md` (F44), `FA_GATE_RECORD.md`/F50, `TERMINUS_round3_mllm_plus3.md`,
`WAVE4_CANDIDATES.md`, `research-wiki/EXP_p9_lmm_rgcl_video.md` (**P9 + P9b in full — the decisive record for candidate 1**),
`research-wiki/experiments/exp-tarc-t0.md` (TARC scope), and the novelty-scope memory (in-field novelty definition).

---

## BOTTOM LINE UP FRONT

1. **The adaptation family opens NO new *performance* route beyond what generic LoRA already tests.** Every adaptation lever
   bottoms out at one of two adapted objects: the **encoder** (Axis-B, generic-LoRA = B3/B4/LoRA-HateMM, D7-encoder-class) or
   the **joint decision** (the retrieval-contrastive loss backpropagated into the LoRA'd MLLM — **that is exactly P9b/D3,
   already run and KILLED**). There is no third adapted object. So there is **no novelty-bearing adaptation that is both
   non-encoder-class AND non-P9b.**
2. **Candidate 1 (retrieval-loss-coupled LoRA) is a P9b variant, not a fresh cell.** P9b already backpropagated the
   retrieval-contrastive term into the LoRA'd MLLM and found a **redistribution law**: at equal weighting the rgcl term moved
   +1.8pt to the kNN read-out at −1.8pt from the head (ZH), an almost-exact swap, and **no cell of the wave — either arm,
   either read-out — beat its protocol-matched floor** (`EXP_p9:340-364`). Candidate 1's non-isomorphism (our align-fusion
   head loss vs RA-HMD's in-LMM head loss; drop the LM/cls co-training) is real but thin, and it is priced *against* that
   mechanism pair. **Recommend documented pre-kill** unless a specific confound-removing design is user-authorized (§5).
3. **Candidate 2 (retrieval-mined hard-negative contrastive SFT *curriculum*) is the one genuinely fresh, cheap, and
   novelty-defensible member — but its performance prior is LOW and it opens no new dataset.** It is a *data-curriculum*
   lever (generative yes/no SFT on retrieval-mined confusable pairs), non-isomorphic to P9b (a *loss* lever) and to TARC (a
   *head* lever, frozen encoder), reusing the B3 SFT machinery at ~one-run cost. Its realistic best case is a **more robust
   ZH pass + inherited HateMM pass** — a cleaner, coupling-*novel* 2-dataset encoder story — **not** a novelty-clean
   non-encoder route and **not** a new second dataset (the second dataset is structurally unavailable to any text-stream
   adaptation lever; §0.2).
4. **Recommended next action = SEQUENCE: hold ALL adaptation GPU behind the LoRA-HateMM verdict; queue nothing now** (§4).
   The adaptation family's only live role is *novelty-strengthening on datasets generic LoRA already passes*, contingent on a
   D7 sub-ruling. Per-branch action in §4. Candidate 2 is pre-registered-in-outline as the reserve variant to run **only** on
   a user D7-sub-ruling request, **never** speculatively.

---

## 0. The three walls every wave-5 candidate inherits (re-verified this recon)

### 0.1 There are only two adapted objects, and both are already spoken for
The pipeline is `LoRA?-MLLM encoder → align-fusion head (triplet-margin+BCE hybrid) → top-20 signed-cosine kNN vote`. An
"adaptation" can touch exactly one of two things:
- **The encoder** (train LoRA, then a *separate* fresh head reads the features). Objective is head-agnostic ⇒ **generic
  encoder LoRA = B3 (ZH +0.0313 PASS), B4 (EN FAIL), LoRA-HateMM (in ceremony).** Axis-B, **D7-encoder-class-dead by ruling
  F24** (`TERMINUS §1 Axis-B`).
- **The joint {encoder+decision}** (backprop a retrieval/contrastive loss *through* the head *into* the LoRA). **This is
  P9b/D3** (`EXP_p9:206-378`): joint LM+cls+rgcl LoRA-SFT, decision by our kNN. **KILLED** (`directions_tried` P9b:
  "head↔memory redistribution, 0/12").

A "novelty-bearing coupling" changes the *objective* (candidate 1) or the *data/curriculum* (candidates 2/3) of one of these
two — it does **not** create a third adapted object. So the D7-encoder-class question is never *escaped* by the adaptation
family; at most it is *reframed* into the narrower sub-ruling "is a retrieval-coupled adaptation objective novel vs generic
LoRA?" (stronger for a data-curriculum lever than for a loss-coupling lever; §2).

### 0.2 The ≥2-dataset arithmetic is structurally fixed by modality-locus (F44/F45)
B3 already passes **ZH final-epoch with generic LoRA**, so a novelty-bearing variant must **HOLD ZH's pass AND add HateMM or
EN.** The second leg is where the family dies on performance:
- **LoRA moves the TEXT stream only** (F45: ZH text AUC frozen 0.847 → LoRA 0.925, **image untouched −0.007**). Every
  adaptation-family curriculum/objective in this family sharpens the *language* representation.
- **HateMM decides on the IMAGE stream** (F44: image train-LOO AUC 0.826, highest of the three; the frozen-swap Pareto is
  image-grounded). A text-stream adaptation lever **inherits** HateMM's frozen/generic-LoRA pass but adds ~nothing on top —
  so it cannot *newly* convert HateMM that generic LoRA didn't already carry.
- **MHC-EN is label-limited, not representation-limited** (F44/SAV #18: net −1 rotation; image stream collapsed 0.734→0.599).
  **No representation lever — adapted or not — clears +0.03 on EN** (B1/B2/B4 all fail; F50 confirms even the best-ever
  fusion AUC 0.898 is unconvertible). So EN is closed to the entire family.

⇒ **The only reachable second dataset is HateMM, and only if generic LoRA already passes it** — in which case a
novelty-bearing variant is *redundant for the performance conjunct* and serves only the D7/novelty story. This is the single
most important structural fact of the wave.

### 0.3 The P9b redistribution law (the direct prior for any retrieval-loss coupling)
P9b is the exact experiment "MLLM trained BY our retrieval-contrastive loss, decision BY our updatable kNN memory"
(`EXP_p9:216`, its own words: "the literal form of the goal"). Its clean D3−C3′ mechanism pair (pure rgcl-term effect, same
branch/recipe): **kNN read-out +1.8pt ZH / +0.2pt EN, MLP head −1.8pt ZH / −1.2pt EN — an almost-exact swap, net ~0**
(`EXP_p9:352-364`). *Conclusion, verbatim:* "the RGCL loss works as designed (shapes the space for the memory), but in this
regime it buys no system-level accuracy — no cell of the wave beats its protocol-matched floor." This is the campaign's
**5th better-signal-no-conversion** pattern applied to *coupling the retrieval loss into the adapter itself.*

---

## 1. Candidate space — adjudicated honestly

| # | candidate | what it adapts | verdict | why (specific ban / finding, scope-quoted) |
|---|---|---|---|---|
| **1** | **Retrieval-loss-coupled LoRA** — backprop our align-fusion head's triplet+BCE-over-kNN loss into the LoRA'd encoder | joint {encoder+decision} | **PRE-KILL (P9b variant)** | This IS P9b's adapted object. P9b `loss_ratio[1,1,1]` already backpropagated a retrieval-contrastive term into the LoRA'd MLLM; **KILL**, redistribution-not-addition, "no cell beats floor" (`EXP_p9:340-364`). Non-iso (our head vs RA-HMD head; no LM/cls co-train) is thin and priced against the D3−C3′ pair (§5). |
| **2** | **Retrieval-mined hard-negative contrastive SFT curriculum** — generative yes/no LoRA-SFT whose training examples/pairs are constructed from the archive's confusable neighbours (own train split + its gold labels) | encoder (data curriculum) | **LEAD (LOW prior; conditional)** | Genuinely fresh: a *data-curriculum* lever, non-iso to P9b (loss), TARC (head), P11 (scores). Reuses B3 SFT machinery. But ≥2-dataset arithmetic (§0.2) caps it at hold-ZH+inherit-HateMM; D7 = user sub-ruling. Full treatment §2. |
| **3** | **Memory-error-focused SFT curriculum** — SFT curriculum = train items the frozen-encoder LOO kNN vote gets wrong | encoder (data curriculum) | **FOLDS INTO 2** | A special case of candidate 2 (retrieval-geometry-shaped SFT curriculum, using LOO-vote errors as the mining signal instead of hard-neg pairs). F47 healthy-signal check: encoder-level LOO vote is 0.72–0.81 (not the memorized trained-head 0.998), so the curriculum target is non-degenerate — but same arithmetic/D7 caps. Treated as a candidate-2 mining variant. |
| **4** | **Retrieval-consistency-regularized SFT** — additive regularizer tying encoder adaptation to kNN self-consistency | joint | **PRE-KILL** | Softer candidate 1; grazes P9b (added-rgcl-term) + TARC V3 (additive aux regularizer "regularization-only; val moves, test flat"). Same redistribution prior. No probe. |
| **5** | **Retrieval-conditioned soft-prompt / prefix adaptation** | encoder (PEFT variant) | **PRE-KILL** | Still encoder-class (D7-dead); soft-prompt is a *weaker* generic PEFT than LoRA with no evidence it beats it. No performance or novelty gain over B3. No probe. |
| **6** | **Encoder-level target-contrastive SFT** (SFT the MLLM to separate same-community hate/benign — TARC's structure moved from head to encoder) | encoder (data curriculum) | **PRE-KILL** | Grazes TARC (test-flat even at GT-target oracle, `exp-tarc-t0 §10.4`: "TEST is flat") + C3-target (7B target predictor +0.0094, "MHC anti-informative") — needs MLLM-predicted target on MHC where the 7B can't predict it. LOW inside candidate-2's space; dominated by candidate 2's label-only mining. No probe. |

**The crux (candidate 1 vs the P9/P9b line — the task's explicit question).** Is "RGCL loss into the encoder" a TARC-banned
*aux loss*, or a fresh *primary* loss? **Answer: neither reframing helps.** TARC's banned term (V3) was an *additive head
regularizer with a frozen encoder* — candidate 1's loss is the *primary* loss into the *encoder*, so it is **not** TARC.
But the primary-retrieval-loss-into-the-LoRA'd-MLLM is precisely **P9b/D3**, which is already run and killed. The
LORA_HATEMM recon's regime line (`§1`: encoder-level *generative* SFT + fresh head, vs decision-level *joint* SFT) does open
a nominal third regime — "encoder-level fresh-head loss backprop'd end-to-end into the LoRA" — but that regime's *mechanism*
is governed by P9b's redistribution law (the retrieval loss reshapes the space *for* the memory at the head's expense, net
~0), and it must additionally **beat the B3 encoder-LoRA floor it is built on top of**, on ≥2 datasets, while carrying
P9b's engineering walls (Qwen-forward embed contract, bs=1 in-batch degeneracy → 4-frame/bs≥4 fix, ~40 min/run × 3-seed ×
2-dataset local SLURM). That is a LOW-prior, high-cost, D7-encoder-class re-run of a killed cell.

---

## 2. LEAD candidate — full treatment: Retrieval-mined hard-negative contrastive SFT curriculum

**(a) Mechanism / bandwidth / injection point.** Generic B3 LoRA-SFT presents each train video *independently* with a
generative "is this hateful? Yes/No" target — the adaptation objective is blind to the retrieval geometry the method uses at
inference. Candidate 2 keeps the identical generative objective and machinery but **constructs the SFT data from the
archive's confusable structure**: for each anchor, mine (via the frozen-encoder kNN over the own-train split) its hardest
**opposite-label** neighbour (and optionally its hardest same-label pseudo-positive), and build SFT records that force the
adapter to resolve exactly the same-community-opposite-intent boundary the kNN vote will later have to decide. Injection =
encoder LoRA-SFT **data curriculum** (which examples, in what pairing/weighting); bandwidth = own-train gold labels + the
frozen retrieval geometry (no new external signal). Read-out = the standard fresh RGCL head + top-20 kNN (B3 protocol,
archive OFF), 3-seed paired vs frozen-CLIP, dual-protocol +0.03/+0.03.

**(b) Non-isomorphism (verified against the cited scope language).**
- **vs P9b/D3 (retrieval-contrastive LOSS into the LoRA):** P9b adds a *contrastive loss term* on embeddings (metric
  learning backprop'd into the encoder). Candidate 2 uses the *unchanged generative LM objective* and changes only the
  *data distribution* (retrieval-curated curriculum). Loss-lever vs data-lever = different mechanism, and candidate 2
  avoids P9b's two engineering walls entirely (no Qwen-forward embed contract, no bs=1 in-batch degeneracy — it is plain
  generative SFT with a curated `train.json`, same code path as B3). *This is the load-bearing non-isomorphism.*
- **vs TARC (`directions_tried`: "predicted-target conditioning of retrieval graph; regularization-only; val moves, test
  flat"):** TARC reshaped the *head's* mining/vote/regularizer over a **frozen** encoder using a **predicted target
  category**. Candidate 2 reshapes the *encoder's SFT curriculum* using **label-only retrieval-mined confusable pairs** — a
  different injection point (encoder SFT vs frozen-head mining) and a different signal (gold-label kNN neighbours, not a
  predicted target class). TARC never adapted the encoder.
- **vs P11 / "MLLM-scores-as-training-signal" (banned):** candidate 2 uses **gold train labels** to define the target and
  the retrieval geometry to select pairs; it uses **no MLLM score as a label or regression target.** Clear.
- **vs "kNN-vote-pool expansion via pseudo-labels" (banned; scope: "representation-training expansion only"):** candidate 2
  adds **no pseudo-labelled entries to the vote pool**; every SFT example is a gold-labelled own-train video. It is exactly
  the *allowed* "representation-training" side of that ban, not the banned vote-pool side.
- **vs the training-data veto (`banned_constraints`: "single-dataset train split ONLY; no cross-dataset mixing"):** the
  mining and the SFT both run **within one dataset's own train split** — clears the veto (identical discipline to B3).

**(c) Expected-effect prior — LOW, grounded in F44/F45/F50.**
- **ZH (the leg it must hold):** the ZH gain is text-stream sharpening (F45). A hard-negative curriculum focuses adaptation
  on the confusable same-community text boundary ⇒ **plausibly sharpens the text stream *further* than generic SFT**, whose
  main realistic payoff is a **more robust ZH pass** (pushing the val-selected protocol over the bar too — B3's val-sel was
  +0.0246 FAIL, a 78-dev selection artifact per F45, not instability). ~30–40% it *strengthens* ZH; ~50% ≈ generic LoRA;
  ~10–20% the curriculum overfits the tiny split and regresses.
- **HateMM:** decided by the image stream (F44); a text-focused curriculum adds ~0 over generic LoRA's image-inherited pass
  ⇒ **inherits, does not newly convert.** No new second dataset here.
- **EN:** label-limited (F44) ⇒ **dead**, like every representation lever.
- **Net:** best case = hold-ZH (more robustly) + inherit-HateMM = a cleaner, coupling-*novel* 2-dataset **encoder** story;
  it does **not** produce a novelty-clean non-encoder pass and does **not** add a dataset generic LoRA lacks. Performance
  prior on the goal (a *new* ≥2-dataset conjunct that generic LoRA doesn't already deliver): **~5–10%.**

**(d) Novelty defensibility vs D7 (honest).** Stronger than generic LoRA, still a user sub-ruling. Generic LoRA is
plainly "a 2024–25-standard technique, Axis-B" (F45) — D7-dead by F24. Candidate 2's contribution is the **coupling**
(retrieval-mined contrastive SFT curriculum specific to the retrieval-contrastive memory), which is the exact
"OBJECTIVE/COUPLING specific to the architecture" the wave sought, and is **novel-in-field** under the project's definition
(no hateful-video method SFT-adapts an encoder on retrieval-mined confusable pairs for a kNN-vote memory; novelty-scope
memory). **But** the adapted object is still the encoder, so D7's "encoder-class levers do not satisfy novelty" *may* cover
it. The honest framing to the user: this is a **narrower, stronger D7 sub-ruling** than option-(c) — "does a *retrieval-
coupled SFT curriculum* count as a novel MLLM integration, distinct from generic LoRA?" — not a route that escapes D7 by
construction.

**(e) Cost / sequencing.** Reuses the **B3/LoRA-HateMM SFT machinery** — the only new artifact is the curated
`data/lora_sft/<DS>/train.json` builder (retrieval-mined pairs from the frozen-encoder kNN, ~CPU) + a config clone.
Per dataset: ~3.5–4 h SFT + ~0.4 h extract + ~2 min head (same as one LoRA-HateMM run). **Local SLURM only** (SFT is a
training run; Modal is features-only, cannot host it) ⇒ **queue-blocked behind the LoRA-HateMM chain** (task #55). No Modal,
no forward surgery.

**(f) Kill-switches (house-style, pre-declared).**
- **K-C2-0 (mining-validity, $0 CPU pre-check):** the retrieval-mined hard-negative set must differ non-trivially from the
  generic all-videos set (else the curriculum ≡ B3 by construction — auto-KILL as "not a distinct method"). Quantify:
  fraction of anchors whose mined hard-neg is a *same-community opposite-label* pair (the confusable case candidate 2
  targets); if the frozen space already makes the nearest opposite-label neighbour community-matched ≥90% of the time (the
  TARC H0 shape, `exp-tarc-t0 §8`), the curriculum is a near-no-op ⇒ **pre-GPU KILL**.
- **K-C2-1 (performance, primary):** ZH LoRA-curriculum − CLIP paired Δacc ≥ +0.030 AND ΔmF1 ≥ +0.030, 3/3, **AND** ≥
  generic-B3-LoRA − 0.014 (must not regress the pass it inherits), judged independently per protocol. Below → KILL.
- **K-C2-2 (add-over-generic, the novelty-earning bar):** ZH LoRA-curriculum must **beat generic B3 LoRA** by a real margin
  on ≥1 protocol (the whole point is that the coupling *adds*); if it merely ties generic LoRA, the coupling earns no
  novelty and the route reduces to "generic LoRA with extra machinery" ⇒ report as no-value.
- **K-C2-3 (regime sanity):** if it lands below the CLIP floor on ZH, the curriculum degraded the one leg it was built to
  strengthen ⇒ bank the negative.

---

## 3. RANKING

| # | candidate | one-line | perf prior (new conjunct) | novelty vs D7 | cost | overall |
|---|---|---|---|---|---|---|
| **1** | **retrieval-mined hard-neg SFT curriculum** (cand 2, incl. cand-3 mining variant) | data-curriculum coupling of memory→encoder-SFT; holds ZH, inherits HateMM | LOW (~5–10%) | strongest in family (narrow D7 sub-ruling) | one SFT run/dataset, local SLURM, reuses B3 machinery | **LEAD (conditional)** |
| **2** | **retrieval-loss-coupled LoRA** (cand 1) | = P9b's adapted object with our head loss | ~dead (P9b redistribution law + must beat B3 floor) | encoder-class, no stronger than generic | P9b forward-surgery + bs≥4 fix, ~40min/run ×6 local | **PRE-KILL** (§5) |
| — | cand 4/5/6 | consistency-reg / soft-prompt / encoder-target-contrastive | pre-killed (§1) | — | — | **PRE-KILL** |

---

## 4. RECOMMENDED NEXT ACTION — SEQUENCE behind the LoRA-HateMM verdict (queue nothing now)

**Do not queue candidate 1/2/3.** The adaptation family opens no non-encoder ≥2-dataset performance route (§0.1); its only
live value is *novelty-strengthening on datasets generic LoRA already passes*, which is gated on the D7 ruling AND on the
LoRA-HateMM outcome. Wait for line A (task #55).

- **BRANCH A — LoRA-HateMM PASSES (~75–85% prior).** The goal's **performance conjunct is met by generic encoder LoRA**
  (ZH + HateMM); the *only* remaining blocker is the D7 novelty ruling on encoder-class levers (terminus option-c). **Action:**
  surface to the user that performance is met and the decision is D7 — do **not** spend GPU on candidate 2 speculatively (it
  cannot add a dataset; §0.2). **IF and only if** the user opens a D7 sub-ruling and asks for a coupling-based novelty variant
  to strengthen the case, **then** run candidate 2 pre-registered as a **novelty upgrade on ZH+HateMM** (hold both passes,
  earn the K-C2-2 add-over-generic margin), never as a new-dataset bet. Candidate 2's outline (§2) is the ready reserve.
- **BRANCH B — LoRA-HateMM FAILS (~15–25%).** Generic encoder LoRA is **ZH-specific**; the performance conjunct is unmet
  (only ZH). The missing second dataset is structurally unavailable to *any* text-stream adaptation lever — HateMM is
  image-borne (LoRA doesn't move image; if generic LoRA failed HateMM, a text-curriculum won't rescue it) and EN is
  label-limited (§0.2). **Candidate 2/3's curriculum is a text-stream sharpener ⇒ cannot convert the dataset generic LoRA
  couldn't.** **Action:** the round-3 terminus stands; escalate to the user (the adaptation family is exhausted on
  performance). Do **not** run candidate 1/2/3.

Either branch: **candidate 1 stays pre-killed** (§5) regardless of the LoRA-HateMM outcome — it is a P9b re-run.

---

## 5. DOCUMENTED PRE-KILLS (so wave-6 does not re-spend)

- **Candidate 1 — retrieval-loss-coupled LoRA (our align-fusion head loss backprop'd into the LoRA'd encoder).**
  **PRE-KILL as a P9b variant.** P9b/D3 already backpropagated a retrieval-contrastive term into the LoRA'd MLLM and decided
  by our kNN; **KILLED** with the redistribution law (kNN +1.8 / head −1.8 ZH swap, net 0; "no cell beats its
  protocol-matched floor", `EXP_p9:340-364`). Candidate 1's differences (our head vs RA-HMD's in-LMM head; drop LM/cls
  co-training; direct 3584-d shaping vs P9b's "indirect" proj_dim=1024 shaping, `EXP_p9:242`) are a real but thin
  non-isomorphism that (i) is governed by the same redistribution mechanism, (ii) must additionally **beat the B3 encoder-LoRA
  floor it sits on**, on ≥2 datasets, (iii) is still **D7-encoder-class**, and (iv) carries P9b's forward-surgery + bs≥4
  engineering cost. **Reopen only** if a user authorizes a specific confound-removing design (pure our-head-only objective,
  no LM/cls, at effective batch ≥8) AND accepts it as a bounded P9b-follow-up, not a fresh axis.
- **Candidate 4 — retrieval-consistency-regularized SFT.** Softer candidate 1; grazes P9b (added-rgcl-term) + TARC V3
  (additive aux regularizer, "val moves, test flat"). Same redistribution prior. No probe.
- **Candidate 5 — retrieval-conditioned soft-prompt / prefix adaptation.** Encoder-class (D7-dead); weaker generic PEFT than
  LoRA, no evidence it beats it. No probe.
- **Candidate 6 — encoder-level target-contrastive SFT.** TARC's target structure moved to the encoder SFT; grazes TARC
  (test-flat at GT oracle) + C3-target (7B target predictor +0.0094, MHC anti-informative — needs MLLM-predicted target on
  MHC). Dominated by candidate 2's label-only mining. No probe.

---

## PROVENANCE
- Adapted-object dichotomy + P9b as the retrieval-loss-coupling cell: `research-wiki/EXP_p9_lmm_rgcl_video.md:206-378`
  (D3 pre-reg + wave + verdict + D3−C3′ mechanism pair); regime disambiguation `LORA_HATEMM_FORENSIC_RECON.md:53-101`.
- ≥2-dataset arithmetic (modality-locus): F44 `ENCODER_SWAP_DIAGNOSIS.md` (`8a48938`: HateMM image 0.826, EN image collapse
  0.599, EN rotation net −1); F45 `B3_ZH_LORA_DECOMPOSITION.md` (`d76e407`: LoRA text 0.847→0.925, image untouched −0.007,
  ZH Pareto +0.111/−0.003); F50 `FA_GATE_RECORD.md` (EN fusion AUC 0.898 unconvertible).
- Generic-LoRA passes ZH: B3 final-epoch +0.0313/+0.0453 3/3 (`directions_tried` positives_bank B3-lora-zh); LoRA-HateMM in
  ceremony, ~75–85% PASS prior (`LORA_HATEMM_FORENSIC_RECON.md:16-49`, task #55).
- Scope quotes: P9b/P11/pseudo-label/veto `state/directions_tried.json`; TARC test-flat `exp-tarc-t0.md:651-679`,
  C3-target +0.0094 `directions_tried` C3-target; D7/terminus option-c `TERMINUS_round3_mllm_plus3.md:104`.
- Novelty-in-field definition: novelty-scope memory (`novelty-scope-and-plan.md`).
- **Required statements:** ZERO GPU / SLURM / Modal spent by this recon; no held-out test metric read or produced; no
  `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated. Committed on `main`, not pushed.
