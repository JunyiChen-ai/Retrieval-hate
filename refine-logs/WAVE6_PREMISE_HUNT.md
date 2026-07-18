# WAVE-6 ADVERSARIAL PREMISE HUNT — is "box empty" hiding a fourth unexamined premise?

**Agent:** wave-6 adversarial premise-hunter (ZERO GPU / ZERO Modal / ZERO test-touch / ZERO user
interaction). Reading + forensic reasoning only. **Date:** 2026-07-18. Deliverable = this committed doc.
Touched no `state/`, prereg, config, `research-wiki/`, or live ceremony artifact.

**Mission.** The campaign's three reopenings each came from an unexamined PREMISE in the *prose*, not the
numbers: **F46** (the banked zeros were only LINEAR-conditional → per-item router gate), **F48** (F44's
fusion dismissal assumed *concat* but the deployed head is *Hadamard* → FA gate), **F54** (‑"LoRA moves
text only" was empirical, not architectural → premise-(d) gate). All three cheap cells were then measured
and **died** (F47/F50/F55), but each was a real, uncovered cell. Wave-6 hunts the next one in the larger
ledger (23–24 dead + F44–F56), adversarial to how the records are *worded*.

**Docs read verbatim:** `state/directions_tried.json` + `findings.jsonl` F44–F56;
`TERMINUS_round3_mllm_plus3.md`; `ROUTER_GATE_RECORD.md` (F47), `FA_GATE_RECORD.md` (F50),
`PREMISE_D_GATE_RECORD.md` (F55), `CAND2_VERDICT_REVIEW.md` (F56), `TIE_BRANCH_RECON.md` (F54),
`LORA_HATEMM_VERDICT_REVIEW.md` (F53), `ENCODER_SWAP_DIAGNOSIS.md` (F44), `B3_ZH_LORA_DECOMPOSITION.md`
(F45); `DRAFT_analysis_chapter.md` §3.6–§3.9 (the four stated laws).

---

## BOTTOM LINE UP FRONT

**No fourth premise reopens the goal.** The four flagged surfaces resolve as: **two out-of-scope**
(head-recipe and epoch-ensembling are generic classifier/protocol levers, not novel-MLLM mechanisms — they
cannot satisfy the D7 clause *regardless of performance*, and one is spirit-isomorphic to the cross-seed
ensemble veto), **one refuted at inspection** (hybrid CLIP+LoRA memory is literally inside the F47 ban
scope — its per-item decision *is* a max-confidence selector over banked votes, no new information source —
and its rank-merge variant is the W2-E reorganisation ban), and **one genuinely-unrun but diagnostic**
(a per-stream decomposition of the *passing* HateMM-LoRA cell — $0, paper-relevant, prior-moving, but not a
conversion lever). The premise-level box is **empty for goal-reopening**: every surface is either
out-of-scope for the novel-MLLM clause or subsumed by an exact ban.

**Single recommended action:** run the **$0 per-stream decomposition of the HateMM-LoRA passing cell**
(surface 4) — the one honestly-unrun cell. It is diagnostic (does **not** reopen the goal), but it is the
best-priced move on the board: it converts the analysis chapter's currently-**reasoned** "HateMM gain is
image-inherited" claim (§3.9) into a **measured** one, empirically tests the F54 "image is architecturally
movable" correction on the passing cell, and is the only thing that would move premise-(a)'s (vision-SFT)
prior off its armchair value. If it too resolves as "image inherited / text secondary" (the modal
outcome), the box is empty at the premise level as well, and the live decision is unchanged: the **D7 user
ruling on generic LoRA** (performance conjunct already met on 2 datasets under final-epoch, F53).

---

## The premise being hunted

Every kill/tie in the ledger operates on the **feature side** (encoder identity F44/B1/B2; encoder
adaptation F45/F53/F56; feature composition F50/F55) or the **decision side** (prior/rerank/pool/threshold/
route P1/P2/P3/B5/F47) of a **head recipe that was frozen campaign-wide for comparability**:
`fusion_mode='align'` (element-wise Hadamard of two L2-normed projections), `num_layers=3`,
`map_dim=proj_dim=1024`, `dropout=[0.2,0.4,0.1]`, `batch_norm=False`, triplet-margin+BCE hybrid loss,
`hard_neg=1`, 30 epochs / warmup 5, top-20 arithmetic rank-weighted signed-cosine kNN vote, memory=train
(`ROUTER_GATE_RECORD.md` §1). The adversarial question: does that freeze *conceal* a live cell — is
anything true only *because* the recipe was held fixed, that a banked negative never actually covered?

---

## SURFACE-BY-SURFACE ADJUDICATION

### Surface 1 — Head architecture / objective variation on the ADAPTED features → **OUT OF SCOPE (partially banked)**

**Mechanism.** The head recipe was frozen for comparability and never ablated *as a lever*. On adapted
(LoRA) features the neighbourhood structure changes materially (F45: LoRA purity **+0.075**, the largest
reorganisation of any lever), so a recipe tuned for CLIP features (topk, dropout, epochs, objective) could
in principle be suboptimal for LoRA features. The task's sharpest sub-form: F47 measured that the head
**memorises its own bank** (CLIP LOO train acc **0.998** vs Qwen 0.800) — is early-stop / regularisation of
that over-fit an untested conversion lever?

**Non-isomorphism vs exact ban language.** There is **no banked negative on pure head-architecture
variation** — honestly uncovered as a *performance* lever. BUT the two goal-relevant realisations are both
covered or out-of-scope:
- *Head-objective coupled to the encoder* is **P9b**, banked dead, and **F51 closes the space**: quote —
  *"adaptation has exactly two adapted objects — encoder (generic LoRA…) and joint encoder+decision
  (retrieval-loss-into-LoRA = P9b/D3, KILLED…). No third object exists."* A head-objective change that
  couples memory into training *is* the P9b object (redistribution ±1.8pt, 0/12).
- *Uncoupled head-recipe tuning* (topk / dropout / epochs / triplet-vs-InfoNCE, no encoder coupling) is
  **generic classifier tuning** — it adds **no MLLM role** (the MLLM's only role here is the D7-dead
  encoder), so it **cannot satisfy the "MLLM meaningfully AND novelly integrated" clause regardless of
  performance.** Per the stall rule it is **tactics** (parameter variation that does not change the
  structural conclusion), not a new direction.

**Expected effect (honest, ≥2-dataset lens).** The only goal-relevant target is EN. F44 rules EN
**label-limited** (residual errors a hard core, net −1 videos on the swap); a better-regularised *head*
cannot fix a **label-limited** core — regularisation trades variance, it does not manufacture the signal a
label-limited problem lacks. Memorisation does not even transfer to a test penalty: deployment votes with
memory=train and query=test (query never in the bank), and val-selection already regularises epoch choice.
**Prior of EN conversion ≈ 0.** HateMM/ZH already pass.

**Gate design / cost / kill bar.** None warranted: the goal-relevant coupled case is banked (P9b/F51); the
uncoupled case is out-of-scope for D7 and tactics-classed. No $0 gate would change the D7/goal picture.

**Verdict: OUT OF SCOPE.** Genuinely unexamined as a performance lever, but it produces no novel MLLM
mechanism (coupled = P9b; uncoupled = generic tuning). "Unexamined" ≠ "goal-relevant."

### Surface 2 — Selection-protocol lever: within-seed epoch-window averaging → **OUT OF SCOPE + spirit-isomorphic to a veto**

**Mechanism.** The ZH val-sel FAIL is documented 78-dev selection noise (F45: LoRA dev-acc plateaus at
0.8718 by ~ep19 while test climbs to ep29, so argmax-dev undershoots). An epoch-window average *within a
single seed* (average the vote/decision over a window of late epochs, NOT across seeds) could de-noise the
selection and repair ZH val-sel.

**Non-isomorphism vs exact ban language.** Not *literally* the `"cross-seed ensembles"` ban (this is
within-seed, across epochs) and not `B5`'s `"per-encoder threshold/operating-point calibration"` (this is
epoch selection, not a τ). So it is literally uncovered. BUT:
- It is a **temporal ensemble** — spirit-isomorphic to the user's `"cross-seed ensembles"` veto (the user
  ruled ensembling out as "not a contribution"; averaging late-epoch decisions is the same move on the
  time axis).
- It adds **no MLLM role** → cannot satisfy the novel-MLLM clause. It is a **generic selection protocol**,
  the same class as B5's operating-point calibration (D7-irrelevant, performance-line only).

**Expected effect (honest).** It targets **only the ZH val-sel leg**. But the blocker there is **not
performance** — the performance conjunct is **already met on 2 datasets under final-epoch** (F53: HateMM
+0.0573/+0.0682 SOLID, ZH +0.0313/+0.0453 MARGINAL). Repairing ZH *val-sel* converts a leg whose binding
blocker is **D7 novelty**, not the protocol. It changes no D7 answer and opens no dataset. The protocol
choice (final-epoch vs val-selected) is already a **known pending user decision** (novelty-scope memory:
"protocol choice = pending user decision"); the epoch-average is just a generic instrument for it.

**Gate design / cost / kill bar.** A diagnostic would need per-epoch head checkpoints to average votes
(only e29 snapshots are banked — the router snapshot). Reading per-epoch *test* acc from trainlogs to
"select a window" is not a deployable method (window can't be chosen on test). Not worth building: it is
out-of-scope and addresses a non-binding constraint.

**Verdict: OUT OF SCOPE.** Generic protocol/ensemble lever; strictly addresses a non-binding (already-met-
under-final-epoch) constraint; needs a user *protocol* ruling, not a run. Not goal-reopening.

### Surface 3 — Hybrid CLIP+LoRA memory (dual-index / score-sum) → **REFUTED AT INSPECTION (literally banned)**

**Mechanism.** After LoRA passes, both query and memory are LoRA-embedded (single encoder). A *hybrid*
memory would combine the CLIP arm and the LoRA arm by a **fixed, non-per-item merge** — either
(a) **score-sum:** `sign(vote_CLIP + vote_LoRA)`, or (b) **rank-merge/union:** one top-20 list interleaved
from both spaces' neighbours. (A single kNN over a literal union is ill-defined — LoRA-query vs CLIP-key
cosine has no shared basis — so both realisations reduce to two per-space reads then a merge.)

**Non-isomorphism vs exact ban language — FAILS.** The score-sum variant is **literally inside the F47 ban
scope.** Quote: *"Do NOT re-propose per-item selectors over frozen channels regardless of feature family or
nonlinearity **unless the selector input is a genuinely NEW information source not derivable from banked
features/votes.**"* Proof it is a selector: on **agreement** items `sign(vote_CLIP)=sign(vote_LoRA)`, so
`sign(sum)` = that shared prediction = best-single; on **disagreement** items the signs differ, so
`sign(sum)` = the sign of the **larger-magnitude vote** = it *selects that arm's prediction*. So the merged
decision equals a per-item selection (the max-|vote| rule) on exactly the disagreement set, and its input =
the two banked votes = **not a new information source**. F47 already measured this class: the label-oracle
selection headroom is **+0.1083 (MHC-EN) / +0.0498 (HateMM)** but the realizable router (GBM+linear over
vote margins, purity, sub-votes, **confidence differential** `|vote_CLIP|−|vote_Qwen|` — which subsumes the
max-confidence rule) delivers **+0.0000** deployable and **−0.0458** at the dev-CV ceiling, below the perm
null. The changed-decision set for *any* fixed vote-combination is a **subset of the disagreement set**, so
the +0.1083 selection oracle is a strict **upper bound** — and even that unrealizable ceiling is unreached.
- The **rank-merge/union** variant is the W2-E ban: quote — *"zero-training unsupervised reorganisation of
  frozen pooled features as accuracy lever (same meta-family as C3geo)."* Interleaving two frozen spaces'
  neighbours into one memory with a fixed rule is exactly that.
- Adding a tuned weight `w·vote_CLIP+(1−w)·vote_LoRA` is still a per-item selector parameterised by a global
  scalar (input still banked votes) → still F47, and the global-w tune is B5-flavoured.

**Expected effect.** On HateMM/ZH the LoRA arm already passes; summing in the weaker CLIP arm **dilutes**
the win. On EN both arms are weak and the core is label-limited (F44) — the disagreement set is near-noise
("no coherent subgroup," net −1). Expected gain **≤ 0** on all three; the one goal-relevant dataset (EN) is
oracle-bounded by an unrealizable +0.1083.

**Verdict: REFUTED at inspection.** Score-sum = a max-confidence per-item selector over banked votes with
no new information source (F47 ban text, verbatim); rank-merge = W2-E reorganisation. Not a genuinely
uncovered cell — **no $0 gate warranted** (unlike GIR's nonlinear-residual case, the changed-decision set
is *provably* the disagreement set F47 already exhausted). This is the surface that most resembles a lever
and is the most decisively closed.

### Surface 4 — Per-stream decomposition of the PASSING HateMM-LoRA cell → **GENUINELY UNRUN, but DIAGNOSTIC**

**Mechanism / what is unrun.** F45 decomposed the *ZH* passing cell (gain lives entirely in the text
stream, Pareto). The analysis chapter §3.9 then **asserts** the HateMM mechanism by *inference from
F44/F45*: "HateMM decides on the image stream (image-only AUC 0.826), which LoRA leaves intact, so LoRA
inherits and preserves frozen-Qwen's image-borne Pareto conversion; the text stream it sharpens is
secondary." This is **reasoned, not measured** — an **F45-style per-stream AUC decomposition of the actual
HateMM-LoRA caches was never run.** It matters *now* because **F54 corrected** the "LoRA moves text only"
premise: `img_feats` pass through the LoRA-adapted LLM backbone (`lora_target: all`), so the image stream is
**architecturally movable**; whether the HateMM SFT target actually *moved* it is an open empirical question
the paper currently answers by assumption.

**Non-isomorphism.** It is not a lever and not banned — it is a **diagnostic** (the same class as F44/F45,
both banked and paper-cited). It touches no new axis.

**Expected effect (honest).** It does **not** reopen the goal (HateMM already passes; the finding is
explanatory). Its payoff is three-fold: (1) it converts a **live paper claim** (§3.9 image-inherited) from
inference to measurement — or **corrects** it if the HateMM SFT *did* move the image stream (a genuinely
new finding about where the pass lives); (2) it is the **only** evidence that would move
**premise-(a)**'s (transcript-dropout / vision-obligatory SFT, F54/TIE-recon) prior off its armchair ~5–8%
— if LoRA demonstrably moves the image stream on the passing cell, a vision-targeted SFT on EN's collapsed
image stream becomes less hopeless (though F50/F55 still price EN's *healthy* image ceiling below the bar,
so this stays low); (3) it closes the KS-2 honesty flag with data rather than a band-check.

**Gate design ($0, feasible — confirmed).** Reuse `scripts/analysis/encoder_swap_geometry.py` +
`b3_zh_lora_trainlog_parse.py` verbatim on the banked HateMM caches:
`data/CLIP_Embedding/HateMM/{train,dev_seen}_{openai_clip-vit-large-patch14-336_HF,
Qwen2.5-VL-7B-Instruct_HF, Qwen2.5-VL-7B-Instruct-LoRA_HF}.pt` — **all present** (LoRA train/dev = Jul-18
job-13234, verified by `ls`). Per-stream train-LOO + held-out-dev image/text/concat AUC for CLIP →
frozen-Qwen → LoRA-Qwen; per-class recall (Pareto-vs-rotation) read from the already-banked
`enc3s_HateMM_*-LoRA_HF_seed*_13235.trainlog` `Test_Retrieval` lines (same provenance discipline as
F45/B3_VERDICT — **no new test evaluation**). CPU, minutes, $0. Zero GPU/Modal/test-touch.

**Kill bars (it's a diagnostic, so "bars" = interpretation gates).** (i) machinery valid iff the concat
read-out reproduces the banked HateMM downstream sign (F44: dev Qwen−CLIP +0.047 → test PASS 3/3); (ii)
report image-stream ΔAUC(LoRA−frozen) with the F54 prediction (≈0 confirms §3.9; materially >0 = new
finding, "HateMM SFT moves the image stream"); (iii) Pareto-vs-rotation of the LoRA−CLIP per-class recall.
No performance claim; feeds the paper and premise-(a)'s prior only.

**Verdict: GENUINELY UNRUN, DIAGNOSTIC.** The one honestly-open $0 cell; recommended below.

### Own additions (checked, none survive as goal-relevant)

- **topk / triplet-vs-InfoNCE on adapted features** — subset of surface 1 (generic tuning, tactics,
  D7-dead).
- **Updatable / editable memory as an accuracy lever** — Axis I closed: archive-as-key claims WITHDRAWN
  (selection artifacts), AUTO two-vote repair NEGATIVE (`C−A=0`, guard-rail only); adding test-time items =
  transductive leakage. Covered.
- **Multiple SFT draws to de-marginalise ZH** — pre-declared out-of-scope stability check (F0.2), and it
  addresses a D7-blocked (not performance-blocked) leg. Not a premise.

---

## RANKING

| # | surface | verdict | goal-relevant? | non-iso vs exact ban | $0 gate? | prior |
|---|---|---|---|---|---|---|
| **4** | HateMM-LoRA per-stream decomposition | **genuinely unrun, diagnostic** | no (paper + prior-mover) | not a lever (diagnostic class, like F44/F45) | **yes, feasible** | n/a (diagnostic) |
| 3 | hybrid CLIP+LoRA memory (score-sum / rank-merge) | **refuted at inspection** | would be EN, but oracle-bounded | **FAILS** — literally F47 ban (max-|vote| selector, banked input) + W2-E | no (banned) | ≈0 |
| 1 | head architecture / objective on adapted feats | out of scope | no (not MLLM / not novel) | coupled = P9b/F51; uncoupled = generic tuning | no | ≈0 (EN label-limited) |
| 2 | within-seed epoch-window averaging | out of scope | no (protocol, non-binding leg) | spirit-iso to cross-seed ensemble veto; B5-class | no | n/a |

---

## SINGLE RECOMMENDED ACTION

**Run the $0 per-stream decomposition of the passing HateMM-LoRA cell (surface 4).** It is the only
honestly-unrun cell, it is cheap and feasible on banked train/dev caches (verified present), it hardens or
corrects a live analysis-chapter claim (§3.9 "HateMM gain is image-inherited," currently *reasoned*), it
empirically adjudicates the F54 "image is architecturally movable" correction on the cell where it matters
most, and it is the sole prior-mover for premise-(a) — the one remaining unpriced SFT variant. It does
**not** reopen the goal; treat it as a paper/prior diagnostic, not a conversion lever.

**Do NOT** spend on surfaces 1–3: two are out-of-scope for the novel-MLLM clause (and one of those is
spirit-isomorphic to the ensemble veto), and the third is literally inside the F47/W2-E bans.

---

## WHAT THIS IMPLIES

The winning F46/F48/F54 pattern **does not repeat here**: those premises each hid a *goal-relevant,
uncovered, cheap-to-measure* cell. Wave-6's four surfaces contain **no such cell** — the two live
possibilities are out-of-scope for the goal's novel-MLLM clause (head-recipe, epoch-ensemble), the one that
points at EN is exactly the F47 ban (a max-confidence selector over banked votes is not a new information
source), and the one genuinely-unrun cell is diagnostic. **The box is therefore empty at the premise level
too**: inside the frozen constraint box, no unexamined premise yields a novel MLLM mechanism on ≥2 datasets.
This is consistent with — and sharpens — `TERMINUS_round3` and the F55/F56 closures: the performance
conjunct is met on 2 datasets under final-epoch by one **encoder-class** lever (F53), and the live decision
is the **D7 user ruling on generic LoRA** (a ruling, not a run, zero further GPU), plus the five terminus
relaxations. Wave-6 adds one honest $0 diagnostic to the queue (surface 4) and closes the premise-hunt
axis: the "box empty" claim survives an adversarial read of the prose.

---

## PROVENANCE

- Ledger: `state/directions_tried.json` (23–24 dead + bans + positives), `state/findings.jsonl` F44–F56.
- Exact ban quotes: F47 `ROUTER_GATE_RECORD.md` (30d0ee1) ban_scope; F50 `FA_GATE_RECORD.md` (e0877c9)
  ban_scope; W2-E `directions_tried.json` W2-E ban_scope; F51 `WAVE5_CANDIDATES.md` (7166232); B5
  `B5_VERDICT_REVIEW.md` (50f01b9); cross-seed-ensemble veto `directions_tried.json:banned_constraints`.
- Stated laws attacked: `research-wiki/DRAFT_analysis_chapter.md` §3.6–§3.9.
- Surface-4 feasibility: `ls data/CLIP_Embedding/HateMM/` — LoRA `train`/`dev_seen` caches present
  (Jul-18, job 13234); scripts `scripts/analysis/encoder_swap_geometry.py`,
  `b3_zh_lora_trainlog_parse.py`; banked test recall in `slurm/logs/enc3s_HateMM_*-LoRA_HF_seed*_13235.trainlog`.
- Head recipe (frozen campaign-wide): `ROUTER_GATE_RECORD.md` §1; `src/model/classifier.py:110–122`
  (align/Hadamard).
- **Required statements:** ZERO GPU / SLURM / Modal spent; no held-out test metric read or produced (only a
  filename `ls`, no `.pt` content opened); no `state/`, prereg, config, `research-wiki/`, or live ceremony
  artifact mutated. Committed on `main`, not pushed.
