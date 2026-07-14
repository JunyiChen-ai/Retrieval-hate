# C5 FORENSIC RECON — 7B relational contrastive representation distillation (CRD) into the head

**Agent:** round-3 C5 forensic recon (ZERO GPU / ZERO SLURM / ZERO Modal; pure code + cache + doc reading).
**Date:** 2026-07-15. **Repo HEAD at recon:** `b143834`.
**Target:** the LAST un-recon'd round-3 queue candidate — **C5 "7B-CRD"** (scout prior **LOW**), pre-recon to
complete the ammo pool.
**Candidate source:** `research-wiki/ROUND3_NOVELTY_CANDIDATES_2026-07-14.md` §C5 (lines 250-261, ranked #5/5,
prior LOW, novelty LOW); descends from `research-wiki/LITERATURE_mllm_integration_2026-07-13.md` **§C4** (CRD)
and is re-affirmed OPEN by `refine-logs/EXHAUSTION_AUDIT_2026-07-14.md` §3(a) / §5 cell #4.
**Ground-truth code read (not memory):** `src/model/loss.py:438-455` (triplet-margin+BCE hybrid, P5 additive-
negative precedent), `src/model/evaluate_rac.py:75-155` (head-projected memory build → `IndexFlatIP`),
`src/utils/metrics.py:262-320` (rank-weighted signed-cosine top-20 vote), `src/utils/retrieval.py` mining loop
(per `C3GEO_FORENSIC_RECON.md` §1). Caches `torch.load`-verified on disk (§D).

---

## VERDICT LINE

> **NO-GO — kill at recon, do NOT open a probe-design cycle.** C5 fails on **both** gates and its oracle
> ceiling is **already measured**, so the kill is pre-GPU and decisive:
> **(1) Novelty gate (D7-tightened): FAIL** — the mechanism is textbook feature/relational KD (CRD, ICLR
> 2020), and the MLLM contribution it distils is *exactly the frozen-Qwen encoder-swap geometry*, which D7
> already ruled non-novel. Same failure class as **R3-C3geo** (killed pre-ceremony) and **W2-E** — C5 is
> C3geo's explicitly-named sibling (`C3GEO_FORENSIC_RECON.md` §5: "the relational-ordering distillation add …
> is closer to C5 … and inherits C5's objection").
> **(2) Performance gate: near-zero prior** — a student can never exceed its teacher's geometry, and here
> **teacher = student-scale (7B)** so there is **no distillation gap** (CRD's entire value is big→small). The
> teacher-as-key oracle is the encoder-swap / B1 result, **already banked**: HateMM +5.3–5.6 (novelty-dead),
> **MHC-EN FAIL, MHC-ZH −0.0112 FAIL**. C5's *ceiling* is therefore a novelty-dead HateMM-only pass, which
> fails the ≥2-dataset goal. No probe can beat a ceiling that is itself dead on 2/3 datasets.
> Cheap-to-run ($0, fully banked — §D) does **not** rescue a double-gate failure.

---

## 0. NAMING-COLLISION DISAMBIGUATION (do this first — it changes the ban analysis)

There are **two different "C5"s** in the state files; conflating them mis-reads the ban surface:

| label | source doc | mechanism | status |
|---|---|---|---|
| **C4 (old)** | `LITERATURE_…2026-07-13.md` §C4 | **CRD**: distil MLLM embedding *geometry* into small head/LoRA-student | deferred "for lack of a 72B teacher," **NOT killed** |
| **C5 (old)** | `LITERATURE_…2026-07-13.md` §C5 | **external unlabeled video + MLLM pseudo-labels**, representation-training only | **BANNED by the data veto** (`directions_tried.json:127`: "conservatively also bans external unlabeled-pool training (C5)") |
| **C5 (round-3, MY TARGET)** | `ROUND3_NOVELTY_…2026-07-14.md` §C5 | **7B-teacher CRD**: distil frozen Qwen-7B *relational geometry* into the head | = the OLD **C4**, re-scoped to a 7B teacher; **OPEN** per `EXHAUSTION_AUDIT` §3(a) |

**Consequence:** my target (round-3 C5 = 7B-CRD) is **the old C4**, and the "(C5)" the training-data veto
bans is the *external-unlabeled-pool* old-C5 — a **different** candidate. So **the target C5 is NOT closed by
the data veto**; it is a genuinely open cell (exhaustion audit §3(a): "technically an open cell not covered by
any epitaph or ban … It is low-prior but it is not closed"). This recon is therefore load-bearing: it is the
document that closes the cell the audit left explicitly open. It closes it on the **novelty + performance**
gates, not on the data veto.

---

## A. MECHANISM — what C5 actually proposes

**The one coherent construction.** C5 adds a **relational/contrastive KD loss term** to the head's training so
the head's projected retrieval geometry **matches Qwen-7B's pooled pairwise-similarity structure**.

- **Teacher signal = banked pooled Qwen-7B vectors ($0, NO new forward).** The 3584-d pooled Qwen features are
  already on disk for all three datasets (§D). The teacher geometry is the matrix of pairwise cosines among
  those frozen vectors. No forward pass is required — this is a CPU load, exactly as C3geo's pooled-distance
  mining needed no GPU (`C3GEO_FORENSIC_RECON.md` §4).
- **Student = the small alignment head.** For the mechanism to be non-vacuous the student head must consume a
  *different, cheaper* input than the teacher — i.e. **head over CLIP (or base) features, distilling Qwen
  geometry** → the only coherent story is *"get Qwen-quality retrieval geometry at CLIP cost"* (efficiency, not
  new accuracy). A head over *Qwen* features distilling *Qwen* geometry is near-circular (the student already
  has the teacher's representation), so that variant collapses into encoder-swap.
- **Exact loss.** CRD (Tian et al., ICLR 2020, arXiv 1910.10699, "feature-KD > logit-KD"), or a lighter
  **relational KD** on pairwise similarities (RKD-style), or feature-map L2. Per §C5 text: "representation-
  level, no logits/scores."

**Slot & compatibility with the live loss (verified code).** The training loss is a **triplet-margin + BCE
hybrid** — `loss.py:453-455`:
`total_loss = torch.mean(torch.relu(in_batch_loss + hard_loss - pseudo_gold_loss + args.triplet_margin))`
(margin default 0.1). A CRD term slots in exactly as **P5's counterfactual add** does — `loss.py:443-448`
already shows the pattern (`hard_loss = hard_loss + cf_sim`, an additive extra term gated by a flag,
byte-identical no-op when off). So a `+ λ·crd_loss` term is **mechanically feasible and gated-clean**. This is
a **D2 representation-level lever** (the favoured class — the head's learned projection geometry is what CRD
reshapes), and it is **train-split-only** (teacher geometry computed over own train∪dev, no external pool).

**Verdict A:** mechanism = "add a relational/contrastive KD term that regularises the head's projection toward
frozen-Qwen-7B pooled geometry; teacher banked ($0); student = head over cheaper features." Loss-compatible.
The mechanism is real and legal — but see B/C: what it distils is a *known dead-on-2-datasets* geometry.

---

## B. COLLISION AUDIT vs the dead list (brutal)

- **(i) vs B3 LoRA (`positives_bank[B3-lora-zh]`, marginal ZH pass).** **Distinct injection point.** B3
  *adapts the Qwen encoder* (LoRA weights change `image_feats`/`text_feats`); C5 keeps **all encoders frozen**
  and changes only the **head's training signal** (adds a CRD term). Non-isomorphic. **BUT B3's finding is a
  headwind, not a tailwind:** B3 measured that the marginal ZH gain "= LoRA **adaptation** not encoder
  identity," and that **frozen-Qwen ZH = −0.0112** (B1, dead route #20). C5 distils the **frozen** Qwen
  geometry — precisely the geometry B1 proved fails ZH and B3 proved only helps *after adaptation C5 does not
  do*. C5 sits on the wrong side of B3's lesson.
- **(ii) vs P4 schema-distill (dead: "redundant; formalized by AAAI25 2412.11917 ensembling").** **NOT
  isomorphic — C5 survives this collision.** P4 distilled **MLLM schema/description *text* fields** (generated
  natural-language attributes); the AAAI-25 result shows that gain is mostly semantic-agnostic ensembling. C5
  distils **embedding *geometry* (pairwise cosines)**, not generated text — a different object entirely. C5
  does **not** die at P4. (This is the one collision C5 passes cleanly.)
- **(iii) vs P11 "MLLM-scores-as-training-signal" ban.** **Escapes the letter, not the spirit-of-redundancy.**
  The ban is literally "MLLM *scores/labels* as training signal"; C5 uses **pairwise representation geometry**,
  no score/label/logit — and both `EXHAUSTION_AUDIT` §3(a) and `C3GEO_FORENSIC_RECON` §2(a) confirm geometry is
  *outside* the ban's letter. **However** the frozen-Qwen pooled geometry **IS the encoder-swap representation**
  (`positives_bank[encoder-swap]`); distilling it imports the encoder-swap signal through the training-loss back
  door — a signal characterised as **dead-on-MHC / redundant-on-HateMM**. Compliant on the letter; the source
  geometry is a known-quantity whose convertible structure is already mapped (identical caveat to C3geo).
- **(iv) vs encoder-swap D7 ruling — the dispositive collision.** C5's novelty claim would be "cross-modal
  relational distillation of an MLLM's geometry into a retrieval memory." Is the MLLM **load-bearing**? The
  *distillation target* is MLLM-derived (so nominally yes), but the **target geometry = the encoder-swap
  geometry D7 already ruled non-novel**, and the **operator (CRD/RKD) is a textbook generic technique**. So the
  novel content reduces to "point standard feature-KD at frozen-Qwen geometry in-domain" — the exact
  generic-trick-in-a-new-domain pattern D7 excludes. This is **isomorphic in outcome-class to R3-C3geo**, which
  forensic recon killed pre-ceremony on this very gate. C3geo's own recon (§5) pre-declared C5 as its sibling:
  *"the relational-ordering distillation add … is closer to C5 (7B relational CRD, priced LOW) … and inherits
  C5's objection: a 7B-teacher relational signal cannot exceed using the 7B encoder directly, which already
  fails to convert on EN/ZH."*

**Verdict B:** C5 is **formally non-isomorphic to every banned id** (not P4, not P11-letter, not B3-slot, not
the old-C5 data-ban) — it is *legal to run*. But it is **outcome-class-isomorphic to C3geo** (frozen-Qwen
geometry as head training signal, D2 class, escapes the score-ban letter, spirit = encoder-swap back door), and
C3geo was killed pre-ceremony. C5 does **not** die at P4; it dies at the **encoder-swap/D7 collision (iv)**,
with C3geo as the binding precedent.

---

## C. HEADROOM LOGIC — the oracle ceiling is ALREADY MEASURED (pre-GPU kill)

The question C poses: if Qwen features already feed the pipeline directly (encoder-swap), what does *distilling*
them into a CLIP-side head add? Answer: **at most a lossy approximation of the teacher, never more.** This gives
a hard, non-vacuous, **already-banked** oracle.

**The oracle for a training-signal distillation = teacher-as-key retrieval = the encoder-swap result.** A
student head distilling Qwen geometry cannot beat *using the Qwen geometry directly as the retrieval key*. That
number is measured (`positives_bank`, dead routes #20/#21):

| dataset | teacher-as-key (frozen-Qwen encoder-swap) = C5's ceiling | goal (≥+0.03/≥+0.03, 3/3, both protocols) |
|---|---|---|
| HateMM | **+5.3–5.6 acc, 3/3, both protocols** | clears — but **D7-DEAD for novelty** (encoder-swap) |
| MHC-EN | **FAIL** (encoder-swap fails EN; SAV falsified the dilution repair) | fail |
| MHC-ZH | **−0.0112 acc, 1/3 sign (B1 #20)** | fail |

So **C5's best conceivable outcome ≤ a novelty-dead HateMM-only pass**, which fails the goal's **≥2-dataset**
requirement outright. There is nothing to convert on the two datasets the goal binds on, because the teacher
geometry itself does not convert there.

**No teacher-student gap.** CRD's documented value (CRD ICLR-2020; and `LITERATURE §C4`'s own cautionary cite
arXiv 2511.17886 "larger teacher ≠ better student") is a **big→small** transfer. Here **teacher = Qwen-7B,
student = a head derivable from ≤7B features** — same scale, no capacity gap for the student to inherit. The
distillation can at best *replicate* the teacher (worse, since KD is lossy), never *exceed* it.

**Does concat/fusion already capture any "fuse both geometries" story?** The only other coherent C5 rationale
would be "CLIP⊕Qwen fused geometry beats either alone." Checked: **Qwen⊕CLIP embedding concat is UNTESTED**
(`EXHAUSTION_AUDIT` §4 row + §5 cell #3: "Untested; classic representation lever… inherits the ZH conversion
wall"). It is a *separate, cleaner* representation-fusion lever than C5 — if fusion is the goal, one would test
concat directly (cheap), not launder it through a KD loss. And the audit's honest prior on concat is
LOW-MODERATE precisely because "on HateMM concat≈Qwen (no new dataset); on ZH/EN both individually fail the
fixed threshold." So the fusion story does not rescue C5, and C5 is a strictly worse way to pursue it.

**Verdict C:** the redundancy is **by-construction**. C5's oracle ceiling = the banked encoder-swap/B1 numbers
= {HateMM-only positive (D7-dead), EN fail, ZH −0.0112}. A distillation cannot beat that ceiling; the ceiling
fails the goal. **Pre-GPU kill — the probe's result is bounded above by numbers we already have.**

---

## D. COST / CACHE — fully banked, $0, zero new GPU (verified on disk this session)

A CRD probe (teacher geometry from banked pooled Qwen; student head over CLIP/base features) needs **no new
extraction**. `torch.load` shapes verified today (`weights_only=False`):

| dataset | train pooled Qwen `{ids,img_feats,text_feats,labels}` | dev_seen | test_seen | student input (CLIP) |
|---|---|---|---|---|
| HateMM | img `(744,3584)` txt `(744,3584)` lab `(744,)` | present | present | `*_openai_clip-vit-large-patch14-336_HF.pt` present |
| MHC-EN | `(549,3584)` / `(549,3584)` / `(549,)` | present | present | present |
| MHC-ZH | `(579,3584)` / `(579,3584)` / `(579,)` | present | present | present |

- Teacher pairwise geometry = CPU cosine over banked `img/text_feats` (no forward). LoRA-Qwen and 32B pooled
  caches also present if a teacher-variant sweep were ever wanted (not needed — all fail the goal).
- Head-training is T4-scale (project-measured 33 s / 30 ep on Modal for this head) ⇒ a full CRD probe would be
  cloud-eligible, **features-only** (only `.pt` float vectors + label JSON leave disk; no raw video — CLAUDE.md
  data boundary respected).
- **This is the only axis on which C5 scores well** ($0, banked) — and, exactly as with W2-E, cheapness does
  **not** rescue a double-gate failure.

---

## E. KILL-BAR SKETCH (for completeness — NOT a recommendation to run)

Had C5 cleared recon, the house-discipline probe (reusing the S2S/W2-B/C3geo instrument verbatim) would be:

- **Sole primary arm:** paired 3-seed **same-seed** contrast — head trained with hybrid loss **vs** hybrid +
  CRD term — on identical banked features; `λ` and KD variant (CRD vs RKD vs L2) are **sensitivity-only, never
  survival-determining** (bar hyperparameter shopping, mirrors B2).
- **Conversion-taxed bars:** HateMM anchor Δacc **AND** ΔmF1 ≥ **+0.05** vs the same-seed hybrid baseline;
  MHC-EN / MHC-ZH survival Δ ≥ **+0.03/+0.03** (P3 shrinkage tax).
- **Oracle arm (the decisive one, and it is ALREADY EVALUATED):** teacher-as-key retrieval = run kNN on the raw
  Qwen teacher geometry = **the encoder-swap/B1 result**. This upper-bounds any distillation. Because that
  oracle is **already banked as {HateMM-only pass, EN fail, ZH −0.0112}**, the mandated oracle kill-switch
  ("DEAD iff oracle Δ < +0.03 on every goal-binding dataset") **fires now, pre-probe**: EN and ZH oracle Δ are
  ≤ 0. This is why the kill is pre-GPU — the oracle need not be re-measured.
- **Calibration/validity guards (house rules):** Fano label-oracle arm ≥ 0.99 else instrument void; permutation
  null ≥ 100 seeds (same-perm across arms); bootstrap 1000, 5th-pct > 0; fail-closed on `test_seen`; assert
  memory counts 851/629/657 (train∪dev).
- **Pre-declared MHC-EN generalisation trap:** even a *teacher-matching* HateMM result would be **HateMM-only**
  = fails the ≥2-dataset goal (the identical trap that caps encoder-swap). Do not let a HateMM-only number read
  as a pass.
- **Fano n/a as a novelty test** — Fano only validates the probe instrument here, not the D7 gate (which is the
  binding failure and is a user ruling, not a measurement).

---

## F. PRIOR & DISPOSITION

**Prior: LOW → recon lowers to near-zero.** Conditioned on:
- **D2-favoured class (+):** distillation *is* a representation-level lever (the only class that ever cleared
  +3). This is C5's single structural merit — **but it is nullified** because the specific geometry distilled is
  the *measured* encoder-swap geometry, dead on the 2 goal-binding datasets. D2 credit requires a *new* geometry;
  C5's is a known dead one.
- **Teacher directly usable as encoder (−−):** the redundancy is by-construction (§C); student ≤ teacher, and
  teacher = student-scale ⇒ no gap. This is the decisive performance minus.
- **B3 / encoder-3seed asymmetric dataset map (−):** frozen-Qwen (what C5 distils) is HateMM-only positive, EN
  fail, ZH −0.0112. The asymmetry does not help C5 — it caps it.
- **D3 noise floor at 3 seeds (−):** any HateMM-only Δ near the teacher would still sit against the ±1-2pt floor
  on 549-744-sample train sets.

**P(C5 converts on ≥2 datasets) ≈ 0.** For ≥2-dataset conversion the teacher itself must carry ≥2 datasets; it
carries **≤1** (HateMM only), and the student cannot exceed the teacher. Even the intra-dataset event
"C5 > teacher-as-encoder on HateMM" has probability ≈ 0 (lossy KD, no scale gap).

**Could C5 claim novelty even if it converted?** **No.** Best case is a **"performance clause only, novelty =
user ruling" B3-type outcome — and strictly worse than B3**, because (a) the operator is textbook CRD/RKD
(C3geo-class D7 fail, not the debatable encoder-*family* question B3 raised), and (b) the distilled geometry is
the encoder-swap signal D7 already ruled non-novel. So even a miracle conversion would not yield a novel
contribution — the exact ceremony-cost trap forensic recon exists to pre-empt (cf. W2-E, C3geo).

---

## RECOMMENDATION TO ORCHESTRATOR

1. **NO-GO — close C5 at recon.** Do not open prereg/review/freeze ceremony. Kill grounds are the **same two**
   that closed C3geo and W2-E, plus one C5-specific reinforcement: (i) **D7 novelty FAIL** (textbook CRD; MLLM
   contribution = the D7-dead encoder-swap geometry); (ii) **near-zero performance prior** with an **already-
   banked oracle ceiling** {HateMM-only / EN-fail / ZH −0.0112} that fails the ≥2-dataset goal; (iii) **teacher =
   student-scale ⇒ no distillation gap** (CRD's raison d'être absent).
2. **Update the open-cell ledger.** This recon closes the cell `EXHAUSTION_AUDIT` §3(a)/§5#4 left explicitly
   open ("7B-teacher CRD … not covered by any epitaph or ban"). It is now covered — by the **novelty gate +
   already-measured oracle**, not by the data veto (the data veto's "(C5)" is the *other*, external-pool C5 —
   §0). Suggested `directions_tried.json` epitaph: *"round-3 C5 (= old-C4 7B-CRD): killed pre-ceremony by
   forensic recon; textbook feature/relational KD fails D7; teacher = student-scale ⇒ no gap; oracle ceiling =
   encoder-swap/B1 (HateMM-only pass, EN fail, ZH −0.0112) already fails ≥2-dataset goal; 0 GPU."*
3. **The one legitimate future for C5's relational-distillation idea** is the same as C3geo's: only if it rides a
   *genuinely novel object* (e.g. S2S frame-set geometry), where the novelty lives in the object/operator, not
   the KD. As a **pooled-vector, same-scale standalone it has no path** to either gate.
4. **Ammo-pool status:** with C5 closed, the round-3 C-line pooled-vector training-objective cells (C3geo, C5)
   are both retired at recon; the live representation-level leads remain **S2S / C2-mem / W2-A** (the frame-set /
   memory / cross-modal-grounded-key routes that carry their own novelty object).

---

## PROVENANCE

- Candidate specs: `research-wiki/ROUND3_NOVELTY_CANDIDATES_2026-07-14.md` §C5 (250-261), ranking (267-273);
  `research-wiki/LITERATURE_mllm_integration_2026-07-13.md` §C4 (CRD, 27-30) + §C5 (external-pool, 31-34).
- Open-cell status + naming collision: `refine-logs/EXHAUSTION_AUDIT_2026-07-14.md` §3(a) (192-207), §4 row
  (220), §5 cell #4 (238-240).
- Sibling kill (binding precedent): `refine-logs/C3GEO_FORENSIC_RECON.md` §2(a), §5 (relational-distill →
  "closer to C5"), verdict; `refine-logs/W2E_FORENSIC_RECON.md` (zero-new-signal / ceremony-cost kill class).
- Live loss + additive-term precedent: `src/model/loss.py:438-455` (triplet-margin+BCE hybrid; P5 `cf_negs`
  additive pattern). Memory build / vote: `src/model/evaluate_rac.py:75-155`, `src/utils/metrics.py:262-320`.
  Mining loop (frozen-Qwen-geometry-as-training-signal reference): `C3GEO_FORENSIC_RECON.md` §1
  (`src/utils/retrieval.py:314-584`).
- Oracle ceiling numbers: `autoresearch/goal_mllm_plus3/state/directions_tried.json` `positives_bank`
  (encoder-swap +5.3–5.6 HateMM, FAILS MHC-EN; B3-lora-zh marginal), dead #20 `B1-qwen-encoder-zh`
  (final −0.0112 acc, 1/3 sign), #21 `B2-32b-encoder` (scale regresses), #22 `B4-lora-en`, `R3-C3geo`; bans +
  `diagnosis_frame` (D1/D2/D3); `banned_constraints[]` incl. data veto "(C5)" = external-pool old-C5.
- Cache shapes verified `torch.load` this session: HateMM 744×3584, MHC 549×3584, MHC_zh 579×3584 (img+text
  pooled Qwen-7B), dev/test + CLIP + LoRA + 32B pooled caches present; `data/CLIP_Embedding/{HateMM,MHC,MHC_zh}/`.
- P4 non-collision anchor: `LITERATURE_mllm_integration_2026-07-13.md:23` (AAAI-25 2412.11917 = schema-text
  ensembling, distinct object from embedding geometry).
- Zero GPU / SLURM / Modal used. Repo HEAD `b143834`.
