# W2-C Forensic Recon — Temporal-order / escalation-aware alignment kernel

**Agent:** zero-GPU / zero-SLURM / zero-Modal forensic recon (pure code + cache + doc reading; one
read-only `torch.load` cache-shape/descriptive pass, no vote / no probe / no training).
**Target:** wave-2 candidate **W2-C** ("temporal-order / escalation-aware alignment kernel", temporal-order
axis, rides the S2S frameset). Ranked #2/5 in `ROUND3_CANDIDATES_WAVE2_2026-07-15.md`, prior MODEST–LOW.
**Nature of this recon:** **PRE-recon.** W2-C's probe rides the S2S Stage-E per-frame frameset
(`frameset_qwen7b_8f/`, T=4 groups from 8 frames) which **does not exist yet** — job **13159 `s2s_extr`
is `PENDING (JobHeldUser)`** (verified `squeue`), and `data/CLIP_Embedding/{HateMM,MHC}/frameset_qwen7b_8f/`
is **ABSENT**. Goal: land the recon now so probe design can start the day the frameset lands.
**Conditioning kills read in full:** `W2B_VERDICT_REVIEW.md` (W2-B KILLED, outcome (d) NEGATIVE),
`W2E_FORENSIC_RECON.md` (W2-E NO-GO at recon, D7 meta-family).
**Ground truth read (not memory):** `src/utils/metrics.py:262-320` (rank-weighted signed-cosine top-20
vote; sim is a multiplicative WEIGHT, `:268-270`), `src/model/evaluate_rac.py:80-155` (memory =
`IndexFlatIP`, one L2-normed head-projected vector per train video), `scripts/analysis/w2b_probe.py`
(1-795, the LOO-kNN + oracle + Fano + perm-null + bootstrap machinery W2-C reuses verbatim),
`refine-logs/S2S_PROBE_DESIGN.md` + `S2S_PREREG_REVIEW.md` (the frameset spec W2-C consumes),
`autoresearch/goal_mllm_plus3/state/directions_tried.json` (26 dead + 9 bans). Repo HEAD at recon =
`954b0cb`.

---

## VERDICT LINE

> **CONDITIONAL — GO-TO-PROBE-DESIGN (as an ADDED KERNEL ARM inside the S2S Stage-P probe), WHEN THE S2S
> FRAMESET LANDS.** W2-C is a genuine D2 (representation-geometry / retrieval-metric) candidate that
> **escapes** the W2-E/C3geo "zero-new-bits reorganisation" meta-family — temporal ORDER is a
> mathematically distinct function of the frozen frame vectors that both POOLED (order-invariant) and
> S2S-MeanMaxSim (permutation-invariant) destroy, so W2-C reads genuinely new bits (this escape argument
> is **structurally sound**, unlike W2-E's). It is therefore **not** a recon-kill. But it is **not** an
> unconditional GO either: (i) it is frameset-blocked; (ii) the cheap-encoder signal-prerequisite
> forensics are **weak-to-negative** (§C: on frozen CLIP the HateMM *hateful* class is MORE static than
> non-hateful, and dynamics magnitude barely separates classes on either dataset — the escalation-arc
> premise is unsupported at the CLIP level, CLIP<Qwen caveat noted); (iii) at T=4 (3 transitions) the
> kernel is thin, the T=8 (16-frame) arm is the only meaningful test. **Conditions, explicit:** run W2-C
> ONLY as a pre-declared added arm in the S2S probe (near-zero marginal cost), never a separate job;
> sole-primary = Δ(order-kernel − S2S-MeanMaxSim) on the **T=8** arm, gated by a mechanism-specific
> **order-shuffle null**; prior is conditional on S2S (§F). **Recommended companion:** a **zero-GPU
> CLIP-K4 order-signal pre-check runnable TODAY** on the already-banked `subclipK4` caches (order-aware
> kernel + order-within-video shuffle null) to sharpen the prior before the frameset even lands — CLIP<Qwen
> caveat inherited, but combined with the static-hateful-class forensics it is a cheap prior-mover.

**One-line prior:** MODEST–LOW, recon-**lowered** toward LOW by the CLIP temporal-variation forensics, but
NOT to a recon-kill because order is a real new-bit source (D1-clean) and the marginal cost of the arm is ~0.

---

## A. MECHANISM — the sharpest W2-C variant, and where it slots

**Baseline object (verified).** S2S extracts, per video, an **ordered sequence** `{g_1 .. g_T}` of
temporal frame-group vectors (`S2S_PROBE_DESIGN.md:120-137`; T=4 for 8 frames, temporal-major, each `g_t`
= mean of the vision tokens in temporal group t, proven contiguous by
`modeling_qwen2_5_vl.py:466-505,560-562`). The frameset `.pt` stores `g` fp16 `[N,T,3584]` **in temporal
order** (§6 storage contract). S2S's primary metric, MeanMaxSim, is
`(1/|Q|) Σ_q max_m cos(ĝ^Q_q, ĝ^M_m)` — a mean of maxes that is **invariant to any permutation of the
query frames AND of the memory frames**. POOLED (`cos(mean_t g^Q_t, mean_t g^M_t)`) is likewise
order-invariant. **Both baselines discard the order axis by construction.**

**The sharpest W2-C variant.** Keep S2S's ordered `{g_1..g_T}` set; replace the order-free score with an
**order-constrained similarity kernel**, in two concrete forms to pre-declare (not shop):
1. **Soft alignment (monotonic OTAM / soft-DTW).** Cross-video cost matrix `C[q,m] = 1 − cos(ĝ^Q_q,
   ĝ^M_m)`; score = the soft-min-cost **monotonic** warping path through `C` (order must be respected:
   only forward/diagonal moves). Two videos match when they share the same benign→escalation→payload
   *trajectory*, not merely one frame. **Crucially, the alignment path is data-driven — NEVER
   label-guided** (see §E leak warning).
2. **Transition-set kernel.** Represent each video by its **first-difference set** `{g_{t+1}−g_t}_{t=1..T-1}`
   (narrative *turns*), then MeanMaxSim over the transition set. This is the cheapest order-carrier: it is
   invariant to a global shift but sensitive to the *direction* of change, and `Δ = g_{t+1}−g_t` flips sign
   under sequence reversal, so it is genuinely order-sensitive.

**Injection point + bandwidth.** Retrieval **metric** over the frame set — **representation-geometry
(D2)**, the only class that ever cleared +3. Same bandwidth as S2S (T vectors/video); it adds an order
constraint, **no decision-side scalar** (so D1 does not bite — see §F). It slots exactly where S2S's
MeanMaxSim slots: as the pairwise `retrieved_scores` fed to the pipeline's real vote
(`metrics.py:262-320`), through the unchanged top-20 LOO machinery. **NOT** a vote re-weighting and **NOT**
a decision-side signal (either would be D1 death, per the diagnosis frame).

**Zero-training vs trained-head.** The pre-declarable version is **zero-training** (a metric swap on frozen
frames, exactly like S2S/W2-B). It **escapes the W2-E/C3geo meta-family ban** — and this is the decisive
mechanism fact, tested honestly below.

**Escape argument, stress-tested.** W2-E was killed at recon because its k-means/prototype reorganisation
is "a deterministic **lossy** function of the *same pooled vector*… **zero** new bits" — a *coarsening*
that can only remove neighbours. W2-C's claim is the opposite: order is a **strictly richer** function of
the same frozen frame vectors that *both* baselines are invariant to. Is that true? **Yes, structurally.**
POOLED is `f(Σ_t g_t)`; MeanMaxSim is a symmetric function invariant under `S_T × S_T` permutations of
the frame indices; an order-constrained kernel (monotonic warp / signed transition set) is **not**
invariant to those permutations — it reads the sequence, which is a distinct coordinate the two baselines
project out. So W2-C is **not** "reorganisation of frozen pooled features with no new information" (the
W2-E/C3geo family); it is a **new order-sensitive metric over the frame SET** (the S2S/W2-B family, D2),
explicitly **not** killed by W2-B ("W2-C not killed — only prior moves", `W2B_VERDICT_REVIEW.md §E`). The
escape is **sound**. The honest residue: escaping W2-E's *no-new-bits* gate is a **different** gate from
clearing D7-tightened *novelty* (§B) and from the *signal-prerequisite* (§C) — W2-C clears the first and
faces the second and third. That is the whole game: W2-C's risk is **signal / thinness / conversion**, NOT
novelty-gate death — a materially better position than W2-E.

---

## B. NOVELTY vs D7-tightened — is the MLLM adjective load-bearing? (brutal)

**Standalone: FAIL D7-tightened (transfer class, same as S2S).** Order-aware video kernels are a large,
mature literature — OTAM (CVPR-2020), soft-DTW, DeepEMD, CMOT/TRX/HyRSM for few-shot action recognition;
and CVPR-2025 "Temporal Alignment-Free Video Matching" explicitly argues order is *sometimes unnecessary*.
Raw novelty is the **same transfer-composite class as S2S** — not novel standalone.

**Composite: the defensible framing** — "**MLLM per-frame semantics × temporal-order kernel for
retrieval-augmented hate detection**": first order-/escalation-aware retrieval for hateful-video, motivated
by hate's reveal structure. In-domain, "Revealing Temporal Label Noise in Multimodal Hateful Video"
(2508.04900) confirms temporal structure is a live concern — but for *label noise*, not retrieval
geometry — so the retrieval-geometry angle is genuinely untouched in-domain.

**Does the MLLM adjective do real work here (unlike W2-E)?** This is the key discriminator, and the
answer is **plausibly yes, but UNPROVEN — and it leans on the same untested S2S premise.** Argument: Qwen
per-frame vectors are **instruction-conditioned semantic** representations, so a transition `g_{t+1}−g_t`
in Qwen space is a **semantic/affective turn** (topic shift, "benign scene → slur"), whereas the same
transition in frozen CLIP space is an **appearance** turn. For a benign→hateful *reveal*, a semantic-order
detector plausibly benefits from semantic frame vectors more than appearance ones. **This is a real
distinction from W2-E** (where "MLLM" was inert because k-means is byte-identical over any encoder). BUT —
brutally — it is (i) an untested plausibility, not a demonstrated mechanism, structurally identical to
S2S's own "first set-to-set over MLLM frame tokens" composite; (ii) partially **testable on CLIP today**
(§C) — if order already carries label signal on CLIP, the MLLM adjective is not *uniquely* load-bearing;
if order carries nothing on CLIP and the escape hangs entirely on "but Qwen semantic order would," that is
an untestable-until-frameset article of faith. Net: the MLLM adjective is **less inert than W2-E's, more
load-bearing than a bolt-on, but not demonstrated** — and whether the composite clears the user's novelty
clause is the **same pending D7 user ruling as S2S/B3**, not decidable here.

**Non-isomorphism (graveyard).** vs S2S: order IS the mechanism (S2S is explicitly order-free and *defers*
OT/temporal alignment as a later arm) — distinct hypothesis (escalation trajectory vs shared segment). vs
TARC (dead): TARC conditioned the retrieval *graph* on a predicted target (decision-side ~3 bits); W2-C
changes the pairwise *geometry* to be order-aware, no target, no graph conditioning. vs W2-B (killed):
different mechanism class (order-aware kernel ≠ MeanMaxSim/Chamfer/ASYM) — W2-B's ban is literally scoped
to "MeanMaxSim / Chamfer / ASYM", not covered. vs W2-E (killed): opposite operation (new-bit metric vs
lossy reorganisation). **No ban collision.**

---

## C. SIGNAL PREREQUISITE — does temporal ORDER carry label-relevant info here?

**The question that decides W2-C.** Order is a real new-bit source (§A), and clears the novelty *escape*
(§B) — but a kernel that reads order only helps if the order **carries label signal**. I ran a **descriptive
cache-forensics pass** (read-only `torch.load`, no vote / no kNN / no label-conditioned probe — a
measurement, not an experiment) on the **already-banked CLIP `subclipK4` caches** (K=4 contiguous temporal
sub-clips over 16 frames; the closest existing proxy for the frame *sequence*):

| dataset | V | within-video sub-clip cosine (1.0 = static) | frac near-static (>0.98 / >0.95) | adjacent-transition/magnitude ratio |
|---|---|---|---|---|
| **HateMM** | 744 | mean **0.884** — **hate 0.899 / non 0.874** | **0.251 / 0.363** | mean **0.370** — **hate 0.301 / non 0.416** |
| **MHC-EN** | 549 | mean **0.855** — hate 0.861 / non 0.852 | 0.024 / 0.120 | mean **0.463** — hate 0.449 / non 0.470 |

**What this establishes (weak-to-negative for the escalation hypothesis):**
- **HateMM: the escalation-arc premise is unsupported at the CLIP level, and mildly *inverted*.** The
  *hateful* class is **more static** (within-cos 0.899 > 0.874) and has **lower** temporal dynamics
  (transition ratio 0.301 < 0.416) than non-hateful — the opposite of "benign setup → dynamic hateful
  reveal." 25% of HateMM videos are near-static (within-cos > 0.98), so for a quarter of the anchor dataset
  there is essentially **no order structure for a kernel to read**. Class separation in dynamics magnitude
  exists but points the "wrong" way for the reveal narrative.
- **MHC-EN: genuine temporal variation exists** (only 2.4% near-static), but **dynamics magnitude barely
  separates the classes** (0.449 vs 0.470 ratio; 0.861 vs 0.852 cosine) — order-*magnitude* is close to
  class-invariant.
- **Both:** the *magnitude* of temporal change is not obviously class-discriminative. (Caveat: magnitude ≠
  order — a video can have large dynamics whose *direction/sequence* carries or does not carry the label;
  this descriptive pass measures "is there dynamics structure at all", not "does the sequence direction
  carry the label." The latter needs the actual order-shuffle **vote** — a probe, out of recon scope.)
- **CLIP<Qwen caveat (load-bearing, per W2-B §E):** these are frozen-CLIP appearance vectors. Qwen's
  instruction-conditioned *semantic* frame vectors could encode narrative turns CLIP misses — so a
  CLIP-negative on order does **not** close the Qwen version. But the static-hateful-class finding is a
  real prior-mover: it is evidence *against* the specific "reveal-structure" motivation, on the cheapest
  available encoder.

**The cheap zero-GPU pre-check that IS possible (recommended, but it is a probe → executor, not recon).**
On the banked `subclipK4` caches, TODAY, CPU/Modal features-only, zero new extraction: add an
**order-aware kernel arm** (transition-set MeanMaxSim, or monotonic soft-DTW over the K=4 ordered
sub-clips) **and an order-within-video shuffle null** (permute the K axis *within each video* — this
leaves POOLED and MeanMaxSim byte-identical, both being order-invariant, and changes ONLY the order
kernel), run through the same W2-B LOO vote. This isolates **order** from set-membership at ~$0. It
**inherits the CLIP<Qwen asymmetry** (a CLIP-null cannot close Qwen; a CLIP-positive corroborates), and
its mechanism (order kernel) is **distinct from and not covered by** the W2-B MeanMaxSim/Chamfer/ASYM ban.
Given the §C forensics, its most likely outcome is "order adds ~nothing on CLIP" — which, with the
static-hateful-class evidence, would revise the W2-C prior down hard before the frameset probe. Worth
queuing as a batch companion; not on the critical path; never GPU.

**What the W2-B per-frame null does and does NOT imply for order.** W2-B's optional per-frame-vector
shuffle null (`W2B_PROBE_RESULTS.md`: HateMM Δacc-95th **−0.1838**, MHC **−0.0302**) shuffled sub-clip
vectors **ACROSS all videos**, *destroying which vectors belong to which video* — it shows MeanMaxSim
collapses when set-membership is scrambled (set structure matters), but it says **nothing about
order-sensitivity** (it does not permute *within* a video). A wrong read would treat that −0.18 as "order
matters a lot"; it does not — it is a set-membership null, not an order null. W2-C needs the **distinct
within-video order-shuffle null** described above; neither S2S (its null also shuffles frame *sets across
videos*, `S2S_PROBE_DESIGN.md:329-333` / K6) nor W2-B tests order — **W2-C must add it.**

**What the S2S temporal positive control σ=[2,0,3,1] would / would NOT establish.** The S2S extractor's
temporal control (`S2S_PROBE_DESIGN.md:152-159`, §4 A1) synthesises a 4-distinct-colour-pair clip, applies
a known input-frame permutation σ (e.g. σ=[2,0,3,1]), and asserts `{g_t}` permutes by the **same** σ (and
each `g_t` is nearest its own colour slab). **It establishes:** the extraction's token→temporal-group
assignment is faithful and temporal-major — i.e. the frameset's order axis is **real and not scrambled by
extraction**, which is a *prerequisite* for any order kernel to be well-defined. **It does NOT establish:**
anything about whether order carries *label* signal — it is an extraction-machine correctness gate
(necessary, not sufficient), exactly analogous to how G-decomp/G-recon gate the aggregate, not the science.
So σ is a **green-light for the order axis being meaningful**, not evidence the axis is discriminative.

---

## D. DEPENDENCY + CACHE PLAN

**Consumes (frameset-blocked).** The S2S Stage-E frameset cache, identical object to S2S:
`data/CLIP_Embedding/{HateMM,MHC}/frameset_qwen7b_8f/{train,dev_seen,test_seen}_frameset.pt`
(`S2S_PROBE_DESIGN.md:379-399`). Each `.pt`: `g` fp16 `[N,T,3584]` (T=4, temporal-order), plus
`n_t,p_S,S,end,labels,grid_thw,zero_guard`. W2-C uses the **ordered** `g[:, 0:T, :]` sequence directly.
**Status: ABSENT** — job **13159 `s2s_extr` PENDING (JobHeldUser)**; `frameset_qwen7b_8f/` not on disk
(verified). Also pending: 13166 `w2a_extr`, 13158 `b5probeC`.

**Meaningful arm needs the T=8 frameset.** At T=4 an order kernel has **3 transitions / a length-4 warp** —
too thin (candidate §D3, and prereg reviewer confirm). The **16-frame arm (`frameset_qwen7b_16f/`, T=8, 7
transitions)** is the meaningful test. S2S pre-declares exactly two frame budgets (8→T=4 primary, 16→T=8
sensitivity, `S2S_PREREG_REVIEW.md` Item 3) — so W2-C's meaningful arm **rides the S2S 16-frame
sensitivity extraction**; confirm that budget is actually extracted, else W2-C's real test is unfunded.

**Additional extraction W2-C needs: NONE beyond S2S's.** It is a pure Stage-P metric added to the S2S probe
— "~0 marginal, rides S2S extraction + S2S probe machinery" (candidate §Cost) is correct.

**Modal-probe feasibility: YES, features-only.** The frameset `.pt` are float vectors + labels (no raw
video); the order kernel + nulls are CPU (soft-DTW over T≤8 is trivially cheap; all-pairs is the same
`V²·T²` batched matmul W2-B already runs in seconds). Cloud-runnable once extracted. **CLIP-K4 pre-check
(§C): runnable TODAY, zero new extraction** — the `subclipK4` caches are banked (`subclip_img_feats`
`[V·K,1024]`, K=4, `num_subclips=4`, contiguous `subclip_parent`; HateMM V=744, MHC V=549 — verified).

---

## E. KILL-BAR SKETCH (pre-registerable; reuse the S2S/W2-B instrument verbatim)

Run **inside the S2S Stage-P probe** as pre-declared arms (`s2s_probe.py`), NOT a separate ceremony/job.

1. **Sole primary arm (declare before results).** ONE contrast:
   **Δ(order-kernel − S2S-MeanMaxSim)** in acc **AND** macro-F1 on the **T=8 (16-frame)** arm, paired LOO.
   The binding claim is "order beats order-*blind*", so the primary is vs **MeanMaxSim**, not merely vs
   POOLED (stricter, and correct — see §F). Report Δ vs POOLED too, but MeanMaxSim is the null that makes
   the mechanism claim. Choose ONE order kernel as primary (transition-set OR monotonic soft-DTW),
   pre-declared; the other is sensitivity, never survival-determining (mirror W2-B B2 / S2S N3).
2. **Oracle kill-switch FIRST — and define the temporal oracle leak-safely.** *The leak the task warns of:*
   NEVER let gold labels choose the cross-video **alignment path** (a label-guided warp injects the label
   into the geometry — inadmissible). The **only admissible** oracle is the W2-B pattern re-applied to
   order-features: gold picks, **per query, which of Q's OWN order-features** (which transition / which
   warp-anchor) to trust (`w2b_probe.py:249-277` `oracle_ceiling`); the memory side and the alignment are
   never oracle-selected (no double-dipping). DEAD-family iff oracle Δ(oracle − POOLED) **< +0.04 on every
   dataset**. Report honestly that at T=4 (3 transitions) this ceiling is thin-by-construction; T=8 is the
   meaningful ceiling. N5 ordering: oracle Δ ≥ raw Δ, else oracle-construction bug (investigate, not
   auto-kill).
3. **Fano ≥ 0.99** (±1 gold-label-agreement key, `w2b_probe.py:283`) — verdict admissible only if the vote
   machine is valid, else VOID.
4. **Conversion-taxed raw bars (P3 shrinkage tax, identical to S2S/W2-B):** HateMM anchor Δacc **AND** ΔmF1
   ≥ **+0.05** vs MeanMaxSim, corroborated by a **rank-only** arm (sim neutralised to 1.0 — de-confounds
   the sim-scale weighting that `metrics.py:270` applies; sign AND own perm-null-95th AND boot-5th>0);
   MHC-EN survival Δ ≥ **+0.03/+0.03**.
5. **Permutation null ≥ 100** (frame-set-across-videos, same-perm-both-arms, `NULL_SEEDS=range(100)`) —
   the generic set↔label null. **PLUS the mechanism-specific ORDER-SHUFFLE null (THE sharpest gate):**
   permute the **T temporal groups WITHIN each video**, which leaves POOLED and MeanMaxSim byte-identical
   and changes ONLY the order kernel; the order kernel's Δ must exceed the 95th pct of this order-shuffle
   null. This is what isolates "order carries signal" from "richer key". Pre-declare it.
6. **Bootstrap 1000**, paired Δ 5th-pct > 0 (D3-fragility guard).
7. **Fail-closed:** no `test_seen` in retrieval memory; assert memory V == 851/629 (train∪val).

**Pre-declared kill (candidate §Probe, tightened):** the order kernel does NOT beat **both** POOLED and
S2S-MeanMaxSim by paired +0.05/+0.03 (per dataset rule) on the **T=8** arm, AND/OR its Δ over MeanMaxSim
does not exceed the **order-shuffle null** → order carries no convertible structure beyond the order-free
set → DEAD (no head GPU). Dataset rule a/b/c/d identical to S2S §6.6.

---

## F. PRIOR — MODEST–LOW, recon-lowered toward LOW; explicit conditional structure

**Diagnosis-law read.** **D1 (decision-side redundancy): does NOT bite** — W2-C is a retrieval *metric*,
no decision scalar (unlike TARC/P1/P2 and unlike the D1-death of low-bandwidth signals). **D2 (only
representation-level levers cleared +3): W2-C IS in this winning class** — a metric over the frame-set
geometry, the same class as S2S. This is a genuine advantage over W2-E (which never entered D2). **D3
(±1–2pt noise floor): binding at T=4** (3 transitions is noise-thin); the T=8 arm is the honest test.

**Conditional structure (the task's core question): is W2-C dead-with-S2S, or does order give it
independent life?**

- **P(W2-C converts | S2S survives) = LOW–MODEST.** S2S surviving means order-*blind* MeanMaxSim already
  converts frame-alignment structure into accuracy. Then order is a **thin increment on an already-winning
  matcher** — W2-C must beat MeanMaxSim by adding order. The §C forensics (dynamics magnitude ~class-
  invariant; HateMM hateful class *more* static) predict the increment is small. But at least the
  "frame-alignment beats pooling on Qwen" premise holds, and order is a plausible refinement → not
  negligible, but below the +0.05-over-MeanMaxSim bar in expectation.
- **P(W2-C converts | S2S dies) = LOW (nonzero — independent life exists in principle).** S2S dying (the
  W2-B-predicted outcome: MeanMaxSim ≈ POOLED) means order-blind set-matching carries nothing. For W2-C to
  convert THEN, order must carry label signal that **both** pooling AND set-matching miss — order is the
  *only* thing that works. This is logically possible (order is a distinct bit-source, §A) — so **W2-C is
  NOT strictly dead-with-S2S; order-awareness does give it a narrow independent life.** But it is a
  strong, specific claim (hate's discriminative structure lives *specifically* in the temporal sequence and
  nowhere in the set or pool), and the §C forensics give it little support. So low-but-nonzero.

**Net prior: MODEST–LOW → recon nudges toward LOW.** The nudge comes from the frozen-CLIP forensics (§C):
the escalation-arc motivation is unsupported (mildly inverted) on the anchor dataset at the cheap encoder,
and dynamics magnitude is near-class-invariant. It is **not** a recon-kill because (i) order is a real
new-bit D2 lever that clears the W2-E meta-family escape soundly, (ii) D1 does not bite, (iii) the marginal
cost is ~0 (rides S2S), and (iv) the CLIP<Qwen caveat leaves the semantic-order hypothesis genuinely
untested. *Falsifiable:* if the order kernel does not beat MeanMaxSim by a paired margin exceeding the
within-video order-shuffle null on the T=8 arm, hate's temporal grammar adds nothing the shared-segment
match already captures — and W2-C dies as a footnote to S2S.

**Because the cost is ~0 and the arm de-risks a whole axis, the honest call is CONDITIONAL-GO as an added
S2S-probe arm — not a NO-GO.** W2-E was a NO-GO because it could not yield a contribution *regardless of the
number* (D7 novelty-gate death) at real ceremony cost. W2-C, by contrast, (a) escapes that gate, (b) costs
~0 marginal, and (c) tests a distinct, falsifiable hypothesis — so the ceremony-cost logic that killed W2-E
does not apply. Recommend: fold W2-C into the S2S probe pre-registration as a pre-declared order-kernel arm
+ order-shuffle null the day the frameset lands; optionally run the zero-GPU CLIP-K4 order pre-check now to
sharpen the prior.

---

## Provenance
- Code: `src/utils/metrics.py:262-320` (vote; sim-as-weight `:268-270`), `src/model/evaluate_rac.py:80-155`
  (memory = IndexFlatIP, one vector/video), `scripts/analysis/w2b_probe.py:1-795` (LOO/oracle/Fano/perm-
  null/bootstrap machinery W2-C reuses; oracle_ceiling `:268-277`, order-null analogue of per_frame_null
  `:365-385`).
- Specs: `refine-logs/S2S_PROBE_DESIGN.md` (frameset contract §6, temporal control §3-4, gate order §5),
  `refine-logs/S2S_PREREG_REVIEW.md` (A1-A5; frame budgets Item 3), `research-wiki/experiments/exp-s2s-r3.md`.
- Kills/bans: `refine-logs/W2B_VERDICT_REVIEW.md` (outcome (d); §E "W2-C not killed, prior moves"; CLIP<Qwen
  §E), `refine-logs/W2E_FORENSIC_RECON.md` (D7 meta-family, no-new-bits), `refine-logs/W2B_PROBE_RESULTS.md`
  (per-frame-across-video null −0.1838/−0.0302), `autoresearch/goal_mllm_plus3/state/directions_tried.json`.
- Cache forensics (read-only `torch.load`, descriptive — NO vote/probe): CLIP `subclipK4` HateMM V=744 /
  MHC V=549, K=4, num_frames=16, contiguous parent; within-video sub-clip cosine + adjacent-transition/
  magnitude ratios by label (§C table). Frameset `frameset_qwen7b_8f/` ABSENT; `squeue`: 13159 s2s_extr
  PENDING (JobHeldUser).
- Repo HEAD at recon: `954b0cb`. Zero GPU / SLURM / Modal used.
