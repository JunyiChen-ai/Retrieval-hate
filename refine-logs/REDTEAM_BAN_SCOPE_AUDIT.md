# RED-TEAM BAN-SCOPE OVERREACH AUDIT — adversarial audit of the kill ledger

**Agent:** red-team recon (adversarial, internal). **Date:** 2026-07-20 NZST.
**Mission.** Refute "we exhausted the space" *from the inside*: for every dead entry in
`autoresearch/goal_mllm_plus3/state/directions_tried.json` and every structural law in
`research-wiki/DRAFT_analysis_chapter.md` §3.6–§3.9 + `state/findings.jsonl` F1–F60, reconstruct the
syllogism **(measured fact → recorded ban)** and mark every place the ban generalizes past the
measurement (an INDUCTIVE LEAP) rather than following from it (a THEOREM). Each leap = a cell that is
"closed" on paper but factually untested.

**Discipline honored.** CPU-only reading + forensic arithmetic. ZERO GPU / SLURM / Modal / training.
No `state/` mutated. One deliverable (this file). Proposed in-gap tests are *recorded for the
orchestrator*, not run here.

**Method.** For each suspect I quote (i) the measured cell from its primary `refine-logs/` record,
(ii) the recorded ban verbatim, (iii) a THEOREM/LEAP verdict. Confirmed leaps get
{ledger id · measured fact+citation · ban quote · gap · minimal in-gap test · cost · prior · in-box
legality}. Airtight closures are listed separately so the orchestrator knows which survived.

---

## BOTTOM LINE

The "exhausted the space" claim is **mostly sound but not airtight.** The frozen-representation
closures (GIR linear-subset, per-item router at 3 supervision sources, P9b redistribution, hybrid-memory
subsumption, cumulative-causal *prefix* closure) survive an adversarial read. But **seven ban scopes
generalize past their measurement.** The single strongest is the **audio axis**: it was screened with a
**classical 88-d whole-video prosody vector on HateMM only** and the record *itself concedes* that vector
"only weakly lower-bounds a learned audio encoder" — yet the ban reads "audio axis PARKED." A learned
general-audio embedding (Whisper-large-v3 encoder, **already on disk, no download**) and the entire
MHC-EN audio axis are **untested**, and audio is a *new-input channel* that structurally escapes the
"representation-lever / label-limited" walls that close everything else on EN. Six further leaps are
recorded below.

---

## PART A — CONFIRMED OVERREACHES

### GAP-1 · APX / F41 — "audio axis parked" leaps from *classical whole-video prosody, HateMM-only* to *all audio*

**Measured fact.** `APX_GATE_RECORD.md` §3 (commit `9c54faf`): the aux block was the openSMILE
**eGeMAPSv02 88-d whole-video functional** vector (means/percentiles/std of ~25 low-level descriptors,
`§1` table). Conditional-info over `Z_best`(8960-d) on **HateMM only** (`§1.2`: "Scope: HateMM only …
N = 851 train∪val"): best `audio_pca_k8` **−0.0038** CI[−0.0113,+0.0033], strictest full-88-d
**+0.0005** CI[−0.0031,+0.0042], calibration accZA = 1.0000. The record's own scope note (`§5`):
> "Because eGeMAPS *upper-bounds* the cheap realization (and only **weakly lower-bounds a learned audio
> encoder**), the classical probe de-risks the whole acoustic axis downward."

**Recorded ban** (`directions_tried.json`, APX entry):
> "classical whole-video prosody as auxiliary channel on HateMM; AVC correspondence variant dies with
> its gate; **audio axis PARKED** — any future audio proposal (incl. W2-D Qwen-Omni download) must first
> explain how it beats a zero-information classical baseline through the same conditional screen."

**Verdict: INDUCTIVE LEAP (two of them).** (1) eGeMAPS 88-d functionals are a whole-video *prosody
summary*; they cannot represent temporal **acoustic events** or **audio semantics** — a hateful chant's
melody, a gunshot, mob/crowd noise, a laugh-track over a slur, music genre. `W2D_FORENSIC_RECON.md` §A
names exactly these as "the true residual … paralinguistics." A learned general-audio encoder captures
that structure; eGeMAPS blurs it into 88 scalars. The record concedes eGeMAPS only *weakly* lower-bounds
a learned encoder, so the zero does **not** transfer to learned audio — yet "audio axis PARKED" sweeps it
in. (2) The screen was **HateMM-only**; the conditional-info of *any* audio channel over `Z_best` on
**MHC-EN (the binding-gap dataset) was never measured at all** — not even classically. Audio is a genuine
**new-input channel** (`W2D` §F/D2), so the F44 "no representation lever converts label-limited EN" wall
does **not** apply to it: a new modality can add signal a representation lever cannot.

**Gap.** `I(label ; learned-audio-emb | Z_best)` on HateMM, and `I(label ; audio | Z_best)` on MHC-EN,
are both untested. The "same conditional screen" the ban *demands* for learned audio was never run.

**Minimal in-gap test.** Local `ffmpeg -ac 1 -ar 16000` → **Whisper-large-v3 encoder mean-pool** (on
disk, `W2D` §D; large-v3 GPU-light, base is CPU-OK) → one `a`-vector/video for HateMM + MHC-EN → reuse
`scripts/analysis/apx_g0cond_gate.py` (= `c3_fusion_probe.py`) conditional-info gate verbatim over
`Z_best`. Oracle-kill-switch FIRST (per `W2D` §C), calibration mandatory.

**Cost.** ~1 local extraction (CPU/GPU-light, ~10 min/dataset by the eGeMAPS precedent 13203 = 8m53s) +
**$0 CPU** probe. **No download** for the Whisper variant.

**Prior.** LOW (~10–15%). Against: F31 hazard is real (Whisper-large-v3 ASR already banks *spoken* hate
into the text channel). For: non-speech events + audio semantics are genuinely uncaptured by both the ASR
transcript **and** eGeMAPS, and EN's audio axis is a blank cell on the one dataset the goal needs.

**In-box legality.** YES — CPU probe + local features-only extraction, raw audio stays local, no model
download (Whisper already cached).

---

### GAP-2 · S2S / F37 (+ F35/F39) — "don't-pool family closed across encoders" leaps from *causal-prefix* to *all* Qwen frame-groups

**Measured fact.** `S2S_PROBE_VERDICT_REVIEW.md` (commit `2c96ab6`): MeanMaxSim SET-vs-POOLED over
Qwen2.5-VL per-frame-group vectors, HateMM Δacc **+0.0035** (fails +0.05, inside perm-null), MHC-EN
**−0.0397**. `S2S_GATE0A_POSTMORTEM.md` (F35, `4358ca1`) proves the group vectors `g_t` are **cumulative
causal prefix summaries** (LLM `is_causal=True`; position dominates content: diff-colour-same-position cos
0.939 vs same-colour-diff-position 0.674), so "pooling is effectively lossless on these representations."
`CTF_GATE_RECORD.md` (F39) adds that even a **supervised** conditional-info probe over the flat
`[g_1…g_T]` tensor finds +0.0000/−0.0029 over `Z_best`.

**Recorded ban** (`directions_tried.json`, S2S + CTF entries):
> "family-level: retrieval-object/don't-pool family **CLOSED across encoders** (W2-B frozen-CLIP + S2S
> Qwen both dead)"; "do not re-propose **temporal structure over Qwen framesets in any operator class**."

**Verdict: INDUCTIVE LEAP.** The Qwen half of the closure rests **entirely** on the causal-**prefix**
property (F35), which is an artifact of the **single-forward** extraction (all frames through one causal
pass, so `g_t` = prefix, not frame-local). F35's own structural argument **does not apply** to
**independently re-encoded** per-segment representations (each segment its own Qwen forward → a genuinely
**frame-local** vector, not a prefix summary). W2-B tested independent segments only with **CLIP** (a
weaker, non-causal encoder); the **frame-local-Qwen-segment** cell sits between W2-B and S2S and was never
extracted. The postmortem F35 chose to *keep* the causal representation and reword the premise
("cumulative causal group summaries") rather than pivot to independent re-encoding — so the pivot cell is
uncovered, and "across encoders … any operator class" papers over the fact that the Qwen measurement never
saw frame-local Qwen semantics. The linear/supervised zeros (F37/F39) also leave a nonlinear **learned**
set kernel over the tensor formally uncovered (F46's linear-conditional-zero caveat), but that residual is
guarded by F35 structurally; the independent-segment residual is **not**.

**Gap.** MeanMaxSim / set-matching over **independently-re-encoded per-segment Qwen** vectors (frame-local,
F35-immune). Never extracted, never probed.

**Minimal in-gap test.** Re-extract Qwen per-segment (K segments, each an *independent* forward on its own
frames) → same oracle-kill-switch + MeanMaxSim probe as S2S. Pre-check: verify the per-segment vectors are
NOT prefix-correlated (onset-invariance control should now *fail*, confirming frame-locality).

**Cost.** Real per-segment Qwen extraction (Modal cloud triage probe per CLAUDE.md — features derivable,
raw video stays local) + CPU probe. **Not $0.**

**Prior.** LOW-MODEST (~10–20%). Against: both prior segment tests (W2-B CLIP, S2S prefix-Qwen) showed
*oracle-exists / operator-can't-convert*; per-segment Qwen is thin (few frames/segment) and loses the
cross-segment context that makes Qwen strong. For: frame-local Qwen semantics were **never seen**, and the
oracle headroom (+0.09/+0.14) is real.

**In-box legality.** YES as a cloud triage probe (features-only export, videos local); not for this
CPU-only audit to run.

---

### GAP-3 · F60 (AUG) — "MLLM data-augmentation dominated by cand-2" leaps from *encoder-SFT* to *head-contrastive* augmentation

**Measured fact.** `AUG_FORENSIC_RECON.md` (commit `f1abd28`) kills MLLM-as-data-generator because "the
adapted object is still the **encoder** → F51 re-entered" and it is "**dominated by cand-2's measured ZH
tie (same object/leg/split**, weaker hook)" (`CAND2_VERDICT_REVIEW.md` F56, `546acc5`: ZH curriculum SFT =
K-C2-2 TIE both protocols). The dominance argument is scoped to **encoder LoRA-SFT** augmentation
throughout (§2 "Wall A — the adapted object is still the encoder").

**Recorded ban** (`directions_tried.json`, AUG entry):
> "MLLM-generated train-data augmentation for **encoder adaptation**: dominated by cand-2's measured ZH
> tie … Do not re-propose without D7 generator-role sub-ruling."

**Verdict: INDUCTIVE LEAP.** The measured/dominated object is the **encoder** (cand-2 = encoder SFT
curriculum; F51's two-object closure is about *adapting the MLLM*). MLLM-synthesized, **label-preserving**
hard-negative examples added to the **RGCL head's triplet-contrastive training set** over **frozen**
features are a **different injection object** (the head, not the encoder) and are **not** cand-2's object.
AUG's own §1 concedes the generator role "clears C3/P4 (features), P11 (scores), TARC (loss),
single-dataset veto, data boundary — the un-enumerated generator role is real," and the banned-constraints
list *permits* "representation-training expansion" (only *vote-pool* pseudo-label expansion is banned). So
head-training-data synthesis with inherited-gold labels is an **allowed-but-unmeasured** cell that the
encoder-scoped domination argument does not reach.

**Gap.** MLLM-generated gold-labeled boundary examples in the **head's** contrastive training pool (frozen
encoder), not the encoder SFT set.

**Minimal in-gap test.** $0 CPU screen FIRST: does adding the paraphrases' *frozen* features change the
head's per-epoch **hardest-opposite-label** mined set vs the existing online mining
(`src/model/retrieval.py:347`)? If the online miner already covers them → auto-kill. Only if the mined set
changes non-trivially does a head-retrain (~2 min GPU) run.

**Cost.** Generation (local GPU, ~0.5–1 A100-h) + $0 CPU screen + optional 2-min head retrain.

**Prior.** LOW (~5–8%). Against: the head *already* mines global-hardest-opposite pairs per-epoch from the
frozen geometry (the C3geo/F25 finding), so synthetic negatives must beat online mining; distribution-shift
risk. For: it is a genuinely un-enumerated object with a cheap kill screen.

**In-box legality.** Partial — the $0 screen is CPU/in-box; the confirm needs local generation + GPU (loop
decision).

---

### GAP-4 · F49 (MJ) — the "alignment > 0.663" pre-measurement bar over-generalizes an MHC-EN-specific number and closes F47's own carve-out

**Measured fact.** `MJ_FORENSIC_RECON.md` (commit `d57d05d`) §1: on the **80-item MHC-EN dev** split
(disagreement sizes 20/23/20, always-Qwen prior 0.588), `gain(q) = 0.2625·q − 0.15415`, so clearing
+0.020 needs **q ≥ 0.6634**; the modality-locus alignment ceiling `a ≤ 0.588` (F44 "no coherent
subgroup"), so a *perfect* modality judge cannot clear it. The 0.663 threshold is a function of that
specific dev geometry (D/N and the 0.588 prior).

**Recorded ban** (`directions_tried.json`, MJ entry):
> "Modality-locus judgments (MLLM or otherwise) as router inputs: dead at alignment ceiling (a<=0.588 <
> q_req 0.663). **F47 carve-out now requires demonstrated alignment>0.663 from banked evidence BEFORE any
> gate.**"

vs F47's carve-out (`ROUTER_GATE_RECORD.md`, F47):
> "unless the selector input is **a genuinely NEW information source not derivable from banked
> features/votes**."

**Verdict: INDUCTIVE LEAP (dataset-specific number + logical catch-22).** (1) `0.663` is **MHC-EN-dev
arithmetic**; a different routing problem (different disagreement geometry / prior) yields a different
`q_req`, so it is not a universal bar. (2) The alignment ceiling `a ≤ 0.588` is specific to the
**modality-locus** input class (F44 measured *modality* ⊥ which-arm-wins); a different new input class is
not bounded by it. (3) The catch-22: F47 carves out a "genuinely NEW information source **not derivable
from banked features/votes**," but F49 then requires that source "**demonstrate alignment>0.663 from
banked evidence** before any gate." A genuinely-new source **cannot** show its alignment from banked
evidence — by definition — so the F49 bar makes F47's own carve-out **unenterable on arithmetic**, letting
a truly-new router input be pre-killed without the $0 gate F47 actually ran (which ran *because* its oracle
exceeded the bar).

**Gap.** Any genuinely-new router-input class (not modality-locus, not banked-derivable) is arithmetic-
pre-killable under F49 without measurement — the very thing F47 carved out.

**Minimal in-gap test.** For any proposed new router input, run the **F47 $0 router gate**
(`cross_channel_router_gate.py`, banked 12 e29 heads, CPU minutes, label-oracle-calibrated) rather than
the F49 arithmetic pre-kill. The gate itself measures realizable alignment.

**Cost.** **$0 CPU** per candidate (F47 machinery is banked and bit-exact-validated).

**Prior.** LOW that any new input converts (F47 closed the *general* router at all 3 supervision sources
with the full meta-feature set) — but the **doctrine is over-scoped**, and the fix costs nothing.

**In-box legality.** YES — $0 CPU, banked heads.

---

### GAP-5 · F55 / F50 / F51 — "EN closed at all levels / to the entire family" omits the *audio* and *vision-adaptation* levels

**Measured fact.** EN closure is asserted at exactly three composition levels:
`PREMISE_D_GATE_RECORD.md` (F55, `6e6061b`) "closed at frozen (F50), collapsed-adapted-deployed (B4/F53),
and healthy-image ⊕ adapted-text composition (F55) levels"; all three concern the **text/vision-feature
composition** of CLIP+Qwen. `TIE_BRANCH_RECON.md` (F54, `6b9985a`) establishes that **every** LoRA-SFT
**froze the vision tower + projector** (`lora_target: all` reaches only the LLM backbone) and every SFT
target was text-decodable, so the image stream stayed flat *for those targets*.

**Recorded ban** (`directions_tried.json`, premise-d + WAVE5):
> "EN closed at all three levels." / "MHC-EN is label-limited … **No representation lever — adapted or not
> — clears +0.03 on EN** … **EN is closed to the entire family**" (`WAVE5_CANDIDATES.md` §0.2).

**Verdict: INDUCTIVE LEAP.** Two admitted-unmeasured EN levels are swept into "all levels / entire
family": (a) **EN + audio** — the APX audio screen was **HateMM-only** (GAP-1), so EN's audio conditional-
info is a blank cell; audio is a *new-input channel*, not a representation lever, so the "no representation
lever converts label-limited EN" clause does **not** bound it. (b) **EN + vision-obligatory SFT** — F54
shows the collapsed Qwen EN image stream (AUC 0.599) is **architecturally movable** by an SFT target that
routes gradient through the vision tokens, but no such target was ever run; F50/F55 price EN's **healthy
CLIP** image stream, not a **repaired Qwen** image stream. `WAVE6_PREMISE_HUNT.md` surface-4 concedes the
per-stream picture on the passing cell is "reasoned, not measured," and premise-(a) (vision-SFT) prior is
"armchair."

**Gap.** EN audio conditional-info screen; EN vision-targeted SFT on the collapsed Qwen image stream.

**Minimal in-gap test.** (a) = the MHC-EN leg of GAP-1's $0 audio screen. (b) an SFT target that forces
gradient through vision tokens (e.g. transcript-dropout / describe-the-frame) on EN, then the standard
3-seed head — GPU, one LoRA run.

**Cost.** (a) $0 CPU + local extract; (b) ~7–9 A100-h (one adaptation chain).

**Prior.** (a) LOW (~8–12%); (b) LOW (~5–10%, F44 label-limited prior stands even with a fixed image
stream, since the residual error core is intrinsic).

**In-box legality.** (a) YES (CPU + local extract); (b) GPU (loop decision).

---

### GAP-6 · B2 / TERMINUS relaxation-(a) — the "scale regresses" counter-evidence is *frozen-only* but is cited against a *download-then-LoRA* larger scale

**Measured fact.** `B2_VERDICT_REVIEW.md` (job 13146) measured **frozen** Qwen2.5-VL-32B: HateMM final acc
CLIP 0.8124 < **32B 0.8450** < 7B 0.8682 (32B *between* CLIP and 7B); 32B-vs-7B FAIL all datasets. F44
attributes the regression to the **frozen** image-stream collapse persisting at 32B (0.608). No *adapted*
(LoRA) larger-scale cell was ever measured (only 7B is downloaded, F8).

**Recorded ban** (`TERMINUS_round3_mllm_plus3.md` relaxation table, row a):
> "**B2 measured scale REGRESSES**: HateMM 32B between CLIP and 7B; 32B-vs-7B fails every dataset … scale
> is not the conversion lever" → prior "**LOW — direct measured counter-evidence**; scale is not the
> conversion lever."

**Verdict: INDUCTIVE LEAP (frozen → adapted).** B2 is a **frozen-encoder** measurement. The campaign's
own Structural Law IV (F45/F53/F58, `DRAFT_analysis_chapter.md` §3.9) states the convertibility line runs
through **adaptation, not encoder identity** — frozen swaps re-rank, LoRA re-decides. A LoRA-32B/72B adapts
the **language backbone** (exactly where conversion lives per F45/F58), which the frozen image-collapse
mechanism does not bound. So citing frozen-B2's "scale regresses" as "direct measured counter-evidence"
against a download-then-**adapt** larger model over-scopes a frozen result onto an adapted cell of a
different mechanism class. The prior label "LOW — direct measured counter-evidence" should read "LOW on
*frozen* scale; the adapted-larger-scale cell is **unmeasured**."

**Gap.** LoRA-adapted 32B/72B encoder — the adapted-scale cell — is unmeasured; the counter-evidence cited
against it is frozen-only.

**Minimal in-gap test.** (download-gated) LoRA-SFT a larger Qwen2.5-VL and run the standard 3-seed encoder
comparison. No cheap screen exists (adaptation adds no conditional info a $0 gate can see — the F60/AUG
argument).

**Cost.** HIGH — model download (prior 32B/72B download attempts failed, F8) + multi-run SFT+extract+head.

**Prior.** LOW-MODEST — genuinely open under Law IV, but download-blocked and expensive; this is already a
flagged user relaxation-(a), so the overreach is in the *prior justification*, not a silently-closed cell.

**In-box legality.** NO — needs a download ruling (relaxation-a); recorded as a scope correction, not an
in-box move.

---

### GAP-7 · F51 two-object closure — airtight for *adapting the MLLM*, but WAVE6 concedes *head/key-map recipe as a performance lever* is uncovered

**Measured fact.** `WAVE5_CANDIDATES.md` (F51, `7166232`): "adaptation has exactly two adapted objects —
encoder (generic LoRA) and joint encoder+decision (retrieval-loss-into-LoRA = P9b/D3, KILLED). No third
object exists." `WAVE6_PREMISE_HUNT.md` surface-1: "There is **no banked negative on pure head-architecture
variation** — honestly uncovered as a *performance* lever."

**Recorded ban** (`directions_tried.json` F51): "= P9b's adapted object … do not re-propose."

**Verdict: NARROW LEAP (out-of-scope, not goal-relevant).** F51's "no third object" is airtight for
**adapting the MLLM** (encoder gradient vs encoder+decision gradient). But the tasking's candidate "adapt
the retrieval **key-map / head recipe** only, encoder frozen" is a *different* object F51 does not address,
and WAVE6 surface-1 concedes it is unmeasured as a performance lever. It is ruled out on **D7** (adds no
MLLM role → generic classifier tuning) and on a **reasoned** (not measured) F44 "EN label-limited, a
better-regularized head cannot manufacture signal." So the cell is genuinely untested but (i) D7-dead
regardless of performance and (ii) its EN-flatness is inferred, not measured.

**Gap.** Uncoupled head/key-map recipe (topk, dropout, epochs, triplet-vs-InfoNCE, early-stop of the
`0.998`-memorized bank) as a *performance* lever — unmeasured, but D7-out-of-scope.

**Minimal in-gap test.** Head-recipe sweep on banked features (CPU/cheap GPU); purely a performance/paper
diagnostic, cannot satisfy the novel-MLLM clause.

**Cost.** Low (banked features). **Prior.** ~0 on the goal (D7-dead + EN label-limited). 

**In-box legality.** YES but off-goal — recorded for completeness; not a refutation of the *goal* closure,
only of the literal "space exhausted as a performance lever" phrasing.

---

## PART B — SPOT-CHECKS OF UNSUSPECTED DEAD ENTRIES (guarding against confirming only my priors)

- **P9b (joint LMM-RGCL)** — `directions_tried` "head↔memory redistribution, 0/12"; measured
  D3−C3′ = kNN +1.8 / head −1.8 ZH, net ~0, 0/12 cells beat floor (`DRAFT_analysis` §3.4). Ban scope
  ("retrieval-loss-into-LoRA … governed by P9b redistribution law") = **AIRTIGHT** for its object; the
  measurement *is* the general mechanism, not one point.
- **W2-E (prototype memory)** — `directions_tried` "zero-training unsupervised reorganisation of frozen
  pooled features." Measured/argued: both variants are "deterministic lossy functions of the SAME single
  pooled vector ⇒ zero new signal" (F28). Ban = **THEOREM** (a lossy function of a fixed vector adds no
  information; no measurement needed). AIRTIGHT.
- **TARC (target-conditioned retrieval graph)** — `exp-tarc-t0.md` G3 "no TEST transfer on either dataset;
  single TARC test-touch consumed." Ban ("regularization-only; val moves, test flat; test-touch spent")
  matches the measurement; closure is **procedural** (test-touch budget spent), not a scope leap. The one
  residual — target-as-*structure* — is separately user-vetoed. Effectively airtight.
- **archive-auto-repair (two-vote AND rule)** — `DRAFT_analysis` §4: measured C−A = +0.0000 (0/4 EN
  seeds); ban = "guard-rail role only." Scope matches (the AND rule "structurally cannot reach memories
  that are semantically contradictory yet not embedding outliers" — a mechanism, not one draw). AIRTIGHT
  for the AND rule; the paper honestly keeps it as a *veto*, not an accuracy claim.
- **P5 (counterfactual twins) / P1–P4** — decision-side, each individually measured (gate-fail+hurts;
  §3.1 numbers). The blanket "P1–P5 re-proposals" ban is housekeeping over five individually-measured
  cells; no untested realization surfaced. AIRTIGHT.

---

## PART C — BANS VERIFIED AIRTIGHT (survived the audit — do NOT reopen these)

| ban | why airtight |
|---|---|
| **GIR / F43** (grounded-residual) | `GIR_GATE_RECORD.md` (`b64a85b`): `r_cache` proven an **EXACT linear function of two baseline columns (residual norm 0)** — a THEOREM, not an induction. The only sliver (`r_field`, cos 0.9986) also measured null. |
| **Router / F47** general per-item selection | closed at **all three** supervision sources (unsupervised K9 zeros, train-supervised degenerate target CLIP-LOO 0.998, dev-supervised CV ceiling −0.046 < perm-null) with the **full** meta-feature set. Residual = genuinely-new-source only (→ GAP-4, which is a *doctrine* over-scope, not a hole in F47). |
| **Cumulative-causal *prefix* closure / F35+F37+F39** | three-level (structural / unsupervised / supervised-linear) closure of temporal structure over **prefix** vectors. Airtight *for the prefix representation*; the only escape is the **independent-segment** representation (→ GAP-2), which F35 explicitly does not cover. |
| **Hybrid CLIP+LoRA memory (WAVE6 surface-3)** | score-sum decision provably **= a max-\|vote\| selector over banked votes** on the disagreement set ⇒ literally inside the F47 ban; rank-merge = W2-E ban. Provable subsumption, no gate warranted. |
| **P9b redistribution / W2-E prototype / GIR** | all resolve to theorems or full-mechanism measurements (Part B). |
| **Encoder-swap mechanism / B2 frozen scale** | AIRTIGHT for **frozen** identity + **frozen** scale (F44 image-collapse retro-predicts B2). Over-scoped only when extended to **adapted** scale (→ GAP-6). |

---

## PART D — RANKED GAP TABLE (by prior × cheapness = expected value per unit cost; best bets first)

| rank | gap | ledger id | prior | cost | in-box | one-line |
|---|---|---|---|---|---|---|
| **1** | **Learned-audio / EN-audio screen** | APX F41 (+F55 EN) | ~0.10–0.15 | **$0 CPU + local extract, no download** | **YES** | eGeMAPS-88d-HateMM ≠ learned-audio-events; EN audio never screened at all |
| **2** | **F47-gate for new router inputs (drop the 0.663 arithmetic pre-kill)** | MJ F49 | low-but-doctrine | **$0 CPU** | **YES** | 0.663 is MHC-EN-specific; the bar closes F47's own carve-out |
| **3** | **Independent-segment (non-prefix) Qwen set-matching** | S2S F37 / F35 | ~0.10–0.20 | cloud extract + CPU probe | cloud | closure rests on causal-prefix artifact; frame-local Qwen never extracted |
| **4** | **Head-contrastive MLLM augmentation** | AUG F60 | ~0.05–0.08 | local gen + $0 screen + 2-min GPU | partial | domination is encoder-SFT-scoped; head object un-enumerated |
| **5** | **EN vision-obligatory SFT (repair collapsed Qwen image)** | F55/F50/F51 | ~0.05–0.10 | ~7–9 A100-h | GPU | F54 says image is movable; only healthy-CLIP-image was priced |
| **6** | **Adapted (LoRA) 32B/72B scale** | B2 / relax-(a) | low-mod | HIGH (download+SFT) | NO (download) | frozen "scale regresses" ≠ adapted scale (Law IV) |
| **7** | **Head/key-map recipe as performance lever** | F51 / WAVE6-s1 | ~0 on goal | low | YES (off-goal) | genuinely unmeasured but D7-dead + reasoned-flat |

**Reading.** Gaps 1–2 are **cheap and in-box** — the strongest refutations of "exhausted": the audio
axis was closed on a classical HateMM-only proxy the record admits is a weak lower bound, and the router
pre-kill doctrine is over-scoped at $0-fixable cost. Gaps 3–5 are real untested cells at moderate GPU/cloud
cost with honestly-low priors. Gap 6 is a *prior-justification* over-scope (download-gated). Gap 7 is
off-goal. None overturns the *goal* verdict on its own, but the "space is exhausted" claim is
**false as stated for the audio axis** and **imprecise** for the router-doctrine, temporal, EN, and scale
closures.

---

## PROVENANCE
- Ledger: `state/directions_tried.json`, `state/findings.jsonl` F1–F60.
- Primary records quoted: `APX_GATE_RECORD.md` (`9c54faf`), `W2D_FORENSIC_RECON.md` (`ad48dcc`),
  `S2S_PROBE_VERDICT_REVIEW.md` (`2c96ab6`), `S2S_GATE0A_POSTMORTEM.md` (`4358ca1`),
  `CTF_GATE_RECORD.md` (`0eb6d33`), `MJ_FORENSIC_RECON.md` (`d57d05d`), `ROUTER_GATE_RECORD.md`
  (`30d0ee1`), `AUG_FORENSIC_RECON.md` (`f1abd28`), `PREMISE_D_GATE_RECORD.md` (`6e6061b`),
  `FA_GATE_RECORD.md` (`e0877c9`), `WAVE4_CANDIDATES.md`, `WAVE5_CANDIDATES.md` (`7166232`),
  `WAVE6_PREMISE_HUNT.md` (`c53cfe1`), `B2_VERDICT_REVIEW.md`, `TIE_BRANCH_RECON.md` (`6b9985a`),
  `GIR_GATE_RECORD.md` (`b64a85b`), `TERMINUS_round3_mllm_plus3.md`,
  `research-wiki/DRAFT_analysis_chapter.md` §3.6–§3.9, `research-wiki/experiments/exp-tarc-t0.md`.
- **Required statements:** ZERO GPU / SLURM / Modal / training spent by this audit; no held-out test
  metric read or produced; no `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated.
  Committed on `main`, not pushed.
