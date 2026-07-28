# NUMERIC PROVENANCE AUDIT — 2026-07-28

**Date:** 2026-07-28 NZST · **Mode:** $0 provenance audit, read-only on all research artefacts.
**Standing rule under audit:** *never transcribe a number without re-reading its source; never
fabricate companion metrics; consistency audits must spot-check primary logs* (the 0.8732 incident;
`research-wiki/` numeric-provenance discipline, ERRATUM commit `66012e9`).

**Compliance log (verifiable):**
- No SLURM submitted, no Modal launched, **no GPU used, nothing trained.** All re-derivation is
  CPU-only `torch.load` + `faiss` over **banked TRAIN-split caches** on the login node.
- **No test split was opened.** Every re-derived number below is a train-split leave-one-out
  quantity. Test-side numbers are quoted from banked verdict records and were re-read, not re-run.
- Scratch scripts live in the session scratchpad; this file plus the corrections listed in §5 are
  the only repo writes.
- Files belonging to other agents in flight were **read only** and are named, not edited:
  `VSW_PREGATE_RECORD.md`, `VSW_ASYMMETRY_RECON.md`, `LITSWEEP7_LANDING_SITE.md`,
  `LITSWEEP8_PATHOLOGY_MATCH.md`, `MEMBANK_C4_PREGATE_RECORD.md`,
  `STREAMCOMP_FORENSIC_RECON.md`, `INSTRUMENT_VALIDATION_RECON.md`, `scripts/analysis/vsw_*`,
  `scripts/analysis/*membank_c4*`.

---

## 0. VERDICT SUMMARY

| # | claim relayed upward today | audit verdict |
|---|---|---|
| 1 | cand-2 curriculum: targeted train-LOO **−0.0538** HateMM / **−0.0402** ZH on the top-30 % confusable head, **0.0000** on the deprived tail | **numbers REPRODUCE bit-exact; the INTERPRETATION is RETRACTED.** The contrast is not curriculum-specific, the tail zero is a tautology, and the second draw of the same curriculum has the opposite sign. |
| 1b | "…buying a deployed **+0.0132** / **0.0000**" | **+0.0132 VERIFIED. "0.0000" RETRACTED** — the ZH quantity is a *TIE adjudication* over −0.0067 / +0.0067, not a measured zero. |
| 1c | "this is the campaign's **tenth law-I datum**" | **RETRACTED IN FULL.** It is not a law-I datum at all: on HateMM the train-side gain **did** convert. |
| 2 | certified law-I count | **NINE.** Enumerated in §2 with a source per entry. Paper docs already correct; one in-flight record says eight and is flagged. |
| 3 | F97's "78/78 parity PASS" | **does not re-assert today** — cross-session CPU-training non-determinism, not a defect. **Zero verdicts move**; one headline *count* ("0 of 36") is session-dependent. §4. |
| 4 | 13 load-bearing numbers in today's F99-F103 records and the reused campaign anchors | **11 PASS, 1 material FAIL** (the fabricated ZH "0.0000"), **1 cosmetic** (OCR char stats off by 1-2; coverage % exact). §5. |
| 5 | litsweep-7: the numbers "co-occur only in an unrelated 2026-07-13 SAV table" | **substring-grep artefact** — neither number is in that table as a quantity. Not evidence of a misread. |

**The one genuinely new result rescued from this audit:** the MAC computation, once the placebo arms
are added, **discharges litsweep-7's own proposed candidate L4 / CURDIAG**
(`LITSWEEP7_LANDING_SITE.md:509-566`) at $0 — with a **split** answer that neither the retracted
framing nor the proposal anticipated. On the curriculum cell the $0 train arena is **valid** (outcome
1: same sign, same dataset ordering, ~2× attenuating). On the encoder-adaptation axis it is
**anti-correlated across datasets** — MHC-EN shows the *largest* train-arena gain (+0.0729) and is the
dataset closed at all three levels. **Arena negatives are informative; arena positives are not.** See
§1.6-1.7.

---

## 1. ITEM 1 — THE MAC CURRICULUM TRAIN-LOO CONTRAST

### 1.1 The artefacts exist; litsweep-7's search was correct but scoped to the repo

Litsweep-7 reported (`LITSWEEP7_LANDING_SITE.md:59-73`) that no curriculum train-LOO contrast exists
anywhere in the repo, including under `git log -S`. **That is true and remains true.** The MAC recon
agent computed these numbers *this session, into the session scratchpad*, and never wrote them to the
repo — so a repo search could not have found them. Both reports were locally honest; the collision was
one of scope.

Located, with mtimes:

| artefact | size | mtime |
|---|---|---|
| `<scratchpad>/mac_recon.py` | 8 339 B | 2026-07-28 08:43 |
| `<scratchpad>/MAC_RECON_OUT.json` | 10 398 B | 2026-07-28 08:43 |
| `<scratchpad>/mac_recon2.py` | 4 866 B | 2026-07-28 08:45 |
| `<scratchpad>/MAC_RECON2_OUT.json` | 1 484 B | 2026-07-28 08:45 |

`MAC_RECON2_OUT.json` contains, verbatim:
`HateMM.curric_minus_generic_LOOerr = {all −0.0202, head −0.0538, deprived 0.0, mid −0.0108}` and
`MHC_zh.curric_minus_generic_LOOerr = {all −0.0086, head −0.0402, deprived 0.0, mid +0.0094}`.

### 1.2 Independent re-derivation — the numbers REPRODUCE

The JSON was **not** trusted. A re-derivation was written from the spec rather than from
`mac_recon2.py` (`<scratchpad>/audit_mac_rederive.py`), differing from it in engine, dtype and
apportionment code path:

* **vote engine** = `scripts/analysis/mechfix_ops.deployed_vote(..., exclude_self=True)`, the frozen
  F89 operator, sha256 re-verified this session =
  `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` ✓
  (MAC's own scripts used a hand-rolled vote loop, not this module.)
* **key space** = `l2n(concat(l2n(img_feats), l2n(text_feats)))`, i.e.
  `mechnov_pairverify.build_space(..., "fused")` — the same arena every $0 pregate since F89 uses.
* **dtype** = float64 load (MAC used float32).
* **strata** defined under φ0 (frozen Qwen): `c = exp(−|v|/0.20)`, `w = 1 + 10·c`, multiplicities by
  largest-remainder apportionment to total N; `head` = `c ≥ p70`, `deprived` = `mult == 0`.

**PARITY GATE (mandatory, run first).** MHC-ZH generic-LoRA fused deployed train-LOO accuracy
= **0.8480**, against the banked F95 anchor **0.8480**
(`refine-logs/MECHNOV_PAIRVERIFY_PREGATE.md:299`; independently re-confirmed at
`VSW_ASYMMETRY_RECON.md:142-143`). **PASS, exact at 4 dp.** The harness is the banked harness.

**Result — the headline reproduces bit-exact:**

| stratum | HateMM curric − generic (LOO error) | MHC-ZH curric − generic (LOO error) |
|---|---|---|
| all | −0.0202 | −0.0086 |
| **head (top-30 % confusable)** | **−0.0538** ✓ | **−0.0402** ✓ |
| deprived tail | 0.0000 | 0.0000 |
| mid | −0.0108 | +0.0094 |

So the arithmetic that was relayed upward is **real, reproducible, and computed in the correct arena
with the correct frozen operator.** Litsweep-7's conclusion "the datum does not exist" is wrong on
existence.

*(Two sub-4dp reconciliations, disclosed. (i) My `n_deprived` is 244 (HateMM) against MAC's 243: the
apportionment quota boundary sits at 0.466774 / 0.466907, a gap of 1.3 × 10⁻⁴, so the float32-vs-float64
vote flips one item across the `mult == 0` line. The head (223 items) and every headline figure are
unaffected. (ii) HateMM full-bank LOO here is 0.8495 on n = 744 against `LITSWEEP8_PATHOLOGY_MATCH.md:116`'s
0.8493 — that record discloses dropping one zero-norm row (n = 743); 632/744 vs 631/743. Reconciled, no
error on either side.)*

### 1.3 …but the interpretation does not survive. Three placebos kill it.

The re-derivation added the control arms that the original recon did not run.

**Placebo A — the generic LoRA, which had no curriculum at all, moves the targeted head MORE.**

| arm contrast (LOO error) | HateMM head | MHC-ZH head | deprived |
|---|---|---|---|
| **generic − φ0 (frozen)** — *no curriculum anywhere* | **−0.0762** | **−0.1207** | 0.0000 |
| curric − generic — *the claimed curriculum effect* | −0.0538 | −0.0402 | 0.0000 |

"A targeted move on the confusable head with zero on the deprived tail" is **not a curriculum
signature**. It is what *any* encoder adaptation produces in this arena, and the plain generic LoRA
produces a larger one. The claimed shape has no discriminating power.

**Placebo B — the deprived-tail zero is a tautology, not a measurement.**

The deprived tail is *defined* as `mult == 0` ⟺ smallest curriculum weight ⟺ smallest `c` ⟺ **largest
|vote|**. Measured: |v| in the deprived tail runs 0.7565–0.9592 (HateMM) and 0.5558–0.9819 (ZH), against
a head that runs 0.0000–0.3617 / 0.0004–0.2211. These are saturated votes.

> **Predictions are bit-identical across all four arms on 244/244 = 100.00 % of the HateMM deprived
> tail and 193/193 = 100.00 % of the ZH deprived tail.** The deprived error *set* is identical
> (pairwise Jaccard 1.0000; 19 items HateMM, 4 items ZH).

The "0.0000 on the deprived tail" therefore carries **zero information about the curriculum**. Every
arm scores exactly 0.0000 there, including the frozen encoder that received no training at all. It was
relayed upward as evidence of targeting; it is an artefact of the stratum definition.

**Placebo C — the second draw of the same curriculum has the opposite sign.**

`train_…-LoRA-curric-rep2_HF.pt` is the banked draw-2 of the *identical* recipe. This is the correct
placebo and its provenance is airtight: `CAND2_REP2_PREREG.md:19` — *"a second, independent
curriculum-SFT draw (a different SFT seed over the **byte-identical curriculum multiset**)"* — and
`:100`, the config *"differs from draw-1's config C by exactly `seed: 1` + rep2 `output_dir`; every
other line byte-identical"*, with `:103` recording the G-repro proof that the builder re-emits
`train_curric.json` bit-exact. Same curriculum, same data, same steps; one seed apart.

| HateMM head LOO error | value | vs generic |
|---|---|---|
| generic | 0.3049 | — |
| curric draw-1 | 0.2511 | **−0.0538** |
| **curric draw-2 (rep2)** | **0.3139** | **+0.0090** |

The −0.0538 is a **single-draw** quantity whose same-recipe replicate moves the head the **wrong way**.
The honest 2-draw HateMM head figure is **−0.0224**, and the honest 2-draw all-item figure is
**−0.0068 error / +0.0068 acc**, not −0.0202 / +0.0202. Under this project's own replication discipline
(F-R0.9: a 2-draw estimate is "survived one independent replication under-bar", never "proven stable"),
a single-draw −0.0538 with a +0.0090 replicate cannot be certified as anything.

### 1.4 The "tenth law-I datum" label is wrong in kind, not only in count

Law I is *better signal without conversion* (`research-wiki/DRAFT_analysis_chapter.md:178`). Applying
its own test to this cell, using the honest 2-draw figures on both sides:

| dataset | train-arena Δacc (2-draw where available) | deployed Δacc | converted? |
|---|---|---|---|
| HateMM | **+0.0068** (2-draw; +0.0202 draw-1 alone) | **+0.0132** (F59 pooled 2×3, 5/6 sign) | **YES — fully, and then some.** The arena *understates* the test gain by ~2×. |
| MHC-ZH | +0.0086 (single draw; below the campaign's own +0.010 bar) | TIE | there is no above-bar train-side gain to fail to convert |

On HateMM the representation gain **converted**; on ZH there is no certifiable train-side gain in the
first place. Neither leg is *better signal without conversion*. **The cell is not a law-I datum, and
"tenth law-I datum" is retracted in full.** The certified count is nine (§2) and this cell does not
change it.

### 1.5 A minor but real forensic by-product: a degenerate row is the curriculum's most-upweighted item

HateMM train row 355, id **`hate_video_95`** (label 1), has **zero-norm image *and* text features in
all four banked caches** (φ0, generic, curric, curric-rep2). Under the deployed rule it gets `v = 0`
⟹ `pred = 1` ⟹ scored **correct** in every arm, and `c = exp(0) = 1.0` — the **maximum possible
confusability** — so cand-2's mining put it in the targeted head and gave it **4 copies**, the joint
maximum. A feature-extraction failure was the single most-upweighted item in the curriculum. It is
constant-correct in every arm, so it cancels exactly in every difference above and changes no number
here; it is recorded because it is a live defect in the mining diagnostic and because
`LITSWEEP8_PATHOLOGY_MATCH.md:115` independently found and disclosed the same row from the other side.

### 1.6 What should be relayed instead — the result that *does* survive

Litsweep-7's §L4 (`LITSWEEP7_LANDING_SITE.md:509-566`) proposes **CURDIAG** as rank-4 candidate: run
the frozen `mechfix_ops.deployed_vote` over the generic and curriculum train caches and report the
paired train-arena Δ, because *"the one lever that converted on the training side … has never been
measured in that arena"* and *"if the correlation is ~0 or negative, the campaign's 0-for-25 base rate
is partly an artifact of the instrument."* It budgets **$0, ~1 hour CPU**, and **P(it changes how the
campaign reads its own kills) ≈ 30 %.**

**That measurement has now been made** — by the MAC recon, before the proposal was written, and
re-derived independently here. Litsweep-7's three declared outcomes and the answer:

1. *Train-arena Δ positive and ordered like the test Δ (HateMM > ZH ≈ 0)* → **THIS IS THE OUTCOME.**
   HateMM +0.0068 (2-draw) / +0.0202 (draw-1) train-arena, +0.0132 test; ZH +0.0086 train-arena, TIE
   test. Same sign, same ordering, HateMM ≫ ZH on both sides.
2. *Train-arena Δ ~0 while test Δ is +0.0132 — "the outcome that would reopen the box"* → **NOT
   observed.** The arena sees the gain.
3. *Train-arena Δ negative while test Δ positive* → **NOT observed** (this is what the retracted
   framing asserted).

**Consequence, on the cell CURDIAG asked about: the instrument is valid, the campaign's $0-pregate
base rate stands, and no confidence discount is owed on that ground** — but see §1.7, which finds the
opposite on a second axis and is the half that matters more. CURDIAG's curriculum leg is
**discharged at $0** and need not be run. One caveat travels with it: on the matched 2-draw comparison
the arena reads **+0.0068** against a test **+0.0132**, so it **attenuates** by roughly 2× — valid in
sign, conservative in magnitude. A kill decided by an arena margin under ~0.007 should not be treated
as decisive on that ground alone.

### 1.7 …but a second axis, which CURDIAG did not ask about, inverts. This is the real caution.

Running the same frozen operator over the frozen-vs-generic-LoRA pair on **all three** datasets (full
bank, deployed rule, `<scratchpad>` re-derivation):

| dataset | frozen Qwen | generic LoRA | **train-arena Δ** | test-side conversion |
|---|---|---|---|---|
| HateMM | 0.8065 | 0.8293 | **+0.0228** (smallest) | **converts** — the one dataset where encoder identity ever converted (§3.9) |
| MHC-ZH | 0.7927 | 0.8480 | +0.0553 | marginal (B3/F45: final-ep PASS marginal, val-sel FAIL) |
| MHC-EN | 0.7687 | 0.8415 | **+0.0729** (largest) | **closed at all three levels** (`findings.jsonl:55`, F55) — and the deployed EN configuration is consequently the **frozen** encoder (`scripts/analysis/mechnov_pairverify.py:84`) |

**The train arena's cross-dataset ordering of the encoder-adaptation move is EN > ZH > HateMM. The
test-side conversion ordering is the exact reverse.** The dataset where the arena sees the *largest*
representational gain is the dataset that converts *least* — indeed the one the campaign closed and
whose deployed encoder is the un-adapted one.

So CURDIAG's question gets a **split answer**, and only the favourable half was in view when the MAC
result was relayed:

* on the **curriculum** cell (the one CURDIAG names): outcome 1, arena valid, mildly attenuating;
* on the **encoder-adaptation** axis (which it does not name): the arena is **anti-correlated across
  datasets**. A large train-arena Δ is not evidence of test conversion, and on EN it is
  counter-evidence.

This is exactly law I restated inside the arena, and it is the honest reason the $0 pregates are run
as *kill* gates rather than as *promote* gates: a negative in the arena is informative, a positive is
not. Nothing here reopens the box; it tightens how arena positives may be read.

*(Provenance note on this table, in the spirit of the standing rule: my first pass computed the EN row
from `train_…-LoRA_HF.pt` and reported a "deployed" EN train-LOO of 0.8415. That is the wrong cache —
`mechnov_pairverify.py:84` declares MHC-EN's deployed model as the **frozen** `Qwen2.5-VL-7B-Instruct_HF`.
Caught by spot-checking against `LITSWEEP8_PATHOLOGY_MATCH.md:116`'s 0.7687, which the corrected run
then reproduces exactly. Recorded because the audit is subject to its own rule.)*

### 1.8 A cross-cutting claim in two of today's in-flight records — INDEPENDENTLY CONFIRMED

Both `VSW_ASYMMETRY_RECON.md:365-372` (sign-only vote) and `LITSWEEP8_PATHOLOGY_MATCH.md:125-131`
(label-majority-only vote) assert that the cosine magnitude is decision-irrelevant. Re-derived here
from the frozen operator on the deployed caches, full bank:

| dataset | deployed | cosine replaced by constant 1 | Δ | decision agreement |
|---|---|---|---|---|
| HateMM (n=744) | 0.8495 | 0.8495 | **+0.0000** | 0.9946 |
| MHC-ZH (n=579) | 0.8480 | 0.8480 | **+0.0000** | **1.0000** |
| MHC-EN (n=549) | 0.7687 | 0.7687 | **+0.0000** | **1.0000** |

**Confirmed, and stronger at full bank than either record claims** (their 5-fold pooled protocol gives
−0.0013 / +0.0000 / −0.0018; at full bank all three are exactly zero, and ZH and EN agree on every
single item). Both records' EN and HateMM anchors also reconcile: EN 0.7687 exact, HateMM 0.8495 on
n = 744 vs LITSWEEP8's 0.8493 on the disclosed n = 743. No provenance defect in either.

---

## 2. ITEM 2 — THE CERTIFIED LAW-I COUNT

**Authoritative count: NINE.**

Law I, as defined at `research-wiki/DRAFT_analysis_chapter.md:178`:

> *Structural law I — **better signal without conversion**. … In each instance a candidate signal is
> demonstrably richer than the pipeline already has, and yet the best in-constraint operator converts
> **none** of it into main-table accuracy. Each shares a sharp form — a gold/label oracle proves the
> convertible headroom is present, but no unsupervised, frozen, or even supervised operator inside the
> constraint box recovers it.*

### 2.1 The enumeration

| # | instance | finding | ordinal source |
|---|---|---|---|
| 1 | **P3** — evidence-density pooling | — | `findings.jsonl:50` (F50) names the list: *"5th better-signal-no-conversion instance (P3, S2S F37, W2-A F42, router F47, FA F50)"* |
| 2 | **S2S** — Qwen frame-group set-matching | F37 | same list, F50 |
| 3 | **W2-A** — transcript-grounded vision key | F42 | same list, F50 |
| 4 | **Router** — per-item cross-channel selection | F47 | same list, F50 |
| 5 | **FA** — modality-fusion / cross-encoder composition | F50 | `findings.jsonl:50` — *"5th better-signal-no-conversion datum"* |
| 6 | **Premise-(d)** — healthy-image ⊕ adapted-text, EN | F55 | `findings.jsonl:55` — *"6th no-conversion datum"* |
| 7 | **LP** — label propagation over the kNN memory graph | F63 | `findings.jsonl:92` (F91) — *"F63 = 7th"* |
| 8 | **Vision-unfreeze LoRA** — ViT tower + projector inside LoRA-SFT | F65 | `findings.jsonl:66` — *"(8th law-I instance)"* |
| 9 | **Molmo2-8B encoder swap** | F91 | `findings.jsonl:92` — *"THIS IS THE 9TH CERTIFIED LAW-I DATUM … ledger count moves 8 → 9"*; record self-label `MOLMO2_PROBE_RECORD.md:104-107` |

**Deliberate non-instances (so the count cannot drift), all three still correct:**

* **F87 / MokA** — explicitly **declined**: *"KS-MOKA-3 NULL-OP … 9th law-I NOT certified"*
  (`findings.jsonl:88`); the image stream read AMBIGUOUS, not MOVED. F91 is a different cell that
  earns the ninth slot on its own measurement.
* **LAUD** (round-5 learned-audio gate) and **CLAP** (round-8) — redundancy nulls with **no oracle
  surplus**, so they fail law I's own defining clause (`DRAFT_analysis_chapter.md:277-281`).
* **F95 / MECHNOV pair-verification** calls itself *"the sharpest instance of law-I yet recorded"*
  (`MECHNOV_PAIRVERIFY_PREGATE.md:436-437`) **without claiming an ordinal**, and
  `DRAFT_analysis_chapter.md` records it as *"the sharpest measurement of the law … not one of the
  nine."* Correct as written — **not certified, and it must stay that way**, otherwise the count moves.
* **VSW / `VSW_ASYMMETRY_RECON.md`** — law-I-*shaped* recon-grade material, **not certified**, no
  ordinal claimed. Correct as written.
* **cand-2 curriculum** (§1) — **not law-I-shaped at all**; it converted.

### 2.2 The reconciliation is already institutionalised

Commit `b4800d7` (2026-07-28 06:25) carries the 8 → 9 move through the wiki with an explicit audit
trail — *"LAW-I COUNT RECONCILED ONCE AND USED CONSISTENTLY: F63 = 7th, F65 = 8th, …"* — and the diff
updates `DRAFT_analysis_chapter.md` §3.6 in all six places (heading, running total, new bullet,
"nine-times-repeated", "nine cells", "nine instances"). Verified by reading the diff. **The paper docs
are correct and needed no repair.**

### 2.3 Who currently disagrees

**Correct as they stand (no edit):**

| location | says | why it is fine |
|---|---|---|
| `research-wiki/DRAFT_analysis_chapter.md:178, 326, 351, 1051` | nine | authoritative |
| `research-wiki/DRAFT_experiments_chapter.md:972-974` | nine, with the F87/F95 guards spelled out | authoritative |
| `research-wiki/DRAFT_experiments_chapter.md:908-910` | *"stays at **eight**"* | **correctly scoped** — it is the round-**7** consistency note, and the very next sentence is *"Round-8 update: … so the current count is nine."* Historically accurate, inline-updated. |
| `refine-logs/MOLMO2_PROBE_RECORD.md:104-107` | "9th law-I datum" | agrees |
| `refine-logs/LITSWEEP7_LANDING_SITE.md:73` | *"the certified law-I count is **nine**, not ten"* | **right, and right for the right reason** (it cites F91's own 8→9 reconciliation) |
| `refine-logs/LITSWEEP6_MEMBANK.md:356` | *"C2 becomes the **tenth** law-I datum"* | **conditional and correct** — a future-tense "if the accuracy does not move" clause, which presupposes nine |
| `MOKA_VERDICT_REVIEW.md:414`, `NCA_PREREG.md:32,91`, `NCA_PREREG_REVIEW.md:158`, `LITSURVEY_NOVEL_MECHANISMS.md:18,149`, `LITSWEEP2_HEAD_OBJECTIVES.md:18` | eight / "8+" | **historical records, correct at write time** (all pre-date F91). Rewriting them would falsify the record; they are frozen prereg/verdict/survey artefacts. |

**Actually wrong (one document, and it is not mine to edit):**

> `refine-logs/LITSWEEP8_PATHOLOGY_MATCH.md:97` and `:604` —
> *"'Ten certified law-I data' over-counts the ledger, **which certifies eight** (F50 '5th', F63
> 'SEVENTH', F65 '8th', F87 '9th law-I NOT certified'). Fix the count before it reaches the paper."*

Litsweep-8 is **right that ten is an over-count and right to demand a re-derivation**, but its own
enumeration stops at F87 and **never reaches F91**, which certifies the ninth explicitly
(`findings.jsonl:92`). **The correct figure is nine, not eight.** Related: `:77`'s *"our ten-odd law-I
data"* should read *"our nine law-I data"* — the campaign has spent a round making this count exact and
"ten-odd" gives it back.

`LITSWEEP8_PATHOLOGY_MATCH.md` is an **in-flight file owned by another agent** and is on this audit's
read-only list, so it is **flagged, not edited**. The correction needed is two words in each of three
places (`:77`, `:97`, `:604`): *eight* → *nine*, *ten-odd* → *nine*, adding `F91` to the enumeration.

**Net: no document required repair by this audit.** The paper docs were already correct; the single
wrong document is in another agent's hands and is flagged above. The count is **nine**, and §1 removes
the only live candidate for a tenth.

---

## 3. ITEM 5 — THE 2026-07-13 SAV "CO-OCCURRENCE" IS A SUBSTRING-GREP ARTEFACT

Litsweep-7 wrote (`LITSWEEP7_LANDING_SITE.md:66`): *"`−0.0538` and `−0.0402` co-occur in exactly one
file, `refine-logs/SAV_F0_EXECUTION_RECORD.md:150,161` — an unrelated 2026-07-13 SAV confidence-interval
table."* Read at source, the three lines that match are:

| line | text that matched | what it actually is |
|---|---|---|
| 149 | `81.0402` | HateMM **C-sparse@40 `L̄_arm`** — a bits value. `"0402"` is digits 3-6 of `81.0402`. |
| 150 | `−.405384` | HateMM **U-1 `ΔL` mean**. `"0538"` is a substring of `405384`. |
| 161 | `−.0538` | MHC-ZH **SAV@40 `Δacc` mean**, CI `[−.1179,.0077]`, not significant. |

**There is no `−0.0402` in that file at all**, and the one real `−.0538` is a ZH Δacc from a different
experiment, a different object and a different dataset from the HateMM head figure it was matched to.
The pairing was produced by digit-substring matching, not by co-occurrence of two quantities.

**Verdict: a genuine coincidence — in fact not even that, a grep artefact.** It is **not** evidence
that the MAC recon lifted its numbers from the wrong table. §1.2 settles that positively: the numbers
were computed this session, in the correct arena, with the frozen operator, and re-derive exactly.

*(No fault attaches to litsweep-7 here. Its search was correct in scope and its `+0.0132` verification
was right; the substring hit is the standard failure mode of `grep`-ing bare decimals, and it is worth
institutionalising: **search for decimals with a leading-boundary anchor** — e.g.
`grep -nE '(^|[^0-9.])0\.0538'` — or the next audit will re-manufacture this.)*

---

## 4. ITEM 3 — THE F97 PARITY ERRATUM: BLAST RADIUS

**Scope note.** The VSW agent's erratum **has landed**, at `VSW_PREGATE_RECORD.md:514-580` (§4.3),
and it is thorough. Nothing here duplicates it and nothing in that file was touched. This section is
the blast radius only: cause, downstream, and what changes.

### 4.1 What fails, and by how much

`scripts/analysis/mechnov_pairverify.py` (sha `77b0defd…b7240d`), re-run **unmodified** on the same
node, same env, same caches, same seeds:

* **Every closed-form quantity reproduces bit-exactly at 4 dp** — `acc_deployed`, `acc_cos_shape`,
  the deployed floors, the per-fold PCA explained-variance sequences, the fitted-pair counts.
* **The torch-fitted MLP arm fails on 44 of 48 trained quantities** (15/16 HateMM, 16/16 ZH,
  13/16 EN). Magnitude is small per quantity (`acc_mlp_max` HateMM 0.8401 → 0.8468, ≈ 5 items on 744;
  pooled `auc_mlp` 0.7753 → 0.7747) but **not sign-preserving**.

### 4.2 Non-determinism or defect? — **NON-DETERMINISM, and the harness is exonerated**

Independently checked here, and it agrees with the VSW diagnosis:

| candidate cause | evidence | verdict |
|---|---|---|
| unseeded randomness | `mechnov_pairverify.py:171` `torch.manual_seed(0)`; `:177` `np.random.RandomState(0)` for the batch permutation; `:210` `StratifiedKFold(..., random_state=0)`; `:227` `PCA(..., random_state=0)`; `:233` per-fold `RandomState(0+fold)`. **Every source is pinned.** | **exonerated** |
| thread count unset / drifting | `:413` `torch.set_num_threads(8)` — set explicitly in the script; VSW measured `∈ {1,4,8}` all give fold-0 `acc_mlp_max = 0.7987` | **exonerated** |
| within-process nondeterminism | VSW: fitting twice on identical inputs gives `max|Δscore| = 0.0` | **exonerated** |
| call ordering / faiss-numpy interaction | VSW: fit before vs after `deployed_vote` gives `max|Δscore| = 0.0` | **exonerated** |
| the §3.2 efficiency deviation | VSW: nominated-pairs-only vs full-matrix-then-index give identical fold-0 predictions | **exonerated** |
| library drift | package dirs unchanged since 2026-03-27; torch 2.6.0+cu124 / numpy 1.26.4 / MKL 2024.2 identical | **exonerated** |

**Residual cause: cross-session CPU GEMM kernel dispatch (oneDNN/MKL) on the 256-core EPYC.** The
mechanism is not mysterious and is worth naming because it determines the fix: `fit_mlp` runs
`MLP_EPOCHS` × mini-batches of **Adam**, which is chaotic in the last bits — a 1-ULP GEMM difference
at step 1 compounds into a visibly different minimum by the end. **This is a property of training a
net on CPU, not a bug in the script, and no amount of additional seeding repairs it.** The only
repairs are (a) bank the fitted parameters/predictions alongside the JSON, or (b) gate against a
same-session re-run — which is exactly the rule VSW already wrote.

### 4.3 Does any VERDICT change? — **NO. Both KILLs stand.**

| conclusion | MLP-derived? | verdict changes? | why |
|---|---|---|---|
| **F95 CONTROL-1** — relational supervision buys a better relation scorer (pair-AUC 0.5843→0.7753 / 0.5123→0.7748 / 0.5057→0.7009) | yes | **no** | passes by **4.3–8.8×** the bar on 18/18 cells; measured drift is **0.0006**, four orders under the margin |
| **F95 CONTROL-2 primary** (4 cells) | yes | **no** | today's anchor still fails all four (+0.0027 / −0.0328 at max aggregation on the two primary datasets) |
| **F95 CONTROL-2b** — the shape cost −0.0417 / −0.0293 / −0.0437 | **no** — `acc_cos_shape` is closed-form | **no** | reproduces bit-exactly 3/3; this is F95's load-bearing term and it is immune |
| **F95 KILL verdict** | — | **NO** | rests on 2b + the primary cells, both intact |
| **F95's "0 of 36 cells" headline count** | yes | **THIS DOES NOT REPRODUCE** | one cell now clears: HateMM × fused × MLP × **mean-top-3**, +0.0107, on a secondary aggregation, 8 items on 744, a hair over the bar |
| **F97 "78/78 parity PASS"** | 15 of 78 are trained | **claim does not re-assert today** | true when made; fails on the HateMM emitter's trained cells |
| **F97 / VGA-VNQ KILL verdict** | — | **NO** | K-VGA-3, the decisive bar, is a **within-session relative** comparison between two feature families — exactly the class of claim this drift cannot touch |

**Net: zero verdicts move. One headline *count* is session-dependent and must be re-worded.**

### 4.4 What will need editing once propagation starts (flagged, NOT edited)

Per tasking, no F95/F97 record or downstream table was edited by this audit. The sites are:

| site | what it says | required change |
|---|---|---|
| `research-wiki/PAPER_MASTER_TABLES.md:551` | `CONTROL-2 端到端 36 个 cell 无一通过(0/36)`; primary `0.8401 / 0.8014 / 0.7650`; exchange rates `0.9474 / 0.5345 / 0.8596` | replace the count with **"0 of 4 primary cells"**; mark the MLP-derived numbers session-dependent |
| `research-wiki/DRAFT_analysis_chapter.md:984, 987` | pair-AUCs; *"Control 2 is then cleared by **0 of 36** end-to-end cells"* | same; §984's AUCs are safe (margin ≫ drift) but should carry the session note once |
| `research-wiki/DRAFT_experiments_chapter.md:952` | *"Control 2 is cleared by **0 of 36** end-to-end cells"* | same |
| `refine-logs/MECHNOV_PAIRVERIFY_PREGATE.md:298-330, 436-437, 454` | the F95 record's own §3.2 / §3.2b tables | erratum pointer to `VSW_PREGATE_RECORD.md:514-580` |
| `refine-logs/VGA_PREGATE_RECORD.md:197` | *"**78/78 gates PASS**"* | erratum note: true when made, does not re-assert; conclusions unaffected |
| `refine-logs/LITSWEEP6_RELGEN.md:43` | ZH exchange rate `0.5345` | session-dependent (now 0.6275) |
| `refine-logs/RDK_FORENSIC_RECON.md:34-36` | **written today**; quotes the three pair-AUCs from `MECHNOV_PAIRVERIFY_PREGATE.md:268-270` | safe (margin ≫ drift); add the session note when the erratum propagates |
| `autoresearch/goal_mllm_plus3/state/findings.jsonl:96` (F95), `:98` (F97) | append-only ledger rows | **erratum row appended, never rewritten** |

**The one sentence that should be institutionalised** (VSW wrote it; it deserves to leave that file):
*any record gating against F95's MLP arm must gate against a same-session `--stage anchor` re-run,
never against the recorded JSON; and any **count** of F95 cells crossing a threshold is
session-dependent.*

---

## 5. ITEM 4 — SPOT-CHECK OF TODAY'S LOAD-BEARING NUMBERS

Method: a number counts as **VERIFIED** only if re-read in a primary artefact (`*_OUT.json`,
`*_OUT.log`, a `data/gt/*` file, a banked cache) or re-derived here. A second `.md` repeating it is
not verification. Train split only.

| # | number | claimed at | primary source | re-derived | verdict |
|---|---|---|---|---|---|
| 1 | ZH deployed train-LOO **0.8480** (the campaign's most-reused anchor) | `MECHNOV_PAIRVERIFY_PREGATE.md:299`, and 12 further sites | banked `MHC_zh/train_…-LoRA_HF.pt` + frozen `mechfix_ops` | **0.8480** | **PASS**, exact |
| 2 | HateMM deployed train-LOO **0.8493** | `LITSWEEP8_PATHOLOGY_MATCH.md:116` | same, n = 743 (one zero-norm row dropped, disclosed) | 0.8495 at n = 744 / 631 of 743 | **PASS** — reconciled, both correct at their stated n |
| 3 | MHC-EN deployed train-LOO **0.7687** | `LITSWEEP8_PATHOLOGY_MATCH.md:116` | banked `MHC/train_…-Instruct_HF.pt` (frozen — the deployed EN encoder, `mechnov_pairverify.py:84`) | **0.7687** | **PASS**, exact |
| 4 | cosine magnitude is decision-irrelevant: Δ ≈ 0, agreement ≥ 0.996 | `VSW_ASYMMETRY_RECON.md:365-372`; `LITSWEEP8:125-131` | re-derived, full bank, all 3 datasets | Δ **+0.0000** ×3; agreement 0.9946 / **1.0000** / **1.0000** | **PASS**, and stronger than claimed |
| 5 | cand-2 targeted head **−0.0538 / −0.0402**, deprived **0.0000** | relayed upward (not in any repo doc) | banked caches + frozen operator | **−0.0538 / −0.0402 / 0.0000** | **arithmetic PASS; interpretation RETRACTED** (§1.3) |
| 6 | deployed **+0.0132** (HateMM pooled 2×3, 5/6 sign) | `findings.jsonl:59`; relayed | `CAND2_REP2_VERDICT_REVIEW.md:120-121` — sum 0.0790 / 6 = **+0.01317** | arithmetic re-checked | **PASS** |
| 7 | deployed **"0.0000"** (ZH) | relayed upward | `CAND2_VERDICT_REVIEW.md:139-140,145` — val-sel mean **−0.0067** (1/3), final-ep **+0.0067** (2/3), *adjudicated* **TIE** | — | **FAIL — fabricated companion metric.** "0.0000" is a verdict label rendered as a measurement. The correct rendering is "TIE (−0.0067 / +0.0067)". |
| 8 | F102 erratum: ZH transcript median "4 words" is a whitespace artefact, **true median 106 chars** | `findings.jsonl:103`; `TVB_FORENSIC_RECON.md` | `data/gt/MHC_zh/train.jsonl`, n = 579 | whitespace-token median **4.0**; char median **106.0** (CJK-char median 69.0) | **PASS**, both halves exact |
| 9 | F103 OCR train coverage **72.6 / 65.2 / 90.3 %** | `findings.jsonl:104`; `OCR_FORENSIC_RECON.md:149-151` | `/data/jehc223/baselines/MoRE/data/{HateMM,MultiHateClip/{en,zh}}/ocr.jsonl` × `data/gt/*/train.jsonl` | 540/744 = **72.6 %**, 358/549 = **65.2 %**, 523/579 = **90.3 %**; row presence 100 % ×3 | **PASS**, exact |
| 10 | F103 new-hate-surface **17.4 % pos vs 8.7 % neg** (HateMM) | `OCR_FORENSIC_RECON.md:197` | denominators 298 pos / 446 neg | 298 + 446 = 744 ✓ and 298/744 = **0.4005** = the banked HateMM `pos_frac` re-derived in §1 ✓; 52/298 = 17.4 %, 39/446 = 8.7 % ✓ | **PASS**, denominators sound |
| 11 | F101 BSY: local class odds **0.1231** against C2's own bar **1.2** ("10× under") — the whole kill | `findings.jsonl:102`; `BSY_FORENSIC_RECON.md:83, 267-268` | `data/gt/HateMM/train.jsonl`, frozen band definition | all five bands re-derived **bit-exactly**: 73/8 → 0.1096 → **0.1231**; 188/55 → 0.4135; 136/52 → 0.6190; 217/111 → 1.0472; 130/72 → **1.2414**. Bands sum 744 ✓, hate sums 298 ✓ | **PASS**, exact |
| 12 | F101 banding is by **whitespace words** | `BSY_FORENSIC_RECON.md:81` | — | the table is on **HateMM** (English), where whitespace tokenisation is valid | **PASS** — checked specifically because this is the third place today a whitespace median could have bitten; it is not one |
| 13 | F100 EUM: "dilution premise MEASURED FALSE on HateMM" — median hate-span coverage **0.8289**, single contiguous **0.7416**, coverage < 0.5 only **0.2181** | `findings.jsonl:101`; `EUM_FORENSIC_RECON.md:163-168` | `data/gt/HateMM/hate_spans.json` (1 083 entries) ∩ `train.jsonl` label == 1 | 298/298 matched, 0 missing; median **0.8289**, mean **0.7174**, single **221/298 = 0.7416**, < 0.5 **65/298 = 0.2181** | **PASS**, all four exact |

**Two defects found, one material.**

1. **MATERIAL — row 7, the "0.0000".** A *TIE adjudication* over two non-zero measured means
   (−0.0067 val-sel, +0.0067 final-epoch) was relayed as a measured **0.0000**. This is precisely the
   failure the standing rule names — *never fabricate companion metrics* — and it did load-bearing
   work: paired against "+0.0132 on HateMM", the fabricated zero manufactured the clean
   "converts on one dataset, exactly nothing on the other" shape that made the cell look law-I-like.
   With the real numbers the shape dissolves. **Retract and re-render as "TIE".**

2. **COSMETIC — `OCR_FORENSIC_RECON.md:149-151` char statistics.** The median/max character figures
   are systematically low by 1–2 against the raw jsonl (median 270/246/498 vs 272/247/499; max
   101,982/7,757/19,092 vs 101,983/7,758/19,093 — an exact +1 on all three maxima, i.e. trailing-
   whitespace handling). The **coverage percentages, which are the load-bearing numbers, are exact**,
   and no conclusion moves. Recorded for completeness, not for repair.

*(A third, procedural: the audit's own first pass mis-selected the MHC-EN cache and would have
published 0.8415 as an EN "deployed" figure. Caught by spot-checking against a primary record before
writing. Disclosed at §1.7.)*

**Coverage, stated honestly.** Two of today's title-level numbers were **not reached** by this pass and
remain **UNVERIFIED, not verified-clean**: F99/RDK's *"pure-permutation oracle"* arithmetic bound
(`RDK_FORENSIC_RECON.md`) and F103's *"MoRE rerun lands 5.6-8.7 acc below us on all three datasets"*
(`OCR_FORENSIC_RECON.md:428-432`), the latter resting on an external baseline tree
(`/data/jehc223/baselines/MoRE/`) rather than on our own artefacts. Both sit in finding **titles**,
which is the highest-risk position — titles are what get relayed. They should be the first two checks
of the next pass.

---

## 6. CORRECTIONS MADE, AND CORRECTIONS OWED

**Made by this audit:** none were required in the paper docs — §2.2 shows commit `b4800d7` had already
carried the law-I count correctly, and §2.3 shows every remaining "eight" is either correctly scoped
in-place or a frozen historical artefact that must not be rewritten. **The corrections this audit
produces are retractions of things relayed in conversation, which is what this document exists to
bank.**

**Retracted, with the retraction to travel wherever the claim went:**

> **RETRACTION 1.** The cand-2 curriculum is **not** the tenth certified law-I datum. It is not a
> law-I datum at all: on HateMM the train-side gain **converted** (+0.0068 arena, 2-draw → +0.0132
> test). The certified count is and remains **nine**.
>
> **RETRACTION 2.** The deprived-tail **0.0000** is not evidence of curricular targeting. Predictions
> are bit-identical across **100.00 %** of that stratum in every arm including the untrained frozen
> encoder; the zero is forced by the stratum's definition.
>
> **RETRACTION 3.** The deployed ZH figure is **not 0.0000**. It is a **TIE** adjudicated over
> −0.0067 (val-sel) / +0.0067 (final-epoch).
>
> **NOT retracted:** the arithmetic **−0.0538 / −0.0402** itself, which re-derives exactly under the
> frozen operator with the F95 parity gate passing at 4 dp — but it is a **single-draw** quantity
> whose same-recipe replicate is **+0.0090**, and the honest 2-draw HateMM head figure is **−0.0224**.

**Owed by others, flagged not edited:**

* `refine-logs/LITSWEEP8_PATHOLOGY_MATCH.md:77, 97, 604` — law-I count reads *eight* / *ten-odd*;
  should read **nine** (its enumeration stops at F87 and never reaches F91). Committed at `2e2805f`;
  three two-word edits (§2.3).
* The F95/F97 propagation set in §4.4 — awaiting whoever propagates the VSW erratum.

---

*Audit performed 2026-07-28. Re-derivation scripts: `<session scratchpad>/audit_mac_rederive.py`,
`audit_mac_diag.py`. Frozen operator `scripts/analysis/mechfix_ops.py` sha256
`635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d`, used unmodified. 0 GPU, $0,
no test contact.*
