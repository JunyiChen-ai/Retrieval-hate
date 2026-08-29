# Gate-0 Reopen — 2026-07-31

**Trigger.** `TARGET_STATE.json::registry_update_2026_07_28.serial_execution.fast_fail`:
*"After two consecutive active-candidate failures, reopen Gate 0 before continuing
the ordered backlog."* C01 (`KILL_CURRENT_ENDPOINT_ROUTE_ONLY`, job `13738`,
2026-07-29) and C02 (`KILL_C02_DENSITY_ORBIT_UNREACHABLE`, job `13847`,
2026-07-31) are those two measured failures.

**Scope.** This reopen governs the post-C04 backlog only. **The C04 lineage was
not read, touched or referenced beyond its registry status string**, and nothing
in C02 was modified. Cost: `$0` — zero GPU, zero SLURM, zero Modal, zero teacher
call, zero model load, zero cache write, zero test-split contact.

**Provenance of every measurement in this record**, restated after round-5 review
because the earlier attestation named only the first item: (i)
`data/gt/{HateMM,MHC_zh,MHC}/{train,val}.jsonl`, the census layer; (ii)
`/data/jehc223/Multihateclip/{English,Chinese}/annotation(new).json` — the
per-dataset title/transcript **medians** in §4.2 are **join-scoped to train+val
ids**, while the `Title`-**presence** counts (`891/891`, `897/897`) and the
`<em>`-location corroboration (`391` Title / `0` Transcript) were taken over **all
rows**, reading only the `Title` and `Transcript` fields and consuming **no label**.
Join-scoped those figures are `629/629`, `657/657` and `277 / 0` — and the C08
disposition is unaffected, because the join-scoped counts make the identical point; (iii) recomputation from the already-banked
result artifacts `artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`
(every C01 accuracy re-derived from its stored confusion matrices) and
`artifacts/c02_edq/v1/a0/C02-A0-v9/C02_A0_{OUT,DECISION}.json`; and (iv) the six
banked `scripts/analysis/headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`, which
supply the fold-head floor values behind §3.7's correction. All four are `$0` and permitted; none is a new experiment; **no label or
prediction was read for any test id.**

**Evidence package under adjudication.**
`refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` — advisory only, creates no
registry status by itself. It recommended: strike C07/C08/C10/C11/C12/C13/C14,
hold C05, gate C06 behind a `$0` falsifier, promote C09.

**Outcome in one line.** The recon's numbers are sound — every `[M]` figure
re-measured independently reproduces, and no fabricated number was found anywhere
in it — but **six of its seven recommended strikes rest on a ban, a screen, a
premise or a boundary clause read wider than its own text, and are downgraded to
HOLD.** Exactly **one** survives fourteen rounds of adversarial review — C14, and only
because that candidate was already registry-ineligible before the recon was
written.

**Review history.** Fourteen rounds of fresh independent review, closing at **`GO (0C/0H/0I)`**:
`REVISE (1C/3H/10I)` → `REVISE (0C/3H/7I)` → `REVISE (1C/1H/4I)` →
`REVISE (0C/2H/2I)` → `REVISE (0C/2H/2I)` → `REVISE (0C/1H/2I)` →
`REVISE (0C/2H/3I)` → `REVISE (0C/2H/4I)` → `REVISE (0C/1H/3I)` → `REVISE (0C/1H/3I)` → `REVISE (0C/1H/2I)` → **`GO (0C/0H/2I)`** → **`GO (0C/0H/1I)`** →
**`GO (0 Critical / 0 High / 0 Important)`**. Every finding is applied. Round 1's Critical moved **C07**
from strike to hold; round 3's Critical moved **C08**, after locating source
annotation files that refute its premise 1; round 4 confirmed all thirty prior
findings applied on all four surfaces; round 5 re-based C13 on its own registry
text after finding the arithmetic leg tag-scoped, and round 6 established that the
remaining basis is a proponent-satisfiable precondition, moving **C13** to hold as
well. Findings are enumerated in §§8-20.

---

## 1. Verification method

The recon is advisory, so every strike was re-derived from primary evidence
rather than adopted. Four independent verification passes were run, each by a
worker with no exposure to this adjudication's reasoning:

1. **Independent re-measurement** of every `[M]` claim, computed from scratch on
   the six gt label files (no prior document supplied).
2. **Cluster 1** — C05's four comparator gates and C07's claims.
3. **Cluster 2** — the three "measured ceilings" of recon §1, which bound the
   whole backlog and are therefore the most load-bearing claims in the document.
4. **Cluster 3** — C06/C08/C09/C10/C11/C12/C13 evidence, with the explicit
   instruction to hunt for bans applied wider than their written scope.

The C01 arm table (the sole basis for the C06 disposition) was verified directly
against `artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json` by
this adjudicator.

---

## 2. Independent re-measurement — the recon's `[M]` layer reproduces

Recomputed from `data/gt/<DS>/{train,val}.jsonl` with no reference to the recon:

| quantity | recon | re-measured | verdict |
|---|---|---|---|
| gt row key-set, all 6 files | `['id','label','text']` | `['id','label','text']`, single key-set, 0 deviations | **exact** |
| `title` field present anywhere | none | **none** — union of all keys is exactly those three | **exact** |
| HateMM whitespace-only `text` | `39` train / `9` val | `39` / `9` | **exact** |
| MHC-ZH, MHC-EN whitespace-only | `0` throughout | `0 / 0 / 0 / 0` | **exact** |
| MHC-ZH rows with an HTML tag | `243` train / `34` val | `243` / `34` | **exact** |
| rows with an HTML **entity** | `64` MHC-EN train / `9` val; `1` MHC-ZH train; `0` HateMM | `64` / `9`; `1`; `0` — MHC-EN entity **occurrences**: train+val `&#39;` ×51, `&quot;` ×22, `&amp;` ×18; **train-only** `&#39;` ×43, `&quot;` ×16, `&amp;` ×17. Row counts use the strict convention `&[a-zA-Z]+;|&#[0-9]+;`; a hex-inclusive `&#?\w+;` gives **2** MHC-ZH train rows, not 1 | **exact** |
| MHC-ZH tag histogram | `em` 254, `/em` 254, nothing else | `em` 254, `/em` 254; **nothing else**, opens/closes balance | **exact** |
| MHC-ZH hate rate, `<em>` present vs absent | `0.580` (141/243) vs `0.116` (39/336); base `0.311` | `0.5802` (141/243) vs `0.1161` (39/336); base `0.3109` | **exact** |
| same, val | `0.588` (20/34) vs `0.182` (8/44); base `0.359` | `0.5882` (20/34) vs `0.1818` (8/44); base `0.3590` | **exact** |
| distinct keywords / occurrences | `49` / `254` | `49` / `254` | **exact** |
| top keywords | 傻逼 42, 阴茎 20, 娘炮 16, 傻屌 11, 公主病 11 | identical | **exact** |
| markup char fraction: median / max / rows >10 % | `0.0000` / `0.8621` / `203` | `0.000000` / `0.862069` / `203` | **exact** |
| markup char fraction p90 | `0.5155` | `0.5051` (`lower`), `0.5071` (`linear`), **`0.5155` (`higher`)** | **convention, not error** |
| MHC-ZH median text length | `106` | `106` | **exact** |
| `data/CLIP_Embedding` entry counts | 100 / 130 / 71 / 6 / 2 | HateMM 100, MHC_zh 130, MHC 71, ImpliHateVid 6, HateClipSeg 2 | **exact** |
| `data/Archive/` contents | `MHC` and `MHC_zh` only, no HateMM | `MHC`, `MHC_zh` only; **no HateMM of any kind** | **exact** |

**Note M-1 (convention, NOT an error).** The markup-fraction `p90` differs by
percentile convention only, in numpy's vocabulary: `0.5051` (`lower`), `0.5071`
(`linear`), and **`0.5155` (`higher`) — which is exactly the recon's figure**. The
same three labels are used in §7, in `TARGET_STATE.json` and in
`TARGET_FINDINGS.md`. Nothing depends on it. The paper note in §7 states the convention
explicitly rather than picking one silently.

**Note M-2 (convention, NOT an error).** HateMM train median `text` length is
`694.5` interpolated and `696` upper — again the recon's figure under the upper
convention, which it applied **consistently** (MHC-ZH val `111` and MHC-EN val
`443` are also exact upper medians). No median in the recon is wrong; the two
documents differ in convention, not in measurement.

---

## 3. Corrections to the recon — findings that changed a disposition

### 3.1 The coverage bound does not bound re-ordering (recon §1.1, §9)

Recon §1.1: *"the **marginal** oracle of expanding **or reordering** the pool
beyond rank 20 is `≤ 1 − coverage(20)` = **+0.0171** head-space."*

The four cited figures are correct **as the pool-expansion marginal**. They do
not bound re-ordering within a fixed pool, and the project's own erratum says so
in advance — `findings.jsonl` F114: *"WHAT CANNOT BACKFILL IT: K-HC-1's coverage
bound (<=+0.0171) bounds POOL EXPANSION not purity-within-a-fixed-pool, as F107
sec6.1 itself states."* Same statement at `HEADCOV_PREGATE_RECORD.md:325-326`.

The re-ordering channel is measured **far above** the bar:
`LITSWEEP8_PATHOLOGY_MATCH.md:198-201` — *"Rank-constrained variant (keep the
deployed profile `w = [20…1]`, oracle chooses only which items and in what order
— i.e. what a pure re-ranker can do): Δ = **+0.0780 / +0.1123 / +0.1876** at pool
20."* The re-ranking family is closed by **reduction** to the measured-dead
AGGNET/VSW re-weighting family (`LITSWEEP8:200-201`, `:268-273`), **not** by an
under-bar oracle. This is a materially different closure and is reflected in the
corrected strategic finding (§6).

### 3.2 The three ceilings are not independent (recon §1, §9)

- Ceiling 1's head-space leg and Ceiling 3 come from the **same table** —
  `HEADCOV_PREGATE_RECORD.md:419`, epoch-29 row `purity 0.8250 | acc 0.8419 |
  coverage 0.9829`. The `0.9829` yielding "+0.0171 head-space" and the `−0.0171`
  coverage fall in Ceiling 3 are one 90-cell, `n = 78`, MHC-ZH-dev, CPU-re-mint
  proxy-head measurement.
- Ceiling 2 is a **strictly dominated sub-case** of Ceiling 1's function class by
  the source's own words: `LITSWEEP8:200-201` — *"strictly weaker than free
  reweighting at the same pool, so it is dominated by the family AGGNET/VSW
  already measured."*

"Three *independent* measured ceilings" is therefore withdrawn.

### 3.3 "Exchange rate never above 1.17 in 36 cells" is stale (recon §1.2)

Faithful to F99, which cites `MECHNOV_PAIRVERIFY_PREGATE.md:458-459` (the quoted
sentence itself sits at `:463`), but the 36
cells are **F95's pair-verifier battery only** (`3 datasets × 3 spaces × 2 models
× 2 aggregations`). Campaign-wide the rate has exceeded 1.17 repeatedly: F96/D1
`1.8889` and `2.2353`, F98/AGGNET `1.8333`, F105 `6.0000`, F112 `2.8889`. The
recon contradicts itself fourteen lines later in §1.4, where it correctly quotes
ER 6.0 and 2.8889. **This does not disturb F99's caps** — those are zero-break
*upper* bounds and hold a fortiori — but the "which has never happened" reassurance
is withdrawn.

### 3.4 F113's kill-direction is marked NOT ESTABLISHED by F113 itself

`findings.jsonl` **F113**'s own honesty clause, verbatim (F114 does not contain
it; the JSON's `V-4` already attributes it correctly): *"**NOT established: that a
raw-space NEGATIVE cannot be a head-space positive** (F95's own limitation L1,
untouched here); that any of this transfers to TEST (all arenas query train-split
items held out from their own head)."* F112 independently carries the
**first half** of that caveat — *"a raw-space null does not entail a head-space
null"* — but **not** the TEST-transfer clause, which is F113's alone (round-11
review). The recon uses "a raw-arena kill is one-sidedly secure" as if
settled. It is the campaign's working convention, not a measured result — which
is precisely why C06 is gated behind a fold-head falsifier rather than struck.

### 3.5 LBOP exists in the repository (recon §4.3)

Recon §4.3 asserts `[M]` that "LBOP" returns zero hits and infers *"the registry's
own dedup boundary points at a body of work that does not exist in the repository
under that name."* The grep is literally true **as scoped** (`refine-logs/*.md`
and `directions_tried.json`), but the inference is refuted repo-wide:

- `research-wiki/TARGET_GATE0_ITER6_LITERATURE.md:246` — a full candidate spec,
  `## 5. 候选 B:LBOP(Label-Blind Lattice-Barycentric Order Projection)`, with
  fast-fail gates LBOP-0/1/2 at `:284-288`, scored `novelty 5.3 / feasibility 5.8
  / +3+3 likelihood 4.0`.
- `TARGET_REVIEW_RAW.md:740` — the independent novelty reviewer's own dedup entry
  for LBOP, including its non-equivalence argument.
- `TARGET_GATE0_ITER6_LITERATURE.md:444` retains **three** distinct candidates
  (`保留恰好三条:LB-SCGP、LBOP、RHT`) and `:271` states the A-vs-B delta
  explicitly, so LBOP is **not** a mis-spelling of LB-SCGP.

**This sharpens rather than weakens C07's first unblock condition.** C07's registry boundary
demands *"a written mathematical delta"* against LBOP; LBOP exists, is a per-video
partial order over a harm/policy lattice — i.e. C07's object — was never run, and
its own LBOP-0 gate demanded `≥ +0.050` on both datasets. No such delta has been
written. The registry's own precondition is unmet against a real, locatable
comparator.

### 3.6 F82's graded structure is richest on EN, not ZH (recon §4.2)

`GRADEDLBL_PREGATE_RECORD.md:134` — *"Per-dataset binding oracle ceiling (max over
arms): EN **+0.0250**, ZH **+0.0256**. BOTH < +0.030."* The recon quotes only ZH
and calls it *"the dataset where the graded structure is richest"*; the record's
own census at `:87-92` gives Offensive as a share of positives **EN 73.2 % vs ZH
62.8 %**. Conclusion unaffected — both ceilings are under the final bar and both
are roughly half the Stage-0 bar — but the hold is recorded with **both**
numbers.

### 3.7 C02's fold-head comparison was paired against the wrong arm (recon §1.4)

Recon: *"C02's A0 used exactly this path (`acc 0.8875 / 0.8912`, matching F113's
fold-head `0.8867 / 0.8923`)."* `0.8875 / 0.8912` is C02's **FULL treatment arm**
3-seed mean; F113's figures are the **deployed floor**. The correct statement is
stronger: C02's own `gates.ARENA2.pooled_native_acc` — HateMM
`0.8884 / 0.8858 / 0.8858`, ZH `0.8929 / 0.8895 / 0.8946` — is **identical to F113's
per-seed fold-head floor, at the 4-dp precision F113's banked artifacts record, on
6/6 seeds** (`headspace_arena_hatemm_s0_OUT.json` stores `head_deployed_acc:
0.8884` against C02's `0.8884408602150538`, so 4 dp is the verifiable precision,
not bit equality). The instrument's reproducibility claim survives; both the
pairing and the precision language are corrected.

### 3.8 Observation filed, not acted on: C02's fifth validity gate

C02's A0 carries five named validity gates — in
`C02_A0_OUT.json` under `datasets.<ds>.gates`, **not** in `C02_A0_DECISION.json`,
which has no gates key — of which **four** have a machine `"pass": true`. `ZERO_CONTRACT` has no `pass` field: two of its four
criteria are `"kind": "DOCUMENTARY_CITATION_NOT_COMPUTED"`, and on MHC-ZH its
population is empty (`banked_text_zero_rows: []`), so it is vacuous there. The
phrase "all five validity gates passed on both datasets" originates in the
existing C02 finding, not in the recon. **C02 is out of this reopen's scope and
nothing about it was edited.** Filed here so the phrase is not re-propagated
uncaveated.

### 3.9 Two corrections that favour C09, the candidate being promoted

- **Kill-risk (iii) is misattributed.** F78 measured **nothing** — it is a `$0`
  PARK whose own status records the *"'$0 banked keys' premise false — head space
  never persisted"*, and no random control appears anywhere in it. The random-
  control null is **F88 null (3)**: HateMM-only, single-draw, deleting *train*
  rows whose own LOO vote disagrees with their label, curated `+0.0016` vs random
  deletion `+0.0031 / +0.0000`, self-labelled *"Pregate-grade null (one rule, one
  proxy head/cell, single draw)"*. Different population, different operator,
  different dataset coverage from C09's. The risk is real but weaker than the
  recon states. (The mislabel is inherited from F106's ledger shorthand, not
  invented by the recon; it should stop propagating.)
- **The Feldman objection's numerical leg is already retracted in-repo.**
  `HEADCOV_PREGATE_RECORD.md:305-310` withdraws *"the Feldman flourish"* because
  the deployed heads sit at 0.82–0.94, not 0.998 — while stating that Feldman's
  *substantive* step (memorising a long-tail singleton does not transfer to an
  unseen member of the same one-member sub-population) *"never depended on the
  head having reached 0.998, and that step stands."* The recon states the
  objection in its surviving substantive form. Correct use; the design below
  addresses that form.

---

## 4. Dispositions

A Gate-0 strike is **registry-level and reversible by a future user ruling**. It
is recorded as `struck_gate0_2026_07_31`, never as a measured kill. No candidate
below was run, and none of these dispositions is evidence for or against any
hypothesis.

### 4.1 Strike CONFIRMED — the one that survives fourteen rounds of adversarial review

**C14 · Multi-prompt Representation Ensemble → `struck_gate0_2026_07_31` from the
performance backlog; diagnostic-only status preserved**

Registry-ineligible by its own construction, quoted verbatim:
`eligible_for_primary_target: false`, and *"May be used only as a frozen
mechanism/upper-bound diagnostic. Ensemble predictions are forbidden as the final
method, and a positive diagnostic cannot satisfy novelty."* Reinforced by
`hard_constraints`: *"no ensemble stacking, cross-seed ensemble, or multi-prompt
ensemble as a final performance method."*

**Correction:** the recon's supporting citation — *"TVB ranked multi-prompt
ensembling 7 of 7 at ~0"* — is a **predicted** expected-value rank in a seven-row
priority table (`LITSWEEP5_COMPLETENESS.md` §4: *"~0 **predicted** (D1; F70+F80
dead neighbors)"*), never a measurement, and TVB's own ban_scope calls the cell
*"flagged-not-banned"*. It is **not** cited in support of this strike. The strike
needs only the registry's own text.

*Unblock:* a user ruling changing `eligible_for_primary_target`. Nothing else.

### 4.2 Strikes DOWNGRADED to HOLD — the six that needed a ban, screen, premise or clause read past its text

**C07 · Harm-Lattice Cone Metric → `held_lattice_delta_unwritten_reachability_unscreened`**

The recon recommended a strike and round 1 of independent review found the strike
unsupportable as posed. It is downgraded.

**The gap.** C07's boundary requires two things before implementation: *"a written
mathematical delta"* against prior lattice work, **and** *"a reachability screen"*.

1. **The delta is un-attempted, not unwritable.** LBOP exists and is locatable
   (§3.5), and no delta has been written — but nobody has tried. That is a
   different kind of fact from C05's, where the gate was **attempted and
   demonstrated unwritable**, and it does not support a strike. The two must not
   carry the same disposition on different evidence.
2. **No reachability screen of C07's object has ever been run.** What exists is
   F82's graded 3-class oracle, and F82's own ban_scope splits the two sides
   explicitly: *"vote-side Offensive reweighting closed both datasets **(any monotone
   weighting, any tau)**; **head-side graded auxiliary = F44-capped + admissibility-gated, only revivable by user
   ruling WITH a new mechanism argument**."* C07 is a **cone metric** — a
   head-side/representation object — so F82's measurement sits on the other side
   of its own source's written boundary. It is a **headwind to price**, not a
   screen of C07, and the earlier draft's phrase "has been run and fails" was
   wrong and is withdrawn.

**What the F82 evidence does support**, stated at its true scope: on the
*vote-side* channel the graded 3-class oracle is **EN `+0.0250` / ZH `+0.0256`**
(`GRADEDLBL_PREGATE_RECORD.md:134`), both under the `+0.030` final bar and about
half the `+0.050` Stage-0 bar, with Offensive down-weighting *monotonically
harmful* (`:118-119`), and `ERRPAT_MHC-ZH §5.3` measuring Offensive `n=28` err
`0.2500` vs Hateful `n=17` err `0.2941` — *"there is no Offensive-specific error
mass to reallocate"* — **a TIER-2 read on the MHC-ZH test split, carrying the same
qualifier this record applies to §5.2 and §5.4**.

**Three limits on that headwind, restored after round-2 review, because an earlier
draft elided them and they cut against the record's own argument:** (i) F82's
ban_scope ends *"; **HateMM out of scope (no Offensive class)**"*, so the headwind
covers at most one of the two datasets a PASS must clear; (ii) the ceiling is a
**dev-label gold cheat computed on the dev splits** (`n = 80` EN, `n = 78` ZH),
and ZH's `+0.0256` is **2 dev items**; (iii) `GRADEDLBL_PREGATE_RECORD.md:72-75`
pre-declares that the oracle *"does NOT bound the head's representation-reshaping"*
— which is the same reason C07's head-side object is not screened by it. A C07
proponent must still beat it, but it is a thin and dataset-partial headwind, not a
closure.

The supervision-source argument (gold forbidden by `banned_constraints[1]` and
non-existent; MLLM forbidden by `[5]`; a binary label inducing only a two-element
chain) is a supporting **inference** whose enumerative form is the same one
downgraded at C10, and it is not load-bearing here either.

*Unblock:* (a) write the mathematical delta against LBOP
(`research-wiki/TARGET_GATE0_ITER6_LITERATURE.md:246-288`) — noting at the point of
use that LBOP-0's own gate is **not** merely `≥ +0.050` on both datasets: `:284`
also requires **macro-F1, per-fold sign agreement and a joint Farkas/gradient-cone
audit**, which makes the unwritten delta *harder*, not easier; (b) name a partial-order
source that is neither gold, nor MLLM output, nor the binary label — noting that LBOP
sourced its order from a label-blind MLLM — though whether that lands on `[5]`/`[6]`
is **construction-dependent under the same EUM-vs-F60 tension recorded at C12
§4.2**, since `[5]`'s literal text is *"MLLM-scores-as-training-signal"*, `[6]` is
*"P1-P5 re-proposals"*, and `TARGET_GATE0_ITER6_LITERATURE.md:260` says LBOP's MLLM
emits policy sets and 不输出 label、score、memory pair/key 或 rationale; (c) **if** the construction
realises the order as a head-side *graded auxiliary*, satisfy F82's head-side
clause, which requires **a user ruling with a new mechanism argument** — stated
conditionally after round-2 review, since no text establishes that a cone metric
over a harm-act partial order *is* a graded auxiliary; and (d) clear a fresh
reachability screen at `+0.050` on two datasets.

**C08 · Provenance-Antisymmetric / Title-Source Encoder → `held_title_channel_separable_route_unscreened`**

The recon recommended a strike; rounds 1 and 3 of independent review dismantled two
of its three premises, and the strike is withdrawn.

**The gap — premise 1 is refuted by primary evidence in this repository.** Both the
recon and revision 2 of this record concluded there is no title channel, then no
*separable* title channel *"without re-deriving source metadata — a data-collection
act declined LOW/~0 in litsweep2."* **That is false for both MultiHateClip
datasets**, verified directly:

- `/data/jehc223/Multihateclip/English/annotation(new).json` carries a non-empty
  `Title` on **891/891** rows; `/data/jehc223/Multihateclip/Chinese/annotation(new).json`
  on **897/897**. (HateMM has none, consistent with the rest of the record.)
- `scripts/prep_mhc.py:72-85` — the very function cited for the folding — reads
  `title = (entry.get("Title") or "").strip()` and `transcript` as **separate
  variables** before concatenating them. `scripts/prep_video_dataset.py:126-139` is
  the same logic for MHC-ZH. Emitting a title-separated gt is a **re-run of an
  existing deterministic CPU script**, not a data-collection act.
- F88 ledger correction (c) reports *"medians: title 15 chars, transcript 76,
  composed 96"* — a measurement only possible from an already-separated title.

`LITSWEEP2_INPUT_FIDELITY.md:56`'s `title_present = 0` is true **of the gt-jsonl key
schema only**; the inference at `LITSWEEP2 §3.3` that recovering it *"means
re-scraping YouTube metadata"* is factually wrong for both MHC datasets. Nothing in
`hard_constraints` or `banned_constraints` bans a title channel.

So a **`≥2`-dataset title route exists on MHC-EN + MHC-ZH** — the same shape as the
C12 downgrade, where a leg that appeared to make the two-dataset arithmetic
impossible turned out not to. The substrate objection that drove the
recommended strike — that no provenance artifact exists on `≥2` datasets — is
arguably met by the title itself. *(An earlier draft put that objection in
quotation marks as C08's "own written unblock"; it is not — C08's registry
`dedup_boundary` contains no such clause, and the phrase came from this record's
own round-3 review text. The quotation is withdrawn.)*

**What survives, and is recorded as the burden a proponent inherits.**

- **Premise 2 stands at marker scope.** The `<em class="keyword">` highlight is
  MHC-ZH-only: `243/579` train and `34/78` val, **`0` HTML tags on HateMM and
  MHC-EN** (re-measured). MHC-EN carries HTML **entities** on `64/549` train and
  `9/80` val rows, but they are ordinary escaping of apostrophes, quotes and
  ampersands encoding **no source identity**. So the *provenance-marker* half has
  no `≥2`-dataset substrate; only the *title* half does.
- **Premise 3 is corroboration only.** `ERRPAT_MHC-ZH_2026-07-26.md:332-335` finds
  quoted-hate false positives at `0.172` vs `0.0776` raw but `5` observed vs `4.57`
  expected at core-error level, `p = 0.5022`. It is a **TIER-2 CPU-re-mint proxy
  read on the MHC-ZH test split** whose own section closes *"Both are recorded as
  hypotheses the ZH test split is too small to settle, not as clusters"* — a
  **non-significant underpowered result, not a measured null**.
- **The representation-level residual is real but belongs elsewhere.** C01's kill
  of endorsement-vs-quotation separation is explicitly route-scoped: *"It does not
  falsify policy contrast under same-pooling caches."*

*Unblock:* C08 must be re-posed around the half that has a substrate. Concretely:
(a) state whether the route is title-source antisymmetry (which has an MHC-EN +
MHC-ZH substrate and needs only a CPU re-prep to expose) or provenance-marker
antisymmetry (MHC-ZH only, no `≥2`-dataset route); (b) for the title route, price
the Stage-0 oracle **per dataset**. The earlier draft priced it from a single
figure — *"title 15 chars … composed 96"* — which is an **MHC-ZH, test-split**
median inherited second-hand via F88 ledger correction (c) from
`ERRPAT_MHC-ZH_2026-07-26.md:272`, and generalising it was wrong. *(Round-11
review: the earlier draft also called that figure "markup-stripped". That is an
**inference**, not a reading — neither `ERRPAT:272` nor F88 states the convention,
and F88 actually describes the title as **carrying** the markup. It is the likely
reading, since `15` sits near this record's own measured stripped value `13`
rather than the raw `27`, but it is recorded as inferred.)* Measured
directly over the train+val ids of each dataset, the title median is **51
characters on MHC-EN** (transcript median 322) against **27 raw / 13
markup-stripped on MHC-ZH** (transcript median 78). The EN leg — the half that
makes the `≥2`-dataset route possible at all — is **3.4× larger than the figure the
"thin channel" verdict rested on, and is unpriced**; and (c) keep it distinct
from C01's same-pooling territory, which is a separate un-falsified candidate
rather than part of C08; and (d) **route around F108** (round-14 review): its
ban_scope closes *"any change to WHICH STREAMS OR IN WHAT PROPORTION enter the
RETRIEVAL KEY … CLOSED BY CONSTRUCTION, SO A RENAME CANNOT EVADE IT"*, and it is
listed under `standing_eliminated_families`. C08 lands in F108's carve-out (ii) —
*content*, not weight — **as written**, and the statement above that no
`hard_constraint` or `banned_constraint` bans a title channel remains literally
true, F108 being a finding-level ban_scope. But a proponent realising (a) as
*"expose the title as its own key block"* walks straight into it. Naming F108
**raises** this unblock's burden, which is the direction an honest record should
err.

**C13 · ZH HTML Markup Invariance → `held_zh_scoped_no_cross_dataset_pairing_named`**

The recon recommended a strike. Rounds 4, 5 and 6 successively dismantled each
basis offered for it, and round 6 established that what remains is a **hold**.

**How the basis moved, recorded because the movement is the finding.** The recon
struck C13 on a plausibility argument (invariance would delete a predictive
feature). Round 4 established that step is an **inference**, not a measurement of
C13, and the strike was re-based on substrate arithmetic. Round 5 established that
the arithmetic was measured for HTML **tags** only while C13's claim says *"HTML/title
markup"*, and the strike was re-based on registry text. **Round 6 established that
the registry-text basis is a precondition a proponent can satisfy** — and this
record's own C07 Critical holds that an un-discharged, proponent-satisfiable
precondition cannot carry a strike. C13 therefore joins the hold column, on the
same principle that moved C07 and C08.

**What is actually established, at its true scope:**

1. **C13's claim is self-scoped**, verbatim: *"Removing sensitivity to native
   HTML/title markup may address **a ZH-specific extraction nuisance** without
   adding a channel."* Its dedup boundary then conditions — it does **not**
   prohibit — a two-dataset claim: *"no claim of a two-dataset final route
   **unless paired with a genuinely cross-dataset mechanism**."* **No such
   mechanism is named in the entry**, so as written C13 cannot meet
   `unified_pilot_gate.stage_0_reachability`'s *"at least two datasets"*. That is
   an unmet condition, not a closure — and unlike C14, C13 carries no
   `eligible_for_primary_target: false` flag and sits in `ordered_backlog` like
   C05–C12.
2. **The `<em>` census corroborates the self-scoping**: the harvest highlight is
   `243/579` train and `34/78` val on MHC-ZH and `0` **tags** on HateMM and
   MHC-EN (re-measured). It does not *establish* a no-substrate result, because
   MHC-EN carries HTML **entities** on `64/549` train and `9/80` val rows
   (`&#39;` ×43, `&quot;` ×16, `&amp;` ×17 train-only) — ordinary escaping rather
   than a harvest artifact, but within the literal reach of *"HTML markup"*.
3. **The strongest headwind is measured, and its step to a verdict is an
   inference.** The markup is **a strong lexical shortcut** in the ZH text channel:
   markup-bearing rows hate at `0.5802` (141/243) against `0.1161` (39/336)
   without and a `0.3109` base rate (val `0.5882` vs `0.1818`, base `0.3590`), and
   round 1 sharpened it — rows carrying a harvested keyword *without* the markup
   hate at only `10/140 = 0.0714`, so the tag itself carries signal beyond the
   keyword. **That is `5×` the no-markup rate and `8×` the bare-keyword rate.** No
   ranking against other lexical features was computed, so this is *a* strong
   shortcut, not a demonstrated *strongest* one. The step from it to *"invariance
   would cost accuracy"* remains a plausibility inference.

*Unblock:* (a) name a genuinely cross-dataset mechanism to pair with, as C13's own
boundary requires; **and** (b) show the invariance does not delete predictive
signal, against a headwind of `5×` and `8×`. A proponent who cannot do (a) has a
single-dataset route that the two-dataset Stage-0 bar excludes on the entry's own
terms.

**The finding survives the disposition and is the more valuable output.** Whatever
happens to C13 as a route, the measurement is paper material and is recorded in §7.

**C10 · Gold-free Reasoning-Boundary Structured Memory → `held_eum_preconditions_unmet`**

The primary leg is sound: C10's own dedup boundary says *"no span/segment gold and
no inherited parent label as segment gold"*, a phrase that only makes sense of
sub-video units, and EUM's ban_scope closes *"Sub-video retrieval UNITS of any
kind — 'evidence units', spans, segments, clips, shots — as the object stored in
and retrieved from the memory bank … **Do NOT re-propose under a new name**."*

**The gap.** The recon's second leg — that the legal unit-definition space is
EMPTY — is an **extension**. EUM's precondition (2) enumerates three illegal
sources (gold spans; MLLM-derived boundaries or weights; per-item selection) and
concludes emptiness *"**as of this recon**"*. A rule-based, gold-free, MLLM-free
boundary (sentence/punctuation split, pixel-difference shot boundary) is not in
that enumeration and is not per-item selection. "Gold-free" is precisely a claim
to a fourth source, and the recon asserts emptiness by quoting EUM rather than by
testing whether C10 names one. The third leg — BSY's *"OPEN USER RULING, BLOCKING
ANY PREREG"* — is scoped in its own text to *"bank-**ADDITION**"* candidates, and whether
C10 adds rows is a property of a construction nobody has written down. *(An
earlier draft hedged that C10 "arguably replaces the bank object rather than adds
rows"; that hedge is **withdrawn** — EUM's own measurement, quoted in the unblock
below, is that a flat unit bank puts only 10.6–11.3 distinct parent videos in a
top-20, i.e. it has strictly more rows than the video bank.)* The BSY block is
also procedural, pending a user ruling, rather than a scientific kill.

HOLD rather than strike is the correct disposition because the burden already
sits on C10: the registry itself requires *"a strict non-isomorphism gate against
EUM, bank synthesis, segment set matching and archive memory."*

*Unblock — EUM's three written preconditions, all required before any recon:*
(1) exhibit a dataset where hateful evidence is genuinely a minority of the
runtime — on HateMM, EUM measures **two separate quantities** over `n = 298` train
hate videos: gold hate-span coverage **median `0.8289`** (mean `0.7174`), and
**`74.2 %` a single contiguous span**. *(EUM's ban_scope compresses these into
"median 83 % … in a single contiguous block"; the two statistics are stated
separately here because the compound is a compression, not a measurement.)*
(2) define the units without gold spans, without MLLM-derived boundaries or
weights, and without per-item selection — i.e. **name the fourth source**;
(3) explain how a flat unit bank adds video-level evidence when it measurably
*reduces* effective video-level depth (10.6–11.3 at K=4 vs the deployed 20, 5.0
at K=30). **EUM's own grade qualifier attaches to (1) and (3)**: its record field
labels these *"recon-grade inline reads of banked TRAIN-split caches, **not
gate-grade** frozen-script output"* — the same qualifier this record applies to its
ERRPAT and F88 reads. Plus resolution of the `banned_constraints[3]` user ruling if the
construction turns out to add bank rows.

**C11 · Null-aware Evidence Representation → `held_thin_evidence_disjunct_unscreened`**

C11's claim is **disjunctive**, verbatim: *"may prevent speech-poor **or
thin-evidence** videos from being forced into the ordinary transcript geometry."*

The recon's `[M]` kills only the first disjunct, and does so correctly:
whitespace-only `text` is `39/744` and `9/107` on HateMM and **`0` on all four
MHC splits** — re-measured exactly. Against a `≥2`-dataset requirement the
literal-null reading has no substrate.

**The gap.** The second disjunct is not only unscreened, it is **measured
positive** on the dataset the recon says has zero instances.
`ERRPAT_MHC-ZH_2026-07-26.md:301-306`, §5.2 *"The one real covariate cluster: thin
transcript"* — the `[31, 76)`-char band holds *"11 of the 22 core errors in 37
items (err rate/seed 0.2973 vs 0.0631 / 0.1532 / 0.1053 in Q1 / Q3 / Q4) — 2.0×
enrichment, permutation p = **0.0048** (50 000 perms; robust to integer vs
exact-quantile cut, p = 0.0051)."*

**Scope of that leg, stated at the same standard applied to C08's premise 3
(§4.1), after round-2 review:** §5.2 is a **TIER-2 CPU-re-mint proxy read on the
MHC-ZH test split, `n = 149`, final-epoch protocol** — the same tier, document and
split as the §5.4 result this record demotes to "non-significant and underpowered"
— and it is a **pooled** effect only: class-stratified, each half is underpowered
(`ERRPAT_MHC-ZH:307-308`, negatives `p = 0.0506`, positives `p = 0.0668`). It is
also, at one remove, a test-split forensic informing a registry disposition; that
sits uncomfortably beside `unified_pilot_gate.test_rule` (*"No test-driven
choice"*), and is recorded here rather than relied on silently. **What it
establishes is therefore narrow: the thin-evidence disjunct is not empty, so it
cannot be dismissed by a whitespace census — not that it is a live lever.**

Short-text populations do exist on all three datasets, but the earlier draft's
cross-dataset census is withdrawn as substrate evidence: "under 76 chars" is
MHC-ZH's own quartile cut, and applying it to corpora with median lengths of 696
(HateMM) and 369 (MHC-EN) characters does not identify the same phenomenon. Any
screen must define thinness **relative to each dataset**.

The recon's fallback — that a continuous reading is *"the quantity channel C02
measured at +0.0009 / −0.0012"* — is also mis-aimed: C02 measured **duplicating
transcript content** (a density orbit on the key), whereas C11 proposes an
**internal missingness state** and its own boundary explicitly distinguishes
itself from "ASR replacement" and "identity-fill at the head". Different operator,
different object.

*Unblock — and the burden is heavy.* Screen the thin-evidence disjunct on its own
terms, against the strongest text on the other side, quoted here **in full**
because the earlier draft truncated it: `ERRPAT_MHC-ZH:405` prices that same
cluster as *"effectively LOCKED … +0.0738 if all 11 flipped, but §7.1 shows no
better transcript exists; the deficit is signal absence. **No legal unmeasured
lever found.**"*

That final sentence is the source's own verdict on the exact cluster this hold
calls unscreened, and a proponent must defeat it. The hold — rather than a strike —
stands on a narrow but real distinction: ERRPAT's search was for levers that
supply *more or better transcript signal*, and it concluded correctly that none
exists, because the deficit is signal absence. C11 proposes something different in
kind: **representing the absence itself** rather than filling it. Whether that is a
"legal unmeasured lever" ERRPAT did not enumerate, or merely the same dead end
renamed, is a question no measurement in the repository answers — which is what a
HOLD records. **If a C11 proponent cannot articulate why representing absence is
not covered by "no legal unmeasured lever found", C11 should be struck without
further measurement.**

**C12 · Archive-version Stability Curriculum → `held_ban_scope_ambiguous_construction_unnamed`**

This is the deepest stretch in the recon, and three of its four legs need text
read past its scope.

1. **Two inference-time nulls used against a training-time boundary that already
   excludes them.** C12's boundary reads *"Training-time invariance curriculum,
   **not adding archive keys/memory at inference**; the archive-key family remains
   eliminated."* The recon cites archive-**as-retrieval-key** (`ΔAcc −0.0014 ±
   0.0313`, zero vote flips) and the AUTO two-vote **inference-time bank
   deletion** (`C − A = 0`). Neither measurement ever computed a cross-version
   disagreement, which is C12's entire quantity — so neither reaches the object
   C12's own boundary defines.
   There is an on-point precedent for refusing exactly this substitution —
   `LITSWEEP3_DATA_CENTRIC.md:80`: *"The AND-rule C−A=0 finding … is a **headwind
   to price, not a coverage of this mechanism**."* **Cited as a general form, not
   as a rule about C12** (round-8 review): that bullet sits inside LITSWEEP3 §4,
   whose subject is *memory-bank curation*, and its stated ground for excluding the
   `C−A=0` null is that the null *"used an **MLLM two-vote** signal; here the
   deletion signal is gold train labels + kNN geometry only. Different information
   source."* C12's stability statistic is archive-derived, i.e. on the **MLLM side**
   of that very distinction, so the source's own ground does not transfer. **The
   downgrade does not need it**: the inference-time/training-time leg above stands
   alone.
2. **`banned_constraints[5]` is construction-dependent, not settled.** Its literal
   text is the bare phrase *"MLLM-scores-as-training-signal"*, carrying no
   construction-level gloss. Broad-reading precedent
   exists, but it is a **stack, not a gloss** (round-10 review): EUM's precondition
   (2) reaches *"MLLM-derived boundaries or **weights**"* via *"**P3 / P11 plus**
   `banned_constraints[5]` … **and `[6]`**"* — four authorities, not `[5]` alone.
   Under that stack, a stability statistic that **weights or selects** training
   examples is closed. *(The mis-attribution ran in the **conservative** direction —
   the stack makes the closure stronger, not weaker — so C12's disposition is
   unaffected.)* But narrow-reading precedent exists too and the recon does not engage
   it — F60/AUG rules **MLLM-as-data-generator admissible**: *"clears C3/P4
   (features), P11 (scores), TARC (loss), single-dataset veto, data boundary — the
   un-enumerated generator role is real"*, and AUG was killed on domination, not
   on `[5]`. If the two archive versions are used as **two views of a multi-view
   training target**, F60 is the governing precedent, not `[5]`.
   **Carried after review, because omitting it would repeat the exact fault this
   record charges the recon with at F80:** F60's ban_scope ends *"Do not
   re-propose without **D7 generator-role sub-ruling**"*, and F60's detail closes
   *"Revisit only under a user D7 generator-role sub-ruling AND acceptance of a
   weaker-than-tied prior."* **The open decision is the D7 generator-role sub-ruling specifically**, not D7 itself — `research-wiki/DECISION_MEMO_pending.md:134,211` records D7 as `RESOLVED 2026-07-14 (RESOLVED-NEGATIVE)` and binding, and F60 asks for a *sub-ruling* on the generator role. The narrow reading is
   therefore not a free pass — it is a route itself gated on a user decision
   nobody has made.
3. **F55 is misread — and this is decisive.** The recon says *"MHC-EN is
   additionally closed at all three levels (F55)."* F55's ban_scope verbatim:
   *"**Cross-encoder composition with ADAPTED text on EN: dead** … F50 ban now
   covers frozen AND adapted feature compositions on EN. EN closed at all three
   levels."* And F55's own finding detail names the levels: *"MHC-EN now closed at
   frozen (F50), collapsed-adapted-deployed (B4/F53), and healthy-img+adapted-text
   composition (F55) levels simultaneously"* — three levels **of the
   encoder-composition question**. F55 does not close MHC-EN for all method
   families, and C12 is not an encoder-composition candidate.

Leg 4 — no HateMM archive — is the one that survives cleanly (`data/Archive/`
holds exactly `MHC` and `MHC_zh`, re-verified). **But it is not fatal**, because
those two datasets are exactly the two that have archives, and with F55 read
correctly, an MHC-EN + MHC-ZH route is arithmetically available.

*Unblock:* C12 must **name its construction**. Note first, after round-14 review,
that **a direct single-authority gloss of `[5]` does exist and raises the burden on
*both* branches**: F103/OCR's ban_scope glosses `[5]` on an **archive field**
specifically — *"It is Qwen-2.5-VL GENERATED TEXT and falls under
`banned_constraints[5]` (MLLM-scores-as-training-signal / the P4-P11 family
boundary)"*. F60 conflicts with it head-on, so the fork is real; but the branch that
looked safer is not free. Then: stability-as-weight (lands on `[5]` under the **EUM
four-authority stack** — `P3 / P11` plus `[5]` and `[6]` — **and independently under
F103's direct archive-field gloss**, and is then dead) versus stability-as-multi-view-target
(governed by F60's admissible generator role, **which is itself gated on the open
D7 generator-role sub-ruling and on accepting a weaker-than-tied prior**). That
single choice decides the ban, and it is C12's to make. Real headwinds to price if it proceeds: the EN
arm's own difficulty record (F53 EN-LoRA failing both protocols; F88's EN
diagnosis of *"LABEL-SEMANTICS MISMATCH, NOT A REPRESENTATION DEFICIT"*), and the
fact that F88 records the deployed EN arm as already carrying `archive-kNN
a0.25` at inference, which C12's own boundary would have to address.

### 4.3 C05 — HOLD, as recommended

**C05 · Full-Bank Signed Discourse Manifold →
`held_nonisomorphism_gate_unwritten_as_posed`**

The registry makes a written non-isomorphism gate a **precondition**: *"Requires a
fresh non-isomorphism gate against SSR/EDCM, RDK and LB-SCGP Global-R2."* The
recon attempted it on paper and could not write it. All four comparator quotes
verified FAITHFUL:

- **vs Global-R2** — verbatim from `TARGET_FINDINGS.md:32`: *"a closed train-only
  label-blind structural-certificate cache compiled into one replayable full-bank
  PSD/unit-diagonal Gram target fitted uniformly by the shared encoder, with test
  as ordinary full-video train-memory top20 kNN."* C05's stages (i)–(iii) are
  identical; the only free variable is the **source** of the relation matrix, and
  Global-R2 is the label-blind-certificate source, dead as the 15th negative
  (*"killed pre-GPU by G0-cond: cache 91-93 % constant, oracle@coverage 10× under
  bar"*, coverage `8.7 % / 6.9 %`, real-A conditional information `≤ 0`).
- **vs RDK (F99)** — RDK *"applies ONE SHARED MAP TO BOTH SIDES, i.e. it falls on
  NCA's side"*, its verifier fitted on *"the SAME label-agreement matrix NCA
  optimises"*, making it *"NCA with a two-stage estimator"* — all verbatim.
- **vs EDCM** — its A0 is the closest existing measurement of C05's oracle and
  reached `+0.0273 / +0.0394` on MHC-EN and `+0.0380 / +0.0444` on MHC-ZH, both
  below `+0.050`; anti-repeat *"The next route must change the video-level
  correctable unit"* — C05 does not.
- **vs SSR** — terminal preflight oracles `+0.0036 / +0.0128 / +0.0052 / +0.0259`,
  all below `+0.050`.

The gate cannot be written **from the three sources anyone has enumerated** —
MLLM-emitted (`[5]`/`[6]`), gold-label agreement (NCA's own objective, F75), and
label-blind structural certificates (which **is** Global-R2), each individually
banned or dead.

**The status string says `unwritten_as_posed`, not `unwritable`, after round-2
review.** That enumeration is not established as exhaustive, and this record
explicitly calls the same enumerative form *"an extension"* at C10 and declines to
make it load-bearing at C07 — so it cannot be treated as a proof of impossibility
here either. C05's own unblock invites precisely the fourth source the enumeration
would exclude. What is established is that **the registry's precondition has been
attempted and could not be discharged from any source now on the table**, which is
a stronger fact than C07's un-attempted delta and a weaker one than impossibility.

**Unblock condition, stated so it is usable.** C05 becomes writable iff a
proponent names a relation source that is **(a) not MLLM output, (b) not the gold
label-agreement matrix, (c) not the Global-R2 certificate family**, and (d)
carries measured conditional information over the deployed key — the G0-cond gate
that killed the A-line. On (a), after round-3 review: whether a **score-free**
MLLM emission is covered at all is **construction-dependent** under the same
EUM-vs-F60 tension recorded at C12 §4.2 — `[5]`'s literal text is
*"MLLM-scores-as-training-signal"* and `[6]` is *"P1-P5 re-proposals"*, neither of
which plainly reaches an emission carrying no label, score, key or rationale. If such a source exists, the first legal step is the `$0`
full-bank membership oracle in the fold-head arena, **not** an encoder fit.

**Scope note.** Because §3.1 withdraws the coverage bound's application to
re-ordering, C05's reachability prior is weaker evidence than the recon presented
— but C05 is held on the **unwritable gate**, which is the registry's own
precondition and is independent of the reachability arithmetic.

### 4.4 C06 — GATED behind a `$0` falsifier, as recommended

**C06 · Prompt-Orbit Tangent/Curvature → `gated_on_zero_cost_falsifier`**

The load-bearing evidence is C01's A0, verified directly by this adjudicator
against `C01_A0_OUT.json` — every accuracy recomputed from the stored confusion
matrices and every net-fix figure matching:

| arm | HateMM acc / net (`n_dev` 107) | MHC-ZH acc / net (`n_dev` 78) |
|---|---|---|
| `endpoint_std` (reference) | `0.8411` / `0` | `0.8590` / `0` |
| `displacement` (real) | `0.8505` / `+1` | `0.8846` / `+2` |
| `common_displacement` (**primary**) | `0.8598` / `+2` | `0.8590` / `0` |
| `common_interaction` (secondary) | `0.8224` / `−2` | `0.8333` / `−2` |
| best **random** rotation `orthrot_83p8` | `0.8692` / `+3` | `0.8974` / `+3` |
| `orthrot_72p7` | `0.8505` / `+1` | `0.8974` / `+3` |

Two further arms the earlier table omitted, added after round-14 review because
their omission **understated** the record's own adverse case: HateMM `common`
`0.8692` / `+3` and MHC-ZH `endpoint_concat` `0.8846` / `+2` — the decision block's
own named strongest ordinary controls, against which `gain_over_strongest_control`
is `−0.0094` and `−0.0256` with `pass: false` and `decision.continue = false`.

**The best of six** matched-block-L2 orthogonal rotations matches or beats the real
prompt displacement on both datasets. **Stated precisely after round-14 review:**
`c01_policy_contrast_a0.py:1272`'s `orthogonal_blocks()` is a **Givens mixing of the
two endpoint blocks**, so the six arms are angles on **one parameter family** that
also contains the primary — the code's own guards confirm `θ = 45°` *is*
`common_displacement` (max abs diff `8.9e-08`–`1.2e-07`) and `θ = 0` *is*
`endpoint_concat`. Calling them "random directions" reads more diffuse than the
object is; the correct reading is **sharper and more adverse**, since the real
displacement is one angle among many on a family where several other angles do
better. Stated at full scope after review:
the rotation spread is `0.8505–0.8692` on HateMM and `0.8462–0.8974` on MHC-ZH, so
against the primary `common_displacement` arm 4 of 6 HateMM rotations and 2 of 6
ZH rotations sit *below* it. The adverse reading is that a random direction with
matched norm **can** reach or exceed the real displacement — not that every random
direction does. That is still a genuine adverse measurement of C06's premise at
the two-point case, and it invokes no ban; it is also precisely why the
disposition is a `$0` falsifier rather than a strike.

**Three corrections, all of which argue for gating rather than striking:**

1. **The supporting bans do not reach C06 — on object mismatch.** F80's object
   is prompt **language**. Quoted at full scope after round-10 review, since the
   earlier draft charged the recon with truncating this entry and then truncated
   its other half: the ban_scope opens with an **unconditional on-dataset** closure
   — *"extraction-instruction language variations (**any language, any stream,
   either encoder arm**) on MHC_zh; prompt-language axis measured null-to-negative"*
   — and only then adds the conditional, elsewhere-scoped clause *"do NOT
   re-propose prompt-language matching **elsewhere** without new mechanism
   (HateMM/EN are English-content = no mismatch exists)"*. **The conditionality
   attaches to *elsewhere*, not to MHC_zh.** Either way the warrant is unchanged,
   because it rests on **object mismatch** — orbit geometry is not
   extraction-instruction language. C06 is not prompt-language matching; it is a claim about orbit
   geometry, i.e. a mechanism claim. F70's object is individual **readout cells**
   (hidden layer L24, one-word prompts, last-token span), not orbit geometry.
   *(Both bans also carve out multi-prompt ensembling explicitly — F80: "a
   SEPARATE user-gated item"; F70: "Does NOT price: … multi-prompt ensembling" —
   but that clause's object is C14, not C06, and it is not the warrant here.)*
   TVB's "7 of 7 at ~0" is a prediction resting on F70+F80 (§4.1, C14). The
   disposition therefore rests on C01 alone.
2. **C01's arena is raw dev keys, not the fold-head path** (`n_dev` 107 / 78), and
   F113 itself marks the kill direction NOT ESTABLISHED (§3.4). A `$0` re-run in
   the deployed head space is exactly the right instrument.
3. **The banked-cache inventory is misstated, and this changes the falsifier's
   design.** The recon claims
   `Qwen2.5-VL-7B-Instruct-LoRA{,-curric}_HF-ro_{L24,L28,ow_L24,ow_L28}` on both
   datasets. Directory listing shows **HateMM has only `-LoRA-curric`** ro-caches
   and **MHC_zh has only `-LoRA`** — one adapter lineage each, not four
   combinations. Additionally
   `src/utils/generate_VideoMLLM_embedding_readout_HF.py:73-89` shows the `ow_` cells change
   the **readout span as well as the prompt**
   (`("ro_ow_L24","oneword","last_token","last_token",LAYER_MID)`), so "2 prompt
   points × 2 layers" is prompt-and-span confounded — the same confound C01's
   review already narrowed its claim for.

**The falsifier, priced not spent.** Re-run C01's real-displacement-versus-
matched-norm-orthogonal-rotation battery in the **fold-head arena** on the
already-banked `ro_*` caches — zero GPU, zero extraction, minutes of CPU on
`scripts/analysis/headspace_{mint,arena}.py`, which exist and are banked. Its
pre-registration must (i) use the per-dataset adapter lineage that actually
exists rather than assuming a matched pair, and (ii) state the prompt/readout-span
confound as a declared limitation exactly as C01's did. If the rotations again
match the real displacement in the deployed head space, C06 closes for `$0` and
the `1.7–2.5 GPU-h` of extraction is never queued. If they do not, C06 has earned
its extraction.

### 4.5 C09 — PROMOTED, and the promotion is stronger than the recon claimed

**C09 · Stable-Inversion Topology Surgery → `next_active_candidate_post_C04`**

Verified supporting evidence, all FAITHFUL to F88:

- Error sets are **~90 % seed-invariant** — HateMM 24–25 of 26–28 errors wrong in
  3/3 seeds (89–93 %); **ZH 22 of the 25-item union wrong 3/3 with NOTHING at
  exactly 2/3, and all 12 false negatives 3/3-stable**.
- Calibration the recon omits, recorded here: EN's shape differs (22 consensus
  errors over 4 seeds plus a 20-item seed-flip noise band), so "~90 %" is a
  HateMM/ZH statement; and F88's HateMM per-item predictions are a
  **CPU-reconstructed proxy**, with F88's own binding caveat that *"a CPU-trained
  arm must be paired against a CPU-TRAINED FLOOR, never against the banked GPU
  floor."* That caveat is carried into the Stage-0 design.

**Legality of using train labels OOF to identify inversions — resolved as LEGAL,
on two written texts, not by inference.** The parent task asked that this be
flagged for the user if ambiguous. It is not ambiguous:

- `autoresearch/goal_mllm_plus3/state/progress.json:25`, the user's own
  oracle-ranked-queue ruling: *"**Legal attack on selection-locked pools = trained
  selector/reshaper on train labels only** (F66 binds only fixed-map phi0)."*
- `refine-logs/LITSWEEP3_DATA_CENTRIC.md:82`, an on-point in-repo adjudication of
  exactly this shape: *"≠ F47 per-item routing / F66 per-item selection
  (law-III): those select **per test instance**; curation selects **train items
  once, globally, applied identically to every test query** — a symmetric
  operator, **so law-III/F66's per-item ban does not apply to the mechanism**
  (though Wall-A still caps the achievable magnitude)."* — the parenthetical
  restored after round-2 review, because the same section prices that mechanism at
  *"+3 any dataset: ~1-2 %"* (`:95`) and *"at most +0.001-0.006"* (`:91`).

Every text on the other side (F47, F66, EUM precondition 2) bans **per-test-
instance** selection. Three boundaries flip C09 to illegal and are written into
the design as HALT conditions: any query-time consultation of the stability
statistic (F47 fires, and its escape clause is closed because an OOF-stability
statistic *is* "derivable from banked features/votes"); any per-item exception
surviving to inference as a per-item rule; and any use of dev/test labels at any
stage.

**The counter-text, carried after review — because this record corrects exactly
this one-sidedness elsewhere and must not commit it here.**
`LITSWEEP5_COMPLETENESS.md` §4(ii), headed *"The contradiction (load-bearing)"*, is
an on-point in-repo adjudication written **after** the ruling: it observes that the
ruling's two blessed classes — *"Trained SELECTOR on train labels"* and *"Trained
symmetric RESHAPER on train labels"* — are *"both already measured dead"*, and that
the ruling *"was written at lit-round-count 3 — before F75/F77/L1 sharpened the
walls."* This does **not** defeat the legality verdict: legality and viability are
different questions, and nothing in LITSWEEP5 withdraws the permission. But the
honest statement is that **C09 is legal under a ruling whose viability premise the
repository itself flags as stale**, and a C09 proponent inherits that as a prior,
not as a formality.

**And the counter-text is itself downgraded, not vacated — recorded after round-7
review, because this record polices exactly this stale-premise class everywhere
else.** §4(ii)'s *first* blessed-class death reads *"Trained SELECTOR on train
labels = F47's train-supervised source. DEAD: the deployed kNN vote memorizes train
(**CLIP LOO 0.998**)"* — and F114 rules that exact premise a **CLIP** number,
against deployed Qwen heads at `0.9406 / 0.8915 / 0.8154`, leaving train-side
headroom 30×–92× larger. `LITSWEEP5_COMPLETENESS.md` is **not** among the nine
records F114 corrected, so the retraction never reached it. §4(ii)'s **independent**
leg — the measured train-disagreement counts `0/109`, `0/102`, `0/92` — is untouched
and stands. Net: the counter-text still weakens C09's prior, but by less than its
wording implies.

**Stage-0 costs zero GPU-hours.** The fold-head arena is banked and verified
present: `scripts/analysis/headspace_{mint,arena,fidelity,report}.py` all exist,
as do all six `headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`. Per-fold head
**checkpoints are not persisted** (`headspace_mint.py:274-281` monkeypatches
`torch.save` to a no-op), so heads are re-minted at ~25–40 s CPU each; a full
2-dataset × 3-seed re-mint is ≈ 36 heads ≈ 30 CPU-minutes.

**The recon's four kill-risks, quoted, with verification notes:**

> (i) any encoder-level pull of an inversion toward its right analogue is a
> label-using metric move ⇒ F75/NCA and §1.3's `+0.0286`;

Stands. F75 is *"the first measured negative for trained-reshaping-unlocks-oracle-
headroom"*. Note §1.3's bound must always be quoted with `R² = 0.027`,
`r = +0.1642`, slope CI `[−0.0221, +0.1637]` straddling zero, MHC-ZH dev only —
F114's standing instruction — and F114 forbids citing F107 as a theory-level
door-closer.

> (ii) Feldman / Feldman & Zhang predict this exact seed-invariant confident error
> set **and** that no operator fixes it, because memorisation is necessary and
> unavailable for held-out items;

**The strongest objection, and it is stated in its surviving form.** Its
numerical leg is retracted in-repo (§3.9); its substantive leg stands. The
Stage-0 design below makes it an explicit measured discriminator rather than a
rhetorical caveat.

> (iii) F78 curation already failed its own random control on this population;

**Misattributed — see §3.9.** F78 measured nothing; the null is F88 (3), HateMM-
only, single-draw, on train-row deletion. It remains a headwind to price, on a
different population and operator.

> (iv) the "constraining break exposure" clause has to beat an exchange rate never
> observed above `1.17` in 36 cells.

**The `1.17` figure is stale (§3.3)** — the true campaign record includes 1.8889,
2.2353, 1.8333, 2.8889 and 6.0000. This makes risk (iv) *weaker* as posed, but
`banned_constraints[10]` is the binding rule and it is unchanged: **exchange rate
is not a screening criterion at all**; the screen is NET ITEMS against
`22.3 / 17.4 / 16.5` for `+0.030`, scaling to `37.2 / 29.0 / 27.5` for `+0.050`.

> **Counterweight:** `NCA_FORENSIC_RECON.md:110` explicitly rules that F66 does
> not bind trained-space reshaping, so this is not foreclosed the way the
> eval-time families are.

Verified verbatim: *"⇒ Ruling: F66 does NOT bind trained-space reshaping. The cell
is not F66-dead — it is legitimately un-measured."* One-sided use, corrected here:
`:112` prices the same cell at *"honest P(≥+3) stays 2–4 %"*, and the cell it
unblocked was subsequently run and killed as F75.

---

## 5. Effective post-C04 order

`C09` → (`C06` `$0` falsifier) → `C05`/`C07`/`C08`/`C10`/`C11`/`C12`/`C13` only if
their named unblock conditions are met. **`C14` struck** — the only one.

The historical `ordered_backlog` array is left untouched as the record of what
was ordered on 2026-07-28; this is the effective order after the reopen.

**C09 goes first and alone.** It is the only candidate aimed at the population the
error forensics actually found, the only one whose Stage-0 needs no extraction,
and the only one that can be adjudicated at zero GPU while C04's tranche holds
the serial-execution lock. Its CPU job waits for C04's tranche to terminate
(serial-execution precedent) and for main-dialogue authorization.

**One strike, seven holds, one gate, one promotion.** Round 1 moved C07 out of the
strike column, round 3 moved C08 and round 6 moved C13; the holds are C05, C07,
C08, C10, C11, C12 and C13, each with a written unblock condition a proponent must
affirmatively satisfy. Nothing in the hold column is dead — but nothing in it is
schedulable either, and that is the reopen's practical finding: **of the recon's
seven recommended strikes, exactly one survived adversarial verification**, and
only because C14 was already registry-ineligible before the recon was written. The
rest were not wrong about the candidates being unpromising; they were wrong about
the evidence being sufficient to close them.

---

## 6. The reopen's main output — corrected strategic finding

The recon's §9 claims the backlog is bounded by *"three independent ceilings"*,
each under the Stage-0 bar. Verification withdraws the independence claim (§3.2)
and the re-ordering scope (§3.1). **The observation survives in a corrected and
sharper form, and the corrected form is more useful:**

**What actually bounds this channel is conversion, not reach.** Every candidate
remaining in the backlog acts on the retrieval-key/representation channel or on
the input-text channel, and this campaign has repeatedly measured **large** oracles
in that channel that **fail to convert**:

- **AGGNET/F98** carried an oracle of `+0.1492 / +0.1520 / +0.2186` with 96–100 %
  of every deployed error inside its function class, and delivered
  `+0.0134 / −0.0069 / +0.0000`. *(Two primary records phrase the superlative
  differently and the narrower one is used throughout: `directions_tried.json`'s
  F98 entry does say verbatim "the LARGEST ORACLE CEILING EVER MEASURED ON THIS
  OBJECT", but `findings.jsonl` F98 says "BY FAR THE LARGEST ORACLE CEILING **ANY
  MEMBER OF THIS FAMILY** HAS EVER HAD" and `AGGNET_PREGATE_RECORD.md:678` is
  narrower still. The family-scoped phrasing is the conservative record and is the
  one this reopen relies on.)* Its own epitaph is the finding: *"What binds is
  neither reach nor capacity but that the local configuration carries no learnable
  signal about which neighbours to trust at n = 549–744."*
- **Re-ranking** has a rank-constrained oracle of `+0.0780 / +0.1123 / +0.1876` —
  2.6× to 6.3× over the final bar — and is closed by **reduction** to that same
  measured-dead re-weighting family, not by an under-bar ceiling.
- **Pool expansion** is the one genuinely small ceiling: `coverage(20) ≥ 0.9829`
  head-space gives a marginal oracle `≤ +0.0171`. This bounds *expansion only*.
- **Purity conversion**, where measured, is `d(acc)/d(purity) = +0.0410` observed,
  with perfect purity implying `+0.0286` at the 95 % upper bound — but this is one
  weak, observational, MHC-ZH-dev-only, `n = 78` within-trajectory association
  (`R² = 0.027`), which F114 requires be quoted with those caveats and forbids
  citing as a door-closer.
- **On the input side**, C02 measured the quantity/shape channel at
  `+0.0009 / −0.0012` with changed items flipping at coin-flip precision
  (`0.5040 / 0.4881`).

**The consequence for the reopen is a screening rule, not a ranking.** A large
oracle is no longer evidence for a candidate in this channel — it is the
precondition every failed candidate already met. What a candidate must now show at
Gate 0 is a reason its mechanism **converts**, and `banned_constraints[10]` already
names the currency: NET ITEMS against `22.3 / 17.4 / 16.5` for `+0.030`, i.e.
`37.2 / 29.0 / 27.5` for `+0.050` — with exchange rate explicitly **not** a
screening criterion.

**And the replenishment problem is real.** After C04 resolves, the backlog holds
one candidate with a live mechanism and a zero-cost Stage-0 (C09), one `$0`
falsifier (C06), and seven holds each blocked on a condition someone must
affirmatively satisfy (C05, C07, C08, C10, C11, C12, C13). A Gate-0 reopen that only re-orders
the remaining candidates will re-derive this document in three candidates' time.

---

## 7. Paper material — recorded so it is not lost

**NOTE (limitations / pillar-4).** The `<em class="keyword">` markup in the
MultiHateClip-ZH text channel is not neutral nuisance: it is **a strong lexical
shortcut** — markup-bearing rows hate at `5×` the no-markup rate and at `8×` the
rate of rows carrying a harvested keyword *without* the markup — and it is a
**collection artifact**. *(No ranking against other lexical features was computed,
so "a strong shortcut" is what the measurement supports; "the strongest" is not
established and is not claimed.)* Measured
on `data/gt/MHC_zh/{train,val}.jsonl` (independently reproduced twice):

| split | hate rate, `<em>` present | hate rate, `<em>` absent | base rate |
|---|---|---|---|
| train | **0.5802** (141/243) | 0.1161 (39/336) | 0.3109 |
| val | **0.5882** (20/34) | 0.1818 (8/44) | 0.3590 |

**On the train split**: 49 distinct keywords over 254 occurrences (`em` 254 /
`/em` 254, no other tag); top terms are slurs (`傻逼` 42, `阴茎` 20, `娘炮` 16,
`傻屌` 11, `公主病` 11). Over **train + val** the census is 50 keywords / 288
occurrences. Markup as a fraction of row characters, train split: median
`0.0000`, **p90 `0.5051`** (`lower`) (`0.5071` `linear`, `0.5155` `higher`), max `0.8621`, with `203/579` rows above 10 % — substantial against a train median text
length of 106 characters (val 108.5), and among the 243 markup-bearing rows the
median fraction is `0.2604`.

Provenance: the markup is the Bilibili search-result highlight around the term the
clip was harvested by, riding on the harvested **title** that F88 ledger
correction (c) identifies as the first component of the deployed ZH text. As
`LITSWEEP3_ZH_SPECIFIC.md:36-37` puts it, the highlight *"is baked into the
current 0.8537 floor and inadvertently surfaces the slur."*

**Why it is paper material rather than a lever.** Part of the reported ZH floor
rests on a marker of how the corpus was collected rather than on video content —
a limitation worth stating plainly, and a natural pillar-4 (auditable/editable
archive) illustration. It is **not** a performance candidate: removing it deletes
a feature that predicts the label at five times the no-markup rate.

---

---

## 8. Independent review — round 1

A fresh reviewer with no exposure to this adjudication's reasoning was given the
recon, this record, the landed state and a written review request scoped to three
questions only: are the strikes faithful to their evidence and applied within its
written scope; are the statuses the correct *kind* of record; is anything recorded
as measured that is not. It was also asked to verify the downgrades, on the
principle that a wrong downgrade is as much a defect as a stretched strike.

**Round 1 verdict: `REVISE (1 Critical / 3 High / 10 Important)`.** All fourteen
findings are applied above. The reviewer independently re-derived the census from
the six gt label files and reproduced every load-bearing figure, re-verified the
C01 arm table cell-for-cell, and confirmed the C12/F55 leg — the decisive
downgrade — as correct. It also stress-tested the C13 measurement in a way this
record had not: rows containing a harvested keyword **without** the markup hate at
only `10/140 = 0.0714` against `141/243 = 0.5802` with it, so the tag itself
carries the signal and C13's regression inference survives.

| # | finding | disposition |
|---|---|---|
| **C-1** | C07 struck on a precondition never *attempted* and a screen never *run for C07's object*; F82's own ban_scope splits vote-side from head-side and C07 is head-side | **C07 moved from strike to `held_lattice_delta_unwritten_reachability_unscreened`** (§4.2) |
| **H-1** | C08 premise 2's "no ≥2-dataset substrate" was measured for HTML *tags* only; MHC-EN carries entities on `64` train / `9` val rows | premise restated as **provenance-*marker*** scoped; entity counts added to §2 and §4.1 |
| **H-2** | C11's headwind citation truncated before *"No legal unmeasured lever found."* | quoted in full; the hold now carries an explicit burden, and says C11 should be struck without further measurement if that burden is not met |
| **H-3** | C12's F60 unblock route omitted F60's own *"Do not re-propose without D7 generator-role sub-ruling"* | D7 clause added to `gap_2` and to the unblock |
| **I-1** | M-1/M-2 are percentile/median *convention* differences, applied consistently by the recon — not errors | restated as convention notes |
| **I-2** | The C06 "bans do not reach C06" warrant cited ensembling carve-outs whose object is C14 | re-based on object mismatch; carve-outs demoted to a parenthetical |
| **I-3** | "random rotation matches or beats … on both datasets" is a best-of-six | full rotation spread stated (`0.8505–0.8692`, `0.8462–0.8974`); "best of six" made explicit |
| **I-4** | "bit-identical on 6/6 seeds" overstates precision — F113's artifacts store 4 dp | restated as identical at the recorded 4-dp precision |
| **I-5** | F88 HateMM seed-invariance transcribed `88–93 %`; source says `89–93 %` | corrected |
| **I-6** | The five validity gates live in `C02_A0_OUT.json`, not `C02_A0_DECISION.json` | path corrected |
| **I-7** | C14's landed registry status and its disposition-block status disagreed; `prior_status` recorded a string appearing nowhere else | both reconciled; `prior_status` set to `ordered_backlog` |
| **I-8** | C09's legality citation one-sided; `LITSWEEP5_COMPLETENESS.md` §4(ii) flags the ruling's viability premise as stale | counter-text added verbatim; legality verdict unchanged, prior explicitly weakened |
| **I-9** | C08 premise 3 used past scope — TIER-2 proxy, ZH test split, source declines to settle it | restated as non-significant and underpowered; strike rests on premises 1–2 |
| **I-10** | The paper-note keyword census is train-only but read corpus-wide | labelled; train+val figures (`50` keywords / `288` occurrences) added |

**Kind-of-record and reversibility** were checked and passed on all ten entries in
round 1: the reversibility language is present and uniform, the historical
`ordered_backlog` is genuinely untouched, C09's preregistration is a draft only,
and nothing in C02 or C04 was modified.

---

## 9. Independent review — round 2

A second fresh reviewer, given the same request plus round 1's verdict, was asked
first to audit whether round 1's findings were genuinely applied.

**Round 2 verdict: `REVISE (0 Critical / 3 High / 7 Important)`.** No Critical: the
C07 downgrade, the four downgrade justifications, the three confirmed strikes, the
reversibility language on all ten entries and the untouched `ordered_backlog` were
all checked and cleared, and the reviewer independently re-derived the census and
the C01 arm table again with every load-bearing figure reproducing. It confirmed
eleven of round 1's fourteen findings as applied and caught that **four had been
applied to this narrative record but not to the machine-readable disposition block
in `TARGET_STATE.json`** — a real defect, since a machine consumer reads the JSON.
All round-2 findings are applied.

| # | finding | disposition |
|---|---|---|
| **H-1** | C11's `ERRPAT §5.2` leg carries no tier/split qualifier, though it is the *same* TIER-2 CPU-re-mint proxy on the *same* MHC-ZH test split as the §5.4 result this record demotes; and the cross-dataset "under 76 chars" census applies MHC-ZH's own quartile cut to corpora with medians of 696 and 369 | §4.2 now states the tier, split (`n = 149`), protocol and pooled-only limitation (class halves `p = 0.0506` / `0.0668`), notes the `test_rule` tension explicitly, and **withdraws** the cross-dataset census as substrate evidence |
| **H-2** | *"the largest oracle ceiling ever measured on this object"* attributed to F98 | **partly upheld, and corrected in the safer direction.** The phrase *is* verbatim in `directions_tried.json`'s F98 entry — the reviewer checked only `findings.jsonl` — but `findings.jsonl` F98 and `AGGNET_PREGATE_RECORD.md:678` are both narrower, so §6 now uses the **family-scoped** phrasing and records that two primary records disagree |
| **H-3** | four round-1 repairs (I-2, I-3, I-10, the C10 hedge) landed in the markdown but not in `TARGET_STATE.json` | all four ported into the JSON verbatim |
| **I-1** | C07's unblock (c) imports F82's head-side clause onto C07's object — the same over-application the Critical was about | made conditional |
| **I-2** | `held_nonisomorphism_gate_unwritable` overclaims from a non-exhaustive enumeration the record calls an "extension" elsewhere | renamed **`held_nonisomorphism_gate_unwritten_as_posed`** in all four surfaces |
| **I-3** | two pinpoint citations wrong in a record about citation fidelity: F99 cites `:458-459` not `:463`; `LITSWEEP8:218-222` is the hubness table | corrected to `:458-459` (noting the sentence itself sits at `:463`) and re-pointed to `:268-273` |
| **I-4** | C14's `prior_status_note` re-asserts a string not recoverable from any committed source | note now states its actual provenance (the pre-edit working tree) |
| **I-5** | the C09 legality quote of `LITSWEEP3:82` truncated before *"(though Wall-A still caps the achievable magnitude)"* — the truncation pattern round 1 charged at C11, reproduced at the promotion | parenthetical restored, with the section's own `~1-2 %` and `+0.001-0.006` pricing |
| **I-6** | the MHC-EN entity histogram is a train+val occurrence count shown beside per-split row counts; and the p90 convention vocabulary was self-inconsistent | histogram labelled train+val with train-only added; numpy's `lower`/`linear`/`higher` used uniformly |
| **I-7** | F82's headwind quoted with *"HateMM out of scope (no Offensive class)"* elided, and the ceiling's resolution unstated | all three limits restored: HateMM out of scope, dev-split gold cheat at `n = 80` / `n = 78`, ZH's `+0.0256` = **2 dev items**, and the record's own pre-declaration that the oracle does not bound representation reshaping |

**What round 2 did not disturb.** The three confirmed strikes, the four downgrades,
the strategic finding, the C09 promotion and its legality verdict, and the
zero-cost/zero-touch boundary all stand. Every finding was one of scope, precision
or surface consistency — none was a fabrication, and the reviewer states plainly
that no fabricated number was found.

---

## 10. Independent review — round 3

A third fresh reviewer audited both prior rounds' findings across all four surfaces
and re-derived the census and the C01 arm table again.

**Round 3 verdict: `REVISE (1 Critical / 1 High / 4 Important)`.** It confirmed all
fourteen round-1 and all ten round-2 findings as applied except two surface gaps
(charged as I-1 and part of H-1), independently verified round 2's H-2 adjudication
as handled honestly, and cleared all four downgrades, both surviving strikes, C06's
gate, C09's promotion and legality, the strategic finding and the three-surface
status agreement. All findings are applied.

| # | finding | disposition |
|---|---|---|
| **C-1** | C08's premise 1, as restated in revision 2, is **refuted by primary evidence in this repository**: `Multihateclip/{English,Chinese}/annotation(new).json` carry a non-empty `Title` on `891/891` and `897/897` rows, `prep_mhc.py:72-85` reads title and transcript as **separate variables**, and F88's *"title 15 chars"* median is only measurable from a separated title. Emitting a title-separated gt is a CPU re-prep, not the "data-collection act" the record claimed — so a `≥2`-dataset title route exists on MHC-EN + MHC-ZH | **C08 moved from strike to `held_title_channel_separable_route_unscreened`** (§4.2), with premise 2 kept at marker scope, premise 3 demoted to corroboration, and the title half named as the unscreened residual |
| **H-1** | round 2's I-6 repair introduced a **transposed** train-only entity triple (`43 / 17 / 16`; true value `43 / 16 / 17`) inside a table certified "exact", was not applied to the other two surfaces, and certified an entity row count without stating its regex convention | corrected to `43 / 16 / 17`, ported to §4.2 and `TARGET_STATE.json`, and both conventions recorded (`1` strict, `2` hex-inclusive on MHC-ZH train) |
| **I-1** | `TARGET_LOOP.md` still said C08 rested on *"three measured premise failures"* after §4.1 had demoted premise 3 | moot under C-1; the row now records the hold and its basis |
| **I-2** | `[5]`/`[6]` applied as a blanket "MLLM output" ban at C05 and C07, without the EUM-vs-F60 tension this record itself sets out at C12 | both unblocks now state the attribution as **construction-dependent**, citing `[5]`'s literal four-word text and LBOP's *不输出 label、score、memory pair/key 或 rationale* |
| **I-3** | round 2's I-5 repair cited `LITSWEEP3:94`; the *"+3 any dataset: ~1-2 %"* figure is at `:95`, and `:69` carries a different `~2 %` belonging to the ELR section | corrected to `:95` |
| **I-4** | M-1 claimed §7 uses all three percentile labels; §7 omitted `higher`/`0.5155`, the very figure M-1 exists to explain | `0.5155` (`higher`) added to §7 |

**The pattern across three rounds, stated because it is the reopen's methodological
finding.** Every Critical was the same defect in a different place: a candidate
recorded as struck on evidence that, read at its own written scope, does not close
it — C07 on a screen never run for its object, C08 on a premise the source files
refute. Both were caught only because each round was given the primary sources and
told to be adversarial. **Of the recon's seven recommended strikes, two survive.**
That is not a criticism of the recon, which was explicitly advisory and whose
measurement layer proved sound throughout; it is the reason the registry requires
adjudication before a recon moves a status.

---

## 11. Independent review — round 4

**Round 4 verdict: `REVISE (0 Critical / 2 High / 2 Important)`.** The reviewer
audited **all thirty** prior findings against all four surfaces and confirmed every
one genuinely applied — including in the JSON, which rounds 2 and 3 had both caught
lagging the prose. It re-derived the census from scratch, recomputed the C01 arm
table cell-for-cell, re-verified the 6/6 fold-head match, and checked every
load-bearing quote at its cited scope; **no fabricated number was found**. It
cleared all five downgrades, both strikes, C06's gate, C09's promotion and legality,
the strategic finding, the `$0`/zero-touch boundary, and confirmed **no
over-cautious hold** — each hold names a usable unblock and is the right kind of
record. All four findings are applied.

| # | finding | disposition |
|---|---|---|
| **H-1** | C08 was filed under the **§4.1 "Strikes CONFIRMED"** heading despite being a hold — the record was the only surface with the grouping wrong (the JSON, `TARGET_LOOP.md` and `TARGET_FINDINGS.md` all had it right) | C08 moved into §4.2 as its fifth entry; §4.1 restored to C13 + C14 |
| **H-2** | C08's unblock priced the title channel from *"title 15 chars … composed 96"* — an **MHC-ZH, test-split, markup-stripped** median inherited second-hand via F88 from `ERRPAT_MHC-ZH:270-271` — and generalised it to a route whose EN leg is far larger | qualified and **measured per dataset**: title median **51 chars on MHC-EN** (transcript 322) vs **27 raw / 13 stripped on MHC-ZH** (transcript 78), so the EN half is `3.4×` larger than the figure the "thin channel" verdict rested on and is explicitly recorded as **unpriced**. The same tier/split qualifier added to ERRPAT §5.3 at C07 |
| **I-1** | (a) *"exhibit a provenance artifact present on `≥2` datasets"* was quoted as **C08's own written unblock**; the string exists nowhere but this record's round-3 review text. (b) F113's honesty clause was attributed to *"F114/F113"*; it is in **F113 only** | (a) quotation withdrawn and the point stated directly, with the provenance noted; (b) attribution corrected |
| **I-2** | C13's strike was recorded as resting on *"measurement alone"*, but the operative step (*"a plausible regression"*) is an **inference**; meanwhile a stronger ban-free measured leg sat unused | **re-based on substrate arithmetic**: the `<em>` marker exists on **one of three datasets** (`0` on HateMM, `0` on MHC-EN) against the two-dataset Stage-0 bar — the same arithmetic used at C11. The hate-rate measurement is retained as a corroborating headwind with its inferential step **labelled as one** |

---

## 12. Independent review — round 5

**Round 5 verdict: `REVISE (0 Critical / 2 High / 2 Important)`.** The reviewer
audited all thirty-four prior findings across all four surfaces and found
**thirty-three genuinely applied**; re-derived the census, the C01 arm table, the
C02 fold-head match and the C08 title measurements from scratch, all reproducing
exactly; and checked every load-bearing quote at its cited scope, finding **no
fabricated number**. It cleared both strikes' kind-of-record, all five downgrades,
the six holds' unblocks, C06's gate, C09's promotion and legality, and confirmed
again that **no hold is over-cautious**. All four findings are applied.

| # | finding | disposition |
|---|---|---|
| **H-1** | round 4's H-2 was unapplied on `TARGET_LOOP.md` — its C08 unblock still priced the title channel at *"15 characters"*, the superseded MHC-ZH test-split markup-stripped figure, twenty lines before the same file described the correction | the per-dataset pricing ported to `TARGET_LOOP.md` |
| **H-2** | C13's sole ban-free leg asserted no `≥2`-dataset substrate, but was measured for HTML **tags** only, while C13's claim says *"HTML/title markup"* and MHC-EN carries entities on `64/549` + `9/80` rows — the identical scoping defect round 1 forced out of C08's premise 2, this time carrying a surviving strike alone | **C13 re-based on its own registry text**: its claim declares the target *"a ZH-specific extraction nuisance"* and its boundary conditions any two-dataset route on a cross-dataset mechanism it never names. The census now **corroborates** rather than establishes, and the tag-scoped arithmetic is withdrawn as a standalone leg |
| **I-1** | the attestation *"the only files opened for measurement were the six gt files"* became false once rounds 3-4 added measurements from `annotation(new).json` and recomputations from `C01_A0_OUT.json` | provenance restated in full on all three surfaces: gt census, the two source annotation files joined to train+val ids, and recomputation from banked result artifacts |
| **I-2** | `TARGET_STATE.json`'s `dispositions.held` array contained **C06**, a gate, contradicting its own tally — the machine-readable mirror of the grouping error round 4 charged at High in the prose | C06 moved to a new `dispositions.gated` array |

---

## 13. Independent review — round 6

**Round 6 verdict: `REVISE (0 Critical / 1 High / 2 Important)`.** The reviewer
re-derived the census, the C01 arm table, the C08 title measurements and the C02
fold-head identity from scratch — all reproducing — checked every load-bearing
quote at its cited scope, found **no fabricated number**, and confirmed all
thirty-eight prior findings applied except the two charged below. It cleared both
strikes' kind-of-record, all downgrades, every hold's unblock, C06's gate, C09's
promotion and legality, and confirmed once more that no hold is over-cautious. All
findings are applied.

| # | finding | disposition |
|---|---|---|
| **H-1** | C13's basis was stated three incompatible ways after round 5's partial repair, and the surviving basis — an un-named cross-dataset pairing — is a **precondition a proponent can satisfy**, which this record's own C07 Critical holds cannot carry a strike. Two supporting claims also failed: *"not an inference"* (the step from self-scoping to the two-dataset bar **is** one) and *"the same basis as C14's"* (C14 carries `eligible_for_primary_target: false`; C13 carries no such flag) | **C13 moved to `held_zh_scoped_no_cross_dataset_pairing_named`**, with the full history of how its basis moved recorded, and the paper finding preserved in §7 |
| **I-1** | round 5's provenance repair was itself incomplete: §3.7's correction is measured from the six banked `headspace_arena_*_OUT.json`, a fourth source the attestation did not name | added as source (iv) on all four surfaces |
| **I-2** | *"the strongest lexical shortcut in the ZH text channel"* is an **unranked superlative** recorded inside the measured layer — what is measured is `5×` the no-markup rate and `8×` the bare-keyword rate, with no ranking against other features computed | softened to *"a strong lexical shortcut"* with both ratios stated, and the absence of a ranking disclosed, on all four surfaces |

**Where six rounds leave the reopen.** Of the recon's seven recommended strikes,
**one** survives — C14, and only because it was already registry-ineligible before
the recon was written. Every other strike collapsed under adversarial reading, and
each collapsed the same way: the evidence offered closed something narrower than
the candidate, or named a precondition a proponent could still satisfy. **That is
the reopen's most important output about method, not about candidates**: an
advisory recon is a hypothesis-generator, and the gap between "this candidate looks
unpromising" and "this candidate is closed" turned out to be seven candidates wide.

---

## 14. Independent review — round 7

**Round 7 verdict: `REVISE (0 Critical / 2 High / 3 Important)`.** Both Highs were
round-6 repairs lagging on a surface — the pattern every round since round 2 has
caught, and the reason each round re-checks all four. Asked explicitly, now that
only one strike remains, whether any hold is **over-cautious**, the reviewer
confirmed none is: at C10 in particular, EUM's ban does name the object, but it
supplies its own three revival preconditions, so a strike would over-read a
conditional closure as an absolute one. All five findings are applied.

| # | finding | disposition |
|---|---|---|
| **H-1** | §5 — the *effective post-C04 order*, the section a consumer schedules from — still read *"`C13`, `C14` struck"*, *"Two strikes, six holds"* and *"only two survived"*, contradicting §4.2, §13 and all three other surfaces | §5 corrected throughout: one strike, seven holds, C13 in the unblock-conditional list |
| **H-2** | round 6's unranked-superlative repair was unapplied on `TARGET_FINDINGS.md`, which still read *"the strongest lexical shortcut"* | softened there too, with the `5×` and `8×` ratios and the no-ranking disclosure |
| **I-1** | `TARGET_LOOP.md` said *"five of its seven"* downgraded (six were), and its *"six downgrades"* section carried only five entries — C13's gap appeared nowhere on that surface but a table row | count corrected and a C13 gap paragraph added |
| **I-2** | the C09 counter-text rests its selector leg on the **F114-retracted** *"CLIP LOO 0.998"* premise, and `LITSWEEP5_COMPLETENESS.md` is **not** among F114's nine corrected records — so the record applied F114 inconsistently, and the error ran *against* the candidate it promotes | recorded as **downgraded, not vacated**, with §4(ii)'s independent train-disagreement leg (`0/109`, `0/102`, `0/92`) noted as untouched |
| **I-3** | C10's unblock transcribed EUM's **compressed** statistic (*"median 83 % … in a single contiguous block"*) rather than the two measurements behind it, and dropped EUM's own *"recon-grade, not gate-grade"* qualifier | both statistics stated separately (coverage median `0.8289`, mean `0.7174`; `74.2 %` single contiguous span, `n = 298`) with the grade qualifier restored |

---

## 15. Independent review — round 8

Round 8 was asked to make the **cross-surface consistency sweep** its first and
most exhaustive task, because every round from 2 to 7 had caught a repair lagging
on exactly one surface.

**Round 8 verdict: `REVISE (0 Critical / 2 High / 4 Important)`.** No disposition
changed. The reviewer re-derived the whole `[M]` layer from scratch, recomputed the
C01 arm table from the stored confusions, re-read the six banked arena files and
C02's gates, and checked every load-bearing quote — **no fabricated number was
found**, every census figure reproduced, the C14 strike verified faithful and
in-scope, **all six downgrades verified justified** (with C10, C11 and C12
independently re-derived), and every hold and the gate confirmed to name a usable
unblock with none over-cautious. It also surfaced a **new corroboration** of the §7
paper note: the ZH `<em>` markup lives in **391 `Title` fields and 0 `Transcript`
fields**, independently confirming that the marker rides on the harvested title.

| # | finding | disposition |
|---|---|---|
| **H-1** | round 7's count repair on `TARGET_LOOP.md` appended *"— six of the seven"* without deleting *"five"*, leaving a self-contradiction in the headline of the surface a consumer schedules from | rewritten cleanly |
| **H-2** | **round 7 did not exist on three of the four surfaces** — `TARGET_LOOP.md`, `TARGET_FINDINGS.md` and two `TARGET_STATE.json` summary strings all still said *"six rounds"* and pointed at a file list omitting `ROUND7.md`, contradicting the JSON's own `independent_review.round_7` block | all three surfaces updated, round-7 verdict and paragraph added, Records lists extended |
| **I-1** | `TARGET_FINDINGS.md` attributed the **derived** `37.2 / 29.0 / 27.5` net-item triple to `banned_constraints[10]`, which names only `22.3 / 17.4 / 16.5` for `+0.030` | restated as the `+0.030` figures scaling to the `+0.050` ones, matching the other three surfaces |
| **I-2** | `LITSWEEP3_DATA_CENTRIC.md:80` was invoked as *"an explicit house rule against exactly this move"*, but its subject is memory-bank curation and its stated ground is that the null used an **MLLM two-vote** signal — the side C12's archive-derived statistic sits on. The record charged the recon with exactly this fault | re-cited as a **general form**, with the source's own subject and ground stated, and the C12 downgrade rested on its independent inference-time leg |
| **I-3** | round 7's I-2 repair (the C09 counter-text is **downgraded, not vacated**) reached the record and the JSON but not `TARGET_LOOP.md` or `TARGET_FINDINGS.md`, which still carried an overstated headwind against the promoted candidate | added to both |
| **I-4** | *"D7 is an open user ruling"* is wrong as written — `DECISION_MEMO_pending.md:134,211` records D7 as `RESOLVED 2026-07-14 (RESOLVED-NEGATIVE)`; what F60 asks for and what is open is the **D7 generator-role sub-ruling**, which the record's own unblock already stated correctly | narrowed on both surfaces |

---

## 16. Independent review — round 9

**Round 9 verdict: `REVISE (0 Critical / 1 High / 3 Important)`.** No disposition
changed. The reviewer re-derived the entire `[M]` layer from scratch, recomputed
the C01 arm table cell-for-cell on both datasets, re-verified the 6/6 fold-head
identity, and checked every load-bearing quote — **no fabricated number was
found**; the C14 strike verified faithful and in-scope; **all six downgrades
verified justified**, with C10, C11 and C12 independently re-derived; every hold
and the gate confirmed to name a usable unblock with **none over-cautious**. Its
findings were that three of round 8's six repairs had not actually landed.

| # | finding | disposition |
|---|---|---|
| **H-1** | round 8's H-1 was recorded as *"rewritten cleanly"* but was not: `TARGET_LOOP.md`'s headline still read *"**five** of its seven … downgraded to HOLD — **six of the seven**"*, because the repair had edited the tail and left the "five" earlier in the same sentence | rewritten properly, once, with balanced emphasis |
| **I-1** | round 8's I-4 did not reach `TARGET_STATE.json`, whose C12 `gap_2` still asserted *"D7 is an OPEN USER RULING"* while the same object's `unblock` said the opposite | narrowed in the JSON: D7 itself is `RESOLVED 2026-07-14 (RESOLVED-NEGATIVE)`; the **generator-role sub-ruling** is what is open |
| **I-2** | round 8's I-2 landed only on the record; the JSON, `TARGET_LOOP.md` and `TARGET_FINDINGS.md` still presented `LITSWEEP3:80` as a *governing* rule rather than a general form | all three re-scoped, with the C12 downgrade rested on its independent leg |
| **I-3** | **the provenance attestation was false again**: it claimed the annotation files were *"joined only to train+val ids"*, but the `Title`-presence counts `891/891` and `897/897` — and round 8's own `391 Title / 0 Transcript` corroboration — are **whole-file**, spanning test ids. Nothing improper followed (only `Title`/`Transcript` were read, no label consumed, and the join-scoped counts make the identical point), but this is the same attestation class rounds 5 and 6 each repaired | attestation restated exactly on all four surfaces, with the join-scoped figures `629/629`, `657/657` and `277 / 0` given alongside, and an explicit "no label or prediction was read for any test id" |

**What nine rounds have settled.** Since round 6 no disposition has moved, and
since round 3 no Critical has been raised. Rounds 7, 8 and 9 found only
cross-surface lag and attestation precision — real defects, all repaired, none
touching a verdict. The dispositions have been stable across three consecutive
adversarial reviews that each independently re-derived the evidence.

---

## 17. Independent review — round 10

**Round 10 verdict: `REVISE (0 Critical / 1 High / 3 Important)`.** No disposition
changed. The reviewer re-derived the `[M]` layer independently and reports **18/18
claims matching exactly**, recomputed the C01 arm table cell-for-cell, confirmed the
6/6 fold-head identity and the §3.7 pairing correction, verified `≈ 36 heads` from
`K_FOLDS = 5` plus the deployed head, and found **no fabricated number**. It cleared
the C14 strike as faithful and in-scope, **all six downgrades** with C10/C11/C12
re-derived from primary text, the kind-of-record and reversibility on all ten
entries, and confirmed **no over-cautious hold**. All four findings are applied.

| # | finding | disposition |
|---|---|---|
| **H-1** | **round 9 did not exist on two of the four surfaces** — `TARGET_LOOP.md` and `TARGET_FINDINGS.md` still said *"eight rounds"* in five places and listed `ROUND{1..8}`, with zero hits for round 9. The third consecutive recurrence of the same lag defect | both surfaces synced, rounds 9 and 10 added, Records lists extended to `{1..10}` |
| **I-1** | the C08 title-median provenance cited `ERRPAT_MHC-ZH:270-271`; `:271` is the composition line and the medians are at **`:272`** | anchor corrected on both surfaces |
| **I-2** | the broad reading of `banned_constraints[5]` was attributed to EUM *glossing* `[5]`; EUM actually reaches "MLLM-derived boundaries or weights" through a **four-authority stack** (*"P3 / P11 plus `[5]` and `[6]`"*) — and in a record whose methodological finding is that authorities must be read at their own scope, the attribution matters | restated as a stack, with the note that the error ran in the **conservative** direction so C12's disposition is unaffected |
| **I-3** | F80 was quoted at partial scope in the C06 warrant: its ban_scope opens with an **unconditional on-dataset** closure (*"any language, any stream, either encoder arm — on MHC_zh"*), and the *"without new mechanism"* conditionality attaches only to *elsewhere* — the record charged the recon with truncating this entry and then truncated its other half | quoted at full scope; the warrant is unchanged, resting on object mismatch |

**Where ten rounds leave it.** No disposition has moved since round 6, and no
Critical has been raised since round 3. Rounds 7-10 found only cross-surface lag
and citation precision — every one real, every one repaired, none touching a
verdict. Four consecutive adversarial reviews have independently re-derived the
evidence and left the ten dispositions standing.

---

## 18. Independent review — round 11

**Round 11 verdict: `REVISE (0 Critical / 1 High / 2 Important)`.** No disposition
changed. The reviewer re-derived all **18** `[M]` census claims from scratch and
reports every one reproducing exactly — including a check the record had not made:
round 1's stress test reproduces **only** under the train-only keyword set
(`10/140 = 0.0714`; train+val gives `10/146`), confirming the record's scoping is
the one that works. It recomputed the C01 arm table across **all 14 arms × 2
datasets to `<1e-9`**, re-verified the 6/6 fold-head identity, and found **no
fabricated number**. It cleared the C14 strike, **all six downgrades** re-derived
from primary text, the reversibility string on all ten entries, and **no
over-cautious hold**.

| # | finding | disposition |
|---|---|---|
| **H-1** | round 10's I-3 repair did not land on `TARGET_FINDINGS.md`, which still said F80's *"prohibition is conditional on 'without new mechanism'"* — the fourth consecutive recurrence of the surface-lag class | corrected to F80's full scope: the on-`MHC_zh` closure is **unconditional**, and the conditionality attaches only to *elsewhere*. The warrant is unchanged, resting on object mismatch |
| **I-1** | *"markup-stripped"* was recorded as a property of a primary measurement, but **neither `ERRPAT:272` nor F88 states the convention**, and F88 actually describes the title as *carrying* the markup. It is a reconciliation inference — `15` sits near this record's own measured stripped value `13` rather than the raw `27` | marked as **inferred** on all four surfaces; the load-bearing claim (EN title median `51`, `3.4×`, unpriced) is a direct measurement and is untouched |
| **I-2** | *"F112 carries the same caveat independently"* is half true: F112 carries the raw-vs-head half but **not** the TEST-transfer clause, which is F113's alone | narrowed on both surfaces; the C06 gating rationale is unaffected because it leans on the half F112 does carry |

---

## 19. Independent review — round 12: **GO**

**Round 12 verdict: `GO (0 Critical / 0 High / 2 Important)`** — the first GO, and
the first round to clear all four scope items explicitly. The reviewer verified
through **four independent passes** (three fact-check workers plus its own
recomputation):

- **Census re-derived from scratch at 100 % reproduction**, including that the
  round-1 stress test reproduces **only** train-scoped (`10/140`; train+val gives
  `10/146`), that Note M-1's percentile-convention claim is right under numpy's
  three conventions, and that Note M-2's upper-median claim holds across four
  further medians.
- **Title census exact**, whole-file and join-scoped; `prep_mhc.py:72-85` and
  `prep_video_dataset.py:126-139` confirmed to read title and transcript as
  separate variables; `LITSWEEP2:56` confirmed gt-schema-scoped and §3.3's
  "re-scraping YouTube metadata" inference confirmed wrong for both MHC datasets.
  **Round 3's Critical stands.**
- **§3.7 fold-head identity recomputed** — identical at 4 dp on **6/6**, at exactly
  the precision claimed.
- **C01 arm table recomputed across all 14 arms × 2 datasets to `<1e-12`**, with
  both rotation spreads and the 4-of-6 / 2-of-6 counts exact.

**The four scope items, as cleared:** (a) the one strike is faithful and in-scope,
with TVB correctly identified as a *prediction* and not relied on; (b) all ten
statuses are the correct kind of record, carrying the byte-exact reversibility
string, with `dispositions` summing to 10 and the historical `ordered_backlog`
intact; (c) **no inference is recorded as a measurement** — every inferential step
is labelled as one; (d) all eight unblocks are concrete and proponent-actionable,
and **no hold is over-cautious**, with C10, C11 and C12 each re-derived from
primary text.

| # | finding | disposition |
|---|---|---|
| **I-1** | §§3.5 and 3.6 and the JSON's `V-5`/`V-6` still called C07 **struck** — residue of round 1's Critical never swept out of §3, five rounds after C07 was downgraded | corrected to *"C07's first unblock condition"* and *"the hold"* on all four places |
| **I-2** | the C12 unblock re-asserted *"lands on `[5]` under EUM's gloss"* thirty-five lines after the record corrected that to a **four-authority stack** — round 10's I-2 had landed only in `gap_2` | corrected to the stack |

Its observations were applied too: the §§8-13 cross-reference, the percentile
vocabulary, two formatting residues from earlier inserts, F82's
*"(any monotone weighting, any tau)"* restored from behind an ellipsis, the
readout script's `src/utils/` path, `TARGET_LOOP`'s C12 unblock narrowed to the
generator-role sub-ruling, and the `277 / 0` join-scoped figure added to the
`TARGET_FINDINGS` attestation.

---

## 20. Independent review — round 13: **GO**

**Round 13 verdict: `GO (0 Critical / 0 High / 1 Important)`** — a second GO, from a
reviewer running three parallel fact-check passes plus its own recomputation. It
cleared all four scope items again: the one strike faithful and in-scope; all ten
blocks carrying the **byte-identical** reversibility string with `dispositions`
summing to exactly 10; the whole `[M]` layer recomputed independently at **100 %
reproduction** with the C01 arm table exact to `<1e-12` across all 14 arms × 2
datasets and the fold-head identity at exactly the claimed precision; and all eight
unblocks concrete and proponent-actionable with **none over-cautious**, C10, C11 and
C12 each re-derived from primary text.

It noted one thing **against this record's own interest**: on C11 the record applies
`ERRPAT`'s *"No legal unmeasured lever found"* **more narrowly** than ERRPAT's own
broadest claim — i.e. it gives that hold a heavier burden than the source requires.

| # | finding | disposition |
|---|---|---|
| **I-1** | `TARGET_STATE.json`'s C12 unblock still read *"lands on `[5]` under EUM's gloss"* — the exact string rounds 10 and 12 had **both** charged, surviving because the repair matched a variant with a comma. The record's markdown was correct; the JSON was not, and round 12's block certified the repair as applied | string corrected in the JSON, and **round 12's certification amended rather than left standing** |

**Observations applied.** Two are worth carrying beyond bookkeeping:

- **`banned_constraints[10]`'s net-item figures are *train-arena* requirements** at
  `n = 744 / 579 / 549` (`LITSWEEP7_LANDING_SITE.md:107-111`). Neither the ban nor
  any earlier revision of this record said so, and a proponent applying the Gate-0
  currency to a test-sized arena would mis-scale by roughly `3.5×`. Now stated
  wherever the figures appear (§6).
- **C07's "head-side/representation object" is this record's reading**, not registry
  text — C07's entry contains no head-side language. It is well-founded and the
  adjacent graded-auxiliary step was already hedged, but it is now labelled as the
  one inferential step inside that downgrade.

The rest: the §§ cross-reference; the JSON's percentile vocabulary unified to
numpy's `lower`/`linear`/`higher`; `banned_constraints[5]` described as a bare
phrase rather than "four words"; the `LITSWEEP5` table described accurately (eight
rows, ranks 1-7 plus a parenthesised `(8)`, the *"7 of 7"* rank claim exact);
archive-as-key given its MHC-ZH-only scope with the EN arm's figures; LBOP-0's
fuller bar recorded (macro-F1, per-fold sign agreement and a Farkas/gradient-cone
audit — which makes the unwritten delta *harder*, against the record's own
argument); and `MHCsmoke` noted as an empty sixth cache directory.

---

## 21. Independent review — round 14: **GO (0 Critical / 0 High / 0 Important)**

**This is the closing verdict.** A fourteenth fresh reviewer, running **four
independent fact-check passes plus its own recomputation**, returned
`GO (0C/0H/0I)` — zero findings at any severity — having cleared all four bar items
explicitly:

- **(a) The one strike is faithful and in-scope.** C14 is the **only** entry of all
  fourteen carrying `eligible_for_primary_target`; its `dedup_boundary` and
  `hard_constraints[4]` are byte-exact; TVB's support is correctly identified as a
  **prediction** and not relied on.
- **(b) Kind-of-record and reversibility.** All ten blocks carry the byte-identical
  reversibility string; `dispositions` sums to exactly 10; C01-C04 carry no gate-0
  key; `new_jobs`/`new_metrics` are empty; the C09 preregistration is a `DRAFT`.
- **(c) Nothing measured that isn't.** The census was re-derived from scratch
  **twice, independently**, at **100 % reproduction** on all 18 `[M]` claims; the
  C01 arm table was recomputed at **every one of 28 arm-cells**; §3.7 was
  recomputed by the reviewer itself as identical at 4 dp on **6/6**, with
  bit-equality correctly disclaimed; every inferential step is labelled.
- **(d) Unblocks usable, none over-cautious.** In the reviewer's own words:
  **"No hold should have been a strike, and no unblock is vacuous."**

**Its observations were applied anyway, and five of them tighten the record against
its own arguments** — which is why they were worth taking after a clean GO:

1. **A direct single-authority gloss of `[5]` does exist.** F103/OCR's ban_scope
   glosses it on an **archive field** specifically. F60 conflicts head-on so C12's
   fork is real, but the branch that looked safer is **not free** — added to C12's
   unblock, raising its burden on both branches.
2. **F108 is now named in C08's unblock.** C08 lands in F108's carve-out (ii) —
   content, not weight — as written, but a proponent realising the title route as
   "expose the title as its own key block" walks straight into it.
3. **C06's six "random rotations" are one parameter family, not six directions.**
   `orthogonal_blocks()` is a Givens mixing of the two endpoint blocks; `θ = 45°`
   *is* `common_displacement` and `θ = 0` *is* `endpoint_concat`. The correct
   reading is **sharper and more adverse**. Two arms omitted from the earlier table
   — HateMM `common` `0.8692`/`+3` and MHC-ZH `endpoint_concat` `0.8846`/`+2` — are
   added, because their omission **understated** the adverse case.
4. **LBOP-0's fuller bar** (macro-F1, per-fold sign agreement, a Farkas/gradient-cone
   audit) is now stated **at the point of use** in C07's unblock, making the
   unwritten delta harder.
5. **C10 gains an unpriced headwind**: EUM's own status field records that a legal
   rule-based unit — uniform `K = 4` windows over Whisper word-level timestamps,
   exactly the fourth source precondition (2) posits — **was already built and
   measured negative** (`dF1 −0.0116`, 3/3 seeds). EN-only and on vote keys rather
   than the bank object, so it does not close C10, but a proponent must price it.

Also applied: the second instance of the gt-schema-scoped title error
(`LITSWEEP3_ZH_SPECIFIC.md:39-40`) named alongside the LITSWEEP2 one, and two
descriptor lags corrected. Noted and not acted on: Global-R2's compressed epitaph
is the source's own and is quoted rather than asserted, with both arms sub-bar
regardless; and `ERRPAT:415` is scoped to what is open *in-box at `$0`* while C11 is
a training-time representation change — where the record already applies `:405`
**more narrowly than ERRPAT's own broadest claim**, i.e. against its own interest.

**Closing state of the review.** Fourteen rounds. Two Criticals, both of which moved
a candidate out of the strike column (C07 at round 1, C08 at round 3). No
disposition has moved since round 6; no Critical since round 3; the last three
rounds all returned GO. **The adjudication is closed.**

---

*No GPU, SLURM, Modal, teacher call, model load, training, cache write, test-split
access or job submission occurred in producing this record. The C04 lineage was
not touched. Nothing in C02 was modified.*
