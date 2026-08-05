# C09 Stage-0 (A0) Preregistration — **DRAFT**

**Candidate.** C09 · Stable-Inversion Topology Surgery
**Registry claim.** *"OOF-stable high-confidence inversions identify topological
defects that can be corrected at encoder level while explicitly constraining break
exposure."*
**Registry dedup boundary.** *"Encoder-level topology intervention, not
thresholding, local reranking, verifier gating, NCA/SupCon, or hard-example
weighting alone."*
**Authorised by.** `TARGET_STATE.json::gate0_reopen_2026_07_31` —
`next_active_candidate_post_C04`.

> ## STATUS: DRAFT. NOT FROZEN. NOT SUBMITTED. **REVIEWED ONCE — `REVISE (4 Critical / 8 High / 10 Important)`.**
>
> ### Round-1 design review: **the design as written is not sound and must not be implemented.**
>
> `refine-logs/C09_A0_PREREG_DRAFT_REVIEW_ROUND1.md`. One round of independent
> review was the authorized scope; **the repairs are deliberately not attempted
> here.** The reviewer confirmed the legality argument (both quotes exact, the three
> HALT boundaries adequate for Stage-0 as scoped), the arena and instrument choice,
> the fold-head floors, the head-count budget and the `37.2 / 29.0` arithmetic — and
> then found four Critical defects, two of which are mine and both of which push
> toward a **false CONTINUE**:
>
> 1. **C-1 — the feature set leaks the scored item's own gold label.** §5.2 lists
>    "top-20 purity" and asserts no feature reads the item's own label, while §4.4
>    defines purity *against `i`'s gold label*. `ERRPAT_HateMM:143-145` measures
>    gold-purity `< 0.5` for 24-27 of 26-28 errors, so the feature nearly **is** the
>    target: AUC would land near 1.0 by construction.
> 2. **C-2 — `D-FELDMAN` is not decidable as specified.** H-MEMORISATION does *not*
>    predict AUC ≈ 0.5: Feldman's long-tail singletons are by definition the
>    low-density, weak-margin items the feature set measures, and the separation is
>    already banked (`ERRPAT_HateMM:130`, median `|vote|` `0.7267` errors vs `0.9873`
>    correct). `K-FELDMAN` essentially cannot fire, so `K-NET` would silently carry
>    the whole decision — and §6 has no row for the realistic outcome.
> 3. **C-3 — cross-seed leakage.** Stability is a per-*item* property but the rows
>    are per (item, seed) and pooled; `GATE-NESTED` groups only by arena fold.
> 4. **C-4 — `NET`'s break accounting is non-conservative**, leaving selected items
>    in neither class uncosted, and mis-scoped against a full-arena currency.
>
> Eight Highs follow, of which two are structural: **H-2**, the decision rule leaves
> an undefined outcome (clear on one dataset, fail on the other → neither KILL nor
> CONTINUE); and **H-8a**, the reopen's *first* quoted kill-risk (F75/NCA) is
> addressed nowhere — and it is a Stage-**0** problem, because `H-L1` bans the only
> region-locator this A0 builds, so a CONTINUE would have **no legal Stage-1
> successor** unless the encoder-training route is named and shown not to be F75's
> object.
>
> The reviewer's bottom line: *"Fix C-1 through C-4 and H-1 through H-4 and this
> becomes a genuinely decisive `$0` pregate … the first design in this channel that
> could actually separate a fixable topology defect from a memorisation-necessary
> error rather than re-measuring atypicality."*
>
> **Nothing below has been revised in response.** Read the sections that follow as
> the reviewed object, not as a corrected design.
>
> ### Original status block
>
> No hash is frozen, no config is written, no code exists, no job is submitted and
> no namespace is created by this document. It is a design for review only.
> Execution requires, in order: (1) C04's teacher tranche to terminate
> (serial-execution precedent, `serial_execution.one_candidate_at_a_time`);
> (2) explicit main-dialogue authorization; (3) implementation plus a fresh
> independent code/resource review to `GO (0C/0H/0I)`; (4) a hash freeze of every
> threshold, control, arm, metric and decision rule **before** the single
> authorized submission.

---

## 1. What A0 asks, and why this one is different

Stage-0 under `unified_pilot_gate.stage_0_reachability` asks whether the
representation-level oracle reaches `+0.050 accuracy` **and** `+0.050 macro-F1` on
at least two datasets before any teacher or GPU spend.

For C09 the oracle question splits in two, and **only the second one is
informative**. The campaign has repeatedly measured large oracles in this channel
that failed to convert — AGGNET/F98 held `+0.1492 / +0.1520 / +0.2186` with
96–100 % of every deployed error inside its function class and delivered
`+0.0134 / −0.0069 / +0.0000`, with the epitaph *"What binds is neither reach nor
capacity but that the local configuration carries no learnable signal about which
neighbours to trust at n = 549–744."* The Gate-0 reopen recorded this as the
campaign's governing lesson: **a large oracle is no longer evidence for a
candidate in this channel — it is the precondition every failed candidate already
met.**

So A0 measures **reach** as a cheap necessary condition and **identifiability** as
the actual test:

1. **Reach (`O1`).** Is the OOF-stable-inversion population large enough that
   fixing all of it would clear `+0.050 / +0.050`? Expected to pass; a fail kills
   at zero cost.
2. **Identifiability (`D-FELDMAN`).** Is the population *distinguishable from
   geometry alone*, without seeing its labels at prediction time? A global,
   symmetric encoder-level operator can only act on a region it can locate. If
   stable inversions are geometrically indistinguishable from correctly-classified
   items in the same local configuration, no such operator exists, and this is
   exactly what Feldman-style long-tail memorisation predicts.
3. **Conversion (`NET`).** At the frozen operating points, does the population
   yield enough net correct-minus-broken items in the currency
   `banned_constraints[10]` mandates?

**A0 does not train an encoder, does not touch test, and does not establish that
any operator exists.** A CONTINUE means only that the target population is large
enough and locatable enough to be worth building an operator for.

---

## 2. Arena, and why it is free

**Path.** The banked **fold-head / deployed-head arena** only. F113 stands and is
carried verbatim into the amendment: *a raw-key arena may KILL but may not PROMOTE*
— so a Stage-0 PASS must be rendered on the fold-head path.

**Instrument (verified present, nothing to build).**
`scripts/analysis/headspace_mint.py`, `headspace_arena.py`, `headspace_fidelity.py`,
`headspace_report.py`, plus the six banked
`headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`.

**Configuration.** 2 datasets × 3 head seeds × 5 item-disjoint folds = 30
fold-heads per dataset-seed sweep. Bank = the fitting pool; queries = the held-out
fifth. Query labels are **train-split** labels held out from the head that judges
them — this is what "OOF" means throughout this document.

**Datasets.** `HateMM` (n = 744 train queries pooled over folds) and `MHC_zh`
(n = 579). **MHC-EN is declared OUT OF SCOPE for A0**: its fold-head arena has
never been minted, and minting it would introduce a new instrument requiring its
own fidelity check. Two datasets is what the bar requires.

**Cost.** `0 GPU-hours`. Per-fold head checkpoints are **not** persisted
(`headspace_mint.py:274-281` monkeypatches `torch.save` to a no-op because
per-epoch dumps are ~34 MB and nothing downstream reads them), so heads are
re-minted at 25–40 s CPU each — ≈ 36 heads ≈ 30 CPU-minutes, plus analysis.
Target shape: one CPU-only SLURM job, 8 CPU / 32 GB / no GPU, following C02's A0
(job `13847`: 8 CPU / 0 GPU / 32 G, `00:29:49` wall).

**F88's binding caveat is satisfied by construction.** F88 requires that *"a
CPU-trained arm must be paired against a CPU-TRAINED FLOOR, never against the
banked GPU floor."* Every arm and every floor in this A0 is minted inside the same
CPU fold-head arena, so no cross-hardware pairing occurs anywhere.

---

## 3. Label-use discipline — resolved LEGAL, on written texts

Identifying OOF-stable inversions requires reading train labels out of fold. This
is **legal**, and the reopen resolved it on two written texts rather than by
inference, so it is not flagged for user adjudication:

- `autoresearch/goal_mllm_plus3/state/progress.json:25` — the user's own
  oracle-ranked-queue ruling: *"Legal attack on selection-locked pools = trained
  selector/reshaper on train labels only (F66 binds only fixed-map phi0)."*
- `refine-logs/LITSWEEP3_DATA_CENTRIC.md:82` — an on-point in-repo adjudication of
  exactly this shape: *"those select **per test instance**; curation selects
  **train items once, globally, applied identically to every test query** — a
  symmetric operator, so law-III/F66's per-item ban does not apply to the
  mechanism."*

Every text on the other side (F47, F66, EUM precondition 2) bans **per-test-instance**
selection. Three boundaries flip C09 to illegal and are written in as **HALT**
conditions:

- **H-L1.** Any query-time consultation of the stability statistic or of the
  `D-FELDMAN` classifier. F47 fires directly, and its escape clause is closed: an
  OOF-stability statistic *is* "derivable from banked features/votes", so it is
  not the *"genuinely NEW information source"* the exception requires.
- **H-L2.** Any per-item exception that survives to inference as a per-item rule.
- **H-L3.** Any read of a dev or test label at any stage, by any code path.

**Stated plainly so it is not over-read: `D-FELDMAN` is a Stage-0 *identifiability
probe*, not a deployable component.** It measures whether a region exists. Whether
a *global operator acting uniformly on that region* is legal and buildable is a
Stage-1 question with its own gate; A0 makes no claim about it.

**The counter-text, carried because legality and viability are different
questions.** `LITSWEEP5_COMPLETENESS.md` §4(ii), headed *"The contradiction
(load-bearing)"*, was written **after** the oracle-queue ruling and observes that
its two blessed classes — *"Trained SELECTOR on train labels"* and *"Trained
symmetric RESHAPER on train labels"* — are *"both already measured dead"*, and
that the ruling *"was written at lit-round-count 3 — before F75/F77/L1 sharpened
the walls."* Nothing there withdraws the permission, so the legality verdict
stands. But C09 inherits a **weakened prior**, not a formality, and this A0 is
designed accordingly: `K-FELDMAN` and `K-NET` are both able to fire on their own.

---

## 4. Population definition — every threshold frozen before any run

### 4.1 Inversions

For dataset `D`, seed `s`, fold `f`, query item `i`: the deployed decision is the
rank-weighted top-20 vote with `w = [20…1]`, `Σw = 210`, decision `[score ≥ 0]`,
exactly as deployed. Item `i` is an **inversion at seed `s`** iff its vote
disagrees with its gold train label.

### 4.2 Stability

`i` is an **OOF-stable inversion** iff it is an inversion in **all three** head
seeds. Stability is computed per seed independently and the population is the
**intersection**. F88 measured this population directly and reports it is the
dominant one: *"ZH 22 of the 25-item union wrong 3/3 (88 %) with NOTHING at
exactly 2/3 and ALL 12 false negatives 3/3-stable"*; HateMM *"24-25 of 26-28
errors wrong in 3/3 seeds (89-93 %)"*.

### 4.3 Confidence — and why the primary threshold is zero

The registry claim says *"high-confidence"* inversions. Confidence is
`c_i = |score_i| / Σw ∈ [0, 1]`.

**PRIMARY: `τ_conf = 0.00`** — i.e. all OOF-stable inversions, with no confidence
restriction. This is deliberate and it is the conservative choice **against** the
candidate's survival being an artifact of threshold search: the reach oracle `O1`
is monotone non-decreasing in population size, so `τ_conf = 0` is the **largest
possible** population and hence the most generous reach test. **A KILL at
`τ_conf = 0` kills every higher threshold by monotonicity**, which is the
strongest possible zero-cost closure.

**DECLARED SENSITIVITY (never the decision): `τ_conf ∈ {0.10, 0.25}`.** Reported
in full for mechanism reading only. No decision rule reads these rows. If the
primary passes and a sensitivity row does not, that is reported as a mechanism
fact, not used to select a threshold.

### 4.4 Frozen ancillary definitions

- **Right analogue** of `i`: the highest-ranked bank item carrying `i`'s gold
  label, with its rank `r_i` in `i`'s current ordering. (Motivating measurement,
  not a threshold: `ERRPAT_MHC-ZH` reports the first same-gold-class train
  neighbour at median rank `1.5` for the 22 ZH core errors, 11 of 22 at rank 1;
  `ERRPAT_HateMM` reports median rank `3.0` with `6/27` errors having no
  true-label neighbour in the top-20 at all.)
- **Purity** `p_i`: fraction of `i`'s top-20 carrying `i`'s gold label.
- **Configuration stratum**: the frozen cross of purity bucket
  `{[0,0.10], (0.10,0.25], (0.25,0.45], (0.45,1]}` (the buckets `ERRPAT` already
  uses) × confidence tercile computed on the **bank** side × analogue-rank bucket
  `{1, 2-5, 6-20, >20}`. Buckets are frozen here and are not tuned.

---

## 5. The three measured quantities

### 5.1 `O1` — reach (necessary, expected to pass)

`Δacc_O1 = |P| / n`, where `P` is the primary population and `n` the pooled query
count, with the paired macro-F1 computed by flipping exactly `P`. Gold-cheating by
construction: it fixes everything and breaks nothing.

### 5.2 `D-FELDMAN` — identifiability, the actual test

**Question.** Can "is this item an OOF-stable inversion?" be predicted from
**geometry alone**, with no access to the item's own label at prediction time?

**Estimator.** A nested-OOF classifier. The arena's 5 folds already hold out the
query items; the identifiability classifier gets its **own** nested split so it
never scores an item it was fit on. Two frozen parameterisations, both reported,
with **logistic regression as the pre-declared primary** and gradient boosting as
a declared capacity check (this mirrors F47's own two-family protocol, which found
*"NO per-item routing signal, GBM or linear"*).

**Frozen feature set — geometry only, no label of the scored item:**
top-20 purity; vote margin `|score|/Σw`; rank of the first same-**predicted**-class
neighbour; rank of the first neighbour whose bank label differs from the majority;
mean and spread of the top-20 similarities; the similarity gap between ranks 1 and
20; local bank density (mean similarity to the 50 nearest bank items); the item's
own norm before L2 normalisation; and the bank-side degree of its top-20 members
(how often each appears in other items' top-20). Every feature is computable from
the banked head keys and bank labels; none reads the scored item's own label.

**Reported.** Held-out AUC with a paired bootstrap 95 % CI, per dataset, pooled
over seeds; plus precision and recall at each frozen operating point of §5.3.

### 5.3 `NET` — conversion, in the mandated currency

`banned_constraints[10]` is explicit: *"EXCHANGE RATE ≥ 1.2 IS NOT A SCREENING
CRITERION … The law is net = changed × (2·precision − 1); the binding screen is
NET ITEMS against 22.3 / 17.4 / 16.5 (HateMM / MHC-ZH / MHC-EN) for +0.030."*
Scaled to the Stage-0 `+0.050` bar: **`37.2` (HateMM) and `29.0` (MHC-ZH)**.

At each frozen operating point, an *idealised global operator* is credited with
flipping every item the classifier selects, to its gold label if it is a stable
inversion and away from its gold label if it is a configuration-matched correct
item. `net = fixes − breaks`. Exchange rate is reported as a diagnostic and reads
no decision rule.

**Frozen operating points.** Classifier score thresholds at the
`{50, 60, 70, 80, 90, 95}`th percentiles of the held-out score distribution — six
points, declared here, with Holm correction across the family of 6 × 2 datasets ×
2 metrics. No other point is evaluated.

---

## 6. The Feldman discriminator, stated as a decidable observation

This is the design's central obligation and the reason the recon named it C09's
sharpest risk. Its **numerical** leg is already retracted in-repo —
`HEADCOV_PREGATE_RECORD.md:305-310` withdraws *"the Feldman flourish"* because the
deployed heads sit at 0.82–0.94, not 0.998 — while its **substantive** leg is
preserved verbatim there and stands: *"memorising a long-tail singleton does not
transfer to an unseen member of the same one-member sub-population."*

**The two hypotheses make opposite, measurable predictions about `D-FELDMAN`.**

| | **H-TOPOLOGY** (C09's claim) | **H-MEMORISATION** (Feldman) |
|---|---|---|
| what the stable inversions are | a **region** of the representation with a shared geometric signature — a topological defect | **singletons**: each wrong for its own reason, sharing only that no analogue was memorised |
| prediction for held-out `D-FELDMAN` AUC | materially above chance, CI lower bound `> 0.5` | ≈ 0.5; configuration carries no signal |
| prediction for `NET` at the frozen points | precision high enough to clear `37.2 / 29.0` | precision ≈ the population base rate; net ≈ 0 or negative |
| what an operator could do | act uniformly on the region | nothing — the fix requires memorising items never seen |

**The discriminating observation, pre-declared:**

> **If `D-FELDMAN`'s held-out AUC 95 % CI lower bound is `≤ 0.5` on both datasets,
> H-MEMORISATION is confirmed on this object and C09 is KILLED at Stage-0.** The
> stable inversions would then be exactly what F98 already described — *"the local
> configuration carries no learnable signal"* — and no global encoder-level
> operator can target a region it cannot locate.
>
> **If the CI lower bound is `> 0.5` on both datasets AND `NET` clears
> `37.2 / 29.0`, the defect is a locatable topology defect rather than a
> memorisation-necessary error**, and C09 earns Stage-1 — where the separate and
> harder question is whether a *legal global operator* realising it exists.

**Two controls make this discriminator honest**, both frozen:

- **`SHUFFLE-POP`.** Permute the stable-inversion indicator within the query set,
  preserving all configuration marginals. `D-FELDMAN` AUC must collapse to chance.
  If it does not, the estimator is leaking and the run **HALTs** — no verdict is
  published.
- **`UNSTABLE-POP`.** Re-run `D-FELDMAN` with the target redefined as
  *unstable* errors (wrong in exactly 1 or 2 of 3 seeds). If stable and unstable
  populations are equally predictable, then **"stability" carries no information**
  and the registry claim's own premise is empty — reported as a mechanism finding
  regardless of the primary verdict. Note F88 measured that ZH has *nothing* at
  exactly 2/3, so this control may be under-powered on ZH; that is declared here,
  in advance, rather than discovered afterwards.

---

## 7. Controls and validity gates — all frozen, all HALT-only

### 7.1 Scientific controls

- **`RANDOM-POP`** — a size-matched random sample of query items in place of the
  stable inversions. Every reported quantity is recomputed against it. This is the
  control the closest prior attempt failed: **F88 null (3)** measured HateMM
  memory-bank LOO curation at `+0.0016` against a random-deletion control of the
  same size at `+0.0031 / +0.0000`, i.e. *"THE CURATED RULE DOES NOT BEAT DELETING
  THE SAME NUMBER OF ROWS AT RANDOM"* — self-labelled *"Pregate-grade null (one
  rule, one proxy head/cell, single draw)"*, HateMM-only, on **train-row deletion**,
  a different population and operator from C09's. It is a headwind to price, not a
  closure, and this control prices it directly.
- **`CONFIG-MATCHED-CORRECT`** — correctly-classified items matched on the frozen
  configuration stratum. Supplies the break-exposure denominator, so that
  "constraining break exposure" is measured rather than asserted.
- **`SHUFFLE-POP`**, **`UNSTABLE-POP`** — §6.

### 7.2 Validity gates (HALT-only; a failure publishes no verdict)

- **`GATE-FID`** — the minted fold-head floors must reproduce F113's banked
  per-seed values **bit-exactly on 6/6 seeds**: HateMM
  `0.8884 / 0.8858 / 0.8858`, MHC-ZH `0.8929 / 0.8895 / 0.8946`. This is free and
  already demonstrated: C02's A0 `gates.ARENA2.pooled_native_acc` reproduced
  exactly these values. Any drift means the instrument moved and nothing may be
  read from the run.
- **`GATE-LEDGER`** — a runtime access ledger asserting **zero** dev-split and
  **zero** test-split path opens and **zero** dev/test label materialisations,
  reported as literal integer counts, not booleans.
- **`GATE-SEED`** — stability is the 3-seed intersection; the per-seed inversion
  sets are emitted in full so the intersection is independently recomputable.
- **`GATE-NULL`** — HateMM train row 355 (`hate_video_95`, the structural
  zero-vector item created by a Decord+PyAV decode failure) is handled under the
  same frozen contract C01-v2 and C02-v9 used: it must remain exact-zero in every
  derived array, must never enter any top-20, and the with-null and remove-null
  routes must agree on every published metric. A structural-null sensitivity read
  is reported alongside the primary, as C02's A0 did.
- **`GATE-ARENA`** — pooled native accuracy must sit between the majority rate and
  saturation on both datasets, as C02's `ARENA-2` did (`0.8858-0.8884` against a
  `0.5995` majority; `0.8895-0.8946` against `0.6891`).
- **`GATE-NESTED`** — the identifiability classifier's nested-OOF split must be
  asserted disjoint from the arena fold that produced each scored item, with the
  assertion emitted as a per-item check count.

---

## 8. Decision rule — frozen, conjunctive, pre-declared

**KILL** if **any** of the following fires:

1. **`K-REACH`** — `O1` `Δacc < +0.050` **or** `Δmacro-F1 < +0.050` on either
   dataset. (By §4.3's monotonicity argument, a fire here closes every confidence
   threshold at once.)
2. **`K-FELDMAN`** — `D-FELDMAN` held-out AUC 95 % CI lower bound `≤ 0.5` on
   **both** datasets.
3. **`K-NET`** — best frozen operating point yields `net < 37.2` (HateMM) or
   `net < 29.0` (MHC-ZH), after Holm correction.

**CONTINUE** only if all three clear on **both** datasets, `SHUFFLE-POP` collapses
to chance, and all six validity gates pass.

**The raw arena is computed and reported but confined to corroborating a KILL**,
which is the only direction F113 permits — and even that direction carries F113's
own caveat, recorded in the reopen: *"NOT established: that a raw-space NEGATIVE
cannot be a head-space positive."* No raw-arena number reaches the decision.

---

## 9. Scope of any verdict this A0 can produce

- A **KILL** closes the C09 Stage-0 oracle **under the frozen Stage-0 rule**. It is
  not an impossibility proof for encoder-level topology intervention. The
  identifiability probe is one particular feature set and one estimator family; a
  richer geometry might locate the region where this one cannot. This boundary is
  stated **now**, in advance, because C02's A0 had to retract exactly this kind of
  overclaim once (the v8 erratum) before it was re-stated correctly.
- A **CONTINUE** establishes only that the population is large enough and locatable
  enough to justify building an operator. **It establishes nothing about whether a
  legal global operator exists**, which is Stage-1's question and carries its own
  live headwinds: F75 is *"the first measured negative for
  trained-reshaping-unlocks-oracle-headroom"*, and F66's β-decomposition found
  91–98 % of oracle headroom reachable only by law-III-banned per-item selection.
- Neither verdict touches the `+0.030 / +0.030` two-dataset target, which remains
  active and unmet.

---

## 10. Open items for the reviewer

Named here rather than left to be discovered:

1. **Is `D-FELDMAN`'s feature set genuinely label-blind for the scored item?**
   Bank-side degree and neighbour-label-derived features read *bank* labels. Bank
   labels are train labels, legal under §3 — but the reviewer should confirm no
   feature is a disguised read of the scored item's own label, and that the
   nested-OOF split makes the "first neighbour whose bank label differs from the
   majority" feature non-circular.
2. **Is `UNSTABLE-POP` viable on MHC-ZH at all?** F88 reports nothing at exactly
   2/3 on ZH, so the control may have an empty or near-empty population. Should it
   be declared HateMM-only in advance, or should the run publish an explicit
   `CONTROL_UNDERPOWERED` flag?
3. **Is the `+0.050 → 37.2 / 29.0` net-item scaling correct?** It is a
   proportional scaling of `banned_constraints[10]`'s `22.3 / 17.4` for `+0.030`.
   The reviewer should confirm the denominators are the same population the
   original figures were computed against.
4. **Six operating points may be too many.** Holm across `6 × 2 × 2 = 24`
   hypotheses is severe. Is a smaller pre-declared set (three points) a better
   trade between power and forking-path control?
5. **Does the confidence-threshold monotonicity argument (§4.3) actually hold for
   `NET`, or only for `O1`?** It is stated for `O1`, where it is arithmetic. The
   reviewer should check whether the primary should also be `τ_conf = 0` for the
   net-item screen, or whether that needs its own justification.
6. **`GATE-FID` bit-exactness** — C02 achieved it, but that was with C02's own
   minting path. The reviewer should confirm nothing in this design perturbs the
   mint (e.g. thread counts, BLAS environment) in a way that would break bit
   equality and turn a scientific run into an engineering HALT, which is how three
   C01 runs died.

---

*Draft only. No hash frozen, no config written, no code implemented, no namespace
created, no job submitted, no cache or test path opened, no metric or result
produced. Zero GPU, zero SLURM, zero Modal, zero teacher call.*
