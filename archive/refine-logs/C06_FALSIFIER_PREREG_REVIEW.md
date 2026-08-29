# C06 `$0` falsifier — independent design review, **ROUND 1**

**Target:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT.md` (DRAFT v1, 2026-08-04)
**Reviewer posture:** fresh, no exposure to the designer's reasoning. Read-only. No GPU,
SLURM, Modal, model load, training, cache write, test-split access, job submission or
commit occurred. `TARGET_STATE.json` was read and not modified. Nothing heavier than
`sha256sum`, file reads, and small arithmetic / linear-algebra re-derivations on the
already-banked train-split caches was executed.

---

# VERDICT

## **REVISE (3C / 6H / 10I)** — plus 4 Minor

Not `GO (0C/0H/0I)`.

The design is unusually strong on ceremony: every digest reproduces, the arithmetic
reproduces, the frozen-constant provenance is real, the HALT semantics are sound, and the
dry check genuinely exercised the payload path. The three Criticals are not about
ceremony. Two of them are about the **instrument**, and I found them by measuring, not by
reading: the head is forwarded over features **near-orthogonal** to the ones it was
trained on (median cosine `0.0234` on HateMM images), and the one gate that is supposed to
catch the resulting degeneracy is watching a quantity — displacement **magnitude** — that
the failure does not have to move.

---

# PART A — INDEPENDENT VERIFICATION OF ALL TWELVE §2 ITEMS

| # | result | what I obtained |
|---|---|---|
| **V1** | **VERIFIED** | All 18 digests recomputed with `sha256sum` and matched character-for-character against §11: 6 imported modules, 4 read-for-definition files, 8 input caches. No mismatch. |
| **V2** | **VERIFIED** | `6a44cce4f65d4a60` / `60054f3be1204ca7` (HateMM) and `1d33fe5d69083479` / `3ad1309dc7500182` (MHC_zh) equal `configs/c01/c01_a0_v2.json::inputs.datasets.<ds>.expected.train.{standard,oneword}_provenance_sha16`. HateMM `ro_L24` full 64-hex `6a44cce4f65d4a60b8863969a2ad9ed72731658cea49e75f487a911000be045f` equals `c01_a0_v3.json::lineage_evidence.diagnostic_train_cache_sha256` **in full**. |
| **V3** | **VERIFIED** | Directory listing: HateMM carries `-LoRA-curric_HF-ro_*` only; MHC_zh carries `-LoRA_HF-ro_*` only. One lineage each. |
| **V4** | **VERIFIED** | All 8 ro caches: `ids` order-identical and `labels` element-identical to the native bank; `n = 744 / 579`; `img_feats` and `text_feats` both `(n, 3584)`. |
| **V5** | **VERIFIED** | HateMM row 355 exactly zero in **both** modalities of **all four** ro caches **and** the native cache; `ids[355] == "hate_video_95"`, matching `c01_a0_v3.json::lineage_evidence.authorized_null` (`row_index 355`, `raw_id hate_video_95`, both policies, both modalities). MHC-ZH: no exact-zero row in any cache. |
| **V6** | **VERIFIED** | `result.acc_deployed` re-read from all six files: HateMM `0.8884 / 0.8858 / 0.8858`; MHC-ZH `0.8929 / 0.8895 / 0.8946`. |
| **V7** | **VERIFIED** | `mechfix_ops.deployed_vote` (`:74-95`): `topk = TOPK = 20`; `w = np.arange(1, k+1)[::-1]` = `[20…1]` descending integer; signed cosine `(lab*2-1) * sim`; `(votes >= 0)`. Against `c01_a0_v2.json::retrieval`: `topk 20`, `rank_weights descending_integer`, `similarity signed_cosine`, `prediction_cutoff 0.0`. Same operator. |
| **V8** | **VERIFIED** | `angles_degrees = [8.3, 17.6, 29.1, 60.4, 72.7, 83.8]`; the config's own text: *"45 degrees excluded because it is the primary common/displacement transform"*. `orthogonal_blocks` is at `c01_policy_contrast_a0.py:1272` and is a Givens mixing (`cos·std + sin·ow`, `−sin·std + cos·ow`); at `θ = 0` it returns `(standard, oneword)`, whose `paired_key` is `endpoint_concat` by construction (`:1318-1322` vs `:1363-1366`). |
| **V9** | **VERIFIED** | `paired_key` (`:1220-1239`) concatenates two `3584`-d blocks per modality → `7168`-d per modality → `14336`-d fused. `classifier_hateClipper.__init__` (`src/model/classifier.py:80-81`): `img_proj = nn.Sequential(nn.Linear(image_dim, map_dim), …)` with `image_dim = 3584`, `map_dim = 1024`. The paired arms **cannot** be fed to the deployed head. The claim is true, not rhetorical. |
| **V10** | **MISMATCH (partial)** — see **C-2** | The precedent exists but is **narrower than the draft states**. `c02_a0_mint.py:214` is `keys[view] = keys_of(tr[1], view_text[view])` — the **native** `img_feats` on every view; only `text_feats` moves. The module says so at `:21-23` (*"img_feats are byte-identical across views by construction … so the view axis is the only thing that moves"*) and `load_view_text` asserts `"img_feats" not in d` (`:68`). C06 replaces **both** streams and moves to a **different readout cell**. |
| **V11** | **VERIFIED** | Every product and both totals re-multiplied independently: `15×40.39=605.9`, `3×49.30=147.9`, `15×34.40=516.0`, `3×38.87=116.6` → `1386.4`; `210×0.0461=9.68`; `240×0.0042=1.008` + `540×0.0218=11.772` + `30×0.0042=0.126` = `12.906`; `3072×0.1241=381.24`; `264×0.126=33.264`; `(12.9+381.2+33.3)/2=213.7`; `+30+60` → **`2127.2 s = 35.5 min`**; `×1.25` → **`2659.0 s = 44.3 min`**. Also confirmed: `195×40.39 + 195×34.40 = 14 584.05 s = 4.05 h`; Phase 3 share `17.9 %`; sensitivity `2508.4 s` and `3652.0 s`. |
| **V12** | **VERIFIED** | `generate_VideoMLLM_embedding_readout_HF.py` `CELLS`: `("ro_L24", "baseline", "prefix", "response", LAYER_MID)` versus `("ro_ow_L24", "oneword", "last_token", "last_token", LAYER_MID)`. The `ow_` cell changes the **prompt kind and both readout spans**. §10.1's quotation is exact. |

## Additional measurements I made, which the draft did not

Computed on the already-banked train-split caches, zero row masked, C01's own block
algebra (`l2` per modality → per-modality `l2`-normalised → concat → row `l2`):

**(a) §7.4's raw-space figures reproduce exactly.** Fused raw key: median
`‖e_ow − e_std‖` = **`0.7016`** (HateMM) / `0.6977` (MHC-ZH); median `cos` = `0.753874` /
`0.756625`. The draft's `0.7016` and `≈ 0.754` are correct.

**(b) Tiny-displacement counts, split by mask.** HateMM: `1 / 744` including row 355,
**`0 / 743` excluding it**. MHC-ZH: `0 / 579`. `GATE-ORBITSCALE` is defined over *unmasked*
items, so the dry-check value the gate would actually see is `0.0000`, not the `1/744 =
0.0013` §7.4 reports. → **I-4**.

**(c) Displacement-direction dispersion in the raw space** — `ρ = ‖mean_i k_i‖` over unit
fused keys, where `ρ → 1` means every item's key points the same way:

| dataset | `ρ`(`displacement`) | `ρ`(`endpoint_std`) | `ρ`(`common`) |
|---|---|---|---|
| HateMM L24 | **0.8917** | 0.9682 | 0.9644 |
| MHC-ZH L24 | **0.9091** | 0.9772 | 0.9697 |

In the raw space the displacement direction is **more** item-dispersed than `endpoint_std`
— it carries real per-item structure. → load-bearing for **C-1**.

**(d) Native cache versus `ro_L24`, row-wise:**

| dataset | median `cos(native_img, ro_L24_img)` | median `cos(native_txt, ro_L24_txt)` |
|---|---|---|
| HateMM | **0.0234** | 0.2300 |
| MHC-ZH | **0.0373** | 0.2495 |

Both caches store unit-norm rows (norm ratio `1.0000`), so this is pure direction: the
image streams are essentially **orthogonal**. → load-bearing for **C-2**.

**(e) L28 is not a sibling of L24.** Per modality, median `‖l2(ow) − l2(std)‖` and median
`cos(std, ow)`:

| dataset | layer | `‖Δ‖` img / txt | `cos` img / txt |
|---|---|---|---|
| HateMM | L24 | 0.7257 / 0.6744 | 0.7367 / 0.7726 |
| HateMM | **L28** | **1.2913 / 1.0993** | **0.1663 / 0.3958** |
| MHC-ZH | L24 | 0.7046 / 0.6899 | 0.7517 / 0.7620 |
| MHC-ZH | **L28** | **1.3059 / 1.1026** | **0.1473 / 0.3922** |

At L28 the two endpoints are near-orthogonal, with `‖Δ‖` approaching the `√2` ceiling. →
**I-8**.

---

# PART B — FINDINGS

## CRITICAL

### C-1. `GATE-ORBITSCALE` guards displacement **magnitude**; the artifact it is declared to catch moves displacement **direction**. A degenerate instrument passes it and publishes a clean-looking CLOSE.
*Attaches to:* §6 `GATE-ORBITSCALE`; §7.4 Finding 2; §14 open issue 2.

Every arm is `l2`-normalised before it reaches `deployed_vote` — §3.3 defines
`displacement = l2(l2(e_ow) − l2(e_std))`, and `mechfix_ops._norm32` re-normalises again
at the index. **The retrieval key therefore discards `‖Δ‖` entirely and keeps only its
direction.** The decision-relevant property of the head-space orbit is the *per-item
dispersion* of that direction, and `GATE-ORBITSCALE` does not measure it.

The concrete failure: suppose the trained head's two policy outputs differ by an
approximately constant offset, `e_ow − e_std ≈ c·u + small per-item residual`. This is
exactly what "the head has learned to be invariant to the readout-span/prompt axis" looks
like, and it is exactly the mechanism §5.7 already names as its structural objection
(*"a fixed prompt injects no per-item information"*). Then every item's `displacement` key
is ≈ `u`, its top-20 neighbourhood is arbitrary, and every `orthrot_θ` arm — which retains
`e_std`'s per-item variation in its first block — beats it trivially. Meanwhile `‖Δ‖ ≈ c`
can sit anywhere above `1e-3`, so `GATE-ORBITSCALE` passes, no other gate in §6 looks at
this, and §5.3 publishes **CLOSE**. That is the false closure the draft names as its top
threat, reached through a door the gate does not cover.

This is not speculative. Measurement (c) above shows the raw-space displacement is
genuinely item-dispersed (`ρ = 0.8917 / 0.9091`, *lower* than `endpoint_std`'s
`0.9682 / 0.9772`), which is why C01 could score it at `0.8505 / 0.8846` at all. A
head-space `ρ` approaching `1.0` would therefore be **the head's doing**, and the design
would attribute it to C06.

Two further points on the threshold, which is what §14 issue 2 asks about:

* The gate is not the numerical-floor guard §7.4 describes. Keys reach `deployed_vote` in
  `float32` (`mechfix_ops._norm32:40-42`) and the head computes in `float32`, so
  cancellation error in `l2(e_ow) − l2(e_std)` is `~1e-7` relative to unit endpoints and
  the direction is noise-dominated only below `‖Δ‖ ≈ 1e-6`. The gate sits **three orders of
  magnitude above** the floor it is said to protect.
* Consequently, changing `1e-3` to any other value does not fix this. The *quantity* is
  wrong, not the constant.

**Repair (`$0`, from data the battery already loads).** Add a pre-registered HALT gate on
displacement **informativeness**, computed identically in the raw key space and the head
space on the same rows — e.g. `ρ_space = ‖mean_i k_i‖` over unit `displacement` keys.
Pre-register the bar **now**, off the raw-space values in measurement (c): those are
banked, label-free and frozen, so fixing the bar before the run costs nothing and reveals
no arm ordering. The raw-vs-head comparison is what separates the two cases §6 currently
conflates:

* both spaces degenerate ⇒ C06's premise is false ⇒ **CLOSE is warranted**;
* raw non-degenerate, head degenerate ⇒ the instrument destroyed the object ⇒ **HALT**.

Add the free prediction-level companion as well: extend `GATE-VIABILITY` — currently
applied to `endpoint_std` alone, an arm that stays perfectly healthy in exactly this
failure mode — to require that both arms in `R` clear the majority-class rate.

### C-2. The head is trained on the native cache and forwarded over features **near-orthogonal** to it. This is not the C02 precedent, and it is the most likely cause of the 219× contraction.
*Attaches to:* §3.3 steps 1-2; §4 A4; §7.4 Finding 2; V10.

`headspace_mint.py:196-199` mints on `train_{model}.pt` from `mechnov_pairverify.DATASETS`
— the **native** pooled cache. §3.3 step 2 then forwards that head over `ro_L24` /
`ro_ow_L24`. Measurement (d): median `cos(native_img, ro_L24_img)` is **`0.0234`** on
HateMM and **`0.0373`** on MHC-ZH; text is `0.2300 / 0.2495`. Both caches are unit-norm, so
this is not a scale issue — the image streams are essentially orthogonal representations.

`classifier_hateClipper.forward` (`src/model/classifier.py:115-124`) under
`fusion_mode = align` is
`mlp( normalize(img_proj(f_img)) ⊙ normalize(text_proj(f_text)) )`, with
`img_proj`/`text_proj` = `Linear(3584 → 1024)` fitted on the native distribution, a
Hadamard fusion, and a ReLU MLP. Forwarding near-orthogonal inputs through that stack is an
out-of-distribution transplant, and a collapsed output orbit is its expected consequence —
which is precisely the `0.7016 → 0.0032`, `cos = 0.999995` contraction §7.4 measured and
was unable to explain.

**The C02 precedent does not cover this, and §3.3's claim that it is "the banked house
pattern, not an invention" is not supported by the artifact it cites.** `c02_a0_mint.py`
holds `img_feats` **byte-identical to the native bank** on every view (`:214`,
`keys_of(tr[1], view_text[view])`), says so in its own docstring (`:21-23`), and refuses
any view file that carries an image stream (`:68`). C02 moved **one** stream **inside one
extraction family**. C06 moves **both** streams to a **different readout cell**.

Under this transplant a "rotations match displacement" result is not attributable to C06,
and §10.2 would scope the closure to something it does not currently say.

**Repair**, in decreasing order of preference:

* **(a)** Add a pre-registered HALT gate on head-space input-domain fidelity, anchored on a
  banked number: head-space `endpoint_std` OOF accuracy against the `GATE-FLOOR` native
  value. If the native forward reproduces `0.8884 / 0.8929` while the `ro_L24` forward
  collapses toward the majority rate, the transplant is the explanation and no arm
  comparison is interpretable. This is free — `endpoint_std` is already voted in Phase 2.
* **(b)** Adopt A4's excluded second reading (head trained on `ro_L24`) on at least one
  dataset as a cross-check. This removes the OOD transplant but forfeits `GATE-FLOOR`, so
  it complements (a) rather than replacing it.
* **(c)** At minimum, state on the face of the verdict (§10.2) that a CLOSE is scoped to a
  **native-trained head applied out of domain**.

### C-3. The head-space arm algebra is a **re-implementation with no fidelity anchor**; `GATE-ALGEBRA` tests only its self-consistency.
*Attaches to:* §3.3 step 3; §6 `GATE-ALGEBRA`; §11.

§11 lists `c01_policy_contrast_a0.py` under *"Read for definitions and thresholds, **not
imported as code**"*, and §3.3 defines all thirteen arms afresh with
`pair(a,b) = l2(concat(l2(a), l2(b)))`, described as *"C01's `paired_key` with the modality
loop collapsed"*. That collapse is a judgment call, not a derivation: C01's `paired_key`
(`:1220-1239`) performs four block `l2`s, two within-modality pair concatenations each
`l2`-normalised, and then `fuse_modalities`, which `l2`s **each modality block again**
before the final concat and row `l2` (`:1208-1217`). Which of those normalisations survives
a collapse to a single fused head key is chosen, not implied.

`GATE-ALGEBRA` cannot detect a wrong choice. It asserts `orthrot_0 ≡ endpoint_concat` and
`orthrot_45 ≡ common_displacement` — and **both sides of each identity are built by the
same re-implemented `pair`/`l2`**. Those identities hold for any internally consistent
definition, including a systematically wrong one. A mis-normalised `common_displacement`
would pass every gate in §6, lose to the rotations, and publish a CLOSE. `GATE-FLOOR` does
not reach this code path either: it exercises the *native* forward, never the arm builder.

**Repair (`$0`, no new files).** Import `c01_policy_contrast_a0` and assert that the
re-implemented builder reproduces `prepare_views`' output **bit-exactly on the raw L24
features**, before the head — the one object on which both definitions are defined. Any
residual above C01's own `2e-6` bar HALTs. This puts a fidelity anchor on the code path
that actually renders the verdict.

---

## HIGH

### H-1. §5.4's Holm family drops `displacement` as a comparator for `common_displacement`, contradicting C01's own frozen comparator set and materially easing SURVIVE.
*Attaches to:* §5.1 (`C`), §5.2 S3/S4, §5.4.

`configs/c01/c01_a0_v2.json::statistics.bootstrap_comparisons.primary_vs_controls` =
`["endpoint_std", "endpoint_ow", "avg_score", "endpoint_concat", "common", "displacement"]`
— **six**, including `displacement`. §5.1 defines `C` from `decision.gain_controls`
(**five**, no `displacement`), so §5.4's family is `5 + 6 = 11` where C01's is `6 + 6 = 12`.
Under the draft's rule, `common_displacement` can SURVIVE **without ever being shown to
beat `displacement`** — a comparison C01 required of its primary.
**Repair:** for `A = common_displacement`, define `C` as the union of `gain_controls` and
`bootstrap_comparisons.primary_vs_controls`, and restate the family size in §5.4 and the
count in §8 Phase 4.

### H-2. The two-real-arm disjunction is asserted to be absorbed; it is not, and no correction is applied.
*Attaches to:* §5.4, §4 A2.

§5.4: *"its multiplicity is absorbed by requiring the entire S1–S5 conjunction of whichever
arm is claimed."* A conjunction controls error *within* an arm; it says nothing about
testing two arms and reporting SURVIVE if **either** passes. The family-wise rate for the
SURVIVE event is up to `≈ 2α`. §4's "conservative = hardest to close" rationale explains why
the disjunction exists, but it does not license leaving it uncorrected — that is arithmetic,
not policy. **Repair:** fold both real arms into one Holm family per dataset
(`11 or 12 comparators × 2 metrics × 2 arms`), or split `α = 0.025` per arm. State which,
before freeze.

### H-3. S4 does not say what the bootstrap is computed over, and the two readings imply different families and different counts.
*Attaches to:* §5.1, §5.2 S4, §8 Phase 4.

§5.1 defines `acc(A,D)` as the **seed-mean**. §5.2 S4 asks for *"paired item-level bootstrap
lower bound on the difference"* on that quantity. §8 Phase 4 counts
`22 × 3 seeds × 2 datasets × 2 layers`, i.e. **per-seed**. So either the decision quantity
is the seed-mean (and §8's count is `3×` too large, family = 22) or it is per-seed (and the
family is 66 per dataset with an **unwritten** rule for combining three per-seed
rejections). This is an un-preregistered element that touches the decision.
**Repair:** state the bootstrap unit — I recommend resampling items once and averaging the
three seeds' per-item correctness inside each resample, giving one family of 22 per
`(real arm, dataset)` exactly as §5.4 claims — and re-derive §8 Phase 4 from it.

### H-4. §12's dev-label claim is false, and `GATE-LEDGER` is weaker than C09's as a result.
*Attaches to:* §12 bullet 2; §6 `GATE-LEDGER`.

§12 states *"`dev_seen` labels are materialised only inside `headspace_fidelity.py`, which
is reporting-only and gates nothing."* That is not what the code does.
`headspace_mint.py:199` loads `dev_seen_*.pt` **unconditionally on every mint**; `:322`
writes `lab_dev` into **every** mint `.npz`; and for `fold == -1` (`:229`) the **real dev
split is the training dev set**, so dev labels enter `run_rac.main` and the `eval_curve` on
6 of the 36 mints.

C09 handled this correctly by *declaring the counts*: `GATE-LEDGER`
(`C09_A0_V17_RECORD.md:1549-1554`) books test-split path opens `0`, test-label
materialisations `0`, dev-split path opens **expected nonzero with its declared value**,
dev-label materialisations outside any decision quantity **expected 36, one per mint**, and
the binding predicate — dev **or** test label materialisations *into any decision quantity*
— at `0`. C06's `GATE-LEDGER` (§6) asserts only `test_path_opens == 0`.
**Repair:** correct §12's sentence to match the code, and give `GATE-LEDGER` C09's full
predicate set with declared expected values, including the binding zero-into-decision leg.

### H-5. `GATE-FLOOR` anchors accuracy only, while macro-F1 is a decision metric in S1, S3 and S4.
*Attaches to:* §6 `GATE-FLOOR`.

§6 lists the six accuracy floors and *"every `fold_acc_deployed` entry"*. C09's
`GATE-FLOOR` (`C09_A0_V17_RECORD.md:1511-1522`) gates **both** metrics and gives the
macro-F1 anchors: HateMM `0.8838 / 0.8811 / 0.8812`, MHC-ZH `0.8747 / 0.8710 / 0.8765`.
A macro-F1 defect would leave C06's floor untouched and flow directly into S1, S3 and S4,
all of which carry a macro-F1 leg.
**Repair:** extend `GATE-FLOOR` to macro-F1 at 4 dp on all six `(dataset, seed)` cells
using the banked values.

### H-6. §7.3's `U4` is **smaller than the sum of the units it is defined to contain**, so Phase 3 is not established as a real-scale measurement.
*Attaches to:* §7.3 `U4`; §8 Phase 3; `rule_1_compute_projection`.

§7.3 defines `U4` as *"one shuffled-pair null draw (rebuild 2 arms in head space + 2 arms ×
5 folds of votes)"* = `0.1241 s`. The two real arms are `displacement` (1024-d, `U2a`) and
`common_displacement` (2048-d, `U2b`), so the vote content alone is
`5 × 0.0042 + 5 × 0.0218 = 0.130 s` — **already above the whole unit**, leaving negative
time for the arm rebuild. Either `U4` was measured over fewer folds or arms than §7.3
states — which is `rule_1_compute_projection`'s falsified pattern verbatim (*"an estimate
extrapolated from a reduced-scale dry run is NOT a projection and may not be recorded as
one"*) — or one of `U2a` / `U2b` / `U4` is mis-transcribed. Phase 3 is `381.2 s`, **17.9 %**
of the corroborating total and the largest non-mint phase; §8 itself names it the
projection's weakest element.
**Repair:** re-measure `U4` at the stated scale and reconcile it against `U2a`/`U2b`, or
declare Phase 3 **UNKNOWN** as the rule directs.

---

## IMPORTANT

### I-1. §8 Phase 1b counts dev forwards over ro-caches that §3.1 and §11 say are never opened.
*Attaches to:* §8 Phase 1b; §3.1 final paragraph; §11; §6 `GATE-SHA`.

Phase 1b: *"the 6 full-train mints add 5 dev forwards"* ⇒ `(30 × 5) + (6 × 10) = 210`. Five
dev forwards per mint means native + `{std, ow} × {L24, L28}` on `dev_seen` — i.e. **four
`dev_seen_*-ro_*` files** that §3.1 says *"are opened by no phase of this battery"*, that
§11 does not list, and that `GATE-SHA` therefore cannot cover. The enumeration and the
input manifest contradict each other.
**Repair:** if only the native dev forward occurs (all `GATE-DEVFID` needs), the count is
`(30 × 5) + (6 × 6) = 186`; if the dev ro-caches are opened, add all four to §11 and
`GATE-SHA` and correct §3.1.

### I-2. Phase 6 contradicts `GATE-DEVFID`'s own scope, and Phases 6-7 are budgets rather than measurements.
*Attaches to:* §8 Phases 6-7; §6 `GATE-DEVFID`.

§8 Phase 6 prices `headspace_fidelity.py` *"× 2 datasets"* while §6 `GATE-DEVFID` says *"on
the **6** deployed-configuration heads"*. Separately, *"C09 measured this at 'seconds';
budgeted 30 s"* and *"Budgeted 60 s as slack"* are precisely what
`rule_1_compute_projection` forbids inside a projection. Nothing turns on the number —
`90 s` of `2127.2 s` — but the rule is categorical, and this design is the one the rule
`applies_immediately_to`.
**Repair:** measure both on the real path (both are seconds-scale), or move them out of the
projection and label them declared slack.

### I-3. §8 omits three per-process loops.
*Attaches to:* §8; `rule_1_compute_projection`.

The battery follows the C02/C09 shape — one process per mint. Each of the 36 processes
therefore independently (a) `torch.load`s the four ro caches it forwards, (b) re-runs
`GATE-SHA` over 8 caches (`≈ 152 MB`) and 11 modules, and (c) pays wrapper import cost
above `headspace_mint.py`'s measured `40.39 s`. None appears in §8. Each is seconds-scale
and the direction is **under**-count — the F118 failure mode in miniature.
**Repair:** enumerate them, or state that `GATE-SHA` runs once in the sbatch driver rather
than per process (cheaper and equally sound), and that the ro caches are loaded once per
process by construction.

### I-4. §7.4's reported `GATE-ORBITSCALE` dry-check value is computed on a population the gate excludes.
*Attaches to:* §7.4 Finding 2; §6 `GATE-ORBITSCALE`.

§6 defines the gate over *"unmasked items"*. §7.4 reports `1 / 744` and *"the gate would
pass at `1/744 = 0.0013`"*. Measurement (b): the raw-space count is `1/744` **including**
row 355 and **`0/743` excluding it**; row 355 is the masked structural null. Under the
gate's own definition the dry-check value is `0.0000`.
**Repair:** restate §7.4's table with masked and unmasked columns and correct the sentence.

### I-5. Five C01 decision conditions are dropped from S1-S5 without disclosure.
*Attaches to:* §5.2; §12 bullet 3.

`c01_a0_v2.json::decision` also carries `require_accuracy_gain_over_deployed_r0_context`,
`minimum_net_fixes` (`{MHC_zh: 2, HateMM: 3}`), `require_no_small_displacement_dominance`,
`require_shuffle_holm_reject`, and `require_primary_and_displacement_above_shuffle_p95`
(**both** arms, where S5 tests only the claimed one). Every omission eases SURVIVE, so all
are consistent with §4's declared direction — but §12 states *"Every threshold in §5 is
either C01's frozen value or fixed in §4 before measurement"*, which reads as completeness.

The net-fix omission is the one that matters. The Gate-0 record's own closing strategic
finding is that Gate 0 *"must now screen on demonstrated conversion in the currency
`banned_constraints[10]` already names — NET ITEMS"*. A SURVIVE that never counts net
corrected-minus-broken items is out of step with the campaign's current currency.
**Repair:** list the dropped conditions explicitly and justify each; in particular say why
net items is not required of a SURVIVE.

### I-6. Three of C09's nine HALT gates have no C06 analogue where one is available for free.
*Attaches to:* §6; `C09_A0_V17_RECORD.md:1509-1572`.

Comparing against C09's nine (`GATE-FLOOR`, `GATE-PARITY-FOLD`, `GATE-FIXK20`,
`GATE-BLIND`, `GATE-LEDGER`, `GATE-NESTED`, `GATE-SELFTEST`, `GATE-ZEROOP`, `GATE-ARENA`):
`GATE-PARITY-FOLD` is folded into C06's `GATE-FLOOR`; `GATE-FIXK20` and `GATE-BLIND` have
no C06 object (`k = 20` always, no learned features); `GATE-SELFTEST` has no analogue
because C06 drops the net-item identity (I-5). Three gaps matter:

* **`GATE-NESTED`** (`:1557-1563`) asserts *per item* that the scoring model excluded that
  item's fold, emitted as a per-item check count equal to the item count. §3.2 states the
  OOF contract in prose; nothing checks it. A bank/query index slip would leak each query
  into its own bank and inflate **every** arm.
* **`GATE-ZEROOP`** (`:1566-1568`) checks the **treatment** path returns exactly nothing
  when the treatment is nil — explicitly *"Checks the treatment path, which `GATE-FLOOR`
  does not."* C06's analogue is free: `orthrot_0` and `endpoint_concat` (and `orthrot_45`
  and `common_displacement`) must produce **identical predictions**, not merely keys
  agreeing at `2e-6`. A `2e-6` key difference can reorder a top-20 neighbourhood.
* **`GATE-ARENA`** (`:1569-1572`) is a two-sided band `majority + 0.02 ≤ acc ≤ 0.98`.
  C06's `GATE-VIABILITY` is one-sided; the **upper** bound is what catches a leak.

**Repair:** add all three.

### I-7. C01's `shuffle_fixed_point_bijection` guard has no §6 counterpart.
*Attaches to:* §5.2 S5; §6; §8 Phase 3.

`c01_a0_v2.json::output.decision_schema.required_halt_only_validity_guards` includes
`shuffle_fixed_point_bijection`, and `statistics.permutation_pairing` requires *"the
registered structural-null train index is fixed and excluded from the remaining-source
bijection"* (echoed by `zero_contract_v2.require_fixed_null_in_shuffle`). S5 and Phase 3
inherit the permutation but §6 has no gate asserting row 355 remains a fixed point.
**Repair:** add it as a HALT gate on HateMM.

### I-8. L28 is not "a second, independent two-point orbit"; §5.6's disagreement clause presupposes a comparability that does not hold.
*Attaches to:* §5.6; §4 A5; §8 (`× 2 layers` throughout).

Measurement (e): at L28 the two endpoints are near-orthogonal (`cos = 0.147–0.396`,
`‖Δ‖ = 1.10–1.31` against a `√2` ceiling), versus `cos = 0.737–0.773`, `‖Δ‖ = 0.674–0.726`
at L24. `generate_VideoMLLM_embedding_readout_HF.py` confirms L28 is `LAYER_FINAL` and the
**R0 clobber-guard** cell, not a sibling of the L24 grid. "Orbit" is the wrong word for the
L28 object, and an L24/L28 disagreement would carry no information about replication.
**Repair:** drop the leg (see my ruling on §14 issue 4), or re-describe it as a
**different-layer probe** and delete §5.6's disagreement clause.

### I-9. The frozen artifacts of the very run being re-tested are absent from §11.
*Attaches to:* §3.1; §11.

The executed C01 A0 is **v4**: `TARGET_STATE.json::c01_a0_v4_typed_audit_repair` pins
`configs/c01/c01_a0_v4.json` (`2d9488e6f9af6be00d500d1c2f13912fd4be0ab9439608d33b0857178efe7ca6`)
and `scripts/analysis/c01_policy_contrast_a0_v4.py`
(`3c545eed876f97aa05f3e85375430bedf8e63226c70f3ee8ea12da02e9bf5514`), the artifact namespace
is `artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/`, and the recon this gate came from
says *"Re-run the **C01 v4** arm battery"* (`C05PLUS_FORENSIC_RECON_2026-07-31.md` §3.4).
§3.1 calls v2 *"C01's frozen configuration"* and §11 hashes v2 and v3 only.

**The draft's thresholds are nevertheless correct, and I traced the chain to confirm it:**
v4 `frozen_v3` → v3 `scientific_base` → v2, with `scientific_thresholds_exact: true` at
each hop and `reuse_policy: sha256_exact_import_then_v4_audit_schema_override_only`; and
`c01_policy_contrast_a0_v4.py:52-62` loads `_v3.py` under a sha256 check, which at `:28,52`
loads `base = scripts/analysis/c01_policy_contrast_a0.py` — the file §11 **does** pin
(`d2b9c2ff…`), and where `orthogonal_blocks` really is at line 1272. So this is a
provenance-completeness defect, not a correctness one. But a reader cannot verify the chain
from the document.
**Repair:** add v4's two digests to §11 and one sentence recording the
v4 → v3 → v2 (config) and v4 → v3 → base (algebra) chain.

### I-10. §5.7 uses the untrained-head contraction as evidence for the expected verdict, which it cannot be.
*Attaches to:* §5.7.

*"§7.4's measured 219× orbit contraction points the same way."* §7.4 itself says the
measurement is at an untrained head and that *"it is not known whether a trained head
contracts more or less"*. At random initialisation, `Linear(3584→1024)` followed by
normalise → Hadamard → ReLU concentrates essentially any two inputs; the contraction is
substantially a property of the initialisation, not of the prompt orbit. §5.7 is
non-decisional, but a pre-declared expectation partly resting on an artifact should not be
scored as a successful prediction later.
**Repair:** delete that clause. §5.7's other two grounds — C01's measurement and the
structural objection — carry it on their own.

---

## MINOR

* **M-1.** §3.1's table header `n_train` does not match the config path
  `inputs.datasets.<ds>.expected.train.n`. Cosmetic.
* **M-2.** §10.4 does not note that `endpoint_std` and `endpoint_ow` **are** the two cells
  F70 priced, entering here as *controls* rather than claims. The object-mismatch warrant
  is unaffected, but this is where a reader will look for the point.
* **M-3.** §5.2 S5 cites `bootstrap_upper_quantile = 0.95` as the source of the shuffle
  p95. That constant governs the bootstrap; the shuffle p95 is named directly in
  `decision.require_primary_and_displacement_above_shuffle_p95`. Cite the latter.
* **M-4.** §6 `GATE-IDPARITY` cites `c02_a0_mint.py:70-77`; the `ids` unwrapping is at
  `:69-71` and the parity assertions at `:72-76`.

---

# PART C — REQUIRED RULINGS

## §3.A — Is the draft's reading of *"the fold-head arena"* the one the written condition requires?

**Yes on the arena; no on the training set — and the second half is C-2.**

*"The FOLD-HEAD ARENA"* is defined by the registry as *"strict train-OOF or untouched
development split using the actual fold-head/deployed-head path"* (`unified_pilot_gate.arena`,
quoted in §1 and consistent with the amendment's condition `g_arena`). §3.3's construction —
five item-disjoint folds over train asserted against the banked `vsw_ckpt`, bank = fitting
pool, queries = the held-out fifth, `deployed_vote` at `k = 20` — **is** that arena, and
A4's three decisive reasons are each independently true: it is the only reading that uses
`headspace_{mint,arena}.py` as the condition names, the only one with a banked
`GATE-FLOOR` anchor, and the only one inside the condition's own *"minutes of CPU"* cost
model. I do not think a different reading is *required*, and I am not raising A4 as a
Critical.

But the condition also says *"on the already-banked `ro_*` caches"*, which the draft
satisfies by **forwarding** rather than **training**. That choice is legitimate. What is
not legitimate is presenting it as the C02 precedent when C02 held the image stream native
(V10), and then leaving the near-orthogonal transplant it creates ungated. **The reading
stands; the instrument built on it cannot render a verdict until C-2 is repaired.**

On the two subsidiary questions in §3.A:

* ***"on the already-banked `ro_*` caches"* and the L28 leg (§5.6).** Permitted on the
  text — the spec says `ro_*`, not `ro_L24`. But §3.1 concedes these four files are outside
  C01's frozen 8-file manifest, and the battery's authority rests on reading the same bytes
  C01 read. Combined with I-8, this is why I rule for dropping the leg.
* **The two named design constraints.** Both are **honoured, not merely mentioned.**
  (i) §3.1 pins one lineage per dataset, asserts it at run time, makes no cross-dataset
  absolute comparison, and structures the two-dataset requirement as a conjunction of
  independently computed verdicts — I verified the lineage claim directly (V3).
  (ii) §10.1 declares the confound on the face of the verdict with the correct citation,
  which I verified against the source (V12).

## §3.G open issue 1 — the direction of "conservative"

**The draft's choice is correct and I would not disturb it.** A closure retires a candidate
permanently and forgoes an extraction that
`iteration_8_stage0_bounded_extraction_amendment` has already authorised in principle; a
SURVIVE only queues a proposal that must still clear the unchanged Stage-0 bar
(`c_bar_unchanged`), its own prereg, its own two review lineages and its own authorisation
(§10.3). The asymmetry in irreversibility is real.

Two conditions on that ruling. First, if "conservative" is doing the work, the design must
**disclose** what it buys — hence I-5. Second, it must not be allowed to excuse H-2: an
uncorrected disjunction is an arithmetic error, not a considered lean.

## §3.G open issue 2 — `GATE-ORBITSCALE`'s threshold

**Not the right bar, and the threshold is the wrong question.** Borrowing C01's two frozen
constants rather than inventing one is the right instinct, but the *quantity* they are
applied to does not carry the failure (**C-1**). The gate also sits ~1000× above the
`float32` noise floor it is described as guarding, so it is not the numerical-floor
protection §7.4 presents it as. Changing `1e-3` does not fix either problem. Add the
raw-vs-head dispersion gate.

## §3.G open issue 3 — must a `$0` closure survive C06's best shot (per-arm retraining, ≈ 4.4 h CPU)?

**No — but not for the draft's reasons, and the answer flips if C-2 is not repaired.**

The written condition is a *falsifier*, not a Stage-0 pilot. It closes what it measures, and
§10.2 already scopes the retrained realisation out under F66's caveat. Spending 4.4 h to
retire a gated candidate is disproportionate, and the exclusion is properly declared as a
scope limit rather than buried. I verified the projection for the excluded variant
(`195 × 40.39 + 195 × 34.40 = 14 584.05 s = 4.05 h`, V11).

The complication: of §3.3's three grounds for excluding it — cost, instrument-anchoring,
and the named instruments — the per-arm-retrained reading is the one that **does not have**
the near-orthogonal transplant problem, because a head trained on `ro_L24` sees its own
distribution. So if C-2 is repaired by a gate, exclusion is fine. If the designer would
rather not add that gate, the retrained reading becomes the cheaper route to a defensible
verdict and the cost argument stops deciding the question.

## §3.G open issue 4 — should the L28 leg be dropped?

**Yes, drop it.** Three reasons, in order of weight:

1. **I-8** — at L28 the endpoints are near-orthogonal, so the leg replicates nothing and
   §5.6's disagreement clause is meaningless as written.
2. It carries four files **outside C01's frozen 8-file manifest** into a battery whose whole
   authority is that it reads the bytes C01 read (§3.1 concedes this).
3. It is every `× 2 layers` factor in Phases 2, 3, 4 and 5 — roughly 40 % of the non-mint
   projection — for a leg that enters no decision.

Dropping it removes the post-hoc-leaning risk the draft names, shrinks the trust surface by
four files, and cuts the projection. If a second orbit is wanted, the honest framing is a
separate probe, not a leg of this one.

## §3.D — Is `GATE-FLOOR` a real fidelity anchor?

**Partly, and the gaps are C-2, C-3 and H-5.** It genuinely anchors the mint recipe, the
fold contract, the vote operator and the native forward — and A4 is right that no other
reading of the condition has it. It does **not** touch the `ro_*` forward path, the
head-space arm algebra, or macro-F1. So yes: **it can pass while the head-space arm
comparison is meaningless**, which is exactly what §3.D asks. Each of those three surfaces
needs its own anchor.

## §3.D — Other rows or angles where a near-zero displacement is amplified?

**No, and I verified the mechanism rather than accepting it.** §7.4 Finding 1 is correct.
`orthogonal_blocks` (`c01_policy_contrast_a0.py:1285-1290`) computes
`−sinθ·std + cosθ·ow`, which at `θ = 45°` equals `sinθ·(ow − std) + δ·ow` with
`δ = cos(π/4) − sin(π/4) = 1.11e-16`. `l2` amplification requires the first term to fall at
or below `δ`, i.e. `‖ow − std‖ ≲ 1.6e-16`. **No merely-small row reaches that**: measurement
(b) finds zero unmasked rows below `1e-3` on either dataset, fifteen orders of magnitude
above the amplification threshold. So the exactly-zero row is the only case and
`GATE-ZEROMASK` covers it; a near-zero-but-nonzero row is not a live risk at these angles.

The residual numerical risk is different in kind: it is the renormalisation of `float32`
rounding noise, which begins around `‖Δ‖ ≈ 1e-6` — three orders **below** the gate. That is
why C-1's repair is about dispersion and not about lowering the epsilon.

## §3.D — Is `GATE-LEDGER` sufficient for `test_path_opens == 0`?

**Yes on test; the gap is on dev.** I confirmed the first of §12's three layers directly:
`headspace_mint.py:106-116` replaces `torch.load` at import with a guard that raises on any
path containing `test_seen` or `/test`, and `c02_a0_mint.py:63-64` shows the wrapper-level
pattern the C06 wrapper will copy. The frozen `c09_guard` `sitecustomize` adds an
`open()`-level, component-wise, repo-scoped predicate in every process. That is stronger
than required, and the existence of `test_seen` ro-caches on disk (which I confirmed — 8
files across the two datasets) does not create an opening. The defect is that the ledger
declares nothing about the **dev** side, where the code does materialise labels — **H-4**.

## §3.C — `rule_1_compute_projection`

The four mint units and five arena units are real-scale on the real banked data as claimed,
and I re-multiplied every product and both totals successfully (V11). The unit costs are
credible: `headspace_mint.py` was run unmodified with its sha256 verified, and the measured
peak RSS (`1.25 GiB`) agrees with C09's measured `1.22 GiB` — the one prediction F118
records as having been accurate. The two-figure conservative/corroborating pattern and the
`2×`/`5×` sensitivity statement are the right shape, and the self-imposed *"if realised
exceeds conservative by more than 2×, that is a reportable process finding"* is a good
addition.

**But the enumeration is not exhaustive and one unit does not reconcile.** The brief told me
to hunt for an uncounted loop; I found four problems: **H-6** (`U4` smaller than its own
constituents — the one that could actually move the total), **I-1** (dev ro-forwards counted
that §3.1 says never happen), **I-2** (Phase 6's count contradicts the gate it prices; Phases
6-7 budgeted rather than measured), **I-3** (three uncounted per-process loops). The draft's
own stated weak point (Phase 3) is the right one to have flagged, and its sensitivity
statement is adequate **in form** — but H-6 means the sensitivity is anchored on a number
that does not reconcile with the other measurements in the same table.

## §3.C — `rule_2_heartbeat`

**Adequate as a specification, with one gap.** §9 is materially better than what F118
punished: a line-buffered handle (`buffering=1`), one line per mint at `≤ 50 s` intervals,
an **independent bash echo per mint** so a phase boundary survives a wedged python process,
and `elapsed ÷ projected` on every line so drift is visible without arithmetic.

The gap: the longest units are the six full-train mints at `49.30 s`, and the driver's echo
brackets the unit rather than sitting inside it. Under the `× 1.25` contention factor §8
itself budgets, the worst-case silent interval is `61.6 s` — over the stated `~60 s`.
`headspace_mint.py`'s `_metrics_spy` already produces one `eval_curve` row per epoch × split,
so an epoch-level line is free. **Repair:** emit a within-mint line.

**What the code-review lineage must verify** (this review does not, and cannot): that the
progress handle is opened `buffering=1` and never re-wrapped; that the bash driver's echo is
unbuffered and does not inherit a block-buffered stdout under `sbatch`; that all 36+
processes **append** rather than truncate and that concurrent appends do not interleave
partial lines; that the HALT path still writes a final line before exit; and that the
`elapsed ÷ projected` denominator is §8's frozen number rather than something recomputed at
run time.

## §3.B — Decision-rule internal consistency, HALT semantics, and NaN paths

**Multiplicity:** two defects, **H-1** (family smaller than C01's) and **H-2** (disjunction
uncorrected). The two-dataset conjunction is correctly characterised as conservative.

**HALT semantics: sound.** §5.5 is genuinely two-valued-plus-HALT, names the record label
`INSTRUMENT_INCONCLUSIVE`, states that a HALT *"may not be reported as a closure"*, keeps
C06's gate exactly where it is, and requires fresh authorisation to re-run. I found no way
to report instrument failure as closure.

**NaN paths: none as written, but the property is fragile.** Every gate in §6 is phrased as
a *pass condition* (`≤ 0.05`, `≥ 0`, "equals", "agree exactly"), so a NaN comparison
evaluates `False` and HALTs — fail-safe. The exposure is at implementation: for example
`GATE-ORBITSCALE`'s fraction is `0/0` if every item were masked, and an implementation
written as `if frac > 0.05: halt` would **pass** on NaN. That is the inverted-comparison
form of exactly the C09-lineage bug the brief cites (an undefined `p` vetoing its partner).
**Repair:** the design should mandate an explicit finiteness assertion on every gate
quantity *before* comparison, and the code lineage must check the comparison **direction**
gate by gate. In S4, a comparator whose paired difference is identically zero yields a
degenerate bootstrap; Holm then fails to reject and S4 fails — also fail-safe, but worth
stating.

**§5.6's non-decisional declaration: structurally airtight as written.** L28 enters no rule
and no family, the verdict follows L24 alone, and the disagreement clause forces disclosure.
The residual risk is rhetorical, not formal — which is why I rule for dropping the leg
rather than tightening the wording.

**§5.7: creates no tuning path.** It is recorded before measurement, S1-S5 are fixed against
frozen constants, and it names CLOSE as the *less* interesting outcome. Its only defect is
evidentiary — **I-10**.

## §3.E — Scope

**Would a CLOSE discharge the state file's rule? Yes — the falsifier is *not* impossible as
written, and A7 is not the obstacle.** The written condition is *"re-run C01's … battery"*;
C01's battery **is** the two-point contrast; re-running it in the fold-head arena is what
the state file asks for. §10.2 is honest that curvature stays unmeasured, and it is right
that `≥ 3` prompt points require extraction and are `$0`-impossible. My reservation is not
A7 — it is that a CLOSE rendered through the *current* instrument would in truth be scoped
to a **native-trained head applied out of domain** (C-2) and to an **unanchored
re-implementation of the arm algebra** (C-3), and §10.2 says neither. With C-1 to C-3
repaired, a CLOSE discharges the rule.

**§10.4's ban analysis holds.** The warrant is object mismatch and I tested it. F80's object
is extraction-instruction **language** — its `ban_scope`, quoted at full scope in
`gate0_reopen_2026_07_31.dispositions.gated[0].why_gated_not_struck` after that record's own
round-10 audit, opens with an unconditional on-`MHC_zh` closure and attaches the *"without
new mechanism"* conditionality only to *elsewhere*. F70's object is individual **readout
cells**. C06's object is the **relation between** two cells, which is neither. The
multi-prompt **ensembling** carve-outs in both bans have C14 as their object, and §10.4
correctly declines to rely on them. The battery forms no ensemble of prompt predictions
(§10.4), which I confirmed against §3.3: `avg_score` is C01's own frozen `gain_control`
serving as a control, not a method arm. One omission, **M-2**: `endpoint_std` and
`endpoint_ow` literally *are* F70's two cells, entering here as controls — worth saying
where a reader will look for it.

**Hard constraints: none touched.** Checked against
`iteration_8_stage0_bounded_extraction_amendment.amended_rule.conditions.d_no_other_relaxation`
— no OCR; no cross-dataset mixing (§3.1 is explicit, and the two-dataset requirement is a
conjunction of independently computed verdicts); no external model API; single-dataset train
split only; parent-video binary label the only gold; no ensemble stacking; no model-size
scaling; SLURM-only with conda `HateVideo` and no `--time` (§13).

## §3.F — Execution boundary

**SLURM is right, and the dismissal of a login-node `nohup` is correct.** I read
`ERRPAT_HateMM_2026-07-26.md` §0.1 myself: it describes re-running the
`enc3seed_lora_curric.sbatch` command CPU-side and prices it at *"Cost: 52 s wall per seed
on 8 CPUs"*, **naming no non-SLURM channel**. A 52-second forensic process is not a
precedent for a 44-minute job. Against it stand CLAUDE.md's standing rule, C01's own frozen
`execution.require_slurm = true, cpu_only = true, required_cpus = 8` (verified in
`c01_a0_v2.json`, with `required_memory: 32G` added in `c01_a0_v3.json`), and the C02/C09 A0
precedents. **SLURM.**

**Cloud routing correctly held inapplicable, on reason (b).** `GATE-FLOOR` anchors to six
floors measured locally on `foscsmlprd01`; the standing ruling's same-table-same-hardware
precondition would require re-minting all six on cloud hardware, which costs more than the
job. Reason (a) is right that `squeue` belongs at submission time. Reason (c) — *"at 44 min
this is not long-running"* — is weaker than the draft implies, because that figure is
precisely the kind of projection F118 says may not be trusted for routing until measured;
but (b) decides it alone.

**§7.6's overrun — I rule the trade was made correctly, and I would have made it the same
way.** `rule_1_compute_projection` is a standing rule in `TARGET_STATE.json`, adopted after
a measured incident, that `applies_immediately_to` this falsifier by name, and it is
categorical about not extrapolating from reduced scale. The *"~5 CPU-minutes"* cap was an
instruction in a task brief. When a brief's budget and a standing rule are incompatible,
the standing rule wins and the overrun is disclosed — which is what happened, at `$0`, zero
GPU, on a 64-core node. The disclosure is also honest in the right direction: it reports
**22 CPU-minutes** rather than the 3.5 wall-minutes that would have sounded compliant.
Nothing in the overrun contaminates the design; §7.5's account is consistent with everything
I could check independently.

One criticism: the incompatibility was knowable **before** the burn — C09's banked mint costs
already put one unit near 4 CPU-minutes — so this should have been raised as a conflict
rather than resolved unilaterally and reported afterwards. That is a process note, not a
finding against the design.

## The dry check's blindness discipline (§7.4, "The head is deliberately UNTRAINED")

**The untrained-head choice does preserve preregistration blindness, and it is the right
call.** A trained head would have revealed an arm ordering before freeze. The choice makes
every *operation* real at real scale — real caches, real mints, real head forwards, real
`deployed_vote` on real minted keys — which is what F114/R3 demands, while making the
*numbers* scientifically void; §7.5 confirms no accuracy was computed, printed or recorded,
and no `dev_seen` or `test_seen` file was opened. I found nothing inconsistent with that
account.

The cost of the choice is that Finding 2's `219×` is measured on an object that tells you
little about the trained head — which the draft states plainly in §7.4 and then partly
forgets in §5.7 (**I-10**), and which is why C-1's repair must be anchored on the **raw**
space rather than on the untrained head's numbers.

---

# PART D — MINIMAL SET OF CHANGES THAT WOULD EARN GO

1. **C-1** — add the raw-vs-head displacement-dispersion HALT gate, with its bar fixed
   before the run off the banked raw-space values; extend `GATE-VIABILITY` to both arms
   in `R`.
2. **C-2** — add a head-space input-domain fidelity HALT gate (head-space `endpoint_std`
   OOF accuracy against the native `GATE-FLOOR` value); correct §3.3 and §4 A4's
   characterisation of the C02 precedent to what `c02_a0_mint.py:214` and `:21-23`
   actually do.
3. **C-3** — import `c01_policy_contrast_a0` and gate the re-implemented arm builder
   bit-exactly against `prepare_views` on the raw L24 features, HALT above `2e-6`.
4. **H-1** — restore `displacement` to the comparator set for `common_displacement`.
5. **H-2** — apply a real multiplicity correction to the two-arm disjunction.
6. **H-3** — state the bootstrap unit and re-derive §8 Phase 4 from it.
7. **H-4** — correct §12's dev-label sentence; give `GATE-LEDGER` C09's full declared-count
   predicate set.
8. **H-5** — extend `GATE-FLOOR` to macro-F1 on all six cells.
9. **H-6** — reconcile or re-measure `U4`; failing that, declare Phase 3 UNKNOWN.
10. **I-1 … I-10** as written, with §14 issue 4 resolved by **dropping the L28 leg** (which
    also discharges I-8 and part of I-3), and an explicit finiteness assertion mandated on
    every gate quantity before comparison.
11. **M-1 … M-4** — citation and wording corrections.

---

*Read-only review. No GPU, SLURM, Modal, model load, training, cache write, test-split
access, job submission or commit occurred. `TARGET_STATE.json`, the draft, and all configs
were read and not modified. A GO on this lineage would authorise nothing to run: the design
would still require freeze with hashes, a **separate** independent code/resource review
lineage over the executable reaching its own `0C/0H/0I`, and main-dialogue authorization.*
