# C06 `$0` falsifier — fresh independent design review request, **ROUND 2**

**Type:** read-only static design review. **No execution of any kind is authorized** — no
SLURM job, no login-node run, no Modal, no GPU, no cache write, no commit, no edit to
`TARGET_STATE.json`.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V2.md` (v2, unfrozen).

You are a **fresh reviewer**. You have not seen the designer's reasoning and must not ask
for it. Round 1 returned **REVISE (3C / 6H / 10I + 4M)**; v2 claims to have adopted **all 23
findings, rebutting none**, with one adopted in a refined form. **Do not trust that claim.**
The house precedent that matters here: on C02, two separate rounds caught a repair the
record asserted and the artifact did not contain. **Verify every claimed repair against the
primary source, and treat v2's disposition table (§14) as an index of things to check, not
as evidence.**

**Nothing is implemented yet.** No battery script exists. This reviews the **design** only;
a separate independent code/resource review lineage runs later over the executable. Do not
grant the implementation credit here.

---

## 0. What C06 is, so you need no prior context

C06 (*Prompt-Orbit Tangent/Curvature*) claims that the tangent and curvature of a video's
representation across a fixed prompt orbit encode policy-bound instability no single prompt
captures. It is **not** an active candidate: its registry status is
`gated_on_zero_cost_falsifier`. An earlier candidate, C01, measured the two-point case in a
**raw-key** arena and found the best of six matched-norm orthogonal rotations of the prompt
endpoints **matched or beat** the real prompt displacement on both datasets — adverse to
C06. Because the registry says a raw-key arena *"may kill but may not promote"*, the Gate-0
adjudicator gated C06 rather than striking it: re-run C01's battery in the **fold-head
(deployed-head)** arena on already-banked caches. If the rotations again match, C06 closes
for `$0` and an authorized `1.7–2.5 GPU-h` extraction is never spent; if not, C06 has
earned that extraction. **The artifact under review is the design of that falsifier.**

---

## 1. Read first, in this order

1. `CLAUDE.md`, `AGENTS.md` — the user's standing instructions. Neither may be edited by
   anyone acting on this review.
2. `TARGET_STATE.json`, four blocks — read the C06 text **verbatim**, not through v2's
   quotation: `gate0_reopen_2026_07_31.dispositions.gated[0]` (hold entry, `falsifier_spec`,
   `falsifier_design_constraints`, `rotation_family_precision_R14`);
   `iteration_8_queue_state_2026_08_04`;
   `process_rule_compute_projection_and_heartbeat_2026_08_04`;
   `iteration_8_stage0_bounded_extraction_amendment`.
3. `TARGET_FINDINGS.md` — **F118** in full; skim **F88** and **F113**.
4. `refine-logs/GATE0_REOPEN_2026-07-31.md` §4.4;
   `refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` §3.
5. `refine-logs/C09_A0_V17_RECORD.md` §2 and §8.1 — the banked arena description and C09's
   nine HALT gates, which v2 borrows from.
6. `refine-logs/C06_FALSIFIER_PREREG_REVIEW.md` — **round 1**, in full.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT.md` — **v1**, superseded. Read it *after* round 1
   so you can judge whether v2's repairs land.
8. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V2.md` — **the artifact under review**.

**Primary sources for the code claims** (read the code, not v2's summary):
`scripts/analysis/c01_policy_contrast_a0.py` (especially `l2_rows:1183-1205`,
`fuse_modalities:1208-1217`, `paired_key:1220-1239`, `contrast_blocks`,
`orthogonal_blocks:1272`, `prepare_views:1294`, and the deferred-import site
`:387` / `import_compute_modules:1048`), `scripts/analysis/headspace_mint.py`,
`scripts/analysis/c02_a0_mint.py`, `scripts/analysis/mechfix_ops.py`,
`scripts/analysis/mechnov_pairverify.py`, `scripts/analysis/headspace_fidelity.py`,
`src/model/classifier.py`, `configs/c01/c01_a0_v{2,3,4}.json`,
`src/utils/generate_VideoMLLM_embedding_readout_HF.py:73-89`.

---

## 2. Verify these facts yourself

Recompute or re-read each from the primary source. Report `VERIFIED` / `MISMATCH` with the
value you obtained, for **every** item — do not skip ones you consider obvious.

| # | claim | where |
|---|---|---|
| **V1** | All 21 sha256 digests in §11 match the files on disk (7 imported modules, 6 read-for-definition files, 8 caches). | §11 |
| **V2** | The four `ro_L24` digests' first 16 hex equal C01's frozen `*_provenance_sha16`; the HateMM one equals C01 v3's `diagnostic_train_cache_sha256` in full. | §3.1, §11 |
| **V3** | The v4 → v3 → v2 config chain and v4 → v3 → base algebra chain are as §11 describes, with `scientific_thresholds_exact: true` at each hop. | §11 |
| **V4** | Every product and both totals in §8 (`2867.1 s`, `3583.9 s`), plus the sensitivity figures (`52 min`, `66 min`), the Phase-3 share (`9.5 %`), the mint share (`82 %`), and the excluded-variant price (`14 584 s = 4.05 h`). | §8, §15.3 |
| **V5** | The Holm family size: `12 + 11 = 23` comparisons × 2 metrics = **46** per `(dataset, lineage)`, and that the per-arm control sets come from C01's `decision.gain_controls` **and** `statistics.bootstrap_comparisons.primary_vs_controls`. | §5.1, §5.5 |
| **V6** | The six banked `GATE-FLOOR` anchors on **both** metrics: acc `0.8884/0.8858/0.8858` and `0.8929/0.8895/0.8946`; mF1 `0.8838/0.8811/0.8812` and `0.8747/0.8710/0.8765`. | §6 |
| **V7** | The majority-class rates `0.5995` (HateMM) and `0.6891` (MHC-ZH) from the train labels. | §6, §7.1 |
| **V8** | The `ρ_raw` table — all 26 values — and that `ρ* = 0.9772` is the maximum over 13 arms × 2 datasets. **Recompute `ρ` yourself** on the raw L24 arms with the zero row masked. | §6.1 |
| **V9** | That C01's `decision` block contains `minimum_net_fixes {MHC_zh: 2, HateMM: 3}`, `require_primary_and_displacement_above_shuffle_p95`, `require_shuffle_holm_reject`, `require_rotation_bootstrap_holm_reject`, `require_no_small_displacement_dominance`, and `require_accuracy_gain_over_deployed_r0_context`. | §5.2, §5.8 |
| **V10** | That `c02_a0_mint.py:214` uses the **native** `img_feats` on every view and `:68` refuses any view file carrying an image stream — i.e. v2's withdrawal of v1's precedent claim is correct and complete. | §3.3 |
| **V11** | The measured near-orthogonality that motivates Head-R: median `cos(native_img, ro_L24_img)` ≈ `0.0234` / `0.0373`, with both caches unit-norm. | §3.3 |
| **V12** | That `headspace_mint.py:199` loads `dev_seen_*.pt` on **every** mint, `:322` writes `lab_dev` into every `.npz`, and `:229` makes the real dev split the training dev set at `fold == −1` — i.e. v2's withdrawal of v1's dev-label sentence is correct. | §12 |

---

## 3. What you must assess

### A. Do the three Critical repairs actually land?

**C-1 → `GATE-ORBITDISP` (§6.1).** Round 1's finding was that the gate watched displacement
**magnitude** while the false-CLOSE failure moves displacement **direction**, because every
key is `l2`-normalised before the vote.

- Does `ρ = ‖mean_i k_i‖` actually capture the named failure (a near-constant offset orbit)?
  Construct the failure yourself and check the gate fires.
- Is `ρ* = 0.9772` — the max over 13 raw arms × 2 datasets — the right calibration
  statistic? A max is the most permissive choice among order statistics; would a quantile,
  or a per-arm bar, be better? **Note the consequence either way:** a more permissive `ρ*`
  makes HALT rarer and CLOSE easier.
- Is the two-case split right — `ρ_raw ≤ ρ* ∧ ρ_head > ρ*` ⇒ HALT, both-degenerate ⇒ no
  HALT? Can you construct a case it mis-classifies?
- Is applying it to **all 13 arms** (not just the two real ones) correct, or does it create
  a HALT path that a true negative could trip?

**C-2 → two head lineages (§3.3, §5.3) + `GATE-ARENA`/`GATE-ARMVIAB`/`GATE-DOMAIN`.**

- Does running Head-R (`ro_L24`-trained) and requiring CLOSE to fail on **both** lineages
  genuinely neutralise the out-of-distribution objection, or does it relocate it?
- **Head-R has no banked floor anchor** — v2 concedes this and raises it as open issue §15.6.
  Rule on it. Is "shares every component except the training cache" sufficient instrument
  fidelity for a lineage that carries a decision consequence? If not, what banked object
  could anchor it?
- Is dropping Head-R's `fold = −1` heads correct, given `GATE-DEVFID` compares against
  banked **native** trainlogs?
- Is `GATE-DOMAIN` (§6.3) — a reported recovery fraction with **no bar** — an acceptable
  discharge of round 1's repair (a), or a gate in name only?

**C-3 → `GATE-C01PARITY` (§3.4, §6).** v2 claims a generic block-list builder that
reproduces `prepare_views` **bit-exactly** (`max|diff| = 0.000e+00`).

- **Reproduce the parity check yourself** on the raw L24 features, both datasets.
- v2 states that v1's `pair` was *wrong* — missing `fuse_modalities`' outer per-block
  normalisation, and `float64` where C01's `l2_rows` returns `float32`. Verify both claims
  in the source.
- The gate anchors the **two-block** instantiation. The verdict is rendered by the
  **one-block** instantiation. Does bit-exactness on two blocks actually constrain the
  one-block path, or is the load-bearing choice simply relocated into the shared helper?
  **This is the sharpest question in the round.**

### B. The decision rule

- **H-1/H-2/H-3:** are the per-arm control sets, the 46-hypothesis Holm family, and the
  bootstrap unit (§5.4: resample items once, average seeds inside each resample) mutually
  consistent, and consistent with §8 Phase 4's count of 92?
- **S6 (net fixes)** is newly imported from C01. Check the direction: net fixes are
  measured against `endpoint_std`. Is that C01's own reference (`fix_break_reference`)?
- Does requiring CLOSE on **both** lineages (§5.3) interact correctly with the two-dataset
  conjunction and the two-arm disjunction, or does it create a fourth multiplicity axis
  nobody has corrected?
- §5.6 mandates a finiteness assertion before every comparison and pass-condition phrasing.
  Is that specification sufficient, and what must the code lineage check?

### C. The process rules

- **`rule_1_compute_projection`.** Round 1 found `U4` smaller than its own constituents;
  v2 withdraws `U2a`/`U2b`/`U4` and re-measures (§7.6), attributing the defect to `float64`
  arms and a repeated single fold. **Check the new numbers reconcile**
  (`5 × 0.00305 + 5 × 0.00629 = 0.04674` against `U4 = 0.08908`) and that the stated cause
  is consistent with `mechfix_ops._norm32` and `c01_policy_contrast_a0.l2_rows:1200-1202`.
  Then **hunt for a loop §8 still does not count** — that was the C09 failure mode and round
  1 found four instances in v1.
- **`rule_2_heartbeat`.** §9 adds a within-mint epoch line to close round 1's `61.6 s`
  worst-case gap. Is any interval still longer than the stated bound? Note §9's list of what
  the code lineage must verify and say whether it is complete.
- §7.7 discloses that the first `GATE-DEVFID` timing was a **failure path** recorded as a
  measurement and then corrected. Judge whether the correction is sound and whether any
  other unit in §7 could have the same defect.

### D. Gates and scope

- v2 has **18 gates** against v1's 11. Are any of the new ones unfalsifiable, redundant, or
  able to fire on a warranted CLOSE? Compare against C09's nine and say which remain
  missing and whether that matters.
- Is `GATE-ZEROOP`'s *identical predictions* form genuinely stronger than `GATE-ALGEBRA`'s
  `2e-6` key form, as §6 claims?
- Does §10.2's scope sentence now say everything a CLOSE is actually scoped to?
- §5.8 discloses one C01 condition deliberately **not** carried
  (`require_accuracy_gain_over_deployed_r0_context`) on the ground that its comparator is a
  raw dev-arena figure and importing it would violate F88's CPU-arm/CPU-floor caveat. Rule
  on that reasoning.
- Does anything touch a `hard_constraint`? Does the L28 removal leave any dangling reference?

### E. Execution and honesty

- §13 keeps SLURM and **withdraws** v1's *"at 44 min this is not long-running"* reason. Is
  the remaining reasoning sufficient?
- §7.8 discloses a further ≈ 30 CPU-minutes of dry-check burn and states the conflict was
  raised rather than resolved silently. Rule on it.
- Is there anywhere v2 claims a repair the artifact does not contain?

---

## 4. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | The design, if executed as written, could publish a **wrong verdict** — an unwarranted closure or an unwarranted survival. Also: any false factual claim in §2, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, any claimed repair the artifact does not contain. |
| **High (H)** | Materially weakens the verdict's authority or its scope statement without inverting it. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**. Report the tally in that form.

The bar is not ceremonial: C09 needed seventeen design rounds to reach `0C/0H/0I`, and a
separate code-review lineage then ran seven more and caught two wrong-verdict paths that
seventeen clean design rounds had missed.

---

## 5. What a GO does and does not authorize

A GO authorizes **nothing to run**. It states only that the design is sound. Before any job:
(1) freeze with hashes; (2) a **separate** independent code/resource review lineage over the
executable reaching its own `0C/0H/0I`; (3) main-dialogue authorization. A GO is not
authority to write `TARGET_STATE.json`.

---

## 6. Deliverable

1. The `0C/0H/0I`-form tally and **GO** or **REVISE**.
2. Every finding with severity, `file:line` citation, and the concrete repair.
3. Your verification result for **all twelve** §2 items, with the values you obtained.
4. An explicit ruling on each of the six items in v2 §15, especially **§15.5** (the one
   refinement: `GATE-ARMVIAB`'s two-case form, adopted instead of round 1's one-sided
   majority-rate HALT) and **§15.6** (Head-R's missing floor anchor).
5. An explicit ruling on §3.A's sharpest question: **does bit-exact parity on the two-block
   instantiation actually constrain the one-block path that renders the verdict?**
6. A statement of whether any round-1 finding is **not** in fact repaired, naming it.
7. If you conclude the falsifier still cannot discharge the written condition at `$0`, say
   so directly and state what would be required instead.

---

*Read-only. No GPU, SLURM, Modal, model load, cache write, test-split access, job
submission or commit is authorized by this document, and `TARGET_STATE.json` must not be
modified.*
