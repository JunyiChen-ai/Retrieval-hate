# C06 `$0` falsifier — fresh independent design review request (round 1)

**Type:** read-only static design review. **No execution of any kind is authorized** — no
SLURM job, no login-node run, no Modal, no GPU, no cache write, no commit, no edit to
`TARGET_STATE.json`.

You are reviewing an **unfrozen DRAFT preregistration** in `/data/jehc223/RGCL`. You have
not seen the designer's reasoning and must not ask for it. Judge the artifact. This is
round 1: there are no prior rounds for this candidate, so nothing has been pre-cleared and
every claim is unverified until you verify it.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT.md`.

**Nothing is implemented yet.** No battery script exists. This review is of the **design**
only. A separate, independent **code/resource review lineage** runs afterwards over the
executable — do not conflate the two, and do not grant the implementation any credit here.

---

## 0. What C06 is, in one paragraph, so you need no prior context

C06 (*Prompt-Orbit Tangent/Curvature*) is a shelved research candidate claiming that the
tangent and curvature of a video's representation across a fixed prompt orbit encode
policy-bound semantic instability that no single prompt captures. It is **not** an active
candidate: its registry status is `gated_on_zero_cost_falsifier`. An earlier candidate,
C01, measured the two-point (first-order) case in a **raw-key** arena and found that the
best of six matched-norm orthogonal rotations of the prompt endpoints **matched or beat**
the real prompt displacement on both datasets — adverse to C06's premise. Because that
measurement was made in the raw key space and the campaign's registry says a raw-key arena
*"may kill but may not promote"*, the Gate-0 adjudicator did not strike C06. It gated it
behind a zero-cost falsifier: re-run C01's battery in the **fold-head (deployed-head)**
arena on already-banked caches. If the rotations again match the displacement there, C06
closes for `$0` and an authorized `1.7–2.5 GPU-h` extraction is never spent; if they do
not, C06 has earned that extraction. **This draft is the design of that falsifier.**

---

## 1. Read first, in this order

1. `CLAUDE.md` and `AGENTS.md` — the user's own standing instructions (execution channel,
   subagent policy, cloud routing). Note that neither may be edited by anyone acting on
   this review.
2. `TARGET_STATE.json`, four blocks, and read the C06 text **verbatim** rather than
   through the draft's quotation:
   - `gate0_reopen_2026_07_31.dispositions.gated[0]` — C06's hold entry, its
     `falsifier_spec`, its `falsifier_design_constraints`, and
     `rotation_family_precision_R14`;
   - `iteration_8_queue_state_2026_08_04` — what is and is not authorized right now;
   - `process_rule_compute_projection_and_heartbeat_2026_08_04` — the two standing rules
     this design must satisfy, adopted four hours before this draft was written;
   - `iteration_8_stage0_bounded_extraction_amendment` — what a SURVIVE would unlock, and
     under what conditions.
3. `TARGET_FINDINGS.md` — **F118** in full (the incident behind the two new process rules,
   and the fold-head arena machinery this design reuses). Skim F88 and F113 for why that
   arena has authority at all.
4. `refine-logs/GATE0_REOPEN_2026-07-31.md` §4.4 — the C06 disposition and the three
   corrections attached to it.
5. `refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` §3 — the recon that proposed C06's
   pre-kill, including its structural objection and its asset/cost audit.
6. `refine-logs/C09_A0_V17_RECORD.md` §2 — the banked description of the fold-head arena
   (mints, folds, floors, cost discipline), which this design says it reuses.
7. Only then: `refine-logs/C06_FALSIFIER_PREREG_DRAFT.md`, the artifact under review.

**Primary sources for the code claims** (read the code, do not trust the draft's summary):
`scripts/analysis/headspace_mint.py`, `scripts/analysis/headspace_arena.py`,
`scripts/analysis/c02_a0_mint.py`, `scripts/analysis/mechfix_ops.py`,
`scripts/analysis/mechnov_pairverify.py`, `scripts/analysis/c01_policy_contrast_a0.py`
(especially `orthogonal_blocks` / `paired_key` / `contrast_blocks` / `prepare_views`),
`configs/c01/c01_a0_v2.json`, `src/utils/generate_VideoMLLM_embedding_readout_HF.py:73-89`.

---

## 2. Verify these facts yourself before assessing any argument

Every one of these is a load-bearing factual claim in the draft. Recompute or re-read each
from the primary source; report any mismatch.

| # | claim to verify | where the draft asserts it |
|---|---|---|
| V1 | All 18 sha256 digests (11 modules/configs + 8 input caches) match the files on disk. | §11 |
| V2 | The four `ro_L24` digests' first 16 hex characters equal C01's frozen `*_provenance_sha16` fields in `configs/c01/c01_a0_v2.json`, and the HateMM one matches C01 v3's `diagnostic_train_cache_sha256` in **full**. | §3.1 |
| V3 | HateMM has **only** `-LoRA-curric` ro-caches and MHC_zh **only** `-LoRA` — one adapter lineage each, not a matched pair. | §3.1 |
| V4 | Every `ro_*` cache's `ids` order and `labels` vector are identical to the native deployed bank; `n = 744 / 579`; features are `(n, 3584)`. | §3.1, §7.1 |
| V5 | HateMM train row 355 is exactly zero in **both** modalities of **all four** ro caches and of the native cache; MHC-ZH has no exact-zero row. Cross-check against `configs/c01/c01_a0_v3.json`'s `authorized_null` block. | §7.1 |
| V6 | The six banked `scripts/analysis/headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json` carry `acc_deployed` = `0.8884 / 0.8858 / 0.8858` (HateMM) and `0.8929 / 0.8895 / 0.8946` (MHC-ZH). | §6 `GATE-FLOOR` |
| V7 | `mechfix_ops.deployed_vote` is numerically the same operator C01's config specifies (`topk = 20`, descending integer weights, signed cosine, cutoff `≥ 0`). | §3.2 |
| V8 | C01's frozen angle grid is `[8.3, 17.6, 29.1, 60.4, 72.7, 83.8]`, that `45°` is excluded *because it is the primary*, and that `orthogonal_blocks` is a Givens mixing in which `θ = 0` is `endpoint_concat`. | §3.3 |
| V9 | The paired C01 arms really are `2 × 3584` per modality while the deployed head's projections are `Linear(3584 → 1024)` — i.e. the draft's claim that those arms **cannot** be fed to the deployed head is true, not rhetorical. | §3.3 |
| V10 | `c02_a0_mint.py` really does train one head per `(dataset, seed, fold)` on the **native** cache and then forward that same head over each view's features — i.e. the precedent the draft leans on exists and says what the draft says. | §3.3 |
| V11 | The draft's arithmetic in §8: every product, and the two totals. Recompute them. | §8 |
| V12 | `generate_VideoMLLM_embedding_readout_HF.py:73-89` really does change the readout span as well as the prompt for the `ow_` cells. | §10.1 |

---

## 3. What you must assess

### A. Does the design implement the written unblock condition, exactly?

Read `falsifier_spec` and `falsifier_design_constraints` **verbatim from
`TARGET_STATE.json`**, then judge whether §3 and §5 of the draft implement them — not
something adjacent, not something better. In particular:

- Does *"in the FOLD-HEAD ARENA"* mean what §3.3 says it means? The draft enumerates three
  implementable readings and picks one on three stated grounds. **This is the single most
  consequential choice in the document.** If you think a different reading is required by
  the condition, say so as a **Critical** — the whole verdict changes with it.
- Does *"on the already-banked `ro_*` caches"* permit the L28 leg (§5.6), which uses files
  outside C01's frozen 8-file manifest?
- Are the two named design constraints (single adapter lineage; prompt/readout-span
  confound declared) actually honoured, or merely mentioned?

### B. Is the decision rule genuinely pre-registered, and is it sound?

- §5.2's `S1`–`S5` and §5.3's closure rule: is every threshold either C01's frozen value
  or fixed before measurement? Check each against `configs/c01/c01_a0_v2.json`. Any
  threshold that is neither is a **High** at minimum.
- §5.4's multiplicity treatment: the Holm family is 11 comparators × 2 metrics per
  `(real arm, dataset)`; the two datasets are a conjunction; the two real arms are a
  **disjunction**. Is the disjunction properly accounted for, or does it inflate the
  chance of a spurious SURVIVE?
- §5.5: is the HALT path genuinely two-valued-plus-HALT, with no way for an instrument
  failure to be reported as a closure?
- §5.6: the L28 leg is declared non-decisional **before** the run. Is that declaration
  airtight, or could it be leaned on after the numbers are visible?
- §5.7: a pre-declared expectation (CLOSE) is recorded. Does recording it create any
  path by which the design could be tuned toward it?

### C. The two process rules from F118 — do they actually bind here?

- **`rule_1_compute_projection`.** §8 must be a *measured unit cost × explicit count*
  projection with the multiplication shown, on the real path at the real scale. Verify
  that the four mint unit costs and five arena unit costs in §7.2/§7.3 are real-scale
  measurements and not reduced-draw extrapolations. Verify §8's counts are exhaustive —
  **the C09 failure was an enumeration that omitted realised draw counts, so hunt for a
  loop that §8 does not count.** Check the draft's own stated weak point (Phase 3) and
  judge whether its sensitivity statement is adequate.
- **`rule_2_heartbeat`.** Is §9's specification sufficient for an external observer to
  distinguish progress from a hang without attaching to the process? Is any phase able to
  run longer than the stated ~60 s interval without emitting a line? Note that §9 is a
  *specification* — the code review lineage will have to verify the implementation, and
  you should say what that lineage must check.

### D. The gates (§6)

- Is `GATE-FLOOR` a real fidelity anchor, or can it pass while the head-space arm
  comparison is still meaningless?
- **`GATE-ORBITSCALE` deserves your closest attention.** §7.4 reports a measured **219×**
  contraction of the prompt orbit between the raw key space (median displacement norm
  `0.7016`) and the deployed head space (`0.0032`), measured at an **untrained** head.
  The draft turns this into a HALT gate using C01's own two frozen constants. Judge: is
  the gate in the right place, is the threshold defensible in a space whose typical
  displacement is 219× smaller than the space that constant was chosen for, and — most
  importantly — **could a degenerate instrument still produce something that looks like a
  clean closure?** If yes, that is a Critical.
- `GATE-ALGEBRA` / `GATE-ZEROMASK` / `GATE-DUALPATH` exist because of a measured finding
  (§7.4, Finding 1: at `θ = 45°` an exact-zero displacement row is renormalised into a
  full-norm arbitrary direction by a 1-ULP `cos–sin` asymmetry, producing an `8.31e-02`
  guard violation attributable entirely to HateMM row 355). Verify the mechanism yourself.
  Then ask the question the draft does not: **are there other rows, or other angles, where
  a near-zero displacement produces the same amplification without being exactly zero?**
  A near-zero-but-nonzero row would pass `GATE-ZEROMASK` and still be amplified.
- Is `GATE-LEDGER` sufficient to guarantee `test_path_opens == 0` given that `test_seen`
  ro-caches **do** exist on disk for both datasets?
- Is any gate missing? Compare against C09's nine §8.1 HALT gates and say which of them
  has no C06 analogue and whether that matters.

### E. Scope (§10)

- Would a CLOSE, as scoped in §10.2, actually discharge the state file's rule — or does
  the curvature caveat (A7: two prompt points give a chord, not a curvature) mean the
  falsifier cannot close C06 as written? **If you conclude the falsifier is impossible as
  written, say so plainly; that is a legitimate review outcome and the designer was
  instructed to report it rather than improvise.**
- Does §10.4's ban analysis hold? Re-read F80's and F70's `ban_scope` yourself. The
  gate rests on **object mismatch**; test that.
- Does anything in the design touch a `hard_constraint` (no OCR, no cross-dataset mixing,
  no external API, single-dataset train split, parent-video binary label only, no
  ensembles, no size scaling, SLURM-only)?

### F. Execution boundary (§13)

- Is SLURM the right channel, and is the draft's dismissal of a login-node `nohup` correct?
  Check `ERRPAT_HateMM_2026-07-26.md` §0.1 yourself — the draft claims F88's `$0`
  forensics do **not** establish a login-node precedent for a 44-minute job.
- Is the cloud-routing ruling correctly held inapplicable?
- §7.6 discloses that the dry check consumed ≈ 22 CPU-minutes against a ≈ 5 CPU-minute
  instruction, on the grounds that one unit of the dominant loop is itself 4.3 CPU-minutes
  and the two constraints are incompatible. **Rule on that trade explicitly.**

### G. The four open issues the draft raises against itself (§14)

Rule on each. They are: the direction of "conservative"; `GATE-ORBITSCALE`'s threshold;
whether a `$0` closure must survive C06's best shot (per-arm head retraining, ≈ 4.4 h CPU,
still `$0` and GPU-free); and whether the L28 leg should be dropped rather than
run-and-reported. **A design that flags its own weak points is not thereby excused from
them** — treat each as an open finding until you close it.

---

## 4. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | The design, if executed as written, could publish a **wrong verdict** — a closure that is not warranted, or a survival that is not warranted. Also: any factual claim in §2 that is false, any test-split exposure, any un-preregistered threshold that touches a decision, any un-counted loop in §8. |
| **High (H)** | A defect that materially weakens the verdict's authority or its scope statement, but does not by itself invert it. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument that is right for a weaker reason than it could be. |

**GO requires 0 Critical, 0 High, 0 Improvement (`0C/0H/0I`).** Anything else is
**REVISE**. Report the tally explicitly in that form.

This bar is not ceremonial. On C09, seventeen design rounds were needed to reach
`0C/0H/0I`, and a **separate** code-review lineage then ran seven further rounds and
caught two wrong-verdict paths that seventeen clean design rounds had missed. Assume this
draft has defects and find them.

---

## 5. What a GO does and does not authorize

A GO on this review authorizes **nothing to run.** It states only that the *design* is
sound. Before any job is submitted, three further things must happen: (1) the design is
frozen with hashes; (2) a **separate, independent code/resource review lineage** reviews
the executable against this design and reaches its own `0C/0H/0I`; (3) main-dialogue
authorization is granted. A GO here is not a substitute for any of the three, and it is
not authority to write `TARGET_STATE.json`.

---

## 6. Deliverable

A written review containing:

1. The `0C/0H/0I`-form tally and a verdict of **GO** or **REVISE**.
2. Every finding with a severity, a `file:line` citation, and the concrete repair.
3. Your independent verification results for **all twelve** items in §2 — state
   `VERIFIED` / `MISMATCH` per item with the value you obtained. Do not skip items you
   consider obvious.
4. An explicit ruling on each of §3.G's four open issues.
5. An explicit ruling on §3.A's central question: **is the draft's reading of "the
   fold-head arena" the one the written condition requires?**
6. If you conclude the falsifier cannot discharge the written condition at `$0` at all,
   say so directly and state what would be required instead.

---

*Read-only. No GPU, SLURM, Modal, model load, cache write, test-split access, job
submission or commit is authorized by this document, and `TARGET_STATE.json` must not be
modified.*
