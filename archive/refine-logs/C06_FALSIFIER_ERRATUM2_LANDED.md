# C06 `$0` falsifier — **ERRATUM 2: LANDED**

*Landed:* 2026-08-05, in one pass, under the user directive that round 7 is the final review
of this erratum. **There is no proposal v8 and no round 8.**

*Design lineage:* proposals v1–v7 and reviews R1–R7, all on disk byte-unmodified:

| proposal | sha256 (16) | review | verdict |
|---|---|---|---|
| `…_ERRATUM2_PROPOSAL.md` | `f063c388c4afabdb` | `…_ERRATUM2_REVIEW.md` | REVISE 0C/3H/3I/3M |
| `…_V2.md` | `4225bea3cc9907d3` | `…_REVIEW_R2.md` | REVISE 0C/2H/3I/3M |
| `…_V3.md` | `48f4e0153103cc60` | `…_REVIEW_R3.md` | REVISE 0C/3H/3I/3M |
| `…_V4.md` | `0b4940416abd1fb4` | `…_REVIEW_R4.md` | REVISE 0C/2H/3I/4M |
| `…_V5.md` | `c41a0223bdf6db70` | `…_REVIEW_R5.md` | REVISE 0C/1H/3I/5M |
| `…_V6.md` | `05c93599b9ee4545` | `…_REVIEW_R6.md` | REVISE 0C/1H/3I/5M |
| `…_V7.md` | `9576da0d12ce3e2a` | **`…_REVIEW_R7.md`** | **REVISE 0C/2H/3I/5M — FINAL** |

Round 7's completeness verdict, carried forward as the warrant for landing without a v8:

> *"the 10×9 table survives both attacks: **no site anywhere in the five artifacts states a moved
> quantity in a form the table does not cover and no sweep reaches.** … That question, open since
> round 1, is closed."*

---

## 1. Files, before and after

| path | before | after |
|---|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E2.md` | *(did not exist)* | `254c0547c8e3579d2b5642747ceb686f9147dd0023a051e669ed9974edce5c4b` |
| `scripts/analysis/c06_falsifier_arena.py` | `0cdfd4f0c9f7095c355a0f0df5389a619576a5254311fdbb6bb9c956d2db0742` | `c0e20054e195152ac0b08f8671984e8ab47c871ce8e5d400f47118f2d933f936` |
| `scripts/analysis/c06_falsifier_mint.py` | `98f7b4a6f10cb39cd180541f91ffd319406c7048254566c4be2f731d9b4ad7f9` | `299d04020489362558c6f4411fb702fad19abf50ef105eccd5321442842474ea` |
| `configs/c06/c06_falsifier.json` | `e26784319a8ea7cd6c4c7b3011515f4c33ccf091d9d7b305d775b84a41e5adeb` | `29f1a57cce41f831819c2c5b9510bfe01fc1d148004323e7c23c944bd6f48ddf` |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `c3647173ff1556801c4c492f05e49e27e7fdb14ef80c1def8b9e492b6fa2fc4d` | `06d554c63cbc498b2af04f3cd4d45f2713cd9bce9db56d51ed35fc2ff5c510b4` |

**Byte-unmodified, verified after landing:**
`C06_FALSIFIER_PREREG_DRAFT_V15E1.md` = `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`;
`C06_FALSIFIER_PREREG_DRAFT_V15.md` = `75e3aa84a39f3c276be8b5b7fb1271e83c0c2e7e3b0ad0480b47465a81408228`;
all seven erratum-2 proposals at the digests tabled above.

**Landing order honoured:** V15E2 written first → its sha256 computed → every code/config edit →
`config:"design_sha256"` set **last**. Startup parity verified: declared == on-disk == `254c0547…`.

---

## 2. Obligations checklist

### H-1 — Phase 1d re-priced by RECOMPUTING the product ✅

`U7 = 0.13 s`. The count rises `1 → 2`, so the product is **re-multiplied**, `2 × 0.13 = 0.26 → 0.3`
at one decimal — by §8's own worked example at Phase 1c (`67 × 0.033 = 2.211 → 2.2`) — **not** by
doubling the already-rounded `0.1`. Every §8-derived figure recomputed and verified by script:

```
total      = 2642.5 + 1.0 + 0.7 + 0.1 + 1.3 + 7.6 + 7.0 + 1013.8 = 3674.0
x1.25      = 4592.5          minutes = 61.23 -> 61.2  /  76.54 -> 76.5   (unchanged)
2x miss    = 4687.8 = 78.1 min      5x miss = 7729.2 = 128.8 min
mint share = 68.27 % -> 68.3        Phase 3 = 27.59 % -> 27.6            (unchanged)
7729.2 / 4592.5 = 1.68x                                                  (unchanged)
```

**Propagated to the heartbeat denominator and its three HALT assertions**, and the sweep-C/F
machinery re-run over all five landed artifacts to guarantee no stale figure survives. The residue
is four hits, every one a **deliberate historical or explanatory** statement:
`config:"what_moved"` (describes the move), `V15E2:1578` (the erratum note explaining
`2642.3 → 2642.5`), `V15E2:1583` (CODE-R1 H-4's provenance arithmetic, row 56 — historical),
`V15E2:1644` (§9's provenance chain naming all three values). **No live figure is stale.**

### H-2 — the resume-safe `C06_MINTS_EXECUTED` clause restored ✅

Round 7's prescription taken in full. `MINT_N` counts **attempts** and is 66 on every run including a
resume; exporting it would HALT a legitimate resume against
`dev_path_opens == mints_executed + expected_sha_dev_opens`. The sbatch now carries a separate
`MINTS_EXECUTED`, incremented on **exactly the mint driver's own skip condition** — the `.npz`
absent at call time, which is what `c06_falsifier_mint.py:218-220` tests before returning at
`MINT-SKIP` ahead of the dev load:

```bash
[ -f "$OUT" ] || MINTS_EXECUTED=$((MINTS_EXECUTED+1))
...
export C06_MINTS_EXECUTED="$MINTS_EXECUTED"
```

`C06_PROJECTED_SECONDS` is exported **from `config:"execution"."projected_seconds"`**, the second
restored provenance clause. §12's *"Why `mints_executed` and not `66`"* warrant is amended in V15E2
to state the semantics rather than leave them to the implementer.

### I-1 / I-2 / I-3 and the five minors ✅

* **I-1** `arena:474`'s string-literal authority claim is inside row 39's rewrite and is corrected
  with the docstring; the whole blocked-predicate paragraph now states the two-term form.
* **I-2** the meta-check subtraction is `102 = 32 + 70` against the twelve sweeps; the vocabulary
  lists (12 phases, 16 units) and the conclusion are unchanged.
* **I-3** `mint:117` lands as **`72 of 74`**, not `73 of 74`, with the driver leg named as the second
  writer — the count is of processes that wrote **nothing**, and the process being added *wrote*.
* **M-1** §9's split restated; **M-2** the `:1583` charge names both patterns; **M-3** rows 49 and 52
  land with their decompositions spelled out (`7 + 6 + 8 + 1 = 22`, `22 + 16 = 38`); **M-4** row 66
  keeps *"and opens no dev_seen file (§12)"*; **M-5** the empty-cell reasons are stated per column.

### CODE-R1 carried items, landed in the same pass ✅

| item | landed as |
|---|---|
| **I-4** GATE-ZEROOP union ranking | guard arms retrieve to depth `2 × 21`; `tie_casualties` takes both arms' index arrays and ranks each arm over the **union** of the two top-21 sets. **CODE-R2 I-1 — corrected claim:** the bound is **exact under the nominal ordering and conservative under near-tie permutation**, not exact in general. `_union_ranked` also drops an arm's own ranks 21–41 that are not union members; those carry weight 0 nominally, but dropping them changes the gap structure on which `vote_bounds_over_orderings` cuts its near-tie groups, so an ordering available on the full 42-list can be unavailable on the restricted one. The residue is bounded by the similarity window (itself bounded by `GATE-ALGEBRA`'s `2e-6 × √2048 = 9.05e-5`), is directed toward **under-counting** casualties — which can only make a lineage more likely to drop, never more likely to publish `SURVIVE` — and the path runs at all only when two algebraically identical arms disagree. |
| **M-6** `:2724`-equivalent assertion | `cell["selected_control"]` is recorded from the selector's own return and asserted equal to `cell["reference"]`, so the comparison is between two independently-carried values rather than a value with itself. |
| **M-7** finiteness on decision quantities | `finite()` now also guards S4's `lower`/`one_sided_raw_p`, S5's `observed`/`p95`/`one_sided_raw_p`, `net_s`, and S7's `fixed_fraction`/`threshold`. |
| **M-10** arena import set | `headspace_mint` and `vsw_pregate` imported; `runtime_block()` called and recorded at `reports["runtime"]`, so the verdict carries runtime provenance. |
| **M-12** resource accounting | `load_ro` **memoised per dataset**: the three call sites executed six load events against §8 Phase 1c's one for the arena. Memoising makes the repeats free, so the code matches the design's count rather than drifting from it — the drift is removed at its cause instead of being re-priced. |

---

## 3. Dry-check battery — outputs

Login node, `$0`, payload-real, CPU only, `artifacts/` untouched.

**GATE-SHA, with the design-pointer gate deriving at startup — 38/38:**
```
GATE-DET1   | 1/1 | thread env verified
GUARD       | 1/1 | c09_guard layer 3 active in this process
PROJECTION  | 1/1 | single source 3674.0 s agrees across config/module/env
GATE-SHA    | 38/38 | all frozen digests match
GATE-SHA-ONLY | 1/1 | driver precondition satisfied
```

**GATE-C01PARITY / ROWSUBSET / RHORAW — bit-exact on both datasets:**
```
GATE-C01PARITY | 1/1 | hatemm max|diff|=0.0 in 11.6s
GATE-ROWSUBSET | 1/1 | hatemm bit-exact bridge
GATE-RHORAW    | 13/13 | hatemm 13 arms at 4 dp
GATE-C01PARITY | 1/1 | zh max|diff|=0.0 in 4.8s
GATE-RHORAW    | 13/13 | zh 13 arms at 4 dp
DRY-PARITY-COMPLETE | 2/2 | no arm accuracy computed
```

**Both new HALT gates falsifier-tested — each fires before any battery compute:**
```
(design digest drifted)  HALT | gate=GATE-SHA | design document ...V15E2.md on-disk digest
                              254c0547... != declared 0000...
(denominator drifted)    HALT | gate=GATE-DET1 | projected_seconds disagree:
                              config 3674.0 / module 9999.0 / env 9999.0
```

**The HALT artifact publishes both digests plus the caveat**, and `sha256_derived` correctly reads
`NOT_DERIVED` when the HALT precedes `gate_sha`:
```json
"design": {"document": "refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E2.md",
           "sha256_declared": "0000…", "sha256_derived": "254c0547…",
           "note": "declared digest is not pinned inside the job; the external anchor is
                    the freeze record in refine-logs/C06_FALSIFIER_IMPLEMENTATION_RECORD.md"}
```

**`expected_sha_dev_opens` derived, not read** — re-derived under `c09guard`'s own predicates:
```
concatenated iterable : 21 files
dev-like              : 2  (dev_seen_...-LoRA-curric_HF.pt, dev_seen_...-LoRA_HF.pt)
test-like             : 0
design doc dev/test?  : False False       <- adding it leaves the count at 2
derived expectation   : 2 x 2 passes = 4  ==  config declares 4     MATCH
ledger (measured)     : test_path_opens, mints_present_before_arena, dev_path_opens,
                        banked_trainlog_opens, processes_reporting
ledger_by_construction: test_label_materialisations,
                        dev_label_materialisations_outside_decisions,
                        dev_or_test_labels_into_decision_quantities
```

**Blindness.** The only occurrence of *"accuracy"* anywhere in the dry outputs is the line asserting
that none was computed (`DRY-PARITY-COMPLETE … no arm accuracy computed`); the only decimals present
are `elapsed ÷ projected` ratios. `deployed_vote` was called zero times on any `ro`-derived battery
arm; no mint ran; no GPU, no SLURM job, no commit, no `TARGET_STATE.json` edit.

---

## 4. Two things to hand the verification lineage, stated rather than buried

**(a) The driver leg now emits FIVE heartbeat lines, not four.** Round-7 I-3 corrected v6's *"three"*
to **four** (`GATE-DET1`, `GUARD`, `GATE-SHA`, `GATE-SHA-ONLY`), citing round 2's enumeration. This
landing adds a fifth, `PROJECTION`, because §7's single-source assertion sits in the
pre-`gate_sha_only` block by round-5 M-3's prescription. The dry-check transcript above shows all
five. V15E2 carries **no** line-count claim (verified by grep), so no design text is stale — but any
future statement of the count must say five.

**(b) An artifact was created and removed during the dry checks — disclosed, not hidden.** The first
falsifier run of the design-digest gate was invoked without `--out`, and `--out` defaults to
`artifacts/c06_falsifier/C06_VERDICT.json`; the HALT path therefore wrote that file into the repo
tree, where every review has verified `artifacts/c06_falsifier/` **absent**. I checked its provenance
(created 04:24 this session, one file, a HALT record from my own deliberate falsifier test), removed
the directory, and re-ran both falsifier tests with `--out` pointed at the session scratchpad. The
directory is absent again, verified. **This is behaviour the design intends** — round 5 noted that a
design-drift HALT in process 1 lands on the canonical verdict path — but it means the dry-check
protocol must pass `--out` explicitly.

**CODE-R2 M-3 — and `--progress` as well.** `Heartbeat.__init__` runs `os.makedirs` on the progress
path's parent and opens the file **before any gate runs**, and `--progress` defaults to
`artifacts/c06_falsifier/progress/C06_PROGRESS.txt`. A dry check that passes only `--out` and lets
`--progress` default therefore recreates `artifacts/c06_falsifier/` anyway. **Every $0 dry check must
pass BOTH `--out` and `--progress` into the session scratchpad.**

---

## 5. What is NOT authorized by this document

Landing an erratum is not authorization to run. Still required before the battery executes:
the scoped code-lineage verification pass (erratum-2 landing fidelity, the five carried CODE-R1
items, and an end-to-end synthetic verdict-path drive), and then the freeze and the user's
submission authorization. `TARGET_STATE.json` is unmodified; nothing is committed; no job is queued.
