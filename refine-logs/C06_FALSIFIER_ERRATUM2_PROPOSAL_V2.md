# C06 `$0` falsifier — **ERRATUM 2, PROPOSAL v2**

**Supersedes** `refine-logs/C06_FALSIFIER_ERRATUM2_PROPOSAL.md` (sha256
`f063c388c4afabdb7964360eda2fe1ef7d6fee611b2a287ca9817929c7f670f5`), which stays on disk as the
record of what round 1 reviewed.
*Against:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md`, sha256
`8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`.
*Adjudication of v1:* `C06_FALSIFIER_ERRATUM2_REVIEW.md` — **REVISE 0C/3H/3I/3M**, endorsing
option (i) and blocking on **scope**.
*Status:* **PROPOSAL. Nothing is landed.** Design, arena, mint, config and sbatch are **unedited**
for this defect and carry their post-CODE-R1 hashes.

---

## 0. What v1 got right, and what it got wrong

**Right, and carried forward unchanged:** the defect diagnosis, option (i)'s form, the rejections of
(ii) and (iii), the archaeology, and the C09 precedent. The reviewer reproduced the `+4` by
execution, verified every archaeology row against the actual files, and independently confirmed both
rejections — adding one ground v1 missed: hashing through `_ORIG_OPEN` would also blind the **test**
limb on the same call path, and `_guarded_open` is the only thing that *raises* on a test path.

**The load-bearing question v1 asked to have measured came back clean.** `torch.load` performs
**exactly one** `builtins.open` per file on this build (torch 2.6.0+cu124, all three caches
zipfile-format, spy-verified per call): `_open_file_like` resolves the path to a file object once
and hands that object to the zip reader. And `headspace_mint.py:198-199` performs both `load_split`
calls **unconditionally**, before the `if a.fold >= 0:` branch at `:218` — so every executed mint
opens the native `dev_seen` exactly once, both lineages, all folds. **`mints_executed` is the right
variable with the right multiplier; the first term needs no static-derivation treatment.**

**Wrong: the scope.** v1 diagnosed the omission — §12 never accounts for the `GATE-SHA` driver
process — and then repaired only **one** of the two binding predicates that omission breaks. The
reviewer measured a clean run at **74** reporting processes against a binding **73**. Landing v1 as
written would pass `dev_path_opens` and then HALT at `processes_reporting`, in the arena, **after
the entire battery has been spent**. v1's closing sentence (*"the battery cannot pass `GATE-LEDGER`
until this erratum lands"*) is true but misleading by omission; the accurate statement was
*"…and still cannot after."*

**Wrong: one overstated claim.** v1 said the rest of the CODE-R1 H-2 wiring *"is correct as it
stands"*. It is not: `dev_label_materialisations_outside_decisions` is binding in both §12 and the
config, is incremented by nothing in the job, and is not evaluated. That is CODE-R1 H-2's original
complaint surviving in one row.

**Wrong: one derivation claim.** v1 said both factors of the new expectation are statically
derivable from the frozen tables. Only one is. §6 and §13 say `GATE-SHA` runs **once**; the `2`
comes from `c06_falsifier_arena.py:1276`, i.e. from code.

---

## 1. The completeness obligation — ALL of §12, reconciled

This is the lesson of being caught twice. **Every §12 predicate is enumerated below with its
expected formula, the artifact the formula is derived from, a measured value where measurable
pre-run, and its binding status.** The point is that a reviewer can check for a *third* collision
without reconstructing the table themselves.

| # | predicate | expected (proposed) | derivation source | measured pre-run | binding | change |
|---|---|---|---|---|---|---|
| 1 | `test_path_opens` | `0` | §12; `c09guard.is_test_like` | **0** — measured across the `--gate-sha-only` process; `_guarded_open` *raises* on a test path, so a non-zero value cannot be reached silently | **yes** | none |
| 2 | `test_label_materialisations` | `0` | §12; by construction — no C06 code path reads a test label | not directly measurable pre-run; **by construction**, and §3.1 states no `test_seen` cache is opened by anything | **yes** | none |
| 3 | `mints_present_before_arena` | `66` = 36 Head-N + 30 Head-R | §3.3's lineage table | not measurable pre-run (no mints exist); `gate_fold_and_ledger_presence` counts `.npz` files and already binds it | **yes** | none |
| 4 | `dev_path_opens` | **`mints_executed + expected_sha_dev_opens`**, `expected_sha_dev_opens = 2 × 2 = 4` | dev-like count from the **concatenated iterable `gate_sha` hashes**; process count audited from the ledger (§3) | **`+2` per GATE-SHA process, measured**; one driver process measured at exactly `dev_path_opens = 2`; `torch.load` = 1 open/file | **yes** | **ERRATUM 2** |
| 5 | `dev_label_materialisations_outside_decisions` | **demoted to REPORTED / by construction**: `lab_dev` is written into every `.npz` by `headspace_mint.py:322-324`; no `open()`-level instrument can see a materialisation | §12's own "Dev labels" paragraph; C09's `by_construction_zero` precedent | **0 measured, against a declared 66** — because nothing increments it (grep: the key exists only in `c09guard.py:43` as an initialiser and in **C09's** arena at `c09_a0_arena.py:1944`) | **no** (was yes) | **ERRATUM 2** |
| 6 | `dev_or_test_labels_into_decision_quantities` | `0` | §12; the binding `H-L3`-class predicate | not directly measurable pre-run; by construction — `lab_dev` is written to disk and read by nothing in the decision path | **yes** | none |
| 7 | `banked_trainlog_opens` | `6` = `GATE-DEVFID` only, `2 ds × 3 seeds` | §12; `headspace_fidelity.floor_dev_curve` | not measurable pre-run (needs the mints); `c09guard.is_banked_trainlog` counts them | no (reported) | none |
| 8 | `processes_reporting` | **`74`** = `1 GATE-SHA + 66 mints + 6 fidelity + 1 arena` | §13's declared order **plus** the driver's own `--gate-sha-only` process | **74 measured** by the reviewer; I reproduced the driver leg: that one process writes **1** ledger file, `argv[-1] = --gate-sha-only`, `dev_path_opens = 2` | **yes** | **ERRATUM 2** |
| 9 | `predicate coverage` | re-derived in-job | §12 | `c09guard.verify_predicate` exists and is read-only | no (reported) | none |

**Which predicates could not be measured pre-run, and why their static derivation is trustworthy.**
Rows 2, 3, 6 and 7 cannot be measured without the 66 mints, which require the run. Their derivations
are trustworthy for different, stated reasons, and the distinction matters:

* **Row 3** (`66`) and **row 7** (`6`) are **counts of declared work**: `2 ds × 3 seeds × (5+1)` and
  `2 ds × 3 seeds`. They are the same two-factor form as row 4's second term and are checked at run
  time by counting files, not by trusting the literal.
* **Rows 2 and 6** are **by-construction zeros**, not counts. Row 2 rests on `is_test_like` raising
  rather than counting — a non-zero value is unreachable without an exception — and on §3.1's
  statement that no `test_seen` cache is opened by anything. Row 6 rests on `lab_dev` being written
  to disk (`headspace_mint.py:322-324`) and read by no decision path. **Neither is a measurement,
  and neither is claimed to be**; both are the class C09 handled in a `by_construction_zero` block
  with a narrative warrant, which is what row 5 is being demoted to.

**Row 5 is the third collision the completeness obligation exists to surface, and it is in this
table rather than in a future erratum.** It was declared a binding integer without an instrument
that can produce it. C09 did not make that mistake: it handled the same quantity as
`by_construction_zero` with a written warrant. C06 promoted it and built nothing.

**I searched for a fourth and did not find one.** Rows 1, 2, 6, 7 and 9 are unaffected by the
`GATE-SHA`-driver omission: the driver process opens no test path (measured `0`), materialises no
label, and reads no trainlog (measured — its ledger shows `banked_trainlog_opens` untouched). Rows
3 and 8 are the two the omission touches; both are repaired here. **That the search is stated does
not make it exhaustive, and I say so rather than claim closure.**

---

## 2. The settled process count

**`74`**, derived from the declared order plus the one process it omits, and confirmed by execution:

| component | n | source |
|---|---|---|
| `GATE-SHA` driver (`--gate-sha-only`) | **1** | `c06_falsifier_cpu.sbatch:63-64`. **Measured:** writes exactly **1** ledger file, `argv[-1] = --gate-sha-only`, `dev_path_opens = 2`, `test_path_opens = 0` |
| mints | 66 | §3.3 — 36 Head-N + 30 Head-R |
| fidelity | 6 | §6 `GATE-DEVFID` — 2 ds × 3 seeds |
| arena | 1 | the aggregator; adds itself as `len(procs) + 1` |
| **total** | **74** | |

**Was the declared `73` simply wrong? Yes, and precisely.** `66 + 6 + 1 = 73` is correct arithmetic
over the three *payload* phases; §13 describes the `GATE-SHA` pass as something *"the driver"* does
*"before any of them"*, which reads as an action of the shell rather than as a python process. It is
a python process: the sbatch invokes `c06_falsifier_arena.py --gate-sha-only`, and because
`PYTHONPATH` and `C09_LEDGER_DIR` are exported **before** that call, the frozen `sitecustomize`
installs the guard in it and its `atexit` flush writes a ledger the arena then counts.

**The corpus contained the contradiction twice, unnoticed** — this is worth recording because it is
the same failure mode as the archaeology in §4. Round 14 wrote *"`66 + 6 + 1 = 73`. I found no
seventy-fourth python process"* while quoting §13's clause that names `GATE-SHA` running before all
of them. Code-review round 1 then enumerated, under the heading *"73 processes in §13's order"*, the
list *"`1 × --gate-sha-only`, then 66 mints …, then 6 fidelity, then 1 arena"* — **74 items counted
as 73**, with the very next line noting that `--gate-sha-only` hashes 37 artifacts. Two independent
lineages held both halves and neither summed them.

**And the precedent v1 quoted already counts this class of process.**
`C09_A0_DECISION.json::GATE_LEDGER.evidence.n_processes_expected_breakdown` reads *"**1
version/preflight heredoc** + 36 mints + 2 GATE-DEVFID runs"*. The answer to the collision v1 left
behind was in the same JSON block v1 cited for the dev-side pattern.

**Not to be repaired by suppressing the driver's ledger.** Unsetting `C09_LEDGER_DIR` for the
`--gate-sha-only` call would make the count pass by discarding the counts of the process performing
the largest new read surface — option (ii)'s laundering under a different name.

---

## 3. `expected_sha_dev_opens` — derived over the right iterable, and audited rather than declared

**First factor — dev-like files, derived over the same concatenated iterable `gate_sha` consumes**
(adopting I-2). `gate_sha` iterates `frozen_sha256` **+** `frozen_sha256_input_caches`. Measured:

```
files in the iterable gate_sha hashes: 21
dev-like among them: 2
   data/CLIP_Embedding/HateMM/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt
   data/CLIP_Embedding/MHC_zh/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt
(subset frozen_sha256_input_caches alone: 8 files, 2 dev-like)
```

The two agree **today**, because none of the 13 module/config paths is dev-like — but v1's whole
warrant for this factor was that it *"cannot drift from the digest table"*, and a derivation over a
strict subset of what the loop hashes can drift from it. Deriving over the same iterable costs
nothing and makes the claim true.

**Second factor — audited, not declared** (adopting H-2's repair). v1 called `2` statically derived
from §13; it is not — §6 and §13 both say **once**, and the `2` is a property of
`c06_falsifier_arena.py:1276`, where `main()` calls `gate_sha()` again. Under this proposal the
arena **measures** it:

```
gate_sha_processes = 1 + #{ledger files whose argv contains --gate-sha-only}
```

and **HALTs if it differs from the declared `2`**. The aggregated ledger already carries each
process's `argv`, so this is ~3 lines, needs no edit to the sha-frozen `c09guard.py`, and converts
the last literal into a cross-checked measurement — which is the principle CODE-R1's H-2 was raised
to enforce.

**And §6 and §13 must be amended in this same erratum** so all four sections agree. Repairing a
§11/§6/§12 contradiction by creating a §12/§6/§13 one is the wrong trade, especially in a document
whose defect record is *"the predicate was frozen and never re-derived after the sections it depends
on changed."*

### Option (iv), considered explicitly and declined — with the reason recorded (I-1)

**Make `GATE-SHA` actually run once, as §6 and §13 say**, by dropping the arena's second pass. The
dev term would become `mints_executed + 2` and §6/§13 would be true as written.

**Declined, and the reason belongs in §6 rather than being left implicit.** The arena runs roughly
an hour after the driver's pass. Re-hashing at the point of use closes a time-of-check/time-of-use
window on the frozen inputs that a single up-front pass leaves open. That is a real provenance
benefit and it argues for keeping two passes — but it is currently an **accident of implementation**
that §6's *"once in the sbatch driver"* actively contradicts. This proposal therefore keeps two
passes **and writes the reason into §6**, so the second pass becomes a design decision rather than a
side effect.

---

## 4. The defect and the archaeology (carried from v1, with round 3 added)

**The defect.** §12's `dev_path_opens == mints_executed + 0` collides with `GATE-SHA`'s coverage of
§11's input caches, two of which are `dev_seen_*.pt`. Hashing opens; `c09guard.is_dev_like` counts;
`GATE-SHA` runs in two processes. Measured `+2` per pass, `+4` total. Two design repairs, each
individually reviewed and correct, jointly unsatisfiable.

| when | what happened |
|---|---|
| **v1 draft** | §11's input-cache table is headed *"train split only"*, eight `train_*` rows, **zero** `dev_seen`. No collision. |
| **v2** | The two `dev_seen` digests enter §11; `GATE-SHA`'s row already reads *"every frozen import and input cache"*; §3.1 already carries the *"covered by `GATE-SHA`"* warrant. **Latent from here.** |
| **round 3** *(added — the reviewer noted v1 skipped it)* | `dev_path_opens` first appears, in H-2 correcting the mint count from 36 to 66. Does not change the account. |
| **round 4, I-5** | *"§12's `dev_path_opens` is binding with an unquantified term. Measured, that term is `0`."* → *"Write the term: `dev_path_opens == mints_executed + 0`."* **Frozen here.** |
| **round 8, H-1** | Widens `GATE-SHA`'s row to name the sixteen banked artifacts. Did not add the dev caches — already in via *"input cache"* — but re-affirmed the breadth. §12 not re-derived. |
| **rounds 6–9** | `dev_path_opens`: **zero** mentions in each. |
| **rounds 10–15** | 1–2 mentions each, all verifying the fidelity-side clause, which is **true**. Round 14: *"the `dev_path_opens == mints_executed` choice is correct and would otherwise HALT a legitimate resume"* — the resume warrant re-checked, the second term never re-derived. |

**Why fifteen rounds missed it:** every round that examined the `+0` verified the *warrant offered
for it* and never re-derived *the term itself*. The complementary question — does any **other**
process open a dev-like path? — was never asked. `GATE-SHA` does, and had since v2.

---

## 5. Options (ii) and (iii) — rejections carried, with the reviewer's added ground

**(ii) hash through a non-counted path: rejected.** It makes the ledger blind to a real open,
generalises to "any inconvenient count can be bypassed by choosing a different I/O primitive", and
blinds the guard to `dev_seen_*-ro_*` — which §3.1 guarantees is never opened. **The reviewer's
added ground, adopted:** `_guarded_open` is the only thing that *raises* on a test path, so
bypassing it for dev-side convenience removes the hard stop on the test side of the same function.

**(iii) drop the two dev caches from `GATE-SHA`: rejected.** §3.1 states the native `dev_seen`
*"is opened by `headspace_mint.py:199` on every mint and is covered by `GATE-SHA`"*. Dropping the
digests falsifies that sentence and leaves a file entering 66 processes unverified. The campaign's
standing direction on `GATE-SHA` has been to widen, never narrow.

---

## 6. Recommendation

**Adopt option (i), widened to reconcile §12 in full**, under all eight obligations of the review's
closing section. The deciding argument is unchanged and was endorsed: a ledger predicate is an
**audit**, and §12's current form does not forbid a leak — it forbids a **lawful read the design
itself mandates**. The house pattern, confirmed at source in `C09_A0_DECISION.json`, is
declare-and-expect with the decomposition written down; and that same block counts its own
driver-side helper process, which is the fix for the second collision.

---

## 7. Implementation delta

| file | change | size |
|---|---|---|
| `c06_falsifier_arena.py` | `gate_ledger`: derive `expected_sha_dev_opens` over the concatenated iterable; **audit** `gate_sha_processes` from ledger `argv` and HALT on mismatch; change `processes_reporting` to `74`; demote `dev_label_materialisations_outside_decisions` to reported with the by-construction warrant recorded on the face | **≈ 25 lines**, one method |
| `configs/c06/c06_falsifier.json` | `processes_reporting.expected` `73 → 74` with the four-part decomposition; `dev_path_opens.expected` → the two-term formula; `dev_label_materialisations_outside_decisions.binding` `true → false` with the warrant; `gate_sha_processes: 2` **declared for the audit to check against**, not to be trusted | **≈ 12 lines** |
| `c06_falsifier_cpu.sbatch` | **obligation 7 only**: count executed-vs-skipped mints and export `C06_MINTS_EXECUTED` — currently a grep-verified single hit at `c06_falsifier_arena.py:1419` with no exporter, so the first term is `== 66` in disguise and **will HALT a legitimate resume**, the exact failure §12 spends four sentences saying it avoided | **≈ 6 lines** |
| V15E2 | §12's rows 4, 5, 8; §13's *"73"* → *"74"* with the decomposition; §6's and §13's *"once"* → *"twice, and why"*; §9's process-count statements | text only |

**Line citations refreshed to the hashed files** (M-1): the driver's `GATE-SHA` call is
`c06_falsifier_cpu.sbatch:63-64` (v1 cited CODE-R1's stale `:57`); the arena's second pass is
`c06_falsifier_arena.py:1276` (not `:1038`).

**M-2 recorded and referred, not fixed here:** `gate_sha` **existence-checks** the sixteen banked
artifacts rather than hashing them (`os.path.exists`, no digest computed), so §6's *"matches its §11
digest"* is met only for presence. The config already declares this openly
(`_gate_sha_count.banked_unhashed_artifacts: 16`, *"whose presence GATE-SHA asserts"*). It does not
affect this erratum's arithmetic — none of the sixteen is dev-like, verified — but a reader deriving
the dev-like count from §6's description rather than from the config would be reasoning about a
scope the code does not implement. **This is code-lineage territory and is flagged, not silently
absorbed.**

**M-3 recorded:** the ledger is blind to anything a forked child opens — `DataLoader` workers and
`multiprocessing` children exit via `os._exit` and skip `atexit`. `GATE-LEDGER`'s guarantee should
be stated as covering **top-level processes only**, and that sentence belongs in §12.

---

## 8. Blindness and no-edit statement

**No battery-arm accuracy or macro-F1 was computed on any ro-derived arm.** `deployed_vote` called
zero times, no arm built, no mint read or run, no GPU, no job, no commit, no `TARGET_STATE.json`
edit. Compute: file and review reads; one dev-like enumeration over the config's digest tables under
an active `c09guard`; one `--gate-sha-only` invocation writing to a scratchpad ledger dir; greps for
the H-3 and I-3 claims. All scratch artifacts live in the session scratchpad;
`artifacts/c06_falsifier/` was never created.

**Nothing is edited for this defect.** `C06_FALSIFIER_PREREG_DRAFT_V15E1.md` (`8cde58aa…`),
`c06_falsifier_arena.py` (`0cdfd4f0…`), `c06_falsifier_mint.py` (`98f7b4a6…`),
`configs/c06/c06_falsifier.json` (`e2678431…`) and `c06_falsifier_cpu.sbatch` (`c3647173…`) all
carry their post-CODE-R1 hashes. The arena still implements `dev_path_opens == mints_executed + 0`
exactly as frozen and fails with `ERRATUM REQUIRED` in the message. **Corrected from v1: the battery
cannot pass `GATE-LEDGER` before this erratum lands, and — under v1 as specified — could not have
passed after it either.**
