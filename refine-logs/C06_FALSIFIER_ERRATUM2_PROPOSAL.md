# C06 `$0` falsifier — **ERRATUM 2, PROPOSAL**

*Against:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` (v15 + Erratum 1 + the CODE-R1 §8
correction), sha256 `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`.
*Raised by:* the implementation lineage, on wiring `GATE-LEDGER` for code-review round 1's **H-2**.
*Adjudicated as erratum territory by:* the code lineage (`C06_FALSIFIER_CODE_REVIEW_R1.md` H-2,
*"This is a design question, not a code choice, and it should go back to the design lineage as an
erratum rather than be decided here"*) and confirmed by the team lead.
*Status:* **PROPOSAL. Nothing is landed.** The design document, the arena and the sbatch are
**unedited** for this defect. A fresh independent reviewer adjudicates; the erratum lands only on
their GO, per the Erratum-1 precedent (proposal → independent review → landed erratum).

---

## 1. The defect, precisely

**§12's binding ledger predicate is unsatisfiable on a clean run.** It reads:

| predicate | expected | binding? |
|---|---|---|
| `dev_path_opens` | **`mints_executed + 0`** — round-4 I-5: `headspace_fidelity.py` opens **no** `dev_seen` file, reading `lab_dev` out of the banked mint `.npz` (`:66`), so the second term is zero, not free | yes |

**`GATE-SHA` opens dev-split files, lawfully, and the ledger counts them.** §11's input-cache table
carries **eight** caches, of which **two are `dev_seen_*.pt`** (the HateMM and MHC-ZH native dev
caches). §6's `GATE-SHA` row covers *"every frozen import, **input cache** and the sixteen banked
artifacts of §11"*. Hashing a file opens it; `c06_falsifier_arena.sha256_of` uses `builtins.open`,
which `c09guard._guarded_open` intercepts, and `c09guard.is_dev_like` counts any basename
containing `dev_seen`.

**Measured, not inferred** (login node, real config, real caches, `c09guard.install()` active):

```
   dev_seen entries in the GATE-SHA digest list: 2
   dev_path_opens incremented by GATE-SHA hashing: 2   (per GATE-SHA-running process)
   GATE-SHA runs in 2 processes (driver --gate-sha-only + arena) -> total +4
   => on a CLEAN run the measured value is mints_executed + 4  -> PREDICATE FAILS
```

**Two design repairs, each individually reviewed and correct, are jointly unsatisfiable.** Nothing
here is a mistake in either repair: `GATE-SHA` *should* cover the dev caches (§3.1 relies on it by
name), and round-4 I-5 *was* right that `headspace_fidelity.py` contributes zero. The defect is the
conjunction.

### Archaeology — which rounds collided, and why fifteen missed it

| when | what happened |
|---|---|
| **v1** | §11's input-cache table carries **zero** `dev_seen` rows. No collision exists. |
| **v2** (after round 1) | The two native `dev_seen_*.pt` digests **enter §11's input-cache table**. `GATE-SHA`'s row already reads *"every frozen import and input cache"*, so from this version the gate is specified to open dev-split files. **The collision becomes latent here.** |
| **round 4, I-5** | Finds §12's `dev_path_opens` *"binding with an unquantified term"* — v4 wrote the second term as *"`mints_executed` + `GATE-DEVFID` reads"*. Measures that term at **0** and prescribes: *"Write the term: `dev_path_opens == mints_executed + 0`."* **The collision becomes a frozen, binding predicate here.** |
| **round 8, H-1** | Widens `GATE-SHA`'s **row** to name the sixteen banked artifacts, because §11 asserted a scope §6 did not carry. This did not add the dev caches — they were already in via *"input cache"* — but it is the round that re-affirmed the gate's breadth. §12 was not re-derived. |
| **rounds 6, 7, 8, 9** | `dev_path_opens` is **not mentioned at all** — zero occurrences in each review. |
| **rounds 10–15** | Mention it once or twice each, always to verify the clause that *justifies* the `+0`: that `headspace_fidelity.py` opens no `dev_seen` file. **That clause is true.** Round 14's C.9 checks it at source and confirms it. |

**Why no round caught it, stated plainly for the record: every round that examined the `+0` term
verified the warrant offered for it, and never re-derived the term itself.** The warrant — *"the
fidelity processes contribute zero"* — is correct and was confirmed repeatedly at source. The
question no round asked is the complementary one: *does any **other** process open a dev-like path?*
`GATE-SHA` does, and had done since v2. **The predicate was frozen in round 4 and never
re-derived after §11's table or §6's `GATE-SHA` row changed** — and both changed after it was
frozen. This is the same shape as Erratum 1: a quantity verified through the clause that motivated
it rather than against the artifact that has to satisfy it.

It is also, precisely, the failure the separate code lineage exists to catch: the collision is
invisible from the design document, where §11, §6 and §12 each read correctly on their own, and
becomes visible the moment a process actually opens a file and something counts it.

---

## 2. The three options, code-anchored

### Option (i) — the predicate expects the lawful `GATE-SHA` opens

`dev_path_opens == mints_executed + expected_sha_dev_opens`, where the second term is **derived
statically from the frozen tables**, not measured at run time:

```
expected_sha_dev_opens = (# dev-like files in §11's GATE-SHA scope)
                       × (# processes that run GATE-SHA)
                       = 2 × 2 = 4
```

* **`2` dev-like files.** §11's input-cache table has eight entries; exactly two have basenames
  matching `c09guard.is_dev_like` — `dev_seen_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` (HateMM)
  and `dev_seen_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` (MHC-ZH). Both are already in
  `configs/c06/c06_falsifier.json::frozen_sha256_input_caches`, so the count is readable from the
  sha-gated config rather than hard-coded.
* **`2` processes.** §13's order is *"66 mints → 6 fidelity → 1 arena"* with *"`GATE-SHA` once in
  the sbatch driver before any of them"*. The driver's call is `--gate-sha-only`
  (`c06_falsifier_cpu.sbatch:57`); the arena runs `gate_sha()` again on its own main path. The 66
  mints do **not** — `c06_falsifier_mint.assert_frozen()` hashes `headspace_mint.py` and
  `mechnov_pairverify.py` only. The 6 fidelity processes do not hash anything.

**The C09 precedent has exactly this shape, and it is the house pattern.** `C09_A0_V17_RECORD.md`
§1549-1553 specifies its own `GATE-LEDGER` as:

> dev-split **path** opens (**expected nonzero** — 36 mint loads of `dev_seen_*.pt` plus 6
> banked-trainlog reads — **reported with its declared expected value**)

and §378-379 records *"its **declared expected dev-label materialisation count is 36**, one per
mint, all outside every decision quantity."* C09 did not forbid its lawful dev reads; it **declared
and expected** them, with the decomposition written down. C06's own §12 already does this for
`banked_trainlog_opens` (`GATE-DEVFID` only, `2 × 3`) — option (i) applies the same treatment to the
one term where a second lawful contributor was missed.

**Direction.** Neutral on the verdict: it changes an instrument predicate from unsatisfiable to
satisfiable-when-clean. It does **not** weaken the audit — the binding predicates that carry the
test-split guarantee (`test_path_opens == 0`, `test_label_materialisations == 0`,
`dev_or_test_labels_into_decision_quantities == 0`) are untouched, and `dev_path_opens` remains
**binding** with an exact expected integer, so an unexpected dev open still HALTs.

### Option (ii) — hash the dev caches through a non-counted path

Use `c09guard._ORIG_OPEN`, `io.FileIO` or `os.open` inside `sha256_of` so the read is not seen by
`_guarded_open`.

**Analysis: this launders the audit, and I recommend against it.** Three grounds.

1. **It makes the ledger blind to a real open.** The ledger's stated purpose (`c09guard.py:28-30`)
   is that *"the arena aggregates them into `GATE-LEDGER`, so the ledger reports **MEASURED** opens
   rather than literals."* An open that happens but is not counted converts the ledger back into a
   literal for that file class — the exact defect code-review H-2 raised.
2. **It generalises badly.** Once a count can be made to pass by choosing a different I/O
   primitive, every inconvenient count has that escape, and no future reader can tell a
   "doesn't-count" open from an absent one.
3. **The audit is directional.** `is_dev_like` also matches `dev_seen_*-ro_*` — files §3.1 says
   *"no `dev_seen_*-ro_*` file is opened by any phase"*. A `sha256_of` that is invisible to the
   guard would be invisible for those too, weakening a guarantee the design states in terms.

**My prior and the team lead's coincide here, and the code-anchored reading confirms both: the
ledger should COUNT the opens and the predicate should EXPECT them.**

### Option (iii) — drop the two dev caches from `GATE-SHA`'s scope

**Analysis: this reopens a provenance hole the design names, and I recommend against it.**

§3.1 states, in terms: *"the native `dev_seen` is opened by `headspace_mint.py:199` on **every**
mint and **is covered by `GATE-SHA`**"*. That sentence is the design's warrant that the dev cache
every one of the 66 mints loads is the file whose digest is frozen. Dropping the two entries would
make it false and would leave a file that enters all 66 mint processes unverified.

Round-8 H-1's reason for widening the gate was that §11 asserted a coverage §6 did not carry, which
left `GATE-FLOOR`'s anchors and `GATE-FOLD`'s parity files unverified. That was about the sixteen
banked artifacts, not these two — but the principle it established runs directly against (iii):
**the campaign's standing direction on `GATE-SHA` has been to widen coverage, never to narrow it**,
and narrowing it to make a bookkeeping predicate pass inverts that for the worst reason.

If the design lineage nonetheless prefers (iii), it must re-examine §3.1's sentence and say what
replaces the guarantee — which is more text and more risk than (i)'s one integer.

---

## 3. Recommendation

**Adopt option (i).** `dev_path_opens == mints_executed + expected_sha_dev_opens`, with
`expected_sha_dev_opens = 4` derived statically as `2 dev-like files × 2 GATE-SHA processes`.

**The deciding argument.** A ledger predicate is an **audit**, and an audit's job is to state what
the run should do and HALT when it does not. §12's current form does not forbid a leak — it forbids
a **lawful read that the design itself mandates** in §3.1 and §6. Making the predicate expect the
reality it audits keeps every guarantee that matters (test opens `0`, dev-or-test labels into
decision quantities `0`, an exact expected integer for dev opens so an *unexpected* one still HALTs)
while removing a contradiction between three sections that are each individually right. **The C09
precedent settles the house pattern: declare and expect lawful dev reads with the decomposition
written down; do not forbid them.**

Two supporting points:

1. **It is the only option that touches one integer.** (ii) changes an I/O primitive and weakens the
   instrument; (iii) changes `GATE-SHA`'s scope and falsifies §3.1's sentence.
2. **It is verifiable statically.** Both factors are readable from frozen artifacts — the dev-like
   count from the sha-gated config's own digest table, the process count from §13's declared
   73-process order — so the expectation is derived, not measured-then-blessed. That distinction is
   what keeps it an audit rather than a rubber stamp.

**One thing I want the reviewer to push on.** I have assumed the mint's `dev_seen` load contributes
**exactly one** count per executed mint. `headspace_mint.py:199` calls `load_split`, which calls
`_ORIG_TORCH_LOAD`; whether `torch.load` performs one `builtins.open` or more on this PyTorch build
is **not something I measured** — I measured only the `GATE-SHA` term. If `torch.load` opens the
file more than once, `mints_executed` is the wrong multiplier and the first term needs the same
static derivation treatment as the second. **That is the one number in this proposal I would most
like independently measured before it lands.**

---

## 4. Implementation delta for option (i)

### 4.1 Code — bounded

| file | change | size |
|---|---|---|
| `c06_falsifier_arena.py` | In `gate_ledger`: derive `expected_sha_dev_opens` from the config's own digest table and the declared `GATE-SHA` process count, and compare against `mints_executed + expected_sha_dev_opens`. **The failure-message decomposition already written for CODE-R1 H-2 becomes the expectation derivation** — same two factors, same provenance, moved from the error path to the predicate. Record both factors and the product in `self.ledger`. | **≈ 15 lines**, one method |
| `configs/c06/c06_falsifier.json` | `ledger.dev_path_opens.expected` → `"mints_executed + expected_sha_dev_opens"`, plus `gate_sha_processes: 2` with its §13 provenance. The dev-like file count is **not** added as a literal — it is counted from `frozen_sha256_input_caches`, so it cannot drift from the digest table. | **≈ 6 lines** |
| `c06_falsifier_cpu.sbatch` | **none** | 0 |

No gate is added or removed, no verdict path changes, no §8 row moves, and the H-2 wiring already
landed (measured aggregation, `c09guard._INSTALLED` assertion, every other predicate as a
pass-condition) is correct as it stands — **this erratum changes one comparison.**

### 4.2 The exact §12 text edit for a V15E2

Replace the `dev_path_opens` row and add one paragraph beneath the table:

> | `dev_path_opens` | **`mints_executed + expected_sha_dev_opens`**, where
> `expected_sha_dev_opens = (dev-like files in §11's GATE-SHA scope) × (processes running GATE-SHA)
> = 2 × 2 = 4` | yes |
>
> **Why the second term is not zero (ERRATUM 2).** Round-4 I-5 fixed this term at `0` on the
> measured fact that `headspace_fidelity.py` opens no `dev_seen` file — which is true and remains
> true. It is not the only contributor. §11's input-cache table carries **two** `dev_seen_*.pt`
> caches and §6's `GATE-SHA` row covers *"every frozen import, input cache and the sixteen banked
> artifacts"*; hashing a file opens it, and `c09guard.is_dev_like` counts it. `GATE-SHA` runs in
> **two** processes — the sbatch driver's `--gate-sha-only` call and the arena — so a clean run
> measures `mints_executed + 4`. Measured `+2` per `GATE-SHA` process. The predicate stays
> **binding** with an exact expected integer, so an unexpected dev open still HALTs; what changes is
> that a **lawful** read the design mandates at §3.1 (*"the native `dev_seen` … is covered by
> `GATE-SHA`"*) is expected rather than forbidden. This follows C09's own `GATE-LEDGER`, which
> declares its dev opens *"expected nonzero … reported with its declared expected value"*
> (`C09_A0_V17_RECORD.md:1549-1553`).

§3.1, §6 and §11 are **unchanged** — the erratum makes §12 agree with them, not the reverse.

---

## 5. Blindness and no-edit statement

**No battery-arm accuracy or macro-F1 was computed on any ro-derived arm in producing this
proposal.** The work was: reading four frozen sources and eleven review files; one measurement that
hashed two dev caches under an active `c09guard` and read the counter; and greps over the draft and
review corpus for the archaeology. No mint was read, no head-space arm built, `deployed_vote` called
zero times, no GPU, no job, no commit, no `TARGET_STATE.json` edit.

**Nothing is edited for this defect.** `C06_FALSIFIER_PREREG_DRAFT_V15E1.md`,
`c06_falsifier_arena.py`, `configs/c06/c06_falsifier.json` and `c06_falsifier_cpu.sbatch` carry
their post-CODE-R1 hashes unchanged. The H-2 ledger wiring **is** already in place and is correct in
every respect except this one predicate: the arena aggregates `c09guard`'s measured counts, asserts
`_INSTALLED` in every process, evaluates each §12 predicate as a pass-condition, and publishes
measured counts on the verdict face. It currently implements `dev_path_opens == mints_executed + 0`
**exactly as frozen**, so it fails on a clean run with a message carrying the decomposition and the
words `ERRATUM REQUIRED`. **The battery cannot pass `GATE-LEDGER` until this erratum lands**, and it
was left that way deliberately rather than adjusted to pass.
