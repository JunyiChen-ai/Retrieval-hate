# C04-A0T-SMALL-v1 v6 CPU-Preflight Unlock Review

Date: 2026-07-30  
Reviewer: independent static reviewer (Opus, fresh context per round, read-only)  
Rounds: 4  
Final verdict: **GO (0 Critical / 0 High / 0 Important)**.

This file is an executor-side immutable transcription of the independent unlock
reviewers' returned verdicts, mirroring
`C04_A0T_SMALL_V1_V6_CODE_RESOURCE_REVIEW.md`. Each round was run by a fresh
reviewer that had not seen the implementer's reasoning, under a hard boundary:
no Python run or imported, no SLURM job submitted/held/released/cancelled
(`sacct`/`squeue` read-only only), no dataset label value, video or model weight
opened, and no file under `/data/jehc223/RGCL` created, modified or deleted.
Working copies for reconstruction were made outside the repository.

## Verdict history

| Round | Verdict | Findings |
|---|---|---|
| 1 | `GO (0C / 0H / 4I)` | stale `implementation_record_sha256` in the manifest; unbound `maps.*` geometry declarations; four dead imports in `preflight.py`; the reviewed `preflight.py` revision was not retained |
| 2 | `REVISE (0C / 1H / 2I)` | **H1** the documented reconstruction recipe was false against the frozen bytes; **I1** "untested surface is exactly the write phase" understated it; **I2** the gates were not run "in the order `main()` calls them" |
| 3 | `GO (0C / 0H / 3I)` | `__pycache__` disclosure factually wrong; manifest pins the post-correction record rather than the code/resource-reviewed one; no unlock-review transcription existed |
| 4 | **`GO (0C / 0H / 0I)`** | none — clean |

Round 2's High is the one that matters for this campaign's discipline. The
record claimed the rounds-1-5 revision could be reconstructed by deleting the
`implementation_authorized` gate block alone. That was true when written and
false once the four unused imports were removed: applied to the frozen bytes it
yields `4d4dd033929560e923394cb704421cb50133bf1463e9027c0132cf79f75b5ebf`, a
revision that never existed. A reconstruction claim a reviewer cannot reproduce
is the same class of unverified static assertion that put job 13805 in the
queue, so it was corrected with an explicit erratum rather than overwritten.

## Round-3 verdict, as returned

`GO (0 Critical / 0 High / 3 Important)`.

The reviewer recomputed 25 distinct objects, all matching: the authorized
config, the normalized config contract reproduced independently via `jq`, the
authority manifest and its `closure_sha256`, the unlock record, the
implementation record, the code/resource review and request, the entrypoint
sbatch and wrapper, the frozen `preflight.py`, all 15 implementation hashes, all
15 frozen-design hashes, the design GO, both train ASR files, and **both**
model tree hashes — the processor tree recomputed from the six files directly,
and the model tree using the HuggingFace content-addressed blob filenames as the
safetensors digests, so both were verified without opening a weight byte.

The four-field revert reproduced the reviewed pre-authority config
`98f2ca60…` byte for byte with exactly four changed lines and no fifth
difference. The corrected two-step reconstruction recipe was executed literally
and reproduced both `7c64ddf6…` and `8f7dcd44…` as written.

On finding 8 — whether any gate would fail again — the reviewer answered **no**,
from its own static reading, and named the line that killed 13805 as now
passing: with all four keys at the sentinel, `freeze_stage` true and
`preflight_materialization_authorized` true, `resolve_prompt_hashes` returns
`SENTINEL_PENDING_CPU_PREFLIGHT_FREEZE` instead of raising. It further verified
that all 1323 train IDs across both datasets have symlinks resolving to regular
files inside the pinned physical roots (0 dangling, 0 escaping), that neither
ASR file contains a surrogate escape that could break canonical encoding, that
the partition's `TIMELIMIT` is `infinite` so omitting `--time` is safe, that all
15 staged paths are namespace-contained, and that `namespace.parent.mkdir` is
the first write and comes after every check, so a HALT anywhere earlier leaves
zero filesystem residue.

## The three Importants, and their closures

1. **`__pycache__` disclosure was wrong.** The implementation record asserted the
   `.pyc` files written by the read-only login-node validation "are never read by
   the SLURM job". The reviewer read all four `.pyc` headers and found flags`=0`
   with embedded source mtime/size exactly matching the current sources, so
   CPython would have loaded that bytecode — a layer outside every hash closure.
   The reviewer rated it Important rather than High because the caches are
   provably derived from the pinned bytes and CPython's mtime+size check is
   fail-closed. **Closed** by deleting
   `scripts/analysis/__pycache__/c04_a0t_small_v1_v6_*.pyc` before submission, so
   the job compiles from the hash-verified sources. The `.pyc` files are in no
   hash map, so their removal moved no hash: config `40ec6d97…`, manifest
   `5e56041a…` and record `1b2d0bef…` are unchanged.
2. **The manifest pins the post-correction record.**
   `reviewer.implementation_record_sha256=1b2d0bef…` is the record as corrected
   in response to unlock rounds 1 and 2, not the revision the five code/resource
   rounds bound, which is
   `208141759d691cf5768eb3195dc4a5d9e0d7c399b9a89d7a4ae1195b576b1862` and is
   pinned inside the code/resource review request. **Closed** by stating this in
   the unlock record. Per the reviewer's own instruction, the manifest was *not*
   rebuilt for it: rebuilding would churn closure → config pin for no safety
   gain, and the code/resource-reviewed revision token is to be added at the next
   legitimate rebuild.
3. **No unlock-review transcription existed.** **Closed** by this file. Nothing
   pins it, so it costs no rebuild.

## Round-4 verdict, as returned

`GO (0 Critical / 0 High / 0 Important)`. "Nothing survived verification."

The reviewer recomputed 31 objects, all matching, and re-derived both model tree
hashes without opening a weight byte. It executed both claims that would have
been easiest to fake and both reproduced exactly: the two-step `preflight.py`
reconstruction (`c86d439c…` → re-insert four imports → `7c64ddf6…` → delete the
gate block → `8f7dcd44…`) and the four-field config revert to `98f2ca60…` with
exactly four changed lines. It also reproduced the erratum's phantom
`4d4dd033…`, confirming the erratum is truthful.

It recorded three items it considered and deliberately did not file: the
implementation record's superseded `.pyc` sentence (now vacuously true, and
quoted and refuted in both companion documents), the manifest's
`implementation_record_sha256` provenance (disclosed, and the reviewed revision
`20814175…` is genuinely pinned inside the review request), and the
documentation-only `maps.*` geometry declarations (checked against the module
constants and found to agree).

On finding 8 it answered **no** — no gate on the CPU-preflight path fails —
naming the killed line as now passing, and confirming `slurm/logs/` exists, the
partition's `TIMELIMIT` is `infinite`, `jsonschema` 4.26.0 exports
`Draft7Validator`, and the post-review gate strictly tightens and sits 321 lines
before the first filesystem write. It named the sole residual as the *content*
of the five safetensors blobs, corroborated by size plus content-addressed blob
name but not re-hashed.

## Authority boundary

This GO unlocks exactly one submission of
`scripts/slurm/c04_a0t_small_v1_v6_preflight.sbatch`. It does not authorize
teacher inference, any GPU allocation, the small tranche, reconciliation,
dev/test evaluation, OCR, external API or network access, label access, chained
submission, release, resubmission, or reuse of any artifact namespace. After the
preflight terminates, an independent collector/reviewer must inspect the frozen
artifacts and issue a fresh payload-review verdict; no GPU or downstream stage
becomes authorized merely because the CPU preflight succeeds.
