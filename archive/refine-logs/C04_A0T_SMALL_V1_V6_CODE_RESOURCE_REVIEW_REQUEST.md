# C04-A0T-SMALL-v1 Implementation-v6 Code/Resource Review Request

Date: 2026-07-30  
Requested verdict form: `GO | REVISE | KILL` with exact
`N Critical / N High / N Important` counts.  
Review mode: **static, read-only**.

## Why v6 exists

Implementation-v5's CPU preflight could not pass its own first static gate.
SLURM job `13805` (`c04_a0t_small_v1_v5_preflight`) was submitted
`2026-07-30T08:48:50`, held as `JobHeldUser`, auto-released and both started and
ended `2026-07-30T16:40:57`, terminating `FAILED` `1:0` with `00:00:00` elapsed,
0 bytes of stdout and 1191 bytes of stderr. The primary log is
`slurm/logs/c04_a0t_small_v1_v5_preflight_13805.err`.

`c04_a0t_small_v1_v5_preflight.py:152` asserts
`prompt_hashes() == cfg["prompt_hashes"]`, while
`configs/c04/c04_a0t_small_v1_v5.json:115-120` holds
`PENDING_CPU_PREFLIGHT_HASH_FREEZE` for all four keys — and computing those four
hashes is what the CPU preflight exists to do. The v5 producer asserts the same
equality at line 177.

No metric, result, decision or CONTINUE/KILL verdict was published by 13805.
`artifacts/c04/` does not exist.

## Exact snapshot to review

This request is **revision 5**. Revision 1: independent
`REVISE (0C / 2H / 2I)`. Revision 2: independent `REVISE (0C / 1H / 2I)`.
Revision 3: independent `REVISE (0C / 1H / 2I)`. Revision 4: independent
`GO (0C / 0H / 1I)`. All eleven findings and the changes made in response are
listed in the next section; verify those closures yourself rather than
accepting them. Only the implementation record changed since revision 4; no
code, schema, wrapper, sbatch or config byte moved.

| Artifact | SHA-256 |
|---|---|
| `configs/c04/c04_a0t_small_v1_v6.json` | `98f2ca603538a22635904c299fa8623352dc516d003da430dcaf642336bdbd94` |
| `refine-logs/C04_A0T_SMALL_V1_V6_IMPLEMENTATION_RECORD.md` | `208141759d691cf5768eb3195dc4a5d9e0d7c399b9a89d7a4ae1195b576b1862` |
| `scripts/analysis/c04_a0t_small_v1_v6_common.py` | `81b10f586cfa5d619db459505ba2c8c43a89fc50e0c1e978e500fb4932633f68` |
| `scripts/analysis/c04_a0t_small_v1_v6_preflight.py` | `8f7dcd44785126a82ba52fe2be4e3c61e4b6f771eb14a839bd016d13faf70111` |
| `scripts/analysis/c04_a0t_small_v1_v6_gpu_ledger.py` | `960cad30528c4bc0633cd103b33943c327ada13b12786c28d23552baadf2e695` |
| `scripts/analysis/c04_a0t_small_v1_v6_producer.py` | `ae02336047d60b8e8ae1f122859598b8b76d78093d07af4a6ba4222f34cfc478` |
| `scripts/wrappers/c04_a0t_small_v1_v6_preflight.sh` | `22bb4a47c6c21b06bb61b994ea873da682c3332bc0a9dec29c96cc0d1429f770` |
| `scripts/wrappers/c04_a0t_small_v1_v6.sh` | `7c8915733ad39288bcbd5aea9358376b869f95a8fc7845528778c9f716b8413c` |
| `scripts/wrappers/c04_a0t_small_v1_v6_reconcile.sh` | `d41f93e4e4c59f451cc1421b52acfb6eefcb5104c6bd74a1fa277f9322fd8f80` |
| `scripts/slurm/c04_a0t_small_v1_v6_preflight.sbatch` | `4051c73eeacf14ca302174c0448b36051685020aa589d8445b048469f46d665f` |
| `scripts/slurm/c04_a0t_small_v1_v6.sbatch` | `d42f9191352509fe2e4bff3e41b8a3b65f1b0e6f42879202a605750fb024b0d7` |
| `scripts/slurm/c04_a0t_small_v1_v6_reconcile.sbatch` | `c5e47308b323b60e07658e2e40e696677cbb937a6f85c68a54f201ded9c435e5` |
| `schemas/c04/c04_a0t_small_v1_v6_prompt_record.schema.json` | `007327448d3173936d5c0368a9ac83980753f5100fc2dbd3cef6ba0902768ff9` |
| `schemas/c04/c04_a0t_small_v1_v6_canonical_record.schema.json` | `7c05774a2e794137c5794bf7869be930766dbc7dfa7116b9cd6a213c1684bf9c` |
| `schemas/c04/c04_a0t_small_v1_v6_stage_authorization.schema.json` | `00f341b140332df3f0b1be7c02d455f894ff7098c0e11c7802f3e76fb3b77e00` |
| `schemas/c04/c04_a0t_small_v1_v6_payload_review.schema.json` | `0557760d97527b23b3fa26ababaaba36fd0eb35961fbaa0c11c31c505a027990` |
| `schemas/c04/c04_a0t_small_v1_v6_resource_final_state.schema.json` | `1274d0ad85125f72add92230c3bb8f71b25de41daf3c7c3ebf044e3951525231` |

Predecessors, for the minimality check (unmodified): the `..._v5_*` files with
config `5228c08fefcd41c354202dfd7a750b46e1a5d4d66a8b3562b736c9d943f526a6` and
record `aa49585cc27853793b349868bb6996ab5f051a44cfa1ff39c2fee0e7e1a704a1`.
Frozen design GO: `refine-logs/C04_V4_DESIGN_REVIEW.md`
(`340ae2c156e7acab8a19dcda9625f883058377ca618bdc4fd59177900738a854`).

## Prior verdict and claimed closures (verify independently)

Revision 1 returned `REVISE (0C / 2H / 2I)`:

- **H1** — `config_contract_sha256` hashed `prompt_hashes` verbatim, so the
  contract hash baked into the code/resource authorization manifest, the genesis
  GPU ledger and the resource ticket was computed over the *pre-freeze* config,
  while downstream stages require the *post-freeze* config. The two states were
  mutually unsatisfiable: leaving the config unfrozen would burn the single GPU
  allocation before the producer's HALT, and amending it would permanently
  invalidate the pinned manifest. **Claimed closure:** `config_contract_sha256`
  now also normalizes `prompt_hashes`, plus a self-test fixture asserting
  invariance across the freeze and non-invariance under tampering.
- **H2** — the GPU-ledger claim stage was not literal-bound, so an unfrozen
  config passed `claim` end-to-end and consumed the single-use ticket before any
  consumer rejected it. **Claimed closure:** new
  `assert_literal_prompt_hash_binding`, called at the top of both
  `validate_gpu_environment` and `validate_cpu_reconciliation_environment`.
- **I1** — the record's statement about the read-only validation was false for
  the snapshotted config, which HALTs on both paths because materialization is
  `false` pre-authority. **Claimed closure:** the record now distinguishes the
  pre-authority, post-authority and post-freeze config states explicitly.
- **I2** — the new manifest guard had no self-test fixture. **Claimed closure:**
  the guard was factored into `assert_literal_prompt_hashes` and given three
  fixtures.

Revision 2 returned `REVISE (0C / 1H / 2I)`:

- **H-1** — the H2 closure did not actually protect the allocation: the GPU
  wrapper arms its `EXIT` trap and writes the entry marker before the first
  Python call, so a claim-time HALT runs `--mode mark-exit`, which fell through
  and bumped the genesis ledger's revision and state even with an empty job
  list, permanently breaking the resource ticket's `genesis_gpu_ledger_sha256`
  pin inside a no-clobber namespace. **Claimed closure:** `mark_exit` returns
  early and leaves the ledger byte-identical when `ledger["jobs"]` is empty; the
  record's §3 claim was corrected to state exactly what the gate does and does
  not prevent, including that `mark-exit` is the one ledger mode with no
  binding gate.
- **I-1** — `maps.expected_hashes` is the only `PENDING_*` config value not
  normalized by `config_contract_sha256`, so a later amendment would wedge the
  run. **Claimed closure:** documented as lifecycle-immutable in both the config
  and the record; no code reads it and no amendment is needed, because
  `verify_payload_review` already binds `map_hashes` to `preflight["map_hashes"]`.
- **I-2** — the new `prompt_hash_freeze` manifest block carried five
  self-attested fields no verifier checked. **Claimed closure:**
  `verify_frozen_prompt_hashes` now asserts `path`, `sha256` against the file,
  `payload_sha256`, `keys`, and all three booleans.

Revision 3 returned `REVISE (0C / 1H / 2I)`:

- **H-A** — the record's read-only-validation section cited a config SHA-256
  (`181cdf9e…`) that matched nothing on disk and contradicted the record's own
  frozen table, i.e. the validation evidence named a different config revision
  than the one being authorized. **Claimed closure:** the dry validation was
  re-run against the final frozen bytes and the citation now names
  `configs/c04/c04_a0t_small_v1_v6.json` explicitly with its current file
  SHA-256. Every 64-hex string in the record was re-audited against disk.
- **I-A** — the `mark_exit` early return keyed on `not ledger["jobs"]`, which is
  false in the window where `claim` has published the allocation claim and the
  consumption record but not yet appended the job row, so an exit in that window
  would skip accounting v5 recorded. **Claimed closure:** the predicate is now
  `not ledger["jobs"] and not resource_consumption.exists()`; the record's
  absolute phrasing was corrected.
- **I-B** — `config_binding_at_freeze` was the one `prompt_hash_freeze` manifest
  field no verifier checked while the record called the block "fully verified".
  **Claimed closure:** `verify_frozen_prompt_hashes` now asserts it against the
  freeze artifact; the record enumerates all eight verified fields.

Revision 4 returned `GO (0C / 0H / 1I)`:

- **I-1** — the record's §3 enumerated the pre-gate entry marker without stating
  its blocking consequence: a claim-time HALT preserves the genesis ledger and
  the unconsumed ticket, but the job-id-and-uptime-pinned
  `allocation_entry_marker.json` still makes the namespace non-re-runnable
  without manual removal. **Claimed closure:** §3 now states this as a third
  boundary, identifies it as a v5 property carried forward in a byte-identical
  wrapper, and records that it must be closed or explicitly accepted before the
  GPU stage is authorized. Record-only change; no code byte moved.

Judge these closures on the current bytes. If a closure is incomplete, or if a
closure introduced a new defect, say so at the appropriate severity.

## Required findings

1. **Repair correctness.** The four prompt-hash keys may equal the pending
   sentinel only on the CPU-preflight freeze run and only when
   `authorization.preflight_materialization_authorized is True`. Any other
   value must HALT. A downstream, non-freeze consumer reading a config that
   still holds the sentinel must HALT. The frozen payload must carry literal
   hashes, and every downstream stage must compare against literal values.
2. **Minimality.** Every other guard, gate, authorization flag and scientific
   semantic must remain identical in meaning to v5. Report any widened
   permission, weakened check, removed HALT, or changed scientific constant as
   at least High.
3. **Self-test adequacy.** The self-test must actually fail closed for:
   sentinel accepted on freeze run; wrong value rejected; sentinel rejected on
   the non-freeze path; post-freeze payload carrying literal hashes.
4. **Bypass hunt.** Report any path by which the sentinel could reach a sealed
   record, a review manifest, a payload binding, or a GPU authorization.
5. **Resource/authorization surface.** The CPU-preflight sbatch must be CPU-only
   with no `--time`, no array, no dependency, no GPU request and no submission,
   release or resubmission path. All of teacher/GPU/Slurm-GPU/small-tranche/
   reconciliation/dev/test/OCR/API/network/cross-dataset/label/chain/release/
   resubmit authorizations must be false.

## Review boundary

Do not run, import or execute any Python. Do not submit, release or modify any
SLURM job. Do not open dataset labels, videos or model weights. Do not modify
any file. Hashing files read-only with `sha256sum` is expected.

A GO here permits preparation and independent review of a strict CPU-preflight
authority snapshot only. It does not by itself authorize the CPU preflight,
teacher generation, GPU execution, label access, or any scientific claim.
