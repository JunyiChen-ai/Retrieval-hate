# C04-A0T-SMALL-v1 Implementation-v6 Code/Resource Review

Date: 2026-07-30  
Reviewer: independent static reviewer (Opus, fresh context per round, read-only)  
Rounds: 5  
Final verdict: **GO (0 Critical / 0 High / 1 Important)**, the single Important
closed in place before the authority snapshot (see below).

This file is an executor-side immutable transcription of the independent
reviewers' returned verdicts. Each round was run by a fresh reviewer that had
not seen the implementer's reasoning, under a hard boundary: no Python run or
imported, no SLURM job submitted/held/released/cancelled (`sacct` read-only
only), no dataset label, video or model weight opened, and no file under
`/data/jehc223/RGCL` created, modified or deleted.

## Verdict history

| Round | Verdict | Findings |
|---|---|---|
| 1 | `REVISE (0C / 2H / 2I)` | H1 config-contract cycle; H2 GPU ledger not literal-bound; I1 false validation claim in record; I2 manifest guard untested |
| 2 | `REVISE (0C / 1H / 2I)` | H-1 `mark_exit` wedges the genesis ledger on any claim-time HALT; I-1 `maps.expected_hashes` undocumented contract trap; I-2 `prompt_hash_freeze` block self-attested |
| 3 | `REVISE (0C / 1H / 2I)` | H-A record cited a config SHA-256 matching nothing on disk; I-A `mark_exit` predicate false in the consume-then-append window; I-B `config_binding_at_freeze` unverified |
| 4 | `GO (0C / 0H / 1I)` | I-1 record understated the entry marker's blocking consequence |
| 5 | `GO (0C / 0H / 1I)` | I-1 `implementation_authorized` never checked by the preflight gate |

Every finding was closed on the bytes and re-reviewed by a fresh reviewer, not
argued away. The three High findings were substantive: rounds 1 and 2 showed
that the first repair had displaced the v5 impossibility one stage downstream
onto the single authorized GPU allocation rather than eliminating it.

## Round-5 verdict, as returned

`GO (0 Critical / 0 High / 1 Important)`.

The reviewer recomputed all 17 snapshot hashes (17/17 exact), audited all 18
distinct 64-hex strings in the implementation record against disk (all real, no
phantom), and confirmed by version-normalized diff that all three wrappers, all
three sbatch files and all five schemas differ from v5 by 0 lines. It confirmed
the prompt constant block and `prompt_hashes()` are byte-identical between v5
and v6, so the four hashes v6 will freeze are provably the four printed in
`slurm/logs/c04_a0t_small_v1_v5_preflight_13805.err`.

On the decisive question — whether any gate in the CPU-preflight path would
still fail after the authority flip — the reviewer answered **no**, and noted
that job 13805's traceback independently proves everything up to the old
`verify_static_config` prompt-hash line already ran and passed in the real SLURM
environment. It verified the remaining unproven segment statically: both train
ASR files exist as non-symlink regular files with matching sizes and hashes;
non-blank row counts are exactly 744 and 579 with unique IDs; **all** 1323 train
IDs across both datasets have `<id>.mp4` symlinks resolving to regular files
inside the pinned physical roots (0 missing, 0 escaping, 0 dangling), so any
200-subset the digest ranking selects resolves; all 14 pinned model files exist
at exactly the pinned sizes; `jsonschema` 4.26.0 is present and still exports
`Draft7Validator`; and `namespace.parent.mkdir` runs after every check, so a
HALT anywhere earlier leaves zero filesystem residue.

It listed the residual runtime risks in order: the authority manifest must
actually exist; its hand-computed `closure_sha256` over canonical JSON; its
exact 16-key schema shape; `authorization_snapshot` byte-matching the config;
the first live `jsonschema` import; model blob content, unverifiable inside the
review boundary; and the then-unguarded `implementation_authorized` flag.

## The single Important, and its closure

`verify_static_config` enumerated 16 of the 17 authorization keys and never read
`implementation_authorized`, which every downstream stage requires true. A
`false` value would therefore have passed the entire CPU preflight, created the
no-clobber namespace, and only then wedged every later stage — the same
"irreversible resource consumed before the rejecting check" pattern this repair
exists to remove. It was v5-inherited and unreachable without a gratuitous edit,
hence Important rather than High.

Closed by the reviewer's first suggested remedy: an explicit
`implementation_authorized is not True → HALT_INVALID_FREEZE` gate in
`scripts/analysis/c04_a0t_small_v1_v6_preflight.py`, placed beside the existing
materialization gate and before anything is materialized. This changed
`c04_a0t_small_v1_v6_preflight.py` and therefore the config's
`implementation_hashes` and the config hash; the resulting bytes are the ones
carried into the authority snapshot and re-reviewed by the unlock review.

## Authority boundary

This GO permits preparation and independent review of a strict CPU-preflight
authorization only. It does not by itself authorize CPU preflight, teacher
generation, dataset/model access, GPU or Slurm-GPU execution, the small tranche,
reconciliation, label access, or any scientific/result claim. The reviewer will
separately audit the resulting CPU-preflight authority snapshot.
