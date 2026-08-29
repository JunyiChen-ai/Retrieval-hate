# M0 REALBANK-RESOURCE-v1 — Result-to-Claim Review (lineage verdict)

Date: 2026-07-13

Reviewer: **Claude Opus 4.8**, fresh independent **result-to-claim** role for
`LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1` (machine `runs[3]`). Separate from realbank-prep/freeze,
amendment/code-review, execution-authorization, and executor. Read-only except this document and the
companion `REALBANK_FULLCHAIN_STATIC_AUDIT.md`. Evidence read directly this session: the two job logs
(`slurm/logs/lbscgp_global_r2_realbank_resource_v1_12994.{out,err}`), the eight frozen entities, the
config/schema/machine plan, and the executor record. I ran no Python, no `sbatch`, no import, and
touched no validation/test/held/cache/query content.

**Model-binding note:** `CLAUDE.md` binds subagents to Opus 4.8; `AGENTS.md`'s "GPT-5.5 xhigh" backend
is unavailable this session, so this review runs on Opus 4.8 with fresh-context re-derivation.
Documented process fact, not a defect.

---

## Verdict

**NO CLAIM SUPPORTED — infrastructure-preflight death; zero scientific information; lineage v1 CLOSED.**

`claim_supported = no` (nothing to claim, neither positive nor negative). Job **12994 FAILED in 2 s**
at the producer's **first** validation-handoff read, **before** any numeric work. This is a
plumbing/infrastructure failure, not a scientific result:

- **No positive claim** is possible: no `resource_peak`, no `rank_tail`, no replay-hash, no
  isolation-injection evidence was produced (all N/A — the pipeline never reached them). The
  `3452K` MaxRSS is interpreter + partial-import overhead, **not** a valid pipeline peak; distance to
  the 96 GiB cap is undefined.
- **No negative claim** is possible either: the run did **not** measure-and-fail any gate (cap
  exceeded / rank_eps>d / replay mismatch / injection-not-rejected). It died upstream of all of them.
  It is therefore **not** the "informative numeric-section" burn the coordination session
  pre-accepted (R-2 cross-process replay, rank window, cap); it is the **preflight class** that per
  the authorization §7 escalation rule **triggers pause + full-chain audit**.
- **The one property that did hold: fail-closed safety.** No `decision.json`, no partial
  `source_manifest`/`access_ledger`/`semantic_verification`, no `*.publish.lock`; the artifact
  directory `artifacts/lb_scgp_global/v1/m0/realbank_resource/` is still absent; the temp validation
  JSON was removed; the final `jq .decision=="PASS"` gate was never reached (`COMPLETE=0`). No false
  PASS. Verified this session (dir absent; no temp leftover under `$TMPDIR`).

## Root cause (one paragraph)

The wrapper (line 59) minted the validator→producer handoff file with
`mktemp "${TMPDIR:-/tmp}/…json"`; on this cluster `$TMPDIR=/data/jehc223/home/tmp`, so the file landed
**outside** the repo root `/data/jehc223/RGCL`. The validator wrote it fine (its `--json-out` write at
`validate.py:186` is an unguarded `Path.write_text`), but the producer consumes it via
`read_json → canonical_root_path` (`common.py:130-139`), an **in-repo-only** hardening that calls
`resolved.relative_to(ROOT)` and raises `RuntimeError: path escapes repository root`. The two sides are
mutually incompatible on any host: unguarded out-of-repo write ↔ in-repo-only guarded read. Full
line-by-line execution calculus and the fix specification are in the companion
`REALBANK_FULLCHAIN_STATIC_AUDIT.md`; the coordination-ruled fix direction **(a)** — write the handoff
JSON **inside** the repo, **without** loosening the path hardening — is confirmed correct there.

## This is the fourth consecutive infrastructure-preflight burn

| run | job | preflight-class root cause | scientific info |
|---|---|---|---|
| Run2 **v1** | (v1) | interface-key mismatch (contract/interface drift) | none |
| Run2 **v2** | (v2) | missing package (`jsonschema` absent from HateVideo) | none |
| Run2 **v3** | (v3) | frozen-document drift (index/doc drift) | none |
| **realbank v1** | **12994** | **runtime file-handoff path escapes repo (`$TMPDIR`)** | **none** |

**Institutional lesson (the thing the simulation table kept missing).** Each ceremony's runtime
cross-check simulation modeled *code assertions vs frozen documents / frozen environment packages*
(hashes, dependency `find_spec`, index pins, resource caps, three-way interface alignment). None of the
four modeled the **runtime file-handoff paths and the environment variables that determine them** —
where a producer/consumer temp file physically lands, and whether an ambient env var (`$TMPDIR`, and by
extension `HOME`/cache/locale/thread vars) can move it. The v4-authorization §7 upgrade rule
("preflight-class death → pause ceremony, run full-chain audit before v-next") exists precisely to stop
this recurrence; the full-chain audit now **adds the missing row class**: a per-handoff writer→path→
reader→path-check table, plus an ambient-env-interaction inventory. That table and inventory are the
companion document's core, and they are the standing addition to every future clone's preflight model.

## Lineage disposition & v2 opening conditions

- **realbank-v1 lineage is CLOSED.** Single-submit budget spent on submission; **not resubmitted**
  (executor instruction 5, correctly honored). No v-next may reuse the v1 wrapper unchanged — it cannot
  pass on any host.
- **realbank-v2 opens only when ALL of the following hold** (single gated path):
  1. **Fix (a) applied** to the wrapper exactly as specified in
     `REALBANK_FULLCHAIN_STATIC_AUDIT.md §5` (handoff JSON written into a repo-internal temp dir;
     path hardening **not** loosened), and **nothing else** changed (blast radius = wrapper entity #7
     only; config/schema/producer/verifier/common/sbatch untouched).
  2. **Full-chain static audit PASSED** — this is `REALBANK_FULLCHAIN_STATIC_AUDIT.md` (verdict there:
     after fix (a), the whole handoff table is PASS; 0 UNPROVABLE rows).
  3. **Re-freeze** — recompute and record the wrapper's new SHA256 in a v2 freeze doc; the other seven
     entities re-`sha256sum -c` unchanged. (The config does **not** hash-bind the realbank wrapper, so
     no config edit is forced; see audit §5.)
  4. **Fresh independent code-review** (0C/0H) of the 3-line wrapper change, separate role.
  5. **Fresh execution authorization** (single-submit ledger re-verified empty, resources, env).
  6. **Exactly one** CPU-only SLURM submission; any fail-closed non-GO is again a
     consciously-accepted STOP, not grounds for a second submit.

## Required statements

- No performance evidence exists and none is claimed. The run emitted no accuracy / macro-F1 and did no
  training or kNN; it produced **no** valid resource/rank/replay/injection measurement (died at
  preflight, before torch import / any bank load / any linear algebra).
- The only project gold is `parent_video_binary_label`; no segment/frame/span/localization/stance/
  target/mechanism/rationale/fragment gold was assumed or introduced. No train label was read (no bank
  was loaded). The `is_science=false` structural placeholder never executed and certifies nothing.
- Run4 (M1 cache), MLLM/cache, validation/test, and training remain **locked**. This failed run unlocks
  nothing.
- Reviewer = Claude Opus 4.8, fresh independent result-to-claim role, separate from realbank-prep/
  freeze, amendment/code-review, authorization, and executor. Wrote only this document and the
  companion full-chain audit; edited no code/config/schema/plan; submitted no job.
