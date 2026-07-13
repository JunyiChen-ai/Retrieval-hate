# M0 REALBANK-RESOURCE-v1 Execution Record

Date: 2026-07-13

Executor: **Claude Opus 4.8**, executor role for `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1`
(machine `runs[3]`), separate from the realbank-prep/freeze, merged amendment/code-review, and
execution-authorization roles. Authorized by
`REALBANK_RESOURCE_V1_EXECUTION_AUTHORIZATION.md` (this session) — scope: exactly one CPU-only
SLURM submission.

## Outcome (one line)

**FAILED — fail-closed, preflight-class death.** The producer refused the validation-handoff JSON
because the wrapper placed it outside the repository root (`$TMPDIR`), tripping the producer's
`read_json` path-escape guard. No artifact, no `decision.json`, no false PASS. Single-submit budget
spent; **not resubmitted**. Routes to a fresh result-to-claim / full-chain audit before any v-next.

## 1. Submission

- **Pre-submit guard (executor, read-only):** re-`sha256sum -c` of all 8 frozen entities → **8/8
  MATCH** (`c436c3dd…` config, `db79cdd3…` schema, `46e1f3fe…` common, `b2bbec02…` validate,
  `dc38d5c3…` producer, `49cc2d9a…` verify, `f80b41ea…` wrapper, `9c4ecc05…` sbatch); `sacct` prior
  rows = 0; artifact dir absent. Submit was guarded on all three (would have aborted on any miss).
- **Command:** `sbatch scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v1.sbatch`
- **Job id: `12994`** (job name `lbscgp_global_r2_realbank_resource_v1`).
- No `--time` set. Requested 16 CPU / 96 GB / 0 GPU / HateVideo.

## 2. Timeline (sacct)

| event | time |
|---|---|
| Submit | 2026-07-13T12:57:24 |
| batch start (auto-released from JobHeldUser within ~8 s) | 2026-07-13T12:57:32 |
| End (FAILED) | 2026-07-13T12:57:34 |
| Elapsed | 00:00:02 |
| State / ExitCode | **FAILED / 1:0** |
| MaxRSS (`12994.batch`) | **3452K (≈3.4 MB)** — pre-torch; **not** a valid pipeline peak |

## 3. Terminal-state evidence collection

| required item | result |
|---|---|
| `decision.json` (GO/STOP) | **does not exist** — producer never published; fail-closed non-publish |
| `resource_peak` (peak RSS) vs 96 GiB cap | **NOT measured** — job died at the producer's first `read_json`, before torch import / any bank load / any linear algebra. The 3.4 MB MaxRSS is interpreter + partial-import overhead, **not** the O(N³) pipeline peak. Distance-to-cap is therefore undefined (no valid measurement). |
| `rank_tail` (rank_eps(G0) ≤ d) | **N/A** — never computed |
| in-job replay-hash consistency | **N/A** — never computed |
| isolation injections (11) + tamper mutations (15) all REJECT | **N/A** — died before the producer's isolation/emit logic |
| `.out` | **empty** (0 bytes) |
| `.err` | the traceback below |

Fail-closed hygiene (all confirmed post-mortem):
- `artifacts/lb_scgp_global/v1/m0/realbank_resource/` **still absent** — `find … -name '*realbank*'`
  = 0 files; no `decision.json`, no `*.publish.lock`, no partial `source_manifest`/`access_ledger`/
  `semantic_verification`. The wrapper `cleanup_on_exit` trap (`COMPLETE=0`) removed all prospective
  outputs (`PROSPECTIVE_OUTPUTS`).
- The temp validation JSON was removed (`rm -f "$VALIDATION_JSON"`); no leftover in `$TMPDIR`.
- No false PASS: the wrapper's final `jq -e '.decision=="PASS"'` gate was never reached; `COMPLETE`
  stayed 0.

## 4. Root cause — wrapper↔producer temp-path incompatibility (preflight-class)

`.err` (verbatim, essential lines):

```
ValueError: '/data/jehc223/home/tmp/lbscgp_global_r2_realbank_resource_v1_validation.bIAbnI.json'
  is not in the subpath of '/data/jehc223/RGCL' OR one path is relative and the other is absolute.
The above exception was the direct cause of the following exception:
  producer.py:119  validation = read_json(args.validation_json)
  common.py:143    fs_path, _ = canonical_root_path(path)
  common.py:138    raise RuntimeError(f"path escapes repository root: {path}") from exc
RuntimeError: path escapes repository root: /data/jehc223/home/tmp/lbscgp_…_validation.bIAbnI.json
```

Chain:

1. **Wrapper (line 59):** `VALIDATION_JSON=$(mktemp "${TMPDIR:-/tmp}/lbscgp_…_validation.XXXXXX.json")`.
   On this cluster `$TMPDIR=/data/jehc223/home/tmp` → the handoff file lands **outside** the repo
   root `/data/jehc223/RGCL`. (The `${TMPDIR:-/tmp}` fallback `/tmp` is **also** outside the repo, so
   the location is out-of-repo under *any* `$TMPDIR`.)
2. **Validator (lines 60–63):** wrote `--json-out $VALIDATION_JSON` **successfully** (it uses a plain
   file write, not the guarded reader) — validation preflight itself did **not** fail.
3. **Producer (lines 65–68 → producer.py:119):** consumed `--validation-json` via
   `read_json` → `canonical_root_path` (common.py:130–139), which resolves the path and calls
   `resolved.relative_to(root)`; an out-of-repo path raises `ValueError`, wrapped to
   `RuntimeError("path escapes repository root")`. Producer aborts on its **first** action.

**Why static review + authorization both missed it:** the handoff path is a **runtime environment
fact** (`$TMPDIR`), not visible in static analysis; the 21-row cross-check table modeled config/
schema/hash/dependency/resource assertions but **no row modeled the validator→producer temp-file
handoff path**. The authorization gates (8-entity hashes, `{numpy,torch,jsonschema}` availability,
single-submit ledger, resource caps) are all **orthogonal** to this plumbing path and remain
correct — they simply do not cover it.

**Nature:** a realbank-implementation **plumbing incompatibility** — the producer's `read_json`
path-escape hardening (`canonical_root_path`, an in-repo-only guard) is incompatible with the
wrapper's out-of-repo `mktemp` handoff. This pair cannot pass on any host until one side changes
(handoff written **inside** the repo, e.g. `mktemp -p` a repo-relative dir; **or** `read_json`
permitted to read the specific validation temp path).

## 5. Classification & escalation

- **Fail-closed, no false PASS, no artifact** — the safety property held exactly as designed.
- **Preflight-class death.** It died at the producer's validation-handoff read, **before** any
  numeric work (before torch import, bank load, eigendecomposition, replay, or isolation logic). Per
  the escalation rule carried from the v4 authorization (§7: "died at a *preflight-class* miss →
  pause ceremony, run full-chain audit before v-next"), this is **exactly** the preflight class that
  triggers a **pause + full-chain audit**. It is **not** the informative numeric-section class
  (R-2 cross-process replay, rank window, cap) that the coordination session pre-accepted as a
  legitimate diagnostic burn.
- **Single-submit budget spent; not resubmitted** (executor instruction 5). The v-next decision
  (fix the temp-path plumbing, re-freeze, re-review, re-authorize) belongs to the coordination
  session / main, not the executor.

## Required statements

- No performance evidence exists and none is claimed; the run emitted no accuracy/macro-F1 and did no
  training or kNN. It produced **no** valid resource/rank/replay/injection measurement (died at
  preflight).
- The only project gold is `parent_video_binary_label`; no segment/frame/span/localization/stance/
  target/mechanism/rationale/fragment gold was assumed or introduced. No train label was read (no
  bank was even loaded).
- Run4 (M1 cache), MLLM/cache, validation/test, and training remain locked. This failed run unlocks
  nothing.
- Executor = Claude Opus 4.8, separate from realbank-prep/freeze, amendment/code-review, and
  execution-authorization roles. The executor wrote only this record and the authorization document;
  no code/config/schema/plan was edited, and the job was submitted exactly once.
