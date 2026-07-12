# LB-SCGP G0 v7 Final Repair Handoff

Thread/session: `g0-v7-final-repair-20260712T000000Z-codex-local`

Scope: v7-only prospective repair under `refine-logs/lb_scgp/v7/` plus this
handoff. No v1-v6 method/config/freeze/artifact/report/result/code file was
modified. No subagents were used. Job `12846` was not controlled, rerun,
cancelled, requeued, released, held, or suspended.

## Fixed Design

- Preregistered eta: `1e-12`
- Frozen tau: `1e-7`
- Signed gap RHS: `tau + eta = 1.00001e-7`
- Design artifact:
  `refine-logs/lb_scgp/v7/G0_V7_PREREGISTERED_DESIGN.json`
- Design SHA256:
  `de77f2f86438d4a20ed436333c91f63bc62c11387506b74d3527e1b3dcff452b`
- Static validator payload:
  `adda172f7a7e57050e0e63227522522be3d0cfa6af66d31395cfdb0980a6ef3a`
- Actual certificate payload:
  `71fe91872d2b72709a76f31716311f96fe46c03dc99650cb5a1cb8e4d50d5f64`
- Independent replay payload:
  `32220860b043dc36622906e4c29f48f2d9003de5b32f2be1b749e823db0277bb`

## Source Hashes

- `v7_common.py`:
  `ff1a1d8eb736ed490549fc21fc0a3c54723c27fda9617d46222b2acd55c1aab4`
- `v7_actual_certificate.py`:
  `cd8875a3e0465d28af5122c51121f866914a8d6c987c015b6dfacc3c96f73492`
- `v7_independent_replay.py`:
  `e51d934bb25d71466915715872053920584eeeed9c8fe349bb73f256c28ab8b4`
- `validate_v7_static.py`:
  `664def390ee20fbab5af19cbff3170b248718f608b70a77b94ba664ac74bc986`
- Static/certificate/replay sbatches:
  `ee4a7622c6f59fc9f648b57b2dc38a9c0ff3b2c3d89062bccb21459f376990b8`,
  `8249a1b6bd0745649175d333f8c6d6bb4451199aee4dfbd8b4230ec621e31f51`,
  `42596129225b4e4d304daffa7202be7c3f80b2fe491007af9a50c71fdbe4ffec`
- Frozen existing hashes were replayed unchanged for v5 config, v5 Dykstra,
  v5 freeze, v6 analytic witness `12866`, and v6 analytic replay `12866`.

## Jobs

| Job | Role | Natural state | Exit | Elapsed | CPUs | ReqMem | MaxRSS | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `12895` | static validator | `COMPLETED` | `0:0` | `00:00:02` | `2` allocated / `1` requested | `4G` | `5352K` | `v7_static_validation_12895.json`, OK |
| `12896` | actual certificate | `COMPLETED` | `0:0` | `00:14:50` | `4` | `24G` | `178420K` | `v7_actual_certificate_12896.json`, pivot pending replay |
| `12898` | independent replay | `COMPLETED` | `0:0` | `00:00:04` | `2` allocated / `1` requested | `4G` | `3336K` | `v7_independent_replay_12898.json`, `REPLAY_OK_PIVOT_TRIGGERED` |

All three sbatches used `conda activate HateVideo`, CPU-only resources, and no
`--time` directive. `12896` and `12898` naturally passed through
`PENDING (JobHeldUser)` before running.

## Canonical Cells

Static validation passed:

- `topk=20`, `tau=1e-7`, `violation=1e-6`, `relative=1e-7`
- max independent orientations `8`, max pivots `32`
- orientation rank `1`
- descriptor: boundary near-tie `p00: p15 vs p18`, G0 score gap `0.0`
- compatible assignments: `[-1]`, `[1]`
- complete adjacent enumeration: `true`
- frozen original residual count: `589`
- additive signed-gap count per cell: `528`
- replay has no solver imports

Cell hashes:

| Cell | Assignment | top20 SHA256 | full outsider SHA256 | signed-gap SHA256 | cell SHA256 |
| --- | --- | --- | --- | --- | --- |
| `0` | `[-1]` | `77c3a833ef562a32e05930cbc145675e174a378d88b34290fdf643b2757af5c3` | `9e6c1bfdcde748bbd45c9ddcfdc8980108ab401132f59493a36047050e64cf13` | `e3000b7801bc7da3a737b5b1a399b1fb14f1127e1522fcb104319e2ed45700cf` | `a0ba199fcbce639fb7859ef34b19bcb300f5641ee66cbce8353e1ab23576601e` |
| `1` | `[1]` | `f88c7ec0e505f1b5b3ea8464836065fd386df4452acee6d8f823f377dbffb9d1` | `a8fe38cbb48d43db90f9f57918c99ed64ed9337caafc46c9b707c03f4633395a` | `602092c7946a29bad5312c3a47dd406a6522f69072754495a59a3707a27feb81` | `06199bab0ec401fb53a088ae65cae7e0281cefdd18f55a2d7e8e4b470b374cd9` |

## Phase I Results

No cell obtained an accepted strict signed-cell compatibility certificate. This
is not an infeasibility proof because no replayed Farkas/conic certificate was
produced.

Cell `0`:

- final status: `NO_COMPATIBILITY_WITNESS_NO_FARKAS`
- selected start: `accepted_12866_0`
- selected max original residual: `1.5452176205243973e-09`
- selected PSD min eigenvalue: `0.004263333136541338`
- selected canonical top20 equals cell: `true`
- selected signed gap min margin: `-6.736095449260811e-15`
- selected signed gap max residual: `6.736095449260811e-15`
- second fixed start `frozen_g0_zero_slack`: signed gap min margin
  `-9.073364282596437e-15`, max original residual
  `2.580594377077361e-09`

Cell `1`:

- final status: `NO_COMPATIBILITY_WITNESS_NO_FARKAS`
- selected start: `frozen_g0_zero_slack`
- selected max original residual: `1.7364167603783898e-09`
- selected PSD min eigenvalue: `0.004697389340520555`
- selected canonical top20 equals cell: `true`
- selected signed gap min margin: `-1.7186069668545097e-14`
- selected signed gap max residual: `1.7186069668545097e-14`
- other fixed start `accepted_12866_0`: signed gap min margin
  `-3.5671283028553954e-14`, max original residual
  `1.081838910121924e-09`

Farkas/conic incompatibility certificates: none.

## Phase II Results

Phase II was skipped for both cells because Phase I did not produce a strict
signed-cell compatibility witness. Therefore there is no accepted original
objective, stationarity, VI, PSD/SOC/linear dual, or complementarity
certificate.

The v7 terminal status is not `LOCAL_STATIONARY_CERTIFIED`.

## Independent Replay

Replay result: `REPLAY_OK_PIVOT_TRIGGERED`

Replay verified:

- certificate payload OK
- cell hashes OK
- source hashes match
- frozen existing hashes unchanged
- supervision boundary OK
- phase-I failures replayed as no-compatibility-witness/no-Farkas, not as
  infeasibility
- no Phase-II local stationarity certificate exists

## Defects, Repairs, and Boundary Notes

- No implementation repair was used after the static validator.
- No thresholds, eta, expected statuses, fixture, or supervision counters were
  changed.
- One procedural deviation occurred before v7 files were created: a local
  Python JSON-shape inspection was run once. It did not mutate artifacts and was
  not a solver/replay/numerical certificate run. All v7 validator, certificate,
  and replay Python execution was then performed through SLURM with
  `conda activate HateVideo`.
- No segment work was performed. Supervision remained:
  `only_gold_supervision=parent_video_binary_label`,
  `segment_gold_exists=false`, `segment_gold_used=false`; MLLM/OCR/teacher/
  held/validation/test counters all zero.

## Terminal Outcome

Pivot is triggered for the v7 authorized path. The strict signed-cell Phase I
certificate failed for both canonical compatible cells, no Farkas/conic
incompatibility certificate was produced, Phase II did not run, and independent
replay validated the pivot route.

This handoff makes no G0 PASS, freeze, formal gate, realfold, G1, performance,
validation/test, teacher, MLLM, OCR, or segment-level claim. The next action is
a fresh result-to-claim review of the v7 pivot-triggered evidence.
