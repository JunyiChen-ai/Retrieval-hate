# C04 V4 Fresh Independent Design Review

**Verdict:** `GO`  
**Severity:** `0 Critical / 0 High / 0 Important`  
**Execution authority:** none

The reviewer exact-matched all five V4 hashes and confirmed the V3 bases were
unchanged. Review was read-only.

Accepted closures:

- STATE_ONLY must lose with positive paired CI; STATE_BLIND must retain at least
  80% and be equivalent within `+/-0.010`; collapse/shuffle/noise are
  `+/-0.010` equivalence controls. Four-state collapse erases all state
  distinction; single-state exact no-op and low-support NA never waive the two
  primary arms. V3's erroneous `+.020` fallback-drop requirement is superseded.
- STANCE_ONLY, HARM_ONLY and CAPACITY_ONLY_NATIVE receive the complete eight
  trials, common outer aggregation, steps, checkpoint and tie rule; capacity
  duplicates must be byte-identical.
- Individual malformed/missing responses enter `single_valid`/`missing`; only
  the five frozen completed-cache rates can scientifically kill semantic
  reliability. Fixture/hash/crash/incomplete failures HALT. Undefined
  sensitivity/calibration KILL language is absent.

No new design defect or regression of an accepted boundary was found.

`GO` authorizes only prospective implementation and code/resource review. It
does not authorize implementation changes, map/cache production, teacher calls,
Python/tests, GPU, SLURM, label/test access or experiment execution.

