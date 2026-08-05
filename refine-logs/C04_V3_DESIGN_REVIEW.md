# C04 V3 Fresh Independent Design Review

**Verdict:** `REVISE`  
**Severity:** `0 Critical / 2 High / 1 Important`  
**Execution authority:** none

All submitted hashes exact-matched. The reviewer accepted the semantic-KILL
principle, global tuning/seed skeleton, dense lower-order map, `N=0` handling,
mechanical perturbation coverage and resource watchdog.

Remaining repairs:

1. Fallback controls must disprove reliance on the reliability mask, not reward
   it. Add state-only/state-blind or equivalent gates; collapse must erase all
   state distinction; stable-only/low-support cases need explicit applicability.
2. Give STANCE_ONLY, HARM_ONLY and CAPACITY_ONLY_NATIVE the same complete
   search/budget/tie treatment.
3. Malformed/missing prompt forms must flow through frozen rate thresholds and
   remain compatible with `single_valid`; delete or quantify undefined
   confidence/agreement/calibration KILL language.

No new user contract is required. No implementation or execution is authorized.

