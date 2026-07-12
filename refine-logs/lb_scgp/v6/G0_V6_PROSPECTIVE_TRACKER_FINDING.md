# G0 v6 Prospective Tracker Finding

Status: prospective nonformal development sanity only.

This v6 repair does not claim G0 PASS, does not freeze any method/config/formal
artifact, does not run formal synthetic, real-fold, replay, decision, G1,
teacher, MLLM, OCR, held, validation, test, or performance work.

The design target is a fail-closed Phase-II local scientific certificate in
Gram space:

- final vote/certificate semantics are top20 only;
- `final_top20_rankings` and `full_outsider_order_for_enumeration` are separate
  certificate objects with independent hashes;
- full `n-1` order is enumeration aid only and is never directly compared
  against a top20 cell;
- Phase-I job `12866` is treated only as a Slater/primal warm start witness,
  not as scientific projection evidence;
- Phase-II acceptance requires original objective stationarity, VI/KKT, PSD,
  complementarity, rank halfspace completeness, independent replay, and
  no-segment/zero-counter checks.

Fresh SLURM order:

1. Run `refine-logs/lb_scgp/v6/runtime/validate_scientific_repair_v6.sbatch`.
2. Only after natural `COMPLETED` and machine `OK`, run
   `refine-logs/lb_scgp/v6/runtime/scientific_repair_sanity.sbatch`.

Any failed validator, failed oracle, failed replay, failed KKT/VI/PSD check, or
unsatisfied adversarial case is terminally recorded as bounded/remove evidence
for this prospective sanity and is never treated as infeasibility proof.

## Completed Prospective Sanity

- Validator `12867`: `COMPLETED`, exit `0:0`, machine status `OK`.
- First oracle `12868`: `FAILED`, exit `2:0`, machine status
  `BOUNDED_REMOVE`; recorded as a failed development run. The local boundary
  fixture accidentally had 21 tie descriptors and was repaired without changing
  thresholds or expected statuses.
- Validator rerun `12869`: `COMPLETED`, exit `0:0`, machine status `OK`.
- Oracle rerun `12870`: `COMPLETED`, exit `0:0`; producer
  `NONFORMAL_SANITY_OK`, independent replay `REPLAY_OK`.

This remains prospective nonformal sanity only: no G0 PASS, freeze, formal
synthetic, real-fold, replay/decision gate, G1, teacher, MLLM, OCR, held,
validation, test, or performance work was run or unlocked.
