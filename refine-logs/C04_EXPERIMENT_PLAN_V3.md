# C04 Experiment Plan V3 — Normative Repair Overlay

**Status:** `FROZEN / PENDING FRESH INDEPENDENT REVIEW`  
**Base:** `C04_EXPERIMENT_PLAN_V2.md`  
**Execution authority:** none

Every unmodified V2 gate remains binding.

## Unique tuning and seed rule

Small and full-bank nested OOF use student seed **0** for FULL and every student
control. DIRECT is deterministic. Each of
`FULL`, `CONCAT_ALL4_MLP`, `RETAINED_INDEPENDENT4`, `LOWER_ORDER_LE3`,
`ADDITIVE`, each REMOVE and each corruption arm receives its **own** complete
eight-point V2 inner grid, identical optimizer/checkpoint budget and identical
tie rule. No arm inherits FULL's winner or receives fewer trials. BASE uses the
already-frozen paired baseline recipe and seed 0.

For a given arm and grid point, average its inner held-fold macro-F1 across the
four inner folds inside each outer fold, then average those five outer-fold
means. Select maximum grand mean macro-F1; tie by the analogous grand mean
accuracy, then lower `lambda_joint`, lower `lambda_slot`, lower `beta`. This
produces exactly one `(lambda_slot,lambda_joint,beta)` per arm for its
full-train/native-dev fit. The aggregation is fixed before OOF/dev results.
Later paired seeds `1,2` reuse the seed-0-selected tuple without retuning, while
covering the complete adaptation+student+downstream lineage.

## Semantic reliability decision

Exact producer/manifest/schema/hash failure is `HALT_INVALID_C04_V3`.
After a valid cache completes, failure of any V2 coverage/conflict/missing/
degeneracy/calibration threshold is
`KILL_C04_TEACHER_SEMANTIC_RELIABILITY`. No prompt/model/schema/threshold rewrite,
retry, redraw or row selection is allowed under C04.

## Binding corruption margins

For both the 200-ID small gate and the completed full-bank gate, separately on
HateMM and MHC-ZH, seed-0 nested OOF `FULL` must exceed each of:

`TUPLE_SHUFFLE`, all four `SLOT_SHUFFLE_f`, `ROLE_PERMUTE`,
`NOISE_MATCHED`, all four `REMOVE_f`, `FALLBACK_COLLAPSE`,
`SHUFFLE_FALLBACK_MASK`, and `NOISE_FALLBACK_MASK`

by `>=+0.020` accuracy **and** `>=+0.020` macro-F1. For every listed paired
comparison and both metrics, the stratified paired-bootstrap 95% lower bound
(2,000 resamples, seed 20260729) must be `>0`. These are conjunctive; strongest-
control-only reporting is insufficient. The existing small BASE/structural
gates and full-bank `+.050/+.050` DIRECT/STUDENT gates are unchanged.

## GPU-hour no-overshoot rule

No SLURM `--time` is set. Before each serial GPU submission, sum exact completed
`sacct ElapsedRaw * allocated_GPU_count` plus any reviewed partial-job ledger.
Let `remaining_seconds = cap_seconds - consumed_seconds` for the active 2-hour
or 8-hour ceiling. If `remaining_seconds <=300`, do not submit.

At allocation start the reviewed wrapper immediately starts a monotonic timer
and launches the producer under an internal watchdog with hard deadline
`remaining_seconds-120`; it sends TERM at the deadline and KILL after 30 seconds.
The producer checks the same deadline before every model forward, writes only
checkpointed incomplete shards, and never seals a partial bank. The 120-second
reserve covers cleanup/accounting. After completion, `sacct` replaces provisional
time in the ledger. Any exact ledger over cap, watchdog event or incomplete bank
is `HALT_RESOURCE_CAP` and forbids another job; prompts/frames/IDs/controls are
never silently reduced.

