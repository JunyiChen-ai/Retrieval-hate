# C04 Claim-Driven Experiment Plan V2

**Status:** `FROZEN / PENDING FRESH INDEPENDENT DESIGN REVIEW`  
**Execution authority:** none

## Claim-to-evidence map

| Claim | Minimum convincing evidence | Kill condition |
|---|---|---|
| C1: four-way binding contributes beyond its fields and lower orders | FULL beats BASE, ADDITIVE, LOWER_ORDER, CONCAT_ALL4_MLP and RETAINED_INDEPENDENT4 in the same OOF arena on both datasets | strongest ordinary/structured control matches FULL, or any full-bank `+.050/+.050` gate fails |
| C2: native student internalizes the effect | STUDENT passes same train-OOF gate, native-only dev improves, and REMOVE/tuple+slot shuffle/role permute/noise/fallback controls behave in the registered direction | DIRECT-only effect, teacher/dev leakage, or any mechanism control matches FULL |

## Frozen fold, hyperparameter and seed protocol

Datasets are fit separately. After the cache seal, assign five outer folds by
sorting IDs independently within each binary class by
`sha256("C04-OUTER5-v2"||dataset||video_id)` and distributing round-robin.
For each outer fold, create four inner folds on its outer-train rows by the same
classwise rule with tag `C04-INNER4-v2||dataset||outer_fold`. No dev row enters a
fold or hyperparameter decision.

DIRECT is always train-only OOF: target/representation probes fit only the
outer-train rows and predict the held outer fold. STUDENT is measured in that
same OOF arena: every student target, normalization, covariance, MLP and
downstream head fits only outer train, then native video input predicts the
outer-held rows with no teacher read. Native-only development is a separate
corroboration after OOF settings freeze.

Inner selection grid is
`lambda_slot in {0.1,0.3}`,
`lambda_joint in {0.1,0.3}`,
`beta in {0.5,1.0}`. Select maximum mean inner macro-F1, then accuracy; exact
ties select lower `lambda_joint`, lower `lambda_slot`, then lower `beta`.
No dev result changes the choice.

Paired seeds are `0,1,2`. A seed covers the complete stochastic lineage:
student/native adaptation, factor/tensor heads, downstream head, batches and
checkpoint. The paired baseline uses the same seed, folds, steps and checkpoint
rule. Seed 0 is the minimal end-to-end pilot; seeds 1/2 are forbidden until its
gate passes. DIRECT fixed-target probes are deterministic but reuse the same
folds.

## A0T-small — exactly 200+200

This stage is forbidden until V2 design GO plus code/resource GO. Use only the
approved hash allowlists and resource envelope.

### Validity and representativeness guards

Before labels:

- exact 200/200 ID coverage, two records per ID, no duplicate/extra ID;
- model/processor/input/prompt/schema/map hashes exact;
- source and access ledgers prove no forbidden field/path;
- each slot has combined `stable+single_valid >=85%`, `missing <=10%`,
  `conflict <=20%`; joint all-four usable coverage is `>=60%`;
- no canonical non-fallback value occupies `>90%` of a slot.

After the cache is sealed and labels are revealed, each selected sample must
contain at least 40 rows of each class and its positive prevalence must be within
0.10 absolute of that dataset's complete train prevalence. Report transcript
length and frame-decode-failure distributions against ID-only full-train
metadata. Any failure is `HALT_SMALL_UNREPRESENTATIVE_OR_UNRELIABLE`; do not
redraw or expand.

### Same-arena scientific survival gates

Run nested OOF BASE, DIRECT and STUDENT plus every primary structural and
permutation control. `PASS_C04_SMALL_V2` requires separately on HateMM and
MHC-ZH:

1. DIRECT and STUDENT each have `Delta accuracy >=+0.040` and
   `Delta macro-F1 >=+0.040` versus paired BASE.
2. For each metric, `Delta_STUDENT / Delta_DIRECT >=0.80` in the **same OOF
   arena**; DIRECT delta must be positive before the ratio is defined.
3. FULL exceeds the strongest of CONCAT_ALL4_MLP, RETAINED_INDEPENDENT4,
   ADDITIVE and LOWER_ORDER by `>=+0.020` accuracy and macro-F1.
4. FULL exceeds tuple shuffle, every slot shuffle, role permutation, matched
   noise and every REMOVE arm by `>=+0.020` in both metrics.
5. Paired stratified bootstrap (2,000 resamples, seed 20260729) lower bounds are
   above zero for BASE deltas and all primary FULL-control deltas.
6. OOF Brier score is not worse than BASE by more than 0.010 and 10-bin ECE is
   not worse by more than 0.020. Confidence-to-A/B-agreement curves and all
   fallback corruption sensitivities are reported, never used to select rows.

The small gate is an internal fast-fail screen, not a performance claim, not
test evidence and not the full-bank registry gate. Any scientific failure is
`KILL_C04_SMALL_MATCHED_SIGNAL`. PASS only permits a fresh independent
result-to-claim review; it does not automatically authorize more extraction.

## A0T-full — conditional complete train banks

Only small PASS plus fresh independent GO plus new code/resource approval may
complete 744 HateMM and 579 MHC-ZH train IDs. Re-seal the complete two-prompt
bank before labels/probes. The total C04 ledger, including small and all later
GPU allocations, remains `<=8 GPU-hours`.

The binding full-bank Stage-0 requires on **both** datasets:

- DIRECT train-OOF `Delta accuracy >=+0.050` and
  `Delta macro-F1 >=+0.050`;
- STUDENT train-OOF the same `+.050/+.050`, with same-arena retention reported;
- FULL exceeds strongest CONCAT/INDEPENDENT4/LOWER_ORDER/ADDITIVE control by
  `>=+0.020/+0.020`;
- all primary BASE/control paired-bootstrap 95% lower bounds above zero;
- at least six net fixes for HateMM and four for MHC-ZH;
- native-only dev corroboration through the frozen student/deployed-head path,
  with teacher paths denied and no primary metric harm.

The `+.050` accuracy and macro-F1 requirements are not waived by small PASS,
reliability, tensor-prediction cosine or a DIRECT-only result.

## Minimal end-to-end and promotion

After full-bank PASS and fresh GO:

1. Seed-0 native student/deployed top-20 kNN must reach
   `>=+0.020` accuracy and macro-F1 on native-only dev for both datasets, with no
   claimed metric below `-0.005`. Otherwise pivot/kill before more seeds.
2. Only then run paired seeds 1/2. Promotion requires 3/3 positive deltas, mean
   `>=+0.030` accuracy and macro-F1 on both datasets, hierarchical paired
   bootstrap/Holm lower bounds above zero, all strong controls, and no teacher
   artifact readable at inference.
3. Test remains closed until a separately frozen final lineage; no test result
   may select a prompt, map, fold, seed, checkpoint or arm.

## Run order and budget stop

`design GO -> code/resource GO -> 200+200 teacher+seal -> CPU nested OOF ->
small result review -> conditional full teacher+seal -> full-bank OOF review ->
seed-0 native dev -> review -> paired seeds -> final freeze -> test`.

All compute, including CPU probes, must use SLURM under `HateVideo`. GPU jobs are
strictly serial, `8 CPU / 64 GB`, no `--time`, no forced `JobHeldUser` release.
The ledger counts allocated GPU count times elapsed wall time. The producer
checks the ledger before every record and exits fail-closed before exceeding
2 GPU-hours (small) or 8 GPU-hours (total). A cap failure returns to the user;
it never silently reduces prompts, frames, IDs or controls.

## Invalidity versus scientific failure

- Hash/path/schema/coverage/leakage/cap/fold/NaN failure:
  `HALT_INVALID_C04_V2`, no claim.
- Small matched-signal/control failure:
  `KILL_C04_SMALL_MATCHED_SIGNAL`.
- Full-bank `+.050/+.050`, control or internalization failure:
  `KILL_C04_FULL_REACHABILITY`.
- Only all serial gates plus independent reviews may promote C04.

