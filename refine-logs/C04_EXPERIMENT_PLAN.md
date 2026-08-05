# C04 Claim-Driven Experiment Plan

**Candidate:** `SPaSH-Tensor`  
**Status:** `FROZEN / PENDING INDEPENDENT DESIGN REVIEW`  
**Execution authority:** none

## Claim map

| Claim | Minimum convincing evidence | Fast falsifier |
|---|---|---|
| C1 joint role binding adds conditional information | FULL beats baseline, ADDITIVE and LOWER-ORDER on actual fold/deployed-head paths on two datasets | DIRECT or STUDENT Stage-0 misses `+0.050/+0.050` on either dataset |
| C2 the joint tensor is internalized into teacher-free kNN geometry | FULL beats capacity-matched REMOVE, SHUFFLE and NOISE; all factors removable; native dev/test has no teacher | direct tensor passes but STUDENT fails, or any ordinary control matches FULL |

## Frozen data boundaries

- Primary datasets: HateMM and MHC-ZH, trained separately.
- Stage-0 may read only existing train/dev summary, P8 text-feature and K4-score
  payloads listed in `C04_STAGE0_ASSET_AUDIT.md`.
- The factor reader may not access container labels. Train labels enter only the
  downstream paired evaluator after the proxy tensor is sealed.
- A new Stage-1 local teacher, if authorized, reads train only. Dev has native
  input and evaluation labels only. Test remains completely closed.
- No cross-dataset training, external pool/API, OCR, segment gold, stance gold,
  target gold, teacher verdict, sample removal or per-item selection.

## Stage 0 — existing-bank reachability

### S0.0 static audit

Before computation:

- exact train/dev ID coverage and order;
- generator/provenance and prompt isolation;
- no test path in the reader;
- exact factor-reader key allowlist;
- frozen bilingual cue compiler, role maps, dimensions, seeds and missingness;
- baseline command and strongest comparator identity;
- no output namespace collision.

Any failure is `HALT_INVALID_STAGE0`, not a scientific KILL.

### S0.1 direct tensor and ordinary controls

Using existing P8 summary embeddings, K4 scores and deterministic S/T cues, run
the actual fold/deployed-head path:

`BASE`, `P_ONLY`, `H_ONLY`, `STANCE_ONLY`, `ADDITIVE`,
`LOWER_ORDER_LE3`, `FULL_Q4`, `SHUFFLE_Q4`, `NOISE_Q4`.

`FULL_Q4` must reach, on both datasets:

- `Delta accuracy >= +0.050`;
- `Delta macro-F1 >= +0.050`;
- at least 6 net fixes on HateMM and 4 net fixes on MHC-ZH;
- at least `+0.020/+0.020` over the strongest ordinary control;
- paired bootstrap 95% lower bound above zero for both metrics.

### S0.2 tensor student removal test

Train proxy tensors only on train and evaluate native-only dev through the same
student representation used by the final method. Thresholds are identical to
S0.1. A DIRECT pass without a STUDENT pass is a KILL: the privileged signal
cannot be internalized.

### S0 decision

- `PASS_EXISTING_BANK_STAGE0`: S0.1 and S0.2 pass on both datasets, all validity
  guards true, and FULL beats ordinary controls. This authorizes design/code
  review for Stage-1 only.
- `KILL_C04_REACHABILITY`: any scientific threshold fails.
- `REVISE_USER_AMENDMENT_REQUIRED`: reviewer rules proxy may not PASS because
  source/stance are not representation-matched. No teacher work may occur until
  the user explicitly changes the registry.

## Stage 1 — new local-teacher signal gate

This stage is forbidden until S0 PASS and independent result-to-claim GO.

### Cache

- local `Qwen2.5-VL-7B-Instruct`, fixed revision and processor;
- one fixed schema prompt plus one frozen semantic paraphrase, greedy decoding;
- train IDs only, eight fixed frames and capped native transcript;
- strict JSON; parse failure becomes all-uncertain, never retry-selected;
- no binary verdict, label, neighbor, prediction, error status or split statistic;
- cache sealed before labels; complete hashes, allowlist, Merkle root and access
  ledger; teacher absent from dev/test.

Two deterministic prompt forms replace useless identical replicas. Agreement is
diagnostic only and never a weight or selector.

### Signal and transfer gate

On train OOF plus untouched dev:

- FULL projected `Delta accuracy >= +0.040` and
  `Delta macro-F1 >= +0.040` on both datasets;
- paired 95% CI lower bound `>0`;
- FULL beats REMOVE/SHUFFLE/NOISE/ADDITIVE/LOWER-ORDER;
- valid label-oracle calibration and permutation sensitivity;
- parse coverage at least 90%, no constant factor in more than 90% of rows, and
  nonzero joint-tensor entropy;
- student retains at least 80% of DIRECT tensor gain and still clears the
  `+0.040/+0.040` bar.

Fail any item: `KILL_C04_SIGNAL_OR_INTERNALIZATION`.

## Stage 2 — seed-0 end-to-end

Only after S1 review GO:

- one dataset at a time, same frozen native encoder/head/retrieval path;
- FULL and the minimum primary controls;
- final-epoch primary, validation-selected corroborative;
- `+0.020/+0.020` on both datasets;
- no primary metric harm below `-0.005`;
- FULL beats REMOVE/SHUFFLE/NOISE and LOWER-ORDER;
- no teacher read outside train and no teacher at inference.

Failure stops before multi-seed expansion.

## Stage 3 — paired seeds 0/1/2

For each dataset and both metrics:

- 3/3 paired deltas positive;
- mean `Delta >= +0.030`;
- hierarchical paired bootstrap over seeds/examples, 95% lower bound `>0`;
- Holm correction over the four dataset-by-metric primary hypotheses;
- FULL removal cost and shuffle/noise costs have same direction with confidence
  bounds excluding zero;
- mean/std, per-seed deltas, fix/break/net and both protocols reported.

## Final test

Test is touched once by the frozen final lineage only. It loads the native
student model and train memory, never factor banks, teacher caches, archive
fields, cue compiler or tensor targets. Failure on final test is reported; no
selection, repair or second test lineage is allowed.

## SLURM and resource order

No stage is currently authorized. If reviewer gates are satisfied, the order is:

1. CPU static audit and existing-bank Stage-0:
   `8 CPU / 32 GB`, estimated under 30 minutes;
2. independent S0 result-to-claim review;
3. sequential teacher cache, one GPU and at most `8 CPU / 64 GB` per dataset;
   preliminary ceiling 4–8 GPU-hours total, to be re-estimated before approval;
4. CPU cache seal and Stage-1 probe;
5. seed-0 end-to-end, one GPU at a time;
6. paired seeds `0/1/2`, datasets serial, controls only after FULL clears;
7. frozen final test lineage.

Every workload uses `conda activate HateVideo`, a project-local
`scripts/slurm/*.sbatch`, no `--time`, no forced release of `JobHeldUser`, and
the per-user ceiling of 16 CPU / 128 GB / 2 GPU. No chained submission is
authorized.

## Required reporting

Every stage records config/source/input/output SHA256, exact IDs, split access
ledger, label-entry point, factor coverage/entropy, validity guards, raw paired
metrics, all control deltas and the fired decision rule. A procedural PASS is
never a performance PASS.

