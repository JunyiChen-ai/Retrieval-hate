# CVoI deviation D2 — C6 timings are binding only inside a GPU-exclusive window

- Timestamp: `2026-08-09T05:05:00+12:00`
- Scope: C6 (measured per-action cost registry and hardware lock) only.
- Registered **before** any C6 cost number exists. `artifacts/cvoi_acq/premetric-v2/c6-cost-v1/`
  contained exactly one file at registration time: `preflight_review_v1.json`
  (`861508f4bed31dcdf64a28f2fb7f968aaa2aa30a9ee832f01cac226df4c7a1dd`), whose status is
  `PENDING_REAL_MEASUREMENT_AND_INDEPENDENT_REVIEW`.
- No candidate metric opened. Test contact count 0.
- Prereg `1208ca27c0e7015cc92913f92cd3c4b56fdad03dce5c5f4bbda432da57aaef12` and appendix
  `8b282d79dea3c521ddadcd605d0615e36941b247d23cce9004c68c8d7a44c0b8` are byte-unchanged by
  this deviation.

## 1. Defect: an unregistered degree of freedom in the cost protocol

Prereg §6.1 requires freezing "engine and weights, environment/container, GPU model, decoder,
resolution, batch size, warm-up, repetition count and synchronization method", and names the
binding x-axis "online measured incremental latency per video **on the frozen hardware**".
Appendix §6 records "Hardware, driver, CUDA/cuDNN, engine, container/conda lock, clocks/power mode
and batch size 1", and appendix §16 requires that "Cost trials share one hardware model and
software image."

All of these pin **hardware identity**. None of them pins **machine state during timing**. The
registered protocol therefore neither requires GPU exclusivity nor forbids measuring while a
foreign compute process occupies the same device, and it registers no fallback hardware and no
contention handling. `costs.hardware_software_lock()` records GPU name, UUID, driver, pstate,
power limit and clocks, but records no co-tenancy field, so a contended run would be
indistinguishable from an exclusive run in the frozen artifacts. This is a registered blank, not a
permission.

## 2. Directional consequences of leaving it blank

A wall-clock latency measured while another user's process holds the device is a contention
artifact, not the frozen hardware's latency. Because it is time-varying and shared between the
OCR and dense extractors unequally, the effect is not a common scale factor. It would propagate
into:

1. the always-acquire denominator, hence every budget `B_i(b)=b*sum_a c_hat_i(a)` (appendix §6);
2. the `b=0.10` primary-budget feasibility test (prereg §6.2), which can flip to "no non-null
   action fits" purely from contention;
3. the cost-heterogeneity test that decides whether B12 is admissible and whether the word
   "knapsack" may be used at all (prereg §7, appendix §11);
4. matched-cost Pareto admissibility (prereg §8.1), which compares arms by measured cost and
   therefore requires all arms' costs to have been measured under the same machine state.

## 3. Registered rule (effective immediately, pre-metric)

- **R1.** A C6 timing run is binding only if, for its entire duration, `nvidia-smi
  --query-compute-apps` reports no compute process outside the measuring job's own process group.
  Exclusivity is confirmed before launch and sampled at a fixed 15 s interval throughout.
- **R2.** Any run during which a foreign compute process is observed, or during which the
  co-tenancy probe fails, is `VOID_CONTENDED`. Its timings are void for every registered purpose:
  they may not be promoted, averaged, corrected, rescaled, minimum-filtered, or used as a prior.
  Void attempts are retained as evidence under `attempt-*/` with a `VOID` marker and are never
  deleted.
- **R3.** The co-tenancy record and its verdict are mandatory C6 evidence. C6 cannot reach `PASS`
  without a `cotenancy_verdict.json` of status `EXCLUSIVE_OK` covering the full timing window,
  alongside the existing cost registry and independent audit.
- **R4.** Exclusivity is a **precondition**, never a substitute for any frozen constant. Warm-ups
  100, repetitions 5 with binding repetitions 2--5, batch size 1, seed 20260814, the phase order
  decode/preprocess/inference/postprocess, and the `torch.cuda.synchronize()` + CUDA-event
  discipline are unchanged.

## 4. Patch

No pinned module changes. `cost_driver.py`, `cost_overhead_driver.py`, `cost_audit.py` and
`costs.py` remain byte-identical to the digests recorded in `preflight_review_v1.json`; the
daemon re-verifies those four digests and halts with `HALT_C6_PINNED_SOURCE_DRIFT` before it will
start a timing run. The guard is external:

| file | sha256 | role |
|---|---|---|
| `scripts/cvoi_acq/c6_exclusivity.py` | `ce4b4fdf8b10d2184cce987472608424662cc05838aac428c8b28d76b5ec3907` | probe / watch / verify |
| `scripts/cvoi_acq/run_c6_exclusive_daemon.sh` | `b8dfbb36fc0135954118476beca8eece529d432dc6c66118f70034443d8bc605` | detached orchestrator |

The superseded foreground poller `scripts/cvoi_acq/run_c6_local_when_free.sh`
(`e61772cc4e0c3c31c2133472f4351a42924575bd2d958e640b7688b2fd5026e6`) is retired from every C6
consumer. It ran in the foreground, coupled to the SSH session, recorded no co-tenancy, and
produced no artifact in ~18 h of polling; it produced no cost number, so nothing is invalidated by
its retirement.

## 5. Payload

The freeze payload of appendix §15 is not yet computed for this study, and no `freeze_manifest.json`
exists, so no existing payload hash changes. At freeze, the two files above enter
`source_file_sha256`, and the `EXCLUSIVE_OK` verdict enters C6's evidence list in the completeness
registry.

## 6. What this deviation does not do

It alters no endpoint, arm, split, budget grid, decision rule, tolerance, model, threshold, action
definition or analysis. It does not promote C6. It grants no permission to inspect any candidate
metric.
