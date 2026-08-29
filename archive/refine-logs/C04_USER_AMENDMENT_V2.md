# C04 User Amendment V2 — Bounded Matched-Teacher Pre-Gate

**Status:** `FROZEN / USER APPROVED / NO EXECUTION AUTHORITY`  
**Date:** 2026-07-29 (Pacific/Auckland)  
**Scope:** C04 only

This amendment changes only the evidence order that previously made C04
unreachable. It does not modify the immutable problem anchor, waive a metric,
authorize implementation, or authorize a teacher/GPU/SLURM action.

## Approved first tranche

- Datasets remain separate: HateMM and MHC-ZH.
- Select exactly 200 **train** IDs per dataset by ascending
  `sha256(C04-A0T-SMALL-v1 || dataset || video_id || 20260729)`.
- The ID-only allowlist and its hash must be sealed before any label value is
  readable. There is no replacement, class balancing, error-based selection or
  redraw.
- Teacher: local, open-weight `Qwen/Qwen2.5-VL-7B-Instruct`, offline and frozen.
  Its local model/processor tree hashes are mandatory code-review fields.
- Each ID receives exactly two fixed prompt forms, eight fixed full-video frames
  and its native transcript capped by the deterministic head/tail rule in the
  V2 proposal. No OCR is allowed.
- The teacher sees no label, prediction, neighbor, rank, margin, error status,
  dataset statistic, fold role or intended use.
- No dev/test content or teacher call, API, external pool, cross-dataset input,
  cross-dataset fit or cross-dataset calibration is allowed.
- The two prompt records and access ledger are sealed before train labels enter.

The first tranche must use reviewed project-local SLURM wrappers under
`conda activate HateVideo`, one GPU at a time, at most `8 CPU / 64 GB`, no
`--time`, and an aggregate maximum of **2 GPU-hours across both datasets and all
C04 jobs**. `JobHeldUser` is allowed to clear automatically; it must not be
force-released. A preflight estimate above the cap or a runtime ledger reaching
the cap halts the tranche.

## Conditional full-bank tranche

Only `PASS_C04_SMALL_V2` followed by a fresh independent result-to-claim `GO`
may request a code/resource review for the remainder. That conditional tranche:

- completes exactly all 744 HateMM train IDs and all 579 MHC-ZH train IDs;
- remains train-only, local/offline, two-prompt, eight-frame, transcript-only;
- remains one GPU at a time with `8 CPU / 64 GB`;
- has an aggregate C04 ceiling of **8 GPU-hours**, including every GPU-second
  consumed by the first tranche and any later C04 extraction/adaptation job;
- still forbids dev/test teacher, API, OCR and cross-dataset data.

## Non-waived scientific gate

The 200-ID tranche is a fast internal survival screen, not a result claim and not
the registry Stage-0. The completed full bank must still establish, separately
on both datasets through the actual fold/deployed-head path:

- train-only DIRECT OOF `Delta accuracy >= +0.050` and
  `Delta macro-F1 >= +0.050`;
- native-only STUDENT OOF `Delta accuracy >= +0.050` and
  `Delta macro-F1 >= +0.050`;
- native-only development corroboration through the frozen student path, with no
  teacher artifact available to the evaluator;
- the mechanism and statistical controls frozen in the V2 experiment plan.

The old P8+K4+cue proxy is diagnostic-only. It may neither pass nor scientifically
kill C04.

## Current hard stop

Until the V2 amendment, asset audit, proposal, plan, tracker and review response
receive a fresh independent design `GO`, the following remain false:
`implementation_authorized`, `teacher_authorized`, `gpu_authorized`,
`slurm_authorized`, `test_authorized`.

