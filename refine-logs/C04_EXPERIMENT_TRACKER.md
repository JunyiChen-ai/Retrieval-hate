# C04 Experiment Tracker

**Current status:** `DESIGN_FROZEN_PENDING_INDEPENDENT_REVIEW`  
**Execution authority:** none  
**Test access:** forbidden

| Item | State | Evidence / next gate |
|---|---|---|
| Problem Anchor | FROZEN | `C04_PROBLEM_ANCHOR.md` |
| Error evidence | COMPLETE_READ_ONLY | HateMM FP1 n=5; EN lexical/counter n=7; ZH stance enrichment nonsignificant |
| Prior collision audit | COMPLETE_READ_ONLY | P4, SSR, LB-SCGP, C3-nontarget, MARS, Intent Projection |
| Existing asset audit | COMPLETE_NO_EXPLICIT_4FACTOR_BANK | `C04_STAGE0_ASSET_AUDIT.md` |
| Four-factor definitions | FROZEN | source is proposition origin; stance is presenter commitment; harm is target-act relation |
| Tensor/student mechanism | FROZEN | ordered four-way Hadamard tensor, native tensor student, ordinary kNN |
| Leakage boundary | FROZEN | train-only label-blind teacher; parent label after seal; no dev/test teacher |
| Stage-0 proxy route | FROZEN_PENDING_REVIEW | existing P8+K4 plus deterministic bilingual S/T compiler |
| Controls | FROZEN | FULL/REMOVE/SHUFFLE/NOISE/ADDITIVE/LOWER-ORDER/remove-factor/capacity |
| Statistics | FROZEN | two datasets; seed-0 gate; seeds 0/1/2; paired bootstrap/Holm |
| Resource order | FROZEN | CPU S0 before any teacher/GPU; serial SLURM afterward only on GO |
| Independent design review | PENDING | reviewer must verify exact file hashes |
| Implementation / Python / tests | NOT_STARTED / FORBIDDEN | wait for reviewer verdict |
| GPU / teacher / SLURM | NOT_STARTED / FORBIDDEN | wait for S0 PASS and later authorization |

## Decision states

- `GO_STAGE0`: implement only the CPU existing-bank S0 after code review.
- `REVISE_USER_AMENDMENT_REQUIRED`: proxy cannot legally PASS; ask user whether
  to authorize the bounded train-only teacher Stage-0 amendment.
- `KILL_C04_COLLISION_OR_INFEASIBILITY`: freeze C04 and advance only through a
  newly reopened Gate 0.

No status in this tracker authorizes a workload.

