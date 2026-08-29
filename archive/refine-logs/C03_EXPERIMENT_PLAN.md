# C03 Experiment Plan — Terminal Gate Record

**Status:** `KILL_C03_DESIGN_INFEASIBILITY / NO_EXECUTION`  
**Candidate:** `Policy-Anchored Native MNTP`  
**Date:** 2026-07-29 (Pacific/Auckland)

## Claim map

| Claim | Minimum convincing evidence | Status |
|---|---|---|
| C1 native MNTP repairs text without damaging image/fusion | matched native bank; two-dataset fold-head gain; image/diversity/synergy belts | BLOCKED before Stage-0 |
| C2 policy conditioning adds value beyond generic MNTP | FULL > MNTP_ONLY, REMOVE, SHUFFLE and NOISE under identical inference | BLOCKED before Stage-0 |

## Frozen data boundary

- Allowed prospective data: HateMM train/dev_seen and MHC-ZH train/dev_seen only.
- Train labels may be used only by the downstream paired RGCL/fold-head evaluator.
- Policy inclusion, token masks, MNTP targets, loss weights, holdout and checkpoint
  are label-blind.
- Dev labels are evaluation-only and never select MNTP steps or hyperparameters.
- Test paths and files are forbidden until a final promoted lineage; C03 never
  reached such a lineage.
- No cross-dataset mixing, external pool, API, pseudo-relation gold, or segment gold.

## Stage order and gates

### A0 — representation-matched existing-bank reachability

Required before GPU: actual fold/deployed-head `+0.050` accuracy and `+0.050`
macro-F1 on both datasets, with at least 6 net HateMM fixes and 4 net MHC-ZH fixes.
Old caches may KILL but cannot PASS a policy-native claim.

**Observed asset status:** no eligible bank exists; see
`refine-logs/C03_ASSET_AUDIT.md`.

**Decision:** `KILL_C03_DESIGN_INFEASIBILITY`. A0 is not implemented or executed.

### S1 — auxiliary/signal gate

Cancelled. Had A0 been legally cleared, the frozen threshold would have been
`+0.040/+0.040` on both datasets with paired 95% CI lower bound above zero and
FULL beating matched REMOVE/SHUFFLE/NOISE signal controls.

### S2 — seed-0 end-to-end

Cancelled. Required `+0.020/+0.020` on both datasets, no claimed-dataset harm below
`-0.005`, image cosine `>=0.98`, stream cosine `<0.55`, and non-destructive fusion.

### S3 — paired three-seed promotion

Cancelled. Required seeds `0/1/2` over both adaptation and head lineages, 3/3
positive paired deltas, mean `+0.030/+0.030` on both datasets, hierarchical paired
bootstrap/Holm lower bounds above zero, and FULL beating all controls. Final-epoch
would be primary and validation-selected corroborative.

## Prospective resource ceiling

No resource is authorized or spent. If the registry were ever changed to permit a
fresh matched Stage-0 asset, the smallest candidate budget would be:

- A0 existing-bank CPU: 8 CPU / 32 GB / under 10 minutes;
- native two-pass FULL smoke+train+dev extraction: one A100 at a time, estimated
  at most 6 GPU-hours across both datasets from prior 3-epoch SFT measurements;
- controls run sequentially only after FULL clears the mechanism belts;
- no `--time`; all work through SLURM under `conda activate HateVideo`.

These are estimates, not authorizations.

## Terminal interpretation

The abstract native policy-MNTP hypothesis is untested. The registered C03 route is
retired because its mandatory evidence ordering cannot be satisfied with existing
assets. The next candidate boundary is Gate-0 reopening before C04; no C04
implementation, teacher, GPU, test, or SLURM action is authorized here.
