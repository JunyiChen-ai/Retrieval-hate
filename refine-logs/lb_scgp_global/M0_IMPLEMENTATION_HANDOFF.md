# LB-SCGP Global-R2 M0 Implementation Handoff

Date: 2026-07-12

## Scope

This handoff covers only Run1:

`LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1`

Run1 is a contract freeze. It does not run Run2, Run3, MLLM, OCR, GPU, training, performance evaluation, validation, held, test, `query_z`, or `query_labels`.

## New Namespace

New config, schema, code, wrapper, and artifact namespace:

- `configs/lb_scgp_global_r2/m0_contract_freeze_v1.json`
- `schemas/lb_scgp_global_r2/scgp_global_cert_v2.schema.json`
- `schemas/lb_scgp_global_r2/scgp_global_contract_freeze_v1.schema.json`
- `scripts/analysis/lb_scgp_global_r2_common.py`
- `scripts/analysis/lb_scgp_global_r2_contract_freeze.py`
- `scripts/analysis/lb_scgp_global_r2_validate.py`
- `scripts/wrappers/lb_scgp_global_r2_run1.sh`
- `scripts/slurm/lb_scgp_global_r2_m0_contract_freeze.sbatch`
- `artifacts/lb_scgp_global/v1/m0/contract_freeze.json`

Preimplementation namespace check found no existing `configs/lb_scgp_global_r2`, `schemas/lb_scgp_global_r2`, `scripts/wrappers/lb_scgp_global_r2_*`, or `artifacts/lb_scgp_global` path.

## Implemented Interfaces

- Restricted `scgp_global_cert_v2` schema with extra-key rejection and no free text, targets, mechanisms, timestamps, spans, localization, verdicts, or rationales.
- Synthetic-only replica consensus, certificate encoding, common basis `Q`, and `M_Q(G)=Q^T(G-I)Q/N` structural operator interface.
- Global projection contract for PSD, unit diagonal, box constraints, coordinate trust, row trust, class trust, structural residual equality, regularized residuals, and robust constraints default off until coverage gate.
- Ambiguous cases are fail-open for geometry and fail-closed for claims.
- H-metric normal-cone/KKT certificate is the only future GO acceptance path; finite VI is diagnostic only.
- Rank-tail audit interface serializes `lambda_d`, `lambda_dplus1`, positive tail mass, tail ratio, negative mass, `lambda_min`, and reconstruction residual. Rank failure is null with no truncation/schema/tolerance rescue.
- Factor and Procrustes interfaces are present and exercised on synthetic fixtures.
- Forbidden-route guards reject local-v7 pass evidence, sample weighting, reranking, key selection, pair/triplet/SupCon, auxiliary heads, test teachers, and segment routes.

## Isolation

Run1 may hash:

- approved source/config/schema files;
- allowlisted train-bank provenance members `data/gt/MHC/train.jsonl` and `data/gt/MHC_zh/train.jsonl`;
- old protected LB-SCGP files only for no-clobber hash comparison.

Run1 records validation/test hashes only as plan-declared provenance strings and does not open validation/test/held content.

## Next Boundary

Run1 FROZEN is not M0 success. The next boundary is a fresh independent code+freeze audit before any Run2, Run3, cache, MLLM/OCR, GPU, training, validation, held, test, or performance work.
