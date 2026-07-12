# M0 Run2-v2 Implementation Fix2/Freeze

Date: 2026-07-12

Scope: static implementation repair and freeze only. I read `AGENTS.md` and the fresh review `M0_RUN2_V2_CODE_REVIEW.md` completely first. I did not run Python, imports, `py_compile`, tests, data/model code, conda, SLURM, `sbatch`, `squeue`, experiments, MLLM/OCR/API/network/model calls, GPU/training/evaluation, or validation/test data/cache inspection. No artifact under `artifacts/lb_scgp_global/v2` was created.

Execution remains locked. This report is not performance evidence, not execution authorization, and not a Run3/M1 unlock.

## Mathematical Repair

The failed implementation tried to keep `G0 = I_N` while requiring `G_star` to be PSD, unit diagonal, rank `<= d=3`, nontrivially moved, and within the unchanged coordinate trust threshold `|G_star-G0|_offdiag <= 0.02`. For `N>d`, that identity-baseline combination is impossible: any rank-`d` correlation matrix obeys the Welch lower bound

`mean_{i<j} G_ij^2 >= (N-d)/(d(N-1))`.

For the FULL fixture, `N=10,d=3`, so some `|G_ij| >= sqrt(7/27) ~= 0.509`, far above `0.02`. Therefore fix2 preserves `d`, fixture identities/counts, thresholds, roles, rank gate, and positive movement by using a unit-diagonal local projection baseline `G0` rather than the impossible identity baseline.

Construction:

1. Build deterministic unit rows `V in R^{N x d}` and set `G_star = V V^T`. Then `G_star >= 0`, `diag(G_star)=1`, and `rank(G_star) <= d`.
2. Let `P_ker` be the orthogonal projector onto `ker(G_star)`. Set `S_psd = tau P_ker` with `tau > 0`, so `S_psd >= 0` and `S_psd G_star = 0`.
3. Use the PSD-cone normal sign for the feasible constraint `G in S_+`: `v_psd = -S_psd`. This is dual feasible and complementary because `tr(S_psd G_star)=0`.
4. Choose structural multiplier `nu`; define `B = A_struct^T nu`. Set `Delta = offdiag(B) + offdiag(S_psd)`, `G0 = G_star - Delta`, `r = -nu/lambda`, and `b = A_struct(G_star) - r`.
5. Use diagonal affine dual `D = diag(B) + diag(S_psd)`. Then G-stationarity is exact:
   `Delta - B + D - S_psd = 0`.
   The r-block is exact:
   `lambda r + nu = 0`.

This materializes the active PSD normal family instead of serializing a zero PSD normal on a boundary point.

## Closure Matrix

| Finding | Fix2 status |
|---|---|
| H5 rank contract | Implemented rank-`<=3` PSD/unit-diagonal `G_star` construction for accepted non-REMOVE cases, with nonzero movement and nonzero PSD dual normal for the FULL KKT certificate. Rank gate, thresholds, fixture counts/roles, controls, and `Q^T(G-I)Q/N` are preserved. |
| H6 verifier completeness | Independent verifier now recomputes every case's `movement_metrics`, `primal_residuals`, `rank_audit`, `factor_replay`, and `robust_coverage`, plus rank-failure probe audit values. Added refreshed-hash targeted injections for each drift class. |
| M2 jsonschema risk | Static metadata does not prove `jsonschema` is installed. The existing single future SLURM validator now preflights `jsonschema` with the job Python and fails closed before producer publish if missing. No install/network/extra job was added. |
| L2 plan wording | `EXPERIMENT_PLAN.md`, tracker, and machine plan now say fix2 is complete but locked for fresh 0C/0H code review, exact hashes/no-clobber review, and separate execution authorization. `ready_for_execution=false` and downstream locks remain. |

## Changed Files

| Path | SHA256 |
|---|---|
| `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json` | `5545826820cd86f588fb43fd73b4070767fb9f904ea683fc2245f0903d48f700` |
| `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v2.schema.json` | `735250091f6a92ef787d9eadccca3c438379cc07f2418439401916270eced9a2` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_common.py` | `5ef8f677e13c0586675a3072b85774b90b7eb5e6ba8da91188f128d3a6d6e24f` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_validate.py` | `4389c4a1d1cbe21ec516ec414c472ebff075bbfb4a8041939163f478bdc3fc36` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py` | `8c4cc842aa53e3d8293744449101282a33abb7715691653bb934b2e563b1cd51` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py` | `795b56f852534c2cfb4264c9dec2f43dd4abc75a4655282465b6546d39111ef1` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.md` | `af1c217c44efc26003150be0bf22fedb499a7019931ca69a0c0164c30557b23a` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json` | `6caa5c2e78961fc7d60f9f970e319534aa6af3b6cc4a033b76b83f2ab18a0492` |
| `refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md` | `327614bbb3a8ce0c493acbdda4ef6b25fa555243bdbe041535a51a9d81800db2` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN_HASHES.sha256` | `2e6d731dba169d4b2487b098e996e77b8089efe3b41d9375f08ee210d774c802` |

Unchanged parity files: case schema `df3616ff50c54dc756bb600c6ef26626afafe070678450d10ca73b9ff83ddcac`, wrapper `14eb036f17b77037fd89624c0f9f7487432fa476da540b0fbc5f503719320716`, sbatch `f91462303e4b28eb0722d546f135c00c2aeacf56151e27d5421f211520ba94bf`.

Fresh review hash bound in config: `be646c3ad2cf55681b44397cfa67cc38313d70cb0b26705d5c4354f3a5c3c9bc`. This fix2 report is intentionally not config-bound to avoid a circular report/config hash.

## Static Checks

- `jq -e` passed for config, payload schema, case schema, and machine plan.
- `sha256sum -c` passed for `EXPERIMENT_PLAN_HASHES.sha256`, `M0_RUN2_V2_PLAN_AMENDMENT_HASHES.sha256`, and `M0_RUN2_V2_PLAN_AMENDMENT_REVIEW_HASHES.sha256`.
- Config-bound authoritative input hashes and Run1 frozen hashes passed `sha256sum -c`; validation/test provenance paths were listed only as declared-not-opened hashes.
- `bash -n` passed for wrapper and sbatch; no `--time` directive was found.
- `git diff --check` passed; `git diff --check --no-index /dev/null <touched-file>` produced no whitespace diagnostics.
- `artifacts/lb_scgp_global/v2` remains absent.
- Static package metadata search did not prove `jsonschema` availability; future SLURM preflight now fails closed if unavailable.

## Remaining Risks

- No Python execution was performed, so numerical/runtime behavior is not claimed.
- If a future reviewer decides `G0=I_N` was a frozen scientific semantic rather than a flawed implementation choice, H5 cannot be closed under `N=10,d=3,coordinate trust=0.02`; the minimal amendment would be to authorize the unit-diagonal local projection baseline used here or change one of those frozen constraints.
- `jsonschema` installation in `HateVideo` remains unproven statically; the future single SLURM validator will fail closed before publish if it is missing.
- Fresh independent implementation/code review with 0 Critical / 0 High is still required before any execution-authorization request.

## Required Statements

- No performance evidence exists and no performance claim is made.
- Only project gold is `parent_video_binary_label`. There is no segment/frame/timestamp/span/localization/stance/target/mechanism/rationale/fragment gold, and none is introduced here.
- Run3, M1, MLLM/cache work, validation/test work, training, and realbank remain locked.
- Execution is unauthorized; no SLURM submission is authorized by this report.

Report SHA256 must be computed externally after this file is written; it is not embedded here.
