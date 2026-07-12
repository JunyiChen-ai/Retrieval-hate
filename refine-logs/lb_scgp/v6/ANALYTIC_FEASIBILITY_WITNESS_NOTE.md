# v6 Analytic Feasibility Witness Supplement Note

Thread/session: fresh DESIGN+IMPLEMENTATION worker in `/data/jehc223/RGCL`, no subagents.

Scope: prospective v6-only supplement under `refine-logs/lb_scgp/v6/`. Existing method, config, freeze, threshold, and prior evidence artifacts are read-only inputs. Job `12846` is inspected only through read-only SLURM status/log/checkpoint commands and is not cancelled, signalled, requeued, released, held, suspended, or otherwise modified.

No-segment-gold contract: the only gold supervision is `parent_video_binary_label`; `segment_gold_exists=false`; `segment_gold_used=false`; all MLLM/OCR/teacher/cache/held/val/test counters must remain zero. The prospective scripts validate this contract from the frozen v5 config before solver work.

Equations and parameterization:
- Variables are a factor `F in R^{n x r}` and slack `xi in R^n`.
- The Gram matrix is `G = F F^T`; PSD and symmetry are exact by construction.
- Unit diagonal is enforced as analytic equality residual `diag(G)-1=0`.
- Semantic zero uses `semantic @ vec_C(G)=0`.
- Inequalities cover off-diagonal box, row trust balls, class-mean trust balls, nonnegative/budgeted slack, vote slack, class/global margin, centroid, and oriented rank halfspaces.

Analytic Jacobians and shapes:
- For any scalar residual with full-Gram gradient `A = d f / dG`, the factor Jacobian is `(A + A^T) F`.
- Linear equality and inequality blocks are vectorized as tensors `A[m,n,n]` plus `xi` coefficients; Jacobian shape is `(m, n*r+n)`.
- Row-trust and class-mean-trust norm residuals use analytic norm gradients, then the same factor-chain rule.
- Validator `analytic_feasibility_witness.py --self-check` checks equality/inequality Jacobian shapes and deterministic directional finite-difference errors under SLURM.

Scaling and objective:
- The solver objective is feasibility-first: `0.5*||scaled equalities||^2 + 0.5*||negative scaled inequalities||^2`.
- This is explicitly distinct from the frozen scientific projection objective; the supplement does not modify frozen scientific search semantics or thresholds.

Deterministic starts:
- Rank schedule is deterministic from the frozen `gram0` positive spectrum plus full rank.
- Starts are deterministic: spectral frozen `g0`, frozen Dykstra result if serialized, spectral semantic-affine warm start, and fixed-seed factor jitters.

Checkpointing and terminal behavior:
- Each multistart appends a JSONL checkpoint under `refine-logs/lb_scgp/v6/results/analytic_feasibility_checkpoint_<job>.jsonl`.
- Solver scripts have finite iteration caps and no SLURM `--time`; they terminate naturally.
- Resource requests are conservative: validator `1 CPU/4G`, solver `4 CPU/24G`.

Replay and labels:
- The solver may only emit `np_primal_feasible_candidate_pending_independent_replay` or `nonconverged_no_feasible_witness`; nonconvergence is never labelled infeasibility.
- `analytic_witness_replay.py` independently replays the serialized witness with NumPy and mpmath at 100 dps. Acceptance label `accepted_feasible_replayed` requires payload hash validity, no-segment-gold contract, witness hash match, NumPy residual/top-20 pass, and mpmath selected residual pass.
- KKT/stationarity diagnostics are reported separately as feasibility-penalty/bound stationarity only and are not used for feasibility acceptance.

Run order and gates:
- Gate 1: submit `runtime/validate_analytic_v6.sbatch`; inspect natural terminal output and `results/validation_analytic_<job>.json`.
- Gate 2: submit `runtime/analytic_feasibility_witness.sbatch` only if Gate 1 passes.
- Gate 3: inspect solver checkpoint, solver JSON, independent replay JSON, log stderr/stdout, hashes, and SLURM terminal state.

Initial slow-path audit of job `12846`:
- Read-only status observed: `RUNNING`, job name `lbscgp_v6_cert_slsqp`, 4 CPUs, 24G, unlimited time, node `foscsmlprd01`.
- Read-only checkpoint observed: first path `scipy_slsqp_frozen_initial` on cell 0 ended `nonconverged` after about 5844 seconds with max residual about `7.09e-05`; this is slow/nonconvergent local-solving evidence only, not infeasibility.

Machine run fields:
- Validator job id: `12865`; natural terminal state `COMPLETED`, exit `0:0`, elapsed `00:00:02`.
- Solver/replay job id: `12866`; natural terminal state `COMPLETED`, exit `0:0`, elapsed `00:00:07`.
- Validator gate result: `status=OK`, payload hash `dc7388265ae6878a286d7c54ada592250c8589118fc6f83d2dbe1f1054f2bee8`, file hash `7cdbab95b51a5c8554587bd044ebe69f5e844bbf7e65b67b606324dab08c7671`.
- Validator self-check: contract OK; cell count `2`; orientation rank `1`; first reduced rank `9`; factor shape `[24,9]`; variable shape `[240]`; equality Jacobian shape `[25,240]`; inequality Jacobian shape `[1160,240]`; directional equality Jacobian error `1.52e-09`; directional inequality Jacobian error `3.89e-10`; objective gradient directional error `2.86e-08`.
- Solver source hashes used by validator and result: `analytic_feasibility_witness.py=cbdb498ba59cf4eb1b156533d204653e9b70cfbb574cdd9cd94d6f8c7892a672`; `analytic_witness_replay.py=87e8386167f02cf1e9a1da3c658b10981f27a4de249751482be5372b18e34ed2`; `validate_analytic_v6.py=e266ca4367fce35c1c6232f772aa4141eb4d2b62adbfd1ade41a7818863af2a0`; `analytic_feasibility_witness.sbatch=f8a248c13e10d2a26ad2fe83330fcbceda79a7f3c77621adf7ef93be9fe2b21e`; `validate_analytic_v6.sbatch=9cdbf134a9a479d9f7bd374f7c8f0871161967fcbd98818ea143bdd90ebdb640`.
- Solver result: `refine-logs/lb_scgp/v6/results/analytic_feasibility_witness_12866.json`, file hash `f9d47eaac29d0ee02102c8bd2989f4c02f10a23ea04f07a103cc111dad805eff`, payload hash `c4929fba9e58e59e0119c6591a293fac898a3684ffbd15758a0c6b17934e8069`.
- Solver checkpoint: `refine-logs/lb_scgp/v6/results/analytic_feasibility_checkpoint_12866.jsonl`, file hash `0d35bad4722e066bc44f84598b1ee9dcaac7a61a651339173b5224a9327940d2`.
- Independent replay result: `refine-logs/lb_scgp/v6/results/analytic_witness_replay_12866.json`, file hash `288dc2b7f2eecb31e1f23d63ba48e518190e2f25a63dfb4a0ec6f1e0cb0867d1`, payload hash `990d7c63fa5e8ba7e81d92eec4acbd003b6e9e4281e8e4ca5ea6b51629509bd9`.
- Solver/replay log hashes: stdout `9e70affac6edabc27f40eadccf9d01c04ea597301587a77c4c58cf31811f0a7e`; stderr `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Accepted numerical witness summary:
- Solver label before replay: `np_primal_feasible_candidate_pending_independent_replay`.
- Replay acceptance label: `accepted_feasible_replayed`.
- Selected cell/rank/start: cell `0`, rank `9`, start `spectral_frozen_g0`.
- Witness hash: `6f0f676dc7a2066a6ccb51507a2a24bb90e14c9ab961445507ca5a48c9a86122`; replay hash matched.
- NumPy replay max residual: `1.7763568394002505e-14`.
- mpmath replay selected max residual: `1.1391406924925865e-14` at `100` dps, PSD check via `mpmath_eigsy_100dps`.
- Top-20 target and realized hash: `77c3a833ef562a32e05930cbc145675e174a378d88b34290fdf643b2757af5c3`.
- KKT/stationarity diagnostic, reported separately: feasibility-penalty objective `8.93e-31`, gradient infinity norm `1.14e-15`, projected-gradient infinity norm `1.14e-15`; not used for acceptance.

Limitations:
- This is an accepted numerical feasibility witness with independent high-precision replay, not a mathematical proof of infeasibility or a proof-grade certificate.
- If a future run fails to find a witness, that must remain labelled nonconvergence/no-witness, not infeasibility.
- The original slow-path job `12846` remains running naturally and was not modified.
