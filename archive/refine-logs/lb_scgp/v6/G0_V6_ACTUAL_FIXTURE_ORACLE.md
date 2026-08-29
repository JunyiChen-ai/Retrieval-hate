# G0 v6 Actual Fixture Oracle Record

Thread/session: fresh exact GPT-5.5 xhigh v6 ACTUAL-FIXTURE ORACLE
IMPLEMENTER in `/data/jehc223/RGCL`; no subagents.

Boundary: job `12846` is read-only and must continue naturally. This record is
prospective v6-only evidence under `refine-logs/lb_scgp/v6/`; it is not a G0
PASS, freeze, formal gate, realfold run, replay decision, or performance claim.

## Frozen 589-Set Map

| Ordinal range | Count | Frozen set formula |
|---:|---:|---|
| 000 | 1 | `symmetry` |
| 001 | 1 | `correlation_diagonal` |
| 002 | 1 | `psd_symmetrized_input` |
| 003 | 1 | `offdiagonal_box` |
| 004-027 | 24 | `row_trust_00` ... `row_trust_23` |
| 028-029 | 2 | `class_mean_trust_0`, `class_mean_trust_1` |
| 030 | 1 | `semantic_radius_zero` |
| 031-032 | 2 | `slack_capped_simplex_0`, `slack_capped_simplex_1` |
| 033-056 | 24 | `vote_slack_00` ... `vote_slack_23` |
| 057-058 | 2 | `class_mean_margin_0`, `class_mean_margin_1` |
| 059 | 1 | `global_mean_margin` |
| 060 | 1 | `centroid_distance` |
| 061-588 | 528 | Per query `q=0..23`: 19 `rank_internal_{22q..22q+18}` then 3 `rank_boundary_{22q+19..22q+21}` |

Total: `589`. Rank halfspaces: `528 = 24 * (19 internal + 3 boundary)`.

## Execution Ledger

All Python/import/solver/replay work below ran through SLURM after
`conda activate HateVideo`. No `--time` directive was used.

| Job | Role | State | Exit | Elapsed | Resources | MaxRSS | Result |
|---:|---|---|---|---:|---|---:|---|
| 12874 | fresh validator | COMPLETED | 0:0 | 00:00:03 | 2 CPU / 4G | 3284K | `actual_fixture_validation_12874.json`: OK |
| 12875 | oracle+replay, first run | COMPLETED | 0:0 | 00:25:38 | 4 CPU / 24G | 175196K | `NO_WITNESS`, replay `REPLAY_OK_BOUNDED_REMOVE` |
| 12882 | repaired validator | COMPLETED | 0:0 | 00:00:03 | 2 CPU / 4G | 3340K | `actual_fixture_validation_12882.json`: OK |
| 12883 | repaired oracle+replay | COMPLETED | 0:0 | 00:41:34 | 4 CPU / 24G | 247032K | `BOUNDED_REMOVE`, replay `REPLAY_OK_BOUNDED_REMOVE` |

Protected job `12846` remained read-only. Last inspected state: RUNNING,
`lbscgp_v6_cert_slsqp`, 4 CPU / 24G.

## Inputs And Hashes

- v5 config:
  `a51981045073e8f5b69da272654d2102ef3f2f5c8739b765d0b161c1f8c75346`
- v5 dykstra JSONL:
  `c28090bbd26da0d6ba89ca67340355c8cebc6e24e20899954282dfeed02a92f6`
- v5 freeze:
  `254e45afe9c0355892824c0c26bc73b4b0854cb20c67c3703982762fad010931`
- accepted 12866 witness:
  `f9d47eaac29d0ee02102c8bd2989f4c02f10a23ea04f07a103cc111dad805eff`
- accepted 12866 replay:
  `288dc2b7f2eecb31e1f23d63ba48e518190e2f25a63dfb4a0ec6f1e0cb0867d1`
- repaired oracle source:
  `f37ea1b68d59125876f6ee29ea06daa11c42e028736731071fad22375c3f82f4`
- replay source:
  `1ba402ab97c7a020819d0051bf230c8fada99f0ae12347cd253c70d195ac16f5`

## Phase I

Frozen binding checks: `constraint_set_count=589`,
`set_order_matches_frozen=true`, orientation rank `1`, compatible cells `2`.
Accepted 12866 was used only as an input witness: rank `9`, witness hash
`6f0f676dc7a2066a6ccb51507a2a24bb90e14c9ab961445507ca5a48c9a86122`.

Run `12875` preserved a search failure: both cells found positive eig margins
and small residuals, but neither realized the target top20 cell, so the result
was `NO_WITNESS` and replayed as bounded remove.

Repair after `12875`: kept the explicit max-eigenvalue-margin attempts, then
added a minimal-perturbation anchor polish at fixed positive eig floors. No
threshold, fixture, expected status, set order, or supervision contract was
relaxed.

Run `12883`:

| Cell | Phase-I status | Best accepted margin | Max 589 residual | Top20 |
|---:|---|---:|---:|---|
| 0 | `FULL_RANK_SLATER_REPLAY_PENDING` | `9.999721969433045e-7` | `3.552713678800501e-15` | matched |
| 1 | `NO_WITNESS` | no accepted witness | best residual `5.838672220806777e-17` in an anchor attempt, but top20 mismatched | mismatched |

Independent replay for cell 0 recomputed `full_rank_replay_ok=true`,
`psd_min_eigenvalue=9.999721969433045e-7`, and mpmath selected residual
`3.552713678800501e-15`. Cell 1 remains `NO_WITNESS`/nonconvergence only.

## Phase II

Phase II was attempted only for cell 0 because cell 1 did not have a
top20-realizing full-rank Slater witness. The cell-0 direct Gram optimization
used upper-triangle Gram variables plus `xi`, analytic gradients/Jacobians,
linear/SOC constraints, and a minimum-eigenvalue PSD constraint.

Cell 0 Phase-II result:

- status: `BOUNDED_REMOVE`
- objective: `1.630798945779586e-05`
- max 589 residual: `5.8836793827797916e-09`
- mpmath max selected residual: `5.883679784762077e-09`
- PSD minimum eigenvalue: `-5.8836793827797916e-09`
- realized top20 equal target: `false`
- stationarity infinity norm: `0.0022782803493536386`
- dual min: `4.3995421488138975e-52`
- complementarity infinity norm: `8.21147387671932e-08`
- VI residual bound from KKT fields: `0.002278362464092406`
- PSD active dual reconstruction: `mu=0.003378569870158688`,
  `S_psd` min eigenvalue `-2.970847666162227e-19`,
  `||SG||=1.987842301829254e-11`,
  `|trace(SG)|=1.9878422904390084e-11`

Although the scalar PSD dual reconstruction was numerically PSD/complementary,
stationarity and top20 coverage failed, so `LOCAL_STATIONARY_CERTIFIED` is not
supportable.

## Replay Boundary

`actual_fixture_replay_12883.json` did not import solver code. It rebound the
frozen fixture/config/v5 hashes, recomputed objective, residual groups, top20
coverage, mpmath PSD/residual checks, KKT fields from stored multipliers,
source hashes, and the no-segment supervision contract. Replay status:
`REPLAY_OK_BOUNDED_REMOVE`; payload
`4ad21907d3a6dcf30206fe21a8e4d1a7c58d0b472d490d01124477d4d6eb3a73`.

## Limitations

- Cell 1 has no accepted full-rank Slater witness in this search; this is
  `NO_WITNESS`/nonconvergence only, never an infeasibility proof.
- Cell 0 has a replayed full-rank Phase-I witness, but its Phase-II optimum
  attempt failed top20 and stationarity checks.
- No feasibility-only or factor-space stationarity result is presented as a
  Gram optimum.
- The next independent-audit boundary is a separate review of the top20 cell
  realization geometry and the active PSD/stationarity system before any
  certification claim.

## Non-Claims

Only gold supervision is `parent_video_binary_label`.
`segment_gold_exists=false`; `segment_gold_used=false`; all MLLM/OCR/teacher
cache/outer-held/validation/test counters must remain zero.
