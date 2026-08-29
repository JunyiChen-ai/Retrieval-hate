# LB-SCGP v6 Numerical Certificate Review

## Determination

The v6 numerical run did not find a LOCAL witness for the frozen
`feasible_oriented_boundary` top-20 cell system. This is a numerical
non-witness result, not a proof of nonexistence: SciPy SLSQP and mpmath replay
show all executed paths remain infeasible above the frozen `1e-6` residual
contract, and no conic/proof-grade infeasibility certificate is available.

This report authorizes no downstream synthetic gate, realfold, replay,
decision, G1, teacher, MLLM, OCR, held, validation, test, training, accuracy,
or macro-F1 claim.

## Protocol Boundary

Only `parent_video_binary_label` is gold. `segment_gold_exists=false` and
`segment_gold_used=false`. MLLM, OCR, teacher/cache, held, validation, and test
counters are all zero in the v6 machine outputs.

Job `12840` is invalid. It was cancelled by a previous reviewer, has no
usable numerical evidence, and is not cited here for feasibility,
infeasibility, LOCAL, BOUNDED, or solver performance.

The machine-readable job manifest is
`refine-logs/lb_scgp/v6/G0_V6_JOB_MANIFEST.json`. It records jobs
`12837` through `12843`, new jobs `12845`, `12846`, `12888`, and `12891`,
commands, states, resources, hashes, and the explicit invalidation of `12840`.

## Formal Object

Identity: the invariant object is the final top-20 exact-vote-safe full-bank
Gram projection for the frozen `feasible_oriented_boundary` synthetic fixture.

Variables: `G in R^{24x24}` is the projected Gram matrix, symmetric with unit
diagonal; `xi in R^{24}` are nonnegative vote-slack variables. The Euclidean
objective is the squared Frobenius displacement from frozen `gram0` plus the
slack penalty used by the v6 numerical certificate.

Constraints: PSD, symmetry, unit diagonal, off-diagonal box, row trust,
class-mean trust, semantic radius, vote-slack nonnegativity and budgets,
class-global margin inequalities, centroid/internal constraints, the 19
internal top-20 rank halfspaces per query, every 20th-vs-all-self-excluded
outsider halfspace, canonical tie rules, and compatible adjacent orientation
cells. The frozen v5 Dykstra object has `n=24`, `topk=20`, `set_count=589`,
semantic shape `1x576`, and labels `[0]*12 + [1]*12`.

Proposition under test: a LOCAL certificate exists if at least one compatible
final-top-20 cell has an independently replayed feasible primal witness with
max residual `<=1e-6`, respecting the frozen `1e-7` tie/relative contracts and
the canonical top-20 controller.

Approximation: v6 used floating-point SciPy SLSQP with mpmath 80-digit
selected residual replay. This is an empirical numerical certificate attempt,
not an exact conic proof.

Interpretation: absence of a feasible replayed witness means LOCAL is not
numerically certified here. It does not prove infeasibility.

## Controller Schema

The v6 controller separates `final_top20_rankings` from
`full_outsider_order_for_enumeration`; full `n-1` order is only an enumeration
aid. Self exclusion is verified. Canonical-ID ties use `tie_tolerance=1e-7`;
rank RHS is `-tol` when `id_a < id_b` and `nextafter(tol,+inf)` otherwise.

Controller hashes from job `12846`:

- compatible cells:
  `19ff753175a956aaf836ee7da132a4a42788d95f2f0d1a6d8fa726a62108c14d`
- reference top-20:
  `f88c7ec0e505f1b5b3ea8464836065fd386df4452acee6d8f823f377dbffb9d1`
- reference full outsider order:
  `a8fe38cbb48d43db90f9f57918c99ed64ed9337caafc46c9b707c03f4633395a`
- reference baseline margins:
  `11f9f929e6c95ccc099c94d12e5eefa5e23a9fc7d1ceb17484714da2ad9230d2`

Compatible adjacent cells: 2 cells, assignments `[-1]` and `[1]`, rank 1
orientation descriptor `["p00","p15","p18"]`. Each cell has 456 internal
rank halfspaces and 72 boundary halfspaces, for 528 rank halfspaces.

Independent verifier obligations are to recompute stable top-20 with
canonical ties, verify all 19 internal inequalities per query, verify every
20th-vs-outsider inequality, rebuild compatible adjacent orientations from
full outsider order, verify objective and residuals without producer imports,
and verify the no-segment-gold/no-MLLM/OCR/teacher/held/val/test boundary.

## V5 Matrix

From `G0_V5_RESULT_TO_CLAIM_REVIEW.md`, v5 synthetic results were partial and
fail-closed. `feasible_oriented_boundary` expected
`LOCAL_STATIONARY_CERTIFIED` but emitted `BOUNDED_SEARCH_FEASIBLE`, exhausted
`500/500` cycles, had `max_set_violation=6.752627108152793e-6`,
`relative_iterate_change=5.1725509722744595e-8`, unstable top-20 cell,
`search_reason=base_cell_not_converged`, 1 independent orientation, 1/1
adjacent checked, and 0 pivots.

The v6 case matrix registered 17 adversarial/negative cases. Only
`one_boundary_known_LOCAL` was executed by the oriented certificate path; all
other cases are registered-only controls and are not execution evidence:
top20-stable ranks 21-to-N shuffled, zero-orientation stable true/false,
one-boundary BOUNDED/nonlocal, just-below/above `1e-6`, relative-change
without feasibility, canonical-ID tie below/at/above `1e-7`, duplicate-ID
tie-map removal, orientation-over-budget removal, pivot-over-budget removal,
PSD/unit/box/trust stress, no-segment manifest negative, and zero-counter
manifest negative.

## Executed Paths

Environment evidence: job `12838` found NumPy `1.26.4`, SciPy `1.17.1`, and
mpmath `1.3.0`; cvxpy and conic solvers were unavailable. Therefore v6 makes
no conic-proof claim.

Job `12846` ran two deterministic SciPy SLSQP starts over two compatible
cells. Options: SLSQP, `ftol=1e-12`, `maxiter=3000`, conda `HateVideo`.
Replay job `12888` independently matched all objectives, residuals, and
witness hashes.

| cell | path | status | objective | max residual | mpmath max selected | top20/full cell | KKT/stationarity proxy |
|---:|---|---|---:|---:|---:|---|---|
| 0 | frozen_initial | nonconverged | `1.5085751224436118e-5` | `7.08996347631449e-5` | `7.08996347630972e-5` | false/false | multipliers only; max abs `0.012261563836223375`, L2 `0.023257586365688955` |
| 0 | spectral_affine | nonconverged | `1.587069503611052e-5` | `2.8708075211567823e-5` | `2.870807521168476e-5` | true/true | multipliers only; max abs `0.02538382422289908`, L2 `0.0410499654725572` |
| 1 | frozen_initial | nonconverged | `5.083264000834039e-5` | `3.6357995540520156e-5` | `3.6357995540593604e-5` | false/false | multipliers only; max abs `0.05569884183847239`, L2 `0.0857999403260439` |
| 1 | spectral_affine | nonconverged | `5.130087590297527e-5` | `3.3575960168320655e-5` | `3.357596016827562e-5` | false/false | multipliers only; max abs `0.04663932077432124`, L2 `0.07612878750246349` |

The active residual in every path is PSD. For the only path that preserved the
target top-20 and full cell (`cell=0`, `spectral_affine`), PSD residual was
`2.8708075211567823e-5`, above `1e-6`. It is therefore not a feasible LOCAL
witness. KKT quality is weak: SciPy multipliers were serialized, but no
proof-grade stationarity certificate was produced.

Replay result: `feasible_replay_count=0`,
`local_feasible_cell_exists_replayed=false`, certificate payload OK, and
replay row count 4.

## Hashes

- certificate JSON:
  `5c824f0d45c5ef3cdfbc25fcd927c077bec5c6c9cdb7425c2333bf89750245e5`
- certificate payload:
  `7d284014c6570ce942bcf92c593b7ef373d02831682597d80a1c798daeef83fb`
- replay JSON:
  `44a0e1479d63b8b26e0d706db7ab097767d363fdc248f66ca8b6966145d3045f`
- replay payload:
  `da66805b40326144545c328391843f3cf5c30bd9cdbe55927d94281303ed3ab2`
- controller JSON:
  `ba8ddb0057697a765a33c2e817ee294dcc94050036e8cc8bb928890eff182565`
- case matrix JSON:
  `f6369d940ed55116d701b8e8f5bd0098ec87716599dcd38dc46477763a18a6d6`
- v6 runtime `oriented_certificate.py`:
  `46d28451ca1feabd51770de03bbb9435306d7dcdf3a647ba339002a4f3fc6ec9`
- v6 replay runtime:
  `f40e179dcc8476daf9f280c1fd463677f4705c6de190c716b1c97734ed34be35`
- v5 synthetic manifest:
  `07dc7d5d17194cd7a2b5d42d539adb9e8248e78b4dc629bbcdaf9d4f64719242`
- v5 Dykstra:
  `c28090bbd26da0d6ba89ca67340355c8cebc6e24e20899954282dfeed02a92f6`

## Recommendation

Do not mark v6 PASS and do not claim LOCAL from this evidence. The correct
state is indeterminate/non-certified: no feasible witness was found, but no
proof-grade infeasibility certificate exists.

For a future v6 executor, the minimal auditable route is:

```text
for each compatible final-top20 cell:
  build constraints from frozen gram0, parent-video labels, and canonical ids
  keep baseline margin/slack budgets tied to the frozen reference top20
  solve feasibility phase first with analytic Jacobians and PSD residual target
  then solve objective refinement only after max residual <= 1e-6
  checkpoint every completed path with G, xi, objective, residuals, and hashes
  replay serialized G, xi independently with mpmath selected residuals
  accept LOCAL only if replay max residual <= 1e-6, relative/tie rules hold,
    top20/full cell obligations hold, and supervision counters are zero
```

Stopping rules must preserve the unchanged `500` cycle, `1e-6`, and `1e-7`
contract. A failed or single SLSQP run cannot prove nonexistence; repeated
non-witness runs should be reported as indeterminate unless paired with a
valid infeasibility proof.

## Residual Risk

The strongest v6 path preserves the target cell but remains PSD-infeasible by
about `2.87e-5`. Other paths fail cell stability and PSD feasibility. The
controller/case matrix is correctly shaped for the final top-20 distinction,
but most adversarial cases are registered-only and still need execution before
they can support a robust implementation claim. No downstream gate is
authorized.
