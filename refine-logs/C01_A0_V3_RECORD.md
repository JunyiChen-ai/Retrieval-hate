# C01 A0 v3 prospective implementation record

**Status:** `V3_READY_NOT_RUN_PENDING_REVIEW`

This record is prospective only. No Python execution, SLURM submission, result,
decision, metric, CONTINUE/KILL verdict, A0 retry, C02 action, or test access is
authorized or claimed.

## Frozen lineage

- Run/schema: `C01-A0-v3`, `c01_a0_result_v3`,
  `c01_a0_decision_v3`.
- Config: `configs/c01/c01_a0_v3.json`, SHA256
  `4ddb0f6f322de06316ea014a77c732b1a593c0fae5d926558d6c64a1be21cda5`.
- Analysis: `scripts/analysis/c01_policy_contrast_a0_v3.py`, SHA256
  `40b35eee2fb6fdbdb21fe9b4acfdcebf003c121c76492b898fbd2ea9b8c34dfb`.
- Wrapper: `scripts/slurm/c01_a0_cpu_v3.sbatch`, SHA256
  `e61b99620622d4161e0baded335e6172bd55b606e425668843cd9d370489af99`.
  Before activating the environment or invoking Python, the wrapper
  exact-checks the frozen v3 analysis and config SHA256 values and fails closed
  on either drift.
- Exclusive namespace:
  `artifacts/c01_policy_contrastive/v3/a0/C01-A0-v3`.
- The v3 config exact-binds frozen v2 config SHA256
  `f3997bddb4788d451ae5f90d9d03d096df3de383f8133a6d3818d97a241563f5`
  and frozen v2 analysis SHA256
  `d2b9c2ff909c07518ae35526db9550df655fb4af395cc7a0899f83e48db1b855`.
  Runtime first validates the complete v2 canonical config, then permits only
  new run/schema/namespace/output identity and HALT-only numerical-equivalence
  names. Any other v2→v3 field difference fails closed.
- The latest guard review is exact-bound as `TARGET_REVIEW_RAW.md` lines
  1165–1338, section SHA256
  `2f4a1b1ff3fdaff176bca52a1a7a0940e949ee8ac75e51d7ac18b9883d28fe54`.
  Later append-only independent reviews do not mutate this frozen section.
- Retrieval diagnostic job `13732` artifact SHA256
  `724c87cd2fbdb763180b663bc6492322887bc2077f378c5b21c4184c4ba80e6f`,
  the approved manifest, zero probe, authorized tuple, exact A0 source/key
  construction, cache hashes and 8-thread CPU environment are exact-bound.

V1/v2 analysis, configs, wrappers, namespaces and job records are not modified
or reused. Job `13730` remains the frozen v2 HALT.
Static freeze hashes at preparation are v1 config
`06368cf1e0693d491ef4e511fe22ffe02c4bbc25ae1b632026f49473f068659b`,
v1 wrapper
`0eb22d3ca072ef753afc097e1ba52ad803c009ff776c578a58ea9d90eb1562ba`,
v2 config
`f3997bddb4788d451ae5f90d9d03d096df3de383f8133a6d3818d97a241563f5`,
v2 wrapper
`2db9f22497f03c977980db3db73be3424e2c5bdf84cf0cf0331a2d2ab1c41092`,
and the shared frozen analysis hash stated above.

The provenance chain is acyclic: v3 source pins v3 config plus frozen
v2/review-section/diagnostic evidence; the wrapper pins v3 source and config;
this immutable record pins the wrapper. TARGET state pins this record after its
final bytes are frozen. Neither source nor config attempts to pin its own hash.

## V3 numerical-equivalence contract

Before applying any tolerance, v3 requires byte-exact raw retained keys,
normalized retained memory and normalized queries. The remove path reuses the
single frozen float32 normalization result. Mapping, raw FAISS neighbors,
deterministic `(-float32 similarity, original index)` neighbors, per-query
sets/order, neighbor labels, prediction bytes, canonical typed
accuracy/macro-F1/ROC-AUC bytes and their deterministic scientific-boolean basis
remain exact. The registered null top-20 count must be zero. NaN, either
infinity, malformed indices, nonfinite bounds and cross-path signed-zero
ambiguity are forbidden.

The only bounded fields are finite similarities and scores. With actual arm
dimension `d`, `u32=2^-24`, and
`gamma32(d)=d*u32/(1-d*u32)`, v3 computes an upward-safe binary64 norm audit and
`rho=max ||x_r||_2 ||q||_2`, then freezes

`B_sim(d,rho)=2^ceil(log2(2*gamma32(d)*rho))`.

No observed job-13732 difference appears in this formula or config. Every
similarity must pass both the absolute bound and the exponent-aware ordered-u32
bound

`max(ord32(upper32(a+B))-ord32(a), ord32(a)-ord32(lower32(a-B)))`,

with outward float32 rounding and finite clipping. Each audit persists the
derivation, observed maximum absolute/ULP difference, maximum allowed ULP, and
observed/bound ratios.

For frozen top-20 weights `20,...,1`, `W=210`, each query uses

`B_score(q)=(1/210) sum_r w_r |Delta a_qr| + 2^-45`.

It also requires the arm sanity bound `B_sim+2^-45`. Closed intervals expanding
either score by `B_score(q)` must not contain or cross cutoff zero, even when the
two observed prediction bits agree. The existing `avg_score` control uses the
prospective source-propagation formula frozen in the v3 config.

Any violation raises `HALT_NUMERICAL_EQUIVALENCE` before result/decision
publication.

The FAISS/binary64 relationship additionally requires the similarity and score
zero-position masks to match exactly and the signbit at every shared zero
position to match exactly. A `+0.0/-0.0` discrepancy is an ambiguity and HALTs;
NaN and either infinity remain forbidden.

## Deterministic binary64 exact-neighbor reference

Every real arm, fixed rotation, and each of 256 fixed-null shuffle retrievals is
re-scored using query chunks of 8, C-order float64 multiplication, and
`numpy.add.reduce` over the feature axis in frozen neighbor rank order. It uses
the same 8-CPU, `OMP/MKL/OPENBLAS/NUMEXPR=8` environment with fixed
NumPy/Torch/FAISS versions and CPU identity.

This is a **rescore-only** ablation: it consumes the exact agreed float32
top-20 original IDs and never mines or replaces neighbors. Candidate selection
is instead protected by the exact raw/stable-neighbor guards. Reference
similarities/scores are a single shared-operand computation and therefore exact
between with/remove paths; reference predictions and canonical metrics must be
exact, and FAISS-with/reference finite differences must lie within the same
prospective float32 envelope. No result-dependent algorithm, thread, chunk-size
or tolerance selection is permitted.

## Static NO-GO repair closure

- The v2→v3 diff whitelist includes both numerical-comparison schema strings,
  including `exact_metric_comparison`; the SHA-frozen scientific config remains
  the only source of scientific/statistical fields.
- The v3 authorized-null tuple is normalized and compared field-by-field with
  the complete frozen v2 zero tuple. Every diagnostic field is also compared
  semantically; its intentionally standard-only policy scope must equal
  `standard` and belong to the frozen `{standard,oneword}` authorization.
- Decision validation now rechecks run/config/result/decision schema,
  manifest/zero/lineage provenance, v2 scientific-base hashes and diff
  whitelist, result filename/SHA type, dataset booleans, `continue` type and
  label, interpretation scope, small-displacement references/rules/role,
  exclusive creation, the complete v3 numerical contract, and every HALT-only
  validity guard.
- Public contract-summary booleans are recomputed from loaded runtime cache
  shapes/types/values, ID/label alignment, zero/derived audits, retrieval and
  shuffle counters, binary64/scientific-basis checks and the access ledger.
  V3 does not overwrite these summaries with literal success values.

## Unchanged science

R0/history parity, ordinary/primary/secondary arms, six rotations, 256
fixed-null shuffles, 2000 paired bootstraps, Holm families, shuffle p95,
small-displacement rules, `+0.020/+0.020`, deployed-R0 accuracy guard and
MHC-ZH/HateMM net-fix minima remain exactly the SHA-frozen v2 settings. V3 adds
only HALT-only validity guards and cannot help a scientific CONTINUE.

## Execution boundary

The wrapper matches v2 job `13730`: CPU-only, 8 CPU, 32 GB,
`OMP/MKL/OPENBLAS/NUMEXPR=8`, conda `HateVideo`, no GPU and no `--time`.
Execution requires fresh independent static review and separate explicit
authorization.
