# M0 Run2-v3 Fresh Static Code Review

Date: 2026-07-13

Reviewer: **Claude Opus 4.8**, fresh, zero-context, zero-history (0C/0H) independent static
code reviewer for the `lb_scgp_global_r2` M0 Run2 **v3** lineage.

## Reviewer boundary

Fresh 0C/0H independent static code review only. I did **not** participate in any prior round
(v1, v2, or the v3 setup/freeze); every ruling below is grounded in files I read directly in
this session, not in any prior-round conclusion. I explicitly re-ran the clone-equivalence proof
myself rather than copying the freeze document's tables. I did **not** run Python, Python
imports, `py_compile`, tests, `conda`, SLURM, `sbatch`, `squeue`, experiments, MLLM/OCR/API/
model/network/GPU/training/evaluation, or validation/test data/cache inspection. Shell was
limited to the allowed static tools: `rg`/`grep`, `sed`/`nl`, `jq`, `awk`, `bash -n`, `diff`,
`sha256sum`, `find`, `ls`, `wc`, `git status`, `git diff`. The only file I wrote is this report.
No artifact under `artifacts/lb_scgp_global/v3` was created (the directory is confirmed absent).

This review does **not** authorize SLURM execution. A PASS permits only entry into a separate
independent execution-authorization step (verdict §4(d.3)) for exactly one future CPU-only SLURM
validation job.

### Model-binding divergence declaration (precedent: `M0_RUN2_V2_CODE_REVIEW_FIX2.md`)

`AGENTS.md:15` binds the main-dialogue subagent to **"GPT-5.5 xhigh"**. That backend is not
available for this session's subagent, so this review runs on the `CLAUDE.md`-bound **Opus 4.8**
(`claude-opus-4-8`). This is a documented model-binding divergence between `AGENTS.md` and
`CLAUDE.md`, recorded here for transparency; it is a documentation/process fact, not a code
defect, and does not affect any ruling below.

---

## Verdict

**PASS_STATIC_REVIEW**

Severity counts:

| Severity | Count |
|---|---:|
| Critical | 0 |
| High | 0 |
| Medium | 2 (M-A, M-B) |
| Low | 3 (L-A, L-B, L-C) |

Critical = 0 and High = 0 → the static-review pass criterion is met. Medium/Low items do not
block but must be understood before execution authorization. Two conditional/attention notes are
attached and highlighted for the execution authorizer (see §7):

- **M-A** carries a live conditional escalation: if a science authority rules that the synthetic
  fixture's `G0` must itself be a realizable rank-`<=d` PSD `Z0 Z0^T`, M-A becomes **High** and
  blocks. No such ruling has been issued, so it remains Medium here.
- **M-B** is fail-closed but, unlike the now-resolved `jsonschema` residual, is **not dissolvable
  by any allowed static means**; it is the single most material residual carried into the
  single-submit attempt and the authorizer must accept its convergence-burn risk consciously.

The former v2 killer **M-C (`jsonschema` availability)** is now **RESOLVED** (installed +
independently confirmed present by read-only site-packages listing); its only residual is the
environment-not-frozen re-confirmation duty, recorded as **L-C**.

---

## 1. Clone equivalence — independently re-proven (not copied from the freeze doc)

### 1.1 SHA256 manifest (both lineages re-hashed here)

All nine v3 entities hash **exactly** to `M0_RUN2_V3_CLONE_FREEZE.md` §1, and all nine v2
source entities still hash exactly to `M0_RUN2_V2_IMPLEMENTATION_FIX2_FREEZE.md` (so v3 derives
from the true frozen v2 bytes):

| # | v3 entity | SHA256 (freeze §1 match) | v2 source (FIX2 match) |
|---|---|---|---|
| 1 | `configs/…/m0_synth_kkt_v3.json` | `e6d33b5d…ceb7d5` ✓ | `5545…f700` ✓ |
| 2 | `schemas/…/…_payload_v3.schema.json` | `1d6f93a1…d2d3` ✓ | `7352…d9a2` ✓ |
| 3 | `schemas/…/…_case_v3.schema.json` | `df3616ff…ddcac` ✓ | `df36…dcac` ✓ (identical) |
| 4 | `scripts/analysis/…_run2_v3_common.py` | `9de62f6d…c411c2` ✓ | `5ef8…e24f` ✓ |
| 5 | `scripts/analysis/…_run2_v3_validate.py` | `2e0bb00b…ca13a6` ✓ | `4389…dc36` ✓ |
| 6 | `scripts/analysis/…_run2_v3_producer.py` | `6ef3a4a8…f5114` ✓ | `8c4c…1cd51` ✓ |
| 7 | `scripts/analysis/…_run2_v3_independent_verify.py` | `4025dbf0…8523c5` ✓ | `795b…1ef1` ✓ |
| 8 | `scripts/wrappers/…_run2_v3.sh` | `8d9123e9…d66d3` ✓ | `14eb…0716` ✓ |
| 9 | `scripts/slurm/…_m0_synth_kkt_v3.sbatch` | `4495ec3c…6b3d9` ✓ | `f914…94bf` ✓ |

The case schema (#3) hashes **identically** to v2 (`df36…dcac`), consistent with its only `v2`
token being the shared/frozen `cert_v2` reference (preserved), so only its filename changes.

### 1.2 Protected-transform diff — EMPTY for all nine (strongest proof)

For each pair I ran the cert-preserving transform and diffed against the actual v3 file:

```
diff <(sed 's/cert_v2/cert__CERTKEEP__/g; s/v2/v3/g; s/cert__CERTKEEP__/cert_v2/g' <v2>) <v3>
```

**Result: EMPTY for all nine pairs.** This proves v3 is byte-for-byte equal to the cert-preserving
`v2`→`v3` transform of the frozen v2 bytes — every intended rename/self-reference change is
present, every `cert_v2` is preserved, and **no other byte differs**. Because an empty diff also
means the transform did not wrongly rewrite any non-cert `v2` token, it independently rules out
the failure mode where a legitimate non-lineage `v2` token should have been preserved but was not.

### 1.3 Corroborating measures

| entity | byte len v2/v3 | residual `v2` in v3 | of which `cert_v2` | blanket-sed diff lines |
|---|---|---|---|---|
| config | 8709/8709 | 2 | 2 | 4 (2 cert lines) |
| payload schema | 26069/26069 | 0 | 0 | 0 (empty) |
| case schema | 9013/9013 | 1 | 1 | 2 (1 cert line) |
| common.py | 52671/52671 | 1 | 1 | 2 (1 cert line) |
| validate.py | 6543/6543 | 0 | 0 | 0 (empty) |
| producer.py | 16398/16398 | 0 | 0 | 0 (empty) |
| independent_verify.py | 60724/60724 | 2 | 2 | 4 (2 cert lines) |
| wrapper.sh | 2215/2215 | 0 | 0 | 0 (empty) |
| sbatch | 758/758 | 0 | 0 | 0 (empty) |
| **TOTAL** | equal each | **6** | **6** | 5 empty / 4 cert-only |

- **Byte length identical** for all nine pairs (both `v2`/`v3` are two-char tokens; no length
  drift possible from other edits).
- In **every** file, `residual v2 == residual cert_v2`, and a `(?<!cert_)v2` scan over all nine
  files returns **zero hits** → the *only* surviving `v2` anywhere is the frozen cert token.
- The blanket `s/v2/v3/g` diff surfaces **only** `cert_v3`↔`cert_v2` lines (the maximal-change
  reference over-rewrites the shared cert; v3 correctly keeps it), matching the freeze doc's
  own counts (config 2, case 1, common 1, independent_verify 2).

**Conclusion: v3 is a verified byte-exact clone of v2 with changes confined to the three
permitted classes (file names, internal self-references, hash bindings — the last vacuously, see
§3). No behavioral byte was changed. Any non-three-class difference would have been Critical;
none exists.**

---

## 2. The six `cert_v2` preservations — each independently adjudicated legitimate

I read every surviving `cert_v2` line. All six are references to the **shared, Run1-frozen,
cross-lineage** certification schema `scgp_global_cert_v2.schema.json`, which is **not** one of
the nine cloned entities — its `v2` is a permanent shared-schema name, not a lineage marker:

| # | site | line | role | verdict |
|---|---|---|---|---|
| 1 | config | L75 | `run1_frozen` hash-binding **path key** for cert schema (`4d3f…22f`) | legit — binds the frozen shared schema |
| 2 | config | L99 | `paths.cert_schema` → the cert schema file | legit — resolves to an existing file (see §3) |
| 3 | case schema | L15 | `schema_version` `const": "scgp_global_cert_v2"` | legit — cert identity constant |
| 4 | common.py | L28 | `CERT_SCHEMA_ID = "scgp_global_cert_v2"` | legit — cert identity constant |
| 5 | independent_verify.py | L212 | `if record["schema_version"] != "scgp_global_cert_v2"` | legit — verifier checks cert identity |
| 6 | independent_verify.py | L234 | emits `{"schema_version": "scgp_global_cert_v2"}` | legit — verifier re-emits cert identity |

Justification (independently checked, not copied): rewriting any of these to `cert_v3` would (a)
point at a nonexistent `scgp_global_cert_v3.schema.json` and (b) break the frozen hash binding
`4d3f…22f`. Preserving `cert_v2` is therefore the **behaviorally correct** choice, exactly as
required for v3 to bind the identical frozen cert schema. This is the single subtlety that makes
a naive blanket `v2`→`v3` incorrect, and v3 handles it correctly at all six sites.

I confirmed the referenced file **exists** and its hash **matches** the binding:
`schemas/lb_scgp_global_r2/scgp_global_cert_v2.schema.json` →
`4d3f1663e633c30ae58e35c0feddaa2fa9bbedba279cdbe6f38ecc35d761f22f` = config L75 bound value. ✓

---

## 3. Cross-reference resolution — all internal references resolve to v3 (cert stays v2)

Verified by `jq` on the config and by reading the wrapper/sbatch/payload directly:

- **config** `run.run_id` = `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v3`; `run.schema_id` =
  `scgp_global_synth_kkt_payload_v3`; `authorization.authorized_run_ids` =
  `["LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v3"]` (equals `run_id`); `run.artifact_path` =
  `artifacts/lb_scgp_global/v3/m0/synth_kkt/manifest.json`.
- **config `paths`**: `payload_schema`/`case_schema`/`wrapper`/`slurm_script` and all four
  `artifacts/lb_scgp_global/v3/…` output paths point at v3 entities; **`cert_schema` correctly
  stays `…scgp_global_cert_v2.schema.json`** and that file exists (§2).
- **config `run.slurm`** = `{cpu:8, ram_gb:64, gpu:0, env:"HateVideo", no_time_flag:true}`.
- **payload schema**: `artifact_schema_id` const = `scgp_global_synth_kkt_payload_v3`; the
  `cases` array `$ref` = `scgp_global_synth_kkt_case_v3.schema.json#` → resolves to the v3 case
  schema (which exists, #3).
- **wrapper** `lb_scgp_global_r2_run2_v3.sh`: `RUN_ID`/`EXPECTED` = `…-v3`, `CONFIG` =
  `m0_synth_kkt_v3.json`, `ARTIFACT_ROOT` = `artifacts/lb_scgp_global/v3/m0/synth_kkt`, invokes
  the three v3 `.py` scripts with the v3 config, and hard-checks `config.run.artifact_path ==
  artifacts/lb_scgp_global/v3/m0/synth_kkt/manifest.json`. All references v3-consistent.
- **validate.py / producer.py** import `lb_scgp_global_r2_run2_v3_common` (v3 module name);
  **independent_verify.py** is intentionally standalone (imports neither producer nor common),
  as in v2.
- **sbatch** `lb_scgp_global_r2_m0_synth_kkt_v3.sbatch`: `--job-name=lbscgp_global_r2_run2_v3`,
  `CONFIG=…m0_synth_kkt_v3.json`, `RUN_ID=…-v3`, calls the v3 wrapper; resources unchanged
  (`--cpus-per-task=8`, `--mem=64G`, no GPU, **no `--time`**), and it activates the environment
  via `source …/conda.sh; conda activate HateVideo` (line 12–13) — the correct pattern that the
  env-repair record identified as necessary to avoid the silent `ExMRD` fallback.

**Class-3 (hash bindings) audit:** the config binds hashes only of external/frozen entities
(authoritative docs, `data/gt` provenance, `old_protected` snapshot, `run1_frozen` incl. the
`cert_v2` schema). **None** of the nine cloned entities has its hash bound in the config, so no
binding is recomputed and the entire `hash_bindings` block is byte-identical v2↔v3 — the class-3
"change" is vacuous, which the empty protected-transform config diff (§1.2) already confirms.

---

## 4. Dependency risk re-assessment (the residual that killed v2, job 12971)

### 4.1 Full import enumeration (top-level **and** in-function) of the four v3 modules

`grep -nE '^\s*(import|from)\s'` over every module (this catches deferred/in-function imports,
which is the audit lesson from v2's death):

- **common.py**: stdlib `{__future__, hashlib, json, math, os, subprocess, tempfile, pathlib,
  typing}` + `numpy` (top, L19) + **deferred** `from jsonschema import Draft7Validator,
  RefResolver` / `from jsonschema.exceptions import SchemaError` at **L182–183**.
- **validate.py**: stdlib `{__future__, argparse, json, os, subprocess, sys, pathlib}` + local
  `from lb_scgp_global_r2_run2_v3_common import …` (L15). No top-level third-party import; the
  `jsonschema` preflight is a `find_spec('jsonschema')` subprocess check (L98–106, appended at
  L148) — fail-closed by design.
- **producer.py**: stdlib `{__future__, argparse, json, os, sys, pathlib, typing}` + `numpy`
  (top, L16) + local `…_v3_common` (L21).
- **independent_verify.py**: stdlib `{__future__, argparse, copy, hashlib, json, math, os,
  tempfile, pathlib, typing}` + `numpy` (top, L20) + **deferred** `jsonschema` imports at
  **L167–168**.

Third-party set = **`{numpy, jsonschema}`**, exactly matching `M0_ENV_REPAIR_RECORD.md` §4. The
two deferred `jsonschema` sites sit at **common.py:182 and independent_verify.py:167** —
**identical line numbers to v2** (the byte-exact clone leaves those lines untouched; their text
carries no `v2` token), confirming the v3 clone reproduces the exact deferred-import structure
that must be satisfied. Both deferred imports are wrapped `try/except Exception → RuntimeError`
(fail-closed; common.py raises `"jsonschema dependency unavailable; refusing to validate Run2-v3
payload"` — note the correctly-retagged `v3` label).

### 4.2 Read-only availability confirmation (method + limitation)

I **cannot** run Python. Per the verdict's read-only method, I listed the `HateVideo`
site-packages directory:

```
/data/jehc223/miniconda3/envs/HateVideo/lib/python3.11/site-packages/
  jsonschema/                        jsonschema-4.26.0.dist-info/
  jsonschema_specifications/         jsonschema_specifications-2025.9.1.dist-info/
  referencing/  referencing-0.37.0.dist-info/   rpds/  rpds_py-2026.6.3.dist-info/
  numpy/
```

`jsonschema 4.26.0` and its full transitive set (`referencing 0.37.0`, `rpds_py 2026.6.3` — the
ABI-sensitive `cp311` C-extension —, `jsonschema_specifications 2025.9.1`) plus `numpy` are all
**present** in the exact interpreter tree that the sbatch's `conda activate HateVideo` resolves.
This is consistent with `M0_ENV_REPAIR_RECORD.md` (install of `jsonschema 4.26.0`).

**Limitation:** a directory listing proves the packages are *installed*, not that they *import*
cleanly at runtime — I did not (and, per protocol, may not) execute the interpreter. The
env-repair record claims a successful runtime `import` in `HateVideo`; I could not re-verify that
step by static means. This residual is therefore **fail-closed** (a broken import would raise →
refuse, never a false PASS) and is downgraded but not zeroed — see **L-C**.

The **former M-C is RESOLVED**: the specific missing dependency that fail-closed job 12971 is now
installed and independently confirmed present.

### 4.3 `RefResolver` deprecation forward-risk

`jsonschema 4.26.0` still ships `RefResolver` (deprecated since 4.18.0); the deferred import
succeeds with a `DeprecationWarning` on stderr. The v3 sbatch sets **no** `PYTHONWARNINGS`
(`grep` count = 0), so the warning stays a warning — no import-time failure. If a future authority
sets `PYTHONWARNINGS=error` or upgrades `jsonschema` past `RefResolver` removal, the deferred
import fails closed (`RuntimeError`/`ImportError` → refuse), never a false PASS. Recorded as a
forward-risk within **L-C**, matching the env-repair §7 disposition.

---

## 5. Residual M-A / M-B independently re-adjudicated (carried from v2 per verdict §4(d.2))

Because the producer/common/independent_verify bytes are identical to v2 (§1.2 empty diffs), the
underlying logic is unchanged; I re-read the frozen proposal and the code myself to re-rule
severity rather than inherit it.

### M-A (Medium) — synthetic `G0` not verified PSD / rank-`<=d` realizable — **maintained Medium; conditional-High retained**

Evidence (independently confirmed): the producer/verifier construct
`G0 = G_star − offdiag(A_struct^T ν) − offdiag(S_psd)` with the diagonal forced to 1
(`…_v3_common.py:420–424`; verifier replay `…_v3_independent_verify.py:422–424`), i.e. derived
*from* `G_star`, not realized as an embedding Gram `Z0 Z0^T`. The verifier checks `G0` shape,
symmetry (`np.allclose(g0, g0.T)`) and unit diagonal (`np.allclose(diag(g0), 1.0)`) at
`…_v3_independent_verify.py:1090–1093` but performs **no** `G0 ⪰ 0` (PSD) or `rank(G0) <= d`
check — a targeted `(g0…(psd|eig|rank))` scan returns nothing. Generically this `G0` is full-rank
and may carry a slightly negative eigenvalue.

Ruling: `FINAL_PROPOSAL.md:129` literally writes `G0 = Z0 Z0^T ∈ S^N` (a PSD, rank-`<=d`
encoder Gram), which is a stronger property than the fixture honors — a genuine **fidelity** gap,
so not Low. But (a) the frozen hard-constraint block (`:264–291`) imposes PSD on the *variable*
`G`, with `G0` appearing only as the anchor `X0=(G0,0)`; (b) the KKT certificate (stationarity +
dual feasibility + complementarity + primal feasibility for a strongly convex program) is
mathematically valid for **any** symmetric unit-diagonal anchor, so the self-test's correctness
does not depend on `G0` PSD-ness; (c) Run2 is explicitly a **synthetic certificate self-test**,
whose contract is certificate structure + `G*` realizability, not `G0` provenance. Net: a
correctness-neutral fidelity gap → **Medium, non-blocking**.

Conditional escalation (retained, **not** currently triggered): if a science authority rules
the synthetic `G0` must itself be a realizable rank-`<=d` PSD `Z0 Z0^T`, this becomes **High**
and blocks. No such ruling exists in the materials I read, so I do **not** unilaterally escalate;
I flag it for the science owner to resolve before any *scientific* claim rests on the fixture.
Amendment paths (unchanged): construct `Z0∈R^{N×d}`, set `G0=Z0Z0^T` and `G*` from an in-`R^{N×d}`
perturbation within `rho_coord`; or add `G0 ⪰ 0` + `rank(G0)<=d` verifier checks; or formally
amend the frozen anchor definition to "symmetric unit-diagonal."

### M-B (Medium, fail-closed) — rank-deficient construction convergence not statically provable — **maintained Medium; flagged as the top single-submit residual**

Evidence (independently confirmed): `rank_deficient_structural_solution`
(`…_v3_common.py:643–700`) runs a **30-step geometric shrink** (`scale *= 0.7`) seeking a `scale`
that simultaneously satisfies `0.005 < movement_off_max <= 0.018`, `movement_fro > 0.005`, and
`r_abs_max <= 0.20`. Iteration-0 is *designed* to land in-window (initial
`scale = movement_target/max_off` with `movement_target = 0.012` puts the structural off-diagonal
at ≈0.012, plus an `S_psd` off-diagonal bounded ≤ `0.25·0.012 = 0.003`, total ≲0.015 ∈
(0.005, 0.018]); the one plausible miss is `r_abs_max > 0.20` when the structural adjoint couples
weakly (`max_off` small → large `scale`). Feasibility of the window depends on per-case geometry
of `Q`/the structural adjoint and is **not statically provable** for all constructed fixtures
(six case-matrix + four `orth_cap_fixture` cases). On failure it raises → the producer refuses to
publish (`…_v3_producer.py:390,396`) — **fail-closed**, so it can never manufacture a false PASS.

Ruling: **Medium, fail-closed** (unchanged from v2 — not a static impossibility like the prior
H5). I explicitly note two things the execution authorizer must weigh: (1) this construction was
introduced in FIX2 and has **never executed to completion** (v1 died on `KeyError`, v2 died at
`validate.py` on `jsonschema` before reaching the producer), so its convergence is genuinely
untested end-to-end; (2) unlike the resolved `jsonschema` residual, M-B is **not dissolvable by
any allowed read-only means** (it requires running the numpy construction). Under the single-submit
regime it is therefore the single most material residual: a non-convergence would burn the one
authorized attempt with zero science (fail-closed, no false PASS). The authorizer should accept
this convergence-burn risk consciously, or obtain an authorized SLURM-only producer dry-run
first. It does **not** block the static review (no Critical/High).

---

## 6. Wrapper L-A and other Low findings

### L-A (Low) — wrapper re-run cleanup can delete a prior success's artifacts — **maintained Low; non-actionable under single-submit**

`cleanup_on_exit` runs `rm -f "${PROSPECTIVE_OUTPUTS[@]}"` whenever `COMPLETE != 1`
(`…_run2_v3.sh:28–30`) without distinguishing files created by the current invocation from
pre-existing ones. On a hypothetical **second** invocation after a prior success,
`validate`/`producer` correctly refuse via no-clobber (non-zero exit), but the ensuing EXIT-trap
cleanup would then delete the prior success's `manifest.json`/`source_manifest.json`/
`access_ledger.json`/`semantic_verification.json` (+ locks). Ruling: **Low**. Under the v3
single-submit regime there is exactly one authorized `sbatch` and the `artifacts/lb_scgp_global/v3`
directory is currently **absent** (confirmed), so there is no prior v3 success to endanger and
the second-run scenario is out of the authorized path. It remains a genuine footgun worth
guarding (scope cleanup to invocation-created outputs, cf. the producer's own
`cleanup_created_outputs` tracking) but is non-blocking and non-actionable for the single
authorized attempt.

### L-B (Low, Run3 scope) — fixture assumes rather than demonstrates a rank-`<=d` projection optimum

The frozen convex program has no hard rank constraint (`FINAL_PROPOSAL.md:291` "no hard rank
constraint"), so its projection solution is generically full-rank; the synthetic fixture
sidesteps this by *constructing* a rank-`d` `G*` on the PSD boundary and back-fitting `G0`. This
is legitimate for a verifier self-test but does not demonstrate that the *real* projection yields
a rank-`<=d` solution — a Run3 / real-method question, explicitly out of Run2 synthetic scope.
**Low, recorded for Run3.**

### L-C (Low, fail-closed) — dependency availability proven by directory listing, not runtime import; environment not frozen

The read-only site-packages listing (§4.2) proves `jsonschema`/`numpy` are *installed* but not
that they *import* cleanly at run time, and the environment state is not frozen between now and
execution. Any breakage is fail-closed (refuse, never false PASS). Per verdict §4(d.3) the
execution authorizer carries the **mandatory dependency-availability evidence** duty and must
re-confirm the third-party set `{numpy, jsonschema}` is present in `HateVideo` at authorization
time. Includes the `RefResolver` forward-risk (§4.3). **Low, fail-closed.**

---

## 7. Notes for the execution authorizer (verdict §4(d.3))

1. **M-B is the top residual and is not statically dissolvable.** It can only fail closed
   (burned attempt), never false-PASS, but its convergence is untested end-to-end. Accept the
   single-submit burn risk consciously, or run an authorized SLURM-only producer dry-run first.
2. **M-A conditional escalation is live but untriggered.** Confirm no science authority requires
   `G0 = Z0 Z0^T` (PSD rank-`<=d`); if such a ruling is issued, M-A becomes High and blocks.
3. **L-C mandatory item:** re-confirm `{numpy, jsonschema}` present in `HateVideo` at
   authorization time (read-only, or authorized SLURM-only preflight — never a login-node
   interpreter run outside SLURM). The env is not frozen.

---

## 8. Static checks performed (record)

- `sha256sum` on all nine v3 entities (= freeze §1) and all nine v2 sources (= FIX2 freeze).
- Protected-transform `diff` for all nine pairs — **EMPTY** (§1.2); blanket `s/v2/v3/g` diff —
  cert-only/empty; byte lengths equal; `(?<!cert_)v2` scan over nine files — zero hits.
- Six `cert_v2` sites read individually and adjudicated legit (§2); cert schema file present with
  matching hash `4d3f…22f`.
- `jq -e .` on config + payload schema + case schema — well-formed; config cross-refs resolved
  via `jq` (§3).
- `grep -nE '(import|from)'` full import enumeration of the four modules incl. in-function (§4.1).
- Read-only `ls` of `HateVideo` site-packages — `jsonschema 4.26.0` + transitive deps + `numpy`
  present (§4.2). No interpreter executed.
- `bash -n` OK for wrapper and sbatch; `grep` confirms no `--time` and no `PYTHONWARNINGS`.
- `git status --porcelain` — the nine v3 entities are untracked (`??`); no v2 entity shows as
  modified; `git diff --check` clean (exit 0).
- `artifacts/lb_scgp_global/v3` absent; only `v1` present. No artifact created by this review.
- No Python / import / `py_compile` / test / conda / SLURM / `sbatch` execution was performed.

---

## Required statements

- No performance evidence exists and no performance claim is made or possible from this static
  review of a byte-exact clone.
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/
  localization/stance/target/mechanism/rationale/fragment gold is assumed or introduced; v3
  fixtures are synthetic and v3 produced no artifact and no counters.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain **locked**.
- This review does **not** authorize SLURM execution. `PASS_STATIC_REVIEW` permits only a
  separate independent execution-authorization review (with the mandatory dependency-availability
  item) for exactly one future CPU-only SLURM validation job. The M-A conditional escalation must
  be resolved (ruling or amendment) if any authority holds that the synthetic `G0` must be
  realizable rank-`<=d` PSD.
- This reviewer role is separate from the v3-setup/freeze, execution-authorization, and executor
  roles.

Report SHA256 is to be computed externally after this file is written; it is not embedded to
avoid a self-referential hash.
