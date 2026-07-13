# M1 CACHE Merged Review — Amendment Ratification + Fresh 0C/0H Static Code Review

Date: 2026-07-13

Reviewer: **Claude Opus 4.8**, fresh **0-context / 0-history independent M1 reviewer** for the
`lb_scgp_global_r2` **M1 block** (`runs[4]` CACHE-MHC-v1 / `runs[5]` CACHE-MHC_zh-v1 / `runs[6]`
CACHE-SEAL-v1). This role is separate from m1-prep (amendment author + implementer + freezer),
the execution authorizer, and the executor. This one document merges: (1) amendment ratification,
(2) execution of the three coordination-session rulings, (3) the 14-entity static code review, and
(4) independent re-derivation of the freeze doc's handoff + simulation tables.

**Boundary / discipline.** Read-only static only. Shell limited to `rg / sed / nl / jq / awk /
bash -n / diff / sha256sum / find / ls / wc / git status / git diff` — **no Python executed** (no
`py_compile`, no unit test, no import). The only file written is this one. No SLURM submission, no
git commit, no GPU/MLLM/OCR/network run, no validation/test/label read.

**Model declaration.** Ran as Claude Opus 4.8 (`claude-opus-4-8`, 1M context), per the project's
subagent-model requirement (CLAUDE.md). No model deviation.

---

## 0. Verdict (summary)

| dimension | result |
|---|---|
| Amendment substance (resource fix / model pin / OCR / replica pins / additive edits) | **SOUND** |
| Hash cascade (4 files) + backup + array invariants | **VERIFIED** |
| Ruling ② (replica = frozen deterministic) | **APPROVED** |
| Ruling ③ (standalone smoke exemption) | **APPROVED w/ refinement** (contract-compatible; see §4) |
| Ruling ① (historical-snapshot rollback) | **NOT APPLIED → HIGH must-fix** |
| Static code review (14 entities) | 1 **HIGH** (GPU guard), else PASS + 3 LOW |
| **Critical / High count** | **Critical = 0, High = 2** |

**Grade: FAIL (conditional) — `AMENDMENT_RATIFIED_PENDING_TWO_FIXES`.** Per the rubric (RATIFIED +
PASS require High = 0), the block is **not ratifiable as-is**. Both High items are **narrow and
trivially fixable** (one documentation revert + hash re-cascade; one preflight-guard idiom swap),
and neither can corrupt an artifact (both are fail-safe / fail-closed). The amendment's science is
correct, the isolation code is strong, and the smoke exemption is approved. Re-review surface after
fix = the two touched entities only.

---

## 1. Amendment ratification (item 1)

### 1.1 Hash cascade — all verified
| file | recorded "after" | recomputed (this review) | match |
|---|---|---|---|
| `EXPERIMENT_PLAN.machine.json` | `93fdd752…861d5ef6` | `93fdd752…861d5ef6` | ✓ |
| `EXPERIMENT_PLAN.md` | `ed38f38c…a6918916` | `ed38f38c…a6918916` | ✓ |
| `EXPERIMENT_TRACKER.md` | `e19b7f26…2c24f591` | `e19b7f26…2c24f591` | ✓ |
| `EXPERIMENT_PLAN_HASHES.sha256` (triple + self) | `5246f208…` self | `5246f208…` self, triple = ed38/e19b/93fd | ✓ |
| `.pre_m1_amendment.bak` | `f4d54b78…830fc9b` ("before") | `f4d54b78…830fc9b` | ✓ |
| `M1_CACHE_PLAN_AMENDMENT.md` | `b9505a7f…` | `b9505a7f…` | ✓ |
| `M1_CACHE_PLAN_AMENDMENT.machine.json` | `c894b829…` | `c894b829…` | ✓ |
| `M1_CACHE_PLAN_AMENDMENT_HASHES.sha256` (self, freeze §1) | `0615b7ee…` | `0615b7ee…` | ✓ |
| Run1 `scgp_global_cert_v2.schema.json` (unchanged) | `4d3f1663…` | `4d3f1663…` | ✓ |
| 14 M1 entities (freeze §1) | all 14 | all 14 recomputed = freeze | ✓ 14/14 |

### 1.2 `.bak` ↔ current diff — every hunk mapped to a declared edit
`diff` produced exactly the following hunks, each matching the amendment's declared changes:

| diff hunk | declared edit | ok |
|---|---|---|
| `runs[4].slurm.gpu 0→1`, `runs[5].slurm.gpu 0→1` | resource fix ① | ✓ |
| `runs[4].budget.estimated_gpu_hours 0→8`, `runs[5] 0→8` | resource fix ① | ✓ |
| `runs[4]/[5].budget.api_calls_semantics` (+block, external_api=0, formula 4·N) | api semantic note | ✓ |
| `runs[4]/[5].model_pin` (Qwen2.5-VL-7B, offline, decoding, authority) | model pin ② | ✓ |
| `runs[4]/[5].evidence_pack_protocol` (16 frames, title, ASR, label_blind, dedup) | OCR/evidence ③ | ✓ |
| `runs[4]/[5].replica_protocol` (4, semantics_pin, parse_failure_policy) | replica pin | ✓ |
| `runs[4]/[5].ocr_policy` (live_ocr=false, ocr_calls=0) | OCR omission ③ | ✓ |
| `runs[4]/[5].m1_amendment` note | provenance | ✓ |
| `runs[6].seal_protocol` (cpu_only, verifies…, model_processor_hash_pin_required) | seal note | ✓ |
| 10× `684→700` aggregate lines + `by_milestone_must.M1 0→16` + `gpu_hours_erratum` | GPU-h aggregation | ✓ (but see §3.1) |
| `budget_ranges.m1_local_mllm_gpu_correction` (+record) | erratum record | ✓ (but see §3.1) |
| `concurrency.m1_cache_parallel_max2.max_gpu_total=2` | concurrency | ✓ |

**No other hunks.** `estimated_api_calls` numeric values are **unchanged** (2196 / 2316 / 0); only
their semantics annotated.

### 1.3 Array + isolation invariants — verified
- `runs` length **66 = 66**; `run_order` length **66 = 66** (unchanged).
- `runs[0..3]` and `runs[7..65]` **byte-identical** pre↔current (per-index sha256 compared; 0
  mismatches). Only `runs[4]/[5]` changed (gpu 0→1, gh 0→8); `runs[6]` gained `seal_protocol` and
  stays gpu=0/gh=0.
- `sum(runs[*].estimated_gpu_hours)` **684 → 700** (Δ+16 = `runs[4]8 + runs[5]8 + runs[6]0`).
  `by_milestone_must.M1.gpu_hours = 16`. Arithmetic internally consistent.
- `concurrency.m1_cache_parallel_max2 = {max_concurrent 2, max_cpu_total 8, max_ram_total_gb 64,
  max_gpu_total 2}` — consistent with 2×(1 GPU / 4 CPU / 32 GB) and within
  `immutable_contract.slurm.max_gpu = 2`.
- `immutable_contract` (claims=2, blocks=5, datasets, seeds, only_gold, forbidden_routes) **untouched**.

**Amendment substance is CORRECT and internally consistent** — with the single exception that the
GPU-h correction was propagated into a historical-snapshot field it should not have touched (§3.1).

---

## 2. Ruling ② — replica semantics (APPROVED)

`FINAL_PROPOSAL.md:170–177` ("Cache protocol") reads verbatim: **"four deterministic calls per
train video; fixed decoding/model/processor/prompt/input/schema hashes; parse failures become
canonical all-unresolved records."** This is **unambiguously frozen, deterministic** semantics —
there is **no** temperature/seed-diversity language anywhere in the proposal's cache protocol.

Therefore the amendment's pin is **faithful**: replicas use **byte-identical evidence-pack input**
+ greedy decoding (`do_sample=false, temperature=0, num_beams=1`), so the four replicas coincide up
to hardware FP nondeterminism, which the M2 `sigma_cache` gate bounds (`sigma_cache ≈ 0` is the
expected fact, correctly annotated). The code matches: `producer.main()` loops
`for _replica in range(REPLICAS)` over the **same** `frames/title/transcript`, with
`torch.manual_seed(0)` and greedy `generate(...)`. **Approve; no diversity semantics exist to
violate.**

---

## 3. Coordination ruling execution

### 3.1 Ruling ① — historical-snapshot rollback: **NOT APPLIED → HIGH must-fix**

The ruling: fields of the **"historical approved snapshot" class (`original_approved_*_before_*`)**
must **roll back to 684 with an annotation**; forward-looking views keep 700.

**Field-by-field classification (all 684→700 sites):**

| field | class | should be | current | verdict |
|---|---|---|---|---|
| `original_approved_r2_envelope_before_v2.must_gpu_hours` | **HISTORICAL snapshot** | **684** | **700** | **VIOLATION** |
| `original_approved_r2_envelope_before_v2.total_gpu_hours` | **HISTORICAL snapshot** | **684** | **700** | **VIOLATION** |
| `paper_plan_substitution_envelope.{must,total}_gpu_hours` | forward (substitution view) | 700 | 700 | OK |
| `remaining_prospective_budget.{must,total}_gpu_hours` | forward ("prospective") | 700 | 700 | OK |
| `lifetime_lineage_envelope.{must,total}_gpu_hours` | forward (includes prospective v2) | 700 | 700 | OK |
| `matrix_estimated_must_run.gpu_hours` | forward (estimate) | 700 | 700 | OK |
| `matrix_estimated_total_with_nice.gpu_hours` | forward (estimate) | 700 | 700 | OK |
| `by_milestone_must.M1.gpu_hours` | forward (estimate) | 16 | 16 | OK |

**Exactly one** `original_approved_*_before_*` field exists; m1-prep set it to **700** and even
**listed both of its fields in `m1_local_mllm_gpu_correction.aggregates_corrected`** — a deliberate
erratum-everywhere choice that **directly contradicts ruling ①**.

**Why the ruling is sound (I concur, independently).** A field literally named
`…_before_v2` is a point-in-time record of what was approved *before* the v2 amendment. At that
time the M1 GPU-h were genuinely recorded as 0 (the miscount was present); the honest historical
value is **684 + an annotation**, not a silent overwrite to 700. Overwriting makes the field
inconsistent with its own name and destroys the audit trail — the exact numeric-provenance failure
mode the ceremony exists to prevent. **Precedent:** all four prior backups (`pre_v4`,
`pre_realbank`, `pre_realbank_v2`, `pre_m1`) held every one of these aggregates at 684 unchanged;
M1 is the first GPU-h correction, so there is no "erratum-everywhere-for-GPU-h" precedent to lean on.

**Severity: HIGH** (per mandate + provenance-integrity). **Fail-safe** — documentation-only; it
gates no run and cannot affect execution or science.

**Fix (before authorization):**
1. Revert `original_approved_r2_envelope_before_v2.must_gpu_hours` and `.total_gpu_hours` to **684**.
2. Add an annotation on that block, e.g. `"gpu_hours_note": "historical snapshot retained at 684
   (M1 local-MLLM GPU-h omission was present at approval time); forward views corrected to 700, see
   m1_local_mllm_gpu_correction"`.
3. Remove those two entries from `aggregates_corrected` (or reclassify them as "intentionally
   retained; historical").
4. Re-run the 4-file hash cascade and re-freeze (`M1_CACHE_FREEZE.md` §1 amendment-lineage +
   configs' `authoritative_inputs` machine hash).

### 3.2 Ruling ③ — non-genealogy smoke exemption: **APPROVED (with refinement)** — the load-bearing ruling

**Question:** authorize one pre-authorization `m1_smoke` non-genealogy job (10 videos, output only
to `slurm/tmp`, independent job-name, declared out of the single-submit budget)? Central concern:
does it violate `mllm_calls_outside_train_cache = 0`?

**Counter scope — proven LOCAL (per-producing-job), not global.** I traced the counter end to end:
- Declared in `ZERO_COUNTER_KEYS` (common.py:95); the ledger **initializes it to 0** and **never
  increments it anywhere** (the only two counters the ledger ever bumps are
  `forbidden_path_read_count` and `non_allowlisted_train_content_read_count`, both of which also
  **raise**, killing the producer fail-closed).
- The producer emits it from **its own** `ledger.counters` into that job's `cache_manifest.json` /
  `access_ledger.json`.
- The seal checks it **only** from the producer's manifest (`zc = manifest.get("zero_counters")`;
  `zero_ok = all(int(zc.get(k,1))==0 …)`). **There is no cross-job or lineage-global MLLM tally
  anywhere in the code.**

**Conclusion:** a separate smoke job (different job, writes no sealed cache, produces no ledger)
**cannot make the sealed cache's `mllm_calls_outside_train_cache` nonzero**. The sealed counter
stays truthfully 0 because `runs[4]/[5]` invoke the MLLM only inside their own train-cache loop.
The smoke therefore does **not** violate the counter, `immutable_contract` (no science/scope
change), or the seal's per-artifact `zero_counters`. On single-submit: pre-declaring the smoke as a
**non-genealogy, out-of-budget, independent-job-name** job means it consumes no run entry's single
submit → compatible.

**Alternatives assessed (per the task):**
- **val/test videos — REJECTED.** Reading val/test content trips `validation_content_read_count` /
  `test_content_read_count` and violates the "val/test content never touched" rule. Off-limits.
- **preflight-inside-`runs[4]` (first-10 leading segment) — REJECTED as the primary de-risk.** A
  preflight failure inside the real job still **burns `runs[4]`'s single submit** (the job ran and
  died), so it does *not* provide the single-submit protection a **separate** smoke does; and it
  forces re-implementing + re-freezing a frozen entity. (A cheap in-job fail-fast is fine as
  belt-and-suspenders, not as a substitute.)
- **HateMM (non-contract) videos ×10 — RECOMMENDED.** Touches **zero** MHC train/val/test content
  and **zero** MHC isolation counters, while exercising exactly the deferred GPU path (model load
  offline → decode → processor `videos=[frames]` → `generate` → `parse_certificate`). Cleanest
  possible isolation narrative.
- **MHC-train videos ×10 — acceptable fallback.** Mechanically safe (sealed counter still 0; train
  *content* reads are the authorized activity and no labels are read), but strictly weaker on the
  narrative than HateMM.

**Approved smoke spec:** standalone job; **HateMM (preferred) or MHC-train** videos ×10; output
**only** to `slurm/tmp/` (throwaway, never under `artifacts/lb_scgp_global/v1/m1/`); independent
job-name (e.g. `lbscgp_global_r2_m1_smoke`); **pre-declared in the authorization doc as outside the
single-submit budget**; exercises the frozen producer path at tiny N to validate (a) the GPU guard
(§5 HIGH), (b) offline model load, (c) `load_video_frames` decode, (d) the processor call, (e)
`generate` + parse. **This smoke is not just compatible — it is strongly recommended, because it is
the empirical validation that would catch the un-provable HIGH in §5 before the single submit.**

---

## 4. Static code review (14 entities)

### 4.1 Interface three-way alignment — PASS
- `zero_counters`: common.py `ZERO_COUNTER_KEYS` (30) **== identical ==** seal schema `required`
  (30) **== identical ==** seal schema `properties` (30); all 30 props are const-0
  (`minimum:0, maximum:0`; 0 props with `maximum≠0`). `diff` empty on both comparisons.
- Replica record: `make_replica_record` returns exactly the 6 keys the replica schema `required`
  lists (`video_id, evidence_pack_sha256, replica_index, schema_version, observables, parse_flags`).
- Observables: schema `observables.required` = **9** == `OBSERVABLE_KEYS(8) + MODALITY_KEY(1)`;
  `canonical_unresolved_observables` and `parse_certificate` build exactly those 9.
- Seal decision fields == seal schema (strict, `additionalProperties:false`, `per_dataset` min/max 2).

### 4.2 Machine lock-step, run_id/path/model pins — PASS
`verify_machine_cache` asserts, for `runs[4|5]`, that `run_order[index]`, `run_id`, `milestone`,
`dataset`, `artifact_paths`, `artifact_schema_ids`, `slurm` (== `expected_cache_slurm_block()`
**and** == config `slurm`), `dependencies`, `model_pin.model_id`, `frames`, `replicas`, and
`ocr_policy.live_ocr` all match code constants / config. `verify_machine_seal` asserts `runs[6]`
similarly with gpu=0 and `dependencies == [MHC, MHC_zh]`. `RUN_INDEX = {…:4,5,6}` is pinned and both
the numeric index and `run_id` are asserted (v3 index-drift lesson). Machine `runs[4..6]` content
independently confirmed to match (dataset MHC/MHC_zh, gpu 1/1/0, gh 8/8/0). **No index literal
points at the wrong run.**

### 4.3 Label-blind self-attestation (freeze §5) — PASS, independently re-derived
- PCRE2 grep over all 4 `.py`: **no** gold/split/seed/neighbor/prediction/margin/stance/target/
  span/timestamp data access (`[...]` or `.get(...)`). None found.
- Evidence-pack builder reads **only** `obj["id"]`, `obj.get("text")`, `obj.get("chunks")`,
  `obj.get("window_text")` — no label field.
- Every `label` token = docstring/disclaimer, the config **flag name** `"label_read_allowed"`
  (checked `== False`, never read as a datum), the output boolean `"labels_enter_after…"`, or the
  error-message **parameter** `label_text`. Every `seed` token = docstring, the counter key
  `"seed_read_count"`, `torch.manual_seed(0)` (determinism WRITE), or comment. Matches §5.

### 4.4 cert_v2 cross-check / parse-failure / retry / Merkle / seal / temps — PASS
- **cert_v2 cross-validation:** the producer validates `cert_v2_object(observables,flags)` against
  the **Run1-frozen** `scgp_global_cert_v2.schema.json` (`4d3f1663`, live dependency) **and** the
  replica record against `scgp_global_cache_replica_v2`.
- **Parse failure → canonical unresolved, NO rescue:** `parse_certificate` returns
  `canonical_unresolved_observables()` on any of {no JSON, decode error, non-object, extra keys,
  missing/malformed key, bad state, bad/boolean/out-of-range confidence}, recording `parse_flags`;
  no re-prompt / schema repair. Transport failures → `["transport_failure"]` → unresolved.
- **Retry capped:** per-slot `while True` bounded by `attempt > REPLICAS` (≤5 tries/slot) **and**
  the dataset hard cap `total_invocations >= retry_cap` (`api_retry_cap` = 4392 MHC / 4632 MHC_zh =
  2× base; per-dataset caps sum to the plan's 9024). No unbounded loop.
- **Merkle:** `merkle_root` sorts leaves + duplicate-last → order-independent; the seal recomputes
  leaves and compares to `manifest.cache_merkle_root`. `call_count == 4·U_D` is enforced at the
  producer (raise) **and** re-checked at the seal.
- **In-repo temps only:** `exclusive_publish_json[l]` use `tempfile.mkstemp(dir=str(fs_path.parent))`
  (explicit in-repo) + `O_EXCL` publish lock + no-clobber; wrappers set `REPO_TMPDIR=slurm/tmp`
  (in-repo, `mkdir -p`) and create no temp of their own. **No `${TMPDIR:-/tmp}` anywhere** (realbank-v1
  landmine closed). No-clobber: producer/seal refuse if artifact or `.publish.lock` exists; wrapper
  trap cleans prospective outputs on non-complete exit; the seal wrapper **keeps** a published STOP
  decision (auditable).
- **sbatch resources:** cache `--gres=gpu:a100:1 --cpus-per-task=4 --mem=32G`, seal no gres / 4 / 32,
  all `no --time`, `HF_HUB_OFFLINE=1`; consistent with the amendment. 5 shells `bash -n` clean; all
  JSON `jq -e .` valid; artifact namespace `artifacts/lb_scgp_global/v1/m1/` **absent**.

### 4.5 HIGH — `require_slurm_cache` GPU-count guard is brittle and un-verifiable
```
seen = None
for key in ("SLURM_GPUS","SLURM_GPUS_ON_NODE","SLURM_STEP_GPUS","SLURM_JOB_GPUS"):
    value = os.environ.get(key)
    if value:
        token = value.split(":")[-1] if ":" in value else value
        if token not in {"(null)","NoDevFiles"}: seen = token
if seen is not None and seen != "1": raise "requires exactly 1 GPU, got {seen}"
```
The loop keeps the **last** non-empty var = `SLURM_JOB_GPUS`, which in Slurm holds the **global GPU
device ID(s)** (e.g. `"0"`, `"3"`), **not the count**. `SLURM_GPUS_ON_NODE` is the count var, but it
gets overwritten. So for a correctly-provisioned 1-GPU allocation whose assigned device ID ≠ `"1"`
(the common case — every real GPU job in this repo runs on the cgroup-remapped `cuda:0` while the
physical/global ID can be any of 0–7), the guard **raises "requires exactly 1 GPU, got 0"** and the
job dies at preflight. It passes only if `SLURM_JOB_GPUS`/`SLURM_STEP_GPUS` are unset in the batch
shell (falling back to `SLURM_GPUS_ON_NODE="1"`) **or** the ID happens to be `"1"` — both
cluster-config-dependent and **not statically provable**. No empirical evidence exists: every
recorded `slurm_*gpus` value in `artifacts/` is `null` (CPU jobs), and freeze §3 row 2 punts this to
runtime.

- **Impact:** fail-closed (cannot corrupt an artifact), but a plausible **false-fail burns the
  single-submit ceremony** — precisely the outcome that burned v1/v2/v3/realbank. It **diverges
  from the project's accepted, proven idiom**: `sq_common.require_runtime(gpu=True)` and
  `lb_scgp_common.require_slurm(expected_gpu=True)` both simply assert `CUDA_VISIBLE_DEVICES` is
  non-empty; the CPU-only siblings (`run2_common`, this module's `_forbid_ambient_gpu_when_cpu_only`)
  use the "any nonzero gpu var → raise" pattern, which is robust. Only this novel exactly-1
  string-compare is fragile.
- **Severity: HIGH** for a single-submit, burn-averse regime.
- **Fix (choose one, before authorization):** (a) replace the block with the accepted idiom —
  assert `os.environ.get("CUDA_VISIBLE_DEVICES")` is non-empty (matches the two accepted GPU
  guards); or (b) if an exact count is desired, read **`SLURM_GPUS_ON_NODE`** explicitly (the count),
  or count comma-separated indices in `SLURM_JOB_GPUS` (`len(v.split(","))==1`), rather than
  string-comparing the last-seen var to `"1"`. **Either way, gate the real submit on the §3.2
  smoke**, which empirically settles this on the actual cluster.

### 4.6 LOW (non-blocking observations)
- **M-1:** the cache sbatch does not export `OMP/MKL/OPENBLAS_NUM_THREADS` (the seal does). Not a
  correctness issue for greedy single-GPU `generate`; replica determinism is bounded by the M2
  `sigma_cache` gate. Optional hardening.
- **M-2:** `verify_machine_cache` does not cross-check `budget.estimated_api_calls` /
  `api_retry_cap` machine↔config; the real guard is `call_count == 4·U_D` (producer raise + seal
  check). Cosmetic.
- **M-3 (= freeze R-2):** `evidence_pack_sha256` includes `video_sha256`, so the builder reads every
  train mp4 once for hashing in addition to the producer's decode. Content-addressing is the safer
  default for a correct `U_D`; flagged, not blocking.
- **R-5:** the processor call `videos=[frames]` (PIL list, `images=None`) is DEFERRED — corroborated
  by the two accepted scripts on the same env; the §3.2 smoke validates it. Fail-closed if rejected.

---

## 5. Handoff + simulation tables — independently re-derived (item 4)

**Dependencies (site-packages corroborated, `ls`/`rg` only):** `torch`, `transformers` **4.49.0**
(exports `Qwen2_5_VLForConditionalGeneration` — 2 hits in `transformers/__init__.py`), `numpy`,
`jsonschema`, `decord`, `av` all present in the HateVideo env; in-repo `load_video_frames` present at
`src/utils/generate_subclip_embedding_HF.py:204`, returning `(frames, ok)` exactly as the producer
unpacks (`decoded, ok = …`; on failure `(None, False)` → text-only call). Function-level imports
(`torch`, `transformers.{AutoProcessor,Qwen2_5_VLForConditionalGeneration}`, `load_video_frames`,
`jsonschema.Draft7Validator`) are enumerated and `dependency_check()` **fails closed** before the
model loads.

**§4 handoff tables (11-row MHC / identical MHC_zh N=579; 9-row seal):** re-traced writer→path→
guard→reader→guard for each artifact. All writes route through `exclusive_publish_json[l]`
(canonical in-repo `dir=` mkstemp, O_EXCL); all reads through `canonical_root_path`. **0 out-of-repo
paths, 0 ambient-env paths, 0 FAIL rows.** Confirmed.

**§3 simulation table (18 rows):** the 15 PASS rows are supported by the static evidence above
(configs↔machine↔code lock-step, schema strictness, no-clobber, Merkle/consensus determinism,
`call_count==4·U_D`, label-blind). The 3 **DEFERRED-TO-RUNTIME** rows are each **fail-closed or
fail-open-by-design**:
- Row 16 (model load offline, R=4 greedy): import / `from_pretrained` raises on missing weights or
  symbol → **fail-closed**.
- Row 17 (`load_video_frames`): unreadable video → `(None,False)` → **text-only call → canonical
  unresolved** (documented fail-open-to-unresolved; contract's "invalid → unresolved, not rescue").
- Row 18 (forbidden zero_counters): ledger **raises** on any forbidden/non-allowlisted read →
  producer dies; the seal additionally enforces `zero_counters` via schema `maximum:0` →
  **fail-closed**.

---

## 6. Fix list (to reach RATIFIED + PASS)

1. **[HIGH · ruling ①]** Revert `original_approved_r2_envelope_before_v2.{must,total}_gpu_hours` →
   **684**; add the historical-snapshot annotation; drop those two from `aggregates_corrected`
   (or reclassify). Re-run the 4-file hash cascade + re-freeze. *(documentation only)*
2. **[HIGH · §4.5]** Replace `require_slurm_cache`'s GPU-count check with the accepted
   `CUDA_VISIBLE_DEVICES`-non-empty idiom (or read `SLURM_GPUS_ON_NODE` explicitly). Re-freeze the
   `common.py` entity (SHA) + configs' `implementation_files` binding. *(fail-closed guard only)*
3. **[Approved]** Run the §3.2 standalone smoke (HateMM ×10 → `slurm/tmp`, out-of-budget,
   independent job-name) as the empirical gate that de-risks fix #2 and the DEFERRED rows before the
   single submit.
4. **[Optional / LOW]** M-1 thread vars in the cache sbatch; M-2/M-3/R-5 as noted.

`ready_for_execution` remains **false**; execution authorization is a separate role and still
requires fixes #1–#2 (and, recommended, #3) plus the exact-hashes/no-clobber and separate execution
authorization steps.

---

## Required statements
- No performance evidence exists or is claimed; this review is static and read-only. No accuracy /
  macro-F1 reproduced; no training / kNN / model / MLLM / OCR / network / GPU run.
- The only project gold is `parent_video_binary_label`; no segment/frame/span/localization/stance/
  target/mechanism/rationale/fragment gold is assumed or introduced. No train label, and no
  validation/test content or label, is read anywhere in the M1 chain (verified by grep + code trace);
  labels enter only after the cache seal.
- M2 (comparator freeze), validation/test, and training remain **locked**. This review authorizes no
  execution and unlocks nothing downstream.
- Reviewer = Claude Opus 4.8, fresh 0C/0H independent M1 reviewer, separate from m1-prep,
  execution-authorization, and executor roles. Wrote only this document; edited no code / config /
  schema / plan; ran no job; executed no Python.

---

# DELTA REVIEW (post-fix) — 2026-07-13

Same reviewer / same read-only discipline (no Python). m1-prep applied both HIGH fixes and ran the
approved non-lineage smoke (job 13002). Re-verified all four delta items below.

## D1. Two fixes strictly match the fix list — no out-of-bounds edits

**FIX-1 (ruling ①) — budget-only, exactly in scope.**
- `original_approved_r2_envelope_before_v2.{must,total}_gpu_hours` = **684** (rolled back), with a new
  `gpu_hours_note` annotation ("historical snapshot retained at 684 … forward views corrected to 700").
- Both fields **removed** from `m1_local_mllm_gpu_correction.aggregates_corrected`; a
  `historical_snapshot_intentionally_retained_at_684` sub-record was added citing ruling ①.
- All 8 forward-view fields (paper_plan_substitution ×2, remaining_prospective ×2, lifetime_lineage
  ×2, matrix_estimated_must_run, matrix_estimated_total_with_nice) stay **700**; `by_milestone M1` = 16.
- **Non-budget diff (bak↔current, `del(.budget_ranges)`) = exactly the amendment's runs[4..6] +
  concurrency edits and nothing else** (`model_pin`×2, `evidence_pack_protocol`×2, `replica_protocol`×2,
  `ocr_policy`×2, `api_calls_semantics`×2, `gpu`×4, `estimated_gpu_hours`×4, `seal_protocol`×1,
  `m1_amendment`×3, `max_gpu_total`×1). runs[4/5/6] content re-confirmed (gpu 1/1/0, gh 8/8/0). FIX-1
  touched **only** `budget_ranges`.

**FIX-2 (§4.5) — common.py-local, exactly in scope.** `require_slurm_cache` now asserts
`CUDA_VISIBLE_DEVICES` non-empty (the accepted `sq_common`/`lb_scgp_common` idiom); the brittle
`SLURM_*_GPUS` last-token `!= "1"` loop is removed; the docstring records the rationale. common.py
699→702 lines; `ZERO_COUNTER_KEYS` still **30**; `merkle_root/consensus_for_video/parse_certificate/
model_processor_hash/exclusive_publish_*/verify_machine_*` all intact. **14-entity recompute: only
common.py (`6d2834e9→601d61e2`) and the 3 configs (rebind: `1f5c0615→23c777de`, `79506228→57bf435d`,
`9e2d487e→94147df7`) changed; the other 10 entities (2 schemas, evidence_pack, producer, seal, 2
wrappers, 3 sbatch) are byte-identical.**

## D2. Hash cascade + config rebind — self-consistent
- Plan cascade `f4d54b78 → 93fdd752 → 7638ac78` (machine); `EXPERIMENT_PLAN.md` `→e5ec9bc4`;
  `EXPERIMENT_TRACKER.md` `→f36e3dec`; `EXPERIMENT_PLAN_HASHES.sha256` self `9de299fd` — all recomputed
  and matching; the HASHES file lists the correct triple.
- All **3 configs** rebind identically: machine `7638ac78`, plan.md `e5ec9bc4`, tracker `f36e3dec`,
  hashes `9de299fd` (= the config-bound value == the file's real sha256).
- `M1_CACHE_FREEZE.md` §1 records the new common.py + 3 config hashes (matching my recompute), row 2
  updated to the FIX-2 idiom, and a `§FIX` section documents both fixes with diffs + the cascade table.
  Manifest is consistent for the execution authorizer.

## D3. Smoke (job 13002) releases HIGH #2 + the DEFERRED rows
Non-lineage smoke: 10 **HateMM** (non-contract) train videos, output only to `slurm/tmp/` (verified
cleaned; dir empty), zero MHC contact, `artifacts/lb_scgp_global/v1/m1/` still **absent**, no sealed
cache/ledger. Imports the **frozen** common + producer `build_messages`, so the sealed code path ran.
- **FIX-2 GPU guard: PASSED** under a real `--gres=gpu:a100:1` alloc with `CUDA_VISIBLE_DEVICES="0"`.
  The record confirms the **old** guard would have read the global id from `SLURM_JOB_GPUS` and
  false-failed ("got 0") — the exact §4.5 scenario, now empirically settled. **HIGH #2 RESOLVED.**
- **DEFERRED rows validated:** row 16 offline model load (5.97 s, `HF_HUB_OFFLINE=1`); row 17 decode
  (`load_video_frames` 10/10); R-5 processor `videos=[frames]` accepted; generate + `parse_certificate`
  (36/40 clean, 1 video's 4 replicas → canonical all-unresolved = contract-correct). Row 18 forbidden
  counters remain code-enforced (fail-closed).
- **R=4 determinism: 10/10 videos byte-identical** across replicas → confirms ruling ② and
  `sigma_cache ≈ 0`. GPU peak 52.71 GiB < 80 GiB. Time extrapolation **7.73 / 8.15 GPU-h** per dataset
  ≈ the amendment's pinned **8 GPU-h/run** → the resource fix is empirically sound.
- Two **benign cosmetic** notes (non-blocking): `.err` "do_sample False but temperature=1e-06" (greedy
  ignores it; determinism proven; optional producer hardening = pass `temperature=None`), and the
  transformers `use_fast` deprecation notice. Neither touches a frozen entity or correctness.

## D4. Parse rate 0.90 vs plan QC — no threshold to clear
`rg` over the plan **and** `FINAL_PROPOSAL.md`: **no numeric parse-rate floor exists**. `parse_rate` is
a **reported diagnostic** in the runs[4]/[5] `metrics` list (alongside `unresolved_count`, `call_count`,
`merkle_leaves`, `zero_forbidden_counters`); the run gate is *"GO iff restricted schema, QC, and zero
forbidden counters pass"* and the **seal code does not reference `parse_rate` at all**. Parse failures
→ canonical all-unresolved is the contract's prescribed behavior ("all records remain in the full
bank"), not a QC failure. So the smoke's 0.90 (a 10-video diagnostic, not the real cache's metric)
clears trivially — there is nothing to clear — and the 10% failure was handled exactly as specified.

## DELTA VERDICT

**Critical = 0, High = 0 → `AMENDMENT_RATIFIED` + `PASS_STATIC_REVIEW`.** Both HIGH must-fixes are
applied cleanly, in-scope, and cascade-consistent; FIX-2 is empirically confirmed on the real cluster;
ruling ② and the resource envelope are corroborated by the smoke. The two review items (amendment
ratification + fresh 0C/0H code review) are **CLOSED**. Execution still requires the separate
exact-hashes / no-clobber review and execution authorization (out of this reviewer's scope);
`ready_for_execution` is not set by this review.

---

# DELTA-2 REVIEW (v2 — input-symlink guard fix, post v1 double-burn) — 2026-07-13

Same reviewer / same read-only discipline (no Python). Context: after DELTA-1 RATIFIED, the v1 cache
runs **double-burned** (jobs 13003/13004) at Stage-1 — the train mp4s are in-repo **symlinks whose
targets escape the repo**, and v1's `canonical_root_path.resolve()` fired at the video site before any
model load or write (`M1_CACHE_V1_RESULT_TO_CLAIM_REVIEW.md`; fail-closed, zero artifacts). m1-prep
built v2 (run_id v1→v2 REPLACE + a symlink-tolerant `canonical_video_path` + 12-entity clone-rename +
real-path smoke2). Re-verified all six delta items.

## DD1. v2 amendment diff — exactly in scope
pre_m1_v2 backup (`7638ac78`) ↔ current (`ab0a06fb`) machine diff = **only**: `run_order[4]/[5]` and
`runs[4]/[5].run_id` `…-MHC-v1→v2` / `…-MHC_zh-v1→v2`; `runs[4]/[5]` gained `v1_burn_and_v2_replace` +
`v1_burn_jobs` (13003/13004) provenance; `runs[6].dependencies` re-pointed to the two v2 ids +
`v2_dependency_sync` note; `dependency_dag.m1_v2_replace_record` added. **Nothing else** — `runs[6].run_id`
stays `…-SEAL-v1`, `runs[4]/[5]` artifact_paths + `artifact_schema_ids` (`scgp_global_cache_replica_v2`)
+ slurm + budget unchanged; runs array length 66=66. Amendment doc hashes verify (`5f0036e8` / `df1dea76`).
Cascade: machine `7638ac78→ab0a06fb`, `EXPERIMENT_PLAN.md` unchanged (`e5ec9bc4`; no literal M1 run_id in
it), TRACKER `f36e3dec→86db7a5f`, HASHES `9de299fd→3d603edc` — all recomputed and matching.

## DD2. Guard-fix semantics — "tolerate input symlink location, zero other isolation weakening" ✓
`canonical_video_path(rel, dataset)` (new in `…_v2_common.py`) contains on the **un-resolved LOCATION**:
rejects `..`, rejects absolute, requires `rel` under `data/video/<dataset>/All/`, requires
`normpath(ROOT/rel)` lexically under ROOT, requires the **parent dir to resolve in-repo** (only the leaf
may be a link), and requires the leaf to be a **regular file or symlink** (`lstat`, no follow). It returns
the in-repo location; **the sole relaxation is that the mp4 leaf's resolved target need not be under
ROOT** — exactly the designed corpus layout.
- **Allowlist/forbidden unchanged:** `note_video_read` still runs `forbidden_reason(rel, …)` →
  `forbidden_path_read_count` on any forbidden token; the video-root allowlist is enforced by the guard's
  `startswith` — so no val/test/held/label path is reachable (they fail the video-root prefix).
  `ZERO_COUNTER_KEYS` still **30**; `forbidden_reason`/`FORBIDDEN_TOKENS`/`evidence_allowlist` byte-unchanged.
- **Output/other paths unchanged:** every other `canonical_root_path` site (config, machine, artifact
  publish, `builder_sha`/`common_sha`, `args.out`) is untouched; `exclusive_publish_*` (in-repo `dir=`,
  O_EXCL, no-clobber) unchanged. Only the **two video sites** (builder `evidence_pack_v2:188`, producer
  `producer_v2:181`) swapped `canonical_root_path`→`canonical_video_path`.
- **`followed_target` is audit-only:** recorded in the ledger (`{is_symlink, followed_target,
  followed_target_in_repo}`); it gates nothing. **Confirmed "log only, no other access granted."**
  `video_sha256` retained (bytes via the OS-followed link).
- **v1↔v2 code diffs prove guard-only:** common = guard fn + `note_video_read` rewrite + `import stat` +
  RUN_MHC/MHC_ZH→v2 (RUN_SEAL stays v1); evidence_pack/producer = the video-site swap + refs; **seal = v1→v2
  reference tokens only, zero behavioral change**.
- *Informational (not a defect):* the guard trusts the symlink **target** (a symlink under `All/` pointing
  at forbidden in-repo content would pass `forbidden_reason` on `rel`). This is out of the threat model
  (the symlinks are the non-adversarial corpus layout, not attacker-controlled) **and** is empirically
  excluded by the readlink audit (DD4): all 790+806+1066 targets resolve into the external video corpora,
  none to a label/test/gt file.

## DD3. 14 SHAs (12 renamed + 2 carried) + 3-config rebind — verified
All **14** recompute = `M1_CACHE_V2_FREEZE.md` §1: the 12 v2 clones (common/evidence/producer/seal ×4,
3 configs, 2 wrappers, 3 sbatch) at the frozen SHAs, and the **2 artifact schemas carried forward
byte-unchanged** (`4bfcfea2` / `f4605bb7` — correct: renaming contract-versioned schema ids would desync
the machine `artifact_schema_ids`; coordination ruled schema-naming stays). All **3 v2 configs** rebind
the post-cascade quartet (machine `ab0a06fb` / plan `e5ec9bc4` / tracker `86db7a5f` / hashes `3d603edc`),
run_ids MHC-v2 / MHC_zh-v2 / SEAL-v1, schema_ids unchanged. v1 entities **retained** as the burned-lineage
record (realbank precedent). v2 sbatch correctly wired (mhc_v2→v2 config+wrapper+`--gres=gpu:a100:1`;
seal RUN_ID=SEAL-v1 with v2 config/wrapper). 5 v2 shells `bash -n` clean.

## DD4. readlink topology — independently re-derived (reproduces freeze §5.1)
My own `find -type l` + `readlink` sweep:

| dataset | video `*.mp4` | escapes? | target root | lora_frames links | gt/train link | ASR/train link |
|---|---|---|---|---|---|---|
| MHC | 790/790 symlink | **YES** | `/data/jehc223/Multihateclip/English/video_mp4/` | 0 | 0 | 0 |
| MHC_zh | 806/806 symlink | **YES** | `/data/jehc223/Multihateclip/Chinese/video/` | 0 | 0 | 0 |
| HateMM | 1066/1066 symlink | **YES** | `/data/jehc223/HateMM/video/` | 0 | 0 | 0 |

The mp4 symlink is the **sole escape class, dataset-universal**; gt/ASR/lora_frames are real in-repo.
`canonical_video_path` tolerates **exactly** this (leaf under `data/video/<ds>/All/` only; parent must
resolve in-repo) and nothing else — the third blind-spot class (input-symlink topology) is now modeled
per dataset (mandatory simulation row 8), and my re-derivation matches it exactly.

## DD5. smoke2 (job 13009) — "real frozen entry" satisfied; DEFERRED released
smoke2 **imports and calls the frozen** `ev.build_dataset_packs(HATEMM, …)` (line 57) and
`C.canonical_video_path` (line 115) — the exact v1 burn surface, **not** a re-implementation (the v1
smoke's fatal gap; HateMM is registered in the in-memory `EXPECTED_TRAIN_N` so the unmodified builder
accepts the non-contract dataset — a constant monkeypatch, not a code fork). Result: frozen
`build_dataset_packs` processed **744/744** real repo-escaping symlinked mp4 **with no raise** in 4.68 s,
744/744 followed-targets recorded as external, **all forbidden zero-counters 0**; 16-frame decode via the
followed link 5/5; **R=4 determinism 5/5** byte-identical; **cert_v2 20/20** validate; offline load 5.94 s;
GPU guard PASS (`CUDA_VISIBLE_DEVICES="0"`); peak 52.71 GiB. This releases simulation rows 2 (GPU guard),
7 (guard tolerates symlink / rejects `..`+non-root), 8 (topology), and 10 (model load + decode on
symlinked mp4). Row 9 (forbidden counters at seal) stays DEFERRED-fail-closed (code-enforced; resolves at
the real seal). Non-lineage hygiene: HateMM-only, `slurm/tmp` (cleaned, dir empty), zero MHC contact,
artifact namespace absent, no sealed cache/ledger. Parse 16/20=0.80 — no plan threshold (DELTA-1 D4);
`hate_video_104` 0/4 → canonical unresolved is contract-correct. Benign cosmetic notes unchanged
(temperature=1e-06; `use_fast`).

## DELTA-2 VERDICT

**Critical = 0, High = 0 → `AMENDMENT_RATIFIED` (v2) + `PASS_STATIC_REVIEW`.** The v2 amendment is exactly
in-scope; the symlink-tolerant guard tolerates precisely the input-mp4-symlink escape with zero other
isolation weakening (allowlist/forbidden/output checks intact, `followed_target` audit-only,
`video_sha256` retained); 14 SHAs + 3-config rebind + cascade are self-consistent; the readlink topology
re-derives exactly; and smoke2 empirically settles the burn on the **real frozen** `build_dataset_packs`.
The two v2 review items (amendment ratification + fresh 0C/0H v2 code review) are **CLOSED**. Remaining:
the six-step gate's step 5 (exact-hashes/no-clobber) and step 6 (one re-submit each for MHC-v2 / MHC_zh-v2,
then seal), which are outside this reviewer's scope; `ready_for_execution` is not set by this review.
