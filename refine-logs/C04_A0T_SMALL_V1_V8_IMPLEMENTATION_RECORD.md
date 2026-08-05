# C04-A0T-SMALL-v1 Implementation-v8 Record

Date: 2026-08-01
Status: **PROSPECTIVE / FRESH INDEPENDENT REVIEW REQUESTED / EXECUTION BLOCKED**
Scientific tag: `C04-A0T-SMALL-v1`
Implementation version: `v8_prospective`
Supersedes: `artifacts/c04/a0t_small_v1_impl_v7` (GPU job `13852`, CUDA OOM in the
vision tower, **no scientific verdict**)

## Why v8 exists

Job 13852 held the tranche's only GPU allocation for 1978 s and died with
`torch.OutOfMemoryError: Tried to allocate 110.50 GiB` inside
`F.scaled_dot_product_attention` of the Qwen2.5-VL **vision** blocks, on the
first MHC-ZH item (`MHC_zh/BV1vs4y127aA`, 1080x1920).

The reconciled cause (finding F115, and re-measured independently here):

| quantity | measured |
|---|---|
| offending item at native resolution | `video_grid_thw = (4,138,78)` |
| pre-merge visual patch tokens | 43,056 |
| vision SDPA score tensor, fp32, 16 heads | 43,056² × 16 × 4 B = **110.50 GiB** |
| same item at `max_pixels=151200` | `(4,36,20)` = 2,880 tokens, **0.49 GiB** |

The v7 producer called `AutoProcessor.from_pretrained(snapshot,
local_files_only=True)` with **no** `max_pixels`, so frames entered the vision
tower at native resolution. The closed form above reproduces the reported
110.50 GiB exactly and is pinned as a self-test fixture.

**No v7 record is salvaged.** v7 completed 400/400 HateMM prompt records, and
none of them enter a v8 bank: applying the deployed cap also changes what the
teacher sees for 178/200 HateMM items, so a mixed bank would hold some items at
native resolution and some capped. Uniform teacher input across one sealed
tranche is a **scientific** bar, decided ahead of any seal-semantics argument.

## The four changes, each flagged

### CHANGE-1 — the tranche reservation, 7200 s → 5222 s

The amendment's FIRST_TRANCHE ceiling is **unchanged** at 7200 s ("an aggregate
maximum of 2 GPU-hours across both datasets and all C04 jobs"). What shrinks is
this tranche's *reservation*, because 13852 really spent GPU time and the
campaign accumulator recorded it from sacct:

```
sacct -X -n -P -j 13852 -o JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES
13852|c04_a0t_small_v1_v7|FAILED|1:0|1978|billing=8,cpu=8,gres/gpu=1,mem=64G,node=1
```

`1978 + 5222 = 7200` exactly. Every stage that asserted
`small_cap_gpu_seconds == 7200` now asserts `SMALL_TRANCHE_CAP_GPU_SECONDS`,
including the `resource_final_state` schema's `cap_gpu_seconds` const
(7200 → 5222) and `TERMINAL_SECONDS_HARD_MAX` (now `cap + 600`, the same 600 s
slack v7 carried).

The number is **binding, not decorative**: a self-test runs
`assert_campaign_aggregate_headroom` against the live accumulator and requires
5222 to be accepted **and 5223 to be refused**. If a later job moves the
aggregate, that fixture goes red before anything is materialised.

A separate confirmation that 13852 is the only C04 job that ever held a GPU:

```
13805 c04_a0t_small_v1_v5_preflight  FAILED     0s  billing=8,cpu=8,mem=64G,node=1
13840 c04_a0t_small_v1_v6_preflight  COMPLETED 19s  billing=8,cpu=8,mem=64G,node=1
13850 c04_a0t_small_v1_v7_preflight  COMPLETED 19s  billing=8,cpu=8,mem=64G,node=1
13852 c04_a0t_small_v1_v7            FAILED  1978s  billing=8,cpu=8,gres/gpu=1,mem=64G,node=1
13853 c04_a0t_small_v1_v7_reconcile  COMPLETED  1s  billing=2,cpu=2,mem=4G,node=1
```

### CHANGE-2 — `max_pixels = 151200`. This is a TEACHER-INPUT change

`AutoProcessor.from_pretrained(..., max_pixels=TEACHER_MAX_PIXELS)`, with
`TEACHER_MAX_PIXELS = 360*420 = 151200`.

It is **not** a convenience constant chosen to make the run fit. It is the cap
every deployed Qwen2.5-VL entrypoint in this repository already uses:

| file | line | value |
|---|---|---|
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | 157 | `default=360*420` |
| `scripts/analysis/predict_target_qwen.py` | 299 | `default=360*420` |
| `scripts/analysis/generate_vision_summary.py` | 69 | `default=360*420` |
| `scripts/analysis/p10_score_segments.py` | 147 | `default=360*420` |
| `scripts/analysis/p10c_score_segments.py` | 137 | `default=360*420` |
| `scripts/analysis/score_segments_mllm.py` | 87 | `default=360*420` |
| `scripts/role3/arbitrate_qwen.py` | 363 | `default=360*420` |
| `src/utils/generate_VideoMLLM_embedding_bidir_textpool_HF.py` | 29 | calls 151200 "the deployed max_pixels" |

So the argument is protocol fidelity in the direction of the project, not away
from it: **v7's uncapped teacher was the anomaly.** The line-29 comment even
records the merged token count it produces — 720 — which this build reproduces
(2,880 pre-merge / 4 = 720).

The change is nonetheless recorded as a teacher-input change in three places
that a reviewer cannot miss: `teacher_contract.max_pixels_is_a_teacher_input_change`
is `true` in the config, `teacher_max_pixels` is in every frame-pack binding, and
`teacher_max_pixels` is in every prompt record's `provenance`.

**Plus a fail-closed runtime ceiling.** `VISUAL_PATCH_TOKEN_HARD_CEILING = 4096`
is checked after the processor and **before `model.generate`**, on the pre-merge
count (which is what vision SDPA runs on — v7's original arithmetic was wrong by
4× precisely because it applied the 2×2 merge first). At 4096 tokens the
worst-case score tensor is 1.00 GiB against 79.27 GiB of device memory. The
producer also requires the runtime count to **equal** the count the CPU preflight
froze for that item. So the v7 failure mode cannot recur even if the cap were
somehow not applied.

### CHANGE-3 — a measured pre-submit projection gate in the CPU preflight

The preflight now measures **every one of the 400 sealed items'** real pre-merge
visual token count by running that item's frozen frames through the Qwen2.5-VL
`image_processor` on CPU — no model weight, `CUDA_VISIBLE_DEVICES` explicitly
empty and refused if not — records the per-item counts in the preflight
manifest, and projects the GPU window from measured inputs.

**Two states, no third.** Fits by measurement → the namespace is published and
the tranche may proceed to review. Does not fit → `HALT_RESOURCE_PROJECTION`
**before the no-clobber namespace exists**, with the numbers on stdout.

The gate basis is deliberately pessimistic in three separate ways, each recorded
in `projection_basis` in the config:

1. per-forward time is v7's measured mean **at native resolution** (4.4248 s over
   400 forwards). Forward time regresses strongly on visual token count —
   least squares over those 400 measured forwards gives
   `elapsed = 1.9913 + 3.567e-4 × pre_merge_patch_tokens` — so a v8 forward on
   the same item cannot be slower;
2. fixed overhead is v7's whole non-forward remainder (1978 − 1769.9 = 208.1 s),
   which still contains the 201 frame packs v7 built inside its allocation and
   v8 does not;
3. MHC-ZH forwards are priced at the HateMM mean. **This is the projection's one
   genuinely unmeasured input** and it is stated in the artifact
   (`unmeasured_input`) rather than hidden.

Corroboration, reported but **not** used as the gate: 46 of v7's 400 HateMM
forwards ran on items already at or below the cap, where native geometry
*equals* capped geometry. Those 46 measured **2.900 s** per forward against
4.623 s for the 354 above the cap. That is a direct measurement in the capped
regime, and it agrees with the regression's prediction at the capped mean
(3.007 s). It is a biased subsample (small-frame videos), hence corroboration
only.

### CHANGE-4 — frame packs move to the CPU preflight

The 400 immutable eight-frame packs are now decoded, written and hash-pinned by
the CPU preflight. The GPU producer may only **load** one and HALTs if a pack is
absent; it has no decode path to fall back to.

Measured motivation, on this hardware:

| work | HateMM/item | MHC-ZH/item |
|---|---|---|
| decode 8 frames | 0.11 s | 0.48 s |
| PNG encode 8 frames | 0.43 s (854×480) | 2.66 s (1080×1920) |

That is roughly 700 s of pure CPU work that v7 performed with the A100 idle,
against a window that is now 4022 s. One MHC-ZH item (`BV18N4y1B7qA`) additionally
fails `decord` open and falls back to pyav, which measured **45.8 s** on its own.

Three consequences, all intended:

* ~700 s of CPU work leaves the GPU allocation;
* the frames become auditable **before** the payload review, instead of being
  created by the very job the review authorises;
* the producer cannot decode a video at all, so an unexpected decode cost cannot
  appear inside the budgeted window.

**The frame bytes are unchanged, and that is proved rather than asserted.** The
dry-execution harness re-derives 12 HateMM packs from the real videos with the
v8 code and compares them to the packs job 13852 wrote: all eight PNG SHA-256s,
the backend, the total frame count, the requested index vector and the
decode-failure vector are identical on all 12.

The pyav fallback is additionally hardened from one pass holding **every** frame
in memory (measured 2,879 frames at 1920×842, of the order of 14 GiB resident) to
two passes retaining only the eight requested. Byte-identical output, verified on
the real fallback item, and **6.6× faster** (45.8 s → 6.9 s).

The frame-pack binding drops the three GPU-stage lineage hashes it could not
have at preflight time (`payload_review_sha256`,
`gpu_execution_authorization_sha256`, `allocation_claim_sha256`) and gains
`code_resource_authorization_sha256` + `teacher_max_pixels`. Nothing becomes
unbound: all 400 manifests enter `staged_output_hashes`, each manifest pins its
own eight PNG digests, the validator re-checks every one, the manifest is pinned
by the payload review and the payload review by the GPU authorization. It is the
same chain walked in the other direction.

## Carried forward from the v7 round-5 non-blocking observations

1. `NO_SEAL_SENTINEL` is now **imported** by the gpu_ledger module rather than
   redefined — the round-5 reviewer's explicit instruction ("any v8 must import
   it"), because a one-sided edit would resurface the round-4 High only after the
   GPU is spent.
2. A template-tamper fixture that injects **no banned token** was added, so
   deleting the template-equality clause inside
   `assert_teacher_visible_containment` now turns a fixture red. A companion
   fixture proves that injection really carries no banned token, so the new
   fixture cannot be passing for the old reason.

## All v7 hardenings retained intact

I-1 fail-closed teacher-visible identifier containment (400 IDs × 2 forms as a
batch precondition **before model load**, re-checked per forward); I-2
known-answer selection digest vector; I-3 allocation-anchored in-job
`BudgetGuard` with an accounting-only breach record and exit 40; the
campaign-ledger headroom check at three points, each before an irreversible step;
and the authorization-flag-before-namespace-creation gate in the GPU wrapper.

## Scientific content: unchanged, and proved unchanged

Selection rule, tag, suffix, count, prompt **text** and its four SHA-256 values,
prompt-hash rule, frame index rule, frame count, transcript normalization and
cap, decoding parameters, the five-rate reliability taxonomy and every threshold
are byte-identical to v7 and to the v6 freeze.

The four frozen prompt hashes still equal the v6 freeze exactly:

| key | value |
|---|---|
| `system` | `1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048` |
| `A` | `cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b` |
| `B` | `9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314` |
| `combined` | `a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a` |

The v8 selection recomputed from the frozen rule reproduces the v6/v7 frozen
allowlists **exactly** on both datasets (200/200 and 200/200, same order).

## Zero-cost dry execution — 19 checks, 0 failed

Per the v7 record's own rule: *before any single-use ticket is consumed or any
no-clobber namespace is entered, a zero-cost dry execution must exercise the
first real operation of the payload path — not a check that the operation
exists, not a reading of its call site, but the operation itself, on real
inputs.*

| check | result |
|---|---|
| self-test fixtures (**65**; v7 had 37) | 0 failed |
| prompt hashes still equal the v6 freeze | pass |
| selection reproduces the v7/v6 frozen allowlists | HateMM 200 same, MHC_zh 200 same |
| teacher-visible containment on all real renderings | **800 renderings, 0 halts** |
| campaign headroom accepts 5222, refuses 5223 | pass (aggregate 1978, phase cap 7200) |
| config cap equals the live ledger headroom | 5222 == 7200 − 1978 |
| v8 frame packs byte-identical to v7's | 12/12 sampled, 0 mismatch |
| two-pass pyav fallback byte-identical to one-pass | pass, 2879 frames, 45.8 s → 6.9 s |
| processor honours the deployed max_pixels | 151200 |
| measured geometry under the ceiling | grid (4,20,36) = 2880 |
| OOM closed form reproduces 110.50 GiB | pass |
| prompt record with `visual_patch_tokens` validates | pass |
| schema refuses a record above the ceiling | pass |
| authority flip + prompt-hash freeze are contract-neutral | pass |
| the tranche cap IS inside the contract hash | pass |
| `max_pixels` IS inside the contract hash | pass |
| projection window == cap − three reserves | 4022 |
| projection prices 800 forwards | pass |
| projection refuses a geometry above the ceiling | pass |

Also: `python -m py_compile` on all four programs, `bash -n` on all three
wrappers and all three sbatch files, `jq` on the config, all five schemas and the
campaign ledger. All 15 v7 implementation files re-verified **unchanged**. All 15
frozen design hashes re-verified. `artifacts/c04/a0t_small_v1_impl_v8/` does not
exist. `PYTHONDONTWRITEBYTECODE=1` throughout.

A new static check *executes* rather than asserts the CHANGE-4 claim: the CPU
preflight parses the producer's own frozen bytes and proves it imports no video
decoder, calls no decoder attribute, has no image `save` call, and imports none
of the frame-writing symbols from the common module. The same predicate applied
to the common module goes red (it does carry `decord` and `av`), so the check is
non-vacuous.

## Projected GPU window

With a placeholder worst-case geometry (3,072 tokens) and 60 s of pack loading;
the real numbers come from the preflight run:

| quantity | value |
|---|---|
| usable teacher window | **4022 s** (5222 − 300 watchdog − 300 item margin − 600 seal reserve) |
| projected, conservative basis | **3807.9 s** → margin 214.1 s (5.3 %) |
| projected, capped-regime corroboration | **2588.1 s** → margin ~36 % |
| affordable mean per forward | 4.6924 s vs 4.4248 s measured at native |
| worst-case vision SDPA at 3,072 tokens | 0.562 GiB (vs 79.27 GiB available) |

Both bases fit. The conservative margin is thin by design — it prices every one
of the 800 forwards at v7's native-resolution mean and adds back overhead v8 does
not incur. If reality still surprises us, the `BudgetGuard` stops **before an
item**, leaving every completed checkpoint intact, no seal, exit 40 and an
accounting-only breach record.

## Execution state

Every teacher/GPU/SLURM/reconciliation authorization is `false`;
`preflight_materialization_authorized` is `false`. `code_resource_verdict` is
`PENDING` pinned to the sentinel `PENDING_V8_CODE_RESOURCE_REVIEW`, which
`_verified_review_file` rejects with `HALT_REVIEW_LINEAGE`. The four
`prompt_hashes` carry the pending sentinel; the CPU preflight is the stage that
materialises them. No v8 review or runtime artifact exists.

The unified pilot gate, the `+0.030/+0.030` two-dataset target and the
amendment's full-bank `+0.050/+0.050` DIRECT-OOF and STUDENT-OOF gates are
untouched and unwaived. **No metric, result, reliability rate or CONTINUE/KILL
verdict is published by anything in this record.**

## Prospective file hashes

| File | SHA-256 |
|---|---|
| `scripts/analysis/c04_a0t_small_v1_v8_common.py` | `1dbcc2c480ecc71abdf200b179382c098b597ff05cdf277b48090a293ca00703` |
| `scripts/analysis/c04_a0t_small_v1_v8_preflight.py` | `51f2ae69207a1cf575bc3756f7be3f209e53950b53b2d98c18e90302540f1f6b` |
| `scripts/analysis/c04_a0t_small_v1_v8_gpu_ledger.py` | `b2e98d23cd5c851b9277a94cdc8875a1fca6293b9511106b573fcf5d11247049` |
| `scripts/analysis/c04_a0t_small_v1_v8_producer.py` | `7997d8ab9659f5ce91e111156d3cb3fcf3ab3360c405c53a7d291272c610f65a` |
| `scripts/wrappers/c04_a0t_small_v1_v8_preflight.sh` | `c52a8ab3b9d8678633c8e00ab831aab1ab231e41edabc814ab933893865f375c` |
| `scripts/wrappers/c04_a0t_small_v1_v8.sh` | `0568ed9381271fe6d901aafef713eff68c0932c74b54bf413a619fb357cfff18` |
| `scripts/wrappers/c04_a0t_small_v1_v8_reconcile.sh` | `2a6a922d2c7ac913141d4ac284734205ce1e9d8fb091273c11872bd66456a7cd` |
| `scripts/slurm/c04_a0t_small_v1_v8_preflight.sbatch` | `9214e2287b5244b86505ff1a9d9685b45eb8c8ad1904fea51a51828a5219d462` |
| `scripts/slurm/c04_a0t_small_v1_v8.sbatch` | `bff81fd58881638ee75a489f1931013d51ca0cafe020ea4d6a5b4ad6d34c1d84` |
| `scripts/slurm/c04_a0t_small_v1_v8_reconcile.sbatch` | `c17f9f5be14b95bdb32b7b8e020dbd1d953d74b40cbd1d01e3dfc2454268e561` |
| `schemas/c04/c04_a0t_small_v1_v8_prompt_record.schema.json` | `eea33f75c196996afa701c32213ca3c6773e4580866e64347722c4a2a01da3ca` |
| `schemas/c04/c04_a0t_small_v1_v8_canonical_record.schema.json` | `e36b36d6559f0b26f363aa2b98ac2257ea2418eb38bccd618f24e7cd14ce92ee` |
| `schemas/c04/c04_a0t_small_v1_v8_stage_authorization.schema.json` | `316930ee7170021cb8f91c8a51f15292d57033864a563d62a4d61f0dcf437600` |
| `schemas/c04/c04_a0t_small_v1_v8_payload_review.schema.json` | `55902820a1160afba3555065f4908dfdf6919e423228d8f75e88d979410f0cfd` |
| `schemas/c04/c04_a0t_small_v1_v8_resource_final_state.schema.json` | `2abdf8f674a31166d489952cca4d5b27b7553d8099793ef001832932bcca698c` |

## The two GPU ledgers — pointer direction, restated so it cannot be inverted

`artifacts/c04/campaign/gpu_ledger.json` is the **authoritative accumulator**
across every C04 GPU job and every implementation version. It reads *from sacct*
and never from a namespace ledger. A namespace ledger
(`artifacts/c04/<impl>/resource/gpu_ledger.json`) is evidence about one
allocation and dies with its namespace. **Never overwrite, regenerate or "sync"
the campaign file from a namespace ledger** — a namespace ledger knows only its
own allocation, so doing so would erase every prior version's spend and reopen
the ceiling. Current state: revision 1, `aggregate_gpu_seconds` 1978, phase
`FIRST_TRANCHE`, phase cap 7200.

## The failure family this build is reviewed against

Four instances now, all the same shape — **an irreversible resource consumed
before the check that would reject the run**:

| # | where | consumed first | what would have rejected it |
|---|---|---|---|
| 1 | v5 static gate | an ~8-hour hold and a queue slot (job 13805) | a config↔computed prompt-hash equality asserted *before* the freeze that produces those hashes |
| 2 | v6 `mark_exit` | the genesis ledger's revision, breaking a no-clobber pin | a claim-time HALT |
| 3 | v6 render defect | the single-use ticket, the namespace, 7B weights on an A100 | `str.format` on a template containing literal JSON braces |
| 4 | **v7 uncapped processor** | **1978 GPU-seconds and the whole tranche** | **any check on visual token count before the vision tower** |

Instance 4 is the one v8 is built around, and it is worth naming precisely why it
survived five code/resource rounds and a payload review: **every safeguard
checked what the code *said*, and none measured what the model would *see*.**
The geometry was computable on a CPU, for free, at any point — and nobody
computed it. That is why the v8 preflight measures all 400 items rather than
asserting a cap, and why the producer re-checks the count before every forward.
