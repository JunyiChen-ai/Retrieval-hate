# C04-A0T-SMALL-v1 impl-v8 — Code/Resource Review Request

Date: 2026-08-01
Stage: `CPU_PREFLIGHT` authorization
Requested verdict: `GO (0C/0H/0I)` or `REVISE` with findings

You are a **fresh independent reviewer**. You have had no exposure to the
implementation reasoning. Read the frozen bytes. Do not accept a claim in this
request as evidence for itself — every claim below is either checkable against
the bytes or is flagged as unverified.

## What you are authorizing, and what you are not

A **CPU-only** SLURM preflight that materialises a no-clobber namespace. It
loads no model weight, reads no label value, touches no dev/test path, makes no
network call, performs no OCR, and submits no job. It does **not** authorize any
GPU, teacher or SLURM execution: those need separate later reviews.

If you return `GO`, the next irreversible step is that CPU preflight creating
`artifacts/c04/a0t_small_v1_impl_v8/`, which can never be recreated.

## Files to review (frozen; verify each SHA-256 before reading)

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
| `configs/c04/c04_a0t_small_v1_v8.json` | (read live; `implementation_hashes` must equal the 15 rows above) |

Binding contract: `refine-logs/C04_USER_AMENDMENT_V2.md`.
Campaign accumulator: `artifacts/c04/campaign/gpu_ledger.json`.
Predecessor, for comparison only, and **not to be edited**:
`scripts/analysis/c04_a0t_small_v1_v7_*.py` and
`refine-logs/C04_A0T_SMALL_V1_V7_IMPLEMENTATION_RECORD.md`.

## Context you need: why v7 died

GPU job `13852` held the tranche's only allocation for 1978 s and failed with
`torch.OutOfMemoryError: Tried to allocate 110.50 GiB` inside
`F.scaled_dot_product_attention` of the Qwen2.5-VL **vision** blocks, on the
first MHC-ZH item at 1080×1920. The v7 producer built its processor with **no**
`max_pixels`, so frames entered the vision tower at native resolution:
`video_grid_thw = (4,138,78)` = 43,056 pre-merge patch tokens, and
43,056² × 16 heads × 4 bytes = 110.50 GiB. Raw log:
`slurm/logs/c04_a0t_small_v1_v7_13852.err`.

No v7 record is carried into v8. The v8 namespace is clean, both datasets, no
item skipped.

## FOUR CHANGES YOU ARE EXPLICITLY ASKED TO ATTACK

### CHANGE-1 — tranche reservation 7200 s → 5222 s

The FIRST_TRANCHE phase ceiling stays 7200 s; the *reservation* becomes
`7200 − 1978 = 5222`. Check: that `SMALL_TRANCHE_CAP_GPU_SECONDS` really replaces
every `== 7200` assertion including the `resource_final_state` schema const; that
`TERMINAL_SECONDS_HARD_MAX` is still coherent; that the self-test which requires
5222 accepted and 5223 refused against the **live** ledger is non-vacuous; and
that no code path can now publish a state the schema forbids, or require a state
no code can produce (that is the exact shape of the v7 round-4 High).

### CHANGE-2 — `max_pixels = 151200`. A TEACHER-INPUT CHANGE

This changes what the teacher sees for 178/200 HateMM items and for the
large MHC-ZH items. It is argued as protocol-faithful matching of the deployed
project cap (`360*420`), which seven other Qwen2.5-VL entrypoints in this
repository already default to and which
`src/utils/generate_VideoMLLM_embedding_bidir_textpool_HF.py:29` calls "the
deployed max_pixels". **Judge that argument yourself.** If you think a
teacher-input change of this kind needs a design-level re-review rather than a
code/resource GO, say so as a finding.

Also check the fail-closed `VISUAL_PATCH_TOKEN_HARD_CEILING = 4096` in
`one_forward`: that it runs **before** `model.generate`, that it uses the
**pre-merge** count, and specifically judge the strict equality against the
preflight-frozen count — a mismatch HALTs a paid-for run mid-way. Is that the
right trade against the alternative of a warn-and-continue?

### CHANGE-3 — a measured pre-submit projection gate in the CPU preflight

`project_gpu_window` in the preflight. Attack the arithmetic and the basis.
Specific things worth doubting:

* the window is `5222 − 300 − 300 − 600 = 4022`. Is that the right window given
  how `BudgetGuard.at_job_start` and the wrapper `timeout` actually interact?
* the conservative projection yields **3807.9 s against 4022 s — a 5.3 % margin.**
  Is a 5.3 % margin on the stated basis acceptable, given that a breach is a
  clean exit-40 HALT that keeps every completed checkpoint but ends the tranche?
* the basis constants live in `configs/.../projection_basis`. They are claimed to
  come from sacct for job 13852 and from that job's own 400 prompt-record
  checkpoints. **Recompute them.** `sacct -X -n -P -j 13852 -o ElapsedRaw` and
  the `elapsed_seconds` field across
  `artifacts/c04/a0t_small_v1_impl_v7/checkpoints/HateMM/prompt_records/*.json`.
* MHC-ZH forwards are priced at the HateMM mean and this is the one unmeasured
  input. Is declaring it enough, or is it a finding?

### CHANGE-4 — frame packs move from the GPU job to the CPU preflight

The 400 eight-frame packs are now written and hash-pinned by the preflight; the
producer only loads them and HALTs if one is missing. This is the largest
structural change and it exists because ~700 s of measured decode+PNG-encode work
was being done with the A100 idle.

Attack:

* the frame-pack binding drops `payload_review_sha256`,
  `gpu_execution_authorization_sha256` and `allocation_claim_sha256` (which do
  not exist at preflight time) and gains `code_resource_authorization_sha256` and
  `teacher_max_pixels`. **Is anything now unbound that was bound in v7?** The
  claim is that all 400 manifests enter `staged_output_hashes`, each manifest
  pins its own eight PNG digests, and the chain is re-established through the
  payload review. Verify that chain end to end.
* `write_frame_pack` / `strict_validate_frame_pack` moved into the common module
  and are now called by two stages. Check for one-sided key-set drift — that is
  the v7 round-1 Critical and round-2 High shape.
* the pyav fallback changed from one pass to two (`_pyav_decode_exact`). Is the
  output really identical, and can the second pass ever select a different frame?
* the preflight now creates its temp namespace **before** the projection gate
  runs. Confirm that a failed gate leaves nothing behind and that the no-clobber
  namespace still appears only at the single atomic rename.
* is there any remaining path by which the producer could decode a video?

## What the builder measured (verify, do not trust)

19 dry-execution checks, 0 failed, including: 65 self-test fixtures; the four
prompt hashes still equal the v6 freeze; selection reproduces the v6/v7 frozen
allowlists exactly on both datasets; **800 real renderings through the
containment guard, 0 halts**; v8 frame packs byte-identical to v7's on 12 sampled
HateMM items; the two-pass pyav fallback byte-identical to the one-pass version
on the real fallback item (`MHC_zh/BV18N4y1B7qA`, 2879 frames, 45.8 s → 6.9 s);
the OOM closed form reproducing 110.50 GiB; and config-contract neutrality of the
authority flip.

## Carried-forward v7 round-5 observations

1. `NO_SEAL_SENTINEL` is now imported by the gpu_ledger module, not redefined.
2. A template-tamper fixture containing **no** banned token was added, plus a
   companion fixture proving the injection carries no banned token.

Confirm both, and confirm the v7 hardenings (I-1 containment precondition, I-2
known-answer vector, I-3 budget guard + campaign ledger, the wrapper
authorization gate before `mkdir`) survived the rebuild intact.

## Verdict format

`GO (0C/0H/0I)`, or `REVISE` with each finding as Critical / High / Important,
naming file and line. A finding that an irreversible step precedes the check
that would reject it is automatically at least High — that family has now cost
this lineage four separate incidents, the most recent being 1978 GPU-seconds and
the entire tranche.
