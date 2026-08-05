# C04-A0T-SMALL-v1 impl-v8 — Payload Review Request

Date: 2026-08-01
Stage: post-CPU-preflight payload review
Requested verdict: `GO` or `REVISE` with findings

You are a **fresh independent payload reviewer**. Zero exposure to the build
reasoning and zero exposure to the earlier code/resource review. Your subject is
the **materialised payload** — the frozen artifacts the CPU preflight actually
produced — not the source code's intentions.

## What exists now, and what you are authorizing

CPU-only SLURM job `13855` (`COMPLETED`, exit `0:0`, elapsed 976 s, no GPU in
`AllocTRES`) published the no-clobber namespace
`artifacts/c04/a0t_small_v1_impl_v8/`. Nothing further has run. No GPU, teacher,
or SLURM execution is authorized; `teacher_authorized`, `gpu_authorized`,
`slurm_authorized` and `small_tranche_execution_authorized` are all `false`.

If you return `GO`, the next irreversible step is **one** A100 allocation that
consumes a single-use resource ticket and reserves 5222 GPU-seconds — the entire
remaining first-tranche budget.

## Anchor hashes

| artifact | value |
|---|---|
| preflight manifest | `artifacts/c04/a0t_small_v1_impl_v8/freeze/preflight_manifest.json`, SHA-256 `bd93adf8496f89af22febc7a76abd8957c59251d259be4cad3012ab1f4d50c1b` |
| code/resource authorization | `refine-logs/C04_A0T_SMALL_V1_V8_CODE_RESOURCE_AUTHORIZATION.json`, SHA-256 `c0a12d1a4cae0105dadfc477c9e9111a81361b087cea52cfd5de735b53150344` |
| config | `configs/c04/c04_a0t_small_v1_v8.json` |
| campaign accumulator | `artifacts/c04/campaign/gpu_ledger.json` (revision 1, aggregate 1978 s, phase FIRST_TRANCHE, phase cap 7200 s) |
| binding contract | `refine-logs/C04_USER_AMENDMENT_V2.md` |

## What the preflight claims it produced

* 414 `staged_output_hashes`, of which **400 are frame-pack manifests**
* 802 guarded-access-audit events
* 74 self-test checks, all passed
* the four prompt hashes materialised as literals, claimed equal to the v6 freeze
* 200 + 200 ID-only allowlists, claimed to reproduce the v6/v7 freeze exactly
* per-item visual geometry for all 400 items, and a projection gate reporting
  `fits_by_measurement: true`

## Measured numbers you should recompute, not accept

Visual geometry, measured through the Qwen2.5-VL `image_processor` on CPU at
`max_pixels=151200`:

| | HateMM | MHC-ZH |
|---|---|---|
| items | 200 | 200 |
| pre-merge patch tokens min | 1456 | 480 |
| median | 2880 | 2880 |
| max | 3072 | 3072 |
| mean | 2848.2 | 2865.4 |

Worst case over all 400: **3072 tokens**, vision SDPA fp32 **0.562 GiB** against a
79.27 GiB card and against the **110.50 GiB** that killed v7.

Decode backends: 388 `decord`, 11 `pyav`, 1 `none`. One item
(`HateMM/hate_video_95`) has all eight frames flagged decode-failed and is the
frozen all-black case — **v7 produced exactly the same all-black pack for it**.

Projection gate (in the manifest under `gpu_window_projection`):

| quantity | value |
|---|---|
| usable teacher window | 4022 s (5222 − 300 watchdog − 300 item margin − 600 seal reserve) |
| projected, conservative basis | 3817.6 s → margin **204.4 s (5.1 %)** |
| projected, capped-regime corroboration | 2597.8 s (≈35 % margin) |
| affordable mean per forward | 4.6803 s vs the 4.4248 s basis |

CPU work the preflight absorbed that v7 performed **inside its GPU allocation**:
decode 273.0 s + PNG write 561.7 s = **834.7 s**. Note what that implies and
please check the arithmetic yourself: had that work stayed in the GPU job, the
projection would be 3817.6 + 834.7 = 4652.3 s against a 4022 s window — the gate
would have **failed**.

## What to attack

1. **Recompute every hash.** Re-hash all 414 staged outputs against the manifest;
   re-verify the manifest's own `payload_sha256`; verify the resource ticket's
   `genesis_gpu_ledger_sha256` pin and the ticket's own digest.
2. **The allowlists.** Recompute the selection from
   `sha256("C04-A0T-SMALL-v1" || dataset || video_id || "20260729")` over the real
   train ASR files and confirm both 200-ID lists and their order. Confirm no label
   value was materialised (`projected_field_counters`).
3. **Label containment.** HateMM identifiers (`hate_video_*`, `non_hate_video_*`)
   *are* the label. Confirm nothing in the frozen payload that a teacher forward
   can read carries an identifier or a label-bearing substring. Confirm no
   dev/test path, no cross-dataset path, no OCR artifact, no network artifact.
4. **The frame packs.** 400 immutable eight-frame packs are new in v8 — v7 created
   them inside the GPU job. Spot-check that each manifest's eight PNG digests
   match the bytes on disk, that the binding fields are the ones the code
   re-derives, and that `requested_indices` follows the frozen rule
   `floor((i+0.5)*N/8)` clamped. Check the zero-frame item and the 11 pyav items.
5. **The geometry and the gate.** Independently re-measure a sample of items
   through the processor and compare to `visual_geometry.items`. Judge whether the
   5.1 % conservative margin is an acceptable basis for spending the entire
   remaining first-tranche budget, given that a breach is a clean exit-40 HALT
   that preserves every completed checkpoint but ends the tranche.
6. **Anything that would make the GPU stage unsatisfiable.** The failure family
   this lineage keeps producing is *an irreversible resource consumed before the
   check that would reject the run*. The GPU job consumes a single-use ticket
   before any payload work. Is there anything in the frozen payload that the GPU
   stage will re-read and reject?

## Prohibitions

Do not edit any file. Do not create or modify anything under
`artifacts/c04/`. Do not submit a SLURM job. Do not use a GPU. Scratch space:
`/data/jehc223/home/tmp/claude-135258174/-data-jehc223-RGCL/3ab1f506-990f-485a-8326-331bed01a558/scratchpad/payload_reviewer_r1/`

## Verdict format

`GO` with `nC/nH/nI`, or `REVISE` with findings by severity, each naming the
artifact and the exact field. State plainly what you recomputed and what each
recomputation gave.
