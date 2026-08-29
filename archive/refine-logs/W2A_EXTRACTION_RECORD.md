# W2-A Stage-E′ Full Extraction Record

**Method line:** W2-A = transcript-first grounded-key extraction (joint forward `[transcript][frames][instruction]` through Qwen2.5-VL-7B; vision-span pool = grounded retrieval key).

**Scope of this record:** RAW transcription only — numbers copied verbatim from the SLURM log. No probe interpretation; the probe verdict reviewer rules later.

---

## 1. Provenance

| Field | Value |
|---|---|
| SLURM job id | 13180 |
| SLURM completion | COMPLETED 0:0 |
| Log file | `/data/jehc223/RGCL/slurm/logs/w2a_extract_13180.log` (377 lines) |
| Host | foscsmlprd01.its.auckland.ac.nz |
| Start (UTC) | 2026-07-16T01:05:45Z |
| End (UTC) | 2026-07-16T02:09:48Z |
| GPU | NVIDIA A100-SXM4-80GB, 81920 MiB |
| HateMM wall | DONE dataset=HateMM in 1686.4s |
| MHC wall | DONE dataset=MHC in 2133.5s |

### Script hashes (banner, log lines 21-25) — MATCH EXPECTED r2c

| Artifact | sha256 (from log) | Expected | Match |
|---|---|---|---|
| `scripts/analysis/w2a_extract.py` | `9e984d61e2bf91d58f15af5e54f14d45a3fabe4e0701ce4492645399d810fa31` | `9e984d61e2bf91d58f15af5e54f14d45a3fabe4e0701ce4492645399d810fa31` | ✅ |
| `scripts/slurm/w2a_extract.sbatch` | `9ed04c14d16799d24e196f1d956698017373e597fd13e0cb2df6919087315153` | `9ed04c14d16799d24e196f1d956698017373e597fd13e0cb2df6919087315153` | ✅ |
| parity-by-import source `src/utils/generate_VideoMLLM_embedding_HF.py` | `d89a912602d763aa055a54f50b0188e302e554b70ff6c0eb872f250bd454b67c` | (parity import) | recorded |

On-disk re-verification (independent `sha256sum` at record time) reproduced both script hashes exactly.

### Config banner (log lines 26, 33, 114)

- `NUM_FRAMES=8 SMOKE=0`
- `model=Qwen/Qwen2.5-VL-7B-Instruct max_pixels=151200 dtype=bfloat16 attn=sdpa transformers=4.49.0`
- grounded transcript block: raw text | empty->'(none)' ; grd_pfx=[first_vis:end]

### self_test (no GPU/model) — PASS both datasets

- HateMM (log line 38): `[self_test] PASS — builders, span indexing, pools, gate-0 raises, placebo pairing all OK.`
- MHC (log line 119): `[self_test] PASS — builders, span indexing, pools, gate-0 raises, placebo pairing all OK.`

---

## 2. Verbatim saved-lines (six splits)

```
[HateMM/train] saved N=744 guard=1 empty=39 grecon_cos_min=0.9999995231628418 grecon_maxabs_max=0.0 grounding_present_median=0.9368 grounding_VOID=False placebo_median=0.9804 placebo_VOID=False -> /data/jehc223/RGCL/data/CLIP_Embedding/HateMM/grounded_qwen7b_8f/train_grounded.pt
[HateMM/dev_seen] saved N=107 guard=0 empty=9 grecon_cos_min=0.9999997019767761 grecon_maxabs_max=0.0 grounding_present_median=0.9475 grounding_VOID=False placebo_median=0.9812 placebo_VOID=False -> /data/jehc223/RGCL/data/CLIP_Embedding/HateMM/grounded_qwen7b_8f/dev_seen_grounded.pt
[HateMM/test_seen] saved N=215 guard=0 empty=26 grecon_cos_min=0.9999996423721313 grecon_maxabs_max=0.0 grounding_present_median=0.9375 grounding_VOID=False placebo_median=0.9826 placebo_VOID=False -> /data/jehc223/RGCL/data/CLIP_Embedding/HateMM/grounded_qwen7b_8f/test_seen_grounded.pt
[MHC/train] saved N=549 guard=0 empty=0 grecon_cos_min=0.9999995231628418 grecon_maxabs_max=0.0 grounding_present_median=0.9605 grounding_VOID=False placebo_median=0.9711 placebo_VOID=False -> /data/jehc223/RGCL/data/CLIP_Embedding/MHC/grounded_qwen7b_8f/train_grounded.pt
[MHC/dev_seen] saved N=80 guard=0 empty=0 grecon_cos_min=0.9999997019767761 grecon_maxabs_max=0.0 grounding_present_median=0.9609 grounding_VOID=False placebo_median=0.9709 placebo_VOID=False -> /data/jehc223/RGCL/data/CLIP_Embedding/MHC/grounded_qwen7b_8f/dev_seen_grounded.pt
[MHC/test_seen] saved N=161 guard=0 empty=0 grecon_cos_min=0.9999997019767761 grecon_maxabs_max=0.0 grounding_present_median=0.9602 grounding_VOID=False placebo_median=0.9692 placebo_VOID=False -> /data/jehc223/RGCL/data/CLIP_Embedding/MHC/grounded_qwen7b_8f/test_seen_grounded.pt
```

(Log lines: HateMM/train 89, HateMM/dev_seen 96, HateMM/test_seen 108, MHC/train 294, MHC/dev_seen 322, MHC/test_seen 373.)

---

## 3. Gate table (per split, raw)

Gate thresholds checked: `grecon_cos_min ≥ 0.9999`; `grecon_maxabs_max ≤ 1e-3`; `grounding_VOID=False` (K2, with `grounding_present_median < 0.999` required for GroundingLive); `placebo_VOID=False` (K3). Expected N: HateMM 744/107/215, MHC 549/80/161.

| Split | N (exp) | grecon_cos_min | ≥0.9999 | grecon_maxabs_max | ≤1e-3 | grounding median | grounding_VOID | <0.999 | placebo median | placebo_VOID | guard | empty | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HateMM/train | 744 (744) | 0.9999995231628418 | ✅ | 0.0 | ✅ | 0.9368 | False | ✅ | 0.9804 | False | 1 | 39 | **PASS** |
| HateMM/dev_seen | 107 (107) | 0.9999997019767761 | ✅ | 0.0 | ✅ | 0.9475 | False | ✅ | 0.9812 | False | 0 | 9 | **PASS** |
| HateMM/test_seen | 215 (215) | 0.9999996423721313 | ✅ | 0.0 | ✅ | 0.9375 | False | ✅ | 0.9826 | False | 0 | 26 | **PASS** |
| MHC/train | 549 (549) | 0.9999995231628418 | ✅ | 0.0 | ✅ | 0.9605 | False | ✅ | 0.9711 | False | 0 | 0 | **PASS** |
| MHC/dev_seen | 80 (80) | 0.9999997019767761 | ✅ | 0.0 | ✅ | 0.9609 | False | ✅ | 0.9709 | False | 0 | 0 | **PASS** |
| MHC/test_seen | 161 (161) | 0.9999997019767761 | ✅ | 0.0 | ✅ | 0.9602 | False | ✅ | 0.9692 | False | 0 | 0 | **PASS** |

All six N counts match the pre-registered split sizes. All four gates PASS on all six splits.

---

## 4. .pt output manifest (on-disk at record time)

Directory: `/data/jehc223/RGCL/data/CLIP_Embedding/{HateMM,MHC}/grounded_qwen7b_8f/`

| File | Size (bytes) | sha256 |
|---|---|---|
| HateMM/grounded_qwen7b_8f/train_grounded.pt | 21373758 | `1cae1f83739d6ed18c5c95a977d9ed880495f4c2dc9e27f4beaf45da9af29e6a` |
| HateMM/grounded_qwen7b_8f/dev_seen_grounded.pt | 3076517 | `41bda7dea0c6ea5ef5e68787117e6b7c8bbcca8eba69057347f2a81cc0125bd0` |
| HateMM/grounded_qwen7b_8f/test_seen_grounded.pt | 6178674 | `23634bcb6a608ee1540d2d0fa1eff91a4a2e8df6df70dd4c338ea3707a9df873` |
| MHC/grounded_qwen7b_8f/train_grounded.pt | 15769726 | `9f8da7a1b1c7d33e8b96975565f918f7785e2ece3be83244da343920f0a48767` |
| MHC/grounded_qwen7b_8f/dev_seen_grounded.pt | 2300773 | `7c1a1a4f5d350c51ef4ec4be5bfb41a3e84dba11fbe3d03658b380acae3eff20` |
| MHC/grounded_qwen7b_8f/test_seen_grounded.pt | 4626738 | `372640a30d16e952882d94ea839d97940da6285620e5f5fd8f01d634d4e9f2ea` |

All 6 `*_grounded.pt` present with plausible sizes (train > test > dev per dataset, tracking N). Companion `*_gatelog.json` (6 files) and per-split `_shards/` subdirs also present under each dataset directory.

---

## 5. K2 / K3 LIVE statements (per dataset)

**HateMM.** K2 GroundingLive: `grounding_present_median` = 0.9368 / 0.9475 / 0.9375 (train/dev/test), all `< 0.999` and `grounding_VOID=False` on all three splits — grounding channel is **LIVE** (not degenerate/collapsed to identity). K3 Placebo: `placebo_VOID=False` on all three splits (`placebo_median` = 0.9804 / 0.9812 / 0.9826) — placebo control is **LIVE**.

**MHC.** K2 GroundingLive: `grounding_present_median` = 0.9605 / 0.9609 / 0.9602 (train/dev/test), all `< 0.999` and `grounding_VOID=False` on all three splits — grounding channel is **LIVE**. K3 Placebo: `placebo_VOID=False` on all three splits (`placebo_median` = 0.9711 / 0.9709 / 0.9692) — placebo control is **LIVE**.

Reconstruction identity (K1-style): `grecon_cos_min` ∈ [0.9999995231628418, 0.9999997019767761] and `grecon_maxabs_max = 0.0` across all six splits — span-pool reconstruction is bit-tight against the joint-forward hidden states.

---

## 6. Decoder-fallback / guard notes (raw)

- **Total `decord failed` WARN lines: 203** (HateMM region: 1; MHC region lines 110-377: 202).
- **`PyAV failed` WARN lines: 1** (the single HateMM undecodable file below).
- **ZERO-GUARD events: 1.**

**HateMM — guard-handled (expected):** `hate_video_95.mp4` (log lines 60-68) failed decord (`partial file` / `av_read_frame failed`), then failed PyAV (`Invalid NAL unit size` / `Error splitting the input into NAL units`), then `no decodable frames` → `[356/744] hate_video_95 ZERO-GUARD (undecodable) -> zero keys`. This is the one HateMM/train `guard=1`. Matches the pre-noted known case.

**MHC — decord→PyAV fallback (benign, all recovered):** 202 MHC files hit the systematic decord error `Check failed: st_nb >= 0 ... cannot find video stream with wanted index: -1` and fell back to PyAV; **0 PyAV failures and 0 guards in the MHC region** (all three MHC splits: `guard=0 empty=0`), so every fallback decoded successfully via PyAV. The specifically pre-noted file `kec4e6w697w.mp4` appears once (log line 313) as one of these decord→PyAV fallbacks. Incidental `[h264] mmco: unref short failure` lines (362-365) during MHC/test are non-fatal decoder warnings; that split still saved N=161 guard=0 empty=0.

**`empty` counts** are empty-transcript rows mapped to `'(none)'` (per config banner "empty->'(none)'"), not decode failures: HateMM 39/9/26; MHC 0/0/0.

---

## 7. Summary verdict

- Banner hashes: **MATCH** expected r2c (extractor + sbatch). SMOKE=0, NUM_FRAMES=8 confirmed. self_test **PASS** both datasets.
- All six splits: N matches pre-registered sizes; all four gates (grecon_cos_min, grecon_maxabs_max, K2 grounding_VOID, K3 placebo_VOID) **PASS**.
- K2 grounding channel **LIVE** on both datasets (medians 0.9368-0.9609, all < 0.999). K3 placebo **LIVE** on both datasets.
- All six `*_grounded.pt` present on disk with plausible sizes; sha256s recorded above.
- Anomalies: none blocking. One HateMM ZERO-GUARD (`hate_video_95.mp4`, expected); systematic-but-recovered MHC decord→PyAV fallback (202 files, 0 residual failures).

**6/6 splits PASS. No gate failures, no hash mismatches.**
