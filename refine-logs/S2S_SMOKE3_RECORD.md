# S2S smoke attempt-3 — gate record (job 13182, r4 gate-0a′ extractor)

**Author:** s2s-implementer (gate-check + record; raw-only transcription — every number below is copied
verbatim from the log). **Date:** 2026-07-16. **Repo HEAD at check:** `c013884`.

This is the record of the **ONE authorized re-smoke** (amendment ruling `S2S_GATE0A_AMENDMENT_RULING.md`
§D.5; code review `S2S_CODE_REVIEW.md` §8 r4 verdict CLEARED-FOR-SMOKE-RESUBMIT) after the gate-0a→0a′
onset-invariance rewrite. The old permutation-equivariance/argmax gate 0a (smoke 13169) was invalid by
construction for cumulative-causal `g_t`; 0a′ tests a property guaranteed true under the causal mask
(a causal-prefix summary is invariant to any change strictly after it).

---

## 1. Provenance

| item | value |
|---|---|
| SLURM job id | **13182** |
| job state / exit | **COMPLETED**, ExitCode **0:0** (sacct) |
| elapsed | 00:00:37 (banner start `2026-07-16T02:10:16Z` → end `2026-07-16T02:10:52Z`, UTC) |
| host | foscsmlprd01.its.auckland.ac.nz |
| GPU | NVIDIA A100-SXM4-80GB, 81920 MiB |
| log | `slurm/logs/s2s_extract_13182.log` |
| throwaway out_root | `slurm/logs/s2s_smoke_out_13182/` (NOT the real cache dir) |
| config echo | `NUM_FRAMES=8 SMOKE=1`; per-run `dataset=… splits=train,val,test num_frames=8 device=cuda limit=1 out_root=<throwaway>` |
| model / parity | `Qwen/Qwen2.5-VL-7B-Instruct max_pixels=151200 dtype=bfloat16 attn=sdpa transformers=4.49.0`; parity-by-import `generate_VideoMLLM_embedding_HF.py` sha256 `d89a9126…54b67c` |

### Hash verification (banner + on-disk == frozen r4)

| artifact | expected r4 sha256 | banner (log:22-23) | on-disk (2026-07-16) | match |
|---|---|---|---|---|
| `scripts/analysis/s2s_extract.py` | `ce23dfe6…d83ff677` | `ce23dfe6810ee74a7311606b6992a747a7267e8754fc0554cd8c1f43d83ff677` | `ce23dfe6810ee74a7311606b6992a747a7267e8754fc0554cd8c1f43d83ff677` | ✅ |
| `scripts/slurm/s2s_extract.sbatch` | `2dc0f90b…d56665dc` | `2dc0f90b03a44f45945cab3194f78ec97012fe7b157727cd50f64d88d56665dc` | `2dc0f90b03a44f45945cab3194f78ec97012fe7b157727cd50f64d88d56665dc` | ✅ |

Real-path check: `data/CLIP_Embedding/HateMM/` and `data/CLIP_Embedding/MHC/` have **no**
`frameset_qwen7b_8f` directory — the smoke wrote artifacts only under the throwaway
`slurm/logs/s2s_smoke_out_13182/` path, per the ruling. ✅

---

## 2. Gate lines (verbatim from `slurm/logs/s2s_extract_13182.log`)

### Gate 0a′ — causal-prefix onset-invariance control (runs once per dataset, before any real video)

HateMM (log:37-38):
```
[gate 0a'] causal-prefix onset-invariance control: encoding 2 synthetic clips (shared frames 0-3, differing frames 4-7) ...
[gate 0a'] PASS: prefix groups invariant (cos 1.0000/1.0000 >= 0.999); changed groups diverge (max 0.9273 < 1.0000-0.002); groups distinct.
```
MHC (log:55-56):
```
[gate 0a'] causal-prefix onset-invariance control: encoding 2 synthetic clips (shared frames 0-3, differing frames 4-7) ...
[gate 0a'] PASS: prefix groups invariant (cos 1.0000/1.0000 >= 0.999); changed groups diverge (max 0.9273 < 1.0000-0.002); groups distinct.
```

Assertion arithmetic (from the printed values):
- **(1) prefix-invariance** shared groups {0,1}: `cos 1.0000/1.0000 ≥ 0.999` ✅ (both datasets)
- **(2) onset-divergence** changed groups {2,3}: `max(c2,c3)=0.9273 < min(c0,c1)−0.002 = 1.0000−0.002 = 0.998` ✅ (margin 0.0707)
- **(3) within-clip distinctness**: "groups distinct" printed (the code HALTs on `max off-diag > 0.999`; it did not HALT) ✅

Both datasets produce identical numbers (the control is deterministic synthetic input, model-only) — expected.

### Per-split assembly lines (verbatim) — carry gate 0b (implicit), gate 1 (decomp_max), gate 2 (grecon)

```
[HateMM/train]     saved N=1 T=4 guards=0 decomp_max=2.9802322387695312e-08 grecon_cos_min=0.9999998807907104 grecon_maxabs_max=0.0
[HateMM/dev_seen]  saved N=1 T=4 guards=0 decomp_max=2.9802322387695312e-08 grecon_cos_min=1.0                grecon_maxabs_max=0.0
[HateMM/test_seen] saved N=1 T=4 guards=0 decomp_max=1.4901161193847656e-08 grecon_cos_min=0.9999998807907104 grecon_maxabs_max=0.0
[MHC/train]        saved N=1 T=4 guards=0 decomp_max=2.9802322387695312e-08 grecon_cos_min=0.9999998807907104 grecon_maxabs_max=0.0
[MHC/dev_seen]     saved N=1 T=4 guards=0 decomp_max=1.4901161193847656e-08 grecon_cos_min=1.000000238418579  grecon_maxabs_max=0.0
[MHC/test_seen]    saved N=1 T=4 guards=0 decomp_max=2.9802322387695312e-08 grecon_cos_min=1.0                grecon_maxabs_max=0.0
```

---

## 3. Per-split table + per-gate verdict

| dataset/split | N | T | guards | decomp_max (≤1e-5) | grecon_cos_min (≥0.9999) | grecon_maxabs_max (≤1e-3) | 0b grid | verdict |
|---|---|---|---|---|---|---|---|---|
| HateMM/train | 1 | 4 | 0 | 2.98e-08 ✅ | 0.9999998808 ✅ | 0.0 ✅ | pass (silent) | GREEN |
| HateMM/dev_seen | 1 | 4 | 0 | 2.98e-08 ✅ | 1.0 ✅ | 0.0 ✅ | pass (silent) | GREEN |
| HateMM/test_seen | 1 | 4 | 0 | 1.49e-08 ✅ | 0.9999998808 ✅ | 0.0 ✅ | pass (silent) | GREEN |
| MHC/train | 1 | 4 | 0 | 2.98e-08 ✅ | 0.9999998808 ✅ | 0.0 ✅ | pass (silent) | GREEN |
| MHC/dev_seen | 1 | 4 | 0 | 1.49e-08 ✅ | 1.0000002384 ✅ | 0.0 ✅ | pass (silent) | GREEN |
| MHC/test_seen | 1 | 4 | 0 | 2.98e-08 ✅ | 1.0 ✅ | 0.0 ✅ | pass (silent) | GREEN |

Notes:
- `grecon_maxabs_max = 0.0` on all six splits = the fresh G-recon vector is **bit-identical** to the
  banked `img_feats` cache (same A100 + sdpa + bf16). The `grecon_cos_min` values `0.99999988…` /
  `1.0000002…` are float32 dot-product accumulation rounding of a unit vector against its bit-identical
  self — fully consistent with maxabs=0.0, and both far inside tolerance.
- `T=4` = `grid_t = num_frames//2 = 8//2` on every video, i.e. the 4 temporal groups the grid gate asserts.

### Per-gate GREEN/NOT-GREEN verdict

- **Gate 0a′ (causal-prefix onset-invariance)** — **GREEN, both datasets.** Prefix groups invariant
  (1.0000/1.0000 ≥ 0.999); changed groups diverge (0.9273 < 0.998, margin 0.0707); groups distinct.
  Explicit `[gate 0a'] PASS` printed for HateMM (log:38) and MHC (log:56).
- **Gate 0b (grid-consistency)** — **GREEN (silent-pass), both datasets, all splits.** *Not
  dropped/renamed.* 0b is a **HALT-on-violation** assertion inside `encode_frameset`
  (`s2s_extract.py:164-180`, `n_vis == grid_t·(grid_h//merge)·(grid_w//merge)` AND
  `(n_vis//T)==per_expected`); it prints **nothing on pass** by design, so its absence from the log is
  expected, **not** a drop. r4 amendment ruling §D.4 lists it "0b … HALT [UNCHANGED]"; r4 code review §8
  confirms the only r4 code change was the 0a→0a′ rewrite and `encode_frameset` (incl. 0b) is byte-
  untouched. Because 0b HALTs (uncaught `RuntimeError` → non-zero exit) on any violation and the job
  **COMPLETED 0:0** writing all 6 split outputs (plus the 2 synthetic clips per dataset that also pass
  through 0b), 0b passed for every encoded video. The `T=4` on every saved line is the grid-consistency
  invariant holding.
- **Gate 1 (G-decomp)** — **GREEN, all 6 splits.** `decomp_max` ∈ {1.49e-08, 2.98e-08} ≤ 1e-5 (≥ 3 orders
  of magnitude of headroom).
- **Gate 2 (G-recon)** — **GREEN, all 6 splits.** `grecon_cos_min ≥ 0.99999988 ≥ 0.9999` AND
  `grecon_maxabs_max = 0.0 ≤ 1e-3`. The stale `(G-recon skipped)` banner echo (log:27) is the known
  cosmetic NOTE (code review §5/§8); G-recon actually ran and is read from the assembly lines above.

**Banner sanity:** both script sha256s = frozen r4 (§1); `NUM_FRAMES=8 SMOKE=1`; `device=cuda`,
`limit=1`; throwaway `out_root`; exit `0:0`. ✅

---

## 4. Overall verdict

**ALL FOUR HARD GATES GREEN on ≥1 real video per dataset (HateMM + MHC), zero anomalies.** The r4
gate-0a′ onset-invariance rewrite passes on both datasets; grid, G-decomp, and G-recon pass on all six
splits with large margins and bit-exact banked parity. The re-smoke authorized by the amendment ruling
§D.5 / code review §8 is satisfied.

Per the queue-on-pass grant, Stage-E **full 8-frame extraction** is submitted (see §5).

---

## 5. Stage-E full extraction submission

Frozen-spec configuration submitted (single sbatch, both datasets sequential, `SMOKE=0`, real
`out_root=data/CLIP_Embedding`, `NUM_FRAMES=8`). Hashes re-verified == frozen r4 immediately before
submit.

- **Submitted job id: `<PENDING-SUBMIT>`** — `sbatch scripts/slurm/s2s_extract.sbatch` (no `--time`;
  `SMOKE` default 0, `NUM_FRAMES` default 8). Expected initial state `PENDING (JobHeldUser)` → wait for
  auto-release, never force. (Filled in by the amend commit after submission.)

### 16-frame sensitivity arm — DELIBERATELY NOT SUBMITTED (reported ambiguity)

The prereg pre-registers a 16-frame (T=8) sensitivity budget (`exp-s2s-r3.md` §6.6, §8), but as a
**"separate forward"** — i.e. a **separate submission event** (`NUM_FRAMES=16 sbatch …`), not part of the
single 8-frame sbatch. It is **held**, not submitted now, because:
1. The team-lead task mandates a **SINGLE submission event**; adding the 16-frame arm would be a second
   sbatch job (second event).
2. The just-passed smoke was **8-frame only** (`NUM_FRAMES=8`); there is **no 16-frame smoke** — the
   smoke-then-full discipline validated only the 8-frame config.
3. The 16-frame arm **cannot run G-recon** (no 16-frame banked cache; §6.6) — it is gated by G-decomp
   only, i.e. a weaker correctness anchor, and neither the amendment ruling §D.5 nor code review §8
   explicitly authorizes a 16-frame full run (both discuss "the full Stage-E extraction" after the
   8-frame smoke).

Per the task's explicit contingency ("if the frozen spec is ambiguous about the 16-frame arm … submit
only the unambiguous 8-frame full run and report the ambiguity"), only the 8-frame run is submitted; the
16-frame arm awaits explicit separate authorization (and, ideally, its own 16-frame smoke).
