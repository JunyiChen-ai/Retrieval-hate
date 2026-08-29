# S2S Stage-E full extraction — gate record (job 13189, 8-frame, r4 extractor)

**Author:** s2s-implementer (full-split gatelog verification; raw-only — every number copied verbatim
from the log / gatelog JSONs). **Date:** 2026-07-16. **Repo HEAD at check:** `77ed845`.

Full Stage-E 8-frame extraction for HateMM + MHC-EN (train/val/test), submitted queue-on-pass after the
smoke-3 four-gate GREEN (`S2S_SMOKE3_RECORD.md`, job 13182). This record verifies the four HARD gates
over **full N** and provides the `.pt` cache manifest. **Stage P is NOT run here** (separate grant).

---

## 1. Provenance

| item | value |
|---|---|
| SLURM job id | **13189** |
| job state / exit | **COMPLETED**, ExitCode **0:0** (sacct, both `13189` and `13189.batch`) |
| elapsed | 00:41:34 (banner start `2026-07-16T02:17:26Z` → end `2026-07-16T02:59:00Z`, UTC) |
| per-dataset | HateMM DONE in 949.1s (log:106); MHC DONE in 1519.1s (log:370); sbatch reached `real run complete for HateMM + MHC (8 frames)` (log:371) → `set -e` success tail |
| host / GPU | foscsmlprd01.its.auckland.ac.nz / NVIDIA A100-SXM4-80GB, 81920 MiB |
| log | `slurm/logs/s2s_extract_13189.log` (372 lines) |
| config echo | `NUM_FRAMES=8 SMOKE=0`; per-run `splits=train,val,test num_frames=8 device=cuda limit=0 out_root=/data/jehc223/RGCL/data/CLIP_Embedding` (real cache dir) |
| model / parity | `Qwen/Qwen2.5-VL-7B-Instruct max_pixels=151200 dtype=bfloat16 attn=sdpa transformers=4.49.0`; parity-by-import `generate_VideoMLLM_embedding_HF.py` sha256 `d89a9126…54b67c` |

### Hash verification (banner == frozen r4; still on-disk r4 at check)

| artifact | frozen r4 sha256 | banner (log:22-23) | match |
|---|---|---|---|
| `scripts/analysis/s2s_extract.py` | `ce23dfe6…d83ff677` | `ce23dfe6810ee74a7311606b6992a747a7267e8754fc0554cd8c1f43d83ff677` | ✅ |
| `scripts/slurm/s2s_extract.sbatch` | `2dc0f90b…d56665dc` | `2dc0f90b03a44f45945cab3194f78ec97012fe7b157727cd50f64d88d56665dc` | ✅ |

---

## 2. Gate 0a′ — causal-prefix onset-invariance control (verbatim)

Runs once per dataset before any real video. HateMM (log:37-38), MHC (log:117-118) — **identical**:
```
[gate 0a'] causal-prefix onset-invariance control: encoding 2 synthetic clips (shared frames 0-3, differing frames 4-7) ...
[gate 0a'] PASS: prefix groups invariant (cos 1.0000/1.0000 >= 0.999); changed groups diverge (max 0.9273 < 1.0000-0.002); groups distinct.
```
- prefix-invariance {0,1}: `1.0000/1.0000 ≥ 0.999` ✅ · onset-divergence {2,3}: `0.9273 < 0.998` (margin 0.0707) ✅ · within-clip distinct ✅ — **GREEN, both datasets.**

---

## 3. Per-split gate table (from the 6 `*_gatelog.json`, cross-checked to the log saved lines)

| dataset / split (outname) | N | expected N | T | guards | decomp_res_max (≤1e-5) | grecon_cos_min (≥0.9999) | grecon_maxabs_max (≤1e-3) | grecon_n_checked | verdict |
|---|---|---|---|---|---|---|---|---|---|
| HateMM / train (train) | **744** | 744 ✓ | 4 | 1 | 5.960464e-08 ✅ | 0.9999995232 ✅ | 0.0 ✅ | 743 | GREEN |
| HateMM / val (dev_seen) | **107** | 107 ✓ | 4 | 0 | 5.960464e-08 ✅ | 0.9999997020 ✅ | 0.0 ✅ | 107 | GREEN |
| HateMM / test (test_seen) | **215** | 215 ✓ | 4 | 0 | 5.960464e-08 ✅ | 0.9999996424 ✅ | 0.0 ✅ | 215 | GREEN |
| MHC / train (train) | **549** | 549 ✓ | 4 | 0 | 5.960464e-08 ✅ | 0.9999995232 ✅ | 0.0 ✅ | 549 | GREEN |
| MHC / val (dev_seen) | **80** | 80 ✓ | 4 | 0 | 5.960464e-08 ✅ | 0.9999997020 ✅ | 0.0 ✅ | 80 | GREEN |
| MHC / test (test_seen) | **161** | 161 ✓ | 4 | 0 | 5.960464e-08 ✅ | 0.9999997020 ✅ | 0.0 ✅ | 161 | GREEN |

- **All six N counts match the pre-registered expected split sizes exactly** (HateMM 744/107/215, MHC 549/80/161; prereg §6.1).
- `grecon_maxabs_max = 0.0` on **all** 1855 non-guard videos = the fresh forward is **bit-identical** to
  the banked `img_feats` cache (same A100 + sdpa + bf16). `grecon_cos_min` values `0.9999995…` are
  float32 dot-product accumulation rounding of a unit vector against its bit-identical self — consistent
  with maxabs=0.0, far inside the ≥0.9999 bar.
- `grecon_n_checked` = N minus guards on every split (HateMM/train 743 = 744−1 guard; all others = N) —
  the guard row correctly carries no banked-vec compare.

### Per-gate GREEN/NOT-GREEN verdict (over full N)

- **Gate 0a′ (causal-prefix onset-invariance)** — **GREEN**, both datasets (§2).
- **Gate 0b (grid-consistency)** — **GREEN (silent-pass)**, every encoded item. 0b is the HALT-on-
  violation assertion inside `encode_frameset` (`s2s_extract.py:164-180`) that prints nothing on pass by
  design; because it HALTs (uncaught `RuntimeError` → non-zero exit) on any violation and the job
  **COMPLETED 0:0** writing all 6 split outputs, 0b passed for all **1856** encoded videos + the 4
  synthetic control clips. `T=4` (=grid_t=8//2) on every saved line + gatelog is the grid invariant
  holding. (Unchanged in r4 per amendment ruling §D.4 / code review §8.)
- **Gate 1 (G-decomp)** — **GREEN**, all 6 splits: `decomp_res_max = 5.96e-08 ≤ 1e-5` (≈2.3 orders of headroom).
- **Gate 2 (G-recon)** — **GREEN**, all 6 splits: `grecon_cos_min ≥ 0.9999995 ≥ 0.9999` AND
  `grecon_maxabs_max = 0.0 ≤ 1e-3`. (Stale `(G-recon skipped)` echo does not appear in the full-run
  branch; it is a smoke-branch-only cosmetic NOTE.)

---

## 4. Anomalies — all benign, all pre-expected

- **1 ZERO-GUARD (undecodable):** `hate_video_95` (HateMM/train, item 356/744; log:65). Both decord AND
  PyAV fail on genuinely corrupt data (log:58 decord `av_read_frame failed 1094995529`; log:62-63 PyAV
  `Invalid data found … Error splitting the input into NAL units`). Handled identically in both retrieval
  arms as a zero frame set (prereg §8). **Matches the banked cache's 1 HateMM-train zero-img guard row**
  (prereg provenance) — expected, not a defect.
- **203 decord→PyAV fallbacks** (`trying PyAV`): **1 in HateMM/All** (= `hate_video_95`, which then also
  fails PyAV → the one guard) and **202 in MHC/All** (all succeed via PyAV → **0 MHC guards**). This is
  the known systematic MHC decord→PyAV fallback (~200 files); the decord warnings are
  `cannot find video stream with wanted index` — benign, PyAV decodes them.
- **No** Traceback / RuntimeError / HALT / OOM / Killed anywhere in the log; the only `Error` substrings
  are inside the two benign decode-fallback WARN lines for `hate_video_95`. Exit `0:0`.

---

## 5. `.pt` cache manifest (real cache dir; sha256 + size in bytes)

Under `data/CLIP_Embedding/<ds>/frameset_qwen7b_8f/`:

| file | size (bytes) | sha256 |
|---|---|---|
| `HateMM/train_frameset.pt` | 26714616 | `10d53f77487058e7df6015ce96d696ab0dc69d4a2d48c13225895463c04191e2` |
| `HateMM/dev_seen_frameset.pt` | 3844380 | `34912910e1cba254f45112f664db7897162dd9c75dd597908c888f2addac4213` |
| `HateMM/test_seen_frameset.pt` | 7721832 | `c1b51dc8b8114d4b81518517c2b7bd16bdcdf037d0d913159a79dc572fc8829b` |
| `MHC/train_frameset.pt` | 19710584 | `9423a818bf22e0d7767e58a03a1e9a3995c5da6f4473afd6802f333d470f2c28` |
| `MHC/dev_seen_frameset.pt` | 2874780 | `d9f4c21d92ddd9ce59f379b8a9a53e288630474d7b09ed36dac511b2f96be3c1` |
| `MHC/test_seen_frameset.pt` | 5782248 | `7afd65ea1f1e07c1a71e8e00ce91ad9140ef3be5765c2b08a46418b4154c945c` |

Total ≈ 66.6 MB (fp16 `{g:[N,4,3584], n_t, p_S, S, end, labels, grid_thw, zero_guard}` per split),
sub-GB as estimated (prereg §12). Each also has a sibling `*_gatelog.json` (the §3 numbers).

**Test-touch note:** the two `test_seen_frameset.pt` files are *extracted and cached* for the later
formal stage but are **NOT scored** — this is the prereg §10 authorized extraction (0 test-touch;
scoring is a Stage-P/formal-stage concern and Stage P retrieves train∪val only, N4-guarded).

---

## 6. Overall verdict

**ALL FOUR HARD GATES GREEN over full N, zero unexpected anomalies.** Banner hashes == frozen r4; N
counts == pre-registered split sizes exactly; G-decomp ≤ 5.96e-08 and G-recon bit-exact (maxabs 0.0,
cos ≥ 0.9999995) on all six splits; the single guard (`hate_video_95`) and the 202 MHC PyAV fallbacks are
the known-benign cases. The Stage-E 8-frame frame-set caches for HateMM + MHC-EN are complete and
verified. **Stage P remains gated behind a separate grant** (not run here); the 16-frame arm stays HELD
(ratified) until the 8-frame probe verdict.
