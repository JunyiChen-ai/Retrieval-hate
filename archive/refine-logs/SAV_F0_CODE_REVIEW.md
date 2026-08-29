# SAV (C2) F-G0/F-G1 — independent code review

**Reviewer:** fresh zero-prior-context independent code reviewer. Read-only except this doc.
Cheap CPU python verification only; **no SLURM job submitted, nothing committed.**
**Date:** 2026-07-13. **Env:** conda `HateVideo` (Python 3.11.8).

**Authority:** `research-wiki/experiments/exp-sav-f0.md` (Rev-2b, APPROVED).
**Code under review:** `scripts/analysis/sav_f0_{common,extract,guard,probe}.py`,
`scripts/wrappers/sav_f0.sh`, `scripts/slurm/sav_f0.sbatch`.
**Impl self-audit checked (not trusted):** `refine-logs/SAV_F0_IMPL_NOTES.md`.

## VERDICT: **APPROVED.**

Every load-bearing claim was re-verified against primary sources (the real transformers
4.49.0 modeling source, the local model config, the real gt/cache files, the banked
extractor, and a live end-to-end execution of the statistics engine). The code faithfully
implements the Rev-2b spec, is fail-closed in the right places, and the one non-trivial
real-data edge case (the undecodable `hate_video_95`) was empirically confirmed to pass the
reproduction guard. No blocking items. Three non-blocking notes at the end. Smoke
prescription for the executor follows the verdict.

---

## Checklist 1 — Runtime cross-check static simulation (load-bearing rows recomputed)

I rebuilt the load-bearing rows independently against the **real** data (not the impl's table):

| quantity | code expectation | independent recomputation | verdict |
|---|---|---|---|
| HateMM train/val | 744 / 107 | `wc -l data/gt/HateMM/{train,val}.jsonl` = 744 / 107 | ✅ |
| MHC train/val | 549 / 80 | 549 / 80 | ✅ |
| MHC_zh train/val | 579 / 78 | 579 / 78 | ✅ |
| cache rows == counts | EXPECTED_COUNTS | loaded all 6 `*_Qwen2.5-VL-7B-Instruct_HF.pt`: rows = 744/107, 549/80, 579/78, shape `[N,3584]` | ✅ |
| cache id order == gt order | dict-lookup + assert | `ids == gt_order` **True** for all 6 splits (guard/probe alignment robust regardless) | ✅ |
| class minima ≥ SELECTION_PER_CLASS(20) | `_selection_draw` assert | HateMM min=298, MHC min=168, MHC_zh min=180 (all ≥20) | ✅ |
| geometry (L,H,hidden,kv) | (28,28,3584) kv=4 | `config.json`: 28 / 28 / 3584 / 4 → head_dim 128; 28×28=784 | ✅ |
| o_proj module count filter | `len(found)==28` | source: 28 decoder layers `self_attn.o_proj`; vision uses `.attn.proj` (see Ck6) | ✅ |
| RUN_ID guard | `RUN_ID != EXPECTED → exit 2` | wrapper L17-25; sbatch passes `RUN_ID=EXPECTED=SAV-F0-FG0-FG1` | ✅ |
| jq gate strings | 3 fail-closed gates | `.complete==true and (.n==.n_expected)`; `.pass==true`; `.status=="COMPLETE"` | ✅ |
| hate_video_95 zero-guard | fresh zero must match cached zero | **empirically confirmed** (see Ck4) | ✅ |
| projected-gain bar | 0.030+0.010=0.040 | constants only; flows to verdict + `_k_pass_mhc` | ✅ |

## Checklist 2 — Deferred-import audit (fresh evidence in HateVideo)

`grep -nE '^[[:space:]]+(import|from)'` over all 4 modules → the **only** function-level
imports are `sav_f0_extract.py:56-57` (`import decord; from decord import VideoReader, cpu`,
inside `_decode_with_decord`) and `:71` (`import av`, inside `_decode_with_pyav`). Both are
**verbatim from the frozen extractor** (`generate_VideoMLLM_embedding_HF.py:156-157,171`) and
are observational-only (they produce PIL frames; they do **not** alter the model forward).
No `try/except ImportError`, no environment-conditional imports, no monkeypatching.

Fresh `python -c` evidence in `HateVideo`:
`numpy 1.26.4 · torch 2.6.0+cu124 · transformers 4.49.0 · PIL 11.1.0 · decord 0.6.0 ·
av 17.0.0 · sklearn 1.5.2`. `jq` = `/usr/bin/jq` (jq-1.6). All resolve. ✅

## Checklist 3 — Full-chain handoff / env audit

- **Fail-closed jq gates.** Wrapper `set -euo pipefail`; each gate is `jq -e '<pred>' … || {
  echo …; exit N; }`. A missing/malformed JSON → `jq -e` non-zero → stage aborts. Gate1 exit3,
  gate2 exit4, gate3 exit5. The guard also `sys.exit(3)` on primary fail (belt-and-braces) —
  under `set -e` the wrapper aborts there with the guard.json already written. Fail-closed. ✅
- **RUN_ID propagation.** sbatch → `RUN_ID=SAV-F0-FG0-FG1 EXPECTED=… bash wrapper`; wrapper
  refuses any mismatch (exit 2). ✅
- **In-repo temp only.** `atomic_write_json`/`atomic_torch_save` use
  `tempfile.mkstemp(dir=<target parent>)` + `os.replace` (same in-repo dir). `$TMPDIR` never
  used (grep-confirmed) — the realbank `$TMPDIR`-escape burn is avoided. `mkdir -p slurm/tmp`
  is an unused in-repo placeholder. ✅
- **Paths.** Reads `data/gt/<ds>/{train,val}.jsonl`, symlinked `data/video/<ds>/All/<id>.mp4`,
  banked `data/CLIP_Embedding/<ds>/{train,dev_seen}_<tag>.pt`, HF cache. Writes only under
  `artifacts/sav_f0/…` + `slurm/logs/`. All in-repo. ✅

## Checklist 4 — Storage topology (symlinked mp4s)

- Extractor decodes `data/video/<ds>/All/<id>.mp4` (the symlinked mp4s) via the same
  decord→PyAV sampler, **not** `data/lora_frames/`. Per video it records `is_symlink`
  (`os.path.islink` = lstat), `followed_target` (`os.path.realpath`), and
  `followed_target_in_repo`; external targets are **expected and allowed** (audit-only, no gate).
- **2 real paths/dataset verified live** — all resolve to the external stores and are readable:
  `HateMM/All/hate_video_100.mp4 → /data/jehc223/HateMM/video/… (2.6 MB)`,
  `MHC/All/01ygFLVdj8s.mp4 → /data/jehc223/Multihateclip/English/video_mp4/… (2.5 MB)`,
  `MHC_zh/All/BV117421N7HM.mp4 → /data/jehc223/Multihateclip/Chinese/video/… (1.7 MB)` (+1 more each). ✅
- **Undecodable-video edge case empirically resolved.** `hate_video_95` (the one cached zero in
  HateMM/train) is a **present** symlink (target 37.5 MB) but is a **partial/corrupt** file:
  decord raises, and running the *real* `sav_f0_extract.load_video_frames` on it returns
  `(None, False)` (PyAV also fails: `Invalid data … Error splitting the input into NAL units`).
  So the fresh extractor emits a **zero** payload, exactly matching the banked zero vector, and
  the guard classes it `zero_matched` (excluded from the min-cosine). The guard PASSES on this
  real video. **This is the single most important real-data check and it holds.** ✅

## Checklist 5 — Sampler-verbatim claim (line-by-line vs `generate_VideoMLLM_embedding_HF.py`)

Diffed the frame sampler + message build + span pooling:

- `_sample_frame_indices` — **identical** (linspace/round/clip/tolist). ✅
- `_decode_with_decord` / `_decode_with_pyav` — **identical** logic (native bridge, index set,
  nearest-fallback). ✅
- `_build_messages` — **identical** (one user turn: video frames + text). ✅
- `forward_once` vs `_encode`: **identical** `apply_chat_template(add_generation_prompt=True)`,
  `processor(text=[text], images=None, videos=[frames])`, `model(..., output_hidden_states=True,
  use_cache=False)`, `last_hidden = hidden_states[-1][0]`, the length assert, the `<|im_start|>`
  boundary for **prefix** (`last_hidden[:boundary].mean`) and **response**
  (`last_hidden[start:].mean`), and `normalize(pooled.float(), p=2, dim=0).cpu()`. Frame count 8,
  `max_pixels = 360*420 = 151200` (set at processor construction), bf16, `attn="sdpa"`,
  `device_map=None` — all match the banked pins. ✅
- **Only deviation:** on a *missing* file, `load_video_frames` **raises** `FileNotFoundError`
  (fail-closed) where the reference returns `(None,False)`→zero. This is safer and cannot
  diverge in practice (the sole cached zero, `hate_video_95`, is present-but-undecodable, which
  both pipelines handle identically). Non-blocking (Note 1).

The hooks are **non-mutating** (`register_forward_pre_hook` returning `None`), so the pooled
read-out is numerically identical to the hookless banked run — the F-G0(b) guard is a real,
un-confounded reproduction check.

## Checklist 6 — Hook correctness (verified against the real transformers 4.49.0 source)

Read `…/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`:

- **o_proj input layout (SDPA path, the kernel actually used).** `Qwen2_5_VLSdpaAttention`
  (L917) L1000-1003: `attn_output = attn_output.transpose(1,2).contiguous(); attn_output =
  attn_output.view(bsz, q_len, self.hidden_size); attn_output = self.o_proj(attn_output)`. So
  the o_proj **input** is `[bsz, q_len, num_heads*head_dim]` with heads **contiguous-major**
  (the eager path L795-798 is identical). The hook captures `args[0][0]` = `[seq, 3584]` and
  reshapes `.view(-1, 28, 128)` → `[seq, head, dim]` in the **correct** order
  (flat index = head*128 + dim). ✅✅
- **Module filter.** `o_proj` (L735, `in_features=28*128=3584`) belongs to the text decoder
  attention (`self.self_attn`, DecoderLayer L1025); the vision tower uses `self.attn.proj`
  (L339 → Linear L178/240/289) — names end `…attn.proj`, **not** `self_attn.o_proj`. So
  `name.endswith("self_attn.o_proj")` selects exactly the **28** decoder layers; `len(found)==28`
  assert holds, and `in_features==3584` per module is asserted. ✅
- **Final-token / span indices.** `img_head_final = t[-1]` (last token of the full sequence at
  every layer = SAV's `x_i^T`); `img_hidden_final = last_hidden[-1]` (C-pos, same final token,
  last-layer full hidden); `img_head_spanmean = t[span_slice].mean` over `slice(0,boundary)` =
  the same prefix span as the cached pooling (C-sparse, per-head identity kept). Consistent with
  the intended pooled / C-pos / SAV / C-sparse axis isolation. ✅
- **VRAM / leak / cleanup.** Hook stores `inp[0].detach()` (GPU); after the img forward each
  layer is `.float().cpu()`-copied and `capture.reset()` clears the buffer per video (no
  cross-video accumulation). Hooks are `enabled=False` during the text forward (fire but return
  immediately) — no capture, no leak. Registered once in `main`, **removed in a `finally`**. ✅
- **Batch/shape guards.** Hook asserts `inp.dim()==3 and inp.shape[0]==1 and inp.shape[-1]==3584`;
  per-layer `t.shape[0]==seq` and `len(capture.buf)==28` asserted. ✅

## Checklist 7 — Statistics engine vs prereg (executed the real code end-to-end)

Verified line-by-line **and** by running the **real** functions on synthetic caches (small
geometry to keep the 100,352-d U-1 fast; monkeypatched `load_extracted_split` only):

- **Primitives** (numeric unit checks pass): `fano_acc(0)=1.0`, `fano_acc(1)=0.5`; `h2inv_lower`
  inverts H₂ to <1e-3; `per_example_bits` gives 1 bit at p=0.5 and ~19.9 bits (clipped, not inf)
  at p→0; `clustered_bootstrap_mean` excludes 0 for a constant +0.05 delta and does **not** for
  zero-mean noise; `clustered_bootstrap_projection` positive+excludes-0 when SAV codelength <
  pooled; `head_nearest_centroid_accuracy` = 1.0 on a perfectly separable head. ✅
- **5 seeds vary the right things (R1a).** `_selection_draw` (20/class, without replacement,
  `rng=1000+seed`) and `_stratified_indices` (80% stratified, `rng=2000+seed`) — genuine
  per-seed variation; head-set stability reported (my run: k=10 intersection 5, Jaccard 0.55 —
  non-degenerate). ✅
- **Clustered bootstrap (R1b).** `dL_ex=(pooled-arm).mean(axis=1)` and `dacc_ex=(arm-pooled).mean
  (axis=1)` average **across seeds first** (axis=1), then resample the **examples** (axis=0);
  `n_effective == n_val`. Seeds reduce variance, do not multiply n. ✅
- **MDL = holdout log-loss (R1c).** `per_example_bits` = −log₂ p̂ over the **val** holdout, probe
  fit on the seed's 80% train only, clip [1e-6, 1−1e-6]. Fano bits→acc = `1 − h2inv_lower(min(ℓ,1))`.
  No prequential/online, no empirical slope. ✅
- **Matched-capacity probe, no leakage.** `make_pipeline(StandardScaler(), LogisticRegressionCV
  (Cs=1/LAMBDAS, cv=StratifiedKFold(5, random_state=seed), penalty l2, neg_log_loss, refit))`;
  scaler + λ-CV fit **inside** `.fit(Xtr,ytr)` on the 80% split; `predict_proba(Xval)` only.
  **Uniform across all arms** incl. U-1 (100,352-d). Val never enters scaler/CV. ✅
- **Arms present & correct axes.** pooled(span-mean/last/full), C-pos(final/last/full),
  SAV@k(final/all/sparse), C-sparse@k(span-mean/all/sparse), U-1(all 784×128), U-2(best single
  head), oracle@k(full-train selection) + SAV majority-vote. ✅
- **Decision (fail-closed).** `_k_pass_mhc`: ΔL mean>0 & CI-low>0 **and** projected-gain
  mean>0.040 & CI-low>0. `_noharm_hatemm`: ΔL CI-high≥0 **and** Δacc CI-low≥−0.010. `decide`:
  PROCEED iff some k has MHC-pass **and** HateMM-no-harm at that k, else KILL; NO_VERDICT if a
  carrying/no-harm arm is absent. Live run confirmed **both** gates are functional and can
  independently fail (my pure-noise synthetic HateMM correctly tripped no-harm → KILL). ✅
- **Verdict config echo.** `projected_gain_bar=0.040`, `noise_band_acc=0.010`,
  `hatemm_noharm_dacc=-0.010`, `probe_stream="img"`, seeds/k/λ all emitted. ✅

## Checklist 8 — Label / gold discipline

`grep -niE 'test_seen|test\.jsonl|"test"'` across all 4 modules + wrapper + sbatch → **NONE**.
`SPLITS=["train","val"]`; `SPLIT_TO_OUTNAME` maps only train→train, val→dev_seen. No test file
is opened anywhere in the chain. The only "gold" reference is the **oracle** arm, which selects
heads with the **full train labels** (`ytr_all`) — the sanctioned §4 oracle upper-bound
pre-check (train labels only; no test, no annotation fields). No `target_group`/`rationale`/
annotation channel consumed. ✅

## Checklist 9 — sbatch / wrapper

- Resources: `--gres=gpu:a100:1` (1 GPU), `--cpus-per-task=16`, `--mem=96G` — within the
  2 GPU / 16 CPU / 128 GB caps. **No `--time`.** Same `partition=slurmpartition` + `gpu:a100:1`
  as the known-good banked `gen_embed_mllm.sbatch`. ✅
- conda activation is robust: `source …/conda.sh; conda activate HateVideo` under `set -euo
  pipefail` (aborts if activation fails) — **not** the fragile `source activate` (v2 burn). ✅
- HF offline coherent: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; `HOME=/data/jehc223/home`,
  default HF cache `…/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct` **exists**
  (config.json read live). `disk_guard.sh` present/executable, called `|| true` (non-fatal). ✅

## Rev-2b consistency

`NOISE_BAND_ACC=0.010 → PROJECTED_GAIN_BAR=0.040` everywhere (common.py, verdict `config`,
`_k_pass_mhc`); no stale `0.020`/`0.050`/single-seed-`+0.015` in code (grep-confirmed).
`PROBE_STREAM="img"` (img-stream scope); text stream forward-passed + pooled only, for the
guard. ✅

---

## Non-blocking notes (do NOT hold the smoke)

1. **Missing-file behavior differs from the reference (fail-closed).** `load_video_frames`
   raises on a *missing* mp4 vs the reference's zero-vector. Safer; cannot diverge in practice
   (the only banked zero, `hate_video_95`, is present-but-undecodable). If any input mp4 ever
   goes missing, the fresh run hard-fails (correct fail-closed behavior; the guard would
   otherwise flag it as `zero_mismatch`).
2. **Guard exit code under `set -e`.** On a PRIMARY fail the guard `sys.exit(3)` aborts the
   wrapper before gate2's `jq` (so SLURM sees exit 3, not the wrapper's exit 4). Both are hard
   failures; `guard.json` is written first. Cosmetic only.
3. **`decide` gates on `SAV@k` presence but not an explicit `pooled` key** (pooled is the
   implicit baseline inside every `_compare`; it must exist for any per-arm result to compute).
   Adequately fail-closed; a future reader may find the `"pooled(implicit)"` sentinel in
   `required_arms` slightly opaque.

---

## SMOKE PRESCRIPTION (for the smoke executor — a separate authorized step)

**This code is APPROVED to smoke.** The smoke exercises the REAL GPU entry point on a tiny real
subset; it MUST run under SLURM (login=compute reaps non-SLURM GPU work).

**Command** (1-GPU allocation, conda `HateVideo`, repo root, HF offline env):
```
python scripts/analysis/sav_f0_extract.py --datasets HateMM,MHC,MHC_zh --splits train,val --limit 2
```
(Guard/probe are NOT smoked — they assert the full `EXPECTED_COUNTS` by design; they were
exercised offline on synthetic caches running the real functions, re-verified here.)

**Expected artifacts:** 12 per-video caches
`artifacts/sav_f0/extract/<ds>/{train,val}/<id>.pt` (2×2×3) each with keys
`id, label, ok, img_pooled[3584], text_pooled[3584], img_hidden_final[3584],
img_head_final[28,28,128], img_head_spanmean[28,28,128], meta`; and 6
`_manifest.json` with `complete==true`, `n==2`.

**Pass criteria:**
1. stdout prints geometry cross-check OK and `registered 28 o_proj hooks`.
2. each `.pt` loads with the shapes above; `img_pooled` is unit-norm (‖·‖≈1).
3. **PRIMARY-guard preview** — for at least one *decodable* id per dataset,
   `cos(fresh img_pooled, cached img_feats[id]) ≥ 0.999` AND
   `cos(fresh text_pooled, cached text_feats[id]) ≥ 0.999`
   (banked `data/CLIP_Embedding/<ds>/{train,dev_seen}_Qwen2.5-VL-7B-Instruct_HF.pt`).
4. `img_head_final` has no NaN and a non-trivial per-head L2-norm spread.

If (3) passes on the previewed videos, the full chain may be launched with the single-submit
ceremony: `sbatch scripts/slurm/sav_f0.sbatch` (RUN_ID-guarded, runs extract → guard → F-G1
serially). A smoke FAIL on (3) means pipeline drift — do **not** proceed to the full guard.
