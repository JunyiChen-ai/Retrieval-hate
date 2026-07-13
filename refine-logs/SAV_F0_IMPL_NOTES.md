# SAV (C2) F-G0/F-G1 — implementation self-audit notes

**Authority:** `research-wiki/experiments/exp-sav-f0.md` (Rev-2a, APPROVED) + the three
non-blocking execution notes in the Rev-2 delta-check of `refine-logs/SAV_F0_PREREG_REVIEW.md`.
**Scope built:** F-G0 extraction + F-G0(b) two-tier reproduction guard + F-G1 statistics engine
(+ wrapper + sbatch). **NOT built / NOT run:** no F-G2/F-G3 (later gates), **no `sbatch` of any
kind** (ceremony requires independent code review + smoke + authorization first).
**Env:** conda `HateVideo` (Python 3.11.8). **Date:** 2026-07-13.

## 0. Files delivered (one line each)

| file | purpose |
|---|---|
| `scripts/analysis/sav_f0_common.py` | shared constants (geometry, frozen extraction pins, thresholds, seeds), atomic in-repo IO, Fano `h2inv`/`fano_acc`, holdout-log-loss bits, nearest-centroid head selection, example-level clustered bootstrap (mean + Fano-projection), per-video cache loaders. No sklearn (keeps the GPU extract import graph light). |
| `scripts/analysis/sav_f0_extract.py` | F-G0 extraction: frozen Qwen2.5-VL-7B forward over train+val of HateMM/MHC/MHC_zh; mirrors the banked extractor's decord→PyAV 8-frame sampler + message/span pooling VERBATIM; forward-pre-hook on the 28 text-decoder `self_attn.o_proj` inputs → per-(layer,head) **final-token AND span-mean** vectors for all 784 heads (img stream) + final-token full-hidden (C-pos) + pooled img/text read-outs (guard). Deterministic, resumable per-video, atomic writes, fail-closed on missing input, symlink `followed_target` audit. |
| `scripts/analysis/sav_f0_guard.py` | F-G0(b) two-tier reproduction guard: PRIMARY per-video cosine ≥ 0.999 (img+text, all train+val) fresh-vs-cached; SECONDARY ±0.010 val-acc confirmatory probe; PRIMARY decides, PRIMARY-fail = hard fail; zero-guard handling. |
| `scripts/analysis/sav_f0_probe.py` | F-G1 statistics engine: SAV nearest-centroid head selection (5 seeds × 20/class draws), stratified-80% probe splits; arms pooled/SAV/C-pos/C-sparse/U-1(100,352-d, L2 λ by 5-fold CV)/U-2 + SAV majority-vote + oracle; primary MDL holdout-log-loss + co-primary accuracy; example-level clustered bootstrap (10k) for ΔL/Δacc/projected-gain; Fano bits→acc; fail-closed machine-readable JSON verdict. Also exposes `fit_logreg_probe` used by the guard. |
| `scripts/wrappers/sav_f0.sh` | fail-closed jq-gated chain: extract → (jq manifests complete) → guard → (jq `.pass==true`) → probe → (jq `.status=="COMPLETE"`). RUN_ID guard; in-repo temp only. |
| `scripts/slurm/sav_f0.sbatch` | 1×A100 + 16 CPU + 96G, no `--time`, conda HateVideo, HF offline; invokes the wrapper with `RUN_ID=SAV-F0-FG0-FG1`. |

Prereg edits (the ONLY two, per execution notes 1 & 3; recorded in §8 as "Rev-2a execution-notes
pinning"): F-G2 pre-flight carry-forward head-set pin (deterministic full-train nearest-centroid =
`oracle_order`, seed draws are stability-diagnostic only) and the §6 stale-"before C1" fix
(C1 = `KILL_CONFIRMED`). Delta-check note 2 (C-sparse span-mean) is realised directly in the
extractor (`img_head_spanmean`), no prereg text change needed.

---

## (a) Runtime cross-check STATIC simulation table

Every assert / drift-guard in the code, simulated against the FROZEN spec values and the REAL
data counts (counted live 2026-07-13; provenance in the last column). "Result" = what the check
does when the frozen pipeline is intact (all must be inert/PASS) and what trips it.

| # | check (file:symbol) | quantity | frozen/expected value | static-sim result | provenance |
|---|---|---|---|---|---|
| 1 | `common:REPO_ROOT` assert | `src/utils/generate_VideoMLLM_embedding_HF.py` exists under repo root | present | PASS (root resolves to `/data/jehc223/RGCL`) | repo tree |
| 2 | `extract.main` geometry | `(num_hidden_layers, num_attention_heads, hidden_size)` | `(28, 28, 3584)` | PASS; trips on any model swap | `config.json` (read live) |
| 3 | `extract.register_head_hooks` count | # modules `name.endswith('self_attn.o_proj')` | `28` | PASS = 28 (vision `.proj`/`.attn.proj` excluded) — **verified offline on a fake Qwen tree: 28 hooked, 5 vision proj rejected** | `modeling_qwen2_5_vl.py:735` (o_proj), `:178/:240/:289` (vision `.proj`) |
| 4 | `extract.register_head_hooks` dim | each `o_proj.in_features` | `3584` (=28×128) | PASS; trips on head-dim drift | `modeling_qwen2_5_vl.py:735` |
| 5 | `extract.HeadCapture` hook shape | o_proj input `(dim, batch, last)` | `(3, 1, 3584)` | PASS; trips on batched/odd input | code |
| 6 | `extract.forward_once` len | `last_hidden.shape[0] == input_ids.numel()` | equal (vision tokens masked_scatter in place) | PASS — same invariant the banked extractor asserts | `generate_VideoMLLM_embedding_HF.py:283` |
| 7 | `extract.forward_once` capture | `len(capture.buf) == 28` and per-layer `seq == last_hidden seq` | 28; equal | PASS; **reshape verified EXACT offline** (final-token + span-mean vs manual) | code + offline test |
| 8 | `extract.process_split` count (HateMM/train) | `len(read_gt)` | **744** | PASS | `data/gt/HateMM/train.jsonl` = 744 |
| 9 | same (HateMM/val) | | **107** | PASS | `.../val.jsonl` = 107 |
| 10 | same (MHC/train, val) | | **549 / 80** | PASS | `data/gt/MHC/{train,val}.jsonl` |
| 11 | same (MHC_zh/train, val) | | **579 / 78** | PASS | `data/gt/MHC_zh/{train,val}.jsonl` |
| 12 | `common.load_extracted_split` count | per-split `len==EXPECTED_COUNTS` | 744/107, 549/80, 579/78 | PASS (same table as 8–11) | live counts |
| 13 | `common.load_extracted_split` id | `obj['id']==gt id` (per video) | equal | PASS; trips on cache/gt misorder | code |
| 14 | `common.load_cached_pooled` shape | `len(ids)==img.rows==txt.rows` | 744/107/549/80/579/78 | PASS | banked `*_Qwen2.5-VL-7B-Instruct_HF.pt` (loaded live: shapes match) |
| 15 | `guard` PRIMARY | `min per-video cosine (img,text) ≥ 0.999` | 0.999 | inert when fresh≈cached; **offline: clean→1.000000 PASS, 1-video drift→−0.026 FAIL** | spec §4 F-G0(b) / R2 |
| 16 | `guard` zero-guard | cached vs fresh both zero-norm (≤1e-6) | matched (excluded from min) | HateMM/train **`hate_video_95`** (symlink present, decode-fails → cached zero) reproduces as zero → `zero_matched`; **offline: zero-in-one-only→FAIL (`zero_mismatch=1`)** | banked cache (norm 0 at `hate_video_95`), live |
| 17 | `guard` SECONDARY | `|Δacc_fresh−Δacc_cached| ≤ 0.010` | 0.010 (confirmatory only) | never blocks a PRIMARY pass (flip-quantised: 1 flip=0.0125>0.010) | spec R2 |
| 18 | `probe._selection_draw` | per-class train count `≥ 20` | ≥20 | PASS (min class = MHC train hate 168) | live class counts: HateMM 298/446, MHC 168/381, MHC_zh 180/399 |
| 19 | `probe.fit_logreg_probe` U-1 dim | `head_final.reshape` cols | `784×128 = 100352` | PASS; L2 λ by 5-fold CV in-train | spec Rec-1 |
| 20 | `probe.decide` fail-closed | all `SAV@k` present for MHC + HateMM | present | verdict `NO_VERDICT_MISSING_ARM` if any absent | spec §3 |
| 21 | `probe` bar constants | `PROJECTED_GAIN_BAR`, `HATEMM_NOHARM_DACC` | `0.030+0.010=0.040`, `−0.010` | printed in verdict `config`; NOISE_BAND=0.010 pinned by Rev-2b main-loop ruling (A-line G0-cond precedent, `refine-logs/lb_scgp_global/M1_G0COND_PROBE_RECORD.md` +0.030+0.01=+0.040) | `REFLECTION:41`, spec F-G1/R3, exp-sav-f0.md §8 Rev-2b |

**Unresolvable rows: none.** Rows 2–7, 15–16 additionally carry live offline evidence (fake-Qwen
hook filter, per-head reshape, guard drift/zero scenarios); rows 8–14, 18 carry live data-count
evidence. Rows requiring the real GPU forward to fully close (6, 15 img/text cosine on real
features) are covered by the reviewer smoke (§e).

---

## (b) Import audit (incl. deferred / function-level), with `python -c` evidence

All third-party modules resolve in `HateVideo` (evidenced live 2026-07-13):

```
numpy 1.26.4 | torch 2.6.0+cu124 | transformers 4.49.0 | PIL 11.1.0
decord 0.6.0 | av 17.0.0 | sklearn 1.5.2
```

**Top-level imports** (loaded at module import; no conditional path-switching):
- `sav_f0_common`: `numpy`, `torch` (+ stdlib `hashlib/json/os/tempfile/pathlib`). **No scipy** —
  `h2inv` is a self-contained numpy bisection, so the stats engine adds no SciPy dependency.
- `sav_f0_extract`: `numpy`, `torch`, `PIL.Image`, `transformers.{AutoProcessor,
  Qwen2_5_VLForConditionalGeneration}`.
- `sav_f0_guard`: `numpy`; local `sav_f0_common`, `sav_f0_probe.fit_logreg_probe`.
- `sav_f0_probe`: `numpy`, `sklearn.{linear_model.LogisticRegressionCV,
  model_selection.StratifiedKFold, pipeline.make_pipeline, preprocessing.StandardScaler}`.

**Deferred (function-level) imports — audited, INTENTIONAL, path-neutral:**
- `sav_f0_extract.py:56-57` `import decord; from decord import VideoReader, cpu` inside
  `_decode_with_decord`; `:71` `import av` inside `_decode_with_pyav`. These are copied VERBATIM
  from the frozen extractor (`generate_VideoMLLM_embedding_HF.py:156-157,171`), where the lazy
  import exists so decord/av load only when a video is actually decoded and the PyAV fallback path
  is only imported if decord fails. **Neither import changes the model forward path** — they
  produce PIL RGB frames that are fed to the processor identically to the banked pipeline. This is
  the only deferred-import site; it is the deliberate mirror of the frozen loader, not a hidden
  path-switch. (The AST scan's `col_offset` heuristic mislabels bare `import decord/av` as
  top-level; their true placement is function-body, confirmed by grep above.)
- Local sibling imports (`import sav_f0_common`, `from sav_f0_probe import fit_logreg_probe`) sit
  after a `sys.path.insert(0, dirname(__file__))` (E402) so the three entry points run both as
  `python scripts/analysis/sav_f0_*.py` and from the repo root. No third-party effect.

**No** `try/except ImportError` fallbacks, **no** environment-conditional imports, **no**
monkeypatching of transformers internals. The o_proj capture is a standard
`register_forward_pre_hook` (added/removed around the run; removed in a `finally`).

---

## (c) File-handoff / env-var table (every path the chain reads/writes)

**Reads (inputs, read-only):**

| path | by | note |
|---|---|---|
| `data/gt/<ds>/{train,val}.jsonl` | extract, common (loaders) | ground truth; TEST never read at F-G0/F-G1 |
| `data/video/<ds>/All/<id>.mp4` | extract | **symlinks** into raw stores; frame source pin (M2b) — NOT `data/lora_frames/` |
| `data/CLIP_Embedding/<ds>/{train,dev_seen}_Qwen2.5-VL-7B-Instruct_HF.pt` | guard, probe(guard sec.) | banked enc3s pooled features (fresh-vs-cached target) |
| Qwen2.5-VL-7B weights (HF cache `~/.cache/huggingface/hub/...`) | extract | offline (`HF_HUB_OFFLINE=1`) |

**Writes (all IN-REPO; atomic; resumable):**

| path | by | handoff / gate |
|---|---|---|
| `artifacts/sav_f0/extract/<ds>/<split>/<id>.pt` | extract | per-video cache (skip-if-exists resume) |
| `artifacts/sav_f0/extract/<ds>/<split>/_manifest.json` | extract | wrapper gate 1: `jq -e '.complete==true and .n==.n_expected'` |
| `artifacts/sav_f0/guard/<ds>/guard.json` | guard | wrapper gate 2: `jq -e '.pass==true'` (PRIMARY) |
| `artifacts/sav_f0/probe/verdict.json` | probe | wrapper gate 3: `jq -e '.status=="COMPLETE"'`; carries the F-G1 verdict |
| `slurm/tmp/` (mkdir) | wrapper | in-repo scratch placeholder (no temp actually written by our code) |
| `slurm/logs/sav_f0_%j.{out,err}` | sbatch | job logs |

**Env vars (set by the sbatch, read by nothing app-specific except HF/W&B):**
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `WANDB_MODE=disabled`, `PYTHONUNBUFFERED=1`,
`TOKENIZERS_PARALLELISM=false`; wrapper contract vars `RUN_ID`/`EXPECTED`/`DATASETS`/`LIMIT`.
**`$TMPDIR` is never used** — atomic writes `tempfile.mkstemp(dir=<target dir>)` + `os.replace`
keep the temp on the SAME in-repo filesystem as the final artifact (realbank `$TMPDIR`
out-of-repo burn lesson). Atomic tmp files are same-dir siblings, auto-removed on success/failure.

---

## (d) Storage-topology / symlink handling at every input

- **Frame source (M2b pin):** `data/video/<ds>/All/<id>.mp4` are **symlinks** into external raw
  stores (`data/video/HateMM/All/hate_video_100.mp4 -> /data/jehc223/HateMM/video/...`,
  `data/video/MHC/All/*.mp4 -> /data/jehc223/Multihateclip/English/video_mp4/...`,
  `MHC_zh -> /data/jehc223/Multihateclip/Chinese/video/...`, all confirmed live). The extractor
  decodes these mp4s via the SAME decord→PyAV 8-frame sampler as the banked extractor (NOT the
  pre-extracted `data/lora_frames/`). Per video it records `is_symlink`, `followed_target`
  (`os.path.realpath`), and `followed_target_in_repo` into the split manifest's `symlink_audit`
  (+ sha256). **External `followed_target` is EXPECTED and allowed** (the raw stores live outside
  the repo by design); the audit records it, it does not gate. A **missing** mp4 is a hard
  `FileNotFoundError` (fail-closed) — a decode-failing but present mp4 (e.g. HateMM/train
  `hate_video_95`) is reproduced as a zero-guard cache, exactly matching the banked cache's zero
  vector, and the guard classes it `zero_matched`.
- **Output topology:** all artifacts under `artifacts/sav_f0/` inside the repo (290G quota),
  atomic same-dir publish. No symlinks are created; no output escapes the repo.
- **A frame-source swap is doubly guarded:** it would (i) diverge from the banked pipeline and
  (ii) trip the F-G0(b) PRIMARY cosine guard (fresh pooled ≠ cached pooled).

---

## (e) Smoke plan (reviewer-run; exercises the REAL frozen entry point on a REAL subset)

The GPU forward MUST run under SLURM (project rule: non-SLURM compute on the login/compute node is
reaped), so the smoke is a **reviewer-authorized 1-GPU SLURM step**, NOT run by the implementation
agent. It drives the **real** `sav_f0_extract.py` entry point (never a re-implementation) on a
tiny REAL subset:

```
# in an interactive/one-off SLURM allocation, conda activate HateVideo, from repo root:
python scripts/analysis/sav_f0_extract.py --datasets HateMM,MHC,MHC_zh --splits train,val --limit 2
```

- Exercises: real Qwen2.5-VL load + geometry cross-check (28,28,3584); the 28 o_proj hooks;
  the real decord→PyAV decode of 2 real mp4s/split/dataset (6 videos × 3 datasets = 12 real
  forwards, img+text); per-head reshape; atomic per-video cache write; manifest (`complete`,
  `symlink_audit`).
- **Expected artifacts:** `artifacts/sav_f0/extract/<ds>/{train,val}/<id>.pt` (each with keys
  `img_pooled[3584]`, `text_pooled[3584]`, `img_hidden_final[3584]`,
  `img_head_final[28,28,128]`, `img_head_spanmean[28,28,128]`, `label`, `ok`, `meta`) and
  `_manifest.json` with `complete==true`, `n==2`.
- **Guard/probe are NOT part of the smoke** — they assert the FULL `EXPECTED_COUNTS` and so
  require the full extraction; a 2-video smoke would (correctly) trip their count asserts. The
  guard/probe code paths were instead validated OFFLINE end-to-end on synthetic caches that run
  the real functions (`run_dataset` over all arms incl. U-1 100,352-d; `guard_dataset` clean /
  drift / zero-mismatch), reported above.
- **Post-smoke sanity for the reviewer:** on 2 real videos load one `<id>.pt` and confirm
  (i) `img_pooled` is unit-norm, (ii) cosine(`img_pooled`, banked cached img_feats for that id)
  ≈ 1.0 (this is the PRIMARY-guard statistic previewed on 12 videos before the full guard), and
  (iii) `img_head_final` has no NaNs and non-trivial per-head norm spread.

An offline pre-check the reviewer can run WITHOUT a GPU (fast, ran clean here): the fake-Qwen hook
filter + per-head reshape test and the synthetic `run_dataset`/`guard_dataset` drivers (see the
transcript in this task); these exercise every non-GPU code path.

---

## VRAM / runtime / storage estimate (with basis)

- **VRAM:** one frozen Qwen2.5-VL-7B bf16 forward (8 frames, `max_pixels=360*420=151200`, sdpa)
  fits **one A100-80G** — the banked enc3s extraction ran exactly this on `gpu:a100:1`
  (`scripts/slurm/gen_embed_mllm.sbatch`), and the review independently cites the P9 LoRA forward
  fitting 80G (`exp-e2eq-e0.md:222-227`); a frozen forward is strictly lighter. The forward-pre-hook
  adds ~0 model VRAM (captures already-computed `[seq,3584]` tensors); transient per-video capture
  of 28 layers × `[seq,3584]` bf16 (seq≈1.5-3k) ≈ 0.3-1.0 GB, freed each video. **Comfortably < 80G.**
- **Runtime (extraction):** 2 frozen forwards/video (img+text) × (744+107+549+80+579+78)=**2137**
  train+val videos. Banked single-stream extraction was "tens of min/dataset"; two forwards/video
  ⇒ ≈ **1-1.5 h** total across the 3 datasets. Fully resumable (skip-if-exists), so restarts are cheap.
- **Runtime (F-G1 probe, CPU):** dominated by U-1 (100,352-d `LogisticRegressionCV`, 5-fold × 7 λ,
  `n_jobs=-1` on 16 CPU). Basis: at N=48 a full U-1 seed took ≈7-8 s in the offline synthetic run;
  at N≈475 (80% of ~590 train) expect ≈1-2 min/U-1-fit ⇒ 15 fits (5 seeds × 3 datasets) ≈ **15-30 min**;
  the ≤5,120-d SAV/C-sparse/oracle and 3,584-d pooled/C-pos arms are seconds each. **Probe ≈ 30-45 min.**
- **Guard:** cosine over 2137×2 videos + 3 small confirmatory probe fits = **seconds-to-minutes.**
- **Total wall (single serial sbatch):** ≈ **1.5-2.5 h** — within the spec's ~2-3 GPU-hr F-G1
  budget line, no `--time`.
- **Storage:** per-video img cache ≈ `img_head_final`(401 KB) + `img_head_spanmean`(401 KB) +
  pooled/hidden(≈43 KB) ≈ **845 KB/video × 2137 ≈ 1.8 GB** fp32 (matches execution-note-2 "~2.4 GB"
  order with `.pt` overhead). Trivial vs the 290G quota.

---

## Open questions — RESOLVED by main-loop rulings 2026-07-13 (exp-sav-f0.md §8 Rev-2b)

1. **`NOISE_BAND_ACC` — RULED: 0.010 (⇒ `PROJECTED_GAIN_BAR = 0.040`). APPLIED.** Originally pinned
   provisionally at the upper end (0.020 ⇒ 0.050) and flagged. Main-loop ruling: **0.010**, for
   protocol consistency with the A-line G0-cond probe precedent
   (`refine-logs/lb_scgp_global/M1_G0COND_PROBE_RECORD.md` used +0.030 + 0.01 = +0.040); a per-gate
   drifting bar invites protocol-inconsistency criticism. Applied in `sav_f0_common.py`
   (`NOISE_BAND_ACC = 0.010`, `PROJECTED_GAIN_BAR = 0.040`), echoed in the F-G1 verdict JSON `config`
   (values flow from the constants — no other code carries the number), simulation-table row 21
   above, and the prereg F-G1 bar paragraph + §8 Rev-2b entry.
2. **Probe stream = IMG only — RULED: ACCEPTED for F-G1. APPLIED.** Scope now pre-declared in
   exp-sav-f0.md §4 F-G1 ("Probe-stream scope PINNED" paragraph): img-stream per-head only; text
   stream remains pooled-only for the F-G0(b) guard (which checks BOTH cached streams);
   **text-stream / concat per-head extraction DEFERRED as an F-G2-stage option only if SAV wins
   F-G1** — pre-declared so it cannot become a post-hoc degree of freedom. No code change needed
   (`PROBE_STREAM="img"` was already the implementation; comment updated to cite the ruling).
3. **Probe preprocessing (`StandardScaler` + L2 `LogisticRegressionCV`, uniform across arms) —
   RULED: accepted as-is.** No change.
4. **U-1 CPU pole — RULED: accepted as-is.** No change.

No commits made. No `sbatch` submitted.
