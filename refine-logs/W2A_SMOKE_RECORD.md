# W2-A Stage-E' SMOKE RECORD (extractor gate 0/1/4 validation)

**Submitted:** job `13166` at `2026-07-14T15:16:23Z` (UTC) — `SMOKE=1 sbatch scripts/slurm/w2a_extract.sbatch`.
**Authorization:** orchestrator "APPROVED FOR SMOKE" (W2A_CODE_REVIEW.md §6 smoke terms; #1 resolution confirmed).
**State at submit:** PENDING (JobHeldUser) → waiting for auto-release, never force.

## --limit decision (reviewer conditional resolved)
Reviewer term §6.4: "if `--limit 1` lands on the single undecodable HateMM row it is zero-guarded and gate 1
is skipped → confirm `grecon_n_checked ≥ 1` per dataset; if 0, re-run with `--limit 3`."
**Verified from the banked caches:** the sole undecodable HateMM row is at **train index 355** (`gt`==cache
order confirmed); every split's index-0 row is decodable (HateMM train[0]=`hate_video_98`,
val[0]=`non_hate_video_58`, test[0]=`hate_video_1`); HateMM val/test and all MHC splits have **0** zero-guard
rows. So `--limit 1` provably lands on decodable rows → `grecon_n_checked = 1` per split (≥1 per dataset)
GUARANTEED. The `--limit 3` fallback is therefore NOT needed; the committed sbatch (`--limit 1`) is used
byte-unchanged (sha `9ed04c14…` valid).

## Frozen hashes the sbatch banner must echo (pre-submit)
| file | sha256 |
|---|---|
| `scripts/analysis/w2a_extract.py` | `2e79599a92d227d9f15366ee17a6644c2f6c77c71f36aa61c76a6274ac9402a9` |
| `scripts/slurm/w2a_extract.sbatch` | `9ed04c14d16799d24e196f1d956698017373e597fd13e0cb2df6919087315153` |
| `src/utils/generate_VideoMLLM_embedding_HF.py` (parity source) | (echoed by the job) |

## No-real-path-write verification (read from the script, not memory)
`scripts/slurm/w2a_extract.sbatch:55-62` — SMOKE path passes `--out_root "$SMOKE_OUT"` with
`SMOKE_OUT=/data/jehc223/RGCL/slurm/logs/w2a_smoke_out_${SLURM_JOB_ID}`. The extractor writes ONLY under
`os.path.join(args.out_root, dataset, grounded_dir, …)` (shards, `<outname>_grounded.pt`,
`<outname>_gatelog.json`); the banked cache is READ-only (`load_banked_imgfeats`). No write path resolves
to `data/CLIP_Embedding/*/grounded_qwen7b_8f/`. Confirmed.

## GREEN criteria (W2A_CODE_REVIEW.md §6) — to verify on terminal
1. config echo + provenance (extractor banner sha + model/max_pixels=151200/bf16/sdpa/transformers ver;
   sbatch 3 sha256s).
2. CPU self-test "[self_test] PASS" before model load.
3. gate 0 — no `[gate0 grid/*]` or `[gate0 contiguity/*]` RuntimeError, both datasets, both forwards.
4. gate 1 — `grecon_n_checked ≥ 1` per dataset with `grecon_cos_min ≥ 0.9999` AND `grecon_maxabs_max ≤ 1e-3`.
5. gate 4 — no `[gate4 len-parity/*]` error.
6. no writes under `data/CLIP_Embedding/*/grounded_qwen7b_8f/` (all artifacts under `w2a_smoke_out_*`).
7. exit 0 (reaching `end=…`).

## TERMINAL RESULT — **GREEN** (all 7 criteria pass)

Job `13166` **COMPLETED** ExitCode `0:0`, elapsed `00:00:43` (host foscsmlprd01; ran start
`2026-07-15T21:15:29Z` → end `2026-07-15T21:16:10Z` UTC). RAW gate lines transcribed from
`slurm/logs/w2a_extract_13166.log` + the six gatelog JSONs:

**Criterion 1 — config echo + 3 sha256s (sbatch banner):**
```
2e79599a92d227d9f15366ee17a6644c2f6c77c71f36aa61c76a6274ac9402a9  scripts/analysis/w2a_extract.py   [== frozen 2e79599a ✓]
9ed04c14d16799d24e196f1d956698017373e597fd13e0cb2df6919087315153  scripts/slurm/w2a_extract.sbatch  [== frozen 9ed04c14 ✓]
d89a912602d763aa055a54f50b0188e302e554b70ff6c0eb872f250bd454b67c  src/utils/generate_VideoMLLM_embedding_HF.py [parity source]
```
extractor banner: `model=Qwen/Qwen2.5-VL-7B-Instruct max_pixels=151200 dtype=bfloat16 attn=sdpa
transformers=4.49.0`; `limit=1 out_root=/data/jehc223/RGCL/slurm/logs/w2a_smoke_out_13166`. **PASS.**

**Criterion 2 — CPU self-test PASS before model load** (both datasets):
`[self_test] PASS — builders, span indexing, pools, gate-0 raises, placebo pairing all OK.` **PASS.**

**Criterion 3 — gate 0** — no `[gate0 grid/*]` / `[gate0 contiguity/*]` RuntimeError, both datasets, both
forwards (job reached `end=` with exit 0). **PASS.**

**Criterion 4 — gate 1 G-recon-IMG** — `grecon_n_checked = 1` for EVERY split (≥1 per split AND per
dataset); all `grecon_cos_min ≥ 0.9999`, all `grecon_maxabs_max = 0.0 ≤ 1e-3`:
```
HateMM/train    : N=1 guard=0 empty=0 grecon_n_checked=1 grecon_cos_min=0.9999998807907104 grecon_maxabs_max=0.0
HateMM/dev_seen : N=1 guard=0 empty=0 grecon_n_checked=1 grecon_cos_min=1.0                grecon_maxabs_max=0.0
HateMM/test_seen: N=1 guard=0 empty=1 grecon_n_checked=1 grecon_cos_min=0.9999998807907104 grecon_maxabs_max=0.0
MHC/train       : N=1 guard=0 empty=0 grecon_n_checked=1 grecon_cos_min=0.9999998807907104 grecon_maxabs_max=0.0
MHC/dev_seen    : N=1 guard=0 empty=0 grecon_n_checked=1 grecon_cos_min=1.000000238418579  grecon_maxabs_max=0.0
MHC/test_seen   : N=1 guard=0 empty=0 grecon_n_checked=1 grecon_cos_min=1.0                grecon_maxabs_max=0.0
```
**PASS** (the img-control forward reproduces banked img_feats to bf16 kernel-drift tolerance).

**Criterion 5 — gate 4 len-parity** — no `[gate4 len-parity/*]` error. **PASS.**

**Criterion 6 — no real-path writes** — `data/CLIP_Embedding/{HateMM,MHC}/grounded_qwen7b_8f/` do **not**
exist; all 18 artifacts (6 `_grounded.pt` + 6 `_gatelog.json` + 6 shards) are under
`slurm/logs/w2a_smoke_out_13166/`. **PASS.**

**Criterion 7 — exit 0** — `end=2026-07-15T21:16:10Z` reached under `set -e`; sacct `COMPLETED 0:0`. **PASS.**

### Observations (non-gating, for the real run)
- HateMM test's single row `hate_video_1` has an **empty transcript** (`empty=1`) — the smoke incidentally
  exercised the `"(none)"` grounded-block branch AND gate 1 still ran on it (grecon 0.99999988); its
  `grounding_present_median=None` (correctly routed to the empty-set diagnostic), `grounding_VOID=False`.
- Single-video `grounding_present_median` (N=1, not a real median): HateMM 0.8759/0.8865, **MHC
  0.9829/0.9739/0.9672**. All `grounding_VOID=False`. MHC single-video grd↔ungrd_vis cos is on the HIGH
  side — worth watching on the REAL run whether the MHC **present-set** median (over 629 videos) approaches
  the 0.999 VOID trigger; it does not gate the smoke.
- Placebo `median=None` everywhere (subset empty under `--limit 1`, per the reviewer's smoke terms). Gate 2/3
  are real-run gates.

**VERDICT: GREEN. Extractor gate 0/1/4 validated on real A100 inputs; the frozen r2 extractor + sbatch are
submission-ready. Full Stage-E' extraction remains a SEPARATE grant (after the C+D re-freeze).**
