# Wave 1 handoff — state at 2026-08-19 23:31

Written so the run does not depend on any live session or watcher. Everything
below is detached (`ppid=1`), idempotent, and guarded.

## Landed

**LaGoVAD — complete, four datasets.** `REPRO_CAMPAIGN_RESULTS.md` §M.
Test-split frame ROC-AUC 0.5579 / 0.5239 / 0.5965 / 0.5000 (HateMM / MHC-EN /
MHC-ZH / HateClipSeg) against a 0.500 random floor. Artifacts verified:
**3081 / 3084 curves**, the three missing being exactly the two audio-only HateMM
containers (`hate_video_147`, `hate_video_292`, freeze D2) and the truncated
`yt_NzvfkIYS5Yg`. Test pools full at 215 / 161 / 149 / 118.

Headline finding: the **text-free binary head is the strongest row** on MHC-EN
(0.6058) and HateClipSeg (0.5431), ahead of all ten hate-definition queries. What
transfers is the checkpoint's generic surveillance-anomaly prior, not the written
definition. Definition paraphrase swings MHC-ZH by 0.18 ROC — more than the gap
between methods.

## In flight

| method | state | owner |
|---|---|---|
| LAVAD | holds the GPU; summarize 570/637 at 2.98 gen/s, `oom=0`, `trunc=0`; scoring stage still to come | LAVAD/URF worker |
| URF | queued behind LAVAD; `max_tokens` restored to 16384; block-wise mask-free SDPA at 219846c | LAVAD/URF worker |
| UniTime | parked on the reservation (pid 1602811); resumes from its own `done_ids` | this worker |
| AV²A | parked (pid 3693884) + eval watcher (3647985); ~7–11 h once it starts | AV²A worker |

`logging/runs/GPU.reservation` gates the queue; only `blip2_caption`, `lavad_llm`,
`lavad`, `urf`, `urf_caption`, `urf_llm` may take the card. Delete it when
LAVAD and URF are both done, and UniTime and AV²A start themselves.

**Roughly 25 GPU-hours remain on one card**, so §J / §K / §L cannot land in a
single night. That is the binding constraint, not the code.

## What to distrust

Curve counts `lavad=5` and `unitime=44` are **stale artifacts**, not progress:
the 5 are from the truncated chain that exited rc=0, the 44 from a converter
test on banked records. Real completion shows as >100 curves per dataset.

## Guards now in place (each verified to pass on real data AND fail on a truncation)

- `run_lagovad_chain.sh` — `set -euo pipefail` + per-dataset curve-count guard.
- `unitime_to_curves.py` — refuses exit 0 when a raw file holds records but
  yields under half as many curves.
- `run_av2a_supervised.sh` — aborts if the smoke fails; `RUN COMPLETE` gated
  behind a curve-count guard (allows only the 17 HateMM + 1 HateClipSeg drops).
- `discrimination_check.py` — `curve_varies`, `embeddings_discriminate`,
  `scores_separate_items`, `patch_applied`.

## Defects found this session

1. **7 corrupt HF checkpoints** — right byte count, wrong content. CLIP returned
   one constant embedding for every frame; recorded for a week as a property of
   LaGoVAD's binary head. Repaired + sha256-verified (`audit_hf_cache.sh`,
   `hf_refetch.py`).
2. **Crash marker vs. SIGTERM** — 12 of the 14 videos the Wave 0 Qwen row
   excludes decode cleanly. Freeze deviation **D3**. `run_qwen_grounding.py`
   still needs the two-strikes fix before it is next run, and §I's prose about
   those 14 files is still wrong.
3. **decord fails on 25.5% of MHC-EN** — `decord_fallback.py` (PyAV fallback).
4. **Runners exiting 0 on empty output** — the LAVAD chain wrote 5 curves and
   reported success. Guards above.

## Retracted

I reported that the flash → sdpa adaptation changed VideoLLaMA3's attention
semantics. It does not: the class that runs is `VisionSdpaAttention`, which
*passes* the bool mask to SDPA rather than adding it (verified block-diagonal to
0.000e+00; the URF worker measured 4.44e-16 against flash). The +1.0 bias exists
only in the eager class, which nobody instantiates. Full retraction in §M.6.

## Open decision for the campaign owner

Whether to re-run the 12 healthy Wave 0 Qwen videos and amend §I's counts and
prose. It is a completion of a run that never evaluated them rather than a second
test call, but it touches freeze red line 4, so it is not mine to settle.
