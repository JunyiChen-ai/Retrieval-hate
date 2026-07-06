# EXP_p8b — vision-grounded evidence summaries as the text channel (P8 extra arm)

**Status:** PRE-REGISTERED (gate frozen before probe/training) · **Started:** 2026-07-07 ·
**Owner:** subagent P8b · **Parent:** `EXP_p8_semantic_compression.md`

**Why.** P8's text-only summary arm gated **CLOSED on ZH** (train probe: B 0.727 < A floor 0.738,
while naive-truncation C hit 0.791) — expected, because ZH transcripts barely truncate (median ~26
tokens) and W2 found **ZH hate evidence is visual / on-screen-text**, which a text-only summary of the
Title+Transcript cannot see. P8b tests the fix: a **vision-grounded** summary — Qwen2.5-VL reads
8 sampled frames + Title+Transcript and writes the same ≤60-word evidence-dense summary, explicitly
instructed to **transcribe on-screen text verbatim** (captions/memes/overlays the channel CLIP image
encoder cannot read, especially Chinese). Same schema/harness as P8 → drop-in extra arm.

## Pre-registration (frozen)

### Generation (label-free, no leakage)
- `scripts/analysis/generate_vision_summary.py` + `scripts/slurm/generate_vision_summary.sbatch`.
  Qwen2.5-VL-7B-Instruct, greedy, 8 uniform frames (same sampler as segment scoring) + Title+Transcript.
  Prompt = P8's exact ≤60-word condensation wording + the single frozen addition: *"Transcribe any
  visible ON-SCREEN TEXT (captions, memes, overlaid words, signs) verbatim into the summary if it
  could be hateful or offensive."* So B_vision vs B_text isolates the vision grounding.
- Output `data/Summaries_vision/<DS>/<split>.jsonl` = {id, label, orig_text, summary} — byte-identical
  schema to P8's `data/Summaries/`, but a **parallel dir** (never clobbers the text-only arm).
- **MHC_zh primary** (the dataset P8 closed on); MHC/HateMM optional secondary.

### Encoding + conditions (reuse P8's harness EXACTLY, apples-to-apples)
- `scripts/analysis/p8b_build_cache.py` imports P8's `encode_single` (≤75 content tokens, one
  CLIPTextModel forward, pooler) and writes `<split>_p8vsum_HF.pt` (copies floor img/ids/labels
  VERBATIM, asserted `torch.equal`; only text_feats replaced). Drop-in `--model p8vsum_HF` swap.
- Conditions carried to test if the gate opens: **A** floor (raw chunk-mean), **B_vision**
  (p8vsum), **C** naive first-70-token truncation (the rent test), **D** optional concat.

### Probe gate (train-side, BEFORE any training) — HIGHER bar than P8-EN
`scripts/analysis/p8b_probe.py` reuses P8's `rep()` + `loo_knn()` (LOO kNN over
`[l2n(img)|l2n(text)]` @k20 on ZH TRAIN). **Gate: B_vision must beat BOTH**
- **A** (floor, 0.738) **AND**
- **C** (naive truncation, **0.791**) — the number to beat, per team-lead.

The C bar is the honest one: on ZH, shorter single-chunk text alone (no MLLM) already scores 0.791,
so B_vision only earns a role if the *vision-grounded semantics* (on-screen-text transcription) push
past that. If B_vision ≤ C → **kill, no training** (vision grounding adds nothing over shorter text).

### Training (only if gate opens)
3 seeds {0,1,2} × {A, B_vision, C}, standard RAC_video_CLIP recipe, GROUP `RAC_video_p8vsum`,
FORCE=False, distinct `--model` tags. Both protocols (val-selected + final-epoch), acc + macro-F1,
paired per-seed deltas vs A. Success (frozen, same as P8): B_vision > A by >0.01 macro-F1, ≥2/3 seeds,
BOTH protocols, **AND B_vision > C** (rent test), no >0.01 harm. Anything weaker = within-noise/no claim.

### Ops
Generation GPU job = `12427` (MHC_zh all splits), queued behind P2c 72B judges + P8 EN training.
Everything downstream (cache build, probe) is CPU. FORCE=False; no .pt in git; foreground sacct polling.

## Results
_(pending: generation 12427 → build p8vsum cache → probe → §gate verdict → training if open)_
