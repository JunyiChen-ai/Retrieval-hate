# P8 — MLLM semantic compression of the speech channel

Front: P8 (campaign goal, user-hardlocked: MLLM meaningfully + novelly integrated AND a
SUBSTANTIAL performance improvement). Highest-priority remaining route because it is grounded
in a POSITIVE finding we already own.

**Grounding.** The EN archive's only real effect was **truncation repair**: the transcript-key
recovered ~75% of the archive's F1 gain (`ABLATION_transcript_vs_archive.md`,
`exp-archive-knn-seeds` addenda). 71% of EN hate evidence is speech-borne; the text channel's
input is the raw Title+Transcript **chunk-mean-pooled over 77-token CLIP windows = dilution**.
What was NEVER tested: an MLLM-written **≤60-word evidence-dense summary** as the text-channel
INPUT, single-chunk encoded, with **end-to-end head retraining** (v1 archives were hallucinated
schema JSON; the transcript/archive key-swaps were eval-only, not a retrained text input).

Data check (2026-07-07): MHC-EN text median ~90 tokens (truncated → headroom); HateMM median
694 chars / max 80k (heavily truncated → strongest headroom; 39/744 train empty-text); MHC-ZH
median ~26 tokens (mostly fits → weak truncation headroom, matches ZH=visual/OCR-borne).

---

## PRE-REGISTRATION (locked before probe/training; written 2026-07-07)

### Generation (label-free preprocessing → no leakage)
- Model: Qwen2.5-VL-7B-Instruct, **text-only**, greedy, fixed prompt. ALL videos
  (train/dev/test) of MHC, MHC_zh, HateMM. Source = the `text` field of
  `data/gt/<DS>/{train,val,test}.jsonl` (Title+Transcript). Empty source → empty summary
  (logged).
- Prompt (frozen): *"Condense the following short-video Title+Transcript into at most 60 words,
  preserving WHO is targeted, WHAT is said or shown that could be hateful or offensive, and the
  overall topic. Output only the condensed text, no commentary."*
- Store `data/Summaries/<DS>/<split>.jsonl` = {id, label, orig_text, summary}. This is
  unsupervised input processing (labels never read) — identical leakage status to CLIP feature
  extraction / ASR / the E0b archive.
- CAVEAT (observed in smoke, recorded before results): the MLLM frequently writes the **ZH
  summary in English** (it translates while condensing). This confounds compression with an
  English-pivot on ZH — but it is the model's natural behaviour under a fixed English prompt,
  and the CLIP text tower is English-centric, so an English summary could *help* ZH the way the
  archive's English pivot did. We do NOT force output language (that would be a second, untested
  variable); the ZH probe gate and the ZH result are interpreted with this confound noted.

### Text encoding (single-chunk = the compression)
- `encode_text_single(t)` = truncate to ≤75 content CLIP tokens (+BOS/EOS), ONE CLIPTextModel
  forward, pooler_output → [768]. NOT chunk-mean (that would re-introduce dilution). Any
  truncation is asserted-logged (a ≤60-word summary should rarely exceed 75 tokens).
- Build ALTERNATIVE text_feats caches (do NOT mutate existing floor caches). Each cache copies
  the floor `img_feats`, `ids`, `labels` VERBATIM (asserted `torch.equal` / identical id order)
  and only replaces `text_feats`, so it is a drop-in `--model` swap for run_rac (P3 pattern).

### Conditions (frozen; one test measurement per cell)
- **A floor** = raw chunk-mean (the existing `openai_clip-vit-large-patch14-336_HF` cache).
  Must reproduce the published floor (EN 0.7826/0.7113, ZH 0.8054/0.7706, HateMM ~0.828 acc).
- **B (ours)** = summary replaces raw (single-chunk encode of the MLLM summary).
- **C naive-truncation control (THE RENT TEST)** = single-chunk encode of the **first 70 tokens**
  of the raw Title+Transcript (no MLLM). **B must beat C**, else the gain is just "shorter
  single-chunk text", not MLLM semantics.
- **D (secondary)** = concat `[l2n(raw chunk-mean) | l2n(summary single-chunk)] / √2` (text dim
  1536); gives the head both the diluted-full and the compressed-dense text.

### Probe gate (train-side, BEFORE training)
Per dataset, TRAIN split, no trained head. Leave-one-out kNN vote (cosine, similarity-weighted
"arithmetic", k=20) over `[l2n(img) | l2n(text)]` — the representation the head consumes.
**Gate: B's LOO-kNN accuracy ≥ A's on that dataset.** Report A/B/C for all three; a dataset's
training arm opens only if its gate passes. Diagnostics (not gates): img-only invariant check,
k∈{1,5,10,20}, macro-F1, and per-dataset summary-token-length distribution.

### Training (only for gated-open datasets)
3 seeds {0,1,2} × {A,B,C,D}, standard RAC_video_CLIP recipe (align fusion, triplet+hybrid BCE,
topk=20 arithmetic vote, 30 ep, Faiss CPU), GROUP `RAC_video_p8sum`, FORCE=False (distinct
`--model` tag per condition → no collision). Report BOTH protocols (val-selected warmup≥5 +
final-epoch), acc + macro-F1, paired per-seed deltas.

### Success criteria (pre-registered)
1. A reproduces the floor (bit-for-bit vs the RAC_video_CLIP floor at seed 0).
2. Probe gate passes on ≥1 dataset.
3. **B beats A with mean ΔmacroF1 > 0.01, ≥2/3 seeds positive, under BOTH protocols, on ≥1
   dataset, AND B > C** (the rent test — the gain is MLLM semantics, not just shorter text),
   with no >0.01 harm on a control dataset. The user's bar is a SUBSTANTIAL improvement —
   honest magnitude reported; anything weaker than (3) is within-noise / no claim / honest kill.

---

## PROBE-GATE RESULTS

_(filled after generation; JSON `scripts/analysis/p8_out/probe_gate.json`)_

<!-- GATE_PLACEHOLDER -->

## RESULTS

_(filled after training; JSON `scripts/analysis/p8_out/p8_results.json`)_

<!-- RESULTS_PLACEHOLDER -->

### Jobs / artifacts / repro
- Generation + cache build: `scripts/analysis/p8_generate_summaries.py` +
  `scripts/slurm/p8_generate_summaries.sbatch`. Summaries `data/Summaries/<DS>/`, caches
  `data/CLIP_Embedding/<DS>/{split}_p8{sum,trunc,concat}_..._HF.pt`.
- Probe gate: `scripts/analysis/p8_probe_gate.py` (CPU). Training: `scripts/slurm/train_p8sum.sbatch`.
