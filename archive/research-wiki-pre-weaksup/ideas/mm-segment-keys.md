---
type: idea
node_id: idea:mm-segment-keys
title: "Evidence-matched multimodal segment keys (per-window ASR + CLIP-text) for consensus denoising"
stage: piloted
outcome: attribution-closed
added: 2026-07-05T00:00:00Z
based_on: ["idea:retrieval-consensus-denoising", "paper:wang2024_multihateclip_multilingual_benchmark"]
target_gaps: []
tags: ["hateful-video", "segment-keys", "ASR", "whisper", "consensus-denoising", "probe-before-train", "attribution-closed", "NEGATIVE-main-table", "wave-final"]
---

# Evidence-matched multimodal segment keys (per-window ASR + CLIP-text) for consensus denoising

**stage:** `piloted`  ·  **outcome:** `attribution-closed` (main-table FAIL; annotator-repair
holds; EN lesion pinned to the segment-supervision channel itself)

## Thesis

EN hate in MHClip is speech/text-carried (89.9% of hateful train videos), but the consensus
E-step voted in a visual-only-varying key space with parent-video text shared by all windows —
votes were effectively video-level. Give each K=4 window its OWN Whisper word-timestamp ASR
transcript, CLIP-text-encode it, and fuse with the frame-CLIP channel
((1-w)·cos_img + w·cos_segtext); at w=0.5 memory keys equal the clip-space round-0 keys, making
the change a clean single-variable attribution probe.

## Outcome (2026-07-05, jobs 12302/12303/12310-12317; `EXP_mm_segment_keys.md`)

- **Annotator repair: FULLY CONFIRMED at the probe layer (EN).** all-pruned hateful videos
  56.0%→19%; voting becomes segment-level (wv-std 0.048→0.12); severity anti-correlation
  removed; the catastrophic clip-consensus training result (−0.117 F1) is fully rescued
  (+0.10~0.13 F1 back to ≈ floor).
- **Main-table claim: FAIL (pre-registered).** EN consensus-mm vs 3-seed CLIP floor:
  final-epoch ΔF1 −0.0116±0.0087 (3/3 seeds below floor, one-signed); val-selected +0.0245 is
  single-seed luck (±0.0881). SECONDARY (w0.5/parent) < PRIMARY (w0.7/zero), matching the
  probe's arm ordering.
- **ZH: probe DEAD, not trained** (pre-registered discipline): window-text rate 48.5% +
  weak CLIP-zh text → ASR channel is noise for ZH; ZH keeps clip-space keys (visual/title
  evidence-matched already).
- **Attribution chain CLOSED:** visual keys = video-level noisy votes → archive/blend voting
  spaces don't rescue (W5) → evidence-matched speech keys fix the annotator but downstream
  still ≤ floor ⇒ the EN lesion is the sub-clip supervision channel itself on speech-carried
  hate, not the voting key/space. Consensus claim stays ZH-scoped.

## Surviving value (paper)

Attribution/analysis chapter (three-step chain) + methodology: **probe-before-train** with
pre-registered dual gates (severity correlation, positive-supervision supply) and the ZH
counterexample ⇒ "evidence-matched segment keys" as a diagnostic method, not a headline lever.
Assets: `data/ASR/*/train_asrK4_whisper-large-v3.jsonl` (word-level timestamps; EN 41%
sentence-level degradation — fix DTW/whisperX before reuse),
`src/utils/generate_segment_asr_HF.py`, `src/utils/generate_subclip_mm_embedding_HF.py`,
`scripts/analysis/consensus_probe_mm.py`, probe JSONs in `scripts/analysis/probe_out/`.

**Anti-repeat:** do not re-propose segment supervision on MHClip-EN with yet another key
modality; the channel itself is the lesion. Open follow-ups moved to MORNING_REPORT §8 TODO
(HateClipSeg localization re-scoring with mm keys; word-ts repair).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
- extends → `idea:retrieval-consensus-denoising` (closes its EN attribution)
- tested_by → `EXP_mm_segment_keys.md`
