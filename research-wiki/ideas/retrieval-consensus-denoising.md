---
type: idea
node_id: idea:retrieval-consensus-denoising
title: "Retrieval-consensus segment denoising (memory washes its own sub-clip labels)"
stage: piloted
outcome: mixed
added: 2026-07-02T21:19:52Z
based_on: ["paper:mei2023_improving_hateful_meme", "paper:yang2025_revealing_temporal_label", "paper:wang2025_hateclipseg_segmentlevel_annotated"]
target_gaps: []
tags: ["hateful-video", "segment-denoising", "retrieval-consensus", "EM", "pseudo-label", "span-free", "MIXED", "language-inconsistent", "iteration-3"]
---

# Retrieval-consensus segment denoising (memory washes its own sub-clip labels)

**stage:** `piloted`  ·  **outcome:** `mixed`

DESIGN_iter3 Method A: sub-clip pseudo-label = agreement(self video label x kNN-neighbor video-label vote), EM rounds, drift demotion; FINAL STATUS (2026-07-05, project close-out): **repair-yes / beat-floor-no / attribution-closed** — ZH claim = de-poisoning repair mechanism only (5-seed replication: beat-floor NOT established), EN refuted with the attribution chain fully closed by the mm-segment-key experiment; analysis-chapter material, not a headline accuracy claim.

## Thesis
Inherited video-level labels are noisy at sub-clip granularity (a hateful video contains benign sub-clips). Let the retrieval memory itself denoise: each sub-clip's pseudo-label = agreement between its own video label and a kNN vote over neighbouring sub-clips' video labels (topk=10, tau=0.2); confident sub-clips train the contrastive embedding, demoted 'drift' sub-clips of positive videos become mined hard negatives; 2 EM rounds re-derive roles in the learned fused space. Span-free (no gold segments). Kill ablation = consensus vs selfscore (MIST/C2FPL-style self-scoring) vs full (inherit labels, Phase-3 repro), gate = both languages >= lambda=0 floor.

## Key risks
STATUS partial (2026-07-03, jobs 12176-12181): ZH VALIDATED — consensus 0.7864 M-F1 / 0.8188 acc wins the kill ablation, repairs the Phase-3 full-mode hole (0.7050/0.7383) and beats the floor (0.7706/0.8054); Phase-3 milmax (0.7875/0.8255) stays numerically top ZH-CLIP overall but destroys EN, so consensus is the best principled/denoising ZH config. EN FAILED HARD — consensus 0.5948/0.7329 vs floor 0.7113/0.7826; gate (both languages same-direction >= floor) NOT passed. Attribution COMPLETE (2026-07-04, W5 jobs 12243-12246): swapping the consensus E-step voting space to the MLLM archive space or a blend fails in BOTH languages (EN archive 0.5663/0.7205, blend 0.6453/0.7143; ZH archive 0.7221/0.7718, blend 0.7232/0.7651 — all below the visual-space consensus and below floor) — the EN failure is NOT a key-space problem rescuable by archive semantics; combined with E1/W2 the chain reads: EN hate is speech-carried → sub-clip supervision itself fails → voting space is not the lesion. Claim is therefore scoped to ZH/visual-carried hate; the "unified mechanism" variant (consensus voting inside the archive memory space) is refuted and must not be re-proposed.

FINAL (2026-07-05, close-out):
1. **ZH multi-seed replication downgrades the win** (`experiments/exp-consensus-zh-seeds.md`,
   jobs 12289-12300): consensus−floor val-selected +0.0115±0.0418 (3/5, p≈0.57), final-epoch
   +0.0247±0.0272 (4/5, p≈0.11) — "beats the floor" NOT established; the surviving claim is
   the REPAIR: the −0.066 inherited-label poisoning never reappears under any seed/criterion,
   and consensus has the best mean of the three arms under both criteria. Paper wording:
   "consensus de-poisons sub-clip supervision (−0.066 → ≈ floor / weakly above)".
2. **EN attribution CLOSED by evidence-matched segment keys** (`idea:mm-segment-keys`,
   `EXP_mm_segment_keys.md`): per-window ASR mm keys fix the annotator completely (supply
   56%→19%, segment-level voting, severity anti-correlation removed, clip-consensus disaster
   rescued) yet training still ≤ floor (final-ep 3/3 seeds) ⇒ the lesion is the sub-clip
   supervision channel itself on speech-carried hate. ZH mm probe dead (pre-registered, not
   trained) — key evidence-matching is language-specific (EN=speech, ZH=visual/title).
   Do not re-propose segment supervision on MHClip-EN with another key modality.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

