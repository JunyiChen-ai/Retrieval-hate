# Round 4: transfer-seeded self-training (frozen 2026-08-31, pre-run)

Goal gap: HateMM +.045 vs MultiHateLoc is non-significant (CI [-.002,+.094]);
HCS trails VERA. Feature-side rounds (OCR, CLIP-L) were killed at their
ceiling gates; the remaining headroom is objective-side: HateMM ceiling .7577
vs method .6766.

## Mechanism (single change)

Classic pseudo-label self-training was rejected earlier as circular (the MIL
model's own confident regions inherit its rank inversion). That objection is
void when the SEED model's ordering comes from external span supervision:

1. Seed = the frozen A1 valsel model (aux-span pretrained + rank-preserved
   adaptation, existing recipe, 3-modal CLIP-B stack).
2. On the TARGET's train split (hateful videos), the seed scores every second;
   per video, the top q=20% seconds become pseudo-positive, bottom q=20%
   pseudo-negative (benign train videos: all seconds pseudo-negative).
3. Retrain the TemporalConv from scratch with frame BCE on the pseudo-labels
   plus the video-label MIL term; one self-training round (R=1); depth of
   nothing else changes.
4. Deployed checkpoint: val within-ROC selects between {seed, self-trained}
   (the seed stays in the candidate set, so the step cannot deploy worse than
   A1 up to val noise).

## Scope and arms (5 seeds, TEST eval, dense scores saved)

- Targets: hatemm and hateclipseg (EN/ZH are at their ceilings; untouched).
- Arms: `st` (the method), `st_milseed` (attribution control: identical
  self-training but seeded by the weak-MIL control model — the circularity
  claim predicts this fails/reverts), seed reference = existing valsel rows.

## Gates (frozen)

- HateMM: `st` mean within-ROC >= valsel (.6766) - .005 AND paired bootstrap
  st vs MultiHateLoc fused CI > 0 (the significance target). If the CI still
  straddles 0 but the mean improves, report honestly; the round is judged a
  partial success only if mean >= .69.
- HCS: informational; gate only non-degradation (>= .5450 - .01).
- Attribution: `st_milseed` must NOT match `st` on HateMM (gap >= .02) —
  otherwise the external-seed story is false.
- Kill: HateMM mean < .6766 - .005 -> round dead, negative result recorded.
