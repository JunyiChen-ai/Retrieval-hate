# Pilot: two-sided MIL with cross-video pseudo-negatives (candidate C3)

Date: 2026-08-30. Prior candidate C1 (VLM order distillation) was killed by its
frozen Stage T gate: teacher within-ROC .578 (HateMM) / .514 (MHC-EN), both
below .60 — the VLM cannot order windows inside hate videos better than the
weak-MIL control, so there is nothing to distill. C3 is the next-ranked open
candidate (`../20260830_powa_within_diagnosis/NOVELTY_SCOUT.md`: novelty
verdict open-with-differentiation, differentiation = cross-video kNN negative
selection + explicit high-positive-fraction framing).

## Mechanism hypothesis (frozen before results)

- Diagnosed failure: in high-positive-fraction hateful videos the benign
  seconds (intros/outros, channel branding, music) receive the HIGHEST scores
  (MHC-ZH: benign-second mean rank .818). Top-k MIL never supervises the
  ordering inside a positive bag, and score-based pseudo-negative selection
  (MIST/BN-WVAD style) is circular exactly when ranks are inverted.
- Benign TRAIN videos are fully labelled by the video label (every second is
  benign). Their frames form a benign appearance bank. Seconds inside hateful
  train videos whose features lie very close to that bank (cross-video kNN)
  are, with high probability, benign filler — a supervision signal external
  to the model's own scores, hence not circular under inversion.
- Method: same TemporalConv on cached CLIP+VGGish+BERT features; loss =
  top-k MIL (unchanged) + BCE pushing DOWN the seconds of hateful train
  videos selected as kNN-pseudo-negatives (selection frozen before training,
  computed once from features).

## Design (frozen)

- Pseudo-negative selection: for each second t of each hateful train video,
  cosine distance to its k=5 nearest benign-train-video seconds (features
  L2-normalized per modality then concatenated). Seconds in the closest
  quartile within their own video AND below the global median distance are
  pseudo-negatives, capped at 50% of any video's seconds.
- Loss: L = MIL + lambda * BCE(pseudo-neg seconds -> 0), lambda = 1.0.
- Seeds 234/2025/3407; 30 epochs; identical optimizer/arch to weak control.
- No other change vs the weak-MIL control (single-mechanism rule).

## Gates (frozen)

- Success: test within-ROC macro beats BOTH the weak-MIL control
  (.5777/.4592/.4126/.5234) AND the best reproduced baseline
  (HateMM .6315 MultiHateLoc, EN .6004 CMHKF-align, ZH .5482, HCS .5619 VERA)
  on >=2 of {HateMM, MHC-EN, MHC-ZH}; and the pos_frac>0.6 stratum must not
  degrade vs control on those corpora.
- Attribution control: position-matched RANDOM pseudo-negatives (same count
  per video, uniformly sampled) must NOT reproduce the gain.
- Kill: fails success criterion -> record negative result, return to step 5.
- All numbers: test split, standard evaluator, within-ROC macro primary.
