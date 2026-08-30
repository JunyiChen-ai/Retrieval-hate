# Round 3: visual encoder upgrade (CLIP-L/14-336) for LOCO-ST

Date: 2026-08-31. Prior rounds this iteration: OCR channel killed (O1 .5827 vs
.5809 on HCS; O1b .7491 vs .7577 on HateMM — no ceiling lift either corpus).
Remaining goal gaps: HCS loses to VERA (.5450 vs .5619; supervised ceiling with
CLIP-B/16 stack only ~.58-.60) and HateMM significance (+.045 ns; ceiling
.7577). Hypothesis: the visual channel (CLIP-B/16, 512-d) is the weakest
feature; CLIP-L/14-336 (768-d, cached locally) lifts per-corpus ceilings, which
the span-transfer method then converts into test gains.

## Stages and frozen gates

- **E1**: extract 1-fps JPEG frames for every video of all four corpora
  (ffmpeg, CPU; ~60% of videos lack cached frames).
- **E2**: CLIP-L/14-336 image embeddings at 1 fps ->
  results/reproduction/features/clip_l14_336_1fps/ (768-d, PROVENANCE recorded).
- **E3 ceiling probes (kill gates, 3 seeds, train-span skyline, TEST eval)**,
  feature stack = clip_l14 + vggish + bert:
  - HCS gate: within-ROC >= .62 (CLIP-B stack: ~.581).
  - HateMM gate: within-ROC >= .77 (CLIP-B stack: .7577).
  - One passing corpus is enough to proceed to E4 for the corpora that pass;
    both failing kills the round.
- **E4**: LOCO-ST A1 valsel + loo_zero on the upgraded stack, 5 seeds, dense
  scores, all four corpora (feature stack is a global method choice — applied
  everywhere, not per-corpus). Gates:
  (i) HCS: mean within-ROC > .5619 AND paired bootstrap vs VERA CI > 0;
  (ii) HateMM: paired bootstrap vs MultiHateLoc CI > 0;
  (iii) EN/ZH must not drop > .015 vs the CLIP-B numbers (.7405/.6310);
  (iv) baselines are NOT rerun on new features (they use their published
  feature configs; our feature stack is part of our method, disclosed).
- Protocol unchanged: test reporting, within-ROC macro primary, shared
  evaluator, seeds 234/2025/3407/42/20260830.

## Scope amendment (2026-08-31, pre-E2, frozen): MHC raw videos unavailable

E1 found 630 (EN) + 663 (ZH) videos with no raw source on this machine (only
their CLIP-B-era features survive). CLIP-L features therefore cannot cover MHC
train splits. Round rescoped:
- Targets: hatemm and hateclipseg only, aux source = the other one (hcs->hatemm
  and hatemm->hcs were each target's strongest/competitive single source in the
  B/16 ablations). EN/ZH method rows keep their 3-modal CLIP-B numbers.
- E3 gates unchanged for these two corpora. E4 gates: (i) HCS beat VERA with
  CI>0; (ii) HateMM bootstrap vs MultiHateLoc CI>0; (iii) same protocol.
- Disclosure: the deployed feature stack differs by corpus (L14 where raw video
  exists, B/16 elsewhere) — stated in every table.

## Round-3 outcome (2026-08-31): KILLED at E3

E3 ceilings with CLIP-L/14-336 stack: hatemm .7605±.020 (gate .77; B/16 ref
.7577 — +.003), hcs .6008±.012 (gate .62; B/16 ref ~.581 — +.02, insufficient).
Both gates fail; E4 not run. Conclusion: the within-video information ceiling
is a property of the 1-fps frozen-feature paradigm on these corpora, not of
the specific encoder (B/16, +OCR, L/14 all land within ~.02). Feature-side
levers exhausted for this iteration; return to objective-side improvements.
