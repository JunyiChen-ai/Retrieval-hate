# C5 span-transfer pilot — PASSED (2026-08-30)

Plan/gates: `PILOT_PLAN.md` (frozen; amendment A1 frozen pre-run; both reviews
PASS). Artifacts: `runs/20260830_spantransfer_pilot/` (results.json/md,
valsel.md, logs). All numbers TEST split, within-hate macro ROC primary,
3 seeds (234/2025/3407), shared evaluator.

## Method (as piloted)

Leave-one-corpus-out span transfer: for target X, pretrain the TemporalConv
(frozen CLIP+VGGish+BERT, 1 fps) with frame BCE on the OTHER three corpora's
train spans; adapt to X with top-k MIL on X's video labels plus pairwise
margin-ranking distillation from the pretrained model (tau .5, lambda 1);
adaptation depth in {0,1,2,4,8,15} epochs selected on X's val within-ROC
(epoch 0 = zero-shot). X's own spans never touched.

## Final table (within-ROC macro, TEST)

| target | valsel (method) | loo_zero | loo_adapt e15 | loo_naive | shuf_span | best baseline | weak-MIL ctrl | skyline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hatemm | **.6801±.016** | .6283 | .6659 | .5840 | .5273 | .6315 (MultiHateLoc) | .5777 | .7495 |
| mhclip_en | **.7326±.001** | .7289 | .6610 | .5798 | .4726 | .6004 (CMHKF align) | .4592 | .7692 |
| mhclip_zh | **.6420±.113** | .6373 | .6263 | .5761 | .5458 | .5482 (MultiHateLoc union) | .4126 | .6217 |
| hateclipseg | .5431±.016 | .5389 | .5297 | .5383 | .5022 | .5619 (VERA) | .5234 | .5989 |

Selected depths (per seed): hatemm 4/1/15, en 15/0/0, zh 2/0/1, hcs 1/15/2.

## Gate verdict

- Beats the best reproduced baseline on 3 of 4 corpora (hatemm/en/zh) ✓
- >= loo_zero - .01 on every corpus ✓ (first fixed-depth run FAILED this on
  EN -.068 / ZH -.011; A1's val-selected depth fixed it as designed)
- Attribution: naive MIL finetune degrades ordering everywhere (rank term is
  load-bearing); shuf_span collapses transfer everywhere (span POSITIONS carry
  it, not corpus statistics) ✓

## Honest notes

- MHC-ZH within-n = 8, sd .113 — direction only, never load-bearing.
- HCS remains unsolved (feature deficit corpus: its supervised skyline is .599;
  no arm moves it; VERA .5619 keeps the corpus lead).
- EN frame AP drops for the epoch-0-selected seeds (zero-shot is uncalibrated
  to the corpus); within-video ordering is the primary metric, but the pooled
  column must be reported honestly in any paper table.
- Setting is target-span-free, not classically weakly-supervised: auxiliary
  corpora contribute span supervision. Every table row must disclose this
  (framing per NOVELTY_C5.md: cross-corpus LOCO protocol).

## Step-14 scale-up results (2026-08-30/31, scale_results.json, bootstrap_ci.md)

5-seed headline (within-ROC macro, TEST): valsel .6766/.7405/.6310/.5450,
loo_zero .6221/.7194/.6834/.5438 (hatemm/en/zh/hcs). 3-of-4 baseline wins hold
at 5 seeds. On ZH the 5-seed zero-shot (.6834) exceeds valsel (adaptation adds
nothing there; val n=8 cannot select reliably) — reported honestly.

Comparators: joint multitask (OSAD-style) loses everywhere it matters
(.5959/.6318/.5024 vs valsel on hatemm/en/zh -> the pretrain+protected-adapt
structure is load-bearing; disclosed caveat: valsel keeps a val-selected
epoch-0 fallback that joint lacks). The ordering-destruction finding is
corpus-dependent (audit correction): significant on EN (naive .6077 vs zero
.7194; valsel-naive +.1328 CI [+.0798,+.1852]) and present on ZH/HCS, but the
HateMM naive rerun landed ABOVE zero-shot (.6453 vs .6221) and valsel-naive is
ns there [-.0028,+.0655] — do not claim "destroys everywhere".

Run-to-run nondeterminism (audit): identical code+seed reruns drift up to
~.06 in a 3-seed within-ROC mean (no deterministic-algorithms flags); treat
3-seed deltas below .06 as unresolved. Code commit for all scale runs:
9aa1aab (and 045407d for the pilot arms).

Source ablations (zero-shot, 3 seeds): HateMM spans are the dominant source
(en: union .7194, minus-hatemm .5978; zh: hatemm-alone .7761 vs union .6834).
Single-source hatemm->hcs reaches within .5686 / frame AP .6321, nominally
above VERA (.5619/.6194) — but the margin (.0067) is inside one seed-sd, no
dense scores were saved for this arm and no paired test exists: a
within-noise observation, not a claimed lead. Source-set selection (by val)
is a legitimate future amendment, not applied post-hoc here.

Win decomposition (audit-corrected): one significant baseline win (EN), one
non-significant mean lead (HateMM), one direction-only n=8 result (ZH — it IS
arithmetically load-bearing for the 3-of-4 gate), one loss (HCS). The
"target-span-free" label must always disclose that adaptation-depth selection
consumes target val frame labels (baselines select on video-level signals
only); mitigation: loo_zero, which uses no target selection at all, also
beats the baselines on EN/ZH.

Sensitivity (hatemm, seed 234, one-at-a-time): within-ROC stays .621-.684
across tau {.25,1}, lambda {.3,3}, top-k {T/4,T/16} — no cliff.

Paired per-video bootstrap (10k, within-AUC):
- EN: valsel - CMHKF +.1402 [+.0626,+.2147] SIGNIFICANT; valsel - naive
  +.1328 [+.0798,+.1852] SIGNIFICANT.
- HateMM: valsel - MultiHateLoc +.0450 [-.0022,+.0938] not significant
  (frozen flag (c) fires: reported as a mean-level, non-significant lead);
  valsel - zero +.0544 [+.0283,+.0803] SIGNIFICANT (adaptation helps).
- ZH: +.0828 vs baseline, CI [-.0912,+.2450] (n=8, underpowered, direction+).
- HCS: -.0169 vs VERA (ns); note the hatemm-single-source row above.

## Next (iteration step 15-16)

Integrity audit; final report.
