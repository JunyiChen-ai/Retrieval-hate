# LOCO-ST integrity audit (iteration step 15) — 2026-08-30

Independent audit of the C5 span-transfer pilot + step-14 scale-up package.
Scope: leakage, score/rank sanity, coverage, baseline fairness, aggregation
honesty, claim-evidence fit, reproducibility. All recomputations were done
with an independent AUC implementation (sklearn roc_auc_score over the dense
score jsonl files + frozen gold npz), not the shared evaluator.

## VERDICT: ISSUES — no CRITICAL finding; 5 MAJOR, 5 MINOR

The core conclusions survive: no test/val leakage into any gradient path; all
recorded aggregates reproduce exactly from the dense score artifacts; the
baseline gates use the strongest branch of every reproduced method; the EN
baseline win is real and bootstrap-significant. What does NOT survive
unqualified are (a) the "naive MIL destroys ordering everywhere" attribution
claim, (b) the framing of the 3-of-4 gate pass as independent of MHC-ZH, and
(c) exact-rerun reproducibility of individual numbers (GPU nondeterminism of
up to .10 per seed measured inside the package itself).

---

## What was verified clean

1. **Leakage (end-to-end)** — CLEAN.
   - `runs/20260830_powa_within_diagnosis/gt_train_diagnosis_only/*.npz` keys
     checked against `results/reproduction/splits/` for ALL 4 corpora (not a
     sample): hatemm 744/744, mhclip_en 532 (all in train, 18 train videos
     absent = no gold/features), mhclip_zh 561, hateclipseg 251/251. Zero ids
     in val, zero in test, zero outside train.
   - Code paths: `pack_spans` excludes the target corpus by construction
     (`spantransfer.run`, `scale_up.run_valsel/run_zero/run_joint`);
     `pack_weak` reads only `load_labels` video labels; target train spans are
     never loaded anywhere. Val is read only in `eval_split(...,"val")` /
     `eval_val_within` for depth selection (no gradients). Test is read only
     in final scoring.
2. **Score/rank sanity** — CLEAN. Independent recomputation of within-ROC
   macro (mean per-video AUC over hate-labeled both-class test videos) from
   the dense jsonl + gold npz for 5 (corpus, arm) combos across all their
   seeds — hatemm/valsel/5seed, mhclip_en/loo_zero/5seed,
   mhclip_zh/valsel/5seed, hateclipseg/loo_naive/3seed, mhclip_en/joint/3seed
   — matches `scale_results.json` mean AND sd to 4 decimals in every case.
3. **Coverage** — CLEAN. All 16 dense score files per corpus cover exactly
   the gold test set (hatemm 214, mhclip_en 158, mhclip_zh 153, hateclipseg
   79 videos): no duplicates, no missing, no extra ids, every score vector
   length equals its gold array length.
4. **Baseline gates** — CLEAN and conservative. .6315 (multihateloc/
   score_fused, hatemm), .6004 (cmhkf/score_align, en), .5482 (multihateloc/
   score_union, zh), .5619 (vera/score_official_postprocessed, hcs) each
   match `runs/20260830_powa_within_diagnosis/summary.md` (backed by
   summary.json) and each is the maximum within-ROC across every reproduced
   method AND every branch on that corpus — including non-official branches
   (zh gate .5482 is score_union, above the official fused .5120), i.e. the
   gate favors the baselines.
5. **Aggregation/CI honesty** — CLEAN numerically. Every number in README's
   step-14 section matches scale_results.json / bootstrap_ci.md. The paired
   per-video bootstrap (10k) was independently re-implemented and reproduces
   bootstrap_ci.md exactly (hatemm +.0450 [-.0022,+.0938]; en +.1402
   [+.0626,+.2147]; hcs -.0169 [-.0736,+.0385]; hatemm vs zero +.0544
   [+.0283,+.0803]). The `L=min(len)` truncation in the baseline reader is a
   no-op in practice (0 length mismatches in 462 scored rows checked). ns
   results (hatemm flag (c), ZH underpowered, HCS loss) are disclosed, and
   the ZH valsel<zero reversal at 5 seeds is reported rather than hidden.
   Reference columns (weak-MIL .5777/.4592/.4126/.5234; skylines
   .7495/.7692/.6217/.5989) trace to weak_control.json / skyline_train.json.

---

## Findings

### MAJOR-1: "naive MIL destroys ordering everywhere" is contradicted on HateMM by the package's own rerun
Pilot README: "naive MIL finetune degrades ordering everywhere (rank term is
load-bearing)"; step-14: "loo_naive rerun confirms MIL destroys ordering (en
.6077 vs zero .7194)". The scale rerun itself shows hatemm loo_naive 3-seed
.6453±.018 vs loo_zero .6221 (5-seed) / .6283 (pilot 3-seed) — naive does
NOT degrade below zero-shot on HateMM at rerun — and the frozen bootstrap
gives valsel−naive on hatemm +.0313 [-.0028,+.0655], not significant. The
README cites only the EN naive number. The rank-term-load-bearing claim is
strongly supported on EN (sig +.1328) and directionally on ZH/HCS, but must
be restricted accordingly; the "everywhere" and "confirms" wording is
overstated. (Note: only shuf_span was the frozen attribution GATE, and it
passed cleanly; the naive claim is narrative, so no gate is invalidated.)

### MAJOR-2: measured run-to-run nondeterminism up to .10 per seed; exact reruns are not possible
Pilot and scale executed identical code with identical seeds for loo_naive
(same ST.pretrain + ST.adapt path). Per-seed within-ROC differs by up to
.098 (zh seed 234: .7169 → .6191; hatemm seed 234: .5502 → .6209; hatemm
seed 3407: .5696 → .6512) and the 3-seed hatemm mean moved .5840 → .6453
(.061, ~2 seed-sd). No torch.use_deterministic_algorithms / cudnn flags are
set anywhere in the experiment code; no torch/CUDA versions, git commit, or
GPU model are recorded in the new run artifacts. Consequences: (a) any
reported number is a draw from a run distribution wider than the seed sd
suggests; 3-seed deltas of order ≤.06 (e.g. some source-ablation gaps, the
hatemm joint margin) carry less resolution than printed; (b) numbers without
dense scores (see MAJOR-4, MINOR-10) cannot be exactly re-derived, only
re-sampled.

### MAJOR-3: the "3-of-4" PASS is arithmetically load-bearing on MHC-ZH (n=8), contradicting the README's own honesty note
The wins are hatemm/en/zh; hcs fails. Remove ZH and the frozen ">=3 of 4"
clause fails (2 of 4). Yet README states "MHC-ZH within-n = 8 … never
load-bearing" (the pilot plan's original wording was "never load-bearing
*alone*" — the qualifier was dropped). Significance decomposition per the
package's own bootstrap: EN significant (+.1402), HateMM not significant
(+.0450, flag (c) correctly fired and disclosed), ZH not significant / n=8.
The gate was frozen pre-run and passed as written — no protocol violation —
but any downstream claim should be framed as: 1 significant baseline win
(EN), 1 mean-level ns lead (HateMM), 1 direction-only (ZH, n=8), 1 loss
(HCS), rather than an unqualified "beats the best baseline on 3 of 4".

### MAJOR-4: the HCS "first lead" note is inside noise and has no dense-score backing
"Single-source hatemm→hcs reaches within .5686 / frame AP .6321 — ABOVE VERA
(.5619/.6194): the first HCS lead in the study." The within margin is .0067
with a 3-seed sd of .0072 (<1 sd), no paired test, and the sources phase ran
with save=False so no dense scores exist to bootstrap it; given MAJOR-2, a
rerun could plausibly land below VERA. The surrounding text handles post-hoc
selection correctly ("legitimate future amendment, not applied post-hoc"),
but the "ABOVE VERA / first lead" sentence needs an explicit within-noise
caveat or a rerun with saved scores before it appears anywhere.

### MAJOR-5: "target is span-free (video labels only)" overstates — depth selection consumes target val FRAME labels, which trained baselines did not get
valsel picks the adaptation depth by val within-ROC computed from val
frame-level gold (span-derived). This is sanctioned project protocol and
disclosed in PILOT_PLAN/README, but the NOVELTY_DEEP framing sentence
"target is span-free (video labels only)" is not literally true for the
headline arm, and the reproduced weakly-supervised baselines select
checkpoints on video-level signals (hate_common.split_train_val), not frame
labels — an asymmetric selection budget in the method's favor. Mitigations
that should be stated wherever this is framed: loo_zero uses no target
selection at all and still beats the baselines on EN (.7194 vs .6004) and ZH
(.6834 vs .5482); VERA is itself validation-selected. Required fix: "target
TRAIN is span-free; target val frame labels are used for depth selection
(disclosed, k=6 candidate depths)".

### MINOR-6: the valsel arm (A1) was designed after a first look at TEST
The first fixed-depth run failed the frozen no-destruction clause on
EN/ZH test, and A1 was then designed and re-run on the same test split. This
is disclosed in README and PILOT_PLAN, A1's gate was frozen before its run,
and it is permitted by iteration rule 11 branch 3 — but it is one adaptive
test look, and the disclosure must survive into any paper.

### MINOR-7: bootstrap CIs cover video resampling only
Per-video AUC is averaged over seeds and then treated as fixed; seed-level
variance is not propagated, making CIs slightly anti-conservative, and the
paired comparisons mix unequal seed counts (valsel 5 vs naive/baselines 3,
VERA 1). None of the stated significance calls appears fragile to this (EN
margins are large; the borderline hatemm calls are already reported ns), but
say "video-resampling CI, seed means fixed" when reporting.

### MINOR-8: sensitivity "no cliff" claim — low end falls below the baseline gate
The reported range .621–.684 is correct (verified against scale_results),
but tau=.25 gives .6213 < the hatemm gate .6315, i.e. one perturbed setting
would lose to the best baseline. "No cliff" is fair; add that footnote.

### MINOR-9: the EN gate baseline is itself unstable
cmhkf/score_align on EN has seed sd ±.175/.215 around mean .6004 (summary.md
vs OFFICIAL_VAL_RESULTS.md sd conventions differ; both large). Using its
mean as the gate is standard and the paired bootstrap already accounts for
it at the video level, but per-seed the "best EN baseline" is not a stable
object; worth one line in the paper.

### MINOR-10: provenance gaps
- OFFICIAL_VAL_RESULTS.md pins a code commit for the baselines; the new
  pilot/scale artifacts record no commit, env, or GPU model.
- scale_results.json stores only mean/sd (per-seed metrics recoverable from
  dense scores for headline/joint/naive arms — verified — but NOT for the
  sources and sens phases, which saved no scores).
- No number was found whose only provenance is a markdown file: the pilot
  table ↔ results.json, depths ↔ valsel.md ↔ results.json selected_epoch,
  gates ↔ summary.json, references ↔ weak_control/skyline_train.json all
  cross-check. The weakest provenance is the sources/sens aggregates
  (json + rerunnable-but-nondeterministic code).

---

## Recommended actions before the final report
1. Reword the attribution claim: rank term load-bearing on EN (significant),
   directional on ZH/HCS; HateMM mixed (pilot degrade, rerun no-degrade,
   valsel−naive ns). Drop "everywhere"/"confirms … destroys".
2. Replace "never load-bearing" with the plan's original "never load-bearing
   alone", and state the win decomposition (1 sig / 1 ns / 1 direction / 1
   loss) next to any "3-of-4" sentence.
3. Add the within-noise caveat to the HCS single-source note, or rerun that
   cell with dense scores + paired bootstrap before using it.
4. Fix the "span-free (video labels only)" framing per MAJOR-5.
5. Record commit hash + torch/CUDA/GPU in run artifacts; consider
   deterministic flags or, failing that, report run-to-run drift explicitly
   (the .5840→.6453 hatemm naive pair is honest ammunition for a variance
   footnote).

## Audit trail
- Leakage/coverage script, AUC recompute, and bootstrap re-implementation
  run 2026-08-30 in conda env HateVideo (sklearn AUC, scipy-free bootstrap
  reproduction with rng seed 20260830, 10k resamples) — all matched to 4
  decimals; outputs quoted above.
- Files read: PILOT_PLAN.md, SCALE_PLAN.md, README.md, NOVELTY_DEEP.md
  (framing section), spantransfer.py, valsel_arm.py, scale_up.py,
  bootstrap_ci.py, results.json/md, scale_results.json, bootstrap_ci.md,
  valsel.md, run.log, scale_up.log, summary.md/json, OFFICIAL_VAL_RESULTS.md,
  weak_control.json, skyline.json, skyline_train.json,
  eval_baseline_scores.py, hate_common/data.py, skyline.py (TemporalConv).
