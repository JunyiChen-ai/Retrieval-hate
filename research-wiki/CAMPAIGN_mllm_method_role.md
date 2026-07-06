# Campaign synthesis — can the MLLM earn a *method* role?

**Question (user mandate).** Beyond serving as a frozen *encoder*, can a 7B/32B-class MLLM earn a
genuine **method role** in this hateful-video pipeline — a component whose **removal measurably
costs accuracy**? Every front below gives the MLLM a distinct, non-encoder job, pre-registers a
bar, and includes the ablation "remove the MLLM." A role is earned only if removing it costs
something beyond the ~1.6-video (≈1 acc pt) noise floor of these ~150-sample test sets.

**Bottom line — the campaign answer.** The MLLM earns exactly **two** removable method roles here:
as the **frozen encoder** (Qwen features beat CLIP on HateMM by +4.2 macro-F1, crossing 0.85) and
as a **span-free localization scorer** (P6 — its per-window evidence scores rank hate windows better
than the retrieval memory and random: within-video AUC **0.5435 vs 0.5140 / 0.5088**, paired b>a
p=0.007, CI excludes null; magnitude modest, statistics solid). The **main-table accuracy role is
exhaustively refuted**: across **eight pre-registered routes at 7B–32B scale**
(P1/P2/P2b/P3-EN,ZH,HateMM/P4/P5) no MLLM component lifts static test accuracy beyond the
~1.6-video (≈1 acc pt) noise floor, and every verdict is guard-backed (reproduction / bit-for-bit /
probe). Two methodology takeaways generalize: **(i) a passing no-head probe is *necessary but not
sufficient*** — HateMM had the cleanest probe of the three yet the learned align-fusion head washed
the input reweighting out (P3-HateMM); **(ii) semantic competence is *orthogonal to, or redundant
with, the decision variable*** — comparability ⊥ vote-correctness, era-drifting verdict rates,
schema fields ⊂ the label (P1/P2/P2b/P4/P5). Semantic aboutness is not the same quantity as which
side of the hate/offensive/benign boundary — and it is that boundary, already directly supervised,
that a main-table lift would need moved.

---

## 1. Scoreboard

| front | MLLM's method job | pre-registered bar | result | why it failed (one line) | doc · commit |
|---|---|---|---|---|---|
| **P1** zero-label prior recal | read archive → label-free HARMFUL/BENIGN → adjusted classify-and-count prior p̂ → quantile-match the drift-gated vote threshold | \|p̂−true\|≤0.07; zero-label recovers ≥60% of the labeled-recal gap (EN); ZH control unharmed | **FAIL.** repro exact; p̂ err **0.22 EN / 0.18 ZH**; corrected recal 0.48 < static 0.63 (EN); ZH forced −0.055 | MLLM verdict **FPR drifts across the very temporal boundary** being adapted to (EN .372→.238, ZH .314→.402) → biased count. Mechanism sound (oracle-prior recovers 80% of EN gap); MLLM can't supply the prior | EXP_p1_zerolabel_recal · `2a69246` |
| **P2** 7B neighbor rerank | margin-gate boundary queries; 7B judges pairwise COMPARABLE/INCOMPARABLE per neighbor (label-blind); drop INCOMPARABLE before revote | B−A>0 EN (≥3/4 gated+); rent test B>C; no ZH harm | **FAIL.** repro exact; B−A **−0.002 EN / −0.020 ZH**; B−C within-noise EN / −0.017 ZH; ZH harmed 4/5 seeds | **over-flags INCOMPARABLE** (83% EN / 70% ZH) off sparse archives (role-3's ratchet relocated); drops **indiscriminate** (selectivity lift +1.1% EN / −3.2% ZH) | EXP_p2_neighbor_rerank · `bc689e1` |
| **P2b** stronger judge + train-side calibration | same harness; TRAIN-side labeled selectivity leaderboard over 7B/32B × archive/+transcript × orig/flip; promote only if EN lift ≥+10pt | a config clears **+10pt EN selectivity lift** on the train benchmark | **FAIL (dies train-side, no test contact).** best EN lift **+2.7pt**; ZH lift **negative for all 6 configs** | **comparability ⊥ vote-correctness**: across 2 models (incl. 32B), 2 evidence sets, 2 prompts, topical comparability is ~independent of label-match. Prompt-flip fixed the *drop-rate* (72→58%) not selectivity; +transcript & 32B added nothing | EXP_p2b_stronger_judge · `cc4ca6e` |
| **P3-EN** evidence-density pooling | MLLM scores each K=4 segment's hate-evidence density 0–3; softmax-reweight the mean-pooled video img embedding toward evidence-bearing segments (label-free input processing) | equal-weights==mean bit-for-bit; **probe gate** weighted≥mean (concat LOO kNN @k20, EN train) | **FAIL (probe KILL, EN not trained).** sanity exact; gate **−0.0055 @k20** | **signal real, intervention doesn't translate**: hateful/benign within-video score var **1.11/0.40**, yet concentrating the localized *visual* signal doesn't separate better than the mean in frozen CLIP once fused with the unchanged text | EXP_p3_evidence_pooling · `c2ba59f` |
| **P3-ZH** same (control) | " | probe pass → train; ΔF1>1pt, ≥2/3 seeds, both protocols | **FAIL (within-noise, no claim).** probe +0.0017 (thin); train val-sel ΔF1 −0.007, final +0.009 — both <1pt | ZH evidence is ASR-poor (score var 0.33/0.12); the thin probe pass predicted the within-noise train result | EXP_p3_evidence_pooling · `15f5f08` |
| **P3-HateMM** same | " | probe pass (**PASS, k-consistent +0.0108**) → train | **FAIL (within-noise, no claim).** Cleanest probe pass of the three (densest evidence, var 1.28/0.71) yet trained wsoftT1 vs floor: val-sel ΔF1 −0.0041, final-ep +0.0004 — both <1pt. Floor reproduces published 0.828 acc. **Decisive: a passing no-head probe does NOT guarantee a training gain — the learned align-fusion head absorbs the input-space reweight.** | EXP_p3_evidence_pooling · `22fe62a` |
| **P4** schema-field distillation | aux linear heads on the fused embedding predict MLLM archive fields (explicitness/modality/mechanism/target_group); L=main+0.1·Σaux; heads dropped at eval | λ=0 bit-for-bit; probe gate (decodable + label-informative); aux beats floor >1pt, ≥2/3 seeds, both protocols | **FAIL (within-noise).** bit-for-bit exact; **probe PASS** (fields decodable AUC .62–.93, label-informative AUC .74–.78); train EN final −0.001, ZH +0.008 (sub-threshold); val-sel negative | fields real but **redundant** with the direct hateful-label supervision the embedding already receives — distilling adds nothing beyond the label | EXP_p4_schema_distill · `6f1f0da`,`00816aa` |
| **P7** score-level fusion | fuse the visual kNN vote share with the MLLM semantic channel (bin=P1 verdict / dens=P3 density) at the SCORE level via two frozen rules (R1 rank-average, R2 band-limited veto-boost) | **train-side gate**: some rule corrects ≥15% of seed-0 LOO errors net-of-damage (no test contact) | **FAIL (train-side KILL, premise refuted).** every rule×channel net **−0.10…−0.38** (damages > corrects); no test spent | **channels are NOT decorrelated**: corr(channel, vote share) **+0.21…+0.51** (positive), and the channel is the weaker classifier (AUC 0.54–0.69 vs floor LOO acc 0.81–0.86) → agrees where vote is right, adds noise where it's wrong | EXP_p7_score_fusion · `8f920e5` |
| **P5** counterfactual twins | MLLM rewrites each TRAIN positive's transcript into a sanitized counterfactual; twin = anchor's REAL img + sanitized-text embedding; one extra per-anchor hard negative | flag-off bit-for-bit; **quality gate** flip≥0.80 + hardness; cf beats floor >1pt, ≥2/3, both protocols | **FAIL.** bit-for-bit exact; **gate CLOSED** (flip **0.503 EN / 0.337 ZH**, hardness pass); diagnostic cf **hurts EN −0.027**, flat ZH; cfrand ≈ cf | MLLM **can't reliably manufacture** the clean counterfactual (half EN / two-thirds ZH still harmful); and clean twins hurt because they **share the anchor's visuals → too close** (cos 0.73), so repelling fights the visual signal | EXP_p5_counterfactual_negs · `fc25cac`,`66d3103` |
| **P6** localization scorer *(the one PASS)* | per-window MLLM evidence scores (frames + ASR) rank HateClipSeg windows for **span-free temporal localization** (memory-free saliency) | within-video mean-AUC(MLLM) > memory AND random; b's 95% CI excludes 0.5; sign-test p<0.05 | **PASS — earns a removable role.** wv-AUC **0.5435** vs memory 0.5140 / random 0.5088; paired b>a **Δ+0.0296** CI[+.009,+.050] **p=0.007**; vs-null p=5.4e-8; seg-AUC 0.635 vs 0.584 | *win, not fail:* the same evidence signal P3 couldn't **pool** is a genuine **localizer** — magnitude modest, statistics solid | EXP_p6_mllm_localization · `c9e3bd8` |

Noise-floor convention (all fronts): 1 acc pt ≈ 1.6 videos on these ~150-sample test sets;
sub-1pt effects are reported as **within-noise, no claim** — the headline is the paired-delta sign
pattern, not a p-value. No cross-seed ensembling anywhere.

---

## 2. What six fronts consistently show the MLLM **CAN** do

- **Read the structured archive competently.** The label-blind archive audit re-finds
  human-flagged label noise (auto-memory-repair), and on clean cases the flip reasons are correct
  (P5 sanitization on EN: "FAGGY FF"→removed, "Cuck Dad"→"Dads"). Archives are judgeable end-to-end
  (P1/P2 ran on them with ~100% strict-parse).
- **Produce genuinely localized evidence signals.** *Strongest positive of the campaign (P3):* the
  per-segment hate-evidence-density scores separate hateful from benign in the right direction on
  all three datasets (within-video score var hate/benign **1.11/0.40 EN, 1.28/0.71 HateMM**,
  0.33/0.12 ZH) — a calibrated, label-free saliency map for *where* the hate is.
- **Produce decodable, label-correlated structured fields.** P4's probe: every archive field is
  linearly decodable from frozen CLIP (AUC .62–.93) *and* the fields predict the video label
  (AUC .74–.78). The semantic content is really in there.

## 3. What it consistently **CANNOT** do (and the unifying reason)

- **Absolute arbitration at the break-even point.** Role-3 (label the deferred queue) and P2
  (flag INCOMPARABLE) both collapse into a one-way **over-flag ratchet** off a generic prior.
- **Supply a decorrelated late-fusion channel.** P7: the MLLM semantic channel is *positively*
  correlated with the visual kNN vote (corr +0.21…+0.51) and the weaker classifier, so score-level
  fusion damages more errors than it corrects (net −0.10…−0.38). The "orthogonal error channels"
  intuition is empirically false here — measured, not assumed.
- **Era-stable estimation under drift.** P1: the verdict's own error rates move across the
  temporal boundary, so a train-calibrated count is biased exactly where it's applied.
- **Reliable counterfactual manufacture.** P5: only ~50% (EN) / ~34% (ZH) of its sanitized
  rewrites pass its *own* harm check.
- **Predict vote-correctness from comparability.** P2/P2b (2 models incl. 32B, 2 evidence sets,
  2 prompts): whether a neighbor is topically comparable is ~independent of whether its label
  matches (|selectivity lift| ≤ 2.7pt EN, wrong-signed ZH).

**Recurring failure shape.** In every case the MLLM has real *semantic* competence, but that
competence is **orthogonal to the decision variable** (comparability ⊥ vote-correctness;
localized-visual-evidence ⊥ frozen-CLIP separability; verdict-rate drifts off the prior it
estimates) **or redundant with it** (schema fields ⊂ the hateful label the head already trains on).
Semantic aboutness is not the same quantity as "which side of the hate/offensive/benign boundary,"
and it is that boundary — already supervised directly — that the method needs moved.

## 4. What survives for the paper (independent of the method-role kills)

1. **Guard-rail / editable-memory role.** The auditable archive memory supports a *veto*: targeted
   deletion of MLLM-flagged noisy entries improves EN (auto-memory-repair). Removal cost shows up
   as **integrity/controllability**, not raw accuracy — a defensible contribution framing.
2. **Human-in-the-loop audit.** The archive re-finds human-labeled noise → an auditable
   memory-hygiene tool, orthogonal to the accuracy claim.
3. **Localization scorer — an EARNED, statistically-validated method role (P6).** The per-window
   MLLM evidence scores (`data/MLLM_scores/<DS>/*_segscoreK4_qwen.jsonl`) rank hate windows on
   HateClipSeg **better than the retrieval memory and random**: within-video AUC 0.5435 vs
   0.5140 / 0.5088, paired b>a p=0.007 (CI excludes 0), vs-null p=5.4e-8. The same vector that
   failed at *pooling* (P3) is a genuine *localizer* — modest magnitude, solid statistics — so this
   is a removable role, not just material. (EXP_p6_mllm_localization; cross-ref
   EVAL_localization_hateclipseg / EVAL_localization_hatemm.)
4. **Quantified oracle bar for future work.** P2's oracle membership editor (drop by *true* label)
   lifts the gated slice to 100% and overall accuracy **+7.5pt EN / +10.6pt ZH, both across 0.85**.
   The gate is sound and the prize is real; P2b shows a *stronger comparability judge* is not the
   key that unlocks it. This is the concrete headroom + the ruled-out approach that scope future
   "membership signal" work.

---

## SLOTS (to close when the last verdicts land)

- **[RESOLVED — `22fe62a`] P3-HateMM training verdict.** The campaign's single cleanest probe positive
  (+0.0108 @k20, k-consistent, densest evidence) trained to **within-noise** anyway: wsoftT1 vs floor
  val-sel ΔF1 −0.0041 / final-ep +0.0004 (both <1pt, fails ≥2/3-seeds too). Floor reproduces the
  published 0.828 acc. So the EN/ZH pattern held: **evidence-density pooling earns NO method role on
  any of EN/ZH/HateMM.** Decisive lesson: a passing no-head probe is necessary but *not sufficient* —
  the learned align-fusion head (img×text) absorbs the small input-space reweight. The per-segment
  MLLM scores remain valuable as a **localization** signal (P6), not a pooling one. §1 P3-HateMM row
  + EXP_p3_evidence_pooling §3.2/§4 updated.
- **[RESOLVED, landed post-assignment — `cc4ca6e`] P2b 32B leaderboard.** Filled above: no config
  cleared the +10pt bar (best EN +2.7, ZH negative for all six incl. both 32B configs); P2b died
  train-side with no test contact. Recorded as final.

*Scoreboard numbers are quoted from each front's committed EXP doc; this synthesis adds no new
measurement. Update the two SLOTS above rather than re-deriving.*
