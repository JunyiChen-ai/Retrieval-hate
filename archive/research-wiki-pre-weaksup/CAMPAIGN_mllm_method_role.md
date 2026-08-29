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
p=0.007, CI excludes null; magnitude modest, statistics solid — later amplified to **modest-plus** by
the 72B A-fuse scorer, HateClipSeg test wv-AUC **0.5755**, §P10-b). The **main-table accuracy role is
exhaustively refuted**: across **eleven pre-registered main-table routes at 7B–72B scale**
(P1/P2/P2b/P3-EN,ZH,HateMM/P4/P5/P7, plus P9/P9b the **decision level** and its rgcl-ON wave — P9/P9b
counted separately) no MLLM component lifts static test accuracy beyond the ~1.6-video (≈1 acc pt)
noise floor, and every verdict is guard-backed (reproduction / bit-for-bit / probe).
**Endgame tally — CAMPAIGN CLOSED (2026-07-09): all 13 pre-registered routes concluded.** The
**eleven main-table-accuracy routes are all dead** (kill / within-noise), settling the
no-main-table-role verdict; the separate **localization track** then ran its full arc: **P6 positive**
→ **P10-b** the 72B A-fuse amplifier promoted (calib 0.5913) and spent the single HateClipSeg test at
wv-AUC **0.5755 = MODEST** → **P10-c** a new-generation open scorer **FAIL** (Qwen3-VL-32B A-fuse
**0.5866**, a *same-tier* generational upgrade that does not surpass the 72B champion; dies
calibration-side, test never re-touched) → **P11** the weak-sup localization-*training* route
**probe-killed** (the 72B zero-shot edge is a coarse×fine aggregation trick, not a better per-segment
labeller — a video-label MIL head already carries most of it). **Three walls now bound the
localization ceiling below the 0.60 substantial line: the reaggregation ceiling 0.5932, the
72B-champion calibration 0.5913→test 0.5755, and the new-gen same-tier 0.5866 — all < the 0.616
waterline the calib→test mapping demands for a substantial test.** The roles the MLLM actually earned
are **encoder + localizer (modest-plus) + guard-rail/audit** (§4); there is **no main-table accuracy
role**. **P9 closed the last architectural locus**: even LoRA-SFT-ing
the *whole* LMM as the classifier only *matches* our existing LoRA-encoder+RGCL route on ZH (+1.0pt
vs the protocol-matched final-epoch floor, within noise), is noise on EN, and confirms the same
pattern on HateMM (head +0.9 ≈ floor) — while reading that same SFT'd backbone through our *retrieval
memory* loses on all three (−2.2 ZH / −2.7 EN / −4.7 HateMM), so the MLLM's own head **displaces**
rather than **enhances** the memory pillar. Two methodology takeaways generalize: **(i) a passing
no-head probe is *necessary but not sufficient*** — HateMM had the cleanest probe of the three yet the
learned align-fusion head washed the input reweighting out (P3-HateMM); **(ii) semantic competence is
*orthogonal to, or redundant with, the decision variable*** — comparability ⊥ vote-correctness,
era-drifting verdict rates, schema fields ⊂ the label (P1/P2/P2b/P4/P5). Semantic aboutness is not the
same quantity as which side of the hate/offensive/benign boundary — and it is that boundary, already
directly supervised, that a main-table lift would need moved.

---

## 1. Scoreboard

| front | MLLM's method job | pre-registered bar | result | why it failed (one line) | doc · commit |
|---|---|---|---|---|---|
| **P1** zero-label prior recal | read archive → label-free HARMFUL/BENIGN → adjusted classify-and-count prior p̂ → quantile-match the drift-gated vote threshold | \|p̂−true\|≤0.07; zero-label recovers ≥60% of the labeled-recal gap (EN); ZH control unharmed | **FAIL.** repro exact; p̂ err **0.22 EN / 0.18 ZH**; corrected recal 0.48 < static 0.63 (EN); ZH forced −0.055 | MLLM verdict **FPR drifts across the very temporal boundary** being adapted to (EN .372→.238, ZH .314→.402) → biased count. Mechanism sound (oracle-prior recovers 80% of EN gap); MLLM can't supply the prior | EXP_p1_zerolabel_recal · `2a69246` |
| **P2** 7B neighbor rerank | margin-gate boundary queries; 7B judges pairwise COMPARABLE/INCOMPARABLE per neighbor (label-blind); drop INCOMPARABLE before revote | B−A>0 EN (≥3/4 gated+); rent test B>C; no ZH harm | **FAIL.** repro exact; B−A **−0.002 EN / −0.020 ZH**; B−C within-noise EN / −0.017 ZH; ZH harmed 4/5 seeds | **over-flags INCOMPARABLE** (83% EN / 70% ZH) off sparse archives (role-3's ratchet relocated); drops **indiscriminate** (selectivity lift +1.1% EN / −3.2% ZH) | EXP_p2_neighbor_rerank · `bc689e1` |
| **P2b/P2c** stronger judge + train-side calibration | same harness; TRAIN-side labeled selectivity leaderboard over **7B/32B/72B** × archive/+transcript × orig/flip; promote only if EN lift ≥+10pt | a config clears **+10pt EN selectivity lift** on the train benchmark | **FAIL (dies train-side, no test contact).** best EN lift **+2.7pt**; ZH lift **negative for all 8 configs** across the full scale ladder | **comparability ⊥ vote-correctness at every open-source scale**: across 3 models (7B/32B/72B), 2 evidence sets, 2 prompts, topical comparability is ~independent of label-match. **Calibration improves with scale** (orig drop-rate 7B 72.5%→32B 64.6%→72B 30.9% EN) but **selectivity does not** (lift ≤+2.7 anywhere, ZH negative everywhere); a bigger judge is better-behaved, not more selective | EXP_p2b_stronger_judge · `cc4ca6e`,`aae1efe` |
| **P3-EN** evidence-density pooling | MLLM scores each K=4 segment's hate-evidence density 0–3; softmax-reweight the mean-pooled video img embedding toward evidence-bearing segments (label-free input processing) | equal-weights==mean bit-for-bit; **probe gate** weighted≥mean (concat LOO kNN @k20, EN train) | **FAIL (probe KILL, EN not trained).** sanity exact; gate **−0.0055 @k20** | **signal real, intervention doesn't translate**: hateful/benign within-video score var **1.11/0.40**, yet concentrating the localized *visual* signal doesn't separate better than the mean in frozen CLIP once fused with the unchanged text | EXP_p3_evidence_pooling · `c2ba59f` |
| **P3-ZH** same (control) | " | probe pass → train; ΔF1>1pt, ≥2/3 seeds, both protocols | **FAIL (within-noise, no claim).** probe +0.0017 (thin); train val-sel ΔF1 −0.007, final +0.009 — both <1pt | ZH evidence is ASR-poor (score var 0.33/0.12); the thin probe pass predicted the within-noise train result | EXP_p3_evidence_pooling · `15f5f08` |
| **P3-HateMM** same | " | probe pass (**PASS, k-consistent +0.0108**) → train | **FAIL (within-noise, no claim).** Cleanest probe pass of the three (densest evidence, var 1.28/0.71) yet trained wsoftT1 vs floor: val-sel ΔF1 −0.0041, final-ep +0.0004 — both <1pt. Floor reproduces published 0.828 acc. **Decisive: a passing no-head probe does NOT guarantee a training gain — the learned align-fusion head absorbs the input-space reweight.** | EXP_p3_evidence_pooling · `22fe62a` |
| **P4** schema-field distillation | aux linear heads on the fused embedding predict MLLM archive fields (explicitness/modality/mechanism/target_group); L=main+0.1·Σaux; heads dropped at eval | λ=0 bit-for-bit; probe gate (decodable + label-informative); aux beats floor >1pt, ≥2/3 seeds, both protocols | **FAIL (within-noise).** bit-for-bit exact; **probe PASS** (fields decodable AUC .62–.93, label-informative AUC .74–.78); train EN final −0.001, ZH +0.008 (sub-threshold); val-sel negative | fields real but **redundant** with the direct hateful-label supervision the embedding already receives — distilling adds nothing beyond the label | EXP_p4_schema_distill · `6f1f0da`,`00816aa` |
| **P7** score-level fusion | fuse the visual kNN vote share with the MLLM semantic channel (bin=P1 verdict / dens=P3 density) at the SCORE level via two frozen rules (R1 rank-average, R2 band-limited veto-boost) | **train-side gate**: some rule corrects ≥15% of seed-0 LOO errors net-of-damage (no test contact) | **FAIL (train-side KILL, premise refuted).** every rule×channel net **−0.10…−0.38** (damages > corrects); no test spent | **channels are NOT decorrelated**: corr(channel, vote share) **+0.21…+0.51** (positive), and the channel is the weaker classifier (AUC 0.54–0.69 vs floor LOO acc 0.81–0.86) → agrees where vote is right, adds noise where it's wrong | EXP_p7_score_fusion · `8f920e5` |
| **P5** counterfactual twins | MLLM rewrites each TRAIN positive's transcript into a sanitized counterfactual; twin = anchor's REAL img + sanitized-text embedding; one extra per-anchor hard negative | flag-off bit-for-bit; **quality gate** flip≥0.80 + hardness; cf beats floor >1pt, ≥2/3, both protocols | **FAIL.** bit-for-bit exact; **gate CLOSED** (flip **0.503 EN / 0.337 ZH**, hardness pass); diagnostic cf **hurts EN −0.027**, flat ZH; cfrand ≈ cf | MLLM **can't reliably manufacture** the clean counterfactual (half EN / two-thirds ZH still harmful); and clean twins hurt because they **share the anchor's visuals → too close** (cos 0.73), so repelling fights the visual signal | EXP_p5_counterfactual_negs · `fc25cac`,`66d3103` |
| **P6** localization scorer *(the one PASS)* | per-window MLLM evidence scores (frames + ASR) rank HateClipSeg windows for **span-free temporal localization** (memory-free saliency) | within-video mean-AUC(MLLM) > memory AND random; b's 95% CI excludes 0.5; sign-test p<0.05 | **PASS — earns a removable role.** wv-AUC **0.5435** vs memory 0.5140 / random 0.5088; paired b>a **Δ+0.0296** CI[+.009,+.050] **p=0.007**; vs-null p=5.4e-8; seg-AUC 0.635 vs 0.584 | *win, not fail:* the same evidence signal P3 couldn't **pool** is a genuine **localizer** — magnitude modest, statistics solid | EXP_p6_mllm_localization · `c9e3bd8` |
| **P9** decision-level LMM-SFT *(LMM-RGCL stage-2)* | LoRA-SFT the **whole** Qwen2.5-VL LMM + its own classifier head (RA-HMD `sft_classifier`, rgcl-OFF); two read-outs — the in-LMM MLP head (C3-mlp) and OUR kNN over the SFT'd embeddings (C3-knn) | C3 beats the frozen floor >1pt, ≥2/3 seeds, both read-outs; user goal = *substantial* + *novel integration* | **FAIL vs protocol-matched floor (EN/ZH/HateMM — same pattern on all three).** C3-mlp test: EN 0.7909 = **+0.6pt** vs frozen best 0.7847 (noise); ZH 0.8635 = **+4.5 vs frozen but only +1.0 vs our LoRA final-epoch 0.8537** (noise); HateMM head +0.9 ≈ floor. **C3-knn (our memory) EN −2.7 / ZH −2.2 / HateMM −4.7 BELOW floor.** | the ZH gain vs frozen is the **LoRA benefit we already had**; the LMM's own head only *matches* our LoRA-encoder+RGCL route, and our retrieval read-out on the SFT'd space **loses on both** ⇒ the MLLM **displaces**, not enhances, the memory pillar. Fork finding: RA-HMD stage-2 ships rgcl-OFF, needs 5 fixes (incl. never reloads classifier.bin) | EXP_p9_lmm_rgcl_video · `455e666` |
| **P9b** rgcl-ON wave *(12-run, 2026-07-08)* | turn **ON** our retrieval-contrastive (rgcl) loss while LoRA-SFT-ing the LMM (arm D3), training the embedding space toward the kNN memory vote; read BOTH the in-LMM MLP head (D3-mlp) and OUR kNN over the SFT'd embeddings (D3-knn); C3′ = rgcl-OFF control | crit 1: D3-knn beats protocol-matched floor by >1.5pt, ≥2/3 seeds, both langs; crit 2: D3-knn ≥ D3-mlp − 1pt | **FAIL (crit 1; crit 2 PASS).** D3-knn test ZH **0.8389±0.005** vs floor 0.8537 (**−1.5pt, 0/3 seeds**), EN **0.7743±0.008** vs 0.7847 (**−1.0pt, 0/3**); crit 2 PASS (knn−mlp: ZH −0.2 / EN −0.6). **0/12 cells over floor.** | rgcl term trains the space toward the memory vote (D3−C3′ knn **+1.8pt ZH / +0.2pt EN**) but the in-LMM head mirrors DOWN (**−1.8 / −1.2 mlp**) → **head↔memory accuracy redistribution, not net gain** | EXP_p9_lmm_rgcl_video "P9b WAVE RESULTS" · `4d28655` (+ LLAMA-FACTORY submodule `b132bc4d`) |
| **P10** localization amplification *(round 1, 2026-07-08)* | freely calibrate the P6 scorer on HateMM spans, single-shot test on HateClipSeg; push P6's *modest* localization gain to *substantial* | a variant clears **≥ +0.04** paired wv-AUC Δ vs the P6 anchor (0.5387) on the HateMM calib set, CI excluding 0 → earns the single HateClipSeg test | **FAIL / no promotion.** best = **A-fuse (K4×K30 coarse×fine) +0.0305** CI[+0.0175,+0.0437] p=7e-7 — significant but **< +0.04 bar**; K60/fewshot/A-gate/A-lex all ≤anchor or ≤+0.006. **Round-1 test NOT touched — P6 stood (wv-AUC 0.5435).** **Later rounds (§P10-b/§P10-c): P10-b DONE** — promoted 72B A-fuse (calib 0.5913) spent the single HateClipSeg test at wv-AUC **0.5755 = MODEST** (CI [0.5581,0.5933]); **P10-c FAIL** — new-gen Qwen3-VL-32B A-fuse **0.5866 < 0.616** gate, same-tier as the 72B champion, test never re-spent. Localizer role upgraded modest → **modest-plus**. | A-fuse is the single significant needle-mover but +0.01 short of the round-1 bar; it amplifies *localization*, not main-table accuracy | EXP_p10_loc_amplify · `7194ee2`,`93e82fa` |
| **P11** MLLM-weak-sup localization *training* (13th pre-reg route) | distil the 72B A-fuse per-segment density into a **trained** segment head; beat (A) video-label-only MIL and (C) a memory-kNN weak-labeller on HateClipSeg weakly-sup localization | **probe gate** (HateMM calib, before any training): wv-AUC(MLLM 72B A-fuse) − wv-AUC(MIL-proxy) ≥ +0.03 AND paired CI excl 0; §3 success line B−A & B−C ≥ +0.05 CI-excl-0 + B abs ≥ 0.65 | **PROBE FAIL → killed (conservative).** committed letter gate (A-fuse − MIL **K4**) **+0.0386** CI[+0.0037,+0.0749] excl 0 **passes** but is granularity/operator-confounded; the binding same-operator matched gate (A-fuse − MIL **A-fuse**, fixed before K30 landed) **+0.0359** CI[−0.0009,+0.0730] (misses by 0.0009), sign-p **0.13** n.s.; raw-vs-raw both K also n.s. (+0.0058 K4 / +0.0143 K30). No training submitted; **HateClipSeg test split frozen, never consumed** | 72B zero-shot edge = coarse×fine **aggregation trick**, not a better per-segment labeller — grant a video-label MIL head the same trick and the gap collapses to n.s.; a 5-fold linear MIL head already ~0.55 wv-AUC ⇒ **video labels alone already contain most of what the MLLM weak label would teach**; §3 line provably unreachable (teacher edge ≤+0.036 n.s. < +0.05; teacher abs 0.59/0.5755 < 0.65). Zero training cost | EXP_p11_weaksup_localization · `eaf72db`,`0b3cf40` |

Noise-floor convention (all fronts): 1 acc pt ≈ 1.6 videos on these ~150-sample test sets;
sub-1pt effects are reported as **within-noise, no claim** — the headline is the paired-delta sign
pattern, not a p-value. No cross-seed ensembling anywhere.

**结题(2026-07-09):13 条预注册路线全部结题。** P11(MLLM 弱监督定位训练)probe-kill 之后,campaign
再无在册未决路线;主表 accuracy 角色被彻底证伪的终局不变,MLLM 挣得的可移除角色 = **encoder + 定位打分器
(P6 → P10-b modest-plus)+ guard-rail/审计**,而**弱监督定位训练**角色现亦被 P11 证伪(72B zero-shot 优势是
聚合技巧而非更好段级标注,video-label MIL 已含大部分可教信息)。

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
   **⚠ F88 correction (2026-07-26, `refine-logs/ERRPAT_MHC-EN_2026-07-26.md` §6.5, commit `ad56a62`):
   "improves EN" is a SINGLE-SEED reading** — the human 2-entry deletion is +0.0124 on seed 0 and
   **zero vote flips on seeds 1–3** (four-seed mean +0.0031); the 14-id rule list is +0.0093 acc /
   +0.0089 mF1, 3/4 seeds, 0 broken, still sub-bar. Cite as *human-in-the-loop capability
   demonstration, single-seed; not an accuracy claim.* The integrity/controllability framing is
   unchanged and is what this cell actually supports.
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
