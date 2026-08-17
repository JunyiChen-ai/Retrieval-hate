# P-A result — Disagreement retrievability gate

- **Freeze:** `idea-stage/PILOT_FREEZE_2026-08-09.md`, section P-A (written before any candidate metric was computed).
- **Script:** `idea-stage/pilot_a_disagreement_retrievability.py`
- **Raw results:** `idea-stage/pilot_a.json`
- **Log:** `logging/runs/pilot_a/run.log`
- **Run:** 2026-08-09 08:36:11–08:36:12, CPU, conda `HateVideo`, single submission, 0.86 s.
- **Test-set contact:** none. Guard armed and logged (`GUARD ARMED: any path whose name contains
  'test' HALTs`). Guard self-tested: `data/gt/MHC/test.jsonl`, `.../MHC/test_seen_*.pt` and
  `mhc_English_test.tsv` all raise `HALT_TEST_CONTACT`; `data/gt/MHC/train.jsonl` passes.
  The complete list of paths the real run touched is in `pilot_a.json:paths_touched` — 12 files,
  all `train` / `val` / `valid` / `dev_seen`.

---

## 1. The frozen rule, transcribed unedited

> **P-A — Disagreement retrievability gate**
>
> **What family it gates.** Every vote-based candidate (agreement-shaped retrieval geometry,
> dissent-preserving memory, contested-item abstention). All of them assume a **necessary
> condition**: that an item's *contestedness* is predictable from its retrieval neighbourhood.
> If a neighbourhood cannot carry disagreement, none of the family has a mechanism.
>
> **Data.** MultiHateClip EN (train 549 + val 80 = 629) and ZH (train 579 + val 78 = 657).
> `test.jsonl` is never opened. Votes joined from the upstream official release
> `{English,Chinese}_data/annotation/{train,valid}.tsv` by `Video_ID` (join verified 100 %).
>
> **Features.** Frozen CLIP ViT-L/14-336 caches already on disk:
> `data/CLIP_Embedding/{MHC,MHC_zh}/{train,dev_seen}_openai_clip-vit-large-patch14-336_HF.pt`.
> Key = `[l2(img_feats) ‖ l2(text_feats)]`, similarity = dot product (= sum of the two per-block
> cosines). This is arm 0 of the late-interaction pilot, chosen so the result is comparable to it.
>
> **Targets (both frozen, both binary, computed per item from the raw vote list).**
> - **T1 `non_unanimous`** — the vote multiset contains ≥2 distinct labels.
> - **T2 `binary_split`** — the votes disagree after mapping to this project's binary protocol
>   (`Normal`, `Counter Narrative` → 0; `Offensive`, `Hateful` → 1). T2 ⊂ T1.
>
> **Predictor.** Leave-one-out over the pooled train+val set. For query *i*, retrieve k=20 nearest
> neighbours (excluding *i*), score
> `s_i = Σ_j w_ij · t_j / Σ_j w_ij`, where `t_j` is the neighbour's contestedness indicator and
> `w_ij` is the similarity. **Similarity-weighted mean, deliberately continuous and
> non-saturating** — per the P2 forensic transferable rule, a bounded neighbour *count* is
> degenerate by construction and must not be used as a selection score.
> Neighbour ordering ties broken lexicographically by `video_id`.
>
> **Endpoints.**
> - **E1** — AUROC of `s_i` predicting T1, per language.
> - **E2** — AUROC of `s_i` predicting T2, per language (secondary).
> - **E3 (the discriminator)** — AUROC of a **label-only hardness baseline** predicting T1:
>   `h_i = 1 − |p_i − 0.5| · 2` where `p_i` is the similarity-weighted fraction of harmful-labelled
>   neighbours. This is what a system with **no access to votes** already knows.
>   The reported increment is `Δ = AUROC(s) − AUROC(h)`.
> - **Null control** — repeat E1 with the contestedness targets randomly permuted (seed 20260909);
>   AUROC must land within [0.45, 0.55] or the pilot is void.
> - Uncertainty: 2000-resample bootstrap over queries, same draws for `s` and `h` (paired).
>
> **Frozen decision rule.**
> - **GO** — in **both** languages: `AUROC(s) on T1 ≥ 0.60` with bootstrap 95 % LB `> 0.55`,
>   **and** `Δ ≥ +0.03`.
> - **AMBIGUOUS** — the AUROC bar is met in both languages but `Δ < +0.03`; or the full GO
>   condition holds in exactly one language.
> - **NO-GO** — `AUROC(s) < 0.60` in either language, or `Δ ≤ 0` in both.
>
> **Reading.** NO-GO means the vote data, though real, is not *retrievable* — the disagreement
> family loses its mechanism and must be closed. AMBIGUOUS with high AUROC but low Δ means
> contestedness is real but already implied by label geometry, i.e. the votes add no new
> information and the idea reduces to hardness-aware retrieval, which is not novel.

---

## 2. Data as loaded

| | EN (MHC) | ZH (MHC_zh) |
|---|---|---|
| n (train + val) | **629** (549 + 80) | **657** (579 + 78) |
| vote join to upstream TSV | **100 %** (0 failures) | **100 %** (0 failures) |
| vote-count histogram | 2:526, 3:102, 4:1 | 2:485, 3:165, 4:7 |
| `No` → `Normal` alias applied | 0× | **1×** |
| base rate T1 `non_unanimous` | 0.2114 (133/629) | 0.3166 (208/657) |
| base rate T2 `binary_split` | 0.1002 (63/629) | 0.1750 (115/657) |
| base rate harmful label | 0.3068 | 0.3166 |
| derived-majority vs cached binary label | 1.0000 | 0.9985 (1 item) |
| negative neighbour weights among the k=20 selected | 0 | 0 |
| neighbour similarity range (dot of the two l2 blocks) | [0.445, 1.880] | [1.005, 2.000] |

Counts match the freeze exactly (629 / 657). T2 ⊂ T1 verified programmatically (HALT otherwise).
No self-neighbour survived the LOO exclusion (checked, HALT otherwise). All neighbour weights
positive, so the weighted mean is well-behaved and never divides by a near-zero denominator.

## 3. Endpoints

| endpoint | EN | ZH |
|---|---|---|
| **E1** AUROC(`s`) on T1 | **0.6855** | **0.7089** |
| E1 bootstrap 95 % CI | [0.6357, 0.7314] | [0.6671, 0.7487] |
| **E2** AUROC(`s`) on T2 | 0.6556 | 0.6849 |
| E2 bootstrap 95 % CI | [0.5806, 0.7260] | [0.6335, 0.7352] |
| **E3** AUROC(`h`) on T1 (label-only hardness) | 0.6272 | 0.6180 |
| E3 bootstrap 95 % CI | [0.5734, 0.6768] | [0.5790, 0.6595] |
| **Δ = E1 − E3** (paired) | **+0.0583** | **+0.0909** |
| Δ paired bootstrap 95 % CI | **[−0.0096, +0.1232]** | [+0.0416, +0.1400] |
| Δ fraction of bootstrap draws > 0 | 0.957 | 0.9995 |

Auxiliary, non-gating (reported so the E2 reading is not hidden behind an interpretation):
AUROC(`s` built from T1) on T2 = 0.6682 (EN) / 0.6426 (ZH); AUROC(`h`) on T2 = 0.6543 (EN) /
0.6249 (ZH). Bootstrap: 2000 resamples, seed 20260908, 0 degenerate draws skipped in either
language; `s` and `h` share the identical resample index draws, so Δ is genuinely paired.

## 4. Null control

Frozen: permute the contestedness targets (seed 20260909), recompute `s` from the permuted
neighbour indicators, re-evaluate E1. Must land in [0.45, 0.55].

| | EN | ZH | required |
|---|---|---|---|
| null AUROC(`s`) on permuted T1 | **0.4933** | **0.5413** | [0.45, 0.55] |
| in range | yes | yes | — |

**Null control PASSES in both languages.** The pilot is not void.

## 5. Verdict against the frozen rule

| clause | EN | ZH |
|---|---|---|
| AUROC(`s`) on T1 ≥ 0.60 | 0.6855 — yes | 0.7089 — yes |
| bootstrap 95 % LB > 0.55 | 0.6357 — yes | 0.6671 — yes |
| Δ ≥ +0.03 | +0.0583 — yes | +0.0909 — yes |
| full GO condition | **met** | **met** |

The GO clause requires all three in **both** languages. It holds.

# VERDICT: **GO**

Raw clause flags recorded in `pilot_a.json:verdict_block`:
`go_condition_triggered=true`, `nogo_condition_triggered=false`,
`ambiguous_condition_triggered=false` — the three frozen clauses did not co-fire, so no
precedence decision was needed.

**Reading, per the freeze.** Contestedness *is* retrievable from CLIP neighbourhood geometry
(AUROC ≈ 0.69–0.71), and it is **not** fully explained by the label-only hardness baseline
(Δ = +0.058 EN, +0.091 ZH). The vote-based family retains a mechanism. Note what this does *not*
say: it is a necessary-condition gate, not an accuracy claim, and the freeze says so explicitly.

---

## 6. Caveats — stated against the result, not for it

1. **The EN Δ is not significantly greater than zero.** Δ_EN = +0.0583 clears the frozen
   +0.03 bar on the point estimate, but its paired bootstrap 95 % CI is **[−0.0096, +0.1232]**,
   i.e. it *contains zero*; 4.3 % of resamples put it at or below zero. The frozen rule was
   written on the point estimate, so the GO stands as written — but half of the GO verdict rests
   on an increment that a stricter rule (CI excluding 0) would not have cleared. ZH is much
   safer (CI [+0.042, +0.140], 99.95 % of draws positive). If the follow-up experiment is
   English-only, this GO is materially weaker than the headline number suggests.
2. **The ZH null sits at the edge of the acceptance window.** 0.5413 against a ceiling of 0.55.
   The synthetic smoke run (random features, n = 629/657) produced nulls of 0.471 and 0.528,
   so the sampling SD of the null at this n is roughly 0.02–0.03 — the frozen [0.45, 0.55]
   window is only about ±1.7 SE wide. A single permutation draw is a weak control at this
   sample size; it detects gross leakage, not subtle bias. Reporting the mean over many
   permutations would have been the stronger design, and the freeze did not ask for it.
3. **`h` may be an unfairly weak discriminator.** `h` is built from the *same* neighbourhood and
   the *same* weights as `s`, so Δ isolates "votes vs binary labels", which is what the freeze
   wanted. But it is not "everything a vote-free system knows": a system with a trained
   classifier could produce a much better-calibrated hardness estimate than a 20-NN weighted
   label fraction, and against *that* baseline Δ could shrink or vanish. The freeze's own
   "Reading" section flags exactly this failure mode (contestedness real but already implied);
   the pilot rules it out only for the crude label-geometry baseline it specified.
4. **T1 is dominated by 2-annotator items.** 526/629 EN and 485/657 ZH items have only two raw
   votes, so T1 "non-unanimous" reduces to "the two annotators differed" for ~80 % of EN and
   ~74 % of ZH. Contestedness is measured at very low annotator resolution, and T1 is partly a
   proxy for single-annotator noise rather than genuine item ambiguity.
5. **T2 is thin.** Only 63 EN / 115 ZH positives. E2's EN CI ([0.581, 0.726]) is wide enough
   that the binary-split target is not independently established in English.
6. **Leave-one-out over a pooled train+val set overstates neighbourhood density** relative to a
   deployment setting where the memory is train-only and the query is unseen. Every item here
   sees 628/656 candidate neighbours including its own split-mates.
7. **Retrievability ≠ usability.** AUROC ≈ 0.70 on a 21 %/32 % base rate is a weak ranker in
   absolute terms. Nothing here shows that routing on `s` improves any detection metric; the
   gate only says the mechanism is not empty.
8. **The 8-frame mean-pooled CLIP key is the same key P-B shows contains degenerate
   bit-identical vectors** (see `PILOT_B_RESULT.md` §7). MHC-EN has zero such vectors and MHC-ZH
   has one pair, so P-A is essentially unaffected — but the two pilots share a feature cache
   whose integrity is not perfect.
9. **`Δ` uses the same 2000 resample draws for `s` and `h`, which is correct for pairing but
   means the two marginal CIs are not independent** — do not read "E1 CI and E3 CI overlap" as
   evidence about Δ. The Δ CI is the only valid statement about the increment.

## 7. Deviations from the freeze (logged, not hidden)

| # | deviation | why / impact |
|---|---|---|
| A-D1 | **Bootstrap seed not specified in the freeze.** Fixed to `20260908` in the source before the real run; only the *null* seed (20260909) was frozen in the document. | Chosen blind, committed in code before any real number existed. No endpoint depends on the choice beyond MC noise. |
| A-D2 | **E2 predictor interpretation.** The freeze says "AUROC of `s_i` predicting T2" without stating whether `t_j` is the T1 or the T2 indicator. Implemented as the matched form (`t_j` = T2). | The unmatched variant is reported as `aux_auroc_sT1_on_T2` so both readings are visible. E2 is secondary and non-gating either way. |
| A-D3 | **Verdict precedence.** The three frozen clauses can co-fire (e.g. full GO in one language, AUROC < 0.60 in the other). Precedence GO > NO-GO > AMBIGUOUS was coded before the run. | Did **not** bind: only the GO clause fired. All three raw flags are in the JSON so the reader can re-adjudicate without re-running. |
| A-D4 | **`No` → `Normal`** alias applied to the single stray Chinese vote token, as instructed. Applied 1×. | Affects at most one ZH item's T1/T2. |
| A-D5 | **Derived-majority vs cached label mismatch on 1 ZH item** (agreement 0.9985). My diagnostic recomputes the majority with the tie rule `harm > half`, which differs from the official `Majority_Voting` column on ties. | Diagnostic only — **no endpoint uses the majority**; T1/T2 come straight from the raw vote list and the binary label comes from the frozen cache. |
| A-D6 | **`logging/runs/pilot_a/run.pid` was not written.** The launcher's `mkdir && conda && setsid … &` chain backgrounded as one unit, so the `echo $! > run.pid` raced the `mkdir`. | Bookkeeping only. `run.log` is complete and the real run finished (exit 0, JSON written). The single-submission red line is intact — P-A's real path was executed exactly once. |

## 8. Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate HateVideo
python idea-stage/pilot_a_disagreement_retrievability.py --smoke synthetic   # random features
python idea-stage/pilot_a_disagreement_retrievability.py --smoke permuted    # label-permuted, seed 999
python idea-stage/pilot_a_disagreement_retrievability.py --out idea-stage/pilot_a.json
```

Pre-run smokes (both executed before the real run, neither reveals a real endpoint):

- **synthetic** (Gaussian features, random targets): E1 = 0.502 / 0.458, E3 = 0.484 / 0.479,
  null = 0.471 / 0.528 — the pipeline is unbiased on noise.
- **permuted** (real features, targets permuted with seed **999**, deliberately *not* the frozen
  null seed): E1 = 0.538 / 0.492, E3 = 0.532 / 0.472, Δ = +0.006 / +0.020 — destroying the target
  destroys the signal, and Δ collapses toward zero.
