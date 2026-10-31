# Diagnostics for the two goals (2026-10-31)

Context: the user's two-goal autonomous iteration (2026-10-31; see `research-wiki/STATUS.md`).
- Goal 1: about 4-5 calls with AP and ROC at most .01 lower than fixed 8, and within not lower.
- Goal 2: the project's own backbone reaches r5's AP and ROC.

All analyses below are rule-10 developmental evidence. They read test predictions and test GT, train nothing, and
select nothing for a method. Numbers are 3-seed means over the test-selected trial of each seed. They come from the
shared core evaluator (`hc.frame_metrics`). Outputs are under `runs/20261031_two_goal_diagnostics/`.

## Scripts

| Script | Question | Output |
|---|---|---|
| `call_allocation.py` | Can allocating calls across videos, by P(G) or other rules, reach 4-5 calls? Includes an oracle label split. | `call_allocation/<corpus>/partial_<host>.json` |
| `backbone_gap.py` | Is the inside-outside (IO) gap to r5 at the video level or in the within-video shape? | `backbone_gap/<corpus>.json` |
| `mix_budgets.py` | Which videos need calls 5-8: positives or negatives? Label-split oracle. | `mix_budgets/*.json` |
| `policy_lab.py` | Does question order matter? EIG vs expected squared-error reduction (EER, `cpolicy._expected_error_reduction`), with copy likelihood, fixed-k and stop rules. `--answer-model val3` uses the validation-fitted three-state answer model (`src/qtl/answer_calib.py`). | `policy_lab/<corpus>/<tag>/summary_<host>.json` |
| `order_anatomy.py` | Which nodes each order asks, and where within ROC is gained or lost (by GT harm fraction tercile). | `policy_lab/order_anatomy.json` |
| `rule_scan.py` | Every stopping rule of `qtl.stop_rules` (plus rules with the selection value as a component) at several target mean calls, on the stored policy runs; per trial. | `policy_lab/<corpus>/rule_scan*.json` |
| `headroom.py` | Where r5's remaining pooled error is: oracle edits of r5's test scores (negatives zeroed, perfect within-video ranking, within-video shape removed) and the level / hate-fraction correlation. | `headroom/<corpus>.json` |

## Findings

1. **Allocating calls alone is not enough** (`call_allocation.py`).
   - HateClipSeg: 87% of the test videos are positive, so giving negatives fewer calls cannot bring the mean to 4-5.
   - HateMM: even with true labels (positives 8 calls, negatives 2, mean 4.3), ROC is .028 lower than fixed 8.
2. **Negatives need calls for pooled AP/ROC; positives need calls only for within** (`mix_budgets.py`, label-split
   oracle).
   - HateMM, r5 with copy likelihood: positives 2 calls and negatives 8 calls give .688 / .890 / .771. That is
     better than all-8 (.686 / .882 / .766).
   - HateClipSeg: cutting positives to 4 calls costs within .010.
3. **The IO gap to r5 is the within-video shape on HateMM and video-level discrimination on HateClipSeg**
   (`backbone_gap.py`).
   - HateMM: with r5's within-video shape, IO recovers within and .009 AP.
   - HateClipSeg: video ROC is .795 vs .865.
4. **Question order and stopping on the r5 backbone with copy likelihood** (`policy_lab.py`). Differences are
   relative to EIG fixed 8 (AP / ROC / within).

   | | HateMM | HateClipSeg |
   |---|---|---|
   | EIG fixed 4 | -.016 / -.012 / -.010 | -.007 / -.002 / -.010 |
   | EIG + qmixSG_rt10, mean ~5 calls | **+.006 / +.002 / +.001** (5.02) | -.006 / -.003 / -.014 (4.72) |
   | EER fixed 4 | -.016 / -.013 / -.021 | **-.009 / -.005 / .000** |
   | EER + qmixSG_rt10, mean ~5 calls | -.004 / +.001 / -.009 (4.99) | -.013 / -.013 / -.009 (5.11) |

   - Stopping on the EER value itself under-asks negatives: ROC is -.035 on HateClipSeg.
   - On the IO backbone neither order reaches the goal on HateClipSeg.
   - No single policy passes on both corpora yet:
     - HateMM is met by the existing qmixSG_rt10 rule on r5 at about 5 calls.
     - HateClipSeg within needs more within-efficient questions; EER at 4 calls gives that.
5. **A validation-fitted answer model does not fix it** (`policy_lab.py --answer-model val3`, tag `r5_val3`).
   - HateMM: within no longer falls with more calls (fixed 16 .771 vs .753 with the trained model), but AP is lower
     at every budget (fixed 8 .673 vs .686).
   - HateClipSeg: within at 4-5 calls is not better (.537-.551).
6. **Where within comes from** (`order_anatomy.py`, test).
   - HateMM: within at 0 calls is .770, at 8 calls .772. Calls move within between videos: the lowest GT-harm-fraction
     tercile gains (.695 to .797 with EIG), the highest loses (.907 to .807).
   - HateClipSeg: within needs calls (.539 at 0, .559 at 8; EER reaches .559 at 4 by asking shorter nodes, 9 s vs 14 s).
7. **Stopping rules at 4-5 calls** (`rule_scan.py`, EIG order, r5 trials; then 6 more r5 trials per corpus, tag
   `r5_more`; DeHate r5; inside-outside trials).
   - Rules on video-level uncertainty or the last change (qmixSG, qmixS, qmixG, qmixD) keep AP / ROC but lose
     .005-.008 within on HateClipSeg; rules on the information left (eigsum, eigtop3) give positives 7-9 calls and
     negatives 1-3 and lose .012-.025 AP / ROC; qmixSG_rt10 at 5 calls loses .014 within on HateClipSeg.
   - Combining the last change with the forward value passes: qmixE3 (top-3 EIG sum) and qmixE1 (the best next EIG,
     no constant). Details and the per-setting table: `experiments/20261031_fewer_calls/README.md`.
   - The test choice among about 190 rules is selection on test; the extra trials, DeHate and inside-outside settings
     did not take part in the choice of the family.
8. **Where r5's remaining pooled error is** (`headroom.py`, r5 test-selected trials, fixed 8, 3-seed means, AP / ROC;
   output `runs/20261031_two_goal_diagnostics/headroom/{hatemm,hateclipseg}.json`).

   | Edit of r5's test scores | HateClipSeg | HateMM |
   |---|---|---|
   | none (official) | .686 / .674 | .691 / .885 |
   | each second = its video's mean score (no within-video shape) | .687 / .680 | .654 / .878 |
   | perfect within-video ranking, same scores per video | .806 / .796 | .749 / .902 |
   | negative videos set to 0 | .705 / .717 | .751 / .929 |

   - HateClipSeg: r5's within-video shape adds nothing to pooled AP / ROC (removing it gives .687 / .680). Pooled
     metrics there depend only on the per-video level. Test seconds: 9900 hate, 6626 non-hate in positive videos,
     2313 in negative videos.
   - The per-video level should rank positive videos by their hate fraction. The Spearman correlation between a
     positive video's mean score and its GT hate fraction is .35 (HateClipSeg) and .36 (HateMM).
   - Consequence for goal 2: a backbone that improves within-video ranking (the scale-aware candidate: HateClipSeg
     within +.019) does not move HateClipSeg pooled metrics unless the within gain is large; the lever on both corpora
     is the per-video level (hate fraction among positives, rejection of negatives).
   - A subagent CPU analysis (scripts not committed, so not used as evidence in the README tables) reported in
     addition: r5's cross-modal attention is close to uniform on test videos (entropy / log T .98-1.00), linear
     probes of r5's pooled representation predict a positive video's hate fraction no better than r5's own level
     (Spearman .2-.46), and a video head that also reads the per-second logits changes pooled AP / ROC by <= .001.
