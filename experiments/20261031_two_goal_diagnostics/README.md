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
| `policy_lab.py` | Does question order matter? EIG vs expected squared-error reduction (EER, `cpolicy._expected_error_reduction`), with copy likelihood, fixed-k and stop rules. | `policy_lab/<corpus>/<r5\|io>/summary_<host>.json` |

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
