# 20261110_answer_likelihood: does a different answer likelihood close the gap to the plain fusion?

User 2026-11-10 (after `experiments/20261109_tree_within/README.md` section 12): change the answer likelihood. Section
12 item 2 found that revision 6 uses a video-level answer worse than a plain logit average of its own 0-call scores
and the raw P(Yes). This round asks why, and whether another answer likelihood fixes it. Everything here is on the
revision-6 test-selected checkpoints (nothing retrained; developmental test analysis, rule 10); host uoa-lab3
(sc474398), CPU; outputs `runs/20261110_answer_likelihood/` (copied to uoa-lab2). Method unchanged (revision 6).
AP / ROC / within, 3-seed means, order HateMM / HCS / DeHate; scores evaluated at 6 decimals.

**Result: no answer likelihood tried improves revision 6 at 8 calls; the 1-call gap comes from the backbone's
over-confident video prior, not from the answer table. Stage B (retraining) was not started.**

## 1. What the revision-6 table does to the answer

Training answers on the logit scale (inline check, numbers from `data/vlm_tree/<C>/soft_both_p1*`): the 8 quantile
edges of P(Yes) are .000 / .001 / .002 / .009 / .042 / .349 / .915 on HateMM, so five of the eight levels lie below
P(Yes) = .05 and the top level spans logit 2.4 to 9; 76% of the positive training roots fall into it (HCS 44%, DeHate
46%). The table of the hateful state is fitted on the positive roots only (297 / 219 / 1484 answers), free per level.
HCS has 32 negative training videos; its fitted log-likelihood ratio is not monotone in the level
(-2.7, -1.4, -.3, -.7, .4, .3, 1.3, 2.2).

## 2. Diagnostic: the video prior is over-confident on held-out videos

`diag_video_evidence.py` (`video_evidence/<corpus>.json`): per video, g = the backbone's video logit, llr = the
revision-6 table's log-likelihood ratio of the root answer, x = logit of the raw root answer. Video-level ROC-AUC
and the coefficients of a logistic regression of the video label on (g, llr); a coefficient of 1 on both would mean
the model's sum g + llr is calibrated.

| | split | AUC g | AUC llr | AUC x | AUC g + llr | AUC g + x | coefficient g | coefficient llr |
|---|---|---|---|---|---|---|---|---|
| HateMM | train | .950 | .860 | .912 | .967 | .964 | 2.09 | .63 |
| | val | .878 | .817 | .876 | .907 | .915 | .65 | .51 |
| | test | .842 | .908 | .924 | .909 | .925 | .29 | .80 |
| HCS | train | .994 | .793 | .783 | .988 | .946 | 13.0 | 3.47 |
| | val | .705 | .797 | .839 | .807 | .853 | .29 | .70 |
| | test | .692 | .691 | .774 | .707 | .761 | .26 | .59 |
| DeHate | train | .930 | .619 | .646 | .924 | .838 | 1.39 | .09 |
| | val | .710 | .694 | .741 | .743 | .785 | .27 | 1.08 |
| | test | .693 | .692 | .760 | .722 | .776 | .24 | 1.08 |

1. The video logit separates the training videos almost perfectly (.93-.99) and the held-out ones much worse
   (.69-.84); on test its calibrated weight is .24-.29, i.e. the model trusts it about four times too much. The root
   answer's evidence is then under-weighted: on HateMM the root answer alone (.908) classifies test videos as well
   as prior plus answer (.909).
2. The 8-level free table loses ordering information of the raw answer: video AUC of the root evidence .908 / .691 /
   .692 against the raw answer's .924 / .774 / .760. (HCS test has 10 negative videos; its video AUCs are coarse.)

## 3. Stage A: candidate answer likelihoods on the revision-6 checkpoints

`stage_a_replay.py` (`stage_a/<corpus>/summary.json`), tables fitted on the same training answers with video labels
only (`src/qtl/answer_density.py`), cached answers re-binned. q8 = revision 6 (reproduces the checkpoint's table
exactly); qN = N train-quantile levels, free table; uN = N uniform bins on logit P(Yes); `tilt1` = hateful table =
no-hate table tilted by beta x (log-likelihood ratio linear in logit P(Yes); fitted slope .57 / .18 / .20 per logit);
`cal` = slope 1 per logit; `smooth4` = both states a degree-4 exponential family. root = tree inference after the
root's answer; u8 = after the 8 equal windows; eig8 = the method (EIG tree, 8 calls).

| | | root (1 call) | 8 equal windows | EIG tree, 8 calls |
|---|---|---|---|---|
| HateMM | q8 (revision 6) | .640 / .856 / .772 | .690 / .879 / .804 | .702 / .889 / .783 |
| | q32 | .651 / .858 / .772 | .688 / .882 / .809 | .693 / .889 / .800 |
| | u32 free | .655 / .860 / .772 | .687 / .882 / .804 | .696 / .890 / .791 |
| | u32 tilt1 | .657 / .866 / .772 | .684 / .880 / .804 | .701 / .883 / .767 |
| | u32 cal | .671 / .881 / .768 | .677 / .851 / .766 | .637 / .840 / .708 |
| | u32 smooth4 | .661 / .866 / .772 | .688 / .884 / .807 | .695 / .889 / .787 |
| HCS | q8 (revision 6) | .660 / .658 / .556 | .653 / .650 / .561 | .691 / .680 / .563 |
| | q32 | .652 / .657 / .556 | .659 / .659 / .565 | .689 / .678 / .558 |
| | u32 free | .635 / .644 / .556 | .651 / .651 / .558 | .693 / .680 / .556 |
| | u32 tilt1 | .665 / .658 / .556 | .664 / .664 / .562 | .677 / .669 / .568 |
| | u32 cal | .691 / .656 / .556 | .668 / .611 / .543 | .642 / .607 / .530 |
| | u32 smooth4 | .646 / .649 / .556 | .655 / .656 / .562 | .692 / .685 / .566 |
| DeHate | q8 (revision 6) | .242 / .733 / .661 | .279 / .755 / .686 | .279 / .770 / .681 |
| | q32 | .249 / .735 / .662 | .270 / .754 / .686 | .274 / .769 / .684 |
| | u32 free | .255 / .736 / .662 | .274 / .756 / .687 | .275 / .771 / .685 |
| | u32 tilt1 | .244 / .735 / .662 | .274 / .756 / .685 | .264 / .763 / .682 |
| | u32 cal | .265 / .768 / .660 | .198 / .675 / .636 | .147 / .637 / .585 |
| | u32 smooth4 | .249 / .735 / .662 | .271 / .753 / .688 | .274 / .769 / .684 |

(q16, u16 free, u16 tilt1, u32 tilt2, u32 gauss in `summary.json`: all between these rows.)

- At 8 calls with the EIG tree no candidate is ahead of revision 6 by more than .002 AP or .005 ROC on any corpus
  (the linear tilt -.001 / -.014 / -.015 AP). Finer levels raise HateMM within by
  .004-.017 at an AP cost of .006-.009.
- The tilt restores the raw answer's ordering at the root (video AUC of the root evidence .921 / .771 against .908 /
  .691; inline check on the Stage A runs) but its pooled gain at 1 call is small (HateMM +.017 / +.010, HCS +.005 /
  .000, DeHate +.002 / +.002).
- Slope 1 per logit (the raw answer as evidence, what the plain fusion does) gains at the root on every corpus (AP
  +.031 / +.031 / +.023, ROC +.025 / -.002 / +.035) and breaks the model beyond the root (EIG 8 calls -.065 / -.049 /
  -.132 AP): a Yes on a sub-window is then trusted far more than the data support (section 11 of the tree-within
  README). The root needs more weight because the prior is over-confident (section 2), not because the table is
  coarse.

## 4. Tempering the prior at test time (diagnostic, not a method)

`stage_a_replay.py --g-scales 1 0.5 0.3 --loc-scales 1 0.5 0` (`summary_scales.json`, `summary_budgets.json`): the
video logit g and the per-second logits / node potentials multiplied by a constant, EIG tree at 1 / 2 / 4 / 8 calls.

| | | 1 call | 2 calls | 4 calls | 8 calls |
|---|---|---|---|---|---|
| HateMM | revision 6 | .647 / .851 / .771 | .667 / .865 / .778 | .687 / .881 / .780 | .702 / .889 / .783 |
| | both priors x .5 | .661 / .865 / .784 | .662 / .875 / .791 | .678 / .884 / .796 | .698 / .888 / .793 |
| | tilt1, g x .3, location x .5 | .669 / .872 / .783 | .700 / .885 / .766 | .705 / .888 / .778 | .705 / .885 / .776 |
| HCS | revision 6 | .661 / .657 / .556 | .658 / .659 / .561 | .676 / .673 / .562 | .691 / .680 / .563 |
| | both priors x .5 | .660 / .656 / .566 | .671 / .661 / .567 | .680 / .670 / .566 | .687 / .677 / .568 |
| | tilt1, g x .3, location x .5 | .688 / .666 / .575 | .690 / .667 / .573 | .682 / .665 / .571 | .683 / .663 / .576 |
| DeHate | revision 6 | .242 / .733 / .662 | .251 / .744 / .662 | .256 / .757 / .669 | .279 / .770 / .681 |
| | both priors x .5 | .251 / .739 / .678 | .259 / .752 / .671 | .272 / .764 / .683 | .289 / .775 / .695 |
| | tilt1, g x .3, location x .5 | .253 / .733 / .679 | .244 / .745 / .670 | .260 / .755 / .680 | .261 / .765 / .690 |

- Halving both priors with the revision-6 table: within rises on every corpus at every budget (+.004 to +.016);
  DeHate also gains AP / ROC at every budget (8 calls +.010 / +.005 / +.014); HateMM and HCS AP / ROC stay within
  .010 of revision 6 except 1 call on HateMM (+.014 / +.014) and 2 calls on HCS (+.013 AP).
- The tilt with a strongly tempered video logit brings HateMM at 2 calls to revision 6's 8-call AP (.700 / .885
  against .702 / .889) and lifts HCS at 1-2 calls by .03 AP, but is below revision 6 on DeHate at 2 and 8 calls (8
  calls -.018 AP).
  No single setting is ahead on all three corpora; choosing one per corpus is not allowed (rule 13).
- The scales here are set by hand. Location prior x .5 with the revision-6 video logit on DeHate (the entry that was
  "running" in the tree-within README section 6): 8 calls +.004 / +.006 / +.016.

## 5. Status

- **Answer likelihood: rejected at Stage A.** Eleven tables (levels 8-32, quantile or logit grid, free / tilted /
  smooth); none improves the method's operating point, and the comparison with the naive-question control at 8 calls
  is unchanged (tree-within README section 12). `src/qtl/answer_density.py` stays as the code of these candidates; no
  training entry uses it.
- **Cause of the 1-call gap identified**: the backbone's video logit is over-confident on held-out videos (calibrated
  weight .24-.29), with the per-second prior (section 12 item 4) the second over-confident part. Tempering both at
  test time gives within +.004 to +.016 on all corpora and AP / ROC gains on DeHate only.
- Open, for the user: whether to make the prior's weight part of the method (estimated on held-out videos; rule 3
  forbids calibration as post-processing, so this needs a ruling) and retrain, or to keep revision 6 and narrow the
  claims.
