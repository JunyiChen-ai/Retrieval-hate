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
| `chain_lever.py` | Would a per-video chain transition help? r5 checkpoints at 0 calls with A fitted to each positive video's GT (ceiling) or predicted from content by ridge (fitted on validation positives). | `chain_lever/<corpus>.json` |
| `feature_ceiling.py` | How much per-second hate information do the fixed features carry? Per-second classifiers fitted on validation per-second GT, test metrics. | `feature_ceiling/<corpus>.json` |
| `weak_probe.py` | A linear per-second classifier from training video labels only, against r5 at 0 calls. | `weak_probe/<corpus>.json` |
| `answer_quality.py` | How much localisation the VLM tree answers carry when every node is answered (node AUC, all-answers per-second score). | `answer_quality/<corpus>.json` |
| `answer_probe.py` | A ridge per-second model fitted to the training answers alone (no GT), and (`--combine-r5`) combined with r5's 0-call scores. | `answer_probe/<corpus>{,_combined}.json` |

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
9. **What limits r5: the weak signal, not the features** (`chain_lever.py`, `feature_ceiling.py`, `weak_probe.py`;
   test, AP / ROC / within; validation per-second GT used only to fit diagnostic probes, no gradient into any method).

   | HateClipSeg | AP / ROC / within | level / hate-fraction Spearman |
   |---|---|---|
   | r5, 0 calls / 8 calls | .658 / .647 / .539, .687 / .674 / .560 | .30, .35 |
   | linear, training video labels only (C by validation video AP+ROC) | .574 / .536 / .514 | -.04 |
   | linear, validation per-second GT (63 videos), x_t | .688 / .680 / .563 | .51 |
   | same, x_t + video mean | .700 / .682 / .567 | .51 |
   | same, x_t + video mean + local means (3, 9, 27 s) | .701 / .696 / .590 | .50 |
   | MLP, validation per-second GT, x_t + video mean + local | .661 / .647 / .556 | .39 |

   - HateClipSeg: a linear read-out of the fixed features with per-second labels from only 63 videos matches or beats
     r5 at 8 calls; video labels alone give nothing (87% positive videos). r5's content model is trained by the video
     label and the VLM answers on all tree nodes, and recovers part of it (0 calls .658 / .647). The features carry the
     information; the weak signal is what limits r5. Local context adds within +.024 and ROC +.014 under supervision.
   - HateMM: 109 validation videos are too few for a supervised probe (best .480 / .743 / .651, below r5 at 0 calls
     .623 / .819 / .768); the weak linear probe gives .467 / .729 / .619. No ceiling can be read on HateMM.
   - Per-video chain transition: GT-fitted per video (ceiling) .705 / .684 (HateClipSeg) and .717 / .853 (HateMM) at 0
     calls vs learned .658 / .647 and .623 / .819; predicted from r5's representation or outputs by ridge,
     .657-.659 / .649-.652 and .611-.626 / .818-.825 (Spearman with the GT-fitted stay probabilities .05-.61). Not a
     usable lever with r5's representation.
10. **The training answers do not teach the features more on HateClipSeg; on HateMM a linear path helps the video
    level** (`answer_quality.py`, `answer_probe.py`; test, AP / ROC / within, 0 calls, no GT in training).

    | | HateClipSeg | HateMM |
    |---|---|---|
    | r5 (3-seed mean) | .658 / .647 / .539 | .623 / .819 / .768 |
    | every node answered, shortest node's p_yes per second (not reachable at test) | .682 / .656 / .615 | .589 / .789 / .652 |
    | ridge on x_t + video mean + local means, fitted to the training answers | .641 / .623 / .542 | .646 / .818 / .663 |
    | that ridge and r5, mean of standardised logits | .662 / .648 / .541 | .656 / .833 / .724 |

    - Node AUC of p_yes for "the node contains hate" (test, nodes of positive videos): .68 HateClipSeg, .75 HateMM;
      p_yes does not follow the node's hate share (Spearman -.10 / -.19 over hate nodes).
    - HateClipSeg: the answers localise within videos (.615) better than r5's content model (.539), but the
      features predict the answers poorly (validation Pearson .27-.28), so a model fitted to them stays below r5 and
      adds nothing to r5 (+.004 / +.001). With finding 9: the per-second information is in the features (dense labels
      reach .70 / .70), but neither the video labels nor the VLM answers carry it to a feature-based model.
    - HateMM: a linear read-out of the raw features fitted to the answers ranks videos better than r5 (AP +.023 alone,
      +.033 / +.014 combined with r5, within -.044 combined).
    - Consequence for goal 2: on HateClipSeg no backbone trained on the present signals is expected to beat r5 by
      more than noise; on HateMM there is room in the video level.
