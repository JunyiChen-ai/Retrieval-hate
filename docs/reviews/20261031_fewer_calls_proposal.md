# Fewer VLM calls, stopping rule qmixE1: independent rule-4 proposal review

Date: 2026-10-31. Reviewer: independent agent (Claude Fable 5.1), the one rule-4 review of
`experiments/20261031_fewer_calls/README.md`. Read: that README and `evaluate_stop.py`, `RESEARCH_ITERATION_RULES.md`
rule 4, `src/qtl/stop_rules.py` (`rule_values`, `QMIX`, `add_quantile_rules`), `src/qtl/policy.py` (`calibrate`,
`stop_calls`), `src/qtl/cpolicy.py` (`run_batch`, how the EIG is recorded), section 17.4 of
`experiments/20260925_query_paradigm/README.md`, item 7 of `experiments/20261031_two_goal_diagnostics/README.md`,
`docs/20260908_adaptive_query_survey.md`, the stored scan files under `runs/20261031_two_goal_diagnostics/policy_lab/`,
and the papers listed in section 3. Rule 4 allows a STOP in four cases only and forbids stopping on
pre-implementation reasoning (identifiability, degeneration, shortcut). This review follows that. The review did not
run experiments and edited no other file.

**Verdict: GO.** No hateful video detection or localisation paper or system found by the searches below uses an
adaptive stopping rule for VLM / LLM queries of any kind, and none uses this rule (stop when both the change caused
by the last answer and the expected information gain of the best next question are small, each normalised to its
label-free validation quantile, one threshold set by a validation call budget). The other three STOP conditions do
not apply. GO means implement and code-review; it does not assert that the rule will meet the goal-1 numbers.

Design update received during the review (coordinator, 2026-10-31, after the rule-6 code review
`docs/reviews/20261031_fewer_calls_code.md` point A found a 32-call maximum binding on HateMM and moving the
threshold): the maximum number of calls is now the fixed budget the rule is compared with (8). The rule only decides
when to stop earlier than that budget; no video gets more calls than in the fixed-8 method. This review covers that
design; the cap-8 numbers are verified in section 5.

Constraint notes for the user (section 2): the rule has no floor or schedule constant of its own and adds no new cap
constant, with two things to state openly: (i) the first call is always made because the backward statistic is
undefined before any answer, which is structurally a floor of one call and was already accepted for qmixSG in 17.4
item 8; (ii) the maximum of 8 is the comparison method's own budget, so the method is "fixed budget 8 with early
stopping", and the maximum does change the threshold and the numbers (the README keeps the 16-call results next to
the 8-call ones for that reason). The operating point (validation mean of 5 calls) is the goal-1 budget, not a rule
constant.

## 1. Object under review

For one video, before call k + 1 (k = 0, 1, ...), two statistics are read from the policy run
(`stop_rules.rule_values`):

- `dlsum[k]` = sum over seconds of |logit p_t after call k - logit p_t after call k - 1| (the per-second posterior
  change the last answer caused, weighted as the pooled metrics weight seconds); +inf at k = 0.
- `eig[k]` = the expected information gain, in bits, of the question the EIG order would ask next. `cpolicy.run_batch`
  records `float(e[j])` with `j = argmax(e)` under `order="eig"` (lines 203-205 and 264), so this is exactly the
  selection value, with the copy likelihood included when `copy_mode="both"`.

`add_quantile_rules` replaces each statistic by its quantile among all finite validation (video, call) values
(`searchsorted(ref, x, side="right") / len(ref)`, so values in (0, 1]; +inf stays +inf) and sets the rule value to the
larger quantile (`QMIX["qmixE1"] = ("dlsum", "eig")`). `policy.calibrate` picks the smallest threshold c whose
validation mean number of calls is <= 5 (no labels); `policy.stop_calls` asks while the rule value >= c and applies the
same c to test. The policy is run to at most `--max-calls` = 8 calls per video on validation and test
(`evaluate_stop.py`, default 8, `fixed_k = (max_calls, target)`), so the rule chooses a stopping point inside the
fixed-8 run. Everything else (prior network, answer model, copy likelihood, tree posterior, EIG order, evaluator) is
the r5 method unchanged. Comparison points: fixed 8 calls (the same run to its end) and fixed 5 calls, both through
the unified evaluator.

## 2. Code check: floors, caps and constants

Checked against the user's rulings (2026-09-29: a minimum number of calls does not count as automatic stopping;
avoid hand-set constants; 2026-09-30 and 10-01 records in 17.4 items 8-11).

| Item | Where | Finding |
|---|---|---|
| Floor | `rule_values`: `ds = [np.inf] + ...`, `vals["dlsum"] = ds[:n]` | `dlsum` is +inf before the first call, so `max(q_dlsum, q_eig) = +inf >= c` and the first call is always made. This is a structural floor of one call. The README states it and gives the reason (no change has been observed before the first answer). Precedent: 17.4 item 8 chose qmixSG as the rule with "no compulsory question beyond the first; all rules ask the first question". Note that the forward component `eig[0]` is finite before any call, so a zero-call stop would be definable with this rule; the proposal chose not to. Not a violation under the project's precedent; the paper must say "at least one call" explicitly. |
| Second-call floor | `QMIX["qmixE1"]` uses `dlsum`, not `dlsum2` / `dlogit2` / `stab2` | None. The rules that look at the last two calls (and so always make the second call) are not used. |
| Cap | `evaluate_stop.py --max-calls 8` (default; "maximum calls = the fixed budget compared with"), `cpolicy.run_batch(..., a.max_calls, ...)`, `stop_calls` returns `len(eigs)` when the value never drops below c | Any video is cut at 8 calls (or at its number of queryable nodes, whichever is smaller), the same cut the fixed-8 comparison method makes. No constant is added beyond the comparison budget: the rule can only stop earlier. Two consequences to state in the paper. (a) The maximum is not neutral: it changes the validation pool from which the quantiles are computed and therefore c; the code review found the earlier 32-call horizon binding on HateMM, and the 8-call and 16-call results differ (section 5; HateClipSeg within +.004 at max 8 versus +.010 at max 16; HateMM within -.003 versus +.004). (b) Videos that reach the maximum are not "stopped" by the rule but by the budget; the evaluator records `videos_at_max_calls` (test) and `validation_videos_at_max_calls`, and both must be reported so the reader can see how much of the saving is the rule's. The quantity "fixed budget B with early stopping" is what is being evaluated, and B = 8 is inherited from the comparison, not chosen by the rule. |
| Schedule constant | `add_relaxed_rules` (delta = .05 / .10) | Not used by qmixE1. The current rule qmixSG_rt10 carried a hand-set delta = .10 and an implied soft maximum of 11-15 calls (17.4 item 10); qmixE1 removes both. |
| Top-k constant | `eigtop3` | Not used; qmixE1 uses the top-1 value. The README records that the switch from qmixE3 to qmixE1 was decided before the qmixE1 result was seen. |
| State-band constants | `hG >= .5 / .2`, `P(G) < .5` | Not used. |
| Operating point | `--target 5.0` | The validation mean number of calls that sets c. This is the goal-1 budget from the user's instruction (about 4-5 calls), not a rule constant; the README reports targets 4.5 and 5.5 as well, and the stored scans confirm them (section 5). The threshold c itself is data-derived. |
| Numerical constants | `_logit` clip 1e-6, `calibrate` 1e-9 | Numerical guards only. |
| Quantile pool | `add_quantile_rules`: pooled over all validation (video, call) pairs | Videos that run longer contribute more values; label-free; one threshold. The pool is built once per trial from the same validation run that sets c. |
| Causality | statistics at position k depend on scores[k], scores[k-1] and the EIG computed before call k + 1 | Later answers do not change an earlier prefix (same property the 2026-10-02 code review confirmed for qmixSG). |

Alias (now stated in README section 3): in the diagnostics scan (`experiments/20261031_two_goal_diagnostics/rule_scan.py`,
dict `NEW`) the same rule is stored as `qmixS1 = ("dlsum", "sel")`, where `sel` is the selection value of the order
in use. It equals qmixE1 under the EIG order only; under the EER order `sel` is the expected error reduction, which
is a different rule. The README's qmixE1 rows come from `qmixS1@...` rows of the EIG-order groups (section 5). The
cap-8 rows come from the stored 16-call runs truncated to 8 calls (`rule_scan.py --cap 8`, function `truncate`:
per-call lists cut at `min(cap, n)`), which is exactly what an 8-call run would have produced because the EIG order
is deterministic given the answers and the prefix does not depend on later calls.

## 3. Literature check

### 3.1 Searches run (WebSearch, 2026-10-31)

1. `"hateful video" adaptive stopping rule VLM queries number of calls localization`
2. `hateful video detection "early stopping" OR "adaptive inference" OR "early exit" frames budget HateMM`
3. `"hateful video" OR "hate speech video" localization adaptive querying multimodal LLM "expected information gain"`
4. `"hate" video detection multimodal agent iterative tool calls adaptive number of frames stop when confident 2025 2026`
5. `MAESTRO "Hear Me Out" hateful video multimodal agentic "global-local" reasoning loop iteration stop confidence DSTA`
6. `"Adaptive Global-Local Reasoning Loop" MAESTRO hateful video frames iteration`
7. `SAGE hateful video detection ACL 2026 gating experts efficiency "2026.acl-long.817"`
8. `"hateful" video VLM "query budget" OR "number of queries" OR "fewer calls" OR "adaptive budget" OR "calls per video" localization`
9. `video content moderation VLM adaptive querying budget "stopping" OR "early stop" OR "early exit" segments posterior Bayesian "information gain" harmful video localization`
10. `VideoAgent long-form video understanding LLM agent self-reflection sufficiency iterative frame selection stopping`
11. `video question answering LLM agent iterative frame retrieval stop criterion "T*" temporal search OR VideoTree OR "Frame-Voyager" OR "adaptive keyframe sampling" confidence stop`
12. `adaptive computation video recognition FrameExit AdaFrame EcoFrame early exit gate frame-by-frame policy`
13. `active learning stopping criteria survey "stabilizing predictions" OR "max confidence" OR "uncertainty" stopping method`
14. `active learning stopping criterion combining multiple criteria "expected model output change" OR "stabilizing predictions" Pullar-Strecker 2021 comparison`
15. `Zhu Wang Hovy 2008 "multi-criteria-based strategy to stop active learning" COLING max-confidence min-error overall-uncertainty combination`
16. `"stopping criterion for active learning with model stability" ACM Transactions on Intelligent Systems and Technology 2018 model change remaining unlabeled examples threshold`
17. `Laws Schütze 2008 "Stopping criteria for active learning of named entity recognition" uncertainty convergence gradient; Vlachos 2008 "A stopping criterion for active learning" Computer Speech and Language`
18. `computerized adaptive testing stopping rules "minimum information" "change in theta" OR "theta change" hybrid termination criterion variable-length CAT`
19. `Choi Grady Dodd 2011 "predicted standard error reduction" stopping rule computerized adaptive testing; Babcock Weiss 2009 termination criteria change in theta`
20. `Babcock Weiss 2009 "Termination criteria in computerized adaptive tests" variable-length CATs "change in theta" minimum information Journal of Computerized Adaptive Testing`
21. `Psychometrika stopping rule computerized adaptive testing "change in theta" "diminished returns" secondary rule premature termination Wang Weiss`
22. `Stafford Runyon Casabianca Dodd 2019 Behavior Research Methods "variable-length stopping rules" generalized partial credit model minimum information change in theta abstract pubmed`
23. `"expected information gain" threshold stopping "active feature acquisition" OR "Bayesian experimental design" stop when EIG below cost sequential`
24. `"active feature acquisition" "when to stop" OR "stopping" information gain threshold cost sequential test-time EDDI Shim 2018`
25. `Uncertainty of Thoughts LLM asking questions expected information gain 20 questions stopping criterion Hu 2024`
26. `Naghshvar Javidi active sequential hypothesis testing stopping posterior threshold Chernoff sequential design of experiments`

Queries 1-9 (hateful video and video moderation) returned no paper or system with an adaptive stopping rule for
VLM / LLM queries, with one unreadable exception (MAESTRO, section 3.4).

### 3.2 Hateful video detection / localisation: primary sources checked

| Source | Task | How many model calls, and any stopping rule | Match to candidate |
|---|---|---|---|
| LELA, arXiv 2602.09637 (HTML read) | training-free frame-level localisation (HateMM, MHC-EN) | GPT-4o mini; captions summarised per frame, rationale + score per modality (image, OCR, music, video) for every frame: "By repeating this scoring process for each frame throughout the entire video, we obtain a complete temporal hatefulness profile." Fixed; no early exit, no information-gain selection; the only threshold is the decision threshold on the final score. | No |
| TANDEM, arXiv 2601.11178 (HTML read) | detection + timestamp spans (HateMM, MHC-EN, ImpliHateVid) | Fixed 30 s chunks, one VL call and one AL call per chunk, up to 24 frames per chunk; chunk predictions aggregated; no stopping rule, confidence gate or adaptive call count. | No |
| MultiHateLoc, arXiv 2512.10408 (WWW 2026; abstract and reviews) | weakly supervised frame-level localisation | Trained network, no VLM queries; "adaptive top-K" is a frame-selection ratio inside MIL, not a query stopping rule. | No |
| HateClipSeg, arXiv 2508.01712 | segment-level dataset and benchmark | ActionFormer and classifiers; no querying. | No |
| SAGE, ACL 2026 (2026.acl-long.817; abstract) | video-level detection | Gating over modality experts at decision level; one pass; no adaptive querying. | No |
| HVGuard, EMNLP 2025 | video-level detection | MLLM chain-of-thought + MoE fusion, one pass. | No |
| CLARA, arXiv 2608.15905 | clip-level detection | VLM-derived rationales, fixed clips. | No |
| RAMF, arXiv 2512.02743 | video-level detection | Fusion network; no querying. | No |
| SafeLens, AAAI 2026 demo (search snippet) | segment-level labels with rationale | Per-segment decisions; nothing adaptive described. | No |
| SCANNER, arXiv 2602.00132 | detection under shift | Test-time adaptation; no querying. | No |
| "Failures to Surface Harmful Contents in Video LLMs", arXiv 2508.10974 | analysis | Fixed frame-sampling budget; shows omission grows with length; proposes no stopping rule. | No |
| KuaiMod, arXiv 2504.14904; UNIVID, 2026.acl-industry.32 | platform moderation | Staged VLM pipelines, a cheap early-screening filter in UNIVID; no per-video adaptive query count with a stopping statistic. | No |
| MAESTRO, DSTA Young Defence Scientists report (PDF not readable, section 3.4) | video-level detection (MultiHateClip), "localisation ability" claimed | Snippets: "Adaptive Global-Local Reasoning Loop", "dynamic iterative reasoning", LLMs as control agents that select relevant frames "without the need to sample every frame". The stopping criterion of the loop is not visible in any snippet. | Unknown; see 3.4 |

The project's own earlier stopping rules (EIG threshold, VOI, output stability, state entropy, qmixSG, qmixSG_rt10;
17.4 items 1-11) are the only adaptive VLM-query stopping rules found for hateful video, and the candidate replaces
one of them.

### 3.3 Nearest precedents outside hateful video (for the paper's related work; not a rule-4 STOP condition)

Stopping rules that look at the change the last observation made (the backward component):

- Bloodgood and Vijay-Shanker, CoNLL 2009 (arXiv 1409.5165): Stabilizing Predictions, stop when Cohen's kappa between
  consecutive models' predictions on a stop set exceeds a threshold (0.99). Pullar-Strecker et al. 2021 rank it best
  on balanced costs.
- Zhu, Wang and Hovy, COLING 2008 (C08-1142, PDF read): classification-change (stop when no unlabeled example changes
  label between two rounds) and a general combination strategy: when a confidence criterion (max-confidence,
  min-error or overall-uncertainty) fires, check classification-change; if it does not hold, tighten the confidence
  threshold and continue. This is a two-criterion cascade of a state-confidence rule and a change rule.
- Vlachos, Computer Speech and Language 2008: contradictory information added per round, stop after it drops for
  three rounds (not fetched; described in Pullar-Strecker et al. 2021).
- Babcock and Weiss, IACAT 2009 / J. Computerized Adaptive Testing 2012: the change-in-theta (CT) rule for CAT. Wang,
  Weiss and Shang, Psychometrika 2019 (PubMed 30511327): "the CT-rule alone can be unstable and it can end the test
  prematurely", but "can be a useful secondary rule to monitor the point of diminished returns". Stafford, Runyon,
  Casabianca and Dodd, Behavior Research Methods 2019: CT performs well only with a lower criterion and a minimum
  test length; the minimum-information rule gives biased estimates.
- EcoFrame (Li et al., arXiv 2608.03918, HTML read): entropy of the VLM output below a round-relaxed threshold
  tau_t = tau_1 + (t - 1) delta_tau, budget B_{t+1} = min(ceil(alpha B_t), B_max), B = [4, 8, 16, 32]; no change or
  information-gain statistic. This is the source of the project's `_rt` relaxation that qmixE1 drops.

Stopping rules that look at the value of the best next query (the forward component):

- Laws and Schütze, COLING 2008 (C08-1059): uncertainty convergence, the entropy of the last instance selected by
  the query strategy, stopped on the gradient over a window; Zhu and Hovy 2007: max-confidence, the same quantity
  with a threshold.
- Zhang, Cai, Wang and Zhang, ACM TIST 9(2) 2018 (DOI 10.1145/3125645): stop when the model change that any
  remaining unlabeled example could cause falls below a threshold (the best remaining example's expected effect).
- Choi, Grady and Dodd, Educational and Psychological Measurement 2011: predicted standard error reduction (PSER),
  stop when the predicted precision gain from one more item is below a criterion. The minimum-information rule
  (Dodd, Koch and De Ayala 1993, cited through Stafford et al. 2019) stops when no remaining item gives information
  above a threshold.
- Cheng and Huan, arXiv 2509.21734 (HTML read): survey sentence that sequential-design stopping rules threshold
  "posterior variance, information gain, or remaining budget" and are "inherently myopic"; proposes learned optimal
  stopping instead. No cited rule combines a last-change statistic with a next-value statistic.
- Krause and Guestrin (VOI in graphical models) and the active feature acquisition line (Shim et al. NeurIPS 2018,
  Janisch et al. AAAI 2019, EDDI ICML 2019, DIME ICLR 2024) already in `docs/20260908_adaptive_query_survey.md`:
  stop when the next feature's value is below its cost, or a learned stop action.

Posterior-confidence stopping (neither component; the family 17.4 tested as hG / hT / vsum):

- Wald's SPRT; Chernoff 1959; Naghshvar and Javidi, Annals of Statistics 2013 (active sequential hypothesis
  testing): stop when the posterior crosses a threshold.
- AdaFrame (Wu et al., CVPR 2019, arXiv 1811.12432): LSTM policy with predicted future return as the stop signal.
  FrameExit (Ghodrati et al., CVPR 2021, arXiv 2104.13400): per-step gates learned self-supervised on the
  recognition loss, deterministic frame schedule. VideoAgent (Wang et al., ECCV 2024, arXiv 2403.10517): LLM
  self-rated confidence 1-3, stop at 3 (8.4 frames on average). AKeyS (arXiv 2503.16032): LLM judges whether
  confidence suffices, else expands the tree. UoT (Hu et al., NeurIPS 2024, arXiv 2402.03271, HTML read): EIG
  reward chooses the next question; stopping is "until the Questioner correctly determines the final answer, or
  the conversation reaches a maximum number of turns" (20 / 5 / 20 turns), i.e. no EIG-based stop.

Reading of the precedents: pairing a change rule with a precision or information rule is an established design in
CAT (CT as a secondary rule next to SE / MI) and in active learning (Zhu et al. 2008 cascade; Pullar-Strecker et
al. 2021 list change-based and next-query-based criteria side by side). What was not found anywhere: (a) a
two-sided rule in which the forward statistic is the EIG the question policy itself maximises; (b) rank /
quantile normalisation of heterogeneous stopping statistics against a label-free validation pool so that one
threshold, set by a mean-calls budget, serves both; (c) any of this on per-second localisation scores evaluated by
pooled AP / ROC, where the project's own 17.4 record shows why each component alone fails (EIG alone starves
negative videos; change alone starves slowly moving HateClipSeg negatives). The paper can cite the precedents
above and claim (a)-(c) narrowly.

### 3.4 Fetches that failed or were partial

- MAESTRO report (dsta.gov.sg): HTTP 403 for WebFetch and for a direct download with a browser user agent (the
  server returned an HTML block page). Only search snippets were available (section 3.2). This is the one
  hateful-video system with an iterative LLM-controlled loop whose stopping criterion I could not read. The
  snippets describe LLM-driven frame selection for video-level detection on MultiHateClip, which, even if it has a
  stop, would be of the VideoAgent self-judgement type rather than a change + EIG quantile rule; but that is
  inference, not a verified reading. Recorded as a residual gap, not a STOP.
- Springer pages for Stafford et al. 2019 and Wang, Weiss and Shang 2019: login redirects; used the PubMed abstract
  (30511327) and the ETS / search-snippet abstract instead.
- Pullar-Strecker et al. 2021 and Zhu, Wang and Hovy 2008: PDFs fetched as binary and read locally with pdftotext.
- UoT and Cheng and Huan: first 100k characters of the HTML read; enough for the stopping statements quoted.

## 4. The four STOP conditions

| Rule 4 condition | Triggered | Reason |
|---|---|---|
| Source method already used for hateful video detection / localisation | No | 26 searches, 13 hateful-video or moderation sources (section 3.2). No hateful-video work adapts the number of VLM / LLM calls per video with a stopping rule, except this project's own earlier rules; the two-sided change + EIG quantile rule appears nowhere. MAESTRO could not be read (3.4). |
| Pure training / test ensemble | No | One network, one answer model, one posterior; the rule only decides when to stop collecting observations. |
| Pure calibration / post-processing / smoothing | No | The rule does not transform scores; it changes which observations enter the posterior, and so the scores themselves, before the evaluator. The threshold calibration on validation is how the single operating point is set, not a score calibration. (Rule 3 forbids inference-time post-processing of scores; a stopping rule is part of the inference procedure, as the 2026-10-02 complete-method review already treated qmixSG_rt10.) |
| Pure engineering trick (hyperparameters, feature swap, augmentation, training config only) | No | The change is to the decision rule of the query policy, with a stated mechanism (stop when the last answer no longer moves the per-second scores and no remaining question is informative) and a one-switch ablation structure (each component alone is already in the scan: `dlsum` and `eig` rules; the max-quantile combination is the device). It is a small change inside an existing family (qmixSG -> qmixE1 swaps the state components for the forward value); rule 4 does not stop on size. Whether it meets goal 1 is for the evaluator run. |

Not grounds for STOP, recorded for the writing: the rule was selected on test among about 190 rules x 2 orders x
2 targets (README section 3; rule 10), and since qmixE1 is the scan's `qmixS1`, the HateMM and HateClipSeg r5 (3)
rows are the selection set, not a check (the README now says so). Section 17.4 item 9 documents that an earlier
three-corpus pass (qmixSG on 8-level trials) did not transfer to 16-level trials. The README's hold-out checks (6
more r5 trials per corpus, DeHate, inside-outside) did not take part in the choice of the family and are the right
answer to this; the paper must present the selection as developmental and the final-backbone run as the
confirmation. The move from a 16-call to an 8-call maximum was made after the selection and shifts the numbers
(section 5), which is one more reason the `evaluate_stop.py` run, not the scan, must supply the claim rows.

## 5. Factual checks on the proposal

Max 8 (the design under review). Verified against
`runs/20261031_two_goal_diagnostics/policy_lab/<corpus>/rule_scan_cap8.json` (groups `r5/eig`, `r5_more/eig`; rows
`qmixS1@5.0` = qmixE1 and `qmixE3@5.0`; differences to the `fixed8` row of the same trials; shared core evaluator):

| README row (max 8) | qmixE1 stored (AP / ROC / within, calls) | qmixE3 stored | Match | Role |
|---|---|---|---|---|
| HateMM r5 (3) | -.0023 / -.0026 / -.0033, 4.92 | -.0026 / -.0024 / -.0010, 4.93 | yes | selection set |
| HateMM r5 more (6) | -.0002 / -.0016 / -.0017, 4.89 | -.0006 / -.0017 / -.0036, 4.90 | yes | check |
| HateClipSeg r5 (3) | -.0095 / -.0089 / +.0044, 4.96 | -.0086 / -.0075 / +.0077, 4.88 | yes | selection set |
| HateClipSeg r5 more (6) | -.0058 / -.0063 / +.0031, 4.49 | -.0058 / -.0063 / +.0030, 4.51 | yes | check |
| DeHate r5 (3) | -.0024 / -.0027 / -.0041, 4.95 | -.0023 / -.0027 / -.0040, 4.94 | yes | check |

Target 4.5 at max 8 (qmixE1): HateMM r5 -.0019 / -.0020 / +.0007 (4.41); HateClipSeg r5 -.0101 / -.0094 / +.0012
(4.39), the AP -.010 miss the README reports; DeHate r5 -.0023 / -.0025 / -.0052 (4.37). No 5.5 rows exist in the
cap-8 files. Reading against the README's goal-1 criterion (AP / ROC at most .01 lower, within not lower by more
than .005): all five target-5 rows pass; the HateClipSeg r5 AP / ROC margin to the bound is .001, and the HateMM
and DeHate within values are now slightly negative (-.002 to -.004) where they were positive at max 16. These are
scan numbers; the claim rows come from `evaluate_stop.py`.

Max 16 (kept in the README for reference). Verified against `rule_scan_targets.json` and `rule_scan_targets_io.json`
(groups `r5/eig`, `r5_more/eig`, `io/eig`):

| README row (max 16) | qmixE1 stored (AP / ROC / within, calls) | qmixE3 stored | Match |
|---|---|---|---|
| HateMM r5 (3) | -.0033 / -.0050 / +.0039, 4.92 | -.0005 / -.0030 / +.0025, 4.95 | yes |
| HateMM r5 more (6) | +.0043 / -.0010 / -.0043, 4.79 | +.0043 / -.0008 / -.0011, 4.77 | yes |
| HateClipSeg r5 (3) | -.0075 / -.0075 / +.0097, 4.98 | -.0080 / -.0082 / +.0066, 5.08 | yes |
| HateClipSeg r5 more (6) | -.0043 / -.0074 / +.0055, 4.55 | -.0047 / -.0086 / +.0025, 4.36 | yes |
| DeHate r5 (3) | -.0023 / +.0009 / +.0011, 4.69 | -.0032 / -.0001 / +.0018, 4.70 | yes |
| HateMM inside-outside (3) | -.0038 / -.0052 / -.0051, 4.53 | -.0021 / -.0050 / -.0045, 4.50 | yes |
| HateClipSeg inside-outside (3) | -.0088 / -.0004 / -.0035, 5.14 | -.0066 / -.0007 / -.0036, 5.16 | yes |

The inside-outside rows exist only at max 16; the README leaves the max-8 cells empty for them, which is correct.

Precision notes (no change to the verdict):

- The key `qmixE1` does not occur in any stored scan file on this machine, uoa-lab1 or uoa-lab3 (checked
  2026-10-31; the scan outputs were written 19:28-19:46, and commit deb3b38 added `qmixE1` to `stop_rules.py` at
  19:33). The README's statement "qmixE1 is the scan's `qmixS1`" is what makes the table traceable; keep it.
- "there is no floor or schedule constant" plus the maximum-of-8 paragraph is accurate. Add to the README and the
  paper: `videos_at_max_calls` on test and validation, so the share of videos stopped by the budget rather than by
  the rule is visible.
- The sentence "Read together, a video stops when the last answer no longer moves the scores and no remaining
  question is informative" is the correct reading of max-of-quantiles (ask while either is large; stop when both
  are small).

## 6. Scope of the novelty claim

Supportable, if the evaluator run meets goal 1: a label-free two-sided stopping rule for adaptive VLM querying on a
time tree, whose backward statistic is the pooled-metric change caused by the last answer and whose forward
statistic is the EIG the query order itself maximises, both normalised by validation quantiles so that one
threshold set by a mean-calls budget serves both, with no floor beyond the first call, no schedule constant and no
maximum other than the fixed budget it is compared with; the first adaptive stopping rule reported for hateful video
localisation (as far as the searches in section 3 reach), evaluated as calls versus pooled AP / ROC and within-video
ROC against the fixed budget it runs inside.

Not supportable: inventing change-based stopping (Stabilizing Predictions, classification-change, CT rule),
next-query-value stopping (uncertainty convergence, Zhang et al. 2018, PSER, minimum information, VOI), or the
pairing of the two kinds (CAT secondary CT rule; Zhu et al. 2008 cascade); first early-exit or adaptive-budget
inference for video (AdaFrame, FrameExit, VideoAgent, EcoFrame).
