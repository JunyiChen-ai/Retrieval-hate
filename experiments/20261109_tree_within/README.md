# 20261109_tree_within: answers that persist across scale (hidden Markov tree answer model)

User 2026-11-09: the question tree is a core novelty; E3 (experiments/20261108_astra_review) showed that with the same
checkpoint and the same calls per video one tree level beats the full tree on within (HateMM 8-16 s / 16-32 s
windows: -.020 / -.028; HCS and DeHate within ±.01). Goal: change the method so the full tree's within is not below
any single level, keeping its AP / ROC advantage. Host: diagnostics on uoa-lab3 (HateMM, HCS) and uoa-lab1 (DeHate),
CPU; outputs `runs/20261109_tree_within/` (copied to uoa-lab2).

## 1. Diagnostics (test error analysis, rule 10; revision-6 test-selected trials, test split unless noted)

Artifacts read: E3 replays and per-video AUCs (`runs/20261108_astra_review/{replay,e3}/`), cached soft answers
(`data/vlm_tree/<C>/soft_both_p1*.jsonl`), validation and test GT.

1. **Where the tree loses** (`diag_tree_within.py`, `runs/20261109_tree_within/diag/diag_*.json`). At the E3 matched
   calls the full tree covers far less of a positive video at fine resolution: share of seconds whose shortest
   covering asked node is at most 2L + 1 s long, HateMM L16: tree .40, single level .95; DeHate L16 .40 vs .96.
   54-82% of the tree's asked nodes are nested with an earlier asked node (HateMM L16 2.9 of 5.4; DeHate L4 5.9 of
   7.2). HateMM's loss is largest in videos that are mostly hateful (GT fraction .89-.99: within -.054 at L16).
2. **The answer depends on whether a node contains hate, not how much** (`diag_answer_fraction.py`,
   `answer_fraction.json`). Among nodes with at least one hateful second, the Spearman correlation of P(Yes) with the
   node's hate fraction is -.06 to .16 per length bin (AUC for fraction > .5: .43-.61) on all three corpora, val and
   test, except HCS nodes longer than 64 s (.22-.42). The model's "any hate in the node" state is the right one; a
   hate-fraction answer model is not supported.
3. **Answers of nested nodes repeat each other** (`diag_answer_redundancy.py`, `answer_redundancy.json`, val + test).
   Residual of a node = logit P(Yes) minus its cell mean (state, length bin). Correlation of residuals, same video,
   same state:

   | | parent-child (length ratio 2) | ratio 4 | disjoint, ratio 2 | disjoint, ratio 4 |
   |---|---|---|---|---|
   | HateMM negative videos | .81 | .70 | .46 | .47 |
   | HateMM hateful nodes | .71 | .55 | .26 | .25 |
   | HCS negative videos | .66 | .42 | .08 | .05 |
   | HCS hateful nodes | .74 | .60 | .30 | .31 |
   | DeHate negative videos | .78 | .64 | .31 | .31 |
   | DeHate hateful nodes | .73 | .56 | .30 | .30 |

   Nested answers are far more alike than disjoint ones, and the ratio-4 correlation is close to the square of the
   ratio-2 one (.55 vs .71² = .50; .64 vs .78² = .61): the residual behaves like a Markov process down the tree.
   The revision-6 answer model treats all answers as independent given the node states.
4. **What the tree policy asks** (`runs_L0.pkl`, one trial per corpus). HateMM / DeHate: the root first in 59% / 85%
   of videos, then both halves, then the quarters - an almost breadth-first, coarse-to-fine order; HCS asks short
   nodes (depth 5-6). Under independence, the root, the halves and the quarters look like separate pieces of
   evidence; with nested correlation .7 they largely repeat one another, so EIG spends the budget re-asking coarse
   evidence while single-level policies spend it on disjoint windows.
5. **Policy changes alone do not fix it** (`diag_policy_variants.py`, `policy_variants/<corpus>/summary.json`).
   Method minus single level, AP / ROC / within at the E3 matched calls:

   | | HateMM L8 | HateMM L16 | HCS L8 | DeHate L16 |
   |---|---|---|---|---|
   | method (EIG) | .013 / .018 / -.020 | .020 / .016 / -.028 | .022 / .008 / -.006 | -.002 / .014 / -.010 |
   | EER order | .012 / .019 / -.016 | .014 / .016 / -.024 | .000 / -.011 / -.002 | -.001 / .012 / -.012 |
   | EIG, no nested asks | -.002 / .006 / -.021 | .005 / .005 / -.023 | .020 / .012 / .003 | -.039 / -.016 / -.016 |

   "No nested asks" is not a usable policy (after the root nothing is askable: DeHate 1.5 calls on average), but on
   HCS, where the root is rarely asked, avoiding nested asks already brings within above every level.

## 2. Mechanism

The VLM's answer to a node depends on the node's state and on a content-driven propensity that nested nodes share:
the child's frames and transcript are part of the parent's. The revision-6 model ignores this, which (a) counts a
child's answer as fresh evidence although it mostly repeats its parent's, and (b) makes EIG overvalue nested and
coarse questions, so the tree asks redundant coarse-to-fine sequences instead of informative disjoint ones. Modelling
the shared propensity should make the posterior discount repeated evidence and make EIG prefer questions whose answer
is not already implied, so the tree keeps its adaptivity (zoom where useful, cover where needed) without wasting calls.

## 3. Method: hidden Markov tree answer model (HMT)

Every tree node n carries a latent answer bias b_n on a support grid. Given the node state s_n (anchored two states as
in revision 6) and b_n,

    P(o_n = l | s_n, b_n, len_n) = softmax_l(theta_{s_n, l} + omega_l u(len_n) + b_n c_l),  c_l = l / (L - 1) - 1/2,

(the revision-6 table with its shared length slope, tilted by b_n). The biases form a Markov tree: b_root ~ pi; for a
child, b_child = b_parent with probability rho, otherwise a fresh draw from pi. So corr(residual of n, residual of m)
decays as rho^(tree distance), which is the pattern in section 1.3. Special cases: rho = 0 gives independent answers
with a pi-mixture table (revision 6 up to the tilt); rho = 1 gives one offset per video (the per-video offset tried
11-01, `src/qtl/answer_offset.py`, not adopted).

- **Fitting (before training, label-only like the anchored table):** theta_0, omega, pi and rho by maximum likelihood on
  the answers of the negative training videos (every node in state 0, biases summed out exactly by an upward pass over
  the tree); pi is a nonparametric mixing distribution on a fixed grid (Laird 1978), points with weight < 1e-4
  dropped; theta_1 on the roots of the positive training videos with b_root ~ pi summed out. No new searched
  hyperparameter; rho and pi are estimated, not set.
- **Inference:** exact. The chain-tree upward pass (`ctree._up`) gets one more index per node, b_n: a child's table is
  carried to the parent's bias by the transition (O(K) per node: rho times the same b plus (1 - rho) times the
  pi-average), the two children are merged per parent bias as before, and the answer factor depends on (b_n, z_n).
  The G = 0 branch (all nodes state 0) is the same pass over the biases alone. Posteriors of seconds, nodes and node
  biases by automatic differentiation as in revision 6.
- **Questions:** EIG of a candidate node with its answer table mixed over its posterior bias given each state,
  P(o | s) = sum_k P(b_n = k | s, answers so far) P(o | s, b_k): after a parent's answer, a child's answer is partly
  predictable from the parent's bias, so its information drops; disjoint nodes far from asked ones keep the pi-mixture.
- **Training:** the tree likelihood of the training answers (the backbone's answer target) uses the same model, so
  training, epoch selection and test use one answer model (as in revision 6).

Source: hidden Markov trees for persistence across scale in wavelet coefficients (Crouse, Nowak and Baraniuk 1998,
IEEE Trans. Signal Processing); here the persistence is in a VLM's answers to nested temporal questions, combined
with the zero-inflated chain over seconds and the any-hate node states. Proposal review: section 5.

## 4. Plan and success criteria (written before any HMT result)

- **Stage A (developmental, CPU):** the HMT at test time on the revision-6 test-selected checkpoints (backbone and
  training unchanged; answer model refitted with the HMT on the same training answers); policy replay on the test
  split; E3 comparison (same checkpoint, same calls per video, tree vs each single level, single levels also under the
  HMT) and fixed 8 calls. Purpose: check the mechanism before retraining.
- **Stage B (the method, GPU):** HMT in training as well; full Optuna search per corpus and seed with the revision-6
  search space (rule 7; trial count by the first trial's duration); then E3, the budget curve and the ablations.
- **Success:** (1) E3 under the new method: full tree within minus each single level (L4 / L8 / L16) >= 0 on all three
  corpora (3-seed means), and full tree AP or ROC still >= .01 above each single level; (2) fixed 8 calls: AP and ROC
  not below revision 6 by more than .005 (3-seed means) on any corpus, within not below revision 6.

## 6. Stage A results and further diagnostics (2026-11-09)

All on the revision-6 test-selected checkpoints, test split, 3-seed means; "tree - level" = full tree minus single
level at the E3 matched calls (AP / ROC / within); "fixed 8" against the revision-6 method.

**HMT version 1 (tilted multinomial table, one tree bias)** (`runs/20261109_tree_within/stage_a_v1_tilt/`): fit
rho .88 / .92, bias sd 7.3 / 7.9 (HateMM / HCS). Fixed 8 vs revision 6: HateMM -.049 / -.020 / -.012, HCS -.024 /
-.028 / +.003. Tree - level within at L16: HateMM -.024, HCS -.003. Rejected. Fit check (`diag_hmt_fit_check.py`,
answer-level correlation of node pairs in negative training videos by tree distance 1 / 2 / 3 / 4 / >= 5):
HateMM data .77 / .62 / .49 / .48 / .43, model .56 / .45 / .42 / .39 / .23 - the tilt cannot make a mid-range level
likely for a given bias.

**HMT version 2 (ordered logit, one tree bias)** (`stage_a_v2_ordinal/`): fixed 8 vs revision 6: HateMM -.030 /
-.011 / -.024, HCS -.034 / -.036 / +.004; share of nested asks HateMM .74 (revision 6 about .6), HCS .51 (.29).
Without a video-wide part, a node far from the asked ones has an unknown bias (sd ~5 logits), so its answer looks
unreliable and the policy prefers nested nodes whose bias the parent already pins down. Fit check: the model has no
floor at large distance (HCS data .58 at distance >= 5, model .13). Rejected.

**Why the tree asks coarse nodes** (`runs_L0.pkl`): on positive test videos with both classes the posterior
P(G = 1) stays at .72-.75 (HateMM) and about .7 (DeHate) after 8 calls, so EIG = I(o; G) + P(G = 1) I(o; y | G = 1)
keeps spending on the video decision; within depends only on P(y | G = 1, answers).

**Two more policy diagnostics** (`diag_policy_variants.py`, revision-6 model; tree - level, AP / ROC / within):

| | HateMM L4 / L8 / L16 within | HCS L4 / L8 / L16 within | DeHate L4 / L8 / L16 within | fixed 8 vs method (HateMM; HCS; DeHate) |
|---|---|---|---|---|
| method | +.005 / -.020 / -.028 | -.004 / -.006 / -.004 | +.003 / -.001 / -.010 | |
| no parent / child of an asked node (`noadj`) | +.023 / -.002 / -.012 | .000 / -.002 / .000 | +.005 / .000 / -.007 | -.005 / -.001 / +.016; -.002 / +.002 / +.004; -.005 / -.006 / .000 |
| I(o; y \| G = 1) only (`eigloc`) | +.011 / -.025 / -.029 | -.006 / -.008 / -.006 | +.005 / +.004 / -.003 | -.011 / -.018 / +.007; -.008 / -.016 / -.002; -.030 / -.015 / +.002 |

`noadj` keeps AP / ROC above every level (+.003 to +.027) and lifts within, but HateMM L16 stays -.012: avoiding
immediately nested questions is most of the effect; dropping the information about G is not.

**HMT version 3 (bias = video part + tree part)**: fit HateMM rho .62, sigma_v 6.3, sigma_h 4.7; HCS .59 / 4.5 / 5.4;
DeHate .57 / 3.3 / 4.8. Fit check HateMM: model .80 / .70 / .65 / .62 / .59 against data .77 / .62 / .49 / .48 / .43
(decay and floor present, floor too high). Stage A (`stage_a/`): tree - level within HateMM +.020 / -.022 / -.032,
HCS -.009 / -.008 / -.013; fixed 8 vs revision 6 HateMM -.038 / -.034 / -.028, HCS -.020 / -.029 / -.005. The same
ordered-logit table with independent answers (`stage_a_none/`, L16 only) already loses at fixed 8 (HateMM -.014 /
-.007 / -.028, HCS -.023 / -.019 / +.003): swapping the answer table under a backbone trained with the anchored table
costs AP by itself, and at equal table the bias model does not help (HateMM L16 tree - level within -.032 vs -.033
independent; fixed 8 AP -.038 vs -.014). Rejected; versions 1-3 also changed the marginal table, which confounds the
test of the dependence.

**More diagnostics on the revision-6 model** (tree - level within L4 / L8 / L16; fixed 8 vs method AP / ROC / within):

| | HateMM | HCS | DeHate |
|---|---|---|---|
| answers weighted by 1 - r(d)^2 (`distw`) | +.015 / -.014 / -.017; -.011 / -.006 / +.011 | -.005 / -.007 / -.005; -.005 / -.001 / -.001 | -.001 / -.002 / -.010; -.009 / -.013 / -.005 |
| prior logits halved, levels too (`halfprior`) | +.012 / -.021 / -.029; +.001 / +.001 / +.007 | -.020 / -.003 / -.009; -.008 / -.005 / +.007 | (running) |
| no asks within tree distance 2 (`noadj2`) | +.025 / -.007 / -.010; -.025 / -.006 / +.022 | +.007 / +.006 / +.007; .000 / +.011 / +.011 | (running) |

The prior shape is over-confident on positive test videos (seconds with prior P(y | G = 1) < .1 are hateful 34-51% of
the time, > .9 only 62-86%), but softening it raises within for every policy and does not close the tree - level
gap. Excluding questions near an asked node closes the gap on HCS and narrows it on HateMM at an AP cost.

**HMT version 4 (Gaussian copula on the anchored table)**: the revision-6 table stays the exact marginal of every
answer; dependence comes from a latent u_n = v + h_n + e_n (video part, Markov-tree part, independent part, variances
summing to 1) binned by the table's probit thresholds. Fit on the negative training videos (HateMM / HCS): s_v .48 /
.55, s_h .43 / .39, s_e .09 / .06, rho .66 / .62; negative log-likelihood of the negative training answers 38659 vs
52091 with independent answers (HateMM; HCS 4298 vs 6283). Implied latent correlation by tree distance (HateMM) .76 /
.67 / .60 / .57 / .53 against the level correlations .77 / .62 / .49 / .48 / .43. Stage A (`stage_a_copula/`, tree -
level AP / ROC / within at L4 | L8 | L16; fixed 8 vs revision 6; share of nested asks):

| | L4 | L8 | L16 | fixed 8 vs r6 | nested |
|---|---|---|---|---|---|
| HateMM | +.017 / +.014 / +.022 | -.005 / +.003 / -.012 | -.013 / +.000 / -.026 | -.047 / -.040 / -.015 | .69 |
| HCS | +.004 / -.014 / -.005 | +.019 / +.004 / -.010 | +.017 / +.008 / -.005 | -.006 / -.012 / +.000 | .31 |
| DeHate | (running) | | | | |

Rejected: the tree still loses within to L8 / L16, the AP / ROC margin over the levels is gone on HateMM, and fixed 8
loses AP. With the bias known from earlier answers, a nested answer becomes a cleaner signal of the node's state
(explaining away), so EIG still prefers nested questions (share .69, revision 6 about .6); and the discounted
evidence makes the video decision less sharp, which costs AP under a backbone trained with independent answers.
Four answer-dependence models have now failed; the cause is not (only) the independence assumption.

**Where the tree loses, by prior quality** (`diag_prior_trust.py`, `diag/prior_trust.json`; positive test videos with
both classes, terciles of the prior's own within AUC before any call; tree / level change of within from the prior at
the E3 matched calls):

| | prior AUC tercile | hate fraction | tree - prior | level - prior | tree - level |
|---|---|---|---|---|---|
| HateMM L16 | <= .67 / .67-.95 / > .95 | .42 / .55 / .85 | +.121 / -.025 / -.056 | +.140 / -.005 / -.012 | -.019 / -.020 / -.044 |
| HateMM L8 | | | +.134 / -.015 / -.063 | +.159 / -.013 / -.030 | -.025 / -.002 / -.034 |
| HCS L16 | <= .48 / .48-.62 / > .62 | .60 / .52 / .64 | +.025 / +.019 / -.024 | +.018 / +.030 / -.014 | +.008 / -.011 / -.010 |

The loss is not concentrated where the prior is wrong (correlation of prior AUC with tree - level: -.05 to .07). In
the videos where the prior already ranks the seconds well (mostly hateful, HateMM), asking lowers within for both,
and the tree's questions lower it about twice as much as the level's.

**Questions chosen under a flat location prior** (`flatpol`: EIG computed with s_t and phi_n set to 0, scores from
the full prior; levels the same way) and **flat prior everywhere** (`flatprior`), tree - level L4 / L8 / L16 within;
fixed 8 vs method AP / ROC / within:

| | HateMM | HCS | DeHate |
|---|---|---|---|
| flatpol | +.013 / -.020 / -.015; -.008 / -.002 / +.007 | -.001 / -.002 / +.003; -.014 / -.009 / +.010 | +.002 / +.001 / -.003; -.001 / +.001 / +.003 |
| flatprior | +.017 / -.027 / -.026; -.035 / -.006 / +.006 | +.003 / -.001 / -.005; -.018 / -.028 / +.049 | +.004 / -.002 / -.014; -.072 / +.004 / -.006 |

Following the prior's idea of where the hate is explains part of the L16 gap on HateMM (-.028 to -.015) and most of
the small gaps on HCS / DeHate, but not HateMM L8 (-.020 under every EIG variant; -.002 only when parent / child of
an asked node are excluded).

## 5. Proposal review

One independent agent (Claude Fable 5.1), 2026-11-09: **PASS** (rule 4: none of the four STOP conditions). Searches
(12 queries, e.g. "hidden Markov tree video temporal localization", "hateful video localization vision-language model
query adaptive questions", "correlated answers nested segments vision-language model latent bias temporal tree",
"\"hidden Markov tree\" \"hateful\" OR \"hate speech\" video") found no hidden Markov tree or cross-scale answer model
in hateful video detection or localization. Closest work: LELA (arXiv 2602.09637, training-free LLM per-frame hate
scores), MultiHateLoc (arXiv 2512.10408, MIL, no VLM queries), SafeLens (AAAI 2026 demo, per-segment LLM pipeline),
TANDEM (arXiv 2601.11178), FV-Action (arXiv 2608.08315, coarse-to-fine yes/no VLM questions for temporal grounding,
answers used independently, not hateful video), VADTree (arXiv 2510.22693, deterministic segment tree for anomaly
detection), the HMT source (Crouse et al. 1998) and arXiv 2607.22951 (HMM chain for LLM reliability). Not an ensemble,
not calibration (the bias enters the training likelihood, the posterior and the question choice), not an engineering
trick. Wording points taken: the "copy with probability rho, else redraw from pi" transition is a restricted
parameterization of the general HMT parent-to-child transition, and the rho^(distance) decay is a property of that
parameterization, not a result of Crouse et al.; rho = 1 equals the 11-01 per-video offset except for the shared
length slope, which the offset model did not have.
