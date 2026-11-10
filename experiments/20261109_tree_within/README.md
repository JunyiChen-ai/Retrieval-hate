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
| DeHate | +.014 / +.009 / +.012 | -.001 / +.005 / +.006 | +.002 / +.002 / -.003 | -.032 / -.034 / +.003 | .82 |

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

**The tree's questions fixed, posterior from a subset of their answers** (`diag_evidence_rules.py`,
`evidence_rules/<corpus>/summary.json`; tree - level AP / ROC / within at L8 | L16; "all" reproduces E3):

| | all | finest asked nodes only | coarsest asked nodes only |
|---|---|---|---|
| HateMM | +.013 / +.018 / -.020 \| +.020 / +.016 / -.028 | -.008 / +.003 / -.029 \| -.003 / +.002 / -.033 | -.023 / -.001 / -.033 \| -.006 / .000 / -.033 |
| HCS | +.022 / +.008 / -.006 \| +.029 / +.016 / -.004 | +.020 / +.010 / -.003 \| +.026 / +.018 / -.001 | +.014 / +.008 / -.005 \| +.021 / +.015 / -.004 |
| DeHate | +.025 / +.024 / -.001 \| -.002 / +.014 / -.010 | +.003 / -.006 / -.004 \| -.023 / -.013 / -.011 | -.015 / -.011 / -.014 \| -.041 / -.017 / -.017 |

Dropping the coarse (or the fine) answers lowers AP / ROC and does not raise within: combining nested answers is
not what costs within; the set of nodes asked is.

**Joint information of the two question sets under the revision-6 model** (`diag_joint_info.py`,
`joint_info/<corpus>.json`; I(o_S; X) in bits, 64 Monte Carlo answer sequences per video; tree's first k_v asked
nodes vs the level's k_v nodes; positive test videos):

| | L4 tree / level | L8 | L16 | share of videos where the level's set has more (L16) |
|---|---|---|---|---|
| HateMM | 1.67 / 1.55 | 1.49 / 1.32 | 1.29 / 1.03 | .37 |
| HCS | 1.37 / 1.29 | 1.37 / 1.11 | 1.33 / .92 | .30 |

The greedy order is not short-sighted: under its own model the tree's questions carry more information than the
level's, in most videos. The model's information simply does not reward covering the video.

**Answers drawn from the model** (`diag_simulated_answers.py`, `simulated/<corpus>/summary.json`): every answer
replaced by a draw from the trial's anchored table given the node's true state (independent across nodes, one draw
per video and node shared by all policies), real prior; tree - level AP / ROC / within:

| | L4 | L8 | L16 | nested share |
|---|---|---|---|---|
| HateMM | -.004 / +.016 / +.006 | +.010 / +.013 / -.010 | +.009 / +.005 / -.022 | .73 |
| HCS | -.003 / +.002 / +.001 | +.011 / -.005 / +.007 | +.030 / +.010 / +.012 | .30 |
| DeHate | +.008 / +.028 / +.007 | +.004 / +.019 / .000 | -.006 / +.015 / -.010 | .83 |

Even when the answers follow the answer model exactly, EIG on the tree loses within to the 16-33 s level on HateMM
(-.022, real answers -.028) and DeHate (-.010, real -.010). The departure of the real answers from the model
(dependence, table errors) explains little; what remains is the question-selection objective (and the prior it is
computed under).

Policy variants on the same drawn answers (tree - level within L4 / L8 / L16; levels re-run with the same policy):

| | HateMM | HCS | DeHate |
|---|---|---|---|
| EIG (method) | +.006 / -.010 / -.022 | +.001 / +.007 / +.012 | +.007 / .000 / -.010 |
| EIG, flat location prior (`flatpol`) | -.005 / -.006 / -.023 | .000 / +.002 / +.010 | +.012 / +.002 / -.003 |
| I(o; y \| G = 1), flat location prior | -.001 / -.003 / -.014 | +.008 / +.008 / +.016 | +.008 / -.002 / -.007 |
| EIG, no parent / child of an asked node | +.016 / -.006 / -.013 | +.005 / +.011 / +.016 | +.007 / +.004 / -.006 |

With answers that follow the model, no EIG-type policy reaches the 16-33 s level on HateMM (k_v = 5.0 calls on
average, the level asks its complete partition). At that budget a complete partition is very hard to beat on within:
any coarser or nested question leaves part of the video at the prior.

**Expected ranking gain instead of EIG** (`spread`: cpolicy._expected_spread_gain, the expected increase of
sum |q_i - q_j| over pairs of videos on P(G = 1) and over pairs of seconds inside the video on P(y | G = 1); single
levels re-run with the same objective; `policy_variants_spread{,_flatpol}/<corpus>/summary.json`; real answers;
tree - level AP / ROC / within at L4 | L8 | L16; fixed 8 vs method):

| | L4 | L8 | L16 | fixed 8 vs method |
|---|---|---|---|---|
| HateMM spread | +.021 / +.027 / +.022 | +.015 / +.018 / -.008 | +.009 / +.015 / -.018 | -.005 / -.001 / +.012 |
| HateMM spread_flatpol | +.021 / +.033 / +.010 | +.012 / +.017 / -.003 | +.013 / +.013 / -.009 | -.009 / -.007 / +.010 |
| HCS spread | -.013 / -.027 / -.017 | -.003 / -.013 / -.007 | +.011 / +.004 / -.001 | -.022 / -.017 / +.004 |
| HCS spread_flatpol | -.018 / -.029 / -.008 | -.008 / -.012 / -.009 | +.003 / -.004 / -.010 | -.029 / -.021 / +.004 |
| DeHate spread | +.030 / +.032 / +.001 | +.020 / +.023 / -.003 | .000 / +.013 / -.011 | +.003 / -.002 / -.002 |
| DeHate spread_flatpol | +.018 / +.033 / +.002 | +.010 / +.025 / +.002 | -.006 / +.014 / -.007 | -.003 / +.001 / .000 |

A ranking objective narrows the HateMM gap (L16 -.028 to -.009) but does not close it, and loses AP on HCS
(-.022 / -.029 at fixed 8). Rejected.

**Budget-aware tree: an exact cover of the video** (`diag_budget_cover.py`, `budget_cover/<corpus>/summary_<variant>.json`;
cpolicy `cover=`, `budgets=`). E3 reads one anytime run after k_v calls; here the tree is told B_v = k_v and its first
B_v questions must be disjoint tree nodes of any sizes that together cover the whole video (the level's partition is
one allowed cover), chosen by EIG; `cover_root` asks the root first and covers with B_v - 1 nodes; `cover_flatpol`
chooses under the flat location prior. Real answers; tree - level AP / ROC / within at L4 | L8 | L16 against the
revision-6 single levels; "same" = share of videos whose cover is exactly the level's partition (L16):

| | L4 | L8 | L16 | same |
|---|---|---|---|---|
| HateMM cover | -.010 / +.010 / +.009 | -.004 / +.005 / -.011 | +.003 / +.004 / -.014 | .34 |
| HateMM cover_root | -.003 / +.019 / +.008 | +.003 / +.015 / -.011 | +.002 / +.011 / -.015 | .18 |
| HateMM cover_flatpol | -.015 / +.007 / +.015 | -.009 / +.003 / -.001 | -.002 / +.002 / -.004 | .37 |
| HCS cover | -.009 / -.015 / -.004 | +.007 / -.004 / -.006 | +.014 / +.003 / -.004 | 0 |
| HCS cover_root | -.014 / -.014 / +.001 | +.002 / -.003 / -.001 | +.009 / +.004 / +.001 | 0 |
| HCS cover_flatpol | -.013 / -.013 / .000 | +.003 / -.003 / -.001 | +.010 / +.004 / .000 | .04 |
| DeHate cover | +.009 / +.013 / +.001 | +.019 / +.006 / .000 | -.005 / -.002 / -.007 | .49 |
| DeHate cover_root | +.008 / +.023 / .000 | +.015 / +.015 / .000 | -.011 / +.006 / -.012 | .29 |
| DeHate cover_flatpol | +.009 / +.015 / +.003 | +.018 / +.008 / +.003 | -.008 / .000 / -.006 | .51 |

Covering the video moves the tree toward the level on both metrics: the within gap shrinks (HateMM L16 -.028 to
-.004 at best) and the AP / ROC margin over the level shrinks with it (+.020 / +.016 to -.002 / +.002). A cover chosen
by EIG is worse on within than the level's equal windows (HateMM L16 -.014); under the flat location prior it is about
equal. Using the cover only for videos the backbone already finds likely hateful (prior P(G = 1) > .5 or > .8, the
anytime EIG run otherwise; offline mix of the two runs) gives HateMM L16 +.005 / +.012 / -.011 and +.006 / +.015 /
-.015, DeHate L16 -.008 / +.011 / -.010: the same trade-off.

Hate density of the positive test videos with both classes (GT fraction): HateMM mean .61 (median .70), HCS .59
(.64), DeHate .40 (.25). With answers of the form "is there any hate in this window", a nested coarse window says
little about where the hate is when most of the video is hateful; in group testing, pooled tests stop beating
testing every item once prevalence exceeds about .38 (Ungar 1960). The largest gaps are on HateMM, the densest corpus.

At the method's own budget the cover also costs AP: fixed 8 calls vs revision 6, AP / ROC / within, HateMM cover
-.018 / -.013 / +.004, cover_root -.011 / -.003 / +.003, cover_flatpol -.023 / -.015 / +.009; HCS -.016 / -.013 /
.000, -.020 / -.011 / +.005, -.019 / -.011 / +.004; DeHate -.009 / -.019 / -.004, -.010 / -.010 / -.003, -.009 / -.018
/ -.001 (`fixed8` rows; an earlier version of the script reused the L8-budget run here, corrected and re-run).

**Status of the tree-within goal (2026-11-09): not met.** Four answer-dependence models (HMT v1-v4) and fifteen
question policies (EER, no nested, no parent / child, no distance <= 2, location information only, distance-weighted
answers, softened / flat prior, flat prior for the questions only, expected ranking gain with either prior, exact
cover, root + cover, cover under the flat prior, cover only for likely hateful videos) were tried on the revision-6
checkpoints. Every policy that raises the tree's within to the level's lowers its AP / ROC margin over the level to
about zero and costs AP at fixed 8 calls; in the AP-within plane the single level lies on the same line as the tree's
policies, not below it. The questions are "is there any hate in this window"; on densely hateful videos (HateMM) the
coarse questions that buy AP say little about where the hate is. A question whose answer carries location (for
example which half of the window holds the hate) is the remaining change that could move the line itself; it needs
new VLM answers for every node. Decision left to the user.

## 7. Naive-question control and our model on equal windows (2026-11-09)

`naive_questions.py` (`runs/20261109_tree_within/naive_questions/<corpus>.json`): the strongest baseline plus the
soft VLM answers to windows chosen without our policy, rule-14(f) fusion (per second the answer of the shortest asked
window covering it; sigmoid(alpha logit baseline + (1 - alpha) logit answer)); 3-seed means, AP / ROC / within.
`diag_uniform8.py` (`uniform8/<corpus>/summary.json`): our model and tree inference with the 8 equal windows of depth 3
as the only questions. Revision 6 at fixed 8 calls for reference.

| | HateMM (MACIL-SD) | HCS (Fed-WSVAD; DSANet) | DeHate (Fed-WSVAD) |
|---|---|---|---|
| revision 6, 8 calls | .702 / .889 / .783 | .691 / .680 / .563 | .279 / .770 / .681 |
| baseline alone | .573 / .807 / .595 | .562 / .509 / .510; .554 / .528 / .545 | .174 / .701 / .508 |
| baseline + our 8 windows (best alpha on test) | .643 / .861 / .647 | .587 / .575 / .513; .595 / .572 / .539 | .248 / .777 / .586 |
| baseline + 8 equal windows, alpha .5 | .707 / .881 / .692 (7.13 calls) | .673 / .621 / .575; .679 / .640 / .580 | .243 / .773 / .601 (6.56) |
| 8 equal windows, best alpha on test | same (alpha .5) | .693 / .651 / .577 (alpha 0, VLM only); .696 / .653 / .577 | .243 / .777 / .601 |
| baseline + top-8 baseline-scored 4-8 s windows | .680 / .866 / .627 | .672 / .641 / .539; .683 / .645 / .566 | .223 / .756 / .566 |
| baseline + every 4-8 s window (calls) | .688 / .860 / .673 (24.3) | .689 / .659 / .622; .691 / .660 / .623 (43.3) | .226 / .757 / .621 (17.4) |
| our model, 8 equal windows | .690 / .879 / .804 (7.13) | .653 / .650 / .561 | .279 / .755 / .686 (6.56) |

1. The earlier fusion controls (14(f), E1, E1b) gave the baseline our policy's windows, which suit our inference
   (nested, coarse) but not a per-second fusion: with 8 equal windows instead, the same fixed fusion gains .072 AP on
   HateMM, .088-.093 on HCS. Against this control, revision 6 has no AP advantage on HateMM (-.005) or HCS (-.002 to
   -.005), and +.036 on DeHate; ROC +.008 / +.027 / -.007; within +.091 / -.014 / +.080.
2. Inside our model, the EIG tree beats the 8 equal windows on AP / ROC: HateMM +.012 / +.010, HCS +.038 / +.029,
   DeHate .000 / +.015 (within -.022 / +.002 / -.005); the tree's question choice matters for our inference.
3. On HCS the VLM answers alone on 8 equal windows (.693 / .651) beat our model on the same windows (.653 / .650) by
   .040 AP: our answer model and backbone use those answers less well than averaging the raw P(Yes).

Correction 2026-11-10: the "baseline + our 8 windows" row was computed with the asked nodes of whichever replay
directory of a seed was listed last (HateMM seed 2025, HCS seeds 234 and 3407, DeHate seeds 2025 and 3407 were
validation-selected trials); `naive_questions.py` now reads the test-selected trials and the row above is the re-run
(before: .635 / .859 / .642; .600 / .580 / .515, .608 / .581 / .540; .240 / .778 / .590; old files in
`runs/20261109_tree_within/naive_questions_before_20261110/`). The other rows do not depend on the trial.

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

## 8. Extent question: No / Some / All (proposal, 2026-11-09; user chose to build a location-bearing question)

**Problem (sections 6-7).** Every policy trades within for AP along one line, and the single level lies on it. The
question "does this part contain hate" is a presence test (OR over the node's seconds). On densely hateful videos
(positive test videos with both classes: GT fraction mean .61 HateMM, .59 HCS, .40 DeHate) a coarse window almost
always answers Yes, which says nothing about where the hate is NOT; only a complete partition of short windows finds
the clean seconds. Motivation from group testing (not a bound here: seconds come in runs, not i.i.d.): for i.i.d.
items, binary presence tests on pools stop beating one test per item once prevalence exceeds (3 - sqrt 5) / 2 = .38
(Ungar 1960).

**Mechanism.** Ask instead: "Does this part contain hateful content? Answer No if no part of it does, Some if only
some of it does, All if all of it does, from its start to its end." The answer reports presence (OR) and fullness
(AND) of the node. With the tree, a coarse node answered All shifts the posteriors of all of its seconds toward
hateful, and No toward clean, in one call, in sparse and in dense videos alike; Some is the answer that sends the
policy deeper. This is symmetric group testing (outcome 0 if every item of the pool is negative, 1 if every item is
positive, inconclusive otherwise; Emad, Shen and Milenkovic 2011, arXiv 1108.2996; a special case of semi-quantitative
group testing, Emad and Milenkovic, IEEE T-IT 2014), here with noisy soft answers from a VLM, nested temporal pools and
a chain prior. A single level also gains (a mixed window can be ranked below a full one), but only nested questions
can move a long run with one coarse call; so the tree's adaptivity should pay on dense videos, where it currently
loses.

**Model.**
- Node state s_n in {G = 0; G = 1 and none; G = 1 and some; G = 1 and all} (revision 6: three states, the last two
  merged). Exact inference: the chain-tree upward pass (`ctree._up`) carries per node (first second's state, last
  second's state, extent in {none, some, all}) = 12 log-weights instead of 8; children merge by none + none = none,
  all + all = all, otherwise some. Node marginals by automatic differentiation with one node factor per extent value.
- Answer: the first-token distribution over No / Some / All (Qwen2.5-VL-7B, same 4 frames, transcript and per-dataset
  definition as revision 6; `extract_tree_soft.py --prompt p3`). Two coordinates: presence p = 1 - P(No) and fullness
  f = P(All) / (P(Some) + P(All)), each binned into 8 training-quantile levels (label-free, as revision 6).
- Answer table: P(o | s, length) per state, the two coordinates independent given the state, each a multinomial
  logistic regression on the standardised log length (revision 6 form). State none: every node of the negative
  training videos (label-only, as revision 6). States some / all need node-level labels: fitted on the validation
  per-second GT (user ruling 2026-09-29: validation labels may fit the answer model, no gradient).
- Questions: EIG over the four states, w = (1 - pG, pG P(none), pG P(some), pG P(all)) per candidate node.
- One VLM call per question, same inputs as revision 6: the cost of a call is unchanged.

**Plan.** Stage A (developmental, CPU): revision-6 test-selected checkpoints (backbone trained with the yes / no
answers, unchanged), extent answers, extent inference and table; E3 (tree vs each single level, both asking the extent
question, same calls per video) and fixed 8 calls vs revision 6; plus the same run with answers drawn from the fitted
tables (does the mechanism work when the answers follow the model). Stage B (the method, GPU): backbone trained with
the extent answers in the tree likelihood, full Optuna search per corpus and seed (rule 7), then E3, budget curve,
ablations (yes / no question vs extent question at equal calls). Success criteria: section 4, unchanged.

## 9. Proposal review of section 8

One independent agent (Claude Fable 5.1), 2026-11-09: **PASS** (rule 4). 15 searches (web and Semantic Scholar), e.g.
"hateful video localization vision-language model question temporal segments none some all extent", "group testing
vision-language model video segments adaptive pooled questions harmful content moderation", "symmetric group testing
Emad Milenkovic ternary", "VLM temporal grounding coarse-to-fine yes/no query tree". Hateful-video work found (LELA
2602.09637, MultiHateLoc 2512.10408, TANDEM 2601.11178, HateClipSeg 2508.01712, 2508.10974) asks no extent question,
uses no tree with AND / OR node states and no group testing. Not an ensemble, not calibration (observation, latent
state, inference, question choice and the Stage B training likelihood all change); a prompt change alone would be an
engineering change, so the paper must present the state space, merge rule and inference as the contribution and keep
the yes / no vs extent ablation at equal calls. Wording taken: symmetric group testing (Emad, Shen and Milenkovic
2011) instead of Damaschke's threshold group testing (binary test, arbitrary answer between the thresholds); Ungar's
.38 cutoff as motivation only; "shifts the posteriors" instead of "settles". Related work to cite: 2608.08315 (yes / no
probes for temporal grounding), VADTree 2510.22693, VideoTree 2405.19209, TimeSearch 2504.01407, VideoTreeSearch
2607.16189, Zoom-Zero 2512.14273, UniTime 2506.18883, Cuturi et al. 2020 (Bayesian sequential noisy group testing,
2004.12508), 2007.13323 (active pooling by posterior prediction).

## 10. Extent answers: the VLM does not tell All from Some (2026-11-09; Stage A not run)

Answers: `extract_tree_soft.py --view both --prompt p3` (Slurm `launch/sbatch_extract_extent.sh`, job 2010 on
uoa-lab1 = shard 0/2, job 2011 on uoa-lab3 = shard 1/2), every validation and test video of the three corpora
(`data/vlm_tree/<C>/ext_both_p3.shard<i>of2.jsonl`, 0 unparsed answers). The answer check below ran first; it showed
the extent answers carry no information beyond the yes / no answers, so both jobs were cancelled before the training
split was finished (HateMM 120 training videos done, HCS / DeHate none) and Stage A was not run.

Check (`diag_extent_answers.py`, `runs/20261109_tree_within/diag/extent_answers.json`, run on uoa-lab1). Nodes
>= 4 s with both answers; state from the per-second GT; fullness = P(All) / (P(Some) + P(All)), presence = 1 - P(No).

AUC of fullness for all vs some, among hateful nodes of a length bin (test; validation in brackets), and the
correlation of fullness with the node's hate fraction:

| | 4-8 s | 8-16 s | 16-32 s | 32-64 s | correlation with hate fraction |
|---|---|---|---|---|---|
| HateMM | .478 (.491) | .495 (.475) | .478 (.463) | .505 (.554) | -.04 to .11 |
| HCS | .567 (.609) | .579 (.625) | .557 (.595) | .549 (.593) | .09 to .44 |
| DeHate | .506 (.439) | .472 (.459) | .475 (.463) | .543 (.495) | -.07 to .10 |

- HateMM and DeHate: no signal (AUC .44-.55 on both splits). HCS: weak (.55-.64), and HCS is the corpus where the tree
  is already within .01 of the levels. The extent model needs All vs Some separation on dense videos (HateMM), where it
  is absent.
- Presence 1 - P(No) is the yes / no answer again: AUC hateful node vs the rest, test, .844 / .710 / .723 (yes / no
  .839 / .703 / .724), validation the same within .013; inside positive videos the two are equal within .03 at every
  length.
- This repeats section 1 item 2 with a question that asks for the extent explicitly: the VLM's answer depends on
  whether a window contains hate, not on how much of it does.

A second finding from the same check: inside positive videos (validation + test, per video with both classes, mean
AUC of presence for none vs hateful nodes of one length):

| | 4-8 s | 8-16 s | 16-32 s | 32-64 s |
|---|---|---|---|---|
| HateMM | .697 (107 videos) | .758 (76) | .836 (47) | .852 (19) |
| HCS | .655 (116) | .680 (110) | .700 (87) | .614 (50) |
| DeHate | .657 (252) | .660 (225) | .626 (179) | .651 (110) |

On HateMM, answers on short windows separate the hateful from the clean parts of a hateful video much worse than
answers on 16-64 s windows (the video sets differ between bins, so this is indicative). On HCS and DeHate the answer
quality is about flat in length. The deep nodes, where the tree spends its later calls, are the least reliable on the
corpus where the tree loses within.

**Status: the extent question is rejected at the answer level; the tree-within goal is still not met.** Options are
in STATUS.md (2026-11-09).

## 11. The answer table over-trusts a Yes inside hateful videos; a validation-fitted table does not close the gap (2026-11-09)

**Check** (`diag_table_trust.py`, `runs/20261109_tree_within/diag/table_trust.json`; revision-6 test-selected
trials, yes / no answers in 8 levels, validation + test nodes). Table AUC = how far apart the revision-6 table puts
the answers of hateful and non-hateful nodes, at the bin's median length (mean of 3 trials); data AUC = how far apart
the real answers are, for hateful nodes against nodes of negative videos, against clean nodes of hateful videos, and
per hateful video (mean over videos):

| | length | table | data: vs negative videos | vs clean nodes of hateful videos | per video | mean level negative / clean / hateful |
|---|---|---|---|---|---|---|
| HateMM | 4-8 s | .899 | .807 | .663 | .680 (107) | 2.26 / 3.56 / 4.68 |
| | 16-32 s | .911 | .878 | .766 | .796 (47) | 2.72 / 4.17 / 5.86 |
| | 32-64 s | .912 | .890 | .821 | .853 (19) | 3.04 / 4.20 / 6.20 |
| HCS | 4-8 s | .823 | .762 | .673 | .644 (116) | 1.87 / 2.57 / 3.91 |
| | 16-32 s | .821 | .803 | .705 | .689 (87) | 2.58 / 3.48 / 4.96 |
| DeHate | 4-8 s | .583 | .696 | .612 | .645 (252) | 2.93 / 3.64 / 4.50 |
| | 16-32 s | .623 | .746 | .624 | .621 (179) | 3.83 / 4.86 / 5.69 |

The table's "no hate" row is fitted on negative videos only (revision 6), but the clean parts of hateful videos get
much higher answers than negative videos (HateMM level 3.56 vs 2.26 at 4-8 s). On HateMM and HCS the table therefore
over-trusts a Yes inside a hateful video, most of all on short nodes (HateMM .899 vs .663), which are the nodes the
tree reaches last. On DeHate the table under-trusts the answers.

**E3 with the validation-fitted three-state table** (`diag_policy_variants.py --variants val3 val3_noadj`,
`runs/20261109_tree_within/policy_variants_val3/<corpus>/summary.json`; `answer_calib.fit_validation_answer_model`:
negative video / clean node of a hateful video / hateful node, own length slope each, validation GT, no gradient;
revision-6 checkpoints unchanged; tree and single levels both use the new table; tree - level AP / ROC / within at
the E3 matched calls; fixed 8 against revision 6):

| | L4 | L8 | L16 | fixed 8 vs revision 6 |
|---|---|---|---|---|
| HateMM val3 | +.022 / +.024 / +.021 | .000 / +.011 / -.022 | +.005 / +.011 / -.004 | -.021 / -.004 / +.003 |
| HateMM val3_noadj | +.024 / +.025 / +.019 | +.002 / +.012 / -.021 | +.002 / +.011 / -.004 | -.020 / -.003 / +.001 |
| HCS val3 | -.014 / -.027 / +.001 | -.005 / -.019 / -.004 | +.003 / -.007 / .000 | -.028 / -.023 / +.002 |
| HCS val3_noadj | -.014 / -.023 / -.001 | -.005 / -.015 / -.006 | +.003 / -.003 / -.001 | -.027 / -.020 / .000 |
| DeHate val3 | +.028 / +.031 / +.004 | +.020 / +.018 / +.001 | -.007 / +.008 / +.001 | -.001 / +.015 / -.003 |
| DeHate val3_noadj | +.028 / +.026 / +.002 | +.021 / +.014 / -.002 | -.005 / +.005 / -.002 | .000 / +.010 / -.005 |

(revision-6 method for reference, section 6: HateMM within L4 / L8 / L16 +.005 / -.020 / -.028.) The calibrated
table closes the HateMM L16 gap (-.028 to -.004) and keeps DeHate within at or above every level (fixed-8 ROC .785
vs .770), but the HateMM L8 gap stays (-.022; -.021 with noadj), HCS loses its AP / ROC margin over the levels, and
fixed-8 AP drops .021 (HateMM) / .028 (HCS), as in `20261101_r5_completion` item 4 on revision 5. Not a solution.

**Status of the tree-within goal: blocked.** Tried on the revision-6 checkpoints: four answer-dependence models,
fifteen question policies, a location-bearing question (section 10: the VLM cannot answer it), and a calibrated
three-state table with and without the noadj rule. None gives tree within >= every single level on all three corpora
while keeping the AP / ROC margin and the fixed-8 AP. HateMM's 8-17 s level stays ahead by .02 in every version
except the cover / noadj policies (-.001 / -.002), which lose the AP margin.

## 12. Why the naive-question control ties revision 6 on pooled AP (2026-11-10)

User 2026-11-10: before deciding how the tree is claimed, find out why equal windows with a simple fusion reach our
pooled AP (section 7 items 1 and 3). Diagnostics only, nothing retrained, method unchanged. Host uoa-lab3 (sc474398),
CPU; revision-6 test-selected trials, test split, 3-seed means, AP / ROC / within. Scores are evaluated at 6 decimals,
as the evaluator reads score files (revision 6 at fixed 8 calls reproduces .702 / .889 / .783, .691 / .680 / .563,
.279 / .770 / .681).

### 12.1 The control at every budget, with the baseline's scores and with our own 0-call scores

`diag_control_curve.py --arms c_label` (`runs/20261109_tree_within/control_curve/<corpus>.json`). K equal windows of
one tree depth (fewer when the video is too short), per second the soft answer of its window, fused with a per-second
prior as in section 7 at alpha .5 (no tuning). Mean calls in brackets.

HateMM:

| K | revision 6, fixed K | VLM answers only | MACIL-SD + answers | our 0-call scores + answers |
|---|---|---|---|---|
| 1 | .647 / .851 / .771 (1.00) | .612 / .862 / .500 (1.00) | .688 / .885 / .595 | .669 / .880 / .771 |
| 2 | .667 / .865 / .778 (1.99) | .612 / .856 / .596 (1.99) | .683 / .881 / .655 | .670 / .874 / .780 |
| 4 | .687 / .881 / .780 (3.94) | .641 / .860 / .640 (3.86) | .693 / .886 / .675 | .685 / .876 / .764 |
| 8 | .702 / .889 / .783 (7.63) | .643 / .845 / .672 (7.13) | .707 / .881 / .692 | .691 / .869 / .765 |
| 16 | .690 / .885 / .791 (14.11) | .627 / .822 / .687 (12.50) | .702 / .871 / .699 | .686 / .861 / .766 |
| 32 | .693 / .879 / .793 (25.02) | .595 / .792 / .670 (19.72) | .681 / .857 / .679 | .668 / .849 / .756 |

HCS (calls = K):

| K | revision 6, fixed K | VLM answers only | Fed-WSVAD + answers | our 0-call scores + answers |
|---|---|---|---|---|
| 1 | .661 / .657 / .556 | .700 / .646 / .500 | .672 / .604 / .510 | .704 / .663 / .556 |
| 2 | .658 / .659 / .561 | .698 / .650 / .502 | .673 / .608 / .508 | .705 / .664 / .549 |
| 4 | .676 / .673 / .562 | .686 / .641 / .545 | .664 / .607 / .544 | .703 / .666 / .562 |
| 8 | .691 / .680 / .563 | .693 / .651 / .577 | .673 / .621 / .575 | .708 / .672 / .578 |
| 16 | .687 / .677 / .572 | .701 / .652 / .629 | .681 / .626 / .628 | .713 / .677 / .627 |
| 32 | .689 / .676 / .606 | .696 / .658 / .617 | .675 / .630 / .619 | .715 / .687 / .626 |

DeHate:

| K | revision 6, fixed K | VLM answers only | Fed-WSVAD + answers | our 0-call scores + answers |
|---|---|---|---|---|
| 1 | .242 / .733 / .662 (1.00) | .182 / .744 / .500 (1.00) | .237 / .772 / .507 | .244 / .768 / .662 |
| 2 | .251 / .744 / .662 (1.99) | .195 / .747 / .549 (1.98) | .242 / .780 / .558 | .259 / .770 / .654 |
| 4 | .257 / .757 / .669 (3.88) | .193 / .741 / .584 (3.71) | .245 / .775 / .587 | .251 / .770 / .661 |
| 8 | .279 / .770 / .681 (7.26) | .182 / .731 / .591 (6.56) | .243 / .773 / .601 | .247 / .769 / .665 |
| 16 | .284 / .782 / .691 (13.00) | .178 / .715 / .618 (10.61) | .238 / .766 / .626 | .239 / .764 / .685 |
| 32 | .286 / .784 / .700 (21.96) | .164 / .689 / .619 (15.37) | .227 / .757 / .625 | .231 / .754 / .687 |

### 12.2 The same 8 equal windows, step by step

`diag_uniform8_decompose.py` (`uniform8_decompose/<corpus>/summary.json`), every row uses the same 8 answers (the
last two rows aside). In brackets the pooled AP when every second gets its video's mean score, i.e. the part of the
pooled AP that the ordering of whole videos gives. `diag_video_level_swap.py` (`video_level_swap/<corpus>.json`) gives
the two rows without answers.

| | HateMM | HCS | DeHate |
|---|---|---|---|
| answers painted on the seconds, raw P(Yes) | .643 / .845 / .672 (.665) | .693 / .651 / .577 (.696) | .182 / .731 / .591 (.199) |
| the same, 8 levels | .569 / .836 / .638 | .644 / .641 / .563 | .156 / .724 / .588 |
| the same, table log-likelihood ratio | .607 / .840 / .651 | .644 / .632 / .553 | .170 / .720 / .602 |
| chain only, no network, no answers | within .723 | within .619 | within .604 |
| tree inference, no network (s, phi, g = 0) | .689 / .875 / .826 (.662) | .663 / .641 / .619 (.674) | .194 / .761 / .698 (.193) |
| tree inference, backbone's video logit only (s, phi = 0) | .700 / .888 / .824 (.674) | .661 / .647 / .619 (.674) | .208 / .766 / .698 (.209) |
| tree inference, full prior (= section 7 last row) | .690 / .879 / .805 (.641) | .653 / .650 / .561 (.645) | .279 / .755 / .686 (.266) |
| our 0-call scores alone | .568 / .801 / .771 (.555) | .650 / .648 / .556 (.653) | .220 / .717 / .661 (.218) |
| our 0-call scores + painted answers, alpha .5 | .691 / .869 / .766 (.682) | .708 / .672 / .578 (.697) | .247 / .769 / .666 (.260) |
| our 0-call scores + table log-likelihood ratio (no tree) | .670 / .861 / .786 | .678 / .666 / .576 | .238 / .737 / .699 |
| baseline + painted answers, alpha .5 (the control) | .707 / .881 / .692 (.706) | .673 / .621 / .575 (.643); DSANet .679 / .640 / .580 (.673) | .243 / .773 / .601 (.254) |
| revision 6, EIG tree, 8 calls | .702 / .889 / .783 (.665) | .691 / .680 / .563 (.681) | .279 / .770 / .681 (.267) |

### 12.3 Findings

1. **The control's pooled AP is the ordering of whole videos.** Its AP equals the AP of its video-mean score (HateMM
   .707 vs .706; HCS answers only .693 vs .696), and one question about the whole video already gives it: HateMM
   baseline + root answer .688 / .885, HCS root answer alone .700 / .646, against revision 6 at 8 calls .702 / .889
   and .691 / .680. More windows add at most .02 AP (HateMM) or nothing (HCS). Revision 6 orders whole videos worse
   (video-mean AP .665 / .681) and gains .037 / .010 AP from the seconds inside the video. The tie on AP is a tie
   between a better video ordering without localization and a worse video ordering with localization. On DeHate
   revision 6 is ahead on both parts (.267 vs .254 video-mean, .279 vs .243).
2. **Revision 6 uses a video-level answer worse than the plain fusion does.** With one call it is below the fusion of
   its own 0-call scores with the root answer: AP -.022 / -.043 / -.001, ROC -.028 / -.007 / -.035 (HCS: revision 6
   rarely asks the root first). With our own scores as the prior, the plain fusion beats revision 6 on HCS at every
   budget (AP +.016 to +.047, within +.015 / +.055 / +.020 at 8 / 16 / 32 calls); revision 6 is ahead on HateMM from 4
   calls (8 calls +.010 / +.020 / +.017) and on DeHate from 8 calls (+.032 / +.001 / +.015; 16 calls +.045 AP).
   Without the tree, on the 8 equal windows: our 0-call scores plus the answer table's log-likelihood ratio (full
   weight, 8 levels) against our 0-call scores plus raw P(Yes) at alpha .5: AP / ROC -.022 / -.008, -.030 / -.007,
   -.009 / -.032 (within +.021 / -.002 / +.033). The table's evidence orders the pooled seconds worse than the raw
   answer in a half-and-half logit average, on every corpus.
3. **MACIL-SD combines better with the answers than our backbone on HateMM** (.707 vs .691 AP in the same fusion;
   alone .573 vs .568). Tested reason "our backbone was trained on the answers": the label-only backbone (arm
   c_label) in the same fusion gives .674 / .857 / .739, HCS .693 / .673 / .592, DeHate .200 / .744 / .652, all below
   our backbone. Not supported; the reason is open.
4. **The per-second prior of the backbone lowers within on equal windows on every corpus**: full prior minus video
   logit only, within -.019 / -.058 / -.012; AP / ROC -.010 / -.009 on HateMM, -.008 / +.004 on HCS, +.071 / -.011 on
   DeHate (DeHate's AP gain is the video ordering: video-mean AP .209 to .266). On HateMM the model with 8 equal
   windows and only the backbone's video logit equals revision 6 on AP / ROC (.700 / .888) with within .824 against
   .783.
5. **The tree's margin over 8 equal windows depends on the corpus and on the per-second prior.** EIG tree minus 8
   equal windows (HateMM, HCS, DeHate): full prior +.012 / +.010 / -.022, +.038 / +.029 / +.002, .000 / +.015 / -.005;
   flat location prior (section 6 `flatprior` at fixed 8 against the row "video logit only") -.033 / -.005 / -.035,
   +.012 / +.005 / -.007, -.001 / +.008 / -.023; no network (arm `noback`, EIG tree, .644 / .876 / .796, .694 / .656 /
   .627, .195 / .756 / .644, against the row "no network"; a separately trained arm, so its chain and table are not
   the row's) -.045 / .000 / -.030, +.032 / +.015 / +.008, +.001 / -.004 / -.054. HateMM: the tree is ahead on AP /
   ROC only with the per-second prior and behind without it. HCS: the tree keeps an AP margin in all three settings
   (+.012 to +.038). DeHate: no AP margin in any setting. Within is lower for the tree on HateMM and DeHate in all
   three settings.
6. **A position prior explains much of within.** The learned chain alone (no network, no answers; the score depends
   only on the second's position and the video length: about half the video mean in the first and last tenth, flat in
   between) has within .723 / .619 / .604; the baselines have .595 / .510-.545 / .508. On HCS the 8 answers add
   nothing to it (.619) and revision 6 is below it at 8, 16 and 32 calls (.563 / .572 / .606).
7. **The 8 answer levels lose little ranking information.** Painted levels against raw P(Yes): ROC -.009 / -.011 /
   -.007, within -.034 / -.014 / -.004; the AP loss (-.073 / -.049 / -.026) is mostly ties (8 distinct values), which
   the tree inference breaks.

Not established: why the baseline's scores combine better with the answers than ours on HateMM (item 3); whether a
method change can keep the per-second prior's AP gain on DeHate without its within cost (item 4 is measured on the
revision-6 checkpoints, which were trained with that prior). `diag_video_level_swap.py` also recombined the video
means of one method with the within-video shape of the other; both recombinations lose AP against both originals
(HateMM .653 and .668 against .702 and .707), so pooled AP does not split into the two parts and those rows are not
used.

**What this means for the claims (decision left to the user).** Against "strongest baseline + the same number of
VLM answers on equal windows", revision 6 has a pooled-AP margin only on DeHate (+.036); on HateMM and HCS pooled AP
is reached by ordering whole videos with one or more answers. What revision 6 has beyond the control is within on
HateMM and DeHate (+.091 / +.080; part of it is the position prior, item 6), ROC on HateMM and HCS (+.008 / +.029 to
+.059), and AP on DeHate. The tree's question choice against 8 equal windows (item 5) gives AP +.012 / +.038 / .000;
on HateMM that margin needs the per-second prior, on HCS it does not, and within is lower for the tree on HateMM and
DeHate; the tree-within goal of section 4 stays unmet.
