# GPT-6-astra 方法审稿：模块一在线查询第 4b 版（2026-09-17 NZST，冻结记录）

审稿人：GPT-6-astra（通过 codex MCP，model `gpt-6-astra`），只凭下方口述的方法描述，未读本仓库任何文件；用户要求只评方法弱点，不评评测细节。原文逐字保存；整理后的 7 条摘要与处理见 `experiments/20260910_online_query_within/README.md` 第 11 节。

## 1. 口述的方法描述（我方 prompt）

```text
Please review the following method (a weakly supervised hateful-video temporal localization method; the current version is what we internally call "4b"). I describe it fully; no files are available.

# Task
Given a video with only a video-level label (hateful / not hateful) at training time, output a per-second hate score at test time (temporal localization). Two corpora: HateMM (~1000 videos, hateful videos are long, hate is often diffuse) and HateClipSeg (~400 videos, hate is more localized). Frozen features at 1 fps: I3D (visual), VGGish (audio), BERT sentence embeddings of the Whisper transcript (text). No new VLM calls are made during our experiments: all VLM verdicts are cached offline.

# Module 1: frozen VLM verdicts (the "queries")
Qwen2.5-VL-7B-Instruct is asked, per video, to rate 30 equal-length fine windows and 4 equal-length coarse blocks. Each window/block is rated in isolation from 4 uniformly sampled frames plus the Whisper transcript of that window, with a rubric 0..3 (0 none, 1 faint, 2 clear but implicit, 3 explicit hate). Empirically the model outputs almost only 0 or 3 (levels 1 and 2 together are < 1% of windows), so we binarize at >= 2. Each verdict is treated as a binary noisy observation of "is there hate in this interval". Test-time errors of these verdicts are strongly correlated within a video (adjacent-window error correlation ~0.65 on HateMM, ~0.39 on HateClipSeg; per-video error rate std is 3-4x what independence would give), i.e. the dominant error mode is video-level (a video where the VLM over-fires everywhere or misses everywhere).

# Module 3: interval evidence HMM (fusion of verdicts into a per-second posterior)
A two-state hidden process s(t) in {0,1} ("this moment is hateful") on the video's normalized time axis. Boundaries of the 30 fine windows and 4 coarse blocks are merged into 32 segments; between segments the state evolves by a continuous-time 2-state Markov chain exp(Q*dt). Each fine (coarse) verdict is a noisy observation of "any segment inside its interval has s=1", with emission parameters q_f, r_f (q_c, r_c) = P(verdict=1 | any hate in interval) and P(verdict=1 | no hate in interval). The augmented state (s, h_fine, h_coarse) has 8 values, forward-backward is exact. 7 parameters (two transition rates, initial p0, four emission probs) are fitted by EM on the training set using only video-level labels: negative videos fix the false-positive rates; positive videos are constrained to have at least one hateful segment. Missing verdicts (windows not yet queried) are simply not emitted, so the posterior is well defined for any subset of queried windows. Outputs per second: posterior log-odds ell(t), P(s_t=1), and posterior for each coarse block P(h_j) used as a soft block-level MIL label. A video-level "reliability mixture" extension (3 latent regimes: normal / over-firing / missing) was tried: it improves held-out verdict prediction log-loss a lot but drops pooled AP by 5-7 points because "hate everywhere" and "over-firing everywhere" are not identifiable without labels; so it is not used.

# Evidence decomposition (added in round 3)
The per-second evidence fed to the backbone is E(t) = ell_fine(t) + x(t) + v, where
- ell_fine(t): HMM posterior log-odds computed with the coarse-block emissions switched off (fine windows only),
- x(t): a centered log-odds from a frozen off-the-shelf text hate classifier (cardiffnlp twitter-roberta-base-hate-latest) applied to the ASR utterance and the OCR text covering second t, max over the two, 0 if no text; the centering constant is the training-set median of the classifier logit,
- v: a video-level constant = log-odds that "at least one segment is hateful" given only the 4 coarse verdicts (from the same HMM).
Motivation: on HateClipSeg the coarse-block verdicts are anti-informative within a video (using them per-second gives within-video ranking below chance) but informative at the video level; on HateMM they are informative per-second. The decomposition lets coarse verdicts act only at the video level. The block-level MIL labels P(h_j) still come from the full posterior.

# Module 2: backbone, "evidence-routed cross-modal attention"
- Two content streams: h_v = fc_v(I3D), h_a = fc_a(VGGish concat BERT-sentence). Content never contains verdicts.
- Evidence code per second: e(t) = Emb[4-cell (b_fine, b_coarse)] + W*[E(t)/scale, P(s_t)], where b_fine / b_coarse in {not queried, 0, 1} for the window / block containing t (a "not queried" explicit state).
- One pre-norm cross-modal attention layer (shared weights) run in both directions (visual queries audio; audio queries visual). The evidence code is added to the query and key inputs only (not to values, not to the residual stream), and a per-head additive score bias beta_h(e_j) computed from the KEY second's evidence is added to attention logits. A video-level calibration vector c = Linear(mean_t e(t)) is added to both stream outputs.
- Per-second logit = fc(a_out) + fc(v_out) + alpha * E(t) (alpha is a learned/searched scalar "prior scale").
- Claim: evidence decides from which seconds content is aggregated and a video-level calibration, but never enters the content representation.
- Losses: (i) top-k (k = ceil(T/16)) bag mean BCE against the video label; (ii) CMAL, MACIL-SD's cross-modal contrastive instance loss (InfoNCE pulling together audio and visual representations of the top-k seconds of videos the model itself scores as hateful, pushing away from bottom-k seconds and from top-k seconds of videos scored normal); (iii) block-level MIL: BCE of the top-k content logit within each coarse block against P(h_j), weighted by |2P(h_j)-1|; (iv) masked-window verdict prediction (see below). Only two method-level scalars (alpha, lambda_block); other hyperparameters (lr, seq len, lambda_cma) are searched.
- An earlier ablation (on a previous revision) showed: removing VLM verdicts entirely costs 11-14 AP points; the interval HMM vs an index-based HMM matters; the attention-routing components matter on HateMM (~2-9 points) but are near zero on HateClipSeg; the whole evidence-routed attention vs simply concatenating verdict columns into the audio stream is only +0.5 AP on HateMM (+0.7 on HateClipSeg). Shuffling the time alignment of evidence at inference hurts HateMM (1-2 points) but not HateClipSeg.

# Training protocol: single-run online query with budget-aware training
- Training-time budget: each training video starts with its 4 coarse verdicts and NO fine-window verdicts. After epochs 1, 2, 3 and 4, an "acquisition event" runs: the current model computes, for every training video and every not-yet-queried fine window, an EOC score = expected change of the per-second sigmoid outputs if that window were revealed, estimated by two counterfactual forward passes (verdict=0 / verdict=1) weighted by the model's own predicted probability that the window's verdict is 1 (sigmoid of the top-k content logit within the window). Nothing about the true cached verdict is read at this stage.
- Cross-video allocation (the "4b" change): each event has total budget N (number of training videos) split into 2 sub-rounds; in each sub-round every video nominates its best window and gain, all nominations are sorted globally, and the top N/2 videos get their window revealed (cached verdict read, counted as one call). The second sub-round recomputes on the updated state. So a video can receive 0, 1 or 2 windows per event; over 4 events the average is 4 fine windows per video (train-time calls = 4 coarse + 4 fine = 8 on average), but the realized distribution is bimodal: ~30% of training videos never receive any fine window, ~30% receive all 8 (2 per event, since the second sub-round almost always goes to the same videos as the first), positives average 5.7-6.9 fine windows, negatives 2.1-2.9. After each event the HMM is refitted on the revealed verdicts.
- Evidence dropout: whenever a training video enters a batch, the number m of revealed fine windows fed as input is drawn uniformly from 0..(number revealed so far); with prob 0.5 the subset is a random subset of that size, with prob 0.5 it is the prefix of that size in acquisition order. Windows not fed are shown as "not queried"; the HMM posterior is recomputed for the fed subset. Coarse blocks are always fed.
- Masked-window verdict prediction loss: for windows whose verdict is revealed but masked out in this item, the top-k content logit within the window is trained by BCE to predict the (known) cached verdict; on negative videos every window's target is 0. Averaged per video then across videos; shares weight lambda_block with block MIL.
- Checkpoint selection on validation using a fixed uniform mask of 4 fine windows, criterion (AP + ROC + within-video ROC)/3, only epochs after the last acquisition event.

# Test-time protocol
Per video: query the 4 coarse blocks; then repeatedly compute EOC for all not-yet-queried fine windows, stop if the max EOC < tau (tau chosen on validation from a small grid), else query that window (read cached verdict), update the HMM posterior and rescore; cap at 4 fine windows (8 calls). Report the fixed 8-call point and the stop-rule point.

# What the results look like (for context only; do not review evaluation)
- At the fixed 8-call point the method beats all trained weakly-supervised baselines on pooled AP/ROC on both corpora and is the first to exceed the strongest baseline on within-video ROC on both corpora.
- Versus the same backbone trained with all 34 verdicts and no querying (a previous revision), the 8-call model is ~1.5-2 AP lower on both corpora.
- On HateMM, the validation-selected stop threshold ends up querying NO fine windows for 212/214 test videos (coarse-only already reaches the reported within-video ROC; adding fine windows gives at most +0.7 AP, and beyond 12 calls performance decreases). On HateClipSeg, performance increases monotonically with calls up to 34.
- EOC-based window selection at test time is only 0.1-0.7 AP better than uniform or random selection of the same number of windows; entropy / HMM-uncertainty / "conflict" policies are similar.
- The cross-video training allocation (4b) vs one window per video per event (round 3): HateClipSeg within-video ROC +1.5 points, HateMM unchanged, pooled AP slightly down on HateClipSeg.
- Putting the frozen text classifier's score x(t) into the final logit via alpha*E(t) has raised the question whether this is an ensemble with an external classifier.

Please give your review comments on the method's weaknesses, ranked by severity, and for each: why it matters, and what you would change or test. Also comment on (a) whether the "evidence never enters the content representation" claim is meaningful given the calibration vector c and the alpha*E(t) prior term, (b) the training-time cross-video allocation producing a bimodal distribution, (c) the test-time query policy being indistinguishable from uniform, (d) the identifiability problem of the reliability mixture and whether there is a better way to model video-level verdict errors, (e) the decomposition E(t) = ell_fine + x + v as a modeling choice, (f) how you would restructure the contribution story if you were the authors. I may follow up with questions.
```

## 2. 审稿意见（原文）

## Overall assessment

The main vulnerability is **not backbone complexity but the probabilistic interpretation of the evidence and the circularity between evidence, predictions, and acquisition**. The method is defensible as budgeted learning from noisy external teachers; it is less convincing as principled Bayesian evidence fusion or reliably informative active querying.

### 1. Critical: the HMM assumes away the dominant error mechanism

**Why it matters.** Conditional independence of emissions is badly mismatched to video-level over-firing and missing. Repeated correlated verdicts can produce unjustified certainty. That certainty then propagates into attention, the additive prior, acquisition, and block supervision. Weighting block labels by \(|2P(h_j)-1|\) can amplify precisely these overconfident mistakes.

Video labels constrain bag positivity, but do not identify temporal occupancy, transition rates, and sensitivity separately. Negative videos help estimate false-positive behavior; they do not resolve diffuse true hate versus correlated false positives in positive videos.

**Change/test.**
- Check parameter identifiability through constructed cases: diffuse hate, isolated hate, and video-wide teacher failure.
- Consider strongly regularized video-level random effects on emission probabilities, rather than unconstrained discrete reliability regimes.
- Crucially, acknowledge that random effects **do not solve identifiability**. Additional anchors or justified constraints are needed.
- Without such anchors, use conservative likelihood tempering or bounded evidence accumulation rather than interpreting the posterior as calibrated certainty.

The failed reliability mixture exposes a genuine ambiguity; removing it does not remove that ambiguity.

### 2. High: \(E=\ell_{\mathrm{fine}}+x+v\) is not a justified log-odds decomposition

**Why it matters.**
- \(\ell_{\mathrm{fine}}\) is already a posterior containing a prior.
- \(v\) concerns **any hate in a video**, not hate at a particular second.
- Median-centering a classifier logit does not make it a likelihood ratio.
- The text classifier and VLM share transcript evidence, so their contributions are dependent.

Their sum therefore risks double-counting both priors and evidence. A single \(\alpha\) cannot independently correct three differently calibrated sources. Moreover, the decomposition does **not** actually confine coarse evidence to video-level use: local coarse bits, apparently full-posterior \(P(s_t)\), and full-posterior block labels still supply coarse temporal information.

**Change/test.** Separate video-level propensity from temporal allocation explicitly. Give text, fine evidence, and coarse context distinct calibrated roles or gates. If coarse evidence is meant to be global only, enforce that restriction throughout the architecture and supervision.

**External-classifier question:** yes, the final additive text score makes this an ensemble component. That is legitimate, but should be disclosed as external supervised knowledge. Also justify applying an utterance-level Twitter hate classifier to ASR/OCR, taking their maximum, and treating missing text as neutral evidence.

### 3. High: EOC measures model sensitivity, not expected task benefit

**Why it matters.** A query that strongly changes predictions need not improve them. With a direct \(\alpha E\) pathway, acquisition may favor windows that mechanically perturb the evidence prior rather than reveal useful content.

The counterfactual weighting is also inconsistent: the sigmoid of a top-k **hate/content** logit is used as the probability of a **VLM verdict**. These are different random variables. Under the stated observation model,
\[
P(Y_w=1\mid D)=r_f+(q_f-r_f)P(H_w=1\mid D),
\]
before accounting for video-level reliability.

Masked verdict prediction only partially addresses this: forcing all negative-video targets to zero contradicts the goal of predicting actual cached verdicts when false positives exist.

**Change/test.** Separate the hate-localization head from a teacher-verdict head, and use a coherent predictive distribution for counterfactual outcomes. Define precisely whether acquisition targets information gain, uncertainty reduction, or expected prediction change.

If the policy is effectively indistinguishable from uniform querying, **do not frame intelligent acquisition as the central contribution**. Budget robustness remains a defensible contribution without that claim.

### 4. High: cross-video allocation creates a self-reinforcing acquisition curriculum

**Why it matters.** The bimodal distribution suggests persistent concentration, although bimodality alone does not prove poor allocation. Early model sensitivity determines which videos receive evidence; those videos then receive more evidence-dependent learning opportunities. Hard but initially unresponsive videos can remain permanently unexplored.

Query count and missingness also become class-correlated signals. The explicit “not queried” code can exploit this acquisition history rather than hateful content. Training permits eight fine queries for some videos, while deployment permits four; dropout does not automatically align these histories.

**Change/test.**
- Add an exploration floor, per-video caps, or diminishing-return penalties.
- Normalize acquisition utility for video duration and the definition of output change.
- Align training acquisition trajectories with deployment.
- Separate learning from observed verdict values from learning the allocation pattern.

Adaptive missingness need not invalidate likelihood estimation when appropriately conditioned on observed history; the immediate concern is coverage and policy-dependent shortcut learning.

### 5. High: sparse MIL assumptions conflict with diffuse hate and interval semantics

**Why it matters.** Top-\(T/16\) pooling assumes a small discriminative subset is sufficient. It provides little incentive to cover sustained hateful content. Conversely, an interval “any hate” verdict gives no information about how much of that interval is hateful.

Block pseudo-labels, top-k pooling, and self-selected contrastive positives can reinforce the same mistaken localization. Cross-modal attraction additionally assumes hateful evidence should align across audio/text and vision, although one modality may be neutral or contradict another.

**Change/test.** Make occupancy assumptions explicit and adaptable; consider duration-aware or coverage-sensitive objectives. Gate cross-modal agreement by modality relevance and separate externally supervised pseudo-label learning from self-selected contrastive learning.

### 6. Medium: “evidence never enters the content representation” is misleading

Evidence-conditioned attention produces
\[
z_i=\sum_j a_{ij}(e,h)V(h_j),
\]
so the resulting representation depends on evidence even when values do not directly contain it. Adding \(c(e)\) makes that dependence explicit. The final \(\alpha E\) term is direct evidence-based prediction.

**Change.** Use the narrower, accurate claim:

> “Evidence is excluded from value projections and initial content embeddings, but conditions aggregation, output calibration, and final prediction.”

Specify which tensors the “content” losses supervise. If they follow evidence-conditioned attention or calibration, they are not evidence-independent content predictions.

### 7. Medium: temporal resolution and duration assumptions are underjustified

**Why it matters.** Thirty normalized windows impose very different physical durations across videos. Normalized-time transition rates encode similar temporal structure relative to video length, not necessarily in seconds. Four frames per interval can miss brief visual events, while isolation removes context needed to distinguish endorsement, quotation, counterspeech, and target identity.

The HMM cannot recover temporal detail absent from its observations; per-second output should not imply per-second evidence resolution.

**Change/test.** Justify normalized versus physical-time dynamics, distinguish evidence resolution from output resolution, and consider duration-sensitive emissions or context-overlapping queries.

## How I would restructure the contribution story

1. **Primary:** budgeted temporal localization using cached, noisy external supervision.
2. **Core modeling contribution:** interval-aware fusion with explicit limitations under correlated teacher errors.
3. **Training contribution:** robustness to partially observed evidence and unequal query histories.
4. **Secondary components:** evidence-conditioned attention, text-score ensembling, and optional acquisition heuristics.

Avoid centering novelty on evidence-free content representations, calibrated Bayesian fusion, or superior active selection unless those claims are structurally supported. A simpler, internally consistent method would be more persuasive than additional coupled mechanisms.

## 3. 追问（我方）

```text
Thank you. Follow-up questions, please answer concretely (this is a discussion, not a new review):

1. On point 3: the masked-window loss deliberately targets hate, not the verdict: on negative videos the cached verdict can be 1 (false positive) but we set the target to 0 because the video label tells us there is no hate. So the content head is trained as a hate head using verdicts as pseudo-labels only on positive videos. You propose separating a "teacher-verdict head" from the hate head. Sketch the minimal version: what does the verdict head predict, what supervises it (on negatives the verdict itself, on positives the verdict too?), and how does the EOC use it (P(Y_w=1|D) = r + (q - r) P(H_w=1|D) with P(H_w) from the hate head?). Would you keep two counterfactual forward passes or replace EOC by the verdict head's predictive entropy / expected information gain under the HMM?

2. On point 1: with only video-level labels, what anchors could make video-level reliability identifiable? Concretely: negatives give r (false-positive rate) per video; positives give "at least one hateful segment". Would a per-video random effect on the false-positive rate only (fitted on negatives, shared prior applied to positives) be identifiable? Or should the video-level reliability be inferred from agreement between the VLM verdicts and the content model's own predictions (i.e., use the trained backbone as the anchor), accepting circularity but breaking it with a held-out scheme?

3. Given the empirical facts (HateMM: coarse-only already sufficient, more fine verdicts hurt beyond 12; HateClipSeg: monotone gain up to 34 calls; EOC ~ uniform), what is the most defensible role for adaptive querying in the paper? Is "decide per video whether fine queries are worth it at all" (a video-level stop after the 4 coarse blocks) a better framing than window selection? How would you build that decision so it is not just a validation-tuned threshold?

4. If you had to pick ONE structural simplification that removes the most reviewer attack surface without giving up the budget story, which is it (drop x(t) text term, drop cross-video allocation, drop the evidence-routed attention in favor of concatenation, drop the alpha*E prior term, drop block MIL)? Justify with respect to the ablation numbers I gave.

5. Our remaining performance gap is pooled AP on HateMM (8-call model ~1-2 points below the all-34-verdict model and below a 12-call earlier version). Which of your suggestions do you expect to actually recover AP on HateMM, versus only cleaning up the story?
```

## 4. 追问答复（原文）

## 1. Minimal verdict head—and whether to retain EOC

Your clarification is correct: **the masked-window objective is a denoising hate objective, not a verdict-prediction objective.** The inconsistency arises only when its output is interpreted as the probability of a teacher verdict.

The minimal separation is:

- **Hate head:** unchanged; negatives target zero, positives use available verdict pseudo-labels.
- **Verdict head:** one small window-pooling head predicting  
  \[
  p_w=P(Y_w=1\mid \text{content},D),
  \]
  where \(D\) contains currently available evidence.
- **Supervision:** actual cached verdicts on **both positive and negative videos**, with the target window’s verdict hidden from the inputs. A false-positive verdict remains target 1 for this head.
- **Acquisition:** use \(p_w\), not the hate score, to weight the two counterfactual outcomes.

There are **two alternative constructions**, not two equations to apply successively:

1. Directly predict \(p_w\) with the verdict head.
2. Predict interval hate probability \(u_w\), then use \(p_w=r_f+(q_f-r_f)u_w\).

The second is coherent only if \(u_w\) means “any hate in this window,” rather than an arbitrary top-k pooled score, and the emission model is credible.

**My preference:** uniform window selection for the main method. If retaining an adaptive policy, keep the two passes initially: fixing their outcome probabilities addresses a concrete defect. Verdict entropy alone is not sufficient—an unpredictable teacher can be uninformative. HMM information gain,
\[
I(Y_w;S\mid D),
\]
is conceptually cleaner, but inherits the HMM’s misspecification. With reliability variables, target information about \(S\), not merely information about teacher reliability.

## 2. Can false-positive-only random effects be identifiable?

**At the population level, potentially; at the individual positive-video level, not generally.**

For example:
\[
\operatorname{logit}r_{v,k}=\mu_k+b_v,\qquad b_v\sim\mathcal N(0,\sigma^2),
\]
with \(k\) distinguishing fine/coarse emissions.

Negative videos can inform \(\mu_k\) and the distribution of \(b_v\), under conditional-independence and exchangeability assumptions. They do **not** give a precise false-positive rate for each video, especially with few observations.

Applying that distribution to positives assumes that their false-positive tendencies follow the same distribution. Within a positive video, dense true hate and high \(b_v\) remain confounded. Fixing the learned distribution regularizes the ambiguity; it does not eliminate it. False-positive-only effects also leave video-wide teacher misses unmodeled.

**Using the backbone as an anchor:** useful as a predictive regularizer, but not automatically an identification argument. Separating its training from the videos it anchors reduces direct self-reinforcement; it does not remove shared errors or dependence induced by VLM pseudo-supervision.

A stronger content anchor would be trained without verdict inputs or verdict-derived targets, then frozen. Even that supplies an imperfect second view—not ground truth. I would describe the result as **content-informed reliability estimation**, not identifiable reliability recovery.

## 3. The defensible role of adaptive querying

Yes: **“whether to acquire fine evidence” is more defensible than “which fine window is optimal.”** Your observations suggest that marginal usefulness varies more than window-selection quality.

However, first distinguish two explanations:

- Fine evidence is genuinely redundant.
- Fine evidence is useful, but the fusion mechanism mishandles correlated observations.

A stopping rule can hide the second problem.

A concrete simplified policy:

1. Acquire four coarse verdicts.
2. A gate chooses between stopping and acquiring a small, uniformly spaced fine-query bundle.
3. Train the gate with an explicit acquisition cost:
   \[
   \mathbb E[L_{\text{video}}+\lambda\,n_{\text{fine}}],
   \]
   or a fixed expected-budget constraint, using exploration to learn the consequences of both actions.

This makes the decision a learned cost-sensitive policy rather than a threshold on arbitrary output movement. All revealed outcomes used to train it must count toward the acquisition budget.

**Limitation:** video-level supervision teaches usefulness for video classification, not necessarily localization. There is no assumption-free way to learn true localization value from video labels alone. If that distinction cannot be addressed, present stopping as a pragmatic budget controller—not optimal temporal information acquisition.

## 4. One simplification: drop evidence-routed attention

**Replace it with straightforward evidence concatenation, retaining the budget protocol and interval processing.**

Your reported incremental advantage over concatenation is only approximately **0.5–0.7 points**, whereas routing introduces:

- evidence-conditioned queries and keys;
- key-specific biases;
- global calibration;
- difficult “content-only” claims;
- additional interactions with pseudo-label losses.

That is substantial conceptual machinery for a small stated incremental benefit. The larger losses from removing individual routing components do not establish that this machinery is necessary when a simpler fusion design remains available.

The numbers come from an earlier revision, so they cannot guarantee the same trade-off in 4b. Nevertheless, this is the strongest simplification on the evidence you supplied. Dropping cross-video allocation would be my second choice.

## 5. What might actually recover HateMM performance?

**I would prioritize fusion robustness, not smarter window selection.**

### Most plausible: temper correlated evidence and separate fusion weights

Replace the shared sum and scale with something like
\[
z_t=z_{\text{content},t}
+a_f\widetilde{\ell}_{\text{fine},t}
+a_xx_t+a_vv,
\]
where fine evidence has bounded or diminishing cumulative influence.

This allows the model to trust coarse video context without equally trusting fine verdicts or the text classifier. Mechanistically, it addresses both video-wide false-positive shifts and excessive confidence as more verdicts arrive.

### Plausible: stop turning uncertain block posteriors into strong supervision

Full-posterior block labels can spread teacher errors into the content predictor. Temper their confidence or remove block MIL if its main effect is to reproduce already-injected evidence. This could help, but could also remove useful supervision.

### Mostly cleanup, not a likely recovery mechanism

- A separate verdict head: improves acquisition coherence, but offers limited leverage when selection itself contributes little.
- Replacing routing with concatenation: primarily simplification.
- Dropping cross-video allocation: primarily removes curriculum and missingness complications.
- A more elaborate stop policy: unlikely to recover missing information when coarse-only already dominates its decisions.

**Bottom line:** independently weighted, correlation-aware fusion is my strongest mechanistic bet. Nothing in the description establishes that the remaining gap is recoverable without additional observations; I would not promise that a cleaner acquisition policy will close it.
