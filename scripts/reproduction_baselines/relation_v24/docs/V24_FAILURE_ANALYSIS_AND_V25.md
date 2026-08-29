# V24 failure analysis and the single recommended V25 mechanism

Scope: read-only analysis of the frozen THVL train weak-label bags and the three
existing seed-234 checkpoints. No test data, temporal ground truth, or new training
trial was opened/run.

## Conclusion

V24 fails for a structural reason: `exact_center` deletes the strongest weak-label
signal before MIL. Raw text/MM levels separate positive and negative bags, whereas
the centered LME mostly measures within-video dispersion. The train objective can
then improve equally well by calibrating the unchanged V16 global score; the
permuted-local arm retains that route. Consequently, “real beats permuted” is not
identified, and validation rationally falls back to epoch 0.

The only priority V25 mechanism should be **negative-reference density-ratio MIL**:
learn/calibrate an instance's evidence relative to frames from negative training
bags, then use a cardinality-stable sparse MIL likelihood on positive bags. Do not
center away the cross-video reference.

## Frozen-data diagnostics

- 314 train videos: 93 positive, 221 negative. Bag length mean 12.69 windows
  (negative 12.29, positive 13.66); median 10 and 90th percentile 21.
- V16 global: AP 0.4849, ROC 0.6909. Mean score is -4.49 for negative bags and
  -2.59 for positive bags.
- Raw text bag mean: AP 0.5814 / ROC 0.7771; raw text maximum:
  0.5555 / 0.7712. Raw MM mean: 0.5792 / 0.7727; raw MM maximum:
  0.6158 / 0.7841.
- After per-video centering, text `max(centered)` drops to AP 0.3997 / ROC
  0.6337; MM drops to 0.4471 / 0.6875. Centered LME gives text
  0.3951 / 0.6380 and MM 0.4446 / 0.6938.
- Positive bags do have more variation, but that is weak localization evidence:
  text within-video variance averages 120.89 vs 64.42; MM 69.47 vs 34.90.
- Text and MM are redundant rather than complementary: pooled Pearson 0.827,
  Spearman 0.688; median within-video Spearman 0.693.
- A fixed, train-only diagnostic (not a selected result) mapping each local score
  through its modality's negative-bag empirical CDF gives AP/ROC 0.5934/0.7808
  with max pooling and 0.6146/0.7958 with top-2 pooling. This establishes that
  negative bags provide a useful absolute instance reference that V24 discards.

## Optimizer/checkpoint evidence

At epoch 0 the full-batch gradients are: global scale delta `+0.7317`, bias
`-0.1762`, gamma `-2.4313`, and exactly zero for both family logits. The family
weights cannot learn initially because gamma is zero. After one epoch the global
delta hits its lower guardrail (-0.5), while gamma is only 0.0534. By epoch 5,
gamma is 0.0343 and the family softmax has drifted to text/MM = 0.112/0.888.

Real-arm train AP across epochs 0..5 is 0.4849, 0.4974, 0.4889, 0.5035,
0.5039, 0.4928. Permuted-arm AP is 0.4849, 0.4962, 0.4870, 0.4949,
0.4992, 0.4849. Their near-identical early gains are explained by global
recalibration, not temporal evidence. The matched global-only arm preserves AP
0.4849 at every epoch (affine calibration preserves ranking) while lowering BCE.
Thus train BCE/AP does not certify instance localization, and a validation
real=epoch0 decision is expected when the required real-over-permuted margin is
not stable.

## V25: negative-reference density-ratio MIL

For each modality, fit a monotone empirical reference distribution using only
instances from negative bags in the same corpus's training split. Convert every
instance score to a negative-tail surprisal (or a regularized log density ratio).
A shared small monotone family calibrator combines text/MM evidence. Train it with:

1. an all-negative instance likelihood for negative bags;
2. a cardinality-stable noisy-OR/top-k latent MIL likelihood for positive bags;
3. sparse responsibility and temporal smoothness regularization; and
4. the unchanged V16 global branch as an exact epoch-0/failure fallback.

The output frame score is the learned instance posterior/responsibility. The video
score is its cardinality-stable MIL aggregation plus the separately calibrated
global branch. This creates the missing bridge: negative weak labels identify what
background instances look like, while positive weak labels identify at-least-one
departures from that reference. It uses no semantic class list, no mixed-dataset
training, and the identical mechanism applies to each of the four corpora with
train-only reference fitting, val-only selection, and frozen test evaluation.

## Minimal falsification experiment

One seed (234), five epochs, no sweep. Compare exactly four preregistered arms on
one corpus first: epoch-0 global, matched global-only, real V25, and length-bucket
permuted-positive-bag V25. Validation must select the epoch and pass all of:

- real video AP gain over epoch 0 >= 0.5 point;
- real video AP gain over permuted >= 0.5 point;
- gamma/responsibility mass is nonzero without one modality weight exceeding .95;
- mixed-video within-video macro ROC > 0.5 and improves over the global prior by
  >= 1 point; and
- time shuffle removes at least 80% of the within-video gain.

If any gate fails, stop V25 rather than expanding trials. Only after this gate
should the same frozen protocol be run independently on the other corpora.

## Limitations

The train-only AP/ROC figures above use weak video labels and do not prove temporal
localization. The negative-reference diagnostic is deliberately in-sample and is
mechanism evidence, not a reportable validation result. Sparse-positive MIL is
also an assumption; diffuse harmful content may require a soft cardinality prior,
which must be fixed on validation rather than inferred from test behavior.

---

# V25 executable protocol addendum (preregister before implementation)

This section replaces every `noisy-OR/top-k` alternative above with one executable
choice. There is no algorithm choice left for a run.

## Frozen constants and split discipline

Each corpus is trained independently. Only its train video labels are used for
optimization/reference construction. Validation video labels select an epoch;
the validation temporal steward evaluates the localization gates. Test is unopened
until the complete configuration is frozen and all validation gates pass.

Constants: seeds `(234, 2025, 3407)`, epochs `e=0..5` (epoch 0 is the exact V16
fallback), `rho=0.20`, aggregation temperature `tau=0.5`, ECDF clipping
`epsilon=1e-4`, no score smoothing, and bootstrap seed `25025` with 2,000
video-cluster replicates.

At epoch 0, and whenever any activation gate fails, the only legal frame output
for video `i` is the constant `score_it=G_i` at every covered second. Its
within-video ROC is exactly 0.5 by the all-ties convention. An unselected or
untrained local posterior must not be emitted, evaluated, or reported.

## Negative reference and instance inference score

For family `f` and an instance raw score `x`, let `R_f` be all instances from
negative **training videos**. Define the mid-rank empirical CDF

`F_f(x) = (#{r in R_f: r < x} + 0.5 #{r in R_f: r = x} + 0.5) / (|R_f| + 1)`

and `z_f(x)=logit(clamp(F_f(x), epsilon, 1-epsilon))`.

Training uses deterministic five-fold video cross-fitting, with
`fold(video_id)=int(SHA256(video_id)[:8],16) mod 5`. Every train video is transformed
using negative reference instances outside its fold. Validation and eventual test
use the single full negative-train reference, frozen before validation. The
preregistered sensitivity report recomputes train predictions using the full
reference and reports mean/max absolute posterior change; it cannot select a model.
Preflight requires every out-of-fold negative reference to be nonempty and all raw
and transformed values finite. The full negative-train reference is materialized
exactly once in canonical order and frozen with per-family SHA-256 plus an aggregate
manifest hash; validation/test may only load that artifact.

The learnable, label-independent instance logit is

`ell_it = b + s * [w_text z_text(x_it) + w_mm z_mm(x_it)]`,

where `s=softplus(s_raw)>=0` and `w=softmax(w_raw)` over the two available families.
If a corpus has a preregistered missing family, its availability mask is fixed for
all splits and weights are renormalized over available families. The inference
output is `p_it=sigmoid(ell_it)` for every valid local window. It depends only on
the local evidence and frozen train reference. MIL responsibility is computed only
inside the train loss and is never reported or used as the inference output.

## The only bag aggregator

For a bag with `T>=1` valid windows, sort logits descending as
`ell_(1)>=...>=ell_(T)`, set `q=rho*T`, `m=floor(q)`, and `a=q-m`. The unique local
bag logit is the fractional top-proportion log-mean-exp

`A(ell)=tau*log([sum_{j=1}^m exp(ell_(j)/tau) + a*exp(ell_(m+1)/tau)]/q)`.

When `m=0`, this reduces exactly to `ell_(1)`. This empirical top-tail integral is
exactly invariant when the whole bag is replicated any integer number of times;
the fractional boundary removes the usual `ceil(rho*T)` artifact. A zero-valid-
window bag is invalid and fails preflight. There is no alternative K, pooling
function, temperature, mask policy, or smoothing candidate.

The real video logit is

`L_video = (1+delta) G + c + gamma A(ell)`,

with `delta in [-.5,.5]`, `c in [-2,2]`, `gamma in [0,2]`. Epoch 0 fixes
`delta=c=gamma=0`, so `L_video=G` exactly. For a positive bag, train-only latent
responsibilities are the normalized exponentials in the same fractional top-tail
(the boundary instance has weight `a`) at temperature `tau`; negative bags treat
every valid instance as negative. The complete per-video loss is

`Loss_i = BCEWithLogits(L_video_i,y_i)
 + 1[y_i=0]*(1/T_i)*sum_t BCEWithLogits(ell_it,0)
 + 1[y_i=1]*1e-3*H(r_i)
 + 1e-3*(1/max(1,T_i-1))*sum_t |p_i,t+1-p_it|`,

where the last term is defined as zero for `T_i=1` and
`H(r)=-sum r log(r+1e-12)`. Thus `lambda_neg=1.0`, `lambda_H=1e-3`, and
`lambda_TV=1e-3`. The total objective is the arithmetic mean of `Loss_i` over
videos: negative instance loss is first averaged within each bag, then across
videos. These are all loss terms and coefficients; none is selectable.

## Four required arms

All arms use identical IDs, splits, references, seeds and epoch accounting.

1. `real`: the model above.
2. `permuted`: identical optimization, but local evidence bags are reassigned to
   target videos before training. Global `G`, video label and ID remain attached
   to the target. Reference distributions are the frozen references constructed
   from the unpermuted real negative training videos.
3. `negative_reference_only`: a matched control with the same
   `(1+delta)G+c+gamma*A(ell)`, loss, optimizer and epochs. Its local mapping is
   frozen at `b=0,s=1,w_text=w_mm=0.5` (renormalized only by the preregistered
   availability mask); only `delta,c,gamma` train. The real arm's sole additional
   capacity is learned local calibration/family weighting under MIL.
4. `global_only`: learns only `delta,c`; `gamma=0` and family parameters remain at
   their initialization.

Permutation buckets are preregistered as exact-length buckets `bucket(T)=T`.
Within each exact-length `T` group containing `n>=2` videos, sorted video IDs are
cyclically shifted by `1 + seed mod (n-1)`. A singleton keeps its own local bag and
is explicitly counted as an unpermutable singleton (it is not silently excluded).
Thus every non-singleton target receives one whole donor bag of exactly the same
length. The producer writes pre/post ID sets, donor mapping, per-video lengths,
singleton IDs, total instance counts and their hashes, proving IDs/counts/lengths
unchanged. Only local evidence moves; global evidence and labels never move.
The manifest reports fractions of scoped videos and instances assigned to a
different donor. If either fraction is below 0.80, the permutation is
unidentifiable and V25 activation fails.

## Window-to-1 Hz output and shuffle

The time grid is `j=0..ceil(duration)-1`, representing second center
`u_j=min(j+0.5,duration-1e-9)`. A valid window covers `u` iff
`start <= u < end`, except the final window also includes `u=end=duration`.
If multiple windows cover a second, average their **instance logits** and then
apply sigmoid once. Seconds with no covering window receive `mask=0` and score
`NaN`; they are excluded from every metric and the coverage fraction is reported.
Coverage below 1.0 is a preflight failure for the fixed 30-second producer. No
boundary interpolation or temporal smoothing is used.

The validation shuffle independently permutes local window logits within each
video using seed `25026 + model_seed`; global scores, labels, masks, durations and window/time
coordinates remain fixed. It never shuffles 1 Hz labels or global evidence.
For seed `s`, define `D_s=ROC_within(real_s)-0.5` and
`Dshuf_s=ROC_within(shuffled_s)-0.5`. The exact removal ratio is
`R_remove=1-(mean_s Dshuf_s)/(mean_s D_s)`, with all three matched seeds averaged
before division. It is undefined and fails when `mean_s D_s<=0`; it is not clipped.

## Epoch selection, uncertainty and gates

For each arm/seed, save epochs 0..5. Compute the mean validation video AP across
the three matched seeds for every real epoch. Select the real epoch maximizing
`(mean AP, mean ROC, -epoch)` lexicographically. Freeze that one shared epoch for
all three seeds. The permuted, negative-reference-only and global-only arms are
evaluated at the same epoch. No control selects its own favorable epoch.

All reported method-minus-control intervals use paired video-cluster bootstrap:
the same 2,000 resampled validation video-index vectors are applied to both methods
and all matched seeds, then seed means are differenced. Invalid one-class ROC
replicates are omitted with their count reported. The real arm must pass every gate:

- mean video AP is at least epoch-0 AP + 0.005 and the paired 95% CI lower bound
  for the difference is greater than 0;
- mean video AP exceeds both permuted and fixed-negative-reference-only by 0.005,
  with both paired 95% CI lower bounds greater than 0;
- mean video ROC is no worse than epoch 0 by 0.005;
- effective `gamma>=0.01` and maximum family weight `<=0.95` in every seed;
- steward-computed mixed-video within-video macro ROC is >0.5 and exceeds the
  global prior by at least 0.01, with paired 95% CI lower bound >0; and
- the exact three-seed `R_remove` is at least 0.80.

Any failed gate freezes epoch 0 as the conclusion and stops expansion. Test has no
role in reference fitting, epoch choice, thresholds, gates, or mechanism revision;
it is evaluated once only after a signed final validation decision.
