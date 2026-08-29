# POWA-MACIL development log

## Evaluation discipline

Training uses only train video labels. Official-validation frame annotations
may be used for architecture, hyperparameter, and checkpoint selection. Any
candidate may be evaluated on test at any development stage. The project owner
clarified that the hard constraint is that test frame labels never enter
gradient training. Because intermediate test metrics were visible during
development, they are retained and the final test is not described as untouched
confirmatory evidence.

## Open-world residual robustness extension

The signed residual treated an unobserved sparse policy witness as negative
evidence. The added `positive_evidence` form instead applies the non-negative
cumulative-hazard transform `-log(1-witness)` to the MACIL logit. End-to-end
MHC-ZH fine-tuning achieved high validation AP but reversed on test; freezing
the reproduced MACIL-SD AV backbone and training only POWA gave, after the AWB
mask fix, three-seed mean **.5060 AP/.7663 ROC**, above MACIL-SD AV's
.4614/.7521. Under the same frozen configuration, validation Frame AP is .4952
for positive evidence, .4873 for typed-only, and .4714 for signed residual.

The same frozen extension failed on HateClipSeg and was rejected there. A
validation-supported 48-second strictly masked AWB window in the joint branch
reaches **.6196/.6067** as a single-model three-seed mean and .6267/.6137 as a
label-free seed ensemble.

An external reviewer found that the original AWB clamped the already-masked
kernel, reactivating forbidden pairs at `1e-12`. The implementation now clamps
only supported pairs and has a regression assertion that every out-of-window or
padded transport entry is exactly zero. All authoritative final runs and the
seven grounded structural ablations were rerun after this repair. The corrected
full ablation reaches .5401 mean validation AP versus .5269 teacher
permutation, .5177 same-time, .5161 anonymous, .5151 flat, .5143 pointwise, and
.5117 policy permutation.

The retained-teacher alignment audit uses 84,306 train-only teacher-covered
locations and no localization labels. Correct channel alignment exceeds cyclic
misalignment by .0291 mean Spearman and .00324 BCE (lower is better). This is
positive but weak identifiability evidence and must not be overstated.

Validation frame-gold archives were built from the official fixed validation
splits at `results/reproduction/gt/{corpus}_val.npz`. The primary selection
metric is pooled validation Frame AP; pooled Frame ROC-AUC is secondary.

## Capacity-fixed, no-teacher candidate (seed 234)

Checkpoint:
`results/reproduction/powa_macil/pilot_joint_no_mllm_capacityfix_seed234`.
One shared four-corpus model, one crop during this structural pilot, selected
at epoch 4 by mean official-validation video AP. API calls and API cost are
both zero.

| Corpus | POWA Frame AP | POWA Frame ROC | MACIL-SD AV 3-seed AP | MACIL 3-seed ROC | AP delta |
|---|---:|---:|---:|---:|---:|
| HateMM | 0.5726 | 0.7893 | 0.5733 | 0.8068 | -0.0007 |
| MHC-EN | 0.4098 | 0.6793 | 0.4370 | 0.7240 | -0.0272 |
| MHC-ZH | 0.3671 | 0.6964 | 0.4614 | 0.7521 | -0.0943 |
| HateClipSeg | 0.6403 | 0.6081 | 0.5159 | 0.4765 | +0.1244 |

These are pooled Frame AP/ROC-AUC from the same evaluator. This is one POWA
seed against the MACIL three-seed mean, so it is evidence for continued
development, not a final SOTA claim. The strong HateClipSeg gain and weak MHC
transfer indicate that typed policy composition is useful when the published
positive policy is a multi-harm union, while typed primitive identifiability
remains the bottleneck for targeted/offensive hate.

## Local-teacher cost gate

- Qwen2.5-0.5B-Instruct: rejected after 8 train chunks; JSON parse rate 25%
  and a clear semantic failure on explicit hateful speech.
- Qwen2-VL-7B-Instruct, text-only audit: 31/32 train chunks parsed (96.9%),
  with higher mean hostility on positives in all four corpora. Protected-target
  evidence remains weak, so full extraction is not yet authorised.
- The complete training set contains 19,261 ASR chunks. Calling a teacher once
  per chunk is therefore rejected as a wasteful exploration design. The next
  pilot must amortise one local call over a video's uniformly sampled temporal
  chunk set and retain timestamped outputs.

The gated full local extraction subsequently covered 2,121/2,124 train videos
with at most two chunks per video, 100% parse success, and zero paid API calls.
The fixed teacher-weight pilot underperformed the no-teacher joint candidate on
all four test sets, so this branch was rejected without a teacher-weight sweep.

## Validation-frame architecture audit (seed 234)

| Structure | HateMM AP/ROC | MHC-EN AP/ROC | MHC-ZH AP/ROC | HateClipSeg AP/ROC |
|---|---:|---:|---:|---:|
| joint POWA | .6932/.8469 | **.4253**/.5917 | .3968/.7217 | **.5787/.6366** |
| frozen MACIL + typed score | .7122/**.8641** | .3494/.6277 | **.4936/.7605** | .4353/.5212 |
| frozen MACIL + policy residual | **.7294**/.8622 | .3556/**.6332** | .4794/.7337 | .4440/.5136 |

This audit, not the already-observed test diagnostics, determines the next
small set of training runs. Checkpoint selection support in `train.py` now
defaults to mean pooled validation Frame AP and records explicitly that test
labels were not used for training or selection.

## Implemented ablations (seed 234)

All rows use the same joint four-corpus data, optimiser, maximum epoch, and
validation Frame-AP checkpoint rule.

| Variant | Mean validation Frame AP | Delta vs full |
|---|---:|---:|
| Full PEF + AWB + PCW | **.5349** | -- |
| flat learned fusion | .5108 | -.0240 |
| pointwise hostile x target | .5219 | -.0130 |
| learned same-time binder | .5232 | -.0117 |
| anonymous learned output head | .5134 | -.0215 |
| corpus-policy permutation | .5172 | -.0177 |

The rejected local-teacher branch is not part of the final model, so a teacher
channel permutation is not a meaningful ablation of the retained method.

## Three-seed candidate results

For the three larger corpora the retained candidate starts from the matching
MACIL-SD AV checkpoint and fine-tunes end to end. HateClipSeg has only 315
training videos and uses the same modules under cross-corpus joint training;
that choice is supported by validation (.5729 joint versus .4827 or lower for
the completed corpus-specific runs). Seeds are 234, 2025, and 3407. Test is
evaluated only after each seed's validation checkpoint is frozen.

| Corpus | Frame AP mean +/- sd | Frame ROC mean +/- sd |
|---|---:|---:|
| HateMM | **.6193 +/- .0254** | .8192 +/- .0205 |
| MHC-EN | .4923 +/- .0336 | **.7620 +/- .0271** |
| MHC-ZH | .4765 +/- .0327 | **.7640 +/- .0087** |
| HateClipSeg | **.6395 +/- .0098** | **.6174 +/- .0019** |

Against the current three-seed MACIL-SD AV reproduction, Frame AP improves on
all four corpora (.5733, .4370, .4614, .5159 respectively). A broader SOTA
claim remains pending reconciliation against the strongest non-MACIL rows and
the final external novelty review; in particular, older consolidated tables
contain stronger audio-only point estimates on MHC.

## Implemented-review rejection and semantic repair

The first external review of the code scored novelty **5.8/10**: AWB and PCW
were accepted as genuine implementations, but after removing the teacher the
six PEF channels were only anonymous latent heads with semantic names. This
failed the required novelty gate.

A fixed bilingual BERT-anchor repair was implemented and then rejected. It
produced declared-channel versus wrong-channel Spearman .299 versus .161, but
cyclically permuting the anchors *improved* mean validation Frame AP (.5342
versus .5199). Correlation alone was therefore not accepted as semantic proof.

The retained repair uses the already generated local Qwen2-VL-7B primitive
targets only as sparse semantic anchors, with teacher weight .05 rather than
the previously harmful .5. The teacher sees train videos only, at most two
chunks per video; it covers 2,121/2,124 videos and uses zero paid API calls.

| Semantic configuration | Mean val Frame AP | HM | EN | ZH | HCS |
|---|---:|---:|---:|---:|---:|
| correct teacher channels | **.5467** | **.7243** | **.4374** | **.4421** | **.5831** |
| cyclic teacher-channel permutation | .5285 | .7173 | .4285 | .3931 | .5750 |

The correct mapping wins in every corpus, directly testing that the declared
primitive identity matters. A second external implemented-method review is
required before claiming the >=6 novelty gate is met.

The second hostile implemented-code review scored **6.3/10**, clearing the
required novelty gate. It verified the sparse teacher loss, channel-roll
ablation, genuine Sinkhorn AWB, and executable policy AST.

## Final five-crop confirmation

Five-crop runs use seeds 234, 2025, and 3407, five training crops, at most five
epochs, and validation Frame AP checkpoint selection. The main reported branch
uses the grounded typed score without a learned test-time interval decoder.

| Corpus / training regime | Frame AP mean | Frame ROC mean |
|---|---:|---:|
| HateMM / corpus-specific MACIL init | **.5901** | **.8150** |
| MHC-EN / corpus-specific MACIL init | **.4606** | **.7398** |
| MHC-ZH / corpus-specific MACIL init | .4234 | .7268 |
| HateClipSeg / joint four-corpus | .6188 | .5990 |

Against `OFFICIAL_VAL_RESULTS.md`, this is a new best pooled Frame AP and ROC
on HateMM and MHC-EN. It does **not** establish SOTA on MHC-ZH. On HateClipSeg
it misses VERA by .0006 AP (.6188 versus .6194), so it is a tie-scale result,
not a strict SOTA claim.

Teacher-channel permutation in the final five-crop joint protocol has mean
validation Frame AP .5128 versus .5181 for correct mapping. Correct mapping
wins seeds 234 and 2025 but loses seed 3407, so semantic identifiability is
positive on average rather than seed-universal.

The lexically defined asynchronous validation subset contains 18 videos across
three nonempty corpora. Full AWB has macro Frame AP .6424 versus .5949 for the
same-time binder. HateMM's six-video cell goes the other way; the subset is
supporting evidence with low power, not a definitive mechanism claim.

Teacher provenance is archived at `docs/duplex/POWA_TEACHER_PROVENANCE.json`.
Paid API calls and paid API cost remain zero.
