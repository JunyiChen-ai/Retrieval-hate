# SYNTH_PAIR_PROBE -- RESULT: TRANSFER FAILURE, direction closed

Freeze: `idea-stage/SYNTH_PAIR_PROBE_FREEZE.md`, committed as `73971c1` before any
training. Single frozen run, 2026-08-17. Raw numbers: `idea-stage/synth_pair_probe.json`.
Log: `logging/runs/synth_pair/run.log`. Zero API cost, ~4 min wall clock on one RTX 5090.

## 1. What was run

CONAN is not on this machine (exhaustive filesystem search found nothing; the
`MM_STANCE_LIT_RECON.md` §7 claim that "we already hold the CONAN family" is not backed by
any file). The pre-registered fallback was used: source sentences from
`data/gt/<ds>/train.jsonl` rows with `label == 1` only.

* 14,396 candidate sentences survived extraction (HateMM 2,634 / MHC 624 / MHC_zh 380 /
  ImpliHateVid 10,758); capped at the frozen 2,600.
* **2,600 pairs = 5,200 synthetic examples** (2,539 `en`, 61 `zh`). Each pair shares one
  source sentence and differs only in the frame, so topic and slur content cannot be the
  discriminative feature.
* 26 EN + 20 ZH frame templates (§2 of the freeze). All text, synthetic and real,
  passed through the same ASR-form normaliser: punctuation and symbols deleted,
  lowercased, whitespace collapsed. Every quotation mark is gone; attribution survives
  only as lexical cues.
* Split 85/15 by source sentence: 4,420 train / 780 synthetic-dev examples.

**Test-set handling.** `test.jsonl` was opened once, at scoring time, by an already-frozen
classifier. No test row touched pair construction, template choice, training, model
selection or thresholding. A prior smoke run exercised every code path with fake
transcripts stitched from the synthetic dev split and never opened `test.jsonl`.

## 2. Synthetic sanity check (not evidence of anything)

| tier | synthetic-dev accuracy |
|---|---|
| A: multilingual-mpnet embeddings + logistic regression | **0.981** |
| B: distilbert-base-multilingual-cased fine-tune (2 ep) | **1.000** |
| length-only control | 0.496 |

The templates are trivially learnable, as expected, and the length control at chance
confirms the two template families are not separable by length alone.

## 3. Primary evaluation -- GOLD_VOICE, n = 37 (21 OWN / 16 NOT_OWN)

Mean-of-chunks aggregation, the frozen primary. Positive class = `NOT_OWN`.

| tier | AUC | acc@0.5 | recall NOT_OWN | recall OWN | mean P(NOT_OWN) on gold-NOT_OWN | on gold-OWN |
|---|---|---|---|---|---|---|
| **A mpnet + logreg** | **0.441** | 0.486 | 0.188 | 0.714 | 0.329 | 0.393 |
| **B distilbert FT** | **0.467** | 0.568 | 0.000 | 1.000 | 0.0020 | 0.0033 |
| length-only control | 0.354 | 0.568 | 0.000 | 1.000 | 0.447 | 0.456 |

Frozen sensitivity aggregations (reported, not decision-bearing) do not change the picture:
Tier A max-of-chunks 0.438, first-256-tokens 0.423; Tier B 0.485 and 0.488.

Two facts beyond the bar:

* **Every tier and every aggregation is at or below 0.50, and the sign is consistently
  inverted**: mean `P(NOT_OWN)` is *higher* on the gold-`OWN` items than on the gold-
  `NOT_OWN` items, in 6 of 6 tier x aggregation cells. This is the direction that risk
  R1 (`2404.01651` Table 7) predicted, now observed at the item level rather than only on
  counter-speech.
* **Tier B collapses to a single class on real transcripts.** After reaching 1.000 on
  synthetic dev it assigns `P(NOT_OWN) ~ 0.002-0.003` to every real transcript -- it calls
  all 37 items `OWN`. Its 0.568 accuracy is exactly the majority-class rate (21/37). The
  fine-tune learned the template surface and nothing that survives the domain shift.

## 4. Secondary -- S_FP / S_FN stratification (primary aggregation)

| stratum | n (OWN / NOT_OWN) | Tier A AUC | Tier A acc | Tier B AUC | Tier B acc |
|---|---|---|---|---|---|
| S_FP | 24 (14 / 10) | 0.443 | 0.458 | 0.429 | 0.583 |
| S_FN | 13 (7 / 6) | 0.429 | 0.538 | 0.595 | 0.538 |

No stratum reaches the bar. The single value above 0.50, Tier B on S_FN (0.595, n = 13),
comes from a model that emitted one class for every item, so its "AUC" is ranking noise
inside a ~0.002-wide probability band, not a usable separation.

## 5. Wrong-sign risk check (R1)

**Frozen marker lexicon prevalence on real transcripts is the headline here.** Of the 99
evaluation transcripts, **only 10 contain any attribution marker at all**: 1 of 21
gold-`OWN`, 3 of 16 gold-`NOT_OWN`, and 6 of 50 controls. Risk R2 (transfer) is therefore
the binding constraint: the lexical attribution cue that the synthetic pairs teach is
almost absent from ASR transcripts, so there is very little for the classifier to key on
even in principle.

**(a) Grounded trap cases (gold `OWN` with a marker present): n = 1.** Both tiers get it
right, but n = 1 is not evidence in either direction. Gold `OWN` without a marker: n = 20,
Tier A accuracy 0.70, Tier B accuracy 1.00 (trivially, it says `OWN` everywhere).

**(b) 50 controls, marker-driven shift in mean P(NOT_OWN):**

| stratum | n | Tier A mean P | Tier B mean P |
|---|---|---|---|
| CTRL_HATE, marker | 3 | 0.470 | 0.157 |
| CTRL_HATE, no marker | 22 | 0.222 | 0.009 |
| CTRL_HATE shift | | **+0.248** | **+0.149** |
| CTRL_NONHATE, marker | 3 | 0.298 | 0.001 |
| CTRL_NONHATE, no marker | 22 | 0.333 | 0.088 |
| CTRL_NONHATE shift | | -0.036 | -0.087 |

The `CTRL_HATE` shift clears the frozen 0.15 threshold in both tiers, i.e. marker presence
does move the classifier toward `NOT_OWN` independently of the true source -- but the
marker stratum is n = 3 on each side, so this is recorded as directionally consistent with
R1 and not as a measured effect. The `CTRL_NONHATE` shift, the stratum the freeze
nominated, is negative in both tiers, so the specific counter-speech wrong-sign pattern of
`2404.01651` is **not** reproduced here; what is reproduced is the item-level sign
inversion in §3.

## 6. Verdict

Frozen rule: SIGNAL at AUC >= 0.70, WEAK at 0.60-0.70, TRANSFER FAILURE below 0.60, on
the better of the two pre-registered tiers, primary aggregation, GOLD_VOICE n = 37.

Best tier AUC = **0.467** (Tier B). Tier A = 0.441.

**TRANSFER FAILURE. Direction closed.**

Rule-synthesised quoting/self-utterance pairs produce a classifier that is essentially
perfect on its own synthetic distribution (0.98-1.00) and worse than a coin flip on real
ASR transcripts. The failure is not marginal and not an artefact of the aggregation
choice: all three aggregations, both tiers, and both S-bucket strata land at or below
chance, with a consistently inverted sign. The mechanism is visible in §5: the written-form
attribution cues the synthetic frames rely on appear in only 10 of 99 real transcripts, so
there is no cue for the transfer to carry.

## 7. Caveats recorded, none of which would flip the verdict

* Chinese coverage in the synthetic set is thin -- 61 of 2,600 pairs -- because MHC_zh
  contributes only 380 of 14,396 candidate sentences and the frozen cap samples
  proportionally. 7 of the 37 primary items are MHC_zh. Removing them cannot lift a 0.44
  / 0.47 AUC to 0.70.
* n = 37 is small; the 95% interval on an AUC of 0.47 comfortably includes 0.60. But the
  probe was pre-registered to make a decision at this n, the point estimate is below
  chance rather than merely below the bar, and every secondary cut agrees.
* Only one fine-tune seed was run, per the freeze. Tier B's degenerate single-class output
  on real data is a domain-shift symptom, not a seed accident -- Tier A, which shares no
  weights and no training procedure, fails in the same direction.
