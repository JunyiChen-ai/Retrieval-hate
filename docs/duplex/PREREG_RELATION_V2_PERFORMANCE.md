# Relation-V2 performance-stage protocol

Frozen before Relation-V2 implementation or results.

## Problems this stage must remove

1. No closed six-class benchmark-derived ontology as the method definition.
2. No hand-written corpus policy AST as the scoring function.
3. Temporal relation modelling must be the centre of the model, not a
   target/hostility-only side branch.
4. No cross-corpus training. Every corpus is an independent experiment.

## Non-negotiable data protocol

For corpus `C`:

- optimization reads only `C/train` features and video labels;
- architecture, scalar hyperparameters, epoch and checkpoint use only
  `C/validation`;
- evaluation uses `C/test`;
- no train sample, label, teacher target, checkpoint or learned parameter from
  another corpus is allowed;
- the reproduced corpus-specific MACIL-SD AV initialization is allowed because
  it was itself trained only on `C/train`;
- test may be evaluated during development as requested by the project owner,
  but test frame labels never enter gradient training;
- final primary numbers are mean and sample standard deviation over seeds 234,
  2025 and 3407, using one model per seed. Seed ensembles are diagnostic only.

Every checkpoint must archive exact train/validation IDs and assert corpus
purity. A runner accepting more than one corpus is invalid for this stage.

## Evaluator and targets

The frozen evaluator reports pooled 1-fps Frame AP (primary) and pooled Frame
ROC-AUC (secondary), matching the existing MultiHateLoc/LELA-style table.

| Corpus | Current fair AP bar | Required AP | Current ROC bar |
|---|---:|---:|---:|
| HateMM | .5733 | >= .6033 | .8068 |
| MHC-EN | .4519 | >= .4819 | .7272 |
| MHC-ZH | .4614 | >= .4914 | .7521 |
| HateClipSeg | .6194 | >= .6494 | .6050 |

The desired margin is 3--4 absolute AP points on every corpus. A candidate
cannot pass by sacrificing ROC below the current bar.

## Working order

1. Implement and independently audit the performance model before full runs.
2. Select a small number of configurations using validation only.
3. Reach the performance target under the independent-corpus protocol.
4. Only then replace remaining heuristic components with an open-vocabulary,
   policy-conditioned relation mechanism and request novelty review.

Performance alone is not completion. Final completion additionally requires an
external reviewer to assign a strong-accept novelty verdict to the implemented,
audited final method.
