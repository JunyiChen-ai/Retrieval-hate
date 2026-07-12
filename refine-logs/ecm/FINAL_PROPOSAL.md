# ECM-RGCL Final Disposition: RETHINK / ABANDONED

## Immutable objective

The project still requires a meaningful, novel and causally removable train-only MLLM integration that improves the unchanged ordinary full-video RGCL kNN by at least `+0.030` accuracy and `+0.030` macro-F1 on MHC-EN and MHC-ZH, paired seeds `0/1/2`, with all signs positive, hierarchical bootstrap/Holm significance, REMOVE and SHUFFLE attribution, and no protocol change.

The parent-video binary label is the only gold supervision. No segment, timestamp, span, stance, target, mechanism, rationale or localization gold exists. Any MLLM output is a weak/privileged train-only pseudo-signal. Validation/test must contain no MLLM, mode, teacher cache or extra head.

## Frozen ECM idea evaluated

ECM proposed to run a frozen MLLM on every train video's strict-OOF, no-direct-outcome-field whole-video prediction trace and emit confidence-bearing posteriors over:

- presentation/context inversion;
- target binding;
- modality conflict;
- surface shortcut;
- evidence dilution;
- undiagnosed.

The modes would form class-balanced retrieval risks and a projected/minimax update of the same shared embedding used by final kNN. The teacher would not receive gold label, correctness/error, loss or true-class margin; it would never run at validation/test.

## Why the method is rejected

For each mode, its risk gradient is a weighted sum of per-example gradients. The proposed QP solution is therefore another weighted sum of those gradients: dynamic sample reweighting plus a generic common-descent/gradient-surgery primitive. It is not cleanly distinct from probabilistic GroupDRO, JTT/EIIL-style reference-model grouping, or PCGrad/MGDA/CAGrad.

The proposal also constrained raw gradients before `AdamW`; momentum, adaptive preconditioning and weight decay transform the actual update, so the claimed mode/base descent constraints were not operational guarantees. Finally, even without a literal correctness field, the MLLM could compare raw evidence with the supplied prediction and reconstruct a scalar error propensity. Error predictiveness would not prove semantic-mode necessity.

These are mechanism and novelty failures. They cannot be repaired by prompt changes, extra controls, larger teachers or explanation prose.

## Terminal decision and anti-repeat

**Verdict:** `RETHINK`; **ECM status:** `ABANDONED`; **score:** `4.98/10` after one independent review round.

Do not re-propose any of the following under a new name:

1. MLLM pseudo-groups followed by standard/soft GroupDRO or minimax weighting;
2. OOF error/margin/difficulty reconstruction followed by JTT-style weighting;
3. mode risks followed by PCGrad, MGDA, CAGrad or the frozen raw-gradient QP;
4. raw-gradient “descent” constraints that are then transformed by AdamW;
5. semantic-mode value proved only by label/error AUC rather than final ordinary-kNN correction beyond matched scalar error propensity;
6. any segment/timestamp/span/localization assumption.

No ECM implementation, experiment, teacher call or performance stage is authorized. No ECM accuracy/macro-F1 result exists.

## Boundary for a genuinely new hypothesis

A future distinct route may investigate a **full-bank semantic proximal target or actual-update-space operator**, but only after a fresh Gate-0 literature/novelty review. It must not be called an ECM revision. It must constrain the realized final-bank geometry or actual adaptive-optimizer step, show that its vector intervention cannot be reproduced by scalar reweighting, beat matched ERROR-PROPENSITY, GroupDRO/JTT/EIIL and PCGrad/CAGrad controls, and retain the same no-segment-gold/test-clean contract.

SQ's now-authoritative S0 result is also terminal for its frozen route: provenance/QC STOP, not a learned accuracy/macro-F1 failure. The global target remains active and unmet.
