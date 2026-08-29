# C02 Independent Design Review

**Candidate:** `C02 Evidence-Density Quotient Geometry`  
**Date:** 2026-07-29 (Pacific/Auckland)  
**Final verdict:** `KILL_C02_DESIGN_COLLISION_OR_INFEASIBILITY`  
**Scope:** design review and read-only asset audit only

## Decision

C02 is frozen before implementation. This is a **design infeasibility under the
current candidate-registry contract**, not a scientific refutation of the abstract
EDQ hypothesis.

The binding Stage-0 rule requires an existing-bank, representation-level oracle to
reach at least `+0.050` accuracy and `+0.050` macro-F1 on at least two datasets before
any new extraction, teacher, or GPU spend. No existing representation bank expresses
the full-transcript-preserving text-density orbit proposed by C02, so that gate cannot
be adjudicated without doing the work that the gate explicitly forbids.

## First review: `REVISE_DESIGN`

The independent reviewer identified two blocking design errors:

1. **A0 proxy-target mismatch.** Existing P3 mean/soft/mild banks change image
   pooling while keeping text byte-identical. The proposed EDQ target changes
   transcript evidence density. A per-item `max_{a,b}` over P3 variants is therefore
   only an optimistic image-routing upper bound; it cannot kill or establish
   reachability for EDQ.
2. **The proposed evidence-core deletion was not a safe equivalence.** Removing
   transcript context can invert the meaning of quotations, counterspeech, archived
   speech, lyrics, or reportage. A label-preserving C02 view must retain the complete
   native transcript as an ordered subsequence and may only add controlled
   repetition.

The reviewer also required, before any end-to-end stage, explicit
`RANDOM_WINDOW_REPEAT`, `MIN_WINDOW_REPEAT`, `REPEAT_ONLY`, and
`LOCALIZED_REPEAT_ONLY` controls; frozen orbit radius, KRR metric, retrieval-length
correlation, confidence/control thresholds, lambda selection, Holm family, and full
self-orbit exclusion; and an explicit account of empty/speech-poor HateMM examples
whose transcript orbit is the identity.

## Read-only asset audit

The follow-up audit found no train/dev representation bank for native versus exact
full-transcript repeat, localized repeat, prefix/suffix repeat, echo, truncation, or
other full-transcript-preserving density views on both HateMM and MHC-ZH.

- `*p3pool_*` varies image pooling only.
- `*bidir/{meanpool,textpool}*` varies attention/readout, not transcript density.
- `*nullop2merge*` is a PEFT merge-path null probe.
- HateMM `*curric-rep2*` is an independent SFT draw, not an input-repeat view
  (`PROVENANCE_AUDIT_2026-07-28.md:143-146`).
- Echo/repeat embeddings were discussed in notes but were never extracted.

Consequently there is no legal existing-bank A0 whose representation matches the
final method. Reusing P3 would answer a different question, while extracting the
needed density views would violate the frozen pre-extraction Stage-0 gate.

## Final adjudication

The independent reviewer returned:

`KILL_C02_DESIGN_COLLISION_OR_INFEASIBILITY`

Required consequences:

- do not implement or execute the proposed A0;
- do not extract new C02 views;
- do not make teacher calls, open test data, use GPU resources, or submit a SLURM job;
- preserve `refine-logs/C02_EXPERIMENT_PLAN.md` as a historical design record;
- advance only to C03 design and independent review.

EDQ remains non-isomorphic to P3, SAV, and MECHFIX at the abstract method level, but
its novelty relative to generic view consistency and recent length-debiasing work
would still require a sharper method-level distinction if it were ever reconsidered.

No Python, pytest, cache loading, feature extraction, teacher call, GPU work, test
access, or SLURM submission occurred in this C02 review.
