# TASK SPEC — goal_mllm_plus3 (Deli_AutoResearch protocol, adopted 2026-07-13)

## Goal (user hook, verbatim binding)
MLLM meaningfully AND novelly integrated into the RGCL hateful-video framework, leading to substantial performance improvement — **at least +3 test accuracy** (project bar: pre-registered test pass at +0.030 acc AND F1, ideally on ≥2 datasets) — before stopping. No asking the user mid-work (zero interaction; ready = execute).

## Hard constraints (violations = invalid result)
- Method proper uses NO gold annotations (no time-span, no target labels; gold ONLY for oracle ceiling probes / scoring MLLM predictions).
- **NO OCR channel** (user veto 2026-07-13 "没啥用" — never re-propose).
- No cross-seed ensembles. No reimplementing codeless baselines. No author emails. Local open weights only (currently ONLY Qwen2.5-VL-7B-Instruct downloaded).
- All GPU via SLURM sbatch, no --time, conda HateVideo, ≤2 GPU/16 CPU/128GB; JobHeldUser waits (never force).
- Ceremony regime: pre-registration → fresh 0-context review → implementation → independent code review → real-frozen-entry smoke → freeze+authorization → SINGLE-SUBMIT per lineage; conclusion-bearing verdicts get an independent verdict reviewer.
- **G0-cond gate mandatory** before any auxiliary-signal GPU spend: conditional MDL probe, capacity-matched, projected gain > +0.030 + 0.010 noise band, oracle-ceiling kill switch (see research-wiki/REFLECTION_mllm_integration_failures.md §4).

## Diagnosis frame (why 16 directions died — enforce on every new direction)
Low-bandwidth decision-side signals are conditionally redundant given the frozen representation ("probe passes, training flat"). Only representation-level levers ever cleared +3 (encoder swap, HateMM only). Any new direction must state its bandwidth + injection point and must not be isomorphic to entries in directions_tried.json.

## Milestones (current pipeline)
1. **C2 SAV (lead)**: exp-sav-f0.md APPROVED (Rev-2/2a/2b). Remaining: main-loop rulings applied → independent code review → real-entry smoke → freeze+authorization → single-submit F-G1 (extract 784 heads + guard + probe, ~2.5h, 1 GPU) → F-G1 verdict (MDL+acc clustered bootstrap, bar +0.040 projected) → if pass: F-G2 RGCL integration (+0.015 val, HateMM no-harm −0.010) → F-G3 single test touch (+0.030 acc AND F1) + independent verdict review.
2. **C3 dense reasoning-text channel** (only if C2 dies or in parallel at zero GPU): G0-cond gate first (needs signal generation ~hours GPU → queue after SAV extraction); gate fail = dead at zero further cost.
3. **If C2+C3 die**: forced structural pivot per protocol §6 — generate fresh directions NOT in directions_tried.json (candidates from LITERATURE doc §1: C4 KD needs 72B download ruling; C5 representation-side pseudo-label expansion needs in-domain unlabeled pool; or new literature sweep with different domains).

## Success criteria
Pre-registered, single-submitted, independently-verdict-reviewed test result: Δacc ≥ +0.030 AND ΔF1 ≥ +0.030 vs the matching frozen floor, correct protocol (no-selection primary), on ≥1 dataset (goal ideally ≥2). Anything less = negative, log and pivot.

## Floors (provenance: exp-encoder-3seed.md, PAPER_MASTER_TABLES, erratum 66012e9)
HateMM: CLIP 0.8279/0.8172 (test), Qwen 0.870 (test, 3-seed); DEV floors seed0: kNN 0.8598 val-sel / 0.8505 final. MHC-EN: ~0.79-0.81. MHC-ZH: 0.8537±0.012 (already above published SOTA 0.80).

## BINDING RULINGS ADDENDUM (2026-07-14, user, D7 resolution)
- **Novelty clause ≠ encoder-class levers.** Frozen encoder swap, LoRA-adapted encoder swap, and generic decision-rule calibration (B5-class) do NOT satisfy the goal's "novel" clause (D7 = RESOLVED-NEGATIVE). They remain valid performance / ablation / diagnosis material ONLY — never a goal-satisfying route.
- **Goal is non-negotiable.** TERMINUS option (c) goal-renegotiation is DEAD. Verbatim (user, 2026-07-14 evening): 「哎呀,这个 encoder swap 肯定不算 novelty 啊」/「我不管,反正这个做不出来就一直做,直到做出来为止。」Continue until solved.
- **Success (bar unchanged, novelty tightened):** a NOVEL MECHANISM — novelty judged within hateful-video detection — that is MLLM-integrated AND demonstrates ≥+3 acc under the standing protocol: pre-registered test pass at +0.030 acc AND +0.030 F1, 3/3 seeds, both protocols (no-selection primary), single-dataset own train split only, no gold aux, no OCR, no APIs, no cross-dataset mixing.
- **Round 3 = novelty-mechanism-first direction generation.** B5 probe (job 13156) continues as a performance/diagnosis line, NOT a goal-satisfying route.
