# C02 A0 — independent static review request

**Type:** read-only static review. **No execution of any kind is authorized.**

You are reviewing a frozen, not-yet-submitted preregistration for a research experiment
in `/data/jehc223/RGCL`. You have not seen the author's reasoning and must not ask for
it. Judge only the artifacts.

## Read first

1. `CLAUDE.md`, `AGENTS.md` — project execution rules.
2. `TARGET_STATE.json`, block `iteration_8_stage0_bounded_extraction_amendment` — the
   registry amendment that authorizes this experiment — and the block
   `registry_update_2026_07_28` (`unified_pilot_gate`, `hard_constraints`,
   `candidate_registry`, `eliminated_directions`).
3. `refine-logs/C02_DESIGN_REVIEW.md` — the 2026-07-29 kill this design must answer.
4. `refine-logs/C02_EXPERIMENT_PLAN.md` — the historical C02 plan.
5. `refine-logs/C01_ZERO_CONTRACT_PROBE.md` — the project's zero-contract criteria.
6. `refine-logs/HEADSPACE_TRANSFER_PREGATE.md` — F113, the head-space arena instrument
   that this design reuses (skim §0-§3 and §4.1-§4.5).

## The frozen artifacts under review

| path | sha256 |
|---|---|
| `refine-logs/C02_A0_RECORD.md` | *(the preregistration; recompute and report)* |
| `configs/c02/c02_a0_v1.json` | `0b8a8289e7438396ce081fdf872f7d18017f870640fa33a687099de4066b53d1` |
| `src/utils/c02_density_views.py` | `e0cd2d2b920a4f5133f30d174d36865843fe23977ff1f8639eea0400d12eab72` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `9ebb80f48d27fd14278b15692d45d3c925efc84ae61941fae1488574bc96832b` |
| `scripts/slurm/c02_density_extract.sbatch` | `e5c29338fab4b0ac1af4c57826e11bde9d96f29b111bd806b98ccc1658acafbc` |
| `scripts/analysis/c02_a0_mint.py` | `3b1b602b145fa362f270ba08a604a1b284ae153f0d22f9a15dafa5c3a0abbfa7` |
| `scripts/analysis/c02_a0_arena.py` | `92abe7d8157a54f89a47657fb1edaf4a8f90e55b873c3fd03840aa940593fa41` |
| `scripts/slurm/c02_a0_cpu.sbatch` | `2b55c67834fc6dfdaf9a932be634c735b5362edcd128cfd5aa6e3829fc82c281` |

Recompute every sha256 with `sha256sum` and report any mismatch as Critical.

## What you must check

**A. Contract compliance.**
- Does every view provably retain the complete native text as an ordered subsequence,
  with only controlled repetition added? Is the proof executed before any forward pass?
- Are the four controls the 2026-07-29 reviewer named (`RANDOM_WINDOW_REPEAT`,
  `MIN_WINDOW_REPEAT`, `REPEAT_ONLY`, `LOCALIZED_REPEAT_ONLY`) actually present and
  actually distinct from the treatment?
- Are frozen orbit radius, KRR metric, retrieval-length correlation,
  confidence/control thresholds, lambda selection status, Holm family and full
  self-orbit exclusion each specified and implemented?
- Is the handling of empty / speech-poor / identity-orbit items explicit, counted and
  fail-closed? Is HateMM `hate_video_95` (train row 355, structural all-zero null)
  handled consistently with `C01_ZERO_CONTRACT_PROBE.md`?

**B. Registry compliance.**
- Extraction limited to `train` + `dev_seen`; test never opened. Look for ANY path by
  which a `test_seen` cache, `test.jsonl`, or a test label could be read, including
  through imported modules.
- `+0.050` / `+0.050` two-dataset bar unchanged; net-fix clause present.
- No OCR, no cross-dataset mixing, no external API, single-dataset train split,
  parent-video binary label only, no ensembles, no model-size scaling.
- F113: is any PASS rendered on the fold-head / deployed-head path, with the raw arena
  confined to corroborating a KILL?
- SLURM-only, `conda HateVideo`, no `--time`, 8 CPU jobs, no dependency/array/force/
  release, one submission each.
- Is the ≤4.0 GPU-hour budget plausible for the declared work?

**C. Scientific validity — be adversarial.**
- The oracle is `s_Q(i,j) = max over view pairs of cosine`. A max over a large view-pair
  product inflates similarities mechanically. Do the `SHUFFLE` and `NOISE` controls
  actually isolate the *correct within-video orbit* from that inflation? Is there any
  way the treatment could pass on an artifact these controls do not cover?
- Is the `k = 20 per view pair` search argument in `orbit_vote` correct? State whether
  the exactness claim holds, and whether ties break it.
- Does `PARITY-NAT` actually bind? Is the tie exemption sound or is it a loophole?
- Is the arena free of leakage: bank/query disjoint, held-out fifth never seen by the
  head in any role, no query's own orbit in its own bank?
- Are the statistics honest — is the bootstrap paired, is the Holm family the right
  size, is anything selected on the data it is then evaluated on?
- Is the `MIN`/`MAX` P3 positional approximation adequately confined to controls, and
  does its weakness threaten the primary read?
- Would a KILL from this design be sound, and would a PASS be sound? These are different
  questions; answer both.

**D. Implementation defects.**
- Read the code, not just the prose. Look for: wrong indices, fold leakage, in-place
  mutation of shared arrays, non-determinism, dead code, guards that cannot fire,
  assertions that are always true, config values contradicted by code constants,
  degenerate cases that crash rather than fail closed, and anything that would make the
  job die after the GPU is already up.
- Verify that every constant in `configs/c02/c02_a0_v1.json` matches the code, and flag
  any config claim the code does not implement.

## Output

Write your review to `refine-logs/C02_A0_PREREG_REVIEW.md` with:
- a findings table classified `Critical` / `High` / `Info`, each with file:line;
- a final verdict line, exactly one of
  `GO (0C/0H/0I)` or `REVISE (nC/nH/nI)`;
- an explicit statement of what you did and did not execute.

`GO` is only permitted with zero Critical and zero High findings. Do not soften a
finding to reach GO. Do not run Python, do not load any cache, do not submit anything,
do not modify any reviewed file.
