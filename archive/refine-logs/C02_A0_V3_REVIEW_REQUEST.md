# C02 A0 v3 — fresh independent static review request

**Type:** read-only static review. **No execution of any kind is authorized.**

You are reviewing a frozen, not-yet-submitted preregistration in `/data/jehc223/RGCL`.
You have not seen the implementer's reasoning and must not ask for it. Judge only the
artifacts. This is the third round; two prior rounds returned REVISE.

## Read first

1. `CLAUDE.md`, `AGENTS.md`.
2. `TARGET_STATE.json` — blocks `iteration_8_stage0_bounded_extraction_amendment` and
   `registry_update_2026_07_28` (`unified_pilot_gate`, `hard_constraints`,
   `candidate_registry`, `eliminated_directions`).
3. `refine-logs/C02_DESIGN_REVIEW.md`, `refine-logs/C02_EXPERIMENT_PLAN.md`,
   `refine-logs/C01_ZERO_CONTRACT_PROBE.md`.
4. `refine-logs/HEADSPACE_TRANSFER_PREGATE.md` (F113) — skim §0–§3, §4.1–§4.5.
5. The freeze chain, in order: `refine-logs/C02_A0_RECORD.md` (v1, superseded),
   `refine-logs/C02_A0_V2_RECORD.md` (v2, superseded),
   `refine-logs/C02_A0_V3_RECORD.md` (**under review**).
6. The prior reviews: `refine-logs/C02_A0_PREREG_REVIEW.md` (round 1, REVISE 0C/4H/20I)
   and `refine-logs/C02_A0_V2_PREREG_REVIEW.md` (round 2, REVISE 0C/2H/19I).

## The frozen v3 artifacts under review

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v3.json` | `3c55214494372457fb8f2702f7ecf1c82c48b13c6b523d99e00272d2b0aa15ca` |
| `src/utils/c02_density_views.py` | `531d4574a6c678132cb76510af0570067891a64ab5aa0a751f638b7f99ffd2fc` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `128461322f0b8f8d66478fc5bc296dba282850bd56f9b4665f06733af99149a0` |
| `scripts/slurm/c02_density_extract.sbatch` | `001a089107e40f486a14eafd2daa52fb80033450b382dd41f204f46e8862de5a` |
| `scripts/analysis/c02_a0_mint.py` | `da8a49187b821f0da15c7cec28421317225213f78f40f20d5c62c58c0ab71d33` |
| `scripts/analysis/c02_a0_arena_v3.py` | `7d04a8ad8e644851fb8e25f77eee30ac12fbf7e33344dbc484bc5da240e21629` |
| `scripts/slurm/c02_a0_cpu_v3.sbatch` | `9463d642756269a77929dd3ffeb8afeab02f81c2b5bd77a20d1566d245bae399` |
| `refine-logs/C02_A0_V3_RECORD.md` | *(recompute and report)* |

Recompute every sha256; any mismatch is Critical. Confirm the v1 and v2 executables
(`configs/c02/c02_a0_v1.json`, `configs/c02/c02_a0_v2.json`,
`scripts/analysis/c02_a0_arena.py`, `scripts/analysis/c02_a0_arena_v2.py`,
`scripts/slurm/c02_a0_cpu.sbatch`, `scripts/slurm/c02_a0_cpu_v2.sbatch`) are ABSENT, and
that `artifacts/c02_edq` and any `*-c02den-*` cache are absent.

## What you must check

**A. Did v3 repair the two round-2 High findings?** State `REPAIRED` / `PARTIALLY
REPAIRED` / `NOT REPAIRED` with file:line evidence for each.
- **H-A**: the retracted "s_Q upper-bounds / failure is decisive" claim must be gone from
  EVERY frozen file, not just the record. Grep for it.
- **H-B**: `SHUFFLE` must no longer be degraded under the design's own null. Verify in
  code that it donates a DISPLACEMENT and never the donor's absolute position, and reason
  independently about whether `FULL > SHUFFLE` is now a real test at H0 or still
  satisfiable when the orbit carries nothing.

**B. Did the repairs introduce new defects?**
- Is the displacement-donation formula correct, deterministic, dtype-safe, and free of
  in-place mutation of shared arrays?
- Is `shuffle_groups` correct? Can a donor group ever straddle the fitting-pool /
  held-out boundary? Does the singleton-merge rule ever leave a fixed point, i.e. an item
  that keeps its OWN displacement in the control arm?
- Does the degeneracy-matched grouping introduce any dependence on the manifest that
  could differ between the head arena and the raw arena?
- Are `degen_mask` and the merge counter defined before every use?

**C. Re-check everything independently — do not take either prior review's word for it:**
extraction limited to `train` + `dev_seen` with no reachable path to any test cache, test
jsonl or test label including through imported modules; the `+0.050`/`+0.050` two-dataset
bar unchanged; F113 respected (PASS only on the fold-head path, raw arena confined to
corroborating a KILL); no OCR, no cross-dataset mixing, no external API, single-dataset
train split, parent-video binary label only, no ensembles, no size scaling; SLURM-only,
`conda HateVideo`, no `--time`, 8 CPU, no dependency/array/force/release, one submission
each; ≤4.0 GPU-hour budget plausible; every constant in `configs/c02/c02_a0_v3.json`
matching the code; the view subsequence contract and its pre-forward proof; `PARITY-NAT`
binding and its tie exemption sound; the zero contract; full self-orbit exclusion.

**D. Be adversarial about the science.** Can the treatment still pass on an artifact that
neither `SHUFFLE` nor `NOISE` covers? Is the `k = 20` per-view-pair exactness argument
correct? Would a KILL from this design be sound? Would a PASS be sound? Answer both
separately.

**E. Implementation defects.** Read the code, not the prose. Anything that would kill the
job after the GPU is already up is at least High.

## Output

Write to `refine-logs/C02_A0_V3_PREREG_REVIEW.md`: a repair-verdict table for H-A and
H-B; a findings table classified `Critical` / `High` / `Info` with file:line; a final
verdict line, exactly one of `GO (0C/0H/0I)` or `REVISE (nC/nH/nI)`; and an explicit
statement of what you did and did not execute.

`GO` is only permitted with zero Critical and zero High findings. Do not soften a finding
to reach GO. Do not run Python, do not load any cache, do not submit anything, do not
modify any reviewed file.
