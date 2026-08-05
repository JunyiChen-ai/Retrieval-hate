# C02 A0 v4 — fresh independent static review request

**Type:** read-only static review. **No execution of any kind is authorized.**

You are reviewing a frozen, not-yet-submitted preregistration in `/data/jehc223/RGCL`.
You have not seen the implementer's reasoning and must not ask for it. Judge only the
artifacts. This is the fourth round; three prior rounds returned REVISE (0C/4H/20I, 0C/2H/19I, 0C/4H/23I). Two of those rounds caught a repair that the record claimed but the code did not contain, so verify every claimed repair in the code and treat the record's assertions as unverified.

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
   `refine-logs/C02_A0_V4_RECORD.md` (**under review**).
6. The prior reviews: `refine-logs/C02_A0_PREREG_REVIEW.md` (round 1, REVISE 0C/4H/20I)
   and `refine-logs/C02_A0_V2_PREREG_REVIEW.md` (round 2, REVISE 0C/2H/19I).

## The frozen v4 artifacts under review

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v4.json` | `8ccd2464699a7029db3952bc18612ea1cfcc79ede2b946e67051df843b26a4a9` |
| `src/utils/c02_density_views.py` | `2ec193cdfa920a2d974db5c8468702614a54fa378a8df324ca5ba47b7d955592` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `66381c40a03c480bceab0af3d4c0497478e00da39a6de7ec0c33f6daf992a319` |
| `scripts/slurm/c02_density_extract.sbatch` | `a1523087253990ce4a38642214aabd2890c34e650007d844ee5b9b4e992d8a9f` |
| `scripts/analysis/c02_a0_mint.py` | `2afbe8b075aefb1cdd02669e0336c53d4306366deeed8714c7f11f58a65e78a5` |
| `scripts/analysis/c02_a0_arena_v4.py` | `71bba0f1bd47517ea8da1bbd922274f66d4b2ef6c62099ca17cc97c1364aba26` |
| `scripts/slurm/c02_a0_cpu_v4.sbatch` | `ae4a237508ebfccde51cd3552903991d60001aad89f483e4861c490a648a8cd6` |
| `refine-logs/C02_A0_V4_RECORD.md` | *(recompute and report)* |

Recompute every sha256; any mismatch is Critical. Confirm the v1, v2 and v3 executables
(`configs/c02/c02_a0_v{1,2,3}.json`, `scripts/analysis/c02_a0_arena{,_v2,_v3}.py`,
`scripts/slurm/c02_a0_cpu{,_v2,_v3}.sbatch`) are ABSENT, and
that `artifacts/c02_edq` and any `*-c02den-*` cache are absent.

## What you must check

**A. Did v4 repair the four round-3 High findings?** State `REPAIRED` / `PARTIALLY
REPAIRED` / `NOT REPAIRED` with file:line evidence for each. Round 3's findings were:
- **H1**: the retracted "s_Q upper-bounds / failure is decisive" claim survived in the
  arena docstring for a second round while the record claimed it was gone. It must now be
  absent from EVERY frozen file. Grep for it yourself; do not trust the record.
- **H2**: `derangement_within`'s pairwise-swap repair could oscillate forever on a size-2
  group and die with a bare `AssertionError` outside the `Halt` path, after the GPU
  extraction was already spent. Verify the replacement always terminates and always
  yields a derangement for every reachable group size, including 2 and 3.
- **H3**: "under H0 FULL and SHUFFLE are EXCHANGEABLE BY CONSTRUCTION" was false (radial
  displacement counter-example). Verify the claim is retracted everywhere and that the
  replacement scope statement is itself correct — in particular, check the argument that
  the radial null cannot manufacture a PASS.
- **H4**: the SHUFFLE self-test re-typed the formula and could not fail. Verify it now
  exercises `build_arms` and would actually catch a regression to absolute-position
  donation.

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
each; ≤4.0 GPU-hour budget plausible; every constant in `configs/c02/c02_a0_v4.json`
matching the code; the view subsequence contract and its pre-forward proof; `PARITY-NAT`
binding and its tie exemption sound; the zero contract; full self-orbit exclusion.

**D. Be adversarial about the science.** Can the treatment still pass on an artifact that
neither `SHUFFLE` nor `NOISE` covers? Is the `k = 20` per-view-pair exactness argument
correct? Would a KILL from this design be sound? Would a PASS be sound? Answer both
separately.

**E. Implementation defects.** Read the code, not the prose. Anything that would kill the
job after the GPU is already up is at least High.

## Output

Write to `refine-logs/C02_A0_V4_PREREG_REVIEW.md`: a repair-verdict table for H1-H4; a findings table classified `Critical` / `High` / `Info` with file:line; a final
verdict line, exactly one of `GO (0C/0H/0I)` or `REVISE (nC/nH/nI)`; and an explicit
statement of what you did and did not execute.

`GO` is only permitted with zero Critical and zero High findings. Do not soften a finding
to reach GO. Do not run Python, do not load any cache, do not submit anything, do not
modify any reviewed file.
