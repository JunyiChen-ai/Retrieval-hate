# C02 A0 v2 — fresh independent static review request

**Type:** read-only static review. **No execution of any kind is authorized.**

You are reviewing a frozen, not-yet-submitted preregistration in `/data/jehc223/RGCL`.
You have not seen the implementer's reasoning and must not ask for it. Judge only the
artifacts.

## Read first

1. `CLAUDE.md`, `AGENTS.md`.
2. `TARGET_STATE.json` — block `iteration_8_stage0_bounded_extraction_amendment` (the
   registry amendment that authorizes this experiment) and block
   `registry_update_2026_07_28` (`unified_pilot_gate`, `hard_constraints`,
   `candidate_registry`, `eliminated_directions`).
3. `refine-logs/C02_DESIGN_REVIEW.md` — the 2026-07-29 kill this design must answer.
4. `refine-logs/C02_EXPERIMENT_PLAN.md` — the historical C02 plan.
5. `refine-logs/C01_ZERO_CONTRACT_PROBE.md` — the project's zero-contract criteria.
6. `refine-logs/HEADSPACE_TRANSFER_PREGATE.md` — F113, the head-space arena instrument
   reused here (skim §0–§3, §4.1–§4.5).
7. `refine-logs/C02_A0_RECORD.md` — the **superseded v1** freeze (sections 1–8 are still
   the design of record except where v2 changes them).
8. `refine-logs/C02_A0_PREREG_REVIEW.md` — the **prior round's** review, `REVISE
   (0C/4H/20I)`.
9. `refine-logs/C02_A0_V2_RECORD.md` — the **v2 freeze under review**, which claims to
   repair all four High findings.

## The frozen v2 artifacts under review

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v2.json` | `2d4b7148154caea6ed41ec95043c15295c63d1abf3c47467b9191d285bd98a6f` |
| `src/utils/c02_density_views.py` | `f6209f04f04b88cfe47fadd5f7c7cd20b079f397a646fe824c8d2c3b35785b34` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `1e40e53013a032e527853cc5e82ca53b054882774b315a6ea6f319ce321b0803` |
| `scripts/slurm/c02_density_extract.sbatch` | `aaee1516f52ff2aabf508580b5451973a0484b3ca0875116be33a92c252e76e8` |
| `scripts/analysis/c02_a0_mint.py` | `f93a9d336c2917ede8737a8a597b7c9e3f83d5173ef4163b2e62118ba466da6b` |
| `scripts/analysis/c02_a0_arena_v2.py` | `7315e3232a42c96f1bf943028bb852eb89c9d85acd902f3890fb83fcd110e01d` |
| `scripts/slurm/c02_a0_cpu_v2.sbatch` | `ccf9881ccae7019d261a393afec4e6504203b947d38a259bf4d68b0238eccbf0` |
| `refine-logs/C02_A0_V2_RECORD.md` | *(recompute and report)* |

Recompute every sha256 with `sha256sum` and report any mismatch as Critical. Also confirm
that `configs/c02/c02_a0_v1.json`, `scripts/analysis/c02_a0_arena.py` and
`scripts/slurm/c02_a0_cpu.sbatch` are **absent** (the superseded v1 executables were
removed so they cannot be submitted), and that `artifacts/c02_edq` and any
`*-c02den-*` cache are absent.

## What you must check

**A. Did v2 actually repair the four prior High findings?** For each of H1 (leaking
`SHUFFLE` derangement), H2 (false `EMPTY_TEXT` expectation), H3 (tautological
`net_fix_rate` gate) and H4 (unproved upper-bound claim), state `REPAIRED` /
`PARTIALLY REPAIRED` / `NOT REPAIRED` with file:line evidence. A repair that only changes
prose while the code still has the defect is `NOT REPAIRED`. Independently re-derive H2's
counts if you can do so without executing Python — if you cannot verify them statically,
say so rather than accepting them.

**B. Did the repairs introduce new defects?** In particular:
- Is `derangement_within` correct, deterministic, and actually fold-local? Does the
  fold-local seed make `SHUFFLE` differ between the head arena and the raw arena in a way
  that matters? Does deranging within the bank create a *different* leakage channel?
- Is the bootstrap now on the same estimand as the bar, and is it still paired?
- Does `oracle_self_test` actually run before any real data is opened, and would it
  catch the failures it claims to catch?
- Do the `__debug__` and `torch.load` guards fire where claimed?
- Does the early no-clobber path in the extractor cover every file the job will write?

**C. Everything the first review checked still applies.** Re-check independently, do not
take the prior review's word for it:
- extraction limited to `train` + `dev_seen`; no reachable path to any test cache, test
  jsonl or test label, including through imported modules;
- `+0.050` / `+0.050` two-dataset bar unchanged; F113 respected (PASS only on the
  fold-head path, raw arena confined to corroborating a KILL);
- no OCR, no cross-dataset mixing, no external API, single-dataset train split,
  parent-video binary label only, no ensembles, no size scaling;
- SLURM-only, `conda HateVideo`, no `--time`, 8 CPU, no dependency/array/force/release,
  one submission each; ≤4.0 GPU-hour budget plausible;
- every constant in `configs/c02/c02_a0_v2.json` matches the code.

**D. Be adversarial about the science.**
- Can the treatment still pass on an artifact that neither `SHUFFLE` nor `NOISE` covers?
- Is the `k = 20` per-view-pair exactness argument correct, and do ties break it?
- Is `PARITY-NAT` binding, and is the tie exemption a loophole?
- Is the arena free of leakage: bank/query disjoint, held-out fifth never seen by the
  head in any role?
- Would a KILL from this design be sound? Would a PASS be sound? Answer both separately.

**E. Implementation defects.** Read the code, not the prose. Wrong indices, fold leakage,
in-place mutation of shared arrays, non-determinism, dead code, guards that cannot fire,
assertions that are always true, config claims the code does not implement, degenerate
cases that crash rather than fail closed, anything that would kill the job after the GPU
is already up.

## Output

Write to `refine-logs/C02_A0_V2_PREREG_REVIEW.md`:
- a per-finding repair verdict table for H1–H4;
- a findings table classified `Critical` / `High` / `Info`, each with file:line;
- a final verdict line, exactly one of `GO (0C/0H/0I)` or `REVISE (nC/nH/nI)`;
- an explicit statement of what you did and did not execute.

`GO` is only permitted with zero Critical and zero High findings. Do not soften a finding
to reach GO. Do not run Python, do not load any cache, do not submit anything, do not
modify any reviewed file.
