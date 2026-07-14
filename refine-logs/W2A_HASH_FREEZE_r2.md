# W2-A r2 HASH-FREEZE (§16 hash-freeze rows — scripts authored to the r1 spec)

**Stage:** implementation complete; awaiting **independent code review** → **smoke** → **single Stage-E' submit**.
**Author:** W2-A implementer. **Date:** 2026-07-15. **NO GPU used; NO submission made.**

This record pins the sha256 of the three authored artifacts **plus** the two governing r1 documents, per
the prereg §16 hash-freeze discipline. It lives in a **separate file** (not appended to the prereg's §16)
on purpose: editing `exp-w2a-grounded.md` would change its own hash and break the "docs unchanged → same
hashes" verification below (the exp doc must still hash to the r1 pin `076bfa5e…`). The r1 §16 pre-declared
CONSTANTS table remains the authority; this file records the code freeze against it.

## Frozen artifacts (r2)

| artifact | sha256 |
|---|---|
| `scripts/analysis/w2a_extract.py`    | `2e79599a92d227d9f15366ee17a6644c2f6c77c71f36aa61c76a6274ac9402a9` |
| `scripts/slurm/w2a_extract.sbatch`   | `9ed04c14d16799d24e196f1d956698017373e597fd13e0cb2df6919087315153` |
| `scripts/analysis/w2a_probe.py`      | `72e25d246890ecd2f52207f64961dc7feebc6dbb29c930c636119a646ae494ce` |

## Governing docs (verified UNCHANGED vs cb59a94)

| doc | sha256 | expected (cb59a94) | match |
|---|---|---|---|
| `research-wiki/experiments/exp-w2a-grounded.md` | `076bfa5eff14fe1321ec98a27e1d7484c129643649eb9fb4156a1233dc432e6b` | `076bfa5e…` | ✓ |
| `refine-logs/W2A_FORENSIC_RECON.md`             | `fedc7e6726385be802f68e367abe5330d71ba1ad7e8a80f0b212674a05a67861` | `fedc7e67…` | ✓ |
| `refine-logs/W2A_PREREG_REVIEW.md`              | `b7f6ee09bb4eaf69d0c1132b4a5247aa1383857ae792a045815c7d0270bb8a8c` | (not pinned) | — |

`git diff cb59a94 -- <exp> <recon> <review>` is empty (docs byte-identical to r1).

## §16 pre-declared constants — implemented values (re-verify at submit)

The scripts encode the r1 §16 hash-frozen constants literally:
- CONCAT-geometry **(b)**: kNN raw bar ADVISORY; the **K9 conditional-info probe vs `Z_best` is the SOLE
  binding performance gate** (`w2a_probe.py`: `RAW_BAR` advisory, `mechanical_gate_check` marks K6/K7/K8 adv).
- `Z_best` = concat(CLIP img[1024], CLIP text[768], Qwen img[3584], Qwen text[3584]) = **8960-d**
  (`build_Zbest` asserts `Z.shape[1]==8960`); secondary Qwen-only = 7168-d, point-arms only.
- triple rule C1 Δacc ≥ **+0.040** (`CI_BAR`), C2 per-video-clustered bootstrap CI-lower > 0, C3 real > all
  perm maxima; label-oracle calibration `accZA ≥ 0.99` else `MACHINERY_INVALID`; ≥**150** perms (`CI_NSEED`).
- oracle-ceiling kill Δacc < **+0.04** on every dataset (`ORACLE_BAR`); Fano ≥ **0.99** (`FANO_BAR`).
- grounding-live VOID present-set **median** cos(grd, ungrd_vis) ≥ **0.999** (`GROUNDING_NOOP_VOID`);
  τ_live + empty-branch = logged diagnostics.
- placebo (binding): cross-video **mismatched** transcript must move grd, subset-median cos < 0.999
  (`PLACEBO_NOOP_VOID`), ≥**50**-video subset (`PLACEBO_N`).
- advisory kNN bar Δacc ≥ **+0.05** AND ΔmF1 ≥ **+0.05** on HateMM (`RAW_BAR`), beat CONCAT-PCA + CONCAT-α
  in sign; permutation null seeds **0..99** (`NULL_SEEDS`); CONCAT-α grid `{0.3,0.4,0.5,0.6,0.7}`
  (`ALPHA_GRID`); bootstrap ≥**1000** (`--n_boot`); near-dup `cos ≥ 0.995` reported at 0.98/0.99/0.995.
- memory sizes HateMM **851** / MHC-EN **629** (`EXPECTED_MEM`, fail-closed); test never opened.
- grid gate `spatial_merge_size` read from the model config; grounded transcript block raw text |
  empty→`"(none)"`; grd_pfx span = `[first_vis : end]`.

## Pre-submit checklist (unchanged from r1 status block)
independent code review → `--limit 1` SMOKE to throwaway path (gate 1 G-recon-IMG DOES run) → re-freeze
these hashes → single Stage-E' submit (local GPU, no `--time`, `JobHeldUser` → wait, never force) → Stage-P'
probe (cloud/CPU; **login-node runs get reaped** — must go through SLURM or Modal, not the login shell) →
independent verdict review (raw-only transcription). Head-training formal stage (§11) NOT authorized.
