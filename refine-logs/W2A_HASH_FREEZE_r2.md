# W2-A r2 HASH-FREEZE (§16 hash-freeze rows — scripts authored to the r1 spec)

**Stage:** implementation complete; awaiting **independent code review** → **smoke** → **single Stage-E' submit**.
**Author:** W2-A implementer. **Date:** 2026-07-15. **NO GPU used; NO submission made.**

This record pins the sha256 of the three authored artifacts **plus** the two governing r1 documents, per
the prereg §16 hash-freeze discipline. It lives in a **separate file** (not appended to the prereg's §16)
on purpose: editing `exp-w2a-grounded.md` would change its own hash and break the "docs unchanged → same
hashes" verification below (the exp doc must still hash to the r1 pin `076bfa5e…`). The r1 §16 pre-declared
CONSTANTS table remains the authority; this file records the code freeze against it.

## Frozen artifacts (r2 → r2b → r2c)

| artifact | sha256 | freeze |
|---|---|---|
| `scripts/analysis/w2a_extract.py`    | `9e984d61e2bf91d58f15af5e54f14d45a3fabe4e0701ce4492645399d810fa31` | **r2c (extractor fixes C+D)** |
| `scripts/slurm/w2a_extract.sbatch`   | `9ed04c14d16799d24e196f1d956698017373e597fd13e0cb2df6919087315153` | **r2 (UNCHANGED)** |
| `scripts/analysis/w2a_probe.py`      | `af4a2f9f5b35461173fd82c176bd52c6fc84bf8fc0d09736f938d38d8f6fe06d` | **r2b (probe fixes A+B)** |

**r2c (extractor-only, post-green-SMOKE re-freeze, 2026-07-16).** The two deferred code-review items,
applied to `w2a_extract.py` ONLY (sbatch `9ed04c14…` + probe `af4a2f9f…` byte-UNCHANGED):
- **C — placebo pairing** (`build_placebo_partners`): replaced the cyclic-successor partner (which wrapped
  the longest transcript to the shortest) with the NEAREST-by-|Δ char-length| ADJACENT partner (non-cyclic),
  for a clean length control. Behavioural only on the real-run gate-3 placebo (≥50 subset); grd/grd_pfx/
  img_recon/ungrd_vis and gates 0/1/4 are byte-identical.
- **D — comment** (module docstring gate-3 line): "kept as a secondary diagnostic" → "deferred (non-gating
  secondary; not implemented)" to match the code (the within-video token-shuffle placebo is not implemented).

The extractor `2e79599a…` frozen at r2/r2b is superseded by `9e984d61…`. **The green SMOKE (job 13166) ran
the r2 extractor `2e79599a…` and validated gates 0/1/4 + G-recon-IMG; r2c changes NEITHER those code paths
NOR the produced keys — only the gate-3 placebo partner selection (a real-run-only gate not exercised by the
`--limit 1` smoke)** — so the smoke result carries over and no re-smoke is required. Re-validated after r2c:
py_compile OK, CPU self-test PASS, Fix-C non-cyclic nearest-length pairing unit-checked. Diff routed to the
independent code reviewer; Stage-E' single-submit remains a SEPARATE grant after CLEARED.

**r2b (probe-only, 2026-07-15).** Applied the two code-review NON-BLOCKING fixes that must land before
probe execution, in `w2a_probe.py` ONLY:
- **A — checkpoint-signature hardening.** `ci_meta` now also carries `probe_sha` = sha256(this script) and
  `grd_sha` = per-dataset sha256(train_grounded.pt)+sha256(dev_seen_grounded.pt), so a re-extraction into
  the same `grounded_dir` or any probe edit invalidates a stale K9 checkpoint (no silent reuse of cached
  point-arms + perm seeds).
- **B — K2/K3 VOID surfaced in `mechanical_gate_check`.** Two rows/dataset (`GroundingLive[ds] (K2)`,
  `Placebo[ds] (K3)`) read the Stage-E' gatelog VOID flags; a K9 `CONDINFO_PROCEED` on a VOID dataset is
  relabelled `VOID(K2/K3-nullified)` and cannot count toward the aggregate `SURVIVES` (closes the
  silent-no-op-grounding foot-gun the reviewer named). `mechanical_gate_check` stays explicitly non-binding.

The prior extractor `2e79599a…` + sbatch `9ed04c14…` rows are **byte-UNCHANGED** — the extract SMOKE (job
13166) runs the r2-frozen extractor; the probe change does not touch it. Re-validated after r2b: py_compile
OK, probe synthetic self-test PASS, Fix-B VOID nullification unit-checked (LIVE→SURVIVES, VOID→nullified).

**Code-review items C + D DEFERRED (extractor-side, non-blocking).** C (placebo cyclic-wrap pairing) and D
(comment "kept"→"deferred") both live in `w2a_extract.py`; editing it now would change the extractor the
pending SMOKE reads at execution time and break the frozen sha echo, so they are deferred to the **post-green-
SMOKE re-freeze** (before full Stage-E' extraction) where the extractor is re-hashed anyway. Both are
non-gating (C is safe-direction on the median VOID; D is doc-only).

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
