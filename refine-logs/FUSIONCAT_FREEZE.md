# FUSIONCAT — HASH-FREEZE

**Frozen by:** independent 0-context reviewer. **Date:** 2026-07-25 NZST.
**Ruling:** `APPROVED-WITH-NOTES` (see `refine-logs/FUSIONCAT_PREREG_REVIEW.md`, V1–V7 all PASS/CONFIRM;
Notes N1/N2 non-blocking, no re-freeze required).
**Base commit:** prereg `511e74c` (`refine-logs/FUSIONCAT_PREREG.md`), sbatch as committed at same HEAD.

## FROZEN ARTIFACTS (byte-exact; sha256)

```
FROZEN refine-logs/FUSIONCAT_PREREG.md  c88332b8972e3270081600d0a8cb892a8d24afefbc73e378a5a3104a433c0830
C      scripts/slurm/fusioncat_family.sbatch  62bfb773beeec325096362c86bae1aca8b90c94804ba3798973bbc88e82517fc
```

(The sbatch sha matches the value declared in prereg §5.1 / §5.3 exactly.)

## REUSED MACHINERY (must be UNCHANGED at submit; ZERO edit)

```
  src/model/classifier.py   e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378
  src/run_rac.py            b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3
  src/model/loss.py         2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b
  src/utils/retrieval.py    d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57
```

## VOID-ON-EDIT CLAUSE

This freeze authorizes submission of **exactly** the two frozen artifacts above, running the concat branch that
already exists in the committed `src/` at the four reused shas. Authorization is **VOID** if, at submit time,
ANY of the following is true:

1. `sha256sum refine-logs/FUSIONCAT_PREREG.md` ≠ `c88332b8…433c0830`, or
   `sha256sum scripts/slurm/fusioncat_family.sbatch` ≠ `62bfb773…2517fc`.
2. Any of the 4 reused-machinery shas differs from the block above.
3. `git status --porcelain src/` is non-empty (any staged/unstaged source edit).

On any VOID: HALT — no `sbatch`. A fresh independent 0-context review + re-freeze is required, and if a source
edit is involved, a **mandatory codex-code-review gate** is re-armed (prereg §4.6). No code edit lands silently
post-freeze.

## EXECUTOR PRE-SUBMIT CHECKLIST (from the frozen prereg)

- G-repro (§4.1): re-run the 3 gates above (src git-clean + reused shas + artifact shas) → all must match.
- Smoke (§4.4): CPU checks (bash -n, CONFIGS=6, collision-absent) + GPU throwaway concat run **per dataset**
  with the `fusion_mode='concat'` args-echo branch-assert (grep concat MUST match, grep align MUST be empty),
  finite losses, no shape error. Any fail ⇒ HALT. Delete all `RAC_video_smoke_fuscat*` dirs + throwaway logs.
- Never-two-16-CPU: confirm no other 16-CPU job queued/running at submit (this 8-CPU job clears trivially).
- Only then: `sbatch scripts/slurm/fusioncat_family.sbatch` → 6 head runs (~0.1 GPU-h). `PENDING (JobHeldUser)`
  ⇒ WAIT for auto-release, never force.
- Test-touch: the 6 head reads are the ONLY budgeted fusioncat evaluations; verdict rendered by an independent
  0-context reviewer against the frozen prereg VERBATIM.
