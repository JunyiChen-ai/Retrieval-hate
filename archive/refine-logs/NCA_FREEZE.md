# NCA / soft-kNN HEAD-LOSS family — HASH-FREEZE

**Frozen by:** independent 0-context pre-registration reviewer.
**Date:** 2026-07-25 NZST.
**Verdict:** `APPROVED-WITH-NOTES` (see `refine-logs/NCA_PREREG_REVIEW.md`; four non-blocking notes).
**Prereg (frozen object):** `refine-logs/NCA_PREREG.md` (commit `9a9f4fe`; on-disk == committed, unmodified).
**Recon:** `refine-logs/NCA_FORENSIC_RECON.md` (`685df9e`).

All shas below were recomputed on disk at freeze time and **match** the prereg §5 freeze block. The prereg was
**NOT** modified (review mandate) — the FROZEN self-sha is the current on-disk sha of the unmodified file.

```
FROZEN 7607863c15bef1d40c1f1f0a5b980123dd84aba260ced6db1fa993f277db5591  refine-logs/NCA_PREREG.md
A e1244adadf16b47c24b05786d1ee4e153fd9c696e3be0924eae43c82f1c3b75b  src/model/loss.py
B b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py
C baf41be8a1264445d6bbc63d2eb54966abed6da33c52176ae62d93bf79c14b94  scripts/slurm/ncafam_family.sbatch
```

### Reused-unchanged machinery (verified at freeze; do NOT edit)

```
e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378  src/model/classifier.py
d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57  src/utils/retrieval.py
00d9e9956549bdf97c6b8913d42d87811f4a2f150e9f459e8b8b86978b306f02  scripts/slurm/enc3seed_lora_curric.sbatch  (same-code anchor)
```

### Banked paired-control inputs (read-only; NOT clobbered)

- ZH floor (job 13150): `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` — present, untouched.
- HateMM floor (job 13241): `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` — present, untouched.
- Floor trainlogs: `slurm/logs/enc3s_MHC_zh_…-LoRA_HF_seed{0,1,2}_13150.trainlog`,
  `slurm/logs/enc3s_HateMM_…-LoRA-curric_HF_seed{0,1,2}_13241.trainlog` — re-parsed to 4dp, bit-match §2.1/§2.2.

### Executor obligations at submit time (from the prereg §4.1 / §4.5 / §5.3)

1. **Re-run `sha256sum`** on A/B/C (and `NCA_PREREG.md`) and confirm `classifier.py e7b61df4…` /
   `retrieval.py d43e3bc4…` / anchor `00d9e995…` unchanged — **any mismatch = authorization VOID.**
2. **CODEX GATE (mandatory, pre-submit)** on the NCA/SupCon/mixup branches (three risk surfaces: LOO indexing,
   grad flow + `log_softmax`/`logsumexp` numerics + A3's re-derived align fusion, per-epoch bank cadence),
   iterative until Claude+Codex agree. **If the codex gate forces a code fix, A/B shas change and this freeze
   block MUST be re-issued** before submission.
3. **Smoke (§4.4)** — per-arm 1-seed 3-epoch GPU throwaway (`_smoke_nca` group / `nca_smoke_*` logs) + the
   no-flag Namespace-equivalence check; **delete** all throwaways so they never persist into the §4.3
   collision surface.
4. **Single-submit:** `sbatch scripts/slurm/ncafam_family.sbatch` (24 head-only runs, ~0.33 GPU-h). NO
   `--time`; `PENDING (JobHeldUser)` → **wait for auto-release, never force**.
5. **One test-touch:** the 24 head reads (8 arm×dataset cells × 3 seeds) are the ONLY budgeted NCA-family
   evaluations. The executor transcribes raw both-protocol per-seed numbers (line-numbered) and applies **NO**
   gates/interpretation; the verdict (KS-arm-dead sign bar → FORMAL +0.030/+0.030 conjunct, both protocols,
   per arm×dataset) is rendered by an independent 0-context reviewer against `NCA_PREREG.md` VERBATIM.

**Freeze statements:** ZERO GPU/SLURM/Modal spent at freeze (CPU-only). Prereg NOT modified. `state/` and
`autoresearch/goal_mllm_plus3/state/` not touched. No job submitted. Not pushed.

---

## REFREEZE-1 — A3 mixup dropout-mode fix (codex STOP) — 2026-07-25 NZST

**Trigger:** prereg **§4.5 code-fix clause** ("Blocking findings ⇒ fix the code + re-freeze the shas (§5)
+ re-run this gate") + the **mandatory codex gate's ONE BLOCKING finding** at submit time
(`NCA_SUBMIT_RECORD.md §2.1`): the A3 (manifold-mixup) arm's BCE forward ran with the classifier's
dropout DISABLED, because the upstream FAISS mining call leaves the model in `eval()` mode
(`retrieval.py:330`) and `_manifold_mixup_bce` re-forwarded the dropout-bearing head submodules without
restoring train mode.

**Fix (surgical, inside `_manifold_mixup_bce` only):** restore train mode on EXACTLY the `nn.Dropout`
submodules reachable by the mixup forward, then restore each module's exact prior mode. 18 insertions,
0 deletions, confined to that function. Floor / A1 / A2 never enter the function ⇒ bit-identical
(evidence E1/E2/E3 in `refine-logs/NCA_REFREEZE_FIX.md`). Full recap, diff, per-constraint rationale,
verbatim E1/E2/E3 output, and the RNG-stream statement: **`refine-logs/NCA_REFREEZE_FIX.md`**.

**Re-issued freeze block (new sha A; B/C byte-identical to the original freeze above):**

```
FROZEN 7607863c15bef1d40c1f1f0a5b980123dd84aba260ced6db1fa993f277db5591  refine-logs/NCA_PREREG.md               (unchanged)
A 2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b  src/model/loss.py                       (was e1244ada… — REFREEZE-1)
B b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py                          (unchanged)
C baf41be8a1264445d6bbc63d2eb54966abed6da33c52176ae62d93bf79c14b94  scripts/slurm/ncafam_family.sbatch      (unchanged)
```
Reused-unchanged machinery re-verified byte-identical: `classifier.py e7b61df4…`, `retrieval.py d43e3bc4…`.

**AUTHORIZATION IS VOID** until (1) an independent 0-context re-review approves this fix + re-freeze, AND
(2) the mandatory codex gate (§4.5) re-runs clean on the patched `loss.py`. The prereg (`NCA_PREREG.md`)
was NOT modified by this re-freeze. ZERO GPU/SLURM/Modal spent (CPU-only equivalence harness). No job
submitted. No `state/` mutation. Not pushed.
