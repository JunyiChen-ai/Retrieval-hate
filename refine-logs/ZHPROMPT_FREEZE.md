# ZHPROMPT — Chinese-Instruction Re-Extraction Pre-Registration — HASH-FREEZE

**Frozen by:** independent 0-context pre-registration reviewer.
**Date:** 2026-07-25 NZST.
**Verdict:** `APPROVED-WITH-NOTES` (see `refine-logs/ZHPROMPT_PREREG_REVIEW.md`; three non-blocking notes N1-N3).
**Prereg (frozen object):** `refine-logs/ZHPROMPT_PREREG.md` (commit `546518a`; on-disk == committed, unmodified).
**Recon:** `refine-logs/ZHPROMPT_FORENSIC_RECON.md` (`47a4e30`).

All shas below were recomputed on disk at freeze time and **match** the prereg §5 freeze block. The prereg was
**NOT** modified (review mandate) — the FROZEN self-sha is the current on-disk sha of the unmodified file, and
the prereg §5.3 self-hash placeholder is left as-is (part of the frozen bytes).

```
FROZEN 07df7c7135e115f41f12be5ee95a06d29fa7360255b0995c5503bf6d3c841aab  refine-logs/ZHPROMPT_PREREG.md
A 1c83d4378678afc12c05ce60dfa9e00b810e5398f436a3f7d51151f8ca35dfa1  src/utils/generate_VideoMLLM_embedding_HF.py
B 8d9bfd43d0a8f63a021280ffb287cc14fc31853d35893e2fb83926193e6e4cf4  src/utils/generate_VideoMLLM_embedding_lora_HF.py
C f69b1aeb44abb554945fd1aeb524c1f5460950702bfaa44910f3dd720807a113  scripts/slurm/zhprompt_extract_head.sbatch
```

### Reused-unchanged machinery (verified git-clean + sha at freeze; do NOT edit)

```
b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py                     (deployed head runner; NCA/head-recipe keys inert)
4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad  scripts/slurm/enc3seed_zh_b3.sbatch (same-code anchor; produced floors 13115/13150)
2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b  src/model/loss.py                  (git-clean; triplet+hybrid path unchanged)
e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378  src/model/classifier.py            (git-clean)
d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57  src/utils/retrieval.py             (git-clean)
```

### Banked paired-control inputs (read-only; NOT clobbered)

- Arm-L floor (job 13150): `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` — present, untouched.
- Arm-F floor (job 13115): `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt` — present, untouched.
- Floor trainlogs: `slurm/logs/enc3s_MHC_zh_…-LoRA_HF_seed{0,1,2}_13150.trainlog`,
  `slurm/logs/enc3s_MHC_zh_…_HF_seed{0,1,2}_13115.trainlog` — independently re-parsed to 4dp, bit-match §2.1/§2.2.
- LoRA adapter (Arm-L, merged read-only): `logging/lora/MHC_zh/{adapter_config.json,adapter_model.safetensors}` (2026-07-02) — present.

### Executor obligations at submit time (from the prereg §4.1 / §4.5 / §4.6 / §6)

1. **Re-run `sha256sum`** on A/B/C (and `ZHPROMPT_PREREG.md`) and confirm `run_rac.py b85eb72…`, `loss.py
   2ae7a73f…`, `classifier.py e7b61df4…`, `retrieval.py d43e3bc4…`, anchor `enc3seed_zh_b3 4379224…` unchanged
   — **any mismatch = authorization VOID.**
2. **CODEX GATE (mandatory, pre-submit)** on the extractor prompt-plumbing + sbatch (default==identity / no
   other constant consumer / Chinese-override flow / single-quote UTF-8 intact / head plumbing), iterative
   until Claude+Codex agree. **If the codex gate forces a code fix, A/B (and possibly C) shas change and this
   freeze block MUST be re-issued** (§4.6) before submission.
3. **Smoke (§4.4)** — (i) KS-parity bit-exact: English-default re-extraction of BOTH extractors reproduces the
   banked cache `img/text max|Δ|==0.0` (HALT on fail); (ii) Chinese-prompt sanity (no crash/OOM, shapes
   (8,3584), one printed assembled `text_prompt`). Delete `logging/_smoke_zhp` so it never persists into §4.3.
   **[Review N1 — recommended] also run the 1-seed no-flag head on the banked English LoRA cache and confirm it
   bit-reproduces 13150 seed0** (§4.1c), to directly close the run_rac.py/loss.py-drift pairing confound.
4. **Single-submit:** `sbatch scripts/slurm/zhprompt_extract_head.sbatch` (ONE job: extract Arm-F + Arm-L →
   Stage-A shape sanity → 6 head runs, ~1.1 GPU-h). NO `--time`; `PENDING (JobHeldUser)` → **wait for
   auto-release, never force**.
5. **One test-touch:** the 6 head reads ({Arm-F,Arm-L}×seed{0,1,2}) are the ONLY budgeted evaluations. The
   executor transcribes raw both-protocol per-seed numbers (line-numbered) and applies **NO** gates. The
   verdict (KS-parity → KS-dead per-arm, NO auto-defund → FORMAL +0.030/+0.030 conjunct both protocols, per
   arm vs its own floor) is rendered by an independent 0-context reviewer against `ZHPROMPT_PREREG.md` VERBATIM.

**Freeze statements:** ZERO GPU/SLURM/Modal spent at freeze (CPU-only). Prereg NOT modified. **Any post-freeze
edit to A/B/C or the prereg voids authorization** (§4.6 re-freeze clause). `state/` and
`autoresearch/goal_mllm_plus3/state/` not touched. `research-wiki/` not touched. No job submitted. Not pushed.
