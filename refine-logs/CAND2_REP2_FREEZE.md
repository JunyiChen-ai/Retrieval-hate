# CAND2_REP2 — Hash-Freeze Record (DRAW-2 replication, HateMM only)

**Executor:** submit executor (this run). **Date:** 2026-07-18. **Mode:** freeze → collision re-check →
single-submit. NO test metric, NO verdict, NO push.

**Authorization chain:**
- Prereg `refine-logs/CAND2_REP2_PREREG.md` at commit `2d15ffb` — sha256
  `365511e91f56577df388266f13d5f8f5d963cf03fc6928be0fd9d576c54a2636` (matches the review target).
- Review `refine-logs/CAND2_REP2_PREREG_REVIEW.md` at commit `e2aee03` (HEAD) — **APPROVED-WITH-NOTES**; the
  reviewer **CONCURRED** with the pre-declared SFT smoke SKIP (13236/13237/13238 precedent), so **no smoke job is
  run**. The live healthy-start gate on the real run is retained and enforced by this executor.
- All four review NOTES are non-blocking (wording/optional-hardening/meta-multiplicity/0.0001-rounding); none
  affect what runs.

## ORCHESTRATOR BINDING (echoed verbatim)

**This is THE single draw-2 attempt. There are NO re-draws under any outcome** (reviewer note-3 hardening). The
prereg's decision tree (§3.4) has four terminal verdicts and no branch loops back to "draw again". `seed: 1` is
pre-committed and single, baked into the hash-frozen yaml A; there is no provision to try another SFT seed and
cherry-pick. A draw-2 FAIL/retire is terminal for this auto-replication ceremony.

## STAGE 1 — sha256 verification (re-run at submit time)

All artifacts re-hashed on disk; every value matches the frozen freeze block (prereg §5.3 / review §"Freeze
block"). **No mismatch ⇒ authorization VALID, proceed.**

### New artifacts (frozen)
```
365511e91f56577df388266f13d5f8f5d963cf03fc6928be0fd9d576c54a2636  refine-logs/CAND2_REP2_PREREG.md            [P]  MATCH
d645de3197739075774b499f335675dad8cd77a3f03b7c6cdc811424506354c6  RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft_rep2.yaml  [A]  MATCH
265f3e736a0e3ae1202cc86bfef562a2e3d830c9d09487eeea9534ab4c763c1e  scripts/slurm/lora_sft_curric_rep2.sbatch   [B]  MATCH
a32fd3bbaaa7140d5d5ffdf1dff3d0df7e26e1fb1ba079c5395e11025861baac  scripts/slurm/enc3seed_lora_curric_rep2.sbatch  [C]  MATCH
```

### Reused-unchanged machinery (must still match)
```
73307ef2e286eddf4fbe12ef13bb3c750f9105d1291494779c7a3a181c91082b  data/lora_sft/HateMM/train_curric.json      (frozen draw-1 curriculum; draw-2 trains THIS)  MATCH
c12c2b6b340151e6c58ed39843aa2cf02a728c17a3296637cae41c2a70b6a4a3  RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft.yaml  (draw-1 fork parent)  MATCH
c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386  scripts/slurm/gen_embed_lora.sbatch         (extraction; out-tag arg 3; NO edit)  MATCH
dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch               (same-code anchor)  MATCH
ba52bc0da3fa14fefa6b93d5d4abcf42e38bcd01261646309ad262a766a6c009  data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct_HF.pt  (frozen mining cache)  MATCH
```

**Verdict: 9/9 sha256 MATCH. Freeze VALID.** The STEP-1b `train_curric.json` bit-exact re-emit gate is enforced
in-job at submit (§4b); the healthy-start check re-verifies the seed took effect.

## STAGE 2 — collision re-check (all targets ABSENT)

- `logging/lora/HateMM_curric_rep2` — **ABSENT** ⇒ fresh SFT, no clobber of draw-1's `HateMM_curric` adapter.
- `data/CLIP_Embedding/HateMM/*LoRA-curric-rep2*.pt` — **ABSENT** ⇒ fresh extraction; frozen/generic/draw-1 caches untouched.
- `logging/Retrieval/HateMM/RAC_video_lora_curric_rep2*` — **ABSENT** ⇒ fresh group; `force=False` never trips an overwrite.
- `slurm/logs/enc3s_*curric-rep2*_seed*_*.trainlog` — **ABSENT** ⇒ no head-log collision.
- `squeue -u $USER` — **clean** (no queued/running jobs at freeze).

## Config sanity (read, not edited)
- yaml A: `seed: 1` (line 50), `dataset: hatemm_lora_curric_train`, `lora_rank 16`, `lora_alpha 32`,
  `output_dir …/logging/lora/HateMM_curric_rep2`, vision+proj frozen — matches F-R0.3 (1-knob diff).
- sbatch B: HateMM hardcoded, no `#SBATCH --time`, ≥20G disk guard, conda `HateVideo`, HF/TRANSFORMERS offline,
  `WANDB_MODE=disabled`; STEP-1a generic build → STEP-1b curriculum re-emit → STEP-2 `python src/train.py <rep2 config>`.
- sbatch C: `LORA=Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF`, `GROUP_NAME=RAC_video_lora_curric_rep2`, seeds
  0/1/2, `--force False`, no `#SBATCH --time`.

Next: single-submit the afterok-wired 3-job chain; verify healthy start; write the submit record.
