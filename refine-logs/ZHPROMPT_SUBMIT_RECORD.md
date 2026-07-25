# ZHPROMPT — SUBMIT EXECUTION RECORD

**Executor:** ZHPROMPT submit executor (conda HateVideo; full-ceremony discipline).
**Frozen object:** `refine-logs/ZHPROMPT_PREREG.md` (FROZEN by `ZHPROMPT_FREEZE.md`, commit `6eb953c`).
**Discipline:** raw facts only in this record; NO gates/deltas/verdict language on the head numbers (rendered by
an independent 0-context reviewer). Incremental local commits, NO push. NO `state/` mutation.

---

## S0 — Authorization re-verify (sha gate + git-clean + collision + inputs) — PASS

**Git HEAD at start:** `6eb953cd8b56da36099074c3b1ddf756785c1ca3`.

### Frozen artifact shas (re-computed on disk; match `ZHPROMPT_FREEZE.md`)

```
FROZEN 07df7c7135e115f41f12be5ee95a06d29fa7360255b0995c5503bf6d3c841aab  refine-logs/ZHPROMPT_PREREG.md      MATCH
A      1c83d4378678afc12c05ce60dfa9e00b810e5398f436a3f7d51151f8ca35dfa1  src/utils/generate_VideoMLLM_embedding_HF.py       MATCH
B      8d9bfd43d0a8f63a021280ffb287cc14fc31853d35893e2fb83926193e6e4cf4  src/utils/generate_VideoMLLM_embedding_lora_HF.py  MATCH
C      f69b1aeb44abb554945fd1aeb524c1f5460950702bfaa44910f3dd720807a113  scripts/slurm/zhprompt_extract_head.sbatch         MATCH
```

### Reused-unchanged machinery (sha re-verify; match `ZHPROMPT_FREEZE.md`)

```
b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py                     MATCH
4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad  scripts/slurm/enc3seed_zh_b3.sbatch  MATCH
2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b  src/model/loss.py                    MATCH
e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378  src/model/classifier.py              MATCH
d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57  src/utils/retrieval.py               MATCH
```

**git-clean:** `git status --porcelain` on run_rac.py, loss.py, classifier.py, retrieval.py, both extractors,
the sbatch, enc3seed_zh_b3.sbatch, ZHPROMPT_PREREG.md = **EMPTY (all clean, working tree == committed).**

### Collision safety (ABSENT) + inputs (PRESENT)

- `data/CLIP_Embedding/MHC_zh/*-zhp.pt` — **ABSENT** (fresh extraction).
- `logging/Retrieval/MHC_zh/RAC_video_zhp*` — **ABSENT** (fresh group).
- `slurm/logs/*zhp*.trainlog` — **ABSENT** (no trainlog collision).
- Banked Arm-F caches `{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt` — **PRESENT** (Jul 2, untouched).
- Banked Arm-L caches `{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` — **PRESENT** (Jul 2, untouched).
- LoRA adapter `logging/lora/MHC_zh/{adapter_config.json,adapter_model.safetensors}` — **PRESENT** (Jul 2 11:06).

### run_one byte-identity

`diff sbatch(L131-172) enc3seed_zh_b3(L42-83)` = **EMPTY (byte-identical).**

**S0 verdict: authorization VERIFIED (not trusted). No mismatch. Proceed to S1 codex gate.**

---
