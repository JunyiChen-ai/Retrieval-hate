# BIDIR_STAGE1_PREREG — HASH-FREEZE

**Frozen by:** independent 0-context pre-registration reviewer. **Date:** 2026-07-25 NZST.
**Verdict:** `APPROVED-WITH-NOTES` (see `refine-logs/BIDIR_STAGE1_PREREG_REVIEW.md`; all 4 notes non-blocking).
**Prereg introduced at commit:** `a7bb2a1` ("prereg: bidir mask-flip stage-1 (ZH+HateMM) DRAFT, unreviewed").
**Repo HEAD at freeze:** `1b3e0c6` (a later sibling prereg; the bidir files are committed and clean).
**CPU-only; no GPU/SLURM/Modal spent; `state/` not touched; prereg NOT modified; not pushed.**

All shas below were recomputed on disk at freeze time and MATCH the prereg §5.1/§5.2/§5.3. The prereg self-sha is
the current committed on-disk content (`a7bb2a1` blob, unchanged).

```
FROZEN 3c532e5370e52b2ed53e0bcc2ad63d2958823f5aca0e6f710495d8cf55565142  refine-logs/BIDIR_STAGE1_PREREG.md (commit a7bb2a1)
A  36cedbac365b2b13c945adbe3437efdc61d8be15ecc85878eb9614225abe367b  src/utils/bidir_patch.py
A2 03f39e09c417bbea291f3c06b787f5220693568cd613705693df1c2bf23e020d  src/utils/generate_VideoMLLM_embedding_bidir_HF.py
B  0f17fce6910981bbc4c5942eae3b18947151bc6990ceee401fc86b252a287ecd  scripts/slurm/gen_embed_mllm_bidir.sbatch
C  82a69e74d570df59a1b686891814c7756b15755901d2a645bb1d3f0164a51264  scripts/slurm/enc3seed_bidir.sbatch
```

**Reused-unchanged machinery (re-verify at submit; do NOT edit):**

```
b6b61a3fa4214f28d2098ca305ca9a981934445afc81ccc5ac3939f8c1fb0ec6  src/utils/generate_VideoMLLM_embedding_lora_HF.py   (causal extractor; operator imported VERBATIM; NOT edited)
dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch                        (head same-code anchor; run_one block sha 13e34e4e…)
f9384d8dbdb8c1e315bb40a96952f068830c9a98cd6107f3b2ac2458e7fc477b  logging/lora/MHC_zh/adapter_config.json              (ZH adapter == 13150; READ-ONLY)
35a510f4ad84542c798939cfdb340b00317a5b8a2c670b07ced8d1869dd7b438  logging/lora/MHC_zh/adapter_model.safetensors        (ZH adapter weights; READ-ONLY)
eaca36dd5cef2a4ff866a0398680d420adb157be815fe335500a387bbf9037b8  logging/lora/HateMM_curric/adapter_config.json       (HateMM curric adapter == 13241; READ-ONLY)
6571d132ef3218e4bdfcee98aab468df21f8aa83b16d623dd2098f8486394efa  logging/lora/HateMM_curric/adapter_model.safetensors (HateMM curric weights; READ-ONLY)
```

The `run_one()`…`PY` head block of `enc3seed_bidir.sbatch` is byte-identical (block sha256
`13e34e4e93c6a76988557e1c609fd54e0353c627fd36eb1c5b9e26ed187c3feb`, 41 lines) to the same block in
`enc3seed.sbatch` AND `enc3seed_lora_curric.sbatch`.

**Banked causal paired-floor caches (must stay present/untouched; distinct `-bidir` out-tags cannot clobber):**
`data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` (present, Jul 2) and
`data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` (present, Jul 18).

**Environment pin:** `transformers 4.49.0`, env `HateVideo`
(`.../miniconda3/envs/HateVideo/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`,
2112 lines) — the source the patch was reviewed against.

**Executor gate (verbatim from §5.3):** re-run `sha256sum` on A/A2/B/C (and this prereg file) + confirm the causal
extractor sha `b6b61a3f…`, the head anchor `dbe3fb81…`, and both adapter shas unchanged at submit time; any
mismatch = authorization VOID. Submission also requires the F0.3 GO-IF gates (the mandatory patch review — the
transformers-source check is discharged by `BIDIR_STAGE1_PREREG_REVIEW.md` Check 1 — plus the one-line D7 user
sub-ruling), and the §6 sequencing (this 8-CPU chain behind / after the parallel readout-recon chain; never two
16-CPU jobs; NO `--time`; JobHeldUser → wait, never force). Re-run the CPU patch self-test
(`python src/utils/bidir_patch.py` → `VERDICT: PASS`) and the smoke plan §4.4 before the real submits.
