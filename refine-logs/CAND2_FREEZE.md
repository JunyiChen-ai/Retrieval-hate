# CAND-2 Curriculum LoRA-SFT — HASH-FREEZE (submit executor)

**Role:** submit executor (freeze → collision → smoke → single-submit → verify start). ZERO user
interaction. NO push. NO test metric read. NO verdict produced.
**Timestamp (UTC):** 2026-07-17T17:08:37Z.
**Prereg:** `refine-logs/CAND2_CURRICULUM_PREREG.md` commit `76ef0e2`,
sha256 `e5a689d9ede0a79e89eb041f028228e70cdc821029349387e7ec9fdff939790e` — **MATCHES on disk.**
**Review:** `refine-logs/CAND2_PREREG_REVIEW.md` commit `c1315cb` — **APPROVED-WITH-NOTES** (3 non-blocking notes).
**Git HEAD at freeze:** `c1315cb9ea804166bc788af591379537193ba83e`.

## VERDICT: **FREEZE PASS** — all freeze-block + reused-machinery shas match; builder bit-exact idempotent.

---

## 1. Freeze-block re-hash (prereg §5.3 + §5.1 I/J) — every artifact MATCHES

| # | path | expected sha256 | on-disk | match |
|---|---|---|---|---|
| P | `refine-logs/CAND2_CURRICULUM_PREREG.md` | `e5a689d9…f939790e` | `e5a689d9…f939790e` | ✔ |
| A | `src/utils/build_curriculum_sft_data.py` | `085384f5…990f66e8` | `085384f5…990f66e8` | ✔ |
| B | `RA-HMD/…/my_configs/hatevideo/mhc_zh_qwen25vl_lora_curric_sft.yaml` | `ac1c5962…8815fa6d` | `ac1c5962…8815fa6d` | ✔ |
| C | `RA-HMD/…/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft.yaml` | `c12c2b6b…a70b6a4a3`* | `c12c2b6b…70b6a4a3` | ✔ |
| D | `scripts/slurm/lora_sft_curric.sbatch` | `6a5abb9e…59ffd57a` | `6a5abb9e…59ffd57a` | ✔ |
| E | `scripts/slurm/enc3seed_lora_curric.sbatch` | `00d9e995…b306f02`* | `00d9e995…8b306f02` | ✔ |
| F | `data/lora_sft/MHC_zh/train_curric.json` (579 rows) | `c8260dd3…7029bc0d` | `c8260dd3…7029bc0d` | ✔ |
| G | `data/lora_sft/HateMM/train_curric.json` (743 rows) | `73307ef2…c91082b`* | `73307ef2…1c91082b` | ✔ |
| H | `RA-HMD/…/data/dataset_info.json` (submodule) | `c2b99d25…0229fd6`* | `c2b99d25…c0229fd6` | ✔ |
| I | `refine-logs/CAND2_KC20_MHC_zh.json` | `38b21db5…a1a0f6f33` | `38b21db5…a1a0f6f33` | ✔ |
| J | `refine-logs/CAND2_KC20_HateMM.json` | `14967d53…3b4048cf6b` | `14967d53…3b4048cf6b` | ✔ |

Full sha256 (verbatim from `sha256sum`):
```
e5a689d9ede0a79e89eb041f028228e70cdc821029349387e7ec9fdff939790e  refine-logs/CAND2_CURRICULUM_PREREG.md
085384f5534ffae9969c95211f7eaefca5cc3d54278734ba76457b84990f66e8  src/utils/build_curriculum_sft_data.py
ac1c596293877e827c9db96bec8aefc8f36ebe5e6d3aa95544889be48815fa6d  .../mhc_zh_qwen25vl_lora_curric_sft.yaml
c12c2b6b340151e6c58ed39843aa2cf02a728c17a3296637cae41c2a70b6a4a3  .../hatemm_qwen25vl_lora_curric_sft.yaml
6a5abb9e7d7427f7e4e9874ee429eaed4ed269e342cff5b6df14d40e59ffd57a  scripts/slurm/lora_sft_curric.sbatch
00d9e9956549bdf97c6b8913d42d87811f4a2f150e9f459e8b8b86978b306f02  scripts/slurm/enc3seed_lora_curric.sbatch
c8260dd3f5a98394c6ef3d7f08e091dad5810e1d22d58db24ac5654d7029bc0d  data/lora_sft/MHC_zh/train_curric.json
73307ef2e286eddf4fbe12ef13bb3c750f9105d1291494779c7a3a181c91082b  data/lora_sft/HateMM/train_curric.json
c2b99d2521b1785a2df8da0fd62b13ea4c0dea086bd783cd724619aec0229fd6  RA-HMD/LLAMA-FACTORY-Ver202512/data/dataset_info.json
38b21db5909d4affc9f57c3a9286eab0e807b00c6b7a0d7de599d6ca1a0f6f33  refine-logs/CAND2_KC20_MHC_zh.json
14967d5313e044a556a8caf365ab4ab00178d51b0ce3fd67d7a6263b4048cf6b  refine-logs/CAND2_KC20_HateMM.json
```

## 2. Reused-unchanged machinery (prereg §5.2) — verify-only, NOT edited, all MATCH

```
c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386  scripts/slurm/gen_embed_lora.sbatch
dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch
135a6e243761fa832c712bf4d02478ac34bc49cabaf888a7b5fe465695d3861e  data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct_HF.pt
ba52bc0da3fa14fefa6b93d5d4abcf42e38bcd01261646309ad262a766a6c009  data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct_HF.pt
```

## 3. DEV-1 fork-source `train.json` (load-bearing single-manipulated-variable guarantee) — MATCH

```
ecfa663d6da6d151e88303c45c37cc0c475186c8bf787e59a3e0db50d31b10d0  data/lora_sft/MHC_zh/train.json   (prereg §1.1: ecfa663d…31b10d0)
93c6d3d1bffbca22b2dd8beba57a33575a48d8ca61d8d56e3148fecdbb93973a  data/lora_sft/HateMM/train.json  (== LORA_HATEMM_PREREG.md pin)
```

## 4. Builder idempotency re-run (prereg §4.1b / STAGE 1 mandate) — bit-exact

Re-ran `build_curriculum_sft_data.py --dataset {MHC_zh,HateMM} --mode softconf` once each (CPU-only,
`HateVideo` interpreter, HF offline). Builder-printed shas equal the frozen F/G bit-exact; on-disk F/G/I/J and
additive H unchanged after the re-run:

- MHC_zh: `train_curric.json sha256 c8260dd3…7029bc0d`; K-C2-0 LOO-err 0.2073 / c-Gini 0.5634 / cov 0.6667 /
  hard-head ×2.1092 / **PASS**.
- HateMM: `train_curric.json sha256 73307ef2…1c91082b`; K-C2-0 LOO-err 0.1935 / c-Gini 0.6497 / cov 0.6756 /
  hard-head ×2.0807 / **PASS**.

Post-re-run on-disk re-hash of F/G/I/J/H all equal §1; git-clean for the tracked targets (F/G/I/J); H lives in
the `RA-HMD/LLAMA-FACTORY-Ver202512` submodule (additive register is idempotent, sha unchanged).

## 5. Review note to carry forward (per team-lead instruction — non-blocking)

Echoing review NON-BLOCKING NOTE (2) for the verdict reviewer's awareness: HateMM KC20 JSON records
`n_train_cache = 744` vs `n_train_sft = 743`, `n_anchor_missing_from_cache = 0` — all 743 SFT anchors are present
in the frozen cache; one cache-only train video acts solely as a potential LOO neighbor. Train-only, no leakage,
predates cand-2. Benign. (Notes 1 and 3 — a 0.1pt class-balance rounding slip, and the DEV-4 sweep grid not
enumerated — are likewise non-blocking transparency items.)

---

**FREEZE OUTCOME: PASS.** Authorization to proceed to collision re-check → smoke → single-submit stands. No
mismatch found. No GPU/SLURM spent in this stage (pure-CPU sha256 + one idempotent builder re-run). Not pushed.
