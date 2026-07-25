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

Commit: `fa75776` (S0 authorization record).

---

## S1 — Mandatory codex gate (frozen extractor diffs + sbatch) — PASS, NO FINDINGS

**Reviewer:** codex `gpt-5.6-sol`, reasoning `xhigh`, sandbox read-only, cwd /data/jehc223/RGCL.
**Thread:** `019f96c8-3758-7a92-a671-b712a8eecaef`.
**Focus (task S1 a-d):** (a) default==identity byte-exact on every path incl. edge cases; (b) no Chinese-override
leakage into English-default runs; (c) `-zhp` cache-naming collision safety + LoRA adapter path; (d) head config
rows byte-match the enc3seed precedent.

**Codex verdict per area (line-cited against source):**
- **(a) CLEAN** — defaults are exactly `IMG_INSTRUCTION`/`TEXT_INSTRUCTION`/`"Title: "`/`"Transcript: "`/`"(none)"`
  (HF.py:96-124, lora_HF.py:119-153); the new assembly (HF.py:397-401, lora_HF.py:420-424) reduces byte-for-byte
  to the old literal. All edge cases match: missing title→"" (HF.py:177), empty transcript→"(none)" (HF.py:176),
  one-slot-empty substitutes only that slot, and inputs equal to "(none)"/whitespace/newline are truthy and
  preserved identically. NO divergent input exists.
- **(b) CLEAN** — each `parse_args_sys()` builds a fresh parser/namespace; constants never reassigned at runtime;
  Arm-F/Arm-L are separate processes (sbatch:58-67 / 75-85); Chinese strings are UNEXPORTED shell vars
  (sbatch:42-46), exported env (sbatch:32-35) carries no prompt text; saved caches store only ids/feats/labels
  (HF.py:476-484). No channel carries Chinese text into a default/parity run.
- **(c) CLEAN** — tags distinct (sbatch:48-51); output `{split}_{tag}.pt` ends `_HF-zhp.pt`/`-LoRA_HF-zhp.pt`,
  cannot equal banked `_HF.pt`/`-LoRA_HF.pt`; two arms cannot collide. Arm-L validates `logging/lora/MHC_zh`
  (`exit 2` if missing), extractor does `PeftModel.from_pretrained`+`merge_and_unload` (lora_HF.py:472-493).
- **(d) CLEAN** — run_one (sbatch:131-172) vs enc3seed_zh_b3 (L42-83): **empty diff, both 2198 bytes, identical
  SHA-256 `286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101`** (codex's independent hash). Only
  per-arm var = `--model`/`--exp_comment "_${MODEL}"`; head loads cache via `--model` (run_rac.py:1083-1085,
  dataset.py:499-503/605-609). Shape-sanity block (sbatch:90-110) asserts `(N,3584)` all 6 caches + `exit 3`
  before any head run. Chinese literals valid UTF-8, no CR, no active metachars. `--num_frames 8`/`--device cuda`
  both arms; omitted `--max_pixels` keeps floor default `360*420=151200`.

**Codex FINAL: NO FINDINGS (no P1/P2/P3).** _(record continues below)_

**Claude independent cross-check (agree):** read all three files in full; git diff `546518a~1..546518a` shows the
edit is ADDITIVE-ONLY (5 argparse keys defaulting to the English constants + the process_split assembly swap) in
BOTH extractors, math/pooling/forward untouched; my own `diff` of run_one vs enc3seed_zh_b3 = EMPTY
(RUN_ONE_DIFF_EMPTY_BYTE_IDENTICAL). Claude + Codex AGREE → **S1 gate PASS**. No code fix ⇒ §4.6 re-freeze NOT
triggered; frozen shas stand. Proceed to S2.

Commit: `ae036ec` (S1 codex gate record).

---

## S2 — CPU smoke (py_compile / argparse identity / bash -n / collision + inputs) — PASS

Run under `conda activate HateVideo` (CPU-only, $0 GPU).

- **py_compile** both extractors → `PY_COMPILE_PASS`.
- **bash -n** `zhprompt_extract_head.sbatch` → `BASH_N_SYNTAX_OK`; `CONFIGS` word-split = **6 rows** (2 arms × 3 seeds).
- **argparse default==identity, byte-wise** (`parse_args_sys([])`, both extractors): all 5 args equal the deployed
  constants/literals (`img==IMG_INSTRUCTION`, `text==TEXT_INSTRUCTION`, `title_label=='Title: '`,
  `transcript_label=='Transcript: '`, `none_placeholder=='(none)'`); the assembled `text_prompt` **byte-matches**
  the pre-edit deployed literal (utf-8 byte compare) across **7 cases** (both empty, title-only, transcript-only,
  both present, literal "(none)" strings, embedded-newline strings, whitespace-only) → `text_assembly
  byte-identical: True` both; `img_default==IMG_INSTRUCTION: True` both. **`OVERALL_IDENTITY_OK`.**
- **Chinese-override sanity (CPU):** override `parse_args_sys([...])` assembles `<TEXT_ZH>\n标题:(无)\n文字记录:<body>`
  and `img==IMG_ZH` for both extractors → True.
- **Collision re-check (this instant):** `*-zhp.pt` **ABSENT**, `RAC_video_zhp*` group **ABSENT**, `*zhp*.trainlog`
  **ABSENT**.
- **Input presence (this instant):** all 6 banked caches (Arm-F/Arm-L × {train,dev_seen,test_seen}) **PRESENT**;
  LoRA adapter `logging/lora/MHC_zh/{adapter_config.json,adapter_model.safetensors}` **PRESENT**.

**S2 verdict: PASS. Proceed to S3 GPU smoke (KS-parity + N1 repro + Chinese-shape).**

Commit: `ac23920` (S2 CPU smoke record).

---

## S3 — GPU smoke (ONE throwaway job; KS-parity + N1 repro + Chinese-shape) — SUBMITTED

**Smoke sbatch (throwaway, NOT a frozen artifact):** `scratchpad/zhprompt_smoke.sbatch` — 8 CPU / 64 G / 1 A100,
NO `--time`, no disk_guard (throwaway). All artifacts under `logging/_smoke_zhp` + `logging/Retrieval/MHC_zh/
_smoke_zhp`, `rm -rf`'d at end; extractions write to `--EXP_FOLDER logging/_smoke_zhp` (NOT data/CLIP_Embedding),
so no `-zhp`/parity cache ever lands in the real §4.3 surface. `bash -n` = SMOKE_BASH_N_OK.

Three checks in ONE job:
- **(i) KS-parity bit-exact** — BOTH extractors run with **English defaults** (no overrides), `--splits test
  --limit 8`, compared against the banked `test_seen_Qwen2.5-VL-7B-Instruct_HF.pt` (frozen) /
  `…-LoRA_HF.pt` (LoRA), first-8 rows matched by id order. Asserts `img max|Δ|==0.0 AND text max|Δ|==0.0` per
  arm (READOUT 13468 R0 precedent). Prints `KS_PARITY_OVERALL_PASS/FAIL`.
- **(ii) N1 (mandatory) 13150-seed0 head repro** — head command flags VERBATIM from `enc3seed_zh_b3` run_one
  (byte-identical to zhprompt run_one), `--model Qwen2.5-VL-7B-Instruct-LoRA_HF` (BANKED English LoRA cache,
  read-only), `--seed 0`, throwaway `--group_name _smoke_zhp`. Expect 4dp match vs 13150 seed0 (val-sel ep20
  acc 0.8322 mF1 0.8023; final ep29 acc 0.8456 mF1 0.8181). Prints `N1_REPRO_PASS_4DP_MATCH/FAIL_MISMATCH`.
  Closes the run_rac.py/loss.py additive-drift confound directly (review N1, now mandatory).
- **(iii) Chinese-override 2-video shape/finite** — both extractors with the frozen Chinese overrides,
  `--splits test --limit 2`; asserts shape `(2,3584)` img+text, all finite; prints one assembled Chinese
  `text_prompt`. Prints `ZH_SHAPE_OVERALL_PASS/FAIL`.

**Queue collision check at submit:** `squeue -u jehc223` = EMPTY (0 CPUs in flight) ⇒ 8-CPU smoke trivially clears
never-2×16-CPU. **Submitted: job `13486` → `PENDING (JobHeldUser)`** (expected; waiting for auto-release, never
forced). ANY smoke FAIL ⇒ STOP.

_Awaiting smoke terminal state → transcribe (i)/(ii)/(iii) results below._

---
