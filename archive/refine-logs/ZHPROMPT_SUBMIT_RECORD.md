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
never-2×16-CPU. **Submitted: job `13486` → `PENDING (JobHeldUser)`** (expected; auto-released ~13:12, never forced).

### S3 RESULTS — job 13486 `COMPLETED`, ExitCode `0:0`, Elapsed `00:02:48` (log `slurm/logs/zhpsmoke_13486.out`)

**(i) KS-parity bit-exact — PASS** (log L17-19):
```
[KS-PARITY frozen] n=8 id_order_match=True img_max|Δ|=0.0 text_max|Δ|=0.0 -> PASS
[KS-PARITY LoRA]   n=8 id_order_match=True img_max|Δ|=0.0 text_max|Δ|=0.0 -> PASS
KS_PARITY_OVERALL_PASS
```
Default-arg English re-extraction of BOTH extractors reproduces the banked cache bit-exact (img AND text
max|Δ|==0.0, 8 rows, id order matched) — the machinery/parity guard holds; default == identity confirmed at
RUNTIME (not just code-level).

**(ii) N1 13150-seed0 head repro (mandatory) — PASS, 4dp bit-match** (log L298-301):
```
N1 VALSEL epoch 20: TEST F1 0.8023 acc 0.8322 roc 0.8825
N1 FINAL  epoch 29: TEST F1 0.8181 acc 0.8456 roc 0.9036
N1_REPRO_PASS_4DP_MATCH
```
Current `run_rac.py b85eb72…` / `loss.py 2ae7a73…` on the flags-off (triplet+hybrid) path reproduces 13150 seed0
EXACTLY to 4dp (val-sel ep20 0.8322/0.8023; final ep29 0.8456/0.8181). The pre-NCA→current head-code additive
drift confound is CLOSED directly (not by inheritance) — pairing the `-zhp` heads vs 13150 is sound.

**(iii) Chinese-override 2-video shape/finite — PASS** (log L318-322):
```
[ZH-SHAPE frozen] N=2 img=(2, 3584) text=(2, 3584) finite=True -> PASS
[ZH-SHAPE LoRA]   N=2 img=(2, 3584) text=(2, 3584) finite=True -> PASS
ZH_SHAPE_OVERALL_PASS
```
Assembled ZH text_prompt (log L321) confirms the Chinese instruction + `\n标题:(无)\n文字记录:<Chinese body>`
scaffold reaches the tokenizer with no mojibake/byte-fallback:
`'你正在分析一段…有害意图。\n标题:(无)\n文字记录:比<em…>妈宝男</em>更可怕的人… .🎼…'`.

**Cleanup verified (post-smoke §4.3 surface re-check):** `*-zhp.pt`/`*parity*.pt` in data/CLIP_Embedding = NONE;
`RAC_video_zhp*` + `logging/_smoke_zhp` + `logging/Retrieval/MHC_zh/_smoke_zhp` = NONE; `*zhp*.trainlog` = NONE.
Banked Arm-F/Arm-L caches intact (mtime still 2026-07-02, sizes unchanged). Smoke left NO residue.

**Independent parser validated** (`scratchpad/indep_parse_zhp.py`, split-tokenizer + line numbers, NOT the sbatch
regex): re-derives banked floors exactly — 13150 s0 val-sel ep20 0.8322/0.8023 @L220, final ep29 0.8456/0.8181
@L302; 13115 s0 val-sel ep22 0.7919/0.7412 @L241, final ep29 0.8188/0.7864 @L305. Ready for S5 cross-verify.

**S3 verdict: ALL SMOKE CHECKS PASS. Proceed to S4 real submission.**

Commit: `083138e` (S3 smoke results).

---

## S4 — Real submission (ONE frozen sbatch, UNMODIFIED) — SUBMITTED

**Submit-instant G-repro re-verify (§4.1a) — all MATCH freeze block, git-clean:**
```
07df7c7…  refine-logs/ZHPROMPT_PREREG.md      MATCH
1c83d43…  generate_VideoMLLM_embedding_HF.py            (A)  MATCH
8d9bfd43… generate_VideoMLLM_embedding_lora_HF.py       (B)  MATCH
f69b1ae…  scripts/slurm/zhprompt_extract_head.sbatch    (C)  MATCH
b85eb72…  src/run_rac.py         MATCH   4379224…  enc3seed_zh_b3.sbatch  MATCH
2ae7a73…  loss.py  MATCH   e7b61df4… classifier.py  MATCH   d43e3bc4… retrieval.py  MATCH
```
`git status --porcelain` (frozen + core) = EMPTY (clean). No post-freeze edit ⇒ authorization stands.

**Queue collision check at submit:** `squeue -u jehc223` = EMPTY, `sum_cpus_in_flight=0` ⇒ 8-CPU single job
trivially clears never-2×16-CPU (smoke 13486 already terminal). 

**Command (verbatim, UNMODIFIED):** `sbatch scripts/slurm/zhprompt_extract_head.sbatch`.
**Submitted: job `13487` → `PENDING (JobHeldUser)`** (expected; auto-released ~13:16, never forced). ONE bite.

---

## S5 — RAW transcription (job 13487 terminal) — RAW FACTS ONLY (no gates/deltas/verdict)

**Terminal state:** job `13487` `COMPLETED`, ExitCode `0:0`, Elapsed `03:43:05` (long wall = disk_guard B2-push
phases, DEV-C; extraction + all 6 heads ran; log ends with b2_push done). Job `.out`: `slurm/logs/zhprompt_13487.out`.

### S5.1 KS-parity evidence (from the S3 smoke job 13486; the machinery guard)
`img max|Δ| == 0.0 AND text max|Δ| == 0.0` for BOTH extractors (English-default re-extraction vs banked cache,
n=8, id order matched): `[KS-PARITY frozen] … img_max|Δ|=0.0 text_max|Δ|=0.0 -> PASS` /
`[KS-PARITY LoRA] … img_max|Δ|=0.0 text_max|Δ|=0.0 -> PASS` → `KS_PARITY_OVERALL_PASS` (smoke .out L17-19).
N1 13150-seed0 head repro: `N1_REPRO_PASS_4DP_MATCH` (val-sel ep20 0.8322/0.8023, final ep29 0.8456/0.8181).

### S5.2 Stage-A extraction stats (job 13487 .out; Dv=Dt=3584 all)

| arm | split | N | zero-vector videos | .out line |
|---|---|---|---|---|
| Arm-F `…_HF-zhp` | train | 579 | **0** | L5220 |
| Arm-F `…_HF-zhp` | dev_seen | 78 | **0** | L5226 |
| Arm-F `…_HF-zhp` | test_seen | 149 | **0** | L5235 |
| Arm-L `…-LoRA_HF-zhp` | train | 579 | **0** | L5311 |
| Arm-L `…-LoRA_HF-zhp` | dev_seen | 78 | **0** | L5317 |
| Arm-L `…-LoRA_HF-zhp` | test_seen | 149 | **0** | L5326 |

Stage-A shape sanity (.out L5328-5334): all 6 `-zhp` caches `img=text=(N,3584)`, `ids==N`, `N>0` → `SHAPE_SANITY_OK`.
(Split counts train 579 / val 78 / test 149 = 806 match prereg §1 Stage 0.)

### S5.3 Six head runs — RAW per-seed, BOTH protocols, re-read from PRIMARY trainlogs with line numbers

Two independent parses agree on every value: **[P1]** = independent split-tokenizer parser
`scratchpad/indep_parse_zhp.py` (VAL/TEST line numbers cited from the primary `.trainlog`); **[P2]** = the sbatch's
embedded regex parser output in `zhprompt_13487.out`. Protocol rule (both parsers): val-sel = epoch≥warmup5 with
max Val_Retrieval acc (roc tie-break); final = max epoch (29). **[P1]==[P2] EXACT, all 6 runs, all metrics.**

**Arm-L (LoRA Chinese-prompt) — trainlogs `enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF-zhp_seed{0,1,2}_13487.trainlog`:**

| seed | val-sel ep | val-sel acc / mF1 | [P1] VAL/TEST line | final ep | final acc / mF1 | [P1] TEST line | [P2] .out line |
|---|---|---|---|---|---|---|---|
| 0 | 7 | 0.7852 / 0.7541 | L99/L100 | 29 | 0.8389 / 0.8065 | L299 | .out L6535-6537 |
| 1 | 8 | 0.8255 / 0.8002 | L108/L109 | 29 | 0.8255 / 0.7904 | L299 | .out L6813-6815 |
| 2 | 5 | 0.7785 / 0.7450 | L79/L80 | 29 | 0.8389 / 0.8065 | L297 | .out L7089-7091 |
| **mean** | | **0.7964 / 0.7664** | | | **0.8344 / 0.8011** | | |

**Arm-F (frozen Chinese-prompt) — trainlogs `enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF-zhp_seed{0,1,2}_13487.trainlog`:**

| seed | val-sel ep | val-sel acc / mF1 | [P1] VAL/TEST line | final ep | final acc / mF1 | [P1] TEST line | [P2] .out line |
|---|---|---|---|---|---|---|---|
| 0 | 25 | 0.7785 / 0.7203 | L261/L262 | 29 | 0.8121 / 0.7608 | L299 | .out L5703-5705 |
| 1 | 7 | 0.7718 / 0.7327 | L99/L100 | 29 | 0.8054 / 0.7613 | L299 | .out L5981-5983 |
| 2 | 5 | 0.7584 / 0.7058 | L79/L80 | 29 | 0.7785 / 0.7158 | L297 | .out L6257-6259 |
| **mean** | | **0.7696 / 0.7196** | | | **0.7987 / 0.7460** | | |

(Per-seed roc, from [P1]/[P2], for completeness — Arm-L val-sel roc s0/s1/s2 = 0.8594/0.8981/0.8712, final =
0.9083/0.8818/0.9028; Arm-F val-sel roc = 0.8929/0.8417/0.8494, final = 0.8915/0.8880/0.8675.)

**Reference floors (VERBATIM from prereg §2, pre-existing facts; NO delta computed here — the independent
0-context reviewer applies §3 against the prereg):** Arm-L floor = 13150 (val-sel mean 0.8322/0.8015, final mean
0.8456/0.8173); Arm-F floor = 13115 (val-sel mean 0.8031/0.7681, final mean 0.8031/0.7712).

### S5.4 Post-run disk state
Six `-zhp` caches now PRESENT in `data/CLIP_Embedding/MHC_zh/` (the real extraction outputs, expected to persist).
Banked English floors `…_HF.pt` / `…-LoRA_HF.pt` intact (mtime 2026-07-02, unchanged). New head dirs live under
`logging/Retrieval/MHC_zh/RAC_video_zhp/`.

**S5: raw transcription complete. Verdict (KS-parity → KS-dead → FORMAL, per arm) is NOT rendered here — it is
rendered by an independent 0-context reviewer against `ZHPROMPT_PREREG.md` VERBATIM.**

Commit: `e03bd25` (S5 raw transcription).

---

## S6 — Closeout

### Frozen sha chain (S0 == submit-instant, all MATCH `ZHPROMPT_FREEZE.md`, no post-freeze edit)
```
FROZEN 07df7c7…  refine-logs/ZHPROMPT_PREREG.md
A      1c83d43…  src/utils/generate_VideoMLLM_embedding_HF.py
B      8d9bfd43… src/utils/generate_VideoMLLM_embedding_lora_HF.py
C      f69b1ae…  scripts/slurm/zhprompt_extract_head.sbatch
reused b85eb72…  src/run_rac.py     4379224…  enc3seed_zh_b3.sbatch
       2ae7a73…  loss.py   e7b61df4… classifier.py   d43e3bc4… retrieval.py
```

### Job IDs
- **Smoke (throwaway):** `13486` — COMPLETED 0:0, 00:02:48. KS-parity bit-exact PASS (both arms, img+text
  max|Δ|=0.0), N1 13150-seed0 repro PASS (4dp bit-match), Chinese-shape PASS; cleanup left no residue.
- **Real bite:** `13487` — COMPLETED 0:0, 03:43:05 (long wall = disk_guard B2-push, DEV-C). Extraction (zero-vec=0
  all 6 caches) + shape-sanity OK + 6 head runs.

### RAW head table (per-seed, both protocols; NO deltas/gates/verdict — reviewer's job)
Arm-L (LoRA, PRIMARY): val-sel s0/s1/s2 acc 0.7852/0.8255/0.7785 mF1 0.7541/0.8002/0.7450 (mean 0.7964/0.7664);
final s0/s1/s2 acc 0.8389/0.8255/0.8389 mF1 0.8065/0.7904/0.8065 (mean 0.8344/0.8011).
Arm-F (frozen, control): val-sel s0/s1/s2 acc 0.7785/0.7718/0.7584 mF1 0.7203/0.7327/0.7058 (mean 0.7696/0.7196);
final s0/s1/s2 acc 0.8121/0.8054/0.7785 mF1 0.7608/0.7613/0.7158 (mean 0.7987/0.7460).
(Reference floors, prereg §2: Arm-L=13150 val-sel 0.8322/0.8015 final 0.8456/0.8173; Arm-F=13115 val-sel
0.8031/0.7681 final 0.8031/0.7712. Delta/verdict deferred to the independent 0-context reviewer.)

### Deviations from the task chain
**NONE material.** Notes: (1) codex gate ran on the account's configured model `gpt-5.6-sol` xhigh (the
`gpt-5.2*` model ids the task suggested are not available to this ChatGPT-account Codex; `gpt-5.6-sol` is a
gpt-5.x-class model at xhigh, satisfying the gate mandate) — verdict NO FINDINGS, cross-checked by Claude. (2)
KS-parity ran BOTH extractors × BOTH streams (stricter than the task's "one stream"), per prereg §3.3/§4.4.1. (3)
N1 (review note, orchestrator-made-mandatory) executed inside the smoke and PASSED 4dp. No frozen file edited; no
`state/` mutation; commits local only (not pushed).

### Local commit chain (this record; NOT pushed)
`fa75776` (S0) → `ac23920`… see per-stage. Final closeout commit below.

**S6: chain complete through the single budgeted test-touch. Handoff to the independent 0-context verdict reviewer.**

