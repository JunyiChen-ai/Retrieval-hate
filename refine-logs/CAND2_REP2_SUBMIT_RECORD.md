# CAND2_REP2 — Submit Record (DRAW-2 replication, HateMM only)

**Provenance of THIS record:** authored **post-hoc by a records clerk** from on-disk evidence. The submit
executor queued the 3-job afterok chain correctly (freeze `6c11988` re-hash → collision re-check → single-submit)
but **exited without writing the submit record**; this file reconstructs the submission from SLURM state, the
frozen artifacts, and the live `13244` SFT log. **No job action, no push, no test metric, no verdict** was taken
by this clerk — read-only reconstruction plus this document + its commit.
**Date:** 2026-07-18. **Branch:** main.

---

## 1. Authorization chain (verified on disk)

| doc | path | commit | status |
|---|---|---|---|
| Prereg | `refine-logs/CAND2_REP2_PREREG.md` | `2d15ffb` | sha256 `365511e91f56577df388266f13d5f8f5d963cf03fc6928be0fd9d576c54a2636` (== review target) |
| Review | `refine-logs/CAND2_REP2_PREREG_REVIEW.md` | `e2aee03` | **APPROVED-WITH-NOTES** (4 NOTES, all non-blocking); reviewer **CONCURRED** with the pre-declared SFT smoke SKIP |
| Freeze | `refine-logs/CAND2_REP2_FREEZE.md` | `6c11988` | **9/9 sha256 MATCH**, freeze VALID; collision targets ABSENT |

Review verdict cite (verbatim head): *"APPROVED-WITH-NOTES … Every load-bearing claim was re-verified from disk
and holds: hashes match, the seed-knob is the genuine and only independent-draw lever, the head code is
byte-identical to the banked controls, the bars are decidable with no interpretive freedom."* The four NOTES
(F-R0.3 CLI-wording; §4b in-script sha assert; explicit serial-re-draw foreclosure; 0.0001 ΔmF1 rounding) are
non-blocking and change nothing about what runs. HEAD at submit time is `87520fba` (post-freeze errata F58);
the frozen artifacts A/B/C are unmodified by those later commits.

## 2. ORCHESTRATOR BINDING (echoed verbatim — single draw-2 attempt)

> **This is THE single draw-2 attempt. There are NO re-draws under any outcome** (reviewer note-3 hardening).
> The prereg's decision tree (§3.4) has four terminal verdicts and no branch loops back to "draw again". `seed: 1`
> is pre-committed and single, baked into the hash-frozen yaml A; there is no provision to try another SFT seed and
> cherry-pick. A draw-2 FAIL/retire is terminal for this auto-replication ceremony.

Enforcement record logged in `autoresearch/goal_mllm_plus3/logs/orchestrator.jsonl` (line 36,
`event: rep2_approved_submit_phase`, `2026-07-18T01:43:56Z`): *"APPROVED-WITH-NOTES e2aee03 (4 non-blocking).
Note-3 hardened at orchestrator level: single draw-2 attempt binding, logged here as the enforcement record.
Submit executor spawned."* Serial "keep drawing SFT seeds until one replicates" is forbidden; any further draw
requires a fresh user-authorized prereg with multiplicity accounting.

## 3. Job chain — 3 SLURM jobs, afterok-wired (single GPU, no `--time`)

```
13244  lora_sft_curric_rep2   RUNNING (now TRAINING)   Dependency=(null)
   └─ afterok ─▶ 13245  lora_embed   PENDING (JobHeldUser)   Dependency=afterok:13244(unfulfilled)
         └─ afterok ─▶ 13246  enc3seed   PENDING (JobHeldUser)   Dependency=afterok:13245(unfulfilled)
```

Dependency evidence (`scontrol show job`, verbatim fields):

```
JobId=13244 JobName=lora_sft_curric_rep2  JobState=RUNNING   Reason=None       Dependency=(null)
   SubmitLine=sbatch --parsable scripts/slurm/lora_sft_curric_rep2.sbatch

JobId=13245 JobName=lora_embed            JobState=PENDING   Reason=JobHeldUser Dependency=afterok:13244(unfulfilled)
   SubmitLine=sbatch --parsable --dependency=afterok:13244 scripts/slurm/gen_embed_lora.sbatch HateMM logging/lora/HateMM_curric_rep2 Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF

JobId=13246 JobName=enc3seed             JobState=PENDING   Reason=JobHeldUser Dependency=afterok:13245(unfulfilled)
   SubmitLine=sbatch --parsable --dependency=afterok:13245 scripts/slurm/enc3seed_lora_curric_rep2.sbatch
```

The `afterok:13244` / `afterok:13245` wiring is confirmed on both downstream jobs; the initial `JobHeldUser` hold
on 13245/13246 is the normal auto-release-pending state (CLAUDE.md: wait, never force).

## 4. SEED EVIDENCE (the manipulated variable) — `seed: 1`, NOT 42

**This log format does not emit a numeric `seed=` line.** LLaMA-Factory does not dump the HF
`Seq2SeqTrainingArguments` repr in `13244`'s stdout; a grep of the full log (406 KB, all lines) finds **no**
`seed=<n>` token — so no numeric echo exists to quote, and none should be waited for. The seed is instead
established by a hash-anchored provenance chain:

1. **Frozen yaml A pins `seed: 1`.** `sha256sum` of
   `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft_rep2.yaml`
   = `d645de3197739075774b499f335675dad8cd77a3f03b7c6cdc811424506354c6` — **MATCHES** the freeze block
   (`6c11988`). That exact (hash-locked) file contains, verbatim:
   ```
   line 46: # (hatemm_qwen25vl_lora_curric_sft.yaml). Draw-1 pinned NO seed => implicit HF default
   line 47: # seed=42; this explicit seed=1 reseeds transformers.set_seed (parser.py:474), which
   line 50: seed: 1
   ```
   So the frozen config **explicitly overrides** the draw-1 implicit HF default 42 with `seed: 1`.
2. **The job trains THAT frozen yaml with no extra CLI args**, so the seed can only come from the yaml (=1).
   Log `logging/slurm/lora_sft_curric_rep2_13244.out`, verbatim:
   ```
   L1322: [lora_sft_curric_rep2] launching training: python src/train.py /data/jehc223/RGCL/RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft_rep2.yaml
   ```
   (per prereg F-R0.3 / review NOTE-1: `read_args` loads the entire config from the `.yaml`; the sbatch passes
   no dotlist override, so `transformers.set_seed(training_args.seed)` at `parser.py:474` receives `1`.)
3. **NOT seed=42:** the frozen yaml carries the explicit `seed: 1` line; there is no `seed=42` (or any numeric
   `seed=`) anywhere in the log. **No violation.** (Had the log shown `seed=42`, this record would have STOPPED
   and reported a VIOLATION; it does not.)

**MUST-CHECK for the verdict reviewer (numeric runtime confirmation):** because stdout carries no numeric seed,
the reviewer **must** read the effective seed from the SFT artifacts written at completion —
`logging/lora/HateMM_curric_rep2/trainer_state.json` (and/or `all_results.json` / `training_args.bin`) — and
confirm it is **`1`** before rendering the verdict. This is the one seed check deferred out of this submit record.

### Dataset / config / output-dir echoes (verbatim, line-numbered from the `13244` log)

```
L1:    [lora_sft_curric_rep2] DATASET=HateMM CONFIG=/data/jehc223/RGCL/RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft_rep2.yaml OUTDIR=/data/jehc223/RGCL/logging/lora/HateMM_curric_rep2
L1316: [lora_sft_curric_rep2] rebuilding CURRICULUM data for HateMM (idempotent; sha must == 73307ef2...) ...
L1317: [register] added ['hatemm_lora_curric_train'] to /data/jehc223/RGCL/RA-HMD/LLAMA-FACTORY-Ver202512/data/dataset_info.json
L1320: [curric]   train_curric.json sha256 73307ef2e286eddf4fbe12ef13bb3c750f9105d1291494779c7a3a181c91082b
L1451: [INFO|2026-07-18 14:25:06] llamafactory.data.loader:143 >> Loading dataset /data/jehc223/RGCL/data/lora_sft/HateMM/train_curric.json...
```

- **dataset echo:** `hatemm_lora_curric_train` (registered L1317) → resolves to
  `data/lora_sft/HateMM/train_curric.json` (loaded L1451); matches frozen yaml A line 18 `dataset: hatemm_lora_curric_train`.
- **config echo:** the rep2 yaml `hatemm_qwen25vl_lora_curric_sft_rep2.yaml` (L1 CONFIG=, L1322 launch).
- **output-dir echo:** `logging/lora/HateMM_curric_rep2` (L1 OUTDIR=; matches frozen yaml A line 27).
- **STEP-1b re-emit gate (prereg §4b):** L1320 shows `train_curric.json sha256 73307ef2e286…1c91082b`, **bit-exact
  == the frozen draw-1 curriculum** — draw-2 trains the identical multiset; the only difference from draw-1 is the
  SFT seed.

### Healthy-start (live gate satisfied)

```
L1585: [INFO|trainer.py:2405] 2026-07-18 14:38:13,659 >> ***** Running training *****
L1593: ... 5/276 [04:10<3:48:43] {'loss': 0.9155, 'grad_norm': 5.58..., 'learning_rate': 3.57e-05, 'epoch': 0.05}
```

First loss finite and decreasing off a sane start; 276-step (3-epoch) schedule as expected. No NaN/OOM/Traceback.

## 5. Collision statement (rep2 targets created by this run only)

At freeze (`6c11988` §2) and at review (`e2aee03`) all four rep2 targets were verified **ABSENT**. Independent
re-check by this clerk: the only rep2-tagged artifact on disk is the SFT stdout log created by job `13244`.

| target | state before this run | now |
|---|---|---|
| `logging/lora/HateMM_curric_rep2/` (SFT adapter) | ABSENT | created by 13244 (fresh; no clobber of draw-1 `HateMM_curric`) |
| `data/CLIP_Embedding/HateMM/*LoRA-curric-rep2*.pt` | ABSENT | still ABSENT (extraction 13245 not yet run) |
| `logging/Retrieval/HateMM/RAC_video_lora_curric_rep2*` | ABSENT | still ABSENT (head 13246 not yet run) |
| `slurm/logs/enc3s_*curric-rep2*_seed*_*.trainlog` | ABSENT | still ABSENT |
| `logging/slurm/lora_sft_curric_rep2_13244.out` | ABSENT | created by 13244 at 2026-07-18 14:09 |

Distinct `-curric-rep2` tags throughout ⇒ the frozen / generic-LoRA (13235) / draw-1-curric (13241) caches and
adapters are never overwritten; `--force False` on the head cannot trip an overwrite.

## 6. What happens next (not this clerk's action)

Auto-release lifts the `JobHeldUser` holds; on `13244` COMPLETE the afterok chain runs 13245 (extraction) then
13246 (3-seed head reads). The verdict is rendered by an **independent 0-context reviewer** against
`CAND2_REP2_PREREG.md` VERBATIM (§3 decision tree), after (a) confirming the runtime seed == 1 from the SFT
artifacts (§4 MUST-CHECK) and (b) transcribing raw both-protocol per-seed head numbers. No test metric is read
before that verdict.

---

**Clerk statements:** read-only reconstruction from disk (SLURM `scontrol`/`squeue`, `sha256sum`, the `13244`
stdout log, committed prereg/review/freeze). No job submitted/held/released, no push, no test metric, no verdict,
no `state/` mutation by this clerk. The submission itself was performed earlier by the submit executor; this
document is the post-hoc record it did not write.
