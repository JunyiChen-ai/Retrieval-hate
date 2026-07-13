# B1 Pre-Registration Review — `exp-encoder-zh-b1.md`

**Reviewer:** fresh zero-prior-context pre-registration reviewer (read-only; CPU verification only; no GPU/SLURM/commits).
**Date:** 2026-07-14.
**Under review:** `research-wiki/experiments/exp-encoder-zh-b1.md` (B1: frozen Qwen2.5-VL-7B encoder vs frozen CLIP on MHC-ZH, 3 seeds, both protocols, single-dataset train split, archive OFF).
**Verdict:** **APPROVED with 3 mandatory (minor, non-blocking) revisions.** Floors in the draft are **correct** — Task A vindicates them; no floor number needs changing. All revisions are framing/wording tightenings + implementation-check notes.

---

## TASK A — FLOOR DISAMBIGUATION (resolved against primary sources)

### A.1 What produced `0.8537 ± 0.0120`? — **LoRA-Qwen, NOT frozen-CLIP, NOT the frozen encoder swap.**

**Attribution: `0.8537 ± 0.0120` = the ZH _LoRA-only floor_ (archive OFF / 无键), final-epoch _accuracy_, encoder = `Qwen2.5-VL-7B-Instruct-LoRA_HF` (LoRA-SFT of the encoder), 5 seeds. The prep agent's reading (0.8537 = LoRA-Qwen final-epoch) is CORRECT.**

Evidence chain (source → provenance → primary logs, all verified):

1. **`PAPER_MASTER_TABLES.md:43`** (the `:43` area named in the task):
   `| MHC-ZH (149) | LoRA-only floor（无键） | Qwen2.5-VL-7B-LoRA | 0.8282±0.0139 | 0.7962±0.0167 | **0.8537±0.0120** | 0.8259±0.0124 | 5 | exp-archive-knn-seeds Add.2 · ebc1988 |`
   Columns are `val-sel acc | val-sel F1 | final-ep acc | final-ep F1`. So `0.8537` is the **final-epoch acc** of the **LoRA-Qwen** stack's **archive-OFF ("无键" = no keys) floor**. The identical value on `:44` (`+ archive-kNN α0.25`) is a *consequence*, not a second measurement: `:49-52` + `exp-archive-knn-seeds.md:164-180` weight-identity audit show the α=0.25 keys flip **zero** votes at ep29, so the archive arm and its LoRA-only floor are byte-identical at final epoch.
   - **The word "floor" here means "no archive keys," NOT "frozen encoder."** This is the source of the project-memory ambiguity ("ZH floor crosses 0.85"): it is the LoRA-Qwen stack's _own_ archive-OFF floor, not the frozen-CLIP floor and not the frozen-encoder arm.

2. **`exp-archive-knn-seeds.md` Addendum 2, `:164-165`** (primary wiki source):
   `ZH LoRA-only (n=5) | final-epoch acc 0.8537±0.0120 | F1 0.8259±0.0124 | per-seed acc 0.8456/0.8389/0.8523/0.8658/0.8658`.
   Jobs = the coordinator-requested LoRA-only control, `arcbase_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0..4}_1222{3..7}` (`:52-61`), post-W5 HEAD, run by **our campaign** (2026-07-04). seed0 (12223) reproduces job 12149 **bit-for-bit** (`:35-37`) → same-code paired control.

3. **Primary trainlogs re-read (this review):**
   ```
   arcbase_..._LoRA_HF_seed0_12223: Test_Retrieval Epoch 29 ... acc: 0.8456  (F1 0.8181)
   seed1_12224: acc 0.8389 (F1 0.8113) · seed2_12225: acc 0.8523 (F1 0.8226)
   seed3_12226: acc 0.8658 (F1 0.8387) · seed4_12227: acc 0.8658 (F1 0.8387)
   ```
   Mean acc = 0.85368 → **0.8537**; sample std = 0.0120; mean F1 = 0.82588 → **0.8259 (std 0.0124)**. **Confirmed to all printed decimals.**
   Namespace check: `model='Qwen2.5-VL-7B-Instruct-LoRA_HF'`, `dataset='MHC_zh'` — this is the **LoRA-SFT** encoder, decisively distinct from the frozen-Qwen arm B1 tests.

**Verdict A.1: `0.8537` is LoRA-Qwen. The prep agent is right; project memory's "ZH floor" phrasing is the loose element.**

### A.2 The CORRECT B1 control floor (frozen-CLIP ZH, archive-OFF, current code) — **prep agent's numbers VERIFIED.**

Source = `exp-consensus-zh-seeds.md:56-64` floor arm (frozen CLIP ViT-L/14-336, λ_seg=0, whole-video baseline; jobs s0=12130, s1-4=12297-12300).

Per-seed (Test macroF1 / acc), from `:60-62`:
| seed | val-sel F1/acc | final F1/acc |
|---|---|---|
| 0 | 0.7706 / 0.8054 | 0.7706 / 0.8054 |
| 1 | 0.7579 / 0.8054 | 0.7542 / 0.8054 |
| 2 | 0.7742 / 0.8121 | 0.7913 / 0.8322 |

**Seeds 0/1/2 mean (the B1 seeds):** val-sel **0.7676 F1 / 0.8076 acc**; final **0.7720 F1 / 0.8143 acc** (recomputed; matches draft `:193-194`). **The prep agent's claim — seeds 0/1/2 = val-sel 0.8076 acc / final 0.8143 acc — is CORRECT.**

5-seed floor (`exp-consensus-zh-seeds.md:80,83`; `PAPER_MASTER_TABLES.md:46`): val-sel 0.8027±0.0139 acc / 0.7649±0.0151 F1; final 0.8027±0.0215 acc / 0.7594±0.0240 F1. **Verified.**

Primary-log spot-check: `mhc_train_seg_12130.out:284` → `Test_Retrieval Epoch 29 ... acc: 0.8054` (F1 0.7706) = floor seed0 final. **Confirmed.**

**Caveat carried into Task B (see Rev-2):** this floor was produced by `train_consensus_seeds.sbatch` (seg-path, λ_seg=0), NOT `train_archive_baseline`. B1's own same-runner CLIP s0/1/2 re-runs are the authoritative control; 12130 is a cross-runner confirmatory reference.

### A.3 CRITICAL SIDE-QUESTION — was LoRA-Qwen-ZH ever paired-tested vs CLIP under +0.03/+0.03? — **NO. It is NOT an unbanked second-dataset near-pass. It is already-excluded-by-construction (different lever + never a valid paired test).**

The naive numbers _look_ like a pass:
- final-epoch: LoRA-Qwen **0.8537** acc / **0.8259** F1 vs frozen-CLIP floor **0.8027** acc / **0.7594** F1 (both 5-seed) = **+0.051 acc / +0.066 F1** — both clear +0.030.

But this is **not** a banked encoder-swap pass, and the "encoder positive = HateMM only" memory is **correct**, for four independent reasons (each with file:line evidence):

1. **Different lever — LoRA-SFT, not a frozen encoder swap.** The banked positive is the **frozen**-Qwen vs **frozen**-CLIP swap (`exp-encoder-3seed.md:25-34`), holding everything else fixed. `0.8537` fine-tunes the encoder (LoRA). The project explicitly classifies LoRA as a *"MIXED performance lever, not novelty"* (draft `:252` citing `query_pack.md:44`). LoRA-Qwen-vs-CLIP conflates encoder-identity + task-specific fine-tuning + 7B-vs-300M capacity; it does not isolate "MLLM-as-encoder."

2. **The pure FROZEN encoder swap on ZH does NOT reproduce the effect.** The one existing frozen-Qwen ZH data point (seed0, `rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_1151518.trainlog`, verified this review):
   - val-sel (e22): F1 0.7412 / acc 0.7919 → **Δ vs CLIP = −0.0294 F1 / −0.0135 acc (LOSES).**
   - final (e29): F1 0.7864 / acc 0.8188 → **Δ = +0.0158 F1 / +0.0134 acc (weak, < +0.030 bar).**
   Direct evidence the `0.8537` gain is **LoRA-driven, not encoder-driven**. (Corroborated by `exp-archive-knn-seeds.md:258-259`: "LoRA-only ZH is remarkably stable … the instability is introduced by the archive-kNN channel, not the encoder.")

3. **Never a registered same-seed paired test.** LoRA-Qwen (5 seeds, `train_archive_baseline` runner, `RAC_video_archive_seeds`) and the frozen-CLIP floor (5 seeds, `train_consensus_seeds` runner, `RAC_video_consensus_seeds`) are **different sub-experiments, different runners, different encoders (3584-dim LoRA vs 1024/768 CLIP), no same-seed pairing, no +0.03/+0.03 decision rule ever applied.** The +0.03/+0.03 rule (`exp-encoder-3seed.md:73-85`) was defined for the frozen swap only.

4. **The project's own master table forbids the comparison.** `PAPER_MASTER_TABLES.md:58`: *"consensus 是 CLIP-base 独立子实验（不同编码器，与 archive-kNN 的 LoRA-Qwen 主栈**不可直接同格并比**）"* — the LoRA-Qwen main stack and the frozen-CLIP floor are declared **not directly comparable side-by-side**. So `0.8537` vs `0.8027` was never a valid paired cell.

**Verdict A.3:** LoRA-Qwen-ZH is **not** a hidden already-banked second-dataset encoder pass. It is a real +5 acc / +6.6 F1 multi-seed gap, but it is a **fine-tuning** result on a *"not-novelty"* lever, cross-runner/cross-experiment, never same-seed paired, and explicitly barred from same-cell comparison by the project's own accounting. The `frozen`-encoder swap — which is exactly what B1 isolates — does **not** show the effect on ZH's one existing seed. The "encoder positive = HateMM only" memory stands.

---

## TASK B — B1 DRAFT REVIEW (standard 0-context pre-registration review)

### Checklist (verified items)

| Item | Result |
|---|---|
| Decision rule transcribed from `exp-encoder-3seed.md:73-85` | ✅ **Verbatim** (draft `:158-170` = parent `:73-85`, word-for-word). |
| Both protocols defined, judged independently | ✅ Draft `:140-147` = parent `:66-71` verbatim (val-sel A / final-epoch B). |
| Final-epoch declared PRIMARY | ⚠️ New element vs parent (parent has no "primary") — see **Rev-3**. |
| Test-touch budget: parent test did **not** consume ZH test | ✅ **VERIFIED.** Parent datasets = HateMM + MHC(EN) only (`exp-encoder-3seed.md:38`); all `enc3s_*` logs read `dataset='MHC'` or `'HateMM'`; **no `enc3s_*zh*` log exists**. ZH-encoder-swap test set is virgin for this question. Budget = 1 touch (draft `:296-298`). Correct. |
| Floors correct (Task A) | ✅ Draft uses frozen-CLIP floor (≈0.808 acc s0-2 / 0.8027 5-seed), **correctly refuses** to use `0.8537` as the frozen floor (`:254-256`). Task A vindicates this. |
| Honest-prior consistent w/ 1151518 | ✅ **VERIFIED.** val-sel e22 = 0.7412/0.7919, final e29 = 0.7864/0.8188 (re-read from log). Val-selection under warmup≥5: max val-acc 0.8205 at {22,26,28}, roc tie-break → e22 (roc 0.8693). Draft `:204-222` matches exactly; single-seed paired deltas (−0.0135/−0.0294 val-sel; +0.0134/+0.0158 final) arithmetically correct. |
| OLD-code single-seed caveat stated | ✅ Draft `:202,214,39` flag old-code, 1-seed, 3584-dim (log head verified: Image/Text dim 3584, `group='RAC_video'`). Comparability handled via reproduction gate `:122-135`. |
| Single-submit ceremony | ✅ Draft `:312-322`: one sbatch, no mid-run resubmit, reproduction gate first, read-back from raw logs. Mirrors parent. |
| Config-edit scope (6-row CONFIGS) vs `enc3seed.sbatch` | ✅ Runner uses a `CONFIGS=("DATASET MODEL SEED")` array (10 rows) → python `--dataset/--model/--seed`, `--lambda_seg 0`, `GROUP=RAC_video_archive_seeds`, frozen `QWEN=Qwen2.5-VL-7B-Instruct_HF` (NOT LoRA). A 6-row ZH block (MHC_zh × {CLIP,Qwen} × {0,1,2}) is the exact, minimal, correct edit. RUNLOG auto-derives `enc3s_MHC_zh_{model}_seed{s}_{jid}.trainlog` as the draft expects (`:320`). |
| Cached feature assets present | ✅ All 6 ZH `.pt` verified: Qwen {train 16.6M/2026-07-02, dev 2.24M, test 4.28M}, CLIP {train 4.17M/2026-07-01, dev 0.56M, test 1.07M}. Loader: `dataset.py:499` lists `MHC_zh`; builds `{path}/{dataset}/{split}_{model}.pt`. |
| ZH splits / expected N | ✅ `data/gt/MHC_zh/{train,val,test}.jsonl` = **579 / 78 / 149** lines; `EXPECTED_TRAIN_N["MHC_zh"]=579` (`lb_scgp_global_r2_m1_cache_v1_common.py:46`). Matches draft `:95-96`. |
| `FORCE=False` output-dir collision | ✅ No collision: existing `RAC_video_archive_seeds/MHC_zh/` dirs are all `-LoRA_HF`; frozen arms create fresh `_Qwen2.5-VL-7B-Instruct_HF` / `_openai_clip-...-336_HF` dirs. |

### Mandatory revisions (minor; none change a floor or the decision rule)

**Rev-1 — Mechanistic rationale #2 mis-uses the LoRA-Qwen `0.8537` as a reason to expect the FROZEN swap to pass (`:74-77`).**
The bullet cites *"the LoRA-Qwen ZH stack reaches final-epoch acc 0.8537 … so ZH is the dataset closest to a clean second-dataset pass"* inside the "why H1 might hold" list. Per Task A.3, `0.8537` is a **different (fine-tuning) lever**; the frozen swap's own ZH seed0 is +0.013 acc (< bar) / −0.014 acc (val-sel). Using LoRA headroom as a frozen-swap prior is the exact conflation the honest-prior section (`:249-256`) warns against.
**Fix:** reframe #2 to cite only the encoder-independent control-quality fact (frozen-CLIP ZH floor already beats MoRE M-F1 0.7706 vs 0.7475). State explicitly that the LoRA/frozen gap is, if anything, evidence the ZH gain is **LoRA-driven not encoder-driven**, i.e. a headwind for H1, not a tailwind. (The draft already says this at `:249-256`; make `:74-77` consistent with it.)

**Rev-2 — Reproduction-gate / kill-rule-1 over-hardens a cross-runner CLIP reference (`:129-135`, `:270-274`).**
The frozen-CLIP s0 reproduction target (job 12130) was produced by a **different runner** (`train_consensus_seeds`, seg-path λ=0) than B1's `train_archive_baseline` (archive-OFF path). This is the **first time** CLIP-ZH runs through `train_archive_baseline` (CLIP-ZH previously existed only via the consensus/seg runner), so the two paths' bit-for-bit equivalence is asserted-by-transitivity, not previously exercised on this cell. Kill rule 1's parenthetical (*"a numeric mismatch is not [a naming difference]"* → auto-HALT) could nuke a clean experiment on a benign runner-path discrepancy.
**Fix:** split the gate. (a) **Hard gate = same-runner reproducibility:** the frozen-Qwen s0 must reproduce 1151518 to 4 decimals (this is the true old-vs-new-code confound, analogous to the parent). (b) **CLIP s0 vs 12130 = confirmatory cross-runner check:** on a >0.0001 mismatch, run a code-path audit that is *permitted to conclude* "benign archive-OFF-vs-seg-λ0 runner difference; same-runner s0 is the authoritative control," rather than mandating an automatic campaign HALT. Keep the same-runner CLIP s0/1/2 as the authoritative control band (draft already designates this at `:133-134` — just make kill rule 1 consistent).

**Rev-3 — "Primary protocol = final-epoch" needs a one-line guard against reading it as a decision gate (`:149-156`).**
The parent test has **no** primary protocol (it judges both independently). Declaring one before running is acceptable pre-registration, but the rationale at `:153` (*"the protocol under which ZH is closest to the goal bar"*) again leans partly on LoRA-Qwen headroom (Rev-1). 
**Fix:** (a) base the primary designation solely on the **encoder-independent** val-selection-tax argument (78-sample dev costs ~2 acc pts — well-documented, `PAPER_MASTER_TABLES.md:56-57`); (b) add one sentence that "primary" is a **reporting-emphasis only**, NOT a decision gate — the ">= 2 datasets" headline still requires ZH to pass under the **same** protocol HateMM passed (HateMM passes both), and the fixed write-up format `"final-epoch: pass/fail; val-selected: pass/fail"` (kill rule 3, transcribed from parent) governs. The draft mostly says this at `:174-177`; make it explicit at the primary declaration.

### Implementation-check notes (for the impl checker, not blockers)

- **IN-1.** Confirm the ZH `CONFIGS` block keeps `QWEN=Qwen2.5-VL-7B-Instruct_HF` (**frozen**), never the `-LoRA_HF` cache — the whole point of B1 is to isolate the frozen encoder from the LoRA lever that produced 0.8537.
- **IN-2.** `FORCE=False` is safe (verified: no frozen-model `RAC_video_archive_seeds/MHC_zh` dirs exist), but re-check at submit time in case a smoke run created a partial dir.
- **IN-3.** Optional 1-epoch smoke (draft `:317`) should assert Qwen 3584→1024 and CLIP (1024 img + 768 txt)→1024 wiring into `classifier_hateClipper`, as the draft states.

---

## FINAL VERDICT

**APPROVED, pending Rev-1/Rev-2/Rev-3 (all minor framing/wording; no floor or decision-rule change).**

The draft's core is sound: floors are **correct** (Task A confirms the draft rightly refuses to treat `0.8537` as the frozen floor), the decision rule and both protocols are transcribed verbatim from the parent, the ZH test set is confirmed **virgin** for this question, all assets/splits/loader facts check out, and the honest-prior is faithful to the one real frozen-Qwen ZH seed. The three revisions purge residual LoRA-vs-frozen conflation from the "why-it-might-pass" framing and harden the cross-runner reproduction gate. After those edits, B1 is ready for implementation check + authorization.

---

# DELTA-CHECK (Rev-1 package) — 2026-07-14, same reviewer

**Package reviewed:** `research-wiki/experiments/exp-encoder-zh-b1.md` (status
`DRAFT-REV1-AWAITING-DELTA-CHECK`), `scripts/slurm/enc3seed_zh_b1.sbatch` (new),
`refine-logs/B1_IMPL_NOTES.md`. Parent `scripts/slurm/enc3seed.sbatch` untouched
(verified by diff below).

## D.1 Revision landing check

| Revision | Landed as specified? | Evidence |
|---|---|---|
| **Rev-1** (0.8537 out of tailwinds, reframed as HEADWIND) | ✅ | Prereg `:77-85`: rationale #2 now cites only the encoder-independent above-SOTA control fact (frozen-CLIP M-F1 0.7706 vs MoRE 0.7475); 0.8537 explicitly marked "Deliberately NOT cited as a reason to expect a pass" and the LoRA-vs-frozen gap (0.8537 vs 0.8188) explicitly called "**a HEADWIND for H1, not a tailwind**". Consistent with honest-prior `:268-275` (unchanged, already correct). The remaining 0.8537 mentions (`:51`, `:94-95`, `:269-273`, `:373`) are all context/anti-conflation usages with correct LoRA attribution — none is a tailwind claim. |
| **Rev-2** (kill-rule split 1a/1b, both flagged sites) | ✅ | Kill rule 1 (`:289-301`): **1a** = hard same-lineage gate, frozen-Qwen s0 must reproduce 1151518 to 4 decimals under **both protocols** → HALT + code-path audit; **1b** = frozen-CLIP s0 vs 12130 cross-runner confirmatory, >0.0001 mismatch → audit **permitted to conclude benign runner difference**, no auto-HALT, same-runner CLIP band authoritative. Config-match section (`:137-146`) updated to match — both flagged sites covered, mutually consistent. |
| **Rev-3** (primary = reporting-emphasis only) | ✅ | `:160-173`: "REPORTING-EMPHASIS ONLY, NOT a decision gate"; sole rationale = encoder-independent 78-dev val-selection tax ("no headroom argument enters it"); both protocols judged independently under verbatim rule (4); fixed write-up format governs; ≥2-datasets headline requires same-protocol pairing ("only within that protocol's column" — correct given HateMM passed both). |

No floor, decision-rule, seed, or budget change anywhere in the r1 diff. Revision
history table (`:377-382`) accurately describes the changes.

## D.2 Sbatch diff — independently verified CONFIGS-only

`diff -u scripts/slurm/enc3seed.sbatch scripts/slurm/enc3seed_zh_b1.sbatch` re-run by
this reviewer: **exactly one hunk** (`@@ -29,16 +29,12 @@`), replacing the 10-row
EN/HateMM `CONFIGS` array with the 6-row ZH array (MHC_zh × {CLIP, QWEN} × {0,1,2}),
matching the prereg run matrix row-for-row. Everything else byte-identical (SBATCH
headers, env, `CLIP`/`QWEN`/`GROUP_NAME`/`WARMUP`, `run_one()` python command, parser,
loop, b2 push). `bash -n`: SYNTAX_OK. Matches the diff shown in `B1_IMPL_NOTES.md` §b.

**Ruling on the 2 cosmetic remnants — IMMATERIAL, LEAVE AS-IS (do NOT edit):**
1. Header comment "Runs 10 configs serially" (actual: 6) — comment-only, zero runtime
   effect. Leave as-is.
2. `#SBATCH --job-name=enc3seed` — stdout goes to `enc3seed_<newJID>.out`; only existing
   file is `enc3seed_12850.out`, `%j` guarantees no overwrite; squeue readability is the
   only cost. Leave as-is.

Rationale for "leave as-is" over "fix": both are functionally inert, and any post-hash
edit would invalidate the authorization hashes below. The execution record should note
remnant #1 so a future reader of the .out is not confused by "10 configs".

## D.3 Impl-notes spot-check — cache-verification table VERIFIED

Re-verified independently by CPU `torch.load` (HateVideo env, no GPU), all 6 caches:
- Dims/rows exactly as tabled: CLIP train/dev/test = (579/78/149, 1024 img / 768 text);
  Qwen = (579/78/149, 3584/3584); labels shapes match; row counts = ZH splits 579/78/149.
- **Paired-arm id audit reproduced: CLIP-vs-Qwen id lists identical (set AND order) on
  all three splits.**
- File sizes/mtimes match both the impl-notes table and this review's original check.

## D.4 Runtime re-checks (at review time)

- **IN-1:** runner `:25-26` — `QWEN=Qwen2.5-VL-7B-Instruct_HF` (frozen); no `LoRA`
  string anywhere in the sbatch. ✅
- **IN-2:** `logging/Retrieval/MHC_zh/RAC_video_archive_seeds/` contains **only**
  `*-LoRA_HF*` dirs (9 dirs) → no `FORCE=False` collision for the frozen arms. ✅
  (Must be re-checked at submit time — condition (ii) below.)
- No `slurm/logs/enc3s_MHC_zh*` file exists → no log collision. ✅

## D.5 DELTA VERDICT: **APPROVED**

All three revisions landed as specified; the sbatch is a CONFIGS-only delta of the
audited parent; the impl-notes are accurate on every spot-checked claim; both cosmetic
remnants are ruled immaterial.

---

# CONDITIONAL EXECUTION AUTHORIZATION (B1)

**Granted 2026-07-14 by the B1 pre-registration reviewer, scope: exactly ONE
submission of:**

```
sbatch scripts/slurm/enc3seed_zh_b1.sbatch
```

by the implementer agent, under ALL of the following conditions:

**(i) Hash pinning.** The authorized artifacts are exactly:

```
91982eb333e61efc34e62794031f6b3f8b672e34ffee7d558fa03e1b2b57972b  research-wiki/experiments/exp-encoder-zh-b1.md
9504dba00ad1ae8351bedbe1ebcd1b5bf1382374273c27df7db7d521f0cbd762  scripts/slurm/enc3seed_zh_b1.sbatch
```

At submit time the implementer MUST re-run `sha256sum` on both files and verify both
digests match the above. Any mismatch = authorization VOID; return for re-review. (This
includes the cosmetic remnants: they are authorized as-is; "fixing" them voids the hash.)

**(ii) `FORCE=False` no-collision re-check at submit time.**
`logging/Retrieval/MHC_zh/RAC_video_archive_seeds/` must contain **only** `*-LoRA_HF*`
directories (i.e., zero dirs matching the frozen tags
`*_openai_clip-vit-large-patch14-336_HF` or `*_Qwen2.5-VL-7B-Instruct_HF` without
`-LoRA`). If any frozen-tag dir exists (e.g., from an interim smoke run), STOP and
report — do not submit.

**(iii) Single-submit discipline.** One `sbatch` invocation, no `--time` flag added,
initial `PENDING (JobHeldUser)` = WAIT for auto-release (never force-release), no
resubmission after the job reaches any terminal state (COMPLETED/FAILED/CANCELLED),
no mid-run intervention.

**Executor obligations:** write `refine-logs/B1_EXECUTION_RECORD.md` containing the
job id, submit/start/end timestamps, and the submit-time re-hash outputs (i); on
terminal state, verify the 6 expected trainlogs exist
(`slurm/logs/enc3s_MHC_zh_{openai_clip-vit-large-patch14-336_HF,Qwen2.5-VL-7B-Instruct_HF}_seed{0,1,2}_<JID>.trainlog`)
with sane readouts (30 epochs, VALSEL/FINAL RESULT_ROW lines parse), then report the
raw per-seed numbers **WITHOUT interpreting them**. Verdict processing = orchestrator +
independent verdict review, per project rule. The reproduction gates (kill rules 1a/1b)
are applied at verdict processing, not by the executor.

**Out of scope of this authorization:** any second submission, any config/seed/dataset
change, any edit to either hashed file, any GPU work beyond this one sbatch job.
