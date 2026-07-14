# B3 Pre-Registration Review — `exp-lora-zh-b3.md`

**Reviewer:** fresh zero-prior-context pre-registration reviewer (read-only; CPU verification only; no GPU / no SLURM / no commits).
**Date:** 2026-07-14.
**Under review:** `research-wiki/experiments/exp-lora-zh-b3.md` (status `DRAFT-UNREVIEWED`) + `scripts/slurm/enc3seed_zh_b3.sbatch` + `refine-logs/B3_IMPL_NOTES.md`.
**Background (read):** `refine-logs/B3_FORENSIC_RECON.md` (authoritative recon), `refine-logs/B1_PREREG_REVIEW.md`, `research-wiki/experiments/exp-encoder-zh-b1.md`.
**Verdict:** **APPROVED** with one non-blocking numeric-consistency erratum (F-1; does NOT void the authorization hashes, applied at verdict-processing time per B1 precedent). Conditional execution authorization granted below.

---

## 0. Checklist result summary

| # | Checklist item | Result |
|---|---|---|
| 1 | Recon-preview numbers verified from primary logs (final-ep + val-sel; 13115 CLIP both protocols; val-sel preview +0.0246 acc / +0.0339 F1 = FAIL shape) | ✅ **All exact** except one 0.0001 contextual blemish (F-1, §1.3) |
| 2 | Rule fidelity (verbatim +0.03/+0.03 AND, 3/3 sign, both protocols independent, categories pre-declared) + marginal-pass reporting ruling | ✅ Rule verbatim; marginal-pass ruling in §2.2 |
| 3 | Honesty clauses present (novelty PENDING; family-claim; single-draw; prior ZH exposure; PMT:58 = user call) | ✅ All present |
| 4 | GROUP decision (RAC_video_b3_lora, FORCE=False) vs run_rac.py:899-908; G-repro anchors preserved; sbatch diff scope | ✅ Verified correct |
| 5 | G-repro hard-gate numbers (0.8456/0.8389/0.8523 acc, 0.8181/0.8113/0.8226 F1) | ✅ Exact vs primary logs |
| 6 | Re-hash 2 files vs impl-notes (71745cf2 / 43792246) | ✅ Both match |

---

## 1. TASK 1 — Numeric verification against primary logs

Method: I re-parsed each primary trainlog with the **exact** parser embedded in the sbatch
(`Val_/Test_Retrieval` regex; val-selection = epoch ≥ warmup 5 with max `Val_Retrieval acc`,
roc tie-break; final = max epoch). Logs read:
`arcbase_MHC_zh_..._LoRA_HF_seed{0,1,2}_1222{3,4,5}.trainlog` (LoRA arm) and
`enc3s_MHC_zh_openai_clip-...-336_HF_seed{0,1,2}_13115.trainlog` (CLIP control) +
`enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_13115.trainlog` (frozen-Qwen, decomposition).

### 1.1 Final-epoch (protocol B) — draft §7a — **VERIFIED EXACT**

| seed | LoRA acc | CLIP acc | Δacc | LoRA F1 | CLIP F1 | ΔF1 |
|---|---|---|---|---|---|---|
| 0 | 0.8456 | 0.8054 | +0.0402 | 0.8181 | 0.7706 | +0.0475 |
| 1 | 0.8389 | 0.8054 | +0.0335 | 0.8113 | 0.7542 | +0.0571 |
| 2 | 0.8523 | 0.8322 | +0.0201 | 0.8226 | 0.7913 | +0.0313 |
| **mean** | **0.8456** | **0.8143** | **+0.0313** | **0.8173** | **0.7720** | **+0.0453** |

Acc mean **+0.0313** (≥ bar, marginal), 3/3 sign; F1 mean **+0.0453** (clears), 3/3 sign.
Every cell in draft §7a reproduced to all printed decimals. Mean-of-per-seed-deltas
(0.0938/3 = 0.03127→+0.0313; 0.1359/3→+0.0453) equals the diff-of-means — internally consistent.

### 1.2 Val-selected (protocol A) — draft §7b — **VERIFIED EXACT (FAIL shape confirmed)**

LoRA val-sel epochs/values: s0 (ep20) 0.8322/0.8023, s1 (ep26) 0.8255/0.7956, s2 (ep19) 0.8389/0.8065 (acc/F1).
CLIP val-sel: s0 (ep29) 0.8054/0.7706, s1 (ep28) 0.8054/0.7579, s2 (ep25) 0.8121/0.7742. All match the draft.

| seed | Δacc | ΔF1 |
|---|---|---|
| 0 | +0.0268 | +0.0317 |
| 1 | +0.0201 | +0.0377 |
| 2 | +0.0268 | +0.0323 |
| **mean** | **+0.0246** | **+0.0339** |

Acc mean **+0.0246 < +0.030 → FAILS the acc bar** (3/3 sign); F1 mean **+0.0339** clears.
Both-metric AND rule ⇒ **val-selected PREVIEW = FAIL**. The prep agent's val-sel preview
(**+0.0246 acc / +0.0339 F1 = val-sel FAIL shape**) is **confirmed exact** from primary logs.

### 1.3 G-repro anchors (draft §10 / impl-notes §b) — **VERIFIED EXACT**

seed0/12223: acc 0.8456 / F1 0.8181 · seed1/12224: 0.8389 / 0.8113 · seed2/12225: 0.8523 / 0.8226.
All three anchors reproduced to 4dp from the arcbase ep29 `Test_Retrieval` lines.

### 1.4 Decomposition (draft §7d) — one 0.0001 blemish → **F-1 (non-blocking)**

frozen-CLIP mean 0.8143 ✅; frozen-Qwen mean **0.8031** ✅ (per-seed 0.8188/0.8054/0.7852, verified);
LoRA 0.8456 ✅. **BUT** draft §7d states the frozen-Qwen Δ-vs-CLIP as **−0.0113**; the primary-log
truth is **−0.0112** (diff-of-means −0.011200; mean-of-per-seed-diffs −0.011200). The draft's own
table is internally off by 0.0001 (0.8031 − 0.8143 = −0.0112, not −0.0113). The **authoritative
recon shares the same rounding** (`B3_FORENSIC_RECON.md:184`), and the **canonical banked B1
verdict is already correct** at −0.0112 (`exp-encoder-zh-b1.md:19`). This value is **contextual**
(it restates the B1 20th-negative decomposition; it is NOT a B3 decision gate, floor, or the
pass/fail number). ⇒ Flagged, non-blocking (see F-1, §7).

---

## 2. TASK 2 — Rule fidelity + marginal-pass ruling

### 2.1 Rule fidelity — VERIFIED

- **Decision rule (§6)** is transcribed **verbatim** from `exp-encoder-3seed.md:73-85` — identical
  to the wording the B1 reviewer verified verbatim (`B1_PREREG_REVIEW.md:85`). Pass = mean Δacc ≥
  +0.030 **AND** mean ΔmF1 ≥ +0.030 **AND** 3/3 sign, **per dataset × protocol**, **both protocols
  judged independently** (§5, §6 rule 4, kill rules 3-4). Outcome categories pre-declared and the
  fixed write-up format "final-epoch: pass/fail; val-selected: pass/fail" is stated (§5, §7c, §10.3).
- **Marginal-pass concern correctly surfaced by the draft itself:** §7a/§7c flag that acc mean
  +0.0313 sits only +0.0013 above the +0.030 bar and call it a fragility. Good — but the framing
  needs the tightening in §2.2 (the "wobble" language is in tension with the deterministic G-repro
  expectation).

### 2.2 RULING on the marginal-pass reporting language (MANDATORY at verdict processing)

The final-epoch acc pass is **marginal and must be reported as such**. The independent verdict
review MUST use this language and MUST NOT upgrade it:

1. **Fixed format, no upgrade.** Report exactly `final-epoch: PASS (MARGINAL); val-selected: FAIL`.
   A marginal acc pass on one protocol, with the other protocol FAILing on acc, does not become a
   clean/headline pass.
2. **Mandatory sensitivity note (three facts, all required):**
   (a) mean Δacc **+0.0313 is only +0.0013 above the +0.030 bar** (≈4% of the bar);
   (b) it is **carried unevenly** — per-seed Δacc spans **+0.0201 … +0.0402**, and **seed2
   (+0.0201) is itself below the per-seed bar**; the pass rests on seeds 0/1 and on F1
   (+0.0453 clears cleanly), not on a uniform margin;
   (c) the acc margin is **smaller than the between-seed spread** — i.e. within head-seed noise.
3. **Recast the "wobble" framing (draft §7a/§7c is slightly self-inconsistent).** Because the
   G-repro hard gate expects **bit-exact** reproduction of the deterministic arcbase anchors
   (cached features + inert-only argv deltas), the fresh LoRA numbers are **not expected to move**;
   there is no run-to-run stochastic wobble on *this* run. The marginality is therefore
   **structural**, not stochastic: it comes from (i) proximity to the bar, (ii) the **single fixed
   CLIP control arm** (13115, one draw), and (iii) the **single LoRA encoder draw** with head-only
   seed variance (§0 fact 2). The verdict must say the fresh run *re-confirms* the same +0.0313
   under clean same-code pairing — not that it proves robustness.
4. **No headline upgrade.** Even a confirmed marginal final-epoch pass yields, with HateMM, only
   (a) a **FAMILY** claim (frozen-on-HateMM + LoRA-on-ZH = different levers) and (b) a
   result **pending the user's novelty ruling** (§8). The +0.0013 margin, single-draw, and
   head-only-variance caveats travel with any claim built on it.

---

## 3. TASK 3 — Honesty clauses — ALL PRESENT

| Clause | Location | OK |
|---|---|---|
| Novelty = PENDING USER RULING (not decided by B3) | front-matter `novelty_clause`, §0, §8, rev-history | ✅ |
| Family-claim framing (frozen-HateMM + LoRA-ZH = different mechanisms) | §0 fact 1, §6 "Application", §8.2 | ✅ |
| Single-encoder-draw limitation (head-seed variance only, not SFT-seed variance) | §0 fact 2, §3, §7d, §10 rule 1, §11 | ✅ |
| Prior ZH test exposure under old code (12223-27 read ZH test) | §11 (prior exposure declared; B3 = re-measurement under current code) | ✅ |
| `PAPER_MASTER_TABLES:58` "不可直接同格并比" override = user call | §8.3 | ✅ |
| H3 conflates encoder-identity + LoRA-SFT + capacity (not isolated) | §3 | ✅ |

The novelty status is faithful to `B1_PREREG_REVIEW.md:64` and `query_pack.md:44` (LoRA = "MIXED
performance lever, not novelty"). The opposite-lever-profile asymmetry (HateMM needs frozen, ZH
needs LoRA) is declared up front (§0 fact 1) and bounds the family claim honestly.

---

## 4. TASK 4 — GROUP decision, collision analysis, sbatch diff

### 4.1 Collision semantics vs `src/run_rac.py:899-908` — VERIFIED

Read directly: `:855 group_name = args.group_name` is a **dead local** (grep of `group_name` in
`run_rac.py` → only `:283` argparse, `:851` comment, `:855` dead local, `:900` live use);
`:899-900` builds `output_path = .../Retrieval/MHC_zh/<args.group_name>/<exp_name>/`; `:901-903`
makedirs if absent; `:904-908` `if not args.force: raise Exception("Output path already exists,
aborting...")` (HARD ABORT). `force=True` would fall through and overwrite in place. **`exp_name`
(`:856`) is seed+model-derived and does NOT include `group_name`**, so a LoRA seed-s dir name is
byte-identical to the existing arcbase dir → under the arcbase group with `force=False` it would
hard-abort; `force=True` would overwrite the anchors. The draft/impl-notes collision analysis is
**correct in every step**.

### 4.2 Filesystem checks — VERIFIED

- `logging/Retrieval/MHC_zh/RAC_video_b3_lora*` → **does not exist** ⇒ fresh group creates fresh
  dirs, `force=False` never trips `:908`, **nothing overwritten**.
- **G-repro anchor dirs preserved:** arcbase LoRA dirs `..._seed{0,1,2,3,4}_hybrid_loss_
  Qwen2.5-VL-7B-Instruct-LoRA_HF` all present under `RAC_video_archive_seeds/MHC_zh/` — the fresh
  group leaves them untouched. ✅
- `slurm/logs/enc3s_MHC_zh_*LoRA_HF*` → **does not exist** ⇒ no trainlog collision.

**Ruling: `GROUP=RAC_video_b3_lora, FORCE=False` is the correct, non-destructive choice.** It
preserves the anchors the G-repro gate reproduces against, keeps `force=False` (matching both 13115
and the arcbase anchors, so the only Namespace deltas vs 13115 are the inert `{model, exp_comment,
group_name, output_path}`), and the group is computationally inert (feeds only `output_path`). I
verified the Namespace substantive fields of arcbase-12223 and 13115-CLIP are identical (fusion=align,
topk=20, metric=cos, loss=triplet, hybrid=True, proj=map=1024, epochs=30, batch=64, lr=1e-4,
warmup=5, lambda_seg=0.0, archive_feats=None, force=False) — so vs arcbase the ONLY delta is
`{group_name, output_path}` (model + exp_comment already match), and the G-repro exact-reproduction
expectation is sound.

### 4.3 sbatch diff scope — VERIFIED, ACCEPTABLE

`diff -u enc3seed_zh_b1.sbatch enc3seed_zh_b3.sbatch` = the described **3 logical hunks**
(unified-diff renders them as 2 adjacent hunks): (i) header comment (INERT — corrects the stale
"frozen-CLIP vs frozen-Qwen / 10 configs" text to the B3 description); (ii) variable block
(`QWEN=…→LORA=…-LoRA_HF`, `GROUP_NAME=RAC_video_archive_seeds→RAC_video_b3_lora`, `CLIP` kept as an
unused breadcrumb); (iii) CONFIGS (6 rows → 3 LoRA rows). **Everything else is byte-identical** —
SBATCH headers, env, `WARMUP=5`, the full `run_one()` python command, the VALSEL/FINAL parser, the
loop, the b2 push. `bash -n` = SYNTAX_OK. **No dangling `$QWEN`** (grep confirms `$QWEN` fully
removed; the CLIP breadcrumb is defined-unused, zero runtime effect).

**Ruling on the QWEN→LORA rename + header-comment edit: ACCEPTABLE.** The rename is clean (nothing
references the old `$QWEN`); leaving B1's "frozen-CLIP vs frozen-Qwen" header on a LoRA test would be
substantively *wrong*, so correcting it (comment-only, inert) is the safer choice — consistent with
the B1 precedent that tolerates inert comment remnants while preferring accuracy. The immaterial
remnants (`--job-name=enc3seed` → `%j` guarantees no `.out` overwrite; `CLIP` unused breadcrumb) are
correctly flagged and ruled leave-as-is (editing them would void the hash for zero benefit).

---

## 5. TASK 5 — G-repro hard-gate numbers — VERIFIED (see §1.3)

0.8456/0.8389/0.8523 acc and 0.8181/0.8113/0.8226 F1 all reproduced exactly from
`arcbase_..._LoRA_HF_seed{0,1,2}_1222{3,4,5}.trainlog` ep29 `Test_Retrieval`. The gate is a genuine
hard gate: cached features + inert-only argv deltas ⇒ bit-exact reproduction is the correct
expectation, and the two corroborating stability gates cited (12223=12149 bit-for-bit;
frozen-Qwen s0 in 13115 = old 1151518 exactly) make a mismatch a real regression, not noise.

---

## 6. TASK 6 — Re-hash — VERIFIED

```
71745cf29de7f03a2bd4d351b30b02637a8d250f493dfb7f49d3459c44f7d802  research-wiki/experiments/exp-lora-zh-b3.md
4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad  scripts/slurm/enc3seed_zh_b3.sbatch
```

Both match `B3_IMPL_NOTES.md` §f exactly.

---

## 7. Findings

- **F-1 (non-blocking numeric-consistency erratum).** Draft §7d states the frozen-Qwen decomposition
  Δ-vs-CLIP as **−0.0113**; the primary-log value is **−0.0112** (`exp-encoder-zh-b1.md:19` already
  banks it correctly as −0.0112; the recon `:184` shares the −0.0113 rounding). This is a 0.0001
  blemish in a **contextual, non-gate** cell (it restates the B1 20th-negative decomposition, not any
  B3 pass/fail number). **Ruling — does NOT block authorization and MUST NOT be fixed pre-submit**
  (editing would void the pinned hashes for zero decision impact, per the B1 immaterial-remnant
  precedent). Instead: (i) the orchestrator MUST correct §7d `−0.0113 → −0.0112` at verdict-
  processing time, when the file is next edited to fold in the fresh results; (ii) if the
  frozen-Qwen−CLIP decomposition ever migrates to a paper table, the correct **−0.0112** must be used.
- **F-2 (framing, folded into the §2.2 ruling).** The draft's "a small downward wobble on the fresh
  run would tip it" (§7a/§7c) is in mild tension with its own deterministic G-repro expectation. The
  verdict must recast marginality as **structural** (proximity-to-bar + single CLIP control + single
  LoRA draw), not run-to-run stochastic. No file edit required pre-submit; this governs the verdict
  write-up.

No other issues. Floors, control arm, decision rule, both protocols, honesty clauses, collision
analysis, anchors, and hashes are all correct.

---

## 8. FINAL VERDICT — **APPROVED**

The B3 pre-registration is sound. Every recon-preview number is verified exact against the primary
logs (final-epoch +0.0313 acc / +0.0453 F1, 3/3 sign, marginal on acc; val-selected +0.0246 acc /
+0.0339 F1 = FAIL shape); the G-repro anchors and hashes match; the decision rule is transcribed
verbatim and both protocols are judged independently; all honesty clauses (novelty PENDING, family
framing, single-draw, prior ZH exposure, PMT:58 = user call) are present; and the
`GROUP=RAC_video_b3_lora / FORCE=False` decision is verified non-destructive against
`run_rac.py:899-908` with the anchors preserved. The single 0.0001 blemish (F-1) is contextual and
non-blocking. The marginal-pass reporting ruling (§2.2) is **binding on the verdict processing**.

No pre-submit revisions are required. F-1 and the §2.2 marginal-pass language are handled at
verdict-processing time (not before submission), preserving the authorization hashes.

---

## 9. CONDITIONAL EXECUTION AUTHORIZATION (B3)

**Granted 2026-07-14 by the B3 pre-registration reviewer.** Scope: **exactly ONE** submission of:

```
sbatch scripts/slurm/enc3seed_zh_b3.sbatch
```

by the executor agent, under ALL of the following conditions:

**(i) Hash pinning.** The authorized artifacts are exactly:
```
71745cf29de7f03a2bd4d351b30b02637a8d250f493dfb7f49d3459c44f7d802  research-wiki/experiments/exp-lora-zh-b3.md
4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad  scripts/slurm/enc3seed_zh_b3.sbatch
```
At submit time the executor MUST re-run `sha256sum` on both files and verify both digests match.
Any mismatch = authorization VOID; return for re-review. (This includes the immaterial remnants and
the F-1 blemish: they are authorized **as-is**; "fixing" either voids the hash.)

**(ii) `FORCE=False` no-collision re-check at submit time.** Immediately before submitting, verify:
`logging/Retrieval/MHC_zh/RAC_video_b3_lora*` **does not exist** AND
`slurm/logs/enc3s_MHC_zh_*LoRA_HF*` **does not exist** AND the arcbase anchor dirs
(`RAC_video_archive_seeds/MHC_zh/..._seed{0,1,2}_hybrid_loss_Qwen2.5-VL-7B-Instruct-LoRA_HF`)
still exist. If a `RAC_video_b3_lora` dir or an `enc3s_*LoRA_HF*` log already exists (e.g. from a
smoke run), STOP and report — do not submit.

**(iii) Single-submit discipline.** One `sbatch` invocation; no `--time` flag added; initial
`PENDING (JobHeldUser)` = WAIT for auto-release (never force-release); no resubmission after any
terminal state (COMPLETED/FAILED/CANCELLED); no mid-run intervention. (An optional 1-epoch smoke of
one LoRA config per draft §13.3 is permitted BEFORE the real submit, provided it writes to a
throwaway group/dir and leaves no `RAC_video_b3_lora` or `enc3s_*LoRA_HF*` artifact that would trip
condition (ii); if in doubt, skip the smoke — the cache dims are already CPU-verified.)

**Executor obligations — write `refine-logs/B3_EXECUTION_RECORD.md`** containing: the job id,
submit/start/end timestamps, and the submit-time re-hash outputs (i). On terminal state:
1. Verify the **3** expected trainlogs exist:
   `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_<JID>.trainlog`,
   each with 30 epochs and parseable VALSEL/FINAL `RESULT_ROW` lines.
2. **Transcribe the raw both-protocol per-seed numbers** (val-sel + final-epoch: Test F1 / acc /
   roc, with line-numbered provenance) from the raw logs. **Apply NO gates and NO interpretation** —
   do not run the G-repro gate, the Namespace-diff gate, or the +0.03/+0.03 decision rule, and do
   not declare pass/fail. That is verdict processing.
3. Report the raw numbers back to the orchestrator.

**Verdict processing = orchestrator + independent verdict review** (per project rule). The
independent verdict reviewer applies, in order: G-repro hard gate (kill rule 1) → Namespace-diff
gate (kill rule 2) → +0.03/+0.03 decision rule under both protocols → the **§2.2 marginal-pass
reporting ruling** (binding) → and folds in the **F-1 correction** (§7d −0.0113→−0.0112) when the
file is edited to record results.

**Out of scope of this authorization:** any second submission; any config/seed/dataset/model/force
change; any edit to either hashed file before submission; any GPU work beyond this one sbatch job.
