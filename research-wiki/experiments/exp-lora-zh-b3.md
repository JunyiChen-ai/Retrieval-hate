---
type: experiment
node_id: exp:exp-lora-zh-b3
title: "B3 — LoRA-Qwen encoder vs frozen-CLIP on MHC-ZH: same-seed paired test under the CURRENT-CODE enc3seed runner (closing the 10-day code gap on the recon preview) (PRE-REGISTRATION, DRAFT-UNREVIEWED)"
idea_id: ""
status: DRAFT-UNREVIEWED
verdict: draft-unreviewed
confidence: n/a
date: "2026-07-14"
hardware: "1x A100 (SLURM), LoRA-Qwen features cached (single shared draw) -> ~20-25 s/run; NO extraction; ~2 min GPU total"
duration: "3 runs (MHC_zh x LoRA x seeds 0/1/2), seconds each, one serial sbatch"
novelty_clause: "PENDING USER RULING. B3 measures the PERFORMANCE clause of the goal ONLY (+0.03 acc AND +0.03 F1). The lever is LoRA-SFT encoder adaptation, an RA-HMD-family technique the project classifies as a 'MIXED performance lever, not novelty' (query_pack.md:44; B1_PREREG_REVIEW.md:64). Whether a LoRA-encoder pass satisfies the goal's 'novel' clause — and whether an 'MLLM-encoder family' (frozen on HateMM + LoRA on ZH) counts as a 'both-datasets' headline — is an EXPLICIT open question for the user, NOT decided here."
provenance: "PRE-REGISTRATION ONLY — NO runs executed yet. Reuses cached LoRA-Qwen features data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt (single shared draw, 2026-07-02). Control arm = B1 job 13115 frozen-CLIP seeds 0/1/2 (existing logs, same enc3seed runner; NOT re-run). Reproduction anchors = arcbase 12223/12224/12225 (LoRA-encoder, old code 2026-07-04). Authoritative recon: refine-logs/B3_FORENSIC_RECON.md. Template + control-arm source: research-wiki/experiments/exp-encoder-zh-b1.md. Floor attribution: refine-logs/B1_PREREG_REVIEW.md Task A."
added: 2026-07-14T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "LoRA-SFT", "RA-HMD", "frozen-CLIP", "encoder-adaptation", "multi-seed", "paired-test", "MHC_zh", "MHC-ZH", "pre-registered", "DRAFT-UNREVIEWED", "B3", "novelty-pending"]
---

# B3 — LoRA-Qwen encoder vs frozen-CLIP on MHC-ZH (PRE-REGISTRATION)

> **STATUS: `CLOSED` 2026-07-14 — VERDICT (`refine-logs/B3_VERDICT_REVIEW.md`; job 13150,
> G-repro bit-exact vs arcbase 12223-25): final-epoch: PASS (MARGINAL); val-selected: FAIL.
> Novelty = PENDING USER RULING (LoRA = RA-HMD-family performance lever). Reporting language
> is BINDING per `B3_PREREG_REVIEW.md` §2.2 — no upgrade.**

**verdict:** `draft-unreviewed` · **confidence:** n/a

## 0. What B3 is (and is NOT) — read first

B3 closes the one impurity the B3 forensic recon (`refine-logs/B3_FORENSIC_RECON.md`)
could not remove from its RECON-PREVIEW: the **10-day code gap**. The recon's paired
LoRA-vs-CLIP preview compares the **old-code** arcbase LoRA runs (12223/24/25, 2026-07-04)
against the **current-code** frozen-CLIP arm (13115, 2026-07-14). B3 re-runs the LoRA arm
**under the identical current-code `enc3seed` runner / python command** that produced 13115,
so the paired verdict is same-code, same-seed, same 149 test videos — a formally clean
head-level paired final-epoch read.

**B3 measures the PERFORMANCE clause ONLY (the goal's "+0.03 acc AND +0.03 F1").** The
lever under test is **LoRA-SFT adaptation of the Qwen2.5-VL-7B encoder** (RA-HMD-family).
The project classifies LoRA as a *"MIXED performance lever, not novelty"*
(`query_pack.md:44`; `B1_PREREG_REVIEW.md:64`). **Whether a LoRA-encoder pass satisfies
the goal's "novel" clause is an EXPLICIT PENDING USER RULING, not decided by B3.** (See §8.)

**Two facts that bound how far a B3 pass can be read (declared up front):**

1. **Opposite-lever-profile across the two datasets (the "unexplained" asymmetry).** The
   two datasets have **opposite encoder-lever profiles**
   (`B3_FORENSIC_RECON.md:213-217`):
   - **HateMM:** the **frozen** Qwen-vs-CLIP swap **wins** (+0.053–0.056 acc / +0.056–0.066
     F1, 3/3 seeds, both protocols — `exp-encoder-3seed.md:25-34`); **LoRA is flat**
     (P9 C3 ≈ floor / below-floor, `B3_FORENSIC_RECON.md:203-212`).
   - **MHC-ZH:** the **frozen** swap **loses** (−0.011 acc paired, B1 job 13115, 20th
     negative — `exp-encoder-zh-b1.md:19`); **LoRA wins** (this test's preview: +0.031 acc
     final-epoch). No single mechanism ("MLLM-as-encoder") passes +0.03/+0.03 on both
     datasets: HateMM needs the frozen encoder, ZH needs LoRA.
2. **Single-encoder-draw limitation.** The 3 B3 LoRA runs all read the **same single shared
   LoRA feature cache** (`..._LoRA_HF.pt`, one SFT draw, 2026-07-02;
   `B3_FORENSIC_RECON.md:86-95,109-124`). The 3 seeds vary **only the downstream RGCL head**
   (init + data-shuffle via `--seed`). So B3's ±band is **head-seed variance on ONE LoRA
   encoder draw** — NOT LoRA-SFT-training-seed variance. This is symmetric with the CLIP
   control (also a single shared cache), so the pairing is a legitimate **head-level**
   same-seed paired test. It is **NOT an encoder-draw paired test.** A full "LoRA-lever"
   claim (encoder-draw stability) would need **≥3 fresh LoRA-SFT retrains + re-extraction**
   (hours of GPU) — **out of B3 scope, pre-declared** (§7, §10).

## 1. Purpose (one line)

Turn the recon's opportunistic same-runner LoRA-vs-CLIP **preview** (final-epoch mean
+0.0313 acc / +0.0453 F1, 3/3 seeds) into a **formally clean same-code same-seed paired
verdict** on MHC-ZH — isolating the LoRA-encoder-adaptation lever from the frozen-encoder
lever that B1 already refuted on ZH — while pre-declaring that this is a PERFORMANCE-clause
measurement on a lever whose NOVELTY status is a pending user ruling.

## 2. Conflict already resolved (recon; not re-litigated here)

The "LoRA-ZH +5.1 vs +1.0" conflict is **resolved** (`B3_FORENSIC_RECON.md:8-78`;
independently `B1_PREREG_REVIEW.md:57-75`): `0.8537` is the **common anchor**; the two
claims measure orthogonal gaps — **+5.1 = LoRA-encoder 0.8537 − frozen-CLIP 0.8027**;
**+1.0 = P9 decision-level LoRA-SFT 0.8635 − LoRA-encoder 0.8537.** Both true. B3 tests the
first gap (LoRA-encoder vs frozen-CLIP) under a clean same-code same-seed protocol.

## 3. Hypothesis (pre-registered)

**H3 (ZH, performance clause).** Replacing the frozen-CLIP video/text encoder with the
**LoRA-SFT-adapted Qwen2.5-VL-7B encoder** (tag `Qwen2.5-VL-7B-Instruct-LoRA_HF`, single
shared cache) on `dataset=MHC_zh` — **every other component identical** (same RGCL head,
topk=20, `lambda_seg=0`, archive OFF, same 579/78/149 split, lr=1e-4 / epochs=30 /
batch=64 / proj=map=1024 / dropout[0.2,0.4,0.1] / hard-neg / hybrid-loss / warmup=5) —
yields **mean paired Δacc ≥ +0.030 AND mean paired ΔmF1 ≥ +0.030 with 3/3 sign** vs the
frozen-CLIP ZH control (13115), judged **independently** under each of the two protocols.

The only manipulated variable is `--model`
(`openai_clip-vit-large-patch14-336_HF` → `Qwen2.5-VL-7B-Instruct-LoRA_HF`). The RGCL head
and all hyperparameters are identical; only the pre-extracted feature `.pt` cache differs.

**H3 conflates three things** (declared, not isolated): encoder identity (7B MLLM vs 300M
CLIP) + **task/language fine-tuning (LoRA-SFT)** + capacity. B1 already isolated encoder
identity **without** fine-tuning (frozen Qwen) and found it **loses** on ZH (−0.011 acc).
So any B3 pass is attributable to **LoRA task/language adaptation of the encoder**, NOT to
MLLM-encoder identity per se — the entire ZH gap rides on the LoRA lever.

**Mechanistic rationale (why LoRA might win on ZH where frozen loses).** The documented ZH
bottleneck is the frozen English-centric CLIP text tower truncating ~97% of Chinese
byte-fragments (`PAPER_MASTER_TABLES.md:188,237`; EXP_p8). Frozen Qwen swaps the tower but
still under-fits the ZH hate task (B1: frozen-Qwen −0.011). LoRA-SFT adapts the encoder to
**task + language**, which is exactly the degree of freedom the frozen swap lacks —
consistent with frozen-Qwen losing while LoRA wins on ZH.

## 4. Design (pre-registered)

- **Dataset:** MHC-ZH only (`dataset=MHC_zh`, Chinese MultiHateClip; Bilibili). Splits
  train 579 / val 78 / test 149 (`data/gt/MHC_zh/{train,val,test}.jsonl`;
  `EXPECTED_TRAIN_N["MHC_zh"]=579`).
- **Training data = the ZH train split ONLY** (user rule). No gold span/attribute
  annotations as supervision; no cross-seed ensembles; no OCR. The RGCL head trains on the
  579-row ZH train split; kNN memory = the ZH train bank. Archive OFF (`archive_feats=None`,
  `lambda_seg=0`).
- **Seeds:** 0 / 1 / 2 — **head-seeds** (init + data-shuffle), paired within seed against
  the same-seed CLIP arm. (Encoder fixed; see §0 limitation 2.)
- **Arms:**
  - **treatment (NEW run):** `LoRA-Qwen` = tag `Qwen2.5-VL-7B-Instruct-LoRA_HF`, single
    shared cache, seeds 0/1/2 head-seeds, under the current-code `enc3seed` runner.
  - **control (EXISTING logs, NOT re-run):** frozen-CLIP arm of **B1 job 13115**, seeds
    0/1/2, **same runner, byte-identical python argv except `--model`**
    (`B3_FORENSIC_RECON.md:129-146`). Final-epoch: s0 0.8054/0.7706, s1 0.8054/0.7542,
    s2 0.8322/0.7913 acc/F1 (`B1_EXECUTION_RECORD.md:105-114`).
- **Feature inputs already cached — NO extraction (asset check, verified this prep):**
  - `data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` (16.6 MB, 2026-07-02, (579,3584)img/(579,3584)txt/(579,)lbl)
  - `dev_seen_..._LoRA_HF.pt` (2.24 MB, (78,3584)/(78,3584)/(78,))
  - `test_seen_..._LoRA_HF.pt` (4.28 MB, (149,3584)/(149,3584)/(149,))
  - Keys `{ids,img_feats,text_feats,labels}`; **single file per split, no seed suffix** —
    the single shared LoRA draw (§0 limitation 2). LoRA-vs-CLIP test ids are set- AND
    order-identical (`B3_FORENSIC_RECON.md:94-95`) → identical 149 test videos.
  - **What is missing to run: NOTHING** (no extraction, no new config). The only artifact
    to author is the sbatch CONFIGS/GROUP edit (§9).

### Config-match verification (to run FIRST — the G-repro hard gate, §6)

Before tabulating, the fresh LoRA s0/s1/s2 must reproduce arcbase 12223/12224/12225
final-epoch to 4 printed decimals. Because the LoRA features are **cached** and the argv is
**byte-identical to arcbase except `--group_name` / derived `output_path`** (both inert),
the run is deterministic given the seed → **exact reproduction is the expectation**, not a
hope. Any mismatch = code drift ⇒ HALT (§6, kill rule 1).

## 5. Protocols (both reported, judged independently — NO protocol selection)

Transcribed from `exp-encoder-3seed.md:66-71` (via B1):

- **(A) val-selected:** pick epoch ≥ warmup 5 with max Val_Retrieval acc (roc tie-break);
  report that epoch's **Test** macroF1 / acc / roc.
- **(B) final-epoch:** report **Test** macroF1 / acc / roc at the last trained epoch (29).

**Primary protocol = (B) final-epoch — REPORTING-EMPHASIS ONLY, NOT a decision gate**
(inherited verbatim from B1's Rev-3 rationale). Sole basis: the 78-sample ZH val set
imposes a documented ~2-acc-point val-selection tax (`PAPER_MASTER_TABLES.md:56-57`;
`exp-consensus-zh-seeds.md:127-133`), so selection-free is the less noisy lens for a 3-seed
paired ZH test. **No headroom argument enters this designation.** Both protocols are judged
independently under the identical rule (§6 rule 4); the fixed write-up format
"final-epoch: pass/fail; val-selected: pass/fail" governs.

## 6. Decision rule (pre-registered, transcribed verbatim from `exp-encoder-3seed.md:73-85`)

> For each dataset x protocol:
> 1. **Per-seed paired difference** delta = (LoRA − CLIP) for acc and macroF1 at seeds 0/1/2.
> 2. **3-seed mean +/- std** of the paired delta; **sign consistency** (how many of 3 positive).
> 3. n=3 is too small for a formal bootstrap; report the paired-t statistic **as an
>    effect-size descriptor only** alongside the mean/std and sign count — no significance
>    claim is made from n=3.
> 4. **Pass criterion (per dataset x protocol):** mean paired delta_acc ≥ +0.030 AND
>    mean paired delta_mF1 ≥ +0.030 AND sign consistency 3/3 positive.
> 5. **Headline claim ("MLLM-as-encoder helps"):** requires the pass criterion met on
>    **≥ 2 datasets** under a stated protocol. Each protocol is judged separately.

**Application to B3 (with the family/novelty caveat).** Judge ZH under the identical rule
(4). A ZH **PASS** under a protocol supplies a candidate second dataset alongside HateMM
(which passed BOTH protocols) **within that same protocol column** — HateMM must be paired
under the same protocol ZH passes. **BUT the two datasets' passes ride on DIFFERENT levers**
(HateMM = frozen swap; ZH = LoRA-SFT). So any resulting "MLLM-as-encoder helps on ≥2
datasets" is a **FAMILY claim** (the MLLM-encoder family: frozen variant on HateMM, LoRA
variant on ZH), **NOT a single-mechanism claim.** Whether that family framing is acceptable
as a "both-datasets" headline is **part of the user's novelty ruling (§8), not decided by
B3.** A ZH **FAIL** under a protocol leaves HateMM as the only formally passing encoder
dataset in that column (status quo, `MEMORY.md`).

## 7. RECON-PREVIEW — same-code same-seed paired LoRA vs frozen-CLIP, MHC-ZH (existing logs, ZERO NEW GPU)

**Not a verdict — a preview from existing primary logs, to be RE-CONFIRMED by B3's fresh
LoRA runs after the G-repro gate.** LoRA = arcbase 12223/12224/12225 (old code); CLIP =
13115 (current code, `B1_EXECUTION_RECORD.md:105-125`). Numbers re-read directly from the
primary trainlogs by this prep agent (numeric-provenance discipline).

### 7a. Final-epoch (protocol B, the recon's preview — primary-verified)

| seed | LoRA acc | CLIP acc | **Δacc** | LoRA F1 | CLIP F1 | **ΔF1** |
|---|---|---|---|---|---|---|
| 0 | 0.8456 | 0.8054 | **+0.0402** | 0.8181 | 0.7706 | **+0.0475** |
| 1 | 0.8389 | 0.8054 | **+0.0335** | 0.8113 | 0.7542 | **+0.0571** |
| 2 | 0.8523 | 0.8322 | **+0.0201** | 0.8226 | 0.7913 | **+0.0313** |
| **mean** | **0.8456** | **0.8143** | **+0.0313** | **0.8173** | **0.7720** | **+0.0453** |

- **Acc:** mean **+0.0313** (≥ +0.030, **marginal**), **3/3 sign positive**. (Per-seed:
  seed2 = +0.0201 < 0.030, but rule 4 requires only mean ≥ bar + 3/3 sign, not per-seed.)
- **F1:** mean **+0.0453** (clears cleanly), **3/3 sign positive**.
- ⇒ **Final-epoch PREVIEW = PASS** (marginal on acc; the +0.0313 mean sits only +0.0013
  above the bar — a small downward wobble on the fresh run would tip it, flagged as a
  fragility, though the G-repro gate expects exact reproduction).

### 7b. Val-selected (protocol A — computed by this prep agent via the sbatch parser; NOT in the recon)

LoRA val-sel from arcbase logs (warmup≥5 max-val-acc, roc tie-break): s0 (ep20) 0.8322/0.8023,
s1 (ep26) 0.8255/0.7956, s2 (ep19) 0.8389/0.8065 acc/F1. CLIP val-sel from 13115: s0 (ep29)
0.8054/0.7706, s1 (ep28) 0.8054/0.7579, s2 (ep25) 0.8121/0.7742.

| seed | LoRA acc | CLIP acc | **Δacc** | LoRA F1 | CLIP F1 | **ΔF1** |
|---|---|---|---|---|---|---|
| 0 | 0.8322 | 0.8054 | **+0.0268** | 0.8023 | 0.7706 | **+0.0317** |
| 1 | 0.8255 | 0.8054 | **+0.0201** | 0.7956 | 0.7579 | **+0.0377** |
| 2 | 0.8389 | 0.8121 | **+0.0268** | 0.8065 | 0.7742 | **+0.0323** |
| **mean** | **0.8322** | **0.8076** | **+0.0246** | **0.8015** | **0.7676** | **+0.0339** |

- **Acc:** mean **+0.0246** (**< +0.030 — FAILS the acc bar**), 3/3 sign positive.
- **F1:** mean **+0.0339** (clears), 3/3 sign positive.
- ⇒ **Val-selected PREVIEW = FAIL** (acc mean below bar; the 78-dev val-selection tax bites
  the LoRA arm too). The metric-shopping rule (§6 rule 4) requires BOTH; acc-short = FAIL.

### 7c. Preview verdict shape (fixed format)

**final-epoch: PASS (marginal, +0.0313 acc / +0.0453 F1); val-selected: FAIL (+0.0246 acc <
bar / +0.0339 F1).** This exact split is the pre-registered expectation; B3's fresh LoRA runs
either confirm it (G-repro passes → identical numbers) or the G-repro gate HALTs on drift.

### 7d. Load-bearing decomposition (all same-runner, seeds 0/1/2 mean, final-epoch)

| ZH arm (final-ep) | acc | Δ vs frozen-CLIP |
|---|---|---|
| frozen-CLIP (13115) | 0.8143 | — (baseline) |
| **frozen-Qwen** encoder swap (13115, B1) | 0.8031 | **−0.0112 (FAILS — B1 20th negative)** |
| **LoRA-Qwen** (12223-25 → B3 re-run) | 0.8456 | **+0.0313** |

⇒ The frozen MLLM-encoder swap *loses* on ZH; the LoRA fine-tune *wins*. LoRA − frozen-Qwen
(paired) = +0.0425 acc: the fine-tuning is what adds on top of the frozen 7B features. **The
entire ZH gap is LoRA task/language adaptation, not MLLM-encoder identity.**

## 8. Novelty clause — EXPLICIT PENDING USER RULING (B3 does not decide this)

B3 answers only: *does the LoRA-encoder swap clear +0.03 acc AND +0.03 F1 on ZH under a
clean same-code same-seed paired test?* The **separate** questions B3 leaves open for the
user:

1. **Does a LoRA-SFT-encoder performance pass count toward the goal's "novel" clause?** LoRA
   is an RA-HMD-family technique the project labels a *"MIXED performance lever, not novelty"*
   (`query_pack.md:44`; `B1_PREREG_REVIEW.md:64`). B3 takes NO position on whether adapting
   it to hateful-video ZH is "novel enough."
2. **Does an "MLLM-encoder family" (frozen on HateMM + LoRA on ZH) count as a "both-datasets"
   headline?** The two passes use different mechanisms (§0 fact 1, §6). Accepting a
   family-level framing is a user call. If the user requires a *single* mechanism to pass
   ≥2 datasets, neither frozen nor LoRA qualifies (frozen: HateMM-only; LoRA: ZH-only on
   this preview).
3. **Barred-comparison accounting.** `PAPER_MASTER_TABLES.md:58` declares the LoRA-Qwen main
   stack and the frozen-CLIP floor *"不可直接同格并比"* (not directly comparable side-by-side)
   in the paper's own tables. B3's same-runner same-seed pairing is the cleanest paired read
   that exists, but whether it overrides that accounting note for a paper claim is a user
   decision.

## 9. Runs to execute (this campaign) — pending authorization

Runner = `scripts/slurm/enc3seed_zh_b3.sbatch` (to author: verbatim copy of
`enc3seed_zh_b1.sbatch` with ONLY the CONFIGS block → 3 LoRA rows AND `GROUP_NAME` →
`RAC_video_b3_lora`; diff in `refine-logs/B3_IMPL_NOTES.md`). One serial sbatch, current
code; each run = the exact `enc3seed`/`train_archive_baseline` python command, differing
only in `--dataset MHC_zh` / `--model Qwen2.5-VL-7B-Instruct-LoRA_HF` / `--seed` /
`--group_name RAC_video_b3_lora`:

| # | dataset | model | seed | role |
|---|---|---|---|---|
| 1 | MHC_zh | Qwen2.5-VL-7B-Instruct-LoRA_HF | 0 | treatment + G-repro hard gate vs 12223 |
| 2 | MHC_zh | Qwen2.5-VL-7B-Instruct-LoRA_HF | 1 | treatment + G-repro hard gate vs 12224 |
| 3 | MHC_zh | Qwen2.5-VL-7B-Instruct-LoRA_HF | 2 | treatment + G-repro hard gate vs 12225 |

Control arm (frozen-CLIP seeds 0/1/2) is **NOT re-run** — B3 re-reads the existing 13115
logs (`B1_EXECUTION_RECORD.md:105-114`); those were produced by the identical runner, so the
pairing is same-code. (3 runs, not 6.)

### GROUP vs FORCE decision — **distinct GROUP `RAC_video_b3_lora`, `FORCE=False` (default)**

**Decision:** use a fresh `--group_name RAC_video_b3_lora`; leave `--force False`. **Do NOT
use `FORCE=True` with the arcbase group.** Rationale (collision semantics verified in
`src/run_rac.py:898-908`; `group_name` usage at `:855,900`):

- **Why there is a collision at all.** `run_rac.py:899-900` builds
  `output_path = .../Retrieval/MHC_zh/<group_name>/<exp_name>/`. `exp_name` is
  seed+model-derived and, for a LoRA seed-s run, is **byte-identical to the existing arcbase
  dir** `RAC_..._seed{s}_hybrid_loss_Qwen2.5-VL-7B-Instruct-LoRA_HF` (seeds 0-4 already exist
  under `RAC_video_archive_seeds/MHC_zh/`, 2026-07-04). With the arcbase group, `:901-908`
  sees the dir exists and — with `force=False` — **raises `"Output path already exists,
  aborting..."` (HARD ABORT, all 3 seeds crash)**. `force=True` would instead **overwrite**
  those dirs in place.
- **Why distinct GROUP beats FORCE=True (3 reasons):**
  1. **Non-destructive.** `FORCE=True` would **overwrite the arcbase 12223-25 output
     artifacts** (ckpt/metrics) — the very anchors the G-repro hard gate reproduces against.
     Distinct GROUP writes to a fresh `RAC_video_b3_lora/` tree and **preserves the anchors
     untouched**.
  2. **Cleaner Namespace.** `force` stays `False`, **matching both** the CLIP control 13115
     (`force=False`) and the arcbase anchors (`force=False`). So the only Namespace deltas
     become `{model, exp_comment, group_name, output_path}` — all output-path/inert fields
     (see §6 kill rule 2). `FORCE=True` would introduce a `force` Namespace divergence.
  3. **`group_name` is computationally inert.** It feeds ONLY `output_path` at
     `run_rac.py:900` (the local at `:855` is dead); it never touches model/data/training.
     Verified: changing the group cannot change any result → the **G-repro reproduction
     expectation is unaffected** by the group swap (features cached + argv otherwise
     byte-identical ⇒ bit-identical training).
- **No-collision / overwrite semantics (explicit):** `RAC_video_b3_lora/MHC_zh/` **does not
  exist** (verified `ls` → none) ⇒ `:901-902` creates it fresh; `force=False` never trips
  the abort; **nothing is overwritten anywhere.** New trainlogs
  `enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_<JID>.trainlog` also do not exist
  (only `arcbase_*-LoRA_HF*` logs exist) ⇒ no log collision (the enc3s log name omits
  group_name, so this holds regardless).

## 10. Kill rules (pre-registered)

1. **G-repro — HARD gate.** Each fresh LoRA seed-s final-epoch **Test** acc AND macroF1 must
   reproduce its arcbase anchor to **4 printed decimals**:
   | seed | anchor job | final-ep acc | final-ep F1 |
   |---|---|---|---|
   | 0 | 12223 | 0.8456 | 0.8181 |
   | 1 | 12224 | 0.8389 | 0.8113 |
   | 2 | 12225 | 0.8523 | 0.8226 |
   Cached features + byte-identical argv (except inert group/output-path) ⇒ **exact match is
   EXPECTED**. Any >0.0001 mismatch = **HALT, do not tabulate, open a code-drift
   investigation** (the arcbase archive-OFF path is code-stable across this window — two
   corroborating gates: 12223 = 12149 bit-for-bit; frozen-Qwen s0 in 13115 = old 1151518
   exactly — so a mismatch is a real regression, not noise).
2. **Namespace-diff gate.** The fresh LoRA Namespace must be identical to the 13115 CLIP
   Namespace **except** `{model, exp_comment, group_name, output_path}` (all inert), and
   identical to the 12223-25 arcbase Namespace **except** `{group_name, output_path}` (same
   model, same exp_comment). Any *substantive* field difference (topk/epochs/batch/lr/
   warmup/proj/map/fusion/loss/metric/hybrid/lambda_seg/archive_feats/…) ⇒ HALT.
3. **No protocol-shopping.** Primary = final-epoch (reporting-emphasis only, §5). Both
   protocols reported regardless of outcome; fixed write-up format
   "final-epoch: pass/fail; val-selected: pass/fail". The ≥2-datasets headline requires ZH to
   pass under the **same** protocol HateMM passed (HateMM passed both) — AND is a FAMILY
   claim subject to §8.
4. **No metric-shopping.** Both mean Δacc AND mean ΔmF1 must clear +0.03 with 3/3 sign for a
   PASS; an F1-only or acc-only move is reported as FAIL-with-direction. (Preview: this is
   exactly why val-selected = FAIL — acc mean +0.0246 short despite F1 +0.0339.)
5. **Single test touch.** The fresh LoRA-arm ZH-test read is the ONE budgeted evaluation for
   this cell under current code (§11). No re-runs with tweaked knobs on the ZH test set under
   this pre-registration.

## 11. Test-touch discipline

- **Prior exposure (declared honestly).** The MHC-ZH test set is **not virgin**: (i) the
  arcbase **12223-27** LoRA-encoder runs (old code, 2026-07-04) already read ZH test
  per-epoch and their final-epoch numbers are the recon preview / G-repro anchors — **prior
  exposure under old code**; (ii) the **13115** CLIP+frozen-Qwen arms (B1) already read ZH
  test (B1's single budgeted touch). B3 re-uses (ii) by re-reading existing logs (no new
  touch) and re-produces (i) under current code.
- **B3's budgeted touch = ONE.** The specific pre-registered question — "does the
  LoRA-encoder swap yield +0.03/+0.03 on ZH under the archive-OFF RGCL head at 3 head-seeds,
  same-code same-seed paired vs frozen-CLIP" — is allotted **exactly one** evaluation: the
  fresh LoRA s0/s1/s2 read (test per-epoch, per the parent protocol precedent that reads test
  each epoch and selects post-hoc). No adaptive re-running against ZH test. Because the
  arcbase runs already touched ZH test under old code, this is a **re-measurement under
  current code**, not a fresh first look — accounted for here, pre-declared.

## 12. GPU budget

- **Extraction: 0 GPU-s** — all 3 LoRA caches exist (§4).
- **Training: 3 runs × ~20-25 s** with cached features (recon §6: "each run ~20-25 s;
  ~2 min GPU total"). Parent job 12850 cached-feature runtimes corroborate (~20-52 s/run).
- **Total: 3 runs serial ≈ 1-2 min compute; < ~10 min wall** incl. conda/disk_guard/parse/
  b2-push. 1× A100 / 8 CPU / 64 GB (inherited parent headers; within 16 CPU / 128 GB / 2 GPU
  cap). No `--time` (project rule); expect initial `PENDING (JobHeldUser)`, wait for
  auto-release, never force.

## 13. Single-submit ceremony (pre-registered)

1. Freeze this pre-registration (review sign-off pending).
2. Author `scripts/slurm/enc3seed_zh_b3.sbatch` (3 LoRA rows + `GROUP_NAME=RAC_video_b3_lora`);
   diff-verified CONFIGS+GROUP-only vs `enc3seed_zh_b1.sbatch` (`refine-logs/B3_IMPL_NOTES.md`).
3. Optional smoke: 1-epoch dry run of one LoRA config to confirm the 3584-d cache loads and
   wires into `classifier_hateClipper` (3584→1024).
4. One `sbatch` submission of the 3-run serial job. No mid-run resubmissions.
5. Read back every number from the raw `enc3s_MHC_zh_*-LoRA_HF_*` trainlogs (line-numbered
   provenance), apply the **G-repro hard gate FIRST** (kill rule 1), then the Namespace-diff
   gate, then tabulate per-seed deltas vs the 13115 CLIP arm and apply the decision rule
   verbatim under both protocols.

## 14. Readiness verdict (what remains before submission)

1. **Fresh pre-registration review** — PENDING (this is `DRAFT-UNREVIEWED`).
2. **Implementation check** — runner to author: `scripts/slurm/enc3seed_zh_b3.sbatch`
   (CONFIGS+GROUP-only copy of `enc3seed_zh_b1.sbatch`; see `refine-logs/B3_IMPL_NOTES.md`
   for the diff, cache check, collision semantics, and hashes).
3. **Conditional authorization** — explicit user/main go (GPUs shared with the user's own
   loop; `CLAUDE.md` — every GPU task via SLURM, subagents do the work).
4. **Single submit** — one serial sbatch, ~10 min, 1 A100.

**Nothing is blocked on data or compute.** Gates = review + authorization.

## 15. Connections
- extends → `exp:exp-encoder-zh-b1` (the frozen-Qwen-vs-CLIP ZH test, B1 20th negative; B3 adds the LoRA-encoder arm on the same runner)
- reuses-control-arm-of → `exp:exp-encoder-zh-b1` (13115 frozen-CLIP seeds 0/1/2 = B3's control, not re-run)
- reproduces → arcbase `12223/12224/12225` (LoRA-encoder, old code; B3 = same runs under current code)
- contrasts-with → `exp:exp-encoder-3seed` (HateMM frozen-swap PASS both protocols — the opposite-lever-profile partner)
- scoped-by → `refine-logs/B3_FORENSIC_RECON.md` (conflict resolution, preview, seed-semantics limitation)
- floor-attribution → `refine-logs/B1_PREREG_REVIEW.md` Task A (0.8537 = LoRA-Qwen, not frozen)
- novelty-clause → PENDING USER RULING (§8)

## 16. Revision history

| rev | date | status | change | authority |
|---|---|---|---|---|
| r0 | 2026-07-14 | DRAFT-UNREVIEWED | Initial pre-registration (recon-scoped; no runs). Both-protocol preview from primary logs (final-ep PASS marginal / val-sel FAIL); GROUP=RAC_video_b3_lora + FORCE=False decision; G-repro + Namespace kill gates; novelty clause declared PENDING USER RULING; single-encoder-draw + opposite-lever-profile limitations pre-declared. | B3 prep agent |
| r1 | 2026-07-14 | CLOSED | Executed + independently reviewed. Verdict (`refine-logs/B3_VERDICT_REVIEW.md`, job 13150): G-repro PASS bit-exact vs arcbase 12223-25 (6/6 readings, 0 mismatch); Namespace PASS; final-epoch PASS (MARGINAL) mean Δacc +0.0313 / ΔmF1 +0.0453, sign 3/3; val-selected FAIL (mean Δacc +0.0246 < +0.030 fails the AND-rule, ΔmF1 +0.0339, sign 3/3). Binding write-up format per `B3_PREREG_REVIEW.md` §2.2 — no upgrade. The §7d frozen-Qwen decomposition erratum (−0.0113 → −0.0112) was applied in place by the independent verdict reviewer (`B3_VERDICT_REVIEW.md` §5a); the number is verified −0.0112 here and NOT re-edited. Novelty = PENDING USER RULING. | B3 verdict reviewer + closure archivist |
