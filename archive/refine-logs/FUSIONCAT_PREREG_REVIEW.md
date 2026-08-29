# FUSIONCAT PREREG — INDEPENDENT 0-CONTEXT REVIEW

**Reviewer:** independent 0-context pre-registration reviewer (fresh context; verified every claim
first-hand). **Date:** 2026-07-25 NZST.
**Discipline:** ZERO GPU / SLURM / Modal / training / test-touch. CPU reading + `git`/`sha256sum`/`bash -n`/
`diff` + floor re-parse only. No `state/` mutation, no `research-wiki/` mutation, no push. NO test metric read
or produced. One local commit on `main`.
**Object under review:** `refine-logs/FUSIONCAT_PREREG.md` (commit `511e74c`) + `scripts/slurm/fusioncat_family.sbatch`.
**Supporting recon:** `refine-logs/FUSIONSWAP_FORENSIC_RECON.md` (`934bc9a`). **Promotion:** finding **F83**.

## RULING: **APPROVED-WITH-NOTES**

All seven checklist items (V1–V7) verified PASS/CONFIRM against primary sources. Two trivial, non-blocking
descriptive imprecisions (Notes N1/N2 below) — neither touches a threshold, a sha, the code path, the
pairing, or the decision rule, so **neither requires a prereg edit or re-freeze**. The prereg is honest,
self-consistent, zero-code, and its LOW prior (P(goal) 3–6 %) + PARK→PROMOTE reversal are transcribed loudly.

---

## V1 — Floors re-derived from primary trainlogs + FORMAL thresholds = floor+0.030 exactly → **PASS**

Re-parsed all 6 floor trainlogs with the EXACT embedded parser (val-sel = epoch ≥ warmup 5, max Val acc, roc
tie-break `max(warm, key=(val_acc, val_roc))`; final = max epoch). Every per-seed and mean number **bit-matches**
prereg §2.1/§2.2:

- **ZH 13150** (`enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`): seed0 val ep20
  0.8322/0.8023, seed1 val ep26 0.8255/0.7956, seed2 val ep19 0.8389/0.8065 → **val-sel mean 0.8322/0.8015**;
  finals 0.8456/0.8181, 0.8389/0.8113, 0.8523/0.8226 → **final mean 0.8456/0.8173**. ✓
- **HateMM 13241** (`enc3s_HateMM_…-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`): val-sel mean **0.8775/0.8711**,
  final mean **0.8791/0.8726** (per-seed all match §2.2). ✓

FORMAL thresholds (§2.3) are floor+0.030 **exactly**: ZH val-sel 0.8622/0.8315, final 0.8756/0.8473; HateMM
val-sel 0.9075/0.9011, final 0.9091/0.9026. Independently recomputed — all four legs correct to 4dp.
Floor trainlog headers confirm `fusion_mode='align'` for both floors (13150 Namespace echo verified verbatim).

## V2 — Zero code diff; concat branch committed & reachable; reused shas match → **PASS**

- `git status --porcelain src/` = **empty (CLEAN)**.
- Concat branch is first-class and committed: `classifier.py:85-86` (`if fusion_mode=='concat': input_shape =
  map_dim*2`), `:138-139` (`x = torch.cat((img_feats, text_feats), dim=1)`) — line numbers verified by grep.
- Reachable via `--fusion_mode`: `run_rac.py:118` argparse (`type=str, default="concat"`), threaded into both
  constructor call-sites `:1269` / `:1273`.
- Reused-machinery shas match on-disk to prereg §5.2 **exactly**:
  `classifier.py e7b61df485…`, `run_rac.py b85eb72a69…`, `loss.py 2ae7a73f6d…`, `retrieval.py d43e3bc417…`. ✓

## V3 — Sbatch audit → **PASS**

- `bash -n scripts/slurm/fusioncat_family.sbatch` = **SYNTAX_OK**; CONFIGS = **6 rows**.
- **6 rows correct:** ZH = `$ZH`=`Qwen2.5-VL-7B-Instruct-LoRA_HF` × seed{0,1,2} (pairs 13150, whose header
  confirms model=LoRA_HF); HateMM = `$HM`=`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` × seed{0,1,2} (pairs 13241).
  The ZH model is the **generic** LoRA_HF (not curric) — correct for the 13150 floor.
- **run_one byte-identity to the anchor:** `diff` of the python command block vs `enc3seed_lora_curric.sbatch`
  = **exactly 2 changed lines** — `--fusion_mode "align"`→`"concat"` and `--exp_comment "_${MODEL}"`→
  `"_${MODEL}_fuscat"`. vs the `ncafam_family.sbatch` parent the command block differs only by those plus the
  removal of the inert `${ARM_FLAGS}` (no NCA arm flags set) — i.e. the 3 declared deltas (`--fusion_mode concat`,
  `_fuscat` exp_comment suffix, `GROUP_NAME=RAC_video_fuscat`) + benign cosmetic renames (job-name, header
  comments, RUNLOG prefix `enc3s_`→`fuscat_`, echo strings, b2-push tag). The `ZH`/`HM`/`GROUP_NAME` variable
  block is byte-identical to ncafam except `GROUP_NAME`. No unexpected token.
- **Resources sane:** `--cpus-per-task=8`, `--mem=64G`, `--gres=gpu:a100:1`, **NO `--time`**. Single 8-CPU job
  trivially clears the 16-CPU/128 G/2 GPU cap and the never-two-16-CPU submit-time wedge rule.
- **Collision surface ABSENT:** `logging/Retrieval/*/RAC_video_fuscat*`, `slurm/logs/*fuscat*.trainlog`, and
  `logging/Retrieval/*/RAC_video_smoke_fuscat*` all **do not exist**; banked floor group `RAC_video_lora_curric`
  exists (read-only). `--force False` guards against overwrite.

## V4 — Binding language coherence → **PASS**

- **KS-arm-dead (§3.3):** per-dataset, sign-based — mean paired Δacc ≤ 0 vs own align floor on **either**
  protocol ⇒ that cell KILLED; secondary mean Δacc < +0.015 on **both** protocols (inside ±0.014 head-seed band)
  ⇒ also KILL. Datasets judged **independently**. Rationale (dual-protocol GOAL ⇒ ≤0 on one protocol can never
  clear FORMAL) is sound.
- **FORMAL (§3.2):** +0.030 acc **AND** +0.030 mF1 conjunct, **3/3** seeds positive, under **BOTH** protocols vs
  each dataset's own floor; a cell clears only if it passes both protocols.
- **One bite (§3.6):** both dataset cells share the single "trained concat fusion" multiplicity bite; scope
  FROZEN (no post-hoc `cross`/gated/param-matched/loss/3rd-dataset arm).
- **D7-DEAD:** stated at F0.3, §3.2, §3.5, §8, framing sentence — a formal PASS is a performance/robustness row,
  never a novelty win. No loophole: both protocols judged independently (no protocol/metric-shopping), verdict
  rendered by an independent 0-context reviewer against the prereg verbatim, executor applies no gates.
- **Protocol pinned to precedent:** decision rule quoted verbatim from `exp-encoder-3seed.md:73-85`; the roc
  tie-break val-sel matches the embedded parser (`max by (Val acc, Val roc)`).

## V5 — Param disclosure honest → **PASS**

F0.6 arithmetic independently recomputed: align first-Linear = 1024·1024+1024 = **1,049,600**; concat =
2048·1024+1024 = **2,098,176**; ratio **1.999 ≈ 2.0×** — bit-matches recon §1.3. The **capacity+operator
bundling** scope limit is stated repeatedly and honestly (F0.6, F0.5(a), §3.2, §8, framing sentence): a PASS is
attributed to "concat-fusion head (2× first-Linear params)" as a whole, **not** the operator in isolation; a
param-matched control is explicitly OUT of scope (a future bite). The in-repo `cross` (bmm, ~1.07B params,
~1000×) is disclosed as comparability-broken and EXCLUDED.

## V6 — Codex-gate exemption → **CONFIRM (not overruled)**

House rule: the codex-code-review gate is mandatory only for **model-internals CODE CHANGES**. This family edits
**no source** (`src/` git-clean; all 4 reused shas match; concat is an already-committed first-class branch
selected by a runtime flag). There is **no new code to gate**. The G-repro sha/diff proof (§4.1–4.2) plus the
smoke runtime branch-assert (§4.4.2) are the correct substitutes for a flag-only family. The §4.6 code-fix⇒
re-freeze clause correctly **re-arms** a mandatory codex gate + fresh review if any circumstance ever forces a
source edit. **Exemption is sound — CONFIRMED.**

## V7 — Smoke spec adequate → **PASS**

CPU checks (bash -n, CONFIGS=6, `src/` git-clean, 4 reused shas, collision-absent) are all reproducible and
were re-run this review. The GPU throwaway (per dataset, 3 epochs, throwaway group `RAC_video_smoke_fuscat`)
asserts: (i) completes with no shape error (proves the `input_shape=map_dim*2` first-Linear accepts the concat
vector), (ii) finite losses, (iii) **branch-taken assert** via `grep -m1 "fusion_mode='concat'"` MUST match AND
`grep "fusion_mode='align'"` MUST be empty in the throwaway trainlog. **Verified `run_rac.py:1065` is literally
`print(args)`** — it echoes the full argparse Namespace (the 13150 floor header confirms the echo renders
`Namespace(…fusion_mode='align'…)`), so a concat run's trainlog will contain `fusion_mode='concat'` and no
`fusion_mode='align'`. The assert is real and adequate; cleanup (`rm -rf` smoke dirs + throwaway trainlogs) is
specified; any fail ⇒ HALT (plumbing bug), not a result.

---

## NOTES (non-blocking; no re-freeze required)

- **N1 (§4.1(d) over-lists `--model`).** §4.1(d) says the concat ZH run "differs from [`enc3seed_zh_b3.sbatch`]
  ONLY in `--fusion_mode`/`--model`/`--group_name`/derived-inert (`exp_comment`)". Verified: `enc3seed_zh_b3.
  sbatch` CONFIGS runs ONLY `$LORA=Qwen2.5-VL-7B-Instruct-LoRA_HF` (the CLIP var is defined but not in CONFIGS),
  which is **identical** to the fusioncat ZH arm's model. So `--model` does **not** differ — the list is
  over-inclusive (harmlessly conservative; it names a superset of the true differences). Actual ZH-arm-vs-floor
  differences = `--fusion_mode` + `--group_name` + `--exp_comment` suffix only. Descriptive text; does not
  weaken the same-code pairing.
- **N2 (`run_rac.py:118` default `fusion_mode="concat"`).** The argparse default is `concat`, but every runner
  (floors and this family) passes `--fusion_mode` **explicitly** (`align` / `concat`), so the default never
  bites. Benign; the smoke branch-assert (§4.4.2) would additionally catch any accidental flag drop.

## FREEZE

Approved ⇒ frozen. Freeze block emitted to `refine-logs/FUSIONCAT_FREEZE.md`.

```
FROZEN refine-logs/FUSIONCAT_PREREG.md  c88332b8972e3270081600d0a8cb892a8d24afefbc73e378a5a3104a433c0830
C      scripts/slurm/fusioncat_family.sbatch  62bfb773beeec325096362c86bae1aca8b90c94804ba3798973bbc88e82517fc
REUSED (must be unchanged at submit):
  classifier.py  e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378
  run_rac.py     b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3
  loss.py        2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b
  retrieval.py   d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57
```

**Void-on-edit:** any byte change to the frozen prereg or sbatch, or any drift in the 4 reused shas / non-empty
`git status --porcelain src/` at submit, VOIDS authorization and forces a fresh 0-context review + re-freeze
(and, if a source edit is involved, a mandatory codex gate per §4.6).

**Required statements:** ZERO GPU/SLURM/Modal/training/test-touch spent by this reviewer. No held-out test
metric read or produced. No `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated (the prereg
is frozen at its committed `511e74c` sha; this review does not edit it). Committed on `main`, not pushed.
