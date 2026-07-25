# FUSIONCAT — Trained concat-fusion head Pre-Registration — 1 arm × {ZH, HateMM}, one bite

**Author:** fusioncat prereg author (CPU-only; no GPU/SLURM/Modal spent; NO job submitted).
**Date:** 2026-07-25 NZST.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced; no job submitted.
**Implements:** `refine-logs/FUSIONSWAP_FORENSIC_RECON.md` (commit `934bc9a`) — its code recon (§1: the `concat` branch
is already first-class and wired, ZERO new code), ban adjudication (§2: F50/F75 letter-overreach both CONFIRMED —
trained fusion is outside both ban letters, `align`/Hadamard is the **only** fusion ever run on video), param-count
table (§1.3: concat 2.10M vs align 1.05M = 2.0×, within the ~2× comparability guidance), floors (§3: ZH 13150 /
HateMM 13241), machinery/timing (§3: the `ncafam_family.sbatch` pattern, 24 runs = 9m28s), and the honest LOW prior
(§5: P(goal) 3–6%). The recon **recommended PARK**; the orchestrator **PROMOTED it to run** — that reversal and its
rationale are transcribed loudly in §11.1 (finding **F83**).
**House-style precedent:** `refine-logs/ZHPROMPT_PREREG.md` (most recent house style: §4.6 code-fix⇒re-freeze clause,
DEV items, sha freeze list, binding-language layout), `refine-logs/NCA_PREREG.md` (single-group additive-flag family
on cached LoRA features), `research-wiki/experiments/exp-encoder-3seed.md:73-85` (the enc3s protocol + decision rule
verbatim).

## Title + claim scope (verbatim)

> This measurement tests **one axis the deployed head never varied — the FUSION OPERATOR that combines the two
> projected modality streams before the MLP.** The deployed head hardcodes `fusion_mode=align` (Hadamard product
> `x = img ⊙ text`, `input_shape = map_dim`); **`align` is the ONLY fusion ever run on video** (recon §1.1). This
> family swaps in the **already-wired, un-measured `concat` branch** (`x = [img ; text]`, `input_shape = map_dim*2`,
> `classifier.py:85-86,:138-139`) — a **trained** nonlinear fusion optimized end-to-end with the **unchanged**
> triplet+0.5·BCE hybrid + FAISS mining, deployment (top-20 rank-weighted signed-cosine kNN vote over own-train
> memory) **unchanged**. It is a **single arm (`concat`) × 2 datasets (`MHC_zh`, `HateMM`) × 3 head-seeds = 6 head
> runs**, paired **within head-seed**, **dual-protocol** (val-selected AND final-epoch), each dataset vs its **own**
> banked `align` floor (ZH = job 13150, HateMM = job 13241), each dataset trained ONLY on its own train split. It is
> a **door-closer-grade PURE-PERFORMANCE bite, NOT an expected-+3 bet**: the recon prices **P(goal) at 3–6 %**
> (§5) and **recommended PARK**; the orchestrator promoted it under the **goal-hook continuation directive** because
> it is the campaign's **cheapest GPU cell (~0.1 GPU-h, ZERO source-code diff)** and a **genuinely never-measured
> first-class axis** whose park rested only on base-rate priors, not a measured arithmetic cap (§11.1, finding F83).
> The fusion operator is an **architecture/capacity knob**; even a formal PASS is a performance/robustness row —
> **D7-DEAD, NO novelty claim regardless of outcome** (§0 F0.3). The honest counter-pressures driving the LOW prior
> are kept in the open (§5): `concat` is a **capacity + operator change bundled** (2.0× the first-Linear params —
> the effect cannot be cleanly attributed to the operator alone, an honest scope limit); the head-side empirical
> base rate is **0-for-~20** promoted; the binding ZH val-sel leg must add ≈+0.030 through the 78-dev selection wall
> (F45/F63); and HateMM's floor (0.879) sits so near ceiling that a +0.030 FORMAL pass is arithmetically implausible.

The cell under test is the deployed RGCL `classifier_hateClipper`, **`fusion_mode=concat`** (`x = torch.cat((img,
text), dim=1)`), triplet+0.5·BCE, AdamW head over cached LoRA embeddings, 30 epochs, warmup 5, top-20 arithmetic
signed-cosine kNN vote, `--force False`, paired **3-seed within head-seed** vs each dataset's banked SAME-ENCODER
`align`-floor, dual-protocol. **The ONLY manipulated variable between a treatment run and its floor is
`--fusion_mode` (`align`→`concat`)**; every other token of the python command is byte-identical to the floor runner
(§4.2 diff = exactly 2 lines). **Any other fusion mode (`cross`/gated/cross-attn), a param-matched control, a
different loss, a third dataset, or a wording/knob edit is OUT of this prereg** — it spends the family and re-costs a
bite (§3.6).

---

## 0. Binding facts / honesty clauses (all present; pre-declared)

**F0.1 — Test is NOT virgin (declared).** Both test splits were already read under the identical `enc3s` protocol by
the banked floors and every prior head family (13115/13150 B1/B3, 13241 curric, 13471 bidir, 13468 readout, 13478
head-recipe, 13482 NCA). This family's reads are **re-measurements under the identical protocol**, not first
exposures. There are **6 budgeted dataset×seed test evaluations** = {ZH, HateMM} × seed{0,1,2}. **Zero test-touch
before the independent verdict.**

**F0.2 — Deterministic feature caches; the ONLY per-run variable vs the floor is the fusion operator (pre-declared,
material).** The concat runs read the **same banked LoRA feature caches** the floors read (fusion happens inside the
head from cached `img_feats`/`text_feats`; the encoder caches are fusion-agnostic — no re-extraction). `--seed`
controls head-init + data-shuffle; pairing is per head-seed (concat seed s − floor seed s), `s∈{0,1,2}`. At a given
seed the **only** difference vs the floor is `--fusion_mode` (`align`→`concat`), which changes the fusion op AND the
first-Linear input dim (`map_dim`→`map_dim*2`); no other RNG-drawing branch is touched (the `mod_dropout` RNG path
is gated to `fusion_mode=='align'` **and** flag-on, `classifier.py:129` — OFF here, draws nothing).

**F0.3 — Novelty = D7-DEAD, say it plainly.** The fusion operator is an **architecture/capacity knob** (which of the
head's 3 already-coded fusion branches combines the two streams). **Novelty-nil / D7-DEAD:** even a formal PASS is a
performance/robustness row ("trained concat fusion vs Hadamard on <dataset>"), the same D7 class as frame-budget
(F67) / head-recipe / readout / loss-family (F75) — **never** a novelty contribution. The reviewer-question value
("why Hadamard and not concat?" answered with a measured number) is a paper *sentence*, not a novelty *mechanism*.
**Pure-performance + door-closer, under an explicit LOW prior.**

**F0.4 — Single-dataset own-train-split VETO compliance (hard user veto; trivially clean).** No SFT, no data build,
no cross-dataset mixing, no re-extraction: each head trains on its dataset's **own train split only** (identical
corpus to its floor). No gold spans/attributes, **NO OCR channel** (user veto), no cross-seed ensemble; raw videos
never leave the machine (only derived `.pt` logging → B2). All standing vetoes cleared.

**F0.5 — Honest prior is LOW (recon §5; pre-declared). This is a door-closer-grade prior run.** The recon prices
**P(clearing the full GOAL bar, ≥+0.030 acc AND +0.030 mF1, 3/3, BOTH protocols, on ≥1 dataset) at 3–6 %** (recon
§5 table, "Goal (≥+3 on ≥2 ds) ~0.03–0.06"). Four disclosed headwinds, none a ban:
- **(a) capacity+operator bundled, un-attributable.** `concat` is a 2.0× first-Linear param bump (§0.6 F0.6) — any
  lift confounds "wider first layer" with "concat operator"; a param-matched control is explicitly OUT of scope (a
  future bite), so a PASS is honestly reported as "concat-fusion arm (2× params) lifts X", never "the concat
  operator is better".
- **(b) F66 near-neighbour analogy (non-binding).** `concat` is a **symmetric reshaper** of the two frozen streams;
  F66's β-decomposition prices the legal symmetric-reshaping slice at **+0.001–0.006** (recon §2). This is an
  **analogy, non-binding on a *trained* reshaper** (the recon's explicit reason the park was base-rate-only, not a
  measured cap, §11.1) — but it is a real headwind: the mean lift a symmetric fusion can convert is small.
- **(c) head-side base rate 0-for-~20.** F70 readout perm-null, F73 SAM/mod-dropout ±noise, F75 loss-family 0/8
  formal — no head-side change has cleared the goal 3/3. Concat is the same head-side family.
- **(d) two structural walls per dataset.** ZH's binding leg is **val-sel through the 78-item dev** (F45/F63
  selection noise ≈ the ±0.014 head-seed band it must beat 3/3); HateMM's floor (0.879) is **near ceiling** so a
  +0.030 FORMAL pass (→0.909) is arithmetically implausible — HateMM is a hold-the-pass leg, not a goal leg.

**F0.6 — Param-count disclosure (recon §1.3; CPU-verified arithmetic this prereg; LOAD-BEARING scope limit).** Fusion
changes ONLY the first MLP `Linear`'s input dim (`img_proj`/`text_proj` are pre-fusion, identical across arms;
`classifier.py:81-82`). At `map_dim=proj_dim=1024`:

| arm | fusion input dim | first-Linear params | ratio vs align | comparability |
|---|---|---|---|---|
| **align (floor)** | 1024 | 1024·1024 + 1024 = **1,049,600** | 1.0× | — |
| **concat (this family)** | 2048 | 2048·1024 + 1024 = **2,098,176** | **1.999× ≈ 2.0×** | within the "~2×" guidance ✓ |

**Honest scope limit:** the concat arm is a **capacity + operator change bundled** (a wider first Linear AND a
different fusion op). We do **not** run a param-matched control in this bite (recon §4 lists it as a separate,
optional arm). Therefore a PASS is attributable to "concat-fusion head (2× first-Linear params)" as a whole, **not**
to the operator in isolation — stated at verdict time. The in-repo `cross` (bmm outer-product, ~1.07B params,
~1000×) is **comparability-broken and EXCLUDED** (recon §1.3); it is NOT this family's arm.

**F0.7 — ZERO source-code diff; the head path is the deployed floor path (pre-declared; CPU-verified this prereg).**
`git status --porcelain src/` = **CLEAN** (working tree == committed). This family **edits no source**: `concat` is
an already-wired first-class branch (`classifier.py:85-86` dim-setup, `:138-139` forward; `run_rac.py:118` parses
`--fusion_mode`, `:1269-1273` threads it into the constructor). The reused machinery shas are frozen at their
**current committed values** (§5.2, verified this prereg):
`classifier.py e7b61df4…`, `run_rac.py b85eb72a…`, `loss.py 2ae7a73f…`, `retrieval.py d43e3bc4…`. The `run_one`
python command is **byte-identical** to `enc3seed_lora_curric.sbatch` (the anchor that produced the HateMM floor
13241) **except exactly 2 lines** — `--fusion_mode "align"`→`"concat"` and `--exp_comment "_${MODEL}"`→`"_${MODEL}
_fuscat"` (§4.2 `diff`); the fresh `GROUP_NAME=RAC_video_fuscat` is a variable definition outside the command block.

**F0.8 — run_rac.py additive keys are inert (pre-declared).** `run_rac.py` currently carries the already-landed
NCA/head-recipe additive-gated keys (`--head_loss` default `triplet`, `--mixup` default `False`, `--sam`/`--mod_
dropout` off, etc.); **this family sets NONE of them** ⇒ the flags-off path is byte-identical to the 13150/13241
floor runner (the NCA-verdict-blessed additive-gating fact). The concat runs pass only the deployed knobs + the one
changed `--fusion_mode` token.

---

## 1. Pipeline spec — fully pinned (head-only; ONE job; nothing left to interpretation)

**Stage 0 — none.** No SFT / no data build / no re-extraction. The heads consume the **banked LoRA feature caches**
directly: ZH `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` (the 13150
inputs); HateMM `…/HateMM/…-LoRA-curric_HF.pt` (the 13241 inputs). Both are read-only inputs.

### 1.1 The single stage layout — ONE sbatch: 6 head runs

- **Submit:** `sbatch scripts/slurm/fusioncat_family.sbatch` (authored this prereg — artifact C, §5).
- **Job-chain shape:** **ONE sbatch, 8 CPU / 64 G / 1×A100**, running 6 chained head runs sequentially in one
  process (the `ncafam_family.sbatch` / `enc3seed_lora_curric.sbatch` single-job precedent — both datasets in ONE
  job). An **8-CPU single job** trivially satisfies the "**never two concurrent 16-CPU jobs**" submit-time
  aggregate wedge rule (one job, 8 < 16). Peak footprint 8 CPU / 64 G / 1 GPU — within the 16 CPU / 128 G / 2 GPU cap.
- **What it runs:** 6 head-only runs on the banked caches (~20–50 s each; recon §3 timing NCA 24 runs = 9m28s):
  {ZH `Qwen2.5-VL-7B-Instruct-LoRA_HF`, HateMM `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`} × seed{0,1,2},
  `--fusion_mode concat`, `--group_name RAC_video_fuscat`, `--exp_comment "_${MODEL}_fuscat"`, `--force False`.
- **Pairing:** ZH per head-seed (concat seed s − **13150** seed s); HateMM per head-seed (concat seed s − **13241**
  seed s). `--seed` controls head-init + data-shuffle; the only difference vs the floor at a given seed is the
  fusion operator.
- **Output:** `slurm/logs/fuscat_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_<JID>.trainlog` (ZH) +
  `fuscat_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_<JID>.trainlog` (HateMM); each trainlog's head
  line is the full `print(args)` Namespace (`run_rac.py:1065`) showing `fusion_mode='concat'` (§4.4 assert target).
- **Cost:** ~0.1 GPU-h total (6 × ~20–50 s + disk_guard/B2 overhead), recon §3/§5.

### 1.2 The patch — NONE

No file is edited. `--fusion_mode concat` selects an existing branch; `input_shape = map_dim*2` and the
`torch.cat` forward are already coded (`classifier.py:85-86,:138-139`). `run_rac.py`, `classifier.py`, `loss.py`,
`retrieval.py` — **NO edit** (§4.1, F0.7; shas unchanged).

---

## 2. Comparison floors — INDEPENDENTLY RE-DERIVED from raw trainlogs (numeric-provenance discipline)

Every number below was independently re-parsed **this prereg** from the raw trainlogs with the EXACT embedded parser
(val-sel = epoch ≥ warmup 5 max `Val_Retrieval` acc, roc tie-break; final = max epoch). Both floor means **bit-match**
the recon §3 table (ZH 0.8322/0.8015 val-sel, 0.8456/0.8173 final; HateMM 0.8775/0.8711 val-sel, 0.8791/0.8726
final) and `NCA_PREREG.md §2` to 4dp.

### 2.1 ZH floor — job **13150** (`Qwen2.5-VL-7B-Instruct-LoRA_HF`, generic-LoRA / B3; group `RAC_video_b3_lora`; goal-relevant, marginal)

| seed | val-sel ep | val-sel acc/mF1 | final ep | final acc/mF1 |
|---|---|---|---|---|
| 0 | 20 | 0.8322 / 0.8023 | 29 | 0.8456 / 0.8181 |
| 1 | 26 | 0.8255 / 0.7956 | 29 | 0.8389 / 0.8113 |
| 2 | 19 | 0.8389 / 0.8065 | 29 | 0.8523 / 0.8226 |
| **mean** | | **0.8322 / 0.8015** | | **0.8456 / 0.8173** |

Files: `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`.

### 2.2 HateMM floor — job **13241** (`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`, curriculum-LoRA; group `RAC_video_lora_curric`; near ceiling)

| seed | val-sel ep | val-sel acc/mF1 | final ep | final acc/mF1 |
|---|---|---|---|---|
| 0 | 29 | 0.8791 / 0.8730 | 29 | 0.8791 / 0.8730 |
| 1 | 14 | 0.8744 / 0.8678 | 29 | 0.8791 / 0.8724 |
| 2 | 10 | 0.8791 / 0.8724 | 29 | 0.8791 / 0.8724 |
| **mean** | | **0.8775 / 0.8711** | | **0.8791 / 0.8726** |

Files: `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`.

### 2.3 Concrete FORMAL promote thresholds (mean +0.030) + noise band

- **ZH (vs 13150):** val-sel mean acc ≥ **0.8622** AND mF1 ≥ **0.8315**; final mean acc ≥ **0.8756** AND
  mF1 ≥ **0.8473** (all with 3/3 per-seed positive).
- **HateMM (vs 13241):** val-sel mean acc ≥ **0.9075** AND mF1 ≥ **0.9011**; final mean acc ≥ **0.9091** AND
  mF1 ≥ **0.9026** (3/3 positive). *Note: floor 0.879 near ceiling ⇒ +0.030 FORMAL is arithmetically implausible
  (F0.5 (d)); HateMM is a hold-the-pass leg.*
- **Head-seed noise band (KS-arm-dead secondary read, §3.3):** ±**0.014** — the established house head-seed spread
  descriptor (`B3_PREREG_REVIEW.md`, `NCA_PREREG.md §2.3`). A 3-seed mean move `< +0.015` on both protocols sits
  inside this band.

---

## 3. Decision rule + kill-bars (paired, both protocols judged independently, 3/3 sign, per-dataset, pre-declared)

### 3.1 Decision rule — verbatim from `exp-encoder-3seed.md:73-85` (treatment = concat arm; control = the dataset's banked align floor)

> For each dataset × protocol: (1) per-seed paired difference δ = (treatment − control) for acc and macro-F1 at
> seeds 0/1/2; (2) 3-seed mean ± std + sign consistency (how many of 3 positive); (3) n=3 too small for a
> bootstrap — report the paired-t **as an effect-size descriptor only**, no significance claim; (4) **pass =
> mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive**; (5) headline claim
> requires pass on ≥ 2 datasets under a stated protocol; both protocols judged separately; verdict written
> exactly "final-epoch: pass/fail; val-selected: pass/fail".

Both protocols judged **independently** (no protocol-shopping, no metric-shopping). Judged **per dataset**; control =
that dataset's OWN banked align floor (ZH §2.1 / HateMM §2.2).

### 3.2 FORMAL promote bar (goal-facing; per dataset)

House **+0.030 acc AND +0.030 mF1** conjunct, **3/3 seeds positive**, under **BOTH** protocols vs the dataset's
banked floor (§2.3). Below the conjunct under a protocol → **NEGATIVE** on that protocol. A dataset cell clears
FORMAL only if it passes BOTH protocols. **D7-DEAD (F0.3): even a formal PASS is a performance/robustness row
("trained concat fusion vs Hadamard on <dataset>"), NEVER a novelty win** — and honestly attributed to the 2× first-
Linear capacity + concat operator BUNDLED (F0.6), not the operator alone.

### 3.3 KS-arm-dead — the KILL bar (per dataset; SIGN-based; task req 1)

- **KS-arm-dead (per-dataset screen kill).** A dataset's **3-seed mean paired Δacc ≤ 0 vs its own align floor on
  EITHER protocol ⇒ that dataset cell KILLED** (banked as the concat-fusion null for that dataset). **Secondary
  read:** mean paired Δacc `< +0.015` on **BOTH** protocols (inside the ±0.014 head-seed band, §2.3) ⇒ also KILL.
  **Per-dataset, datasets INDEPENDENT:** a KS-arm-dead ZH cell does NOT auto-kill the HateMM cell, and vice versa;
  each is judged only vs its own floor. State each killed cell explicitly at verdict time. (Rationale for the
  "either-protocol" gate: the GOAL bar is dual-protocol, so a cell ≤0 on even one protocol can never clear FORMAL.)
- **Family one bite (§3.6):** the two dataset cells share the single "trained concat fusion" multiplicity bite.

### 3.4 KS-regression note (per dataset)

If concat − floor **mean Δacc ≤ −0.014** on a leg (below the full head-seed spread, §2.3), concat fusion
**degraded** the head → bank "concat fusion hurts on <dataset>." A note within the KS-arm-dead frame, not a separate
multiplicity bite.

### 3.5 D7-DEAD closure (no novelty claim regardless of outcome)

Whatever the numbers, the fusion operator is a **generic architecture/capacity knob** — a formal PASS is a
robustness/ablation row, a KILL is a door-closer for the fusion axis; **neither yields a novelty contribution**
(F0.3). The paper role is: "we measured trained concat fusion against the deployed Hadamard fusion — here is the
number." That is the entire deliverable.

### 3.6 Multiplicity + scope of THIS submit (pre-declared)

- **ONE sbatch = ONE pre-registered family = ONE multiplicity bite** across both dataset cells. `concat` is the ONLY
  arm; the two datasets **share** the single "trained concat fusion" bite.
- **Scope FROZEN.** **NO** post-hoc arm additions: a **`cross`/gated/cross-attention** arm, a **param-matched
  control** (a narrower first Linear to isolate the operator from capacity), a **different loss**, a **third
  dataset/encoder**, or any knob edit is a **new** pre-declared family and re-costs a bite.
- **A surviving cell still owes the full ceremony** (this prereg → independent 0-context review → freeze-hash →
  SLURM); this prereg does **not** discharge that. This is the ONLY fusion-operator bite.

### 3.7 Gate order

G-repro (§4.1: `src/` git-clean + reused-machinery sha re-verify + run_one 2-line-diff-vs-anchor proof) → smoke
(§4.4: `bash -n` + config count + collision check + GPU throwaway concat run per dataset incl. the **fusion_mode='
concat'** args-echo assert) → single test-touch (the 6 head reads) → per dataset: **KS-arm-dead** → **FORMAL promote
bar (both protocols)**. The verdict is rendered by an **independent 0-context reviewer against this prereg VERBATIM**;
the executor transcribes raw both-protocol per-seed numbers (line-numbered) and applies NO gates/interpretation.

---

## 4. G-repro + smoke plan + collision safety

### 4.1 G-repro discipline

- **(a) src/ git-clean gate.** At submit time re-run `git status --porcelain src/` — must be **empty** (this family
  edits no source; F0.7). Any staged/unstaged src edit = authorization VOID.
- **(b) Reused-machinery sha gate.** Re-run `sha256sum` on the 4 reused files — must match §5.2:
  `classifier.py e7b61df4…`, `run_rac.py b85eb72a…`, `loss.py 2ae7a73f…`, `retrieval.py d43e3bc4…`. Any mismatch =
  authorization VOID.
- **(c) Artifact sha gate.** Re-run `sha256sum scripts/slurm/fusioncat_family.sbatch` + this file — must match the
  §5.3 freeze block (filled by the reviewer at freeze).
- **(d) Same-code proof (INCLUDING the floor runner).** The `run_one` python command of `fusioncat_family.sbatch` is
  **byte-identical** to `enc3seed_lora_curric.sbatch` (the HateMM-floor 13241 runner) **except exactly 2 lines**
  (`--fusion_mode`, `--exp_comment`); `diff` of the command block = those 2 lines only (§4.2). The ZH floor 13150
  runner (`enc3seed_zh_b3.sbatch`) carries the same command; the concat ZH run differs from it ONLY in
  `--fusion_mode`/`--model`/`--group_name`/derived-inert (`exp_comment`).

### 4.2 CPU verification (run this prereg — PASS)

- `git status --porcelain src/` = **empty (CLEAN)** — **PASS**.
- `sha256sum` on the 4 reused files matches §5.2 (`e7b61df4…`/`b85eb72a…`/`2ae7a73f…`/`d43e3bc4…`) — **PASS**.
- `bash -n scripts/slurm/fusioncat_family.sbatch` = **SYNTAX_OK**; CONFIGS = **6 rows** (2 datasets × 3 seeds) —
  **PASS**.
- `diff` of the `run_one` python command (`python ./src/run_rac.py … --Faiss_GPU`) vs
  `enc3seed_lora_curric.sbatch` = **exactly 2 changed lines**: `--fusion_mode "align"`→`"concat"` and
  `--exp_comment "_${MODEL}"`→`"_${MODEL}_fuscat"` — **PASS** (the GROUP_NAME change is a variable definition,
  outside the command block).
- Param-count arithmetic (F0.6): concat first-Linear `2048·1024+1024 = 2,098,176`; align `1024·1024+1024 =
  1,049,600`; ratio `1.999 ≈ 2.0×` — **PASS** (bit-matches recon §1.3).

### 4.3 Collision safety (verified this prereg — ABSENT; re-check at submit)

- `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_fuscat*` — do NOT exist ⇒ fresh group; `--force False` never trips
  the `run_rac.py:1059-1062` hard-abort; distinct `--model`+`_fuscat` `exp_comment` keeps the dirs distinct from the
  banked align floors (`RAC_video_b3_lora`, `RAC_video_lora_curric`) and from each other.
- `slurm/logs/*fuscat*.trainlog` — do NOT exist ⇒ no trainlog collision (the `fuscat_` prefix + `_${MODEL}_` tag
  separate the rows).
- `scripts/slurm/fusioncat_family.sbatch`, `refine-logs/FUSIONCAT_PREREG.md` — created by this prereg (no prior).
- Banked floor caches/trainlogs (13150/13241) + the LoRA feature caches are **read-only inputs**; this family writes
  none. Smoke throwaways (`logging/_smoke_fuscat/`, `_smoke_fuscat` group) — deleted after smoke; must NOT persist.

### 4.4 Smoke plan (executor runs BEFORE the real submit; leave no artifact that trips §4.3)

1. **CPU checks (already run this prereg, $0; re-run at submit):** `bash -n` = SYNTAX_OK; CONFIGS word-split = 6
   rows; `git status --porcelain src/` empty; the 4 reused shas match §5.2; `ls logging/Retrieval/*/RAC_video_
   fuscat*` = **absent** (collision check, task req 4). Reference for the executor.
2. **GPU throwaway concat run per dataset (~1–2 min each; task req 4).** For EACH dataset, run ONE short concat head
   to a throwaway group, e.g. `python ./src/run_rac.py … --dataset MHC_zh --model Qwen2.5-VL-7B-Instruct-LoRA_HF
   --fusion_mode "concat" --epochs 3 --group_name RAC_video_smoke_fuscat --exp_comment "_smoke" --force False`
   (and the HateMM `…-LoRA-curric_HF` variant). Assert: (i) it **completes** (no crash / no shape error — proves the
   `input_shape=map_dim*2` first-Linear accepts the concat vector), (ii) **losses are finite** across the 3 epochs
   (grep the loss lines, no `nan`/`inf`), (iii) **the concat branch is actually taken** — a one-line grep of the
   `run_rac.py:1065` args echo in the throwaway trainlog: **`grep -m1 "fusion_mode='concat'" <trainlog>` MUST match
   AND `grep "fusion_mode='align'" <trainlog>` MUST be empty** (NO code edits — this uses the existing `print(args)`
   Namespace echo). Then `rm -rf logging/Retrieval/*/RAC_video_smoke_fuscat*` + the throwaway trainlogs. **Any fail
   ⇒ HALT** (plumbing/branch bug), not a result.

### 4.5 CODEX GATE — NOT REQUIRED (pre-declared; recon §3)

Per recon §3, **Arm A (concat) needs ZERO new code ⇒ no codex-code-review gate** (the gate is mandatory only for the
gated/cross-attn arms that require a `classifier.py` diff, which are OUT of this bite, §3.6). The G-repro sha/diff
proof (§4.1-4.2: `src/` git-clean + 2-line command diff) replaces the code-review gate for this flag-only family. The
smoke branch-assert (§4.4.2) is the runtime confirmation that the flag selects the intended branch.

### 4.6 Code-fix ⇒ re-freeze clause (verbatim-ported from `ZHPROMPT_PREREG.md §4.6` / `NCA_PREREG.md §4.5`)

**This family is flag-only and MUST NOT need a code fix.** If ANY circumstance forces a source edit (e.g. the smoke
reveals a latent bug in the shared `concat` branch), the affected file shas change, the ZERO-code-diff premise is
BROKEN, and **the freeze block (§5.3) MUST be re-issued** with a fresh independent 0-context review (and a codex gate
becomes mandatory, per §4.5's condition). No code edit lands silently post-freeze; the executor re-runs `sha256sum`
+ `git status --porcelain src/` at submit and any mismatch = authorization VOID.

---

## 5. Artifacts authored this prereg + hash-freeze block

### 5.1 New artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current; reviewer freezes) |
|---|---|---|---|
| P | `refine-logs/FUSIONCAT_PREREG.md` | **NEW** — this file | *(reviewer fills at freeze)* |
| C | `scripts/slurm/fusioncat_family.sbatch` | **NEW** — ONE job: 6 concat head runs (2 datasets × 3 seeds), `RAC_video_fuscat`, `run_one` python command byte-identical to `enc3seed_lora_curric.sbatch` except `--fusion_mode "concat"` + `--exp_comment "_${MODEL}_fuscat"` | `62bfb773beeec325096362c86bae1aca8b90c94804ba3798973bbc88e82517fc` |

### 5.2 Reused-unchanged machinery (verified sha / git-clean THIS prereg; do NOT edit)

| path | role | sha256 (current, verified) |
|---|---|---|
| `src/model/classifier.py` | deployed head + the `concat`/`align`/`cross` fusion branches (`:85-90`/`:138-143`; ZERO edit) | `e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378` |
| `src/run_rac.py` | deployed head runner (`--fusion_mode` parse `:118`, threading `:1269-1273`, args echo `:1065`; NCA/head-recipe keys inert; ZERO edit) | `b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3` |
| `src/model/loss.py` | deployed triplet+BCE hybrid (unchanged; ZERO edit) | `2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b` |
| `src/utils/retrieval.py` | deployed FAISS mining + top-20 kNN vote (unchanged; ZERO edit) | `d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57` |
| `scripts/slurm/enc3seed_lora_curric.sbatch` | same-code anchor for §4.2 (produced HateMM floor 13241) | *(committed; git-clean verified §4.2)* |
| `scripts/slurm/enc3seed_zh_b3.sbatch` | ZH floor 13150 runner (same command modulo `--model`/`--group_name`) | *(committed; git-clean)* |
| ZH `data/CLIP_Embedding/MHC_zh/…-LoRA_HF.pt` + HateMM `…-LoRA-curric_HF.pt` caches | the 13150/13241 head inputs (read-only; not clobbered) | *(present; verified untouched)* |

### 5.3 Hash-freeze (to be filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file FUSIONCAT_PREREG.md, after review>
C 62bfb773beeec325096362c86bae1aca8b90c94804ba3798973bbc88e82517fc  scripts/slurm/fusioncat_family.sbatch
REUSED (must be unchanged at submit):
  classifier.py  e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378
  run_rac.py     b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3
  loss.py        2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b
  retrieval.py   d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57
```
Executor re-runs `sha256sum` on C (+ this file) + confirms the 4 reused shas + `git status --porcelain src/` empty
at submit time; any mismatch = authorization VOID. **If any circumstance forces a source edit, C (and the affected
reused sha) change and the freeze block MUST be re-issued (§4.6).**

---

## 6. Single-submit / execution plan + resource plan

**Order (ONE SLURM job):**

1. Pre-submit: G-repro (§4.1) → smoke incl. **GPU throwaway concat run per dataset + the `fusion_mode='concat'`
   args-echo assert** (§4.4). Only on all-clear:
2. `sbatch scripts/slurm/fusioncat_family.sbatch` → 6 head runs (~0.1 GPU-h). Produces the 6 `fuscat_*` trainlogs +
   the `RAC_video_fuscat` output dirs.

**Resource plan (STANDING INFRA RULE compliant):** the sbatch requests **`--cpus-per-task=8`, `--mem=64G`, 1×A100**.
Single 8-CPU job ⇒ peak footprint **8 CPU / 64 G / 1 GPU** — within the 16 CPU / 128 G / 2 GPU cap, and **NEVER two
16-CPU jobs in flight** (an 8-CPU single job trivially clears the 29 h-wedge submit-time aggregate rule — re-check no
other 16-CPU job is queued/running at submit). `conda activate HateVideo`; `sbatch` with **NO `--time`**; initial
`PENDING (JobHeldUser)` = **WAIT for auto-release, never force** (CLAUDE.md). Sources `conda.sh` directly, runs the
≥20 G `disk_guard.sh` (wall-time padding, `|| true`, touches `slurm/logs/disk_guard.log` — expected, non-fatal),
B2-pushes derived `logging` at the end (videos never leave — CLAUDE.md boundary).

**Test-touch:** the 6 head reads are the ONLY budgeted fusioncat test evaluations (2 datasets × 3 seeds); zero
test-touch before the verdict. **The executor transcribes raw both-protocol per-seed numbers (line-numbered) and
applies NO gates/interpretation** — the verdict (KS-arm-dead → FORMAL, per dataset) is rendered by an **independent
0-context reviewer against this prereg VERBATIM.**

**No job is submitted by this prereg author.** Submission happens only after the independent 0-context review +
hash-freeze.

---

## 7. Outcome table template (filled ONLY from raw trainlogs at verdict time)

### 7.1 Per-dataset table (fill from `fuscat_<DATASET>_<MODEL>_seed{0,1,2}_<JID>.trainlog`)

**ZH concat vs floor 13150 (§2.1):**

| seed | protocol | concat acc/F1 | floor acc/F1 | Δ(concat−floor) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8322/0.8023 | ___ |
| 1 | val-sel | ___ | 0.8255/0.7956 | ___ |
| 2 | val-sel | ___ | 0.8389/0.8065 | ___ |
| **mean** | **val-sel** | ___ | **0.8322/0.8015** | **___** |
| 0 | final-ep | ___ | 0.8456/0.8181 | ___ |
| 1 | final-ep | ___ | 0.8389/0.8113 | ___ |
| 2 | final-ep | ___ | 0.8523/0.8226 | ___ |
| **mean** | **final-ep** | ___ | **0.8456/0.8173** | **___** |

**HateMM concat vs floor 13241 (§2.2):**

| seed | protocol | concat acc/F1 | floor acc/F1 | Δ(concat−floor) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8791/0.8730 | ___ |
| 1 | val-sel | ___ | 0.8744/0.8678 | ___ |
| 2 | val-sel | ___ | 0.8791/0.8724 | ___ |
| **mean** | **val-sel** | ___ | **0.8775/0.8711** | **___** |
| 0 | final-ep | ___ | 0.8791/0.8730 | ___ |
| 1 | final-ep | ___ | 0.8791/0.8724 | ___ |
| 2 | final-ep | ___ | 0.8791/0.8724 | ___ |
| **mean** | **final-ep** | ___ | **0.8791/0.8726** | **___** |

### 7.2 Fixed write-up format (per §3.1 rule 5 + the bars §3.2/§3.3)

```
Smoke: fusion_mode='concat' branch-assert <PASS | HALT>.  (must PASS before any cell is judged)
ZH concat:     final-epoch: <pass/fail>; val-selected: <pass/fail>  [FORMAL §3.2]. KS-arm-dead: <KILLED | survives>.
HateMM concat: final-epoch: <pass/fail>; val-selected: <pass/fail>  [FORMAL §3.2]. KS-arm-dead: <KILLED | survives>.
(+ KS-regression note if any mean Δacc ≤ −0.014; capacity+operator-bundled caveat (F0.6) on any PASS; D7-DEAD always.)
```

---

## 8. What a PASS / FAIL means for the goal (D7 boundary — fusion-operator is DEAD, not user-pending)

- **Both cells KS-arm-dead (recon prior — the honest expected outcome, P(goal) 3–6 %):** trained concat fusion
  carries no net vote signal beyond Hadamard on either dataset ⇒ the fusion-operator axis is **CLOSED** at ~0.1
  GPU-h, and the live reviewer question ("why Hadamard, not concat?") is answered with a **measured null**.
  Cleanest cheap outcome: a genuinely un-enumerated first-class axis converted to a measured door-closer.
- **A cell survives KS but < FORMAL bar:** measured-not-promoted limbo (bank the weak positive; still D7-DEAD, still
  capacity-confounded per F0.6). Most plausibly ZH on ONE protocol, the other eaten by 78-dev selection noise (wall
  (d)); HateMM cannot clear FORMAL (near ceiling).
- **A cell clears the FORMAL bar (≥+0.030/+0.030, 3/3, both protocols):** a paper-worthy **robustness/ablation** row
  ("trained concat fusion (2× first-Linear params) lifts the <dataset> cell"), **honestly attributed to the
  capacity+operator bundle, NOT the operator alone** (F0.6), and **NOT a novelty win** (F0.3). A surviving cell owes
  the full ceremony + owes a param-matched control before any operator-level claim (a future bite, §3.6).

**Framing sentence (verbatim):** *this measurement tests one axis the deployed head never varied — the fusion
operator combining the two projected streams — by swapping the already-wired, never-run `concat` branch for the
deployed Hadamard `align`, trained end-to-end with the unchanged triplet+BCE + kNN vote, 3-seed paired dual-protocol
on ZH and HateMM vs each dataset's banked align floor; concat is a capacity+operator change bundled (2× first-Linear
params); a pass is a performance/robustness row, NEVER a novelty win — fusion-operator is D7-DEAD; run under an
explicit LOW prior (P(goal) 3–6 %) as the campaign's cheapest zero-code door-closer.*

---

## 9. Provenance index

- Recon (PARK-recommended, orchestrator-PROMOTED; code branches, param counts, ban adjudication, floors, timing,
  prior): `refine-logs/FUSIONSWAP_FORENSIC_RECON.md` (`934bc9a`).
- Promotion decision: `autoresearch/goal_mllm_plus3/state/findings.jsonl` finding **F83** (2026-07-25T08:55:00Z) —
  transcribed §11.1.
- Fusion code (verbatim): `src/model/classifier.py:71,73` (`fusion_mode` arg/attr), `:85-90` (dim-setup),
  `:138-143` (forward branches); `src/run_rac.py:118` (`--fusion_mode` parse, default `concat`), `:1269-1273`
  (instantiation), `:1065` (`print(args)` Namespace echo — the smoke branch-assert target).
- Floors (re-derived §2): ZH `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`;
  HateMM `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`.
- Protocol + decision rule (verbatim): `research-wiki/experiments/exp-encoder-3seed.md:73-85`.
- Same-code anchor + head runner: `scripts/slurm/enc3seed_lora_curric.sbatch` (HateMM-floor 13241 runner),
  `scripts/slurm/ncafam_family.sbatch` (single-group two-dataset family precedent), `scripts/slurm/enc3seed_zh_b3.
  sbatch` (ZH-floor 13150 runner); output-path keying `src/run_rac.py:1010-1062`.
- Walls / counter-pressure: F45/F63 (78-dev selection noise), F66 (symmetric-reshaper legal slice +0.001–0.006,
  non-binding analogy on a trained reshaper), F70 (readout perm-null), F73 (SAM/mod-dropout ±noise), F75 (loss-
  family 0/8 formal) — the head-side 0-for-~20 base rate.

**Required statements:** ZERO GPU/SLURM/Modal spent by this prereg author (only pure-CPU login-node floor
re-parsing, `sha256sum`, `bash -n`, `git status`, `diff`, and param arithmetic, seconds; no held-out test metric
produced). All floor numbers re-parsed from banked completed-run trainlogs (numeric-provenance discipline; both bit-
match recon §3). `src/` git-clean; NO source-code edit. No `state/` mutated. No `research-wiki/` mutated. NO job
submitted. Committed on `main`, not pushed.

---

## 10. DEV items — foreseeable execution pitfalls

1. **DEV-A (run_rac.py evaluates test EVERY epoch — no-peek discipline).** The head logs `Test_Retrieval` every
   epoch; the val-sel protocol selects the epoch by **Val** only (parser §2), then reads Test at that epoch.
   Discipline: **selection uses ONLY Val**; the per-epoch Test lines are transcribed but never used to pick an epoch
   (identical to all banked floors). Any tie-break goes on a **throwaway** group, never the banked `RAC_video_fuscat`
   dirs.
2. **DEV-B (JobHeldUser wait).** Initial `PENDING (JobHeldUser)` is expected — **WAIT for auto-release, never force**
   (CLAUDE.md).
3. **DEV-C (disk_guard wall-time padding).** The sbatch runs `disk_guard.sh` at start (≥20 G reclaim, gated on
   verified B2 copies); it can add minutes and touch `slurm/logs/disk_guard.log` — expected, non-fatal (`|| true`).
4. **DEV-D (naming collisions).** `RAC_video_fuscat` + `fuscat_*` trainlogs are collision-checked ABSENT (§4.3) and
   distinct from every banked group/tag. Re-check at submit — if any `fuscat` artifact exists, HALT (a prior partial
   run) rather than overwrite; `--force False` also guards the output dirs.
5. **DEV-E (concat first-Linear shape).** The concat branch builds an `nn.Linear(map_dim*2, proj_dim)` = 2048→1024
   first layer (vs align's 1024→1024). The smoke throwaway run (§4.4.2) confirms the head instantiates and trains
   without a shape error before the real submit; a shape mismatch here = HALT, not a result.
6. **DEV-F (never-2×16-CPU at submit).** Before `sbatch`, confirm no other 16-CPU job is queued/running (the
   submit-time aggregate wedge rule). This job is 8-CPU so it clears trivially, but the check is mandatory.
7. **DEV-G (HateMM near ceiling — read the FORMAL bar honestly).** HateMM floor 0.879 ⇒ FORMAL needs 0.909 val-sel /
   0.909 final; a within-noise concat move cannot clear it (F0.5 (d)). HateMM is a hold-the-pass leg; its most
   probable verdict is KS-arm-dead or survives-but-<FORMAL, NOT a FORMAL pass. Do not over-read a small HateMM lift.

---

## 11. Promotion note + deviations from the recon — flagged loudly

### 11.1 The recon recommended PARK; the orchestrator PROMOTED it — transcribed (finding F83, LOAD-BEARING)

The recon (`934bc9a`) **BOTTOM LINE = PARK** (§0/§6): a genuine F50/F75 letter-gap, cheap to close, but **no
defensible ≥+3-on-≥2-datasets prior** — binding ZH-val-sel leg selection-walled + F66-arithmetic-analogy-capped,
head-side base rate 0-for-~20, **P(goal) 3–6 %**. The orchestrator **overruled the park** (finding **F83**,
2026-07-25T08:55:00Z), verbatim rationale:

> ORCHESTRATOR OVERRULE RATIONALE (goal-hook 试到最后 + stop-hook enforcement): unlike F71/F78/F79/F82 parks (each
> had measured arithmetic caps or absent targets), fusion park rests only on base-rate priors; cost 0.1 GPU-h
> zero-code is the cheapest GPU cell of the campaign; genuinely never-measured first-class axis.

**This prereg therefore runs a door-closer-grade LOW-prior cell BY DESIGN**, under the user's never-stop goal-hook
continuation directive — NOT because the arithmetic favors a win (it does not: P(goal) 3–6 %, §0.5). The prereg's job
is to convert a genuinely un-measured first-class axis into a measured number at minimum cost, with the honest prior
declared up front. This is stated so no downstream reader mistakes the LOW prior for a hidden expectation of success.

### 11.2 Deviations from the recon design (§4)

1. **DEV-1 (concat-ONLY, drop gated + cross-attn). Recon-aligned.** Recon §4 sketched Arm A (concat, $0 code) + an
   optional Arm B (gated, ~20 new lines + codex gate) and dropped C (cross-attn) / bmm-`cross` (comparability-
   broken). I take **A only** — the ZERO-code arm — matching the orchestrator's F83 promotion ("concat arm = zero
   new code"). Gated/cross-attn are OUT of scope (§3.6); adding either re-costs a bite and re-arms the codex gate.
2. **DEV-2 (KS-arm-dead pinned as "≤0 on EITHER protocol"; task req 1). Task-aligned.** The task and recon §4 pin
   mean Δacc ≤ 0 on EITHER protocol ⇒ that dataset cell dead (+ secondary `< +0.015` both protocols). Justified: the
   GOAL bar is dual-protocol, so a cell ≤0 on even one protocol can never clear FORMAL. Datasets judged INDEPENDENTLY
   (no cross-dataset auto-defund), §3.3.
3. **DEV-3 (single group `RAC_video_fuscat` + distinct `exp_comment` `_${MODEL}_fuscat`). Same-code-favorable.**
   Recon §4 sketched `RAC_video_fusionswap`; I use `RAC_video_fuscat` (task req 2's group name) with the two
   datasets separated by `--model`⇒`exp_comment` (the NCA single-group precedent), which lets the `run_one` python
   command stay byte-identical to `enc3seed_lora_curric.sbatch` modulo the 2 declared lines (§4.2).
4. **DEV-4 (ONE sbatch, both datasets, 6 rows, cloned from `ncafam_family.sbatch`). Budget-aligned.** The exact
   parent precedent is `scripts/slurm/ncafam_family.sbatch` (the two-dataset two-model single-group family: ZH =
   `…-LoRA_HF`/13150, HateMM = `…-LoRA-curric_HF`/13241 — the identical floor pairing this family needs), whose
   python command is itself byte-identical to the `enc3seed_lora_curric.sbatch` anchor. I clone it down to 1 arm ×
   6 rows. ONE submission = the budget (§6, task req 5).
5. **DEV-5 (no codex gate; §4.5). Recon-aligned.** Recon §3 explicitly exempts the flag-only concat arm from the
   codex gate; the G-repro sha/diff proof + smoke branch-assert replace it. (A codex gate re-arms only if §4.6 is
   ever triggered.)
