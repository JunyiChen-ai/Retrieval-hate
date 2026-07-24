# HEAD-RECIPE Pre-Registration — SAM + modality-dropout head-training family (ZH + HateMM, 2 arms, one bite)

**Author:** head-recipe prereg author (CPU-only; no GPU/SLURM/Modal spent; NO job submitted).
**Date:** 2026-07-25 NZST.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced; no job submitted.
**Implements:** `refine-logs/HEADRECIPE_FORENSIC_RECON.md` (commit `44918e0`, the GO recon) — its 2-arm family
design (one bite, multiplicity-safe), the two pre-pinned hyperparameters + the identity-fill ruling, the
ban-collision rulings, and the kill-bar skeleton transcribed and re-verified below. Deviations from the recon
are flagged **loudly** in §11.
**House-style precedent:** `refine-logs/FRAME16_PREREG.md`, `refine-logs/CAND2_CURRICULUM_PREREG.md` (binding
language, F0.x honesty clauses, pinned pipeline, re-derived line-cited floors, freeze block, single-submit plan,
outcome-table template); `research-wiki/experiments/exp-encoder-3seed.md` (the 12850 encoder-swap protocol +
decision rule verbatim :73-85).

## Title + claim scope (verbatim)

> This measurement tests **two cheap head-TRAINING recipes** — **ARM A: SAM** (sharpness-aware minimization, a
> flat-minima optimizer) and **ARM B: modality-dropout** (an identity-fill stream-dropout regularizer) — on the
> deployed RGCL align-fusion head, run entirely on **CACHED LoRA features**, **3-seed paired** within head-seed
> against each dataset's own banked floor (ZH generic-LoRA job 13150; HateMM curriculum-LoRA job 13241),
> dual-protocol (val-selected AND final-epoch), on **ZH (`MHC_zh`) and HateMM, each trained ONLY on its own
> train split**. It is a **PURE-PERFORMANCE** family: both recipes are generic training-dynamics knobs
> (**D7-DEAD**, F0.3) — even a formal PASS is a performance/ablation row, never a novelty contribution. The two
> arms are pre-registered as **ONE family = ONE multiplicity bite** with **hyperparameters FROZEN at recon-pin**
> (SAM `rho=0.05`; mod-dropout `p=0.3`, identity-fill mandatory under Hadamard align); this is a **cheap,
> one-bite closure** of the two remaining head-training-dynamics escape hatches, whose realistic best case is
> hardening the **marginal ZH** cell. This prereg decides the **performance clause only.**

The cells under test are the deployed RGCL `classifier_hateClipper`, `fusion_mode=align`, triplet+BCE hybrid loss,
AdamW head over cached embeddings, 30 epochs, warmup 5, `--force False`, paired **3-seed within head-seed** vs the
banked floors, dual-protocol. **The treatment and its floor read the byte-identical banked feature cache** (no
re-extraction, no re-SFT) — so the ONLY difference at a given seed is the head-training recipe (§0 F0.2). ARM A
adds `--sam True --sam_rho 0.05`; ARM B adds `--mod_dropout True --mod_dropout_p 0.3`; all other tokens identical
to the floor command (§4.2). **32f/other knobs, and any re-tuning of `rho`/`p`, are OUT of this prereg** — touching
a knob spends the family and re-costs a bite (§3.6).

---

## 0. Binding facts / honesty clauses (all present; pre-declared)

**F0.1 — Test is NOT virgin (declared).** ZH and HateMM test were already read, under the identical `enc3s`
protocol, by: frozen-CLIP (ZH 13115 / HateMM 12850), frozen-Qwen-8f (12850), generic-LoRA (ZH B3 job 13150;
HateMM job 13235), curriculum-LoRA (ZH+HateMM job 13241), the LoRA-HateMM verdict, and frame16 (13353). This
prereg's head-recipe reads are **re-measurements under the identical protocol**, not first exposures. There are
**four** budgeted head-recipe test evaluations — {ARM A, ARM B} × {ZH, HateMM} — each = the **3 head-seed reads**
of one arm×dataset cell (12 reads total). **Zero test-touch before the independent verdict.**

**F0.2 — The paired test isolates the head-recipe cleanly; single-encoder-draw caveat, and why it does NOT
confound this comparison (pre-declared, material).** Both the treatment and its floor consume the **SAME banked
single-SFT-draw LoRA feature cache** (ZH `…-LoRA_HF.pt`; HateMM `…-LoRA-curric_HF.pt`) — there is **no
re-extraction and no re-SFT** here. So the single-encoder-draw limitation that burdens the LoRA cells is **shared
identically** by treatment and floor and **cannot** confound the head-recipe delta: the ONLY thing that differs at
a given seed is the head-training recipe (SAM optimizer / mod-dropout regularizer). `--seed` controls head-init +
data-shuffle; the pairing is per head-seed (arm seed s − floor seed s), `s ∈ {0,1,2}`.
*Arm-specific RNG note (pre-declared):* **ARM A (SAM) draws NO new RNG** (the ascend is deterministic given the
grads) ⇒ its data-shuffle, head-init and existing-`nn.Dropout` streams match the floor **exactly except the weight
trajectory** — the cleanest possible pairing. **ARM B (mod-dropout) DOES draw `torch.rand` inside forward** (that
IS the regularizer) ⇒ its stochastic training stream diverges from the floor from the first such draw; head-init
and data-order at a given seed are set before/independently of the forward-time draws, so the pairing still
isolates "floor recipe vs +mod-dropout recipe" under matched initialization. This divergence is expected and
confined to the treatment; the flag-OFF path (both floors, and ARM A) draws nothing new.

**F0.3 — Novelty = D7-DEAD, say it plainly (not a user-pending boundary — it is dead).** SAM is a generic
training-time optimizer; modality-dropout is a textbook stochastic regularizer. Both are **novelty-nil / D7-DEAD**
(recon §3(c)/§4(c)/§8): even a formal PASS is a **performance/ablation row** ("head recipe: SAM / mod-dropout on
the align head"), same D7 class as C4 (head-eng) and C5 (recipe) — **never** a novelty contribution. (Contrast the
LoRA/curriculum cells whose D7 is a *user ruling*; these two are dead on arrival, so this is a pure-performance
family — the mandate this round.)

**F0.4 — Single-dataset own-train-split VETO compliance (hard user veto; trivially clean).** No SFT, no data
build, no cross-dataset mixing: each arm's head trains on its own dataset's own-train feature cache only, exactly
as the banked floor did. No gold spans/attributes, no OCR channel, no cross-seed ensemble, raw videos never leave
the machine. All standing vetoes cleared.

**F0.5 — Honest priors are LOW and TWO disclosed headwinds LOWER them; none raise them (pre-declared, material).**
Recon §0/§8: both arms **~8–12%**, revised **down** from the litsurvey ~10–15% for regime-specific reasons the
litsurvey could not see:
- **(a) ARM A headwind — F69 counter-evidence (measured 2026-07-24, the day before this prereg).** On this exact
  tiny retrieval head, the **flatness proxy anti-correlates with test accuracy**: grad-norm↔acc Spearman
  **+0.61 / +0.72 / +0.62, wrong sign 3/3 seeds** (F69). SAM seeks flat minima; F69 is regime-specific evidence
  that *flatter ≠ better here*. It is **NOT a ban** (F69's kill scope was checkpoint *selection* by grad-norm, not
  a training-time optimizer, and SAM's m-sharpness in an ε-ball ≠ point grad-norm — §3.5), but it is a genuine
  headwind, and it is disclosed. Litsurvey headline was ~10–15%; revised to ~8–12%.
- **(b) ARM B headwind — mod-dropout is downside-skewed on text-carried datasets.** Both targets are text-carried
  (F45: ZH gain lives *entirely* in the text stream; F58: HateMM pass is text-carried). A per-sample dropout that
  with prob p/2 removes text and forces the img-only survivor trains the head on the **non-carrying** stream ~15%
  of samples → most likely **hurts** on ZH/HateMM (recon §4(d)). Revised to ~8–12%, **downside-skewed.**
- **(c) HateMM is near-ceiling.** The HateMM floor is 0.8775 (val-sel) / 0.8791 (final) — project-best, thin
  headroom; the +0.030 promote bar sits at ~0.909 everywhere. **ZH is the realistic target** (marginal cell, real
  headroom); HateMM is a hold-the-line leg.
- **(d) Family = one bite, knobs frozen.** SAM `rho=0.05` and mod-dropout `p=0.3` (+ identity-fill) are pinned
  now; **no post-hoc tuning** — if a knob is touched the family is spent and re-costs a bite (§3.6).

**F0.6 — SAM re-mine-reuse invariant (LOAD-BEARING G-repro fact; the one structural risk, HANDLED and asserted).**
The RGCL head mines hard negatives via a FAISS index over the train features, rebuilt **inside `compute_loss`**
and gated **solely** by `retrieval.py:341` (`if train_feats is None or train_labels is None:`). With the deployed
`reindex_every_step=False`, the rebuild fires **once per epoch** (first step) and the returned
`train_feats/train_labels` are reused for the rest of the epoch. SAM's TWO-step requires a **second**
`compute_loss` at the perturbed weights `w+ε`; that second call **MUST reuse the SAME `train_feats/train_labels`
returned by the first call**, so `train_feats is not None` ⇒ **the FAISS index is NOT rebuilt at `w+ε`**
(retrieval.py:341 gate stays False). A re-mine here would build a *different* train index at the perturbed weights
and break byte-consistency with the once-per-epoch baseline semantics. The patch threads the first call's
`train_feats/train_labels` into the second `compute_loss` **and guards the invariant with a runtime `assert`**
(`run_rac.py`, the SAM branch): the arm crashes loudly rather than silently re-mining. The post-training E-step
FAISS rebuild (`run_rac.py:1315-1329`) is on the **EM path only** (`seg_mode consensus/selfscore`), which the
deployed config (`seg_mode full`, else-branch) does **not** take. Risk = LOW, one asserted line of care. **This
invariant is the mandatory focus of the pre-submit codex review (§4.5).**

**F0.7 — Additive-gating / same-code fact (pre-declared).** Every new code path (SAM branch, mod-dropout block,
4 argparse keys) is gated behind `getattr(args, <flag>, <default>)` defaulting OFF. With the flags absent, the
`else:` branch in `model_pass` reproduces the old optimizer block **byte-for-byte** and the classifier forward is
unchanged ⇒ **behaviour is byte-identical to the banked floors, which therefore need NO re-run.** Adding the
argparse keys grows the Namespace by exactly 4 inert keys (`sam=False, sam_rho=0.05, mod_dropout=False,
mod_dropout_p=0.3`) with **zero behaviour diff** when unset — the established additive-flag pattern (cf.
`--tarc_vote_gamma`, `run_rac.py`). The Namespace diff of a treatment run vs its floor = `--model` + the two
derived-inert fields (`group_name`, `exp_comment`) + the arm's ≤2 active flags; nothing else.

**F0.8 — Identity-fill magnitude wrinkle (honest, pre-declared).** Under `batch_norm=False` (deployed), a dropped
sample has `‖x‖ = ‖img⊙1‖ = 1` (unit survivor) while a joint sample has `‖img⊙text‖ < 1`, so identity-fill
injects a magnitude shift the head must absorb — inherent to multiplicative (Hadamard) fusion, and another reason
ARM B's prior is low (recon §4(b)). Recorded for full transparency; it is intrinsic to the mechanism, not a hidden
confound.

---

## 1. Pipeline spec — fully pinned (nothing left to interpretation)

**Deployed head recipe (identical for both floors and both treatment arms):** RGCL `classifier_hateClipper`,
`fusion_mode=align` (Hadamard `x = img ⊙ text`, `classifier.py:140-141` post-patch), triplet+BCE hybrid loss (`--loss triplet --hybrid_loss
True`), AdamW over `model.parameters()` (`run_rac.py`, `lr 1e-4`), 30 epochs, `batch 64`, `grad_clip 0.1`,
`reindex_every_step=False` (FAISS re-mine once per epoch), `warmup 5`, `lambda_seg 0 / seg_mode full` ⇒ single
`model_pass` (else-branch, NOT the EM path). Command byte-verified vs `enc3seed_lora_hatemm.sbatch` (§4.2).

### 1.1 The single stage — 12 head-only runs on cached features (ONE sbatch)

- **Submit:** `sbatch scripts/slurm/headrecipe_family.sbatch` (authored this prereg — artifact C, §5).
- **What it runs:** **2 arms × 2 datasets × 3 seeds = 12 head-only runs** on the banked LoRA caches (~25 s/base
  run; SAM ~40–50 s/run due to the second forward-backward):
  - **ARM A (SAM):** `--sam True --sam_rho 0.05`, exp tag `_..._SAM_rho0.05`, on {`MHC_zh`:`…-LoRA_HF`,
    `HateMM`:`…-LoRA-curric_HF`} × seed{0,1,2}.
  - **ARM B (mod-dropout):** `--mod_dropout True --mod_dropout_p 0.3`, exp tag `_..._MODDROP_p0.3`, same
    dataset×seed grid.
  - `--group_name RAC_video_headrecipe`, `--force False`.
- **Frozen knobs (recon-pin; NO tuning — one bite):** SAM `rho = 0.05` (Foret default); mod-dropout `p = 0.3`,
  **identity-fill (ones) mandatory** under align (zero-fill is rejected — recon §4(b): `img⊙0 = 0` degenerates the
  fused vector; ones-fill passes the survivor through, `img⊙1 = img`).
- **Output:** `slurm/logs/hr_{SAM_rho0.05,MODDROP_p0.3}_{MHC_zh,HateMM}_<MODEL>_seed{0,1,2}_<JID>.trainlog`.
- **GPU budget:** ARM B ≈ base (~25–30 s/run); ARM A ≈ base + one extra head forward-backward/step (FAISS re-mine
  NOT doubled — F0.6) ≈ ~40–50 s/run. Total ≈ 6×45 s + 6×30 s ≈ **~8 min wall, < 0.15 GPU-h**, one A100, serial,
  ONE sbatch.

### 1.2 The patches (3 files; all additive, all getattr-gated)

1. **`src/run_rac.py`** (additive):
   - **+4 argparse keys** (no-op defaults): `--sam` (bool, `False`), `--sam_rho` (float, `0.05`), `--mod_dropout`
     (bool, `False`), `--mod_dropout_p` (float, `0.3`).
   - **Module-level `_sam_ascend(model, rho)` / `_sam_restore(eps_list)`** (Foret two-step, inline; no new pip
     dependency): `_sam_ascend` reads `.grad`, computes the **global** 2-norm over params-with-grad,
     `ε = rho·g/(‖g‖+1e-12)`, `p.add_(ε)` under `torch.no_grad()`, returns `[(p, ε)]`; `_sam_restore` does
     `p.sub_(ε)`. Deterministic (no RNG).
   - **SAM branch in the `model_pass` step loop** (gated `getattr(args,'sam',False)`): the `else:` branch is the
     **byte-identical** old 5-line optimizer block; the SAM branch does `zero_grad → total_loss.backward →
     _sam_ascend → [ASSERT train_feats/labels not None] → second compute_loss REUSING the same train_feats/labels
     (no re-mine, F0.6) → zero_grad → total_loss_perturbed.backward → _sam_restore → clip_grad_norm_ →
     optimizer.step`.
2. **`src/model/classifier.py`** (additive, ~+21 lines net): `__init__` stores `self.mod_dropout` /
   `self.mod_dropout_p` from `args` (default off; `args` is already threaded via `build_model`, so NO run_rac
   wiring is needed beyond the argparse keys). `forward`, **after** L2-normalize / **before** fusion, gated
   `if self.training and getattr(self,'mod_dropout',False) and self.fusion_mode=='align'`: per-sample
   Bernoulli(`p`) `drop` + fair img/text `coin`, `drop_img = drop & coin`, `drop_text = drop & ~coin`
   (**at most one stream per sample**), `img_feats = where(drop_img, ones_like, img_feats)` /
   `text_feats = where(drop_text, ones_like, text_feats)`. Eval path untouched (`self.training` gate).
3. **`src/model/loss.py`** — **NO change** (SAM re-uses `compute_loss` as-is via the threaded `train_feats`).

CPU-verified this prereg: `python -m py_compile` PASS on both edited files; the mod-dropout masking logic
(standalone, n=20000) yields drop-rate **0.2953 ≈ p** with img/text ≈ 0.1525/0.1427 and **BOTH-dropped == 0** (the
at-most-one invariant holds) — this is the mask-rate check pinned as smoke (§4.4.2).

---

## 2. Comparison floors — INDEPENDENTLY RE-DERIVED from raw trainlogs (numeric-provenance discipline)

Every number below was independently re-parsed **this prereg** from the raw trainlogs with the EXACT
`enc3seed_lora_hatemm.sbatch` embedded parser (val-sel = epoch ≥ warmup 5 max `Val_Retrieval` acc, roc tie-break;
final = max epoch). Both floor means bit-match `HEADRECIPE_FORENSIC_RECON.md §1` to 4dp — no discrepancy.

### 2.1 ZH floor — job **13150** (`Qwen2.5-VL-7B-Instruct-LoRA_HF`, generic-LoRA / B3; goal-relevant, marginal)

| seed | val-sel ep | val-sel acc/mF1 | final ep | final acc/mF1 | provenance (Test line) |
|---|---|---|---|---|---|
| 0 | 20 | 0.8322 / 0.8023 | 29 | 0.8456 / 0.8181 | seed0 log :220 (val) / :302 (final) |
| 1 | 26 | 0.8255 / 0.7956 | 29 | 0.8389 / 0.8113 | seed1 log :275 / :303 |
| 2 | 19 | 0.8389 / 0.8065 | 29 | 0.8523 / 0.8226 | seed2 log :207 / :298 |
| **mean** | | **0.8322 / 0.8015** | | **0.8456 / 0.8173** | |

Files: `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`. Bit-matches
`CAND2_CURRICULUM_PREREG.md §2.1` (generic-LoRA/B3) and `B3_PREREG_REVIEW.md`.

### 2.2 HateMM floor — job **13241** (`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`, curriculum-LoRA; near-ceiling)

| seed | val-sel ep | val-sel acc/mF1 | final ep | final acc/mF1 | provenance (Test line) |
|---|---|---|---|---|---|
| 0 | 29 | 0.8791 / 0.8730 | 29 | 0.8791 / 0.8730 | seed0 log :331 (val=final) |
| 1 | 14 | 0.8744 / 0.8678 | 29 | 0.8791 / 0.8724 | seed1 log :178 (val) / :329 (final) |
| 2 | 10 | 0.8791 / 0.8724 | 29 | 0.8791 / 0.8724 | seed2 log :140 (val) / :331 (final) |
| **mean** | | **0.8775 / 0.8711** | | **0.8791 / 0.8726** | |

Files: `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`. This is the
project-best HateMM cell (cand-2 curriculum, `goal-round3-terminus` memory).

### 2.3 Concrete promote thresholds + noise band

- **ZH promote (+0.030 per seed):** val-sel acc ≥ {0.8622, 0.8555, 0.8689}; final acc ≥ {0.8756, 0.8689, 0.8823}.
- **HateMM promote (+0.030 per seed):** ≥ ~0.909 everywhere (near-ceiling; thin surface — F0.5(c)).
- **Head-seed noise band (for KS-regression, §3.4):** ±**0.014** — the established house head-seed-spread
  descriptor (`CAND2_CURRICULUM_PREREG.md §2.3`, largest observed generic-arm between-seed acc spread 0.0140);
  the local floors here are within it (ZH val-sel/final spread 0.0134; HateMM ≤ 0.0047). A 3-seed **mean** move
  beyond ±0.014 is beyond the full head-seed spread.

---

## 3. Decision rule + kill-bars (paired, both protocols judged independently, 3/3 sign, pre-declared)

### 3.1 Decision rule — verbatim from `exp-encoder-3seed.md:73-85` (treatment = arm; control = the arm's banked floor)

> For each dataset × protocol: (1) per-seed paired difference δ = (treatment − control) for acc and macro-F1 at
> seeds 0/1/2; (2) 3-seed mean ± std + sign consistency (how many of 3 positive); (3) n=3 too small for a
> bootstrap — report the paired-t **as an effect-size descriptor only**, no significance claim; (4) **pass =
> mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive**; (5) headline claim
> requires pass on ≥ 2 datasets under a stated protocol; both protocols judged separately; verdict written
> exactly "final-epoch: pass/fail; val-selected: pass/fail".

Both protocols judged **independently** (no protocol-shopping, no metric-shopping). Control = the arm's OWN
banked floor (ZH §2.1, HateMM §2.2). Judged **per arm × per dataset**; the family verdict is per-arm (§3.6).

### 3.2 FORMAL promote bar (goal-facing; per arm × dataset)

House **+0.030 acc AND +0.030 mF1** conjunct, **3/3 seeds positive**, under **BOTH** protocols vs the banked floor
(§2). Below the conjunct under a protocol → **NEGATIVE** on that protocol. **D7-DEAD (F0.3): even a formal PASS is
an engineering/ablation row, NEVER a novelty win.**

### 3.3 KS-arm-dead — the KILL bar (SIGN-based; per arm × dataset)

Per the **frame16 DEV-1 sign discipline** (house n=3 = **no bootstrap**; the kill uses SIGN, not a CI-straddles-0
test): an arm×dataset cell is **KILLED** iff, on **BOTH protocols**, `mean paired Δacc ≤ 0` **OR** the acc sign is
not 3/3 positive — i.e. **neither** protocol produces a clean positive-mean-and-3/3-sign result (a tie-or-regress
on both protocols = no net improvement over the floor). This is the sign-based analog of the recon's "≤ floor on
both protocols." At verdict time, state each killed cell explicitly.

### 3.4 KS-regression — BELOW-FLOOR-BY-SPREAD note (per arm × dataset)

If arm − floor **mean Δacc ≤ −0.014** on a leg (below the full head-seed spread, §2.3), the recipe **degraded**
the head → bank "SAM / mod-dropout hurts on <dataset>." (This is the pre-declared expected direction for ARM B on
text-carried ZH/HateMM — F0.5(b).) A note within the KS-arm-dead frame, not a separate multiplicity bite.

### 3.5 Ban-collision closure (carried from recon §3/§4; the two headwinds disclosed, NOT bans)

- **SWA F62/F62b (KILLED):** scope = weight **AVERAGING** across epochs. SAM is a **single-trajectory training
  optimizer** — no averaging. **NOT covered.**
- **F69 (KILLED 2026-07-24):** scope = checkpoint **SELECTION** by grad-norm. SAM **selects nothing**; it changes
  the optimizer. **NOT a ban** — but F69 is the sharpest *measured* headwind for SAM's premise on this head
  (grad-norm↔acc wrong sign 3/3), disclosed in F0.5(a).
- **F50 (FA fusion/composition gate, KILLED):** scope = fixed **inference-time** reweightings over **frozen**
  features. Modality-dropout is a **training-time stochastic regularizer** shaping the head's learned weights, not
  a fixed inference-time combiner (the align op itself is unchanged). **NOT covered.**
- **cross-seed ensemble / OCR / gold-in-method / single-dataset-split / P1–P5 / external-API / target-as-structure**
  — none reach a training-time head recipe. **Clear.** Both arms are IN-BOX (head training over cached embeddings;
  no gold/ensemble/encoder-touch); both D7-DEAD (§3.2 / F0.3).

### 3.6 Multiplicity + scope of THIS submit (pre-declared)

- **ONE sbatch = ONE pre-registered family = ONE multiplicity bite** whether one or both arms survive. The two
  arms **share** the single "F68 wave-2 head-recipe" bite.
- **Hyperparameters FROZEN** (SAM `rho=0.05`; mod-dropout `p=0.3` + identity-fill). **NO post-hoc knob tuning** —
  a `{rho}` or `{p}` sweep, zero-fill, a second `p`, or any knob touch is a **new** pre-declared arm and re-costs
  a bite. `error`-mode / concat-fusion variants are OUT.
- **Family verdict is per-arm × per-dataset** (each judged only vs its own floor). A surviving arm×dataset cell
  still owes the **full ceremony** (this prereg → independent 0-context review → freeze-hash → SLURM); this prereg
  does **not** discharge that, and this family is the ONLY head-recipe bite.

### 3.7 Gate order

G-repro (patched-file sha re-verify + no-flag Namespace-diff + additive-gating proof, §4.1) → codex review of the
SAM branch (§4.5) → smoke (§4.4) → single test-touch (the 12 head reads) → per arm×dataset: KS-arm-dead → FORMAL
promote bar (both protocols). The verdict is rendered by an **independent 0-context reviewer against this prereg
VERBATIM**; the executor transcribes raw both-protocol per-seed numbers (line-numbered) and applies NO
gates/interpretation.

---

## 4. G-repro + smoke plan + collision safety + codex gate

### 4.1 G-repro discipline

- **(a) Patched-file sha gate.** At submit time re-run `sha256sum` on `src/run_rac.py`, `src/model/classifier.py`,
  `scripts/slurm/headrecipe_family.sbatch` (and this file) — must match the §5 freeze block; any mismatch =
  authorization VOID. `src/model/loss.py` sha `4879663…` (unchanged) re-verified (SAM touches no loss code).
- **(b) Additive-gating / no-flag byte-identity proof (F0.7).** A run with the flags OFF must be byte-identical in
  Namespace (modulo the 4 inert new keys + derived-inert `model`/`group_name`/`exp_comment`) to the banked floor
  command. The `else:` branch of the SAM edit reproduces the old optimizer block byte-for-byte (git-diff verified:
  the only "deleted" lines are that block, re-emitted verbatim under `else:`; the classifier "deletions" are two
  whitespace-only lines). Optional stronger check: a 1-seed **no-flag** head run on the ZH LoRA cache bit-matches
  the banked 13150 seed0 trajectory (classic G-repro bit-exact, cf. `exp-encoder-3seed.md:126-146`).
- **(c) Same-code (INCLUDING the floors).** The `python ./src/run_rac.py …` block of `headrecipe_family.sbatch` is
  token-identical to `enc3seed_lora_hatemm.sbatch` **except** the two intended deltas: `--exp_comment
  "_${MODEL}_${ARM}"` (derived-inert) and the trailing `${ARM_FLAGS}` (additive-gated). The readout `PY` block is
  **BYTE-IDENTICAL** (`diff` empty). Both verified this prereg (§4.2).

### 4.2 Same-code + syntax verification (run this prereg — PASS)

- `python ./src/run_rac.py` invocation diff vs `enc3seed_lora_hatemm.sbatch`: exactly 2 lines
  (`--exp_comment "_${MODEL}"` → `"_${MODEL}_${ARM}"`; trailing `` `` → `${ARM_FLAGS}` before `2>&1`). Nothing else.
- Readout `WARMUP=… python - …PY` block vs `enc3seed_lora_hatemm.sbatch`: **BYTE-IDENTICAL**.
- `bash -n scripts/slurm/headrecipe_family.sbatch` = **SYNTAX_OK**; CONFIGS word-split dry-run = 12 rows, each
  `run_one` receiving `(DATASET, MODEL, SEED, ARM, ARM_FLAGS…)` correctly (trailing remainder captured).
- `python -m py_compile src/run_rac.py src/model/classifier.py` = **PASS**.

### 4.3 Collision safety (verified this prereg — ABSENT; re-check at submit)

- `scripts/slurm/headrecipe_family.sbatch`, `refine-logs/HEADRECIPE_PREREG.md` — created by this prereg (no prior).
- `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_headrecipe*` — do NOT exist ⇒ fresh group; `--force False` never
  trips the `run_rac.py` hard-abort; the `RAC_video_headrecipe` group + per-arm `exp_comment` (`_..._SAM_rho0.05` /
  `_..._MODDROP_p0.3`) keep dirs distinct from every banked arm.
- `slurm/logs/hr_*.trainlog` — do NOT exist ⇒ no trainlog collision. The `hr_${ARM}_…` prefix + arm tag guarantees
  ARM A and ARM B (same dataset/model/seed) never collide.
- Banked caches (ZH `…-LoRA_HF.pt`, HateMM `…-LoRA-curric_HF.pt`) and floor trainlogs (13150 / 13241) are
  **read-only inputs**; this family writes none of them (distinct group, no extraction, no SFT).
- Smoke throwaways (`_smoke_hr` group / `hr_smoke_*` logs) — deleted after smoke; must NOT persist into §4.3.

### 4.4 Smoke plan (executor runs BEFORE the real submit; leave no artifact that trips §4.3)

1. **ARM A (SAM) smoke (GPU, ~1 min):** 1-seed (seed 0) **3-epoch throwaway** on the ZH LoRA cache with
   `--sam True --sam_rho 0.05 --epochs 3 --group_name _smoke_hr` → confirm: (i) train loss **finite** (no NaN)
   across the 3 epochs; (ii) run **completes** (FAISS re-mine fires once/epoch, no crash, the re-mine-reuse assert
   does NOT trip); (iii) the **SAM double-step is visible** — per-step / per-epoch wall-time is meaningfully above
   the flag-off baseline (the second forward-backward; grep the tqdm timing). Then delete
   `logging/Retrieval/*/…_smoke_hr*` + the smoke trainlog.
2. **ARM B (mod-dropout) smoke (GPU, ~1 min + $0 CPU):** (a) 1-seed 3-epoch throwaway on the ZH cache with
   `--mod_dropout True --mod_dropout_p 0.3 --epochs 3 --group_name _smoke_hr` → loss finite, completes; delete
   artifacts. (b) **$0 CPU mask-rate line** (pinned; already run this prereg): construct the masking on a synthetic
   batch (n large) with `p=0.3` and print the observed **drop-rate (≈0.30), img/text split (≈0.15/0.15), and
   BOTH-dropped count (must be 0)** in `training` mode, and confirm the `self.training` gate yields NO fill in eval
   mode. Reference result this prereg: `drop 0.2953 / img 0.1525 / text 0.1427 / both 0`.
3. **No-flag Namespace / byte-identity proof (§4.1b):** run the base command with NO arm flags (or
   `headrecipe_family.sbatch` with `ARM_FLAGS=""`), dump `vars(args)`, confirm it differs from the banked floor
   Namespace ONLY by the 4 inert new keys + `model`/`group_name`/`exp_comment`. If in doubt, the optional 1-seed
   bit-exact floor reproduction (§4.1b) settles it.

### 4.5 CODEX GATE (mandatory, pre-submit — house `codex-code-review` pattern)

Before ANY SLURM submission, the executor **MUST** run a codex review (iterative loop until Claude + Codex agree)
focused on the **SAM double-step + re-mine-reuse** branch: `_sam_ascend` / `_sam_restore` (global grad-norm,
`ε` scale, in-place add/sub under `torch.no_grad()`, exact restore), the sam-gated block ordering (backward at w →
ascend → second `compute_loss` at w+ε → backward → restore → clip → step), and **critically** the F0.6 invariant
(the second `compute_loss` reuses the first call's `train_feats/train_labels`; the assert guards it; no second
FAISS re-mine). **Blocking findings ⇒ fix the code + re-freeze the shas (§5) + re-run this gate.** The mod-dropout
block (identity-fill, at-most-one-stream, `self.training` gate) should also be included but is lower-risk.

---

## 5. Artifacts authored this prereg + hash-freeze block

### 5.1 New / edited artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current) |
|---|---|---|---|
| P | `refine-logs/HEADRECIPE_PREREG.md` | **NEW** — this file | *(reviewer fills at freeze)* |
| A | `src/run_rac.py` | **EDITED (additive)** — +4 argparse keys; `_sam_ascend`/`_sam_restore`; SAM-gated branch in `model_pass` (else-branch byte-identical to old block) | `1012c9e378905e5c10a0447475560de4a32904af691e457bf4ce77a3d36cc20d` |
| B | `src/model/classifier.py` | **EDITED (additive)** — store `mod_dropout`/`mod_dropout_p` in `__init__`; identity-fill per-sample mask in `forward`, gated `training ∧ mod_dropout ∧ align` | `e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378` |
| C | `scripts/slurm/headrecipe_family.sbatch` | **NEW** — clone of `enc3seed_lora_hatemm.sbatch`; `run_one` python block token-identical + `${ARM_FLAGS}`; readout `PY` byte-identical; `RAC_video_headrecipe`; 12 rows (2 arms × 2 ds × 3 seeds) | `c88f685f68f83611fde3f91751f330d30b6be278693a405f4b9fb80f53ebb009` |

### 5.2 Reused-unchanged machinery (verify sha at submit time; do NOT edit)

| path | role | sha256 |
|---|---|---|
| `src/model/loss.py` | `compute_loss` (SAM re-uses as-is; **NO edit**) | `48796638fdd60fcfb313e97e7f89d73226d96f23369f8c8ebb61ca5814f9cd64` |
| `src/utils/retrieval.py` | FAISS re-mine gate (`:341`, the F0.6 invariant; **NO edit**) | `d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57` |
| `scripts/slurm/enc3seed_lora_hatemm.sbatch` | same-code anchor for §4.2 (produced neither floor, but the byte source of `run_one`) | `19c76b177f7dc883a9e03524450ad2e6cb302cdd0a6704d69da68a62188a06fc` |
| `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | ZH floor cache (paired input; NOT clobbered) | train `b2e8e78…`, dev `4c07af7…`, test `4e107bf…` |
| `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` | HateMM floor cache (paired input; NOT clobbered) | train `5e80f39…`, dev `46ee4fd…`, test `b50ae4e…` |

### 5.3 Hash-freeze (to be filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file HEADRECIPE_PREREG.md, after review>
A 1012c9e378905e5c10a0447475560de4a32904af691e457bf4ce77a3d36cc20d  src/run_rac.py
B e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378  src/model/classifier.py
C c88f685f68f83611fde3f91751f330d30b6be278693a405f4b9fb80f53ebb009  scripts/slurm/headrecipe_family.sbatch
```
Executor re-runs `sha256sum` on A/B/C (and this file) + confirms `loss.py 4879663…` / `retrieval.py d43e3bc…`
unchanged at submit time; any mismatch = authorization VOID. **If the codex gate (§4.5) forces a code fix, A/B/C
shas change and the freeze block MUST be re-issued.**

---

## 6. Single-submit / execution plan + resource plan

**Order (ONE SLURM job):**

1. Pre-submit: G-repro (§4.1) → **codex gate (§4.5)** → smoke (§4.4). Only on all-clear:
2. `sbatch scripts/slurm/headrecipe_family.sbatch` → 12 head runs sequential inside (2 arms × 2 ds × 3 seeds),
   ~8 min wall, < 0.15 GPU-h. Produces `slurm/logs/hr_{SAM_rho0.05,MODDROP_p0.3}_{MHC_zh,HateMM}_<MODEL>_seed{0,1,2}_<JID>.trainlog`.

**Resource plan (STANDING INFRA RULE compliant):** the sbatch requests **`--cpus-per-task=8`, `--mem=64G`,
1×A100** (inherited from `enc3seed_lora_hatemm.sbatch`; verified). Single job ⇒ peak footprint **8 CPU / 64 G /
1 GPU** — well within the 16 CPU / 128 G / 2 GPU cap, and **NEVER two 16-CPU jobs in flight** (the 29 h-wedge
infra rule). `conda activate HateVideo`; `sbatch` with **NO `--time`**; initial `PENDING (JobHeldUser)` = **WAIT
for auto-release, never force** (CLAUDE.md). Sources `conda.sh` directly and runs the ≥20 G `disk_guard.sh`.

**Test-touch:** the 12 head reads are the ONLY budgeted head-recipe test evaluations (4 arm×dataset cells × 3
seeds); zero test-touch before the verdict. **The executor transcribes raw both-protocol per-seed numbers
(line-numbered) and applies NO gates/interpretation** — the verdict (G-repro → codex → smoke → KS-arm-dead →
FORMAL bar, per arm×dataset) is rendered by an **independent 0-context reviewer against this prereg VERBATIM.**

**No job is submitted by this prereg author.** Submission happens only after the independent 0-context review +
hash-freeze (+ codex gate) run by the orchestrator/executor.

---

## 7. Outcome table template (filled ONLY from raw trainlogs at verdict time)

### 7.1 ARM A — SAM vs floor (fill from `hr_SAM_rho0.05_{MHC_zh,HateMM}_…_seed{0,1,2}_<JID>.trainlog`)

| dataset | seed | protocol | SAM acc/F1 | floor acc/F1 (§2) | Δ(SAM−floor) acc/F1 |
|---|---|---|---|---|---|
| MHC_zh | 0 | val-sel | ___ | 0.8322/0.8023 | ___ |
| MHC_zh | 1 | val-sel | ___ | 0.8255/0.7956 | ___ |
| MHC_zh | 2 | val-sel | ___ | 0.8389/0.8065 | ___ |
| MHC_zh | **mean** | **val-sel** | ___ | **0.8322/0.8015** | **___** |
| MHC_zh | 0 | final-ep | ___ | 0.8456/0.8181 | ___ |
| MHC_zh | 1 | final-ep | ___ | 0.8389/0.8113 | ___ |
| MHC_zh | 2 | final-ep | ___ | 0.8523/0.8226 | ___ |
| MHC_zh | **mean** | **final-ep** | ___ | **0.8456/0.8173** | **___** |
| HateMM | 0 | val-sel | ___ | 0.8791/0.8730 | ___ |
| HateMM | 1 | val-sel | ___ | 0.8744/0.8678 | ___ |
| HateMM | 2 | val-sel | ___ | 0.8791/0.8724 | ___ |
| HateMM | **mean** | **val-sel** | ___ | **0.8775/0.8711** | **___** |
| HateMM | 0 | final-ep | ___ | 0.8791/0.8730 | ___ |
| HateMM | 1 | final-ep | ___ | 0.8791/0.8724 | ___ |
| HateMM | 2 | final-ep | ___ | 0.8791/0.8724 | ___ |
| HateMM | **mean** | **final-ep** | ___ | **0.8791/0.8726** | **___** |

### 7.2 ARM B — modality-dropout vs floor (fill from `hr_MODDROP_p0.3_…` trainlogs)

*(same 16-row structure as §7.1, ARM B acc/F1 vs the identical §2 floors.)*

### 7.3 Fixed write-up format (per §3.1 rule 5 + the bars §3.2/§3.3)

```
ARM A (SAM):        MHC_zh:  final-epoch: <pass/fail>; val-selected: <pass/fail>  [FORMAL §3.2]. KS-arm-dead: <KILLED | survives>.
                    HateMM:  final-epoch: <pass/fail>; val-selected: <pass/fail>. KS-arm-dead: <KILLED | survives>.
ARM B (mod-drop):   MHC_zh:  final-epoch: <pass/fail>; val-selected: <pass/fail>. KS-arm-dead: <KILLED | survives>.
                    HateMM:  final-epoch: <pass/fail>; val-selected: <pass/fail>. KS-arm-dead: <KILLED | survives>.
(+ KS-regression note if any Δacc ≤ −0.014; + MARGINAL note if a within-noise pass per B3 §2.2 precedent.)
```

---

## 8. What a PASS / FAIL means for the goal (D7 boundary — recipes are DEAD, not user-pending)

- **KS-arm-dead on all four cells (recon prior — the honest expected outcome, ~8–12% each ⇒ likely ≥1 survivor is
  low):** the two remaining head-training-dynamics escape hatches (SAM flat-minima, mod-dropout rebalancing) carry
  no net signal on ZH/HateMM ⇒ both are **CLOSED** at < 0.15 GPU-h. The cleanest, cheapest outcome: two
  prose-argued gaps converted to measured door-closers in one bite.
- **A cell survives KS but < FORMAL bar:** measured-not-promoted limbo (bank the weak positive; still D7-DEAD).
- **A cell clears the FORMAL bar (≥ +0.030/+0.030, 3/3, both protocols):** a paper-worthy **robustness/ablation**
  row ("head recipe helps on <dataset>"), most plausibly the **marginal ZH** cell (F0.5). **NOT a novelty win**
  (F0.3): SAM and mod-dropout are generic training knobs. A surviving cell still owes the full ceremony (§3.6).

**Framing sentence (verbatim):** *this measurement tests two cheap head-training recipes — SAM (flat-minima
optimizer) and modality-dropout (identity-fill stream-dropout regularizer) — on the deployed align-fusion head over
cached features, 3-seed paired dual-protocol vs each dataset's banked floor; it is a one-bite, knobs-frozen closure
of the last two head-training-dynamics escape hatches, and a pass is a performance/ablation row, NEVER a novelty
win — both recipes are D7-DEAD.*

---

## 9. Provenance index

- Recon (GO; 2-arm family design, pinned knobs, identity-fill ruling, ban rulings, kill skeleton):
  `refine-logs/HEADRECIPE_FORENSIC_RECON.md` (`44918e0`).
- Cell source: `LITSURVEY_NOVEL_MECHANISMS.md` top-5 #3 (SAM) + #5 (modality-dropout); F69 headwind
  (`state/findings.jsonl`, 2026-07-24).
- Deployed head + fusion: `src/model/classifier.py:115-149` (forward; align/Hadamard `torch.mul` at `:141`),
  `:71-113` (`__init__`; mod-dropout storage at `:77`), mod-dropout block at `:129-137` (all post-patch).
- SAM re-mine invariant (F0.6): `src/model/loss.py:245-285` (`compute_loss` → `dense_retrieve_…`);
  `src/utils/retrieval.py:341` (the `train_feats is None` re-mine gate); `run_rac.py` `model_pass` step loop
  (SAM branch + else-branch).
- Floors (re-derived §2): ZH `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`
  (`B3_PREREG_REVIEW.md`, `CAND2_CURRICULUM_PREREG.md §2.1`); HateMM
  `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog` (`goal-round3-terminus`).
- Protocol + decision rule (verbatim): `research-wiki/experiments/exp-encoder-3seed.md:73-85`.
- Same-code anchor: `scripts/slurm/enc3seed_lora_hatemm.sbatch` (sha `19c76b1…`).
- Mechanism headwinds: F45 (ZH text-stream Pareto), F58 (HateMM text-carried), F69 (grad-norm↔acc wrong sign).

**Required statements:** ZERO GPU/SLURM/Modal spent by this prereg author (only pure-CPU login-node floor
re-parsing, `py_compile`, a synthetic-tensor mask-rate check, and collision/syntax/same-code verification, seconds;
no held-out test metric produced). All floor numbers re-parsed from banked completed-run trainlogs
(numeric-provenance discipline). No `state/` mutated. No `research-wiki/` mutated. NO job submitted. Not pushed.

---

## 10. (reserved)

---

## 11. DEVIATIONS FROM THE RECON — flagged loudly

1. **DEV-1 (`run_one` python block is NOT byte-identical to `enc3seed` — it adds a trailing `${ARM_FLAGS}` +
   arm-tagged `exp_comment`/trainlog). MATERIAL / inherent to a flagged family.** Unlike frame16/cand-2 (which
   varied only `--model`/`--group_name`, both pre-existing template fields), this family MUST inject per-arm flags
   (`--sam …` / `--mod_dropout …`), so the run_one command cannot be byte-identical. Resolution: the base command
   is **token-identical** to `enc3seed_lora_hatemm.sbatch` (verified §4.2) with exactly two intended deltas —
   `--exp_comment "_${MODEL}_${ARM}"` (derived-inert, the blessed class) and the trailing `${ARM_FLAGS}`
   (additive-gated). The readout `PY` block is byte-identical. The same-code guarantee is preserved at the
   token/Namespace level (F0.7): flags-off ⇒ byte-identical behaviour; each arm adds only its ≤2 inert-unless-set
   flags. The arm tag is **required** in the trainlog name + `exp_comment` so ARM A and ARM B (same dataset/model/
   seed) never collide (§4.3).

2. **DEV-2 (KILL bar uses SIGN, not "bootstrap CI straddles 0"). MATERIAL — house discipline, per frame16 DEV-1.**
   The recon phrases KS as "≤ floor on both protocols." A CI-straddles-0 formalism conflicts with the house n=3
   **no-bootstrap** rule (`exp-encoder-3seed.md:78-79`). I pin the **sign-based** KS-arm-dead (§3.3): killed iff on
   BOTH protocols `mean Δacc ≤ 0` OR acc sign not 3/3 positive. Only the significance formalism changes; the
   qualitative bar (tie/regress on both protocols ⇒ dead) is identical to the recon.

3. **DEV-3 (patches EDIT `run_rac.py` + `classifier.py` in place — not new files). MATERIAL / recon-mandated,
   same-code preserved.** Unlike frame16/cand-2 (new sbatch, existing code untouched), the recon's design requires
   additive edits to two tracked source files. Resolution: every edit is `getattr`-gated OFF-by-default; the SAM
   `else:` branch reproduces the old optimizer block byte-for-byte; the classifier "deletions" are two
   whitespace-only lines (git-diff verified) ⇒ flags-off behaviour is byte-identical and the floors need no re-run
   (F0.7). The patched-file shas are hash-frozen (§5) and re-verified at submit.

4. **DEV-4 (mod-dropout wiring needs NO run_rac.py build_model change). Neutral / favorable.** The recon §7 patch
   summary item 1 lists "pass mod-dropout args into build_model." In fact `build_model` **already** passes
   `args=args` into `classifier_hateClipper` (`run_rac.py:1117-1120`), so the classifier reads `mod_dropout`/
   `mod_dropout_p` directly from `args` in `__init__` — no run_rac wiring beyond the 4 argparse keys. Smaller diff,
   same effect.

5. **DEV-5 (SAM invariant enforced by a runtime `assert`, not only code-structure). Favorable.** The recon offered
   "a runtime assert OR a code-structure guarantee" for the re-mine-reuse invariant (F0.6). I pin **both**: the
   structure threads the first call's `train_feats/train_labels` into the second `compute_loss`, AND a runtime
   `assert train_feats is not None and train_labels is not None` in the SAM branch converts any config where mining
   did not happen into a loud failure rather than a silent re-mine at `w+ε`. Stronger reproducibility guarantee.

6. **DEV-6 (mask-rate line delivered as a $0 CPU synthetic-tensor check, not a permanent hot-path log). Favorable.**
   To avoid polluting trainlogs / the same-code surface with a permanent per-step print, the mod-dropout "mask-rate
   line" smoke (§4.4.2b) is a standalone synthetic-batch check (drop-rate / img-text split / both-dropped==0 /
   eval-gate-off), already run this prereg (0.2953 / 0.1525 / 0.1427 / 0). The classifier hot path stays clean.

7. **DEV-7 (both arms + both datasets in ONE bite, per the task; HateMM floor = curriculum-LoRA 13241, ZH floor =
   generic-LoRA 13150). Documented.** The task pins the HateMM comparison against the **curriculum** LoRA (project-
   best 13241) and ZH against the **generic** LoRA (13150) — each dataset's deployed/best cache. Both arms run on
   both datasets under the single family bite (§3.6); no arm is a new-dataset bet (HateMM is near-ceiling hold, ZH
   is the marginal target — F0.5).
