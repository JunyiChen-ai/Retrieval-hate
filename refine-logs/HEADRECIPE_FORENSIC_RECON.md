# HEAD-RECIPE FORENSIC RECON — SAM + modality-dropout (F68 wave-2)

**Author:** headrecipe-forensic-recon (ZERO GPU / CPU-only; banked logs + code read).
**Date:** 2026-07-25 NZST.
**Mission:** cheap head-level training changes that run entirely on CACHED features (GPU minutes),
sourced from `LITSURVEY_NOVEL_MECHANISMS.md` top-5 **#3 SAM flat-minima optimizer** and
**#5 modality-dropout regularizer**. Pure-performance mandate this round (D7-novelty irrelevant).
Produce GO/NO-GO + execution skeleton treating **BOTH arms as ONE pre-registered family (one bite,
multiplicity-safe)**. NO job submission, NO prereg, `state/` untouched.

---

## 0. VERDICT UP FRONT

**GO — as a single 2-arm family, with two hyperparameters and one fill-rule pre-pinned now.**
The whole family costs **< 0.15 GPU-h in one sbatch**, so the cheap-kill *is* the measurement — no
$0 pre-gate is warranted. Both arms are legal (no ban collision) and both are genuinely cheap.

Honest priors are **below** the litsurvey headline for *this* regime, for reasons the litsurvey could
not see (they postdate it or emerge from the deployed fusion):

- **ARM A (SAM):** litsurvey ~10–15% → **revise DOWN to ~8–12%.** F69 (measured *yesterday*, 2026-07-24)
  found the flatness proxy **anti-correlates** with test accuracy on this exact tiny retrieval head
  (grad-norm↔acc Spearman **+0.61/+0.72/+0.62, wrong sign 3/3 seeds**). SAM seeks flat minima; F69 is
  regime-specific counter-evidence that flatness ≠ generalization *here*. Not a ban, but a real headwind.
- **ARM B (modality-dropout):** litsurvey ~10–15% two-sided → **revise DOWN to ~8–12%, downside-skewed.**
  Two independent reasons: (i) the deployed fusion is **`align` = Hadamard (element-wise multiply)**, on
  which naive zero-fill **degenerates the whole fused vector to bias-only** — modality-dropout is not even
  well-defined without a fill-rule decision (I pre-pin **identity/ones-fill**, ruled below); (ii) both
  target datasets are **text-carried** (F45: ZH gain lives *entirely* in the text stream; F58: HateMM pass
  is text-carried) — dropping the carrying stream trains on a stream that cannot carry → likely **hurts**.

Neither arm collides with any kill or banned constraint (rulings in §3–§4). Run both; expect at most one
marginal ZH hardening if anything survives.

---

## 1. DEPLOYED TARGET + RE-DERIVED FLOORS (banked, no re-run)

**Head recipe (identical for both floors and both treatment arms):** RGCL `classifier_hateClipper`,
`fusion_mode=align` (Hadamard, `classifier.py:110-122`), triplet + BCE hybrid loss
(`--loss triplet --hybrid_loss True`, `ce_weight` default 0.5, `loss.py:545-554`), AdamW over
`model.parameters()` (`run_rac.py:558`, **lr 1e-4, no weight_decay passed**), 30 epochs, batch 64,
`grad_clip 0.1`, `reindex_every_step=False` (FAISS re-mining **once per epoch**), `warmup 5`,
`seg_mode full / lambda_seg 0` ⇒ single `model_pass` (else-branch `run_rac.py:1333`, **not** the EM path).
Command verified byte-identical across `enc3seed*.sbatch`.

### ZH floor — job **13150**, model `Qwen2.5-VL-7B-Instruct-LoRA_HF`, group `RAC_video_b3_lora`
(goal-relevant: val-sel FAIL, needs +3; final marginal)

| seed | val-sel ep | val-sel acc | val-sel mF1 | final ep | final acc | final mF1 |
|---|---|---|---|---|---|---|
| 0 | 20 | 0.8322 | 0.8023 | 29 | 0.8456 | 0.8181 |
| 1 | 26 | 0.8255 | 0.7956 | 29 | 0.8389 | 0.8113 |
| 2 | 19 | 0.8389 | 0.8065 | 29 | 0.8523 | 0.8226 |
| **mean** | | **0.8322** | **0.8015** | | **0.8456** | **0.8173** |

### HateMM floor — job **13241**, model `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`, group `RAC_video_lora_curric`
(near ceiling; little headroom)

| seed | val-sel ep | val-sel acc | val-sel mF1 | final ep | final acc | final mF1 |
|---|---|---|---|---|---|---|
| 0 | 29 | 0.8791 | 0.8730 | 29 | 0.8791 | 0.8730 |
| 1 | 14 | 0.8744 | 0.8678 | 29 | 0.8791 | 0.8724 |
| 2 | 10 | 0.8791 | 0.8724 | 29 | 0.8791 | 0.8724 |
| **mean** | | **0.8775** | **0.8711** | | **0.8791** | **0.8726** |

**House promote bar** (per-arm, per-dataset): **+0.030 acc AND +0.030 mF1, 3/3 seeds, on BOTH protocols**
(val-selected and final-epoch). Concretely a promote needs e.g. ZH val-sel ≥ {0.8622, 0.8555, 0.8689}
per seed; HateMM ≥ ~0.909 everywhere (near-ceiling — HateMM headroom is thin, ZH is the realistic target).

**Kill-switch (KS):** an arm is KILLED if it lands **≤ floor on both protocols** (no net improvement).
The band between KS and the +0.030 promote bar is "measured, not promoted."

---

## 2. FAMILY DESIGN (one bite, multiplicity-safe)

- **2 arms × 2 datasets × 3 seeds = 12 runs**, all on cached LoRA features (~25 s/base run).
- **ONE sbatch, ONE pre-registered family = ONE multiplicity bite.** Hyperparameters frozen at recon-pin
  (SAM `rho=0.05`; mod-dropout `p=0.3`, identity-fill) — **no post-hoc knob tuning**; if a knob is touched
  the family is spent and re-costs a bite.
- Each arm judged **only** vs its own banked floor (ZH 13150, HateMM 13241) under both protocols.
- The two arms **share** the family's multiplicity budget: this recon spends the single "F68 wave-2
  head-recipe" bite whether one or both arms are tested. Any surviving arm then owes the full ceremony
  (prereg → independent review → freeze-hash → SLURM) — this recon does **not** discharge that.
- **GPU budget:** mod-dropout ≈ base (~25–30 s/run); SAM ≈ base + extra head forward-backward per step
  (FAISS re-mine NOT doubled — see §5) ≈ ~40–50 s/run. Total ≈ 6×45 s + 6×30 s ≈ **~8 min wall,
  < 0.15 GPU-h**, one A100, serial, one sbatch.

---

## 3. ARM A — SAM (sharpness-aware minimization) on the head

**(a) Mechanism.** Standard two-step SAM (Foret et al., ICLR 2021) wrapping the existing AdamW: at each
step compute grad g at w, ascend to `w + ε` with `ε = rho·g/‖g‖`, recompute grad at `w+ε`, then apply the
AdamW update at the original w using the perturbed-point grad. Steers the tiny head to a wider minimum.
`rho` pre-pinned **0.05** (SAM default); base optimizer, lr, grad_clip, loss all unchanged.

**(b) Ban-collision rulings (quoted scopes).**
- **F62 / F62b (SWA, Family C, KILLED):** scope = *weight **AVERAGING*** — "SWA **averages** per-epoch head
  weights and lost dev points on HateMM's mid-peak-dev seeds" (F62 body: "SWA lands 0.9–6.6 dev-acc pts
  BELOW the val-sel max"). SAM is a **training-time optimizer on a single trajectory / single model** — no
  post-hoc averaging. **NOT covered.** The SWA failure mechanism (averaging drags a mid-peak optimum) does
  not apply.
- **F69 (grad-norm checkpoint SELECTION, KILLED 2026-07-24):** scope = *selecting an existing checkpoint by
  gradient-norm score* — "paper 2601.16874 premise … argmin picks good epochs — INVERTS on our tiny
  retrieval head". SAM **selects nothing**; it changes the optimizer. **NOT covered as a ban.** **HOWEVER**
  F69 is the sharpest *empirical* datum against SAM's premise on this regime: grad-norm (a sharpness proxy)
  correlated with acc at **+0.61/+0.72/+0.62 (wrong sign, 3/3 seeds)** — i.e. *flatter = worse* here. This
  lowers the SAM prior below the litsurvey's; it does not forbid the measurement (SAM's m-sharpness in an
  ε-ball ≠ point grad-norm), but the direction is a headwind. **Report this in the prereg honestly.**
- **banned_constraints:** none touch a training-time optimizer (`cross-seed ensembles`, `MLLM-scores`,
  `OCR`, `gold-in-method`, `single-dataset-split`, `P1–P5`, `external-API`, `target-as-structure`,
  `kNN-pool-expansion` — none apply). **Clear.**

**(c) Legality.** IN-BOX (head training over cached embeddings; no gold/ensemble/encoder-touch).
D7 = generic optimizer → **D7-dead / performance-only** (irrelevant this round).

**(d) Patch surface (run_rac.py only; classifier & loss untouched).**
- **argparse (additive, no-op defaults):** `--sam` (bool, default `False`), `--sam_rho` (float, default `0.05`).
- **`model_pass` step block (`run_rac.py:625-629`)** — gate on `getattr(args,'sam',False)`:
  - else-branch = existing 5 lines, **byte-identical**.
  - SAM branch (~1 helper + ~10 lines): `zero_grad → loss.backward → _sam_ascend(model, rho)` (compute
    global grad-norm over `model.parameters()`, `ε=rho·g/(‖g‖+1e-12)`, `p.add_(ε)`, cache ε) → **recompute
    `compute_loss` at perturbed w, passing the SAME `train_feats/train_labels` returned by the first call**
    (no re-mine, §5) → `zero_grad → loss2.backward → _sam_restore(model)` (`p.sub_(ε)`) → existing
    `clip_grad_norm_(model.parameters(), grad_clip)` → `optimizer.step()`.
  - Implement SAM inline (Foret's two closures); **no new pip dependency** (env untouched).
- **Est. size:** ~30–40 lines in `run_rac.py`, +2 argparse lines. No RNG draws added (SAM is deterministic
  given grads ⇒ the arm's RNG stream matches the floor except for the weight trajectory).

**(e) Cost:** LOW (~+15–25 s/run vs base). **(f) Prior:** ~8–12% (see §0), best case = harden marginal ZH.

---

## 4. ARM B — modality-dropout regularizer on the head

**(a) Mechanism (as intended).** During head training, with prob `p` per sample drop (mask) exactly one of
the img / text projection streams, forcing the fused space not to over-rely on one stream. `p` pre-pinned **0.3**.

**(b) CRITICAL ruling — the deployed fusion breaks naive modality-dropout; fill-rule must be pre-pinned.**
The deployed fusion is `align` = **Hadamard** `x = img_feats ⊙ text_feats` (`classifier.py:120`), applied
to **L2-normalized** streams (`:114-115`). Modality-dropout in the literature assumes **concat** fusion,
where zeroing one half leaves the other half intact so the head is forced onto the survivor. Under Hadamard,
**zero-filling a dropped stream zeros the ENTIRE fused vector** (`img ⊙ 0 = 0`) → the sample degenerates to
a bias-only constant input, which is *not* modality-dropout but zero-embedding noise injection. **Ruling:
naive zero-fill is rejected.** The multiplicatively-correct analog is **identity-fill (ones):** set the
dropped normalized stream to `ones` so `x = img ⊙ 1 = img` (the surviving stream passes through unchanged).
**Pre-pin identity-fill.** Honest wrinkle to record: with `batch_norm=False` (deployed), a dropped sample
has `‖x‖=1` (unit survivor) while a joint sample has `‖img⊙text‖<1`, so dropout injects a magnitude shift
the head must absorb — inherent to multiplicative fusion, and another reason the prior is low.

**(c) Ban-collision rulings.**
- **cross-seed ensemble (banned):** modality-dropout is a **single-model, single-training-run** stochastic
  regularizer. **NOT an ensemble. NOT covered.**
- **F50 (FA fusion/composition gate, KILLED):** scope = *fixed, inference-time re-weightings/temperatures
  over **frozen** features* — "modality-fusion door closed; per-modality temperatures/reweights over banked
  frozen features." Modality-dropout is a **training-time stochastic regularizer** that shapes the head's
  learned weights, **not** a fixed feature-level composition applied at inference. **NOT covered.** (The
  fusion op itself, Hadamard-align, is unchanged; dropout perturbs training inputs, it does not add a
  learned combiner.)
- **F65 vision-LoRA / F44 EN-collapse:** different object (adapter capacity / encoder). Not relevant to a
  head-training regularizer.
- **banned_constraints:** none apply.
- **Legality:** IN-BOX. D7 = generic → D7-dead / perf-only.

**(d) Honest two-sided-to-negative prior.** F45 (ZH gain lives *entirely* in the text stream) and F58
(HateMM pass is text-carried, frozen-swap-sufficient) mean the **text stream is the carrying stream** on
both targets. A per-sample dropout that with prob p/2 removes text and forces the img-only survivor is
training the head on the **non-carrying** stream ~15% of samples → most likely **hurts** ZH and HateMM.
The upside case (rebalancing a text-dominant fusion) is exactly the case F44/F55 flags as **label-limited,
not imbalance-curable** on the collapsed dataset (EN — not even in this family). So on ZH/HateMM the honest
prior is **downside-skewed ~8–12%.** It remains a legal, cheap, publishable-either-way measurement.

**(e) Patch surface (classifier.py + run_rac.py wiring).**
- **argparse (additive, no-op defaults):** `--mod_dropout` (bool, default `False`), `--mod_dropout_p`
  (float, default `0.3`).
- **`classifier_hateClipper.__init__`** (already accepts `args=None`, `classifier.py:71`): store
  `self.mod_dropout`, `self.mod_dropout_p` from args (default off).
- **`classifier_hateClipper.forward`**, inserted **after** normalize (`:115`), **before** fusion (`:117`),
  gated `if self.training and getattr(self,'mod_dropout',False) and self.fusion_mode=='align'` (~10 lines):
  per-sample Bernoulli(`p`) mask + a fair img-vs-text coin, `drop_img = drop & coin`,
  `drop_text = drop & ~coin` (**at most one stream dropped per sample**), then
  `img_feats = where(drop_img, ones_like, img_feats)` / `text_feats = where(drop_text, ones_like, text_feats)`
  (identity-fill). Eval path untouched (`self.training` gate).
- **Est. size:** ~10 lines in `classifier.py` + build-wiring + 2 argparse lines.
- **RNG note:** this arm *does* add `torch.rand` draws inside forward (that IS the regularizer) → its RNG
  stream diverges from the floor. That is expected and confined to the treatment; the floor (flag off) and
  ARM A are unaffected.

**(f) Cost:** LOW (~base run time). **(g) Prior:** ~8–12%, downside-skewed on text-carried ZH/HateMM.

---

## 5. IMPLEMENTATION RISK

- **SAM double-step vs FAISS re-mining (the one structural risk) — HANDLED.** Re-mining lives inside
  `compute_loss` and fires only when `train_feats is None`; deployed `reindex_every_step=False` ⇒ it fires
  **once per epoch** (first step), and the returned `train_feats/train_labels` are reused for the rest of
  the epoch (mining already "stale-within-epoch" in the baseline). SAM's **second** `compute_loss` call
  must pass the `train_feats` returned by the first call ⇒ **no second re-mine at the perturbed weights**,
  byte-consistent with base semantics. The post-training E-step FAISS rebuild (`run_rac.py:1315-1329`) is
  on the **EM path only** (`seg_mode consensus/selfscore`), which the deployed config does **not** take
  (`seg_mode full`, else-branch `:1333`). So SAM touches no active re-mining hook beyond the one threaded
  call. Risk = LOW, one line of care.
- **Same-code / G-repro discipline — PRESERVED.** Both arms gate strictly behind `getattr(args,flag,False)`
  defaulting `False`. With the new flags absent, every code path is byte-identical to the banked floors ⇒
  **the floors need NOT be re-run**; a fresh control with the identical old command reproduces them
  bit-for-bit. The treatment sbatch runs the **SAME python command + the arm's flags** — the only Namespace
  diff is the two new keys, and they are inert unless set. This is the established additive-flag pattern in
  this codebase (cf. `getattr(args,'tarc_vote_gamma',0.0)`, `run_rac.py:675`). Answer to "does adding the
  argparse arg change the Namespace diff?": yes it adds keys, but with no-op defaults the *behavior* diff is
  zero when unset — same-code holds.
- **Collision naming.** group `RAC_video_headrecipe`; per-arm `--exp_comment` tags `_SAM_rho0.05` /
  `_MODDROP_p0.3`; fresh output dirs, `--force False` ⇒ **never** overwrites 13150 / 13241 or any banked arm.
- **Cleanest approach (recommended):** a **new sbatch** (`scripts/slurm/headrecipe_family.sbatch`) that
  calls the byte-identical enc3seed python command **plus** the arm flags; `run_rac.py` + `classifier.py`
  **additively** extended. Banked runs unaffected; no edit to any existing sbatch.

---

## 6. $0 PRE-GATE

**None needed / none warranted.** The entire 12-run family costs **< 0.15 GPU-h in one sbatch**; the
cheap-kill *is* the measurement. Any separate CPU pre-check would cost more analyst time than it saves GPU.
(Contrast the litsurvey C1/MCR arm, which is MED–HIGH GPU and *does* warrant its 0.599-floor pre-check.)

---

## 7. EXECUTION SKELETON (for the eventual ceremony — NOT run here)

```
# scripts/slurm/headrecipe_family.sbatch  (skeleton — 1×A100, 8 CPU, 64G, NO --time)
# group RAC_video_headrecipe ; force False ; fresh dirs (never overwrites 13150/13241)
# base cmd = the exact enc3seed python line; arms add ONLY the two flags below.
ZH=Qwen2.5-VL-7B-Instruct-LoRA_HF          # floor 13150
HM=Qwen2.5-VL-7B-Instruct-LoRA-curric_HF   # floor 13241
CONFIGS:                                    # 2 arms × 2 ds × 3 seeds = 12 runs
  ARM_A (SAM):        --sam True  --sam_rho 0.05       exp _SAM_rho0.05      {MHC_zh:ZH, HateMM:HM} × seed{0,1,2}
  ARM_B (mod-drop):   --mod_dropout True --mod_dropout_p 0.3  exp _MODDROP_p0.3  {MHC_zh:ZH, HateMM:HM} × seed{0,1,2}
# each run: python ./src/run_rac.py --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
#   --dataset <D> --model <M> --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
#   --fusion_mode align --hard_negatives_loss True --no_hard_negatives 1 --final_eval False \
#   --seed <S> --group_name RAC_video_headrecipe --metric cos --loss triplet --batch_norm False \
#   --hybrid_loss True --warmup 5 --majority_voting arithmetic --no_pseudo_gold_positives 1 \
#   --lambda_seg 0 --seg_mode full --num_subclips 4 --em_rounds 2 --consensus_topk 10 \
#   --consensus_margin 0.2 --Faiss_GPU False --force False <ARM FLAGS>
# same VAL-SELECTED + FINAL-EPOCH readout block as enc3seed*.sbatch → RESULT_ROW per run.
```

**Patch summary (3 files, additive):**
1. `src/run_rac.py` — `+--sam/--sam_rho`, `+--mod_dropout/--mod_dropout_p` argparse; SAM branch in
   `model_pass` step loop (helper `_sam_ascend/_sam_restore` + threaded 2nd `compute_loss`); pass
   mod-dropout args into `build_model`. (~40 lines.)
2. `src/model/classifier.py` — store `mod_dropout/mod_dropout_p` in `__init__`; identity-fill per-sample
   mask after normalize, gated `training ∧ mod_dropout ∧ align`. (~12 lines.)
3. `src/model/loss.py` — **no change** (SAM re-uses `compute_loss` as-is via threaded `train_feats`).

**Codex pre-submit review recommended** on the SAM branch (model-internal double-backward + param
perturb/restore correctness) before any SLURM submission, per project discipline.

---

## 8. PRIORS (honest, regime-specific)

| Arm | Litsurvey prior | Recon-revised | Direction | Why revised |
|---|---|---|---|---|
| A — SAM | ~10–15% | **~8–12%** | low | F69 (yesterday): flatness proxy *anti*-correlates with acc on this head (wrong sign 3/3) |
| B — mod-dropout | ~10–15% two-sided | **~8–12%** | downside-skewed | Hadamard needs identity-fill (zero-fill degenerate); ZH/HateMM text-carried ⇒ dropping text hurts |

Both D7-dead (irrelevant this round). Best realistic payoff for either = hardening the **marginal ZH**
val-selected leg; HateMM is near-ceiling (thin headroom).

---

## 9. FINAL VERDICT

**GO** — run the 2-arm family in ONE sbatch, ONE bite, hyperparameters/fill-rule frozen at recon-pin
(SAM `rho=0.05`; mod-dropout `p=0.3`, **identity-fill mandatory** under align). Ban collisions cleared:
**SWA F62/F62b** (averaging, ≠ optimizer), **F69** (selection, ≠ optimizer — but a measured flatness
headwind to disclose), **F50** (fixed inference-time fusion reweight, ≠ training-time regularizer),
**cross-seed ensemble** (both arms single-model). Total **< 0.15 GPU-h**. No $0 pre-gate needed. Any
surviving arm owes the full ceremony (prereg → independent review → freeze-hash → SLURM) before it counts.
Priors are low (~8–12% each) and honest; the family's value is a cheap, publishable-either-way closure of
the two remaining head-training-dynamics escape hatches on the marginal ZH cell.
