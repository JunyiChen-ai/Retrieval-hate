# SYNIB MASKED-CONSISTENCY PORT — FORENSIC RECON (zero GPU)

**Candidate:** `REPRO_SURVEY_2025.md` rank-1 — SynIB (arXiv 2606.09853, `kkontras/SynIB`, clone at
`external/baselines/SynIB`, single commit `3f3d3b8` "Initial public release").
**Brief:** port the "masked-consistency symmetric-KL term" into our RGCL align-head training as an
**additive** regularizer (triplet + BCE unchanged).
**Executor budget spent:** $0 — no GPU, no SLURM, no test touch, no `state/` mutation, no push.

---

## 0. VERDICT SUMMARY

**RECOMMENDATION: CONDITIONAL — default PARK; GO only on a specific LSMI branch (§8), and even then
as a 6-run single-dataset door-closer, not the 12-run family.**

Three findings drive it, in order of importance:

1. **The object named in the brief does not exist in the upstream code.** SynIB's live objective is
   **not** a symmetric KL between the intact-input prediction and the masked-input prediction. The
   intact prediction **never enters any KL**. What is actually computed is a **KL from the
   *masked-branch* predictive distribution to an uninformative prior** (`synergy_type="gaussian"`,
   a learned-variance Gaussian KL on the masked logits) or, in the config the paper's own
   `REPRODUCE.md` uses for Hateful Memes, a **forward KL from the masked-branch prediction toward
   the complementary *unimodal* head's (detached) prediction** (`synergy_type="unimodal_anchor"`).
   The only symmetric-ish logit-KL helper in the file (`_logit_kl`) is **commented out**
   (`synib_mask_model.py:668-673`). The survey's *mechanism intuition* — "penalise the head for
   staying confident when one modality is withheld" — is **correct**; its *stated form*
   ("symmetric-KL between intact and masked predictions") is **wrong**. Porting the brief's literal
   term would be porting an object SynIB does not implement, under SynIB's name. See §1 + §1.5
   (ERRATUM).
2. **The faithful term is mechanistically anti-aligned with our measured regime.** SynIB's HM
   weighting (`--l 0.01 --l_pareto 0.1`) puts **10× the weight on the image-destroyed branch**, i.e.
   it most strongly demands *uncertainty when the image is corrupted*. Our datasets are
   **text-carried** (F44: MHC-EN image stream collapses to near-chance; F73 ban text: modality
   dropout was "flat-to-harmful on **text-carried** datasets"). On a stream that carries ~0
   conditional information, the correct behaviour under corruption is to be *exactly as confident*;
   the term injects the opposite prior. This is a plausible mechanistic account of *why* F73 was
   harmful — and the KL makes that pressure **explicit and stronger**, not weaker. §5.
3. **The non-isomorphism to F73 is real but narrower than the survey claims, and it is load-bearing
   only through the KL.** Verified directly against the F73 `ban_scope` text (§4.2): the masking
   half (per-dimension instead of per-stream, cross-sample permutation fill instead of ones-fill) is
   defensible as a *different operator* but is exactly what a hostile reader calls "knobs" — and
   F73's ban says **"Do not re-tune knobs (one-bite family consumed)"**. The port is therefore
   admissible **only** as masking **+** KL, and a `masking_only` arm is **inside F73's ban** and must
   not be proposed.

Ban adjudication (§4): **OUTSIDE the F75 letter** (additive, not a swap, not toward
vote-consistent/contrastive/mixup-BCE) — same adjudication class as F79/ELR. **OUTSIDE the F73
letter** for the masking+KL object only. **No ban forecloses it.**

Honest prior (§5): **P(≥+0.030 on ≥2 datasets) ≈ 1–2 %**; **P(≥+0.010 stable on HateMM alone) ≈
5–8 %**; **P(clearing the binding ZH val-sel leg 3/3) ≈ 2–4 %**.

---

## 1. THEIR LOSS — WHAT THE CODE ACTUALLY COMPUTES (Duty 1a)

All line numbers are `external/baselines/SynIB/src/synib/models/vlm/synib_mask_model.py` unless
stated. The HM model class is `FusionIBModel_Mask` (`:1157`), whose synergy module is `SynIB`
(`:562`).

### 1.1 Loss assembly (the top-level equation)

`FusionIBModel_Mask._base_forward_synib` (`:1264-1332`) emits a `losses` dict; the trainer
(`src/synib/training/pipeline/helpers/Trainer.py:132-158`) forms

```
total = Σ_k  w_k · CE(preds[k], y)          # w_k = multi_loss.multi_supervised_w, defaultdict(int)
      + Σ    output["losses"].values()       # the KL terms, ALREADY weighted inside the model
```

For the HM method configs (`run/configs/hateful_memes/methods/synib.json`,
`synib_u.json`), `multi_supervised_w = {combined: 1, c: 0, g: 0}` — so **no CE is applied to any
masked prediction** and the whole SynIB contribution is the KL block. The deployed HM objective is

```
L = CE(logits_clean, y)  +  l · KL_branch1  +  l · l_pareto · KL_branch2
```

with `_branch_weight` (`:1070-1088`) supplying `l` to the `_1` branch (z1 clean, **z2 destroyed**)
and `l · l_pareto` to the `_2` branch (**z1 destroyed**, z2 clean). In the HM tier `enc_0` is the
**text** (DeBERTa) encoder and `enc_1` the **image** (CLIP) encoder, so branch `_1` = **image
destroyed** (weight `l`) and branch `_2` = **text destroyed** (weight `l·l_pareto`).

### 1.2 The KL term — three variants, none of them a clean-vs-masked symmetric KL

| Variant | selector | code | exact form |
|---|---|---|---|
| **gaussian** (default; `synib.json`) | `synergy_type: "gaussian"` | `_kl_loss` **`:1089-1097`** → `_gaussian_kl` **`:634-636`** | `0.5·Σ_c( exp(logvar) + mu² − 1 − logvar ).mean()` where **`mu` = the masked branch's logits** (`num_classes`-dim) and `logvar = self.logvar_head(feat_masked)`, `logvar_head = nn.Linear(fc_inner, num_classes)` (**a new parameter**, `:591`). This is `KL( N(mu, e^{logvar}) ‖ N(0, I) )` → **shrinks the masked-branch logits toward 0** (= max uncertainty). |
| **dirichlet** | `synergy_type: "dirichlet"` | `_dirichlet_kl` **`:638-652`** | evidential `KL( Dir(softplus(evidence_head(feat_masked))+1) ‖ Dir(prior_conc·1) )` — again a **KL to an uninformative prior**, needs `evidence_head` (`:594`). |
| **unimodal_anchor** (`synib_u.json`; **the config `docs/REPRODUCE.md:112-127` uses for the paper's HM runs**) | `synergy_type: "unimodal_anchor"` / `anchor_unimodal: true` | `_kl_unimodal_anchor` **`:1101-1106`** | `F.kl_div( log_softmax(pred_masked), softmax(pred_unimodal_target.detach()), reduction="batchmean" ) · branch_weight`. **Forward KL, not symmetric**; target = the **complementary modality's unimodal head**, detached. |

**There is no term anywhere in the live code that compares the intact-input prediction with the
masked-input prediction.** `preds["combined"]` (the clean logits) is used **only** for
`CE(combined, y)` and for diagnostics (`:1338-1390`). The one helper that would have formed a
logit-space KL between two predictions, `_logit_kl` (`:668-673`), and the categorical helpers
`_cat_kl` / `kl_to_uniform_multiclass_from_logits` (`:654-666`) are **all commented out**.

Call-site census (`grep -rn "_kl_unimodal_anchor\|_kl_pass\|_kl_loss\|_logit_kl" --include=*.py`):
`_kl_pass` `:1098` → `:1132,1133,1312,1313,1329,1330`; `_kl_unimodal_anchor` `:1101` →
`:1309,1310,1326,1327`; `_logit_kl` → **0 call sites** (commented).

### 1.3 What is masked, and how (the fill)

`SynIB.get_random_mask_multiclass` (**`:691-780`**):

```python
def make_keep(z, p):                      # :709  — NB: "keep" is really a DESTROY mask
    return (torch.rand_like(z) < p).to(z.dtype)

def fill_func(z, keep, ema=None):         # :723
    eps = z[torch.randperm(z.size(0))]    # cross-sample permutation along the BATCH dim
    # return (1 - keep) * z + keep * noise_fn(z, ema=ema)     <-- COMMENTED OUT
    return (1 - keep) * z + keep * eps
```

- **Granularity: per-element.** `torch.rand_like(z) < p` is drawn over the *full* tensor — for the
  `cls_type=="mlp"` path that is **per (sample, feature-dimension)** on the pooled projected
  vectors `z1,z2`; for the HM tier (`cls_type=="tf"`, `tiers/small_tf_deberta.json`) it is per
  (sample, token, dimension) on the **non-aggregated token sequences** `na_z1,na_z2` (`:749-763`).
- **Fill: cross-sample batch permutation, always.** `eps = z[torch.randperm(B)]` — the masked
  coordinate takes **another sample's value at the same coordinate**. The `zeros | noise | ema`
  branches of `noise_fn` (`:715-721`) are **dead code**: the only line that called `noise_fn` is
  commented out at `:725`. `FeatureStatsMasker.ema_update` still runs every step (`:750-751`,
  `:757-758`) but **its output is never consumed** on any live path. **`"fill": "ema"` in every HM
  config is therefore inert.**
- **`p` plumbing quirk (verify before quoting any p):** `self.p = float(self.perturb.get("p_min", 0.5))`
  (`:602`) — the config key **`"p": 0.5` is never read**. With `p_type != "diff"` (`"type": "rand"`
  in all HM configs; `--rmask random` sets it to `"random"`), `make_tilde_once` (`:729-736`) uses
  `self.p` = **`p_min`**, and **`p_max` is unused**. So the paper's HM command
  `--perturb_pmin 0.3 --perturb_pmax 0.5` runs at an effective corruption rate of **0.30**.
- `K = perturb.num_samples = 1` in every HM config → `repeat_k` (`:706`) is a no-op there.
- `p_type == "diff"` (not used at HM) instead sweeps a **cosine-schedule** `p(t)` across the `K`
  repeats, `_get_diff_p` (`:675-689`), `p ∈ [p_min, p_max]`, decreasing.

### 1.4 The forward passes and the branch structure

`_base_forward_synib` (`:1264-1332`) runs **two independently-gated branches**, both defaulting ON
(`synib_use_random_ce` / `synib_use_learnable_kl`, `:1288-1289`; `--rmask random|learned` sets only
`perturb.type` and does **not** gate them — `entrypoints/train.py:448-451`; the gates are the
separate `--no_random_ce` / `--no_learnable_kl` flags, `train.py:209-215`):

- **random branch** (`:1298-1315`): 3 extra fusion forwards — `randmask0` (z1 clean / z2 destroyed),
  `randmask1` (z1 destroyed / z2 clean), `randmask01` (both) — then KL on the first two.
- **learnable branch** (`:1317-1330`): `get_learnable_mask_multiclass` (`:779-1045`) runs an
  **inner optimisation** (`perturb.steps = 5`, `lr = 0.1`, hard-concrete gates, model frozen during
  the inner loop) to *learn* which dimensions to destroy, then 3 more fusion forwards + KL.

So the paper's HM command (which passes neither gate flag) runs **6 extra fusion forwards + a
5-step inner mask optimisation per training step**. `masking_only` (`src/synib/baselines/masking_only.py`,
configs `masking_only_{random,learned}.json`) keeps the masked forwards and drops **all** KL, moving
supervision to `multi_supervised_w = {randmask0: 1, randmask1: 1, ...}`.

`forward` (`:1392-1404`) gates the whole thing on `synergy_weight > 0` and
`perturb.ending_epoch`, and supports a linear warm-up of `l` over `bias_infusion.l_anneal_epochs`
(0 = off in all HM configs).

### 1.5 ERRATUM against `REPRO_SURVEY_2025.md` §4.1 — three corrections

The survey (`refine-logs/REPRO_SURVEY_2025.md:137-160`) is right about the plug point, the legality
and the `masking_only` ablation, and right that the fill is a batch permutation rather than
ones-fill. It is **wrong** on three points that matter for the port:

| Survey claim | Fact | Evidence |
|---|---|---|
| "adds a **symmetric-KL penalty between the intact-input prediction and the masked-input prediction**" | **No such term exists.** KL is either masked→uninformative-prior (gaussian/dirichlet) or masked→complementary-unimodal (forward KL, asymmetric). Intact prediction never enters a KL. | `:1089-1106`, `:1132-1133`, `:1309-1330`; `_logit_kl` commented `:668-673` |
| "masked coordinates are filled by a permuted other-sample value … alternative fills `zeros \| noise \| ema \| token \| shuffle`, EMA statistics tracked by a `FeatureStatsMasker`" | Permutation fill is correct and is the **only** live fill; the alternatives are **dead code** (the `noise_fn` call is commented out). EMA buffers are updated but unused. | `:715-727`, `:750-758` |
| implies the HM headline is `synib.json` | `docs/REPRODUCE.md:112-127` runs HM with **`synib_u.json`** (`unimodal_anchor`), `--l 0.01 --l_pareto 0.1 --perturb_pmin 0.3 --perturb_pmax 0.5`. `README.md:183-189` quick-start uses `synib.json` (gaussian). The two disagree. | as cited |

**Additional forensic flag on the HM headline config, stated as a fact and not adjudicated here:**
in `synib_u.json` the KL targets are `uni_pred_1/uni_pred_2`, produced by the encoders' own
`head = nn.Linear(d_model, num_classes)` (`models/vision_text/hf_text.py:43,123-135`). Those heads
receive gradient **only** through `CE(preds["c"|"g"], y)` weighted by `multi_supervised_w`, which
that config sets to **0/0**, and they are additionally `.detach()`-ed inside `_kl_unimodal_anchor`.
`Trainer.py:339-346` builds `w_loss = defaultdict(int)` from exactly that dict. On this reading the
HM anchor targets come from **untrained, randomly-initialised** linear heads. Anyone attempting a
faithful reproduction (survey §5 #2a) must resolve this before quoting the HM row.

### 1.6 Hyperparameters and their Hateful-Memes values

| Knob | Where | HM value |
|---|---|---|
| `bias_infusion.l` (λ) | `synib.json` / `synib_u.json` | `0.1` in-file; **overridden to `0.01`** by the `REPRODUCE.md` command |
| `bias_infusion.l_pareto` | CLI `--l_pareto` (`train.py:233-249`) | **`0.1`** → image-destroyed branch `0.01`, text-destroyed branch `0.001` |
| `perturb.p_min` (= the effective `p`) | `--perturb_pmin` | **`0.3`** |
| `perturb.p_max` | `--perturb_pmax` | `0.5` (**inert** at `type != "diff"`) |
| `perturb.fill` | tier + method | `"ema"` (**inert**, §1.3) |
| `perturb.num_samples` (K) | tier + method | `1` |
| `perturb.steps / lr / tau / lsparse` (learnable mask) | tier | `5 / 0.1 / 1.0 / 1.0` |
| `synergy_type` | method config | `gaussian` (`synib.json`) / `unimodal_anchor` (`synib_u.json`, used by REPRODUCE) |
| `multi_supervised_w` | method config | `{combined: 1, c: 0, g: 0}` |
| optimiser / schedule | `default_config_hm.json` | AdamW `lr 1e-4`, wd `1e-4`, cosanneal `max_lr 1e-3`, warmup 100 steps |
| batch / epochs / seeds / folds | `default_config_hm.json`, `REPRODUCE.md:139-140` | **32** / max 50 (early stop `n_steps_stop 20`) / **109, 27, 3407** / folds 0,1,2 |
| backbones | `tiers/small_tf_deberta.json` | frozen CLIP-ViT-B/16 + DeBERTa-v3-base, `d_model 512`, `fc_inner 128`, `cls_type "tf"`, 2 fusion layers |

---

## 2. OUR INSERTION MAP (Duty 1b)

### 2.1 The receiving code

- `src/model/classifier.py:115-147` — `classifier_hateClipper.forward`: `img_proj` → `text_proj` →
  L2-normalise each → (`align`) `x = torch.mul(img, text)` → `mlp` → `output_layer` (**a single
  logit**, `:109`/`:114`). Deployed geometry (`scripts/slurm/enc3seed_lora_curric.sbatch:54-67`):
  `--map_dim 1024 --proj_dim 1024 --fusion_mode align --batch_size 64 --epochs 30 --dropout 0.2 0.4 0.1
  --batch_norm False --loss triplet --hybrid_loss True` (⇒ `ce_weight` default 0.5,
  `run_rac.py:156`).
- `src/model/loss.py:12` — `compute_loss`; `:31-32` `model.train(); output, feats = model(img, txt, return_embed=True)`;
  triplet assembly `:~100-577`; hybrid BCE `:577-597`; **additive-term region `:599-630`**
  (`lambda_seg`, `lambda_aux`, `lambda_tarc` — the established `L = L_main + λ·L_extra` pattern with
  "λ==0 ⇒ EXACT no-op, byte-identical" comments); return `:632`.
- `src/model/loss.py:697-744` — `_manifold_mixup_bce`: **the exact precedent** for a second head
  forward on modified post-projection features. It re-derives `norm(img_proj(x)) * norm(text_proj(x))`
  **inside loss.py without touching classifier.py**, and carries the REFREEZE-1 dropout guard.

### 2.2 What would be masked, concretely

**Per-dimension on each projected, L2-normalised stream, pre-fuse** — i.e. SynIB's `cls_type=="mlp"`
path, which is the only one our architecture has (we hold pooled vectors, not token sequences; the
HM tier's `tf` path has no analogue here).

```
img = normalize(img_proj(image_feats))            # [B, 1024]
txt = normalize(text_proj(text_feats))            # [B, 1024]
m_i = (rand_like(img) < p)                        # per (sample, dim)
img~ = where(m_i, img[randperm(B)], img)          # cross-sample permutation fill
x_m0 = img~ * txt      ;   z_m0 = output_layer(mlp(x_m0))     # image-destroyed branch
x_m1 = img  * txt~     ;   z_m1 = output_layer(mlp(x_m1))     # text-destroyed branch
```

**Hadamard-specific note (this is the F73 distinguisher, and it is real).** Under `x = img ⊙ txt`,
F73's **ones-fill** sets `img_d = 1` so `x_d = txt_d` — the surviving stream passes through
*unchanged*, an unusually weak perturbation. The permutation fill sets `x_d = img_d^{(π(i))}·txt_d`,
a genuinely different value drawn from the same normalised marginal. **Do not port a zeros-fill**:
`img_d = 0 ⇒ x_d = 0` would null the fused coordinate entirely (the reason F73 chose ones-fill in
the first place, `classifier.py:120-124`).

### 2.3 Where the KL comes from — and the binary-head problem

Our head emits **one** logit and trains with `BCEWithLogitsLoss`. Consequences:

- **`unimodal_anchor` is not portable as-is.** We have **no unimodal heads**; adding two would mean
  new parameters, a new optimiser branch (the `aux_pack` precedent, `run_rac.py:660-667`), and a
  decision about how they are trained — which upstream leaves in the state described in §1.5. Rule
  it out for a minimal port.
- **`gaussian` needs a `logvar_head`** = `nn.Linear(proj_dim, 1)`, again new parameters + optimiser
  surgery.
- **`softmax`/`log_softmax` over a 1-logit vector is identically 1 / 0** — any port that naively
  reuses `F.kl_div(F.log_softmax(z), ...)` on our single logit is **degenerate and silently
  computes 0**. This is the single most likely implementation bug in this port and belongs in the
  smoke list.

**Recommended faithful, parameter-free form (PORT-A).** Take the *exact binary analogue* of SynIB's
"KL from the masked-branch predictive distribution to an uninformative prior":

```
L_synib = λ · [  E_B  KL( Bern(σ(z_m0)) ‖ Bern(0.5) )
        + λ_p · E_B  KL( Bern(σ(z_m1)) ‖ Bern(0.5) ) ]
        = λ · [ E_B (log2 − H_b(σ(z_m0))) + λ_p · E_B (log2 − H_b(σ(z_m1))) ]
```

bounded in `[0, log2]` per sample. **DEVIATION-1 (must be pre-registered):** SynIB learns the
variance via `logvar_head`; we use the fixed-variance / exact-Bernoulli analogue so the port adds
**zero parameters**. This preserves the mechanism (shrink masked-branch confidence toward the
uninformative prior) and drops only the learned-scale freedom.

**PORT-B (the brief's literal term — symmetric KL between clean and masked predictions) is a
different object and must not be run under SynIB's name.** It is a *consistency / invariance*
regulariser (R-Drop-shaped: pull masked toward clean), whose sign of intent on the confidence axis
is **opposite** to PORT-A's. It has no upstream implementation, no `masking_only` ablation backing
it, and — critically — it is *much* closer to "masking-as-augmentation with a consistency penalty",
which is the reading of F73 that the ban's "do not re-tune knobs" clause reaches. If PORT-B is
wanted it needs its own literature anchor and its own non-isomorphism argument.

### 2.4 Where the second forward comes from, and the four hazards

Insert as a new additive block at `loss.py:631` (after the `lambda_tarc` block, before `return`),
with a helper `_synib_masked_ib(model, image_feats, text_feats, args)` next to `_manifold_mixup_bce`.
`classifier.py` is **not** touched (the mixup precedent re-derives the align forward locally). Four
hazards, all with an existing precedent:

1. **Model is in `eval()` mode at that point.** The FAISS mining call sets `model.eval()`
   (`src/utils/retrieval.py:330`) and never restores it — this is exactly the REFREEZE-1 bug the
   codex gate caught on A3 mixup (`loss.py:709-721`, `NCA_SUBMIT_RECORD.md §2.1`). The masked
   forwards **must** re-enable the `nn.Dropout` submodules and restore each module's prior mode, or
   the masked branch runs dropout-off while the clean logits were computed dropout-on. Copy the
   mixup guard verbatim.
2. **RNG.** ON-path draws per step: `2 × torch.rand_like([B,1024])` (CUDA), `2 × torch.randperm(B)`
   (**CPU** RNG unless `device=` is passed — pass it explicitly), plus the dropout draws of two
   extra head forwards. The arm's RNG stream therefore diverges from the floor — accepted and
   precedented (`classifier.py:125-128` says the same for F73 ARM B). **The OFF path must draw
   nothing**, which is what makes the banked floors (13150 ZH, 13241 HateMM) reusable without
   re-running.
3. **Mining inertness.** The masked forwards must **not** call any `dense_retrieve_*`, and must not
   touch `train_feats/train_labels`. Assert it (NCA harness check C4 precedent).
4. **Triplet inertness.** `feats` (the grad-tracked clean embedding feeding the triplet term and the
   kNN memory) must be read from the **clean** forward only. The masked branches feed the KL and
   nothing else, mirroring A3's "kNN memory reads real neighbours".

### 2.5 Cost per step

Two extra `mlp + output_layer` forwards at `[64, 1024] → [64, 1024] → [64, 1]` plus two
`rand_like` + `randperm` draws. Against a step already dominated by FAISS mining over the train
bank, this is **≈ +5–15 % wall per step**, and **0 new parameters ⇒ no optimiser change**. (SynIB's
own HM setting costs 6 extra fusion forwards + a 5-step inner optimisation; the minimal port takes
**2** and **no** learnable-mask branch — see §6 for why the learnable branch is deliberately
excluded.)

---

## 3. WHAT WOULD BE *REPRODUCED* vs. WHAT WOULD BE *NEW*

| SynIB piece | Ported? | Note |
|---|---|---|
| per-element Bernoulli(p) mask on each stream | **yes** | per-dimension on pooled vectors (their `mlp` path) |
| cross-sample permutation fill | **yes** | the only live fill upstream |
| KL from masked branch to uninformative prior | **yes, in binary form** | DEVIATION-1, §2.3 |
| asymmetric branch weights `l`, `l·l_pareto` | **yes, as a knob** | but see §5 on the direction of the asymmetry |
| learned-variance `logvar_head` | **no** | avoids new parameters + optimiser surgery |
| `unimodal_anchor` variant | **no** | no unimodal heads exist here (§2.3) |
| learnable-mask branch (5-step inner loop) | **no** | ~6× the cost, second multiplicity axis, and its own hyperparameters (`lsparse`, `tau`, warmup, schedule) |
| `masking_only` ablation arm | **NO — banned** | this arm is inside F73's `ban_scope` (§4.2) |
| cosine `p_type="diff"` schedule | **no** | inert at HM anyway (`K=1`) |
| EMA feature statistics | **no** | dead code upstream (§1.3) |

---

## 4. BAN / PRECEDENT ADJUDICATION (Duty 2)

### 4.1 F75 (head-loss family) — **OUTSIDE the letter**

`state/directions_tried.json` F75 `ban_scope`, verbatim:

> "head-loss **swaps** of the triplet+BCE hybrid **toward vote-consistent (NCA/soft-kNN), contrastive
> (SupCon), or mixup-BCE objectives** at 7B frozen-encoder feature scale; tau/alpha retunes =
> tactics, banned; …"

Three tests, following the F79/ELR precedent (`ELR_FORENSIC_RECON.md §1`):

- **Is it a swap?** **No.** `L = 0.5·triplet + 0.5·BCE + λ·L_synib`. The triplet term is untouched
  (NCA/SupCon *replaced* it) and the BCE **target** is untouched (mixup *rewrote* it). Structurally
  identical to the additive `lambda_seg` / `lambda_aux` / `lambda_tarc` pattern already in the file.
- **Is it "toward" one of the three named families?** **No.** It is a masked-input
  information-bottleneck / confidence-calibration term — a fourth mechanism, not soft-kNN, not
  SupCon, not mixup-BCE.
- **Is it a τ/α retune of a banned arm?** **No** — new term, new hyperparameters (λ, p, λ_pareto).

**Ruling: outside F75's letter, same adjudication class as F79's ELR (which was ruled admissible on
exactly this reasoning and parked on *prior*, not on ban).**

**Honest counter-note (spirit).** F75 bans **mixup-BCE**, itself a *feature-space perturbation +
modified target* rather than a pure swap — and PORT-A is also a feature-space perturbation with an
auxiliary objective. A reviewer can and should raise this. My rejection of that reading is the same
as ELR's: F75 enumerates three **named** families each killed on its own measured arm (A3 mixup
dead), and this is a genuinely fourth mechanism. **What does transfer is the F75 *finding*** — "first
measured negative for trained-reshaping-unlocks-oracle-headroom" — as a **prior**, and it is a heavy
one (§5).

### 4.2 F73 (SAM + modality dropout) — **OUTSIDE the letter for masking+KL; the `masking_only` arm is INSIDE**

F73 `ban_scope`, verbatim:

> "SAM (rho 0.05) on the retrieval head: … **Modality-dropout (p 0.3, ones-fill) on Hadamard head:
> flat-to-harmful on text-carried datasets. Do not re-tune knobs (one-bite family consumed).**"

I verified the survey's non-isomorphism argument **against this text and against our own ARM-B code**
(`classifier.py:118-136`) rather than accepting it:

| Axis | F73 ARM B (measured, dead) | SynIB port | Different? |
|---|---|---|---|
| granularity | **per-stream** — `torch.where(drop_img, ones_like(img), img)` drops the *entire* vector for a sampled subset of rows | **per-dimension**, every row, fraction `p` of coordinates | **yes** |
| incidence | `p=0.3` of **samples**, exactly one stream each | `p=0.3` of **coordinates**, both branches always | **yes** |
| fill | **ones** ⇒ under `⊙`, surviving stream passes through *unchanged* | cross-sample permutation ⇒ a real off-manifold value | **yes** |
| objective | **none** — pure augmentation; loss unchanged | **+ explicit KL to uninformative prior on the masked branch** | **yes — and this is the load-bearing one** |
| upstream corroboration | — | SynIB ships `masking_only` (`baselines/masking_only.py` docstring: "isolates the data-augmentation effect of masked inputs **from the KL objective itself**") | the non-isomorphism is asserted by the **source repo**, not by us |

**Ruling.** The masking half alone is *exactly* what "do not re-tune knobs" reaches — `p` and fill
are knobs of ARM B, and granularity is arguably one too. **The port is admissible only because and
only insofar as the KL term is present.** Two binding consequences for the prereg:

1. **No `masking_only` arm.** It is a re-tune of a consumed one-bite family. If we ever want that
   number, it is already measured: **F73 is our `masking_only`.**
2. **λ = 0 is not a legal arm** (it is the F73 cell). The floor comparison is the banked floor.

### 4.3 F79 / ELR — precedent for the adjudication class

ELR was ruled **outside** F75 on the additive-vs-swap test and then **parked on prior**
(`ELR_FORENSIC_RECON.md:39-77`, `directions_tried.json` F79: "PARKED at recon, $0 (P(+3)~1-2%)").
The SynIB port sits in the **same class**: legally admissible, additive, cheap — and must be judged
on prior, not on ban. That is what §5 does.

### 4.4 Other constraints — all clear

In box on every standing veto: no new data, no OCR (user veto), no gold annotations, no cross-dataset
mixing (single-dataset own train split), no cross-seed ensembles, no test-time component, no raw
video anywhere, no external model weights. The port is a training-time term on our own two cached
streams.

---

## 5. HONEST HEADWINDS AND PRIOR (Duty 2c)

1. **Head-side base rate: 0-for-~20.** F70 readout grid inside perm-null, F73 SAM/mod-dropout ±noise,
   F75 loss-family 0/8 FORMAL. No head-side change has ever cleared the goal 3/3
   (`FUSIONCAT_PREREG.md:90`, `FUSIONSWAP_FORENSIC_RECON.md:105`). PORT-A is head-side.
2. **The nearest measured neighbour is a kill, not a null.** F73's ARM B is the *same operator
   family on the same head with the same p*. Its verdict was "**flat-to-harmful on text-carried
   datasets**".
3. **Mechanism anti-alignment (the strongest of the honest headwinds).** SynIB's HM weighting puts
   `l` on the image-destroyed branch and `l·0.1` on the text-destroyed branch — i.e. it maximally
   demands *uncertainty when the image is corrupted*. Our regime is the opposite of the one that
   licenses this: **F44** measured the MHC-EN image stream at near-chance and equal-weight concat
   cancelling the text gain; **F73** named text-carried datasets as where the masking hurt. Forcing
   the head toward `p=0.5` when a near-uninformative stream is perturbed asks it to **discard a
   legitimately text-only decision**. Any port that keeps `l_pareto = 0.1` in SynIB's orientation is
   pushing on the wrong stream; flipping the orientation is a *knob* choice we would be making from
   our own data, which is a forking-path we must pre-register.
4. **Binding leg arithmetic (ZH val-sel).** The align floor sits at **+0.0246 acc over CLIP
   (val-sel FAIL)** and needs **+0.0054 more, 3/3**, through a **78-item dev** whose selection noise
   is ≈ the **±0.014** head-seed band (`FUSIONSWAP_FORENSIC_RECON.md:103`,
   `CAND2_REP2_PREREG.md:152-155`). A regulariser whose expected mean effect is ≲1 pt cannot
   reliably push all three val-selected seeds across a bright line there.
5. **HateMM is a hold-the-pass leg, not a goal leg.** Floor ≈ 0.879 (project best 0.8775–0.879);
   a FORMAL +0.030 → ≈0.909 is arithmetically implausible (`FUSIONCAT_PREREG.md:93-95`).
6. **Law-I / F66.** 91–98 % of oracle headroom is formally selection-locked; F75 is the first
   *measured* negative for "trained reshaping unlocks it". PORT-A is a trained reshaper.

**Prior.** P(≥+0.030 acc on ≥2 datasets, i.e. the goal) **≈ 1–2 %**. P(≥+0.010 stable, 3/3, on
HateMM alone) **≈ 5–8 %**. P(clearing the binding ZH val-sel leg 3/3) **≈ 2–4 %**. These are at or
just below the ELR/F79 park line, and the "nearest neighbour is a measured kill" factor is worse
here than it was for ELR.

**What we learn even if it fails** (the only genuinely positive column): a negative **completes the
F73 story** — F73 measured masking-as-augmentation with the weakest possible fill, and the source
repo's own ablation says that is the wrong half. Measuring the KL half closes the modality-masking
family with a documented non-isomorphism instead of leaving a letter-overreach in the ban scope.
That is analysis-chapter value, not goal value, and it should be priced as such.

---

## 6. MINIMAL DECISIVE FAMILY DESIGN (Duty 3)

### 6.1 Arms

**Two arms, one hyperparameter moved.** The natural "their-default vs 0.5×" contrast is **not**
decisive here: SynIB's `l = 0.01` is calibrated against a CE of weight 1 with a multi-class Gaussian
KL of magnitude O(1–10); our term is bounded by `log 2 ≈ 0.693` and sits against
`0.5·triplet + 0.5·BCE`. At λ = 0.01 the term would contribute **<1 %** of the objective and a 0.5×
arm would be measuring noise twice. The decisive contrast is **"is the term inert?" vs "is the term
too strong?"**:

| Arm | flags | rationale |
|---|---|---|
| **S1** `λ = 0.01`, `λ_pareto = 1.0`, `p = 0.3` | `--synib_lambda 0.01 --synib_pareto 1.0 --synib_p 0.3` | SynIB's HM λ verbatim, **symmetric** branch weighting (their CREMA-D default `l_pareto = 1.0`), their effective `p`. Symmetric is chosen over HM's 0.1 to avoid picking a stream orientation from our own data (§5.3) |
| **S2** `λ = 0.1`, `λ_pareto = 1.0`, `p = 0.3` | `--synib_lambda 0.1 --synib_pareto 1.0 --synib_p 0.3` | the in-file `bias_infusion.l` of `synib.json`/`synib_u.json`, and the scale at which the term is a non-vestigial fraction of our objective |

Explicitly **not** in the family: `masking_only` / λ=0 (banned, §4.2), the learnable-mask branch
(cost + second multiplicity axis), `p` variants, `λ_pareto` variants, PORT-B. **One sbatch = one
pre-registered family = one multiplicity bite.**

### 6.2 Shape and pairing

**2 arms × 2 datasets × 3 seeds = 12 head-only runs**, single sbatch, sequential, 8 CPU / 64 G /
1×A100 (satisfies the never-two-concurrent-16-CPU rule). Floors are the **banked** paired controls —
**not re-run** (guaranteed by the no-flag inertness proof, §6.4):

- ZH `Qwen2.5-VL-7B-Instruct-LoRA_HF` vs job **13150**, per head-seed;
- HateMM `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` vs job **13241**, per head-seed.

Everything else byte-identical to `enc3seed_lora_curric.sbatch:54-67`; fresh `--group_name`,
`--force False`.

### 6.3 Diff size (codex gate WILL apply — model internals)

| File | change | est. lines |
|---|---|---|
| `src/run_rac.py` | 3 additive `getattr`-gated flags (`--synib_lambda` default **0.0**, `--synib_p` default 0.3, `--synib_pareto` default 1.0) + the standard inertness comment block | **+18** |
| `src/model/loss.py` | additive block at `:631` (`if synib_lambda > 0 …`) | **+14** |
| `src/model/loss.py` | `_synib_masked_ib(...)` helper beside `_manifold_mixup_bce` (dropout-mode guard, two masked forwards, binary KL, `assert fusion_mode == "align"`) | **+50** |
| `src/model/classifier.py` | **none** | **0** |
| `scripts/slurm/synib_family.sbatch` | new, `ncafam_family.sbatch` clone | ~110 (not model internals) |

**≈ +82 lines across 2 source files, 0 lines modified or deleted.** Codex review is mandatory
(second forward, RNG, dropout mode, model internals) — the same trigger that produced the A3
REFREEZE-1 catch.

### 6.4 Smoke requirements (CPU, pre-submit)

1. **No-flag inertness proof** (the NCA C4 analogue, and the load-bearing one — it is what lets the
   banked floors stand as controls): with the flags absent, (a) `Namespace` gains `synib_lambda=0.0`
   and the block is not entered; (b) a 3-step CPU harness produces a **bit-identical** loss trace
   vs. the pre-change file; (c) `torch.random.get_rng_state()` (CPU **and** CUDA) after N steps is
   **identical** to the pre-change run ⇒ zero extra draws.
2. **Dropout-mode guard**: after `_synib_masked_ib` returns, every `nn.Dropout` module's `.training`
   equals its pre-call value; and during the masked forwards all of them are `True`
   (`restored_to_eval_after` style assertion, NCA C1 analogue).
3. **Degenerate-softmax trap**: assert the implementation does **not** route the single logit through
   `log_softmax` (§2.3); assert `L_synib > 0` on random data and `L_synib → 0` as `|z_m| → ∞`.
4. **Mask statistics**: realised corruption fraction ≈ `p ± 0.02` at `B=64, d=1024`; fill values are
   drawn from *other* rows (self-map fraction ≈ 1/B ≈ 0.016); `randperm` is created on the feature
   tensor's device.
5. **Mining inertness**: `dense_retrieve_*` call count and `train_feats`/`train_labels` object
   identity unchanged with the flag ON.
6. **Triplet inertness**: the triplet term's inputs are the clean `feats` only; the masked branches
   carry no path into `in_batch_loss`/`hard_loss`/`pseudo_gold_loss`.
7. **Bounds + finiteness**: per-sample KL ∈ `[0, log 2]`; total finite; gradient reaches
   `img_proj`, `text_proj`, `mlp`, `output_layer`.
8. **Align pin**: `assert model.fusion_mode == "align"` (mixup precedent, `loss.py:708`).

### 6.5 Cost

NCA precedent: **24 head runs = 9 m 28 s** wall of actual compute (`FUSIONCAT_PREREG.md:142`; job
13482 `elapsed 00:19:19` total, the balance being `disk_guard.sh` housekeeping), ~0.33 GPU-h charged.
PORT-A adds 2 head forwards/step (§2.5).

- **12 runs ≈ 5–6 min compute**, ≈ **0.10–0.17 GPU-h** charged (allow the disk-guard overhead).
- Reduced single-dataset door-closer: **6 runs ≈ 3 min**, ≈ **0.05–0.08 GPU-h**.
- CPU smoke: minutes, no GPU.

---

## 7. DEPENDENCY CHECK (Duty 4)

- **Beyond torch: nothing.** PORT-A needs `torch.rand_like`, `torch.randperm`,
  `torch.nn.functional.logsigmoid`/`softplus` (or `binary_cross_entropy_with_logits`) — all core.
  SynIB's `requirements.txt` pins torch 2.9.1 / transformers 4.57.1 / wandb / easydict etc., but
  **none of that is needed for the port**; we lift ~15 lines of tensor algebra, not the package.
  (`wandb` is imported at module scope in `synib_mask_model.py` — another reason to lift the algebra
  rather than import their module.) No `HateVideo` env change; no new conda package.
- **EMA state: none, and this is a genuine finding rather than a design choice.** SynIB's EMA fill is
  **dead code** (§1.3) — `fill_func`'s `noise_fn` call is commented out and only the batch
  permutation is live. So the port carries **no EMA buffers**, hence **no interaction with the
  per-epoch FAISS re-mining loop** (`run_rac.py:684-697`), no epoch-boundary state, no bank staleness
  question, and no new optimiser parameter group. Contrast with the NCA arm, which *did* need a
  per-epoch detached bank.
- **Batch-size sensitivity at Bz64.** The fill is a **within-batch** cross-sample permutation, so the
  fill distribution is the empirical batch marginal. At `B=64` (vs SynIB's HM `B=32`) this is
  strictly better-conditioned. Two second-order notes: (i) `randperm` has ≈1/B ≈ 1.6 % fixed points,
  so ~1.6 % of "masked" coordinates are self-filled and thus unperturbed — negligible; (ii) the last
  batch of an epoch is smaller, which only sharpens the same effect. No batch-composition coupling
  with the triplet term, because the masked branches never enter it (§2.4 hazard 4).
- **Determinism / repro**: `torch.randperm` defaults to the **CPU** generator; pass `device=` so the
  arm's draw lands on the same generator as `rand_like` and the G-repro audit has one stream to
  reason about.

---

## 8. WAIT-FOR-LSMI: HOW THE SIBLING GATE MODULATES THE LAUNCH (Duty 5)

The LSMI recon/exec running in parallel (`state/progress.json` phase line; survey §4.2) estimates
**per-sample synergy `s`, redundancy `r`, uniqueness `u1,u2`** on our banked feature caches. SynIB's
entire premise is that there is **synergy to protect** — the term's job is to stop the head from
becoming confident on evidence that is not jointly grounded. That premise is exactly what LSMI
measures, which makes it a true pre-gate rather than a nice-to-have.

| LSMI branch | Reading | Decision on this port | Prior |
|---|---|---|---|
| **(a) `s ≈ 0` on all datasets** | there is no synergy to preserve; the KL would penalise confidence that is legitimately unimodal (§5.3 becomes not a headwind but a proof) | **PARK, no GPU.** Write the F73-completion note from the recon alone; the negative is already argued, not measured | **< 1 %** |
| **(b) estimator unstable at n≈600** | LSMI is uninformative; we are back to the unconditional prior | **PARK** (default). The port would be launched on an unverified premise against a 0-for-~20 base rate | 1–2 % |
| **(c) `s > 0` on HateMM only** | the stratum SynIB targets exists on our strongest dataset | **CONDITIONAL GO — HateMM-only door-closer**, S1+S2 × 3 seeds = **6 runs ≈ 0.05–0.08 GPU-h**. Honest label: HateMM is a hold-the-pass leg (floor ≈0.879 near ceiling), so this is **paper/analysis value**, not a goal lever | 5–8 % for ≥+0.010 on HateMM; ≈0 for the goal |
| **(d) `s > 0` on ZH** | the synergy stratum exists on the **binding goal leg** | **GO — ZH-first**, S1+S2 × 3 seeds = 6 runs; escalate to the full 12-run family only if ZH shows sign-3/3 | 2–4 % for the ZH val-sel leg |
| **(e) `s > 0` on both ZH and HateMM** | best case available | **GO — full 12-run family** as specced in §6 | ≈ 3–5 % for the goal |

**Standing recommendation absent LSMI: PARK.** Branch (a) or (b) — which the survey itself calls the
more likely outcomes given F50/F44 — closes the port at $0. Do not submit before the LSMI reading
lands; the whole point of the sibling gate is that it is free and this is not.

**One thing that should be recorded regardless of branch:** the §1.5 erratum. `REPRO_SURVEY_2025.md`
§4.1 and §6 both describe SynIB's core as a "symmetric-KL between intact and masked predictions",
and any downstream prereg or paper sentence built on that phrasing would misattribute the method.
That correction is worth more than the port's expected numeric value.

---

## 9. FILES READ (provenance)

Upstream (`external/baselines/SynIB/`): `src/synib/models/vlm/synib_mask_model.py` (1417 L, read
`:58-73, 110-170, 444-500, 565-790, 1040-1420`), `src/synib/baselines/masking_only.py`,
`src/synib/training/pipeline/helpers/Trainer.py:125-160, 339-346`,
`src/synib/entrypoints/train.py:100-260, 354-462`, `src/synib/models/vision_text/hf_text.py`,
`run/configs/hateful_memes/{tiers/small_tf_deberta.json, default_config_hm.json,
methods/{synib,synib_u,vanilla,uni_text,masking_only_random}.json}`, `run/hateful_memes/train.sh`,
`docs/REPRODUCE.md:100-145`, `README.md`, `requirements.txt`, `pyproject.toml`.

Ours: `src/model/loss.py:1-80, 570-640, 697-780`, `src/model/classifier.py`,
`src/run_rac.py:150-600, 660-760`, `src/utils/retrieval.py:325-345`,
`scripts/slurm/{enc3seed_lora_curric,ncafam_family}.sbatch`,
`autoresearch/goal_mllm_plus3/state/{directions_tried.json (F72,F73,F75,F76,F78,F79,F80),
progress.json}`, `refine-logs/{REPRO_SURVEY_2025, ELR_FORENSIC_RECON, FUSIONSWAP_FORENSIC_RECON,
FUSIONCAT_PREREG, NCA_SUBMIT_RECORD, CAND2_REP2_PREREG}.md`.

**Nothing in this document is quoted from a README or a paper abstract; every claim about SynIB's
objective is from the source file at the cited line.**
