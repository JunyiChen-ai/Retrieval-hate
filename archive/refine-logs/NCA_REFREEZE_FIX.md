# NCA / soft-kNN HEAD-LOSS family — REFREEZE-1 FIX RECORD (codex STOP → surgical fix)

**Role:** re-freeze fix executor. CPU-only; NO SLURM, NO GPU, NO test-touch, NO `state/` mutation,
NO push. Applies the minimal surgical fix for the ONE blocking codex finding that HALTED the NCA
family submission, produces equivalence evidence, and re-issues the freeze block. **Authorization
remains VOID until an independent 0-context re-review approves and the codex gate re-runs
(prereg §4.5).**
**Date:** 2026-07-25 NZST.

---

## 1. Codex finding recap (verbatim source: `refine-logs/NCA_SUBMIT_RECORD.md §2.1`)

The mandatory codex gate (prereg §4.5; `codex exec`, model `gpt-5.4`, `xhigh`, two rounds) escalated
ONE finding to **BLOCKING** for the **A3 (manifold-mixup)** arm and HALTED the submission:

> **A3's mixup BCE forward runs with the classifier's dropout DISABLED (eval mode).**

Confirmed by independent source re-read (all three legs) THIS record:

1. **The mixup forward uses dropout-bearing head submodules.** `_manifold_mixup_bce`
   (`src/model/loss.py`) forwards through `model.img_proj`, `model.text_proj`, `model.mlp`,
   `model.output_layer`. In `classifier_hateClipper`: `img_proj`/`text_proj` each
   `= nn.Sequential(nn.Linear, nn.Dropout(dropout[0]=0.2))` (`classifier.py:81-82`); `mlp` starts
   `nn.Dropout(dropout[1]=0.4)` then per-layer `nn.Dropout(dropout[2]=0.1)` (`classifier.py:96,103`).
   These dropouts are train/eval-mode sensitive.
2. **Mining leaves the model in eval mode and never restores it.**
   `dense_retrieve_hard_negatives_pseudo_positive` calls `model.eval()` (`retrieval.py:330`) and
   returns (`retrieval.py:582/584`) WITHOUT restoring the prior mode.
3. **In A3 the mixup forward runs AFTER mining with no intervening `model.train()`.** For A3
   (`head_loss='triplet'`, `mixup=True`): `model.train()` (`loss.py:31`) → train-mode forward
   (`loss.py:32`) → [triplet, so NO early return] → FAISS mining (`loss.py:310`, since
   `no_pseudo_gold_positives=1>0`) → **model now EVAL** → triplet assembly → hybrid block →
   `_manifold_mixup_bce` (`loss.py:585`). The ONLY `model.train()` in `loss.py` are line 31 (before
   mining) and line 973 (inside `compute_segment_loss`, NOT called at `lambda_seg=0`). So A3's mixup
   BCE forward executed with `img_proj/text_proj/mlp` **dropout OFF**.

**Why BLOCKING:** the FLOOR's BCE (`mixup=False`, `loss.py:588-594`) uses `output` from the
train-mode forward at `loss.py:32` → dropout **ON**. So A3 differed from its floor in TWO ways: the
intended manifold-mixup interpolation AND an UNINTENDED loss of classifier-BCE dropout regularisation
— confounding the A3 delta (mixup vs dropout-disabling). A1 (nca) / A2 (supcon) **early-return before
mining** (`loss.py:43-64`) and never call `_manifold_mixup_bce`; the floor has no second post-mining
head forward inside `compute_loss`. So the confound is **A3-specific**, not shared with the paired
control. It fires in every A3 run, does not crash/NaN, and is invisible to a `_manifold_mixup_bce`
unit probe (model in train mode) or the §4.4 smoke — exactly the confound the codex gate exists to
catch. It contradicts the prereg's DEV-3 §11 binding claim that the mixup re-forward reproduces
"exactly the deployed align forward" (which runs in train mode, dropout ON).

---

## 2. The fix — surgical, inside `_manifold_mixup_bce` only (`src/model/loss.py`)

### 2.1 Chosen variant: restore train mode on EXACTLY the `nn.Dropout` submodules, restore prior modes

Per constraint 3 I audited every module type reachable by the mixup forward BEFORE choosing (E3
below). The head under the frozen `--batch_norm False` config contains only `Dropout / Linear / ReLU`
— **NO BatchNorm / InstanceNorm / LayerNorm / other running-stat module**. Under that audit the
whole-model `model.train(); …; model.train(was_training)` variant would ALSO be side-effect-free.
I nonetheless chose the **Dropout-only** variant because it is:

- **unconditionally correct** — it introduces zero running-stat side effect *even if* `--batch_norm`
  were ever flipped True (in which case a whole-model `model.train()` would spuriously double-update
  BatchNorm running stats during A3's second forward, since the floor's `loss.py:32` forward already
  updated them once). It does not depend on the audited invariant staying true.
- **minimally scoped to what mixup needs** — it enables exactly the dropout regularisation the
  deployed align forward applies, and nothing else.
- **an exact-prior-mode restore** — it snapshots each Dropout module's `.training` flag and restores
  it, so post-call behaviour is unchanged.

### 2.2 Exact diff (`git diff src/model/loss.py`)

```diff
@@ -706,6 +706,22 @@ def _manifold_mixup_bce(model, image_feats, text_feats, labels, args):
     mixup only regularises the classifier path. Pinned to fusion_mode == 'align'.
     """
     assert model.fusion_mode == "align", "A3 mixup is pinned to align (Hadamard) fusion"
+    # REFREEZE-1 (codex STOP; NCA_SUBMIT_RECORD.md §2.1): the upstream FAISS mining call
+    # (dense_retrieve_hard_negatives_pseudo_positive) put the model in EVAL mode via
+    # model.eval() (retrieval.py:330) and never restored it. Without the guard below this
+    # SECOND head forward would run with the head's Dropout submodules DISABLED
+    # (img_proj/text_proj/mlp; classifier.py:81-82,96,103), confounding the mixup BCE with
+    # dropout-off and contradicting the prereg's "reproduce the deployed align forward"
+    # (dropout ON) claim. Fix: restore train mode on EXACTLY the nn.Dropout submodules
+    # reachable by this forward, run the forward, then restore each module's exact prior
+    # mode. The head carries NO BatchNorm/running-stat module (frozen --batch_norm False;
+    # ncafam_family.sbatch:86), so touching only Dropout has zero running-stat side effect.
+    # This function is entered ONLY by A3 (mixup=True); the floor and A1/A2 never call it,
+    # so their model mode AND RNG streams are byte-untouched by this fix.
+    _mixup_dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
+    _mixup_prev_modes = [m.training for m in _mixup_dropouts]
+    for _m in _mixup_dropouts:
+        _m.train()
     img = nn.functional.normalize(model.img_proj(image_feats), p=2, dim=1)
     txt = nn.functional.normalize(model.text_proj(text_feats), p=2, dim=1)
     x = torch.mul(img, txt)                                      # fused post-projection rep
@@ -715,6 +731,8 @@ def _manifold_mixup_bce(model, image_feats, text_feats, labels, args):
     perm = torch.randperm(B, device=x.device)
     x_mix = lam * x + (1.0 - lam) * x[perm]
     logit = model.output_layer(model.mlp(x_mix))                # [B, 1]
+    for _m, _mode in zip(_mixup_dropouts, _mixup_prev_modes):
+        _m.train(_mode)
     y = labels.float().reshape(-1, 1)
     y_mix = lam * y + (1.0 - lam) * y[perm]
     if getattr(args, "pos_weight_value", None) is not None:
```

`git diff --stat`: `src/model/loss.py | 18 ++++++++++++++++++  1 file changed, 18 insertions(+)`.
Both hunks are **inside `_manifold_mixup_bce`**; nothing outside the mixup call path changed.
`python -m py_compile src/model/loss.py` = **COMPILE_OK**. No edit to `retrieval.py`, no edit to the
floor path, no edit to `run_rac.py` / `ncafam_family.sbatch` (shas B/C intact, §4).

### 2.3 Rationale per constraint 3 (running-stat audit → variant choice)

The E3 audit shows the mixup forward reaches only `Dropout / Linear / ReLU`. The task permits the
whole-model variant when no running-stat module exists; I selected the strictly-more-robust
Dropout-only variant so the guard remains correct independent of the `--batch_norm` flag and enables
exactly the dropout regularisation (not, e.g., a spurious BatchNorm running-stat update). Restore
loop reinstates each module's exact pre-call `.training` flag (E2 confirms both the model and every
dropout return to their pre-call `eval` state).

---

## 3. Equivalence evidence (CPU-only throwaway harness; verbatim output)

Harness (throwaway, NOT in repo):
`…/scratchpad/nca_refreeze_evidence.py`; pre-fix snapshot `…/scratchpad/loss_prefix.py`
(sha `e1244ada…`, the frozen pre-fix `loss.py`). `torch.manual_seed` fixed; synthetic tensors only.

### E1 — Floor-path bit-exactness (mixup=False, triplet hybrid INCL. FAISS mining), pre-fix vs post-fix

The deployed floor / A1 / A2 never enter `_manifold_mixup_bce`; the fix is textually confined to it.
E1 empirically confirms the full `compute_loss` floor path (including the eval-mode-leaking mining
call) is bit-identical pre/post-fix. Same model instance, same batch, same `train_dl`, RNG snapshot
restored between the two calls.

```
E1 floor-path (mixup=False, triplet hybrid incl. FAISS mining):
    total_loss       pre=0.8732562065124512  post=0.8732562065124512  bit-exact=True
    in_batch         pre=0.9493409395217896  post=0.9493409395217896  bit-exact=True
    hard             pre=0.9823682308197021  post=0.9823682308197021  bit-exact=True
    pseudo           pre=0.9820610880851746  post=0.9820610880851746  bit-exact=True
    loss_classifier  pre=0.6968642473220825  post=0.6968642473220825  bit-exact=True
E1 VERDICT: PASS (bit-exact)
```
(All five returned scalar components bit-exact. The `0.8732…` value is a coincidental synthetic
number, not the historic provenance datum; what matters is pre == post to the last bit.)

### E2 — Fix positive control (dropout ON during the mixup forward; mode restored after)

Model set to `eval()` to simulate the post-mining leaked state; forward-hooks on each `nn.Dropout`
record `module.training` at call time. Pre-fix = bug reproduction; post-fix = the guard.

```
--- PRE-FIX (bug reproduction: dropout OFF during mixup forward) ---
E2[pre-fix]: dropout forwards observed=5  all-in-train-during-forward=False  (modes during fwd=[False, False, False, False, False])
E2[pre-fix]: model.training  pre=False post=False  (restored=True)
E2[pre-fix]: dropout modes  pre=[False, False, False, False, False] post=[False, False, False, False, False]  (restored=True)
E2[pre-fix]: loss=0.6959605217  lam=0.821353
--- POST-FIX (dropout ON during mixup forward, mode restored) ---
E2[post-fix]: dropout forwards observed=5  all-in-train-during-forward=True  (modes during fwd=[True, True, True, True, True])
E2[post-fix]: model.training  pre=False post=False  (restored=True)
E2[post-fix]: dropout modes  pre=[False, False, False, False, False] post=[False, False, False, False, False]  (restored=True)
E2[post-fix]: loss=0.6962630153  lam=0.376669
E2 VERDICT: bug-present-prefix=True  fix-dropout-on-postfix=True  mode-restored-to-eval=True  -> PASS
```
Pre-fix: all 5 dropout forwards ran with `training=False` (bug confirmed). Post-fix: all 5 ran with
`training=True` (dropout ON), and both the model and every dropout returned to their pre-call `eval`
state. The pre/post loss & `lam` differ only because dropout-ON draws additional RNG during
`img_proj`/`text_proj` before the `Beta`/`randperm` draws — the intended behavioural change; A3 has
no prior baseline to match (zero test-touch), so the internal RNG order is free.

### E3 — Module audit (types reachable by the mixup forward)

```
E3 module-audit: types reachable by the mixup forward (img_proj/text_proj/mlp/output_layer):
    Dropout: 5
    Linear: 5
    ReLU: 2
    Sequential: 3
E3 running-stat (BatchNorm/InstanceNorm) modules present: 0  []
E3 layernorm present: False
E3 VERDICT: NO running-stat module -> Dropout-only guard is sufficient AND whole-model train() would also be side-effect-free
```
Corroborated statically: `--batch_norm False` is pinned in the frozen `ncafam_family.sbatch:86`
(and the anchor `enc3seed_lora_curric.sbatch:61`), and the `run_rac.py` argparse default is `False`
(`run_rac.py:175`), so the `nn.BatchNorm1d` branch (`classifier.py:100-101`) is never constructed.

**E1 PASS / E2 PASS / E3 PASS (no running-stat module).**

---

## 4. Shas — OLD → NEW, and B/C intact

| # | file | OLD sha256 (frozen `NCA_FREEZE.md`) | NEW sha256 (this fix) | status |
|---|---|---|---|---|
| A | `src/model/loss.py` | `e1244adadf16b47c24b05786d1ee4e153fd9c696e3be0924eae43c82f1c3b75b` | `2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b` | **CHANGED (this fix)** |
| B | `src/run_rac.py` | `b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3` | `b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3` | **UNCHANGED (byte-identical)** |
| C | `scripts/slurm/ncafam_family.sbatch` | `baf41be8a1264445d6bbc63d2eb54966abed6da33c52176ae62d93bf79c14b94` | `baf41be8a1264445d6bbc63d2eb54966abed6da33c52176ae62d93bf79c14b94` | **UNCHANGED (byte-identical)** |

Reused-unchanged machinery also re-verified byte-identical: `classifier.py`
`e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378`; `retrieval.py`
`d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57`.

---

## 5. RNG-stream statement (constraint 4)

`_manifold_mixup_bce` is entered **ONLY** by the A3 arm (`mixup=True`). The deployed **floor**
(`mixup=False`, `head_loss='triplet'`) and arms **A1 (nca)** / **A2 (supcon)** (which early-return at
`loss.py:43-64` **before** the mining call and before the hybrid block) **never call this function**,
so their model-mode and RNG streams are completely untouched by this fix — confirmed empirically by
E1 (floor path bit-exact) and structurally by the confined 18-insertion diff. Within A3 the fix DOES
change RNG consumption (dropout-ON now draws dropout RNG during `img_proj`/`text_proj`, shifting the
subsequent `Beta`/`randperm` draws — see E2). This is intended: A3 was never run (zero test-touch, §6
of `NCA_SUBMIT_RECORD.md`), so it has no prior trajectory to reproduce, and A3's arm-confined RNG
divergence is already pre-declared (prereg F0.2; `NCA_SUBMIT_RECORD.md` note 3). The A3 arm now
regularises its BCE path with dropout ON, exactly as the deployed align forward does.

---

## 6. Status / obligations before submission

Per prereg §4.5 ("Blocking findings ⇒ fix the code + re-freeze the shas (§5) + re-run this gate")
the fix is applied and the freeze block is re-issued (`NCA_FREEZE.md` REFREEZE-1). **Authorization is
VOID until:** (1) an **independent 0-context re-review** approves this fix + re-freeze, AND (2) the
**mandatory codex gate re-runs** clean on the patched `loss.py` (§4.5 three risk surfaces + the
now-fixed A3 dropout-mode). No SLURM submitted. Zero GPU / test-touch spent. No `state/` mutation.
Not pushed.
