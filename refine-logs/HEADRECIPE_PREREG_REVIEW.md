# HEAD-RECIPE (SAM + modality-dropout) — INDEPENDENT 0-CONTEXT PRE-REGISTRATION REVIEW

**Reviewer:** independent 0-context pre-registration reviewer (no prior project context; adversarial
mandate; zero user interaction; no job submitted; prereg NOT modified).
**Date:** 2026-07-25 NZST. **CPU-only** (no GPU/SLURM/Modal spent; `state/` not touched;
`autoresearch/goal_mllm_plus3/state/` unmodified).
**Target:** `refine-logs/HEADRECIPE_PREREG.md` (commit `83bb76e`; on-disk sha256
`68be61ac5ad77d2cfc66f3b355b574855bb426ed2c2a6e6b89f8050347ba232d`).
**Recon:** `refine-logs/HEADRECIPE_FORENSIC_RECON.md` (`44918e0`).
**House precedents:** `refine-logs/FRAME16_PREREG_REVIEW.md`, `refine-logs/READOUT_PREREG_REVIEW.md`.
**Method:** every load-bearing fact re-derived from primary artifacts on disk — the patch commit `83bb76e`
diff read line-by-line against `run_rac.py` / `classifier.py`; `compute_loss` (`loss.py`) and the FAISS
re-mine gate (`retrieval.py:341`) traced end-to-end; `build_model` args-threading confirmed; the sbatch
`run_one` python + readout blocks diffed against the `enc3seed_lora_hatemm.sbatch` anchor; the banked
13150/13241 trainlogs re-parsed with an **independently written** parser (not the prereg's embedded one);
the mod-dropout masking reproduced on a synthetic batch; every freeze-block hash recomputed; all collision
paths `ls`-checked on disk; the sbatch `bash -n`'d and the edited python `py_compile`d; the two disclosed
headwinds (F69, F45/F58) verified present in `state/findings.jsonl` (read-only). The prereg's and recon's
numbers were treated as untrusted until independently reproduced.

## VERDICT: **APPROVED-WITH-NOTES** (all four notes non-blocking)

The prereg is hash-integral, floor-faithful to 4dp (both protocols, both datasets, all per-seed values +
selected epochs), same-code-paired at the token/Namespace level, patch-correct on both the load-bearing SAM
re-mine-reuse invariant and the additive no-flag byte-identity, identity-fill-correct (ones, not zeros, per
the Hadamard-degeneracy argument), leakage/veto-clean, collision-free on disk, and its kill-ladder
(KS-arm-dead sign bar → FORMAL +0.030/+0.030 conjunct, both protocols, per arm×dataset) is fully decidable
from raw logs by a 0-context verdict reviewer with no interpretive freedom. Both novelty-nil recipes are
plainly D7-DEAD (a formal PASS is an engineering/ablation row, never a novelty win), both honest headwinds
are disclosed AND real (F69 grad-norm↔acc wrong-sign 3/3; F45/F58 text-carried ⇒ mod-dropout downside-
skewed), knobs are frozen (SAM `rho=0.05`, mod-dropout `p=0.3` + identity-fill), and the whole family is ONE
multiplicity bite. The four notes below are within-mechanism observations (mod-dropout also perturbs the
retrieval query; SAM clips the perturbed gradient; the mask-rate reference numbers are seed-specific; the
re-mine assert is conservatively over-broad) — none affects decidability, leakage, clobber-safety, or the
honesty of any bar, and none can manufacture an unsupported pass. **Cleared to freeze + single-submit**
(codex gate on the SAM branch first, per the prereg's own §4.5).

---

## Rationale (one paragraph)

The family measures the two remaining head-training-dynamics escape hatches — a flat-minima optimizer (SAM)
and a stream-dropout regularizer (modality-dropout) — over the **byte-identical banked LoRA feature caches**
(ZH `…-LoRA_HF`, HateMM `…-LoRA-curric_HF`), 3-seed paired within head-seed against each dataset's own banked
floor, dual-protocol. Its validity hinges on two properties, both of which hold under audit. **(1) No-flag
byte-identity:** every new path is `getattr(args,<flag>,default)`-gated OFF; with the flags absent the SAM
`else:` branch is byte-for-byte the pre-patch 5-line optimizer block (confirmed against `44918e0`) and the
classifier mod-dropout block is skipped by its `self.training ∧ mod_dropout ∧ align` gate, so a fresh no-flag
run reproduces the banked floors and they need no re-run; the diff's only "deletions" are trailing-whitespace
blank lines, and the 4 new argparse keys default inert. **(2) SAM re-mine-reuse:** the FAISS train index is
rebuilt only inside `compute_loss`→`retrieval.py:341` when `train_feats is None`; with
`reindex_every_step=False` it fires once per epoch (first step) at the unperturbed weights, and the SAM
branch threads the SAME `train_feats/train_labels` the first call returned into the second `compute_loss` at
`w+ε`, so the `:341` gate stays False and the index is **never** rebuilt at the perturbed weights — a runtime
`assert train_feats is not None and train_labels is not None` guards it and crashes loudly rather than
silently re-mining. Because `compute_loss` calls `model.train()` (loss.py:30) BEFORE its loss-bearing forward
(line 31) and retrieval's `model.eval()` (retrieval.py:330) fires only afterward, mod-dropout applies on the
correct forward and the eval/mining index is clean. All floors re-derive to 4dp, every freeze-block hash
matches disk, `build_model` already threads `args=args` into `classifier_hateClipper` (line 1203) so no
run_rac wiring is needed, and every collision path is verified absent. The grid is a hardcoded 12-row
`CONFIGS` array (no sweepable flag), the executor transcribes raw per-seed numbers with the verdict rendered
independently, and the knobs are frozen at one bite — so the motivated-executor attack surface (re-tune
rho/p, protocol/metric shop, bury a regression, clobber a floor, silently re-mine) is closed by
construction. Novelty is repeatedly and correctly stamped D7-DEAD.

---

## CHECK-BY-CHECK

### 1. PATCH CORRECTNESS — SAM two-step + re-mine-reuse invariant (LOAD-BEARING) — **PASS**

- **`_sam_ascend` (run_rac.py:551-573):** `params = [p for p in model.parameters() if p.grad is not None]`;
  `grad_norm = ‖stack([p.grad.detach().norm(2)])‖₂` = the correct **global** 2-norm (√Σ of per-param squared
  norms = √Σ over all grad elements); `scale = rho/(grad_norm+1e-12)`; `p.add_(p.grad.detach()*scale)` under
  `torch.no_grad()`; returns `[(p, ε)]`. Textbook Foret ascend, deterministic (no RNG). **`_sam_restore`**
  does `p.sub_(ε)` for the stored ε — an **exact** `w+ε → w` restore (uses the cached ε, not the new grad, so
  restore is correct even after the perturbed backward repopulates `.grad`).
- **SAM branch ordering (run_rac.py:675-709):** `assert train_feats/labels not None` → `zero_grad` →
  `total_loss.backward()` (grad at w) → `_sam_ascend` (→ w+ε) → **second `compute_loss` at w+ε REUSING the
  first call's `train_feats/train_labels`** → `zero_grad` → `total_loss_perturbed.backward()` (grad at w+ε) →
  `_sam_restore` (→ w) → `clip_grad_norm_` → `optimizer.step()` (AdamW update at w with the w+ε grad). Correct
  two-step SAM.
- **RE-MINE-REUSE INVARIANT — VERIFIED at BOTH the structural and assert level.** The second `compute_loss`
  call passes `train_feats=train_feats, train_labels=train_labels` — the exact (detached / numpy) objects the
  FIRST call returned (run_rac.py:642-663, then 668-674). `compute_loss`→`dense_retrieve_hard_negatives_…`→
  `retrieval.py:341` rebuilds the index **only** `if train_feats is None or train_labels is None`; the
  reused objects are non-None ⇒ the gate stays **False** ⇒ **no FAISS rebuild at w+ε**, byte-consistent with
  the once-per-epoch baseline. In the deployed config (`no_pseudo_gold_positives=1 > 0`, `hard_negatives_loss
  True`, `sparse_dictionary None`) the first `compute_loss` ALWAYS mines (loss.py:269-288) and returns
  non-None train_feats, so the assert holds; if a config ever skipped mining, the assert converts it to a loud
  crash rather than a silent re-mine at w+ε (conservative — see Note 4). The post-training E-step rebuild
  (`run_rac.py:1315-1329`) is EM-path-only (`seg_mode consensus/selfscore`); the deployed `seg_mode full`
  else-branch never reaches it.
- **Train/eval-mode trace:** `compute_loss` sets `model.train()` (loss.py:30) then does the loss-bearing
  forward (line 31); retrieval's unconditional `model.eval()` (retrieval.py:330) fires only afterward, so the
  loss-bearing forward is always in train mode and the mining index is always encoded in eval mode. The model
  being left in eval mode at `total_loss.backward()` time is **identical to the baseline** (the else branch
  backward-at-w has the same property) — SAM does not perturb this, and the second `compute_loss` re-sets
  `model.train()`. No confound.

### 2. PATCH CORRECTNESS — additive gating / no-flag byte-identity — **PASS**

- **`else:` branch byte-identical to the banked config.** `git show 44918e0:src/run_rac.py` optimizer block =
  `optimizer.zero_grad()` / `total_loss.backward()` / `clip_grad_norm_(model.parameters(), args.grad_clip)` /
  `optimizer.step()` — the post-patch `else:` branch (run_rac.py:710-715) is these exact 5 lines, only
  re-indented under `else:`. With `sam=False` the executed path is byte-identical.
- **4 argparse keys inert-by-default.** `--sam`/`--mod_dropout` parse `str(x).lower()=='true'` (default
  `False`); `--sam_rho 0.05` / `--mod_dropout_p 0.3` floats. Absent ⇒ 4 inert Namespace keys, zero behaviour
  diff — the established `--tarc_vote_gamma`-style additive pattern.
- **classifier `__init__` diff = 2 inert attributes + whitespace.** `self.mod_dropout = getattr(args,
  'mod_dropout', False)` / `self.mod_dropout_p = getattr(args,'mod_dropout_p',0.3)`; the "deleted" line is a
  trailing-whitespace blank. No behaviour change when unset. `build_model` already threads `args=args` into
  `classifier_hateClipper` (run_rac.py:1203) ⇒ DEV-4 confirmed, no run_rac wiring needed.
- **`py_compile` on both edited files = PASS; `bash -n` on the sbatch = SYNTAX_OK; loss.py / retrieval.py
  untouched** (shas `4879663…` / `d43e3bc…` re-verified). A no-flag run therefore differs from the banked
  floor Namespace ONLY by the 4 inert keys + derived `model`/`group_name`/`exp_comment` (§4.1b), exactly as
  the smoke §4.4.3 will confirm.

### 3. PATCH CORRECTNESS — classifier identity-fill — **PASS**

- **Ones-fill, not zeros** (classifier.py forward): `img_feats = where(drop_img, ones_like, img_feats)` /
  `text_feats = where(drop_text, ones_like, text_feats)`. The fusion at `classifier.py:141` is
  `x = torch.mul(img_feats, text_feats)` (Hadamard) — zero-fill would give `img⊙0 = 0` (degenerate
  bias-only), ones-fill gives `img⊙1 = img` (survivor passes through). The recon's degeneracy ruling is
  correct and the code obeys it.
- **At-most-one stream per sample:** `drop = rand(B) < p`, `coin = rand(B) < 0.5`, `drop_img = drop & coin`,
  `drop_text = drop & ~coin` ⇒ `drop_img ∧ drop_text` is always empty. Independently reproduced on n=200k:
  drop-rate 0.2988 ≈ p, img 0.1489 / text 0.1499 (fair split), **both-dropped == 0**. Invariants hold.
- **Does NOT fire at eval / does NOT alter the no-flag path:** gated
  `if self.training and getattr(self,'mod_dropout',False) and self.fusion_mode=='align'` — `model.eval()`
  (eval + the mining re-encode) skips it; `mod_dropout=False` skips it; non-align fusions skip it. Inserted
  AFTER L2-normalize (lines 118-119), BEFORE fusion (line 138+) — the pre-registered position.

### 4. FLOORS — **PASS (independently re-parsed; all 4dp-exact, both protocols)**

Re-parsed with a freshly written parser (val-sel = epoch ≥ warmup 5 max `Val_Retrieval` acc, roc tie-break →
that epoch's `Test_Retrieval`; final = max epoch):

| leg | protocol | s0 acc/mF1 | s1 acc/mF1 | s2 acc/mF1 | mean acc/mF1 | prereg |
|---|---|---|---|---|---|---|
| ZH 13150 (generic-LoRA) | val-sel (ep 20/26/19) | 0.8322/0.8023 | 0.8255/0.7956 | 0.8389/0.8065 | 0.8322/0.8015 | ✓ |
| ZH 13150 | final (ep 29) | 0.8456/0.8181 | 0.8389/0.8113 | 0.8523/0.8226 | 0.8456/0.8173 | ✓ |
| HateMM 13241 (curric-LoRA) | val-sel (ep 29/14/10) | 0.8791/0.8730 | 0.8744/0.8678 | 0.8791/0.8724 | 0.8775/0.8711 | ✓ |
| HateMM 13241 | final (ep 29) | 0.8791/0.8730 | 0.8791/0.8724 | 0.8791/0.8724 | 0.8791/0.8726 | ✓ |

Every per-seed value, selected epoch, and 3-seed mean bit-matches §2.1/§2.2 to 4dp. The §2.3 promote
thresholds are arithmetically correct (ZH val-sel +0.030 = {0.8622, 0.8555, 0.8689}; final = {0.8756,
0.8689, 0.8823}). ZH is correctly flagged the marginal target; HateMM the near-ceiling hold (+0.030 ≈ 0.909,
thin surface).

### 5. BARS + HONESTY — **PASS**

- **FORMAL (§3.2):** +0.030 acc AND +0.030 mF1, 3/3 seeds positive, BOTH protocols vs the arm's own floor —
  quoted verbatim from `exp-encoder-3seed.md:73-85` (per-seed δ; 3-seed mean±std + sign; n=3 paired-t as
  effect-size descriptor only / no significance claim; pass = the conjunct; both protocols judged
  independently). Fully decidable, no protocol/metric shopping.
- **KS-arm-dead (§3.3):** sign-based — KILLED iff on BOTH protocols `mean Δacc ≤ 0` OR acc sign not 3/3
  positive. This is the frame16 DEV-1 discipline (house n=3 = NO bootstrap); decidable from raw per-seed
  numbers; can only ever KILL, never fabricate a pass. KS-regression (§3.4) at mean Δacc ≤ −0.014 (the CAND2
  §2.3 head-seed spread) is the pre-declared expected direction for ARM B.
- **Honesty clauses present and REAL.** F0.3 D7-DEAD (both recipes generic training knobs — never a novelty
  win). F0.5 discloses two headwinds that LOWER the prior, both verified in `state/findings.jsonl`:
  (a) F69 grad-norm↔acc Spearman **+0.61/+0.72/+0.62 wrong-sign 3/3** (present) — flatter ≠ better on this
  head, correctly ruled NOT a ban (F69 scope = checkpoint selection, not a training optimizer) but a genuine
  headwind; (b) F45 (ZH gain text-carried) present ⇒ dropping the carrying stream ~15% of samples is
  downside-skewed on both text-carried targets. F0.6 (the re-mine invariant), F0.7 (additive gating), F0.8
  (identity-fill magnitude wrinkle under `batch_norm=False`) all accurate.
- **Knobs frozen, one bite (§3.6):** SAM `rho=0.05` / mod-dropout `p=0.3` + identity-fill pinned; any
  rho/p/zero-fill/second-p touch is a NEW bite. ONE sbatch = ONE family = ONE bite whether one or both arms
  survive. Single test-touch (§4/F0.1): the 12 head reads (4 arm×dataset cells × 3 seeds) are the ONLY
  budgeted evaluations; zero test-touch before the verdict; a surviving cell still owes the full ceremony.

### 6. SMOKE + CODEX GATE — **PASS**

- **Smoke (§4.4) proves the two load-bearing things.** ARM A: 1-seed 3-epoch throwaway with `--sam True` on
  the ZH cache → loss finite, completes, the **re-mine-reuse assert does NOT trip**, and the SAM double-step
  is VISIBLE (per-epoch wall-time meaningfully above the flag-off baseline = the second forward-backward).
  ARM B: 3-epoch throwaway (loss finite, completes) + the $0 CPU mask-rate line (drop≈0.30, img/text≈0.15,
  both==0, eval-gate off). §4.4.3: a NO-flag run dumps `vars(args)` and confirms the Namespace differs from
  the floor ONLY by the 4 inert keys + `model`/`group_name`/`exp_comment` (with the optional 1-seed bit-exact
  floor reproduction as a settle). All throwaways use the `_smoke_hr` group / `hr_smoke_*` logs and are
  **deleted** so they never persist into the §4.3 collision surface.
- **Codex gate (§4.5) pinned pre-submit,** iterative until Claude+Codex agree, focused on the SAM
  double-step + re-mine-reuse (global grad-norm, ε scale, in-place add/sub under `no_grad`, exact restore,
  block ordering, and the F0.6 invariant/assert). Blocking findings ⇒ fix + re-freeze A/B/C shas + re-run the
  gate. Correctly load-bearing.

### 7. COLLISION / SUBMIT — **PASS**

- **Collisions ABSENT on disk (re-check at submit):** `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_headrecipe*`
  = none; `slurm/logs/hr_*.trainlog` = none; `_smoke_hr` groups/logs = none. Banked caches + floor trainlogs
  (13150/13241) are read-only inputs; a fresh `RAC_video_headrecipe` group + arm-tagged
  `exp_comment`/trainlog keep every write distinct, and `--force False` would abort (never overwrite) if a
  path ever pre-existed. The `hr_${ARM}_…` prefix guarantees ARM A and ARM B (same dataset/seed) never
  collide.
- **Same-code.** The `run_one` python invocation is token-identical to `enc3seed_lora_hatemm.sbatch` with
  EXACTLY two deltas — `--exp_comment "_${MODEL}"→"_${MODEL}_${ARM}"` (derived-inert) and trailing
  `${ARM_FLAGS}` (additive-gated) — and the readout `PY` block is **BYTE-IDENTICAL** (`diff` empty). The
  hardcoded `CONFIGS` word-splits to exactly 12 rows, each `run_one` receiving `(DATASET, MODEL, SEED, ARM,
  ARM_FLAGS…)` correctly (SAM rows get `--sam True --sam_rho 0.05`, MODDROP rows get `--mod_dropout True
  --mod_dropout_p 0.3`).
- **Resource plan.** ONE sbatch, 12 runs sequential, `--cpus-per-task=8 --mem=64G --gres=gpu:a100:1` ⇒ peak
  8 CPU / 64 G / 1 GPU — within the 16/128/2 cap and never two 16-CPU jobs (the 29h-wedge rule). **NO
  `--time`** (L8: "intentionally NO --time"). `conda activate HateVideo`; `PENDING (JobHeldUser)` → **wait
  for auto-release, never force** (§6). Sources `conda.sh`, runs `disk_guard.sh`. ~8 min wall, < 0.15 GPU-h.

### 8. DEVIATIONS §11 (DEV-1..DEV-7) — all favorable / neutral / documented

- **DEV-1** (run_one not byte-identical — adds `${ARM_FLAGS}` + arm-tagged `exp_comment`/trainlog) —
  **NECESSARY/documented.** A flagged family cannot be byte-identical; resolution = token-identical base + the
  two blessed deltas (verified) + byte-identical readout `PY`. Same-code holds at the Namespace level; the
  arm tag is REQUIRED to keep ARM A/B logs distinct. OK.
- **DEV-2** (KILL bar = SIGN, not bootstrap-CI) — **FAVORABLE.** Pins the house n=3 no-bootstrap discipline
  (same call the FRAME16 review ruled favorable); only the significance formalism changes; can only kill.
- **DEV-3** (patches EDIT run_rac.py + classifier.py in place) — **recon-mandated, same-code preserved.**
  Every edit `getattr`-gated OFF; else-branch byte-identical; classifier "deletions" whitespace-only; shas
  hash-frozen. Verified.
- **DEV-4** (no `build_model` change needed) — **neutral/favorable, verified** (args already threaded at
  run_rac.py:1203).
- **DEV-5** (SAM invariant enforced by runtime `assert` AND code structure) — **favorable, verified** (both
  present).
- **DEV-6** (mask-rate as $0 CPU synthetic check, not a hot-path log) — **favorable, verified** (hot path
  stays clean; invariants independently reproduced).
- **DEV-7** (both arms + both datasets one bite; HateMM=curric 13241, ZH=generic 13150) — **documented,
  matches the task pinning** (each dataset's deployed/best cache; ZH marginal target, HateMM near-ceiling
  hold).

---

## NON-BLOCKING NOTES (for the executor / verdict reviewer)

1. **Mod-dropout perturbs the retrieval query as well as the classifier output.** The `feats` produced by the
   train-mode forward (loss.py:31, mod-dropout applied) is the query fed into hard-negative mining
   (loss.py:254). So ARM B's regularization touches the retrieval/triplet loss, not only the final head
   logits. This is **within-mechanism** (the recon frames mod-dropout as "shaping the head's learned
   weights", and the whole training step operating on the dropped representation is the intended regularizer);
   the mining INDEX is still clean (encoded under `model.eval()`). Recorded so the verdict reviewer knows the
   knob's reach. Non-material.

2. **SAM clips/steps on the PERTURBED gradient**, whereas the baseline `else:` branch clips the w-gradient.
   This is standard SAM (the update uses the w+ε grad, and `clip_grad_norm_` runs after
   `total_loss_perturbed.backward`), and `grad_clip=0.1` therefore acts on the ascended grad. Inherent to the
   treatment definition, not a defect; the FORMAL/KS bars are outcome-based so it does not affect
   decidability. Informational.

3. **The §4.4.2 mask-rate reference numbers (0.2953 / 0.1525 / 0.1427 / 0) are one seed's draw**, not a fixed
   target. My independent reproduction (0.2988 / 0.1489 / 0.1499 / 0) confirms the load-bearing invariants
   (drop-rate ≈ p, fair img/text split, both-dropped ≡ 0, eval-gate off) but differs in the exact fractional
   values, as expected for a finite stochastic sample. Non-material.

4. **The re-mine assert is conservatively over-broad.** `assert train_feats is not None and train_labels is
   not None` would also crash a hypothetical `no_hard_negatives=0 ∧ no_pseudo_gold_positives=0` SAM run (where
   mining is skipped entirely and there is no re-mine to prevent). The deployed config always mines, so the
   assert never trips here; and crashing loudly is the SAFE direction. The gate can only ever block, never
   fabricate a pass. Informational.

---

## Reviewer's independent ruling on the two named invariants (≤3 sentences)

The **SAM re-mine-reuse invariant HOLDS**: the second `compute_loss` at `w+ε` is passed the exact non-None
`train_feats/train_labels` the first call returned, so the `retrieval.py:341` (`train_feats is None`) rebuild
gate stays False and the FAISS index is never rebuilt at the perturbed weights — guarded by both the threading
structure and a runtime assert that crashes loudly rather than silently re-mining (and in the deployed
`no_pseudo_gold_positives=1` config the first call always mines, so the assert is satisfied). The **no-flag
byte-identity HOLDS**: with `--sam`/`--mod_dropout` absent, the SAM `else:` branch is the pre-patch
5-line optimizer block verbatim (confirmed against `44918e0`) and the classifier mod-dropout block is skipped
by its `self.training ∧ mod_dropout ∧ align` gate, so a fresh no-flag run reproduces the banked 13150/13241
floors and they need no re-run. **Cleared to freeze + single-submit** after the pre-submit codex gate on the
SAM branch.

## HASH-FREEZE

Recorded in `refine-logs/HEADRECIPE_FREEZE.md` (prereg NOT modified, per review mandate). All freeze-block
shas re-verified on disk at freeze time and **match**: prereg self-sha `68be61ac…`, A `1012c9e3…`
(run_rac.py), B `e7b61df4…` (classifier.py), C `c88f685f…` (headrecipe_family.sbatch); reused-unchanged
`loss.py 48796638…`, `retrieval.py d43e3bc4…`, anchor `enc3seed_lora_hatemm.sbatch 19c76b17…`.

**Reviewer statements:** ZERO GPU/SLURM/Modal spent — CPU-only login-node re-parse of the banked 13150/13241
trainlogs with an independently written parser, plus a synthetic-tensor mask-rate reproduction, `sha256sum` /
`py_compile` / `bash -n` / `ls` collision checks and a read-only `findings.jsonl` headwind check (seconds);
no held-out test metric produced; `state/` and `autoresearch/goal_mllm_plus3/state/` not touched; the prereg
was **NOT** modified; no job submitted; not pushed. Cloud/external numbers were never mixed with local
G-repro numbers.
