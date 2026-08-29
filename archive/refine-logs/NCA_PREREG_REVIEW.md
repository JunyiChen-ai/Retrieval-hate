# NCA / soft-kNN HEAD-LOSS family — INDEPENDENT 0-CONTEXT PRE-REGISTRATION REVIEW

**Reviewer:** independent 0-context pre-registration reviewer (no prior project context; adversarial
mandate; zero user interaction; no job submitted; prereg NOT modified).
**Date:** 2026-07-25 NZST. **CPU-only** (no GPU/SLURM/Modal spent; `state/` not touched;
`autoresearch/goal_mllm_plus3/state/` unmodified).
**Target:** `refine-logs/NCA_PREREG.md` (commit `9a9f4fe`; on-disk sha256
`7607863c15bef1d40c1f1f0a5b980123dd84aba260ced6db1fa993f277db5591`; on-disk == committed, unmodified).
**Recon:** `refine-logs/NCA_FORENSIC_RECON.md` (`685df9e`, the GO recon).
**House precedents:** `refine-logs/HEADRECIPE_PREREG_REVIEW.md`, `refine-logs/FRAME16_PREREG_REVIEW.md`.
**Method:** every load-bearing fact re-derived from primary artifacts on disk — the prereg's committed
loss.py / run_rac.py diff (commit `9a9f4fe`) read line-by-line against the live `src/model/loss.py`,
`src/run_rac.py`, and `src/model/classifier.py::classifier_hateClipper.forward`; the three risk surfaces
(LOO self-mask, bank stop-grad, mixup λ) independently exercised by importing the ACTUAL
`_nca_head_loss`/`_supcon_head_loss`/`_manifold_mixup_bce` on synthetic tensors; the banked 13150/13241
trainlogs re-parsed with an **independently written** parser (not the prereg's embedded one); the sbatch
diffed token-by-token against its `enc3seed_lora_curric.sbatch` anchor and the readout heredoc hashed for
byte-identity; every freeze-block hash recomputed; all collision paths `ls`-checked on disk; the sbatch
`bash -n`'d and both edited files `py_compile`d. The prereg's and recon's numbers were treated as untrusted
until independently reproduced.

## VERDICT: **APPROVED-WITH-NOTES** (all four notes non-blocking)

The prereg is hash-integral (all A/B/C + reused-unchanged shas match disk), floor-faithful to 4dp (both
protocols, both datasets, all per-seed values + selected epochs + 3-seed means), same-code-paired at the
token/Namespace level (the `run_rac.py` invocation differs from the anchor by exactly the two blessed deltas
and the readout heredoc is byte-identical), and patch-correct on all three load-bearing invariants — the
LOO self-exclusion-by-id (`−inf` mask, self-mass driven to exactly 0 even at maximal self-similarity), the
belt-and-suspenders bank stop-grad (a hostile `requires_grad=True` bank receives exactly zero gradient while
the anchor grad flows), and the additive no-flag byte-identity (the only loss.py deletions are the 6 BCE
lines re-emitted verbatim under the new `else:`, and run_rac.py is purely additive) so the banked floors need
NO re-run. The F66-non-binding ruling is honestly stated and paired with law-I as the disclosed +3
counter-pressure (P(≥+3)=2–4%, "NOT an expected-+3 bet", D7-DEAD) — the prereg promises no lift and keeps the
diagnostic framing in the open. The kill-ladder (KS-arm-dead sign bar → FORMAL +0.030/+0.030 conjunct, both
protocols, per arm×dataset) is fully decidable from raw logs by a 0-context verdict reviewer with no
interpretive freedom; the τ grid is pre-declared with both values reported and no winner-reselect; the family
is ONE multiplicity bite with knobs frozen; the 24 reads (8 arm×dataset cells × 3 seeds) are the ONLY
budgeted test-touch and the test-not-virgin list is honest. The four notes below are within-mechanism /
descriptor observations (an insertion-count mislabel, a conservative over-statement of A2's RNG divergence,
the disclosed treatment-arm RNG divergence for A1/A3, and A3's second classifier forward) — none affects
decidability, leakage, clobber-safety, hash integrity, or the honesty of any bar, and none can manufacture an
unsupported pass. **Cleared to freeze + single-submit** (codex gate on the NCA/SupCon/mixup branches first,
per the prereg's own §4.5).

---

## Rationale (one paragraph)

The family measures the one genuinely un-enumerated axis — a head-training objective that directly optimizes
the deployed top-20 signed-cosine kNN vote (NCA τ∈{0.1,0.2}, neighborhood-SupCon, manifold mixup) — over the
**byte-identical banked LoRA feature caches** (ZH `…-LoRA_HF`, HateMM `…-LoRA-curric_HF`), 3-seed paired
within head-seed against each dataset's own banked floor, dual-protocol. Its validity hinges on four
properties, all of which hold under audit. **(1) The surrogate lives in the vote's space.** Both the anchor
`feats` and the bank `f` are the retrieval embedding `embed = mlp[:-2](x)` returned by
`model(...,return_embed=True)` — the exact space `metrics.compute_metrics_retrieval` votes over — so the NCA
softmax pulls anchors toward same-class neighbours in the space the decision reads. **(2) LOO by id is
correct.** `_build_nca_bank` asserts the train ids are unique, every batch anchor is a train-split point and
therefore in the bank, and `logits[arange(B), own_rows] = −inf` masks exactly the anchor's own row before the
softmax; the loss is training-only (`compute_loss` is never called on the eval path, which routes through
`retrieve_evaluate_RAC_`), so dev/test queries never touch the mask. My import-level smoke drove an anchor to
equal its own bank row and the retained self-softmax-mass was exactly 0.000e+00. **(3) The bank is
stop-grad'd twice** (`no_grad`+`.detach()` at construction, local `bank_feats.detach()` in the loss); a
hostile `requires_grad=True` bank received a gradient of exactly 0.000e+00 while the anchor grad flowed
(0.264), and the loss decreased 0.4705→0.1795 under SGD. **(4) Flag-off is byte-identical.** With no new
flags `head_loss` defaults `'triplet'` (early branch skipped), `mixup` defaults `False` (the hybrid hook's
`else` re-emits the deployed 6-line BCE verbatim — git confirms those are the only 6 deletions), `nca_bank`
stays `None` (built only for `head_loss=='nca'`, otherwise unused), and the A1/A2 early return keeps
`train_feats/train_labels` `None` all epoch with the downstream `torch.is_tensor` guard (run_rac.py:738)
skipping the detach — so the FAISS mining is inert and nothing breaks. All floors re-derive to 4dp, every
freeze hash matches disk, the sbatch requests 8-CPU/64G/1-GPU with NO `--time`, every collision path is
verified absent, and the executor transcribes raw per-seed numbers with the verdict rendered independently —
so the motivated-executor attack surface (re-tune τ/α, protocol/metric shop, bury a regression, clobber a
floor, silently mine at the perturbed weights) is closed by construction. Novelty is repeatedly and correctly
stamped **D7-DEAD**.

---

## CHECK-BY-CHECK

### 0. HASH INTEGRITY — **PASS**

| role | file | freeze-block sha | disk sha | match |
|---|---|---|---|---|
| A | `src/model/loss.py` | `e1244ada…` | `e1244ada…` | ✓ |
| B | `src/run_rac.py` | `b85eb72a…` | `b85eb72a…` | ✓ |
| C | `scripts/slurm/ncafam_family.sbatch` | `baf41be8…` | `baf41be8…` | ✓ |
| reused | `src/model/classifier.py` | `e7b61df4…` | `e7b61df4…` | ✓ |
| reused | `src/utils/retrieval.py` | `d43e3bc4…` | `d43e3bc4…` | ✓ |
| anchor | `scripts/slurm/enc3seed_lora_curric.sbatch` | `00d9e995…` | `00d9e995…` | ✓ |

Prereg self-sha `7607863c…` (on-disk == committed `9a9f4fe`, zero-diff). Commit `9a9f4fe` touches exactly the
4 declared files (loss.py, run_rac.py, ncafam_family.sbatch, NCA_PREREG.md).

### 1. LOSS-CODE CORRECTNESS (LOAD-BEARING) — **PASS**

- **(a) NCA LOO by id — VERIFIED.** `_build_nca_bank` (run_rac.py:609-639) returns `id_to_row` and asserts
  `len(id_to_row) == bank_feats.shape[0]` (train ids unique ⇒ mask well-defined). `_nca_head_loss` builds
  `own_rows = [id_to_row[v] for v in batch_ids]` and sets `logits[arange(B), own_rows] = −inf`, masking
  **exactly** the anchor's own row (one entry per anchor; ids unique ⇒ full self-exclusion). Every batch id is
  a train-split anchor and the bank is a full pass over `train_dl`, so `id_to_row[v]` never KeyErrors; if the
  batch/bank ever disagreed it would raise loudly rather than mis-mask. `compute_loss` is invoked ONLY in the
  `model_pass` training loop — the eval/selection path (`retrieve_evaluate_RAC_`, run_rac.py:813+) never calls
  it, so the LOO mask is training-only and dev/test are untouched. **Independent smoke:** with an anchor set
  equal to its own bank row (maximal self-similarity), the retained self-softmax-mass was `0.000e+00`.
- **(b) Bank stop-grad, both layers — VERIFIED.** Construction: `model.eval()` + `torch.no_grad()` +
  `f.detach()` (run_rac.py:609-639). Loss: local `bank_feats = bank_feats.detach()` (loss.py:651). Anchor
  `feats` (= the grad-on retrieval embedding from the loss-bearing forward) is normalized to `q`; `k` is
  normalized from the detached bank ⇒ `logits = q @ k.t()/τ` grads only to `q`. **Independent smoke:** a
  hostile bank with `requires_grad=True` received grad-norm `0.000e+00`; anchor grad-norm `2.636e-01` (flows).
- **(c) Soft-vote surrogate direction — VERIFIED.** `logits = cos/τ`, `log_softmax` over the bank, same-class
  `logsumexp`, `−logP.mean()`; higher similarity ⇒ higher softmax weight ⇒ (if same-class) higher `P_i` ⇒
  lower loss ⇒ the gradient pulls the anchor toward same-class bank points. Sign/temperature semantics
  correct. **Independent smoke:** loss decreased `0.4705→0.1795` over 20 SGD steps. The `−inf` self/other-class
  masks and the `clamp(min=−30)` all-`−inf` guard produced no NaN (the guard is unreachable at N≫1 binary).
- **(d) A3 mixup — VERIFIED.** `_manifold_mixup_bce` re-derives the align forward
  `x = norm(img_proj(img)) ⊙ norm(text_proj(text))` — byte-for-byte the `classifier_hateClipper.forward`
  align path with `mod_dropout` OFF (confirmed against classifier.py:115-150) — mixes `x` and the label with a
  single per-batch `λ ~ Beta(2,2)` and a random permutation, forwards `mlp`+`output_layer`, and BCEs the mixed
  logit against the mixed soft label. The triplet term above is untouched (real un-mixed `feats`), so the kNN
  memory reads real neighbours and mixup regularises the classifier path only. `classifier.py` is untouched
  (sha `e7b61df4…`). **Independent smoke:** `λ=0.590∈[0,1]`, BCE finite, param-grads flow.
- **(e) Flag-off byte-identity — VERIFIED.** `head_loss` default `'triplet'` ⇒ the early branch
  (loss.py:42-64) is skipped; `mixup` default `False` ⇒ the hybrid hook's `else` runs the deployed BCE; the
  git diff's only 6 loss.py deletions ARE those BCE lines, re-emitted verbatim (+4 indent) under the new
  `else:` (confirmed by `git show 9a9f4fe`). run_rac.py is `68/0` (purely additive); `nca_bank` defaults
  `None` and is unused off-path. `python -m py_compile src/model/loss.py src/run_rac.py` = PASS.
- **(f) A1/A2 early-return / mining inert — VERIFIED.** The `head_loss∈{nca,supcon}` branch returns the
  7-tuple `(total_loss, _zero, _zero, _zero, loss_classifier, train_feats, train_labels)` **before** the FAISS
  mining, so `dense_retrieve_hard_negatives_pseudo_positive` is never entered and `train_feats/train_labels`
  pass through as `None`. Downstream, run_rac.py:735-741 guards the detach with `torch.is_tensor(...)`, so the
  `None` values are skipped (no crash); the `if step % log_interval` logging reads the `_zero` tensors via
  `.item()` (= 0.0); the SAM branch is bypassed (`--sam` False → else at run_rac.py:778). The 7-tuple shape
  matches the normal return, and nothing downstream consumes the (absent) mined pairs.

### 2. FLOORS — **PASS (independently re-parsed; all 4dp-exact, both protocols)**

Re-parsed with a freshly written parser (val-sel = epoch ≥ warmup 5 max `Val_Retrieval` acc, roc tie-break →
that epoch's `Test_Retrieval`; final = max epoch):

| leg | protocol | s0 acc/mF1 | s1 acc/mF1 | s2 acc/mF1 | mean acc/mF1 | prereg |
|---|---|---|---|---|---|---|
| ZH 13150 (generic-LoRA) | val-sel (ep 20/26/19) | 0.8322/0.8023 | 0.8255/0.7956 | 0.8389/0.8065 | 0.8322/0.8015 | ✓ |
| ZH 13150 | final (ep 29) | 0.8456/0.8181 | 0.8389/0.8113 | 0.8523/0.8226 | 0.8456/0.8173 | ✓ |
| HateMM 13241 (curric-LoRA) | val-sel (ep 29/14/10) | 0.8791/0.8730 | 0.8744/0.8678 | 0.8791/0.8724 | 0.8775/0.8711 | ✓ |
| HateMM 13241 | final (ep 29) | 0.8791/0.8730 | 0.8791/0.8724 | 0.8791/0.8724 | 0.8791/0.8726 | ✓ |

Every per-seed value, selected epoch, and 3-seed mean bit-matches §2.1/§2.2 to 4dp. The §2.3 promote
thresholds are arithmetically correct (ZH val-sel +0.030 acc = {0.8622, 0.8555, 0.8689}, mF1 = {0.8323,
0.8256, 0.8365}; ZH final acc = {0.8756, 0.8689, 0.8823}; HateMM +0.030 ≈ 0.909 everywhere). ZH is correctly
flagged the marginal target; HateMM the near-ceiling hold.

### 3. F66 NON-BINDING + law-I COUNTER-PRESSURE — **PASS (honest, present)**

The prereg states the ruling plainly (§0 claim scope, F0.5(a), §3.5): F66's β-decomposition bounds
*inference-side symmetric* operators over a **fixed** φ₀ Gram matrix (ISR / re-agg / vote-reweighting) to
+0.001–0.006; an NCA/soft-kNN *training* loss reshapes φ₀→φ′ (a different Gram matrix, oracle, and
symmetric/selection split — objects F66 never measured), so the cell is legitimately **un-measured, not
F66-dead**. The honest counter-pressure is disclosed and drives the prior down, not up: law-I (the 8-instance
"better representation ⇒ zero vote conversion" pattern) warns the ceiling may live in the frozen features, so
**P(≥+3)=2–4%**, the family is explicitly "**NOT an expected-+3 bet**", and even a formal PASS is **D7-DEAD**
(a generic training objective, never a novelty win). The diagnostic framing (NCA is the one operator that
discriminates "wrong objective" from "feature ceiling") is present and the prereg promises no lift. Honest.

### 4. BARS + HONESTY — **PASS**

- **FORMAL (§3.2):** +0.030 acc AND +0.030 mF1, 3/3 seeds positive, BOTH protocols vs the arm's own floor —
  quoted verbatim from `exp-encoder-3seed.md:73-85` (per-seed δ; 3-seed mean±std + sign; n=3 paired-t as
  effect-size descriptor only / no significance claim). Fully decidable; both protocols judged independently
  (fixed §7.2 write-up "final-epoch: pass/fail; val-selected: pass/fail"); no protocol/metric shopping.
- **KS-arm-dead (§3.3):** sign-based — KILLED iff on BOTH protocols `mean Δacc ≤ 0` OR acc sign not 3/3
  positive (the frame16 DEV-1 house n=3 no-bootstrap discipline). Decidable from raw per-seed numbers; can only
  KILL, never fabricate a pass. KS-regression (§3.4) at mean Δacc ≤ −0.014 (the CAND2/HEADRECIPE head-seed
  spread) is a within-frame note.
- **Multiplicity (§3.6):** ONE sbatch = ONE family = ONE bite whether one or all four arms survive; the τ grid
  (A1a/A1b) is the ONLY pre-declared multiplicity, both values reported with the per-arm pass rule and **no
  winner-reselect-and-rerun**; knobs frozen (τ∈{0.1,0.2}, SupCon τ=0.1, α=2.0, ce_weight=0.5 via the shared
  default, bank-detach, LOO-by-id, per-epoch rebuild); any re-tune / NCA-only / ProxyNCA++ / in-batch-grad /
  hard-top20 is a NEW bite.
- **Single test-touch (F0.1):** 24 head reads = 8 arm×dataset cells × 3 seeds = the ONLY budgeted NCA-family
  evaluations; zero test-touch before the verdict; a surviving cell still owes the full ceremony. The
  test-not-virgin list (frozen-CLIP, frozen-Qwen-8f, generic-LoRA, curriculum-LoRA, the LoRA-HateMM verdict,
  frame16, head-recipe) is honest — these are re-measurements under the identical `enc3s` protocol.

### 5. SMOKE + CODEX GATE — **PASS**

- **Smoke (§4.4)** records the CPU synthetic checks (NCA finite+decreasing, hostile-bank grad 0, LOO self-mass
  0, SupCon finite, mixup λ∈[0,1]) — all independently reproduced by me at the import level against the ACTUAL
  functions — plus the GPU pre-submit plan (per-arm 1-seed 3-epoch throwaway on the ZH cache: finite+decreasing
  loss, completes, A1 builds the bank and does NOT enter FAISS mining, A3 fires mining once/epoch, the LOO /
  bank-detach / mixup-λ asserts do not trip) and the no-flag Namespace-equivalence check. All throwaways use
  the `_smoke_nca` group / `nca_smoke_*` logs and are **deleted** so they never persist into §4.3.
- **Codex gate (§4.5)** is pinned pre-submit, iterative until Claude+Codex agree, focused on the three risk
  surfaces (LOO indexing, grad flow / `log_softmax`+`logsumexp` numerics + A3's re-derived fusion, per-epoch
  bank cadence). Blocking findings ⇒ fix + re-freeze A/B shas + re-run the gate. Correctly load-bearing.

### 6. COLLISION / SUBMIT — **PASS**

- **Collisions ABSENT on disk (re-check at submit):** `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_ncafam*` =
  none; `slurm/logs/nca_*.trainlog` = none; `_smoke_nca` group / `nca_smoke_*` = none. The 6 banked caches
  (ZH `…-LoRA_HF.pt`, HateMM `…-LoRA-curric_HF.pt`) and floor trainlogs (13150/13241) are read-only inputs; a
  fresh `RAC_video_ncafam` group + per-arm `exp_comment` + `nca_${ARM}_` trainlog prefix keep every write
  distinct (arms never collide with banked runs or each other), and `--force False` would abort (never
  overwrite) if a path ever pre-existed.
- **Same-code.** The `python ./src/run_rac.py …` invocation is token-identical to `enc3seed_lora_curric.sbatch`
  with EXACTLY two deltas — `--exp_comment "_${MODEL}"→"_${MODEL}_${ARM}"` (derived-inert) and trailing
  `${ARM_FLAGS}` (additive-gated) — and the readout `PY` heredoc is **BYTE-IDENTICAL** (block sha256
  `267a1505…` for both). `--ce_weight` is passed by neither (both use the run_rac default 0.5). The hardcoded
  `CONFIGS` word-splits to exactly **24 rows**, each `run_one` receiving `(DATASET, MODEL, SEED, ARM,
  ARM_FLAGS…)` correctly (`shift 4; ARM_FLAGS="$*"`; A3's multi-word `--mixup True --mixup_alpha 2.0` splits
  cleanly). `bash -n` = SYNTAX_OK.
- **Resource plan.** ONE sbatch, 24 runs sequential, `--cpus-per-task=8 --mem=64G --gres=gpu:a100:1` ⇒ peak
  8 CPU / 64 G / 1 GPU — within the 16/128/2 cap and never two 16-CPU jobs (the 29h-wedge rule). **NO
  `--time`** (L8: "intentionally NO --time"). `conda activate HateVideo`; `PENDING (JobHeldUser)` → **wait for
  auto-release, never force** (§6). Sources `conda.sh`, runs `disk_guard.sh`, B2-pushes `logging` only
  (derived artifacts; videos never leave). ~20 min wall, ~0.33 GPU-h.

### 7. DEVIATIONS §11 (DEV-1..DEV-7) — all favorable / neutral / documented

- **DEV-1** (KILL bar = SIGN, not bootstrap-CI) — **FAVORABLE.** Pins the house n=3 no-bootstrap discipline
  (same call the FRAME16 + HEADRECIPE reviews ruled favorable); only the significance formalism changes; can
  only kill.
- **DEV-2** (patches EDIT loss.py + run_rac.py in place) — **recon-mandated, same-code preserved.** Every edit
  `getattr`/`head_loss`-gated OFF; the mixup hook's `else` is byte-identical (the only 6 loss.py deletions);
  run_rac.py purely additive; shas hash-frozen. Mirrors the just-landed HEADRECIPE precedent. Verified.
- **DEV-3** (mixup re-derives the align rep in loss.py, classifier.py untouched) — **FAVORABLE.** The
  re-derivation reproduces the align forward exactly (mod_dropout OFF); classifier sha unchanged; the A3-confined
  second-dropout divergence is disclosed. Verified.
- **DEV-4** (bank stop-grad at construction AND locally) — **FAVORABLE, verified** (hostile-bank grad = 0).
- **DEV-5** (LOO as explicit id→row map + `−inf` mask + `clamp(min=−30)` guard) — **FAVORABLE, verified**
  (self-mass = 0; no NaN).
- **DEV-6** (single `--nca_tau` for the NCA grid and the SupCon temperature) — **NEUTRAL, documented.** No
  behaviour change; SupCon has a single pinned value (0.1); ProxyNCA++/in-batch-grad/NCA-only parked.
- **DEV-7** (per-epoch bank in `model_pass`, mining inert via the early return) — **NEUTRAL, recon-aligned,
  verified.** `nca_bank` built only for `head_loss=='nca'`; `train_feats/train_labels` stay None all epoch;
  nothing consumes the absent mined pairs; A3 keeps triplet + mining exactly as the floor.

---

## NON-BLOCKING NOTES (for the executor / verdict reviewer)

1. **§1.2 / §5.1 mislabel loss.py as "+147/−6"; the true numstat is 141 insertions / 6 deletions.** git's
   `--stat` bar renders 147 = 141 insertions + 6 deletions combined; the prereg transcribed the bar total as
   the insertion count. The **deletion** count (6) and their identity (the BCE block, re-emitted verbatim under
   the new `else:`) are correct, and no hash / bar / gate is affected. Descriptor-only slip (numeric-provenance
   nit). Non-material.

2. **F0.2 over-states A2 (SupCon)'s RNG divergence — conservatively.** run_rac builds the per-epoch bank
   ONLY for `head_loss=='nca'` (run_rac.py:697), and `_supcon_head_loss` draws no RNG, so A2 iterates
   `train_dl` once per epoch (like the floor) and stays in RNG lockstep — A2 is in fact the **cleanest** paired
   arm (matched shuffle + dropout, only the loss functional differs). F0.2's blanket "the NCA/SupCon/mixup arms
   all diverge from the floor's RNG stream" claims MORE divergence than exists for A2; the direction is
   conservative and touches no bar. Non-material.

3. **A1 (NCA) and A3 (mixup) DO carry the disclosed treatment-arm RNG divergence.** A1's per-epoch
   `_build_nca_bank` consumes a `train_dl` shuffle before the step loop, so A1's per-epoch data order is a
   different permutation than the floor's; A3 draws a `Beta` λ + a permutation each step. Their paired delta
   therefore carries a seed-noise-level data-order / regularisation component on top of the objective swap —
   the same class of caveat the HEADRECIPE review recorded for mod-dropout. F0.2 discloses the mechanism
   explicitly (two sentences before its slightly-loose "matched data-order" clause); head-INIT is matched and
   the divergence averages over 3 seeds. Within-precedent, non-blocking; recorded so the verdict reviewer reads
   A1/A3 as "floor-objective vs +arm-objective under matched init, treatment-shuffled order."

4. **A3's `_manifold_mixup_bce` re-forwards the classifier a second time.** In the mixup arm the BCE logit
   comes from a fresh `img_proj/text_proj/mlp/output_layer` forward on the mixed rep, NOT from the `output`
   computed at loss.py:32 — so line-32's `output` is computed-but-unused-for-loss in A3 (a few dead flops). This
   is inherent to manifold mixup (the mixed rep must be re-forwarded) and harmless; it does not affect the
   triplet term (real feats) or the floor. Informational.

---

## Reviewer's independent ruling on the LOO mask + stop-grad + flag-off byte-identity (≤3 sentences)

The **LOO self-exclusion HOLDS**: `id_to_row` (from a bank asserted to have unique train ids) resolves every
batch anchor's own row and `logits[arange(B), own_rows] = −inf` zeroes exactly that row's softmax mass — my
import-level smoke drove an anchor to equal its own bank row and the retained self-mass was exactly
`0.000e+00` — and the loss is training-only (the eval path routes through `retrieve_evaluate_RAC_`, never
`compute_loss`), so dev/test never enter the mask. The **bank stop-grad HOLDS**, enforced twice
(`no_grad`+`.detach()` at construction, local `bank_feats.detach()` in the loss): a hostile
`requires_grad=True` bank received a gradient of exactly `0.000e+00` while the anchor grad flowed. The
**no-flag byte-identity HOLDS**: with the flags absent `head_loss` defaults `'triplet'` (early branch
skipped), `mixup` defaults `False` (the hybrid `else` re-emits the 6 deployed BCE lines verbatim — git
confirms those are the only 6 deletions), and `nca_bank` stays `None` and unused, so a fresh no-flag run
reproduces the banked 13150/13241 floors and they need no re-run. **Cleared to freeze + single-submit** after
the pre-submit codex gate on the NCA/SupCon/mixup branches.

## HASH-FREEZE

Recorded in `refine-logs/NCA_FREEZE.md` (prereg NOT modified, per review mandate). All freeze-block shas
re-verified on disk at freeze time and **match**: prereg self-sha `7607863c…`, A `e1244ada…` (loss.py), B
`b85eb72a…` (run_rac.py), C `baf41be8…` (ncafam_family.sbatch); reused-unchanged `classifier.py e7b61df4…`,
`retrieval.py d43e3bc4…`, anchor `enc3seed_lora_curric.sbatch 00d9e995…`; the 6 banked caches present and
untouched.

**Reviewer statements:** ZERO GPU/SLURM/Modal spent — CPU-only login-node re-parse of the banked 13150/13241
trainlogs with an independently written parser, an import-level synthetic-tensor smoke of the ACTUAL
`_nca_head_loss`/`_supcon_head_loss`/`_manifold_mixup_bce` (LOO self-mass, hostile-bank stop-grad,
decreasing NCA, mixup λ), plus `sha256sum` / `py_compile` / `bash -n` / `ls` collision checks (seconds); no
held-out test metric produced; `state/` and `autoresearch/goal_mllm_plus3/state/` not touched; the prereg was
**NOT** modified; no job submitted; not pushed. Cloud/external numbers were never mixed with local G-repro
numbers.
