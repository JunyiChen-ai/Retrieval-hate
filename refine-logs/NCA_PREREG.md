# NCA / soft-kNN HEAD-LOSS family Pre-Registration — deployed-vote-aligned head objective (ZH + HateMM, 4 arms, one bite)

**Author:** NCA-family prereg author (CPU-only; no GPU/SLURM/Modal spent; NO job submitted).
**Date:** 2026-07-25 NZST.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced; no job submitted.
**Implements:** `refine-logs/NCA_FORENSIC_RECON.md` (commit `685df9e`, the GO recon) — its 4-arm family
design (one bite, multiplicity-safe: A1a/A1b NCA τ-grid, A2 neighborhood-SupCon, A3 manifold mixup), the
frozen hyperparameters, the F66-non-binding + law-I-counter-pressure rulings, the ban-collision closures, the
bank-detach / LOO-by-id / mining-inert implementation pins, and the kill-bar skeleton — transcribed and
re-verified below. Deviations from the recon are flagged **loudly** in §11.
**House-style precedent:** `refine-logs/HEADRECIPE_PREREG.md` (closest shape: flag-gated head-only family on
cached features, additive `getattr`-gated patches, F73 additive-gating precedent), `refine-logs/FRAME16_PREREG.md`
(binding language, F0.x honesty clauses, pinned pipeline, re-derived line-cited floors, freeze block, single-submit
plan, outcome-table template); `research-wiki/experiments/exp-encoder-3seed.md` (the 12850 encoder-swap protocol +
decision rule verbatim :73-85).

## Title + claim scope (verbatim)

> This measurement tests **one cheap head-TRAINING objective family** — replace/augment the deployed head's
> `triplet(m=0.1)+0.5·BCE` contrastive term with a loss that **directly optimizes the deployed top-20 signed-cosine
> kNN vote** — run entirely on **CACHED LoRA features**, **3-seed paired** within head-seed against each dataset's
> own banked floor (ZH generic-LoRA job 13150; HateMM curriculum-LoRA job 13241), dual-protocol (val-selected AND
> final-epoch), on **ZH (`MHC_zh`) and HateMM, each trained ONLY on its own train split**. The family has **4 arms
> under ONE multiplicity bite**: **A1a NCA τ=0.1 / A1b NCA τ=0.2** (a pre-declared 2-point temperature grid, both
> reported, no winner-reselect), **A2 neighborhood-SupCon (τ=0.1)**, **A3 manifold mixup (α=2.0)** on the existing
> triplet+BCE; BCE kept at `ce_weight=0.5` in all arms. It is a **PURE-PERFORMANCE + DIAGNOSTIC family, NOT an
> expected-+3 bet**: even a formal PASS is a performance/ablation row (a training objective is a generic recipe,
> **D7-DEAD**, F0.3), never a novelty contribution. **F66 does NOT bind this cell** — F66's β-decomposition is
> conditional on a *fixed* embedding map φ₀ and bounds only *inference-side* symmetric operators over φ₀'s Gram
> matrix; an NCA/soft-kNN *training* loss produces a *different* map φ′ with a different Gram matrix, oracle, and
> symmetric/selection split, objects F66 never measured — so the cell is legitimately un-measured, not F66-dead.
> The honest counter-pressure, kept in the open and driving the low prior, is **law-I** (the 8-instance empirical
> pattern "better representation ⇒ zero vote conversion"): NCA is the one operator that discriminates a *wrong
> objective* (beatable) from a *feature-information ceiling* in the frozen LoRA-Qwen embeddings (not beatable),
> which is exactly its diagnostic value and exactly why the honest **P(≥+3)=2–4%** and the realistic deliverable is
> **ZH val-sel hardening**, not a new +3 dataset. Hyperparameters are **FROZEN at recon-pin** (τ∈{0.1,0.2} the only
> pre-declared multiplicity; SupCon τ=0.1; mixup α=2.0; ce_weight=0.5; bank-detach + per-epoch rebuild + LOO-by-id
> all fixed); this prereg decides the **performance clause only.**

The cells under test are the deployed RGCL `classifier_hateClipper`, `fusion_mode=align`, AdamW head over cached
embeddings, 30 epochs, warmup 5, `--force False`, paired **3-seed within head-seed** vs the banked floors,
dual-protocol. **The treatment and its floor read the byte-identical banked feature cache** (no re-extraction, no
re-SFT) — so the ONLY difference at a given seed is the head-training objective (§0 F0.2). A1a/A1b set
`--head_loss nca --nca_tau {0.1,0.2}`; A2 sets `--head_loss supcon --nca_tau 0.1`; A3 sets `--mixup True
--mixup_alpha 2.0`; all other tokens identical to the floor command (§4.2). **Any re-tuning of τ/α/ce_weight, a
NCA-only (no-BCE) variant, ProxyNCA++, or in-batch-grad NCA are OUT of this prereg** — touching a knob spends the
family and re-costs a bite (§3.6).

---

## 0. Binding facts / honesty clauses (all present; pre-declared)

**F0.1 — Test is NOT virgin (declared).** ZH and HateMM test were already read, under the identical `enc3s`
protocol, by: frozen-CLIP (ZH 13115 / HateMM 12850), frozen-Qwen-8f (12850), generic-LoRA (ZH B3 job 13150; HateMM
job 13235), curriculum-LoRA (ZH+HateMM job 13241), the LoRA-HateMM verdict, frame16 (13353), and the head-recipe
family (SAM/mod-dropout, if run). This prereg's NCA-family test reads are **re-measurements under the identical
protocol**, not first exposures. There are **eight** budgeted arm×dataset test evaluations — {A1a, A1b, A2, A3} ×
{ZH, HateMM} — each = the **3 head-seed reads** of one arm×dataset cell (**24 reads total**). **Zero test-touch
before the independent verdict.**

**F0.2 — The paired test isolates the head objective cleanly; single-encoder-draw caveat, and why it does NOT
confound this comparison (pre-declared, material).** Both the treatment and its floor consume the **SAME banked
single-SFT-draw LoRA feature cache** (ZH `…-LoRA_HF.pt`; HateMM `…-LoRA-curric_HF.pt`) — there is **no
re-extraction and no re-SFT** here. So the single-encoder-draw limitation that burdens the LoRA cells is **shared
identically** by treatment and floor and **cannot** confound the head-objective delta: the ONLY thing that differs
at a given seed is the head training objective. `--seed` controls head-init + data-shuffle; the pairing is per
head-seed (arm seed s − floor seed s), `s ∈ {0,1,2}`.
*Arm-specific RNG note (pre-declared):* the NCA/SupCon/mixup arms all **diverge** from the floor's RNG stream from
the first new draw — A1's per-epoch detached bank build advances the sampler RNG before the step loop; A3 draws a
`Beta` λ + a permutation each step. This is expected and confined to the treatment arms (exactly the mod-dropout
divergence precedent); the flag-OFF path (`head_loss='triplet'`, `mixup=False`) draws NOTHING new and is
byte-identical to the floor (F0.7). Arm-vs-floor pairing therefore isolates "floor objective vs +arm objective"
under matched head-init + data-order at each seed.

**F0.3 — Novelty = D7-DEAD, say it plainly (a training objective is a generic recipe).** NCA (NIPS'04),
soft-NN/Soft-kNN (Frosst ICML'19), SupCon (Khosla NeurIPS'20), and manifold mixup (Verma ICML'19) are all textbook
training objectives / regularizers. Even a formal PASS is a **performance/ablation row** ("head objective:
vote-aligned NCA / SupCon / mixup on the align head"), same D7 class as C4 (head-eng) and C5 (recipe) — **never** a
novelty contribution. The *diagnostic* value (discriminating wrong-objective from feature-ceiling, F0.5/§3.1) is a
strong paper *sentence*, not a novelty *mechanism*. This is a pure-performance + door-closer family.

**F0.4 — Single-dataset own-train-split VETO compliance (hard user veto; trivially clean).** No SFT, no data build,
no cross-dataset mixing: each arm's head trains on its own dataset's own-train feature cache only, exactly as the
banked floor did. The NCA bank is built **only from that dataset's own train split** (the same corpus the deployed
FAISS vote reads). No gold spans/attributes, no OCR channel, no cross-seed ensemble, raw videos never leave the
machine. All standing vetoes cleared.

**F0.5 — Honest priors are LOW; law-I lowers the +3 prior; none raise them (pre-declared, material; recon §0/§2.4).**
Per litsweep, restated per arm (§2.4 table): **P(≥+1 stable ZH val-sel) ≈ 12–18% (NCA A1a/A1b), ~10% (SupCon A2),
12–15% (mixup A3); P(≥+3 any dataset) ≈ 1–4%.** Two disclosed headwinds, neither a ban:
- **(a) law-I counter-pressure (the honest +3 cap).** law-I is an 8-instance empirical pattern — better image
  stream (F65), better fusion (F50), better readout (F70), denser frames (F67), per-segment re-encode (F66) all
  improved the *representation* and gained *zero* on the vote. If the ceiling is a **feature-information ceiling**
  in the frozen LoRA-Qwen embeddings fed to the head, NCA cannot beat it either (law-I is agnostic about *why* the
  ceiling exists). NCA is the single operator that discriminates "wrong objective" (beatable) from "feature
  ceiling" (not) — which is why **P(≥+3) stays 2–4%** and the value is **diagnostic-per-dollar**, not expected lift.
- **(b) SupCon regime headwind.** triplet ≥ SupCon on small/mid datasets (2510.02161); SupCon needs a larger batch
  and can collapse on imbalanced binary as batch grows — batch 64 is borderline (recon §2.2). A2 prior is the
  lowest of the family.
- **(c) HateMM is near-ceiling.** The HateMM floor is 0.8775 (val-sel) / 0.8791 (final) — project-best, thin
  headroom; the +0.030 promote bar sits at ~0.909 everywhere. **ZH is the realistic target** (marginal cell, real
  headroom); HateMM is a hold-the-line leg.
- **(d) Family = one bite, knobs frozen.** τ∈{0.1,0.2} (the only pre-declared multiplicity), SupCon τ=0.1,
  α=2.0, ce_weight=0.5 pinned now; **no post-hoc tuning** — if a knob is touched the family is spent and re-costs a
  bite (§3.6). If the whole family fails, law-I upgrades from "operators mismatch the vote" to "even the
  vote-matched objective can't" — a strong closing paper sentence (recon §0).

**F0.6 — Bank-detach / LOO-by-id / mining-inert invariants (LOAD-BEARING; the three risk surfaces, HANDLED and
asserted).** Three implementation facts carry the family's correctness; each is pinned in code and re-checked at
smoke + the codex gate (§4.4/§4.5):
- **(i) Bank stop-grad.** The per-epoch NCA bank is built **detached** (`_build_nca_bank`, `model.eval()` +
  `torch.no_grad()` + `.detach()`, run_rac.py:609-639) AND `_nca_head_loss` **re-detaches locally**
  (`bank_feats = bank_feats.detach()`, loss.py:651) — belt-and-suspenders stop-grad: anchor grad-on, bank grad-off,
  no O(N²) grad, no in-epoch bank drift. Smoke asserts a hostile bank tensor with `requires_grad=True` receives
  **zero** gradient (§4.4.1).
- **(ii) LOO self-exclusion BY ID.** At training time the anchor **is** in the train bank; its own row MUST be
  masked from its softmax (classic NCA `j≠i`; deployment self-excludes because dev/test queries are disjoint from
  the train bank). The mechanism is **by video id**: `_build_nca_bank` returns `id_to_row = {train_id → bank row}`;
  `_nca_head_loss` computes `own_rows = [id_to_row[v] for v in batch_ids]` and sets those `[anchor, own_row]`
  logits to `−inf` before the softmax (loss.py:656-659). Smoke asserts the anchor's retained self-mass is `<1e-6`
  for anchors whose ids are in the bank (§4.4.1).
- **(iii) Triplet machinery goes inert (A1/A2).** When `head_loss∈{nca,supcon}` the early branch (loss.py:34-64)
  **returns before** the FAISS mining call, so `dense_retrieve_hard_negatives_pseudo_positive` is never entered and
  the mined `hard_negative_features`/`pseudo_positive_features` are never consumed; `train_feats/train_labels` stay
  `None` all epoch (no mining forward). **Verified nothing else consumes the mined pairs** (recon §7.2: `loss.py`
  routes both ONLY into the triplet assembly; the eval path and `run_rac` do not read them). A3 (mixup) keeps
  `head_loss='triplet'` ⇒ triplet + FAISS mining run **exactly as the floor**; mixup only rewrites the BCE target.

**F0.7 — Additive-gating / same-code fact (pre-declared).** Every new code path (the NCA/SupCon early branch, the
mixup BCE hook, the `_build_nca_bank` builder, the per-epoch bank build, the 4 argparse keys) is gated behind
`getattr(args, <flag>, <default>)` / `head_loss=='triplet'`. With the flags absent, `head_loss` defaults to
`'triplet'`, `mixup=False`, `mixup_alpha=0.0`, `nca_tau=0.1` (inert): the early branch is skipped, the mixup hook's
`else` re-emits the deployed BCE **verbatim** (whitespace-only reindent — git-diff verified: the only loss.py
"deletions" are the 6 BCE lines re-emitted under the new `else:`), `nca_bank` stays `None`, and the classifier
forward is untouched ⇒ **behaviour is byte-identical to the banked floors, which therefore need NO re-run.** Adding
the argparse keys grows the Namespace by exactly 4 inert keys (`head_loss='triplet', nca_tau=0.1, mixup=False,
mixup_alpha=0.0`) with **zero behaviour diff** when unset — the established additive-flag pattern (cf. `--sam`,
`--mod_dropout`, `--cf_negs`, `--lambda_seg`, `--lambda_aux`). run_rac.py is **purely additive** (68 insertions, 0
deletions).

**F0.8 — Surrogate-fidelity is a declared approximation, not an identity (honest, pre-declared; recon §2.0).** The
deployed decision is a rank-decay AND cosine-magnitude weighted SIGNED top-20 kNN vote over the train bank
(`s(q)=Σ_j w_rank(j)·(2y_j−1)·cos(φ(q),φ(k_j))/Σw_rank`, decision `[s≥0]`, metrics.py:262-284). L_NCA is its **soft
surrogate** with four declared, standard relaxations: (1) it drives same-class softmax mass `P_i→1`, the smooth
relaxation of the 0/1 vote-correctness indicator; (2) soft full-bank softmax vs hard top-20 (the rank-15–20 tail
carries ≤5/210 arithmetic weight AND negligible `exp(cos/τ)` mass ⇒ soft-full-bank ≈ hard-top20 at the margin;
hard truncation rejected — non-differentiable); (3) the softmax `exp(cos/τ)` rank-weighting is a *sharper*
similarity-driven surrogate for the arithmetic `[20..1]` decay — we do NOT replicate the exact profile (deliberate
surrogate choice); (4) for binary labels "same-class mass >0.5 ⟺ signed vote >0" (monotone), so NCA's
argmax-improving direction aligns with signed-vote-correctness. These are surrogate gaps by design, not defects;
they are the reason the objective is *vote-aligned*, not *vote-identical*.

---

## 1. Pipeline spec — fully pinned (nothing left to interpretation)

**Deployed head recipe (identical for both floors and all four treatment arms; pinned from
`enc3seed_lora_curric.sbatch`, byte-identical across the enc3seed floors):** RGCL `classifier_hateClipper`,
`fusion_mode=align` (Hadamard `x=img⊙text`), triplet+BCE hybrid (`--loss triplet --hybrid_loss True
--ce_weight 0.5`), AdamW over `model.parameters()` (`lr 1e-4`), 30 epochs, `batch 64`, `topk 20`, `proj_dim 1024`,
`map_dim 1024`, `dropout 0.2 0.4 0.1`, `hard_negatives_loss True`, `no_hard_negatives 1`, `no_pseudo_gold_positives
1`, `metric cos`, `majority_voting arithmetic`, `warmup 5`, `Faiss_GPU False`, `reindex_every_step=False` (FAISS
re-mine once/epoch), `lambda_seg 0 / seg_mode full` ⇒ single `model_pass` (else-branch, NOT the EM path). Model
selection = Val_Retrieval acc (roc tie), warmup≥5.

### 1.1 The single stage — 24 head-only runs on cached features (ONE sbatch)

- **Submit:** `sbatch scripts/slurm/ncafam_family.sbatch` (authored this prereg — artifact C, §5).
- **What it runs:** **4 arms × 2 datasets × 3 seeds = 24 head-only runs** on the banked LoRA caches (~20–50 s/run):
  - **A1a (NCA τ=0.1):** `--head_loss nca --nca_tau 0.1`, exp tag `_..._nca_tau0.1`.
  - **A1b (NCA τ=0.2):** `--head_loss nca --nca_tau 0.2`, exp tag `_..._nca_tau0.2`.
  - **A2 (n-SupCon):** `--head_loss supcon --nca_tau 0.1`, exp tag `_..._supcon_tau0.1`.
  - **A3 (manifold mixup):** `--mixup True --mixup_alpha 2.0`, exp tag `_..._mixup_a2.0`.
  - each × {`MHC_zh`:`…-LoRA_HF` (13150 floor), `HateMM`:`…-LoRA-curric_HF` (13241 floor)} × seed{0,1,2}.
  - `--group_name RAC_video_ncafam`, `--force False`.
- **Frozen knobs (recon-pin; NO tuning — one bite):** τ∈{0.1,0.2} (the only pre-declared multiplicity, both
  reported, per-arm pass rule, no winner-reselect); SupCon τ=0.1; mixup α=2.0 (`Beta(2,2)`); ce_weight=0.5 in all;
  bank-detach + LOO-by-id + per-epoch bank rebuild fixed (§0 F0.6).
- **Output:** `slurm/logs/nca_{nca_tau0.1,nca_tau0.2,supcon_tau0.1,mixup_a2.0}_{MHC_zh,HateMM}_<MODEL>_seed{0,1,2}_<JID>.trainlog`.
- **GPU budget:** head-only on cached feats; A1/A2 add one detached bank forward/epoch (~+1–2 s/epoch × 30, ~12
  train batches ⇒ negligible); A3 ~floor cost. **24 × ≤50 s ≈ 20 min ≈ ~0.33 GPU-h < 0.5.** ONE A100, serial, ONE
  sbatch.

### 1.2 The patches (2 files; all additive, all getattr / head_loss-gated)

1. **`src/run_rac.py`** (additive, +68 lines, 0 deletions):
   - **+4 argparse keys** (no-op defaults, run_rac.py:550-574): `--head_loss {triplet,nca,supcon}` (default
     `triplet`), `--nca_tau` (float, `0.1`), `--mixup` (bool, `False`), `--mixup_alpha` (float, `0.0`).
   - **Module-level `_build_nca_bank(train_dl, model, args)`** (run_rac.py:609-639): builds the DETACHED per-epoch
     memory bank in the current embedding space — `model.eval()` + `no_grad` forward over the whole train split,
     returns `(bank_feats [N,proj_dim] detached, bank_labels [N] long, id_to_row {train_id→row})`; restores the
     model's prior mode; asserts train ids are unique (LOO-by-id well-defined).
   - **Per-epoch bank build in `model_pass`** (run_rac.py:693-698, gated `head_loss=='nca'`): `nca_bank=None` reset
     alongside the existing `train_feats=None` reindex reset; `nca_bank=_build_nca_bank(...)` once per epoch,
     reused all epoch (mirrors the once-per-epoch FAISS reindex; supcon/mixup ⇒ `nca_bank` stays None).
   - **`nca_bank=nca_bank` threaded into BOTH `compute_loss` calls** (the main call run_rac.py:729 and the SAM
     second call run_rac.py:770 — the latter for cross-family safety only; NCA arms set `--sam` False).
2. **`src/model/loss.py`** (additive, +147/−6, the 6 "deletions" = the reindented byte-identical BCE block):
   - **`compute_loss` gains `nca_bank=None`** (loss.py:24).
   - **NCA/SupCon early branch** (loss.py:34-64, after the forward, before the label-matrix / mining): when
     `head_loss∈{nca,supcon}` sets the contrastive term to `_nca_head_loss` / `_supcon_head_loss`, keeps `0.5·BCE`,
     and RETURNS the 7-tuple (zeros for the logged in_batch/hard/pseudo terms) **before** any FAISS mining.
   - **Mixup BCE hook** (loss.py:579-586, in the `hybrid_loss` block): when `mixup and mixup_alpha>0`, the BCE is
     `_manifold_mixup_bce(...)` on the mixed fused rep; else the deployed BCE runs verbatim.
   - **Helpers `_nca_head_loss` / `_supcon_head_loss` / `_manifold_mixup_bce`** (loss.py:635-717).
3. **`src/model/classifier.py`** — **NO change** (align forward re-derived read-only inside `_manifold_mixup_bce`
   via `model.img_proj/text_proj/mlp/output_layer`; classifier sha unchanged, §5.2).
4. **`src/utils/retrieval.py`** — **NO change** (A1/A2 bypass the FAISS mining path entirely; A3 uses it exactly as
   the floor; retrieval sha unchanged, §5.2).

**CPU-verified this prereg (§4):** `python -m py_compile src/model/loss.py src/run_rac.py` PASS; a synthetic-tensor
CPU smoke (n small) confirms NCA finite + **decreasing** (0.71→0.17 over 20 steps), hostile-bank grad-norm **0.0**
(internal stop-grad), LOO self-mass **0.0** (anchor never its own neighbour), SupCon finite + grad-flows, mixup
**λ∈[0,1]** finite + param-grads, and all default gates OFF (no-flag path byte-identical).

---

## 2. Comparison floors — INDEPENDENTLY RE-DERIVED from raw trainlogs (numeric-provenance discipline)

Every number below was independently re-parsed **this prereg** from the raw trainlogs with the EXACT
`enc3seed_lora_curric.sbatch` embedded parser (val-sel = epoch ≥ warmup 5 max `Val_Retrieval` acc, roc tie-break;
final = max epoch). Both floor means bit-match `NCA_FORENSIC_RECON.md §1.4`, `HEADRECIPE_PREREG.md §2.1/§2.2`, and
`CAND2_CURRICULUM_PREREG.md` to 4dp — no discrepancy.

### 2.1 ZH floor — job **13150** (`Qwen2.5-VL-7B-Instruct-LoRA_HF`, generic-LoRA / B3; goal-relevant, marginal)

| seed | val-sel ep | val-sel acc/mF1 | final ep | final acc/mF1 | provenance (Test line, re-read this prereg) |
|---|---|---|---|---|---|
| 0 | 20 | 0.8322 / 0.8023 | 29 | 0.8456 / 0.8181 | seed0 log :197 (val) / :270 (final) |
| 1 | 26 | 0.8255 / 0.7956 | 29 | 0.8389 / 0.8113 | seed1 log :246 / :271 |
| 2 | 19 | 0.8389 / 0.8065 | 29 | 0.8523 / 0.8226 | seed2 log :185 / :266 |
| **mean** | | **0.8322 / 0.8015** | | **0.8456 / 0.8173** | |

Files: `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`. Bit-matches
`B3_PREREG_REVIEW.md`, `CAND2_CURRICULUM_PREREG.md §2.1`, `HEADRECIPE_PREREG.md §2.1`.

### 2.2 HateMM floor — job **13241** (`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`, curriculum-LoRA; near-ceiling, project-best)

| seed | val-sel ep | val-sel acc/mF1 | final ep | final acc/mF1 | provenance (Test line, re-read this prereg) |
|---|---|---|---|---|---|
| 0 | 29 | 0.8791 / 0.8730 | 29 | 0.8791 / 0.8730 | seed0 log :299 (val=final) |
| 1 | 14 | 0.8744 / 0.8678 | 29 | 0.8791 / 0.8724 | seed1 log :161 (val) / :297 (final) |
| 2 | 10 | 0.8791 / 0.8724 | 29 | 0.8791 / 0.8724 | seed2 log :127 (val) / :299 (final) |
| **mean** | | **0.8775 / 0.8711** | | **0.8791 / 0.8726** | |

Files: `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`. This is the
project-best HateMM cell (cand-2 curriculum, `goal-round3-terminus` memory; 0.8775 = project best). Bit-matches
`HEADRECIPE_PREREG.md §2.2`.

### 2.3 Concrete promote thresholds + noise band

- **ZH promote (+0.030 per seed):** val-sel acc ≥ {0.8622, 0.8555, 0.8689}, mF1 ≥ {0.8323, 0.8256, 0.8365};
  final acc ≥ {0.8756, 0.8689, 0.8823}, mF1 ≥ {0.8481, 0.8413, 0.8526}.
- **HateMM promote (+0.030 per seed):** acc ≥ ~0.909 everywhere (near-ceiling; thin surface — F0.5(c)).
- **Head-seed noise band (for KS-regression, §3.4):** ±**0.014** — the established house head-seed-spread descriptor
  (`CAND2_CURRICULUM_PREREG.md §2.3`, `HEADRECIPE_PREREG.md §2.3`; largest observed generic-arm between-seed acc
  spread 0.0140); the local floors here are within it (ZH val-sel/final acc spread 0.0134; HateMM ≤ 0.0047). A
  3-seed **mean** move beyond ±0.014 is beyond the full head-seed spread.

---

## 3. Decision rule + kill-bars (paired, both protocols judged independently, 3/3 sign, pre-declared)

### 3.1 Decision rule — verbatim from `exp-encoder-3seed.md:73-85` (treatment = arm; control = the arm's banked floor)

> For each dataset × protocol: (1) per-seed paired difference δ = (treatment − control) for acc and macro-F1 at
> seeds 0/1/2; (2) 3-seed mean ± std + sign consistency (how many of 3 positive); (3) n=3 too small for a
> bootstrap — report the paired-t **as an effect-size descriptor only**, no significance claim; (4) **pass =
> mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive**; (5) headline claim
> requires pass on ≥ 2 datasets under a stated protocol; both protocols judged separately; verdict written
> exactly "final-epoch: pass/fail; val-selected: pass/fail".

Both protocols judged **independently** (no protocol-shopping, no metric-shopping). Control = the arm's OWN banked
floor (ZH §2.1, HateMM §2.2). Judged **per arm × per dataset**; the family verdict is per-arm (§3.6).

### 3.2 FORMAL promote bar (goal-facing; per arm × dataset)

House **+0.030 acc AND +0.030 mF1** conjunct, **3/3 seeds positive**, under **BOTH** protocols vs the banked floor
(§2). Below the conjunct under a protocol → **NEGATIVE** on that protocol. **D7-DEAD (F0.3): even a formal PASS is
an engineering/ablation row, NEVER a novelty win.** The **legal oracle-conversion ceiling** on the headroom story
is F66's **+0.001–0.006** symmetric slice (recon §3.1) — a formal +0.030 pass would therefore be an *empirical*
result about a *re-trained* φ′, consistent with F66 being silent on φ′ (it never measured it), not a violation of it.

### 3.3 KS-arm-dead — the KILL bar (SIGN-based; per arm × dataset)

Per the **frame16 DEV-1 sign discipline** (house n=3 = **no bootstrap**; the kill uses SIGN, not a CI-straddles-0
test): an arm×dataset cell is **KILLED** iff, on **BOTH protocols**, `mean paired Δacc ≤ 0` **OR** the acc sign is
not 3/3 positive — i.e. **neither** protocol produces a clean positive-mean-and-3/3-sign result (a tie-or-regress
on both protocols = no net improvement over the floor). This is the sign-based analog of the recon's "≤ floor on
both protocols." At verdict time, state each killed cell explicitly. **A whole-family KS-arm-dead sweep is the
honest expected outcome (F0.5) and is itself a strong door-closer:** it upgrades law-I from "operators mismatch the
vote" to "even the vote-matched objective can't move it" (the ceiling lives in the frozen features).

### 3.4 KS-regression — BELOW-FLOOR-BY-SPREAD note (per arm × dataset)

If arm − floor **mean Δacc ≤ −0.014** on a leg (below the full head-seed spread, §2.3), the objective **degraded**
the head → bank "NCA/SupCon/mixup hurts on <dataset>." (SupCon on batch-64 imbalanced binary is the most plausible
regressor — F0.5(b).) A note within the KS-arm-dead frame, not a separate multiplicity bite.

### 3.5 Ban-collision closure (carried from recon §3; F66 is the centerpiece, disclosed NOT a ban)

- **F66 (β-decomposition, KILLED for inference-side operators):** F66 bounds *symmetric inference-side* operators
  over a **fixed** φ₀ Gram matrix (ISR / re-agg / vote-reweighting) to +0.001–0.006 legal. An NCA/soft-kNN
  **training** loss reshapes φ₀→φ′ (a *different* Gram matrix, oracle, and symmetric/selection split, objects F66
  never measured) ⇒ **F66 does NOT arithmetically bind trained-space reshaping; the cell is legitimately
  un-measured, not F66-dead** (recon §3.1, three-point ruling). Honest counter-pressure = law-I (F0.5(a)).
- **Not F73** (SAM optimizer + mod-dropout input-masking): NCA/SupCon change the **loss functional**; mixup is an
  **input-interpolation regularizer on the BCE path** — none is an optimizer perturbation or a masking operator.
  **Distinct** (and A3 mixup ≠ mod-dropout: interpolation ≠ dropout).
- **Not P9b** (encoder-in-loop): every arm is **head-only on cached deployed features**; no encoder gradients.
- **Not cand-2** (data reweighting/curriculum): the arms change the **loss functional / a feature-mixup
  regularizer**, not the data, its weights, or its curriculum (the curriculum LoRA cache is the *floor* input).
- **Not F50** (FA fusion/composition, inference-time channel op): NCA/SupCon/mixup are **training-time
  objectives**; the align-Hadamard fusion and the eval path are untouched.
- **Not F62/F62b (SWA averaging) / F69 (grad-norm selection) / F70 (readout):** none is a loss functional; A3 mixup
  is a **variance-reduction** regularizer that F66 does not gate at all (it does not convert φ₀'s oracle headroom;
  it stabilizes the argmax) and attacks the diagnosed ZH failure (F45: 78-dev selection noise, dev saturates ep19
  while test climbs to ep29). **cross-seed ensemble / OCR / gold-in-method / single-dataset-split / P1–P5 /
  external-API / target-as-structure** — none reach a training-time head objective. **Clear.**

### 3.6 Multiplicity + scope of THIS submit (pre-declared)

- **ONE sbatch = ONE pre-registered family = ONE multiplicity bite** whether one or all four arms survive. The four
  arms **share** the single "litsweep2 wave-3 NCA head-loss" bite.
- **The ONLY pre-declared multiplicity is the 2-point τ grid** (A1a/A1b), justified by documented NCA
  tau-sensitivity (Frosst ICML'19 learns T); **both values are reported and the per-arm pass rule applies to each
  — we do NOT pick the winner and re-run** (that would be selection-laundering).
- **Hyperparameters FROZEN** (τ∈{0.1,0.2}; SupCon τ=0.1; α=2.0; ce_weight=0.5; bank-detach; LOO-by-id; per-epoch
  rebuild). **NO post-hoc knob tuning** — a τ/α/ce sweep, a NCA-only (no-BCE) variant, ProxyNCA++, in-batch-grad
  NCA, or hard-top20 truncation is a **new** pre-declared arm and re-costs a bite.
- **Family verdict is per-arm × per-dataset** (each judged only vs its own floor). A surviving arm×dataset cell
  still owes the **full ceremony** (this prereg → independent 0-context review → freeze-hash → SLURM); this prereg
  does **not** discharge that, and this family is the ONLY NCA-head-loss bite.

### 3.7 Gate order

G-repro (patched-file sha re-verify + no-flag Namespace-diff + additive-gating proof, §4.1) → **codex review of the
NCA/SupCon/mixup branches (§4.5)** → smoke (§4.4) → single test-touch (the 24 head reads) → per arm×dataset:
KS-arm-dead → FORMAL promote bar (both protocols). The verdict is rendered by an **independent 0-context reviewer
against this prereg VERBATIM**; the executor transcribes raw both-protocol per-seed numbers (line-numbered) and
applies NO gates/interpretation.

---

## 4. G-repro + smoke plan + collision safety + codex gate

### 4.1 G-repro discipline

- **(a) Patched-file sha gate.** At submit time re-run `sha256sum` on `src/model/loss.py`, `src/run_rac.py`,
  `scripts/slurm/ncafam_family.sbatch` (and this file) — must match the §5 freeze block; any mismatch =
  authorization VOID. `src/model/classifier.py` sha `e7b61df…` and `src/utils/retrieval.py` sha `d43e3bc…`
  (unchanged) re-verified (NCA touches neither).
- **(b) Additive-gating / no-flag byte-identity proof (F0.7).** A run with the flags OFF must be byte-identical in
  Namespace (modulo the 4 inert new keys + derived-inert `model`/`group_name`/`exp_comment`) to the banked floor
  command. The mixup hook's `else` re-emits the deployed BCE verbatim (git-diff verified: the only loss.py
  "deletions" are the 6 BCE lines re-indented under `else:`); run_rac.py is purely additive. Optional stronger
  check: a 1-seed **no-flag** head run on the ZH LoRA cache bit-matches the banked 13150 seed0 trajectory (classic
  G-repro bit-exact, cf. `exp-encoder-3seed.md:126-146`).
- **(c) Same-code (INCLUDING the floors).** The `python ./src/run_rac.py …` block of `ncafam_family.sbatch` is
  token-identical to `enc3seed_lora_curric.sbatch` **except** the two intended deltas: `--exp_comment
  "_${MODEL}_${ARM}"` (derived-inert) and the trailing `${ARM_FLAGS}` (additive-gated). The readout `PY` block is
  **BYTE-IDENTICAL** (`diff` empty). Both verified this prereg (§4.2).

### 4.2 Same-code + syntax verification (run this prereg — PASS)

- `python ./src/run_rac.py` invocation diff vs `enc3seed_lora_curric.sbatch`: exactly 2 lines
  (`--exp_comment "_${MODEL}"` → `"_${MODEL}_${ARM}"`; trailing `2>&1` → `${ARM_FLAGS} 2>&1`). Nothing else.
- Readout `WARMUP=… python - …PY` block vs `enc3seed_lora_curric.sbatch`: **BYTE-IDENTICAL** (`diff` empty).
- `bash -n scripts/slurm/ncafam_family.sbatch` = **SYNTAX_OK**; CONFIGS word-split dry-run = **24 rows**, each
  `run_one` receiving `(DATASET, MODEL, SEED, ARM, ARM_FLAGS…)` correctly (trailing remainder captured; A3's
  multi-word `--mixup True --mixup_alpha 2.0` splits cleanly).
- `python -m py_compile src/model/loss.py src/run_rac.py` = **PASS**.

### 4.3 Collision safety (verified this prereg — ABSENT; re-check at submit)

- `scripts/slurm/ncafam_family.sbatch`, `refine-logs/NCA_PREREG.md` — created by this prereg (no prior).
- `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_ncafam*` — do NOT exist ⇒ fresh group; `--force False` never trips
  the `run_rac.py` hard-abort; the `RAC_video_ncafam` group + per-arm `exp_comment`
  (`_..._nca_tau0.1` / `_..._nca_tau0.2` / `_..._supcon_tau0.1` / `_..._mixup_a2.0`) keep dirs distinct from every
  banked arm AND from each other.
- `slurm/logs/nca_*.trainlog` — do NOT exist ⇒ no trainlog collision. The `nca_${ARM}_…` prefix + arm tag
  guarantees the four arms (same dataset/model/seed) never collide.
- Banked caches (ZH `…-LoRA_HF.pt`, HateMM `…-LoRA-curric_HF.pt`, all 6 files verified present) and floor trainlogs
  (13150 / 13241) are **read-only inputs**; this family writes none of them (distinct group, no extraction, no SFT).
- Smoke throwaways (`_smoke_nca` group / `nca_smoke_*` logs) — deleted after smoke; must NOT persist into §4.3.

### 4.4 Smoke plan (executor runs BEFORE the real submit; leave no artifact that trips §4.3)

1. **Per-arm 1-seed 3-epoch throwaway on the ZH cache (GPU, ~1 min each):** for each of `{--head_loss nca
   --nca_tau 0.1}`, `{--head_loss nca --nca_tau 0.2}`, `{--head_loss supcon --nca_tau 0.1}`, `{--mixup True
   --mixup_alpha 2.0}` run 1 seed × 3 epochs with `--group_name _smoke_nca --epochs 3`, confirm: (i) train loss
   **finite** (no NaN) and **decreasing** across the 3 epochs; (ii) run **completes** (A1: the per-epoch bank
   builds, the run does NOT enter FAISS mining; A3: FAISS mining fires once/epoch as the floor); (iii) the pinned
   asserts do NOT trip — the **LOO self-exclusion** assert (anchor never its own neighbour: the loss's `−inf`
   self-mask; add a one-shot debug print that the max retained self-softmax-mass is `<1e-6` on a real ZH batch),
   the **bank-detach** check (the NCA bank tensor carries no grad — `_build_nca_bank` returns `.detach()`ed tensors
   AND `_nca_head_loss` re-detaches; verify via a one-shot `bank_feats.requires_grad==False`), and the **mixup λ**
   logged in `[0,1]`. Then delete `logging/Retrieval/*/…_smoke_nca*` + the smoke trainlogs.
2. **No-flag Namespace / byte-identity proof (§4.1b):** run the base command with NO arm flags (or
   `ncafam_family.sbatch` with `ARM_FLAGS=""`), dump `vars(args)`, confirm it differs from the banked floor
   Namespace ONLY by the 4 inert new keys + `model`/`group_name`/`exp_comment`. If in doubt, the optional 1-seed
   bit-exact floor reproduction settles it.
3. **CPU synthetic smoke (already run this prereg, $0):** NCA finite + decreasing (0.71→0.17), hostile-bank
   grad-norm 0.0, LOO self-mass 0.0, SupCon finite, mixup λ∈[0,1] finite + grads, default gates OFF — reference for
   the executor's GPU smoke.

### 4.5 CODEX GATE (mandatory, pre-submit — house `codex-code-review` pattern)

Before ANY SLURM submission, the executor **MUST** run a codex review (iterative loop until Claude + Codex agree)
focused on the **three risk surfaces** (recon §7, F0.6):
- **LOO indexing:** `_nca_head_loss`'s `own_rows = [id_to_row[v] for v in batch_ids]` + the `[arange(B), own_rows]`
  `−inf` self-mask — the id→row map is correct, every batch id is in the bank (train anchors), and the mask
  excludes exactly the anchor's own row (no off-by-one, no wrong-axis).
- **Grad flow:** anchor grad-on, bank stop-grad — `_build_nca_bank` (`no_grad`+`.detach()`) AND the local
  `bank_feats.detach()` in `_nca_head_loss`; `log_softmax`/`logsumexp` numerics (the `−inf` self/other-class masks,
  the `clamp(min=−30)` guard) do not NaN; A3's re-derived align fusion matches `classifier.forward` (mod_dropout
  OFF) and the mixed logit path grads to `mlp`+`output_layer`.
- **Bank rebuild cadence:** the per-epoch `nca_bank` build fires once/epoch (start-of-epoch, alongside the
  `train_feats=None` reset), is reused all epoch, and only when `head_loss=='nca'`; the early return keeps
  `train_feats/train_labels` None (mining inert) and nothing downstream consumes the (absent) mined pairs.

**Blocking findings ⇒ fix the code + re-freeze the shas (§5) + re-run this gate.** The mixup block (α range, soft
label, permutation) is included but lower-risk.

---

## 5. Artifacts authored this prereg + hash-freeze block

### 5.1 New / edited artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current) |
|---|---|---|---|
| P | `refine-logs/NCA_PREREG.md` | **NEW** — this file | *(reviewer fills at freeze)* |
| A | `src/model/loss.py` | **EDITED (additive)** — `nca_bank` kwarg; NCA/SupCon early branch; mixup BCE hook; `_nca_head_loss`/`_supcon_head_loss`/`_manifold_mixup_bce` (the 6 "deletions" = the reindented byte-identical BCE block) | `e1244adadf16b47c24b05786d1ee4e153fd9c696e3be0924eae43c82f1c3b75b` |
| B | `src/run_rac.py` | **EDITED (additive, 0 deletions)** — +4 argparse keys; `_build_nca_bank`; per-epoch bank build in `model_pass`; `nca_bank` threaded into both `compute_loss` calls | `b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3` |
| C | `scripts/slurm/ncafam_family.sbatch` | **NEW** — clone of `enc3seed_lora_curric.sbatch`; `run_one` python block token-identical + `${ARM_FLAGS}`; readout `PY` byte-identical; `RAC_video_ncafam`; 24 rows (4 arms × 2 ds × 3 seeds) | `baf41be8a1264445d6bbc63d2eb54966abed6da33c52176ae62d93bf79c14b94` |

### 5.2 Reused-unchanged machinery (verify sha at submit time; do NOT edit)

| path | role | sha256 |
|---|---|---|
| `src/model/classifier.py` | align/Hadamard forward (re-derived read-only in mixup; **NO edit**) | `e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378` |
| `src/utils/retrieval.py` | FAISS mining (A1/A2 bypass; A3 uses as floor; **NO edit**) | `d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57` |
| `scripts/slurm/enc3seed_lora_curric.sbatch` | same-code anchor for §4.2 (produced the floors' byte source of `run_one`) | `00d9e9956549bdf97c6b8913d42d87811f4a2f150e9f459e8b8b86978b306f02` |
| `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | ZH floor cache (paired input; NOT clobbered) | *(present; verified untouched)* |
| `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` | HateMM floor cache (paired input; NOT clobbered) | *(present; verified untouched)* |

### 5.3 Hash-freeze (to be filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file NCA_PREREG.md, after review>
A e1244adadf16b47c24b05786d1ee4e153fd9c696e3be0924eae43c82f1c3b75b  src/model/loss.py
B b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py
C baf41be8a1264445d6bbc63d2eb54966abed6da33c52176ae62d93bf79c14b94  scripts/slurm/ncafam_family.sbatch
```
Executor re-runs `sha256sum` on A/B/C (and this file) + confirms `classifier.py e7b61df…` / `retrieval.py d43e3bc…`
unchanged at submit time; any mismatch = authorization VOID. **If the codex gate (§4.5) forces a code fix, A/B
shas change and the freeze block MUST be re-issued.**

---

## 6. Single-submit / execution plan + resource plan

**Order (ONE SLURM job):**

1. Pre-submit: G-repro (§4.1) → **codex gate (§4.5)** → smoke (§4.4). Only on all-clear:
2. `sbatch scripts/slurm/ncafam_family.sbatch` → 24 head runs sequential inside (4 arms × 2 ds × 3 seeds),
   ~20 min wall, ~0.33 GPU-h. Produces
   `slurm/logs/nca_{nca_tau0.1,nca_tau0.2,supcon_tau0.1,mixup_a2.0}_{MHC_zh,HateMM}_<MODEL>_seed{0,1,2}_<JID>.trainlog`.

**Resource plan (STANDING INFRA RULE compliant):** the sbatch requests **`--cpus-per-task=8`, `--mem=64G`,
1×A100** (inherited from `enc3seed_lora_curric.sbatch`; verified). Single job ⇒ peak footprint **8 CPU / 64 G /
1 GPU** — well within the 16 CPU / 128 G / 2 GPU cap, and **NEVER two 16-CPU jobs in flight** (the 29 h-wedge infra
rule). `conda activate HateVideo`; `sbatch` with **NO `--time`**; initial `PENDING (JobHeldUser)` = **WAIT for
auto-release, never force** (CLAUDE.md). Sources `conda.sh` directly and runs the ≥20 G `disk_guard.sh`; B2-pushes
`logging` at the end (derived artifacts only — videos never leave, CLAUDE.md data boundary).

**Test-touch:** the 24 head reads are the ONLY budgeted NCA-family test evaluations (8 arm×dataset cells × 3 seeds);
zero test-touch before the verdict. **The executor transcribes raw both-protocol per-seed numbers (line-numbered)
and applies NO gates/interpretation** — the verdict (G-repro → codex → smoke → KS-arm-dead → FORMAL bar, per
arm×dataset) is rendered by an **independent 0-context reviewer against this prereg VERBATIM.**

**No job is submitted by this prereg author.** Submission happens only after the independent 0-context review +
hash-freeze (+ codex gate) run by the orchestrator/executor.

---

## 7. Outcome table template (filled ONLY from raw trainlogs at verdict time)

### 7.1 Per-arm × dataset table (fill from `nca_<ARM>_{MHC_zh,HateMM}_…_seed{0,1,2}_<JID>.trainlog`)

For each arm ∈ {A1a nca_tau0.1, A1b nca_tau0.2, A2 supcon_tau0.1, A3 mixup_a2.0}, a 16-row block:

| dataset | seed | protocol | arm acc/F1 | floor acc/F1 (§2) | Δ(arm−floor) acc/F1 |
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

### 7.2 Fixed write-up format (per §3.1 rule 5 + the bars §3.2/§3.3)

```
A1a NCA τ=0.1:   MHC_zh:  final-epoch: <pass/fail>; val-selected: <pass/fail>  [FORMAL §3.2]. KS-arm-dead: <KILLED | survives>.
                 HateMM:  final-epoch: <pass/fail>; val-selected: <pass/fail>. KS-arm-dead: <KILLED | survives>.
A1b NCA τ=0.2:   MHC_zh:  final-epoch: <pass/fail>; val-selected: <pass/fail>. KS-arm-dead: <KILLED | survives>.
                 HateMM:  final-epoch: <pass/fail>; val-selected: <pass/fail>. KS-arm-dead: <KILLED | survives>.
A2  n-SupCon:    MHC_zh:  final-epoch: <pass/fail>; val-selected: <pass/fail>. KS-arm-dead: <KILLED | survives>.
                 HateMM:  final-epoch: <pass/fail>; val-selected: <pass/fail>. KS-arm-dead: <KILLED | survives>.
A3  mixup α=2.0: MHC_zh:  final-epoch: <pass/fail>; val-selected: <pass/fail>. KS-arm-dead: <KILLED | survives>.
                 HateMM:  final-epoch: <pass/fail>; val-selected: <pass/fail>. KS-arm-dead: <KILLED | survives>.
(+ KS-regression note if any Δacc ≤ −0.014; + MARGINAL note if a within-noise pass per B3 §2.2 precedent.)
```

---

## 8. What a PASS / FAIL means for the goal (D7 boundary — objectives are DEAD, not user-pending)

- **KS-arm-dead on all eight cells (recon prior — the honest expected outcome, F0.5):** the vote-aligned objective
  carries no net signal on ZH/HateMM ⇒ the loss↔inference-mismatch axis is **CLOSED** at ~0.33 GPU-h, and law-I is
  **upgraded** to "even the vote-matched objective can't move it" — the ceiling lives in the frozen LoRA-Qwen
  features, not the objective. The cleanest, cheapest outcome: a genuinely un-enumerated axis converted to a
  measured door-closer in one bite, with a strong closing paper sentence.
- **A cell survives KS but < FORMAL bar:** measured-not-promoted limbo (bank the weak positive; still D7-DEAD).
  Most plausibly the **marginal ZH val-sel** cell (the realistic deliverable = ZH val-sel *hardening*, F0.5).
- **A cell clears the FORMAL bar (≥ +0.030/+0.030, 3/3, both protocols):** a paper-worthy **robustness/ablation**
  row ("vote-aligned head objective helps on <dataset>"), most plausibly ZH. **NOT a novelty win** (F0.3): NCA /
  SupCon / mixup are generic training objectives. A surviving cell still owes the full ceremony (§3.6). Empirically
  it would say φ′ (the re-trained map) has more symmetric-reachable vote accuracy than φ₀ — consistent with F66
  being silent on φ′, not a violation.

**Framing sentence (verbatim):** *this measurement tests one cheap head-training objective family — a
deployed-vote-aligned NCA / soft-kNN loss (τ∈{0.1,0.2}), neighborhood-SupCon, and manifold mixup — on the deployed
align-fusion head over cached features, 3-seed paired dual-protocol vs each dataset's banked floor; F66 does not
bind a loss that reshapes φ (it measured only inference-side operators over a fixed φ₀), so the cell is legitimately
un-measured, with law-I as the honest ~2–4% +3 counter-pressure; it is a one-bite, knobs-frozen probe of the only
un-enumerated axis (loss↔inference mismatch), and a pass is a performance/ablation row, NEVER a novelty win — the
objective is D7-DEAD.*

---

## 9. Provenance index

- Recon (GO; 4-arm family design, pinned knobs, F66/law-I rulings, ban closures, kill skeleton, implementation
  pins): `refine-logs/NCA_FORENSIC_RECON.md` (`685df9e`).
- Cell source: `refine-logs/LITSWEEP2_HEAD_OBJECTIVES.md` (lit2-objectives ROUND-2 #1 axis).
- Deployed head + loss + vote: `src/model/classifier.py::classifier_hateClipper.forward` (align/Hadamard `x=img⊙text`),
  `src/model/loss.py::compute_loss` (triplet+0.5·BCE), `src/utils/metrics.py:262-284` (the top-20 signed-cosine
  arithmetic vote the loss must match), `src/utils/retrieval.py:341` (per-epoch FAISS reindex gate).
- Patched code (this prereg): `src/model/loss.py:24` (`nca_bank` kwarg), `:34-64` (NCA/SupCon early branch),
  `:579-586` (mixup BCE hook), `:635-717` (`_nca_head_loss`/`_supcon_head_loss`/`_manifold_mixup_bce`);
  `src/run_rac.py:550-574` (argparse), `:609-639` (`_build_nca_bank`), `:693-698` (per-epoch bank build),
  `:729`/`:770` (`nca_bank` threaded).
- Floors (re-derived §2): ZH `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`
  (`B3_PREREG_REVIEW.md`, `CAND2_CURRICULUM_PREREG.md §2.1`, `HEADRECIPE_PREREG.md §2.1`); HateMM
  `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`
  (`goal-round3-terminus`, `HEADRECIPE_PREREG.md §2.2`).
- Protocol + decision rule (verbatim): `research-wiki/experiments/exp-encoder-3seed.md:73-85`.
- Same-code anchor: `scripts/slurm/enc3seed_lora_curric.sbatch` (sha `00d9e99…`).
- Mechanism headwinds: F45 (ZH text-stream Pareto / 78-dev selection noise), law-I 8-instance pattern
  (F50/F65/F66/F67/F70…), F66 β-decomposition (`state/findings.jsonl`, a6e41f8).

**Required statements:** ZERO GPU/SLURM/Modal spent by this prereg author (only pure-CPU login-node floor
re-parsing, `py_compile`, a synthetic-tensor CPU smoke, and collision/syntax/same-code verification, seconds; no
held-out test metric produced). All floor numbers re-parsed from banked completed-run trainlogs (numeric-provenance
discipline). No `state/` mutated. No `research-wiki/` mutated. NO job submitted. Not pushed.

---

## 10. (reserved)

---

## 11. DEVIATIONS FROM THE RECON — flagged loudly

1. **DEV-1 (KILL bar uses SIGN, not "bootstrap CI straddles 0"). MATERIAL — house discipline, per frame16 DEV-1.**
   The recon §6 phrases KS-arm-dead as "val-sel AND final ≤ floor on both datasets/protocols." A CI-straddles-0
   formalism conflicts with the house n=3 **no-bootstrap** rule (`exp-encoder-3seed.md:78-79`). I pin the
   **sign-based** KS-arm-dead (§3.3): killed iff on BOTH protocols `mean Δacc ≤ 0` OR acc sign not 3/3 positive.
   Only the significance formalism changes; the qualitative bar (tie/regress on both protocols ⇒ dead) is identical
   to the recon.

2. **DEV-2 (patches EDIT `loss.py` + `run_rac.py` in place — not new files). MATERIAL / recon-mandated, same-code
   preserved.** Unlike frame16/cand-2 (new sbatch, existing code untouched), the recon's design requires additive
   edits to two tracked source files. Resolution: every edit is `getattr`/`head_loss`-gated OFF-by-default; the
   mixup hook's `else` re-emits the deployed BCE byte-for-byte (git-diff verified: the only loss.py "deletions" are
   the 6 BCE lines re-indented under `else:`); run_rac.py is purely additive (0 deletions) ⇒ flags-off behaviour is
   byte-identical and the floors need no re-run (F0.7). The patched-file shas are hash-frozen (§5) and re-verified
   at submit. This mirrors the just-landed HEADRECIPE precedent (SAM/mod-dropout edits to the same two files).

3. **DEV-3 (mixup implemented in `loss.py` by re-deriving the align fused rep, NOT by editing `classifier.py`).
   MATERIAL / favorable — keeps the patch to the 2 files the task pins.** The recon §2.3/§8 pins mixup at "the
   fused post-projection representation." I implement it in `_manifold_mixup_bce` (loss.py) by re-deriving
   `x = norm(img_proj(img)) * norm(text_proj(text))` read-only from the model's submodules (mod_dropout OFF, exactly
   the deployed align forward), mixing `x` + the soft label and re-forwarding `mlp`+`output_layer`. `classifier.py`
   is **untouched** (sha unchanged). The re-derivation draws a second projection-dropout realisation, confined to
   the A3 arm (expected divergence, F0.2); the flag-off path re-derives nothing. Pinned to `fusion_mode=='align'`
   (asserted).

4. **DEV-4 (bank stop-grad enforced BOTH at construction AND locally in the loss). Favorable.** The recon pins
   "anchor grad-on, full bank detached." I pin **both**: `_build_nca_bank` returns `.detach()`ed tensors under
   `no_grad`, AND `_nca_head_loss` re-detaches (`bank_feats.detach()`) so a future caller cannot leak grad into the
   bank. Smoke asserts a hostile `requires_grad=True` bank receives zero gradient. Stronger reproducibility /
   grad-flow guarantee (the mandatory codex focus, §4.5).

5. **DEV-5 (LOO self-exclusion pinned as an explicit id→row map + `−inf` self-mask; the numerical guard is a
   `clamp(min=−30)`). Favorable / documented.** The recon pins "LOO self-exclusion, matched by batch-id vs bank-id."
   I pin the concrete mechanism: `_build_nca_bank` returns `id_to_row`; `_nca_head_loss` masks
   `logits[arange(B), [id_to_row[v] for v in batch_ids]] = −inf`. A `clamp(logP, min=−30)` guards the (unreached at
   N≫1 binary) all-`−inf` same-class row so the tiny synthetic smoke cannot NaN. Both are the codex gate's LOO +
   numerics focus.

6. **DEV-6 (single `--nca_tau` flag serves both the NCA grid and the SupCon temperature). Neutral / documented.**
   The recon §8 named `--head_loss_tau`; the task pins `--nca_tau`. I use `--nca_tau` for BOTH the NCA τ-grid
   (0.1/0.2) and the SupCon τ (fixed 0.1, passed explicitly on the A2 rows). No behaviour change — the flag is the
   cosine-softmax temperature for whichever surrogate `--head_loss` selects; SupCon has a single pinned value (no
   grid). ProxyNCA++ is dropped (recon §0, dominated by the τ grid); in-batch-grad NCA and NCA-only (no-BCE) are
   parked follow-ups, not spent here.

7. **DEV-7 (per-epoch bank cadence built in `run_rac.model_pass`, mining path left inert for A1/A2 via the early
   return). Neutral / recon-aligned.** The recon §8 offered "the mining call may stay (pairs ignored) or be
   skipped." I pin **skipped**: the NCA/SupCon early branch returns before the FAISS mining call, so
   `train_feats/train_labels` stay None all epoch (no mining forward, no O(N²) work) and the bank is built
   independently once/epoch in `model_pass` (mirroring the FAISS reindex reset). Verified nothing else consumes the
   (absent) mined pairs (F0.6(iii)). A3 keeps triplet + mining exactly as the floor.
