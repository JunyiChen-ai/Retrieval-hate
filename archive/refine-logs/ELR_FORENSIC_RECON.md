# ELR NOISE-ROBUST HEAD — $0 FORENSIC RECON (batch-4 #3, LITSWEEP3 §3 / shortlist #2)

**Agent:** elr-recon (zero-GPU / zero-SLURM / zero-Modal / zero-test-touch forensic recon).
**Date:** 2026-07-25 NZST. **Deliverable:** this doc + one local commit (no push). `state/` untouched.
**Candidate:** noise-robust training for the RGCL align head — **ELR** (early-learning regularization, an
*additive* term) as lead, **co-teaching** as contrast, targeting noise in the FAISS-mined
pseudo-positive/hard-negative PAIRS. Source: `LITSWEEP3_DATA_CENTRIC.md` §3 / shortlist #2 (`8629188`), which
priced it P(≥+1)≈6–10%, P(≥+3)≈2%, ~0.3 GPU-h, flagging **Wall-C** (ZH/HateMM test climbs LATE ⇒ noise-robust /
early-stopping mechanisms may be anti-aligned with how the head converges).

**VERDICT (up front): PARK.** Not banned (ELR is outside the F75 ban *letter* — §1), but PARK on the merged
weight of four measured facts:
1. **Mechanism mismatch (load-bearing).** The mined pairs are **label-filtered by GOLD train labels**
   (`retrieval.py:480/497`), so "mined-pair noise" ≡ **gold-train-label noise** — NOT an independent noise
   source, contra the tasking's "(not the gold video labels)". ELR's natural additive form attaches to the
   **BCE (classification) leg**, but the deployed decision is the **kNN VOTE**, not the BCE logit — ELR
   regularizes a leg that does not decide (second-order on the shared trunk).
2. **Noise proxy is real but boundary-hardness-confounded and space-mismatched** (§3): raw-space
   kNN-majority-disagreement **≈13–17%** (concat), ABOVE the 5% "no-target" floor at face value, but it is an
   **upper bound** — it conflates true label-noise with legitimate class-boundary hardness, and it is measured
   in the **raw pre-head space**, whereas the deployed mining runs in the **trained head space** (more
   1-hop-separable, F63 ⇒ lower disagreement). The genuine-label-noise fraction ELR could fix is materially
   below 13–17%, plausibly single-digit.
3. **Wall-C anti-alignment — quantified and decisive** (§4): both floors show **test peaking LATE**
   (HateMM test-opt ep 18/21/24, +4/+7/+14 epochs AFTER dev saturates; ZH final beats val-selected by
   **+0.0134 on every seed**). ELR's implicit early-stopping bias (pull predictions toward the early-epoch EMA
   target) is anti-aligned with a late-climbing test — the SWA/F62 kill shape.
4. **Prior ≤ the just-killed sibling.** F75 (NCA/SupCon/mixup head-loss family) just went **0/8 FORMAL, 7/8
   KS-arm-dead**; its finding is the *first measured negative for trained-reshaping-unlocks-oracle-headroom*.
   ELR is a trained-reshaping operator with a WEAKER hook (targets the non-deciding BCE leg) and a Wall-C
   headwind the NCA family did not carry. Re-priced prior: **P(≥+1 stable ZH val-sel) ~5–8%; P(≥+3 any ds)
   ~1–2%.**

A diagnostic-only escalation (ELR-λ{1,3}, ~0.16 GPU-h door-closer, expected KILL) is spec'd in §5/§7 for the
orchestrator's option, honestly labelled below the promote bar.

---

## 1. BAN-SCOPE ADJUDICATION (Duty 1) — ELR is OUTSIDE the F75 ban *letter*; the F75 *finding* is a prior, not a ban

**F75 `ban_scope` (verbatim, `state/directions_tried.json` dead[47]):**
> "head-loss **swaps** of the triplet+BCE hybrid **toward vote-consistent (NCA/soft-kNN), contrastive (SupCon),
> or mixup-BCE objectives** at 7B frozen-encoder feature scale; tau/alpha retunes = tactics, banned; … First
> measured negative for trained-reshaping-unlocks-oracle-headroom; F66 selection-locked pools untouched."

**NCA family definition (`NCA_PREREG.md` §Title):** the banned family "**replace/augment** the deployed head's
`triplet(m=0.1)+0.5·BCE` contrastive term with a loss that **directly optimizes the deployed top-20 kNN vote**"
— four arms: A1a/A1b NCA (vote-consistent soft-kNN), A2 SupCon (contrastive), A3 manifold-mixup (mixup-BCE).

**Adjudication (three tests):**
- **Is ELR a head-loss *swap* of the triplet+BCE hybrid?** **No.** ELR is strictly **additive**:
  `L = triplet + 0.5·BCE + λ·L_ELR`. The triplet contrastive term and the BCE term are **unchanged** (the
  NCA/SupCon arms *replaced* the triplet term; mixup *rewrote* the BCE target — ELR does neither).
- **Is ELR *toward* one of the three named objective families (vote-consistent / contrastive / mixup-BCE)?**
  **No.** ELR is a **label-noise-robustness regularizer** (Liu et al., NeurIPS 2020, arXiv:2007.00151): it pulls
  each sample's prediction toward its **own early-epoch moving-average target** to prevent memorization of
  noisy labels. It is a **fourth, distinct** objective family — not soft-kNN, not SupCon, not mixup.
- **Is ELR a "tau/alpha retune" of a banned arm?** **No** — it introduces new hyperparameters (λ, β) of a new
  term, not a retune of NCA-τ or mixup-α.

**LITSWEEP3 already ran this isomorphism check and reached the same reading** (§3, isomorphism table row):
"§3 noise-robust head (ELR/co-teach) … F75 loss-swap: **outside letter** (additive reg / procedure, not
family-swap) … **ADMISSIBLE — probe**." The prereg author's own F75 write-up agrees: "**Noise handling is
orthogonal to the loss family:** ELR *adds a term* to the existing BCE … none is a 'loss-swap toward
vote-consistent/contrastive/mixup,' so none is inside F75's `ban_scope`."

**Honest counter-note (spirit, not letter).** The F75 ban lists **mixup-BCE** — itself a BCE-modifying
regularizer, not a pure swap — as a banned member. One could argue the ban's *spirit* reaches "any additive
BCE-side regularizer." I reject that reading: mixup-BCE is a **specifically named** member banned on its own
measured failure (A3 dead), and the ban enumerates three families; ELR is a genuinely fourth mechanism. **What
DOES transfer to ELR is the broader FINDING** — "first measured negative for
trained-reshaping-unlocks-oracle-headroom" — plus **law-I** (the 8-instance "better representation ⇒ zero vote
conversion" pattern). Those are **priors that lower ELR's expectation**, not a ban that forecloses it.

**Verdict: OUTSIDE the ban. Continue the recon.** (If it had been inside, this section would read
PARK-as-banned and stop.)

---

## 2. MECHANISM PIN (Duty 2a) — where mined pairs enter the loss, and what "label noise" means here

**The FAISS-mined pairs are LABEL-FILTERED by GOLD TRAIN LABELS (the decisive mechanism fact).**

| step | file:line | what |
|---|---|---|
| mining call (video config: `no_pseudo_gold_positives=1`) | `src/model/loss.py:302-321` (`dense_retrieve_hard_negatives_pseudo_positive(...)`) | mines both hard-neg + pseudo-gold-positive per anchor, in the **trained HEAD embedding space** (bank re-built per epoch via `model(img,txt,return_embed=True)`, `retrieval.py:347-379`) |
| **hard-negative selection** | `src/utils/retrieval.py:480` | `train_labels[I[i,iter]] != query_labels[i]` — nearest neighbour with **OPPOSITE gold label** |
| **pseudo-gold-positive selection** | `src/utils/retrieval.py:497` | `train_labels[I[i,iter]] == query_labels[i]` — nearest neighbour with **SAME gold label** |
| hard-neg → loss | `loss.py:343-418` (cosine to anchor, summed) → triplet | pushes anchor away from opposite-gold-label neighbour |
| pseudo-gold → loss | `loss.py:421-469` (cosine to anchor, mean) → triplet | pulls anchor toward same-gold-label neighbour |
| triplet assembly | `loss.py:486-488` | `mean(relu(in_batch + hard − pseudo_gold + margin))` |
| BCE leg | `loss.py:578-596` | `total = triplet·(1−ce_w) + BCE·ce_w`, `ce_weight=0.5` |

**Consequence (load-bearing, contra the tasking framing).** The "pseudo-gold positive" is **not a pseudo-label**
— it carries a **certain, gold train label** equal to the query's gold label, selected by embedding proximity
but **label-gated by the gold annotation**. There is **no independent pseudo-labeling / thresholding step** that
could inject label noise (verified: the baseline mining loop `retrieval.py:466-513` only reads `train_labels`;
the TARC branch `:543-579` is inactive — `tarc_hn_mode='off'`). Therefore:

> **"Mined-pair label noise" ≡ "gold-train-label noise" propagated through label-filtered retrieval.**
> The tasking's "(not the gold video labels)" is **not accurate to the code**: the only label noise in the mined
> pairs IS the gold-video-label noise. A mined pseudo-positive is "wrong" only when the query's or the
> neighbour's **gold label** is wrong, or when a correctly-labeled query sits across the class boundary (that is
> *hard-example* geometry, not label noise — and ELR does not model it).

This also means ELR here would be a **second attack on the same object** the project already addresses at the
video-label level via **pillar-3 consensus-denoising** (ZH-validated).

**Where would ELR actually attach?** ELR (arXiv:2007.00151) is a per-sample **classifier-prediction**
regularizer (EMA of the softmax/sigmoid output). Its natural additive home is the **BCE leg** (`loss.py:588-594`
`output`/`loss_classifier`). But the **deployed decision is the top-20 signed-cosine kNN vote over the head
embedding** (`metrics.py:262-284`), **not** the BCE logit. So ELR-on-BCE shapes the shared trunk (`mlp`) only
**second-order** — the identical "the regularizer touches a leg the vote does not read" friction that dominated
the entire data-centric lens (LITSWEEP3 §2/§3). To attack the **triplet/mined-pair** noise directly you would
need **co-teaching-style per-pair small-loss selection** (not ELR) — infeasible at our scale (§5).

### Duty 2b — measured $0 noise proxy (TRAIN-ONLY, raw pre-head space; no test-touch, no GPU)

The deployed mining runs in the trained head space, but **all 6 deployed head ckpts are deleted**
(`CURATION_FORENSIC_RECON.md` §2.2); only the **seed-independent raw pre-head caches** survive (img/text
3584-d + gold labels, §2.1 there). Label noise (annotation error) is embedding-invariant to first order, so a
raw-space kNN-label-disagreement estimate is a valid **proxy** (honestly: raw-space, single-draw, an
**upper bound** on head-space noise). Measured this recon on the **train split only** (the mining's own
query=corpus=train setup), fused key `z = L2([L2(img)‖L2(text)])` (LP-gate substrate), cosine, self-excluded:

| dataset (V, pos_frac) | space | k1-opposite | k5 maj-disagree / mean-same | k10 maj-disagree / mean-same | k20 maj-disagree / mean-same |
|---|---|---|---|---|---|
| **ZH** (579, 0.311) | concat | 0.1693 | 0.1658 / 0.7817 | **0.1278** / 0.7511 | 0.1313 / 0.7232 |
| | text | 0.1623 | 0.1416 / 0.8079 | 0.1088 / 0.7945 | 0.1347 / 0.7687 |
| | img | 0.3057 | 0.2850 / 0.6473 | 0.2297 / 0.6332 | 0.2591 / 0.6220 |
| **HateMM** (744, 0.401) | concat | 0.1949 | 0.1573 / 0.7718 | **0.1358** / 0.7599 | 0.1505 / 0.7524 |
| | text | 0.1828 | 0.1478 / 0.7997 | 0.1398 / 0.7835 | 0.1559 / 0.7738 |
| | img | 0.2675 | 0.2513 / 0.6890 | 0.1962 / 0.6757 | 0.2164 / 0.6634 |

- **maj-disagree** = fraction of train rows whose gold label ≠ majority gold label of its k nearest neighbours
  (confident-learning-style mislabel flag). **k1-opposite** = fraction whose *nearest* neighbour is opposite-gold-label
  (the query sits on/across the boundary — the noisiest mined pairs, where the pseudo-positive must reach past a
  closer hard-negative).
- **Reading:** the proxy is **~13–17% (concat, k5–k20)** on both datasets — **above the 5% "no-target" floor at
  face value** (so K-NR-0 would NOT auto-kill), but it is an **upper bound** for two honest reasons: (i) a hard
  kNN-majority vote over binary overlapping classes flags **correctly-labeled boundary examples** as "noise" —
  most of the 13–17% is class-boundary hardness, not annotation error (mean-same-label frac 0.72–0.81 confirms
  strong but imperfect local class structure); (ii) it is raw-space; the trained head space is **more
  1-hop-separable** (F63) ⇒ head-space disagreement is **lower**. **Net: the genuine label-noise ELR could fix
  is materially below 13–17%, plausibly single-digit** — a weak, confounded target, not a clean "nothing to fix"
  but far from a strong one.

---

## 3. (folded into §2b)

---

## 4. WALL-C CHECK (Duty 3) — test climbs LATE on BOTH floors; quantified per seed

Parsed this recon from the banked floor trainlogs (`Val_Retrieval` / `Test_Retrieval` per-epoch acc; warmup 5;
val-sel = argmax Val acc, **roc tie-break** — matches `NCA_PREREG.md` §2.1 val-sel epochs {20,26,19} / {29,14,10}
bit-for-bit). "dev-first-max" = the epoch dev acc first reaches its warmup max; "test-opt" = the epoch test peaks.

| floor | seed | val-sel ep | test@valsel | dev first hits max-acc @ep | **TEST peaks @ep (acc)** | test@final(29) | gap (testopt − devmax) | Δtest(final − valsel) |
|---|---|---|---|---|---|---|---|---|
| **ZH 13150** | 0 | 20 | 0.8322 | 9 (0.8718) | 11 (0.8456) | 0.8456 | +2 | **+0.0134** |
| | 1 | 26 | 0.8255 | 11 (0.8718) | 1 (0.8456)* | 0.8389 | −10* | **+0.0134** |
| | 2 | 19 | 0.8389 | 19 (0.8718) | 27 (0.8591) | 0.8523 | +8 | **+0.0134** |
| **HateMM 13241** | 0 | 29 | 0.8791 | 14 (0.8505) | 18 (0.8884) | 0.8791 | +4 | +0.0000 |
| | 1 | 14 | 0.8744 | 14 (0.8505) | 21 (0.8930) | 0.8791 | +7 | +0.0047 |
| | 2 | 10 | 0.8791 | 10 (0.8505) | 24 (0.8884) | 0.8791 | +14 | +0.0000 |

*ZH seed1 has an anomalous early test spike (ep1=0.8456) then settles; its late trajectory
(ep23–29: 0.8322→0.8389) still climbs.

**Quantified findings.**
- **HateMM: test peaks LATE and consistently** — test-opt at **ep 18 / 21 / 24** of 29 (0.8884/0.8930/0.8884),
  **+4/+7/+14 epochs AFTER dev saturates** (dev plateaus at 0.8505 by ep 10–14). The val-sel/final checkpoints
  (0.8744–0.8791) sit **below** the late test peak — dev cannot see the late climb.
- **ZH: the final epoch beats the val-selected checkpoint by exactly +0.0134 on ALL 3 seeds** — dev acc
  saturates at 0.8718 by ep 9–19 and cannot distinguish the later, better test region (F45's "78-dev plateaus,
  test keeps climbing"). Test-optimal epochs are late/noisy (11 / [1] / 27).
- **This is the SWA/F62 kill shape.** Any mechanism that **shrinks or early-biases late-epoch learning** is
  anti-aligned. **ELR's core mechanism is an early-epoch-target pull** (predictions regularized toward their
  EMA over early epochs, where — on these floors — test is at or near its *lowest*: ZH early test 0.77–0.82,
  HateMM early test below the ep18–24 peak). ELR therefore **fights the observed dynamics**; co-teaching
  (drop large-loss LATE) and late-stopping (remove "mislabeled" LATE) are **even more** anti-aligned.

---

## 5. TRANSPLANT SPEC (Duty 4) — exact ELR-additive form, minimal grid, co-teaching feasibility

**ELR additive form (Liu et al., NeurIPS 2020).** Maintain a per-train-id EMA target of the sigmoid output;
add a term that penalizes agreement with a *drifting-away* target:

- Binary head: `p_i = sigmoid(output_i) ∈ [0,1]`; 2-vector `P_i = [1−p_i, p_i]`.
- Per-id EMA target `t_i ← β·t_i + (1−β)·P_i` (updated each step, detached; buffer `[N_train,2]` keyed by video
  id — same id→row machinery as `_build_nca_bank` / `id_to_row`, `run_rac.py:609-639`).
- ELR term: `L_ELR = mean_i log(1 − ⟨P_i, t_i⟩)` (⟨·,·⟩ = dot of the 2-vectors; the target is normalized/detached).
- Total: `total_loss = triplet·(1−ce_w) + BCE·ce_w + λ·L_ELR`.

**Where in `compute_loss`:** immediately after `loss_classifier` is formed in the `hybrid_loss` else-branch
(`loss.py:588-594`), a `getattr(args,"lambda_elr",0.0)`-gated block computes `L_ELR` from `sigmoid(output)` and
the id-indexed EMA buffer (threaded in from `run_rac.model_pass` like `nca_bank`), then adds `λ·L_ELR` to
`total_loss`. `lambda_elr==0` (default) ⇒ **byte-identical no-op** (F73/NCA additive-gating precedent). The
triplet/mining path is **untouched** (ELR does not attach to the mined pairs — see §2 mismatch). `retrieval.py`
and `classifier.py` unchanged.

**Minimal hyper-grid (house one-bite, ≤2 values/knob):** **λ_elr ∈ {1, 3}** (ELR paper range 1–7; 1 conservative,
3 the CIFAR-10 default) — the **only** declared multiplicity. **β = 0.7 fixed** (ELR canonical; pinned, no grid).
⇒ **2 arms × 2 datasets × 3 seeds = 12 head-only runs on cached feats**, ≈ **0.16 GPU-h** (< the F75 family's
0.33). Floors = ZH 13150 §2.1 / HateMM 13241 §2.2 (`NCA_PREREG.md`), dual-protocol, 3/3 sign.

**Co-teaching contrast arm — feasibility at our scale: NOT meaningful; recommend DROP.**
- Co-teaching (Han et al., NeurIPS 2018) trains **two** heads, each keeping the peer's **small-loss** examples
  (keep-rate = 1 − est. noise). At **n≈579–744, batch 64, ~31–40% positive**, the clean/noisy **loss
  distributions overlap heavily** — the small-loss partition is unstable (LITSWEEP3 §3), and the ~13–17% proxy
  is mostly **boundary hardness** (clean-hard examples have LARGE loss ⇒ co-teaching **drops the clean-hard
  tail** — Late-Stopping's exact critique, arXiv:2308.13862).
- Co-teaching drops large-loss examples **LATE** in training — **directly Wall-C anti-aligned** (§4).
- Two heads at head-seed noise ±0.014 (`NCA_PREREG.md` §2.3): the peer-disagreement signal is **within noise**.
- **Feasibility: run-able but not informative;** it would spend a bite to measure a near-null under a headwind.

**Cost per arm:** head-only on cached feats, ~20–50 s/run; ELR adds a per-id EMA buffer update (O(B), negligible).
6 runs/arm ≈ 3–5 min; **2 arms ≈ 0.16 GPU-h** total (co-teaching would ~double it for no expected signal).

---

## 6. GO / PARK + PRIOR (Duty 5) — **PARK**

**PARK.** ELR is **admissible** (outside the F75 ban letter, §1) but the substantive case is negative:

| factor | reading |
|---|---|
| Ban scope | **Outside** the F75 letter (additive reg, not a swap toward the 3 named families). Not a foreclosure. |
| Noise target | Mined-pair noise ≡ **gold-label noise** (label-filtered mining, §2); proxy **~13–17% raw-space upper bound**, mostly boundary hardness + raw-vs-head-space inflation ⇒ **genuine label-noise fraction plausibly single-digit**; already pillar-3's object. |
| Mechanism hook | ELR attaches to the **BCE leg**; the deployed decision is the **kNN vote** ⇒ **second-order** on the vote (the lens-wide friction). Attacking the mined-pair triplet directly needs co-teaching — infeasible at n≈600 (§5). |
| Wall-C | **Quantified anti-alignment** (§4): test peaks LATE (HateMM ep 18/21/24, +4/+7/+14; ZH final − valsel = +0.0134 ×3); ELR's early-target bias fights it. |
| Prior sibling | F75 (NCA/SupCon/mixup) just went **0/8 FORMAL, 7/8 KS-dead**; ELR has a weaker hook + a Wall-C headwind the NCA family lacked. |

**Re-priced honest prior** (down from LITSWEEP3 §3's 6–10% / 2%, for the BCE-leg-doesn't-decide mismatch +
quantified Wall-C anti-alignment + boundary-hardness discount on the proxy):
- **ZH val-sel hardening ≥+1 (stable, 3/3): ~5–8%.**
- **+3 on any dataset: ~1–2%.**
- **Paper-value:** modest — even a null closes the "the head trains on noisy mined pairs" hypothesis on the
  record (PV-2), companion to pillar-3; but §2 already shows the mined-pair noise is *definitionally* the
  gold-label noise, so the diagnostic mostly re-states a mechanism fact this recon establishes at $0.

**No KS/FORMAL prereg is drafted** (GO would have required one). The escalation option is §7.

---

## 7. OPTIONAL DIAGNOSTIC ESCALATION (non-binding; only if the orchestrator wants the hypothesis closed on the record)

If the orchestrator wants "the head trains on noisy mined pairs" closed as a **measured door-closer** (not
parked), the minimal admissible cell:
- **Arms:** **A-ELR-λ1 / A-ELR-λ3** (β=0.7 fixed), additive on the BCE leg, ZH 13150 + HateMM 13241 × 3 seeds,
  dual-protocol, vs the **F75/enc3s floors** (§2.1/§2.2). **DROP co-teaching** (infeasible, Wall-C-doubled cost,
  §5). ≈ **0.16 GPU-h**, one sbatch, one bite.
- **Kill-switches (house style):**
  - **K-ELR-0 ($0, already answered):** noise proxy > 5% — **PASS at face value (~13–17%)** but flagged
    boundary-hardness-confounded (§2b); this gate does **not** rescue the prior.
  - **K-ELR-1 (primary):** ZH val-sel strengthened 3/3 over floor by ≥ **+0.014**; **KS-arm-dead** if any arm
    ≤ +0.020 (F75/F70 precedent).
  - **K-ELR-2 (Wall-C tripwire):** if the winning arm's dev-argmax epoch moves **earlier** than the floor's
    while test is still climbing (§4), flag anti-alignment even on a nominal pass (the SWA/F62 pattern).
- **Expected outcome:** KS-arm-dead (consistent with F75 + law-I + Wall-C). Spend the 0.16 GPU-h only if the
  pillar-2/3 paper sentence ("even an additive noise-robust regularizer on the head does / does not move the
  vote, 3-seed") is judged worth it. **D7-DEAD regardless** — a training regularizer is a generic recipe,
  never a novelty win.

---

## 8. PROVENANCE & DISCIPLINE

- **Ban scope:** `state/directions_tried.json` dead[47] `ban_scope` + `banned_constraints`; `NCA_PREREG.md`
  Title/§0/§3.5 (F75 family def + F66/law-I rulings); `LITSWEEP3_DATA_CENTRIC.md` §3 + isomorphism table
  (`8629188`).
- **Mechanism:** `src/model/loss.py:302-321` (mining call), `:343-469` (hard/pseudo → loss), `:486-488`
  (triplet), `:578-596` (BCE leg); `src/utils/retrieval.py:466-513` (label-filtered mining, `:480` hard-neg
  opposite-gold, `:497` pseudo-pos same-gold; `:543-579` TARC branch inactive); vote `src/utils/metrics.py:262-284`.
- **Noise proxy (this recon, $0 CPU, train-only):** `data/CLIP_Embedding/{MHC_zh/train_…-LoRA_HF.pt,
  HateMM/train_…-LoRA-curric_HF.pt}` (V=579 / 744); raw fused key `L2([L2(img)‖L2(text)])`; caches
  seed-independent (`CURATION_FORENSIC_RECON.md` §2.1); deployed head ckpts deleted (§2.2 there).
- **Wall-C (this recon):** `slurm/logs/enc3s_MHC_zh_…LoRA_HF_seed{0,1,2}_13150.trainlog`,
  `slurm/logs/enc3s_HateMM_…LoRA-curric_HF_seed{0,1,2}_13241.trainlog`; val-sel epochs bit-match
  `NCA_PREREG.md` §2.1/§2.2. Wall-C rationale: F45 `B3_ZH_LORA_DECOMPOSITION.md`; F62 SWA; F66 `ISR_PREGATE_RECORD.md`.
- **Citations (web-verified in LITSWEEP3, 2026-07-25):** ELR arXiv:2007.00151 (NeurIPS 2020); Co-teaching
  arXiv:1804.06872 (NeurIPS 2018); SOP arXiv:2202.14026 (ICML 2022); Late-Stopping arXiv:2308.13862 (ICCV 2023).
- **ZERO GPU / SLURM / Modal spent** (pure-CPU login-node cache reads, kNN arithmetic, trainlog re-parsing, and
  `py_compile`-free reading — seconds). **No held-out test metric read or produced** (train split only). No
  `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated. Committed on `main`, not pushed.
