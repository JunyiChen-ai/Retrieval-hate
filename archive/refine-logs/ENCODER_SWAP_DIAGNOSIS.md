# ENCODER-SWAP MECHANISM DIAGNOSIS — why frozen Qwen2.5-VL-7B beats CLIP on HateMM but not MHC

**Author:** diagnostic analyst (ZERO GPU, banked artifacts only; no Modal, no model
inference, zero test-touch — train+dev features only). **Date:** 2026-07-17.
**Question (project's biggest unexplained fact):** the frozen Qwen2.5-VL-7B → RGCL-head
encoder swap gives **+5.3–5.6 acc on HateMM** (3/3 seeds, both protocols;
`exp-encoder-3seed.md`, `040adb8`) yet **fails on MHC-EN** (both protocols) and **MHC-ZH**
(frozen B1: −0.0112 acc). The "dilution hypothesis" was FALSIFIED by C2-SAV
(`SAV_F1_VERDICT_REVIEW.md`); no accepted mechanism existed.

**Scripts (committed alongside, reproduce every number here):**
`scripts/analysis/encoder_swap_geometry.py` (core kNN/geometry read-outs),
`scripts/analysis/encoder_swap_diagnosis_tables.py` (→ `..._out.json`, tables T1–T6).
All read-outs are on **train + dev only**; the RGCL head is a shallow per-modality linear
map + L2-norm feeding a top-20 cosine kNN vote, so raw-frozen-feature kNN geometry is a
faithful substrate for what the head can do. Machinery is validated (below) by reproducing
the downstream sign on the two datasets whose dev tracks test.

---

## TOP-LINE MECHANISM (one paragraph)

The encoder swap is a pure **representation** upgrade, and Qwen's upgrade is **real and
roughly equal in magnitude on all three datasets** (top-20 kNN neighborhood purity rises
+0.023/+0.023/+0.021; the text stream ranks +0.041/+0.054/+0.045 AUC better everywhere).
It converts to **accuracy on HateMM only**, for two compounding reasons that are both
properties of the *datasets*, not fixable properties of the method:

1. **Modality locus.** HateMM's hate is **visually grounded** — the image stream alone
   ranks hate at AUC 0.83, and Qwen's stronger multimodal text stream (+0.041 AUC) rides on
   top of a *neutral, already-strong* image stream, so the fused gain is a **clean
   Pareto move: hate-recall +0.116 at zero cost to non-hate recall.** On **MHC-EN the visual
   channel is near-useless for the VLM** — Qwen's image stream **collapses to near-chance
   (AUC 0.734→0.599)** — and because RGCL fuses image and text as **equal-weight L2-normed
   blocks**, that collapse **cancels the text gain in the concat**, netting ≈0.

2. **Representation-limited vs label-limited errors.** On HateMM the residual errors are
   *representation-limited* (a better encoder fixes borderline cases). On MHC the residual
   errors are a **hard / label-limited core** (SAV: MHC-EN is "data/label-limited at frozen
   7B read-out"; native Qwen centroid *below* the majority baseline). A better representation
   there **reshuffles the ranking (AUC up) but rotates rather than Pareto-improves the
   decision** (hate-recall +0.040 bought with non-hate −0.036) — an easy-example-ordering /
   unconvertible gain, exactly what B5 proved for ZH at the label-oracle threshold.

**This is non-isomorphic to the falsified dilution hypothesis** (which was a *quantity*
claim: MHC hate diluted across more benign content). This is a *locus + error-type* claim,
evidenced by a per-modality AUC decomposition and per-class recall the dilution work never
did. **It opens no new axis** (§7): every lever it suggests is encoder-class (Axis B,
D7-novelty-dead) or frozen-feature reorganization (W2-E/C3geo family, banned) and still
hits the MHC conversion wall. It **re-confirms the terminus** and sharpens the paper's
"encoder-swap is HateMM-specific *because…*" story.

---

## 1. Machinery validation (does raw-frozen kNN reproduce the downstream signs?)

Train→dev kNN vote (memory = train, query = dev, k=20, cosine-weighted signed vote), concat:

| dataset | CLIP dev acc/mF1 | Qwen dev acc/mF1 | Δacc (dev) | downstream Δacc (test) | match? |
|---|---|---|---|---|---|
| HateMM | 0.785 / 0.781 | 0.832 / 0.831 | **+0.047** | +0.053…+0.056 (PASS 3/3) | ✔ sign+size |
| MHC-EN | 0.762 / 0.721 | 0.750 / 0.715 | **−0.012** | +0.006 final / +0.019 val (FAIL) | ✔ ≈0/fail |
| MHC-ZH | 0.705 / 0.593 | 0.859 / 0.848 | +0.154 | −0.0112 (B1 frozen FAIL) | ✘ dev≫test |

The read-out reproduces HateMM (sign **and** magnitude) and MHC-EN (≈0/fail). **MHC-ZH dev is
not trustworthy** and must defer to the formal test verdicts: the CLIP-ZH dev collapses
(hate-recall 0.25, mF1 0.593 on n=78) so the dev Qwen−CLIP gap is inflated; on test CLIP-ZH
is fine (project floor ≈0.85) and frozen-Qwen ZH fails (B1). B5 already established *why* the
frozen-Qwen ZH edge does not survive: its roc +0.050 is easy-example ordering, unconvertible
at any threshold including the label-oracle cut. ZH is therefore **not a counterexample** to
the mechanism — it is the same conversion wall as MHC-EN, reached by a different proximate
route (§4).

---

## 2. The representation gain is REAL and roughly EQUAL across datasets

Two encoder-agnostic quality measures (train, larger sample) show Qwen improves the space by
a **similar amount everywhere** — the datasets do NOT differ in *how much* Qwen helps the
representation:

| measure (Qwen7B − CLIP) | HateMM | MHC-EN | MHC-ZH |
|---|---|---|---|
| top-20 kNN neighborhood purity (T2) | **+0.023** | **+0.023** | **+0.021** |
| dev hard-case acc, low-CLIP-margin third (T5) | +0.114 | +0.115 | +0.231 |

The near-identical purity gain (+0.023 on all three) is the crux: **Qwen makes local
neighborhoods cleaner by the same margin on every dataset, yet only HateMM's accuracy moves.**
Whatever explains the differential is therefore NOT "Qwen helps HateMM's representation more"
— it is *where the gain lands* and *whether the decision metric can absorb it*.

---

## 3. Modality decomposition — the core finding (T1, T2)

Split each encoder into image-only and text-only kNN read-outs. Train-LOO AUC (k=20, n=549–744,
the reliable large-sample view):

| stream | HateMM CLIP→Qwen | MHC-EN CLIP→Qwen | MHC-ZH CLIP→Qwen |
|---|---|---|---|
| **image** | 0.826 → 0.817 (**−0.009**) | 0.734 → **0.599 (−0.135)** | 0.718 → 0.721 (+0.003) |
| **text** | 0.847 → 0.888 (**+0.041**) | 0.797 → 0.851 (**+0.054**) | 0.802 → 0.847 (**+0.045**) |
| concat | 0.867 → 0.883 (+0.016) | 0.801 → 0.825 (+0.023) | 0.764 → 0.840 (+0.076) |

Two facts:

- **Qwen's TEXT stream is uniformly better** (+0.041/+0.054/+0.045 AUC) — a stable,
  dataset-independent gain (Qwen's LLM text/transcript+context representation beats CLIP's
  77-token text encoder). It is **not** an empty-transcript artifact: the gain is *largest on
  MHC*, which has ≈0 empty transcripts (§5).
- **Qwen's IMAGE stream is dataset-dependent and is the whole story.** On HateMM it is neutral
  (−0.009) on top of an *already strong* image signal (0.83). On **MHC-EN it collapses to
  near-chance (0.599)** — Qwen's 8-frame LLM-pooled image summary carries almost no hate signal
  for MHC's content, where CLIP's dense patch pooling at least reaches 0.734.

Because RGCL fuses image and text as two **equal-weight L2-normed 1024-d blocks**, the fused
outcome is dominated by whichever stream is worse:

- **HateMM:** strong image (neutral swap) + better text → fused gain **converts**.
- **MHC-EN:** collapsed Qwen image **cancels** the +0.054 text gain in the 50/50 concat →
  fused Δacc ≈ 0 (dev −0.012). Confirmed structurally: for CLIP, adding the image stream lifts
  MHC-EN text 0.650→0.762; for Qwen it barely moves 0.738→0.750 — Qwen's image simply has less
  to add, so the swap forfeits the image contribution CLIP was providing.

> The trained head has *some* capacity to attenuate the collapsed image block, but the banked
> **test** result (MHC-EN frozen-Qwen FAIL) shows it does **not** net-recover — so the
> fusion-cancellation is a faithful account of the observed failure, not merely a raw-feature
> artifact.

---

## 4. Why the MHC gain does not CONVERT — Pareto vs rotation (T3, T4, T5)

Per-class recall at the concat dev vote (minority = hate):

| dataset | encoder | non-hate recall | hate recall | reading |
|---|---|---|---|---|
| HateMM | CLIP → Qwen | 0.766 → **0.766** | 0.814 → **0.930** | **Pareto:** +0.116 hate recall, **zero** non-hate cost |
| MHC-EN | CLIP → Qwen | 0.836 → **0.800** | 0.600 → **0.640** | **rotation:** +0.040 hate bought with −0.036 non-hate → wash |
| MHC-ZH | CLIP → Qwen | 0.960 → 0.880 | 0.250 → 0.821 | dev-collapse of CLIP (see §1); defer to B1 test |

- HateMM's gain is a **clean minority-recall improvement** — the fingerprint of an encoder that
  *reads hateful imagery CLIP misses*, on a dataset where that imagery is decisive.
- MHC-EN's gain is a **rotation** the ranking metric (AUC +0.057 dev) rewards but accuracy/mF1
  does not — the encoder reshuffles a hard core without net-resolving it.

Error-set overlap (T4) says the same: on HateMM Qwen **net-fixes +5** dev videos; on **MHC-EN
net −1** (fixes 11, breaks 12) — no coherent subgroup is repaired, errors are swapped in and
out. This is the SAV "data/label-limited" core made concrete, and it matches B5's proof that
the MHC/ZH ranking edge is unconvertible at any operating point.

---

## 5. Dataset composition differentials that drive the above (T2, T6)

- **Image informativeness.** Hate is far more visually grounded in HateMM: image-only train-LOO
  AUC = **0.826 (HateMM)** vs 0.734 (MHC-EN) / 0.718 (MHC-ZH). HateMM's founding content is
  explicit visual hate (violence, gore, symbols/memes); MHC (YouTube/Bilibili) hate is largely
  implicit / speech- and context-borne over ordinary footage. **This is the root asymmetry:**
  the modality Qwen most improves-or-degrades (image) only *matters* on HateMM.
- **Transcript coverage (T6).** HateMM has **5.6%** degenerate CLIP text embeddings (empty/near-
  empty transcripts → identical vector); MHC-EN/ZH have **0.2%/0.3%**. This is a *secondary*
  factor: it slightly handicaps CLIP's HateMM text stream, but the Qwen text gain is uniform and
  actually largest on the transcript-complete MHC datasets, so empty transcripts are **not** the
  driver.
- **Scale does not rescue the image collapse.** Qwen-**32B** image-only AUC on MHC-EN = **0.608**
  (still collapsed), and 32B MHC-EN concat train-LOO AUC = 0.767 < 7B 0.825 < CLIP 0.801. The
  mechanism therefore **predicts B2** ("scale regresses on MHC; not the conversion lever"): the
  MHC visual channel is uninformative for the VLM family regardless of size.

---

## 6. Reconciliation with the campaign's prior verdicts (this diagnosis unifies them)

| prior finding | how this diagnosis explains / localizes it |
|---|---|
| **SAV #18** — dilution FALSIFIED; "MHC-EN data/label-limited," native Qwen centroid < majority | localized: the *image stream* is the collapsed component (AUC 0.599); the errors are a hard rotation core, not a dilution of quantity |
| **B5** — ZH/MHC Qwen ranking edge = easy-example ordering, unconvertible at any threshold | the "rotation not Pareto" pattern (§4) IS the unconvertible edge, seen here as +AUC / ≈0-acc |
| **B2** — 32B "scale regresses," not the lever on MHC | 32B image stays collapsed on MHC (0.608); scale cannot fix an uninformative visual channel |
| **encoder-swap #positive** — HateMM +5.3, robust | fully accounted: text-gain + neutral-strong image on a visually-grounded, representation-limited problem → Pareto minority-recall +0.116 |

---

## 7. Does this open a NON-isomorphic axis? — honest verdict: **NO**

Judged against the 9 closed axes (`TERMINUS_round3_mllm_plus3.md`) and the banned list
(`directions_tried.json`):

1. **The whole finding lives inside Axis B (encoder identity / representation),** which the user
   ruled **D7-novelty-dead** (F24). Explaining *why* the swap is HateMM-specific does not lift
   that ruling.
2. **The one lever the modality asymmetry surfaces — a cross-encoder hybrid (Qwen text-stream ⊕
   CLIP image-stream)** — is dead three ways: (i) still encoder-class (Axis B, D7-dead); (ii) a
   zero-training reorganization of frozen features = the **W2-E / C3geo banned meta-family**;
   (iii) low performance prior — on MHC it would at best lift **AUC**, the exact quantity B5
   proved **unconvertible to accuracy**, and the core is label-limited (SAV), so it cannot clear
   the +0.03 **accuracy** bar on ≥2 datasets.
3. **"Down-weight/gate the collapsed image stream on MHC"** = modality gating / learned fusion
   weights = textbook, decision-side (Axis A conditional-redundancy), D7-dead; the trained head
   already has attenuation capacity and still failed on test.
4. Nothing here escapes the frozen constraint box `{7B-local, no gold-in-method, no OCR,
   single-dataset train, no cross-seed ensembles, no external APIs, D7 encoder-exclusion}`.

**Conclusion:** the diagnosis is **explanatory, not generative** — it converts the "biggest
unexplained fact" into a mechanism (modality-locus × error-type, equal-weight fusion), unifies
SAV/B5/B2 under it, and confirms the ≥2-dataset goal is unreachable inside the box because
MHC's ceiling is intrinsic (visual channel uninformative to the VLM + label-limited decision
core), not a method gap a novel MLLM integration could close. **Publishable value:** it upgrades
the paper's encoder-swap result from "works on HateMM, mysteriously not MHC" to a mechanistic,
evidence-backed account of *when a frozen-VLM encoder upgrade converts to accuracy* — a
transferable characterization (visually-grounded + representation-limited ⇒ converts; otherwise
ranking-only), and a fourth instance of the campaign's "better signal, no conversion" law.

---

## 8. Ranked mechanism hypotheses (by evidence strength)

1. **[LEAD, strong] Modality-locus × equal-weight fusion.** Qwen's uniform text gain converts
   only where the image stream is not a net drag AND the visual signal is decisive: HateMM
   (image neutral+strong) converts; MHC-EN (image collapse −0.135 AUC cancels text in 50/50
   concat) does not. Evidence: T2 per-modality AUC (large-sample), T1 dev decomposition, the
   CLIP-vs-Qwen image-informativeness gap, 32B image still collapsed.
2. **[LEAD, strong] Representation-limited vs label-limited errors (Pareto vs rotation).** HateMM
   gain = +0.116 hate-recall at zero non-hate cost; MHC-EN gain = a wash rotation. Evidence: T3
   per-class recall, T4 net-fix (+5 vs −1), and consistency with SAV (data/label-limited) and B5
   (unconvertible ranking edge). Hypotheses 1 and 2 are complementary, not competing: 1 is why the
   *fused representation* gain is HateMM-specific; 2 is why even the residual gain converts only
   there.
3. **[secondary] Empty-transcript handicap to CLIP on HateMM** (5.6% vs ~0%). Real but minor;
   the uniform, MHC-largest text gain rules it out as the driver.
4. **[REJECTED] Dilution** (MHC hate diluted across benign content) — falsified by SAV and not
   supported here (the purity gain is equal across datasets; the failure is fusion-cancellation +
   label-limited core, not signal quantity).

---

## 9. Provenance / reproduction

- Scripts (this analysis, committed): `scripts/analysis/encoder_swap_geometry.py`,
  `scripts/analysis/encoder_swap_diagnosis_tables.py` (+ `..._out.json`). conda `HateVideo`,
  CPU, `OMP_NUM_THREADS=4`; each runs in <2 s (kNN) / seconds (all tables).
- Banked inputs (read-only): `data/CLIP_Embedding/{HateMM,MHC,MHC_zh}/{train,dev_seen}_{openai_clip-vit-large-patch14-336_HF,Qwen2.5-VL-7B-Instruct_HF,Qwen2.5-VL-32B-Instruct_HF}.pt`
  (per-video pooled `img_feats`/`text_feats`/`labels`/`ids`). **No test caches touched.**
- Anchors: `research-wiki/experiments/exp-encoder-3seed.md` (`040adb8`), `SAV_F1_VERDICT_REVIEW.md`
  (dilution falsified), `B5_VERDICT_REVIEW.md` (`50f01b9`, unconvertible edge), B2 (scale),
  `directions_tried.json` (bans / axes), `TERMINUS_round3_mllm_plus3.md`.
- **Required statements:** no held-out **test** metric was read or produced; all acc/mF1/AUC are
  train/dev used solely to measure representation geometry (conditional/diagnostic). Gold read =
  train + dev `labels` only. No `state/`, prereg, config, or frozen artifact mutated. Not pushed.
