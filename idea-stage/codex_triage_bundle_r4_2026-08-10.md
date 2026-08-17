# Round-4 triage bundle — objective feasibility facts, then your adversarial pass

You produced 14 candidates in `idea-stage/section9_round4_candidates_2026-08-10.md` (re-read it).
The executor eliminated **nothing** on quality grounds. What follows is the **objective feasibility
gate**: facts about what is actually on this machine's disk, verified today. These are resource
facts, not opinions. Several of them materially change your ranking, which is why you are getting a
second pass rather than a rubber stamp.

---

## §A — Feasibility gate results (verified on disk 2026-08-10)

### A.1 F1 MDL — **FULLY FEASIBLE, and it is the only top-ranked candidate that runs on all four datasets**

Encoder caches present (`data/CLIP_Embedding/<ds>/<split>_<enc>.pt`, keys `ids/img_feats/text_feats/labels`):

| dataset | CLIP ViT-L/336 | frozen Qwen2.5-VL-7B | LoRA-Qwen |
|---|---|---|---|
| HateMM | yes | yes | yes (`-LoRA-curric`) |
| MHC-EN | yes | yes | yes (`-LoRA`) |
| MHC-ZH | yes | yes | yes (`-LoRA-curric`) |
| ImpliHateVid | yes | yes | **NO LoRA CACHE — 2 encoders only** |

Cost: a bare-head training run is ~30–60 s on the single RTX 5090. Your MVE (3 encoders × 5 folds ×
3 seeds × 4 datasets) is ~180 runs ≈ 1.5–3 GPU-hours — inside budget but at the top of it. Fold count
is the obvious dial.

### A.2 T1 PRES — **FEASIBLE ON EXACTLY ONE DATASET, and it is the contaminated one**

- OCR window assets exist for **HateMM only** at split granularity: `ocr_windows_K30.jsonl`
  (25,530 window rows over the 851 train+val videos), `ocr_windows_K30_test.jsonl` (6,450 rows over
  215 test videos), and pre-computed CLIP-text window vectors `pilot_ocr_window_vecs.npz`
  (**6,565 × 768** non-empty train+val windows) and `test_ocr_window_vecs.npz` (**2,111 × 768** test
  windows). Video-level means also cached (744/107/215 × 768).
- **HateClipSeg cannot supply a second dataset.** It has `ocr_windows_K30.jsonl` (11,850 rows) but
  **no train/test split at all** — all 395 items are a single partition (`data/gt/HateClipSeg/test.jsonl`,
  395 rows) — and its transcript channel is already known to be **395/395 constant**. There is no
  train pool to estimate a background from and no held-out pool to adapt to.
- MHC-EN / MHC-ZH / ImpliHateVid have **no OCR cache of any kind**.
- ⇒ PRES's decisive comparison (test-pool background vs train-only background) is **structurally
  single-dataset**, and that dataset is HateMM, whose official test split carries 12.1 % vs 5.2 %
  whitespace-only transcripts (OR 2.49, p = 0.001) and 3.3 % content-duplicate test items.

### A.3 I1 IPPO — **FULLY FEASIBLE, but structurally single-dataset by construction**

ImpliHateVid subtype labels are real and verified: train 325 EX / 324 IM / 634 NH; val 83/77/165;
test 92/108/201 — matching the ACL 2025 paper exactly. **No other dataset in the project has a
hate-subtype annotation**, so IPPO can only ever produce a one-dataset result.

**A leakage hazard you must design around**: the subtype is carried *in the item id string*
(`EX_`/`IM_`/`NH_`), and `NH_` is exactly the negative class. Any use of the id at inference is
label leakage, not subtype supervision. The audit already flagged that HateMM and ImpliHateVid ids
encode the label and that ids must never reach a model, a hash bucket, or a sort key.

### A.4 R1 B-SRTD — **NOT PILOTABLE THIS ROUND; the asset build is a separate expenditure**

Re-verified today: `data/Counterfactual/MHC/train_twins.jsonl` = **168 records, every one label=1**;
`MHC_zh` = **180 records, every one label=1**. Schema is
`{id, label, orig_text, sanitized_text, orig_verdict, san_verdict, regen_used, flipped}` — one
intervention axis (toxicity-sanitising transcript rewrite), positives only. Your MVE's precondition
(≥200 train + 80 val balanced 2×2 lattices over both hard labels and two intervention axes)
**requires generating an asset that does not exist**. That is hours of Claude-API work plus human
verification — a real and bounded expenditure, but it is not a pilot and cannot report inside this
round.

### A.5 The rest

- **B1 JLR, I2 SHC, I3 CNV, B3 NTC, B2 PCD**: all feasible on cached features, CPU/GPU-minutes.
  B2 additionally needs 12 policy-clause sentences encoded with the matching frozen text encoder —
  trivial.
- **T2 TMN**: needs MultiHateClip title/description text embedded. The raw metadata exists in the
  MHC release; no embedding cache exists yet. Small build, not zero.
- **T3 JRSA, F2 SCRA**: weeks of theory work by your own estimate; out of scope for a pilot round.
- **R2 EAPD**: needs 330 Claude-annotated videos against a pre-registered edge schema. Same class of
  expenditure as A.4, larger.
- **R3 SRCP**: K=30 CLIP subclip caches exist for **HateMM only**; MHC/MHC_zh have **K=4 only**.
  Its cross-dataset arm is not buildable.

---

## §B — What this round can actually run

Budget: one RTX 5090, ~8 GPU-hours total for all pilots, **at most 3 pilots**. Everything must be a
single submission under rules frozen before implementation, ≥3 seeds, train/val/test protocol,
all cells reported.

So the real choice set for *piloting now* is: **F1 MDL, T1 PRES, I1 IPPO, B1 JLR, I2 SHC, I3 CNV,
B3 NTC, B2 PCD** — and nothing else.

---

## §C — Your task: the adversarial pass, and the final pilot selection

For each of the eight runnable candidates:

1. **What is the strongest objection a reviewer would raise?** You already gave these for the top 3;
   give them for the rest, and say whether the objection is fatal or differentiable.
2. **What is the most likely failure mode** — i.e. what will the pilot actually show, and would that
   be informative either way?
3. **Does the single-dataset constraint kill it?** Be specific. A one-dataset gain has killed
   candidates in this project before, and the project's own §5.5 lesson is that signals which look
   strong against a weak comparator die against a trained one. State for T1 and I1 whether a
   single-dataset result can support a top-venue methods claim at all, or whether they are
   *diagnostics wearing a method's clothes*.
4. **Re-rank all eight** given the feasibility facts, especially A.2 and A.3.
5. **Name the 2–3 you would actually pilot now**, in run order, and for each state the **frozen
   decision rule** you want applied — the exact quantity, the exact bar, and the exact null — such
   that the rule can be written down before a single line of implementation exists and cannot be
   argued with afterwards. Include what a KILL looks like, not only a GO.

Two constraints on your answer:

- **Confirmatory-by-construction disclosure.** F1 MDL and I1 IPPO were both motivated by
  measurements already taken on the test sets (§4.1 and §4.3 of the generation bundle). Say plainly
  whether you think that is disqualifying, survivable-with-disclosure, or requires a specific design
  change to fix.
- **Do not pad.** If your honest read after the feasibility facts is that fewer than three of these
  deserve a pilot, say so and say why. A well-argued "run two, the third is not worth the GPU" is
  more useful than three.
