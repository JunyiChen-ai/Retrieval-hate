# EXHAUSTION AUDIT — adversarial completeness review of the round-2 MLLM-integration search

**Auditor:** independent adversarial completeness auditor (zero prior context by design; read-only
except this file; ZERO GPU/SLURM/commits). **Date:** 2026-07-14.
**Mandate:** try to REFUTE "the search space for the goal is exhausted within constraints." Win by
finding a genuinely open cell OR by confirming exhaustion with independent reasoning. No rubber-stamp.

**Goal under audit:** integrate Qwen2.5-VL-7B meaningfully+novelly into the retrieval-contrastive
hateful-video pipeline (frozen encoder → alignment head, triplet+BCE → top-20 kNN vote); bar = ≥+3 acc
AND ≥+3 macro-F1, 3/3 seeds, both protocols, per dataset; binding gap = a SECOND passing dataset
(HateMM already +5.3 frozen). Datasets HateMM (744 tr) / MHC-EN (549) / MHC-ZH (579).

---

## VERDICT — EXHAUSTION REFUTED (narrowly, and on a cheap decisive probe the team skipped)

The 22-route closure is, for the routes it actually measured, honest, well-instrumented, and mostly
airtight. **But it is not complete.** One cell is genuinely open, sits exactly on the binding gap (a
second dataset), is mechanistically grounded in the team's OWN logged numbers, is non-isomorphic to
all 22 dead routes and the banned list, and — decisively — was **never measured**: the closure that
supposedly covers it (B1, the 20th negative) measured only a *sub-cell*.

**The open cell: convert the frozen-Qwen ranking advantage on MHC-ZH into accuracy by calibrating the
decision rule the pipeline currently hard-codes.**

The exhaustion claim rests on B1 disposing of "frozen-Qwen encoder on MHC-ZH" as a clean FAIL. B1 is
correct *about what it measured* — the encoder swap under the **unchanged, uncalibrated decision rule**.
It never measured the quantity that decides whether the representation advantage is real-but-blocked
vs absent: **the oracle-threshold / calibrated-threshold accuracy of the frozen-Qwen ZH representation.**
That number is cheap (≈1 min GPU, or CPU-replay from cached features) and has never been computed.
Until it is, "exhausted within constraints" is *unproven* for this cell — and the team's own logged
data point toward convertible headroom.

I rank this the top open cell, give it a MODEST (not high) prior — the realistic outcome is a B3-style
*marginal* second-dataset pass, not a clean +3 — and specify its G0-cond probe in §7. Three weaker open
cells survive at low prior (§5). Everything else is genuinely closed (§6).

---

## 1. THE OPEN CELL — ROC→acc conversion of frozen-Qwen on MHC-ZH

### 1.1 The load-bearing anomaly (all numbers re-derived from the banked verdicts)

On MHC-ZH the frozen-Qwen encoder ranks **far** better than CLIP but scores **worse** on thresholded
accuracy — a huge, 3/3-seed-consistent, unconverted AUC advantage:

| ZH final-epoch (job 13115) | acc | roc/AUC | **roc − acc (calibration gap)** |
|---|---|---|---|
| frozen-CLIP mean | 0.8143 | ~0.839 | **0.025 (near-optimal operating point)** |
| **frozen-Qwen mean** | **0.8031** | **~0.888** | **0.085 (badly mis-placed operating point)** |
| Δ (Qwen − CLIP) | **−0.0112** | **+0.049** | — |

Provenance: `refine-logs/B1_VERDICT_REVIEW.md:35-40` (per-seed Qwen roc 0.8906/0.8951/0.8806, acc
0.8188/0.8054/0.7852; CLIP roc 0.8382/0.8342/0.8444, acc 0.8054/0.8054/0.8322); the reviewer's own
note `B1_VERDICT_REVIEW.md:160-163`: *"the Qwen arm does carry consistently higher Test roc/AUC —
0.88–0.90 vs CLIP 0.83–0.84 on every seed — i.e. better ranking that never converts to better
thresholded acc/F1; the rule is on acc AND F1, so this does not move the verdict."*

**Read this carefully.** The team saw the anomaly, correctly noted it was verdict-neutral under the
pre-registered rule, and moved on. **They never asked whether the operating point is fixable.** The
Qwen ZH roc−acc gap is **0.085 — 3.4× CLIP's 0.025.** Frozen-Qwen's representation ranks ZH videos
excellently (AUC 0.888) and the decision rule throws that away; CLIP's decision rule is already near
its own ceiling. That asymmetry is the signature of a *convertible* representation advantage blocked
at the decision layer, not an absent one.

### 1.2 The code fact — the pipeline hard-codes an uncalibrated, class-balance-blind threshold

The per-epoch Val/Test lines call `compute_metrics_retrieval(..., use_sim=True)`
(`src/run_rac.py:673-674,691-692`). Inside (`src/utils/metrics.py:294-301`):

```python
roc = roc_auc_score(labels, list_majority_voted)          # threshold-FREE ranking (AUC)
...
list_majority_voted_round = (sigmoid(np.array(list_majority_voted)) >= 0.5)*1   # FIXED cut at sim-vote ≥ 0
acc = np.mean(list_majority_voted_round == labels)
```

So **accuracy thresholds the similarity-weighted signed kNN vote at a fixed 0** (sigmoid≥0.5 ⇔ vote≥0),
and macro-F1 (`:307-309`) is computed at the same fixed cut. **The threshold is never calibrated on
validation** — model selection (`run_rac.py:757,775`) only picks the epoch with max val-acc *at that
fixed cut*; it never moves the cut.

Now the kicker: **MHC-ZH is 30% positive** (test 45/149 = 0.302; train 180/579 = 0.311; dev 28/78 =
0.359 — measured this audit via `torch.load` on `data/CLIP_Embedding/MHC_zh/*_Qwen2.5-VL-7B-Instruct_HF.pt`).
A symmetric, class-balance-blind threshold on a 30/70 dataset is *a priori* suboptimal for both
accuracy and (especially) macro-F1. The encoder with the largest unconverted ranking advantage sits on
top of the single most threshold-suboptimal setting in the project. **No route ever moved this
threshold on frozen-Qwen ZH features.**

### 1.3 Why it escapes D1 / D2 / D3 and every epitaph

- **D1 (redundancy law) does not bite.** D1 kills *adding a low-bandwidth decision-side MLLM signal*
  that is conditionally redundant given the frozen representation. This cell adds **no signal at all**
  — no MLLM scalar, no extra channel. It re-places the operating point of an *already-better*
  representation using only class labels on val (labels are explicitly allowed). There is no
  "conditional information beyond Z" question because nothing is concatenated to Z; Z itself is the
  Qwen representation whose ranking is measurably superior (AUC +0.049, 3/3 seeds).
- **D2 (representation law) actively PREDICTS this.** D2: "only representation-level levers ever cleared
  +3; increases come from representation, never decoration." Frozen-Qwen-ZH **is** a representation
  with a large ranking advantage. It failed to clear +3 *only because the decision rule discards the
  advantage.* Converting it is not "decoration" (it adds no signal) and not a new representation — it
  is unlocking the representation lever D2 says is the only thing that works. D2 is the reason to run
  this, not to skip it.
- **D3 (measurement law) is the real risk, and it is bounded here.** D3: 78-dev selection is noisy
  (±~2 acc pts). A **single scalar threshold** is far less overfit-prone than epoch- or feature-subset
  selection, and the AUC advantage (+0.049) is well above the noise floor. D3 is why the *honest*
  (val-calibrated) number may land marginal rather than clean — not a reason the cell is closed. The
  G0-cond probe (§7) puts an **oracle-threshold arm** as the kill-switch so D3 is quantified, not
  assumed.

Epitaph check — the cell clears every one of the 22:
- **B1 (20th, `directions_tried.json:86-89`)** "frozen Qwen encoder on MHC-ZH: FAIL both protocols" —
  measured the encoder swap under the **unchanged fixed-threshold decision rule** only. The sub-cell
  "frozen-Qwen-ZH representation + calibrated decision rule" is untouched (see §3, epitaph over-reach).
- **B3 (`exp-lora-zh-b3.md`)** converts ZH via **LoRA** (re-training the encoder). This cell converts
  via the **decision rule on the frozen features** — a different, cheaper, LoRA-free lever, and one
  that sidesteps the "LoRA = RA-HMD-family performance lever, not novelty" pending-ruling that clouds
  B3.
- **P1 (`CAMPAIGN_mllm_method_role.md:51`)** did adjust "the drift-gated vote threshold" — but via an
  **MLLM-generated HARMFUL/BENIGN prior** (a decision-side MLLM signal, bandwidth = bits) and to correct
  temporal drift; it died because the MLLM verdict's FPR drifts across the temporal boundary. This cell
  uses **no MLLM signal** and **val labels only**; injection point = decision threshold, signal source =
  labels + the encoder's own ranking. Non-isomorphic (different signal, different bandwidth class,
  different failure mode).
- **encoder-swap (`exp-encoder-3seed.md`)** injection point = the encoder feature. This cell's injection
  point = the decision rule. Different injection point ⇒ non-isomorphic by the diagnosis frame's own
  test.
- **SAV (18th), P9/P9b, P10, TARC, A-line** — all touch representation-mining or decision-side signals,
  none touches threshold/operating-point calibration of the frozen-Qwen ZH vote.

### 1.4 Injection point, bandwidth, non-isomorphism, cost, second-dataset assessment

- **Injection point:** the decision rule (vote threshold; optionally a class-balanced/operating-point-
  aware training loss on the same head). **Not** the encoder, **not** an added channel.
- **Bandwidth class:** zero added signal bandwidth (a single calibrated scalar from val labels). This
  is a *new* bandwidth class relative to the 22 dead routes, which are all either representation-swap
  (encoder) or low-bandwidth-decision-side-ADD.
- **Could it plausibly clear +3 on the SECOND dataset?** This IS the second dataset (ZH). Honest
  answer: **plausible but likely MARGINAL.** The roc−acc gap of 0.085 shows real blocked headroom, and
  30/70 imbalance guarantees the fixed symmetric cut is suboptimal (helps macro-F1 in particular). But
  an AUC 0.888 does not guarantee a large best-threshold accuracy — under an equal-variance score model
  it could already be near-optimal; the kNN sim-vote is bounded/non-Gaussian, so the true convertible
  headroom is **empirically undetermined and must be probed.** Realistic band: paired Δacc after
  honest val-calibration ≈ +0.02 to +0.05 on ZH — i.e. a B3-class *marginal* pass or near-miss, not a
  clean +3. **That is still a refutation of "exhausted":** a binding-gap cell with a live, cheap,
  never-computed path to a second passing dataset is not an exhausted search.
- **GPU cost:** ~1 min (re-run the 13115 heads on cached features à la B3, instrumented to dump
  per-video votes), or **zero GPU** by retraining the tiny 1024-d head on CPU / replaying the vote.

### 1.5 The novelty caveat (honest; a user ruling, not mine to make)

Threshold calibration by itself is trivial and not "novel MLLM integration." The defensible framing:
*MLLM encoders produce better-ranked but operating-point-miscalibrated representations for a
retrieval-vote head; a calibration-aware decision rule is what lets the MLLM-as-encoder role clear the
≥2-dataset bar (HateMM + ZH).* That is a small, real methodological finding, and it targets the goal's
**performance clause** (the binding gap) directly. Whether it satisfies the **novelty clause** is a
user ruling — exactly the same class of ruling already pending for B3. The auditor's job is the
performance clause and the open-cell question; on those, the cell is open.

---

## 2. EPITAPH-STRENGTH AUDIT — one genuine over-reach found

**B1 (20th negative) over-generalizes from a sub-cell to the whole cell.** `directions_tried.json:86-89`
and `TERMINUS §2 (D2)` read B1 as "frozen-Qwen ZH FAIL ⇒ ZH is a decision-layer-non-conversion, scale
is not the lever, ZH gain is a LoRA lever." The measured fact is narrower: *frozen-Qwen ZH fails **under
the fixed-threshold decision rule**.* The verdict doc itself flags the unconverted AUC advantage
(`B1_VERDICT_REVIEW.md:160-163`) and then discards it as rule-neutral. The leap from "fails under the
current decision rule" to "ZH representation carries no convertible advantage" is **not supported by any
measurement** — the oracle/calibrated-threshold accuracy was never computed. This is the classic
whole-cell-vs-sub-cell over-closure the mandate asked me to hunt: B1 closed "frozen-Qwen-ZH + unchanged
head," and the state files bank it as closing "frozen-Qwen-ZH." The delta between those two is precisely
the open cell in §1.

Other epitaphs checked and found **fairly scoped** (no over-reach):
- **encoder-swap positive** is scoped HateMM-only (`exp-encoder-3seed.md:18-21,235-239`) — not
  over-claimed to ≥2 datasets. Clean.
- **SAV (18th)** kill is decisive for attention-head-space mining: the corrected-machinery U-1 arm (full
  784-head concat spanning all layers) recovers to ≈pooled and carries no label info beyond last-layer-
  pooled on MHC and HateMM (`SAV_F1_VERDICT_REVIEW.md:170-180`), and SAV's native read-out is
  sub-majority on MHC (`:203-205`). The "MHC-EN is data/label-limited at this frozen read-out capacity"
  conclusion (`:209-211`) is correctly scoped to *read-out capacity* — it does NOT claim to close a
  *decision-rule* lever (which SAV never tested). Fair.
- **B4 (22nd, EN LoRA)** is explicitly scoped as "formalizing a banked seed0 negative, not opening new
  ground" (`B4_FORENSIC_RECON.md:146-150`). Honest, no over-reach.
- **C1/C3-target/C3-nontarget/A-line** epitaphs match their measured sub-cells. Fair.

---

## 3. ISOMORPHISM-ABUSE AUDIT — one minor gap, one correctly-closed

**(a) Does "MLLM-scores-as-training-signal" wrongly absorb representation-geometry distillation?**
Partially yes, but the team did NOT actually over-absorb it: the reflection §3.1/§3.3 and literature C4
explicitly keep **representation-level contrastive distillation (CRD)** alive
(`LITERATURE_mllm_integration_2026-07-13.md:27-30`). C4 was **deferred only for lack of a 72B teacher**,
not killed. A **7B-teacher** CRD (distill Qwen-7B embedding *geometry* — pairwise relational structure —
into the small head, representation-level, no scores, no pseudo-labels) is therefore **technically an
open cell not covered by any epitaph or ban.** It is low-prior (§5) but it is not closed, and the state
files do not list it. Minor flag: the banned-list phrasing "MLLM-scores-as-training-signal" could be
*read* to absorb it; it should not.

**(b) Does the pseudo-label pool-expansion ban absorb label-free self-supervised representation
expansion?** No over-absorption: the ban text (`directions_tried.json:105`) already carves out
"representation-training expansion only." That carve-out is then correctly **closed by the training-data
veto** (`:110`: single-dataset own-train-split only, "conservatively also bans external unlabeled-pool
training (C5)"). So label-free representation expansion is closed — but by the *data* veto, not by the
pseudo-label ban. Correctly closed; no abuse.

---

## 4. UNEXPLORED-AXES SWEEP (mandate angle 3) — dispositions

| axis | tested? | disposition |
|---|---|---|
| **Threshold/operating-point conversion of frozen-Qwen ROC (ZH)** | **NO** | **OPEN — top cell (§1).** Fixed cut, never calibrated (`metrics.py:298-300`); 30/70 imbalance; Qwen roc−acc gap 0.085. |
| Mid-layer / penultimate **pooled** hidden states (residual stream) | NO (SAV tested attention-HEAD outputs, not residual-stream pooled) | Near-closed by SAV U-1 null (all-layer head-space ≈ last-layer-pooled, `SAV_F1:170-180`); residual-stream mid-layer specifically never extracted. LOW prior — still faces the same conversion wall on the 2nd dataset. |
| Token-level (vs pooled) representations | NO | Untested but out of the frozen-cache design (cache is pooled 3584-d); would need re-extraction; and the U-1 aggregate-null argues against extra label info. LOW prior. |
| Mixed-encoder retrieval (Qwen keys + CLIP head, or vice versa) | NO | Untested combinatorial cell; no mechanism to reach +3 on a 2nd dataset (Qwen≈CLIP at the threshold on EN/ZH). LOW prior (§5). |
| Qwen⊕CLIP **embedding** concat (representation fusion) | NO (C3-nontarget fused MLLM *reasoning text*, a different object) | Untested; classic representation lever, non-isomorphic to C3-nontarget. LOW-MODERATE prior but inherits the ZH conversion wall (§5). |
| MLLM-embedding-geometry distillation into the head (7B CRD) | NO (C4 deferred for 72B only) | Open per §3(a). LOW prior — can't beat directly using the 7B encoder, which already fails to convert. |
| Loss/head change targeting ROC→acc conversion | NO | Same open cell as §1 via the training-time route (class-balanced / operating-point-aware loss). Part of the top cell. |

---

## 5. RANKED OPEN CELLS (prior × goal-relevance / cost)

1. **[LEAD] Decision-rule conversion of frozen-Qwen ZH ranking** — §1. Prior MODEST (marginal pass
   plausible), goal-relevance MAXIMAL (binding gap = 2nd dataset), cost ~1 min GPU / zero-GPU. Escapes
   D1, invited by D2, D3-bounded. Non-isomorphic to all 22 + bans.
2. **EN version of the same conversion** — frozen-Qwen EN roc +0.02 over CLIP (`exp-encoder-3seed.md:165-170`)
   with both encoders mis-calibrated; smaller advantage + 549-sample data limit (SAV H0). Prior LOW.
   Fold into the same probe (marginal cost) but do not bet on it.
3. **Qwen⊕CLIP embedding concatenation into the head** — untested representation fusion, non-isomorphic
   to C3-nontarget (which fused MLLM text, killed as encoder-redundant, `C3_NONTARGET_VERDICT_REVIEW.md`).
   Prior LOW-MODERATE: on HateMM concat≈Qwen (no new dataset); on ZH/EN both individually fail the fixed
   threshold, so concat still needs §1's conversion to matter. Cost ~2 min GPU. Test only if §1 clears
   its oracle gate.
4. **7B-teacher CRD geometry distillation into the head** — open per §3(a); representation-level, not
   scores. Prior LOW: cannot exceed using the 7B encoder directly, which fails to convert on EN/ZH.
   Cost ~1–2 h (distillation training). Lowest priority.

---

## 6. IF YOU READ THIS AS "MOSTLY EXHAUSTED" — the strongest confirmed-closed near-misses and why they die

- **72B-AWQ encoder (scale axis)** — dies pre-GPU on B2's monotone regression: HateMM 32B sits *between*
  CLIP and 7B (`directions_tried.json:91-94`); scale REGRESSES, so 72B prior ≈ 0. Correctly closed.
- **SAV attention-head mining** — dies on corrected-machinery re-run: MHC "+0.0875" is a crushed-baseline
  + Fano-floor artifact, HateMM harm is real and grows when the baseline is un-crushed, U-1 resolves to a
  clean null (`SAV_F1_VERDICT_REVIEW.md:96-180`). Decisively closed.
- **LoRA-Qwen EN encoder (B4)** — seed0 already banked negative (val-sel −0.031 acc), LoRA below *both*
  frozen encoders on EN, mechanism understood (549-sample SFT degrades the encoder)
  (`B4_FORENSIC_RECON.md:56-63,154-178`). Closed.
- **All decision-side MLLM signals (P1–P5, P7, P10, P11, TARC)** — D1 redundancy, five independent hits
  (`REFLECTION §2`, `CAMPAIGN §1`). The conversion cell in §1 is the one route that is neither a
  decision-side signal add nor an encoder/scale swap, which is exactly why it survives.

---

## 7. G0-cond PROBE DESIGN for the lead cell (near-zero GPU, oracle kill-switch built in)

**Object:** does calibrating the decision threshold convert frozen-Qwen's ZH ranking advantage into a
paired ≥+3 acc AND ≥+3 macro-F1 vs frozen-CLIP, 3/3 seeds, both protocols?

**Setup (reuse existing artifacts; no new training of the encoder):**
- Reproduce the frozen-Qwen and frozen-CLIP ZH heads for seeds 0/1/2 exactly as B1's job 13115
  (`enc3seed`/`train_archive_baseline` archive-OFF, cached features
  `data/CLIP_Embedding/MHC_zh/*_{Qwen2.5-VL-7B-Instruct,openai_clip-...-336}_HF.pt`). Instrument
  `compute_metrics_retrieval` to **dump the per-video signed sim-vote** for dev and test, both encoders,
  both protocols (val-sel epoch + final epoch). Cost: ~1 min GPU (à la B3) or CPU head-retrain.
- G-repro hard gate: the dumped fixed-threshold acc/F1 must reproduce B1's banked 13115 readings to 4 dp
  (`B1_VERDICT_REVIEW.md:29-40`), else HALT.

**Three threshold arms per (encoder, seed, protocol):**
1. **deployed** — fixed cut (sim-vote ≥ 0). Must equal the banked number (sanity).
2. **oracle** — cut maximizing **test** acc (and separately macro-F1). UPPER BOUND, label-oracle arm,
   **G0-cond kill-switch**: if paired oracle Δ (Qwen − CLIP) < +0.03 on ZH for acc OR macro-F1, the
   ranking advantage is *not convertible even with a perfect threshold* ⇒ **KILL the whole conversion
   route, zero further GPU.** (This is the mandated label-oracle calibration arm: gold labels used only
   to compute the upper bound, never in-method.)
3. **honest** — cut chosen on **dev** (78 samples) to max dev-acc / dev-macroF1, applied to test. The
   deployable number; the gap oracle→honest quantifies the D3 selection tax on a *single scalar*.

**Decision rule (pre-registered, same bar as B1/B3):** proceed to a formal single-submit ceremony only
if (i) oracle arm clears +0.03 on both metrics (headroom exists) AND (ii) honest arm paired Δacc ≥ +0.030
AND ΔmF1 ≥ +0.030, sign 3/3, **both protocols**. Report per-protocol verbatim ("final-epoch: …;
val-selected: …") with the B3-style marginality sensitivity note. **Mandatory paired control:** report
Δ = (Qwen_honest − CLIP_honest), not Qwen alone — the claim requires the calibration to help Qwen *more*
than CLIP (predicted by the 0.085-vs-0.025 roc−acc asymmetry; if it helps both equally the cell dies).
**Class-imbalance variant (optional, same run):** also evaluate a threshold chosen to max dev **balanced
acc / macro-F1** given the 30/70 skew, since macro-F1 is the more threshold-sensitive goal metric here.

**Interpretation ladder:** oracle < +0.03 → route dead (representation advantage genuinely
non-convertible; exhaustion confirmed for this cell). Oracle ≥ +0.03 but honest < +0.03 → dies on D3
(threshold-selection noise on 78-dev); banked as a *quantified* near-miss with the oracle headroom
recorded. Honest ≥ +0.03 both protocols 3/3 → **first frozen-encoder second dataset**, pending user
novelty ruling. All three outcomes are decision-useful; none has been produced.

---

## 8. Provenance
- Frozen-Qwen/CLIP ZH roc>acc anomaly: `refine-logs/B1_VERDICT_REVIEW.md:29-40,160-163`.
- Fixed-threshold code: `src/utils/metrics.py:294-301`; per-epoch call `src/run_rac.py:673-674,691-692`;
  model selection `src/run_rac.py:757,775`.
- ZH class balance (this audit, `torch.load`): test 45/149 pos (0.302), train 180/579 (0.311),
  dev 28/78 (0.359) from `data/CLIP_Embedding/MHC_zh/*_Qwen2.5-VL-7B-Instruct_HF.pt`.
- Encoder-swap positive (HateMM-only): `research-wiki/experiments/exp-encoder-3seed.md:18-21,150-239`.
- B3 LoRA-ZH marginal pass: `refine-logs/B3_VERDICT_REVIEW.md:20-24,110-147`.
- SAV corrected-machinery kill + U-1 null: `refine-logs/SAV_F1_VERDICT_REVIEW.md:96-180,203-211`.
- B4 EN-LoRA banked negative: `refine-logs/B4_FORENSIC_RECON.md:56-63,146-178`.
- 22-route dead list + bans + diagnosis frame: `autoresearch/goal_mllm_plus3/state/directions_tried.json`.
- D1/D2/D3 laws: `research-wiki/REFLECTION_mllm_integration_failures.md:23-29`; TERMINUS §2
  `research-wiki/TERMINUS_round2_mllm_plus3.md:19-24`.
- C4 (CRD) deferred-not-killed: `research-wiki/LITERATURE_mllm_integration_2026-07-13.md:27-30`.
- P1 threshold-via-MLLM-prior (non-isomorphic): `research-wiki/CAMPAIGN_mllm_method_role.md:51`.

**Scope statement:** zero GPU/SLURM/commits. Read-only except this file. One CPU `torch.load` of cached
frozen features to read ZH split sizes/labels (labels read for class-balance context only, never
in-method, no test-time modeling). No prereg/config/CLAUDE.md/state mutated.
