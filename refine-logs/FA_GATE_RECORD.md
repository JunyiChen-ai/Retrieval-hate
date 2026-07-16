# FA GATE RECORD — fusion/composition $0 probe (does the cancelled Qwen-text gain convert on MHC-EN?)

**Executor:** FA gate executor (ZERO GPU, ZERO test-touch, banked caches only; no Modal, no user).
**Date:** 2026-07-17. **Repo HEAD at design time:** `6032d32` (recorded in provenance §6).
**Raw-only record.** The executor applies the pre-declared mechanical kill/pass rules and attaches a
**NON-binding** label; the binding verdict is the orchestrator's.

Direction: wave-4 candidate **FA** (`WAVE4_CANDIDATES.md` §2, commit `6032d32`). The wave-4 recon's §0.3
correction showed the deployed head fuses via **`fusion_mode='align'` = a parameter-free element-wise
**Hadamard** product of two L2-normed projections** (`src/model/classifier.py:120`,
`x = torch.mul(imghat, texthat)`) — **not** the "equal-weight concat" F44's prose described. In align mode a
linear `img_proj` cannot map varying inputs to a constant (zero weights => `normalize(0)` NaN), so the head
**structurally cannot down-weight the collapsed Qwen image factor**; F44's dismissal of a modality-reweighting
fusion lever ("the head already has attenuation capacity and still failed") rests on a premise the align head
does not satisfy. That cell was therefore **unmeasured, not closed**. FA measures it: does recovering the
F44-cancelled Qwen-text gain via a **different modality composition** convert to *accuracy* (a Pareto move) or
only re-rank (a rotation, B5-dead)?

**Architecture claim independently verified.** `src/model/classifier.py:110-122`: both projections are
`nn.functional.normalize(., p=2, dim=1)` then, for `fusion_mode=='align'`, `torch.mul` (Hadamard). Confirmed.

---

## 0. LOCKED BEFORE READ (house rule: design fixed before any judged number)

The FA section of `WAVE4_CANDIDATES.md` §2.1(d) pre-declares the arms and the K-FA-1/2/3 bars. Parameters it
left underspecified were fixed **before** computing any judged number, as follows:

**Machinery = the F44 concat-kNN proxy, reused exactly.** Raw frozen per-video features, per-modality
L2-norm, cosine top-20 rank/sim-weighted signed kNN vote (memory=train, decision `score>0`) — the
`scripts/analysis/encoder_swap_geometry.py:knn_vote` semantics. This is the §0.3-validated proxy that
reproduces the deployed align (Hadamard) head's downstream dev **sign** (F44 §1). Because it uses **raw
features (no trained head, no training seed)** the probe is **deterministic**; bootstrap / permutation over
dev items give the CIs. (The router gate needed 3 seeds only because it reloaded *trained* heads; FA does not.)

**Arms** (all over the same proxy; MHC-EN primary, HateMM sanity/positive-control):
- **A0 CLIP-concat** = `[imghat_CLIP , texthat_CLIP]` (w=0.5) — the baseline reference for every delta.
- **A1 Qwen weighted-concat** = `z = [sqrt(w).imghat_Q , sqrt(1-w).texthat_Q]`, `w in {0.00,0.05,…,1.00}` (21).
  Because each block is L2-unit, `||z||=1` and `cos(z_a,z_b) = w.img_cos + (1-w).txt_cos` — a clean convex
  reweight of the two modality cosines. `w->0` = Qwen-text-only; `w->1` = Qwen-image-only. This is the reweight
  the align head structurally cannot perform.
- **A2 cross-encoder concat** = `z = [sqrt(w).imghat_CLIP , sqrt(1-w).texthat_Q]`, same w-grid (angle-(b):
  CLIP's strong image ⊕ Qwen's better text). Ids aligned CLIP<->Qwen by common id (orders already match).
- **A3a Qwen-align** = `imghat_Q (.) texthat_Q` (raw-feature Hadamard control; both Qwen streams 3584-d). NB a
  raw CLIP-Hadamard is undefined (1024 vs 768 dims) — the deployed align only Hadamards after learned 1024-d
  projections — so no raw CLIP-align arm exists; A3a is a control, not the F44 proxy.

**w-selection (locked):** two reads, mirroring the router gate's ceiling-then-deployable structure.
- *Ceiling / judged read:* over A1∪A2∪A3a, the config maximising **dev** acc (dev-oracle-w) — the
  maximally-favorable point, so a KILL here is airtight. Pareto feasibility (K-FA-1 point bars) is checked over
  the **whole grid**; if any config meets the point bars, the max-Δacc feasible one is the K-FA-1 candidate.
- *Deployable read (supplementary):* per arm, w that maximises **train-LOO** acc, evaluated on dev.

**Oracle-threshold (K-FA-2, locked):** B5 port — both the candidate and the CLIP-concat baseline get their
**own label-oracle decision threshold** (the τ on the vote score that maximises dev acc; dev labels touch the
threshold only, test never read). `d_oracle = candidate@oracle − CLIP-concat@oracle`. Giving *both* arms the
oracle cut is the fair B5 test; giving only the candidate a cut would inflate it. **Rule (verbatim from the
mandate): `d_oracle < +0.03` => the AUC edge is easy-example ordering, unconvertible => KILL.**

**K-FA-3 tolerance (locked):** the concat proxy's MHC-EN dev `(Qwen − CLIP)` acc must be **negative AND within
±0.010 of F44's −0.012**; else MACHINERY_INVALID. (Same machinery as F44 => near-exact reproduction expected.)

**Pre-declared bars.**
- **K-FA-1 (binding):** candidate must be Pareto — `Δhate-recall ≥ +0.03` AND `Δnon-hate-recall ≥ −0.01` AND
  `Δacc ≥ +0.02` AND **bootstrap CI-low > 0** (1000×, over dev items). A rotation (`Δacc ≤ 0` with the classic
  +hate/−non-hate trade) = KILL.
- **K-FA-2 (binding):** `d_oracle < +0.03` = KILL (B5 kill-switch, ported).
- **K-FA-3:** machinery validity (above).
- **selection-null (house "the usual"):** shuffle dev labels (1000×), recompute the arm's max-over-w Δacc; the
  observed max Δacc must exceed the null p95 (controls for selecting w on dev inflating the best-w Δacc).
- **HateMM sanity:** FA arms must not lose the HateMM Qwen-concat win; and the Pareto detector **must fire** on
  it (positive control / planted-signal recovery — a MHC zero is a calibrated zero only if the detector fires
  on a known real conversion).
- **PASS** = K-FA-1 ∧ K-FA-2 ∧ selection-null ∧ deployable(train-w Δacc ≥ +0.02) ∧ HateMM-sanity, all true.
  Only a PASS promotes to CC (§2.3 of the recon) + a D7-composition ruling.

**Determinism:** fixed RNG=20260717, no wall-clock seeding; `OMP_NUM_THREADS=4`; ~seconds CPU.

---

## 1. Machinery validation (K-FA-3) and calibration (planted-signal recovery)

**K-FA-3 — the concat proxy reproduces the align-head dev sign, bit-close.** MHC-EN concat proxy dev acc:
CLIP **0.7625**, Qwen **0.7500**, `Qwen − CLIP = −0.0125` vs F44's **−0.012** (`|Δ|=0.0005 < 0.010`, sign
negative). **MACHINERY VALID** — the proxy is the faithful §0.3-validated substrate F44 used.

**Calibration — the Pareto/rotation detector is live (not a dead switch).** Two synthetic planted sets on a
MHC-EN-like 25/55 balance:
- *planted pure Pareto* (fix 6 hate FN, touch no non-hate): `Δhate +0.24 / Δnon-hate +0.00 / Δacc +0.075` →
  **detector flags Pareto** ✔.
- *planted symmetric trade* (fix 6 hate FN, break 6 non-hate): `Δhate +0.24 / Δnon-hate −0.109 / Δacc 0.00` →
  **detector flags rotation** ✔.

**Positive control — the *real* HateMM conversion clears every bar the MHC candidate misses.** Same machinery,
HateMM Qwen-concat vs CLIP-concat: `Δacc +0.0467 / Δhate +0.1163 / Δnon-hate +0.000` (clean Pareto), and
**K-FA-2 `d_oracle = +0.0467 ≥ +0.03`** (converts at the oracle threshold). So the K-FA-2 kill-switch **fires on
MHC (+0.025) and does NOT fire on HateMM's genuine win (+0.0467)** — the MHC kill below is a **calibrated** kill.

---

## 2. RAW RESULTS — MHC-EN (PRIMARY), n_train=549, n_dev=80 (25 hate)

Baseline **A0 CLIP-concat**: dev acc **0.7625**, hate-recall **0.600**, non-hate-recall **0.836**,
oracle-threshold acc **0.800**.

### 2.1 A1 — within-Qwen reweight (the fusion the align head cannot do). Deltas vs A0.

| config | dev acc | Δacc | Δhate-rec | Δnon-hate-rec | dev AUC | d_oracle | reading |
|---|---|---|---|---|---|---|---|
| A1 w=0.00 (Qwen-text-only) | 0.7375 | −0.0250 | **+0.120** | −0.091 | **0.8575** | +0.000 | pure unconvertible edge (AUC↑, acc↓) |
| A1 w=0.15 (train-LOO best) | 0.7875 | +0.0250 | +0.120 | −0.018 | 0.8778 | +0.025 | within noise (see §3) |
| A1 w=0.50 (Qwen-concat) | 0.7500 | −0.0125 | +0.040 | −0.036 | 0.8473 | +0.000 | **F44 rotation, exact** |
| A1 w=1.00 (Qwen-image-only) | 0.7750 | +0.0125 | −0.280 | +0.146 | 0.6865 | −0.025 | collapsed image (F44's 0.599 regime) |
| A3a Qwen-align (Hadamard) | 0.7250 | −0.0375 | −0.280 | +0.073 | 0.7884 | −0.025 | deployed fusion; **worst** |

The within-Qwen reweight recovers the +0.054 text AUC (w→0: AUC 0.8575) but **no w converts**: it is either the
exact F44 rotation (w=0.5: +0.040 hate / −0.036 non-hate) or a pure ranking edge that accuracy does not absorb
(w→0: AUC 0.857, acc −0.025). The align (Hadamard) control is the *worst* accuracy of all — consistent with
§0.3 (a collapsed factor corrupts multiplicatively).

### 2.2 A2 — cross-encoder CLIP-image ⊕ Qwen-text (the angle-(b) / CC object). Deltas vs A0.

The AUC **peaks at w=0.15 = 0.8982 — the highest anywhere on MHC-EN** (CLIP-img 0.734 ⊕ Qwen-text 0.851 does
improve the ranking, exactly as F44 predicted). But the accuracy read is a **jumpy, non-monotonic** function of
w (adjacent near-identical compositions swing acc by ±0.04 = ±3 videos), the noise signature:

| w | dev acc | Δacc | Δhate-rec | Δnon-hate-rec | dev AUC | d_oracle |
|---|---|---|---|---|---|---|
| 0.05 | 0.8000 | +0.0375 | +0.240 | −0.055 | 0.8749 | +0.025 |
| 0.10 | 0.7875 | +0.0250 | +0.200 | −0.055 | 0.8895 | +0.013 |
| **0.15** | **0.8125** | **+0.0500** | **+0.120** | **+0.018** | **0.8982** | **+0.025** |
| 0.20 | 0.7750 | +0.0125 | +0.040 | +0.000 | 0.8727 | +0.025 |
| 0.50 | 0.7750 | +0.0125 | −0.120 | +0.073 | 0.8218 | +0.013 |

**Ceiling / K-FA-1 candidate = A2_cross w=0.15** (max dev acc): `Δacc +0.050 / Δhate +0.120 / Δnon-hate
+0.018` — a **Pareto-SHAPED point** that meets the K-FA-1 *point* bars. It is the *only* config on the entire
grid that does. But it survives **none** of the inferential guards (§3).

Deployable (train-LOO-selected w): A1 w=0.15 → dev Δacc **+0.025** (Δnon-hate −0.018); A2 w=0.10 → dev Δacc
**+0.025** (Δnon-hate −0.055). Both inside the noise floor and not Pareto (non-hate down).

### 2.3 Degenerate-supervision check (F47 warning, measured)

F47 flagged that the *trained* CLIP head memorises train (LOO 0.998), degenerating the routing target. **FA uses
raw-feature kNN (no trained head), so that pathology cannot recur, and does not:** raw train-LOO accs sit at a
healthy **0.72–0.81** across all arms (A2 w=0.10 = 0.8106, the deployable pick). The w-selection target is a
low-capacity scalar (1-of-21), non-degenerate. The train-selected w transfers to only +0.025 dev (within noise)
— consistent with **no memorization but also no real signal**, not with a degenerate target.

---

## 3. The three inferential guards on the K-FA-1 candidate (A2_cross w=0.15) — all fail

| guard | value | verdict |
|---|---|---|
| **K-FA-1 bootstrap CI-low** (1000×, Δacc over dev items) | Δacc +0.050, **CI [−0.0625, +0.150]** | CI-low ≤ 0 → **fails** |
| **K-FA-2 oracle-threshold** (both arms @ label-oracle τ) | candidate 0.8250 − baseline 0.8000 = **+0.025** | < +0.03 → **KILL fires** |
| **selection-null** (shuffle dev y, max-over-w Δacc, 1000×) | observed +0.050 vs **null p95 +0.1375**, null-mean +0.076, **p=0.766** | +0.05 sits at the 23rd pct of the *noise* floor → **not survived** |

The +0.050 point estimate is **4 videos on n=80**, with a bootstrap spread of ±5 videos, and is **smaller than
the median Δacc a blind best-of-21-w search returns on shuffled labels** (+0.076). The Pareto *shape* is real at
the point estimate; the Pareto *effect* is indistinguishable from selection noise.

---

## 4. MECHANICAL KILL/PASS (pre-declared rules applied verbatim; PRIMARY = MHC-EN)

| switch | rule | value | fires? |
|---|---|---|---|
| **K-FA-3** | concat proxy MHC-EN dev (Qwen−CLIP) reproduces F44 −0.012 (±0.010, sign−) | **−0.0125** | machinery **VALID** |
| calibration | Pareto & rotation detectors fire on planted signals | both **fire** | detector **live** |
| **K-FA-2** | `d_oracle < +0.03` ⇒ easy-example ordering ⇒ KILL | **+0.025** | **KILL fires** |
| **K-FA-1** | Pareto point bars AND boot CI-low > 0 | point bars met; **CI-low −0.0625** | **not passed** |
| selection-null | observed max Δacc > null p95 | +0.050 vs +0.1375 (p=0.766) | **not survived** |
| HateMM positive-control | Pareto detector fires on the known win; d_oracle ≥ +0.03 | fires; **+0.0467** | **OK (calibrates)** |
| HateMM sanity | FA arms not worse than Qwen-concat win | best HateMM arm (A1 w=0.30) **0.8598** ≥ Qwen-concat **0.8318** | **OK** |
| deployable | train-w Δacc ≥ +0.02 | +0.025 | met but non-Pareto & within noise |

## 5. NON-BINDING EXECUTOR LABEL: **KILL** (machinery valid; calibrated; B5 kill-switch fires)

The F44 modality-fusion cell is **measured and closed**. Neither route recovers a Pareto conversion on MHC-EN:

- **Within-Qwen reweight (A1)** — the fusion the align head structurally cannot do — is a pure **rotation /
  unconvertible ranking edge** at every w (F44-exact +0.040 hate / −0.036 non-hate at concat; AUC 0.857 with
  −0.025 acc at text-only; align-Hadamard the worst). Recovering the text AUC does **not** buy accuracy.
- **Cross-encoder CLIP-img ⊕ Qwen-text (A2 / CC)** raises the ranking to the **best AUC on MHC-EN (0.898)** — a
  genuine representation improvement — but the accuracy conversion is a **selection artifact**: the sole
  point-bar-passing config (w=0.15, Δacc +0.05) fails the bootstrap CI (crosses 0), fails the selection-null
  (p=0.766, below the noise median), and its **oracle-threshold edge is only +0.025 < +0.03** — the pre-declared
  **K-FA-2 B5 kill-switch fires**. This is calibrated: the identical test gives **+0.0467 (would-pass)** on
  HateMM's genuine Pareto win.

**Scientific reading (for the reviewer, non-binding).** FA converts F44's *asserted* fusion-dismissal (built on
the mis-described concat premise) into a *measurement*, and the measurement confirms the dismissal on the
correct (align) architecture: the cancelled Qwen-text gain is **not** convertible to MHC-EN accuracy by any
modality reweighting or by a CLIP-img⊕Qwen-text composition. The cross-encoder is the **fifth "better-signal /
no-conversion" datum** (after P3 · S2S F37 · W2-A F42 · router F47) — it lifts the *AUC* (the exact quantity B5
proved unconvertible) but not the *accuracy* at the label-oracle operating point. **No PASS ⇒ CC does not run,
no D7-composition ruling is owed.** The terminus is strengthened, not reopened; no headline/family claim
created. (One honest caveat recorded for the binding review: on n=80 the *point* estimate is Pareto-shaped, so
this is a "within-noise, does-not-survive-guards" close, not a sign-flipped rotation like A1's — the guards, not
the point sign, carry the kill. The K-FA-2 oracle switch is the decisive, calibrated pre-declared rule.)

> **Executor note (transparency):** the mechanical decision tree first emitted `AMBIGUOUS` because its KILL
> branch was coded only for the "no point-bar config" case, so a point-bar-passing-but-guard-failing config fell
> through. The pre-declared **K-FA-2 rule ("`d_oracle < +0.03` ⇒ KILL", verbatim from the mandate)** was already
> computed (`K_FA_2_pass=False`); the label wiring was corrected to route that pre-declared kill-switch to the
> final label. No threshold was changed; this implements the locked rule, it does not move it. Both the raw
> per-switch booleans and the final label are in `FA_GATE_OUT.json`.

---

## 6. Provenance / hygiene

- **Script (committed):** `scripts/analysis/fa_fusion_gate.py`, sha256
  `9e2fcbf39966cf85f6f5184eb29cf31cd6c577db1cfa5717ee70739a010b8b04` (also in `FA_GATE_OUT.json:script_sha256`).
  Deterministic (RNG=20260717); CPU-only (`OMP_NUM_THREADS=4`); ~seconds. kNN vote machinery reused faithfully
  from `scripts/analysis/encoder_swap_geometry.py` (cosine top-20 rank/sim-weighted signed vote, `score>0`); the
  §0.3 align-Hadamard claim re-verified against `src/model/classifier.py:110-122`.
- **Outputs:** `refine-logs/FA_GATE_OUT.json` (full grid, all deltas, guards, calibration, per-switch booleans).
- **Banked inputs (read-only):** `data/CLIP_Embedding/{MHC,HateMM}/{train,dev_seen}_{openai_clip-vit-large-
  patch14-336_HF, Qwen2.5-VL-7B-Instruct_HF}.pt` (per-video pooled `img_feats`/`text_feats`/`labels`/`ids`).
- **ZERO GPU, ZERO test-touch** (train + dev features/labels only; test caches never opened), **no Modal**, no
  download, no user interaction. Gold read = train + dev `labels` only (train-LOO w-selection; dev scoring +
  label-oracle threshold). No `state/`, prereg, config, `research-wiki/`, LoRA-HateMM ceremony artifact, SLURM
  job, or frozen artifact mutated. Not pushed.
- **Repo HEAD at run:** `6032d32`; recorded via the commit that lands this record.
