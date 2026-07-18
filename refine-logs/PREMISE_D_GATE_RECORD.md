# PREMISE-(d) GATE RECORD — does CLIP-img ⊕ LoRA-EN-Qwen-text convert on MHC-EN?

**Executor:** premise-(d) gate executor (ZERO GPU, ZERO Modal, ZERO test-touch, banked caches only,
no user interaction; CPU/numpy only). **Date:** 2026-07-18. **Repo HEAD at design time:** `6b9985a`
(= the recon commit that is this gate's design source). **Raw-only record.** The executor applies the
pre-declared mechanical kill/pass rules and attaches a **NON-binding** label; the binding close is the
orchestrator's.

**Direction.** TIE-branch recon LEAD candidate **premise-(d)** (`refine-logs/TIE_BRANCH_RECON.md`
commit `6b9985a`, §2). F50's FA gate composed the *healthy frozen CLIP image stream* (EN AUC 0.734)
with the **frozen** Qwen-text stream (AUC 0.851) → composite best-ever EN AUC **0.898**, but
`d_oracle +0.025 < +0.03` (K-FA-2 kill: the AUC edge is easy-example ordering, unconvertible). F50's
own ban language carves the untested cell out — *"do not re-propose fixed compositions … over banked
**FROZEN** features; **conversion requires adaptation (F45)** or a new information source with
alignment>0.663."* Premise-(d) swaps the **frozen** Qwen-text block for the **LoRA-EN-adapted**
Qwen-text block (the B4-arm extraction cache) — the adaptation the ban itself names — keeps the
healthy CLIP image block, and re-runs the FA oracle machinery. **Question: does the LoRA text swap
close the +0.005 oracle gap and convert as a PARETO move, or not?**

---

## 0. DESIGN LOCKED BEFORE THE JUDGED READ (house rule)

Design source = `TIE_BRANCH_RECON.md` §2(e)/(f). Machinery = the validated FA gate
(`scripts/analysis/fa_fusion_gate.py`, FA sha256 `9e2fcbf3…`) + `FA_GATE_RECORD.md` §0 locked params,
reused **verbatim**. Parameters underspecified by the recon were fixed here **before** any judged
number, as follows.

**Machinery = the F44 concat-kNN proxy, reused exactly.** Raw frozen per-video features, per-modality
L2-norm, cosine top-20 rank/sim-weighted signed kNN vote (memory=train, decision `score>0`) —
`encoder_swap_geometry.knn_vote` semantics, the §0.3-validated substrate that reproduces the deployed
align (Hadamard) head's dev **sign** (F44). Raw features ⇒ **deterministic** probe (no head, no
training seed); bootstrap / permutation over dev items give the CIs.

**Arms** (all over the same proxy; MHC-EN primary, HateMM sanity/positive-control):
- **A0 CLIP-concat** = `[imghat_CLIP , texthat_CLIP]` (w=0.5) — baseline reference for every delta.
- **A2F FROZEN-text cross** = `z = [sqrt(w).imghat_CLIP , sqrt(1-w).texthat_Qwen(frozen)]`, w-grid
  {0.00..1.00} (21 steps). This is the **exact FA-A2 arm**, recomputed here as the K-D-0b
  machinery-reproduction anchor — it must match `FA_GATE_OUT.json` bit-close.
- **A2L LoRA-text cross (JUDGED)** = `z = [sqrt(w).imghat_CLIP , sqrt(1-w).texthat_Qwen(LoRA-EN)]`,
  same w-grid. The premise-(d) object: the frozen→LoRA text swap the ban carves out.
- **A1c Qwen-concat control** = `[sqrt(0.5).imghat_Q , sqrt(0.5).texthat_Q]` (frozen Qwen img+txt) —
  used only for the K-FA-3 substrate check + the HateMM positive control.

**Cache provenance (banked, verified; sha256 in §6).** All caches read-only.
- CLIP-img/text: `data/CLIP_Embedding/MHC/{train,dev_seen}_openai_clip-vit-large-patch14-336_HF.pt`
  (mtime 2026-07-01).
- Frozen-Qwen text: `…/{train,dev_seen}_Qwen2.5-VL-7B-Instruct_HF.pt` (2026-07-02).
- **LoRA-EN Qwen text (the judged block):** `…/{train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`
  (mtime **2026-07-02 11:34/11:37** — the stable **B4-era** extraction, *not* the Jul-18 live
  cand-2/HateMM artifacts). Adapter provenance (`B4_FORENSIC_RECON.md:71-93`): `logging/lora/MHC`,
  base `Qwen2.5-VL-7B-Instruct`, r=16/α=32, SFT on `mhc_lora_train` = **EN own-train-split only (549
  records)**, binary hateful/normal (no gold aux, no OCR, no cross-dataset mixing) — clears all three
  standing vetoes. B4 verified id-alignment LoRA↔CLIP **set-equal AND order-equal**, labels
  bit-identical; re-verified here in-code (per-split assert on common-id alignment + label equality).

**w-selection (locked).** Two reads, mirroring FA.
- *Ceiling / judged read:* over the A2L grid, the config maximising **dev** acc (dev-oracle-w) — the
  maximally-favorable point, so a KILL here is airtight. Pareto feasibility (K-D point bars) checked
  over the whole grid; if any config meets the point bars, the max-Δacc feasible one is the candidate,
  else the max-dev-acc ceiling is.
- *Deployable read (supplementary):* w maximising **train-LOO** acc, evaluated on dev.

**Oracle-threshold (K-D-1, locked, BINDING).** B5 port — both the candidate and the CLIP-concat
baseline get their **own** label-oracle decision threshold (the τ maximising dev acc; dev labels touch
the threshold only, test never read). `d_oracle = candidate@oracle − CLIP-concat@oracle`.

**Pre-declared kill-switches (from `TIE_BRANCH_RECON.md` §2(f); FA-ported).**
- **K-D-0 (machinery, VOID on fail):** planted Pareto/rotation detectors fire (calibration) **AND** the
  A2F arm reproduces FA-A2 bit-close (AUC 0.898 / ceiling `d_oracle +0.025`) **AND** the concat proxy
  reproduces F44 −0.012 (substrate) **AND** the HateMM positive control would-pass (Pareto fires +
  `d_oracle ≥ +0.03`). Any fail ⇒ MACHINERY_INVALID.
- **K-D-1 (oracle, BINDING):** candidate `d_oracle < +0.03` ⇒ **KILL** (B5 port — AUC edge is
  easy-example ordering, unconvertible).
- **K-D-2 (Pareto-not-rotation):** rotation at the ceiling (`d_acc ≤ 0` with +hate/−non-hate trade) ⇒
  KILL. Pareto-vs-rotation decomposition (hate-recall vs non-hate cost separately) is **mandatory**.
- **K-D-3 (deployable-w sanity):** train-LOO-selected w must not regress below the CLIP-concat floor.
- **+ point bars** (K-D): `Δhate ≥ +0.03` AND `Δnon-hate ≥ −0.01` AND `Δacc ≥ +0.02` AND bootstrap
  CI-low > 0 (1000×). **+ selection-null** (1000×, shuffle dev y, max-over-w Δacc vs null p95).
- **PASS** = K-D-1 ∧ point-bars+boot ∧ selection-null ∧ deployable(train-w Δacc ≥ +0.02) ∧
  HateMM-sanity, all true. Only a PASS reactivates the moot relaxation-(f) D7-composition sub-ruling.

**Determinism:** fixed RNG=20260717 (same as FA), `OMP_NUM_THREADS=4`, ~seconds CPU.

---

## 1. MACHINERY VALIDATION (K-D-0) — all four legs pass

**(a) Calibration — the Pareto/rotation detector is live (FA-verbatim planted signals).**
- planted pure Pareto (fix 6 hate FN, touch no non-hate): `Δhate +0.24 / Δnon-hate +0.00 / Δacc +0.075`
  → **detector flags Pareto** ✔.
- planted symmetric trade (fix 6 hate FN, break 6 non-hate): `Δhate +0.24 / Δnon-hate −0.109 / Δacc 0.00`
  → **detector flags rotation** ✔.

**(b) K-D-0b — the A2F arm reproduces the FA-A2 frozen result BIT-EXACT.** Recomputing the frozen
CLIP-img⊕Qwen-text cross arm over the whole w-grid and comparing every field (dev_acc, dev_auc, d_acc,
d_hate, d_nonhate, oracle_acc, d_oracle) against `FA_GATE_OUT.json`:
**max absolute difference vs FA = `0.000000`** across all 21 configs. Anchor numbers reproduced:
**peak dev AUC `0.8982`** (FA "0.898 best-ever EN"), ceiling `A2F_frozen_w0.15` **`d_oracle +0.0250`**
(FA +0.0250). The machinery is the identical FA substrate on the identical caches — validity anchor
confirmed.

**(c) K-FA-3 substrate — the concat proxy reproduces the align-head dev sign.** MHC-EN Qwen-concat −
CLIP-concat dev acc = **−0.0125** vs F44's −0.012 (`|Δ|=0.0005 < 0.010`, sign negative). Substrate
faithful.

**(d) HateMM positive control — the detector fires on the KNOWN real conversion (calibrated zero).**
Frozen HateMM Qwen-concat vs CLIP-concat: `Δacc +0.0467 / Δhate +0.1163 / Δnon-hate +0.000` (clean
Pareto), **`d_oracle = +0.0467 ≥ +0.03`** (converts at the oracle threshold), and the FA arms do not
lose the HateMM win. So the K-D-1 kill-switch **fires on MHC-EN (+0.025) and does NOT fire on HateMM's
genuine win (+0.0467)** — the MHC-EN kill below is a **calibrated** kill. (HateMM control uses the
**frozen** Jul-2 caches only; the Jul-18 HateMM LoRA cache was never opened.)

⇒ **K-D-0 machinery_valid = TRUE.**

---

## 2. RAW RESULTS — MHC-EN (PRIMARY), n_train=549, n_dev=80 (25 hate)

Baseline **A0 CLIP-concat**: dev acc **0.7625**, hate-recall **0.600**, non-hate-recall **0.836**,
oracle-threshold acc **0.800**.

### 2.1 A2L — CLIP-image ⊕ **LoRA-EN**-Qwen-text (the judged premise-(d) arm). Deltas vs A0.

| w | dev acc | Δacc | Δhate-rec | Δnon-hate-rec | dev AUC | oracle acc | d_oracle |
|---|---|---|---|---|---|---|---|
| 0.00 (LoRA-text-only) | 0.7625 | +0.0000 | +0.2400 | −0.1091 | 0.8458 | 0.7875 | −0.0125 |
| 0.05 | 0.7750 | +0.0125 | +0.2400 | −0.0909 | 0.8451 | 0.7875 | −0.0125 |
| 0.10 | 0.7875 | +0.0250 | +0.2800 | −0.0909 | 0.8400 | 0.8000 | +0.0000 |
| 0.15 | 0.8000 | +0.0375 | +0.2800 | −0.0727 | 0.8560 | 0.8000 | +0.0000 |
| **0.20 (ceiling)** | **0.8125** | **+0.0500** | **+0.2800** | **−0.0545** | **0.8698** | **0.8250** | **+0.0250** |
| 0.25 | 0.8000 | +0.0375 | +0.2000 | −0.0364 | 0.8633 | 0.8125 | +0.0125 |
| 0.30 | 0.8000 | +0.0375 | +0.1600 | −0.0182 | 0.8589 | 0.8125 | +0.0125 |
| 0.50 | 0.7750 | +0.0125 | −0.0400 | +0.0364 | 0.8327 | 0.8125 | +0.0125 |
| 1.00 (CLIP-img-only) | 0.7500 | −0.0125 | −0.2800 | +0.1091 | 0.7367 | 0.7750 | −0.0250 |

(Full 21-row grid in `PREMISE_D_GATE_OUT.json:MHC-EN.A2L_all_rows`.)
**MAX `d_oracle` anywhere on the entire A2L grid = `+0.0250` < +0.03.** No config converts at the
oracle threshold. **MAX dev AUC = `0.8698`.**

### 2.2 The two decisive comparisons — the LoRA swap does NOT help, it hurts.

| quantity | FROZEN text (A2F, = FA-A2) | LoRA-EN text (A2L, premise-d) | swap effect |
|---|---|---|---|
| peak dev AUC | **0.8982** | **0.8698** | **−0.0284 (WORSE)** |
| max `d_oracle` on grid | +0.0250 | +0.0250 | **+0.0000 (gap NOT closed)** |
| ceiling `d_acc` | +0.0500 | +0.0500 | tie (both +0.05, both selection noise) |
| any Pareto point-bar config? | one (w0.15) | **NONE** | worse shape |

The adaptation the ban names as the conversion mechanism (F45: ZH text AUC 0.847→0.925) does **not**
occur on EN: the LoRA-EN text stream *degrades* the composite ranking (peak AUC −0.028) and leaves the
binding oracle threshold pinned at exactly +0.025 — identical to frozen. This is consistent with the
converging EN negatives the recon flagged (F44 EN label-limited, EN image CLIP 0.734→Qwen 0.599; B4
EN LoRA regresses below both frozen floors; B5 EN AUC-edges = easy-example ordering).

### 2.3 Pareto-vs-rotation decomposition (mandatory, K-D-2).

- **Candidate `A2L_lora_w0.20` shape = NEITHER Pareto NOR clean rotation.** `Δhate +0.2800` (up) but
  `Δnon-hate −0.0545` (a non-hate cost far exceeding the −0.01 tolerance); net `Δacc +0.05` because the
  hate-class gain outweighs the non-hate loss on this class balance. It fails the Pareto point bar
  (`Δnon-hate` too negative) — no config on the grid is Pareto (§2.1). It is not the clean B5 rotation
  either (`Δacc > 0`), so **K-D-2's rotation switch does not fire** — the KILL is carried by the
  binding K-D-1 oracle switch, not by a sign-flip.
- **Ceiling rotation flag = False.** (The kill is the oracle switch, exactly as in FA-A2.)

### 2.4 Degenerate-supervision check. Raw-feature kNN (no trained head) ⇒ the F47 memorisation
pathology cannot recur; train-LOO accs are healthy (deployable pick `A2L_lora_w0.15` train-LOO
**0.8689**), the w-selection target is a low-capacity 1-of-21 scalar. Train-selected w transfers to
only +0.0375 dev (within noise), consistent with no memorisation and no real signal.

---

## 3. INFERENTIAL GUARDS on the candidate (A2L_lora_w0.20) — all fail

| guard | value | verdict |
|---|---|---|
| **point bars** (Δhate ≥ +0.03 ∧ Δnon-hate ≥ −0.01 ∧ Δacc ≥ +0.02) | Δnon-hate **−0.0545** < −0.01 | **fails (not Pareto)** |
| **bootstrap CI-low** (1000×, Δacc over dev items) | Δacc +0.050, **CI [−0.0503, +0.1625]** | CI-low ≤ 0 → **fails** |
| **K-D-1 oracle-threshold** (both arms @ label-oracle τ) | candidate 0.8250 − baseline 0.8000 = **+0.0250** | < +0.03 → **KILL fires** |
| **selection-null** (shuffle dev y, max-over-w Δacc, 1000×) | observed +0.050 vs **null p95 +0.1375**, null-mean +0.0745, **p=0.7532** | +0.05 at the 25th pct of the *noise* floor → **not survived** |

The +0.050 point estimate is **4 videos on n=80** with a bootstrap spread of ±5 videos, smaller than
the median Δacc a blind best-of-21-w search returns on shuffled labels (+0.0745).

---

## 4. MECHANICAL KILL/PASS (pre-declared rules applied verbatim; PRIMARY = MHC-EN)

| switch | rule | value | fires? |
|---|---|---|---|
| **K-D-0 calibration** | Pareto & rotation detectors fire on planted signals | both fire | detector **live** |
| **K-D-0b A2F reproduction** | A2F reproduces FA-A2 (AUC 0.898 / ceiling d_oracle +0.025), max|diff| ≤ 1e-4 | **max|diff| 0.000000** | machinery **VALID** |
| **K-FA-3 substrate** | concat proxy MHC-EN dev (Qwen−CLIP) reproduces F44 −0.012 (±0.010, sign−) | **−0.0125** | substrate **VALID** |
| **K-D-0 HateMM control** | Pareto fires on the known win AND d_oracle ≥ +0.03 | fires; **+0.0467** | **OK (calibrates)** |
| **K-D-1** | candidate `d_oracle < +0.03` ⇒ KILL (B5 port) | **+0.0250** | **KILL fires** |
| point bars + boot | Pareto point bars AND boot CI-low > 0 | no Pareto config; **CI-low −0.0503** | **not passed** |
| selection-null | observed max Δacc > null p95 | +0.050 vs +0.1375 (p=0.7532) | **not survived** |
| **K-D-2** | rotation at ceiling | Δacc +0.05 > 0 (not a rotation) | not fired (kill is K-D-1) |
| **K-D-3** | train-w Δacc ≥ CLIP floor (≥ 0) | +0.0375 | met (but non-Pareto, within noise) |
| deployable | train-w Δacc ≥ +0.02 | +0.0375 | met but non-Pareto & within noise |

## 5. NON-BINDING EXECUTOR LABEL: **KILL** (machinery valid; calibrated; K-D-1 B5 kill-switch fires)

The premise-(d) F50 carve-out — **CLIP-img ⊕ LoRA-EN-Qwen-text** — is **measured and closed on
MHC-EN.** The frozen→LoRA text swap the ban itself names as the conversion mechanism does **not**
convert:

- **The +0.005 oracle gap does not close.** Max `d_oracle` anywhere on the A2L grid is **+0.0250**,
  identical to FA-A2 frozen (+0.025) and below the pre-declared +0.03 bar — the pre-declared **K-D-1
  B5 kill-switch fires**. The candidate's Pareto *shape* is worse than frozen's (a non-hate cost
  −0.0545 vs frozen's +0.018), no Pareto point-bar config exists on the grid, the bootstrap CI crosses
  0 [−0.05, +0.16], and the selection-null is not survived (p=0.7532).
- **The LoRA adaptation actively degrades EN.** Peak composite AUC drops **0.8982 → 0.8698 (−0.0284)**
  — the LoRA-EN text stream is a *worse* text block than frozen Qwen for this composition, the mirror
  image of ZH (F45: LoRA lifts ZH text 0.847→0.925). This matches the standing EN verdicts (F44 EN
  label-limited; B4 EN LoRA regresses below both frozen floors; B5 EN AUC-edge = easy-example
  ordering): 549 EN SFT samples degrade the encoder rather than adapt it.

This is **calibrated**: the identical test gives **+0.0467 (would-pass)** on HateMM's genuine Pareto
win, and the machinery reproduces the FA-A2 frozen result to `0.000000` absolute error.

**Scientific reading (for the reviewer, non-binding).** Premise-(d) was the single genuinely-uncovered,
$0, goal-relevant cell the TIE-branch recon identified — the literal adaptation carve-out of the F50
ban. The measurement closes it: the healthy CLIP image stream composed with the **adapted** (not just
frozen) Qwen text stream still does **not** convert to MHC-EN accuracy at the label-oracle operating
point; the adaptation makes the ranking *worse*, not better. This is the **sixth "better-signal or
adapted-signal / no-conversion" datum** on EN (after P3, S2S-F37, W2-A-F42, router-F47, FA-A2-F50) and
the second time the FA oracle machinery has fired its calibrated K-D-1/K-FA-2 kill on this exact
composition family. **No PASS ⇒ the moot relaxation-(f) D7-composition sub-ruling stays moot; no
CC/ceremony is owed; no headline or family claim is created.** Per the recon §4: with premise-(d)
dead, EN is closed at the frozen, collapsed-adapted, and healthy-img-adapted composition levels
simultaneously; the round-3 terminus is complete for the adaptation family, and the live decision
reverts to the **D7 ruling on generic LoRA** (a user ruling, zero further GPU) plus the 5 TERMINUS
relaxations. **Binding close remains the orchestrator's.**

---

## 6. Provenance / hygiene

- **Script (committed):** `scripts/analysis/premise_d_gate.py`, sha256
  `909f9d1a6b52f77eaf837476e3dd2c5921b1905e24cfe3d135a7cf19db9a931d` (also in
  `PREMISE_D_GATE_OUT.json:script_sha256`). Deterministic (RNG=20260717); CPU-only
  (`OMP_NUM_THREADS=4`); ~seconds. kNN vote machinery reused faithfully from `fa_fusion_gate.py`
  (FA sha256 `9e2fcbf3…`) / `encoder_swap_geometry.py`.
- **Reproduction anchor:** `refine-logs/FA_GATE_OUT.json`, sha256
  `23e5bfda1440a0a1489a57f3c9949d3c91b714d98771166ac0a70d4eeb552c3d` (compared field-by-field, max
  abs diff 0.000000).
- **Outputs:** `refine-logs/PREMISE_D_GATE_OUT.json` (full grid, all deltas, guards, calibration,
  reproduction check, per-switch booleans, cache sha256s).
- **Banked inputs (read-only), sha256:**
  - MHC-EN CLIP: `train` `deea74ff78d81836…`, `dev_seen` `cd08e35fd1a8dc28…`
  - MHC-EN frozen-Qwen: `train` `05a9b2def2923025…`, `dev_seen` `cd5d4c7dc08311f8…`
  - **MHC-EN LoRA-Qwen (judged):** `train` `50293e9a71aa1cb6…`, `dev_seen` `404a3a07cbdae973…`
    (both mtime 2026-07-02 11:34/11:37 — stable B4-era, not the Jul-18 live caches)
  - HateMM CLIP (control): `train` `0802b6ba00669ec5…`, `dev_seen` `ab9cd8a070b93afb…`
  - HateMM frozen-Qwen (control): `train` `ba52bc0da3fa14fe…`, `dev_seen` `1b219e12a5a03d77…`
  (full digests in `PREMISE_D_GATE_OUT.json:cache_sha256`.)
- **ZERO GPU, ZERO Modal, ZERO test-touch** (train + dev_seen features/labels only; `test_seen`
  caches never opened). Gold read = train + dev `labels` only (train-LOO w-selection; dev scoring +
  label-oracle threshold). **Did NOT touch** any live cand-2 chain artifact (jobs 13237-13241,
  `logging/lora/*_curric`, curric caches), the Jul-18 HateMM LoRA cache, `state/`, prereg, config,
  `research-wiki/`, any SLURM job, or any frozen ceremony artifact. Committed on `main`, **not pushed**.
- **Repo HEAD at design/run:** `6b9985a` (the recon commit); recorded via the commit that lands this
  record.
