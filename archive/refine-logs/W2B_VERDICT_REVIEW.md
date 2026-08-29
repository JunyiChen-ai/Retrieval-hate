# W2-B Verdict Review — BINDING RULING (sub-clip set-matching, cloud triage, K4-primary)

**Reviewer:** fresh zero-context independent verdict reviewer (did not design, implement, or execute this
probe). Read-only; no GPU / no SLURM / no Modal. This document is the **binding ruling** on W2-B, rendered
strictly against the pre-registered verdict rules; the probe script's own `mechanical_gate_check` arithmetic
is explicitly **NOT** the binding verdict (house rule) — this review is.

**Evidence read directly (not summaries):**
- `refine-logs/W2B_PROBE_DESIGN.md` (r1, B1–B3 + N1–N5 folded; verdict rules §4/§5, dataset rule a/b/c/d).
- `refine-logs/W2B_PREREG_REVIEW.md` (APPROVED-WITH-AMENDMENTS §1–§9; §10 code re-check → CLOUD EXECUTION
  CLEARED; binding terms §8).
- `refine-logs/W2B_FORENSIC_RECON.md` (cache reality, Delta-1 non-isomorphism, CLIP<Qwen asymmetry).
- `refine-logs/W2B_PROBE_RECORD.md` (RAW execution record — the transcribed Modal numbers) **and**
  `refine-logs/w2b_probe_results.json` (raw source; I re-derived the paired Δ from the per-arm acc/mF1 to
  confirm the record's transcription is faithful — it is, to 4dp).

**Provenance checks (cheap, done):**
- `git log`: the probe was committed at `07d1982` ("W2-B probe executed on Modal … raw record, awaiting
  verdict review"); code+amendments at `bc1810b`. HEAD at ruling time = `2bf00cb` (W2-A r2b, unrelated).
- Probe script hash: `sha256sum scripts/analysis/w2b_probe.py` =
  `d22aac02b4c50f2952e1aa06b4609dd158d69ff54dd184cd9885fec1d3a15776` — **matches** the design §11 r1 freeze
  and the record's provenance claim (unchanged pre/post run). Code was independently reviewed +
  hash-frozen before dispatch (prereg review §10, "CLOUD EXECUTION CLEARED").
- JSON↔record fidelity spot-check: HateMM SET acc 0.7520564 / POOLED acc 0.7567568 ⇒ Δacc −0.0047 (record
  −0.0047); oracle `d_acc` 0.07755581668625144 (record +0.0776); MHC Δacc +0.0015898 (record +0.0016). All
  match. No transcription drift.

---

## A. ORACLE KILL-SWITCH — checked FIRST (pre-registered gate 2, K4-primary per B3)

**Rule (design §4 gate 2 / §3.8, pinned to the K4 PRIMARY arm by amendment B3):** DEAD-family (outcome (a))
**iff** oracle Δ(oracle − POOLED) **< +0.04 on EVERY dataset**. The oracle is computed on the K4 primary
(train∪dev, 851/629) only; a K30-inflated ceiling cannot bypass this early kill.

**Raw numbers (verbatim from the record, K4 primary):**
- **HateMM:** oracle acc **0.8343** ⇒ Δ vs POOLED acc **+0.0776**, mF1 **+0.0754**.
- **MHC-EN:** oracle acc **0.7886** ⇒ Δ vs POOLED acc **+0.0700**, mF1 **+0.1015**.

**Ordering-sanity (design §3.8 "raw Δ materially above oracle ⇒ oracle bug"):** HateMM raw Δacc −0.0047 ≤
oracle +0.0776; MHC-EN raw Δacc +0.0016 ≤ oracle +0.0700. Oracle ≥ raw on both — **no oracle bug**, no
auto-kill/investigate condition.

**OUTCOME — ORACLE KILL-SWITCH DOES NOT FIRE → SURVIVES.** Oracle headroom is **ABOVE +0.04 on BOTH**
datasets (+0.0776 / +0.0700 acc), so the "< +0.04 on every dataset" DEAD-family condition is **not** met.
Consequence: outcome **(a) DEAD-family is ruled OUT**. There is genuine convertible sub-clip-alignment
structure that pooling discards (oracle beats pool by ~+0.07–0.10 acc) — but see §C: the **unsupervised
MeanMaxSim metric does not realize any of it at the decision level.** The verdict therefore turns on the raw
K4 bars, not the oracle.

---

## B. K4 SOLE-SURVIVAL ARM (amendment B2) — sensitivity arms are NON-determining

Per amendment **B2** (design §5), the **K4 PRIMARY (train∪dev, 851/629) is the SOLE survival-determining
arm.** The ruling below is rendered **exclusively** on the K4-primary paired Δ. All other arms are
explicitly non-survival-determining and cannot rescue, convert, or veto the K4-primary decision:

- **`_mm` (MHC-EN train-only, 549):** modality **sensitivity** report only. Recorded for completeness:
  Δ(MM−POOLED) acc +0.0128 / mF1 +0.0542; Δ(MM−VIS) acc **−0.0073** / mF1 +0.0180; mm-vs-vis obs>null95
  **False**, bootstrap-5th −0.0328. Non-determining; does not enter the verdict.
- **SET-Chamfer, WITH-TEXT, ASYM:** sensitivity arms; ASYM `beats_set` = False (HateMM) / True-but-tiny
  (MHC-EN, Δacc +0.0048, not significant). Non-determining.
- **K30 granularity sensitivity (HateMM train-only, 744): DEFERRED — not run on cloud** (`--k30_sensitivity
  0`, TERM-2). By B2 it is a **breadth-modifier** that can only characterize whether a *negative* is
  K4-specific or persists across granularity; it **can never rescue a failed K4 primary and can never
  convert a survival into a kill.**

**Ruling on the deferred local K30 sensitivity: MOOT for the binding verdict.** The verdict (below) is a
KILL decided entirely on K4; K30 cannot change it. Because this is a cloud-triage kill whose only downstream
consequence is a *prior-down* update on the don't-pool family (no local comparison table, no GPU authorized
on this line), the K30 breadth read is **optional color, not required** — it would at most add a footnote of
the form "the negative also holds / does not hold at 30-way granularity," which does not alter the ruling,
the ban scope, or the S2S decision. **Recommendation: do NOT spend local GPU on the K30 breadth read** unless
a specific granularity-breadth footnote is later wanted for the writeup; it is not needed to close the line.

---

## C. FOUR DATASET-RULE ROWS (raw numbers quoted verbatim; K4 primary)

### Row 1 — HateMM RAW BAR (anchor): Δacc ≥ +0.05 AND ΔmF1 ≥ +0.05, corroborated by rank-only
- **Primary paired Δ(SET−POOLED):** acc **−0.0047**, macro-F1 **−0.0077**.
- **RULING: FAILS — decisively.** SET is not merely below the +0.05 bar; it is **worse than POOLED** (both
  Δ are negative). The +0.05/+0.05 AND-bar is missed by the sign, not by a margin.

### Row 2 — HateMM CORROBORATION CHAIN (rank-only / permutation-null / bootstrap / near-dup-exclusion)
Verbatim from the record:
- **Rank-only (A2 credit rule, sign AND null-95th AND boot-5th):** obs Δacc **+0.0000** (does not even match
  the negative primary sign), vs rank-only null-95th +0.0213; rank-only bootstrap-5th −0.0153 ⇒
  **corroborates = False** (sign=False, null=False, boot=False).
- **Permutation null (N1, 100 seeds):** obs Δacc **−0.0047** vs null-95th **+0.0235** → below; obs ΔmF1
  **−0.0077** vs null-95th **+0.0271** → below. **Not significant.**
- **Bootstrap (1000 resamples):** Δacc 5th-pct **−0.0188** (not > 0 ⇒ **D3-FRAGILE**); ΔmF1 5th-pct
  **−0.0227** (not > 0 ⇒ **D3-FRAGILE**).
- **Near-dup exclusion (A3):** 125 flagged pairs (≥0.995); excluded-retrieval Δ(SET−POOLED) acc **−0.0141**,
  mF1 **−0.0181** → the SET advantage **does not survive** (it was negative and worsens on exclusion).
- **Fano (machine validity):** ±1 gold-label key vote acc **1.0000** ≥ 0.99 → vote machine **VALID**; the
  negative verdict is **admissible** (not VOID).
- **RULING: the entire corroboration chain FAILS.** With Fano valid, the negative is admissible; every
  corroboration leg (rank-only, permutation null, bootstrap, near-dup exclusion) is BELOW its bar. HateMM is
  an unambiguous, machine-valid negative.

### Row 3 — MHC-EN SURVIVAL BAR: Δacc ≥ +0.03 AND ΔmF1 ≥ +0.03
- **Primary paired Δ(SET−POOLED):** acc **+0.0016**, macro-F1 **+0.0031**.
- Corroboration (for completeness, though the raw magnitude alone fails): rank-only corroborates = **False**
  (obs Δacc +0.0079 vs null-95th +0.0175, boot-5th −0.0048); permutation null obs Δacc +0.0016 vs null-95th
  **+0.0160** → below; bootstrap Δacc 5th-pct **−0.0127** (not > 0). Fano **1.0000** (valid).
- **RULING: FAILS.** The signs are positive but the magnitudes (+0.0016 / +0.0031) are **~20× under** the
  +0.03 bar and within the permutation null. SET ≈ POOLED on MHC-EN.

### Row 4 — COMBINED DATASET RULE a/b/c/d (design §4 gate 8, K4-primary determined)
- **(a) DEAD-family** — oracle < +0.04 on every dataset? **NO** (oracle survived both: +0.0776 / +0.0700).
- **(b) BOTH** — HateMM raw bar (corroborated) AND MHC-EN survival bar both cleared? **NO** (neither cleared).
- **(c) SINGLE** — exactly one of {HateMM raw, MHC-EN survival} cleared? **NO** (zero cleared).
- **(d) NEGATIVE** — neither raw bar clears but oracle survived on ≥1 dataset? **YES.** Oracle survived on
  **both**; neither raw bar clears. → **Outcome (d) NEGATIVE**: *set ≈ pool at the decision level on frozen
  CLIP despite oracle headroom → weak-negative family prior update.*

**RULING: outcome (d) NEGATIVE** is the unique pre-registered dataset-rule row that fires.

---

## D. BINDING VERDICT

> ## **KILLED — outcome (d) NEGATIVE (weak-negative don't-pool-family prior update).**
> The pre-registered **KILL** (design §5, K4-primary only) fires: HateMM K4 primary Δacc AND ΔmF1 are
> **< +0.05** (in fact negative, rank-only-uncorroborated — which the rule states "counts as below") **AND**
> MHC-EN K4 primary Δ is **< +0.03/+0.03**. Set-matching does **not** beat pooling at the sub-clip
> granularity on banked frozen-CLIP features on either anchor dataset. This is **NOT** a SURVIVE/ESCALATE
> outcome (b), and **NOT** a SINGLE outcome (c). It is the KILL branch, at the (d) NEGATIVE (not (a)
> DEAD-family) intensity — because the label-oracle retains ~+0.07–0.10 acc headroom on both datasets, i.e.
> there IS convertible sub-clip-alignment structure, but the unsupervised MeanMaxSim/Chamfer/ASYM metric
> cannot realize it at the decision level on frozen CLIP.

**Note on the (a)-vs-(d) intensity wording:** design §5's KILL box calls the KILL a "strong negative prior
update" for outcomes (a)/(d) jointly, while §4 gate 8 assigns the more precise per-row label — (a) = strong,
(d) = **weak-negative** (oracle headroom remains). Ruling on the precise pre-registered row: this is **(d)
weak-negative**, not (a) strong/DEAD-family. This is not a renegotiation of any threshold — the KILL fires
either way; the (d) label is simply the correct intensity given the surviving oracle.

**What is now BANNED from re-proposal (mechanism-level, NOT broader than pre-registered):**
- **Zero-training, frozen-CLIP sub-clip (K=4) set-matching (MeanMaxSim / Chamfer / ASYM) as a retrieval
  metric that beats pooled-cosine, on hate-video video-level LOO kNN, on HateMM and MHC-EN.** That specific
  mechanism at that specific scope is refuted (SET ≤ POOLED at the decision level, machine-valid).
- Combined with the prior Delta-1 kill (the *trained* multi-granularity version, `exp-seg-mode-ablation.md`,
  MHClip, "do not re-attempt segment-level temporal retrieval without gold spans"), the **frozen-CLIP
  sub-clip don't-pool retrieval-metric family is now closed** on these datasets: both the trained and the
  zero-training witnesses are negative.

**What is explicitly NOT banned (scope discipline — do not over-read):**
- The **Qwen-token S2S line is NOT killed** by this ruling (see §E; separate encoder, separate prereg).
- **W2-C (temporal kernel)** and **C2** are not killed — only their priors move (§E).
- **Supervised / gold-span** sub-clip use is untouched (the oracle headroom shows the ceiling exists; this
  probe only refutes the *unsupervised frozen-CLIP metric*, not the existence of segment structure).
- **`_mm` multimodal, WITH-TEXT, K30, MHC_zh** — sensitivity/breadth arms; not adjudicated, not banned.

---

## E. FAMILY INTERPRETATION MATRIX (EXPLICITLY NON-BINDING for other lines)

This ruling binds **only** W2-B. The following are honest directional prior updates, not verdicts on other
lines (each of which carries its own prereg + bars):

- **S2S (Qwen2.5-VL per-frame set-matching) — prior lowered, NOT vetoed.** Per the pre-declared **CLIP<Qwen
  encoder asymmetry** (recon §4 / design §9 threat #1 / §4.8 N5), a frozen-CLIP null **cannot close** the
  Qwen-token version. W2-B's flat result predicts S2S will likely also be flat *if the effect were
  encoder-agnostic*, and it argues **against** spending S2S's Qwen-frame extraction GPU cheaply — but two
  facts keep S2S's hypothesis genuinely untouched: (i) S2S uses a **stronger, different encoder** (MLLM
  per-frame token vectors vs frozen-CLIP 4-frame-mean sub-clips, a coarser unit), and (ii) the **oracle
  survived** here (+0.0776/+0.0700), so the failure is "the *unsupervised metric* can't convert existing
  segment structure on CLIP," not "there is no segment structure" — a stronger encoder could plausibly make
  that same structure separable to MeanMaxSim. **Prior-update sentence:** *W2-B's flat/negative frozen-CLIP
  result lowers the S2S prior and argues against a cheap Qwen-GPU bet, but the CLIP<Qwen encoder asymmetry
  plus the surviving oracle headroom mean it does NOT veto S2S — S2S must still be judged on its own
  separate prereg bars.*
- **W2-C (temporal kernel):** same don't-pool family; prior revised **down** by this frozen-CLIP witness,
  bounded by the identical CLIP<Qwen caveat. Non-binding; not killed.
- **C2:** prior nudged down per §4.8(a) family logic; non-binding.

---

## F. TRIAGE STAMP

**Every number in this review is CLOUD-TRIAGE tier.** The probe ran on Modal CPU (app
`ap-qRhIPZPGASmeO9JZVuJmMQ`, ephemeral/stopped), features-only. Cloud results carry ~1.4pt cross-hardware
drift (seed-noise magnitude) and are triage-only: they may kill a line or authorize queueing a formal local
validation, but they **NEVER enter a local comparison table**. This ruling is a KILL / decline-to-spend-GPU
update, which the house rules (design §7 G-repro note; prereg review §4/§8.6) explicitly sanction as a
sufficient use of a cloud number — **no local re-run is required to decline the GPU.** The magnitudes here
are not close calls (HateMM SET is *negative* vs POOLED; MHC-EN is ~20× under bar and within the null), so
the ~1.4pt drift band cannot flip the verdict. None of these numbers may be mixed into any local table.

---

## G. HYGIENE / PROVENANCE GAPS — none affect the ruling's validity

1. **Mid-chain runner plumbing patch (`features.commit()`).** `scripts/cloud/modal_probe_runner.py`
   `_execute` gained a post-subprocess `features.commit()` so `/root/data` output writes persist to the
   volume for `modal volume get` (record §Provenance; prereg review TERM-1 anticipated exactly this). This
   is **runner plumbing only** — the probe script's logic and its sha256 (`d22aac02…d3a15776`) are unchanged
   and match the hash-frozen, code-reviewed version. It affects only whether the output files were
   retrievable, not any computed number. **No impact on validity.**
2. **Reduced per-frame optional null (30 seeds vs default 100).** `--n_perframe_null 30`. This is the
   OPTIONAL, explicitly **non-gating** per-sub-clip-vector shuffle null (design §4 gate 5, "reported not
   gating"). All **GATING** statistics — the permutation null (0..99 = 100 seeds) and bootstrap (1000
   resamples) — ran at pre-registered defaults. The reduced per-frame null does not enter any bar. (For
   record: its 95th-pct are strongly negative — HateMM Δacc-95th −0.1838, MHC −0.0302 — consistent with the
   negative verdict, but irrelevant to gating.) **No impact.**
3. **K30 deferred (not run on cloud).** Non-survival-determining by B2; ruled MOOT in §B. **No impact.**
4. **MHC_zh optional third-dataset arm not run.** Out of the binding gap, non-primary, non-gating (design
   §1 N1). **No impact.**
5. **Structural integrity confirmed:** video-count guards PASSED (851/629 — the B1 no-flat-sub-clip-bank
   guard held, so no trivial sibling-retrieval leakage); Fano = 1.0000 both datasets (machine valid, verdict
   admissible not VOID); code was independently reviewed + hash-frozen pre-dispatch (prereg §10). The
   synthetic planted-shared-unit self-test (MMS>POOLED) was cleared at code review.

**Conclusion on validity:** all provenance gaps are in **non-gating / plumbing / deferred-non-determining**
territory. The gating machinery (K4 primary paired Δ, permutation null 100, bootstrap 1000, rank-only credit
rule, near-dup exclusion, Fano, oracle on K4) ran to spec at pre-registered defaults. **The ruling is
valid.**

---

## VERDICT LINE

**W2-B: KILLED — dataset-rule outcome (d) NEGATIVE (weak-negative don't-pool-family prior update). Cloud
triage, K4-primary. Oracle kill-switch SURVIVED (headroom exists) but neither raw bar clears → the KILL
fires at the (d), not (a), intensity. Bans, at mechanism scope, zero-training frozen-CLIP sub-clip
set-matching as a retrieval-metric accuracy lever on HateMM/MHC-EN; does NOT veto the Qwen-token S2S line.**
