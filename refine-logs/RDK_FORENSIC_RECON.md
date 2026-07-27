# RDK — RELATIONAL DISTILLATION INTO THE KEY SPACE — FORENSIC RECON (zero-GPU)

**Agent:** forensic-recon (2026-07-28 adversarial wave, candidate 1 of 4) · **Date:** 2026-07-28 NZST.
**Discipline honoured.** CPU-only reading + forensic arithmetic. **ZERO GPU / SLURM / Modal / training /
test-touch.** No prereg written, no job submitted, no frozen artifact mutated. Every number below is either
(a) quoted from a banked record at `file:line`, or (b) re-derived in this write-up from quantities that are
themselves quoted at `file:line` — and each is labelled.

**Status of this document.** This is a **recon-level PRE-CLOSURE**, not a measured KILL. Nothing was run.
The distinction is load-bearing and is used consistently below: a *measured KILL* (F89, F94, F95, F96, F97,
F98) is a frozen-bar verdict against numbers produced by a hash-frozen script; a *recon-level pre-closure*
is a ban-scope + arithmetic argument that says the cell should not be opened, and which a later measurement
could in principle overturn. RDK is pre-closed, **not** killed.

---

## 0. THE CELL, PRECISELY

**Mechanism statement.** Train a map `ψ` on the **banked keys** (train split) such that the cosine geometry
of `ψ(z)` reproduces the **relation geometry of the F95 pair-verifier** — i.e. distil the verifier's
pairwise scores into the key space so that a plain cosine kNN over `ψ(z)` inherits the verifier's ordering.
The **deployed decision rule is untouched**: same retrieval, same `k=20`, same rank weights `[20..1]`, same
threshold at 0, same label field. Only the key map changes.

The candidate's whole appeal is F95's headline asymmetry: the verifier is a **much** better relation scorer
than the deployed cosine and converts **nothing** when it is used *as a decision rule*. RDK's premise is
that the failure was the *deployment surface*, not the *relation*: put the relation back into the metric and
let the deployed averaging cash it.

**Why it looked live.** F95's own numbers say the relation is real:

| dataset (fused space) | cosine pair-AUC | MLP verifier pair-AUC | Δ | source |
|---|---|---|---|---|
| HateMM | 0.5843 | 0.7753 | **+0.1910** | `MECHNOV_PAIRVERIFY_PREGATE.md:268` |
| MHC-ZH | 0.5123 | 0.7748 | **+0.2625** | `:269` |
| MHC-EN | 0.5057 | 0.7009 | **+0.1952** | `:270` |

18/18 cells clear the +0.03 bar with 5/5 fold signs, "by margins of **4.3× to 8.8×** on the primary fused
space" (`MECHNOV_PAIRVERIFY_PREGATE.md:278-279`).

---

## 1. PRE-CLOSURE TABLE

| # | prior finding | binding? | quoted binding text (`file:line`) | ruling for RDK |
|---|---|---|---|---|
| 1 | **F66** (ISR β-decomposition) | **NOT BINDING** | F66 is φ₀-conditional; the NCA recon already litigated this: "F66's β-decomposition is *conditional on a fixed embedding map φ₀* … it bounds **inference-side** operators acting on a fixed similarity structure" (`NCA_FORENSIC_RECON.md:13,106`); ruling "**F66 does NOT bind trained-space reshaping**" (`NCA_FORENSIC_RECON.md:110`) | RDK produces a new map ψ∘φ₀, hence a new Gram matrix. F66's arithmetic does not evaluate on it. **Clean.** |
| 2 | **F75 / NCA** | **PARTIALLY → effectively BINDING on the shared-map form** | LITSWEEP-6 draws the line itself, on the *adjacent* RevisedKey/INK family: "**INK** … move the **datastore keys one-sidedly** — query and its datastore twin may end up in different places — which is genuinely not NCA (**F75, which applies one shared map to both sides**)" (`LITSWEEP6_MEMBANK.md:707-710`) | RDK as specified applies **one shared map to both sides** — it therefore falls on **NCA's** side of the line LITSWEEP-6 drew, not INK's. Worse: the verifier RDK distils was fitted on "**Verifier target** `y = 1[lab_i == lab_j]`" (`MECHNOV_PAIRVERIFY_PREGATE.md:188`) — the **same label-agreement matrix NCA optimises**. RDK is NCA with a two-stage estimator. **F75 binds the shared-map form.** |
| 3 | **F89 / T2b** | **PARTIALLY BINDING** | T2b whitening "de-collapses the cosine? **train top-1 sim 0.9999 → 0.5220**" (`MECHFIX_PREGATE_2026-07-27.md:238`) and "**whitening raises the length organisation sharply** (HateMM 0.52 → 0.87 mean)" (`:288-289`, per-dataset ρ table `:276-280`) | RDK's mechanism **is** T2b's move — a linear re-metrication that de-collapses the cone — performed **with labels** instead of a closed-form whitener. T2b's measured outcome is the honest prior: ΔmF1 **negative on 3/3** datasets (−0.0097 / −0.0053 / −0.0395, `:308`) and Δacc negative on 2/3 with ZH at exactly ±0.0000. See §4 correction. |
| 4 | **F94 alone** | **NOT BINDING** | F94 is about vote *depth*: "k=20 IS AT OR ABOVE THE PLATEAU ON ALL 6 ARMS, and the plateau starts at k~10-15" (F94 body, `findings.jsonl` id F94) | RDK does not change k. |
| 5 | **F94 ∧ F98** | **EFFECTIVELY BINDING (the decisive conjunction)** | F94: "ranks 11-20 are already inert … Items whose prediction differs from k=20 at k=10 is ZERO on 215/215 in 5 of 6 HateMM cells"; "The noise ERRPAT found is at ranks 1-5, where the LABELS THEMSELVES are wrong" (F94 body). F98: family oracle "**+0.1492 / +0.1520 / +0.2186**" (`AGGNET_PREGATE_RECORD.md:368-370`), realised "**+0.0134 / −0.0069 / +0.0000**" (`:682`) | With the top-20 **set** essentially unchanged (§3), RDK's realisable channel is a **per-item re-weighting of a fixed top-20** — exactly F98's function class, whose oracle is 96-100 % of every deployed error and which delivered **+0.0134** at best anywhere in 45 cells. RDK is inside a family measured at its own ceiling. |
| 6 | **D7 (novelty ruling)** | **BINDING ON THE CLAIM** | `REDTEAM_BAN_SCOPE_AUDIT.md` **GAP-7**: the "adapt the retrieval **key-map / head recipe** only, encoder frozen" cell "is ruled out on **D7** (adds no MLLM role → generic classifier tuning)" (`:303-306`); the gap is named "Uncoupled head/key-map recipe … as a *performance* lever — unmeasured, but D7-out-of-scope" (`:310-311`) | RDK is a key-map. Even a positive is **D7-dead as a novelty claim**; it could only ever be a performance/analysis datum. |
| 7 | **Training-signal bans** | **NOT BINDING ON THE LETTER** | `banned_constraints[5]` = "MLLM-scores-as-training-signal"; `[6]` = "P1-P5 re-proposals"; `[1]` = "gold annotations inside method" (`directions_tried.json:454-461`, 0-indexed per the `SEG_REENCODE_FORENSIC_RECON.md:125` convention) | RDK's teacher is **our own** F95 verifier fitted on **gold train labels**, not an MLLM score and not a gold *annotation* (spans/targets). None of [1]/[5]/[6] fires on the letter. |
| 8 | **C5's teacher-as-key ceiling rule** | **BINDING AS A PRECONDITION** | "**The oracle for a training-signal distillation = teacher-as-key retrieval = the encoder-swap result.** A student head distilling Qwen geometry cannot beat *using the Qwen geometry directly as the retrieval key*." (`C5_FORENSIC_RECON.md:133-135`) | Transposed to RDK: RDK's ceiling = **using the verifier's relation directly inside the retrieval weighting**. That object is **VSW** (LITSWEEP6-relgen C4, `LITSWEEP6_RELGEN.md:256-297`), which "**was not tasked and was not run**" (`VGA_PREGATE_RECORD.md:446`). **RDK's oracle is therefore unmeasured, and VSW is its precondition.** |

---

## 2. THE INVALID ARGUMENT — RECORDED AS INVALID

An earlier framing of this recon proposed to kill RDK on an **interaction-share** argument: "only
26.6-37.7 % of the score variance is query×bank interaction, so a bilinear/metric map cannot carry the
verifier's relation." **This argument is INVALID and must not be reused.** Three reasons, in order of
decisiveness:

1. **The figure is the COSINE's, not the verifier's.** `MECHNOV_PAIRVERIFY_PREGATE.md:431-432`: "In the
   deployed raw key space, **only 26.6-37.7 % of the cosine's score variance is query×bank interaction**".
   The **verifier's** share is the opposite: "The trained verifier inverts this to **77-93 % interaction**"
   (`:433-434`); per-dataset MLP interaction 0.9329 / 0.7895 / 0.7745 (`:425,427,429`). Quoting 26.6-37.7 %
   against RDK inverts the sign of the evidence — the verifier is *more* relational, which is an argument
   **for** the cell, not against it.
2. **An ANOVA main/interaction split says nothing about bilinear representability.** A two-way variance
   decomposition of `S[query, bank]` measures how much of the score varies with the *pair* rather than with
   either margin. A metric `⟨ψ(q), ψ(b)⟩` is *pure interaction* by construction up to the norms; there is no
   theorem taking a variance share to a rank/approximation bound on the map class.
3. It was never measured on ψ.

**The VALID proxy, and it is already banked.** F95 fitted two model families on the identical pairs. The
**logistic** arm is, by construction, a label-supervised **diagonal re-metric** of the key space — i.e. the
nearest banked object to "distil the relation into a metric":

> "the logistic arm is a label-supervised **diagonal re-metric**, which lives in the same shape as F89's
> operators" (`MECHNOV_PAIRVERIFY_PREGATE.md:107`)

Its share of the MLP's pair-AUC advantage over the raw cosine, **re-derived in this recon** from the fused
rows of `MECHNOV_PAIRVERIFY_PREGATE.md:268-270`:

| dataset (fused) | MLP Δ vs cos | logistic Δ vs cos | logistic / MLP |
|---|---|---|---|
| HateMM | +0.1910 | +0.1292 | **0.676** |
| MHC-ZH | +0.2625 | +0.1687 | **0.643** |
| MHC-EN | +0.1952 | +0.1410 | **0.722** |

**A metric-shaped student recovers 64–72 % of the relational advantage.** That is the honest read: RDK is
*not* pre-killed by representability — a metric can carry most of the relation. It is pre-closed for the
reasons in §1 and §3, not for this one.

*(Caveat carried, not hidden: the logistic arm additionally fires F95 control 4 — "the plain logistic arm
collapses" to positive rate 0.0237-0.0604 on ZH/EN, `MECHNOV_PAIRVERIFY_PREGATE.md:347-350` — so its
**end-to-end** numbers are contaminated. The **pair-AUC** numbers used above are held-out ranking
quantities and are not affected by the decision-side collapse.)*

---

## 3. NEWLY DERIVED ARITHMETIC — THE PURE-PERMUTATION ORACLE

This is the most transferable thing this recon produced, and it is **pure arithmetic on quantities already
in the ledger** — no new measurement, reproducible by hand.

### 3.1 The set does not move

RDK re-metricates; it does not change *which* items are in the bank. Its realisable effect on the decision
is therefore (i) a possible reordering inside the retrieved list, and (ii) a possible change of membership
at the boundary of the top-20. The banked evidence says (ii) is small and (i) is where all the mass is:

- "**72-92 % of all deployed errors are in the pathology population** (HateMM 88/116, ZH 79/88, EN 109/121)"
  where the pathology population is "deployed-wrong items whose **nearest same-gold-class bank item sits
  within rank 5** by full-space cosine" (`MECHNOV_PAIRVERIFY_PREGATE.md:369-374`).
  *(Erratum noted, not carried: the three counts compute to 75.9 % / 89.8 % / 90.1 %, not "72-92 %". The
  prose range in that sentence is wider than its own parenthesised counts. The counts are the primary datum.)*
- The right analogue is already retrieved and top-ranked; it is **out-voted**, not missing
  (`MECHNOV_PAIRVERIFY_PREGATE.md:372-376`; ERRPAT-ZH "median rank **1.5**", `ERRPAT_MHC-ZH_2026-07-26.md:235`).

So the live question is: **how much accuracy can a pure re-ordering of an unchanged top-20 buy, at most?**

### 3.2 The flip condition (derived here)

The deployed score is (`NCA_FORENSIC_RECON.md:30-35`, `metrics.py:262-284`)

```
s(q) = Σ_{j∈top20} w_j · (2·y_j − 1) · cos(φ(q), φ(k_j)) / Σ w_j ,   w = [20,19,…,1],  Σ w = 210
decision = [s ≥ 0]
```

In the **cone-collapsed** regime all twenty cosines are equal to first order — and the collapse is measured,
not assumed: "Cosine is saturated at ~0.9999 for both errors and correct items (the head space is collapsed
onto a narrow cone)" (`ERRPAT_HateMM_2026-07-26.md:139-141`); median top-1 neighbour cosine 0.999852 for
errors vs 0.999976 for correct (`:131`). With equal cosines the sign of `s` depends only on `Σ_j w_j σ_j`.

Let `m` = number of top-20 neighbours carrying the query's **true** class. The **best possible permutation**
puts those `m` at the head of the list, giving weight mass `W(m) = Σ_{i=1..m}(21−i) = 21m − m(m+1)/2`, so

```
Σ w σ = 2·W(m) − 210 ,   which is ≥ 0  ⟺  m(41 − m) ≥ 210
```

- **hate query** (true class = +1): needs `s ≥ 0`, and `m(41−m) ≥ 210` first holds at **m = 6** (6·35 = 210,
  the tie, which `[s ≥ 0]` resolves toward hate). ⇒ **flippable iff purity ≥ 6/20 = 0.30**.
- **non-hate query** (true class = −1): needs `s < 0` **strictly**, and `k(41−k) > 210` first holds at
  **k = 7** (7·34 = 238; k = 6 gives exactly 210 and fails the strict inequality).
  ⇒ **flippable iff purity ≥ 7/20 = 0.35**.

**Below purity 6/20, no permutation of the retrieved list — and hence no re-metrication that preserves the
set — can flip the prediction.** This is the exact, elementary form of F94's "ranks 11-20 are inert /
the noise is at ranks 1-5".

### 3.3 Crossing it with the measured error purity

| dataset | banked purity distribution over deployed errors | source | errors above the 0.30/0.35 threshold | test n | **permutation cap** |
|---|---|---|---|---|---|
| **HateMM** | "purity <0.5 for **24-27 of 26-28** errors in every cell … with **purity <0.25 for 21/27**" | `ERRPAT_HateMM_2026-07-26.md:143-144` | ≤ 27 − 21 = **6** | **215** | **≤ +0.0279** |
| **MHC-ZH** | "**8 at ≤ 0.10, 7 in (0.10, 0.25], 7 in (0.25, 0.45], and zero above 0.45**" (22 stable-core items) | `ERRPAT_MHC-ZH_2026-07-26.md:222-223` | ≤ **7** (the whole (0.25,0.45] band, an over-count) | **149** | **≤ +0.0470** |

Test sizes `HateMM clean = 215 / ZH = 149` are the wiki-consistent figures re-affirmed at
`research-wiki/PAPER_MASTER_TABLES.md:418`, and are the same n used by F89 (`MECHFIX_PREGATE_2026-07-27.md:303`).

**Both caps assume ZERO breaks** — every flip is a fix and no currently-correct item is lost. That has never
happened in this campaign: the exchange rate is "0.53-0.95 in the primary cells and never exceeds 1.17
anywhere" (`MECHNOV_PAIRVERIFY_PREGATE.md:458-459`); LITSWEEP-6 states it as a law
(`LITSWEEP6_MEMBANK.md:39-44`). Under any realised exchange rate < 1 the true cap is strictly smaller.

**Reading.** The **zero-break** upper bound is **below the +0.030 house bar on HateMM** and clears it on ZH
only by 0.0170 — 7 test items — against a **≥2-dataset** requirement. Even granting RDK a perfect,
break-free re-ordering of an unchanged top-20, it cannot satisfy the goal. This is a *sufficient* pre-closure
of the set-preserving channel on its own, independent of every ban in §1.

**Honest limitation of the arithmetic, stated plainly.** The bound is exact only for the set-preserving
channel. A trained ψ can also *change membership* of the top-20 (pull a new item in from rank 21+). That
channel is not bounded by §3.2. It is, however, (a) F98's family by another name once the new item is
inside, and (b) the channel the F95 verifier already exercised end-to-end and lost on: −0.0040 / −0.0466 /
−0.0146 (`MECHNOV_PAIRVERIFY_PREGATE.md:298-300`).

---

## 4. CORRECTIONS TO THE TASKING (claims that did NOT check out verbatim)

Recorded here rather than silently propagated, per house discipline.

1. **"T2b was negative on 3/3."** Precisely: T2b is **ΔmF1-negative on 3/3** (−0.0097 HateMM / −0.0053 ZH /
   −0.0395 EN) but **Δacc-negative on 2/3** — MHC-ZH Δacc is **exactly +0.0000**
   (`MECHFIX_PREGATE_2026-07-27.md:308`). The per-seed ZH vector is [+0.0134, −0.0067, −0.0067] (`:314`).
2. **The "72-92 %" pathology range** does not match its own counts (75.9 / 89.8 / 90.1 %); see §3.1.
3. **"F95's oracle was independently reproduced".** The +0.1492 / +0.1520 / +0.2186 triple is **F98's**
   (`AGGNET_PREGATE_RECORD.md:368-370`), not F95's; F95's adjudication-family ceiling is
   +0.0726 / +0.0535 / +0.0893 (`findings.jsonl` F97 body). They are different families and must not be merged.

---

## 5. VERDICT

> **ALIVE ONLY BEHIND A RULING — and D7-DEAD ON THE CLAIM. Recorded as a recon-level PRE-CLOSURE,
> pending the VSW result.**

- **On performance:** the set-preserving channel is capped by §3 at **≤ +0.0279 (HateMM) / ≤ +0.0470 (ZH)**
  under a *zero-break* assumption the campaign has never met. The membership-changing channel is F98's
  family, measured at **+0.0134** best-anywhere against a **+0.1492-to-+0.2186** oracle.
- **On novelty:** GAP-7's D7 ruling ("adds no MLLM role → generic classifier tuning",
  `REDTEAM_BAN_SCOPE_AUDIT.md:305-306`) fires on the *claim* regardless of the number.
- **On bans:** F75 binds the shared-map form (`LITSWEEP6_MEMBANK.md:707-710` + the `1[lab_i==lab_j]` target
  identity at `MECHNOV_PAIRVERIFY_PREGATE.md:188`). A **one-sided** (INK-shaped) variant escapes *that*
  sentence — and only that one; it still meets §3 and GAP-7, and LITSWEEP-6 already ranks the one-sided
  family "the highest overfitting risk in the sweep, on the axis nearest to closed territory. **Do not spend
  before C1-C4**" (`LITSWEEP6_MEMBANK.md:710-713`) — and C1/C3 are now measured dead (F96/F98).
- **The one legitimate precondition, if anyone ever wants to reopen this:** run **VSW**
  (`LITSWEEP6_RELGEN.md:256-297`), which is the direct-use ceiling C5's rule
  (`C5_FORENSIC_RECON.md:133-135`) says bounds any distillation of the same teacher. VSW is a **~1 hour CPU
  rider on the existing emitter** ("the emitter now exists, so C4's lambda-sweep exchange-rate curve is a
  much cheaper rider than when the sweep record priced it", `findings.jsonl` F97 body). If VSW's λ-curve
  never crosses exchange rate 1.0, RDK is closed *arithmetically* rather than by recon.

### P(pass) estimates (honest, per bar)

| bar | estimate | reasoning |
|---|---|---|
| P(≥ +0.030 acc on ≥2 datasets, both protocols) — the goal bar | **≤ 1 %** | §3 caps HateMM below the bar outright; ZH needs 7 of 7 eligible flips with zero breaks |
| P(≥ +0.030 on ≥1 dataset) | **2–4 %** | ZH only, and only in the zero-break corner |
| P(≥ +0.010 on ≥1 dataset, train-arena pregate) | **15–25 %** | F98's +0.0134 shows this band is reachable by re-weighting |
| P(the D7 novelty claim survives even given a pass) | **~0 %** | GAP-7 is a ruling, not a measurement |

**Cost if it were ever run:** the pregate is $0 CPU on banked train-split keys (F95/F98 harness reusable);
the full version needs a head re-mint per dataset per seed and is therefore **not** 0 GPU-h — see
`BSY_FORENSIC_RECON.md §4` for the same cost correction.

---

## PROVENANCE

- Ban ledger: `autoresearch/goal_mllm_plus3/state/directions_tried.json` (`banned_constraints`,
  `dead` entries for F89/F94/F95/F96/F97/F98); `state/findings.jsonl` F66, F70, F75, F89, F94, F95, F97, F98.
- Records read directly: `MECHNOV_PAIRVERIFY_PREGATE.md` (F95), `MECHFIX_PREGATE_2026-07-27.md` (F89),
  `AGGNET_PREGATE_RECORD.md` (F98), `RESTRANS_PREGATE_RECORD.md` (F96), `VGA_PREGATE_RECORD.md` (F97),
  `LITSWEEP6_MEMBANK.md` §0/§7.6, `LITSWEEP6_RELGEN.md` C4/VSW, `NCA_FORENSIC_RECON.md` (F66 ruling),
  `C5_FORENSIC_RECON.md` §C, `REDTEAM_BAN_SCOPE_AUDIT.md` GAP-7,
  `ERRPAT_{HateMM,MHC-ZH}_2026-07-26.md`, `research-wiki/PAPER_MASTER_TABLES.md`.
- **Computed in this write-up** (arithmetic only, no data touched): the logistic/MLP pair-AUC ratio table
  (§2) from `MECHNOV_PAIRVERIFY_PREGATE.md:268-270`; the permutation flip thresholds and the
  ≤ +0.0279 / ≤ +0.0470 caps (§3) from the ERRPAT purity bands and the `[20..1]` weight profile.
- **Required statements:** ZERO GPU / SLURM / Modal / training spent by this recon; no held-out test metric
  read or produced; no `state/` mutated by this file (the finding row and ban entry are written separately);
  no prereg, config, or frozen artifact touched.
