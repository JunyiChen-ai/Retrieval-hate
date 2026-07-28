# LITSWEEP-7 — THE LANDING SITE

**Date:** 2026-07-28 · **Agent:** litsweep-7 · **Cost: $0** — WebSearch/WebFetch + read-only repo
forensics + arithmetic on already-banked recorded numbers. **ZERO GPU / SLURM / Modal / training /
code execution against any dataset file. TEST-SPLIT CONTACT: NONE** (no `test_seen_*`, no
`data/gt/*/test.jsonl`, no `errpat_*`/`mechfix_*`/`ksweep`/`p2_out` artifact opened).
`scripts/analysis/vsw_*` was **read only**, never edited or run.

**Lens ordered by the team lead:** not a better signal, not a better operator — mechanisms that change
**where a gain lands**: which items are decision-active, what functional of the same geometry the
decision reads, or the conditioning of the decision problem itself.

**Inputs read in full before searching:** `refine-logs/LITSWEEP6_{RELGEN,PARADIGM,MEMBANK}.md`,
`refine-logs/LITSWEEP5_COMPLETENESS.md`, `autoresearch/goal_mllm_plus3/state/directions_tried.json`
(70 dead entries, 9 banned constraints, 6 positives), `state/findings.jsonl` F88-F102, and the six
pregate records F89/F94/F95/F96/F97/F98 plus the four recons F99-F102 and `VSW_PREGATE_RECORD.md`.

---

## §0. TWO CORRECTIONS TO THE TASKING, BEFORE ANYTHING ELSE

Both are load-bearing: one changes the sweep's premise, one removes a "certified datum" from the
ledger. Neither is a quibble.

### 0.1 VSW did **not** return a pass. It returned a KILL with an above-bar *hindsight* ceiling on one dataset.

The tasking says *"VSW just returned the campaign's first permutation-validated positive on the
relational asset: HateMM verifier-soft-reweighting +0.0255 at p=0.0050 (null mean -0.0005 +/- 0.0053)"*.
What the record says:

* `refine-logs/VSW_PREGATE_RECORD.md:579-581` — **"0 of 3 datasets reach +0.030 under any family. The
  best λ-selected number anywhere in the battery is +0.0255 … i.e. 85 % of the bar on one dataset and
  ~0 on the other two."** K-VSW-1 **FAIL**.
* `VSW_PREGATE_RECORD.md:902-905` — **"# KILL as a performance lever."**
* `VSW_PREGATE_RECORD.md:785-796` — **MHC-ZH is killed on its own terms by BOTH mandatory degeneracy
  controls** (DEG-A 0.9516, DEG-B 0.9706 at k=20 — *the deployed rule itself*), against a 0.95 kill
  line. MHC-EN is **+0.0018**, not "-0.0017 on ZH and nothing else".
* **The permutation p-values for the real datasets are NOT in the record.** `:889-896` is a placeholder
  (`<!-- NULL RESULTS GO HERE -->`) with the note "RUNNING at the time this section was drafted". The
  only permutation p-values written anywhere in the file are the **synthetic self-test** ones (`:381-385`).
  The per-draw deltas exist in `scripts/analysis/vsw_perm_{hatemm,zh,en}_OUT.json`; **the p=0.0050 /
  null-mean figure quoted in the tasking is not sourced from the record and I could not verify it**
  (I did not run the frozen reporter, per instruction). Treat it as unverified until §8 is filled in
  by `vsw_pregate_report.py`.

**The number that actually matters, and it is worse for the campaign than the tasking suggests:**
`VSW_PREGATE_RECORD.md:596-605` — with **full hindsight over 3 multiplier families × 48 λ values**,
the ceiling is **HateMM +0.0309 / MHC-ZH +0.0069 / MHC-EN +0.0164**, so *"K-VSW-1 is not merely unmet —
it is **arithmetically unreachable on ≥2 of 3 datasets for the entire declared operator space**."*

**One thing VSW *did* refute, and every future record must stop repeating the old law:**
`VSW_PREGATE_RECORD.md:942-947` — *"The sweep's hoped-for arithmetic closure is FALSE … The exchange
rate reaches **6.0000** on HateMM and exceeds 1.2 at every one of the 23 non-zero λ there, against
F95's best-in-36-cells of 1.1667. **Anyone citing 'the exchange rate never exceeds ~1.2' as a law of
this system must stop**; the correct law is §6.3's."* §2 below is built on §6.3's law.

### 0.2 The "tenth certified law-I datum" — the cand-2 curriculum train-LOO number — **does not exist in the repo.**

The tasking lists as new-today: *"the cand-2 curriculum itself (targeted train-LOO move of −0.0538
HateMM / −0.0402 ZH buying +0.0132 / 0.0000)"*. An exhaustive search (`refine-logs/**`, `autoresearch/**`,
`research-wiki/**`, `scripts/**`, `src/**`, and `git log --all -S`) finds:

* **No document anywhere contains a curriculum train-LOO contrast.** No `curric`-vs-`generic`
  train-side arm appears in `CAND2_CURRICULUM_RECON.md`, `CAND2_CURRICULUM_PREREG.md`, `CAND2_FREEZE.md`,
  `CAND2_SUBMIT_RECORD.md`, `CAND2_VERDICT_REVIEW.md`, `CAND2_REP2_*`, or any `*_OUT.json`.
* `−0.0538` and `−0.0402` co-occur in exactly one file, `refine-logs/SAV_F0_EXECUTION_RECORD.md:150,161`
  — an unrelated 2026-07-13 SAV confidence-interval table.
* The only curriculum-adjacent `0.0402` is `CAND2_CURRICULUM_RECON.md:224` — a **positive** ZH
  *generic*-LoRA-vs-CLIP **test** per-seed Δacc from B3, wrong object and wrong sign.
* The `+0.0132 / 0.0000` half **is** real: `findings.jsonl:59` pooled 2×3 mean **+0.01317** (5/6 signs)
  on HateMM, ZH **TIE** (`positives_bank[4]`, `directions_tried.json:516-521`).
* What *does* exist is a **pre-curriculum** frozen-Qwen train LOO used as a mining input only:
  `CAND2_KC20_HateMM.json` `loo_acc 0.8065`, `CAND2_KC20_MHC_zh.json` `loo_acc 0.7927`.

**Consequence.** The certified law-I count is **nine**, not ten (F91 explicitly reconciles the count at
8→9: `findings.jsonl:91`). More importantly: **the campaign's only training-side positive has never
been measured in the arena where every $0 pregate since F89 has been run.** That is not a gap in the
ledger's honesty — nobody claimed it — but it is the single most exploitable hole in the sweep, and it
becomes candidate **L4** below, at $0, on two banked caches that are already on disk.

---

## §1. THE ARITHMETIC EVERY CANDIDATE MUST DEFEAT (rebuilt from the four newest records)

The tasking asks me to filter candidates on the family's largest gold-cheating oracle. **That filter is
now close to vacuous here, and saying so is the most useful thing in this section.**
`AGGNET_PREGATE_RECORD.md:674-676`: *"**delivery is uncorrelated with ceiling** — and C3 is the datum
that establishes it."* The ceiling/delivery table, `AGGNET_PREGATE_RECORD.md:678-682`:

| family | oracle ceiling (HateMM / ZH / EN) | delivered |
|---|---|---|
| F94 global-k | +0.0145 (max per-seed oracle k over 6 arms) | −0.0140 … +0.0041 |
| F95/F97 adjudication gate | +0.0726 / +0.0535 / +0.0893 | +0.0269 / +0.0104 / +0.0182 |
| F98 conditional weighting | **+0.1492 / +0.1520 / +0.2186** | +0.0134 / −0.0069 / +0.0000 |

An operator with a 10× larger oracle delivered *less*. So I replace the oracle filter with four walls
that are actually binding. Every candidate in §3 is scored against all four.

### Wall 1 — the replacement law (the *correct* successor to the dead exchange-rate law)

`VSW_PREGATE_RECORD.md:753-775`: **`net = changed × (2·precision − 1)`**, where `precision` is the
fraction of changed decisions that are fixes. Exchange rate is a *ratio*; net is what buys accuracy,
and the two are anti-correlated across the sharpness continuum (HateMM: precision 0.8571 at 21 changed
→ 0.5696 at 79 changed; net pinned in **+11 … +21** over the whole curve).

Required net for +0.030 in the train arena, where the campaign's $0 pregates live
(n = 744 / 579 / 549, `MECHNOV_PAIRVERIFY_PREGATE.md:296-300`):

| | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| **net items required for +0.030** | **22.3** | **17.4** | **16.5** |

### Wall 2 — the measured precision-volume frontier, per dataset (re-derived, not transcribed)

Every operator this campaign has measured in that arena, converted to net items:

| operator (train arena, raw fused, 5-fold item-disjoint) | HateMM | MHC-ZH | MHC-EN | source |
|---|---|---|---|---|
| F97 `f47ctrl_full:gbm` adjudication gate | 36−16 = **+20** | +6.0 | +10.0 | `VGA_PREGATE_RECORD.md:293-297` |
| VSW `pow`, λ nested-selected | 36−17 = **+19** | 8−9 = **−1** | 22−21 = **+1** | `VSW_PREGATE_RECORD.md:565-575` |
| VSW, best over **all 3 families × 48 λ (hindsight)** | **+23** | **+4** | **+9** | `VSW_PREGATE_RECORD.md:596-600` |
| F98 `C3_net` learned aggregation profile | 22−12 = **+10** | 21−25 = **−4** | 1−1 = **0** | `AGGNET_PREGATE_RECORD.md:385-413` |
| F98 `FIXBEST_oracle` (**gold-cheating** profile selection) | 16−3 = **+13** | 24−15 = **+9** | 21−5 = **+16** | `AGGNET_PREGATE_RECORD.md:393,402,411` |
| F98 `THRESH_best` (bare threshold twin) | 38−24 = **+14** | −4 | −9 | `AGGNET_PREGATE_RECORD.md:385-413` |
| F95 verifier ungated (max aggregation) | −3 | −27 | −8 | `MECHNOV_PAIRVERIFY_PREGATE.md:380-387` |

**Best honest net: +20 / +6 / +10, against a requirement of 22.3 / 17.4 / 16.5 — i.e. 90 % / 35 % / 61 %.
Best net including gold-cheating within-family selection: +23 / +9 / +16 — 103 % / 52 % / 97 %.**

Read that carefully. **Even permitted to cheat inside the family, only ONE dataset crosses, and it
crosses by one item.** A ≥2-dataset pass requires roughly **tripling MHC-ZH's** or **+60 % on MHC-EN's**
*entire measured frontier*, not merely a better operator. This is the binding filter and it is the
number every candidate below is scored against.

### Wall 3 — the pure-permutation cap kills HateMM for set-preserving re-ordering, outright

`RDK_FORENSIC_RECON.md:130-174` derives, from `Σ w σ = 2·W(m) − 210` with `w = [20..1]`:

* a **hate** query is flippable by *any* permutation of the fixed weight vector **iff top-20 purity ≥ 6/20 = 0.30**;
* a **non-hate** query **iff purity ≥ 7/20 = 0.35** (`:146-150`);
* `:152-154` — *"**Below purity 6/20, no permutation of the retrieved list — and hence no re-metrication
  that preserves the set — can flip the prediction.**"*
* Crossed with measured purity (`:158-161`): **HateMM ≤ +0.0279, MHC-ZH ≤ +0.0470**, both under a
  *zero-break* assumption the campaign has never met.

**HateMM's cap is BELOW the +0.030 bar.** So: no operator that (i) preserves the retrieved set and
(ii) acts as a permutation of the deployed rank weights can ever pass on HateMM — the dataset that is
90 % of the way there on every other frontier. **Wall 3 is why "re-rank the neighbours better" is
finished, arithmetically, and not merely empirically.**

(Scope, stated because it is the crack §3 exploits: the cap binds *permutations of the fixed weights*.
A **free non-negative re-weighting** is more expressive — its oracle is Wall-1's +0.1492/+0.1520/+0.2186
— and a functional that is **not a weighted sum of signed labels at all** is not bounded by this
derivation. `RDK_FORENSIC_RECON.md:176-180` states the same limitation for the membership channel.)

### Wall 4 — one global hyperparameter is already too much selection on 2 of 3 datasets

This is **new with VSW and no prior sweep states it.** `VSW_PREGATE_RECORD.md:609-622`:

| dataset | nested-selected λ (deployable) | best fixed λ, pooled hindsight | fraction of ceiling kept |
|---|---|---|---|
| HateMM | +0.0255 | +0.0282 | **90 %** |
| MHC-ZH | −0.0017 | +0.0052 | **negative** |
| MHC-EN | +0.0018 | +0.0128 | **14 %** |

`:934-938` — *"a single global hyperparameter is already too much selection for this arena."*

**Design consequence, binding on everything in §3: a candidate with even one tuned scalar is
pre-priced to lose 86-100 % of whatever it finds on MHC-ZH and MHC-EN. Only parameter-free operators,
or operators whose every hyperparameter is fixed a priori and reported per-arm without selection, are
worth building.** No previous litsweep applied this filter; three of the last four candidates
(C1 τ/λ arms, C3 λ grid, VSW λ grid) violated it.

### The composite filter

A candidate is worth a pregate only if it satisfies **all five**:

* **F1** — not a permutation of the fixed rank weights (Wall 3 caps HateMM below bar).
* **F2** — its input is **not** the (cosine-to-query, label) profile of the deployed top-20.
  `AGGNET_PREGATE_RECORD.md:711-713` defines the closure by exactly that input, so this is the
  quotable boundary, not a rhetorical one.
* **F3** — parameter-free, or every hyperparameter frozen a priori and reported per-arm (Wall 4).
* **F4** — a *stated mechanism* by which MHC-ZH or MHC-EN could reach net +17 / +16, given that their
  entire measured frontiers are +6 / +10 (Wall 2). **This is where almost everything dies, and a
  candidate that cannot answer it should be killed at the desk.**
* **F5** — a $0 pregate on banked **train-split** artifacts only.

---

## §2. WHAT THE LEAD ASKED ABOUT VSW: is there a mechanism reason HateMM converts and ZH does not, and does it name a second dataset?

**Yes to the first, and the second answer is: it names MHC-EN, not MHC-ZH — and MHC-EN's remaining gap
is not mechanism-addressable.**

**Why HateMM converts.** Not because HateMM has more relational signal. It has *less*:
`VSW_PREGATE_RECORD.md:103` records the verifier's within-query AUC gain as
**+0.1572 (HateMM) / +0.2302 (ZH) / +0.1785 (EN)** — **ZH has the most relational signal and converts
the least**, and the record explicitly does not reconcile that inversion. The mechanism is **Wall 4**:
on HateMM the inner-fold λ selector keeps 90 % of the pooled-hindsight ceiling; on ZH it keeps a
negative fraction and on EN 14 % (`:609-622`). HateMM converts because it is the only dataset where
**selection over a single scalar is affordable at n = 744 with 5 folds**. ZH additionally collapses to
the deployed rule outright — `:791-796`: λ\* ∈ {0.25, 0.25, 0.25, **0**, 0.5}, DEG-B's arg-max is
**k = 20 = the deployed rule**, 17 changed items out of 579.

Secondary, and worth recording as the campaign's first decision-level relational win:
`:841-846` — on HateMM the trained relation score contributes **+0.0188 of the +0.0255** over a
like-for-like cosine twin, *"the first measurement in this campaign in which the verifier profile beats
a like-for-like cosine control at the decision level rather than the relation level"* — **but on MHC-EN
the cosine twin WINS (+0.0200 vs +0.0018), so the effect does not replicate.**

**Does the reason name a second dataset?** By Wall 2, the second-closest dataset is **MHC-EN** (net +10
honest, +16 gold-cheating, vs ZH's +6 / +9), not MHC-ZH — the tasking's framing had it the other way.
But EN is the dataset where `ERRPAT_MHC-EN_2026-07-26.md` measures **~41 % of the consensus error set as
non-group-harm Offensive**, i.e. a construct-validity boundary, and where the error set is only ~52 %
seed-invariant. **The 39 % of EN's requirement that is missing is, by the campaign's own forensics,
mostly not a model quantity.** So the honest answer to "does it bring a second dataset along" is **no**,
and it is no for a different reason on each of the two candidates.

**Does a modification bring one along?** The only modification Wall 4 licenses is **remove the
hyperparameter**. A parameter-free version of VSW would keep ZH's +0.0052 and EN's +0.0128 instead of
losing them to selection — i.e. net +3 and +7 against 17.4 and 16.5. **That is not a route; it is a
17 %/42 % recovery of a number that is already 4× short.** VSW's own routing (`:991-1019`) closes the
λ-interpolated monotone re-weighting family by measurement, and I concur: this specific asset is done.

---

## §3. RANKED CANDIDATES

Five survive my own filter; two more are listed because the ledger explicitly nominates them and a
sweep that silently dropped them would be dishonest. **Nothing here is sold. Four of the seven are
diagnostics whose value is a clean kill, and I say so in each entry.**

---

### L1 — **ATC: aggregate-then-compare, class-subspace reconstruction residual** ★ rank 1
*(= LITSWEEP6 membank C4, the one arm the ledger nominates and nobody has run)*

**What it changes about WHERE the gain lands.** Under the deployed vote, *all twenty* retrieved items
are decision-active with weights fixed in advance, and one correct analogue must **out-vote** nineteen
wrong-class neighbours — a contest it loses on the whole pathology population (purity 0.12-0.22).
ATC changes the functional: split the *same* top-20 by gold label, form a rank-`r` basis from each
class's members, and decide by which class's subspace **reconstructs the query with smaller residual**.
Under that functional a *single* correct analogue can decide by **spanning** the query, and the other
nineteen are decision-active only to the extent they add span in the query's direction. The
decision-active set becomes query-determined by geometry, with **no learned selection and no weights.**

**Literature (verified this sweep).**
* **SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling** — Camile Lendering,
  Erkut Akdag, Egor Bondarev. arXiv:**2602.23013**, v1 26 Feb 2026, v3 13 May 2026. **CVPR 2026**
  (stated). Code `https://github.com/CLendering/SubspaceAD` (stated). *Fetched and confirmed:* fits a
  PCA model to estimate the normal subspace and scores **via the reconstruction residual with respect
  to that subspace**; training-free, one-shot/few-shot; MVTec-AD 97.1 % image-level AUROC.
* **ProCon: Projection-Consistency Memory for Training-Free Anomaly Detection** — arXiv:**2607.04894**,
  6 Jul 2026. Verified by LITSWEEP6-membank §4(a) (fetched there; **I did not re-fetch** — flagged).
  Softly projects the query onto the span of retrieved memory vectors and uses the projection residual,
  with **median** aggregation across perturbed banks.
* **Regression Networks for Meta-Learning Few-Shot Classification** — arXiv:**1905.13613**. Classifies
  by *"finding the nearest class subspace by comparing regression errors"* — the exact functional,
  predating the AD line. *Surfaced by search, abstract read via search snippet only; not fetched.*
* **Learning to Compare: Relation Network** — Sung et al., CVPR 2018, arXiv:1711.06025 — supplies the
  composition order (aggregate the support set **first**, one relation per class).

**Pre-closure table.**

| ban / finding | binding text (file:line) | ruling |
|---|---|---|
| **F98 conditional-aggregation closure** | `AGGNET_PREGATE_RECORD.md:711-720`: *"The closure covers operators whose **input is the (cosine, label) profile of the deployed top-20**. It does **not** cover: … **LITSWEEP6 C4** (aggregate-then-compare subspace residual) — its input is the retrieved **vectors**, not their cosine/label profile, so it is **outside C3's function class by information content**, not merely by functional form. **C4's $0 pregate is untouched by anything measured here.**"* | **NOT CLOSED — explicitly.** Also nominated as next arm at `:728-730`. |
| **F96 residual-transport kill** | `RESTRANS_PREGATE_RECORD.md:443-446`: *"**Route next to C4 (aggregate-then-compare subspace residual)** … **C4's own $0 pregate is untouched by anything measured here.**"* | **NOT CLOSED — explicitly nominated.** |
| **F95 pair-verification ban** | `MECHNOV_PAIRVERIFY_PREGATE.md:472-479` bans (a) head-space pair verification, (b) other **pair-scorer architectures**, (c) verifier-as-reranker unpriced | **Does not fire.** ATC has no pair scorer, no verifier, no nomination — the candidate set is bit-identically the deployed top-20. |
| **F95 control-2b shape cost** | `:322-325`: shape cost **−0.0417 / −0.0293 / −0.0437** for "shortlist-per-class + max" | **Fires as a warning, not a ban.** ATC does abandon the rank-weighted average, so it starts in the same hole. This is the candidate's largest single liability and its bar 1 must be read against it. |
| **F89 eval-time operators** | `directions_tried.json` `dead[56]`: "eval-time symmetric vote/retrieval operators on deployed head keys: class-balanced quota, CSLS, LW whitening, 1-D excision"; new variants "need fresh recon + freeze" | **Does not fire on the letter.** A query-dependent subspace residual is not `cos(Az, Az')` for any fixed `A`. Fresh freeze required — supplied below. |
| **F94 k-sweep** | `dead[61]`: "k in [1,60] … CLOSED BOTH DIRECTIONS" | **Does not fire.** k = 20, unchanged. |
| **F99 RDK permutation cap** | `RDK_FORENSIC_RECON.md:152-154` | **Does not fire.** The cap is derived for `Σ w σ` with `w = [20..1]`; ATC's decision is not a weighted sum of signed labels, so it can flip items with purity 1/20 that no permutation can reach. **This is ATC's whole arithmetic case.** |
| **F66 selection lock** | `NCA_FORENSIC_RECON.md:106-110`: F66's arithmetic is *"conditional on a single fixed map φ₀"* and bounds **inference-side symmetric re-weighting** of a fixed Gram | **Partially fires.** ATC *is* inference-side on a fixed Gram — but F66's symmetric slice was measured for **non-selecting aggregations of a vote** (`ISR_PREGATE_RECORD.md:36-39`), not for a non-vote functional. Honest reading: F66 is evidence against, not a bound. |
| **banned_constraints[0-8]** | `directions_tried.json` | None fire. No OCR, no gold spans, no ensembles, no pseudo-labels, no target-as-structure, no MLLM scores, no P1-P5, no APIs, no cross-dataset data. |
| **D7 (novelty)** | `REDTEAM_BAN_SCOPE_AUDIT.md:305-306` via `RDK_FORENSIC_RECON.md:206-207` | **Live risk.** A decision-rule swap with no MLLM role can be read as "generic classifier tuning". The defensible framing is mechanism-first (it is the answer to a pathology this campaign *measured*), not architecture-first. Flag to the user, do not finesse. |

**Filter score:** F1 ✓ · F2 ✓ (quotable) · F3 ✓ if `r` is declared a priori · **F4 — unanswered, see risk** · F5 ✓.

**Largest oracle for the family.** Any operator over the deployed top-20 that can act on a *single*
neighbour has ceiling `AGGNET_PREGATE_RECORD.md:363-370`: **+0.1492 / +0.1520 / +0.2186** (class-mixed
top-20 on 86.8 / 82.9 / 97.1 % of items; **96-100 % of every deployed error reachable**). ATC's
candidate set is that same set, so this is its ceiling too. **Above bar on 3/3 — the candidate is not
self-killed by criterion 5.** But per §1, ceiling is not the binding filter here; Wall 2 is.

**$0 pregate (frozen bars, house style).** Same arena as F95/F96/F97/F98: `StratifiedKFold(5,
shuffle=True, random_state=0)`, item-disjoint, train split only, `mechfix_ops.py` (sha256
`635c1312…c83fc8d`) and `mechnov_pairverify.py` (sha256 `77b0defd…b7240d`) **imported unmodified and
sha-asserted**. Inputs: `data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_*.pt` + `data/gt/*/train.jsonl`.
Cost: **minutes, ≤8 CPU threads, $0, training-free** (a least-squares projection).

* **PARITY (mandatory, first):** reproduce the deployed floors **0.8441/0.8419 · 0.8480/0.8281 ·
  0.7796/0.7286** and the 116/88, 88/79, 121/109 wrong/pathology counts at 4 dp against
  `mechnov_pairverify_*_OUT.json`, ≥81/81 cells as F96 did. Miss ⇒ harness void, no number reported.
* **DEG-0 (degeneracy, fires a KILL, read BEFORE bar 1):** the distribution of
  `residual_0 − residual_1` must be non-degenerate at the declared `r` — require
  **IQR ≥ 0.01** in normalised residual units **and** `< 90 %` of items within 1e-3 of zero. In a
  cone-collapsed space (deployed top-1 cosine 0.9439-0.9686 raw, `MOLMO2` via
  `RESTRANS_PREGATE_RECORD.md:452-456`; 0.999852 in head space) both class spans may be near-universal.
  If DEG-0 fires the arm reports **"untested by degeneracy"**, never "aggregate-then-compare is dead".
* **DEG-A (threshold twin, ≥0.95 agreement ⇒ KILL):** pooled agreement of ATC's held-out predictions
  with `THRESH_best`. Form and threshold taken verbatim from `AGGNET_PREGATE_RECORD.md:245-247`.
* **DEG-B (fixed-k twin, ≥0.95 ⇒ KILL):** max agreement over the eight F94 `FIXK_k` profiles.
  Same source.
* **ISO (isomorphism control, mandatory — this is the bar that can refute ATC's own novelty claim):**
  re-run **F98's `C3_net`** (the best *free re-weighting* of the identical top-20, identical folds,
  identical budget) and require **ATC − C3_net ≥ +0.010 on ≥2 of 3 datasets**. If ATC does not beat
  the best re-weighting of the same set, then "reads the vectors, not the profile" bought nothing,
  ATC is inside F98's closure **by delivery**, and it dies regardless of its Δ. `C3_net`'s numbers are
  already banked (`aggnet_pregate_OUT.json`), so this control is free.
* **PERM (permutation null, 200 draws, `PERM_SEED = 12345`):** shuffle the bank labels within the
  fitting folds; `p = (1 + #{null ≥ obs}) / 201`. Require **p ≤ 0.01**. *And* report the null's
  **maximum**, because `AGGNET_PREGATE_RECORD.md:572-589` showed a null can give p = 0.0099 on an arm
  that changes 2 items — the null must be read together with the net.
* **CLASS BALANCE:** decision positive rate within **0.10** of the bank rate (0.4005 / 0.3109 / 0.3060).
  Outside ⇒ the nulls for that cell are **VOID** (`VSW_PREGATE_RECORD.md:291-307`).
* **K-ATC-1 (primary, decisive):** pooled item-disjoint **Δacc ≥ +0.030 on ≥2 of 3 datasets**, fold
  signs ≥4/5 on those datasets, **and** net ≥ Wall-1's 22.3 / 17.4 / 16.5. Miss ⇒ **KILL, the
  non-vote-functional axis closes.**
* **K-ATC-0 (interest only, licenses nothing):** Δacc ≥ +0.010 on ≥1 dataset, 5/5 fold signs ≥ 0,
  ≥3/5 strictly positive, **and** ATC must beat the −0.0293/−0.0437 shape-cost hole it starts in.
* **Arms declared a priori, reported per-arm, no selection (Wall 4):** `r ∈ {1, 2, 3, 5}` with a fixed
  ridge `δ = 1e-3`; **PRIMARY = r = 2**, declared before any number. Spaces: fused (PRIMARY), text, img.

**Cost.** Pregate **$0 CPU, ~1 hour**. Full version: training-free at inference, but a deployed-arena
verdict needs a head re-mint per dataset per seed — `BSY_FORENSIC_RECON.md:145-162` and F78 (6/6 floor
head ckpts deleted) make that **not** 0 GPU-h. Budget ~0.3-1 GPU-h *only after* K-ATC-1 passes.

**Honest probabilities.** P(clears DEG-0, i.e. the arm is even testable) ≈ **55 %** — cone collapse is
the named killer. P(clears K-ATC-0 on ≥1 dataset) ≈ **18 %**. P(clears the ISO control given K-ATC-0)
≈ **35 %**. **P(clears K-ATC-1, ≥2 datasets) ≈ 2-3 %.** P(reaches +0.030 acc AND mF1 on ≥2 datasets
3/3 seeds on test under final-epoch) ≈ **1 %**.
It is ranked first because it is the **only** live cell that (i) two independent kill records
explicitly nominate, (ii) escapes Wall 3's HateMM cap by construction, (iii) is quotably outside F98's
stated function class, and (iv) costs one afternoon of CPU and returns a decisive answer either way.
**F4 is unanswered:** I have no mechanism argument for why ATC would triple MHC-ZH's frontier, and the
honest expectation is a HateMM-only result that fails the ≥2-dataset conjunct exactly as VSW did.

---

### L2 — **CORRVOTE: deflate the vote by the retrieved neighbours' mutual redundancy** ★ rank 2

**What it changes about WHERE the gain lands.** The deployed vote treats twenty neighbours as twenty
independent pieces of evidence. ERRPAT says the pathology is a **tight, mutually-similar, wrong-class
cluster** out-voting one correct analogue sitting at median rank 1.5 (ZH). If those nineteen are
mutually redundant they are, informationally, close to **one** vote — and the correct analogue, which
is *dissimilar to them by construction* (different class), is the only independent observation in the
set. CORRVOTE re-weights by the **neighbour × neighbour** Gram, so the decision-active set is deflated
to its effective size. Nothing about the signal, the retrieval, the set, `k`, or the labels changes.

**Literature (verified this sweep).**
* **Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels** — Guneet
  Kohli, arXiv:**2605.29800**, 28 May 2026 (cs.CL; no venue stated). *Fetched and confirmed:* 9 frontier
  LLM judges supply *"only about 2 independent votes' worth of information"*; quantified via the
  **Kish effective sample size `n_eff`** and a Condorcet null. **The paper is also the strongest
  published warning against this candidate** and I quote it as such: *"Neither adding more judges nor
  using smarter aggregation algorithms helps — established methods close at most 11 % of this gap,
  even with access to the correct answers. … The bottleneck is correlated judges, not the aggregation
  algorithm."* Our case is not identical (we are not trying to *extract* more information, we are
  trying to stop a correlated bloc swamping one independent observation), but a candidate whose
  headline anchor says "smarter aggregation recovers ≤11 %" must be priced accordingly.
* **Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information** — Ai, Pan,
  Simchi-Levi, Tambe, Xu. arXiv:**2510.01499**, **ICML 2026**. Verified by LITSWEEP6-relgen §2 C4
  (fetched there; **not re-fetched here** — flagged). Optimal Weight / Inverse Surprising Popularity
  use **cross-voter correlation** and provably mitigate majority voting's limitations.
* **Principled and Scalable Diversity-Aware Retrieval via Cardinality-Constrained Binary Quadratic
  Programming** — Qiheng Lu, Nicholas D. Sidiropoulos. arXiv:**2604.02554**, 2 Apr 2026. *Fetched and
  confirmed:* diversity retrieval as CCBQP with an interpretable relevance/diversity trade-off,
  Frank-Wolfe algorithm with convergence guarantees. **Caveat: the abstract does not state an MMR/DPP
  comparison and no code link is on the page** — the search snippet claimed a Pareto-frontier win over
  MMR and DPP and I could **not** verify that from the abstract. Do not cite it for that.

**Pre-closure table.**

| ban / finding | binding text (file:line) | ruling |
|---|---|---|
| **F98 closure (a)** | `AGGNET_PREGATE_RECORD.md:693-697`: *"(a) any **learned** re-weighting, soft-mixture-over-k, attention, or gating **over the deployed top-20**"* — and the scope definition at `:711-713`: *"operators whose **input is the (cosine, label) profile of the deployed top-20**"* | **Arguably does not fire, and this is the candidate's load-bearing legal claim.** CORRVOTE is (i) **not learned** — a closed-form function; (ii) its input is the **bank×bank** Gram of the retrieved items, which is **not** in the (cosine-to-query, label) profile. Same escape clause, same sentence, as C4's. **Flag to the reviewer up front: this is one sentence away from a re-weighting ban and it should be pre-cleared, not argued after a positive.** |
| **F89 T2a CSLS / hubness** | `MECHFIX_PREGATE_2026-07-27.md:235`: hubness `r(x)` IQR **1.0e-4 – 8.8e-4**, inert | **Does not fire on the object** — `r(x)` is a global bank-item statistic over its own nearest *queries*; CORRVOTE's redundancy is within-neighbourhood and query-conditional. **But it names the exact failure mode**, and it is why bar 0 below exists. |
| **F94** | `dead[61]` | Does not fire; k = 20. |
| **F99 permutation cap** | `RDK_FORENSIC_RECON.md:152-154` | **Does not fire** — CORRVOTE is a *free* non-negative re-weighting, not a permutation of `[20..1]`; oracle is Wall-1's +0.1492/+0.1520/+0.2186. |
| **VSW routing** | `VSW_PREGATE_RECORD.md:995-999`: closed by measurement — *"any **λ-interpolated, monotone re-weighting of the deployed top-20 by the F95 pair-verifier score**"* | **Does not fire.** No verifier, no λ, no fitting. The *closest measured relative* is nevertheless VSW, and its whole-continuum net cap (+23/+4/+9) is the number CORRVOTE must beat. |
| **F63 label propagation** | `dead[36]` | Does not fire; one hop, no propagation between bank items, labels untouched. |
| **banned_constraints / D7** | — | None fire; D7 risk identical to L1. |

**Filter score:** F1 ✓ · F2 ✓ (arguable, quotable) · **F3 ✓✓ — parameter-free, the only candidate that
is** · **F4 — partially answerable**: the mechanism predicts the largest effect where within-top-20
redundancy is highest, which is *measurable at $0 before anything else runs* and is the one route by
which ZH/EN could differ from their re-weighting frontiers. · F5 ✓.

**Largest oracle:** same as L1, **+0.1492 / +0.1520 / +0.2186**, above bar 3/3.

**$0 pregate.** Same frozen arena and parity gate as L1. Two declared arms, **both parameter-free**:
* **A-GLS (PRIMARY):** `w ∝ (R + δI)^{-1} 1` over the 20 retrieved keys, `R` = their pairwise cosine
  matrix, `δ = 1e-3` fixed a priori, weights clipped at 0 and renormalised — the textbook
  correlated-observations weighting.
* **A-KISH (SECONDARY):** deflate each neighbour's deployed rank weight by
  `1 / (1 + Σ_{j≠i} max(0, cos(k_i, k_j)))` — Kish `n_eff`, zero parameters at all.
* **λ = 0 parity assert:** with `R = I` both arms must reproduce the deployed vote **bit-exactly**
  (the `PARITY-λ0 54/54` form used at `VSW_PREGATE_RECORD.md:484-490`).

* **BAR 0 (dynamic-range control — RUN FIRST, $0, ~5 minutes, can pre-kill the whole candidate):**
  per query, the IQR of the within-top-20 pairwise cosines. **If the median IQR < 0.01 the operator has
  no dynamic range and is inert by exactly F89-T2a's argument** (`MECHFIX_PREGATE_2026-07-27.md:235`),
  the arm is void, and no accuracy number is reported. **I estimate this fires with probability
  ~45-55 %** given the measured cone collapse — and if it fires, that single table is a clean,
  publishable closure of the redundancy axis for ~5 minutes of CPU.
* **DEG-A / DEG-B / CLASS BALANCE / PERM:** identical to L1's, verbatim thresholds.
* **ISO (refutes CORRVOTE's own novelty claim):** an arm using a **query-side** redundancy proxy only
  (each neighbour's cosine to the query, i.e. F98's own profile). If it matches A-GLS, the "bank×bank
  Gram is new information" claim — the entire licence to escape F98's closure — is **falsified by
  measurement** and the candidate dies regardless of net. This is the K-VGA-3 pattern
  (`VGA_PREGATE_RECORD.md:56-59`) applied to CORRVOTE's own escape clause; it must be declared, because
  the last candidate to invoke an escape clause without one had it falsified.
* **K-CV-1 (primary):** Δacc ≥ **+0.030 on ≥2 of 3**, fold signs ≥4/5, net ≥ 22.3/17.4/16.5. Miss ⇒ KILL.
* **K-CV-0 (interest):** Δacc ≥ +0.010 on ≥1, 5/5 fold signs ≥ 0, ≥3/5 strict, **and** net above VSW's
  whole-continuum ceiling on that dataset (+23 / +4 / +9) — otherwise it is inside a family already
  measured to its ceiling.

**Cost.** Bar 0 alone: **~5 min CPU**. Full pregate: **~1 hour CPU, $0**. No new parameters anywhere,
so the deployed-arena version is a re-evaluation, not a retrain — but the same head-re-mint caveat as
L1 applies to any floor-grade verdict.

**Honest probabilities.** P(bar 0 passes) ≈ **50 %**. P(K-CV-0 on ≥1 | bar 0 passes) ≈ **25 %**.
P(survives ISO | K-CV-0) ≈ **40 %**. **P(K-CV-1 on ≥2 datasets) ≈ 2 %.** P(goal bar on test) ≈ **1 %**.
Ranked second, not first, because F98's ban clause (a) is one sentence away from covering it and
because its own headline anchor (arXiv:2605.29800) reports that smarter aggregation over correlated
voters recovers ≤11 % of the gap **even with the answer key**. Its virtues are that it is the only
parameter-free candidate in the sweep, it has a 5-minute self-kill, and its mechanism is the exact
inverse image of the measured pathology.

---

### L3 — **XBANK: decouple the head-fit set from the memory bank** ★ rank 3

**What it changes about WHERE the gain lands — and this is a genuinely unexamined axis.** In this
pipeline the memory bank **is** the head's own training set. `directions_tried.json:171` (F47) records
the consequence: *"the RGCL head memorises train (**CLIP LOO train acc 0.998** vs Qwen 0.800)"*.
So **every bank key sits at a memorised location and every test query is a stranger**: the two sides of
the cosine are drawn from structurally different regimes of the same map. That is a covariate shift
**inside the retrieval geometry**, and no dead entry, no ban, and neither completeness audit (F61, F81)
has an axis for it. XBANK removes it: fit the head on a subset and use the *complement* as the bank, so
both sides of every comparison are out-of-sample with respect to the map. It changes which items are
decision-active by moving them, without adding a signal, a parameter, or a label.

**Literature.** The mechanism is **cross-fitting / sample splitting** (Chernozhukov et al., double
machine learning; Wolpert stacking) — textbook, and the right citation posture is "we import a standard
estimation-theory correction into retrieval-memory classification", not "we propose cross-fitting".
Two verified-by-search-snippet-only anchors on the retrieval side (**neither fetched** — flag):
**Great Memory, Shallow Reasoning: Limits of kNN-LMs** (Geng, Zhao, Rush, arXiv:**2408.11815**, NAACL
2025 short), which shows via **oracle retrieval** that kNN-LMs still fail with perfect retrieval; and
**On the Theoretical Limitations of Embedding-Based Retrieval** (arXiv:**2508.21038**, ICLR 2026),
which bounds the top-k sets representable at a given embedding dimension. **I found no 2024-2026 paper
that measures or corrects the train/bank memorisation asymmetry in a retrieval-augmented classifier.**
That is either a real gap or a search failure; ~6 targeted searches returned nothing, which is absence
of evidence and should be re-run with the `novelty-check` skill before any claim.

**Pre-closure table.**

| ban / finding | binding text (file:line) | ruling |
|---|---|---|
| **F78 memory-bank curation** | `dead[49]`: *"no curation GPU without user invoking the door-closer … W2-E prototype-select ban stands"* | **Does not fire.** Curation is *influence-based deletion of items*. XBANK deletes nothing and selects nothing; it changes which items the **map** was fitted on. Different object. |
| **F99 RDK key-space distillation** | `RDK_FORENSIC_RECON.md:208-212` bans a **trained map ψ** distilling the verifier's relation geometry; `:206-207` D7 fires on the claim | **Does not fire on the letter** — XBANK trains no new map class, uses the identical architecture/recipe/loss, and has no teacher. **D7 fires on the claim** exactly as for RDK: "adapt the head recipe only" reads as generic tuning unless framed as the memorisation-asymmetry correction it is. |
| **F66** | `NCA_FORENSIC_RECON.md:110`: *"**F66 does NOT bind trained-space reshaping.**"* | **Does not fire.** XBANK produces a different map, hence a different Gram, hence an object F66 never evaluated. |
| **F75 loss family** | `dead[47]` | Does not fire. Identical loss. |
| **F68 head-recipe family** | `dead[46]`: *"SAM … modality-dropout … Do not re-tune knobs (one-bite family consumed)"* | **Adjacent.** XBANK is not an optimizer or a regulariser knob; it is a data-partition protocol. But a strict reader will place it in the recipe family, and that should be pre-cleared. |
| **banned_constraints[2] ensembles** | — | Does not fire in the disjoint-split form (one head, one bank). **Fires** on any K-fold form that averages K heads — which is why the PRIMARY arm is the single disjoint split. |
| **banned_constraints[8]** | — | Does not fire; own train split only, no mixing. |

**Filter score:** F1 ✓ · F2 ✓ (outside every profile-based family) · F3 ✓ (one declared split fraction,
fixed a priori at 0.5) · **F4 — the only candidate with an argument**: the asymmetry is a property of
the *head*, not of a dataset covariate, so unlike CP1 (HateMM-only, ρ = +0.2842 / −0.1152 / −0.0050,
`RESTRANS_PREGATE_RECORD.md:379-383`) it should be present on all three datasets. · F5 — **partial**,
see cost.

**Largest oracle.** Unbounded by any of the four walls: XBANK changes membership *and* the map, so
neither Wall 3's permutation cap nor F98's family ceiling applies. **This is also its weakness — there
is no oracle to quote, so criterion 5 cannot be discharged in advance and the candidate is a genuine
bet rather than a priced one.**

**$0(-ish) pregate.** The **diagnostic half is strictly $0** and must run first, because it decides
whether the mechanism exists at all:
* **D-1 (asymmetry measurement, $0, no training):** under the deployed head keys, compare the
  distributions of (a) bank-item→bank-item top-1 cosine and (b) *held-out-fold* item→bank top-1 cosine,
  and the LOO vote accuracy in each regime. The pregates since F89 all run in **raw** encoder space
  where the deployed-equivalent LOO is 0.8441/0.8480/0.7796 — i.e. **no memorisation gap at all** —
  whereas F47 measures 0.998 in head space. **Quantifying that gap is the whole mechanism claim and it
  costs nothing.** If the gap is small, XBANK dies on the spot.
* **D-2 (matched-size control, mandatory):** any accuracy comparison must hold the bank size fixed —
  a full-head bank **subsampled** to the XBANK bank's size — so that "memorisation removed" is not
  confounded with "half the bank". Without D-2 the arm is uninterpretable.
* **Treatment arm** needs a head re-mint. `findings.jsonl:88` records *"HateMM head trains in 52 s on
  8 CPUs"*; if that holds for ZH/EN the whole thing is CPU-only. **It is recorded for HateMM only and
  I did not verify it for the other two** — and `BSY_FORENSIC_RECON.md:145-162` warns that the $0
  banked replay is **deletion-only**, so any key-space change needs a re-mint (F78: 6/6 floor head
  ckpts deleted). Budget: **$0 if the 52 s figure generalises, ≤0.3 GPU-h if not.**
* **Bars:** PARITY (D-2 matched-size floor reproduces the deployed floor within seed noise);
  **K-XB-1** Δacc ≥ +0.030 on ≥2 of 3 vs the **matched-size** floor, 3/3 seeds; **K-XB-0** ≥ +0.010 on
  ≥1 with 3/3 seed signs; **DEG-SIZE** — if the matched-size control alone explains ≥80 % of the Δ,
  KILL (the effect is bank size, not memorisation); PERM (shuffle which items go to the fit half);
  CLASS BALANCE ±0.10.

**Honest probabilities.** P(D-1 shows a large head-space asymmetry) ≈ **80 %** — F47's 0.998 vs 0.84
essentially guarantees it. P(K-XB-0 on ≥1 dataset against the *matched-size* floor) ≈ **15 %**.
**P(K-XB-1 on ≥2) ≈ 3 %.** The dominant risk is arithmetic and obvious: halving the fit set and the
bank are both large known negatives, and the memorisation correction must beat both. The dominant
*value* is that D-1 is a $0 measurement of a quantity nobody has ever measured, and it is the direct
answer to "is the campaign's whole pregate arena structurally different from the deployed one in the
dimension that matters" — see §5.

---

### L4 — **CURDIAG: measure what the curriculum did in the train arena** ★ rank 4 (diagnostic; the instrument-validity test)

**What it changes about where the gain lands: nothing — it *localises* the campaign's only
training-side gain.** Every $0 pregate since F89 measures operators in the raw banked train-split arena
under 5-fold LOO. Every one of them has been negative. Meanwhile the **one** lever that converted on
the training side — the cand-2 confusion-weighted SFT curriculum, `positives_bank[4]/[5]` — has
**never been measured in that arena** (§0.2). Nobody knows whether train-arena movement predicts test
movement, correlates with it, or is orthogonal to it. **If the correlation is ~0 or negative, the
campaign's "0-for-25" base rate is partly an artifact of the instrument, and every kill since F89 needs
a confidence discount.** That is a first-order claim about the whole programme and it costs one hour.

**It is now free, and I verified the inputs exist.** Both caches are on disk:

| dataset | generic-LoRA train cache | curriculum train cache |
|---|---|---|
| HateMM | `data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` (21 358 815 B, 2026-07-18 04:00) | `…-LoRA-curric_HF.pt` (21 358 864 B, 2026-07-18 12:26) |
| MHC-ZH | `data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | `…-LoRA-curric_HF.pt` |

Run the frozen `mechfix_ops.deployed_vote` under the frozen F95 fold protocol on **both** and report
the paired train-arena Δ. The test-side answer is already banked and needs no new measurement:
HateMM pooled **+0.01317** (5/6 signs, `findings.jsonl:59`), ZH **TIE** (`positives_bank[4]`).

**What it decides.** Three outcomes, all informative:
1. **Train-arena Δ is positive and ordered like the test Δ (HateMM > ZH ≈ 0).** The instrument is
   valid, the 0-for-25 base rate stands, and the closure argument in §5 is sound.
2. **Train-arena Δ is ~0 while test Δ is +0.0132.** The instrument is *insensitive* — it cannot see a
   real, replicated, deployed gain — and every $0 kill since F89 is a statement about the arena, not
   about the mechanism. **This is the outcome that would reopen the box**, and it would do so without
   any new mechanism at all.
3. **Train-arena Δ is negative while test Δ is positive** (what the tasking's unsourced datum asserts).
   The instrument is *anti-correlated*, and the campaign has been systematically killing the wrong
   things. This is the strongest possible reopening and also the one I consider least likely.

**Pre-closure.** No ban fires — this is a measurement of a banked positive, not a new operator.
`dead[?]`/queue note `directions_tried.json:534` says *"do NOT re-run curriculum variants (tactics)
without new structural premise"*; **CURDIAG runs no variant** — it re-reads two existing caches.

**Bars.** PARITY: the curric arm must reproduce F95/F96/F98's HateMM train floor **0.8441/0.8419**
exactly (those records were computed on the curric cache — `mechnov_pairverify_hatemm_OUT.json`
`meta.model = Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`), which makes this a free parity gate on the
harness. Then report paired Δ per dataset, 5/5 fold signs, fixed/broken, and the **rank correlation
between train-arena Δ and test Δ across every operator for which both exist**.

**Cost: $0, ~1 hour CPU.** **P(it changes how the campaign reads its own kills) ≈ 30 %.**
P(it produces a performance gain) = **0 %, by construction** — it is a diagnostic and must never be
dressed as anything else.

---

### L5 — **FLIPSET: are the four HateMM positives the same twenty items four times?** ★ rank 5 (diagnostic)

**The purest landing-site question the campaign can ask, and it has never been asked.** Four
independent operators produce a positive net on HateMM in the identical arena, on the identical folds:
`f47ctrl_full:gbm` **+20**, VSW `pow` **+19**, `THRESH_best` **+14**, `C3_net` **+10** (§1 Wall 2).
The requirement is **+22.3**. **If those flip sets are largely disjoint, the family is
operator-limited and its union clears the bar; if they are the same items, the family is
information-limited and closed by measurement rather than by argument.** Nobody knows which, and the
per-item substrate is banked: `vga_emit_{hatemm,zh,en}_OUT.json` (318/247/236 KB) carries per-item
gold, fold, deployed prediction, adjudicated predictions, `sc_rank`, and all three feature blocks
(`VGA_PREGATE_RECORD.md:460`), and `aggnet_*_OUT.json` / `vsw_*_OUT.json` carry the rest.

**Legality, stated plainly and not finessed.** As a **measurement** this is pure forensics and no ban
touches it. As a **lever**, combining the operators is squarely inside `AGGNET_PREGATE_RECORD.md:699-701`
clause (c) *"a bigger / better aggregator"* — and it would also need to answer Wall 4, since a
combination adds selection. **So: run the measurement, and treat any disjointness finding as a
premise for a new prereg with independent review, never as a promotion inside the diagnostic.**

**Bars.** Report, per dataset: pairwise Jaccard of the fixed sets and of the broken sets; the union net
and the intersection net; and — decisively — the **majority-of-four net**. **K-FS-1:** union net ≥ 22.3
on HateMM **and** ≥ 17.4 on ZH or ≥ 16.5 on EN ⇒ escalate to prereg. Anything less ⇒ the
re-weighting/gating family is **information-limited**, which is a materially stronger closure than the
four separate nulls, and §5's proof structure becomes complete.

**Cost: $0, ~1-2 hours CPU** (some per-item flip identities may need re-emission from the existing
frozen emitters; no verifier refit needed). **P(union clears on ≥2 datasets) ≈ 8 %** — ZH's four
operators net +6, −1, −4, −4, so even perfect disjointness gives ~+6, a third of requirement. **The
realistic value is the kill**, and it is the last piece §5 is missing.

---

### L6 — **QLB: query-local (anchor-local) bank densification** ★ rank 6 — ruling-gated, do not build yet

The one construction BSY's arithmetic explicitly does **not** bound.
`BSY_FORENSIC_RECON.md:277-281`: *"the **only** version worth writing is the **anchor-local** variant
(`LITSWEEP6_MEMBANK.md:296-299`) … because that is the only construction that is **not** query-agnostic
and therefore the only one §2's derivation does not bound. It would, however, then need to explain why
it is not per-item selection (Law III)."*

* **Why the query-agnostic version is dead:** `BSY_FORENSIC_RECON.md:73-95` — injected mass buys fixes
  and breaks in the ratio of the **local class odds**, which in the (0-1 word × hate) cell C2 targets
  are **0.1231** against C2's own ≥1.2 bar. Ten-fold under.
* **Two blockers before any prereg**, both from the same record: (i) `:236-258` — the open user ruling
  on `banned_constraints[3]`, reading **(A)** "the banned object is the pseudo-label" (C2 legal) vs
  **(B)** "the banned object is vote-pool expansion" (C2 and *every* future bank-addition candidate
  banned by name). **Under (B) no arithmetic matters.** (ii) `RESTRANS_PREGATE_RECORD.md:393-397` — the
  placement criterion **cannot** use `p̂`, and `BSY_FORENSIC_RECON.md:138` records that C2 currently has
  **no legal placement criterion at all**.
* **Filter score:** F1 ✓ · F2 ✓ (membership) · F3 ✗ (ρ, r, δ all need selection) · F4 ✗ (CP1 is
  HateMM-only: ρ = +0.2842 / **−0.1152 sign-inverted** / −0.0050) · F5 ✗ (`BSY:145-162` — additions
  need a head re-mint, the $0 replay is deletion-only).
* **Oracle:** reach oracle **+0.1492 / +0.1520 / +0.2186**, independently reproduced at
  `BSY_FORENSIC_RECON.md:53-56` (+0.1505 / +0.1520 / +0.2313). Above bar — **the candidate does not die
  of a small ceiling**, which is why it stays on the list.
* **P(goal bar) ≈ 1 %** (BSY's own estimate, `:287-291`, is <1 % for ≥2 datasets; 60-80 % that its
  near-duplicate control fires). **Recommendation: obtain ruling (A)/(B) first; do not spend a
  sweep-hour on it before that.**

---

### L7 — **VEA: the pair verifier as an evidence ranker for pillar ④** ★ rank 7 — not a performance candidate

Carried because it is the **one legal, unmeasured use** of the relational asset and three independent
records now say so in the same words: `MECHNOV_PAIRVERIFY_PREGATE.md` ban_scope (*"the pair verifier as
an EVIDENCE RANKER for the auditability pillar … **NEVER an accuracy claim**"*),
`VGA_PREGATE_RECORD.md:410-416`, `VSW_PREGATE_RECORD.md:1013-1016`. Headline numbers already exist
(median first same-class analogue rank 1.0 over all items vs 2.0-3.0 over the deployed vote's errors;
72-92 % of deployed errors in the pathology population, `MECHNOV_PAIRVERIFY_PREGATE.md:373-375`).
**It cannot satisfy the goal clause and must never be presented as if it could.** Listed so the sweep
is complete, not because it is a lever.

---

### Ranking summary

| # | candidate | landing site it attacks | F1 | F2 | F3 | F4 | F5 | family oracle | P(pregate) | **P(goal bar)** |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | **ATC** subspace residual | the **functional** the decision reads | ✓ | ✓ | ✓ | ✗ | ✓ | +0.149/+0.152/+0.219 | 18 % (K-ATC-0) | **1 %** |
| L2 | **CORRVOTE** redundancy deflation | how many items are **effectively** decision-active | ✓ | ~ | ✓✓ | ~ | ✓ | +0.149/+0.152/+0.219 | 25 %·50 % | **1 %** |
| L3 | **XBANK** fit/bank decoupling | the **conditioning** of the decision problem | ✓ | ✓ | ✓ | ✓ | ~ | unbounded/unpriced | 15 % (K-XB-0) | **3 %** |
| L4 | **CURDIAG** curriculum train-arena read | — (instrument validity) | — | — | — | — | ✓ | — | — | **0 % by construction** |
| L5 | **FLIPSET** disjointness of the 4 positives | — (is the family info- or operator-limited) | — | — | — | — | ✓ | — | 8 % (union clears) | **0 % by construction** |
| L6 | **QLB** anchor-local densification | **membership**, query-conditionally | ✓ | ✓ | ✗ | ✗ | ✗ | +0.149/+0.152/+0.219 | ruling-gated | **1 %** |
| L7 | **VEA** evidence ranking | — (pillar ④) | — | — | — | — | ✓ | n/a | high | **n/a — never an accuracy claim** |

**What I would actually run, in order:** CORRVOTE's **bar 0** (5 minutes; may pre-kill L2 outright) →
**CURDIAG** (1 hour; may invalidate the instrument that produced the last six kills) → **ATC** full
pregate (1 afternoon) → **FLIPSET** (1-2 hours) → **XBANK D-1** ($0 diagnostic only). Total ≈ one day
of ≤8-thread CPU, zero GPU, zero test contact. Freeze all bars in a prereg **before** any emitter
touches real data, per standing ceremony.

---

## §4. PRE-KILLS — candidates I considered and killed myself

Each of these was a live idea at the start of this sweep. All are killed at the desk, at $0, with
in-repo arithmetic. **These are the sweep's most reliable output.**

**PK-1 — Selective prediction / conformal risk control / learning-to-defer over this decision.**
LITSWEEP6-PARADIGM ranked this **R1, first of five**. It is now **dead by measurement**, killed the
same day it was proposed. `AGGNET_PREGATE_RECORD.md:697-698` bans *"(b) any per-item selector, router
or **adjudication gate** over the same neighbourhood, **with any feature family**"*, and
`VGA_PREGATE_RECORD.md:404-408` bans *"(b) verifier-based selective prediction / abstention / risk
ordering — K-VNQ-1/2 close it, and the free vote margin dominates it"*. The measurement:
`VGA_PREGATE_RECORD.md:331-352` — ΔAUGRC vs the kNN-UE baseline **−0.0029 / −0.0024 / −0.0052**, 0 of 3;
the **free** deployed vote margin (AUROC 0.7395/0.7927/0.7375) beats every fitted risk score. And at
full coverage it satisfies the goal clause **by construction, never**. **Do not re-propose R1.**

**PK-2 — Prior-matched / label-shift-corrected batch decision (Sinkhorn/OT/BBSE/EM over the test batch).**
Already pre-killed by LITSWEEP6-PARADIGM PK-1 and independently re-confirmed: for a 1-D score,
constraining the batch to a class prior **is** a quantile threshold, and thresholds are dead
(`MECHFIX_PREGATE_2026-07-27.md:241-242` T1 **identical predictions on 215/215 HateMM and 149/149 ZH**;
ZH test-fitted threshold oracle **+0.0201**, below bar). Also transductive ⇒ F63's ban scope rules it
out-of-box regardless. **Confirmed dead; do not spend.**

**PK-3 — Per-stratum decision conditioning (stratified thresholds / cell-wise calibration).**
**Killed by a gold-cheating oracle that is under the bar** — exactly the kill the tasking asked me to
perform on myself. `BSY_FORENSIC_RECON.md:186-197` measures the in-sample, gold-fitted stratum-shift
ceiling at C2's own granularity (4 length strata): **+0.0202 / +0.0207 / +0.0255 — 0 of 3 over bar**,
with zero generalisation cost assumed. Going to 8 strata reaches +0.0282/+0.0311/+0.0437 but at
68-71 items per bin, i.e. *"a stratification that would put 68-71 items per bin and overfit outright"*
(`:196-197`), and `:199-204` warns that a class-conditioned version degenerates into the label oracle.
**A family whose gold-cheating ceiling is below the bar is dead before any operator is written.**

**PK-4 — Matched / caliper retrieval on the nuisance covariate (retrieve neighbours matched on transcript volume).**
Self-killing on the campaign's own numbers. `BSY_FORENSIC_RECON.md:81-87`: in the 0-1-word band
`P(hate) = 0.1096`, **local class odds 0.1231**. Matching the query to same-stratum neighbours
*concentrates* the neighbourhood in precisely the band with the most adverse class odds — it makes the
confident inversion **more** confident. `:101-105` confirms the cell is already correctly reached
(short queries retrieve neighbours of median 1.0 word) and is ~89 % non-hate. And CP1 is HateMM-only
(ρ = +0.2842 / **−0.1152** / −0.0050, `RESTRANS_PREGATE_RECORD.md:379-383`), so on ZH the matching
would run the wrong way and on EN it would match on noise. **Dead on 3/3 for two independent reasons.**

**PK-5 — Full-bank class-conditional energy / log-density-ratio readout (soft vote over all n, temperature τ).**
The family **interpolates between two measured-dead endpoints**: as τ → 0 it becomes
"best neighbour per class", whose *shape cost alone* is **−0.0293 to −0.0437 before any scoring runs**
(`MECHNOV_PAIRVERIFY_PREGATE.md:322-325`); as τ → ∞ it becomes the class mean, i.e. a prototype, and
W2-E prototype memory is dead with the prototype-select ban standing. The interior is one global
scalar, and **Wall 4** measures a single global scalar to be already too much selection on 2 of 3
datasets (`VSW_PREGATE_RECORD.md:609-622`). *(Recorded honestly: F94 measured k > 20 **only on MHC-EN**
— the HateMM and ZH tables at `KSWEEP_RECORD.md:132-180` terminate at k = 20 — so `dead[61]`'s
"nor raising it for more evidence" is a **ban-scope letter-overreach** on two of three datasets. It
does not rescue this candidate: rank weights `[k..1]` already make the tail inert, 215/215 identical at
k=10 in 5 of 6 HateMM cells.)*

**PK-6 — Pseudo-relevance feedback / query expansion (Rocchio-style `q' = q + β·mean(top-m)`).**
Kills itself on ERRPAT. The top-20 is **78-88 % wrong-label** on the pathology population (purity
0.12-0.22), so moving the query toward its own neighbourhood centroid moves it **deeper into the
wrong-class cluster** — a self-reinforcing error, not a correction. In a cone-collapsed space
(top-1 cosine 0.9439-0.9686) the displacement is also near-zero. Adjacent to F63 (diffusion over the
frozen cosine graph, monotone-negative in α). **Do not spend.**

**PK-7 — Diversity-aware *set selection* (MMR/DPP/CCBQP choosing 20 from a top-60 pool).**
Distinct from L2, and killed for a reason L2 escapes: selecting a subset is a **0/1 re-weighting of a
retrieved list**, which is `AGGNET_PREGATE_RECORD.md:693-695` clause (a) in substance, and its cardinality
constraint plus trade-off parameter both need selection (Wall 4). More decisively, ERRPAT says the
correct analogue is already at median rank ~1.5 — **it does not need surfacing**, so a selection rule
that promotes it buys nothing the deployed ordering does not already have. L2 survives because it
changes the *weight given to correlated mass*, not the membership. **Dead as posed.**

**PK-8 — DRO / CVaR / worst-group aggregation of the head's training loss.**
Tempting because it is the cleanest "spend the model's capacity where the decision is contested" lever,
because head training is CPU-cheap (`findings.jsonl:88`: 52 s on 8 CPUs for HateMM), and because
**F66 provably does not bind it** (`NCA_FORENSIC_RECON.md:110`). Killed on three grounds:
(i) `dead[46]` head-recipe family — *"Do not re-tune knobs (**one-bite family consumed**)"*;
(ii) `dead[50]` prices the noise-robust/example-reweighting family by measured proxy —
*"boundary-dominated 13-17 % upper bound, **single-digit fixable** — new noise-robust proposals must
beat this arithmetic first"*, and single-digit is under Wall 1's 22.3/17.4/16.5;
(iii) `dead[47]` F75 is *"the first measured negative for trained-reshaping-unlocks-oracle-headroom"*,
0/8 formal. **D7 additionally kills the claim.** Recorded rather than dropped because it is the only
pre-kill I would reverse on a user ruling: if the user wants performance and will accept a D7-dead
lever, this is the cheapest untried training-side arm in the box.

**PK-9 — TabR-style difference-vector re-valuation of each neighbour.**
`LITSWEEP6_MEMBANK.md:729-737` already prices it out of budget (its grid runs 10k-1.2M objects, ~15×
our largest split) and its minimal form is *"re-value each neighbour"* = `AGGNET_PREGATE_RECORD.md:693-695`
clause (a). **Dead.**

**PK-10 — A fourth evaluation dataset (ImpliHateVid arXiv:2508.06570; Ex-HateMM / Ex-ImpliHateVid,
arXiv:2606.11953; HateClipSeg's frozen unconsumed 60/10/30 split).**
Not a mechanism, so out of scope for this sweep's charge — **but it is the one structural move that
neither F61 nor F81 could see, because both enumerated the *pipeline's decision surface* and none of
them enumerated the *evaluation target*.** `banned_constraints[8]` bans cross-dataset **training**
mixing, not evaluating a fourth dataset trained on its own split; the goal's "≥2 of 3" is a user-set
target and changing the denominator is a **user ruling, not an experiment**. Cost is GPU extraction
plus head training, so it has no $0 pregate. **Surfaced in §5, not ranked.**

---

## §5. COMPLETENESS-CRITIC PASS — what F61 and F81 could not see

My bar was to find something those two audits missed. **I found three things, and then I found the
proof structure that closes the box anyway. Both halves are reported.**

### 5.1 The three genuine gaps

**GAP-A (the biggest): nobody has validated the measuring instrument.**
Every $0 pregate since 2026-07-26 — F95, F96, F97, F98, VSW — runs in the **raw banked encoder key
space, train split, 5-fold LOO**. Every one is negative. Every one *also* states, first and against
itself, that the arena is not the deployed one (`MECHNOV_PAIRVERIFY_PREGATE.md:491-495`;
`VGA_PREGATE_RECORD.md:422-426`; `AGGNET_PREGATE_RECORD.md:744-751`; `VSW_PREGATE_RECORD.md:1025-1061`).
**The correlation between train-arena Δ and deployed-test Δ has never been measured, on any operator.**
It cannot be waved away by the "relative comparisons are less arena-sensitive" argument those records
make, because the arenas differ in a first-order way: F47 measures head-space LOO train accuracy at
**0.998** while the raw-space deployed-equivalent LOO in the pregate arena is **0.8441/0.8480/0.7796**.
Those are not the same object. F61 and F81 audited *cells*; neither audited the *instrument*, and the
instrument did not exist when F61 ran. **This is L4/CURDIAG and it costs $0 on two caches already on
disk.** Until it is run, "0-for-25" carries an unquantified instrument discount.

**GAP-B: the head's memorisation of its own memory bank is an unenumerated axis.**
F61's decision-surface table (`LITSWEEP5_COMPLETENESS.md:24-52`) has rows for frames, resolution,
prompt, layers, readout, fusion, loss, mining, memory-curation, memory-prototypes, vote, k, protocol,
encoder, LoRA recipe, vision-unfreeze, audio, OCR. **It has no row for the fit-set/bank coupling.**
F47 measured the symptom (`directions_tried.json:171`: "the RGCL head memorises train, CLIP LOO train
acc 0.998") and used it only to explain why a *train-supervised selector* is degenerate; nobody asked
what it does to the **retrieval geometry** the deployed decision runs on. This is L3/XBANK, and its
diagnostic half is $0.

**GAP-C: two ban-scope over-reaches that are load-bearing and contradict each other.**
1. **F66 is applied outside its own ruling.** `LITSWEEP5_COMPLETENESS.md:84` kills ArcFace with
   *"**F66 caps it** — ArcFace is a symmetric embedding-geometry operator … it can recover at most
   +0.001-0.006"*. But `NCA_FORENSIC_RECON.md:110` is a formal ruling in the opposite direction:
   ***"F66 does NOT bind trained-space reshaping. The cell is not F66-dead — it is legitimately
   un-measured."*** F99 applies the correct reading (`RDK_FORENSIC_RECON.md:47`: "RDK produces a new
   map ψ∘φ₀, hence a new Gram matrix. F66's arithmetic does not evaluate on it. **Clean.**"). The
   campaign therefore has **two incompatible readings of its most-cited arithmetic**, and the stricter
   one has been used to price at least one cell dead. *(It does not reopen ArcFace — F75's 0/8
   measurement does that job independently — but any future training-side proposal must be priced
   under the F99/NCA reading, not the LITSWEEP5 one.)*
2. **F49's q > 0.663 gate is MHC-EN-specific arithmetic promoted to a campaign constant.**
   `MJ_FORENSIC_RECON.md:42-63` derives it from **MHC-EN dev, N = 80, D ≈ 21, p_Q = 0.588, bar +0.020**
   via `gain(q) = 0.2625·q − 0.15415`. **There is no HateMM or MHC-ZH re-derivation anywhere in the
   repo**, yet `directions_tried.json:179` states it unconditionally: *"any future carve-out candidate
   must first show alignment > 0.663 from banked evidence"*. On a different dataset with a different
   disagreement rate and base rate the required q is a different number. **The gate should be
   re-derived per dataset before it is used to pre-kill anything again.**
3. *(Minor, recorded for completeness)* **F94's upward-k ban is measured on MHC-EN only.**
   `KSWEEP_RECORD.md:132-180` terminates at k = 20 for HateMM and ZH; only the EN tables run to k = 60.
   `dead[61]`'s *"nor raising it for more evidence"* is therefore a letter-overreach on 2 of 3 datasets.
   Priced ~0 by the rank-weight inertness argument (215/215 identical at k = 10), so it is a letter-gap,
   not a cell.

### 5.2 And here, honestly, is the proof structure that closes the enumeration anyway

I set out to break the closure and instead assembled it. The argument is now **arithmetic on measured
quantities**, not a base-rate appeal, and it is short:

1. **Every legal operator sits in exactly one of four channels**, and the taxonomy is exhaustive by
   construction: it either (a) changes the **representation**, (b) preserves the retrieved set and
   **re-orders/re-weights** it, (c) **changes the set** (membership), or (d) changes the
   **training** of the map.
2. **Channel (a) is closed by nine certified law-I data** — P3, S2S, W2-A, Router, FA, premise-d, LP,
   vision-unfreeze, Molmo2 (count reconciled at `findings.jsonl:91`), the cleanest being F91 where the
   image stream genuinely improved by **+0.0558**, the best ever on HateMM, and the deployed number
   *fell*.
3. **Channel (b) splits, and both halves are capped.** *Permutation of the fixed weights* caps at
   **+0.0279 HateMM / +0.0470 ZH** under a zero-break assumption never met — **HateMM below bar**
   (`RDK_FORENSIC_RECON.md:158-174`). *Free re-weighting* has a huge ceiling (+0.1492/+0.1520/+0.2186)
   but its **measured precision-volume frontier** across four independent operators tops out at net
   **+20 / +6 / +10** against a requirement of **22.3 / 17.4 / 16.5**, and **+23 / +9 / +16** even when
   permitted to cheat inside the family (§1 Wall 2). One dataset, by one item, with the answer key.
4. **Channel (c) is closed for every query-agnostic construction** by the local-class-odds identity
   (`BSY_FORENSIC_RECON.md:73-95`: injected mass buys at the local odds, **0.1231** in the target cell
   against a 1.2 bar), for sub-video units three times over (`EUM_FORENSIC_RECON.md:183-188`), and for
   depth by F94. The residue is the anchor-local variant — **query-conditional, hence Law-III-exposed,
   and blocked behind an unresolved ban ruling.**
5. **Channel (d) is closed empirically** (F75 0/8, F68 head-recipe one-bite, F53/B3 LoRA already
   banked as the campaign's largest positive) though **not arithmetically** (GAP-C1). Its one live
   positive, the curriculum, is +0.0132 pooled on one dataset with ZH tied.
6. **And the cross-channel wall:** `AGGNET_PREGATE_RECORD.md:674-676` — **"delivery is uncorrelated
   with ceiling"**. So enlarging the oracle is not a strategy, and the four measured frontiers in
   step 3 are the operative bound.
7. **Finally, the ≥2-dataset conjunct is what actually kills everything.** Every positive this
   campaign has produced since F53 lands on **HateMM**. MHC-EN is dead at five levels with **~41 % of
   its residual measured as label semantics**; MHC-ZH's binding wall is **78-item dev selection noise**,
   which is a **protocol ruling**, not an operator — and Wall 4 now shows ZH cannot even afford one
   selected scalar. **There is no second dataset for a mechanism to land on.**

**Two clean ways to falsify this proof, both $0 and both in §3.** **L5/FLIPSET** falsifies step 3 if
the four HateMM flip sets turn out disjoint (union ≥ +22.3). **L4/CURDIAG** falsifies steps 3-6
wholesale if the train arena is measured insensitive to a gain the test split saw. Until one of them
fires, my verdict is:

> **The enumeration is now closed for CONVERSION, and the closure is arithmetic rather than a base
> rate. Three genuine gaps survive an adversarial read — the pregate instrument has never been
> validated (GAP-A), the head/bank memorisation coupling has never been enumerated (GAP-B), and two
> load-bearing bans are applied outside their derivations (GAP-C) — but none of the three is a lever:
> two are diagnostics and one (XBANK) is an unpriced bet whose natural operator halves both the fit
> set and the bank. The remaining upside is where LITSWEEP-5 left it a week ago and every measurement
> since has narrowed rather than widened it: user rulings (ZH val-selection retirement; the
> `banned_constraints[3]` (A)/(B) reading; whether a fourth evaluation dataset counts toward "≥2 of 3";
> whether a D7-dead performance-only lever is acceptable), not operators.**

---

## §6. LIMITATIONS OF THIS SWEEP

1. **No measurement of any kind was performed.** Every number here is either transcribed with a
   file:line or re-derived by arithmetic from transcribed numbers (net = fixed − broken; net = Δacc × n).
   I ran no script and opened no dataset file. The five pregates exist because these arguments are not
   evidence.
2. **The VSW permutation p-values are unresolved** (§0.1). `VSW_PREGATE_RECORD.md:889-896` is a
   placeholder; the per-draw JSONs exist and the frozen reporter has not been run. Any statement about
   VSW's significance — including the tasking's — is currently unsourced. **Someone should run
   `vsw_pregate_report.py` and fill in §8 before VSW is cited anywhere.**
3. **A campaign-level reproducibility erratum is open and I did not audit it.**
   `VSW_PREGATE_RECORD.md:514-556` reports that the F95-frozen torch-fitted MLP arm *"fails to
   reproduce … on 44 of 48 trained quantities"* (HateMM `acc_mlp_max` 0.8401 → 0.8468; exchange rate
   0.9474 → 1.0377), so **F97's "78/78 parity" claim was true on 2026-07-27 and would fail today on
   the 15 trained cells.** Several numbers I quote from F95/F97 are on the affected side of that
   erratum. They are used here for *ordering* arguments, not to 4dp, but the record needs a
   determinism fix before any of them enters a paper.
4. **Citation verification is uneven.** Fetched and confirmed this sweep: arXiv:2602.23013 (SubspaceAD,
   CVPR 2026, code stated), arXiv:2604.02554 (CCBQP diversity retrieval — **and its MMR/DPP comparison
   claim is NOT verifiable from the abstract; do not cite it for that**), arXiv:2605.29800 (Nine Judges,
   Two Effective Votes). **Relayed from LITSWEEP-6's verification log, not re-fetched here:**
   arXiv:2607.04894 (ProCon), arXiv:2510.01499 (ICML 2026 OW/ISP), arXiv:1711.06025 (RelationNet).
   **Surfaced by search only, abstract not fetched:** arXiv:1905.13613, arXiv:2408.11815,
   arXiv:2508.21038, arXiv:2508.06570, arXiv:2606.11953. Re-verify before any prereg or paper.
5. **The "no prior work on the train/bank memorisation asymmetry" claim (L3) is absence of evidence
   across ~6 searches**, not a systematic review. Re-run with the `novelty-check` skill.
6. **P-estimates are calibrated judgement, not statistics.** My priors are anchored on the campaign's
   own base rate (0-for-25 post-F61 door-closers) and on the fact that the last four sweeps' rank-1
   candidates all died at or before their own second bar.
7. **I did not read `scripts/analysis/vsw_*` source or run anything in that path**, per instruction, so
   all VSW statements come from the record file.
8. **No `state/` file, prereg, config, frozen artifact, or `research-wiki/` document was mutated.**
   Nothing was written outside this file. No commit was made.

---

## §7. SOURCES

**Fetched and confirmed during this sweep** (title/authors/date/venue checked against the arXiv page):

- Lendering, Akdag, Bondarev. *SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace
  Modeling.* **CVPR 2026**, arXiv:2602.23013 (v1 26 Feb 2026, v3 13 May 2026). Code:
  `https://github.com/CLendering/SubspaceAD` (stated). https://arxiv.org/abs/2602.23013
- Lu, Sidiropoulos. *Principled and Scalable Diversity-Aware Retrieval via Cardinality-Constrained
  Binary Quadratic Programming.* arXiv:2604.02554, 2 Apr 2026. **No venue stated, no code stated;
  MMR/DPP comparison NOT confirmed from the abstract.** https://arxiv.org/abs/2604.02554
- Kohli. *Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels.*
  arXiv:2605.29800, 28 May 2026 (cs.CL). **No venue stated.** https://arxiv.org/abs/2605.29800

**Relayed from LITSWEEP-6's verification log; NOT re-fetched here — re-verify before use:**

- Chae et al. *ProCon: Projection-Consistency Memory for Training-Free Anomaly Detection.*
  arXiv:2607.04894, 6 Jul 2026 (preprint, code stated).
- Ai, Pan, Simchi-Levi, Tambe, Xu. *Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order
  Information.* **ICML 2026**, arXiv:2510.01499.
- Sung, Yang, Zhang, Xiang, Torr, Hospedales. *Learning to Compare: Relation Network for Few-Shot
  Learning.* **CVPR 2018**, arXiv:1711.06025.

**Surfaced by search only, abstract NOT fetched — do not cite without verification:**

- *Regression Networks for Meta-Learning Few-Shot Classification.* arXiv:1905.13613.
- Geng, Zhao, Rush. *Great Memory, Shallow Reasoning: Limits of kNN-LMs.* NAACL 2025 short,
  arXiv:2408.11815.
- *On the Theoretical Limitations of Embedding-Based Retrieval.* ICLR 2026, arXiv:2508.21038.
- *ImpliHateVid: A Benchmark Dataset and Two-stage Contrastive Learning Framework for Implicit Hate
  Speech Detection in Videos.* arXiv:2508.06570.
- *Decoding Multimodal Cues: Unveiling the Implicit Meaning Behind Hateful Videos* (Ex-HateMM,
  Ex-ImpliHateVid). arXiv:2606.11953, Jun 2026.
- Chernozhukov et al., *Double/Debiased Machine Learning* (cross-fitting) — classical, cited from
  knowledge, not fetched.

**In-repo sources (all read this sweep, all read-only):**
`autoresearch/goal_mllm_plus3/state/{directions_tried.json, findings.jsonl, progress.json}` ·
`refine-logs/{LITSWEEP6_RELGEN, LITSWEEP6_PARADIGM, LITSWEEP6_MEMBANK, LITSWEEP5_COMPLETENESS}.md` ·
`refine-logs/{MECHFIX_PREGATE_2026-07-27, KSWEEP_RECORD, MECHNOV_PAIRVERIFY_PREGATE,
RESTRANS_PREGATE_RECORD, VGA_PREGATE_RECORD, AGGNET_PREGATE_RECORD, VSW_PREGATE_RECORD}.md` ·
`refine-logs/{RDK, EUM, BSY, TVB, MJ, NCA, C5}_FORENSIC_RECON.md` ·
`refine-logs/{ISR_PREGATE_RECORD.md, ISR_PREGATE_OUT.json}` ·
`refine-logs/CAND2_*` · `refine-logs/ERRPAT_*` (prose only; no test artifact opened) ·
`data/CLIP_Embedding/*/` directory listings only (no `.pt` loaded).

**Required statements.** ZERO GPU / SLURM / Modal / training spent. No held-out test metric read or
produced; no test-split file opened. No `state/`, prereg, config, frozen artifact or `research-wiki/`
document mutated. Nothing committed.
