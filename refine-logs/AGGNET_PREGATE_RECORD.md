# AGGNET PREGATE — C3, the learned aggregation profile network

**Date:** 2026-07-27 NZST · **Agent:** aggnet-pregate · **Cost: $0** (CPU only, ≤8 threads,
**zero GPU, zero SLURM, zero Modal, zero test-split contact**). Repo sha at freeze time `db2eae8`
(working tree dirty). Env: conda `HateVideo`, numpy 1.26.4, scipy 1.17.1, scikit-learn 1.5.2,
torch 2.6.0+cu124 (CPU), faiss.

**Test-split contact: NONE.** The only files opened are `data/CLIP_Embedding/<DS>/train_<model>.pt`
and (for the SECONDARY Δlength profile arm only) `data/gt/<DS>/train.jsonl`. `dev_seen` and
`test_seen` are never loaded by any script in this record.

**Binding design source:** `refine-logs/LITSWEEP6_MEMBANK.md` §3(a)-(f), read in full before any
code was written. Its §3(e) frozen bars are quoted **verbatim** in §2 below, before any number in
this document was computed. **Context records read in full first:**
`refine-logs/VGA_PREGATE_RECORD.md` (F47-family gate = the +0.0269 benchmark; verifier features
measured dead) and `refine-logs/RESTRANS_PREGATE_RECORD.md` (the label-field correction's collapse
to a threshold shift = the degeneracy this record must rule out for a *weighting* correction).

---

## §0. WHAT IS UNDER TEST, AND THE THREE FACTS THAT SHAPE IT

The deployed decision (`src/utils/metrics.py:262-301`, `src/model/evaluate_rac.py:405-465`, replayed
by the F89-frozen `mechfix_ops.deployed_vote`) is

```
v = Σ_i (2·lab_i − 1)·cos_i·w_i / Σ_i w_i ,   top-20 own-train neighbours,  w = [20,19,…,1]
predict 1 iff v ≥ 0
```

C3 replaces the **fixed scalar weight vector** `w` by a **learned function of the query's own
neighbourhood configuration**:

```
v = Σ_i s_i·cos_i·g_θ(profile)_i / Σ_i g_θ(profile)_i ,   s_i = 2·lab_i − 1,  g_θ ≥ 0
predict 1 iff v ≥ 0
```

Retrieval, the key space, `k = 20`, the candidate set, the threshold and the label field are all
untouched. **Only the weighting changes, and it changes per query.** Per the tasking, the plain
`s_i` summand is used — **not** the `r_i` residual composition of LITSWEEP6 §3(d), because C1
(residual transport) was measured dead and closed (`RESTRANS_PREGATE_RECORD.md` §7).

**Three inherited facts fix the design and are not re-litigated here.**

1. **F94 closed the global-`k` axis in both directions** (`KSWEEP_RECORD.md`): `k ≤ 3` is
   algebraically 1-NN and costs −0.0157 to −0.0388; ranks 11-20 flip zero predictions; the plateau
   starts at k≈10-15 and k=20 sits on it; dev-selected k is deployment-legally negative. C3's entire
   licence is therefore **conditional, non-monotone** weighting that no single global k can express.
   That licence is the load-bearing distinctness claim and §2's bars are built to attack it.
2. **The F47 feature family carries a real, permutation-validated conditional signal**
   (`VGA_PREGATE_RECORD.md` §4.3): +0.0269 HateMM (p = 0.0050, 5/5 folds), +0.0104 ZH (p = 0.0050),
   +0.0182 EN (p = 0.0100) as an adjudication gate — below the +0.030 bar. **The verifier-profile
   features are measured dead** as a signal source (lost to the F47 features 3/3, indistinguishable
   from the label-shuffled null). Consequence, binding on this record: **no verifier feature appears
   anywhere in C3's profile.** The profile is the F47-family / local-configuration object only.
3. **A correction that collapses to a global threshold move is dead by the existing ban**
   (`RESTRANS_PREGATE_RECORD.md` §5.3: C1 agreed with a pure threshold shift on 95.03 / 97.75 /
   99.45 % of items and was killed on that alone, *regardless of its Δ*). C3 must discharge the same
   duty in the form appropriate to a *weighting* correction, and must additionally discharge the
   F94 version of it — see DEG-A and DEG-B in §2.3.

**The distinctness duty of this record, stated plainly.** C3 is interesting only if the learned
weighting is (a) not a threshold shift, (b) not any single fixed `k`, and (c) worth more than the
+0.0269 the cheap F47 gate already banked. Bars that only compare C3 to the deployed `[20..1]` would
be uninformative, and are not the bars this record is read on.

---

## §1. FROZEN HARNESS

### 1.1 Arena — the F95 harness verbatim

LITSWEEP6 §3(e): *"Same 5-fold item-disjoint harness; θ fitted on fitting-fold profiles only.
549-744 LOO examples of a ~60-d → 1 problem. $0, CPU, minutes."*

Banked **RAW encoder key spaces** (seed-independent), **train split only**, item-disjoint
`StratifiedKFold(5, shuffle=True, random_state=0)`, **PRIMARY space = fused**, `text`/`img`
SECONDARY. This is the arena `mechnov_pairverify.py` (F95) froze and the one both sibling pregates
used. The trained RGCL head is **not** the arena: it memorises its own train split (LOO train acc
0.998, F47), so a train-side screen in head space measures memorisation. No head-space read is run;
the limitation is recorded in §8. **[ERRATUM — see the appended ERRATUM at the end of this record.
0.998 is F47's CLIP head; the deployed Qwen heads measure 0.9406 / 0.8915 / 0.8154. Downgraded, not
vacated; the raw-space justification is superseded by F113's unsaturated fold-head arena.]**

Consequence, stated up front: this arena has **no seeds** (raw encoder features are
seed-independent), so sign evidence is **per fold**, not per seed — as in both sibling records. A
network-initialisation seed does exist and is handled separately (§1.4).

### 1.2 Parity gate — 81 cells, train-side, abort on any mismatch

Identical to `RESTRANS_PREGATE_RECORD.md` §1.5 and for the same reason (the MECHFIX 15/15
floor-parity cells are *test* reads, and LITSWEEP6 binds this pregate to train only):

1. `sha256(mechfix_ops.py)` must equal `635c1312…c83fc8d` — the F89-frozen file carrying the 15/15
   test-side floor parity of MECHFIX §2.4.
2. The deployed floor recomputed inside this harness must reproduce **F95's recorded train-side
   numbers at 4 dp, per cell**, read from `mechnov_pairverify_{hatemm,zh,en}_OUT.json`: pooled
   `acc_deployed` + `mF1_deployed`, all five per-fold `acc_deployed`, and the integer counts
   `n_deployed_wrong`, `n_pathology_pop` — for every dataset × every space =
   **9 + 9 + 45 + 18 = 81 asserted cells.**

**IMPL gate (asserted every fold, every space).** The C3 vote engine evaluated at the constant
profile `g = [20..1]` must equal `mechfix_ops.deployed_vote` **bit-for-bit** in votes, predictions
and the retrieved index matrix. This proves the treatment differs from the floor **only** in the
weighting.

### 1.3 The profile — declared in full, no verifier feature anywhere

Per LITSWEEP6 §3(d) (`[cos_1..cos_20 ; s_1..s_20 ; purity-prefix counts ; (optionally) Δlength]`)
and per the tasking's binding restriction to F47-family / local-configuration features:

| block | dims | definition |
|---|---|---|
| `cos_1..20` | 20 | the top-20 cosines in deployed rank order |
| `s_1..20` | 20 | signed neighbour labels `2·lab_i − 1` in deployed rank order |
| `pfx_1..20` | 20 | purity-prefix: `pfx_i = (1/i)·Σ_{j≤i} lab_j`, the binary-label form of Meta-k's prefix label-count profile |
| **PRIMARY total** | **60** | |
| `dlen_1..20` | +20 | **SECONDARY arm only:** `|log1p(vol_q) − log1p(vol_i)|` per neighbour, `vol` = the F89-T3 frozen transcript-volume scalar (words for HateMM/MHC-EN, characters for MHC-ZH) |

The rank-1 margin (`cos_1 − cos_2`), the vote itself, purity and the label ratio are all
**deterministic functions of this profile** and are therefore inside the function class without
being listed separately. Δlength is declared SECONDARY, not PRIMARY, because
`RESTRANS_PREGATE_RECORD.md` §6 measured that the length covariate carries conditional information
on **HateMM only** (Spearman ρ +0.2842) and is **sign-inverted** on MHC-ZH (−0.1152) and **null** on
MHC-EN (−0.0050, p = 0.906); importing it as a primary input would import a covariate measured dead
on two of three datasets.

Profiles are standardised on the **fitting-fold rows only**.

### 1.4 g_θ — the network, frozen

`Linear(d, 16) → tanh → Linear(16, 20) → softplus(·) + 1e-6`.
Parameters: **1316** at d = 60 (PRIMARY), 1636 at d = 80 (`+dlen`). Inside LITSWEEP6's declared
~1-3k budget.

**Deployed-anchored initialisation (frozen, and load-bearing).** The output layer's weight matrix is
initialised to **zero** and its bias to `softplus⁻¹(w_i / w_1)` with `w = [20..1]`, so that **at
initialisation `g_θ` is exactly the deployed profile and the arm is bit-identical to the floor.**
The network can therefore only move away from the deployed rule if the fitting-fold data pushes it
there, and "Δacc = 0 at epoch 0" is an available sanity check. This removes an otherwise arbitrary
starting point (a zero-initialised softplus head starts at *uniform*, which is a different — and
already-measured — member of the fixed-profile family).

**Scale invariance, and why it matters.** `v` is invariant to any positive rescaling of `g`, so the
network cannot manufacture confidence by inflating magnitudes; the only thing it can change is the
**relative** weighting. Training therefore cannot degenerate into a magnitude hack.

Fit: full-batch **Adam**, `lr = 1e-2`, **300 epochs**, no early stopping on any held-out quantity,
loss `BCEWithLogits(v, y)` with the logit taken as `v` itself — the deployed convention, since the
deployed decision is `sigmoid(v) ≥ 0.5 ⟺ v ≥ 0`. `NET_SEED = 0` is PRIMARY (the F95 `MLP_SEED = 0`
precedent); seeds 1 and 2 are run as a **stability read only** and cannot carry a bar.

**Shrinkage toward the deployed rule, and nested selection of its strength.** The objective carries
an explicit penalty anchored on the deployed profile:

```
loss = BCEWithLogits(v, y) + λ · ( ‖l2.weight‖² + ‖l2.bias − b_deployed‖² )
```

Both penalised terms sit at their deployed-anchored values at initialisation, so **λ → ∞ returns
exactly the deployed rule `[20..1]`** and **λ → 0 returns the free conditional network**. (Naive
`weight_decay` on the output bias would pull the profile toward *uniform* — a different, already
measured member of the fixed family — and would make "fall back to the floor" unreachable.)
`λ` is selected by an **inner `StratifiedKFold(5, shuffle=True, random_state=17)` inside the fitting
pool** over the frozen grid `λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}`, refitting on the whole pool
at the chosen `λ`; ties break toward the **largest** `λ`, i.e. toward the deployed rule. The
held-out fold never sees a fit and never chooses `λ` (the VGA §2.4 nesting precedent).

**Why this is in the freeze rather than a later repair: §2.6 measured, on synthetic data before the
freeze, that without it the harness is structurally incapable of returning an honest null** — an
unregularised 1316-parameter net fitted on 560 examples destroyed a deployed-optimal rule by
**−0.1714**. A negative result from that harness would have been uninterpretable.

**Training set construction.** For each fitting-fold item `j`, the top-20 are retrieved from the
fitting fold **excluding `j` itself** (`exclude_self=True`, the same leave-one-out construction the
sibling record used for its D1 control); the profile is built from those and the target is `lab[j]`.
Held-out items retrieve from the full fitting-fold bank — the identical call that produces the
floor. **No held-out item ever enters a fit, as a query or as a bank row of its own fold.**

### 1.5 Arms

| id | status | what |
|---|---|---|
| `C3_net` | **PRIMARY** | the profile network above, 60-d profile, seed 0 |
| `C3_net_s1`, `C3_net_s2` | stability | seeds 1, 2 — reported, cannot carry a bar |
| `C3_net_dlen` | SECONDARY | 80-d profile (+Δlength) |
| `C3_metak` | SECONDARY / mechanism | the **Meta-k form**: same trunk, output = `softmax` over the 8 F94 k-profiles, `g = Σ_k π_k(profile)·ŵ^{(k)}`. A convex mixture of monotone profiles is **monotone**, so this arm is C3 *restricted to F94's family*, per-query. `C3_net` − `C3_metak` is the cleanest available price of non-monotonicity. Output bias initialised to `8.0` on the k = 20 component and `0` elsewhere, so the initial mixture is 0.9997 on the deployed profile |
| `FIXBEST_mono` | **BAR 2** | best **fixed monotone** profile selected on the fitting folds (§1.6) |
| `FIXBEST_oracle` | ORACLE ceiling | same family selected on the **held-out** fold — reported as a ceiling only, never a pass |
| `THRESH_best` | **DEG-A** | deployed vote with a **global threshold** τ chosen on the fitting folds — the RESTRANS §5.3 degeneracy twin |
| `FIXK_{1,2,3,5,7,10,15,20}` | **DEG-B** | the eight F94 grid profiles `[k..1, 0…]` |
| `DIRECT_logit` | **DEG-C** | L2 logistic regression on the *same standardised profile* → gold label. The unconstrained readout. If C3 ≈ DIRECT, C3 is a profile classifier wearing an aggregation costume, not an aggregation rule |

### 1.6 The fixed monotone profile family (27 members, declared before the run)

All non-increasing in rank: `dep` = `[20..1]`; `k{K}` = `[K..1, 0×(20−K)]` for
K ∈ {1,2,3,5,7,10,15,20} (the F94 grid); `unif{K}` = `[1×K, 0×(20−K)]` for the same K;
`exp{γ}` = `γ^(i−1)` for γ ∈ {0.5,0.7,0.8,0.9,0.95,0.99}; `pow{α}` = `i^(−α)` for
α ∈ {0.25,0.5,1,2}. Selection for `FIXBEST_mono` is by **leave-one-out accuracy on the fitting-fold
items**, ties broken toward the earlier member of the declared order (which begins with `dep`).

### 1.7 Permutation null (mandatory, per the tasking)

`N_PERM = 200`, `PERM_SEED = 12345` (the VGA constants). The **fitting-fold targets only** are
shuffled — the bank labels `s_i`, the retrieval and the held-out labels are untouched — and the full
pipeline (standardisation, 300-epoch fit, held-out evaluation) is re-run per permutation, i.e. the
null is at the **same fitting budget**. Run on the **PRIMARY arm × PRIMARY space** for all three
datasets. Reported `p = (1 + #{null ≥ observed}) / (N_PERM + 1)`; significance at `p < 0.05`.

---

## §2. FROZEN BARS

### 2.1 Quoted verbatim from `LITSWEEP6_MEMBANK.md` §3(e)

> **Frozen bars:**
> 1. Δacc ≥ **+0.010**, 5/5 fold signs, ≥3/5 strictly positive, ≥1 dataset.
> 2. **The control that decides whether C3 means anything (declare in advance, F95-2b style):** the
>    learned aggregator must beat **the best fixed monotone profile chosen on the fitting folds** — not
>    merely the deployed `[20..1]`. Beating `[20..1]` but not the best fixed profile is a win for
>    *profile tuning*, which F94 has effectively closed, and must be reported as such.
> 3. **Non-monotonicity read:** what fraction of held-out queries receive a **non-monotone** learned
>    weight profile, and is the Δ concentrated on them? If the learned profile is monotone almost
>    everywhere, C3 has collapsed into F94's family and is dead by that precedent.
> 4. Exchange rate ≥ 1.2 on the pathology population; class-balance sanity.

### 2.2 The decisive bar, quoted verbatim from the tasking

> The conditional-signal cash measured so far tops out at **+0.0269 (HateMM train-arena)**. C3 only
> matters if the richer function class extracts MORE of the +0.07-0.09 family oracle:
> **promotion-interest bar = pooled item-disjoint Δacc > +0.030 on ≥1 dataset with ≥4/5 fold signs
> AND materially above the +0.0269 gate benchmark on that dataset**. Below that → the whole
> conditional-aggregation family closes (record it as such).

**K-C3-P** is that bar. It is the bar the verdict is read on; LITSWEEP6's own bar 1 (+0.010) is
retained as the weaker *interest* threshold and reported alongside, so a result between +0.010 and
+0.030 is visible rather than hidden.

### 2.3 Degeneracy controls (mandatory, per the tasking; each fires a KILL, not a caveat)

> the record's own degeneracy control: agreement rate with (a) global threshold shifts and (b) any
> single fixed k from F94's grid — must be materially below ~95 % or the arm is a dead thing in a
> costume.

* **DEG-A — threshold-shift twin.** Pooled agreement between `C3_net`'s held-out predictions and
  `THRESH_best`'s. **≥ 95 % ⇒ KILL** (the RESTRANS §5.3 verdict form, applied to a weighting).
* **DEG-B — fixed-k twin.** `max_k` pooled agreement between `C3_net` and `FIXK_k` over the eight
  F94 grid profiles. **≥ 95 % ⇒ KILL** (C3 has re-derived a member of the closed family).
* **DEG-C — unconstrained-readout twin.** Pooled agreement between `C3_net` and `DIRECT_logit`, plus
  their Δacc gap. Reported as a *distinctness* reading, not an automatic kill: high agreement means
  the aggregation form is doing no work beyond what a plain readout of the same profile does, which
  is F70/F47 territory and must be stated as such.
* **Class-mixed coverage.** Fraction of held-out items whose top-20 contains **both** classes. On a
  class-pure neighbourhood no non-negative weighting can change the sign of `v`, so C3 is a no-op
  there **by construction**; this fraction is the hard ceiling on C3's reach and is reported before
  any Δ.

### 2.4 Operationalisation notes, frozen with the bars

* **Bar 1 / K-C3-P** are read on the **PRIMARY space (fused)**, pooled over all held-out items (each
  train item held out exactly once), against the deployed floor on the same fitting-fold bank.
  Secondary spaces are reported but cannot carry a bar.
* **Bar 4**'s exchange rate is the F95 definition — `fixed / broken` over all held-out items,
  reported with the fraction of the pathology population fixed (deployed-wrong held-out items whose
  nearest same-gold-class bank item sits within rank 5), so the number is directly comparable to
  F95's 0.53-0.95 and its 1.1667 ceiling, and to RESTRANS's 0.2647-0.9474.
* **Bar 3**'s non-monotonicity test: a learned profile is *non-monotone* if
  `max_i (g_{i+1} − g_i) > MONO_TOL · max_i g_i` with `MONO_TOL = 0.01` — i.e. any rise between
  adjacent ranks exceeding 1 % of the profile's own maximum. Reported with the Δacc restricted to
  the non-monotone subpopulation.
* **Class balance** (bar 4, F95 control 4): decision positive rate vs bank positive rate.
* The exchange rate is also reported against the **F47-gate benchmark population** wherever the two
  are comparable, per the tasking's "exchange-rate on the pathology population".

Gate order: §0-§2 written and both sha256 frozen → machinery self-test on **synthetic data only** →
parity gate (81 cells) + IMPL gate → arms and controls → permutation null → verdict. Machine output:
`scripts/analysis/aggnet_pregate_OUT.json`, run log `…_OUT.log`. Every number in §3 onward is
re-read from that JSON at report time, 4 dp.

### 2.5 Machinery validity — three synthetic arms, run BEFORE the freeze

`aggnet_pregate.py --selftest`, **synthetic data only**, at the real problem's scale (n = 700, 560
fitted / 140 held out, class rate 0.4). No real-dataset number was computed before the sha in §2.6
was frozen.

| arm | construction | Δacc | acc C3 | λ chosen | non-monotone |
|---|---|---|---|---|---|
| **A_conditional** | two populations needing **opposite** monotone treatments, keyed by the rank-1 margin — LITSWEEP6 §3(c)'s own worked example | **+0.5429** | **1.0000** | 0.01 | 0.4571 |
| **B_priorfallback** | the neighbourhood says nothing about `y` | +0.0929 | 0.5643 | 0.001 | 0.0000 |
| **C_deployedoptimal** | the deployed rule is already optimal; its residual errors are not a function of the profile | **0.0000** | 0.8429 | 100 | 0.0000 |

**A** shows the harness recovers a genuine conditional weighting rule exactly. **C** shows it returns
the deployed floor **bit-exactly** when there is nothing conditional to find — so a null below is a
property of the data, not of the code. These are the two properties VGA §2.7 established for its own
gate, and C is the one that required the §1.4 shrinkage design.

**B is not a null and must not be read as one.** C3's held-out accuracy there is **0.5643 = exactly
the majority-class rate**. Because `g ≥ 0` and the neighbourhoods are class-mixed, the sign of `v`
is freely choosable per query, so **C3's function class contains the constant/prior predictor and,
more generally, very nearly the full class of classifiers of the profile**. This is the DEG-C concern
made concrete before any real number was seen, and it is why the deployed floor is *not* the control
that matters: **bar 2 (best fixed monotone profile), DEG-C (unconstrained readout) and the
permutation null are.**

**One pre-registered correction to bar 3.** Bar 3's stated inference is
*"monotone almost everywhere ⇒ C3 has collapsed into F94's family and is dead by that precedent"*.
That inference is **unsound as written**, and the reason is definitional rather than empirical:
**F94's family is a single GLOBAL k**, whereas a *per-query* monotone profile is not in it. The
declared `C3_metak` arm is exactly that object — a per-query softmax mixture over the eight F94
k-profiles, every emitted profile monotone by construction, yet not expressible as any one global k
— so "monotone" and "inside F94's family" are demonstrably different predicates. Bar 3 is therefore
still **reported exactly as specified**, but the sound version of the same worry is **DEG-B**
(agreement with any *single fixed* k), which the tasking independently required, and **that is the
one a kill is read on**. `C3_metak`'s own measured result (§4) is what prices the per-query-monotone
sub-family.

> **Erratum, disclosed rather than silently fixed.** An earlier draft of this paragraph cited
> "arm A reaches perfect accuracy at λ = 1.0 with 0 % non-monotone profiles" as measured support for
> separating conditionality from non-monotonicity. That number came from a **superseded pre-freeze
> iteration of the harness** (the naive `weight_decay` regulariser of §1.4, replaced before the
> freeze) and is **not** a property of the frozen script: in the frozen self-test, arm A selects
> λ = 0.01 and solves the task with **0.4571** non-monotone profiles, and λ ≥ 0.1 does not solve it
> at all. The claim above has been restated so it rests on a definitional argument plus the declared
> `C3_metak` arm, neither of which depends on the withdrawn number. No bar changed.

### 2.6 Frozen script shas

| path | sha256 |
|---|---|
| `scripts/analysis/aggnet_pregate.py` | `8e95c2fc796c7c94e6671d785f8e6d21d2b5edfdb4cfb0b376234d129c11e8a9` |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` (F89, imported unmodified, sha asserted at run time) |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` (F95, imported unmodified for the fold protocol, loaders and space constructors) |

<!-- ============ EVERYTHING BELOW THIS LINE WAS WRITTEN AFTER THE RUN ============ -->

---

## §3. GATES — 81/81 PARITY, 45/45 IMPL, AND A FREE DETERMINISM CHECK

`sha256(mechfix_ops.py)` asserted equal to `635c1312…c83fc8d` and
`sha256(mechnov_pairverify.py)` to `77b0defd…b7240d` before anything ran. The deployed floor
recomputed inside this harness then reproduced **every** F95 train-side number at 4 dp, read directly
from `mechnov_pairverify_{hatemm,zh,en}_OUT.json` — **81/81 cells PASS** (27 per dataset: pooled acc
+ pooled mF1 + 5 per-fold acc + `n_deployed_wrong` + `n_pathology_pop`, over 3 spaces). Any mismatch
aborts the run. The floors reproduce `RESTRANS_PREGATE_RECORD.md` §3 exactly.

| dataset | n | floor acc / mF1 (fused) | per-fold acc (fused) | wrong | pathology |
|---|---|---|---|---|---|
| HateMM | 744 | 0.8441 / 0.8419 | 0.7987, 0.8322, 0.8926, 0.8255, 0.8716 | 116 | 88 |
| MHC-ZH | 579 | 0.8480 / 0.8281 | 0.8534, 0.8534, 0.9224, 0.8017, 0.8087 | 88 | 79 |
| MHC-EN | 549 | 0.7796 / 0.7286 | 0.7091, 0.7545, 0.8091, 0.8182, 0.8073 | 121 | 109 |

**IMPL gate: 45/45 (dataset × space × fold).** The C3 vote engine evaluated at the constant profile
`g = [20..1]` was asserted **bit-for-bit equal** to `mechfix_ops.deployed_vote` in votes, predictions
and the retrieved index matrix, every fold. The treatment therefore differs from the floor **in the
weighting and in nothing else**.

**Determinism gate (free, and not planned as a control).** The main battery and the permutation
battery were run as independent processes; their fused cells were asserted identical across the two
runs on 52 quantities per dataset and matched exactly. The pipeline is deterministic.

### 3.1 Coverage — what C3 can reach at all, before any Δ

A non-negative weighting can change `sign(v)` only if the top-20 contains **both** classes (every
`cos_i > 0` here, so a class-pure neighbourhood is unflippable by construction).

| dataset | class-mixed top-20 | deployed errors reachable | **family oracle Δacc** |
|---|---|---|---|
| HateMM | 646 / 744 = **0.8683** | 111 / 116 = 0.9569 | **+0.1492** |
| MHC-ZH | 480 / 579 = **0.8290** | 88 / 88 = 1.0000 | **+0.1520** |
| MHC-EN | 533 / 549 = **0.9709** | 120 / 121 = 0.9917 | **+0.2186** |

(`aggnet_pregate_diag_OUT.json`, post-hoc, adds no arm.) **C3's function class can in principle fix
96-100 % of every deployed error.** That is 2-4× the F95 adjudication-gate oracle
(+0.0726 / +0.0535 / +0.0893, VGA §1) and 10-15× F94's per-seed oracle-k ceiling (+0.0145 max over
six arms, `KSWEEP_RECORD.md` §5). **C3 enters this pregate with by far the largest ceiling any
member of this family has ever had.** Keep that number in view for §7.

---

## §4. PRIMARY READ — fused space, pooled over all held-out train items

Every number re-read at report time from the merged `aggnet_pregate_OUT.json`, 4 dp. All declared
arms ran; none was dropped. `ER` = exchange rate = fixed / broken over all held-out items.

| dataset | arm | Δacc | ΔmF1 | fold signs | ≥0 | fixed/broken | **ER** | changed | pos-rate (bank) |
|---|---|---|---|---|---|---|---|---|---|
| **HateMM** | **C3_net (PRIMARY)** | **+0.0134** | +0.0117 | `-0+++` | 4/5 | 22 / 12 | **1.8333** | 34 | 0.4355 (0.4005) |
| | C3_net seed 1 | +0.0121 | +0.0105 | `0++++` | 5/5 | 21 / 12 | 1.7500 | 33 | 0.4395 |
| | C3_net seed 2 | +0.0134 | +0.0117 | `0++++` | 5/5 | 23 / 13 | 1.7692 | 36 | 0.4355 |
| | C3_net_dlen | +0.0121 | +0.0112 | `+++++` | 5/5 | 17 / 8 | 2.1250 | 25 | 0.4556 |
| | C3_metak (monotone twin) | +0.0040 | +0.0024 | `+-++-` | 3/5 | 22 / 19 | 1.1579 | 41 | 0.4422 |
| | `FIXBEST_mono` (**bar 2**) | −0.0027 | −0.0034 | `--+0+` | 3/5 | 16 / 18 | 0.8889 | 34 | 0.4651 |
| | `FIXBEST_oracle` *(ORACLE)* | +0.0175 | +0.0171 | `+++++` | 5/5 | 16 / 3 | 5.3333 | 19 | 0.4637 |
| | `THRESH_best` (**DEG-A**) | **+0.0188** | +0.0152 | `0+++-` | 4/5 | 38 / 24 | 1.5833 | 62 | 0.3978 |
| | `DIRECT_logit` (**DEG-C**) | **+0.0134** | +0.0104 | `+-+++` | 4/5 | 32 / 22 | 1.4545 | 54 | 0.4113 |
| **MHC-ZH** | **C3_net (PRIMARY)** | **−0.0069** | −0.0159 | `++-+0` | 4/5 | 21 / 25 | **0.8400** | 46 | 0.2971 (0.3109) |
| | C3_net seed 1 | −0.0121 | −0.0193 | `0--+0` | 3/5 | 13 / 20 | 0.6500 | 33 | 0.3126 |
| | C3_net seed 2 | −0.0121 | −0.0199 | `0+-+0` | 4/5 | 14 / 21 | 0.6667 | 35 | 0.3092 |
| | C3_net_dlen | −0.0138 | −0.0193 | `0--0-` | 2/5 | 6 / 14 | 0.4286 | 20 | 0.3247 |
| | C3_metak | −0.0155 | −0.0216 | `00--0` | 3/5 | 12 / 21 | 0.5714 | 33 | 0.3230 |
| | `FIXBEST_mono` | −0.0138 | −0.0161 | `0--+-` | 2/5 | 11 / 19 | 0.5789 | 30 | 0.3454 |
| | `FIXBEST_oracle` *(ORACLE)* | +0.0156 | +0.0174 | `++0++` | 5/5 | 24 / 15 | 1.6000 | 39 | 0.3472 |
| | `THRESH_best` | −0.0069 | −0.0083 | `+0-0-` | 3/5 | 2 / 6 | 0.3333 | 8 | 0.3454 |
| | `DIRECT_logit` | +0.0017 | −0.0077 | `+--++` | 3/5 | 24 / 23 | 1.0435 | 47 | 0.2850 |
| **MHC-EN** | **C3_net (PRIMARY)** | **+0.0000** | +0.0000 | `+-000` | 4/5 | 1 / 1 | **1.0000** | 2 | 0.2605 (0.3060) |
| | C3_net seed 1 | −0.0055 | −0.0073 | `+000-` | 4/5 | 5 / 8 | 0.6250 | 13 | 0.2587 |
| | C3_net seed 2 | −0.0055 | −0.0073 | `+000-` | 4/5 | 5 / 8 | 0.6250 | 13 | 0.2587 |
| | C3_net_dlen | +0.0018 | +0.0017 | `+0000` | 5/5 | 1 / 0 | n/a | 1 | 0.2587 |
| | C3_metak | −0.0018 | −0.0017 | `0-000` | 4/5 | 0 / 1 | n/a | 1 | 0.2623 |
| | `FIXBEST_mono` | +0.0036 | +0.0034 | `+-0+0` | 4/5 | 13 / 11 | 1.1818 | 24 | 0.2568 |
| | `FIXBEST_oracle` *(ORACLE)* | +0.0291 | +0.0419 | `+0++0` | 5/5 | 21 / 5 | 4.2000 | 26 | 0.2860 |
| | `THRESH_best` | −0.0164 | −0.0174 | `+-0+-` | 3/5 | 10 / 19 | 0.5263 | 29 | 0.2696 |
| | `DIRECT_logit` | −0.0036 | −0.0034 | `--+--` | 1/5 | 25 / 27 | 0.9259 | 52 | 0.2641 |

### 4.1 The whole battery in one line

Across **45 C3 cells** (3 datasets × 3 spaces × 5 C3 arms): **17 positive, 8 exactly zero,
20 negative; the maximum anywhere is +0.0134 and NO cell reaches +0.030.** The three best cells are
all HateMM × fused (+0.0134 seed 2, +0.0134 seed 0, +0.0121 seed 1). **Secondary spaces do not
rescue anything.** The PRIMARY arm is negative in **3/3** MHC-ZH spaces (−0.0069 / −0.0035 / −0.0069)
and is +0.0000 / +0.0018 / −0.0073 on MHC-EN. Taking the single best cell per dataset over all
5 arms × 3 spaces — a 15-way maximum, reported as an upper bound and not as a result — gives
**+0.0134** (HateMM × fused × seed 2), **+0.0069** (MHC-ZH × text × seed 1) and **+0.0036**
(MHC-EN × text × seed 1). Even that shopped maximum is below half the bar on the best dataset and
below a quarter of it on the other two.

### 4.2 Two facts about HateMM that must travel together

**The good one, stated without inflation.** C3's exchange rate on HateMM is **1.8333** (22 fixed /
12 broken) — above the frozen 1.2 bar, above F95's ceiling of 1.1667 over 36 cells, and above every
C1 cell in `RESTRANS_PREGATE_RECORD.md` (0.2647-0.9474). C3 is the first *treatment* arm in this
campaign to buy fixes at better than par on a dataset where it also moves accuracy. The `+dlen` arm
reaches 2.1250 with 5/5 fold signs.

**The one that prices it.** It buys them on **34 items out of 744**, so a 1.83 exchange rate is worth
**+0.0134** — less than half the promotion bar, and **below the +0.0269 the F47-feature gate already
banked on the same dataset in the same arena** (VGA §4.3, re-read from `vga_pregate_OUT.json`:
`f47ctrl_full:gbm`, 36 fixed / 16 broken, ER 2.25, p = 0.0050, fold signs `+++++`). **The richer
function class extracted less of the family oracle than the cheap gate did**, which is the exact
question the tasking posed.

---

## §5. CONTROLS

### 5.1 Bar 2 — vs the best fixed monotone profile chosen on the fitting folds — **PASS on 2 of 3**

| dataset | C3_net | `FIXBEST_mono` | **C3 − FIXBEST** | profile chosen per fold |
|---|---|---|---|---|
| HateMM | +0.0134 | −0.0027 | **+0.0161** | `k7`, `pow1.0`, `unif20`, `exp0.7`, `k10` |
| MHC-ZH | −0.0069 | −0.0138 | **+0.0069** | `pow0.5`, `exp0.8`, `pow1.0`, `exp0.7`, `k15` |
| MHC-EN | +0.0000 | +0.0036 | **−0.0036** | `exp0.95`, `exp0.99`, `exp0.9`, `pow0.25`, `pow1.0` |

This is the one bar C3 clears on its own terms: on HateMM and MHC-ZH the learned conditional
weighting **is** better than the best fixed monotone profile the fitting folds can pick, so the
result is not merely "profile tuning", which LITSWEEP6 §3(e) bar 2 was written to catch. Note also
that fixed-profile selection is itself **unstable** — a different profile family member wins in every
fold on every dataset, and `FIXBEST_mono` is *negative* on two datasets, i.e. picking a fixed profile
on the fitting folds actively costs accuracy. **Bar 2 passing is real but it is a low bar**: the
thing C3 beats is itself worse than doing nothing.

### 5.2 Bar 3 — non-monotonicity — **does NOT fire, and the Δ lives entirely on the non-monotone half**

| dataset | non-monotone profiles | Δacc \| non-monotone | Δacc \| monotone | changed (nm / mono) | mean rise / max |
|---|---|---|---|---|---|
| HateMM | 741 / 744 = **0.9960** | +0.0135 | 0.0000 | **34 / 0** | 0.2021 |
| MHC-ZH | 555 / 579 = **0.9585** | −0.0054 | −0.0417 | 45 / 1 | 0.2025 |
| MHC-EN | 190 / 549 = **0.3461** | −0.0053 | +0.0028 | 1 / 1 | −0.0007 |

Bar 3 asked whether the learned profile is monotone almost everywhere, in which case C3 would have
"collapsed into F94's family". **It is not** — 96-100 % of HateMM/ZH profiles are non-monotone, and
on HateMM **every one of the 34 changed decisions falls on a non-monotone profile**. By the letter of
bar 3, C3 passes.

**But the mean learned profile shows what the non-monotonicity actually is**, and it is not what the
candidate was sold on. Mean `g` over held-out items, HateMM: `[2.0086, 0.9602, 0.7991, 0.7559,
0.6863, 0.6518, …, 0.1375, 0.0947, 0.0480]`; MHC-ZH: `[2.0161, 1.0259, 0.8747, …]`; MHC-EN:
`[1.0132, 0.9574, 0.9111, …, 0.0500]` — the deployed profile, unmoved. **On the two datasets where
C3 moved at all, what it learned is "double the rank-1 weight and otherwise keep the deployed
decay".** That is a *global* re-shaping with a small conditional ripple on top, not the
"conditionally trust rank 1" rule the licence was written for — and a steeper fixed profile is
already inside the bar-2 family (`exp0.7`, `pow1.0`, `k7`), which is why the bar-2 margin is only
+0.0161 and why §5.3's fixed-k agreement is 0.96.

### 5.3 Degeneracy controls — **DEG-A and DEG-B both FIRE on the only dataset with a positive**

Pooled agreement with `C3_net`'s held-out predictions. Frozen kill threshold: **≥ 0.95**.

| dataset | **DEG-A** global threshold shift | **DEG-B** best single fixed k | DEG-C unconstrained readout | (vs deployed) | (vs FIXBEST) |
|---|---|---|---|---|---|
| **HateMM** | **0.9570 → FIRES** | **0.9610 (k=15) → FIRES** | 0.9516 | 0.9543 | 0.9543 |
| MHC-ZH | 0.9206 | 0.9206 (k=7) | 0.9257 | 0.9206 | 0.9378 |
| **MHC-EN** | 0.9508 → FIRES | **0.9964 (k=20) → FIRES** | 0.9053 | 0.9964 | 0.9526 |

**On HateMM — the only dataset where C3 produced a positive Δacc — it agrees with a plain global
decision-threshold shift on 95.70 % of items and with a single fixed k=15 vote on 96.10 %.** Both
sit above the frozen 0.95 line. This is `RESTRANS_PREGATE_RECORD.md` §5.3's verdict form arriving
again on a different operator: the arm is **a threshold move and a fixed-k move wearing a
conditional costume** on 96 % of the items it touches. On MHC-EN, DEG-B is 0.9964 against **k = 20**,
i.e. C3 *is* the deployed rule there (2 items changed).

The arithmetic that makes this decisive rather than a technicality: **`THRESH_best` alone scores
+0.0188 on HateMM, more than C3's +0.0134**, using no network, no profile and no conditioning — and
`DIRECT_logit`, an unconstrained logistic readout of the identical profile, scores **+0.0134,
exactly C3's number**. The aggregation form contributes nothing that either of its two degenerate
twins does not already contribute.

### 5.4 Class balance (bar 4, F95 control 4) — **PASS, no collapse**

| dataset | bank pos-rate | deployed pos-rate | C3_net pos-rate | deviation from bank |
|---|---|---|---|---|
| HateMM | 0.4005 | 0.4812 | 0.4355 | 0.0350 |
| MHC-ZH | 0.3109 | 0.3489 | 0.2971 | 0.0138 |
| MHC-EN | 0.3060 | 0.2605 | 0.2605 | 0.0455 |

No arm collapses to one class and every emitted decision sits within 0.05 of its bank rate — unlike
C1, which drifted to 0.5181 / 0.5501 against bank rates of 0.31. This bar passing is what licenses
reading the others as measurements rather than artefacts.

### 5.5 Exchange rate (bar 4) — **1 of 3 clears 1.2**

HateMM **1.8333** (clears), MHC-ZH **0.8400** (fails), MHC-EN **1.0000** on 2 changed items
(fails, and is not interpretable at that count). Over all 45 C3 cells the exchange rate ranges
0.4286-5.0000, but every cell above 1.2 changes ≤ 46 items.

### 5.6 The shrinkage the inner CV actually chose

`λ` per fold, `C3_net`: HateMM `[0.01, 0.01, 0.01, 0.01, 0.01]`; MHC-ZH `[0.1, 0.01, 0.01, 0.01,
0.01]`; MHC-EN `[0.1, 0.1, 1.0, 1.0, 1.0]`. The inner CV never selects the free end of the grid
(1e-4) for the primary arm and selects heavy shrinkage on MHC-EN — where the arm consequently sits
at the floor. On stability seeds it twice selects the maximum `λ = 100` on MHC-ZH, i.e. **the inner
CV chooses "be the deployed rule" outright in 2 of 5 folds**. The harness is doing exactly what
§2.5's arm C showed it would when there is nothing conditional to learn.

### 5.7 An erratum-strength sharpening of `RESTRANS_PREGATE_RECORD.md` §7.1

That record flagged, as a note and not a candidate, that its **dead relative D1** (`logistic(deployed
vote, log(1+volume)) → gold`, a *query-side* score-level length de-bias) produced the campaign's first
above-bar exchange rates on HateMM: **+0.0215 acc / ER 1.8889** (fused) and **+0.0282 / 2.2353**
(text). It attributed the effect to the length covariate, "the only dataset where the covariate
actually carries information".

`THRESH_best` in this record is the same operator **with the covariate removed** — a bare global
threshold on the deployed vote, no features at all — and it is measured here, in the identical arena
and protocol, at **+0.0188 / ER 1.5833** (fused) and **+0.0242 / ER 1.7200** (text). Re-read from
`restrans_pregate_OUT.json` and `aggnet_main_hatemm_OUT.json` respectively:

| operator | HateMM fused Δacc | HateMM text Δacc |
|---|---|---|
| D1 = threshold **+ length covariate** | +0.0215 | +0.0282 |
| `THRESH_best` = threshold **only** | **+0.0188** | **+0.0242** |
| **increment attributable to the covariate** | **+0.0027** (≈ 2 items) | **+0.0040** (≈ 3 items) |

**D1's positive is ~87 % a bare decision-threshold move, not a length de-bias.** The covariate is
worth 2-3 items. This does not change RESTRANS's verdict (D1 was already a note on a measured-dead
lever, in a raw train-side arena, below the house bar) but it removes the one remaining optimistic
reading of it, and it should travel with the number from here on. It is also the mechanical reason
DEG-A fires in §5.3: on HateMM's train arena the deployed vote's **threshold** is simply mis-set,
and any sufficiently flexible operator — a length logistic, a bare threshold, or a 1316-parameter
conditional aggregator — converges to the same correction. Three independent measurements now agree
on this, and the lever remains measured **dead in the deployed head space on test**
(ERRPAT-HateMM §2.1: +0.0000 / +0.0016).

---

## §6. PERMUTATION NULL — satisfied on all three, and **uninformative on all three**

Mandatory per the tasking. `N_PERM = 100`, `PERM_SEED = 12345`, fitting-fold targets shuffled
(bank labels, retrieval and held-out labels untouched), the **full nested pipeline including the
inner-CV `λ` selection** re-run per draw, PRIMARY arm × PRIMARY space.

| dataset | observed | null mean ± sd | null q95 | **null max** | draws ≥ 0 | **p** |
|---|---|---|---|---|---|---|
| HateMM | +0.0134 | −0.1063 ± 0.0274 | −0.0590 | **−0.0323** | **0 / 100** | **0.0099** |
| MHC-ZH | −0.0069 | −0.1063 ± 0.0187 | −0.0776 | **−0.0259** | **0 / 100** | **0.0099** |
| MHC-EN | +0.0000 | −0.0523 ± 0.0221 | −0.0181 | **−0.0109** | **0 / 100** | **0.0099** |

All three clear the bar at the smallest p the design can produce (1/101 = 0.0099). **This must not be
banked as evidence, and the reason is visible in the table.**

**Not one of the 300 null draws reached zero** (maxima −0.0323 / −0.0259 / −0.0109; minima −0.1626 /
−0.1485 / −0.1129). The null therefore **never falls back to the deployed floor**: with shuffled
targets the inner CV is choosing `λ` by an argmax over seven noisy, uninformative estimates, so it
lands on a small `λ` roughly at random and the refit is then dragged 5-16 accuracy points below the
floor. Any arm that *does* fall back to the floor consequently scores p = 0.0099 automatically.

**MHC-EN is the proof, and it is unambiguous.** There, `C3_net` changes **2 items** out of 549, its
Δacc is **exactly +0.0000**, and it is nonetheless "significant at p = 0.0099". A test that certifies
a two-item no-op as significant is measuring the null's fragility, not the arm's signal.

**What this null does and does not discharge.** It discharges *"the fitted rule is better than a rule
fitted to shuffled labels"* — a real but very weak claim, made nearly automatic by §1.4's
deployed-anchored fallback. It does **not** discharge *"the fitted rule beats the deployed floor"* or
*"the conditional part is doing the work"*. Those are exactly what **bar 2** (best fixed monotone
profile), **DEG-A/B** (threshold-shift and fixed-k twins) and **DEG-C** (unconstrained readout) test,
and they are the controls the verdict rests on. This is a **design limitation of the permutation null
as specified for an arm with a floor fallback**, recorded here so it is not repeated: a more
informative null for this arm shape would shuffle targets *and* force the same `λ`, or would compare
against the floor rather than against a noise-fitted net.

Contrast with VGA, where the same test was informative: there the gate's fallback ("fire on nothing",
net exactly 0) **was** reachable by the null, its null means were −0.0024 to −0.0072 rather than
−0.05 to −0.11, and the primary verifier arm consequently returned honest non-significance
(p = 0.8706 / 0.5174 / 0.9751). The difference is entirely in whether the null can reach the
fallback.

---

## §7. VERDICT

# **KILL — C3 misses the decisive bar by a factor of more than two, and the two mandatory degeneracy controls both fire on the only dataset where it is positive.**

| bar | requirement | measured | verdict |
|---|---|---|---|
| **K-C3-P** (decisive) | Δacc > **+0.030** on ≥1 dataset, ≥4/5 fold signs, **and materially above +0.0269** on that dataset | best C3 cell **anywhere** = **+0.0134** (HateMM × fused); **0 of 45 C3 cells** reach +0.030; on HateMM it lands **0.0135 BELOW** the F47-gate benchmark it had to exceed | **FAIL** |
| **bar 1** (LITSWEEP6) | Δacc ≥ +0.010, **5/5** folds Δ ≥ 0, ≥3/5 strictly positive, ≥1 dataset | HateMM PRIMARY +0.0134 but fold signs `-0+++` = **4/5**; MHC-ZH −0.0069; MHC-EN +0.0000 | **FAIL** (see note) |
| **bar 2** | must beat the best fixed monotone profile chosen on the fitting folds | **+0.0161** (HateMM), **+0.0069** (ZH), −0.0036 (EN) | **PASS 2/3** |
| **bar 3** | non-monotonicity read | 0.9960 / 0.9585 / 0.3461 non-monotone; on HateMM **34/34** changed decisions sit on non-monotone profiles | **does not fire** |
| **bar 4a** exchange rate | ≥ 1.2 on the pathology population | HateMM **1.8333**; ZH 0.8400; EN 1.0000 on 2 items | **PASS 1/3** |
| **bar 4b** class balance | pos-rate near bank rate | max deviation **0.0455**, no collapse | **PASS** |
| **DEG-A** (mandatory) | agreement with a global threshold shift **< 0.95** | HateMM **0.9570**, EN **0.9508** | **FIRES → KILL** |
| **DEG-B** (mandatory) | agreement with any single fixed k **< 0.95** | HateMM **0.9610** (k=15), EN **0.9964** (k=20) | **FIRES → KILL** |
| **DEG-C** | distinctness from an unconstrained readout | HateMM agreement 0.9516 **and identical Δacc (+0.0134 vs +0.0134)** | aggregation form adds nothing |
| permutation null (mandatory) | must beat the label-shuffled null at the same fitting budget | p = **0.0099** on all three — but **0 of 300 null draws reached zero**, so an arm that merely falls back to the floor passes automatically (MHC-EN: 2 items changed, Δ = +0.0000, p = 0.0099) | **PASS but UNINFORMATIVE** (§6) |

**Note on bar 1, stated so the boundary is visible rather than hidden.** The PRIMARY arm misses the
5/5 fold-sign clause by **one fold**. Two of the three stability seeds (`0++++`, +0.0121 / +0.0134)
and the SECONDARY `+dlen` arm (`+++++`, +0.0121) would clear it. Under the freeze those cannot carry
a bar, so bar 1 fails — but the honest reading is that C3 sits **right at the boundary of the weak
+0.010 interest threshold and nowhere near the +0.030 decision bar**. Nothing about the kill turns on
that one fold sign.

**Why the kill is mechanistic rather than budgetary.** C3 did not fail to reach the pathology: it
entered this pregate with the **largest oracle ceiling of anything ever tried on this object**
(§3.1: +0.1492 / +0.1520 / +0.2186, i.e. 96-100 % of every deployed error is inside its function
class), it was given a network that §2.5 proved can recover a genuine conditional weighting rule
*exactly*, it was given a nested shrinkage selection that §2.5 proved returns the floor bit-exactly
when there is nothing to find, and it still converged — on the one dataset where it moved — to
**a global threshold shift (95.70 % agreement) and a fixed k=15 vote (96.10 % agreement)**, both of
which are already-closed levers, one of which (`THRESH_best`, +0.0188) **outscores it** while using
no network at all.

This is `RESTRANS_PREGATE_RECORD.md` §5.3's verdict arriving on a second, independent operator: the
label-field correction was a threshold shift in an item-level costume on 95-99 % of items; the
weighting correction is a threshold shift **and** a fixed-k truncation in a conditional costume on
96 %. Two different halves of the vote, edited two different ways, degenerate to the same place.

---

## §7.1 THE FAMILY CLOSURE — the conditional-aggregation axis is settled

Per the tasking: *"Below that → the whole conditional-aggregation family closes (record it as such)."*
Recording it.

**The family.** Every measured attempt to convert the deployed top-20 neighbourhood's own **local
configuration** — its cosines, its neighbour labels, and deterministic functions of them — into
accuracy, **without** changing retrieval, the bank, the key space or the candidate set. Four members
are now measured, all in the same raw train arena, all item-disjoint, all permutation-controlled:

| # | member | what it changes | delivered (HateMM / ZH / EN) |
|---|---|---|---|
| 1 | **F95** pair-verification | replaces the vote with a trained relation adjudicator | −0.0040 / −0.0466 / −0.0146 (ungated) |
| 2 | **VGA C1** verifier-gated adjudication | gates that replacement per item on the verifier profile | best +0.0108; PRIMARY arm indistinguishable from its own null (p = 0.8706 / 0.5174 / 0.9751); **K-VGA-3 fired** — the F47 features beat the verifier features 3/3 |
| 3 | **VGA C2 / VNQ** neighbourhood-quality risk | reads the same profile as a risk ordering | loses to the **free deployed vote margin** on all three |
| 4 | **C3** learned aggregation profile net (this record) | keeps the vote, learns the **weighting** conditional on the same configuration | **+0.0134 / −0.0069 / +0.0000**; 0/45 cells at +0.030; DEG-A and DEG-B both fire |

**The one positive in the family, and it has not moved.** The **F47-feature adjudication gate**:
**+0.0269 / +0.0104 / +0.0182**, permutation-validated at p = 0.0050 / 0.0050 / 0.0100 (re-read from
`vga_pregate_OUT.json`; HateMM `f47ctrl_full:gbm` 36 fixed / 16 broken, ER 2.25, signs `+++++`).
It is below the +0.030 bar on all three, it gates an adjudicator that is **net-negative ungated**,
and VGA already ruled it **analysis-grade only**. C3 was the test of whether a richer function class
could extract more of the same oracle. **It extracted less.**

**The arithmetic that closes the axis, and the part worth keeping for the paper.** Within this
family, **delivery is uncorrelated with ceiling** — and C3 is the datum that establishes it, because
it is the member with the largest ceiling by a wide margin:

| member | oracle ceiling (HateMM / ZH / EN) | delivered |
|---|---|---|
| F94 global-k | **+0.0145** (max per-seed oracle k over 6 arms) | −0.0140 to +0.0041 (dev-legal) |
| F95 / VGA adjudication gate | +0.0726 / +0.0535 / +0.0893 | +0.0269 / +0.0104 / +0.0182 |
| **C3 conditional weighting** | **+0.1492 / +0.1520 / +0.2186** | **+0.0134 / −0.0069 / +0.0000** |

The standing alternative explanation for every previous member of this family was *"the operator
could not reach the errors"*. C3 removes it: **96-100 % of every deployed error is inside C3's
reach**, it is free to place any non-negative weight on any of the 20 already-retrieved neighbours,
and §2.5 arm B measured that this makes its function class very nearly **the full class of
classifiers of the profile**. It still delivers nothing. What binds is not reach and not capacity: it
is that **the local configuration does not carry a learnable signal about which neighbours to trust**
at n = 549-744. The +0.0269 F47 gate is the whole of what this information object is worth, and it
was already measured before this pregate ran.

**Therefore, closed. Do not re-propose, on this neighbourhood object:**
* **(a)** any learned re-weighting, soft-mixture-over-k, attention, or gating **over the deployed
  top-20** — C3 spans that class and DEG-B shows what it converges to;
* **(b)** any per-item selector, router or adjudication gate over the same neighbourhood, **with any
  feature family** — the verifier features are dead by K-VGA-3, and the F47 features have a measured
  ceiling of +0.0269 that is already banked;
* **(c)** "a bigger / better aggregator" — capacity is measured not to be the binding constraint
  (§2.5 arm B), and more capacity without more regularisation is measured to be actively harmful
  (§2.5 arm C, −0.1714 unregularised);
* **(d)** the **static** sibling, **LITSWEEP6 C5** (per-entry soft reliability weights `α_i`), is not
  formally inside this closure — `α_i` keys on bank-item identity, which C3's rank-indexed profile
  cannot see — but it is **strongly downgraded** by this result: it is a *query-independent*
  reweighting of a fixed support with a strictly smaller effective ceiling, and the
  query-*conditional* generalisation with a 96-100 % ceiling delivered ~0. LITSWEEP6's own filter (i)
  already priced it dead; this is the second, measured reason. **Recommend it be dropped as a
  performance candidate and retained only for its pillar-④ auditability role**, which is what §5 of
  the sweep record said its real value was.

**What is NOT closed by this, stated precisely so the boundary is usable.** The closure covers
operators whose **input is the (cosine, label) profile of the deployed top-20**. It does **not**
cover:
* **LITSWEEP6 C2** (cell-conditional synthesis into the bank) — it changes **membership**, i.e.
  *which items are retrievable at all*, which no member of this family touches. It is the only
  remaining candidate that can create support where there is none. Its prereg still needs the
  rewrite `RESTRANS_PREGATE_RECORD.md` §6 mandated (its placement criterion cannot use `p̂`).
* **LITSWEEP6 C4** (aggregate-then-compare subspace residual) — its input is the retrieved
  **vectors**, not their cosine/label profile, so it is outside C3's function class by information
  content, not merely by functional form. C4's $0 pregate is untouched by anything measured here.

---

## §7.2 ROUTING

1. **Do not spend GPU. Do not promote. Do not ceremony.** No arm in this record approaches the house
   bar, and the two mandatory degeneracy controls fire on the only positive dataset.
2. **Next candidate: C4 (aggregate-then-compare subspace residual)** — nominated by
   `RESTRANS_PREGATE_RECORD.md` §7.2 before this run and **unaffected** by this closure (§7.1). It
   is $0, CPU, and its degeneracy bar (both class residuals collapse at every rank `r`) is already
   written.
3. **Then C2 (synthesis into the bank)** — the only operator left that changes membership, and the
   sweep record's best paper story. Its prereg must be rewritten first (no `p̂` placement).
4. **C5: drop as a performance candidate** (§7.1(d)); keep the per-entry weight as pillar-④ material.
5. **Carry §5.7 forward**: RESTRANS §7.1's D1 note is now measured to be ~87 % a bare threshold move.
   The HateMM train-arena threshold observation has three independent measurements and remains
   **dead in the deployed head space on test** — it is a diagnostic about the train arena, not a
   lever, and it should stop being re-derived.

---

## §8. LIMITATIONS

1. **Arena.** Banked **raw** encoder key space, **train** split, not the deployed head space and not
   test — the F95 precedent, inherited by both sibling pregates (F47: head LOO train acc 0.998, so a
   train-side screen in head space measures memorisation). A raw-space, train-side null does not
   logically entail a head-space or test null. This is stated first because it cuts *against* the
   negative verdict. It is mitigated, not removed, by the fact that the bars that actually fire here
   are **relative** comparisons measured inside one arena under one protocol (C3 vs the best fixed
   monotone profile, vs a global threshold shift, vs a fixed k, vs its own label-shuffled null), and
   relative comparisons are far less arena-sensitive than an absolute Δacc.
2. **No seeds in the arena; three seeds in the network.** The raw features are seed-independent, so
   the sign evidence is **5 folds**, not 3 seeds — as in both sibling records. The network's own
   initialisation seed is varied (0 PRIMARY, 1 and 2 as a stability read), which controls the
   *fitting* variance but not the fold-draw variance.
3. **One fold draw.** `FOLD_SEED = 0`, the frozen F95 assignment. No resampling of the outer folds.
4. **Train-side only; no test quantity exists in this document by design** (LITSWEEP6 §3(e)). For the
   same reason the ERRPAT stable-core id lists could not be used as the exchange-rate population —
   they are test-split objects — and the F95 pathology population (deployed-wrong held-out items
   whose nearest same-gold-class bank item is within rank 5) is the train-side substitute, exactly as
   in `RESTRANS_PREGATE_RECORD.md` §1.6.
5. **One network family, one loss, one epoch budget.** `Linear(d,16)→tanh→Linear(16,20)→softplus`,
   BCE on the deployed logit, 300 full-batch Adam epochs at `lr = 1e-2`. Only the shrinkage strength
   `λ` is selected (nested, 7-point grid). A different architecture, a margin loss, or a
   learned-temperature readout are unmeasured. What bounds the value of that unmeasured space is
   §2.5's arm B: the function class is *already* nearly the full class of profile classifiers, so
   more capacity is not the missing ingredient.
6. **The Δlength profile block is SECONDARY and its covariate is measured dead on 2 of 3 datasets**
   (`RESTRANS_PREGATE_RECORD.md` §6). It is reported for completeness, not as a live arm.
7. **The permutation null shuffles the fitting targets only**, at the same fitting budget including
   the nested `λ` selection. It therefore tests "is the fitted conditional rule better than one
   fitted to noise", not "is the deployed floor beatable at all". `N_PERM = 100` rather than VGA's
   200, because each draw re-runs 31 network fits; the p-value resolution is 1/101.
8. **No test-split file was opened, no test label read, no oracle used to choose anything.** The two
   oracle quantities that appear (`FIXBEST_oracle`, and the family ceiling in §6) are **reported
   ceilings computed on held-out train items**, never used to select an arm or an operating point.

---

## §9. FILE MANIFEST

| path | contents |
|---|---|
| `scripts/analysis/aggnet_pregate.py` | frozen implementation, sha256 `8e95c2fc…c11e8a9`; imports `mechfix_ops.py` and `mechnov_pairverify.py` unmodified with both shas asserted at run time; includes the pre-freeze `--selftest` |
| `scripts/analysis/aggnet_pregate_report.py` | merge + report only; asserts the fused cell is identical across the two runs (determinism gate) and re-reads every number at 4 dp |
| `scripts/analysis/aggnet_pregate_diag.py` | **post-hoc** oracle-ceiling diagnostic; adds no arm, promotes nothing |
| `scripts/analysis/aggnet_main_{hatemm,zh,en}_OUT.json` (+ `.log`) | main battery: 3 spaces × 5 folds × all arms and controls |
| `scripts/analysis/aggnet_perm_{hatemm,zh,en}_OUT.json` (+ `.log`) | permutation battery: fused space + the 100-draw label-shuffled null |
| `scripts/analysis/aggnet_pregate_OUT.json` | merged result — the file the record is read from |
| `scripts/analysis/aggnet_pregate_diag_OUT.json` | family oracle ceilings per dataset × space |

Read-only inputs: `data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_*.pt` (**train split only** —
`dev_seen`/`test_seen` were never opened), `data/gt/*/train.jsonl` (SECONDARY Δlength arm only),
`scripts/analysis/mechnov_pairverify_{hatemm,zh,en}_OUT.json` (parity anchors, read-only),
`scripts/analysis/vga_pregate_OUT.json` (the +0.0269 benchmark, re-read from source rather than
transcribed from the record). Read for context, not modified: `LITSWEEP6_MEMBANK.md`,
`VGA_PREGATE_RECORD.md`, `RESTRANS_PREGATE_RECORD.md`, `KSWEEP_RECORD.md`.
Nothing under `autoresearch/goal_mllm_plus3/state/` was written. No file deleted or moved.
**Zero GPU, zero SLURM submissions, zero Modal calls, zero test contact.**

---

## ⚠ ERRATUM (appended 2026-07-28, closeout) — the inherited "head memorises train at LOO ≈ 0.998" premise is a **CLIP** number

**No verdict moves.** This is a framing correction to an inherited premise, not to any measurement
taken in this record.

**The error.** This record repeats — from `mechnov_pairverify.py:21-25` / F95 — that the trained RGCL
head *"memorises its own train split (LOO train acc 0.998, F47)"*, and uses it to justify screening in
the **raw** encoder key space rather than the deployed head space.

**0.998 is F47's CLIP head, not the deployed head.** F47's own `ban_scope`
(`directions_tried.json:171`) reads *"train-supervised = memorization-degenerate target, **CLIP LOO
0.998**"*, and the memory index pairs it with *"vs **Qwen 0.800**"*. The deployed system does not use
the CLIP head.

**The deployed Qwen heads, newly computed** (`INSTRUMENT_VALIDATION_RECON.md` §0.2, F111; re-read from
`scripts/analysis/mechfix_{hatemm,zh,en}_OUT.json` → `train_side_sanity.deployed_loo_train_acc`):

| | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| **deployed head train LOO** | **0.9406** | **0.8915** | **0.8154** |
| raw-arena deployed train LOO | 0.8441 | 0.8480 | 0.7796 |
| gap between the two arenas | +0.0965 | +0.0435 | +0.0358 |

**The two arenas differ by 3.6–9.7 accuracy points on the same train items, not by the 0.998-vs-0.84
chasm the premise asserts.** The argument *"a train-side screen in head space measures memorisation"*
is therefore **weaker than stated — downgraded, not vacated**: 0.9406 against a 0.8441 raw floor still
means the head reproduces its own train split far better than its deployed test behaviour.

**CONSEQUENCE 1 — the raw-space screening justification is superseded.** The saturation claim applies
**only to full-train LOO**. `HEADSPACE_TRANSFER_PREGATE.md` (F113) demonstrates the fix nobody used:
**train the head on 4/5 of the train split and query it with the held-out fifth.** That **fold-head
arena is unsaturated**, is a strictly better proxy for deployment than the raw arena, and costs
**~35 s of CPU per fold-head**. The existing `mechfix_ops` / `vsw_pregate` battery runs in it
unmodified. **The head space was available the whole time**, and F113 recommends it become the default
`$0` pregate arena.

**CONSEQUENCE 2 — F107's Q1 argument depended on this figure and has been adjudicated.**
`HEADCOV_PREGATE_RECORD.md` §6.1 claimed *"the objective is already at its optimum on its own training
signal, with ≤0.002 of headroom"*. On the corrected figures the remaining train-side headroom is
**0.0594 / 0.1085 / 0.1846** — 30× / 54× / 92× larger. That step is **RETRACTED**; F107's conclusion
(the metric channel is closed) **survives but is SCOPED and WEAKENED — it is now empirical, not
analytic**, resting on the F75/NCA isomorphism (a measured GPU negative) plus a weak observational
conversion bound (R² = 0.027, MHC-ZH dev only) plus F113's head-space fitting evidence. **F107 must no
longer be cited as a theory-level door-closer.** See `HEADCOV_PREGATE_RECORD.md` §6.1 ERRATUM in full.

**Provenance note.** `scripts/analysis/mechnov_pairverify.py:21-25` still carries the wrong premise and
has been **deliberately left byte-identical**: its sha256 `77b0defd…b7240d` is asserted at run time by
five scripts, so editing even a comment would break the reproducibility of F95, F97, F98, F105, F112
and F113 at once. The correction lives in `MECHNOV_PAIRVERIFY_PREGATE.md` §E.1–E.3.

*Authority: `INSTRUMENT_VALIDATION_RECON.md` §0.2 (F111) · `HEADSPACE_TRANSFER_PREGATE.md` §8 (F113).
Ledger: F114. `$0` — no GPU, no SLURM, no Modal, no training, no test contact.*
