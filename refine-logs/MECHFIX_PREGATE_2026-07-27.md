# MECHFIX PREGATE — eval-time vote-operator replacements vs the deployed kNN vote

**Date:** 2026-07-27 NZST · **Agent:** mechfix pregate · **Cost: $0** (CPU only, ≤8 threads,
**zero GPU, zero SLURM, zero Modal**). Repo sha at freeze time `ad56a62` (working tree dirty;
this record and the `mechfix_*` scripts are the only things it commits).

**What this is.** A $0 pregate on five *eval-time decision-operator* replacements for the deployed
top-20 rank-weighted signed-cosine kNN vote, run on banked features with existing proxy/snapshot
heads. Nothing is trained. Nothing is tuned. Every arm is a global, symmetric operator applied
identically to all items.

**What this is not.** Not a formal verdict, not a prereg, not a promotion. Heads are proxies
(HateMM, MHC-ZH) or snapshot recomputes (MHC-EN); the **primary claim object is the paired
same-head Δ**, never the absolute number, and the proxy is never presented as the deployed floor.

---

## §1. FROZEN ARMS AND DISTINCTNESS AUDIT

### 1.1 The diagnosis being treated

From `refine-logs/ERRPAT_{HateMM,MHC-EN,MHC-ZH}_2026-07-26.md`, all three read in full before any
arm was written. The deployed decision (`src/utils/metrics.py:262-301`,
`src/model/evaluate_rac.py:405-465`) is

```
keys      = mlp[:-2]( normalize(img_proj(x_img)) * normalize(text_proj(x_txt)) )     # fused head embedding
retrieval = faiss.IndexFlatIP over float32 L2-normalised keys, memory = own train split, top-20
vote      = Σ_i (2·lab_i − 1)·cos_i·w_i / Σ_i w_i,   w = [20, 19, …, 1]
decision  = predict 1 iff sigmoid(vote) ≥ 0.5  ⟺  vote ≥ 0
```

and the residual errors are **confident neighbourhood inversions**, not boundary cases:

* ZH: median top-20 purity toward the true label **0.15**, median |vote| **0.7137**, and the first
  same-gold-class train neighbour sits at **median rank 1.5** in the raw fused space (11 of 22 at
  rank 1) — the right analogue is retrieved and then out-voted (ERRPAT-ZH §3).
* HateMM: median top-1 neighbour cosine **0.999852** for errors vs **0.999976** for correct items —
  the space is collapsed onto a narrow cone, so distance carries no dynamic range (ERRPAT-HateMM §2).
* HateMM: retrieval is length-organised, Spearman ρ(query words, median words of its top-20)
  = **0.5817** (p = 7.4e-21), and the bank's hate base rate runs 0.1096 → 0.5538 across word bins,
  so the vote inherits the bank's **length-conditional class prior** (ERRPAT-HateMM §4.3).
* ZH: the trained head **sharpens** the inversion — core-error purity raw-fused 0.400 → head
  **0.1167**, while correct items sharpen 0.85 → 0.9833 (ERRPAT-ZH §3).

Each arm below attacks one named term of that mechanism.

### 1.2 The five arms (frozen configs — literature defaults, no tuning anywhere)

Implementation: `scripts/analysis/mechfix_ops.py`,
**sha256 `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d`** (197 lines), frozen
before any treatment number was computed. All arms operate in the **deployed head key space**, at
**eval time only**, applied **symmetrically to every item**, with **no test label** and **no
per-item branch**.

| arm | operator | frozen constants | mechanism term it removes |
|---|---|---|---|
| **T1** class-balanced vote | retrieve top-10 of **each** class separately from the train bank; `score_c = Σ_{i=1..10} w_i·cos_i / Σw`, `w=[10..1]`; predict hate iff `score_hate ≥ score_nonhate` | k=10 per class (total budget 20 = deployed) | the neighbourhood's **local class prior**: per-class neighbour count is fixed 10/10 for every item, so the bank's length-conditional base rate cannot enter |
| **T2a** CSLS hubness correction | `r(x)` = mean cosine of bank item `x` to its 10 nearest **other** bank items (bank-side, train-only, precomputed); `adjsim(q,x) = 2·cos(q,x) − r(x)`; top-20 under `adjsim`; then the deployed signed vote **using the adjusted sims** | hub neighbourhood k=10 (Lample et al. 2018 default) | **hubness**: bank items that are everyone's neighbour are penalised continuously |
| **T2b** whitened keys | mean-centre bank+query keys with the **train** bank mean; whiten with the Ledoit-Wolf shrinkage covariance (`sklearn.covariance.LedoitWolf`, fit on train bank only); L2-renormalise; deployed vote unchanged | LW shrinkage is chosen in closed form by the estimator — **no free parameter** | **cone collapse**: restores dynamic range to the cosine |
| **T3** length-direction excision | `v` = least-squares regression direction of `log(1+transcript_volume)` on train-bank keys (train only, 1 dimension, **no labels**); project `v̂` out of bank+query keys; L2-renorm; deployed vote unchanged | 1 direction; `log1p`; min-norm lstsq (d>n) | the **length axis** in the retrieval geometry |
| **T4** combo | T2b whitening, then the T1 class-balanced vote | as above | cone collapse + local class prior together |

Transcript volume for T3 is read from `data/gt/<DS>/{train,val,test}.jsonl` `"text"` —
whitespace-token count for HateMM and MHC-EN, **character count of the composed text for MHC-ZH**
(per tasking; ZH text is `Title + " . " + Transcript` and whitespace tokens are not meaningful).

**Engine uniformity.** Every arm draws its cosines from the same `faiss.IndexFlatIP` search over
float32 L2-normalised keys that the deployed path uses, so no arm-to-arm Δ can be an artefact of a
different similarity engine. T1 and T2a, whose ranking is not plain cosine, take the full faiss
similarity row and rank within it.

### 1.3 Structural-distinctness audit (mandatory)

The house rule is that a re-run of a measured-dead lever is not a new measurement. Each arm is
argued against every dead lever it could be confused with.

**vs. global decision threshold (dead on all 3 datasets: HateMM dev-fitted +0.0000/+0.0016
ERRPAT-HateMM §2.1; ZH test-fitted oracle only +0.0201 ERRPAT-ZH §3; EN dev-selected −0.0083 on
0/6 arms ERRPAT-EN §6.2).** A threshold re-prices one fixed scalar by a constant and cannot change
which items are ranked above which. **T1** computes a different scalar that is *not a monotone
function of the deployed vote*: two items with an identical deployed vote receive opposite T1
decisions depending on how their top-20 splits by class, because T1 fixes that split at 10/10.
**T2a, T2b, T3, T4** change the retrieved *set* itself. None of the five can be written as
`vote ≥ τ` for any τ. DISTINCT.

**vs. train-LOO logistic recalibration of the vote (dead: −0.0016 acc / −0.0017 mF1,
ERRPAT-HateMM §2.1).** Same argument: that is a monotone 1-D transform of a fixed score. All five
arms change the score's inputs, not its calibration. DISTINCT.

**vs. length-logistic score de-bias (dead: train-LOO fit −0.0016 acc, dev fit +0.0000,
ERRPAT-HateMM §4.3/§6.2).** This is the closest call and the audit's load-bearing entry. The dead
lever fits `logistic(vote, log(1+n_words))` — a monotone reweighting of the **final scalar**, applied
*after* retrieval. **T3 acts on the geometry, before retrieval**: it excises the length direction
from the keys, so the *retrieved neighbour set changes*. The ERRPAT-HateMM report states the reason
this distinction is the whole point: "the bias lives in the retrieval geometry, not in a monotone
miscalibration of the score" (§4.3). T3 is the untested arm that sentence points at. DISTINCT — and
the distinctness is exactly what is under test.

**vs. LOO bank curation (measured null and fails its own random-deletion control: curated +0.0016
vs random +0.0031/+0.0000, ERRPAT-HateMM §6.1; EN 14-id prune +0.0093 sub-bar, ERRPAT-EN §6.5).**
Curation **deletes** bank rows. No arm here deletes anything: T1 keeps the whole bank and imposes a
per-class retrieval quota; T2a down-weights hubs **continuously** (a hub is demoted, never removed);
T2b/T3 are linear maps of the key space with the bank intact. T1 in particular is the *opposite*
move — it guarantees that both classes are represented rather than removing rows. DISTINCT.

**vs. the NCA / soft-kNN trained loss family (F75, 7/8 cells KS-arm-dead).** F75 changed the
*training objective* so the learned space would suit the kNN vote. Every arm here is **eval-time,
zero training, zero gradient steps**; T2b/T3's parameters are closed-form functions of the frozen
train bank (a covariance and a least-squares direction), and T1/T2a have no fitted parameters at
all. DISTINCT.

**vs. per-item selection / routing (F47 dead at all 3 supervision sources; F66: 91-98 % of oracle
headroom formally selection-locked).** All five arms are **global symmetric operators**: the same
transform and the same decision rule for every item, no channel choice, no branch, no oracle, no
test label. Nothing in any arm is conditioned on the item's identity, its margin, or its label.
DISTINCT — and this is the property that makes them legal at all.

**vs. the fusion-operator axis (F83/F85 concat null; F50 fixed composition).** Those change how the
two streams are combined. All five arms act on the **already-fused deployed key** and never touch
the stream combination. DISTINCT.

**No arm was dropped.** All five clear the audit.

### 1.4 What would count as a pass

House bar: **+0.030 acc AND +0.030 mF1, 3/3 seeds**, on ≥1 dataset, to earn promotion to the formal
prereg ceremony. Anything positive but below that is reported as a **sub-bar positive** and is not
dressed up. The ±0.014 project seed-noise band is quoted where relevant.

---

## §2. HARNESS, AND THE PARITY GATES IT MUST CLEAR FIRST

### 2.1 The paired same-head design (this is the load-bearing part)

For each dataset × seed × protocol: load **the same head the errpat analysis used**, recompute the
train-bank and test-query embeddings from the banked feature caches, compute the **deployed** vote
and **every treatment** vote *from those same embeddings*, and report

```
Δ(arm) = metric(arm) − metric(deployed)          per seed, acc and macro-F1
```

Both arms of every comparison therefore share the head, the features, the memory bank, the
similarity engine and the floating-point path; the only difference is the operator. Absolute
accuracies are proxy-grade and are reported only as context. **The Δ is the claim object.**

### 2.2 Per-dataset artifacts, protocols and parity targets

| dataset | head basis | memory bank | protocols | parity target (must match at 4 dp before any treatment runs) |
|---|---|---|---|---|
| **HateMM** (test n=215) | errpat CPU **proxy** heads, `<scratch>/errpat/…/ckpt/epoch_model_{25,15,29}` (val-sel) and `epoch_model_29` (final), per seed 0/1/2 | own train, V=744 | val-sel (proxy exact vs floor at 4 dp) + final | proxy trainlog: val-sel 0.8791/0.8730, 0.8744/0.8684, 0.8791/0.8730; final 0.8698/0.8632, 0.8791/0.8735, 0.8791/0.8730 (ERRPAT-HateMM §0.2) |
| **MHC-ZH** (test n=149) | errpat CPU **re-mint** heads, `logging/Retrieval/MHC_zh/errpat_zh_remint_v2/…/ckpt/epoch_model_29` (primary) and the re-mint's own dev-argmax epoch (secondary) | own train, V=579 | **PRIMARY = final-epoch** (the only device-reproducible readout, ERRPAT-ZH §0.2); val-sel-epoch reported secondary with the proxy caveat | re-mint dump ep29: 0.8456/0.8158, 0.8389/0.8090, 0.8523/0.8226 (`errpat_remint_dumps/errpat_zh_remint_seed{0,1,2}.pkl`) |
| **MHC-EN** (test n=161) | **ARM-F snapshot** heads `refine-logs/router_ckpt_snapshot/MHC_Qwen_s{0,1,2}_e29.pt` | own train, V=549 | final-epoch, 3 seeds | primary trainlog anchors 0.8012/0.7596, 0.7702/0.7203, 0.7826/0.7475 (ERRPAT-EN §2.2) |

**Scope limits, stated up front.**
* MHC-EN **ARM-V** (the master-table headline val-selected stack, 4 seeds) **cannot** support these
  operators: only its top-60 neighbour lists are banked, its head ckpt is gone, and T1/T2a/T2b/T3
  all need full-bank similarities or a re-projection. EN is therefore measured on ARM-F only, at
  3 seeds, final-epoch. This is a scope limit of the artifact, not a choice.
* HateMM and MHC-ZH heads are CPU proxies. Their val-sel 3-seed mean matches the HateMM floor
  exactly at 4 dp; the ZH re-mint matches banked ep29 accuracy at 4 dp on all 3 seeds. The Δ design
  makes proxy-vs-floor offset irrelevant to the claim, since both arms sit on the same head.
* ZH's val-sel-epoch read uses the **re-mint's own** dev argmax (same-path selection), not the
  banked run's epoch, because mixing a GPU-selected epoch with a CPU head would compare two
  different paths.

### 2.3 Gate order (test-touch discipline)

1. §1 + §2 of this record written and the operator sha256 frozen **before any treatment number
   exists**. ✔ (this section)
2. **Floor parity gates**: the deployed-vote reproduction must match the recorded values above at
   4 dp, per dataset per seed per protocol, as hard `assert`s that abort the run. Results → §2.4.
3. **Train-side sanity** per arm (train items only, LOO, no test): T1 must not collapse to one
   class; T2a's `r(x)` must have spread; T3's direction must correlate with length as intended;
   T2b's whitener must actually de-collapse the cone. **Recorded, never used to tune** — the configs
   are already frozen. → §3.
4. Test reads: 5 arms × 3 datasets × 3 seeds × protocols as scoped. Dev/val reads are computed too,
   as free same-head corroboration (no selection is made on them). → §4.
5. Core-error flip/break accounting against the errpat stable-core id lists. → §5.
6. Verdict per arm vs the house bar. → §6. Limitations → §7.

Scripts: `scripts/analysis/mechfix_ops.py` (frozen operators),
`scripts/analysis/mechfix_run.py` (harness). Machine outputs:
`scripts/analysis/mechfix_{hatemm,zh,en}_OUT.json`. Every number in §2.4-§6 is re-read from those
JSONs at report time, 4 dp.

<!-- RESULTS BELOW THIS LINE WERE WRITTEN AFTER THE GATES RAN -->

### 2.4 Floor parity gate results — 15/15 PASS on test, 15/15 PASS on dev

Re-read from `mechfix_{hatemm,zh,en}_OUT.json` → `floor_parity`. Every cell is a hard `assert`; the
run aborts on any mismatch. `ops_sha256` recorded inside all three JSONs is
`635c1312…c83fc8d`, i.e. the frozen file.

| dataset | cell | epoch | anchor acc / mF1 | recomputed acc / mF1 | test | dev anchor → recomputed |
|---|---|---|---|---|---|---|
| HateMM | valsel s0 | 25 | 0.8791 / 0.8730 | 0.8791 / 0.8730 | **PASS** | 0.8505 → 0.8505 PASS |
| HateMM | valsel s1 | 15 | 0.8744 / 0.8684 | 0.8744 / 0.8684 | **PASS** | 0.8505 → 0.8505 PASS |
| HateMM | valsel s2 | 29 | 0.8791 / 0.8730 | 0.8791 / 0.8730 | **PASS** | 0.8505 → 0.8505 PASS |
| HateMM | final s0 | 29 | 0.8698 / 0.8632 | 0.8698 / 0.8632 | **PASS** | 0.8505 → 0.8505 PASS |
| HateMM | final s1 | 29 | 0.8791 / 0.8735 | 0.8791 / 0.8735 | **PASS** | 0.8037 → 0.8037 PASS |
| HateMM | final s2 | 29 | 0.8791 / 0.8730 | 0.8791 / 0.8730 | **PASS** | 0.8505 → 0.8505 PASS |
| MHC-ZH | final s0 | 29 | 0.8456 / 0.8158 | 0.8456 / 0.8158 | **PASS** | 0.8462 → 0.8462 PASS |
| MHC-ZH | final s1 | 29 | 0.8389 / 0.8090 | 0.8389 / 0.8090 | **PASS** | 0.8333 → 0.8333 PASS |
| MHC-ZH | final s2 | 29 | 0.8523 / 0.8226 | 0.8523 / 0.8226 | **PASS** | 0.8462 → 0.8462 PASS |
| MHC-ZH | valsel s0 | 5 | 0.7987 / 0.7695 | 0.7987 / 0.7695 | **PASS** | 0.8718 → 0.8718 PASS |
| MHC-ZH | valsel s1 | 19 | 0.8322 / 0.8023 | 0.8322 / 0.8023 | **PASS** | 0.8718 → 0.8718 PASS |
| MHC-ZH | valsel s2 | 6 | 0.8188 / 0.7958 | 0.8188 / 0.7958 | **PASS** | 0.8718 → 0.8718 PASS |
| MHC-EN | final s0 | 29 | 0.8012 / 0.7596 | 0.8012 / 0.7596 | **PASS** | 0.7625 → 0.7625 PASS |
| MHC-EN | final s1 | 29 | 0.7702 / 0.7203 | 0.7702 / 0.7203 | **PASS** | 0.7875 → 0.7875 PASS |
| MHC-EN | final s2 | 29 | 0.7826 / 0.7475 | 0.7826 / 0.7475 | **PASS** | 0.7750 → 0.7750 PASS |

The EN dev anchors 0.7625 / 0.7875 / 0.7750 are the ones ERRPAT-EN §2.2 cites from the ckpt-filename
suffixes, reproduced here independently. HateMM's proxy val-sel epochs recomputed from the proxy
trainlog dev argmax (warmup ≥ 5) come out {25, 15, 29}, asserted equal to the errpat record.

Split sizes re-read at report time: HateMM train 744 / dev 107 / test 215 (86 pos, train pos-rate
0.4005); MHC-ZH 579 / 78 / 149 (45 pos, 0.3109); MHC-EN 549 / 80 / 161 (49 pos, 0.3060).

---

## §3. TRAIN-SIDE SANITY (train items only, LOO; recorded, used to tune nothing)

From `train_side_sanity` in each OUT json (final-epoch cells shown; the val-sel cells agree) and
`mechfix_diag_OUT.json`. All three checks the tasking asked for were run, and **two of them came back
telling us the arm is inert before we ever looked at a test number** — which is the most useful thing
this section could have done.

| check | HateMM (s0/s1/s2) | MHC-ZH | MHC-EN | reading |
|---|---|---|---|---|
| **T1 collapses to one class?** | **No** — LOO pos-rate 0.4019 vs bank 0.4005 | No | No | T1 is well-behaved: it predicts both classes at the bank's own base rate |
| T1 LOO train acc vs deployed LOO | 0.9476 vs **0.9476** | — | — | *identical*, first sign of the degeneracy in §3.1 |
| **T2a `r(x)` spread** | IQR **1.38e-4 / 1.24e-4 / 1.02e-4**, range [0.9829, 0.999994] | IQR 2.4-3.4e-4 | IQR 3.8-8.8e-4 | the hubness term has **almost no dynamic range** — the cone collapse leaves nothing to correct with |
| **T3 direction encodes length?** | Pearson(proj, log-len) = **1.0000** on train, residual after removal 4.3e-9 | 1.0000 | 1.0000 | the direction is fitted and excised exactly — *and* see §3.2 |
| T2b LW shrinkage / eigen-condition | 0.00041 / **2.4e6** | 0.0011 / 9.0e5 | 0.0027 / 3.5e5 | at d=1024 > n the closed-form shrinkage is ~0, so the whitener amplifies the smallest eigendirections ~1800× |
| T2b de-collapses the cosine? | train top-1 sim **0.9999 → 0.5220** | 0.9999 → 0.53-0.61 | 0.9996 → 0.39-0.46 | yes, decisively |

### 3.1 T1 is not merely null — it is the *same classifier* (measured, with a coding control)

On HateMM and MHC-ZH, T1's prediction equals the deployed prediction on **215/215** and **149/149**
items, in every seed. On MHC-EN it differs on 1 / 4 / 0 items of 161.

This is not an implementation artefact. An **independent float64 numpy re-implementation** of T1 with
no faiss in the loop (`mechfix_diag.py:t1_numpy_control`) agrees with the frozen faiss implementation
on **161/161, 149/149, 215/215** predictions in all 9 cells, max margin difference 2.2e-7.

The reason is visible in the per-class scores: the class whose top-10 sits at cosine ≈ 0.9999 wins,
and the losing class's top-10 sits materially lower (HateMM s0 median class-0 score **0.999890** vs
class-1 **0.994208**). The gap *between* classes is two orders of magnitude larger than the
within-class dispersion, so `sign(score_hate − score_nonhate)` and the deployed rank-weighted signed
sum over the mixed top-20 are **the same statistic on this data**. Both also agree with the plain
top-1-neighbour label rule on 204/215 items.

**Consequence for the diagnosis.** The errpat reports describe the vote as inheriting "the bank's
local class prior". T1 was the arm designed to remove that term by construction. It turns out the
term is not separable: fixing the neighbour count at 10/10 per class does not change any decision,
because what decides the vote is *which class owns the very top of the ranking*, and a per-class
quota cannot change that. The local class prior is not a removable bias sitting on top of the
retrieval signal — in this geometry it **is** the retrieval signal.

### 3.2 T3 excises the direction exactly, and the length organisation does not move

The frozen T3 direction is fitted and removed to numerical precision (Pearson 1.0000 before,
|residual projection| ≤ 8.6e-9 after). But two measurements show the excision cannot reach the
mechanism it targets:

1. The fitted direction carries **essentially none of the key variance**: variance share
   **6.4e-15 / 7.2e-15 / 3.4e-15** (HateMM), 3.6-6.2e-14 (ZH), 8.4e-14 to 5.9e-13 (EN). It is a
   numerically-null direction of the key covariance.
2. The **length organisation of retrieval is unchanged**. Measuring the errpat statistic directly —
   Spearman ρ(query transcript volume, median volume of its top-20 retrieved bank rows) — before and
   after excision:

| dataset | deployed ρ (s0/s1/s2) | after T3 excision | after T2b whitening |
|---|---|---|---|
| HateMM | 0.5805 / 0.5005 / 0.4924 | **0.5808 / 0.5006 / 0.4965** | **0.8770 / 0.8489 / 0.8686** |
| MHC-ZH | 0.3667 / 0.4813 / 0.3986 | **0.3671 / 0.4805 / 0.4005** | **0.8134 / 0.8043 / 0.8154** |
| MHC-EN | 0.1794 / 0.2939 / 0.1717 | **0.1813 / 0.2947 / 0.1686** | **0.6629 / 0.6619 / 0.6973** |

(The deployed HateMM s0 value 0.5805 independently reproduces ERRPAT-HateMM §4.3's ρ = 0.5817 on the
proxy head.) Excising the best linear length predictor moves ρ by ≤ 0.004. **The length
organisation of retrieval is not carried by any single linear direction of the key space** —
so the 1-D excision specified for T3 is structurally incapable of removing it, and its null is
explained rather than merely observed.

The third row is the surprise, and it drives §6's T2b verdict: **whitening raises the length
organisation sharply** (HateMM 0.52 → 0.87 mean). Whitening amplifies low-variance directions, and
the length axis is a low-variance direction, so the literature-default whitener promotes exactly the
nuisance axis the diagnosis flagged.

---

## §4. RESULTS — paired same-head Δ on test

All Δ are `treatment − deployed` on the **identical head**, per seed, re-read from
`means_3seed` in the OUT jsons. Sign patterns are per seed (s0 s1 s2). House bar =
**+0.030 acc AND +0.030 mF1, 3/3 seeds**.

### 4.1 Primary protocols

| arm | HateMM final-ep (n=215) Δacc / ΔmF1 | signs | MHC-ZH final-ep (n=149) Δacc / ΔmF1 | signs | MHC-EN final-ep (n=161) Δacc / ΔmF1 | signs |
|---|---|---|---|---|---|---|
| deployed (floor) | *0.8760 / 0.8699* | — | *0.8456 / 0.8158* | — | *0.7847 / 0.7425* | — |
| **T1** class-balanced | **+0.0000 / +0.0000** | 000 / 000 | **+0.0000 / +0.0000** | 000 / 000 | **+0.0021 / +0.0031** | −+0 / −+0 |
| **T2a** CSLS | **+0.0000 / +0.0000** | 000 / 000 | **+0.0000 / +0.0000** | 000 / 000 | **−0.0021 / −0.0032** | −00 / −−0 |
| **T2b** whitening | **−0.0078 / −0.0097** | −−0 / −−− | **+0.0000 / −0.0053** | +−− / +−− | **−0.0104 / −0.0395** | −+− / −+− |
| **T3** length excision | **+0.0000 / +0.0000** | 000 / 000 | **+0.0000 / +0.0000** | 000 / 000 | **+0.0000 / +0.0000** | 000 / 000 |
| **T4** whiten + balanced | **−0.0046 / −0.0060** | −−+ / −−+ | **+0.0067 / +0.0052** | +0+ / +−+ | **+0.0041 / −0.0122** | −+0 / −+− |

Per-seed Δacc for the arms that moved at all:
HateMM final T2b [−0.0047, −0.0186, +0.0000], T4 [−0.0093, −0.0093, +0.0047];
ZH final T2b [+0.0134, −0.0067, −0.0067], T4 [+0.0134, +0.0000, +0.0067];
EN final T1 [−0.0062, +0.0124, +0.0000], T2b [−0.0435, +0.0248, −0.0124], T4 [−0.0062, +0.0186, +0.0000].

**0 of 15 arm×dataset cells clears the bar. The largest 3-seed mean anywhere is T4 on MHC-ZH,
+0.0067 acc / +0.0052 mF1 — 4.5× under the bar, inside the ±0.014 seed band, and not 3/3 on mF1.**
**[ANNOTATION 2026-07-28 (F113/F114): a second, independent head-space arena (train-fold heads, 3 seeds)
gives **−0.0063** for the same cell — same magnitude, opposite sign ⇒ seed/arena noise. Verdict
unchanged.]**

### 4.2 Secondary protocols

HateMM val-selected (the protocol on which the proxy is exact at 4 dp vs the floor; deployed 3-seed
0.8775 / 0.8715):

| arm | Δacc / ΔmF1 | signs (acc) | per-seed Δacc |
|---|---|---|---|
| T1 | +0.0000 / +0.0000 | 000 | [0.0000, 0.0000, 0.0000] |
| T2a | +0.0000 / +0.0000 | 000 | [0.0000, 0.0000, 0.0000] |
| T2b | **−0.0155 / −0.0187** | −−0 | [−0.0186, −0.0279, 0.0000] |
| T3 | +0.0000 / +0.0000 | 000 | [0.0000, 0.0000, 0.0000] |
| T4 | −0.0062 / −0.0080 | −−+ | [−0.0093, −0.0140, +0.0047] |

MHC-ZH val-sel-epoch (secondary, **proxy caveat applies**: ERRPAT-ZH §0.2 established that the
val-selected readout is not reproducible across a device change, and the re-mint's own dev argmax
lands at epochs 5/19/6, not the banked 20/26/19; deployed 3-seed here 0.8166 / 0.7892):

| arm | Δacc / ΔmF1 | signs (acc) | per-seed Δacc |
|---|---|---|---|
| T1 | +0.0000 / +0.0008 | +0− | [+0.0067, 0.0000, −0.0067] |
| T2a | −0.0022 / −0.0109 | +0− | [+0.0201, 0.0000, −0.0268] |
| T2b | −0.0112 / −0.0455 | +−− | [+0.0201, −0.0268, −0.0268] |
| T3 | +0.0022 / +0.0022 | 00+ | [0.0000, 0.0000, +0.0067] |
| T4 | −0.0112 / −0.0303 | 0−− | [0.0000, −0.0067, −0.0268] |

This protocol is where the arms look most active and most random — up to 20 items flipped per seed
with signs in both directions — which is the per-item face of ERRPAT-ZH §1: an epoch chosen off a
78-item dev set is sitting in a lottery, so operator Δ measured on top of it is mostly epoch noise.
It is reported for completeness and carries no weight.

### 4.3 Dev-side corroboration (free, same head, no selection made on it)

Not a legality requirement — the configs were frozen before any read — but a same-head dev Δ that
agrees with the test Δ is evidence against a single unlucky test draw. 3-seed mean Δacc on dev:

| arm | HateMM final (n=107) | MHC-ZH final (n=78) | MHC-EN final (n=80) |
|---|---|---|---|
| T1 | +0.0000 | +0.0000 | +0.0042 |
| T2a | +0.0000 | +0.0000 | +0.0042 |
| T2b | **−0.0062** | **+0.0128** (3/3 +) | **−0.0667** |
| T3 | +0.0000 | +0.0000 | +0.0000 |
| T4 | −0.0062 | +0.0043 | **−0.0625** |

Dev agrees with test on the negative verdicts for T2b/T4 on HateMM and EN, and strongly so on EN
(−0.0667 / −0.0625 with 3/3 negative signs). The one disagreement is ZH-final T2b: dev is **+0.0128
with 3/3 positive signs** while test is +0.0000 acc / −0.0053 mF1. At 78 dev items one item is
0.0128, so the entire dev signal there is one video per seed; it does not survive contact with the
149-item test split. Recorded, not promoted.

---

## §5. CORE-ERROR FLIP / BREAK ACCOUNTING

This is where the arms tie back to the mechanism story. Counts are per seed
(s0 / s1 / s2), primary protocol. "core" = the errpat stable-core error id lists:
HateMM 25 items wrong in 3/3 seeds (final) from `errpat_hatemm_forensics_OUT.json`
(27 at ≥2/3 from `errpat_hatemm_peritem.csv`); MHC-ZH 22 items wrong in 3/3 from
`errpat_zh_taxonomy_OUT.json`; MHC-EN the documented 22-item ARM-V 4/4 consensus set from
`errpat_mhc_en_out.json:consensus_error_ids` (ARM-V is a different key space from the ARM-F arm
measured here, so this is an overlap read, not a same-arm core).

| dataset | arm | errors fixed | previously-correct broken | **net items** | of the fixed, in the documented core |
|---|---|---|---|---|---|
| HateMM final (28/26/26 deployed errors) | T1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 / 0 / 0 |
| | T2a | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 / 0 / 0 |
| | **T2b** | 3 / 1 / 2 | 4 / 5 / 2 | **−1 / −4 / 0** | 2 / 0 / 1 |
| | T3 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 / 0 / 0 |
| | **T4** | 3 / 2 / 3 | 5 / 4 / 2 | **−2 / −2 / +1** | 2 / 1 / 2 |
| MHC-ZH final (23/24/22) | T1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 / 0 / 0 |
| | T2a | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 / 0 / 0 |
| | **T2b** | 2 / 3 / 2 | 0 / 4 / 3 | **+2 / −1 / −1** | 1 / 1 / 2 |
| | T3 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 / 0 / 0 |
| | **T4** | 2 / 2 / 3 | 0 / 2 / 2 | **+2 / 0 / +1** | 1 / 1 / 3 |
| MHC-EN final (32/37/35) | T1 | 0 / 3 / 0 | 1 / 1 / 0 | −1 / +2 / 0 | 0 / 0 / 0 |
| | T2a | 0 / 2 / 0 | 1 / 2 / 0 | −1 / 0 / 0 | 0 / 0 / 0 |
| | **T2b** | 6 / 11 / 8 | 13 / 7 / 10 | **−7 / +4 / −2** | 3 / 2 / 2 |
| | T3 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 / 0 / 0 |
| | **T4** | 8 / 10 / 9 | 9 / 7 / 9 | **−1 / +3 / 0** | 5 / 3 / 2 |

Three readings.

1. **T1, T2a and T3 flip literally nothing** on HateMM and MHC-ZH, and near-nothing on EN. Their
   nulls are not "small effects lost in noise" — they are **inert operators**, each for a measured
   reason (§3.1, §3.2, and T2a's 1e-4 hubness range). Notably T2a changes the retrieved top-20 *set*
   on 88-114 of 215 HateMM items and 98-114 of 161 EN items while flipping 0-4 decisions: the
   neighbours it swaps are label-interchangeable.
2. **T2b and T4 do reach the hard core** — they fix 1-3 (HateMM/ZH) and 2-5 (EN) of the documented
   stable-core errors, which no measured lever in the errpat reports managed. But they break at
   least as many previously-correct items in almost every cell. Net is ≤ 0 in **12 of the 18**
   T2b/T4 seed cells and the 3-seed mean Δ is negative or ~0 on every dataset.
3. That fix-some/break-more shape is the **same arithmetic** F47/F66 priced for per-item channel
   selection and ERRPAT-HateMM §3.2 measured for the image stream (fixes 11-14, breaks 40-43). A
   global geometric operator lands in the same place: the information to fix the core exists in the
   space, and every symmetric way of surfacing it costs at least as much elsewhere. Realising it
   would again require the per-item selection the campaign has closed.

---

## §6. VERDICT PER ARM vs THE HOUSE BAR

House bar for promotion to formal prereg ceremony: **+0.030 acc AND +0.030 mF1, 3/3 seeds, on ≥1
dataset**. Nothing in this battery is close.

| arm | 3-seed mean Δacc / ΔmF1 (primary protocol) | verdict |
|---|---|---|
| **T1** class-balanced vote | HateMM +0.0000/+0.0000 · ZH +0.0000/+0.0000 · EN +0.0021/+0.0031 | **NULL — degenerate.** Identical predictions to the deployed vote on 215/215 (HateMM) and 149/149 (ZH); 1/3 seeds positive on EN. The class-prior term it was built to remove is not separable from the retrieval signal (§3.1). Verified against an independent numpy re-implementation. |
| **T2a** CSLS hubness | HateMM +0.0000/+0.0000 · ZH +0.0000/+0.0000 · EN −0.0021/−0.0032 | **NULL — inert.** `r(x)` spans an IQR of 1.0e-4 to 8.8e-4; the cone collapse leaves the hubness statistic no dynamic range. Changes half the retrieved sets, flips 0-4 decisions, mean sign negative on EN. |
| **T2b** whitened keys | HateMM −0.0078/−0.0097 · ZH +0.0000/−0.0053 · EN −0.0104/−0.0395 | **NEGATIVE.** The only arm that reliably changes decisions, and it changes them for the worse: negative 3-seed mean mF1 on all three datasets, dev-corroborated negative on HateMM and EN (−0.0062, −0.0667). Mechanism measured: at d=1024 > n the closed-form shrinkage is ~0.001-0.003, so the whitener amplifies near-null eigendirections ~1000× and **raises** the length organisation of retrieval from ρ≈0.52 to ρ≈0.87 (HateMM). It de-collapses the cone as intended and promotes the nuisance axis while doing it. |
| **T3** length-direction excision | HateMM +0.0000/+0.0000 · ZH +0.0000/+0.0000 · EN +0.0000/+0.0000 | **NULL — inert, and informatively so.** The direction is fitted exactly (Pearson 1.0000) and removed exactly (residual ≤ 8.6e-9), yet ρ(length, retrieved length) moves by ≤ 0.004 in 9/9 cells and not one prediction changes in 9/9 cells. **New structural result: the length organisation of retrieval is not carried by any single linear direction of the deployed key space.** This closes the geometry-level 1-D version of the length lever, complementing the score-level version already dead at ERRPAT-HateMM §4.3/§6.2. |
| **T4** whitening + class-balanced | HateMM −0.0046/−0.0060 · ZH **+0.0067/+0.0052** · EN +0.0041/−0.0122 | **Sub-bar positive on MHC-ZH only; null-to-negative elsewhere.** The ZH read is the best number in the battery and it is +0.0067 acc / +0.0052 mF1 — 4.5× under the bar, inside the ±0.014 seed band, sign pattern +0+ on acc but +−+ on mF1 (so not 3/3), on a proxy head, with a dev Δ of +0.0043. Reported as a **sub-bar positive**, not a lead. **[ANNOTATION 2026-07-28 (F113, F114): a SECOND, independent head-space arena — train-fold heads, 3 seeds — gives **−0.0063** for this cell: same magnitude, opposite sign. This is direct evidence that the +0.0067 is seed/arena noise rather than a small real effect. This record's own reading ("sub-bar positive, not a lead", "inside the ±0.014 seed band", "not 3/3 on mF1") is confirmed, and the MECHFIX verdict — 0 of 15 cells clears — is unchanged.]** Negative on HateMM under both protocols; mF1-negative on EN. |

**Batch verdict: 0 for 5. No arm is promotable; none reaches half the bar on any dataset.** The
genuinely-open set this pregate was probing is **empty**, and it closes at $0 rather than at a queue
slot.

### 6.1 What the battery bought, given that it bought no accuracy

Three door-closers and one new structural fact, all at zero GPU:

1. **The vote-operator axis is now measured, not assumed.** Class-balancing, hubness correction and
   whitening are the three standard eval-time repairs for exactly the pathology the errpat reports
   diagnosed (local class prior, hub-proneness, cone collapse). All three are now measured on all
   three datasets under a paired same-head design with 15/15 floor-parity gates passed. Previous
   closures on this family were score-level (thresholds, logistic recalibration); this is the
   retrieval-level closure.
2. **The "local class prior" is not a separable term** (§3.1). This sharpens the errpat diagnosis:
   the vote is not a good signal contaminated by a base-rate bias that a quota can strip; the
   base-rate structure and the signal are the same statistic in a cone-collapsed space.
3. **The length organisation of retrieval is not 1-D linear** (§3.2). Excising the exact linear
   length predictor leaves ρ unchanged. Any future length-de-biasing proposal must be priced against
   this: the axis is exactly decodable and yet not excisable in one dimension.
4. **De-collapsing the cone is not free and points the wrong way.** Whitening restores cosine
   dynamic range (0.9999 → ~0.5) and simultaneously *increases* length-organisation to ρ≈0.87. Any
   future "fix the collapsed geometry" proposal inherits this: the variance ordering of the key space
   is doing useful work, and flattening it surfaces the nuisance axis first.

---

## §7. LIMITATIONS

1. **Proxy and snapshot heads, not the deployed floors.** HateMM and MHC-ZH heads are the errpat CPU
   proxies (floor ckpts deleted, F78); MHC-EN is the ARM-F snapshot recompute. The paired same-head Δ
   design makes the proxy-vs-floor offset irrelevant to the Δ, and all 15 cells reproduce their
   recorded anchors at 4 dp — but a Δ measured on a proxy head is not proof of the same Δ on the
   floor head. This is a pregate, not a verdict.
2. **MHC-EN's headline stack (ARM-V, val-selected, 4 seeds) was not measured.** Its head ckpt is gone
   and only top-60 neighbour lists are banked, which cannot support any of these operators (all need
   full-bank similarities or re-projection). EN is 3 seeds, final-epoch, ARM-F only. The EN core-error
   overlap read in §5 therefore compares against an ARM-V-defined core in a different key space.
3. **T3's spec is under-determined at d > n.** With d=1024 and n=744/579/549 the least-squares
   direction is an interpolating solution that fits log-length exactly by construction, and it lands
   in a ~1e-14-variance subspace. The measured null is a null **about that construction**. A ridge-
   regularised direction, a whitened-space direction, or a multi-dimensional length subspace were
   **not** tested and would each be a new arm needing its own freeze. What is established
   independently of the construction is the ρ-invariance in §3.2, which is a property of the
   retrieval, not of the fit.
4. **Single draw per cell.** Every number is one deterministic evaluation per seed×protocol; there is
   no resampling and no confidence interval. With 1 test item = 0.0047 (HateMM), 0.0067 (ZH),
   0.0062 (EN), every non-zero Δ in §4 is 1-7 videos.
5. **ZH's val-sel-epoch reads use the re-mint's own dev argmax** (epochs 5/19/6), not the banked run's
   (20/26/19), because pairing a GPU-selected epoch with a CPU head would compare two paths. Those
   rows are secondary and, per ERRPAT-ZH §1, sit inside an epoch lottery.
6. **T1/T2a/T3's exact zeros are a property of this geometry, not a general result.** They say these
   operators are inert on cone-collapsed 1024-d Hadamard-fused RGCL keys with these banks. They do
   not say class-balanced voting or CSLS are useless operators in general.
7. **The dev reads in §4.3 are corroboration only.** No config, epoch, threshold or arm was selected
   using them; the arms were frozen (sha256 `635c1312…`) before any read.
8. **No test-fitted quantity appears anywhere in this record.** No oracle, no per-item selection, no
   threshold search. The five arms are the five frozen configs of §1.2 and nothing else was run.

---

## §8. FILE MANIFEST

| path | contents |
|---|---|
| `scripts/analysis/mechfix_ops.py` | **frozen** operator implementations, sha256 `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` |
| `scripts/analysis/mechfix_run.py` | harness: head loading, parity gates, train-side sanity, test/dev reads, flip accounting |
| `scripts/analysis/mechfix_diag.py` | mechanism diagnostics (T1 numpy control, hubness range, length-organisation ρ before/after, whitening spectrum) |
| `scripts/analysis/mechfix_hatemm_OUT.json` | HateMM: 6 cells (2 protocols × 3 seeds) |
| `scripts/analysis/mechfix_zh_OUT.json` | MHC-ZH: 6 cells |
| `scripts/analysis/mechfix_en_OUT.json` | MHC-EN: 3 cells |
| `scripts/analysis/mechfix_diag_OUT.json` | 9 cells of mechanism diagnostics (final-epoch, 3 datasets × 3 seeds) |
| `refine-logs/MECHFIX_PREGATE_2026-07-27.md` | this record |

Read-only inputs: the three ERRPAT reports and their OUT jsons; the HateMM errpat proxy run dir
under the session scratchpad; `logging/Retrieval/MHC_zh/errpat_zh_remint_v2/*/ckpt/`;
`scripts/analysis/errpat_remint_dumps/*.pkl`; `refine-logs/router_ckpt_snapshot/MHC_Qwen_s{0,1,2}_e29.pt`;
`data/CLIP_Embedding/{HateMM,MHC,MHC_zh}/*.pt`; `data/gt/{HateMM,MHC,MHC_zh}/*.jsonl`;
`slurm/logs/enc3s_MHC_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog`;
`slurm/logs/arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed{1,2}_1227{5,6}.trainlog`.
Nothing under `autoresearch/goal_mllm_plus3/state/` was read or written. No file deleted or moved.
Zero GPU jobs, zero SLURM submissions, zero Modal calls, zero training runs.
