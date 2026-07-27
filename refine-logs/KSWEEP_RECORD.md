# KSWEEP — top-k sweep of the deployed kNN vote (2026-07-27)

**Question (from the user).** The deployed retrieval vote uses top-20 neighbours. Has anyone tried
**reducing** k to cut neighbourhood noise? No explicit sweep existed in the records. This is it.

**Status: FORENSIC.** Read-only replay of per-item neighbour lists that were **already banked and
already test-consumed** (identical basis to the three ERRPAT reports, 2026-07-26). **Zero GPU, zero
SLURM, zero Modal, zero retraining, zero new test inference.** CPU only, ≤4 threads, ~40 s wall.
No selection, tuning, or promotion decision is derived from any test-labelled quantity below; every
such quantity is labelled **ORACLE / FORENSIC**. The one deployment-legal read (dev picks k) is
computed separately in §4 and reported including where it loses.

| artefact | path |
|---|---|
| script | `scripts/analysis/ksweep_vote.py` |
| machine-readable output | `scripts/analysis/ksweep_OUT.json` |

---

## 0. VERDICT

**Reducing k does not help. On every arm k=20 sits at or above the plateau, and the plateau starts
at k≈10-15 — there is nothing in ranks 11-20 to cut.** The two directions the user's hypothesis
could have gone both fail:

* **Small k is actively harmful.** k ∈ {1,2,3} costs **−0.0157 to −0.0388 acc** (3-seed mean) on
  every one of the 6 arms, 0+/3− or 0+/4− seed signs on 5 of 6. §5 shows why: under rank weights
  `[k..1]` with descending cosines, **k ≤ 3 is algebraically a plain 1-NN classifier** (verified
  identical to the top-1 label vector in 19/19 cells).
* **The mid-range gains are sub-item oracle artefacts.** The largest forensic shared-k gain over
  the whole grid is **+0.0104 acc** (EN ARM-F at k=7 = 1.7 test items of 161); every other arm's
  best shared k is either 20 itself or worth ≤ +0.0045 (ZH final k=5 = 0.67 items of 149).
* **Deployment-legally it is negative.** Dev-selected k (§4) gives 3-seed-mean Δtest acc of
  **−0.0140** (HateMM final), **−0.0157** (ZH final), **+0.0041** (EN ARM-F) per-seed, and
  **0.0000 / 0.0000 / −0.0179 / −0.0041** under the pooled one-k-per-config rule. Dev cannot find
  the oracle k; where dev moves off k=20 it usually moves to k=3 and loses ~0.045 acc.

Nothing here approaches the +0.030-on-≥2-datasets bar; nothing is even a candidate. **Axis closed.**

---

## 1. WHAT WAS SWEPT, AND WHY IT IS EXACT

### 1.1 The vote formula (parity-gated, not assumed)

Re-read from `src/utils/metrics.py:228-231, 262-301` (`use_sim=True`,
`majority_voting="arithmetic"`, the deployed path):

```
w    = [k, k-1, ..., 1]                                  (metrics.py:229-231 with topk=k)
v    = Σ_i (2·lab_i − 1) · cos_i · w_i  /  Σ_i w_i
pred = 1  iff  sigmoid(v) ≥ 0.5   ⟺   v ≥ 0              (metrics.py:300)
```

`w` is **re-derived at each k**, as the deployed code does — not the 20-vector truncated. This
matters: `[k..1]` is not proportional to `[20..21−k]`.

### 1.2 Why truncating banked lists == re-running with `--topk k`

`--topk` is consumed in exactly three places in `src/run_rac.py`: the retrieval depth
(`largest_retrieval=args.topk`, lines 818 and 836), the vote (`topk=args.topk`, lines 828 and 846),
and the experiment-name string (line 1016). **It never enters the loss, the hard-negative miner,
the optimiser, or the memory bank.** So at a fixed (seed, epoch) the trained head is *identical*
under any k, and the top-k neighbour list is the length-k prefix of the banked top-20 list. The
similarity threshold is not binding — every banked item has `n_retrieved == 20`, asserted in the
loader. Truncation therefore reproduces a `--topk k` re-run **exactly**, not approximately.

Caveat retained honestly: changing k changes dev accuracy, hence *could* change which epoch the
val-selection rule picks. §3 holds the epoch at the deployed choice (isolating k); §4 reports the
dev-selected-k read at that same epoch. Joint (epoch, k) re-selection was not swept.

### 1.3 Data sources

| arm | n_test | seeds | protocol | source | nature |
|---|---|---|---|---|---|
| **HateMM** | 215 | 3 | val-sel + final | ERRPAT proxy dumps, `<scratch>/errpat/Retrieval/HateMM/RAC_errpat_proxy/*/{dev,test}epoch_*_retrieval_logging_dict.pkl` | **CPU PROXY** of job 13241 (floor head ckpts deleted, F78) |
| **MHC_zh** | 149 | 3 | val-sel + final | `scripts/analysis/errpat_remint_dumps/errpat_zh_remint_seed{0,1,2}.pkl` | **CPU re-mint PROXY** of job 13150 (floor ckpts deleted) |
| **MHC-EN ARM-V** | 161 | 4 | val-sel (deployed headline stack, archive-kNN α=0.25) | `scripts/analysis/p2_out/cache_MHC_s{0,1,2,3}.json`, top-**60** banked | **EXACT** (banked per-item, no proxy) |
| **MHC-EN ARM-F** | 161 | 3 | final e29, no-archive-key | recomputed from `refine-logs/router_ckpt_snapshot/MHC_Qwen_s{0,1,2}_e29.pt` | **EXACT to 4 dp** vs primary trainlogs |

Dev neighbour lists are available for HateMM (n=107), ZH (n=78) and EN ARM-F (n=80).
**EN ARM-V supports no dev read**: no dev neighbours were banked and its val-selected checkpoints
(`best_model_24_0.7875.pt` etc.) are deleted — verified absent under `logging/`. This is disclosed
rather than papered over; the EN dev read in §4 comes from ARM-F only.

---

## 2. PARITY GATE — k=20 reproduces the recorded numbers, 19/19 cells

The script **aborts** on any mismatch (`assert ok, ("PARITY GATE FAILED", ...)`). It did not abort.
Recorded values are the ERRPAT-report / banked-header numbers; replay is this script's k=20 vote.

| cell | epoch/ckpt | recorded acc / mF1 | replay acc / mF1 | 4dp |
|---|---|---|---|---|
| HateMM s0 val-sel | 25 | 0.8791 / 0.8730 | 0.8791 / 0.8730 | OK |
| HateMM s1 val-sel | 15 | 0.8744 / 0.8684 | 0.8744 / 0.8684 | OK |
| HateMM s2 val-sel | 29 | 0.8791 / 0.8730 | 0.8791 / 0.8730 | OK |
| HateMM s0 final | 29 | 0.8698 / 0.8632 | 0.8698 / 0.8632 | OK |
| HateMM s1 final | 29 | 0.8791 / 0.8735 | 0.8791 / 0.8735 | OK |
| HateMM s2 final | 29 | 0.8791 / 0.8730 | 0.8791 / 0.8730 | OK |
| MHC_zh s0 final | 29 | 0.8456 / 0.8158 | 0.8456 / 0.8158 | OK |
| MHC_zh s1 final | 29 | 0.8389 / 0.8090 | 0.8389 / 0.8090 | OK |
| MHC_zh s2 final | 29 | 0.8523 / 0.8226 | 0.8523 / 0.8226 | OK |
| MHC_zh s0 val-sel | 5 | 0.7987 / 0.7695 | 0.7987 / 0.7695 | OK |
| MHC_zh s1 val-sel | 19 | 0.8322 / 0.8023 | 0.8322 / 0.8023 | OK |
| MHC_zh s2 val-sel | 6 | 0.8188 / 0.7958 | 0.8188 / 0.7958 | OK |
| EN ARM-V s0 | best_model_24 | 0.8075 / 0.7626 | 0.8075 / 0.7626 | OK |
| EN ARM-V s1 | best_model_29 | 0.7640 / 0.7145 | 0.7640 / 0.7145 | OK |
| EN ARM-V s2 | best_model_21 | 0.7950 / 0.7505 | 0.7950 / 0.7505 | OK |
| EN ARM-V s3 | best_model_27 | 0.8075 / 0.7713 | 0.8075 / 0.7713 | OK |
| EN ARM-F s0 | 29 | 0.8012 / 0.7596 | 0.8012 / 0.7596 | OK |
| EN ARM-F s1 | 29 | 0.7702 / 0.7203 | 0.7702 / 0.7203 | OK |
| EN ARM-F s2 | 29 | 0.7826 / 0.7475 | 0.7826 / 0.7475 | OK |

Two additional gates passed silently: the HateMM per-cell anchors were cross-checked against the
proxy trainlogs re-parsed with the deployed selection rule, and the EN ARM-V k=20 vote reproduces
every banked `floor_vote` **bit-exactly** (max |Δ| = 0.0, all 4 seeds × 161 items).

**Two ZH provenance notes.** (a) ZH val-sel epochs are **5 / 19 / 6** here, not the banked 20 / 26 /
19 — this is the known re-mint argmax relocation already documented in ERRPAT MHC-ZH §1.7, so the
ZH val-sel anchors are self-derived from the dump rather than external. ZH's binding arm for this
sweep is **final-epoch**, which is externally anchored and passes. (b) HateMM and ZH are proxies;
their k-sweep *deltas* are the object of interest, and deltas are computed within-cell so the
proxy↔floor offset cancels.

---

## 3. THE SWEEP (test, mean over seeds, Δ vs k=20)

One test item = **0.0047** (HateMM, n=215), **0.0067** (ZH, n=149), **0.0062** (EN, n=161).

### HateMM — val-selected, 3 seeds, n=215

| k | mean acc | mean mF1 | Δacc | ΔmF1 | per-seed Δacc | signs |
|---|---|---|---|---|---|---|
| 1 | 0.8419 | 0.8328 | −0.0357 | −0.0387 | [−0.0372, −0.0279, −0.0419] | 0+/3− |
| 2 | 0.8419 | 0.8328 | −0.0357 | −0.0387 | [−0.0372, −0.0279, −0.0419] | 0+/3− |
| 3 | 0.8419 | 0.8328 | −0.0357 | −0.0387 | [−0.0372, −0.0279, −0.0419] | 0+/3− |
| 5 | 0.8713 | 0.8644 | −0.0062 | −0.0071 | [−0.0093, −0.0093, 0.0000] | 0+/2− |
| 7 | 0.8775 | 0.8709 | +0.0000 | −0.0006 | [0.0000, +0.0047, −0.0047] | 1+/1− |
| 10 | 0.8775 | 0.8715 | +0.0000 | +0.0000 | [0.0000, 0.0000, 0.0000] | 0+/0− |
| 15 | 0.8775 | 0.8715 | +0.0000 | +0.0000 | [0.0000, 0.0000, 0.0000] | 0+/0− |
| **20** | **0.8775** | **0.8715** | — | — | — | — |

### HateMM — final-epoch, 3 seeds, n=215

| k | mean acc | mean mF1 | Δacc | ΔmF1 | per-seed Δacc | signs |
|---|---|---|---|---|---|---|
| 1 | 0.8372 | 0.8276 | −0.0388 | −0.0424 | [−0.0326, −0.0419, −0.0419] | 0+/3− |
| 2 | 0.8372 | 0.8276 | −0.0388 | −0.0424 | [−0.0326, −0.0419, −0.0419] | 0+/3− |
| 3 | 0.8372 | 0.8276 | −0.0388 | −0.0424 | [−0.0326, −0.0419, −0.0419] | 0+/3− |
| 5 | 0.8698 | 0.8628 | −0.0062 | −0.0071 | [−0.0047, −0.0140, 0.0000] | 0+/2− |
| 7 | 0.8760 | 0.8697 | +0.0000 | −0.0002 | [+0.0093, −0.0047, −0.0047] | 1+/2− |
| 10 | 0.8744 | 0.8684 | −0.0016 | −0.0015 | [0.0000, −0.0047, 0.0000] | 0+/1− |
| 15 | 0.8760 | 0.8699 | +0.0000 | +0.0000 | [0.0000, 0.0000, 0.0000] | 0+/0− |
| **20** | **0.8760** | **0.8699** | — | — | — | — |

### MHC_zh — final-epoch (binding arm), 3 seeds, n=149

| k | mean acc | mean mF1 | Δacc | ΔmF1 | per-seed Δacc | signs |
|---|---|---|---|---|---|---|
| 1 | 0.8277 | 0.7925 | −0.0179 | −0.0233 | [+0.0067, −0.0134, −0.0470] | 1+/2− |
| 2 | 0.8277 | 0.7925 | −0.0179 | −0.0233 | [+0.0067, −0.0134, −0.0470] | 1+/2− |
| 3 | 0.8277 | 0.7925 | −0.0179 | −0.0233 | [+0.0067, −0.0134, −0.0470] | 1+/2− |
| **5** | **0.8501** | **0.8196** | **+0.0045** | **+0.0038** | [+0.0201, 0.0000, −0.0067] | 1+/1− |
| 7 | 0.8479 | 0.8181 | +0.0022 | +0.0024 | [+0.0134, −0.0067, 0.0000] | 1+/1− |
| 10 | 0.8456 | 0.8150 | +0.0000 | −0.0008 | [+0.0067, 0.0000, −0.0067] | 1+/1− |
| 15 | 0.8479 | 0.8181 | +0.0022 | +0.0023 | [+0.0067, 0.0000, 0.0000] | 1+/0− |
| **20** | **0.8456** | **0.8158** | — | — | — | — |

### MHC_zh — val-selected (proxy argmax relocated, see §2), 3 seeds, n=149

| k | mean acc | mean mF1 | Δacc | ΔmF1 | per-seed Δacc | signs |
|---|---|---|---|---|---|---|
| 1 / 2 / 3 | 0.8009 | 0.7676 | −0.0157 | −0.0216 | [−0.0201, −0.0134, −0.0134] | 0+/3− |
| 5 | 0.8054 | 0.7770 | −0.0112 | −0.0122 | [−0.0403, +0.0201, −0.0134] | 1+/2− |
| 7 | 0.7942 | 0.7660 | −0.0224 | −0.0232 | [−0.0336, 0.0000, −0.0336] | 0+/2− |
| 10 | 0.8054 | 0.7793 | −0.0112 | −0.0099 | [−0.0067, 0.0000, −0.0268] | 0+/2− |
| 15 | 0.8166 | 0.7907 | +0.0000 | +0.0015 | [0.0000, 0.0000, 0.0000] | 0+/0− |
| **20** | **0.8166** | **0.7892** | — | — | — | — |

### MHC-EN ARM-V — deployed headline stack, val-selected, 4 seeds, n=161 (top-60 banked)

| k | mean acc | mean mF1 | Δacc | ΔmF1 | per-seed Δacc | signs |
|---|---|---|---|---|---|---|
| 1 / 2 / 3 | 0.7547 | 0.7068 | −0.0388 | −0.0429 | [−0.0435, −0.0062, −0.0435, −0.0621] | 0+/4− |
| 5 | 0.7857 | 0.7400 | −0.0078 | −0.0098 | [−0.0124, +0.0186, −0.0124, −0.0248] | 1+/3− |
| 7 | 0.7857 | 0.7391 | −0.0078 | −0.0107 | [−0.0248, +0.0124, −0.0062, −0.0124] | 1+/3− |
| 10 | 0.7935 | 0.7489 | +0.0000 | −0.0008 | [−0.0186, +0.0248, −0.0062, 0.0000] | 1+/2− |
| 15 | 0.7919 | 0.7483 | −0.0016 | −0.0015 | [0.0000, 0.0000, −0.0062, 0.0000] | 0+/1− |
| **20** | **0.7935** | **0.7497** | — | — | — | — |
| 30 | 0.7935 | 0.7480 | +0.0000 | −0.0017 | [−0.0062, +0.0062, 0.0000, 0.0000] | 1+/1− |
| 40 | 0.7935 | 0.7472 | +0.0000 | −0.0026 | [−0.0124, +0.0062, +0.0062, 0.0000] | 2+/1− |
| 60 | 0.7919 | 0.7431 | −0.0016 | −0.0066 | [−0.0124, +0.0062, 0.0000, 0.0000] | 1+/1− |

On the deployed EN stack, **k=20 is tied for the accuracy argmax over the full 1-60 range** (with
k=10, 30 and 40, all at 0.7935) and is the **strict mF1 argmax** (0.7497 vs 0.7489 at k=10, 0.7480
at k=30). Going *up* is as dead as going down.

### MHC-EN ARM-F — final-epoch no-key floor, 3 seeds, n=161

| k | mean acc | mean mF1 | Δacc | ΔmF1 | per-seed Δacc | signs |
|---|---|---|---|---|---|---|
| 1 / 2 / 3 | 0.7660 | 0.7251 | −0.0186 | −0.0174 | [−0.0373, −0.0124, −0.0062] | 0+/3− |
| 5 | 0.7805 | 0.7399 | −0.0041 | −0.0026 | [−0.0311, +0.0124, +0.0062] | 2+/1− |
| **7** | **0.7950** | **0.7594** | **+0.0104** | **+0.0169** | [−0.0062, +0.0186, +0.0186] | 2+/1− |
| 10 | 0.7909 | 0.7516 | +0.0062 | +0.0091 | [−0.0124, +0.0062, +0.0248] | 2+/1− |
| 15 | 0.7847 | 0.7427 | +0.0000 | +0.0002 | [−0.0124, +0.0124, 0.0000] | 1+/1− |
| **20** | **0.7847** | **0.7425** | — | — | — | — |
| 30 | 0.7805 | 0.7405 | −0.0041 | −0.0019 | [−0.0124, 0.0000, 0.0000] | 0+/1− |
| 40 | 0.7743 | 0.7338 | −0.0104 | −0.0087 | [−0.0248, 0.0000, −0.0062] | 0+/2− |
| 60 | 0.7785 | 0.7379 | −0.0062 | −0.0046 | [−0.0248, +0.0062, 0.0000] | 1+/1− |

EN ARM-F k=7 (+0.0104 acc, +0.0169 mF1, 2/3 seeds) is the single largest positive in the whole
sweep. It is **1.7 test items**, it is not the deployed EN arm (ARM-V is), it does not replicate on
ARM-V (which is −0.0078 at k=7), and §4 shows dev does not pick it on 2 of 3 seeds. It is noise.

---

## 4. THE DEPLOYMENT-LEGAL READ — what k would DEV have picked?

This is the only number here that could ever license a change. Rule declared before looking at any
test number: **argmax dev accuracy at the deployed epoch; keep the incumbent k=20 whenever it is
among the tied argmax set (a tie is not evidence to move); otherwise take the largest tied k.**

### 4.1 Per-seed dev pick

| arm | seed | dev k* | dev acc @k* (vs @20) | test acc/mF1 @k* | test acc/mF1 @20 | Δtest |
|---|---|---|---|---|---|---|
| HateMM val-sel | 0 | 20 | 0.8505 (0.8505) | 0.8791 / 0.8730 | 0.8791 / 0.8730 | +0.0000 / +0.0000 |
| HateMM val-sel | 1 | 20 | 0.8505 (0.8505) | 0.8744 / 0.8684 | 0.8744 / 0.8684 | +0.0000 / +0.0000 |
| HateMM val-sel | 2 | 20 | 0.8505 (0.8505) | 0.8791 / 0.8730 | 0.8791 / 0.8730 | +0.0000 / +0.0000 |
| HateMM final | 0 | 20 | 0.8505 (0.8505) | 0.8698 / 0.8632 | 0.8698 / 0.8632 | +0.0000 / +0.0000 |
| HateMM final | 1 | **3** | 0.8318 (0.8037) | 0.8372 / 0.8279 | 0.8791 / 0.8735 | **−0.0419 / −0.0456** |
| HateMM final | 2 | 20 | 0.8505 (0.8505) | 0.8791 / 0.8730 | 0.8791 / 0.8730 | +0.0000 / +0.0000 |
| ZH val-sel | 0 | 15 | 0.8846 (0.8718) | 0.7987 / 0.7719 | 0.7987 / 0.7695 | +0.0000 / +0.0024 |
| ZH val-sel | 1 | 20 | 0.8718 (0.8718) | 0.8322 / 0.8023 | 0.8322 / 0.8023 | +0.0000 / +0.0000 |
| ZH val-sel | 2 | 15 | 0.8974 (0.8718) | 0.8188 / 0.7978 | 0.8188 / 0.7958 | +0.0000 / +0.0020 |
| ZH final | 0 | 20 | 0.8462 (0.8462) | 0.8456 / 0.8158 | 0.8456 / 0.8158 | +0.0000 / +0.0000 |
| ZH final | 1 | 20 | 0.8333 (0.8333) | 0.8389 / 0.8090 | 0.8389 / 0.8090 | +0.0000 / +0.0000 |
| ZH final | 2 | **3** | 0.8590 (0.8462) | 0.8054 / 0.7646 | 0.8523 / 0.8226 | **−0.0470 / −0.0580** |
| EN ARM-F | 0 | 30 | 0.7750 (0.7625) | 0.7888 / 0.7477 | 0.8012 / 0.7596 | −0.0124 / −0.0119 |
| EN ARM-F | 1 | 60 | 0.8000 (0.7875) | 0.7764 / 0.7329 | 0.7702 / 0.7203 | +0.0062 / +0.0126 |
| EN ARM-F | 2 | 7 | 0.8125 (0.7750) | 0.8012 / 0.7653 | 0.7826 / 0.7475 | +0.0186 / +0.0178 |

3-seed-mean Δtest acc / mF1: HateMM val-sel **+0.0000 / +0.0000**, HateMM final **−0.0140 /
−0.0152**, ZH val-sel **+0.0000 / +0.0015**, ZH final **−0.0157 / −0.0193**, EN ARM-F **+0.0041 /
+0.0062**. EN ARM-V: **not available** (§1.3).

### 4.2 Pooled dev pick (one k per config, shared across seeds — the normal way k is set)

| arm | pooled dev k* | tied | pooled dev acc (vs @20) | mean test acc/mF1 | Δtest acc/mF1 | per-seed Δacc |
|---|---|---|---|---|---|---|
| HateMM val-sel | **20** | [15, 20] | 0.8505 (0.8505) | 0.8775 / 0.8715 | +0.0000 / +0.0000 | [0, 0, 0] |
| HateMM final | **20** | [10, 15, 20] | 0.8349 (0.8349) | 0.8760 / 0.8699 | +0.0000 / +0.0000 | [0, 0, 0] |
| ZH val-sel | 15 | [15] | 0.8846 (0.8718) | 0.8166 / 0.7907 | +0.0000 / +0.0015 | [0, 0, 0] |
| ZH final | **3** | [1, 2, 3] | 0.8462 (0.8419) | 0.8277 / 0.7925 | **−0.0179 / −0.0233** | [+0.0067, −0.0134, −0.0470] |
| EN ARM-F | 30 | [15, 30] | 0.7833 (0.7750) | 0.7805 / 0.7405 | −0.0041 / −0.0019 | [−0.0124, 0, 0] |

**Read honestly: dev selection of k is worthless-to-harmful here.** On HateMM it re-picks the
incumbent. On ZH final it picks k=3 on a +0.0043 pooled-dev margin and pays −0.0179 test acc. On EN
ARM-F it picks k=30 and pays −0.0041. The dev sets (n=107 / 78 / 80) are far too small to resolve
differences that are worth 1-2 test items; this is the same dev-resolution wall already recorded for
epoch selection (ERRPAT ZH §1.5, "dev argmax 0.8718 = 68/78 in all three seeds — information content
essentially nil"). **No deployment-legal version of this lever gains anything.**

---

## 5. MECHANISM

### 5.1 k ≤ 3 is a 1-NN classifier, exactly

With cosines sorted descending (`s_0 ≥ s_1 ≥ s_2`) and weights `[3,2,1]`, the rank-1 term satisfies
`3·s_0 ≥ 2·s_1 + 1·s_2` always, with equality only when all three cosines are equal. The same holds
a fortiori for k=1 and k=2. So the sign of the vote at k ≤ 3 is **the top-1 neighbour's label**,
regardless of what ranks 2 and 3 say.

Verified empirically, not just algebraically: the k=1, k=2 and k=3 prediction vectors are
element-wise identical to the top-1-label vector in **19 of 19 cells**
(`ksweep_OUT.json:mechanism.k_le_3_is_1nn`). This is why the k ∈ {1,2,3} rows in §3 are
byte-identical to each other in every table, and it prices the "cut the noise" intuition: shrinking
k does not sharpen the neighbourhood, it **discards the vote entirely** and falls back to 1-NN,
which is 2-4 accuracy points worse on every arm.

### 5.2 Ranks 11-20 are already inert — there is no tail noise to cut

Number of test items whose prediction differs from k=20 (`mechanism.items_changed_vs_k20`):

| arm/seed | k=1 | k=5 | k=7 | k=10 | k=15 |
|---|---|---|---|---|---|
| HateMM s0 val-sel | 18 | 6 | 2 | **0** | **0** |
| HateMM s1 val-sel | 14 | 8 | 5 | **0** | **0** |
| HateMM s2 val-sel | 13 | 4 | 3 | **0** | **0** |
| HateMM s0 final | 11 | 9 | 6 | **0** | **0** |
| HateMM s1 final | 15 | 3 | 1 | 1 | **0** |
| HateMM s2 final | 13 | 4 | 3 | **0** | **0** |
| ZH final s0/s1/s2 | 11 / 6 / 11 | 3 / 4 / 3 | 2 / 3 / 2 | 1 / 2 / 1 | 1 / 0 / 0 |
| EN ARM-V s0..s3 | 23 / 23 / 25 / 16 | 10 / 15 / 14 / 8 | 12 / 14 / 9 / 6 | 11 / 8 / 5 / 2 | 2 / 4 / 5 / 0 |

On HateMM the top-**10** already fixes the decision on 215/215 items in 5 of 6 cells (214/215 in the
sixth): neighbours 11-20 carry rank weights 10..1 against 20..11 for the leaders and **never flip a
single prediction**. The rank weighting has already down-weighted the tail into irrelevance. The
user's hypothesis — that ranks 11-20 inject noise worth removing — is therefore **structurally
false on HateMM**, and only weakly live on ZH/EN where the tail moves 0-5 items with no consistent
sign.

### 5.3 Flip accounting at the forensic oracle k

"Stable core" = items wrong at k=20 in **every** seed. Counts independently reproduce the ERRPAT
reports (HateMM 24 val-sel / 25 final; EN ARM-V 22 — matching ERRPAT MHC-EN §0.2's "22 items wrong
in all 4 seeds"), which is a further cross-check on this script.

| arm | shared oracle k | core size | core fixed in ALL seeds | core fixed in ≥1 seed | per-seed items fixed | per-seed items broken | net |
|---|---|---|---|---|---|---|---|
| HateMM val-sel | **20** | 24 | 0 | 0 | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |
| HateMM final | **20** | 25 | 0 | 0 | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |
| ZH val-sel | **20** | 14 | 0 | 0 | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |
| ZH final | 5 | 22 | **0** | 4 | [3, 2, 1] | [0, 2, 2] | [+3, 0, −1] |
| EN ARM-V | 10 | 22 | **0** | 5 | [4, 6, 2, 1] | [7, 2, 3, 1] | [−3, +4, −1, 0] |
| EN ARM-F | 7 | 26 | **1** | 9 | [6, 7, 7] | [7, 4, 4] | [−1, +3, +3] |

**Three of six arms have shared oracle k = 20 — i.e. no k in the grid beats the incumbent even when
test labels choose it.** Where the oracle k does move, it is churn, not repair: the stable core is
fixed in all seeds for **exactly one item across all six arms** (EN ARM-F), and every fix is paid
for by a comparable number of previously-correct items breaking (EN ARM-V: 13 fixed, 13 broken;
EN ARM-F: 20 fixed, 15 broken; ZH final: 6 fixed, 4 broken). This is the same
selection-locked geometry ERRPAT already documented — consensus errors retrieve a neighbourhood
that is majority-wrong at *every* depth, so no truncation of that neighbourhood can rescue them.

Even the per-seed oracle k (a strictly larger cheat: a different k per seed, chosen on test) is
worth only +0.0016 / +0.0031 / +0.0067 / +0.0067 / +0.0077 / +0.0145 acc across the six arms —
under half the +0.030 bar as a pure upper bound.

---

## 6. BOTTOM LINE

1. **Answer to the user's question: yes, it has now been swept; no, reducing k does not help.**
   k=20 is on the plateau, the plateau is flat from k≈10-15 to k=20 (and to k=60 on EN), and
   everything below k≈7 is a cliff.
2. **The premise does not hold.** Rank weighting `[k..1]` has already neutralised the tail —
   ranks 11-20 flip zero predictions on HateMM. There is no "neighbourhood noise" at the tail to cut;
   the noise the ERRPAT reports found is at ranks 1-5, where the *labels themselves* are wrong.
3. **Small k is not a sharper vote, it is 1-NN.** k ≤ 3 provably discards the vote (§5.1), costing
   0.016-0.039 acc.
4. **Nothing is deployable.** The best forensic shared-k gain is +0.0104 acc (1.7 items, on a
   non-deployed arm); dev-selected k is 0.0000 / −0.0140 / −0.0157 / +0.0041 per-seed and
   0.0000 / 0.0000 / −0.0179 / −0.0041 pooled. No prereg is warranted and no GPU should be spent.
5. **Axis CLOSED.** Record this as a measured negative on the vote-aggregation axis. It joins F49 /
   F66 / F86: the residual is selection-locked in the *representation*, and re-weighting or
   truncating the retrieved list cannot reach it.

**Test-touch note.** This consumed no new test budget: every prediction replayed here was already
banked and already reported in the 2026-07-26 ERRPAT reports. The oracle-k readings in §3/§5.3 are
labelled forensic and must never appear in a results table.
