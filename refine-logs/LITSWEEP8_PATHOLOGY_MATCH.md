# LITSWEEP-8 — PATHOLOGY-SHAPE MATCH

**Date:** 2026-07-28 NZST · **Agent:** litsweep-8 (pathology lens) · **Cost: $0** — CPU reading, grep,
WebSearch/WebFetch, and ~90 s of numpy on **banked TRAIN-split feature caches only**.
**ZERO GPU / SLURM / Modal / training.** **Test-split contact: NONE** — no script in this record opens
`dev_seen` or `test_seen`; the only `.pt` files loaded are `data/CLIP_Embedding/<DS>/train_*.pt`.
Nothing under `autoresearch/goal_mllm_plus3/state/` was written. `refine-logs/LITSWEEP7_LANDING_SITE.md`
and `refine-logs/VSW_PREGATE_RECORD.md` / `scripts/analysis/vsw_*` were **not touched**
(VSW read only, and only after it had already written its own §9 verdict).

**Lens.** Previous sweeps (LITSWEEP2/3/5/6) searched *by mechanism concept* — "what technique could we
add". This one searches *by pathology shape*: take the measured symptom profile, find the fields where
that exact profile is a named, studied phenomenon, and bring back their diagnosis, their fix, and their
negative results.

---

## §0. BOTTOM LINE (read first)

1. **The central finding has a name — but not the name the brief expected, and not one name.** There is
   **no** published term for "a retrieval system whose ranking quality far exceeds its decision quality"
   *as a phenomenon of retrieval-augmented classifiers* (§4). But the **mechanism** underneath it is
   classical, and the project has been re-deriving two theorems from 1978 and 2012:
   * **Bailey & Jain (1978)** — for fixed `k`, the asymptotically optimal weight vector for a weighted
     k-NN rule is **uniform**; distance/similarity weighting is asymptotically useless.
   * **Samworth (2012, Ann. Statist. 40(5):2733-2763, arXiv:1101.5783)** — the regret ratio of the
     *optimally* weighted NN classifier to the unweighted k-NN classifier "depends asymptotically only
     on the dimension `d` … The improvement is greatest when `d = 4`, but thereafter **decreases as
     `d → ∞`**."
   Our `d` is **1024 (head) / 7168 (raw)**. **F98/AGGNET and the VSW C4 arm are exactly the family these
   two theorems bound, and their measured realisations (+0.0134, +0.0255, "net capped below the bar at
   every point of a 16 000× λ continuum") are what the theory predicts.** This is a paper-level citation
   the project currently does not have.

2. **This sweep contributes two new $0 measurements that close the retrieval axis arithmetically** (§2,
   train-LOO raw fused arena, the F95/AGGNET arena):
   * **The similarity values are decision-irrelevant.** Discarding every cosine and voting on the
     retrieved **labels alone** with the same rank weights reproduces the deployed decision on
     **99.60 % / 100.00 % / 100.00 %** of train items (HateMM / MHC-ZH / MHC-EN) and the accuracy to
     within 0.0013 / 0.0000 / 0.0000. **The deployed classifier is a rank-weighted label majority over
     the top-20; the metric enters only through *which* 20 items are retrieved and *in what order*.**
   * **Retrieval coverage is already ~perfect and pool expansion is worthless.** At least one
     correct-label neighbour is present in the deployed top-20 for **99.33 % / 100.00 % / 99.82 %** of
     items. Expanding the candidate pool 20 → 400 raises the free-weight oracle by
     **+0.0067 / +0.0000 / +0.0018** — **under +0.030 on 3/3**. **The entire re-ranking / candidate-set
     family (k-reciprocal, mutual proximity, local scaling, NICDM, any "retrieve better") SELF-KILLS on
     its own gold-cheating oracle.**

3. **Five phenomenon matches are real and their theories jointly PREDICT law-I; none of them supplies a
   legal fix.** Weighted-NN futility (§3.1) predicts the aggregation nulls; the **rare/outlier
   minority-example taxonomy** of Napierała & Stefanowski (2016) names our error population *by its own
   published thresholds* and its literature's own finding is that resampling/reweighting does not reach
   it (§3.2); **neural collapse** predicts both the 0.9999 cone and — precisely — the F89-T2b failure
   mode where whitening surfaces the length nuisance (§3.4); **long-tail memorization** (Feldman) predicts
   the ~90 % seed-invariant confident error set and predicts that no label-free operator can fix it (§3.5).
   And **law-I has already been published in another field**: Xu, Alon & Neubig (arXiv:2301.02828) ask why
   kNN-LM works *"even when the k-nearest neighbor component retrieves examples from the same training set
   the LM was trained on"* — our exact configuration — and conclude the gain is **not** better retrieval,
   tracing it instead to input representation, approximate search and **softmax temperature of the kNN
   distribution** (= our VSW λ-continuum, measured dead). Cuconasu et al. (arXiv:2401.14887) report that
   **adding random documents improves RAG accuracy by up to 35 %** while high-scoring-but-irrelevant ones
   hurt. **We have been re-deriving a known result** (§3.6).

4. **Hubness is present but measured non-causal** (§3.3, new numbers): skewness of the k=20 occurrence
   count is 1.54 / 1.74 / 1.45 and the busiest bank item is the top-20 neighbour of 103-117 queries
   (5.2-5.8× expected) — real hubness. But bad occurrences are only weakly concentrated (top-5 % of
   items carry 17-29 % of them, vs 5 % under no concentration), and F89-T2a measured the canonical fix
   (CSLS) inert. **I also correct MECHFIX's own explanation of that null** (§6): the reason CSLS is inert
   is *not* "no dynamic range" — F95 §4.1 shows the bank main effect carries 31-44 % of the cosine's
   score variance, and F89 records that T2a "changes half the retrieved sets". CSLS is inert because the
   *decision* does not read the similarity at all (§2).

5. **Net for the campaign: this sweep proposes ZERO new GPU spend and closes one family.** One candidate
   survives to a $0 pregate sketch and it is a **door-closer / analysis deliverable**, not a goal bet
   (§5.3). Everything else self-kills on oracle or is pre-closed. The honest headline is the one the
   brief said would be an outstanding result: **the phenomenon is (partly) known, the theory predicts our
   ten-odd law-I data, and the same theory predicts that nothing in the legal class can work.**

---

## §1. SYMPTOM PROFILE — VERIFIED, WITH FOUR CORRECTIONS

Every claim in the tasking was re-read against its source. Verdicts below; corrections in **bold**.

| # | Tasking claim | Source (file:line) | Verdict |
|---|---|---|---|
| 1a | pair-AUC +0.13 to +0.27 over cosine on 18/18 cells, 0/36 end-to-end | `refine-logs/MECHNOV_PAIRVERIFY_PREGATE.md:266-279` (18/18, +0.0515…+0.2685), `:315` ("`Δ ≥ +0.010` is achieved by 0 of 36 cells") | **VERIFIED** |
| 1b | temporal ROC 0.8484 vs random-split 0.7175, macro-F1 −0.084 | `research-wiki/EVAL_temporal_memory_W4.md:28-29,33,41-44` | **VERIFIED**, and the source itself calls it "an **operating-point (calibration) failure, not a separability failure**" (`:44`) — and *fixes it*: 20 labelled new-period samples used for **threshold recalibration only** fully recover the drop (0.6273 → 0.7336 ≥ the 0.7113 random floor, `:87,92-96`). **This is the ONE place in the whole campaign where a ranking-vs-decision gap was converted, and the converter was a threshold, not an operator.** |
| 1c | correct-class analogue at median rank ~1.5 (ZH) / ~3.0 (HateMM) | `refine-logs/ERRPAT_MHC-ZH_2026-07-26.md:234` (median rank 1.5, 11 of 22 at rank 1); `refine-logs/ERRPAT_HateMM_2026-07-26.md:134` (median rank 3.0) | **VERIFIED** |
| 1d | EN dev AUC 0.898 converts nothing (F50/F48) | not re-read this sweep | **UNVERIFIED here** (accepted from the tasking; F50's ban is quoted verbatim in `LITSWEEP5_COMPLETENESS.md:77`) |
| 2a | train top-1 cosine ~0.9999; 0.999852 (error) vs 0.999976 (correct) | `MECHFIX_PREGATE_2026-07-27.md:238`, `ERRPAT_HateMM_2026-07-26.md:131` | **VERIFIED** — **but this is a property of the TRAINED HEAD space only.** New measurement (§2): in the **raw** encoder key space the median top-1 cosine is **0.9459 / 0.9537 / 0.9416** with a median rank-1→rank-20 spread of **0.0246 / 0.0210 / 0.0227**. **The cone collapse is manufactured by the head, not inherited from the encoder.** That matters — see §3.4. |
| 2b | whitening 0.9999 → 0.52 but length-Spearman 0.52 → 0.87, negative 3/3 | `MECHFIX_PREGATE_2026-07-27.md:238,278,288-290,308,427` | **VERIFIED** |
| 3 | errors ~90 % seed-invariant; purity 0.12-0.22; median \|vote\| 0.7267 | `ERRPAT_HateMM_2026-07-26.md:110-113` (24-25 of 26-28 = 89-93 %), `:130,133`; `ERRPAT_MHC-ZH_2026-07-26.md:203` (22/25 = 88 %), `:218,220` | **VERIFIED** (EN 52 % not re-checked) |
| 4a | F66/ISR: 91-98 % of oracle headroom formally disjoint from symmetric operators | `refine-logs/ISR_PREGATE_RECORD.md:110-118` (HateMM +0.0776 = +0.0012 symmetric + +0.0764 selection; EN +0.0700 = +0.0064 + +0.0636) | **VERIFIED**, with a scope note: the decomposition is computed on the **W2-B per-segment** oracle, not on an arbitrary Gram-matrix operator. Generalising it to "every symmetric operator on a fixed Gram" is the project's standing *inference*, not that record's measurement. §2 supplies an independent and much tighter bound that does not need the inference. |
| 4b | F98/AGGNET oracle +0.1492 / +0.1520 / +0.2186, realised +0.0134 | `refine-logs/AGGNET_PREGATE_RECORD.md:368-370,387,418` | **VERIFIED**, and **independently reproduced this sweep** (§2: +0.1440 / +0.1520 / +0.2295 on full-bank train LOO; ZH matches to 4 dp) |
| 4c | F94 truncation closed both directions | `AGGNET_PREGATE_RECORD.md:44-48` (quoting KSWEEP) | **VERIFIED** |
| 5 | "Ten certified law-I data" | `state/findings.jsonl` F50 ("5th"), F63 ("SEVENTH"), F65 ("8th"), F87 ("**9th law-I NOT certified**") | **CORRECTION — the ledger certifies EIGHT.** F87 explicitly declines to certify a 9th. F95 calls itself "the sharpest instance of law-I yet recorded" (`MECHNOV_PAIRVERIFY_PREGATE.md:436-437`) without claiming an ordinal. The cand-2 figure "−0.0538 train-LOO buying +0.0132" could not be reconciled: **+0.0132 is verified** (`CAND2_REP2_VERDICT_REVIEW.md:120,182`) but the only `0.0538` in `refine-logs/` is HateMM's **LOO-disagreement rate** in a different experiment (`ERRPAT_HateMM_2026-07-26.md:390`). **Do not put "ten certified" in the paper without re-deriving the list.** |
| 6 | test n=215/149/161; head-seed band ±0.014; banks 549-744 in 7168-d | `MECHFIX_PREGATE_2026-07-27.md:307` (n via 1-item = 0.0047/0.0067/0.0062), `:429`; `MECHNOV_PAIRVERIFY_PREGATE.md:155-157,361` | **VERIFIED** |
| 7 | VSW "IN PROGRESS", HateMM +0.0255 at p=0.0050 vs ZH −0.0017 p=0.5522 | `refine-logs/VSW_PREGATE_RECORD.md:567,570,902-960` | **CORRECTION — VSW has already written its verdict: "KILL as a performance lever."** K-VSW-1 FAIL and *arithmetically unreachable*; DEG-A **and** DEG-B both fire on MHC-ZH (0.9516 / 0.9706, with DEG-B's arg-max at k=20 = the deployed rule). +0.0255 clears only the weaker K-VSW-0 interest threshold, on one dataset, and remains below the F47-gate benchmark of +0.0269 on that same dataset. Read `:902-960` before treating it as a "live exception". |

**Two further facts from the record that this sweep treats as load-bearing and the tasking did not mention:**

* **F95 §4.1 variance decomposition** (`MECHNOV_PAIRVERIFY_PREGATE.md:420-437`): in the raw key space only
  **26.6-37.7 %** of the cosine's score variance is query×bank *interaction*; 62-73 % is item-level
  offsets. A trained relation scorer inverts this to **77-93 %** interaction and **still does not decide
  better**. This is the single most diagnostic number in the campaign and §2/§3.3 explain it.
* **VSW §6.3** (`VSW_PREGATE_RECORD.md:748-777`): `net = changed · (2·precision − 1)`; precision decays
  monotonically with aggregation sharpness at almost exactly the rate that cancels the rise in volume,
  so the net is pinned to +11…+21 items on HateMM across a 16 000× λ range and is **below +0.030 at
  every point**. §3.1 gives that empirical law its classical name.

---

## §2. WHAT THIS SWEEP MEASURED ($0, train-split only) — the arithmetic that reorganises the diagnosis

Arena: the **F95/AGGNET arena** — banked raw fused train keys,
`fused = L2norm(concat(L2norm(img_feats), L2norm(text_feats)))`, per-dataset deployed encoder caches
`data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_*.pt`, full-bank leave-one-out, deployed rule replayed
from `src/utils/metrics.py:262-301` (`v = Σ (2·lab_i − 1)·sim_i·w_i / Σ w_i`, `w = [20,…,1]`,
`predict 1 iff v ≥ 0` — confirmed by reading lines 262-301 and 297-301). One HateMM row has zero-norm
img **and** text features and is dropped (n 744 → 743, disclosed).

**Parity check against the record:** deployed train-LOO accuracy **0.8493 / 0.8480 / 0.7687**
(F95 on a 4/5 bank: 0.8441 / **0.8480** / 0.7796); free-weight top-20 oracle Δ **+0.1440 / +0.1520 /
+0.2295** (AGGNET: +0.1492 / **+0.1520** / +0.2186). MHC-ZH matches both quantities to 4 dp against two
independent prior records; the HateMM/EN differences are the full-bank-vs-4/5-bank difference.

### 2.1 RESULT A — the similarity values are decision-irrelevant

Define `M = Σ_i (2·lab_i − 1)·w_i / Σ_i w_i` — the rank-weighted **label** majority of the top-20, with
**every cosine discarded**.

| dataset | deployed acc | label-majority-only acc | decisions identical | median \|v − M\| | median \|M\| |
|---|---|---|---|---|---|
| HateMM (n=743) | 0.8493 | 0.8506 | **99.60 %** (3 items differ) | 0.0445 | 0.7238 |
| MHC-ZH (n=579) | 0.8480 | 0.8480 | **100.00 %** (0 differ) | 0.0299 | 0.5429 |
| MHC-EN (n=549) | 0.7687 | 0.7687 | **100.00 %** (0 differ) | 0.0293 | 0.4190 |

Algebraically: writing `sim_i = c̄ + δ_i` with `c̄` the w-weighted mean, `v = c̄·M + D` with
`D = Σ s_i δ_i w_i / Σ w_i`. Measured `|D| ≈ 0.03-0.045` median against `|M| ≈ 0.42-0.72` median — a
factor 10-17. And `M` is quantised: flipping the label at rank `i` moves it by `2w_i/210 ∈ [0.0095, 0.19]`.
So `sign(v) = sign(M)` outside a vanishing tie band. **In the head space the margin is far larger still**,
because there `sim` spreads over ~1e-4 (`ERRPAT_HateMM_2026-07-26.md:131`) while `M`'s smallest possible
step is 0.0095 — a factor ~100 — so in the deployed head space the identity is essentially exact.

**Consequences, each of which is a previously-unstated corollary of the deployed design:**

* **C1 (the kernel).** The decision map factors through `query ↦ (ordered 20-tuple of retrieved bank
  labels)`. **Any change to the similarity function that leaves that tuple fixed is EXACTLY
  decision-invariant.** Not approximately — exactly.
* **C2 (why F95's +0.13-0.27 pair-AUC bought nothing).** The verifier is a better *scorer*. The deployed
  rule does not consume scores. Its improvement can only enter through re-ordering, and the pool it
  re-orders already contains what it needs (Result B) but the *fixed* rank profile `[20…1]` limits what
  re-ordering can do — which is precisely the family Bailey-Jain and Samworth bound (§3.1).
* **C3 (why CSLS was inert).** CSLS subtracts a bank-item offset from `sim`. Under C1, an offset that does
  not change the retrieved label tuple is a no-op **by construction**, and F89 records exactly that
  signature: "changes half the retrieved sets, flips **0-4** decisions" (`MECHFIX:426`).
* **C4 (why the campaign's own "law-I" is not mysterious on this axis).** Every "better signal, no
  conversion" datum that improves a *score* rather than a *retrieved label tuple* is in the kernel of the
  decision map. Law-I on the retrieval/aggregation axis is a **design fact**, not an empirical mystery.

### 2.2 RESULT B — coverage is ~perfect and pool expansion is worthless

Free-weight oracle (any non-negative reweighting of a pool of size `M` ⇒ correct iff the pool contains
≥1 correct-label item; this is exactly AGGNET's family oracle, generalised to `M > 20`):

| dataset | coverage @20 | @50 | @100 | @400 | oracle Δ @20 | oracle Δ @400 | **marginal of pool expansion** |
|---|---|---|---|---|---|---|---|
| HateMM | 0.9933 | 1.0000 | 1.0000 | 1.0000 | +0.1440 | +0.1507 | **+0.0067** |
| MHC-ZH | **1.0000** | 1.0000 | 1.0000 | 1.0000 | +0.1520 | +0.1520 | **+0.0000** |
| MHC-EN | 0.9982 | 1.0000 | 1.0000 | 1.0000 | +0.2295 | +0.2313 | **+0.0018** |

Rank-constrained variant (keep the deployed profile `w = [20…1]`, oracle chooses only *which* items and
*in what order* — i.e. what a pure re-ranker can do): Δ = **+0.0780 / +0.1123 / +0.1876** at pool 20,
saturating at the free-weight value by pool 100. Note this is **strictly weaker** than free reweighting
at the same pool, so it is dominated by the family AGGNET/VSW already measured.

**Consequences:**

* **C5 (the retrieval axis is closed by its own oracle).** *Every* candidate whose mechanism is "retrieve
  a better/larger/differently-ordered candidate set" — k-reciprocal re-ranking (Zhong et al., CVPR 2017,
  arXiv:1701.08398), mutual proximity (Schnitzer et al., JMLR 2012), local scaling / NICDM, QB-Norm
  (Bogolin et al., CVPR 2022, arXiv:2112.12777), NNN (arXiv:2410.24114), CSLS, larger `k` — has a
  gold-cheating marginal oracle of **+0.0067 / +0.0000 / +0.0018**. **Per the tasking's own rule
  ("if the family's gold-cheating oracle is under +0.030, kill it yourself"), I kill the entire family
  here.** It is not a matter of which re-ranker; there is nothing outside the top-20 worth retrieving.
* **C6 (the headroom is entirely a *trust* problem).** 100 % of the +0.144/+0.152/+0.230 oracle is
  "which of the 20 already-retrieved items to believe". That is a per-item weighting problem, it is the
  F98/VSW family, it is bounded by §3.1's theorems, and F66/L1's train-non-transferability wall
  (`LITSWEEP5_COMPLETENESS.md:128`) blocks learning the selector.

### 2.3 RESULT C — hubness, quantified

Standard Radovanović statistics on the same train keys (k-occurrence count `N_k`, "bad" = retrieved
neighbour's label ≠ query's label):

| dataset | S(N_20) skew | max N_20 (× expected) | bad-occurrence rate | random-label baseline | BN share of top-1 % / top-5 % hubs |
|---|---|---|---|---|---|
| HateMM | 1.539 | 117 (5.8×) | 0.2476 | 0.480 | 0.050 / 0.200 |
| MHC-ZH | 1.740 | 103 (5.2×) | 0.2768 | 0.429 | 0.062 / 0.276 |
| MHC-EN | 1.450 | 109 (5.5×) | 0.3455 | 0.425 | 0.034 / 0.173 |

Reading: **hubness is real** (positive skew 1.45-1.74; one bank item is a top-20 neighbour of 103-117 of
549-744 queries against an expectation of 20). **Bad hubs are not the mechanism**: the bad-occurrence rate
is far *below* the random-label baseline, so retrieval carries genuine label signal on average, and bad
occurrences are only 3.5-5.9× concentrated in the top 5 % of items — removing or re-scoring them reaches
at most 17-29 % of them. Meanwhile the *error* population's purity is 0.12-0.22, i.e. a bad rate of
0.78-0.88 — **four times the global rate**. The pathology is **local**, not hub-borne.

### 2.4 Reproduction

Three self-contained numpy snippets, run under `OMP_NUM_THREADS=8`, ~30 s each, no files written. Inputs:
`data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt`,
`data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`,
`data/CLIP_Embedding/MHC/train_Qwen2.5-VL-7B-Instruct_HF.pt` (the exact three caches F95 §2.1 names).
Procedure per dataset: drop zero-norm rows → `fused` as above → `S = Z Zᵀ`, `diag = −inf` →
`order = argsort(−S)` → `top20 = order[:, :20]` → `v = ((2y[top20]−1) · S[top20] · w).sum(1)/w.sum()`,
`M = ((2y[top20]−1) · w).sum(1)/w.sum()` → coverage `= (y[order[:,:M]] == y[:,None]).any(1)`.

---

## §3. PHENOMENON MATCHES

Every external claim below was surfaced by WebSearch and the load-bearing ones confirmed by fetching the
arXiv/ACL/publisher page during this sweep. Anything not confirmed is marked **UNVERIFIED**.

### 3.1 ★ THE MATCH FOR THE AGGREGATION AXIS — *weighted nearest-neighbour futility*

**Name in the field:** there is no catchy label; the results are cited as "Bailey and Jain's theorem" and
"the optimal weighted nearest neighbour classifier". The phenomenon is: **weighting the votes of a fixed
k-neighbourhood cannot help.**

| paper | id / venue | claim |
|---|---|---|
| Bailey, T. & Jain, A. K., *A note on distance-weighted k-nearest neighbor rules* | IEEE Trans. Syst. Man Cybern. **8**:311-313 (1978) — pre-arXiv, **id UNVERIFIABLE**; the *claim* is confirmed by multiple independent secondary sources fetched this sweep | For fixed `k`, the asymptotic error probability of a weighted k-NN rule is minimised **for all distributions** by **uniform** weights; i.e. the unweighted k-NN rule's asymptotic error is **no worse than any** weighted k-NN rule. |
| Samworth, R. J., *Optimal weighted nearest neighbour classifiers* | Ann. Statist. **40**(5):2733-2763 (2012), **arXiv:1101.5783** — abstract fetched verbatim | Derives the asymptotically optimal non-negative weight vector. **"The ratio of the regret of this classifier to that of an unweighted k-nearest neighbour classifier depends asymptotically only on the dimension `d` … The improvement is greatest when `d = 4`, but thereafter decreases as `d → ∞`."** |
| MacLeod, Luk & Titterington, *A Re-Examination of the Distance-Weighted k-Nearest Neighbor Classification Rule* | IEEE Xplore doc 4075685; **venue/year UNVERIFIED** (IEEE Trans. SMC, ~1987) | The finite-sample counter-argument: Bailey-Jain is asymptotic and does not entail that weighting is useless at finite `n`. **Cite this as the honest counterweight.** |

**Does the diagnosis match?** Exactly, and it is *predictive*, not merely consistent:

* §2 Result A measures that our similarity-weighted vote **is** the unweighted-in-similarity label
  majority to 99.6-100 %. Bailey-Jain says that is the asymptotically *right* thing to be.
* F98/AGGNET (learned per-query non-negative reweighting) and VSW C4 (verifier-monotone reweighting) are
  **precisely the weighted-NN family Samworth optimises over**. At `d = 1024` (head) / `7168` (raw), his
  result says the available regret improvement is at its asymptotic floor. Measured: **+0.0134** and
  **+0.0255**, one dataset each, both under bar, with VSW §6.3's net "capped below the bar at every point
  of a 16 000× λ continuum". **That empirical law is Samworth's theorem showing up at finite `n`.**
* F94/KSWEEP's finding that `k = 20` sits on a flat plateau and rank-weighting is direction-inconsistent
  is the same statement about the *k* half of the family.

**Which law-I data does it explain?** All of the *aggregation/decision-side* ones: F63 (LP), F94 (ksweep),
F95 (pair-verify), F96 (RESTRANS), F97 (VGA/VNQ), F98 (AGGNET), the VSW C4 arm, and F89-T1/T2a. It does
**not** explain the representation-side ones (F44/F50/F65/F91).

**What fixed it there?** *Nothing, by theorem.* Samworth's own escape clause is explicit: "improvements in
the rate of convergence are possible under stronger smoothness assumptions, **provided we allow negative
weights**." Negative weights on a binary vote mean *inverting* a neighbour's label — which requires
knowing it is wrong, i.e. the banned per-item selection of F66/Law-III. **The literature's one escape is
our one ban.** That is a genuinely satisfying closure of the axis.

**Legal / pre-closed?** N/A — this is a *citation*, not a lever. It costs $0 and it is the single most
valuable import in this sweep: it converts "we tried nine reweightings and they all failed" into
"the family is bounded by a 1978 theorem and a 2012 asymptotic expansion, and our measurements sit where
the theory puts them."

### 3.2 ★ THE MATCH FOR THE ERROR POPULATION — *rare and outlier minority examples*

**Name in the field:** the **safe / borderline / rare / outlier** taxonomy of minority examples.

* Napierała, K. & Stefanowski, J., *Types of minority class examples and their influence on learning
  classifiers from imbalanced data*, **J. Intelligent Information Systems** 46(3), 2016, DOI
  `10.1007/s10844-015-0368-1`. The type of an example is defined by **the class composition of its
  k-neighbourhood**, with same-class fractions **≥0.8 = safe, ≥0.5 = borderline, ≥0.2 = rare, <0.2 =
  outlier** (thresholds as reported by secondary sources fetched this sweep; **exact threshold values
  UNVERIFIED against the paper PDF**).
* Skryjomski & Krawczyk (?), *Influence of minority class instance types on SMOTE imbalanced data
  oversampling*, PMLR **v74** (2017) — **authorship/venue partially UNVERIFIED**. Finding: oversampling
  helps safe and borderline instances and **does not reach rare/outlier instances.**

**Does the diagnosis match?** It is a numerical bullseye. Our error population's median top-20 true-label
purity is **0.12-0.22** (`ERRPAT_HateMM:133` = 0.1667; `ERRPAT_MHC-ZH:220` = 0.15), and 6/27 HateMM errors
have **zero** true-label neighbour in the top-20 (`ERRPAT_HateMM:135`). **By Napierała-Stefanowski's own
thresholds our errors are "rare" (0.2) and "outlier" (<0.2) instances**, and the campaign's global
statistics say the *non*-errors are "safe" (purity 1.00). This gives the paper a published name and a
published taxonomy for its central error population, replacing the project's coined phrase "confident
neighbourhood inversion".

**Which law-I data does it explain?** The literature's own result — that resampling and reweighting do not
reach rare/outlier instances — is the field-external prediction of F78/curation (measured +0.0016 vs a
random-deletion control of +0.0031, `ERRPAT_HateMM:390-396`), F101/BSY (bank synthesis pre-closed because
the local class odds in the target cell are 0.1231), and the whole AGGNET/VSW reweighting family.

**What fixed it there?** In that literature: **nothing that operates on the existing sample.** Their
prescription is data collection targeted at the rare sub-concepts. Under ban [8] (own train split only)
that is unavailable to us.

**Legal / pre-closed?** Citation only. $0.

### 3.3 HUBNESS — present, quantified, and measured non-causal

* Radovanović, M., Nanopoulos, A. & Ivanović, M., *Hubs in Space: Popular Nearest Neighbors in
  High-Dimensional Data*, **JMLR 11:2487-2531 (2010)** — verified. Defines hubness as the skew of the
  k-occurrence distribution `N_k`, shows it is driven by **intrinsic dimensionality**, and defines
  **"bad hubs"** — frequent neighbours whose labels mismatch — which "account for a surprisingly large
  portion of the total error" in k-NN classification.
* Standard fixes: local scaling / NICDM and **mutual proximity** (Schnitzer et al., JMLR 2012 — **id
  UNVERIFIED**); **CSLS** (Conneau et al., word translation, ICLR 2018 — **id UNVERIFIED**);
  **QB-Norm** (Bogolin et al., CVPR 2022, arXiv:**2112.12777** — **id UNVERIFIED**); **NNN**
  (arXiv:2410.24114, EMNLP 2024 — id taken from `LITSWEEP6_MEMBANK.md:658`, not re-verified here).

**Does the diagnosis match?** **Partially, and the mismatch is the informative part.** §2.3 measures real
hubness (`S(N_20)` = 1.45-1.74; busiest bank item is a top-20 neighbour of 103-117 of 549-744 queries,
5.2-5.8× expected). F95 §4.1's decomposition — 62-73 % of the cosine's score variance is item-level
offsets — is a textbook hubness signature and is the strongest hubness evidence in the campaign. **But
bad hubs are not the mechanism:** the global bad-occurrence rate (0.25-0.35) is *below* the random-label
baseline (0.43-0.48), and bad occurrences are only 3.5-5.9× concentrated in the top 5 % of bank items,
while the error population's bad rate is 0.78-0.88 — four times global. The pathology is **local to
specific queries**, not carried by a small set of universal bad hubs.

**What fixed it there / does it work here?** CSLS was measured **inert** on our decisions (F89-T2a: 0.0000
/ 0.0000 / −0.0021, `MECHFIX:307,426`). §2 explains *why* (corollary C3) and §6 corrects MECHFIX's own
explanation. Every remaining member of the family (mutual proximity, NICDM, QB-Norm, NNN) changes only
*which* items are retrieved, and **§2 Result B's marginal oracle for that is +0.0067 / +0.0000 / +0.0018.**
**Family KILLED here, on oracle, per the tasking's own rule.** This also supersedes `LITSWEEP6_MEMBANK.md`
§7.1's argument, which closed only NNN and only by a rank-invariance argument; §2 closes the whole family
including the set-changing members that argument did not reach.

### 3.4 ★ THE MATCH FOR THE CONE AND FOR THE F89-T2b FAILURE — *neural collapse* + *isotropy-cluster incompatibility*

| paper | id / venue | relevance |
|---|---|---|
| Papyan, Han & Donoho, *Prevalence of Neural Collapse during the terminal phase of deep learning training* | **PNAS 117(40):24652-24663 (2020)**, arXiv:**2008.08186** — verified | **NC1:** during the Terminal Phase of Training (begins when training error first vanishes), "cross-example **within-class variability of last-layer training activations collapses to zero**, as the individual activations themselves collapse to their class-means." NC2: class means → simplex ETF. NC4: the decision collapses to nearest-class-mean. |
| Mickus, Grönroos & Attieh, *Isotropy, Clusters, and Classifiers* | **ACL 2024**, arXiv:**2402.03191** — abstract fetched verbatim | "**isotropy imposes requirements on the embedding space that are not compatible with the presence of clusters — which also negatively impacts linear classification objectives.** We demonstrate this fact both mathematically and empirically." |
| Forooghi, Sadeghi & Lu, *Whitening Not Recommended for Classification Tasks in LLMs* | arXiv:**2407.12886** (2024-07-16; venue not stated on page) — abstract fetched verbatim | "**whitening degenerates embeddings for classification tasks**" — across PCA, ZCA, PCA-Cor, ZCA-Cor and Cholesky whitenings, model- and task-dependent. |
| Timkey & van Schijndel, *All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality* | **EMNLP 2021**, ACL Anthology `2021.emnlp-main.372`, arXiv:**2109.04404** — verified | 1-3 "rogue dimensions" dominate cosine similarity and are mismatched to the dimensions that matter for behaviour; **standardization** (diagonal, not full whitening) corrects them. |
| *Isotropy Matters: Soft-ZCA Whitening of Embeddings for Semantic Code Search* | arXiv:**2411.17538** — title/id from search result, **abstract UNVERIFIED** | Soft-ZCA adds an eigenvalue regulariser `ε` precisely because near-zero eigenvalues make `Λ^{-1/2}` explode and whitening amplifies noise directions. |

**Does the diagnosis match? Yes, and it is the sharpest predictive match in the sweep.**

1. **The cone is manufactured by the head, and NC says so.** New measurement (§1, row 2a): the **raw**
   encoder space is *not* collapsed (median top-1 cosine 0.9459/0.9537/0.9416, rank-1→20 spread ~0.021-0.025).
   The 0.9999 collapse exists only in the **trained head** space, and F47 records the head's train LOO
   accuracy at **0.998** — i.e. the head is *in* the Terminal Phase of Training. **NC1 predicts exactly the
   0.9999.** It also predicts something the campaign has been treating as an anomaly: the memory bank is
   built from **train** features, so under NC1 the bank is approximately **two points plus noise**.
2. **NC predicts that the residual directions are nuisance — which is exactly why F89-T2b failed.** If
   within-class variability has collapsed, the surviving small-eigenvalue directions are by construction
   the ones the objective did *not* use. Ledoit-Wolf whitening at `d = 1024 > n` has shrinkage ~0.0004-0.0027
   and an eigen-condition of 2.4e6, "so the whitener amplifies near-null eigendirections ~1000×"
   (`MECHFIX:237,427`) — and the length nuisance duly rises from ρ≈0.52 to ρ≈0.87 (`MECHFIX:288-290`).
   **This is a published, replicated failure mode**, not a project-specific accident: arXiv:2407.12886
   measured whitening degrading *classification* across five whitening variants, and arXiv:2402.03191
   proves mathematically that isotropy is incompatible with clustered spaces and harms linear classification.
   **F89-T2b is a re-derivation of published science.** The paper should say so and cite both.
3. **The brief's specific question — "what restored isotropy WITHOUT surfacing a nuisance axis?" — has an
   answer, and it does not help us.** The literature's answer is *do not fully whiten*: use **diagonal
   standardization** (Timkey & van Schijndel) or **shrinkage/Soft-ZCA** with an explicit `ε`
   (arXiv:2411.17538). MECHFIX's own limitation 3 (`:472`) concedes "a ridge-regularised direction, a
   whitened-space direction, or a multi-dimensional length subspace were **not** tested". So these are
   legal, un-run, and $0. **But they self-kill on §2:** they are re-metrications, i.e. they change only
   *which* 20 items are retrieved and in what order, and Result B prices that family at
   **+0.0067 / +0.0000 / +0.0018**. **KILLED here on oracle.**

**Does NC theory predict which interventions can and cannot work?** Yes, and its answer is the campaign's
`diagnosis_frame` verbatim: NC is a property of **the trained map**. Interventions *downstream* of the
collapsed features (re-metrication, reweighting, re-ranking, vote surgery) operate on information the
training objective already deleted; interventions *upstream* (a different encoder / a different training
objective / stopping before TPT) change what is deleted. **The only class of lever that ever cleared +3 in
this campaign is the upstream one (encoder swap, HateMM).** NC explains why, arithmetically.

**Honest note on the one upstream intervention NC suggests:** stop before the Terminal Phase (early-stop /
variability-preserving regularisation). It is **not** clean: F62b killed SWA, F79 quantified "Wall-C"
(HateMM test peaks at epochs 18/21/24 of 29 and ZH final-epoch > val-selected), the epoch-selection axis is
already the contested protocol question, and a head-loss regulariser is F75-family and **D7-novelty-dead**.
I do not propose it; I record that NC is what would motivate it.

### 3.5 ★ THE MATCH FOR SEED-INVARIANCE — *long-tail memorization*

* Feldman, V., *Does Learning Require Memorization? A Short Tale about a Long Tail*, arXiv:**1906.05271**
  (STOC 2020 — **venue UNVERIFIED**). Thesis: natural data distributions are long-tailed with rare,
  atypical examples, and **memorization of those examples is necessary for close-to-optimal generalization**.
* Feldman, V. & Zhang, C., *What Neural Networks Memorize and Why: Discovering the Long Tail via Influence
  Estimation*, **NeurIPS 2020**, arXiv:**2008.03703** — verified. Empirically confirms it via
  subsampled-influence estimation.
* Related and worth citing for the seed-invariance measurement itself: Jiang et al., **C-score**
  (consistency score), arXiv:**2002.03206** — **id and claim UNVERIFIED this sweep**.

**Does the diagnosis match?** Yes, and it predicts three of our measurements at once:

1. **Seed-invariance.** ~90 % of errors wrong in 3/3 seeds (`ERRPAT_HateMM:110-113`, `ERRPAT_MHC-ZH:203`).
   Under Feldman's account, long-tail singletons are the examples whose correct classification *requires
   memorising them*; a test item cannot be memorised, so it is wrong under every training run. Seed
   variance touches ≤5 items of 215 — exactly the "everything else is safe" prediction.
2. **The right analogue at rank ~1.5, and still out-voted.** Feldman-Zhang's influence estimation says the
   *single* closest sub-population member carries almost all of the memorization value. That is our
   measurement: the correct analogue is there, at rank 1-2, and it is one item against nineteen.
3. **Large oracle, no legal operator.** The oracle knows which of the 20 is the sub-population twin. No
   label-free statistic identifies it, because by construction the sub-population has ~1 member — there is
   no density for a density-based statistic to see. **This is the mechanism behind F66's selection lock and
   §2's C6, expressed as a data-distribution property rather than an operator property.**

**What fixed it there?** Nothing operator-side. Feldman's theory says memorization is *necessary*; the only
remedies are more data from the sub-population (ban [8]) or explicit example-level memorization (impossible
for held-out items). **The theory predicts that nothing in our legal class can work.**

### 3.6 ★ kNN-LM / RAG — *retrieval quality is not what makes retrieval augmentation work*

This is the field with the **closest independent replication of law-I itself**, and all three headline
papers were fetched and confirmed verbatim this sweep.

| paper | id / venue | verified finding |
|---|---|---|
| Xu, F. F., Alon, U. & Neubig, G., *Why do Nearest Neighbor Language Models Work?* | arXiv:**2301.02828** (7 Jan 2023, rev. 17 Jan 2023). **Page states "Preprint, 21 pages" — no venue; do NOT cite a venue.** | Asks why kNN-LM beats a parametric LM *"even when the k-nearest neighbor component retrieves examples from the same training set that the LM was originally trained on"* — our exact setting (bank = own train split). Conclusion: the gain is **not** better retrieval. The three identified causes are **a different input representation for predicting the next token, approximate kNN search, and the softmax temperature of the kNN distribution**. They then fold these into the parametric LM and get the gain **"without the need for an explicit retrieval component."** |
| Cuconasu, F. et al., *The Power of Noise: Redefining Retrieval for RAG Systems* | arXiv:**2401.14887** (26 Jan 2024, rev. 1 May 2024). **Page lists cs.IR/cs.CL only — no venue on page; do NOT cite SIGIR without checking.** | *"the retriever's highest-scoring documents that are not directly relevant to the query … negatively impact the effectiveness of the LLM. Even more surprising, we discovered that **adding random documents in the prompt improves the LLM accuracy by up to 35 %**."* |
| Salemi, A. & Zamani, H., *Evaluating Retrieval Quality in Retrieval-Augmented Generation* | **SIGIR 2024**, arXiv:**2404.13781**, DOI `10.1145/3626772.3657957` | *"Evaluation of the retrieval model's performance based on query-document relevance labels shows a small correlation with the RAG system's downstream performance."* Their fix, **eRAG**, redefines relevance **as downstream utility**: score each retrieved item by running it through the consumer. |

**Does the diagnosis match? Yes — this is law-I, published, in a different field, three times.** Xu et al.
is the strongest match: a retrieval-augmented system whose gains are traced to everything *except* retrieval
quality, in the same own-train-datastore configuration we run. Cuconasu et al. is the strongest form of the
"ranking metric ⊥ decision utility" claim anywhere in the literature — ranking *higher* on relevance can be
actively worse.

**Which law-I data does it explain?** It reframes them all: the campaign has been assuming retrieval quality
is the causal channel and measuring surprise when improving it converts nothing. Xu et al. says that
assumption is wrong for retrieval-augmented models generally. §2 makes it exactly true for ours.

**What fixed it there, and is the fix legal?** Three fixes, and each maps onto something already dead here:

1. **Xu et al.'s "different input representation"** ⇒ a representation-class lever. That is the *one* class
   that ever cleared +3 in this campaign (encoder swap, HateMM) and it is D7-constrained. Consistent with the
   project's standing `diagnosis_frame`, which §2 independently re-derives.
2. **Xu et al.'s "softmax temperature of the kNN distribution"** ⇒ a monotone reshaping of the vote weights.
   That is **exactly the VSW λ-continuum** (`VSW_PREGATE_RECORD.md:748-777`), measured across a 16 000× range
   with the net capped below the bar at every point. **Already measured dead.**
3. **eRAG's per-item downstream-utility relabelling** ⇒ per-entry bank weights = LITSWEEP6 C5, whose binary
   realisation fails its own random-deletion control (`ERRPAT_HateMM:390-396`). **Already measured dead.**

**One important disanalogy, stated so the paper does not over-claim.** In RAG the consumer is an LLM with an
opaque utility function, so the retrieval-utility mismatch is an *empirical* finding. **In our system the
consumer is a rank-weighted label majority and the mismatch is exactly computable** (§2 C1) — we can prove
what eRAG can only estimate. That is a contribution, not a limitation.

**What fixed it there?** eRAG's fix is to *train/select the retriever against downstream utility*. Our
analogue is a per-entry downstream-utility weight — which is LITSWEEP6 C5, already ranked last with survival
"low" (`LITSWEEP6_MEMBANK.md:634`), and whose binary realisation (drop LOO-disagreeing bank rows) measured
**+0.0016 against a random-deletion control of +0.0031** — i.e. it **fails its own random control**
(`ERRPAT_HateMM:390-396`). **Not recommended; not proposed.**

### 3.7 RANKING-VS-ACCURACY — *AUC is incoherent across classifiers*

* Hand, D. J., *Measuring classifier performance: a coherent alternative to the area under the ROC curve*,
  **Machine Learning 77(1):103-123 (2009)** — verified. Argument: **AUC is incoherent because it uses a
  different implicit misclassification-cost distribution for each classifier**; comparing two classifiers'
  AUCs is comparing them under two different metrics. Fix: the **H-measure** (fix a Beta(2,2) cost prior).
* Hernández-Orallo, Flach & Ferri, *A Unified View of Performance Metrics: Translating Threshold Choice into
  a Loss Function*, JMLR 2012 — **UNVERIFIED this sweep**. Its relevance is that AUC maps to expected loss
  **only under a specified threshold-choice method**, so an AUC gain need not convert at a fixed threshold.
* Cortes & Mohri, *AUC Optimization vs. Error Rate Minimization*, NIPS 2003 — **UNVERIFIED this sweep**.

**Does the diagnosis match?** For the *temporal-split* datum, **yes, and the record already says so and
already fixed it**: `EVAL_temporal_memory_W4.md:44` calls the −0.084 macro-F1 at ROC 0.8484 "an
**operating-point (calibration) failure, not a separability failure**", and threshold recalibration on 20
labelled samples recovers the entire drop (`:87,92-96`). **This is the campaign's single successful
ranking→decision conversion, and it is a Hand-style fix.**

For the *random-split* data it does **not** apply, and saying so is important: F88 measured the
**test-fitted** threshold oracle at ZH **+0.0201 < bar**, HateMM threshold-recalibration dead, and EN
dev-selected thresholds deployably dead (`LITSWEEP6_PARADIGM.md:376-379`). **Calibration is not the binding
constraint on the in-distribution splits** — §2 says why: the vote is `sign(M)` with `|M|` median 0.42-0.72,
so the errors are not near any threshold.

---

## §4. THE NAMING QUESTION — a direct answer

> **Is there a published name and theory for "a retrieval system whose ranking quality far exceeds its
> decision quality"?**

**Partly. The composite phenomenon is unnamed; three of its four components are named, and the fourth — the
one that is actually load-bearing for us — is not a phenomenon at all but a design property.**

**(a) NO name exists for the composite.** Targeted searches for a term covering "retrieval ranks well but
the retrieval-augmented classifier decides badly" returned nothing: no "retrieval-decision gap",
"ranking-decision gap", "aggregation bottleneck" or equivalent is established terminology. The nearest
published framings are descriptive, not nominal — eRAG's "retrieval metrics show small correlation with
downstream performance" (arXiv:2404.13781) and the metric-learning "reality check" genre. **If the project
wants a name, the slot is open.**

**(b) The components that ARE named:**

| our symptom | published name | citation |
|---|---|---|
| reweighting a fixed neighbourhood cannot help | *(Bailey-Jain theorem / optimal weighted NN)* | Bailey & Jain 1978; Samworth 2012, arXiv:1101.5783 |
| the error population (purity 0.12-0.22) | **rare** and **outlier** minority examples | Napierała & Stefanowski 2016 |
| item-level offsets dominating the similarity | **hubness**, and **bad hubs** | Radovanović et al., JMLR 2010 |
| the 0.9999 train cone | **neural collapse (NC1)** | Papyan, Han & Donoho, PNAS 2020 |
| whitening de-collapses but hurts | **isotropy-cluster incompatibility**; "whitening not recommended for classification" | Mickus et al., ACL 2024, arXiv:2402.03191; Forooghi et al., arXiv:2407.12886 |
| stable confident errors on singletons | **long-tail memorization** | Feldman arXiv:1906.05271; Feldman & Zhang arXiv:2008.03703 |
| AUC gains that do not convert | **AUC incoherence** / threshold-choice dependence | Hand, Mach. Learn. 2009 |
| retrieval-augmentation gains that do not come from retrieval | *(no name; stated as a finding)* | Xu, Alon & Neubig, arXiv:2301.02828; Cuconasu et al., arXiv:2401.14887; Salemi & Zamani, SIGIR 2024, arXiv:2404.13781 |

**(c) The load-bearing part is not a phenomenon — and that IS the paper-level result.** §2 Result A shows
the gap in *our* system is not an emergent pathology at all: **the deployed decision is a function of the
retrieved ordered label tuple alone, so every similarity-space improvement that does not change that tuple
is exactly in the kernel of the decision map.** Combined with Result B (the tuple already has ~100 %
coverage), the gap is *forced*. I could find no paper that states this for retrieval-augmented
classification. **A precise, defensible name for it — with the arithmetic of §2 and the theorems of §3.1
behind it — is available and, as far as this sweep can determine, unclaimed.** Suggested framing, offered
for the user's judgement, not asserted:

> **The label-tuple bottleneck.** In a retrieval-augmented classifier that decides by a weighted vote over
> the labels of `k` retrieved exemplars, the representation influences the decision only through the
> *ordered label tuple* it induces. Once neighbourhood coverage saturates, further improvement to the
> similarity function is decision-invariant by construction, and the residual headroom is confined to a
> weighting problem that Bailey-Jain and Samworth bound to a vanishing improvement in high dimension.

That statement is falsifiable, it is measured on three datasets (§2), it has classical theory behind it, and
it explains the campaign's eight certified law-I data on the decision-side axis without appeal to any
project-specific fact.

---

## §5. LEGALITY AUDIT AND CANDIDATE DISPOSITION

Bans checked against `autoresearch/goal_mllm_plus3/state/directions_tried.json → banned_constraints`
(0-indexed, read this sweep): [0] OCR, [1] gold annotations in method, [2] cross-seed ensembles,
[3] kNN-vote-pool expansion via pseudo-labels, [4] target-as-structure at 7B, [5] MLLM-scores-as-training-signal,
[6] P1-P5 re-proposals, [7] external model APIs, [8] single-dataset own-train-split only. Plus D7
(encoder-class levers do not satisfy novelty) and the pre-closed lists at `LITSWEEP6_RELGEN.md:368-386` and
`LITSWEEP6_PARADIGM.md:371-390`.

### 5.1 SELF-KILLED BY ORACLE (per the tasking's rule) — one whole family

**CAND-A: the candidate-set / re-ranking family.** Members: k-reciprocal encoding (Zhong et al., CVPR 2017,
arXiv:1701.08398), mutual proximity, local scaling / NICDM, QB-Norm, NNN, Soft-ZCA / diagonal
standardization / any re-metrication, and any `k > 20` pool expansion.
**Largest oracle I can derive for the family:** the *marginal* free-weight oracle of expanding the pool from
the deployed 20 to 400 = **+0.0067 (HateMM) / +0.0000 (MHC-ZH) / +0.0018 (MHC-EN)** (§2 Result B).
**KILL — under +0.030 on 3/3, by a factor of 4.5× on the best dataset and infinitely on ZH.** No re-ranker
can beat "the correct-label neighbour is already in the top-20 for 99.3-100 % of items". This closes
`LITSWEEP6_MEMBANK.md` §7.1's family completely rather than partially, and it retires the hubness axis.

### 5.2 PRE-CLOSED OR ALREADY PRICED (recorded so no future sweep re-derives them)

| candidate the pathology lens surfaces | status |
|---|---|
| Bailey-Jain-optimal / Samworth-optimal weight profiles | = F98/AGGNET bar-2 fixed monotone profile family, measured **−0.0027** (`AGGNET:392`); and Samworth says the optimum's advantage → 0 as `d → ∞`. **Dead.** |
| eRAG-style per-entry downstream-utility weights (LITSWEEP6 C5) | binary realisation measured **+0.0016 vs random-deletion control +0.0031** → fails its own control (`ERRPAT_HateMM:390-396`); LITSWEEP6 already ranks it last. **Do not spend.** |
| NC-motivated early-stop / variability-preserving regulariser | F62b (SWA) dead, F79 Wall-C anti-aligned, F75-family loss engineering, **D7-novelty-dead**. Recorded, not proposed (§3.4). |
| threshold / operating-point moves at full coverage | F88: ZH test-fitted threshold **oracle +0.0201 < bar**; HateMM dead; EN deployably dead. **Dead on the goal**; alive only as the temporal-drift deliverable already scoped at `LITSWEEP6_PARADIGM.md` R2. |
| conformal / selective prediction variants | pre-closed for a full-coverage accuracy bar (`LITSWEEP6_MEMBANK.md:695-707`); I found **no** variant that lifts full-coverage accuracy. |
| negative vote weights (Samworth's own escape) | requires knowing a neighbour is wrong = per-item selection = **F66 Law-III banned**. |

### 5.3 THE ONE THING WORTH RUNNING — a $0 falsification of §2, not a goal bet

§2 is measured in the **raw** train-LOO arena (the F95/AGGNET arena). The deployed decision lives in the
**head** space, where `ERRPAT_HateMM:135` reports 6/27 test errors with **zero** true-label neighbour in the
top-20 — i.e. head-space coverage on the error population may be materially worse than raw-space coverage.
**If head-space coverage at k=20 is well below 99 %, my kill of CAND-A weakens.** That is the honest
falsification test and it is the only experiment this sweep recommends.

**PREGATE SKETCH — HEADCOV (cost: $0, CPU ≤8 threads, zero GPU/SLURM/Modal, TRAIN SPLIT ONLY).**

* **Arena.** The F89-frozen machinery: `scripts/analysis/mechfix_ops.py`
  (sha256 `635c1312…c83fc8d`, 15/15 floor-parity gates passed) with the ERRPAT CPU proxy heads for HateMM
  and MHC-ZH. **MHC-EN cannot be run** — its head checkpoint is gone (`MECHFIX:462-465`); declare EN
  out-of-scope *before* running, do not shop for it afterwards.
* **Measure, on train items under full-bank LOO in head space, exactly the three §2 quantities:**
  (i) decision-identity between the deployed vote and the label-majority-only vote `M`;
  (ii) coverage (≥1 true-label item) at pool 20 / 50 / 100 / 400;
  (iii) the marginal free-weight oracle of pool expansion 20→400.
* **Frozen kill switches (declare before any number exists):**
  * **K-HC-1 (confirms §2, closes CAND-A permanently).** Marginal pool-expansion oracle **< +0.030 on both
    runnable datasets** ⇒ the re-ranking/candidate-set family is **CLOSED at $0**, in the deployed space,
    and §2's arithmetic goes in the paper as a measured result rather than a raw-space inference.
  * **K-HC-2 (falsifies §2).** Marginal pool-expansion oracle **≥ +0.030 on ≥1 dataset** ⇒ §5.1's kill is
    **WITHDRAWN**, and exactly one re-ranker (k-reciprocal encoding, as the canonical set-changing operator
    that is neither CSLS nor label-propagating) earns a further $0 pregate under the F98 degeneracy-control
    discipline (DEG-A threshold twin, DEG-B fixed-k twin, DEG-D cosine twin, class balance, permutation null).
  * **K-HC-3 (integrity).** Decision-identity between the deployed vote and `M` must be **≥ 0.98** in head
    space. If it is not, §2 Result A does not transfer and *this record's central claim is wrong* — say so,
    loudly, and re-open the metric axis.
* **Expected outcome, stated in advance:** K-HC-1 fires. The head space is *more* collapsed than the raw
  space (cosine spread 1e-4 vs 2e-2), so Result A's margin is ~100× larger there, and ERRPAT's own
  `median rank of first TRUE-label neighbour = 3.0` on errors (`ERRPAT_HateMM:134`) implies coverage is high
  in head space too — the 6/27 zero-coverage errors are 2.8 % of test items, consistent with 97-99 % coverage.
* **What a pass buys:** a door-closer and a paper section. **Not** a performance lever. Nothing in §5 can
  clear +0.030 on ≥2 datasets, and this sweep does not claim otherwise.

---

## §6. CORRECTIONS TO THE IN-REPO RECORD (four, all load-bearing)

1. **`MECHFIX_PREGATE_2026-07-27.md:235,426` — the explanation of T2a's null is wrong.** It says the CSLS
   hubness term "has almost no dynamic range — the cone collapse leaves nothing to correct with"
   (`r(x)` IQR 1.0-1.4e-4). But (a) that IQR is the same order as the *cosine's own* spread in that space,
   (b) the same record states T2a "changes half the retrieved sets" (`:426`), and (c) F95 §4.1 — written
   *after* MECHFIX — measures the bank main effect at **31-44 % of the cosine's score variance**
   (`MECHNOV_PAIRVERIFY_PREGATE.md:424-429`). CSLS is not inert for lack of dynamic range; **it is inert
   because the decision does not read the similarity at all** (§2 C1/C3). The corrected sentence is
   *stronger*, not weaker: *hubness reduction is measured inert on the decision even though hubness carries
   a third to a half of the similarity's variance.* Arena caveat: MECHFIX T2a is head-space, F95 §4.1 is
   raw-space; the direction is unaffected but a same-arena confirmation is exactly what §5.3's pregate buys.
2. **"Ten certified law-I data" over-counts the ledger, which certifies eight** (F50 "5th", F63 "SEVENTH",
   F65 "8th", **F87 "9th law-I NOT certified"**). Fix the count before it reaches the paper.
3. **VSW is not "in progress" — it has written `KILL as a performance lever`** (`VSW_PREGATE_RECORD.md:902`),
   with DEG-A **and** DEG-B both firing on MHC-ZH and K-VSW-1 declared "arithmetically unreachable"
   (`:920-928`). It should not be described as a live exception.
4. **The cone collapse is head-induced, not encoder-inherited** (§1 row 2a, new measurement). Every
   statement of the form "our features live on a 0.9999 cone" must be scoped to the trained head space; the
   raw Qwen keys sit at median top-1 cosine 0.94-0.95 with a 0.021-0.025 top-20 spread. This matters because
   it moves the collapse from "a property of the encoder we inherited" to "a property of the objective we
   chose", which is what makes the neural-collapse citation apply.

---

## §6b. POST-FREEZE ADDENDA (2026-07-28, after this record was first written and committed at `2e2805f`)

Three pieces of evidence landed after §0-§7 were frozen. **Nothing above was edited**; the additions are
recorded here so the ordering is auditable.

**A1 — §2 Result A is confirmed in the DEPLOYED HEAD SPACE.** `refine-logs/HEADCOV_PREGATE_RECORD.md`
(F107) ran the §5.3 falsification, with bars frozen before any treatment number. Result A's decision
identity is **1.0000** on MHC-ZH dev (0/78 differing items in each of 3 seeds, final epoch; 0.9989 over all
90 seed×epoch cells, min 0.9872), and Result B's coverage(20) is **0.9829**, bounding the pool-expansion
oracle at **+0.0171** — so **§5.1's kill of CAND-A stands in the space the system actually retrieves in**,
and the pre-registered falsifier K-HC-2 did not fire. Two honest scope notes: HateMM and MHC-EN are **out
of scope by instrument availability** (all deployed head checkpoints are deleted — 97 empty `ckpt/` dirs,
0 of 9 P2-era ckpts extant, which confirms and *extends* F78 beyond HateMM), and the head-space identity
is **forced by collapse** (spread 1.95e-04 vs `M`'s smallest step 0.0095), so the *mechanism* evidence for
"this is a property of the decision rule" remains the **raw**-arena measurement in §2, not the head-space one.

**A2 — §2 Result A is independently corroborated in the raw arena.**
`refine-logs/VSW_ASYMMETRY_RECON.md` §5 deletes the cosine magnitude from the deployed vote outright
(`cos_i := 1`) and measures Δacc **−0.0013 / +0.0000 / −0.0018** at **99.60 / 99.65 / 99.82 %** agreement
with the deployed decision. That is the same conclusion as §2 Result A, reached by a different agent with
a different operator. **Three independent measurements across two arenas now support the kernel argument.**

**A3 — §3.1's Samworth citation has become a measurement (F105/VSW, commit `e9a17fe`).** VSW's
pre-declared door-closer, outcome (b) — "the exchange rate is bounded below 1 across the continuum" — was
measured **FALSE**: the exchange rate reaches **6.0** on HateMM (against F95's best-of-36-cells 1.1667) and
VSW *still* fails. The corrected law is **`net = changed × (2·precision − 1)`**, with precision decaying
monotonically as sharpness rises (HateMM 0.8571 at 21 changed → 0.5696 at 79 changed), pinning the net to
**+11…+21 items across a 16 384× λ range**. **This is §3.1's theorem at finite `n`.** VSW's λ-continuum is
a search over exactly the weighted-NN family Samworth optimises analytically; Samworth says the family's
best member beats uniform by an amount that vanishes as `d → ∞`, and VSW measures that the best member's
*net* is capped below the bar at every point *even where its per-item exchange rate is excellent*. A high
rate with a capped net is what a vanishing weighting advantage looks like when it can only be spent on a
shrinking population. **Anyone citing "the exchange rate never exceeds ~1.2" as a law of this system must
stop** — that was withdrawn by F105.

**A4 — determinism scope.** A confirmed defect makes frozen modules reproduce closed-form quantities
bit-exactly but drift on **trained** ones (44 of 48 in the F95 module; oneDNN/MKL kernel selection).
**No number in §2 or §2.3 of this record comes from a trained estimator** — they are closed-form numpy
over banked feature caches — so no tolerance caveat attaches to them. The same holds for every number in
HEADCOV (F107), which re-fits nothing.

---

## §7. PROVENANCE AND COMPLIANCE

**In-repo sources read this sweep (all read-only):**
`refine-logs/LITSWEEP5_COMPLETENESS.md`; `refine-logs/LITSWEEP6_{MEMBANK,PARADIGM,RELGEN}.md`;
`refine-logs/MECHNOV_PAIRVERIFY_PREGATE.md`; `refine-logs/MECHFIX_PREGATE_2026-07-27.md`;
`refine-logs/AGGNET_PREGATE_RECORD.md`; `refine-logs/ISR_PREGATE_RECORD.md`;
`refine-logs/ERRPAT_{HateMM,MHC-ZH,MHC-EN}_2026-07-26.md`; `refine-logs/VSW_PREGATE_RECORD.md` (read only,
after its own §9 verdict existed); `refine-logs/CAND2_{CURRICULUM_RECON,REP2_VERDICT_REVIEW}.md`;
`research-wiki/EVAL_temporal_memory_W4.md`; `src/utils/metrics.py:258-307`;
`autoresearch/goal_mllm_plus3/state/{directions_tried.json,findings.jsonl}`.
**`refine-logs/LITSWEEP7_LANDING_SITE.md` was not opened**, per the tasking.

**Computation performed (§2):** three numpy snippets over
`data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_*.pt`, ≤8 CPU threads, ~90 s total, **no files written, no
model run, no checkpoint loaded, no dev/test split opened**.

**External sources — verification status.**

*Fetched, abstract read verbatim this sweep (safe to cite as stated):* arXiv:**1101.5783** (Samworth, Ann.
Statist. 40(5):2733-2763); arXiv:**2402.03191** (Mickus et al., ACL 2024); arXiv:**2407.12886** (Forooghi
et al., **no venue on page**); arXiv:**2301.02828** (Xu, Alon & Neubig, **"Preprint, 21 pages" — no venue on
page**); arXiv:**2401.14887** (Cuconasu et al., **cs.IR/cs.CL only — no venue on page**).

*Confirmed via search-result metadata from primary hosts (JMLR / PNAS / ACL Anthology / Springer / ACM):*
JMLR 11:2487-2531 (2010); PNAS 117(40):24652-24663 + arXiv:2008.08186; ACL Anthology `2021.emnlp-main.372` +
arXiv:2109.04404; DOI 10.1007/s10844-015-0368-1; Machine Learning 77(1):103-123 (2009); arXiv:2404.13781 +
DOI 10.1145/3626772.3657957; arXiv:1906.05271; arXiv:2008.03703 (NeurIPS 2020).

*Explicitly **UNVERIFIED** and flagged in place — do not cite without fetching first:* Bailey & Jain 1978
(pre-arXiv; the *claim* is corroborated by ≥2 independent secondary sources, the primary was not fetched);
MacLeod et al. re-examination (venue/year); Napierała-Stefanowski's exact 0.8/0.5/0.2 thresholds; Skryjomski
PMLR v74 authorship; Schnitzer JMLR 2012; Conneau CSLS; QB-Norm arXiv:2112.12777; NNN arXiv:2410.24114
(id carried over from `LITSWEEP6_MEMBANK.md:658`); Soft-ZCA arXiv:2411.17538; k-reciprocal arXiv:1701.08398;
Hernández-Orallo/Flach/Ferri JMLR 2012; Cortes & Mohri NIPS 2003; Jiang et al. C-score arXiv:2002.03206.

**Two venue corrections to guard against:** Xu et al. (2301.02828) is a **preprint** on its arXiv page — the
common attribution to ICML 2023 is *not* supported there. Cuconasu et al. (2401.14887) shows **no venue** on
its arXiv page — the common attribution to SIGIR 2024 is *not* supported there.

**Sweep scope note.** Four field-specific search agents (kNN-LM/RAG, deep metric learning/hubness,
calibration/AUC, neural collapse/OOD/few-shot) were dispatched in parallel at the start of this sweep and had
not returned by the time the record was frozen; every citation above was therefore verified by this agent
directly. §3.7 (calibration) is the thinnest section and is the one to extend first if the parent wants more.

**Required statements.** ZERO GPU / SLURM / Modal / training / test-touch spent. No held-out test metric was
read or produced. No `state/`, prereg, config, `research-wiki/`, or frozen artifact was mutated. No entry
appended to `findings.jsonl`. **Not committed** (adjacent paths are being committed by other agents).

