# EXP_p7_score_fusion — score-level fusion of the MLLM channel with the kNN vote

**Status:** **DONE — TRAIN-SIDE KILL, no test contact. The "decorrelated channels" premise is
empirically REFUTED (channel↔vote corr +0.21…+0.51, positive), and every frozen rule×channel damages
more LOO errors than it corrects (net −0.10…−0.38). Score-level fusion earns no method role.** ·
**Started:** 2026-07-07 · **Finished:** 2026-07-07 · **Owner:** subagent P7

**Question (campaign mandate).** Can a score-level combination of the visual-embedding **kNN vote**
and the MLLM's **zero-label semantic channel** earn main-table accuracy? This is one of the two
never-tested routes: every prior front touched embeddings (P3/P4), membership (P2/P2b), or the
threshold (P1); **none fused at the score level**. Premise: the kNN vote (visual similarity) and the
MLLM verdict (speech/semantic) are plausibly **decorrelated error channels**, so a late fusion could
correct the vote exactly where it is wrong.

**Discipline:** a mandatory **train-side gate** (no test contact) decides whether any test measurement
happens. If the channel does not demonstrably correct the vote's errors on train, this is killed
train-side and the correlation numbers are reported (they settle the "decorrelated channels" claim).

---

## 0. Pre-registration (frozen before any gate/test run)

### 0.1 Assets (all on disk; gate is 100% CPU)

- **Winner heads** = the 9 val-selected **archive-kNN α=0.25** heads (EN `MHC` seeds 0-3, ZH
  `MHC_zh` seeds 0-4), the identical checkpoints of `exp-archive-knn-seeds.md` /
  `EXP_auto_memory_repair.md`, loaded via `scripts/analysis/p2_rerank_eval.py`
  (`CKPT_FILE`, `build_head`, `project_split`, `augment` α=0.25, `retrieve`, `sim_vote`, `vote_pred`).
  The vote is therefore **bit-identical to the logged floor**.
- **kNN vote share** `s_v(x) = sigmoid(vote)` ∈ [0,1] = P(hateful); `vote` = top-20 similarity-signed
  rank-weighted arithmetic vote over the archive-augmented train memory. Floor decision = `s_v ≥ 0.5`
  (⇔ vote ≥ 0). `margin = |vote|`.
- **MLLM channel** `c(x)` ∈ [0,1], two pre-registered variants:
  - `bin`  = P1 HARMFUL/BENIGN verdict (`scripts/analysis/p1_out/harmful_verdicts.json`,
    `d[ds]["v2"][id]`): HARMFUL→1, BENIGN→0. Available **EN + ZH, all splits**.
  - `dens` = P3 mean per-segment density / 3 (`data/MLLM_scores/<ds>/<split>_segscoreK4_qwen.jsonl`,
    mean of the K scores ÷ 3). Available **EN train, ZH all, HateMM all**; EN dev/test missing (one
    small scoring job — fired ONLY if the gate passes and `dens` beats `bin`).
- HateMM has no archive-kNN winner heads (archive heads are EN/ZH only), so HateMM is out of scope
  for the fusion test; its `dens` channel is available if a HateMM head set is later provided.

### 0.2 Fusion rules (freeze exactly TWO; parameter-free except one pre-frozen constant)

- **R1 — rank-average (operating point inherited from the floor).** Within an eval set: `r_v` =
  rank-percentile of `s_v`, `r_c` = rank-percentile of `c` (ties → mid-rank); `fused = (r_v+r_c)/2`.
  Decision: predict hateful for the **top-N_pos** videos by `fused`, where `N_pos` = number of
  floor-predicted positives on that set (ties broken by `s_v`). Inheriting `N_pos` from the floor
  keeps the positive rate fixed, so R1 isolates the *re-ranking*; **no tunable parameter**.
- **R2 — band-limited veto-boost.** Uncertain band `U = {margin < τ}`, `τ` = 25th percentile of the
  seed's VAL margins (the campaign's Role-3 / P2 deferral band, pre-registered). For `x ∈ U`:
  `fused_score = s_v + 0.25·(c − 0.5)`, decision `fused_score ≥ 0.5`. For `x ∉ U`: floor decision
  unchanged. The `0.25` and the 25% band are pre-frozen; nothing tuned on test.

### 0.3 TRAIN-SIDE GATE (mandatory, seed 0 only, NO test contact)

On TRAIN, LOO kNN vote share with the seed-0 winner head (each train video retrieves top-20 over the
OTHER train videos in the augmented memory; self excluded). For each `(dataset, channel)`:

- **(a) Net-of-damage LOO error correction** (the hard bar). Simulate each fusion rule on the train
  LOO predictions vs floor LOO predictions: `corrected` = # LOO floor-errors that flip to correct;
  `damaged` = # LOO floor-corrects that flip to wrong; `net = corrected − damaged`. Report
  `net / n_errors`. For R2 the band `τ` uses the **train LOO** margins' 25th percentile (train-only).
  **Promotion bar (frozen):** some fusion rule achieves `net > 0` AND `net / n_errors ≥ 0.15`.
- **(b) Diagnostics** (report always; they settle the premise): AUC(`c`, gold), point-biserial(`c`,
  gold), and Pearson corr(`c`, `s_v`) — the decorrelation premise wants **low |corr|** with a
  **decent AUC**.

**Gate decision:** only `(dataset, channel, rule)` combos that clear the bar in (a) proceed to test.
If nothing clears on a dataset → **kill that dataset train-side**, report (a)+(b), no test contact.

### 0.4 TEST (only for gate-passing combos; one measurement per cell)

All applicable winner heads (EN s0-3, ZH s0-4), BOTH protocols (val-selected + final-epoch are not
applicable — these are fixed val-selected heads; instead report the single logged head per seed),
conditions: **A** floor (vote only), **B** fused R1, **C** fused R2, **D** random-channel control
(`c` replaced by a seeded label-free permutation of the same marginal). **B/C must beat D.** Metrics:
acc + macro-F1 overall (and on the gated subset for R2). Per-seed paired deltas B−A, C−A, B−D, C−D;
mean±std.

**Success (frozen campaign bar):** mean gain > 1 pt beyond the ~1-video noise floor, with ≥2/3 seeds
(EN) / ≥3/5 seeds (ZH) positive, BOTH the winning rule's overall metric, B/C > D, and no >1pt harm
elsewhere. Anything weaker = **within-noise, no claim**.

### 0.5 Hard rules
Reproduction gate first (floor A == logged per-seed floor, bit-identical). No cross-seed ensembles.
No λ/threshold tuning on test. Gate is CPU; the only possible GPU job is EN dev/test `dens` scoring,
and only post-gate. FORCE=False; no .pt in git.

---

## 1. Gate results (seed-0 winner heads, TRAIN LOO, CPU) — FAIL both datasets

Winner-head ckpts pulled from B2 (`logs/<ds>/Retrieval/...archive.../ckpt`). LOO reproduces the
head quality (floor LOO acc EN 0.8124, ZH 0.8618 — consistent with logged test floors 0.8075/0.8523).

| ds | channel | coverage | AUC(c,gold) | pt-biserial | **corr(c, vote_share)** | R1 net / n_err | R2 net / n_err | gate |
|---|---|---|---|---|---|---|---|---|
| MHC (EN) | bin | 100% | 0.675 | 0.326 | **+0.513** | −14/103 (−0.136) | −13/103 (−0.126) | fail |
| MHC (EN) | dens | 100% | 0.662 | 0.313 | **+0.361** | −28/103 (−0.272) | −24/103 (−0.233) | fail |
| MHC_zh | bin | 100% | 0.688 | 0.349 | **+0.483** | −30/80 (−0.375) | −20/80 (−0.250) | fail |
| MHC_zh | dens | 100% | 0.544 | 0.174 | **+0.209** | −8/80 (−0.100) | −23/80 (−0.287) | fail |

Every rule×channel has **net < 0** (damages more LOO errors than it corrects); bar was net ≥ +15%.
**GATE FAIL on both datasets → killed train-side, NO test contact** (per pre-registration §0.3).

## 2. Test results
None — the gate killed both datasets before any test contact (as pre-registered).

## 3. Verdict — score-level fusion earns NO method role; the premise is refuted

- **The "decorrelated error channels" premise is empirically FALSE.** The MLLM channel correlates
  **positively** with the kNN vote share (corr +0.21…+0.51 across both channels, both datasets), not
  near-zero. Both channels are *weaker* classifiers than the vote (channel AUC 0.54–0.69 vs floor LOO
  acc 0.81–0.86), so the channel mostly agrees with the vote where the vote is already right and adds
  noise exactly where the vote is wrong. Fusing therefore corrects a few errors but damages more:
  net LOO error-correction is negative for **all 8** (channel × rule × dataset) combinations.
- This is the cleanest kind of negative: it does not just fail a threshold, it **refutes the
  mechanism the route was premised on**. Late score-level fusion of the visual-similarity vote and
  the MLLM semantic verdict cannot help because the two are not independent enough and the MLLM
  channel is the weaker of the two.
- Consistent with the rest of the campaign: the MLLM is semantically competent (channel AUC > 0.5,
  positively signed against gold) but its competence is **redundant with**, not orthogonal to, the
  decision variable the retrieval head already optimizes. No test measurement was spent.
- No GPU used (gate CPU-only); EN dev/test density scoring never fired (gate closed first).
