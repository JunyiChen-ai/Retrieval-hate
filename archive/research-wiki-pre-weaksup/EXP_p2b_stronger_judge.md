# EXP: P2b — stronger comparability judge with a TRAIN-side calibration stage

> **Status: PRE-REGISTERED (design frozen before any test-set evaluation).**
> P2 (`research-wiki/EXP_p2_neighbor_rerank.md`) killed the 7B neighbour-reranker: the judge
> over-flagged INCOMPARABLE (83% EN / 70% ZH) and its drops were indiscriminate (selectivity
> lift +1.1% EN / −3.2% ZH). But the oracle showed the prize is huge (+7.5 pt EN / +10.6 pt ZH,
> both splits across 0.85). P2b keeps the exact frozen harness and adds the one thing P2 lacked:
> a **TRAIN-side calibration stage** where judge configs are iterated against a labelled
> selectivity benchmark, and only a config that clears a pre-registered bar earns a single test
> pass. This section is committed before any test measurement; the train-side leaderboard and
> the (≤1) test pass are appended below.

## The metric being optimised (train-side, labelled, no test contact)

For a gated query and one retrieved neighbour, the neighbour is a **wrong-vote** neighbour if
its label ≠ the query's gold label (it pushes the kNN vote the wrong way) and a **correct-vote**
neighbour otherwise. A useful comparability judge drops (calls INCOMPARABLE) the wrong-vote
neighbours **more** than the correct-vote ones:

> **selectivity lift = drop_rate(wrong-vote) − drop_rate(correct-vote)**

P2's 7B judge: **+1.1% (EN) / −3.2% (ZH)** — the numbers to beat. Perfect selectivity (drop all
wrong-vote, keep all correct-vote) = the oracle. This lift is the direct proxy for the eventual
revote gain B−A, and it is computable on TRAIN with labels only.

## Train-side benchmark population (frozen)

- One head per language (seed 0 — the config ranking is head-robust because the verdict is a
  pure function of the two videos' evidence, not of the head). Cheap to iterate.
- **Leave-one-out gate on TRAIN**: each train query retrieves over the train memory with itself
  excluded (drop the rank-0 self-match), the LOO similarity-signed vote and `margin=|vote|` are
  computed with the identical vote code, and the **bottom-25% margin** train queries form the
  gated (boundary) population — the train analogue of the gated test set.
- For those gated train queries, the top-20 LOO neighbours are split into correct-vote /
  wrong-vote, and a **balanced subsample of ≈1.5k pairs per language** (equal correct/wrong,
  fixed rng seed 0) is the benchmark. Iterating judge configs over this 1.5k-pair set is cheap;
  the full test pass runs only for the promoted config.

## Candidate grid (iterated on TRAIN only — full logging, no silent shopping)

Three levers, from P2's failure analysis (over-flag ratchet + thin archive fields):

| lever | options |
|---|---|
| (a) judge model | 7B = Qwen2.5-VL-7B-Instruct (P2 baseline, text-only) · 32B = Qwen2.5-32B-Instruct (text-only, bf16, 1×A100-80G, no quant) |
| (b) evidence | `archive` (P2: v2 card only) · `archive+transcript` (card + capped title/transcript snippet for BOTH query and neighbour) |
| (c) prompt | `orig` (P2 wording) · `flip` (default COMPARABLE; INCOMPARABLE is the burden-of-proof verdict, only when the two are positively about different things; frames neighbours as already-nearest so the prior is comparable) |

Configs (executed **cheap-first** — exhaust the cached 7B before downloading 32B):

- **C0** 7B · archive · orig  — reproduces P2's judge on the train benchmark (reference).
- **C1** 7B · archive · flip
- **C2** 7B · archive+transcript · orig
- **C3** 7B · archive+transcript · flip
- **C4** 32B · archive+transcript · flip — the strong-judge shot; run **only if** the best 7B
  config misses the promotion bar.
- **C5** 32B · archive+transcript · orig — model-effect isolator; run only alongside C4 if C4 is
  close, to attribute prompt vs model.

Every config's train-side (drop_rate, lift, per-group n) is logged to
`scripts/analysis/p2_out/p2b_trainbench.json` and tabulated here before any test decision.

## Promotion bar (frozen — decides who, if anyone, touches test)

A config is promoted to the single test pass **iff**, on the train benchmark:

1. **selectivity lift ≥ +10 pt on the PRIMARY dataset (EN/MHC)**, AND
2. its **overall drop-rate ∈ [15%, 50%]** on EN (sane band — not the P2 over-flag, not a no-op),
   AND
3. **selectivity lift > 0 on ZH** (does not regress the control).

If **several** configs clear it, the one with the **highest EN lift** (ties → higher ZH lift →
cheaper model) is promoted. If **none** clears it, **P2b dies train-side with no test contact** —
a valid, cheap kill; the train leaderboard is the deliverable.

## Test pass (at most ONE promoted config)

Identical to P2: same 9 val-selected heads (EN s0–3, ZH s0–4), same 25%-val-margin gate, same
frozen revote rule (drop INCOMPARABLE, keep COMPARABLE/UNSURE, extend if <3 survive, cap 3·K=60,
else fall back to floor), same conditions **A floor / B ours / C random-drop / D oracle**, one
test measurement per condition × seed. The random-drop control **C is recomputed** to match the
promoted config's per-query drop counts. Verdicts are judged with the promoted config on the
test pairs (the union of gated top-60 pairs already emitted by
`p2_rerank_eval.py --mode collect`). Success criteria carry over from P2 (floor repro exact;
mean B−A > 0 EN with ≥3/4 gated-positive; rent test B > C; no consistent ZH harm; sub-noise
positives reported as "within noise floor").

## Hard rules

All GPU via SLURM (no `--time`, `HF_HUB_OFFLINE=1`, `WANDB_MODE=disabled`); model pre-downloaded
on the login node then offline in jobs. No package installs into the shared env (bf16 transformers
only, no vLLM/AWQ). No cross-seed ensembles. No `.pt` in git; checkpoints pulled/deleted; commit
(no push). Poll `sacct`. Report the train-side leaderboard before any test pass.

---

## TRAIN-SIDE LEADERBOARD

Run 2026-07-06. Benchmark = seed-0 LOO-gated train queries, 1500 balanced pairs/language
(750 correct-vote / 750 wrong-vote); EN gated 137/549 train, ZH 145/579. Judge jobs: 7B C0–C3
= 12371–12374; 32B C4–C5 = 12387–12388 (Qwen2.5-32B-Instruct, text-only bf16, 1×A100-80G).
Parse-fallbacks negligible (0 for all 7B; C4 8/3000, C5 0). Numbers by
`scripts/analysis/p2b_score.py` → `p2_out/p2b_trainbench.json`.

**selectivity lift = drop_rate(wrong-vote) − drop_rate(correct-vote)** (higher = the judge
preferentially removes the neighbours that would misvote; the oracle = +100%).

| cfg | model · evidence · prompt | EN drop% (corr/wrong) | EN lift | ZH drop% (corr/wrong) | ZH lift | promote |
|---|---|---|---|---|---|---|
| C0 | 7B · archive · orig (= P2) | 72.5 (72.5/72.5) | **+0.0** | 58.2 (60.9/55.5) | **−5.5** | no |
| C1 | 7B · archive · flip | 58.1 (56.8/59.5) | **+2.7** | 45.3 (46.9/43.7) | **−3.2** | no |
| C2 | 7B · archive+transcript · orig | 73.5 (73.3/73.7) | **+0.4** | 57.6 (60.9/54.3) | **−6.7** | no |
| C3 | 7B · archive+transcript · flip | 60.1 (59.1/61.1) | **+2.0** | 39.9 (43.3/36.5) | **−6.8** | no |
| C4 | **32B** · archive+transcript · flip | 59.2 (58.3/60.1) | **+1.9** | 48.5 (51.6/45.3) | **−6.3** | no |
| C5 | **32B** · archive+transcript · orig | 64.6 (64.0/65.2) | **+1.2** | 50.7 (54.0/47.3) | **−6.7** | no |

## VERDICT — P2b dies TRAIN-side (no test contact)

**No config clears the promotion bar** (EN lift ≥ +10 pt, ZH lift > 0, EN drop-rate ∈ [15,50]%).
Best EN lift across the whole grid = **+2.7 pt** (C1) — a quarter of the way to the bar; **ZH
lift is negative for all six configs** (−3.2 to −6.8 pt: on Chinese the judge, if anything,
*prefers* dropping the correct-vote neighbours). Per the pre-registration this is a valid,
cheap kill: **P2b never touches the test set.**

What the three levers bought, isolated on the labelled benchmark:

1. **Prompt flip (the over-flag fix) works — but only on drop-rate, not selectivity.** Flipping
   INCOMPARABLE to burden-of-proof cut the drop-rate from 72.5%→58.1% (EN) and 58.2%→45.3% (ZH)
   and nudged EN lift from +0.0 to +2.7 pt. So the P2 over-flag ratchet was real and fixable —
   but a judge that drops a *sane* 40–60% of neighbours is **still not selective**: it removes
   correct-vote and wrong-vote neighbours at nearly equal rates.
2. **Transcript evidence adds nothing.** C2/C3 ≈ C0/C1 (EN lift +0.4/+2.0 vs +0.0/+2.7); the
   thin-archive-field hypothesis is not the bottleneck — richer text did not make the match
   track label-relevance.
3. **Model scale (7B→32B) is NOT the lever.** The 32B is no more selective than the 7B — EN
   lift +1.9 (C4) ≤ the 7B flip's +2.7 (C1); ZH still −6.3. A 4.5× bigger judge with the best
   prompt+evidence combo does not move selectivity off zero.

**Mechanism claim (confirmed, and the reason the whole line is closed): comparability ⊥
vote-correctness.** The P2b premise was that a *better* comparability judge would recover the
oracle's +7.5/+10.6 pt. The train benchmark refutes the premise, not just the executor: across
2 models, 2 evidence sets and 2 prompts, whether a neighbour is topically COMPARABLE to the
query is essentially **independent of whether its label matches the query's** (|lift| ≤ 2.7 pt
EN, and wrong-signed on ZH). Dropping incomparable neighbours therefore cannot preferentially
remove misvoters, so no comparability-based reranker — 7B, 32B, or (by this evidence) larger —
can convert the oracle headroom into accuracy. The oracle (drop by *true label*) remains the
only rule that captures it, and it is not implementable without the labels it is scoring
against. **The reranking line is closed; the surviving carrot from P2 (the gate + the oracle's
+7.5/+10.6 pt) needs a fundamentally different membership signal, not a stronger judge.**

*(Leaderboard by `scripts/analysis/p2b_score.py` from the six `tb_verdicts_*` files; verdict
prose human-written against `p2b_trainbench.json`. 32B model + seed-0 heads deleted after the
run; no `.pt` in git.)*

---

## P2c — 72B tier (pre-registered extension; completes the 7B→32B→72B scale ladder)

> **Pre-registered before any 72B judgment.** P2b left the ladder incomplete (7B, 32B). P2c adds
> the 72B rung on the **identical frozen** train benchmark (`p2b_train_benchmark.py` /
> `p2b_score.py`, same 1500 balanced correct/wrong-vote pairs per language) with the **unchanged
> promotion bar** (EN selectivity lift ≥ +10 pt AND EN drop-rate ∈ [15,50]% AND ZH lift > 0). No
> other change. If a config clears, the single best is promoted to the one A–D test pass (the 0.85
> shot); if neither clears, the ladder is complete and flat and the kill is definitive.

- **Configs:** **C6 = 72B · archive+transcript · flip**, **C7 = 72B · archive+transcript · orig**
  (the model-effect isolator). Same evidence/prompt as the 32B rung so C6−C4 / C7−C5 isolate scale.
- **Load path (documented deviation):** the intended `Qwen2.5-72B-Instruct-AWQ` (41 G) and the
  GPTQ-Int4 fallback **both require backends absent from the installed stack** (no `autoawq`, no
  `auto_gptq`/`gptqmodel`), and installing them into the shared `HateVideo` env would risk
  downgrading `transformers` 4.49 and breaking the sibling Qwen2.5-VL jobs mid-campaign.
  `bitsandbytes` 0.49.2 + `accelerate` 1.5.2 ARE installed, so P2c loads the **bf16
  `Qwen/Qwen2.5-72B-Instruct` under on-the-fly 4-bit nf4 (double-quant, bf16 compute)** —
  ~40 G on 1×A100-80G, zero env mutation, zero sibling risk. Cost: the bf16 checkpoint (~145 G)
  is downloaded instead of the 41 G AWQ; deleted after the run. Text-only, greedy, same judge
  script (`--quant bnb4`).

### P2c leaderboard

Run 2026-07-07. 72B = `Qwen/Qwen2.5-72B-Instruct` bf16 under bitsandbytes 4-bit nf4, text-only,
greedy, on the identical 1500-pair benchmark. Jobs: C7 = 12425, C6 = 12429 (resumed from a
1102-verdict cache after a scheduling swap to free a GPU for P8; resume-by-key, no data loss).
0 parse-fallbacks. Full 7B→32B→72B ladder (drop-rate = correct/wrong):

| cfg | model · evidence · prompt | EN drop% (corr/wrong) | EN lift | ZH drop% (corr/wrong) | ZH lift | promote |
|---|---|---|---|---|---|---|
| C0 | 7B · archive · orig | 72.5 (72.5/72.5) | +0.0 | 58.2 (60.9/55.5) | −5.5 | no |
| C1 | 7B · archive · flip | 58.1 (56.8/59.5) | +2.7 | 45.3 (46.9/43.7) | −3.2 | no |
| C2 | 7B · archive+transcript · orig | 73.5 (73.3/73.7) | +0.4 | 57.6 (60.9/54.3) | −6.7 | no |
| C3 | 7B · archive+transcript · flip | 60.1 (59.1/61.1) | +2.0 | 39.9 (43.3/36.5) | −6.8 | no |
| C4 | 32B · archive+transcript · flip | 59.2 (58.3/60.1) | +1.9 | 48.5 (51.6/45.3) | −6.3 | no |
| C5 | 32B · archive+transcript · orig | 64.6 (64.0/65.2) | +1.2 | 50.7 (54.0/47.3) | −6.7 | no |
| **C6** | **72B · archive+transcript · flip** | 35.7 (34.8/36.5) | **+1.7** | 25.5 (27.2/23.7) | **−3.5** | **no** |
| **C7** | **72B · archive+transcript · orig** | 30.9 (30.7/31.2) | **+0.5** | 14.9 (16.3/13.5) | **−2.8** | **no** |

### P2c verdict — definitive scale-ladder kill (7B → 32B → 72B flat on selectivity)

**No config clears the bar at any rung; P2c never touches the test set.** The ladder is now
complete and the two axes cleanly separate:

- **Calibration IMPROVES monotonically with scale.** Orig-prompt drop-rate collapses 7B 72.5% →
  32B 64.6% → **72B 30.9%** (EN) and 58.2% → 50.7% → **14.9%** (ZH): the over-flag ratchet that
  the flip prompt had to hand-fix at 7B, the 72B fixes *on its own* — it drops far fewer
  neighbours and uses UNSURE judiciously (72B-orig: 254 EN / 362 ZH UNSURE). Scale buys a
  well-behaved, non-trigger-happy judge.
- **Selectivity does NOT move with scale — it stays pinned at ~0.** EN lift never exceeds **+2.7
  pt** anywhere on the ladder (72B: +1.7 / +0.5, ≤ the 7B flip's +2.7); ZH lift is **negative at
  every one of the 8 configs** (−2.8 to −6.8). A well-calibrated 72B that drops a sane 31% of
  neighbours *still* removes correct-vote and wrong-vote neighbours at the same rate.

This is the decisive completion of the P2b mechanism claim: **comparability ⊥ vote-correctness at
every open-source scale.** Whether a neighbour is topically comparable to the query is independent
of whether its label matches — and making the judge bigger (up to 72B) makes it *better behaved*
without making its comparability calls track label-relevance at all. No comparability-based
reranker — 7B, 32B, or 72B — can preferentially drop the misvoters, so none can convert the P2
oracle headroom (+7.5 EN / +10.6 ZH) into accuracy. **The neighbour-reranking line is closed at
open-source scale; the 0.85-crossing oracle prize is real but needs a fundamentally different
membership signal, not a stronger judge.**

*(72B leaderboard by `p2b_score.py` from the eight `tb_verdicts_*` files. Load path = bnb-4bit
on the bf16 checkpoint, documented above; 136 G 72B cache deleted after the run; no `.pt` in
git.)*
