# EXP: P2 — margin-gated MLLM reranking of retrieved neighbors

> **Status: PRE-REGISTERED (design frozen before any test-set evaluation).**
> This top section (motivation, gate, judging task, revote rule, conditions, success
> criteria) was written and committed BEFORE running any condition on any test set. Results
> are appended in a separate `## RESULTS` section and were not used to tune any threshold,
> prompt, or rule. Numbers are produced by `scripts/analysis/p2_rerank_eval.py`; MLLM
> comparability verdicts by `scripts/analysis/judge_neighbor_compat.py`
> (SLURM `scripts/slurm/judge_neighbor_compat_{MHC,MHC_zh}.sbatch`).

## Motivation — why this is not the dead Role-3 arbiter

Role-3 (`research-wiki/EVAL_role3_selective_reasoning.md`) let a 7B MLLM **decide the label**
of gate-deferred hard samples. It died: deferred-accuracy 0.462–0.615, all below the 0.667
break-even, with a one-way over-flag ratchet (a generic safety prior pushes the whole
boundary queue to "harmful"). Two assets survived that post-mortem and are reused here
verbatim:

- **The margin gate works.** `|similarity-signed arithmetic vote|` concentrates errors: on
  EN test at the 30% working point, 24% of samples captured 42% of the kNN errors, on the
  hate-vs-offensive boundary. Oracle accuracy at 20–30% deferral = 0.857–0.888, i.e. there is
  real headroom on the gated slice.
- **The gate is cheap** (CPU faiss, ms) and honest (thresholds chosen on val only).

**P2 changes the MLLM's job so it can never ratchet.** The MLLM **never outputs a label.**
For each gated query it judges, per retrieved neighbor, a **pairwise comparability** question —
"is this precedent a fair basis for voting on this query?" — an easier, evidence-grounded call.
Incomparable neighbors are removed before the kNN vote is recomputed. An MLLM error therefore
only **dilutes** a vote; it can never **decide** a sample. The vote still consumes the
neighbors' ground-truth labels exactly as before; the MLLM only edits *membership*.

## Base system (fixed, zero training)

- 9 val-selected archive-kNN α=0.25 heads: **EN (MHC) seeds 0–3**, **ZH (MHC_zh) seeds 0–4** —
  the identical checkpoints used in `EXP_auto_memory_repair.md` / `exp-archive-knn-seeds.md`
  (EN jobs 12210/12219/12220/12221, ZH jobs 12207/12215/12216/12217/12218; epochs pinned in
  `scripts/analysis/auto_memory_repair.py::CKPT_FILE`). Pulled from B2
  (`logs/Retrieval/...`) by `scripts/analysis/p2_pull_ckpts.sh`, deleted after the CPU phase.
- Memory bank = TRAIN split only. Key = `[l2n(fused) | 0.25·l2n(archive_CLIP_text)]` with the
  **v1** archive feats (`--archive_feats auto`). Decision = faiss cosine, **topk=20**,
  arithmetic rank weights (20,19,…,1 by retrieval rank), **similarity-signed** vote
  (`use_sim=True`); `pred = sigmoid(vote) ≥ 0.5 ⇔ vote ≥ 0`; `margin = |vote|`. This vote is
  imported/replicated bit-for-bit from `src/utils/metrics.compute_metrics_retrieval` and must
  reproduce the logged per-seed floor exactly (reproduction gate below).
- **v1 keys are mandatory** (reproduction gate pins them). The **MLLM reads the v2 archives**
  (cleaner structured text; `data/Archive/{ds}/v2/`), the same split of duties as the
  auto-memory-repair robustness pass: v1 defines *who* the neighbors are and *how they vote*;
  v2 text is *what the MLLM reads* to judge comparability. Verdicts are label-blind and depend
  only on (query archive, neighbor archive), so each unique pair is judged **once** and reused
  across all seeds (seed-independent, like the auto-repair semantic vote).

## The gate (fixed a priori — no tuning)

Deferral fraction **fixed at 25%** a priori (mid of the 20–30% oracle window). Per seed, the
margin threshold `t` is the **25th percentile of that seed's VAL margins** (midpoint between
the k-th and (k+1)-th smallest, k=round(0.25·N_val); identical rule to
`scripts/role3/gate_margin.py`). A test sample is **gated ⇔ its margin < t**. The threshold is
computed on val only; test never participates. The realised test deferral fraction is reported
(≈25%, not forced to exactly 25%, because we do not peek at test margins to set the gate).

## The MLLM judging task (per gated query × retrieved neighbor)

Qwen2.5-VL-7B-Instruct, greedy (`do_sample=False`), **text-only**. Input = the query's v2
archive card + **one** neighbor's v2 archive card (target_groups / mechanism / modality_cues /
explicitness / neutral_summary). **No labels shown** (comparability, not classification).
Output strict JSON `{"verdict": "COMPARABLE"|"INCOMPARABLE"|"UNSURE", "reason": "..."}`.
Fixed criteria (frozen here):

> A precedent is **COMPARABLE** if it shares with the query **at least one** of: (a) the same
> or an overlapping **target group**; (b) the same **attack / harm mechanism**; (c) the same
> **evidence modality** carrying the salient content (visual / speech / on-screen text).
> **INCOMPARABLE** only when **clearly none** of (a)/(b)/(c) match. **UNSURE** if the archives
> are too thin/ambiguous to tell.

Parse-failure fallback: strict JSON first, then a bare-word fallback (`COMPARABLE`/
`INCOMPARABLE`/`UNSURE`), then default **UNSURE** (keeps the neighbor — the conservative,
vote-preserving default). Fallbacks are counted.

Judging depth **N_JUDGE = 60 = 3·K** neighbors per gated query (enough to cover the extension
cap below). Unique (query, neighbor) pairs are deduplicated across seeds before judging.

## The revote rule (ONE rule, frozen — no alternatives tried on test)

For each gated query, in retrieval order:

1. Take the top-K=20 voting neighbors. **Drop** those judged **INCOMPARABLE**; keep
   **COMPARABLE** and **UNSURE** at full weight.
2. If **≥3** neighbors survive → revote over the survivors.
3. If **<3** survive → **extend** retrieval depth (ranks 21…59), appending any non-INCOMPARABLE
   neighbor to the survivor tail until **3** survive; **cap at 3·K=60**. If still <3 at rank 60
   → **fall back to the original top-20 vote** (unchanged).
4. **Revote** = the identical similarity-signed rank-weighted arithmetic vote applied to the
   surviving neighbor list *in retrieval order* (rank weights 20,19,… reassigned by surviving
   position, exactly as `compute_metrics_retrieval` assigns them). With nothing dropped this
   equals the floor vote bit-for-bit, so ungated and un-edited samples are untouched.

Non-gated samples keep their floor prediction in every condition.

## Conditions (one test measurement per condition × seed — no repeats, no selection)

| id | condition | drop set on each gated query |
|----|-----------|------------------------------|
| **A** | floor (no reranking) | ∅ — must reproduce the logged per-seed floor exactly |
| **B** | **MLLM reranking (ours)** | top-20 neighbors judged INCOMPARABLE (+ extension rule) |
| **C** | random-drop control | the **same number** B dropped on that query, uniformly at random, `random.Random(0)` advanced deterministically per query; same extension/fallback machinery |
| **D** | oracle reranking (upper bound) | neighbors whose **label disagrees with the query's gold label** (keeps only same-gold-label neighbors; extension finds same-label deeper). Uses test labels for the BOUND only, never in the method. |

C isolates *which* neighbors are dropped (MLLM judgment vs random of equal count) — the **rent
test**. D is the theoretical ceiling of any drop rule (perfect membership editing); B's fraction
of D = (B−A)/(D−A) says how much of the achievable headroom the MLLM captures.

## Metrics & reporting

Per seed: accuracy + macro-F1, **overall** and on the **gated subset**, for A/B/C/D. Mean±std
across seeds. Paired per-seed deltas **B−A**, **B−C** (overall + gated subset), and **B's
fraction of D**. Also: per-query drop counts, extension/fallback counts, verdict distribution
(COMPARABLE/INCOMPARABLE/UNSURE), and a 10-query kept/dropped-neighbor audit. EN primary, ZH
control.

## Pre-registered success criteria

1. **Reproduction gate (must pass first).** Condition A equals the logged per-seed floor
   (EN s0 0.8075/0.7626; ZH s0 0.8523/0.8270; all 9 seeds match
   `exp-archive-knn-seeds.md`). If this fails, stop and debug.
2. **B helps on EN.** mean paired **B−A > 0** on EN, with **≥3/4 seeds positive on the gated
   subset**.
3. **The MLLM earns its place (rent test): B > C.** mean paired **B−C > 0** — MLLM-chosen drops
   beat random drops of equal count. If C ≥ B, the MLLM adds nothing over blind dilution and we
   say so plainly.
4. **No harm on the ZH control.** mean B−A on ZH not consistently negative (no consistent
   negative sign across the 5 seeds).
5. **Honesty guard.** On these ~150-sample test sets 1 acc pt ≈ 1.6 videos; seed/selection
   noise dominates. Sub-1.6-video (<1 pt) positive results are reported as **"within the noise
   floor"** with **no accuracy claim** — the headline is the sign pattern of the paired deltas,
   not a p-value. If B fails, a 10-query qualitative audit of MLLM comparability verdicts
   (right/wrong and why) documents the failure mechanism.

## Hard rules honoured

- All GPU work via SLURM (`sbatch`, no `--time`, `HF_HUB_OFFLINE=1`, `WANDB_MODE=disabled`,
  `--Faiss_GPU False`), one judging job per language. The gate/retrieval/revote are CPU-only.
- No cross-seed ensembling — per-seed only, aggregated as mean±std / paired deltas.
- No checkpoint/cache overwrites; no `.pt`/`.safetensors` committed to git; pulled checkpoints
  deleted after the CPU collect phase; disk kept under quota.

---

## RESULTS

Run 2026-07-06. Gate + top-60 retrieval + floor-repro + pair dedup = SLURM job 12354
(`p2_collect`, CPU). MLLM comparability verdicts = jobs 12358 (EN, 4811 pairs, 60 min) +
12359 (ZH, 5419 pairs, 69 min); text-only Qwen2.5-VL-7B-Instruct, greedy; **0 parse
fallbacks** on 10230 pairs (100% strict-JSON). A–D revote = `p2_rerank_eval.py --mode revote`
(CPU). Machine tables: `scripts/analysis/p2_out/p2_results_block.md` /
`p2_results.json`; audit: `scripts/analysis/p2_audit.py`.

**Reproduction gate — PASS (bit-identical, all 9 heads):** EN s0 0.8075/0.7626 … s3
0.8075/0.7713; ZH s0 0.8523/0.8270 … s4 0.7852/0.7266 — every floor equals the logged
val-selected value.

### The MLLM over-flags INCOMPARABLE (the failure mechanism)

Verdict distribution over the **used** top-20 gated neighbours:

| dataset | COMPARABLE | INCOMPARABLE | UNSURE | mean neighbours dropped / gated query |
|---|---|---|---|---|
| MHC (EN) | 403 (16.0%) | **2103 (83.4%)** | 14 (0.6%) | **16.3 / 20** |
| MHC_zh (ZH) | 1058 (29.9%) | **2469 (69.8%)** | 13 (0.4%) | **13.9 / 20** |

The judge calls ~70–83% of retrieved neighbours INCOMPARABLE, so the revote almost always
collapses onto <3 survivors and triggers the extension/fallback path (EN 4–10 fallbacks per
seed, ZH 2–7). This is the **same over-flagging ratchet role-3 hit** — there a generic safety
prior pushed the queue to "harmful"; here a strict "must share target OR mechanism OR modality"
bar, read off **sparse archives** (many benign videos have empty `target_groups`/`mechanism`,
leaving only a noisy summary), pushes the queue to "incomparable".

### Discriminativeness diagnostic — the drop is indiscriminate (≈ random)

A useful comparability judge would drop **wrong-vote** neighbours (label ≠ query gold) more
often than **correct-vote** neighbours (label = gold). It does not:

| dataset | correct-vote dropped | wrong-vote dropped | selectivity lift |
|---|---|---|---|
| MHC (EN) | 82.9% (n=1343) | 84.0% (n=1177) | **+1.1%** (≈0, indiscriminate) |
| MHC_zh (ZH) | 71.3% (n=1829) | 68.1% (n=1711) | **−3.2%** (anti-selective — drops *helpful* neighbours more) |

So comparability is nearly orthogonal to vote-correctness: it removes good and bad precedents
at the same rate on EN, and slightly *prefers* removing the good ones on ZH — which is exactly
why B ≈ noisy-random on EN and net-harmful on ZH.

### A–D — overall (mean ± std across seeds, accuracy / macro-F1)

| dataset | A floor | B MLLM (ours) | C random-drop | D oracle |
|---|---|---|---|---|
| MHC (EN, 4 seeds) | 0.7935 / 0.7497 | 0.7919 / 0.7451 | 0.7857 / 0.7361 | **0.8680 / 0.8355** |
| MHC_zh (ZH, 5 seeds) | 0.8268 / 0.7915 | 0.8067 / 0.7605 | 0.8242 / 0.7883 | **0.9329 / 0.9188** |

### A–D — gated subset (mean ± std accuracy)

| dataset | A floor | B MLLM | C random | D oracle |
|---|---|---|---|---|
| MHC (EN) | 0.6150 ± 0.050 | 0.6125 ± 0.057 | 0.5775 ± 0.054 | **1.0000** |
| MHC_zh (ZH) | 0.5463 ± 0.151 | 0.4659 ± 0.097 | 0.5421 ± 0.083 | **1.0000** |

### Paired per-seed deltas (accuracy)

| delta | EN overall | EN gated | ZH overall | ZH gated |
|---|---|---|---|---|
| **B − A** (ours vs floor) | −0.0016 ± 0.010 (+2/4) | −0.0025 ± 0.050 (+2/4) | **−0.0201 ± 0.023 (+1/5)** | −0.0803 ± 0.089 (+1/5) |
| **B − C** (rent test) | +0.0062 ± 0.019 (+3/4) | +0.0350 ± 0.106 (+3/4) | **−0.0174 ± 0.019 (+1/5)** | −0.0762 ± 0.077 (+1/5) |
| **D − A** (oracle headroom) | +0.0745 ± 0.015 (+4/4) | +0.3850 (+4/4) | +0.1060 ± 0.030 (+5/5) | +0.4537 (+5/5) |

**B's fraction of the oracle headroom (overall acc): EN −0.02, ZH −0.19** — the 7B judge
captures none of the (large, real) headroom and on ZH is worse than the floor.

### 10-query neighbour audit (EN seed 0; full dump in p2_audit.py --examples)

The audit shows the drop is driven by **noisy archive fields and inconsistent reasoning**, not
by real comparability:

- `8zLoOqXvk64` (gold 0): **all 20 neighbours dropped** (kept 0/20 → fallback). Its own card
  reads "a group around a motorcycle" yet lists `mechanism=slur`; the judge then finds nothing
  to match against and drops everything.
- `01ygFLVdj8s` (gold 1, trans-mockery): drops 18/20 including four other correct-vote
  hateful precedents, each with the boilerplate reason "different target groups and mechanisms"
  — even neighbours that are also anti-trans mockery, because the neighbour cards' `mechanism`
  fields differ in wording.
- `-cgm2EZcLC8` (gold 1): keeps `t2KWh5O9798` (nlab 0, WRONG-vote) for "same mechanism (slur)"
  while dropping `O_i2jeFqsCQ` (nlab 1, correct-vote) for "different mechanisms" — the kept /
  dropped split is uncorrelated with which precedent would vote correctly.
- Recurrent pattern: identical, generic justifications ("different target groups and
  mechanisms") applied to both correct- and wrong-vote neighbours; matches hinge on exact
  string overlap of thin `mechanism`/`target_groups` fields rather than semantic aboutness.

### Verdict vs the pre-registered success criteria

1. **Reproduction gate — PASS** (bit-identical, all 9 heads).
2. **B helps on EN — FAIL.** mean B−A = −0.0016 (overall) / −0.0025 (gated), and only **2/4**
   seeds positive on the gated subset (criterion asked ≥3/4). Within the noise floor and
   directionless.
3. **Rent test B > C — FAIL (not consistent).** EN B−C = +0.0062 (3/4, but < the 1.6-video ≈
   1-pt noise floor → **within noise, no claim**); ZH B−C = **−0.0174 (1/5), negative**. The
   MLLM's drops do **not** reliably beat random drops of equal count — the discriminativeness
   lift (+1.1% EN, −3.2% ZH) says why.
4. **No harm on ZH — FAIL.** B−A = −0.0201 acc / −0.0310 macro-F1, negative on **4/5** ZH
   seeds (0.8268 → 0.8067). Dropping ~14/20 neighbours destabilises an otherwise-strong floor.

### Plain-language bottom line

**Margin-gated MLLM neighbour reranking does not work with a 7B judge, and the reason is the
same over-flagging pathology that killed role-3 — just relocated from "harmful" to
"incomparable".** The gate itself is sound and the headroom is large and real: the oracle
membership editor (condition D) drives the gated subset to 100% and lifts overall accuracy
+7.5 pt (EN) / +10.6 pt (ZH), comfortably across 0.85 — a perfect comparability judge would be
a genuine win. But the 7B judge calls 70–83% of neighbours INCOMPARABLE off sparse, noisy
archive fields, dropping ~14–16 of every 20 voting neighbours, and its drops are **statistically
indiscriminate** (it removes vote-correct and vote-wrong precedents at the same rate on EN, and
*prefers* removing the correct ones on ZH). The net effect is therefore a noisy random thinning
of the vote: **within the noise floor on EN (B−A ≈ 0, B−C ≈ +0.6 pt, no claim), and net-harmful
on ZH (B−A ≈ −2 pt, B−C ≈ −1.7 pt).** The rent test fails: the MLLM adds nothing over random
dropping, and captures ~0 (EN) to negative (ZH) of the oracle headroom.

Because the verdicts are seed/head-independent (a pure function of the two archives), the
failure mechanism is **not** a checkpoint-selection artefact and would reproduce on the
final-epoch heads unchanged; running that second protocol would re-confirm the same negative at
2× the judging cost, so it is not run. The one constructive read-through, consistent with
role-3's oracle finding: the ceiling (D) is high enough to justify a **stronger** comparability
judge (bigger model, or archives dense enough that a target/mechanism/modality match is a real
signal rather than a string-overlap lottery) — but on the current 7B + current archives, this
line is closed.

*(Numbers from `scripts/analysis/p2_rerank_eval.py` / `p2_audit.py`; verdict prose
human-written against `p2_results.json` and the audit dump. Checkpoints pulled from B2 and
deleted after the collect phase.)*
