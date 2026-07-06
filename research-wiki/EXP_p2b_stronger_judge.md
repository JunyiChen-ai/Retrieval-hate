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

_(appended after each config; test pass gated on the promotion bar)_

## RESULTS (test pass)

_(only if a config is promoted)_
