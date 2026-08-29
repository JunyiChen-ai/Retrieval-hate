---
type: code-review
node_id: review:tarc-g1-code
title: "TARC B-line G1 probe — fresh pre-submit code review"
reviewer: "fresh / zero-context (read-only)"
date: "2026-07-13"
scope: "src/run_rac.py, src/model/loss.py, src/utils/retrieval.py, src/model/evaluate_rac.py, src/utils/metrics.py, scripts/analysis/build_tarc_target_map.py, scripts/slurm/tarc_g1.sbatch, data/gt/HateMM/target_map.json (all vs HEAD; changes uncommitted)"
verdict: PASS_FOR_SUBMIT
critical: 0
high: 0
medium: 0
low: 3
---

# TARC G1 code review — verdict: **PASS_FOR_SUBMIT**

Zero-context read-only review of the TARC (target-aware retrieval-contrastive) B-line G1
oracle-ceiling probe against the pre-registration `exp-tarc-t0.md` (§1-9). All changes are
uncommitted (`git diff HEAD`). No SLURM submitted, no training run, no code modified —
only `py_compile`, a CPU parse of the job-12850 trainlogs, and a `target_map.json`↔CSV
cross-check were executed.

**Critical = 0, High = 0 ⇒ PASS_FOR_SUBMIT.** Three Low advisories (non-blocking) at the end.

---

## Item 1 — V1 row alignment (head risk): **CORRECT, no misalignment**

The alignment holds by a *lockstep-rebuild* argument, and the draft's flagged shuffle
hazard is neutralised:

- **Same-pass collection.** In the index rebuild block (`src/utils/retrieval.py:341-384`),
  `train_ids_accum.extend(batch["ids"])` (`:347-349`) runs in the *same single pass* over
  `train_dl` that accumulates `train_feats`/`train_labels` (`:350-379`). After the loop,
  `target_pack["_train_targets"]` is written from `train_ids_accum` (`:380-384`). So
  `_train_targets[r] ↔ train_feats[r]` for whatever (shuffled) FAISS row order that rebuild
  produced. Order-agnostic by construction.
- **No stale cache across epochs.** `model_pass` sets `train_feats = None; train_labels =
  None` at the **start of every epoch** (`src/run_rac.py:581-582`). The rebuild block runs
  **iff** `train_feats is None` (`:341`), and it rewrites *both* `train_feats` and
  `_train_targets` inside that one block. There is no path that rebuilds `train_feats`
  without rewriting `_train_targets`, and none that rewrites `_train_targets` without
  rebuilding `train_feats`. When `train_feats` is cached (mid-epoch, not `None`), the
  rebuild is skipped and the cached `_train_targets` still matches the cached rows. Hence
  epoch-N's target vector can never be applied to epoch-(N+1)'s new shuffle order.
  `--reindex_every_step` (`:586-589`) only makes the lockstep rebuild happen every step —
  still consistent (and it is off for G1).
- **`tarc_active` is constant within a run** (depends only on `target_pack`, `query_ids`,
  and two args, all fixed), so the ON/OFF branch never flips mid-run.
- **Query side.** `query_targets_np[i]` is keyed off the current batch's `query_ids`
  (`:457-459`), which are the same `ids` whose `query_feats` were retrieved; `I[i,iter]`
  indexes a train row `r`, and `train_labels[r]`/`train_targets_np[r]` index the *same* `r`
  (`:562-577`). Fully aligned.

**Verdict: the row-alignment implementation is correct.** The `_train_targets is None`
cache-miss fallback (`:460-464`) reverts that step to baseline mining rather than
mis-aligning — it is a dead path given the epoch-start rebuild, but it fails safe (see L2).

---

## Item 2 — no-op guarantee at default: **GENUINE no-op**

- With `--tarc_target_source off`, the `main()` builder block is skipped (`src/run_rac.py:1218`
  `if tarc_source != "off"`), so `target_pack` stays `None`.
- `dense_retrieve_...`: `tarc_active` is `False` when `target_pack is None`
  (`src/utils/retrieval.py:324-328`); `train_ids_accum = None` (`:346`), all `if tarc_active`
  blocks skip, and the **original mining loop runs verbatim** under `if not tarc_active:`
  (`:466-550`). I diffed the re-added baseline loop against the pre-change loop: it is a
  byte-for-byte copy (only re-indented), including the `if j == largest_retrieval and k ==
  args.no_pseudo_gold_positives: break` condition.
- V3 (`src/model/loss.py:582-588`): `lambda_tarc == 0` ⇒ block skipped, `compute_target_loss`
  never called.
- V2 (`src/utils/metrics.py:274`): the multiplier is guarded by `tarc_vote_gamma > 0 and
  "query_target" in value`; with `gamma=0` no numpy op is added. `retrieve_evaluate_RAC_`
  sets `tarc_v2=False` (`src/model/evaluate_rac.py:327-328`) ⇒ no extra logging-dict keys.
- **No new torch tensor or torch-RNG draw on the OFF path** — the only id→code work is
  numpy/python and only executes when `tarc_active`. Seed reproduction is therefore preserved.
- The OFF python command is training-identical to the enc3seed CLIP command (Item 6);
  differences are name-only (`--group_name`, `--exp_comment`) plus the inert
  `--tarc_target_source off`.

**Verdict: default path is a true no-op.** The empirical 4-decimal reproduction gate (§7.2)
must still be confirmed at runtime, but the code path is provably inert.

---

## Item 3 — V3 loss math & gradient: **CORRECT**

`compute_target_loss` (`src/model/loss.py:626-670`):
- Per target `t` present in the batch with **both** a hate and a benign example, computes
  `mu_hate`/`mu_benign` as means of the **grad-tracked** `feats` (`:664-665`), then
  `L_t = relu(margin + sim(mu_hate, mu_benign))` with `sim = _pair_similarity` (`:666-667`).
- `_pair_similarity` (`src/model/loss.py:716-725`) returns a **similarity** (cos: higher =
  closer). Minimising `relu(margin+sim)` drives `sim` down toward `-margin`, i.e. **pushes
  the two same-community centroids apart** — exactly the §1 intent (see Item 7.1 for the
  sign adjudication). `margin = triplet_margin = 0.1` (`:656`), no new hyperparameter.
- Degenerate cases handled: targets with code `< 0` excluded (`:658` `codes >= 0`); a target
  missing either class in the batch is skipped (`:662-663`), covering the tail classes
  (Asian n=1, Sexits n=5); empty batch of eligible targets returns `zeros(())` (`:668-669`).
- Gradient flows through `feats` → model; the loss is **symmetric** in hate/benign, so even
  a flipped label convention would not change correctness.
- `lambda_tarc = 0` ⇒ never invoked (Item 2).

---

## Item 4 — V2 vote reweight: **CORRECT, identity when off**

- `src/utils/metrics.py:271-278`: the `(1+gamma)` multiplier lives **only** in the
  `elif use_sim:` branch (`:262`), applied to `retrieved_labels_map` after the existing
  sim-weighting (`:270`). `tmult = where((rtg==qt) & (qt>=0), 1+gamma, 1.0)` — a neighbour
  is up-weighted only when its target matches the query's *and* the query target is present
  (`qt>=0`). `gamma=0` or a missing `query_target` key ⇒ identity ⇒ byte-identical vote.
  Non-renormalised denominator matches the pre-existing sim convention (`:284`); only the
  vote **sign** determines acc, so this is the designed reweighting, not a defect.
- `src/model/evaluate_rac.py:475-480`: `query_target`/`retrieved_target` are attached only
  when `tarc_v2` (`target_pack` set **and** `gamma>0`), so OFF/V1/V3 arms carry no extra keys.
  `retrieved_target` is built from the same `retrieved_ids` as `retrieved_label`/`_scores`,
  so the three per-neighbour arrays are equal-length and aligned for the elementwise multiply.
- Cross-arm isolation is clean: V2 is inert in V1/V3 arms (`gamma=0`), and V1/V3 are inert in
  V2 arms (`tarc_hn_mode off`, `lambda_tarc 0`).

---

## Item 5 — gold-annotation isolation (§5 ledger): **ENFORCED**

- **Single GT reader.** `grep` over `src/` + `scripts/` shows `target_map.json` is read at
  exactly one training-pipeline site: `src/run_rac.py:1232` (the `gt_oracle` branch), plus
  the builder that writes it. `hate_spans.json` / `hate_snippet` are read only by unrelated
  localization scripts (`p10_eval_hatemm.py`, `p11_probe_hatemm.py`, `eval_localization_ours.py`,
  `hatemm_spans.py`) — never by the TARC training path. `mllm_pred` file only at `:1237`.
- **Assert precedes all side effects.** The `gt_oracle ⇒ oracle_probe` assertion
  (`src/run_rac.py:884-887`) runs before the first filesystem side effect (`os.makedirs`,
  `:901-903`), before the GT read (`:1232`), and before training. A `gt_oracle` run without
  `--oracle_probe True` aborts with `AssertionError` before any directory/file/GT touch.
- **Provenance stamping works.** `_ORACLE_CEILING` is appended to `exp_name`
  (`:896-897`, gated on `gt_oracle` ∧ `oracle_probe`) and echoed in the trainlog header
  (`:913-921`). `mllm_pred` raises `FileNotFoundError` this round (`:1238-1241`), so no GT
  can leak into a main-table path.

---

## Item 6 — sbatch 21 arms + OFF expectations: **CORRECT, expectations verified**

- **Arms.** `ARMS=(off v1prefer v1require v3lt0.1 v3lt0.5 v2vg0.5 v2vg1.0)` × seeds `0/1/2`
  = **21**. `arm_flags` (`scripts/slurm/tarc_g1.sbatch:41-52`) matches the §9.3 table
  arm-for-arm; every non-OFF arm carries `--tarc_target_source gt_oracle --oracle_probe True`.
- **Command parity.** The python invocation (`:64-78`) equals the enc3seed CLIP command
  (`scripts/slurm/enc3seed.sbatch:49-62`) — `lr 1e-4, epochs 30, topk 20, proj/map_dim 1024,
  dropout 0.2 0.4 0.1, align, hn 1, pp 1, cos, triplet, hybrid, warmup 5, arithmetic,
  lambda_seg 0, Faiss_GPU False` — differing only by name-only flags and `$FLAGS`.
- **Resources.** `--gres=gpu:a100:1`, `--cpus-per-task=8`, `--mem=64G`, **no `--time`**,
  `conda activate HateVideo`, no MLLM. ✓
- **OFF expectation provenance (verified live from job-12850 trainlogs).** I re-parsed
  `slurm/logs/enc3s_HateMM_openai_clip-vit-large-patch14-336_HF_seed{0,1,2}_12850.trainlog`
  with the sbatch's own selection logic (warmup≥5, val-acc→roc tie-break):

  | seed | best ep | val-sel TEST acc / F1 | final ep | final TEST acc / F1 |
  |---|---|---|---|---|
  | 0 | 24 | **0.8279 / 0.8172** | 29 | **0.8186 / 0.7997** |
  | 1 | 26 | **0.8279 / 0.8163** | 29 | **0.8047 / 0.7822** |
  | 2 | 24 | **0.8047 / 0.7920** | 29 | **0.8140 / 0.7988** |

  All six pairs match the sbatch header (`tarc_g1.sbatch:19-21`) **exactly** — the OFF
  expectation is faithfully sourced (numeric-provenance clean).
- **G1 decidability.** `RESULT_ROW` (`:102`) emits per-arm/seed `VALacc VALf1 TESTacc TESTf1`,
  which is sufficient for the pre-registered G1 rule (paired **val** Δacc ≥ +0.015, sign
  ≥2/3). The OFF val-selected **VAL** acc (the delta baseline) is 0.8411/0.8505/0.8692 in the
  same logs. Adequate. (Header labels these val-selected numbers "VALSEL acc"; they are
  val-*selected* TEST acc — cosmetic, see L3.)

---

## Item 7 — §9.5 deviations: **all five are faithful repairs, none a substantive departure**

1. **V3 loss sign (load-bearing) — FAITHFUL.** `_pair_similarity` is a similarity
   (`loss.py:716-725`). The literal §2 formula `relu(m − sim)` would *pull* centroids
   together (wrong sign). The implemented `relu(margin + sim)` (`loss.py:667`) drives the
   same-community hate/benign centroids to `sim ≤ −margin`, i.e. apart — realising the §1
   hypothesis. Margin reuses `--triplet_margin`; no new knob. Correct fix.
2. **`target_map.json` over all 1083 CSV rows — FAITHFUL.** `id_to_target` covers every
   query (train+val+test); my CSV↔map cross-check found **0 mismatches / 1083 entries**; the
   17 non-split stems are never queried.
3. **`--tarc_multitarget any` stub — FAITHFUL.** Raises `NotImplementedError`
   (`run_rac.py:1219-1222`); `primary` is fully wired; G1 uses `primary`.
4. **Missing-target (code −1) — FAITHFUL.** Never target-matches: `qt>=0` guards in V1
   (`retrieval.py:565`) and V2 (`metrics.py:277`), `codes>=0` in V3 (`loss.py:658`).
5. **EM path not wired — FAITHFUL but a latent footgun (L1).** `model_pass` at the EM
   driver (`run_rac.py:1301`) is called without `target_pack`; the non-EM call
   (`:1334`, `target_pack` at `:1349`) is wired. G1/G3 use `seg_mode full` (non-EM), so TARC
   is active. See L1.

---

## Low advisories (non-blocking)

- **L1 — TARC silently no-ops under the EM driver.** `gt_oracle` + `seg_mode
  consensus/selfscore` would build `target_pack` but never thread it into the EM `model_pass`
  (`run_rac.py:1301`), disabling TARC with no warning. Documented (§9.5-5) and out of scope
  for G1/G3, but a one-line guard (`assert not (tarc_source!="off" and seg_mode in
  ("consensus","selfscore"))`) would prevent a future silent-null result.
- **L2 — V1 cache-miss fallback is silent.** `retrieval.py:460-464` reverts a step to
  baseline mining if `_train_targets` is absent (a dead path given epoch-start rebuild).
  Harmless, but a one-line warn would make any future regression auditable rather than
  invisible.
- **L3 — sbatch header wording.** `tarc_g1.sbatch:19-21` labels the numbers "VALSEL acc",
  which are val-*selected* **TEST** acc (the reported result), not val-split acc. The values
  are correct (Item 6); only the label is loose. Cosmetic.

---

## Bottom line

All seven priority items pass; every §9.5 deviation is a faithful repair of the
pre-registration intent (notably the load-bearing V3 sign correction). Row alignment — the
head risk — is correct by lockstep rebuild. OFF is a genuine no-op and its expected numbers
reproduce exactly from the job-12850 logs. Gold-target isolation is code-enforced with the
assert ahead of any side effect.

**Critical = 0, High = 0, Medium = 0, Low = 3 ⇒ PASS_FOR_SUBMIT.**
