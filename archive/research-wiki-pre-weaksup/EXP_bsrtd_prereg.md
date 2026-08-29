# EXP — B-SRTD pre-registration (Balanced Semantic Response-Tensor Distillation, C4/R1 revived)

**Status:** FROZEN. **Freeze timestamp:** `2026-08-10T07:05+12:00` (`2026-08-09T19:05Z`).
**Scope:** one pilot, single submission, MHC-EN + MHC-ZH primary.
**Nothing below may change without a numbered deviation record** in `idea-stage/`.

Everything in §5 (gates), §7 (loss forms), §8 (protocol) and **§9 (decision rules)** was written
before any B-SRTD number existed. The implementation
(`idea-stage/bsrtd_pilot.py`, `bsrtd_lattice.py`, `bsrtd_teacher_score.py`,
`bsrtd_embed_cells.py`) was written after this document and only ever rehearsed on
**synthetic** data fabricated by `bsrtd_lattice.py --signal {planted,nosignal}` (§13).

| item | value |
|---|---|
| candidate | **R1 B-SRTD**, the revival of round-3 **C4** (`idea-stage/IDEA_REPORT.md` §7.1, §7.2, §8.3, §8.9, §8.11) |
| prior disposition | jury 7.0 (round-4 high score); **never killed by a mechanism failure, an occupant or a null** — blocked twice by a missing asset |
| blocker now removed by | the balanced two-axis lattice being built in `data/Counterfactual/BSRTD/` (parallel agent) |
| machine | single RTX 5090, no SLURM; conda env `HateVideo` |
| budget | ~130 head trainings (84 primary + 20 null + 24 secondary), ≈1.5–3 GPU-h, plus ~2.5k teacher API calls and ~2.2k Qwen-VL text forwards |

---

## 1 — Mechanism, in one paragraph

Each seed video contributes a **2×2 semantic intervention lattice**: axis **A** = *attack-target
substitution*, axis **B** = *stance reversal* (assertion ↔ condemnation / counter-speech), giving
four cells `orig / targetsub / stancerev / both`. A teacher scores the hate content of all four
cells; three **finite differences** form the item's *response tensor*

```
dA  = T[targetsub] - T[orig]                                first difference, axis A
dB  = T[stancerev] - T[orig]                                first difference, axis B
dAB = (T[both] - T[stancerev]) - (T[targetsub] - T[orig])   mixed second difference
```

The student — the project's deployed frozen-feature head — produces per-cell probabilities and
therefore its own `(dA, dB, dAB)`. An auxiliary loss matches the student's response tensor to the
teacher's. The claim is narrowly **the named-intervention response tensor**: not "distilling
counterfactual behaviour" (DISCO `2212.10534`), not generic Jacobian matching (`1803.00443`), not
counterfactually-augmented data (Kaushik et al., `1909.12434`) — all three are pre-existing and all
three appear below as *controls*, not as the contribution.

Both sides live in `[0,1]`, so the residual is scale-free and the loss carries **no temperature
hyper-parameter**.

---

## 2 — Data contract (frozen)

Produced by the parallel data-build agent; this pilot **never writes** into
`data/Counterfactual/BSRTD/` except `teacher_scores.jsonl`.

```
data/Counterfactual/BSRTD/{train,val}_lattices_{en,zh}.jsonl
  {"seed_id", "split", "lang", "seed_label",
   "cells": {"orig","targetsub","stancerev","both"},
   "cell_expected_labels": {...},
   "verify": {"verdicts": {cell: {axis_fidelity, fluency, label_consistency,
                                  minimal_pair, reason}}, "round": n}}
```

Read against `data/Counterfactual/BSRTD/BUILD_RECORD_2026-08-10.md` (the build agent's own frozen
record). Two facts from it are load-bearing here:

- **Expected labels (their §2.3, frozen).** Seed label 1 → `1, 1, 0, 0`; seed label 0 → `0, 0, 1, 1`.
  Axis A is designed label-preserving, axis B label-flipping **in both directions**. So the gold
  pattern takes exactly **two** values across the corpus, one per seed class — which is precisely why
  G0.d ("residual sd after removing the per-gold-pattern mean") is the right test for whether the
  teacher carries anything CAD does not, and why `dB` genuinely changes sign between classes.
- **Quotas (their §2.5).** 110 train lattices per language and 40 (en) + 44 (zh) val — i.e. the MVE's
  "≥200 train + 80 val" is met **pooled across languages**, not per language. G0.a and G2's floors
  are written pooled for that reason, with per-language minima set below quota so that ordinary
  verification attrition cannot HALT the pilot while a collapsed language still would.
- **`verify` carries no boolean.** A lattice counts as verified iff every criterion of every
  generated cell reads `PASS` (`bsrtd_lattice.verify_pass`).

**Cell order is frozen** as `("orig","targetsub","stancerev","both")` = indices 0,1,2,3, and the
three difference operators above are applied at those indices, for the teacher and for the student
alike. `lang → dataset`: `en → MHC`, `zh → MHC_zh`. **Splits `train` and `val` only** — the pilot,
the teacher scorer and the cell embedder all assert that `test` is never requested.

---

## 3 — Student (frozen)

- Encoder: **`Qwen2.5-VL-7B-Instruct_HF`**, the same frozen cache all four datasets already have.
- **img_feats are banked, not re-extracted.** In `src/utils/generate_VideoMLLM_embedding_HF.py`
  the image stream is 8 frames + a *fixed neutral instruction* and never sees the title or the
  transcript, so it is invariant under every text intervention **by construction**. Re-extracting it
  would only inject GPU non-determinism into a quantity that must be identical across the four
  cells. Same pattern as the C02 density views
  (`src/utils/generate_c02_density_view_text_embedding_HF.py`).
- **text_feats are re-extracted per cell** by `idea-stage/bsrtd_embed_cells.py`, which imports the
  frame sampler, chat template, pooling span, instruction constants and prompt assembly *unmodified*
  from the deployed extractor. Deployed extractor sha256 at freeze time:
  `1c83d4378678afc12c05ce60dfa9e00b810e5398f436a3f7d51151f8ca35dfa1`
  (pass it as `--pin-sha` to make the run refuse a changed extractor).
- Head: `idea-stage/r4_harness.py::Head` — `classifier_hateClipper`, align/Hadamard fusion,
  map 1024, proj 1024, 3 layers, dropout (0.2, 0.4, 0.1), AdamW lr 1e-4, batch 64, 30 epochs,
  warmup 5. sha256 `31fa182d9e9028c676284ee2b8d38f1eb43a0760b40367ef54c27cf212fbd547`.
- Output cache: `data/CLIP_Embedding/<DS>/{train,dev_seen}_bsrtdcells_Qwen2.5-VL-7B-Instruct_HF.pt`,
  which deliberately carries **no `img_feats` key** so `src/run_rac.py`'s loader cannot silently
  consume it as a full cache.

---

## 4 — Teacher (frozen)

**Frozen order.** Primary **`Qwen/Qwen2.5-72B-Instruct`**; alternate 1
**`meta-llama/Llama-3.3-70B-Instruct`**; alternate 2 **`deepseek-ai/DeepSeek-V3`**. At most two
re-exams (§5, G1) are permitted, in this order, and the order may not be changed after the freeze.

**Why these, and why not Claude.**

1. **Generator ≠ teacher — this is the decisive constraint.** The lattice is being generated by
   Claude under the project's frame/text exemption. If Claude also scored the cells, the "response
   tensor" would largely re-read the generator's own design intent, which is *already* recorded in
   `cell_expected_labels`. That is precisely the round-3 jury's objection to C4 ("amplifies
   generator bias"). Claude is therefore **ineligible as teacher**, notwithstanding the exemption.
2. **Reproducibility.** All three candidates are open-weight, so any third party can re-derive
   `teacher_scores.jsonl` from the released lattice. A closed frontier model would make the central
   artefact of the paper unreproducible.
3. **Not self-distillation.** The student encoder is Qwen2.5-VL-**7B**; every teacher is ≥ 70B, an
   order of magnitude larger, and the primary is a *text* LLM rather than the student's VL sibling.
4. **Bilingual competence.** MHC-ZH requires a teacher that is strong in Chinese; Qwen2.5-72B and
   DeepSeek-V3 both are, which is why the alternates are ordered as they are.

**Scoring protocol (frozen).** Prompt id `BSRTD-T1` (verbatim in `bsrtd_teacher_score.py`), text
only, one deterministic call per cell (`temperature=0, top_p=1, max_tokens=8, seed=20260810`),
integer 0–100 with a rubric anchored on the project's hate definition and an explicit
use/mention carve-out ("content that REPORTS, QUOTES, CONDEMNS or ARGUES AGAINST hate is not hate
evidence"). Parse rule: first integer in the reply, clipped to [0,100], divided by 100. Cache:

```
data/Counterfactual/BSRTD/teacher_scores.jsonl
  {"key":"<lang>|<split>|<seed_id>|<cell>", "text_sha256", "model", "engine",
   "prompt_id", "score", "raw", "ts"}
```

Append-only, idempotent, resumable: a cell is skipped iff a row exists with the same key,
`text_sha256`, `model` and `prompt_id`. Changing the prompt bumps `PROMPT_ID` and invalidates the
cache. Rows tagged `engine="synthetic"` exist only for rehearsal and the pilot **refuses to emit a
primary verdict** if any are present.

---

## 5 — Teacher-quality protection chain (frozen gates)

*Added at freeze time in response to the user's 2026-08-10 challenges: (i) the teacher may simply
be wrong on a cell, and (ii) the gold labels already exist, so why is a teacher needed at all?*
Gate G0 and gates G1/G2 run **before any test contact**; the CAD control that answers (ii) is
in §6 and is rule-bearing in §9 R3(a).

**All three gates are computed on train + val only. Any failure ⇒ HALT: no test run, no verdict.
HALT is not KILL** — it is a statement about the asset or the teacher, not about the mechanism.
If either language HALTs, the whole pilot HALTs; the primary table is not allowed to shrink to one
language, because choosing which language to report after seeing a gate outcome would be selection.

### G0 — asset, encoder responsivity, and information beyond the gold labels

| clause | requirement |
|---|---|
| G0.a size / balance | **pooled over the two languages**: ≥ **200** train and ≥ **80** val lattices; **per language**: ≥ **75** train and ≥ **28** val, ≥ **0.80** passing the builder's verification rubric, and each seed class ≥ **0.30** of lattices |
| G0.b teacher moves | `sign(dB)` agrees with the intended direction in ≥ **0.60** of lattices, and mean `|dB|` ≥ **0.05** |
| G0.c encoder responsivity | a logistic probe trained on train-lattice deltas, asked "is this an axis-A delta or an axis-B delta?", reaches ≥ **0.65** accuracy on val lattices |
| G0.d beyond-gold information | after removing the mean *within each gold-label pattern*, the residual sd of **each** of `dA, dB, dAB` is ≥ **0.03** |

G0.c is the clause that would have caught the failure mode C4 was never able to test: if the frozen
text encoder barely moves under the interventions, the student's finite differences are numerically
degenerate and the mechanism is untestable *with this encoder* regardless of merit.

G0.d is the clause that makes §9 R3(a) answerable in advance: if the teacher's response tensor is a
deterministic function of the gold-label pattern, B-SRTD is **reducible to CAD by construction** and
there is nothing to distil. A G0.d failure permits one re-exam with the next teacher in the §4 list.

### G1 — teacher qualification exam (stratified)

Binarise every teacher score at **0.5** and score it against `cell_expected_labels` (the designed,
human-verified intent for that cell). Requirement, per language:

> **accuracy ≥ 0.90 in EVERY cell type** (`orig`, `targetsub`, `stancerev`, `both`) **and**
> overall accuracy ≥ 0.90.

**Why stratified.** A pooled mean can hide exactly the failure mode that would poison the response
tensor — e.g. a teacher that answers *stance-reversed* cells wrongly as a class would invert the
sign of `dB` on those items while the pooled accuracy still looked acceptable. The per-cell-type
line makes that systematic pattern blocking. False-positive and false-negative rates are reported
per cell type as well, so the direction of any residual bias is on the record.

**Why 0.90 and not the usual 0.75–0.80 one would expect against human labels.** The exam is *not*
agreement with noisy aggregate human annotation; it is agreement with a **deliberately constructed
and individually verified** manipulation. The quantity B-SRTD distils is the *sign and magnitude of
the response to that manipulation*. A teacher that cannot recover a planted, verified stance or
target edit 9 times in 10 cannot supply a trustworthy response tensor, and any gain obtained from it
would be uninterpretable. 0.90 is therefore a competence floor for this specific, easier task, not a
claim about hate-speech classification difficulty in general.

**On failure:** move to the next teacher in the §4 order and re-run the exam (≤ 2 re-exams). If no
teacher passes, **HALT and report**. Teacher order is frozen, so this is not fishing.

### G2 — disagreement-cell filter (frozen, whole-lattice drop)

A lattice enters the distillation loss **iff the teacher's binarised verdict equals
`cell_expected_label` in all four cells**. A single disagreement drops the **whole lattice** —
dropped, not down-weighted, because a clean rule is worth more than a tuned one. Cells whose
expected label is unspecified count as agreeing.

**Post-filter floors:** **pooled** ≥ **150** train and ≥ **60** val lattices, and **per language**
≥ **60** train and ≥ **25** val, else HALT.
The drop count, and the per-cell-type breakdown of *which* cell caused each drop, go in the final
report.

**Foreseeable risk, stated in advance.** With the builder's quotas (220 train / 84 val pooled) the
val floor has only ~13 % headroom: a teacher/gold disagreement rate above roughly 22 % of lattices
will trip G2 and HALT the pilot. That is the intended behaviour — a teacher disagreeing with a
verified rubric on a fifth of the lattices is not a teacher worth distilling — but it means G1 and
G2 are likely to be the binding constraints, not the decision rules.

**The filter is applied identically to every aux-loss arm** (CAD, KD-ABS, JAC, B-SRTD) so the arms
stay matched on data. A useful side effect: after G2 the teacher's binarised verdicts and the gold
labels **coincide by construction on the kept set**, so the CAD control in §6 cannot be handicapped
by label disagreement, and any B-SRTD advantage over CAD cannot come from the teacher "correcting"
a gold label. It can only come from the continuous structure.

---

## 6 — Arms (all same seeds, same hardware, same lattice set)

| arm | objective | what it isolates |
|---|---|---|
| `bce` | BCE only | the project's original baseline |
| `pair` | pairwise-AUC + 0.1·BCE | the §8.10(2b) baseline (pairwise beat BCE in 4/4 cells) |
| **`cad`** | `pair` + BCE on the four cells against their **gold** `cell_expected_labels` | **counterfactually-augmented data** (Kaushik et al. 2020), already occupied in hate speech. **No teacher at all.** |
| `kdabs` | `pair` + per-cell `Huber(p_c, T_c)` | ordinary score distillation on the augmented set: teacher **levels**, no response structure |
| `jac` | `pair` + first-order terms only | Jacobian matching (`1803.00443`); isolates the mixed partial |
| **`bsrtd`** | `pair` + first-order + mixed partial | **candidate** |
| `null` | `bsrtd` with teacher tensors permuted across lattices within seed hard label | coordinate-permutation null |

**Why the teacher can beat the gold labels (explicit hypotheses, stated in advance).**
After G2 the gold pattern is, for most lattices of a given seed class, the *same four bits*. The
teacher's tensor is not. Four specific things it carries that four bits cannot:

- **(H-a) graded magnitude.** Gold says "still hate" after target substitution; the teacher says
  "still hate, but 0.31 less". The size of the target-substitution effect is a real, item-varying
  quantity the binary label cannot express.
- **(H-b) per-example grading.** Two lattices with identical gold patterns can have very different
  `(dA, dB)` magnitudes; CAD gives them identical supervision, B-SRTD does not.
- **(H-c) the interaction term.** In a 2×2 design, four binary labels admit no non-trivial mixed
  second difference once the pattern is fixed — `dAB` computed from gold is identically 0 for the
  dominant pattern. The mixed partial is therefore information CAD **cannot carry at all**, which is
  why it is a separate term in the loss and why `jac` is reported separately.
- **(H-d) dark knowledge.** Soft targets encode the teacher's uncertainty and the local geometry of
  its decision boundary (Hinton et al.).

G0.d measures whether any of this is actually present *before* training; §9 R3(a) measures whether
it is worth anything *at test*. **If it is not, that is a KILL, not a caveat.**

---

## 7 — Loss forms (frozen)

Task loss, identical in every arm (this is `single_pairwise` from `idea-stage/r4_pilot2_jlr.py`):

```
L_task = mean softplus(-(s_pos - s_neg)) + 0.1 * 0.5 * (BCE(s_pos,1) + BCE(s_neg,0))
```
with 2048 sampled positive/negative pairs per step, `max(1, n_train//64)` steps per epoch.

Auxiliary loss, on a minibatch of 64 kept lattices per step, with `p_c = sigmoid(head(img_seed, txt_c))`
and `Huber` delta = **0.1**:

```
L_cad   = BCE(logit_c, gold_c)                                   over cells with a gold label
L_kdabs = Huber(p, T)                                            over the 4 cells
L_first = 0.5 * ( Huber(dA_p, dA_T) + Huber(dB_p, dB_T) )
L_jac   = L_first
L_bsrtd = 0.5 * ( L_first + Huber(dAB_p, dAB_T) )
L_total = L_task + lambda * L_arm
```

`lambda` grid is frozen at **{0.3, 1.0, 3.0}**, selected per (dataset, arm) on **mean validation
macro-F1 across the 3 seeds**, and the *same* grid and *same* selection rule apply to every aux arm
so no control is handicapped. Epoch selection: validation macro-F1 of the head, warmup 5, as in
`r4_harness`. Threshold: `pick_threshold` on validation (max val macro-F1, ties → closest to 0.5).

---

## 8 — Evaluation protocol (user standing instruction, highest priority)

- **train → train; val → select; test → report.** Seeds `{0,1,2}`; every test number is the mean
  over the 3 seeds. **Single submission.** No test quantity enters any selection path.
- Primary (rule-bearing) datasets: **MHC-EN (`MHC`)** and **MHC-ZH (`MHC_zh`)** — the two corpora
  the lattice is built from.
- **Frozen comparator**, per dataset: the arm among `{bce, pair, cad, kdabs}` with the highest mean
  *validation* ROC; ties broken in the fixed order `kdabs > cad > pair > bce` (the
  strongest-looking control wins). This is the R4-2 frozen-comparator machinery, unchanged.
- **Secondary transfer arm (exploratory, NOT rule-bearing):** heads trained on **HateMM** and
  **ImpliHateVid** with the MHC-EN lattices in the auxiliary loss, `lambda` fixed to MHC-EN's
  selection, arms `{pair, cad, kdabs, bsrtd}`. Reported with deltas; it can support no GO and no
  KILL. It is included because the CAD control absorbs the obvious confound (both arms get the same
  out-of-domain cells; only the supervision differs), and excluded from the rules because a
  cross-domain auxiliary set confounds "response structure transfers" with "extra data helps".
  If the lattice build yields no usable MHC-EN cells, this arm is dropped silently — it carries no
  verdict.
- Hardware/frame: all arms in one process, one GPU, one harness; every reported quantity is a
  seed-paired delta computed inside that harness.

---

## 9 — Decision rules (FROZEN — written before any B-SRTD number existed)

Notation: for dataset `d ∈ {MHC, MHC_zh}`, `ΔROC_d` = mean over 3 seeds of
(`bsrtd` test ROC − frozen-comparator test ROC); `ΔF1_d` likewise on macro-F1 at the
validation-selected threshold. Six "cells" = 2 datasets × 3 seeds.

> **R1 — effect size, and no dataset collapses.**
> `mean_d ΔROC_d ≥ +0.005` **AND** `min_d ΔROC_d ≥ −0.005` **AND** `mean_d ΔF1_d ≥ 0`.
>
> **R2 — multi-seed sign consistency.**
> per-seed `ΔROC > 0` in **≥ 5 of the 6** (dataset × seed) cells.
>
> **R3 — the mechanism, not a multi-task regulariser and not CAD.** Both must hold:
> **(a) vs CAD:** `mean_d [ROC(bsrtd) − ROC(cad)] ≥ +0.005` and `> 0` on both datasets.
> **(b) vs KD-ABS:** `mean_d [ROC(bsrtd) − ROC(kdabs)] ≥ +0.005` and `> 0` on both datasets.
>
> **R4 — above the null, and not a coin flip.**
> `mean_d ΔROC_d ≥ Null95 + 0.005` **AND** `LCB95(mean_d ΔROC_d) > 0`, where `Null95` is the 95th
> percentile of `max(0, delta)` over **20** coordinate-permutation null trainings on MHC-EN at the
> selected `lambda`, and `LCB95` is the 5th percentile of a **paired label-stratified bootstrap**
> over test items, resampled independently per dataset, **10,000** reps, `rng = 20260810`.

**GO iff R1 ∧ R2 ∧ R3 ∧ R4. Otherwise KILL.** Explicit KILL conditions, stated so they cannot be
argued around afterwards:

- **R3(a) fails ⇒ KILL.** The gain does not exceed gold-label counterfactual augmentation, so the
  teacher is demonstrably redundant and the method degenerates to CAD — an occupied, unpublishable
  trick. *This is the KILL the user's second challenge demanded, and it is not inferred from any
  other number; it is measured directly by the `cad` arm.*
- **R3(b) fails ⇒ KILL.** The gain survives deleting the response-tensor terms while keeping the
  same teacher calls and the same augmented data, so it is multi-task regularisation from teacher
  levels, not response structure.
- **R4's null clause fails ⇒ KILL.** The gain is inside what permuted response tensors produce.
- **Any gate in §5 fails ⇒ HALT, not KILL**, with the failing clause reported verbatim.

**Reported but NOT rule-bearing** (diagnostics, so a KILL can be explained): `bsrtd − jac` (does the
mixed partial contribute?), the per-`lambda` validation curve, G2 drop counts by cell type, the
teacher's `sd(dA), sd(dB), sd(dAB)` and their beyond-gold residuals, and the entire secondary
transfer arm.

---

## 10 — Null instrument, and the D1-mandated double check

**The null (frozen).** Coordinate permutation: reattach each lattice's teacher response tensor to a
*different* lattice **of the same seed hard label**, leaving every feature, every cell and every
task-loss item untouched. This preserves the marginal distribution of response tensors exactly and
destroys only the item-specific pairing.

**Why this is not D1's mistake.** `idea-stage/R4_DEVIATION_D1_2026-08-10.md` showed that
within-label permutation of *encoder logits* manufactures idealised complementarity, because the
downstream step was a **combination** of encoders and conditional independence given the label is
the best case for combination. Here there is no combination step: the permuted tensor is a
regression *target*, and a target uncorrelated with the item cannot be easier to fit in a way that
helps the primary task. That argument is a prediction, not a proof, so per §8.10(5) it is checked:

**Mandatory pre-run smokes** (`--mode smoke-planted` / `--mode smoke-nosignal`, always with
`--holdout val` so the test cache stays closed):

| smoke | teacher scores replaced by | frozen pass criterion |
|---|---|---|
| planted-signal | `sigmoid(3·(x·w − mean))` for a fixed random direction `w` in the cell text space — i.e. a response tensor the student demonstrably *can* learn | `Δ(bsrtd − pair) − mean(null) ≥ +0.005` **and** `Δ(bsrtd − pair) > 0` |
| no-signal | i.i.d. uniform noise per cell | `|Δ(bsrtd − pair)| ≤ 0.01` **and** `mean(null) − Δ(bsrtd − pair) ≤ 0.01` |

The no-signal criterion is the direct D1 guard: it fails if the permutation makes things *better*
than the real thing. Both smokes must pass before the primary run. Note that R4's null is the
*weakest* of the four rules by design — R3 is the real mechanism guard — so a null that turns out
uninformative cannot by itself manufacture a GO.

---

## 11 — Novelty position (bounded, from `idea-stage/IDEA_REPORT.md` §7.3)

The claim is **only** the named-intervention response tensor as a distillation target. Explicitly
*not* claimed, and each present as a control: counterfactual distillation (DISCO `2212.10534`,
`2510.21631`), Jacobian matching (`1803.00443` — the `jac` arm), counterfactually-augmented data
(Kaushik et al. — the `cad` arm), ordinary logit/score distillation (the `kdabs` arm). A per-candidate
novelty check has **not** been re-run for this freeze; per §8.12 no verdict in this document rests on
a novelty claim, and a fresh check is a precondition for any paper, not for this pilot.

**Known limitation, declared:** the interventions are text-only, so the visual channel does not
respond to them. `img_feats` are held fixed across the four cells (§3). B-SRTD as tested is therefore
a response tensor over the *text* channel of a multimodal head, and the paper may not claim more.

---

## 12 — Run order and launch commands

Nothing may run until the lattice files exist. Order is fixed:

```bash
cd /home/jehc223/Retrieval-hate && conda activate HateVideo

# 0. launcher writes log + pid atomically (avoids the recorded pid-race defect, IDEA_REPORT §8.12)
bash idea-stage/bsrtd_launch.sh teacher     # 1. teacher scores  -> data/Counterfactual/BSRTD/teacher_scores.jsonl
bash idea-stage/bsrtd_launch.sh embed       # 2. cell text_feats -> data/CLIP_Embedding/*/{train,dev_seen}_bsrtdcells_*.pt
bash idea-stage/bsrtd_launch.sh gates       # 3. G0/G1/G2 only, no training, no test contact
bash idea-stage/bsrtd_launch.sh smoke-planted
bash idea-stage/bsrtd_launch.sh smoke-nosignal
bash idea-stage/bsrtd_launch.sh primary     # 4. THE single submission
```

Each stage writes `logging/runs/bsrtd_<stage>/run.log` and `run.pid`; follow with
`tail -f logging/runs/bsrtd_primary/run.log`, check liveness with
`ps -p $(cat logging/runs/bsrtd_primary/run.pid)`. Every stage prints parseable `PROGRESS ...` lines.

**Two external prerequisites, both verified absent on this machine at freeze time:**

1. **Teacher credentials** (not stored in the repo): `BSRTD_TEACHER_API_KEY`, optionally
   `BSRTD_TEACHER_BASE_URL` (default `https://openrouter.ai/api/v1`) and `BSRTD_TEACHER_MODEL`
   (default `Qwen/Qwen2.5-72B-Instruct`). No key is present in the environment.
2. **`Qwen/Qwen2.5-VL-7B-Instruct` weights.** `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct`
   contains only `.incomplete` blobs — the download was started and never finished — so stage 2
   (`embed`) will fetch ~16 GB on first run, or must be pointed at a local snapshot with
   `--model /path/to/snapshot`. Stages 1, 3 and 4 do not need it.

Stage 2 cost estimate: ~2×280 lattices × 4 cells ≈ 2.2k Qwen-VL forwards, with the 8 frames
decoded once per seed and reused across its four cells; ≈ 30–60 min on the 5090. Stage 4:
~130 head trainings, ≈ 1.5–3 GPU-h.

**Stop conditions:** any gate failure ⇒ stop at stage 3 and report. Either smoke failing ⇒ stop and
raise a deviation record before touching stage 4. Stage 4 runs **once**.

---

## 13 — Appendix: synthetic end-to-end rehearsal

The implementation was exercised only on fabricated lattices
(`python idea-stage/bsrtd_lattice.py --signal planted`), which write a complete parallel bundle —
base caches, cell caches, lattice jsonl, teacher jsonl — under `idea-stage/bsrtd_synth/`. No real
lattice, no real teacher call and no real test split was touched during development. **The synthetic
verdict below is not evidence about B-SRTD**; the synthetic teacher is a stand-in and the synthetic
task is fabricated. What is validated is the plumbing: that every gate fires, that the null
instrument discriminates, and that the report path produces a verdict from frozen rules.

**Synthetic bundle:** 260 train + 90 val lattices per language, 500/130/200 base items, `d=64`.
Generative story: a hidden hate level `h` drives both the label and the teacher score; axis A moves
the text along a separate direction and changes `h` by a *per-item* amount; axis B removes the hate
content outright; a non-additive interaction term makes `dAB` non-zero; the teacher reads
`sigmoid(1.8h + 0.15 + noise)` so a few lattices genuinely contradict the gold pattern.

**(1) Gates** — `--mode gates`, all three PASS:

```
G0: PASS   probe acc en 0.9778 / zh 0.9611   (floor 0.65)
           resid sd beyond gold  en [0.151, 0.152, 0.170] / zh [0.146, 0.141, 0.163]  (floor 0.03)
G1 teacher exam [en]: overall 0.9593  orig 0.9686 targetsub 0.9257 stancerev 0.9629 both 0.9800 -> PASS
G1 teacher exam [zh]: overall 0.9636  orig 0.9771 targetsub 0.9200 stancerev 0.9714 both 0.9857 -> PASS
G2 [en]: train 260->221 (floor 150), val 90->81 (floor 60) -> PASS
         dropped by cell: {'orig': 11, 'targetsub': 21, 'stancerev': 9, 'both': 6}
G2 [zh]: train 260->227 (floor 150), val 90->79 (floor 60) -> PASS
gates -> PASS
```

An earlier, noisier synthetic teacher (overall 0.894 / worst cell 0.866) produced
`G1 -> FAIL` and `gates -> HALT (no test run, no verdict)`, confirming the gate blocks rather than
warns, and that `G0.d` fails when axis A carries no per-item variance (`resid sd dA` 0.007 < 0.03).

**(2) Null-instrument smokes** — both PASS, `--holdout val` (test cache never opened):

```
SMOKE smoke-planted:  Delta(bsrtd-pair)=+0.0177  mean(null)=-0.0407  Null95=0.0000
  criterion: Delta(bsrtd-pair) - mean(null) >= +0.005 AND Delta(bsrtd-pair) > 0   ->  PASS
SMOKE smoke-nosignal: Delta(bsrtd-pair)=+0.0024  mean(null)=-0.0409  Null95=0.0000
  criterion: |Delta(bsrtd-pair)| <= 0.01 AND mean(null) - Delta(bsrtd-pair) <= 0.01  ->  PASS
```

This is the D1 check: with a learnable response structure the real tensor gains and the permuted
tensor loses; with no signal at all neither gains, and critically **the permutation does not
manufacture an advantage**. The failure mode that killed R4-1's null is absent here.

**(3) Full report path** — `--mode primary --allow-synthetic-teacher`, 20 epochs, 8 null reps,
400 bootstrap reps (reduced for speed; the frozen run uses 30 / 20 / 10,000):

```
-- MHC, frozen comparator = bce, lambda = {'cad': 1.0, 'kdabs': 3.0, 'jac': 3.0, 'bsrtd': 3.0}
     bce  0.7019   pair 0.6674   cad 0.6800   kdabs 0.6757   jac 0.6911   bsrtd 0.6791   (test ROC)
     => DeltaROC=-0.0228  vsCAD=-0.0009  vsKDABS=+0.0033  vsJAC=-0.0120
-- MHC_zh, frozen comparator = bce, ...
     => DeltaROC=-0.0296  vsCAD=+0.0016  vsKDABS=+0.0038  vsJAC=-0.0030
MeanDeltaROC=-0.0262  Null95=0.0000  bootLCB95=-0.0353  positive_cells=0/6
R1=False R2=False R3a(vs CAD)=False R3b(vs KD-ABS)=False R4=False
VERDICT: KILL
```

Every branch of the verdict block executed, including the frozen-comparator selection, the null,
the paired bootstrap and the KILL print.

**(4) Component rehearsals**

- `bsrtd_teacher_score.py --engine synthetic` on a 3-lattice fixture built from real MHC ids:
  12 cells scored, second invocation reported `cached=12 to_score=0 nothing to do` — idempotent
  and resumable.
- `bsrtd_embed_cells.py --dry-run` on the same fixture: 3 lattices, 0 missing videos, prompt
  assembly byte-identical to the deployed `TEXT_INSTRUCTION`; `--splits test` aborts with
  `REFUSING: B-SRTD lattices are train/val only`.
- `bsrtd_launch.sh gates` against the (absent) real lattice: pid file written by the child,
  log captured, pilot failed fast with `FileNotFoundError: .../train_lattices_en.jsonl` — the
  correct behaviour while the data build is still running.

---

## 14 — Revision log

| when | what | why |
|---|---|---|
| 2026-08-10T07:05+12:00 | initial freeze | — |
| same freeze, pre-publication | **§5 G1 + G2** (teacher qualification exam, stratified by cell type; disagreement-cell whole-lattice filter with post-filter floors) and **§5 G0.d** | user 2026-08-10 challenge: the teacher may score a genuinely hateful cell low, and a systematic per-cell-type failure would be hidden by a pooled mean |
| same freeze, pre-publication | **§6 `cad` arm** promoted to a first-class control and **§9 R3(a)** made a rule-bearing KILL clause; **§6 hypotheses (H-a)–(H-d)** written | user 2026-08-10 challenge: every cell already has a gold label, so the simplest method is plain CAD (Kaushik et al.) — the teacher must be shown to add something beyond it, measured directly rather than inferred |

| same freeze, pre-publication | **§2** rewritten against `data/Counterfactual/BSRTD/BUILD_RECORD_2026-08-10.md`; **G0.a and G2 floors changed from per-language to pooled** (with per-language minima), and `verify` parsing changed from a boolean to the builder's per-cell rubric | the build agent's frozen quotas are 110 train lattices *per language* (220 pooled), and its `verify` field carries per-criterion verdicts, not a `pass` flag. The earlier per-language floors would have HALTed the pilot on a correctly-built asset — a false HALT, the same class of defect as deviation D1's false KILL |

All three revisions were made **before** the freeze was published and **before** any B-SRTD number
existed, so they add gates and controls, or align a threshold with the asset that actually exists,
rather than modifying an already-frozen clause after seeing a result.
