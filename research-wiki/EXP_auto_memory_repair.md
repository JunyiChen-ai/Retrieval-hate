# EXP: Automatic memory repair — can a two-vote rule recover the manual editing gain without a human?

> **Status: PRE-REGISTERED (design frozen before any test-set evaluation).**
> This top section (motivation, thresholds, conditions, success criteria) was written
> and committed BEFORE running condition C/D/E on any test set. Results are appended in a
> separate section below and were not used to tune any threshold. Numbers are produced by
> `scripts/analysis/auto_memory_repair.py`; MLLM verdicts by
> `scripts/analysis/judge_memory_archive.py` (SLURM job
> `scripts/slurm/judge_memory_archive.sbatch`).

## Motivation

The manual memory-editing demo (`research-wiki/DEMO_memory_editing.md`,
`scripts/analysis/memory_editing_demo.py`) showed a **retraining-free** repair: two
train-side memory entries (`XScP1AiMkNM` avocado-jam, `QvPp8Q7QhWE` counting-money —
labelled hateful but the MLLM archive describes them as plainly benign) were found by
train-side forensics and deleted from the kNN memory bank, lifting MHC-EN test accuracy
**0.8075 → 0.8199** (macro-F1 0.7626 → 0.7748) at seed 0 with **zero retraining**. A human
was in the loop: the two ids came from W2 forensics + manual audit.

**Question under test:** does a fully AUTOMATIC rule — no human, no per-dataset tuning —
recover that gain, and does an MLLM semantic check add anything over a purely embedding-based
label-noise detector (Cleanlab-style)?

## Method (frozen decision path — identical to the winning configs and the manual demo)

- Frozen head per seed: the val-selected best checkpoint of the archive-kNN α=0.25 winners
  (EN frozen-Qwen jobs 12210/12219/12220/12221 seeds 0-3; ZH LoRA-Qwen jobs
  12207/12215/12216/12217/12218 seeds 0-4). No retraining anywhere in this experiment.
- Memory bank = **TRAIN split only** (matches the manual demo: EN N=549, ZH N=579), keys =
  `[l2n(fused) | 0.25·l2n(archive_CLIP_text)]` with the **v1 archives** (`--archive_feats
  auto`). kNN decision = faiss cosine, topk=20, arithmetic weights, similarity-signed vote
  (`use_sim=True`) — imported verbatim (`augment`, `knn_eval`) from
  `memory_editing_demo.py`, which reproduces the training-log floor bit-for-bit.
- **Why v1 archives throughout (keys AND the MLLM judge):** conditions A and B must reproduce
  the manual demo (0.8075 / 0.8199) as a gate, which pins the keys to v1. The manual finding
  and its two noise ids were derived from v1 archives, so a like-for-like "can the machine
  replace the human's reading of the *same* evidence" test also feeds the MLLM the v1 archive.
  As a robustness note, the MLLM additionally judges the v2 archives; v2 verdicts are reported
  but do NOT define the primary condition C.

## The automatic deletion rule — two independent votes over TRAIN entries only

All decisions use train-side information only; test sets are touched exactly once per
(condition × seed). Both thresholds below are FIXED now and are not tuned on any val/test data.

1. **Embedding vote (automatic, per seed).** Leave-one-out kNN over the memory bank: for
   train entry *i*, retrieve its **k=10** nearest OTHER entries in the SAME augmented-key
   cosine space the decision uses (self excluded). Let *d_i* = fraction of those 10 neighbours
   carrying the **opposite** label to *i*. **Flag** *i* iff *d_i ≥ 0.80* (≥8/10 neighbours
   disagree). *d_i* (continuous) is also recorded.
2. **Semantic vote (MLLM, automatic, seed-independent).** Qwen2.5-VL-7B-Instruct reads entry
   *i*'s archive JSON (target_groups / mechanism / modality_cues / explicitness /
   neutral_summary) **plus its dataset label** and emits exactly one of
   {`SUPPORT`, `CONTRADICT`, `UNSURE`} + a one-line reason. Greedy decoding (temperature 0,
   `do_sample=False`). Prompted to answer `CONTRADICT` **only** for a clear mismatch
   (description plainly benign but label hateful, or plainly hateful but label benign).
   The verdict depends only on (archive, label), so it is computed ONCE per dataset and reused
   across all seeds.
3. **Deletion rule (the method, condition C).** Delete entry *i* **iff** embedding vote flags
   it **AND** semantic vote = `CONTRADICT`. No threshold tuning; no human.

## Conditions (each cell = ONE single test measurement per seed — no repeats, no selection)

| id | condition | deletion set |
|----|-----------|--------------|
| A | floor (no deletion) | ∅ |
| B | manual (reproduce demo) | {`XScP1AiMkNM`, `QvPp8Q7QhWE`} — EN only; **N/A for ZH** (no manual noise ids ⇒ B≡A) |
| C | **auto two-vote (the method)** | {embedding-flagged AND CONTRADICT} |
| D | embedding-vote-only (Cleanlab-style control) | {embedding-flagged} |
| E | random control | \|C\| entries drawn with `random.Random(0)` (count matched to C; if \|C\|=0 then E≡A) |

- Datasets: **MHC (EN, primary — where the manual gain lives)**, **MHC_zh (ZH, negative /
  robustness control)**. HateMM has no archives, so ZH is the only negative control (expected
  by the task).
- Metrics: **accuracy + macro-F1** via the standard kNN vote (`knn_eval`, `use_sim=True`).
  Report per-seed values AND mean±std for A–E on both datasets; plus paired per-seed deltas
  **C−A, C−D, D−A**. Report WHICH entries each of C and D deletes (video ids + archive
  one-liner) for post-hoc human audit.

## Pre-registered success criteria

1. **Reproduction gate (must pass first).** A(EN, seed 0) = 0.8075 / 0.7626 and B(EN, seed 0)
   = 0.8199 / 0.7748, bit-for-bit vs the manual demo. If this fails, stop and debug — no other
   result is trustworthy.
2. **C recovers the manual gain.** C−A ≥ **+1.0 acc pt** on the manual-demo setup (EN seed 0),
   AND **mean C−A > 0** across EN seeds.
3. **C does no harm on the ZH control.** mean |C−A| small on MHC_zh, with no consistent
   negative sign across seeds.
4. **The MLLM earns its place: C > D.** mean paired **C−D > 0** — i.e., the semantic gate
   prevents harmful over-deletions that the embedding vote alone (D) would make. **If D alone
   already captures the gain (C ≈ D, or D ≥ C), we say so plainly — that kills the MLLM's role,
   and that verdict is equally valuable.**
5. **Honesty guard.** Per the multi-seed post-mortem (`exp-archive-knn-seeds.md`), on a
   ~150-sample test 1 acc pt ≈ 1.6 videos and seed/selection noise dominates. We therefore
   report the per-seed sign pattern and mean±std with an explicit small-N caveat and make **no
   significance claim** from n=4–5 seeds; the headline is directional (sign of the paired
   deltas), not a p-value.

## Hard rules honoured

- All GPU work via SLURM (`sbatch`, no `--time`, `HF_HUB_OFFLINE=1`, `WANDB_MODE=disabled`).
  Only the MLLM judge needs a GPU (a few hundred train entries × 2 archive versions per
  dataset); the kNN evals are CPU-only.
- No cross-seed ensembling — per-seed only, aggregated as mean±std / paired deltas.
- No checkpoint/cache overwrites; no `.pt`/`.safetensors` committed to git; pulled checkpoints
  deleted locally when done.

---

## RESULTS

Run 2026-07-06. Semantic vote = SLURM job 12347 (Qwen2.5-VL-7B, greedy), verdicts in
`scripts/analysis/memory_repair_out/verdicts.json`; CPU evals via
`scripts/analysis/auto_memory_repair.py` (`results_v1.json` primary, `results_v2.json`
robustness). Reproduction gate passed exactly on both datasets (EN s0 A=0.8075/0.7626,
B=0.8199/0.7748; ZH s0 A=0.8523/0.8270); all 9 seeds' floor A match the logged
val-selected multi-seed values (`exp-archive-knn-seeds.md`).

### Semantic-vote distribution over the TRAIN memory bank

| dataset / version | n | SUPPORT | CONTRADICT | UNSURE |
|---|---|---|---|---|
| MHC (EN) v1 | 549 | 466 | **80** (14.6%) | 3 |
| MHC (EN) v2 | 549 | 453 | 88 | 8 |
| MHC_zh (ZH) v1 | 579 | 454 | **119** (20.6%) | 6 |
| MHC_zh (ZH) v2 | 579 | 438 | 132 | 9 |

The MLLM flags **both** manual noise ids as CONTRADICT with correct reasoning
(`XScP1AiMkNM` guacamole: "ordinary, harmless content without targeting any protected
group"; `QvPp8Q7QhWE` money-counting: "affirms ordinary content ... contradicts hateful
label"). So the semantic signal *is* right. But their embedding-vote disagreement is only
0.50 / 0.60 (< 0.80), so **neither is embedding-flagged** — the AND rule (C) cannot delete
them by construction. This is the central mechanism of the result below.

### MHC (EN) — primary (v1 verdicts), 4 seeds — accuracy / macro-F1

| seed | A floor | B manual | C auto (method) | D emb-only | E random | F sem-only* | flag | \|C\| | \|D\| | \|F\| |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8075/0.7626 | **0.8199/0.7748** | 0.8075/0.7626 | 0.8012/0.7533 | 0.8012/0.7533 | 0.8075/0.7686 | 35 | 11 | 35 | 80 |
| 1 | 0.7640/0.7145 | 0.7640/0.7145 | 0.7640/0.7145 | 0.7640/0.7180 | 0.7640/0.7180 | 0.7764/0.7360 | 40 | 11 | 40 | 80 |
| 2 | 0.7950/0.7505 | 0.7950/0.7505 | 0.7950/0.7505 | 0.7826/0.7387 | 0.7826/0.7387 | 0.8012/0.7625 | 39 | 11 | 39 | 80 |
| 3 | 0.8075/0.7713 | 0.8075/0.7713 | 0.8075/0.7713 | 0.8075/0.7713 | 0.8075/0.7713 | 0.8075/0.7713 | 41 | 13 | 41 | 80 |
| **mean±std acc** | 0.7935±0.0178 | 0.7966±0.0208 | **0.7935±0.0178** | 0.7888±0.0170 | 0.7888±0.0170 | 0.7981±0.0128 | | | | |
| **mean±std F1** | 0.7497±0.0216 | 0.7528±0.0240 | **0.7497±0.0216** | 0.7453±0.0195 | 0.7453±0.0195 | 0.7596±0.0140 | | | | |

Paired per-seed deltas (acc / macro-F1):

| delta | acc | macro-F1 |
|---|---|---|
| **C − A** (method vs floor) | **+0.0000 ± 0.0000 (+0/4)** | +0.0000 ± 0.0000 (+0/4) |
| **C − D** (method vs emb-only) | **+0.0047 ± 0.0052 (+2/4)** | +0.0044 ± 0.0063 (+2/4) |
| D − A (emb-only vs floor) | −0.0047 ± 0.0052 (+0/4) | −0.0044 ± 0.0063 (+1/4) |
| F − A (sem-only vs floor)* | +0.0047 ± 0.0052 (+2/4) | +0.0099 ± 0.0079 (+3/4) |

### MHC_zh (ZH) — control (v1 verdicts), 5 seeds — accuracy / macro-F1

B (manual) is **N/A** for ZH: the two manual noise ids are EN-specific and absent from the
ZH bank, so B ≡ A by definition.

| seed | A floor | C auto (method) | D emb-only | E random | F sem-only* | flag | \|C\| | \|D\| | \|F\| |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8523/0.8270 | 0.8523/0.8270 | 0.8389/0.8090 | 0.8591/0.8359 | 0.8456/0.8278 | 26 | 14 | 26 | 119 |
| 1 | 0.8456/0.8158 | 0.8456/0.8158 | 0.8456/0.8158 | 0.8456/0.8158 | 0.8255/0.7956 | 30 | 14 | 30 | 119 |
| 2 | 0.8322/0.8046 | 0.8322/0.8046 | 0.8389/0.8135 | 0.8322/0.8046 | 0.8255/0.8002 | 36 | 18 | 36 | 119 |
| 3 | 0.8188/0.7837 | 0.8188/0.7837 | 0.8188/0.7837 | 0.8188/0.7837 | 0.8054/0.7734 | 36 | 19 | 36 | 119 |
| 4 | 0.7852/0.7266 | **0.8054/0.7579** | 0.7919/0.7330 | 0.7919/0.7330 | **0.8456/0.8242** | 24 | 11 | 24 | 119 |
| **mean±std acc** | 0.8268±0.0238 | **0.8309±0.0172** | 0.8268±0.0196 | 0.8295±0.0231 | 0.8295±0.0151 | | | | |
| **mean±std F1** | 0.7915±0.0355 | **0.7978±0.0246** | 0.7910±0.0312 | 0.7946±0.0351 | 0.8042±0.0200 | | | | |

Paired per-seed deltas (acc / macro-F1):

| delta | acc | macro-F1 |
|---|---|---|
| **C − A** (method vs floor) | **+0.0040 ± 0.0081 (+1/5)** | +0.0063 ± 0.0125 (+1/5) |
| **C − D** (method vs emb-only) | **+0.0040 ± 0.0081 (+2/5)** | +0.0068 ± 0.0126 (+2/5) |
| D − A (emb-only vs floor) | +0.0000 ± 0.0074 (+2/5) | −0.0005 ± 0.0094 (+2/5) |
| F − A (sem-only vs floor)* | +0.0027 ± 0.0293 (+1/5) | +0.0127 ± 0.0430 (+2/5) |

\* F (semantic-only) is EXPLORATORY / post-hoc, not one of the pre-registered A–E
conditions; it isolates the MLLM signal without the embedding gate.

### v2 robustness (re-run with the v2 archives feeding the semantic vote)

Same pipeline, semantic vote reads v2 archives instead of v1 (keys/decision unchanged =
v1, so A/B are identical). Paired deltas (acc):

| dataset | C − A | C − D | F − A |
|---|---|---|---|
| MHC (EN) | −0.0016 ± 0.0027 (+0/4) | +0.0031 ± 0.0054 (+1/4) | −0.0124 ± 0.0098 (+0/4) |
| MHC_zh (ZH) | +0.0013 ± 0.0027 (+1/5) | +0.0013 ± 0.0066 (+1/5) | −0.0067 ± 0.0225 (+1/5) |

Under v2, C−A is if anything slightly negative on EN and near-zero on ZH; the lone positive
cell (ZH seed 4 under v1) shrinks (C=0.7919 vs 0.8054), i.e. it is archive-version-fragile —
consistent with noise, not a real repair.

### What each rule actually deletes (audit)

- **C (auto two-vote)** deletes 11–19 entries/seed that are BOTH embedding outliers AND
  CONTRADICT. These split into (i) benign-labelled videos whose archive quotes a slur
  (`n6YTbpnaLnA` "cunt", `H_dqS8HBRNQ`, `b7WbQg-4LCY`) and (ii) hateful-labelled videos the
  archive read as benign (`5snzFreG79c`, `EU-dip0ITa4`, `MZFOUge3kM0`). Deleting them flips
  **zero** EN test votes on every seed (C ≡ A).
- **D (embedding-only, Cleanlab-style)** deletes 35–41 entries/seed. Crucially it deletes
  **genuinely hateful memory that is merely embedding-hard** — e.g. `OC7D6mi_Dao`
  (targets=lesbian/gay, explicit; MLLM SUPPORT), `UPG99ifsalw` (survivor recounting sexual
  abuse; SUPPORT, emb_disagree 1.00), `aeOm9oT0_qk` (2002 Gujarat gang-rape case; SUPPORT).
  These are correctly-labelled hard positives sitting among benign neighbours; removing them
  is why D is net-negative on EN (−0.47 acc pt).
- **The semantic gate in C vetoes exactly those D-deletions** (they are SUPPORT, not
  CONTRADICT) — that is the entire, and only, source of C > D.
- **F (semantic-only)** deletes 80 EN / 119 ZH entries (incl. both manual ids). Most are
  benign-labelled videos that *quote/discuss* a slur or sexual term (counter-speech,
  reclaimed language, satire — correctly benign), so deleting them removes good memory; the
  net effect is ~neutral on EN (the true noise-id gain is cancelled by collateral) and very
  high-variance on ZH.

### Verdict vs the pre-registered success criteria

1. **Reproduction gate — PASS** (exact, both datasets).
2. **C recovers the manual gain — FAIL.** C − A = **0.0000** on EN seed 0 and on all 4 EN
   seeds (criterion asked for ≥ +1.0 acc pt). The manual +1.24 pt is never reproduced: the
   two noise ids the manual edit removed are semantic contradictions, not embedding-space
   outliers (disagreement 0.50/0.60 < 0.80), so the AND rule structurally cannot delete them.
3. **C does no harm on ZH — PASS.** C − A ≥ 0 on all 5 ZH seeds (mean +0.40 acc pt, driven
   entirely by seed 4 and fragile to the archive version). No consistent negative.
4. **The MLLM earns its place (C > D) — PASS, but only defensively and within noise.**
   C − D = +0.47 pt (EN) / +0.40 pt (ZH). The semantic vote's value is that it *vetoes* the
   embedding-only rule's deletion of correctly-labelled hard-positive memory — a real,
   auditable mechanism — but it is sub-1-point (≈ the 1.6-video noise floor) and yields no
   positive repair on top of the floor.

### Plain-language bottom line

The automatic two-vote rule **does not recover the manual repair, and does not beat the
floor.** The MLLM semantic check is genuinely competent — on its own it correctly re-flags
the exact two entries a human flagged, with the right reasons — so the archive makes the
memory bank **auditable**: a machine reading the same evidence reaches the same call. But
"auditable" did not convert into unsupervised "repairable." Two failures compound: (a) the
manual gain was a **human-precision, 2-entry surgical edit**, and the semantic vote at full
strength is far blunter (it fires on 15–21% of the bank, mostly on correctly-labelled videos
that merely quote hateful language), so semantic-only deletion washes the gain out in
collateral; and (b) the pre-registered AND with the embedding vote — meant as a
precision filter — instead **excludes the very entries the manual edit targeted**, because
those entries are label/description contradictions rather than embedding-space label-noise.
The two votes are near-orthogonal noise detectors. The one thing the MLLM demonstrably buys
is **defense**: it stops a Cleanlab-style embedding-only rule from deleting genuinely hateful
but embedding-hard memory (abuse testimony, gang-rape reporting, slur-targeting videos),
which is why C > D. That is a real result — the semantic vote earns a *guard-rail* role, not
a *repair* role — but on these ~150-sample test sets it is within the seed/selection noise
band, so we make no accuracy claim for it. The manual editability demo stands; automating it
with this specific two-vote rule does not close the human-in-the-loop gap.

*(Tables/numbers generated by `scripts/analysis/auto_memory_repair.py` from
`verdicts.json`; verdict prose human-written against `results_v1.json` / `results_v2.json`.)*
