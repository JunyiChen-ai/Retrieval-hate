# CAT close-out — frozen design

**Scope.** The four validation items listed in `idea-stage/IDEA_REPORT.md` §13.6 as *not done*
(items 2, 3, 4, 5) plus the disclosure paragraph (item 6). **No new method idea is tested here.**
Nothing in this document proposes a new candidate; every leg is validation of the single banked
entry `CAT`.

**This file is committed to git BEFORE any code for these legs is written and before any arm
metric on any of the seed ranges below exists.** The commit hash of this file is the freeze point.

**Red lines held.** (1) No test-set label is used to select any epoch, threshold, arm, span, fold
or hyper-parameter — test labels enter only the final reported metric, under the user's
2026-08-09 test-set protocol ruling. (2) Every decision rule below is fixed here, before results.
(3) No candidate metric is computed during design or implementation. (4) Each leg is a single
submission; re-runs are only permitted for infrastructure failure and must be filed as a
deviation.

**Cost class.** Local RTX 5090, zero API, zero DashScope, no cloud. Estimated ≈5.5 GPU-hours.

---

## 0. What `CAT` is (restated, no re-derivation)

Per dataset, from ONE frozen merged Qwen2.5-VL-7B forward per stream:

- `img_feats` = `n(mean(h_28[0:hdr]))` — the deployed "prefix" span of the **image-prompt**
  forward (`RO.IMG_INSTRUCTION`), 3584-d.
- `A0` (the deployed text read-out) = `n(mean(h_28[hdr:]))` — the trailing
  `<|im_start|>assistant\n` header, 3 tokens, of the **text-prompt** forward
  (`RO.TEXT_INSTRUCTION` + title + transcript), 3584-d.
- `TXT` = `n(mean(h_28[v_end:hdr]))` — the transcript/instruction content positions of the SAME
  text forward, 3584-d.
- **arm `A0`**: `text_feats = A0` (3584). **arm `CAT`**: `text_feats = [A0 ‖ TXT]` (7168).
  **arm `RAND`**: `text_feats = [A0 ‖ n(A0·R)]` (7168), matched-width control, `R` the fixed
  Gaussian whose sha256 is pinned in `idea-stage/r6_readout/build_meta.json`.

`hdr` = index of the LAST `<|im_start|>`; `v_end` = 1 + index of the LAST `<|video_pad|>`;
degenerate guard `v_end >= hdr → v_end = 0`. Layer 28 only. Head: `src/run_rac.py` with the
byte-identical hyper-parameter line of `idea-stage/reaudit/run_grid.sh`.

**Read-out protocols** (both computed from the same runs, unchanged from every prior round):
`P1` = epoch `argmax_{e≥5}` dev macro-F1 (earliest tie), test macro-F1 @0.5. `P2` = epoch 29.
`P1` is the judging protocol everywhere below; `P2` is reported for sign agreement only.

**Statistics.** Paired bootstrap over the pairing unit (seed, or CV cell), B = 20000,
RNG seed 20260817, percentile 95 % CI. Bar = **+0.005** macro-F1.

---

## 1. Seed ledger

Consumed before this freeze (enumerated from every `*_s<seed>.trainlog` under `logging/runs/`
plus the R10/R11/R12 documents): **0–89, 100–129, 300–329, 400–429, 500–529, 600–629, 700–729,
800–829, 900–929, 41000–41029, 50700–50729.**

Allocated here, disjoint from all of the above and from each other:

| leg | seeds |
|---|---|
| A — MHC-ZH re-extraction grid | **1300–1319** (20) |
| B — MHC-EN transport grid | **1400–1429** (30) |
| C — repeated stratified CV | **1500–1524** (25, one per CV cell, shared by both arms) |

Leg D consumes no seeds (it re-reads the already-dumped R10-COMBO per-item logits).

---

## 2. Leg A — end-to-end extraction reproducibility (§13.6 item 2)

### 2.1 What is and is not being replicated

The four banked `CAT` replications (R10 leg 1, R10-COMBO, R11-UNION, R12-ANCHOR) are four
disjoint **head-seed** ranges over **one** feature extraction. Leg A repeats the extraction from
raw video and re-measures.

**In scope of the perturbation:** execution order. Both streams for MHC-ZH are re-extracted in a
**single fresh pass with the item order within each split reversed**, then permuted back to
ground-truth order by id. Every mathematically-relevant setting is held frozen and identical:
model `Qwen/Qwen2.5-VL-7B-Instruct`, LoRA `logging/lora/MHC_zh` (sha256 verified against
`35a510f4ad84542c798939cfdb340b00317a5b8a2c670b07ced8d1869dd7b438`), `num_frames=8`,
`max_pixels=200704`, bf16, sdpa, `use_cache=False`, the two prompt strings, the frame sampler,
the span definitions, layer 28.

**Declared in advance, so it cannot be presented as a finding later:** this pipeline has no
stochastic component by construction — `np.linspace` frame sampling, deterministic decoding,
greedy single forward, no dropout at inference. The only way the two passes can differ is
non-deterministic GPU kernel scheduling. **Leg A is therefore a determinism audit plus a
representation-level re-measurement, not a sampling-noise study.** Both possible outcomes are
pre-interpreted in §2.4.

### 2.2 Extraction

New script `idea-stage/cat_closeout/extract_cc.py`: a thin fork that imports the model load, the
frame sampler, the prompt strings, `_build_messages`, `_encode_readout` and `_pool_span`
**verbatim** from `src/utils/generate_VideoMLLM_embedding_readout_HF.py` (module `RO`) and the
span decomposition **verbatim** from `idea-stage/r10_tokpos/extract_tokpos.py` (module `TP`).
It re-implements only (a) that exactly two forwards are run per item — the image-prompt forward
(pooled with the frozen `RO._pool_span(span="prefix")` at layer 28) and the text-prompt forward
(pooled with the frozen `TP._spans_from_hidden` at layer 28) — and (b) the item order.

Output: `data/CLIP_Embedding/<DS>/{split}_<BASE>-cc.pt`, suffix `cc` never used before, so no
banked cache can be clobbered. A hard guard raises if the suffix is not `cc`.

**Belt A1 (extraction, gating).** On the first 12 items of each split, the `A0` span produced by
`TP._spans_from_hidden` must equal `RO._pool_span(span="response")` computed on the *same*
forward with **max abs diff exactly 0.0**, and the `PRE`/img span must equal
`RO._pool_span(span="prefix")` with max abs diff exactly 0.0. Failure raises and halts. (This is
the R10 deviation-D2 belt, applied to both streams.)

**Belt A2 (identity, gating).** id order and labels of the `-cc` cache must equal the
ground-truth jsonl order after the reverse-permutation, elementwise. Failure halts.

### 2.3 Numerical comparison (descriptive, no verdict power)

Against the banked R10 `-tp` cache (text spans) and the banked `-ro_L28` cache (img), per split
and per span, report: max abs diff, mean/min cosine, fraction of rows bit-identical. The `-ro_L28`
comparison is **cross-hardware** (A100-extracted) and is explicitly descriptive drift, exactly as
in `R10_TOKPOS_DEVIATION_D1/D2`; the `-tp` comparison is same-hardware and is the determinism
read-out.

### 2.4 The grid and the frozen rule

Arms `A0`, `CAT`, `RAND` built by `idea-stage/cat_closeout/build_cc_arms.py` (a fork of
`idea-stage/r10_tokpos/build_arms.py` that takes both streams from the `-cc` cache instead of
carrying `img_feats` over from the banked cache), tag prefix `CCA`. MHC-ZH, **20 seeds
1300–1319**, `--dump_head_scores` on.

**Frozen judgement contrast: `CAT − A0` under P1.**

> **REPRODUCED** iff mean ≥ **+0.005** and the paired-bootstrap 95 % CI excludes zero.
> Anything else is **NOT REPRODUCED AT THE FROZEN BAR**, and is reported as such with the
> observed mean and CI. No re-run, no seed extension, no protocol switch is permitted on a miss.

Secondary, reported without verdict power: `CAT − RAND`, `RAND − A0`, P2 signs, and the §2.3
numerical table.

**Pre-committed interpretation, fixed before any number exists:**
- If the two extractions are **bit-identical** (max abs diff 0.0 on every span and split), Leg A
  establishes that the extraction is deterministic and therefore that the four banked
  replications share a cache that carries **no** hidden extraction variance — the effect is not
  an artefact of one lucky extraction draw. The head grid is then a fifth head-seed replication
  on a re-derived cache and is reported as such, with the honest statement that it adds
  **optimisation** evidence, not independent representation evidence.
- If the two extractions **differ numerically**, the head grid is the representation-level
  replication §13.6 asked for and the verdict above carries its full weight.

---

## 3. Leg B — MHC-EN transport check (§13.6 item 4)

### 3.1 Standing, declared in advance

MHC-EN's test split participated in the broader campaign. This is therefore a **transport check,
not a fresh confirmation**, and must be described that way in every downstream sentence. The
exact frozen `CAT` configuration is carried over with **no retuning of anything**.

### 3.2 Encoder provenance

The MHC-EN task LoRA is not on this workstation. It is restored from Backblaze B2:
`b2:junyi-data/RGCL_video/logs/lora/MHC` → `logging/lora/MHC`, **top-level files only**
(`--exclude "checkpoint-*/**"`). Its sha256 is recorded in the result document.

**No prior sha256 record for the MHC-EN adapter exists in this repository** (unlike the ZH and
HateMM adapters, whose hashes are pinned in `refine-logs/MNTP_S1_RECORD.md` §1.1). This is a
provenance gap and is stated as such. The substitute belt is behavioural:

**Belt B1 (adapter identity, gating).** The re-extracted deployed spans must reproduce the banked
MHC-EN deployed cache `{split}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` (A100-extracted, produced by
this adapter) at **mean cosine ≥ 0.95 and min cosine ≥ 0.90**, for both `img_feats` (vs the
img-prompt prefix span) and `text_feats` (vs `A0`). A wrong adapter, wrong prompt or wrong span
reads ≈0.3–0.6 (measured in `R10_TOKPOS_RESULT.md` §2.0 and `build_meta_MHC_zh.json`), so this
floor separates "right encoder, platform drift" from "wrong encoder". Failure halts the leg.
The observed cosines are descriptive drift, never a gate on any arm.

Belts A1 and A2 of §2.2 apply unchanged to this extraction.

### 3.3 The grid and the frozen rule

Same `extract_cc.py`, dataset `MHC`, splits train/val/test, LoRA `logging/lora/MHC`, base tag
`Qwen2.5-VL-7B-Instruct-LoRA_HF`, **forward item order** (no reversal — the reversal is Leg A's
perturbation, not part of the transport configuration). Arms `A0`, `CAT`, `RAND`, tag prefix
`CCB`. **30 seeds 1400–1429**, `--dump_head_scores` on.

**Frozen judgement contrast: `CAT − A0` under P1.**

> **TRANSPORTS** iff mean ≥ **+0.005** and the paired-bootstrap 95 % CI excludes zero.
> Otherwise **DOES NOT TRANSPORT AT THE FROZEN BAR**, reported with the observed mean and CI.
> A miss is a real finding about the effect's coverage and is written into the close-out as
> such. No re-run, no seed extension, no protocol switch.

Secondary, no verdict power: `CAT − RAND`, `RAND − A0`, P2 signs.

### 3.4 ImpliHateVid

`data/video/ImpliHateVid/All` does not exist (the directory holds only `_id2b2path.tsv`); the raw
videos are gone from this workstation. The dataset is therefore **not measurable on this axis**
and is recorded as such, with no substitute and no estimate.

---

## 4. Leg C — repeated stratified cross-validation on train+dev (§13.6 item 5)

### 4.1 Purpose and limits, declared in advance

This shows whether `CAT − A0` is carried by the one fixed split. **It cannot and does not undo
adaptive dataset reuse** (see §6). It is sampling robustness, nothing more.

### 4.2 Construction

Zero new extraction: the pooled population is the **train ∪ dev_seen** rows of the already-banked
`R10CB-A0` and `R10CB-CAT` caches — the exact caches behind the R10-COMBO / R11 / R12 `CAT`
numbers. MHC-ZH: 579 + 78 = **657** rows. HateMM: 744 + 107 = **851** rows.

**The official `test_seen` split is never loaded in this leg.** Belt C1 (gating): the id set of
every fold is asserted disjoint from the official `test_seen` id list; failure halts.

For repeat `r ∈ {0..4}`: `StratifiedKFold(n_splits=5, shuffle=True, random_state=20260818 + r)`
over the pooled population, stratified on the binary label.
For fold `f ∈ {0..4}`: `EVAL` = fold `f`; `REST` = pool \ `EVAL`. `REST` is split by
`StratifiedShuffleSplit(n_splits=1, test_size = round(|REST| · d), random_state = 1000·(20260818+r) + f)`
into `INNER_TRAIN` / `INNER_DEV`, where `d = |dev_seen| / (|train| + |dev_seen|)` (= 78/657 for
MHC-ZH, 107/851 for HateMM) so the inner dev fraction matches the deployed protocol.

Cache cell `(r,f)` is written as `{train: INNER_TRAIN, dev_seen: INNER_DEV, test_seen: EVAL}`
under tags `CCCVr{r}f{f}-A0` and `CCCVr{r}f{f}-CAT`. `INNER_DEV` drives the P1 epoch rule; `EVAL`
is scored once at that epoch and is never used for any selection.

Head seed for cell `(r,f)` = **1500 + 5r + f**, identical for both arms (this is what makes the
25 cells a paired sample). 25 cells × 2 arms × 2 datasets = **100 runs**. Both datasets are run;
`RAND` is not run in this leg.

### 4.3 The frozen read-out

Pairing unit = the CV cell (25 per dataset). Report, per dataset: the mean of `CAT − A0` over the
25 cells, its paired-bootstrap 95 % CI (B = 20000, seed 20260817), the across-cell SD, the
per-repeat means (5 numbers), the number of cells with a positive difference, and the same
quantities under P2.

> **CV-SUPPORTED** iff the P1 mean over cells is **> 0** with the 95 % CI excluding zero.
> Reported additionally, without gate status: whether the mean also clears **+0.005**.
> If the CI includes zero the leg reads **NOT CV-SUPPORTED** and the close-out must say the
> effect is not separable from split choice at this sample size.

The CV mean is **not** comparable in level to the fixed-split delta (different training-set size,
different evaluation-set size, different dev size). Only its sign, its CI and its dispersion are
interpreted. This is fixed here so it cannot be re-negotiated after seeing the number.

---

## 5. Leg D — per-item read-out audit (§13.6 items 3 and 4-adjacent)

Descriptive. **No verdict, no gate, no bar.** Zero new runs, zero GPU.

Source: the per-item head logits already dumped by R10-COMBO
(`logging/runs/r10_combo/{zh,hm}/logs/*.scores.jsonl`), arms `A0` and `CAT`, MHC-ZH seeds
600–629 and HateMM seeds 600–614, each seed read at its own P1 epoch selected from its own dev
curve.

**Belt D1 (gating on the audit's own validity).** macro-F1 recomputed from the dumped logits must
agree with the trainlog-logged macro-F1 to 1e-4 for every run used; otherwise halt.

Procedure:
1. Per item, per arm: the fraction of seeds that misclassify it at threshold 0.5.
2. **FIXED** = items misclassified by `A0` in ≥ 2/3 of seeds and by `CAT` in ≤ 1/3.
   **BROKEN** = the mirror image. (Thresholds fixed here, before looking.)
3. Rank each set by the seed-fraction gap; take the top ~10 of each on MHC-ZH.
4. For those items, read the transcript (`data/gt/MHC_zh/test.jsonl` field `text`) and the OCR
   record (`data/OCR/MHC_zh_test/ocr_video.jsonl`), plus the mechanical per-item read-out audit
   §13.6 item 3 asks for: token count, `v_end`, `hdr`, `TXT` span length, special-token
   composition of the 3-token `A0` span, and the feature norms — computed from the Leg A `-cc`
   extraction (which stores the per-item span statistics).
5. Report the transcript-length distribution of FIXED vs BROKEN vs the rest of the test split,
   and the empty-transcript rate in each, as descriptive context.
6. Write one paragraph, in plain language, on where the gain comes from. It is explicitly labelled
   a description, not evidence.

---

## 6. Leg E — disclosure text (§13.6 item 6)

Not an experiment. The result document must contain, verbatim in substance:

> Roughly 90 candidate configurations were evaluated against these official test splits before
> `CAT` was selected as the surviving entry. The paired-bootstrap intervals reported for `CAT`
> are **conditional descriptive intervals, not post-selection-valid confirmatory intervals**;
> they do not account for the selection. Leg C (repeated stratified CV on train+dev) is offered
> as mitigation of split-specific luck only — **it does not correct adaptive reuse of the test
> split**, because the candidate that is being cross-validated was itself chosen with knowledge
> of test-split outcomes. No uncontaminated confirmatory population exists for this result on
> this workstation.

---

## 7. Execution

Single background driver `idea-stage/cat_closeout/run_all.sh`, `nohup`, log
`logging/runs/cat_closeout/run.log`, pid `logging/runs/cat_closeout/run.pid`, periodic parseable
progress lines. Order: Leg A extraction → Leg A grid → Leg B restore+extraction → Leg B grid →
Leg C build+grid → all analyzers → Leg D (CPU). A leg that halts on a belt stops that leg only;
subsequent independent legs still run, and the halt is reported.

Analyzer `idea-stage/cat_closeout/analyze_cc.py`: computes dev/test macro-F1 **exactly from the
dumped per-item logits** (no confusion-matrix reconstruction, no hard-coded split sizes), selects
P1 from the dev curve, and cross-checks every run against the trainlog-logged macro-F1 at 1e-4
(**Belt E1**, gating). It applies the frozen rules of §2.4, §3.3 and §4.3 mechanically and writes
its verdicts to JSON before any prose is written.

Every leg's raw JSON, the verdict JSON, the logs and the build metadata are listed in the result
document `idea-stage/CAT_CLOSEOUT_RESULT.md`.

## 8. What this close-out cannot produce

Fixed here so it cannot drift: **no outcome of any leg upgrades `CAT` to a publishable method
contribution.** A clean sweep (A reproduces, B transports, C supports) licenses exactly one
sentence — that the effect is reproducible from raw inputs, extends to a third dataset, and is
not an artefact of one split — and it leaves §13.6's defensible final statement and the
method-paper-only verdict of §13.5 unchanged. Misses narrow the coverage statement further.
