# MHC-ZH baseline gap — diagnosis

**Date** 2026-08-17. **Cost** zero API, local GPU, 92 head-level runs (~14 min wall).
**Trigger** `idea-stage/IDEA_REPORT.md` §10.7: the `ro_L28` MHC-ZH baseline reads 0.8014–0.8080
while the recorded MHC-ZH contrast line is 0.7821; the R6-1C report owed "a matched re-check of the
deployed MHC-ZH extraction".

**Test-set discipline.** Test labels were read only for the final macro-F1 readings below. No
threshold, epoch rule, arm or cache was selected on test. `src/` was not modified. All three
read-out protocols compared here were fixed in project documents before this work
(`idea-stage/RGCL_ABLATION_FREEZE.md` §3 for P1b, `idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md`
for P1/P2); none was invented here.

---

## 1. The two pipelines, item by item

| item | contrast line 0.7821 | `ro_` family / R6-1C |
|---|---|---|
| feature cache | `data/CLIP_Embedding/MHC_zh/{split}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` | `..._Qwen2.5-VL-7B-Instruct-LoRA_HF-ro_L28.pt` |
| encoder | Qwen2.5-VL-7B + **curriculum** LoRA adapter `logging/lora/MHC_zh_curric` (job 13237/13239) | Qwen2.5-VL-7B + **generic** LoRA adapter `logging/lora/MHC_zh` (job 13150) |
| extraction script | `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | `src/utils/generate_VideoMLLM_embedding_readout_HF.py`, a declared strict clone of the same file |
| layer | `hidden_states[-1]` = 28 | 28 |
| pooling span | img = prefix mean, text = response (post-last-`<\|im_start\|>`) mean | identical |
| prompts | deployed `IMG_INSTRUCTION` / `TEXT_INSTRUCTION` | byte-identical (pinned as a clobber guard) |
| frames | 8, uniform | 8, uniform |
| text input | frames + title + transcript + fixed analytic instruction | identical |
| dims / rows | 3584 / 3584, 579-78-149 | identical |
| head training config | `scripts/rgcl_ablation_grid.sh`, `--contrast_mode none` (L1) | `idea-stage/r6_confirm/run_confirm.sh` — every other flag byte-identical |
| epoch selection | `argmax_{e≥5} (dev acc, dev roc)` — called **P1b** below | `argmax_{e≥5} dev macro-F1` (**P1**) and last epoch (**P2**) |
| seeds | 3 (0,1,2) | 60 (30–89) |

**Verified by direct tensor comparison** (`data/CLIP_Embedding/MHC_zh`, all three splits):

- `{split}_..-LoRA_HF.pt` vs `{split}_..-LoRA_HF-ro_L28.pt`: `img max|Δ| = 0.0`, `text max|Δ| = 0.0`,
  ids identical. The `ro_L28` cache **is** the deployed generic-LoRA cache, bit for bit. `R6RO-A0`
  is a verbatim copy of it.
- `{split}_..-LoRA_HF.pt` vs `{split}_..-LoRA-curric_HF.pt`: `img max|Δ|` = 0.053 / 0.047 / 0.030 on
  l2-normalised rows — a real but modest difference, and the **only** difference between the two
  pipelines other than protocol and seed count.

**So the difference list reduces to three items: the LoRA adapter, the epoch-selection key, and the
number of seeds.** Everything else (script, layer, span, prompts, frames, inputs, head
hyperparameters) is identical or provably bit-identical.

### 1b. Which MHC-ZH encoder is the deployed one

Every other record in the project treats **generic LoRA (job 13150, tag `-LoRA_HF`)** as the deployed
MHC-ZH encoder: `scripts/slurm/gen_embed_readout.sbatch` hardcodes
`MHC_zh logging/lora/MHC_zh Qwen2.5-VL-7B-Instruct-LoRA_HF`; the bidir family
(`gen_embed_mllm_bidir*.sbatch`) does the same; `refine-logs/C02_A0_RECORD.md` does the same. The
curriculum adapter was evaluated in `refine-logs/CAND2_VERDICT_REVIEW.md` and rendered **TIE / no
novelty on ZH under both protocols** (K-C2-2, ZH val-sel −0.0067 acc sign 1/3, final-ep +0.0067 acc
sign 2/3) — it was never adopted.

`idea-stage/RGCL_ABLATION_RESULT.md` §1 states its LoRA-variant choice was made by local file
existence ("LoRA 变体按各数据集本地实际存在者选取"), and MHC_zh happens to have **both** caches on
disk. It picked `-LoRA-curric_HF`. **That is the bookkeeping defect: the contrast line was measured
on an encoder the project's own verdict declined to adopt.**

---

## 2. Attribution — 60 seeds, paired, same training code

`idea-stage/mhczh_gap/run_gap.sh` re-ran the deployed head on the **curriculum** cache for seeds
30–89 with `run_confirm.sh`'s hyperparameters byte-for-byte. The **plain** arm did not need
re-running: `MHC_zh_PLAIN_s30` reproduces `logging/runs/r6_confirm/logs/MHC_zh_A0_s30.trainlog`
line-for-line on all 60 dev+test epoch lines, so the r6_confirm A0 logs are reused directly.
Analysis `idea-stage/mhczh_gap/analyze_gap.py`, results `idea-stage/mhczh_gap/results.json`.

Absolute test macro-F1, MHC-ZH, 60 seeds (30–89):

| protocol | CURRIC (`-LoRA-curric_HF`) | PLAIN (`-LoRA_HF` = `ro_L28`) | CAT (L28‖L24) | seed std CURRIC / PLAIN |
|---|---|---|---|---|
| P1 (dev macro-F1) | 0.7816 | **0.8014** | **0.8199** | 0.0212 / 0.0141 |
| P2 (last epoch) | 0.8052 | 0.8080 | 0.8194 | 0.0110 / 0.0111 |
| P1b (dev acc, dev roc) — **the key that produced 0.7821** | 0.7851 | 0.7855 | 0.7798 | 0.0215 / 0.0256 |

Paired deltas over the same 60 seeds (paired bootstrap 95 % CI, 20 000 resamples):

| pair | protocol | mean | MC SE | 95 % CI | pos/60 |
|---|---|---|---|---|---|
| PLAIN − CURRIC | P1 | **+0.0198** | 0.0035 | [+0.0128, +0.0263] | 49 |
| PLAIN − CURRIC | P2 | +0.0028 | 0.0019 | [−0.0011, +0.0065] | 36 |
| PLAIN − CURRIC | P1b | +0.0004 | 0.0040 | [−0.0075, +0.0080] | 34 |
| CAT − CURRIC | P1 | +0.0383 | 0.0041 | [+0.0302, +0.0462] | 53 |
| CAT − CURRIC | P2 | +0.0142 | 0.0027 | [+0.0086, +0.0190] | 52 |
| CAT − CURRIC | P1b | −0.0054 | 0.0050 | [−0.0151, +0.0045] | 25 |

### Decomposition of the 0.7821 → 0.8014 gap (+0.0193)

| step | value | size |
|---|---|---|
| recorded contrast line: CURRIC, P1b, **3 seeds (0,1,2)** | 0.7821 | — |
| same cache, same protocol, **60 seeds** | 0.7851 | +0.0030 seed sampling |
| swap the cache (CURRIC → PLAIN) **under P1b** | 0.7855 | **+0.0004** (CI includes 0) |
| swap the protocol (P1b → P1) **on PLAIN** | 0.8014 | **+0.0159** |

The two factors interact: the cache is worth +0.0004 under P1b and +0.0198 under P1. Neither is
individually "the" cause; the reading only rises when the plain cache is read with a dev-macro-F1
selector. The direction of the interaction is that **the curriculum cache's dev macro-F1 is a worse
epoch selector than its (acc, roc) selector, and the generic cache's is a much better one.**

### The 3-seed instrument cannot see any of this

At the original seeds 0,1,2, run under the deployed protocol P1b, the *plain* cache scores **0.7603**
— 2.2 points *below* the curriculum cache's 0.7821, the opposite sign from the 60-seed result
(+0.0004). With a P1b seed std of 0.022–0.026, a 3-seed mean carries SE ≈ 0.013. This is the same
finding as `idea-stage/IDEA_REPORT.md` §10.6, arriving from a second direction.

### Answer to the question asked

**The gap is not caused by a defective MHC-ZH feature extraction.** The `ro_L28` cache is bit-exact
to the deployed generic-LoRA cache; the extraction is fine. What the diagnosis found instead is
two smaller, real, separable defects:

1. **Wrong encoder on the contrast line.** The MHC-ZH row of the RGCL ablation was run on the
   curriculum adapter, which the project's own CAND2 verdict declined to adopt and which no other
   MHC-ZH pipeline uses. Worth +0.0004 under the ablation's own protocol, +0.0198 under P1.
2. **An epoch-selection key that does not match the reported metric.** `(dev acc, dev roc)` is
   inherited from `scripts/rgcl_ablation_analyze.py::parse_run`; it selects on accuracy and reports
   macro-F1, and on MHC-ZH it is the noisiest of the three protocols (std 0.0215–0.0256 vs 0.0110
   for last-epoch). Worth +0.0159 here, and it was already flagged as a protocol deviation in
   `idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md`.

---

## 3. Repair, and what the new baseline would be

**Repair action.** No re-extraction is required. Two bookkeeping changes:

1. **Retire `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` as the MHC-ZH baseline cache**; use
   `Qwen2.5-VL-7B-Instruct-LoRA_HF` (job 13150, the adapter every other MHC-ZH pipeline merges).
   No GPU cost — the cache already exists and is bit-identical to `ro_L28`/`R6RO-A0`.
2. **Retire `(dev acc, dev roc)` epoch selection** in favour of selecting on the metric that is
   reported. Pre-declare **P1 (dev macro-F1, ties → earliest, epochs ≥ warmup 5)** as the standing
   read-out, with **P2 (last epoch)** as the corroboration. Re-baseline all four datasets under it
   before any further comparison is made.
3. **≥30 seeds for any MHC-ZH number.** 3 seeds has an SE of ~0.008–0.013 on this split.

**New baselines under the recommended protocol** (measured, 60 seeds, seeds 30–89, not extrapolated):

| dataset | line | P1 | P2 |
|---|---|---|---|
| MHC-ZH | A0 = plain-LoRA L28 (new baseline) | **0.8014** | 0.8080 |
| MHC-ZH | + R6-1C L24‖L28 concatenation | **0.8199** | 0.8194 |
| HateMM | A0 = LoRA-curric L28 (unchanged encoder) | 0.8747 | 0.8675 |
| HateMM | + L24‖L28 | 0.8731 | 0.8696 |

**Position vs HVGuard 0.822 on MHC-ZH:** the bare head moves from 0.7821 to **0.8014** (−0.021 vs
HVGuard, down from −0.040); adding the confirmed layer concatenation reaches **0.8199** (−0.002).
That is a statistical tie with HVGuard, not a win, and it is a bare frozen head plus a free
extraction-time detail — worth banking as the new contrast line, not as a claim.

**Caveat that must travel with these numbers.** Under P1b the ordering reverses: CAT 0.7798 <
A0 0.7855 (CAT − A0 = −0.0054, CI [−0.0151, +0.0045]). The R6-1C gain is real under the two
protocols its freeze specified and absent under the legacy one. Choosing P1 *after* seeing all three
protocols is test-informed selection; the honest form of recommendation 2 is that P1/P2 are
defensible on their construction (select on what you report / do not select at all) and P1b is not,
and that the choice is pre-declared **for future runs** rather than used to pick the largest number
from this table.

---

## 4. The other three datasets

| dataset | deployed cache | re-extracted (`ro_`) cache? | comparable numbers | verdict |
|---|---|---|---|---|
| HateMM | `-LoRA-curric_HF` (job 13241 adapter) | yes, `-LoRA-curric_HF-ro_*`, and `ro_L28` is bit-exact to the deployed cache (`refine-logs/READOUT_SUBMIT_RECORD.md` §"R0 FULL-CACHE bit-exact", 3 splits × 2 streams) | contrast line 0.8774 (3 seeds, P1b) vs 60-seed same-cache P1b **0.8734**, P1 0.8747, P2 0.8675 | **consistent.** Same cache, same encoder. The −0.0040 is pure seed sampling (3-seed SE ≈ 0.0039). No gap, no action. |
| MHC (EN) | frozen `Qwen2.5-VL-7B-Instruct_HF` for the 0.7331 contrast line; only `-LoRA_HF` LoRA variant exists | **no** `ro_` cache | none | nothing to compare; encoder choice is unambiguous (only one LoRA variant on disk, and frozen Qwen beat it 0.7331 vs 0.7133) |
| ImpliHateVid | CLIP `openai_clip-vit-large-patch14-336_HF` (0.9118); **no LoRA cache exists at all** | **no** `ro_` cache | none | nothing to compare |

The dual-cache ambiguity that produced the MHC-ZH defect exists **only** on MHC-ZH: it is the only
dataset carrying both a `-LoRA_HF` and a `-LoRA-curric_HF` cache.

---

## 5. `-degenfix1` propagation to the `ro_` family — disposition

**Census (this session, all splits, exact zero-norm rows):**

| cache family | HateMM train | HateMM dev/test | MHC-ZH any split |
|---|---|---|---|
| `Qwen2.5-VL-7B-Instruct_HF` (frozen) | `hate_video_95` (img **and** text) | none | none |
| `-LoRA_HF` | `hate_video_95` | none | none |
| `-LoRA-curric_HF` | `hate_video_95` | none | none |
| `-LoRA-curric_HF-ro_{L28,L24,ow_L28,ow_L24}` | `hate_video_95` | none | none |
| `-LoRA_HF-ro_{L28,L24,ow_L28,ow_L24}` (MHC-ZH) | — | — | none |
| CLIP `openai_clip-...-336_HF` | `hate_video_95` (img only) | none | none |
| CLIP `...-degenfix1` | none | none | none |

Three corrections to the framing in `idea-stage/IDEA_REPORT.md` §10.4:

1. The all-zero row is **not specific to the `ro_` family**. It is present in *every* Qwen-family
   HateMM train cache including the deployed one, because the extractors share the same
   zero-fill-on-decode-failure path. Repairing "the `ro_` family" alone would make the `ro_` caches
   disagree with the deployed cache they are pinned bit-exact to.
2. It is **train-split only** (row 355 of 744). Dev and test are clean, so no reported test metric
   contains it directly.
3. **MHC-ZH has no all-zero row in any cache**, so this is unrelated to the gap in §2.

**Blast-radius measurement, this session.** `idea-stage/mhczh_gap/build_zi.py` built `R6RO-A0ZI` =
`R6RO-A0` with that one train row replaced by the l2-normalised mean of the other 743 train rows;
dev/test copied verbatim. This is not the repair — it bounds how much *any* non-degenerate value in
that slot can move the head. 30 seeds (30–59), `run_confirm.sh` hyperparameters:

Seeds 30–59, so the A0 column is a 30-seed subset of the 60-seed HateMM values in §3.

| protocol | A0 (zero row as banked) | A0ZI (row imputed) | paired Δ | MC SE | 95 % CI | pos/30 |
|---|---|---|---|---|---|---|
| P1 | 0.8754 | 0.8722 | −0.0031 | 0.0024 | [−0.0071, +0.0020] | 7/30 |
| P2 | 0.8694 | 0.8650 | −0.0044 | 0.0024 | [−0.0093, −0.0000] | 10/30 |
| P1b | 0.8734 | 0.8701 | −0.0034 | 0.0018 | [−0.0066, +0.0002] | 10/30 |

Putting a plausible non-degenerate vector in that slot moves HateMM test macro-F1 by **−0.003 to
−0.004**, i.e. the slot is worth well under half a seed-std (0.0057–0.0123) and the sign is
*negative*: the banked zero row is not costing anything recoverable. The imputed vector is not the
true feature, so the correct reading is a magnitude bound — |Δ| ≲ 0.004 in either direction — not a
prediction of what the real repair would give.

This is consistent with the CLIP-side measurement in `refine-logs/DEGEN_FEATURE_FIX_2026-08-09.md`
§4 (paired POST−PRE = −0.0033, mixed sign, smaller than the seed std).

**Disposition: do not propagate now.** Reasons, in order:

1. **Blocked.** The repair requires re-running `generate_VideoMLLM_embedding_lora_HF.py` with a
   prefix-tolerant decode, which needs the merged LoRA adapter. `logging/lora/` **does not exist on
   this workstation** — neither `MHC_zh` nor `HateMM_curric` is present. The raw video
   (`~/data/HateMM/video/hate_video_95.mp4`) is present, so only the adapters are missing.
2. **Below noise.** Measured effect above, plus the CLIP-side −0.0033 (mixed sign).
3. **Scope.** If repaired at all it must be repaired across all six HateMM Qwen-family caches at
   once (frozen, `-LoRA_HF`, `-LoRA-curric_HF`, and the four `ro_*`), otherwise the `ro_L28`
   bit-exactness guard against the deployed cache breaks. That is ~6 extractor passes, not one.
4. `generate_VideoMLLM_embedding_lora_HF.py` has no id-subsetting flag (open item 3 of the
   2026-08-09 record), so a merge script analogous to `scripts/analysis/degen_feat_fix.py` would
   have to be written first.

**Recorded fix recipe, for when the adapters are recovered:** decode `hate_video_95.mp4` with the
prefix-tolerant PyAV path from `scripts/analysis/degen_feat_fix.py::decode_prefix` (12 346 of 13 461
declared frames are decodable), sample the uniform 8-frame grid over the decodable prefix, run the
unchanged `_encode()` for both spans under each adapter, and merge into a new `-degenfix2` cache
family — never overwriting the banked caches, and rebuilding the `ro_*` and `R6RO-*` derivatives
from the same repaired source in the same pass.

---

## 6. Artifacts

| what | path |
|---|---|
| this document | `idea-stage/MHCZH_GAP_RECON.md` |
| gap grid driver (59 CURRIC + 3 PLAIN + 1 smoke pair) | `idea-stage/mhczh_gap/run_gap.sh` |
| analysis / decision-free read-out | `idea-stage/mhczh_gap/analyze_gap.py` |
| numbers | `idea-stage/mhczh_gap/results.json` |
| zero-row sensitivity cache builder | `idea-stage/mhczh_gap/build_zi.py` |
| trainlogs | `logging/runs/mhczh_gap/logs/`, `logging/runs/zi_probe/logs/` |
| reused, not re-run | `logging/runs/r6_confirm/logs/MHC_zh_{A0,CAT}_s{30..89}.trainlog`, `logging/runs/rgcl_ablation/logs/LORA_MHC_zh_L1_s{0,1,2}.trainlog` |
