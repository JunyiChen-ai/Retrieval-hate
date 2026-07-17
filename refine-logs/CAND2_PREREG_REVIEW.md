# CAND-2 Curriculum LoRA-SFT — INDEPENDENT 0-CONTEXT PRE-REGISTRATION REVIEW

**Reviewer:** independent 0-context pre-registration reviewer (no prior context; zero user interaction; no
job submitted; prereg not modified).
**Date:** 2026-07-18.
**Target:** `refine-logs/CAND2_CURRICULUM_PREREG.md` (commit `76ef0e2`).
**Hash gate (pre-condition):** expected sha256 `e5a689d9ede0a79e89eb041f028228e70cdc821029349387e7ec9fdff939790e`
— **on-disk sha256 MATCHES.** Review proceeded.

## VERDICT: **APPROVED-WITH-NOTES** (notes are non-blocking)

The prereg is hash-integral, single-manipulated-variable, leakage-clean, decidable-from-raw-logs, and
numerically bit-exact against the banked trainlogs it re-derives. Its mining is deterministic and idempotent,
its hyperparameters are baked into a hash-frozen builder, its head code is byte-identical to the banked
controls, and every kill-switch is a pre-declared threshold on raw numbers with no interpretive freedom. The
K-C2-2 tie path is a pre-committed, non-embarrassing "generic LoRA with reshuffled data" verdict. Three minor
transparency/rounding notes (below) do not affect decidability, leakage, or the honesty of any bar and are
therefore non-blocking.

---

## Rationale (one paragraph)

Cand-2 measures whether a confusion-weighted (retrieval-mined) reweighting of the encoder-LoRA-SFT example
distribution *adds over* the generic-LoRA arm on `MHC_zh` (primary) and `HateMM` (hold), each trained on its own
train split only. The design's validity hinges on one property — that the curriculum arm is byte-identical to
the generic arm *except* the multiplicity of each SFT record — and that property holds under audit: the two SFT
YAMLs differ by exactly the dataset pointer and output_dir; the curriculum records fork the exact word-variant
`train.json` the banked generic arms trained on (fork-source sha256 identical to `LORA_HATEMM_PREREG.md`'s pin);
and the Stage-3 head `run_one` block is byte-identical across `enc3seed.sbatch`, `enc3seed_lora_hatemm.sbatch`,
and `enc3seed_lora_curric.sbatch`, so the only manipulated head inputs are `--model` and `--group_name`. The
mining reads only the frozen-Qwen **train** cache with leave-one-out `exclude_self` (no dev/test file is opened
anywhere in the builder), uses own-train gold labels only for weighting (training-allowed), and the deployed
extraction path uses fixed label-free single-video prompts — so gold never enters inference. All comparison
floors/arms re-derive to 4dp from the raw trainlogs with the byte-identical embedded parser; the K-C2-0
mining-validity gate re-computes bit-exact (I re-ran the builder twice: `train_curric.json`, the KC20 JSONs, and
the additive `dataset_info.json` are all unchanged, git-clean). Because the mining is deterministic, the
hyperparameters are frozen inside the hash-pinned builder, and the executor transcribes raw per-seed numbers
with the verdict rendered independently, the motivated-executor attack surface (re-mine until favorable, choose
τ/λ/cap post-hoc, bury a KS-regression, cherry-pick a protocol or an SFT draw) is closed by construction. The
prereg correctly and repeatedly defers novelty (D7) to the user and pre-declares a ~50–60% tie as the most
likely outcome.

---

## CHECK-BY-CHECK

### 1. Hash integrity + data build — **PASS**
- Prereg sha256 matches the expected value (gate).
- Freeze-block A–H all match disk: A `085384f5…`, B `ac1c5962…`, C `c12c2b6b…`, D `6a5abb9e…`, E `00d9e995…`,
  F `c8260dd3…`, G `73307ef2…`, H `c2b99d25…`. KC20 I `38b21db5…`, J `14967d53…` match. Reused machinery matches:
  `gen_embed_lora.sbatch c76bb422…`, `enc3seed.sbatch dbe3fb81…`, ZH frozen cache `135a6e24…`, HateMM frozen
  cache `ba52bc0d…`.
- Row counts: ZH 579, HateMM 743 (== N each). Unique/maxdup: ZH 386/3, HateMM 502/4 — match §1.1.
- **Idempotency: re-ran `build_curriculum_sft_data.py` for both datasets (CPU-only). Printed shas equal F/G
  bit-exact; on-disk F/G/I/J and additive H all unchanged; `git status` clean for every re-run target.** K-C2-0
  numbers reproduced exactly (ZH LOO-err 0.2073 / c-Gini 0.5634 / cov 0.6667 / 2.11×; HateMM 0.1935 / 0.6497 /
  0.6756 / 2.08×; PASS both).

### 2. Single-manipulated-variable — **PASS**
- ZH & HateMM curric-vs-generic YAML `diff` = **exactly 2 lines each** (L18 `dataset:` pointer, L27
  `output_dir:`); `eval_dataset` unchanged. Output_dir is a non-collision necessity, not a training variable.
- `run_one`…`PY` block **byte-identical** across `enc3seed.sbatch` / `enc3seed_lora_hatemm.sbatch` /
  `enc3seed_lora_curric.sbatch` (extracted-block sha256 `286a9e44…` for all three; head config matches §1.4
  verbatim). Full-file diff vs the HateMM head sbatch = header comment, `LORA` tag, `GROUP_NAME`, and the 6
  CONFIGS rows (`MHC_zh`×3 + `HateMM`×3) only. `bash -n` OK on both new sbatch.
- **DEV-1 (load-bearing) verified:** current fork-source `train.json` shas — ZH
  `ecfa663d…31b10d0`, HateMM `93c6d3d1bffbca22b2dd8beba57a33575a48d8ca61d8d56e3148fecdbb93973a` — match prereg
  §1.1, and the HateMM sha is byte-identical to `LORA_HATEMM_PREREG.md:76`'s pin (743 recs, 297 hateful/446
  normal). The curriculum forks the exact records the 13235 generic arm trained on; answer-format is NOT a second
  manipulated variable.

### 3. Comparison-number provenance — **PASS (all 4dp-exact, independently re-derived)**
Re-parsed all 12 banked trainlogs with the byte-identical embedded parser (val-sel = max val-acc epoch≥5,
roc tie-break; final = max epoch):

| leg | protocol | 3-seed mean acc/F1 | Δ vs CLIP acc/F1 |
|---|---|---|---|
| ZH CLIP floor (13115) | val-sel | 0.8076 / 0.7676 | — |
| ZH CLIP floor (13115) | final | 0.8143 / 0.7720 | — |
| ZH generic-LoRA (13150) | val-sel | 0.8322 / 0.8015 | **+0.0246 / +0.0339** (acc FAIL) |
| ZH generic-LoRA (13150) | final | 0.8456 / 0.8173 | **+0.0313 / +0.0453** (per-seed +0.0402/+0.0335/**+0.0201** ⇒ MARGINAL) |
| HateMM CLIP floor (12850) | val-sel | 0.8202 / 0.8085 | — |
| HateMM CLIP floor (12850) | final | 0.8124 / 0.7936 | — |
| HateMM generic-LoRA (13235) | val-sel | 0.8620 / 0.8545 | **+0.0419 / +0.0460** (3/3 PASS) |
| HateMM generic-LoRA (13235) | final | 0.8698 / 0.8618 | **+0.0573 / +0.0682** (3/3 PASS) |

All per-seed values in §2.1/§2.2 reproduce exactly. Noise-band §2.3: largest head-seed acc spread = 0.0140
(HateMM val-sel) ⇒ ±0.014 band confirmed.

### 4. Bars decidability — **PASS**
K-C2-0 (fixed thresholds, already computed + reproduced), K-C2-1 (mean Δacc & ΔmF1 ≥ +0.030, 3/3 sign, AND ≥
generic−0.014), K-C2-2 (mean paired Δacc ≥ +0.010 AND 3/3 positive sign AND mean ΔmF1 ≥ 0), KS-regression
(mean Δacc ≤ −0.014), KS-below-floor, and the ZH-robustness clause (val-sel conjunct passes OR final mean ≥
+0.040 with 3/3 per-seed ≥ +0.030) are each decidable from raw per-seed logs with no interpretive freedom. The
K-C2-2 **tie** outcome is a pre-declared, non-embarrassing verdict ("generic LoRA with reshuffled data; bank the
negative; do not claim the coupling"). Both protocols are judged and reported separately (fixed §7.3 write-up),
so no protocol cherry-pick.

### 5. Leakage & vetoes — **PASS**
- Audited `build_curriculum_sft_data.py` in full: `_load_frozen_cache` opens only `train_{FROZEN_TAG}.pt`;
  `build_curriculum` reads only that cache + the generic `train.json`; **no dev/test file is opened anywhere**;
  LOO uses `keep = idx != i` (`exclude_self`). Vote machinery is byte-identical to
  `cross_channel_router_gate.py:73-78,120-131`.
- Own-train gold labels are used only to compute confusability (training-allowed). The deployed extraction path
  (`generate_VideoMLLM_embedding_lora_HF.py:59-66`) uses **fixed neutral single-video prompts** applied
  identically to every video regardless of label — gold never enters inference.
- Single-dataset own-train-split per arm; no OCR; no cross-dataset mixing; raw videos never leave the machine.

### 6. Honesty clauses — **PASS (all present, pre-declared)**
F0.1 test-not-virgin; F0.2 single-curriculum-draw limitation tied to K-C2-2 validity; F0.3 D7 deferred to user;
F0.4 structural ceiling / ~5% new-dataset prior; F0.7 ~50–60% tie prior (most-likely outcome); F0.8 class-balance
shift disclosed; DEV-4/§3.5 ~33% easy-tail drop tied explicitly to the KS-regression risk.

### 7. Adversarial — **PASS (one minor transparency note, non-blocking)**
- *Re-mine until favorable:* blocked — mining is deterministic (no RNG on the softconf path), idempotent
  (verified), and its hyperparameters are baked into the hash-frozen builder (A); any change voids authorization.
- *τ/λ/cap post-hoc:* pinned pre-GPU, tuned only on $0 train-cache **properties** (~2.1× hard-head mass, ~67%
  coverage), not on any test metric; K-C2-0 independently validates the resulting curriculum; frozen by hash.
  **Note:** the prereg states the sweep gave a "fair contrast" but does not enumerate the swept grid; because the
  values are frozen, train-only-tuned, and reproduced, this is non-blocking.
- *Bury a KS-regression:* blocked — the executor transcribes raw per-seed numbers and applies no gates; the
  verdict is rendered independently against the prereg verbatim, using the byte-identical parser.
- *SFT-draw / protocol cherry-pick:* blocked — single test-touch budget per dataset (F0.1/§6) forbids re-drawing
  SFT; both protocols reported.

### 8. Deviations DEV-1..DEV-7 — all favorable / neutral / documented, sound grounds
- **DEV-1** (yn→word variant) — LOAD-BEARING, **favorable**; fork-source shas verified against the generic arms.
- **DEV-2** (fused img+text vote) — **neutral**; targets the align-fusion head's joint space; machinery verbatim.
- **DEV-3** (deterministic largest-remainder, no RNG) — **favorable**; removes resampling artifact, yields the
  bit-exact idempotency I verified.
- **DEV-4** (τ/λ/cap pinned by $0 CPU sweep) — **neutral**; see §7 note (grid not enumerated; frozen + train-only).
- **DEV-5** (new sbatch vs case-add) — **favorable**; leaves the generic path byte-untouched.
- **DEV-6** (K-C2-2 threshold self-consistent) — **documented**; sound (seed-spread band vs 3-seed-mean gain,
  noise ≈ std/√3 ≈ 0.003; 3/3-sign is the teeth).
- **DEV-7** (both datasets, not branch-gated) — **documented**; grounded: `LORA_HATEMM_VERDICT_REVIEW.md:160`
  = "HateMM: final-epoch: PASS; val-selected: PASS."

---

## NON-BLOCKING NOTES (for the executor / verdict reviewer)
1. **HateMM class-balance figure (§0.8/§1.1).** Prereg states "40.1%→37.7% hateful"; the generic arm's hateful
   fraction is 297/743 = **40.0%** (39.97%, rounds to 40.0 not 40.1). Curric 37.7% (280/743) is exact. A 0.1pt
   rounding slip on a disclosure number; direction/magnitude of the shift are correct. Non-material.
2. **HateMM cache/SFT count (KC20 JSON).** `n_train_cache = 744` vs `n_train_sft = 743`,
   `n_anchor_missing_from_cache = 0`: all 743 SFT anchors are present in the frozen cache; one cache-only train
   video acts solely as a potential LOO neighbor. Train-only, no leakage, predates cand-2. Benign.
3. **DEV-4 sweep provenance** (see §7): frozen + train-only-tuned + K-C2-0-validated ⇒ non-gameable post-freeze,
   but the swept grid is not written into the prereg. Purely a transparency note.

---

## HASH-FREEZE (recorded here; prereg not modified, per review mandate)

```
FROZEN e5a689d9ede0a79e89eb041f028228e70cdc821029349387e7ec9fdff939790e  refine-logs/CAND2_CURRICULUM_PREREG.md (commit 76ef0e2)
A 085384f5534ffae9969c95211f7eaefca5cc3d54278734ba76457b84990f66e8  src/utils/build_curriculum_sft_data.py
B ac1c596293877e827c9db96bec8aefc8f36ebe5e6d3aa95544889be48815fa6d  mhc_zh_qwen25vl_lora_curric_sft.yaml
C c12c2b6b340151e6c58ed39843aa2cf02a728c17a3296637cae41c2a70b6a4a3  hatemm_qwen25vl_lora_curric_sft.yaml
D 6a5abb9e7d7427f7e4e9874ee429eaed4ed269e342cff5b6df14d40e59ffd57a  lora_sft_curric.sbatch
E 00d9e9956549bdf97c6b8913d42d87811f4a2f150e9f459e8b8b86978b306f02  enc3seed_lora_curric.sbatch
F c8260dd3f5a98394c6ef3d7f08e091dad5810e1d22d58db24ac5654d7029bc0d  data/lora_sft/MHC_zh/train_curric.json
G 73307ef2e286eddf4fbe12ef13bb3c750f9105d1291494779c7a3a181c91082b  data/lora_sft/HateMM/train_curric.json
H c2b99d2521b1785a2df8da0fd62b13ea4c0dea086bd783cd724619aec0229fd6  RA-HMD/.../data/dataset_info.json
I 38b21db5909d4affc9f57c3a9286eab0e807b00c6b7a0d7de599d6ca1a0f6f33  refine-logs/CAND2_KC20_MHC_zh.json
J 14967d5313e044a556a8caf365ab4ab00178d51b0ce3fd67d7a6263b4048cf6b  refine-logs/CAND2_KC20_HateMM.json
```
Executor MUST re-run `sha256sum` on the prereg + A–J at submit time; any mismatch = authorization VOID. STEP 1b
must reproduce F/G bit-exact (the generic builder is RNG-free, so STEP 1a will not perturb the fork source).

*Reviewer spent zero GPU/SLURM/Modal. Only pure-CPU verification (re-ran the curriculum builder twice; re-parsed
banked trainlogs). Prereg not modified. Not pushed.*
