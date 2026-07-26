# CLAP general-audio channel — G0-cond gate RECORD (F90)

**Date:** 2026-07-27 NZST. **Executor:** CLAP lane executor (CPU/SLURM, conda `HateVideo`).
**Binding design:** `refine-logs/CLAP_GATE_SPEC_2026-07-27.md` (commit `6c8929d`), followed VERBATIM.
**Status of sections 0-2 (this header):** **PRE-DECLARATION + PRE-GATE VERIFICATION — committed
BEFORE any CLAP feature existed and before any gate number was computed.** Extraction results
(§3) and gate numbers (§4-6) are appended AFTER this header is committed. Full raw-only
transcription discipline: every number in §4-6 is copied verbatim from
`refine-logs/CLAP_G0COND_GATE_OUT.json`.

The axis: **CLAP (`laion/larger_clap_general`) general-audio SEMANTIC embeddings as a new input
stream**, targeting the ERRPAT **FN1** cluster ("speech-poor visual hate", 7 of 27 stable HateMM
errors, ceiling **+0.0326 acc**). CLAP is the designated **closer** for the learned-audio axis that
both prior audio kills named explicitly.

---

## 0. Pre-declared kill-switches (FROZEN at `6c8929d`, no adjustment after seeing numbers)

**Machinery** (VERBATIM `c3_fusion_probe.py` → `apx_g0cond_gate.py` → `laud_g0cond_gate.py` fork):
`Z` standardized ALONE at its Z-only inner-CV-optimal `C_Z`; aux block standardized × `s=50`
(effectively un-penalized), refit at `C_Z`; aux via train-fold PCA (leak-free); decision family
k ∈ {8,16} (+ {32,64} context); 5×5 `RepeatedStratifiedKFold` `rs=1000+rep`; per-video-clustered
bootstrap **B=5000** on Δacc (`BOOT_SEED=20260714`); shuffled-aux control (`seed=12345`);
permutation null over **≥150** fresh permutations, computed **only** to confirm a would-be pass.
Binding point = best of {k8, k16}. **Dev-side only** (train∪val), **single gate read**.

- **K-CLAP-1 (calibration):** label-oracle `accZA < 0.99` → **MACHINERY_INVALID**.
- **K-CLAP-0 (kill / promote — K9 house standard, bit-consistent with APX/LAUD):** on the **`proj`**
  block, best-decision-k point **Δacc < +0.040** OR bootstrap **CI-lower ≤ 0** OR (would-be pass
  only) **not > all ≥150 perm maxima** → **KILL**. **A PASS must clear BOTH `Z` arms.**
- **K-CLAP-2 (honest-partial):** `+0.030 ≤ Δacc < +0.040` with CI-lower > 0 on both arms →
  documented near-miss, **NOT** an auto-promote.
- **K-CLAP-3 (FN1 stratum read):** train∪val items with **≤25 transcript words** (the identical FN1
  rule and word-count function as `errpat_hatemm_clusters.py:131`), **N=232, 42 hate, base rate
  0.1810**. AUC-based. **NARROW-GO** requires all three: (C1) AUC ≥ 0.65 with boot CI-low > 0.55;
  (C2) CLAP stratum AUC − Whisper stratum AUC ≥ **+0.05**; (C3) conditional ΔAUC over `Z_deployed`
  within the stratum > 0 with boot CI-low > 0. **KILL-side** iff AUC ≤ 0.60 OR CI-low ≤ 0.50.
  Anything between = **INCONCLUSIVE-NARROW**, treated as a KILL for action purposes.
- **`hidden` block role (§4.4 of the spec), fixed before any number:** the pre-projection block
  **cannot produce a PASS**; a lone pass there is recorded as **DISCORDANT** — an escalation
  *request* to the main loop, not a promote and not a prereg trigger.

**Z arms** (HateMM only, N=851 train∪val, 341 pos): `deployed_7168` = LoRA-curric Qwen img⊕text;
`strict_8960` = CLIP img⊕text ⊕ frozen-Qwen img⊕text (the exact W2-A/APX/LAUD `Z_best`).

**Honest prior: LOW ~10-15%.** The audio axis is **0-for-2**: F41/APX killed classical eGeMAPS-88-d
prosody (best-k **−0.0038**, full-88-d **+0.0005**, calib 1.0000, HateMM); F64/LAUD killed the
learned Whisper-large-v3 encoder on all 3 datasets and both Z arms (global max over 6 cells ×
{k8,k16} = **+0.0041**, every CI straddling 0, calib 1.0000 everywhere). Both share the **F31
hazard**: the whisper-large-v3 *transcript* already banks spoken hate into the deployed `text_feats`.
CLAP is nonetheless a genuinely distinct measurement — it is trained by contrastive alignment of
audio to free-text descriptions of **general sound events**, so it is organised by *what the audio
is* rather than by prosodic scalars or by phonetic content for transcription. Both prior records
scope their own kills narrowly and name this successor: APX §5 concedes eGeMAPS "only **weakly**
lower-bounds a learned audio encoder"; the audio recon §0.2 (binding on LAUD) states a Whisper null
"**must not** close the whole learned-audio axis: a **general-audio encoder** is the proper
*closer*". **D7 novelty is thin regardless** — learned audio is catch-up to SOTA inputs (all three
baselines already fuse audio), so even a PASS is a performance/ablation row, never a novelty win.

**Declared tension (spec §4.3):** the global +0.040 bar **exceeds FN1's entire +0.0326 ceiling**. It
is retained anyway for graveyard comparability with F41/F64 — and that is precisely why K-CLAP-3
was pre-declared, with its own strict independent bars, rather than reached for afterwards.

---

## 1. Representation + provenance (FROZEN)

**Model:** `laion/larger_clap_general`, snapshot `ada0c23a36c4e8582805bb38fec3905903f18b41`,
`pytorch_model.bin` **776.4 MB** (total repo 0.780 GB), downloaded 2026-07-27 to the shared HF hub
cache. `projection_dim` **512**, audio tower HTSAT `hidden_size` **1024**, `enable_fusion=False`.
Feature extractor: `sampling_rate` **48000**, `nb_max_samples` **480000** (= 10.0 s), 64 mel bins,
`padding=repeatpad`, `truncation=rand_trunc` (never entered — see §2 D-1).

**Choice justified vs the FN1 content:** FN1 is *defined by the absence of speech* (median 6
transcript words, two members with zero; `hate_video_10`'s entire transcript is `"🎼 In and and.🎼And."`).
`larger_clap_general` carries the **broadest non-speech event vocabulary** (music, chanting, crowd,
screaming, effects) of the LAION releases; `larger_clap_music_and_speech` is narrower on events and
would re-import the speech bias that made Whisper the wrong instrument. **No second CLAP variant
will be tried** — trying another after seeing the first's numbers would be bar-shopping.

**Video vector:** per video, PyAV-decode **48 kHz mono** (`decode_audio_pyav` imported VERBATIM from
`src/utils/generate_segment_asr_HF.py`, sha256 `c52cba69…`; no ffmpeg binary, no torchaudio) → split
into **consecutive non-overlapping 10.0 s windows** (480000 samples, exactly `nb_max_samples`) →
one forward per window → **mean-over-windows ⊕ max-over-windows**. Max is the FN1-critical term: a
single 7 s chant inside a 150 s video survives a max and is washed out by a mean.

| block | object | per-window | pooled | role |
|---|---|---|---|---|
| **`proj`** | L2-normalised projected joint audio-text embedding (== `get_audio_features()`) | 512 | **1024** | **BINDING PRIMARY** |
| `hidden` | HTSAT `pooler_output`, pre-projection | 1024 | **2048** | SECONDARY (cannot PASS) |

Both come from **one** forward: verified this session that
`F.normalize(audio_projection(pooler_output), dim=-1) == get_audio_features(...)` **exactly**
(max abs diff **0.0**), so the single-forward path is bit-identical to the spec's `get_audio_features()`
at half the compute.

| item | value |
|---|---|
| extractor | `scripts/analysis/clap_extract.py` sha256 **`c9dab6c28daf113ec70b59b1baa39efc51d11107d45fe22b06b962b7e1e92b09`** |
| extraction sbatch | `scripts/slurm/clap_extract.sbatch` sha256 **`a4324f50b42e4b391a1aa962b772b2df2621fbf4074501f44c2c4f76ac7e7a80`** |
| gate | `scripts/analysis/clap_g0cond_gate.py` sha256 **`956e3fb0daa7b157e84b4bc930ee99c5e5e66741032af5af0da27baca75b9a92`** |
| machinery source (fork target) | `scripts/analysis/laud_g0cond_gate.py` sha256 `b601013a13727973536c16e199276ae30c2d475c423fe52e090d006668e60594` |
| machinery source (verbatim template) | `scripts/analysis/c3_fusion_probe.py` sha256 `9091e2c3443d4826144f820217e37d43d26d282d334b0b35bea7cb4ae9748b3c` |
| extraction job | SLURM **13647**, **CPU-only** (no `--gres`), 8 CPU / 48 G, no `--time` |

**Splits.** `train`+`val` → `data/audio/HateMM/clap_larger_clap_general_trainval.pt` (**the only
file the gate opens**); `test` → `..._test.pt` (written, then untouched by this lane).
**Declared deviation from the APX/LAUD wording:** extracting *inputs* for the test split is not a
test read (no test label loaded, no test-set metric computed anywhere, gate hard-codes `_trainval`),
and it is how every other feature cache in this repo is built. This record therefore makes the
weaker, true claim and does **not** copy LAUD's "the test split was never enumerated" sentence.

---

## 2. Pre-gate verification (all completed BEFORE any CLAP gate number existed)

Three checks, all committed before the gate ran. None of them can move a bar (the bars were frozen
and committed at `6c8929d`); two of them use no CLAP data at all.

### 2.1 Extraction smoke gate — **PASS** (`refine-logs/CLAP_SMOKE_OUT.json`, 20 videos, no labels read)

| check | result |
|---|---|
| **D-1 determinism** | **True on 20/20** — re-encoding a video gives bit-identical `proj` and `hidden`. `is_longer` is **all-False** for exact-10 s windows, so the **nondeterministic `rand_trunc` path is never entered**. The spec's determinism assumption is thus *measured*, not assumed. |
| **D-2 sanity** | **True on 20/20** — no NaN, no all-zero row, `n_windows == ceil(duration/10 s)` exactly (durations 13.0 s → 684.9 s, 2 → 69 windows). |
| **D-3 discriminative** | **True** — `proj` off-diagonal cosine min **0.1828** / mean **0.5455** / max **0.8512**: embeddings are well spread, the feature-extractor contract is intact. |

### 2.2 Machinery parity vs the published F64 — **PASS, bit-exact at 4 dp on both Z arms**

(`scripts/analysis/clap_machinery_parity.py`, `refine-logs/CLAP_MACHINERY_PARITY_OUT.json`.) The
forked gate is fed the already-banked Whisper-large-v3 block and must reproduce
`LAUD_GATE_RECORD.md` §3:

| Z arm | accZ (pub) | best-k (pub) | Δacc (pub) | CI (pub) | full (pub) | parity |
|---|---|---|---|---|---|---|
| deployed_7168 | 0.8712 (0.8712) | k16 (k16) | **+0.0014** (+0.0014) | [−0.0075,+0.0103] (same) | +0.0063 (+0.0063) | **PASS** |
| strict_8960 | 0.8383 (0.8383) | k8 (k8) | **+0.0014** (+0.0014) | [−0.0073,+0.0106] (same) | +0.0052 (+0.0052) | **PASS** |

Calibration 1.0000 on both. **This is what makes holding the +0.040 bar meaningful:** the CLAP
number lands directly comparable to F41 (−0.0038) and F64 (+0.0014) because the fork provably
introduced **no** machinery drift.

> **Documented gotcha.** `arm_cor_allk` fits ONE PCA at `n_components = max(ks)` and slices it, and
> sklearn's randomized SVD solver depends on `n_components` — so the k8/k16 estimates are **not**
> invariant to passing `KS_DECISION` (kmax=16) vs `KS_REPORT` (kmax=64). `point_arms()`, like LAUD,
> passes `KS_REPORT`; parity holds only on that path. A first attempt passing `KS_DECISION`
> produced a spurious +0.0019 on the deployed arm. Sensitivity ~5e-4 — two orders of magnitude
> under the bar and immaterial to any verdict, but real, and recorded so it is not later
> misdiagnosed as drift.

### 2.3 K-CLAP-3 comparator + confound pre-check — **the bars are slack, and are NOT being changed**

(`scripts/analysis/clap_stratum_precheck.py`, `refine-logs/CLAP_STRATUM_PRECHECK_OUT.json`. Uses
**only** banked caches + gt; **no CLAP data touched**; run while job 13647 was still in its
`disk_guard` preamble, i.e. before any CLAP feature existed.)

On the FN1 ≤25-word stratum (**n=232, 42 hate, base rate 0.1810**):

| quantity | value |
|---|---|
| **Whisper-alone (the ALREADY-KILLED block)** stratum AUC | **0.8482** CI[0.7844, 0.9053] |
| **`Z_deployed`-alone** stratum AUC | **0.8937** CI[0.8355, 0.9427] |
| `n_words` alone | 0.6610 |
| duration alone | 0.5214 |
| ρ(whisper score, `n_words`) | **+0.4765** (p = 1.5e-14) |
| ρ(whisper score, duration) | +0.0046 (p = 0.944) |

**Load-bearing consequence, documented rather than patched.** The spec's absolute C1 thresholds
(KILL ≤ 0.60 / GO ≥ 0.65) are **slack** in this stratum: 0.65 sits *below* even the `n_words`-alone
covariate baseline (0.6610), and far below what a channel already measured as conditionally null
achieves *marginally* (0.8482). **The bars are deliberately NOT changed** — moving them after
measuring the comparator would be a relaxation, and the KILL-side branch is exactly the case the
spec already routes to INCONCLUSIVE-NARROW ("treated as a KILL for action purposes"). The operative
NARROW-GO conditions are therefore **C2** and **C3**, both of which remain strict and are now
pinned to concrete numbers:

* **C2** requires CLAP stratum AUC **≥ 0.8982** (beat Whisper by ≥ +0.05);
* **C3** requires conditional ΔAUC **> 0 with boot CI-low > 0** over a `Z` baseline of **0.8937**.

**Independent scientific value.** A marginal stratum AUC of **0.8482** from a block whose
*conditional* contribution LAUD measured at **+0.0014** is a crisp, quantified demonstration of the
redundancy mechanism: marginal audio signal in the speech-poor stratum is **real but almost entirely
already carried by `Z`**. This is the sharpest statement of the F31 hazard the project has produced,
and it is the correct lens for reading §4 below.

### 2.4 End-to-end gate dry-run — **PASS** (synthetic stand-in, deleted immediately)

Before the real gate ran, the full gate was executed end-to-end against a **synthetic** stand-in
cache built from slices of the Whisper block (`clap_DRYRUN_trainval.pt`), to exercise every code
path — 4 point cells, both strata, the confound diagnostics, the verdict logic, DISCORDANT
detection and the JSON write. It completed cleanly in **694 s**. **The synthetic cache was deleted
immediately afterwards** and none of its numbers appear anywhere in this record or in
`CLAP_G0COND_GATE_OUT.json`; they were stand-in values with no scientific content. This fixes the
expected real-gate cost at ~12 CPU-minutes.

---

*(§3 extraction results and §4-6 gate numbers are appended below after this header is committed.)*
