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

---

## 3. Stage A — extraction results (job 13647)

SLURM **13647** (`clap_extract`) **COMPLETED `0:0`**, elapsed **01:39:07** wall, **CPU-only**
(`AllocTRES=cpu=8,mem=48G,node=1,billing=8` — **no `gres/gpu` component**, verified), MaxRSS 3.10 G.
Log `slurm/logs/clap_extract_13647.log`.

> **Wall-time note.** Only ~66 min of the 1:39:07 was extraction. The first ~32 min were the standard
> `disk_guard` sbatch preamble, which found `/data` at 287 G against its 250 G target and pruned 169
> checkpoints (almost all another lane's `RAC_video_moka_umfloor` MHC_zh runs) to B2 at ~8 s each.
> Several of its B2 pushes failed (`ERROR: push failed; NOT deleting local. Skipping.`) — it
> correctly refused to delete anything it had not verified as uploaded. Extraction itself ran at
> **~3.7 s/video**. This preamble tax is paid by **every** sbatch until the footprint is under 250 G.

| group | N (exp) | n_pos | proj | hidden | n_zero | n_nan | status | windows (min/med/max/total) |
|---|---|---|---|---|---|---|---|---|
| **trainval** | **851** (851 ✓) | 341 | (851, 1024) f32 | (851, 2048) | 0 | 0 | `ok:851` | 1 / 11 / 581 / 13052 |
| **test** | **215** (215 ✓) | 86 | (215, 1024) f32 | (215, 2048) | 0 | 0 | `ok:215` | 1 / 12 / 100 / 3034 |

Audio coverage **100%** — no `no_audio` row, no decode failure, no NaN, no all-zero vector anywhere,
exactly as the F64 audio census predicted.

### 3.1 Cache verification — **ALL_CHECKS_PASS = True**

(`scripts/analysis/clap_cache_verify.py`, `refine-logs/CLAP_CACHE_VERIFY_OUT.json`.)

| cache | md5 | sha256 |
|---|---|---|
| `clap_larger_clap_general_trainval.pt` | **`13b0b289f6f78d5a5b4ad92a02e41ca5`** | `c0e8e14386674ac4f5d241f85bb5948729ac4ced424ccbe753492ad379e33f02` |
| `clap_larger_clap_general_test.pt` | **`3d7d3e6df6cd827a333ef5b0a03bb9ad`** | `fa6025d4ae9f9a022e3eec2f21d36e08220fb9e684d6cdfdfe7c9a4b8761b6a9` |

Row-count sanity and every structural check passed with **zero** failures: id order == gt
`train.jsonl` ⊕ `val.jsonl` (and == gt `test.jsonl` for the test cache); N == 851 / 215 as expected;
labels agree with gt **and** with all three baseline feature caches (CLIP, frozen-Qwen,
LoRA-curric); dims 1024 / 2048 as declared; **trainval ∩ test = ∅ (overlap 0)**; and the CLAP id
order is **identical** to the banked Whisper cache, which is what makes the K-CLAP-3 head-to-head
legitimate.

---

## 4. Stage B — gate results (single read; verbatim from `CLAP_G0COND_GATE_OUT.json`)

Gate on local login-node CPU, elapsed **681 s**. `CLAP_G0COND_GATE_OUT.json` sha256
**`0a3f8629045f2f8d383922056c6b72f0e74c69a2edafdc1a059d1b60926f156b`**; run log
`refine-logs/CLAP_G0COND_GATE_run.log`. **K-CLAP-1 VALID in all 4 cells** (label-oracle
`accZA = 1.0000`, headroom-fraction 1.000 everywhere → genuine nulls, not machinery artifacts).
**No perm-null computed on any cell** — correct per spec: it runs only to confirm a would-be pass,
and no arm cleared C1 & C2, so C3 is `None` throughout.

### 4.1 `proj` — **BINDING PRIMARY** (1024-d, N=851, 341 pos)

| Z arm | accZ | C_Z | k8 | k16 | k32 | k64 | full_cvC | shuffled k8/k16 | **best-k Δacc** | K-CLAP-0 |
|---|---|---|---|---|---|---|---|---|---|---|
| deployed_7168 | 0.8712 | 1.0 | **−0.0009** [−0.0075,+0.0056] | −0.0033 [−0.0108,+0.0040] | −0.0063 | −0.0157 [−0.0284,−0.0026] | +0.0089 [+0.0007,+0.0174] | −0.0157 / −0.0129 | **−0.0009 (k8)** | **KILL** |
| strict_8960 | 0.8383 | 0.01 | **−0.0038** [−0.0118,+0.0045] | −0.0113 [−0.0209,−0.0019] | −0.0132 | −0.0193 [−0.0348,−0.0045] | +0.0026 [−0.0059,+0.0110] | −0.0035 / −0.0096 | **−0.0038 (k8)** | **KILL** |

### 4.2 `hidden` — SECONDARY, cannot PASS (2048-d)

| Z arm | accZ | k8 | k16 | k32 | k64 | full_cvC | **best-k Δacc** | K-CLAP-0 |
|---|---|---|---|---|---|---|---|---|
| deployed_7168 | 0.8712 | +0.0007 [−0.0049,+0.0061] | **+0.0009** [−0.0078,+0.0096] | −0.0028 | −0.0169 [−0.0291,−0.0045] | +0.0059 [−0.0031,+0.0148] | **+0.0009 (k16)** | **KILL** |
| strict_8960 | 0.8383 | **−0.0005** [−0.0082,+0.0075] | −0.0009 [−0.0118,+0.0103] | −0.0078 | −0.0188 [−0.0334,−0.0038] | +0.0033 [−0.0063,+0.0129] | **−0.0005 (k8)** | **KILL** |

**Structural read.** Every decision-family point estimate is **at or below zero** — the global max
over all 4 cells × {k8,k16} is **+0.0009**, an order of magnitude under the +0.040 bar and *below
even F64's +0.0041*. Every decision CI straddles 0. No cell reaches the +0.030 honest-partial band.
The context arms degrade **monotonically** with k (k64 down to −0.0193), the classic
**pure-redundancy dilution** signature: added CLAP dimensions are noise the head cannot zero out.
Calibration at full headroom (1.0000) proves the machinery *can* convert real information, so these
are genuine nulls. **`DISCORDANT = False`** — the secondary block agrees with the primary.

> **The one arm with a positive CI, reported because it is the most favourable reading available.**
> `proj | deployed_7168 | full_cvC` = **+0.0089, CI [+0.0007, +0.0174]** — the only arm anywhere
> whose CI excludes zero on the positive side. It is **not** the decision family (the binding point
> is best-of-{k8,k16} PCA, per the frozen machinery, identically to F41/F64), and at +0.0089 it is
> **4.5× under the +0.040 bar** and under even the +0.030 honest-partial floor. It is recorded here
> so the record cannot be accused of hiding the single non-negative signal; it changes no verdict.

### 4.3 K-CLAP-3 — the FN1-targeted stratum read

**Primary stratum, ≤25 words (the FN1 rule): n=232, 42 hate, base rate 0.1810.**

| leg | quantity | value |
|---|---|---|
| **(a) marginal** | CLAP-alone AUC | **0.8411** CI[0.7640, 0.9073] |
| **(b) head-to-head** | Whisper-alone AUC | 0.8482 CI[0.7844, 0.9053] |
| | **CLAP − Whisper** | **−0.0071** CI[−0.0701, +0.0524] |
| **(c) conditional** | `Z_deployed` alone AUC | 0.8937 |
| | best-k16 ΔAUC over Z | **+0.0113** CI[**−0.0283**, +0.0533] |
| | (k8 ΔAUC) | +0.0064 CI[−0.0176, +0.0322] |
| confound | ρ(CLAP score, `n_words`) | **+0.3729** (p = 4.6e-09) |

| condition | requirement | observed | fires? |
|---|---|---|---|
| **C1** | AUC ≥ 0.65 **and** CI-low > 0.55 | 0.8411, CI-low 0.7640 | ✔ **true** (but slack — see §2.3) |
| **C2** | CLAP − Whisper ≥ **+0.05** | **−0.0071** | ✗ **false** (CLAP is *behind* the killed channel) |
| **C3** | ΔAUC > 0 **and** CI-low > 0 | +0.0113, **CI-low −0.0283** | ✗ **false** |

**K-CLAP-3 verdict = `INCONCLUSIVE_NARROW`** → per the frozen spec, **treated as a KILL for action
purposes** (no GPU, no prereg). C1 passing is exactly the slack the §2.3 pre-check predicted in
advance; the two conditions that carry real information both fail. **C2 is the decisive one:** CLAP
does not merely fail to clear its bar, it scores **−0.0071 *below* the Whisper block we already
killed**, on the very stratum built to favour it.

**Context stratum, ≤1 word (empty-transcript analogue): n=87, 8 hate, base rate 0.0920. Declared
UNDERPOWERED CONTEXT ONLY in advance — renders no verdict.**

| leg | value |
|---|---|
| CLAP-alone AUC | 0.8180 CI[0.6781, 0.9354] |
| Whisper-alone AUC | 0.6915 CI[0.4060, 0.9403] |
| **CLAP − Whisper** | **+0.1266** CI[**−0.1169, +0.4300**] |
| `Z` alone AUC | 0.8655; best-k16 ΔAUC **+0.0585** CI[−0.0572, +0.1830] |

**This is the one place CLAP beats Whisper, and it is exactly where the FN1 mechanism predicted it
would** — on zero-speech items, where an ASR encoder has nothing to encode and a general-audio
encoder should shine. It is also exactly where the data cannot resolve it: **8 positives**, and a
CI running from −0.12 to +0.43. Honest reading: **suggestive, unresolvable, and pre-declared as
non-decisional before any number was seen.** It is reported because it is the only positive trace
of the hypothesised mechanism, and it is *not* promoted because 8 positives cannot carry a verdict —
this is precisely why the ≤25-word stratum (42 positives) was pre-registered as the binding unit.

### 4.4 Global marginal diagnostic (spec §5.1; moves no bar)

| quantity | value |
|---|---|
| CLAP-alone global AUC (N=851) | 0.8497 CI[0.8228, 0.8752] |
| Whisper-alone global AUC | 0.8954 CI[0.8734, 0.9164] |
| **CLAP − Whisper (global)** | **−0.0458** |
| ρ(CLAP score, `n_words`) | +0.4430 (p = 3.2e-42) |

CLAP is a **weaker** marginal predictor than Whisper globally, and its score is **strongly
length-correlated** (ρ = +0.44) — i.e. a substantial part of what CLAP "knows" about hatefulness is
the same transcript-length prior that ERRPAT §4.3 identified as the *bias* driving FN1 in the first
place, not new acoustic evidence.

---

## 5. Verdict — **KILL** (F90). The general-audio axis is closed.

**`promote_head_gpu = False`. `narrow_prereg_draft_authorized = False`.**

The CLAP general-audio stream carries **no conditional label information** over the deployed encoder
(7168-d) **or** the strict W2-A/APX/LAUD `Z_best` (8960-d) on HateMM, in **either** the projected
joint-space block or the pre-projection block. K-CLAP-0 fires on all four cells; K-CLAP-3 returns
INCONCLUSIVE-NARROW with both informative conditions failing.

### 5.1 The graveyard, now three deep — and the numbers are directly comparable

Comparability is **earned, not assumed**: §2.2 proved the forked machinery reproduces F64's
published numbers bit-exactly at 4 dp on both Z arms.

| id | representation level | binding point (deployed / strict) | verdict |
|---|---|---|---|
| **F41 / APX** | classical prosody — eGeMAPSv02, 88-d hand-crafted functionals | — / **−0.0038** (full-88d +0.0005) | KILL |
| **F64 / LAUD** | learned **speech-ASR** encoder — Whisper-large-v3 hidden, 2560-d | **+0.0014** / **+0.0014** | KILL |
| **F90 / CLAP** | learned **general-audio semantics** — CLAP `proj`, 1024-d | **−0.0009** / **−0.0038** | **KILL** |

**All three representational levels of the audio axis are now measured and null.** The scope caveat
that kept the axis open after F64 — the audio recon §0.2 / LAUD §4 ruling that "a Whisper-encoder
null must **not** close the whole learned-audio axis; a **general-audio encoder** is the proper
*closer*" — **has now been discharged on its own terms.** Per the pre-declared consequence in spec
§4.2: **no training GPU, no prereg, no second CLAP variant, and no AST/BEATs/wav2vec2 escalation.**
The axis is closed.

### 5.2 Mechanism — this is the F31 hazard, quantified more sharply than before

The three §2.3 + §4.3 + §4.4 numbers together give the cleanest statement of the redundancy
mechanism the project has produced:

1. On the speech-poor stratum, audio carries **real marginal signal** — CLAP 0.8411, Whisper 0.8482
   AUC, both far above the `n_words`-alone covariate at 0.6610.
2. That signal is **almost entirely already carried by `Z`**: `Z` alone scores 0.8937 on the same
   items, and adding CLAP moves it by **+0.0113 with a CI spanning zero**.
3. So the audio channel is **redundant, not uninformative** — exactly the F31 hazard. The
   whisper-large-v3 *transcript* already banks the content into the deployed `text_feats`, and what
   remains is a **length prior** (ρ = +0.44 globally, +0.37 within stratum), which ERRPAT §4.3
   already measured as the *bias* that produces FN1 rather than a cure for it.

The FN1 hypothesis — "speech-poor hate videos fail because no channel carries their non-speech
audio" — is therefore **refuted in its actionable form**: the non-speech audio *is* partly legible
to a general-audio encoder, but it duplicates what the multimodal representation already has, and
the increment does not survive conditioning.

### 5.3 D7 novelty status — unchanged and thin

Learned audio was always **catch-up to SOTA inputs** (HateMM 2023 and both MultiHateClip baselines
already fuse audio), so even a PASS would have been a performance/ablation row, never a novelty win.
A KILL costs the project no novelty claim.

---

## 6. Cost ledger and required statements

| stage | resource | cost |
|---|---|---|
| model download | network, 0.78 GB | one-time |
| smoke gate (20 videos) | login-node CPU | 88 s |
| **extraction (1066 videos, all splits)** | **local SLURM, CPU-only, 8 CPU** | **1:39:07 wall (~32 min of it `disk_guard`), 0 GPU-h** |
| comparator/confound pre-check | login-node CPU | 13 s |
| machinery parity vs F64 | login-node CPU | ~250 s |
| end-to-end dry-run | login-node CPU | 694 s |
| **gate (4 cells + 2 strata + diagnostics)** | login-node CPU | **681 s** |
| **TOTAL GPU** | | **ZERO** |

**Required statements.**
- **No performance / accuracy claim** on any held-out benchmark. Every accuracy/AUC number above is
  train∪val cross-validation used **solely** to measure conditional information and audit the probe.
- **Test contact: none.** The gate opened only `clap_larger_clap_general_trainval.pt`; no test label
  was read and **no test-set metric was computed anywhere in this lane**. The test cache was written
  by the extractor and never opened by the gate (`trainval ∩ test = ∅`, verified). Per §1, this
  record makes that weaker-but-true claim and deliberately does **not** copy LAUD's stronger "the
  test split was never enumerated" wording.
- **Raw media never left local.** ffmpeg-free PyAV decode of the local mp4s on-node; only derived
  float caches produced. No Modal; the `modal_probe_runner.py` hard block untouched. This executor
  created/stopped **zero** Modal apps.
- **Single gate read.** All bars frozen at `6c8929d` **before any CLAP weight was downloaded**. No
  bar, threshold, stratum boundary or arm role was adjusted after any number was seen — including
  when the §2.3 pre-check showed C1's absolute thresholds were slack, where the deliberate choice
  was to document the slack rather than relax the bar.
- **Raw-only transcription:** every number above copied verbatim from `CLAP_G0COND_GATE_OUT.json`
  (sha256 `0a3f8629…`) / the manifests; no companion metric fabricated.
- **No `state/` mutation, no prereg written, no training job submitted, no GPU consumed.**
- **Write scope:** `scripts/analysis/clap_{extract,g0cond_gate,stratum_precheck,machinery_parity,cache_verify}.py`,
  `scripts/slurm/clap_extract.sbatch`, `refine-logs/CLAP_{GATE_SPEC_2026-07-27,GATE_RECORD,SMOKE_OUT,
  G0COND_GATE_OUT,G0COND_GATE_run,STRATUM_PRECHECK_OUT,MACHINERY_PARITY_OUT,CACHE_VERIFY_OUT}.*`,
  and derived caches under `data/audio/HateMM/` (gitignored). Local commits only, **not pushed**.
