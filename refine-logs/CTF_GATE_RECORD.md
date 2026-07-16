# CTF G0-cond conditional-information gate — Wave-3 candidate #1 (NON-BINDING $0 pre-check)

**Date:** 2026-07-16. **Executor:** CTF gate executor (CPU-only, conda `HateVideo`, no GPU/SLURM/network).
**Status of this record:** **NON-BINDING prior-mover / cheap-kill screen** (pre-ceremony; precedent = the
W2-C CLIP-K4 pre-check, `ad48dcc`). No prereg freeze was taken; full **raw-only** transcription discipline
applies (every number below copied verbatim from the primary JSON `refine-logs/CTF_G0COND_GATE_OUT.json`).
**Design followed VERBATIM:** `refine-logs/WAVE3_CANDIDATES.md` CANDIDATE 1 sections (d)/(e) (commit `0ee06df`).

---

## 0. Headline

**KILL-side (clean).** On the binding object — the flattened causal-prefix frame-group tensor
`[g_1..g_T]` (14336-d), PCA-reduced — the conditional-information point estimate over
`Z_best = concat(CLIP img+text, Qwen img+text)` (8960-d) is **+0.0000 on HateMM** (best-k CI `[-0.0031,+0.0031]`)
and **−0.0029 on MHC-EN** (CI `[-0.0076,+0.0016]`). Both fail the +0.040 bar (C1) **and** have bootstrap
CI-lower ≤ 0 (C2); K-CTF-1 is an OR of kill conditions, so the gate fires on either alone. Calibration
(K-CTF-2) is **VALID** everywhere (label-oracle accZA = 1.0000, headroom-fraction 1.000) → this is a genuine
null, not a machinery artifact. The arc channel `Δ = g_T − g_1` (1a) is likewise dead (HateMM −0.0049,
MHC −0.0010). **`CTF_SURVIVES = False` — both realizations (1a arc, 1b learned-pool) die at $0, no GPU.**

---

## 1. Provenance

| item | value |
|---|---|
| script | `scripts/analysis/ctf_g0cond_gate.py` sha256 **`948596e665a11f1b6cbbba3f18efc2abe2cb899954d6b0ff573439510ab85c2c`** |
| machinery source (reused VERBATIM) | `scripts/analysis/c3_fusion_probe.py` sha256 **`9091e2c3443d4826144f820217e37d43d26d282d334b0b35bea7cb4ae9748b3c`** |
| repo HEAD at run | `0ee06df` (= the WAVE3_CANDIDATES.md commit; design frozen) |
| invocation | `conda run -n HateVideo python3 scripts/analysis/ctf_g0cond_gate.py` (env `NSEED` unset → default **150**) |
| where run | **LOCAL login-node CPU** (foscsmlprd01), **NOT Modal**; no GPU, no SLURM, no network. Single checkpointed process, exit 0, **elapsed 550 s (~9 min)** |
| env | conda `HateVideo`: sklearn 1.5.2, numpy 1.26.4, torch 2.6.0+cu124 |
| raw outputs | `refine-logs/CTF_G0COND_GATE_OUT.json` (primary) + `refine-logs/CTF_G0COND_GATE_run.log` (console) |

### 1.1 Input cache sha256 cross-check vs the frozen S2S manifest (`cc3d90e`, `S2S_EXTRACTION_RECORD.md` §5)

Only **train + dev_seen** framesets were opened (canonical id order = frameset train ⊕ dev_seen). The two
`test_seen_frameset.pt` files were **never opened** → zero test-touch.

| frameset file | sha256 on disk this run | manifest (cc3d90e §5) | match |
|---|---|---|---|
| `HateMM/frameset_qwen7b_8f/train_frameset.pt` | `10d53f77…04191e2` | `10d53f77487058e7df6015ce96d696ab0dc69d4a2d48c13225895463c04191e2` | ✅ |
| `HateMM/frameset_qwen7b_8f/dev_seen_frameset.pt` | `34912910…ac4213` | `34912910e1cba254f45112f664db7897162dd9c75dd597908c888f2addac4213` | ✅ |
| `MHC/frameset_qwen7b_8f/train_frameset.pt` | `9423a818…0f2c28` | `9423a818bf22e0d7767e58a03a1e9a3995c5da6f4473afd6802f333d470f2c28` | ✅ |
| `MHC/frameset_qwen7b_8f/dev_seen_frameset.pt` | `d9f4c21d…96be3c1` | `d9f4c21d92ddd9ce59f379b8a9a53e288630474d7b09ed36dac511b2f96be3c1` | ✅ |

Baseline `Z_best` pooled caches (features-only, read-only): `data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen}_
{openai_clip-vit-large-patch14-336_HF, Qwen2.5-VL-7B-Instruct_HF}.pt`. Dimension audit confirmed at load:
CLIP img 1024 + CLIP text 768 + Qwen img 3584 + Qwen text 3584 = **8960-d**; ids equal across CLIP=Qwen=frameset
in both splits (asserted per-video, label agreement asserted per-video). N = HateMM 744+107 = **851**
(1 zero-guard `hate_video_95`, kept — its `Z_best` img is the matching banked zero-guard), MHC 549+80 = **629**.

---

## 2. Machinery (reused VERBATIM from `c3_fusion_probe.py`; only the data layer differs)

Identical constants and functions to the C3-template conditional-info probe (`pick_C`, `pick_C_combined`,
`_fit_cor`, `baseline_cor`, `oracle_cor`, `full_cor`, `arm_cor_allk`, `boot_ci`, `_perm_stats`):
Z standardized ALONE at its Z-only inner-CV-optimal `C_Z` (grid {0.001,0.01,0.1,1.0}, `StratifiedKFold rs=0`);
aux block appended standardized × **s=50** (effectively un-penalized, refit at `C_Z` — the REFLECTION §4 fix so
shared L2 cannot crush the aux columns); aux via **train-fold PCA** (leak-free), k sliced from one kmax PCA;
**5×5 RepeatedStratifiedKFold** (rs=1000+rep), per-video correctness averaged; example-clustered (per-video)
**bootstrap B=5000** on Δacc; **bar = +0.040**; mandatory **label-oracle calibration arm** (2-col one-hot gold
× s=50; accZA ≥ 0.99 or MACHINERY_INVALID); permutation null (≥150) available on the pass-branch only.

**Data-layer swap (the only change):** the C3 `A_text` (generated-text 3584-d) is replaced by two aux blocks
derived from the banked frameset `g` ([N,4,3584]), aligned to `Z_best` by id:
- **`flat` = `[g_1..g_T]` flattened = T·3584 = 14336-d** — the **BINDING object** per K-CTF-1.
- **`delta` = `g_T − g_1` = 3584-d** — measured **SEPARATELY** (bears directly on the 1a arc channel).

Arm family per (dataset × source), verbatim: `aux_pca_k{8,16}` (decision family + max-over-k), `aux_pca_k{32,64}`
(context), `aux_full_cvC` (full-dim capacity-matched secondary at combined CV-tuned C), `shuffled` (seed 12345,
continuity). Cells: HateMM (N=851) and MHC-EN (N=629), train∪val — matching design N=851/629.

### 2.1 Interpretation note (single load-bearing reading; not an improvisation)

WAVE3 §(d) frames the measurement as "codelength/MDL, not accuracy," while the **binding** kill-switch §(e)
K-CTF-1 states the threshold in accuracy units — "conditional-info point-estimate … **< +0.040 (projected acc,
bits→acc)**". Reusing "the C3-template conditional-info probe **verbatim**" (§d) operationalizes the quantity as
the C3 machinery's **held-out Δacc = accZA − accZ** with the +0.040 acc bar. That is the only reading under which
"verbatim C3-template" and the acc-unit kill-switch are mutually consistent, so it is the reading used here. No
other design detail required interpretation.

---

## 3. Raw results (verbatim from `CTF_G0COND_GATE_OUT.json`)

**Calibration (K-CTF-2 — machinery validity):** every cell label-oracle **accZA = 1.0000**, headroom-fraction
**1.000**, `PASS = True` → machinery **VALID** on all four cells (the aux-column-crush pathology is absent; a
negative read is admissible).

### HateMM (N=851, n_pos=341, accZ=0.8383)

| source | arm | accZA | Δacc | per-video 95% CI |
|---|---|---|---|---|
| **flat [g_1..g_T]** (14336-d, C_Z=0.01) | aux_pca_k8 | 0.8383 | **+0.0000** | [−0.0031, +0.0031] |
| | aux_pca_k16 | 0.8383 | −0.0000 | [−0.0042, +0.0040] |
| | aux_pca_k32 (ctx) | 0.8357 | −0.0026 | [−0.0085, +0.0033] |
| | aux_pca_k64 (ctx) | 0.8275 | −0.0108 | [−0.0197, −0.0019] |
| | aux_full_cvC (sec) | 0.8228 | −0.0155 | [−0.0261, −0.0052] |
| | shuffled k8/k16 | — | −0.0066 / −0.0047 | — |
| **delta g_T−g_1** (3584-d, C_Z=0.01) | aux_pca_k8 | 0.8324 | −0.0059 | [−0.0141, +0.0019] |
| | aux_pca_k16 | 0.8334 | −0.0049 | [−0.0136, +0.0038] |
| | aux_pca_k32 (ctx) | 0.8266 | −0.0118 | [−0.0221, −0.0019] |
| | aux_pca_k64 (ctx) | 0.8237 | −0.0146 | [−0.0263, −0.0028] |
| | aux_full_cvC (sec) | 0.8353 | −0.0031 | [−0.0122, +0.0056] |
| | shuffled k8/k16 | — | −0.0019 / −0.0078 | — |

`real_max_over_kdec` (best of {k8,k16}): flat **+0.0000**, delta **−0.0049**.

### MHC-EN (N=629, n_pos=193, accZ=0.7971)

| source | arm | accZA | Δacc | per-video 95% CI |
|---|---|---|---|---|
| **flat [g_1..g_T]** (14336-d, C_Z=0.001) | aux_pca_k8 | 0.7943 | −0.0029 | [−0.0076, +0.0016] |
| | aux_pca_k16 | 0.7895 | −0.0076 | [−0.0140, −0.0016] |
| | aux_pca_k32 (ctx) | 0.7825 | −0.0146 | [−0.0242, −0.0057] |
| | aux_pca_k64 (ctx) | 0.7669 | −0.0302 | [−0.0423, −0.0184] |
| | aux_full_cvC (sec) | 0.7787 | −0.0184 | [−0.0315, −0.0051] |
| | shuffled k8/k16 | — | −0.0124 / −0.0118 | — |
| **delta g_T−g_1** (3584-d, C_Z=0.001) | aux_pca_k8 | 0.7949 | −0.0022 | [−0.0095, +0.0051] |
| | aux_pca_k16 | 0.7962 | −0.0010 | [−0.0127, +0.0111] |
| | aux_pca_k32 (ctx) | 0.7921 | −0.0051 | [−0.0188, +0.0079] |
| | aux_pca_k64 (ctx) | 0.7933 | −0.0038 | [−0.0216, +0.0137] |
| | aux_full_cvC (sec) | 0.7952 | −0.0019 | [−0.0118, +0.0079] |
| | shuffled k8/k16 | — | −0.0067 / −0.0178 | — |

`real_max_over_kdec` (best of {k8,k16}): flat **−0.0029**, delta **−0.0010**.

**Structural read.** The decision-family point estimates sit at ≈0 (HateMM flat k8 accZA 0.8383 = baseline
accZ 0.8383 to 4dp — literally zero marginal information from the top PCs of the frameset over `Z_best`), and
the **higher-k and full-dim arms go increasingly NEGATIVE** (HateMM flat full_cvC −0.0155; MHC flat k64 −0.0302).
That monotone dilution — with calibration at full headroom (1.0000) proving the machinery *can* convert real
information — is the signature of pure **redundancy** (the per-group decomposition adds only noise dimensions the
head cannot zero out), not of a capped-but-present signal. This is the same redundancy read F37 recorded for the
frameset, now confirmed against a **supervised, generalizing** operator class (train-fit logistic head) — the exact
cell F37's own oracle had left untested.

---

## 4. Mechanical kill/pass evaluation (design arithmetic quoted verbatim; NON-binding)

Quoted verbatim from `WAVE3_CANDIDATES.md` §(e):

> **K-CTF-1 ($0 gate, BINDING):** conditional-info point-estimate of `[g_1..g_T]` over `Z_best` **< +0.040**
> (projected acc, bits→acc) OR bootstrap CI-lower **≤ 0** OR real ≤ max permutation-null → the per-group
> structure is redundant with the pooled+CLIP config → **CTF DEAD (both realizations), no GPU.**
>
> **K-CTF-2 (calibration, BINDING):** label-oracle arm accZA < ~0.99 → MACHINERY_INVALID, no negative credited.

Evaluated mechanically on the **binding object (`flat`)**; `delta` shown as 1a context. C1 = point ≥ +0.040;
C2 = CI-lower > 0; C3 = real > max permutation-null. K-CTF-1 fires (KILL) when **any** of {point < 0.040,
CI-lower ≤ 0, real ≤ perm-max} holds — i.e. when NOT (C1 ∧ C2 ∧ C3).

| dataset | source | best-k Δacc | C1 (≥+0.040)? | C2 (CI-low>0)? | C3 (>perm-max)? | calib≥0.99 (K-CTF-2)? | verdict |
|---|---|---|---|---|---|---|---|
| **HateMM** | **flat (BINDING)** | **+0.0000** | ✗ (0.0000<0.040) | ✗ (−0.0031≤0) | — moot | ✔ (1.0000) | **KILL** |
| HateMM | delta (1a ctx) | −0.0049 | ✗ | ✗ (−0.0136≤0) | — moot | ✔ | KILL |
| **MHC-EN** | **flat (BINDING)** | **−0.0029** | ✗ | ✗ (−0.0076≤0) | — moot | ✔ (1.0000) | **KILL** |
| MHC-EN | delta (1a ctx) | −0.0010 | ✗ | ✗ (−0.0127≤0) | — moot | ✔ | KILL |

**Permutation null (C3) not required.** K-CTF-1 is an OR of three kill conditions; on every cell **both** C1 and
C2 already fail, so the OR-kill fires regardless of C3. The ≥150-permutation null (which only discriminates a
would-be *pass* with point ≥ +0.040 **and** CI-lower > 0) is therefore moot and was not computed — faithful to
the kill-switch's own logic. Calibration (K-CTF-2) is VALID on all cells, so the KILL is **credited** (not
MACHINERY_INVALID).

**Binding CTF verdict (from `OUT.json['ctf_binding_verdict']`):**
`{"flat_per_dataset": {"HateMM": "KILL", "MHC": "KILL"}, "CTF_SURVIVES": false, "any_machinery_invalid": false}`.

The survival ladder in §(e) ("if the gate clears → 1a arc paired-LOO on HateMM must clear ≥+0.05 before 1b earns
a GPU") is **not reached**: the gate did not clear, so **neither 1a nor 1b earns any GPU** (S2S no-OR-ing discipline).

---

## 5. Verdict (NON-binding)

**KILL-side.** The CTF $0 conditional-information gate fires cleanly: the full causal-prefix frame-group tensor
`[g_1..g_T]` carries **no** conditional label information over `Z_best` on either dataset (HateMM +0.0000
CI[−0.0031,+0.0031]; MHC −0.0029 CI[−0.0076,+0.0016]), and its long-range increment `Δ=g_T−g_1` (the 1a arc
channel) is equally dead (HateMM −0.0049; MHC −0.0010). Because the gate is **operator-agnostic** — every pool of
`{g_t}` (fixed-mean, arc-increment 1a, learned attention-pool 1b) is a function of the same tensor whose aggregate
is already inside `Z_best` — **both CTF realizations die at $0 with no GPU spend.** Calibration is VALID
(label-oracle accZA = 1.0000, full Fano headroom) on all four cells, so this is a **genuine null, not a
capped-signal / machinery artifact**; the monotone-negative higher-capacity arms confirm redundancy-and-dilution.
This matches the design's own **honest LOW prior** (F35 causal-ancestor redundancy; F29 inverted-escalation on the
anchor) and closes the supervised-temporal-pool cell that F37's unsupervised-only oracle had left open.

**This is a NON-BINDING pre-ceremony screen.** It authorizes no prereg and consumes no formal budget; it recommends
retiring CTF (candidate #1) from the Wave-3 pool at zero cost. Any decision to record CTF as a formal
pre-registered negative is the team lead's call.

---

## 6. Required statements

- **No performance / accuracy claim** on any held-out benchmark. All accuracy numbers are train∪val
  cross-validation used **solely** to measure conditional information and audit the probe (the C3-template usage).
- **Zero test-touch:** the two `test_seen_frameset.pt` files and all `test_seen_*` pooled caches were never opened
  (canonical order built from train ⊕ dev_seen only). Gold labels used **PROBE-ONLY** (calibration arm + CV
  stratification).
- **Raw-only:** every number transcribed verbatim from `CTF_G0COND_GATE_OUT.json`; no companion metric fabricated.
- **Compute posture:** login-node CPU, no GPU, no SLURM, no Modal app created, no network. This executor created
  and stopped **zero** Modal apps.
- **Write scope:** `scripts/analysis/ctf_g0cond_gate.py`, `refine-logs/CTF_G0COND_GATE_OUT.json`,
  `refine-logs/CTF_G0COND_GATE_run.log`, this record. No prereg / config / CLAUDE.md / state files mutated.
  Committed (script + this record); **not pushed**.
