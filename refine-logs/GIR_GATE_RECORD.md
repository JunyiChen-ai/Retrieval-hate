# GIR G0-cond conditional-information gate — Wave-3 candidate #2 (NON-BINDING $0 pre-check)

**Date:** 2026-07-17. **Executor:** GIR gate executor (CPU-only, conda `HateVideo`, no GPU/SLURM-GPU/Modal/network).
**Status of this record:** **NON-BINDING prior-mover / cheap-kill screen** (pre-ceremony; precedent = the
CTF gate `0eb6d33`, APX gate `9c54faf`, W2-C CLIP-K4 pre-check `ad48dcc`). No prereg freeze was taken; full
**raw-only** transcription discipline applies (every number below copied verbatim from the primary JSON
`refine-logs/GIR_G0COND_GATE_OUT.json`, cross-checked against the console log `refine-logs/GIR_G0COND_GATE_run.log`).
**Design followed VERBATIM:** `refine-logs/WAVE3_CANDIDATES.md` CANDIDATE 2 sections (a)/(b)/(d)/(e) (commit `0ee06df`).
**Design-owner comparator clarification** (wave3-recon, in-session, binding on the spec): the ungrounded comparator
is the **banked `ungrd_vis` video-first vision-span control field**, not `img_feats`. That field IS banked (§1.3);
it is used as the design-canonical binding residual `r_field`. The `img_feats` reading from the task INPUTS is run
in parallel as `r_cache` — the exact-subsumption bound (§2.1). **Both kill on both datasets** (§3), so the binding
designation is immaterial to the verdict.

---

## 0. Headline

**KILL-side (clean).** The grounded-incongruity residual `r = grd − ungrd_vis` carries **no** conditional label
information over the strictest baseline that already contains the full grounded key —
`Z_best-incl-grd = concat(CLIP img+text, Qwen img+text, grd)` (**12544-d**) — on **both** datasets.

- **Design-canonical residual `r_field = grd − ungrd_vis(field)`** (the video-first control the design (a) names):
  best decision-family point estimate **+0.0000** on HateMM (k8, CI `[-0.0054,+0.0056]`) and **−0.0064** on MHC-EN
  (k8, CI `[-0.0159,+0.0029]`). Both fail the +0.040 bar (C1) **and** have bootstrap CI-lower ≤ 0 (C2).
- **Exact-subsumption residual `r_cache = grd − img_feats`** (task-INPUT comparator; `img_feats` == the "Qwen img"
  column literally inside the baseline, so `r_cache` is an EXACT linear function of two baseline columns —
  numerically verified `‖r_cache − (grd − Qwen_img)‖_max = 0.000e+00`): best-k **+0.0012** HateMM (CI `[-0.0040,+0.0066]`),
  **−0.0051** MHC-EN (CI `[-0.0143,+0.0038]`). Both fail C1 and C2.
- **Dilution-control** (HateMM covered-rows-only, empty-transcript+guard removed, n=802): `r_cache` best-k **−0.0035**
  (CI `[-0.0085,+0.0015]`) — DEAD, so the null is **not** an empty-transcript dilution artifact (mirrors the W2-A K9
  Amdt-5 covered-rows view, which also returned DEAD).

Calibration (K-GIR-2) is **VALID** on every cell (label-oracle accZA = 1.0000, headroom-fraction 1.000) → this is a
genuine null, not a machinery artifact. **`GIR_SURVIVES = False` — GIR dies at $0, no GPU.** This is the
subsumption outcome the team-lead pre-flagged and the design (b)/(f) pre-declared as the LOW-prior expectation: a
linear residual channel over `[Z_best, grd]` is mathematically subsumed by the W2-A K9 null (grd adds ≈0 over Z_best).

---

## 1. Provenance

| item | value |
|---|---|
| gate script | `scripts/analysis/gir_g0cond_gate.py` sha256 **`e4e585add107fe39b2066aad29b72ec2c4e908dabf18507a0e06be41d63b4567`** |
| machinery source (reused VERBATIM) | `scripts/analysis/c3_fusion_probe.py` sha256 **`9091e2c3443d4826144f820217e37d43d26d282d334b0b35bea7cb4ae9748b3c`** (same source the CTF/APX gates pin) |
| repo HEAD at run | `f003e7a` (design frozen upstream at `0ee06df` = the WAVE3_CANDIDATES.md commit) |
| invocation | `python3 scripts/analysis/gir_g0cond_gate.py` (env `NSEED` unset → default **150**) |
| where run | **LOCAL login-node CPU** (foscsmlprd01), **NOT Modal, no GPU, no SLURM, no network.** Single checkpointed process, exit 0, **elapsed 957 s (~16 min)**, minutes-scale per the CTF/APX login-node precedent + task-INPUT authorization ("local login-node CPU if minutes-scale"). Not reaped. |
| env | conda `HateVideo`: sklearn 1.5.2, numpy 1.26.4, torch 2.6.0+cu124 (OMP/BLAS threads = 4) |
| raw outputs | `refine-logs/GIR_G0COND_GATE_OUT.json` (primary) + `refine-logs/GIR_G0COND_GATE_run.log` (console) |

### 1.1 Machinery byte-identity check (substantiates "VERBATIM")

The 10 shared machinery functions (`pick_C`, `pick_C_combined`, `_fit_cor`, `baseline_cor`, `oracle_cor`,
`full_cor`, `arm_cor_allk`, `dmean`, `boot_ci`, `_perm_stats`) are **byte-identical** (per-function sha256) between
`gir_g0cond_gate.py` and `c3_fusion_probe.py`. Only the data layer differs (§2).

### 1.2 Input grounded-cache sha256 cross-check vs the frozen extraction manifest (`c013884`, `W2A_EXTRACTION_RECORD.md` §4)

Only **train + dev_seen** grounded caches were opened (canonical id order = grounded train ⊕ dev_seen). The two
`test_seen_grounded.pt` files were **never opened** → zero test-touch.

| grounded file | sha256 on disk this run | manifest (c013884 §4) | match |
|---|---|---|---|
| `HateMM/grounded_qwen7b_8f/train_grounded.pt` | `1cae1f83…f29e6a` | `1cae1f83739d6ed18c5c95a977d9ed880495f4c2dc9e27f4beaf45da9af29e6a` | ✅ |
| `HateMM/grounded_qwen7b_8f/dev_seen_grounded.pt` | `41bda7de…0125bd0` | `41bda7dea0c6ea5ef5e68787117e6b7c8bbcca8eba69057347f2a81cc0125bd0` | ✅ |
| `MHC/grounded_qwen7b_8f/train_grounded.pt` | `9f8da7a1…a48767` | `9f8da7a1b1c7d33e8b96975565f918f7785e2ece3be83244da343920f0a48767` | ✅ |
| `MHC/grounded_qwen7b_8f/dev_seen_grounded.pt` | `7c1a1a4f…3eff20` | `7c1a1a4f5d350c51ef4ec4be5bfb41a3e84dba11fbe3d03658b380acae3eff20` | ✅ |

Baseline pooled caches (features-only, read-only, train + dev_seen): `data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen}_
{openai_clip-vit-large-patch14-336_HF, Qwen2.5-VL-7B-Instruct_HF}.pt`. Dimension audit at load: CLIP img 1024 +
CLIP text 768 + Qwen img 3584 + Qwen text 3584 + grd 3584 = **12544-d**; ids equal across CLIP=Qwen=grounded per
split; label agreement asserted per-video (grd == qwen == clip). N = HateMM 744+107 = **851** (1 zero-guard
`hate_video_95` kept, matching W2-A K9 / CTF), MHC 549+80 = **629** — matching the W2-A K9 memory sizes.

### 1.3 Ungrounded comparator: identity, banking, and the exact-subsumption fact

The grounded `.pt` carries three vision pools of the same frozen forward: `grd` (transcript-first grounded key),
`ungrd_vis` (video-first ungrounded control — the K2 GroundingLive comparator), and `img_recon` (the K1
reconstruction). Measured on train+dev (per-video cosine, float64):

- `img_recon` **== `img_feats`** (standard pooled Qwen img cache): cos median **1.00000**, max-abs-diff ≤ 2.4e-4
  → confirms W2-A K1 ("the harness IS the banked forward").
- `ungrd_vis`(field) vs `img_feats`: cos median **0.99864** (min ~0.9958 off-guard), max-abs-diff ~0.039 → the
  video-first control is **near-identical but not bit-identical** to the standard cache.
- `cos(grd, ungrd_vis)` median 0.940–0.961 (reproduces the extraction record's `grounding_present_median`), i.e. the
  grounding channel is LIVE (consistent with the K2 gatelogs).

Consequence for the gate: with `r_cache = grd − img_feats` and `img_feats` == the "Qwen img" column of the baseline,
`r_cache` is an **exact linear function of two baseline columns** (verified `‖r_cache − (grd − Qwen_img)‖_max = 0`)
→ the **strictest** possible test, subsumption exact. `r_field = grd − ungrd_vis(field)` retains a tiny non-subsumed
sliver (the ~0.039-max `img_feats − ungrd_vis` difference, `‖r_field − r_cache‖_max = 0.039` HateMM / 0.026 MHC), so
it is the **most generous** admissible residual. Running both brackets the design-owner reading (field) and the
task-INPUT reading (cache); both kill.

### 1.4 W2-A branch-conditioning confirmed (GIR non-vacuously activated)

GIR is live ONLY in the "P3-pattern" cell (design (b)): W2-A raw-fails the kNN bar but its oracle ceiling survives.
Per `W2A_PROBE_VERDICT_REVIEW.md` (`7228373`): binding **K9 FAIL on both datasets** (grd Δacc −0.0000 HateMM /
−0.0038 MHC over Z_best — grd does **not** beat concat → NOT the DEAD-by-subsumption "grd wins" branch), **K5 oracle
SURVIVES** (+0.0635 HateMM / +0.0970 MHC → NOT the oracle-fired DEAD branch), **advisory raw kNN FAIL** (K6 Δacc
−0.0259). Headroom exists (oracle) but the full-key operator cannot convert it (K9/raw) — exactly the live cell where
GIR's question ("does isolating the interaction term recover the operator?") is non-vacuous. Activation valid.

---

## 2. Machinery (reused VERBATIM from `c3_fusion_probe.py`; only the data layer differs)

Identical constants and functions to the C3-template conditional-info probe (byte-identical per §1.1): Z standardized
ALONE at its Z-only inner-CV-optimal `C_Z` (grid {0.001,0.01,0.1,1.0}, `StratifiedKFold rs=0`); aux block appended
standardized × **s=50** (effectively un-penalized, refit at `C_Z` — the REFLECTION §4 fix so shared L2 cannot crush
the aux columns); aux via **train-fold PCA** (leak-free), k sliced from one kmax PCA; **5×5 RepeatedStratifiedKFold**
(rs=1000+rep), per-video correctness averaged; example-clustered (per-video) **bootstrap B=5000** on Δacc; **bar =
+0.040**; mandatory **label-oracle calibration arm** (2-col one-hot gold × s=50; accZA ≥ 0.99 or MACHINERY_INVALID);
permutation null (≥150) available on the pass-branch only.

**Data-layer swap (the only change):** the C3 `A_text` aux is replaced by the residual, and the baseline is raised to
**include the full grounded key `grd`** (design (d)):
- **baseline `Z_best-incl-grd`** = concat(CLIP img 1024, CLIP text 768, Qwen img 3584, Qwen text 3584, **grd 3584**) = **12544-d**.
- **`r_field` = grd − ungrd_vis(field)** — design-canonical (design (a); the video-first control).
- **`r_cache` = grd − img_feats** — task-INPUT comparator; exact-subsumption bound (§1.3).

Arm family per cell, verbatim: `aux_pca_k{8,16}` (decision family + max-over-k), `aux_pca_k{32,64}` (context),
`aux_full_cvC` (full 3584-d capacity-matched secondary at combined CV-tuned C), `shuffled` (seed 12345, continuity).
Cells: BINDING {HateMM,MHC}×`r_cache`×full + design-canonical {HateMM,MHC}×`r_field`×full + HateMM×`r_cache`×covered.

### 2.1 Interpretation note (single load-bearing reading; not an improvisation)

The design (d) frames the measurement as "conditional-info of r … over Z_best," and (e) states the K-GIR-1 threshold
in accuracy units ("conditional-info point-est of r … **< +0.040**"). Reusing "the C3-template conditional-info probe
verbatim" operationalizes the quantity as the C3 machinery's held-out **Δacc = accZA − accZ** with the +0.040 acc bar,
read on the **decision family** best-of-{k8,k16} (the C3-template's pre-declared decision arm). `aux_full_cvC` is the
non-decision full-dim secondary/context arm. This is the only reading under which "verbatim C3-template" and the
acc-unit kill-switch are mutually consistent. The comparator ambiguity (`ungrd_vis` field vs `img_feats`) is resolved
by running BOTH (§1.3) rather than improvising one; no other design detail required interpretation.

---

## 3. Raw results (verbatim from `GIR_G0COND_GATE_OUT.json`)

**Calibration (K-GIR-2 — machinery validity):** every cell label-oracle **accZA = 1.0000**, headroom-fraction
**1.000**, `PASS = True` → machinery **VALID** on all five cells (the aux-column-crush pathology is absent; a negative
read is admissible).

### HateMM (N=851, n_pos=341, accZ=0.8350, C_Z=0.01) — full

| source | arm | accZA | Δacc | per-video 95% CI |
|---|---|---|---|---|
| **`r_cache` = grd−img_feats** (exact-subsumption, BINDING) | aux_pca_k8 | 0.8362 | **+0.0012** | [−0.0040, +0.0066] |
| | aux_pca_k16 | 0.8308 | −0.0042 | [−0.0103, +0.0016] |
| | aux_pca_k32 (ctx) | 0.8294 | −0.0056 | [−0.0134, +0.0021] |
| | aux_pca_k64 (ctx) | 0.8294 | −0.0056 | [−0.0183, +0.0071] |
| | aux_full_cvC (sec) | 0.8397 | +0.0047 | [−0.0038, +0.0129] |
| | shuffled k8/k16 | — | −0.0073 / −0.0066 | — |
| **`r_field` = grd−ungrd_vis** (design-canonical) | aux_pca_k8 | 0.8350 | **+0.0000** | [−0.0054, +0.0056] |
| | aux_pca_k16 | 0.8310 | −0.0040 | [−0.0101, +0.0019] |
| | aux_pca_k32 (ctx) | 0.8301 | −0.0049 | [−0.0129, +0.0031] |
| | aux_pca_k64 (ctx) | 0.8291 | −0.0059 | [−0.0183, +0.0066] |
| | aux_full_cvC (sec) | 0.8353 | +0.0002 | [−0.0075, +0.0078] |
| | shuffled k8/k16 | — | −0.0073 / −0.0071 | — |

`real_max_over_kdec` (best of {k8,k16}): `r_cache` **+0.0012**, `r_field` **+0.0000**.

### MHC-EN (N=629, n_pos=193, accZ=0.7965, C_Z=0.001) — full

| source | arm | accZA | Δacc | per-video 95% CI |
|---|---|---|---|---|
| **`r_cache` = grd−img_feats** (BINDING) | aux_pca_k8 | 0.7914 | **−0.0051** | [−0.0143, +0.0038] |
| | aux_pca_k16 | 0.7876 | −0.0089 | [−0.0213, +0.0035] |
| | aux_pca_k32 (ctx) | 0.7882 | −0.0083 | [−0.0242, +0.0076] |
| | aux_pca_k64 (ctx) | 0.7847 | −0.0118 | [−0.0299, +0.0057] |
| | aux_full_cvC (sec) | 0.7911 | −0.0054 | [−0.0143, +0.0035] |
| | shuffled k8/k16 | — | −0.0022 / −0.0083 | — |
| **`r_field` = grd−ungrd_vis** (design-canonical) | aux_pca_k8 | 0.7901 | **−0.0064** | [−0.0159, +0.0029] |
| | aux_pca_k16 | 0.7866 | −0.0099 | [−0.0223, +0.0022] |
| | aux_pca_k32 (ctx) | 0.7857 | −0.0108 | [−0.0270, +0.0048] |
| | aux_pca_k64 (ctx) | 0.7860 | −0.0105 | [−0.0286, +0.0070] |
| | aux_full_cvC (sec) | 0.7911 | −0.0054 | [−0.0143, +0.0038] |
| | shuffled k8/k16 | — | −0.0022 / −0.0099 | — |

`real_max_over_kdec` (best of {k8,k16}): `r_cache` **−0.0051**, `r_field` **−0.0064**.

### HateMM covered-rows-only (n=802, n_pos=337, accZ=0.8259, C_Z=0.1) — dilution control, non-binding

| source | arm | accZA | Δacc | per-video 95% CI |
|---|---|---|---|---|
| **`r_cache`** | aux_pca_k8 | 0.8224 | **−0.0035** | [−0.0085, +0.0015] |
| | aux_pca_k16 | 0.8185 | −0.0075 | [−0.0147, −0.0007] |
| | aux_pca_k32 (ctx) | 0.8195 | −0.0065 | [−0.0162, +0.0032] |
| | aux_pca_k64 (ctx) | 0.8247 | −0.0012 | [−0.0135, +0.0107] |
| | aux_full_cvC (sec) | 0.8357 | +0.0097 | [+0.0022, +0.0175] |
| | shuffled k8/k16 | — | −0.0070 / −0.0047 | — |

`real_max_over_kdec` (best of {k8,k16}): **−0.0035** (KILL). **Honest note:** the covered-rows `aux_full_cvC` is the
one arm with CI-lower > 0 (+0.0097 CI[+0.0022,+0.0175]) — but it is the **non-decision full-dim secondary** (full
3584-d residual at a combined-CV-tuned C on the n=802 subset), it is **≈4× under the +0.040 bar**, and on the binding
full-N=851 view the same arm is +0.0047 with CI straddling 0. The gate's decision arm (best-of-{k8,k16}) is −0.0035
→ KILL. No decision-family arm on any cell reaches the bar or has CI-lower > 0.

**Structural read.** Decision-family point estimates sit at ≈0 (HateMM `r_field` k8 accZA 0.8350 = baseline accZ
0.8350 to 4dp — literally zero marginal information; `r_cache` k8 +0.0012 within its own shuffled-null band −0.0073),
and higher-k / MHC arms drift negative (MHC k64 −0.0118) — the pure-**redundancy** signature (added residual
dimensions are noise the head cannot zero out), with calibration at full headroom (1.0000) proving the machinery
*can* convert real information. This is the W2-A K9 null re-confirmed on the isolated interaction term: since `grd`
and `Qwen_img` are both already in the baseline, their difference carries nothing new — the linear residual channel
is subsumed exactly (`r_cache`) and near-exactly (`r_field`, plus a null sliver).

---

## 4. Mechanical kill/pass evaluation (design arithmetic quoted verbatim; NON-binding)

Quoted verbatim from `WAVE3_CANDIDATES.md` §(e):

> **K-GIR-1 ($0):** conditional info of `r` over `Z_best`-incl-`grd` **< +0.040** or CI-lower **≤ 0** or **≤ perm-null
> max** → DEAD. **K-GIR-2:** calibration accZA < 0.99 → MACHINERY_INVALID. Survival → paired LOO kNN `concat(key,
> r_pca)` vs `key` clears the S2S raw bar on HateMM before any GPU.

Evaluated mechanically on the decision family best-of-{k8,k16}. C1 = point ≥ +0.040; C2 = CI-lower > 0; C3 = real >
max permutation-null. K-GIR-1 fires (KILL) when **any** of {point < 0.040, CI-lower ≤ 0, real ≤ perm-max} holds.

| dataset | source (mask) | best-k Δacc | C1 (≥+0.040)? | C2 (CI-low>0)? | C3 (>perm-max)? | calib≥0.99 (K-GIR-2)? | verdict |
|---|---|---|---|---|---|---|---|
| **HateMM** | **`r_cache` (full, BINDING)** | **+0.0012** | ✗ | ✗ (−0.0040≤0) | — moot | ✔ (1.0000) | **KILL** |
| **MHC-EN** | **`r_cache` (full, BINDING)** | **−0.0051** | ✗ | ✗ (−0.0143≤0) | — moot | ✔ (1.0000) | **KILL** |
| HateMM | `r_field` (full, design-canonical) | +0.0000 | ✗ | ✗ (−0.0054≤0) | — moot | ✔ | KILL |
| MHC-EN | `r_field` (full, design-canonical) | −0.0064 | ✗ | ✗ (−0.0159≤0) | — moot | ✔ | KILL |
| HateMM | `r_cache` (covered, dilution ctrl) | −0.0035 | ✗ | ✗ (−0.0085≤0) | — moot | ✔ | KILL |

**Permutation null (C3) not required.** K-GIR-1 is an OR of three kill conditions; on every cell **both** C1 and C2
already fail, so the OR-kill fires regardless of C3. The ≥150-permutation null (which only discriminates a would-be
*pass* with point ≥ +0.040 **and** CI-lower > 0) is therefore moot and was not computed — faithful to the
kill-switch's own logic and to the CTF/APX precedent. Calibration (K-GIR-2) is VALID on all cells, so the KILL is
**credited** (not MACHINERY_INVALID).

**Binding GIR verdict (from `OUT.json['gir_binding_verdict']`):**
`{"binding_per_cell": {"HateMM|r_cache|full": "KILL", "MHC|r_cache|full": "KILL"}, "GIR_SURVIVES": false,
"any_machinery_invalid": false}`. The design-canonical `r_field` cells return the identical verdict (KILL both
datasets), so the outcome is invariant to the comparator reading.

The survival ladder in §(e) ("if the gate clears → paired LOO kNN `concat(key, r_pca)` vs `key` clears the S2S raw
bar on HateMM before any GPU") is **not reached**: the gate did not clear, so **no GPU is spent** (S2S no-OR-ing
discipline).

---

## 5. Verdict (NON-binding)

**KILL-side.** The GIR $0 conditional-information gate fires cleanly on both comparator readings and both datasets:
the grounded-incongruity residual `r = grd − ungrd_vis` carries **no** conditional label information over the
strictest baseline `Z_best-incl-grd` (12544-d, which already contains the full grounded key `grd`). The
design-canonical `r_field` is +0.0000 (HateMM) / −0.0064 (MHC); the exact-subsumption `r_cache` is +0.0012 / −0.0051;
the HateMM covered-rows dilution control is −0.0035 — all decision-family estimates an order of magnitude under the
+0.040 bar with bootstrap CI-lower ≤ 0. Calibration is VALID (label-oracle accZA = 1.0000, full Fano headroom) on all
five cells, so this is a **genuine null, not a machinery artifact**; the monotone-negative higher-capacity/MHC arms
confirm redundancy-and-dilution. Because `r_cache` is an **exact linear function of two baseline columns** (`grd` and
`Qwen_img`, both inside the baseline) and `r_field` differs only by a null sliver, the residual is mathematically
**subsumed** by the W2-A K9 null (grd adds ≈0 over Z_best) — the outcome the team-lead pre-flagged and the design
(b)/(f) pre-declared. GIR converts the banked W2-A grounded forward into a second, sharper shot at the same signal
and finds the same null. **`GIR_SURVIVES = False`; GIR dies at $0 with no GPU spend.**

**This is a NON-BINDING pre-ceremony screen.** It authorizes no prereg and consumes no formal budget; it recommends
retiring GIR (candidate #2) from the Wave-3 pool at zero cost. Any decision to record GIR as a formal
pre-registered negative is the team lead's call. Independent verdict review recommended before any DEAD/PASS is
minuted, per the CTF/APX precedent.

---

## 6. Required statements

- **No performance / accuracy claim** on any held-out benchmark. All accuracy numbers are train∪val cross-validation
  used **solely** to measure conditional information and audit the probe (the C3-template usage).
- **Zero test-touch:** the two `test_seen_grounded.pt` files and all `test_seen_*` pooled caches were never opened
  (canonical order built from train ⊕ dev_seen only). Gold labels used **PROBE-ONLY** (calibration arm + CV
  stratification).
- **Raw-only:** every number transcribed verbatim from `GIR_G0COND_GATE_OUT.json` and cross-checked against
  `GIR_G0COND_GATE_run.log`; no companion metric fabricated.
- **Compute posture:** login-node CPU, no GPU, no SLURM, **no Modal app created or stopped**, no network,
  minutes-scale (957 s) per the CTF/APX login-node precedent and the task-INPUT authorization. Zero-GPU / zero
  test-consumption respected throughout.
- **Branch pre-check:** GIR ran only after confirming W2-A's live "raw-fail / oracle-survive" cell (§1.4) and that the
  `ungrd_vis` comparator is banked (§1.3) — the two pre-conditions the design (b) / design-owner flagged.
- **Write scope:** `scripts/analysis/gir_g0cond_gate.py`, `refine-logs/GIR_G0COND_GATE_OUT.json`,
  `refine-logs/GIR_G0COND_GATE_run.log`, this record. No prereg / config / CLAUDE.md / `state/` / frozen artifacts
  mutated. Committed (script + record + probe outputs); **not pushed**.
