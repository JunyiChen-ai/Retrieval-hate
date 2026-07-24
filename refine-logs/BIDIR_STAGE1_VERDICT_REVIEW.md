# BIDIR stage-1 (bidirectional-attention LoRA-Qwen vs banked CAUSAL-LoRA, ZH + HateMM) — INDEPENDENT 0-CONTEXT VERDICT REVIEW

**Reviewer role:** independent 0-context verdict reviewer. No prior project context; trusts ONLY primary
artifacts. Renders the binding verdict strictly against the frozen pre-registration
`refine-logs/BIDIR_STAGE1_PREREG.md` VERBATIM. Zero user interaction. CPU-only (no GPU/SLURM/Modal). Modified
nothing except this file; `autoresearch/goal_mllm_plus3/state/` untouched; nothing pushed.
**Out of scope (F0.3 / §8):** the D7 novelty boundary (whether removing the causal mask counts toward the goal's
"novel" clause is the USER's ruling, DEFERRED) and goal-level satisfaction — this review decides the **PERFORMANCE
clause only**.
**Date:** 2026-07-25 NZST.

---

## 0. Hash-freeze verification (done FIRST, before any metric was read)

```
on-disk sha256(refine-logs/BIDIR_STAGE1_PREREG.md)
  = 3c532e5370e52b2ed53e0bcc2ad63d2958823f5aca0e6f710495d8cf55565142
expected (task + refine-logs/BIDIR_STAGE1_FREEZE.md frozen block, commit a7bb2a1)
  = 3c532e5370e52b2ed53e0bcc2ad63d2958823f5aca0e6f710495d8cf55565142
```
**MATCH.** The prereg on disk is the frozen binding text. NOT VOID. Proceeding.

**Frozen artifacts + reused machinery re-verified on disk at verdict time (no drift since submit):**
```
A  36cedbac365b2b13c945adbe3437efdc61d8be15ecc85878eb9614225abe367b  src/utils/bidir_patch.py                                 [MATCH]
A2 03f39e09c417bbea291f3c06b787f5220693568cd613705693df1c2bf23e020d  src/utils/generate_VideoMLLM_embedding_bidir_HF.py       [MATCH]
B  0f17fce6910981bbc4c5942eae3b18947151bc6990ceee401fc86b252a287ecd  scripts/slurm/gen_embed_mllm_bidir.sbatch                [MATCH]
C  82a69e74d570df59a1b686891814c7756b15755901d2a645bb1d3f0164a51264  scripts/slurm/enc3seed_bidir.sbatch                      [MATCH]
causal extractor b6b61a3fa4214f28d2098ca305ca9a981934445afc81ccc5ac3939f8c1fb0ec6  src/utils/generate_VideoMLLM_embedding_lora_HF.py [MATCH]
head anchor      dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch                     [MATCH]
ZH adapter cfg   f9384d8dbdb8c1e315bb40a96952f068830c9a98cd6107f3b2ac2458e7fc477b  logging/lora/MHC_zh/adapter_config.json           [MATCH]
ZH adapter wts   35a510f4ad84542c798939cfdb340b00317a5b8a2c670b07ced8d1869dd7b438  logging/lora/MHC_zh/adapter_model.safetensors     [MATCH]
HM curric cfg    eaca36dd5cef2a4ff866a0398680d420adb157be815fe335500a387bbf9037b8  logging/lora/HateMM_curric/adapter_config.json    [MATCH]
HM curric wts    6571d132ef3218e4bdfcee98aab468df21f8aa83b16d623dd2098f8486394efa  logging/lora/HateMM_curric/adapter_model.safetensors [MATCH]
```

**Same-code head block:** `run_one`…`PY` of `enc3seed_bidir.sbatch` == `enc3seed.sbatch` == `enc3seed_lora_curric.sbatch`
(block sha `13e34e4e93c6a76988557e1c609fd54e0353c627fd36eb1c5b9e26ed187c3feb`, per freeze). The ONLY manipulated head
variables vs the banked causal controls are `--model` (`-bidir` cache tag) and `--group_name` (`RAC_video_bidir`),
plus derived `--exp_comment` / `--output_path` — confirmed at runtime by the Namespace diff (§6).

**Measurement provenance (raw logs only, job IDs per `BIDIR_STAGE1_SUBMIT_RECORD.md`):** bidir arm = job **13471**
(chain: extract **13470** → head **13471**, `afterok`; smoke **13469**). Trainlogs
`slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA-bidir_HF_seed{0,1,2}_13471.trainlog` and
`slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir_HF_seed{0,1,2}_13471.trainlog`. Comparison arms =
the CAUSAL banked arms: ZH generic-LoRA job **13150**, HateMM curric job **13241** — **re-parsed, not re-run**
(§4.1e). Context CLIP floors: ZH **13115**, HateMM `openai_clip` **12850** (Review Note 1 — parsed the `openai_clip`
trainlog, NOT the co-resident frozen-Qwen 12850 arm). Every number below re-derived with a byte-identical
reimplementation of the `enc3seed.sbatch` embedded parser (val-sel = epoch ≥ warmup 5 maximising `(Val_Retrieval
acc, roc)`, report that epoch's TEST metrics; final = max-epoch TEST), independently written and hand-verified
against raw lines.

---

## 1. Comparison floors — re-derived vs the prereg's §2.1/§2.2/§2.3 pinned tables (numeric-provenance discipline)

Independent re-parse of the raw 13150 / 13241 / 13115 / 12850 trainlogs. **Every paired-anchor mean, every per-seed
value, every selected epoch, and every TEST-line line number reproduce the prereg EXACTLY to 4dp** — no discrepancy,
no blocking flag.

### 1.1 Paired anchors (the FORMAL bars)

| anchor | protocol | s0 (sel ep, L) | s1 | s2 | mean (mine) | prereg |
|---|---|---|---|---|---|---|
| **ZH causal-LoRA 13150** (§2.1) | val-sel | 0.8322/0.8023 (e20 L220) | 0.8255/0.7956 (e26 L275) | 0.8389/0.8065 (e19 L207) | **0.8322/0.8015** | ✔ |
| | final-ep | 0.8456/0.8181 (e29 L302) | 0.8389/0.8113 (e29 L303) | 0.8523/0.8226 (e29 L298) | **0.8456/0.8173** | ✔ |
| **HateMM causal-curric 13241** (§2.2) | val-sel | 0.8791/0.8730 (e29 L331) | 0.8744/0.8678 (e14 L178) | 0.8791/0.8724 (e10 L140) | **0.8775/0.8711** | ✔ |
| | final-ep | 0.8791/0.8730 (e29 L331) | 0.8791/0.8724 (e29 L329) | 0.8791/0.8724 (e29 L331) | **0.8791/0.8726** | ✔ |

All four pinned anchor means reproduce to 4dp: **ZH val-sel 0.8322/0.8015, final 0.8456/0.8173; HateMM val-sel
0.8775/0.8711, final 0.8791/0.8726.** (HateMM s0 val-sel selects ep29 = final, so its two rows coincide, exactly as
§2.2 states.)

### 1.2 Context floors (orientation only, NOT the formal bar — §2.3)

| context | protocol | mean (mine) | prereg §2.3 |
|---|---|---|---|
| ZH frozen-CLIP 13115 | val-sel / final | 0.8076/0.7676 · 0.8143/0.7720 | ✔ ✔ |
| HateMM frozen-CLIP 12850 (`openai_clip`) | val-sel / final | 0.8202/0.8085 · 0.8124/0.7936 | ✔ ✔ |

Both context floors reproduce to 4dp (Review Note 1 discharged: I parsed
`enc3s_HateMM_openai_clip-vit-large-patch14-336_HF_seed*_12850.trainlog`, not the co-resident
`…Qwen2.5-VL-7B-Instruct_HF…` frozen-Qwen arm). These carry orientation only; the formal bar pairs vs the causal-LoRA
arms (§1.1), which reproduce exactly.

---

## 2. Bidir arm — raw measured numbers (job 13471), re-parsed + line-verified

Val-selection argmax hand-verified from the raw `Val_Retrieval` epochs (warmup ≥ 5, max acc, roc tie-break) and all
six match the parser.

### 2.1 ZH bidir (`…-LoRA-bidir_HF`, seeds 0/1/2)

| protocol | s0 acc/F1 (sel ep, L) | s1 | s2 | mean acc/F1 |
|---|---|---|---|---|
| **val-sel** | 0.6980/0.5362 (e19 L209) | 0.7181/0.6079 (e26 L272) | 0.7315/0.6108 (e17 L190) | **0.7159/0.5850** |
| **final-ep** | 0.6711/0.5379 (e29 L300) | 0.7315/0.6337 (e29 L300) | 0.7114/0.6025 (e29 L299) | **0.7047/0.5914** |

Val-argmax: s0→e19 (val acc 0.6795), s1→e26 (0.7179), s2→e17 (0.7308). Line numbers are the `Test_Retrieval … macroF1`
lines in `enc3s_MHC_zh_…-LoRA-bidir_HF_seed{s}_13471.trainlog`.

### 2.2 HateMM curric-bidir (`…-LoRA-curric-bidir_HF`, seeds 0/1/2)

| protocol | s0 acc/F1 (sel ep, L) | s1 | s2 | mean acc/F1 |
|---|---|---|---|---|
| **val-sel** | 0.7349/0.7244 (e22 L261) | 0.7767/0.7674 (e23 L271) | 0.7581/0.7490 (e21 L252) | **0.7566/0.7469** |
| **final-ep** | 0.7581/0.7471 (e29 L332) | 0.7488/0.7339 (e29 L332) | 0.7535/0.7406 (e29 L333) | **0.7535/0.7405** |

Val-argmax: s0→e22 (val acc 0.8224), s1→e23 (0.8037), s2→e21 (0.8131). Line numbers are the `Test_Retrieval …`
lines in `enc3s_HateMM_…-LoRA-curric-bidir_HF_seed{s}_13471.trainlog`. (bidir ROC also collapses vs causal — ZH
~0.70–0.75 vs causal ~0.88–0.91; HateMM ~0.83 vs causal ~0.92–0.93 — concordant with the acc/F1 degrade, not
scored here.)

---

## 3. Outcome tables — paired within head-seed (Δ = bidir − causal)

### 3.1 ZH — bidir vs CAUSAL-LoRA floor 13150 (prereg §7.1)

| seed | protocol | bidir acc/F1 | causal-LoRA acc/F1 (§2.1) | Δ(bidir−causal) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | 0.6980/0.5362 | 0.8322/0.8023 | **−0.1342/−0.2661** |
| 1 | val-sel | 0.7181/0.6079 | 0.8255/0.7956 | **−0.1074/−0.1877** |
| 2 | val-sel | 0.7315/0.6108 | 0.8389/0.8065 | **−0.1074/−0.1957** |
| **mean** | **val-sel** | **0.7159/0.5850** | **0.8322/0.8015** | **−0.1163/−0.2165** |
| 0 | final-ep | 0.6711/0.5379 | 0.8456/0.8181 | **−0.1745/−0.2802** |
| 1 | final-ep | 0.7315/0.6337 | 0.8389/0.8113 | **−0.1074/−0.1776** |
| 2 | final-ep | 0.7114/0.6025 | 0.8523/0.8226 | **−0.1409/−0.2201** |
| **mean** | **final-ep** | **0.7047/0.5914** | **0.8456/0.8173** | **−0.1409/−0.2260** |

**Sign vectors — Δacc(bidir−causal):** val-sel `[−0.1342, −0.1074, −0.1074]` = **0/3 positive**; final-ep
`[−0.1745, −0.1074, −0.1409]` = **0/3 positive**. Δmacro-F1: **0/3** both protocols. Every one of the 6 ZH
per-seed deltas is negative.

### 3.2 HateMM — bidir vs CAUSAL-curric floor 13241 (prereg §7.2)

| seed | protocol | bidir acc/F1 | causal-curric acc/F1 (§2.2) | Δ(bidir−causal) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | 0.7349/0.7244 | 0.8791/0.8730 | **−0.1442/−0.1486** |
| 1 | val-sel | 0.7767/0.7674 | 0.8744/0.8678 | **−0.0977/−0.1004** |
| 2 | val-sel | 0.7581/0.7490 | 0.8791/0.8724 | **−0.1210/−0.1234** |
| **mean** | **val-sel** | **0.7566/0.7469** | **0.8775/0.8711** | **−0.1210/−0.1241** |
| 0 | final-ep | 0.7581/0.7471 | 0.8791/0.8730 | **−0.1210/−0.1259** |
| 1 | final-ep | 0.7488/0.7339 | 0.8791/0.8724 | **−0.1303/−0.1385** |
| 2 | final-ep | 0.7535/0.7406 | 0.8791/0.8724 | **−0.1256/−0.1318** |
| **mean** | **final-ep** | **0.7535/0.7405** | **0.8791/0.8726** | **−0.1256/−0.1321** |

**Sign vectors — Δacc(bidir−causal):** val-sel `[−0.1442, −0.0977, −0.1210]` = **0/3 positive**; final-ep
`[−0.1210, −0.1303, −0.1256]` = **0/3 positive**. Δmacro-F1: **0/3** both protocols. Every one of the 6 HateMM
per-seed deltas is negative.

**Both datasets, both protocols, both metrics: 0/12 per-seed deltas positive — a maximally concordant regression.**

---

## 4. Per-switch rulings (frozen text VERBATIM; each ruled exactly as worded)

### KS-bidir-dead — the KILL bar (§3.2) → **KILL condition MET both datasets, but OVERRIDDEN by the §3.3 DEGRADE carve-out**

> KILL iff, on BOTH protocols, `mean paired Δacc ≤ 0` **OR** the acc sign is not 3/3 positive (so **neither**
> protocol produces a clean positive-mean-and-3/3-sign result). Then: **the bidir cell is DEAD on that dataset, AND
> Stage-2 MNTP is AUTO-DEFUNDED on that dataset** — this is the **Law-I / FLAT** outcome.

- **ZH:** val-sel mean Δacc **−0.1163 ≤ 0** (sign 0/3); final-ep mean Δacc **−0.1409 ≤ 0** (sign 0/3). BOTH
  protocols regress ⇒ KILL condition MET.
- **HateMM:** val-sel mean Δacc **−0.1210 ≤ 0** (sign 0/3); final-ep mean Δacc **−0.1256 ≤ 0** (sign 0/3). BOTH
  protocols regress ⇒ KILL condition MET.

The plain KS-bidir-dead reading would AUTO-DEFUND MNTP on both datasets. **BUT this is NOT the Law-I / FLAT
outcome** — it is the §3.3 DEGRADE case, which the prereg explicitly carves out of KS-bidir-dead (§3.6 ladder) and
which **overrides** the §3.2 auto-defund. Ruled under §3.3 below.

### DEGRADE branch — "Llama-pattern, MNTP-motivated" (§3.3) → **FIRES on BOTH datasets, BOTH protocols**

> Iff `mean paired Δacc ≤ −0.014` on BOTH protocols (the banked between-seed acc spread is ≤ ~0.014), record the
> outcome as **"Llama-pattern, MNTP-motivated"**: the causally-trained LoRA weights **break** under bidirectional
> attention (distribution shift) — precisely the LLM2Vec Llama-precedent under which **MNTP is the designed
> repair**. This is a **perf-dead result for Stage-1** but it does **NOT auto-defund Stage-2**: MNTP becomes a
> **SEPARATE, user-visible funding decision**. (This is the one case that overrides the §3.2 auto-defund: a strong
> concordant degrade is *evidence for* MNTP, not against.)

- **ZH:** val-sel mean Δacc **−0.1163 ≤ −0.014** ✔; final-ep mean Δacc **−0.1409 ≤ −0.014** ✔. BOTH protocols ⇒
  **DEGRADE FIRES.**
- **HateMM:** val-sel mean Δacc **−0.1210 ≤ −0.014** ✔; final-ep mean Δacc **−0.1256 ≤ −0.014** ✔. BOTH protocols
  ⇒ **DEGRADE FIRES.**

Every mean Δacc is ~7–10× beyond the −0.014 threshold, and all 12 per-seed deltas are individually negative (the
least-negative single seed is HateMM val-sel s1 at −0.0977) ⇒ this is a **strong, maximally concordant DEGRADE**,
the textbook LLM2Vec **Llama-pattern**: the causally-trained LoRA adapters catastrophically break when the causal
mask is flipped to bidirectional at inference. Per §3.3 this is **perf-DEAD for Stage-1 on both datasets** and
**Stage-2 MNTP is NOT auto-defunded** — MNTP becomes a **SEPARATE, user-visible funding decision** (a conditional
future prereg), NOT auto-funded and NOT auto-killed. *(Per Review Note 3: the −0.014 boundary is a spend/labeling
decision, not a scientific claim; either KS/DEGRADE branch is a negative Stage-1 verdict — the datum here lands
decisively in the DEGRADE bin.)*

### CONTINUE-to-stage-2 gate (§3.4, internal spend gate) → **NOT CLEARED (positive route)**

> Continue iff `mean paired Δacc ≥ +0.010` AND acc sign 3/3 positive on ≥ 1 protocol.

- **ZH:** val-sel −0.1163 (< +0.010), sign 0/3; final-ep −0.1409, sign 0/3. **Fails both protocols.**
- **HateMM:** val-sel −0.1210, sign 0/3; final-ep −0.1256, sign 0/3. **Fails both protocols.**

Neither dataset clears the positive CONTINUE route. MNTP is therefore not funded *via the positive gate*; its
disposition is governed by the §3.3 DEGRADE relabel above (separate user decision), not by the "weak-limbo /
not-funded" default (which applies only below-gate-but-not-DEGRADE).

### FORMAL verdict bar (§3.5, goal-facing) → **NEGATIVE on both protocols, both datasets (no PASS)**

> +0.030 acc AND +0.030 mF1, 3/3 seeds positive, under EACH protocol independently, vs the CAUSAL-LoRA arm.
> Headline claim requires FORMAL PASS on ≥ 2 datasets under a stated protocol.

- **ZH val-sel:** −0.1163 / −0.2165, sign 0/3. **FAIL (NEGATIVE).**
- **ZH final-ep:** −0.1409 / −0.2260, sign 0/3. **FAIL (NEGATIVE).**
- **HateMM val-sel:** −0.1210 / −0.1241, sign 0/3. **FAIL (NEGATIVE).**
- **HateMM final-ep:** −0.1256 / −0.1321, sign 0/3. **FAIL (NEGATIVE).**

No formal pass under any protocol on either dataset (every delta is a large regression, not merely below +0.030).
**Headline (≥2 datasets, one protocol): NOT met.** D7 novelty remains the user's ruling (F0.3), DEFERRED — even a
PASS would have been a performance result here; there is no PASS.

---

## 5. Fixed write-up lines (prereg §7.3)

```
ZH (bidir vs causal-LoRA):       final-epoch: FAIL; val-selected: FAIL   [FORMAL bar §3.5].
HateMM (bidir vs causal-curric): final-epoch: FAIL; val-selected: FAIL   [FORMAL bar §3.5].
KS-bidir-dead: KILL condition met both datasets (both protocols regress), OVERRIDDEN by §3.3 DEGRADE ⇒ MNTP NOT auto-defunded.
DEGRADE: Llama-pattern, MNTP-motivated (both datasets, both protocols; mean Δacc ZH −0.116/−0.141, HateMM −0.121/−0.126).
CONTINUE gate (§3.4): NOT cleared either dataset → MNTP disposition = SEPARATE user funding decision (per §3.3).
Headline (≥2 datasets, one protocol): NOT met.   D7 novelty: DEFERRED to user.
```

Task-format one-liners (all bars):

`ZH-bidir vs causal-LoRA 13150: final-epoch FAIL (Δacc −0.1409 / ΔmF1 −0.2260, acc sign 0/3); val-selected FAIL
(Δacc −0.1163 / ΔmF1 −0.2165, acc sign 0/3); KS-bidir-dead KILL-condition met but §3.3 DEGRADE fires (both
protocols ≤ −0.014) → "Llama-pattern, MNTP-motivated", MNTP NOT auto-defunded = user decision; CONTINUE gate NOT
cleared; FORMAL NEGATIVE both protocols.`

`HateMM-curric-bidir vs causal-curric 13241: final-epoch FAIL (Δacc −0.1256 / ΔmF1 −0.1321, acc sign 0/3);
val-selected FAIL (Δacc −0.1210 / ΔmF1 −0.1241, acc sign 0/3); KS-bidir-dead KILL-condition met but §3.3 DEGRADE
fires → "Llama-pattern, MNTP-motivated", MNTP NOT auto-defunded = user decision; CONTINUE gate NOT cleared; FORMAL
NEGATIVE both protocols.`

---

## 6. Compliance clauses (prereg binds; checked)

- **Patch applied + non-causality self-test recorded (§4.1a/§4.4.1) — COMPLIANT.** CPU non-causality self-test
  reproduced and recorded in submit record §3: patched mask `(1,1,6,6)` all-zero=True; `d_causal(pos0, future
  perturbed) = 0.000e+00`; `d_causal(last, sanity) = 1.042e+01`; `d_bidir(pos0) = 6.387e-02`; **VERDICT: PASS** —
  bit-reproduces prereg §4.4.1 and the prereg-review Check-1 independent run. On the REAL merged 7B, both the smoke
  (13469) and the real extraction (13470) print `[BIDIR] mask-flip patch installed on model.model; is_causal=False
  on 28 decoder attention module(s); attention is now bidirectional.` for both the ZH and HateMM arms (SDPA assert
  passed — no AssertionError).
- **Bidir caches DIFFER from causal (bidir != causal check recorded) — COMPLIANT.** Smoke record §4 Part B
  `DIFFER_CHECK: PASS` — id-matched vs the banked HateMM-curric causal cache: `hate_video_1` img max|Δ|=2.452e-01 /
  text 3.166e-01; `non_hate_video_4` img 1.315e-01 / text 2.592e-01; **no smoke row bit-identical to the causal
  row** ⇒ the mask flip genuinely changed the representation end-to-end (patch did NOT silently fail). Structurally
  corroborated: the bidir caches are distinct out-tag files (`…-LoRA-bidir_HF.pt` / `…-LoRA-curric-bidir_HF.pt`)
  with byte sizes distinct from the banked causal caches. Part C real-model non-causality belt `BELT: PASS`
  (`model.model` is `Qwen2_5_VLModel`; bound `_update_causal_mask` returns a NON-None all-zeros `(1,1,10,10)` mask).
- **Same-code head — Namespace diff = model/group/derived ONLY — COMPLIANT.** Runtime Namespace (trainlog `:1`) of
  each bidir head run vs its banked causal control differs ONLY in `model` (`…-LoRA-bidir_HF` /
  `…-LoRA-curric-bidir_HF` vs `…-LoRA_HF` / `…-LoRA-curric_HF`), `group_name` (`RAC_video_bidir` vs
  `RAC_video_b3_lora` / `RAC_video_lora_curric`), and the derived `output_path` / `exp_comment`. Every substantive
  config pin is IDENTICAL across bidir and causal: `fusion_mode='align'`, `topk=20`, `metric='cos'`,
  `loss='triplet'`, `hybrid_loss=True`, `proj_dim=1024`, `map_dim=1024`, `dropout=[0.2,0.4,0.1]`,
  `batch_norm=False`, `epochs=30`, `batch_size=64`, `lr=0.0001`, `no_hard_negatives=1`, `hard_negatives_loss=True`,
  `warmup=5`, `lambda_seg=0.0`, `archive_feats=None` (archive OFF), inert `tarc_target_source='off'`,
  `oracle_probe=False`, `lambda_tarc=0.0`, and matched `seed`. This is exactly the §4.1c Namespace-diff pin.
- **Single test-touch (F0.1) — COMPLIANT.** The 3 bidir head reads per dataset (job 13471) are the ONLY budgeted
  bidir-encoder test evaluations = exactly ONE new single-test-touch per dataset; zero test-touch before this
  verdict (the submit executor transcribed raw-only per-seed numbers, applied NO gates/deltas/pass-fail — submit
  record §5). Prior ZH/HateMM test exposures under the identical `enc3s` protocol are pre-declared (F0.1) and are
  re-measurements, not first exposures.
- **Collision safety / banked causal floors untouched (§4.3) — COMPLIANT.** Banked causal caches present with
  mtimes **bit-identical to the submit-record §2 pre-run table**: ZH `train_…-LoRA_HF.pt` @ 2026-07-02
  12:08:59.501321227, `dev_seen` @ 12:11:47.839858186, `test_seen` @ 12:17:25.706949549; HateMM
  `train_…-LoRA-curric_HF.pt` @ 2026-07-18 12:26:57.405769081, `dev_seen` @ 12:29:24.237503621, `test_seen` @
  12:34:15.123051972 — **untouched** (distinct `-bidir` out-tags cannot clobber the causal tags). Fresh bidir
  caches exist under the distinct tags (ZH 579/78/149, HateMM 744/107/215; dim 3584; dated 2026-07-25, job 13470).
  `RAC_video_bidir` head group is the real run; the smoke throwaway `logging/_smoke_bidir` was deleted (submit
  record §4 cleanup).
- **G-repro items the prereg pins — COMPLIANT.** (a) Patch self-test PASS (above). (b) Extractor same-code: causal
  extractor `b6b61a3f…` byte-unchanged at verdict time (§0); the bidir runner adds exactly one `apply_bidir_mask`
  call. (c) Head same-code (Namespace diff above). (d) Extraction shape sanity: both bidir `.pt` sets load with
  `img_feats`/`text_feats` shape `(N, 3584)`, N = split size, finite; zero-vector guard counts ZH 0/0/0, HateMM
  train 1 / dev 0 / test 0 (the single HateMM-train zero-vector guard is a <8-decodable-frame video handled
  gracefully per F0.2's deterministic sub-caveat; it is on TRAIN, not the test read, and cancels in the pairing).
  (e) Banked causal controls re-paired from banked logs, not re-run.
- **Freeze integrity (carried) — COMPLIANT.** Prereg self-sha, A/A2/B/C, and all reused-machinery shas MATCH on
  disk at verdict time (§0); no drift since submit.

**Carried Review Notes (from `BIDIR_STAGE1_PREREG_REVIEW.md`, APPROVED-WITH-NOTES; all non-blocking):**
1. **§2.3/§9 HateMM-CLIP context-floor provenance granularity** — job 12850 ran BOTH a CLIP arm and a frozen-Qwen
   arm; the §2.3 orientation number (0.8202/0.8085 · 0.8124/0.7936) is the CLIP arm. **Discharged:** I parsed
   `enc3s_HateMM_openai_clip-vit-large-patch14-336_HF_seed*_12850.trainlog` and reproduced it to 4dp; the context
   floor is orientation-only, not a paired anchor. Non-material.
2. **§3.6 "nested bars" is an ordering, not literal set-nesting.** The DEGRADE branch is carved out of / overrides
   KS-bidir-dead; FORMAL ⟹ CONTINUE. The ordering held cleanly in this ruling. Cosmetic.
3. **§3.3 −0.014 DEGRADE threshold and §3.5 demanding HateMM bar are spend/labeling boundaries, not scientific
   claims.** Recorded: the measured means (~−0.12 to −0.14) land the datum decisively in the DEGRADE bin (far past
   the boundary), and the labeling drives the MNTP SPEND decision (user-visible), never a goal-facing claim.
4. **CPU self-test coverage** — the self-test proves the mask function flips causality; `apply_bidir_mask`'s SDPA
   assert + `is_causal=False` loop are exercised by the real runner (28-module install line) and the smoke Part C
   belt (`BELT: PASS`). Gap closed. Non-material.

**No compliance violations found.**

---

## 7. FINAL VERDICT BLOCK (performance clause only)

**Prereg §7.3 fixed write-up:**
```
ZH (bidir vs causal-LoRA):       final-epoch: FAIL; val-selected: FAIL   [FORMAL bar §3.5].
HateMM (bidir vs causal-curric): final-epoch: FAIL; val-selected: FAIL   [FORMAL bar §3.5].
KS-bidir-dead: KILL condition met both datasets, OVERRIDDEN by §3.3 DEGRADE ⇒ MNTP NOT auto-defunded.
DEGRADE: Llama-pattern, MNTP-motivated (both datasets, both protocols).
CONTINUE gate (§3.4): NOT cleared either dataset → MNTP disposition = SEPARATE user funding decision.
Headline (≥2 datasets, one protocol): NOT met.   D7 novelty: DEFERRED to user.
```

**Per-switch (verbatim rulings):**
- **KS-bidir-dead (§3.2):** KILL condition **MET** on both datasets (both protocols regress: ZH val-sel −0.1163 /
  final −0.1409; HateMM val-sel −0.1210 / final −0.1256; acc sign 0/3 everywhere) — **but this is NOT the Law-I /
  FLAT outcome**: the §3.3 DEGRADE carve-out fires and **OVERRIDES the auto-defund**. Stage-1 is perf-DEAD; MNTP is
  NOT auto-defunded.
- **DEGRADE (§3.3):** **FIRES on both datasets, both protocols** (all four mean Δacc ≤ −0.014, by ~7–10×; 0/12
  per-seed deltas positive) ⇒ **"Llama-pattern, MNTP-motivated"** — the causally-trained LoRA adapters break under
  bidirectional attention, the LLM2Vec Llama-precedent. Stage-1 perf-dead; **Stage-2 MNTP = a SEPARATE,
  user-visible funding decision** (a conditional future prereg), not auto-funded, not auto-killed.
- **CONTINUE gate (§3.4):** **NOT CLEARED** on either dataset (no protocol reaches +0.010 acc with 3/3 sign — all
  regress). MNTP is not funded via the positive route; its disposition is the §3.3 user decision.
- **FORMAL bar (§3.5):** **FAIL (NEGATIVE) on both protocols, both datasets** — every delta is a large regression,
  far below the +0.030/+0.030 3/3 conjunct. **Headline NOT met.** D7-novelty DEFERRED (F0.3).

**Composite (performance clause only):** Flipping the LoRA-Qwen2.5-VL-7B decoder's attention topology from causal
to bidirectional at inference — the LLM2Vec / NV-Embed recipe, on the SAME banked adapters, SAME prefix-mean
readout, SAME 8 frames — **catastrophically REGRESSES** the head, paired within head-seed on BOTH datasets and BOTH
protocols: ZH −0.1163 (val-sel) / −0.1409 (final) acc and up to −0.2802 macro-F1; HateMM −0.1210 (val-sel) /
−0.1256 (final) acc, −0.14 macro-F1; 0/12 per-seed deltas positive. This is **not** the prereg's most-likely FLAT /
Law-I kill (§3.2 auto-defund) — it is the pre-declared **DEGRADE / "Llama-pattern, MNTP-motivated"** branch (§3.3,
F0.7): the causal mask flip breaks the causally-trained LoRA weights exactly as the LLM2Vec Llama-precedent
predicts, which is precisely the failure mode MNTP is designed to repair. Per the frozen rules this is a
**perf-DEAD Stage-1 result on both datasets** (no FORMAL pass, no CONTINUE), with the crucial carve-out that
**Stage-2 MNTP is NOT auto-defunded** — it becomes a **separate user-visible funding decision**, not the reviewer's
to make. No compliance violation; the banked causal floors are untouched (byte-tags distinct, mtimes unchanged);
the head is byte-identical same-code (Namespace diff = model/group/derived only); the bidir caches genuinely differ
from causal (DIFFER PASS); single test-touch per dataset honored.

**Out of scope for this reviewer (F0.3 / §8):** whether removing the causal mask is an architecture-level object
that counts toward the goal's "novel" clause (D7) is the USER's ruling, DEFERRED; **and the Stage-2 MNTP funding
decision that the DEGRADE branch routes to the user is likewise the USER's, NOT rendered here.** Goal-level
satisfaction is a USER ruling. This review renders the **performance clause only**, as the frozen prereg mandates.

---

*Reviewer statements: hash verified before any metric was read; all four comparison floors (ZH 13150, HateMM 13241,
ZH-CLIP 13115, HateMM-CLIP 12850) re-derived from raw trainlogs with a byte-identical reimplementation of the
enc3seed parser and reproduce the prereg §2.1/§2.2/§2.3 tables to 4dp including per-seed selected epochs and TEST-line
line numbers; the bidir arm (job 13471) re-parsed from raw and line-verified (per-seed both protocols, selected-epoch
argmax, TEST-line line numbers); the head Namespace confirmed runtime same-code vs each causal control; banked causal
caches confirmed untouched by mtime vs the submit record; the CPU non-causality self-test and the bidir≠causal DIFFER
check confirmed recorded; no GPU/SLURM/Modal spent; no state/ mutated; nothing pushed; no goal/novelty claim made and
no MNTP funding decision made.*
