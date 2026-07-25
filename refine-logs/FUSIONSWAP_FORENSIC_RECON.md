# TRAINED FUSION-MODE SWAP — FORENSIC RECON (zero-GPU)

**Agent:** fusionswap forensic-recon executor. **Date:** 2026-07-25 NZST.
**Discipline:** ZERO GPU / SLURM / Modal / training / test-touch. CPU reading + code grep only.
`autoresearch/state/` untouched. One local commit on `main`, NO push. NO test metric read/produced.
**Candidate:** litsweep-5 S3 never-measured cell (a)-1 + ban-scope challenge #1 (`refine-logs/LITSWEEP5_COMPLETENESS.md` §1 L54, §2 challenge #1 L76-79; commit 4e3b09a).
**Object:** TRAINED alternative fusion modes in the align head — concat+MLP / gated / shallow cross-attention — trained end-to-end exactly like the floor (triplet+BCE hybrid, FAISS mining), deployment (top-20 rank-weighted signed-cos kNN vote over own-train memory) unchanged.

---

## 0. BOTTOM LINE — **PARK**

The candidate is a **genuine F50/F75 letter-gap** (both ban-scope over-reaches CONFIRMED, §2) and the `concat`/`cross` flags already exist in code, so the cheapest arm costs **$0 new code + ~0.2 GPU-h**. But it carries **no defensible ≥+3-on-≥2-datasets prior**: the binding leg is **ZH val-sel through a 78-item dev**, where the floor already sits at **+0.0246 acc over CLIP (FAIL, needs +0.0054 more to cross)** and a trained *symmetric* fusion swap over the *same two frozen streams* is **F66-arithmetic-capped at +0.001–0.006** — below the **±0.014 head-seed band** it must clear 3/3. The empirical head-side base rate is **0-for-~20** promoted (F70 readout perm-null, F73 SAM/mod-dropout ±noise, F75 loss-family 0/8 formal — F75's *symmetric-reshaper* null is the nearest neighbor to a fusion swap). **Arithmetic says the family cannot plausibly clear the binding ZH-val-sel leg 3/3; do not manufacture a launch. Recommend PARK** (worth ONE cheap door-closer bite for paper-completeness IFF the user later wants the fusion axis closed on the record — minimal honest form in §4).

---

## 1. CODE RECON (file:line)

### 1.1 Fusion modes already in-repo
The deployed head is `classifier_hateClipper` (`src/model/classifier.py:70-150`). `fusion_mode` is a **first-class constructor arg** (`classifier.py:71,73`) with THREE branches already wired, twice (dim-setup + forward):

| mode | dim-setup | forward op | file:line |
|---|---|---|---|
| `concat` | `input_shape = map_dim*2` | `torch.cat((img,text),dim=1)` | `classifier.py:85-86, 138-139` |
| `align` (floor) | `input_shape = map_dim` | `torch.mul(img,text)` (Hadamard) | `classifier.py:87-88, 140-141` |
| `cross` | `input_shape = map_dim**2` | `torch.bmm(img.unsqueeze(2),text.unsqueeze(1)).flatten(1,2)` (outer product) | `classifier.py:89-90, 142-143` |

`--fusion_mode` is parsed (`src/run_rac.py:118`, default `concat`) and threaded straight into the constructor (`run_rac.py:1269-1273`, both the plain and Archive variants). The deployed floor sbatch **hardcodes** `--fusion_mode "align"` (`scripts/slurm/enc3seed_lora_curric.sbatch`, the anchor). **`align` is the ONLY fusion ever run on video** — grep of `refine-logs/` finds no banked video fusion sweep in F1–F80 (litsweep5 concurs, §1 L78). The archive-stream twin `classifier_hateClipperArchive` (`run_rac.py:30-92`) also carries the same three branches — irrelevant here (archive OFF at the floor).

**Consequence:** the **concat arm needs ZERO new code** — swap one token (`align`→`concat`) in the sbatch CONFIGS. `cross` is also flag-only but comparability-broken (§1.3).

### 1.2 What a gated / cross-attn arm would need (new code → codex gate)
Neither exists. The in-repo `cross` is a *fixed-form outer product* (bmm), **not** attention.
- **Gated fusion** (e.g. `g = σ(W·[img;text]); x = g⊙img + (1−g)⊙text`, or FiLM): **~15–25 new lines** in `classifier.py` (one `nn.Linear(2·map_dim, map_dim)` in `__init__` + a `fusion_mode=='gated'` branch in forward + arg plumbing). `input_shape = map_dim` (same as align).
- **Shallow (1-layer) cross-attention** (Q/K/V over the 2 modality "tokens" + softmax + residual/LN → pooled `map_dim`): **~30–55 new lines** (a small attention block class + branch + args). Any new branch draws new RNG → same treatment-arm-RNG-divergence disclosure as NCA A1/A3; requires the mandatory **codex-code-review gate** (house precedent: NCA §2 escalated a *blocking* A3 dropout-mode bug that pure reading missed) + prereg-freeze + independent review.

### 1.3 Parameter counts / comparability (map_dim=proj_dim=1024, verified arithmetic)
Fusion changes ONLY the first MLP `Linear`'s input dim (img_proj/text_proj are pre-fusion, identical across arms):

| arm | fusion input dim | first-Linear params | ratio vs align | comparability |
|---|---|---|---|---|
| **align (floor)** | 1024 | **1,049,600** | 1.0× | — |
| **A concat** | 2048 | **2,098,176** | **2.0×** | within the "~2×" guidance ✓ |
| **B gated** | 1024 | ~1,049,600 + ~2,098,176 gate = **~3.1M** | ~3× | slightly over 2×; a param-matched control (narrower gate) available |
| **C cross-attn 1L** | 1024 (pooled) | ~1.05M + ~3× map_dim² QKV ≈ **~4M** | ~4× | over 2×; declare as capacity-uncontrolled or param-match |
| ~~`cross` (bmm, in-repo)~~ | **1,048,576** | **1,073,742,848 (~1.07B)** | **~1000×** | **BROKEN** — ~1.07B params on ~1k train items = guaranteed overfit + memory blow-up |

**Do NOT use the in-repo `cross` (bmm) as the "cross-attention" arm** — it is a different (fixed outer-product) operator, 1000× the floor's params, comparability-dead. A real 1-layer cross-attention (new code) is the intended C arm; if a bmm-cross is wanted at all it needs `map_dim≈32` (32²=1024) to param-match, a design compromise.

---

## 2. BAN ADJUDICATION (independent re-read)

### F50 (verbatim, `directions_tried.json` dead[], finding F50, `refine-logs/FA_GATE_RECORD.md` e0877c9)
> "…do not re-propose **fixed compositions, reweights, or per-modality temperatures** over banked frozen features; conversion requires adaptation (F45) or a new information source with alignment>0.663 (F49 bar)."

**S3's letter-overreach call → CONFIRMED.** F50 names exactly three *fixed* objects. A trained fusion head (`concat`-MLP / gated / cross-attn) optimized end-to-end with triplet+BCE is a **trained nonlinear operator** — none of the three. F50 actually *measured* only (A1) a scalar within-Qwen reweight (rotation at every w) and (A2) a **fixed** cross-encoder CLIP-img+Qwen-text concat scored by kNN/oracle — **neither is a trained fusion-mode swap**. Letter does NOT cover it.
**BUT** F50's *conversion thesis* (the clause after the semicolon) is the substantive headwind: converting the selection-locked frozen-feature headroom requires **encoder adaptation (F45)** or a **new >0.663 channel (F49)**. A fusion head does *neither* — it re-mixes the same two frozen streams. So F50's letter is over-reached (S3 right) but F50's *mechanism* predicts near-zero conversion.

### F75 (verbatim, dead[], finding F75)
> "head-loss swaps of the triplet+BCE hybrid toward **vote-consistent (NCA/soft-kNN), contrastive (SupCon), or mixup-BCE** objectives … tau/alpha retunes = tactics, banned … First measured negative for trained-reshaping-unlocks-oracle-headroom; F66 selection-locked pools untouched."

**S3's "F75 covers objectives not architecture" → CONFIRMED.** F75 bans **loss objectives**. A fusion-mode swap **keeps triplet+BCE unchanged** and swaps the fusion *operator* (architecture). Not covered by the F75 letter. **BUT** F75's *mechanism* generalizes: it is "the first measured negative for **trained-reshaping-unlocks-oracle-headroom**," and a trained fusion head is exactly another **trained symmetric reshaper** of the fused space. So F75 is the *nearest measured neighbor* — a same-family null, not a letter ban.

### F66 (the binding arithmetic, dead[], finding F66; `ISR_PREGATE_RECORD.md` a6e41f8)
The β-decomposition prices **any symmetric reshaper** of the fused representation: **91–98 % of the ZH/EN oracle headroom is selection-locked**, the **legal convertible slice is +0.001–0.006**. A fusion-mode swap produces a symmetric fused space (concat/gated/cross-attn are all order-symmetric over the two streams) → **falls squarely under the F66 cap.** This is the same class-argument that admitted NCA *past* F66's letter and then saw NCA *measured dead* by F75. **The fusion swap inherits that fate by construction.**

**Adjudication summary:** both S3 letter-overreach calls are **correct** — the candidate is genuinely *outside the F50 and F75 ban letters*. But it is *inside* F50's conversion thesis + F75's mechanism + **F66's arithmetic cap**. It is a legitimate **door-closer for the fusion axis**, not a goal cell.

---

## 3. FLOOR COMPARABILITY & MACHINERY

**Same floors as NCA/HEADRECIPE** (paired controls, NOT re-run): **ZH = job 13150** (`Qwen2.5-VL-7B-Instruct-LoRA_HF`, group `RAC_video_b3_lora`); **HateMM = job 13241** (`…-LoRA-curric_HF`, group `RAC_video_lora_curric`). Floor means (authoritative, `NCA_VERDICT_REVIEW.md`):

| dataset | protocol | floor acc / mF1 |
|---|---|---|
| ZH (13150) | val-sel | **0.8322 / 0.8015** |
| ZH (13150) | final | 0.8456 / 0.8173 |
| HateMM (13241) | val-sel | 0.8775 / 0.8711 |
| HateMM (13241) | final | 0.8791 / 0.8726 |

**Machinery supports the concat flag CLEANLY, no diff.** The head retrains from scratch per arm on **cached LoRA features** (fusion happens inside the head from cached img/text feats — the encoder caches are fusion-agnostic). The `ncafam_family.sbatch` pattern (`scripts/slurm/ncafam_family.sbatch`) is a byte-identical reuse of the anchor `enc3seed_lora_curric.sbatch` python command + per-arm additive flags. **Arm A (concat)** = change the `--fusion_mode "align"` token to `concat` per config → **no code diff, no codex gate** (still needs prereg-freeze + independent review per house style). **Arms B/C (gated/cross-attn)** = a `classifier.py` diff (new fusion branch + args) → **freeze + mandatory codex gate + review**, exactly like NCA's `loss.py` diff.

**Timing (measured):** NCA job 13482 = **24 head runs in 9m28s wall** (`slurm/logs/ncafam_13482.out`), features cached, ~20–50 s/run, ~0.16 GPU-h. A fusion family is the same cost class.

---

## 4. FAMILY DESIGN (minimal decisive set — *specified for completeness; recon recommends PARK*)

IF the user later greenlights a door-closer, the minimal honest form:

- **Arm A — concat+MLP** (flag-only, $0 code, 2.0× params = within comparability).
- **Arm B — gated fusion** (new code ~20 lines, param-matched control) — *optional; adds a codex gate*.
- (Drop C cross-attn and the bmm-`cross`: cross-attn is the highest-code/lowest-prior arm; bmm-`cross` is comparability-broken.)

**Grid:** 2 arms × {ZH 13150, HateMM 13241} × 3 head-seeds = **12 runs ≈ 5–6 min wall, ~0.1 GPU-h** (A-only = 6 runs ≈ 3 min). Same `ncafam` harness, fresh `GROUP_NAME=RAC_video_fusionswap`, `--force False` (never overwrites banked arms). **ONE sbatch = ONE family = ONE multiplicity bite.** Judgement per house style: **KS-arm-dead** (arm−floor mean Δacc ≤ −0.014 on a leg → objective-dead, D7-DEAD limbo) + **FORMAL dual-protocol conjunct** (val-sel AND final, both datasets, 3/3 sign, vs each arm's own align floor), rendered by an independent 0-context reviewer against a frozen prereg. Codex gate on B (skip for A).

---

## 5. HONEST PRIOR (priced against F73 / F75 / F70 + the ZH val-sel selection wall)

**The binding leg = ZH val-sel through the 78-item dev.** The align floor already sits at **+0.0246 acc over CLIP (val-sel FAIL; AND-rule fails on acc; `B3_VERDICT_REVIEW.md` L21), +0.0313 over CLIP (final, PASS-MARGINAL)**. For the goal (≥+0.030 on ≥2 datasets), the fusion head must **lift ZH val-sel by ≈+0.0054 over the align floor, 3/3**, to cross — while the per-arm change lives inside the **±0.014 head-seed band** (KS threshold; ZH LoRA head-seed std ≈0.014, `exp-archive-knn-seeds`; B3 seed spread already ~15× the pass margin). F66 caps the legal symmetric-reshaping slice at **+0.001–0.006**. A within-noise ~+0.005 mean shift cannot reliably push **all 3** val-selected seeds above a bright line on a 78-item dev (val-sel selection alone costs ~2 acc pts of noise; B3's seed2 already landed +0.0201 < bar).

**Base rate of head-side changes clearing the goal 3/3: 0-for-~20** — F70 readout grid inside perm-null, F73 SAM/mod-dropout ±noise (~+0.5pt-not-3/3 on HateMM, hurts ZH), F75 loss-family 0/8 formal (sole KS survivor +0.0112 within-noise, NOT promoted). The fusion swap is the same head-side, symmetric-reshaper family.

| leg | P(≥+1 stable acc over align floor, 3/3) | P(clears goal-bar 3/3) | note |
|---|---|---|---|
| **ZH val-sel (BINDING)** | ~0.05–0.10 (F66 legal slice exists in the *mean*, seed-noise-fragile) | **~0.03–0.06** | needs +0.0054 to cross; ±0.014 seed noise on 78-dev dominates; F66-capped |
| **ZH final** | ~0.10 | ~0.05 | already passes over CLIP; fusion adds ~0 |
| **HateMM (both)** | ~0.15 (hold-the-pass likely ~0.8) | **~0.00** | floor 0.879 near ceiling; +0.030 impossible; not the binding leg |
| **Goal (≥+3 on ≥2 ds)** | — | **~0.03–0.06** | binds on ZH val-sel; below any launch threshold |

**Arithmetic verdict:** the family **cannot plausibly clear the binding ZH-val-sel leg 3/3.** The one theoretically-convertible slice (F66 legal +0.001–0.006) is smaller than the +0.0054 it must add AND smaller than the ±0.014 noise it must beat 3/3 on a selection-underpowered dev.

---

## 6. RECOMMENDATION — **PARK**

Genuine F50/F75 letter-gap, cheap to close, but **no defensible ≥+3-on-≥2-datasets prior**: the binding ZH-val-sel leg is selection-noise-walled and F66-arithmetic-capped, and the head-side empirical base rate is 0-for-~20. **Do not manufacture a launch.** It qualifies as a **≤0.1-GPU-h, A-concat-only door-closer** for the paper's fusion-axis completeness **iff the user explicitly wants that row on the record** — but as a goal lever it is dead on arrival. **PARK.**

---

## 7. PROVENANCE
- Code: `src/model/classifier.py:70-150` (fusion branches), `:85-90/138-143` (dim-setup/forward), `:129-136` (mod-dropout, off); `src/run_rac.py:118` (--fusion_mode parse), `:1269-1273` (instantiation), `:30-92` (Archive twin). Param arithmetic verified (map_dim=proj_dim=1024).
- Machinery/timing: `scripts/slurm/ncafam_family.sbatch`, `scripts/slurm/enc3seed_lora_curric.sbatch` (anchor); `slurm/logs/ncafam_13482.out` (24 runs / 9m28s).
- Ban ledger: `autoresearch/goal_mllm_plus3/state/directions_tried.json` dead[] (F50/F66/F70/F73/F75 verbatim) + banned_constraints[]; `refine-logs/FA_GATE_RECORD.md` (e0877c9), `ISR_PREGATE_RECORD.md` (a6e41f8), `NCA_VERDICT_REVIEW.md`, `HEADRECIPE_*`, `READOUT_*`, `B3_VERDICT_REVIEW.md` (L20-27,79-81), `LITSWEEP5_COMPLETENESS.md` (4e3b09a §1/§2).
- **Required statements:** ZERO GPU / SLURM / Modal / training / test-touch spent. No held-out test metric read or produced. No `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated. Committed on `main`, not pushed.
