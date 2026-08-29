# LoRA-HateMM Forensic Recon — completing the encoder-level LoRA performance matrix (ZH pass / EN fail / HateMM UNMEASURED)

**Agent:** LoRA-HateMM forensic recon (read-only; **ZERO GPU / SLURM / Modal**; evidence audit +
prereg design only). **Date:** 2026-07-17. **Siblings mirrored:** `refine-logs/B3_FORENSIC_RECON.md`
(ZH lineage, +0.0313 marginal pass), `refine-logs/B4_FORENSIC_RECON.md` (EN lineage, banked
negative). **Cell under recon:** an **encoder-level** LoRA-SFT-adapted Qwen2.5-VL-7B *encoder* on
**HateMM**, features fed to the standard archive-OFF RGCL head (`enc3s`/`arcbase` protocol), paired
3-seed vs frozen-CLIP, dual-protocol (val-selected + final-epoch), decision rule = mean Δacc ≥ +0.030
AND mean ΔmF1 ≥ +0.030 AND sign 3/3 (identical to B3/B4 / `exp-encoder-3seed.md:73-85`).

**This measurement completes the performance evidence for terminus relaxation option (c); it makes NO
novelty claim — D7 remains the user's ruling.**

---

## BOTTOM LINE UP FRONT — **GO** (measure it), but it is a genuine ~3.5–4 h open run, NOT a ~2 min formalization

- **Regime verdict: GO. P9 does NOT pre-kill this cell.** P9's banked HateMM negative
  (C3-knn −4.7 below floor) is a **decision-level** LLAMA-FACTORY `sft_classifier` result
  (r128 α256, joint LM+binary-head SFT, raw-kNN read-out over the joint-SFT'd embeddings, *no*
  trained fusion head). This cell is the **encoder-level** regime (r16 α32, pure `stage: sft`
  CAUSAL_LM generative yes/no SFT, features → a **freshly-trained RGCL align-fusion head + kNN**).
  The two regimes are proven **non-isomorphic** by their **opposite ZH behavior** (encoder-level ZH
  kNN read-out **+0.031**, B3; decision-level ZH C3-knn **−2.2**, P9) — so P9's HateMM C3-knn −4.7 is
  a different-regime datum and cannot pre-close the encoder-level cell (§1).
- **The cell is genuinely UNMEASURED.** Unlike B3 (ZH) and B4 (EN), which had the LoRA adapter AND
  the extracted feature cache already on disk (→ ~2 min head-only runs), **HateMM has NO encoder-level
  LoRA adapter** (`logging/lora/` = `{MHC, MHC_zh}` only, no `HateMM`) and **NO arcbase-style LoRA
  cache** (`data/CLIP_Embedding/HateMM/` has only the frozen-Qwen cache + the P9 per-seed
  `p9c3_hatemm_s{0,1,2}` decision-level caches). Completing it requires the **full (b) branch**:
  fresh LoRA-SFT (~3.1 h) + extraction (~20–30 min) + 3-seed head (~2 min) = **~3.5–4 h GPU**, one A100
  (§2).
- **Mechanism-informed prior: PASS-leaning (~75–85% clears +0.03/+0.03 vs CLIP), but the pass is
  expected to be substantially INHERITED from the frozen image-grounded conversion, not LoRA-specific.**
  HateMM's decisive modality is the **image** stream (train-LOO AUC 0.826, the highest of the three
  datasets — F44); LoRA sharpens the **text** stream (F45), HateMM's *secondary* modality. So LoRA most
  likely ≈-preserves frozen-Qwen's +5.3 Pareto conversion (image untouched + text a little sharper)
  rather than adding much on top. **HateMM resembles ZH (representation-limited, image NOT collapsed) —
  NOT EN (label-limited, image collapsed to 0.599) — so the EN sign-flip degradation mechanism does
  not apply** (§3).
- **Net:** GO. The run is decision-relevant either way — a PASS supplies the goal's **second dataset**
  under one lever (encoder-level LoRA: ZH + HateMM), leaving only D7; a FAIL banks the informative
  "encoder-level LoRA is ZH-specific" negative. Machinery is ready (build script already supports
  `--dataset HateMM`; extraction runner is dataset-generic; both comparison floors — frozen-CLIP AND
  frozen-Qwen 3-seed — are on disk). Only two tiny artifacts must be authored (a HateMM SFT config +
  a `lora_sft.sbatch` case; §2, §5).
- **GPU-hour estimate: ~3.5–4.0 A100-hours** (SFT ~3.1 h dominates; extraction ~0.4 h; head ~0.03 h).
  Optional B4-EN formal closure adds ~1 min (adapter+cache exist) — bundle it into the same head submit
  for a fully-formal 3-dataset encoder-level LoRA matrix (§4).

---

## 1. REGIME DISAMBIGUATION — encoder-level LoRA (this cell) vs P9 decision-level LoRA — **GO, P9 does not cover it**

There are **two distinct LoRA systems** in this project; conflating them is the central risk this
recon must retire. The B3/B4 recons already drew the line (`B4_FORENSIC_RECON.md:127-150`); this
section states it precisely for HateMM.

### 1.1 The two regimes, side by side

| axis | **ENCODER-LEVEL LoRA (THIS cell, = B3/B4 regime)** | **DECISION-LEVEL LoRA (P9 C3/C3′/D3)** |
|---|---|---|
| SFT stage | LLaMA-Factory `stage: sft` — **pure CAUSAL_LM** generative yes/no SFT | LLaMA-Factory `stage: sft_classifier` — **joint LM + binary classifier-head** SFT |
| LoRA config | **r=16, α=32**, dropout 0.0, target q,k,v,o+gate,up,down proj (`mhc_qwen25vl_lora_sft.yaml`) | **r=128, α=256**, dropout 0.05, `lora_target=all` (`EXP_p9:47`) |
| recipe | lr 1e-4, 3 ep, cosine, warmup 0.05, per_dev_bs 1 × accum 8 (eff 8), 8-frame, bf16, vision+proj **frozen** | lr 4e-5, cls_lr 1e-4, 3 ep, 8-frame, bf16, `loss_ratio [1,1]` (lm,cls) |
| feature-extraction point | **separate post-hoc pass** (`generate_VideoMLLM_embedding_lora_HF.py`, merge_and_unload) → last-token 3584-d img/text into a single `..._LoRA_HF.pt` cache | last-token embeddings of the **same joint-SFT'd** checkpoint (`p9c3_*` per-seed caches) |
| head training | features → a **freshly-trained RGCL align-fusion head + top-20 kNN** (`src/run_rac.py`, archive OFF) | **none for kNN** — C3-knn is a *raw* kNN vote over the embeddings; C3-mlp is the in-LMM MLP head |
| protocol | 3 head-seeds on ONE encoder draw, paired vs frozen-CLIP, dual-protocol +0.03/+0.03 rule | dev-gated seed expansion, single test-touch/cell, >1pt bar |

### 1.2 The decisive proof they are non-isomorphic — **opposite ZH behavior on the SAME read-out name ("kNN")**

- **Encoder-level, ZH (B3, job 13150):** LoRA features → trained RGCL head → kNN vote = **+0.0313 acc
  (final-epoch PASS marginal)** vs frozen-CLIP (`exp-lora-zh-b3.md:20-21`, `B3_VERDICT_REVIEW.md`).
- **Decision-level, ZH (P9 C3):** joint-SFT'd embeddings → raw kNN = **C3-knn 0.7964 = −2.2 BELOW
  floor** (`EXP_p9:135,164`).

Same base model family, same dataset, same word "kNN" — **opposite sign** (+3.1 vs −2.2). The
difference is entirely the regime: the encoder-level path (lower rank, generative SFT, **plus a
freshly-trained fusion head** that adapts to the LoRA features) converts; the decision-level path
(joint-classifier SFT that reshapes the space *for the MLP head and against raw kNN* — P9's own
finding, `EXP_p9:210-213`) does not. P9b's mechanism read makes this explicit: rgcl-OFF joint SFT
"reshaped the embedding space for the MLP head and AGAINST our kNN" (`EXP_p9:213`).

### 1.3 Does P9's HateMM negative pre-kill this cell? — **NO (different regime), but it IS a tempering yellow flag**

- **P9 on HateMM (`EXP_p9:180-196`):** C3-mlp 0.8698 (s0 only) ≈ floor 0.8605 (+0.9pt); **C3-knn
  0.814 (s0/s1/s2 .823/.814/.805) = −4.7pt BELOW floor** — the *largest* decision-level kNN regression
  of the three datasets.
- **Why it does NOT pre-kill:** C3-knn is the **decision-level raw-kNN** read-out (no trained fusion
  head); this cell is the **encoder-level trained-RGCL-head** read-out — the regime that *flipped* a ZH
  −2.2 into a +3.1. There is no encoder-level LoRA measurement on HateMM at all (§2), so the cell is
  genuinely open.
- **Why it IS a yellow flag (fold into the prior, §3):** P9's HateMM C3-knn −4.7 is the empirical
  proof that "SFT-ing Qwen on HateMM and reading a kNN over the last-token embeddings" *can* be
  kNN-hostile on this dataset. The encoder-level regime differs (lower rank, generative SFT, trained
  fusion head), and that difference rescued ZH — but HateMM's decision-level kNN fell *harder* than
  ZH's (−4.7 vs −2.2), so the encoder-level RGCL head has more to overcome here than it did on ZH. The
  prior is favorable but not a foregone conclusion.

**REGIME VERDICT: GO.** The encoder-level LoRA-HateMM cell is not covered by P9; it is the untested
completion of the B3/B4 encoder-level matrix.

---

## 2. RECIPE + COST AUDIT — the full (b) branch (nothing on disk; this is NOT a formalization)

### 2.1 Inventory — what exists, what is missing (verified this recon)

| artifact | ZH (B3) | EN (B4) | **HateMM (this cell)** |
|---|---|---|---|
| encoder-level LoRA adapter (`logging/lora/<DS>`) | **EXISTS** (2026-07-02) | **EXISTS** (2026-07-02) | **MISSING** (`logging/lora/` = MHC, MHC_zh only) |
| arcbase-style LoRA cache (`..._Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`) | **EXISTS** (single draw) | **EXISTS** (single draw) | **MISSING** (only `p9c3_hatemm_s{0,1,2}` = decision-level) |
| SFT data (`data/lora_sft/HateMM/{train,val,test}.json`) | exists | exists | **EXISTS** (built 2026-07-07, 743/107/215 records) |
| `hatemm_lora_train/val` dataset registration | registered | registered | **NOT registered** (dataset_info.json has mhc_*, mhc_zh_* only) |
| frozen-CLIP 3-seed control (enc3s 12850) | 13115 | **EXISTS** | **EXISTS** (`enc3s_HateMM_openai_clip-...-336_HF_seed{0,1,2}_12850.trainlog`) |
| frozen-Qwen 3-seed (secondary floor) | 13115 | exists | **EXISTS** (`enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_12850.trainlog`) |
| **cost to complete** | **~2 min** (head only) | **~2 min** (head only) | **~3.5–4 h** (SFT + extract + head) |

**The load-bearing asymmetry:** B3/B4 were ~2 min *formalizations of near-complete measurements*
(adapter + cache pre-existed). **HateMM is a ~3.5–4 h genuine open measurement** — the encoder-level
LoRA has never been trained or extracted for HateMM. Do not price this like B3/B4.

### 2.2 The three stages + cost (one A100; scale-anchored to the MHC precedent)

| stage | command | status | cost estimate | anchor |
|---|---|---|---|---|
| **0. build+register SFT data** | `python src/utils/build_lora_sft_data.py --dataset HateMM` | data BUILT (Jul-7); registration MISSING — idempotent CPU step | **~0 GPU, seconds** | `build_lora_sft_data.py:43,175` (HateMM in `choices`/`DS_PREFIX`) |
| **1. LoRA-SFT (own train split)** | `sbatch scripts/slurm/lora_sft.sbatch HateMM` (needs a HateMM case + config — §5) | adapter MISSING | **~3.0–3.5 h GPU** | MHC 549→8222 s (2.28 h); HateMM 744 → 8222×744/549 ≈ **11,140 s ≈ 3.1 h** |
| **2. feature extraction** | `sbatch scripts/slurm/gen_embed_lora.sbatch HateMM logging/lora/HateMM` | cache MISSING; runner already dataset-generic | **~20–30 min GPU** | MHC 790 videos @8fr; HateMM 1066 videos → ~1.35× |
| **3. 3-seed enc3s head** | add 3 HateMM-LoRA rows to `enc3seed.sbatch` CONFIGS (cached feats) | control on disk | **~2 min GPU** (~20–25 s/run) | B3/B4 head-run precedent |

**Total NEW GPU: ~3.5–4.0 A100-hours**, chainable as {SFT → extract} (one job) then {head} (one
job), or a single serial job. Wall time longer under `PENDING (JobHeldUser)` — auto-release, never
force (CLAUDE.md).

**VETO compliance (single-dataset own-train-split — PASS):** stage-1 SFT trains on
`data/lora_sft/HateMM/train.json` = **HateMM own train split only** (744/743 records), binary
hateful/normal yes/no target, no gold spans/attributes, no OCR, no cross-dataset mixing — clears all
three standing vetoes, identical discipline to the MHC/MHC_zh adapters
(`B4_FORENSIC_RECON.md:78-87`). One prerequisite to confirm at run-time (low-risk): `data/video/HateMM/`
videos present for extraction — implied present since the frozen-Qwen HateMM cache was extracted.

---

## 3. HONEST PRIOR — mechanism-informed, falsifiable (no advocacy)

**Prediction: LoRA-HateMM most likely PASSES the +0.03/+0.03 conjunct vs the frozen-CLIP floor
(~75–85%), but the pass is expected to be substantially INHERITED from the frozen image-grounded
Pareto conversion, not a LoRA-specific gain.** The dominant expected outcome is
**LoRA-HateMM ≈ frozen-Qwen-HateMM** (both well above CLIP).

### 3.1 The mechanism (F44 + F45)

- **F44 (`ENCODER_SWAP_DIAGNOSIS.md`, the frozen swap):** HateMM converts (+5.3 acc, 3/3, both
  protocols) because its hate is **visually grounded** — image train-LOO AUC **0.826** (highest of the
  three datasets) — and Qwen's uniform **text** gain (+0.041 AUC) rides on a *neutral, already-strong*
  image stream (frozen-Qwen image −0.009), yielding a **Pareto move: hate recall +0.116 at zero
  non-hate cost.** The residual HateMM errors are **representation-limited**, not label-limited.
- **F45 (`B3_ZH_LORA_DECOMPOSITION.md`, the LoRA lift):** LoRA's entire gain lives in the **text
  stream** (ZH text AUC frozen-Qwen 0.847 → LoRA **0.925**, +0.078; **image untouched**, −0.007). LoRA
  converts ZH's frozen *rotation* into a *Pareto* gain by sharpening the one stream where ZH hate lives
  (text/context-borne).

### 3.2 Applying the mechanism to HateMM

- **LoRA leaves HateMM's decisive modality (image) untouched.** By F45, LoRA on the LLM backbone moves
  the text stream, not the image stream. HateMM's image AUC (0.817 frozen-Qwen) — the modality that
  *decides* HateMM — would stay ≈unchanged. So LoRA **inherits** frozen-Qwen's image-grounded Pareto
  conversion.
- **LoRA sharpens HateMM's SECONDARY modality (text).** Frozen-Qwen HateMM text is already 0.888
  (higher than ZH's 0.847); LoRA would push it further (diminishing returns near ceiling), but on an
  **image-dominated** dataset the marginal text AUC has **less to convert** than it did on ZH (where
  text was the decisive modality). ⇒ LoRA's *additive* effect over frozen is expected small.
- **HateMM resembles ZH, NOT EN — the EN sign-flip does not apply.** EN degraded under LoRA (B4:
  regresses below both frozen floors) because EN is **label-limited** (F44/SAV) with a **collapsed
  image stream** (Qwen image AUC 0.599) that cancels the text gain in the 50/50 fusion, and its 549-row
  SFT overfit. **HateMM is the opposite on every axis:** representation-limited (not label-limited),
  image strong-and-uncollapsed (0.826, no cancellation), and **more training data** (744 > 579 ZH > 549
  EN → less overfit risk). The mechanism therefore predicts HateMM behaves like **LoRA-ZH (converts)**
  or better, not like **LoRA-EN (degrades)**.

### 3.3 Does LoRA ADD over frozen's +5.3, preserve it, or degrade? — outcome distribution

| scenario | LoRA vs CLIP | LoRA vs frozen-Qwen | reading | ~P |
|---|---|---|---|---|
| **(a) ≈-preserve** (most likely) | PASS (~+0.05) | ≈ (within ±0.014) | image-inherited pass; text sharpening adds ~0 on image-dominated HateMM | ~50–60% |
| **(b) add** | PASS (>+0.05) | LoRA > frozen | text sharpening genuinely helps; LoRA = best HateMM encoder (cleanest family story) | ~15–20% |
| **(c) trail-but-pass** | PASS (+0.03–0.05) | LoRA < frozen−noise | SFT slightly perturbs the space (P9 C3-knn −4.7 echo, milder in this regime); passes conjunct, frozen still best | ~15–20% |
| **(d) FAIL** | < +0.03 | — | encoder-level SFT degraded HateMM à la EN — requires HateMM to be label-limited/image-collapsed, which F44 says it is NOT; residual risk from P9's harsh HateMM C3-knn | ~10–15% |

**Expected value: a PASS that completes the performance conjunct (2 datasets under one lever:
ZH-LoRA + HateMM-LoRA), with the honest caveat that the two passes convert via DIFFERENT modalities —
ZH's is text-borne and LoRA-specific; HateMM's is image-borne and inherited-from-frozen.** That
modality-divergence is exactly the nuance D7 must weigh (§6).

### 3.4 Pre-declared kill-switches + comparison floors (all from banked 12850 logs, `exp-encoder-3seed.md:148-170`)

**Comparison floors (per-seed, 3-seed mean, computed this recon from the 12850 trainlogs):**

| HateMM floor | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | **3-seed mean acc / F1** |
|---|---|---|---|---|---|
| **frozen-CLIP (PRIMARY floor)** | val-sel | 0.8279/0.8172 | 0.8279/0.8163 | 0.8047/0.7920 | **0.8202 / 0.8085** |
| **frozen-CLIP (PRIMARY floor)** | final-ep | 0.8186/0.7997 | 0.8047/0.7822 | 0.8140/0.7988 | **0.8124 / 0.7936** |
| **frozen-Qwen (SECONDARY floor)** | val-sel | 0.8698/0.8606 | 0.8651/0.8586 | 0.8837/0.8753 | **0.8729 / 0.8648** |
| **frozen-Qwen (SECONDARY floor)** | final-ep | 0.8605/0.8507 | 0.8605/0.8514 | 0.8837/0.8753 | **0.8682 / 0.8591** |

(Known frozen-Qwen−CLIP paired pass, for context: val-sel +0.0527 acc/+0.0563 F1; final-ep +0.0558/+0.0656 —
`exp-encoder-3seed.md:185,196`. The "0.870/0.861" memory shorthand = frozen-Qwen val-sel mean 0.8729/0.8648.)

- **KS-1 — PERFORMANCE CONJUNCT (primary kill).** LoRA−CLIP paired: mean Δacc ≥ +0.030 **AND** mean
  ΔmF1 ≥ +0.030 **AND** sign 3/3, judged **independently** under each protocol (no protocol-shopping,
  no metric-shopping). Below → **NEGATIVE** (encoder-level LoRA does not generalize to HateMM) — a
  valid, informative kill outcome.
- **KS-2 — FAMILY-COHERENCE HONESTY FLAG (not a performance kill).** Compare LoRA vs the frozen-Qwen
  floor (0.8729/0.8648 val-sel; 0.8682/0.8591 final). If **LoRA < frozen-Qwen − 0.014** (the seed
  band), pre-declare: *"on HateMM the best encoder remains frozen-Qwen; the LoRA pass is
  image-inherited, not LoRA-driven."* This weakens (does not break) the single-lever family narrative
  and is material to D7. LoRA ≥ frozen-Qwen strengthens it.
- **KS-3 — REGIME SANITY / P9 CROSS-CHECK.** If LoRA-HateMM lands **below the CLIP floor** (echoing
  P9's decision-level C3-knn −4.7), the encoder-level regime failed to convert on HateMM despite
  converting on ZH → bank as the "encoder-level LoRA is ZH-specific too" negative.
- **G-repro discipline (adapted — no bit-exact anchor exists, first LoRA draw).** (a) SFT 20-step
  smoke: loss sane/decreasing, no NaN, ckpt saves, recipe pattern matches MHC (eval_loss ~0.12–0.16
  range). (b) Head runs **same-code as 12850**: Namespace diff must be `--model` +
  derived-inert-fields only (`exp-encoder-3seed.md:126-146` retired old-vs-new-code confound bit-for-bit
  for the archive-OFF path). (c) frozen-CLIP control re-paired from 12850 (code-stable, verified).
- **SINGLE-ENCODER-DRAW limitation (pre-declared, same as B3 §0.2).** The 3 head-seeds read ONE HateMM
  LoRA draw (init + data-shuffle vary; encoder fixed) → the ±band is head-seed variance, NOT LoRA-SFT
  encoder-draw variance. An encoder-draw-stability claim would need ≥3 fresh SFT retrains
  (~9 h) — out of scope, pre-declared. Symmetric with the frozen-CLIP control (also single-draw) ⇒ a
  legitimate head-level paired test.

---

## 4. OPTIONAL EN CLOSURE — cheap; bundle into the same head submit

B4 already priced the "2-min 3-seed formal closure" of the EN LoRA cell
(`B4_FORENSIC_RECON.md:97-124`): the **EN adapter + feature cache + frozen-CLIP 3-seed control all
exist**; only the `enc3s_MHC_*-LoRA_HF_seed{0,1,2}` head arm is missing. **Honest prior: FAIL both
protocols** (seed0 anchor: val-sel **−0.031** acc, final-ep **+0.006** acc; LoRA below both frozen
floors on EN — `exp-lora-sft-encoder.md:21`). It would **formally close the 22nd negative**, not open
ground.

**Recommendation:** include EN as an **optional arm in the SAME head submit** — add 3 EN-LoRA rows
alongside the 3 HateMM-LoRA rows in the enc3s CONFIGS (EN adapter/cache already on disk, marginal cost
**~1 min GPU**). This yields a **fully-formal 3-dataset encoder-level LoRA matrix in one head job**:
ZH (PASS marginal, B3) / EN (FAIL formal) / HateMM (new). Cheap, tidy, and closes the open B4 loop.

---

## 5. PREREG SKELETON (for the downstream full ceremony — prereg → independent review → freeze → single-submit)

**H-HM (HateMM, performance clause).** Replacing the frozen-CLIP encoder with an **encoder-level
LoRA-SFT-adapted Qwen2.5-VL-7B** encoder (tag `Qwen2.5-VL-7B-Instruct-LoRA_HF`, trained on HateMM own
train split only), every other component identical (same RGCL head, topk=20, `lambda_seg=0`, archive
OFF, same 744/107/215 split, lr=1e-4/ep30/bz64/proj=map=1024/dropout/hard-neg/hybrid-loss/warmup=5) —
yields mean paired Δacc ≥ +0.030 AND mean paired ΔmF1 ≥ +0.030 with 3/3 sign vs the frozen-CLIP HateMM
control (12850), judged independently under each protocol. Only manipulated variable in the head run =
`--model`.

- **Protocols:** (A) val-selected (epoch ≥ warmup 5 max Val acc, roc tie-break); (B) final-epoch (ep29).
  Both reported, judged independently, fixed write-up "final-epoch: pass/fail; val-selected: pass/fail".
- **Gates:** G-repro (§3.4); Namespace-diff (head run = 12850 argv except `--model` + inert
  group/path); KS-1/KS-2/KS-3; single test-touch (the fresh LoRA head read is the ONE budgeted HateMM-test
  evaluation for this cell — NB HateMM test is **not virgin**: the frozen-CLIP/Qwen 12850 arms already
  read it; this is a re-measure under the same protocol).
- **Artifacts to author (2 tiny, diff-verified):** (i)
  `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_sft.yaml` = verbatim copy of
  `mhc_qwen25vl_lora_sft.yaml` with `dataset→hatemm_lora_train`, `eval_dataset→hatemm_lora_val`,
  `output_dir→logging/lora/HateMM`; (ii) a `HateMM)` case in `lora_sft.sbatch` (CONFIG + OUTDIR). Head
  runner = `enc3seed.sbatch` + 3 HateMM-LoRA rows (+ optional 3 EN-LoRA rows) + fresh GROUP.
- **Single-submit plan:** freeze prereg → independent review sign-off → author+diff the 2 artifacts →
  chain {SFT (`lora_sft.sbatch HateMM`) → extract (`gen_embed_lora.sbatch HateMM logging/lora/HateMM`)}
  → apply G-repro smoke → one head sbatch (HateMM + optional EN LoRA rows) → read back every number from
  raw trainlogs (line-numbered), apply KS-1/2/3 + decision rule verbatim under both protocols.
- **Framing sentence (verbatim, per instruction):** *this measurement completes the performance
  evidence for terminus relaxation option (c); it makes NO novelty claim — D7 remains the user's ruling.*

---

## 6. WHAT A PASS/FAIL MEANS FOR THE GOAL (D7 boundary — recon does NOT decide)

- **PASS (expected ~75–85%):** the goal's **performance conjunct** (+0.03/+0.03 on ≥2 datasets) is met
  by **one lever — encoder-level LoRA** (ZH + HateMM), a *cleaner* framing than the frozen(HateMM) +
  LoRA(ZH) *family* the B3 doc had to hedge (`exp-lora-zh-b3.md:44-54`). **But** the honest mechanistic
  reading (KS-2/§3.3) is that the two passes convert via **different modalities** (ZH text-borne,
  LoRA-specific; HateMM image-borne, inherited-from-frozen). Whether "encoder-level LoRA passes on 2
  datasets" (with divergent underlying mechanisms, one marginal +0.0313, one image-inherited) satisfies
  the goal's **"novel"** clause — and whether LoRA-SFT encoder adaptation (a 2024-25-standard technique,
  Axis-B, D7-novelty-dead by ruling F24) counts at all — is **the user's D7 ruling, not decided here.**
- **FAIL:** banks the 23rd/24th pre-registered negative (encoder-level LoRA is ZH-specific), and the
  performance conjunct remains unmet under any single-lever framing — HateMM's only formal encoder pass
  stays the **frozen** swap.

---

## 7. Provenance index (file:line / path)
- Two-regime distinction: `refine-logs/B4_FORENSIC_RECON.md:127-150`; encoder-level ZH pass
  `research-wiki/experiments/exp-lora-zh-b3.md:20-21`; decision-level ZH C3-knn −2.2 / HateMM C3-knn −4.7
  `research-wiki/EXP_p9_lmm_rgcl_video.md:135,164,180-196,210-213`.
- Encoder-level SFT recipe: `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/mhc_qwen25vl_lora_sft.yaml`
  (stage sft, r16 α32, lr1e-4, 3ep); `logging/lora/MHC/{adapter_config.json,README.md,all_results.json}`
  (MHC 549→8222 s=2.28 h, 204 steps, eval_loss 0.162); runner `scripts/slurm/lora_sft.sbatch` (MHC/MHC_zh
  cases only); data builder `src/utils/build_lora_sft_data.py:43,175` (HateMM supported).
- HateMM inventory (this recon): `logging/lora/` = {MHC, MHC_zh} (no HateMM); `data/CLIP_Embedding/HateMM/`
  = frozen-Qwen + `p9c3_hatemm_s{0,1,2}` only (no `..._LoRA_HF.pt`); `data/lora_sft/HateMM/{train,val,test}.json`
  built 2026-07-07 (743/107/215); `data/gt/HateMM/{train,val,test}.jsonl` = 744/107/215; dataset_info.json
  has no `hatemm_lora_*`.
- Extraction: `scripts/slurm/gen_embed_lora.sbatch` (dataset-generic; `<DATASET> <LORA_DIR>` args),
  `src/utils/generate_VideoMLLM_embedding_lora_HF.py`; precedent `slurm/logs/lora_embed_12146.out`
  (MHC 549+80+161 @8fr, merge_and_unload).
- Floors (per-seed): `research-wiki/experiments/exp-encoder-3seed.md:152-170` (HateMM CLIP + Qwen s0/s1/s2,
  both protocols), `:185,196` (frozen-Qwen−CLIP paired pass); ERRATUM 66012e9 (CLIP floor 0.8279/0.8172).
- Mechanism: `refine-logs/ENCODER_SWAP_DIAGNOSIS.md` (F44, `8a48938`: HateMM image 0.826 / Pareto +0.116),
  `refine-logs/B3_ZH_LORA_DECOMPOSITION.md` (F45, `d76e407`: LoRA lift = text-stream, image untouched).
- EN closure: `refine-logs/B4_FORENSIC_RECON.md:97-124,154-178` (2-min, FAIL prior).
- Control logs verified on disk: `slurm/logs/enc3s_HateMM_{openai_clip-...-336_HF,Qwen2.5-VL-7B-Instruct_HF}_seed{0,1,2}_12850.trainlog`.

**Required statements:** ZERO GPU/SLURM/Modal spent by this recon; no held-out test metric produced;
all floor numbers read from banked completed-run trainlogs (numeric-provenance discipline). No `state/`,
prereg, config, or frozen artifact mutated. Not pushed.
