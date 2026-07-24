# BIDIR-SURGERY Pre-Registration (STAGE-1) — bidirectional-attention LoRA-Qwen vs banked CAUSAL-LoRA (ZH + HateMM)

**Author:** bidir stage-1 prereg author (CPU-only; no GPU/SLURM/Modal spent; NO job submitted).
**Date:** 2026-07-25 NZST.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced; no job submitted; `state/` untouched.
**Implements:** `refine-logs/BIDIR_SURGERY_FORENSIC_RECON.md` (commit `ec5add8`, the GO-IF recon) — its **stage-1**
design (training-free mask-flip on ZH-primary + HateMM-hold, extraction + head only, single lever = attention
topology), mechanics (§1), cost ledger, and outcome/kill skeleton (§5) transcribed and re-verified below.
Deviations from the recon are flagged **loudly** in §11.
**House-style precedent:** `refine-logs/FRAME16_PREREG.md` (`0b5cbb5`; closest shape — extraction-side change,
frozen/adapter weights unchanged, small chain, F0.x honesty clauses, re-derived floors, freeze block, single-submit
plan, outcome-table template) and `refine-logs/VISION_UNFREEZE_PREREG.md` (ceremony depth, import-machinery-verbatim
pattern). Decision rule verbatim from `research-wiki/experiments/exp-encoder-3seed.md:73-85`.

## Title + claim scope (verbatim)

> This measurement tests **the one axis every banked Qwen arm shares and none ever varied — the LLM decoder's
> causal attention topology** — by **flipping `is_causal=True` to bidirectional at inference** (LLM2Vec / NV-Embed
> recipe) when harvesting embeddings from the **already-banked LoRA adapters**, on **MHC-ZH (F45 val-sel target,
> primary)** and **HateMM (mechanism hold, curric adapter)**. It is a **PERFORMANCE lever aimed directly at our
> OWN diagnosed pathology** (F35: `is_causal=True` makes every Qwen frame-group vector a cumulative causal-prefix
> summary — no run ever removed the mask; this cell IS that removal). It re-extracts with the **SAME frozen
> adapter, SAME baseline prefix-mean readout, SAME 8 frames — the ONLY difference from the banked causal arm is the
> attention mask.** It makes **NO novelty claim here — D7 (is this an architecture-level third structural object
> that counts toward the goal's "novel" clause) is the USER's ruling, DEFERRED** (F0.3); this prereg decides the
> **performance clause only**, against the CAUSAL-LoRA arm. Stage-1 (this submit) is training-free; **Stage-2 MNTP
> is a CONDITIONAL FUTURE prereg, NOT authored or submitted here**, and its funding is governed by the Stage-1
> outcome shape (§3/§8).

The cell under test is the **bidirectional-attention** LoRA-Qwen2.5-VL-7B encoder (mask flipped via
`src/utils/bidir_patch.apply_bidir_mask`, LoRA weights unchanged), features fed to the standard archive-OFF RGCL
align-fusion head + top-20 kNN (`enc3s`/12850 protocol), paired **3-seed within head-seed vs the banked
CAUSAL-LoRA arm** (ZH = job **13150** generic-LoRA / B3; HateMM = job **13241** curric — the SAME adapter each
bidir arm re-extracts, so the pairing isolates the mask topology exactly), dual-protocol (val-selected AND
final-epoch), each dataset trained ONLY on its own train split (hard veto). **The banked frozen-CLIP floors
(ZH 13115, HateMM 12850) are carried for orientation only (§2.3), NOT as the formal bar.**

---

## 0. Binding facts / honesty clauses (all present; pre-declared)

**F0.1 — Test is NOT virgin (declared).** Under the identical `enc3s` protocol, the held-out test was already read
by: **ZH (MHC_zh)** — frozen-CLIP + frozen-Qwen (job 13115, B1), **generic-LoRA (job 13150, B3)**, curric-LoRA
(job 13241, cand-2); **HateMM** — frozen-CLIP + frozen-Qwen (job 12850), generic-LoRA (job 13235, F53),
**curric-LoRA (job 13241, cand-2)**, curric-rep2 (job 13246), frozen-16f (frame16 chain), plus the LoRA-HateMM
verdict. This prereg's bidir head reads are **re-measurements under the identical protocol**, not first exposures.
They consume exactly ONE budgeted **bidir-encoder** test evaluation per dataset = the **3 head-seed reads**.
**Zero test-touch before the independent verdict.**

**F0.2 — Single-extraction "draw", deterministic, and the single-SFT-draw confound CANCELS in the pairing
(pre-declared; STRONGER than the LoRA cells' F0.2).** The bidir arm re-extracts with the **SAME banked adapter**
the causal arm used (`logging/lora/MHC_zh` for ZH; `logging/lora/HateMM_curric` for HateMM) — **NO new SFT.** The
frozen forward is fully deterministic given (adapter weights, mask, sampled frame indices, max_pixels): frame
sampling is `np.linspace` (deterministic, no RNG), `attn=sdpa`, `bf16`, `no_grad`, single forward — **no stochastic
encoder draw at all** (identical to frame16 F0.2). Therefore (i) the reported ±band is **purely head-seed
variance**, symmetric with the banked causal arm (also 3 head-seeds over one deterministic extraction); and (ii)
**the single-SFT-draw caveat that burdens every LoRA cell is SHARED between the paired arms** (both read the same
adapter) and **cancels** — the paired delta isolates the **attention topology** alone. The LoRA adapter modifies
the attention **linear weights** (`q/k/v/o_proj` + mlp; verified §1.4, zero `visual.*`); the mask patch modifies
the attention **topology**; the two are orthogonal (recon §2), so "same adapter, mask flipped" is a clean
single-variable manipulation.
*Sub-caveat (deterministic, conservative, identical to the causal arm):* videos with <8 decodable frames yield
duplicated `linspace` indices (graceful; both arms use 8 frames, so this cancels in the pairing).

**F0.3 — Novelty = D7, DEFERRED to the USER (NOT decided here); this prereg decides the performance clause only.**
Removing the causal mask is an **architecture-level change to attention topology** — the recon (§4/§5.4) argues it
is a **third/fourth structural object** (neither a generic encoder-weight swap [F24] nor the two adapted objects of
F51 nor P9b), that measuring it is **in-doctrine under the F65 precedent** (F65 already refuted F51's "two-object
closure" wording by *measuring* the vision-reach object), and that it is the **highest-D7-payoff cell in the
litsurvey** (a named/cited SOTA mechanism — LLM2Vec 2404.05961 / NV-Embed 2405.17428 — aimed at our OWN diagnosis
F35). **BUT whether a pass counts toward the goal's "novel" clause is the USER's D7 ruling, and it is DEFERRED.**
Per the current user mandate the goal is **performance-only (substantial gains)**; this prereg decides that clause
against the causal arm and makes NO novelty claim. **GO-IF GATE (recon §0, the author flags; the author does NOT
authorize GPU):** before any GPU the orchestrator must obtain **(1) a codex/0-context review of the mask patch
against the installed transformers source** (mandatory reviewer check, §4.5) and **(2) a one-line D7 user
sub-ruling** acknowledging this is an architecture-level object worth the ~0.5 GPU-h even at a low perf prior.

**F0.4 — Single-dataset own-train-split VETO compliance (hard user veto; trivially clean — Stage-1 has NO
training).** Stage-1 re-extracts with the **already-banked own-split-trained adapters** (ZH: `logging/lora/MHC_zh`
trained on MHC_zh's own train split; HateMM: `logging/lora/HateMM_curric` on HateMM's own train split — the SAME
adapters the banked causal comparators used). The bidir extractor reads only each video + the two FIXED
instructions (`IMG_INSTRUCTION`/`TEXT_INSTRUCTION`) — **no labels enter the encoder path**. The RGCL head trains on
each dataset's own train split only (identical to the causal arm). NO cross-dataset mixing, NO gold spans/attributes,
NO OCR channel, raw videos never leave the machine. All standing vetoes cleared. (Stage-2 MNTP — NOT this submit —
would be **self-supervised masked-token pretraining on the SAME own-train videos**, own-split → veto-legal per
recon §3, but it is not funded here.)

**F0.5 — Honest prior is LOW, and Law-I lowers it; nothing raises it (pre-declared, material).** Recon §5.3: **~10–15%**
for ≥+1pt on ≥1 dataset. Discounted by **Law-I** (F37/F39: on the Qwen reps the pooled prefix-mean already
integrates the whole sequence, so re-pooling/matching adds nothing). The one thing keeping the prior above zero:
bidir changes the **content** of each token vector (every token computed with full past+future context), whereas
F37/F39 only showed re-pooling the same **causal** vectors is lossless — bidir is the **sole lever that changes the
vectors themselves**, not how they are pooled. Realistic target = **ZH val-sel hardening** (F45 selection tax; the
B3 ZH causal arm is val-sel FAIL / final PASS-MARGINAL vs CLIP). **HateMM already passes strongly under curric
(F53/cand-2, 0.8775 val-sel = project best), so bidir on HateMM can at most sharpen an already-passing leg** — its
FORMAL bar (+0.030 over 0.8775/0.8791) is ~0.9075/0.9091, very demanding; ZH is the live perf surface.

**F0.6 — Ban-scope check (verified vs recon §4; NO collision).** **NOT F24/encoder-swap** — same encoder, same
frozen adapter weights, only attention **topology** changes at inference (not a weight-space lever). **NOT F51's two
adapted objects** — topology is neither generic-LoRA-weights nor joint head-training; it is a third structural
object, and F65 already breached the F51 "closure" wording by measuring one. **NOT P9b** — no retrieval loss, no
head coupling, no training at Stage-1. **NOT F35–F39 (don't-pool family)** — those pool/match/supervise OVER the
cumulative-causal vectors; bidir **ATTACKS the F35 mechanism** (removes the `is_causal=True` that creates them)
rather than pooling over their output. Distinct family. (Full quoted scopes: recon §4.)

**F0.7 — Pre-declared honest most-likely outcome (recon §5.1, informative either way).** The single most likely
result is **FLAT** on ZH (Law-I): the topology change carries no *new convertible* dev/test signal that the pooled
mean did not already see ⇒ **KS-bidir-dead**, a clean cheap kill that converts F35 from a passive diagnosis into an
actively-tested named mechanism (paper value even on a kill). A **DEGRADE** (causally-trained weights break under
bidir) is the LLM2Vec-**Llama-pattern** that *motivates* Stage-2 MNTP as the designed repair — a distinct outcome
label, not a plain null (§3.3). A **MOVE** (bidir clears the CONTINUE gate) funds Stage-2 MNTP; a **FORMAL PASS**
(both protocols, ≥2 datasets) would be the headline, with D7 novelty still the user's ruling.

---

## 1. Pipeline spec — fully pinned (2 stages; NO SFT; nothing left to interpretation)

**Stage 0 — none.** No SFT / no data build: the bidir extractor reuses the **banked adapters** and reads
`data/gt/<DS>/{train,val,test}.jsonl` + `data/video/<DS>/All/<id>.mp4` directly. The RGCL head consumes the derived
`.pt` caches.

### 1.1 Stage 1 — bidirectional-attention feature extraction (mask flip; SAME adapter; baseline readout)

- **Submit:** `sbatch scripts/slurm/gen_embed_mllm_bidir.sbatch` (artifact B, §5 — hardcodes both rows).
- **Runner:** `src/utils/generate_VideoMLLM_embedding_bidir_HF.py` (artifact A2) — a **thin fork** of the causal
  LoRA extractor that **imports its pooling operator VERBATIM** (`read_gt` / `process_split` / `SPLIT_TO_OUTNAME`,
  which use its `_encode` / `load_video_frames` / `IMG_INSTRUCTION` / `TEXT_INSTRUCTION`), so the ONLY difference
  from the causal arm is the mask. Its `main()` loads the base Qwen2.5-VL (`attn_implementation="sdpa"`), merges
  the LoRA adapter (`PeftModel.from_pretrained` → `merge_and_unload`), then — **the ONE new step** — calls
  `bidir_patch.apply_bidir_mask(model)` **after the merge and before any forward** (binds the all-zeros mask to
  `model.model._update_causal_mask`, asserts SDPA, clears `is_causal`; §1.3). The banked causal extractor
  `generate_VideoMLLM_embedding_lora_HF.py` is **NOT edited** (sha unchanged; all other provenance chains stay
  valid — §11 DEV-2).
- **Two rows, each reusing its OWN banked adapter (no re-SFT):**
  - **ZH:** `--dataset MHC_zh --lora_dir logging/lora/MHC_zh --out_model_tag Qwen2.5-VL-7B-Instruct-LoRA-bidir_HF`
    (same adapter as the 13150 generic-LoRA arm).
  - **HateMM:** `--dataset HateMM --lora_dir logging/lora/HateMM_curric --out_model_tag
    Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir_HF` (same adapter as the 13241 curric arm).
- **8 frames, baseline (prefix-mean) readout, NOTHING else changed** vs the causal LoRA arm (recon §2/§6: bidir is
  its own single arm on the baseline readout — NO bidir×readout grid).
- **DISTINCT `-bidir` out-tags** ⇒ the banked causal caches (`…-LoRA_HF` / `…-LoRA-curric_HF`) are **NEVER
  clobbered** (collision-checked ABSENT, §4.3).
- **Output:** `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-bidir_HF.pt` and
  `data/CLIP_Embedding/HateMM/{…}_Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir_HF.pt` (loader contract
  `{ids, img_feats, text_feats, labels}`, Dv=Dt=3584; both datasets route via `load_feats_MHC`,
  `dataset.py:499-503`, verified). Then B2-push (derived `.pt` only; videos never leave).
- **Cost:** ~0.5–0.7 GPU-h total (ZH ~579 videos + HateMM ~1066 videos; one frozen forward/video, same order as any
  8f re-extraction; bidir's full-attention adds negligibly at these sequence lengths).

### 1.2 Stage 2 — 3-seed RGCL align-fusion head + kNN (paired vs the banked CAUSAL-LoRA arm)

- **Submit:** `sbatch scripts/slurm/enc3seed_bidir.sbatch` (artifact C, §5).
- **What it runs:** 6 head-only runs (features cached, ~20–25 s each): MHC_zh-bidir seeds 0/1/2
  (`--model Qwen2.5-VL-7B-Instruct-LoRA-bidir_HF`) and HateMM-curric-bidir seeds 0/1/2
  (`--model Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir_HF`), `--group_name RAC_video_bidir`, `--force False`.
- **CRITICAL same-code guarantee (verified this prereg — §4.2):** the `run_one`…`PY` block of `enc3seed_bidir.sbatch`
  is **BYTE-IDENTICAL** to both `enc3seed.sbatch` (the 12850 runner) and `enc3seed_lora_curric.sbatch` (the 13241
  runner) — `diff` empty. The **ONLY** manipulated variables vs the banked causal arms are `--model` (the `-bidir`
  cache) and `--group_name`. Config verbatim: `--batch_size 64 --lr 0.0001 --epochs 30 --topk 20 --proj_dim 1024
  --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align --hard_negatives_loss True --no_hard_negatives 1
  --metric cos --loss triplet --hybrid_loss True --warmup 5 --lambda_seg 0 --archive OFF`. Identical to
  `exp-encoder-3seed.md` H1 / B3 / cand-2 / frame16 / vision-unfreeze.
- **Pairing:** per head-seed (bidir seed s − causal seed s), `s ∈ {0,1,2}`. `--seed` controls head-init +
  data-shuffle; the only difference between treatment and the banked causal arm at a given seed is the feature cache
  (bidir vs causal, **same adapter**). ZH pairs vs job **13150** (`…-LoRA_HF`); HateMM pairs vs job **13241**
  (`…-LoRA-curric_HF`). Banked causal arms are **NOT re-run**.
- **Output:** `slurm/logs/enc3s_{MHC_zh,HateMM}_…-bidir…_seed{0,1,2}_<JID>.trainlog`. Cost ~2 min GPU total.

**Total NEW GPU: ~0.5–0.7 A100-h** (extraction dominates; head ~0.03 h). No SFT, no ban collision.

**The recon's "$0 dev screen" is FOLDED INTO this verdict (recon deviation — LOUD, §11 DEV-1).** The recon (§2)
proposed a train+dev-only head screen (zero test-touch) as the cheap kill. Per the task and the **frame16 house
pattern**, this prereg instead **mirrors frame16 exactly**: run the full 3-seed head (which reads test), transcribe
raw both-protocol per-seed numbers, and let the **0-context verdict's KS bar BE the screen** (§3). The single
test-touch per dataset (the 3 head reads) is the only budgeted read; the KS-bidir-dead bar (§3.2) is the "screen."

---

## 2. Comparison floors — INDEPENDENTLY RE-DERIVED from raw trainlogs (numeric-provenance discipline)

Every number below was independently re-parsed **this prereg** from the raw `slurm/logs/enc3s_*_{13150,13115,13241,12850}.trainlog`
with the EXACT `enc3seed.sbatch` embedded parser (val-sel = epoch ≥ warmup 5 max `Val_Retrieval` acc, roc tie-break;
final = max epoch). **Cross-validation: my parser reproduces the published B3 verdict bit-exactly** — ZH-LoRA − CLIP
final-ep mean **Δacc +0.0313 / ΔmF1 +0.0453**, val-sel **Δacc +0.0246** (`exp-lora-zh-b3.md` / B3 verdict) — and the
frame16 HateMM-CLIP floor (0.8202/0.8085 val-sel; 0.8124/0.7936 final) to 4dp. No discrepancy.

### 2.1 ZH generic-LoRA floor (job 13150) — the PAIRED anchor for the ZH bidir arm (delta = bidir − causal-LoRA)

| protocol | s0 acc/F1 (sel ep, Test line) | s1 acc/F1 | s2 acc/F1 | 3-seed mean acc/F1 |
|---|---|---|---|---|
| **val-sel** | 0.8322/0.8023 (e20, L220) | 0.8255/0.7956 (e26, L275) | 0.8389/0.8065 (e19, L207) | **0.8322 / 0.8015** |
| **final-ep** | 0.8456/0.8181 (e29, L302) | 0.8389/0.8113 (e29, L303) | 0.8523/0.8226 (e29, L298) | **0.8456 / 0.8173** |

Provenance: `enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog` (line numbers above, re-read
this prereg).

### 2.2 HateMM curric floor (job 13241) — the PAIRED anchor for the HateMM bidir arm (delta = bidir − causal-curric)

| protocol | s0 acc/F1 (sel ep, Test line) | s1 acc/F1 | s2 acc/F1 | 3-seed mean acc/F1 |
|---|---|---|---|---|
| **val-sel** | 0.8791/0.8730 (e29, L331) | 0.8744/0.8678 (e14, L178) | 0.8791/0.8724 (e10, L140) | **0.8775 / 0.8711** |
| **final-ep** | 0.8791/0.8730 (e29, L331) | 0.8791/0.8724 (e29, L329) | 0.8791/0.8724 (e29, L331) | **0.8791 / 0.8726** |

Provenance: `enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`. (s0 val-sel selects
ep29 = final, so its two rows coincide; s1/s2 val-sel pick early epochs but land at the same/near test acc — a very
stable, HIGH floor: 0.8775 val-sel = project best, so the HateMM FORMAL bar is demanding, §F0.5.)

### 2.3 Context floors (NOT paired anchors — for orientation only)

**ZH frozen-CLIP (job 13115)** — val-sel **0.8076/0.7676**, final **0.8143/0.7720**
(`enc3s_MHC_zh_openai_clip-…-336_HF_seed{0,1,2}_13115.trainlog`; s0 val=final e29 L305; s1 val e28 L294 / final e29
L304; s2 val e25 L264 / final e29 L301). **HateMM frozen-CLIP (job 12850)** — val-sel **0.8202/0.8085**, final
**0.8124/0.7936** (matches frame16 §2.2). These orient the "K-V1-style" gap only; **the FORMAL bar is vs the CAUSAL
arm (§2.1/§2.2), NOT vs CLIP** — bidir *replaces the representation* of the same LoRA adapter, so the apples-to-apples
comparison is bidir-vs-causal-LoRA.

---

## 3. Decision rule + kill-bars (paired, both protocols judged independently, 3/3 sign, pre-declared)

### 3.1 Decision rule — verbatim from `exp-encoder-3seed.md:73-85` (treatment = bidir; control = CAUSAL-LoRA)

> For each dataset × protocol: (1) per-seed paired difference δ = (treatment − control) for acc and macro-F1 at
> seeds 0/1/2; (2) 3-seed mean ± std + sign consistency (how many of 3 positive); (3) n=3 too small for a bootstrap
> — report the paired-t **as an effect-size descriptor only**, no significance claim; (4) **pass = mean paired Δacc
> ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive**; (5) headline claim requires pass on ≥ 2
> datasets under a stated protocol; both protocols judged separately; verdict written exactly "final-epoch:
> pass/fail; val-selected: pass/fail".

Both protocols judged **independently** (no protocol-shopping, no metric-shopping). Control = the CAUSAL-LoRA arm
(ZH §2.1 / HateMM §2.2).

### 3.2 KS-bidir-dead — the KILL bar (auto-defunds Stage-2 MNTP on that dataset)

**KILL iff, on BOTH protocols,** `mean paired Δacc ≤ 0` **OR** the acc sign is not 3/3 positive (so **neither**
protocol produces a clean positive-mean-and-3/3-sign result). Then: **the bidir cell is DEAD on that dataset, AND
Stage-2 MNTP is AUTO-DEFUNDED on that dataset** — this is the **Law-I / FLAT** outcome (F37/F39): the pooled
prefix-mean already integrates the whole sequence, so a topology change carrying no new convertible signal proves
MNTP cannot re-supervise information the mask flip shows is absent. State this at verdict time. *(Sign bar per
frame16 DEV-1 precedent + house n=3 discipline — NOT a bootstrap CI, §11 DEV-3.)*

### 3.3 DEGRADE branch — "Llama-pattern, MNTP-motivated" (a DISTINCT outcome label; NOT auto-defund)

**Iff `mean paired Δacc ≤ −0.014` on BOTH protocols** (the banked between-seed acc spread is ≤ ~0.014), record the
outcome as **"Llama-pattern, MNTP-motivated"**: the causally-trained LoRA weights **break** under bidirectional
attention (distribution shift) — precisely the LLM2Vec Llama-precedent under which **MNTP is the designed repair**
(recon §3/§5.1). This is a **perf-dead result for Stage-1** but it does **NOT auto-defund Stage-2**: MNTP becomes a
**SEPARATE, user-visible funding decision** (a conditional future prereg), NOT auto-funded and NOT auto-killed.
(This is the one case that overrides the §3.2 auto-defund: a strong concordant degrade is *evidence for* MNTP, not
against.)

### 3.4 CONTINUE gate — Stage-2 MNTP fundable (INTERNAL spend gate; NOT a paper claim)

**Continue iff `mean paired Δacc ≥ +0.010` AND acc sign 3/3 positive on ≥ 1 protocol** — the minimum that justifies
spending the ~2–4 GPU-h/dataset on Stage-2 MNTP (self-supervised own-split; a CONDITIONAL FUTURE prereg, §8). Below
this gate but not KS-bidir-dead / DEGRADE = **weak-limbo**: MNTP **NOT funded** (banked "bidir moves too little to
justify MNTP"). Spend gate, not a goal-facing claim.

### 3.5 FORMAL verdict bar (goal-facing; the bar that matters for the user's substantial-gains mandate)

House **+0.030 acc AND +0.030 mF1** conjunct, **3/3 seeds positive**, under **EACH** protocol independently, **vs
the CAUSAL-LoRA arm** (§2.1/§2.2). Below the conjunct under a protocol → **NEGATIVE** on that protocol. Headline
claim ("bidir helps") requires FORMAL PASS on **≥ 2 datasets** under a stated protocol (rule 5). **D7 novelty
remains the user's ruling (F0.3) — even a FORMAL PASS is a performance result here; whether it counts toward the
"novel" clause is DEFERRED.**

### 3.6 Ladder summary (nested bars, one measurement per dataset)

`KS-bidir-dead (both protocols tie/regress → MNTP auto-defunded)` — with the `DEGRADE (≤ −0.014 both → "Llama-pattern,
MNTP-motivated", MNTP = user decision)` **branch** carved out of it — ⊂ `weak-limbo (some positive but < +0.010 / not
3/3 → MNTP not funded)` ⊂ `CONTINUE gate (≥ +0.010 acc, 3/3, ≥1 protocol → MNTP fundable)` ⊂ `FORMAL PASS (≥
+0.030/+0.030, 3/3, both protocols → goal-facing; ≥2 datasets = headline)`. Single test-touch (the 3 head reads)
resolves all bars.

### 3.7 Gate order

G-repro (patch self-test + extractor/head sha re-verify + shape sanity + Namespace-diff) → single test-touch (3
head reads/dataset) → KS-bidir-dead (+ DEGRADE branch) → CONTINUE gate → FORMAL bar (both protocols). The verdict
is rendered by an **independent 0-context reviewer against this prereg VERBATIM**; the executor transcribes raw
both-protocol per-seed numbers (line-numbered) and applies NO gates/interpretation.

---

## 4. G-repro + patch self-test + smoke plan + collision safety

### 4.1 G-repro discipline

- **(a) Patch self-test (LOAD-BEARING; CPU, $0 — already PASSED this prereg, §4.4.1).** `python
  src/utils/bidir_patch.py` builds a tiny random Qwen2.5-VL decoder on CPU and runs the **non-causality
  discriminator**: perturb a FUTURE token, measure the change at an EARLY token. Under causal the early token is
  invariant; under the patch it changes. Must print `VERDICT: PASS` (mask all-zeros non-None + `d_causal(pos0)`≈0 +
  `d_bidir(pos0)`>0). Re-run at submit.
- **(b) Extractor same-code gate.** `generate_VideoMLLM_embedding_bidir_HF.py` imports the causal extractor's
  operator VERBATIM; the causal extractor `generate_VideoMLLM_embedding_lora_HF.py` is **unchanged** (sha §5.2,
  re-verify at submit). The bidir runner's only behavioural difference from the causal path is the one
  `apply_bidir_mask` call (§1.1) — the reviewer confirms the diff is exactly the model-construction wiring + that
  one line.
- **(c) Head same-code as the banked causal arms.** `run_one` byte-identical to `enc3seed.sbatch` +
  `enc3seed_lora_curric.sbatch` (§4.2). The Namespace diff between a bidir head run and the banked causal control
  MUST be `--model` + derived-inert fields (`exp_comment`, `group_name`, `output_path`) ONLY — plus the inert
  argparse defaults already blessed by the encoder-swap / B3 / cand-2 verdicts.
- **(d) Extraction shape sanity (post-extraction, $0 CPU).** Before the head job, confirm each new `-bidir` `.pt`
  loads with `img_feats`/`text_feats` shape `(N, 3584)`, `N` = split size, labels present, finite (no all-zero rows
  beyond the extractor's zero-guard count).
- **(e) Banked causal controls re-paired from banked logs (§2), NOT re-run.**

### 4.2 Same-code + syntax verification (run this prereg — PASS)

- `run_one`…`PY` block of `enc3seed_bidir.sbatch` == `enc3seed.sbatch` **and** == `enc3seed_lora_curric.sbatch`:
  **BYTE-IDENTICAL** (`diff` empty, both).
- `bash -n` on `gen_embed_mllm_bidir.sbatch` and `enc3seed_bidir.sbatch` = **SYNTAX_OK**.
- `python -m py_compile bidir_patch.py generate_VideoMLLM_embedding_bidir_HF.py` = **OK**; the bidir runner's
  sibling-import chain resolves under env `HateVideo` (transformers 4.49.0).

### 4.3 Collision safety (verified this prereg — ABSENT; re-check at submit)

- `data/CLIP_Embedding/MHC_zh/*-LoRA-bidir_HF.pt`, `data/CLIP_Embedding/HateMM/*-LoRA-curric-bidir_HF.pt` — do NOT
  exist ⇒ fresh extraction; banked causal caches (`…-LoRA_HF`, `…-LoRA-curric_HF`) **untouched** (distinct out-tags).
- `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_bidir*` — do NOT exist ⇒ fresh group; `--force False` never trips the
  `run_rac.py:906-908` hard-abort; the `-bidir` model tag differs from CLIP/Qwen/LoRA/LoRA-curric regardless.
- `slurm/logs/enc3s_*bidir*_seed*_*.trainlog` — do NOT exist ⇒ no trainlog collision.
- The banked adapters `logging/lora/{MHC_zh,HateMM_curric}` are READ-ONLY (loaded + merged in-memory; never written).

### 4.4 Smoke plan (executor runs BEFORE the real submits; leave no artifact that trips §4.3)

1. **Patch self-test (CPU, $0 — LOAD-BEARING):** `python src/utils/bidir_patch.py` → must print `VERDICT: PASS`.
   *(Already PASSED this prereg under `HateVideo`: mask shape (1,1,6,6) all-zero; `d_causal(pos0)=0.000e+00`;
   `d_bidir(pos0)=6.4e-02`.)*
2. **Bidir extraction smoke (GPU, ~1–2 min):** `python src/utils/generate_VideoMLLM_embedding_bidir_HF.py --dataset
   HateMM --lora_dir logging/lora/HateMM_curric --num_frames 8 --splits test --limit 2 --out_model_tag _smoke_bidir
   --EXP_FOLDER logging/_smoke_bidir --device cuda` — confirm the `[BIDIR] mask-flip patch installed …` line prints,
   the masked-scatter assert (extractor L306) holds, it writes `logging/_smoke_bidir/HateMM/test_seen__smoke_bidir.pt`
   with `img_feats`/`text_feats` shape `(2, 3584)`, no OOM; **AND the real-model non-causality check**: assert
   `type(model.model).__name__ == "Qwen2_5_VLModel"` and that the bound `_update_causal_mask` returns an all-zeros
   non-None mask on a real forward (belt for the merge-preserves-`model.model` fact); then `rm -rf
   logging/_smoke_bidir`. (Redirected `--EXP_FOLDER` ⇒ never writes into `data/CLIP_Embedding/`.)
3. **mtime discipline (frame16):** after any smoke, confirm the banked causal `.pt` caches' mtimes are UNCHANGED.
4. **1-seed head smoke (optional):** on an existing causal LoRA cache, run ONE `run_rac.py` head with throwaway
   `--group_name _smoke` to confirm the align-fusion path loads + completes 30 epochs; delete the `_smoke` dir. If
   in doubt skip — the same-code guarantee (§4.2) + cache dims are CPU-verified.

### 4.5 MANDATORY REVIEWER CHECK — patch semantics against installed transformers source (write-in for the 0-context/codex review)

The independent review **MUST** verify, against the installed `transformers 4.49.0`
`modeling_qwen2_5_vl.py` in env `HateVideo`, that:
1. **No config flag exists** — `Qwen2_5_VLAttention.__init__` hard-codes `self.is_causal = True` (**:723**); there
   is no `is_causal`/`bidirectional` config field. A monkey-patch is mandatory.
2. **The SDPA re-causalization trap is defeated** — for a single unpadded sample `_update_causal_mask` returns
   **None** (`_ignore_causal_mask_sdpa`, **:1278**), whereupon `Qwen2_5_VLSdpaAttention.forward` sets
   `is_causal = True if causal_mask is None and q_len > 1 else False` (**:989**) and SDPA masks causally INTERNALLY.
   Confirm `_bidir_update_causal_mask` returns a **NON-None all-zeros** 4D mask so :989 evaluates `is_causal=False`
   and the zero additive bias imposes no masking ⇒ bidirectional (consumed as `attn_mask` at **:995**). **Nulling
   the mask is the trap; the patch must NOT null it.**
3. **The binding target survives the LoRA merge** — after `merge_and_unload`, `model` is the base
   `Qwen2_5_VLForConditionalGeneration` and `model.model` is the `Qwen2_5_VLModel` decoder (**:1519-1520**); confirm
   on the real merged 7B (smoke §4.4.2) that `apply_bidir_mask` binds to that decoder instance.
4. **The SDPA pin holds** — `apply_bidir_mask` asserts `model.model.config._attn_implementation == "sdpa"`; a silent
   flash fallback would re-causalize via `is_causal=self.is_causal` (**:904**), which the mask patch alone does not
   cover (the defensive `is_causal=False` loop covers it, but the assert is the guarantee).
5. **Vision untouched** — the patch binds only to `model.model`; the vision tower/merger (`model.visual`) already
   uses block-diagonal `cu_seqlens` attention (**:265-269**) and is not modified.

---

## 5. Artifacts authored this prereg + hash-freeze block

### 5.1 New artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current) |
|---|---|---|---|
| P | `refine-logs/BIDIR_STAGE1_PREREG.md` | **NEW** — this file | *(reviewer fills at freeze)* |
| A | `src/utils/bidir_patch.py` | **NEW** — the mask-flip patch (`_bidir_update_causal_mask` ~9 lines + `apply_bidir_mask` bind/sdpa-assert/`is_causal=False` + CPU `bidir_self_test`) | `36cedbac365b2b13c945adbe3437efdc61d8be15ecc85878eb9614225abe367b` |
| A2 | `src/utils/generate_VideoMLLM_embedding_bidir_HF.py` | **NEW** — thin bidir runner; imports the causal extractor's operator VERBATIM; inserts `apply_bidir_mask` after merge | `03f39e09c417bbea291f3c06b787f5220693568cd613705693df1c2bf23e020d` |
| B | `scripts/slurm/gen_embed_mllm_bidir.sbatch` | **NEW** — extraction; 2 hardcoded rows (ZH `logging/lora/MHC_zh`→`-LoRA-bidir_HF`; HateMM `logging/lora/HateMM_curric`→`-LoRA-curric-bidir_HF`); 8 frames | `0f17fce6910981bbc4c5942eae3b18947151bc6990ceee401fc86b252a287ecd` |
| C | `scripts/slurm/enc3seed_bidir.sbatch` | **NEW** — 3-seed head × 2 datasets; `run_one` BYTE-IDENTICAL to `enc3seed.sbatch`; `GROUP_NAME=RAC_video_bidir` | `82a69e74d570df59a1b686891814c7756b15755901d2a645bb1d3f0164a51264` |

### 5.2 Reused-unchanged machinery (verify sha at submit time; do NOT edit)

| path | role | sha256 |
|---|---|---|
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | causal extractor (operator imported VERBATIM; NOT edited) | `b6b61a3fa4214f28d2098ca305ca9a981934445afc81ccc5ac3939f8c1fb0ec6` |
| `scripts/slurm/enc3seed.sbatch` | same-code anchor for §4.2 (12850 runner) | `dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d` |
| `logging/lora/MHC_zh/adapter_config.json` | ZH adapter cfg (== 13150 arm; READ-ONLY) | `f9384d8dbdb8c1e315bb40a96952f068830c9a98cd6107f3b2ac2458e7fc477b` |
| `logging/lora/MHC_zh/adapter_model.safetensors` | ZH adapter weights (READ-ONLY) | `35a510f4ad84542c798939cfdb340b00317a5b8a2c670b07ced8d1869dd7b438` |
| `logging/lora/HateMM_curric/adapter_config.json` | HateMM curric adapter cfg (== 13241 arm; READ-ONLY) | `eaca36dd5cef2a4ff866a0398680d420adb157be815fe335500a387bbf9037b8` |
| `logging/lora/HateMM_curric/adapter_model.safetensors` | HateMM curric adapter weights (READ-ONLY) | `6571d132ef3218e4bdfcee98aab468df21f8aa83b16d623dd2098f8486394efa` |
| `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | ZH causal cache (paired floor; NOT clobbered) | *(present; verified untouched)* |
| `data/CLIP_Embedding/HateMM/{…}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` | HateMM causal cache (paired floor; NOT clobbered) | *(present; verified untouched)* |

### 5.3 Hash-freeze (to be filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file BIDIR_STAGE1_PREREG.md, after review>
A  36cedbac365b2b13c945adbe3437efdc61d8be15ecc85878eb9614225abe367b  src/utils/bidir_patch.py
A2 03f39e09c417bbea291f3c06b787f5220693568cd613705693df1c2bf23e020d  src/utils/generate_VideoMLLM_embedding_bidir_HF.py
B  0f17fce6910981bbc4c5942eae3b18947151bc6990ceee401fc86b252a287ecd  scripts/slurm/gen_embed_mllm_bidir.sbatch
C  82a69e74d570df59a1b686891814c7756b15755901d2a645bb1d3f0164a51264  scripts/slurm/enc3seed_bidir.sbatch
```
Executor re-runs `sha256sum` on A/A2/B/C (and this file) + confirms the causal-extractor sha `b6b61a3f…` and both
adapter shas unchanged at submit time; any mismatch = authorization VOID.

---

## 6. Single-submit / execution plan + resource plan

**Order (2 SLURM jobs, SEQUENTIAL, chained via `--dependency=afterok:`):**

1. `sbatch scripts/slurm/gen_embed_mllm_bidir.sbatch` → extracts BOTH `-bidir` caches (ZH then HateMM; ~0.5–0.7
   GPU-h). Gate: patch self-test (§4.4.1, CPU) + bidir extraction smoke (§4.4.2) BEFORE this real submit; on
   COMPLETE apply the §4.1d shape sanity + §4.4.3 mtime discipline.
2. `sbatch --dependency=afterok:<1> scripts/slurm/enc3seed_bidir.sbatch` → 6 bidir head runs, ~2 min. Produces
   `slurm/logs/enc3s_{MHC_zh,HateMM}_…-bidir…_seed{0,1,2}_<JID>.trainlog`.

**Resource plan (STANDING INFRA RULE compliant):** each sbatch requests **`--cpus-per-task=8`, `--mem=64G`,
1×A100** (inherited from `gen_embed_lora.sbatch` / `enc3seed.sbatch`; verified). The head has an
`afterok:<extraction>` dependency ⇒ **the two jobs never run concurrently**; peak footprint is **8 CPU / 64 G / 1
GPU** — within the 16 CPU / 128 G / 2 GPU cap, and **NEVER two 16-CPU jobs in flight** (the 29h-wedge rule). `conda
activate HateVideo`; `sbatch` with **NO `--time`**; initial `PENDING (JobHeldUser)` = **WAIT for auto-release,
never force** (CLAUDE.md). Both sbatch source `conda.sh` directly and run the ≥20 G `disk_guard.sh`.

**Sequencing vs the parallel readout-recon cell (recon §6 + task):** bidir is its OWN single arm on the baseline
readout — **NO bidir×readout grid**. Submit this chain **only after the parallel readout-recon cell's chain has
cleared the queue, OR behind it in a dependency chain** — keep total in-flight ≤ 16 CPU (never two 16-CPU jobs;
these are 8-CPU each, so at most one other 8-CPU job may co-run).

**Cost ledger:** Stage-1 total ~**0.5–0.7 GPU-h** (extraction dominates; head ~0.03 h). Conditional future (NOT this
submit): Stage-2 MNTP ~2–4 GPU-h/dataset (self-supervised own-split; funded only per §3.4/§3.3). $0 CPU: all floor
re-derivation + shape sanity + the patch self-test.

**Test-touch:** the Stage-2 head reads are the ONLY budgeted bidir-encoder test evaluations (one per dataset = 3
head-seed reads); zero test-touch before the verdict. **No job is submitted by this prereg author.** Submission
happens only after the independent 0-context review + hash-freeze + the F0.3 GO-IF gates (codex patch review + D7
user sub-ruling), run by the orchestrator.

---

## 7. Outcome table template (filled ONLY from raw trainlogs at verdict time)

### 7.1 ZH — bidir vs CAUSAL-LoRA floor 13150 (fill from `enc3s_MHC_zh_…-LoRA-bidir_HF_seed{0,1,2}_<JID>.trainlog`)

| seed | protocol | bidir acc/F1 | causal-LoRA acc/F1 (§2.1) | Δ(bidir−causal) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8322/0.8023 | ___ |
| 1 | val-sel | ___ | 0.8255/0.7956 | ___ |
| 2 | val-sel | ___ | 0.8389/0.8065 | ___ |
| **mean** | **val-sel** | ___ | **0.8322/0.8015** | **___** |
| 0 | final-ep | ___ | 0.8456/0.8181 | ___ |
| 1 | final-ep | ___ | 0.8389/0.8113 | ___ |
| 2 | final-ep | ___ | 0.8523/0.8226 | ___ |
| **mean** | **final-ep** | ___ | **0.8456/0.8173** | **___** |

### 7.2 HateMM — bidir vs CAUSAL-curric floor 13241 (fill from `enc3s_HateMM_…-curric-bidir_HF_seed{0,1,2}_<JID>.trainlog`)

| seed | protocol | bidir acc/F1 | causal-curric acc/F1 (§2.2) | Δ(bidir−causal) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8791/0.8730 | ___ |
| 1 | val-sel | ___ | 0.8744/0.8678 | ___ |
| 2 | val-sel | ___ | 0.8791/0.8724 | ___ |
| **mean** | **val-sel** | ___ | **0.8775/0.8711** | **___** |
| 0 | final-ep | ___ | 0.8791/0.8730 | ___ |
| 1 | final-ep | ___ | 0.8791/0.8724 | ___ |
| 2 | final-ep | ___ | 0.8791/0.8724 | ___ |
| **mean** | **final-ep** | ___ | **0.8791/0.8726** | **___** |

### 7.3 Fixed write-up format (per §3.1 rule 5 + the ladder §3.6)

`ZH (bidir vs causal-LoRA):      final-epoch: <pass/fail>; val-selected: <pass/fail>  [FORMAL bar §3.5].`
`HateMM (bidir vs causal-curric): final-epoch: <pass/fail>; val-selected: <pass/fail>  [FORMAL bar §3.5].`
`KS-bidir-dead: <KILLED (MNTP auto-defunded) | survives> per dataset.  DEGRADE: <Llama-pattern MNTP-motivated | n/a>.`
`CONTINUE gate (§3.4): <cleared → MNTP fundable | not cleared → banked> per dataset.`
`Headline (≥2 datasets, one protocol): <met/NOT met>.  D7 novelty: DEFERRED to user.`
(+ MARGINAL note if a within-noise pass per B3 §2.2 precedent.)

---

## 8. What a PASS / FAIL means for the goal (D7 boundary — DEFERRED to the user)

- **KS-bidir-dead / FLAT (recon prior — the MOST likely outcome, F0.7):** the topology change carries no new
  convertible signal through the encoder+pool ⇒ the bidir cell is **CLOSED** and **Stage-2 MNTP auto-defunded** on
  that dataset. The strongest clean outcome: F35 converted from a passive diagnosis into an **actively-measured
  named mechanism** at ~0.6 GPU-h — paper value even on a kill.
- **DEGRADE / "Llama-pattern, MNTP-motivated" (§3.3):** bidir breaks the causally-trained adapter ⇒ Stage-1 perf-dead
  but Stage-2 MNTP becomes a **user funding decision** (the LLM2Vec Llama-precedent's designed repair), not
  auto-killed.
- **CONTINUE-gate cleared (≥ +0.010 acc, 3/3, ≥1 protocol):** bidir moves enough to **fund** Stage-2 MNTP (a
  CONDITIONAL FUTURE prereg; self-supervised own-split; ~2–4 GPU-h/dataset). A spend decision.
- **FORMAL PASS (≥ +0.030/+0.030, 3/3, both protocols; ≥2 datasets = headline):** a goal-facing performance result.
  **Whether removing the causal mask counts toward the goal's "novel" clause is the USER's D7 ruling (F0.3), DEFERRED
  — this prereg does NOT decide it.** Caveat: single deterministic extraction (F0.2), single test-touch per dataset.

**Framing sentence (verbatim):** *this measurement tests the one axis every banked Qwen arm shares and none ever
varied — the LLM decoder's causal attention topology — by flipping is_causal to bidirectional at inference on the
SAME banked adapters, SAME readout, SAME 8 frames; it decides the performance clause against the CAUSAL arm, its
most likely outcome is a clean Law-I kill that measures the F35 mechanism, and D7 novelty stays the user's ruling.*

---

## 9. Provenance index

- Recon (GO-IF; stage-1 design, mechanics, cost, kill skeleton): `refine-logs/BIDIR_SURGERY_FORENSIC_RECON.md` (`ec5add8`).
- Cell source / litsurvey: `refine-logs/LITSURVEY_MLLM_EMBEDDING.md` §B3 (b807722); F68-P3; F35 diagnosis.
- Mechanics (verified this prereg vs installed `transformers 4.49.0`): `modeling_qwen2_5_vl.py:723` (is_causal
  hard-coded), `:989/:995` (SDPA is_causal trap + attn_mask), `:1244-1325` (`_update_causal_mask`), `:1278`
  (`_ignore_causal_mask_sdpa` None trap), `:904` (flash is_causal), `:265-269` (vision block-diagonal),
  `:1519-1520` (model.model / model.visual).
- Extractor / operator: `src/utils/generate_VideoMLLM_embedding_lora_HF.py:301` (single forward + hidden_states),
  `:306` (masked-scatter assert), `:313-341` (`_encode` prefix/response spans), `:419-441` (base load + LoRA merge).
- Loader routing: `src/data_loader/dataset.py:499-503` (MHC_zh/HateMM → `load_feats_MHC`), `:605-608`
  (`{split}_{model}.pt`).
- Paired floors (re-derived §2): ZH 13150 `enc3s_MHC_zh_…-LoRA_HF_seed{0,1,2}_13150.trainlog`; HateMM 13241
  `enc3s_HateMM_…-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`; context CLIP ZH 13115 / HateMM 12850. Cross-validated
  vs `exp-lora-zh-b3.md` (B3 verdict) + `FRAME16_PREREG.md §2` (HateMM-CLIP).
- Protocol + decision rule (verbatim): `research-wiki/experiments/exp-encoder-3seed.md:73-85`.
- Ban scopes (no collision): F24 / F51 / P9b / F35–F39 — recon §4 + `findings.jsonl`.
- House precedent: `refine-logs/FRAME16_PREREG.md` (`0b5cbb5`), `refine-logs/VISION_UNFREEZE_PREREG.md`.

**Required statements:** ZERO GPU/SLURM/Modal spent by this prereg author (only pure-CPU login-node floor re-parsing,
collision/syntax/same-code verification, and the CPU patch self-test on a tiny random model — seconds each; no
held-out test metric produced). All floor numbers re-parsed from banked completed-run trainlogs (numeric-provenance
discipline). No `autoresearch/.../state/` mutated. No `research-wiki/` mutated. NO job submitted. DRAFT awaiting
independent 0-context review + hash-freeze + the F0.3 GO-IF gates. Not pushed.

---

## 10. (reserved)

---

## 11. DEVIATIONS FROM THE RECON — flagged loudly

1. **DEV-1 (the "$0 dev screen" is FOLDED INTO the verdict — screen replaced by the full 3-seed test's KS bar).
   MATERIAL — pinned to the task + the frame16 house pattern, over the recon's wording.** Recon §2 proposed a
   train+dev-only head screen (ZERO test-touch) as the cheap kill, with the full 3-seed test only on a MOVE. The
   task's binding instruction is to **mirror frame16 exactly**: run the full 3-seed head (which reads test),
   transcribe raw both-protocol numbers, and let the **0-context verdict's KS-bidir-dead bar BE the screen** (§3.2).
   Rationale: a dev-only screen would need a modified head run (the `enc3s` runner prints test lines), whereas the
   house n=3 pattern (frame16) ran the full head and let the verdict rule; this keeps the runner byte-identical to
   the banked controls. Consequence: this prereg spends **one test-touch per dataset** (declared F0.1), NOT zero.
   The qualitative screen (FLAT ⇒ kill; DEGRADE ⇒ Llama-pattern; MOVE ⇒ CONTINUE) is unchanged — only its footing
   moves from dev to the pre-registered test verdict.

2. **DEV-2 (a dedicated bidir RUNNER that imports the causal operator VERBATIM, instead of editing the banked
   extractor or a raw instance monkey-patch). MATERIAL / provenance-favorable.** The recon §1.3 pins an instance
   bind `model.model._update_causal_mask = types.MethodType(...)`. Editing
   `generate_VideoMLLM_embedding_lora_HF.py` in place (e.g. a `--bidir` flag) would change the sha of a file other
   frozen preregs pin as "reused-unchanged" (vision-unfreeze §5.2), risking their authorization-void checks.
   Following the house FORK discipline (frame16 DEV-2, vision-unfreeze DEV-3), I authored a NEW thin runner
   `generate_VideoMLLM_embedding_bidir_HF.py` that **imports the causal extractor's pooling operator VERBATIM** and
   inserts exactly one `apply_bidir_mask(model)` call after the merge. The banked extractor is byte-unchanged; the
   patch semantics are identical to the recon's instance bind (`apply_bidir_mask` does the same `types.MethodType`
   bind on `model.model`), plus a **belt-and-suspenders**: an SDPA assert (recon §1.5 recommendation) and an
   `is_causal=False` loop over decoder attention modules (covers the flash trap the recon flags, harmless for SDPA).

3. **DEV-3 (KILL bar uses SIGN, not "bootstrap CI straddles 0"). Neutral — house discipline.** The recon phrases the
   FLAT kill loosely; I pin the **sign-based** bar (`mean paired Δacc ≤ 0 OR sign not 3/3, both protocols`, §3.2),
   consistent with the house n=3 no-bootstrap rule (`exp-encoder-3seed.md:78-79`) and frame16 DEV-1. Only the
   significance formalism changes; the qualitative bar (tie/regress both protocols ⇒ dead) is identical.

4. **DEV-4 (HateMM paired anchor = the CURRIC arm 13241, not the generic-LoRA arm). Per the task; documented.** The
   recon §2 lists HateMM's control loosely as "curric adapter, hold." The task pins the HateMM causal anchor as job
   **13241** (curric) — the SAME adapter the bidir HateMM arm re-extracts. This makes the pairing clean (same
   adapter, mask flipped) but sets a HIGH floor (0.8775 val-sel = project best), so the HateMM FORMAL bar is very
   demanding (F0.5); ZH (generic-LoRA 13150) is the live perf surface. The frozen-CLIP floors (12850/13115) are
   carried as orientation-only context (§2.3), NOT the formal bar.

5. **DEV-5 (Stage-2 MNTP is a CONDITIONAL FUTURE prereg; DEGRADE does NOT auto-defund it — a distinct user
   decision). Per the task; documented.** The recon §3/§5.1 makes Stage-2 conditional on the Stage-1 outcome shape.
   The task refines the funding logic: FLAT/KS-bidir-dead **auto-defunds** MNTP (Law-I), but a strong concordant
   DEGRADE (≤ −0.014 both protocols) is relabeled **"Llama-pattern, MNTP-motivated"** and leaves MNTP as a
   **SEPARATE user-visible funding decision** (§3.3), not auto. No Stage-2 / MNTP artifact is authored or submitted
   here.

6. **DEV-6 (D7 novelty is DEFERRED to the user, not asserted). Per the task; documented.** The recon (§5.4) argues
   bidir is the highest-D7-payoff cell and cites the F65 precedent for measuring it. This prereg carries that as
   **motivation** but decides the **performance clause only**; whether a pass counts toward the goal's "novel"
   clause is the USER's D7 ruling, DEFERRED (F0.3). The recon's GO-IF gates (codex patch review + one-line D7 user
   sub-ruling before GPU) are flagged as orchestrator pre-conditions the author does NOT clear.
