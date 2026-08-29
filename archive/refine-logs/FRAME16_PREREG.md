# FRAME16 Pre-Registration — frozen-Qwen-16f vs banked frozen-Qwen-8f, RGCL head, HateMM (stage-1 only)

**Author:** frame-budget prereg author (CPU-only; no GPU/SLURM/Modal spent; NO job submitted).
**Date:** 2026-07-21 NZST.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced; no job submitted.
**Implements:** `refine-logs/FRAME_BUDGET_FORENSIC_RECON.md` (commit `ec601a6`, the GO-IF recon) — its **stage-1**
design (frozen-Qwen-16f HateMM, extraction + head only, single-variable), mechanics, cost ledger, and
kill-bar skeleton transcribed and re-verified below. Deviations from the recon are flagged **loudly** in §11.
**House-style precedent:** `refine-logs/LORA_HATEMM_PREREG.md`, `refine-logs/VISION_UNFREEZE_PREREG.md`
(binding language, F0.x honesty clauses, pinned pipeline, re-derived floors, freeze block, single-submit
plan, outcome-table template); `research-wiki/experiments/exp-encoder-3seed.md` (the 12850 encoder-swap
protocol + decision rule verbatim :73-85).

## Title + claim scope (verbatim)

> This measurement tests **the one axis 8-frame sampling never varied — visual sampling density (16 frames vs
> the hard-coded 8)** — through the **frozen** Qwen2.5-VL-7B encoder + mean-pool, on **HateMM** (the one dataset
> whose image stream is healthy enough to convert added coverage). It is a **PERFORMANCE lever + a door-closer**
> that converts a currently prose-argued gap ("denser frames untested") into a measured-and-closed result. It
> makes **NO novelty claim — sampling density is an engineering knob (D7-DEAD, F0.3)**; even a formal PASS is a
> performance/ablation row, never a novelty contribution. This prereg decides the **performance clause only**,
> and is the **CHEAP, SINGLE-VARIABLE, DECISIVE stage-1** gate that governs whether the expensive contaminated
> LoRA-16f follow-up (stage-2) is ever funded.

The cell under test is the **frozen** Qwen2.5-VL-7B encoder run at **16 frames** (one changed arg,
`--num_frames 8 → 16`; byte-identical pooled `img_feats`/`text_feats` operator), features fed to the standard
archive-OFF RGCL align-fusion head + top-20 kNN (`enc3s`/12850 protocol), paired **3-seed within head-seed**
vs the banked **frozen-Qwen-8f** floor (job 12850), dual-protocol (val-selected AND final-epoch), on **HateMM,
trained ONLY on its own train split**. **ZH stage-1.5 and LoRA-16f stage-2 are CONDITIONAL FUTURE preregs, NOT
authored or submitted here; 32f is declared OUT of this prereg (multiplicity, §3.6).**

---

## 0. Binding facts / honesty clauses (all present; pre-declared)

**F0.1 — Test is NOT virgin (declared).** HateMM test was already read, under the identical `enc3s` protocol, by:
the **frozen-CLIP** and **frozen-Qwen-8f** arms (job 12850), the **generic-LoRA** arm (job 13235, F53), the
**LoRA-HateMM verdict** (`LORA_HATEMM_VERDICT_REVIEW.md`), and **cand-2 curriculum**. This prereg's frozen-16f
head reads are **re-measurements under the identical protocol**, not first exposures. They consume exactly ONE
budgeted **frozen-16f-encoder** test evaluation = the **3 head-seed reads**. **Zero test-touch before the
independent verdict.**

**F0.2 — Single-extraction "draw" — and why NO draw-variance caveat applies (pre-declared; STRONGER than the
LoRA F0.2).** The 3 head-seeds read ONE frozen-16f extraction per dataset. **Unlike the LoRA arms** (whose F0.2
warns of single-SFT-draw luck), the frozen forward is **fully deterministic given (weights, sampled frame
indices, max_pixels)**: frame sampling is `np.linspace(0, N−1, 16)` (deterministic, no RNG), `attn=sdpa`,
`bf16`, `no_grad`, single forward — there is **no stochastic encoder draw at all**. The reported ±band is
**purely head-seed variance**, exactly symmetric with the banked frozen-8f floor (also 3 head-seeds over one
deterministic extraction). **So the single-draw caveat that burdens every LoRA cell does NOT exist here** — the
16f-vs-8f comparison is a clean head-seed-paired test with no hidden extraction-luck confound.
*Sub-caveat (deterministic, conservative):* videos with <16 decodable frames yield duplicated `linspace`
indices (graceful; those clips degenerate toward the 8f-equivalent — recon §2 Finding B), which can only make
16f resemble 8f more, i.e. it **biases against** finding a 16f effect (a conservative bias, not a
false-positive risk); no crash — the masked-scatter invariant `assert last_hidden.shape[0]==input_ids.numel()`
(extractor L283) holds at any even frame count.

**F0.3 — Novelty = D7-DEAD, say it plainly (not a user-pending boundary — it is dead).** Visual sampling
density is **how many frames to feed a fixed encoder+pool** — an **engineering knob**, NOT an MLLM-novelty
mechanism. **Novelty-nil / D7-DEAD** (recon §6): even a formal PASS is a **performance/ablation row**
("frame-budget: 16f vs 8f"), same D7 class as C4 (head-eng) and C5 (recipe) — **never** a novelty contribution.
This is a door-closer + robustness ablation, not a goal-reacher. (Contrast the LoRA/vision cells, whose D7 is a
*user ruling*; this one is not — density is dead on arrival.)

**F0.4 — Single-dataset own-train-split VETO compliance (hard user veto; trivially clean — there is no
training).** Stage-1 has **no SFT**: the frozen extractor reads only each video + the two FIXED instructions
(`IMG_INSTRUCTION`/`TEXT_INSTRUCTION`, extractor L45-52) — **no labels enter the encoder path**. The RGCL head
trains on **HateMM's own train split only** (identical to the 8f floor). NO cross-dataset mixing, NO gold
spans/attributes, NO OCR channel, raw videos never leave the machine. All standing vetoes cleared.

**F0.5 — Honest prior is LOW-MODEST, and two banked mechanisms LOWER it; none raise it (pre-declared,
material).** Recon §4: no $0 pre-gate exists (frozen-16f `img_feats` cannot be derived from 8f caches — a new
forward is required), and no banked evidence RAISES the prior. Two lower it: **(a)** F37/F35 cumulative-causal
redundancy — on these Qwen representations the pooled mean over a denser prefix grid is partly redundant with
the running causal summary (late frames already summarise early ones); **(b)** mean-pool dilution — a single
hateful frame among 16 contributes half the weight it does among 8, so denser sampling raises coverage (↑) but
lowers per-event pooled contribution (↓); the two partly cancel and the **net sign is genuinely unknown**,
which is exactly why it must be measured. Red-team §0 fact-1: the missing-dataset bottleneck (EN) is encoder
collapse, not frame count. **Net honest prior: LOW-MODEST ~8-12% for ≥+1pt on any dataset** (recon VERDICT; not
revised up). HateMM is chosen because its image stream is the healthiest (train-LOO img AUC: CLIP 0.836 /
frozen-Qwen 0.820, red-team §0), so it is the only dataset with a real conversion surface for added coverage;
EN is mechanistically mis-aimed (collapsed vision tower) and deprioritized; ZH is a conditional stage-1.5.

**F0.6 — Ban-scope check: SAME operator class as the deployed method, NO ban collision (verified vs
F35/F37/F39 wording, recon §5).** The pooled object is unchanged — mean of last-layer hidden states over the
prefix span (img) / trailing span (text), L2-normed (extractor `_encode` L290-322). None of the temporal-family
bans reach it: **F37** killed the retrieval-object / don't-pool / set-matching family over the 8 frame groups
(the *pooled* side is the survivor — this cell IS the pooled object); **F39 (CTF)** killed the supervised
temporal-pool of the `[g_1..g_T]` frame-group tensor as a key (a different object frozen-16f never forms);
**F35** is a causal-prefix *mechanism note*, not a kill, and concerns how the 8 groups relate, not whether
denser sampling covers more of the video. ⇒ No ban collision (matches recon C2(b)). The 16f arm never forms a
frame-group tensor, never does set-matching — it is the deployed mean-pool operator with a denser frame grid.

---

## 1. Pipeline spec — fully pinned (2 stages; no SFT; nothing left to interpretation)

**Stage 0 — none.** There is no SFT / no data build: the frozen extractor consumes `data/gt/HateMM/{train,val,
test}.jsonl` + `data/video/HateMM/All/<id>.mp4` directly (both verified present this prereg). No `dataset_info`
registration, no `lora_sft` data.

### 1.1 Stage 1 — frozen-Qwen-16f feature extraction (ONE changed arg; NO extractor code edit)

- **Submit:** `sbatch scripts/slurm/gen_embed_mllm_16f.sbatch HateMM` (authored this prereg — artifact B, §5).
- **What changes vs the 8f cache:** exactly **one variable**, `--num_frames 8 → 16`, in the frozen extractor
  `src/utils/generate_VideoMLLM_embedding_HF.py` — an **already-existing argparse arg** (L90-95, default 8);
  **no code edit**. The pooled operator is byte-for-byte the same (`_encode` L254-323). The out-tag is set to a
  **DISTINCT** value `--out_model_tag Qwen2.5-VL-7B-Instruct_HF-16f` (also a pre-existing arg, L84-89) so the
  banked 8f cache `..._HF.pt` is **never** clobbered.
- **No cutoff wall, no VRAM concern (recon §2 Finding A).** The extractor passes no `max_length`/`truncation`;
  `model(**inputs, output_hidden_states=True)` is a single forward. At 16f the sequence is ~1440-1536 visual
  tokens + text (vs ~720-768 at 8f) — trivial for a 7B on the A100 that already runs 8f. `--num_frames 16` is
  even ⇒ `grid_t=8`, clean. **This is why stage-1 needs ZERO second changed variable** (the `cutoff_len=4096`
  wall lives only in the SFT builder, which stage-1 does not touch — that is stage-2's contamination, §3.6/§8).
- **Output:** `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF-16f.pt`
  (loader contract: `{ids, img_feats, text_feats, labels}`, Dv=Dt=3584; consumed by
  `src/data_loader/dataset.py:load_feats_MHC` — HateMM routes there, dataset.py:499-503, verified this prereg).
- **Cost:** ~0.4-0.6 GPU-h (HateMM 1066 videos; 16f ≈ 1.5-2× the banked 8f anchor 949.1 s from
  `S2S_EXTRACTION_RECORD.md`; upper bound generous — recon §6). Then B2-push (derived `.pt` only; videos never
  leave — CLAUDE.md data boundary).

### 1.2 Stage 2 — 3-seed RGCL align-fusion head + kNN (paired vs the banked frozen-Qwen-8f floor)

- **Submit:** `sbatch scripts/slurm/enc3seed_fb16.sbatch` (authored this prereg — artifact C, §5).
- **What it runs:** 3 head-only runs (features cached, ~20-25 s each): HateMM-16f seeds 0/1/2,
  `--model Qwen2.5-VL-7B-Instruct_HF-16f`, `--group_name RAC_video_fb16`, `--force False`.
- **CRITICAL same-code guarantee (verified this prereg — §4.2):** the `run_one`…`PY` block of
  `enc3seed_fb16.sbatch` is **BYTE-IDENTICAL** to `enc3seed.sbatch` (`diff` empty) — the SAME runner that
  produced BOTH the frozen-CLIP and the frozen-Qwen-**8f** floors (job 12850). The **ONLY** manipulated
  variables vs the banked 8f floor are `--model` (`…_HF` → `…_HF-16f`, i.e. the 16f feature cache) and
  `--group_name`. Config verbatim: `--batch_size 64 --lr 0.0001 --epochs 30 --topk 20 --proj_dim 1024
  --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align --hard_negatives_loss True --no_hard_negatives 1
  --metric cos --loss triplet --hybrid_loss True --warmup 5 --lambda_seg 0 --archive OFF`. Identical to
  `exp-encoder-3seed.md` H1 / LoRA-HateMM / cand-2 / vision-unfreeze.
- **Pairing:** per head-seed (16f-seed s − 8f-seed s), `s ∈ {0,1,2}`. `--seed` controls head-init + data-shuffle;
  the only difference between treatment and the 8f floor at a given seed is the feature cache (16f vs 8f). A
  clean within-seed paired increment on the **already-banked frozen-Qwen encoder swap** — this is 16f-vs-8f,
  **NOT** vs CLIP.
- **Output:** `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF-16f_seed{0,1,2}_<JID>.trainlog`. Cost ~1.5 min.

**Total NEW GPU: ~0.5-0.7 A100-h** (extraction dominates; head ~0.03 h). No SFT, no cutoff, no ban collision.

---

## 2. Comparison floor — INDEPENDENTLY RE-DERIVED from raw 12850 trainlogs (numeric-provenance discipline)

Every number below was independently re-parsed **this prereg** from the raw
`slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_12850.trainlog` with the EXACT `enc3seed.sbatch`
embedded parser (val-sel = epoch ≥ warmup 5 max `Val_Retrieval` acc, roc tie-break; final = max epoch). The
final-epoch acc **mean bit-matches 0.8682** (recon §5 / F53 KS-2 line), and every value matches
`LORA_HATEMM_VERDICT_REVIEW.md` §2.1 / `exp-encoder-3seed.md` to 4dp — no discrepancy.

### 2.1 frozen-Qwen-8f HateMM floor (the paired anchor; treatment = frozen-Qwen-16f, delta = 16f − 8f)

| protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | 3-seed mean acc/F1 |
|---|---|---|---|---|
| **val-sel** (sel ep 28/22/29) | 0.8698/0.8606 | 0.8651/0.8586 | 0.8837/0.8753 | **0.8729 / 0.8648** |
| **final-ep** (ep 29) | 0.8605/0.8507 | 0.8605/0.8514 | 0.8837/0.8753 | **0.8682 / 0.8591** |

(seed2 val-sel selects ep29 = the final epoch, so its two protocol rows coincide.) Provenance (file:line, re-read
this prereg): `enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog:293`(val e28)/`:303`(final e29);
`…_seed1_…:235`(val e22)/`:299`(final e29); `…_seed2_…:302`(val=final e29). Cross-checked bit-exact against
`exp-encoder-3seed.md:155-159`.

### 2.2 Context (NOT a paired anchor — for orientation only)

frozen-CLIP HateMM floor (the 8f-vs-CLIP encoder-swap that already PASSED both protocols): val-sel 0.8202/0.8085,
final 0.8124/0.7936 (`exp-encoder-3seed.md`; ERRATUM 66012e9). The frozen-Qwen-8f→CLIP gain was +0.053-0.056 acc
3/3 both protocols. **This cell does NOT re-pair vs CLIP** — it isolates the *sampling-density* increment on top
of the banked frozen-Qwen encoder, so the only anchor is §2.1.

---

## 3. Decision rule + kill-bars (paired, both protocols judged independently, 3/3 sign, pre-declared)

### 3.1 Decision rule — verbatim from `exp-encoder-3seed.md:73-85` (treatment = frozen-Qwen-16f; delta = 16f − 8f)

> For each dataset × protocol: (1) per-seed paired difference δ = (treatment − control) for acc and macro-F1 at
> seeds 0/1/2; (2) 3-seed mean ± std + sign consistency (how many of 3 positive); (3) n=3 too small for a
> bootstrap — report the paired-t **as an effect-size descriptor only**, no significance claim; (4) **pass =
> mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive**; (5) headline claim
> requires pass on ≥ 2 datasets under a stated protocol; both protocols judged separately; verdict written
> exactly "final-epoch: pass/fail; val-selected: pass/fail".

Both protocols judged **independently** (no protocol-shopping, no metric-shopping). Control = frozen-Qwen-8f (§2.1).

### 3.2 KS-16f-dead — the KILL bar (auto-kills the LoRA-16f follow-up)

**KILL iff, on BOTH protocols, the 16f arm ties-or-regresses the 8f floor** — i.e. under each protocol
`mean paired Δacc ≤ 0` **OR** the acc sign is not 3/3 positive (so **neither** protocol produces a clean
positive-mean-and-3/3-sign result). Then: **the frozen-16f cell is KILLED, AND the expensive LoRA-16f stage-2
is AUTO-DEAD** (banked, never run) — you cannot re-SFT your way into information the frozen forward proves
denser frames do not carry through the encoder+pool (recon §1/§5). State this explicitly at verdict time.

### 3.3 CONTINUE-to-stage-2 gate (INTERNAL spend gate — NOT a paper claim)

**Continue iff** frozen-16f `mean paired Δacc ≥ +0.010` **AND** acc sign 3/3 positive on **≥ 1 protocol** — the
minimum that would justify spending the ~4-6 GPU-h/dataset on the contaminated LoRA-16f stage-2 (a **CONDITIONAL
FUTURE prereg**, §8). Below this gate (but not KS-16f-dead) = a measured weak-positive limbo: **LoRA-16f is NOT
funded** (banked as "16f moves too little to justify the ≥2-variable follow-up"). This is a spend gate, not a
goal-facing claim.

### 3.4 FORMAL verdict bar (goal-facing, paper-worthy frame-budget effect)

House **+0.030 acc AND +0.030 mF1** conjunct, **3/3 seeds positive**, under **BOTH** protocols vs the banked 8f
floor (§2.1) — identical to the encoder-swap criterion. Below the conjunct under a protocol → **NEGATIVE** on
that protocol. **D7-DEAD (F0.3): even a formal PASS is an engineering / ablation row ("16f vs 8f"), NEVER a
novelty win** — it converts a prose gap into a measured robustness result, nothing more.

### 3.5 Ladder summary (three nested bars, one measurement)

`KS-16f-dead (both protocols tie/regress)` ⊂ `weak-limbo (some positive but < +0.010/not-3-sign)` ⊂
`CONTINUE gate (≥ +0.010 acc, 3/3, ≥1 protocol → LoRA-16f funded)` ⊂ `FORMAL PASS (≥ +0.030/+0.030, 3/3, both
protocols → engineering row)`. Single test-touch (the 3 head reads) resolves all four.

### 3.6 Multiplicity + scope of THIS submit (pre-declared)

- **16f is the single PRIMARY arm** (2× is the minimal density step). **32f is declared OUT of this prereg** — a
  {16,32} sweep uncorrected is a forking path; 32f would be a separate pre-declared arm gated on 16f moving.
- **ZH stage-1.5** (frozen-16f MHC_zh + head) and **LoRA-16f stage-2** (SFT `NUM_FRAMES 8→16` **plus** forced
  `cutoff_len 4096→~8192` — inherently **≥2 changed variables**, ~4-6 GPU-h/dataset) are **CONDITIONAL FUTURE
  preregs, NOT authored or submitted here.** Their gate is §3.3 (CONTINUE). ZH would additionally require its
  own recon/prior (off-mechanism: ZH is text-borne, F45).

### 3.7 Gate order

G-repro (extractor/head sha re-verify + shape sanity + Namespace-diff) → single test-touch (3 head reads) →
KS-16f-dead → CONTINUE gate → FORMAL verdict bar (both protocols). The verdict is rendered by an **independent
0-context reviewer against this prereg VERBATIM**; the executor transcribes raw both-protocol per-seed numbers
(line-numbered) and applies NO gates/interpretation.

---

## 4. G-repro + smoke plan + collision safety

### 4.1 G-repro discipline

- **(a) Extractor same-code + no-edit gate.** `generate_VideoMLLM_embedding_HF.py` is **unchanged** (sha §5.2);
  `--num_frames`/`--out_model_tag` are pre-existing args (L84-95) — the executor re-verifies the sha at submit
  and confirms NO code edit was made. `gen_embed_mllm_16f.sbatch` differs from its fork source
  `gen_embed_mllm.sbatch` ONLY in the header comment, `job-name`, the echo line, and the two python-arg lines
  (`--num_frames 16` hardcoded + `--out_model_tag Qwen2.5-VL-7B-Instruct_HF-16f`) — verified this prereg (§4.2).
- **(b) Head same-code as 12850 (INCLUDING the 8f floor).** The `run_one`…`PY` block of `enc3seed_fb16.sbatch`
  is BYTE-IDENTICAL to `enc3seed.sbatch` (§4.2), the exact runner that produced the frozen-Qwen-8f floor. The
  Namespace diff between a 16f head run and the banked 8f control MUST be `--model` + derived-inert fields
  (`exp_comment`, `group_name`, `output_path`) ONLY — plus the inert argparse defaults already blessed by the
  encoder-swap / LoRA-HateMM verdicts (proven no-op by the 12850 bit-exact seed0 reproductions,
  `exp-encoder-3seed.md:126-146`).
- **(c) Extraction shape sanity (post-extraction, $0 CPU).** Before the head job, confirm each new
  `{split}_Qwen2.5-VL-7B-Instruct_HF-16f.pt` loads with `img_feats`/`text_feats` shape `(N, 3584)`, `N` = split
  size (744/107/215), labels present, finite (no all-zero rows beyond the extractor's own zero-guard count).
- **(d) Frozen-Qwen-8f control re-paired from banked 12850 logs (§2.1), NOT re-run.**

### 4.2 Same-code + syntax verification (run this prereg — PASS)

- `run_one`…`PY` block of `enc3seed_fb16.sbatch` == `enc3seed.sbatch`: **BYTE-IDENTICAL** (`diff` empty).
- Full-file `diff` of `gen_embed_mllm_16f.sbatch` vs `gen_embed_mllm.sbatch`: header comment, `job-name`
  (`mllm_embed_16f`), echo line, and the python args (`--num_frames 16` + `--out_model_tag
  Qwen2.5-VL-7B-Instruct_HF-16f`) only — the fork adds the out-tag and hardcodes 16 frames; nothing else.
- `bash -n` on both new sbatch = **SYNTAX_OK**. Loader routing for HateMM verified (dataset.py:499-503 →
  `load_feats_MHC` → `{path}/{dataset}/{split}_{model}.pt`), so head `--model …_HF-16f` reads exactly the
  extractor's `--out_model_tag …_HF-16f` output.

### 4.3 Collision safety (verified this prereg — ABSENT; re-check at submit)

- `data/CLIP_Embedding/HateMM/*_Qwen2.5-VL-7B-Instruct_HF-16f.pt` — do NOT exist ⇒ fresh extraction; banked 8f
  `…_HF.pt` (3 files, verified present, Jul-2) **untouched** (distinct out-tag).
- `logging/Retrieval/HateMM/RAC_video_fb16*` — do NOT exist ⇒ fresh group; `--force False` never trips
  `run_rac.py:904-908` hard-abort; the `-16f` model tag differs from CLIP/Qwen/LoRA regardless, so dirs are
  distinct.
- `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF-16f_seed*_*.trainlog` — do NOT exist ⇒ no trainlog collision.
- Smoke throwaways (`logging/_smoke_fb16/`, `_smoke` group) — deleted after smoke; must NOT persist into §4.3.

### 4.4 Smoke plan (executor runs BEFORE the real submits; leave no artifact that trips §4.3)

1. **16f extraction smoke (GPU, ~1 min):** `python src/utils/generate_VideoMLLM_embedding_HF.py --dataset HateMM
   --num_frames 16 --splits test --limit 3 --out_model_tag _smoke16f --EXP_FOLDER logging/_smoke_fb16
   --device cuda` — confirm it writes `logging/_smoke_fb16/HateMM/test_seen__smoke16f.pt`, loads with
   `img_feats`/`text_feats` shape `(3, 3584)`, labels present, **no OOM** and the masked-scatter assert (L283)
   holds at 16f; then `rm -rf logging/_smoke_fb16`. (Redirected `--EXP_FOLDER` ⇒ never writes into
   `data/CLIP_Embedding/HateMM/`.)
2. **1-seed head smoke (optional):** on the existing frozen-Qwen-**8f** cache
   (`data/CLIP_Embedding/HateMM/*_Qwen2.5-VL-7B-Instruct_HF.pt`), run ONE `run_rac.py` head with throwaway
   `--group_name _smoke` to confirm the align-fusion path loads + completes 30 epochs; delete the `_smoke` dir.
   If in doubt, skip — the same-code guarantee (§4.2) + cache dims are CPU-verified.

---

## 5. Artifacts authored this prereg + hash-freeze block

### 5.1 New artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current) |
|---|---|---|---|
| P | `refine-logs/FRAME16_PREREG.md` | **NEW** — this file | *(reviewer fills at freeze)* |
| B | `scripts/slurm/gen_embed_mllm_16f.sbatch` | **NEW** — clone of `gen_embed_mllm.sbatch`; hardcodes `--num_frames 16` + DISTINCT `--out_model_tag Qwen2.5-VL-7B-Instruct_HF-16f` (no 8f clobber); arg-driven `DATASET` (default HateMM) inherited from the fork source | `a600e74c0a6483095329f9ce15a3df19c842554362f7a3ef1f6e76e26fe3c750` |
| C | `scripts/slurm/enc3seed_fb16.sbatch` | **NEW** — clone of `enc3seed.sbatch`; `run_one` BYTE-IDENTICAL; `GROUP_NAME=RAC_video_fb16`; CONFIGS = 3 HateMM-16f seed rows | `99e7e8b10286e22d7913e85c14141c8fa02c90ae27adc0da6facaceeb703864a` |

### 5.2 Reused-unchanged machinery (verify sha at submit time; do NOT edit)

| path | role | sha256 |
|---|---|---|
| `src/utils/generate_VideoMLLM_embedding_HF.py` | frozen extractor (`--num_frames`/`--out_model_tag` pre-existing; NO edit) | `d89a912602d763aa055a54f50b0188e302e554b70ff6c0eb872f250bd454b67c` |
| `scripts/slurm/gen_embed_mllm.sbatch` | fork source for B | `9357fa1087e775d059779e6c5f86e19e71b78b2d166f904fa3c71a1a1cbb3268` |
| `scripts/slurm/enc3seed.sbatch` | same-code anchor for §4.2 (produced the frozen-Qwen-8f floor) | `dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d` |
| `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt` | banked 8f cache (paired floor; NOT clobbered) | *(present; verified untouched)* |

### 5.3 Hash-freeze (to be filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file FRAME16_PREREG.md, after review>
B a600e74c0a6483095329f9ce15a3df19c842554362f7a3ef1f6e76e26fe3c750  gen_embed_mllm_16f.sbatch
C 99e7e8b10286e22d7913e85c14141c8fa02c90ae27adc0da6facaceeb703864a  enc3seed_fb16.sbatch
```
Executor re-runs `sha256sum` on B/C (and this file) + confirms the extractor sha `d89a9126…` unchanged at
submit time; any mismatch = authorization VOID.

---

## 6. Single-submit / execution plan + resource plan

**Order (2 SLURM jobs, SEQUENTIAL, chained via `--dependency=afterok:`):**

1. `sbatch scripts/slurm/gen_embed_mllm_16f.sbatch HateMM` → produces
   `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF-16f.pt` (~0.4-0.6 GPU-h).
   Gate: 16f extraction smoke (§4.4.1) BEFORE this real submit; on COMPLETE, apply the §4.1c shape sanity.
2. `sbatch --dependency=afterok:<1> scripts/slurm/enc3seed_fb16.sbatch` → 3 HateMM-16f head runs, ~1.5 min.
   Produces `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF-16f_seed{0,1,2}_<JID>.trainlog`.

**Resource plan (STANDING INFRA RULE compliant):** each sbatch requests **`--cpus-per-task=8`, `--mem=64G`,
1×A100** (inherited from `gen_embed_mllm.sbatch` / `enc3seed.sbatch`; verified). The head has an
`afterok:<extraction>` dependency ⇒ **the two jobs never run concurrently**; peak footprint is **8 CPU / 64 G /
1 GPU** — well within the 16 CPU / 128 G / 2 GPU cap, and NEVER two 16-CPU jobs in flight. `conda activate
HateVideo`; `sbatch` with **NO `--time`**; initial `PENDING (JobHeldUser)` = **WAIT for auto-release, never
force** (CLAUDE.md). Both sbatch source `conda.sh` directly and run the ≥20 G `disk_guard.sh`.

**Cost ledger:** stage-1 total ~**0.5-0.7 GPU-h** (extraction ~0.4-0.6 h + head ~0.03 h). Conditional futures
(NOT this submit): ZH stage-1.5 ~0.6-0.9 h; LoRA-16f stage-2 ~4-6 h/dataset (≥2 changed variables). $0 CPU: all
floor re-derivation + shape sanity.

**Test-touch:** the Stage-2 head reads are the ONLY budgeted frozen-16f-encoder test evaluations; zero
test-touch before the verdict. **No job is submitted by this prereg author.** Submission happens only after the
independent 0-context review + hash-freeze (run by the orchestrator).

---

## 7. Outcome table template (filled ONLY from raw trainlogs at verdict time)

### 7.1 HateMM — frozen-Qwen-16f vs frozen-Qwen-8f floor (fill from `enc3s_HateMM_…_HF-16f_seed{0,1,2}_<JID>.trainlog`)

| seed | protocol | 16f acc/F1 | 8f floor acc/F1 (§2.1) | Δ(16f−8f) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8698/0.8606 | ___ |
| 1 | val-sel | ___ | 0.8651/0.8586 | ___ |
| 2 | val-sel | ___ | 0.8837/0.8753 | ___ |
| **mean** | **val-sel** | ___ | **0.8729/0.8648** | **___** |
| 0 | final-ep | ___ | 0.8605/0.8507 | ___ |
| 1 | final-ep | ___ | 0.8605/0.8514 | ___ |
| 2 | final-ep | ___ | 0.8837/0.8753 | ___ |
| **mean** | **final-ep** | ___ | **0.8682/0.8591** | **___** |

### 7.2 Fixed write-up format (per §3.1 rule 5 + the ladder §3.5)

`HateMM (16f vs 8f):  final-epoch: <pass/fail>; val-selected: <pass/fail>  [FORMAL bar §3.4].`
`KS-16f-dead: <KILLED (LoRA-16f auto-dead) | survives>.  CONTINUE gate (§3.3): <cleared → LoRA-16f fundable | not cleared → banked>.`
(+ MARGINAL note if a within-noise pass per B3 §2.2 precedent.)

---

## 8. What a PASS / FAIL means for the goal (D7 boundary — density is DEAD, not user-pending)

- **KS-16f-dead (recon prior — plausible given F0.5 dilution/redundancy):** denser frames carry no extra label
  signal through the frozen encoder+pool ⇒ the frame-budget cell is **CLOSED** and **LoRA-16f is auto-dead**
  (banked). The strongest, cleanest outcome: a prose-argued gap converted to a measured door-closer at ~0.6
  GPU-h. This is the honest expected result.
- **CONTINUE-gate cleared (≥ +0.010 acc, 3/3, ≥1 protocol):** the density knob moves enough to **fund** the
  contaminated LoRA-16f stage-2 (a CONDITIONAL FUTURE prereg; ≥2 changed variables, ~4-6 GPU-h/dataset). Still
  **D7-DEAD** — a spend decision, not a goal-reacher.
- **FORMAL PASS (≥ +0.030/+0.030, 3/3, both protocols):** a paper-worthy **robustness/ablation** row
  ("frame-budget: 16f > 8f on HateMM"). **NOT a novelty win** (F0.3): sampling density is engineering; it adds
  a performance data point on a dataset already converted by the frozen encoder swap, nothing toward the goal's
  "novel" clause.

**Framing sentence (verbatim):** *this measurement tests the one axis 8-frame sampling never varied — visual
sampling density (16 vs 8 frames) — through the frozen encoder+pool on HateMM; it is the cheap, single-variable,
decisive gate for the expensive LoRA-16f follow-up, and a pass is a performance/ablation row, NEVER a novelty
win — density is D7-DEAD.*

---

## 9. Provenance index

- Recon (GO-IF; stage-1 design, mechanics, cost, kill skeleton): `refine-logs/FRAME_BUDGET_FORENSIC_RECON.md` (`ec601a6`).
- Cell source: `refine-logs/REDTEAM_UNTESTED_CELLS.md` §C2 (RANK 3) + §0 stream table; F61 (`findings.jsonl`).
- Extractor / no-cutoff / args: `src/utils/generate_VideoMLLM_embedding_HF.py:84-95` (`--num_frames`/`--out_model_tag`),
  `:254-323` (`_encode`, single forward, in-place scatter assert L283), `:146-152` (`_sample_frame_indices`),
  `:437` (out-path `{outname}_{out_model_tag}.pt`).
- Loader routing: `src/data_loader/dataset.py:499-503` (HateMM → `load_feats_MHC`), `:605-608` (`{split}_{model}.pt`).
- Banked 8f floor (re-derived §2.1): `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_12850.trainlog`;
  `research-wiki/experiments/exp-encoder-3seed.md`; F53 KS-2 line (6b8f634).
- Protocol + decision rule (verbatim): `research-wiki/experiments/exp-encoder-3seed.md:73-85`.
- Ban scopes (no collision): F35/F36 (S2S causal-prefix), F37 (retrieval-object/pooling-lossless; 16f arm
  cancelled), F39 (CTF supervised temporal-pool) — `findings.jsonl` + recon §5.
- Timing anchor: `refine-logs/S2S_EXTRACTION_RECORD.md` (HateMM 949.1 s, 8f); head ~29 s from trainlog tqdm.
- Same-code anchor: `scripts/slurm/enc3seed.sbatch` (sha `dbe3fb81…`, also pinned by LoRA-HateMM / vision preregs).

**Required statements:** ZERO GPU/SLURM/Modal spent by this prereg author (only pure-CPU login-node floor
re-parsing + collision/syntax/same-code verification, seconds; no held-out test metric produced). All floor
numbers re-parsed from banked completed-run trainlogs (numeric-provenance discipline). No `state/` mutated. No
`research-wiki/` mutated. NO job submitted. Not pushed.

---

## 10. (reserved)

---

## 11. DEVIATIONS FROM THE RECON — flagged loudly

1. **DEV-1 (KILL bar uses SIGN, not "bootstrap CI straddles 0"). MATERIAL — pinned to the task + house
   discipline, over the recon's wording.** Recon §5 phrases KS-16f-dead as "mean paired Δacc ≤ 0 OR bootstrap
   CI straddles 0 on both protocols." A bootstrap-CI test **conflicts with the house n=3 no-bootstrap
   discipline** (`exp-encoder-3seed.md:78-79` rule 3: "n=3 too small for a bootstrap — report paired-t as an
   effect-size descriptor only, no significance claim"). The task's binding kill-bar is **sign-based** ("mean
   paired Δacc ≤ 0 OR sign-inconsistent, both protocols"), which I pin verbatim (§3.2). Only the significance
   *formalism* changes; the qualitative bar (tie/regress on both protocols ⇒ dead) is identical.

2. **DEV-2 (dedicated 16f extraction sbatch instead of reusing `gen_embed_mllm.sbatch` with `NUM_FRAMES=16`).
   MATERIAL / safety-favorable.** The existing `gen_embed_mllm.sbatch` plumbs `--num_frames` (via `NUM_FRAMES`
   env) but **NOT** `--out_model_tag` — so `NUM_FRAMES=16 sbatch gen_embed_mllm.sbatch HateMM` would write the
   16f features to the **default** tag `…_HF.pt` and **CLOBBER the banked 8f floor**. To make the DISTINCT
   out-tag a hash-frozen invariant (no submit-time typo can clobber), I authored `gen_embed_mllm_16f.sbatch`
   with both `--num_frames 16` and `--out_model_tag Qwen2.5-VL-7B-Instruct_HF-16f` hardcoded. The extractor
   Python is UNCHANGED (no code edit — both args pre-exist); the only new artifact is the sbatch wrapper.

3. **DEV-3 (no code edit needed for stage-1 — confirmed, recon §2 Finding B holds).** Neutral. The recon's core
   mechanical claim — `--num_frames 16` needs no code edit, no `cutoff_len` raise, no VRAM concern (the
   extractor has no truncation; 16f ≈ 1.5k visual tokens fits trivially) — is verified: `--num_frames` (L90-95)
   and `--out_model_tag` (L84-89) are both pre-existing argparse args; `_sample_frame_indices` = `np.linspace`
   works at any count; the masked-scatter assert holds at even frame counts. Stage-1 is genuinely
   single-variable.

4. **DEV-4 (group tag `RAC_video_fb16`, job tags `enc3seed_fb16`/`mllm_embed_16f`). Neutral.** Follows the
   recon §6 naming (`fb16`, `RAC_video_fb16`); collision-checked ABSENT (§4.3).

5. **DEV-5 (ZH stage-1.5 + LoRA-16f stage-2 + 32f are declared CONDITIONAL FUTURE / OUT, per the task).
   Documented.** The recon frames stage-2 as "NO-GO unless stage-1 moves" and ZH as "stage-1.5 if HateMM moves";
   the task restricts THIS submit to HateMM frozen-16f stage-1 and makes ZH/LoRA-16f conditional future preregs
   gated on §3.3, and 32f out-of-scope on multiplicity. No ZH/LoRA-16f/32f artifact is authored or submitted here.
