# READOUT-AXIS FORENSIC RECON — F68 ledger P1 (which layer / which token / which prompt)

**Executor:** readout-forensic-recon subagent. **Date:** 2026-07-25 NZST.
**Mission:** zero-GPU forensic recon for the READOUT-AXIS cell (F68 ledger candidate **P1**). Our extraction
pipeline has NEVER varied which transformer **layer**, which **token/pooling span**, or which **prompt** the
video-level embedding is read from. VidVec / PromptEOL / E5-V evidence says intermediate-layer + one-word-prompt
last-token readout beats naive pooling for classification embeddings. Produce GO/NO-GO + full execution skeleton.
**Source read first:** `refine-logs/LITSURVEY_MLLM_EMBEDDING.md` §0 (four-lever map) + §2.A (readout axis, A1–A4)
+ §2.A3 (Echo). This record honors that survey's isomorphism verdict (readout axis = SURVIVES, un-varied).

**Discipline honored.** CPU-only forensic recon. ZERO GPU / SLURM / Modal / downloads / training / job
submission. NO prereg authored (this is recon, not authorization). `autoresearch/goal_mllm_plus3/state/`
UNMODIFIED. Deliverable committed on `main`, **not pushed**. Every code fact is cited to a line number and was
read from the working tree; every banked number is cited to its source file.

---

## 0. VERDICT UP FRONT

**GO-IF.** The readout axis genuinely SURVIVES isomorphism (F68-P1; litsurvey §2.A) — layer/token/prompt were
never varied and this is not any temporal/pooling/encoder-class kill. The design is clean, small (≤4 cells),
multiplicity-safe, and ZH-goal-relevant (harden the marginal val-selected leg, F45). **BUT this is NOT a $0 gate.**
Unlike ISR (F66) and LP, which screened on *already-banked* caches, the readout axis **cannot be screened on any
banked artifact** — the current extractor saves only the pooled **final-layer** vector (`_encode` discards the
intermediate hidden states and per-token positions after pooling). So the axis costs **~2 GPU-h of mandatory
local re-extraction to even reach the $0 CPU screen**, plus a small **codex-review-gated** extractor code change
(new read of `out.hidden_states[L]` + a last-token span + one-word prompts).

- **GO** if the orchestrator accepts ~2 GPU-h of local re-extraction (videos never leave the node — extraction is
  local-SLURM-only by policy) as the entry price, then a $0 CPU screen decides whether any verdict GPU is spent.
- **NO-GO reduction** if that ~2 GPU-h is judged not worth a LOW–MODEST (~15–20% ZH; EN capped) prior on a
  D7-thin readout choice — in which case the axis stays a paper-only "readout was fixed at final-layer/mean-pool"
  limitations note, no code, no GPU.

**Recommended:** GO. Prior is non-zero and ZH-relevant, cost is modest and bounded, the $0 screen kills cheaply if
flat (KS at ~2 GPU-h), and the axis is the single un-enumerated one inside our paradigm (litsurvey verdict).

**Headline facts for the reply:** grid = baseline `R0` + **3** new cells (`R1`=intermediate-layer L24,
`R2`=one-word+last-token@L28, `R3`=one-word+last-token@L24); **2 prompt-passes per dataset** (the layer and
token axes are harvested GPU-FREE inside each pass via `output_hidden_states=True`, which is already set — **only
the one-word PROMPT change forces the second forward**); **~2 GPU-h total** for ZH+HateMM re-extraction; then a
$0 CPU screen. Current readout = **final layer (`hidden_states[-1]`), mean-pool** (img over vision+instruction
prefix, text over the ~3-token assistant-header tail), two **fixed non-one-word prompts**.

---

## 1. THE CURRENT READOUT, EXACTLY (verified in code, cited)

Both extractors share a byte-identical `_encode`. The LoRA variant is a strict superset (merges the adapter, then
runs the *identical* frozen forward + pooling); when `--lora_dir` is empty it equals the frozen extractor.

- Frozen: `src/utils/generate_VideoMLLM_embedding_HF.py`
- LoRA:   `src/utils/generate_VideoMLLM_embedding_lora_HF.py`  (adapter merged at `main`, lines 429-444)

**(a) Which LAYER.** `generate_VideoMLLM_embedding_HF.py:278-279`:
```
out = model(**inputs, output_hidden_states=True, use_cache=False)
last_hidden = out.hidden_states[-1][0]      # [seq_len, D]   <-- FINAL layer only
```
`output_hidden_states=True` is **already on**, so the forward already materialises **all** layer outputs — but
the code reads only `[-1]`. **Qwen2.5-VL-7B LLM = 28 decoder layers, hidden_size 3584** (verified from
`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/.../config.json`: `num_hidden_layers=28`,
`hidden_size=3584`). Therefore `out.hidden_states` is a **tuple of length 29** (index 0 = embedding output;
indices 1..28 = layer outputs; `[-1]` = index 28 = final layer). The intermediate layers are computed and then
thrown away.

**(b) Which TOKEN / pooling span.** Two streams, both **mean-pool** (never last-token, never a learned pool):
- `img_feats` — `span="prefix"` (`:290-303`): `pooled = last_hidden[:end].mean(dim=0)` where `end` = position of
  the **last** `<|im_start|>` (start of the assistant header). = **mean over the whole vision + instruction
  prefix**, everything up to (not including) the assistant generation-prompt tail.
- `text_feats` — `span="response"` (`:304-318`): `pooled = last_hidden[start:].mean(dim=0)` where `start` = the
  **last** `<|im_start|>` position. = **mean over the trailing assistant-header tail** (`<|im_start|>assistant\n`,
  ~3 tokens). NOTE: the docstring calls this "last-token decoder embedding" but the code takes a **mean over ~3
  header tokens**, NOT a single last token, and there is **no one-word constraint** on the prompt. So even the
  text stream is *not* a PromptEOL/E5-V last-token-after-"in one word:" readout.
- Both streams L2-normalised (`:321-322`): `pooled = F.normalize(pooled.float(), p=2, dim=0)`.

**(c) Which PROMPT.** Two **fixed, non-one-word** instructions (`:44-52`, identical in the LoRA variant `:58-66`):
- `IMG_INSTRUCTION` = "Describe the people, symbols, gestures, and on-screen text in this video."
- `TEXT_INSTRUCTION` = "You are analysing a short video for potentially hateful or offensive content. Considering
  the frames together with the provided title and transcript, summarise the targets, symbols, tone, and any
  harmful intent conveyed." (+ `"\nTitle: … \nTranscript: …"` appended at `:349-355`).
- Neither carries the one-word summarisation constraint ("…in one word:" / a dedicated `<emb>` token) that
  PromptEOL/E5-V/VidVec use to force the final position to compress the input.

**Baseline `R0` (the deployed readout) is therefore: {layer 28 (final), mean-pool, fixed analytic/descriptive
prompts}.** All three readout knobs — layer, token, prompt — are at a single un-swept setting.

---

## 2. AMORTIZATION — how many forward passes the grid actually costs (the critical question)

**Answer: the LAYER axis and the TOKEN/pooling axis are GPU-FREE within a forward; only a PROMPT change forces a
new forward.** Because `output_hidden_states=True` is already set, ONE forward pass yields every layer's hidden
state (`out.hidden_states[0..28]`) and the full token sequence in memory. Reading layer 24 instead of / in
addition to layer 28 is a free tensor index; reading the last token instead of the mean is a free re-pool of the
same tensors. What is NOT free is a **different input prompt** (a one-word readout prompt changes the token
sequence → a genuinely new forward), and **echo** (doubled context → new forward, longer sequence).

Honest per-item forward count (the deployed extractor runs **2 forwards/item** = img-prefix + text-response):

| pass | prompt | forwards/item | cells harvested (FREE across layers/spans inside the pass) |
|---|---|---|---|
| **Pass A** | current IMG/TEXT instructions | 2 (img, text) | `R0` = L28 + current-span (**re-computes the banked baseline** → bit-exact clobber-guard); `R1` = L24 + current-span |
| **Pass B** | one-word variants of both instructions | 2 (img, text) | `R2` = L28 + last-token; `R3` = L24 + last-token |

**= 4 forwards/item total (2 prompt-passes), i.e. 2× the deployed per-item cost.** The grid can run as ONE SLURM
job/dataset doing both prompt variants internally and writing all 4 cell-caches, or two jobs; either way the GPU
is the same **2 prompt-passes**. The layer sweep {L24,L28} and span choice {mean, last-token} add **zero**
forwards. (A full 28-layer sweep is *also* GPU-free but is a multiplicity trap — see §5; we pin one literature
layer a priori and do NOT sweep.)

---

## 3. THE ARM GRID (pre-declarable, small, multiplicity-safe)

Grid = **baseline `R0` + 3 new cells**, per dataset (≤4 new-cell cap honored: 3 new cells). Two literature
readout knobs, one literature-pinned intermediate layer:

| cell | layer | token/span | prompt | recipe / cite | forward source |
|---|---|---|---|---|---|
| **R0** (baseline) | 28 (final) | mean (current spans) | current descriptive/analytic | **deployed** (`_encode` as-is) | banked cache = deployed; re-computed in Pass A as clobber-guard |
| **R1** | **24** | mean (current spans) | current | intermediate-layer (**VidVec**, 2602.08099) | Pass A, free |
| **R2** | 28 (final) | **last token** @ gen-position | **one-word** ("…in one word:" / `<emb>`) | PromptEOL/E5-V (2307.16645 / 2407.12580) | Pass B, free |
| **R3** | **24** | **last token** @ gen-position | **one-word** | **VidVec full recipe** (intermediate + one-word) | Pass B, free |

**Layer L\*=24, justified.** Qwen2.5-VL-7B LLM has **28** decoder layers. VidVec (Tzachor et al., Feb 2026,
2602.08099) reads the embedding from **intermediate layer 24** of VideoLLaMA3-7B — a same-class ~28-layer 7B
decoder — and reports intermediate > final for video-text embeddings. 24/28 ≈ **0.857 relative depth**. So
`L*=24` = `out.hidden_states[24]` is the single **literature-pinned** intermediate; we do **not** sweep layers
(GPU-free availability of all 29 layers is a forking-path trap, resisted by pinning L24 a priori).

**One-word readout on both streams.** For R2/R3 both streams switch to a one-word summarisation prompt and read
the **last token at the generation position** (where the model would emit the one word) — img: "Describe this
video in one word:"; text: analytic instruction + "…summarise … in one word:". This unifies both streams onto the
PromptEOL/E5-V last-token-after-constraint readout (vs R0's mean-pool over descriptive prompts).

**Optional 5th cell — Echo (deferred, NOT in the core grid).** `R4` = echo/repetition readout (litsurvey A3,
2402.15449 / training-free 2502.20726): feed the input twice, read the second occurrence so early tokens see full
context. **Ban status: SURVIVES** (readout-time, training-free, a *single* read of the 2nd copy — not an average).
**Cost:** transcript-only echo is cheap (+~0.3 GPU-h/dataset, +1 prompt-pass); full-multimodal echo doubles the
visual token count → near-16f expensive. **Prior:** weakest of the family (~10%, Law-I discounted — our pooled
vector already integrates the causal prefixes, so per-token echo gains largely wash out at the pooled level).
**Recommendation: hold R4 out of the core grid** (keeps the grid at 3 new cells and its prior is the thinnest);
add it only as a transcript-only companion if R1–R3 show any life. Excluded from the pre-declared family for now.

---

## 4. ENCODERS / DATASETS / GPU-h

| dataset | role | encoder (adapter, deployed) | cache tag (banked R0) | dev n | train n |
|---|---|---|---|---|---|
| **MHC_zh** (ZH) | **PRIMARY** — harden marginal val-sel leg (F45) | `logging/lora/MHC_zh` (LoRA_HF) | `Qwen2.5-VL-7B-Instruct-LoRA_HF` | 78 (28 pos) | 579 |
| **HateMM** | SECONDARY — hold check | `logging/lora/HateMM_curric` (curric) | `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` | 107 (43 pos) | 744 |
| **MHC (EN)** | **SKIP** for the grid | — | — | 80 | 549 |

Adapter + banked-cache pairing verified present: `logging/lora/MHC_zh/adapter_model.safetensors` +
`logging/lora/HateMM_curric/adapter_model.safetensors` both exist; banked LoRA_HF (ZH) and LoRA-curric_HF
(HateMM) caches present for all three splits. **The encoder is byte-identical to the deployed one** — the SAME
frozen base + SAME already-trained adapter, merged and run frozen; ONLY the read changes.

**EN skipped, honestly:** MHC-EN is proven **label-limited at five levels** (F44 frozen, F50 collapsed-adapted,
F55 healthy-image, F58, F65 vision-unfreeze). A readout change is a *representation-read* change, and the F44/F55
arithmetic caps representation-side gains on EN independent of how the vector is read. Prior ≈ near-0; a frozen-EN
hold-check could be added at ~1 GPU-h if a reviewer insists, but it is not recommended.

**GPU-h (honest).** Deployed 8f full-video extraction ≈ **25–35 min/dataset** for the 2-stream pass (consistent
with FRAME16_SUBMIT_RECORD: the 16f J1 extract "proper" was ~30 min after subtracting the ~30-min disk_guard B2
prune at sbatch start; 8f is lighter). Two prompt-passes/dataset ≈ **~1 GPU-h/dataset**. ZH + HateMM =
**~2 GPU-h** to reach the $0 screen. Verdict-stage 3-seed head runs are ~1.5–4 min each (FRAME16 J2 = 04:13 for
3 seeds), so the *dominant* cost is the ~2 GPU-h extraction; the verdict adds well under an hour of GPU only if a
cell clears the screen.

---

## 5. SELECTION PROTOCOL ($0 screen, forking-path discipline)

**The grid is screened DEV-SIDE ONLY, as ONE pre-registered family, ONE bite.** Mirrors ISR (F66) and LP exactly.

**Cheapest honest screen = $0 CPU, head-FREE, on the raw fused key** (the only object uniform across all four
readout key-spaces — there are no trained head ckpts for the readout spaces, same situation LP §1 documents for
the LoRA spaces). Machinery reused **verbatim**, no vote reimplemented:
- **Fused key** (identical construction for every cell): per video take `img_feats` (3584-d) + `text_feats`
  (3584-d) from the cell's cache, **L2-norm each stream, concat → 7168-d, L2-renorm** (LP_GATE_RECORD §1).
- **Vote:** deployed **rank-weighted signed-cosine top-20** `_weighted_signed_vote`
  (`scripts/analysis/cross_channel_router_gate.py:73-79`), two arms both reading only train + dev_seen:
  (i) **video-level LOO** over train∪dev (diagonal self-exclusion) and (ii) **strict dev-query → train-memory**
  (ISR_PREGATE_RECORD §0.2 verbatim). Test split is **never** loaded (hard-assert `split ∈ {train, dev_seen}`).
- **Screen statistic:** per cell, dev Δacc = (cell fused-key vote acc) − (R0 fused-key vote acc), on the **same**
  videos, both arms, per dataset. Report exact **items fixed / broken / net** (dev n = 78 / 107 → count-level).
- **Machinery-validity guards (report, mirror ISR/LP):** (a) permutation-null ≥200 perms — real Δ must exceed the
  null 95th pct to count as signal; (b) bootstrap-1000 5th-pct of Δ; (c) a degenerate-recovery assert (R0-cache
  re-fed through the screen reproduces the banked R0 vote bit-exact). These are validity guards, not extra bars.

**Why head-free is the right primary, and its honest limit.** The raw-key kNN vote is the ISR/LP-consistent $0
object and needs no head training or head-seed noise. Its **limit**: a readout change alters the embedding
*geometry* (different layer/token/prompt = different manifold), and a *trained* enc3s head might exploit geometry
the raw-kNN vote cannot — so the $0 screen is a **proxy** for, not a substitute for, the deployed trained head.
**Mitigation (pre-declared tie-break):** the head-retrain is only **~25 s GPU/cell**; if the $0 screen is
**ambiguous** (winner within ±1 dev item of R0, OR the LOO and dev-query arms disagree on the winner), run a
**~25 s/cell head-retrain screen** (enc3s head on the cell's train cache, eval dev) to break the tie before
advancing. Cheapest-first: $0 CPU decides unless it can't.

**One dev-winner advances.** The screen selects **at most one** cell (highest dev Δacc clearing the advance bar,
on ≥1 dataset). Only that one cell proceeds to a **single** test-touch at the verdict stage. Multiplicity is
controlled: the whole {R1,R2,R3} family is one pre-registered bite, one winner, one test measurement. **No layer
sweep, no prompt sweep beyond the two pre-declared, no per-cell test peeking.** Pre-registration (if the screen
promotes) declares the full grid + the single-winner-advances rule BEFORE any dev number is computed
(ISR/LP §0 forking-path discipline).

---

## 6. BAN / ISOMORPHISM CHECK (quote why each near-neighbor does NOT apply)

- **NOT P4 (schema-distill).** P4 changed the **SFT training data** (distilled a schema/rationale into the
  generative SFT *target*). The readout grid changes **nothing about training** — no SFT, no new training data,
  no adapter re-fit. It is a pure **extraction-time read** of the SAME frozen (or SAME-adapter-merged) forward
  pass. Orthogonal object.
- **NOT multi-prompt ENSEMBLING.** Every cell is a **single** readout (one layer, one span, one prompt). The grid
  **compares** single readouts and picks ONE — it never **averages** across prompts/layers. The MetaEOL
  multi-prompt *ensemble* (2402.18458), flagged-not-banned by the `cross-seed ensembles` rule (litsurvey §3, F68
  micro-ruling), is **explicitly excluded** here; the core grid needs **no** user micro-ruling because it does no
  ensembling. (The one-line micro-ruling is only needed IF an ensemble readout is ever proposed — it is not.)
- **NOT F24 / encoder-class.** No new encoder, no retraining, no LoRA re-fit, no mask surgery. The base + adapter
  are byte-identical to the deployed encoder; the merged frozen forward is identical. Only which layer/token/prompt
  the frozen output is read from changes. This **sidesteps the F24 encoder-class veto** (litsurvey §2.A note:
  "nothing about the encoder changes"). (Contrast the litsurvey's B-axis candidates — SupCon/hybrid/bidirectional
  — which ARE encoder-class and need the D7 sub-ruling; this cell does not.)
- **NOT F35–F39 / F67 (temporal / frame axis).** No temporal frame-group object, no per-segment re-encode
  (that's ISR/F66), no frame set/pool/order (S2S/CTF/ISR, F37/F39/F66), no denser sampling (F67 killed 8→16;
  frames are held at **8** here). Same 8-frame jointly-processed pooled object; the layer/token/prompt axis is
  orthogonal to frame handling. (Litsurvey §0 cross-check: SOTA video-MLLM-embedding works use NO temporal
  operator — the borrowable video insight is the *readout*, corroborating F35/F37/F67, not a new frame lever.)
- **Echo (R4) ban status.** Readout-time, training-free, a **single** read of the 2nd occurrence (not an average
  → not ensembling). SURVIVES (litsurvey A3); deferred on cost + weakest prior (see §3).

---

## 7. KILL-BAR SKELETON

- **House dev-screen advance bar (permissive, to catch signal):** the winner cell's dev **Δacc ≥ +0.02** over R0
  on **≥1 dataset** (ZH or HateMM), on the $0 raw-key vote, in **either** arm, with the machinery-validity guards
  sane (real Δ > perm-null p95 as a signal check; degenerate-recovery bit-exact). The +0.02 screen bar is
  deliberately **below** the verdict bar — a screen should be permissive (don't miss a candidate), the verdict is
  the strict judgment. If the screen is ambiguous, the ~25 s/cell head-retrain tie-break (§5) decides.
- **Verdict bar (strict, house standard — B3 / frame16 / encoder-3seed):** the single advanced cell, **3-seed
  paired** vs the banked deployed **R0** arm, **house +0.030 acc / +0.030 mF1, 3/3 seeds, DUAL protocol**
  (final-epoch AND val-selected). **Single test-touch, verdict stage only.** ZH val-selected leg is the target
  (F45 78-dev selection tax); HateMM is the hold check (already passes, F53 — must not regress).
- **Kill-switch (KS-readout-dead):** if **ALL** grid cells are **≤ R0** on the $0 dev screen (no cell clears
  +0.02 on either dataset in either arm) → **cell DEAD at ~2 GPU-h total** (extraction only; zero verdict GPU).
  Mirrors ISR NO-GO / frame16 KS-16f-dead: a flat $0 screen closes the axis without any test-touch.

---

## 8. COLLISION-SAFE NAMING · COST LEDGER · PRIORS · D7

**Collision-safe cache tags** (verified: no `-ro_*` / `readout` cache exists today). Append a readout-variant
suffix to the deployed tag so the banked R0 cache is NEVER clobbered (frame16 `-16f` precedent):

| cell | ZH tag | HateMM tag |
|---|---|---|
| R0 re-compute (clobber-guard) | `…-LoRA_HF-ro_L28` | `…-LoRA-curric_HF-ro_L28` |
| R1 (L24, current prompt) | `…-LoRA_HF-ro_L24` | `…-LoRA-curric_HF-ro_L24` |
| R2 (one-word, last-tok, L28) | `…-LoRA_HF-ro_ow_L28` | `…-LoRA-curric_HF-ro_ow_L28` |
| R3 (one-word, last-tok, L24) | `…-LoRA_HF-ro_ow_L24` | `…-LoRA-curric_HF-ro_ow_L24` |

(`…` = `Qwen2.5-VL-7B-Instruct`.) The R0-recompute (`-ro_L28`, current span) must reproduce the banked R0 cache
**bit-exact** — a clobber-guard sanity like frame16's mtime/sha check (deterministic frozen forward).

**Cost ledger.**
| stage | GPU-h | notes |
|---|---|---|
| ZH re-extract (2 prompt-passes) | ~1.0 | one job, 4 forwards/item, writes R0/R1/R2/R3 caches |
| HateMM re-extract (2 prompt-passes) | ~1.0 | one job, same |
| **subtotal to reach $0 screen** | **~2.0** | mandatory; the axis has no banked screen |
| $0 CPU dev screen | 0 | ISR/LP machinery, CPU, <1 min |
| (tie-break head-retrain, if ambiguous) | ~0.01 | ~25 s/cell × ≤4 cells × 2 ds |
| verdict 3-seed head (only if screen promotes) | <0.1 | ~1.5–4 min/dataset, 3 seeds |
| **total if KILLED at screen** | **~2.0** | **total if promotes to verdict** ≈ ~2.1 |

**Honest priors.** LOW–MODEST, **~15–20% ZH** (litsurvey §2.A1–A2), thinner on HateMM (already passes; target =
no-regression robustness), ~near-0 EN (label-capped, 5 levels). Realistic best case = the intermediate-layer
and/or one-word readout sharpens the ZH representation enough to survive the F45 val-selected 78-dev selection tax
— a **performance/robustness** result, not a novelty claim. Most-likely case (consistent with terminus) = flat $0
screen, KS fires, axis closed at ~2 GPU-h.

**D7 status.** Readout = **extraction engineering**; **novelty-thin alone** (a layer/token/prompt choice is not a
method contribution). Its value is (a) hardening the marginal ZH leg (perf/robustness) and (b) as a **component of
the MLLM-embedding-paradigm story** ("principled readout for hate-video MLLM embeddings", F68). Even a positive
result needs the paradigm framing to carry D7 weight; the D7 call is the user's, same as every encoder-adjacent
cell. (This cell does NOT need the encoder-class D7 sub-ruling that the litsurvey B-axis candidates do — see §6.)

---

## 9. EXECUTION SKELETON (what actually has to be built — no code written here)

**Sequencing:** code change (codex-gated) → smoke → 2 extraction jobs (ZH, HateMM) → $0 CPU screen → [if promote]
prereg → review → freeze → 3-seed verdict. Extraction and $0 screen are the only steps before a kill decision.

1. **Extractor code change (codex-review-gated — touches hidden-state indexing / model-internals read).** Add to
   `generate_VideoMLLM_embedding_lora_HF.py` (the superset; frozen path falls out with empty `--lora_dir`):
   - `--readout_layer` (int, default 28 = final; the grid uses {28, 24}) → in `_encode`, read
     `out.hidden_states[args.readout_layer][0]` instead of the hardcoded `[-1]`. Guard `0 ≤ L ≤ 28`.
   - `--readout_span` ∈ {`current` (existing prefix/response mean, default), `last_token` (single last token at
     the generation position)} → add a `last_token` branch returning `last_hidden[-1]`.
   - `--readout_prompt` ∈ {`baseline` (current IMG/TEXT instructions, default), `one_word` (append the one-word
     summarisation constraint to both)} → two new module-level prompt constants; select by flag.
   - `--out_model_tag` already exists → supply the `-ro_*` suffix per §8 (no clobber).
   - **Amortization hook (recommended):** let one invocation loop over the {baseline, one_word} prompts and dump
     both L28 and L24 caches, so 4 caches come out of 4 forwards/item in a single job (§2). Keep it single-variable
     and deterministic; NO change to the frame sampler, the fusion, or the loader contract.
   - **Codex gate rationale (CLAUDE.md / precedent):** any change reading model internals (`hidden_states` layer
     index, generation-position token) is codex-review-gated before GPU submission.
2. **Smoke (throwaway, `--limit 3`, scratchpad sbatch, mirror FRAME16 §3):** confirm shapes (N,3584) for img+text,
   0 NaN, R0-recompute (`-ro_L28`, current span) reproduces a 3-row slice of the banked R0 cache bit-exact
   (clobber-guard), then discard the smoke caches.
3. **Extraction jobs (local SLURM only — videos never leave the node):** one job/dataset, template = a `--readout_*`
   generalization of `scripts/slurm/gen_embed_lora.sbatch` (8 CPU / 64 G / 1×A100, no `--time`, `afterok` nothing
   — but **never two 16-CPU jobs concurrent** per the infra rule; these are 8-CPU so fine). ZH: `--dataset MHC_zh
   --lora_dir logging/lora/MHC_zh`; HateMM: `--dataset HateMM --lora_dir logging/lora/HateMM_curric`. Each dumps
   R0/R1/R2/R3 caches with the §8 tags. ~1 GPU-h each.
4. **$0 CPU screen:** a new `scripts/analysis/readout_screen.py` that reuses the fused-key + `_weighted_signed_vote`
   machinery (import/lift from `cross_channel_router_gate.py:73-79`, LOO + dev-query arms), computes per-cell dev
   Δacc + fixed/broken/net + perm-null + bootstrap + degenerate-recovery assert, writes
   `refine-logs/READOUT_SCREEN_OUT.json`. CPU-only, `CUDA_VISIBLE_DEVICES=""`, test split never opened. This is the
   KILL-or-promote decision.
5. **If promote (winner clears +0.02 on ≥1 dataset):** escalate to the normal ceremony — prereg (declares the
   frozen grid + single-winner + verdict bar) → 0-context review → freeze (sha) → 3-seed test verdict via
   `enc3seed*.sbatch` on the winner's cache, paired vs banked R0. Else KS fires → bank the negative (candidate for
   a new F-finding, orchestrator's call — this recon modifies no `state/`).

---

## 10. PROVENANCE

- Code (working tree, read this session): `src/utils/generate_VideoMLLM_embedding_HF.py` (`_encode` :254-323,
  layer read :278-279, spans :290-318, prompts :44-52); `_lora_HF` variant (identical `_encode` :277-346, adapter
  merge :429-444, prompts :58-66). Qwen config: `num_hidden_layers=28`, `hidden_size=3584`.
- $0-screen machinery: `scripts/analysis/cross_channel_router_gate.py:73-79` (`_weighted_signed_vote`),
  fused-key construction `refine-logs/LP_GATE_RECORD.md §1`, two-arm protocol `refine-logs/ISR_PREGATE_RECORD.md
  §0.2`. Verdict/kill-bar template: `refine-logs/FRAME16_PREREG.md` + `FRAME16_VERDICT_REVIEW.md`.
- Extraction cost anchor: `refine-logs/FRAME16_SUBMIT_RECORD.md` (16f J1 ~30 min extraction-proper; J2 3-seed head
  04:13). Adapters/caches/dev-sizes verified present this session (§4).
- Literature: `refine-logs/LITSURVEY_MLLM_EMBEDDING.md` §0/§1/§2.A (readout axis A1–A4, Echo A3), F68 ledger
  (`state/findings.jsonl` F68, candidate P1). External anchors: VidVec 2602.08099 (intermediate layer 24 on
  VideoLLaMA3-7B), PromptEOL 2307.16645, E5-V 2407.12580, Echo 2402.15449 / 2502.20726, MetaEOL 2402.18458.
- Discipline: CPU-only recon, ZERO GPU/SLURM/Modal/downloads/training/submission; no prereg authored; `state/`
  unmodified; cloud/external numbers are triage context only, never mixed with local G-repro numbers. Nothing here
  is a prereg or a GPU authorization — it is a GO-IF recommendation with a fully-specified execution skeleton.
