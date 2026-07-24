# READOUT-GRID (R0–R3) — INDEPENDENT 0-CONTEXT PRE-REGISTRATION REVIEW

**Reviewer:** independent 0-context pre-registration reviewer (no prior project context; adversarial
mandate; zero user interaction; no job submitted; prereg NOT modified).
**Date:** 2026-07-25 NZST. **CPU-only** (no GPU/SLURM/Modal spent; `state/` not touched;
`autoresearch/goal_mllm_plus3/state/` unmodified).
**Target:** `refine-logs/READOUT_PREREG.md` (commit `1b3e0c6`; on-disk sha256
`f6d43096332acb2e9f9743f5e51967f6a93b3006770be3520effc13929aac543`).
**Recon:** `refine-logs/READOUT_FORENSIC_RECON.md` (`61a9f4a`).
**Method:** every load-bearing fact re-derived from primary artifacts on disk — the readout extractor,
its fork source, the screen, the sbatch, the vote source and the head clone source read directly; the
banked 13150/13241 and 13115/12850 trainlogs re-parsed with an **independently written** parser (not the
prereg's embedded one); the Qwen config `num_hidden_layers` read from the HF cache; every freeze-block
hash recomputed; all collision paths `ls`-checked on disk; both python artifacts `py_compile`d and the
sbatch `bash -n`'d. The prereg's and recon's numbers were treated as untrusted until independently
reproduced.

## VERDICT: **APPROVED-WITH-NOTES** (all three notes non-blocking)

The prereg is hash-integral, floor-faithful to 4dp (and line-accurate), extractor-mechanics-correct,
same-code-paired (the pooling math is byte-identical to the deployed `_encode`, and the promote-time head
`run_one` is pinned byte-identical to the banked 13150/13241 runner), layer-indexing-correct,
clobber-impossible (every out-path carries a distinct `-ro_*` suffix so the banked deployed cache is never
written), leakage-clean (label-free prompts, no OCR, own-train-split only), veto-compliant, collision-free
on disk, and its kill-ladder is fully decidable from raw logs/JSON by a 0-context verdict reviewer with no
interpretive freedom. The only manipulated axis is genuinely the READOUT: the fork source
(`generate_VideoMLLM_embedding_lora_HF.py`) is git-clean and byte-untouched, and the variant's changes are
exactly {multi-layer harvest from `output_hidden_states`, a `last_token` span, two one-word prompt
constants, `-ro_*` out-tag plumbing + the frozen `CELLS` table + a `num_hidden_layers==28` guard}. The
$0 screen forks the deployed vote **byte-verbatim**, hard-blocks the test split two ways
(`assert split ∈ {train, dev_seen}` + a `CUDA_VISIBLE_DEVICES==""` assert), and advances a **single**
winner on a pre-declared `Δacc ≥ +0.020` bar with `R0-bit-exact PASS` as a hard conjunct. The verdict bar
is quoted verbatim from `exp-encoder-3seed.md:73-85` (`+0.030/+0.030`, 3/3, dual protocol, paired vs the
banked deployed R0). The one-word readout is repeatedly and correctly framed as a **single** readout, not
multi-prompt ensembling, and the ensembling micro-ruling is left explicitly unconsumed. The three notes
below are (1) an honest FP-reproducibility risk on the R0 bit-exact gate that can only ever **block**, never
manufacture a pass, (2) a torch-vs-numpy re-implementation of the (non-load-bearing) fused-key recipe while
the load-bearing vote is byte-verbatim, and (3) an acc/F1 column-order slip on the context floors whose
values are all correct. None affects decidability, leakage, clobber-safety, or the honesty of any bar.
**Cleared to freeze + single-submit** (extractor codex-gate first, per the prereg's own step 0).

---

## Rationale (one paragraph)

The grid measures the one axis extraction never varied — which transformer LAYER, TOKEN/pooling span, and
PROMPT the video-level embedding is read from — off the **deployed merged encoder** (frozen base + the
already-trained LoRA adapter), on ZH (primary; harden the marginal val-selected leg) and HateMM (hold).
Its validity hinges on one property: the merged frozen forward must be byte-identical to the deployed one,
so a within-cell change is a pure readout change. That property holds under audit. `_encode` is byte-
identical between the frozen and LoRA extractors (`diff` empty), and the readout variant's `_pool_span`
reproduces the deployed prefix/response pooling math character-for-character (same `im_start` boundary, same
`mean(dim=0)`, same `float()`+`F.normalize(p=2,dim=0)`), with `last_token` = `last_hidden[-1]` added only
for R2/R3. The layer read is `out.hidden_states[L]` for `L ∈ {28, 24}`: with `num_hidden_layers=28`
(verified from the HF config) the tuple has 29 entries (0 = embeddings, 1..28 = layer outputs), so
`hidden_states[28]` **is** the deployed `hidden_states[-1]` (R0) and `hidden_states[24]` is the intended
~0.857-depth intermediate; a runtime `assert num_hidden_layers == 28` makes the pinned indices safe. R0
(`-ro_L28`) is therefore expected to reproduce the banked deployed cache bit-exact, and the screen enforces
this as a hard `max|Δ| == 0.0` gate over train+dev on both streams — the G-repro anchor that certifies the
extraction run reproduces the deployed forward and thereby justifies pairing the winner against the banked
13150/13241 heads instead of a redundant fresh R0 head. The $0 screen reuses `_weighted_signed_vote`
byte-verbatim from `cross_channel_router_gate.py:73-79`, the LP §1 fused key, and the ISR §0.2 two arms
(LOO + dev→train), never opens test, and promotes at most one cell on `Δacc ≥ +0.020`. All floors
re-derive to 4dp from the raw trainlogs with an independently written parser, every freeze-block hash
matches disk, and every collision path is verified absent. Because the forward is deterministic (no RNG
frame sampling via `np.linspace`, no dropout, bf16+sdpa, `no_grad`), the grid is hardcoded in a frozen
`CELLS` constant (no sweepable flags), the executor transcribes raw per-seed numbers with the verdict
rendered independently, and the tie-break is constructed test-safe, the motivated-executor attack surface
(sweep layers/prompts, protocol/metric shop, peek at test, bury a regression, clobber the floor) is closed
by construction. Novelty is repeatedly deferred to the USER as a D7-thin performance/robustness row.

---

## CHECK-BY-CHECK

### 1. Extractor variant — **PASS (fork source untouched; changes exactly in-scope)**

- **Fork source untouched.** `git status --porcelain` on `generate_VideoMLLM_embedding_lora_HF.py` is
  **empty** and `git diff --stat HEAD` is **empty** ⇒ byte-untouched; its sha256
  `b6b61a3fa4214f28d2098ca305ca9a981934445afc81ccc5ac3939f8c1fb0ec6` **matches** the prereg §5.2 pin.
  `_encode` is byte-identical between `generate_VideoMLLM_embedding_HF.py` and `..._lora_HF.py`
  (independently confirmed: extracted `_encode` blocks compare equal).
- **Variant changes = exactly the declared set.** `generate_VideoMLLM_embedding_readout_HF.py` (sha
  `ef05f3d4…`, matches §5.1):
  - **Multi-layer harvest:** `_encode_readout` runs ONE forward and reads `out.hidden_states[L][0]` for
    each `L` in `[LAYER_FINAL=28, LAYER_MID=24]`, preserving the masked-scatter invariant
    `assert last_hidden.shape[0] == input_ids.numel()` (the deployed L306 guard).
  - **`last_token` span:** `_pool_span(..., span="last_token")` → `pooled = last_hidden[-1]`; the
    `prefix`/`response` branches are the deployed pooling math verbatim (same `im_start_id` boundary via
    `positions[-1]`, `last_hidden[:end].mean` / `last_hidden[start:].mean`, then `float()`+
    `F.normalize(p=2,dim=0)`+`detach().cpu()`).
  - **One-word prompts (two module constants):** `IMG_INSTRUCTION_OW = "Describe this video in one word:"`
    and `TEXT_OW_TAIL = "\nSummarise the above in one word:"`; the text one-word prompt is
    `TEXT_INSTRUCTION + "\nTitle: … \nTranscript: …" + TEXT_OW_TAIL`. The baseline `IMG_INSTRUCTION` /
    `TEXT_INSTRUCTION` are byte-identical to the deployed constants (verified) — the R0/R1 forwards match
    the deployed prompt exactly, and R2/R3's prompts are pinned **verbatim** matching §1.1, are **label-
    free** (no gold, no OCR), and tokenize as ordinary text.
  - **Out-tag plumbing:** `--out_model_base_tag` + the frozen `CELLS` table drive four caches
    `{outname}_<BASE>-ro_{L28,L24,ow_L28,ow_L24}.pt`; the un-suffixed deployed tag is **never** produced
    (`out_tag = "{}-{}".format(base_tag, suffix)` at L518 always carries a suffix).
  - **Safety guard added:** `assert model.config.num_hidden_layers == LAYER_FINAL` before any extraction.
- **`py_compile` on A + C = COMPILE_OK; `bash -n` on B = SYNTAX_OK** (re-run this review).

**LAYER-INDEXING RULING (the named risk): CORRECT, no off-by-one.** With `num_hidden_layers=28` (read from
`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/.../config.json`; `hidden_size=3584`),
`out.hidden_states` is a length-29 tuple (index 0 = embedding output, indices 1..28 = decoder-layer
outputs), so `hidden_states[28]` is byte-identical to the deployed `hidden_states[-1]` (R0 final) and
`hidden_states[24]` is the intended VidVec-style intermediate at 24/28 ≈ 0.857 relative depth; the runtime
`assert num_hidden_layers == 28` makes both pinned indices safe against any model-swap.

### 2. R0 bit-exact gate — **PASS (decidable; conservative failure direction)** — see Note 1

- The screen's GUARD 1 loads the banked deployed cache (`base_tag`, no suffix) and the R0 recompute
  (`base_tag + "-ro_L28"`) for **both** `train` and `dev_seen`, both streams, and computes
  `di = float((bimg - rimg).abs().max())`, `dt = …`, `exact = bool(di == 0.0 and dt == 0.0)`; the gate is
  `R0_bit_exact_pass = all(v["exact"] …)`.
- **Decidability ruling:** the gate is fully decidable — a hard boolean requiring **exact float equality to
  zero** of the max absolute element-wise difference across every row of train+dev on both `img_feats` and
  `text_feats` (i.e. byte/float-exact, not an epsilon tolerance). It is the correct G-repro anchor:
  bit-exact R0 certifies THIS extraction run reproduces the deployed forward, which is the necessary
  precondition for the winner cell (extracted off the same forwards) to be a pure readout change pairable
  against the banked 13150/13241 heads. A mismatch is pre-declared **VOID / investigate** (§4.1a — never
  silently proceed), so the gate can only ever **block**, never manufacture a pass. `advance` is
  `(best Δacc ≥ 0.02) AND R0_bit_exact_pass`, so a failed gate also blocks promotion. See Note 1 for the
  FP-reproducibility caveat.

### 3. $0 screen — **PASS**

- **Vote byte-verbatim.** `readout_screen._weighted_signed_vote` is byte-identical (modulo docstring) to
  `cross_channel_router_gate.py:73-79` (`w = np.arange(1,TOPK+1)[::-1]…[:k]`, `lm = (nb_lab*2-1)*nb_sim`,
  `sum(lm*w)/sum(w)`), `TOPK=20` in both. `knn_votes` reproduces the deployed faiss `IndexFlatIP` +
  `normalize_L2` + `TOPK+1`/`keep = idx != i` LOO exactly.
- **Fused key + two arms.** `fused_key` is the LP §1 recipe (L2-norm each stream → concat 7168 → L2-renorm;
  torch re-implementation — Note 2). Two arms match ISR §0.2: `loo` (memory = train∪dev, self-exclusion,
  score dev) and `devtrain` (memory = train, dev queries, score dev). Fused key is the decision object;
  img-only/text-only are diagnostics.
- **Test hard-blocked, two ways.** `load_cache` asserts `split ∈ {"train","dev_seen"}` ("test-touch
  blocked"); every call site passes only `train`/`dev_seen`; `main()` asserts `CUDA_VISIBLE_DEVICES == ""`.
  (The extraction sbatch does produce the `test_seen` `-ro_*` caches for the eventual verdict head, but the
  screen never opens them — producing a feature cache is not a test-touch; evaluating a metric on test is,
  and that happens only at the verdict head.)
- **Screen rule pre-declared, single winner.** `ADVANCE_BAR = 0.020`; the winner is the best fused-key dev
  `Δacc` over R0 across {R1,R2,R3} in either arm; `advance = (best Δacc ≥ 0.02) AND R0_bit_exact_pass`;
  `verdict = "PROMOTE" if any dataset advances else "KS-readout-dead"`. Perm-null (≥200) + bootstrap (1000)
  are computed and written as **validity guards** (not folded into the hard advance boolean — DEV-4). One
  family, one bite, one winner, one test-touch.

### 4. Floors — **PASS (independently re-parsed; all 4dp-exact; line-accurate)**

Re-parsed with a freshly written parser (val-sel = epoch ≥ warmup 5 max `Val_Retrieval` acc, roc
tie-break → that epoch's `Test_Retrieval`; final = max epoch):

| leg | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | mean acc/F1 | prereg |
|---|---|---|---|---|---|---|
| ZH R0 generic-LoRA (13150) | val-sel | 0.8322/0.8023 | 0.8255/0.7956 | 0.8389/0.8065 | 0.8322/0.8015 | ✓ |
| ZH R0 generic-LoRA (13150) | final | 0.8456/0.8181 | 0.8389/0.8113 | 0.8523/0.8226 | 0.8456/0.8173 | ✓ |
| HateMM R0 curric (13241) | val-sel | 0.8791/0.8730 | 0.8744/0.8678 | 0.8791/0.8724 | 0.8775/0.8711 | ✓ |
| HateMM R0 curric (13241) | final | 0.8791/0.8730 | 0.8791/0.8724 | 0.8791/0.8724 | 0.8791/0.8726 | ✓ |
| ZH CLIP context (13115) | val-sel/final | — | — | — | 0.8076/0.7676 · 0.8143/0.7720 | ✓ |
| HateMM CLIP context (12850) | val-sel/final | — | — | — | 0.8202/0.8085 · 0.8124/0.7936 | ✓ |

Every per-seed value, selected epoch (ZH 20/26/19; HateMM 29/14/10), and 3-seed mean bit-matches §2.1/§2.2
to 4dp; the context floors bit-match §2.3 and `CAND2_CURRICULUM_PREREG.md §2`. The §2.1/§2.2 line citations
(e.g. 13150 s0 L199/L272, 13241 s0 L301) point exactly at the `Test_Retrieval … macroF1` lines (`grep -n`
confirmed). The ZH B3 leg is correctly flagged MARGINAL (final +0.0313 acc 3/3 but seed2 +0.0201 < the
per-seed +0.030 bar; val-sel +0.0246 acc FAIL) — the leg a readout gain must strengthen; HateMM is the
project-best hold (val-sel 0.8775).

### 5. Bars + multiplicity — **PASS**

- **§3.3 verdict bar** is quoted verbatim from `exp-encoder-3seed.md:73-85` (verified line-by-line: per-seed
  paired δ, 3-seed mean±std + sign, n=3 paired-t as effect-size-descriptor-only / no significance claim,
  pass = Δacc ≥ +0.030 AND Δmacro-F1 ≥ +0.030 AND 3/3 sign, headline needs ≥2 datasets, both protocols
  judged separately, verdict written exactly "final-epoch: …; val-selected: …"). Treatment = the promoted
  winner cell; control = the banked deployed R0 arm (13150 ZH / 13241 HateMM). Both protocols judged
  independently ⇒ no protocol/metric shopping.
- **§3.4 KS-regression** = mean Δacc ≤ −0.014 on a held leg (HateMM), decidable from raw per-seed numbers;
  the ±0.014 band = the largest banked head-seed spread (CAND2 §2.3).
- **Multiplicity.** The whole {R1,R2,R3} family is one pre-registered bite, one winner, one test
  measurement; the grid is hardcoded in a frozen `CELLS` constant (DEV-3) so no submit-time flag can widen
  it into a layer/prompt sweep. The one-word cell is a **single** readout — the prereg quotes the
  distinction (title scope, F0.6: "SINGLE readout, NOT multi-prompt ensembling") and explicitly leaves the
  ensembling micro-ruling **unconsumed**.

### 6. Collision / cost / submit — **PASS**

- **Collisions ABSENT on disk (re-check at submit):** `data/CLIP_Embedding/{MHC_zh,HateMM}/*-ro_*.pt` = 0;
  `logging/Retrieval/*/RAC_video_readout*` = 0; `slurm/logs/*-ro_*.trainlog` = 0. Banked deployed caches
  present with sha16 matching §5.2 exactly (ZH train `b2e8e78d19c71d2c` / dev `4c07af75098391c9` / test
  `4e107bf65f58745a`; HateMM train `5e80f39327a74314` / dev `46ee4fd9fcaec80b` / test `b50ae4ecb077a833`).
- **Clobber-impossible.** The extractor writes only `-ro_*`-suffixed tags (L518/L525); it never writes
  `{split}_<BASE>.pt`, so the banked R0 caches cannot be overwritten. Per-dataset tags are hardcoded in the
  sbatch `CONFIGS` (DEV-2) so no submit-time arg can point at the wrong adapter.
- **Cost ledger** ~2.0 GPU-h to reach the $0 screen (4 forwards/item × 8 frames × 2 datasets), verdict head
  adds <0.1 GPU-h only if promoted ⇒ ~2.0/2.1 total — sane, matches the recon §8 ledger.
- **Submit plan.** ONE combined sbatch (ZH then HateMM sequential), `--cpus-per-task=8 --mem=64G
  --gres=gpu:a100:1`, peak 8 CPU / 64 G / 1 GPU — within the 16/128/2 cap and never two 16-CPU jobs.
  **NO `--time`** (L8: "intentionally NO --time"). `conda activate HateVideo`; `PENDING (JobHeldUser)` →
  **wait for auto-release, never force** (§6). The **readout chain submits BEFORE the bidir chain**: §6 and
  the title scope pre-declare that the bidirectional-encoder chain submits **after** this readout chain
  clears its extraction+screen.
- **Single-test-touch.** The promote-time head's 3 seed reads on the winner cache are the ONLY budgeted
  readout-cell test evaluations; zero test-touch before the independent verdict; no job submitted by the
  prereg author.

### 7. Deviations §11 (DEV-1..DEV-5) — all favorable / neutral / documented

- **DEV-1** (tie-break made test-safe via a `test_seen := dev_seen` throwaway `-tb` copy) — **FAVORABLE,
  closes a real gap.** Verified the premise: `run_rac.py:745` prints `Test_Retrieval … macroF1` **every
  epoch** unconditionally (reads `test_seen_dl`, no skip flag), so a naive tie-break head-retrain WOULD
  open the real test cache before promote. The resolution (throwaway cache whose `test_seen` is a copy of
  `dev_seen`; winner chosen by `Val_Retrieval` dev acc only; the dev-on-dev "Test" lines discarded unread)
  keeps the single budgeted real test-touch at the verdict head, by construction. Optional, not auto-run.
- **DEV-2** (one combined extraction job, hardcoded per-dataset tags) — **FAVORABLE/neutral.** One
  hash-frozen artifact, one submit, no submit-time adapter typo, never two concurrent CPU jobs, partial
  progress recoverable. Mirrors the frame16 DEV-2 anti-clobber discipline.
- **DEV-3** (grid hardcoded in a frozen `CELLS` constant, not sweepable `--readout_*` flags) —
  **FAVORABLE.** Makes the frozen grid a code-level invariant; no runtime flag can widen it into a
  forking-path sweep. One-word prompts are frozen module constants.
- **DEV-4** (screen bar = +0.020; perm-null/bootstrap reported as validity guards, not in the hard advance
  boolean) — **DOCUMENTED/faithful.** `advance = (Δacc ≥ 0.02) AND R0-bit-exact`; the guards are computed
  and written for reviewer weighing (a Δ inside the perm-null band is flagged for skepticism/tie-break, not
  auto-killed). Matches ISR/LP treatment.
- **DEV-5** (R4 Echo excluded; verdict pairs vs the banked deployed R0, no fresh R0 head) — **DOCUMENTED.**
  Valid because `ro_L28` is bit-exact to the deployed cache (gated in §1.2c/§4.1a), so a fresh R0 head would
  reproduce 13150/13241 seed-for-seed; EN excluded per F0.4. (If the R0 bit-exact gate ever VOIDs, this
  pairing basis is gone — see Note 1.)

---

## NON-BLOCKING NOTES (for the executor / verdict reviewer)

1. **R0 bit-exact gate uses exact float-equality-to-zero; bf16+sdpa cross-run bitwise reproducibility is not
   guaranteed by PyTorch.** The gate correctly requires `max|Δ| == 0.0`. On the same GPU with the same
   libraries a deterministic `no_grad` forward usually reproduces bit-exact, but PyTorch does **not**
   guarantee it (kernel/reduction-order nondeterminism), so a benign nonzero FP drift could force the
   pre-declared **VOID / investigate** path and strand the ~2 GPU-h extraction without a verdict. This is
   the **conservative** direction — the gate can only ever block, never fabricate a pass, and a failed gate
   also correctly invalidates the DEV-5 banked-R0 pairing basis (so stopping is the right response). The
   note is only that the executor should be prepared to **investigate a small nonzero Δ as a hardware/lib
   drift** rather than assume a code bug, and MUST NOT relax the gate to an epsilon to "rescue" a run — a
   nonzero R0 Δ means the winner cell is not a clean readout of the deployed forward and the paired
   comparison is invalid. Substantive but non-blocking (safe by construction).

2. **The $0 screen's `fused_key` is the LP §1 recipe re-implemented in torch, not a byte-verbatim fork.**
   `lp_gate.fused_key` uses numpy (`np.clip(n, 1e-12, …)`); `readout_screen.fused_key` uses
   `torch.nn.functional.normalize(p=2, dim=1)` (default eps 1e-12). Mathematically identical (L2 each
   stream → concat 7168 → L2-renorm); float paths may differ at the ~1e-7 level, irrelevant to a
   sign-thresholded kNN vote. The **load-bearing** vote (`_weighted_signed_vote`) IS byte-verbatim from
   `cross_channel_router_gate.py:73-79`. Non-material.

3. **§2.3 context floors are written acc/F1 order while the §2.1/§2.2 tables are labeled F1/acc.** The
   context floors "ZH CLIP 0.8076 / 0.7676" and "HateMM CLIP 0.8202 / 0.8085" are acc-first (matching
   `CAND2 §2.1`), whereas the §2.1/§2.2 tables use the labeled "Test F1 / acc" order. All values are
   correct and independently reproduced; only the cross-section column order is inconsistent. Cosmetic.

---

## HASH-FREEZE

Recorded in `refine-logs/READOUT_FREEZE.md` (prereg NOT modified, per review mandate). All freeze-block
shas re-verified on disk at freeze time and **match**: prereg self-sha
`f6d43096332acb2e9f9743f5e51967f6a93b3006770be3520effc13929aac543`, A `ef05f3d4…`, B `948db851…`,
C `f56badb6…`, fork source `b6b61a3f…`, vote source `d4adf545…`, head-clone source `00d9e995…`; banked
deployed caches present and untouched (sha16 as §5.2).

**Reviewer statements:** ZERO GPU/SLURM/Modal spent — CPU-only login-node re-parse of the banked
13150/13241/13115/12850 trainlogs with an independently written parser, plus `sha256sum` / `py_compile` /
`bash -n` / `ls` collision checks and a `num_hidden_layers` config read (seconds); no held-out test metric
produced; `state/` and `autoresearch/goal_mllm_plus3/state/` not touched; the prereg was **NOT** modified;
no job submitted; not pushed. Cloud/external numbers were never mixed with local G-repro numbers.
