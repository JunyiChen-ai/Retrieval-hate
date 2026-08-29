# Independent 0-Context Pre-Registration Review — `BIDIR_STAGE1_PREREG.md`

**Reviewer:** independent 0-context pre-registration reviewer (no prior project context; adversarial mandate;
zero user interaction; no job submitted; prereg NOT modified).
**Date:** 2026-07-25 NZST. **CPU-only** (no GPU/SLURM/Modal spent; `state/` not touched).
**Target:** `refine-logs/BIDIR_STAGE1_PREREG.md` (introduced at commit `a7bb2a1`; on-disk sha256
`3c532e5370e52b2ed53e0bcc2ad63d2958823f5aca0e6f710495d8cf55565142`; current HEAD `1b3e0c6`, a later
sibling prereg — the bidir files are committed and clean).
**Implements recon:** `refine-logs/BIDIR_SURGERY_FORENSIC_RECON.md` (commit `ec5add8`).
**Method:** every load-bearing fact re-derived from primary artifacts on disk — the pinned patch module read
line-by-line against the **installed** `transformers 4.49.0` `modeling_qwen2_5_vl.py` (env `HateVideo`), the
SDPA re-causalization path traced through the model forward, the CPU non-causality self-test **run
independently** (`python src/utils/bidir_patch.py`), all four comparison floors re-parsed from the raw 13150 /
13241 / 13115 / 12850 trainlogs with an **independently written** parser, the `run_one` head block hashed and
diffed across all three sbatch, every freeze-block hash recomputed, all collision paths `ls`-checked on disk, and
both new sbatch `bash -n`'d. The prereg's and recon's numbers were treated as untrusted until independently
reproduced.

## VERDICT: **APPROVED-WITH-NOTES** (all notes non-blocking)

The prereg is patch-correct against the installed transformers source (the SDPA re-causalization trap is genuinely
defeated), hash-integral, floor-faithful to 4dp on both paired anchors, same-code-paired (head `run_one`
byte-identical to BOTH banked causal runners), clobber-safe (distinct hardcoded `-bidir` out-tags), leakage-clean,
veto-compliant (Stage-1 has no training), collision-free on disk, and its kill-ladder is fully decidable from raw
logs by a 0-context verdict reviewer with no interpretive freedom. The single manipulated variable — the LLM
decoder's causal attention topology — is genuinely single: the runner is a thin fork that imports the causal
extractor's pooling operator VERBATIM and inserts exactly one `apply_bidir_mask(model)` call after the LoRA merge
and before any forward, so the within-seed paired delta isolates the attention mask alone. The load-bearing
correctness point (the recon's own flagged gotcha) holds under audit: for a single unpadded sequence the stock
SDPA path returns a **None** mask and silently re-causalizes via `is_causal=True` at `:989`; the patch fully
replaces `_update_causal_mask` with a **NON-None all-zeros** 4D additive mask, forcing `is_causal=False` and a
zero additive bias = bidirectional — verified against the installed source AND empirically via the CPU self-test
(early-token change under future-token perturbation: `0.000e+00` causal → `6.4e-02` bidir = PASS). Stage-2 MNTP is
correctly held out as a CONDITIONAL FUTURE prereg with funding gated on the Stage-1 outcome shape, and D7 novelty
is deferred to the user throughout. The notes below are a floor-provenance granularity item (Note 1), a loose
"nested" descriptor (Note 2), a spend-threshold clarification (Note 3), and a self-test coverage observation
(Note 4); none affects decidability, leakage, or the honesty of any bar, and none can be used to manufacture an
unsupported pass. **Cleared to freeze + single-submit** (subject to the author's own GO-IF gates: the mandatory
codex/patch review — for which this review discharges the transformers-source check — and the one-line D7 user
sub-ruling; both are orchestrator pre-conditions, not the reviewer's to clear).

---

## Rationale (one paragraph)

The cell measures the one axis every banked Qwen arm shares and none ever varied — the LLM decoder's causal
attention topology — by flipping `is_causal` to bidirectional at inference on the **same** banked LoRA adapters
(ZH `logging/lora/MHC_zh`, HateMM `logging/lora/HateMM_curric`), same prefix-mean readout, same 8 frames, paired
within head-seed against the banked causal-LoRA arms (ZH 13150, HateMM curric 13241), dual-protocol. Validity
hinges on two properties, both of which hold under audit. First, the manipulation is single and correct: the
`_bidir_update_causal_mask` function is a signature-exact drop-in for `Qwen2_5_VLModel._update_causal_mask` (the
model forward calls `self._update_causal_mask(...)` at `:1177` and threads the result to every decoder layer as
`attention_mask` at `:1210`), it returns an all-zeros `[bsz,1,seq,seq]` additive mask (dtype/device from
`input_tensor`, 2D-padding fold a no-op at bsz=1), which makes `Qwen2_5_VLSdpaAttention.forward` evaluate
`is_causal = False` (`:989`, because the mask is non-None) and pass a zero additive bias as `attn_mask` (`:995`)
= attend-everywhere; the patch binds only to `model.model`, leaving `model.visual` (the already block-diagonal
vision tower, `:265-269`, `:1519-1520`) untouched, and asserts `sdpa` to close the flash re-causalization path
(`:904`) with a defensive `is_causal=False` loop as belt-and-suspenders. Second, everything downstream of the mask
is byte-identical to the banked causal arm: the runner imports `read_gt`/`process_split`/`SPLIT_TO_OUTNAME` (hence
`_encode`, `load_video_frames`, and the fixed label-free `IMG_INSTRUCTION`/`TEXT_INSTRUCTION`) verbatim from the
unedited causal extractor (`b6b61a3f…`), the masked-scatter length invariant at `:306` is independent of the
attention mask, the head `run_one` block is byte-identical (sha `13e34e4e…`) to both `enc3seed.sbatch` and
`enc3seed_lora_curric.sbatch`, the loader routes `MHC_zh`/`HateMM` → `load_feats_MHC` → `{split}_{model}.pt`, and
the distinct hardcoded `-bidir` out-tags cannot clobber the banked causal caches (verified absent on disk). All
four floors re-derive to 4dp with an independently written parser (both paired anchors exact; the B3 cross-check
Δacc +0.0313 final / +0.0246 val-sel / ΔmF1 +0.0453 reproduces), the frozen forward is deterministic (np.linspace
sampling, no RNG, `no_grad`, bf16+sdpa) so there is no single-encoder-draw confound and the single-SFT-draw caveat
cancels in the pairing, and the executor transcribes raw per-seed numbers with the verdict rendered independently
— so the motivated-executor attack surface (re-draw, protocol/metric shop, bury a regression, clobber the floor)
is closed by construction. The honest prior is LOW (~10–15%, Law-I-discounted) and pre-declared; KS-bidir-dead is
stated as the most likely outcome; D7 novelty is the user's ruling throughout.

---

## CHECK-BY-CHECK

### 1. The patch vs installed transformers 4.49.0 — **PASS (load-bearing; SDPA trap defeated, verified + run)**

Read `src/utils/bidir_patch.py` (`36cedbac…`) and `generate_VideoMLLM_embedding_bidir_HF.py` (`03f39e09…`)
against the installed `modeling_qwen2_5_vl.py` (2112 lines, env `HateVideo`). Every cited line matches:

- **(a) SDPA re-causalization trap — genuinely defeated.** `Qwen2_5_VLAttention.__init__` hard-codes
  `self.is_causal = True` (**:723**; no config flag — a monkey-patch is mandatory). For SDPA the causality comes
  from two places: the 4D additive mask consumed as `attn_mask=causal_mask` (**:995**) and the fallback
  `is_causal = True if causal_mask is None and q_len > 1 else False` (**:989**). For a single unpadded sample the
  stock `_update_causal_mask` returns **None** — `_ignore_causal_mask_sdpa` at **:1278/1285** — so `:989` sets
  `is_causal=True` and SDPA masks causally INTERNALLY (**this is the trap: nulling the mask stays causal**). The
  patch's `_bidir_update_causal_mask` returns a **NON-None** all-zeros `[bsz,1,seq,seq]` tensor, so `causal_mask`
  is non-None ⇒ `:989` evaluates `is_causal=False` and the zero additive bias at `:995` imposes no masking ⇒
  bidirectional. Because the patch **fully replaces** `_update_causal_mask`, `_ignore_causal_mask_sdpa` /
  `_unmask_unattended` (`:1278`,`:1323`) never run (no interaction hazard). **Interception point is correct:** the
  model forward calls `causal_mask = self._update_causal_mask(...)` at **:1177-1179** and passes it to every
  decoder layer as `attention_mask=causal_mask` at **:1210**, so binding on the `model.model` instance intercepts
  exactly. The `_bidir_update_causal_mask` signature is positionally exact to the call
  (`attention_mask, input_tensor, cache_position, past_key_values, output_attentions`). The real-path case (the
  processor returns a 2D all-ones `attention_mask`, not None) is also covered: the fold `(1-ones)*min = 0` is a
  no-op ⇒ still all-zeros ⇒ bidirectional. **I ran the CPU self-test independently**
  (`python src/utils/bidir_patch.py`, `CUDA_VISIBLE_DEVICES=""`, transformers 4.49.0): mask shape `(1,1,6,6)`
  all-zero=True; `d_causal(pos0, future perturbed) = 0.000e+00`; `d_causal(last, sanity) = 1.042e+01`;
  `d_bidir(pos0) = 6.387e-02`; **VERDICT: PASS** — reproducing the prereg §4.4.1 numbers.
- **(b) Vision tower/merger untouched.** `apply_bidir_mask` binds only to `decoder = model.model`
  (`Qwen2_5_VLModel`, **:1520**); `model.visual` (**:1519**) is the vision transformer whose attention builds a
  block-diagonal `cu_seqlens` mask (**:265-269**, full-within-window, never causal) and is not modified. ✓
- **(c) Binding works on BOTH arms in the pinned call chain.** The runner is a single thin fork whose `main()` is
  faithful to the causal extractor's `main()` (identical `from_pretrained(attn_implementation="sdpa")`,
  `PeftModel.from_pretrained`+`merge_and_unload`, processor, splits loop, `process_split`, save) with **exactly one
  added line**: `apply_bidir_mask(model)` at **:71 — after the merge, before any forward**. `gen_embed_mllm_bidir.sbatch`
  passes `--lora_dir logging/lora/MHC_zh` and `--lora_dir logging/lora/HateMM_curric` (both `--num_frames 8`), so
  each arm loads base → merges its OWN banked adapter → applies the patch. After `merge_and_unload`, `model` is the
  base `Qwen2_5_VLForConditionalGeneration` and `model.model` is the same `Qwen2_5_VLModel` instance (the patch is
  applied post-merge; smoke §4.4.2 asserts `type(model.model).__name__=="Qwen2_5_VLModel"` at runtime). The patch is
  bound before `.to(device)` — harmless, since the mask is created at forward time on `input_tensor.device` and the
  bound method / `is_causal` attribute survive the device move. ✓
- **(d) The self-test would catch a silently-ineffective patch.** It is a genuine backward-propagation
  discriminator: perturb the LAST (future) token, measure the change at position 0. Under causal, position 0 cannot
  see the future ⇒ change ~0 (measured `0.000e+00`); under the patch, position 0 attends to the future ⇒ change > 0
  (measured `6.4e-02`). A patch that failed to flip causality would leave `d_bidir(pos0) ≈ 0` and the assertion
  (`d_bidir_pos0 > 1e-4`) FAILS. The `d_causal_last > 1e-4` sanity leg confirms the perturbation is real; the
  mask-shape/all-zero asserts confirm the structural form. The propagation-backward-iff-active property holds. ✓
- **(e) Flash-attention exclusion asserted.** `apply_bidir_mask` asserts
  `model.model.config._attn_implementation == "sdpa"` (`bidir_patch.py:82`) and the runner loads with
  `attn_implementation="sdpa"` (`:48`); a silent flash fallback would re-causalize via `is_causal=self.is_causal`
  at **:904**, which the mask patch alone does not cover — the assert is the guarantee, and the defensive
  `is_causal=False` loop over decoder `Qwen2_5_VLAttention` modules covers it belt-and-suspenders (harmless for
  SDPA, which recomputes `is_causal` locally at `:989`). ✓

### 2. Floor re-derivation — **PASS (independently re-parsed; both anchors exact to 4dp)**

I wrote a standalone parser implementing the `enc3seed.sbatch` embedded rule (val-sel = epoch ≥ warmup 5 with max
`Val_Retrieval` acc, roc tie-break → that epoch's `Test_Retrieval` acc/macroF1; final = max epoch) and ran it on
the raw trainlogs. **Both PAIRED ANCHORS (the formal bars) reproduce §2.1/§2.2 exactly** — every per-seed value,
every selected epoch, and both 3-seed means:

| arm | protocol | re-parsed mean acc/F1 | prereg |
|---|---|---|---|
| **ZH generic-LoRA 13150 (§2.1 anchor)** | val-sel / final | 0.8322/0.8015 · 0.8456/0.8173 | ✓ ✓ |
| **HateMM curric 13241 (§2.2 anchor)** | val-sel / final | 0.8775/0.8711 · 0.8791/0.8726 | ✓ ✓ |
| ZH frozen-CLIP 13115 (§2.3 context) | val-sel / final | 0.8076/0.7676 · 0.8143/0.7720 | ✓ ✓ |
| HateMM frozen-CLIP 12850 (§2.3 context) | val-sel / final | 0.8202/0.8085 · 0.8124/0.7936 | ✓ ✓ |

Per-seed selected epochs match (ZH 20/26/19; HateMM 29/14/10 — s0 val-sel selects ep29=final so its two rows
coincide, exactly as §2.2 states). The §7.1/§7.2 outcome tables pre-fill the causal floors matched by seed index;
all pre-filled cells match my re-parse. **B3 cross-validation reproduces:** ZH-LoRA − CLIP final Δacc +0.0313 /
ΔmF1 +0.0453, val-sel Δacc +0.0246 (bit-exact vs the §2 claim). The HateMM-CLIP context floor is the
ERRATUM-corrected value (0.8279/0.8172 per-seed peak → 0.8202/0.8085 mean), not the withdrawn 0.8732. All floors
consistent with `FRAME16_PREREG.md §2` and `VISION_UNFREEZE_PREREG.md §2`. See Note 1 on the 12850 provenance
granularity (non-material).

### 3. Same-code + syntax + collision — **PASS**

- **Head `run_one` byte-identical to BOTH banked causal runners.** The `run_one()`…`PY` block of
  `enc3seed_bidir.sbatch` is 41 lines, sha256 `13e34e4e93c6a76988557e1c609fd54e0353c627fd36eb1c5b9e26ed187c3feb`,
  **identical** (`diff` empty) to the same block in `enc3seed.sbatch` (12850/13150 runner) **and**
  `enc3seed_lora_curric.sbatch` (13241 runner). The only manipulated head variables vs the banked causal controls
  are `--model` (the `-bidir` cache tag) and `--group_name` (`RAC_video_bidir`) plus derived `--exp_comment`. The
  config pins are verbatim (`--batch_size 64 --lr 0.0001 --epochs 30 --topk 20 --proj_dim 1024 --map_dim 1024
  --dropout 0.2 0.4 0.1 --fusion_mode align --hard_negatives_loss True --no_hard_negatives 1 --metric cos --loss
  triplet --hybrid_loss True --warmup 5 --lambda_seg 0 --force False`, archive OFF via the default).
- **Runner faithful to the causal extractor.** `generate_VideoMLLM_embedding_bidir_HF.py` imports the operator
  verbatim; its `main()` differs from `generate_VideoMLLM_embedding_lora_HF.py:main()` only in the single
  `apply_bidir_mask(model)` line and cosmetic prints. The causal extractor `b6b61a3f…` is **byte-unchanged**
  (matches §5.2). `python -m py_compile` on both new python files = OK.
- **Syntax.** `bash -n` on `gen_embed_mllm_bidir.sbatch` and `enc3seed_bidir.sbatch` = SYNTAX_OK.
- **Collision list verified ABSENT on disk (re-check at submit):**
  `data/CLIP_Embedding/MHC_zh/*LoRA-bidir*.pt` (none), `data/CLIP_Embedding/HateMM/*LoRA-curric-bidir*.pt` (none),
  `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_bidir*` (none), `slurm/logs/enc3s_*bidir*seed*.trainlog` (none). The
  banked causal caches are PRESENT and untouched (ZH `…-LoRA_HF.pt` dated Jul 2; HateMM `…-LoRA-curric_HF.pt` dated
  Jul 18). Distinct hardcoded `-bidir` out-tags ⇒ the banked caches cannot be clobbered by any submit-time env var.
  `--force False` hits `run_rac.py:905-908` (`raise Exception("Output path already exists, aborting...")`) only if
  the fresh `RAC_video_bidir` path already exists (it does not) — so it never trips, and if it ever did it aborts
  rather than overwrites. The banked adapters are loaded/merged in-memory (READ-ONLY, never written).

### 4. Bars + kill ladder — **PASS (fully decidable; sign-based per house n=3 discipline)**

- **§3.1 decision rule** is quoted faithfully from `exp-encoder-3seed.md:73-85` (verified line-by-line: per-seed
  paired δ, 3-seed mean±std + sign, n=3 paired-t as effect-size descriptor only / no significance claim, pass =
  Δacc≥+0.030 AND ΔmF1≥+0.030 AND 3/3 sign, headline needs ≥2 datasets, both protocols judged separately, exact
  write-up string). Treatment = bidir, control = CAUSAL-LoRA (the correct paired-control adaptation of the source's
  Qwen−CLIP framing).
- **Ladder internal consistency.** Mutually-exclusive, jointly-exhaustive outcome bins on one measurement:
  **KS-bidir-dead (§3.2)** = neither protocol yields (mean Δacc > 0 AND acc sign 3/3) ⇒ KILL + Stage-2 MNTP
  auto-defunded; **DEGRADE (§3.3)** = mean Δacc ≤ −0.014 on BOTH protocols ⇒ "Llama-pattern, MNTP-motivated"
  (perf-dead Stage-1 but MNTP becomes a SEPARATE user funding decision, the one carve-out overriding §3.2's
  auto-defund); **weak-limbo** = some positive but < +0.010 / not 3/3 ⇒ MNTP not funded; **CONTINUE (§3.4)** =
  mean Δacc ≥ +0.010 AND 3/3 on ≥1 protocol ⇒ MNTP fundable (internal spend gate, not a paper claim);
  **FORMAL (§3.5)** = +0.030/+0.030 AND 3/3 under EACH protocol vs the causal arm ⇒ goal-facing; ≥2 datasets =
  headline. FORMAL ⟹ CONTINUE, so the ordering is coherent; each bar is a fixed pre-registered threshold on raw
  per-seed numbers with no interpretive freedom, both protocols judged independently (fixed §7.3 write-up) ⇒ no
  protocol/metric shopping. **Sign-based, not bootstrap-CI** — DEV-3 pins the house n=3 no-bootstrap rule
  (`exp-encoder-3seed.md:78-79`), matching the frame16 precedent; it can only ever fire on the negative side and
  cannot manufacture a pass. **Stage-2 MNTP is correctly gated as a separate CONDITIONAL FUTURE decision** — no
  MNTP artifact is authored or submitted here, funding is a downstream spend decision keyed to the Stage-1 outcome
  shape, and even a strong DEGRADE leaves MNTP as a user-visible decision rather than auto-funding it.
- The HateMM FORMAL bar is very demanding (curric anchor 0.8775 val-sel = project best ⇒ +0.030 → ~0.9075), a
  favorable/conservative consequence of DEV-4 pre-declared in F0.5; the prereg is honest that ZH is the live perf
  surface and HateMM can at most sharpen an already-passing leg. Not a defect (see Note 3).

### 5. Leakage / veto / single-test-touch — **PASS**

- **Label-free extraction.** The runner reuses the causal operator's fixed `IMG_INSTRUCTION`/`TEXT_INSTRUCTION`
  (`process_split` → `_encode`) applied identically to every video; the label enters only the saved `labels`
  tensor — gold never enters the encoder path. Stage-1 has NO training ⇒ the single-dataset own-train-split veto
  (F0.4) is trivially clean; the head trains on each dataset's own train split only (identical to the causal arm).
  No OCR channel, no external API, raw videos never leave (`b2_push` copies only `.pt` caches).
- **Own-banked-adapter pairing.** Each bidir arm re-extracts with the SAME banked adapter its causal comparator
  used (ZH `logging/lora/MHC_zh` == 13150; HateMM `logging/lora/HateMM_curric` == 13241; both adapter shas match
  §5.2), so the paired delta isolates the mask topology and the single-SFT-draw caveat cancels between arms.
- **Deterministic extraction (F0.2).** `np.linspace` frame sampling (no RNG), `attn=sdpa`, bf16, `no_grad`, single
  forward ⇒ no stochastic encoder draw; the reported ±band is purely head-seed variance, symmetric with the causal
  arm. Argument sound.
- **Single-test-touch accounting.** The 3 head-seed reads per dataset are the ONLY budgeted bidir-encoder test
  evaluations = exactly ONE new single-test-touch per dataset; zero test-touch before the independent verdict (the
  author ran nothing on held-out test; my CPU self-test uses a tiny RANDOM model, no test data). Prior HateMM/ZH
  test exposures under the identical enc3s protocol are correctly enumerated (F0.1); spot-checked jobs 13150,
  13241, 13235 (F53), 13246 (curric-rep2) all present on disk. These reads are re-measurements, not first
  exposures.

### 6. Cost + submit plan — **PASS**

- **Sequential, chained.** Extraction (`gen_embed_mllm_bidir.sbatch`) → head (`--dependency=afterok:<1>
  enc3seed_bidir.sbatch`); the head cannot start until extraction succeeds ⇒ the two jobs never run concurrently.
- **Aggregate CPU rule respected.** Both sbatch request `--cpus-per-task=8`, `--mem=64G`, `--gres=gpu:a100:1`
  (verified headers); peak footprint with the afterok chain = 8 CPU / 64 G / 1 GPU — within the 16 CPU / 128 G /
  2 GPU cap and **never two 16-CPU jobs in flight** (the 29h-wedge rule). §6 correctly instructs sequencing this
  8-CPU chain AFTER the parallel readout-recon chain clears, or behind it, keeping total in-flight ≤ 16 CPU (at
  most one other 8-CPU job may co-run) — the submit-time discipline is the executor's, and it is stated.
- **NO `--time`** in either sbatch (intentional). `conda activate HateVideo`; `PENDING (JobHeldUser)` → **wait for
  auto-release, never force** stated (§6). Both source `conda.sh` directly and run `disk_guard.sh`. Cost ledger
  ~0.5–0.7 A100-h (extraction dominates; head ~0.03 h). Stage-2 MNTP (~2–4 GPU-h/dataset) is declared CONDITIONAL
  FUTURE and is not authored or submitted here. **No job is submitted by this prereg author** — submission follows
  the independent review + freeze + the F0.3 GO-IF gates (codex patch review + one-line D7 user sub-ruling).

### 7. Deviations §11 (DEV-1..DEV-6) — all favorable / neutral / documented

- **DEV-1** (the recon's "$0 dev screen" folded into the full 3-seed test's KS bar, mirroring frame16) —
  **MATERIAL, favorable**; matches the task's binding instruction and the frame16 house pattern, keeps the head
  runner byte-identical to the banked controls, spends one declared test-touch per dataset (F0.1). The qualitative
  screen (FLAT⇒kill / DEGRADE⇒Llama-pattern / MOVE⇒CONTINUE) is unchanged; only its footing moves from dev to the
  pre-registered test verdict.
- **DEV-2** (dedicated bidir RUNNER importing the causal operator VERBATIM, vs editing the banked extractor or a raw
  instance monkey-patch) — **MATERIAL / provenance-favorable**; editing the banked extractor would change a sha that
  other frozen preregs pin as reused-unchanged. Following frame16 DEV-2 / vision DEV-3, the fork leaves the causal
  extractor byte-unchanged (`b6b61a3f…` verified) and the patch semantics equal the recon's instance bind plus the
  SDPA assert and defensive `is_causal=False` loop.
- **DEV-3** (KILL bar uses SIGN, not bootstrap-CI) — **neutral, house discipline**; pins `exp-encoder-3seed.md:78-79`,
  matching frame16 DEV-1. Only the significance formalism changes; cannot manufacture a pass.
- **DEV-4** (HateMM paired anchor = curric 13241, not generic-LoRA) — **per task, documented**; makes the pairing
  clean (same adapter, mask flipped) but sets a HIGH floor (0.8775 = project best), so the HateMM FORMAL bar is very
  demanding (see Note 3). Frozen-CLIP floors carried as orientation-only context.
- **DEV-5** (Stage-2 MNTP = CONDITIONAL FUTURE; DEGRADE does NOT auto-defund it) — **per task, documented**; FLAT
  auto-defunds MNTP (Law-I), a strong concordant DEGRADE relabels to "Llama-pattern, MNTP-motivated" and leaves
  MNTP a SEPARATE user decision. No Stage-2/MNTP artifact authored or submitted here.
- **DEV-6** (D7 novelty DEFERRED to the user, not asserted) — **per task, documented**; the recon's highest-D7-payoff
  argument is carried as motivation only; this prereg decides the performance clause. The GO-IF gates (codex patch
  review + one-line D7 user sub-ruling before GPU) are flagged as orchestrator pre-conditions the author does not
  clear.

---

## NON-BLOCKING NOTES (for the executor / verdict reviewer)

1. **§2.3 / §9 HateMM-CLIP context-floor provenance granularity.** The prereg attributes the HateMM frozen-CLIP
   orientation floor (0.8202/0.8085 · 0.8124/0.7936) to "job 12850". Job 12850 ran **two** HateMM arms in one job:
   the `openai_clip-vit-large-patch14-336_HF` arm (whose 3 trainlogs give exactly 0.8202/0.8085 val-sel /
   0.8124/0.7936 final — the number the prereg reports) **and** a `Qwen2.5-VL-7B-Instruct_HF` frozen-Qwen arm
   (which gives 0.8729/0.8648 · 0.8682/0.8591). The prereg's number and "frozen-CLIP" label are **correct** (parsed
   from the CLIP trainlog, matching frame16 §2.2), but a reader who re-parses "12850" with the Qwen tag lands on the
   frozen-Qwen numbers. **Why non-blocking:** §2.3 is explicitly orientation-only, NOT a paired anchor — the FORMAL
   bars pair vs the causal-LoRA arms (13150 / 13241), which reproduce exactly. **Recommendation for the executor:**
   when re-verifying the HateMM-CLIP orientation floor at submit, parse
   `enc3s_HateMM_openai_clip-vit-large-patch14-336_HF_seed*_12850.trainlog` (not the `…Qwen2.5-VL-7B-Instruct_HF…`
   ones). Descriptor-granularity only (parallels frame16 Note 1).

2. **§3.6 "nested bars" is an ordering, not literal set-nesting.** KS-bidir-dead / DEGRADE / weak-limbo / CONTINUE /
   FORMAL are outcome bins ordered by increasing positive strength; the `⊂` chain reads as a threshold ordering.
   FORMAL ⟹ CONTINUE holds (Check 4), so the ordering is sound; the "nested"/"carved-out" wording is merely loose.
   Cosmetic (parallels frame16 Note 3).

3. **§3.3 DEGRADE −0.014 threshold and §3.5 HateMM demanding bar are spend/labeling boundaries, not scientific
   claims.** The −0.014 DEGRADE threshold is justified as "the banked between-seed acc spread ≤ ~0.014"; the ZH-LoRA
   banked spread is 0.0134 (val-sel and final) and HateMM-curric is far tighter (≤0.0047), so −0.014 ≈ one ZH spread
   — reasonable as a "clearly below noise" boundary. It only governs the LABEL (KS-bidir-dead auto-defund vs
   Llama-pattern MNTP-motivated) and the downstream MNTP SPEND, never a goal-facing claim; either branch is a
   negative Stage-1 verdict. Likewise the near-unreachable HateMM FORMAL bar (~0.9075) is a conservative consequence
   of pinning the project-best curric anchor, pre-declared in F0.5 (ZH is the live surface). Noted so the verdict
   reviewer records these as spend/labeling decisions, not proofs.

4. **CPU self-test coverage.** `bidir_self_test` binds `_bidir_update_causal_mask` directly and does not exercise
   `apply_bidir_mask`'s `assert_sdpa` or the `is_causal=False` loop; those are covered by the runner (loads sdpa;
   the `[BIDIR] … patch installed` line reports the decoder-module count) and the smoke §4.4.2 real-model
   non-causality check. The self-test's own job — proving the mask function flips causality — is fully discharged
   (verified by independent run). Minor coverage observation; the smoke plan closes the gap. Non-material.

---

## HASH-FREEZE

Recorded in `refine-logs/BIDIR_STAGE1_FREEZE.md` (prereg NOT modified, per review mandate). All freeze-block shas
in §5 re-verified on disk at freeze time and **match**: prereg self-sha `3c532e53…`, A `36cedbac…`, A2
`03f39e09…`, B `0f17fce6…`, C `82a69e74…`; reused-unchanged causal extractor `b6b61a3f…`, head anchor
`dbe3fb81…`, both ZH and both HateMM-curric adapter shas as pinned; banked causal caches present and untouched.

**Reviewer statements:** ZERO GPU/SLURM/Modal spent — CPU-only login-node work: reading the patch against the
installed transformers source, an independently written re-parse of the banked 13150/13241/13115/12850 trainlogs,
hashing / `diff` / `bash -n` / `ls` collision checks, and one CPU (`CUDA_VISIBLE_DEVICES=""`) run of the tiny
random-model non-causality self-test (seconds). No held-out test metric produced; `state/` not touched; the prereg
was **NOT** modified; no job submitted; not pushed.
