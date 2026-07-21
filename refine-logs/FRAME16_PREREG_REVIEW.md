# Independent 0-Context Pre-Registration Review — `FRAME16_PREREG.md`

**Reviewer:** independent 0-context pre-registration reviewer (no prior project context; adversarial mandate;
zero user interaction; no job submitted; prereg NOT modified).
**Date:** 2026-07-21 NZST. **CPU-only** (no GPU/SLURM/Modal spent; `state/` not touched).
**Target:** `refine-logs/FRAME16_PREREG.md` (commit `0b5cbb5`; on-disk sha256
`5c240518217e1ab69cbd52de34c8849450d8b37ad163d68b4247c5f2c791c725`).
**Method:** every load-bearing number re-derived from primary artifacts on disk — the raw 12850 HateMM
trainlogs re-parsed with an **independently written** parser (not the prereg's); the extractor, both new sbatch,
their fork sources, the loader, and `run_rac.py` read directly; every freeze-block hash recomputed; all
collision paths `ls`-checked on disk; both sbatch `bash -n`'d. The prereg's and recon's numbers were treated as
untrusted until independently reproduced.

## VERDICT: **APPROVED-WITH-NOTES** (all three notes non-blocking)

The prereg is hash-integral, floor-faithful to 4dp, extractor-mechanics-correct, same-code-paired (head
`run_one` byte-identical to the banked 8f-floor runner), clobber-safe (a distinct hardcoded out-tag that no
submit-time env var can override), leakage-clean, veto-compliant, collision-free on disk, and its kill-ladder is
fully decidable from raw logs by a 0-context verdict reviewer with no interpretive freedom. The single manipulated
variable (`--num_frames 8→16`) is genuinely single: it needs no code edit, no `cutoff_len` change, and no second
knob, because the frozen extractor has no truncation wall (that wall lives only in the SFT builder, correctly
deferred to the conditional stage-2). The DEV-1 substitution of a **sign-based** KS bar for the recon's
bootstrap-CI bar is **correct and favorable** — it removes a criterion that directly violates the house n=3
no-bootstrap discipline. The three notes below are a diff-enumeration undercount (Note 1), a strong-but-conservative
pre-declared spend inference (Note 2), and a loose "nested" descriptor (Note 3); none affects decidability,
leakage, or the honesty of any bar, and none can be used to manufacture an unsupported pass. **Cleared to freeze +
single-submit.**

---

## Rationale (one paragraph)

The cell measures the one axis 8-frame sampling never varied — visual sampling density (16 vs the hard-coded 8) —
through the **frozen** Qwen2.5-VL-7B encoder + mean-pool on HateMM, paired within head-seed against the banked
frozen-Qwen-**8f** floor (job 12850), dual-protocol. The design's validity hinges on one property: that the 16f
arm is byte-identical to the banked 8f arm **except** the feature cache (16f vs 8f), so the within-seed paired
delta isolates sampling density exactly. That property holds under audit: the head `run_one`…`PY` block is
byte-identical to `enc3seed.sbatch` (block sha256 `286a9e44…`, the exact runner that produced the 8f floor); the
only manipulated head variables vs the 8f control are `--model` (`…_HF`→`…_HF-16f`) and `--group_name`; the
extractor Python is unchanged (`--num_frames`/`--out_model_tag` are pre-existing argparse args, so 16f is a
one-arg wrapper with **no code edit**); the pooled operator (`_encode`, prefix/response mean of last-layer hidden
states, L2-normed) is byte-for-byte the same at any even frame count and the masked-scatter invariant
(`assert last_hidden.shape[0]==input_ids.numel()`, L283) holds at 16; the distinct out-tag is **hardcoded** in the
sbatch (L37) so the banked 8f `…_HF.pt` cannot be clobbered even if `NUM_FRAMES` is set. All comparison floors
re-derive to 4dp from the raw 12850 trainlogs with an independently written parser, the loader routes
`…_HF-16f` to the 16f cache (`dataset.py:499-503` → `load_feats_MHC` → `{split}_{model}.pt`), and every collision
path is verified absent on disk. Because the frozen forward is deterministic (no RNG frame sampling, no dropout,
bf16+sdpa, `no_grad`), there is genuinely no single-encoder-draw confound (F0.2 is *stronger* than the LoRA
cells'), the executor transcribes raw per-seed numbers with the verdict rendered independently, and the FORMAL
bar is unchanged from the encoder-swap criterion — so the motivated-executor attack surface (re-draw, protocol/
metric shop, bury a regression, clobber the floor) is closed by construction. Novelty is **D7-DEAD** (density is
an engineering knob), stated plainly and repeatedly; even a formal PASS is pre-declared as a performance/ablation
row, never a novelty win, and KS-16f-dead is pre-declared as the honest most-likely outcome.

---

## CHECK-BY-CHECK

### 1. Floor re-derivation — **PASS (independently re-parsed; all match to 4dp)**

I wrote a standalone parser implementing the `enc3seed.sbatch` embedded rule (val-sel = epoch ≥ warmup 5 with max
`Val_Retrieval` acc, roc tie-break → that epoch's `Test_Retrieval` acc/macroF1; final = max epoch) and ran it on
the raw `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_12850.trainlog`. **Every per-seed value,
every selected epoch, and both 3-seed means reproduce the prereg §2.1 exactly:**

| protocol | s0 acc/mF1 | s1 acc/mF1 | s2 acc/mF1 | mean (mine) | prereg |
|---|---|---|---|---|---|
| val-sel (sel ep 28/22/29) | 0.8698/0.8606 | 0.8651/0.8586 | 0.8837/0.8753 | 0.8729/0.8648 | ✓ |
| final-ep (ep 29) | 0.8605/0.8507 | 0.8605/0.8514 | 0.8837/0.8753 | 0.8682/0.8591 | ✓ |

Selected epochs 28/22/29 match; seed2's val-sel selects ep29 (= final), so its two protocol rows coincide, exactly
as the prereg states. Final-epoch acc mean = 0.868233 → **0.8682** (bit-matches the recon §5 / F53 KS-2 line). The
§2.1 line-number provenance (`:293/:303`, `:235/:299`, `:302`) points to the `Test_Retrieval … macroF1` line
(the acc line +2, which carries both acc and macroF1); all five offsets check out. The §2.2 CLIP context row
(0.8202/0.8085 · 0.8124/0.7936) is the ERRATUM-corrected floor and is correctly flagged as **orientation only,
not a paired anchor** — this cell pairs vs frozen-Qwen-8f, never vs CLIP.

### 2. Extractor mechanics — **PASS**

- `--num_frames` (argparse L90-95, default 8) and `--out_model_tag` (L84-89, default `Qwen2.5-VL-7B-Instruct_HF`)
  both exist as pre-existing args ⇒ 16f needs **no code edit**. `_sample_frame_indices` (L146-152) =
  `np.linspace(0, N-1, num_frames)` → round → clip; handles 16 (and any count) with no branch. The masked-scatter
  invariant `assert last_hidden.shape[0] == input_ids.numel()` (L283) is a length check between hidden states and
  input_ids that grow together with frame count ⇒ **holds at 16** (and any even count). The pooled operator
  (`_encode` L254-323: prefix-span mean for img, trailing-span mean for text, `float()`+L2-normalize) is
  byte-identical regardless of frame count. Model load is `bf16` + `attn_implementation="sdpa"` under
  `@torch.no_grad()` (L396-402) — confirming F0.2's deterministic-forward claim (no RNG in sampling, no dropout).
- **Out-path routing (L437):** `out_path = {EXP_FOLDER}/{dataset}/{outname}_{out_model_tag}.pt`, `EXP_FOLDER`
  default `./data/CLIP_Embedding`, `SPLIT_TO_OUTNAME = {train:train, val:dev_seen, test:test_seen}`. With
  `--out_model_tag Qwen2.5-VL-7B-Instruct_HF-16f` the outputs are
  `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF-16f.pt` — **distinct** from the
  banked 8f `…_HF.pt`.
- **CLOBBER CATCH — the danger is real, and defused.** The fork source `gen_embed_mllm.sbatch` plumbs
  `--num_frames "${NUM_FRAMES:-8}"` (L32) but passes **NO** `--out_model_tag`, so
  `NUM_FRAMES=16 sbatch gen_embed_mllm.sbatch HateMM` would run at 16f **and write the default `…_HF.pt` tag,
  overwriting the banked 8f floor.** The new `gen_embed_mllm_16f.sbatch` **hardcodes both** `--num_frames 16`
  (L36) and `--out_model_tag Qwen2.5-VL-7B-Instruct_HF-16f` (L37) — no env override, so the distinct out-tag is a
  hash-frozen invariant. `bash -n` on both new sbatch = **SYNTAX_OK**. DEV-2 is thereby a genuine safety
  improvement, not cosmetic.

### 3. Head same-code + loader routing — **PASS**

- `run_one`…`PY` block of `enc3seed_fb16.sbatch` (L37-78) vs `enc3seed.sbatch` (L44-85): **BYTE-IDENTICAL**
  (`diff` empty; extracted-block sha256 `286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101` for
  both — the same block hash the cand-2 review recorded for the banked controls). The load-bearing
  `python ./src/run_rac.py …` argv, the config pins (`--batch_size 64 --lr 0.0001 --epochs 30 --topk 20
  --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align --hard_negatives_loss True
  --no_hard_negatives 1 --metric cos --loss triplet --hybrid_loss True --warmup 5 --lambda_seg 0 --force False`),
  and the embedded val-sel/final parser are identical. The **only** manipulated head variables vs the 8f control
  are `--model` (`…_HF-16f` via `QWEN16`) and `--group_name` (`RAC_video_fb16`) plus derived `--exp_comment`.
  Full-file diff vs `enc3seed.sbatch` = header comment, job-name, the var block (`CLIP/QWEN`→`QWEN16`), group
  name, the CONFIGS array (10 rows → 3 HateMM-16f seed rows), and the final b2_push path — nothing load-bearing.
- **Loader:** `dataset.py:499-503` routes `HateMM` → `load_feats_MHC`; `:605-608` reads
  `{path}/{dataset}/{split}_{model}.pt`. So head `--model …_HF-16f` reads exactly the extractor's `…_HF-16f`
  output, with identical `N`/id-order to the 8f cache (both derived deterministically from the same gt files) ⇒
  the within-seed pairing isolates the feature cache.

### 4. Bars + kill ladder — **PASS (DEV-1 ruled favorable below)**

- **§3.1 decision rule** is quoted faithfully from `exp-encoder-3seed.md:73-85` (verified line-by-line: per-seed
  paired δ, 3-seed mean±std + sign, n=3 paired-t as effect-size descriptor only / no significance claim, pass =
  Δacc≥+0.030 AND ΔmF1≥+0.030 AND 3/3 sign, headline needs ≥2 datasets, both protocols judged separately).
- **Ladder internal consistency.** Four mutually-exclusive, jointly-exhaustive outcome bins on one measurement:
  - **KS-16f-dead (§3.2):** neither protocol yields (mean Δacc > 0 AND acc sign 3/3) ⇒ KILL, LoRA-16f auto-dead.
  - **weak-limbo:** some protocol clean-positive but < +0.010 / not-3/3-at-threshold.
  - **CONTINUE (§3.3):** ∃ protocol with mean Δacc ≥ +0.010 AND acc sign 3/3 ⇒ LoRA-16f funded.
  - **FORMAL (§3.4):** both protocols Δacc ≥ +0.030 AND ΔmF1 ≥ +0.030 AND 3/3 ⇒ engineering/ablation row.
  FORMAL ⟹ CONTINUE (both-protocol +0.030/3-3 implies ≥1-protocol +0.010/3-3), so the nesting is coherent; each
  bar is a fixed pre-registered threshold on raw per-seed numbers with no interpretive freedom. Both protocols
  are judged **independently** (fixed §7.2 write-up) ⇒ no protocol/metric shopping. The FORMAL bar is byte-for-byte
  the encoder-swap criterion, so a single-dataset PASS is correctly framed as an engineering row (rule 5's
  "≥2 datasets" headline is unreachable by a HateMM-only cell — the prereg does not over-claim).
- **DEV-1 RULING — FAVORABLE (I concur with the substitution).** The recon §5 (L166) phrases KS-16f-dead as
  "mean paired Δacc ≤ 0 OR **bootstrap CI straddles 0** on both protocols." A bootstrap-CI kill decision on n=3
  is a **significance claim from n=3**, which `exp-encoder-3seed.md:78-79` (rule 3) explicitly forbids
  ("n=3 is too small for a formal bootstrap … no significance claim is made from n=3"). The prereg replaces it
  (§3.2) with a **sign-based** bar ("mean Δacc ≤ 0 OR acc sign not 3/3 positive"), reusing the house's own teeth
  (rule 4's 3/3-sign requirement). This changes only the significance *formalism*, keeps the qualitative bar
  (tie/regress on both protocols ⇒ dead) identical, and can only ever fire on the **negative** side (kill/defund) —
  it cannot manufacture a PASS, and the FORMAL/paper bar is untouched. Pinning the sign bar over the recon's
  CI bar is the correct call.

### 5. Leakage / veto / collision / single-test-touch — **PASS**

- **Label-free extraction.** `IMG_INSTRUCTION`/`TEXT_INSTRUCTION` (L45-52) are fixed and applied identically to
  every video; `process_split` (L344-368) feeds only the video + its own title/transcript into `_encode` and
  writes the label solely into the saved `labels` tensor — **gold never enters the encoder path.** No SFT in
  stage-1 ⇒ F0.4 veto is trivially clean. **No OCR channel. No external API.**
- **Own-train-split / no cross-mixing.** The RGCL head trains on HateMM's own train split only (identical to the
  8f floor); raw videos never leave (b2_push copies only the `.pt` caches under `data/CLIP_Embedding/HateMM`).
- **Clobber-proof out-path** (see Check 2): distinct hardcoded tag ⇒ banked 8f `…_HF.pt` cannot be overwritten.
- **Collision list verified ABSENT on disk (re-check at submit):**
  `data/CLIP_Embedding/HateMM/*_Qwen2.5-VL-7B-Instruct_HF-16f.pt` (none),
  `logging/Retrieval/HateMM/RAC_video_fb16*` (none), `slurm/logs/enc3s_*_HF-16f_seed*.trainlog` (none),
  `logging/_smoke_fb16` / `*_smoke*` groups (none). The banked 8f caches (`train`/`dev_seen`/`test_seen`_…_HF.pt,
  dated Jul 2) are present and untouched. `--force False` hits the protective abort at `run_rac.py:905-908` only
  if the fresh `RAC_video_fb16` output path already exists (it does not) — so it never trips, and if it ever did
  it would abort rather than overwrite.
- **Single-test-touch accounting.** The 3 head-seed reads are the ONLY budgeted frozen-16f-encoder test
  evaluations = exactly ONE new single-test-touch under the F0.1 house convention; **zero test-touch before the
  independent verdict** (the author ran nothing; CPU-only). Prior HateMM-test exposures under the identical enc3s
  protocol are correctly listed: frozen-CLIP + frozen-Qwen-8f (job 12850, banked trainlogs present), generic-LoRA
  (job 13235 / F53, trainlogs present on disk), the LoRA-HateMM verdict, and cand-2 curriculum (frozen+submitted).
  This cell's reads are re-measurements under that protocol, not first exposures.

### 6. Cost + submit plan — **PASS**

- **Sequential, chained.** Extraction (`gen_embed_mllm_16f.sbatch HateMM`) → head
  (`--dependency=afterok:<1> enc3seed_fb16.sbatch`); the head cannot start until extraction succeeds ⇒ the two
  jobs **never run concurrently.**
- **Resource aggregates within cap at all times.** Both sbatch headers request `--cpus-per-task=8`, `--mem=64G`,
  `--gres=gpu:a100:1` (verified L3-5 of each). With the afterok chain, peak footprint = **8 CPU / 64 G / 1 GPU** —
  well within the 16 CPU / 128 G / 2 GPU cap, and never two 16-CPU jobs in flight (the standing infra rule after
  the 13303 wedge). **NO `--time`** in either sbatch ("intentionally NO --time"). `conda activate HateVideo`;
  `PENDING (JobHeldUser)` → **wait for auto-release, never force** stated (§6). Both source `conda.sh` directly
  and run `disk_guard.sh`.
- **Cost ledger:** stage-1 ~0.5–0.7 A100-h (extraction ~0.4–0.6 + head ~0.03), faithfully transcribed from the
  recon (HateMM 949.1 s 8f anchor × 1.5–2×). ZH stage-1.5 and LoRA-16f stage-2 (≥2 changed variables incl.
  `cutoff_len 4096→~8192`) are declared CONDITIONAL FUTURE and are **not** authored or submitted here; 32f is
  declared OUT on multiplicity.

### 7. Deviations §11 (DEV-1..DEV-5) — all favorable / neutral / documented

- **DEV-1** (sign-based KS bar vs recon's bootstrap-CI) — **FAVORABLE**; ruled in Check 4. Pins the house n=3
  no-bootstrap discipline; only the significance formalism changes; cannot manufacture a pass.
- **DEV-2** (dedicated `gen_embed_mllm_16f.sbatch` vs reusing `gen_embed_mllm.sbatch` with `NUM_FRAMES=16`) —
  **FAVORABLE / safety-critical**; the fork source cannot set `--out_model_tag`, so reuse would clobber the 8f
  floor. Verified in Check 2. No extractor code edit.
- **DEV-3** (no code edit for stage-1) — **neutral, verified**; both args pre-exist, `np.linspace` + the L283
  assert work at 16, no cutoff/VRAM wall in the extractor.
- **DEV-4** (tags `RAC_video_fb16` / `enc3seed_fb16` / `mllm_embed_16f`) — **neutral**; collision-checked absent.
- **DEV-5** (ZH stage-1.5 + LoRA-16f stage-2 + 32f = CONDITIONAL FUTURE / OUT) — **documented**; no such artifact
  is authored or submitted here; the recon's "stage-2 NO-GO unless stage-1 moves" framing is preserved via §3.3.

---

## NON-BLOCKING NOTES (for the executor / verdict reviewer)

1. **§4.1a/§4.2 diff-enumeration undercounts by two cosmetic hunks.** The prereg says
   `gen_embed_mllm_16f.sbatch` differs from `gen_embed_mllm.sbatch` "ONLY in the header comment, job-name, the
   echo line, and the two python-arg lines." The actual `diff` has **six** hunks: job-name, header comment, the
   `[mllm_embed…]` echo line, the two python-arg lines (`--num_frames 16` + `--out_model_tag`), **plus** the
   pre-`b2_push` comment (`# … push MLLM-16f embeddings …`) and the trailing `[mllm_embed_16f] done` echo. The two
   un-enumerated hunks are a comment and a log-tag echo — **zero load-bearing**. Descriptor imprecision only
   (parallels the vision review's Note 2 line-count slip); the manipulated python args are exactly the two
   claimed. Non-material.

2. **§3.2 KS-16f-dead → LoRA-16f "auto-dead" is a strong pre-declared inference, but conservative.** A *frozen*
   forward tying/regressing at 16f does not strictly *prove* a LoRA-adapted forward cannot extract more from 16
   frames (LoRA changes the encoder). The prereg's argument (F0.5 dilution/redundancy + "cannot re-SFT into
   information the frozen forward proves the pool discards") is a reasonable mechanism prior, and the gate only
   ever **defunds** the expensive contaminated stage-2 (conservative direction) — it never manufactures a PASS.
   Pre-declared as a spend decision, not a scientific claim; acceptable as a pre-registered risk budget. Noted so
   the verdict reviewer records it explicitly as a *spend* verdict, not a proof that 16f is scientifically inert.

3. **§3.5 "three nested bars" is an ordering, not literal set-nesting.** KS-dead / weak-limbo / CONTINUE / FORMAL
   are four mutually-exclusive outcome bins ordered by increasing positive strength; the `⊂` chain reads as a
   threshold ordering. FORMAL ⟹ CONTINUE holds (Check 4), so the ordering is sound; the "nested" wording is
   merely loose. Cosmetic.

---

## HASH-FREEZE

Recorded in `refine-logs/FRAME16_FREEZE.md` (prereg NOT modified, per review mandate). All freeze-block shas in
§5 re-verified on disk at freeze time and **match**: prereg self-sha `5c240518…`, B `a600e74c…`, C `99e7e8b1…`,
extractor `d89a9126…`, fork source `9357fa10…`, head anchor `dbe3fb81…`; banked 8f caches present and untouched.

**Reviewer statements:** ZERO GPU/SLURM/Modal spent — CPU-only login-node re-parse of the banked 12850 trainlogs
with an independently written parser, plus hashing / `bash -n` / `ls` collision checks (seconds); no held-out test
metric produced; `state/` not touched; the prereg was **NOT** modified; no job submitted; not pushed.
