# MOKA — Modality-Routed LoRA (MokA `A`-split) Pre-Registration — ZH single arm, one bite

**Author:** MokA implementation + prereg author (CPU-only; **ZERO GPU / SLURM / Modal spent; NO job
submitted; no test-touch; no `state/` mutation; no push**).
**Date:** 2026-07-26 NZST.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced.
**Implements:** `refine-logs/MOKA_FORENSIC_RECON.md` (commit `dbf30f1`, the GO recon) — its locked
decisions transcribed and re-verified below: `A`-split only / cross-attention OUT (§3.2), `r_v = r_t = 16`
(§3.3), modality mask `(input_ids==151655)|(input_ids==151656)` (§2.4), monkey-patch
`llamafactory.model.adapter.get_peft_model` with **zero vendored-tree edits** (§2.3), the
`merge_and_unload()` blocker and its `--moka` unmerged-extraction fix (§2.5/§3.1), ZH-first with the
HateMM leg auto-defunded (§4.2), curriculum compatibility (§3.4), the honest prior 5–8 % and the
**text-side** bet (§5.1), and the KS skeleton (§6).
**MokA licence:** **USER RULING 2026-07-26 — UNGATED.** The code may be used directly; MokA is
credited in the paper (`state/progress.json`, commit `6c4766e`). Every ported line carries a credit
header naming `GeWu-Lab/MokA` (NeurIPS 2025) @ `b28e834`.
**House-style precedent:** `refine-logs/ZHPROMPT_PREREG.md` (structure, F0.x honesty clauses,
default==identity parity guard, single-submit plan, freeze block), `refine-logs/NCA_PREREG.md`
(§4.5 codex gate, code-fix⇒re-freeze clause), `refine-logs/FRAME16_PREREG.md` (extraction+head
pipeline, outcome-table template), `research-wiki/experiments/exp-encoder-3seed.md:73-85` (the
`enc3s` protocol + decision rule verbatim).

## Title + claim scope (verbatim)

> This measurement tests **one axis no banked adapter in this campaign ever varied — whether the LoRA
> down-projection is SHARED across modalities or ROUTED per modality.** Every banked adapter (B3,
> F53, curric, vis, bidir) used **one shared `A` for vision-pad and text positions alike**;
> modality-routed adaptation has **never been run on this project**. The cell ports MokA's `A`-split
> (`lora_A_v` for `<|image_pad|>`/`<|video_pad|>` positions, `lora_A` for every other position, a
> **SHARED** `lora_B`, `r_v = r_t = 16`, `alpha 32`, dropout 0.0, the same 7 projections × 28 decoder
> layers, **NO cross-attention**) into the deployed ZH LoRA-SFT recipe **changing nothing else**, then
> re-extracts the ZH dual-stream through an **unmerged** adapter forward and retrains the deployed
> RGCL align-fusion head + top-20 kNN, **3 head-seeds paired within seed** against the banked
> generic-LoRA floor **13150**, **dual protocol** (val-selected AND final-epoch), on **`MHC_zh`
> trained ONLY on its own train split**. **ONE arm, ONE bite.** The cell does **NOT** bet MokA's
> advertised premise. Recon §0/§5.1, transcribed **verbatim** because it is load-bearing:
>
> > "The GO rests on **one** premise that is *not* priced dead: MokA's arithmetic side-effect gives
> > the **dominant TEXT stream its own undiluted down-projection**, and text is the stream that
> > carries **both** measured passes (F45 ZH, F58 HateMM). MokA's *advertised* premise (protect the
> > weak visual modality) is **already priced dead** by F58 + F65 and is explicitly **not** what this
> > cell bets on. That distinction is load-bearing and must survive into the prereg verbatim."
>
> **EN is NOT re-opened** (§3.6). **HateMM is a stage-2 hold-the-pass leg, auto-defunded by
> `KS-MOKA-1`** and out of scope for this submit. **Novelty is D7-BOUNDED (F0.3):** a PASS is a
> performance row plus a mechanism sentence; transplant novelty is claimable **only** as
> *first application of modality-routed PEFT to hateful-video encoders*, and **only** with MokA
> credited — never as an invented mechanism.

The cells under test are the deployed RGCL `classifier_hateClipper`, `fusion_mode=align`
(Hadamard `x=img⊙text`), triplet+0.5·BCE, AdamW head over cached embeddings, 30 epochs, warmup 5,
top-20 arithmetic signed-cosine kNN vote, `--force False`, paired **3-seed within head-seed** vs the
banked floor 13150, dual-protocol. **The ONLY manipulated variable between the arm and its floor is
`A`-sharing → `A`-routing inside the SFT.** Same base model, same data (`mhc_zh_lora_train`), same
epochs / lr / schedule / batch / rank / alpha / dropout / targets / freezes, same extractor, same head
command. **Any of: porting the cross-attention, changing `r_v ≠ r_t`, adding a second dataset, an EN
arm, a different mask definition, or a routing-`B` variant is OUT of this prereg** — each re-costs a bite (§3.8).

---

## 0. Binding facts / honesty clauses (all pre-declared)

**F0.1 — Test is NOT virgin (declared).** ZH test was already read, under the identical `enc3s`
protocol, by: frozen-CLIP (13115 CLIP arm), frozen-Qwen-English (13115), **generic-LoRA (job 13150 —
this prereg's floor)**, curriculum-LoRA (13241), bidir-LoRA (13471), readout-grid (13468), head-recipe
(13478), the NCA family (13482) and the ZHPROMPT family. This prereg's reads are **re-measurements
under the identical protocol**, not first exposures. **Budgeted test evaluations = 3** (1 arm × 3
head-seeds). `KS-MOKA-0b` costs **0** (feature cosine only; the caches carry no labels through the
comparison). **Contingent +3** only if `KS-MOKA-0b` forces the same-path unmerged floor (§3.4).
**Zero test-touch before the independent verdict.**

**F0.2 — SINGLE-ENCODER-DRAW declaration (mandatory; B3/F53 precedent, recon §6).** **ONE** MokA SFT
run. `--seed` varies the **head** only (head-init + data-shuffle); pairing is per head-seed
(arm seed *s* − floor seed *s*, *s*∈{0,1,2}). The encoder is a **single draw** and **its seed variance
is NOT estimated** — declared exactly as F53 did. Unlike ZHPROMPT (where treatment and floor shared
one adapter), here the arm's encoder is a **different SFT draw** from the floor's, so **encoder-seed
noise is confounded with the routing effect and is NOT separable within this budget.** This is a
material limitation and must be restated at verdict time; it is the same limitation every banked
encoder cell in this campaign carries (B3, F53, curric, bidir), accepted for the same reason (a
3-seed SFT sweep costs ~9 GPU-h and the family budget is 4.6).

**F0.3 — Novelty = D7-BOUNDED, and the boundary is stated plainly (USER RULING).** MokA is a
published PEFT variant (NeurIPS 2025). Recon §5.1: *"Novelty class: D7-relevant but weak — a published
PEFT variant transplanted onto our encoder. A PASS is a performance row plus a mechanism sentence; it
is **not** a novelty mechanism."* Per the user ruling, transplant novelty is claimable **ONLY** as
(a) *first application* of modality-routed PEFT to hateful-video encoders, **and** (b) with **MokA
explicitly credited** (paper acknowledgement + a citation; code credit headers already in
`src/moka/routed_lora.py`). Any phrasing that implies we invented modality-routed LoRA is **banned**.

**F0.4 — Standing-veto compliance (recon §4.3, re-verified).** Single-dataset own-train-split only
(`mhc_zh_lora_train`, **no** cross-dataset mixing) ✅; **NO OCR channel** ✅; no gold spans/attributes ✅;
no cross-seed ensembles ✅; no MLLM-scores-as-training-signal ✅; raw videos never leave the machine
(SFT reads local frame JPGs; only derived `.pt` → B2) ✅; all GPU via SLURM ✅.

**F0.5 — Honest prior is LOW and is the recon's, transcribed unchanged.**
**P(goal: ≥ +0.030 acc AND +0.030 mF1, 3/3 sign, BOTH protocols, on ≥ 1 dataset) = 5–8 %.**
**P(any KS-surviving movement worth reporting) = 25–30 %.**
**Against (all measured, all binding on MokA's *advertised* premise; recon §5.1 verbatim):**
- **F58** — HateMM's pass is text-carried and frozen-sufficient; image stream LoRA−frozen =
  **+0.0045/+0.0062 (FLAT)**. **F45** — the ZH gain lives **entirely** in the text stream; image stream
  flat (0.718 → 0.721 → 0.714). ⇒ on **both** converting datasets the image stream contributes ≈ 0.
- **F65** — the image stream is **movable** (+0.0320 trLOO on EN) and moving it **converts nothing**
  (K-V2 TIE everywhere). **8th law-I instance.** MokA's stated goal — protect the dominated visual
  modality — is priced at ~0.
- `LITSWEEP2_FRESH_2026` HUNT-3 priced "richer PEFT adapters (MoLE, Task-Adapter++)" at ~0.
- ZH's binding val-sel leg must clear +0.030 through the **78-item dev** selection wall (F45/F63),
  head-seed noise band ±0.014.
**For (the one premise not priced dead — what the cell actually bets on; recon §5.1 verbatim):**
- MokA's arithmetic side-effect is that the **dominant TEXT stream gets its own `A_t`, undiluted by
  image-token gradient**. **No finding prices a sharpened text-side adaptation subspace.** F45 shows
  ZH text train-LOO AUC moving 0.847 → 0.925 under a *shared* `A`; whether an undiluted `A_t` moves it
  further is **unmeasured**.
- Encoder-SFT is the **only** axis in the campaign that ever formally converted (F53, F45/B3).
- Every banked adapter used **one shared `A`**. Modality-routed adaptation has **never been run here**.
- Per-token effective rank identical ⇒ the cleanest single-variable comparability available (§F0.7).

**F0.6 — NEW MEASURED COUNTER-PRESSURE (this prereg, CPU; the recon did NOT have it). The token
balance in our SFT is the MIRROR IMAGE of MokA's, and it argues for the LOW end of 5–8 %.**
Measured this prereg on the real deployed records (`scripts/analysis/moka_smoke.py` S8, plus a
tokenizer sweep over all 579 `data/lora_sft/MHC_zh/train.json` rows):
- At the deployed `image_max_pixels: 262144`, one ZH SFT record tokenizes to **2,688 vision-pad tokens
  (8 frames × 336) + a median 153 text tokens ⇒ median vision share 94.6 %** (measured record #0:
  2,688 + 135 = 2,823 tokens, 95.2 % vision; text tokens over 579 rows: min 81 / p25 112 / median 153 /
  p75 210 / max 393).
- MokA's own shipped setting is the OPPOSITE: their in-code debug trace
  (`external/baselines/MokA/VisualText/modified_peft/tuners/lora/layer.py:565-566`) records
  `my_text_mask 16128` vs `my_image_mask 256` on a 16,384-token sequence = **98.4 % TEXT**.
- **Consequence, stated honestly:** the routing makes `A_t` *undiluted* (the bet) but also
  *data-starved* — under a shared `A` the down-projection received gradient from **100 %** of
  positions; routed, `A_t` receives it from **~5.4 %**. Over the same 3 epochs that is ~18× fewer
  token-gradients into the matrix the bet depends on. Whether "undiluted but starved" beats "diluted
  but data-rich" is precisely the unmeasured question this cell buys — but it is a reason to sit at
  **5 %**, not 8 %. It also means the transfer from MokA's reported gains is **weak**: their
  text-dominant regime is not ours. **This paragraph must be restated at verdict time whatever the
  outcome.**

**F0.7 — Comparability: +44.89 % parameters, per-token effective rank UNCHANGED (recon §3.3; the
FLOPs half is AMENDED, §11 DEV-1).** Verified arithmetically this prereg (smoke S7, exact integers):
deployed `A+B` r=16 = **40,370,176** params (byte-matches `logging/lora/HateMM/adapter_model.safetensors`
and `logging/slurm/lora_sft_13233.out:308`); MokA `A_t+A_v+shared B` r=16 = **58,490,880** =
**1.448864×**. **At any given token exactly one `A` output is used, so the applied ΔW is rank-16 —
identical capacity per token to the deployed adapter; the extra parameters are inactive per token and
the manipulated variable is *which* `A` a token is routed through.** FUSIONCAT accepted 2.0× as
"within the ~2× comparability guidance" for a change that altered per-token capacity; 1.4489× at
unchanged per-token rank is strictly inside that precedent. **AMENDMENT (honest):** the frozen
implementation is the *dense-select* formulation, which **evaluates both `A`s and discards one**, so
adapter `A` FLOPs are ~2× (≈ +1 % of total layer FLOPs, since `A` is rank-16 against 3584/18944-wide
base projections). The recon's "FLOPs are identical" sentence is therefore replaced by
**"per-token effective rank is identical; compute is ≈ +1 %"** — see §11 DEV-1 for why.

**F0.8 — Default-flag == identity == the parity guard (CPU-verified this prereg).** The extractor's
two new flags `--no_merge` / `--moka` both DEFAULT to `False`, and with both off the code path is the
**byte-identical deployed `merge_and_unload()` path**. CPU-verified this prereg: `py_compile` PASS;
`parse_args_sys([])` gives `moka=False, no_merge=False`; the ZHPROMPT default==identity assembly proof
(`text_prompt` byte-matches the deployed literal, all 5 prompt args at English defaults) **still
holds unchanged** after this edit. `KS-parity` (§3.2) upgrades this to a runtime bit-exact check.

**F0.9 — `run_rac.py` and the whole head path are UNTOUCHED (pre-declared, verified).**
`git status --porcelain src/run_rac.py src/model/loss.py src/model/classifier.py src/utils/retrieval.py`
= **CLEAN**; `sha256(src/run_rac.py) = b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3`
— **identical to the sha frozen in `ZHPROMPT_PREREG.md §5.2`**. The `run_one()` block of
`scripts/slurm/moka_extract_head.sbatch` is **BYTE-IDENTICAL** to `scripts/slurm/enc3seed_zh_b3.sbatch`
lines 42-83 (the runner that produced floor 13150) — verified by exact line-list comparison this
prereg. The only per-run variables are `--model` and the derived `--exp_comment "_${MODEL}"`.

**F0.10 — ZERO vendored-tree edits (verified).** `RA-HMD/LLAMA-FACTORY-Ver202512` is a git **gitlink**
(`git ls-files -s` mode `160000`); **not one line of it is modified**. The routing enters through
`llamafactory.model.adapter.get_peft_model` — a module attribute — replaced from our wrapper entry
point. **Runtime-verified this prereg (CPU import, no GPU):** after importing `src/moka/train_moka.py`,
`llamafactory.model.adapter.get_peft_model is train_moka._moka_get_peft_model` → **True**, and
`inspect.getsource(adapter._setup_lora_tuning)` contains the literal `get_peft_model(model, peft_config)`
→ the patch is on the hot path (adapter.py:312). The new SFT **yaml** lives inside that gitlink
(`my_configs/hatevideo/`), exactly like every deployed config; it is therefore **hash-frozen, not
committed** — the `CAND2_FREEZE.md` / `LORA_HATEMM_FREEZE.md` precedent (§5.2).

**F0.11 — Curriculum compatibility is out of scope for stage 1 but pre-recorded (recon §3.4).** The
curriculum is a deterministic multiset re-weighting emitting byte-identical records capped to
`N_train` (`build_curriculum_sft_data.py:172-174,:209`), **not** an ordering ⇒ MokA composes with it
by construction. The stage-2 HateMM cell would reduce to "same `train_curric.json`, `--moka on`". It
is **auto-defunded** unless `KS-MOKA-1` is survived (§3.5) and would need a prereg amendment + a new
yaml before any submission.

---

## 1. Pipeline spec — fully pinned (2 SLURM jobs, submitted SEQUENTIALLY)

### 1.0 Job-chain shape (CHOSEN + documented)

**TWO jobs, submitted sequentially, NOT chained by `--dependency`.** Job-1 (SFT) needs the deployed
16 CPU / 120 G footprint; job-2 (extract+heads) needs 8 CPU / 64 G. They are submitted **one at a
time** — job-2 goes in only after `sacct` reports job-1 `COMPLETED`. Rationale: the standing infra
rule ("**never two concurrent 16-CPU jobs**"; submit-time aggregate cap, the 29 h wedge) is satisfied
*by construction* if only one job of this family is ever in the queue, whereas `--dependency=afterok`
would put 16+8 = 24 CPU of submit-time aggregate demand in front of a 16-CPU user cap. **Peak
footprint at any instant = 16 CPU / 120 G / 1 GPU.**

### 1.1 Job 1 — `sbatch scripts/slurm/lora_sft_moka.sbatch` (default `MHC_zh`)

A clone of the deployed `scripts/slurm/lora_sft.sbatch` (which produced floor-adapter job 12143).
Differences, exhaustively:
1. entry point `python src/train.py <yaml>` → `python /data/jehc223/RGCL/src/moka/train_moka.py <yaml>`
   (same cwd `$LF_ROOT`, same `run_exp()`);
2. the yaml (below);
3. disk preflight ≥ **25 G** instead of ≥ 20 G (MokA adapter set ≈ 1.0 G on disk: 4 saves × ~234 MB);
4. a collision guard (`ABORT` if `logging/lora/MHC_zh_moka/adapter_model.safetensors` exists);
5. `MOKA_ROUTE_IMPL=dense`, `MOKA_STRICT=1` exported;
6. a **$0 post-run block**: asserts the saved adapter has **196 `lora_A` + 196 `lora_A_v` + 196
   `lora_B`** tensors and **58,490,880** params, then emits the **`KS-MOKA-2`** readout (per-layer
   `‖A_v − A_t‖_F / ‖A_t‖_F`, min/median/max) to `refine-logs/MOKA_KS2_routing_report.json`.
Everything else (`DISABLE_VERSION_CHECK`, nvcc shim, `HF_HUB_OFFLINE`, `disk_guard.sh`, the
`build_lora_sft_data.py --dataset MHC_zh` idempotent pre-step) is verbatim.

**YAML** `RA-HMD/…/my_configs/hatevideo/mhc_zh_qwen25vl_lora_moka_sft.yaml` — a copy of the deployed
`mhc_zh_qwen25vl_lora_sft.yaml`; `diff` = **exactly one line** (`output_dir` →
`/data/jehc223/RGCL/logging/lora/MHC_zh_moka`), verified this prereg. So the recipe is pinned at
`lora_rank 16`, `lora_alpha 32`, targets `q,k,v,o,gate,up,down`, `freeze_vision_tower/projector: true`,
`lr 1.0e-4`, `3.0` epochs, bs 1 × grad-accum 8, cosine, warmup 0.05, bf16, gradient checkpointing,
`cutoff_len 4096`, `save_strategy epoch`, `lora_dropout` unset ⇒ **0.0**.

**Output:** `logging/lora/MHC_zh_moka/` (adapter_config.json + adapter_model.safetensors with the
`lora_A_v` keys + 3 epoch checkpoints). **Cost ≈ 3.1 GPU-h** (floor `train_runtime` 8,635.998 s =
2.399 h, job 12143 wall `02:39:49`; × ~1.2 routing/eval overhead + ~0.25 h build/load).

### 1.2 Job 2 — `sbatch scripts/slurm/moka_extract_head.sbatch` (8 CPU / 64 G / 1 A100)

**Stage A0 — `KS-MOKA-0b` merge-drift probe (0 test-touch).** Re-extract the **already-banked generic
ZH adapter** `logging/lora/MHC_zh` through the **UNMERGED** path (`--no_merge`, plain PEFT forward, no
routing) to tag `Qwen2.5-VL-7B-Instruct-LoRA_HF-um`, all 3 splits; then report **mean and min per-item
cosine** against the banked merged cache `…-LoRA_HF.pt`, per split × stream (6 numbers). Rationale
(recon §3.6): the banked floors came from a **merged** encoder (`W+BA` folded, one bf16 matmul);
MokA's must come from an **unmerged** encoder (`Wx + B(A_m x)`, different bf16 accumulation order), so
a routing-OFF MokA **cannot** reproduce the floor cache bit-exactly and the G-repro must be at the
**layer** level, not the pipeline level. Cost ≈ **0.6 GPU-h** (banked `lora_embed` jobs 13234/13239/
13240/13245/13302 ran `00:26:17`–`00:37:25`; unmerged ≈ ×1.1).

**Stage A1 — MokA extraction.** `generate_VideoMLLM_embedding_lora_HF.py --dataset MHC_zh --lora_dir
logging/lora/MHC_zh_moka --out_model_tag Qwen2.5-VL-7B-Instruct-LoRA-moka_HF --moka --num_frames 8
--device cuda`. `--moka` ⇒ **no merge**, install `MokaLinear` + the `input_ids` modality pre-hook,
then **explicitly load the `lora_A_v` tensors** (`PeftModel.from_pretrained` drops them silently under
`strict=False` when the class is absent — recon §3.5 item 4). Cost ≈ **0.7 GPU-h**.

**Stage S — shape sanity (before ANY budgeted test read).** Asserts all 3 MokA caches exist with
`img_feats.shape == text_feats.shape == (N, 3584)`, `len(ids[0]) == N > 0`, and all-finite; a fail
exits non-zero **before** the 3 head runs, protecting the test-touch.

**Stage B — 3 RGCL head + kNN runs.** `{Qwen2.5-VL-7B-Instruct-LoRA-moka_HF} × seed{0,1,2}`,
`--group_name RAC_video_moka`, `--force False`, `run_one()` **BYTE-IDENTICAL** to
`enc3seed_zh_b3.sbatch`. Cost ≈ **0.05 GPU-h** (13150 ran 3 seeds in `00:02:46`). Output
`slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA-moka_HF_seed{0,1,2}_<JID>.trainlog`.
Then B2-push the derived `.pt` + logs (raw videos never leave — CLAUDE.md boundary).

### 1.3 The code (5 new files, 1 in-repo file modified, 0 vendored lines)

| file | status | lines | content |
|---|---|---|---|
| `src/moka/routed_lora.py` | NEW | 372 | `MokaLinear(peft.tuners.lora.layer.Linear)` + `lora_A_v`; routed forward (`dense` frozen / `gather` cross-check); `install_moka()` (in-place `__class__` swap + `register_forward_pre_hook(with_kwargs=True)`); `load_moka_a_v()`; merge guard; `moka_param_report()` / `moka_routing_report()` (`KS-MOKA-2`); credit header naming GeWu-Lab/MokA @ `b28e834` |
| `src/moka/train_moka.py` | NEW | 52 | monkey-patches `llamafactory.model.adapter.get_peft_model` → `get_peft_model` + `install_moka`, then `run_exp()`. **Zero vendored edits.** |
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | **MODIFIED** | **+33 / −2** | `--no_merge` and `--moka` flags, both default OFF ⇒ byte-identical deployed merged path |
| `scripts/analysis/moka_smoke.py` | NEW | 324 | the CPU gate S1–S8 (§4.3) |
| `scripts/slurm/lora_sft_moka.sbatch` | NEW | 113 | job 1 (§1.1) |
| `scripts/slurm/moka_extract_head.sbatch` | NEW | 162 | job 2 (§1.2) |
| `RA-HMD/…/mhc_zh_qwen25vl_lora_moka_sft.yaml` | NEW (in gitlink) | 48 | deployed ZH yaml, `output_dir` changed **only** (1-line `diff`) |

**Total: 1,023 new in-repo lines + 48 lines of new yaml inside the gitlink + 33 added / 2 removed
in one in-repo file; 0 vendored lines touched.**

---

## 2. Comparison floor — INDEPENDENTLY RE-DERIVED from raw trainlogs (numeric-provenance discipline)

Re-parsed **this prereg** from the raw trainlogs with the **exact** `enc3seed_zh_b3.sbatch` embedded
parser (val-sel = epoch ≥ warmup 5, max `Val_Retrieval` acc with roc tie-break; final = max epoch).

### 2.1 Floor — job **13150** (`Qwen2.5-VL-7B-Instruct-LoRA_HF`, generic LoRA / B3)

| seed | val-sel ep | val-sel acc/mF1 | final ep | final acc/mF1 |
|---|---|---|---|---|
| 0 | 20 | 0.8322 / 0.8023 | 29 | 0.8456 / 0.8181 |
| 1 | 26 | 0.8255 / 0.7956 | 29 | 0.8389 / 0.8113 |
| 2 | 19 | 0.8389 / 0.8065 | 29 | 0.8523 / 0.8226 |
| **mean** | | **0.8322 / 0.8015** | | **0.8456 / 0.8173** |

Files: `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`.
Bit-matches recon §4.2, `B3_EXECUTION_RECORD.md`, `NCA_PREREG.md §2.1`, `ZHPROMPT_PREREG.md §2.1`.
**NOT** the ledger's `0.8537` (a different ZH cell — 0.8732-incident discipline).

### 2.2 Stage-2 reference only (NOT in scope) — HateMM job **13241** (`…-LoRA-curric_HF`)

val-sel **0.8775 / 0.8711**; final **0.8791 / 0.8726** (per-seed val-sel 0.8791/0.8730, 0.8744/0.8678,
0.8791/0.8724; final 0.8791/0.8730, 0.8791/0.8724, 0.8791/0.8724). Re-derived this prereg; bit-matches
recon §4.2. Recorded **only** so `KS-MOKA-1`'s auto-defund arithmetic is auditable: +0.030 from 0.8791
would require **0.9091**, which recon §4.2 calls "**arithmetically implausible**".

### 2.3 Concrete promote thresholds + noise band

- **FORMAL (vs 13150):** val-sel mean acc ≥ **0.8622** AND mF1 ≥ **0.8315**; final mean acc ≥ **0.8756**
  AND mF1 ≥ **0.8473**; **3/3 per-seed positive on both metrics, both protocols.**
- **Head-seed noise band:** ±**0.014** — the established house ZH descriptor (`B3_PREREG_REVIEW.md`,
  `CAND2_CURRICULUM_PREREG.md §2.3`, `NCA_PREREG.md §2.3`, `ZHPROMPT_PREREG.md §2.3`).

---

## 3. Decision rule + kill bars (pre-declared, binding, judged by an independent 0-context reviewer)

### 3.1 Decision rule — verbatim from `exp-encoder-3seed.md:73-85`

> For each dataset × protocol: (1) per-seed paired difference δ = (treatment − control) for acc and
> macro-F1 at seeds 0/1/2; (2) 3-seed mean ± std + sign consistency (how many of 3 positive); (3) n=3
> too small for a bootstrap — report the paired-t **as an effect-size descriptor only**, no
> significance claim; (4) **pass = mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND
> sign 3/3 positive**; (5) headline claim requires pass on ≥ 2 datasets under a stated protocol; both
> protocols judged separately; verdict written exactly "final-epoch: pass/fail; val-selected: pass/fail".

Both protocols judged **independently** (no protocol-shopping, no metric-shopping). Rule (5)'s
≥2-dataset headline is **structurally unreachable in stage 1** (single dataset).

### 3.2 `KS-MOKA-0` — pre-GPU machinery gate (MANDATORY, codex-gated). Any failure ⇒ **NO submission.**

All of §4.3's S1–S8 must PASS, in particular:
- **S2 IDENTITY CONTROL (the recon §3.5 item 1 threshold, unrelaxed):** with `lora_A_v` weights copied
  from `lora_A`, `MokaLinear.forward` must equal upstream PEFT `Linear.forward` at **max|Δ| = 0.0**
  in fp32, for **all-text, all-vision AND mixed** masks. **RUN THIS PREREG (CPU): PASS, `0.000e+00`
  in all 6 cells** (2 modules × 3 mask patterns) under the frozen `dense` implementation.
- **S8 MASK COVERAGE on a real tokenized ZH SFT record:** `(input_ids==151655)|(input_ids==151656)`
  count must equal the processor's grid arithmetic `Σ prod(image_grid_thw_i)/merge_size²`, and
  vision ∪ text must partition the sequence exactly once. **RUN THIS PREREG: PASS** (2,688 = 2,688 at
  the deployed 262,144-px cap; 21,528 = 21,528 uncapped; all masked ids ∈ {151655}).
- **S4 GRAD FLOW (MokA's D1 dead-parameter trap):** after one backward, `lora_A`, `lora_A_v` **and**
  the shared `lora_B` all have non-zero grads and **no** `requires_grad=True` parameter has
  `grad is None`. **RUN THIS PREREG: PASS.** (Note pre-declared: PEFT zero-inits `lora_B`, so
  `dL/dA ≡ 0` at step 0 for *every* LoRA; the check is made at a perturbed-`B` point.)
- **S5 SAVE/LOAD round-trip** (the `lora_A_v` keys survive `save_pretrained`, are silently dropped by
  a bare `from_pretrained`, and are restored bit-exactly by `install_moka` + `load_moka_a_v`; a
  **generic** adapter passed to `--moka` is **refused**). **RUN THIS PREREG: PASS.**
- **S6 MERGE GUARD:** `merge_and_unload()`, `merge()`, `get_delta_weight()` all **raise**.
  **RUN THIS PREREG: PASS.**
- **S7 PARAM BUDGET:** exactly 40,370,176 → 58,490,880 = 1.448864×. **RUN THIS PREREG: PASS.**
- **S3 STRICT GUARD:** a routed layer invoked with no stashed mask **raises** rather than silently
  falling back to plain LoRA (which would make the whole arm a null-op). **RUN THIS PREREG: PASS.**
- **GPU smoke (executor, ~0.2 GPU-h, §4.4):** 10-step SFT (loss finite, grads reach **both** `A_t`
  and `A_v` with norm > 0 each, exactly ONE `lora_B` per layer, `fallback_calls == 0`); `--moka`
  2-video extraction (shapes `(2,3584)`, finite); and **KS-parity: a no-flag (`--moka` OFF)
  re-extraction of one stream on the banked generic adapter reproduces the banked cache
  `max|Δ| == 0.0` bit-exact** (the READOUT job-13468 R0 precedent).

### 3.3 `KS-parity` (machinery guard, pre-science)

Covered by the last bullet of §3.2: **fail ⇒ HALT (plumbing/stack drift), not a result.**

### 3.4 `KS-MOKA-0b` — merged-vs-unmerged extraction drift (pre-verdict, §1.2 Stage A0)

**Bar:** **mean per-item cosine(unmerged, merged) ≥ 0.9999** on the banked ZH cache, on **all** 6
(split × stream) cells. **If ANY cell < 0.9999**, a same-path **unmerged** floor head run (3 seeds,
+0.05 GPU-h, **+3 test evaluations**) becomes **MANDATORY** before any verdict, and the arm is then
paired against **that** floor instead of 13150. Pre-declared here so the contingent test-touch is
budgeted, not improvised. This probe reads **no labels** ⇒ **0 test-touch**.

### 3.5 `KS-MOKA-1` — ZH decisive, mid-family (the auto-defund switch)

**If on BOTH protocols the 3-seed mean paired Δacc ≤ 0, OR the acc sign is not 3/3 positive, the ZH
arm is DEAD and the HateMM stage-2 leg is AUTO-DEFUNDED** (saves ≈ 4.2 GPU-h). Bank as Law-I / FLAT.
**Secondary read (within-noise kill):** mean paired Δacc `< +0.015` on **both** protocols (inside the
±0.014 band, §2.3) ⇒ also **KILL**. State the kill explicitly at verdict time.

### 3.6 `KS-MOKA-2` — routing-is-real ($0, emitted automatically by job 1)

Report per-layer `‖A_v − A_t‖_F / ‖A_t‖_F`. **If the median layer is < 0.05 (5 % relative), the two
down-projections have converged: the arm is a NULL-OP and any observed delta is head-seed noise, NOT
routing** — report it exactly that way and do **not** claim a routing effect. (Reference: at
initialisation two independent Kaiming draws give ≈ 1.41; smoke S4 measured 1.4135 / 1.4210.)

### 3.7 `KS-MOKA-3` — stream decomposition ($0, MANDATORY before ANY claim). The honest-core clause.

Re-run the F45/F58 train-LOO img/text-AUC machinery (`scripts/analysis/encoder_swap_geometry.py`
read-outs; movement rule verbatim from `scripts/analysis/hatemm_lora_stream_decomp.py`: **MOVED** iff
`dAUC ≥ +0.010` on train-LOO **and** `≥ +0.005` on dev with the same sign; **FLAT** iff
`|dAUC| < 0.010` on train-LOO) on the **MokA cache vs the generic-LoRA floor cache**, **TRAIN + DEV
only, zero test-touch**. Three **pre-declared** readings, transcribed from recon §6:

- **text moved** ⇒ the §F0.5 bet is confirmed and the result **must be reported as a *text-side*
  mechanism — NEVER as "MokA protected the visual modality."**
- **image moved, head flat** ⇒ the **9th law-I instance**; report as such.
- **neither moved** ⇒ null-op; cross-check `KS-MOKA-2`.

**This clause is binding on the write-up regardless of the performance outcome.** MokA's advertised
premise is priced dead by F58 + F65 (§F0.5); a PASS may not be narrated as visual-modality protection
unless `KS-MOKA-3` independently shows the image stream moved *and* the head followed.

### 3.8 `KS-regression`

MokA below floor by **≥ 0.030** mean Δacc on either protocol ⇒ report as a measured **REGRESSION**
finding ("modality-routed `A` degrades the ZH adaptation"). A note within the KS frame, not a new bite.

### 3.9 Ban-collision closure (disclosed; NOT bans)

- **EN is NOT re-opened and no EN arm exists** (recon §4.1, transcribed): EN is closed at frozen (F50),
  collapsed-adapted-deployed (B4/F53) and healthy-img+adapted-text (F55) levels; `REDTEAM_BAN_SCOPE_AUDIT.md`
  GAP-5 named exactly two unmeasured levels — (a) EN+audio, (b) EN+vision-obligatory SFT — and **MokA
  is neither**; more decisively **F65 discharged GAP-5b by measurement** ("first lever ever to move the
  collapsed EN image stream", K-V2 TIE on both datasets and both protocols). **Proposing an EN MokA
  cell would be a re-burn.**
- **Not F65 / vision-unfreeze:** F65 unfroze the vision **tower**; MokA changes the **LLM-decoder
  adapter's routing** and leaves `freeze_vision_tower: true` untouched. Different object.
- **Not F70 / readout-grid / ZHPROMPT:** those vary the extraction *readout* or *prompt* over a fixed
  encoder; this changes the **SFT adapter structure**. Different axis.
- **Not F66/F63 (selection-lock):** those bound inference-side headroom over a **fixed** φ₀; a new SFT
  produces a **different** φ. F66 is silent on trained reshaping (the NCA precedent). Honest
  counter-pressure (§F0.5), not a ban.
- **Not LITSWEEP2 HUNT-3:** HUNT-3 priced richer PEFT adapters at ~0 as a **prior**, not a measurement;
  it is the reason the prior is 5–8 %, not a ban.
- **NOT** cross-seed ensemble / OCR / gold-in-method / cross-dataset mixing / external API /
  target-as-structure. **Clear.**

### 3.10 Multiplicity + scope of THIS submit

- **ONE pre-registered family = ONE multiplicity bite**, spanning both jobs.
- **Scope FROZEN.** Cross-attention, `r_v ≠ r_t`, a routed `B`, a third mask (MokA's `question_mask`),
  a different mask definition, an EN arm, a HateMM arm, or a second SFT seed are each a **new**
  pre-declared arm and re-cost a bite.
- **Verdict is rendered by an independent 0-context reviewer against this prereg VERBATIM.** The
  executor transcribes raw both-protocol per-seed numbers (line-numbered) and applies **no** gates.

### 3.11 Gate order

G-repro sha re-verify (§4.1) → **codex gate (§4.5)** → CPU smoke `moka_smoke.py` all-PASS (§4.3) →
GPU smoke incl. **KS-parity bit-exact** (§4.4) → **job 1** → job-1 post-run asserts + **`KS-MOKA-2`** →
**job 2** Stage A0 **`KS-MOKA-0b`** → Stage S shape sanity → the 3 budgeted test reads →
**`KS-MOKA-1`** → **`KS-MOKA-3`** → FORMAL bar (§2.3, both protocols) → `KS-regression` note.

---

## 4. G-repro, smoke plan, collision safety, codex gate

### 4.1 G-repro discipline

- **(a) Patched-file sha gate.** At submit time re-run `sha256sum` on all 7 artifacts (§5.1) + this
  file — any mismatch = **authorization VOID**.
- **(b) Untouched-machinery gate.** `src/run_rac.py` must still be
  `b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3` (the `ZHPROMPT_PREREG.md §5.2`
  sha) and `src/model/loss.py`, `src/model/classifier.py`, `src/utils/retrieval.py` git-clean.
- **(c) Vendored-tree gate.** `RA-HMD/LLAMA-FACTORY-Ver202512` gitlink must still point at
  `a912747c408b3c661b4029ecf1d88b9d91c7f1a8`; the only new file under it is the frozen yaml.
- **(d) Head same-code.** `run_one()` in `moka_extract_head.sbatch` must remain **byte-identical** to
  `enc3seed_zh_b3.sbatch` lines 42-83 (sha `4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad`).
- **(e) Default==identity.** `--moka` / `--no_merge` default OFF ⇒ deployed merged path; the ZHPROMPT
  prompt-arg identity proof still holds (§F0.8). Runtime confirmation = KS-parity (§3.2).

### 4.2 CPU verification RUN THIS PREREG (all PASS; `$0`)

- `python -m py_compile` on `routed_lora.py`, `train_moka.py`, the extractor, `moka_smoke.py` — **PASS**.
- `python scripts/analysis/moka_smoke.py` → **ALL SMOKE CHECKS PASS** (S1–S8; the S2 identity control
  at `max|Δ| = 0.000e+00` in every cell; dense-vs-gather cross-check `≤ 1.19e-07`).
- **Global-RNG neutrality of `install_moka`** (CPU): `manual_seed(99); randn(4)` vs
  `manual_seed(99); install_moka(); randn(4)` → **bit-identical**; and the `A_v` draw is identical
  under two different global RNG states (§11 DEV-3) — **PASS**.
- Monkey-patch runtime check (CPU import of the real LLaMA-Factory): `get_peft_model` replaced
  (**True**), original preserved (`peft.mapping`), `run_exp` resolves to `llamafactory.train.tuner`,
  and `adapter._setup_lora_tuning` source contains `get_peft_model(model, peft_config)` — **PASS**.
- `diff` deployed ZH yaml vs MokA yaml = **1 line** (`output_dir`) — **PASS**.
- `run_one()` byte-identity vs `enc3seed_zh_b3.sbatch` lines 42-83 — **PASS** (exact line-list compare).
- `bash -n` on both sbatch — **SYNTAX_OK**.
- Extractor no-flag `parse_args_sys([])`: `moka=False`, `no_merge=False`, prompt args at the English
  defaults, assembled `text_prompt` byte-matches the deployed literal — **PASS**.
- Floors §2 re-parsed from raw trainlogs with the embedded `enc3seed_zh_b3` parser — **PASS**.

### 4.3 CPU smoke contents (`scripts/analysis/moka_smoke.py`, the `KS-MOKA-0` gate)

`S1` install/hook/`A_v` presence · `S2` **identity control** (tied `A_v`; all-text / all-vision /
mixed; `max|Δ| == 0.0`) **+ dense-vs-gather cross-impl agreement** (an indexing-bug guard) · `S3`
strict-mode raise · `S4` grad flow to `A_t`, `A_v`, shared `B` + no dead params + `KS-MOKA-2` report ·
`S5` save/load round-trip incl. the silent-drop demonstration and the generic-adapter refusal · `S6`
merge guard · `S7` parameter budget (exact integers) · `S8` mask coverage on a **real** tokenized ZH
SFT record at the deployed 262,144-px cap.

### 4.4 GPU smoke plan (executor runs BEFORE the real submit; ≈ 0.2 GPU-h; leave no artifact)

1. **10-step SFT** — a throwaway copy of the yaml with `max_steps: 10`,
   `output_dir: logging/_smoke_moka`. Assert: loss finite; the `[moka] routed 196 lora.Linear layers`
   line present; `[moka] trainable params` shows `lora_A_v == lora_A_t` and `trainable_total ==
   58,490,880`; after the run, `moka_stats()['fallback_calls'] == 0`; `‖A_v−A_t‖` report emitted.
   Then `rm -rf logging/_smoke_moka`.
2. **`--moka` 2-video extraction** — `--splits test --limit 2 --EXP_FOLDER logging/_smoke_moka
   --out_model_tag _mokasmoke`; assert shapes `(2,3584)`, all finite, `[moka] routed layers: 196 |
   lora_A_v tensors loaded: 196`. Then `rm -rf logging/_smoke_moka`.
3. **KS-parity bit-exact (`§3.2`, REQUIRED).** Run the edited extractor with **NO** new flags on
   `--splits test --limit 8` against the **banked generic** adapter to a throwaway folder and assert
   `img max|Δ| == 0.0 AND text max|Δ| == 0.0` vs `data/CLIP_Embedding/MHC_zh/test_seen_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`.
   **Fail ⇒ HALT.**

### 4.5 CODEX GATE — MANDATORY (recon §6: "not a flag-only arm; the FUSIONCAT §4.5 exemption does NOT apply")

Iterative `codex-code-review` loop until Claude **and** Codex agree, focused on **model internals**:
1. **`MokaLinear.forward`** — routing algebra; `vision ∪ ~vision` partitions the sequence exactly once;
   `torch.where` broadcast over the rank axis; dtype handling vs upstream PEFT (we keep upstream's
   `x.to(lora_A.weight.dtype)` and the `torch_result_dtype` restore that MokA commented out, recon D7);
   the shared-`B` application; that **no** `lora_B_v` exists (MokA's D1 dead twin is not reproduced).
2. **The mask pre-hook** — registered on the base `Qwen2_5_VLForConditionalGeneration`; reads
   `kwargs["input_ids"]`; **never cleared at end-of-forward** so gradient-checkpointing recompute
   still sees it (recon §2.4); overwritten once per batch; correctness under `eval_strategy: epoch`.
3. **Positional validity of the mask** — that Qwen2.5-VL `masked_scatter`s vision embeddings **in
   place** (transformers 4.49 `modeling_qwen2_5_vl.py:1803-1809,:1821-1827`) so the sequence axis is
   never shifted, and that all 7 targeted projections share that axis (`down_proj` sees 18,944-wide
   inputs but the same `S`).
4. **PEFT 0.14.0 API surface** — `adapter_layer_names` extension; `_mark_only_adapters_as_trainable`
   prefix `"lora_"` keeping `lora_A_v` trainable; `get_peft_model_state_dict` (`save_and_load.py:85`)
   keeping it; `_insert_adapter_name_into_state_dict` round-tripping `lora_A_v.weight` ↔
   `lora_A_v.default.weight`; the fp32 cast at `adapter.py:314-316` reaching `lora_A_v`; the in-place
   `__class__` swap being safe before optimizer construction.
5. **The unmerged extraction path** — `install_moka` **before** `load_moka_a_v`; `PeftModel.config`
   resolution through `__getattr__` (the extractor reads `model.config.hidden_size`); `--moka` /
   `--no_merge` default-OFF byte-identity; the merge guard raising rather than producing a wrong ΔW.
6. **Strict mode** — that a missing mask raises instead of silently degrading the arm to plain LoRA.
7. **Both sbatch** — sequential submission (never 2×16 CPU), collision guards, the shape-sanity gate
   preceding the budgeted test reads, `run_one` byte-identity.

**Blocking findings ⇒ fix the code + re-freeze the shas (§5.3) + re-run this gate (§4.6).**

### 4.6 Code-fix ⇒ re-freeze clause (verbatim-ported from `NCA_PREREG.md §4.5/§5.3`)

**If the codex gate (§4.5), the CPU smoke (§4.3) or the GPU smoke (§4.4) forces a code fix, the
affected artifact shas change and the freeze block (§5.3) MUST be re-issued** — a new independent
0-context review is re-run against the amended files before submit. No code edit lands silently
post-freeze; the executor re-runs `sha256sum` at submit and any mismatch = **authorization VOID**.

### 4.7 Collision safety (verified ABSENT this prereg — re-check at submit)

- `logging/lora/MHC_zh_moka` — **does not exist** (job 1 also aborts if the adapter file appears).
- `data/CLIP_Embedding/MHC_zh/*-moka_HF.pt` and `*-LoRA_HF-um.pt` — **do not exist**; the `-moka_HF` /
  `-um` tags are distinct from every banked tag (`_HF`, `-LoRA_HF`, `-LoRA-curric_HF`, `-LoRA-bidir_HF`,
  `-ro_{L28,L24,ow_L28,ow_L24}`, `-zhp`, `-32B`, `p3pool*`, `p8*`, `p9*`) ⇒ **cannot clobber**.
- `logging/Retrieval/MHC_zh/RAC_video_moka*` — **does not exist** ⇒ fresh group; `--force False`
  never trips the `run_rac.py:1059-1062` hard-abort.
- `slurm/logs/*moka*.trainlog` — **do not exist**.
- `refine-logs/MOKA_PREREG.md`, both sbatch, the yaml, `src/moka/*` — created by this prereg (no prior).
- Banked inputs `logging/lora/MHC_zh`, floor caches and 13150 trainlogs are **read-only**; this family
  writes none of them.
- Smoke throwaways (`logging/_smoke_moka/`) must be deleted before submit.

---

## 5. Artifacts + hash-freeze block

### 5.1 New / edited artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current) |
|---|---|---|---|
| P | `refine-logs/MOKA_PREREG.md` | **NEW** — this file | *(reviewer fills at freeze)* |
| A | `src/moka/routed_lora.py` | **NEW** (372 L) — `MokaLinear` + install/load/report; MokA credit header | `9b0fc502193b0521f1359978f85b35b6ce98034d1371f174315c8f1a219a8386` |
| B | `src/moka/train_moka.py` | **NEW** (52 L) — `get_peft_model` monkey-patch + `run_exp()` | `fae40487263fd7f65e2d0566205e57d3a2caf1b9d1477c693c7cdfdc891c9749` |
| C | `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | **EDITED** (+33/−2) — `--no_merge` / `--moka`, both default OFF (identity) | `75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399` |
| D | `scripts/analysis/moka_smoke.py` | **NEW** (324 L) — the `KS-MOKA-0` CPU gate S1–S8 | `843dace46611d3a2f5e6942a5f0e780fe8b2fb623a3c969949c142bd559d7793` |
| E | `scripts/slurm/lora_sft_moka.sbatch` | **NEW** (113 L) — job 1 (16 CPU/120 G/1 A100) + `KS-MOKA-2` readout | `df3c9a6a8721f5d97935e4b87137732726329877c14ee8635902976eaf70e38b` |
| F | `scripts/slurm/moka_extract_head.sbatch` | **NEW** (162 L) — job 2 (8 CPU/64 G/1 A100), `run_one` byte-identical to `enc3seed_zh_b3` | `fd1b7f295cdb7106e0e64629cb1a2391355f9daad4ab0f2ad773292f48b31bde` |
| G | `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/mhc_zh_qwen25vl_lora_moka_sft.yaml` | **NEW** (48 L, inside the gitlink ⇒ hash-frozen, not committed — CAND2/LORA_HATEMM precedent) | `51b883e9f0a78c26d9b4af185b54a4703a250a3cab4c947756782c6c8fe49764` |

### 5.2 Reused-unchanged machinery (verify sha / git-clean at submit; do NOT edit)

| path | role | sha256 |
|---|---|---|
| `src/run_rac.py` | deployed head runner (NCA/head-recipe keys inert; NO edit; git-clean) | `b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3` |
| `scripts/slurm/enc3seed_zh_b3.sbatch` | same-code anchor (`run_one` source; produced floor 13150) | `4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad` |
| `scripts/slurm/lora_sft.sbatch` | clone source for job 1 | `e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f` |
| `RA-HMD/…/mhc_zh_qwen25vl_lora_sft.yaml` | copy source for the MokA yaml (1-line diff) | `2f2429fbd8f6b0b82fba173a4efc5f12ae75cf5e5bce791c3819663c5f1439ea` |
| `src/model/loss.py`, `src/model/classifier.py`, `src/utils/retrieval.py` | deployed head/loss/fusion/retrieval (NO edit) | *(git-clean verified §4.2)* |
| `RA-HMD/LLAMA-FACTORY-Ver202512` | vendored tree, **ZERO lines edited** | gitlink `a912747c408b3c661b4029ecf1d88b9d91c7f1a8` |
| `logging/lora/MHC_zh` | banked generic ZH adapter (`KS-MOKA-0b` input; read-only) | *(present, 2026-07-02)* |
| `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | `KS-MOKA-0b` / KS-parity targets; not clobbered | *(present)* |
| `external/baselines/MokA` @ `b28e834` | the credited source (read-only; `external/` is gitignored) | *(clone present)* |

### 5.3 Hash-freeze (filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file MOKA_PREREG.md, after review>
A 9b0fc502193b0521f1359978f85b35b6ce98034d1371f174315c8f1a219a8386  src/moka/routed_lora.py
B fae40487263fd7f65e2d0566205e57d3a2caf1b9d1477c693c7cdfdc891c9749  src/moka/train_moka.py
C 75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399  src/utils/generate_VideoMLLM_embedding_lora_HF.py
D 843dace46611d3a2f5e6942a5f0e780fe8b2fb623a3c969949c142bd559d7793  scripts/analysis/moka_smoke.py
E df3c9a6a8721f5d97935e4b87137732726329877c14ee8635902976eaf70e38b  scripts/slurm/lora_sft_moka.sbatch
F fd1b7f295cdb7106e0e64629cb1a2391355f9daad4ab0f2ad773292f48b31bde  scripts/slurm/moka_extract_head.sbatch
G 51b883e9f0a78c26d9b4af185b54a4703a250a3cab4c947756782c6c8fe49764  RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/mhc_zh_qwen25vl_lora_moka_sft.yaml
```
Executor re-runs `sha256sum` on A–G (and this file) + confirms `run_rac.py b85eb72…`, the
loss/classifier/retrieval git-clean, and the LF gitlink `a912747…` at submit time; any mismatch =
**authorization VOID**. **§4.6 applies: any code fix ⇒ re-freeze + re-review.**

---

## 6. Execution plan + GPU budget

**Order:**
1. Pre-submit: G-repro (§4.1) → **codex gate (§4.5)** → CPU smoke all-PASS (§4.3) → GPU smoke incl.
   **KS-parity bit-exact** (§4.4) → delete `logging/_smoke_moka`. Only on all-clear:
2. `sbatch scripts/slurm/lora_sft_moka.sbatch` (job 1). Initial `PENDING (JobHeldUser)` is expected —
   **WAIT for auto-release, never force** (CLAUDE.md). On `COMPLETED`, read the job-1 post-run asserts
   and the **`KS-MOKA-2`** median.
3. Only after job 1 is `COMPLETED`: `sbatch scripts/slurm/moka_extract_head.sbatch` (job 2).
4. Executor transcribes raw per-seed both-protocol numbers (line-numbered) and applies **no** gates;
   an **independent 0-context reviewer** renders the verdict against this prereg **verbatim**.

**Budget (all A100-h; wall times from `sacct`, not estimated):**

| item | basis | GPU-h |
|---|---|---|
| GPU smoke (10-step SFT + 2-video extract + KS-parity) | — | 0.2 |
| `KS-MOKA-0b` unmerged extract, banked adapter, 3 splits | `lora_embed` 13234/13239/13240/13245/13302 = `00:26:17`–`00:37:25` | 0.6 |
| MokA-ZH SFT | `train_runtime` 8,635.998 s (job 12143 wall `02:39:49`) × ~1.2 + ~0.25 h build/load | **3.1** |
| MokA-ZH extraction (`--moka`, 3 splits) | 0.6 × ~1.15 | 0.7 |
| 3 head-seeds (`enc3s`) | job 13150 = `00:02:46` for exactly 3 runs | 0.05 |
| **Stage-1 total** | | **≈ 4.65** |

**CAP = 4.7 GPU-h for this submit.** Contingent `+0.05` if `KS-MOKA-0b` forces the same-path floor
(§3.4). **Stage 2 (HateMM, ≈ +4.2) is AUTO-DEFUNDED by `KS-MOKA-1` and NOT authorised here.**

**Resource plan (STANDING INFRA RULE compliant):** job 1 = 16 CPU / 120 G / 1 A100; job 2 = 8 CPU /
64 G / 1 A100; **submitted sequentially so at most one is ever queued/running** ⇒ never two 16-CPU
jobs, never above the 16 CPU / 128 G / 2 GPU cap. `conda activate HateVideo`; `sbatch` with **NO
`--time`**; both sources `conda.sh` directly and runs `disk_guard.sh`; B2-push of derived `.pt` + logs
only (**raw videos never leave the machine**).

**Test-touch:** the **3** head reads of job 2 are the ONLY budgeted MokA test evaluations.
`KS-MOKA-0b` = 0. Contingent +3 (§3.4). **Zero test-touch before the independent verdict.**

**No job is submitted by this prereg author.**

---

## 7. Outcome table template (filled ONLY from raw trainlogs at verdict time)

**MokA-ZH vs floor 13150 (§2.1)** — from
`slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA-moka_HF_seed{0,1,2}_<JID>.trainlog`:

| seed | protocol | MokA acc/F1 | floor acc/F1 | Δ(MokA−floor) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8322 / 0.8023 | ___ |
| 1 | val-sel | ___ | 0.8255 / 0.7956 | ___ |
| 2 | val-sel | ___ | 0.8389 / 0.8065 | ___ |
| **mean** | **val-sel** | ___ | **0.8322 / 0.8015** | **___** |
| 0 | final-ep | ___ | 0.8456 / 0.8181 | ___ |
| 1 | final-ep | ___ | 0.8389 / 0.8113 | ___ |
| 2 | final-ep | ___ | 0.8523 / 0.8226 | ___ |
| **mean** | **final-ep** | ___ | **0.8456 / 0.8173** | **___** |

**Fixed write-up format:**

```
KS-MOKA-0  (machinery, incl. identity control + KS-parity bit-exact): <PASS | HALT>
KS-MOKA-0b (merge drift): worst mean per-item cosine = ____   -> <>=0.9999 OK | same-path floor MANDATORY>
KS-MOKA-2  (routing is real): median ||A_v-A_t||_F/||A_t||_F = ____  -> <routing real | NULL-OP>
MokA-ZH:   final-epoch: <pass/fail>; val-selected: <pass/fail>   [FORMAL §2.3]
KS-MOKA-1: <ZH DEAD, HateMM stage-2 AUTO-DEFUNDED | survives>
KS-MOKA-3: <text moved -> TEXT-SIDE mechanism | image moved + head flat -> 9th law-I | neither -> null-op>
           (NEVER narrate any outcome as "MokA protected the visual modality" unless KS-MOKA-3 shows
            the image stream moved AND the head followed.)
KS-regression: <note if mean Δacc <= -0.030 on either protocol>
Restate F0.2 (single encoder draw, encoder-seed noise NOT separable) and F0.6 (94.6% vision tokens;
MokA's own regime is 98.4% text) in the verdict, whatever the outcome.
```

---

## 8. What a PASS / FAIL means for the goal

- **`KS-MOKA-1` fires (the honest modal case, ~70 %):** modality-routed adaptation carries no net vote
  signal on the one dataset where the adaptation operator is measurably load-bearing (F58: "ZH's is
  **LoRA-SPECIFIC** (frozen fails)") ⇒ the **PEFT-structure axis is CLOSED** at ≈ 4.6 GPU-h, the
  HateMM leg is auto-defunded (−4.2 GPU-h), and — depending on `KS-MOKA-3` — the result is banked
  either as a null-op or as the **9th law-I instance**. A genuinely un-run axis converted to a
  measured door-closer, with a decomposition attached.
- **Survives `KS-MOKA-1` but below FORMAL (~20–25 %):** measured-not-promoted limbo; bank the weak
  positive with the `KS-MOKA-3` mechanism reading. Most plausibly one protocol only, the other eaten
  by the 78-dev selection wall (F45/F63).
- **Clears FORMAL (≥ +0.030/+0.030, 3/3, both protocols; ~5–8 %):** the ZH leg — the **binding goal
  leg** — converts. That is a **performance row plus a mechanism sentence**, and per F0.3 the novelty
  claim is bounded to *first application of modality-routed PEFT to hateful-video encoders, crediting
  MokA*. It would also unlock the stage-2 HateMM leg under a **new prereg amendment** (recon §4.2
  still judges HateMM's +0.030 from 0.8791 "arithmetically implausible", so the amendment must argue
  its own case). A single-dataset pass is **not** the ≥2-dataset headline of rule (5).

**Framing sentence (verbatim):** *this measurement tests the one adapter-structure axis no banked
adapter ever varied — shared vs modality-routed LoRA down-projection — on the ZH cell where the
adaptation operator is measurably load-bearing, 3 head-seeds paired dual-protocol against the banked
generic-LoRA floor 13150, at unchanged per-token rank; the bet is a sharpened, undiluted TEXT-side
subspace (the stream that carries both measured passes), NOT MokA's advertised visual-modality
protection, which F58 + F65 already price dead; a pass is a performance row plus a mechanism sentence,
with transplant novelty claimable only as first-application and only with MokA credited.*

---

## 9. Provenance index

- Recon (design authority; GO, ZH-first, all locked decisions): `refine-logs/MOKA_FORENSIC_RECON.md` (`dbf30f1`).
- MokA source (credited, user-ungated): `external/baselines/MokA` @ `b28e834` —
  `VisualText/modified_peft/tuners/lora/layer.py:548-681` (routed forward, shared `B` at `:657`,
  cross-attention at `:627-653` **not ported**, debug token counts at `:565-566`),
  `VisualText/train/train.py:210-211` (mask by token-id equality),
  `AudioVisualText/peft_hyper/tuners/lora.py:460-532` (the dense formulation we freeze).
- Deployed SFT invocation replicated: `scripts/slurm/lora_sft.sbatch:78-80`
  (`cd $LF_ROOT && python src/train.py $CONFIG`) with
  `CONFIG = RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/mhc_zh_qwen25vl_lora_sft.yaml`
  (the config that produced `logging/lora/MHC_zh`, job **12143**, 2026-07-02, wall `02:39:49`,
  `train_runtime` 8,635.998 s).
- PEFT insertion point: `RA-HMD/LLAMA-FACTORY-Ver202512/src/llamafactory/model/adapter.py:19` (import),
  `:300-305` (`LoraConfig`), `:312` (`get_peft_model`), `:314-316` (fp32 cast of trainables);
  called from `model/loader.py:190`.
- PEFT 0.14.0 internals relied on: `tuners/lora/layer.py:106-163` (`update_layer`), `:164-188`
  (`reset_lora_parameters`, Kaiming `a=sqrt(5)`), `:596-640` (upstream `Linear.forward`);
  `tuners/lora/model.py:281-284` (`_mark_only_adapters_as_trainable`, prefix `"lora_"`);
  `utils/save_and_load.py:85` (state-dict filter), `:277` (adapter-name strip), `:310-326`
  (`_insert_adapter_name_into_state_dict`), `:451` (`load_state_dict(strict=False)` — the silent-drop trap).
- Mask validity: transformers 4.49 `models/qwen2_5_vl/modeling_qwen2_5_vl.py:1803-1809,:1821-1827`
  (`masked_scatter` in place); the extractor's own invariant assert at
  `src/utils/generate_VideoMLLM_embedding_lora_HF.py:347`; token ids from `logging/lora/MHC_zh/added_tokens.json`
  (`<|image_pad|>` 151655, `<|video_pad|>` 151656).
- Floor (re-derived §2.1): `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`
  (`B3_EXECUTION_RECORD.md`, `NCA_PREREG.md §2.1`, `ZHPROMPT_PREREG.md §2.1`). Stage-2 reference §2.2:
  `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`.
- Protocol + decision rule (verbatim): `research-wiki/experiments/exp-encoder-3seed.md:73-85`.
- Same-code anchor: `scripts/slurm/enc3seed_zh_b3.sbatch:42-83`; output-path keying `src/run_rac.py:1010-1062`.
- Stream-decomposition machinery (`KS-MOKA-3`): `scripts/analysis/encoder_swap_geometry.py`,
  `scripts/analysis/hatemm_lora_stream_decomp.py` (its pre-declared MOVED/FLAT rule), `refine-logs/B3_ZH_LORA_DECOMPOSITION.md` (F45).
- Walls / counter-pressure: F45, F53, F55, F58, F63, F65, F66, F70, F75, F83
  (`autoresearch/goal_mllm_plus3/state/findings.jsonl`); GAP-5 `refine-logs/REDTEAM_BAN_SCOPE_AUDIT.md:217-260`;
  `LITSWEEP2_FRESH_2026` HUNT-3. Licence ruling: `state/progress.json` (`6c4766e`).
- Freeze precedent for a gitlink-resident yaml: `refine-logs/CAND2_FREEZE.md:21-22,35-36`,
  `refine-logs/LORA_HATEMM_FREEZE.md:18,24`.

**Required statements:** **ZERO GPU / SLURM / Modal spent** by this prereg author (only pure-CPU
login-node work: `py_compile`, the `moka_smoke.py` CPU gate, a tokenizer sweep, floor re-parsing,
`sha256sum`, `bash -n`, `diff`, `sacct` read-only queries — seconds to a couple of minutes; **no
held-out test metric produced**). **No `state/` mutated. No `research-wiki/` mutated. NO job
submitted. Not pushed.** All floor numbers re-parsed from banked completed-run trainlogs
(numeric-provenance discipline; the ZH means bit-match recon §4.2 and `ZHPROMPT_PREREG.md §2.1`).

---

## 10. DEV items — foreseeable execution pitfalls

1. **DEV-A (their bf16 hard-code — NOT ported).** MokA's `update_layer` hard-codes
   `dtype=torch.bfloat16` on both `A` and `B` (recon D2) and comments out the dtype casts and the
   `torch_result_dtype` restore (recon D7). **We follow upstream PEFT instead:** `lora_A_v` is created
   at `lora_A`'s dtype/device, `x.to(lora_A.weight.dtype)` is kept, and the result is restored to
   `torch_result_dtype`. Under `bf16: true` LLaMA-Factory casts trainables to fp32 at
   `adapter.py:314-316`, and `lora_A_v` is created **before** that line, so it is cast identically to
   `lora_A`. If a future recipe sets `lora_dropout > 0`, `install_moka` **refuses to install** (a
   non-Identity dropout draws different RNG per modality group and would void the identity control).
2. **DEV-B (print-spam removed).** MokA prints `'train mode'` per decoder layer per step
   (recon D3, `modeling_llama.py:316`). Nothing in our port prints inside `forward`; the only prints
   are two lines at `get_peft_model` time and the post-run readouts.
3. **DEV-C (PEFT 0.14.0 API surface).** Verified this prereg: `lora_A_v` is kept trainable (prefix
   `"lora_"`), saved by `get_peft_model_state_dict`, and round-trips through
   `_insert_adapter_name_into_state_dict`. **The trap:** `set_peft_model_state_dict` loads with
   `strict=False`, so a bare `PeftModel.from_pretrained` **silently drops** `lora_A_v` — hence
   `--moka` calls `install_moka` **then** `load_moka_a_v`, which raises unless every routed layer got
   its tensor and every checkpoint tensor was consumed (smoke S5 exercises both branches). If PEFT is
   ever upgraded (`requirements.txt:6` allows `<=0.17.1`), **re-run the full smoke before trusting it.**
4. **DEV-D (JobHeldUser).** Initial `PENDING (JobHeldUser)` is expected on both jobs — **WAIT for
   auto-release, never force** (CLAUDE.md).
5. **DEV-E (disk / `save_total_limit`).** The FS holding `/data/jehc223/RGCL` was at **521 G avail /
   97 % used** at prereg time. The MokA adapter set is ≈ **1.0 G** (4 saves × ~234 MB, i.e. 1.4489× the
   deployed 161 MB), vs 678 M for a deployed adapter dir. Job 1 preflights **≥ 25 G** and runs
   `disk_guard.sh`. **`save_total_limit` is deliberately NOT set in the yaml** so the `diff` vs the
   deployed recipe stays at exactly one line (`output_dir`); it is training-inert, so if the reviewer
   prefers a hard cap, adding `save_total_limit: 1` is an acceptable amendment that changes only what
   is retained on disk — it would change sha G and trigger §4.6. Alternative with zero yaml change:
   prune `logging/lora/MHC_zh_moka/checkpoint-*` **after** job 2's extraction completes.
6. **DEV-F (gradient-checkpointing recompute).** The mask stash is written in the forward-**pre** hook
   and is **never cleared at end-of-forward**, because checkpointing replays each decoder block during
   backward. It is overwritten once per batch. Codex gate item 2. A regression here would surface as a
   `MOKA_STRICT` raise, not as silent wrong numbers.
7. **DEV-G (strict mode / `MOKA_STRICT`).** Both sbatch export `MOKA_STRICT=1`: a routed layer invoked
   without a valid mask **raises**. This is deliberate — a silent plain-LoRA fallback would make the
   whole arm a null-op that still produces plausible numbers. `MOKA_STRICT=0` exists only as a
   debugging escape hatch and **must not** be set for the pre-registered runs.
8. **DEV-H (`--force False` / group hygiene).** `--group_name RAC_video_moka` is fresh (§4.7) and
   `--exp_comment "_${MODEL}"` keys a distinct output dir, so `--force False` never trips the
   `run_rac.py:1059-1062` hard-abort.
9. **DEV-I (`run_rac.py` evaluates test EVERY epoch).** Selection uses **Val only** (the embedded
   parser); the per-epoch `Test_Retrieval` lines are transcribed but never used to pick an epoch —
   identical to every banked floor. Any tie-break exploration happens on a throwaway group only.
10. **DEV-J (adapter path correctness).** Job 2 asserts both
    `logging/lora/MHC_zh/adapter_model.safetensors` (the `KS-MOKA-0b` input) and
    `logging/lora/MHC_zh_moka/adapter_model.safetensors` (the MokA adapter) exist, and exits 2
    otherwise. `--lora_dir` must point at the top-level adapter dir, **not** a `checkpoint-*` subdir.
11. **DEV-K (unmerged extraction is slower and heavier).** The unmerged forward keeps the adapter live
    (`Wx + B(A_m x)` per targeted projection) and adds a rank-16 pass; budgeted at ×1.1–1.15 of the
    merged extraction (§6). If VRAM becomes tight, reduce `--max_pixels` is **NOT** allowed (it would
    change the encoder input); the correct response is to halt and re-plan.

---

## 11. DEVIATIONS FROM THE RECON — flagged loudly

1. **DEV-1 (routing implemented in MokA's *dense-select* form, not the VisualText gather/scatter).
   MATERIAL / measured.** Recon §3.1 sketched the gather/scatter of `layer.py:603-621`. I implemented
   **both** and froze the **dense** one (`torch.where(vision, A_v(x), A_t(x))`, MokA's own
   `AudioVisualText/peft_hyper/tuners/lora.py:460-532` formulation, which recon §1.3 records as
   "mathematically identical given disjoint masks"). **Reason, measured this prereg:** the recon's
   `KS-MOKA-0` demands **`max|Δ| == 0.0`** against upstream PEFT under a tied `A_v`. The gather/scatter
   form changes the `A` GEMM's `M` dimension and therefore its BLAS blocking, producing a residual of
   **5.96e-08 / 1.19e-07** (fp32, CPU, torch 2.6.0) on **mixed** masks — a tolerance argument the
   recon's threshold does not admit. The dense form gives each `A` the identical full-sequence GEMM and
   reproduces upstream **bit-exactly in all three mask patterns (verified `0.000e+00`)**. The
   gather/scatter path is retained behind `MOKA_ROUTE_IMPL=gather` **solely** so the smoke can
   cross-check the two formulations against each other (they agree to ≤ 1.19e-07 — an indexing-bug
   guard). **Cost of the deviation:** adapter `A` FLOPs ~2× (≈ +1 % of layer FLOPs) because both `A`s
   are evaluated and one output is discarded; **per-token effective rank is unchanged at 16**, so the
   comparability argument survives with the amendment already made in §F0.7.
2. **DEV-2 (text route = `~vision`, i.e. pad positions ARE routed through `A_t`; MokA zeroes them).
   MATERIAL / identity-control-enabling.** MokA pads both masks with `False`
   (`train.py:283-296`), so pad positions receive no LoRA delta at all. Here `vision ∪ ~vision`
   partitions the sequence exactly once, which is what makes the tied-`A_v` identity control **exact**;
   it is also how the **deployed** adapter treats those positions (standard PEFT applies to every
   position). With `per_device_train_batch_size: 1` and single-sample extraction, **no pad position
   exists** in either path, so the two conventions cannot differ in practice.
3. **DEV-3 (`A_v` init is seeded independently of the global RNG). Neutral / reproducibility.** The
   recon pinned "Kaiming (PEFT's `reset_lora_parameters`)". Implemented exactly
   (`kaiming_uniform_(a=sqrt(5))`) but drawn inside `torch.random.fork_rng` with
   `MOKA_INIT_SEED = 20260726 + 8·layer_index`, so the draw is reproducible regardless of when
   `install_moka` runs relative to LLaMA-Factory's global seeding. **Every RNG-consuming statement
   (including `nn.Linear.__init__`'s own `reset_parameters`) is inside the fork, so installing MokA
   leaves the process-global RNG stream bit-identical to an un-patched run — CPU-VERIFIED this prereg
   ("RNG-neutral: True"; and the `A_v` draw is identical under two different global RNG states).**
   `lora_B` stays zero-init ⇒ `ΔW = 0` at step 0, exactly as standard LoRA.
4. **DEV-4 (TWO sbatch submitted sequentially, not one job and not `--dependency=afterok`).
   MATERIAL / infra-rule-aligned.** Recon §3.1 listed two sbatch. Sequential manual submission is
   pinned (§1.0) so the family never presents more than 16 CPU of submit-time aggregate demand,
   avoiding the 29 h-wedge failure mode; `afterok` (24 CPU aggregate) is explicitly rejected.
5. **DEV-5 (extractor patch is +33/−2, not the recon's "+25"). Minor / scope-honest.** The extra lines
   are the **second** flag `--no_merge` — the recon's §3.6 `KS-MOKA-0b` needs a *plain* unmerged
   extraction of the **generic** adapter, which `--moka` cannot provide (it requires `lora_A_v` keys
   and refuses a generic adapter). Both flags default OFF, so the identity guarantee is unchanged.
6. **DEV-6 (`KS-MOKA-2` bar pinned at median < 0.05 ⇒ NULL-OP, and it is emitted automatically).
   Neutral / recon-faithful.** Recon §6 said "< 5 % relative difference at the median layer"; pinned
   verbatim and wired into job 1's post-run block so the number exists before any test read.
7. **DEV-7 (ONE yaml authored, ZH only; no HateMM yaml). Neutral / budget-aligned.** Recon §3.1
   sketched two yamls. Authoring the HateMM one now would put an unauthorised stage-2 artifact inside
   the freeze surface; job 1 hard-refuses `HateMM` with a message naming `KS-MOKA-1` (§1.1).
8. **DEV-8 (NEW material fact the recon did not have: the 94.6 % vision-token share and MokA's
   inverted 98.4 % text regime — §F0.6).** Not a deviation from a recon decision, but a
   prereg-time measurement that changes how the result must be read and argues for the low end of the
   frozen 5–8 % prior. The prior itself is **left at the recon's 5–8 %** rather than unilaterally
   re-priced; the reviewer may tighten it.
