# Independent 0-Context Pre-Registration Review — `VISION_UNFREEZE_PREREG.md`

**Reviewer:** independent 0-context pre-registration reviewer (no prior project context; adversarial mandate;
zero user interaction; no job submitted; prereg NOT modified).
**Date:** 2026-07-20 NZST. **CPU-only** (no GPU/SLURM/Modal spent; `state/` not touched).
**Target:** `refine-logs/VISION_UNFREEZE_PREREG.md` (commit `c1592bb`; on-disk sha256
`a2bb1c45b44de35fdfb959fdf07d1b1146546b0944f37fe95434e49e4ed2be2d`).
**Configs A/B** live in submodule `RA-HMD/LLAMA-FACTORY-Ver202512` at commit `a912747c`
("hatevideo: vision-unfreeze LoRA-SFT configs (EN + HateMM)").
**Method:** every load-bearing number re-derived from primary artifacts on disk — raw 12850/13235/arcbase
trainlogs re-parsed with an independently written parser; the §2.3 image-stream anchor re-computed with the
committed F58 operator over the banked EN train/dev caches; configs/scripts diffed byte-for-byte; the
LLaMA-Factory LoRA resolver read directly; every freeze-block hash recomputed. The prereg's and recon's numbers
were treated as untrusted until independently reproduced.

## VERDICT: **APPROVED-WITH-NOTES** (all three notes non-blocking)

The prereg is hash-integral, floor-faithful to 4dp, config-diff-exact, resolver-correct (the `all` +
`freeze_vision_tower:false` path provably lands LoRA on the ViT blocks while the merger + patch_embed stay
frozen), same-code-paired, leakage-clean, veto-compliant, and its kill-switches + outcome tables are fully
decidable from raw logs by a 0-context verdict reviewer with no interpretive freedom. The K-V2 clean-superset
premise (generic adapter = LLM-only, ZERO `visual.*`) is verified at both the config and the safetensors level,
and the load-bearing §4.1a smoke gate (`n_visual_lora_tensors > 0`) is a genuine runtime abort that catches the
one way the cell could silently degenerate into the generic arm. The three notes below are a
justification-text imprecision (Note 1), a line-count descriptor slip (Note 2), and a data-prep transparency
item (Note 3); none affects decidability, leakage, or the honesty of any bar, and none can be used to
manufacture an unsupported pass. **Cleared to freeze + single-submit.**

---

## Rationale (one paragraph)

The cell measures the one lever the LoRA-SFT family never pulled — LoRA reaching the Qwen2.5-VL ViT blocks
(`freeze_vision_tower:false` + `lora_target:all`, projector frozen) — on MHC-EN (refutation target) and HateMM
(hold/upside), each trained on its own train split only. The design's validity hinges on one property: that the
vis arm is byte-identical to the banked generic-LoRA arm **except** the LoRA reach (LLM ⊕ ViT vs LLM-only), so
K-V2 (vis − generic, paired by head-seed) isolates the vision contribution exactly. That property holds under
audit: the two vis YAMLs differ from their generic parents by **exactly** the 3 claimed lines
(`lora_target`, `freeze_vision_tower`, `output_dir`); the LLaMA-Factory resolver confirms `all` +
`freeze_vision_tower:false` makes the ViT-block Linears LoRA-eligible while `visual.merger` (projector) and
`patch_embed` (Conv3d, `lora_conflict_keys`) remain excluded, and the LLM coverage under `all` is byte-identical
to the generic 7-suffix list; the banked generic adapters carry **88 target modules, ZERO `visual.*`** (config)
and **392 tensors / 40,370,176 params, ZERO `visual.*`** (safetensors); the Stage-3 head `run_one`…`PY` block is
byte-identical to `enc3seed.sbatch`; the EN image-MOVED gate reuses the committed F58 operator verbatim and
reads **train + dev_seen only** (zero test-touch). All comparison floors re-derive to 4dp from the raw
trainlogs with the byte-identical embedded parser, the §2.3 image anchor re-derives exactly (generic-LoRA EN img
0.6236 train / 0.6756 dev), and the DEV-4 machinery-validation delta reproduces bit-for-bit
(+0.0245 train / −0.0109 dev ⇒ FLAT). Because the mining/training is a fixed recipe, the head code is
byte-identical, the branch is a submit-time argument (no hash-void edit), and the executor transcribes raw
per-seed numbers with the verdict rendered independently, the motivated-executor attack surface
(re-draw-and-cherry-pick, protocol/metric shopping, bury a regression, drop the EN kill quietly) is closed by
construction. Novelty (D7) is deferred to the user throughout; a ~10–15% EN pass and a HateMM K-V2 tie are
pre-declared as the most likely outcomes.

---

## CHECK-BY-CHECK

### 1. Floor re-derivation — **PASS (independently re-parsed; all match to 4dp)**

I wrote a standalone parser implementing the `enc3seed.sbatch` embedded rule (val-sel = epoch ≥ warmup 5 with
max `Val_Retrieval` acc, roc tie-break → that epoch's `Test_Retrieval` acc/F1; final = max epoch) and ran it on
the raw `slurm/logs/enc3s_*_{12850,13235}.trainlog` (+ `arcbase_MHC_Qwen…_1227{5,6}` for EN-Qwen s1/s2).
**Every per-seed value and every 3-seed mean in §2.1/§2.2 reproduces exactly** — all 24 seed-level acc/F1 pairs
and all 12 means. Spot table (means):

| arm | protocol | re-parsed mean acc/F1 | prereg |
|---|---|---|---|
| HateMM CLIP (K-V1) | val-sel / final | 0.8202/0.8085 · 0.8124/0.7936 | ✓ ✓ |
| HateMM frozen-Qwen | val-sel / final | 0.8729/0.8648 · 0.8682/0.8591 | ✓ ✓ |
| HateMM generic-LoRA (K-V2) | val-sel / final | 0.8620/0.8545 · 0.8698/0.8618 | ✓ ✓ |
| EN CLIP (K-V1) | val-sel / final | 0.7619/0.6715 · 0.7785/0.7202 | ✓ ✓ |
| EN frozen-Qwen (honesty bar) | val-sel / final | 0.7805/0.7219 · 0.7847/0.7425 | ✓ ✓ |
| EN generic-LoRA (K-V2) | val-sel / final | 0.7598/0.7114 · 0.7785/0.7389 | ✓ ✓ |

The §7.1/§7.2 outcome tables pre-fill the CLIP and generic floors **matched by seed index** (paired-by-seed);
those pre-filled cells also all match my re-parse. The CLIP floor is the ERRATUM-corrected value
(`0.8279/0.8172`), not the withdrawn `0.8732`. Floors are consistent with `LORA_HATEMM_VERDICT_REVIEW.md` and
`CAND2_CURRICULUM_PREREG.md`.

**§2.3 EN image-stream AUC anchor — re-derived with the committed F58 operator** (`encoder_swap_geometry.py`,
image-only stream, EN train n=549 + dev_seen n=80, CPU, `CUDA_VISIBLE_DEVICES=""`):

| encoder (EN, image-only) | train-LOO AUC | dev AUC | prereg |
|---|---|---|---|
| CLIP (healthy) | 0.7338 | 0.7367 | ✓ ✓ |
| frozen-Qwen (collapsed) | 0.5992 | 0.6865 | ✓ ✓ |
| **generic-LoRA (gate anchor)** | **0.6236** | **0.6756** | **✓ ✓** |

Full-precision DEV-4 machinery-validation delta (frozen→generic): train **+0.024450** (4dp `+0.0245`), dev
**−0.010909** (4dp `−0.0109`) ⇒ MOVED-rule vs frozen = **False** ⇒ **FLAT** — reproduces the prereg's DEV-4
claim bit-for-bit (F44 collapse 0.7338→0.5992 reproduced; LLM-only LoRA leaves the EN image FLAT under the
committed operator because dev fails to corroborate the train move).

### 2. Config diffs — **PASS (exactly 3 lines each; no drift)**

`diff mhc_qwen25vl_lora_sft.yaml mhc_qwen25vl_lora_vis_sft.yaml` and the HateMM pair each return **exactly the 3
claimed changed lines**:
- L13 `lora_target: q_proj,…,down_proj` → `all`
- L14 `freeze_vision_tower: true` → `false`
- L27 `output_dir: …/lora/<DS>` → `…/lora/<DS>_vis`

No extra drift. The full vis YAML confirms every §1.2 recipe pin (`stage: sft`, `finetuning_type: lora`,
`lora_rank: 16`, `lora_alpha: 32`, `freeze_multi_modal_projector: true`, `lr 1.0e-4`, `num_train_epochs 3.0`,
cosine, `warmup_ratio 0.05`, bs1 × accum8, bf16, gradient-checkpointing, `cutoff_len 4096`, `save/eval_strategy
epoch`, `dataset: mhc_lora_train`/`eval_dataset: mhc_lora_val`). `lora_dropout` is omitted (LF default 0.0;
confirmed 0.0 in the generic adapter config). The vis arm keeps the generic word-variant dataset pointer ⇒
clean superset (F0.6).

### 3. Superset claim + resolver — **PASS**

- **Generic adapters carry ZERO `visual.*` (config):** `logging/lora/{HateMM,MHC}/adapter_config.json` each list
  **88 `target_modules`, n_visual = 0**, `r: 16`, `lora_alpha: 32`, `lora_dropout: 0.0`.
- **Generic adapters carry ZERO `visual.*` (safetensors):** both `adapter_model.safetensors` = **392 tensors,
  0 visual tensors, 40,370,176 LoRA params** — matching the §1.2 LLM-LoRA footprint bit-exact. So every banked
  "encoder adaptation" adapted the **LLM only** (F0.3 / GAP-5b verified).
- **Resolver (this LLaMA-Factory version) lands LoRA on ViT under `all` + `freeze_vision_tower:false`:**
  `adapter.py:215-216` dispatches `lora_target:all` → `find_all_linear_modules(model, freeze_vision_tower)`.
  In `misc.py`, `lm_head` + `projector_key` are forbidden **unconditionally** (L31, L37-38), and the vision keys
  are added to `forbidden_modules` **only if** `freeze_vision_tower` (L40-41) — so with `false` the ViT-block
  Linears become LoRA-eligible. `visual.py:344-352` registers `qwen2_5_vl` with `projector_key="visual.merger"`,
  `vision_model_keys=["visual.patch_embed","visual.blocks"]`, `lora_conflict_keys=["patch_embed"]`.
  `patch_target_modules` (`visual.py:182-197`, called at `adapter.py:233`) re-applies `get_forbidden_modules`
  (which forbids `visual.merger` because `freeze_multi_modal_projector:true`) **plus** `lora_conflict_keys`
  (`patch_embed`) — so the merger stays frozen and `patch_embed` (a Conv3d, not Linear anyway) is excluded. The
  LLM-side suffix set under `all` is exactly the generic 7 (`q/k/v/o/gate/up/down_proj`); the extra ViT suffixes
  (`qkv`, bare `.proj`) match only ViT modules (PEFT boundary-matched). ⇒ **vis = generic ⊕ ViT-LoRA, a clean
  superset; K-V2 isolates the ViT contribution exactly.**
- **Load-bearing runtime backstop:** §4.1a smoke gate inspects the throwaway smoke adapter for `visual.blocks`
  tensors and **ABORTs if `n_visual_lora_tensors == 0`** — a genuine runtime check that the config actually
  reached the ViT (if it did not, the arm would equal generic and K-V2 would be vacuous). Correctly specified.

### 4. Same-code + syntax + probe leakage — **PASS**

- `run_one`…`PY` block of `enc3seed_lora_vis.sbatch` vs `enc3seed.sbatch`: **BYTE-IDENTICAL** (`diff` empty).
  The load-bearing `python ./src/run_rac.py …` argv is identical; the only manipulated variables reaching the
  run are `--model` (`…-LoRA-vis_HF`) and `--group_name` (`RAC_video_lora_vis`) plus derived `--exp_comment`.
  Full-file diff vs `enc3seed_lora_hatemm.sbatch` = header comments, `-vis` tag, group name, and the arg-driven
  `DATASETS` loop (DEV-2) — nothing load-bearing.
- `bash -n` on both new sbatch = **OK**; `python -m py_compile vis_image_moved_probe.py` = **OK**.
- **Probe imports the committed F58 operator + touches train/dev only:** `vis_image_moved_probe.py:39` imports
  `encoder_swap_geometry as G` — the exact module `hatemm_lora_stream_decomp.py:72` imports; it pins F58's
  `MOVE_TR=0.010`/`MOVE_DV=0.005` (matching `hatemm_lora_stream_decomp.py:86-89`). `G.load` is called **only**
  with `"train"` and `"dev_seen"` (L48-49, and the `--context` path L100); the sole `test` reference is the
  docstring "test is never read." **Zero test-touch, verified in code.** `encoder_swap_geometry.load` reads
  exactly `{BASE}/{ds}/{split}_{tag}.pt` — the split passed, never test.

### 5. Bar arithmetic + internal consistency — **PASS (one non-blocking note; see Note 1)**

- **K-V1** (vis − CLIP): mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND sign 3/3, judged independently per
  protocol — matches `LORA_HATEMM_PREREG` KS-1 and `CAND2` K-C2-1. Decidable from the §2.1/§2.2 per-seed floors.
- **K-V2** (vis − generic, paired by head-seed): mean paired Δacc ≥ +0.010 AND 3/3 positive sign AND mean
  ΔmF1 ≥ 0 — matches `CAND2` K-C2-2 verbatim. TIE = "vision reach adds nothing." F0.2 single-draw caveat
  pre-attached to any PASS.
- **EN image-MOVED gate:** MOVED iff dAUC_img ≥ +0.010 (train-LOO) AND ≥ +0.005 (dev) [F58 verbatim]; anchor
  0.6236/0.6756 (re-derived, §1). Branch point is a submit-time arg to a hash-frozen sbatch — no frozen-file
  edit needed.
- **EN honesty flag (§3.5):** vis must beat frozen-Qwen (EN val-sel 0.7805/0.7219, final 0.7847/0.7425) — the
  bar generic could NOT clear (generic 0.7598 ≤ 0.7805 val-sel; 0.7785 ≤ 0.7847 final — **both verified ≤**).
- **KS-regression:** vis − generic mean Δacc ≤ −0.014 ⇒ KILL. Matches `CAND2`.
- **eval_loss band anchors (§3.7b):** re-read from the generic adapters' `all_results.json` — HateMM
  **0.10844** (→ 0.1084 ✓), MHC **0.16196** (→ 0.1620 ✓). Exact.

All thresholds are fixed pre-registered numbers on raw per-seed logs; both protocols reported separately (fixed
§7.3 write-up) ⇒ no protocol/metric shopping.

### 6. Leakage + veto compliance — **PASS**

- **Single-dataset own-train-split:** EN trains on `mhc_lora_train` (`train.json`, 549), HateMM on
  `hatemm_lora_train` (`train.json`, 743) — both configs verified; both byte-identical (shas below) to the data
  the banked generic comparators trained on. No cross-dataset mixing.
- **No gold in deployed path:** `generate_VideoMLLM_embedding_lora_HF.py:59-66` uses **fixed** label-free
  instructions (`IMG_INSTRUCTION`/`TEXT_INSTRUCTION`, "never sampled from the model; pure encoder use") applied
  identically to every video regardless of label. Gold never enters inference.
- **Adapter-generic extraction (no edit):** `:419` loads the base VLM; `:429-441` `PeftModel.from_pretrained` +
  `merge_and_unload` merges **whatever** LoRA modules the adapter contains (incl. `visual.*`) — no module-name
  special-casing. `gen_embed_lora.sbatch` takes the out-tag as **arg 3** (`Qwen2.5-VL-7B-Instruct-LoRA-vis_HF`,
  a DISTINCT tag) and B2-pushes **only** `data/CLIP_Embedding/<DS>` `.pt` caches — raw videos never leave.
- **No OCR channel.** **No external API.** LoRA weights stay on disk.
- **Collision safety re-verified (all ABSENT):** `logging/lora/{MHC_vis,HateMM_vis}`,
  `data/CLIP_Embedding/{MHC,HateMM}/*LoRA-vis*.pt`, `logging/Retrieval/{MHC,HateMM}/RAC_video_lora_vis*`,
  `slurm/logs/enc3s_*LoRA-vis*.trainlog`, and `logging/lora/_smoke_vis` — none exist ⇒ fresh SFT/extract/head,
  `force=False` never trips, no 12850/13235 arm overwritten.
- **Single-test-touch accounting incl. the EN early-kill branch:** the EN image-MOVED gate runs on train+dev
  **before** the head, so a FLAT/DEGRADED EN spends **zero** EN test-touch (EN head cancelled); a MOVED EN spends
  exactly one. HateMM always spends exactly one. ⇒ ≤ 1 EN + 1 HateMM new vis-encoder test evaluations, each a
  NEW single test-touch/dataset (F0.1). Clean.
- **F0.x honesty clauses present:** F0.1 test-not-virgin, F0.2 single-encoder-draw (band = head-seed variance,
  travels with any K-V2 pass), F0.3 D7-deferred, F0.4 structural ceiling, F0.5 veto compliance, F0.6
  clean-superset, F0.7 honest most-likely outcome (EN MOVES-but-FAILS / HateMM K-V2 TIE).

### 7. Deviations §11 (DEV-1..DEV-7) — all favorable / neutral / documented

- **DEV-1** (config = 3 lines not "2") — **neutral/clarifying**; the 3rd line is `output_dir` (non-collision),
  data pointer unchanged. Verified: diff = exactly those 3.
- **DEV-2** (arg-driven `DATASETS` vs hardcoded 6-row array) — **favorable**; makes the EN branch a submit-time
  argument, keeping the hash-freeze intact. `run_one` kept byte-identical (verified).
- **DEV-3** (dedicated `vis_image_moved_probe.py` vs the HateMM-hardcoded F58 script) — **favorable/neutral**;
  imports `encoder_swap_geometry.py` verbatim and pins F58's thresholds; validated (reproduces the FLAT delta).
- **DEV-4** (image-MOVED anchor operator corrected to the committed operator) — **favorable, LOUD**; the recon's
  0.659/0.695 came from an uncommitted scratch probe; re-derived with the committed operator = 0.6236/0.6756.
  Gate unchanged (same-operator delta, operator-independent threshold). Independently reproduced (§1).
- **DEV-5** (kill-bar set follows the task, F55 oracle carried as a prior damper not a hard gate) —
  **documented**; consistent with the task's binding kill-bar list.
- **DEV-6** (EN generic-adapter data = word `mhc_lora_train`, not `train_yn.json`) — **favorable/clarifying**;
  verified: `logging/lora/MHC/README.md` = "fine-tuned … on the mhc_lora_train dataset"; the vis EN config points
  to the same `mhc_lora_train → train.json` (549, sha `7fe4c654…`). ⇒ K-V2 is a clean superset on EN
  (answer-format is NOT a second manipulated variable).
- **DEV-7** (ZH = GO-IF clause only) — **documented**; no ZH artifact authored or submitted.

### 8. Cost + chain — **PASS**

Execution order enforces SFT → extract → **EN image-MOVED gate** → head (§6); Stage-2/Stage-3 chain via
`--dependency=afterok:`; the EN gate is a $0-CPU branch point before the EN head budget. Both new sbatch set
**NO `--time`**; `PENDING (JobHeldUser)` → **wait for auto-release, never force** is stated (§6). `lora_sft_vis.sbatch`
is a clean clone of `lora_sft.sbatch` (diff = job-name/header/CONFIG-OUTDIR mapping only; all commands identical),
sources `conda.sh` directly, sets the offline/CUDA-shim env, and carries the ≥20 G disk guard. New GPU budget
~9–12 A100-h; EN gate + all floor/anchor re-derivations are $0 CPU.

---

## NON-BLOCKING NOTES

1. **§3.3 K-V2 justification — "banked generic between-seed acc spread ≤ 0.014" is inaccurate for the EN leg.**
   This sentence is imported verbatim from `CAND2` (whose datasets were ZH+HateMM). For the datasets here it
   holds only on **HateMM** (val-sel spread 0.0140 = cand-2's cited max; final 0.0093) — on the **EN generic
   arm it does not**: val-sel per-seed accs {0.7516, 0.7391, 0.7888} span **0.0497** (3.5× the cited band) and
   final {0.7702, 0.7764, 0.7888} span **0.0186**. **Why non-blocking:** the between-seed-spread sentence is
   auxiliary *color* for why the +0.010 threshold is meaningful, **not** part of the decision rule; K-V2 is a
   **head-seed-paired** test (δ_s = vis_s − generic_s at matched seeds), whose real teeth are the **3/3-sign +
   mean Δacc ≥ +0.010 + mean ΔmF1 ≥ 0** conjunct — all decidable from raw logs, non-gameable, and identical to
   the cand-2 bar that was approved. The pairing removes the shared head-seed variance the "spread" sentence
   describes, and the F0.2 single-draw caveat is already pre-attached. **Recommendation for the verdict
   reviewer:** for any **EN** K-V2 pass, explicitly note that the EN generic between-seed spread
   (0.0497 val-sel / 0.0186 final) exceeds the §3.3-cited 0.014, so on EN the 3/3-sign requirement carries the
   full discriminative load and the F0.2 single-draw caveat should be reported prominently. (HateMM, the leg
   most likely to be reported, is within the band.)

2. **§1.5/§4.2 "run_one … 42 lines" descriptor.** My extraction of the `run_one()`…`PY` block is 41 lines and
   **byte-identical** to `enc3seed.sbatch` (`diff` empty). The line-count is an off-by-one descriptor slip
   (boundary counting); the load-bearing property — byte-identity to the banked-control head code — holds.
   Non-material.

3. **STEP 1 (`build_lora_sft_data.py`) builds all three splits, including a `test.json` ShareGPT file.** This is
   inherited verbatim from the generic `lora_sft.sbatch` parent and from the already-approved LoRA-HateMM /
   cand-2 chains. The built `test.json` does **not** enter SFT (the config trains on `mhc_lora_train` only) and
   produces **no held-out test metric** — it is a data-prep artifact identical to the banked generic
   comparator's, not a test-touch in the single-test-touch accounting (which counts RGCL-head held-out
   evaluations). Precedented and clean; noted for a 0-context reader.

---

## HASH-FREEZE (recorded in `VISION_UNFREEZE_FREEZE.md`; prereg NOT modified, per review mandate)

All values re-verified on disk at freeze time (prereg self-sha, artifacts A–E, reused machinery, SFT data):

```
FROZEN a2bb1c45b44de35fdfb959fdf07d1b1146546b0944f37fe95434e49e4ed2be2d  refine-logs/VISION_UNFREEZE_PREREG.md (commit c1592bb)
A 7d551460239aaf537ecbb62f4c77d859cfeea3403867ccb99b34d31eeeb7fd3f  mhc_qwen25vl_lora_vis_sft.yaml       (submodule a912747c)
B 634bd0bb02789a1728728be19efdf91b69b36aab27a5f1dd9eab229e3041700b  hatemm_qwen25vl_lora_vis_sft.yaml    (submodule a912747c)
C 3e895420e308b30d8371c54a7a03ab9cf033ebe4804143a511989e68f3ef7946  lora_sft_vis.sbatch
D ca7749149fd836bd84404cad8436fd868c51c1ff2930c3ed9e91657c6933e2fb  enc3seed_lora_vis.sbatch
E 719ab1fe837ad4c9f75c750b8e8e5d5853bd64cdcf3c526da35fe0177944c4a6  vis_image_moved_probe.py
```

Reused-unchanged machinery (re-verify at submit; do NOT edit):
`encoder_swap_geometry.py 974771775e15fd58c31bd07bfd26d6dac43eab304b5fd888235a8449009190f6`,
`gen_embed_lora.sbatch c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386`,
`enc3seed.sbatch dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d`,
fork sources `mhc_qwen25vl_lora_sft.yaml db371c18…`, `hatemm_qwen25vl_lora_sft.yaml d2f415cd…`.
SFT data: `MHC/train.json 7fe4c654…e1bba` (549), `MHC/val.json 575c84f2…76571` (80),
`HateMM/train.json 93c6d3d1…73973a` (743), `HateMM/val.json 9e103ed3…9cc9ef` (107) — HateMM train sha is
byte-identical to the pin in `LORA_HATEMM_PREREG.md` / `CAND2_CURRICULUM_PREREG.md`.

**Reviewer statements:** ZERO GPU/SLURM/Modal spent — CPU-only re-parse + hashing on the login node, plus one
CPU (`CUDA_VISIBLE_DEVICES=""`) kNN re-derivation of the §2.3 anchor over banked train/dev caches; no held-out
test metric produced; `state/` not touched; the prereg was NOT modified; no job submitted; not pushed.
