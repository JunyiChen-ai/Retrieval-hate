# ZHPROMPT — Chinese-Instruction Re-Extraction Pre-Registration — ZH dual-stream, 2 arms (LoRA PRIMARY), one bite

**Author:** zhprompt prereg author (CPU-only; no GPU/SLURM/Modal spent; NO job submitted).
**Date:** 2026-07-25 NZST.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced; no job submitted.
**Implements:** `refine-logs/ZHPROMPT_FORENSIC_RECON.md` (commit `47a4e30`, the GO recon) — its verbatim deployed
English prompts (§1), faithful Chinese translations (§3), the LoRA-SFT-Chinese-instruction decisive finding
(§4: LoRA arm = PRIMARY, frozen null does NOT auto-defund the LoRA arm), measured cost (§6: 0.54 GPU-h per
dual-stream extraction; both arms + heads ≈ 1.1 GPU-h), the `-zhp` cache tag (collision-free, §6), and the
draft kill-switches (§7) — transcribed and re-verified below. Deviations from the recon are flagged **loudly**
in §11.
**House-style precedent:** `refine-logs/NCA_PREREG.md` (§2 re-derived floors, §3 binding KS/FORMAL clauses, §4
execution chain incl. codex gate + smoke + single-submit, §4.5 code-fix⇒re-freeze clause, F0 RNG/multiplicity
notes), `refine-logs/FRAME16_PREREG.md` (extraction+head pipeline, F0.x honesty clauses, re-derived
line-cited floors, freeze block, single-submit plan, outcome-table template), `research-wiki/experiments/
exp-encoder-3seed.md:73-85` (the enc3s protocol + decision rule verbatim).

## Title + claim scope (verbatim)

> This measurement tests **one axis the deployed extraction path never varied — the LANGUAGE of the injected
> instruction/scaffolding prompt** (recon §0.4/§1.B: B1 killed encoder-language swap, P8c killed
> summary-channel language, but the *extraction-instruction language on the deployed path* is virgin). It
> re-extracts the ZH **dual-stream** (img span=`prefix`, text span=`response`) with **faithful CHINESE
> translations** of the frozen English `IMG_INSTRUCTION`/`TEXT_INSTRUCTION` + scaffolding (标题/文字记录/(无)),
> retrains the deployed RGCL align-fusion head + top-20 kNN, **3-seed paired within head-seed**, dual-protocol
> (val-selected AND final-epoch), on **ZH (`MHC_zh`) trained ONLY on its own train split**. It has **2 arms
> under ONE multiplicity bite**: **Arm-L = LoRA-Qwen ZH Chinese-prompt (PRIMARY)** — because the deployed ZH
> LoRA adapter was **SFT-trained with a CHINESE instruction** yet floor 13150 feeds it the **ENGLISH**
> extraction prompt, so a Chinese-prompt extraction *removes a measured train/inference instruction-LANGUAGE
> mismatch* (recon §4), a previously-unexamined mechanism on the *actual deployed floor*; and **Arm-F =
> frozen-Qwen ZH Chinese-prompt** — the mechanism control (no instruction-language SFT, so a frozen null does
> **NOT** predict a LoRA null; **arms are independent, NO auto-defund**, correcting L2's F67 auto-defund per
> recon §4). It is a **PURE-PERFORMANCE + door-closer / reviewer-question bite, NOT an expected-+3 bet**: even a
> formal PASS is a performance/robustness row (prompt-language is a generic extraction knob, **D7-DEAD**), never
> a novelty contribution. The honest counter-pressures driving the LOW prior are kept in the open (§5, recon §5):
> the frozen encoder is native-bilingual (likely no-op); the **text readout span pools language-INDEPENDENT
> assistant-header tokens** (attenuates any text effect); the **chat-template English system prompt stays
> English in both arms** (scope limit — never a fully-Chinese frame); and ZH's real wall is **78-dev
> val-selection noise** (F45/F63/F66), not representation. All strings are **FROZEN at recon-pin** (verbatim
> translations, no wording edits; §4.4 verified byte-exact vs recon §3); this prereg decides the **performance
> clause only.**

The cells under test are the deployed RGCL `classifier_hateClipper`, `fusion_mode=align` (Hadamard `x=img⊙text`),
triplet+0.5·BCE, AdamW head over cached embeddings, 30 epochs, warmup 5, top-20 arithmetic signed-cosine kNN vote,
`--force False`, paired **3-seed within head-seed** vs each arm's banked SAME-ENCODER English-prompt floor,
dual-protocol. **The ONLY manipulated variable between a treatment arm and its floor is the injected prompt
LANGUAGE** (English→Chinese); the extractor's 5 new prompt args DEFAULT to the byte-identical English constants
(§1.2, default == identity == the KS-parity guard). **Any wording/structure edit of the Chinese strings beyond a
faithful translation, a system-prompt translation, a different pooling span, or a third arm is OUT of this prereg**
— it spends the family and re-costs a bite (§3.6).

---

## 0. Binding facts / honesty clauses (all present; pre-declared)

**F0.1 — Test is NOT virgin (declared).** ZH test was already read, under the identical `enc3s` protocol, by:
frozen-CLIP (13115 CLIP arm), **frozen-Qwen-English (job 13115 — this prereg's Arm-F floor)**, **generic-LoRA
(job 13150 / B3 — this prereg's Arm-L floor)**, curriculum-LoRA (13241), bidir-LoRA (13471), readout-grid (13468),
head-recipe (SAM/mod-dropout 13478), and the NCA family (13482). This prereg's re-extraction test reads are
**re-measurements under the identical protocol**, not first exposures. There are **6 budgeted arm×seed test
evaluations** = {Arm-F, Arm-L} × seed{0,1,2} (the 3 head-seed reads of each arm). **Zero test-touch before the
independent verdict.**

**F0.2 — Single-extraction "draw"; shared identically by treatment and floor, so it does NOT confound the
prompt-language delta (pre-declared, material).** Each arm's 3 head-seeds read ONE Chinese-prompt extraction;
its floor read ONE English-prompt extraction of the SAME encoder. The Qwen forward is deterministic given
(weights/adapter, sampled frame indices, max_pixels, prompt) — frame sampling is `np.linspace(0,N−1,8)` (no RNG),
`attn=sdpa`, `bf16`, `no_grad`, single forward (extractor `_encode`). For **Arm-L** the encoder is a **single**
SFT-draw LoRA adapter (`logging/lora/MHC_zh`, 2026-07-02) — but that **same** adapter produced the 13150 floor,
so the single-SFT-draw limitation is **shared identically** by Arm-L and its floor and **cannot** confound the
prompt-language delta: the ONLY thing that differs at a given seed is the injected prompt language. `--seed`
controls head-init + data-shuffle; pairing is per head-seed (arm seed s − floor seed s), `s∈{0,1,2}`.

**F0.3 — Novelty = D7-DEAD, say it plainly.** The injected-instruction LANGUAGE is an extraction knob (which
string to feed a fixed encoder+pool). **Novelty-nil / D7-DEAD:** even a formal PASS is a performance/robustness
row ("Chinese vs English extraction prompt on ZH"), same D7 class as frame-budget (F67) / head-recipe / readout
— **never** a novelty contribution. The *reviewer-question* value ("why English prompts for Chinese inputs?"
answered with a measured null-or-lift) is a strong paper *sentence*, not a novelty *mechanism*. Pure-performance +
door-closer.

**F0.4 — Single-dataset own-train-split VETO compliance (hard user veto; trivially clean).** No SFT, no data
build, no cross-dataset mixing: the extractors read only each ZH video + the (now Chinese) FIXED
instruction/scaffolding — **no labels enter the encoder path** (`gt["text"]`=Bilibili description, `title` always
"(none)", `label` never touches the prompt). The RGCL head trains on **`MHC_zh`'s own train split only** (identical
corpus to both floors). NO gold spans/attributes, **NO OCR channel** (user veto), no cross-seed ensemble; raw
videos never leave the machine (only derived `.pt` `-zhp` caches → B2). All standing vetoes cleared.

**F0.5 — Honest prior is LOW; four disclosed walls lower it, none raise it (pre-declared; recon §5/§8).** Recon
§8: **prior of clearing the full GOAL bar (≥+0.030/+0.030, 3/3, BOTH protocols vs 13150) ≈ 8–12%** on the PRIMARY
Arm-L — revised **up** from L2's 4–6% by the §4 SFT-language-mismatch finding, still below a "likely win".
Outcome sketch (recon §8): clean null → PV1 ~60–65% · one-protocol lift ~25% · both-protocol pass ~8–12%. Four
disclosed walls, none a ban:
- **(a) native-bilingual frozen encoder.** Qwen2.5-VL is a top native-Chinese model (2502.13923; mE5/MMTEB
  consensus = one English instruction is fine across document languages) ⇒ **Arm-F is a likely no-op** at the
  embedding level.
- **(b) readout-token language-independence (text arm).** `text_feats` pools the FIXED trailing
  `<|im_start|>assistant\n` header tokens (span=`response`); the Chinese instruction acts only through
  attention-mediated context — attenuates any text effect (recon §3/§5). The **img** arm pools the instruction
  tokens directly (span=`prefix`) so it has the larger surface, but img is the weaker vote (dev img acc
  ~0.74–0.78 vs text ~0.85–0.87).
- **(c) English system frame stays constant.** `apply_chat_template(add_generation_prompt=True)` injects a
  default ENGLISH system prompt `You are a helpful assistant` (confirmed present in the cached
  `chat_template.json`, both arms) — held constant (not a confound) but a **LIMIT**: the model is never in a
  fully-Chinese frame (§4 confound discipline).
- **(d) ZH's real wall = 78-dev val-selection noise, not representation** (F45/F63/F66): LoRA text AUC already
  0.925; oracle-union headroom 91–98% selection-locked. A representation lift can be eaten by the 78-item dev
  selection on the val-sel protocol — which is exactly the harder protocol the dual-protocol goal bar requires.

**F0.6 — The §4 SFT-language-mismatch finding: Arm-L is NOT a frozen null, and NO auto-defund (LOAD-BEARING;
recon §4).** The deployed ZH LoRA (`logging/lora/MHC_zh`) was **SFT-trained with a CHINESE instruction**
(`data/lora_sft/MHC_zh/train.json`, recon §4), yet floor 13150 extracts its embeddings with the **ENGLISH**
`IMG/TEXT_INSTRUCTION` ⇒ the deployed ZH floor already contains a train/inference instruction-**LANGUAGE**
mismatch. Chinese-prompt extraction on Arm-L *removes* that mismatch = the train/inference-consistent config, a
genuine un-tested mechanism on the *actual deployed floor*. **Consequence (correcting L2's F67 auto-defund):** a
frozen-arm null does **NOT** predict a LoRA-arm null (the frozen model has no instruction-language SFT) ⇒
**arms are evaluated INDEPENDENTLY, each vs its own floor; a KS-dead frozen arm does NOT kill Arm-L** (§3.3).
*Honest caveat (recon §4):* the SFT prompt differs from the deployed extraction prompt in TASK/STRUCTURE, not
only language — a faithful *translation* of the summarization prompt aligns language but not task, so the
alignment recovered is **PARTIAL** (using the SFT prompt itself would change the task = a different confounded
cell, OUT of scope).

**F0.7 — Default-arg == identity == the parity guard (pre-declared; CPU-verified this prereg).** The 5 new
extractor args (`--img_instruction`, `--text_instruction`, `--title_label`, `--transcript_label`,
`--none_placeholder`) DEFAULT to the byte-identical deployed English constants/literals (`IMG_INSTRUCTION`,
`TEXT_INSTRUCTION`, `"Title: "`, `"Transcript: "`, `"(none)"`). With no override, `process_split` assembles
`args.text_instruction + "\n" + args.title_label + (title or none) + "\n" + args.transcript_label + (transcript
or none)` = the **byte-identical** deployed literal `TEXT_INSTRUCTION + "\nTitle: " + (title or "(none)") +
"\nTranscript: " + (transcript or "(none)")`, and `args.img_instruction == IMG_INSTRUCTION`. **CPU-verified this
prereg** (§4.2): `py_compile` PASS on both extractors; a no-override `parse_args_sys([])` reproduces the deployed
literal byte-for-byte across 4 title/transcript cases (empty/present, incl. Chinese body); a Chinese-override
`parse_args_sys([...])` assembles exactly recon §3's `\n标题:(无) / \n文字记录:…`. So a no-flag re-extraction is
byte-identical to the banked cache ⇒ default == identity == the KS-parity guard (§4.1a, machinery-verified
bit-exact at smoke, §3.3-KS-parity).

**F0.8 — run_rac.py is UNTOUCHED by this prereg; the head path is the deployed floor path (pre-declared).**
`git status --porcelain src/run_rac.py src/model/loss.py src/model/classifier.py src/utils/retrieval.py` =
**CLEAN** (working tree == committed; §4.1). run_rac.py currently carries the already-landed NCA/head-recipe
**additive-gated** keys (`--head_loss` default `triplet`, `--mixup` default `False`, `--sam` default off, etc.),
but **the zhprompt head command sets NONE of them** ⇒ the flags-off path is **byte-identical** to the 13115/13150
floor runner (NCA `F0.7` additive-gating fact, already blessed by the NCA verdict). The head `run_one` block is
**BYTE-IDENTICAL** to `enc3seed_zh_b3.sbatch` (the runner that produced BOTH floors 13115/13150) — `diff` empty
(§4.2). The ONLY per-arm variables are `--model` (`…_HF-zhp` / `…-LoRA_HF-zhp`) and the derived
`--exp_comment "_${MODEL}"`.

---

## 1. Pipeline spec — fully pinned (extraction + head; ONE job; nothing left to interpretation)

**Stage 0 — none.** No SFT / no data build: the extractors consume `data/gt/MHC_zh/{train,val,test}.jsonl` +
`data/video/MHC_zh/All/<id>.mp4` directly (ZH split counts train 579 / val 78 / test 149 = 806, recon §2). No
`lora_sft` rebuild — the deployed ZH adapter `logging/lora/MHC_zh` (2026-07-02; `adapter_config.json` +
`adapter_model.safetensors` verified present, §4.3) is merged read-only.

### 1.1 The single stage layout — ONE sbatch: extract both arms → 6 head runs

- **Submit:** `sbatch scripts/slurm/zhprompt_extract_head.sbatch` (authored this prereg — artifact C, §5).
- **Job-chain shape (CHOSEN + documented, task req 2/5):** **ONE sbatch, 8 CPU / 64 G / 1 A100 throughout**,
  running Stage A (extract Arm-F + Arm-L) then Stage B (6 chained head runs) **sequentially in one process**.
  This is the `gen_embed_readout.sbatch` / `ncafam_family.sbatch` precedent (both datasets/arms in ONE job),
  **not** the frame16 two-job `--dependency=afterok` chain. Rationale: (i) the prereg budget is **ONE
  submission** (§6); (ii) an **8-CPU single job** trivially satisfies the "**never two concurrent 16-CPU jobs**"
  submit-time aggregate wedge rule (it is one job, and 8 < 16); (iii) a single job removes the inter-job
  `afterok` race and keeps extraction+heads atomic. Peak footprint = 8 CPU / 64 G / 1 GPU — well within the
  16 CPU / 128 G / 2 GPU cap.

**Stage A — Chinese-prompt dual-stream re-extraction (recon §3 strings, verbatim):**
- **Arm-F (frozen):** `generate_VideoMLLM_embedding_HF.py --dataset MHC_zh --out_model_tag
  Qwen2.5-VL-7B-Instruct_HF-zhp --num_frames 8 --img_instruction "$IMG_ZH" --text_instruction "$TEXT_ZH"
  --title_label "$TITLE_ZH" --transcript_label "$TRANSCRIPT_ZH" --none_placeholder "$NONE_ZH" --device cuda`.
- **Arm-L (LoRA, PRIMARY):** `generate_VideoMLLM_embedding_lora_HF.py --dataset MHC_zh --lora_dir
  logging/lora/MHC_zh --out_model_tag Qwen2.5-VL-7B-Instruct-LoRA_HF-zhp --num_frames 8` + the same 5 Chinese
  overrides + `--device cuda`.
- The Chinese strings are **HARDCODED** at the top of the sbatch (frame16/readout no-submit-time-typo discipline;
  single-quoted, no shell interpolation) — verbatim from recon §3 (§4.4 byte-exact check). `--num_frames 8` +
  default `max_pixels=360*420=151200` = the floor config (recon §1). Both extractions write **DISTINCT `-zhp`
  caches** and never touch the banked `…_HF.pt`/`…-LoRA_HF.pt` (§4.3 collision-checked).
- **Output:** `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF-zhp.pt` (Arm-F) +
  `…-LoRA_HF-zhp.pt` (Arm-L); loader contract `{ids,img_feats,text_feats,labels}`, Dv=Dt=3584.
- **Stage-A shape sanity (in-sbatch, $0 GPU):** before ANY head run, a python block asserts all 6 `-zhp` caches
  exist with `img_feats.shape==text_feats.shape==(N,3584)`, `len(ids)==N`, `N>0`; a fail exits `3` **before**
  the budgeted 6 head reads (protects the test-touch). Then B2-push the derived `-zhp` `.pt` (videos never leave).
- **Cost:** ~0.5 GPU-h/arm (recon §6: job-12116 frozen ZH dual-stream = 00:32:26 = 0.54 GPU-h; LoRA + adapter
  merge ~min more) ⇒ ~1.0 GPU-h extraction.

**Stage B — 6 RGCL align-fusion head + kNN runs (paired vs the banked English floors):**
- **What it runs:** 6 head-only runs on the `-zhp` caches (~20–30 s each; 13150 ran 3 seeds in 2m46s):
  {Arm-F `…_HF-zhp`, Arm-L `…-LoRA_HF-zhp`} × seed{0,1,2}, `--group_name RAC_video_zhp`, `--force False`.
- **BYTE-IDENTICAL same-code guarantee (verified §4.2):** the `run_one`…`PY` block of
  `zhprompt_extract_head.sbatch` is **BYTE-IDENTICAL** to `enc3seed_zh_b3.sbatch` (`diff` empty) — the runner
  that produced BOTH floors (13150 LoRA; 13115 frozen was the byte-identical B1 command per B3_IMPL_NOTES,
  differing only in `--model`/`--group_name`). Config verbatim: `--batch_size 64 --lr 0.0001 --epochs 30
  --topk 20 --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align --hard_negatives_loss True
  --no_hard_negatives 1 --metric cos --loss triplet --hybrid_loss True --warmup 5 --majority_voting arithmetic
  --no_pseudo_gold_positives 1 --lambda_seg 0 --seg_mode full --exp_comment "_${MODEL}" --Faiss_GPU False
  --force False`. Distinct `--model` per arm ⇒ distinct `exp_comment` ⇒ **distinct `logging/Retrieval/MHC_zh/
  RAC_video_zhp/<exp_name>/` dirs** (run_rac.py:1010-1054 embeds `exp_comment` in `exp_name`) ⇒ no collision under
  the single group; `--force False` never trips the hard-abort (run_rac.py:1059-1062).
- **Pairing:** Arm-L per head-seed (Arm-L seed s − **13150** seed s); Arm-F per head-seed (Arm-F seed s − **13115**
  seed s). `--seed` controls head-init + data-shuffle; the only difference vs the floor at a given seed is the
  feature cache (Chinese-prompt vs English-prompt, SAME encoder) — a clean single-variable (prompt-language) test.
- **Output:** `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF-zhp_seed{0,1,2}_<JID>.trainlog` (Arm-F) +
  `…-LoRA_HF-zhp_seed{0,1,2}_<JID>.trainlog` (Arm-L). Cost ~3 min total.

**Total NEW GPU: ~1.1 A100-h** (extraction ~1.0 h dominates; 6 heads ~0.05 h) — recon §6 ledger.

### 1.2 The patches (2 files; additive; default == identity)

Both extractors (`src/utils/generate_VideoMLLM_embedding_HF.py` frozen; `…_lora_HF.py` LoRA) receive the
**identical** additive edit:
1. **+5 argparse keys** (inserted after `--out_model_tag`), each `default=` the byte-identical deployed
   constant/literal: `--img_instruction`(=`IMG_INSTRUCTION`), `--text_instruction`(=`TEXT_INSTRUCTION`),
   `--title_label`(=`"Title: "`), `--transcript_label`(=`"Transcript: "`), `--none_placeholder`(=`"(none)"`).
2. **`process_split` uses the args instead of the module constants** for the img instruction and the text-prompt
   assembly. The module-level `IMG_INSTRUCTION`/`TEXT_INSTRUCTION` remain the **single source of truth** (the
   argparse defaults reference them). No other code path used the constants (grep-verified: only their definition
   + `process_split`). **Default == identity** (F0.7, §4.2). The `_encode`/pooling/forward math is **untouched**.

`run_rac.py`, `classifier.py`, `retrieval.py`, `loss.py` — **NO edit** (§4.1, F0.8; shas unchanged).

---

## 2. Comparison floors — INDEPENDENTLY RE-DERIVED from raw trainlogs (numeric-provenance discipline)

Every number below was independently re-parsed **this prereg** from the raw trainlogs with the EXACT
`enc3seed_zh_b3.sbatch` embedded parser (val-sel = epoch ≥ warmup 5 max `Val_Retrieval` acc, roc tie-break;
final = max epoch). The Arm-L (13150) means **bit-match** the recon §2 provenance-flagged table and
`NCA_PREREG.md §2.1` to 4dp — and are **NOT** the ledger's `0.8537` (a different ZH cell; recon §2 NUMERIC
FLAG, 0.8732-incident discipline).

### 2.1 Arm-L floor — job **13150** (`Qwen2.5-VL-7B-Instruct-LoRA_HF`, generic-LoRA / B3; PRIMARY; goal-relevant, marginal)

| seed | val-sel ep | val-sel acc/mF1 | final ep | final acc/mF1 |
|---|---|---|---|---|
| 0 | 20 | 0.8322 / 0.8023 | 29 | 0.8456 / 0.8181 |
| 1 | 26 | 0.8255 / 0.7956 | 29 | 0.8389 / 0.8113 |
| 2 | 19 | 0.8389 / 0.8065 | 29 | 0.8523 / 0.8226 |
| **mean** | | **0.8322 / 0.8015** | | **0.8456 / 0.8173** |

Files: `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`. Bit-matches recon §2 /
`B3_EXECUTION_RECORD.md` / `NCA_PREREG.md §2.1` / `CAND2_CURRICULUM_PREREG.md §2.1`. (Task req 1 quoted the val-sel
mF1 as `0.8014`; the re-derived 3-seed mean is `0.80147 → 0.8015`; used **0.8015** per numeric-provenance.)

### 2.2 Arm-F floor — job **13115** (`Qwen2.5-VL-7B-Instruct_HF`, frozen-Qwen-English ZH; the mechanism control's same-encoder anchor)

| seed | val-sel ep | val-sel acc/mF1 | final ep | final acc/mF1 |
|---|---|---|---|---|
| 0 | 22 | 0.7919 / 0.7412 | 29 | 0.8188 / 0.7864 |
| 1 | 25 | 0.8121 / 0.7871 | 29 | 0.8054 / 0.7759 |
| 2 | 28 | 0.8054 / 0.7759 | 29 | 0.7852 / 0.7514 |
| **mean** | | **0.8031 / 0.7681** | | **0.8031 / 0.7712** |

Files: `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_13115.trainlog`. This is the **frozen-English
ZH floor** the recon §7 names ("frozen-zh vs the frozen-English floor") — pairing Arm-F against it isolates the
**single** variable (prompt language) on the frozen encoder, rather than confounding encoder-identity with
language (see §11 DEV-2 for why Arm-F is paired vs 13115, not 13150).

### 2.3 Concrete promote thresholds (mean +0.030) + noise band

- **Arm-L (vs 13150):** val-sel mean acc ≥ **0.8622** AND mF1 ≥ **0.8315**; final mean acc ≥ **0.8756** AND
  mF1 ≥ **0.8473** (all with 3/3 per-seed positive).
- **Arm-F (vs 13115):** val-sel mean acc ≥ **0.8331** AND mF1 ≥ **0.7981**; final mean acc ≥ **0.8331** AND
  mF1 ≥ **0.8012** (3/3 positive).
- **Head-seed noise band (KS-dead secondary read, §3.3):** ±**0.014** — the established house ZH head-seed spread
  descriptor (`B3_PREREG_REVIEW.md`, `CAND2_CURRICULUM_PREREG.md §2.3`, `NCA_PREREG.md §2.3`). A 3-seed mean move
  `< +0.015` on both protocols sits inside this band.

---

## 3. Decision rule + kill-bars (paired, both protocols judged independently, 3/3 sign, per-arm, pre-declared)

### 3.1 Decision rule — verbatim from `exp-encoder-3seed.md:73-85` (treatment = arm; control = the arm's banked English-prompt floor)

> For each dataset × protocol: (1) per-seed paired difference δ = (treatment − control) for acc and macro-F1 at
> seeds 0/1/2; (2) 3-seed mean ± std + sign consistency (how many of 3 positive); (3) n=3 too small for a
> bootstrap — report the paired-t **as an effect-size descriptor only**, no significance claim; (4) **pass =
> mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive**; (5) headline claim
> requires pass on ≥ 2 datasets under a stated protocol; both protocols judged separately; verdict written
> exactly "final-epoch: pass/fail; val-selected: pass/fail".

Both protocols judged **independently** (no protocol-shopping, no metric-shopping). Judged **per arm**; control =
the arm's OWN banked English floor (Arm-L §2.1 / Arm-F §2.2). (Note: rule (5)'s ≥2-dataset headline is
structurally unreachable here — this is a **single-dataset** (ZH) bite; a formal pass is a ZH robustness row, not
a headline, §8.)

### 3.2 FORMAL promote bar (goal-facing; per arm)

House **+0.030 acc AND +0.030 mF1** conjunct, **3/3 seeds positive**, under **BOTH** protocols vs the arm's banked
floor (§2). Below the conjunct under a protocol → **NEGATIVE** on that protocol. **D7-DEAD (F0.3): even a formal
PASS is a performance/robustness row ("Chinese vs English extraction prompt on ZH"), NEVER a novelty win.**

### 3.3 KS-parity + KS-dead — the KILL bars (per arm; SIGN-based; NO auto-defund)

- **KS-parity (machinery guard, pre-science; task req 3).** BEFORE any Chinese-prompt judging, an **English-DEFAULT**
  re-extraction of ONE stream must reproduce the banked cache **BIT-EXACT**. **Specified exactly:** at smoke
  (§4.4), run each edited extractor with **no prompt overrides** (all 5 args at their English defaults) on a
  small sample to a **throwaway `--EXP_FOLDER`**, and compare `img_feats`/`text_feats` (first-N rows, matched id
  order) against the banked `…_Qwen2.5-VL-7B-Instruct_HF.pt` (frozen) and `…-LoRA_HF.pt` (LoRA). **Threshold =
  `img max|Δ| == 0.0 AND text max|Δ| == 0.0`** (bit-exact), the exact `READOUT` job-13468 R0 precedent for the
  LoRA stack (recon §2, "img/text max|Δ| = 0.0, all 3 splits"). This is REQUIRED in addition to the **code-level
  identity proof** already CPU-verified this prereg (F0.7/§4.2: `py_compile` PASS + no-override assembly
  byte-matches the deployed literal). **Fail ⇒ HALT (plumbing bug), not a result.**
- **KS-dead (per-arm screen kill; recon §7 "≤0 on EITHER protocol" gate).** A treatment arm's **3-seed mean paired
  Δacc ≤ 0 vs its own floor on EITHER protocol ⇒ that arm KILLED** (banked as the prompt-language null). **Secondary
  read:** mean paired Δacc `< +0.015` on **BOTH** protocols (inside the ±0.014 ZH seed-noise band, §2.3) ⇒ also
  KILL. **Per-arm, arms INDEPENDENT, NO auto-defund** (F0.6, correcting L2's F67 auto-defund): a KS-dead **frozen**
  arm does **NOT** kill the LoRA arm; each is judged only vs its own floor. State each killed arm explicitly at
  verdict time. (Rationale for the "either-protocol" gate vs frame16's "both-protocol" gate: the GOAL bar is
  dual-protocol, so an arm ≤0 on even one protocol can never clear FORMAL — the recon §7 pins the stricter screen;
  see §11 DEV-1.)

### 3.4 KS-regression note (per arm)

If arm − floor **mean Δacc ≤ −0.014** on a leg (below the full head-seed spread, §2.3), the Chinese prompt
**degraded** the stream → bank "Chinese extraction prompt hurts on ZH <arm>." A note within the KS-dead frame, not
a separate multiplicity bite.

### 3.5 Ban-collision closure (carried from recon; disclosed, NOT a ban)

- **Not B1** (encoder-language swap, killed): B1 swapped the *encoder model* language identity; this swaps the
  *injected instruction* language on a **fixed** encoder — a different object (recon §0.4). **Virgin.**
- **Not P8c** (summary-channel language, killed): P8c varied the *generated-summary* channel's language; this
  varies the *extraction-instruction* prompt on the deployed dual-stream path — a different channel (recon §0.4).
- **Not F70 / readout-grid** (readout prompt *structure*, KS-dead 2026-07-25): F70/readout varied the *pooling
  span / prompt structure*; this varies prompt *language* only (span/structure **unchanged** — same `prefix`/
  `response` pooling). Thematically adjacent (recon §5 "near a fresh grave") but a distinct axis. **Virgin.**
- **Not F45/F63/F66** (selection-lock / conversion-ceiling): those bound *inference-side* headroom over a fixed
  φ₀; a Chinese-prompt re-extraction produces a **different** φ (new Gram/oracle) — F66 is silent on it (as it is
  on any re-encode), it is legitimately un-measured, not F66-dead. F45/F63 are the honest counter-pressure (wall
  (d), F0.5), not a ban.
- **NOT** cross-seed ensemble / OCR / gold-in-method / cross-dataset mixing / external-API / target-as-structure —
  none reach an extraction-prompt-language swap. **Clear.**

### 3.6 Multiplicity + scope of THIS submit (pre-declared)

- **ONE sbatch = ONE pre-registered family = ONE multiplicity bite** whether one or both arms survive. The two
  arms **share** the single "ZH Chinese-instruction re-extraction" bite.
- **Strings FROZEN** (recon §3 verbatim translations; §4.4 byte-exact). **NO** post-hoc string tuning — a
  reworded/prompt-engineered Chinese variant, an "instruction-constants-only" (English-scaffold) variant, a
  **system-prompt** translation, a different pooling span, a Chinese-SFT-prompt (task-changing) arm, or a third
  encoder is a **new** pre-declared arm and re-costs a bite.
- **Verdict is per-arm.** A surviving arm still owes the **full ceremony** (this prereg → independent 0-context
  review → freeze-hash → SLURM); this prereg does **not** discharge that, and this is the ONLY ZH-prompt-language
  bite.

### 3.7 Gate order

G-repro (patched-file sha re-verify + run_rac.py/core untouched-verify + default==identity proof, §4.1) →
**codex review of the extractor prompt-plumbing + sbatch (§4.5)** → smoke incl. **KS-parity bit-exact** (§4.4) →
Stage-A shape sanity (§1.1) → single test-touch (the 6 head reads) → per arm: **KS-parity must have passed** →
KS-dead → FORMAL promote bar (both protocols). The verdict is rendered by an **independent 0-context reviewer
against this prereg VERBATIM**; the executor transcribes raw both-protocol per-seed numbers (line-numbered) and
applies NO gates/interpretation.

---

## 4. G-repro + KS-parity + smoke plan + collision safety + codex gate

### 4.1 G-repro discipline

- **(a) Patched-file sha gate.** At submit time re-run `sha256sum` on the two extractors + the sbatch (+ this
  file) — must match the §5 freeze block; any mismatch = authorization VOID. Re-verify `src/run_rac.py`,
  `src/model/loss.py`, `src/model/classifier.py`, `src/utils/retrieval.py` **git-clean / sha unchanged** (this
  prereg edits none of them; F0.8).
- **(b) Default==identity proof (F0.7).** A no-override extraction reproduces the banked cache byte-for-byte: the
  code-level proof (assembled prompt == deployed literal) is CPU-verified (§4.2); the **runtime bit-exact**
  confirmation is the KS-parity smoke (§4.4, `max|Δ|==0.0`).
- **(c) Head same-code (INCLUDING the floors).** The `run_one`…`PY` block of `zhprompt_extract_head.sbatch` is
  **BYTE-IDENTICAL** to `enc3seed_zh_b3.sbatch` (`diff` empty, §4.2) — the runner that produced BOTH English
  floors. The Namespace diff between a `-zhp` head run and its banked floor command must be `--model` +
  derived-inert (`exp_comment`/`group_name`/`output_path`) + the already-blessed inert NCA/head-recipe argparse
  defaults (F0.8) ONLY. Optional stronger check: a 1-seed **no-flag** head on the *banked English* LoRA cache
  bit-reproduces 13150 seed0 (READOUT R0 precedent).

### 4.2 CPU verification (run this prereg — PASS)

- `python -m py_compile` on both extractors = **PASS**.
- No-override `parse_args_sys([])` (both extractors): `img_instruction==IMG_INSTRUCTION`,
  `text_instruction==TEXT_INSTRUCTION`, `title_label=="Title: "`, `transcript_label=="Transcript: "`,
  `none_placeholder=="(none)"`; assembled `text_prompt` **byte-matches** the deployed literal across 4
  title/transcript cases (empty/present incl. Chinese body) — **PASS**.
- Chinese-override `parse_args_sys([...])` assembles exactly `\n标题:(无)` / `\n文字记录:<body>` (recon §3) —
  **PASS**.
- Chinese strings in the sbatch **byte-exact** vs recon §3 (L92 IMG / L96 TEXT / L98 labels), incl. the recon's
  ASCII commas in TEXT and ASCII colons in the labels — **PASS** (§4.4).
- `bash -n scripts/slurm/zhprompt_extract_head.sbatch` = **SYNTAX_OK**; CONFIGS word-split = **6 rows**
  (2 models × 3 seeds); `run_one` diff vs `enc3seed_zh_b3.sbatch` = **empty (byte-identical)** — **PASS**.

### 4.3 Collision safety (verified this prereg — ABSENT; re-check at submit)

- `data/CLIP_Embedding/MHC_zh/*_Qwen2.5-VL-7B-Instruct{_HF,-LoRA_HF}-zhp.pt` — do NOT exist ⇒ fresh extraction;
  the `-zhp` tag is distinct from every banked tag (`_HF`, `-LoRA_HF`, `-LoRA-curric_HF`, `-LoRA-bidir_HF`,
  `-ro_{L28,L24,ow_L28,ow_L24}`, `-32B`, `p3pool*`, `p8*`, `p9*`) ⇒ **cannot clobber** a banked cache.
- `logging/Retrieval/MHC_zh/RAC_video_zhp*` — do NOT exist ⇒ fresh group; `--force False` never trips the
  run_rac.py:1059-1062 hard-abort; distinct `--model`⇒`exp_comment` keeps the two arms' dirs distinct.
- `slurm/logs/*zhp*.trainlog` — do NOT exist ⇒ no trainlog collision (the `_${MODEL}_` tag separates the arms).
- `scripts/slurm/zhprompt_extract_head.sbatch`, `refine-logs/ZHPROMPT_PREREG.md` — created by this prereg (no prior).
- LoRA adapter `logging/lora/MHC_zh` — verified present (`adapter_config.json` + `adapter_model.safetensors`,
  2026-07-02). Banked floor caches/trainlogs (13115/13150) are **read-only inputs**; this family writes none.
- Smoke throwaways (`logging/_smoke_zhp/`, `_smoke_zhp` group) — deleted after smoke; must NOT persist into §4.3.

### 4.4 Smoke plan (executor runs BEFORE the real submit; leave no artifact that trips §4.3)

1. **KS-parity bit-exact (GPU, ~2 min each; the machinery guard, §3.3).** For **each** extractor, run with **no
   prompt overrides** on a small sample to a throwaway folder, e.g.
   `python src/utils/generate_VideoMLLM_embedding_HF.py --dataset MHC_zh --splits test --limit 8
   --out_model_tag _parityF --EXP_FOLDER logging/_smoke_zhp --device cuda` (and the `…_lora_HF.py` variant with
   `--lora_dir logging/lora/MHC_zh --out_model_tag _parityL`). Load each and compare `img_feats`/`text_feats`
   first-8 rows (matched id order) vs the banked `data/CLIP_Embedding/MHC_zh/test_seen_Qwen2.5-VL-7B-Instruct_HF.pt`
   (frozen) / `…-LoRA_HF.pt` (LoRA). **Assert `img max|Δ|==0.0 AND text max|Δ|==0.0`** (bit-exact). Then
   `rm -rf logging/_smoke_zhp`. **Fail ⇒ HALT.**
2. **Chinese-prompt sanity (GPU, ~2 min).** Run each extractor with the Chinese overrides on `--splits test
   --limit 8 --EXP_FOLDER logging/_smoke_zhp --out_model_tag _zhpF/_zhpL`; confirm it completes (no crash, no OOM),
   shapes `(8,3584)`, finite (no unexpected all-zero rows beyond the extractor's own zero-guard), and print one
   assembled `text_prompt` to confirm the Chinese instruction + `\n标题:` / `\n文字记录:` scaffold + `(无)`
   reached the tokenizer (no mojibake / byte-fallback blowup — Chinese is native to the Qwen tokenizer, recon §3).
   Then `rm -rf logging/_smoke_zhp`.
3. **CPU checks (already run this prereg, $0):** §4.2 — reference for the executor.

### 4.5 CODEX GATE (mandatory pre-submit — house `codex-code-review` pattern; recon §6 recommends it)

Before ANY SLURM submission, the executor **MUST** run a codex review (iterative loop until Claude + Codex agree),
focused on the identity guarantee (the parity guard) + the plumbing:
- **Default == identity:** the 5 new args default to the byte-identical English constants/literals; a no-override
  run's assembled `text_prompt`/`img` instruction is byte-for-byte the deployed literal (no stray whitespace,
  no reordered concatenation, `"\n"+label` reproducing `"\nTitle: "`/`"\nTranscript: "`).
- **No other consumer of the module constants:** `process_split` is the only site now reading the prompt (the
  constants survive ONLY as argparse defaults); `_encode`/pooling/forward untouched.
- **Chinese-override path:** the args flow into `process_split` correctly; the sbatch single-quoting passes the
  UTF-8 strings intact (no shell interpolation of `、,。()`); `--none_placeholder` substitutes for BOTH empty
  title and empty transcript.
- **Head plumbing:** `run_one` byte-identical to `enc3seed_zh_b3.sbatch`; distinct `--model`⇒`exp_comment`⇒dir;
  Stage-A shape sanity aborts before the head reads on any malformed `-zhp` cache.

**Blocking findings ⇒ fix the code + re-freeze the shas (§5) + re-run this gate** (§4.6). (Lower-complexity than
the NCA gate — no new numeric kernel, only string plumbing — but mandatory because default-args bit-exactness IS
the KS-parity guard.)

### 4.6 Code-fix ⇒ re-freeze clause (verbatim-ported from `NCA_PREREG.md §4.5/§5.3)

**If the codex gate (§4.5) or the KS-parity smoke (§4.4.1) forces a code fix, the affected artifact shas change
and the freeze block (§5.3) MUST be re-issued** (a new independent 0-context review is re-run against the amended
files before submit). No code edit lands silently post-freeze; the executor re-runs `sha256sum` at submit and any
mismatch = authorization VOID.

---

## 5. Artifacts authored this prereg + hash-freeze block

### 5.1 New / edited artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current) |
|---|---|---|---|
| P | `refine-logs/ZHPROMPT_PREREG.md` | **NEW** — this file | *(reviewer fills at freeze)* |
| A | `src/utils/generate_VideoMLLM_embedding_HF.py` | **EDITED (additive)** — +5 prompt args defaulting to the English constants/literals; `process_split` reads the args (default==identity) | `1c83d4378678afc12c05ce60dfa9e00b810e5398f436a3f7d51151f8ca35dfa1` |
| B | `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | **EDITED (additive)** — identical +5 prompt args + `process_split` change | `8d9bfd43d0a8f63a021280ffb287cc14fc31853d35893e2fb83926193e6e4cf4` |
| C | `scripts/slurm/zhprompt_extract_head.sbatch` | **NEW** — ONE job: Stage A (Arm-F + Arm-L Chinese-prompt extraction, verbatim recon §3 strings hardcoded) + Stage-A shape sanity; Stage B `run_one` BYTE-IDENTICAL to `enc3seed_zh_b3.sbatch`, `RAC_video_zhp`, 6 rows (2 arms × 3 seeds) | `f69b1aeb44abb554945fd1aeb524c1f5460950702bfaa44910f3dd720807a113` |

### 5.2 Reused-unchanged machinery (verify sha / git-clean at submit; do NOT edit)

| path | role | sha256 |
|---|---|---|
| `src/run_rac.py` | deployed head runner (NCA/head-recipe keys inert; NO zhprompt edit; git-clean) | `b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3` |
| `scripts/slurm/enc3seed_zh_b3.sbatch` | same-code anchor for §4.2 (produced floors 13150/13115) | `4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad` |
| `src/model/loss.py`, `src/model/classifier.py`, `src/utils/retrieval.py` | deployed head/loss/fusion/retrieval (NO edit; git-clean) | *(unchanged; git-clean verified §4.1)* |
| `logging/lora/MHC_zh` | deployed ZH LoRA adapter (Arm-L; merged read-only) | *(present; `adapter_config.json`+`adapter_model.safetensors`, 2026-07-02)* |
| `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt` | Arm-F parity target (bit-exact) + not clobbered | *(present; verified untouched)* |
| `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | Arm-L parity target (READOUT R0 bit-exact) + not clobbered | *(present; verified untouched)* |

### 5.3 Hash-freeze (to be filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file ZHPROMPT_PREREG.md, after review>
A 1c83d4378678afc12c05ce60dfa9e00b810e5398f436a3f7d51151f8ca35dfa1  src/utils/generate_VideoMLLM_embedding_HF.py
B 8d9bfd43d0a8f63a021280ffb287cc14fc31853d35893e2fb83926193e6e4cf4  src/utils/generate_VideoMLLM_embedding_lora_HF.py
C f69b1aeb44abb554945fd1aeb524c1f5460950702bfaa44910f3dd720807a113  scripts/slurm/zhprompt_extract_head.sbatch
```
Executor re-runs `sha256sum` on A/B/C (and this file) + confirms `run_rac.py b85eb72…` + loss/classifier/retrieval
git-clean at submit time; any mismatch = authorization VOID. **If the codex gate (§4.5) or KS-parity smoke forces
a code fix, A/B (and possibly C) shas change and the freeze block MUST be re-issued (§4.6).**

---

## 6. Single-submit / execution plan + resource plan

**Order (ONE SLURM job):**

1. Pre-submit: G-repro (§4.1) → **codex gate (§4.5)** → smoke incl. **KS-parity bit-exact** (§4.4). Only on
   all-clear:
2. `sbatch scripts/slurm/zhprompt_extract_head.sbatch` → Stage A (Arm-F + Arm-L Chinese-prompt extraction,
   ~1.0 GPU-h) → Stage-A shape sanity (aborts on malformed cache) → Stage B (6 head runs, ~3 min). Produces the
   6 `-zhp` caches + `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct{_HF,-LoRA_HF}-zhp_seed{0,1,2}_<JID>.trainlog`.

**Resource plan (STANDING INFRA RULE compliant):** the sbatch requests **`--cpus-per-task=8`, `--mem=64G`,
1×A100** (inherited; §1.1). Single 8-CPU job ⇒ peak footprint **8 CPU / 64 G / 1 GPU** — within the 16 CPU / 128 G
/ 2 GPU cap, and **NEVER two 16-CPU jobs in flight** (an 8-CPU single job trivially clears the 29 h-wedge
submit-time aggregate rule). `conda activate HateVideo`; `sbatch` with **NO `--time`**; initial `PENDING
(JobHeldUser)` = **WAIT for auto-release, never force** (CLAUDE.md). Sources `conda.sh` directly, runs the ≥20 G
`disk_guard.sh`, B2-pushes derived `-zhp` `.pt` + `logging` at the end (videos never leave — CLAUDE.md boundary).

**Test-touch:** the 6 head reads are the ONLY budgeted zhprompt test evaluations (2 arms × 3 seeds); zero
test-touch before the verdict. **The executor transcribes raw both-protocol per-seed numbers (line-numbered) and
applies NO gates/interpretation** — the verdict (KS-parity → KS-dead → FORMAL, per arm) is rendered by an
**independent 0-context reviewer against this prereg VERBATIM.**

**No job is submitted by this prereg author.** Submission happens only after the independent 0-context review +
hash-freeze (+ codex gate) run by the orchestrator/executor.

---

## 7. Outcome table template (filled ONLY from raw trainlogs at verdict time)

### 7.1 Per-arm table (fill from `enc3s_MHC_zh_<MODEL>_seed{0,1,2}_<JID>.trainlog`)

**Arm-L (LoRA Chinese-prompt, PRIMARY) vs floor 13150 (§2.1):**

| seed | protocol | Arm-L acc/F1 | floor acc/F1 | Δ(arm−floor) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | ___ | 0.8322/0.8023 | ___ |
| 1 | val-sel | ___ | 0.8255/0.7956 | ___ |
| 2 | val-sel | ___ | 0.8389/0.8065 | ___ |
| **mean** | **val-sel** | ___ | **0.8322/0.8015** | **___** |
| 0 | final-ep | ___ | 0.8456/0.8181 | ___ |
| 1 | final-ep | ___ | 0.8389/0.8113 | ___ |
| 2 | final-ep | ___ | 0.8523/0.8226 | ___ |
| **mean** | **final-ep** | ___ | **0.8456/0.8173** | **___** |

**Arm-F (frozen Chinese-prompt, control) vs floor 13115 (§2.2):**

| seed | protocol | Arm-F acc/F1 | floor acc/F1 | Δ(arm−floor) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | ___ | 0.7919/0.7412 | ___ |
| 1 | val-sel | ___ | 0.8121/0.7871 | ___ |
| 2 | val-sel | ___ | 0.8054/0.7759 | ___ |
| **mean** | **val-sel** | ___ | **0.8031/0.7681** | **___** |
| 0 | final-ep | ___ | 0.8188/0.7864 | ___ |
| 1 | final-ep | ___ | 0.8054/0.7759 | ___ |
| 2 | final-ep | ___ | 0.7852/0.7514 | ___ |
| **mean** | **final-ep** | ___ | **0.8031/0.7712** | **___** |

### 7.2 Fixed write-up format (per §3.1 rule 5 + the bars §3.2/§3.3)

```
KS-parity: <PASS bit-exact | HALT>.   (must PASS before any arm is judged)
Arm-L (LoRA, PRIMARY):  final-epoch: <pass/fail>; val-selected: <pass/fail>  [FORMAL §3.2]. KS-dead: <KILLED | survives>.
Arm-F (frozen, control): final-epoch: <pass/fail>; val-selected: <pass/fail>. KS-dead: <KILLED | survives>.
(+ KS-regression note if any mean Δacc ≤ −0.014; + MARGINAL note if a within-noise pass per B3 §2.2 precedent.)
```

---

## 8. What a PASS / FAIL means for the goal (D7 boundary — prompt-language is DEAD, not user-pending)

- **Both arms KS-dead (recon prior — the honest expected outcome, ~60–65% PV1):** the injected-prompt language
  carries no net vote signal on ZH ⇒ the extraction-instruction-language axis is **CLOSED** at ~1.1 GPU-h, and
  the live reviewer question ("why English prompts for Chinese inputs?") is answered with a **measured null**
  (PV1). Cleanest cheap outcome: a genuinely un-enumerated axis converted to a measured door-closer.
- **An arm survives KS but < FORMAL bar (recon ~25% one-protocol lift):** measured-not-promoted limbo (bank the
  weak positive; still D7-DEAD). Most plausibly Arm-L on ONE protocol (the SFT-mismatch mechanism, §4/F0.6), with
  the other protocol eaten by 78-dev selection noise (wall (d)).
- **An arm clears the FORMAL bar (≥+0.030/+0.030, 3/3, both protocols; recon ~8–12%, Arm-L):** a paper-worthy
  **robustness/ablation** row ("train/inference-consistent Chinese extraction prompt lifts the ZH LoRA cell"),
  strengthening the ZH story. **NOT a novelty win** (F0.3) and NOT a headline (single dataset, rule (5)). A
  surviving arm still owes the full ceremony (§3.6).

**Framing sentence (verbatim):** *this measurement tests one axis the deployed extraction path never varied —
the LANGUAGE of the injected instruction/scaffolding prompt — through the ZH dual-stream on the frozen (control)
and the deployed LoRA (PRIMARY, which removes a measured Chinese-SFT × English-extraction train/inference
mismatch) encoders, 3-seed paired dual-protocol vs each arm's banked English-prompt floor; arms are independent
with NO auto-defund; a pass is a performance/robustness row, NEVER a novelty win — prompt-language is D7-DEAD.*

---

## 9. Provenance index

- Recon (GO; deployed prompts, translations, SFT-mismatch finding, cost, `-zhp` tag, kill skeleton):
  `refine-logs/ZHPROMPT_FORENSIC_RECON.md` (`47a4e30`).
- Cell source: `refine-logs/LITSWEEP3_ZH_SPECIFIC.md` shortlist **C1** (`d4af64b`); virgin-axis args §0.4/§1.B.
- Deployed prompts (verbatim): `src/utils/generate_VideoMLLM_embedding_HF.py:45-52` (constants),
  `:351-355` (text-prompt assembly), `:254-323` (`_encode` span pooling); LoRA `…_lora_HF.py:59-66` / `:374-378`.
- Chinese strings (verbatim): `ZHPROMPT_FORENSIC_RECON.md §3` (L92 IMG / L96 TEXT / L98 scaffold).
- SFT-mismatch (Arm-L PRIMARY): `ZHPROMPT_FORENSIC_RECON.md §4`; `data/lora_sft/MHC_zh/train.json`;
  adapter `logging/lora/MHC_zh` (2026-07-02).
- Floors (re-derived §2): Arm-L `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`
  (`B3_EXECUTION_RECORD.md`, `NCA_PREREG.md §2.1`); Arm-F
  `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_13115.trainlog`.
- KS-parity precedent (bit-exact): READOUT job 13468 R0 re-extraction, `READOUT_SUBMIT_RECORD.md §5` (recon §2).
- Protocol + decision rule (verbatim): `research-wiki/experiments/exp-encoder-3seed.md:73-85`.
- Same-code anchor + head runner: `scripts/slurm/enc3seed_zh_b3.sbatch` (sha `4379224…`); output-path keying
  `src/run_rac.py:1010-1062`.
- Walls / counter-pressure: F45 (78-dev selection noise), F63/F66 (selection-lock), B1 (encoder-language, killed),
  P8c (summary-language, killed), F70/readout-grid (prompt-structure, KS-dead 2026-07-25).

**Required statements:** ZERO GPU/SLURM/Modal spent by this prereg author (only pure-CPU login-node floor
re-parsing, `py_compile`, argparse-identity/assembly checks, byte-exact string verification, and
collision/syntax/same-code verification, seconds; no held-out test metric produced). All floor numbers re-parsed
from banked completed-run trainlogs (numeric-provenance discipline; Arm-L bit-matches recon §2; used 13150 raw
numbers, NOT 0.8537). No `state/` mutated. No `research-wiki/` mutated. NO job submitted. Not pushed.

---

## 10. DEV items — foreseeable execution pitfalls (task req 6)

1. **DEV-A (run_rac.py evaluates test EVERY epoch — throwaway-cache / no-peek discipline).** The head logs
   `Test_Retrieval` every epoch; the val-sel protocol selects the epoch by **Val** only (parser §4.2), then reads
   Test at that epoch. Discipline: **selection uses ONLY Val**; the per-epoch Test lines are transcribed by the
   executor but never used to pick an epoch (identical to all banked floors). If any tie-break is ever needed, do
   it on a **throwaway** `_smoke_zhp` group, never on the banked `RAC_video_zhp` dirs.
2. **DEV-B (JobHeldUser wait).** Initial `PENDING (JobHeldUser)` is expected — **WAIT for auto-release, never
   force** (CLAUDE.md).
3. **DEV-C (disk_guard wall-time padding).** The sbatch runs `disk_guard.sh` at start (≥20 G reclaim, gated on
   verified B2 copies); it can add minutes and touch `slurm/logs/disk_guard.log` — expected, non-fatal (`|| true`).
4. **DEV-D (cache naming collisions).** The `-zhp` tag is collision-checked ABSENT (§4.3) and distinct from every
   banked tag; the two arms differ by base tag (`_HF-zhp` vs `-LoRA_HF-zhp`). Re-check at submit — if any `-zhp`
   cache exists, HALT (a prior partial run) rather than overwrite.
5. **DEV-E (LoRA adapter path correctness — Arm-L).** `--lora_dir logging/lora/MHC_zh` must point at the DEPLOYED
   ZH adapter (the one that produced 13150), not a checkpoint subdir; the sbatch asserts the dir exists (`exit 2`
   if not) and the extractor raises `FileNotFoundError` on a bad peft dir. The adapter is merged
   (`merge_and_unload`) read-only; verify `adapter_config.json`+`adapter_model.safetensors` present at submit.
6. **DEV-F (Stage-A must complete before Stage B; shape sanity gate).** In one sequential job, extraction
   precedes heads automatically; the in-sbatch Stage-A shape-sanity block (`exit 3` on any malformed `-zhp`
   cache) prevents 6 wasted head reads on a bad extraction.
7. **DEV-G (bit-exactness is GPU/library-stack dependent).** The KS-parity `max|Δ|==0.0` threshold relies on the
   deterministic bf16/sdpa forward proven by READOUT 13468 on THIS stack. If the smoke re-extraction is NOT
   bit-exact (e.g. a driver/transformers change), that is a **HALT** (plumbing/stack drift), not a result —
   re-establish parity before judging.
8. **DEV-H (empty title always "(none)"→"(无)").** ZH `title` is constant "(none)" every row (recon §1, F74); the
   `--none_placeholder "(无)"` therefore appears on every title line. This is intended (faithful scaffold
   translation), not a bug; the `Transcript`/`文字记录` body carries the Bilibili description (median ~106 chars).

---

## 11. DEVIATIONS FROM THE RECON — flagged loudly

1. **DEV-1 (KS-dead pinned as the recon §7 "≤0 on EITHER protocol" SIGN gate — stricter than frame16's
   "both-protocol" gate). MATERIAL / recon- + task-aligned.** The task (req 1) and recon §7 both pin **mean Δacc
   ≤ 0 on EITHER protocol ⇒ arm dead**; frame16's KS-16f-dead used "tie/regress on BOTH protocols." I pin the
   recon/task "either-protocol" screen (§3.3), justified because the GOAL bar is **dual-protocol** (an arm ≤0 on
   even one protocol can never clear FORMAL). Only the significance formalism is sign-based (house n=3
   no-bootstrap); the secondary within-noise read (`< +0.015` both protocols) is added from recon §7.
2. **DEV-2 (Arm-F is paired vs the frozen-English floor job 13115, NOT vs 13150). MATERIAL / recon-faithful,
   task-clarifying.** Task req 1's phrase "vs floor 13150" names the **PRIMARY (Arm-L)** comparison and supplies
   its numbers; recon §4/§7 makes the frozen arm the **mechanism control** paired against the **frozen-English
   floor**. Pairing Arm-F vs its **same-encoder** English floor (job 13115, re-derived §2.2) isolates the SINGLE
   variable (prompt language); pairing it vs 13150 (a LoRA floor) would confound encoder-identity with language
   and make the control meaningless. Arm-L stays vs 13150. Both arms thus test the identical single variable
   (prompt language) each against its own encoder's English baseline.
3. **DEV-3 (argparse-default plumbing — 5 args, default==identity — chosen over the clone-a-`_zhprompt`-script
   route). MATERIAL / recon-recommended.** Recon §6 offered (a) argparse-default args (default bit-exactness =
   the KS-parity guard) or (b) cloning each extractor to a `_zhprompt` variant. I take (a) per the recon's
   recommendation + task req 2: it makes the parity guard a code-level invariant (F0.7) and avoids duplicating
   ~450 lines twice. The 5 args (not just the 2 instruction args) parameterise the **scaffolding** too
   (`--title_label`/`--transcript_label`/`--none_placeholder`) so the injected side is single-language (recon §3
   "translate instruction + scaffolding + placeholder"); all 5 default to the byte-identical English literals.
4. **DEV-4 (ONE sbatch extract→heads, NOT frame16's two-job `afterok` chain). MATERIAL / budget-aligned.** Recon
   §6 lists both the readout single-job precedent and the frame16 two-job chain; task req 2 lets me pick the
   shape respecting never-2×16-CPU. I take the readout/NCA **single-job** shape (§1.1) — it is the ONE submission
   the budget (§6, task req 5) requires, and an 8-CPU single job trivially clears never-2×16-CPU.
5. **DEV-5 (single group `RAC_video_zhp` + distinct `exp_comment`, NOT recon §6's `RAC_video_zhp_lora`). Neutral /
   same-code-favorable.** Recon §6 sketched a fresh `RAC_video_zhp_lora` group. I use ONE group with the two arms
   separated by `--model`⇒`exp_comment` (run_rac.py:1010-1054 embeds it in `exp_name` ⇒ distinct dirs), which
   lets `run_one` stay **BYTE-IDENTICAL** to `enc3seed_zh_b3.sbatch` (the NCA `RAC_video_ncafam` single-group
   precedent). Collision-checked ABSENT (§4.3).
6. **DEV-6 (full dual-stream both arms in one bite, NOT L2's "text-only first"). Recon-aligned.** Recon §7
   recommends the dual-stream one-bite (the deployed vote is dual; the img `prefix` span pools the instruction
   tokens directly, the larger effect surface). I pin dual-stream both arms; marginal cost of img ≈ +0.25 GPU-h.
