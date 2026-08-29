# ZHPROMPT — Chinese-Instruction Re-Extraction · FORENSIC RECON

**Candidate:** batch-4 #2 (LITSWEEP3_ZH_SPECIFIC.md shortlist **C1**, commit d4af64b) — re-extract the
deployed ZH text (+img) stream with **Chinese** instruction prompts (faithful translations of the frozen
English ones), retrain the RGCL head, compare vs **floor 13150** (B3 LoRA-Qwen ZH cell), dual protocol.
**Executor:** zhprompt forensic-recon agent · **Date:** 2026-07-25 · **GPU spent: 0** (zero-GPU recon;
no SLURM submit, no test-touch, no `state/` mutation, no push).
**Virgin cell:** B1 killed encoder-language swap; P8c killed summary-channel language; extraction-instruction
language **on the deployed path was never varied** (LITSWEEP3 §0.4, §1.B).

---

## 1. DEPLOYED PROMPTS — pinned (verbatim, git-verified)

**Extractor for floor 13150's caches:** `src/utils/generate_VideoMLLM_embedding_lora_HF.py`
(loads frozen base Qwen2.5-VL-7B + merges the peft adapter `logging/lora/MHC_zh`).
The frozen-arm extractor `src/utils/generate_VideoMLLM_embedding_HF.py` is a **byte-identical superset** for
the prompt constants and pooling math.

- Both extractors last touched by commit **`ece6a3b` "adapt RGCL core to hateful video detection" (2026-07-02)**;
  `git status --porcelain` on both = **clean** (working tree == committed). No prompt drift.
- `IMG_INSTRUCTION` / `TEXT_INSTRUCTION` strings are **byte-identical** between the frozen and LoRA extractors
  (verified by diff — "IMG identical", "TEXT identical").
- The prompts are **module-level constants, NOT argparse args** (unlike `--num_frames` / `--out_model_tag`
  that frame16 could flip flag-only). Changing them requires a code path (see §3 plumbing).

**IMG stream** (`generate_VideoMLLM_embedding_HF.py:45-47`, LoRA `:59-61`), span=`prefix`
(mean over vision+instruction, i.e. every token up to the last `<|im_start|>` assistant header — the
instruction tokens ARE included in this pool):

```
Describe the people, symbols, gestures, and on-screen text in this video.
```

**TEXT stream** (`:48-52`, LoRA `:62-66`), span=`response` (mean over the trailing assistant-header tokens
`<|im_start|>assistant\n` — see §5 wall on language-independence):

```
You are analysing a short video for potentially hateful or offensive content. Considering the frames together with the provided title and transcript, summarise the targets, symbols, tone, and any harmful intent conveyed.
```

**Text-prompt assembly** (`generate_VideoMLLM_embedding_HF.py:351-355`) — English scaffolding labels appended
to `TEXT_INSTRUCTION`:

```
<TEXT_INSTRUCTION>
Title: (none)            # title is ALWAYS "(none)" for ZH — constant every row (LITSWEEP3 §0.3, F74)
Transcript: <gt["text"]>   # ZH deployed text = Bilibili description, median ~106 Chinese chars (LITSWEEP3 §0.1-0.2)
```

**Config baked into floor 13150 caches:** `num_frames=8`, `max_pixels=360*420=151200` (`:99`), bf16,
`torch.no_grad()`, sdpa, last-layer hidden states, L2-norm. Matches the "8f/151200px" the candidate cites.

---

## 2. FLOOR 13150 — provenance + raw numbers (compare-against target)

- **Job 13150** = B3 `enc3seed_zh_b3.sbatch`, LoRA-Qwen ZH treatment arm, group `RAC_video_b3_lora`,
  cache tag `Qwen2.5-VL-7B-Instruct-LoRA_HF` (3584-d dual stream). COMPLETED exit 0, elapsed 2m46s
  (source: `refine-logs/B3_EXECUTION_RECORD.md`, raw transcription).
- **RAW per-seed → means** (from B3_EXECUTION_RECORD, RAW only):

  | protocol | seed0 acc | seed1 acc | seed2 acc | **mean acc** | **mean mF1** |
  |---|---|---|---|---|---|
  | final-epoch (29) | 0.8456 | 0.8389 | 0.8523 | **0.84560** | **0.81733** |
  | val-selected     | 0.8322 | 0.8255 | 0.8389 | **0.83220** | **0.80147** |

- **NUMERIC-PROVENANCE FLAG:** the ledger's "ZH floor final-epoch **0.8537**" is a *different* ZH cell,
  NOT job 13150. The floor-vs-treatment comparison for this probe uses the **13150 LoRA raw numbers above**
  (final-epoch mean acc 0.8456 / mF1 0.8173; val-sel 0.8322 / 0.8014). Do not transcribe 0.8537 as the 13150
  floor (per the 0.8732-incident discipline in MEMORY).
- **Banked caches present** (`data/CLIP_Embedding/MHC_zh/`, distinct-suffix, untouched):
  `{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` (2026-07-02) and the frozen
  `{...}_Qwen2.5-VL-7B-Instruct_HF.pt` (2026-07-02). ZH split counts: **train 579 / val 78 / test 149 = 806**.

**KS-parity precedent is already proven for this exact stack.** READOUT job 13468 (2026-07-25) re-extracted the
ZH LoRA `R0` cache (same deployed adapter, same English spans, current pooling) and it reproduced the banked
deployed cache **BIT-EXACT** — `img max|Δ| = 0.0`, `text max|Δ| = 0.0`, id-order match, all 3 splits
(`READOUT_SUBMIT_RECORD.md §5`, "a fresh R0 head would reproduce 13150 seed-for-seed"). The frozen/LoRA forward
is deterministic on this GPU/library stack ⇒ an English-default re-extraction is bit-exact-verifiable as the
clobber-guard, and the head reproduces 13150 seed-for-seed.

---

## 3. PROPOSED CHINESE PROMPTS (faithful translation — LANGUAGE only, no prompt-engineering)

The cell tests **instruction LANGUAGE**; any wording/structure change beyond translation is a confound.
Proposed strings translate exactly the injected instruction + scaffolding, leaving gt content untouched:

**IMG (zh):**
```
描述这段视频中的人物、符号、手势以及画面上的文字。
```
**TEXT (zh):**
```
你正在分析一段可能包含仇恨或冒犯性内容的短视频。请结合视频画面以及所提供的标题和文字记录,概括其中的攻击对象、符号、语气,以及所传达的任何有害意图。
```
**Scaffolding labels (zh)** (`:353-354` equivalents): `"\n标题:" + (title or "(无)") + "\n文字记录:" + (transcript or "(无)")`
(`Transcript`→`文字记录` matches the literal English label AND the ZH-SFT usage; `Title`→`标题`; `(none)`→`(无)`).

**Design note (which surface to translate):** recommend translating **instruction + scaffolding + placeholder**
(everything WE inject) so the injected side is single-language — the clean "instruction-language" isolation.
A narrower "instruction-constants-only" variant (leaving `Title:/Transcript:/(none)` English) is defensible but
leaves an English↔Chinese mix in the injected scaffold; not recommended. The scaffold is a tiny span either way.

**Tokenizer / chat-template pitfalls (verified against the cached processor):**
- `apply_chat_template(..., add_generation_prompt=True)` **injects a default ENGLISH system prompt**
  `<|im_start|>system\nYou are a helpful assistant.<|im_end|>` whenever no system message is present (our
  `_build_messages` supplies only a user turn). This system frame stays English in BOTH arms — held constant
  (not a confound) but a LIMIT: the model is never fully in a Chinese frame. (Confirmed from
  `chat_template.json`: `"You are a helpful assistant" in template = True`.)
- The **TEXT readout span is language-INDEPENDENT tokens** — span=`response` pools the trailing
  `<|im_start|>assistant\n` header tokens, identical tokens regardless of instruction language; only their
  *contextualized* hidden states shift via attention over the Chinese body. Strong "likely no-op" mechanism for
  text_feats (§5).
- The **IMG readout span includes the instruction tokens directly** (span=`prefix` pools everything up to the
  header), so the Chinese IMG_INSTRUCTION tokens enter the img pool directly — a larger surface for any effect,
  though the img arm is the weaker vote (dev img acc ~0.74–0.78 vs text ~0.85–0.87, READOUT_SCREEN_OUT.json).
- No Qwen-specific Chinese-tokenization pitfall: Chinese is native to the Qwen tokenizer; no OOV, no byte-fallback
  blowup. READOUT smoke already confirmed "one-word tokenized" sanity on this stack.

---

## 4. DECISIVE-FINDING: the LoRA arm is NOT a frozen null (SFT-language mismatch)

**The ZH LoRA (`logging/lora/MHC_zh`, adapter 2026-07-02) was SFT-trained with a CHINESE instruction.**
First train record (`data/lora_sft/MHC_zh/train.json`):
> `请结合这8帧画面和下面的文字记录分析这段视频。判断它是仇恨/有害内容…请综合考虑画面、屏幕文字、手势、符号以及语音。文字记录：… 请只用一个词回答：仇恨 或 正常。`

But floor 13150's embedding extraction feeds that Chinese-SFT'd adapter the **ENGLISH** IMG/TEXT_INSTRUCTION.
⇒ **The deployed ZH floor already contains a train/inference instruction-LANGUAGE mismatch** (Chinese-SFT
adapter × English extraction prompt). Chinese-prompt extraction on the LoRA arm *removes* that mismatch =
the train/inference-consistent configuration — a genuine, previously-unexamined mechanism, distinct from the
frozen arm.

**Consequence for the L2 spend-rule:** L2's C1 sketch makes the frozen arm decisive ("frozen flat ⇒ LoRA
auto-defunded", F67 pattern). **That auto-defund MISFIRES here** — the frozen model has no instruction-language
SFT, so a frozen null does NOT predict a LoRA null. The LoRA arm must be given **independent standing**.
(Honest caveat: the SFT prompt differs from the deployed extraction prompt in TASK/STRUCTURE, not only
language — a faithful *translation* of the summarization prompt aligns language but not task, so the alignment
gained is PARTIAL. Using the SFT prompt itself would change the task = a different confounded cell.)

---

## 5. HONEST WALLS

- **Native-bilingual encoder (frozen arm).** Qwen2.5-VL is a top native-Chinese model (2502.13923, leads
  OCRBench-v2 Chinese +20.6%); mE5/E5-mistral/MMTEB field consensus = a single **English** instruction is fine
  across document languages. Frozen instruction-language swap is a likely **no-op** at the embedding level.
- **Readout-token language-independence (text arm).** text_feats pools the fixed assistant-header tokens; the
  Chinese instruction acts only through attention-mediated context (§3). Attenuates any effect.
- **System-prompt stays English** (§3) — the frame is never fully Chinese.
- **ZH's real wall = 78-dev val-selection noise, NOT representation** (LITSWEEP3 §0.5; F45/F63/F66): LoRA text
  AUC already 0.925; oracle union headroom +0.1026 is 91–98% selection-locked. A representation lift can be
  eaten by the 78-item dev selection on the val-sel protocol.
- **Goal bar is dual-protocol.** 13150 is already ONE protocol away (final-epoch +0.0313/+0.0453 3/3 PASS-marginal
  vs CLIP; val-sel +0.0246 FAIL). To *matter for the goal*, a gain must clear **+0.030/+0.030 3/3 on BOTH**
  protocols vs 13150 — the harder protocol is where selection noise bites.
- **Adjacency to dead axes:** readout prompt-*structure* variants dead (F70); readout-grid KS-dead fired
  2026-07-25 (all ZH cells ≤ +0.0128 dev). This is prompt-*language*, virgin, but thematically near a fresh grave.

---

## 6. COST + PLUMBING

**Direct timing precedent (not estimated):** frozen ZH dual-stream extraction (job **12116**, 806 items × 2
forwards, 8f/151200px, `Qwen2.5-VL-7B-Instruct_HF`) = **elapsed 00:32:26 = 0.54 GPU-h**. This is the *identical
operation* with different prompt strings.

| arm | forwards | GPU-h | note |
|---|---|---|---|
| frozen-zh re-extract (img+text) | 806×2 | **~0.5** | = job-12116 clone, prompts swapped |
| LoRA-zh re-extract (img+text)   | 806×2 | **~0.5** | + adapter merge (~min); PRIMARY arm (§4) |
| 3-seed heads (per arm)          | —     | ~3 min | run_rac cached-feature head; 13150 ran 3 seeds in 2m46s |
| **total (both arms, one job)**  |       | **~1.1 GPU-h** | single A100, 8 CPU / 64 G — well under 16CPU/128G/2GPU cap; 8-CPU job so the "never 2×16-CPU concurrent" wedge rule is not engaged |

L2's 0.3–0.6 GPU-h estimate = the single-arm figure; both arms ≈ 1.1 GPU-h. (Readout job 13468 ran 2 configs ×
4-forwards/item in 2h00m — consistent.)

**Extraction script + sbatch to clone:**
- Extractors: `generate_VideoMLLM_embedding_HF.py` (frozen) + `generate_VideoMLLM_embedding_lora_HF.py` (LoRA).
- sbatch precedents: `gen_embed_mllm.sbatch` (frozen) + `gen_embed_lora.sbatch` (LoRA) + `gen_embed_readout.sbatch`
  (the closest template — re-extracts ZH-LoRA with a distinct `-ro_*` suffix, hardcoded CONFIGS, never clobbers).
- Head sbatch: clone `enc3seed_zh_b3.sbatch` (produced 13150). It is `run_rac.py --model <TAG> --group_name <G>`;
  the Chinese arm only changes `--model …-zhp` + a fresh `--group_name` (e.g. `RAC_video_zhp_lora`), byte-identical
  otherwise ⇒ clean single-variable compare vs 13150.

**Prompt-override plumbing (the one non-flag change).** Because prompts are constants (§1), recommend the minimal
auditable diff: add optional `--img_instruction` / `--text_instruction` argparse args to both extractors,
**defaulting to the exact current English constants** (default run = provable identity; the R0 bit-exact guard
verifies it at runtime), then hardcode the Chinese strings in the new sbatch (frame16/readout no-submit-time-typo
discipline). Zero-touch alternative: clone each extractor to a `_zhprompt` variant with hardcoded Chinese
constants (frame16/readout used dedicated scripts) — safer for the frozen script but duplicates ~450 lines.
Recommend the argparse-default form + codex-code-review, since default-args bit-exactness IS the KS-parity guard.

**New cache paths (collision-checked: `-zhp` tag is FREE, no existing zhprompt/zhp cache):**
- frozen arm: `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF-zhp.pt`
- LoRA arm:   `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF-zhp.pt`
- English parity re-extract → diff against the banked `…_HF.pt` / `…-LoRA_HF.pt` (bit-exact per R0); no separate
  cache needed. `-zhp` is distinct from `-16f`, `-ro_*`, `-LoRA`, `-curric`, `-bidir` ⇒ cannot clobber a frozen cache.

---

## 7. KILL-SWITCHES (draft — reconciled with house style)

- **KS-parity** (machinery guard, pre-science): English-default re-extraction reproduces the banked cache
  **bit-exact** (`img/text max|Δ| == 0`, R0 precedent) AND a fresh English head reproduces 13150 within
  G-repro (seed-for-seed). Fail ⇒ HALT (plumbing bug), not a result.
- **KS-dead** (screen kill; house-style ≤0 gate, supersedes L2's ≤+0.015 sketch): treatment 3-seed **mean
  Δacc ≤ 0 vs its floor on EITHER protocol ⇒ KILL**. Secondary read: mean Δacc `< +0.015` on BOTH protocols =
  inside the ±0.014 ZH seed-noise band ⇒ also KILL. Bank as the PV1 prompt-language null.
  - **Per-arm, NOT auto-defund** (correcting L2 per §4): the LoRA arm is evaluated on its own vs 13150; a frozen
    null does NOT kill the LoRA arm.
- **FORMAL** (only if KS-dead not triggered): house bar **Δacc ≥ +0.030 AND ΔmF1 ≥ +0.030, 3/3 seeds, BOTH
  protocols**, LoRA-zh vs floor **13150** (frozen-zh vs the frozen-English floor).

**One-bite minimal-decisive set (recommendation):** ONE SLURM job, **both arms, full dual-stream (img+text)**.
Rationale vs L2's "text-only first": (a) the deployed vote is dual, so the faithful mirror of 13150 is dual;
(b) the img readout pools the instruction tokens directly (§3) — the img arm is where a language effect is most
detectable, so dropping it weakens the single bite; (c) marginal cost of adding img ≈ +0.25 GPU-h. The LoRA arm
is PRIMARY (§4 SFT-mismatch); the frozen arm is the mechanism control in the same job. Full-null on both cleanly
closes the axis; a lift on either advances to FORMAL.

---

## 8. VERDICT — **GO** (conditional, single cheap bite ~1.1 GPU-h)

**Prior of clearing the full GOAL bar (both protocols vs 13150): ~8–12%** — revised UP from L2's 4–6% by the §4
SFT-language-mismatch finding (the deployed LoRA floor is English-prompt-on-Chinese-SFT; Chinese-prompt extraction
is the train/inference-consistent config = a real un-tested mechanism, not the frozen native-bilingual null L2
priced). Still below a "likely win": the §5 walls (readout-token language-independence, English system frame,
78-dev selection lock, dual-protocol requirement) make the modal outcome a **paper-value null or a single-protocol
lift that misses the formal bar**. Outcome sketch: clean null → PV1 ~60–65% · one-protocol lift ~25% ·
both-protocol pass ~8–12%.

**Why GO not PARK:** (i) cost is genuinely ~1 GPU-h with a proven bit-exact parity guard; (ii) the LoRA arm has an
un-examined mechanism on the *actual deployed floor* (§4) that L2's frozen-centric prior missed; (iii) **guaranteed
paper value regardless of sign** — PV1 answers the live reviewer question "why English prompts for Chinese inputs?"
with a measured null (or a positive that would strengthen the whole ZH story). Discipline unchanged: prereg →
independent review → freeze-hash → single-submit; correct L2's auto-defund (evaluate the LoRA arm on its own).

**If the orchestrator prefers strict ≥10%-or-park:** the LoRA-only arm sits at the boundary (~8–12%); a defensible
narrower GO is **LoRA arm only** (drop the frozen control) at ~0.5 GPU-h, since §4 makes the frozen arm the less
informative one here.
