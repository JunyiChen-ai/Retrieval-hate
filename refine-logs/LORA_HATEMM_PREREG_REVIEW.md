# Independent 0-Context Pre-Registration Review — `LORA_HATEMM_PREREG.md`

**Reviewer:** independent 0-context pre-registration reviewer (no prior context; adversarial mandate).
**Date:** 2026-07-17. **Zero user interaction; no job submitted; prereg NOT modified.**
**Target:** `refine-logs/LORA_HATEMM_PREREG.md` (commit `3ebd880`).
**Method:** every load-bearing number re-derived from primary artifacts on disk (raw 12850 trainlogs
re-parsed with an independently written parser; configs/scripts diffed byte-for-byte; hashes recomputed).
The prereg's and recon's numbers were treated as untrusted until independently reproduced.

## VERDICT: **APPROVED-WITH-NOTES** (all notes non-blocking)

The prereg is internally sound, hash-consistent, floor-faithful, same-code-paired, veto-compliant, and
its kill-switches + outcome table are fully decidable from raw logs by a 0-context verdict reviewer. The
four notes below are wording/optional-step caveats that do not affect decidability and cannot be used to
manufacture an unsupported pass. Cleared to freeze + single-submit.

---

## Hash integrity — **PASS**

- Prereg self-sha `sha256sum LORA_HATEMM_PREREG.md` = `da1759a4b18fd7689c9086d5343a0e6805dedd24b76c3f2933bdc83a1941cd4b` — **matches the expected freeze hash.**
- Freeze-block artifact shas (§6.1/§6.2) all match files on disk:
  - A `hatemm_qwen25vl_lora_sft.yaml` = `d2f415cd93fa6f7b439fd4b4573a536baf48ad42186dc8bd50f3fab20553e36a` ✓
  - B `lora_sft.sbatch` = `e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f` ✓
  - C `enc3seed_lora_hatemm.sbatch` = `19c76b177f7dc883a9e03524450ad2e6cb302cdd0a6704d69da68a62188a06fc` ✓
  - `gen_embed_lora.sbatch` = `c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386` ✓
  - `mhc_qwen25vl_lora_sft.yaml` (copy source) = `db371c18f306c5a3a00eeef8550964c3ddacf9e20400439324009ef2e69b1b52` ✓
  - `enc3seed.sbatch` (12850 same-code anchor) = `dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d` ✓
  - `enc3seed_zh_b3.sbatch` (B3 template) = `4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad` ✓
  - §1.1 data shas: `train.json` `93c6d3d1…`, `val.json` `9e103ed3…`, `test.json` `c12ad356…`, `dataset_info.json` `ebf14b47…` — all ✓
- Prereg, B, and C are git-tracked and committed at `3ebd880` with **no on-disk drift** (`git diff HEAD` empty).

## Floor provenance — **PASS (re-parsed independently; all match to 4dp)**

I wrote a standalone parser (val-sel = epoch ≥ warmup 5 with max `Val_Retrieval` acc, roc tie-break →
that epoch's `Test_Retrieval` macroF1/acc; final = max epoch) and ran it on the raw
`slurm/logs/enc3s_*_12850.trainlog` (+ reused `arcbase_MHC_Qwen…_1227{5,6}` for EN-Qwen s1/s2). **Every floor
in the prereg is reproduced exactly.**

HateMM frozen-CLIP (PRIMARY, KS-1 pairs vs this):
- val-sel s0 `0.8279/0.8172` · s1 `0.8279/0.8163` · s2 `0.8047/0.7920` → **mean 0.8202/0.8085** ✓
- final   s0 `0.8186/0.7997` · s1 `0.8047/0.7822` · s2 `0.8140/0.7988` → **mean 0.8124/0.7936** ✓

HateMM frozen-Qwen (SECONDARY, KS-2 pairs vs this):
- val-sel s0 `0.8698/0.8606` · s1 `0.8651/0.8586` · s2 `0.8837/0.8753` → **mean 0.8729/0.8648** ✓
- final   s0 `0.8605/0.8507` · s1 `0.8605/0.8514` · s2 `0.8837/0.8753` → **mean 0.8682/0.8591** ✓

MHC-EN frozen-CLIP (EN PRIMARY):
- val-sel s0 `0.7826/0.7113` · s1 `0.7329/0.6034` · s2 `0.7702/0.6997` → **mean 0.7619/0.6715** ✓
- final   s0 `0.7640/0.7145` · s1 `0.7826/0.7159` · s2 `0.7888/0.7303` → **mean 0.7785/0.7202** ✓

MHC-EN frozen-Qwen (EN SECONDARY, context):
- val-sel s0 `0.7888/0.7378` · s1 `0.7826/0.7283` · s2 `0.7702/0.6997` → **mean 0.7805/0.7219** ✓
- final   s0 `0.8012/0.7596` · s1 `0.7702/0.7203` · s2 `0.7826/0.7475` → **mean 0.7847/0.7425** ✓

The prereg §8.1/§8.2 outcome tables pre-fill the per-seed CLIP floors matched to seed index (paired-by-seed);
these pre-filled cells also all match my re-parse. The CLIP floor is the ERRATUM-corrected value
(`0.8279/0.8172`, commit `66012e9`), not the withdrawn `0.8732` — correct.

## Kill-switch well-formedness — **PASS**

- **KS-1 (primary conjunct):** mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND sign 3/3, judged
  **independently under each protocol** (val-sel AND final-ep). Fully decidable from raw logs: executor reads
  per-seed LoRA acc/F1, subtracts the frozen §2.1 per-seed CLIP floor, means the three paired deltas, counts
  signs. No protocol- or metric-shopping possible (both protocols reported; the write-up format §8.3 is fixed).
- **KS-2 (family-coherence honesty flag):** "LoRA < frozen-Qwen − 0.014" → pre-declared statement; **explicitly
  NOT a performance kill; explicitly does not change the KS-1 verdict.** Threshold (0.014 seed band) and
  non-kill status are unambiguous. See Note 2 for a benign under-specification.
- **KS-3 (P9-echo):** LoRA mean below the CLIP floor → bank as "encoder-level LoRA is ZH-specific too" negative.
  Decidable; a labeling rule subordinate to KS-1, not an independent kill.

## Same-code pairing — **PASS**

`diff` of the `run_one`-through-`PY` block (`enc3seed_lora_hatemm.sbatch` L48–89 vs `enc3seed.sbatch` L44–85)
returns **exit 0 — byte-identical.** The full-file diff differs ONLY in: header comments, the
`CLIP=`/`LORA=` breadcrumb comment lines, `GROUP_NAME`, and the `CONFIGS` rows — all non-load-bearing. The
load-bearing `python ./src/run_rac.py …` argv is identical; the only manipulated variables reaching the run
are `--model` (CLIP→LoRA) and `--group_name` (fresh) plus derived `--exp_comment` — exactly the
`--model` + inert-fields claim in §4.1(b). `group_name` inertness is independently corroborated by the 12850
node's bit-for-bit seed0 reproduction across `RAC_video` / `RAC_video_archive_seeds` groups.

## Config diff — **PASS**

`diff mhc_qwen25vl_lora_sft.yaml hatemm_qwen25vl_lora_sft.yaml` = **exactly 3 changed lines**: L18
`dataset: mhc_lora_train→hatemm_lora_train`, L19 `eval_dataset: mhc_lora_val→hatemm_lora_val`, L27
`output_dir: …/lora/MHC→…/lora/HateMM`. Recipe pins verified in the yaml: `stage: sft`, `finetuning_type: lora`,
`lora_rank: 16`, `lora_alpha: 32`, `lora_target: q,k,v,o,gate,up,down_proj`, `freeze_vision_tower: true`,
`freeze_multi_modal_projector: true`, `learning_rate: 1.0e-4`, `num_train_epochs: 3.0`, `lr_scheduler_type: cosine`,
`warmup_ratio: 0.05`, `per_device_train_batch_size: 1` × `gradient_accumulation_steps: 8` (eff-bs 8), `bf16: true`.
`lora_dropout` is omitted (LLaMA-Factory default) — and the "dropout 0.0" claim is confirmed against
`logging/lora/MHC/adapter_config.json` (`lora_dropout: 0.0`, `r: 16`, `lora_alpha: 32`). The dataset uses the
**word** variant `hatemm_lora_train`, distinct from the P9 `hatemm_lora_yn_*` `_yn` variant (which exists,
points to `*_yn.json`, and is correctly unused).

## Hard-constraint compliance — **PASS**

- **Single-dataset own-train-split:** Stage-1 trains on `data/lora_sft/HateMM/train.json` only — verified
  **743 rows** (297 hateful / 446 normal), val 107 (43/64), test 215 (86/129), matching §1.1. No cross-dataset
  mixing.
- **No gold annotations in the deployed path:** I scanned every train row. The user turn is a **constant
  instruction** ("Analyze this video using the 8 frames and the transcript below. Decide whether it is
  hateful/harmful … or normal…") + 8 `<image>` frame refs + the video's own ASR **transcript**. The label
  appears ONLY as the assistant target (`hateful`/`normal`) — i.e. the training-split supervision target
  itself, which is permitted. No gold spans/attributes leak into the prompt.
- **No OCR channel:** the prompt merely instructs the model to consider on-screen text it can see in frames;
  no separate OCR-extracted text input is injected. Identical template to the already-accepted MHC/MHC_zh
  encoder adapters (this yaml is a verbatim copy).
- **Raw videos stay local:** extraction reads `data/video/HateMM/` (present) locally; only derived `.pt`
  feature caches are B2-pushed. No raw video leaves the machine.
- **All GPU via SLURM, no `--time`:** three `sbatch` jobs, none set `--time`; `PENDING (JobHeldUser)` wait
  honored. `lora_sft.sbatch` carries a ≥20 G free-space disk guard (L40–45).

## Honesty clauses — **PASS (all present + binding)**

F0.1 test-not-virgin (declared; one budgeted LoRA test-eval per dataset) · F0.2 single-encoder-draw (±band =
head-seed variance, symmetric with the single-draw CLIP control) · F0.3 no-novelty-claim (D7 = user ruling;
performance clause only) · F0.4 image-inherited-vs-LoRA-driven framing (material to D7; quantified by KS-2) ·
F0.5 single-dataset veto compliance · F0.6 two-regime disambiguation (P9 decision-level ≠ this encoder-level;
non-isomorphism evidenced by opposite ZH sign). The bundled **B4-EN expected-FAIL** arm is carried with its
own EN CLIP floors (§2.2, §3.5) and a pre-declared FAIL prior — it cannot be spun into a headline (which
requires ≥2 datasets and the EN prior is FAIL). The "≥2 datasets under one lever" framing is confined to the
performance clause throughout (§9), with the modality-divergence caveat pre-attached.

## Decidability / anti-gaming (adversarial pass) — **PASS**

Every §8 cell is fillable from raw trainlogs; the verdict is rendered by an independent 0-context reviewer
(not the executor, who "applies NO gates/interpretation"); the frozen per-seed floors are banked and
re-verified here; val-selection is fixed by the byte-identical inline parser (no epoch cherry-picking); the
decision rule is a hard AND-conjunct judged per protocol (no protocol/metric shopping). The one irreducible
residual — an executor could in principle retrain the SFT and cherry-pick the best encoder draw — is
pre-declared as a single draw (F0.2), is symmetric with the control, and would leave multiple adapter dirs /
trainlogs visible to the verdict reviewer. No path lets a motivated executor claim a pass the evidence does
not support. Collision safety re-verified: `logging/lora/HateMM`, `data/CLIP_Embedding/HateMM/*LoRA*.pt`,
`logging/Retrieval/{HateMM,MHC}/RAC_video_lora_hm*`, and the LoRA trainlogs all **do not exist** ⇒ no clobber
(matches §4.3; `force=False`). EN LoRA adapter (`logging/lora/MHC`) + cache (`…/MHC/*LoRA*.pt`, all 3 splits)
present ⇒ bundled arm + head-smoke are genuinely head-only.

## Non-blocking notes

1. **§1.1 "git status shows dataset_info.json byte-unchanged vs HEAD"** is loose wording:
   `RA-HMD/LLAMA-FACTORY-Ver202512` is a **git submodule** and `data/lora_sft/HateMM/*.json` are untracked by
   the parent repo, so there is no parent-repo HEAD to diff against. Immaterial — the real freeze mechanism is
   the sha256 table (verified matching on disk) + the submit-time re-hash clause (§6.4), which is independent
   of git tracking. The idempotent-rebuild claim (snapshot→build→snapshot sha unchanged) is verifiable by
   re-hashing at submit.
2. **KS-2 metric scope** is not spelled out (does "LoRA < frozen-Qwen − 0.014" trigger on acc, F1, or both?).
   Immaterial: KS-2 is explicitly a non-kill honesty flag that does not change the KS-1 pass/fail. Recommend
   the verdict reviewer report both acc and F1 gaps and apply the flag if either falls below the band.
3. **Artifact A freeze** rests on sha256 rather than git (A lives inside the submodule). Same as Note 1; the
   re-hash-at-submit clause covers it.
4. **SFT smoke (§4.4.1)** does not spell out *how* `max_steps: 20` is injected (CLI override vs throwaway yaml).
   Non-blocking: the smoke is explicitly optional/skippable and must not write into `logging/lora/HateMM`.

## Favorable deviation (honestly flagged, confirmed)

The recon (`edeaedc`) recorded the `hatemm_lora_{train,val,test}` **registration as MISSING**; the prereg
§1.1 flags "loudly, favorable" that it is in fact **present** and points to the word-variant files. I confirm
the registration is present and correct ⇒ Stage 0 is a verified no-op confirm, not a build. This is an honest,
correctly-disclosed upgrade over the recon, not a discrepancy that weakens the prereg.

## Freeze values (for the orchestrator to apply at freeze time — I do not edit the prereg per mandate)

```
FROZEN da1759a4b18fd7689c9086d5343a0e6805dedd24b76c3f2933bdc83a1941cd4b   LORA_HATEMM_PREREG.md
A      d2f415cd93fa6f7b439fd4b4573a536baf48ad42186dc8bd50f3fab20553e36a   hatemm_qwen25vl_lora_sft.yaml
B      e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f   lora_sft.sbatch
C      19c76b177f7dc883a9e03524450ad2e6cb302cdd0a6704d69da68a62188a06fc   enc3seed_lora_hatemm.sbatch
```

**Reviewer statements:** ZERO GPU/SLURM/Modal spent (read-only re-parse + hashing on the login node only); no
held-out test metric produced; all floor numbers independently re-parsed from banked completed-run trainlogs;
the prereg was NOT modified; no job submitted; not pushed.
