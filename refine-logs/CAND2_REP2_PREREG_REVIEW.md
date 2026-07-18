# CAND2_REP2_PREREG — Independent 0-Context Pre-Registration Review

**Reviewer:** independent 0-context pre-registration reviewer (no prior context; judged only what is committed on
disk). **Date:** 2026-07-18. **Mode:** read-only — no push, no job submission, no prereg edit.
**Target:** `refine-logs/CAND2_REP2_PREREG.md` at commit `2d15ffb` (HEAD).

## VERDICT: **APPROVED-WITH-NOTES**

The prereg pre-registers a DRAW-2 replication of the single live novelty-bearing positive (F56 HateMM val-sel
add-over-generic PASS). Every load-bearing claim was re-verified from disk and holds: hashes match, the seed-knob
is the genuine and only independent-draw lever, the head code is byte-identical to the banked controls, the bars
are decidable with no interpretive freedom, and I independently re-derived the banked draw-1 per-seed deltas
bit-for-bit from the raw trainlogs. Two minor NOTES (neither blocking); details below.

---

## Hash integrity — **PASS**

- **This file** `CAND2_REP2_PREREG.md`: sha256 `365511e91f56577df388266f13d5f8f5d963cf03fc6928be0fd9d576c54a2636`
  — **matches** the review target exactly; HEAD is `2d15ffbcff5ed36fb1559f2f6eadd8f23900bb38`. No mismatch ⇒ proceeded.
- **Artifact A** `hatemm_qwen25vl_lora_curric_sft_rep2.yaml` (submodule): `d645de31…506354c6` — **match**.
- **Artifact B** `lora_sft_curric_rep2.sbatch`: `265f3e73…c4763c1e` — **match**. `bash -n` = **SYNTAX_OK**.
- **Artifact C** `enc3seed_lora_curric_rep2.sbatch`: `a32fd3bb…5861baac` — **match**. `bash -n` = **SYNTAX_OK**.
- **Reused-unchanged** all match: `train_curric.json` `73307ef2…1c91082b`; draw-1 config C `c12c2b6b…2a70b6a4a3`;
  `gen_embed_lora.sbatch` `c76bb422…2f260a46f386`; `enc3seed.sbatch` anchor `dbe3fb81…0d4de0815c3d`; frozen mining
  cache `train_Qwen2.5-VL-7B-Instruct_HF.pt` `ba52bc0d…766a6c009`.
- **Collision targets ABSENT** (verified): `logging/lora/HateMM_curric_rep2`, `data/CLIP_Embedding/HateMM/*LoRA-curric-rep2*.pt`,
  `logging/Retrieval/HateMM/RAC_video_lora_curric_rep2*`, `slurm/logs/enc3s_*curric-rep2*.trainlog` — none exist ⇒
  no clobber of draw-1 / generic / frozen artifacts; `force=False` cannot trip an overwrite.

## The seed-knob claim (load-bearing) — **PASS** (1 factual NOTE)

- **(a) builder is deterministic/RNG-free — VERIFIED.** `src/utils/build_curriculum_sft_data.py` (sha `085384f5…`
  matches) declares a `SEED` constant at line 72 that is **never consumed** anywhere else (grep-confirmed);
  the softconf path is largest-remainder apportionment with a fully deterministic tiebreak
  `sorted(..., key=lambda i:(-frac[i], -w[i], i))`. A curriculum re-draw therefore cannot come from re-running the
  builder — it re-emits identical bytes (confirmed on-disk `train_curric.json` == frozen `73307ef2…`).
- **(b) parser.py:474 — VERIFIED verbatim.** Line 474 is `transformers.set_seed(training_args.seed)`, which seeds
  the global torch/numpy/random RNGs that govern LoRA-A kaiming init and the dataloader shuffle. So the SFT seed is
  the correct and only lever for an independent draw of a fixed curriculum multiset.
- **(c) draw-1 pinned NO seed ⇒ HF default 42 — VERIFIED.** The frozen draw-1 config has no `seed:` line (see (e));
  a live import under `HateVideo` (transformers 4.49.0) gives `Seq2SeqTrainingArguments.seed default = 42`. Draw-2
  pins explicit `seed: 1`, a genuinely distinct SFT draw.
- **(e) rep2-vs-draw-1 diff — VERIFIED by direct diff.** The ONLY differences are: `output_dir` →
  `…/HateMM_curric_rep2`, an added `seed: 1`, and a 6-line comment block documenting the seed. Every other line is
  byte-identical. Matches F-R0.3 exactly.
- **(d) NOTE — the "CLI override is ignored" justification is imprecise.** F-R0.3 states `read_args` "ignores extra
  CLI args." It does **not**: `parser.py:73-76` computes `OmegaConf.merge(dict_config, OmegaConf.from_cli(sys.argv[2:]))`,
  so a dotlist `seed=1` passed on the command line **would** override the yaml. What is actually true and
  sufficient: (i) an HF-style `--seed 1` flag is not dotlist-parseable by OmegaConf, and (ii) — decisively — the
  frozen sbatch B invokes `python src/train.py "${CONFIG}"` with **no extra args**, so no CLI override is ever in
  play, and the seed is baked into the **hash-frozen** yaml A. The chosen design (bake seed into the frozen yaml)
  is correct and unambiguous; only the stated rationale for *why* CLI wasn't used is loosely worded. **No effect on
  what runs.** Optional: soften the F-R0.3 wording at freeze.

## Same-code — **PASS**

- The `run_one … PY` block of Artifact C is **byte-identical (42 lines, empty diff)** to BOTH
  `enc3seed_lora_curric.sbatch` (draw-1) AND `enc3seed.sbatch` (anchor). Verified by direct `diff`.
- Full head-sbatch delta vs draw-1 = documentation comment + `LORA` model tag
  (`…-LoRA-curric-rep2_HF`) + `GROUP_NAME` (`RAC_video_lora_curric_rep2`) + dropped ZH `CONFIGS` rows (HateMM-only,
  per F-R0.5) + final echo string. The only run-affecting manipulated variables are `--model` and `--group_name`;
  consistent with the prereg.
- Extraction reuses `gen_embed_lora.sbatch` **unchanged** (sha match; `OUT_MODEL_TAG` is arg 3 at line 34;
  distinct `…-curric-rep2_HF` tag never clobbers frozen / generic / draw-1 caches).
- Dataset resolution verified: config `dataset: hatemm_lora_curric_train` → LF `dataset_info.json` →
  `/data/jehc223/RGCL/data/lora_sft/HateMM/train_curric.json` (the frozen `73307ef2…` multiset). SFT trains the
  frozen curriculum; the STEP-1b sha gate re-proves this at submit.

## Bars decidable + independent re-derivation of draw-1 deltas — **PASS**

I re-parsed the raw `13241` (draw-1 curric) and `13235` (generic-LoRA) trainlogs from scratch, applying the frozen
selection logic (warmup≥5, argmax by val-acc then val-F1). **Independently re-derived draw-1 per-seed deltas:**

| protocol | s0 Δacc | s1 Δacc | s2 Δacc | mean | sign | mean ΔmF1 |
|---|---|---|---|---|---|---|
| **val-sel** | **+0.0186** | **+0.0046** | **+0.0233** | **+0.0155** | **3/3** | +0.0165* |
| final-ep | +0.0140 | +0.0047 | +0.0093 | +0.0093 (<0.010 ⇒ TIE) | 3/3 | — |

These match the prereg §2 line-150/152 vectors exactly. Every banked §2 arm value (generic, curric, and the
erratum-corrected CLIP floor s0 val-sel `0.8279/0.8172`) reproduces bit-for-bit from the raw logs.

- **\*ΔmF1 micro-note (trivia, non-blocking):** the prereg/verdict-review state val-sel ΔmF1 `+0.0166`
  (diff of the pre-rounded means `0.8711−0.8545`); the mean-of-per-seed-diffs is `+0.01654 → +0.0165`. A
  rounding-order artifact of 0.0001, immaterial to every bar (the ΔmF1 bar is ≥ 0).
- **K-REP-1** is the frozen §3.4 K-C2-2 rule (`mean Δacc ≥ +0.010 AND sign 3/3 AND ΔmF1 ≥ 0`) applied verbatim to
  draw-2, **bound to val-sel only** (F-R0.7). Binding to the single protocol that F56 passed (rather than the
  original "≥1 protocol") makes replication **harder**, not easier — it forecloses protocol-shopping. Decidable.
- **K-REP-2** pooled arithmetic **re-derived and correct:** draw-1 sum `+0.0465` (3/3); pooled mean ≥ +0.010 ⇔
  draw-2 sum ≥ +0.0135 ⇔ draw-2 mean ≥ ~+0.0045; a wash (draw-2 sum ≈ 0) yields pooled mean `+0.00775 ≈ +0.0078`
  with ≤4/6 sign ⇒ **NOT hardened**. Matches §3.2.
- **KS-REP** `≤ −0.014` = the frozen KS-regression threshold (= HateMM val-sel largest head-seed spread 0.0140,
  §2.3). Retire is an explicit terminal branch banked as a strong negative.
- **Decision tree (§3.4)** renders four **terminal** verdicts (hardened / weakly-hardened / downgraded /
  ruled-draw-noise); the comparison arm is the **banked** generic 13235 ("NOT re-run", §2). **No branch loops back
  to "draw again"** — in-ceremony seed-shopping is structurally foreclosed. §7.3 forces the verdict word into a
  fixed template, so a retire cannot be buried.

## Smoke-skip — **SOUND (concur)** (1 minor NOTE)

- Precedent verified on disk: job **13236** smoke `COMPLETED` (`train_loss 0.23189724`, checkpoint written, no
  NaN/OOM — `CAND2_SUBMIT_RECORD.md` §1); **13237** (ZH curric SFT) and **13238** (HateMM curric SFT) both produced
  completed `all_results.json` (HateMM `eval_loss 0.1186`, `epoch 2.97` = 3 epochs; it is the adapter that yielded
  F56). The recipe is byte-identical to draw-1 minus the SFT seed + output_dir; a reseed only changes
  `set_seed(1)` vs `set_seed(42)` (which init / which data order) and **cannot** introduce NaN/shape/OOM
  pathologies, which are fixed by the recipe/schema/footprint. The live healthy-start gate (first SFT log line sane
  + STEP-1b sha re-verify) is retained. **The skip is sound; I concur.**
- **NOTE (minor):** §4b frames the STEP-1b `train_curric.json` sha re-verify as a mandatory executor gate, but
  sbatch B does not *automate* the comparison — line 68 only echoes a comment ("sha must == 73307ef2…") and runs
  the builder. Given the builder's proven RNG-free idempotence and the hash-frozen mining inputs the risk is low,
  but an in-script `sha256sum … | grep`-assert-and-`exit` would make the gate self-enforcing rather than
  executor-dependent. Optional hardening.

## Hard constraints — **PASS**

Single-dataset own-train (HateMM `train_curric.json`; ZH not re-run; no cross-dataset mixing) · no gold in the
deployed path (extraction uses fixed neutral instructions; gold enters only own-train SFT/curriculum construction,
unchanged from draw-1 clearance) · no OCR · videos stay local (operates on cached `.pt`; local SLURM, not Modal) ·
no `#SBATCH --time` directive in either sbatch (only the "NO --time" comment) · single test-touch = the three
draw-2 head reads = one budgeted rep2 evaluation, zero pre-verdict touch · D7 novelty + goal satisfaction
explicitly deferred to the USER (title, F-R0.4, §7.3).

## Adversarial — **PASS** (1 hardening NOTE)

- `seed: 1` is **pre-committed and single**, baked into hash-frozen yaml A; there is exactly **one** SFT-draw seed
  and no provision to try seed 2/3/… and cherry-pick. Head seeds 0/1/2 are the paired head-init/shuffle seeds that
  match the banked generic arm — legitimate pairing, not the SFT-draw lever.
- Retire/downgrade outcomes are explicit terminal branches, banked and template-forced — cannot be buried.
- **NOTE (hardening, non-blocking):** the prereg binds one draw-2 attempt and its decision tree never loops to a
  re-draw, so *in-ceremony* shopping is closed. It does not, however, state in words that a draw-2 FAIL/retire is
  **terminal for the auto-replication ceremony** — i.e., that "keep drawing fresh SFT seeds until one replicates"
  is forbidden and any further draw would require a fresh user-authorized prereg with multiplicity accounting. The
  heavy ceremony (new prereg + independent review + user rulings) and the pooled K-REP-2 read mitigate this at the
  meta level; adding one explicit sentence would fully close the garden-of-forking-paths. Recommend, do not require.

---

## Freeze block — values confirmed for §5.3

```
FROZEN 365511e91f56577df388266f13d5f8f5d963cf03fc6928be0fd9d576c54a2636  CAND2_REP2_PREREG.md
A d645de3197739075774b499f335675dad8cd77a3f03b7c6cdc811424506354c6  hatemm_qwen25vl_lora_curric_sft_rep2.yaml
B 265f3e736a0e3ae1202cc86bfef562a2e3d830c9d09487eeea9534ab4c763c1e  lora_sft_curric_rep2.sbatch
C a32fd3bbaaa7140d5d5ffdf1dff3d0df7e26e1fb1ba079c5395e11025861baac  enc3seed_lora_curric_rep2.sbatch
--- reused (must still match) ---
  73307ef2e286eddf4fbe12ef13bb3c750f9105d1291494779c7a3a181c91082b  train_curric.json (HateMM, draw-2 trains this)
  c12c2b6b340151e6c58ed39843aa2cf02a728c17a3296637cae41c2a70b6a4a3  hatemm_qwen25vl_lora_curric_sft.yaml (fork parent)
```

All A–C and reused shas match disk at review time. The `FROZEN` line above records this file's current sha (no
prereg edit was made by this review). At submit the executor must re-run `sha256sum` on A–C and this file and
verify STEP-1b re-emits `train_curric.json` bit-exact; any mismatch = authorization VOID.

## Summary of NOTES (all non-blocking)

1. F-R0.3's "CLI ignores extra args" wording is imprecise (OmegaConf actually *merges* CLI dotlist overrides); moot
   because the frozen sbatch passes no extra args and the seed is in the hash-frozen yaml. Optional wording fix.
2. §4b STEP-1b sha gate is not automated inside sbatch B (comment only). Optional in-script assert.
3. Consider one explicit sentence foreclosing serial re-draws until one replicates (meta-level multiplicity).
4. Trivia: draw-1 val-sel ΔmF1 is +0.0165 (mean-of-diffs) vs the stated +0.0166 (diff-of-means) — 0.0001
   rounding-order, immaterial to every bar.

None of these affect the design's validity, the bars' decidability, or the honesty of the retire path. **APPROVED-WITH-NOTES.**
