# ZHPROMPT — Chinese-Instruction Re-Extraction Pre-Registration — INDEPENDENT 0-CONTEXT REVIEW

**Reviewer:** independent 0-context pre-registration reviewer (no prior context on this probe).
**Date:** 2026-07-25 NZST.
**Object under review:** `refine-logs/ZHPROMPT_PREREG.md` (commit `546518a`, on-disk == committed).
**Supporting recon:** `refine-logs/ZHPROMPT_FORENSIC_RECON.md` (`47a4e30`).
**Mandate:** ZERO GPU / no SLURM / no test-touch / no `state/` mutation / no push. Every item re-verified
independently from primary sources (raw trainlogs, git, on-disk files, CPU import). No claim taken on the
prereg's word.

**RULING: `APPROVED-WITH-NOTES`.** All seven checklist items pass on independently re-derived evidence. Three
non-blocking advisory notes (N1–N3) recorded below; none requires a code change and none blocks the freeze.

---

## V1 — Floors re-derived from PRIMARY sources (PASS)

Independently re-parsed the six raw trainlogs with the `enc3seed_zh_b3` protocol (val-sel = epoch ≥ warmup 5
with max `Val_Retrieval` acc, roc tie-break; final = max epoch = 29) using my own parser (not the prereg's).
Results **bit-match the prereg §2.1/§2.2 to 4dp, every seed**:

**Arm-L floor — job 13150** (`enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`):
| seed | val-sel ep / acc/mF1 | final ep / acc/mF1 |
|---|---|---|
| 0 | 20 / 0.8322 / 0.8023 | 29 / 0.8456 / 0.8181 |
| 1 | 26 / 0.8255 / 0.7956 | 29 / 0.8389 / 0.8113 |
| 2 | 19 / 0.8389 / 0.8065 | 29 / 0.8523 / 0.8226 |
| **mean** | **0.83220 / 0.80147** | **0.84560 / 0.81733** |

**Arm-F floor — job 13115** (`enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_13115.trainlog`):
| seed | val-sel ep / acc/mF1 | final ep / acc/mF1 |
|---|---|---|
| 0 | 22 / 0.7919 / 0.7412 | 29 / 0.8188 / 0.7864 |
| 1 | 25 / 0.8121 / 0.7871 | 29 / 0.8054 / 0.7759 |
| 2 | 28 / 0.8054 / 0.7759 | 29 / 0.7852 / 0.7514 |
| **mean** | **0.80313 / 0.76807** | **0.80313 / 0.77123** |

Prereg means (Arm-L val-sel 0.8322/0.8015, final 0.8456/0.8173; Arm-F val-sel 0.8031/0.7681, final
0.8031/0.7712) all match my re-derivation at 4dp (0.80147→0.8015, 0.76807→0.7681, 0.77123→0.7712 are correct
round-to-4dp). Promote thresholds §2.3 re-checked by arithmetic: Arm-L {0.8622/0.8315 val-sel, 0.8756/0.8473
final}, Arm-F {0.8331/0.7981 val-sel, 0.8331/0.8012 final} — all = floor + 0.030 exactly.
**`0.8537` appears in the prereg ONLY twice (L227, L597), both explicit disavowals** ("NOT the ledger's 0.8537",
"used 13150 raw numbers, NOT 0.8537") — never used as a floor. PASS.

## V2 — Default == identity == KS-parity guard (PASS)

Read the git diff of both extractors vs pre-diff (`git show 546518a`). The edit is additive and identical in
both files: +5 argparse keys (`--img_instruction`=`IMG_INSTRUCTION`, `--text_instruction`=`TEXT_INSTRUCTION`,
`--title_label`="Title: ", `--transcript_label`="Transcript: ", `--none_placeholder`="(none)"), and
`process_split` switched from the module constants/literals to the args. The module constants (frozen L45-52,
LoRA L59-66) are byte-verbatim the recon §1 English strings and are now consumed **only** as argparse defaults
(grep: no other consumer). **CPU-verified myself** (HateVideo env):
- `py_compile` PASS on both.
- `parse_args_sys([])` on both: all 5 args equal the deployed constants/literals; the default-path assembled
  `text_prompt` is **byte-identical to the deployed literal across 4 title/transcript cases** (empty/present,
  incl. Chinese body) and `img_instruction==IMG_INSTRUCTION` — `text_assembly 4/4=True img=True` both extractors.
- Chinese-override `parse_args_sys([...])` assembles `…\n标题:(无)\n文字记录:某段中文描述` = recon §3 scaffold.

The no-override assembly path is provably byte-identical to the pre-diff code; the `_encode`/pooling/forward
math is untouched (diff confined to the arg block + the assembly expression). PASS.

## V3 — Chinese strings verbatim across prereg / sbatch / recon §3 (PASS)

Code-point-level comparison (UTF-8): sbatch `IMG_ZH` (L42) and `TEXT_ZH` (L43) are **byte-identical** to recon
§3 L92/L96 (`MATCH: True` both). Scaffold labels `标题:` / `文字记录:` / `(无)` (sbatch L44-46) are present
verbatim in recon §3 L98, and the prereg's own assembled examples (F0.7, §4.2, DEV-H) use the same fragments
with **ASCII colon U+003A** after 标题/文字记录 — consistent with the sbatch labels (also U+003A) and the
deployed English `Title: `/`Transcript: `. `(无)` uses ASCII parens U+0028/U+0029 mirroring `(none)`. Punctuation
audit of TEXT: full-width `。`U+3002 / `、`U+3001 with ASCII `,`U+002C — identical between sbatch and recon.
Faithful translation, no wording drift between the three sources. PASS (see N3 for an orthographic observation
that is constant within the Chinese arm and therefore not a confound).

## V4 — Binding-language coherence (PASS)

- **KS-parity** (§3.3): English-default re-extraction of BOTH extractors must reproduce the banked cache
  `img max|Δ|==0.0 AND text max|Δ|==0.0` (READOUT 13468 R0 precedent). Bit-exact, both extractors, HALT-on-fail.
- **KS-dead** (§3.3, per-arm): mean paired Δacc ≤ 0 on EITHER protocol ⇒ arm KILLED; secondary mean Δacc
  < +0.015 on BOTH protocols ⇒ also KILL. **Per-arm, arms independent, NO auto-defund** (F0.6 corrects L2's
  F67 auto-defund — a frozen null does not predict a LoRA null since the frozen model has no
  instruction-language SFT). Coherent, and the "either-protocol" gate is a valid strict screen (an arm ≤0 on
  one protocol can never clear the dual-protocol FORMAL bar).
- **FORMAL** (§3.2): +0.030 acc AND +0.030 mF1 conjunct, 3/3 seeds, BOTH protocols, per arm vs own floor.
  Judged independently, no protocol/metric-shopping. D7-DEAD framing consistent throughout (F0.3/§8).
- No promotion loophole: the secondary KS threshold only makes killing *easier* (never a false promote); the
  measured-not-promoted band (§8) is an explicit coherent third outcome, not a gap.
- **Protocol tie-break pinned to precedent:** "val-sel = max Val acc, roc tie-break, warmup≥5" matches the
  `enc3seed_zh_b3.sbatch` embedded parser exactly — independently confirmed because my own re-derivation with
  that rule reproduced both floors bit-exact (V1).
- **±0.014 noise band verified as a real house descriptor** (not fabricated): `CAND2_CURRICULUM_PREREG.md §2.3`
  derives it as the largest observed head-seed spread (HateMM val-sel 0.0140; ZH val-sel/final 0.0134);
  `NCA_PREREG.md §2.5` reuses it. See N2. PASS.

## V5 — Sbatch audit (PASS)

`scripts/slurm/zhprompt_extract_head.sbatch`: `--gres=gpu:a100:1`, `--cpus-per-task=8`, `--mem=64G` (peak
8CPU/64G/1GPU, within 16/128/2 and trivially clear of the never-2×16-CPU wedge); **no `--time`** (only a comment
noting intentional absence); `bash -n` = SYNTAX_OK. Collision-free: `-zhp` caches, `RAC_video_zhp*` group dir,
and `*zhp*.trainlog` all verified **ABSENT** on disk; LoRA adapter `logging/lora/MHC_zh/{adapter_config.json,
adapter_model.safetensors}` (2026-07-02) and both banked parity-target caches verified **PRESENT**. CONFIGS =
exactly **6 rows** (2 arms × 3 seeds). **`run_one` block (L131-172) is BYTE-IDENTICAL to
`enc3seed_zh_b3.sbatch` L42-83** — verified by `diff` (empty) and md5 (`d8bdaa90…` == `d8bdaa90…`). Single
submission (one sbatch, one job doing extract→heads). Extractor/smoke CLI flags all exist (`--EXP_FOLDER`
default `./data/CLIP_Embedding`, `--splits` default `train,val,test`→`train/dev_seen/test_seen` matching the
shape-sanity gate). PASS.

## V6 — Multiplicity / budget / test-touch (PASS)

ONE sbatch = ONE family = ONE multiplicity bite (§3.6). **6 budgeted test reads** = {Arm-F,Arm-L}×seed{0,1,2}
(F0.1); zero test-touch before the independent verdict; executor transcribes raw numbers, applies no gates.
Budget ~1.1 GPU-h (recon §6: job-12116 frozen ZH dual-stream = 0.54 GPU-h/arm; heads ~3 min) — capped and
plausible. §4.6 code-fix⇒re-freeze clause present and verbatim-ported from `NCA_PREREG.md §4.5/§5.3`. DEV items
adequate and cover the required surfaces: DEV-A (run_rac.py evaluates Test every epoch — selection uses Val
only), DEV-B (JobHeldUser → wait), DEV-C (disk_guard padding), DEV-E (LoRA adapter path `logging/lora/MHC_zh`,
`exit 2` if missing), plus DEV-D/F/G/H. PASS.

## V7 — Untouched-core verify (PASS)

`git diff 546518a~1 546518a` touches only 4 files (prereg, sbatch, 2 extractors). `src/run_rac.py`,
`src/model/loss.py`, `src/model/classifier.py`, `src/utils/retrieval.py` are **not in the prereg commit** and
`git status --porcelain` on all four = **clean** (working tree == committed). `run_rac.py` sha256
`b85eb72a…` matches prereg §5.2; `enc3seed_zh_b3.sbatch` sha256 `4379224…` matches §5.2. The head path sets
none of the additive NCA/head-recipe keys (`--loss triplet --hybrid_loss True`, no `--head_loss`/`--mixup`/
`--sam`) so behaviour on the flags-off path is identical to the floor-era runner — the same additive-gating
already reviewed and frozen for the NCA family (`NCA_PREREG_REVIEW.md`/`NCA_FREEZE.md`, pairing the same
13115/13150 floors on the same `run_rac.py b85eb72…`). PASS.

## Freeze-candidate shas re-computed on disk (all match §5.1/§5.3)

```
07df7c7135e115f41f12be5ee95a06d29fa7360255b0995c5503bf6d3c841aab  refine-logs/ZHPROMPT_PREREG.md   (self, unmodified)
1c83d4378678afc12c05ce60dfa9e00b810e5398f436a3f7d51151f8ca35dfa1  src/utils/generate_VideoMLLM_embedding_HF.py       (A)
8d9bfd43d0a8f63a021280ffb287cc14fc31853d35893e2fb83926193e6e4cf4  src/utils/generate_VideoMLLM_embedding_lora_HF.py  (B)
f69b1aeb44abb554945fd1aeb524c1f5460950702bfaa44910f3dd720807a113  scripts/slurm/zhprompt_extract_head.sbatch         (C)
```
A/B/C all match the prereg §5.1/§5.3 claimed shas exactly.

---

## Non-blocking notes (advisory)

- **N1 (head-path pairing guard — recommend making the optional check mandatory).** Floors 13115/13150 were
  produced 2026-07-14 by a pre-NCA `run_rac.py`/`loss.py`; the treatment `-zhp` heads run on the current
  `run_rac.py b85eb72…` / `loss.py 2ae7a73f…` (both have since gained additive code gated OFF on the triplet+
  hybrid path). The prereg relies on additive-gating (F0.8) — sound and precedent-blessed by the NCA freeze,
  and the surgical NCA `loss.py` fix lives inside `_manifold_mixup_bce` which the triplet path never enters.
  The prereg's direct runtime confirmation — a 1-seed no-flag head on the banked English LoRA cache
  bit-reproducing 13150 seed0 (§4.1c, READOUT R0 precedent) — is currently marked **Optional**. Recommend the
  executor run it as a **mandatory** pre-judge guard (≈20 s, $0 test-touch) to close the run_rac.py/loss.py
  drift confound directly rather than by inheritance. Not blocking.
- **N2 (secondary KS-dead threshold slightly generous-to-kill).** "mean Δacc < +0.015 on both protocols ⇒ KILL"
  sits marginally above the ±0.014 band max (0.0140), so a mean of +0.0141–0.0149 would be killed as
  within-noise. This is conservative (only ever makes killing easier, never a false promote) and house-cited;
  recorded for transparency, not a defect.
- **N3 (ASCII comma inside the Chinese TEXT string).** `TEXT_ZH` carries `,` U+002C (ASCII) rather than a
  full-width `,`. This is byte-faithful to recon §3 (the recon pinned it) and identical across all three
  sources, and it is constant within the Chinese arm, so it introduces **no** treatment-vs-floor confound (the
  manipulated variable remains English→Chinese language). Flagged only as an orthographic observation; the
  prereg §4.2 already documents the ASCII commas/colons as intentional.

**FREEZE PERFORMED** — see `refine-logs/ZHPROMPT_FREEZE.md`. Any post-freeze edit to A/B/C or the prereg voids
authorization (§4.6). ZERO GPU/SLURM/Modal spent (CPU-only: floor re-parse, py_compile, argparse-identity
import, byte/codepoint string compare, sha256, git). No `state/` or `research-wiki/` mutation. No job submitted.
Not pushed.
