# VISION-UNFREEZE LoRA-SFT — INDEPENDENT 0-CONTEXT VERDICT REVIEW

**Reviewer role:** independent 0-context verdict reviewer. No prior project context; trusts ONLY primary
artifacts. Renders the binding verdict strictly against the frozen pre-registration
`refine-logs/VISION_UNFREEZE_PREREG.md` VERBATIM. Zero user interaction. CPU-only (no GPU/SLURM/Modal).
Modified nothing except this file; `autoresearch/goal_mllm_plus3/state/` untouched; nothing pushed.
**Out of scope (USER's rulings, per prereg F0.3/§8):** the D7 encoder-class novelty boundary and goal-level
satisfaction — this review decides the **PERFORMANCE clause only.**
**Date:** 2026-07-21 NZST.

---

## 0. Hash-freeze verification (done FIRST)

```
on-disk sha256(refine-logs/VISION_UNFREEZE_PREREG.md)
  = a2bb1c45b44de35fdfb959fdf07d1b1146546b0944f37fe95434e49e4ed2be2d
expected (task + VISION_UNFREEZE_FREEZE.md frozen block)
  = a2bb1c45b44de35fdfb959fdf07d1b1146546b0944f37fe95434e49e4ed2be2d
```
**MATCH.** The prereg on disk is the frozen binding text. Proceeding.

**Frozen artifacts + machinery re-verified on disk at verdict time (no drift since submit):**
```
C 3e895420e308b30d8371c54a7a03ab9cf033ebe4804143a511989e68f3ef7946  scripts/slurm/lora_sft_vis.sbatch       [MATCH]
D ca7749149fd836bd84404cad8436fd868c51c1ff2930c3ed9e91657c6933e2fb  scripts/slurm/enc3seed_lora_vis.sbatch  [MATCH]
E 719ab1fe837ad4c9f75c750b8e8e5d5853bd64cdcf3c526da35fe0177944c4a6  scripts/analysis/vis_image_moved_probe.py [MATCH]
  974771775e15fd58c31bd07bfd26d6dac43eab304b5fd888235a8449009190f6  scripts/analysis/encoder_swap_geometry.py [MATCH]
  dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch (same-code)  [MATCH]
```
Configs A/B live in the submodule; the submit record re-verified `A 7d551460…` / `B 634bd0bb…` MATCH at submit.

**Measurement provenance (raw logs only, job IDs per VISION_UNFREEZE_SUBMIT_RECORD.md):** vis-LoRA head reads =
job **13330**, six trainlogs `slurm/logs/enc3s_{HateMM,MHC}_Qwen2.5-VL-7B-Instruct-LoRA-vis_HF_seed{0,1,2}_13330.trainlog`
(chain: SFT EN 13301 → extract 13302 → EN gate → SFT HateMM **13328** [after the 13303 wedge/cancel, §4] →
extract 13329 → head 13330). Comparison arms re-parsed from raw: HateMM frozen-CLIP **12850**, HateMM
generic-LoRA **13235**, HateMM frozen-Qwen **12850**; MHC frozen-CLIP **12850**, MHC generic-LoRA **13235**,
MHC frozen-Qwen (s0 **12850**, s1 **12275**, s2 **12276**). Every number below re-derived with the
**byte-identical `enc3seed.sbatch` readout parser** (val-sel = epoch ≥ warmup 5 maximising `(Val_Retrieval acc,
roc)`, report that epoch's TEST metrics; final = max-epoch TEST metrics). Parser faithfulness hand-verified
against raw lines (e.g. HateMM-vis s0 val-sel `Test_Retrieval Epoch 12 macroF1: 0.8469 … acc: 0.8558`, final
`Epoch 29 macroF1: 0.8706 … acc: 0.8791`; MHC-vis s1 val-sel `Epoch 9 macroF1: 0.7342 … acc: 0.7888`; MHC-vis s0
val-selection confirmed ep26 = the max-val-acc warmup epoch, val acc 0.8000).

### 0.1 Comparison arms — re-derived vs the prereg's §2.1/§2.2 pinned tables (numeric-provenance discipline)

Every floor/generic mean below matches the prereg's §2.1/§2.2 to **4dp** (independent re-parse):

| arm | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | mean acc/F1 | §2 match |
|---|---|---|---|---|---|---|
| HateMM frozen-CLIP (12850) | val-sel | 0.8279/0.8172 | 0.8279/0.8163 | 0.8047/0.7920 | 0.8202/0.8085 | ✔ |
| HateMM frozen-CLIP (12850) | final-ep | 0.8186/0.7997 | 0.8047/0.7822 | 0.8140/0.7988 | 0.8124/0.7936 | ✔ |
| HateMM frozen-Qwen (12850) | val-sel | 0.8698/0.8606 | 0.8651/0.8586 | 0.8837/0.8753 | 0.8729/0.8648 | ✔ |
| HateMM frozen-Qwen (12850) | final-ep | 0.8605/0.8507 | 0.8605/0.8514 | 0.8837/0.8753 | 0.8682/0.8591 | ✔ |
| HateMM generic-LoRA (13235) | val-sel | 0.8605/0.8521 | 0.8698/0.8620 | 0.8558/0.8495 | 0.8620/0.8545 | ✔ |
| HateMM generic-LoRA (13235) | final-ep | 0.8651/0.8580 | 0.8744/0.8660 | 0.8698/0.8613 | 0.8698/0.8618 | ✔ |
| MHC frozen-CLIP (12850) | val-sel | 0.7826/0.7113 | 0.7329/0.6034 | 0.7702/0.6997 | 0.7619/0.6715 | ✔ |
| MHC frozen-CLIP (12850) | final-ep | 0.7640/0.7145 | 0.7826/0.7159 | 0.7888/0.7303 | 0.7785/0.7202 | ✔ |
| MHC frozen-Qwen (12850/12275/12276) | val-sel | 0.7888/0.7378 | 0.7826/0.7283 | 0.7702/0.6997 | 0.7805/0.7219 | ✔ |
| MHC frozen-Qwen (12850/12275/12276) | final-ep | 0.8012/0.7596 | 0.7702/0.7203 | 0.7826/0.7475 | 0.7847/0.7425 | ✔ |
| MHC generic-LoRA (13235) | val-sel | 0.7516/0.6916 | 0.7391/0.6920 | 0.7888/0.7506 | 0.7598/0.7114 | ✔ |
| MHC generic-LoRA (13235) | final-ep | 0.7702/0.7302 | 0.7764/0.7360 | 0.7888/0.7506 | 0.7785/0.7389 | ✔ |

The prereg's floor/generic transcriptions are trustworthy; all comparisons below use these re-derived numbers.

---

## 1. Vis-LoRA arm — raw measured numbers (job 13330, re-parsed + spot-checked vs raw lines)

| arm | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | mean acc/F1 |
|---|---|---|---|---|---|
| **HateMM-vis** (13330) | val-sel | 0.8558/0.8469 (ep12) | 0.8698/0.8620 (ep22) | 0.8558/0.8453 (ep22) | **0.8605/0.8514** |
| **HateMM-vis** (13330) | final-ep | 0.8791/0.8706 (ep29) | 0.8744/0.8653 (ep29) | 0.8558/0.8461 (ep29) | **0.8698/0.8607** |
| **MHC-EN-vis** (13330) | val-sel | 0.7888/0.7561 (ep26) | 0.7888/0.7342 (ep9) | 0.7826/0.7448 (ep24) | **0.7867/0.7450** |
| **MHC-EN-vis** (13330) | final-ep | 0.7826/0.7448 (ep29) | 0.7702/0.7302 (ep29) | 0.7640/0.7274 (ep29) | **0.7723/0.7341** |

---

## 2. Outcome tables — filled cell-by-cell (prereg §7.0 / §7.1 / §7.2)

### 2.0 EN image-MOVED gate (§7.0) — MOVED (independently reproduced from the committed F58 operator)

| footing | generic-LoRA img AUC (§2.3) | vis-LoRA img AUC | dAUC(vis−gen) | MOVED? (≥+0.010 tr / ≥+0.005 dv) |
|---|---|---|---|---|
| train-LOO | 0.6236 | 0.6556 | **+0.0320** | ✓ (≥ +0.010, + sign) |
| dev | 0.6756 | 0.6822 | **+0.0065** | ✓ (≥ +0.005, + sign) |

`EN gate: MOVED → EN head PROCEEDS.` **Independently reproduced** this review by replicating the probe's
img-stream computation against the committed `encoder_swap_geometry.py` on the banked EN train (n=549) +
dev_seen (n=80) caches (CPU, zero test-touch, artifact NOT mutated): generic img trLOO 0.623610 / dev 0.675636,
vis img trLOO 0.655606 / dev 0.682182 ⇒ dAUC +0.031996 / +0.006545 — **bit-for-bit identical** to the on-disk
gate JSON (`scripts/analysis/vis_image_moved_MHC_out.json`) and to the submit record §6.3. The generic anchor
also reproduces the frozen §2.3 value exactly (0.6236 / 0.6756). Both footings clear F58's thresholds with the
same positive sign ⇒ MOVED; DSLIST branch = `"HateMM MHC"` correctly matched the MOVED rule.

### 2.1 HateMM — vis-LoRA vs frozen-CLIP (K-V1) AND vs generic-LoRA (K-V2)

| seed | protocol | vis acc/F1 | CLIP acc/F1 | Δ(vis−CLIP) acc/F1 | generic acc/F1 | Δ(vis−generic) acc/F1 |
|---|---|---|---|---|---|---|
| 0 | val-sel | 0.8558/0.8469 | 0.8279/0.8172 | **+0.0279/+0.0297** | 0.8605/0.8521 | **−0.0047/−0.0052** |
| 1 | val-sel | 0.8698/0.8620 | 0.8279/0.8163 | **+0.0419/+0.0457** | 0.8698/0.8620 | **0.0000/0.0000** |
| 2 | val-sel | 0.8558/0.8453 | 0.8047/0.7920 | **+0.0511/+0.0533** | 0.8558/0.8495 | **0.0000/−0.0042** |
| **mean** | **val-sel** | **0.8605/0.8514** | 0.8202/0.8085 | **+0.0403/+0.0429** | 0.8620/0.8545 | **−0.0016/−0.0031** |
| 0 | final-ep | 0.8791/0.8706 | 0.8186/0.7997 | **+0.0605/+0.0709** | 0.8651/0.8580 | **+0.0140/+0.0126** |
| 1 | final-ep | 0.8744/0.8653 | 0.8047/0.7822 | **+0.0697/+0.0831** | 0.8744/0.8660 | **0.0000/−0.0007** |
| 2 | final-ep | 0.8558/0.8461 | 0.8140/0.7988 | **+0.0418/+0.0473** | 0.8698/0.8613 | **−0.0140/−0.0152** |
| **mean** | **final-ep** | **0.8698/0.8607** | 0.8124/0.7936 | **+0.0573/+0.0671** | 0.8698/0.8618 | **0.0000/−0.0011** |

Sign vectors — vis−CLIP: val-sel acc `[+,+,+]` (3/3), final acc `[+,+,+]` (3/3). vis−generic: val-sel acc
`[−,0,0]` (0/3 positive), final acc `[+,0,−]` (1/3 positive).

### 2.2 MHC-EN — vis-LoRA vs frozen-CLIP (K-V1) AND vs generic-LoRA (K-V2) [EN gate MOVED ⇒ EN rows live]

| seed | protocol | vis acc/F1 | CLIP acc/F1 | Δ(vis−CLIP) acc/F1 | generic acc/F1 | Δ(vis−generic) acc/F1 |
|---|---|---|---|---|---|---|
| 0 | val-sel | 0.7888/0.7561 | 0.7826/0.7113 | **+0.0062/+0.0448** | 0.7516/0.6916 | **+0.0372/+0.0645** |
| 1 | val-sel | 0.7888/0.7342 | 0.7329/0.6034 | **+0.0559/+0.1308** | 0.7391/0.6920 | **+0.0497/+0.0422** |
| 2 | val-sel | 0.7826/0.7448 | 0.7702/0.6997 | **+0.0124/+0.0451** | 0.7888/0.7506 | **−0.0062/−0.0058** |
| **mean** | **val-sel** | **0.7867/0.7450** | 0.7619/0.6715 | **+0.0248/+0.0736** | 0.7598/0.7114 | **+0.0269/+0.0336** |
| 0 | final-ep | 0.7826/0.7448 | 0.7640/0.7145 | **+0.0186/+0.0303** | 0.7702/0.7302 | **+0.0124/+0.0146** |
| 1 | final-ep | 0.7702/0.7302 | 0.7826/0.7159 | **−0.0124/+0.0143** | 0.7764/0.7360 | **−0.0062/−0.0058** |
| 2 | final-ep | 0.7640/0.7274 | 0.7888/0.7303 | **−0.0248/−0.0029** | 0.7888/0.7506 | **−0.0248/−0.0232** |
| **mean** | **final-ep** | **0.7723/0.7341** | 0.7785/0.7202 | **−0.0062/+0.0139** | 0.7785/0.7389 | **−0.0062/−0.0048** |

Sign vectors — vis−CLIP: val-sel acc `[+,+,+]` (3/3), final acc `[+,−,−]` (1/3). vis−generic: val-sel acc
`[+,+,−]` (2/3), final acc `[+,−,−]` (1/3).

EN honesty flag (§3.5): vis-LoRA ≥ frozen-Qwen (val-sel 0.7805/0.7219; final 0.7847/0.7425)? **val-sel: CLEARS**
(0.7867/0.7450 vs 0.7805/0.7219, +0.0062 acc/+0.0231 F1); **final-ep: DOES NOT CLEAR** (0.7723/0.7341 vs
0.7847/0.7425, −0.0124 acc/−0.0084 F1).

---

## 3. Per-switch rulings (frozen text VERBATIM; each ruled exactly as worded)

### EN IMAGE-MOVED gate (§3.4) — **MOVED → EN-HEAD-PROCEEDS** (branch executed correctly)

`MOVED iff dAUC_img ≥ +0.010 (train-LOO) AND ≥ +0.005 (dev)`. Measured +0.0320 train-LOO (≥ +0.010 ✓) AND
+0.0065 dev (≥ +0.005 ✓), same positive sign ⇒ **MOVED**. Reproduced bit-for-bit from the committed operator
(§2.0). EN head budget correctly spent; DSLIST `"HateMM MHC"`. This is the pre-declared mechanical branch point
(train+dev only, zero test-touch), NOT a head-accuracy verdict.

### K-V1 — HOUSE PERFORMANCE CONJUNCT (vis-LoRA − frozen-CLIP; §3.2)

Rule per dataset × protocol: **mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND sign 3/3**, judged independently
under each protocol. (Prereg §3.2: *"A K-V1 pass that merely equals generic earns nothing — K-V2 is the decisive bar."*)

- **HateMM val-sel:** mean +0.0403 acc / +0.0429 F1, sign 3/3. **PASS** (marginal-adjacent note: seed0 Δacc
  +0.0279 sits just under the +0.030 per-seed bar, per B3 §2.2 precedent; the mean clears +0.040 and F1 is
  +0.0429 3/3 — a solid pass with one sub-bar seed).
- **HateMM final-ep:** mean +0.0573 acc / +0.0671 F1, sign 3/3, all per-seed Δacc ≥ +0.030. **PASS (non-marginal).**
- **MHC-EN val-sel:** mean **+0.0248** acc (**< +0.030**) / +0.0736 F1, sign 3/3. Acc conjunct not met ⇒ **FAIL.**
- **MHC-EN final-ep:** mean −0.0062 acc, sign 1/3. **FAIL.**

**K-V1: HateMM = PASS both protocols; MHC-EN = FAIL both protocols.** Per §3.2 the HateMM K-V1 pass **earns
nothing new** — HateMM already clears K-V1 under generic LoRA (banked F53), and the vis arm here **equals or is
fractionally below** generic on HateMM (K-V2 below), so the CLIP-relative pass re-confirms an inherited result
rather than adding a dataset.

### K-V2 — ADD-OVER-BANKED-GENERIC (THE DECISIVE ViT-CONTRIBUTION BAR; vis-LoRA − generic-LoRA, head-seed-paired; §3.3)

Rule PASS (per dataset, ≥1 protocol) = **mean Δacc ≥ +0.010 AND sign 3/3 positive AND mean ΔmF1 ≥ 0**.
TIE (= NO ViT CONTRIBUTION, the F0.7 outcome) = **mean |Δacc| < +0.010 OR sign not 3/3.**

- **HateMM val-sel:** mean −0.0016 acc, sign 0/3 positive. |Δacc| < +0.010 AND sign not 3/3 ⇒ **TIE.**
- **HateMM final-ep:** mean +0.0000 acc (< +0.010), sign 1/3. ⇒ **TIE.**
- **MHC-EN val-sel:** mean +0.0269 acc (≥ +0.010) BUT sign **2/3** (seed2 −0.0062). Sign-not-3/3 clause trips ⇒ **TIE.**
- **MHC-EN final-ep:** mean −0.0062 acc, sign 1/3. ⇒ **TIE.**

**K-V2: TIE on both datasets, both protocols (NO ViT contribution).** This is exactly the prereg's pre-declared
F0.7 most-likely outcome ("vision reach adds nothing over LLM-only LoRA"; report *"generic LoRA with vision
reach that did not matter,"* bank the negative, do NOT claim a vision contribution). **The MHC-EN val-sel
near-miss (+0.0269 acc / +0.0336 F1 mean) is correctly a TIE, not a pass:** its sign is only 2/3, and — applying
the mandatory Review Note (a) — the banked EN generic val-sel between-seed spread is **0.0497** (generic seed1
val-selected ep5 at acc 0.7391, a low draw), so the positive mean on seeds 0/1 is spread-driven; the 3/3-sign
teeth (the K-V2 rule's own guard) correctly catch it. No K-V2 pass exists, so the single-vis-draw F0.2 caveat
carries no pass — but had EN val-sel been read as a pass, both the F0.2 single-draw limitation and Note (a) wide
spread would have flagged it.

### EN HONESTY FLAG (§3.5) — moved but did not clear the frozen-Qwen ceiling on the deciding protocol

On EN, a claim requires vis-LoRA to beat the frozen-Qwen floor on the claimed protocol. Measured: **val-sel
CLEARS** frozen-Qwen (0.7867/0.7450 ≥ 0.7805/0.7219) but that protocol is a K-V1 FAIL + K-V2 TIE; **final-ep DOES
NOT CLEAR** (0.7723/0.7341 < 0.7847/0.7425). Since EN registers **no K-V1 pass and no K-V2 pass on either
protocol**, no EN performance claim survives to honour. Per §3.5 wording, the EN read is reported as **"moved but
did not clear the frozen-Qwen ceiling"** in the decision-relevant sense (on final-ep it does not even reach
frozen-Qwen; on val-sel it edges past frozen-Qwen but fails K-V1 and ties K-V2).

### KS-REGRESSION — BELOW-GENERIC KILL (mean Δacc(vis−generic) ≤ −0.014 on a held leg; §3.6)

Most-negative leg mean = MHC-EN final-ep **−0.0062** (next: HateMM val-sel −0.0016); all others ≥ 0.0000. None
≤ −0.014. **NOT triggered — no KILL.** The vision reach did not degrade adaptation below the banked head-seed spread.

### OVERFIT TRIPWIRES (§3.7)

- **(a) image-stream sanity = the §3.4 EN gate:** MOVED (passed). The optional record-only HateMM $0 diagnostic
  (`--dataset HateMM`) was not written to disk (`vis_image_moved_HateMM_out.json` absent) — **NON-BLOCKING**: §3.7a
  declares the HateMM image diagnostic explicitly **"not a kill"** and non-gating (HateMM's image is expected
  FLAT/swap-neutral per F58), so its absence is a documentation gap only, decision-inert.
- **(b) eval_loss band (~0.10–0.18; generic anchors HateMM 0.1084 / MHC 0.1620):** MHC_vis **0.1731** (in band,
  HIGHER than generic ⇒ no fire); HateMM_vis **0.1036** (in band, only 0.0048 below the generic anchor — **not
  "much lower"** ⇒ no fire). The "much-lower eval_loss + widening val-sel↘final gap" conjunct fires on neither.
  Observation (caveat, not a tripwire): on MHC the head does degrade val-sel↘final (final < val-sel by
  0.006–0.019 acc), consistent with EN's known weaker final-ep protocol, but the eval_loss is not much-lower, so
  §3.7b does not fire.

---

## 4. Compliance clauses (prereg binds; checked)

- **ViT-LoRA-present LOAD-BEARING census (§4.1a) — CONFIRMED on the real adapters (not just the smoke):** CPU
  census of `logging/lora/{HateMM_vis,MHC_vis}/adapter_model.safetensors` → **n_visual_lora_tensors = 320** (32
  ViT blocks × 5 Linears × {A,B}), **n_llm = 392** (byte-identical count to the banked generic adapter's 392),
  **n_merger = 0**, **n_patchembed = 0**, total 712; banked generic `logging/lora/HateMM` = 0 visual / 392 llm.
  The clean-superset premise (vis = generic ⊕ ViT-LoRA) **holds bit-for-bit ⇒ K-V2 is NOT vacuous.** Adapter
  sizes corroborate (vis 206 MB vs generic 161 MB; +44.6 MB ≈ 11.15M fp32 ViT-LoRA params). **COMPLIANT.**
- **Same-code Namespace diff (§4.1c/§4.2):** head-run Namespace, HateMM-vis(13330) vs HateMM-generic(13235) same
  seed = **only** `model`, `group_name`, and the derived-inert `exp_comment` + `output_path` differ; every
  substantive field identical (fusion_mode align, topk 20, proj/map 1024, dropout [0.2,0.4,0.1], batch_size 64,
  lr 1e-4, epochs 30, triplet, hybrid_loss True, warmup 5, no_hard_negatives 1, cos, lambda_seg 0.0,
  archive_feats None ⇒ archive OFF), and the inert TARC/oracle argparse defaults (tarc_target_source 'off',
  oracle_probe False, lambda_tarc 0.0 …) identical in both. **COMPLIANT.**
- **Single test-touch accounting (F0.1):** the job-13330 vis-LoRA head reads are the ONLY budgeted
  vis-LoRA-encoder test evaluations (EN + HateMM); no earlier vis-encoder test exposure. Prior held-out reads of
  this project already spent by other arms (pre-declared, not this cell): frozen-CLIP (12850), frozen-Qwen
  (12850 s0 + arcbase 12275/12276 s1/s2), generic-LoRA (13235, both datasets), the LoRA-HateMM verdict, and
  cand-2. The vis arm is a NEW single test-touch per dataset. **COMPLIANT.**
- **Single-encoder-draw (F0.2):** one vis-LoRA SFT draw per dataset (EN 13301, HateMM 13328), one extraction
  each (13302, 13329), 3 head-seeds (13330); the ±band is head-seed variance, not vis-SFT-draw variance. Honored;
  since K-V2 = TIE on every leg, no pass carries the caveat, but it is declared and would attach to any K-V2 pass.
  **COMPLIANT.**
- **13303 cancel / 13328 resubmit (ruled):** J3 13303 (SFT-HateMM) sat `PENDING (JobHeldUser)` ~29 h and never
  ran (submit record §7: aggregate per-user CPU/mem cap wedged the second concurrent 16-CPU SFT); it was
  cancelled with its dependents (`scancel 13303 13304 13307`), and — verified in the record — (i) the frozen
  script/config/data shas were **re-verified MATCH** before resubmit (C `3e895420…`, D `ca774914…`, config B
  `634bd0bb…`, HateMM data `93c6d3d1…`/`9e103ed3…`), (ii) HateMM-leg collision re-check was **CLEAN** (HateMM_vis
  adapter, `*LoRA-vis*` caches, `RAC_video_lora_vis*` dirs, `enc3s_*LoRA-vis*` logs all ABSENT — 13303 produced
  nothing), (iii) resubmitted **sequentially** 13328 → 13329 → 13330. **No double test-touch** (13303 never ran;
  the only HateMM vis SFT that produced an adapter is 13328), **same frozen script sha**, and the **EN gate was
  NOT re-run** (DSLIST carried from the already-PASSED §6.3 gate). **COMPLIANT.**
- **EN image-MOVED gate compliance:** the gate ran pre-head (after the EN extract 13302, before the head 13330),
  its raw output (+0.0320 train-LOO / +0.0065 dev) **reproduces bit-for-bit from the committed probe script**
  (§2.0, this review's independent CPU re-run), and the DSLIST branch (`"HateMM MHC"`) matched the prereg's MOVED
  rule. **COMPLIANT.**
- **Collision safety (§4.3):** fresh `-vis` tags throughout; no 12850/13235 arm, no frozen/`-LoRA_HF`/`-curric`
  cache overwritten (submit record §2 + §7 re-checks). **COMPLIANT.**
- **Freeze integrity (carried):** prereg sha matches; A–E + reused-machinery shas match on disk at verdict time
  (§0); no on-disk drift. **COMPLIANT.**

**No compliance violations found.** (One non-blocking documentation gap: the record-only HateMM image diagnostic
JSON was not persisted; §3.7a declares it non-gating.)

---

## 5. FINAL VERDICT BLOCK (performance clause only)

**Prereg §7.3 fixed write-up format:**

```
EN gate:  MOVED.
HateMM:   final-epoch: PASS (K-V1) · K-V2: tie · KS-regression: ok.
          val-selected: PASS (K-V1, seed0 sub-bar note) · K-V2: tie · KS-regression: ok.
MHC-EN:   final-epoch: FAIL (K-V1) · K-V2: tie · frozen-Qwen honesty: does-not.
          val-selected: FAIL (K-V1, acc +0.0248 < +0.030) · K-V2: tie · frozen-Qwen honesty: clears.
```

**Per-switch (verbatim rulings):**
- **EN image-MOVED gate:** MOVED → EN-HEAD-PROCEEDS (dAUC +0.0320 train-LOO / +0.0065 dev; reproduced bit-for-bit).
- **K-V1 (vs frozen-CLIP):** HateMM = PASS both protocols (final-ep non-marginal +0.0573/+0.0671 3/3; val-sel
  +0.0403/+0.0429 3/3, one seed fractionally sub-bar) — but earns nothing new (equals generic, §3.2);
  MHC-EN = FAIL both protocols (val-sel acc +0.0248 < +0.030; final-ep −0.0062 acc, sign 1/3).
- **K-V2 (add-over-generic, the DECISIVE ViT bar):** **TIE on both datasets, both protocols — NO ViT
  contribution** (HateMM val-sel −0.0016/0-3 sign, final +0.0000/1-3; MHC-EN val-sel +0.0269 acc but sign 2/3 ⇒
  TIE, final −0.0062/1-3). The F0.7 pre-declared outcome. The EN val-sel near-miss is a sign-2/3 TIE, inflated by
  the wide EN generic between-seed spread 0.0497 (mandatory Review Note (a)); F0.2 single-draw caveat noted.
- **EN honesty flag:** clears frozen-Qwen on val-sel only (K-V1 FAIL / K-V2 TIE there); does NOT clear on
  final-ep ⇒ "moved but did not clear the frozen-Qwen ceiling"; no EN claim survives.
- **KS-regression:** NOT triggered (most-negative leg −0.0062 > −0.014). **Overfit tripwires:** NOT fired
  (eval_loss HateMM_vis 0.1036 / MHC_vis 0.1731, both in band, neither much-lower).

**Composite (performance clause only):** The vision reach **is real and reaches the ViT** (320 LoRA tensors,
LOAD-BEARING check passed) and **demonstrably MOVED the EN image geometry** (gate MOVED, reproduced
bit-for-bit) — so the WORDING "EN is closed to the entire representation family / no vision lever was ever tried"
(F51 two-object closure / GAP-5b / C1) is **refuted at the mechanism level.** But on the head it adds **NOTHING
over LLM-only generic LoRA: K-V2 = TIE on both datasets and both protocols** — the decisive ViT-contribution bar
fails everywhere. K-V1 passes only on HateMM, where it merely re-confirms the inherited generic pass (vis ≈
generic) and earns no new dataset; MHC-EN fails K-V1 on both protocols and, on final-ep, does not even clear
frozen-Qwen. No kill-switch fired, no below-generic regression, no overfit tripwire, no compliance violation.
This is precisely the prereg's pre-declared **F0.7 honest most-likely outcome** — *"the EN image MOVES but the
K-V1 conjunct still FAILS"* combined with *"HateMM K-V2 TIE"* — the informative closure of the vision-adaptation
axis: the vision path is reachable and moves the EN image, but the r16 ViT-LoRA on <750 videos does not convert
that movement into a head-level gain over the LLM-only adapter.

**Explicitly out of scope for this reviewer (USER's rulings, prereg F0.3/§8):** whether the mechanism-level
refutation of the F51/GAP-5b wording (EN image demonstrably moved) carries any weight toward the goal's "novel"
clause under the **D7 encoder-class novelty boundary**, and whether the overall result satisfies the **goal**.
This review renders the **performance clause only**, as the frozen prereg mandates.

---

*Reviewer statements: hash verified before reading any metric; every floor/generic/vis number re-derived from
raw trainlogs with the byte-identical enc3seed parser and spot-checked against raw lines; the EN image-MOVED gate
independently reproduced from the committed F58 operator over banked train/dev caches (CPU, zero test-touch,
artifact not mutated); the ViT-LoRA-present census re-run on the real HateMM_vis + MHC_vis adapters (CPU); no
GPU/SLURM/Modal spent; no state/ mutated; nothing pushed; no goal/novelty claim made.*
