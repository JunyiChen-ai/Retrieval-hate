---
type: experiment
node_id: exp:exp-cand2-curriculum
title: "cand-2 curriculum LoRA-SFT (round-4 closing) — confusion-weighted single-video SFT curriculum vs generic LoRA and vs frozen-CLIP on MHC-ZH + HateMM: 3-seed paired, dual protocol, single manipulated variable = example multiplicity"
idea_id: "idea:memory-adaptation-coupling-curriculum"
status: CLOSED
verdict: tie
confidence: high
date: "2026-07-18"
hardware: "1x A100 x2 (SLURM); fresh curric LoRA-SFT ZH ~2.8-3.3 h (job 13237) + HateMM ~3.1-3.5 h (13238), extractions ~0.4 h each (13239/13240), 3-seed head ~2 min (13241); one budgeted curriculum-LoRA-encoder test evaluation per dataset (ZH + HateMM)"
duration: "job chain 13237/13238 (curric lora_sft ZH/HateMM) -> 13239/13240 (gen_embed_lora) -> 13241 (enc3seed head; 3 ZH-curric + 3 HateMM-curric rows)"
novelty_clause: "PENDING USER D7 SUB-RULING. This cell decides the PERFORMANCE clause only (K-C2-0/1/2, KS-regression, KS-below-floor, ZH-robustness, per the frozen prereg). The lever is a memory->adaptation-coupling SFT curriculum; whether it counts as distinct from generic encoder LoRA (i.e. clears the narrower D7 memory-coupling sub-ruling) is the USER's ruling, NOT decided here (prereg F0.3). Not folded into any main table (PAPER_MASTER_TABLES.md PUR-banner / PUR-4)."
provenance: "CLOSED under full single-submit ceremony. Recon (GO-IF, design (i) only) refine-logs/CAND2_CURRICULUM_RECON.md (7087b5a); prereg refine-logs/CAND2_CURRICULUM_PREREG.md (76ef0e2, sha256 e5a689d9...f939790e); independent 0-context prereg review refine-logs/CAND2_PREREG_REVIEW.md (c1315cb, APPROVED-WITH-NOTES); hash-freeze refine-logs/CAND2_FREEZE.md (7804324, freeze PASS + K-C2-0 PASS both); single-submit record refine-logs/CAND2_SUBMIT_RECORD.md (1ea3c13, job chain 13237/13238->13239/13240->13241); independent 0-context verdict refine-logs/CAND2_VERDICT_REVIEW.md (546acc5). Comparison arms re-parsed with the byte-identical enc3seed parser: ZH generic-LoRA 13150, ZH frozen-CLIP 13115, HateMM generic-LoRA 13235, HateMM frozen-CLIP 12850; every floor/generic mean matches the prereg s2.1/s2.2 to 4dp."
added: 2026-07-18T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "LoRA-SFT", "curriculum", "confusion-weighting", "hard-example-mining", "memory-adaptation-coupling", "RA-HMD", "frozen-CLIP", "multi-seed", "paired-test", "MHC-ZH", "HateMM", "dual-protocol", "pre-registered", "CLOSED", "tie", "F56", "novelty-pending", "D7", "rep2", "F59", "weakly-hardened"]
---

# cand-2 curriculum LoRA-SFT (round-4 closing) — memory-mined confusion-weighted SFT curriculum vs generic LoRA

> **STATUS: `CLOSED` 2026-07-18 — VERDICT (`refine-logs/CAND2_VERDICT_REVIEW.md`, commit `546acc5`;
> job 13241, independent 0-context reviewer, hash-verified vs the frozen prereg `76ef0e2`):**
>
> ```
> ZH:     final-epoch: PASS (K-C2-1, MARGINAL) · K-C2-2: tie · ZH-robustness: not strengthened.
>         val-selected: FAIL (K-C2-1)          · K-C2-2: tie.
> HateMM: final-epoch: PASS (K-C2-1, hold)     · K-C2-2: tie.
>         val-selected: PASS (K-C2-1, hold)     · K-C2-2: pass (single-draw caveat, F0.2).
> ```
>
> **Novelty = PENDING USER D7 SUB-RULING** (memory->adaptation coupling). Not folded into any main table.

**verdict:** `tie` (ZH ties generic on both protocols = pre-declared F0.7 outcome; HateMM K-C2-2 adds only
val-selected, single-draw; ZH-robustness NOT strengthened; no kill fired) · **confidence:** `high`

## 0. What this cell is (and is NOT) — read first

The **round-4 closing** probe tests whether *coupling the retrieval memory into the encoder-adaptation
objective* upgrades the generic LoRA leg from "encoder-class (D7-dead)" to "memory-coupled." The cell is a
**confusion-weighted single-video SFT curriculum** (recon design (i-a)): the RGCL memory's leave-one-out
top-20 signed-cosine kNN vote over the **banked frozen-Qwen train features** assigns each train video a
confusability `c_i = exp(-|vote_i| / tau)` (peaks at the decision boundary); multiplicity `w_i = 1 + lambda*c_i`
**reweights how often each SFT record appears**. The SFT records are **byte-identical** to the generic-LoRA
arm (same 8 frames, same instruction, same word target); the **ONLY manipulated variable is example
multiplicity**, and the reweighted multiset is capped to `N_train` by largest-remainder apportionment so the
3-epoch step count is **identical** to generic (cost-neutral). Features feed the unchanged archive-OFF RGCL
align-fusion head + top-20 kNN (`enc3s`/12850 protocol), 3 head-seeds paired vs both the banked frozen-CLIP
floor (K-C2-1) and the banked generic-LoRA arm (K-C2-2), dual protocol.

- **PERFORMANCE clause only.** Whether the coupling counts as novel-in-field is the user's **D7 sub-ruling**,
  not decided here (prereg F0.3). Curriculum learning / hard-example mining is textbook outside hateful-video;
  the sibling idea (retrieval-mined hard *negatives*) was already D7-dead (`C3GEO_FORENSIC_RECON.md`). cand-2
  survives that kill on one thin difference (F0.6): the head's own per-epoch mining reads *frozen* features and
  can only exploit existing separation; the curriculum is the only lever that makes the encoder *allocate*
  r16/3-epoch capacity to the confusable region.
- **Opens NO new dataset (pre-declared F0.4).** By F44/F45 modality-locus arithmetic a text/curriculum lever
  holds ZH and can add only HateMM (inherited, frozen-swap-sufficient; its convertible signal is text-carried,
  not image-borne — F58, `refine-logs/HATEMM_LORA_STREAM_DECOMP.md`, `51eb95b`) or EN (label-limited, dead) — so cand-2's realistic
  best case is a cleaner/robuster story on datasets generic LoRA already passes, not a new performance route.
- **Single-curriculum-draw limitation (pre-declared, F0.2, CRITICAL for K-C2-2).** All 3 head-seeds read ONE
  curriculum-SFT encoder draw per dataset; the +-band is head-seed variance, NOT curriculum-SFT-draw variance.
  K-C2-2 is therefore **one curriculum draw vs one generic draw, both read by 3 head-seeds** — it cannot
  separate the curriculum effect from SFT-draw luck. A stability claim would need >=3 fresh curriculum retrains
  (out of scope, pre-declared).
- **Class-balance shift is a pre-declared property, not a hidden confound (F0.8):** confusion-weighting shifts
  SFT class balance ZH 31.1%->41.1% / HateMM 40.1%->37.7% hateful (mechanism-aligned per F45: the ZH gain is a
  minority hate-recall Pareto move).

## 1. Comparison arms (re-derived to 4dp vs the prereg; enc3seed parser, spot-checked vs raw lines)

| arm | protocol | 3-seed mean acc / mF1 |
|---|---|---|
| ZH frozen-CLIP (13115) | val-sel | 0.8076 / 0.7676 |
| ZH frozen-CLIP (13115) | final-ep | 0.8143 / 0.7720 |
| ZH generic-LoRA (13150) | val-sel | 0.8322 / 0.8015 |
| ZH generic-LoRA (13150) | final-ep | 0.8456 / 0.8173 |
| HateMM frozen-CLIP (12850) | val-sel | 0.8202 / 0.8085 |
| HateMM frozen-CLIP (12850) | final-ep | 0.8124 / 0.7936 |
| HateMM generic-LoRA (13235) | val-sel | 0.8620 / 0.8545 |
| HateMM generic-LoRA (13235) | final-ep | 0.8698 / 0.8618 |

## 2. Curriculum arm — measured (job 13241; 3-seed mean, both protocols)

| arm | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | mean acc/F1 |
|---|---|---|---|---|---|
| ZH-curric | val-sel | 0.8188/0.7837 (e13) | 0.8523/0.8270 (e18) | 0.8054/0.7734 (e5) | **0.8255/0.7947** |
| ZH-curric | final-ep | 0.8591/0.8339 | 0.8523/0.8249 | 0.8456/0.8158 | **0.8523/0.8249** |
| HateMM-curric | val-sel | 0.8791/0.8730 (e29) | 0.8744/0.8678 (e14) | 0.8791/0.8724 (e10) | **0.8775/0.8711** |
| HateMM-curric | final-ep | 0.8791/0.8730 | 0.8791/0.8724 | 0.8791/0.8724 | **0.8791/0.8726** |

### 2.1 vs frozen-CLIP (K-C2-1, hold the inherited pass)

| dataset | protocol | mean Delta(curric-CLIP) acc/F1 | sign (acc) | K-C2-1 |
|---|---|---|---|---|
| ZH | val-sel | +0.0179 / +0.0271 | 2/3 | **FAIL** (mean Delta-acc < +0.030) |
| ZH | final-ep | +0.0380 / +0.0529 | 3/3 | **PASS (MARGINAL)** (seed2 +0.0134 < +0.030 per-seed; mean < +0.040) |
| HateMM | val-sel | +0.0573 / +0.0626 | 3/3 | **PASS (hold)** |
| HateMM | final-ep | +0.0667 / +0.0790 | 3/3 | **PASS (hold)** |

### 2.2 vs generic-LoRA (K-C2-2, the add-over-generic / novelty-earning bar; head-seed paired)

Rule PASS (per dataset, >=1 protocol) = mean Delta-acc >= +0.010 AND sign 3/3 positive AND mean Delta-mF1 >= 0.
TIE (= NO novelty, the F0.7 outcome) = mean |Delta-acc| < +0.010 OR sign not 3/3.

| dataset | protocol | mean Delta(curric-generic) acc/F1 | sign (acc) | K-C2-2 |
|---|---|---|---|---|
| ZH | val-sel | -0.0067 / -0.0068 | 1/3 | **TIE** |
| ZH | final-ep | +0.0067 / +0.0076 | 2/3 | **TIE** (< +0.010, sign not 3/3) |
| HateMM | val-sel | **+0.0155 / +0.0166** | 3/3 | **PASS** (single-draw caveat F0.2) |
| HateMM | final-ep | +0.0093 / +0.0108 | 3/3 | **TIE** (+0.0093 < +0.010 by 0.0007) |

## 3. Kill-switch + clause rulings (frozen prereg text applied verbatim)

- **K-C2-0 (mining validity, $0 CPU pre-GPU gate): PASS both** — banked result STANDS. ZH LOO-err 0.2073 /
  c-Gini 0.5634 / cov 0.6667 / hard-head x2.11; HateMM 0.1935 / 0.6497 / 0.6756 / x2.08; neither at the ~0
  memorization auto-KILL; `train_curric` shas equal frozen F/G.
- **K-C2-1 (hold inherited pass): ZH final-ep PASS (MARGINAL) / val-sel FAIL; HateMM PASS both (held).**
- **K-C2-2 (add-over-generic, novelty bar): ZH = TIE both protocols (NO NOVELTY on ZH, the F0.7 outcome on the
  a-priori-most-likely leg); HateMM = PASS via val-sel only (+0.0155 acc, 3/3, ΔmF1 +0.0166), final-ep TIE
  (+0.0093).** Novelty signal "K-C2-2 PASS on >=1 dataset" MET on HateMM only — carrying the F0.2
  single-curriculum-draw caveat and a protocol-split (val-sel-only) caveat, on the hold/inherited leg
  (F0.4; frozen-swap-sufficient, convertible signal text-carried — F58, `51eb95b`), NOT on ZH.
- **KS-regression (below-generic KILL, mean Delta-acc(curric-generic) <= -0.014 on a held leg): NOT triggered**
  (most-negative leg-mean = ZH val-sel -0.0067 > -0.014).
- **KS-below-floor (curric below CLIP floor on ZH): NOT triggered** (ZH-curric above CLIP floor both protocols).
- **ZH-robustness clause (pre-declared "ZH leg strengthened"): NOT strengthened.** (a) val-sel conjunct does not
  pass (mean +0.0179 acc, sign 2/3); (b) final-ep does not become non-marginal (mean +0.0380 < +0.040; seed2
  +0.0134 < +0.030 per-seed). The primary declared performance goal of cand-2 — strengthen the marginal ZH leg
  — was NOT achieved. ZH final-ep is essentially B3's status (B3 +0.0313; curric +0.0380, higher but still
  sub-+0.040 with seed2 below the per-seed bar).

## 4. Compliance (prereg binds; clean)

- **Same-code pairing (§4.1c/§4.2):** HateMM-curric(13241) vs HateMM-generic(13235) same seed = only `model` /
  `group_name` / `exp_comment` / derived `output_path` differ; **76/80 fields identical** (fusion align, topk 20,
  proj/map 1024, dropout [0.2,0.4,0.1], bz 64, lr 1e-4, epochs 30, triplet, hybrid loss, warmup 5, hard_neg 1,
  cos, lambda_seg 0, archive OFF). Code-version confound retired. COMPLIANT.
- **Single-curriculum-draw (F0.2):** one SFT draw per dataset (13237 ZH / 13238 HateMM), one extraction each
  (13239/13240), 3 head-seeds (13241). Honored; caveat travels with the HateMM K-C2-2 PASS. COMPLIANT.
- **Single test-touch:** the job-13241 curriculum-LoRA head reads are the ONLY budgeted curriculum-encoder test
  evaluations (ZH + HateMM); zero earlier curriculum test exposure. COMPLIANT.
- **Class-balance disclosure (F0.8):** DISCLOSED (non-blocking 0.1pt rounding slip noted by prereg review).
- **Freeze integrity (carried):** prereg sha matches; freeze-block A-H + reused-machinery shas matched at freeze;
  builder bit-exact idempotent; KC20 JSON `train_curric` shas equal frozen F/G. Non-blocking echoed note: HateMM
  KC20 `n_train_cache 744` vs `n_train_sft 743`, `n_anchor_missing_from_cache 0` (all 743 SFT anchors present;
  one cache-only train video is a potential LOO neighbour only; train-only, no leakage, predates cand-2). BENIGN.

**No compliance violations found.**

## 5. What the tie means (D7 boundary — this cell does NOT decide)

The curriculum **held** both inherited K-C2-1 passes (HateMM both; ZH final-ep, still marginal) but did **not**
deliver the primary declared upgrade — the ZH leg is **not strengthened** and **ties** generic on both protocols
(K-C2-2 TIE = "generic LoRA with reshuffled data," the prereg's own pre-declared wording). The add-over-generic
bar is met on **exactly one dataset — HateMM, val-selected only, single-draw** — off the a-priori-favoured leg
and not coinciding with a ZH-robustness upgrade. The honest reading: the **memory->adaptation coupling's
measurable effect over generic LoRA is dataset- and protocol-local** — the head's own per-epoch mining
re-extracts most of what the curriculum injects into the encoder, one level up on the P3/C3-geo redundancy
pattern (F0.7). cand-2 opens no new dataset (F0.4) and is not folded into any main table; whether its one-cell
add-over-generic suffices for the D7 memory->adaptation-coupling novelty sub-ruling is the user's decision.

## 6. Connections

- tests-coupling-upgrade-of -> `exp:exp-lora-zh-b3` (ZH generic encoder-level LoRA marginal pass — cand-2's
  primary "strengthen" target; NOT strengthened) and `exp:exp-lora-hatemm` (HateMM generic LoRA both-protocol
  pass — cand-2's "hold" leg; held)
- ties-against -> `exp:exp-lora-zh-b3` / `exp:exp-lora-hatemm` generic-LoRA arms (K-C2-2 head-seed-paired)
- non-redundancy-basis (thin) -> `refine-logs/C3GEO_FORENSIC_RECON.md` (sibling retrieval-mined hard-negative
  idea, D7-dead; cand-2 survives on the encoder-capacity-allocation distinction F0.6)
- mechanism -> `refine-logs/ENCODER_SWAP_DIAGNOSIS.md` (F44) + `refine-logs/B3_ZH_LORA_DECOMPOSITION.md` (F45);
  the confusion-weighting class shift is F45-aligned (minority hate-recall Pareto)
- analysed-as -> `DRAFT_analysis_chapter.md` §3.9 (completes the adaptation phase diagram; coupling effect is
  dataset/protocol-local) and `DRAFT_experiments_chapter.md` §7 Table 8
- decision-support -> `refine-logs/D7_RULING_DOSSIER.md` (`def6ce3`, D7 ruling — evidence-only)
- novelty-clause -> PENDING USER D7 SUB-RULING (`PAPER_MASTER_TABLES.md` PUR-4 / PUR-banner)

## 8. Draw-2 replication (rep2, F59) — WEAKLY-HARDENED

> **STATUS: rep2 draw-2 CLOSED 2026-07-18 — VERDICT (`refine-logs/CAND2_REP2_VERDICT_REVIEW.md`, commit
> `aa48275`; job 13246, independent 0-context reviewer, hash-verified vs the frozen rep2 prereg `2d15ffb`):**
>
> ```
> HateMM draw-2: K-REP-1 (val-sel add-over-generic): NOT-PASS (mean +0.0108 acc, sign 2/3, ΔmF1 +0.0120).
>                K-REP-2 (pooled 6-pt): HARDENED (pooled mean +0.01317 acc, sign 5/6).
>                KS-REP: NOT fired.  final-ep add-over-generic (non-binding): mean +0.0140 acc, sign 3/3.
> VERDICT: F56 HateMM val-sel add-over-generic = WEAKLY-HARDENED.
> (D7 novelty + goal satisfaction remain the USER's — not decided here.)
> ```

The §2.2 HateMM K-C2-2 val-selected PASS (draw-1: +0.0155 acc / +0.0166 mF1, 3/3, single-curriculum-draw
caveat F0.2) was the one live novelty-bearing positive. rep2 ran **exactly one** independent second SFT
draw — seed = 1 the single manipulated variable (draw-1 was HF default 42), curriculum multiset bit-exact
to draw-1 (sha `73307ef2…82b`) — on **HateMM only**, to test whether it replicates.

### 8.1 Draw-2 val-selected add-over-generic (K-REP-1, PRIMARY/BINDING; vs banked generic-LoRA 13235)

| seed | rep2 acc/mF1 | generic acc/mF1 | Δacc | ΔmF1 |
|---|---|---|---|---|
| 0 | 0.8744/0.8678 | 0.8605/0.8521 | +0.0139 | +0.0157 |
| 1 | 0.8651/0.8574 | 0.8698/0.8620 | −0.0047 | −0.0046 |
| 2 | 0.8791/0.8745 | 0.8558/0.8495 | +0.0233 | +0.0250 |
| **mean** | 0.8729/0.8666 | 0.8620/0.8545 | **+0.0108** | **+0.0120** |

Point bar (mean Δacc ≥ +0.010) **cleared** (+0.0108), but the **3/3 sign gate failed** (seed1 −0.0047 →
2/3) ⇒ **K-REP-1 does NOT PASS.** Non-binding final-epoch add-over-generic: per-seed [+0.0186, +0.0140,
+0.0093], mean **+0.0140**, sign **3/3**, ΔmF1 +0.0162 (reported, not decision-bearing).

### 8.2 Pooled 2-draw read (K-REP-2, SECONDARY)

| draw | s0 Δacc | s1 Δacc | s2 Δacc | draw sign |
|---|---|---|---|---|
| draw-1 (re-derived) | +0.0186 | +0.0046 | +0.0233 | 3/3 |
| draw-2 (measured) | +0.0139 | −0.0047 | +0.0233 | 2/3 |

Pooled mean = **+0.01317** (sum +0.0790 / 6), sign **5/6** positive → clears ≥ +0.010 AND ≥ 5/6 ⇒
**K-REP-2 = HARDENED.** **KS-REP** (retirement kill; fires iff draw-2 mean Δacc ≤ −0.014) **NOT fired**
(observed +0.0108).

### 8.3 What it means (D7 boundary — this cell does NOT decide)

The HateMM val-selected add-over-generic is now **pooled weakly-hardened across two draws (5/6 sign),
per-draw 3/3 gate not met** — it did not fully replicate on the binding protocol (seed1 flipped −0.0047),
did not reverse, and the pooled read agreed in direction, so it is **not** a single-draw cherry-pick but
is **weaker than a clean replication**, and remains a 2-draw estimate (F-R0.9). Seed compliance verified
(`training_args.bin` seed = 1); the single draw-2 attempt is **binding and consumed** — no further draws
are possible. rep2 measured HateMM only, so the ZH leg is untouched and **ZH-robustness remains not
strengthened** (§3, F56). Novelty stays **PENDING USER D7 SUB-RULING**; not folded into any main table.

**Provenance (rep2 ceremony chain):** prereg `2d15ffb` → independent 0-context prereg review `e2aee03`
(APPROVED-WITH-NOTES) → hash-freeze `6c11988` → single-submit record `d06ad07` → independent 0-context
verdict `aa48275` (job 13246).

## 7. Revision history

| rev | date | status | change | authority |
|---|---|---|---|---|
| r0 | 2026-07-18 | CLOSED | Initial per-experiment note, authored at paper-integration time from the committed ceremony chain (recon `7087b5a` -> prereg `76ef0e2` -> review `c1315cb` -> freeze `7804324` -> submit `1ea3c13` -> verdict `546acc5`, job chain 13237/13238->13239/13240->13241). ZH K-C2-2 TIE both protocols (F0.7 outcome), ZH-robustness NOT strengthened; HateMM K-C2-1 held both, K-C2-2 PASS val-sel only (+0.0155 acc/+0.0166 F1, 3/3, single-draw; final-ep tie by 0.0007). No kill fired; compliance clean. Novelty = PENDING USER D7 SUB-RULING; not folded into any main table. | paper integrator |
| r1 | 2026-07-18 | CLOSED | Appended §8 draw-2 replication (rep2, F59, verdict `aa48275`, job 13246; prereg chain `2d15ffb`->`e2aee03`->`6c11988`->`d06ad07`->`aa48275`). HateMM-only second SFT draw (seed=1): K-REP-1 NOT-PASS (val-sel mean +0.0108 acc, sign 2/3, seed1 −0.0047), KS-REP NOT fired, K-REP-2 pooled 6-pt HARDENED (+0.01317, 5/6). VERDICT: F56 HateMM val-sel add-over-generic = WEAKLY-HARDENED (pooled weakly-hardened across two draws, per-draw 3/3 gate not met; 2-draw estimate, binding attempt consumed). ZH untouched, ZH-robustness still not strengthened. Novelty PENDING USER D7 SUB-RULING; not folded into any main table. Draw-1 sections §0–§7-r0 byte-unchanged. | F59 addendum clerk |
