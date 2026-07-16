# MJ FORENSIC RECON — MLLM modality-reliability judgment as a NEW router input

**Agent:** MJ forensic recon (wave-4 candidate #2). **Date:** 2026-07-17.
**ZERO GPU / ZERO Modal / ZERO test-touch / ZERO user interaction.** Reading + forensic
arithmetic only. Deliverable = this committed doc + GO/NO-GO.
**Repo HEAD at recon:** `6032d32`.

**Docs read (verbatim):** `refine-logs/WAVE4_CANDIDATES.md` §2.2 MJ (`6032d32`),
`refine-logs/ROUTER_GATE_RECORD.md` (F47, `30d0ee1`), `refine-logs/ENCODER_SWAP_DIAGNOSIS.md`
(F44, `8a48938`), `refine-logs/C3_REAL_PREDICTOR_PROBE.md` + `C3_NONTARGET_PILOT_RECORD.md`
(C3-target/nontarget), `research-wiki/EXP_p2b_stronger_judge.md` (P2/P2b judge quality),
`autoresearch/goal_mllm_plus3/state/directions_tried.json` (bans F47/C3/P2/P5) +
`findings.jsonl` F11/F12/F44/F46/F47. Data inventoried on disk (read-only).

---

## RULING: **NO-GO** (arithmetic-decisive), with a subordinate **$0 GO-IF closure option**

MJ dies on the **required-judge-accuracy arithmetic before judge quality or cost even
enter**: to clear +0.020 on the 80-item MHC-EN dev split, the router must pick the correct
arm on the disagreement subset with accuracy **q ≥ 0.663**, but the *alignment ceiling* —
how well the true modality locus predicts which arm is correct on exactly that subset — is
**≤ 0.588 (generous) and ≈ 0.41–0.50 as F44/F47 actually measured it.** Therefore **even a
perfect modality judge cannot clear the bar** (a perfect judge gives gain ≈ 0 at the most
generous ceiling and −0.023 at F44's measured alignment). Judge imperfection (P2/C3/P5
ledger) only makes it worse. Two independent secondary kills confirm it: MJ is caught by the
**P2 "comparability ⊥ vote-correctness" meta-pattern** (its exact structural twin), and the
**routing signal it would supply is already banked** in the RGCL archive `modality_cues`
field — so it is not "genuinely NEW" in F47's carve-out sense, and it is **$0, not a GPU/queue
candidate.** The one thing keeping the door literally ajar is F47's new-source clause; a
**$0 CPU closure probe** over the banked archive would shut it as a paper-grade fifth
"better-signal-no-conversion" datum. That is a door-closer, **not** a goal-hit.

---

## 1. The required-judge-accuracy arithmetic (decisive, F44/F47 numbers only)

**Setup (all from `ROUTER_GATE_RECORD.md` §3.1–3.2, MHC-EN, the F47 gate MJ would re-use):**
dev N = 80 (25 hate); best single channel = **Qwen** (dev-optimal); routing changes the
prediction *only* on the CLIP↔Qwen disagreement subset, where exactly one arm is correct.

| seed | disagreement size D | "Qwen-correct" base rate p_Q | oracle (perfect-router) gain |
|---|---|---|---|
| s0 | 20 | 0.550 | +0.1125 |
| s1 | 23 | 0.565 | +0.1250 |
| s2 | 20 | 0.650 | +0.0875 |
| **3-seed mean** | **21** | **0.588** | **+0.1083** |

Routing gain for a router that sends items to the correct arm with accuracy `q`
(constant across seeds), on the full 80-item dev split:

```
gain(q) = mean_s [ (q − p_Q,s) · D_s / 80 ]  =  q·(21/80)  −  0.15415
        = 0.2625·q − 0.15415
```
(sanity: `gain(1.0)` = 0.2625 − 0.15415 = **+0.1084** = the oracle +0.1083 to rounding ✔.)

**Required routing accuracy to clear the +0.020 bar (K-R1):**
```
0.2625·q − 0.15415 ≥ 0.020  ⇒  q ≥ 0.6634
```
The router must pick the winning arm on **66.3 %** of disagreement items — **7.5 points above
the global "always-Qwen" prior of 0.588.**

**MJ is not a router; it is a modality judgment feeding the rule {visual→CLIP, textual→Qwen}.**
Decompose `q` via the mechanism-alignment `a` (routing accuracy of the *true* modality locus)
and the judge accuracy `j` on the modality label, as a noisy binary channel:
```
q = j·a + (1−j)·(1−a)   ⇒   perfect judge (j=1): q = a
```

**The alignment ceiling `a` is well below 0.663 — measured three ways, all F44/F47:**

| reading of `a` | value | gain(a) = 0.2625·a − 0.15415 | source |
|---|---|---|---|
| generous: modality = the global prior | 0.588 | **+0.0002 ≈ 0** | p_Q itself |
| F44 direct: "no coherent subgroup" | ≈ 0.50 | **−0.023** | error-set overlap fixes 11 / breaks 12 = net −1 on MHC-EN dev (F44 §4) |
| F47 realizable (dev-CV, all meta-features incl. per-modality sub-votes vimg/vtxt) | ≈ 0.413 | **−0.046** | ROUTER_GATE §3.3 dev-CV GBM −0.0458 (reproduces this row exactly ✔) |

**Conclusion (arithmetic alone):** a **perfect** modality judge (j=1 ⇒ q=a) yields **at best
≈ 0 and realistically −0.02 to −0.05** — it **cannot** reach +0.020, because per-item modality
locus does not predict which-arm-wins on the disagreement subset (F44 measured "no coherent
subgroup"; the +0.108 oracle is the *which-arm-correct* oracle, **not** a modality oracle).
Required q = 0.663 exceeds the perfect-judge ceiling q = a ≤ 0.588. **NO-GO on arithmetic,
independent of judge quality.** The model is calibrated: at a = 0.413 it reproduces F47's
realizable −0.046 to the digit, so this is not a hand-wave — it is F47's own number
re-derived from the modality-alignment angle.

**Judge quality (compounds, secondary).** Even granting a fictitious a ≥ 0.663, `j` for a 7B
meta-cognitive judgment is empirically near-chance on the routing-relevant axis:
- **P2** 7B comparability judge over-flagged INCOMPARABLE **83 % (EN)** and the axis was
  orthogonal — **"comparability ⊥ vote-correctness"** (EN lift +1.1 %, ZH −3.2 %).
- **P2b** 32B stronger judge + flip prompt: best **+2.7 EN train-side, never reached test**;
  orthogonality confirmed at 2 models × 2 evidence sets × 2 prompts.
- **C3-target** real Qwen-7B predictor: conditional Δacc **+0.0094** (near-chance,
  MHC anti-informative).
- **P5** counterfactual twins: **gate-fail + hurts.**

An imperfect judge (`j` < 1) pulls `q` toward 0.5 via `q = j·a + (1−j)(1−a)`, i.e. **below**
the already-failing perfect-judge ceiling. The ledger says the modality judgment MJ needs is
exactly the class of 7B meta-cognitive judgment measured at chance on the target quantity.

---

## 2. Non-isomorphism ruling (honest, against each cited record)

**vs C3-target (dead, 17th negative) — NOT strictly isomorphic, but the distinction does not
save it.** C3-target used an MLLM *predicted target/community* as a **content FEATURE** fused
into the retrieval key; MJ uses an MLLM *modality-reliability judgment* as a **ROUTER INPUT**
on the disagreement subset. Different question type (which-modality-is-reliable vs
who-is-the-target) and different injection point (per-item arm-selection vs key-fusion). The
structural distinction **is real** — MJ is genuinely not C3-target. But "not C3-target" is
necessary, not sufficient.

**vs P2 (neighbour-rerank, dead) — this is MJ's exact structural twin, and it is the binding
kill.** P2's premise was that a good MLLM *meta-cognitive judgment* (topical comparability)
would fix the vote; it failed because **the judged axis was orthogonal to what determines vote
correctness.** MJ's premise is that a good MLLM meta-cognitive judgment (modality locus) will
fix the vote; F44 already measured that **modality locus ⊥ which-arm-correct** on the
disagreement subset ("no coherent subgroup," net −1). MJ **is P2 with "modality reliability"
substituted for "comparability."** Same family, same orthogonality wall, now measured in
advance by F44 instead of discovered after a GPU spend.

**vs F47 (per-item router, dead, 22nd negative) — the carve-out is only *literally* satisfied,
and MJ's own input undercuts it.** F47 closed per-item channel selection at all three
supervision sources but wrote: *"…unless the selector input is a genuinely NEW information
source not derivable from banked features/votes."* Two problems for MJ:
1. F47's own feature set **already included per-modality sub-votes `vimg_c` / `vtxt_c`** for
   both arms (ROUTER_GATE §2) — the model-internal answer to "which modality carries the
   signal for this item." MJ's judgment is a **lossy, coarse, generative restatement of that
   same quantity.** The information is not new; only its wrapper is.
2. The judgment is **already banked** (§3) — so it is derivable from a banked artifact, i.e.
   the opposite of "genuinely new." At most it is a *different representation* of an
   already-banked modality signal, which F44 shows the encoder's own representation does not
   convert.

So MJ passes the carve-out only on a literal-text technicality (a generative output is not
*linearly* in `Z_best`), while failing its spirit (the underlying quantity is banked and was
partially in F47's features) — and the arithmetic in §1 shows the loophole is empty anyway.

**vs the D1 banked meta-frame** (`directions_tried.json` `diagnosis_frame`): "low-bandwidth
decision-side MLLM signals are conditionally redundant given frozen representation." MJ is a
few-bits, decision-side, MLLM signal at the vote injection point — **squarely inside D1.**

---

## 3. Banked-artifact inventory — the judgment ALREADY EXISTS ($0, cost picture corrected)

**DECISIVE: `data/Archive/MHC/*_Qwen2.5-VL-7B-Instruct_archive.jsonl` already contains a
per-video Qwen2.5-VL-7B modality-locus judgment with FULL dev coverage.**

- **Field:** each record's `archive.modality_cues = {visual, speech, on_screen_text}`
  free-text cue extraction (plus `mechanism`, `target_groups`, `explicitness`).
- **Coverage (MHC-EN):** train **552**, **dev_seen 80 (parse_ok 80/80)**, test_seen 161 —
  the exact 80 dev items the F47/MJ gate scores on.
- **Populated, not empty:** dev **hate** (n=25) → 8 visual / **20 speech** / 7 on-screen-text
  cues non-empty; dev **non-hate** (n=55) → 26 / 40 / 17. Example dev hate `0ATva49qP4w`:
  `speech`="…father forbidding a 'HARLOT'… females should be submissive…", `visual`="" —
  a genuine "the hate is speech-borne here" signal, exactly MJ's intended router input.
- **Provenance:** committed at **`d0f9e7b`** ("…archive JSONL (v1+v2)…"); generator
  `src/utils/generate_video_archive_HF.py` + `src/utils/p4_archive_fields.py` (the RGCL
  consensus-denoising / archive-card pillar).

**⇒ MJ needs NO generation.** WAVE4 §2.2(d)'s "generate the judgment on Modal cloud" plan is
**doubly wrong**: (i) Qwen2.5-VL generation needs the actual **frames** = raw content, hard-
banned from Modal (`modal_probe_runner.py` guard; CLAUDE.md 原始视频永不上云) — features-only
export cannot produce a *new* VLM judgment; (ii) no generation is needed at all, because the
judgment is already banked. **The data-boundary conflict the recon flagged is moot.**

**Caches checked and rejected as MJ inputs** (for completeness):
| cache | why not usable as MJ's dev router input |
|---|---|
| `artifacts/c3_nontarget/{MHC,HateMM}/text/` | 300 **train-sampled** dense-reasoning texts, **no dev coverage**; content, not modality locus |
| `data/gt/MHC/target_pred_qwen7b.json` | community/target only (`{"community": …}`), no modality |
| `data/Counterfactual/MHC/train_twins.jsonl` (P5) | text rewrites, train only, no modality |
| `data/CLIP_Embedding/MHC/train_p3pool_*.pt` (P3) | segment **hate-density** pool weights, not modality locus; train/dev/test but wrong signal |
| `data/MLLM_scores/{HateMM,HateClipSeg}/…` | P6/P10 **localization** scores, not MHC-EN, not modality |

**Had it not been banked**, the correct path would be **local SLURM** (Modal blocked by the
frame boundary), following the c3_nontarget precedent (job 13101, 1×A100, **85 min for 300
videos** ⇒ ~629 MHC-EN videos ≈ 3 h), **QUEUE-BLOCKED behind LoRA-HateMM line A** (jobs
13228/13229 `PD JobHeldUser`). Because it **is** banked, none of that applies — **MJ is a $0
CPU probe or nothing.**

---

## 4. GO-IF: the only admissible move — a $0 archive-modality-router CLOSURE probe

If (and only if) the loop wants F47's literal new-source carve-out **formally shut** with a
paper-grade fifth "better-signal-no-conversion" datum (parallel to FA closing the F44 fusion
cell), run a **$0 CPU** closure probe. It is **not** a goal candidate — the §1 arithmetic
pre-registers a KILL.

**Design (reuse `scripts/analysis/cross_channel_router_gate.py` verbatim + one feature):**
1. Derive a per-item modality feature from banked `modality_cues` on train+dev — e.g.
   `speech_strength − visual_strength` (cue-text lengths / non-empty indicators) and an
   `on_screen_text` indicator; also a binary `argmax(cue)` variant. **Zero generation.**
2. Append it to F47's meta-feature set; re-run F47's gate **unchanged** on the 12 banked e29
   heads (bit-exact machinery already validated, ROUTER_GATE §1).
3. **Pre-declared kill-switches (ported from F47 verbatim):**
   - **K-MJ-1** (= K-R1): routed − best-single **< +0.020** MHC-EN dev (3-seed mean) **OR**
     bootstrap CI-low ≤ 0 ⇒ **KILL.**
   - **K-MJ-2** (= K-R2): label-oracle calibration accZA ≥ 0.99 else MACHINERY_INVALID.
   - **K-MJ-3** (= K-R3, decisive): dev-CV realizable ceiling with the MJ feature must exceed
     perm-null p95. F47's ceiling was **−0.046**; MJ must lift a negative ceiling above null —
     the §1 arithmetic says it will not.
4. **Cost:** $0 CPU, minutes, banked archive + banked heads only. **Zero test-touch** (the
   test archive exists but is not opened). No Modal, no SLURM, no download.
5. **Outcome accounting:** a KILL (expected) closes F47's carve-out and books a paper datum; a
   surprise pass (arithmetically near-impossible) would promote to a pre-registered local
   router under the standard ceremony. Either way **no GPU inside MJ.**

**Recommendation:** hold MJ as **NO-GO** for the goal. Fire the §4 closure probe **only** if
the loop is actively closing F47's carve-out for the terminus writeup (it is cheaper and more
decisive than leaving the clause "ajar"); otherwise spend nothing.

---

## 5. Kill-switch already tripped (why this is NO-GO, not GO-IF-by-default)

- **Arithmetic pre-kill:** required q = 0.663 > perfect-judge ceiling q = a ≤ 0.588; gain(a)
  ≤ +0.0002 at the most generous alignment, −0.023 at F44's measured alignment. A perfect
  judge fails. This fires **before** cost or judge quality.
- **Structural pre-kill:** MJ = P2 with "modality reliability" for "comparability";
  F44 pre-measures the orthogonality (modality locus ⊥ which-arm-correct) that P2 discovered
  post-GPU.
- **Carve-out pre-kill:** the signal is already banked (archive `modality_cues`, `d0f9e7b`)
  and partially in F47's per-modality sub-votes ⇒ not "genuinely new."

---

## 6. Provenance / hygiene
- **HEAD:** `6032d32`. **Banked inputs inspected (read-only):**
  `data/Archive/MHC/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_archive.jsonl`
  (provenance `d0f9e7b`); `refine-logs/ROUTER_GATE_RECORD.md` (`30d0ee1`);
  `refine-logs/ENCODER_SWAP_DIAGNOSIS.md` (`8a48938`). Numbers §1 recomputed from
  ROUTER_GATE §3.1–3.2 only.
- **Required statements:** ZERO GPU / SLURM / Modal spent by this recon; no held-out **test**
  metric read or produced (dev/train archive `parse_ok` counts and F47's already-published
  dev disagreement numbers only); no `state/`, prereg, config, `research-wiki/`, or frozen
  artifact mutated. Not pushed.
