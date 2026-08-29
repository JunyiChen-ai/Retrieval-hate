# PAPER CONSISTENCY AUDIT — round 3 (post-fold QA of commits dc224b9 / 247fcaf / 2c1900f)

**Auditor:** zero-context adversarial consistency auditor. **Date:** 2026-07-17.
**Scope:** READ-ONLY except this file. NO GPU / SLURM / Modal. No `state/` touch. No push.
**Commits under audit:**
- `dc224b9` — folds rounds 2–3 into `DRAFT_analysis_chapter.md` §3.6/§3.7, `DRAFT_experiments_chapter.md` §7,
  `PAPER_MASTER_TABLES.md` T5.
- `247fcaf` — intro limitation citation (`DRAFT_intro_related_limitations.md`) + B3 decomposition (F45) folded into
  experiments §7.
- `2c1900f` — two figures + `research-wiki/figures/fig_data.json` + `scripts/analysis/make_encoder_figures.py`.

**Method.** Every number in the new/edited sections was traced to its PRIMARY committed record (verdict-review /
gate-JSON / diagnosis / trainlog-parse) and compared to 4 dp. House numeric-provenance rule applied: a draft number
is a MATCH only if it equals the primary record, not a summary/memory. 37 numeric checks performed (≥15 required).

---

## 1. NUMERIC SPOT-CHECK TABLE

Legend: draft value → primary record value → primary file → verdict. Rounding to the draft's stated precision.

| # | draft loc | draft value | primary record value | primary file | verdict |
|---|---|---|---|---|---|
| 1 | anal §3.6 / exp §7 T5 / T5.2 | S2S oracle HateMM **+0.0917** | oracle Δacc +0.0917 (gate 4) | S2S_PROBE_VERDICT_REVIEW.md | MATCH |
| 2 | same | S2S oracle MHC **+0.1399** | oracle Δacc +0.1399 | S2S_PROBE_VERDICT_REVIEW.md | MATCH |
| 3 | anal §3.6 / exp §7 / T5.2 | S2S MeanMaxSim HateMM **+0.0035 acc / +0.0003 mF1** | gate 5: +0.0035 / +0.0003 | S2S_PROBE_VERDICT_REVIEW.md | MATCH |
| 4 | same | S2S MHC-EN **−0.0397** | Δacc −0.0397 (SET−POOLED) | S2S_PROBE_VERDICT_REVIEW.md | MATCH |
| 5 | anal §3.6 / exp §7 / T5.2 | W2-A oracle **+0.0635 (HateMM) / +0.0970 (MHC)** | K5 Δ(oracle−CONCAT) +0.0635 / +0.0970 | W2A_PROBE_VERDICT_REVIEW.md | MATCH |
| 6 | anal §3.6 / exp §7 | W2-A K9 HateMM **−0.0000, CI [−0.0052,+0.0049]** | K9 −0.0000, CI[−0.0052,+0.0049] | W2A_PROBE_VERDICT_REVIEW.md | MATCH |
| 7 | anal §3.6 / exp §7 | W2-A K9 MHC **−0.0038, CI [−0.0099,+0.0019]** | K9 −0.0038, CI[−0.0099,+0.0019] | W2A_PROBE_VERDICT_REVIEW.md | MATCH |
| 8 | exp §7 / T5.2 | W2-A advisory kNN **−0.0259 / −0.0509** | K6 HateMM −0.0259; MHC −0.0509 | W2A_PROBE_VERDICT_REVIEW.md | MATCH |
| 9 | anal §3.6/§4 / exp §4 / intro | MHC-EN image collapse **0.734 → 0.599** | T2 MHC-EN img CLIP 0.7338 → 7B 0.5992 | encoder_swap_diagnosis_tables_out.json / ENCODER_SWAP_DIAGNOSIS.md | MATCH |
| 10 | anal §3.6 / exp §4 | 32B MHC-EN image AUC **0.608** | T2 MHC-EN img Qwen32B 0.6076 | encoder_swap_diagnosis_tables_out.json | MATCH |
| 11 | anal §3.6 / exp §4 | purity **+0.023 / +0.023 / +0.021** | HateMM/EN/ZH +0.023/+0.023/+0.021 | ENCODER_SWAP_DIAGNOSIS.md §2 | MATCH |
| 12 | anal §3.6 / exp §4 | text-AUC **+0.041 / +0.054 / +0.045** | §3: +0.041/+0.054/+0.045 | ENCODER_SWAP_DIAGNOSIS.md §3 | MATCH |
| 13 | anal §3.6 | HateMM hate-recall **+0.116 at zero cost** (DEV) | T3 dev: 0.8140→0.9302 = +0.1163; non-hate 0.766→0.766 | encoder_swap_diagnosis_tables_out.json T3 | MATCH (dev — see §3-B) |
| 14 | anal §3.6 | MHC-EN rotation **+0.040 hate / −0.036 non-hate** (DEV) | T3 dev: +0.040 / −0.036 | encoder_swap_diagnosis_tables_out.json T3 | MATCH (dev — see §3-B) |
| 15 | anal §3.6 | net-fix **+5 HateMM / −1 MHC-EN** | T4 net_fix 5 / −1 | encoder_swap_diagnosis_tables_out.json T4 | MATCH |
| 16 | anal §3.6 / exp §4 | MHC-EN fusion **net dev −0.012** | §1 machinery-val Δacc(dev) −0.012 | ENCODER_SWAP_DIAGNOSIS.md §1 | MATCH |
| 17 | anal §4 | encoder cross: **frozen-Qwen 0.870/0.861 vs CLIP floor** | T1.1: 0.870/0.861; floor 0.8279/0.8172 | PAPER_MASTER_TABLES.md T1.1 | MATCH |
| 18 | anal §3.7 / exp §7 / T5.2 | CTF flat **+0.0000 (HateMM) / −0.0029 (MHC)** | best_dacc 0.0 / −0.0028617 | CTF_G0COND_GATE_OUT.json | MATCH |
| 19 | anal §3.7 / exp §7 / T5.2 | CTF arc **−0.0049 / −0.0010** | delta best_dacc −0.0049354 / −0.0009539 | CTF_G0COND_GATE_OUT.json | MATCH |
| 20 | anal §3.7 | CTF calibration **accZA = 1.0** | label_accZA 1.0 (all 4 cells) | CTF_G0COND_GATE_OUT.json | MATCH |
| 21 | anal §3.7 (F35) | temporal cos **0.939** (diff-colour-same-pos) vs **0.674** (same-colour-diff-pos) | A3·B3=0.939; Blue A2·B0=0.674 | S2S_GATE0A_POSTMORTEM.md | MATCH |
| 22 | exp §7 / T5.2 | APX **best arm −0.0038, raw-88-d +0.0005** | audio_pca_k8 −0.0037603; audio_full_cvC +0.0004700 | APX_G0COND_GATE_OUT.json | MATCH |
| 23 | exp §7 / T5.2 | GIR r_cache **+0.0012 (HateMM) / −0.0051 (MHC)** | +0.0011751 / −0.0050874 | GIR_G0COND_GATE_OUT.json | MATCH |
| 24 | exp §7 / T5.2 | GIR r_field **+0.0000 / −0.0064**; 5 cells | 0.0 / −0.0063593; 5 gate_eval cells | GIR_G0COND_GATE_OUT.json | MATCH |
| 25 | exp §7 / T5.1 | B3 final-epoch **+0.0313 acc / +0.0453 mF1**, val-sel **+0.0246** | +0.0313/+0.0453; val-sel +0.0246 | B3_VERDICT_REVIEW.md | MATCH |
| 26 | exp §7 / T5.1 | B3 margin **+0.0013**; seed-2 **+0.0201** | §4a: +0.0013; seed-2 +0.0201 | B3_VERDICT_REVIEW.md | MATCH |
| 27 | exp §7 (F45) | B3 text AUC **0.802 → 0.847 → 0.925** | 0.8019 / 0.8468 / 0.9254 | b3_zh_lora_trainlog_parse_out.json / T2 | MATCH |
| 28 | exp §7 (F45) | B3 Pareto **+0.1111 / −0.0032** (LoRA test) | LoRA−CLIP: +0.1111 / −0.0032 | b3_zh_lora_trainlog_parse_out.json final_mean | MATCH |
| 29 | exp §7 (F45) | frozen rotation **+0.0741 / −0.0481, net −0.0112** | frozen−CLIP: +0.0741 / −0.0481 / −0.0112 | b3_zh_lora_trainlog_parse_out.json final_mean | MATCH |
| 30 | fig_data AUC (24 cells) | CLIP/7B/32B all datasets | byte-exact copy of T2 | encoder_swap_diagnosis_tables_out.json T2 | MATCH |
| 31 | fig_data AUC MHC-ZH LoRA | img 0.7141 / text 0.9254 / concat 0.9131 | deterministic recompute; text 0.925 in decomp | make_encoder_figures.py + B3_ZH_LORA_DECOMPOSITION.md | MATCH |
| 32 | fig_data pareto HateMM | **+0.1279 / +0.0078 / +0.0558** (test) | trainlog-derived 3-seed mean | make_encoder_figures.py (enc3s 12850) | MATCH |
| 33 | fig_data pareto MHC-EN | **+0.0953 / −0.0328 / +0.0062** (test) | trainlog-derived 3-seed mean | make_encoder_figures.py (enc3s/arcbase) | MATCH |
| 34 | fig_data pareto ZH-LoRA / ZH-frozen | +0.1111/−0.0032/+0.0313 ; +0.0741/−0.0481/−0.0112 | B3 final_means | b3_zh_lora_trainlog_parse_out.json | MATCH |
| 35 | exp §7 T5.1 / T5.2 (#15 A-line) | 91–93% constant, **264 GPU-h**, +0.040 bar | 91–93% 常数; 264 GPU-h; +0.040 | TERMINUS_round2 §1 + A_LINE_PAUSE_DECISION.md | MATCH |
| 36 | exp §7 T5.1 (#16 C1) | **+0.7** priced, DEV kNN **≈ −0.02** (job 13039) | +0.7 acc in-domain; −0.02 (13039) | C1_KILL_REVIEW.md + TERMINUS_round2 §1 | MATCH |
| 37 | exp §7 T5.1 (#17 C3-target) | oracle **+0.0487**, real best **+0.0094**, MHC anti-info | oracle +0.0487; real {+0.0094,+0.0077,−0.0111,−0.0136} | C3_REAL_PREDICTOR_PROBE.md | MATCH |

Also verified (#22 B4): draft "val-sel −0.0310 / final +0.0062 ≈ 1/5 bar" = B4_FORENSIC_RECON.md §(ii) exactly. MATCH.

**Numeric-provenance result: 37/37 MATCH — ZERO MISMATCH.** No transcription drift found; the fig_data.json AUC block
is a byte-exact copy of the committed T2 tensor, and the round-3 gate numbers reproduce the gate-JSON `best_dacc`
fields to 4 dp.

**Commit-hash existence (git cat-file -e):** all 11 hashes cited in the new prose EXIST —
`2c96ab6 7228373 8a48938 50f01b9 4358ca1 20c0bf2 0eb6d33 9c54faf b64a85b 0f43bdd d76e407`. None dangling.

---

## 2. CROSS-CHAPTER CONSISTENCY

**2.1 [FINDING — MEDIUM] The 13-route partition is stated with two different localization sub-counts; the
count-discipline note reconciles only the P9/P9b half.**
- `DRAFT_analysis_chapter.md` §1: "**Eleven routes** … main-table accuracy (…9 families…); **two** target temporal
  localization (the P6 scorer and its P10/P11 … thread)." → partition **(11 main + 2 loc) = 13**.
- `DRAFT_experiments_chapter.md` §7 count-discipline: "ten main-table accuracy rows plus **three** localization rows
  (P6 / P10 / P11); the analysis chapter's finer count instead splits P9/P9b to report eleven main-table routes."
  → partition **(10 main + 3 loc) = 13**, and it explains the 11-vs-10 main-table difference (P9/P9b split).
- `PAPER_MASTER_TABLES.md` tension #7 agrees with experiments: localization = **P6 + P10 + P11 = 3 rows**, main = 10.
- Consequence: both chapters correctly total **13**, but they reach it by opposite granularity choices on TWO route
  pairs — analysis §1 *splits* P9/P9b (main 10→11) AND *merges* P10/P11 (loc 3→2); experiments/tension #7 *merges*
  P9/P9b (main 10) AND *splits* P10/P11 (loc 3). The count-discipline note documents only the P9/P9b split, so a
  reader who combines "analysis reports **11** main-table" (experiments' own words) with "localization = **3**"
  (experiments' own words) computes 11 + 3 = **14**, not 13. The localization 2-vs-3 discrepancy is unreconciled.
- Not a numeric-provenance error (no primary contradicted); it is the exact "partition stated inconsistently"
  risk. Suggested fix (through team-lead): have the experiments §7 note add "…and the analysis chapter groups
  P10/P11 as one localization thread (two localization routes), so its partition is 11+2 = 13," OR make §1 say
  "…**three** localization routes (P6, P10, P11)."

**2.2 [CLEAN] Round-2 extension counting.** Experiments §7 / T5.1: "seven sprint negatives (#15–21) plus a pre-GPU
forensic close (#22, B4) and one marginal positive held pending a novelty ruling (B3)." Verified verbatim against
`TERMINUS_round2_mllm_plus3.md` §1 ("7 条路线 第 15–21 条"), §7 ("B4 … 第 22 条"), §6/§8 (B3 held, D7). Table 4 has
exactly 8 rows (#15–22). The cumulative "21 pre-registered negatives" (14 prior + 7 sprint) is consistent; #22 (B4)
is the pre-GPU close appended after the "21" tally. No conflict.

**2.3 [CLEAN] Round-3 counting.** Experiments §7: "round 3 adds six directions" (Table 5 = S2S, CTF, APX, AVC, W2-A,
GIR = 6) and "retired six recon-/triage-stage companions" (W2-B, W2-E, W2-C, C5, R3-C3geo, B5 = 6). Matches
`TERMINUS_round3_mllm_plus3.md` §0/§9 ("six kills in one day") and the Axis A–F member lists. "wave-3 pool is empty /
every injection point closed" matches TERMINUS_round3 §0 Status line ("wave-3 pool is empty … unconditional").

**2.4 [CLEAN] B3 held-pending status is uniform; never called a pass.** Every occurrence keeps the scoped language:
exp §7 "one marginal positive **held pending** a user novelty ruling"; the B3 paragraph reports the verbatim binding
`final-epoch: PASS (MARGINAL); val-selected: FAIL` and adds "Whether B3 counts toward the goal's *novel* clause is an
explicit user ruling; it is not folded into any main table"; T5.1 "唯一 marginal 正例,pending novelty 裁决,不并入主表."
No chapter upgrades B3 to a goal-level pass, and no chapter that calls a route a KILL is contradicted elsewhere
(S2S/CTF/APX/W2-A/GIR are KILL in every appearance; B1/B2/B4 negatives consistent). Consistent with the newest
primary (B3_ZH_LORA_DECOMPOSITION.md `d76e407`), which itself frames the LoRA-family novelty acceptance as still a
user call (option c) even though the general encoder-class D7 ruling is negative.

**2.5 [CLEAN] Verdict-commit hashes in prose.** All 11 exist (see §1). Each cited file basename also resolves in
`refine-logs/` (A_LINE_PAUSE_DECISION.md lives under `refine-logs/lb_scgp_global/`; a basename citation, not wrong).

---

## 3. CLAIM-STRENGTH

**3-A [CLEAN] Intro limitation edit (247fcaf) is claim-neutral and faithful.** The inserted sentence
("This ceiling is now mechanistically attributed … a collapsed Qwen image stream under equal-weight fusion plus a
label-limited error core, so no encoder upgrade converts there") adds a mechanistic gloss to the *existing*
MHClip-EN ceiling claim (≈0.78–0.80, does not cross 0.85). It neither weakens nor strengthens the performance claim
(the ceiling numbers are untouched), sits correctly in the EN paragraph (the collapse is EN-specific), and the
attribution is faithful to `ENCODER_SWAP_DIAGNOSIS.md` (`8a48938`). No numbers introduced.

**3-B [FINDING — MEDIUM] §3.6 reports DEV per-class recall unlabeled while the companion figure reports TEST — the
two footings are not cross-labeled.** The analysis §3.6 mechanism prose uses the DEV concat-vote per-class recalls
from `ENCODER_SWAP_DIAGNOSIS.md` T3: HateMM "+0.116 at zero non-hate cost" and MHC-EN "+0.040 hate / −0.036 non-hate."
Only the adjacent "net **dev** −0.012" carries a dev label; the two recall figures do not. `fig_pareto_rotation` (and
`fig_data.json`, note field) report the **TEST final-epoch 3-seed** deltas for the same phenomenon: HateMM
+0.1279/+0.0078, MHC-EN +0.0953/−0.0328. Each value is individually faithful to its own primary (dev = T3; test =
trainlog-derived), so this is NOT a MISMATCH — but a reader flipping between §3.6 prose and the figure sees
different numbers for the "same" move, and the MHC-EN hate-recall in particular differs materially (dev **+0.040**
vs test **+0.095**, ~2.4×). The figure is explicitly labeled "TEST"; the prose recalls are not labeled "dev."
Suggested fix (through team-lead): label the §3.6 per-class recalls as dev-vote (e.g. "hate-recall +0.116 (dev) …")
and/or note the figure shows the test-side of the same effect, so the two are not conflated.

**3-C [FINDING — LOW · pre-existing, adjacent to an edited paragraph] "+4.2 macro-F1" in §4 mislabels the metric.**
`DRAFT_analysis_chapter.md` §4 (line 267): "Qwen2.5-VL features beat CLIP on HateMM by **+4.2 macro-F1**." Against
T1.1 (frozen-Qwen 0.870/0.861 vs CLIP floor 0.8279/0.8172): the **accuracy** delta is 0.870−0.8279 = **+0.0421**
(≈ +4.2), while the **macro-F1** delta is 0.861−0.8172 = **+0.0438** (≈ **+4.4**). So "+4.2" is the accuracy gap
mislabeled as macro-F1; the macro-F1 gap is +4.4. This line was introduced in commit `0587bf4` (original analysis
draft) and is **NOT a regression from the three audited commits** — but it sits inside the §4 encoder paragraph that
`dc224b9` appended text to, so it is surfaced here for completeness. Both metrics do cross 0.85, so the crossing
claim is unaffected. Suggested fix (through team-lead): change to "+4.2 accuracy / +4.4 macro-F1" or "+4.4 macro-F1."

**3-D [CLEAN] §3.6/§3.7 stay within negative-evidence framing.** Structural law I ("a signal being measurably better
is necessary but not sufficient") and law II ("cumulative-causal three-level closure … a concrete, transferable
caution") are framed as characterizations of the negative graveyard and a methodological caution, matching the
primaries' "explanatory, not generative" language (ENCODER_SWAP_DIAGNOSIS §7; TERMINUS_round3 §0). Neither is
presented as a positive main-table result. The §4 encoder addition keeps the encoder as the "frozen-encoder identity,
not the new method role," now "mechanistically bounded (will not generalise to MHC)" — a scoping, not an upgrade.

**3-E [CLEAN] Figure captions/annotations match committed data.** `make_encoder_figures.py` reads every number from
`encoder_swap_diagnosis_tables_out.json` (T2) and `b3_zh_lora_trainlog_parse_out.json` (no hand-typed values); the
Δacc annotations render the computed cell values (`{da:+.4f}`), and the fig caption is explicitly labeled
"(TEST, final-epoch, 3-seed mean)". The MHC-EN "image stream collapses to chance; 32B does not fix it" callout
matches T2 (0.7338 → 0.5992 → 0.6076). Consistent.

---

## 4. OVERALL VERDICT

**CLEAN on numeric provenance — 37/37 checks MATCH, ZERO MISMATCH; all 11 cited commit hashes exist.** The fold of
rounds 2–3 into the analysis/experiments chapters, master-table T5, and the two figures transcribes every number
faithfully from its primary verdict/gate/diagnosis record.

Issues, ranked by severity (all consistency/labeling, NONE a numeric error; report-only, fixes route through the
team lead):

1. **MEDIUM (§2.1)** — 13-route partition: analysis §1 counts localization as **2** (P6 + P10/P11 thread) vs
   experiments §7 / tension #7's **3** (P6/P10/P11); the count-discipline note reconciles only the P9/P9b (11-vs-10)
   half, leaving "11 main + 3 loc = 14" as an apparent contradiction with the stated total of 13.
2. **MEDIUM (§3-B)** — §3.6 mechanism prose reports DEV per-class recalls (HateMM +0.116/0; MHC-EN +0.040/−0.036)
   unlabeled, while the companion figure reports the TEST deltas (+0.1279/+0.0078; +0.0953/−0.0328); the MHC-EN
   hate-recall differs ~2.4× between footings. Both faithful to source; the prose just needs a dev label.
3. **LOW · pre-existing (§3-C)** — §4 "+4.2 macro-F1" is the accuracy delta; the macro-F1 delta is +4.4. Introduced
   in `0587bf4`, not by the audited commits; surfaced because it sits in an edited paragraph.

No MISMATCH rows to report.
