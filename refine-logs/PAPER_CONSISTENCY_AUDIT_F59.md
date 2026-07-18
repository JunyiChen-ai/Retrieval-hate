# PAPER CONSISTENCY AUDIT — round-4 → F59 fold

**Auditor:** independent 0-context paper-consistency auditor. **Date:** 2026-07-18.
**Scope:** everything folded into `research-wiki/` since commit `0180521` (inclusive) through F59 —
commits `0180521` (F47–F51 fold), `6de8af4` (F53 fold), `9a09406` (F54–F56 fold), `87520fb` (F58
locus errata), `a31c80b` (F59 rep2 fold), plus the round-4 exp notes and the `D7_RULING_DOSSIER.md`
§7/§8 addenda.
**Discipline:** ZERO GPU / ZERO Modal / ZERO test-touch / ZERO user interaction. Read-only against the
primary verdict/gate records; five numbers spot-checked all the way to the raw trainlogs (§4). No push.

**Audited draft sections:** `DRAFT_analysis_chapter.md` (§3.6 / §3.8 / §3.9 / §4), `DRAFT_experiments_chapter.md`
(§7 Tables 4–8 + rep2 + premise-(d)), `PAPER_MASTER_TABLES.md` (T5.1–T5.4, PUR-1…4, tension ledger),
`experiments/{exp-router-r4,exp-fa-r4,exp-lora-hatemm,exp-premise-d,exp-cand2-curriculum}.md`,
`refine-logs/D7_RULING_DOSSIER.md` §7/§8.

**Primary sources cross-checked against:** `LORA_HATEMM_VERDICT_REVIEW.md` (`6b8f634`),
`HATEMM_LORA_STREAM_DECOMP.md` (`51eb95b`), `ROUTER_GATE_RECORD.md` (`30d0ee1`),
`FA_GATE_RECORD.md` (`e0877c9`), `PREMISE_D_GATE_RECORD.md` (`6e6061b`),
`CAND2_VERDICT_REVIEW.md` (`546acc5`), `CAND2_REP2_VERDICT_REVIEW.md` (`aa48275`).

---

## 0. HEADLINE

- **Number mismatches (pure-typo, all-fidelity): 0.** Every number in the audited draft sections matches
  its primary source to 4dp, including five spot-checked all the way down to the raw trainlogs.
- **Fixes applied: 4** — (i) the two pre-authorized single-draw→pooled-weakly-hardened phrasing fixes in
  `DRAFT_analysis_chapter.md` (lines 415 and 429), + a companion citation extension so the introduced
  rep2 claim is sourced; (ii) the two F58 locus fixes in `experiments/exp-cand2-curriculum.md`
  (lines 56, 119), applied in the errata follow-through commit after the orchestrator authorized them
  (2026-07-18, same scope as `87520fb`).
- **Flags left (judgmental, not fixed): 0** — the two locus leftovers below were flagged in the first
  pass, then authorized and fixed in the follow-through commit.
- **Claim consistency (b)/(c)/(d): PASS.** Verbatim F56 verdict blocks preserved; protocol qualifiers
  (final-epoch + ZH-marginal) present at every 2-dataset performance statement; T4 13-route count
  untouched; extension counts and the tension ledger self-consistent.
- **Cross-doc: PASS.** D7 dossier §7/§8 numbers match F58/F59 records; every exp-note provenance chain
  is complete and every DOC commit-hash pointer resolves to the correct landing commit.

---

## 1. FINDINGS TABLE

| # | file:line | issue | severity | disposition |
|---|---|---|---|---|
| 1 | `research-wiki/DRAFT_analysis_chapter.md:415` | §3.9 own-voice: HateMM K-C2-2 add described as "a single curriculum draw" — pre-rep2 status, now superseded by F59 pooled-weakly-hardened. | medium | **FIXED** (pre-authorized) — → "3/3 on the draw-1 curriculum; pooled weakly-hardened across two draws, 5/6 sign, per-draw 3/3 gate not met"; +F59 citation on L416. |
| 2 | `research-wiki/DRAFT_analysis_chapter.md:429` | Phase-diagram bullet: cand-2 add "on HateMM val-selected only (single-draw)" — pre-rep2 status. | medium | **FIXED** (pre-authorized) — → "(pooled weakly-hardened across two draws, 5/6 sign; per-draw 3/3 gate not met)". |
| 3 | `research-wiki/experiments/exp-cand2-curriculum.md:56` | §0 own-voice locus claim: "can add only HateMM **(image-borne, inherited)**". F58 (`51eb95b`) REFUTED the image-borne locus (HateMM's convertible signal is **text-carried**, frozen-swap-sufficient). Contradicts the errata'd analysis §3.9 L430–431 ("HateMM is inherited (frozen-swap-sufficient, its convertible signal text-carried)"). | medium | **FIXED** (flagged first pass → authorized 2026-07-18 → follow-through commit): → "(inherited, frozen-swap-sufficient; its convertible signal is text-carried, not image-borne — F58, `51eb95b`)". |
| 4 | `research-wiki/experiments/exp-cand2-curriculum.md:119` | §3 own-voice: HateMM K-C2-2 pass lands "on the **hold/image-inherited leg** (F0.4)". "image-inherited" is the F58-refuted locus term (the inheritance is real; its *image* attribution is not). | medium | **FIXED** (flagged first pass → authorized 2026-07-18 → follow-through commit): → "hold/inherited leg (F0.4; frozen-swap-sufficient, convertible signal text-carried — F58, `51eb95b`)". |

**Notes on the two flags (§ claim-consistency 2a).** The `87520fb` F58 errata correctly scrubbed the
image-borne / image-inherited / secondary-modality locus phrasing from every *primary paper* section it
touched — `DRAFT_analysis_chapter.md` §3.9 (all cross-refs), `DRAFT_experiments_chapter.md` §7,
`PAPER_MASTER_TABLES.md` (PUR/T5), and `experiments/exp-lora-hatemm.md` — and from the intro / limitations /
method / abstract drafts (grep-clean). The **only** research-wiki survivors are the two lines above, in the
one exp-note the errata commit did not open (`exp-cand2-curriculum.md`, later touched by `a31c80b` for the
rep2 fold but not for the locus errata). Both are the note's own analytical voice, not `​`​`-fenced verbatim
verdict blocks, so correcting them is defensible; they were FLAGGED in the first pass (not fixed) because the
errata-scope call — whether the F0.4-pre-declared framing in a per-experiment note must be retro-corrected —
was judgmental, not a pure-typo mismatch. **The orchestrator then authorized both (2026-07-18, same scope as
`87520fb`); they are now FIXED in the errata follow-through commit, citing `51eb95b`, with the pre-declared
"By F44/F45 modality-locus arithmetic" basis preserved and the two `​`​`-fenced F56 verdict blocks untouched.**
The refine-logs records that retain "image-borne" (`CAND2_*_PREREG`,
`*_RECON`, `WAVE5/6_*`, `LORA_HATEMM_PREREG/FORENSIC_RECON`, `TIE_BRANCH_RECON`, and the F53
`LORA_HATEMM_VERDICT_REVIEW.md` itself) are **correctly** left as-is: they are pre-F58 historical records,
not paper drafts, and F58's own synthesis names them as the framing it supersedes.

---

## 2. NUMBER-FIDELITY VERIFICATION (all PASS — representative cells)

Every audited number was checked against its source; a representative slice below. No mismatch found.

**LoRA-HateMM / F53** (`LORA_HATEMM_VERDICT_REVIEW.md` → analysis §3.9, exp §7 Table 7, exp-lora-hatemm,
PUR-2/PUR-3): HateMM val-sel mean Δacc **+0.0419** / ΔmF1 **+0.0460** (3/3); final-ep **+0.0573 / +0.0682**
(3/3); LoRA final 0.8698/0.8618 ≥ frozen-Qwen 0.8682/0.8591 (KS-2 +0.0015/+0.0026); CLIP floors
0.8202/0.8085 (val-sel) · 0.8124/0.7936 (final); MHC-EN FAIL both (val-sel −0.0021 / final +0.0000);
cushion ≈ 9× B3's +0.0013; eval_loss 0.1084. All match to 4dp.

**F58 stream decomposition** (`HATEMM_LORA_STREAM_DECOMP.md` → analysis §3.9, exp-lora-hatemm §5, D7 §7):
text-only AUC 0.847/0.837 → 0.888/0.875 → 0.920/0.899 (CLIP/frozen/LoRA, tr/dv); image ΔAUC(LoRA−frozen)
+0.0045 tr / +0.0062 dv; frozen−CLIP +0.0558 acc (hate +0.128 at +0.008 non-hate); LoRA−frozen +0.0015
final / −0.0108 val-sel. All match.

**Router / F47** (`ROUTER_GATE_RECORD.md` → analysis §3.6/§3.8, exp-router-r4, exp §7 Table 6, T5.3):
oracle headroom +0.1083 MHC-EN (0.1125/0.1250/0.0875) / +0.0498 HateMM; deployable +0.0000; dev-CV
−0.0458 CI[−0.0875,0], linear −0.0333; perm-null p95 +0.0042 (p=0.97); CLIP LOO 0.998 vs Qwen 0.800;
0/109·0/102·0/92; accZA 1.000; disagreement 20/23/20. All match.

**FA / F48+F50** (`FA_GATE_RECORD.md` → analysis §3.6, exp-fa-r4, exp §7 Table 6, T5.3): dev AUC peak
0.8982; A2 w=0.15 Δacc +0.050 / +0.120 hate / +0.018 non-hate; boot CI [−0.0625,+0.150]; oracle edge
+0.025 < +0.03; selection-null p=0.766; A1 w=0.5 +0.040/−0.036; concat proxy −0.0125 vs F44 −0.012;
HateMM control +0.0467. All match.

**Premise-(d) / F55** (`PREMISE_D_GATE_RECORD.md` → analysis §3.6, exp-premise-d, exp §7, T5.4(b)):
A2F↔FA-A2 max|diff| 0.000000; peak AUC 0.8982→0.8698 (−0.0284); max d_oracle +0.0250 < +0.03; A2L w=0.20
Δacc +0.050 / Δnon-hate −0.0545; boot CI [−0.0503,+0.1625]; selection-null p=0.7532; HateMM control
+0.0467; A0 0.7625 (oracle 0.800); sixth no-conversion datum (P3·S2S-F37·W2-A-F42·router-F47·FA-A2-F50).
All match.

**cand-2 / F56** (`CAND2_VERDICT_REVIEW.md` → analysis §3.9, exp §7 Table 8, exp-cand2, T5.4(a)):
ZH val-sel 0.8255/0.7947 (+0.0179/+0.0271, −0.0067 1/3); ZH final 0.8523/0.8249 (+0.0380/+0.0529, +0.0067
2/3); HateMM val-sel 0.8775/0.8711 (+0.0573/+0.0626, +0.0155 3/3 pass); HateMM final 0.8791/0.8726
(+0.0667/+0.0790, +0.0093 3/3 tie); K-C2-0 ZH 0.2073/0.5634/0.6667/2.11× · HateMM 0.1935/0.6497/0.6756/2.08×;
ZH-robustness not strengthened. All match.

**rep2 / F59** (`CAND2_REP2_VERDICT_REVIEW.md` → analysis §3.9 [now cited], exp §7 rep2 para, exp-cand2 §8,
T5.4(a-rep2), D7 §8): draw-2 val-sel [+0.0139, −0.0047, +0.0233] mean +0.0108 (ΔmF1 +0.0120), sign 2/3 →
K-REP-1 NOT-PASS; KS-REP not fired; pooled +0.01317, 5/6 → HARDENED; final-ep +0.0140, 3/3; verdict
WEAKLY-HARDENED. All match.

*(Immaterial, non-actionable: `CAND2_REP2_VERDICT_REVIEW.md` §4 records that draw-1 mean ΔmF1 re-derives to
+0.0165 vs the frozen banked +0.0166 — a 0.0001 rounding of a non-binding quantity. The drafts uniformly use
the frozen banked +0.0166, which is the correct primary value; no draft mismatch.)*

---

## 3. CLAIM-CONSISTENCY (b)/(c)/(d) — PASS

**(b) Verbatim F56 preservation.** The `​`​`-fenced F56 verdict block ("val-selected: PASS (K-C2-1, hold) ·
K-C2-2: pass (single-draw caveat, F0.2)") is preserved byte-for-byte everywhere it is quoted —
`DRAFT_experiments_chapter.md:613`, `PAPER_MASTER_TABLES.md:312`, `exp-cand2-curriculum.md:27`. The two fixes
touch only own-voice prose, never a quoted block.

**(c) Protocol qualifiers.** The 2-dataset performance statement carries its final-epoch qualifier + ZH-marginal
note at every occurrence: analysis §3.9 (L383–386), exp §7 (L546–550, "protocol-qualified"), exp-lora-hatemm
§7 (L160–164), D7 dossier §7 (L352–353), PUR-2/PUR-3. No unqualified "passes on two datasets" claim survives.

**(d) Count discipline.** T4's 13-route campaign count is untouched by every sprint (analysis §1 / exp §7
L393–419 / master-tables tension #7). The P9/P9b split (11 fine-grained vs 10 T4 accuracy rows, same results)
is coherent (#7). The round-4 ordinal tension (findings.jsonl 22nd/23rd/24th vs directions_tried B4=#22) is
disclosed, not silently reconciled, and the paper uses the round-by-round framing without minting a contested
grand total (#9, incl. the F53 and F55/F56 addenda). Extension bookkeeping self-consistent: round-2 #15–22 +
B3 marginal positive; round-3 six axes; round-4 = router(F47)/FA(F50)/MJ(F49)/wave-5(F51)/line-A(F53) +
closing pair cand-2(F56)/premise-(d)(F55); none a main-table route.

---

## 4. TRAINLOG SPOT-CHECKS (5/5 bit-exact)

Read directly from `slurm/logs/*.trainlog` (zero-GPU); each reproduces the value carried through the records
into the drafts:

| # | quantity | draft/record value | trainlog line (verbatim) |
|---|---|---|---|
| 1 | LoRA-HateMM s0 val-sel (ep19) | 0.8605 / 0.8521 | `enc3s_HateMM_…-LoRA_HF_seed0_13235`: `Epoch 19 macroF1: 0.8521 … acc: 0.8605` |
| 2 | HateMM-curric s0 final (ep29) | 0.8791 / 0.8730 | `…-LoRA-curric_HF_seed0_13241`: `Epoch 29 macroF1: 0.8730 … acc: 0.8791` |
| 3 | rep2 s1 val-sel (ep16) | 0.8651 / 0.8574 | `…-LoRA-curric-rep2_HF_seed1_13246`: `Epoch 16 macroF1: 0.8574 … acc: 0.8651` |
| 4 | generic-LoRA s1 val-sel (ep14) | 0.8698 / 0.8620 | `…-LoRA_HF_seed1_13235`: `Epoch 14 macroF1: 0.8620 … acc: 0.8698` |
| 5 | CLIP floor s0 val-sel (ep24) | 0.8279 / 0.8172 | `…clip-vit-large-patch14-336_HF_seed0_12850`: `Epoch 24 macroF1: 0.8172 … acc: 0.8279` |

---

## 5. PROVENANCE / DOC-HASH CHECK (all correct)

Each DOC commit-hash pointer in the audited sections resolves to the record's actual landing commit
(`git log --diff-filter=A`): ROUTER `30d0ee1`, FA `e0877c9`, PREMISE_D `6e6061b`, STREAM_DECOMP `51eb95b`,
CAND2 `546acc5`, CAND2_REP2 `aa48275`, LORA_HATEMM `6b8f634`, B3_ZH_LORA_DECOMPOSITION `d76e407` — all match.
Exp-note ceremony chains complete (exp-lora-hatemm `edeaedc→3ebd880→2e41332→8de0991→56a732a→6b8f634`;
exp-cand2 draw-1 `7087b5a→76ef0e2→c1315cb→7804324→1ea3c13→546acc5` + rep2
`2d15ffb→e2aee03→6c11988→d06ad07→aa48275`; exp-premise-d / exp-fa-r4 / exp-router-r4 all cite record + script
sha + RNG).

---

## 6. AUDITOR STATEMENTS

No GPU / Modal / SLURM / test-touch spent; only banked records, drafts, and completed-run trainlogs read.
Nothing pushed. Mutations across the two commits: (1) audit commit `734c389` — the two pre-authorized
single-draw phrasing fixes (+ their companion rep2 citation) in `DRAFT_analysis_chapter.md` and this audit
file; (2) errata follow-through commit — the two authorized F58 locus fixes in
`experiments/exp-cand2-curriculum.md` (lines 56, 119) + this report's disposition update. No number was
altered; no verbatim F56 verdict block was touched.
