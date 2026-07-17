# D7 NOVELTY RULING — DECISION-SUPPORT DOSSIER

**Author:** D7-dossier author (evidence-organizing only; ZERO advocacy, ZERO GPU, ZERO test-touch).
**Date:** 2026-07-18.
**Purpose:** assemble, with source-pinned provenance, everything the USER needs to rule on the pending
**D7 novelty question** for `goal_mllm_plus3` (hateful-video detection). This document takes **no side**
and issues **no recommendation** — it ends at the decision matrix (§5) plus a factual open-evidence
register (§6). Every number is transcribed from a re-read primary record; each is cited to its
commit/file. Disputed, marginal, or convention-dependent items are flagged inline as **[MARGINAL]**,
**[DISPUTED]**, or **[PENDING]**.

---

## 1. THE RULING QUESTION (stated precisely, with D7 precedent quoted verbatim)

### 1.1 The goal being ruled against (frozen, user-set)

An MLLM **meaningfully AND novelly** integrated into the retrieval-contrastive method, delivering a
**substantial performance improvement**, operationalized as the user's loop bar: **+0.03 acc AND +0.03
macro-F1 on ≥2 of the 3 datasets** (HateMM, MHC-EN, MHC-ZH), under the frozen constraint box (7B-only,
no gold-in-method, no OCR, single-dataset own-train split, no cross-seed ensembles, no external APIs).
Novelty is judged **strictly within the hateful-video-detection field**; prior work in other domains
(incl. hateful-meme RGCL/RA-HMD, which this project adapts from) counts as *inspiration*, not
novelty-defeating prior art.
[source: `refine-logs/TERMINUS_round3_mllm_plus3.md:7`; `research-wiki/query_pack.md:16`]

### 1.2 The D7 precedent — VERBATIM

D7 was raised and **resolved-negative** on 2026-07-14. The binding record is
`research-wiki/DECISION_MEMO_pending.md` §"D7" (lines 74–97) and its origin
`research-wiki/TERMINUS_round2_mllm_plus3.md` §8 (lines 74–85).

**User ruling, verbatim (2026-07-14 evening), quoted in both records:**

> - 「哎呀,这个 encoder swap 肯定不算 novelty 啊」
>   *(roughly: "oh, this encoder swap obviously does not count as novelty")*
> - 「我不管,反正这个做不出来就一直做,直到做出来为止。」
>   *(roughly: "I don't care — if it isn't achieved, keep working until it is")*

**Orchestrator interpretation, binding (verbatim from `DECISION_MEMO_pending.md:80–85`):**

> **编排解读(binding):D7 = RESOLVED-NEGATIVE。** encoder-class 杠杆——frozen swap、LoRA-adapted
> swap,及推而广之的通用决策规则校准(如 B5)——**均不满足 goal 的 novelty 子句**;它们保留为
> **合法的性能 / 消融 / 诊断素材**。TERMINUS 选项 (c)「goal 重议」= **DEAD**;goal 现要求一个
> **NOVEL MECHANISM**(novelty 在 hateful-video 检测范围内判定)× MLLM-integrated × 交付 **≥+3 acc**。

**In English (this dossier's non-binding gloss, kept for readability):** D7 = resolved-negative.
Encoder-class levers — *frozen encoder swap*, *LoRA-adapted encoder swap*, and by extension generic
decision-rule calibration (e.g. B5) — do **NOT** by themselves satisfy the goal's novelty clause. They
remain legitimate **performance / ablation / diagnostic** material. The "renegotiate the goal" escape
(TERMINUS option c) is dead; the goal now requires a **NOVEL MECHANISM** (novelty judged within
hateful-video detection) × MLLM-integrated × delivering ≥+3 acc.

### 1.3 The precise question now on the table

The performance conjunct has since been **met on 2 datasets by one encoder-class lever** (LoRA-adapted
Qwen encoder: HateMM both protocols + MHC-ZH final-epoch — see §2.2). D7's standing language classes
that exact lever as **non-novel by itself**. Therefore the live question is:

> **Does the method bundle actually delivered (§2) constitute a "novel MLLM integration" in the sense
> the goal requires — either (a) the bundle as a whole clears novelty, (b) it clears only if the
> pending cand-2 memory→adaptation coupling verdict upgrades the LoRA leg from "generic" to
> "memory-coupled", or (c) it does not clear, so the goal remains unmet and needs a new axis or
> renegotiation?**

Two sub-facts make this a genuine ruling rather than a lookup:
- D7's original wording targeted encoder-class levers **standalone**; it did not rule on whether the
  *full four-pillar bundle* plus the *negative-results contribution* clears novelty collectively.
- cand-2 is a **pre-registered attempt to move the LoRA leg out of "generic encoder-class"** into
  "retrieval-memory-coupled adaptation curriculum," whose verdict is **[PENDING]** (§6). The prereg
  itself declares "**the ruling remains the USER's**" regardless of its own PASS/FAIL
  [`refine-logs/CAND2_CURRICULUM_PREREG.md:461,468`].

---

## 2. THE METHOD BUNDLE AS IT WOULD BE CLAIMED (with per-component evidence status)

Source: `research-wiki/DRAFT_intro_related_limitations.md` §Contributions (lines 63–107),
`research-wiki/DRAFT_analysis_chapter.md` §4, `research-wiki/PAPER_MASTER_TABLES.md`. The paper frames
**four capability contributions (C1–C4)**, an **MLLM-roles framing (C5)**, and a **negative-results
contribution (C6)**.

| # | Claimed component | Evidence status (validated where / refuted where) | Source |
|---|---|---|---|
| **C1** | Retrieval-guided-contrastive + kNN core, ported to hateful video, beating MoRE head-to-head | **Validated (performance)** on all 3 benchmarks: +5.6/+8.7/+6.7 acc and +6.2/+22.9/+9.7 mF1 (HateMM/EN/ZH) vs stronger MoRE variant. *Mechanism* claim, not a "first to bring retrieval to hateful video" claim. | `BASELINE_MoRE_rerun.md`, `PAPER_MASTER_TABLES.md` T1.2 |
| **C2** | Zero-retrain updatable memory + O(1) temporal-recalibration protocol | **Validated**: beats target-majority on 5/6 informative cross cells; temporal drift (MHC-EN −0.084 mF1) fully recovered from k=20 new-period labels (0.6273→0.7336), ZH no-drift control | `exp-cross-dataset-transfer.md`, `EVAL_temporal_memory_W4.md` |
| **C3** | Consensus denoising of label-inherited segment supervision | **Validated on Chinese as a repair** (removes −0.066 mF1 hole); **refuted / does-not-transfer on English**, with full attribution chain pinning residual on segment-supervision channel itself (language-conditioned) | `exp-consensus-zh-seeds.md`, `EXP_mm_segment_keys.md` |
| **C4** | Auditable + human-editable archive memory | **Validated as integrity/controllability, NOT accuracy**: audit faithful on 77% of records; deleting 2 human-flagged entries lifts MHC-EN 0.8075→0.8199 (zero retrain, human-in-loop). **AUTO two-vote repair = NEGATIVE** (survives only as guard-rail veto); archive-as-key accuracy claims **WITHDRAWN** (selection artifacts) | `AUDIT_archive_faithfulness.md`, `DEMO_memory_editing.md`, `EXP_auto_memory_repair.md` |
| **C5** | MLLM's three earned roles + one explicit non-role | **Encoder** (frozen Qwen > CLIP on HateMM, crosses 0.85 — but **D7-dead standalone**); **span-free localization scorer** (P6/P10-b, paired-significant, §3.3); **guard-rail/audit**; and **NO main-table-accuracy role** at 7B–72B | `CAMPAIGN_mllm_method_role.md`, `DRAFT_analysis_chapter.md` §4 |
| **C6** | Negative results as a first-class methodological contribution | **Validated as evidence**: 13-route pre-registered campaign, 11 main-table-accuracy routes all honest kills or within-noise, each guard-backed; two transferable mechanistic conclusions | `PAPER_MASTER_TABLES.md` T4, `DRAFT_analysis_chapter.md` |

**MLLM roles earned (C5 detail, for the ruling):**
- **Encoder via LoRA adaptation / frozen swap** — performance-validated, but **the exact lever D7
  names as non-novel** (§2.2). This is the crux.
- **Localizer (P6 / P10-b)** — paired-significant but MODEST; the one lane where the MLLM exceeds
  memory/random (§3.3). Test-touch spent once.
- **Guard-rail / audit** — validated as controllability, not accuracy (C4).

### 2.2 The performance evidence that gates D7 (source-pinned, re-read)

**HateMM — LoRA-adapted Qwen encoder vs frozen-CLIP** (F53, `LORA_HATEMM_VERDICT_REVIEW.md` `6b8f634`,
job 13235, hash-verified vs frozen prereg):
- val-selected: LoRA **0.8620/0.8545** vs CLIP **0.8202/0.8085** = **+0.0419 acc / +0.0460 mF1**, sign **3/3 PASS**.
- final-epoch: LoRA **0.8698/0.8618** vs CLIP **0.8124/0.7936** = **+0.0573 acc / +0.0682 mF1**, sign **3/3 PASS**.
- **KS-2 honesty flag NOT tripped**: final LoRA 0.8698 ≥ frozen-Qwen final 0.8682 → the pass is **not
  merely inherited from the frozen-Qwen image stream**; LoRA matches/exceeds frozen-Qwen.
  [`LORA_HATEMM_VERDICT_REVIEW.md` §1.1/§2.1; F53]

**MHC-ZH — LoRA-adapted Qwen encoder vs frozen-CLIP** (B3, `B3_VERDICT_REVIEW.md`/`B3_PREREG_REVIEW.md`
§2.2, job 13150, G-repro bit-exact vs arcbase):
- final-epoch: **+0.0313 acc / +0.0453 mF1**, sign **3/3 PASS (MARGINAL)**.
- val-selected: **FAIL** (+0.0246 acc). Margin over the +0.030 bar = **+0.0013 acc, structural**; seed2
  sits below the per-seed bar. **[MARGINAL]**
- Frozen-Qwen (no LoRA) on ZH = **−0.0112** ⇒ the gain is **LoRA adaptation, not encoder identity**.

**MHC-EN — LoRA-adapted Qwen encoder vs frozen-CLIP** (B4 closure arm, bundled in F53):
- **FAIL both protocols**: val-sel −0.0021 acc / final +0.0000 acc (mF1 +0.0399 / +0.0187). Formally
  closes the EN-LoRA-encoder cell; matches the B4 seed0 anchor exactly. [`LORA_HATEMM_VERDICT_REVIEW.md` §2.2]

**Frozen-Qwen encoder swap — HateMM** (the separate F24-ruled lever): paper T1.1 headline stated as
**+4.2 acc / +4.4 mF1** (frozen-Qwen 0.870/0.861 vs CLIP floor, crosses 0.85)
[`DRAFT_analysis_chapter.md` §4]. **[DISPUTED — floor convention]** the same verdict review's 3-seed
paired floors (job 12850) give frozen-Qwen − frozen-CLIP = **+0.0527/+0.0563 (val-sel), +0.0558/+0.0655
(final)** [`LORA_HATEMM_VERDICT_REVIEW.md` §1.1], consistent with the exp-encoder-3seed "+5.3–5.6 acc"
figure (`040adb8`). The two numbers differ by floor/protocol convention; this is **not load-bearing for
D7** because the frozen swap is D7-dead regardless — flagged only for paper-table reconciliation.

**Performance-conjunct ledger (verbatim frame, F53):** under the **final-epoch** protocol, one lever
(encoder-level LoRA-Qwen) meets the bar on **2 datasets** — HateMM (+0.0573/+0.0682, SOLID) AND MHC-ZH
(+0.0313/+0.0453, MARGINAL). Under the **val-selected** protocol, **HateMM only** (ZH val-sel FAILs).

---

## 3. CASE THAT NOVELTY IS SATISFIED (steelman, from the record)

*Constructed as the strongest good-faith case the banked record can support. Each point is cited; none
is endorsed here.*

**3.1 First working video port of LMM-RGCL (systems/integration first).** P9b delivered the **first
working port of RA-HMD's (released-broken) LMM-RGCL stage-2 to video** — 5 fork fixes incl. a
never-reloaded classifier, bs=1 in-batch degeneration, 4-frame/bs4 fix
[`EXP_p9_lmm_rgcl_video.md:374`, `455e666`/`4d28655`]. *Load-bearing caveat for this case:* the port
was **net-FAIL on accuracy** (head↔memory redistribution ±1.8pt, 0/12 cells beat the protocol-matched
floor) — the "first" is an integration/engineering first, not a performance win
[`CAMPAIGN_mllm_method_role.md` P9]. **[MARGINAL as a novelty pillar]**

**3.2 Adaptation-not-identity as a scientific contribution (F44→F45→F53 chain).** A three-step
mechanistic result: (i) F44 (`8a48938`) — the encoder swap converts **iff** hate is visually-grounded
AND errors are representation-limited, otherwise it merely rotates the ranking; (ii) F45 (`d76e407`) —
ZH LoRA decomposes to a **genuine Pareto conversion in the text stream** (hate-recall +0.1111 at
non-hate −0.0032) vs frozen-Qwen's **rotation** (hate +0.0741 / non-hate −0.0481), so *"the
convertibility line runs through ADAPTATION, not encoder identity"*; (iii) F53 — KS-2 not tripped
confirms the HateMM LoRA pass is not merely image-inherited. The steelman: this is a **novel
mechanistic finding about *how* MLLM adaptation converts in hateful video**, distinct from "swap the
encoder." **[MARGINAL — this is an *analysis* contribution; whether a mechanism *explanation* satisfies
a *method-novelty* clause is exactly the ruling.]**

**3.3 The localization role (P6 / P10-b), paired-significant.** The per-window MLLM evidence scorer
ranks HateClipSeg windows at within-video AUC **0.5435** vs memory **0.5140** vs random **0.5088**
(paired-over-memory Δ+0.0296, CI [+.009, +.050], **p=0.007**) [`DRAFT_analysis_chapter.md` §3.5,
`EXP_p6_mllm_localization.md`]. Amplified, A-fuse reaches wv-AUC **0.5755** (bootstrap CI
[0.5581, 0.5933], sign-p 1.4e-9, n=329) [`DRAFT_analysis_chapter.md`]. This is the **one lane where the
MLLM exceeds both memory and random with statistical support** — a genuinely earned, removable MLLM
role. *Caveat:* explicitly **MODEST, not substantial** (0.5755 < 0.60 line); single test-touch spent
[`TERMINUS_mllm_campaign_DRAFT.md` §6]. **[MARGINAL]**

**3.4 The negative-results ledger itself as novel systematic evidence (C6).** A 13-route
pre-registered campaign with 11 main-table-accuracy routes all honestly killed, plus a coherent
**law**: **5 instances of "better semantic signal → no accuracy conversion"** (F44 = 4th, F50 = 5th
datum) and a **3-supervision closure** of per-item channel-selection (F47: closed at all three
supervision sources; F49: fails even at the alignment ceiling with a perfect judge). The steelman:
publishing this ledger as a **first-class methodological contribution** ("semantic competence is
orthogonal to the decision variable; a passing no-head probe is necessary but not sufficient") is
itself a novel contribution to the field. **[This is a novelty-of-*findings* claim, not a
novelty-of-*method* claim — noted as such.]**

**3.5 The four-pillar bundle is more than any one encoder lever.** D7's original wording targeted
encoder-class levers **standalone**. The delivered method is a **bundle** — retrieval-contrastive+kNN
core (C1, validated 3/3 datasets vs MoRE), updatable memory + O(1) temporal recalibration (C2,
validated), consensus denoising (C3, ZH-validated), auditable/editable archive (C4). The steelman: the
*integration* of an MLLM into **this specific memory architecture** (swappable + temporally
recalibratable + auditable + editable), which "no hateful-video method offers … simultaneously"
[`DRAFT_intro_related_limitations.md:188`], is the novel object, and the encoder lever is one earned
role inside it. **[Whether "novel memory architecture with MLLM roles" reads as "novel MLLM
integration" per the goal is the ruling; the pillars C2–C4 do not themselves deliver the ≥+3 acc
conjunct — that comes from the encoder lever D7 names.]**

**3.6 cand-2 (if it clears) supplies a coupling-novelty upgrade.** If the pending cand-2 verdict shows
the retrieval-memory-mined hard-negative SFT curriculum **adds over generic LoRA** (K-C2-2 PASS), the
LoRA leg re-labels from "generic encoder-class (D7-dead)" to "**memory→adaptation coupling**," a
mechanism that couples the retrieval memory *into* the adaptation objective — arguably novel-in-field
[`CAND2_CURRICULUM_PREREG.md:458–461`]. **[PENDING and low-prior — see §4.4/§6.]**

**Strongest single point of this case:** the **F44→F45→F53 adaptation-not-identity mechanism chain** is
a concrete, reproducible, source-pinned scientific result unique to this project — *if* the user counts
a novel mechanistic finding about MLLM adaptation as satisfying novelty.

---

## 4. CASE THAT NOVELTY IS NOT SATISFIED (steelman, from the record)

*Constructed as the strongest good-faith case the banked record can support. Each point is cited; none
is endorsed here.*

**4.1 D7's own language already covers LoRA — verbatim.** The binding record names it explicitly:
"encoder-class 杠杆——frozen swap、**LoRA-adapted swap**……**均不满足 goal 的 novelty 子句**"
[`DECISION_MEMO_pending.md:80–81`]. The 2-dataset performance conjunct is delivered by exactly this
lever (§2.2). On a plain reading, the thing that passes is the thing D7 already excluded. This is a
**definitional** closure, not an empirical one [`TERMINUS_round3_mllm_plus3.md:104`].

**4.2 Encoder swap / LoRA are textbook levers.** LoRA-SFT of an encoder and swapping a pretrained
backbone are standard, widely-used techniques; RA-HMD/LMM-RGCL already ship LoRA-tuned LMM contrastive
adaptation in the meme domain [`DESIGN_iter1.md:120`, `TARGET_GATE0_NOVELTY_REVIEW.md:124`]. The
project's own historical classification calls the LoRA/RA-HMD-family a **"MIXED performance lever, not
novelty"** [`query_pack.md:44`, `B1_PREREG_REVIEW.md:64`]. A top-venue reviewer sees a known technique
applied to a new domain, which the goal explicitly excludes from novelty (domain transfer =
inspiration, not novelty).

**4.3 The passing configuration is protocol-fragile and marginal.** The ≥2-dataset story exists **only
under the final-epoch protocol**, and its second leg (ZH) is **[MARGINAL]**: +0.0313 acc, margin over
bar +0.0013 (structural), seed2 sub-bar, and **val-selected FAILs** [`B3_PREREG_REVIEW.md` §2.2;
F53]. Under the no-selection primary protocol the loop treats as more honest, the 2-dataset conjunct is
**not clean**. If one demands a single mechanism passing ≥2 datasets under *both* protocols, it does not
exist (HateMM passes both; ZH passes one).

**4.4 A cand-2 tie collapses the coupling claim (prior ~50–60%).** The pre-registered honest prior is
**~50–60% that K-C2-2 ties**, collapsing cand-2 to "generic LoRA with reshuffled data" — i.e. even
encoder-capacity-allocation is **redundant with the head's frozen-feature mining**
[`CAND2_CURRICULUM_PREREG.md:34,462`; F52]. cand-2 also **opens no new dataset** (it holds ZH + inherits
HateMM; HateMM decides on the image stream LoRA doesn't touch, EN is label-limited) — so even a PASS
does not add a third dataset [F51/F52]. On the majority prior, cand-2 does **not** rescue novelty.

**4.5 The family/scale/EN failures bound any "MLLM-encoder family" headline.** The two encoder passes
ride **different mechanisms** (HateMM = frozen-swap image-borne; ZH = LoRA text-borne), so there is no
single mechanism spanning ≥2 datasets [`DECISION_MEMO_pending.md:101–106`, D8]. Scale **regresses**
(CLIP<32B<7B on the HateMM anchor; B2, 21st negative), and **every** encoder lever fails MHC-EN
(P9/B1/B2/B4) because EN is data/label-limited [`TERMINUS_round3_mllm_plus3.md:52–53`]. The
representation-gain class is "real but boxed."

**4.6 What top-venue reviewers would likely say (grounded in the drafts' own positioning).** The
related-work section positions the contribution by "clean difference from nearest competitors, **not by
any first-in-the-world claim**" [`DRAFT_intro_related_limitations.md:130`], and repeatedly flags
**honest novelty risks/withdrawals**: archive "auditable" wording clashes with SafeLens and was scoped
down [`:184–188`]; several ranking claims were **withdrawn** as within-noise/selection artifacts
[`:204–206`]. The C1 "we bring retrieval to hateful video" claim is deliberately **avoided** in favor
of a narrower mechanism-delta claim [`:75`]. A hostile reviewer reads: strong engineering + honest
negatives, but the accuracy delta rides a **known encoder-adaptation lever** the authors themselves
decline to call novel.

**Strongest single point of this case:** D7's binding text **names "LoRA-adapted swap" as
not-satisfying novelty**, and that is precisely the lever delivering the 2-dataset pass — a direct
verbatim collision.

---

## 5. DECISION MATRIX

*For each possible ruling, what follows operationally — paper claims, further GPU, goal status — all
consistent with the banked ledger. No option is recommended.*

| Ruling | Paper claims that become sayable | Further GPU implied | Goal status |
|---|---|---|---|
| **(A) Novelty SATISFIED by the bundle as-is** | Headline: MLLM meaningfully+novelly integrated into a swappable/recalibratable/auditable/editable retrieval-memory architecture; ≥+3 acc on 2 datasets (final-epoch) via earned encoder role; C1–C4 pillars + P6/P10-b localizer + F44/F45/F53 mechanism as the novelty substrate. Must carry the [MARGINAL] ZH + [DISPUTED] protocol caveats and the "no first-in-world" framing already in the drafts. | **None required** — all supporting runs are banked (F53 6b8f634, B3 13150, P6/P10-b test-touch spent). cand-2 becomes optional strengthening only. | **MET** (2 datasets, final-epoch, one lever + bundle). Campaign closes to writing. |
| **(B) Novelty SATISFIED only if cand-2 clears (K-C2-2 PASS ≥1 dataset + ZH-robustness strengthened)** | On PASS: upgrade the LoRA leg from "generic encoder LoRA (D7-weak)" to "memory-coupled adaptation curriculum (protocol-robust on ZH)"; claim memory→adaptation coupling as the novel mechanism [`CAND2_CURRICULUM_PREREG.md:458–461`]. On TIE: no coupling-novelty claim; revert to (C). | cand-2 chain **already running** (submit 1ea3c13; J1–J5, ~7–8 A100-h); **no additional GPU** beyond it. On TIE, no further runs on this axis (family exhausted). | **CONDITIONAL** — resolves to MET-with-coupling (PASS, prior ~40–50%) or unmet-on-this-axis (TIE, prior ~50–60%). Single curriculum draw caveat (F0.2) travels with any PASS. |
| **(C) Novelty NOT SATISFIED (bundle + LoRA = encoder-class, D7 stands)** | LoRA/frozen-swap enter as **formal performance/ablation/diagnostic rows only**, not novelty [`DECISION_MEMO_pending.md:81`]; novelty narrative rests on C1–C4 pillars + C6 negative-results contribution + P6 localizer, none of which alone delivers the ≥+3 conjunct. No "goal-met via encoder" headline (option-c goal-renegotiation is DEAD per §8). | **None on the closed encoder axis.** Requires a **NEW novelty-bearing axis** that defeats the banked meta-patterns (K9 calibrated-zero; oracle-unconvertible ×3; better-signal-no-conversion ×5; adaptation-not-identity F45) — none currently in the pool (F43: pool empty). | **UNMET** on frozen terms; per user's standing "keep working until achieved" directive, requires either a new axis (none pooled) or a user re-scoping of the novelty bar. |

**Cross-cutting facts true under all three rulings:**
- The performance numbers do not change: HateMM LoRA +0.0573/+0.0682 (final) & +0.0419/+0.0460
  (val-sel), 3/3; ZH LoRA +0.0313/+0.0453 (final only, MARGINAL); EN LoRA FAIL both.
- No test-touch remains to be spent on localization (P10-b spent once); HateClipSeg split frozen.
- TERMINUS option (c) "renegotiate the goal" is already **DEAD** by the same 2026-07-14 ruling
  [`TERMINUS_round2_mllm_plus3.md:81`] — so "unmet" does not reopen goal-renegotiation automatically;
  it routes to new-axis search or an explicit fresh user re-scope.

---

## 6. OPEN EVIDENCE THAT COULD STILL MOVE THE RULING

*Factual register of what is not yet resolved. No recommendation.*

1. **cand-2 curriculum verdict — [PENDING], the single largest open item.** Chain LIVE (submit record
   `1ea3c13`): J1=13237 (ZH-SFT), J2=13238 (HateMM-SFT), J3/J4 extractions, J5 heads (6 rows), chained
   `afterok`. Smoke 13236 PASS. On J5 completion → 0-context verdict vs `CAND2_CURRICULUM_PREREG.md`
   verbatim (K-C2-0/1/2, KS-regression, ZH-robustness). **Honest prior ~50–60% TIE** (collapses to
   "generic LoRA with reshuffled data"), ~40–50% PASS. Even a PASS is a **single curriculum draw vs
   single generic draw** (F0.2 caveat) and **adds no third dataset**. The prereg pre-commits that the
   novelty ruling **remains the user's** either way [`CAND2_CURRICULUM_PREREG.md:461,468`; F52].
   [state: `progress.json` active_line]

2. **ZH protocol-sensitivity / marginality — [MARGINAL], partially open.** The ZH leg passes final-epoch
   only, margin +0.0013 over bar, seed2 sub-bar, val-sel FAIL. cand-2's ZH-robustness read (§7.1 of the
   prereg) could strengthen or not-strengthen this; F45 already attributes the val-sel failure to
   78-dev selection noise (dev plateaus ~ep19 while test climbs to ep29) rather than absence of effect,
   but that is an *interpretation*, not a protocol pass [F45, `d76e407`].

3. **Negative-results count ordinal tension — [DISPUTED], bookkeeping only.** F53 records B4-EN as the
   "24th pre-registered negative (per drafts' count discipline; **ordinal tension noted**)," while
   round-2 terminus counts 21/22 and PAPER_MASTER_TABLES holds "13 total routes, 11 main-table." The
   total-negative count is **not reconciled to a single figure** across rounds; this dossier does not
   cast a single total. It does not affect the ruling substance, only how the C6 ledger is worded.
   [`PAPER_MASTER_TABLES.md:208–212`, F53]

4. **Frozen-swap HateMM headline number — [DISPUTED], convention only.** +4.2 acc/+4.4 mF1 (T1.1
   convention) vs +5.3–5.6 acc (3-seed paired convention) — to reconcile in the paper table; not
   load-bearing for D7 (frozen swap is D7-dead regardless).

5. **No new novelty-bearing axis currently pooled.** Round-3 pool is empty (F43); any new direction
   must defeat the banked meta-patterns (K9 calibrated-zero, oracle-unconvertible ×3,
   better-signal-no-conversion ×5, adaptation-not-identity F45). If the user rules (C), the operational
   gap is that **no candidate is presently queued** to supply a fresh novel mechanism.
   [`progress.json` notes; `TERMINUS_round3_mllm_plus3.md`]

---

*End of dossier. This document organizes evidence for the D7 ruling and deliberately stops before any
recommendation; the ruling is the user's.*
