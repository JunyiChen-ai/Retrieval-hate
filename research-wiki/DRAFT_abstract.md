# DRAFT — Abstract

_Assembled 2026-07-09. All numbers transcribed from committed sources
(`PAPER_MASTER_TABLES.md` T1/T2, `DRAFT_intro_related_limitations.md` §1,
`DRAFT_method_chapter.md` §1/§6, `DECISION_MEMO_pending.md` D3/D6). No number
introduced beyond the master tables. Two decision-gated spots are marked
`[TODO-D3]` (EN/ZH wording) and `[TODO-D6]` (efficiency emphasis / venue), each
with a recommended replacement per the DECISION MEMO recommendation column._

---

## Candidate titles (not finalized — pick one)

- **A (span-free):** *Span-Free Hateful-Video Detection with a Retrieval-Guided
  Contrastive Memory and a Bounded Multimodal-LLM Role*
- **B (updatable memory):** *Updatable, Auditable Memory for Hateful-Video
  Detection: Retrieval-Guided Contrast and the Limits of Multimodal LLMs*

---

## Abstract (~215 words)

Hateful-video detection splits between heavy always-on reasoning VLMs that emit
a throwaway per-clip verdict and light fusion heads with no reusable memory. We
port retrieval-guided contrastive learning from hateful memes to video, building
a detector around a run-once frozen or LoRA-adapted encoder, a
few-million-parameter head, and a kNN read-out over an updatable memory bank
`[TODO-D6]`. The method rests on four pillars: (1) retrieval-guided contrastive
embedding with kNN-vote inference; (2) an updatable memory with a
temporal-recalibration protocol for evolving hate; (3) consensus denoising of
label-inherited segment supervision, validated on Chinese; and (4) an auditable,
human-editable archive memory. We delimit the multimodal LLM to three earned
roles — frozen encoder, span-free localization scorer, and guard-rail/auditor —
and one explicit non-role: at 7B–72B scale it adds no main-table accuracy. The
detector reaches 0.870 accuracy / 0.861 macro-F1 on HateMM and wins the
same-arena MoRE comparison on all three shared benchmarks (+5.6 / +8.7 / +6.7
accuracy) `[TODO-D3]`. For localization, a span-free per-window MLLM scorer ranks
HateClipSeg hate windows at within-video AUC 0.5755, paired-significantly above
the retrieval-memory read-out (0.5140; Δ +0.0615, p = 4.9e-5) though modest.
Finally, thirteen preregistered routes map where a multimodal LLM does and does
not help a retrieval-memory detector — a negative-result contribution delivered
alongside a memory that is swappable with zero retraining, recalibrated in O(1),
and surgically edited by a human at inference.

---

## Recommended replacement copy for the two `[TODO]` spots

**`[TODO-D6]` — efficiency emphasis / venue (DECISION MEMO D6 recommends
foregrounding efficiency as a co-headline; lean toward a negative-results /
methodology-friendly venue such as ICWSM or ACL-ARR).**
Recommended inline replacement of the marker with a co-headline efficiency
clause:
> "…a kNN read-out over an updatable memory bank — a run-once-encode,
> light-head design that matches or beats far heavier reasoning-VLM detectors
> while keeping every decision inspectable."

**`[TODO-D3]` — EN/ZH wording (DECISION MEMO D3 recommends reporting the
dual-protocol EN number under a same-arena-win + near-ceiling framing, not an
absolute-SOTA claim; the ZH 0.85 crossing is protocol-contingent and reported
but not adopted).**
Recommended inline replacement of the marker with:
> "; on the two MultiHateClip splits it sits near a documented ceiling —
> MHClip-EN ≈ 0.78–0.80, a decisive same-arena win over MoRE (0.69–0.72) rather
> than an absolute-SOTA claim, while MHClip-ZH crosses 0.85 only under a
> selection-free final-epoch protocol we report but do not adopt."

---

## Number provenance (for the citation audit)

| Claim in abstract | Value | Source |
|---|---|---|
| HateMM accuracy / macro-F1 (frozen-Qwen best stack) | 0.870 / 0.861 | `PAPER_MASTER_TABLES.md` T1.1 (`ebc1988`) |
| Same-arena MoRE deltas (HateMM / MHClip-EN / MHClip-ZH, accuracy) | +5.6 / +8.7 / +6.7 | `DRAFT_intro_related_limitations.md` §1; `PAPER_MASTER_TABLES.md` T1.2 |
| Localization span-free scorer wv-AUC (72B coarse×fine A-fuse) | 0.5755 | `PAPER_MASTER_TABLES.md` T2.1 / T2.2; `DECISION_MEMO_pending.md` D4 |
| Retrieval-memory read-out wv-AUC (contrast baseline) | 0.5140 | `PAPER_MASTER_TABLES.md` T2.1 |
| Localization paired significance | Δ +0.0615, p = 4.9e-5 | `DRAFT_intro_related_limitations.md` §1 |
| MHClip-EN band / MoRE same-arena band | 0.78–0.80 / 0.69–0.72 | `DRAFT_intro_related_limitations.md` §1; `HEADTOHEAD_FEASIBILITY.md` §3 |
| MHClip-ZH selection-free vs val-selected | 0.854 vs 0.827 | `DECISION_MEMO_pending.md` D2 |
| Preregistered routes bounding the MLLM role | 13 | `DECISION_MEMO_pending.md` D1; `CAMPAIGN_mllm_method_role.md` |
| Four pillars / three roles + one non-role wording | — | `DRAFT_method_chapter.md` §1, §6 |
