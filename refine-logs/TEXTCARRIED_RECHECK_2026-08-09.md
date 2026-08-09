# TEXT-CARRIED RECHECK — HateMM CLIP text-only vs image-only kNN AUC, empty transcripts removed

**Date** 2026-08-09. **CPU only, zero GPU, zero test-set touch.** Only
`data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt` and
`data/gt/HateMM/train.jsonl` are read. No dev/test metric enters the judgement.

**Trigger:** `refine-logs/EMPTY_TEXT_AUDIT_2026-08-09.md` §2d + §3b RECHECK-1 + §3d.
39/744 HateMM train rows carry `" "` as transcript; the CLIP text tower encodes them as
`[BOS, EOS]` → one identical 768-d vector, and that single point is 36/39 non-hate (P(hate)=0.077)
against a 0.401 split base rate. Any text-inclusive kNN read-out gets a near-free correct answer
on 5.2 % of the memory bank from a constant.

**Conclusion under test:** "HateMM is TEXT-carried" — CLIP text-only train-LOO kNN AUC
**0.8471** ≥ image-only **0.8255**, margin **+0.0216**, recorded in
`refine-logs/HATEMM_LORA_STREAM_DECOMP.md` §Q1 / `scripts/analysis/hatemm_lora_stream_decomp_out.json`
(`geometry.CLIP.{text,img}.train_loo_knn.auc`), propagated into `DRAFT_analysis_chapter.md` and
`DRAFT_experiments_chapter.md`.

---

## FROZEN DECISION RULE (written before any 705-row number was computed)

Primary footing = **train-LOO kNN AUC, k=20, cosine-weighted signed vote**, exactly the protocol of
`scripts/analysis/encoder_swap_geometry.py::loo_knn` reused by `hatemm_lora_stream_decomp.py`
(per-stream L2-norm, `np.fill_diagonal(S, -inf)`, `score = Σ_j cos_ij · (2y_j−1)` over the top-20).
Clean condition = the 39 empty-transcript rows deleted from **both** the query and the memory side
(n = 705), per `EMPTY_TEXT_AUDIT_2026-08-09.md` §3d.

Let `Δ705 = AUC_text(705) − AUC_img(705)`.

- **SURVIVES (conclusion stands, footnote added)** iff `Δ705 > 0` **and** `Δ705 ≥ 0.0216/2 = 0.0108`.
- **RETRACTED** iff `Δ705 ≤ 0` (text advantage gone or reversed), **or** `0 < Δ705 < 0.0108`
  (advantage survives in sign but is more than half manufactured by the constant cluster —
  the recorded margin is not defensible as stated).
- In the RETRACTED branch, mark the conclusion RETRACTED and annotate every downstream item on
  the `EMPTY_TEXT_AUDIT_2026-08-09.md` §3 RECHECK list.

Alignment check (must pass before the judgement is read): the 744-row rerun must reproduce
0.8471 / 0.8255 to ±0.0005. If it does not, the protocol is not the original one and the
judgement is void.

Script: `scripts/analysis/textcarried_recheck.py` → `scripts/analysis/textcarried_recheck_OUT.json`.

---

## RESULTS

*(filled in after the frozen rule above; see git history of this file for the pre-run state)*
