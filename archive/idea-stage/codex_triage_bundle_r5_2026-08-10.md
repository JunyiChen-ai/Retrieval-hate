# Round-5 triage bundle — training-level mechanisms only

## 0. Your job
Score and rank the 10 candidates in §4. Output for each: score /10, GO / HOLD / DEAD, one-line
reason. Then name **at most two** to pilot, in order, and say explicitly if you think **zero**
should be piloted. You may also declare a candidate dead on novelty or on an asset fact.
Do not soften: this project has generated 55 candidates over four rounds with zero survivors,
and the executor's bias is toward finding something to run.

## 1. Scope, fixed by the user (not negotiable)
- Only the four existing datasets: HateMM, MHC-EN, MHC-ZH, ImpliHateVid. **No new dataset.**
- Video-level binary classification only.
- **The mechanism must change training or representation** — LoRA objective/target, how the MLLM
  is used, encoder-level input integration, generative decision. Head-level and late-fusion
  mechanisms are forbidden this round (that space was exhausted in round 4).
- **Hard exclusion: anything of the form "learn something else on top of frozen features".**

## 2. Phase-A measurements taken today (this session), zero-GPU, on the test sets
Baseline = the round-4 best three-encoder ensemble comparator per dataset, 3 seeds,
thresholds picked on validation (§8.7 of `idea-stage/IDEA_REPORT.md`):
HateMM 0.8732 · MHC-EN 0.7776 · MHC-ZH 0.8183 · ImpliHateVid 0.9276 macro-F1.

**(A1) Annotation ceiling from MultiHateClip per-annotator votes** (`data/gt/mhc_votes/`,
2.18 EN / 2.26 ZH annotators per item; binary positive = Hateful ∪ Offensive; the majority
reproduces the project's binary label on 100 % of test ids in both languages):

| | pairwise raw agreement | Krippendorff α | split-vote rate | 1-annotator-vs-rest macro-F1 | **panel-resample ceiling (test split)** | base | **prize** |
|---|---|---|---|---|---|---|---|
| MHC-EN | 0.817 | 0.803 | 12.3 % | 0.928 | **0.9276** (p05 0.896) | 0.7776 | **+15.0** |
| MHC-ZH | 0.781 | 0.772 | 16.1 % | 0.908 | **0.9387** (p05 0.909) | 0.8183 | **+12.0** |

HateMM and ImpliHateVid ship no per-annotator votes; their headroom to a perfect predictor is
+12.7 and +7.2. **The prize pool is ≥ 7 points on every dataset — annotation noise is not the
binding constraint.**

**(A2) Error attribution.** All 108 test errors of the baseline were read individually
(full transcript, plus the video-level OCR cache for HateMM) and coded into buckets before any
count was taken. Buckets: **S** = stance / use-vs-mention (reportage, quotation, counter-speech,
satire, archival, or hate-adjacent-but-labelled-normal); **O** = decisive evidence is burned-in
on-screen text, not speech; **M** = transcript empty or music-only; **A** = annotators split or
label conflicts with the material; **D** = item named as a duplicate/degenerate in the project's
own train↔test audit; **X** = ordinary ranking error.

| dataset | n err | S | O | M | A | D | X | **macro-F1 gained by oracle-fixing S** | fixing S+O+M |
|---|---|---|---|---|---|---|---|---|---|
| HateMM | 26 | 8 | 5 | 1 | 1 | 2 | 9 | **+3.47** | +6.35 |
| MHC-EN | 31 | 16 | 0 | 2 | 5 | 0 | 8 | **+10.48** | +11.99 |
| MHC-ZH | 24 | 12 | 0 | 2 | 1 | 0 | 9 | **+8.90** | +10.31 |
| ImpliHateVid | 27 | 13 | 0 | 0 | 2 | 1 | 11 | **+3.00** | +3.00 |
| **total** | 108 | **49 (45.4 %)** | 5 (4.6 %) | 5 (4.6 %) | 9 (8.3 %) | 3 (2.8 %) | 37 (34.3 %) | **mean +6.46** | mean +7.91 |

Concrete S-bucket examples, so you can judge whether the coding is honest:
HateMM false positives include a John Lennon protest song containing a slur, 1950s archival
documentary footage of segregationists, and a video exposing a named neo-Nazi (counter-speech).
ImpliHateVid false positives include a monologue *arguing against* using racial slurs (which
therefore mentions one) and a satirical sketch about sexism. False negatives include a CTV news
*report* about international students labelled implicit hate.

**(A3) MHC errors are 5–6× enriched on items where the annotators themselves split**
(MHC-EN 38.7 % of errors vs 16.1 % of the test set, OR 5.23; MHC-ZH 37.5 % vs 13.4 %, OR 6.22).

**(A4) New measurement made today: OCR is redundant with ASR on MultiHateClip.** PaddleOCR
K=30 was run on both MHC test splits for the first time (161 + 149 videos, 15 min). 95 % / 99 %
of videos carry ≥20 chars of on-screen text, but inspection of every error item shows the text is
burned-in captions of the speech already in the transcript, plus uploader watermarks — **not one
MHC error is decidable from on-screen text that the transcript lacks.** On HateMM the opposite
holds (MEMRI-TV translated subtitles, Britain First title cards, a slur that appears only in the
burned-in text of a video whose transcript is empty). **The O bucket is a HateMM-only
phenomenon in this project.**

## 3. Constraints that kill candidates before you score them
- **K1. ImpliHateVid has no raw video on this machine** (B2 only). Every training-level or
  encoder-level candidate is therefore **at most 3 of 4 datasets**.
- **K2. The GPU is not currently available.** One RTX 5090 (32 GB), shared; another user's job
  has held 21–28 GB at 97 % utilisation for 10 h. Only ~5 GB free — a 7B MLLM does not fit even
  in 4-bit right now.
- **K3. The LoRA training stack is not on disk.** `RA-HMD/LLAMA-FACTORY-Ver202512` is an
  uninitialised submodule; the `my_configs/hatevideo/*.yaml` were authored locally and are gone;
  `llamafactory` is not pip-installed; all paths point at `/data/jehc223/RGCL`, which does not
  exist here; **no trained LoRA adapter exists on disk**. Assets that DO exist: Qwen2.5-VL-7B
  weights (16 GB), pre-extracted 8-frame JPGs for all 2,661 videos (2.6 GB), ShareGPT SFT JSONs.
- **K4. Recorded cost of one LoRA-SFT arm: 2.28 h (MHC) / 2.85 h (HateMM) on an A100-80GB**,
  8 frames, `image_max_pixels 262144`, bs1×accum8, 3 epochs. Feature re-extraction adds ~0.54
  GPU-h per dataset. **No 5090 measurement exists.** A 3-dataset × 2-arm × 3-seed design is
  ~100 GPU-h.
- **K5. The project has already measured the generative-decision null.** `EXP_p9_lmm_rgcl_video`:
  LoRA-SFT'ing the MLLM and reading its own decision head lands at the protocol-matched floor —
  EN +0.6, ZH +1.0, HateMM +0.9 pts, all inside seed noise — and reading it through the retrieval
  memory is 2.2–4.7 pts **below** floor.
- **K6. No leakage-free stance supervision exists on disk.** MultiHateClip's `Target_Victim`
  field is ~95 % determined by the label (EN: 631/662 negatives empty, 278/339 positives filled),
  so it cannot be an auxiliary target. The one clean signal is the **`Counter Narrative`
  annotator vote**, which never survives majority aggregation: 63 EN + 76 ZH items corpus-wide,
  and its rate is nearly balanced across classes (EN 7.7 % of positives vs 5.6 % of negatives;
  ZH 7.1 % vs 7.8 %). **But only 1 of the 55 MHC test errors carries a CN vote** — the asset does
  not mark the bucket it is meant to target.
- **K7. Standing prohibitions.** Teacher-score distillation and counterfactual data augmentation
  are closed by user ruling (`research-wiki/EXP_bsrtd_KILL_2026-08-10.md`) and must not be
  revived. Curriculum LoRA and retrieval-contrastive QLoRA are dead from earlier rounds.
  Eleven never-claim items in `research-wiki/NOVELTY_RECON_2026-08-09.md` App. B.
- **K8. The comparator bar is the three-encoder ensemble of pairwise-trained heads**, not the
  bare head (round-4 §8.10). A candidate that beats a single BCE head but not the ensemble has
  shown nothing.

## 4. Novelty verdicts (independent recon, this session; every arXiv id fetched and title-matched)

| family | verdict | strongest occupant | rejection citation |
|---|---|---|---|
| **F1 rationale-then-verdict SFT / RLVR** | **OCCUPIED** | IARE `2606.11953` (CoT-SFT+DPO on **hateful video**, Ex-HateMM 85.86→90.14 at n=749; Ex-ImpliHateVid 89.50→91.75 at n=1205); LEAF `2026.findings-acl.604`; ExPO-HM `2510.08630` (ICLR 2026) | ExPO-HM: Direct-SFT **75.0** F1 > CoT-SFT 74.5 > GRPO 74.5 on Qwen2.5-VL-7B — naive explain-then-detect *loses*. Plus `2409.12183` |
| **F2 generative MLLM as classifier** | ADJACENT | RA-HMD `2502.13061` (EMNLP 2025 oral); WWW 2025 `2501.15438`; HateClipSeg `2508.01712` | RA-HMD App. G: label-token 90.2 vs head 91.1 AUC in-domain. `2603.02546` (ICLR 2026): discriminative +2.5 % on video. **Small-n evidence is a loss: MHC-EN n=1000 generative 0.78 vs head 0.79.** Compounded by K5 |
| **F3 stance / use-vs-mention as SUPERVISION** | **OPEN** | `2404.01651` (NAACL 2024) is **prompting-only** and its Limitations explicitly leave fine-tuning unexplored; TANDEM `2601.11178` supervises *target*, not stance; ImpSH `2606.18852` contrasts *implied statement* | `2404.01651` kills any inference-time-prompt framing; `2307.03377` (IJCNN 2023) kills bare auxiliary-head MTL via negative transfer. **No isolated ablation at n<1000 published** |
| **F4 annotator votes as training target** | **DROP** | AI Wizards EXIST 2026 `2607.04410` (multimodal port done) | `2605.20642`: with few annotations/example, hard labels beat soft. On HS-Brexit (1120 items, **6** annotators) the flagship multi-annotator architecture ranked 19th and last, both worse than majority-class. We have **2** annotators |
| **F5a/b OCR integration (encoder or prompt)** | **OCCUPIED** | **MM-HSD `2508.20546` (ACM MM 2025): PaddleOCR at 1 fps as the cross-modal attention query, M-F1 0.874 on HateMM** — i.e. equal to our ensemble; `2602.09637` puts OCR in an LLM prompt on HateMM + MultiHateClip | MM-HSD + `2602.09637`. Also killed by our own A4 measurement outside HateMM |
| **F5c text-bearing frame selection for moderation** | OPEN | SFA `2511.20190` (text-region focus, not moderation); AKS `2502.21271`, Q-Frame `2506.22139` (query-relevance, not text) | AKS/Q-Frame for naive relevance selection; `2508.10974` (AAAI 2026): relevance sampling still misses >90 % of harmful content |
| **F6 missing-modality / silent-video training** | ADJACENT | `2602.01101` (WWW 2026, memes); IMOL `2025.acl-long.1494` (fake-news video) | Dai et al. CVPR 2024 `2403.04245`: plain modality dropout buys robustness by *inducing modality bias*, costing accuracy on complete data — fatal when 88 % of the split has speech |

## 5. The 10 candidates

Each is stated as: mechanism · which bucket · what it needs · what would kill it.

**C1 — Conditional-Mask Stance Auxiliary LoRA (CMS).** Train the LoRA with the verdict loss on
every item plus an auxiliary "does the speaker endorse or contest the harmful material?" head
supervised **only** on the 63 EN + 76 ZH items carrying a `Counter Narrative` annotator vote,
masked elsewhere (CondMTL `2302.07372` is the template for sparse auxiliary labels).
Bucket S (+10.5 EN / +8.9 ZH). Needs: nothing new. Kill: negative transfer (`2307.03377`);
and K6 — only 1 of 55 MHC errors carries a CN vote.

**C2 — Stance-Contrast LoRA (SCL).** Same asset, different mechanism: use CN-vote items as the
*contrast axis* of a supervised-contrastive term inside the LoRA (pull CN items away from
same-topic hate items), rather than as a classification head. Bucket S. Kill: LAHN `2406.07886`
and ImpSH `2606.18852` occupy label- and implication-axis contrast; 139 anchors is very thin.

**C3 — Stance-Conditioned Extraction Prompt (SCEP).** Pure encoder-level, no training: the
feature extractor's `TEXT_INSTRUCTION` currently reads *"…summarise the targets, symbols, tone,
and any harmful intent conveyed"* and the pooled representation is the mean over the assistant's
generation. Replace it with an instruction that requires the model to first state whether the
video's author endorses, condemns, reports or quotes the harmful material, then summarise.
The extractor already exposes `--text_instruction`. Cost ≈ 0.54 GPU-h per dataset per variant.
Bucket S. Kill: it is a prompt change, so the novelty claim is weak; and §8.13 of the report
already found the frozen text spaces cannot represent policy exemptions.

**C4 — Hate-Relevant-Evidence Frame Selection (HEFS).** Replace uniform `linspace` 8-frame
sampling with a selector that picks frames by predicted hate-relevant on-screen-text evidence
yield, then LoRA-train and extract on the selected frames. Bucket O. F5c is the only OPEN half
of the OCR family. Kill: our own A4 measurement says the O bucket exists on HateMM only
(5 of 108 errors, +2.37 macro-F1 on one dataset).

**C5 — Bias-Controlled Transcript Dropout (BCTD).** LoRA-train with the transcript stochastically
blanked, plus a dominant-modality correction so full-transcript performance is not sacrificed
(the `2403.04245` requirement). Bucket M (+0.5 / +1.6 / +1.5 / 0). Kill: prize is ~+0.9 mean and
the family is ADJACENT at best.

**C6 — Naturally-Silent-Subset Training (NSS).** Same family as C5 but the honest framing: real
silent videos are a biased subpopulation, so train and *evaluate* on the natural 12.1 % silent
subset of HateMM rather than i.i.d. random masking. Bucket M. Kill: one dataset; reads as an
evaluation protocol, not a mechanism.

**C7 — Pairwise-Ranking LoRA Objective (PRLO).** Lift round-4's §8.10 finding 2b (a pairwise/AUC
objective beats BCE on ranking in 4 of 4 cells at the head level) down into the LoRA: replace
token cross-entropy on the verdict with a pairwise ranking loss on the verdict-token logit
between a hate and a non-hate item in the same batch. Bucket: all. Kill: standard machinery,
no novelty claim available; and it may simply reproduce the head-level gain we already have.

**C8 — Rationale-then-verdict SFT (self-generated, rejection-sampled).** STaR-style: the model
generates its own rationale, keep only those leading to the correct label, SFT on them. Bucket S.
Kill: **F1 is OCCUPIED** — IARE does CoT-SFT+DPO on this exact dataset lineage — and ExPO-HM
measured naive CoT-SFT *below* direct SFT.

**C9 — Calibrated Generative Verdict (CGV).** Matched-conditions comparison of label-token
logprob vs verbalised confidence vs a head on the same model's features, with the known logprob
repairs (DC-PMI, Platt, prompt-family ensembling). Bucket: all. Kill: K5 — the project already
measured this null in-house — and the published small-n evidence is a loss on MHC-EN.

**C10 — Vote-Fraction Soft-Target SFT (VFS).** SFT the model to emit the annotator vote fraction
rather than a hard label. Bucket A (+2.8 EN / +0.9 ZH). Kill: **F4, drop outright** — 2
annotators per item, and the flagship architecture lost to majority-class on a better-annotated
corpus.

## 6. Questions you must answer explicitly
1. Given that F3 is the only OPEN family and it coincides with the largest error bucket
   (45 % of errors, mean +6.5 macro-F1), but its only leakage-free supervision is 139 sparse
   labels that do **not** mark the observed errors (K6) — is C1/C2 a real candidate or a
   candidate with no fuel? If no fuel, say so.
2. Is there a stance-supervision source we have missed that does not require new annotation and
   does not leak the label?
3. Under K2–K4 (no GPU, no training stack, ~100 GPU-h for a properly powered design), is the
   correct action to pilot anything at all this round, or to freeze a pre-registration and stop?
4. If you would pilot, name the cheapest design that could still return a **decisive** verdict,
   and state its GPU-hour cost.
