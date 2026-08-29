# EUM — EVIDENCE-UNIT MEMORY — FORENSIC RECON (zero-GPU)

**Agent:** forensic-recon (2026-07-28 adversarial wave, candidate 2 of 4) · **Date:** 2026-07-28 NZST.
**Discipline honoured.** CPU-only reading + $0 descriptive arithmetic on **banked TRAIN-split derived
caches**. **ZERO GPU / SLURM / Modal / training.** **TEST-SPLIT CONTACT: NONE** — no test feature file, no
test label, no test metric was opened or produced. No prereg written, no job submitted, no frozen artifact
mutated.

**Status of this document.** **Recon-level PRE-CLOSURE**, not a measured KILL. The distinction is used
consistently: a *measured KILL* is a frozen-bar verdict against hash-frozen-script numbers; a *pre-closure*
is a ban-scope + arithmetic argument. EUM's *trained* version, however, **was already measured** (Δ-1,
§3) — so EUM is unusual: it is pre-closed at the family level **and** carries a banked negative.

---

## 0. THE CELL, PRECISELY

**Mechanism statement.** Make the unit of retrieval an **evidence unit** — a sub-video span carrying one
piece of hate evidence — rather than the whole video. The bank becomes a flat list of unit keys; a query
retrieves units; the vote runs over unit labels inherited from their parent videos. The claimed pathology it
attacks: "a pooled whole-video key is diluted, because the hateful evidence occupies a small fraction of the
video, so the key is mostly non-hate content."

**Everything hinges on that premise.** §4(c) measures it and finds it **FALSE on the anchor dataset**.

---

## 1. PRE-CLOSURE TABLE

| # | prior finding / ban | binding? | quoted binding text (`file:line`) | ruling for EUM |
|---|---|---|---|---|
| 1 | **F37 / S2S family ban** | **BINDING BY NAME** | "family-level: **retrieval-object/don't-pool family CLOSED across encoders** (W2-B frozen-CLIP + S2S Qwen both dead); no SS11 downstream, no head GPU" (`directions_tried.json:133`, the `ban_scope` of the S2S dead entry whose `direction` is at `:130`) | EUM *is* the retrieval-object family: it changes what a bank row **is**. Closed by name. |
| 2 | **W2-B** — CLIP frame-local object | **CLOSED (measured)** | W2-B's own scope, as quoted by the ISR recon: "zero-training **frozen-CLIP** sub-clip set-matching … does **NOT** veto the Qwen-token S2S line" (`SEG_REENCODE_FORENSIC_RECON.md:92-93`); operator results "HateMM SET −0.0047; EN +0.0016, ~20× under bar" (`:138`) | Representation object #1 dead. |
| 3 | **S2S / CTF** — Qwen causal-prefix object | **CLOSED (measured)** | F37: "HateMM primary Δ(SET−POOLED) acc **+0.0035**/mF1 +0.0003 fails … +0.05 bar on all six sub-conditions; MHC-EN **NEGATIVE: acc −0.0397**" (`findings.jsonl` F37). F39/CTF, the *supervised* closure: "Binding flat [g_1..g_T] 14336d over Z_best 8960d: HateMM best-k **+0.0000** CI[−0.0031,+0.0031]" (`findings.jsonl` F39) | Representation object #2 dead — and dead at **exactly zero conditional information** under a supervised operator. |
| 4 | **F66 / ISR** — independent-segment object | **CLOSED (measured)** | "the sole ban-surviving operator (per-segment-kNN vote-mean, uniform, no selection) is **FLAT**: HateMM +0.0012 / EN +0.0032"; "HateMM oracle +0.0776 = symmetric +0.0012 (legal) + selection +0.0764 (banned)" (`findings.jsonl` F66) | Representation object #3 dead, **with the decomposition proving the surviving headroom is the wrong kind**. |
| 5 | **The operator pincer** | **BINDING — no third branch** | "Any operator is either (a) a **fixed symmetric aggregation** — which is the **don't-pool / retrieval-object family** (BAN A); or (b) a **per-item choice** — which is **selection** (BAN B: Law III + P11). There is **no third kind**." (`SEG_REENCODE_FORENSIC_RECON.md:120-123`); and "The moment you make the segment combiner *non-uniform* … you re-enter selection (Law III) or, if the weights come from MLLM hate-density, **P3** … or the **MLLM-scores-as-signal** ban (constraint [5])" (`:123-126`) | EUM's operator space is exhausted before it starts. |
| 6 | **Law II / cost** | **BINDING AT $0** | Qwen per-segment features do not exist: "**NO Qwen/VideoMLLM per-segment features exist anywhere** (`find` over `data/**` = 0)" (`SEG_REENCODE_FORENSIC_RECON.md:198-199`); reaching them costs "K=4 primary, HateMM+EN, all splits ≈ **~2–3 GPU-h**" and is **local-GPU-only** because it consumes raw video (`:213-221`) — the extraction F66 declined to authorise | The deployed-encoder arena is **unreachable at $0**. Any $0 EUM probe runs on the *weaker* CLIP caches, which are the object W2-B already killed. |
| 7 | **Gold spans** | **BANNED** | `banned_constraints[1]` = "gold annotations inside method (time-span, target)" (`directions_tried.json:456`); ISR: "HateClipSeg `gold_segments.json` exists but is the localization set … do **NOT** introduce gold spans (banned constraint [1])" (`SEG_REENCODE_FORENSIC_RECON.md:70-72`) | The **only unit definition Δ-1 says could work** is the banned one. See §3. |
| 8 | **MLLM-derived unit boundaries/weights** | **BANNED (two ways)** | `banned_constraints[5]` = "MLLM-scores-as-training-signal"; `[6]` = "P1-P5 re-proposals" (`directions_tried.json:460-461`, 0-indexed per `SEG_REENCODE_FORENSIC_RECON.md:125`). P3 = "MLLM segment hate-density pooling weights — probe pass, train flat, 3 datasets"; P11 = "MLLM segment scores as weak-sup training labels — probe fail; **MIL already carries it**" (`SEG_REENCODE_FORENSIC_RECON.md:108-109`) | **Tripwire.** The obvious way to define an "evidence unit" without gold spans is to let the MLLM say where the evidence is. That is P3 (weights) or P11 (labels), and it additionally trips constraint [5] and/or [6]. |

---

## 2. THE THREE-OBJECT CLOSURE, STATED AS ONE SENTENCE

There are exactly three ways this campaign has ever represented a sub-video unit, and all three are
measured dead: **frame-local CLIP** (W2-B), **causal-prefix Qwen** (S2S/F37, closed at *zero* conditional
information by CTF/F39), and **independent-segment** (ISR/F66, closed with the symmetric/selection
decomposition). EUM proposes no fourth representation — it proposes a fourth *name* for the same object
plus a claim about what the unit **means**. The meaning does not change the geometry, and §4 measures the
geometry.

---

## 3. THE TRAINED VERSION WAS ALREADY RUN — Δ-1

This is the fact that most cleanly separates EUM from a genuinely untested cell: **the trained,
unit-level-retrieval version has a banked 3-arm result.**

`research-wiki/experiments/exp-seg-mode-ablation.md` (`verdict: no`, `:6`, `:18`), with the paired numbers
at `research-wiki/ITERATION_LOG.md:593-602` (header `:593`, baselines `:591`):

| dataset | config | job | Test macro-F1 | acc | Δ vs λ=0 baseline (F1 / acc) |
|---|---|---|---|---|---|
| MHC (EN) | λ=0 baseline | 12128 | 0.7113 | 0.7826 | — |
| MHC (EN) | **milmax** (λ=0.5) | 12134 | 0.6089 | 0.7205 | **−0.1024 / −0.0621** |
| MHC_zh | λ=0 baseline | 12130 | 0.7706 | 0.8054 | — |
| MHC_zh | **milmax** (λ=0.5) | 12135 | 0.7875 | 0.8255 | **+0.0169 / +0.0201** |

> "Sign-flips by language. … **milmax rescues ZH but collapses EN.** … **No seg_mode beats whole-video
> baseline on BOTH languages**; no config crosses acc 0.85." (`exp-seg-mode-ablation.md:21`)
>
> "Verdict=no: diagnosed as **noisy MIL pseudo-positives without gold segment labels.** Demoted from
> headline to honest ablation. **Highest anti-repeat value — do not re-attempt segment-level temporal
> retrieval on these datasets without gold spans.**" (`exp-seg-mode-ablation.md:24`)

**The anti-repeat flag and the gold-span ban are jointly closing.** Δ-1's own diagnosis says the failure
mode is *unit definition without gold spans*; `banned_constraints[1]` forbids gold spans inside the method.
EUM is therefore asked to succeed at exactly the thing Δ-1 says needs the one input it may not have.

**And the best legal evidence unit was already built.** `research-wiki/EXP_mm_segment_keys.md` — multimodal
segment keys with a severity gate as the unit-selection criterion — final-epoch PRIMARY paired
**ΔF1 −0.0116 ± 0.0087, 3/3 seeds negative** (`EXP_mm_segment_keys.md:195`), Δacc −0.0083 (2/3 negative);
SECONDARY −0.0250, 3/3 negative (`:196`); verdict "**FAIL**" (`:198`), with the record's own reading:
"final-epoch (selection-free): PRIMARY 3/3 seed below floor … a consistent small negative effect, not noise"
(`:199-200`).

---

## 4. NEWLY MEASURED ARITHMETIC (this recon, $0 CPU, TRAIN SPLIT ONLY)

Three quantities decide whether a "unit" is a different object from a "video key" at all. All three are
computed here from **banked train-split derived caches**; the protocol is stated in full so the numbers are
reproducible without a script.

### (a) A flat unit bank does not deepen the neighbourhood — it *shallows* it

**Protocol.** Artifacts: `data/CLIP_Embedding/{HateMM,MHC,MHC_zh}/train_subclipK4_openai_clip-vit-large-patch14-336_HF.pt`
(fields `subclip_img_feats [V·K,1024]`, `subclip_parent [V·K]`, `video_ids [V]`) and
`data/CLIP_Embedding/HateMM/train_subclipK30_...pt`. L2-normalise every unit. Query = each video's **pooled
key** (L2-normalised mean of its own K unit vectors — the deployed pooled retrieval object). Bank = the flat
unit list with **the query's own units excluded** (video-level LOO, the W2-B B1 leakage guard). Take the
top-20 units by cosine and count **distinct parent videos**. Mean over the V queries. Train split only.

| cache | V | K | units | **mean distinct parent videos in top-20** | median | min |
|---|---|---|---|---|---|---|
| HateMM K=4 | 744 | 4 | 2 976 | **10.61** | 10 | 5 |
| MHC-EN K=4 | 549 | 4 | 2 196 | **10.83** | 11 | 5 |
| MHC-ZH K=4 | 579 | 4 | 2 316 | **11.28** | 11 | 5 |
| HateMM K=30 | 744 | 30 | 22 320 | **5.00** | 4 | 1 |

*(Newly computed this session; not previously persisted to any repo artifact.)*

**Why this matters, against F94.** F94 swept the vote depth and found "k=20 IS AT OR ABOVE THE PLATEAU ON
ALL 6 ARMS, and **the plateau starts at k~10-15**" (`findings.jsonl` F94). A flat unit bank at K=4 delivers
an **effective video-level depth of ~10.6-11.3** — i.e. it lands on the **bottom edge** of the measured-flat
plateau, where F94 says accuracy is unchanged (HateMM val-sel k=10 Δacc **+0.0000**, k=15 **+0.0000**). At
K=30 the effective depth collapses to **5.00**, i.e. **below the plateau floor**, moving toward — though not
yet inside — the band F94 measured as actively harmful ("k in {1,2,3} costs **−0.0157 to −0.0388** acc …
on every one of the 6 arms", F94 body).

> **Correction to the tasking, recorded:** the tasking said depth 5.00 "collaps[es] into F94's harmful
> region". It does not. F94's *measured* harmful region is **k ≤ 3** (where the vote provably degenerates to
> 1-NN); k=5 is below the plateau but not measured harmful — indeed F94's best ZH forensic gain anywhere is
> "ZH final best **k=5 = +0.0045**". The honest statement is **"below the plateau floor and trending toward
> the collapse band"**, not "in the harmful region".

**Net:** switching to units buys **no** new video-level evidence at K=4 and **destroys** video-level
evidence at K=30. The mechanism EUM sells — "see more, finer evidence" — is arithmetically a *depth
reduction*.

### (b) A "unit" is not a different object from the key it is supposed to refine

Same caches, same L2 normalisation, train split.

| dataset | **mean within-video unit↔unit cosine** (off-diagonal, K=4) | **mean unit ↔ own-video pooled-key cosine** |
|---|---|---|
| HateMM | **0.8842** | **0.9538** |
| MHC-EN | **0.8548** | **0.9432** |
| MHC-ZH | **0.8773** | **0.9523** |
| HateMM K=30 | 0.8401 | 0.9155 |

*(Newly computed this session; not previously persisted.)*

**A unit sits at cosine ≈ 0.95 to the very key it is meant to disambiguate.** In a space whose deployed
top-1 neighbour cosine is 0.9439-0.9686 in raw form (F91, quoted at `LITSWEEP6_MEMBANK.md:294`) and
~0.9999 in head space (`ERRPAT_HateMM_2026-07-26.md:131`), a 0.95 unit-to-parent cosine means the units are
**inside the parent's own retrieval radius**. Replacing one key by four such units is not a finer
representation; it is the same point counted four times, which is exactly what (a) measures.

This also independently re-derives F35's conclusion at the *pooling* level rather than the causal-prefix
level: pooling is near-lossless on these representations because the things being pooled are nearly
identical.

### (c) The dilution premise is FALSE on the anchor dataset

**Protocol.** Artifacts: `data/gt/HateMM/hate_spans.json` (1 083 entries; per video `{duration, spans,
label}`) intersected with `data/gt/HateMM/train.jsonl` restricted to gold `label == 1`, **n = 298**.
Coverage = Σ(span end − start) / duration, spans **un-merged and un-clipped** (the merged/clipped variants
move the median by ≤ 0.004; both were computed). **TRAIN SPLIT ONLY — the test rows of `hate_spans.json`
were not read.**

| statistic | value |
|---|---|
| **median hate-span coverage of the video** | **0.8289** |
| mean coverage | **0.7174** |
| fraction with a **single contiguous** hate span | **0.7416** (221/298) |
| fraction with coverage **< 0.5** | **0.2181** (65/298) |

*(Newly computed this session; not previously persisted. Sensitivity: merging overlapping spans gives
median 0.8249 / single-span 0.7517; clipping coverage at 1.0 gives mean 0.7164.)*

**Reading.** On HateMM — the campaign's anchor and the one dataset whose image stream converts — the
hateful evidence occupies a **median 83 % of the video** and is a **single contiguous block in 74 % of
cases**. Only 22 % of hate videos are even candidates for the dilution story. **The premise that motivates
evidence-unit retrieval is false where EUM would most need it to be true**, which is the mechanistic reason
(a) and (b) come out the way they do: there is no concentrated minority of hateful frames for a unit to
isolate.

---

## 5. VERDICT

> **PRE-CLOSED.** Family closed by name (F37 ban text, `directions_tried.json:133`); all three
> representation objects measured dead (W2-B / S2S+CTF / ISR); the operator space is a two-way pincer with
> no third branch (`SEG_REENCODE_FORENSIC_RECON.md:120-123`); the deployed-encoder arena is unreachable at
> $0 and reachable only via the ~2-3 GPU-h extraction F66 declined; the trained version is a banked negative
> with an explicit anti-repeat flag (Δ-1); the best *legal* evidence unit is a 3/3-seed negative
> (`EXP_mm_segment_keys.md:195`); and this recon measures the motivating premise **false** on HateMM.

**No GPU is warranted and none should be requested.** If the direction is ever revived, the revival must
first defeat §4(c) — i.e. exhibit a dataset where hate evidence is genuinely a minority of the runtime —
and must define its units **without gold spans, without MLLM scores, and without per-item selection**, which
§1 rows 5/7/8 jointly leave empty.

### P(pass) estimates

| bar | estimate | reasoning |
|---|---|---|
| P(≥ +0.030 acc on ≥2 datasets, both protocols) | **< 1 %** | Δ-1 already sign-flipped by language; the object is 3-for-3 dead; the premise is false on the anchor |
| P(≥ +0.030 on ≥1 dataset) | **1–2 %** | ZH only, replicating Δ-1's milmax +0.0201 acc — which was *not* a legal unit definition |
| P(a $0 CPU pregate on the existing CLIP unit caches returns positive) | **3–5 %** | this is ISR gate α, which F66 already ran and measured flat (+0.0012 / +0.0032) |
| P(the ~2-3 GPU-h Qwen unit extraction changes the verdict) | **≤ 5 %** | the only new variable is the encoder, and Law IV says a frozen encoder converts on HateMM only |

---

## PROVENANCE

- Ban ledger: `autoresearch/goal_mllm_plus3/state/directions_tried.json` (S2S entry `:130-133`,
  `banned_constraints` `:454-463`); `state/findings.jsonl` F35, F37, F39, F42, F66, F94.
- Records read directly: `SEG_REENCODE_FORENSIC_RECON.md` (the ISR recon — operator pincer, cache audit,
  extraction cost), `research-wiki/experiments/exp-seg-mode-ablation.md` (Δ-1),
  `research-wiki/ITERATION_LOG.md:591-602` (Δ-1 paired table), `research-wiki/EXP_mm_segment_keys.md`
  (multimodal segment keys), `ERRPAT_HateMM_2026-07-26.md`, `LITSWEEP6_MEMBANK.md`.
- **Caches read (train split only, read-only):**
  `data/CLIP_Embedding/{HateMM,MHC,MHC_zh}/train_subclipK4_openai_clip-vit-large-patch14-336_HF.pt`,
  `data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt`,
  `data/gt/HateMM/hate_spans.json`, `data/gt/HateMM/train.jsonl`.
- **Reproducibility note, stated rather than hidden:** §4's numbers were produced by an inline `python3`
  read, **not** by a hash-frozen script, and are therefore **recon-grade, not gate-grade**. They are
  descriptive geometry/coverage statistics used to price a direction, not measurements against a frozen bar.
  If any of them is ever cited in the paper, it must first be re-emitted by a frozen, reviewed script under
  the standard ceremony. The protocol above is stated in enough detail to make that mechanical.
- **Required statements:** ZERO GPU / SLURM / Modal / training spent by this recon; **no test-split file was
  opened**; no held-out test metric read or produced; no `state/` mutated by this file; no prereg, config,
  or frozen artifact touched.
