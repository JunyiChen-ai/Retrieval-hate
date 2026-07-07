# P8 — MLLM semantic compression of the speech channel

Front: P8 (campaign goal, user-hardlocked: MLLM meaningfully + novelly integrated AND a
SUBSTANTIAL performance improvement). Highest-priority remaining route because it is grounded
in a POSITIVE finding we already own.

**Grounding.** The EN archive's only real effect was **truncation repair**: the transcript-key
recovered ~75% of the archive's F1 gain (`ABLATION_transcript_vs_archive.md`,
`exp-archive-knn-seeds` addenda). 71% of EN hate evidence is speech-borne; the text channel's
input is the raw Title+Transcript **chunk-mean-pooled over 77-token CLIP windows = dilution**.
What was NEVER tested: an MLLM-written **≤60-word evidence-dense summary** as the text-channel
INPUT, single-chunk encoded, with **end-to-end head retraining** (v1 archives were hallucinated
schema JSON; the transcript/archive key-swaps were eval-only, not a retrained text input).

Data check (2026-07-07): MHC-EN text median ~90 tokens (truncated → headroom); HateMM median
694 chars / max 80k (heavily truncated → strongest headroom; 39/744 train empty-text); MHC-ZH
median ~26 tokens (mostly fits → weak truncation headroom, matches ZH=visual/OCR-borne).

---

## PRE-REGISTRATION (locked before probe/training; written 2026-07-07)

### Generation (label-free preprocessing → no leakage)
- Model: Qwen2.5-VL-7B-Instruct, **text-only**, greedy, fixed prompt. ALL videos
  (train/dev/test) of MHC, MHC_zh, HateMM. Source = the `text` field of
  `data/gt/<DS>/{train,val,test}.jsonl` (Title+Transcript). Empty source → empty summary
  (logged).
- Prompt (frozen): *"Condense the following short-video Title+Transcript into at most 60 words,
  preserving WHO is targeted, WHAT is said or shown that could be hateful or offensive, and the
  overall topic. Output only the condensed text, no commentary."*
- Store `data/Summaries/<DS>/<split>.jsonl` = {id, label, orig_text, summary}. This is
  unsupervised input processing (labels never read) — identical leakage status to CLIP feature
  extraction / ASR / the E0b archive.
- CAVEAT (observed in smoke, recorded before results): the MLLM frequently writes the **ZH
  summary in English** (it translates while condensing). This confounds compression with an
  English-pivot on ZH — but it is the model's natural behaviour under a fixed English prompt,
  and the CLIP text tower is English-centric, so an English summary could *help* ZH the way the
  archive's English pivot did. We do NOT force output language (that would be a second, untested
  variable); the ZH probe gate and the ZH result are interpreted with this confound noted.

### Text encoding (single-chunk = the compression)
- `encode_text_single(t)` = truncate to ≤75 content CLIP tokens (+BOS/EOS), ONE CLIPTextModel
  forward, pooler_output → [768]. NOT chunk-mean (that would re-introduce dilution). Any
  truncation is asserted-logged (a ≤60-word summary should rarely exceed 75 tokens).
- Build ALTERNATIVE text_feats caches (do NOT mutate existing floor caches). Each cache copies
  the floor `img_feats`, `ids`, `labels` VERBATIM (asserted `torch.equal` / identical id order)
  and only replaces `text_feats`, so it is a drop-in `--model` swap for run_rac (P3 pattern).

### Conditions (frozen; one test measurement per cell)
- **A floor** = raw chunk-mean (the existing `openai_clip-vit-large-patch14-336_HF` cache).
  Must reproduce the published floor (EN 0.7826/0.7113, ZH 0.8054/0.7706, HateMM ~0.828 acc).
- **B (ours)** = summary replaces raw (single-chunk encode of the MLLM summary).
- **C naive-truncation control (THE RENT TEST)** = single-chunk encode of the **first 70 tokens**
  of the raw Title+Transcript (no MLLM). **B must beat C**, else the gain is just "shorter
  single-chunk text", not MLLM semantics.
- **D (secondary)** = concat `[l2n(raw chunk-mean) | l2n(summary single-chunk)] / √2` (text dim
  1536); gives the head both the diluted-full and the compressed-dense text.

### Probe gate (train-side, BEFORE training)
Per dataset, TRAIN split, no trained head. Leave-one-out kNN vote (cosine, similarity-weighted
"arithmetic", k=20) over `[l2n(img) | l2n(text)]` — the representation the head consumes.
**Gate: B's LOO-kNN accuracy ≥ A's on that dataset.** Report A/B/C for all three; a dataset's
training arm opens only if its gate passes. Diagnostics (not gates): img-only invariant check,
k∈{1,5,10,20}, macro-F1, and per-dataset summary-token-length distribution.

### Training (only for gated-open datasets)
3 seeds {0,1,2} × {A,B,C,D}, standard RAC_video_CLIP recipe (align fusion, triplet+hybrid BCE,
topk=20 arithmetic vote, 30 ep, Faiss CPU), GROUP `RAC_video_p8sum`, FORCE=False (distinct
`--model` tag per condition → no collision). Report BOTH protocols (val-selected warmup≥5 +
final-epoch), acc + macro-F1, paired per-seed deltas.

### Success criteria (pre-registered)
1. A reproduces the floor (bit-for-bit vs the RAC_video_CLIP floor at seed 0).
2. Probe gate passes on ≥1 dataset.
3. **B beats A with mean ΔmacroF1 > 0.01, ≥2/3 seeds positive, under BOTH protocols, on ≥1
   dataset, AND B > C** (the rent test — the gain is MLLM semantics, not just shorter text),
   with no >0.01 harm on a control dataset. The user's bar is a SUBSTANTIAL improvement —
   honest magnitude reported; anything weaker than (3) is within-noise / no claim / honest kill.

---

## PROBE-GATE RESULTS — EN OPENS, ZH + HateMM CLOSE

Generation job **12423** COMPLETED (45 min; 790 EN / 806 ZH / 1066 HateMM summaries; empty
summaries 0/0/74). Probe = train-side LOO kNN acc@k20 over `[l2n(img)|l2n(text)]`. JSON:
`scripts/analysis/p8_out/probe_gate.json`.

| dataset | A floor (raw chunk-mean) | B summary (ours) | C first-70-tok | gate B≥A | B≥C |
|---|---|---|---|---|---|
| **MHC (EN)** | 0.7359 | **0.7523** | 0.7067 | **OPEN** | ✓ |
| MHC_zh | 0.7375 | 0.7271 | **0.7910** | CLOSED | ✗ |
| HateMM | 0.7715 | 0.7702 | **0.7876** | CLOSED (tie −0.001) | ✗ |

- **EN opens** — the truncation-repair target: the ≤60-word summary beats the diluted chunk-mean
  floor (+1.6pt) AND the naive first-70-tok control (+4.6pt). Trained (job 12426, A/B/C/D × 3
  seeds). Whether the probe gain survives the learned align-fusion head is the open question
  (cf. the P3-HateMM "probe necessary-but-not-sufficient" lesson).
- **ZH / HateMM close.** On both, the naive first-70-tok control C is the BEST probe (ZH 0.791,
  HateMM 0.788): single-chunk (no dilution) helps, but the MLLM summary does NOT beat raw
  single-chunk — "shorter helps" ≠ "MLLM semantics help". ZH: summary hurts (short ~26-tok text
  + partial English-translation loss). HateMM: B≈A tie. Not trained (pre-registration).
- **ZH English-translation rate = 13%** (87% ZH summaries stayed Chinese — the English-pivot
  confound is much smaller than the smoke suggested). Truncation on single-chunk encode: B
  (summary) rarely truncated (EN 13/549, HateMM 13/744) — summaries fit; ZH B 362/579 truncated
  is an artifact of CJK tokenization, not length.

## RESULTS — FAIL (the strongest probe of the campaign still does not survive training)

EN training job **12426** COMPLETED (12 runs = A/B/C/D × 3 seeds). Only EN opened the probe.
JSON: `scripts/analysis/p8_out/p8_results.json`.

- **Bit-for-bit A PASS:** A seed0 val-sel = 0.7826 acc / 0.7113 maF1, exact vs the RAC_video_CLIP
  floor. Trustworthy.

EN, mean over 3 seeds, TEST macro-F1 (acc):
| condition | val-selected | final-epoch | Δ maF1 vs A (val / final) | seeds+ (val / final) |
|---|---|---|---|---|
| **A** floor (raw chunk-mean) | 0.6715 (0.762) | 0.7202 (0.779) | — | — |
| **B** summary (ours) | 0.6482 (0.739) | 0.6409 (0.733) | **−0.023 / −0.079** | 1/3 · 0/3 |
| **C** first-70-tok (rent) | 0.6056 (0.733) | 0.6620 (0.758) | −0.066 / −0.058 | 1/3 · 0/3 |
| **D** concat[raw\|sum] | 0.7335 (0.791) | 0.6127 (0.743) | +0.062 / −0.108 | 3/3 · 0/3 |

### What happened
- **B (summary) HURTS EN under both protocols** (val-sel −0.023, final-epoch **−0.079**, 0/3
  seeds positive on the stable protocol). The pre-registered bar (B−A>0.01, ≥2/3 seeds, both
  protocols, AND B>C) FAILS on every clause.
- **The rent test fails in the wrong direction:** on final-epoch B (−0.079) is *worse* than the
  naive first-70-token control C (−0.058) — the MLLM summary does not even beat blind truncation
  once trained.
- **D is pure val-selection noise:** val-sel +0.062 (3/3) but final-epoch −0.108 (0/3, the WORST
  cell). The two protocols flip sign by 0.17 maF1 — a textbook selection artifact, no claim.
- **The decisive lesson (sharpest instance in the campaign):** P8 had the *strongest* no-head
  probe of any front — the EN summary beat both the floor (+1.6pt) and the rent-test control
  (+4.6pt) at the probe — yet the trained retrieval head does WORSE on the compressed text than
  on the raw chunk-mean. **A passing no-head probe is necessary but not sufficient** (cf.
  P3-HateMM). Mechanism: the single-chunk summary is a *lossy* re-encoding; the learned
  align-fusion head exploits the full raw (even diluted) text better than the MLLM's paraphrase,
  and the compression discards signal the head would otherwise use.

### Verdict
**MLLM speech-channel semantic compression does NOT earn a method role and does not produce the
substantial improvement the user's goal requires.** On the one dataset whose probe opened (EN),
the compressed summary text input is net-negative vs both the floor and naive truncation once the
head is trained; the only "positive" (D val-sel) is a selection artifact that inverts on the
stable protocol. The truncation-repair premise is real at the probe level but does not translate
through end-to-end training. Honest kill. (The vision-grounded variant P8b, run in parallel by
p3-pool on ZH under GROUP RAC_video_p8vsum, is a separate arm — its verdict is reported there.)

### Cross-reference — the ZH summary-input family is fully closed (P8b/P8c)
The vision-grounded (P8b) and Chinese-language (P8c) summary arms on ZH, run by p3-pool
(`EXP_p8b_vision_summary.md`), close the family with a mechanistic diagnosis: ZH train-probe
acc A 0.7375 / B_text 0.7271 / B_vision(EN) 0.7409 / **B_vision_zh(CN) 0.7168** / C(raw first-70)
**0.7910**. The CN summary is 99.8% Chinese-compliant and evidence-dense yet scores WORST —
because the frozen **English-centric CLIP text tower byte-fragments Chinese** (≤90-char CN
summary → ~140 CLIP tokens, 97% truncated at 75) and encodes it weakly, so raw-ZH truncation
(short, one chunk, exact surface forms) wins. So on ZH the summary-input bottleneck is the
**frozen encoder**, not summary content/length/language — the real lever is a Chinese-capable
text tower (multilingual/CN CLIP or mpnet-zh), a different experiment family.

### Jobs / artifacts / repro
- Generation + cache build: `scripts/analysis/p8_generate_summaries.py` +
  `scripts/slurm/p8_generate_summaries.sbatch`. Summaries `data/Summaries/<DS>/`, caches
  `data/CLIP_Embedding/<DS>/{split}_p8{sum,trunc,concat}_..._HF.pt`.
- Probe gate: `scripts/analysis/p8_probe_gate.py` (CPU). Training: `scripts/slurm/train_p8sum.sbatch`.
