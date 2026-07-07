# EXP_p8b — vision-grounded evidence summaries as the text channel (P8 extra arm)

**Status:** **DONE — ZH gate CLOSED, no training. Vision grounding measurably beats text-only
summaries (B_vision 0.7409 > B_text 0.7271, the on-screen-text clause works) but is dominated by
naive raw-text truncation (C 0.7910): MLLM summarization translates ZH→EN and paraphrases away the
ZH lexical signal that C keeps.** · **Started:** 2026-07-07 · **Finished:** 2026-07-07 ·
**Owner:** subagent P8b · **Parent:** `EXP_p8_semantic_compression.md`

**Why.** P8's text-only summary arm gated **CLOSED on ZH** (train probe: B 0.727 < A floor 0.738,
while naive-truncation C hit 0.791) — expected, because ZH transcripts barely truncate (median ~26
tokens) and W2 found **ZH hate evidence is visual / on-screen-text**, which a text-only summary of the
Title+Transcript cannot see. P8b tests the fix: a **vision-grounded** summary — Qwen2.5-VL reads
8 sampled frames + Title+Transcript and writes the same ≤60-word evidence-dense summary, explicitly
instructed to **transcribe on-screen text verbatim** (captions/memes/overlays the channel CLIP image
encoder cannot read, especially Chinese). Same schema/harness as P8 → drop-in extra arm.

## Pre-registration (frozen)

### Generation (label-free, no leakage)
- `scripts/analysis/generate_vision_summary.py` + `scripts/slurm/generate_vision_summary.sbatch`.
  Qwen2.5-VL-7B-Instruct, greedy, 8 uniform frames (same sampler as segment scoring) + Title+Transcript.
  Prompt = P8's exact ≤60-word condensation wording + the single frozen addition: *"Transcribe any
  visible ON-SCREEN TEXT (captions, memes, overlaid words, signs) verbatim into the summary if it
  could be hateful or offensive."* So B_vision vs B_text isolates the vision grounding.
- Output `data/Summaries_vision/<DS>/<split>.jsonl` = {id, label, orig_text, summary} — byte-identical
  schema to P8's `data/Summaries/`, but a **parallel dir** (never clobbers the text-only arm).
- **MHC_zh primary** (the dataset P8 closed on); MHC/HateMM optional secondary.

### Encoding + conditions (reuse P8's harness EXACTLY, apples-to-apples)
- `scripts/analysis/p8b_build_cache.py` imports P8's `encode_single` (≤75 content tokens, one
  CLIPTextModel forward, pooler) and writes `<split>_p8vsum_HF.pt` (copies floor img/ids/labels
  VERBATIM, asserted `torch.equal`; only text_feats replaced). Drop-in `--model p8vsum_HF` swap.
- Conditions carried to test if the gate opens: **A** floor (raw chunk-mean), **B_vision**
  (p8vsum), **C** naive first-70-token truncation (the rent test), **D** optional concat.

### Probe gate (train-side, BEFORE any training) — HIGHER bar than P8-EN
`scripts/analysis/p8b_probe.py` reuses P8's `rep()` + `loo_knn()` (LOO kNN over
`[l2n(img)|l2n(text)]` @k20 on ZH TRAIN). **Gate: B_vision must beat BOTH**
- **A** (floor, 0.738) **AND**
- **C** (naive truncation, **0.791**) — the number to beat, per team-lead.

The C bar is the honest one: on ZH, shorter single-chunk text alone (no MLLM) already scores 0.791,
so B_vision only earns a role if the *vision-grounded semantics* (on-screen-text transcription) push
past that. If B_vision ≤ C → **kill, no training** (vision grounding adds nothing over shorter text).

**P8 ZH probe numbers (from p1-prior, the bars to beat):** A(floor) **0.7375**, B_text **0.7271**
(this is why P8 closed on ZH), C(naive-trunc) **0.7910**. So B_vision must clear **0.7910** — the
strong control on ZH — to earn a role; merely beating B_text (0.727) or A (0.738) is not enough.

### Training (only if gate opens)
**Efficiency (coordinated with p1-prior): A and C are the SAME floor/trunc caches for both arms, so
their P8 A/C results ARE this arm's baseline — do NOT retrain them.** Train only **B_vision**, 3 seeds
{0,1,2}, `scripts/slurm/train_p8vsum.sbatch` (flags matched EXACTLY to p1-prior's P8 recipe:
align/triplet/cos/hybrid, topk20, warmup5, dropout .2/.4/.1, `--model p8vsum_HF --exp_comment _p8Bvis
--group_name RAC_video_p8vsum`, FORCE=False). Trainlog `p8vsum_<ds>_Bvis_s<seed>.trainlog` (foldable
by p1-prior's `p8_collect.py`). Compare B_vision − A vs p1-prior's A; rent test B_vision vs their C.
Both protocols. Success (frozen, same as P8): B_vision > A by >0.01 macro-F1, ≥2/3 seeds, BOTH
protocols, **AND B_vision > C** (rent test), no >0.01 harm. Anything weaker = within-noise/no claim.

### Ops
Generation GPU job = `12427` (MHC_zh all splits), queued behind P2c 72B judges + P8 EN training.
Everything downstream (cache build, probe) is CPU. FORCE=False; no .pt in git; foreground sacct polling.

## Results — ZH gate CLOSED (no training)

Generation 12427 COMPLETED (44 min): MHC_zh train/val/test = 579/78/149 summaries, **0% empty**
(vision covers empty-transcript videos, unlike text-only). The on-screen-text clause works — samples
pull Chinese overlay text into the summary, e.g. *'The text "真正的痞子" appears, which translates to
"Real scumbag"'* (text the CLIP image encoder cannot read). 155/579 train summaries exceed 75 content
tokens (truncated at encode, same as P8).

**ZH TRAIN probe (LOO kNN @k20 over [l2n(img)|l2n(text)], reusing P8's rep/loo_knn):**

| cond | text channel | acc@k20 | macro-F1 | note |
|---|---|---|---|---|
| A | floor raw chunk-mean | **0.7375** | 0.6367 | matches p1-prior exactly |
| B_text | text-only summary (P8) | 0.7271 | — | (p1-prior) why P8 closed on ZH |
| **B_vision** | vision-grounded summary | **0.7409** | 0.6710 | +0.0034 vs A, **+0.0138 vs B_text** |
| C | naive first-70-token trunc | **0.7910** | 0.7527 | the strong control |

**GATE: B_vision ≥ A ✓ but B_vision ≥ C(0.791) ✗ → CLOSED. No training** (per pre-registration).

## Verdict

- **Vision grounding is a real, measurable improvement over text-only summarization** (B_vision 0.7409
  > B_text 0.7271, +1.4 pt) — the on-screen-text transcription clause did exactly what it was meant
  to (surfacing Chinese overlay/caption text CLIP's image encoder is blind to), and it lifts the ZH
  summary probe above the floor A. That part of the hypothesis is confirmed.
- **But it does not earn a method role**, because on ZH the MLLM summary — text OR vision — is
  **dominated by naive single-chunk truncation of the raw text (C 0.791)**. Root cause: Qwen writes
  the summary in **English** (translating/paraphrasing the ZH Title+Transcript), so the CLIP text
  encoder loses the ZH surface forms that the raw-ZH truncation C retains. Compression/translation
  throws away more discriminative lexical signal than the on-screen-text grounding recovers.
- Consistent with W2 and P8: on ZH the win is "keep the raw (short) text in one chunk", not "have an
  MLLM rewrite it". The vision channel helps at the margin but cannot overcome the translation loss.
- Honest kill at the probe (no test measurement spent). If a future variant wants to salvage this,
  the lever is **summarize in Chinese** (avoid the ZH→EN translation loss) — pre-registered as P8c below.

---

# P8c — Chinese-language vision summaries (the diagnosed salvage lever)

**Status:** **DONE — ZH gate CLOSED, no training. The Chinese-forced summary is 99.8% compliant and
evidence-dense yet scores the WORST of all arms (0.7168) because CLIP's English-centric text encoder
byte-fragments Chinese (97% of summaries truncated at 75 tok) and encodes it weakly. Root cause of the
ZH summary-input failure is the FROZEN ENCODER, not the summary content. Summary-input family closed on
ZH with complete three-arm attribution.** · **Started:** 2026-07-07 · **Finished:** 2026-07-07 ·
**Owner:** subagent P8c.
Team-lead fired this after P8b: P8b measured the OCR gain (+1.4 pt through translation loss) and
diagnosed the loss as ZH→EN paraphrase dropping ZH surface forms. P8c isolates it by forcing the
summary INTO CHINESE. p1-prior's caveat (probe-maybe / train-no; the whole summary-input family hit a
probe≠training wall on EN) is noted — hence cheap scope: one gen job + CPU probe, **no training unless
the gate opens**.

## Pre-registration (frozen)
- **Generation:** byte-identical to P8b except the frozen prompt forces Chinese output and keeps
  on-screen text verbatim (`要求:摘要必须用中文书写;画面中出现的屏幕文字…请原样保留`), ≤90 汉字
  (≈ the ≤60-word budget). Same 8 frames, greedy, all ZH splits.
  `generate_vision_summary.py --summary_lang zh --out_dir data/Summaries_vision_zh`
  (parallel dir; P8b's `data/Summaries_vision/` untouched). Gen job = **12430**.
- **Encoding:** `p8b_build_cache.py --summaries_dir data/Summaries_vision_zh --tag p8vsumzh`
  → `<split>_p8vsumzh_HF.pt` (reuses P8's encode_single; floor img/ids/labels verbatim).
- **Probe gate (ZH TRAIN LOO @k20, same harness):** `p8b_probe.py --vsum_tag p8vsumzh --label B_vision_zh`.
  **Gate: B_vision_zh must beat BOTH A (0.7375) AND C (0.7910)** — the same higher bar as P8b.
  Reference points: B_text 0.7271, B_vision(EN) 0.7409. B_vision_zh needs ≥ +0.050 over B_vision(EN)
  just from keeping Chinese to clear C.
- **Training:** only if the gate opens — 3 seeds, `--model p8vsumzh_HF`, trainlog
  `p8vsumzh_MHC_zh_Bvis_s<seed>.trainlog`, else HONEST KILL. If it fails too, the summary-input family
  is closed on ZH with a complete three-arm attribution (text-only 0.727 < vision 0.741 < vision-zh ?
  vs the naive-trunc bar 0.791).

## P8c results — ZH gate CLOSED (worst arm), summary-input family closed on ZH

Gen 12430 COMPLETED (41 min): 579/78/149 summaries, 0% empty, **Chinese-compliance 99.8–100%**
(meanCJK 0.97–0.98 — the 强制中文 clause held). Summaries are fully Chinese + evidence-dense, e.g.
`WHO：男性 WHAT：手淫对前列腺炎的影响及危害 主题：男性健康教育…`. **But at CLIP-text encode, 563/579
(97%) exceed 75 content tokens** (Chinese byte-fragments into many CLIP BPE tokens: "141 > 77") → heavy
single-chunk truncation.

**Three-arm attribution (ZH TRAIN probe, LOO kNN @k20, same harness):**

| cond | text channel | acc@k20 | macro-F1 |
|---|---|---|---|
| A | floor raw chunk-mean (multi-chunk, no single-chunk trunc) | 0.7375 | 0.6367 |
| B_text | EN text-only summary | 0.7271 | — |
| B_vision | EN vision summary | 0.7409 | 0.6710 |
| **B_vision_zh** | **CN vision summary** | **0.7168** | 0.5792 |
| C | naive first-70-tok raw ZH | **0.7910** | 0.7527 |

**GATE: B_vision_zh ≥ A ✗ (0.7168 < 0.7375), ≥ C ✗ → CLOSED. No training.**

## P8c verdict + P8-family close-out on ZH

- **The Chinese hypothesis is refuted, and the real bottleneck is now identified: the FROZEN ENCODER,
  not the summary content or its language.** The CN summary is fully compliant and dense, but CLIP's
  English-centric text tokenizer **byte-fragments Chinese** (a ≤90-char CN summary → ~140 CLIP tokens,
  97% truncated at 75) and its text tower encodes Chinese weakly. So the CN summary — despite keeping
  ZH surface forms — is encoded worse than the shorter raw-ZH truncation and even worse than the EN
  summaries. Keeping Chinese *hurts* here because of the encoder, not because of the content.
- **Complete three-arm attribution on ZH:** text-EN 0.727 < vision-EN 0.741 (OCR helps) but **CN 0.717
  (encoder byte-fragmentation dominates)**, all < naive raw-ZH truncation **C 0.791**. C wins because
  raw ZH is short (median ~26 tok → fits one chunk) AND keeps the exact surface forms CLIP was (weakly)
  trained on. **No MLLM summary variant — text or vision, EN or CN — beats naive raw-text truncation on
  ZH.** The summary-as-text-input family is **closed on ZH**.
- The real lever for ZH would be a **Chinese-capable text encoder** (e.g. a multilingual/Chinese CLIP
  or mpnet-zh text tower), not a better summary — but that changes the frozen encoder (a different
  experiment family, not this one). Honest kill at the probe; no test/training spent (one gen job total).
