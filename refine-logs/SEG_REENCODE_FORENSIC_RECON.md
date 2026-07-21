# SEG-RE-ENCODE FORENSIC RECON — independently re-encoded per-segment Qwen features

**Agent:** forensic-recon subagent (zero-GPU). **Date:** 2026-07-21 NZST.
**Cell:** independently re-encoded per-segment Qwen2.5-VL features (red-team **untested-cell #5** /
`REDTEAM_BAN_SCOPE_AUDIT.md` **§GAP-2**, ranked **#3** in that audit's Part D table / **F61 cell (5)**).
The tasking's "ban-scope GAP-3" = the audit's Part-D **rank-3** row; the section body is labelled GAP-2.
Same cell; no ambiguity.

**Discipline honored.** CPU-only reading + forensic arithmetic. **ZERO GPU / SLURM / Modal / training /
test-touch.** No `state/` mutated, no prereg written, no job submitted. One deliverable (this file).
Committed on `main`, not pushed.

**Mission.** GO / GO-IF / NO-GO on the cell, with the **operator-survival analysis as the centerpiece**:
does *any* operator survive **both** the S2S/F37 don't-pool ban **and** the F47/Law-III selection ban?
If none survives, the cell is NO-GO regardless of extraction.

---

## 0. BOTTOM LINE

**VERDICT: GO-IF (mandatory $0 CLIP pre-gate) — else NO-GO. Realistic expectation: the $0 gate kills it
and no GPU is ever spent.**

- **Operator survival (load-bearing).** **Exactly one** operator class survives the *letter* of both bans:
  a **fixed, uniform, per-segment aggregation with NO per-item selection** — either symmetric MeanMaxSim
  set-matching, or the not-yet-tried **per-segment-kNN vote-mean**. Every **selection** operator
  (best-segment / route-to-most-hateful-segment) is **dead on two independent bans** (Law III per-item
  selection + P11 segment-score-selection). The survivor lives only on an **inductive-leap escape** (the
  don't-pool family ban is F35-*prefix*-scoped for Qwen and *CLIP*-scoped for W2-B; frame-local Qwen is
  neither), and it is precisely the object that **already died twice** (W2-B frame-local CLIP, S2S prefix
  Qwen), in both cases leaving an oracle it structurally cannot reach.
- **The decisive sharpening.** In *both* dead neighbor cells the surviving oracle headroom (+0.07–0.14) is a
  **per-segment SELECTION** ceiling (per-query argmax over segment index, using gold). The only operator
  that survives the selection ban does **no selection**, so it can only realize the symmetric/pooled part
  of that headroom — which both neighbors measured at **≈ 0** at the decision level. The headroom is the
  **wrong kind** for the surviving operator. This is the whole ballgame.
- **The only genuinely-new variable** vs the two dead cells is the **encoder** (frozen Qwen vs frozen CLIP,
  at frame-local granularity). W2-B's own verdict already priced that axis: "prior lowered, NOT vetoed"
  (CLIP<Qwen). And Structural Law IV says a **frozen** encoder converts on **HateMM only** — so even a
  positive would most likely be HateMM-only, **not** the ≥2-dataset EN gain the goal needs.
- **Cost.** Extraction needs **raw video** → **LOCAL GPU only** (Modal is hard-blocked from raw video per
  CLAUDE.md). K=4 primary on HateMM+EN, all splits ≈ **~2–3 GPU-h**; a $0 CPU probe on top. The $0
  pre-gate below can pre-empt all of it.
- **Honest prior.** **~5–10%** of a ≥+1pt gain on any single dataset (below the red-team's 10–20%, after
  the two-for-two-dead + selection-locked-headroom sharpening); **near-0** for the EN ≥+0.03 the goal
  requires. **D7:** in-pillar retrieval-method probe (not encoder-identity), so D7-clean as a probe — but
  a frozen-feature zero-training operator carries the "probe-pass ≠ train-gain" caution (fired ≥5×).

---

## 1. THE CELL, PRECISELY

**Object.** For each video, cut it into **K contiguous segments**; run **K independent frozen
Qwen2.5-VL-7B forwards** (one per segment, over that segment's own frames) → **K frame-local**
img/text embeddings per video. This is **F35-immune by construction**: F35 (`S2S_GATE0A_POSTMORTEM.md`,
`4358ca1`) proved that a *single*-forward frame-group `g_t` is a **cumulative causal-prefix summary**
(`is_causal=True`, position dominates content, diff-colour-same-position cos 0.939 > same-colour-diff-
position 0.674). An **independent** per-segment forward has no such prefix contamination — each segment
vector summarizes **only its own frames**. That is the entire premise of the cell.

**Encoder(s).** Frozen Qwen2.5-VL-7B (local, F8). The LoRA'd variant is a *possible* extension but is
**out of scope for the first probe**: (a) it doubles cost; (b) Law IV says LoRA's gain is text-borne and
leaves the image stream flat (`51eb95b`), so a per-segment *image* operator gains nothing from LoRA;
(c) keep the probe a clean frozen-feature retrieval-metric test, exactly as S2S/W2-B were.

**Segmentation machinery (already fixed by the campaign).** Canonical K ∈ **{4, 30, 60}** — the
`data/MLLM_scores/**/train_segscoreK{4,30,60}_qwen.jsonl` P6/P10 grid and W2-B's frozen partition
(K4 primary, K30 sensitivity). **Adopt W2-B's exact contract:** K=4 **contiguous** blocks primary
(matches `train_subclipK4_*` on disk), K=30 sensitivity — so the operator comparison is apples-to-apples
against the existing frame-local CLIP caches (see §4). HateClipSeg `gold_segments.json` exists but is the
localization set (P6/P11), not a classification-video segmentation — do NOT introduce gold spans (banned
constraint [1]).

**Operators on top (the candidate set that must clear the two bans — §2):**
1. **POOLED** (incumbent null) — mean the K segment vectors → one key → deployed pooled-cosine kNN.
2. **SET / MeanMaxSim / Chamfer / ASYM** — symmetric set-matching (the S2S/W2-B operator).
3. **Best-segment / per-segment SELECTION** — pick the segment that (by some score) decides the video.
4. **Per-segment-kNN vote-mean** — each segment runs its own kNN retrieval; the K segment-votes are
   combined by a **fixed uniform** rule. (The one operator W2-B did *not* run.)

---

## 2. OPERATOR-SURVIVAL ANALYSIS — THE CENTERPIECE

Two ban scopes must **both** be cleared. Verbatim:

**BAN A — S2S/F37 retrieval-object "don't-pool" family** (`directions_tried.json` S2S + CTF; F37/F35/F39):
> "family-level: retrieval-object/don't-pool family **CLOSED across encoders** (W2-B frozen-CLIP + S2S
> Qwen both dead)"; "do not re-propose **temporal structure over Qwen framesets in any operator class**."

but its two constituent measurements are **narrower** than that family sentence:
- **W2-B ban is CLIP-scoped** (`W2B_VERDICT_REVIEW.md §D`): "zero-training **frozen-CLIP** sub-clip
  set-matching … does **NOT** veto the Qwen-token S2S line" (CLIP<Qwen asymmetry, §E).
- **S2S ban is prefix-scoped** (`S2S_PROBE_VERDICT_REVIEW.md`; W2-C ban): "any revival must first **defeat
  F35** (groups are cumulative prefixes, not frame-local states) **with a different representation
  object**." Independent re-encoding **is** that different representation object.

So the family-level "across encoders / any operator class" sentence is an **inductive leap** for the
frame-local-Qwen corner (the red-team GAP-2 finding, and correct on the record). BAN A is a **theorem** for
{CLIP-frame-local} and {Qwen-prefix}; it is a **leap** for {Qwen-frame-local}.

**BAN B — F47 / Law III, per-item selection** (`ROUTER_GATE_RECORD.md`; `DRAFT_analysis_chapter.md §3.8`):
> "Do NOT re-propose per-item selectors over frozen channels regardless of feature family or nonlinearity
> unless the selector input is a genuinely NEW information source not derivable from banked
> features/votes." Transferable caution: "a per-item selector is admissible only if its input can be
> shown, **from banked evidence, to align with which-arm-wins above q = 0.663**."

Plus **P11** (`directions_tried.json`): "MLLM segment scores as weak-sup training labels — probe fail;
**MIL already carries it**" — the segment-selection route specifically.

### 2.1 Operator-by-operator

| operator | vs BAN B (selection) | vs BAN A (don't-pool) | survives both? |
|---|---|---|---|
| **1. POOLED** | n/a (no selection) | n/a (it IS pooling — the null) | — (baseline, no novelty) |
| **2. SET/MeanMaxSim/Chamfer/ASYM** | ✔ no per-item selection | LEAP-escape only; **the exact operator dead in W2-B (frame-local CLIP) AND S2S (prefix Qwen)** | **letter: YES; empirically 2-for-2 dead** |
| **3. best-segment / per-segment SELECTION** | ✘ **COLLIDES** — a per-item selector choosing which segment decides; needs alignment > q_req from banked evidence, none exists; **P11 already killed it** ("MIL carries it"); the W2-B/S2S **oracle IS this selection** and "can NEVER be claimed as a result" | (moot) | **NO — dead on Law III + P11** |
| **4. per-segment-kNN vote-mean (fixed uniform)** | ✔ no per-item selection (uniform combiner) | LEAP-escape only; a *distinct* don't-pool object W2-B did not run; but Law I predicts a uniform vote-mean ≈ pooled-vote | **letter: YES (the cleanest survivor)** |

**The pincer.** Any operator is either (a) a **fixed symmetric aggregation** — which is the **don't-pool /
retrieval-object family** (BAN A), re-openable only on the F35-immunity + CLIP<Qwen leap; or (b) a
**per-item choice** — which is **selection** (BAN B: Law III + P11). There is **no third kind**. The moment
you make the segment combiner *non-uniform* to escape the "it's just pooling" objection, you re-enter
selection (Law III) or, if the weights come from MLLM hate-density, **P3** ("MLLM segment hate-density
pooling weights — probe pass, train flat, 3 datasets") or the **MLLM-scores-as-signal** ban (constraint
[5]). So the **only** legal non-pooling aggregation is the **uniform** one — the one closest to pooling
and least likely to add signal.

### 2.2 The decisive fact: the headroom is selection-locked

Both dead neighbors leave a **large oracle headroom** — W2-B +0.0776 (HateMM) / +0.0700 (EN);
S2S +0.0917 / +0.1399. The red-team cites this ("oracle headroom is real") as the reason to extract.
**But that oracle is a per-segment SELECTION ceiling** — `t*(Q)=argmax_t (2y_Q−1)·v_t(Q)`, choosing the
best segment index **with gold** (`W2B_PROBE_DESIGN.md §3.8`; `s2s` §6.4). Realizing it requires a
**selector** — which is exactly what **BAN B forecloses** (Law III: alignment > 0.663 unmeetable in-box;
P11: MIL already carries the max-pool). The **only operator that survives the selection ban does no
selection**, so it can access only the **symmetric/pooled** slice of the headroom — and W2-B/S2S measured
that slice at **≈ 0** (HateMM SET −0.0047; EN +0.0016, ~20× under bar). **The convertible headroom and the
legal operator are disjoint.** This is not a data gap the Qwen encoder can close; it is a structural
mismatch that holds for any frozen encoder.

### 2.3 Verdict of the operator analysis

> **One operator survives the letter of both bans** — a **fixed uniform per-segment aggregation** (MeanMaxSim
> or per-segment-kNN vote-mean) over frame-local Qwen segments, legal via the F35-immunity / CLIP<Qwen
> escape from the leap-scoped family ban. **All selection operators are dead** (Law III + P11 + the
> oracle-is-selection identity). The survivor is the family's **two-for-two-dead** object whose **only new
> variable is the encoder**, aimed at a headroom that is the **wrong kind** (selection, not aggregation)
> for it to reach. The cell is therefore **not NO-GO on operator-emptiness** — a legal operator exists —
> but it is NO-GO-leaning on **prior**, and it must be gated by the $0 pre-gate in §3 before any GPU.

---

## 3. $0 PRE-GATE (the cost control — run BEFORE any Qwen extraction)

The frame-local independent-segment **CLIP** caches **already exist on disk** (verified §4). They ARE the
frame-local segment answer at the *weaker* encoder. Two $0 CPU probes over them settle the cell before a
GPU-hour is spent:

**Gate α — the untried operator on existing CLIP caches.** Run operator **#4** (per-segment-kNN vote-mean,
fixed uniform) — the one operator W2-B never ran — over `train_subclipK4_*` (HateMM + EN) vs the pooled
key, reusing `w2b_probe.py`'s loader + `compute_metrics_retrieval` vote (video-level LOO, no test-touch,
Fano-calibrated). W2-B already showed MeanMaxSim/Chamfer/ASYM die on these caches; if #4 **also** fails to
beat pooling on frame-local **CLIP** (Law I predicts a uniform vote-mean ≈ pooled-vote), then the sole
surviving operator is flat on frame-local segments at the CLIP encoder, and the **only** remaining hope is
the CLIP→Qwen swap.

**Gate β — selection-ceiling arithmetic ($0, no run).** The W2-B oracle already quantifies the per-segment
**selection** headroom on frame-local CLIP (+0.0776 / +0.0700). Per Law III, converting it needs a selector
with banked alignment > q_req; none exists (F47 closed all three supervision sources; F49 arithmetic;
P11). So the convertible headroom is **selection-locked** and the surviving (non-selecting) operator cannot
touch it — an arithmetic pre-kill of the "large oracle ⇒ worth extracting" argument.

**Decision rule (pre-declared here, honored by the orchestrator — this recon does not run it):**
- **Gate α flat AND gate β confirms selection-locked** (the expected outcome) → **NO-GO.** The
  encoder-swap-only reed (frozen CLIP→Qwen) is too thin against a two-for-two-dead family + Law IV
  (frozen ⇒ HateMM-only) + selection-locked headroom, and spending Qwen GPU here would **repeat the exact
  W2-B→S2S mistake** the campaign already made once (S2S burned Qwen extraction on a family W2-B had
  already de-risked down; it died identically).
- **Gate α PASSES on CLIP** (operator #4 beats pooling on frame-local CLIP where MeanMaxSim did not —
  low-prior surprise) → **GO** for the Qwen extraction: a working frame-local operator on the *weaker*
  encoder predicts an even stronger Qwen result, and the CLIP<Qwen asymmetry then flips from headwind to
  tailwind.

This gate is **$0 CPU on banked caches** (W2-B machinery is authored, hash-frozen `d22aac02…`,
code-reviewed) and can run as a batch companion. It is the difference between spending 0 and ~3 GPU-h.

---

## 4. CACHES, EXTRACTION, COST

**Existing (frame-local CLIP, for the $0 gate) — verified on disk:**
```
data/CLIP_Embedding/{HateMM,MHC,MHC_zh}/{train,dev_seen,test_seen}_subclipK4_openai_clip-...-336_HF.pt
data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-...-336_HF.pt        # K30 sensitivity (train)
data/CLIP_Embedding/{HateMM,MHC}/train_subclipK4_mm_...pt                     # +Whisper-ASR text (train)
```
**Absent (the cell's object):** **NO Qwen/VideoMLLM per-segment features exist anywhere** (`find` over
`data/**` = 0). The cell is genuinely unextracted at the Qwen level — consistent with S2S having produced
only single-forward *prefix* framesets (`data/CLIP_Embedding/*/frameset_qwen7b_8f/`), never independent
segments.

**Extractor gap.** `generate_VideoMLLM_embedding_lora_HF.py` samples frames by `np.linspace` over the
**whole** video (`:172`) — one forward per video, no per-segment mode. `s2s_extract.py` does **one** forward
→ K causal-prefix groups (the thing being avoided). **New code needed:** a per-segment loop that cuts each
video into K contiguous frame-ranges and runs **K independent forwards**, mirroring W2-B's contiguous-block
contract (assert `subclip_parent == repeat_interleave(arange(V), K)`; video-level LOO; no flat `[V·K,D]`
bank — the W2-B B1 leakage guard transfers verbatim). This is a moderate, clean modification of
`s2s_extract.py` (swap "single forward → group" for "K forwards → K vectors"), reusing its bit-parity
G-recon gate per segment.

**Raw video present (extraction feasible LOCALLY):** `data/video/{HateMM,MHC,MHC_zh}/All/*.mp4` as
symlinks (HateMM 1066). **Data boundary:** per-segment extraction consumes raw video ⇒ **LOCAL GPU only**;
Modal is hard-blocked from raw video (`modal_probe_runner.py`), so — unlike S2S/W2-B — the *extraction*
cannot be a cloud probe. The **probe over derived features** could run on Modal (features-only), but a
local $0 CPU probe is simpler.

**Cost (anchored on the S2S single-forward extraction: HateMM 1066 vids / 949 s; MHC 790 vids / 1519 s on
one A100-80GB):** K=4 independent = ~4 forwards/video (fewer frames each). Empirically ≈ 3–4× S2S wall
time. **K=4 primary, HateMM+EN, all splits ≈ ~2–3 GPU-h** + $0 CPU probe. +~1.5 GPU-h if MHC_zh robustness
arm included. **K=30 sensitivity ≈ ~18 GPU-h — NOT recommended** (W2-B rule B2: sensitivity can never
rescue a failed K4 primary; it only characterizes a negative's breadth). Head-training §11 escalation, only
if the probe passes, is ~20–25 s/run × 3 seeds (trivial).

---

## 5. KILL-BARS SKELETON (reuse the frozen S2S/W2-B ladder verbatim — do not invent new bars)

Gate order, pre-declared, mechanical-arithmetic-then-independent-verdict-reviewer (house rule):
1. **Machine validity (Fano)** — ±1 gold-label key LOO vote acc **≥ 0.99** both datasets, else VOID.
2. **Bit-parity G-recon per segment** — fresh forward == banked, `maxabs ≤ 1e-3`, `cos ≥ 0.9999` (the S2S
   gate, applied per independent segment).
3. **Oracle kill-switch FIRST** — per-segment selection oracle Δ(oracle − POOLED) **< +0.04 on EVERY
   dataset ⇒ DEAD-family**. (Necessary-not-sufficient; the ceiling "can NEVER be claimed as a result".)
4. **HateMM raw bar** — paired Δ(operator − POOLED) **acc ≥ +0.05 AND mF1 ≥ +0.05**, bootstrap-5th > 0,
   above permutation-null 95th, **rank-only corroborating** (the S2S/W2-B P3-priced bar).
5. **MHC-EN survival bar** — paired Δ **acc ≥ +0.03 AND mF1 ≥ +0.03** (the binding-gap increment).
6. **House conditional-info floor** — if the operator is recast as a key, the G0-cond gate: Δacc over
   `Z_best` **< +0.040** with calibration accZA ≈ 1.0 ⇒ kill (the K9/APX/CTF standard).
7. **Dataset rule** — (a) DEAD-family / (b) BOTH clear / (c) SINGLE / (d) NEGATIVE. Only (a/b/c) that also
   clears 1–5 authorizes a §11 head-training prereg; no sensitivity-arm OR-ing.

**Pre-declared honest prior:** red-team said 10–20%; **adjusted to ~5–10%** (single-dataset ≥+1pt) after
the two-for-two-dead + selection-locked-headroom sharpening, **near-0 for the EN ≥+0.03** the goal needs.
The **expected** result is outcome (d) or (a) at the $0 CLIP pre-gate — i.e. **no Qwen GPU is spent.**

---

## 6. D7 / NOVELTY, DATASET ORDER, COLLISION-SAFETY

- **D7 / novelty.** In-pillar: a **segment-level retrieval operator** sits inside the retrieval-contrastive
  pillar, not the encoder-identity axis, so it is **D7-clean as a method probe** (unlike audio = catch-up).
  A frozen-feature, zero-training operator is treated exactly as S2S/W2-B were (legitimate probe). If it
  passed, the novelty claim would be "frame-local independent-segment retrieval object beats pooled-cosine"
  — genuinely method-family, but it must survive the "probe-pass ≠ train-gain" caution (fired ≥5×: P3, S2S,
  W2-A, router, FA) via the §11 head-training escalation.
- **Dataset order.** HateMM **primary** (image-borne / visually-grounded — where frame-local segment
  structure is most plausible, and the raw-bar anchor), EN the **binding survival** dataset (the goal's
  ≥+0.03 lives here). Reuse the S2S/W2-B two-dataset rule exactly; MHC_zh optional robustness color only.
  Note Law IV: a **frozen** encoder converts on HateMM-only, so a HateMM-only pass would **not** satisfy
  the goal — EN is the whole point.
- **Collision-safe naming.** The W2 wave is recorded "fully dead" (F42); reopening under a W2 label invites
  confusion. Propose **`ISR` (Independent-Segment Re-encode)**, or `SEG-RE`. Distinct from S2S (prefix),
  W2-B (CLIP), W2-C (order-kernel), CTF (supervised-pool). Caches must be named
  `*_segindepK4_qwen7b_*.pt` under a fresh `data/CLIP_Embedding/<ds>/segindep_qwen7b/` dir — **never**
  overwrite the `frameset_qwen7b_8f/` (S2S prefix) or `subclipK4` (W2-B CLIP) caches.

---

## 7. EXECUTION SKELETON (for the orchestrator — nothing here is run by this recon)

1. **$0 CLIP pre-gate (§3)** — author `scripts/analysis/isr_clip_pregate.py` reusing `w2b_probe.py`
   machinery; run gates α (operator #4 on `subclipK4` CLIP) + β (selection-ceiling arithmetic on the banked
   W2-B oracle). Independent code-review + hash-freeze, then local $0 CPU (or Modal features-only).
   **Expected: KILL → STOP, 0 GPU-h.**
2. **Only if gate α passes** — prereg `ISR` (adapt `S2S_PROBE_DESIGN.md` bars §5 verbatim; single-submit,
   independent review, hash-freeze). Author `scripts/analysis/isr_extract.py` (K independent forwards,
   per-segment G-recon parity, W2-B contiguous-block + no-flat-bank guards) → separate code review.
3. **Local SLURM extraction** — `sbatch scripts/slurm/isr_extract.sbatch` (no `--time`; K=4 HateMM+EN, all
   splits; ~2–3 GPU-h). Test splits extracted-but-unconsumed per S2S precedent (0 test-touch).
4. **$0 CPU probe** (local or Modal features-only) → RAW record → **independent zero-context verdict
   review** against the frozen bars → finding + `directions_tried` entry.

---

## PROVENANCE

- Ban ledger: `autoresearch/goal_mllm_plus3/state/directions_tried.json` (S2S, CTF, W2-C, W2-B, router
  `router_cross_channel_per_item`, MJ, P3, P11 entries; `banned_constraints`), `state/findings.jsonl`
  F35/F37/F39/F42/F47/F49/F57/F61/F64.
- Primary records read directly: `REDTEAM_BAN_SCOPE_AUDIT.md` (GAP-2/§rank-3), `REDTEAM_UNTESTED_CELLS.md`,
  `S2S_PROBE_VERDICT_REVIEW.md` (`2c96ab6`), `S2S_EXTRACTION_RECORD.md` (timings, `cc3d90e`),
  `W2B_PROBE_DESIGN.md` (r1) + `W2B_VERDICT_REVIEW.md` (frame-local CLIP precedent, oracle-as-selection,
  CLIP<Qwen scope), `W2A_EXTRACTION_RECORD.md` (Qwen-forward timings),
  `research-wiki/DRAFT_analysis_chapter.md §3.6–3.9` (Laws I–IV).
- Code/caches: `scripts/analysis/s2s_extract.py` (single-forward-K-groups, the avoided path),
  `src/utils/generate_VideoMLLM_embedding_lora_HF.py:172` (whole-video linspace, no per-segment mode),
  `data/CLIP_Embedding/*/…subclipK4…pt` (present), Qwen per-segment features (absent),
  `data/video/*/All/*.mp4` (present as symlinks; local extraction feasible),
  `data/MLLM_scores/*/train_segscoreK{4,30,60}_qwen.jsonl` (K-grid).
- **Required statements:** ZERO GPU / SLURM / Modal / training spent by this recon; no held-out test metric
  read or produced; no `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated. Committed on
  `main`, not pushed.
