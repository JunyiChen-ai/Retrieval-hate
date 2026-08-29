---
type: experiment
node_id: exp:exp-s2s-r3
title: "S2S — set-to-set retrieval over frozen Qwen2.5-VL per-frame token sets vs pooled-cosine (round-3 lead candidate C1): G0-cond zero-training oracle screen (PRE-REGISTRATION, DRAFT-UNREVIEWED)"
idea_id: ""
status: PREREG APPROVED-WITH-AMENDMENTS (r1 applied); scripts authored, AWAITING INDEPENDENT CODE REVIEW — no submission authorized
verdict: approved-with-amendments
confidence: n/a
date: "2026-07-14"
hardware: "Stage E (extraction, the ONLY GPU): ~1-2 GPU-h, 1x A100, single sbatch, one frozen Qwen2.5-VL-7B forward per video (visual/prefix span only; text_feats already banked). Stage P (probe): CPU only, minutes; all-pairs set-kNN over <=851 memory videos."
duration: "Stage E: ~1-2 GPU-h. Stage P: minutes on CPU."
provenance: "PRE-REGISTRATION ONLY — NO runs executed. Reuses banked pooled cache data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt (2026-07-02; verified 2026-07-14: HateMM 744/107/215, MHC-EN 549/80/161, Dv=Dt=3584, L2-normed, HateMM train has 1 zero-img guard row). Extraction lineage: src/utils/generate_VideoMLLM_embedding_HF.py (banked img_feats = normalize(mean over the PREFIX TOKEN SPAN [0:last-<|im_start|>] = vision-pad tokens UNION system/user-header/instruction text tokens) — NOT a mean over per-frame vectors; see §4/§6). Retrieval core: src/utils/metrics.py:262-320 (vote), src/model/evaluate_rac.py (faiss IndexFlatIP), src/model/classifier.py:110-127 (align head, downstream only). Candidate spec: research-wiki/ROUND3_NOVELTY_CANDIDATES_2026-07-14.md §C1. Gate mandate: research-wiki/REFLECTION_mllm_integration_failures.md §4. Executable probe spec + config-parity table + G-decomp/G-recon anchors: refine-logs/S2S_PROBE_DESIGN.md."
added: 2026-07-14T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "frozen-Qwen", "set-to-set", "late-interaction", "MaxSim", "Video-ColBERT", "DeepEMD", "OTAM", "HateMM", "MHC-EN", "G0-cond", "oracle-kill-switch", "representation-geometry", "pre-registered", "DRAFT-UNREVIEWED", "S2S"]
---

# S2S — set-to-set retrieval over frozen Qwen per-frame token sets (round-3 lead candidate C1) (PRE-REGISTRATION)

> **STATUS: `PREREG APPROVED-WITH-AMENDMENTS (r1 applied)`. The independent pre-registration review
> (`refine-logs/S2S_PREREG_REVIEW.md`) returned APPROVED-WITH-AMENDMENTS; the five blocking A1–A5 and
> seven non-blocking N1–N7 are folded in place (tagged `(r1: A#/N#)`; see §16). The three executable
> scripts (`scripts/analysis/s2s_extract.py`, `scripts/slurm/s2s_extract.sbatch`,
> `scripts/analysis/s2s_probe.py`) are now authored and are `AWAITING INDEPENDENT CODE REVIEW` — a
> SEPARATE gate. NOTHING is submitted; no submission is authorized until that code review passes and the
> scripts are hash-pinned (`S2S_PROBE_DESIGN.md` §10). Stage P consumes ZERO test touches. The downstream
> head-training formal stage (§11) is NOT authorized here — it is a later, separate pre-registration
> gated behind the Stage P oracle kill-switch (§7).**

**verdict:** `approved-with-amendments`. · **confidence:** n/a · **line name:** S2S (set-to-set). This is
round-3 candidate **C1**-by-mechanism but is named **S2S** throughout to avoid collision with the
dead QLoRA route `C1 e2eq` (`directions_tried.json`, 16th negative).

## 1. Purpose (one line) + audit lineage

Test — with a **zero-training, paired, oracle-gated** screen before any head GPU — whether keeping a
video as a **set of per-frame Qwen token vectors** and retrieving by a **set-matching** score
(MaxSim late interaction) recovers label-discriminative **alignment** structure that **mean-pooling
destroys before the first retrieval operation**, on HateMM and MHC-EN.

**Audit lineage.** The round-3 novelty scout (`ROUND3_NOVELTY_CANDIDATES_2026-07-14.md`) derived the
one structural axis every one of the 22 dead routes shares: **they all retrieve over ONE pooled
vector per video** and only ever (a) swapped which encoder produces that pooled vector (encoder-swap
/ B1 / B2 / P9 — the only class that ever cleared +3, HateMM only) or (b) bolted a low-bandwidth
decision-side signal onto the pooled-vector vote (P1–P5, P10, P11, TARC, archive-repair — all dead by
**D1**). **No route ever changed the retrieval OBJECT.** Qwen2.5-VL is a video-language model: a video
is natively a set of per-frame language-aligned token embeddings, and the pipeline throws all of that
away by mean-pooling before retrieval. This is the one untouched component, it is a
**representation-geometry** lever (D2 — the only class that ever converted), and it is **not** a
decision-side add (so D1 does not bite it).

Two project-internal facts say the signal is here: **P6 (POSITIVE)** — the MLLM *segment* localizer
beats the pooled memory (wv-AUC 0.5435 vs 0.5140, paired p=0.007): hate is **locally** concentrated,
and the project already proved segment structure carries label signal the pooled vector dilutes.
**encoder-swap (POSITIVE)** — the only +3 lever, and it is representation-level. The cross-domain
literature says the same independently: in few-shot video classification (the closest analog to a
~600–850-video kNN task) set-to-set / temporal-alignment matching over per-frame features robustly
beats pooling by **3–8 pts** — OTAM (CVPR 2020), DeepEMD (CVPR 2020), CMOT/TSAM (OT set-matching); in
retrieval, ColBERT (SIGIR 2020), ColPali (ICLR 2025), Video-ColBERT (CVPR 2025) show preserving
token/frame vectors + a late-interaction operator beats single-pooled-vector retrieval. **None of
this exists in hateful-video or hateful-meme** (RGCL / RA-HMD / MoRE all retrieve over pooled single
vectors).

## 2. The mechanism under test (cell definition)

**One-line mechanism.** Stop mean-pooling the visual stream. Run **one** frozen Qwen2.5-VL-7B forward
(the SAME forward that produces the banked `img_feats`), but keep a **per-frame set**
`{g_1..g_T}` of pooled hidden-state vectors (one vector per temporal frame-group — Video-ColBERT-style
frame vectors, NOT all spatial tokens) for every video. Retrieval distance becomes a **set-matching**
score (MaxSim / late interaction) instead of cosine between two pooled vectors. The kNN vote (top-20,
rank-weighted signed cosine — `metrics.py`) is UNCHANGED; only the retrieval object and the pairwise
score change. **(amend-0a′, ruling 20c0bf2)** Each set element `g_t` is a **cumulative causal group
summary** — the mean of the last-LLM-layer states over temporal group `t`, conditioned on frames `0..t`
(the Qwen LLM is **causal** over the video-first token stream, so a group's tokens attend only to earlier
positions; verified `modeling_qwen2_5_vl.py:723,:1244,:1371-1380`), **not** a frame-local segment
descriptor. Two videos that pass through **similar cumulative visual states** at some temporal stage now
match via MeanMaxSim on the aligned states, whereas pooled cosine collapses the whole trajectory to one
vector before matching. Whether set-MaxSim over these cumulative states beats the pooled vector is the
empirical question Stage P adjudicates under the oracle kill-switch (§6.4); a synthetic-stimulus probe
found content weakly encoded relative to context-depth in these reps (a prior-lowering, not
disqualifying, signal — `S2S_GATE0A_AMENDMENT_RULING.md` §B).

**Frame granularity is fixed by Qwen, and we state it honestly.** Qwen2.5-VL-7B has
`temporal_patch_size = 2` and `spatial_merge_size = 2` (config verified 2026-07-14). With the banked
`num_frames = 8`, the vision tokens partition into **T = 4 temporal groups** (each spanning a 2-frame
pair), laid out temporal-major. So the config-parity primary set has **T = 4 elements per video**, not
8 — each element is a 2-frame group's spatial-token mean. The 16-frame sensitivity arm (§8) gives
T = 8. We call these **frame-group vectors** and never overclaim "8 frames."

**Injection point + bandwidth class.** Retrieval representation **and** metric
(representation-geometry). Bandwidth = **T×** the current (a set of 4–8 frame-group vectors vs one
pooled vector) — strictly *higher*-bandwidth, the opposite of the low-bandwidth decision-side adds D1
kills. Encoder unchanged; no MLLM scalar/score; no added channel; no gold annotations in-method.

## 3. Why this cell is open — escapes D1/D2/D3 and every epitaph

- **vs P3 (segment hate-density pooling, dead):** P3 kept ONE pooled vector and re-weighted segments
  by an **MLLM hate-density SCORE** (decision-side, score-driven). S2S keeps the **full frame set** and
  changes the **distance metric**; no score, no weighting — matching is data-driven (MaxSim).
  Different injection point, different bandwidth class, **no MLLM score**. Non-isomorphic.
- **vs C2-SAV (attention-head mining, 18th, U-1 null):** SAV asked "do other heads/layers carry label
  *information* beyond last-layer-pooled?" → null, and **falsified the dilution hypothesis on MHC-EN**
  (concluded MHC-EN is data/label-limited). S2S makes **no** extra-information claim: it claims the
  **matching geometry** is better when you align frame-to-frame rather than pool-then-match (the
  DeepEMD/OTAM thesis — pooling discards ALIGNMENT, not bits; even if the pooled average contains every
  frame's information, pooled cosine cannot *align* the shared hateful frame across two videos). SAV's
  information-content null does not touch this. **CRITICAL HONESTY CARRY-OVER (§9, §13):** SAV's
  MHC-EN result means the prior that S2S rescues MHC-EN is **weaker** than for HateMM; the probe's
  paired Δ resolves it empirically, and MHC-EN is treated as the *binding-gap co-primary*, not
  assumed to convert.
- **vs encoder-swap (positive):** swapped WHICH encoder, kept pooled retrieval. S2S keeps the encoder,
  changes pooled→set. Different injection point; **composable** with the encoder swap. This also means
  a HateMM-only S2S win does **not** newly satisfy the ≥2-dataset goal (HateMM already passes via the
  swap) — see §7/§9.
- **vs P6 (localizer, positive):** P6 *scores* segments for a localization read-out. S2S *matches*
  frame-sets for classification retrieval. P6 supplies the premise (hate is local); S2S is the first
  route to convert that premise into the **main-table retrieval** object.
- **vs MoRE (WWW 2025, in-domain closest prior):** MoRE retrieves whole instances with a pooled joint
  retriever → mixture-of-experts. S2S changes the retrieval **distance** itself and votes; no MoE, no
  instance-as-context. Different object.
- **Bans check (`directions_tried.json:banned_constraints`):** single-dataset own-train-split memory
  ✓; no OCR ✓; no gold annotations in-method ✓ (frames are unlabeled tokens; gold used only as a
  probe ceiling, §6.4, compliant with REFLECTION §4); no cross-seed ensemble ✓; no external API ✓; no
  MLLM-score-as-signal ✓; no kNN-pool expansion ✓ (same memory, richer keys); local Qwen-7B only ✓;
  not a P1–P5 re-proposal ✓.

## 4. The extraction-correctness problem — banked pooled cache is NOT a mean over frames (KEY finding)

The team-lead-proposed extraction gate — *"the mean of per-frame vectors must match the banked pooled
vector per video to float tolerance"* — is **invalid as stated**, because the banked pooled cache is
**not** a mean over per-frame vectors. From `src/utils/generate_VideoMLLM_embedding_HF.py` (r1: N7 —
the prefix pooling is at `:303`, `pooled = last_hidden[:end].mean(dim=0)`; `end` is set at `:296-302`;
the `.float()` + L2-norm at `:321-322`; the earlier `:290-322` span-cite is corrected to these exact
lines):

```
banked img_feats[v] = L2normalize( mean_{i in [0:end]} h_i )
   end   = position of the LAST <|im_start|> (start of the assistant header)
   [0:end] span  = { system + <|im_start|>user header tokens }
                 ∪ { ALL vision-pad tokens }              # per merged spatio-temporal patch, NOT per frame
                 ∪ { fixed IMG_INSTRUCTION text tokens }
                 ∪ { <|im_end|> }
```

Two facts break the naive gate: (i) the pooled mean **folds in the non-vision tokens** (system /
header / instruction text) alongside the vision tokens, so it is not a pure visual pool; (ii) the
vision tokens themselves are per **merged spatio-temporal patch**, and there is **no per-frame
partition stored** anywhere. (`text_feats` is worse for the naive gate — it is a trailing
assistant-header *last-token-style* pool, `:304-318`, unrelated to frames. S2S does not use text_feats
as a set; it is a single vector, reused as-is for the pipeline anchor / with-text sensitivity arm.)

**Adapted extraction-correctness gate (replaces the naive gate; full spec + tolerances in
`S2S_PROBE_DESIGN.md` §4).** We define the frame-group vector `g_t` = mean of the last-layer hidden
states over the vision tokens in temporal group `t` (these last-layer states are **cumulative-causal**:
each conditions on frames `0..t`), and gate the frame set with **four** checks. Two of them (the grid gate
and the causal-prefix control) actually certify the frame set; the other two (G-decomp, G-recon) certify
the *aggregate*.

> **(r1: A1) Correction of the v1 overclaim — G-decomp and G-recon are grouping-INVARIANT.** Because
> `vis_pos` and `nonvis_pos` are complementary within `[0:end]` by construction,
> `Σ_t n_t·g_t + p_S = Σ[0:end]` and `(…)/end = mean[0:end]` **for ANY vision/text mask and ANY grouping
> of the vision tokens**. So G-decomp (and G-recon, which checks the pooled aggregate) would pass even
> with a **wrong `video_pad_id`** (wrong vision/text boundary) or a **wrong temporal grouping**
> (spatial-major misread). The v1 statements "any residual > 1e-5 means the token→frame assignment …
> is wrong" and "G-decomp proves the frame set is a faithful decomposition" are **retracted**; those
> gates prove only span-`end` match, partition completeness, and arithmetic self-consistency. The frame
> set itself is certified by the grid gate + causal-prefix control below.

- **(r1: A1) Grid-consistency gate (mandatory, exact, free).** From the model's own `video_grid_thw` and
  `spatial_merge_size` (=2): assert `n_vis == grid_t·(grid_h//2)·(grid_w//2)` (catches a wrong
  `video_pad_id`) **and** `T == grid_t` and `(n_vis // T) == (grid_h//2)·(grid_w//2)` (catches a wrong
  per-group size). This is the check that actually pins the vision/text boundary and the equal partition.
  **HALT** on violation.
- **(amend-0a′) Causal-prefix positive control (mandatory, HALT).** Two 8-frame clips sharing frames
  0–3 (groups 0,1) and differing at frames 4–7 (groups 2,3): assert the **shared prefix groups are
  invariant** (`cos(ĝ^P_k,ĝ^Q_k) ≥ 0.999`, `k∈{0,1}`) and the **changed groups diverge** (onset), plus
  within-clip distinctness — the only check that exercises the token→temporal-group assignment, and one
  that is **valid under causal cumulation** (a prefix summary is invariant to later frames; the old
  permutation-equivariance/argmax control was invalid by construction — smoke 13169 failed it
  scientifically, `S2S_GATE0A_AMENDMENT_RULING.md`). Temporal-major contiguity remains proved by the
  modeling source (`modeling_qwen2_5_vl.py:466-505` `get_window_index` builds
  `arange(...).reshape(grid_t, llm_grid_h, llm_grid_w)` (t,h,w) row-major; `:529-534` reorder;
  `:560-562` `merger` then `hidden_states[argsort(window_index)]` restores the original order).
- **G-decomp (mandatory, exact, free) — aggregate arithmetic only.** With `n_t` = #vision-tokens in
  group `t`, `p_S` = sum of hidden states over the **non-vision** prefix tokens, `|S|` their count, and
  `end = Σ_t n_t + |S|`:

  ```
  reconstructed_prefix_mean = ( Σ_t n_t · g_t  +  p_S ) / end
  L2normalize(reconstructed_prefix_mean)  ==  (this forward's own banked-formula pooled vector)   to <= 1e-5
  ```

  A pure algebraic identity; a residual > 1e-5 means a bug in the decomposition arithmetic (dropped
  token, wrong `end`, incomplete partition) → **HALT**. Necessary but **not sufficient** (grouping-
  invariant, per the A1 correction above).
- **G-recon (banked-cache parity anchor, tolerance-based).** The fresh forward's `L2normalize(full
  prefix mean)` vs the **banked** `img_feats[v]` cache: require **cosine ≥ 0.9999 AND max-abs-diff
  ≤ 1e-3** on the L2-normed vector (expected near-bit-exact on the same A100 + sdpa + bf16; the
  tolerance only absorbs cross-run bf16 kernel drift). This documents that the fresh forward **is** the
  banked forward — the analog of B5's "reproduce the deployed numbers to 4 dp" gate; also grouping-
  invariant.

Storage keeps `{g_t, n_t, p_S, |S|, end, grid_thw}` per video so the reviewer can recompute all four
checks offline.

## 5. Arms — paired, on identical frame vectors (binding design rule 1)

The scientific contrast is **pool-then-match vs match-then-pool on the SAME per-frame vectors**, which
cancels seed/representation noise and isolates the mechanism:

| arm | retrieval score s(Q, M) between query set Q, memory set M | role |
|---|---|---|
| **POOLED** (baseline) | `cos( mean_t g^Q_t , mean_t g^M_t )` — pool the frames, then cosine | the "pool destroys alignment" null |
| **SET (primary)** | **MeanMaxSim** `= (1/|Q|) Σ_{q∈Q} max_{m∈M} cos(ĝ^Q_q, ĝ^M_m)` (frame vecs L2-normed) | match-then-pool late interaction |
| **SET-Chamfer (single sensitivity arm)** | `0.5·[MeanMaxSim(Q→M) + MeanMaxSim(M→Q)]` | symmetric robustness variant |
| **PIPELINE-ANCHOR pooled** | pooled-cosine kNN over the **banked** `img_feats` (with G-decomp tie to §4) | internal reference to the banked cache (folds in non-vision text; NOT the primary null) |
| **WITH-TEXT (sensitivity)** | POOLED / SET visual score **+** fixed `cos(text_feats^Q, text_feats^M)`, identical additive channel in both arms | shows the visual mechanism survives alongside text |
| **ASYM (r3: C2 ablation cell)** | `max_{m∈M} cos(ĝ^Q_pooled, ĝ^M_m)` — pooled query × set memory (the `\|Q\|=1` reduction of MeanMaxSim; `ĝ^Q_pooled` = L2-normed pooled query) | the folded C2 candidate: the pooled-query × set-memory off-diagonal cell of the S2S grid, adjudicated vs symmetric SET |

The **primary decision** is on the **visual-isolated** POOLED-vs-SET paired Δ, because that is where
the mechanism lives and mixing in the shared text channel only dilutes the paired contrast. Every arm
feeds the **identical, real** vote (`metrics.py` `compute_metrics_retrieval`, `use_sim=True`,
`majority_voting='arithmetic'`, `topk=20`): only the pairwise *score* substituted into the vote
changes across arms; the vote machine is byte-identical. **OT / Sinkhorn is deliberately excluded from
the probe** (its temperature is a metric-shopping surface); it is deferred to a later stage only if the
probe survives. **One primary set-metric is declared before results: MeanMaxSim.** Chamfer is the
single pre-declared sensitivity arm.

**(r1: A2) Rank-only sim-neutralized co-diagnostic (mandatory).** The `metrics.py` vote multiplies each
neighbour's signed label by the arm's pairwise score as a WEIGHT (`metrics.py:262-320`), so MeanMaxSim's
compressed range vs pooled cosine could move the paired Δ even at identical neighbours — a false-PASS/
false-KILL surface. A **rank-only** variant of both POOLED and SET is therefore computed: retrieve
top-20 by the arm's own score, but neutralise the sim to a constant `1.0` so both arms use identical
rank-position weighting and differ only in *which* neighbours they retrieve. The sim-weighted
MeanMaxSim−POOLED Δ stays the reported **primary** (it mirrors the downstream vote), but is credited only
if the rank-only paired Δ **corroborates it** (same sign AND its own permutation/bootstrap significance).
Both are reported; an uncorroborated primary is a sim-scaling artifact.

**(r1: A3) Near-duplicate audit + near-dup-excluded sensitivity (mandatory).** SET can "win" by
re-discovering near-duplicate / same-source clips rather than aligning hateful segments across distinct
videos (the permutation null does not catch this). Pre-declared: flag any distinct memory pair with
`pooled_cos ≥ 0.995` OR `MeanMaxSim ≥ 0.995` (a lone shared hateful frame gives MeanMaxSim ≈ 1/T ≪ 0.995,
so the flag cannot swallow the signal); report the flagged-pair count at 0.98/0.99/0.995 for both metrics
plus the single-frame max-cosine distribution; and re-run LOO dropping flagged neighbours — the SET
advantage must survive. Full spec: `S2S_PROBE_DESIGN.md` §5.

**(r3: C2 fold) ASYM ablation arm — the folded C2 candidate (no separate ceremony).** The round-3 C2
candidate (asymmetric / multi-view MLLM memory) is **not** a separate route: its parameter-free core is
literally the pooled-query × set-memory off-diagonal cell of S2S's own MeanMaxSim 2×2 grid
(`ĝ^Q_pooled × set-memory`), computed on the **identical** frozen frame vectors and run through the
**identical** Stage-P LOO vote, paired, same seeds, with symmetric treatment in the permutation-null and
bootstrap machinery (the same per-seed permutations as the other arms). Forensic recon:
`refine-logs/C2MEM_FORENSIC_RECON.md` (verdict FOLD-INTO-S2S; standalone novelty fails the D7 bar =
MUVERA/ColBERT asymmetric-multi-vector; C2's learned best case is already upper-bounded by S2S's per-query
oracle-ceiling). **Pre-declared C2 kill logic:** (a) if S2S's oracle Δacc < +0.04 on **every** dataset
(the §6.4 kill-switch fires), the whole "don't-pool" family — S2S **and** ASYM — is **DEAD together**, no
separate ASYM adjudication; (b) if S2S's symmetric SET survives (oracle did not fire), ASYM is **dead
unless it beats symmetric SET on acc AND macro-F1 (paired) on ≥1 dataset** — otherwise asymmetric
multi-view memory adds nothing over S2S's symmetric operator, and a beating ASYM would escalate only as
the asymmetric arm of the §11 downstream stage (never a standalone route). **C2 has no separate ceremony;
the family is adjudicated by S2S's kill-switch.**

## 6. G0-cond probe — exact procedure + oracle kill-switch (binding design rules 2–4)

### 6.1 Mechanics (Stage E = the only GPU; Stage P = CPU, zero test touch)

- **Stage E (extraction).** One `sbatch` (spec: `S2S_PROBE_DESIGN.md` §3). One frozen Qwen-7B forward
  per video over HateMM (744+107+215) and MHC-EN (549+80+161) — **all** splits extracted so test
  frame sets exist for the *later* formal stage, but **Stage P touches only train+val** (§6.6 ledger).
  Byte-level config parity with the banked cache (§4; parity table in `S2S_PROBE_DESIGN.md` §2). Dump
  `{g_t, n_t, p_S, |S|, end}` per video. Run G-decomp + G-recon (§4) as HARD gates during/after
  extraction.
- **Stage P (probe, CPU).** No head, no training. Build the memory from **train ∪ val only**
  (HateMM = 851 videos incl. 1 zero-img guard; MHC-EN = 629). Leave-one-out: each train/val video is a
  query against the rest; retrieve top-20 by each arm's score; call the real vote; report AUC + acc +
  macro-F1, per dataset, per arm, **paired** (SET − POOLED on identical `g_t`).

### 6.2 Primary metric (declared before results)

**Leave-one-out (LOO) retrieval-vote on TRAIN ∪ VAL only**, per dataset, reported as **paired
Δ(SET − POOLED)** in **both accuracy and macro-F1**, plus AUC. Test is untouched at Stage P. LOO =
each memory video held out of its own retrieval. The vote is the pipeline's continuous rank-weighted
signed-cosine vote; acc uses the pipeline cut `sigmoid(vote) ≥ 0.5 ⇔ vote ≥ 0`; macro-F1 is the goal
metric (`metrics.py:309`).

### 6.3 Machine-validity calibration arm (Fano — MANDATORY, REFLECTION §4)

Before any Δ is interpreted, run a **label-oracle calibration arm**: replace the retrieval score with
gold-label agreement. **(r1: N2)** the pairwise score is `+1` if `label(q)==label(m)` else `−1`
(deterministic tie-break by memory index). The vote acc **must reach ≥ 0.99** on both datasets (full
Fano headroom). If it does not, the vote machine is **void** and **no negative verdict may be accepted**
(this is the exact C3-erratum guard: a probe that cannot convert the gold signal itself is broken). Gold
labels here are a machine check only.

### 6.4 ORACLE-CEILING kill-switch (pre-declared, binding — decides whether ANY head GPU is spent)

The mechanism's **upper bound** = a **gold-guided oracle frame-selection** MaxSim. **(r1: A4) Exact
deterministic procedure — per-query oracle frame selection, video-level gold ONLY, no time-span gold:**
for query `Q` with gold label `y_Q ∈ {0,1}` and L2-normed frame-groups `{ĝ^Q_t}`, for each candidate `t`
compute the single-query-frame score `s_t(Q,M) = max_{m∈M} cos(ĝ^Q_t, ĝ^M_m)` (LOO, `M≠Q`), run the real
vote to get the pre-sigmoid margin `v_t(Q)`, and select

```
t*(Q) = argmax_t  (2·y_Q − 1) · v_t(Q)          # frame that most confidently votes the CORRECT label
        tie-break: smallest index t
```

The oracle score for `Q` is `s_{t*(Q)}(Q,M)`; the memory side keeps FULL sets (never oracle-selected —
no double-dipping). Report paired **Δ(oracle − POOLED)**. Gold enters only to pick which of `Q`'s **own**
frames to trust (per-query, video-level, no per-frame/time-span annotation). This upper-bounds how much
frame-level alignment structure — that pooling discards — *could* buy if we knew the discriminative
frame. **(r1: N5)** oracle Δ should generally **≥** raw Δ; a raw Δ materially exceeding the oracle ⇒ an
oracle-construction bug (investigate, do NOT auto-KILL).

> **KILL-SWITCH (binding).** If the **oracle-ceiling** paired Δacc **< +0.04** on **every** dataset,
> then pooling was **not** discarding convertible alignment structure — the whole "don't-pool" family
> (S2S + its C2 multi-view cousin) is **DEAD**, zero head GPU, exhaustion re-confirmed for the
> retrieval-object cell. The oracle ceiling is an upper bound and can **NEVER** be claimed as a result.

### 6.5 Raw-effect survival bar (pre-declared; prices in P3's "probe pass ≠ train gain")

Survival additionally requires the **raw** (non-oracle) paired LOO effect to clear a bar set **high
enough to price the probe→train pessimism**:

> **RAW BAR (binding).** On the **primary dataset (HateMM)**, mean paired **Δacc ≥ +0.05 AND mean
> paired Δmacro-F1 ≥ +0.05** (MeanMaxSim vs POOLED), with the bootstrap 5th-pct of the paired Δ **> 0**
> (D3), the observed Δ **above the 95th pct of the permutation null** (§6.6), **and (r1: A2) the
> rank-only sim-neutralized arm corroborating the sign + significance** (an uncorroborated Δ is a
> sim-scaling artifact, not a mechanism effect). **Bar derivation:**
> the goal is +0.03 test acc+F1; the probe measures a **zero-training LOO Δ on train∪val**, which is
> **optimistic** on two counts — (i) LOO over train∪val has near-duplicate self-retrieval and no
> held-out generalization gap; (ii) at zero training the POOLED arm is un-adapted, so the head can
> later partially **compensate** the pooled deficit and shrink the SET advantage (this is exactly the
> P3 "probe passes, training goes flat" failure — repeated ≥4× in the graveyard). We price a ~1.7×
> pessimism factor → require **+0.05 raw** to believe **+0.03 converts** downstream. +0.05 is well
> inside the +3–8 pt band the few-shot-video set-matching literature reports, so the bar is demanding
> but not a moonshot.

### 6.6 Permutation null as a DISTRIBUTION + dataset rule + frame-budget

- **Permutation null (≥100 fresh seeds).** **(r1: N1)** the pre-declared seed set is **0..99**; for each
  seed, shuffle the per-video **frame sets across videos** (destroying the query↔memory alignment while
  preserving the marginal frame-vector distribution), with the **same** permutation applied to both arms
  within a seed so the paired Δ is preserved; recompute the paired Δ. The observed Δ must exceed the
  **95th percentile** of this null. Report the full null distribution, not a point. An optional finer
  per-frame-vector shuffle null is additionally reported (separates "alignment" from a generic
  "richer-key" effect).
- **Dataset rule (pre-declared; HateMM primary vs MHC-EN binding-gap — the honest ruling).** HateMM is
  the **primary mechanism-existence** dataset: highest prior (P6 locality strongest there; encoder-swap
  shows Qwen represents HateMM well; 851 memory videos). MHC-EN is the **binding-gap co-primary**: it
  is the dataset that actually advances the ≥2-dataset goal (HateMM already passes via the encoder
  swap), but its prior is **weaker** (SAV falsified dilution on MHC-EN; 629 memory videos,
  data/label-limited). Pre-declared outcomes: **(a) HateMM clears + MHC-EN clears** → strongest; both
  license a formal stage. **(b) HateMM clears, MHC-EN fails** → mechanism is **real** and licenses a
  HateMM formal stage (a scientific result, composable with the encoder swap), but the binding gap is
  **NOT closed** and this MUST be reported as such, not as "goal met." **(c) MHC-EN clears, HateMM
  fails** → advances the goal on the binding dataset (report the HateMM null honestly). **(d) neither
  clears** → DEAD, retrieval-object family closed. No post-hoc dataset shopping: these four rows are
  fixed now.
- **Frame-budget sensitivity (exactly 2 budgets, no shopping).** Primary = **8 frames (T = 4)** —
  byte-parity with the banked cache. Sensitivity = **16 frames (T = 8)** — a separate forward, gated by
  its OWN internal G-decomp (it cannot use G-recon against the 8-frame banked cache). No third budget.

## 7. Gate order (what runs, in order)

0. **(r1: A1 / amend-0a′) Grid-consistency gate + causal-prefix onset-invariance control (0a′)** (Stage
   E) — HALT if the vision/text boundary or per-group size mismatches the grid, or the two shared-prefix
   clips' groups {0,1} are not invariant / changed groups {2,3} do not diverge / groups collapse.
1. **G-decomp** (exact, per video) — HALT on any residual > 1e-5 (aggregate arithmetic; grouping-
   invariant — necessary not sufficient, per A1).
2. **G-recon** (banked-cache parity) — HALT on cosine < 0.9999 or max-abs-diff > 1e-3.
3. **Fano machine-validity arm** (§6.3) — void the probe if ±1 gold-label-key acc < 0.99.
4. **Oracle-ceiling kill-switch** (§6.4) — if oracle Δacc < +0.04 on every dataset → DEAD, no head GPU.
5. **Raw-effect bar + rank-only corroboration + permutation null + bootstrap** (§6.5, §6.6, A2) —
   survival test.
6. **Near-duplicate audit + near-dup-excluded sensitivity** (§5, A3) — SET advantage must survive.
7. **Dataset rule** (§6.6) — assign outcome (a)/(b)/(c)/(d).

**(r1: N3)** The sensitivity arms (Chamfer, WITH-TEXT, 16-frame, near-dup-excluded) **cannot rescue a
failed primary**: the single pre-declared primary is the MeanMaxSim visual-isolated paired Δ (acc AND
F1) under the §6.6 dataset rule, corroborated by the rank-only arm — no OR-ing beyond the four fixed
dataset-rule rows.

Only outcome (a)/(b)/(c) that also clears gates 0–6 authorizes drafting the **downstream head-training
formal pre-registration** (§11) — which is a **new, separately reviewed** document, NOT authorized
here.

## 8. D3 guards (binding design rule 4)

- **Bootstrap** ≥1000 resamples over the LOO query set (per dataset), re-report 5/50/95 pct of paired
  Δacc / Δmacro-F1; a pass whose 5th-pct crosses 0 is labelled **D3-FRAGILE**.
- **No per-seed training variance exists** (zero training) — so "3/3 seeds" is *not* claimed at Stage
  P; the paired design cancels representation noise instead. The formal stage (§11) is where 3-seed
  head variance enters.
- **Frame-budget sensitivity** (§6.6): 8 vs 16 frames, pre-declared, no post-hoc pick.
- **Zero-img guard rows** (HateMM train has 1) get a zero frame set and are handled **identically** in
  both arms, so the paired contrast is unaffected; their count is logged.

## 9. Novelty scope statement (honest; a user ruling, not decided here)

The raw *mechanism* (set-matching / late interaction for few-shot video and retrieval) is established
(OTAM, DeepEMD, CMOT; ColBERT, ColPali, Video-ColBERT). The honest novelty is **domain +
representation transfer**: first set-to-set retrieval in hateful-video, first over **MLLM
video-language frame tokens** (not raw CNN features), inside a retrieval-contrastive kNN-vote head.
Whether this clears the **novelty clause** is the **same pending user ruling** as B3-LoRA (a
D7-class decision), NOT decided here. This file decides only the **performance** clause, and only its
G0-cond screen. **Binding-gap honesty:** because HateMM already passes via the encoder swap, only a
**MHC-EN** (or ZH, later) pass newly advances the ≥2-dataset goal; a HateMM-only S2S win is a
mechanism result, not goal closure (§6.6 rule (b)).

## 10. Test-touch ledger (binding design rule 7)

| stage | test data used? | touches |
|---|---|---|
| Stage E extraction | test frame sets are *extracted and cached* (for the later formal stage) but **not scored** | 0 |
| Stage P probe (all arms, incl. oracle ceiling + Fano) | **train ∪ val only**; test never retrieved, never voted | **0** |
| Downstream formal stage (§11, not authorized here) | test scored under the frozen 3-seed both-protocol ceremony | (spent later, in that prereg) |

Stage P is a **ZERO test-touch** screen. The oracle-ceiling and Fano arms use **gold labels of
train∪val only**, as a probe ceiling / machine check (REFLECTION §4 compliant). **(r1: N4)** the probe
script enforces this fail-closed: it never constructs or opens any `test_seen*` file and asserts the
loaded memory size is exactly 851 (HateMM) / 629 (MHC-EN) so a stray test row cannot enter the memory or
the vote.

## 11. Downstream head-training design sketch (NOT authorized; later prereg only)

Only if Stage P survives: adapt the triplet+BCE head to the set representation. Sketch — a **shared
per-frame projection head** `φ` applied to every `g_t`, retrieval distance = MeanMaxSim over
`{φ(g_t)}`, trained by the existing triplet-margin + BCE objective with the set score substituted for
the pooled cosine; inference vote unchanged (top-20). Keep φ parameter-light (single linear
`3584→map_dim`, shared across frames) to bound overfit on 549–744 samples. Composability with the
encoder swap and with C2 (multi-view memory, which reuses this exact extraction) is noted. This is a
**brief sketch**; the full formal pre-registration (3-seed, both protocols, G-repro anchors,
verbatim +0.03/+0.03 rule) is written **after** the screen passes and is **separately reviewed**.

## 12. Cost estimate

| item | cost |
|---|---|
| Stage E extraction (both datasets, 8 frames, 1856 videos, 1 prefix forward each) | **~1–2 GPU-h**, 1× A100, single sbatch |
| Stage E storage (frame sets + gate side-artifacts, fp16) | **~80–110 MB** (8 frames) / ~160–210 MB (16 frames) — **sub-GB** |
| Stage P probe (all arms, both datasets, ≥100 null seeds, ≥1000 bootstrap) | **minutes on CPU** |
| Downstream head-training formal stage (only if screen passes) | ~1–2 GPU-h (later prereg) |

Extraction is ~**half** the banked 7B extraction (that ran two forwards/video — prefix + response; we
need only the prefix/visual forward, since `text_feats` is already banked).

## 13. What-would-kill-this table

| # | killer | where |
|---|---|---|
| K0 | (r1: A1 / amend-0a′) grid gate: `n_vis` ≠ grid count (wrong video_pad_id) or per-group size ≠ grid; or causal-prefix control 0a′ fails (shared-prefix groups {0,1} not invariant / changed groups {2,3} do not diverge / groups collapse) | §4, §7 gate 0 → HALT |
| K1 | G-decomp residual > 1e-5 (decomposition arithmetic bug — dropped token / wrong `end` / incomplete partition; **NB (r1: A1)** grouping-invariant, so this does NOT catch a wrong video_pad_id or grouping — K0 does) | §4, §7 gate 1 → HALT |
| K2 | G-recon cosine < 0.9999 or max-abs > 1e-3 (fresh forward ≠ banked) | §4, §7 gate 2 → HALT |
| K3 | Fano ±1 gold-label-key acc < 0.99 (vote machine void) | §6.3, §7 gate 3 → probe void |
| K4 | oracle-ceiling Δacc < +0.04 on **every** dataset (no headroom) | §6.4, §7 gate 4 → DEAD, no head GPU |
| K5 | raw HateMM Δacc < +0.05 OR ΔmF1 < +0.05 (pooling was ~lossless / effect too small to survive P3 shrinkage) | §6.5, §7 gate 5 |
| K5b | (r1: A2) rank-only sim-neutralized arm does NOT corroborate the primary Δ (sign/significance) → sim-scaling artifact | §5, §6.5, §7 gate 5 |
| K6 | observed Δ ≤ 95th pct of the frame-shuffle permutation null (matching artifact) | §6.6 |
| K6b | (r1: A3) SET advantage does not survive near-dup-excluded retrieval (duplicate-rediscovery artifact) | §5, §7 gate 6 |
| K7 | bootstrap 5th-pct of paired Δ crosses 0 (D3-fragile) | §8 |
| K8 | MHC-EN fails while HateMM passes → mechanism real but binding gap not closed (honest partial) | §6.6 rule (b), §9 |

## 14. Honest prior / expected outcome (declared before running)

**Prior: FAIR** (best of anything left, per the scout). Falsifiable: if per-frame set-matching LOO
does not beat pooled-vector LOO by a paired margin projecting to +3 test acc on at least one dataset's
oracle arm, S2S is dead — pooling was lossless for retrieval here. Realistic band: strongest on
**HateMM** (locality/P6 evidence); **MHC-EN is the honest coin-flip** (SAV says it may be
data-limited, not dilution-limited). The most likely *informative* outcomes are (b) HateMM-only
(mechanism real, gap open) or (d) both fail (family closed) — either is a decisive, cheap result.

## 15. Connections

- Round-3 candidate spec: `research-wiki/ROUND3_NOVELTY_CANDIDATES_2026-07-14.md` §C1 (+ §C2 shares
  this extraction).
- Graveyard + bans + D1/D2/D3: `autoresearch/goal_mllm_plus3/state/directions_tried.json`,
  `research-wiki/REFLECTION_mllm_integration_failures.md`.
- Gate mandate + calibration erratum: `REFLECTION_mllm_integration_failures.md` §4.
- House rigor precedent: `refine-logs/B3_PREREG_REVIEW.md`, `refine-logs/B5_PREREG_REVIEW.md`.
- Executable spec + config-parity table + G-decomp/G-recon anchors + probe script plan:
  `refine-logs/S2S_PROBE_DESIGN.md`.
- Extraction lineage: `src/utils/generate_VideoMLLM_embedding_HF.py`; retrieval core
  `src/utils/metrics.py:262-320`, `src/model/evaluate_rac.py`, `src/model/classifier.py:110-127`.

## 16. Revision history

- **2026-07-14 — v1 DRAFT-UNREVIEWED.** Initial pre-registration. Establishes Stage E (extraction,
  only GPU), Stage P (zero-training zero-test-touch oracle screen), the adapted extraction-correctness
  gate (G-decomp + G-recon; the naive "mean-of-frames == pooled" gate is **invalid** — banked pooled =
  mean over the vision∪text prefix token span, §4), the MeanMaxSim primary set-metric + Chamfer single
  sensitivity arm, the oracle-ceiling kill-switch (+0.04), the raw survival bar (+0.05 acc AND F1,
  P3-priced), the Fano machine-validity arm, the permutation-null distribution, and the honest
  HateMM-primary / MHC-EN-binding-gap dataset rule. Awaiting independent pre-registration review.
- **2026-07-14 — r1 APPROVED-WITH-AMENDMENTS (amendments applied).** Independent pre-registration review
  (`refine-logs/S2S_PREREG_REVIEW.md`) returned APPROVED-WITH-AMENDMENTS. Folded the five blocking
  A1–A5 + seven non-blocking N1–N7 in place: **A1** retracted the G-decomp/G-recon overclaim (they are
  grouping-invariant, §4) and added the HARD grid-consistency gate + synthetic temporal positive control
  (gate 0) + the `modeling_qwen2_5_vl.py:466-505,529-534,560-562` layout citation; **A2** rank-only
  sim-neutralized co-diagnostic + corroboration rule (§5, §6.5); **A3** near-duplicate audit +
  near-dup-excluded sensitivity (§5, §7 gate 6); **A4** pinned the exact deterministic per-query oracle
  frame-selection formula, video-level gold only (§6.4); **A5** hash-freeze + this history (in
  `S2S_PROBE_DESIGN.md` §10–§11). Non-blocking: **N1** null seeds 0..99 same-permutation-both-arms +
  optional per-frame null (§6.6); **N2** Fano ±1 score (§6.3); **N3** sensitivity-cannot-rescue-primary
  (§7); **N4** fail-closed no-test-touch guard (`S2S_PROBE_DESIGN.md` §5, §10 ledger); **N5** oracle≥raw
  ordering (§6.4); **N6** independent-verdict output split (`S2S_PROBE_DESIGN.md` §5); **N7** provenance
  cite corrected to `:303` (§4). The three executable scripts are authored to the amended spec and are
  AWAITING INDEPENDENT CODE REVIEW; no submission is authorized. Spec ambiguities resolved by the
  implementer are recorded in `S2S_PROBE_DESIGN.md` §12.
- **2026-07-14 — r2 CODE-REVIEW FIXES APPLIED.** Independent code review
  (`refine-logs/S2S_CODE_REVIEW.md`, verdict APPROVED AFTER FIXES) landed three blocking + two
  non-blocking + four note fixes in the scripts (docs unchanged in substance): **B1** G-recon compared a
  CUDA tensor to a CPU tensor (`.cpu()` both before comparing) — the extractor would have crashed on the
  first real video; **B2** decoupled G-recon from `--limit` so the mandated `SMOKE=1` run exercises all
  four hard gates (PREREG_REVIEW §5(iii)); **B3** the A2 rank-only corroboration now has the rank-only
  arm's OWN permutation null + bootstrap significance (not sign-only), per the pre-registered credit
  rule; **NB-a** NEG_INF filter in the near-dup-excluded vote; **NB-b** the `gpu:a100:1` gres verified
  schedulable (node advertises `gpu:a100:8`; banked-cache producer used it); notes N-i/N-ii/N-iii (dead
  no-op removed, unused param dropped, sbatch exit cosmetic) and N-iv (offline-G-decomp wording:
  authoritative residual is the inline f32 number, `S2S_PROBE_DESIGN.md` §4). Scripts re-hashed (r2 table
  in `S2S_PROBE_DESIGN.md` §10). Still AWAITING the reviewer's one-line hunk re-check; no submission
  authorized.
- **2026-07-15 — r3 FOLD C2 AS ASYM ABLATION ARM (probe-only amendment).** Per
  `refine-logs/C2MEM_FORENSIC_RECON.md` (verdict FOLD-INTO-S2S), the round-3 C2 candidate is folded into
  S2S as one pre-declared ablation cell rather than a separate route/ceremony: **ASYM** =
  `max_{m∈M} cos(ĝ^Q_pooled, ĝ^M_m)` (pooled-query × set-memory, the off-diagonal cell of the MeanMaxSim
  grid), added to §5 arms and to `s2s_probe.py` through the identical LOO vote, paired, same frozen
  frames/seeds, with symmetric permutation-null + bootstrap treatment. Pre-declared C2 kill logic: (a)
  S2S oracle Δ<+0.04 everywhere → don't-pool family (S2S+ASYM) dead together; (b) SET survives → ASYM
  dead unless it beats symmetric SET on acc AND macro-F1 (paired) on ≥1 dataset. **Probe-only change —
  the r2 extractor/sbatch remain byte-identical and their §10 hashes are UNCHANGED**; only `s2s_probe.py`
  + both docs are re-hashed (r3 table in `S2S_PROBE_DESIGN.md` §10). The queued smoke 13159 (extractor,
  r2 pins) is untouched. Awaiting the code reviewer's diff-only re-check before Stage P; Stage P is anyway
  gated on extraction.
- **2026-07-16 — r4 POST-FAILURE AMENDMENT: gate 0a → 0a′ (causal-prefix onset-invariance control) +
  premise reword.** Binding independent amendment ruling `refine-logs/S2S_GATE0A_AMENDMENT_RULING.md`
  (commit 20c0bf2, verdict **(B)-REPLACE**), triggered by smoke 13169 failing old gate 0a as a *scientific*
  gate (`S2S_GATE0A_POSTMORTEM.md`). Old gate 0a assumed frame-local permutation-equivariance, which is
  invalid **by construction** for cumulative-causal group vectors (Qwen LLM `is_causal=True`; smoke 13169
  `match=[1,0,3,3]≠σ`). Replaced with the causal-prefix **onset-invariance** control (0a′): two clips
  P=[R,G,B,Y]/Q=[R,G,Y,B] sharing frames 0–3 — shared groups {0,1} invariant (`cos≥0.999`), changed groups
  {2,3} diverge, within-clip distinct — a check **valid under cumulation and still discriminative** of
  spatial-major/reversed/interleaved grouping (§D.1). §2/§4 premise reworded to "cumulative causal group
  summaries" (§D.2/§D.3); gate table → 0a′; §7 gate-0 + §13 K0 reworded. **This is evidence-driven, not
  outcome-driven: no Stage-P bar moved** — oracle +0.04, raw +0.05/+0.05, rank-only corroboration,
  permutation null, bootstrap, dataset rule all stand verbatim; the premise is honestly weakened, not the
  gate softened. Extractor-only code change (`temporal_positive_control` → `causal_prefix_control`);
  sbatch + `s2s_probe.py` UNCHANGED. Re-smoke REQUIRED (gates 0b/1/2 never reached on a real video);
  authorized submissions: ONE, after the reviewer's diff-only re-check. r4 hash table in
  `S2S_PROBE_DESIGN.md` §10.
