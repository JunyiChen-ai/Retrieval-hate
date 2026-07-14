# W2-E Forensic Recon — Unsupervised fine-grained hate-mode **prototype memory**

**Agent:** zero-GPU / zero-SLURM / zero-Modal forensic recon (pure code + cache + literature-note reading).
**Target:** wave-2 candidate **W2-E** ("unsupervised hate-mode prototype memory", memory-organisation axis).
**Source candidate text:** `research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md` §W2-E (ranked #5/5, prior LOW,
novelty LOW; author's own note: "included because the seed named it and it is a cheap TODAY probe").
**Conditioning kill:** `refine-logs/W2B_VERDICT_REVIEW.md` (W2-B KILLED, outcome (d) NEGATIVE) — read in full.
**Ground truth read (not memory):** `src/utils/metrics.py:262-320` (rank-weighted signed-cosine top-20 vote),
`src/model/evaluate_rac.py:80-155` (memory build → `IndexFlatIP`, one head-projected L2-normed vector per
train video), `scripts/analysis/w2b_probe.py:1-380` (the S2S/W2-B LOO-kNN + oracle + Fano + perm-null + boot
machinery W2-E would reuse verbatim), `autoresearch/goal_mllm_plus3/state/directions_tried.json` (24 dead +
9 bans). Cache shapes verified on disk (see §D).

---

## VERDICT LINE

> **NO-GO** (fails D7-tightened at the novelty gate *before* performance matters — same failure class as
> R3-C3geo which recon killed pre-ceremony; carries **zero new information** over the flat kNN so D1 bites
> maximally and D2's winning class is not even entered; W2-B's surviving-oracle headroom argument does
> **not** transfer; zero-training nearest-prototype/mode-local kNN is a strict *coarsening* of full kNN over
> ~600 exemplars → performance prior near-zero and below the D3 noise floor). Cheap-to-probe (caches
> sufficient, §D) does **not** rescue a novelty-gate failure. Recommend closing at recon; do **not** spend
> prereg/review/freeze ceremony. Optional fallback in §F if the user wants a family-closing datapoint.

---

## A. MECHANISM — what would W2-E actually change?

**Baseline (verified in code).** `evaluate_rac.py:80-155` builds the memory as **one L2-normed,
head-projected vector per train video** added to a `faiss.IndexFlatIP`; `metrics.py:262-320` retrieves
top-20 and votes with **rank-weight × signed-cosine** (`retrieved_labels*2-1` × sim, rank weight
`flip(arange(1,k+1))`). In the **zero-training probe** form (what W2-E proposes, on banked pooled Qwen) the
head projection is dropped and the raw pooled 3584-d Qwen vector is the key — exactly as W2-B/S2S probes do.

**W2-E's proposed change** (sharpest train-split-only / no-gold / plausibly-MLLM version): discover
`K` prototypes **per class** by **unsupervised clustering (k-means/ProtoNet-style centroids) of the banked
Qwen pooled representation**, then restructure retrieval as one of:
1. **Prototype-as-key** — replace the ~600 per-video memory vectors with `2K` class-prototype centroids;
   query votes over nearest prototypes ("denoised keys"); or
2. **Mode-local kNN** — assign query → nearest prototype (mode), then run the top-20 vote **only within that
   mode's exemplars** (local neighbour pool).

**The decisive structural fact.** *Both variants operate on the SAME single banked pooled Qwen vector per
video.* Clustering is a **deterministic, lossy function of those very vectors**. Neither variant introduces
any new signal, channel, view, or interaction term — unlike W2-A (adds a cross-modal interaction *uncomputable*
from the marginals) or W2-D (adds an acoustic channel). W2-E only **reorganises / compresses information the
flat kNN already has**. Variant (1) is a nearest-centroid classifier = a strict coarsening of kNN; variant
(2) restricts the neighbour set to a subset of what flat kNN already ranks, so it can only **remove** good
neighbours, never add one. This is a **memory-organisation / decision-side reorganisation**, *not* a
representation-level lever. (The candidate text hedges "D2-ish"; honestly it is **not** D2 — the representation
is byte-identical to the flat baseline's.)

Requirements check: (i) train-split-only ✓ (cluster on own train∪dev pooled). (ii) no gold annotations ✓
(unsupervised k-means; note the *oracle* arm in §C is where gold would enter, and that is where it becomes
vacuous). (iii) "MLLM-integrated enough" — **NO**: k-means over Qwen vectors is identical in mechanism to
k-means over CLIP vectors; nothing about the operation is MLLM-specific. This is a textbook trick with a Qwen
coat of paint — the exact anti-pattern D7-tightened names.

---

## B. NOVELTY vs literature + vs our own dead list

**Standalone: FAIL D7-tightened.** Prototype / centroid / condensed memory in retrieval-augmented
classification is textbook (ProtoNet, Snell et al. 2017; nearest-centroid; k-means memory condensation;
prototype-based few-shot is a whole subfield). The candidate author concedes this ("Prototype/centroid memory
is textbook few-shot (ProtoNet)"). It is a *generic trick transfer*, precisely the class D7 was tightened to
exclude — and directly analogous to **R3-C3geo** (`directions_tried.json` id `R3-C3geo`), which forensic recon
**killed pre-ceremony** for "textbook 2024-25 trick, cannot clear D7-tightened novelty bar." W2-E is arguably
worse than C3geo: C3geo at least altered the training signal (hard-negative mining); W2-E alters nothing —
it's a zero-training memory reshuffle carrying no new information.

**Composite: also FAIL.** The proposed composite ("MLLM-representation-derived fine-grained hate-mode
prototypes in a video retrieval-contrastive head") does not survive because the "MLLM" adjective is inert:
prototypes work over any encoder, so the composite reduces to "k-means over a frozen encoder's vectors," with
no *new MLLM-specific integration*. Contrast W2-A, whose composite ("MLLM cross-modally-grounded internal
representation as the retrieval key, so implicit visual–transcript incongruity enters retrieval geometry")
does carry an MLLM-specific mechanism (an interaction term a dual encoder structurally cannot produce). W2-E
has no such sliver. In-domain prior-art proximity: MoRE (WWW 2025) already does pooled retrieval → MoE
routing over hate-video experts; "hate has distinct modes routed to experts" is *closer to already-published*
than the candidate acknowledges.

**Non-isomorphism vs specific dead ids** (mechanism, not outcome-class): non-isomorphic to P3
(score-reweighted single vector), archive-auto-repair (MLLM-vote *deletion*), C2 (multi-view *expansion* — W2-E
is the opposite, cross-exemplar compression), W2-B (set-matching, retrieval-object axis). So W2-E is **not
banned** by any existing id and does **not collide** with the W2-B ban scope (frozen-CLIP sub-clip
set-matching). *However* it is isomorphic in **outcome-class** to the meta-family "zero-training unsupervised
reorganisation of frozen features as a retrieval-geometry accuracy lever" — the family W2-B just produced the
cleanest negative in (see §C). Banned-constraint scan: no OCR ✓, no gold-in-method ✓, no cross-seed ensemble
✓, single-dataset train split ✓, no API ✓, not P1–P5 ✓, no target-as-structure ✓. It is flagged
"memory-curation-adjacent" (the archive-auto-repair ban is on **MLLM-vote deletion**, which W2-E is not) — not
banned, but adjacent.

---

## C. ORACLE / HEADROOM LOGIC — **does NOT transfer from W2-B**

**Why W2-B's oracle survived.** W2-B's memory is a genuine **set of K = 4 *different* sub-clip vectors** per
video. Its oracle (`w2b_probe.py:249-277`) used the video-level gold label to **select, per query, which of the
K sub-clips** to trust — a meaningful "does the *right view* exist among K genuinely different views?"
question. It survived (+0.0776 / +0.0700 acc) because such a view exists; the unsupervised MeanMaxSim simply
could not pick it → outcome (d) NEGATIVE (structure present, unsupervised metric can't realise it on frozen
CLIP).

**Why the argument fails to transfer to W2-E.** W2-E has **one vector per query**, not K different views. There
is no "which-of-K-views" question, so no analogous non-vacuous oracle:
- **Gold-label prototype routing** (build class-conditional prototypes, route each query to its *gold* class's
  prototypes) is **label-leaking / circular** — it inflates accuracy by injecting the label into routing, and
  measures nothing about convertible structure. Inadmissible as a headroom oracle.
- The only *admissible* (non-leaking) formulation is **oracle cluster-routing**: cluster the memory
  *unsupervised* into K modes, then use gold knowledge only to pick, per query, the mode whose **local** kNN
  vote is most correct. But mode-local candidate pools are **subsets** of the global kNN pool, so this oracle
  can only help by **excluding out-of-mode false neighbours** — and the flat vote already discounts those by
  cosine × rank (`metrics.py:270`). Expected headroom is **thin**.
- **Decisive:** even a *surviving* W2-E oracle would **inherit W2-B's lesson directly** — W2-B just proved that
  when frozen-feature headroom exists, the *unsupervised zero-training conversion step fails*. W2-E's conversion
  step (unsupervised mode-routing must match oracle routing) is the *same* frozen-feature unsupervised step
  that just failed. So a surviving oracle would not upgrade the prior; it would reproduce (d).

**Net:** the headroom argument that kept W2-B from being (a) DEAD-family does **not** rescue W2-E; W2-E has
*less* structure than W2-B (1 view vs K) and *no* non-vacuous surviving-oracle path.

---

## D. CACHE SUFFICIENCY — **YES, zero new GPU** (verified on disk)

A full zero-training W2-E probe (cluster train∪dev pooled Qwen → prototype-anchored / mode-local LOO kNN vs
flat kNN) runs entirely on **already-banked pooled Qwen 7B caches**, CPU-only, ~$0 on Modal, no extraction.
Verified shapes (`torch.load`, `weights_only=False`):

| dataset | train pooled Qwen | dev_seen pooled Qwen | primary memory (train∪dev) | pos(train/dev) |
|---|---|---|---|---|
| HateMM | `train_Qwen2.5-VL-7B-Instruct_HF.pt` img `(744,3584)` txt `(744,3584)` | `dev_seen_…_HF.pt` `(107,3584)` pos 43 | **851** | — |
| MHC-EN | `train_…_HF.pt` `(549,3584)` | `dev_seen_…_HF.pt` `(80,3584)` pos 25 | **629** | — |
| MHC_zh | `train_…_HF.pt` `(579,3584)` | `dev_seen_…_HF.pt` `(78,3584)` pos 28 | **657** | — |

- Each cache dict = `{ids, img_feats, text_feats, labels}` — **one vector per video**, labels present
  (needed for oracle/Fano arms and for `concat(img,text)` baselines). Counts **851/629** match W2-B's
  `EXPECTED_MEM_PRIMARY` exactly → the W2-B loader/guard/vote/oracle/Fano/perm-null/bootstrap machinery
  (`w2b_probe.py`) is **reusable verbatim** (swap the sub-clip set-matrix builder for a k-means +
  prototype/mode-local retrieval builder; the LOO vote `run_vote`, `oracle_ceiling`, `fano`,
  `permutation_null`, bootstrap stay identical). `sklearn 1.5.2`, `scipy 1.17.1`, `faiss 1.13.2`,
  `numpy 1.26.4` all present in `HateVideo` → k-means available on CPU.
- **`concat(img,text)`** (7168-d) is the honest baseline (flat kNN over the full pooled key); it is
  constructed on the fly from the two banked tensors — no new file.
- Modal volume `rgcl-features` already holds HateMM+MHC+MHC_zh pooled per orchestrator note; **no new
  uploads** (and none would be raw video regardless — features-only `.pt` + labels).
- **Not needed / not proposed:** subclip K4/K30 (`(2976,1024)` etc., that is the W2-B object), 32B, LoRA,
  audio (`data/audio/MHC` empty). W2-E is pooled-Qwen-only.

**Conclusion: cache-sufficient TODAY, zero GPU.** (This is the *only* dimension on which W2-E scores well — and
it does not rescue the novelty-gate failure.)

---

## E. KILL-BAR SKETCH (only if a probe were run — for completeness, not a recommendation)

House-discipline bars, reusing the W2-B/S2S instrument so no new machinery risk:
- **Sole primary arm (K4-analogue):** declare **train∪dev pooled Qwen, flat-kNN vs prototype/mode-local kNN**
  the sole survival-determining contrast; K (cluster count) and variant (prototype-key vs mode-local) sweeps
  are **sensitivity only, never survival-determining** (mirrors B2), to bar hyperparameter shopping across
  K∈{2..8}.
- **Oracle kill-switch FIRST** (admissible = *unsupervised-cluster oracle-routing*, §C; the gold-class-routing
  form is inadmissible/label-leaking and must be excluded): DEAD-family iff oracle Δ(oracle-routed − flat)
  `< +0.04` on **every** dataset. Report the oracle honestly as thin-by-construction.
- **Fano validity** ≥ 0.99 (±1 gold-label-agreement key, `w2b_probe.py:283`) — verdict admissible only if valid.
- **Conversion-taxed raw bars:** HateMM anchor Δacc **AND** ΔmF1 ≥ **+0.05** vs flat kNN, corroborated by a
  rank-only arm; MHC-EN survival Δ ≥ **+0.03/+0.03** — identical to W2-B, since the P3 shrinkage tax applies.
- **Permutation null ≥ 100 seeds** (same-perm across arms, `permutation_null`); **bootstrap 1000**, 5th-pct > 0
  required (D3-fragility guard).
- **Fail-closed:** never open `test_seen`; assert memory video-count == 851/629/657.

Pre-declared expectation (falsifiable): prototype/mode-local Δ will land **≤ 0 or within the perm-null**, and
bootstrap-5th-pct **< 0** — because nearest-prototype is a coarsening and mode-local strictly shrinks the
neighbour pool. Any positive would be K-selection noise on ~600 exemplars (D3).

---

## F. PRIOR — **LOW**, verdict **NO-GO**

**Prior: LOW** (author agreed; recon lowers it further conditioned on the W2-B kill and D1/D2/D3).

- **D1 (decision-side low-bandwidth signals conditionally redundant):** bites *maximally*. The prototype
  assignment / mode label is a **deterministic lossy function of the same pooled vector** the flat kNN already
  votes over → **zero** new bits, not merely "low bandwidth." Strictly weaker than every dead decision-side
  signal (P1/P2/TARC at least injected an *external* MLLM read); W2-E injects nothing.
- **D2 (only representation-level levers ever cleared +3):** W2-E does **not** change the representation, so it
  never enters the one winning class. Its "D2-ish" self-label is not supported by the code — the key is
  identical to the flat baseline's.
- **D3 (±1–2pt noise floor):** K∈{2..8} prototypes over ~600 exemplars is clustering-init- and seed-noise
  dominated; any observed Δ sits inside the floor and inside the perm-null (as W2-B's +0.0016 MHC did).
- **W2-B conditioning:** W2-B is the freshest, cleanest negative in the exact meta-family W2-E belongs to
  ("zero-training unsupervised reorganisation of frozen features as an accuracy lever"). W2-B had **more**
  going for it — genuine multi-view structure *and* a surviving oracle (+0.07–0.10 headroom) — and still
  realised **none** of it at the decision level. W2-E has **less** structure (one view) and **no** admissible
  surviving-oracle path (§C). So W2-B's result pushes W2-E's performance prior toward zero, not merely down.
- **Novelty gate is dispositive.** Even setting performance aside, W2-E **fails D7-tightened** standalone and
  composite (§B) — a textbook k-means trick with an inert Qwen label, the same gate that let recon kill C3geo
  *before* any ceremony. Recommending a probe would spend prereg-design + independent-review + hash-freeze
  ceremony on a mechanism that **cannot yield a novel contribution regardless of the number that comes back**.
  That is precisely the ceremony-cost forensic recon exists to prevent.

**Recommendation: NO-GO — close W2-E at recon.** Do not open a probe-design cycle.

**Optional fallback (user-facing, not recommended by recon):** if the user later wants a *family-closing
negative datapoint* for the writeup (e.g., "memory-organisation / prototype compression also does not convert
on hate-video kNN"), the cache-sufficiency (§D) + verbatim W2-B machinery reuse make such a probe **near-free
(~$0 CPU-minutes)**. But its *maximum* upside is a negative prior-update footnote — it can never be a
contribution (D7-fail), so it should be run, if at all, only as batch-companion color, never on the critical
path, and never with GPU.

---

## Provenance
- Code: `src/utils/metrics.py:262-320`, `src/model/evaluate_rac.py:80-155`, `scripts/analysis/w2b_probe.py`
  (lines 1-380 read; loader/vote/oracle/Fano/perm-null/bootstrap machinery).
- Caches: direct `torch.load` shape verification (§D table); `sklearn/scipy/faiss/numpy` import check in
  `HateVideo`; `data/audio/MHC` confirmed empty.
- Priors/bans: `autoresearch/goal_mllm_plus3/state/directions_tried.json` (24 dead ids incl. R3-C3geo +
  W2-B; 9 banned_constraints), `refine-logs/W2B_VERDICT_REVIEW.md` (outcome (d), ban scope), candidate text
  `research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md` §W2-E.
- Repo HEAD at recon: `0f43bdd`. Zero GPU / SLURM / Modal used.
