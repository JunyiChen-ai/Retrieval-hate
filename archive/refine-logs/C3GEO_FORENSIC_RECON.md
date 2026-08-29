# C3-GEOMETRY FORENSIC RECON — Qwen-embedding hard-negative mining for the triplet head

**Agent:** round-3 C3-geometry forensic recon (ZERO GPU; recon + probe-design sketch only, NO prereg/ceremony).
**Date:** 2026-07-15.
**Candidate source:** `research-wiki/ROUND3_NOVELTY_CANDIDATES_2026-07-14.md` §C3 (lines 187-227), scout prior **MODEST-LOW**.
**Status of this doc:** PIPELINE PRE-WORK. The round-3 lead is S2S set-matching (`exp-s2s-r3.md`). If S2S's probe kills, C3-geometry is the next candidate; this recon means zero dead time.

**HEADLINE VERDICT (one line):** **KILL-BEFORE-CEREMONY as a standalone.** The pipeline *already* mines global hardest-opposite-label negatives in the head's own evolving space; C3-geo's only delta is swapping that mining space for the frozen Qwen pooled space — a textbook trick (dozens of 2024-25 papers) that (a) cannot clear the D7-tightened novelty bar and (b) has a near-zero performance prior because the frozen-Qwen source is either strictly weaker than the current mining space (task-aligned) or is the encoder-swap signal through the back door, already RESOLVED-NEGATIVE on MHC (B1/B2/B4) and redundant on HateMM.

---

## 1. CURRENT MINING BASELINE (load-bearing — the candidate's delta must be stated against THIS, not a strawman)

**One-liner:** The head is trained with a **triplet-margin + BCE** hybrid whose hard negatives are mined **dynamically, dataset-wide (global, NOT in-batch), as the nearest opposite-label train videos in the head's OWN evolving projected embedding space, re-indexed once per epoch** (or every step under `--reindex_every_step`). This is **online / curriculum-style hardest-opposite-label mining** — the strongest static-mining strawman is already beaten by what exists.

Evidence, quoted file:line:

- **Loss = triplet-margin + BCE hybrid.** `src/model/loss.py:453-455`:
  `total_loss = torch.mean(torch.relu(in_batch_loss + hard_loss - pseudo_gold_loss + args.triplet_margin))`.
  Margin = `--triplet_margin` default 0.1 (`src/run_rac.py:144-145`). (Memory `numeric-provenance-discipline` /
  method-chapter note: loss is triplet-margin+BCE, NOT InfoNCE — consistent with this code.)

- **Mining is FAISS over the head's evolving projected space, dataset-wide.** `src/utils/retrieval.py:347-353` rebuilds
  `train_feats` by running the **trainable head** over the whole train loader:
  `_, all_feats = model(image_feats, text_feats, return_embed=True)` — so the mining space is the head's *current*
  L2-normalized projection, not the raw frozen encoder features. `retrieval.py:422` `index.add(train_feats_normalized)`
  indexes **all** train rows; `retrieval.py:426-427` `index.search(query_feats_normalized, largest_retrieval*args.hard_negatives_multiple)`
  searches globally. **This is global, not in-batch, not semi-hard.**

- **Selection rule = hardest OPPOSITE-LABEL neighbor.** `retrieval.py:480`:
  `if train_labels[I[i,iter]].item() != query_labels[i].item() and j < args.no_hard_negatives:` → take it as a hard
  negative, walking the FAISS-ranked list from nearest outward. (Same loop also mines pseudo-gold positives =
  nearest *same*-label, `retrieval.py:497`.) So the current negatives are already the *most confusable cross-class*
  train videos — exactly the object C3-geo proposes to "newly" select.

- **Refresh cadence = per-epoch (curriculum).** `src/run_rac.py:581-582` sets `train_feats=None` at each epoch start
  (forces re-index against the *updated* head); reused across steps within the epoch (`run_rac.py:607-608`);
  `run_rac.py:586-589` re-indexes every step when `--reindex_every_step`. The mining space therefore **tracks the head as
  it trains** — this is the online-mining property that static foreign-space mining was invented to beat, not lack.

- **Config surface already exists:** `--no_hard_negatives` (default 1), `--hard_negatives_multiple` (search depth),
  `--hard_negatives_loss`, `--reindex_every_step` (`run_rac.py:202-240`). Hard-negative machinery is a first-class,
  tuned part of the pipeline, not an unexplored slot.

**Precise statement of the C3-geo delta against this reality:**

| axis | current baseline | C3-geo proposal | is this the delta? |
|---|---|---|---|
| **source space** | head's **evolving** projected embedding (task-adaptive, curriculum) | **frozen Qwen pooled** 3584-d (static, foreign to the head) | **YES — this is the ONLY real change** |
| scope | global, dataset-wide FAISS over all train | global, dataset-wide | NO (identical) |
| selection principle | hardest opposite-label (nearest cross-class) | hardest opposite-label (nearest cross-class) | NO (identical) |
| object | one pooled/projected vector per video | one pooled vector per video | NO (identical) |
| loss integration | into `hard_loss`, inside triplet relu | into `hard_loss`, inside triplet relu | NO (identical) |

So C3-geo is **not** "add hard-negative mining" (already present) and **not** "make mining global" (already global). It is **swap
the mining space from the head's own evolving space to a frozen foreign (Qwen) space.** That is the entire mechanism, and it is a
*downgrade in task-alignment* of the mining space, not obviously an upgrade. (The optional "relational-ordering distillation" add
is a second, separate lever — see §5.)

---

## 2. NON-ISOMORPHISM AUDIT (vs `state/directions_tried.json`, updated 2026-07-14)

- **(a) Banned "MLLM-scores-as-training-signal" — does frozen-embedding GEOMETRY escape it?**
  **CONCEDE the letter, flag the spirit.** The ban (`banned_constraints[]`) is literally "MLLM *scores/labels* as training signal."
  C3-geo uses frozen Qwen **pairwise cosine distances** (embedding structure), never a generated score/label/logit. The candidate
  doc §C3 non-isomorphism (lines 202-205) and the exhaustion audit §3(a) explicitly keep representation-geometry distillation as an
  open cell. So **on the letter, C3-geo is compliant — this is NOT the banned route.** BUT the spirit is uncomfortable: the frozen
  Qwen pooled embedding is *exactly the encoder-swap representation* (`positives_bank[encoder-swap]`). Mining a CLIP-head's negatives
  by Qwen distance imports the encoder-swap signal through the training-loss back door. On MHC-EN/ZH that signal is
  **RESOLVED-NEGATIVE** (dead routes `B1-qwen-encoder-zh` #20, `B2-32b-encoder` #21, `B4-lora-en` #22, and `C2-SAV` #18 which
  falsified the dilution hypothesis: MHC-EN is data/label-limited, not dilution-limited). On HateMM the Qwen encoder already wins
  by +5.3 — so you would just swap the encoder, not mine from its frozen distances. **Compliant, but the source geometry is a known
  quantity whose convertible cross-class structure is already characterized as dead-on-MHC / redundant-on-HateMM.**

- **(b) vs P5 counterfactual twins (dead #P5, "gate fail + hurts").** Non-isomorphic on mechanism: P5 *synthesized* one extra
  per-anchor negative (a sanitized-text twin fused via `model(anchor_img, sanitized_text)`, added into `hard_loss` at
  `loss.py:443-448`); C3-geo *re-selects real existing* train videos by frozen-Qwen distance — no synthesis. **BUT P5's epitaph is a
  direct warning:** injecting extra/other-sourced hard negatives into this *already-hard-mining* triplet loss **hurt**. The loss is
  already saturated with the hardest cross-class exemplars; changing where they come from is the same class of intervention that P5
  showed the objective does not reward.

- **(c) vs B3/P9/encoder-swap (representation levers).** Non-isomorphic on injection point: those swap or *train* the encoder
  (change `image_feats`/`text_feats`); C3-geo keeps the encoder and changes only the negative-sampling distribution. Different slot.
  Fine.

- **(d) The D1 law — does it bite?** **Literal D1 does NOT bite; a generalized redundancy argument DOES.** D1 (`diagnosis_frame`) is
  about *low-bandwidth decision-side* signals being redundant given the frozen representation. C3-geo is training-time, adds no
  inference-side channel → literal D1 is inapplicable, and the candidate correctly claims the D2 (representation-shaping) class. **The
  honest problem is not D1 but redundancy *within* the mining itself:** the current loss already draws the hardest opposite-label
  negatives from the head's own (more task-aligned) space; substituting a frozen foreign space changes *which* hard negatives, not
  *whether* the objective sees hard structure. This is the exact "probe passes, training flat" signature that killed P3
  (`dead[P3]`: "probe pass, train flat, 3 datasets") — and mining tweaks are the textbook instance of that failure. **D1's generalized
  spirit ("is the extra structure redundant given what the head already extracts?") bites hard, even though its letter does not.**

**Non-isomorphism verdict:** C3-geo is **formally non-isomorphic** to every dead route (no synthesis, no score, different injection
point from encoder-swap) and **not on the banned list**. It is legal to run. But it is *mechanistically adjacent* to two dead
findings — its source geometry is the encoder-swap signal (dead on MHC, redundant on HateMM), and its intervention class (re-source
hard negatives in a saturated hard-mining loss) is the P3/P5 "objective doesn't reward it" pattern.

---

## 3. NOVELTY VERDICT (against the D7-tightened bar)

**The bar (binding).** D7 = **RESOLVED-NEGATIVE** (`DECISION_MEMO_pending.md:74-85`, `state/task_spec.md:28-30`,
`findings.jsonl F24`, user 2026-07-14 verbatim: "encoder swap 肯定不算 novelty… 一直做直到做出来为止"). Encoder-class levers *and*
"推而广之的通用决策规则校准 (generic decision-rule calibration, B5-class)" do **not** satisfy the novelty clause. The goal now
requires a **NOVEL MECHANISM** (novel within hateful-video) × MLLM-integrated × ≥+3 acc. The ruling's operative principle is that
**generic, off-the-shelf techniques do not become novel by being pointed at this domain.**

**Lit check (WebSearch, 2026-07-15) — is embedding-space hard-negative mining a standard trick?** Emphatically yes; it is a
*saturated* research area with named SOTA recipes:
- **NV-Retriever** (arXiv 2407.15831, 2024) — positive-aware hard-negative mining from embedding models; explicitly the crucial lever
  for SOTA text embeddings.
- **CoRNStack** (arXiv 2412.01007) and **Hard Negative Mining for Domain-Specific Retrieval** (arXiv 2505.18366, 2025) — embedding-
  distance hard-negative mining with denoising thresholds.
- **Dynamic threshold hard-negative mining** and **positive-aware mining** (NV-Retriever) — adaptive "hardness" per dataset.
- **MoCHi / SynCo** — synthesize hard negatives directly in representation space.
- **Graph CL via subspace-preserving hard-neg mining** (WWW 2024, dl.acm 10.1145/3589334.3645327);
  **Uncertainty-aware CL with hard-negative sampling** (AAAI 2025).
Sources: [NV-Retriever](https://arxiv.org/pdf/2407.15831), [CoRNStack](https://arxiv.org/pdf/2412.01007),
[Domain-Specific HNM](https://arxiv.org/pdf/2505.18366), [WWW'24 subspace HNM](https://dl.acm.org/doi/10.1145/3589334.3645327),
[AAAI'25 uncertainty-aware HNM](https://csse.szu.edu.cn/attachment/cglr/1749432531_AAAI-2025%E8%AE%BA%E6%96%87-%E5%88%98%E6%B6%B5.pdf).

**Verdict: CANNOT clear the D7-tightened novelty bar as a standalone → KILL-BEFORE-CEREMONY.** "Hard-negative mining from a foundation-
model embedding space" is precisely the class of generic, textbook trick that D7 says does not count merely for being first-in-
hateful-video. The candidate doc itself concedes C3 is "the weakest-novelty of the three… a training-recipe change" (§C3 line 209).
Under D7 that concession is fatal: a training-recipe change whose recipe is standard everywhere else is exactly what the ruling
excludes. It is *strictly weaker* on novelty than encoder-swap (which D7 already ruled out), because encoder-swap at least changes the
representation; C3-geo only re-samples negatives within an existing, tuned mining loop.

**What (if anything) would make it novel:** only **composition with the S2S set-structure** — i.e. mine hard *set*-negatives via
set-matching geometry (MaxSim/OT over frozen Qwen *per-frame* tokens), so the novel object (frame-set) and the novel operator (set
alignment) carry the contribution and the mining is a downstream detail. But that is *S2S's* novelty, inherited, not C3-geo's; it
should be pursued inside S2S (or its C1/C2 successors), never as an independent route. As a pooled-vector standalone, C3-geo has no
path to the novelty clause.

---

## 4. G0-COND PROBE SKETCH (if ever run despite §3 — pre-declared kill logic; ~$0)

**Data reality (better than the candidate doc assumed).** The frozen Qwen-7B **pooled** caches already exist locally — e.g.
`data/CLIP_Embedding/{HateMM,MHC,MHC_zh}/train_Qwen2.5-VL-7B-Instruct_HF.pt` (+ dev/test; 32B and LoRA variants too). So the pooled-
distance geometry needs **NO new GPU forward** — it is a CPU numpy/torch load. (Only the S2S/C1 *per-frame* variant needs a new
forward; C3-geo's pooled mining does not.) Under the CLAUDE.md 云端探针 policy this runs on Modal features-only, but honestly it is a
few CPU minutes; cost ≈ **$0**.

**Probe stages (paired, both protocols, oracle kill-switch first):**
1. **Overlap screen (dead-on-arrival gate, pure CPU, zero training).** For each anchor, compute the current mining's picks (hardest
   opposite-label in the *frozen input* space as a cheap proxy for the head space at init) vs C3-geo's picks (hardest opposite-label
   in frozen **Qwen** space). Report set-overlap of the `no_hard_negatives × hard_negatives_multiple` candidate pool.
   **Kill if overlap > ~90% on all datasets** → C3-geo selects the same negatives the pipeline already uses ⇒ dead, no training.
2. **Oracle ceiling arm (mandated, per C3-probe erratum / candidate doc line 129).** Label-aware hardest-negative ceiling: does *any*
   choice of Qwen-mined negatives, taken at its oracle-best, project to ≥+3 acc over the current mining under the paired protocol?
   Include a **label-oracle calibration arm that must reach ~100% Fano headroom**, else the probe instrument is void.
   **Kill if oracle Δ < +0.03 on every dataset** → mined-hardness source is not the binding constraint ⇒ dead, no formal ceremony.
3. **Triage A/B (Modal T4, ~1 GPU-h only if 1-2 clear).** Retrain the tiny head with Qwen-mined vs current negatives; paired Δ,
   oracle + dev-honest acc/F1, both protocols, on the surviving dataset only.
   **Kill if paired Δ < +0.03 OR no-selection protocol shows overfit (train↑ test↓)** — D3 is the live risk on 549-744 train samples.
4. Only a clean pass at stage 3 on ≥1 dataset would justify the formal 3-seed both-protocol ceremony.

**Pre-declared expectation:** stage 1 or 2 kills it. The current mining already takes the *hardest* cross-class negatives from a
*more* task-aligned space; the frozen-Qwen picks will either overlap heavily (stage-1 kill) or, where they differ, differ *because the
Qwen space is not the head's space* — which is a reason to expect no gain, not a gain.

---

## 5. HONEST PRIOR

Engaging D3 (±1-2pt noise on 549-744-sample train sets), the scout's **MODEST-LOW**, and the fact that the current loss is *already*
a per-epoch-refreshed global hardest-opposite-label miner in the task-aligned head space:

**Prior: LOW (below the scout's MODEST-LOW).** The scout priced C3-geo before this recon established that the pipeline is not doing
naive/random/in-batch mining but **online global hardest mining in the evolving head space** — which removes the usual upside of
"add real hard-negative mining." The residual lever (swap the mining space to frozen Qwen) is a *downgrade in task-alignment* dressed
as an upgrade, and its source geometry is the encoder-swap signal that is dead on MHC and redundant on HateMM.

**One falsifiable sentence:** *If Qwen-frozen-geometry-mined hard negatives do not beat the pipeline's existing evolving-head-space
hardest-opposite-label mining by a paired ≥+0.03 acc AND ≥+0.03 macro-F1 on ≥1 dataset under the no-selection protocol (with the
overlap screen showing the two miners actually pick different negatives), then the mining source is redundant with what the triplet+BCE
head already extracts — the failure signature that closed P3.*

**The optional relational-ordering distillation add** (candidate §C3 line 195, "preserve Qwen's pairwise ordering of neighbors") is a
*separate* lever that is closer to C5 (7B relational CRD, priced **LOW** in the candidate doc §C5) than to mining, and inherits C5's
objection: a 7B-teacher relational signal cannot exceed using the 7B encoder directly, which already fails to convert on EN/ZH. It does
not rescue C3-geo's prior.

---

## RECOMMENDATION TO LEAD

1. **KILL-BEFORE-CEREMONY as a standalone route.** It cannot clear the D7 novelty bar (textbook trick, weakest-novelty-of-three by the
   scout's own admission) and has a LOW performance prior (mining is already hardest-global-online; the frozen-Qwen source is dead-on-
   MHC / redundant-on-HateMM). No prereg, no GPU.
2. **If S2S survives:** C3-geo's *only* legitimate future is **absorbed into S2S/C1** as "hard set-negative mining via set-matching
   geometry" — the novelty rides on the set object/operator, not on the mining. Do not spin it out separately.
3. **If S2S dies and the lead needs the next candidate:** prefer **C1 set-to-set** (the ranked LEAD, FAIR prior, D2's richest untried
   member) or **C2 asymmetric multi-view memory** over C3-geo. C3-geo should be the *last* C-line cell touched, and only if a
   representation-geometry lever is wanted at the training-objective slot specifically.
4. **Cheapest honest closure if the lead wants C3-geo formally retired:** run only **probe stage 1 (overlap screen, pure CPU, ~$0, no
   GPU)** — a >90% overlap number is a clean, one-measurement epitaph ("Qwen-geometry mining picks the negatives the pipeline already
   mines") that costs nothing and pre-empts any re-proposal.

---

## PROVENANCE

- Current mining loop: `src/utils/retrieval.py:314-584` (`dense_retrieve_hard_negatives_pseudo_positive`), esp. :347-353 (head-space
  index rebuild), :422-427 (global FAISS search), :480/:497 (opposite-label / same-label selection).
- Triplet+BCE hybrid + margin: `src/model/loss.py:453-455`; `src/run_rac.py:142-145`.
- Per-epoch / per-step re-index cadence: `src/run_rac.py:581-589`, :607-608.
- Hard-negative config surface: `src/run_rac.py:202-240`.
- Candidate spec + priors: `research-wiki/ROUND3_NOVELTY_CANDIDATES_2026-07-14.md` §C3 (187-227), §C5 (251-261), ranking table (267-273).
- Dead routes + bans + D-laws: `autoresearch/goal_mllm_plus3/state/directions_tried.json` (P3, P5, C2-SAV #18, B1 #20, B2 #21,
  B4 #22; `banned_constraints[]`; `diagnosis_frame`; `positives_bank[encoder-swap]`).
- D7 ruling: `research-wiki/DECISION_MEMO_pending.md:74-85`; `autoresearch/goal_mllm_plus3/state/task_spec.md:28-30`;
  `state/findings.jsonl F24`; `logs/orchestrator.jsonl` d7_ruling_round3_retarget.
- Local pooled Qwen caches (no new GPU for C3-geo pooled probe): `data/CLIP_Embedding/{HateMM,MHC,MHC_zh}/train_Qwen2.5-VL-7B-Instruct_HF.pt` (+ dev/test/32B/LoRA).
- Novelty lit (embedding-space HNM is standard): NV-Retriever 2407.15831, CoRNStack 2412.01007, Domain HNM 2505.18366,
  WWW'24 10.1145/3589334.3645327, AAAI'25 uncertainty-aware HNM (verified via WebSearch 2026-07-15).
