# C2-MEMORY FORENSIC RECON — asymmetric / multi-view MLLM memory bank

**Agent:** round-3 C2 candidate forensic recon (ZERO GPU; recon + probe-design sketch only, NO prereg/ceremony).
**Date:** 2026-07-15.
**Candidate source:** `research-wiki/ROUND3_NOVELTY_CANDIDATES_2026-07-14.md` §C2 (lines 141-183), scout prior **MODEST**.
**Sibling recon mirrored:** `refine-logs/C3GEO_FORENSIC_RECON.md` (killed its candidate pre-ceremony against the D7-tightened novelty bar; same severity applied here).
**Status of this doc:** PIPELINE PRE-WORK. Round-3 lead is S2S set-matching (`research-wiki/experiments/exp-s2s-r3.md`). C2 shares S2S's per-frame extraction; this recon means zero dead time whichever way S2S's probe goes.

**HEADLINE VERDICT (one line):** **FOLD-INTO-S2S, do NOT spin out as a separate route or ceremony.** C2's parameter-free core (fixed multi-view memory × pooled/max-query aggregation) is *literally the off-diagonal ablation cell* of S2S's own MeanMaxSim operator (query∈{pooled,set} × memory∈{pooled,set}; S2S declares the two diagonal cells, C2 = pooled-query × set-memory); its only genuine extra lever (a *learned* query attention-pool / learned view-prototypes) is an S2S §11 downstream variant with strictly worse D3 overfit exposure; and — decisively — C2's learned-optimal best case is **already upper-bounded by S2S's declared per-query oracle-ceiling kill-switch** (`exp-s2s-r3.md` §6.4, which *by name* already calls "S2S + its C2 multi-view cousin" DEAD together if the oracle Δ < +0.04). C2 therefore **cannot outlive S2S** and adds no route S2S's screen does not already decide. Standalone novelty additionally **fails the D7 bar** (asymmetric multi-vector memory = MUVERA / ColBERT / asymmetric-dual-encoder, a saturated general-retrieval trick); the composite novelty is **S2S's, not C2's** (C2 varies the symmetry of S2S's novel operator, it does not add a novel component).

---

## 1. MEMORY REALITY (load-bearing — C2's delta must be stated against THIS, not a strawman)

**One-liner:** The retrieval memory is built by running the **trainable align head** over each train video to produce **exactly one L2-normalized projected vector per video**, indexed in a flat inner-product FAISS index; the top-k neighbours vote by a **rank-weighted signed-cosine sum**. BOTH the query side and the memory side are **single pooled vectors**; the match is **symmetric cosine**. There is no per-video multiplicity anywhere in the memory.

Evidence, quoted file:line (eval-time / main-table path, `src/model/evaluate_rac.py` `retrieve_evaluate_RAC`):

- **One vector per memory video.** `evaluate_rac.py:96` `_, all_feats = model(image_feats, text_feats, return_embed=True)` produces one projected embedding per batch row; `train_feats` is accumulated **one row per train video** (`:100-115`). The query side (`evaluate_feats`) is built identically, one row per eval video.
- **Symmetric single-vector kNN.** `evaluate_rac.py:139` `index = faiss.IndexFlatIP(dim)`; `:154` `index.add(train_feats)`; `:155` `D, I = index.search(evaluate_feats, largest_retrieval)` with `largest_retrieval = args.topk`. One pooled query vector searched against one-pooled-vector-per-video memory; cosine/IP.
- **Vote = rank-weighted signed-cosine sum over top-k.** `src/utils/metrics.py:262-320` (`use_sim=True`, `majority_voting='arithmetic'`): neighbour labels mapped `{0,1}→{-1,+1}` (`:268`), multiplied by the neighbour similarity (`:270`), weighted by rank position `topk..1` (`weight[:length]`, `:284`), summed and `sigmoid(·)≥0.5` (`metrics.py:300`). Macro-F1 (the goal metric) at `:309`.
- **`topk`:** default `5` (`src/run_rac.py:121`), deployed at **20** in the round-3 configs (as `exp-s2s-r3.md` §5 states); the vote weighting is `topk..1`. Not load-bearing for the verdict — the object (one vector/video) is.

**Precise statement of the C2 delta against this reality:**

| axis | current memory reality | C2 proposal | is this the delta? |
|---|---|---|---|
| **memory object** | **one** pooled/projected vector per video | **multi-view SET** per video (per-frame/segment token vectors, or a small learned set of "view" prototypes) | **YES — the memory-side multiplicity is the change** |
| query object | one pooled/projected vector | pooled (or a **learned** query-side attention-pool over the query's own frames) | PARTIAL (asymmetry / learned query aggregation) |
| match geometry | symmetric cosine, single×single | **asymmetric**: pooled/learned query × set memory, vote aggregates over matched views | YES (asymmetry) |
| vote machine | rank-weighted signed-cosine over top-20 | **unchanged** (same `metrics.py` vote; only the pairwise score feeding it changes) | NO (identical) |
| training coupling | head trained by triplet+BCE (`src/model/loss.py:453-455`) | same head, **plus** a learned query-pool / learned prototypes (extra params) | the learned-pool variant only |

So C2 = **make the memory side multi-view (a set per video) + make retrieval asymmetric (pooled/learned query × set memory)**. The vote machine is byte-identical to the baseline; only the pairwise score and the memory multiplicity change. (`archive-auto-repair` — dead — only *deleted* pooled memory rows; `P2/P2b` — dead — *reranked* pooled neighbours with an MLLM. Neither changed the memory *object*; C2 is the first to do so — the candidate's one true claim, `ROUND3…§C2:155-160`, `directions_tried.json` dead[P2/P2b], dead[archive-auto-repair].)

---

## 2. RELATION TO S2S (critical — the load-bearing finding)

**Read S2S's arms first (`exp-s2s-r3.md` §5, lines 198-204).** S2S retrieves over **per-frame Qwen token sets** `{g_1..g_T}` and declares:

- **POOLED (baseline):** `cos(mean_t g^Q_t , mean_t g^M_t)` — pool BOTH sides, then cosine. (query = pooled, memory = pooled)
- **SET (primary):** MeanMaxSim `= (1/|Q|) Σ_{q∈Q} max_{m∈M} cos(ĝ^Q_q, ĝ^M_m)` — set on BOTH sides. (query = set, memory = set)
- **SET-Chamfer (sensitivity):** symmetric `0.5·[MeanMaxSim(Q→M) + MeanMaxSim(M→Q)]`.

**C2-as-specified is the off-diagonal cell of S2S's own 2×2 grid.** Lay S2S's operator on the grid {query ∈ pooled|set} × {memory ∈ pooled|set}:

| | memory = pooled | memory = **set** |
|---|---|---|
| **query = pooled** | S2S **POOLED** (declared baseline) | **← C2 (parameter-free core)** |
| **query = set** | (mirror; unused) | S2S **SET** (declared primary) |

C2's parameter-free mechanism — "fixed multi-view memory + query-max aggregation," which the C2 doc **itself** names as its cheapest first probe (`ROUND3…§C2:174-177`) — is exactly `s(Q,M) = max_{m∈M} cos(ĝ^Q_pooled, ĝ^M_m)`: the `|Q|=1` (pooled-query) reduction of MeanMaxSim against a full memory set. That is **not a new operator; it is the `|Q|=1` ablation of S2S's operator.** It sits inside S2S's natural ablation grid and can be computed on S2S's *identical* frozen frame vectors at zero marginal cost.

**The one genuine C2-specific lever — and why it too is bounded by S2S.** C2's only mechanism outside S2S's declared cells is the **learned** asymmetric query aggregation (a trained attention-pool that *selects which memory views to match*) and/or **learned view-prototypes** on the memory side. But S2S already declares an **oracle-ceiling arm** (`exp-s2s-r3.md` §6.4, lines 265-288) that, for each query, selects the query's single most-discriminative frame `t*(Q)` and scores `s = max_{m∈M} cos(ĝ^Q_{t*}, ĝ^M_m)` against the **full** memory set. That is precisely **the upper bound of C2's learned-query-selection asymmetric mechanism** — an oracle query selector dominates any *learned* query selector, and the memory side is the full set (which dominates any learned prototype compression, since prototypes are a function of the frames and cannot carry more alignment than the frames themselves). Therefore:

> **S2S's per-query oracle-ceiling kill-switch already upper-bounds C2's best case.** If S2S's oracle Δacc < +0.04 on every dataset, C2's learned asymmetric best case is *also* < +0.04. C2 dies with S2S, at zero additional measurement.

This is not my inference alone — **the S2S authors already institutionalized it.** `exp-s2s-r3.md` §6.4 KILL-SWITCH (verbatim): *"the whole 'don't-pool' family (S2S + its C2 multi-view cousin) is DEAD, zero head GPU."* And §11: *"Composability with … C2 (multi-view memory, which reuses this exact extraction) is noted."* And §13 K4 lists the oracle Δ<+0.04 as the family killer. S2S already treats C2 as (a) sharing its extraction, (b) bound by its oracle kill-switch, (c) a downstream-composable variant — not a separate route.

**C2-vs-S2S distinctness verdict: FOLD-AS-ABLATION (not a separate route).**
- Parameter-free C2 ⊂ S2S's 2×2 ablation grid (the pooled-query × set-memory off-diagonal cell) → add it as a **pre-declared asymmetric ablation arm** in `exp-s2s-r3.md` §5, credited only if it beats the symmetric SET arm.
- Learned-pool C2 → an **asymmetric variant of S2S's §11 downstream head-training stage**, with strictly worse D3 exposure (extra params on 549-744 samples; the candidate concedes this, `ROUND3…§C2:170`), pursued only if the ablation cell clears.
- There is **no scenario** where C2 is a live standalone route that S2S's screen does not already decide: S2S-survives → C2 is an S2S ablation axis; S2S-dies (symmetric SET fails or oracle kill-switch fires) → C2's best case is below the same +0.04 bound and dies with it.

*(Contrast with C3-geo: C3-geo's composite path at least added a distinct training-time operation to S2S; C2's "composite" adds nothing S2S does not already own — it only varies the symmetry of S2S's operator. C2 is therefore **more** subsumed by S2S than C3-geo was.)*

---

## 3. NON-ISOMORPHISM (vs `state/directions_tried.json`, 20-dead + banned)

- **vs the memory-adjacent dead routes.** `archive-auto-repair` (dead, "AND-rule C−A=0, guard-rail only") only *deleted* pooled memory rows; `P2/P2b` (dead, "comparability ⊥ vote-correctness") *reranked* pooled neighbours with an MLLM. Neither altered the memory *object*. C2 is the first to make the memory multi-view — **formally non-isomorphic** on injection point. ✓ (the candidate's accurate claim).
- **vs encoder-swap (positive, HateMM-only).** Encoder-swap changed *which encoder* produces the pooled vector, kept the memory pooled. C2 keeps the encoder, changes memory structure — different slot. BUT the per-frame material is the **same frozen Qwen-7B representation** whose pooled form already **failed to convert MHC-EN** (`dead[B1-qwen-encoder-zh]` #20; `dead[C2-SAV]` #18 "dilution hypothesis FALSIFIED; MHC-EN data/label-limited"). C2's only escape is the alignment-not-bits argument — **which is entirely inherited from S2S** (`exp-s2s-r3.md` §3); C2 adds nothing to it.
- **vs banned_constraints.** Single-dataset own-train-split memory ✓; no OCR ✓; no gold-in-method ✓ (frames are unlabeled tokens); no cross-seed ensemble ✓ (within-model kNN); no external API ✓; no MLLM-score-as-training-signal ✓; **no kNN-vote-pool expansion ✓** (same videos, richer keys *per video* — NOT adding pseudo-labeled videos to the pool; the ban is on pool expansion via pseudo-labels, `banned_constraints[3]`); not a P1-P5 re-proposal ✓. **Legal to run** — same status as C3-geo, and (as with C3-geo) legality is not the binding constraint.
- **D-laws.** **D1** (low-bandwidth decision-side redundant) does **not** bite — C2 is memory/representation-level, no decision-side scalar. **D2** (only representation-level ever cleared +3): C2 claims this class correctly, but the one D2 positive (encoder-swap) did not convert MHC-EN, and C2's raw material is that same representation. **D3** (±1-2pt noise on 549-744 samples): the learned query-pool **adds parameters → strictly worse D3 exposure than S2S's parameter-free set distance** (candidate concedes, §C2:170). C2 is the *worse-D3* sibling of S2S.

**Non-isomorphism verdict:** formally clean against all 20 dead + bans, but **mechanistically dominated by S2S** and sharing S2S's single load-bearing risk (same frozen representation that is dead-on-MHC / HateMM-redundant-via-swap).

---

## 4. NOVELTY (against the D7-tightened bar) — standalone AND composite

**The bar (binding, D7 = RESOLVED-NEGATIVE).** `DECISION_MEMO_pending.md` D7 (user, 2026-07-14 verbatim): *"encoder swap 肯定不算 novelty … 做不出来就一直做,直到做出来为止"*; orchestration reading: encoder-class levers **and 推而广之的通用决策规则校准 (generic decision-rule / off-the-shelf-technique calibration)** do **not** satisfy the novelty clause; goal now requires a **NOVEL MECHANISM** (novel within hateful-video) × MLLM-integrated × ≥+3 acc. Operative principle (from C3-geo): **a generic off-the-shelf technique does not become novel by being pointed at this domain.**

**Standalone lit check (WebSearch, 2026-07-15) — is asymmetric multi-view memory a standard trick?** Emphatically yes; it is a *named, saturated* general-retrieval area:
- **ColBERT / ColBERTv2** (SIGIR 2020 / arXiv 2112.01488) — multi-vector late interaction, one embedding per token, MaxSim (Chamfer) scoring.
- **MUVERA** (NeurIPS 2024, arXiv 2405.19504) — *literally* "**asymmetrically** generates Fixed Dimensional Encodings of **queries and documents**"; the FDE asymmetry (query embeddings summed, document embeddings averaged) and the "large offline document encoder / pruned distilled query encoder" design are the standard asymmetric-multi-vector recipe. **This is C2's mechanism, named.**
- **Jina-ColBERT-v2** (arXiv 2408.16672), **asymmetric dual-encoder retrievers** — established.
- Multi-view / prototype **memory** for few-shot **video** classification is likewise established (SlowFast memory networks ACM-MM 2020; multi-grained temporal prototypes; memory-prototype learning) — WebSearch 2026-07-15.

**In-domain check.** RGCL (ACL 2024) uses "a simple **KNN classifier**" over a single RGCL-trained embedding space; RA-HMD (EMNLP 2025) and MoRE (WWW 2025) retrieve over pooled single vectors (WebSearch 2026-07-15; `ROUND3…§C2:162-164`, `C3GEO_FORENSIC_RECON` provenance). Multi-vector/late-interaction memory is **absent in hateful-video/meme** — the gap is real.

**Standalone novelty verdict: FAILS the D7 bar → KILL as a standalone.** "Apply ColBERT/MUVERA-style asymmetric multi-view memory to hateful-video kNN" is precisely the generic-off-the-shelf-technique-pointed-at-the-domain that D7 excludes. It is *weaker* on the D7 axis than encoder-swap in one respect (encoder-swap at least changed the representation; C2 re-organizes an established retrieval memory), on par in the other (both are domain transfers of an external standard technique). Same fate as C3-geo.

**Composite novelty verdict: the composite is S2S's, not C2's.** The lead flagged that (set-structured MLLM memory × retrieval-contrastive hateful-video pipeline) *together* might be the defensible story. It is — but **S2S already IS that composite** ("first set-to-set late-interaction retrieval over MLLM video-language tokens in a retrieval-contrastive hateful-video kNN head," `exp-s2s-r3.md` §9). C2 does not add a novel *component* to that composite; it varies a *design choice within S2S's novel component* (symmetric-vs-asymmetric matching; raw-frames-vs-learned-prototypes memory). Even the sharpest pro-C2 framing — "asymmetry is itself the contribution: query-side and memory-side discriminative frames differ" — is (a) exactly MUVERA/asymmetric-dual-encoder territory, and (b) directly testable as the off-diagonal cell of S2S's grid, i.e. a *finding S2S reports*, not a separate contribution. **The composite framing confirms FOLD, not spin-out.**

---

## 5. G0-COND PROBE SKETCH + cost (fold as pre-declared S2S ablation arms; ~$0 marginal)

**Data reality.** C2 shares S2S's Stage E per-frame extraction verbatim (`exp-s2s-r3.md` §6.1, all splits extracted). The frameset caches are **derived float vectors, features-only** → cloud-eligible under the CLAUDE.md 云端探针 policy; syncing them to Modal is the only marginal step, and the probe itself is CPU-minutes. **Marginal cost of the parameter-free C2 screen ≈ $0** (it is one extra score function over the same cached frame sets that S2S's Stage P already loads).

**Fold — add these to `exp-s2s-r3.md` §5 arms table (NOT a new prereg):**
1. **ASYM (pooled-query × set-memory):** `s(Q,M) = max_{m∈M} cos(ĝ^Q_pooled, ĝ^M_m)` — the parameter-free C2 core (the off-diagonal grid cell). Run through the identical Stage P LOO vote, paired, both protocols, on the SAME frozen frame vectors and same seeds as S2S's POOLED/SET arms.
2. *(optional, for a complete 2×2)* **ASYM-mirror (set-query × pooled-memory).**

**Pre-declared kill logic (binding sketch):**
- **Free kill via S2S's oracle (no extra measurement):** S2S's §6.4 oracle-ceiling arm upper-bounds C2's learned-query best case. **If S2S's oracle Δacc < +0.04 on every dataset (S2S kill-switch K4 fires), C2 is DEAD with S2S** — the "don't-pool" family (S2S + C2) is closed together, exactly as `exp-s2s-r3.md` §6.4 already declares. No head GPU, no learned pool.
- **Ablation-cell kill (if S2S symmetric SET survives):** compute the 2×2 grid on identical frames, paired. **If symmetric SET ≥ ASYM everywhere (all datasets, both acc AND macro-F1), the asymmetric memory-structure lever adds nothing over S2S's symmetric operator** ⇒ C2 dead as a route, zero learned-pool GPU. Only if ASYM **beats** symmetric SET by a paired margin projecting to +3 would the learned query-pool warrant ~1 GPU-h — and it then runs as the **asymmetric arm of S2S's §11 formal stage**, never a standalone ceremony.
- **Calibration guard (per the C3-probe erratum, inherited from S2S §6.3):** the Fano label-oracle machine-validity arm must reach ≥0.99, else the probe instrument is void and no negative verdict is accepted.

**GPU cost.** $0 marginal for the parameter-free ablation cell (rides S2S Stage E + Stage P). ~1 GPU-h **only if** the ablation cell clears AND the learned pool is pursued inside S2S §11.

---

## 6. HONEST PRIOR (one falsifiable sentence)

**Prior: LOW** (below the scout's MODEST). The scout priced C2 before this recon established that (i) its parameter-free core is an ablation cell of S2S's own operator, (ii) its learned best case is already upper-bounded by S2S's declared oracle kill-switch, and (iii) its only extra lever adds parameters → worse D3 on 549-744 samples. The memory *is* the last untouched pipeline component, but C2's way of touching it is dominated by S2S at every point.

**One falsifiable sentence:** *If, on S2S's frozen per-frame vectors, the asymmetric pooled-query × set-memory ablation cell does not beat S2S's symmetric SET arm by a paired ≥+0.03 acc AND ≥+0.03 macro-F1 on ≥1 dataset — while S2S's own per-query oracle-ceiling arm, which upper-bounds C2's learned-query best case, already governs the +0.04 kill-switch — then asymmetric multi-view memory adds nothing beyond S2S's symmetric operator and is not a route S2S's screen does not already decide.*

---

## RECOMMENDATION TO LEAD

1. **FOLD, do not spin out.** C2 is not a separate candidate; it is (a) the pooled-query × set-memory **ablation cell** of S2S's MeanMaxSim grid (parameter-free core, ~$0, add as a pre-declared arm to `exp-s2s-r3.md` §5) and (b) an **asymmetric variant of S2S's §11 downstream stage** (learned pool). No independent prereg, no independent ceremony.
2. **Correct the queue position.** "C2 next-in-queue if S2S dies" is **wrong**: C2 **cannot outlive S2S**. S2S's oracle-ceiling kill-switch (§6.4) already upper-bounds C2's best case and already names "S2S + its C2 multi-view cousin" as DEAD together. If S2S dies, the retrieval-object family (S2S + C2) closes in one measurement; there is no separate C2 to run, and the shared-extraction cost is moot.
3. **Novelty:** standalone **fails D7** (MUVERA/ColBERT asymmetric-multi-vector = saturated off-the-shelf trick); the composite novelty **belongs to S2S** (C2 varies the symmetry of S2S's operator, adds no novel component). Do not present C2 as an independent novelty claim.
4. **Cheapest honest handling:** when S2S's Stage P runs, add the one **ASYM off-diagonal cell** to its 2×2 grid (free) and let S2S's existing oracle kill-switch adjudicate the whole "don't-pool" family. That yields, at zero marginal GPU, a clean pre-registered answer to "does asymmetric multi-view memory beat symmetric set-matching?" — reportable as an S2S ablation whichever way it falls, and a one-line epitaph for C2-as-a-route if symmetric ≥ asymmetric.

---

## PROVENANCE

- Memory reality (one vector/video, symmetric IndexFlatIP kNN, rank-weighted signed-cosine vote): `src/model/evaluate_rac.py:96` (`model(...return_embed=True)`), `:100-115` (one row/video accumulation), `:139` (`IndexFlatIP`), `:154-155` (`index.add(train_feats)` / `index.search(evaluate_feats, args.topk)`), `:160-194` (top-k logging_dict); vote `src/utils/metrics.py:262-320` (`use_sim`, `arithmetic`, `:268` label map, `:270` sim-weight, `:284` rank-weight, `:300` sigmoid, `:309` macro-F1); `topk` default `src/run_rac.py:121`; hard-neg mining loop (train-time) `src/utils/retrieval.py:314-500`; loss `src/model/loss.py:453-455`.
- Candidate spec: `research-wiki/ROUND3_NOVELTY_CANDIDATES_2026-07-14.md` §C2 (141-183), §0 (15-54), ranking (267-273).
- S2S (distinctness centrepiece): `research-wiki/experiments/exp-s2s-r3.md` §5 arms (198-204), §6.4 oracle kill-switch (265-288, verbatim "S2S + its C2 multi-view cousin … DEAD"), §11 downstream + C2 composability (394-403), §13 K4 (425), §9 novelty scope (368-378).
- Dead routes + bans + D-laws: `autoresearch/goal_mllm_plus3/state/directions_tried.json` (dead[P2/P2b], dead[archive-auto-repair], dead[C2-SAV #18], dead[B1 #20], dead[R3-C3geo]; `banned_constraints[]`; `diagnosis_frame` = D1/D2; `positives_bank[encoder-swap]`).
- D7 ruling: `research-wiki/DECISION_MEMO_pending.md` §D7 (RESOLVED-NEGATIVE, user 2026-07-14 verbatim).
- Sibling severity: `refine-logs/C3GEO_FORENSIC_RECON.md`.
- Novelty lit (asymmetric multi-vector memory is standard; in-domain pooled-only): MUVERA arXiv 2405.19504 (NeurIPS 2024, asymmetric query/document FDE); ColBERT/ColBERTv2 (SIGIR 2020 / arXiv 2112.01488); Jina-ColBERT-v2 arXiv 2408.16672; few-shot-video memory-prototype (SlowFast memory nets ACM-MM 2020, multi-grained temporal prototypes); RGCL "simple KNN classifier" (ACL 2024, rgclmm.github.io), RA-HMD (EMNLP 2025, arXiv 2502.13061), MoRE (WWW 2025) all pooled single-vector — verified via WebSearch 2026-07-15.
