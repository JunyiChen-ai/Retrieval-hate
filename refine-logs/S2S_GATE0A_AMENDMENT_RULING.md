# S2S gate-0a amendment ruling — independent, binding

**Reviewer:** fresh zero-prior-context independent pre-registration **amendment** reviewer (read-only;
CPU-only source inspection; NO GPU / NO SLURM / NO submission; NO edit to any code/prereg file — this
ruling doc is the only artifact). **Date:** 2026-07-16. **Repo HEAD at ruling:** `a1b2439`.

**Under adjudication:** the S2S extraction smoke (job **13169**, extractor sha `07fd1621…`) FAILED at its
pre-registered gate **0a** (temporal positive control). The executor's forensic postmortem
(`refine-logs/S2S_GATE0A_POSTMORTEM.md`, commit `4358ca1`, doc-only) claims the control is invalid **by
construction** and asks for a disposition ruling (its options A/B/C). This file renders the binding
ruling.

**Read in full to rule:** the postmortem; the pre-registration (`research-wiki/experiments/exp-s2s-r3.md`
r3/r3a) + executable spec (`refine-logs/S2S_PROBE_DESIGN.md`); the binding prereg review
(`S2S_PREREG_REVIEW.md`, A1–A5) + code review (`S2S_CODE_REVIEW.md`, r1→r3a); the primary evidence
(`slurm/logs/s2s_extract_13169.log`); the control source
(`scripts/analysis/s2s_extract.py:115-286`); the installed **transformers 4.49.0** Qwen2.5-VL modeling
source (structural claim verified line-by-line, §A); and the two cross-line recons
(`W2C_FORENSIC_RECON.md`, `W2C_ORDER_PRECHECK_RECORD.md`, `W2A_FORENSIC_RECON.md`).

> **RULING (one line): (B) — REPLACE gate-0a with a causal-consistent positive control (the
> causal-prefix ONSET-INVARIANCE control, specified verbatim in §D), AND correct the §2/§4 premise
> wording to "cumulative causal group summaries." Assignment correctness stays gated by the new 0a′ +
> 0b grid + G-decomp + G-recon; the S2S PREMISE is adjudicated (unchanged) by the Stage-P oracle
> kill-switch. Re-smoke REQUIRED (gates 0b/1/2 were never reached); authorized submissions: ONE.**

I chose **B over A** deliberately: dropping gate-0a with no replacement (option A) would re-open exactly
the ungated-grouping gap that the prereg review made **BLOCKING amendment A1**. The correct response to
a *mis-written* correctness gate is to write it *correctly*, not to delete the protection — and a valid,
cheap, discriminative causal-consistent control **does** exist (§D). B is therefore strictly more
rigorous than A at trivial extra cost (one function rewrite + a diff-only code re-check).

---

## A. TECHNICAL VALIDATION — is the postmortem right? (each claim independently verified)

I re-derived every number from the raw log and re-read the installed transformers source myself.

### A.0 Evidence transcription — the postmortem matrix + argmax are FAITHFUL to the log
`slurm/logs/s2s_extract_13169.log:45-49` prints `match [1, 0, 3, 3] != expected sigma [2, 0, 3, 1]` and
the 4×4 matrix; the postmortem `§1` transcribes both **verbatim**. I recomputed `M.argmax(dim=0)`
(per-B-group-column best A-group row) from the printed matrix and got **[1, 0, 3, 3]** — matches the log
and the code (`s2s_extract.py:270-272`, `M = ga @ gb.t()` with `M[i,j]=cos(A_i,B_j)`, `match =
M.argmax(dim=0)`). ✔

> **One transcription correction (not the postmortem's — the orchestrator brief's).** The brief's
> parenthetical "(rows=B groups, cols=A groups)" is **inverted**: the code and the postmortem both have
> **rows = A-group `i`, cols = B-group `j`** (`M = ga @ gb.t()`, `M[i,j]=cos(A_i,B_j)`). The reported
> argmax `[1,0,3,3]` is self-consistent with the code's convention (verified), so nothing downstream is
> affected — but the axis label in the brief is backwards; the postmortem's is right.

### A.(i) "The match pattern refutes an orientation/axis bug" — CORRECT ✔
`σ=[2,0,3,1]` ⇒ `σ⁻¹=[1,3,0,2]` (I recomputed: `σ[0]=2,σ[1]=0,σ[2]=3,σ[3]=1` ⇒ inverse `[1,3,0,2]`).
A pure transpose / argmax-over-wrong-dim / B→A-vs-A→B bookkeeping bug reproduces `σ⁻¹` **exactly**. The
observed `match=[1,0,3,3]` (a) agrees with `σ⁻¹` only at index 0, and (b) **contains a repeat (`3`
twice), so it is not a permutation at all** — impossible under any clean relabelling. Orientation bug is
**refuted**, exactly as the postmortem (and the lead) predicted. ✔

### A.(ii) "Group means are cumulative-causal by construction" — CORRECT ✔ (I verified the source myself)
I read the installed `transformers==4.49.0`
`.../models/qwen2_5_vl/modeling_qwen2_5_vl.py` directly:

- **LLM decoder is causal, uniformly, with NO vision-token unmasking.** `Qwen2_5_VLAttention.is_causal =
  True` (**line 723**). The SDPA attention path passes the mask built by `_update_causal_mask` (**lines
  975-998**: `causal_mask = attention_mask[..., :key_len]`; `is_causal = True if causal_mask is None and
  q_len>1 else False`; `scaled_dot_product_attention(..., attn_mask=causal_mask, ...)`).
  `_update_causal_mask` (**lines 1244-1325**) → `_prepare_4d_causal_attention_mask_with_cache_position`
  (**lines 1327-1394**) builds a **standard lower-triangular** mask: **line 1371**
  `diagonal_attend_mask = torch.arange(target_length) > cache_position.reshape(-1,1)` and **line 1380**
  `causal_mask *= diagonal_attend_mask`. The mask is a pure function of sequence position; **nothing in
  it reads `input_ids`, `video_pad_id`, or vision-token spans** — there is **no bidirectional unmasking
  of vision tokens** anywhere in the LLM. The only other mask edit is the padding mask (lines 1382-1393);
  merged vision tokens are injected as ordinary sequence positions. So a vision token at sequence
  position `p` attends to `0..p` only.
- **Only the VISION ENCODER has block-diagonal (frame/window-local) attention.** `Qwen2_5_VLVisionAttention`
  (eager, **lines 265-269**) builds `attention_mask` full of `−inf` then sets `0` only within each
  `cu_seqlens[i-1]:cu_seqlens[i]` block; the SDPA vision path (**lines 314-316**) builds a bool mask
  `True` only within each `cu_seqlens` block. That attention is bidirectional **within** a block, never
  causal, never global — and it is **pre-LLM** (sees no text).

Therefore `g_t` = mean of `out.hidden_states[-1]` over temporal-group-`t` vision tokens
(`s2s_extract.py:130,184-190`) is the **mean of last-LLM-layer states conditioned on the causal prefix
through frames 0..t** — a cumulative-causal prefix summary **by construction**, not a frame-local
descriptor. The observed matrix corroborates this empirically: the block means are **monotone
both-late (0.900) > both-early (0.831) > mixed (0.727)** and the global max is `A3·B3 = 0.939` (last
group vs last group, *different* colours Y vs G). I recomputed all three block means and the global-max
cell — all match the postmortem. A scrambled/buggy grouping would give unstructured noise, **not** this
clean depth gradient; the gradient is positive evidence the grouping is *working* and the reps are
*cumulative*. ✔

### A.(iii) "The permutation-equivariance assumption is unsatisfiable for ANY stimulus under causal masking" — CORRECT for the load-bearing assumption; the *coded* argmax gate is slightly weaker (flag, not fatal)
The control's docstring load-bearing assumption (`s2s_extract.py:246-248`) is that *"permuting the input
pair order permutes `{g_t}` by the SAME permutation."* Under cumulation this is **structurally
impossible for any discriminative stimulus**: `g^B_0` summarizes `(text-prefix, slot-0 content)` with
**no history**, whereas after a permutation the content that landed in slot 0 of clip A appears at slot
`σ⁻¹` of B **with a different accumulated history**, so `g^B_0` cannot equal any `g^A_k` (different
context depths are different functions). The only stimulus for which permuting is a no-op is **all-frames-identical**,
which then fails the control's own within-clip distinctness sub-check (`:280-284`). So the equivariance
premise is genuinely unsatisfiable for any *distinct-content* stimulus — the postmortem is right, and
"patch the frames and resubmit will not work" is the correct operational call. ✔

**One honest over-statement to flag (does not change the disposition).** The postmortem's §4 phrasing
"*No choice of synthetic stimulus can make a permutation-equivariance / **content-argmax** control
pass*" conflates two things. The *equivariance* assumption is provably unsatisfiable (above). The
*coded gate* is the weaker `argmax_i cos(A_i,B_j)=σ[j]` (content recovery). That is not a pure logical
impossibility — a stimulus with content signal strong enough to dominate the context-depth gradient
could in principle recover the content argmax. It is defeated **empirically** here (content advantage is
only **+0.017**, §A(iv)), and it tests a premise (equivariance) known to be false, so a pass would be
for the wrong reason. Net: the gate is invalid because it encodes a false premise and is empirically
dominated by position — *not* because the argmax is a theorem-level impossibility for every conceivable
stimulus. This nuance is why I do not accept "unsatisfiable for ANY stimulus" as literally exact for the
coded gate, but it **supports the same ruling** (drop/replace, do not patch).

### A.(iv) The "same-colour advantage +0.017" arithmetic — CORRECT ✔
Same-colour cross pairs `(A_i,B_{j}: i=σ[j])`: `Red A0·B1=0.853, Green A1·B3=0.807, Blue A2·B0=0.674,
Yellow A3·B2=0.902` → **mean 0.809**. Other 12 entries → **mean 0.79208**. Advantage = **+0.01692 ≈
+0.017** (I recomputed from the matrix; exact). Content is almost absent from the similarity structure;
position/context-depth dominates. ✔

**Postmortem verdict on §A:** technically sound on (i), (ii), (iv), and on the operational conclusion of
(iii); the only blemish is the mild over-reach in the §4 "any stimulus / content-argmax" phrasing
(flagged above). The postmortem was appropriately disciplined — it did **not** patch-and-resubmit, it
escalated for independent review, and it correctly separated "control-flaw" from "premise-reframe."

---

## B. PREMISE ASSESSMENT — does "set element = cumulative causal prefix summary" invalidate S2S, or reinterpret it?

**Ruling: it REINTERPRETS the hypothesis; it does not invalidate it — but it legitimately LOWERS the
prior, and the §2/§4 wording must be corrected.**

- **(i) The cumulative reps already power the project's only strong positive.** The banked `img_feats`
  pooled vector is the L2-normed mean over the **same** cumulative last-layer states (prefix span,
  vision∪text; `generate_VideoMLLM_embedding_HF.py:303`), and that pooled form produced the
  encoder-swap **HateMM +5.3** — the single robust win. "Cumulative" is therefore **not disqualifying per
  se**; the open question is only whether the *set* geometry over these cumulative states carries more
  alignable label signal than their *pool*. That question is untouched by the gate failure.

- **(ii) What MaxSim over cumulative states actually measures.** Each `g_t` is a "story-so-far" state
  (frames 0..t). Under cumulation, early content is re-summarized inside every later group (all later
  vision tokens attend back to it) while the newest frame enters only the last group — so the pool is a
  blur of context-depth-graded states, and MeanMaxSim over `{g_t}` is a **soft alignment of narrative /
  context-depth trajectories** ("do two videos pass through similar accumulated visual states at some
  stage?"), **not** the clean "align the one shared hateful frame across two videos" story the prereg §2
  told. That is a real reinterpretation and the prereg must say so. It is still a coherent, testable
  hypothesis (trajectory alignment can carry label signal), just a blurrier and lower-prior one than the
  frame-local picture that motivated S2S (and that P6-locality suggested).

- **(iii) The pre-declared oracle kill-switch bounds the risk of proceeding.** §6.4's oracle-ceiling
  (per-query gold-guided frame selection; DEAD if oracle Δacc `< +0.04` on *every* dataset) is exactly
  the instrument that adjudicates whether cumulative-state set-matching can convert — at **zero head
  GPU** if it cannot. The premise nuance changes the *interpretation* of a set element, not the validity
  of the probe. I do **not** weaken this or any Stage-P bar.

- **Is the empirical content-vs-position finding a legitimate prior-lowering signal for Stage P?**
  **YES — legitimately prior-lowering, but NOT disqualifying and NOT a Stage-P result.** On the synthetic
  stimulus, content is barely encoded relative to position (+0.017), i.e. on *these* reps the discriminative
  frame-content signal is faint versus the generic context-depth axis. That is a real, direction-lowering
  datum for "set-alignment recovers label-discriminative content pooling discards." Its weight is bounded
  by two honest caveats: (a) solid colours are pathologically low-content for a vision encoder trained on
  natural images (so +0.017 is a floor, not an estimate of natural-frame content), and (b) it is n=1
  synthetic clip pair. So it nudges the prior down modestly; it must **not** be cited as evidence the
  probe will fail, and it is not a substitute for the Stage-P oracle/raw/null adjudication.

**Bottom line for §B:** premise-REFRAME (record it honestly, reword §2/§4), not premise-falsification.
The line proceeds to Stage P under the unchanged oracle kill-switch, carrying a modestly lowered,
explicitly-stated prior.

---

## C. RULING — (B) REPLACE gate-0a + premise reword; plus the moral-hazard analysis

**Chosen: (B).** Rejected alternatives, briefly:
- **(A) DROP + reword only** — rejected. Gate-0a exists because prereg-review **A1** (BLOCKING) found
  that 0b-grid + G-decomp + G-recon are **grouping-invariant** and 0a is "the **only** check that
  actually exercises the token→temporal-group assignment." Dropping it with no replacement re-opens the
  precise silent-drift gap A1 was raised to close (a transformers upgrade or an unexpected `video_grid_thw`
  shape could reorder tokens undetected). A is acceptable-but-weaker; B dominates it at trivial cost.
- **(C) representation-source fork** (pool the vision-encoder pre-LLM, frame-local features) — rejected as
  the S2S disposition. It **breaks banked-parity**: the frame-local ViT features have **no G-recon twin**
  in the banked `img_feats` cache (which is a *post-LLM* pool), so the entire extraction-correctness
  anchor (G-recon cos≥0.9999 vs banked) collapses, and the composability-with-encoder-swap story is gone.
  Per the postmortem's own §5(C), that is "a design fork, not a bugfix" — i.e. a **NEW line** requiring
  fresh forensic recon + its own pre-registration. If the lead wants it, it goes to the candidate queue
  as a distinct route; it is **not** an amendment to S2S-as-frozen and is out of scope here.
- **(D) KILL S2S** — rejected. The failure is orthogonal to the measured hypothesis (see moral-hazard
  below); the oracle kill-switch already bounds the downstream risk to zero head-GPU; killing now would
  discard a cheap, pre-registered, honestly-priced screen on the one structurally-untouched component
  (the retrieval object) on the basis of a *control-design* error, not a hypothesis refutation.

### Moral-hazard analysis (this is a post-failure amendment — why is it evidence-driven, not outcome-driven?)

The decisive question the brief poses: **did the gate fail for a reason orthogonal to the measured
hypothesis, or a reason that also undermines it?** My finding: **primarily orthogonal (a false premise
in the control), with a bounded, honestly-recorded prior-lowering side-effect on the hypothesis.**

Evidence the failure is orthogonal / not a masked bug:
1. It is **not an implementation bug**: the match is not `σ⁻¹` and not even a permutation (§A(i)); a
   real orientation/assignment bug has a signature this isn't.
2. The failure mode is a **structural property of the model I independently verified from source**
   (causal LLM ⇒ cumulative reps, §A(ii)) — it is *what the code correctly computes*, not a defect.
3. The property gate-0a legitimately protects (temporal-major assignment) is **not** what failed — the
   clean monotone depth gradient in the matrix shows the grouping IS assigning tokens to temporal slabs
   correctly; only the equivariance *expectation layered on top* failed.
4. The same cumulative reps power the pooled baseline (encoder-swap +5.3), so the surfaced fact is not a
   flaw in the extraction.

Why this is nonetheless **not** a whitewash (the honest counterweight): the finding is not *purely*
orthogonal — it genuinely lowers the S2S prior (the frame-local motivation is now known to be
over-optimistic; content is faintly encoded vs position). An outcome-driven amendment would (i) delete
the failed gate, (ii) keep the rosy "shared-segment" wording, and (iii) protect the line by softening the
Stage-P bars. This ruling does the opposite on all three: (i) it **replaces** the gate with a *stricter-in-spirit,
correct* control (§D) rather than deleting the protection; (ii) it **rewrites §2/§4 to the weaker, more
honest "cumulative causal group summaries"** and records the +0.017 content finding as a prior-lowering
signal; (iii) it **touches no Stage-P bar** — the oracle `+0.04`, raw `+0.05/+0.05`, rank-only
corroboration, permutation null, bootstrap, and dataset rule all stand verbatim. The amendment makes the
line's claim *weaker and better-gated*, not easier to pass. That is the evidence-driven signature.

---

## D. EXACT AMENDMENT TEXT (binding) + re-smoke + submission count

The amendment is applied by the implementer under a **fresh diff-only code review + re-hash** (house
practice; the reviewer already re-checks single hunks). This ruling **specifies** the text; it does not
edit the source files. Every quoted "ORIGINAL" is verbatim from the r3/r3a-frozen files; every "AMENDED"
is the binding replacement.

### D.1 New gate 0a′ — the causal-prefix ONSET-INVARIANCE positive control (REPLACES old gate 0a)

**Design (valid under cumulation AND discriminative).** Instead of the false equivariance/argmax test,
exercise a property **guaranteed true** by causal masking: *a causal prefix summary is invariant to any
change strictly after it.* Two 8-frame clips built from the same solid-colour palette:
- **Clip P** pairs = colours `[R,G,B,Y]` (order `[0,1,2,3]`).
- **Clip Q** pairs = colours `[R,G,Y,B]` (order `[0,1,3,2]`) — **identical to P for frames 0–3 (groups
  0,1); differs for frames 4–7 (groups 2,3).**

Encode both (`banked_vec=None`), take the raw group means `g^P,g^Q` (`[4,D]`), L2-normalize per group to
`ĝ`. Assertions (ALL HALT on violation):

1. **PREFIX-INVARIANCE (load-bearing).** For each shared group `k ∈ {0,1}`:
   `cos(ĝ^P_k, ĝ^Q_k) ≥ 0.999`. *(Under the causal mask, group-`k` tokens attend only to positions ≤
   their own; P and Q share every frame through group 1, so `g^P_0=g^Q_0` and `g^P_1=g^Q_1` up to bf16
   kernel ULP. A spatial-major, reversed, or scrambled temporal grouping breaks this — the changed later
   frames would leak into an "early" group and drop the cosine well below 0.999.)*
2. **ONSET-DIVERGENCE (non-degeneracy + later-content routing).** `max(cos(ĝ^P_2,ĝ^Q_2),
   cos(ĝ^P_3,ĝ^Q_3)) < min(cos(ĝ^P_0,ĝ^Q_0), cos(ĝ^P_1,ĝ^Q_1)) − 0.002` *(the groups that received
   the differing frames must be detectably less identical than the untouched prefix groups — confirms the
   later frames actually entered groups 2/3 and the stimulus is non-degenerate).*
3. **WITHIN-CLIP DISTINCTNESS (retained from old 0a).** For each clip, `max off-diagonal group cosine <
   0.999` (the four temporal groups are distinct, not collapsed).

**Why it is valid AND discriminative:** (a) prefix-invariance is *guaranteed* by causal masking, so a
correct extraction cannot fail it — **no false-KILL from the cumulative structure** (the exact defect
that sank old 0a); (b) it **FAILS** for spatial-major, reversed, or interleaved temporal grouping (all
break the invariance), so it discriminates the assignment errors old 0a was meant to catch; (c) it is 2
forwards — **same cost as old 0a**, smoke-time. Combined with 0b (equal contiguous blocks) + the
source-proof of temporal-major contiguity (`modeling_qwen2_5_vl.py:466-505,529-534,560-562`), the
token→temporal-group assignment is runtime-gated again.

**ORIGINAL control code to replace** — `scripts/analysis/s2s_extract.py:245-286` (`temporal_positive_control`),
whose load-bearing assertions are:
```
    order_a = [0, 1, 2, 3]
    sigma = [2, 0, 3, 1]  # clip B's group j carries clip A's colour sigma[j]
    ...
    M = ga @ gb.t()                        # M[i, j] = cos(A_i, B_j)
    match = M.argmax(dim=0).tolist()       # for each B-group j -> best A-group i
    if match != sigma:
        raise RuntimeError(... "temporal assignment FAILED: argmax match {} != expected sigma {}" ...)
```
**AMENDED:** replace the σ-permutation/argmax body with the three assertions above (clips P=`[R,G,B,Y]`,
Q=`[R,G,Y,B]`; prefix-invariance on groups {0,1}; onset-divergence on groups {2,3}; retained within-clip
distinctness). Rename the function/log tag to `causal_prefix_control` / `[gate 0a']` so logs are
unambiguous. The function keeps its call site (before any real video) and HALT-on-failure semantics.

### D.2 §2 premise reword (`exp-s2s-r3.md` §2)

**ORIGINAL (`exp-s2s-r3.md:72`):**
> Two hateful videos that share a hateful *segment* but differ globally now match on that segment,
> whereas pooled cosine averages the match away.

**AMENDED:**
> Each set element `g_t` is a **cumulative causal group summary** — the mean of the last-LLM-layer states
> over temporal group `t`, conditioned on frames `0..t` (the Qwen LLM is **causal** over the video-first
> token stream, so a group's tokens attend only to earlier positions; verified
> `modeling_qwen2_5_vl.py:723,:1244,:1371-1380`), **not** a frame-local segment descriptor. Two videos
> that pass through **similar cumulative visual states** at some temporal stage now match via MeanMaxSim
> on the aligned states, whereas pooled cosine collapses the whole trajectory to one vector before
> matching. Whether set-MaxSim over these cumulative states beats the pooled vector is the empirical
> question Stage P adjudicates under the oracle kill-switch (§6.4); a synthetic-stimulus probe found
> content weakly encoded relative to context-depth in these reps (a prior-lowering, not disqualifying,
> signal — `S2S_GATE0A_AMENDMENT_RULING.md` §B).

### D.3 §4 g_t definition + gate-0a bullet reword (`exp-s2s-r3.md` §4 and design §3/§4)

**ORIGINAL (`exp-s2s-r3.md:164-166`, the temporal-control bullet):**
> - **(r1: A1) Temporal positive control (mandatory, HALT).** On a synthetic 8-frame clip = 4 distinct
>   solid-colour pairs, verify each `g_t` is nearest its intended temporal slab and that permuting the
>   input frame-pair order permutes `{g_t}` identically — the only check that exercises the grouping.

**AMENDED:**
> - **(amend-0a′) Causal-prefix positive control (mandatory, HALT).** Two 8-frame clips sharing frames
>   0–3 (groups 0,1) and differing at frames 4–7 (groups 2,3): assert the **shared prefix groups are
>   invariant** (`cos(ĝ^P_k,ĝ^Q_k) ≥ 0.999`, `k∈{0,1}`) and the **changed groups diverge** (onset), plus
>   within-clip distinctness — the only check that exercises the token→temporal-group assignment, and one
>   that is **valid under causal cumulation** (a prefix summary is invariant to later frames; the old
>   permutation-equivariance/argmax control was invalid by construction — `S2S_GATE0A_AMENDMENT_RULING.md`).
>   Temporal-major contiguity remains proved by the modeling source
>   (`modeling_qwen2_5_vl.py:466-505,529-534,560-562`).

Also add, at the `g_t` definition (`exp-s2s-r3.md:144-145` / `S2S_PROBE_DESIGN.md:120`), the clause
"(these last-layer states are **cumulative-causal**: each conditions on frames 0..t)". Apply the
identical bullet/wording swap in `S2S_PROBE_DESIGN.md:152-159` (§3 control), `:200-211` (§4), the §7
anchor table (`:412`), and `exp-s2s-r3.md` §7 gate 0 (`:353-354`) + §13 K0 (`:438`) — replacing every
"permuting … permutes `{g_t}` identically" / "each `g_t` nearest its slab" phrasing with the
prefix-invariance/onset control. **No number, bar, or Stage-P arm changes.**

### D.4 Gate list AFTER the amendment (the exact amended gate set)

```
0a′  Causal-prefix ONSET-INVARIANCE control (NEW, replaces old 0a)  — HALT
       shared groups {0,1} invariant (cos≥0.999); changed groups {2,3} diverge; within-clip distinct
0b   Grid-consistency  n_vis == grid_t·(grid_h//2)·(grid_w//2) AND per-group == (grid_h//2)·(grid_w//2)  — HALT   [UNCHANGED]
1    G-decomp  L2norm((Σ n_t g_t + p_S)/end) == this-forward banked-formula pooled, max-abs ≤ 1e-5      — HALT   [UNCHANGED]
2    G-recon   fresh banked-formula vec vs banked img_feats[v],  cos ≥ 0.9999 AND max-abs ≤ 1e-3        — HALT   [UNCHANGED]
3    Fano      ±1 gold-label-key LOO vote acc ≥ 0.99 both datasets                                       — VOID  [UNCHANGED]
4    Oracle    per-query oracle-frame Δacc < +0.04 on EVERY dataset ⇒ DEAD (no head GPU)                 — KILL  [UNCHANGED]
5    Raw bar + rank-only corroboration + permutation null + bootstrap                                    [UNCHANGED]
6    Near-dup audit + near-dup-excluded sensitivity                                                      [UNCHANGED]
7    Dataset rule (a/b/c/d)                                                                              [UNCHANGED]
```
Only gate 0a is replaced (0a→0a′). Everything from 0b onward is byte-unchanged.

### D.5 Re-smoke requirement — YES, REQUIRED (one submission)

Gates **0b + 1 + 2 on ≥1 real video were NEVER exercised**: smoke 13159 crashed at the device bug
(pre-data), and smoke 13169 failed the scientific gate 0a (still pre-data, before any real video was
encoded). The extractor is also being **code-changed** (new 0a′). Therefore, before Stage-E full
extraction is authorized:

1. Fold the D.1–D.3 amendments into `exp-s2s-r3.md` + `S2S_PROBE_DESIGN.md` + rewrite
   `s2s_extract.py::temporal_positive_control` → `causal_prefix_control`; re-`sha256`; re-pin the §10
   hash table (new "amend-0a′" row); the sbatch + `s2s_probe.py` are UNCHANGED (restate their r3a hashes).
2. **Independent diff-only code re-check** of the single rewritten function (house practice — not a full
   re-review) confirming the three assertions match §D.1 and still HALT-propagate.
3. **ONE** `SMOKE=1 sbatch scripts/slurm/s2s_extract.sbatch` submission — single job, **no `--time`**,
   `PENDING (JobHeldUser)` = **WAIT** (never force-release), throwaway `--out_root`, **no artifact under
   the real `data/CLIP_Embedding/<ds>/frameset_qwen7b_8f/` path**. The smoke log MUST show, on ≥1 real
   video per dataset, **all four HARD gates green**: **0a′** (prefix-invariance PASS, onset diverges,
   groups distinct), **0b** grid, **1** G-decomp ≤ 1e-5, **2** G-recon `grecon_cos_min ≥ 0.9999 AND
   grecon_maxabs_max ≤ 1e-3` (read from the assembly lines; the stale "(G-recon skipped)" echo remains a
   cosmetic NOTE), plus the config echo + both script sha256 lines.

**Authorized submission count: ONE** (this re-smoke). Stage-E full extraction remains a **separate**
authorization granted only after the re-smoke shows all four green; Stage P stays gated behind the full
extraction. No other GPU/SLURM is authorized by this ruling.

---

## E. CROSS-LINE IMPLICATIONS (non-binding directional updates)

**W2-C (temporal-order / escalation kernel) — net LOWER (prior-lowering; cuts both ways, dominant side is
LESS promising).** The S2S finding is double-edged and I state both:
- *More-promising read:* order/context-depth is now known to be **strongly, literally encoded** in the
  Qwen cumulative reps (`g_0..g_3` is a monotone depth ramp; block means 0.900>0.831>0.727) — a
  well-defined order axis exists, and the first-difference `g_{t+1}−g_t` (W2-C's transition kernel)
  isolates the *incremental* content, partially un-doing the cumulation.
- *Less-promising read (dominant):* the strongly-encoded order axis is **generic context-depth** — a
  monotone "context accumulates" ramp that is ~identical across videos and **class-invariant**, not the
  discriminative benign→escalation *content* trajectory W2-C needs. Content itself is weakly encoded
  (+0.017 synthetic). So an order kernel most strongly reads a non-discriminative depth ramp, which is
  exactly what W2-C's mandatory **within-video order-shuffle null** absorbs — and this is precisely what
  the CLIP-K4 pre-check already showed (`W2C_ORDER_PRECHECK_RECORD.md`: ORDER-DTW's edge over order-blind
  MeanMaxSim = +0.006/+0.005 acc, **at or below the order-shuffle null**, bootstrap CI straddles 0). The
  S2S finding **reinforces** that read on the Qwen side and validates the order-shuffle null as the
  correct instrument. Net: **prior-lowered, not killed** — the binding test remains the Qwen **T=8**
  within-video order-shuffle null, and "order is literally encoded" must not be mis-sold as "discriminative
  order signal exists."

**W2-A (transcript-first grounded key) — read-across is CONFIRMATORY of its mechanism, neutral-to-slightly-positive
on architecture; its own low prior is unaffected.** W2-A's re-specified PRIMARY *deliberately* uses the
same causal cumulation: it places the transcript **before** the frames so the vision tokens causally
attend back to it, then pools the vision span. The S2S structural finding (`is_causal=True`, no
bidirectional vision unmasking) is the **same fact W2-A independently verified** and depends on — so the
two lines cross-validate the architectural premise, and S2S's "problem" (vision tokens see preceding
context) is literally W2-A's "feature" (grounding). Read-across: the grounded key is architecturally
**real**, not a no-op, in the transcript-first order — S2S corroborates that. Caveat carried over: S2S
shows visual *content* is weakly encoded vs position on degenerate stimuli, a mild caution that a
conditioned-vision pool's own visual signal may be faint — but that is a low-content-stimulus artifact,
not disqualifying. **W2-A's LOW–MODEST prior stands on its own grounds** (the interaction term is already
partly banked in `text_feats`; the concat-must-lose D1 arm is the verdict) — unchanged by this ruling.

---

## F. HYGIENE

- **No code/prereg changed since the failure.** The postmortem commit `4358ca1` is **doc-only**
  (`git show --name-only`: `refine-logs/S2S_GATE0A_POSTMORTEM.md` + `refine-logs/S2S_SMOKE_RECORD.md`
  only). `git status --porcelain` on `s2s_extract.py`, `exp-s2s-r3.md`, `S2S_PROBE_DESIGN.md`,
  `S2S_GATE0A_POSTMORTEM.md` = clean (no uncommitted edits). The on-disk extractor sha
  `07fd162196a7e61e…` matches the r3a freeze and the 13169 log header. This ruling edits **no** source
  file; it writes only this doc.
- **Scope guard.** This amendment is **S2S gate-0a ONLY** (0a→0a′) + the §2/§4 premise wording. It sets
  **no precedent** for amending any other gate, bar, or line, and does not license any change to the
  encoder, dataset, frame budgets, Stage-P arms/bars, or the oracle/raw/null/bootstrap thresholds — all
  of which stand verbatim. Any representation-source fork (option C) is a NEW line requiring fresh recon
  + prereg, not covered here.

---

### Provenance
- Log/matrix: `slurm/logs/s2s_extract_13169.log:37-49` (argmax `[1,0,3,3]`, matrix); recomputed in
  Python (argmax [1,0,3,3]; σ⁻¹=[1,3,0,2]; same-colour 0.809 / diff 0.79208 / adv +0.01692;
  both-late 0.8998 > both-early 0.831 > mixed 0.7272; global max (3,3)=0.939).
- Control source: `scripts/analysis/s2s_extract.py:115-235` (`encode_frameset`), `:245-286`
  (`temporal_positive_control`).
- Structural claim (installed `transformers==4.49.0`, HateVideo env)
  `.../models/qwen2_5_vl/modeling_qwen2_5_vl.py`: LLM causal — `:723` `is_causal=True`, `:975-998` SDPA
  path, `:1244-1325` `_update_causal_mask`, `:1327-1394` mask build (`:1371` `arange>cache_position`,
  `:1380` `causal_mask *= diagonal_attend_mask`), no vision-token unmasking; vision encoder block-diagonal
  — `:265-269` (eager), `:314-316` (sdpa); temporal-major layout `:466-505,529-534,560-562`.
- Governing docs: `refine-logs/S2S_GATE0A_POSTMORTEM.md`, `research-wiki/experiments/exp-s2s-r3.md`,
  `refine-logs/S2S_PROBE_DESIGN.md`, `refine-logs/S2S_PREREG_REVIEW.md` (A1), `refine-logs/S2S_CODE_REVIEW.md`.
- Cross-line: `refine-logs/W2C_FORENSIC_RECON.md`, `refine-logs/W2C_ORDER_PRECHECK_RECORD.md`,
  `refine-logs/W2A_FORENSIC_RECON.md`.
- Repo HEAD at ruling: `a1b2439`. Zero GPU / SLURM / Modal used.
