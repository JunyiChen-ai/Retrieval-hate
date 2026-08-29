# S2S gate-0a (temporal positive control) — zero-GPU forensic postmortem

**Trigger:** smoke job **13169** (r3a extractor `07fd1621…`) FAILED in 18 s at gate 0a
(`s2s_extract.py:274`), *not* a crash — the r3a device fix held (execution passed the old line-199
site). The temporal positive control failed as a **scientific gate**: the synthetic-clip argmax
colour-matching assertion did not hold.
**Method:** zero GPU, no code change, no resubmit. Evidence = the raw failure log, the extractor
source, and the installed `transformers==4.49.0` Qwen2.5-VL modeling source. No smoke output survived
(the control runs *before* any real video; nothing is written to `slurm/logs/s2s_smoke_out_13169/`,
confirmed absent; no artifact under the real `data/CLIP_Embedding/…` path).

---

## 1. Raw evidence (transcribed from `slurm/logs/s2s_extract_13169.log`)

```
[gate 0a] temporal positive control: encoding 2 synthetic 4-pair clips ...
RuntimeError: [gate 0a] temporal assignment FAILED: argmax match [1, 0, 3, 3] != expected sigma [2, 0, 3, 1].
Cross-clip cosine matrix:            # M[i,j] = cos(A_i, B_j),  i = A-group (rows), j = B-group (cols)
[[0.801 0.853 0.705 0.635]
 [0.819 0.851 0.829 0.807]
 [0.674 0.808 0.877 0.881]
 [0.614 0.746 0.902 0.939]]
```
Job state: `FAILED`, ExitCode `1:0`, Elapsed `00:00:18`.

## 2. Exactly how the control is constructed (`s2s_extract.py:241-278`)

- `_solid(color, size=336)` → a **solid-colour** 336×336 PIL image. `colours = [R(220,30,30),
  G(30,200,30), B(30,30,220), Y(230,210,30)]`.
- `clip(order)` builds 8 frames = 4 **pairs** (2 identical frames per pair), pair `k` painted
  `colours[order[k]]`.
- Clip **A** = `order_a=[0,1,2,3]` → A-group content by temporal position: `A0=R, A1=G, A2=B, A3=Y`.
- Clip **B** = `sigma=[2,0,3,1]` → B-group content by position: `B0=B, B1=R, B2=Y, B3=G`.
- Both encoded with `encode_frameset(..., banked_vec=None)` → each returns `g` = `[T=4, D]`, where
  `g_t` = **mean of the LAST LLM-layer hidden states** over the vision tokens of temporal group `t`
  (`s2s_extract.py:129-130` `out.hidden_states[-1]`; `:184-191` group pooling).
- Assertion: `M=ga@gb.t()`; `match = M.argmax(dim=0)` (best A-group per B-group); require
  `match == sigma` (i.e. B-group `j`'s nearest A-group is the one sharing its colour, `A_{sigma[j]}`).
  Docstring assumption (`:247-248`): *"each g_t reflects its temporal slab and … permuting the input
  pair order permutes {g_t} by the SAME permutation."*

## 3. Hypothesis tests (in the priority order the lead requested)

### H-orientation (axis / σ-inverse bookkeeping bug) — **REFUTED**
`sigma=[2,0,3,1]`; its inverse `sigma⁻¹=[1,3,0,2]` (since σ[1]=0,σ[3]=1,σ[0]=2,σ[2]=3). A pure
transpose/axis bug (comparing A→B when the expectation was defined B→A, or argmax over the wrong dim)
reproduces `sigma⁻¹` **exactly**. Observed `match=[1,0,3,3]`. Compare:
`match` vs `sigma⁻¹` = `[1,3,0,2]` → agrees only at position 0 (1==1), disagrees at 1/2/3; and `match`
has a **repeat** (`3` twice) so it is **not a permutation at all** — impossible for any clean
orientation relabelling. **Orientation/bookkeeping bug is refuted** (as the lead predicted).

### H-content-tracking (are same-colour pairs the most similar?) — **NO** (content barely encoded)
Same-colour cross pairs (should be ≈1 if `g_t` were frame-local): `Red A0·B1=0.853`, `Green
A1·B3=0.807`, `Blue A2·B0=0.674`, `Yellow A3·B2=0.902` → **mean 0.809**. The other 12
different-colour entries → **mean 0.792**. The same-colour advantage is **+0.017** — negligible. If the
group vectors described frame-local content, same-colour would sit near 1.0 and dominate every column;
instead colour is almost absent from the similarity structure.

### H-cumulative-causal-context (is similarity governed by temporal position / context-depth?) — **YES**
Block means of `M` by position (early = groups 0–1, late = groups 2–3):
- both-late (i,j∈{2,3}): 0.877,0.881,0.902,0.939 → **0.900**
- both-early (i,j∈{0,1}): 0.801,0.853,0.819,0.851 → **0.831**
- mixed (one early, one late): → **0.73** (top-right 0.744 / bottom-left 0.711)

Monotone **both-late > both-early > mixed**, and the global max is `A3·B3=0.939` (last group vs last
group, **different** colours Y vs G). The single sharpest contrast: **same-colour, different-position**
`Blue A2(pos2)·B0(pos0)=0.674` ≪ **different-colour, same-late-position** `A3(pos3)·B3(pos3)=0.939`.
Similarity is dominated by *how deep the causal context is*, not by frame content.

**This is not a hypothesis about the code — it is what the code computes.** Verified in the installed
`transformers==4.49.0` source: the Qwen2.5-VL **LLM decoder is causal** (`modeling_qwen2_5_vl.py:723`
`self.is_causal=True`; `:975-997` SDPA with `is_causal`; `:1177,:1244` `_update_causal_mask` builds a
standard causal mask) with **no bidirectional unmasking of vision tokens** in the LLM — only the
*vision encoder* uses block-diagonal window attention (`:265-320`, within-frame). Therefore a vision
token at sequence position `p` attends to `0..p` only, and `g_t` = mean of the **last LLM layer** over
group-`t` tokens is a **cumulative causal-prefix summary conditioned on frames 0..t** — *by
construction*, not frame-local. `g_3` (all four groups of history) resembles any other late group
regardless of colour; `g_0` (no history) is the outlier. This exactly reproduces the observed matrix.

### H-control-flaw (are solid-colour stimuli too low-information?) — **CONTRIBUTING, but not sufficient alone**
Solid colours are pathologically low-content for a vision encoder trained on natural images, which
weakens the content axis and *amplifies* position dominance (H-content shows content ≈ 0.017). But low
content **alone** would give near-uniform similarity with unstructured noise, **not** the systematic
monotone position gradient of §H-cumulative. The stimulus weakness is a real confound that *co-occurs*
with — and cannot substitute for — the cumulative-causal mechanism, which is independently certain from
the source.

## 4. Verdict hypothesis

**PRIMARY: CONTROL-DESIGN FLAW — invalid *by construction* (not a fixable-stimulus flaw, not an
orientation one-liner).** The control's load-bearing assumption (`:247-248`, `:271-273`) is that `{g_t}`
are **frame-local and permutation-equivariant** — "permuting the input pair order permutes `{g_t}` by
the same permutation," tested via same-colour argmax. That assumption is **false for causal LLM
hidden states**: permuting input frames does not permute `g_t`, it **re-contextualizes** every
downstream group (each `g_t` conditions on frames `0..t`). No choice of synthetic stimulus can make a
permutation-equivariance / content-argmax control pass on a cumulative-causal representation — so
**"patch the frames and resubmit" will not work**, and the failure does **not** indicate a wrong
token→temporal-group assignment. (The assignment arithmetic is still validly gated by 0b grid-consistency
+ G-decomp, which the crash pre-empted before any real video.)

**SECONDARY (genuine, must escalate): PREMISE-REFRAME — not premise falsification.** The prereg §2/§4
framing ("per-frame set element … two videos that share a hateful *segment* … match on that segment")
is **imprecise**: the set elements are **cumulative causal group summaries**, not frame-local segment
descriptors. Two caveats keep this from being an automatic S2S kill:
1. The **banked `img_feats`** that S2S decomposes are *themselves* a pool of these same causal
   last-layer hidden states, and their pooled form already produced the **encoder-swap HateMM +3
   positive** — so "cumulative" is not disqualifying per se.
2. Whether **set-MaxSim over cumulative group vectors beats the pooled vector** is exactly the
   **empirical** question Stage P measures, guarded by the pre-declared **oracle kill-switch** (§6.4).
   The premise nuance changes the *interpretation* of a set element, not the validity of the probe.

**NOT orientation-bug** (H-orientation refuted).

## 5. Recommended disposition (for independent review — NOT executed here)

Do **not** patch-and-resubmit. Options for the reviewer to rule on:
- **(A) DROP the temporal positive control** as a gate (it tests a false frame-local/equivariance
  premise) and rely on the already-valid **0b grid-consistency + G-decomp** for the token→group
  assignment and on **Stage-P's oracle kill-switch** as the real premise test; correct the prereg §2/§4
  wording to "cumulative causal group summaries."
- **(B) REPLACE it with a causal-consistent control** that respects the representation — e.g. a
  *monotone-onset* test: two clips sharing the **same position-0 frame** but differing later must have
  `g_0` matching near-1 while later groups diverge (validly exercises the pos-0 = no-history invariant);
  or an *injection* test (a distinctive high-content frame at position `p` shifts every group whose
  cumulative window includes `p` and none strictly earlier). These test grouping/causality **without**
  assuming permutation-equivariance.
- **(C) If the reviewer judges cumulative reps incompatible with the "shared-segment" premise**, escalate
  a representation-source decision: last-LLM-layer (cumulative, banked-parity, current) vs **vision-encoder
  output before the causal LLM** (frame-local / within-frame-bidirectional, but a *different*
  representation with no G-recon parity to the banked cache). This is a design fork, not a bugfix.

**Bottom line:** the gate did its job — it exposed that the control encodes a false premise and that
S2S's set elements are cumulative causal summaries. This is a **control-flaw + premise-reframe**, not an
orientation bug and not (on this evidence) a premise falsification. Awaiting independent review before
any amendment, resubmit, or kill.

## 6. Connections
- W2-A forensic recon independently surfaced a **causal-masking finding** (task #39) — consistent with
  §3 H-cumulative here.
- S2S premise: `research-wiki/experiments/exp-s2s-r3.md` §2/§4 (set element wording), §6.4 (oracle
  kill-switch = the real premise test), §4 (banked img_feats = last-LLM-layer pool).
- Layout/causality source: `modeling_qwen2_5_vl.py:265-320` (vision window attn), `:723/:975-997/:1177/
  :1244` (LLM causal mask).
- Control code: `scripts/analysis/s2s_extract.py:241-284`.
