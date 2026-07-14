# S2S G0-cond Probe — Executable Design Spec

**Companion to** `research-wiki/experiments/exp-s2s-r3.md` (pre-registration). This file is the
implementation spec: the exact extraction plan with a byte-level config-parity table vs the banked
pooled cache, the sbatch skeleton, the probe-script plan, the G-decomp / G-recon extraction-correctness
anchors, and the storage layout. **NOTHING here is authorized to run** until this file + the prereg
pass independent pre-registration review AND the three authored scripts pass a separate independent
code review and are hash-pinned (§10). Status: **APPROVED-WITH-AMENDMENTS (r1 applied 2026-07-14);
scripts authored, AWAITING INDEPENDENT CODE REVIEW — no submission authorized.**

> **r1 amendment note (2026-07-14).** The independent pre-registration review
> (`refine-logs/S2S_PREREG_REVIEW.md`, verdict APPROVED-WITH-AMENDMENTS) required five blocking
> amendments A1–A5 + seven non-blocking N1–N7. All are folded **in place** below (each amended passage
> tagged `(r1: A#/N#)`). A hash-freeze section (§10) and a revision history (§11) are added, and the
> five spec ambiguities the implementer had to resolve are recorded in §12.

Line name: **S2S** (set-to-set). Do NOT reuse "C1" (collides with the dead QLoRA route `C1 e2eq`).

---

## 1. What the banked pooled cache actually is (the correctness reference)

From `src/utils/generate_VideoMLLM_embedding_HF.py` (read 2026-07-14). The **img_feats** stream — the
only stream S2S turns into a set — is built as:

- `_encode(..., span="prefix")` (`:290-303`): `out = model(**inputs, output_hidden_states=True)`;
  `last_hidden = out.hidden_states[-1][0]` `[seq_len, 3584]`; `end` = index of the **last** `<|im_start|>`
  token (start of the assistant header); `pooled = last_hidden[:end].mean(dim=0)`; then
  `pooled.float()`, `F.normalize(p=2)` (`:321-322`).
- So **banked `img_feats[v]` = L2normalize( mean over tokens[0:end] )**, and `tokens[0:end]` =
  system + `<|im_start|>user` header + **ALL** vision-pad tokens (per merged spatio-temporal patch) +
  fixed `IMG_INSTRUCTION` text + `<|im_end|>`. It is **NOT** a per-frame mean, and **NOT** even a pure
  visual pool — it folds in the non-vision text tokens.
- `text_feats` (`:304-318`) is a trailing assistant-header last-token-style pool — **not a set**. S2S
  leaves it as the banked single vector (pipeline anchor + with-text sensitivity arm only).

Config facts (verified 2026-07-14): `num_frames=8`, `max_pixels=360*420=151200`, model
`Qwen/Qwen2.5-VL-7B-Instruct`, `torch_dtype=bfloat16`, `attn_implementation="sdpa"`,
`transformers==4.49.0`; vision config `temporal_patch_size=2`, `spatial_merge_size=2`,
`out_hidden_size=3584`. With 8 frames → **T = 4 temporal groups**, vision tokens laid out
temporal-major. Banked cache verified: HateMM 744/107/215, MHC-EN 549/80/161, `img/text` L2-normed,
HateMM train has **1 zero-img guard row** (undecodable video).

---

## 2. Config-parity table — S2S extraction MUST match the banked forward byte-for-byte

| knob | banked img_feats (`generate_VideoMLLM_embedding_HF.py`) | S2S extraction | parity |
|---|---|---|---|
| model id | `Qwen/Qwen2.5-VL-7B-Instruct` | same | **must** |
| class | `Qwen2_5_VLForConditionalGeneration` | same | **must** |
| dtype | `torch.bfloat16` | same | **must** |
| attn impl | `sdpa` | same | **must** |
| device | single CUDA (A100) | same | **must** (bf16 kernel parity for G-recon) |
| processor | `AutoProcessor.from_pretrained(model, max_pixels=151200)` | same | **must** |
| transformers | 4.49.0 | same | **must** |
| num_frames | 8 | 8 (primary) | **must** for G-recon |
| frame sampler | `_sample_frame_indices` (`np.linspace` round-clip) | reuse verbatim | **must** |
| decode | decord → PyAV fallback | reuse verbatim | **must** |
| messages | `_build_messages(frames, IMG_INSTRUCTION)` | reuse verbatim | **must** |
| IMG_INSTRUCTION | fixed string (`:45-47`) | reuse verbatim | **must** |
| chat template | `apply_chat_template(add_generation_prompt=True)` | same | **must** |
| span end | last `<|im_start|>` position | same | **must** |
| pooling (pre-norm) | `last_hidden[:end].mean(0)` | recompute for G-recon; ALSO decompose into groups | derived |
| spatial_merge_size | 2 (vision config) | read from `model.config.vision_config.spatial_merge_size` | **must** (r1: A1 grid gate) |
| zero-guard | undecodable → zeros | same policy → zero frame set | **must** |

**The only additions** vs the banked script: after `last_hidden`/`input_ids` are computed, (a) locate
the vision-pad tokens and partition them into T temporal groups; (b) pool each group → `g_t`; (c) pool
the non-vision prefix tokens → `p_S`, `|S|`; (d) run the **A1 HARD grid-consistency gate** + G-decomp +
G-recon; (e) save the frame-set cache. **No change to any forward-affecting knob** → the forward is
byte-identical → G-recon is expected near-bit-exact.

**(r1: A1) HARD grid-consistency gate (free, from the saved `video_grid_thw`).** Before pooling groups,
assert the vision-token partition is exactly the one the model laid out:

```
grid_t, grid_h, grid_w = video_grid_thw[0]            # saved per video
merge = model.config.vision_config.spatial_merge_size # = 2
per_expected  = (grid_h // merge) * (grid_w // merge)  # LLM tokens per temporal group
n_vis_expected = grid_t * per_expected
assert n_vis == n_vis_expected                         # catches a WRONG video_pad_id (mask count != grid count)
assert T == grid_t and n_vis % T == 0 and (n_vis // T) == per_expected   # catches a WRONG per-group size
```

This is **strictly stronger** than the old `n_vis % T == 0` check: it independently pins the vision/text
boundary (`video_pad_id`) via the token count and the per-group size via the grid, both from the
model's own `video_grid_thw`. The temporal-major contiguity that makes each group a **contiguous**
`per`-token block is proved by the modeling source (see §4, `modeling_qwen2_5_vl.py:466-505,529-534,
560-562`). HALT on any violation.

---

## 3. Stage E — extraction sbatch skeleton (the ONLY GPU)

New script **`scripts/analysis/s2s_extract.py`** (r1: §12 resolution 1 — the team-lead-directed path;
this file's earlier draft name `src/utils/generate_VideoMLLM_frameset_HF.py` is retired). It **imports
and reuses verbatim** from `src/utils/generate_VideoMLLM_embedding_HF.py`: `_sample_frame_indices`,
`_decode_with_decord`, `_decode_with_pyav`, `load_video_frames`, `_build_messages`, `read_gt`,
`SPLIT_TO_OUTNAME`, and the `IMG_INSTRUCTION` string — so every forward-affecting knob is the banked
one by construction, not a copy. Pseudocode for the added block inside the prefix forward:

```python
out = model(**inputs, output_hidden_states=True, use_cache=False)
last_hidden = out.hidden_states[-1][0]            # [seq_len, 3584]
input_ids   = inputs["input_ids"][0]              # [seq_len]
assert last_hidden.shape[0] == input_ids.numel()  # banked preflight invariant

video_pad_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
vis_mask = (input_ids == video_pad_id)            # vision-pad positions
im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
end = int((input_ids == im_start).nonzero()[-1]) if (input_ids==im_start).any() else seq_len
end = max(end, 1)

# --- G-recon reference: the banked-formula pooled vector, recomputed in THIS forward ---
prefix = last_hidden[:end].float()                # [end, 3584]
full_prefix_mean = prefix.mean(0)                 # pre-norm
banked_formula_vec = F.normalize(full_prefix_mean, p=2, dim=0)

# --- frame-group decomposition over the vision tokens WITHIN the prefix span ---
vis_pos = vis_mask[:end].nonzero(as_tuple=True)[0]        # vision positions inside prefix
grid_t, grid_h, grid_w = [int(x) for x in inputs["video_grid_thw"][0].tolist()]
T = grid_t                                                # temporal-group count (=4 for 8 frames)
n_vis = vis_pos.numel()
# (r1: A1) HARD grid-consistency gate — strictly stronger than `n_vis % T == 0`.
merge = int(model.config.vision_config.spatial_merge_size)   # = 2
per_expected  = (grid_h // merge) * (grid_w // merge)
n_vis_expected = grid_t * per_expected
assert n_vis == n_vis_expected, f"grid gate: n_vis {n_vis} != grid {n_vis_expected} (wrong video_pad_id?)"
assert n_vis % T == 0 and (n_vis // T) == per_expected, f"grid gate: per-group size mismatch"
per = n_vis // T
g   = []; n_t = []
for t in range(T):
    idx = vis_pos[t*per:(t+1)*per]
    g.append(prefix[idx].mean(0)); n_t.append(idx.numel())   # g_t unnormalized group mean
g = torch.stack(g)                                # [T, 3584]
n_t = torch.tensor(n_t, dtype=torch.float32)      # [T]

# --- non-vision prefix contribution (for exact reconstruction) ---
nonvis_pos = (~vis_mask[:end]).nonzero(as_tuple=True)[0]
p_S = prefix[nonvis_pos].sum(0)                    # [3584]
S   = float(nonvis_pos.numel())

# --- G-decomp (exact, free): weighted recombination == full prefix mean ---
recon_mean = (g * n_t[:,None]).sum(0).add(p_S).div(end)     # ( Σ n_t g_t + p_S ) / end
decomp_res = (F.normalize(recon_mean,p=2,dim=0) - banked_formula_vec).abs().max().item()
assert decomp_res <= 1e-5, f"G-decomp FAIL {decomp_res}"    # HALT

save g (fp16), n_t, p_S (fp16), S, end   # per video
```

**(r1: A1) Temporal-structure positive control (HALT gate, runs once at start of Stage E).** Before any
real video, the extractor synthesises **one** 8-frame clip = **4 distinct solid-colour frame-pairs**
(frames 0-1 colour C0, 2-3 C1, 4-5 C2, 6-7 C3), runs the identical forward, and asserts (i) each `g_t`
is nearest (max cosine) its own intended temporal slab's colour prototype and strictly farther from the
other three, and (ii) permuting the input frame-pair order permutes `{g_t}` by the same permutation.
This is the **only** check that actually exercises the token→temporal-group assignment (G-decomp and
G-recon are grouping-invariant — §4). HALT on failure; the real run does not start unless it is green.

**sbatch** (`scripts/slurm/s2s_extract.sbatch`, house ceremony): `conda activate HateVideo`; NO
`--time`; single submit; `PENDING (JobHeldUser)` → wait for auto-release, never force. Two invocations
of the script inside one job (dataset=HateMM, then MHC), `--splits train,val,test`, `--num_frames 8`.
Writes frame-set caches to the layout in §6 and a per-video gate log; prints the config echo + script
sha256 at start. A 1-item `--limit 1` smoke to a throwaway path is permitted before the real run
(leaves no real-path artifact). **Resumable:** the extractor writes one atomic per-video shard, and on
requeue skips any video whose shard already exists AND passes a re-loaded integrity check (keys present,
`g` shape `[T,3584]`, saved G-decomp residual ≤ 1e-5); the final per-split `.pt` is assembled from
shards in gt order. A requeue therefore never recomputes a completed video.

**Cost:** ~1856 videos × 1 prefix forward × ~1–2 s/video on A100 + model load ≈ **~1–2 GPU-h**. Half
the banked extraction (no response forward — `text_feats` already banked).

---

## 4. Extraction-correctness anchors (the adapted gate — REPLACES "mean-of-frames == pooled")

The naive gate is **invalid**: the banked pooled vector is not a mean over per-frame vectors (§1). Three
anchors replace it: a **grid-consistency gate** and a **temporal positive control** (both actually gate
the frame set), plus G-decomp and G-recon (which gate the *aggregate*).

> **(r1: A1) What G-decomp does and does NOT prove — correction of the v1 overclaim.** By construction
> `vis_pos` and `nonvis_pos` are **complementary within `[0:end]`**, so
> `Σ_t n_t·g_t + p_S = Σ(vision) + Σ(non-vision) = Σ[0:end]` and `(…)/end = mean[0:end]` **for ANY
> `vis_mask` and ANY grouping of the vision tokens.** G-decomp is therefore **grouping-invariant**: it
> proves only (i) the span `end` matches, (ii) the vision/non-vision partition is *complete* (covers
> `[0:end]`), and (iii) arithmetic self-consistency. It does **NOT** verify that `video_pad_id` is the
> correct vision/text boundary, and it does **NOT** verify the temporal grouping — a wrong
> `video_pad_id` or a spatial-major misread would still pass. G-recon is likewise grouping-invariant
> (it checks the pooled aggregate). The v1 claim "G-decomp proves the frame set is exactly the banked
> representation, decomposed … any residual > 1e-5 means the token→frame assignment is wrong" is
> **false as stated** and is retracted here. The frame set is gated instead by the two checks below.

### (r1: A1) Grid-consistency gate (MANDATORY, exact, free) — the vision/text boundary + per-group size
From the model's own `video_grid_thw` and `spatial_merge_size` (=2): assert
`n_vis == grid_t·(grid_h//2)·(grid_w//2)` (catches a wrong `video_pad_id` — mask count ≠ grid count)
**and** `T == grid_t` and `(n_vis // T) == (grid_h//2)·(grid_w//2)` (catches a wrong per-group size).
HALT on violation. This is what actually pins the vision/non-vision boundary and the equal partition —
the checks G-decomp cannot make.

### (r1: A1) Temporal positive control (MANDATORY, HALT) — the token→temporal-group assignment
On ≥1 synthetic clip (8 frames = 4 distinct solid-colour pairs) verify each `g_t` is nearest its
intended temporal slab and that permuting the input frame-pair order permutes `{g_t}` identically
(§3). This is the **only** check that exercises the grouping itself. The temporal-major contiguity it
confirms is proved by the modeling source: `modeling_qwen2_5_vl.py:466-505` (`get_window_index` builds
`index = arange(grid_t·llm_grid_h·llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)` — (t,h,w)
row-major), and `:529-534` (tokens are reordered for window attention) then `:560-562`
(`hidden_states = self.merger(...)`; `reverse_indices = argsort(window_index)`;
`hidden_states[reverse_indices]` **restores** the original (t,h,w) row-major order). So the merged
vision tokens emitted into the LLM sequence are temporal-major, each temporal group a **contiguous**
`(grid_h//2)·(grid_w//2)`-token block. (Verified against the installed `transformers==4.49.0` source in
both the HateVideo and ExMRD envs, 2026-07-14.)

### G-decomp (MANDATORY, exact, ~free) — arithmetic self-consistency of the decomposition
`L2normalize( (Σ_t n_t·g_t + p_S) / end )` must equal **this forward's own** banked-formula pooled
vector to **max-abs ≤ 1e-5**. Pure algebra (count-weighted group means + non-vision sum = overall
mean); bit-exact within a forward. Any residual > 1e-5 ⇒ an implementation bug in the decomposition
arithmetic (e.g. a dropped token, wrong `end`, incomplete partition) ⇒ **HALT**. **(r2: N-iv)** the
**authoritative** residual is the inline **float32** `decomp_res` computed at extraction (stored per
shard + `decomp_res_max` in the gatelog); an *offline* recompute from the saved `{g, p_S}` lands only to
~1e-3 because those tensors are stored **fp16** (§6) — so an offline check verifies the decomposition to
fp16 precision, not to the 1e-5 gate, which is the inline f32 number. **NB (r1: A1):** a green G-decomp
is *necessary but not sufficient* — it is the grid-consistency gate + temporal control above that certify
the frame set; G-decomp only certifies the aggregate arithmetic.

### G-recon (banked-cache parity anchor, tolerance-based) — fresh forward == banked forward
The fresh `banked_formula_vec` vs the **banked** `img_feats[v]` (`data/CLIP_Embedding/<ds>/<split>_
Qwen2.5-VL-7B-Instruct_HF.pt`, matched by id): **cosine ≥ 0.9999 AND max-abs-diff ≤ 1e-3** on the
L2-normed vectors, for every non-zero-guard video. Expected near-bit-exact (same A100 + sdpa + bf16 +
byte-parity §2). Report the distribution of cosines / max-abs over all videos; HALT if any video
breaches (excluding the 1 HateMM zero-guard row, which is zeros by policy in both). This is the analog
of B5's "reproduce the deployed numbers to 4 dp."

> There is **no banked raw pooled-kNN LOO number** to reproduce (the pipeline's retrieval runs over
> **headed** embeddings — `classifier.py:110-127`, `align`: `normalize(img_proj)⊙normalize(text_proj)`,
> map_dim 1024; the probe deliberately runs **un-headed**). So the extraction-correctness gate is
> **G-decomp + G-recon**, not a retrieval-number reproduction. The un-headed POOLED-LOO AUC is reported
> as an internal reference only.

---

## 5. Stage P — probe script plan (CPU, zero test touch)

New **`scripts/analysis/s2s_probe.py`** (r1: §12 resolution 1 — team-lead-directed path; earlier draft
name `s2s_g0cond_probe.py` retired). No head, no training, no GPU. Inputs: the §6 frame-set caches
(**train + val only** for retrieval) + the banked `text_feats` (with-text arm) + gold labels (train+val,
for Fano + oracle ceiling only).

**(r1: N4) Fail-closed no-test-touch guard.** The probe **never constructs, never opens, and asserts it
has not loaded** any `test_seen*` file. Concretely: it refuses to build a `test_seen` path; after
loading the memory it asserts `len(memory) == 851` (HateMM) / `629` (MHC-EN) — the exact train∪val
counts — so a stray test row cannot enter the memory or the vote. Any deviation is a HARD failure.

**Retrieval + vote (reuse the REAL vote, like B5):** build a `logging_dict` of top-20 neighbors per
LOO query (their labels + the arm's pairwise score as `retrieved_scores`), then call the pipeline's
`compute_metrics_retrieval(logging_dict, labels, majority_voting='arithmetic', topk=20, use_sim=True)`
from `src/utils/metrics.py` — do NOT reimplement the vote. Only the pairwise score differs across arms.
(Verified `metrics.py:262-320`: with `use_sim=True` the neighbour's signed label is **multiplied by the
arm's score** as a vote WEIGHT, then rank-weighted — hence A2 below.)

**Arms (per dataset, memory = train ∪ val, LOO):**
1. `POOLED` — `cos(mean_t g^Q_t, mean_t g^M_t)`, `g_t` = **unnormalized** group means (visual-isolated
   null; the "pool destroys alignment" baseline paired against SET on the identical `g_t`).
2. `SET` (primary) — `MeanMaxSim(Q,M) = (1/|Q|) Σ_q max_m cos(ĝ^Q_q, ĝ^M_m)`, `ĝ` = L2-normed frame
   vecs.
3. `SET-Chamfer` (single sensitivity) — `0.5[MeanMaxSim(Q→M)+MeanMaxSim(M→Q)]`.
4. `PIPELINE-ANCHOR pooled` — pooled-cosine over the **banked** `img_feats` (internal reference tying
   the probe to the actual pipeline cache; **not** the primary null, since it folds in the non-vision
   text tokens — §1).
5. `WITH-TEXT` — arms 1/2 visual score **+** fixed `cos(text_feats^Q,text_feats^M)` (identical channel
   both arms).
6. `ASYM` (r3: C2 fold) — `max_{m∈M} cos(ĝ^Q_pooled, ĝ^M_m)`, `ĝ^Q_pooled` = L2-normed pooled query
   (`normalize(mean_t g^Q_t)`), `ĝ^M_m` = L2-normed memory frame vecs. The pooled-query × set-memory
   off-diagonal cell of the MeanMaxSim grid (the `|Q|=1` reduction of arm 2) — the folded C2 candidate,
   computed on the **same** frozen `g_t`, run through the **same** LOO vote, paired, same seeds, with
   symmetric permutation-null + bootstrap treatment (`C2MEM_FORENSIC_RECON.md`).

**(r3: C2 fold) ASYM adjudication (pre-declared kill logic).** ASYM is credited/killed by two branches,
NOT a separate ceremony: **(a)** if S2S's oracle-ceiling Δacc < +0.04 on **every** dataset (§6.4
kill-switch fires), the whole don't-pool family — S2S **and** ASYM — is DEAD together, no ASYM
adjudication; **(b)** if symmetric SET survives (oracle did not fire), ASYM is dead unless it **beats
symmetric SET on acc AND macro-F1 (paired) on ≥1 dataset** — a beating ASYM escalates only as the
asymmetric arm of the §11 downstream stage. The probe reports `Δ(ASYM − SET)` (acc + mF1), its
permutation-null-95th and bootstrap-5th (same machinery as the SET/rank-only arms), and `asym_beats_set`
per dataset; the mechanical gate check emits the (a)/(b) branch outcome (NOT the binding verdict).

**(r1: A2) Rank-only sim-neutralized co-diagnostic — MANDATORY corroboration arm.** The `metrics.py`
vote uses the arm's pairwise score as a multiplicative WEIGHT, not merely a ranking key. MeanMaxSim (a
mean of maxes) has a compressed, upward-shifted range vs pooled cosine, so at *identical* retrieved
neighbours the two arms produce different votes → the raw paired Δ conflates neighbour-quality with
sim-scale weighting (a live false-PASS *and* false-KILL surface). To de-confound, compute a **rank-only**
variant of BOTH POOLED and SET: retrieve top-20 by the arm's own score (so *which* neighbours are
retrieved still differs), but set every `retrieved_scores` entry to the constant **1.0**, so the vote
reduces to the identical rank-position weighting in both arms and the only remaining difference is the
retrieved neighbour set. **Pre-declared corroboration rule:** the sim-weighted MeanMaxSim−POOLED primary
Δ (kept as the reported primary, since it mirrors the downstream vote) is credited only if the
**rank-only** paired Δ **matches its sign AND is itself significant** (its observed Δ above the 95th pct
of the §6.6 permutation null AND bootstrap 5th-pct > 0). Both arms are reported. A primary Δ that the
rank-only arm does not corroborate is treated as a sim-scaling artifact, not a mechanism effect.

**(r1: A3) Near-duplicate audit + near-dup-excluded sensitivity — MANDATORY.** SET matches on shared
segments; near-duplicate / same-source clips (plausible for MHC YouTube/Bilibili re-uploads, HateMM
re-uploads) let SET "win" by re-discovering duplicates rather than aligning hateful segments across
genuinely distinct videos — and the permutation null does NOT catch this (near-dups satisfy true
set↔label structure). Pre-declared (§12 resolution 3):
- **Audit:** for every distinct memory pair `(i,j)`, flag it near-duplicate if `pooled_cos(i,j) ≥ 0.995`
  **OR** `MeanMaxSim(i,j) ≥ 0.995` (the binding threshold; `MeanMaxSim ≥ 0.995` requires ~all query
  frames to have a near-identical memory match ⇒ GLOBAL near-duplication, **not** single-segment sharing
  — a lone shared hateful frame gives MeanMaxSim ≈ 1/T ≪ 0.995, so the flag cannot swallow the signal).
  Report per dataset: the flagged-pair count at thresholds **0.98 / 0.99 / 0.995** for both metrics, plus
  the single-frame `max_{q,m} cos` distribution (review-requested transparency).
- **Excluded sensitivity:** re-run the LOO retrieval dropping, for each query, every neighbour whose pair
  is flagged (≥ 0.995), and re-report the paired Δ(SET − POOLED). The SET advantage must **survive**.

All-pairs scoring is CPU-cheap: HateMM 851² ≈ 7.2e5 pairs × T²(=16) × 3584 ≈ 4e10 MACs → seconds with
batched `torch` (vectorize MeanMaxSim as `(Q̂ @ M̂ᵀ).max(dim=memory-frame).mean(dim=query-frame)`).

**Metrics:** AUC + acc + macro-F1 per arm; **paired Δ(SET − POOLED)** in acc and macro-F1 (primary).

**(r1: A4) Oracle-ceiling — exact deterministic procedure (video-level gold labels ONLY; no time-span
annotations anywhere).** Per-query oracle frame selection; memory keeps full sets. For query video `Q`
with gold label `y_Q ∈ {0,1}` and frame-groups `{ĝ^Q_1..ĝ^Q_T}` (L2-normed): for each candidate frame
index `t`, form the single-query-frame score `s_t(Q,M) = max_{m∈M} cos(ĝ^Q_t, ĝ^M_m)` to every memory
video `M` (LOO, `M ≠ Q`), run the real vote to get the continuous margin `v_t(Q)` (pre-sigmoid), and
select

```
t*(Q) = argmax_t  (2·y_Q − 1) · v_t(Q)          # frame that most confidently votes the CORRECT label
        tie-break: smallest index t
```

Then the oracle score for `Q` is `s_{t*(Q)}(Q, M)`; run the real vote over all queries → oracle acc/mF1;
report **paired Δ(oracle − POOLED)**. Gold enters ONLY to pick which of `Q`'s **own** frames to trust
(per-query, video-level label, no per-frame/time-span gold); the memory side is never oracle-selected
(no double-dipping). This upper-bounds "how much frame-level alignment structure pooling discards *could*
buy if we knew the discriminative frame." **(r1: N5) Ordering expectation:** oracle Δ should generally
**≥** raw Δ; a raw Δ materially exceeding the oracle ⇒ an oracle-construction bug (investigate, do NOT
auto-KILL). The oracle number is an upper bound and is **NEVER** reported as a result.

**(r1: N2) Fano-arm score:** retrieval by gold-label agreement uses the pairwise value `+1` if
`label(q)==label(m)` else `−1` (deterministic tie-break by memory index); vote acc must reach ≥ 0.99 on
both datasets, else the vote machine is VOID and no negative verdict is acceptable.

**Gates, in order (mirror prereg §7):**
- (i) **Fano** (`§6.3`, N2): retrieval by ±1 gold-label agreement → vote acc ≥ 0.99 each dataset, else
  VOID.
- (ii) **Oracle-ceiling** (`§6.4`, A4): per-query oracle-frame MaxSim; paired Δacc; DEAD if < +0.04 on
  every dataset.
- (iii) **Raw bar** (`§6.5`): HateMM mean paired Δacc ≥ +0.05 AND ΔmF1 ≥ +0.05, **corroborated by the
  rank-only arm (A2)**.
- (iv) **Permutation null** (`§6.6`, N1): the pre-declared **seed set 0..99** (≥100), shuffle frame sets
  across videos, the **same** permutation applied to both arms within a seed (paired Δ preserved);
  observed Δ > 95th pct. An optional finer per-frame-vector shuffle null is also reported (separates
  "alignment" from a generic "richer-key" effect).
- (v) **Bootstrap** (`§8`): ≥1000 query resamples; 5th-pct of paired Δ > 0 else D3-FRAGILE.
- (vi) **Dataset rule** (`§6.6`): assign outcome (a)/(b)/(c)/(d).

**(r1: N3)** The sensitivity arms (Chamfer, WITH-TEXT, 16-frame, near-dup-excluded) **cannot rescue a
failed primary**; the single pre-declared primary is the MeanMaxSim visual-isolated paired Δ (acc AND
F1) under the §6.6 dataset rule, corroborated by the rank-only arm — no OR-ing across arms/metrics
beyond the four fixed dataset-rule rows.

**Output (r1: §12 resolution 2 — reconciles the house raw-only rule with the team-lead's
auto-evaluation directive):**
- `refine-logs/S2S_PROBE_RESULTS.md` — the human deliverable: raw per-arm AUC/acc/mF1, paired Δ tables
  (sim-weighted AND rank-only), the Fano number, the oracle ceiling, the near-dup audit table, the null
  distribution percentiles, the bootstrap percentiles, the G-decomp max-residual + G-recon cos/max-abs
  distributions from Stage E, per dataset, and the raw arrays so one (dataset, arm) cell can be
  hand-recomputed — **with NO pass/fail interpretation** (verdict processing is independent, per house
  rule).
- `refine-logs/s2s_probe_results.json` — the same raw numbers PLUS a `mechanical_gate_check` block that
  does the pre-registered threshold arithmetic (raw Δ vs +0.05, oracle Δ vs +0.04, Fano vs 0.99,
  observed Δ vs null-95th, bootstrap-5th vs 0, rank-only corroboration) and prints each to stdout as
  `GATE …: value=… threshold=… → BELOW/ABOVE`. This block is **explicitly stamped "mechanical
  pre-registered arithmetic, NOT the binding verdict — the independent verdict reviewer renders the
  ruling"**; it automates the comparison the team-lead asked for without letting the executor pre-judge.

---

## 6. Storage layout (sub-GB, pre-declared)

Guard-excluded path (outside `logging/`), one `.pt` per (dataset, split):

```
data/CLIP_Embedding/<HateMM|MHC>/frameset_qwen7b_8f/
    train_frameset.pt   dev_seen_frameset.pt   test_seen_frameset.pt
```

Each `.pt` (contract mirrors the banked cache so a loader can consume it):
```
{ "ids":   [ [id, ...] ],                       # one sublist, banked contract
  "g":      fp16  [N, T, 3584],                 # frame-group vectors (unnormalized)
  "n_t":    int16 [N, T],                        # per-group vision-token counts (G-decomp)
  "p_S":    fp16  [N, 3584],                     # non-vision prefix sum (G-decomp)
  "S":      int32 [N],                           # non-vision prefix token count
  "end":    int32 [N],                           # prefix span length
  "labels": long  [N],
  "grid_thw": int32 [N, 3],                      # (T,H,W) per video, for audit
  "zero_guard": bool [N] }                       # undecodable → zero frame set
```

**Size:** N_total = 1856. Dominant term `g`: 1856 × 4 × 3584 × 2 B ≈ **53 MB**; `p_S` ≈ 13 MB; rest
negligible → **~80–110 MB** total (8 frames). 16-frame arm (`frameset_qwen7b_16f/`, T=8) ≈ 2×
→ ~160–210 MB. **Sub-GB confirmed.** (Disk 2026-07-14: 471 G free on /data; trivial.)

---

## 7. G-repro / consistency anchors summary (for the reviewer)

| anchor | check | tolerance | HALT? |
|---|---|---|---|
| **grid gate (r1: A1)** | `n_vis == grid_t·(grid_h//2)·(grid_w//2)` AND `n_vis//T == (grid_h//2)·(grid_w//2)` (vision/text boundary + per-group size) | exact | yes |
| **temporal control (r1: A1)** | synthetic 4-pair clip: each `g_t` nearest its slab; input-order permutation permutes `{g_t}` | exact | yes |
| G-decomp | `L2norm((Σ n_t g_t + p_S)/end)` == this-forward banked-formula pooled (aggregate arithmetic only — grouping-invariant, r1: A1) | max-abs ≤ 1e-5 | yes |
| G-recon | fresh banked-formula vec vs **banked** `img_feats[v]` | cos ≥ 0.9999 AND max-abs ≤ 1e-3 | yes |
| Fano (r1: N2) | ±1 gold-label-key LOO vote acc | ≥ 0.99 both datasets | probe VOID if fail |
| zero-guard | HateMM train zero-img rows handled identically both arms | count logged (expect 1) | no |

Banked-cache anchor files (for G-recon):
`data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt`.

---

## 8. Scope / veto / test-touch (executable restatement)

- **Stage E** = one sbatch, ~1–2 GPU-h, single submit, no `--time`, JobHeldUser=wait. Extracts all
  splits (test frame sets cached for the later formal stage) but **scores none**.
- **Stage P** = CPU, minutes, **ZERO test touch** (train∪val only; gold labels only for Fano + oracle
  ceiling, REFLECTION §4 compliant).
- Vetoes honored: single-dataset own-train memory; no OCR; no gold in-method; no cross-seed ensemble;
  no external API; no MLLM-score-as-signal; no kNN-pool expansion; local Qwen-7B only.
- **Not authorized here:** the downstream head-training formal stage (prereg §11) — a separate,
  independently-reviewed pre-registration, gated behind the Stage P oracle kill-switch.

---

## 9. Deliverable file map

- `research-wiki/experiments/exp-s2s-r3.md` — the pre-registration (this spec's companion).
- `refine-logs/S2S_PROBE_DESIGN.md` — this file.
- `refine-logs/S2S_PREREG_REVIEW.md` — the independent pre-registration review (A1–A5 + N1–N7).
- **`scripts/analysis/s2s_extract.py`** — Stage-E extractor (authored r1; awaiting code review).
- **`scripts/slurm/s2s_extract.sbatch`** — Stage-E sbatch (authored r1; awaiting code review).
- **`scripts/analysis/s2s_probe.py`** — Stage-P CPU probe (authored r1; awaiting code review).
- (produced at execution) `refine-logs/S2S_PROBE_RESULTS.md`, `refine-logs/s2s_probe_results.json`.

---

## 10. (r1: A5) Hash-freeze

Both design docs are pinned **now** (post-r1); the three scripts are hash-pinned **after** the separate
independent code review, before the single Stage-E submit (B3/B5 precedent). At submit time, every hash
below is re-verified; any mismatch aborts. The hash table is appended by the implementer after computing
sha256 over the final artifacts (see the r1 commit). **NOTHING is authorized to submit until the code
review passes and this table is filled and re-verified.**

| artifact | sha256 | status |
|---|---|---|
| `refine-logs/S2S_PROBE_DESIGN.md` (this file, post-r1) | _(filled at r1 commit — the file's own hash is recorded in the commit message / a sidecar, since a file cannot contain its own hash)_ | pinned r1 |
| `research-wiki/experiments/exp-s2s-r3.md` (post-r1) | see r1 hash table below | pinned r1 |
| `scripts/analysis/s2s_extract.py` | see r1 hash table below | authored; awaiting code review |
| `scripts/slurm/s2s_extract.sbatch` | see r1 hash table below | authored; awaiting code review |
| `scripts/analysis/s2s_probe.py` | see r1 hash table below | authored; awaiting code review |

**r1 hash table (computed 2026-07-14, appended at commit):**

<!-- S2S-R1-HASH-TABLE-START -->
| artifact | sha256 (r1, 2026-07-14) |
|---|---|
| `research-wiki/experiments/exp-s2s-r3.md` | `ffcc3a679b628e32d600bc0ecaeda0a0d7ac2d8da6387fbc6de2143506e3fe5b` |
| `scripts/analysis/s2s_extract.py` | `91637ccd52ce8fec5e29c8c8a30621e046d19aca3a5096d7e116beab059c4b74` |
| `scripts/slurm/s2s_extract.sbatch` | `818e0cd2f865fcb17e864e2519574abcb9ce3f8894b15d7a72e2811dddb47561` |
| `scripts/analysis/s2s_probe.py` | `53c6a6c80e3db336070521c5cf3daf5877e82cb24cfbb749162862ad4753c9c4` |
| `refine-logs/S2S_PROBE_DESIGN.md` (this file) | recorded in the r1 commit message (a file cannot embed its own hash) |

Re-verify every row at submit time; any mismatch aborts. The three scripts are pinned here as authored but
remain **AWAITING INDEPENDENT CODE REVIEW** — a mismatch after a review-driven edit is expected and the
table is re-pinned at that point.
<!-- S2S-R1-HASH-TABLE-END -->

**r2 hash table (re-pinned 2026-07-14 after `S2S_CODE_REVIEW.md` fixes B1–B3 + guards; SUPERSEDES r1).**
The r1 hashes above are retained for the audit trail; the **r2 hashes below are the current freeze** —
re-verify these at submit time. Still AWAITING the reviewer's one-line hunk re-check of B1/B2/B3.

<!-- S2S-R2-HASH-TABLE-START -->
| artifact | sha256 (r2, 2026-07-14) |
|---|---|
| `research-wiki/experiments/exp-s2s-r3.md` | `587f9b9b8e103758c34ffbb4c81aaa6796f231528b4612cca7c3d513504811c7` |
| `scripts/analysis/s2s_extract.py` | `41979f6a41c95e38a3cd875e11dc54a5a48eac9a5b908f295bad4d8d051cd23a` |
| `scripts/slurm/s2s_extract.sbatch` | `2dc0f90b03a44f45945cab3194f78ec97012fe7b157727cd50f64d88d56665dc` |
| `scripts/analysis/s2s_probe.py` | `949ebbdd432c9d72b1b164bc715da1cbba9fafc7f363337893f9813ff826f209` |
| `refine-logs/S2S_PROBE_DESIGN.md` (this file) | recorded in the r2 commit message (a file cannot embed its own hash) |
<!-- S2S-R2-HASH-TABLE-END -->

**r3 hash table (re-pinned 2026-07-15 after the C2/ASYM fold; SUPERSEDES r2 for the probe + docs).**
This is a **probe-only** amendment: the **extractor and sbatch are byte-identical to r2 and their r2
hashes are UNCHANGED** (explicitly re-stated below); only `s2s_probe.py` + both docs changed. Re-verify
these at submit time.

<!-- S2S-R3-HASH-TABLE-START -->
| artifact | sha256 (r3, 2026-07-15) | vs r2 |
|---|---|---|
| `scripts/analysis/s2s_extract.py` | `41979f6a41c95e38a3cd875e11dc54a5a48eac9a5b908f295bad4d8d051cd23a` | **UNCHANGED** (r2 = r3) |
| `scripts/slurm/s2s_extract.sbatch` | `2dc0f90b03a44f45945cab3194f78ec97012fe7b157727cd50f64d88d56665dc` | **UNCHANGED** (r2 = r3) |
| `scripts/analysis/s2s_probe.py` | `141a0441845d6175646d642a57b4534f78a48d96521ef3dc3a2d9fcf0f2301b3` | changed (ASYM fold) |
| `research-wiki/experiments/exp-s2s-r3.md` | `3f1f5b09e24c142dc07a76c5c21d2189a6d4a4b332c8f93dbb7e2eecc08b75b0` | changed (§5/§11) |
| `refine-logs/S2S_PROBE_DESIGN.md` (this file) | recorded in the r3 commit message (a file cannot embed its own hash) | changed (§5/§10/§11) |
<!-- S2S-R3-HASH-TABLE-END -->

---

## 11. Revision history

- **v1 (2026-07-14) DRAFT-UNREVIEWED.** Initial executable spec (config-parity table, G-decomp/G-recon
  anchors, storage layout, probe plan).
- **r1 (2026-07-14) APPROVED-WITH-AMENDMENTS, amendments applied.** Folded the five blocking A1–A5 and
  seven non-blocking N1–N7 from `S2S_PREREG_REVIEW.md`, in place:
  - **A1** — retracted the G-decomp overclaim (§4: it is grouping-invariant, proves aggregate arithmetic
    only); added the HARD grid-consistency gate (§2/§3/§4/§7) and the synthetic temporal positive control
    (§3/§4/§7); cited the layout proof `modeling_qwen2_5_vl.py:466-505,529-534,560-562`.
  - **A2** — added the rank-only sim-neutralized co-diagnostic vote arm + the pre-declared corroboration
    rule (§5).
  - **A3** — added the near-duplicate audit (pooled-cos OR MeanMaxSim ≥ 0.995) + near-dup-excluded
    sensitivity arm (§5).
  - **A4** — pinned the exact deterministic per-query oracle frame-selection formula, video-level gold
    only, tie-break, N5 ordering expectation (§5).
  - **A5** — added this hash-freeze section (§10) and revision history (§11).
  - **N1** null seeds 0..99 same-permutation-both-arms + optional per-frame null (§5); **N2** Fano ±1
    score (§5); **N3** sensitivity-cannot-rescue-primary (§5); **N4** fail-closed no-test-touch guard
    (§5); **N5** oracle≥raw ordering (§5); **N6** independent-verdict output split (§5); **N7** the
    provenance-cite blemish is corrected in the companion prereg §1 lineage.
- **r2 (2026-07-14) CODE-REVIEW FIXES APPLIED (`S2S_CODE_REVIEW.md`, APPROVED AFTER FIXES).** Script-only
  fixes (this spec's substance unchanged apart from N-iv wording + the §10 r2 hash table):
  - **B1** — `s2s_extract.py` G-recon compared a CUDA `grecon_vec` to a CPU banked vector → would crash
    on the first real video; both are `.cpu()`-ed before the compare.
  - **B2** — `s2s_extract.py` loaded the banked cache for G-recon only when `--limit` was unset, so the
    mandated `SMOKE=1` (`--limit 1`) run skipped gate 2; the banked cache is now always loaded so the
    smoke exercises all four hard gates (PREREG_REVIEW §5(iii)).
  - **B3** — `s2s_probe.py` A2 rank-only corroboration was sign-only; the rank-only arm now has its OWN
    permutation null (same permutation, both arms) + bootstrap, and the credit rule is sign AND
    rank-only observed Δ > null-95th AND rank-only bootstrap-5th > 0.
  - **NB-a** — `run_vote` drops NEG_INF (excluded/self) entries before the vote so a near-dup-excluded
    query can never multiply a label by ~−1e30.
  - **NB-b** — the `gpu:a100:1` gres verified schedulable (node advertises `gpu:a100:8`; the banked-cache
    producer `gen_embed_mllm.sbatch` used `gpu:a100:1` and ran) — kept, comment added.
  - **N-iv** — §4 G-decomp wording: the **authoritative** residual is the inline **f32** `decomp_res`;
    an offline recompute from the fp16-stored `{g, p_S}` only reaches ~1e-3 (fp16 precision).
  - N-i (dead no-op removed), N-ii (unused `T_nominal` param dropped from `shard_ok`), N-iii (sbatch exit
    cosmetic).
  Scripts + prereg re-hashed (§10 r2 table). Still AWAITING the reviewer's one-line hunk re-check; no
  submission authorized.
- **r3 (2026-07-15) FOLD C2 AS ASYM ABLATION ARM — probe-only amendment (`C2MEM_FORENSIC_RECON.md`).**
  Folded the round-3 C2 candidate into S2S as one pre-declared ablation cell (not a separate route): the
  **ASYM** arm `max_{m∈M} cos(ĝ^Q_pooled, ĝ^M_m)` (pooled-query × set-memory) added to §5 and to
  `s2s_probe.py` — computed on the same frozen frame vectors, run through the identical LOO vote, paired,
  same seeds, with symmetric permutation-null + bootstrap treatment (same per-seed permutations as the
  SET/rank-only arms). Pre-declared C2 kill logic: (a) S2S oracle Δ<+0.04 everywhere → don't-pool family
  (S2S+ASYM) dead together; (b) SET survives → ASYM dead unless it beats symmetric SET on acc AND
  macro-F1 (paired) on ≥1 dataset (a beating ASYM escalates only as the §11 asymmetric arm). The
  mechanical gate check emits the (a)/(b) outcome. **PROBE-ONLY: the r2 extractor + sbatch are
  byte-identical and their §10 hashes are UNCHANGED** (r3 table restates them); only `s2s_probe.py` +
  both docs are re-hashed. The queued smoke 13159 (extractor, r2 pins) is untouched. Awaiting the code
  reviewer's diff-only re-check before Stage P (which is anyway gated on extraction).

---

## 12. Spec ambiguities the implementer resolved (recorded for the code reviewer)

1. **Script paths.** The team-lead task named `scripts/analysis/s2s_extract.py`,
   `scripts/slurm/s2s_extract.sbatch`, `scripts/analysis/s2s_probe.py`; the v1 design §9 named
   `src/utils/generate_VideoMLLM_frameset_HF.py` and `scripts/analysis/s2s_g0cond_probe.py`.
   **Resolution:** follow the team-lead paths (operative directive); the extractor lives in
   `scripts/analysis/` and **imports the banked helpers verbatim** from
   `src/utils/generate_VideoMLLM_embedding_HF.py` rather than being a sibling superset — parity is
   preserved by import, not copy. Docs updated to these paths so hash-freeze/code-review reference real
   files.
2. **Pass/fail in the results.** House rule (design §5, review N6): executor writes RAW only, NO
   interpretation. Team-lead: "every kill-bar evaluated automatically … PASS/KILL printed."
   **Resolution:** the human `S2S_PROBE_RESULTS.md` stays raw-only (no verdict); the machine
   `s2s_probe_results.json` + stdout carry a `mechanical_gate_check` block doing the pre-registered
   threshold arithmetic, explicitly stamped "mechanical arithmetic, NOT the binding verdict — independent
   reviewer rules." Both requirements satisfied without the executor pre-judging.
3. **Near-dup threshold.** Review A3 example `≥ 0.98`; team-lead example `> 0.995`. **Resolution:**
   binding pre-declared flag = `pooled_cos ≥ 0.995 OR MeanMaxSim ≥ 0.995` (team-lead's stricter value);
   report the full distribution at 0.98/0.99/0.995 for both metrics + the single-frame max-cosine
   distribution (review's transparency ask). MeanMaxSim (not raw single-frame max) is the set-level flag
   so a lone shared hateful frame — the signal — is never mistaken for a duplicate.
4. **Oracle statistic (A4).** Under-specified in v1. **Resolution:** per-query oracle frame selection,
   query selects its own most-correct-voting frame via `argmax_t (2y_Q−1)·v_t(Q)` (smallest-index
   tie-break), memory keeps full sets, score = single-selected-query-frame-to-memory-set MaxSim, paired
   Δ(oracle − POOLED); video-level gold only, no time-span gold; N5 ordering expectation stated.
5. **Synthetic temporal control placement.** A1 item 3 is an extraction-time check (real forward on
   synthetic frames); the team-lead also listed it under the probe. **Resolution:** the authoritative
   g_t-assignment temporal control (real forward on 4-pair synthetic clip) runs in the **extractor** as a
   HALT gate; the **probe** additionally runs a CPU-only synthetic set-matching positive control (a
   planted shared-segment pair for which MeanMaxSim must exceed POOLED) validating the set-metric
   implementation. Both fold A1's spirit at the correct stage.
