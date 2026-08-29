# C04 Refined Proposal V2 — Reliability-Sealed SPaSH Tensor

**Status:** `FROZEN / PENDING FRESH INDEPENDENT DESIGN REVIEW`  
**Execution authority:** none  
**Immutable anchor:** `refine-logs/C04_PROBLEM_ANCHOR.md`

## One contribution, two claims

The sole contribution candidate is an ordered source–proposition–
presenter-stance–harm interaction used as train-only privileged supervision for
the one native embedding consumed by ordinary top-20 kNN.

- **C1:** the four-way binding contains conditional information not explained by
  the same four fields, their lower-order interactions or a flexible
  concatenation.
- **C2:** that interaction can be internalized into the native video embedding;
  the teacher, fields and reliability record are absent at development/test.

No claim is made for first tensor fusion, first rationale distillation, first
stance reasoning, first pragmatic decomposition or first train-only teacher.

## Exact teacher record

For each train video the same frozen local
`Qwen/Qwen2.5-VL-7B-Instruct` sees eight uniform full-video frames and the native
transcript only. For a video with `N` decodable frame indices, requested indices
are `floor((i+0.5)N/8), i=0..7`; a requested decode failure becomes a fixed black
frame plus `frame_decode_failed=true`, with no neighboring-frame retry.
Transcript text is NFKC normalized, line endings become `\n`, and content over
2,048 Unicode scalar values becomes first 1,024 + `\n<TRUNCATED>\n` + last
1,024. Decoding is greedy: `do_sample=false`, `temperature=0`,
`num_beams=1`, `max_new_tokens=256`.

Both prompts require JSON only:

```json
{
  "source_relation": "current_presenter|quoted_or_embedded|performed_or_lyric|mixed|uncertain",
  "proposition": "one neutral bounded clause",
  "presenter_stance": "endorse_or_promote|reject_or_counter|report_or_describe|perform_without_clear_commitment|uncertain",
  "protected_target": "race|ethnicity|religion|nationality|gender|sexual_orientation|disability|other_protected|no_protected_target|uncertain",
  "harm_act": "attack|dehumanize|threaten|exclude|harass|other|none|uncertain",
  "confidence": {"S": 0, "P": 0, "T": 0, "H": 0}
}
```

Each confidence is an integer `0..4`. `H` is the ordered rendering of
`protected_target` and `harm_act`. `none` is not a video label. The proposition
is at most 32 whitespace tokens for English or 64 Unicode scalar values for
Chinese. Neither prompt may request or emit a hate verdict, moderation action,
free rationale, timestamp or evidence span.

**Prompt A, primary:** “Extract the origin relation, the single neutral
proposition currently presented, the current presenter's stance toward it, and
any protected-target harm relation. Describe roles, not whether the video is
hateful. Use only the supplied frames/transcript and return exactly the schema.”

**Prompt B, challenger:** “Independently restate one bounded literal proposition
from the supplied evidence, then identify who voices it, whether the present
speaker supports/counters/reports/performs it, and its protected-target plus harm
act if any. Do not judge the video. Return exactly the same schema.”

The full byte-exact system wrapper, templates, enum order and schema must be
hash-frozen at code review. Prompt A supplies canonical content when the two
forms agree; Prompt B is never merged into A.

## Reliability, conflict and fallback

Validity is per slot. Enums require exact membership; P requires a nonempty
bounded clause. Normalize P by NFKC, case-folding Latin, whitespace collapse and
terminal-punctuation removal. Proposition agreement is cosine `>=0.80` under
the frozen teacher token-embedding encoder; enum/H agreement is exact.

For every slot:

- `stable`: A and B are valid, agree, and both slot confidences are `>=3`;
- `single_valid`: exactly one form is valid with confidence `>=3`, while the
  other is parse-invalid/missing;
- `conflict`: both are valid but disagree, or any available valid form has
  confidence `<3`;
- `missing`: neither form is valid.

Canonical content is A for `stable`, the one valid form for `single_valid`,
`CONFLICT_<slot>` for `conflict`, and `MISSING_<slot>` for `missing`. Every ID is
retained. There is no retry, sample selection, confidence weight, loss weight,
router or threshold-conditioned branch.

The reliability state and numeric confidence are audit sidecars and are not
model inputs. The explicit conflict/missing content sentinels necessarily enter
the slot rendering; therefore `FALLBACK_COLLAPSE`, `SHUFFLE_FALLBACK_MASK` and
`NOISE_FALLBACK_MASK` are mandatory corruption controls. Report prompt parse
rate, each slot's four-state distribution, joint all-four coverage, conflict/
missing/fallback rates, A/B disagreement and every corruption sensitivity by
dataset. No human factor gold is asserted.

## Exact ordered tensor

Let `E(s)` be the mean of the frozen teacher's input token-embedding rows for
the exact canonical slot rendering, with hidden size `d0=3584`. Let
`eps=1e-12`. For nonzero `x`, `safe(x)=x/max(||x||2,eps)`; for `||x||2<=eps`,
`safe(x)=0`.

Each role map is a deterministic pseudorandom signed row-orthogonal operator
`R_f in {-1,0,+1}^{256 x 3584}`. Starting from identity indices `0..3583`, run
Fisher–Yates using an infinite SHA256 counter stream
`sha256("C04-SPASH-ROLEMAP-v2" || role || uint64_be(counter))`; consume unbiased
64-bit rejection samples, then consume one bit per selected coordinate for its
sign. The first 256 signed rows define `R_f`. The roles are ordered
`S,P,T,H`. No floating QR is used.

```text
u_f = safe(R_f E(render_f))
p_A = elementwise_product(u_f for f in A)
q_A = concat(safe(p_A), [1 if ||p_A||2 <= eps else 0])
q4  = q_{S,P,T,H}
```

Thus `q4` has 257 dimensions and the all-zero case is exactly
`[0,...,0,1]`, never NaN. The materialized index/sign arrays, canonical
serialization and four exact SHA256 values are mandatory code-review inputs;
missing hashes block execution.

`LOWER_ORDER_LE3` contains all 14 subsets with `1<=|A|<=3`, concatenates their
257-D `q_A`, and uses a separately seeded frozen signed row-orthogonal
compression to 257 dimensions. `ADDITIVE` similarly compresses
`[u_S;u_P;u_T;u_H]` plus its zero flag. Their compression payload hashes are
also mandatory pre-execution fields. Labels never generate, rotate, sign,
select or normalize a target.

## Retained student and strong alternatives

All student arms start from the same native RGCL representation `z`. Four
capacity-matched heads predict `u_S,u_P,u_T,u_H`; all are retained when their
arm uses them. `FULL_Q4` composes predicted slots by the fixed operator above,
aligns both slots and `q4`, and appends a learned 257-D tensor branch to `z`.
The sole query/memory vector is the normalized concatenation; final inference
is the existing ordinary top-20 kNN with no teacher, second index, score fusion
or router.

Mandatory alternatives:

- `CONCAT_ALL4_MLP`: same four heads and slot losses; a two-layer MLP over their
  concatenation replaces the tensor and is retained at inference.
- `RETAINED_INDEPENDENT4`: same four heads/losses; four within-slot projections
  are summed without cross-slot multiplication and retained.
- `LOWER_ORDER_LE3`, `ADDITIVE`, `STANCE_ONLY`, `HARM_ONLY`.
- `CAPACITY_ONLY_NATIVE`: identical retained branch architecture but no teacher
  target.

For each comparison, output dimension is identical, trainable parameter counts
must differ by at most 1%, optimizer steps/batches are exact, and measured
forward FLOPs must differ by at most 5%. Hidden widths and exact counts are
frozen at code review before labels/results. This retained independent-four arm,
not historical discarded-head P4, is the binding P4/C5/structured-KD control.

Mandatory perturbations:

- `TUPLE_SHUFFLE`: within dataset and outer split, sort IDs by
  `sha256("C04-TUPLE-v2"||video_id)` and rotate complete tuples by one.
- `SLOT_SHUFFLE_f`: independently sort with the slot tag and rotate only slot
  `f` by one; the other three remain at the original ID.
- `ROLE_PERMUTE`: cycle payload assignments
  `S<-P, P<-T, T<-H, H<-S` while preserving each destination role map.
- `REMOVE_f`: replace only slot `f` with literal `REMOVED_<f>`; never zero,
  conflict, missing, or sample deletion.
- `NOISE_MATCHED`: fit mean/covariance on the outer-train target only, clip
  negative eigenvalues to zero, generate deterministic Gaussian vectors from
  the frozen arm seed, apply the observed outer-train norm distribution by
  hash-rank pairing, and preserve the original fallback-mask frequency.
- `FALLBACK_COLLAPSE`, `SHUFFLE_FALLBACK_MASK`, `NOISE_FALLBACK_MASK` as defined
  above.

All permutations are label-blind derangements. A split of size below two is an
operational HALT. Held-fold and dev transformations use parameters learned from
the corresponding outer-train portion only.

## Novelty boundary after adversarial comparison

| Prior | Occupied mechanism | What C04 may still test; forbidden claim |
|---|---|---|
| RAMF, TMLR 2026 | objective/hate-assumed/non-hate-assumed reasoning text fused with multimodal features | C04 removes the teacher and distills typed role binding into one kNN embedding; it cannot claim first structured reasoning/fusion |
| LEAF, Findings ACL 2026 | gold-label-grounded Self-Grounding CoT and stage-wise generative SMM distillation with explain-then-predict inference | C04 teacher is label-blind and its student predicts no rationale; it cannot claim first hateful-video reasoning distillation |
| TFN / LMF | outer-product and low-rank tensor fusion of modality features | C04's object is a privileged semantic-role interaction, but tensor multiplication itself is old; survival requires beating CONCAT/ADDITIVE/LOWER-ORDER |
| DR-HM, Findings ACL 2026 | cognition-aware synthesis, SFT and A-GRPO | C04 has no rationale synthesis objective or RL; it cannot claim first decomposition/reasoning supervision |
| Intent Projection | literal/pragmatic orthogonal projection | C04 binds source/stance/target-act and removes the teacher at inference; it cannot claim first literal/pragmatic factorization |
| P4/C5 and generic KD | independent structured targets or train-only semantic teacher transfer | C04 must beat the retained independent-four and flexible concat controls; train-only KD is not novel |

The defensible paper-level delta is therefore conditional, not conjunction-based:
**a label-blind typed role interaction has an effect beyond flexible fusion and
independent structured KD, and that excess effect survives teacher removal in
the ordinary kNN memory geometry.** Failure against either strong control kills
the C04 novelty/mechanism claim even if accuracy rises.

## Authorization boundary

This file permits only fresh independent design review. No implementation,
prompt call, cache opening, label access, map materialization, Python/test run,
GPU, SLURM or test action is authorized.

