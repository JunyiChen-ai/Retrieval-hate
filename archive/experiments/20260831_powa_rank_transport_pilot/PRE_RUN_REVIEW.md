# Independent pre-run review — POWA rank transport pilot

Date: 2026-08-31  
Reviewer role: independent Rule-9 implementation/evaluation review  
Verdict: **PASS**

This is the final re-review after the initial BLOCK and the subsequent fixes.
It authorizes the six preregistered **Stage-V validation runs only**. It does
not authorize candidate test inference unless the frozen two-corpus Stage-V
supervisor later returns `ADVANCE_TO_STAGE_P`; `infer.py` independently
recomputes that authority before opening its one-shot test path.

The review covered `PILOT_PLAN.md`, `model.py`, `train.py`, `infer.py`,
`complete_run.py`, `summarize_stage_v.py`, `run_stage_v.sh`,
`audit_interventions.py`, `test_rank_transport.py`, `val_rank_diagnosis.py`,
the shared insertion implementation, the imported POWA/MACIL-SD code, and the
single shared evaluator. The PASS applies only to the hashes recorded below.
Any change to a source included in `CURRENT_SOURCE_MAP`, the frozen plan, or
this review invalidates the binding and requires a new review/snapshot.

## CRITICAL

No open critical finding remains.

### C1 resolved — shifted-mask feasibility and semantics

The shifted control now selects a same-length contiguous interval disjoint
from the donor interior. Insertion positions are prefiltered for that
feasibility, and recipients shorter than the label-independent boundary
buffer are excluded identically in all arms. The interval's permitted use of
excluded donor-boundary rows is explicit in the preregistered amendment and
tests the intended seam shortcut.

The exhaustive real-data audit was rerun against the final insertion source.
It completed without a crash and reproduced byte-for-byte the authoritative
artifact:

- HateMM: 18,600 items, 7,425 insertions, 25 identically ineligible short
  positive items, one recorded stability exception;
- HateClipSeg: 6,275 items, 5,475 insertions, no ineligible item or stability
  exception;
- every insertion had a valid 12--36-row donor window and a same-length,
  donor-interior-disjoint shifted mask.

Authority:
`runs/20260831_powa_rank_transport_pilot/preflight_intervention_audit.json`,
SHA256 `130ac220bc34a2c96104b3f90aa0c610d6a9ec438a0c79321fc0b3250f5246eb`.

### C2 resolved — matched causal interventions

Random draws use arm-free named keys. The exhaustive audit verified all
7,425 HateMM and 5,475 HateClipSeg insertions: negative-donor and shifted-mask
items match exactly in donor identity/crop/start/duration, insertion point,
recipient map, donor mask and all three composite tensors. Positive-donor
items share recipient, donor duration, crop and insertion draws; only donor
identity and identity-dependent window placement may differ. The manifest
persists the recipient map, both relevant intervals, and stability support.

### C3 resolved — validation-first, fail-closed test authorization

Inference defaults to validation. Test inference now requires all of the
following at the time of the request:

- the requested checkpoint is the seed-234 negative-donor core named by the
  stored two-corpus Stage-V summary;
- the current review is PASS and plan/review hashes match metadata and
  completion records;
- all six Stage-V records, evaluator artifacts, checkpoints, anchors, frozen
  hyperparameters, completion hashes and source snapshots pass a fresh
  `build_stage_v_summary` recomputation;
- the recomputed summary is exactly equal to the stored PASS summary and all
  six runs have the same source inventory;
- current source files match that inventory.

The test output is fixed to `<checkpoint>/test_scores.jsonl`. An atomic
`O_EXCL` claim in the checkpoint prevents concurrent, repeated, or
alternate-output exports. The claim deliberately fails closed: a runtime
failure after claiming requires explicit audit rather than an automatic retry.

## MAJOR

No open major finding remains.

### M1 resolved — authoritative control scores and metrics

Every selected validation artifact persists all 11 branches: POWA,
rank-transport, direct-additive, raw order, two tie controls, random
permutation, chronological/reverse, edge-first and center-first. The runner
passes the complete file through
`scripts/reproduction_baselines/eval_baseline_scores.py`; the supervisor
requires its score hash, corpus/split, full coverage, and all three metric
values for every branch to match the training record.

### M2 resolved — complete frozen Stage-V gates

The supervisor now explicitly gates epoch-0/no-insertion exact identity,
pooled feasibility, within-video gain, improved-video ratio, order uniqueness,
raw-order agreement, both tie controls, positive- and shifted-donor controls,
position controls, HateClipSeg high-positive-fraction behavior, hard multiset
identity, and direct-additive constraint attribution. The known HateMM
center-first shortcut remains a required falsification control; its prior
within-video ROC was `.76550` versus POWA `.57193`, so a position-driven gain
cannot advance.

### M3 resolved — run identity, completion and source provenance

The supervisor rejects partial, stale, relabelled, differently configured or
source-divergent runs. It validates corpus/arm/seed/output directory, all
frozen hyperparameters, plan/review hashes, actual anchor path and `model.pth`
hash, corpus-only metadata, evaluator and score hashes, selected files,
rank-head hash, completion record, and the complete source inventory. All
three arms within a corpus and all six runs globally must share the identical
snapshot content. Test authorization recomputes these checks rather than
trusting editable PASS booleans in an old summary.

### M4 resolved — bit-exact epoch-0 identity

The zero-residual path copies anchor scores before transport and asserts
pointwise equality for every validation video. The final HateMM smoke artifact
contains 109/109 exact rows, zero maximum pointwise error, equal epoch-0 metric
objects, and zero sorted-array error. This removes the earlier near-tie swap.

### M5 resolved — short-recipient exception is explicit

All valid recipient positions contribute to the position-sensitive order
loss. Only the centered stability term excludes the seam. If no stability
support remains, its term is zero while order supervision remains active; the
exception is preregistered, counted, and written to the manifest. The
exhaustive audit found exactly one such HateMM insertion and none on
HateClipSeg.

## Checks rerun on the final source

- 12/12 mechanism and real-data tests passed, including the historical
  HateMM shifted-mask failure, exact negative/shifted composites, shared
  positive draws, same-corpus donors, maps/masks, POWA freezing, hard
  permutation, epoch-0 identity, fail-closed authorization, and canonical
  atomic one-shot behavior.
- The exhaustive five-epoch intervention audit reproduced the checked-in
  authoritative JSON exactly (`cmp` success).
- The shared evaluator self-test passed its pooled ROC, pooled AP, within-video
  and 1 fps grid checks. No evaluation formula is copied into the experiment.
- Python source compilation, `bash -n run_stage_v.sh`, and `git diff --check`
  all passed.
- Usable train/val/test cohorts are disjoint: HateMM 744/109/215 and
  HateClipSeg 251/63/79. Both anchors declare only their matching corpus and
  no test-label training or selection.
- POWA is frozen by `requires_grad=False`, held in eval mode, executed under
  `no_grad`, detached before the rank head, and excluded from the optimizer.
- Crop scores are averaged on the existing POWA path and mapped to the final
  1 fps grid before hard assignment. Stable transport is an exact
  per-video permutation on that grid.
- The loss is position-sensitive: it indexes the supervised composite
  interval and frozen recipient candidates, while stability compares mapped
  non-seam recipient positions. It is not a permutation-invariant bag loss.

## Reviewed source hashes

| source | SHA256 |
|---|---|
| `PILOT_PLAN.md` | `e157b8e8a0cfd1babdc1103fe29b09d5a06158297d7054685010847d35ef3ef8` |
| `model.py` | `121db14492066d36dd24e9801b845e6d74453121f63a7e6978e20c3bebfd1552` |
| `train.py` | `565bd21117625350e8df61b486f1f5f844c06d8d1a70b2eb54e112c2a78af0c7` |
| `infer.py` | `c8bd3a6b0dc8a755cf26c9e30e7f9aa7a610aee48ac1a66d9421b1d2635b8576` |
| `summarize_stage_v.py` | `290e31b9c57321ac873ccb4f9483be881be3a4a783c848100639d74fb1cd4553` |
| `run_stage_v.sh` | `236019c2bfda9d764f9a32177a3be52820730316ca0921ea4fd7ab94b0077ede` |
| `complete_run.py` | `1a8cd86fe23ce3e0aa1cae0fd941e34a906345c0f08b3be5d3f1048ff0846951` |
| `audit_interventions.py` | `82f76d19206c87762ef415e745beb7479af5535caabf23ac8323585f82715820` |
| `test_rank_transport.py` | `3d5f602d90ec93042c043ea4ee1596f3dcc18b2a88f927afc83c8f7170cdc6b2` |
| `same_corpus_insertion.py` | `c1c7995e85a501bc33adad820ac26bb91e5afd14987f97e6c45493bfeed95275` |
| shared `eval_baseline_scores.py` | `2da04398c6e8bb66e275afe5664a06ec7d0f85d0f973ef5f67c01d0252093bdf` |
| shared `frame_eval_common.py` | `8b8c07d483af8ca53138a4a6144e9095781cdd563543308a55a4c2b6a03f801b` |

The formal runner additionally snapshots and hashes `README.md`,
`NOVELTY_SCOUT.md`, this PASS review, `src/weak_supervision/__init__.py`, and
the imported POWA/MACIL-SD model/data files. The supervisor compares that
complete inventory across all six runs and against the current working tree.

## Disposition

The implementation and evaluation pipeline satisfy Rule 9 for the frozen
pilot. The formal Stage-V runner may start from the source hashes above. Do
not export candidate test scores unless the unmodified supervisor reports
`ADVANCE_TO_STAGE_P` for both corpora and all controls; any Stage-V gate failure
is `KILL_BEFORE_TEST` under the frozen plan.
