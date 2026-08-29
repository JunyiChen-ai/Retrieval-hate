# C03 Stage-0 Asset Audit

**Candidate:** `C03 Policy-Anchored Native MNTP`  
**Date:** 2026-07-29 (Pacific/Auckland)  
**Verdict:** `NO_MATCHED_TWO_DATASET_STAGE0_ASSET`  
**Scope:** filename/document/code inspection only; no cache was opened

## Binding eligibility rule

Before any teacher, GPU, extraction, or native MNTP training, the registry requires
an actual fold/deployed-head representation oracle with at least `+0.050` accuracy
and `+0.050` macro-F1 on both HateMM and MHC-ZH.

For C03, an eligible bank pair would have to differ only in the mechanism FULL will
train:

- native adaptation at each dataset's deployed Qwen2.5-VL task-LoRA weight point;
- identical train/dev rows, prompt, readout, pooling, fusion, head and retrieval path;
- a label-blind policy-conditioned MNTP mask/loss versus matched native MNTP-only;
- no test, external corpus, API, teacher output, per-item router, or alternate
  inference prompt.

No such pair exists.

## Existing train/dev assets

| Existing family on HateMM + MHC-ZH | What it changes | Why it is ineligible |
|---|---|---|
| deployed causal `...LoRA{,-curric}_HF.pt` | baseline task-LoRA representation | no native MNTP/policy pair |
| `...bidir_HF.pt` | attention mask only, no training | F72 crater; mask surgery is closed |
| `...bidir-meanpool_HF.pt` / `...bidir-textpool_HF.pt` | readout span under bidirectional attention | F92 readout route is closed and pooling is not C03's training mechanism |
| `...bidir-mntp_HF.pt` | published McGill text-model MNTP LoRA transplanted after task-LoRA merge | wrong weight point, no policy anchor, no native training; F93 is below the causal floor and kills only the zero-training shortcut |
| `...-ro_{L28,L24,ow_L28,ow_L24}.pt` | layer, prompt and pooling/readout endpoint | F70 is closed; standard/one-word endpoints are confounded and C01's existing endpoint route is scientifically killed |
| `...nullop2merge_HF.pt` | PEFT double-merge null path | numerical-path control only |

The filename/size audit covered only top-level `train_*` and `dev_seen_*` files.
No `test_seen` cache was opened, hashed, loaded, or used.

## Why an optimistic salvage oracle is still insufficient

A global interpolation or projection between causal, plain-bidirectional, and F93
transplant embeddings would measure whether a mismatched external MNTP residual can
be post-hoc salvaged. It would not isolate the label-blind policy-conditioned native
MNTP axis. Such an old-cache construction may strengthen a KILL, but it cannot PASS
the mandatory representation-matched Stage-0 gate.

Likewise, F70/C01 standard-versus-one-word banks cannot be combined with F93 to
manufacture a policy-MNTP proxy: those banks change prompt/pooling/readout, were
extracted on a different endpoint contrast, and their coordinate interaction is not
the mechanism C03 would train.

## Terminal consequence

Creating the missing bank would require implementing native policy-MNTP, training it,
and re-extracting train/dev representations. That is precisely the GPU work the
Stage-0 rule forbids before the matched oracle passes.

Therefore C03 is **design-infeasible under the current registry contract** and must
be frozen before implementation:

`KILL_C03_DESIGN_INFEASIBILITY`

This is not an experimental refutation of native policy-MNTP. It is a sequencing and
evidence-availability failure under the current mandatory gate.

No Python, cache loading, feature extraction, teacher call, GPU use, test access, or
SLURM submission occurred.
