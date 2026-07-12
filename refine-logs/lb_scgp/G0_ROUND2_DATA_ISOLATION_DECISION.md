# LB-SCGP G0 Round2 Data-Isolation Decision

**Date:** 2026-07-11  
**Launcher metadata:** model `gpt-5.5`, `model_reasoning_effort=xhigh`, `--strict-config`. This records supplied launcher metadata only; no identity was inferred from runtime introspection.  
**Status:** decision document only. No sanitizer, G0, replay, or SLURM job was run.

## Metadata Inspection Outcome

Round2 inspected only filesystem metadata, code, and path conventions for a physically separated fold4 outer-train feature/subclip source. No `artifacts/lb_scgp/inputs/MHC_zh/fold4` train-only artifacts existed, and the visible CLIP feature/subclip paths were dataset-wide train caches by name. No `torch.load`, mixed feature tensor read, fold-label read, held-label/content read, `query_z`, or `query_labels` access was performed in this worker turn.

Because no physically separated source was found and no LB-SCGP formal freeze/job exists, the self-imposed byte-level non-opening contract is revised before formal freeze to a dedicated quarantine sanitizer. This revision does not weaken experimental leakage controls.

## Adopted Quarantine Protocol

- Sanitizer is a separate SLURM-only preprocessing job and namespace before G0 freeze.
- Quarantine source locators and source hashes live only in `configs/lb_scgp/lb_scgp_sanitizer_sources.json` and `artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json`; neither is a formal G0 input.
- Sanitizer imports/calls no model, optimizer, evaluator, teacher, MLLM, or OCR code.
- Selection is solely by exact `memory_ids`; `query_ids` is an exclusion sentinel only.
- Output parent labels come only from allowed `memory_labels`; source labels are ignored and not emitted.
- Held/query rows may exist transiently inside quarantined source storage, but cannot be selected, serialized into train-only outputs, logged as rows, scored, or passed downstream.
- Output schemas are whitelist-only and physically train-only:
  - `outer_train_features.pt`: `ids`, `img_feats`, `text_feats`, `labels`.
  - `outer_train_subclips.pt`: `subclip_img_feats`, `subclip_parent`, `labels`.
- Subclip labels must equal inherited parent labels.
- Sanitizer manifest records output/source hashes, code hash, row counts, ID hashes, access disclosure, zero network/external calls, SLURM ID, and no-clobber evidence.
- Formal G0 reads only sanitizer outputs, sanitized provenance, sanitizer decision, checkpoint, allowed bank members, remove ledger, code/config/docs. It never reads mixed caches or quarantine source config/manifest.
- Formal records distinguish `quarantine_mixed_storage_read=true` in the quarantine manifest from formal model/optimizer/evaluator outer-held read counts, which remain zero.

## Decision

Proceed only to independent review of the Round2 repair. Do not run sanitizer or G0 jobs until a reviewer accepts the revised data-isolation contract and implementation. The global target remains active and unmet; no G0 performance evidence exists.
