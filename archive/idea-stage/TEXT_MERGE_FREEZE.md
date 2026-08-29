# TEXT_MERGE — freeze

**Frozen 2026-08-13 (Pacific/Auckland), before any downstream metric of this experiment
was computed.** Repo HEAD at freeze time: `744d31ca3bd8a4cfd1df9b15a971338fdf6bad40`.
Nothing in §§1–8 is edited after results exist. Results go to
`idea-stage/TEXT_MERGE_RESULT.md`.

---

## 1. Question

`idea-stage/DESC_CHANNEL_RESULT.md` killed the MLLM perceptual description **as a new
768-d third input stream**, and its controls located the cause in the *stream*, not the
*content*: routing the plain transcript through the same new stream cost −0.0327, pure
noise cost −0.0539, shuffled descriptions (−0.0150) cost *less* than real ones (−0.0270).
§8 of that document states the untested alternative explicitly — put the description into
the **text channel proper**, so no new stream and no new parameter exists.

This experiment does exactly that. The description text is merged into the transcript
string **before** the text encoder runs; the encoder, the prompt scaffolding, the pooling
span, the feature dimensionality and the head architecture are all unchanged. The only
thing that differs between arms is the characters in the `Transcript: ` slot of the
encoder prompt.

Zero paid-API cost: the descriptions are the already-generated, already-paid
`idea-stage/desc_channel/descriptions_hatemm.jsonl` (1034/1066 rows with a parsed
six-field description). No API of any kind is called by this experiment.

---

## 2. Deviation D0 — the encoder cell is the **base** Qwen2.5-VL-7B, not the LoRA cell

The task specified the `LORA/HateMM/L1/I1` cell whose banked test macro-F1 is
**0.8774 ± 0.0041**. That cell's text features were produced by
`src/utils/generate_VideoMLLM_embedding_lora_HF.py`, which merges the LoRA adapter
`logging/lora/HateMM_curric` into the frozen base before the forward passes.

**That adapter does not exist and cannot be recovered.** Verified before freezing:
`logging/lora/` is absent on this workstation; `find / -maxdepth 6 -name
adapter_config.json` returns nothing; the B2 backups contain no such adapter
(`RGCL_video/adapters/` holds only the `lora_p9` family; `manual_backup_2026-08-06/RGCL/`
contains `logging/{slurm,temporal_memory}` and `data/lora_{frames,sft}` but no
`logging/lora`). `refine-logs/LORA_HATEMM_FORENSIC_RECON.md` already recorded
`logging/lora/` as `{MHC, MHC_zh}` only. Re-encoding in the LoRA text space is therefore
impossible, which is the same blocker `DESC_CHANNEL_FREEZE.md` §4 recorded — it is what
forced that experiment into the third-stream design in the first place.

**Frozen substitution:** the encoder for this experiment is the **frozen base
`Qwen/Qwen2.5-VL-7B-Instruct`**, i.e. the project's `QWEN` cell, extracted by
`src/utils/generate_VideoMLLM_embedding_HF.py` — the *parent* of the LoRA script, byte-
identical in prompt scaffolding, frame sampler, pooling span and cache contract. Base
weights are present in the local HF cache. Published banked value of this cell:
`QWEN/HateMM/L1/I1` test macro-F1 **0.8640 ± 0.0097**
(`idea-stage/RGCL_ABLATION_RESULT.md` §3), 1.3 points below the LoRA cell.

Consequences, accepted and frozen:

- The in-table baseline is **arm A0 of this experiment**, re-extracted here on this
  hardware. It is **not** comparable to 0.8774 and no row of this experiment will be
  placed in the same table as any LoRA-cell number.
- **Every arm, including A0, is re-extracted from scratch** by the same script in the same
  process on the same GPU, so no cross-hardware drift enters any paired comparison.
  The banked `Qwen2.5-VL-7B-Instruct_HF` cache is used only (a) for `img_feats`, which do
  not depend on the transcript and are therefore identical across all four arms, and
  (b) as a drift diagnostic (cosine of our A0 text vectors against the banked ones, to be
  reported, not used in any decision).

---

## 3. Arm text construction (frozen)

Imported, not redefined:

- **Defect rule** — `idea-stage/desc_channel/defect.py` verbatim:
  `DEFECT(i) ⟺ U_i < 10 OR nwr_i ≥ 0.30` over `/usr/share/dict/american-english`.
  Counts: train 140 / val 26 / test 52 = **218 of 1066**, of which 74 are literally empty
  and 144 are long-but-garbled. `sha256` of the sorted 218-id defect list:
  **`ca1b6f8941e2d672dfbf356a0d011a6985539e48638cc887413545df38e01003`**.
- **Descriptions** — `idea-stage/desc_channel/descriptions_hatemm.jsonl`,
  `sha256 755f911674f34ddaa6f5527cfdb5faa11c1503a741c2e7da9a1d82188cee6289`,
  1066 rows, 1034 with a parsed six-field description.
- **Description text** `DESC(v)` — the same six-field join used by
  `desc_channel/build_desc_feats.py`:
  `"Scene: {scene}\nPeople: {people}\nActions: {actions}\nOn-screen text: {on_screen_text}\nFormat: {production_format}\nAudio cues: {audio_visible_cues}"`.
  Empty string for the 32 videos without a description.
- **Mismatch permutation** `π` — the same fixed derangement,
  `numpy.random.default_rng(20260813).permutation` over the sorted 1066 ids, redrawn
  until it has no fixed point.

**Merge rule** (`idea-stage/text_merge/textmerge.py::merge`):

```
merge(t, d) = t                if d == ""            # no description available
            = d                if t.strip() == ""    # empty transcript -> REPLACED
            = t + "\n" + d     otherwise             # garbled transcript -> APPENDED
```

| arm | string placed in the `Transcript: ` slot | rows differing from A0 |
|---|---|---|
| **A0** | `TRANSCRIPT(v)` verbatim — baseline | 0 |
| **TMt** | `merge(TRANSCRIPT(v), DESC(v))` if `DEFECT(v)` else `TRANSCRIPT(v)` — **headline** | 215 |
| **TMall** | `merge(TRANSCRIPT(v), DESC(v))` for every v — undifferentiated control | 1034 |
| **TMshuf** | `merge(TRANSCRIPT(v), DESC(π(v)))` if `DEFECT(v)` else `TRANSCRIPT(v)` — mismatch control | 215 |

`sha256` of the frozen `{id: text}` map per arm (printed by
`python idea-stage/text_merge/textmerge.py`):

| arm | sha256 |
|---|---|
| A0 | `2f5d893726d59dfe51ed797fb99b2a0098f4440f6a75fd2c6072e198b4621ac0` |
| TMt | `55149b3f39f9611a91992be653a3e95d9199c28833331ce5f6a7f5ae90db28cb` |
| TMall | `0553f0d1d6089a6ef04df0c73607ad454f7f8e43b46a0ad6a6703be61bb1e269` |
| TMshuf | `6bb32046dc5029ee2a43744100b78f10efba37b8fe2b4a1624ead470a88a7448` |

Nothing else about the prompt changes: `TEXT_INSTRUCTION`, `"\nTitle: "`,
`"\nTranscript: "` and the `"(none)"` placeholder are the deployed English literals
(HateMM carries no `title` field, so the title slot is `(none)` in every arm of every
video, exactly as in the banked cache).

---

## 4. Encoding (frozen)

`idea-stage/text_merge/extract_text_feats.py`, importing
`src/utils/generate_VideoMLLM_embedding_HF.py` as a module — the frame sampler
(`load_video_frames`, decord → PyAV), the message builder, `_encode(..., span="response")`
and `read_gt` are the production functions, not copies.

- Model `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, `attn_implementation="sdpa"`, `eval()`,
  `torch.no_grad()`, single forward, no generation.
- 8 uniformly sampled frames per video, `max_pixels = 360*420` set at processor
  construction — identical to the banked extraction.
- `text_feats` = L2-normalised mean of the last-layer hidden states over the trailing
  assistant-header span; 3584-d.
- The 8 frames are decoded **once per video** and reused for every arm; identical prompts
  across arms are encoded once (deduplicated by prompt sha256), so a defect video costs 3
  forwards and a clean video costs 2 — **2350 forwards for 1066 videos**.
- `img_feats` are copied from the banked `{split}_Qwen2.5-VL-7B-Instruct_HF.pt`; they are
  byte-identical across all four arms and enter every arm identically.
- Output caches `data/CLIP_Embedding/HateMM/{split}_TEXTMERGE-{ARM}.pt` in the loader
  contract `{ids: [[...]], img_feats, text_feats, labels}`.
- **Truncation**: the production extractor passes no `max_length` to the processor, so no
  truncation is applied at any length; the model context is 128 k tokens. Full prompt
  token counts (vision + text) are recorded per video per arm and the count exceeding the
  context window is reported (expected 0 — a dry run over 6 videos gives 755–1295 tokens).
- Videos whose frames cannot be decoded receive a zero vector in every arm (the
  production zero-vector guard); the count is reported.

---

## 5. Head training (frozen)

Identical to `idea-stage/desc_channel/run_arms.sh` arm A0 in every flag except `--model`
and `--exp_comment`; **no `--archive_feats`, no third stream, no extra parameter anywhere**.

```
python ./src/run_rac.py \
  --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
  --dataset HateMM --model TEXTMERGE-{ARM} \
  --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align \
  --hard_negatives_loss True --no_hard_negatives 1 --final_eval False \
  --seed {SEED} --group_name TEXT_MERGE_20260813 \
  --metric cos --loss triplet --batch_norm False --hybrid_loss True --warmup 5 \
  --majority_voting arithmetic --no_pseudo_gold_positives 1 --lambda_seg 0 \
  --contrast_mode none --exp_comment "_TM_{ARM}" \
  --Faiss_GPU False --force False --keep_epoch_ckpts True
```

- 4 arms × seeds **0, 1, 2** = 12 runs, paired by seed, **single submission** in one
  background process (`setsid nohup`, log `logging/runs/text_merge/run.log`, PID file
  `logging/runs/text_merge/run.pid`). No re-run, no tuning after any number is seen.
- **train** trains, **val** (`dev_seen`, 107) selects the epoch, **test** (`test_seen`,
  215) is reported. Epoch selection reused verbatim from
  `scripts/rgcl_ablation_analyze.py::parse_run`, head rung I1: `argmax` over epochs
  ≥ warmup(5) of (dev head acc, dev head roc).
- `--keep_epoch_ckpts True` so the defect-subset per-sample readout can be recomputed at
  the val-selected epoch (same reason as `DESC_CHANNEL_RESULT.md` deviation D3; it changes
  only which files are deleted after training).
- Test labels are read **only** to compute the reported test metrics, after the epoch was
  already selected on val.

---

## 6. Reported quantities

1. Test macro-F1 and test ROC-AUC per arm, per seed, mean ± std.
2. Paired-by-seed deltas vs A0 for TMt, TMall, TMshuf; plus TMt−TMall and TMt−TMshuf.
3. Truncation accounting: min/median/max prompt token count per arm and the number of
   prompts exceeding the 128 k context (expected 0).
4. **Defect-subset readout**: on the 52 DEFECT test videos, correct predictions per arm
   per seed at the val-selected epoch, mean over seeds, paired change vs A0; also for the
   26 literally-empty-transcript test videos, and for the 163 clean videos. Descriptive;
   it does not gate the verdict.
5. Drift diagnostic: cosine similarity of our re-extracted A0 text vectors against the
   banked `Qwen2.5-VL-7B-Instruct_HF` cache (per split, mean/min). Diagnostic only.
6. Distinction from Pro-Cap / HVGuard (targeted repair vs undifferentiated concatenation),
   written honestly against whatever TMall does relative to TMt.

---

## 7. Decision rule (frozen; primary = seed-paired mean Δ test macro-F1 of **TMt** vs **A0**)

| # | clause | requirement for **GO** |
|---|---|---|
| 1 | `mean(TMt − A0)` on test macro-F1 | `≥ +0.005` |
| 2 | sign agreement | positive on **3/3** seeds |
| 3 | mismatch control | `mean(TMshuf − A0) < 0.5 × mean(TMt − A0)` **and** `mean(TMshuf − A0) < +0.005` |

**GO** iff clauses 1–3 all hold. **KILL** otherwise. No AMBIGUOUS band: this is a cheap
experiment and a borderline result is a kill.

Clause 4 (**reported, not gating**): if `mean(TMall − A0) ≥ mean(TMt − A0)`, the targeted
gate carries no incremental value over undifferentiated concatenation, and
`TEXT_MERGE_RESULT.md` must say so in those words — i.e. the differentiation claim against
the Pro-Cap / HVGuard family has failed — regardless of clauses 1–3.

---

## 8. What this experiment cannot show

- One dataset (HateMM), one encoder cell (base Qwen2.5-VL-7B), one loss rung (L1), one
  readout (I1). No claim of generality, and — because of D0 — no claim about the LoRA cell.
- 3 seeds cannot separate a +0.005 effect from seed noise; the bar is a screening bar, not
  a significance test.
- A negative result here does not re-open the third-stream question, and a positive result
  here does not transfer to the LoRA cell without re-encoding, which is impossible until
  the adapter is retrained.
