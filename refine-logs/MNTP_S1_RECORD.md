# MNTP S1 — bidir + LLM2Vec mean-pool readout, NO training (execution record)

**Agent:** mntp-s1 · **Date:** 2026-07-27 NZST · **Repo HEAD at build:** `a3db06e`
**Stage:** S1 of the staged MNTP plan (`refine-logs/MNTP_FORENSIC_RECON.md` §6).
**Scope:** DEV ONLY. **ZERO test-touch.** No training, no adapter beyond the banked task LoRA,
no corpus ruling, no download. No `src/` file edited (only a new fork added). No `state/`,
prereg, or frozen artifact mutated.

**What S1 decides:** whether the F72 crater is **H1** (bidirectional attention breaks the
causally-trained weights) or **H2** (the deployed EOS-class text readout is mismatched to
bidirectional topology). These prescribe very different spends — H2 costs ~1 GPU-h and no
ruling; H1 routes to S2a/S2b at 2-8 GPU-h plus a corpus ruling — so the recon's cheapest-kill
discipline splits them before funding MNTP.

---

## 1. BUILD PROVENANCE

### 1.1 Artifacts

| file | sha256 | status |
|---|---|---|
| `src/utils/bidir_patch.py` | `36cedbac365b2b13c945adbe3437efdc61d8be15ecc85878eb9614225abe367b` | **frozen, VERIFIED unchanged** (matches recon §2.4) |
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | `75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399` | causal extractor, **unchanged** |
| `src/utils/generate_VideoMLLM_embedding_bidir_HF.py` | `03f39e09c417bbea291f3c06b787f5220693568cd613705693df1c2bf23e020d` | F72 artifact A2, **unchanged** |
| `scripts/analysis/mntp_rawkey_devscreen.py` | `8bc009e68833d8bad3aecb531c7c8b9879e05a2e00430465e0b2b4f05f9dede0` | recon screen, **unchanged** |
| `src/utils/generate_VideoMLLM_embedding_bidir_meanpool_HF.py` | *(recorded in §6)* | **NEW** — the S1 fork |
| `scripts/slurm/gen_embed_mllm_bidir_meanpool.sbatch` | *(recorded in §6)* | **NEW** — S1 extraction runner |
| `scripts/analysis/mntp_s1_devscreen.py` | *(recorded in §6)* | **NEW** — KS-MNTP-1 |
| `scripts/analysis/mntp_s1_cpuhead.py` | *(recorded in §6)* | **NEW** — KS-MNTP-2 |
| `scripts/slurm/mntp_s1_cpuhead.sbatch` | *(recorded in §6)* | **NEW** — KS-MNTP-2 runner (CPU, no GPU) |

Encoders (banked, unchanged, no re-SFT): HateMM = `logging/lora/HateMM_curric`
(`adapter_model.safetensors` sha256 `6571d132ef3218e4bdfcee98aab468df21f8aa83b16d623dd2098f8486394efa`,
the adapter behind `…-LoRA-curric_HF` / `…-LoRA-curric-bidir_HF`); MHC_zh = `logging/lora/MHC_zh`
(sha256 `35a510f4ad84542c798939cfdb340b00317a5b8a2c670b07ced8d1869dd7b438`, the adapter behind
`…-LoRA_HF` / `…-LoRA-bidir_HF`). Env: transformers 4.49.0, peft 0.14.0, torch 2.6.0+cu124.

### 1.2 The fork, and exactly what it changes

`generate_VideoMLLM_embedding_bidir_meanpool_HF.py` follows the F72 artifact-A2 thin-fork
pattern: it imports `read_gt` / `process_split` / `SPLIT_TO_OUTNAME` / `parse_args_sys`
**verbatim** from the causal extractor and re-implements only `main()`. Verified at runtime:
`S1.process_split is causal.process_split → True`, `S1.read_gt is causal.read_gt → True`.

Exactly **two** things differ from the banked causal arm:

1. **Attention mask.** `bidir_patch.apply_bidir_mask(model)` applied **post-`merge_and_unload`,
   pre-any-forward** — the ordering the recon §4.3 makes binding (a pre-merge bind would attach
   to a `PeftModel` whose `.model` is the wrapper, not the decoder). Identical to the F72 runner.
2. **The TEXT readout span**, and nothing else.

### 1.3 The exact pooling spans (the whole point of S1)

| stream | deployed causal + F72 bidir | **S1** |
|---|---|---|
| **img** (`span="prefix"`) | `last_hidden[:end].mean(0)`, `end` = index of the **last** `<|im_start|>` → mean over the vision+instruction prefix | **unchanged — delegated to the frozen `_encode` function object itself**, not a copy |
| **text** (`span="response"`) | `start` = index of the **last** `<|im_start|>`; `last_hidden[start:].mean(0)` → mean over the trailing `<|im_start|>assistant\n` header, **3-4 format tokens**. An **EOS-class / last-token** readout. | `last_hidden.mean(0)` over **ALL non-padding positions, span `[0, seq_len)`** — every video-pad token + title + transcript + instruction + every chat-format token, **including** the trailing assistant header. **LLM2Vec mean pooling.** |

Measured composition of the S1 text span (recon §1.5): 768 video tokens (constant; 8 frames →
4 temporal groups × 192 merged) + median 162.5 text tokens = **median 930.5 positions, 82.5 % vision**.

Implementation note carried from the codex review: at bsz=1 unpadded extraction the attention
mask is all-ones, so S1 takes the `last_hidden.mean(dim=0)` branch — **the same reduction call
the frozen prefix readout uses**, giving numerical parity by construction (same kernel, same
bf16 accumulation) rather than by assumption. A masked-mean branch exists for the padded case
and is arithmetically identical at all-ones (verified on CPU).

Cache tags (all distinct — no banked cache can be clobbered):
`HateMM → Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-meanpool_HF`,
`MHC_zh → Qwen2.5-VL-7B-Instruct-LoRA-bidir-meanpool_HF`. Splits: **`train,val` only**.

### 1.4 Pre-GPU review and hardening

The fork + sbatch went through the project's codex gate before submission (model internals +
unattended run). Verdict: **GO, no blockers**, with three non-blocking nits, all closed:

- **bf16 precision over ~930 positions** → S1 now uses the identical `.mean(dim=0)` reduction as
  the frozen prefix readout (§1.3), removing the question rather than arguing about it.
- **`assert` disabled under `python -O`/`PYTHONOPTIMIZE`** → both the test-touch guard and the
  readout-installed guard are now explicit `raise RuntimeError`, and they run **first** in
  `main()`, before any filesystem or GPU side effect.
- **`S1_LIMIT=1_0` parses differently in bash vs python** (could yield a truncated cache without
  the protective `-smoke` suffix) → the sbatch now rejects any non-digit `S1_LIMIT`.

Two further hardenings applied on top: an out-tag guard refusing any tag without `meanpool`
(so a manual invocation cannot inherit the causal default and clobber a banked cache), and
HateMM ordered **first** in the sbatch so the primary dataset is banked even if the job aborts
mid-run.

---

## 2. KILL-SWITCH RESULTS

### 2.1 `KS-MNTP-0a` — installation belt ($0 CPU, pre-GPU) — **PASS**

`bidir_patch.bidir_self_test()`:

| measurement | value | bar | verdict |
|---|---|---|---|
| patched mask shape | `(1, 1, 6, 6)` | `[bsz,1,seq,seq]` | PASS |
| patched mask all-zero | `True` | all-zero additive | PASS |
| `d_causal(pos0, future perturbed)` | `0.000e+00` | `< 1e-05` (control genuinely causal) | PASS |
| `d_causal(last pos, sanity)` | `1.042e+01` | `> 1e-04` (perturbation is real) | PASS |
| `d_bidir(pos0, future perturbed)` | `6.387e-02` | `> 1e-04` (patch makes it non-causal) | PASS |

SDPA assert and the `[BIDIR] … is_causal=False on 28 decoder attention module(s)` runtime line
are re-checked inside the sbatch before the model loads; `set -e` aborts the job on failure.

**Additional $0 belts run before GPU** (all PASS):

| belt | result |
|---|---|
| verbatim-imported `process_split` dispatches through the S1 readout | `_encode is _encode_s1_meanpool → True` |
| frozen causal `_encode` captured **before** the swap (img path cannot recurse) | `True` |
| masked mean == full mean at all-ones mask; differs from trailing-tail mean | `True` / `True` |
| clobber guard: causal out-tag rejected | raises, **no directory created** |
| test-touch guard: `--splits train,val,test` rejected | raises |
| test-touch guard: `--splits "train, test "` (whitespace) rejected | raises |

### 2.1b Smoke run — job **13650**, `COMPLETED 00:01:36` (S1_LIMIT=4) — **PASS**

Smoke mode (`S1_LIMIT=4`, out-tags suffixed `-smoke`, 4 items/split/dataset). Runtime log
confirms, for **both** datasets: the KS-MNTP-0a self-test PASS block; `[BIDIR] mask-flip patch
installed on model.model; is_causal=False on **28 decoder attention module(s)**`; the
`[S1] text readout = LLM2Vec mean pool over ALL non-padding positions [0, seq_len)` line;
`splits='train,val'`, `limit=4`, `max_pixels=151200`, `num_frames=8`; `zero-vector videos=0`.

**The decisive smoke check** — S1's 4-item vectors against the banked F72 bidir cache, id-matched
(mean per-item cosine):

| dataset | split | **img** cos(S1, banked-bidir) | **text** cos(S1, banked-bidir) | text cos(S1, causal) |
|---|---|---|---|---|
| HateMM | train | **1.000000** | 0.4255 | 0.3817 |
| HateMM | dev_seen | **1.000000** | 0.5159 | 0.4528 |
| MHC_zh | train | **1.000000** | 0.5885 | 0.4545 |
| MHC_zh | dev_seen | **1.000000** | 0.5641 | 0.4477 |

This is the S1 analogue of `KS-MNTP-0b`, and it is exactly the two-sided result the design needs:

- **img is a verified EXACT null-op (cosine 1.000000, not merely ≥0.9999)** — the delegation to
  the frozen `_encode` function object works, and the model/mask/merge path is bit-identical to
  the F72 runner's. Any S1-vs-F72 difference is therefore attributable to the text readout alone.
- **text genuinely changed** (cos 0.43-0.59, far from 1.0) — the readout swap is not a silent
  no-op, so the arm cannot be an accidental duplicate of F72.

Smoke caches were deleted after this check.

### 2.1c Execution incident — CPU-head arm filter silently dropped (job 13652)

Job **13652** (`COMPLETED 00:05:23`) was submitted as `MNTP_ARMS=causal,bidir` but measured the
**causal arm only** (6 of the intended 12 cells; verified in `mntp_s1_cpuhead_OUT.json`).

**Cause:** `sbatch --export=ALL,MNTP_ARMS=causal,bidir`. `--export` takes a **comma-separated
variable list**, so sbatch read it as `ALL` + `MNTP_ARMS=causal` + a variable named `bidir`
(unset, ignored). The value was truncated at the comma. This is the same escaping class the
codex review had flagged for `S1_LIMIT` — which was itself safe, having no comma in its value
(job 13650's log confirms `limit=4` and the `-smoke` suffix applied correctly).

**Impact: none on any measurement.** The 6 causal cells that ran are valid and are reused as-is;
the failure was one of coverage, not correctness. **Fix:** set the variable in the submitting
shell and omit the explicit `--export` flag, relying on sbatch's default `--export=ALL`
propagation. The bidir arm was resubmitted as job **13655** under the fixed form.

**CPU-trained causal floor from 13652** (dev, final epoch 29, 3 seeds — the KS-MNTP-2 floor):

| dataset | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| HateMM | 0.8318 | 0.8411 | 0.8131 | **0.8287** |
| MHC_zh | 0.8462 | 0.8333 | 0.8462 | **0.8419** |

### 2.1d Full extraction — job **13654**, `COMPLETED 00:49:16` — cache sanity **PASS**

| dataset | split | N | expected | shapes | id order == causal | zero-vec rows | md5 |
|---|---|---|---|---|---|---|---|
| HateMM | train | 744 | 744 | (744, 3584) ×2 | ✔ | 1 | `f4a294154b78b12dbcf005fe2c428f2f` |
| HateMM | dev_seen | 107 | 107 | (107, 3584) ×2 | ✔ | 0 | `5fb60ca68731bfcca192d2cc12075dee` |
| MHC_zh | train | 579 | 579 | (579, 3584) ×2 | ✔ | 0 | `a349878dcce812764c84583bef350503` |
| MHC_zh | dev_seen | 78 | 78 | (78, 3584) ×2 | ✔ | 0 | `27f81b75b6df0df46708b0e61e921790` |

Total **1508 rows = 851 HateMM + 657 MHC_zh**, exactly as expected for train+dev.
**No `test_seen_*bidir-meanpool*` file exists in either dataset directory** (checked explicitly).
The single zero-vector row is HateMM train idx 355, `hate_video_95`, and it is present
**identically in the causal, F72-bidir and S1 caches** — a pre-existing undecodable video, not
an S1 artifact.

### 2.2 `KS-MNTP-1` — raw-key dev screen — **STOP (do not continue)**

Bars are FROZEN in recon §5.2, quoted not recomputed. All values DEV, raw untrained key space,
deployed vote operator imported verbatim from the frozen `8bc009e6` screen. Format acc/mF1/roc.

**HateMM**

| stream | causal | bidir-lasttoken (F72) | **bidir-MEANPOOL (S1)** | Δ S1 vs bidir | Δ S1 vs causal |
|---|---|---|---|---|---|
| img | 0.7570 / 0.7491 / 0.8141 | 0.7664 / 0.7540 / 0.8127 | 0.7664 / 0.7540 / 0.8127 | **+0.0000** | +0.0093 |
| **text** | 0.8037 / 0.8003 / 0.8935 | 0.7570 / 0.7377 / 0.8368 | **0.7477 / 0.7318 / 0.8743** | **−0.0093** | **−0.0561** |
| concat | 0.8505 / 0.8489 / 0.9052 | 0.7944 / 0.7862 / 0.8674 | 0.7570 / 0.7405 / 0.8539 | **−0.0374** | −0.0935 |

**MHC_zh**

| stream | causal | bidir-lasttoken (F72) | **bidir-MEANPOOL (S1)** | Δ S1 vs bidir | Δ S1 vs causal |
|---|---|---|---|---|---|
| img | 0.7436 / 0.7057 / 0.8379 | 0.7564 / 0.7173 / 0.8414 | 0.7564 / 0.7173 / 0.8414 | **+0.0000** | +0.0128 |
| **text** | 0.8462 / 0.8353 / 0.9407 | 0.6282 / 0.5203 / 0.6886 | **0.7051 / 0.6578 / 0.8079** | **+0.0769** | **−0.1410** |
| concat | 0.8590 / 0.8519 / 0.9214 | 0.6410 / 0.5439 / 0.7400 | 0.7436 / 0.7168 / 0.8157 | **+0.1026** | −0.1154 |

**Gate arithmetic against the frozen bars (text stream):**

| dataset | S1 text acc | frozen bidir | frozen floor25 | frozen bar50 | recovery fraction | cell verdict |
|---|---|---|---|---|---|---|
| HateMM | **0.7477** | 0.7570 | 0.7687 | 0.7804 | **−0.1999** | below the crater — KILL-side |
| MHC_zh | **0.7051** | 0.6282 | 0.6827 | 0.7372 | **+0.3529** | PARTIAL (25-50 %) |

Applying the frozen rule verbatim: **≥50 % on ≥1 dataset ⇒ CONTINUE** — not met (max 35.3 %).
**<25 % on BOTH ⇒ KILL** — not met (ZH is 35.3 %). Remainder: *"Between ⇒ partial; continue
only if the **sign** is consistent across both datasets."* The signs are **opposite**
(HateMM −0.1999, ZH +0.3529). **⇒ DO NOT CONTINUE.**

**Belts on this screen.** img null-op belt (S1 img vs banked F72 bidir img, mean per-item
cosine over decodable rows): HateMM train **1.000000**, dev **1.000000**; MHC_zh train
**1.000000**, dev **1.000000** — bar 0.9999, **PASS**. The img readout is a verified *exact*
null-op, so every S1-vs-F72 difference is attributable to the text readout alone.

> **Belt-design erratum, recorded because the first run reported it wrong.** The belt initially
> read HateMM train **0.998656 → FAIL**. That was an artifact of my own belt, not of the data:
> `hate_video_95` is all-zeros in *both* arms (the arms agree perfectly) but `cosine(0, 0)`
> returns 0.0, and one such row among 744 drags the mean to exactly 0.998656. Median per-item
> cosine was 1.00000012 and **exactly one** row fell below the bar. The belt now excludes rows
> whose norm is zero in either arm, and reports 1.000000. Both numbers are stated here.

Feature drift, mean per-item cosine, id-matched, decodable rows:

| dataset | | train img | train text | dev img | dev text |
|---|---|---|---|---|---|
| HateMM | S1 vs causal | 0.7913 | 0.4365 | 0.7946 | 0.4540 |
| | S1 vs F72 bidir | **1.0000** | 0.4821 | **1.0000** | 0.5028 |
| MHC_zh | S1 vs causal | 0.6786 | 0.4352 | 0.6762 | 0.4292 |
| | S1 vs F72 bidir | **1.0000** | 0.6004 | **1.0000** | 0.5948 |

**The load-bearing new measurement — STREAM COLLAPSE.** Mean per-item cosine between the two
streams *within* each arm:

| arm | HateMM train | HateMM dev | MHC_zh train | MHC_zh dev |
|---|---|---|---|---|
| causal | 0.3523 | 0.3499 | 0.3105 | 0.3027 |
| bidir-lasttoken (F72) | 0.4314 | 0.4511 | 0.4977 | 0.4898 |
| **bidir-MEANPOOL (S1)** | **0.9273** | **0.9404** | **0.9320** | **0.9316** |

Under S1 the "text" vector is ~0.93 cosine-identical to the img vector. **The two streams have
very nearly merged into one.** This is mechanically forced and should have been foreseen: the S1
text readout means over all ~930 positions, of which **768 (82.5 %) are vision tokens from the
very same 8 frames the img stream pools** (recon §1.5). The pooled "text" vector is therefore
dominated by shared visual content.

### 2.3 `KS-MNTP-2` — CPU head dev screen — **NOT RUN for the S1 arm (gate said stop)**

KS-MNTP-1 returned STOP, and the recon makes KS-MNTP-2 conditional on KS-MNTP-1 continuing.
Running it anyway after a stop verdict would be an unfunded extra look on dev — a forking path
the family's multiplicity discipline (§5.3) exists to prevent. **The 6 meanpool cells were
deliberately not run.** This is a protocol decision, not an oversight.

The **control** half of the screen was already measured and is banked for any future arm at zero
additional cost (CPU-only jobs **13652** causal + **13655** bidir; dev, final epoch 29, 3 seeds,
CPU-to-CPU):

| dataset | arm | seed 0 | seed 1 | seed 2 | mean | Δ vs CPU causal floor |
|---|---|---|---|---|---|---|
| HateMM | causal (floor) | 0.8318 | 0.8411 | 0.8131 | **0.8287** | — |
| HateMM | bidir-lasttoken | 0.7850 | 0.7944 | 0.7757 | **0.7850** | **−0.0436** (per-seed −0.0467/−0.0467/−0.0374) |
| MHC_zh | causal (floor) | 0.8462 | 0.8333 | 0.8462 | **0.8419** | — |
| MHC_zh | bidir-lasttoken | 0.7179 | 0.6667 | 0.6923 | **0.6923** | **−0.1496** (per-seed −0.1282/−0.1667/−0.1538) |

This independently reproduces the F72 crater **in dev/CPU space** (F72's own −0.121/−0.141 were
**test**-side), which is a useful same-protocol reference the campaign did not previously have.

---

## 3. H1 vs H2 — HONEST READING

**Headline: S1 refutes the *naive* form of H2 and does NOT vindicate H1. The decisive finding is
that S1's manipulation was confounded, so the underlying H2 remains untested.**

**1. The naive readout fix does not work.** "Swap the EOS-class readout for LLM2Vec mean pooling
and the crater closes" is false as stated. HateMM went *further down* (text 0.7477, below even
the F72 crater 0.7570); ZH came up only 35 % of the way. No dataset reached the 50 % bar, and
the two datasets moved in opposite directions.

**2. ZH's apparent partial recovery is not readout repair — it is stream substitution.** The
collapse measurement explains both datasets with one mechanism. Under S1 the text vector is a
~0.93-cosine near-copy of the img vector, so each dataset's "text" row simply migrates toward
its *img* row:

| dataset | causal: text vs img | S1: text vs img | img row (bidir) |
|---|---|---|---|
| HateMM | 0.8037 vs 0.7570 → text is **+0.0467 better** | 0.7477 vs 0.7664 → text is **−0.0187 worse** | 0.7664 |
| MHC_zh | 0.8462 vs 0.7436 → text is **+0.1026 better** | 0.7051 vs 0.7564 → text is **−0.0513 worse** | 0.7564 |

Under causal, the text stream is clearly the *stronger* of the two on both datasets. Under S1 it
is worse than img and ~93 % correlated with it. Where the bidir text stream was catastrophically
broken (ZH, 0.6282) replacing it with a copy of the healthy img stream *looks* like recovery;
where it still carried real signal (HateMM, 0.7570) the same substitution *loses* information.
That single mechanism produces the opposite signs, which is exactly why the sign-consistency
clause in the frozen gate fired. **The gain is not a repaired text representation; it is the
text channel being replaced by the image channel.**

**3. Therefore S1 did not cleanly test H2.** "LLM2Vec mean pooling" does not transplant naively
into a multimodal sequence. LLM2Vec mean-pools over sequences that are **pure text**; ours are
**82.5 % vision**. Pooling over *all* positions is a materially different operator, and it
destroys the text channel's independence. The clean test of H2 — mean-pool over the **text
positions only**, excluding the 768 video-pad tokens — is the true multimodal analogue of the
LLM2Vec recipe and was **not** run here. **This is a design limitation of S1 as specified, and
it is the single most important thing this record hands back.** (Flagged, not acted on:
escalation is the main loop's call, not this agent's.)

**4. H1's strong form remains refuted, independently of all the above.** The mean-pooled img
stream is unharmed by the mask flip and slightly better (HateMM +0.0093, ZH +0.0128) — and S1
reproduces the F72 img vectors at cosine **1.000000**, so this is now confirmed on two
independently extracted caches. A mask flip that leaves one stream intact on both datasets is
not a global "the weights are broken" event. **The crater remains specifically a text-stream,
readout-adjacent phenomenon** — which is what makes the untested text-positions-only variant the
live question rather than a closed one.

**5. What this does NOT license.** No claim that bidirectional attention is viable; no claim that
the readout hypothesis is dead; no goal-clause progress whatsoever. Every S1 number is *below*
its causal floor on both datasets. Per `KS-MNTP-3`, even full recovery would have been a
mechanism result, not a goal result — and S1 did not recover.

---

## 4. COST

| job | what | elapsed | GPU |
|---|---|---|---|
| 13650 | S1 smoke (`S1_LIMIT=4`) | 00:01:36 | 1× A100 |
| 13654 | S1 full extraction, both datasets, train+dev | 00:49:16 | 1× A100 |
| 13652 | CPU head, causal arm (6 cells) | 00:05:23 | **none** (no `--gres`) |
| 13655 | CPU head, bidir arm (6 cells) | 00:05:11 | **none** (no `--gres`) |

**GPU spent: 00:50:52 = 0.848 GPU-h**, against the recon's S1 budget of ~1.0 GPU-h — **under
budget**, helped by extracting train+dev only and skipping test entirely. CPU screens: $0.
No download, no Modal, no training, no corpus ruling consumed.

---

## 5. REQUIRED STATEMENTS

**ZERO test-touch, in the sense that matters: no held-out test data entered any measurement,
gate, or selection decision at any point in S1.** Enforced by four independent belts: (i) the S1
extractor raises on any `--splits` containing `test`, before any side effect; (ii) the sbatch
hard-codes `--splits train,val`; (iii) the S1 caches have no `test_seen` file at all; (iv) the
KS-MNTP-2 harness replaces the loader with a dev-only variant returning `(train, dev, dev)` and
wraps `load_feats_split` in a guard that raises on any path containing `test_seen` — applied
**uniformly to every arm**, so no arm reads test and the harness is identical across arms.

> **Correction (S1b review, external gate).** An earlier draft of this section claimed *"no
> held-out test file was opened, read, or produced."* **That literal claim was too strong and is
> withdrawn.** Job 13654's final step ran `scripts/b2_push.sh data/CLIP_Embedding/<DS>` for the
> backup, which is a recursive `rclone copy` over the whole embedding directory — and that
> directory contains pre-existing `test_seen_*.pt` caches belonging to *other* arms (59 files
> across both datasets). Those files were therefore **enumerated and copied** to B2.
>
> **What this does and does not mean.** It is a byte-level file copy performed after all
> extraction finished. No test label, feature, or metric entered the S1 pipeline, the KS-MNTP-1
> screen, the CPU head screen, or any bar — every belt above still holds, and every reported
> number remains dev-only. So the *scientific* zero-test-touch property is intact. But the
> sentence as originally written asserted something stronger than what happened, and this
> project's provenance discipline does not permit leaving that standing.
>
> **Fixed forward:** the S1b sbatch pushes **only its own two cache files by name**, never the
> directory. The S1 sbatch (sha `20a020cb…`) is deliberately left byte-unchanged because it is
> the provenance record of what job 13654 actually executed.

No user-gated resource consumed: **no download, no corpus ruling, no `llm2vec` install, no
Modal**. No `src/` file edited. No banked cache overwritten. No `state/`, prereg, or frozen
artifact mutated. Committed on `main`, **not pushed**.

---

## 6. ARTIFACT SHAS (final, at commit time)

**Frozen artifacts — VERIFIED BYTE-UNCHANGED by this cell:**

| file | sha256 |
|---|---|
| `src/utils/bidir_patch.py` | `36cedbac365b2b13c945adbe3437efdc61d8be15ecc85878eb9614225abe367b` |
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | `75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399` |
| `src/utils/generate_VideoMLLM_embedding_bidir_HF.py` | `03f39e09c417bbea291f3c06b787f5220693568cd613705693df1c2bf23e020d` |
| `scripts/analysis/mntp_rawkey_devscreen.py` | `8bc009e68833d8bad3aecb531c7c8b9879e05a2e00430465e0b2b4f05f9dede0` |

**New S1 artifacts:**

| file | sha256 |
|---|---|
| `src/utils/generate_VideoMLLM_embedding_bidir_meanpool_HF.py` | `8f2de58efc696f5146419cd3ea7c82ce5bdbdb5e7790174fc32487bf7c7f7200` |
| `scripts/slurm/gen_embed_mllm_bidir_meanpool.sbatch` | `20a020cba578b33c0a3343941afb4d754dcf370b09191dc6ecac61d67ba0aaa4` |
| `scripts/analysis/mntp_s1_devscreen.py` | `d4a4a9f3d79f174f5b08cd4e699166c63ffd47d758d2a8b9337b34401dfe5cc2` |
| `scripts/analysis/mntp_s1_cpuhead.py` | `b42f6888f21ce1078046afe84e503b415305c4f734eeef2631e03ef5e1ad3a82` | *(sha updated: the S1b textpool arm was added to its ARMS map after the S1 commit; the 13652/13655 causal+bidir results are unaffected — those cells were already measured and are reloaded from the OUT json, not recomputed)*
| `scripts/slurm/mntp_s1_cpuhead.sbatch` | `e71d5a95d204bdaccb11f17115fb2cb76b0e06c8f7559b69633c8ff0b160715b` |

**S1b artifacts (amendment §6b/§6c):**

| file | sha256 |
|---|---|
| `src/utils/generate_VideoMLLM_embedding_bidir_textpool_HF.py` | `df3b8ee4ca501938d612ebb97576e26e015271de33843b0d943d1e8965924405` |
| `scripts/slurm/gen_embed_mllm_bidir_textpool.sbatch` | `e4e9701f7857cc63e99dd58166a359d3f8cc17ce90692d372d9adb14e47cd6b4` |
| `scripts/analysis/mntp_s1b_devscreen.py` | `2ebd61ace902ad109dbee35283ac870ecf9dd504bd0e58933ac0a49aa6bd4380` |

**S2a artifacts (amendment §6d/§6e):**

| file | sha256 |
|---|---|
| `src/utils/generate_VideoMLLM_embedding_bidir_mntp_HF.py` | `c15bb95c3e6e36e2a50063cddd39a00696e9814806ee53096fa4b0dbc0af35e2` |
| `scripts/slurm/gen_embed_mllm_bidir_mntp.sbatch` | `61b332fbfdd476ecaa2681c0b5c47510eb4a230ab45acb596fbeb4de695ae3d1` |
| `scripts/slurm/gen_embed_nullop2merge.sbatch` | `d0c871fa7f15970724cae1ff32c307428f23253f6dc1c412fe1d37b6cef06f8c` |
| `scripts/analysis/mntp_s2a_devscreen.py` | `4d6cfb1a223cceb4b6c9e368fcc4db8bbb086f0ead29bb7f04615d7ac4dbf5b8` |
| McGill MNTP adapter (`adapter_model.safetensors`, downloaded) | `5e3fb47d2448ce3302261019fb40b9ba7b335740b23a45df2da9bf3a1e158feb` |

Primary outputs: `scripts/analysis/mntp_s1_devscreen_OUT.json`, `mntp_s1b_devscreen_OUT.json`, `mntp_s2a_devscreen_OUT.json`,
`scripts/analysis/mntp_s1_cpuhead_OUT.json`. S1 caches:
`data/CLIP_Embedding/{HateMM,MHC_zh}/{train,dev_seen}_*-bidir-meanpool_HF.pt` (md5s in §2.1d).

---

## 6b. AMENDMENT — S1b: TEXT-POSITIONS-ONLY MEAN POOL (declared BEFORE building)

**Status at declaration:** funded by main-loop ruling after S1 reported. Still S1-family
(readout hypothesis, **zero training**, no adapter, no corpus ruling, no download). S2a transplant
stays parked. **This section was written and committed BEFORE the S1b fork was built**, per the
project's freeze discipline; nothing below was chosen after seeing an S1b number.

### 6b.1 Why S1b exists

S1 refuted the *naive* form of H2 but did not test the real one: it pooled **all** ~900 positions,
of which ~80 % are vision tokens from the same 8 frames the img stream pools, so the "text" vector
collapsed onto the img vector (cos 0.93) and stopped being a text readout. LLM2Vec mean-pools
sequences that are **pure text**. The faithful multimodal analogue pools **text positions only**.
That is S1b, and it is the clean test of H2.

### 6b.2 The manipulated readout — exact rule

Text positions are selected by **token id**, not by span arithmetic:

```
keep[i]  ==  attention_mask[i] == 1              (non-padding; all-ones at bsz=1 unpadded)
         AND input_ids[i] != <|video_pad|>       (id 151656 — vision CONTENT positions)
         AND input_ids[i] != <|image_pad|>       (id 151655 — defensive; we pass videos, not images)
text_feats = L2( mean_{i : keep[i]} last_hidden[i] )
```

Verified on CPU with the deployed processor (`max_pixels=151200`, `num_frames=8`) before writing
this section:

| token | id | role | disposition |
|---|---|---|---|
| `<|video_pad|>` | 151656 | the vision content positions | **EXCLUDED** |
| `<|image_pad|>` | 151655 | image content (unused on this path) | **EXCLUDED** (defensive) |
| `<|vision_start|>` / `<|vision_end|>` | 151652 / 151653 | **one each**, structural markers, not content | **KEPT** (chat-format text) |
| `<|im_start|>` / `<|im_end|>` | 151644 / 151645 | 3 / 2 per sequence, chat format | **KEPT** |
| all instruction / title / transcript / system wordpieces | — | the actual text | **KEPT** |

Decoding the kept positions reproduces the full prompt intact — system turn, user turn, the
vision markers with the video content elided, the analytic instruction, title, transcript, and the
trailing assistant header. That is exactly the object LLM2Vec mean-pools.

**Example span decomposition** (deployed processor, 8 frames at `max_pixels=151200`, 60 real train
prompts per dataset):

| dataset | seq median | vision positions | text positions (median / min / max) | vision share |
|---|---|---|---|---|
| HateMM | 908.5 | 720, constant | 188.5 / 70 / 2994 | 79.3 % |
| MHC_zh | 847.5 | 720, constant | 127.5 / 83 / 328 | 85.0 % |

*Provenance note:* the recon §1.5 reports 768 constant vision tokens; measured here at the deployed
`max_pixels` the count is **720**. The count is a function of the frame geometry the sampler hands
the processor (both are "4 temporal groups × N merged tokens"), so it is **not** a fixed constant
across differently-shaped source videos. **This does not affect the rule, which masks by token id
and never by count.** The S1b extractor will log the *actual* per-item decomposition over the real
run, and those measured numbers — not these dummy-frame estimates — go in the S1b results table.

The **img readout is unchanged**: still the frozen causal prefix-mean, still delegated to the
frozen `_encode` function object. Expected to remain an exact null-op; the belt is retained.

### 6b.3 Gates — bars do NOT move

1. **`KS-MNTP-1`**, identical frozen bars: HateMM text ≥ **0.7804** (50 %), floor **0.7687** (25 %);
   MHC_zh ≥ **0.7372**, floor **0.6827**; and the same sign-consistency clause for the partial band.
2. **NEW mandatory belt — stream-collapse.** Mean per-item cos(text_S1b, img) must stay in the
   causal regime, **< 0.60** (causal is 0.3105-0.3523; S1 was 0.9273-0.9320). **If it collapses
   toward img again the arm SELF-REFUTES regardless of accuracy** — record and stop. This bar is
   declared here, before any S1b number exists.
3. Carried belts: `KS-MNTP-0a`; smoke with 4-item cosine checks against the banked arms (img must
   be an exact null-op, text must differ); clobber and test-touch guards; **the zero-norm-row
   exclusion fix carries over** to every cosine in this arm.
4. Pre-GPU external review pass on the fork diff (model-internals code).
5. **train+dev only, both datasets, HateMM first.** ZERO test-touch.
6. If `KS-MNTP-1` CONTINUEs → `KS-MNTP-2` CPU screen against the **banked** 13652/13655 floors
   (HateMM causal 0.8287 / bidir 0.7850; ZH causal 0.8419 / bidir 0.6923), CPU-to-CPU.

Cache tags: `HateMM → Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-textpool_HF`,
`MHC_zh → Qwen2.5-VL-7B-Instruct-LoRA-bidir-textpool_HF`. Budget ~1.0 GPU-h.

### 6b.4 What S1b can and cannot conclude — declared in advance

- **CONTINUE** (≥50 % recovery on ≥1 dataset **and** collapse belt < 0.60) ⇒ H2 is live: the crater
  is substantially a readout mismatch, and the S1 result was an artifact of a bad pooling choice.
- **Collapse belt ≥ 0.60** ⇒ arm self-refutes; the multimodal readout cannot be de-confounded this
  way and the readout route is exhausted at zero training.
- **Partial with inconsistent sign, or <25 % on both, with the collapse belt passing** ⇒ **H2 is
  refuted on its strongest available test.** The readout is not the story, H1's weaker form stands,
  and the campaign routes to S2a (transplant) or stops.
- Under **no** outcome does S1b advance the goal clause: recovery to the causal floor is a
  mechanism result (`KS-MNTP-3`), and every arm so far sits below that floor.

---

## 6c. S1b EXECUTION

### 6c.1 Pre-GPU external review — **NO-GO, then GO after fixes**

The fork + sbatch went to the external gate before submission. First pass returned **NO-GO** with
two blockers. Both were legitimate and are recorded because one of them invalidates a sentence in
this record's own §5.

- **BLOCKER — backup exposed held-out test caches.** The sbatch's final step ran
  `b2_push.sh data/CLIP_Embedding/<DS>`, which is a recursive `rclone copy` over the *whole*
  embedding directory — and that directory holds pre-existing `test_seen_*.pt` caches belonging to
  other arms (59 files across both datasets). **Job 13654 (S1) already did this.** Copying bytes
  leaks no test information into any gate, so every reported number stands, but the literal
  zero-test-touch sentence in §5 was too strong; it is **corrected there**, not quietly dropped.
  **Fix:** S1b pushes only its own two cache files **by name**, never the directory.
- **BLOCKER — a third, undeclared behavioural divergence.** `parse_args_sys` is inherited
  verbatim, so it also exposes the MOKA cell's `--no_merge` / `--moka`. The frozen causal
  extractor honours them; this fork always takes the merge path and would have **silently ignored
  them**. The arm declared *exactly two* differences from the control, so a reachable third is a
  spec violation even though the sbatch never sets those flags. **Fix:** `main()` now raises on
  either flag, before any side effect.
- **NIT — span-stats filename collision** (a later smoke run could overwrite full-run stats):
  filename now carries the out-tag. **NIT — oversized `S1_LIMIT`**: an all-digit value beyond
  bash's signed-int range passes the digits-only regex but makes `-gt` fail *inside an `if`*, so
  `set -e` does not fire and the run would take the FULL out-tag while python truncated the split
  — a partial cache wearing a full cache's name. Length cap added. **NIT — `b2_push.sh` in file
  mode** treats arg 2 as the full destination *including* filename, so `embeddings/$DS` would have
  landed files in `embeddings/`; basename now appended.

Re-review after fixes: **GO**, all four cleared. The reviewer also confirmed the §6b.3 four-item
smoke cosine belt is acceptable as an **operator step** between the smoke and full submissions
(the S1 pattern) rather than automated inside the sbatch, provided the full job is not submitted
until it passes and the result is recorded. It was run and recorded (§6c.3).

### 6c.2 Measured span decomposition — and a correction to the recon

The S1b extractor logs the **real** per-item decomposition (`SPANSTATS_*.json`), so the record
carries measured numbers rather than dummy-frame estimates. From the smoke (n=4/split):

| dataset | split | seq median | vision positions (median, min-max) | **constant?** | text positions (median) | vision share |
|---|---|---|---|---|---|---|
| HateMM | train | 1381.0 | 700.0 (676-720) | **NO** | 681.0 | 50.7 % |
| HateMM | dev | 1076.0 | 720.0 (720-768) | **NO** | 332.0 | 66.9 % |
| MHC_zh | train | 820.0 | 720.0 (704-720) | **NO** | 105.5 | 87.8 % |
| MHC_zh | dev | 829.0 | 702.0 (684-720) | **NO** | 127.0 | 84.7 % |

**This settles the §6b.2 provenance note empirically.** The recon §1.5 reports "video tokens: 768,
**constant**". On real videos the count is **not** constant — it ranges 676-768 across items,
because the sampler hands the processor frames of differing geometry. **Any implementation that
had assumed a fixed 768-token vision prefix and sliced by position would have silently mixed
vision and text tokens on most items.** S1b masks by token id and is unaffected; this is recorded
as the concrete justification for that choice.

### 6c.3 Smoke — job **13656**, `COMPLETED 00:01:42` — belts **PASS**, with one early warning

Runtime log confirms for both datasets: KS-MNTP-0a PASS; `is_causal=False on 28 decoder attention
module(s)`; `[S1b] vision-content token ids: video_pad=151656 image_pad=151655 (EXCLUDED …);
vision_start/vision_end and chat-format tokens are KEPT`; `splits=train,val`; `limit=4`.

Mandatory 4-item cosine belt (mean per-item cosine, decodable rows, id-matched):

| dataset | split | **img** vs banked bidir | text vs bidir | text vs S1 meanpool | text vs causal |
|---|---|---|---|---|---|
| HateMM | train | **1.000000** | 0.4515 | 0.9148 | 0.3273 |
| HateMM | dev | **1.000000** | 0.5638 | 0.9052 | 0.4460 |
| MHC_zh | train | **1.000000** | 0.7185 | 0.8667 | 0.3861 |
| MHC_zh | dev | **1.000000** | 0.6786 | 0.8860 | 0.4095 |

Both smoke gates pass: img is again an **exact** null-op (1.000000), so every S1b-vs-F72
difference is attributable to the text readout alone; and text differs from **both** the F72 arm
and the S1 arm, so S1b is neither a duplicate of the control nor of the previous arm.

> **EARLY WARNING, recorded before the full run was submitted so it cannot be read as
> hindsight.** The within-arm collapse statistic on the 4-item smoke was
> **0.6889 / 0.7699 / 0.7482 / 0.7815** — **above the 0.60 bar declared in §6b.3 on all four
> cells.** The full run was submitted anyway, deliberately: the collapse belt is declared as a
> gate on *the arm's measurement*, and n=4 is not that measurement. Killing the campaign's last
> live gate on four items would be the one irreversible error available at this step, and
> ~0.85 GPU-h is cheap against it. **If the full-dev belt confirms the smoke, the arm
> self-refutes as declared.**

### 6c.4 Full extraction — job **13657**, `COMPLETED 00:48:54` — cache sanity **PASS**

1508 rows (744/107 HateMM, 579/78 MHC_zh), shapes (N, 3584) ×2, **id order identical to causal**
on every split, **no `test_seen_*textpool*` file exists**. md5: HateMM train
`261d13656d1493123bf94fa1686f41aa`, dev `16c3a4905601ee46ac5990cdc8922b67`; MHC_zh train
`1ce3afc5541a871b1942cbf419fc4aff`, dev `2e7082e109bc51b2f0e9df11cb17857b`.

**Measured span decomposition over the FULL run** (`SPANSTATS_*.json`; HateMM train n=743 because
the undecodable `hate_video_95` takes the zero-guard path and never reaches the readout):

| dataset | split | n | seq median | vision median (min-max) | **constant?** | text median | vision share |
|---|---|---|---|---|---|---|---|
| HateMM | train | 743 | 963.0 | 720.0 (**264-768**) | **NO** | 240.0 | 74.8 % |
| HateMM | dev | 107 | 885.0 | 720.0 (**364-768**) | **NO** | 168.0 | 81.4 % |
| MHC_zh | train | 579 | 867.0 | 720.0 (**120-768**) | **NO** | 145.0 | 83.0 % |
| MHC_zh | dev | 78 | 875.0 | 720.0 (**676-768**) | **NO** | 146.0 | 82.3 % |

The vision-token count ranges **120-768** on real data. The recon's "768, constant" is wrong on
this corpus, and the deviation is large, not marginal. **A position-sliced implementation would
have silently mixed vision and text tokens on a large fraction of items.** The id-based mask is
what makes S1b correct.

### 6c.5 `KS-MNTP-1` + collapse belt — **STOP: the arm SELF-REFUTES as declared**

All values DEV, raw untrained key space, frozen vote operator. Format acc/mF1/roc.

**HateMM**

| stream | causal | bidir-lasttoken | S1 meanpool | **S1b textpool** | S1b vs bidir | vs causal | vs S1 |
|---|---|---|---|---|---|---|---|
| img | 0.7570/0.7491/0.8141 | 0.7664/0.7540/0.8127 | 0.7664/0.7540/0.8127 | **0.7664/0.7540/0.8127** | +0.0000 | +0.0093 | +0.0000 |
| **text** | 0.8037/0.8003/0.8935 | 0.7570/0.7377/0.8368 | 0.7477/0.7318/0.8743 | **0.7664/0.7540/0.8390** | +0.0093 | **−0.0374** | +0.0187 |
| concat | 0.8505/0.8489/0.9052 | 0.7944/0.7862/0.8674 | 0.7570/0.7405/0.8539 | **0.8037/0.7914/0.8735** | +0.0093 | **−0.0467** | +0.0467 |

**MHC_zh**

| stream | causal | bidir-lasttoken | S1 meanpool | **S1b textpool** | S1b vs bidir | vs causal | vs S1 |
|---|---|---|---|---|---|---|---|
| img | 0.7436/0.7057/0.8379 | 0.7564/0.7173/0.8414 | 0.7564/0.7173/0.8414 | **0.7564/0.7173/0.8414** | +0.0000 | +0.0128 | +0.0000 |
| **text** | 0.8462/0.8353/0.9407 | 0.6282/0.5203/0.6886 | 0.7051/0.6578/0.8079 | **0.6923/0.5966/0.7364** | +0.0641 | **−0.1538** | −0.0128 |
| concat | 0.8590/0.8519/0.9214 | 0.6410/0.5439/0.7400 | 0.7436/0.7168/0.8157 | **0.7051/0.6408/0.7600** | +0.0641 | **−0.1538** | −0.0385 |

**Belt 1 — COLLAPSE (declared §6b.3 before the arm was built), bar < 0.60. Result: FAIL.**

| arm | HateMM train | HateMM dev | MHC_zh train | MHC_zh dev |
|---|---|---|---|---|
| causal | 0.3523 | 0.3499 | 0.3105 | 0.3027 |
| bidir-lasttoken | 0.4314 | 0.4511 | 0.4977 | 0.4898 |
| S1 meanpool | 0.9273 | 0.9404 | 0.9320 | 0.9316 |
| **S1b textpool** | **0.7566** | **0.7624** | **0.7565** | **0.7538** |

Worst cell **0.7624** (HateMM) and **0.7565** (ZH), both **≫ 0.60**. Removing the vision positions
cut collapse from ~0.93 to ~0.76 but **did not restore the causal regime (0.31-0.35)**. Per the
pre-declared rule, **the arm self-refutes regardless of accuracy.** The full-dev result confirms
the 4-item smoke warning (0.6889-0.7815) rather than overturning it.

**Belt 2 — img null-op: PASS**, 1.000000 on all four cells. Every S1b-vs-F72 difference is the
text readout alone.

**KS-MNTP-1 against the frozen bars:**

| dataset | S1b text acc | frozen bidir | floor25 | bar50 | recovery | cell verdict |
|---|---|---|---|---|---|---|
| HateMM | 0.7664 | 0.7570 | 0.7687 | 0.7804 | **+0.2003** | KILL-side (<25 %) |
| MHC_zh | 0.6923 | 0.6282 | 0.6827 | 0.7372 | **+0.2941** | PARTIAL (25-50 %) |

No dataset reached 50 %. **Note that the accuracy gate alone would have said CONTINUE this time**
— unlike S1, both signs are positive (+0.2003, +0.2941), so the sign-consistency clause is
satisfied. **The collapse belt is the binding constraint, and it is why it was declared in
advance.** Without it, S1b's uniformly positive movement over F72 (+0.0093 HateMM, +0.0641 ZH on
both text and concat) would have been read as partial readout repair and escalated.

**The smoking gun.** On HateMM the S1b **text row is numerically identical to the S1b img row** in
acc and macro-F1 — 0.7664/0.7540 for both, differing only in roc (0.8390 vs 0.8127). The text
channel is returning the image channel's answers. That is substitution, not repair.

**`KS-MNTP-2` NOT RUN for S1b.** The collapse belt is an overriding stop condition ("record and
stop"), declared before the arm existed. Running the head screen after a self-refutation would be
exactly the unfunded extra look the belt exists to prevent.

### 6c.6 H1 vs H2 — UPDATED READING (S1b closes the readout route)

**H2 is refuted on its strongest available test, and the mechanism is now identified.**

S1b is the faithful multimodal analogue of the LLM2Vec recipe — the experiment S1 failed to be.
It still does not recover: HateMM reaches 20 % of the crater, ZH 29 %, neither near the 50 % bar,
and both remain far below their causal floors (−0.0374 / −0.1538 on text).

**Why, mechanistically.** Under bidirectional attention **every text token attends to all ~720
vision tokens**, so the text tokens' hidden states are *themselves* saturated with visual content.
Excluding vision *positions* from the pool does not exclude vision *information* from the
representations being pooled. That is why collapse only falls 0.93 → 0.76 and never approaches the
causal 0.31-0.35. **The two streams converge under bidirectional attention regardless of which
positions are pooled, because the convergence happens inside the attention, not in the readout.**
Under causal attention the streams stay separated (0.31-0.35) precisely because the img readout
pools a prefix that cannot see the text and the text readout pools a tail that can.

**Consequence: the readout hypothesis is exhausted at zero training.** No choice of pooling span
can undo an information mixture that the topology itself creates. Three spans were tried —
EOS-class tail (F72), all positions (S1), text positions only (S1b) — spanning the full range of
what a readout can select, and all three land far below the causal floor while the streams
converge monotonically with how much of the sequence is pooled.

**H1's strong form remains refuted**, unchanged and now on three independent extractions: the img
stream is *better* under bidir (+0.0093 / +0.0128) and S1b reproduces it at cosine **1.000000**.
The weights are not broken. **But the earlier framing — "the crater is a text-stream, readout-
adjacent phenomenon" — is now sharpened: it is a text-stream phenomenon that a readout cannot
fix.** The remaining live hypothesis is that bidirectional attention needs *weight adaptation*
(the actual MNTP claim), which is S2a/S2b — not a readout change.

**What this does NOT license.** No goal-clause progress: every S1b number is below its causal
floor on both datasets. Per `KS-MNTP-3`, even full recovery would be a mechanism result. And the
S1b "gains" over F72 must not be quoted as method gains — they are the text channel being
partially replaced by the image channel.

### 6c.7 S1b cost

| job | what | elapsed | GPU |
|---|---|---|---|
| 13656 | S1b smoke (`S1_LIMIT=4`) | 00:01:42 | 1× A100 |
| 13657 | S1b full extraction, both datasets, train+dev | 00:48:54 | 1× A100 |

**S1b GPU: 00:50:36 = 0.843 GPU-h** (budget ~1.0). **S1 + S1b combined: 1.691 GPU-h.**
No training, no download, no corpus ruling, no Modal. KS-MNTP-2 not run (belt override), so the
banked 13652/13655 CPU floors remain unspent and reusable.

---

## 6d. AMENDMENT — S2a: PUBLISHED MNTP ADAPTER TRANSPLANT (declared BEFORE building)

**Status at declaration:** funded by main-loop ruling after S1b closed the readout route. Zero
training, zero corpus ruling. The download gate is the only user-gated resource, and it is spent
(§6d.2). **Written and committed BEFORE the S2a fork was built.**

### 6d.1 Why S2a, and what it tests

S1/S1b established that the F72 crater is **not** a readout problem — the vision/text mixture
happens inside bidirectional attention, before any pooling. The surviving hypothesis is the actual
MNTP claim: **bidirectional attention needs WEIGHT ADAPTATION**. S2a tests it at zero training cost
by transplanting the published McGill MNTP LoRA onto our merged VL trunk.

**This is a TRANSPLANT, and its main risk is stated up front:** the adapter is a low-rank delta
fitted to `Qwen2.5-7B-Instruct`'s weight point. Qwen2.5-VL's trunk was *initialised* from Qwen2.5
and then further trained during VL pretraining, so the base has drifted. The delta may carry the
generic "use bidirectional context" adaptation, or it may be noise at the new weight point. That
is the empirical question, and it costs ~1 GPU-h and no training to answer.

### 6d.2 Download gate — SPENT and verified

`McGill-NLP/LLM2Vec-Qwen25-7B-Instruct-mntp`, snapshot `2c7b5e8605db3caa2593d1ae8afb09c826c0d887`.
`adapter_model.safetensors` = **80,790,528 bytes**, sha256
`5e3fb47d2448ce3302261019fb40b9ba7b335740b23a45df2da9bf3a1e158feb`.

`adapter_config.json` vs recon §3.5 — **exact match on every field**: `base_model_name_or_path =
Qwen/Qwen2.5-7B-Instruct`, `r = 16`, `lora_alpha = 32`, `lora_dropout = 0.05`, `peft_type = LORA`,
`task_type = None`, `bias = none`, `target_modules = [gate_proj, q_proj, down_proj, v_proj, k_proj,
up_proj, o_proj]`.

**Shape verification against the LOCAL VL trunk (tensors, not just the config table):** 392
tensors = 28 layers × 7 modules × {lora_A, lora_B}; layers 0-27 complete; every shape correct for
`hidden=3584, kv=4 (512), intermediate=18944`:

| module | lora_A | lora_B | expected | |
|---|---|---|---|---|
| q_proj / o_proj | (16, 3584) | (3584, 16) | ✔ | |
| k_proj / v_proj | (16, 3584) | (512, 16) | ✔ | GQA kv width |
| gate_proj / up_proj | (16, 3584) | (18944, 16) | ✔ | |
| down_proj | (16, 18944) | (3584, 16) | ✔ | |

Same rank/alpha/target set as our own task LoRA (recon §2.3), and our SFT already freezes the
vision tower and projector, so "MNTP scoped to the LLM trunk" is the scope the deployed adapter
already has. **No new scoping argument is needed.**

### 6d.3 Arm construction — adapter order DECLARED

```
base Qwen2.5-VL-7B  →  PeftModel(task LoRA)  → merge_and_unload      [unchanged, all arms]
                    →  PeftModel(MNTP LoRA)  → merge_and_unload      [S2a ONLY]
                    →  bidir_patch.apply_bidir_mask(model)           [POST-merge, PRE-forward]
                    →  process_split(...)                            [imported VERBATIM]
```

**Declared order: the CHEAP order** (MNTP stacked on top of the merged task LoRA), per recon §4.2.
This preserves the deployed encoder identity and costs no SFT. It is acknowledged that the
scientifically faithful LLM2Vec order is MNTP-first-then-task-adaptation; per recon §4.2 that order
is funded **only** if the cheap order shows partial recovery. **Both orders are not run blind.**

**Readout = the DEPLOYED EOS-class tail, unchanged.** S1/S1b killed the readout route, so the
crater-comparable readout is the deployed one — which also keeps the frozen KS-MNTP-1 bars valid
without reinterpretation. The S2a runner is therefore the **F72 bidir fork plus one adapter
load/merge**, and nothing else.

Cache tags: `HateMM → Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-mntp_HF`,
`MHC_zh → Qwen2.5-VL-7B-Instruct-LoRA-bidir-mntp_HF`.

### 6d.4 Belts — declared in advance

1. **Weight-level transplant proof (new, strongest, $0-ish).** Snapshot a sample of decoder weight
   tensors immediately **before** the MNTP merge and re-read them **after**; they MUST change.
   This proves the transplant took effect *at the weights*, independent of any downstream metric,
   and it catches the PEFT-key-mismatch failure mode directly rather than inferring it. **Abort on
   no-change.**
2. **`KS-MNTP-0b` adapter-is-not-a-null-op (MANDATORY, F87 shape).** Mean per-item
   cos(S2a text, plain-bidir text) must be **< 0.9999** on all cells. **≥ 0.9999 ⇒ the adapter
   silently failed to load ⇒ ABORT before spending head time.**
3. **Img stream is READ, not asserted.** Unlike S1/S1b — whose img readout was delegated and had to
   be an exact null-op — the MNTP LoRA modifies the decoder, so **img MAY legitimately move.**
   If it does, that is **data, not failure**, and it is recorded. (An unchanged img stream would
   itself be suspicious here.)
4. **`KS-MNTP-0b` symmetric — same-path double-merge floor (recon §4.3).** The arm now has **TWO**
   merges, so it is strictly more exposed to the F87 bf16 merge-drift artifact (which manufactured
   a −0.0268 phantom "method effect"). Probe: identical runner, identical double-`merge_and_unload`,
   MNTP adapter replaced by a **freshly-initialised LoRA of identical shape** (PEFT inits `lora_B`
   to zeros, so it is a mathematical null-op — only the merge *path* differs), **bidir OFF**, vs the
   banked causal cache. Bar **≥ 0.9999**. **If it fails, the same-path double-merge causal floor
   becomes BINDING and must be extracted before any accuracy is read.** Run on a subset (≥60
   items/split) — merge drift is a numerical property of the path, not of particular items, so a
   subset is sufficient to detect it at the 1e-4 level; a failure escalates to the full floor.
5. Carried: `KS-MNTP-0a`; external review of the fork diff; smoke with 4-item cosine checks;
   clobber + test-touch guards; zero-norm-row exclusion in every cosine; train+dev only; HateMM
   first; **ZERO test-touch**.
6. **Frozen `KS-MNTP-1` bars, UNCHANGED**: HateMM text ≥ **0.7804** (50 %), floor **0.7687**;
   MHC_zh ≥ **0.7372**, floor **0.6827**; same sign-consistency clause.
7. **Collapse belt carried, bar < 0.60.** Under the deployed readout this should sit near the F72
   regime (~0.43-0.49). A rise toward 0.60+ is **reportable drift**, and ≥0.60 self-refutes as
   before.
8. If `KS-MNTP-1` CONTINUEs → `KS-MNTP-2` against the banked 13652/13655 CPU floors, **or** against
   the double-merge floor if belt 4 makes it binding.

### 6d.5 What S2a can and cannot conclude — declared in advance

- **Adapter is a null-op (belt 2 fails)** ⇒ load failure, not a result. Abort and fix.
- **CONTINUE** (≥50 % recovery on ≥1 dataset, collapse belt passing) ⇒ weight adaptation is the
  right axis; routes to S2b (train our own MNTP) with the corpus ruling now live.
- **<25 % on both, or partial with inconsistent sign** ⇒ the transplant is noise at the VL weight
  point. That does **not** by itself refute the MNTP hypothesis — it refutes *this* zero-training
  shortcut to it — and the honest routing is then S2b (needs the corpus ruling) or stop.
- Under **no** outcome does S2a advance the goal clause; recovery to the causal floor is a
  mechanism result (`KS-MNTP-3`), and every bidir arm so far sits below that floor.

---

## 6e. S2a EXECUTION AND RESULT

### 6e.1 Pre-GPU external review — **NO-GO ×2, then GO. It caught a silent-no-op blocker.**

**This review prevented a wasted GPU-hour and an uninterpretable arm.** First pass returned
**NO-GO** on a blocker I had not anticipated, which I then confirmed empirically on meta device
before fixing:

- **BLOCKER — the transplant would have loaded ZERO weights.** The McGill checkpoint ships keys
  `base_model.model.layers.N…`. Wrapping the **outer** `Qwen2_5_VLForConditionalGeneration` yields
  PEFT keys `base_model.model.**model**.layers.N…` — one `.model` deeper, because the outer
  model's decoder sits at `.model`. **PEFT loads non-strictly: it warns and continues**, leaving
  every `lora_B` at its zero init. The merge would then have been a no-op and S2a would have
  silently duplicated F72 while *looking* like it ran.
- **BLOCKER — undeclared vision-tower binding.** Suffix-matching the 7 target names on the outer
  model hits **292** modules, **96 of them in the vision tower**
  (`visual.blocks.N.mlp.{gate,up,down}_proj`) — a tower our SFT freezes. On `model.model` it hits
  exactly **196 = 28 × 7, zero vision**.

Both verified before fixing (meta-device enumeration). **Fix:** `_attach_second_adapter()` binds
the adapter to `model.model` (the bare `Qwen2_5_VLModel`; `model.visual` and `lm_head` are
siblings and untouched), merges, and assigns back.

Further fixes: belt 1 now requires **all 5** probes found **and all 5 changed** (a partial delta
must not pass); the parser handles `--mntp_dir=VALUE`, a trailing `--mntp_dir`, and a flag-shaped
value; only the **two preregistered flag/tag combinations** are accepted, so a real MNTP run can
never be banked under a `nullop` tag or vice versa; stale mean-pool provenance text removed from
the sbatch. Re-review: **GO**.

### 6e.2 Belts — all PASS

| belt | result |
|---|---|
| `KS-MNTP-0a` self-test | **PASS** (`is_causal=False` on 28 modules, both datasets) |
| CPU guard drills (combination, tag-coherence, parser, test-touch) | **PASS** (9/9) |
| **Adapter scope** | **196** LoRA-wrapped modules (= 28×7), **0** vision-tower, **196** with non-zero `lora_B` |
| **Belt 1 — weight-level transplant proof** | **5/5 probed tensors changed**, `max|Δ|` 1.96e-02 … 1.29e-01, rel 3.3 %-21 % |
| **`KS-MNTP-0b` adapter-not-null-op** | **PASS** — cos(S2a text, plain-bidir text) = **0.3639** (HateMM) / **0.3076** (ZH), bar < 0.9999 |
| **Belt 4 — same-path double-merge floor** (job 13659) | **PASS** |

The `196/196 non-zero lora_B` line is the one that proves the fix worked: before it, that count
would have read **0**. Combined with 5/5 weight probes changing, the transplant is certified to
have landed at the weights, not merely to have opened a file.

**Belt 4 detail (job 13659, `COMPLETED 00:08:55`, 60 items/split).** Fresh zero-init LoRA
(`lora_B` all-zero confirmed: 0/196 non-zero, 0/5 weight probes changed ⇒ a true mathematical
null-op), bidir OFF, double merge, vs the banked causal cache:

| dataset | split | img mean cos | text mean cos | worst per-item |
|---|---|---|---|---|
| HateMM | train / dev | 1.00000000 / 1.00000012 | 1.00000000 / 1.00000012 | 0.99999952 |
| MHC_zh | train / dev | 1.00000000 / 1.00000000 | 1.00000000 / 1.00000000 | 0.99999964 |

**All 8 cells ≥ 0.9999. The extra merge is numerically inert**, so the banked causal cache remains
a valid floor and **no same-path floor extraction was required** (saving ~0.85 GPU-h). This also
refines the F87 lesson: F87's drift came from a merge-vs-**unmerged**-path difference, not from
merge *count* — a second `merge_and_unload` of a zero delta costs nothing.

### 6e.3 Full extraction — job **13660**, `COMPLETED 00:49:20` — cache sanity **PASS**

1508 rows (744/107, 579/78), id order identical to causal on every split, **no `test_seen_*mntp*`
file**. md5: HateMM train `cebf42d6e9bb0772504dcae4c157d4c6`, dev
`77687475a690212d005a0a168764f07b`; MHC_zh train `fc2aeff17002965c1d8c7d31afdeee76`, dev
`ab30cb9d12d44bfe8d0db2aaa334c6dc`.

### 6e.4 `KS-MNTP-1` + collapse belt — **STOP (overdetermined)**

**HateMM**

| stream | causal | bidir-lasttoken | S1 meanpool | S1b textpool | **S2a MNTP** | vs bidir | vs causal |
|---|---|---|---|---|---|---|---|
| img | 0.7570/0.7491/0.8141 | 0.7664/0.7540/0.8127 | 0.7664 | 0.7664 | **0.7290/0.7170/0.8014** | **−0.0374** | **−0.0280** |
| **text** | 0.8037/0.8003/0.8935 | 0.7570/0.7377/0.8368 | 0.7477 | 0.7664 | **0.7850/0.7667/0.8790** | **+0.0280** | **−0.0187** |
| concat | 0.8505/0.8489/0.9052 | 0.7944/0.7862/0.8674 | 0.7570 | 0.8037 | **0.7383/0.7232/0.8605** | **−0.0561** | **−0.1121** |

**MHC_zh**

| stream | causal | bidir-lasttoken | S1 meanpool | S1b textpool | **S2a MNTP** | vs bidir | vs causal |
|---|---|---|---|---|---|---|---|
| img | 0.7436/0.7057/0.8379 | 0.7564/0.7173/0.8414 | 0.7564 | 0.7564 | **0.7308/0.6803/0.7621** | **−0.0256** | **−0.0128** |
| **text** | 0.8462/0.8353/0.9407 | 0.6282/0.5203/0.6886 | 0.7051 | 0.6923 | **0.6923/0.5667/0.8157** | **+0.0641** | **−0.1538** |
| concat | 0.8590/0.8519/0.9214 | 0.6410/0.5439/0.7400 | 0.7436 | 0.7051 | **0.7051/0.6408/0.7957** | **+0.0641** | **−0.1538** |

**`KS-MNTP-1` against the frozen bars — the first CONTINUE of the entire campaign:**

| dataset | S2a text acc | frozen bidir | floor25 | bar50 | recovery | cell verdict |
|---|---|---|---|---|---|---|
| **HateMM** | **0.7850** | 0.7570 | 0.7687 | 0.7804 | **+0.6006** | **CONTINUE (≥50 %)** |
| MHC_zh | 0.6923 | 0.6282 | 0.6827 | 0.7372 | **+0.2941** | PARTIAL |

**Collapse belt — FAIL on both:**

| arm | HateMM train / dev | MHC_zh train / dev |
|---|---|---|
| causal | 0.3523 / 0.3499 | 0.3105 / 0.3027 |
| bidir-lasttoken | 0.4314 / 0.4511 | 0.4977 / 0.4898 |
| S1 meanpool | 0.9273 / 0.9404 | 0.9320 / 0.9316 |
| S1b textpool | 0.7566 / 0.7624 | 0.7565 / 0.7538 |
| **S2a MNTP** | **0.6494 / 0.6550** | **0.6386 / 0.6433** |

Worst 0.6550 / 0.6433, both **≥ 0.60**. Declared rule: **self-refutes regardless of accuracy —
record and stop.** `KS-MNTP-2` **NOT RUN**.

**THE STOP IS OVERDETERMINED — four independent reasons, so the belt is not load-bearing alone:**

1. **The collapse belt fires** (pre-declared, ≥0.60 on both datasets).
2. **Fusion inverts from additive to destructive.** Under causal, concat *beats* the best single
   stream (**+0.0467** HateMM, **+0.0128** ZH). Under S2a, concat is *worse* than the best single
   stream (**−0.0467** HateMM, **−0.0256** ZH). The deployed system is a fusion head, so a
   text-only gain that destroys fusion is not a usable gain. **This independently corroborates the
   diversity loss the belt is designed to detect.**
3. **Every S2a number is below its causal floor**, on both datasets and all three streams.
4. **`KS-MNTP-3` is not met and cannot be**: escalation requires Δdev ≥ **+0.020** *over* the
   causal floor; S2a text is **−0.0187** below it on HateMM and **−0.1538** on ZH.

**Honest note on belt transferability, raised for the main loop and NOT resolved here.** The
collapse belt was designed for a *readout* pathology (S1/S1b pooled vision tokens, so text became
a copy of img). S2a is a *weights* arm with the readout untouched, and on **HateMM the text stream
retains — indeed slightly widens — its margin over img** (S2a +0.0561 vs causal +0.0467), which is
**not** the "text becomes a copy of img" signature the belt was written for. A reasonable reader
could argue the belt is firing on a different phenomenon here. **I did not move the bar**: it was
declared before the arm existed, and post-hoc reinterpretation of a pre-declared gate is exactly
the failure mode the freeze discipline exists to prevent. It is recorded because reasons 2-4
make the stop hold regardless, so nothing turns on it — but whether this belt should carry
forward unchanged to future *weights* arms is a live design question for the main loop.

### 6e.5 H1 vs H2 — UPDATED READING

**The MNTP transplant produces the first real movement on the bidirectional text axis of the whole
campaign, and simultaneously demonstrates why it is not usable.**

- **The signal is real.** HateMM text +0.0280 over F72 bidir = **60 % of the crater recovered**,
  the first time any arm cleared the 50 % bar. ZH +0.0641 (29 %). Both signs positive. Weight
  adaptation moves the text stream in a way three different readouts could not — **which is
  consistent with the S1b conclusion that the problem lives in the weights/topology, not the
  readout.** To that extent the MNTP hypothesis is *supported in direction*.
- **The cost is global.** The same perturbation **degrades the img stream on both datasets**
  (−0.0374 / −0.0256 vs F72 bidir; also below causal), collapses stream diversity to ~0.64-0.66,
  and **inverts fusion from additive to destructive**. Features move enormously: cos vs plain
  bidir is 0.31-0.36 (text) and 0.65-0.72 (img).
- **This is exactly the transplant risk declared in §6d.1**, now measured: a delta fitted to
  `Qwen2.5-7B-Instruct`'s weight point is a **large, blunt perturbation** at the VL weight point,
  which drifted during VL pretraining. It carries *some* of the bidirectional adaptation — enough
  to move text 60 % of the way back — while damaging representations the VL trunk had learned.
- **What is refuted:** the zero-training shortcut to MNTP. **What is NOT refuted:** the MNTP
  hypothesis itself. §6d.5 declared this distinction in advance, and the result lands squarely on
  it. A transplant under-delivering at a drifted weight point is weak evidence about what a
  *correctly fitted* MNTP would do.

**H1's strong form remains refuted** (img is fine under a bare mask flip, +0.0093/+0.0128, cosine
1.000000 across three extractions). **H2 stays refuted** (S1b). The surviving live hypothesis is
still **weight adaptation fitted to this model** — i.e. **S2b**, which needs the corpus ruling.

### 6e.6 S2a cost

| job | what | elapsed |
|---|---|---|
| 13658 | S2a smoke (`S1_LIMIT=4`) | 00:02:05 |
| 13659 | belt-4 same-path double-merge floor probe | 00:08:55 |
| 13660 | S2a full extraction, both datasets, train+dev | 00:49:20 |

**S2a GPU: 01:00:20 = 1.006 GPU-h** (budget ~1.0). **Campaign total S1 + S1b + S2a = 2.697
GPU-h.** Download gate spent (80 MB). No training, no corpus ruling, no Modal. `KS-MNTP-2` not
run, so the banked 13652/13655 CPU floors remain unspent.

---

## 7. GATE-BY-GATE SUMMARY

| gate | outcome |
|---|---|
| `KS-MNTP-0a` installation belt | **PASS** (self-test, SDPA assert, `is_causal=False` on 28 modules) |
| pre-GPU $0 belts (dispatch, delegation, pooling arithmetic, clobber + test-touch guards) | **PASS** (6/6) |
| codex pre-submission review | **GO, no blockers**; 3 nits closed + 2 extra hardenings |
| smoke 13650 (img null-op / text changed) | **PASS** (img cos 1.000000; text cos 0.43-0.59) |
| cache sanity 13654 | **PASS** (1508 rows, id order matches causal, no test file) |
| **`KS-MNTP-1` raw-key dev screen** | **STOP** — HateMM −0.1999 (below crater), ZH +0.3529 (partial); no dataset ≥50 %, signs inconsistent |
| `KS-MNTP-2` CPU head (S1 arm) | **NOT RUN** — gate said stop; controls banked (13652/13655) |
| `KS-MNTP-3` escalation flag | **NOT REACHED** |
| — **S1b (amendment, §6b/§6c)** — | |
| external pre-submission review | **NO-GO → GO** — 2 blockers (test-cache backup exposure; undeclared `--no_merge`/`--moka` divergence) + 3 nits, all fixed |
| `KS-MNTP-0a` + $0 belts (S1b) | **PASS** (incl. token-mask verification: 720 vision dropped, 0 `video_pad` survived, markers kept) |
| smoke 13656 (img null-op / text differs from BOTH prior arms) | **PASS** (img cos 1.000000; text vs F72 0.45-0.72, vs S1 0.87-0.91) — with a recorded collapse early warning |
| cache sanity 13657 | **PASS** (1508 rows, id order matches causal, no test file) |
| **collapse belt (declared in advance, < 0.60)** | **FAIL — 0.7624 / 0.7565. ARM SELF-REFUTES.** |
| `KS-MNTP-1` (S1b) | HateMM +0.2003 (KILL-side), ZH +0.2941 (partial); no dataset ≥50 % — **accuracy gate alone would have said CONTINUE; the belt overrode it** |
| `KS-MNTP-2` CPU head (S1b arm) | **NOT RUN** — belt override |

| — **S2a (amendment, §6d/§6e)** — | |
| external pre-submission review | **NO-GO ×2 → GO** — caught that the transplant would have loaded **ZERO weights** (PEFT key one `.model` short, loads non-strictly ⇒ silent no-op) and would have bound **96 vision-tower modules** |
| adapter scope + belt 1 | **PASS** — 196 modules (28×7), 0 vision, **196/196 non-zero `lora_B`**, 5/5 weight probes changed |
| `KS-MNTP-0b` adapter-not-null-op | **PASS** — cos(S2a text, plain-bidir) 0.3639 / 0.3076 ≪ 0.9999 |
| belt 4 same-path double-merge floor (13659) | **PASS** — all 8 cells ≥0.9999 (mean 1.00000000) ⇒ extra merge inert, **no binding floor**, ~0.85 GPU-h saved |
| `KS-MNTP-1` (S2a) | **HateMM +0.6006 = CONTINUE (first ≥50 % of the campaign)**; ZH +0.2941 partial |
| **collapse belt** | **FAIL — 0.6550 / 0.6433 ⇒ self-refutes** |
| `KS-MNTP-2` (S2a arm) | **NOT RUN** — belt override; stop is **overdetermined** (see below) |

**Verdict routing — the readout route is CLOSED.** Three pooling spans were tried across the full
range of what a readout can select — EOS-class tail (F72), all positions (S1), text positions only
(S1b) — and all three land far below the causal floor while the streams converge monotonically
with how much of the sequence is pooled (causal 0.31-0.35 → S1b 0.76 → S1 0.93). The mechanism is
identified: under bidirectional attention every text token attends to all ~720 vision tokens, so
the text representations are saturated with visual content **before** any pooling happens.
**A readout cannot undo an information mixture created by the topology.** H2 is refuted on its
strongest available test; H1's strong form stays refuted (img unharmed, +0.0093/+0.0128,
reproduced at cosine 1.000000 on three independent extractions).

**S2a then tested the surviving weight-adaptation hypothesis at zero training cost, and produced
the campaign's first real signal — which is nonetheless not promotable.** The published MNTP
transplant recovers **60 % of the HateMM crater** on the text stream (the first arm ever to clear
the 50 % bar) and +29 % on ZH, confirming in direction that the problem lives in the weights
rather than the readout. But the same perturbation degrades the img stream on both datasets,
collapses stream diversity to ~0.65, and **inverts fusion from additive (+0.0467 over the best
single stream under causal) to destructive (−0.0467 under S2a)**. Every S2a number sits below its
causal floor, and `KS-MNTP-3` (Δdev ≥ +0.020 *over* the floor) is unreachable — S2a text is
−0.0187 below it. The stop is **overdetermined**: pre-declared collapse belt, fusion inversion,
sub-floor everywhere, and an unmeetable escalation bar.

**What S2a refutes is the ZERO-TRAINING SHORTCUT, not the MNTP hypothesis** — a distinction
declared in §6d.5 before the arm ran. A delta fitted to `Qwen2.5-7B-Instruct` is a large, blunt
perturbation at the VL weight point (features move to cos 0.31-0.36), so it is weak evidence about
what a correctly-fitted MNTP would do.

**The one live hypothesis remaining in this lane is S2b — MNTP trained by us on this model.** It
requires the user's corpus ruling (recon §3.4: wikitext needs the veto relaxation; own-split
multimodal (a′) is legal without a ruling but supplies only ~1.5 % of the reference token budget).
That is a main-loop/user decision, not this agent's. Nothing in S1, S1b, or S2a advances the goal
clause, and no arm's movement over F72 may be quoted as a method gain.
