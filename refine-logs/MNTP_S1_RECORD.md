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

**ZERO test-touch.** No held-out test file was opened, read, or produced at any point in S1.
Enforced by four independent belts: (i) the S1 extractor raises on any `--splits` containing
`test`, before any side effect; (ii) the sbatch hard-codes `--splits train,val`; (iii) the S1
caches have no `test_seen` file at all; (iv) the KS-MNTP-2 harness replaces the loader with a
dev-only variant returning `(train, dev, dev)` and wraps `load_feats_split` in a guard that
raises on any path containing `test_seen` — applied **uniformly to every arm**, so no arm reads
test and the harness is identical across arms.

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
| `scripts/analysis/mntp_s1_cpuhead.py` | `65bb1dcbc33dbe8b38cd498968d7b6924f1ec0262b92924f2c2bd64c24ffdd77` |
| `scripts/slurm/mntp_s1_cpuhead.sbatch` | `e71d5a95d204bdaccb11f17115fb2cb76b0e06c8f7559b69633c8ff0b160715b` |

Primary outputs: `scripts/analysis/mntp_s1_devscreen_OUT.json`,
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

**Verdict routing:** S1 does not escalate. The naive mean-pool readout is dead; the *underlying*
readout hypothesis is **untested**, because the S1 operator collapsed the two streams
(cos 0.93). The cheapest remaining discriminator is a **text-positions-only** mean pool
(exclude the 768 video-pad tokens) — same ~0.85 GPU-h, no ruling, no download. Whether to fund
it, or to route instead to S2a (the published McGill MNTP transplant, download-gated), is the
main loop's decision.
