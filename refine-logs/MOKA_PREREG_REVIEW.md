# MOKA-ZH Pre-Registration — INDEPENDENT 0-CONTEXT REVIEW

**Reviewer:** independent 0-context pre-registration reviewer. **ZERO GPU / SLURM / Modal spent**
(pure-CPU login-node work only: the committed CPU smoke, my own re-implementations of the identity
control at deployed dimensions, a tokenizer sweep, floor re-parsing from raw trainlogs, `sha256sum`,
`bash -n`, `py_compile`, `diff`, read-only `sacct`). **NO job submitted. NO test metric read. NO
`state/` mutation. NO `research-wiki/` mutation. Not pushed.**
**Timestamp (`date -u`):** `Sat Jul 25 13:23:02 UTC 2026`.
**Object under review:** `refine-logs/MOKA_PREREG.md` (commit `7c4a22e`, 842 lines) + the 7 artifacts
of its §5.1/§5.3 freeze block.
**Design authority checked against:** `refine-logs/MOKA_FORENSIC_RECON.md` (`dbf30f1`).
**Upstream reference checked:** `external/baselines/MokA` @ `b28e83431d057e2b83c8b7f5bd7cde9f33d6393a`.
**Environment of my own runs:** `HateVideo`, torch `2.6.0+cu124`, peft `0.14.0`, transformers `4.49.0`.
**Repo HEAD at review start:** `7c4a22ec4dc9ff51ee66b580595d5ae6f3a460cb`.

---

## RULING: **APPROVED-WITH-NOTES**

Every mandatory gate verified **on my own execution**, not by reading the author's claims. All 7
freeze-block shas recomputed and match byte-for-byte. The machinery is bit-exact where the prereg
says it is bit-exact. Six notes are recorded below; **none is blocking** — no note changes a
threshold, a kill bar, a gate ordering, the test-touch budget, or any frozen artifact. Notes **N1**
and **N2** are binding on the **write-up at verdict time** and are carried into `MOKA_FREEZE.md`.

---

## V1 — Floors + thresholds: **PASS** (re-derived from primary trainlogs)

I re-parsed the three raw floor trainlogs with the **embedded parser lifted verbatim from
`scripts/slurm/enc3seed_zh_b3.sbatch:63-82`** (val-sel = max `Val_Retrieval` acc over epochs ≥ warmup
5, `roc` tie-break; final = max epoch). Files:
`slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`.

| seed | val-sel ep | val-sel acc / mF1 | final ep | final acc / mF1 |
|---|---|---|---|---|
| 0 | 20 | 0.8322 / 0.8023 | 29 | 0.8456 / 0.8181 |
| 1 | 26 | 0.8255 / 0.7956 | 29 | 0.8389 / 0.8113 |
| 2 | 19 | 0.8389 / 0.8065 | 29 | 0.8523 / 0.8226 |
| **mean** | | **0.8322 / 0.801467** | | **0.8456 / 0.817333** |

**Identical, epoch-for-epoch and digit-for-digit, to prereg §2.1.** No transcription drift.

**FORMAL = floor + 0.030 exactly** (my arithmetic, full precision → 4 dp):

| protocol / metric | floor mean | +0.030 | prereg §2.3 | match |
|---|---|---|---|---|
| val-sel acc | 0.832200 | 0.862200 | 0.8622 | ✔ |
| val-sel mF1 | 0.801467 | 0.831467 | 0.8315 | ✔ |
| final acc | 0.845600 | 0.875600 | 0.8756 | ✔ |
| final mF1 | 0.817333 | 0.847333 | 0.8473 | ✔ |

Stage-2 reference §2.2 (HateMM 13241) also re-derived from primary logs: per-seed val-sel
0.8791/0.8730, 0.8744/0.8678, 0.8791/0.8724 → mean **0.8775 / 0.8711**; final all 0.8791 with mF1
0.8730/0.8724/0.8724 → mean **0.8791 / 0.8726**. **Bit-matches §2.2.** The auto-defund arithmetic
(0.8791 + 0.030 = 0.9091) is therefore auditable as claimed. §2.1's explicit refusal of the ledger's
`0.8537` (0.8732-incident discipline) is correct — that is a different ZH cell.

---

## V2 — Identity / parity architecture: **PASS** (my own smoke run + independent re-derivation)

### (a) I ran the committed CPU gate end-to-end myself

`python scripts/analysis/moka_smoke.py` → **`==== ALL SMOKE CHECKS PASS ====`**, exit 0. S1–S8 all
green on my run. Load-bearing cells, verbatim from my output:

- **S2 identity control:** `max|Δ| = 0.000e+00` in **all 6 cells** (2 modules × {all-text,
  all-vision, mixed}) — the recon's unrelaxed `0.0` threshold, met.
- **S2 dense-vs-gather cross-impl:** `0.0` on all-text/all-vision; `1.192e-07` / `5.960e-08` on
  mixed — reproducing the exact residuals the prereg's DEV-1 reports.
- **S4 grad flow:** `hook_calls 1`, `routed_calls 2`, `fallback_calls 0`, `dead=[]`; non-zero grads
  on `lora_A`, `lora_A_v` **and** the shared `lora_B` at both modules.
- **S5** round-trip incl. the silent-drop demonstration and the generic-adapter refusal; **S6** all
  three merge entry points raise; **S7** `40370176 → 58490880 = 1.448864×`.
- **S8** on the real record #0: `mask 2688 vs grid_thw arithmetic 2688`, `seq_len 2823 = vision 2688
  + text 135 (vision share 95.2%)`, masked ids `= [151655]`; uncapped branch `21528 = 21528`.

### (b) I re-derived the identity control independently, at **deployed** dimensions

The committed smoke proves identity on a 32-wide toy. I rebuilt the check on the **real Qwen2.5-VL-7B
projection shapes** — `q_proj` 3584→3584, `k_proj` 3584→512, `up_proj` 3584→18944, `down_proj`
18944→3584 — with `lora_A_v` tied to `lora_A`, a non-trivial (perturbed) `lora_B`, and **four** mask
patterns including the **real measured ZH pattern** (`seq 2823`, 2688 contiguous vision positions =
94.6 %). Result, all 16 cells:

```
dense  vs upstream PEFT : 0.000e+00  in 16/16 cells  (BIT-EXACT)
```

**The dense-select routing reproduces upstream PEFT bit-exactly at deployed dimensions, on mixed
masks, including the real vision/text layout.** This is the claim the whole single-variable argument
rests on, and it holds under my own independent implementation.

### (c) DEV-1 honesty check — the gather residual is real but **shape-specific** (→ NOTE N3)

I swept the gather variant across widths × shapes to test *why* dense was frozen:

| in_features | bsz × seq | dense vs upstream | gather vs upstream |
|---|---|---|---|
| 32 | 2 × 11 | 0.000e+00 | 5.960e-08 |
| 256 | 2 × 11 | 0.000e+00 | 1.192e-07 |
| 1024 | 2 × 11 | 0.000e+00 | 1.192e-07 |
| 3584 | 2 × 11 | 0.000e+00 | 2.831e-07 |
| 18944 | 2 × 11 | 0.000e+00 | 5.364e-07 |
| **all of the above** | **1 × 2823** | **0.000e+00** | **0.000e+00** |
| **all of the above** | **2 × 512** | **0.000e+00** | **0.000e+00** |

So: DEV-1's measurement is **honestly and accurately reported** (I reproduce `5.96e-08 / 1.19e-07`
exactly), and the conclusion "dense is bit-exact, gather is not" is **true at the smoke's shape**.
But the residual is a **small-row-count** BLAS-blocking artifact: at the deployed shape
(`per_device_train_batch_size 1`, seq ≈ 2823) **gather is also bit-exact at every width tested.**
The freeze of dense is therefore **strictly conservative** (dense is bit-exact everywhere I tested)
and its cost was already disclosed and amended into §F0.7 (+1 % FLOPs, per-token rank unchanged at
16). The rationale text over-generalises a toy-shape measurement — see **N3**. Not blocking: the
frozen implementation is the safer of the two and the comparability arithmetic (1.448864× params,
per-token rank 16) is unaffected.

### (d) Structural safety I verified beyond the checklist

The banked ZH adapter contains exactly **196 `lora_A` + 196 `lora_B` = 392 tensors, 40,370,176
params**, spanning **28 decoder layers × 7 projections** (`q,k,v,o,gate,up,down`) with **zero
`visual.*` / `merger` keys**. Consequence: every layer `install_moka` converts sits on the **decoder
sequence axis**, where the `input_ids`-derived mask is positionally valid. The failure mode where a
vision-tower LoRA layer receives a patch-axis input and trips `MOKA_STRICT` is **structurally
absent**, and job-1's `n_a == n_av == n_b == 196` assert would catch any regression. I also confirmed
LLaMA-Factory applies `image_max_pixels` by PIL-resizing **before** the processor
(`mm_plugin.py:229-236` `_preprocess_image`, sqrt factor + int truncation) — exactly what smoke S8's
`_lf_resize` replicates, so S8's 2,688 is the genuinely deployed token count. And **no record is
truncated** at `cutoff_len 4096`: worst case = 2688 + 407 = **3095**.

---

## V3 — Single-variable discipline: **PASS**

- **YAML diff = exactly 1 line.** `diff mhc_zh_qwen25vl_lora_sft.yaml mhc_zh_qwen25vl_lora_moka_sft.yaml`
  → `27c27  output_dir: …/MHC_zh → …/MHC_zh_moka`. Nothing else. 48 lines. Recipe therefore pinned at
  `lora_rank 16`, `lora_alpha 32`, all 7 targets, `freeze_vision_tower/projector true`, `lr 1.0e-4`,
  `3.0` epochs, bs 1 × accum 8, cosine, warmup 0.05, bf16, grad-ckpt, `cutoff_len 4096`,
  `save_strategy epoch`, `lora_dropout` unset ⇒ 0.0 — as §1.1 states.
- **`train_moka.py` replicates the deployed invocation surface.** Deployed =
  `cd $LF_ROOT && python src/train.py <yaml>` (`lora_sft.sbatch:78-80`). The wrapper puts
  `$LF_ROOT/src` at `sys.path[0]` (same as the deployed entry), `os.chdir(LF_ROOT)` before
  `run_exp()`, and calls the identical `llamafactory.train.tuner.run_exp`. The sbatch also `cd`s to
  `$LF_ROOT` first, so cwd is right regardless. Confirmed at runtime: `run_exp.__module__ ==
  llamafactory.train.tuner`.
- **Extractor flags default OFF, no default-ON leak.** `parse_args_sys([])` → `moka=False`,
  `no_merge=False` (both `action="store_true"`, no `default=True`, no `dest` aliasing). The gating is
  `if args.moka or args.no_merge: … else: <the two original lines verbatim>` — I read the diff: the
  `else` branch is byte-identical to the pre-edit `merge_and_unload()` path, so **default ⇒ deployed
  path**. The `--moka` sub-branch is nested one level deeper (`if args.moka:` inside), so `--no_merge`
  alone never installs MokA. The ZHPROMPT prompt-arg identity is intact (`img_instruction` /
  `text_instruction` still the deployed English literals).
- **Diff size honest:** `git diff --numstat` = `33  2` — exactly the declared **+33 / −2** (DEV-5).
- **`import routed_lora` is deferred inside the `if args.moka:` block** (extractor line 510), as is
  the `sys.path.insert` — so a non-MokA extraction never touches MokA code at all.

---

## V4 — Vendored-tree integrity: **PASS**

- **Gitlink unchanged.** `git ls-tree` at `HEAD`, `HEAD~1`, `HEAD~2` all → `160000 commit
  a912747c408b3c661b4029ecf1d88b9d91c7f1a8`; `git ls-files -s` agrees; submodule `git rev-parse HEAD`
  = `a912747c…`. **Commit `7c4a22e` did not move the gitlink.**
- **Zero edits inside.** Inside the submodule, `git status --short` shows **no modified tracked
  file** and `git diff --stat HEAD` is empty. The only untracked entries are
  `my_configs/hatevideo/mhc_zh_qwen25vl_lora_moka_sft.yaml` (the frozen yaml, expected: gitlink-resident
  ⇒ hash-frozen not committed, `CAND2_FREEZE.md:21-22,35-36` / `LORA_HATEMM_FREEZE.md:18,24`
  precedent — verified those files do exactly this) and a pre-existing `.cuda_home_shim/` (the nvcc
  shim the deployed `lora_sft.sbatch` already points `CUDA_HOME` at).
- **Monkey-patch is runtime-only and does not leak.** Importing `src/moka/train_moka.py` against the
  real LLaMA-Factory (CPU, `DISABLE_VERSION_CHECK=1`, `CUDA_HOME` = the shim):

```
adapter.get_peft_model patched            : True
original preserved (module)               : peft.mapping
adapter._setup_lora_tuning contains call  : True      # the hot path, adapter.py:312
GLOBAL LEAK? peft.get_peft_model patched  : False
GLOBAL LEAK? peft.mapping.get_peft_model  : False
cwd unchanged by import                   : <unchanged>
```

  The patch rebinds **one module attribute** on `llamafactory.model.adapter`; `peft`'s own namespace
  is untouched, so nothing in the process outside LLaMA-Factory's adapter path can pick it up.
- **Nothing but the MokA family imports it.** Repo-wide grep for `routed_lora | install_moka |
  src/moka` (excluding `external/`, `refine-logs/`, the vendored tree) hits **only**
  `scripts/analysis/moka_smoke.py`, `src/moka/*`, `scripts/slurm/lora_sft_moka.sbatch`, and the
  deferred import at `generate_VideoMLLM_embedding_lora_HF.py:510`. `src/run_rac.py` — the head path —
  never sees it. **No import side-effect reaches a non-MokA run.**

---

## V5 — Binding-language coherence: **PASS with notes N1, N2, N4, N5**

**Present and verbatim / correct:**

| clause | location | verdict |
|---|---|---|
| `KS-MOKA-0` (S1–S8 mandatory, any failure ⇒ NO submission) | §3.2 | present, and I ran the gate |
| `KS-parity` bit-exact re-extraction ⇒ **HALT** not a result | §3.2 last bullet, §3.3, §4.4(3) | present |
| `KS-MOKA-0b` bar ≥ 0.9999 on **all 6** cells, contingent floor +3 evals pre-budgeted | §3.4, §6 | present, implemented in job-2 Stage A0 |
| `KS-MOKA-1` auto-defund of the HateMM leg | §3.5 + job-1 hard refusal | present (see **N4**) |
| `KS-MOKA-2` null-op detector, $0, emitted by job 1 | §3.6 + sbatch lines 96-110 | present (see **N1**) |
| `KS-MOKA-3` mandatory before ANY claim, 3 pre-declared readings | §3.7 | present (see **N5**) |
| *"NEVER narrate any outcome as 'MokA protected the visual modality' unless KS-MOKA-3 shows the image stream moved AND the head followed"* | §3.7 **and** §7 template | **verbatim present in both places** ✔ |
| "text moved ⇒ **text-side** mechanism" / "image moved + head flat ⇒ **9th law-I**" | §3.7, §7, §8 | verbatim ✔ |
| `KS-regression` (≤ −0.030 ⇒ measured REGRESSION note, not a new bite) | §3.8, §7 | consistent ✔ |
| **one-bite** — ONE family = ONE multiplicity bite spanning both jobs; scope FROZEN with the re-cost list | §3.10 + title-scope para | present, and the OUT-list matches §3.10 ✔ |
| **D7 transplant-credit** — novelty claimable ONLY as first-application AND only with MokA credited; "we invented modality-routed LoRA" **banned** | F0.3, §8 | present; and the code credit header (`routed_lora.py:1-6`) names `GeWu-Lab/MokA` (NeurIPS 2025) @ `b28e834` — **I verified that commit resolves to `b28e83431d057e…`** ✔ |
| **§4.6 code-fix ⇒ re-freeze + re-review**, submit-time re-`sha256sum`, mismatch = authorization VOID | §4.6, §5.3 | present; NCA_PREREG.md:422,458 precedent confirmed ✔ |
| Verdict rendered by an independent 0-context reviewer; executor applies **no** gates | §3.10, §6 | present ✔ |
| Ban-collision closure (EN not re-opened; not F65/F70/F66/HUNT-3) | §3.9 | coherent; `REDTEAM_BAN_SCOPE_AUDIT.md` GAP-5 argument checks out ✔ |

`KS-MOKA-3`'s machinery exists and the rule is transcribed faithfully:
`scripts/analysis/encoder_swap_geometry.py` and `scripts/analysis/hatemm_lora_stream_decomp.py` are
both present, and the latter's own header (`:32-33`, constants `MOVE_TR = 0.010`, `MOVE_DV = 0.005`)
reads *"MOVED iff dAUC_s ≥ +0.010 on train-LOO AND ≥ +0.005 on dev (same + sign); FLAT iff |dAUC_s| <
0.010 on train-LOO"* — matching §3.7 word for word.

### NOTE **N1** (substantive; binding on the write-up) — `KS-MOKA-2`'s 0.05 bar is **structurally unable to fire**, so "routing is real" is not a supported inference

The bar: *median per-layer `‖A_v − A_t‖_F / ‖A_t‖_F < 0.05` ⇒ the two down-projections have converged
⇒ NULL-OP*. I measured both ends of the scale on real objects:

| quantity | measured | source |
|---|---|---|
| two **independent** Kaiming draws at the deployed `A` shape (16 × 3584), 20 trials | mean **1.4136** (min 1.4089 / max 1.4191) | my run; matches §3.6's "≈ 1.41" and smoke S4 |
| **total displacement of a trained `lora_A` over the real deployed 3-epoch ZH SFT** (`checkpoint-73` → final, i.e. epoch 1 → epoch 3) | median **0.0506**, min 0.0164, **max 0.1267** | `logging/lora/MHC_zh/{checkpoint-73,}/adapter_model.safetensors` |

`A_v` and `A_t` **start** ≈ 1.41 apart (independent Kaiming draws — smoke S1 asserts they differ, and
`moka_init` draws `A_v` fresh). For the median to reach < 0.05 they would have to converge by ≈ 1.36
relative-Frobenius, i.e. move roughly **27× the entire measured training displacement of an `A`
matrix in this exact recipe**, in a perfectly convergent direction. **`KS-MOKA-2` will therefore
report "median ≥ 0.05" with probability ≈ 1 whether or not routing does anything.**

- The clause is **not wrong** (a fire *would* mean null-op) and it is **recon-faithful** (DEV-6:
  "pinned verbatim"); it also cannot manufacture a PASS, since it gates nothing and is emitted after
  training at $0.
- What it **cannot** support is §7's label `KS-MOKA-2 (routing is real): median ____ -> <routing real
  | NULL-OP>`. **The `median ≥ 0.05 ⇒ routing real` direction is unsupported.**
- **Binding on the verdict write-up:** `KS-MOKA-2` must be reported as a **non-degeneracy floor**
  (the two `A`s did not collapse onto each other), **not** as evidence that routing changed the
  learned function. The claims that *do* carry that weight are (i) `fallback_calls == 0` under
  `MOKA_STRICT=1` (§4.4(1), DEV-G) — the genuine null-op guard — and (ii) `KS-MOKA-3`'s stream
  decomposition (§3.7), which is already MANDATORY.
- **Constructive, $0, non-blocking recommendation:** because DEV-3 pins `A_v`'s init deterministically
  (`MOKA_INIT_SEED = 20260726 + 8·layer_index` inside `torch.random.fork_rng`), `A_v`'s initial value
  is **exactly reconstructible offline**. The genuinely informative statistic — *did the vision route
  train at all* — is `‖A_v^final − A_v^init‖_F / ‖A_v^init‖_F` per layer, computable at $0 from the
  saved adapter with no GPU. Adding it would give `KS-MOKA-2` real discriminating power. Adding it
  **after** freeze is a code edit ⇒ **§4.6 fires** (re-freeze + re-review); it may equally be computed
  as a separate post-hoc analysis at verdict time without touching a frozen artifact.

### NOTE **N2** (binding on the write-up) — carry F0.2 and F0.6 into the verdict, as §7 already requires

§7's template already mandates restating **F0.2** (single encoder draw; encoder-seed noise confounded
with the routing effect and **not separable**) and **F0.6** (94.6 % vision tokens vs MokA's own 98.4 %
text regime). I confirm both are materially correct (F0.2 is the same limitation B3/F53/curric/bidir
carry; F0.6 verified in V6) and record here that the reviewer treats **both restatements as
conditions of the approval**, not as optional colour.

### NOTE **N3** — DEV-1's rationale over-generalises a toy-shape measurement

See V2(c). The measurement is honest and I reproduced it exactly; the *inference* "gather cannot meet
the 0.0 threshold" holds at `bsz 2 × seq 11` but **not** at the deployed `bsz 1 × seq ≈ 2823`, where
both formulations are bit-exact at every width I tested. The frozen choice (dense) is the
conservative one and its ~+1 % FLOPs cost is already amended into §F0.7. **Non-blocking**; if the
DEV-1 paragraph is ever restated in the paper, the shape-dependence should be stated with it.

### NOTE **N4** — `KS-MOKA-1`'s scope quantifier is ambiguous; pin the reading before the verdict

§3.5 reads: *"If on BOTH protocols the 3-seed mean paired Δacc ≤ 0, OR the acc sign is not 3/3
positive, the ZH arm is DEAD…"*. Two parses exist: **(A)** "on both protocols: (Δacc ≤ 0 **or** sign
≠ 3/3)" vs **(B)** "(on both protocols Δacc ≤ 0) **or** (sign ≠ 3/3)" — under (B) the sign clause has
no protocol scope. I traced the consequence: under either parse the kill is **strictly easier to
trigger than FORMAL**, so **no false PASS is reachable**; the only reachable divergence is a
one-protocol-positive arm that "survives" `KS-MOKA-1` and so does not auto-defund stage 2. That
outcome is contained anyway — stage 2 additionally requires a prereg amendment, a new yaml, and is
hard-refused by job 1 (`exit 2`). **Non-blocking**; recommend the executor state which parse is
applied when transcribing the `KS-MOKA-1` line.

### NOTE **N5** — §3.7's three readings are not exhaustive; §2.3's FORMAL phrasing bundles protocols

(i) §3.7 enumerates *text moved* / *image moved + head flat* / *neither moved*. The **both moved**
case is unenumerated. It is adequately fenced by the same section's stronger guard ("a PASS may not
be narrated as visual-modality protection unless `KS-MOKA-3` independently shows the image stream
moved *and* the head followed"), so no narration loophole opens — but the executor should state the
both-moved case explicitly if it arises.
(ii) §2.3 states FORMAL as one bundled bar ("both protocols"), while §3.1(clause 5 / final sentence)
and §7's template judge and report **per protocol** ("final-epoch: pass/fail; val-selected:
pass/fail", the B3 house form). The bundled reading is **stricter**, so this is not a shopping
loophole; report per protocol, and treat "clears FORMAL" (the goal conjunct, §8 bullet 3) as
requiring both. **Non-blocking.**

---

## V6 — F0.6 counter-pressure honesty: **PASS** (with a minor provenance note, N6)

- **Not buried.** F0.6 is a full bolded clause in **§0** (the binding-facts section, before any
  method text), is **restated as a mandatory line in §7's verdict template** ("Restate F0.2 … and
  F0.6 (94.6 % vision tokens; MokA's own regime is 98.4 % text) in the verdict, whatever the
  outcome"), and is repeated as **DEV-8** in §11. Three placements, one of them mandatory at verdict
  time.
- **The prior implication is stated, not implied.** F0.6 says outright: *"it is a reason to sit at
  **5 %**, not 8 %"* and *"the transfer from MokA's reported gains is **weak**: their text-dominant
  regime is not ours"*; DEV-8 adds that the prior is *"left at the recon's 5–8 % rather than
  unilaterally re-priced; the reviewer may tighten it."* The **5 % end** is disclosed exactly as the
  checklist requires. As reviewer I accept the 5–8 % band as filed and note the low end is the
  honest reading.
- **Upstream 98.4 % verified at source.** `external/baselines/MokA` @ `b28e834`,
  `VisualText/modified_peft/tuners/lora/layer.py:565-566`: `my_text_mask tensor(16128.)` /
  `my_image_mask tensor(256.)` on `x torch.Size([1, 16384, 3584])` → 16128 + 256 = 16384 exactly,
  **98.4 % text**. The citation is accurate and the mirror-image framing is correct.
- **Record #0 exact.** My independent tokenization: text 135, vision 2688, total 2823, vision share
  **95.22 %** — matching F0.6's "2,688 + 135 = 2,823 tokens, 95.2 % vision" and my own smoke-S8 run.
- **NOTE N6 (minor, numeric-provenance).** The 579-row quantile tuple is **not bit-reproducible**
  because the prereg does not define what counts as a "text token". Over all 579 rows of
  `data/lora_sft/MHC_zh/train.json` (I confirm 579 rows, all with exactly 8 images) I get, per
  convention: **all non-`image_pad` ids** → min 95 / p25 126 / median 167 / p75 223 / max 407
  (⇒ 94.15 % median vision share); **also excluding `vision_start`/`vision_end`** → 79 / 110 / 151 /
  207 / 391 (⇒ **94.68 %**). The prereg reports 81 / 112 / 153 / 210 / 393 (⇒ 94.6 %) — closest to
  the second convention, offset ≈ 2 tokens. **The headline 94.6 % reproduces to within 0.5 pt under
  every convention I tried, and the qualitative claim (vision dominates at ~94–95 %, the mirror image
  of MokA's 98.4 % text) is robust.** F0.6 gates nothing — it shades a prior — so this is a
  transparency note only: state the counting convention when F0.6 is restated at verdict time.

---

## V7 — Budget / infra: **PASS** (every basis re-read from `sacct` / primary files)

| item | prereg basis | my verification | GPU-h |
|---|---|---|---|
| GPU smoke (10-step SFT + 2-video extract + KS-parity) | — | plan reviewed §4.4; throwaway dirs, `rm -rf` after | 0.2 |
| `KS-MOKA-0b` unmerged extract, 3 splits | `lora_embed` 13234/13239/13240/13245/13302 = `00:26:17`–`00:37:25` | `sacct`: **00:28:41 / 00:33:17 / 00:26:17 / 00:37:25 / 00:34:37**, all COMPLETED — range exact | 0.6 |
| MokA-ZH SFT | `train_runtime` 8,635.998 s; job 12143 wall `02:39:49` | `logging/lora/MHC_zh/all_results.json` → `train_runtime 8635.9986`; `sacct` 12143 = `02:39:49`, 16 CPU, 120 G | 3.1 |
| MokA-ZH extraction (`--moka`, 3 splits) | 0.6 × ~1.15 | consistent | 0.7 |
| 3 head-seeds | job 13150 = `00:02:46` for exactly 3 runs | `sacct` 13150 = `00:02:46` COMPLETED | 0.05 |
| **total** | | **0.2+0.6+3.1+0.7+0.05 = 4.650** ≤ **cap 4.7** ✔ | **4.65** |

SFT term recomputed: 8635.998 s = 2.3989 h; ×1.2 + 0.25 = **3.1287** — the prereg rounds to 3.1; at
the unrounded 3.13 the total is 4.68, **still under the 4.7 cap**. Contingent +0.05 / +3 evals (§3.4)
is pre-declared, so the cap holds under the contingency too.

- **Resources sane and deployment-matched.** Job 1 = `gpu:a100:1`, **16 CPU / 120 G** — *identical*
  to the deployed `lora_sft.sbatch` (`--cpus-per-task=16`, `--mem=120G`, `gpu:a100:1`) that produced
  the floor adapter. Job 2 = **8 CPU / 64 G / 1 A100** — identical to the deployed
  `gen_embed_lora.sbatch` and `enc3seed_zh_b3.sbatch`. Both within the 16 CPU / 128 G / 2 GPU cap.
- **No `--time`** in either sbatch (both carry the explicit "intentionally NO --time" note). ✔
- **Sequential submission, no `--dependency`.** I grepped both files: **no `--dependency` anywhere**.
  §1.0 + DEV-4 pin manual sequential submission (job 2 only after `sacct` says job 1 `COMPLETED`), so
  submit-time aggregate demand never exceeds 16 CPU ⇒ **never two concurrent 16-CPU jobs**; the
  `afterok` 24-CPU-aggregate wedge mode is explicitly rejected. Peak instantaneous footprint 16 CPU /
  120 G / 1 GPU. ✔
- **Job-1 HateMM refusal present and hard.** `lora_sft_moka.sbatch:32-35`: the `HateMM)` case prints
  the `KS-MOKA-1` auto-defund message, names the three missing prerequisites, and `exit 2`. Unknown
  datasets also `exit 2`. ✔
- **Disk.** `df` now: **518 G avail, 97 % used** (prereg said 521 G / 97 % — consistent drift). Job 1
  preflights **≥ 25 G** (`exit 3` otherwise) and runs `disk_guard.sh` (threshold 250 G — this is the
  standing "256 G watch"; it is a *reclaim-above* trigger and is a no-op at current usage). MokA
  adapter set ≈ 1.0 G: I confirmed the deployed dir is **678 M** with 3 epoch checkpoints + the final
  save (4 saves), so 1.4489 × 161 MB × 4 ≈ 0.94 G — DEV-E's figure is right. **`save_total_limit` is
  deliberately unset** to keep the yaml diff at one line; DEV-E discloses this and offers two
  remedies. At 518 G free, 1.0 G is not binding, so I do **not** require the amendment (adding
  `save_total_limit: 1` would change sha **G** and fire §4.6). Recommended zero-yaml-change hygiene,
  as DEV-E already states: prune `logging/lora/MHC_zh_moka/checkpoint-*` **after** job 2's extraction
  completes.
- **Collision surfaces absent — I re-checked every one.** `logging/lora/MHC_zh_moka` ✗ absent;
  `logging/Retrieval/MHC_zh/RAC_video_moka*` ✗ absent; `logging/_smoke_moka` ✗ absent; no
  `*moka*` or `*-um*` file in `data/CLIP_Embedding/MHC_zh/`; no `*moka*` trainlog in `slurm/logs/`.
  Job 1 additionally aborts (`exit 4`) if the MokA adapter file already exists; job 2 aborts (`exit
  2`) if either input adapter is missing. `--group_name RAC_video_moka` is a fresh group ⇒
  `--force False` cannot trip the `run_rac.py` hard-abort. ✔
- **Gate order sane.** Job 2 runs Stage A0 (`KS-MOKA-0b`, 0 test-touch) → Stage A1 → **Stage S shape
  sanity, which exits non-zero before any of the 3 head runs** → Stage B. The budgeted test reads are
  the last thing that happens. ✔ `bash -n` on both sbatch: **SYNTAX_OK**; `py_compile` on all four
  Python artifacts: **PASS**.

---

## V8 — Sha ledger: **PASS** (all recomputed on disk, all match)

Every §5.1/§5.3 artifact re-hashed by me. **7 / 7 match byte-for-byte:**

```
9b0fc502193b0521f1359978f85b35b6ce98034d1371f174315c8f1a219a8386  src/moka/routed_lora.py                                  (A) ✔
fae40487263fd7f65e2d0566205e57d3a2caf1b9d1477c693c7cdfdc891c9749  src/moka/train_moka.py                                   (B) ✔
75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399  src/utils/generate_VideoMLLM_embedding_lora_HF.py        (C) ✔
843dace46611d3a2f5e6942a5f0e780fe8b2fb623a3c969949c142bd559d7793  scripts/analysis/moka_smoke.py                           (D) ✔
df3c9a6a8721f5d97935e4b87137732726329877c14ee8635902976eaf70e38b  scripts/slurm/lora_sft_moka.sbatch                       (E) ✔
fd1b7f295cdb7106e0e64629cb1a2391355f9daad4ab0f2ad773292f48b31bde  scripts/slurm/moka_extract_head.sbatch                   (F) ✔
51b883e9f0a78c26d9b4af185b54a4703a250a3cab4c947756782c6c8fe49764  RA-HMD/…/my_configs/hatevideo/mhc_zh_qwen25vl_lora_moka_sft.yaml  (G) ✔
```

Prereg itself (the value §5.1 row P leaves for the reviewer):
`dc3f1078a89fc2e1de30c870103c2b7f2986fd419698d6c49b5b9ec0966c53f8  refine-logs/MOKA_PREREG.md`.

**§5.1 completeness:** the 7 rows are exactly the 7 files touched by commit `7c4a22e` (`git show
--name-status`: 6 × `A`, 1 × `M`) plus the gitlink-resident yaml. **Nothing changed on disk is
missing from the ledger, and nothing in the ledger is absent from disk.**

**Reused-unchanged machinery (§5.2) — verified:**

```
b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py                 UNCHANGED ✔
4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad  scripts/slurm/enc3seed_zh_b3.sbatch   ✔
e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f  scripts/slurm/lora_sft.sbatch         ✔
2f2429fbd8f6b0b82fba173a4efc5f12ae75cf5e5bce791c3819663c5f1439ea  RA-HMD/…/mhc_zh_qwen25vl_lora_sft.yaml ✔
```

- `run_rac.py` = `b85eb72a…` and I confirmed this is **the same string frozen in
  `ZHPROMPT_PREREG.md` §5.2 (line 462)** — not merely asserted, grep-verified.
- `git status --porcelain src/run_rac.py src/model/loss.py src/model/classifier.py
  src/utils/retrieval.py` → **empty (CLEAN)**. ✔
- **`run_one()` byte-identity:** `sed -n '42,83p' enc3seed_zh_b3.sbatch` vs `sed -n '112,153p'
  moka_extract_head.sbatch` → `diff` **empty**, both blocks hash to
  `286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101`. **Byte-identical.** ✔
  *(Ledger nit: §4.1(d) attaches the sha `4379224671…` to the phrase "lines 42-83"; that value is the
  **whole-file** sha of `enc3seed_zh_b3.sbatch` — correct as a file pin, which I verified — while the
  42-83 **block** sha is `286a9e44…`. Both are recorded in `MOKA_FREEZE.md` so the executor can check
  either.)*
- F0.7's provenance for 40,370,176 independently confirmed: the banked adapter sums to exactly
  **40,370,176** params, and `logging/slurm/lora_sft_13233.out:308` reads `trainable params:
  40,370,176 || all params: 8,332,536,832 || trainable%: 0.4845`. ✔
- Gitlink-yaml freeze precedent confirmed to exist as cited (`CAND2_FREEZE.md:21-22,35-36`,
  `LORA_HATEMM_FREEZE.md:18,24`). ✔

---

## Additional finding (documentation-only, non-blocking)

**Stale docstring in a frozen artifact.** `src/moka/routed_lora.py:34-35` still reads *"per-token rank
and FLOPs are IDENTICAL (exactly one `A` fires per token)"*. Under the frozen **dense** formulation
both `A`s are evaluated and one output is discarded, so this contradicts the prereg's own §F0.7
amendment and DEV-1 (per-token rank identical; compute ≈ +1 %). The same file's `_ROUTE_IMPL` comment
(lines 60-61) states it correctly ("~2x the (tiny) `A` FLOPs"), so the file is internally
inconsistent, not wrong throughout. **Recorded, not required:** fixing it edits artifact **A** and
therefore fires **§4.6** (re-freeze + re-review) for a comment-only change. The prereg text — which
is what binds the write-up — is already correct and amended. If the paper ever quotes the code
comment, use §F0.7's wording.

---

## What I did NOT do

No GPU, no SLURM, no Modal, no job submitted, no held-out test metric read, no `state/` or
`research-wiki/` mutation, no push, no edit to any of the 7 frozen artifacts or to
`MOKA_PREREG.md`. §5.1 row P is filled in `MOKA_FREEZE.md` rather than inside the prereg, because
writing a file's own sha256 into that file is self-invalidating — this follows the
`CAND2_FREEZE.md` / `LORA_HATEMM_FREEZE.md` house pattern where the freeze document carries the
prereg sha.

---

## RULING

**APPROVED-WITH-NOTES.** Proceed to the §3.11 gate order: G-repro sha re-verify → codex gate (§4.5)
→ CPU smoke all-PASS → GPU smoke incl. `KS-parity` bit-exact → job 1 → post-run asserts +
`KS-MOKA-2` → job 2 → `KS-MOKA-0b` → shape sanity → the 3 budgeted test reads → `KS-MOKA-1` →
`KS-MOKA-3` → FORMAL → `KS-regression`.

**Conditions of approval (write-up-binding, no code change required):**
1. **N1** — report `KS-MOKA-2` as a **non-degeneracy floor**, never as "routing is real"; rest any
   routing-is-active claim on `fallback_calls == 0` and `KS-MOKA-3`.
2. **N2** — restate **F0.2** and **F0.6** at verdict time, as §7 already mandates.
3. **N4** — state which `KS-MOKA-1` parse was applied.
4. **N5** — if both streams move, say so explicitly rather than forcing one of the three readings.
5. **N3 / N6** — if DEV-1 or F0.6's quantiles are restated in the paper, carry the shape-dependence
   (N3) and the token-counting convention (N6) with them.

Any **code** fix (including the N1 recommendation or the stale-docstring fix) changes an artifact sha
and **fires §4.6**: re-freeze + a new independent 0-context review before submit.
