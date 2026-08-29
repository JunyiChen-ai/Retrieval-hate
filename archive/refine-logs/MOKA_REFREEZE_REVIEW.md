# MOKA-ZH — REFREEZE-1 INDEPENDENT 0-CONTEXT RE-REVIEW

**Reviewer:** independent 0-context re-freeze reviewer (`/data/jehc223/RGCL`, conda `HateVideo`).
Nothing below is taken on the word of `MOKA_REFREEZE_FIX.md`, `MOKA_SUBMIT_RECORD.md` or the commit
message; every claim was re-derived with the reviewer's own harnesses (scratchpad, outside the repo).
**Review timestamp (`date -u`):** `Sat Jul 25 14:41 UTC 2026` (local `Sun Jul 26 02:41 NZST`).
**Env re-read at review time:** `torch 2.6.0+cu124`, `peft 0.14.0`, `transformers 4.49.0`,
`accelerate 1.5.2`.

**Under re-review:** commit `72a947b` (amended **A** `src/moka/routed_lora.py`,
**D** `scripts/analysis/moka_smoke.py`, **E** `scripts/slurm/lora_sft_moka.sbatch`) against
`MOKA_FREEZE.md §7 REFREEZE-1`, triggered by prereg §4.6 after the §4.5 codex gate returned
`GATE: BLOCK` (commit `9c4adba`).

## VERDICT: **REFREEZE-1 APPROVED** (authorization RESTORED, still gated on the codex re-gate + the owed §4.4 GPU smoke)

Both P1s are fixed, each fix is verified by the reviewer's own runtime evidence on the deployed
class, each is covered by a check that **demonstrably fails on the frozen pre-fix code**, the
amendment surface is confined to A/D/E, and no scientific clause (§2 floors, §3 KS bars, §3.11 gate
order, §4.4/§4.5 gate content, the 3-evaluation test-touch budget) is touched. Three non-blocking
notes are recorded (**N7**, **P3-2**, **OBS-1**); none requires a further code edit.

**ZERO GPU / SLURM / Modal spent at this review. NO job submitted. NO `sbatch`/`scancel` issued. NO
held-out test metric read. NO `state/` or `research-wiki/` mutation. NO frozen artifact edited. NO
vendored LLaMA-Factory line edited. Not pushed.** Cost: CPU login-node work only (`sha256sum`,
`git show`, `bash -n`, `py_compile`, four reviewer harnesses, two CPU smoke runs, read-only
`squeue`). Queue observed read-only throughout: the only job under this account is
`13531 lsmi_power_cpu PENDING 16 CPU` (another executor's).

---

## R1 — Diff confinement: **PASS**

`git show --name-only 72a947b` returns exactly five paths: the two re-freeze documents
(`refine-logs/MOKA_FREEZE.md`, `refine-logs/MOKA_REFREEZE_FIX.md`) and the three code artifacts
**A**, **D**, **E**. **No other file is in the commit.**

| claim | reviewer's own check | result |
|---|---|---|
| only A/D/E changed | `git show --name-only 72a947b` | ✔ A, D, E + the 2 docs, nothing else |
| **B**, **C**, **F**, **G** byte-identical to the original freeze | `sha256sum` on disk vs `MOKA_FREEZE.md §1` | ✔ 4/4 MATCH (see R5) |
| prereg NOT edited | `git show --stat 72a947b -- refine-logs/MOKA_PREREG.md` → **empty**; `sha256sum` on disk | ✔ `dc3f1078a89fc2e1de30c870103c2b7f2986fd419698d6c49b5b9ec0966c53f8` — re-hashed, MATCH |
| working tree == commit for all MokA artifacts | `git status --porcelain` over `src/moka scripts/analysis/moka_smoke.py scripts/slurm/{lora_sft_moka,moka_extract_head}.sbatch src/utils/generate_VideoMLLM_embedding_lora_HF.py refine-logs/MOKA_*` | ✔ **empty** — no uncommitted drift |

**Content of the diff, re-read line by line (not summarised from the fix doc):**

- **A** — exactly 2 hunks. (i) `MokaLinear.forward:186-196`: `adapter_names = kwargs.pop(...)` with
  verbatim re-insertion when non-`None` (the P2-2 fix); (ii) `install_moka:295-312`: hook site moved
  from `peft_model.get_base_model()` to `peft_model` itself, idempotence check moved from `getattr`
  to `__dict__`, plus 16 comment lines. **The routing algebra, the mask definition, the strict
  guard, dtype discipline, the shared-`B` application, the merge guards, `load_moka_a_v`, and both
  `$0` readouts are untouched** (no other line of A appears in the diff).
- **D** — the diff removes exactly **two** lines (`base = m.get_base_model()` and the old
  `check("S1.hook", getattr(base, "_moka_hook", None) is not None, "")`) and adds the docstring
  paragraph, `S1.hook-on-outermost` + `S1.hook-not-on-base`, `s9_production_wrapper_class()` and its
  `main()` call. Reviewer-verified by `git show … | grep "^-"`: **no other S1–S8 assertion was
  removed or weakened**; the one replaced check was strengthened from 1 assertion to 2.
- **E** — the diff removes exactly **two** lines (`import json, os, sys` and
  `med = vals[len(vals) // 2]`) inside the STEP-3 heredoc. The `196/196/196` and `58,490,880`
  asserts, the JSON dump, the sbatch header, the disk/collision guards and STEP-1/STEP-2 are
  untouched.

---

## R2 — P1-A fix semantics on a REAL `PeftModelForCausalLM`: **PASS**

**Reviewer's own harness** (`r2_hook_surface.py`, scratchpad), deliberately *not* the fix executor's:
a tiny offline **`LlamaForCausalLM`** (`vocab 151700, 2 layers, hidden 32`) — the Qwen2.5-VL LM
family's architecture, not GPT2 — with the **deployed** `target_modules =
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` (yaml **G** `lora_target`), `r=16`,
`alpha=32`, `dropout=0.0`, `task_type=CAUSAL_LM`, `MOKA_STRICT=1`, `MOKA_ROUTE_IMPL=dense`,
`CUDA_VISIBLE_DEVICES=""`. A plain-PEFT reference forward is taken **before** `install_moka` with
`lora_B` perturbed (`std 0.02`, so the LoRA delta is not identically zero); `lora_A_v` is then tied
to `lora_A`, which makes the routed forward algebraically the plain-PEFT forward.

The pre-fix module was obtained by the reviewer as `git show 9c4adba:src/moka/routed_lora.py` into
the scratchpad and **re-hashed to `9b0fc502…19a8386`** = the original freeze's **A** — i.e. the
pre-fix baseline is authentic, not a reconstruction.

```
[PREFIX ] routed_lora sha256: 9b0fc502193b0521f1359978f85b35b6ce98034d1371f174315c8f1a219a8386
[PREFIX ] wrapper class    : PeftModelForCausalLM   get_base_model(): LlamaForCausalLM
[PREFIX ] routed layers    : 14   (7 proj x 2 layers)
[PREFIX ] _moka_hook owners: ['base_model.model']
[PREFIX ] hook in outer __dict__: False | in get_base_model().__dict__: True
[PREFIX ] model(**inputs)  : RAISED RuntimeError: MokA: modality mask absent or shape-mismatched
                             at a routed layer (x=(1, 14, 32), mask=None). …
[PREFIX ]   stats          : {'hook_calls': 0, 'routed_calls': 0, 'fallback_calls': 1, 'strict': True}

[POSTFIX] routed_lora sha256: 6b7bdb6c13262cbfce81f212fcf2ed596a8b466f9ae928ba340ac0eee37c85be
[POSTFIX] _moka_hook owners: ['<self>']
[POSTFIX] hook in outer __dict__: True | in get_base_model().__dict__: False
[POSTFIX] model(**inputs)  : OK
[POSTFIX]   stats          : {'hook_calls': 1, 'routed_calls': 14, 'fallback_calls': 0, 'strict': True}
[POSTFIX]   max|delta vs plain-PEFT ref| = 0.000e+00   bit-exact=True
```

⇒ pre-fix `hook_calls = 0`, `fallback_calls = 1`, strict raise on the trainer/extractor call shape;
post-fix the hook fires **once**, **every** routed layer routes (`routed_calls == 14 == n`),
`fallback_calls == 0`, and the **whole-model logits are bit-exact** (`0.000e+00`) against the
plain-PEFT reference. **P1-A is repaired on the deployed class.**

**Mechanism re-derived from the installed libraries (not quoted from the fix doc):**
`peft/tuners/tuners_utils.py:197` `return self.model.forward(*args, **kwargs)` — a direct
`.forward()`; `peft/peft_model.py:912-920` `get_base_model()` returns exactly that `self.model` for
LoRA; `torch/nn/modules/module.py:1743-1770` `_call_impl` runs `_forward_pre_hooks` **inside
`__call__` only** and resolves `forward_call = self.forward` *after* the hooks. Contrast
`peft/peft_model.py:843-849` (generic `PeftModel.forward`) which goes through `__call__` — which is
precisely why S1–S8 stayed green on the frozen code.

### R2 supplementary — the four load-bearing sub-claims, each re-measured (`r2b_surfaces.py`)

| sub-claim | reviewer's measurement |
|---|---|
| `getattr` would have been fooled ⇒ the `__dict__` idempotence check is necessary | own probe: an attribute set **only** on `get_base_model()` is returned by `getattr(m, …)` and `getattr(m.base_model, …)` while absent from both `__dict__`s. Forwarding confirmed at `peft/peft_model.py:821-828` and **`peft/tuners/lora/model.py:368-375`** (see **P3-2**). |
| no hook stacking | outer `len(_forward_pre_hooks)` = 1 before and 1 after re-running the guard ⇒ **no duplicate registration** |
| accelerate's bf16 patch does not defeat the hook | accelerate patches the **instance attribute** `model.forward` (`accelerator.py:1462-1473`, re-read). Simulated: patched forward invoked 1×, stats `{'hook_calls': 1, 'routed_calls': 14, 'fallback_calls': 0}` ⇒ `__call__` still runs pre-hooks first |
| train-shaped call (labels + backward) | `model(**inputs, labels=…)` → `{'hook_calls': 1, 'routed_calls': 14, 'fallback_calls': 0}`, loss `11.94836`, **dead trainable params `[]`**, grad mass `A_t 9.4971e+00 / A_v 1.0486e+01 / B 6.7395e+01` |
| mask refreshed per batch, no shape carry-over | two consecutive batches `S=14` then `S=9` → stash shapes `(1,14)` then `(1,9)`, `hook_calls 1→2`, `routed_calls 14→28`, `fallback_calls 0` |

### R2 — the "two deployed call surfaces" claim: **verified independently**

| surface | reviewer's check | result |
|---|---|---|
| job 1 builds `PeftModelForCausalLM` | `llamafactory/model/adapter.py:301-305` `LoraConfig(task_type=TaskType.CAUSAL_LM, …)`; `:312 model = get_peft_model(model, peft_config)`; `:320 return model` — the **same object** the patched `get_peft_model` (artifact **B**) installs the hook on is the one handed to the trainer | ✔ |
| job 1 train step | `transformers/trainer.py:3759` `outputs = model(**inputs)` (`compute_loss`); LF `CustomSeq2SeqTrainer.compute_loss` is a bare `super()` call | ✔ `__call__` |
| job 1 eval (`eval_strategy: epoch` is ON in **G**) | `trainer.py:4525` and `trainer_seq2seq.py:352`, both `outputs = model(**inputs)`; `trainer_seq2seq.py:294-296` short-circuits to `super().prediction_step` when `predict_with_generate` is false | ✔ `__call__` |
| **no `.generate()` surface in this cell** | `predict_with_generate` **absent from the frozen yaml G** (grep: 0 hits) and `training_args_seq2seq.py:52-54` **defaults `False`** ⇒ the `self.model.generate(...)` branch at `trainer_seq2seq.py:333` is unreachable | ✔ |
| job 2 extractor | `src/utils/generate_VideoMLLM_embedding_lora_HF.py:503` `model = PeftModel.from_pretrained(...)` → `:513 install_moka(model)` → `:514 load_moka_a_v(...)`; **`model` is not rebound in the `--moka` branch** (the `merge_and_unload` rebind is in the `else`), and `:360` is `out = model(**inputs, output_hidden_states=True, use_cache=False)`. `grep "generate("` over the extractor: **0 hits** | ✔ |
| job 2 also builds `PeftModelForCausalLM` | banked `logging/lora/MHC_zh/adapter_config.json:115` `"task_type": "CAUSAL_LM"` | ✔ |

**Residual, independently reproduced and accepted:** a caller that invokes `peft_model.forward(**inputs)`
*directly* still fires no hook. Measured post-fix: `{'hook_calls': 0, 'routed_calls': 0,
'fallback_calls': 1}` + strict **raise** — i.e. loud failure, never a silent plain-LoRA null-op.
No deployed path does this (table above), and S9 asserts the invariant on that surface.

---

## R3 — P1-B: the amended STEP-3 statistic is a true median: **PASS**

The reviewer extracted the STEP-3 heredoc **byte-for-byte** out of the amended
`scripts/slurm/lora_sft_moka.sbatch` (`awk` between the `<<'PY'` and `PY` delimiters):
**31 lines, sha `a53ec99225d7fc48d137314188a3e2c9f0d1bd6ed86bd87c2a4c691c11165a29`** — **matches** the
value recorded in `MOKA_REFREEZE_FIX.md §3 E3b`. `diff` against the same block extracted from the
pre-fix `9c4adba` shows exactly the two claimed hunks and nothing else.

Both blocks were then run against the reviewer's **own** synthetic 196-layer adapter at the deployed
shapes (28 layers × the 7 projections, `r=16`, `A_v = A_t·(1+c)` ⇒ `rel_fro_diff == c` exactly;
file order randomly permuted so the block must do its own sorting; **588 tensors, 58,490,880 params**
⇒ the block's own asserts are exercised end-to-end). The two middle order statistics were pinned to
straddle the bar: `vals[97] = 0.0400`, `vals[98] = 0.0550`. Run with cwd in the scratchpad so the
block's relative `refine-logs/MOKA_KS2_routing_report.json` write landed there —
**re-verified absent from the repo afterwards.**

```
AMENDED  block: [moka_sft] KS-MOKA-2 rel : min 0.0300 median 0.0475 max 0.0650
                [moka_sft] KS-MOKA-2 median >= 0.05 ? False
PRE-FIX  block: [moka_sft] KS-MOKA-2 rel : min 0.0300 median 0.0550 max 0.0650
                [moka_sft] KS-MOKA-2 median >= 0.05 ? True

cross-check on the amended block's own JSON:
  median=0.047500 == statistics.median=0.047500 == (v[97]+v[98])/2=0.047500   equal=True
  old expression vals[len//2] = 0.055000
  KS-MOKA-2 'median>=0.05':  amended=False  old=True   DECISION FLIP = True
```

⇒ **the fix executor's decision-flip numbers (old `0.0550` vs new `0.0475`) reproduce exactly.**
*Explained delta:* my `min/max` are `0.0300 / 0.0650` where the fix doc reports `0.0303 / 0.0647` —
the two synthetic ramps differ away from the middle; the only positions that were pinned identically
by construction are `vals[97]`/`vals[98]`, which is what the statistic and the flip depend on.

**Even/odd control on the amended expression** (`n = 196` even, `n = 195` odd):

```
n=196 even=True  | old vals[len//2]=98.0000 | statistics.median=97.5000 | identical=False
n=195 even=False | old vals[len//2]=97.0000 | statistics.median=97.0000 | identical=True
```

⇒ `statistics.median` is the mean of the two middle order statistics for even `n` and the middle
element for odd `n` — a true median under both parities. The deployed `n` is **196** (the block's own
`assert n_a == n_av == n_b == 196`), i.e. the even branch. The amended code now matches prereg §3.6's
own wording ("median layer < 0.05"), which the frozen code did not.

*Interaction with note N1:* the true median is ≤ the old `vals[98]`, so the statistic moves
marginally **toward** the `< 0.05` NULL-OP trip. N1's finding stands unchanged (two independent
Kaiming draws sit at ≈ 1.41 relative-Frobenius, ~28× the bar), and N1's write-up restriction —
report `KS-MOKA-2` as a non-degeneracy floor, never as "routing is real" — is untouched by this fix.

---

## R4 — S9 blind-spot closure: **PASS (and non-vacuous)**

**(a) Amended smoke on the post-fix tree, reviewer's own run**
(`python scripts/analysis/moka_smoke.py`, repo root, `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=""`):
**`==== ALL SMOKE CHECKS PASS ====`, exit 0**, S1–S9 including S8. Every frozen number reproduced:

- `S1.hook-on-outermost` **PASS** (`registered on PeftModel`), `S1.hook-not-on-base` **PASS**
- **S2 identity control `max|Δ| = 0.000e+00` in all 6 cells**; dense-vs-gather `0.000e+00`
  (all-text, all-vision) and `1.192e-07 / 5.960e-08` (mixed) — bit-matches `MOKA_FREEZE.md §6`
- S4 `{'hook_calls': 1, 'routed_calls': 2, 'fallback_calls': 0}`, `dead=[]`,
  `rel_fro_diff = [1.4145, 1.484]`
- S5 round-trip 2/2 bit-exact + plain-reload drops `A_v` + generic-adapter refusal; S6 all 3 raises
- S7 `40,370,176 → 58,490,880 = 1.448864×`
- S8 deployed cap `2688 == 2688`, `seq 2823 = 2688 + 135` (95.2 % vision), ids `[151655]`;
  processor-default `21528 == 21528`, `21663 = 21528 + 135` (99.4 %)
- S9 all 8 sub-checks PASS, incl. `S9.identity-vs-plain-PEFT max|delta| = 0.000e+00` and
  `S9.direct-forward-no-silent-nullop raises (strict)`

**(b) The same amended smoke pointed at the PRE-FIX `routed_lora.py`.** Built as a shadow tree in the
scratchpad (`prefix_repo/scripts/analysis/moka_smoke.py` = amended **D** `bd258553…`,
`prefix_repo/src/moka/routed_lora.py` = frozen **A** `9b0fc502…`), run with `--skip-mask`.
**The repo tree was not touched.** Result:

```
  [FAIL] S1.hook-on-outermost   registered on PeftModel
  [FAIL] S1.hook-not-on-base
  [PASS] S9.class-is-PeftModelForCausalLM PeftModelForCausalLM
  [PASS] S9.install-count routed 2 lora.Linear layers
  [FAIL] S9.hook-on-wrapper …
  RuntimeError: MokA: modality mask absent or shape-mismatched at a routed layer
                (x=(1, 12, 32), mask=None). …
  EXIT=1
```

⇒ **the amended CPU gate would have blocked the frozen code**: 3 explicit FAILs plus an uncaught
strict raise, exit status **1**. The check is not vacuous, and the `KS-MOKA-0` gate now covers the
deployed wrapper class. *(See **OBS-1**: on the pre-fix code S9 aborts with a traceback instead of
printing the `FAILURES: …` summary. Exit status is 1 either way, so the "smoke must exit 0" gate
discipline is intact; cosmetic only.)*

---

## R5 — Sha ledger: **PASS (12/12 re-hashed by the reviewer on disk)**

```
P  dc3f1078a89fc2e1de30c870103c2b7f2986fd419698d6c49b5b9ec0966c53f8  refine-logs/MOKA_PREREG.md          UNCHANGED  (freeze §1)
A  6b7bdb6c13262cbfce81f212fcf2ed596a8b466f9ae928ba340ac0eee37c85be  src/moka/routed_lora.py             = REFREEZE-1 (was 9b0fc502…)  395 L
B  fae40487263fd7f65e2d0566205e57d3a2caf1b9d1477c693c7cdfdc891c9749  src/moka/train_moka.py              UNCHANGED  (freeze §1)
C  75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399  src/utils/generate_VideoMLLM_embedding_lora_HF.py  UNCHANGED
D  bd2585536e7982e021ead4974910fb7df32498eee57625cbe4331d7bbf46c4ef  scripts/analysis/moka_smoke.py      = REFREEZE-1 (was 843dace4…)  422 L
E  020dd10bd7cfcabb381d76ea44441cbd9607d2db0b98ef92e2388e71482745e6  scripts/slurm/lora_sft_moka.sbatch  = REFREEZE-1 (was df3c9a6a…)  116 L
F  fd1b7f295cdb7106e0e64629cb1a2391355f9daad4ab0f2ad773292f48b31bde  scripts/slurm/moka_extract_head.sbatch  UNCHANGED
G  51b883e9f0a78c26d9b4af185b54a4703a250a3cab4c947756782c6c8fe49764  RA-HMD/…/mhc_zh_qwen25vl_lora_moka_sft.yaml  UNCHANGED
```

Reused-unchanged pins (`MOKA_FREEZE.md §2`), re-hashed — **4/4 MATCH**:

```
b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py
4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad  scripts/slurm/enc3seed_zh_b3.sbatch
e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f  scripts/slurm/lora_sft.sbatch
2f2429fbd8f6b0b82fba173a4efc5f12ae75cf5e5bce791c3819663c5f1439ea  RA-HMD/…/mhc_zh_qwen25vl_lora_sft.yaml
```

§4 VOID-ON-EDIT items 1–5, all re-verified by the reviewer:

| item | result |
|---|---|
| 1 `src/run_rac.py` = `b85eb72a…` | ✔ |
| 2 `loss.py` / `classifier.py` / `retrieval.py` / `run_rac.py` git-clean | ✔ `git status --porcelain` **empty** |
| 3 gitlink `RA-HMD/LLAMA-FACTORY-Ver202512` | ✔ `git ls-files -s` → `160000 a912747c408b3c661b4029ecf1d88b9d91c7f1a8` — **unchanged**; inner `git diff --stat HEAD` **empty**; only untracked entries are `.cuda_home_shim/` and artifact **G** ⇒ **zero vendored lines edited** |
| 4 `run_one()` byte-identity `moka_extract_head.sbatch:112-153` vs `enc3seed_zh_b3.sbatch:42-83` | ✔ `diff` **empty**, both blocks sha `286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101` |
| 5 §4.7 collision surfaces | ✔ **all ABSENT**: `logging/lora/MHC_zh_moka`, `logging/_smoke_moka`, `logging/Retrieval/MHC_zh/RAC_video_moka*`, `data/CLIP_Embedding/MHC_zh/*moka*`, `…/*-um*`, `slurm/logs/*moka*`, `logging/slurm/lora_sft_moka*`, `refine-logs/MOKA_KS2_routing_report.json` |

Also re-run: `bash -n` both sbatch **SYNTAX_OK**; `py_compile` on all four Python artifacts **PASS**.

---

## R6 — P2 / P3 dispositions: **SOUND**

**P2-2 (FIXED) — inert for the deployed sites.** Reviewer's own control (`r6_p22.py`), pre-fix vs
post-fix module, same routed layer, same input, mask set:

```
[PREFIX ] no kwarg (DEPLOYED shape)           -> OK  (2,5,32)   identical-to-no-kwarg=True
[PREFIX ] adapter_names=None                  -> TypeError: Linear.forward() got an unexpected keyword argument 'adapter_names'
[PREFIX ] adapter_names=['default','default'] -> OK (delegated to upstream)
[POSTFIX] no kwarg (DEPLOYED shape)           -> OK  (2,5,32)   identical-to-no-kwarg=True
[POSTFIX] adapter_names=None                  -> OK  (2,5,32)   identical-to-no-kwarg=True
[POSTFIX] adapter_names=['default','default'] -> OK (delegated to upstream)
```

The deployed shape is bit-identical across the fix; `None` no longer `TypeError`s; a non-`None` list
still delegates to upstream (the value is re-inserted verbatim before `super().forward`, and
upstream `peft/tuners/lora/layer.py:598` pops it itself). Semantics of the popped kwarg on the routed
path are unchanged: `LoraLayer._check_forward_args` returns immediately when `adapter_names is None`,
so removing an always-`None` key changes nothing. **`grep -rn "adapter_names"` over the extractor,
`train_moka.py`, both sbatch and the entire `llamafactory` tree: 0 hits** ⇒ no deployed call site
passes the kwarg. Fix confirmed inert-and-correct.

**P2-4 (ACCEPTED) — the `set_seed` ordering claim holds.** Re-read at source:
`transformers/trainer.py:455` `enable_full_determinism(...) if args.full_determinism else
set_seed(self.args.seed)` inside `Trainer.__init__`, and LLaMA-Factory `train/sft/workflow.py:52`
builds the model (`load_model`, which is where `install_moka`'s `A_v` draw happens) **before**
`:82 trainer = CustomSeq2SeqTrainer(...)`. So the CUDA stream is reseeded after MokA's init and
before any training RNG is consumed ⇒ the CPU-only `fork_rng(devices=[])` caveat is non-blocking as
stated. CPU-side neutrality independently re-confirmed by the smoke reproducing the frozen
`S1.A_v-differs` and `S4 rel_fro_diff = [1.4145, 1.484]` bit-for-bit.

**P2-5 (ACCEPTED) — surfaces genuinely inactive.** Re-read `moka_extract_head.sbatch:41-45`: the only
guard is on the two adapter **inputs** (`exit 2` on a missing `adapter_model.safetensors`); there is
indeed **no** guard on the `-um` / `-moka_HF` feature-cache **outputs**. Both output families are
re-verified **ABSENT on disk right now** (R5 item 5), **F** is byte-identical to the original freeze,
and the submit executor re-checks §4.7 at S0 by standing instruction. Accepting it keeps the
re-review diff minimal; the risk is a repeat-execution overwrite, which the collision re-check
catches. Disposition accepted.

**P2-1, P2-3 (ACCEPTED, not fixed)** — both would add new code to hot/extraction paths (a new
`ok`-conjunction raise surface, and inspection of PEFT's `strict=False` load result), i.e. neither is
inert; the mandate confined the amendment to the P1s + the blind spot + one trivial P2. P2-1's
premise re-checked: the 7 deployed targets are `nn.Linear` projections whose `base_layer` output is
`[B,S,out]` by construction, and S2/S9 verify bit-exact agreement with upstream on exactly that path.
P2-3's residual re-checked: `load_moka_a_v` **does** raise unless every `.lora_A_v.` key maps to an
existing module with matching shape *and* every routed layer received a tensor (S5 covers both, plus
the generic-adapter refusal); the unenforced part is upstream PEFT behaviour shared with every LoRA
run in this project. Both carried, unresolved, to the codex re-gate. **Accepted.**

**P3-1 (ACCEPTED, pre-existing)** — `routed_lora.py:34-35` still says "FLOPs are IDENTICAL" against
§F0.7/DEV-1's "rank identical, compute ≈ +1 %". Documentation-only; already on the freeze doc §5.
Fixing it edits **A** and re-fires §4.6 for a comment. **Left as-is, correctly.**

---

## R7 — Ruling on the prereg-wording item: **the freeze-side amendment is SUFFICIENT; no prereg edit, no new gate. One write-up condition (N7) and one re-gate instruction are attached.**

**The divergence, verbatim.** `MOKA_PREREG.md:497` (§4.5 item 2): *"**The mask pre-hook** — registered
on the base `Qwen2_5_VLForConditionalGeneration`; reads `kwargs["input_ids"]`; **never cleared at
end-of-forward** …; overwritten once per batch; correctness under `eval_strategy: epoch`."* The
amendment registers on the outer `PeftModelForCausalLM` instead.

**Ruling and its basis (each leg checked by the reviewer, not assumed):**

1. **It is a description inside the gate checklist, not a scientific clause.** §4.5 is the list of
   items the codex reviewer must inspect. It sets no bar, no threshold, no kill switch, no gate
   order, no budget. The reviewer re-read the binding sections: §2 (floors / FORMAL), §3.1–§3.7
   (`KS-MOKA-0/0b/1/2/3`), §3.11 (gate order), §4.4 (GPU smoke incl. KS-parity), §4.7 (collision) —
   **none mentions the hook's owning module, and the prereg file's sha is unchanged, so all of them
   are byte-identical.**
2. **§4.6 is the clause that governs this exact situation, and it prescribes the freeze side.** It
   says the affected artifact shas change, **the freeze block (§5.3) MUST be re-issued**, and a new
   independent 0-context review is re-run. It does **not** call for editing the prereg — and it
   cannot, because the prereg's own sha is pinned as row **P** in the freeze document (the house
   pattern; editing the prereg would void row P and the entire freeze). The mechanism is
   supersession-by-re-issued-freeze, and that is what REFREEZE-1 §7.1 does.
3. **The prereg already contains values that §4.6 expects to be superseded.** `MOKA_PREREG.md:552`
   pins **E** at `df3c9a6a…` and `:256` describes **A** as "372 L" — both now stale for exactly the
   §4.6-anticipated reason. If stale *shas and line counts* are handled by re-issuing the freeze
   block, an *implementation-site description* in the same document is a fortiori handled the same
   way. Treating item 2 differently would imply the prereg must be edited on every code fix, which
   §4.6 explicitly forbids by design.
4. **Every other assertion in item 2 survives the move, and the reviewer measured each:** reads
   `kwargs["input_ids"]` ✔ (the deployed calls are `model(**inputs …)`, so `input_ids` is in the
   outer `kwargs`); never cleared at end-of-forward ✔ (unchanged code; and the DEV-F
   gradient-checkpointing rationale at `MOKA_PREREG.md:766-769` names no module, so it is unaffected);
   overwritten once per batch ✔ (measured: two batches → `hook_calls 1→2`, stash reshaped `(1,14)`→`(1,9)`);
   correctness under `eval_strategy: epoch` ✔ (the eval path also enters via `__call__`,
   `trainer.py:4525` / `trainer_seq2seq.py:352`). **Only the module name changed, and it changed
   because the named module was proven unhookable.**
5. **Direction of the change is toward the prereg's intent, not away from it.** Item 2's purpose is
   that the mask reaches the routed layers; on the named module it demonstrably did not (R2 PREFIX).
   The same is true of P1-B: the amended statistic now matches §3.6's own word "median", which the
   frozen code did not.

**Therefore: the REFREEZE-1 note is sufficient. No prereg amendment. No new gate, bar or budget.**
Two conditions attach:

- **N7 (NEW — binding on the write-up and on the re-gate).** Any restatement of §4.5 item 2 — in the
  verdict, the method section or the re-gate brief — must state that the modality-mask
  forward-pre-hook is registered on the **outer `PeftModelForCausalLM` wrapper**, not on the base
  `Qwen2_5_VLForConditionalGeneration`, citing `MOKA_FREEZE.md §7 REFREEZE-1` and
  `MOKA_REFREEZE_FIX.md §2`. The prereg text is superseded on this point and must never be quoted
  bare. **The codex re-gate must be pointed at the amended site explicitly**, otherwise item 2 is
  reviewed against a spec the gate itself invalidated.
- **P3-2 (NEW, non-blocking, documentation-only).** The comment at `routed_lora.py:308-310` and
  `MOKA_REFREEZE_FIX.md §2` leg 4 cite the attribute-forwarding `__getattr__` as
  `tuners_utils.py:368-375`. The reviewer checked: `peft/tuners/tuners_utils.py` contains **no**
  `__getattr__` at all (`:368-375` is `_check_merge_allowed`); the correct file is
  **`peft/tuners/lora/model.py:368-375`** — the line range is right, the filename is wrong. The
  substantive claim is **true** and was re-measured (R2 supplementary). Fixing the comment edits **A**
  and re-fires §4.6 for a typo; **left as-is**, same disposition as P3-1. Recorded so the write-up
  and the re-gate carry the correct citation.

---

## Non-blocking observation

- **OBS-1.** On the pre-fix code, `S9`'s `m(**inputs)` at `moka_smoke.py:379` is outside a
  `try/except`, so the smoke aborts with a traceback instead of printing the
  `==== FAILURES: … ====` summary. Exit status is `1` in both cases, so the "CPU smoke must exit 0"
  gate discipline is preserved and the closure claim is unaffected. Cosmetic; a fix would edit **D**
  and re-fire §4.6. **No action required.**
- **OBS-2 (pre-existing, unchanged by the fix).** `install_moka` is not idempotent at the
  layer-conversion level: a second call on an already-installed model raises
  `"MokA install: found ZERO peft lora.Linear layers to route."` at the `n == 0` check, which sits
  **before** the hook block in both the pre-fix and post-fix code. Both deployed paths install
  exactly once (job 1 inside the patched `get_peft_model`, job 2 once at extractor `:513`). The
  `__dict__` guard itself was exercised directly and does **not** stack hooks (R2 supplementary).
  Behaviour identical pre- and post-fix ⇒ not introduced by this amendment.

---

## RE-ISSUED FREEZE BLOCK (REFREEZE-1) — **AUTHORISED**

```
FROZEN dc3f1078a89fc2e1de30c870103c2b7f2986fd419698d6c49b5b9ec0966c53f8  refine-logs/MOKA_PREREG.md
A      6b7bdb6c13262cbfce81f212fcf2ed596a8b466f9ae928ba340ac0eee37c85be  src/moka/routed_lora.py             (395 L)
B      fae40487263fd7f65e2d0566205e57d3a2caf1b9d1477c693c7cdfdc891c9749  src/moka/train_moka.py
C      75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399  src/utils/generate_VideoMLLM_embedding_lora_HF.py
D      bd2585536e7982e021ead4974910fb7df32498eee57625cbe4331d7bbf46c4ef  scripts/analysis/moka_smoke.py      (422 L)
E      020dd10bd7cfcabb381d76ea44441cbd9607d2db0b98ef92e2388e71482745e6  scripts/slurm/lora_sft_moka.sbatch  (116 L)
F      fd1b7f295cdb7106e0e64629cb1a2391355f9daad4ab0f2ad773292f48b31bde  scripts/slurm/moka_extract_head.sbatch
G      51b883e9f0a78c26d9b4af185b54a4703a250a3cab4c947756782c6c8fe49764  RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/mhc_zh_qwen25vl_lora_moka_sft.yaml
```

**VOID-ON-EDIT stands unchanged.** `MOKA_FREEZE.md §4` items 1–5 and prereg §4.6 apply verbatim to
this block: any byte change to **P** or **A–G** voids the authorization, and any further code fix
forced by the re-gate or either smoke re-fires §4.6 (amend → re-hash → re-issue → new independent
0-context review). `MOKA_FREEZE.md §2`'s 4 reused pins and the `run_one` block sha `286a9e44…` remain
verify-only.

**Conditions carried into the verdict:** **N1–N6** unchanged (§5 of the freeze doc) — in particular
N1 (`KS-MOKA-2` is a non-degeneracy floor, never "routing is real") and N2 (restate F0.2 and F0.6) —
**plus N7** (hook-site supersession, above). Documentation-only **P3-1** and **P3-2** carry forward
unfixed. **P2-1, P2-3, P2-4, P2-5 remain open** and must be re-put to the codex re-gate.

**Still owed before any `sbatch`, in this order:**

1. **Codex re-gate** (prereg §4.5, **all 7 items**, on the amended **A/D/E**) returning `GATE: PASS`
   — with N7's instruction that item 2's hook site has moved.
2. **§4.4 GPU smoke**, including the **KS-parity bit-exact** leg (unmerged re-extraction vs the
   banked cache) and `fallback_calls == 0` under `MOKA_STRICT=1` — the leg the CPU gate cannot cover.
3. Submit-time `sha256sum` re-verification of **P + A–G** against the block above, plus §4 items 1–5.
4. The standing infra rule: job 1 needs 16 CPU; never two concurrent 16-CPU jobs (`13531
   lsmi_power_cpu` currently holds the 16-CPU budget).

---

**Required statements.** ZERO GPU / SLURM / Modal spent. NO job submitted, cancelled or released. NO
held-out test metric produced or read — the 3 budgeted test evaluations remain **unspent**. NO
`state/` mutated. NO `research-wiki/` mutated. NO frozen artifact edited by this review. NO vendored
LLaMA-Factory line edited. All reviewer harnesses and the shadow pre-fix tree live in the session
scratchpad, outside the repo. Not pushed.
