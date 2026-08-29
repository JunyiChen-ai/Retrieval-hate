# MOKA-ZH — REFREEZE-1 FIX RECORD (codex `GATE: BLOCK` → amended code, pending re-review)

**Executor:** MokA re-freeze fix executor (`/data/jehc223/RGCL`, conda `HateVideo`).
**Trigger:** `refine-logs/MOKA_SUBMIT_RECORD.md` (commit `9c4adba`) — the mandatory §4.5 codex gate
returned **`GATE: BLOCK`** with **2 P1 findings**, both runtime-confirmed by the submit executor.
Prereg **§4.6 fires**: the affected artifacts must be amended, their shas change, the freeze block
must be re-issued and a **new independent 0-context review + a codex re-gate** must run before submit.
**Scope of this pass:** the 2 P1s + the smoke blind spot that hid P1-A + exactly one trivial P2.
**ZERO GPU / SLURM / Modal. NO job submitted. NO test metric read. NO `state/` or `research-wiki/`
mutation. Not pushed. `refine-logs/MOKA_PREREG.md` NOT edited (sha unchanged).**
**Fix timestamp (`date -u`):** `Sat Jul 25 14:06:20 UTC 2026`. Env at fix time: `peft 0.14.0`,
`torch 2.6.0+cu124`, `transformers 4.49.0`, `accelerate 1.5.2`.

---

## 1. Findings recap (as ruled by the gate) and what was done

| # | finding | disposition | artifact |
|---|---|---|---|
| **P1-A** | mask forward-pre-hook registered on `peft_model.get_base_model()` never fires on the production `PeftModelForCausalLM` (its call chain reaches the base model through a **direct `.forward()`**, and `nn.Module` hooks fire in `__call__` only) ⇒ `hook_calls=0`, `fallback_calls=1`, `MOKA_STRICT` raise on batch 1 | **FIXED** — hook moved to the **outermost** module (`peft_model` itself) | **A** `src/moka/routed_lora.py` |
| **P1-B** | `KS-MOKA-2` reported `vals[len(vals)//2]` over **196** values = `vals[98]`, an upper-neighbour order statistic, not the median | **FIXED** — `statistics.median(vals)` | **E** `scripts/slurm/lora_sft_moka.sbatch` |
| **blind spot** | S1–S8 build their `LoraConfig` **without `task_type`** ⇒ generic `PeftModel` (base reached via `__call__`), so the smoke was structurally blind to P1-A | **CLOSED** — new **S9** on a real `PeftModelForCausalLM` | **D** `scripts/analysis/moka_smoke.py` |
| **P2-2** | explicitly passed `adapter_names=None` was not popped and reached `nn.Linear` ⇒ `TypeError` | **FIXED** (trivial, inert; matches upstream `peft/tuners/lora/layer.py:598`) | **A** |
| P2-1, P2-3, P2-4, P2-5, P3 | see §5 | **ACCEPTED, documented, NOT touched** | — |

---

## 2. Fix design for P1-A — which module is hooked, and why

**Chosen site: the outermost object the caller invokes — the `PeftModel` wrapper itself**
(`install_moka`'s own `peft_model` argument), `src/moka/routed_lora.py:311-312`.

Rationale, each leg verified in this pass against the installed libraries:

1. **Hooks only fire in `__call__`.** `nn.Module._call_impl` runs `_forward_pre_hooks`; a direct
   `mod.forward(...)` bypasses it. Any module whose `.forward` is called directly is unhookable —
   so the fix cannot be "hook a different inner module", it must be the module that is entered
   through `__call__`.
2. **The peft chain that broke it.** `PeftModelForCausalLM.forward` → `self.base_model(...)`
   (`peft/peft_model.py:1704-1719`) = `LoraModel.__call__` → `BaseTuner.forward`
   (`peft/tuners/tuners_utils.py:196-197`) → **`return self.model.forward(*args, **kwargs)`** —
   the direct call. `get_base_model()` returns exactly that `self.model`, hence hook_calls = 0.
3. **Both deployed call sites enter the wrapper via `__call__`** (this is what makes ONE
   registration sufficient — verified by reading the deployed code, not assumed):
   * **job 1 (SFT).** `transformers/trainer.py:3759` `outputs = model(**inputs)` in
     `compute_loss`; eval loop `trainer.py:4525` and `trainer_seq2seq.py:352`, both
     `outputs = model(**inputs)`. LLaMA-Factory's `CustomSeq2SeqTrainer.compute_loss`
     (`src/llamafactory/train/sft/trainer.py:116-117`) is a bare `super()` call and its
     `prediction_step` (`:120-143`) delegates to `super().prediction_step`. The frozen yaml **G**
     does not set `predict_with_generate` (default OFF), so the `self.model.generate(...)` branch
     (`trainer_seq2seq.py:333`) is **not reachable** in this cell — there is no `.generate()`
     surface to cover. `accelerate` (1.5.2) patches the *instance attribute* `model.forward`
     (`accelerator.py:1463-1473`) for bf16 autocast; `__call__` still runs pre-hooks before
     dispatching to that patched `forward`, so the hook is unaffected.
   * **job 2 (extraction).** `src/utils/generate_VideoMLLM_embedding_lora_HF.py:360`
     `out = model(**inputs, output_hidden_states=True, use_cache=False)`, where `model` is the
     `PeftModel` returned by `from_pretrained` at `:503` and left **unmerged** under `--moka`
     (`:504-515`). The extractor never calls `.generate()` (`:387` "we do NOT generate").
   The two surfaces are therefore the *same* class entered the *same* way; artifact **C** did not
   need to change (its sha is unchanged, §4).
4. **Idempotence check uses `__dict__`, not `getattr`.** `PeftModel.__getattr__`
   (`peft/peft_model.py:821-828`) and `BaseTuner.__getattr__` (`tuners_utils.py:368-375`) forward
   missing attributes **down** to the wrapped model, so `getattr(peft_model, "_moka_hook", None)`
   would report a handle owned by an inner module. Measured in E1 pre-fix: with the hook on the
   base model, `_moka_hook` was visible through `''`, `'base_model'` **and** `'base_model.model'`.
   `"_moka_hook" not in peft_model.__dict__` is forwarding-immune.
5. **Kept identical:** the hook function, the mask definition, the never-clear-the-stash rule
   (recon §2.4 — gradient-checkpoint recompute must still see the batch mask), and the
   raise-if-strict backstop in `MokaLinear.forward:198-208` (unchanged semantics: a missing or
   shape-mismatched mask still raises under `MOKA_STRICT=1`, never a silent plain-LoRA null-op).
   Hook firing frequency is unchanged (once per outer forward = once per batch).

**Residual, stated plainly:** a caller that invokes `peft_model.forward(**inputs)` *directly*
(rather than `peft_model(**inputs)`) still fires no hook. Neither deployed path does this (leg 3).
The behaviour on that surface is now asserted by S9: it **raises** under strict rather than silently
degrading (measured, §3 E1(d)). One inherent caveat of the never-clear design, unchanged by this
fix: a hook-less surface entered *after* a successful hooked forward would reuse the previous
batch's stashed mask when the shape happens to match, instead of raising.

**Prereg-text divergence the re-reviewer must rule on (documentation-level, no threshold moves).**
`MOKA_PREREG.md §4.5 item 2` describes the hook as "registered on the base
`Qwen2_5_VLForConditionalGeneration`". That description is precisely what the gate proved
non-functional. The prereg is **frozen and was not edited**; the amended implementation registers on
the outer `PeftModelForCausalLM`. No bar, threshold, gate order, kill switch or test-touch budget is
affected.

---

## 3. Evidence (all CPU, login node, no GPU, harness in the executor scratchpad)

### E1 — P1-A reproduction and repair on a REAL `PeftModelForCausalLM`

Harness `e1_hook_surface.py`: tiny offline `GPT2LMHeadModel` (`vocab 151700, n_layer 2, n_embd 32`),
`LoraConfig(task_type=CAUSAL_LM, r=16, alpha=32, dropout=0.0, target_modules=["c_attn"])`,
`MOKA_STRICT=1`, `MOKA_ROUTE_IMPL=dense`, `CUDA_VISIBLE_DEVICES=""`. A plain-PEFT reference forward
is taken **before** `install_moka` with a perturbed `lora_B` (std 0.02, so the LoRA delta is not
identically zero); then `lora_A_v` is tied to `lora_A`, which makes the routed forward algebraically
the plain PEFT forward.

**PRE-FIX** (run against the frozen `src/moka/routed_lora.py`, sha `9b0fc502…19a8386`), verbatim:

```
(a) class returned by get_peft_model(task_type=CAUSAL_LM): PeftModelForCausalLM
    base_model: LoraModel | base_model.model: GPT2LMHeadModel
    install_moka routed layers: 2 | hook on self(PeftModel)=False | hook on named submodules=['', 'base_model', 'base_model.model']
(b/c) trainer-style  model(**inputs): RAISED RuntimeError: MokA: modality mask absent or shape-mismatched at a routed layer (x=(1, 12, 32), mask=None). Refusing to fall back to plain LoRA silently, which would make the arm a null-op. Set MOKA_STRICT=0 to allow the plain fallback.
        stats={'impl': 'dense', 'hook_calls': 0, 'routed_calls': 0, 'fallback_calls': 1, 'strict': True}
(d)   direct         model.forward(**inputs): RAISED RuntimeError: MokA: modality mask absent or shape-mismatched at a routed layer (x=(1, 12, 32), mask=None). …
        stats={'impl': 'dense', 'hook_calls': 0, 'routed_calls': 0, 'fallback_calls': 1, 'strict': True}
```

**POST-FIX** (same harness, amended `routed_lora.py`), verbatim:

```
(a) class returned by get_peft_model(task_type=CAUSAL_LM): PeftModelForCausalLM
    base_model: LoraModel | base_model.model: GPT2LMHeadModel
    install_moka routed layers: 2 | hook on self(PeftModel)=True | hook on named submodules=['']
(b/c) trainer-style  model(**inputs): OK  max|delta vs plain-PEFT reference| = 0.000e+00  bit-exact=True  stats={'impl': 'dense', 'hook_calls': 1, 'routed_calls': 2, 'fallback_calls': 0, 'strict': True}
(d)   direct         model.forward(**inputs): RAISED RuntimeError: MokA: modality mask absent … Set MOKA_STRICT=0 to allow the plain fallback.
        stats={'impl': 'dense', 'hook_calls': 0, 'routed_calls': 0, 'fallback_calls': 1, 'strict': True}
```

⇒ hook fires (`hook_calls 0 → 1`), every routed layer routes (`routed_calls 0 → 2 == n_layers`),
`fallback_calls 1 → 0`, and the whole-model forward is **bit-exact** vs the dense/plain reference
(`max|Δ| = 0.000e+00`). Leg (d) is the documented residual of §2.

### E2 — identity control and full CPU smoke, post-fix

`python scripts/analysis/moka_smoke.py` (amended **D**) → **`==== ALL SMOKE CHECKS PASS ====`, exit 0.**
S2 identity control, all 6 cells, verbatim:

```
  [PASS] S2.identity[all-text][m0]   max|delta| = 0.000e+00      [PASS] S2.dense-vs-gather[all-text][m0]   max|delta| = 0.000e+00
  [PASS] S2.identity[all-text][m1]   max|delta| = 0.000e+00      [PASS] S2.dense-vs-gather[all-text][m1]   max|delta| = 0.000e+00
  [PASS] S2.identity[all-vision][m0] max|delta| = 0.000e+00      [PASS] S2.dense-vs-gather[all-vision][m0] max|delta| = 0.000e+00
  [PASS] S2.identity[all-vision][m1] max|delta| = 0.000e+00      [PASS] S2.dense-vs-gather[all-vision][m1] max|delta| = 0.000e+00
  [PASS] S2.identity[mixed][m0]      max|delta| = 0.000e+00      [PASS] S2.dense-vs-gather[mixed][m0]      max|delta| = 1.192e-07
  [PASS] S2.identity[mixed][m1]      max|delta| = 0.000e+00      [PASS] S2.dense-vs-gather[mixed][m1]      max|delta| = 5.960e-08
```

**⇒ identity control still `0.000e+00` in all 6 cells**, and the dense-vs-gather residuals
(`1.192e-07 / 5.960e-08`) reproduce `MOKA_FREEZE.md §6` and `MOKA_SUBMIT_RECORD.md §S2` unchanged.
Other frozen numbers reproduced unchanged this pass: S4
`{'hook_calls': 1, 'routed_calls': 2, 'fallback_calls': 0}`, `dead=[]`, `rel_fro_diff = [1.4145, 1.484]`;
S5 round-trip 2/2 bit-exact + generic-adapter refusal; S6 all three raises; S7
`40,370,176 → 58,490,880 = 1.448864×`; S8 `2688 == 2688`, `seq 2823 = 2688 + 135` (95.2 % vision),
masked ids `[151655]`, processor-default `21528 == 21528`.

New S1/S9 lines, verbatim:

```
  [PASS] S1.hook-on-outermost registered on PeftModel
  [PASS] S1.hook-not-on-base
S9 — DEPLOYED PeftModelForCausalLM wrapper class (closes the P1-A blind spot)
  [PASS] S9.class-is-PeftModelForCausalLM PeftModelForCausalLM
  [PASS] S9.install-count routed 2 lora.Linear layers
  [PASS] S9.hook-on-wrapper hook owner = PeftModelForCausalLM (NOT get_base_model() = GPT2LMHeadModel)
  [PASS] S9.hook-fires-on-model(**inputs) {'impl': 'dense', 'hook_calls': 1, 'routed_calls': 2, 'fallback_calls': 0, 'strict': True}
  [PASS] S9.routed-calls {'impl': 'dense', 'hook_calls': 1, 'routed_calls': 2, 'fallback_calls': 0, 'strict': True}
  [PASS] S9.no-fallback {'impl': 'dense', 'hook_calls': 1, 'routed_calls': 2, 'fallback_calls': 0, 'strict': True}
  [PASS] S9.identity-vs-plain-PEFT max|delta| = 0.000e+00
  [PASS] S9.direct-forward-no-silent-nullop raises (strict) instead of a silent plain-LoRA fallback: {'impl': 'dense', 'hook_calls': 0, 'routed_calls': 0, 'fallback_calls': 1, 'strict': True}
```

**Self-check that S9 is not vacuous:** S9 is exactly the E1 harness folded into the smoke, and E1
run against the *frozen* file **fails** it (`hook_calls 0`, `fallback_calls 1`, strict raise). The
new check therefore *would have* blocked the frozen code at the CPU gate.

### E3 — P1-B median control

**E3a — order statistic** (`e3_median.py`), verbatim:

```
  n=196 (even=True)  old vals[len//2] = 98.0000 | statistics.median = 97.5000 | explicit (v[97]+v[98])/2 = 97.5000 | identical=False
  n=195 (even=False) old vals[len//2] = 97.0000 | statistics.median = 97.0000 | explicit (v[97]+v[97])/2 = 97.0000 | identical=True
```

**E3b — the amended STEP-3 block run VERBATIM.** The heredoc was extracted byte-for-byte out of the
amended `scripts/slurm/lora_sft_moka.sbatch` (31 lines, extracted-block sha
`a53ec99225d7fc48d137314188a3e2c9f0d1bd6ed86bd87c2a4c691c11165a29`) and executed against a
**synthetic** 196-layer adapter at the deployed shapes (28 layers × 7 projections, r=16), built so
that `A_v = A_t·(1+c)` ⇒ `rel_fro_diff = c` with the sorted values deliberately straddling the
`KS-MOKA-2` bar: `vals[97] = 0.0400`, `vals[98] = 0.0550`. It was run with **cwd inside the
scratchpad** so the block's relative `refine-logs/MOKA_KS2_routing_report.json` write landed there
and **never in the repo** (verified absent, §4). Verbatim:

```
  | [moka_sft] adapter keys: lora_A 196 | lora_A_v 196 | lora_B 196 | total tensors 588 | params 58490880
  | [moka_sft] KS-MOKA-2 rel ||A_v-A_t||_F/||A_t||_F : min 0.0303 median 0.0475 max 0.0647
  | [moka_sft] KS-MOKA-2 median >= 0.05 ? False
  n = 196 | vals[97] = 0.040000 | vals[98] = 0.055000
  reported median (amended)        = 0.047500  -> KS-MOKA-2 'median >= 0.05' = False
  OLD expression vals[len//2]      = 0.055000  -> KS-MOKA-2 'median >= 0.05' = True
  statistics.median cross-check    = 0.047500 | explicit (v[97]+v[98])/2 = 0.047500 | equal=True
  DECISION FLIP on this control: old=True vs true=False
  repo untouched: refine-logs/MOKA_KS2_routing_report.json in repo? False
```

⇒ the amended block reports the true median, agrees with the explicit `(vals[97]+vals[98])/2` to
the bit, and the control exhibits a **decision flip** vs the frozen expression. The run also
re-validates the block's own asserts end-to-end (`196/196/196`, `58,490,880`) and its JSON dump.

### E4 — no-flag extractor path unaffected

Artifact **C** was **not edited** (sha unchanged, §4), so the no-flag path is byte-identical by
construction; the args-default check was re-run anyway (`e4_extractor_and_p2.py`), verbatim:

```
  parse_args_sys([]) -> moka=False | no_merge=False | lora_dir='' | out_model_tag='Qwen2.5-VL-7B-Instruct-LoRA_HF'
  img_instruction default == deployed literal : True
  text_instruction default == deployed literal: True
  NO-FLAG PATH UNCHANGED (both flags OFF => the byte-identical merge_and_unload path): True
```

The `§4.4` GPU-smoke KS-parity leg (bit-exact re-extraction vs the banked cache) is **not** run here
— it is a GPU check and remains **owed** before submit.

### E5 — P2-2 control (pre-fix vs post-fix) and RNG parity

`e5_p2_prefix_and_rng.py`, run twice: `prefix` imports the frozen file extracted from `HEAD`
(re-hashed `9b0fc502…19a8386`), `postfix` imports the amended file. Verbatim:

```
PRE-FIX   adapter_names=None -> TypeError: Linear.forward() got an unexpected keyword argument 'adapter_names'
POST-FIX  adapter_names=None -> OK, out (2, 5, 32)
(post-fix, e4) no kwarg (deployed call shape) -> OK, out (2, 5, 32)
(post-fix, e4) adapter_names=['default','default'] -> delegated to upstream, OK
```

**RNG parity (both variants, identical):** global CPU stream immediately after `get_peft_model`,
with vs without `install_moka`:

```
  without install: [-0.81014305, -0.52058148, -0.90126741, 0.14274217]
  with    install: [-0.81014305, -0.52058148, -0.90126741, 0.14274217]
  CPU RNG stream bit-identical: True
```

⇒ **RNG statement:** `moka_init`'s `fork_rng(devices=[])` + fixed `MOKA_INIT_SEED` block was not
touched by this pass; CPU RNG neutrality is re-measured bit-identical, and the `A_v` initialisation
draw is unchanged (same seeds `MOKA_INIT_SEED + 8n`, same values — S1.A_v-differs and S4's
`rel_fro_diff = [1.4145, 1.484]` reproduce the frozen numbers exactly). P2-4's CUDA-side caveat is
unchanged and still open (§5).

**Parity statement:** the only executable behaviour changed by this pass is (i) *where* the mask
hook is registered, (ii) `adapter_names` popping, (iii) the median expression in job 1's post-run
readout. The routing algebra, the mask definition, dtype discipline, the shared-`B` application,
merge guards, save/load, the parameter budget, all thresholds/bars, the gate order and the
test-touch budget are untouched.

---

## 4. Old → new shas (`sha256sum`, on disk after the fix)

```
P  dc3f1078a89fc2e1de30c870103c2b7f2986fd419698d6c49b5b9ec0966c53f8  refine-logs/MOKA_PREREG.md          UNCHANGED
A  9b0fc502193b0521f1359978f85b35b6ce98034d1371f174315c8f1a219a8386  ->
   6b7bdb6c13262cbfce81f212fcf2ed596a8b466f9ae928ba340ac0eee37c85be  src/moka/routed_lora.py             CHANGED (372 -> 395 L)
B  fae40487263fd7f65e2d0566205e57d3a2caf1b9d1477c693c7cdfdc891c9749  src/moka/train_moka.py              UNCHANGED
C  75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399  src/utils/generate_VideoMLLM_embedding_lora_HF.py  UNCHANGED
D  843dace46611d3a2f5e6942a5f0e780fe8b2fb623a3c969949c142bd559d7793  ->
   bd2585536e7982e021ead4974910fb7df32498eee57625cbe4331d7bbf46c4ef  scripts/analysis/moka_smoke.py      CHANGED (324 -> 422 L)
E  df3c9a6a8721f5d97935e4b87137732726329877c14ee8635902976eaf70e38b  ->
   020dd10bd7cfcabb381d76ea44441cbd9607d2db0b98ef92e2388e71482745e6  scripts/slurm/lora_sft_moka.sbatch  CHANGED (113 -> 116 L)
F  fd1b7f295cdb7106e0e64629cb1a2391355f9daad4ab0f2ad773292f48b31bde  scripts/slurm/moka_extract_head.sbatch  UNCHANGED
G  51b883e9f0a78c26d9b4af185b54a4703a250a3cab4c947756782c6c8fe49764  RA-HMD/…/mhc_zh_qwen25vl_lora_moka_sft.yaml  UNCHANGED
```

Reused-unchanged pins (`MOKA_FREEZE.md §2`) re-hashed after the fix — **all 4 MATCH**:

```
b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py
4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad  scripts/slurm/enc3seed_zh_b3.sbatch
e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f  scripts/slurm/lora_sft.sbatch
2f2429fbd8f6b0b82fba173a4efc5f12ae75cf5e5bce791c3819663c5f1439ea  RA-HMD/…/mhc_zh_qwen25vl_lora_sft.yaml
```

Other post-fix state checks: `bash -n` both sbatch **SYNTAX_OK**; `py_compile` on all four Python
artifacts **PASS**; `run_one()` block in **F** untouched (F's sha unchanged ⇒ byte-identity vs
`enc3seed_zh_b3.sbatch:42-83`, block sha `286a9e44…`, still holds); §4.7 collision surfaces re-checked
and **all still ABSENT** (`logging/lora/MHC_zh_moka`, `logging/_smoke_moka`, `data/CLIP_Embedding/MHC_zh/*moka*`,
`*-um*`, `slurm/logs/*moka*`, `refine-logs/MOKA_KS2_routing_report.json`); `git status --porcelain`
under `src/moka scripts/analysis scripts/slurm` shows **only the three amended MokA files** (the
`lsmi_*` entries belong to another executor and were not touched); the
`RA-HMD/LLAMA-FACTORY-Ver202512` gitlink is untouched (**zero vendored lines edited**).

### Diffs

`src/moka/routed_lora.py` — 2 hunks (+31/−8 over the file):

```diff
@@ MokaLinear.forward (:186-196)
-        if self.disable_adapters or self.merged or kwargs.get("adapter_names") is not None:
+        # `adapter_names` is POPPED (upstream peft/tuners/lora/layer.py:598 does the same) so an
+        # explicitly passed `adapter_names=None` cannot reach `self.base_layer(...)` at the routed
+        # path below and TypeError out of `nn.Linear`.  It is re-inserted verbatim when it is not
+        # None, so the mixed-batch delegation to upstream is unchanged.  No deployed call site
+        # passes the kwarg at all, so this is inert for both jobs.
+        adapter_names = kwargs.pop("adapter_names", None)
+        if self.disable_adapters or self.merged or adapter_names is not None:
+            if adapter_names is not None:
+                kwargs["adapter_names"] = adapter_names
             return super().forward(x, *args, **kwargs)

@@ install_moka (:295-312)
-    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
-    if getattr(base, "_moka_hook", None) is None:
-        base._moka_hook = base.register_forward_pre_hook(_mask_pre_hook, with_kwargs=True)
+    # -- modality-mask pre-hook: registered on the OUTERMOST module the caller actually invokes ---
+    # [16 comment lines: hooks fire in __call__ only; the PeftModelForCausalLM ->
+    #  LoraModel.__call__ -> BaseTuner.forward -> self.model.forward(...) direct call; the two
+    #  deployed call sites (trainer.py:3759/4525, extractor:360); why __dict__ not getattr]
+    if "_moka_hook" not in peft_model.__dict__:
+        peft_model._moka_hook = peft_model.register_forward_pre_hook(_mask_pre_hook, with_kwargs=True)
     return n
```

`scripts/slurm/lora_sft_moka.sbatch` — 2 hunks (+6/−2), inside the STEP-3 heredoc only:

```diff
-import json, os, sys
+import json, os, statistics, sys
...
 vals = sorted(r["rel_fro_diff"] for r in rows)
-med = vals[len(vals) // 2]
+# TRUE median: n == 196 is EVEN, so the median is (vals[97] + vals[98]) / 2, not vals[98].
+# `statistics.median` implements exactly that (mean of the two middle order statistics for even n,
+# the middle element for odd n) on the already-sorted list.
+med = statistics.median(vals)
```

`scripts/analysis/moka_smoke.py` — +102/−2: module docstring gains the S9 paragraph; `S1.hook`
(`getattr(get_base_model(), "_moka_hook")`) is replaced by `S1.hook-on-outermost` +
`S1.hook-not-on-base` (both `__dict__`-based); new `s9_production_wrapper_class()` (`:319-401`)
called from `main()` (`:414`) between S7 and S8. **No existing S1–S8 assertion was weakened**: the
only pre-existing check whose text changed is the hook-location check, which was *strengthened*
(it now also asserts the hook is NOT on the base model).

---

## 5. P2 / P3 dispositions

| # | disposition | justification |
|---|---|---|
| **P2-2** (`adapter_names` not popped) | **FIXED** | Trivial and provably inert for the deployed paths: no deployed call site passes the kwarg (E4/E5 control shows the no-kwarg shape unchanged, `adapter_names=None` no longer `TypeError`s, non-None still delegates to upstream). Matches upstream `peft/tuners/lora/layer.py:598` semantics exactly. |
| **P2-1** (`ok` guard does not validate that `result` is 3-D with `[B,S]` leading dims) | **ACCEPTED, not fixed** | A fix would have to add conditions to the `ok` conjunction, which changes what `fallback_calls` counts and adds a new raise surface inside the hot path — not inert. The deployed targets are the 7 Qwen linear projections, whose `base_layer` output is `[B,S,out]` by construction (S8 re-verifies the `[B,S]` alignment on a real record; S2/S9 verify bit-exact agreement with upstream on that path). Out of scope per the fix mandate; re-reviewer rules. |
| **P2-3** (`load_moka_a_v` does not enforce "every checkpoint tensor consumed" for non-`A_v` keys) | **ACCEPTED, not fixed** | `load_moka_a_v` already raises unless every `.lora_A_v.` key maps to an existing module with matching shape **and** every routed layer received a tensor (S5 covers both, plus the generic-adapter refusal). The unenforced part is PEFT's own `strict=False` load of the non-`A_v` keys, i.e. upstream behaviour shared with every LoRA run in this project, including the banked floor. Fixing it means inspecting PEFT's load-result object — new code on the extraction path, outside this mandate. |
| **P2-4** (RNG neutrality is CPU-only; `fork_rng(devices=[])` does not restore CUDA state) | **ACCEPTED, not fixed** | Any faithful fix either (a) calls `fork_rng` with `devices=None`, which force-initialises CUDA and would break the CPU-only smoke discipline, or (b) hand-saves/restores CUDA generator state — new conditional code in the install path. Non-blocking per the gate: `Trainer.__init__` calls `set_seed(args.seed)` *after* model construction (LLaMA-Factory builds the model in `run_sft` before instantiating `CustomSeq2SeqTrainer`), so the CUDA stream is reseeded before any training RNG is consumed. CPU neutrality re-measured bit-identical (E5). Seeds `MOKA_INIT_SEED + 8n`, no collisions (gate-confirmed). |
| **P2-5** (job 2 has no collision guard on the two feature-cache output families) | **ACCEPTED, not fixed** | Would add a new abort path to artifact **F**, which this pass otherwise leaves byte-identical (keeping the re-review diff minimal). The surfaces are verified ABSENT at freeze, at S0, and again post-fix (§4), and the submit executor re-verifies them at S0 by standing instruction. |
| **P3** (`routed_lora.py:34-35` "FLOPs are IDENTICAL" contradicts §F0.7/DEV-1's "rank identical, compute ≈ +1 %") | **ACCEPTED, not fixed** | Documentation-only, already recorded in `MOKA_FREEZE.md §5`; the prereg text that binds the write-up is correct. Deliberately left untouched to keep the amendment surface confined to the gate's P1s + the blind spot + one trivial P2. |

---

## 6. Status after this pass

- The 2 P1 defects are fixed and each is covered by a check that **fails on the frozen code**
  (S9 for P1-A; E3b's decision-flip control for P1-B).
- **Authorization is VOID** until: (i) an independent **0-context re-review** of the amended **A / D /
  E** and a re-issued freeze block, and (ii) a **codex re-gate** (prereg §4.5, all 7 items) returning
  `GATE: PASS`. Then the §4.4 GPU smoke (still owed, incl. the KS-parity bit-exact leg) and only then
  submit.
- **No GPU / SLURM / Modal spent. No job submitted. No held-out test metric produced. No `state/`
  mutated. No `research-wiki/` mutated. No vendored LLaMA-Factory line edited. Not pushed.**
