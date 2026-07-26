# MOKA-ZH — SUBMIT EXECUTION RECORD

**Executor:** MOKA-ZH submit executor (`/data/jehc223/RGCL`, conda `HateVideo`).
**Authorization:** `refine-logs/MOKA_FREEZE.md` (APPROVED-WITH-NOTES), freeze commit `0ed3807`.
**Prereg:** `refine-logs/MOKA_PREREG.md` (843 L), FROZEN sha `dc3f1078…0966c53f8`.
**Chain executed:** S0 authorization → S1 codex gate → S2 CPU smoke. **HALTED AT S1/S2 BOUNDARY.**

> **⚠ ROUND 1 (below) IS SUPERSEDED — see `## ROUND 2 — REFREEZE-1` at the end of this file.**
> Round 1's STOP was adjudicated correct; the code was amended (`72a947b`), re-frozen
> (`MOKA_FREEZE.md §7` REFREEZE-1) and independently re-reviewed (`cf29665`), and the executor was
> **RESUME-AUTHORIZED** to re-run the chain from the top against the REFREEZE-1 values. Round 1 is
> retained verbatim as the provenance of the two P1 defects.

## ROUND-1 OUTCOME: **STOP — prereg §4.6 FIRES.** The mandatory codex gate returned `GATE: BLOCK` with **2 P1 findings**, both **independently confirmed at runtime by the executor**. Per §4.6 the affected artifact shas must change, the freeze block must be re-issued, and a new independent 0-context review must be re-run against the amended files **before submit**.

**NO GPU spent. NO SLURM job submitted. NO GPU smoke run. NO test metric read. NO frozen file edited.
NO `state/` mutation. NO push.** Cost of this stage: pure CPU login-node work (sha256sum, `bash -n`,
`diff`, the CPU smoke, one ~1-minute CPU confirmation script, read-only `squeue`/`df`/`sacct`).

---

## S0 — Authorization re-verify: **PASS (12/12 sha MATCH)**

`sha256sum` re-run at execution time. All 8 frozen objects (P + A–G) and the 4 reused-unchanged
pins MATCH `MOKA_FREEZE.md §1`/`§2` byte-for-byte:

```
dc3f1078a89fc2e1de30c870103c2b7f2986fd419698d6c49b5b9ec0966c53f8  refine-logs/MOKA_PREREG.md          (P)
9b0fc502193b0521f1359978f85b35b6ce98034d1371f174315c8f1a219a8386  src/moka/routed_lora.py             (A)
fae40487263fd7f65e2d0566205e57d3a2caf1b9d1477c693c7cdfdc891c9749  src/moka/train_moka.py              (B)
75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399  src/utils/generate_VideoMLLM_embedding_lora_HF.py (C)
843dace46611d3a2f5e6942a5f0e780fe8b2fb623a3c969949c142bd559d7793  scripts/analysis/moka_smoke.py      (D)
df3c9a6a8721f5d97935e4b87137732726329877c14ee8635902976eaf70e38b  scripts/slurm/lora_sft_moka.sbatch  (E)
fd1b7f295cdb7106e0e64629cb1a2391355f9daad4ab0f2ad773292f48b31bde  scripts/slurm/moka_extract_head.sbatch (F)
51b883e9f0a78c26d9b4af185b54a4703a250a3cab4c947756782c6c8fe49764  RA-HMD/…/mhc_zh_qwen25vl_lora_moka_sft.yaml (G)
b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py
4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad  scripts/slurm/enc3seed_zh_b3.sbatch
e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f  scripts/slurm/lora_sft.sbatch
2f2429fbd8f6b0b82fba173a4efc5f12ae75cf5e5bce791c3819663c5f1439ea  RA-HMD/…/mhc_zh_qwen25vl_lora_sft.yaml
```

Additional §4 VOID-ON-EDIT conditions, all confirmed:

| gate | result |
|---|---|
| `git status --porcelain src/run_rac.py src/model/loss.py src/model/classifier.py src/utils/retrieval.py` | **empty (CLEAN)** |
| LF gitlink | `git ls-files -s` → `160000 a912747c408b3c661b4029ecf1d88b9d91c7f1a8` — **unchanged** |
| gitlink inner tree | `git -C … status --porcelain` → only `?? .cuda_home_shim/` + `?? my_configs/hatevideo/mhc_zh_qwen25vl_lora_moka_sft.yaml`; `git diff --stat HEAD` **empty** ⇒ **zero modified tracked files, zero vendored lines edited** |
| `run_one()` byte-identity | `sed -n '42,83p' enc3seed_zh_b3.sbatch` vs `sed -n '112,153p' moka_extract_head.sbatch` → `diff` **empty**, both blocks sha `286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101` — **MATCH** |
| §4.7 collision surfaces | `logging/lora/MHC_zh_moka` ✗ · `logging/_smoke_moka` ✗ · `logging/Retrieval/MHC_zh/RAC_video_moka*` ✗ · `data/CLIP_Embedding/MHC_zh/*moka*` ✗ · `*-um*` ✗ · `slurm/logs/*moka*` ✗ — **all ABSENT** |
| banked read-only inputs | `logging/lora/MHC_zh/adapter_model.safetensors` (161,533,192 B, 2026-07-02) and all 3 `…_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` caches present, untouched |
| disk preflight | `df -h /data/jehc223/RGCL` → **517 G avail / 97 % used** at S0; **514 G avail / 97 %** at S2 re-check (job-1 bar is ≥ 25 G ⇒ satisfied) |

Environment pins read at execution time: `peft 0.14.0`, `torch 2.6.0+cu124`, `transformers 4.49.0`.

---

## S1 — MANDATORY CODEX GATE (prereg §4.5): **GATE: BLOCK**

**Reviewer:** codex MCP, model `gpt-5.6-sol`, `model_reasoning_effort=xhigh`, sandbox `read-only`,
`approval-policy never`, cwd `/data/jehc223/RGCL`. Thread `019f997a-1a50-72d1-9e01-11e6462bd949`.
All 7 §4.5 items reviewed. Verbatim closing line: **`GATE: BLOCK`**.

| §4.5 item | codex result |
|---|---|
| 1 `MokaLinear.forward` routing algebra | FINDING (2× **P2**) |
| 2 monkey-patch scope / no global leak | **PASS** |
| 3 save/load round-trip of `lora_A_v` | FINDING (1× **P2**) |
| 4 extractor `--moka`/`--no_merge` gating | FINDING (**1× P1 BLOCKING**) |
| 5 RNG neutrality of `install_moka` | FINDING (1× **P2**) |
| 6 merge guard | **PASS** |
| 7 grad flow + both sbatch | FINDING (**2× P1 BLOCKING**, 1× P2) |

### P1-A (items 4 + 7) — the modality-mask forward-pre-hook NEVER FIRES on the production PEFT class

**Codex's claim.** `install_moka` registers the mask hook on the module returned by
`peft_model.get_base_model()` (`src/moka/routed_lora.py:287-290`). The production wrapper is
`PeftModelForCausalLM`, whose `forward` calls `self.base_model(...)` = `LoraModel.__call__` →
`BaseTuner.forward` → **`self.model.forward(*args, **kwargs)`** — a *direct* `.forward()` call that
bypasses the wrapped model's `Module.__call__` and therefore all of its forward-pre-hooks. Hence
`_STASH.mask` stays `None` and, under `MOKA_STRICT=1`, the first routed decoder projection raises.

**Executor's independent confirmation — 4 separate legs, all reproduced:**

1. **torch semantics.** A registered `forward_pre_hook` fires on `mod(x)` and does **not** fire on
   `mod.forward(x)` (hook counter 1 → 1). Measured.
2. **PEFT source.** `peft/tuners/tuners_utils.py:196-197`:
   `def forward(self, *args, **kwargs): return self.model.forward(*args, **kwargs)` — the direct call.
   `peft/peft_model.py:912-920` `get_base_model()` returns `self.base_model.model` for LoRA, i.e. the
   hook is registered on exactly the module reached by that direct `.forward()`.
   Contrast `peft/peft_model.py:843-849` (generic `PeftModel.forward`): `self.get_base_model()(*args,
   **kwargs)` — via `__call__`, so the hook *does* fire there.
3. **Both production paths really are the causal wrapper.** *Job 1 (SFT):*
   `RA-HMD/LLAMA-FACTORY-Ver202512/src/llamafactory/model/adapter.py:300-303` constructs
   `LoraConfig(task_type=TaskType.CAUSAL_LM, …)`, and `peft/mapping.py:211-224` dispatches a set
   `task_type` to `MODEL_TYPE_TO_PEFT_MODEL_MAPPING[...]` ⇒ `PeftModelForCausalLM`.
   *Job 2 (extraction):* the banked `logging/lora/MHC_zh/adapter_config.json:115` carries
   `"task_type": "CAUSAL_LM"`, and `peft/peft_model.py:564-573` dispatches `from_pretrained` on that
   field ⇒ `PeftModelForCausalLM` again (and job 1's saved MokA adapter would carry the same field).
4. **End-to-end runtime reproduction** (CPU, offline, `GPT2LMHeadModel(n_layer=2, n_embd=32)`,
   `target_modules=["c_attn"]`, `MOKA_STRICT=1`, frozen `routed_lora.py` imported unmodified):

   ```
   class returned by get_peft_model(task_type=CAUSAL_LM): PeftModelForCausalLM
   install_moka routed layers: 2 ; hook registered on: GPT2LMHeadModel
   peft_model.base_model: LoraModel | base_model.model: GPT2LMHeadModel
   forward RAISED RuntimeError: MokA: modality mask absent or shape-mismatched at a routed layer
                                (x=(1, 12, 32), mask=None). Refusing to fall back to plain LoRA …
   moka_stats: {'impl':'dense','hook_calls':0,'routed_calls':0,'fallback_calls':1,'strict':True}
   ```

   `hook_calls == 0`, `routed_calls == 0`, `fallback_calls == 1`. **CONFIRMED.**

**Consequence as measured, stated without embellishment.** `MOKA_STRICT=1` (exported by both sbatch:
`lora_sft_moka.sbatch:67`, `moka_extract_head.sbatch:33`) converts the defect into a hard raise, so it
would **not** silently produce generic-LoRA features — but job 1 would die on its first training batch
*after* base-model + vision GPU work, and job 2's Stage A1 likewise. `fallback_calls` would be ≥ 1, not
`0`, so the §4.4 GPU-smoke assertion `fallback_calls == 0` was the designed backstop; the codex gate
caught it first, at zero GPU cost.

**Why the CPU smoke could not see it (executor-confirmed blind spot).**
`scripts/analysis/moka_smoke.py:82-89` builds its `LoraConfig` **without `task_type`** over a bare
`nn.Module` toy, so `get_peft_model` returns the *generic* `PeftModel`, whose `forward` reaches the base
through `__call__`. Reproduced side by side in the same script: with `task_type=None` the same frozen
`install_moka` yields `{'hook_calls': 1, 'routed_calls': 2, 'fallback_calls': 0}` and the forward
succeeds. The smoke's S1–S8 therefore **PASS while the production class path fails** — the PASS is real
but does not cover the deployed wrapper class.

### P1-B (item 7) — `KS-MOKA-2` computes a non-median statistic

`scripts/slurm/lora_sft_moka.sbatch:104-105` takes `vals = sorted(...)` over the **196** per-layer
`‖A_v−A_t‖_F/‖A_t‖_F` values (asserted `n_a == n_av == n_b == 196` at `:94`) and reports
`med = vals[len(vals) // 2]` = `vals[98]`. For even *n* = 196 the median is `(vals[97]+vals[98])/2`.

Executor confirmation on a 196-element control sequence: `vals[len(vals)//2] = 98.0` vs true median
`97.5` — **not identical**. The pre-registered `KS-MOKA-2` bar (`median < 0.05` ⇒ NULL-OP,
prereg §3.6) is therefore evaluated against an upper-neighbour order statistic rather than the median.
Note: the same quantity computed by `src/moka/routed_lora.py:361-372` (`moka_routing_report`) returns
raw rows and is not implicated; the defect is in job 1's inline post-run block only.
(Reviewer note **N1** already restricts what `KS-MOKA-2` may be claimed to show — it is a
non-degeneracy floor, never a routing-is-real claim — but the statistic itself is still mis-specified.)

### P2 / P3 findings (recorded, non-blocking; carried to whoever amends the code)

| # | item | finding | cite |
|---|---|---|---|
| P2-1 | 1 | the `ok` guard validates `x` and `mask` shape but not that `result` is 3-D with `[B,S]` leading dims; the final reshape could reinterpret a nonstandard base-layer result. Frozen Qwen linear targets preserve the shape. | `routed_lora.py:190-197,212-218,248` |
| P2-2 | 1 | an **explicitly passed** `adapter_names=None` is not popped and reaches `nn.Linear` ⇒ `TypeError`. Executor-confirmed upstream contrast: `peft/tuners/lora/layer.py:598` does `kwargs.pop("adapter_names", None)`; `routed_lora.py:187` uses `kwargs.get(...)` and forwards `kwargs` intact at `:212`. Deployed call sites omit the kwarg. | `routed_lora.py:186-188,211-212` |
| P2-3 | 3 | prereg §3.2/§10 DEV-C's "every checkpoint tensor was consumed" is **not enforced**: `load_moka_a_v` filters only `.lora_A_v.` keys and never inspects PEFT's `strict=False` load result, so non-A_v tensors can go unconsumed; the key-suffix parse also admits malformed keys mapping to one destination. | `routed_lora.py:306-331`; `peft/utils/save_and_load.py:441-451` |
| P2-4 | 5 | RNG neutrality is **CPU-only**, not process-global: `fork_rng(devices=[])` restores CPU state but no CUDA state, while `torch.manual_seed` also reseeds CUDA. Trainer reseeding after model construction makes this non-blocking. Seeds `MOKA_INIT_SEED + 8n`, n=0…195 — codex confirmed **no collisions**. | `routed_lora.py:154-160`; `torch/random.py:32-60` |
| P2-5 | 7 | job 2 guards its adapter *inputs* but has **no collision guard on either feature-cache output family**; a repeat execution can overwrite existing `-um` / `-moka_HF` caches. | `moka_extract_head.sbatch:41-83` |
| P3 | — | pre-existing, already flagged by the freeze doc §5: `routed_lora.py:34-35` still says "FLOPs are IDENTICAL", contradicting §F0.7/DEV-1 ("rank identical, compute ≈ +1 %"). | `routed_lora.py:34-35` |

**Confirmed-PASS items (codex, with executor spot-checks):** item 2 — only
`llamafactory.model.adapter.get_peft_model` is rebound, `peft.get_peft_model` / `peft.mapping.get_peft_model`
untouched, patch lands at `adapter.py:312` i.e. **before** the fp32 trainable cast at `:314-316`.
Item 6 — `merge()` / `get_delta_weight()` always raise, `unmerge()` no-ops while unmerged, and PEFT
0.14's `merge_and_unload` (incl. `safe_merge`) dispatches through the guarded `merge()`. Item 1's core
algebra — `vision ∪ ~vision` partitions exactly once, `[B*S,1]` broadcasts over the rank axis, tied
dense-select is **bit-exact** vs upstream on the frozen 3-D/zero-dropout path, gather/scatter is
semantically equivalent with unique replacement indices, dtype restore / shared `B` / scaling match
upstream. Item 7's gradient algebra — one shared `B`, no `B_v`, no dead parameters; `set -euo pipefail`
does propagate heredoc and `tee`-pipeline failures; Stage S precedes all 3 budgeted reads; exit codes
and the 196/196/196 + 58,490,880 arithmetic are correct.

---

## S2 — CPU smoke + static gates: **PASS (and shown to be blind to P1-A)**

Run fresh at execution time, unmodified frozen `scripts/analysis/moka_smoke.py`:

- `python scripts/analysis/moka_smoke.py` → **`==== ALL SMOKE CHECKS PASS ====`, exit 0**, S1–S8.
  - **S2 identity control** `max|Δ| = 0.000e+00` in all 6 cells (2 modules × all-text/all-vision/mixed);
    dense-vs-gather cross-impl agreement `0.000e+00` (all-text, all-vision) and
    `1.192e-07 / 5.960e-08` (mixed) — reproduces the freeze doc's numbers.
  - **S4** `hook-fired {'hook_calls': 1, 'routed_calls': 2, 'fallback_calls': 0}`; grads non-zero on
    `lora_A`, `lora_A_v` **and** shared `lora_B` at both modules; `dead=[]`;
    routing report `rel_fro_diff = [1.4145, 1.484]`.
  - **S5/S6** state-dict keeps `A_v`; plain reload **drops** it; `load_moka_a_v` restores 2/2
    bit-exactly; generic-adapter refusal fires; `merge_and_unload` / `merge` / `get_delta_weight`
    all raise.
  - **S7** `40,370,176 → 58,490,880 = 1.448864×`.
  - **S8** deployed 262,144-px cap: mask `2688 == 2688` grid arithmetic, `seq 2823 = 2688 + 135`
    (95.2 % vision), masked ids `[151655]`; processor-default: `21528 == 21528`,
    `21663 = 21528 + 135` (99.4 % vision).
- `bash -n scripts/slurm/lora_sft_moka.sbatch` → **SYNTAX_OK**;
  `bash -n scripts/slurm/moka_extract_head.sbatch` → **SYNTAX_OK**.
- Collision surfaces re-checked at S2: **all still ABSENT** (list above).
- Disk: **514 G avail / 97 % used** (`/dev/mapper/data-data`, 14 T).

**The S1–S8 PASS is recorded as genuine but non-covering:** §S1's P1-A blind-spot leg shows the smoke
instantiates a generic `PeftModel`, not the deployed `PeftModelForCausalLM`.

---

## S3 — GPU smoke: **NOT RUN** (blocked by §4.6). S4 — real submission: **NOT SUBMITTED**.

Zero GPU-h of the 4.7 GPU-h cap consumed. **Job IDs: none — no `sbatch` was issued.**
`refine-logs/MOKA_KS2_routing_report.json` was not created (job 1 never ran).

**Queue state observed throughout (read-only `squeue`), relevant to the standing infra rule.** The only
job under this account is **`13531 lsmi_power_cpu` — 16 CPU, `PENDING (JobHeldUser)`** (another
executor's CPU job). Job 1 needs 16 CPU, so `13531 + job-1 = 32 CPU` of submit-time aggregate demand
against a 16-CPU user cap ⇒ **job 1 was in any case not submittable at this instant** and the executor
would have had to poll for `13531` to clear first. This is recorded for the next executor: the CPU
budget is fully occupied by `13531`, and the §4.6 STOP is the *primary* reason for non-submission, the
queue is secondary. **No other executor's job was cancelled or altered.**

---

## S5 — RAW transcription: **N/A — no run produced any number.**

Nothing to transcribe: no SFT trainlog, no extraction stats, no head runs, no `KS-MOKA-0b` cosines, no
`KS-MOKA-2` readout, no `KS-MOKA-3` decomposition. **Zero test-touch. The 3 budgeted test evaluations
remain unspent.** The §7 outcome table stays empty.

## Reviewer notes N1–N6 status

Carried, unconsumed — all six are **write-up-binding at verdict time** and no verdict exists. N1 is the
only one that interacted with this stage: it already forbids reading `KS-MOKA-2` as "routing is real"
(non-degeneracy floor only), which is orthogonal to P1-B's separate finding that the statistic as coded
is not the median.

## Deviations from the task script

1. **Chain terminated at S1/S2 instead of running S3–S6.** Mandated: prereg §4.6 + the executor
   instruction "codex/smoke finds a defect ⇒ STOP and report, do NOT edit frozen files."
2. **S2 was run despite S1 already blocking.** Deliberate and zero-cost: the fresh CPU smoke is the
   direct evidence that S1's P1-A is invisible to the frozen `KS-MOKA-0` gate, which is the single
   most useful fact for the amendment round. It touched nothing and created no artifact.
3. **No frozen file was edited**, so **all 8 freeze shas are still intact at this commit** — the
   authorization is *not* voided by anything the executor did; it is suspended by §4.6 pending a code
   amendment that the executor is not permitted to make.
4. Codex model is `gpt-5.6-sol` xhigh (the account's available model; the `gpt-5.x` family requested).
   Same substitution and rationale as `ZHPROMPT_SUBMIT_RECORD.md:307-308`.

## What §4.6 now requires (not performed here)

A code amendment (at minimum: the hook-registration site in artifact **A** and the median expression in
artifact **E**), which changes shas **A** and **E** ⇒ **the freeze block MUST be re-issued and a new
independent 0-context review re-run against the amended files before any submit.** A re-run of the
full CPU smoke is additionally required, and — per DEV-C — it should be extended to exercise the
deployed `task_type=CAUSAL_LM` wrapper, since the present S1–S8 cannot see this class of defect.

**Required statements:** ZERO GPU / SLURM / Modal spent. NO job submitted. No held-out test metric
produced. No `state/` mutated. No `research-wiki/` mutated. No frozen artifact edited. Not pushed.

---
---

# ROUND 2 — REFREEZE-1

**Resume authorization.** Round 1's `GATE: BLOCK` was adjudicated **correct**. The fix
(`72a947b`), the re-issued freeze (`MOKA_FREEZE.md §7` REFREEZE-1) and the independent 0-context
re-review (`cf29665`, `MOKA_REFREEZE_REVIEW.md`, **APPROVED**) are complete, and the executor was
instructed to re-run the chain **from the top** against the REFREEZE-1 values.

**Binding condition N7** (from `MOKA_REFREEZE_REVIEW.md`, carried into this record): the codex
re-gate was pointed **explicitly at the AMENDED hook site**. Prereg §4.5 item 2's original wording —
"registered on the base `Qwen2_5_VLForConditionalGeneration`" — **describes the defect round 1
invalidated and is obsolete**. Everywhere in this record, and in any restatement of item 2:
**the mask forward-pre-hook sits on the OUTER `PeftModel` wrapper** (`install_moka`'s own
`peft_model` argument), which is the object both deployed paths invoke via `__call__`. The
prereg-wording divergence was ruled covered by the REFREEZE-1 note; no bar, threshold, gate order or
test-touch budget is affected.

## S0' — Authorization re-verify vs REFREEZE-1: **PASS**

`sha256sum` re-run at execution time; **3 amended + 5 unchanged, 8/8 as recorded in §7.1**:

```
A  6b7bdb6c13262cbfce81f212fcf2ed596a8b466f9ae928ba340ac0eee37c85be  src/moka/routed_lora.py       (AMENDED, was 9b0fc502…)
D  bd2585536e7982e021ead4974910fb7df32498eee57625cbe4331d7bbf46c4ef  scripts/analysis/moka_smoke.py (AMENDED, was 843dace4…)
E  020dd10bd7cfcabb381d76ea44441cbd9607d2db0b98ef92e2388e71482745e6  scripts/slurm/lora_sft_moka.sbatch (AMENDED, was df3c9a6a…)
FROZEN dc3f1078a89fc2e1de30c870103c2b7f2986fd419698d6c49b5b9ec0966c53f8  refine-logs/MOKA_PREREG.md  (unchanged)
B  fae40487263fd7f65e2d0566205e57d3a2caf1b9d1477c693c7cdfdc891c9749  src/moka/train_moka.py        (unchanged)
C  75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399  src/utils/generate_VideoMLLM_embedding_lora_HF.py (unchanged)
F  fd1b7f295cdb7106e0e64629cb1a2391355f9daad4ab0f2ad773292f48b31bde  scripts/slurm/moka_extract_head.sbatch (unchanged)
G  51b883e9f0a78c26d9b4af185b54a4703a250a3cab4c947756782c6c8fe49764  RA-HMD/…/mhc_zh_qwen25vl_lora_moka_sft.yaml (unchanged)
```

All 4 reused-unchanged pins re-hashed and **MATCH** (`run_rac.py b85eb72a…`,
`enc3seed_zh_b3.sbatch 4379224671…`, `lora_sft.sbatch e767eba0…`, deployed ZH yaml `2f2429fb…`).
`src/run_rac.py`, `loss.py`, `classifier.py`, `retrieval.py` **git-clean**. LF gitlink still
`160000 a912747c408b3c661b4029ecf1d88b9d91c7f1a8`, inner tree carries **zero modified tracked files**
(`git diff --stat HEAD` empty; only `?? .cuda_home_shim/` + `?? …_moka_sft.yaml`) ⇒ **zero vendored
lines edited**. `run_one()` block `diff` **empty**, sha `286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101`.
§4.7 collision surfaces **all ABSENT** (incl. `MOKA_KS2_routing_report.json`). Disk **511 G avail /
97 % used** (job-1 bar ≥ 25 G).

## S1' — MANDATORY CODEX RE-GATE (full, all 7 items): **GATE: PASS**

**Reviewer:** codex MCP, model `gpt-5.6-sol`, `model_reasoning_effort=xhigh`, sandbox `read-only`,
cwd `/data/jehc223/RGCL`. Thread `019f99b0-2f8d-73e0-8840-16aa829c3774`. Given the `72a947b` diff,
`MOKA_REFREEZE_FIX.md`, round 1's findings, the N7 scoping condition, and the accepted-not-fixed
P2-1/P2-3/P2-4/P2-5/P3 justifications as context.

| §4.5 item | round 1 | round 2 |
|---|---|---|
| 1 routing algebra (+ new `adapter_names` pop) | FINDING 2×P2 | **PASS** |
| 2 monkey-patch scope + **amended outer-wrapper hook site** | PASS | **PASS** |
| 3 save/load round-trip of `lora_A_v` | FINDING 1×P2 | **PASS** |
| 4 extractor `--moka`/`--no_merge` gating | **P1 BLOCKING** | **PASS** |
| 5 RNG neutrality | FINDING 1×P2 | **PASS** |
| 6 merge guard | PASS | **PASS** |
| 7 grad flow + both sbatch (+ median, + S9 audit) | **2× P1 BLOCKING** | **PASS** |

**Verbatim closing line: `GATE: PASS`. Zero new P1 / P2 / P3 findings.** Load-bearing confirmations
codex reached by reading source (not by taking the fix record's word):

- **Item 2 — the amended hook covers every surface these two jobs use.** The hook is on the outer
  `PeftModel` with `with_kwargs=True`, so it observes the original `input_ids` kwargs before dispatch,
  avoiding PEFT's inner direct-`.forward()` bypass (`peft/tuners/tuners_utils.py:196-197`). Job 1
  training + eval both go through `model(**inputs)` (`transformers/trainer.py:3744-3759,4513-4525`;
  `llamafactory/train/sft/trainer.py:115-139`). The **frozen yaml enables no generation, DeepSpeed,
  FSDP or compilation**, so `predict_with_generate` stays false ⇒ **no `.generate()` surface exists**;
  the 1-GPU launch introduces no DDP; **accelerate patches the same instance's `forward`, leaving its
  outer `__call__` hooks intact**; gradient checkpointing replays decoder layers, not the outer
  wrapper. Job 2 keeps the `PeftModel`, installs on it, and calls `model(**inputs, …)`
  (`generate_VideoMLLM_embedding_lora_HF.py:360,500-518`). The `"_moka_hook" not in
  peft_model.__dict__` ownership check is correct despite PEFT attribute forwarding.
- **Item 1 — the new `adapter_names` pop matches upstream for all three cases** (absent / explicit
  `None` / non-`None`); the non-`None` path still delegates to upstream mixed-batch handling.
- **Item 7 — `statistics.median` on the sorted 196-element list yields the true even-sample median
  `(vals[97]+vals[98])/2`**, and nothing else in that heredoc regressed. **S9 genuinely detects the
  old defect**: it builds the production `PeftModelForCausalLM`, makes `B` nonzero, captures the
  reference *before* installation, then ties `A_v := A_t` — and with the pre-fix base-model hook site
  the cleared stash stays empty and strict mode exits nonzero.
- The five accepted-not-fixed justifications were each re-checked and **none was found wrong**.

## S2' — CPU smoke S1–S9, fresh: **ALL SMOKE CHECKS PASS, exit 0**

- **S9 (the new check that closes the P1-A blind spot)** — all green on the real deployed class:
  `S9.class-is-PeftModelForCausalLM`; `S9.hook-on-wrapper` → *hook owner =
  `PeftModelForCausalLM` (NOT `get_base_model()` = `GPT2LMHeadModel`)*;
  `S9.hook-fires-on-model(**inputs)` → `{'hook_calls': 1, 'routed_calls': 2, 'fallback_calls': 0}`;
  `S9.identity-vs-plain-PEFT max|Δ| = 0.000e+00`; `S9.direct-forward-no-silent-nullop` → raises under
  strict rather than degrading to plain LoRA (`{'hook_calls': 0, 'routed_calls': 0,
  'fallback_calls': 1}`).
- **S1** strengthened: `S1.hook-on-outermost` PASS and `S1.hook-not-on-base` PASS.
- **S2** identity control still `0.000e+00` in all 6 cells; dense-vs-gather `0.000e+00` (all-text,
  all-vision) / `1.192e-07`, `5.960e-08` (mixed).
- **S4** grads non-zero on `lora_A`, `lora_A_v` and shared `lora_B` at both modules, `dead=[]`.
- **S5/S6** round-trip + silent-drop + generic-adapter refusal + all three merge raises.
- **S7** `40,370,176 → 58,490,880 = 1.448864×`. **S8** `2688 == 2688`, `seq 2823 = 2688 + 135`
  (95.2 % vision), ids `[151655]`; processor-default `21528 == 21528`, `21663 = 21528 + 135`.
- `bash -n` both frozen sbatch **SYNTAX_OK**.

### Executor's own independent re-confirmation (round-1 harness, unmodified)

The **same executor-written probe that detected P1-A in round 1** was re-run against the amended
code. On a real `PeftModelForCausalLM` it now reports:

| | round 1 (pre-fix) | round 2 (post-fix) |
|---|---|---|
| `hook_calls` | **0** | **1** |
| `routed_calls` | **0** | **2** |
| `fallback_calls` | **1** | **0** |
| forward | **RuntimeError** (strict raise) | **OK** |

i.e. the defect-detector flipped. (The probe's own console label "hook registered on:
`GPT2LMHeadModel`" is a **stale string in the round-1 script** — it prints
`type(pm.get_base_model()).__name__`, naming the base model, not the hook's owner. The authoritative
ownership assertion is smoke `S9.hook-on-wrapper`, which reads `__dict__` and reports
`PeftModelForCausalLM`.) The generic-`PeftModel` contrast leg is unchanged
(`hook_calls: 1`) — consistent with round 1's blind-spot diagnosis.

## S3' — GPU smoke (prereg §4.4): **SUBMITTED, job `13537`**

Throwaway artifacts live in the executor scratchpad (**nothing added to the repo**); the only
on-disk product is `logging/_smoke_moka`, which the job deletes at the end (§4.7).

**Resource choice — 8 CPU, and why.** `sacct` forensics settle the standing infra rule precisely:
job **13303** (`lora_sft_vis`, **16 CPU**) was co-submitted at `2026-07-20T05:21:10` with **13301**
(`lora_sft_vis`, **16 CPU**); 13301 started immediately, **13303 never started at all**
(`Start=None`) and sat until it was cancelled `2026-07-21T10:19:47` — **the 29 h wedge, caused by
two 16-CPU jobs**. At that same submit instant **13302** (`lora_embed`, **8 CPU**) was also queued
and ran fine right after 13301; the same 16+8 pattern repeats at 13328/13329/13330. So the rule is
literally *never two 16-CPU jobs*, and an 8-CPU job alongside a pending 16-CPU job is
demonstrated-safe. Every smoke in this campaign is 8 CPU (`smoke_readout`, `nca_smoke`, `zhpsmoke`,
`fuscat_smoke`, `hr_smoke`, `smoke_bidir`, `smoke_fb16`). **Job 1 (16 CPU) still waits for 13531.**

**Legs.** (1) 10-step MokA SFT through the **frozen** `src/moka/train_moka.py`, imported as a module
so its monkey-patch applies verbatim, with `run_exp(callbacks=[…])` — `run_exp` accepts `callbacks`
(`llamafactory/train/tuner.py:126`), so the probe only **observes**; zero frozen bytes touched.
Asserts loss finite, grad norms > 0 for `A_t` **and** `A_v` **and** shared `B` (measured at
`on_pre_optimizer_step`, before grads are zeroed; the A-side bar is max-over-steps because PEFT
zero-inits `B` ⇒ `dL/dA ≡ 0` at step 0, prereg §3.2 S4), and `fallback_calls == 0`.
(2) job-1 STEP-3 post-run rehearsal on the smoke adapter (196/196/196, 58,490,880, the amended
`statistics.median`). (3) `--moka` 2-video extraction: `(2,3584)`, finite, non-zero.
(4) **KS-parity**: no-flag extraction on the **banked generic** adapter must reproduce the banked
cache with `max|Δ| == 0.0` on **both** streams — fail ⇒ HALT. (5) merge-drift machinery rehearsal.

**Throwaway-yaml deviations from the frozen recipe (smoke only, documented):** `output_dir` →
`logging/_smoke_moka`; `max_steps: 10`; `save_strategy: "no"` (the final `trainer.save_model()` at
`llamafactory/train/sft/workflow.py:97` is unconditional under `do_train`, so the smoke adapter is
still written); and `eval_strategy: epoch` → `steps` / `eval_steps: 5` **deliberately**, so the smoke
exercises the **eval** call surface as well as the training one — 10 steps would never reach an
epoch boundary otherwise, and the eval loop is the second surface the amended hook must cover.

**Deviation on `KS-MOKA-0b`.** The resume message lists merge-drift under the GPU smoke, but the
**pre-registered** `KS-MOKA-0b` is Stage A0 of **job 2** (frozen artifact **F**, all 3 splits, the
`-um` tag, bar mean per-item cosine ≥ 0.9999). Running the full probe twice would cost an extra
~0.6 GPU-h and would pre-create the `-um` caches that job 2 then rewrites (P2-5: job 2 has no
output-cache collision guard). The smoke therefore runs an **8-item machinery rehearsal to a
throwaway tag** only; **the pre-registered KS-MOKA-0b number is the one job 2 Stage A0 produces.**

## S5' tooling — validated in advance ($0, zero test-touch)

- **Independent trainlog cross-parser** (scratchpad; deliberately a different implementation from
  the parser embedded in artifact **F** — token-splitting instead of one regex, and it records the
  1-based source line of every value). Validated against the **banked floor 13150**: it reproduces
  **all 12 numbers** of prereg §2.1 bit-exactly —
  seed 0 val-sel ep 20 `0.8322/0.8023` (line 220) / final ep 29 `0.8456/0.8181` (line 302);
  seed 1 val-sel ep 26 `0.8255/0.7956` (line 275) / final ep 29 `0.8389/0.8113` (line 303);
  seed 2 val-sel ep 19 `0.8389/0.8065` (line 207) / final ep 29 `0.8523/0.8226` (line 298).
  30 val + 30 test epochs parsed per seed.
- **`KS-MOKA-3` stream-decomposition readout** (scratchpad; imports
  `scripts/analysis/encoder_swap_geometry.py` verbatim as a library and applies the MOVED/FLAT rule
  transcribed from `hatemm_lora_stream_decomp.py:232-237`, `MOVE_TR 0.010` / `MOVE_DV 0.005`).
  Dry-run on a **banked** ZH LoRA variant confirms the machinery and the provenance: it reads
  `n_train 579`, `n_dev 78` (the 78-item dev wall) and a floor **text train-LOO AUC 0.9254**, which
  matches **F45's published ZH text train-LOO 0.925**. Opens **train + dev_seen only** — `test_seen`
  is never touched.

### S3'-a — job `13537` **FAILED in 11 s** (`ExitCode 1:0`). Root cause: **the throwaway harness, NOT a frozen artifact.**

```
yaml.constructor.ConstructorError: while constructing a mapping
  in ".../scratchpad/moka_smoke_gpu.yaml", line 2, column 1
found duplicate key save_strategy
  in ".../scratchpad/moka_smoke_gpu.yaml", line 51, column 1
```
Raised at `llamafactory/hparams/parser.py:75` (`OmegaConf.load`), reached from
`tuner.py:127 read_args` — i.e. **before any model was constructed**. `slurm/logs/moka_smoke_13537.out`.

**Diagnosis.** The frozen ZH recipe already carries `save_strategy: epoch` (frozen yaml line 29).
The executor's scratchpad build step **appended** a second `save_strategy: "no"` instead of
substituting, and OmegaConf rejects duplicate top-level keys. **The defect is 100 % in the executor's
own throwaway yaml.** Frozen artifact **G** re-hashed immediately after the failure:
`51b883e9f0a78c26d9b4af185b54a4703a250a3cab4c947756782c6c8fe49764` — **UNCHANGED**, as were A–F and
the prereg. **No frozen artifact is implicated, so §4.6 does NOT fire**; the throwaway was repaired
under the standing "throwaways are not frozen artifacts" rule.

**Cost:** 11 s wall, GPU allocated but idle (the crash preceded model load) ⇒ **~0 GPU-h**, budget
unaffected. `set -euo pipefail` aborted at LEG 1 exactly as designed — the harness failed safe and
wrote nothing: `logging/_smoke_moka` absent afterwards, all collision surfaces still ABSENT.

**Fix (throwaway only).** The `save_strategy` override was **removed entirely** rather than
de-duplicated: 579 train rows ÷ grad-accum 8 = **72 optimizer steps per epoch**, and `max_steps: 10`
never reaches an epoch boundary, so the frozen `save_strategy: epoch` can never fire and writes no
mid-training checkpoint. The final `trainer.save_model()`
(`llamafactory/train/sft/workflow.py:97`) is unconditional under `do_train`, so the smoke adapter is
still produced. The rebuilt throwaway yaml is now **3 clean substitution hunks** off the frozen
recipe — `output_dir` → `logging/_smoke_moka`, `num_train_epochs: 3.0` → `max_steps: 10`,
`eval_strategy: epoch` → `steps` + `eval_steps: 5` — with **no appended keys at all**.

**Process fix — the pre-flight whose absence caused this.** Before resubmitting, the throwaway yaml
is now validated locally by (a) a duplicate-top-level-key scan, (b) `OmegaConf.load` (the exact call
at `parser.py:75`), and (c) **the real `run_exp` argument path** `read_args()` → `get_train_args()`.
All three pass. (c) additionally **runtime-confirms on the parsed arguments** three properties codex
had asserted from source at item 2: **`predict_with_generate = False`** ⇒ **no `.generate()`
surface**, **`deepspeed = None` / `fsdp = []`** ⇒ no wrapper interposition, and
**`lora_dropout = 0.0`** ⇒ `install_moka`'s `require_zero_dropout` guard is satisfied. Also
confirmed: `max_steps 10`, `eval_strategy STEPS`/`eval_steps 5`, `save_strategy EPOCH` (inert here),
`load_best_model_at_end False`, `freeze_vision_tower True`, `lora rank/alpha 16/32`.
*(The parser must see exactly one visible GPU — `parser.py:296` raises `ParallelMode.NOT_DISTRIBUTED`
otherwise. That is a login-node artifact only; `--gres=gpu:a100:1` gives the job a single device,
which is how the banked floor job 12143 ran.)*

### S3'-b — resubmitted as job **`13551`** (8 CPU/64 G/1 A100, no `--time`)

Frozen shas A–G + prereg re-verified **MATCH** at the resubmit instant; `bash -n` SYNTAX_OK (the
sbatch itself was never at fault and is unchanged); all collision surfaces ABSENT; queue empty of any
other job of this account (13531 was **CANCELLED by its owner at 03:26:03 having never started**, so
the 16-CPU aggregate is now free for job 1 once the smoke passes).

### S3'-b RESULT — job `13551` **COMPLETED, ExitCode 0:0, elapsed `00:31:55`. ALL 5 LEGS PASS.**

Log: `slurm/logs/moka_smoke_13551.out`, closing line
`======== MOKA GPU SMOKE ALL LEGS PASS (13551) ========`.

**LEG 1 — 10-step MokA SFT on the REAL production wrapper** (log l.260-261, 274-373, 521-530):

```
[moka] routed 196 lora.Linear layers -> MokaLinear (A_v added, B shared)
[moka] trainable params: {'trainable_total': 58490880, 'lora_A_t': 18120704,
                          'lora_A_v': 18120704, 'lora_B': 22249472}
```
`18,120,704 + 18,120,704 + 22,249,472 = 58,490,880` ✔ (= the frozen §F0.7 figure, 1.448864×).

| step | grad-norm `A_t` | grad-norm `A_v` | grad-norm shared `B` | `fallback_calls` |
|---|---|---|---|---|
| 1 | 0.000000e+00 | 0.000000e+00 | 9.999997e-01 | 0 |
| 2 | 0.000000e+00 | 0.000000e+00 | 9.999996e-01 | 0 |
| 3 | 1.876191e-01 | 4.481185e-03 | 9.822312e-01 | 0 |
| 4 | 1.986559e-01 | 1.330844e-02 | 9.799784e-01 | 0 |
| 5 | 1.084415e-01 | 5.173314e-03 | 4.651682e-01 | 0 |
| 6 | 4.611471e-01 | 1.070130e-02 | 8.872584e-01 | 0 |
| 7 | 2.273781e-01 | 1.926310e-02 | 9.736151e-01 | 0 |
| 8 | 4.622498e-01 | 1.180384e-02 | 8.866705e-01 | 0 |
| 9 | 2.516530e-01 | 1.442473e-02 | 7.600222e-01 | 0 |
| 10 | 5.297920e-01 | 1.297502e-02 | 8.480275e-01 | 0 |

`max A_t 5.297920e-01`, `max A_v 1.926310e-02`, `max B 9.999997e-01` — **all three > 0 ⇒ grad flow
reaches `A_t` AND `A_v` AND the shared `B`.** The two leading zeros at steps 1-2 are the
**pre-declared** prereg §3.2 S4 behaviour (PEFT zero-inits `lora_B`, so `dL/dA ≡ 0` until `B` has
moved), not a defect. Losses `[0.2373, 0.1737]`, **all finite**. 10 optimizer steps, **3 eval loops**
completed. Final `moka_stats = {'impl': 'dense', 'hook_calls': 314, 'routed_calls': 77224,
'fallback_calls': 0, 'strict': True}` — **`fallback_calls == 0` at every single readout, across both
the training and the eval call surfaces, on the real `PeftModelForCausalLM` under `MOKA_STRICT=1`.**
This is the runtime discharge of round-1's P1-A on the deployed path.

*Raw observation, no interpretation:* `A_v`'s grad-norm runs ~25-40× below `A_t`'s throughout,
despite vision tokens being 94.6 % of positions (§F0.6).

**LEG 1b — job-1 STEP-3 post-run block rehearsal** (l.532-535):
`lora_A 196 | lora_A_v 196 | lora_B 196 | tensors 588 | params 58490880` ✔ (both frozen asserts pass).
`KS-MOKA-2 (smoke, 10 steps only) min 1.4046 median 1.4140 max 1.4242`;
`median expression check: statistics.median == (v97+v98)/2 -> True` ⇒ **the amended P1-B expression is
correct on a real 196-layer adapter.** *(This smoke median is ~1.41, i.e. still at the two-independent-
Kaiming-draws value after only 10 steps — it is **not** the cell's KS-MOKA-2, which job 1 emits after
the full 3-epoch run. Per **N1** it is a non-degeneracy floor either way.)*

**LEG 2 — `--moka` 2-video extraction** (l.542-547):
`[moka] routed layers: 196 | lora_A_v tensors loaded: 196` — the silent-drop trap is defeated in
practice. `shapes img (2, 3584) text (2, 3584)`, all finite, both rows non-zero.

**LEG 3 — KS-parity, the HALT gate** (l.557-560):
```
KS-parity img_feats  max|delta| = 0.000000e+00  over N=8
KS-parity text_feats max|delta| = 0.000000e+00  over N=8
KS-parity BIT-EXACT (both streams max|delta| == 0.0) ? True
```
**PASS.** The edited extractor with **no** new flags reproduces the banked generic-LoRA cache
**bit-exactly** ⇒ §F0.8's default==identity claim is now runtime-confirmed, and there is no stack
drift between the banked floor and this cell.

**LEG 4 — merge-drift machinery rehearsal** (l.570-572): path works; see the flag below.

**Cleanup verified:** `logging/_smoke_moka removed: YES`; no stray `*moka*`, `*-um*`, `_parity`,
`_umsmoke`, `_mokasmoke` caches in `data/CLIP_Embedding/MHC_zh`; `RAC_video_moka*` absent;
`MOKA_KS2_routing_report.json` absent; frozen artifacts **git-clean**. Disk 463 G avail / 97 %.

### ⚠ MATERIAL FORECAST from LEG 4 — the §3.4 `KS-MOKA-0b` contingency looks likely to FIRE

```
merge-drift(8 items) img_feats  mean cos 0.99976921  min cos 0.99947494
merge-drift(8 items) text_feats mean cos 0.99945784  min cos 0.99917126
```
The pre-registered `KS-MOKA-0b` bar (§3.4) is **mean per-item cosine ≥ 0.9999 on ALL 6 (split ×
stream) cells**. On this 8-item rehearsal **both** streams sit **below** it (`0.99977` and
`0.99946`). This is **not** a leg failure — LEG 4 was declared machinery-only and the machinery works
— and it is **not** the pre-registered measurement, which job 2 Stage A0 produces over all 3 splits
at full N. But the drift is a property of **bf16 accumulation order** (merged `W+BA` in one matmul vs
unmerged `Wx + B(Ax)`), not of sample size, so the forecast is robust.

**If it fires, §3.4 is already pre-declared and pre-budgeted:** a same-path **unmerged** floor head
run (3 seeds, **+0.05 GPU-h, +3 test evaluations** — the contingent budget reserved in §F0.1) becomes
**MANDATORY before any verdict**, and the arm is paired against **that** floor **instead of 13150**.
Note the frozen job-2 sbatch (**F**) does **not** contain that run, so it would need a separate
submission on the `-um` caches Stage A0 leaves behind. **Flagged now for planning; no action taken,
and this does not gate job 1.** This is exactly the scenario recon §3.6 / prereg §3.4 anticipated
("a routing-OFF MokA **cannot** reproduce the floor cache bit-exactly").

### Budget disclosure (honest)

GPU smoke actual **`00:31:55` = 0.532 GPU-h** (plus 13537's 11 s) against a planned **0.2**.
**Overrun +0.33 GPU-h**, cause identified: the executor's own throwaway `eval_strategy: steps` /
`eval_steps: 5` deviation triggered **3 full 78-item eval passes at ~4:57 each ≈ 15 min**, i.e. about
half the runtime. It bought the evidence that the amended hook fires on the **eval** surface too
(`fallback_calls == 0` across all 3 eval loops), which is a surface the frozen job 1 exercises every
epoch. Projected family total ≈ **4.98 GPU-h** vs the **4.7 cap** (≈ 5.03 if the §3.4 contingency
fires). Per standing instruction the cap is a planning figure: **recorded, and execution continues.**

## S4' — REAL SUBMISSION, job 1 of 2: `sbatch scripts/slurm/lora_sft_moka.sbatch` → **job `13552`**

Gate order §3.11 satisfied in full before this submit: G-repro sha (S0') → **codex gate `GATE: PASS`**
(S1') → **CPU smoke S1-S9 all-PASS** (S2') → **GPU smoke incl. KS-parity bit-exact `0.000000e+00`**
(S3'-b, job 13551) → `logging/_smoke_moka` deleted.

**Submit-instant re-verification (all MATCH / CLEAN):** frozen A–G + prereg re-hashed at the instant
of `sbatch` — `A 6b7bdb6c…`, `B fae40487…`, `C 75bb8156…`, `D bd258553…`, `E 020dd10b…`,
`F fd1b7f29…`, `G 51b883e9…`, `FROZEN dc3f1078…`; `src/run_rac.py b85eb72a…`; `loss.py` /
`classifier.py` / `retrieval.py` **git-clean**; LF gitlink still `160000
a912747c408b3c661b4029ecf1d88b9d91c7f1a8`; `logging/lora/MHC_zh_moka` **ABSENT** (job 1's `exit 4`
collision guard clear); disk **463 G** avail (bar ≥ 25 G).

**Resource / infra-rule compliance:** `NumCPUs=16`, `ReqTRES=cpu=16,mem=120G,gres/gpu=1`, **no
`--time`**. The account's queue was **completely empty** at submit — 13531 was cancelled by its owner
at 03:26:03 and the smoke 13551 had already reached terminal — so **exactly one job of this family is
in flight and there is no second 16-CPU job anywhere.** The 13303-wedge condition cannot arise.
Job 2 (8 CPU) is **not** submitted until `sacct` reports 13552 terminal (prereg §1.0 / DEV-4:
sequential, never `--dependency=afterok`).

Initial state `PENDING (JobHeldUser)` — **expected; awaiting auto-release, never forced** (CLAUDE.md,
prereg §10 DEV-D). Watcher armed on 13552.

**Expected products:** `logging/lora/MHC_zh_moka/{adapter_config.json, adapter_model.safetensors,
checkpoint-*}` (196 `lora_A` + 196 `lora_A_v` + 196 `lora_B`, 58,490,880 params), the job-1 STEP-3
asserts, and the **`KS-MOKA-2`** readout → `refine-logs/MOKA_KS2_routing_report.json`. Per **N1** that
median is reported as a **non-degeneracy floor only** — never as evidence that routing is active;
the routing-activity claim rests on `fallback_calls == 0` (already discharged at LEG 1) and on
`KS-MOKA-3`.

### S4' job-1 RESULT — `13552` **COMPLETED, ExitCode `0:0`**, wall `03:24:49` (11:02:37 → 14:27:26 UTC)

Log `logging/slurm/lora_sft_moka_13552.out`; closing line
`[moka_sft] DONE DATASET=MHC_zh -> /data/jehc223/RGCL/logging/lora/MHC_zh_moka` (l.1996).

**Routing installed on the real deployed path** (l.1348, 1604-1605):
```
[moka] patched llamafactory.model.adapter.get_peft_model; cwd=.../LLAMA-FACTORY-Ver202512
[moka] routed 196 lora.Linear layers -> MokaLinear (A_v added, B shared)
[moka] trainable params: {'trainable_total': 58490880, 'lora_A_t': 18120704,
                          'lora_A_v': 18120704, 'lora_B': 22249472}
```
`MOKA_STRICT=1` was exported for the whole run and **no strict raise occurred over the full 3 epochs**
⇒ every routed layer saw a valid modality mask at every step (the mask-hook invariant held on both
the training and the per-epoch eval surface).

**Train-loss curve endpoints** (43 logged points, `logging_steps: 5`):

| point | epoch | loss |
|---|---|---|
| first logged | 0.07 | **0.2488** |
| second | 0.14 | 0.1868 |
| third | 0.21 | 0.1378 |
| … | … | … |
| third-last | 2.82 | 0.0115 |
| second-last | 2.88 | 0.0256 |
| **last logged** | **2.95** | **0.0160** |

**Per-epoch eval loss** (`eval_strategy: epoch`, the frozen recipe): epoch 1.0 → **0.11615663766860962**
(l.1714); epoch 2.0 → **0.09505169838666916** (l.1824); epoch 2.97 → **0.10931766778230667** (l.1942).

**`all_results.json`** (`logging/lora/MHC_zh_moka/all_results.json`, verbatim):
`epoch 2.9671848013816926`, `train_loss 0.11931405959788847`, `train_runtime 8784.611`,
`train_samples_per_second 0.198`, `train_steps_per_second 0.025`, `eval_loss 0.10931766778230667`,
`eval_runtime 301.5816`, `total_flos 2.28718331744256e+17`.

**Runtime vs the banked floor:** MokA `train_runtime` **8,784.611 s** vs floor job 12143's
**8,635.9986 s** (`logging/lora/MHC_zh/all_results.json`) ⇒ **ratio 1.0172, i.e. +1.72 %**. The prereg
budgeted "× ~1.2 routing/eval overhead" (§1.1); the measured routing overhead is **an order of
magnitude smaller than budgeted**, consistent with §F0.7/DEV-1's amended "compute ≈ +1 %".

**Job-1 STEP-3 post-run asserts — both PASS** (l.1993):
`[moka_sft] adapter keys: lora_A 196 | lora_A_v 196 | lora_B 196 | total tensors 588 | params 58490880`
⇒ `n_a == n_av == n_b == 196` ✔ and `tot == 58,490,880` ✔.

**Artifacts on disk:** `logging/lora/MHC_zh_moka/adapter_model.safetensors` = **234,042,880 B (234 MB)**
— matching the prereg's ~234 MB/save prediction (1.4489× the deployed 161,533,192 B) — plus
`adapter_config.json` and 3 epoch checkpoints (`checkpoint-73`, `-146`, `-216`).

#### `KS-MOKA-2` (l.1994-1995) — reported as a **NON-DEGENERACY FLOOR ONLY**, per reviewer note **N1**

```
[moka_sft] KS-MOKA-2 rel ||A_v-A_t||_F/||A_t||_F : min 1.4039 median 1.4170 max 1.4292
[moka_sft] KS-MOKA-2 median >= 0.05 ? True
```
`refine-logs/MOKA_KS2_routing_report.json`: **196 rows**, min **1.4039**, **median 1.4170**, max **1.4292**
(computed with the amended `statistics.median`, i.e. the true even-sample median).

**N1 is binding and is applied here verbatim.** This number is **NOT** evidence that routing is real
and must never be reported as such. N1 measured at freeze that two independent Kaiming draws at the
deployed `A` shape (16 × 3584) already sit at **1.4136**, while a *trained* `lora_A`'s **total**
displacement across a real 3-epoch deployed ZH SFT is median **0.0506** / max **0.1267**. The measured
median **1.4170** is therefore statistically indistinguishable from the two-independent-draws value —
i.e. the check confirms only that the two down-projections did **not** collapse onto each other; it
carries **no** information about whether routing did anything. **The routing-activity claim rests
entirely on `fallback_calls == 0` (discharged at GPU-smoke LEG 1 on the real wrapper: 10 optimizer
steps + 3 eval loops, `hook_calls 314`, `routed_calls 77,224`, zero fallbacks) and on `KS-MOKA-3`.**

#### Budget running total (honest)

| item | planned | actual |
|---|---|---|
| GPU smoke | 0.20 | **0.535** (13537 11 s + 13551 `00:31:55`) |
| MokA-ZH SFT | 3.10 | **3.414** (`03:24:49`) |
| KS-MOKA-0b + extraction + heads | 1.35 | pending |
| **total** | **4.65** (cap 4.7) | **projected ≈ 5.30** |

Over the 4.7 planning cap. Per standing instruction the cap is a planning figure — **recorded, and
execution continues; no healthy run was killed.** Note the SFT overrun is *not* routing compute (that
was +1.72 %); the wall/`train_runtime` gap (`03:24:49` vs `2:26:24`) is dominated by the three
~301 s per-epoch eval passes plus model build/load, all of which the frozen recipe prescribes.

## S4' — job 2 of 2: `sbatch scripts/slurm/moka_extract_head.sbatch` → **job `13566`** (8 CPU / 64 G / 1 A100)

Submitted **only after** `sacct` reported 13552 terminal (`COMPLETED 0:0`) — sequential per §1.0 /
DEV-4, never `--dependency=afterok`. At submit the account queue was **empty**, so peak footprint of
this family is 8 CPU / 64 G / 1 GPU and the never-two-16-CPU rule is satisfied by construction.

**Submit-instant re-verification (all MATCH / CLEAN / ABSENT):**
- frozen A–G + prereg re-hashed: `A 6b7bdb6c…`, `B fae40487…`, `C 75bb8156…`, `D bd258553…`,
  `E 020dd10b…`, `F fd1b7f29…`, `G 51b883e9…`, `FROZEN dc3f1078…` — **MATCH**;
  `src/run_rac.py b85eb72a…` — **MATCH**.
- §4 condition 4 — `run_one()` block still **byte-identical** to `enc3seed_zh_b3.sbatch:42-83`
  (`diff` empty, block sha `286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101`).
- `loss.py` / `classifier.py` / `retrieval.py` **git-clean**; LF gitlink `a912747c…` unchanged.
- **DEV-J input asserts** both satisfied: `logging/lora/MHC_zh/adapter_model.safetensors` (banked
  generic, the `KS-MOKA-0b` input) and `logging/lora/MHC_zh_moka/adapter_model.safetensors` (job 1's
  MokA adapter) both present ⇒ job 2's `exit 2` guard is clear.
- **Output surfaces ABSENT:** no `*-moka_HF*` / `*-um*` caches, no `RAC_video_moka*` group, no
  `*moka_HF*.trainlog` ⇒ `--force False` cannot trip `run_rac.py:1059-1062` and nothing can clobber.
- Disk **1,570 G** avail (disk_guard pruning during job 1 freed a large amount).

**Stage order this job will execute:** A0 `KS-MOKA-0b` merge-drift (3 splits × 2 streams, **0
test-touch** — features carry no labels) → A1 MokA extraction (`--moka`, 3 splits) → **Stage S shape
sanity, which aborts BEFORE any budgeted test read** → Stage B 3 head-seeds (`RAC_video_moka`,
seeds 0/1/2) = **the 3 and only 3 budgeted test evaluations**.

**Pre-committed handling of the §3.4 branch.** Per the coordinator's standing instruction, if
`KS-MOKA-0b`'s full-N measurement puts **any** of the 6 (split × stream) cells below the **0.9999**
bar, the executor will **report before making any further submission**: the same-path unmerged-floor
head run is a separate **+3 test-evaluation** spend against the §F0.1 reservation and requires
explicit coordinator acknowledgement. The GPU-smoke LEG-4 rehearsal (8 items) already forecasts this
will fire (`img 0.99977`, `text 0.99946`). **No extra job will be submitted on the executor's own
authority.**

### ⚠ `KS-MOKA-0b` (job 2 Stage A0) — **FIRES. All 6 (split × stream) cells are below the 0.9999 bar.**

Raw, from `slurm/logs/moka_eh_13566.out:155-162`:

| split | stream | mean per-item cos | min per-item cos | ≥ 0.9999 ? |
|---|---|---|---|---|
| train (N=579) | `img_feats` | **0.99984443** | 0.99894041 | **NO** |
| train | `text_feats` | **0.99957055** | 0.99807644 | **NO** |
| dev_seen (N=78) | `img_feats` | **0.99987048** | 0.99896044 | **NO** |
| dev_seen | `text_feats` | **0.99954879** | 0.99750400 | **NO** |
| test_seen (N=149) | `img_feats` | **0.99983090** | 0.99933851 | **NO** |
| test_seen | `text_feats` | **0.99955094** | 0.99884427 | **NO** |

```
[KS-MOKA-0b] WORST mean per-item cosine over all 6 (split x stream) = 0.99954879
[KS-MOKA-0b] >= 0.9999 ? False
```
`-um` caches written for all 3 splits with `zero-vector videos=0` in each (l.139/145/154).
**This probe reads no labels ⇒ 0 test-touch, as pre-declared.**

**The pre-declared §3.4 consequence, transcribed verbatim:** *"If ANY cell < 0.9999, a same-path
unmerged floor head run (3 seeds, +0.05 GPU-h, +3 test evaluations) becomes MANDATORY before any
verdict, and the arm is then paired against THAT floor instead of 13150."* The trigger is "ANY cell";
here **all six** fail, so the branch is not marginal.

Structure of the drift (raw observation, no interpretation): the **text** stream drifts ~3× further
from merged than the **image** stream (text means ~0.99955, image means ~0.99985) and carries the
worst per-item minima (0.99750 on dev). The GPU-smoke LEG-4 8-item rehearsal predicted both the
magnitude and the ordering (`img 0.99977` > `text 0.99946`). This is the merged-vs-unmerged bf16
accumulation-order effect recon §3.6 / prereg §3.4 anticipated — **not** a MokA effect: it is measured
on the **banked generic adapter** with routing entirely absent.

**EXECUTOR ACTION: NONE. HOLDING for coordinator acknowledgement.** Per the coordinator's standing
instruction and the executor's own pre-commitment recorded above, the same-path unmerged-floor head
run is a **separate +3 test-evaluation spend** against the §F0.1 reservation and **will not be
submitted on executor authority**. Job 2 itself continues unmodified — it is the frozen,
pre-registered bite (Stage A1 → Stage S → Stage B's 3 budgeted reads) and needs no further approval.

### §3.4 contingency — coordinator **ACK GRANTED**; throwaway runner built and verified (NOT submitted yet)

Authorized against the §F0.1 contingent reservation: 3 head-seeds on tag
`Qwen2.5-VL-7B-Instruct-LoRA_HF-um`, ≈0.05 GPU-h, **+3 test evaluations**. Pairing switches to the
unmerged floor per §3.4 verbatim. This is the **frozen contingency executing on its own
pre-registered terms — not a scope expansion**; the verdict reviewer is to be instructed accordingly.

**Condition (1) — `run_one()` byte-identity: satisfied at the STRONGEST level.** The block was not
re-typed or adapted; it is `sed -n '112,153p'`-lifted from frozen artifact **F** at build time. Both
`diff`s are **empty** and all three shas agree:

```
286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101  enc3seed_zh_b3.sbatch:42-83   (anchor, produced floor 13150)
286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101  moka_extract_head.sbatch:112-153  (FROZEN artifact F, the arm)
286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101  moka_umfloor.sbatch:61-102        (THROWAWAY, this contingency)
```
The condition allowed the block to differ in the model tag and group_name; it does **not** — those
two differences live entirely in the **surrounding variables**, so the executed block is byte-identical
to the one that produced the banked floor:

| | arm (frozen **F**) | unmerged floor (throwaway) |
|---|---|---|
| `LORA` | `Qwen2.5-VL-7B-Instruct-LoRA-moka_HF` | `Qwen2.5-VL-7B-Instruct-LoRA_HF-um` |
| `GROUP_NAME` | `RAC_video_moka` | `RAC_video_moka_umfloor` |
| `DS` / `WARMUP` | `MHC_zh` / `5` | `MHC_zh` / `5` (same) |

**Conditions (2)-(4).** (2) The runner lives **only** in the executor scratchpad and is **never
committed as a frozen artifact** — it is reproduced in this record by provenance (lift command + block
sha), not vendored. (3) It is submitted **only after `13566` reaches terminal**, sequentially, even
though 8 + 8 CPU would coexist safely — sequential keeps the ledger clean. (4) Its RAW transcription
will carry the **same per-seed dual-protocol readout as the arm**, with **no verdict language**.

Additional guards carried over from the frozen job: a **Stage-S-equivalent shape-sanity block runs
before any budgeted test read** (asserts all 3 `-um` caches are `(N,3584)`, finite, id-aligned,
non-zero) so a malformed cache aborts the job *before* spending test evaluations;
`RAC_video_moka_umfloor*` verified **ABSENT** ⇒ fresh group, `--force False` cannot trip
`run_rac.py:1059-1062`; `bash -n` **SYNTAX_OK**; no `--time`; 8 CPU / 64 G / 1 A100.

---

# S5' — RAW TRANSCRIPTION (executor applies **NO** gates; prereg §3.10)

**Every number below was re-read from the raw trainlogs at transcription time and carries its source
line. No number is copied from a summary. No verdict, pass/fail, KS-MOKA-1 determination, or FORMAL-bar
comparison is made here — those belong to the independent 0-context verdict reviewer.**

Two parsers were run over every log: the parser **embedded in the frozen sbatch** (`RESULT_ROW` lines)
and an **independent executor-written cross-parser** (different implementation: token-splitting rather
than one regex, independent val-selection). **They agree on all 24 arm+floor values.** The cross-parser
was pre-validated by reproducing all 12 numbers of prereg §2.1 from the banked floor 13150.

## S5'.1 — Job inventory

| job | what | state | elapsed |
|---|---|---|---|
| 13537 | GPU smoke (throwaway harness defect) | FAILED 1:0 | 00:00:11 |
| 13551 | GPU smoke, 5 legs | COMPLETED 0:0 | 00:31:55 |
| 13552 | **job 1** — MokA-ZH LoRA-SFT | COMPLETED 0:0 | 03:24:49 |
| 13566 | **job 2** — KS-MOKA-0b + `--moka` extraction + 3 arm head-seeds | COMPLETED 0:0 | 01:12:42 |
| 13573 | **§3.4 contingency** — 3 unmerged-floor head-seeds | COMPLETED 0:0 | 00:24:46 |

## S5'.2 — Extraction stats + zero-vector guards (job 13566)

`[moka] routed layers: 196 | lora_A_v tensors loaded: 196` (l.169) — the silent-drop trap defeated.

| split | N | Dv | Dt | zero-vector videos | Stage-S nonzero img / txt |
|---|---|---|---|---|---|
| train | 579 | 3584 | 3584 | **0** | 579 / 579 |
| dev_seen | 78 | 3584 | 3584 | **0** | 78 / 78 |
| test_seen | 149 | 3584 | 3584 | **0** | 149 / 149 |

(l.239/245/254 saves; l.255-257 Stage-S.) The `-um` caches (job 13566 Stage A0, l.139/145/154) and the
umfloor job's own Stage-S-equivalent (13573 l.669-671) report the same N and **zero-vector videos=0**.

## S5'.3 — PER-SEED DUAL-PROTOCOL READOUT (line-numbered, cross-parsed)

**MokA arm** — `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA-moka_HF_seed{s}_13566.trainlog`

| seed | val-sel ep | val-sel acc / mF1 | line | final ep | final acc / mF1 | line |
|---|---|---|---|---|---|---|
| 0 | 23 | 0.8121 / 0.7679 | 244 | 29 | 0.8456 / 0.8107 | 299 |
| 1 | 27 | 0.8389 / 0.8039 | 286 | 29 | 0.8456 / 0.8080 | 305 |
| 2 | 28 | 0.8456 / 0.8107 | 289 | 29 | 0.8456 / 0.8107 | 299 |

**UNMERGED floor (§3.4 contingency, the BINDING pairing)** —
`enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF-um_seed{s}_13573.trainlog`

| seed | val-sel ep | val-sel acc / mF1 | line | final ep | final acc / mF1 | line |
|---|---|---|---|---|---|---|
| 0 | 5 | 0.7718 / 0.7259 | 80 | 29 | 0.8456 / 0.8181 | 297 |
| 1 | 25 | 0.8121 / 0.7742 | 267 | 29 | 0.8255 / 0.7956 | 304 |
| 2 | 26 | 0.8322 / 0.8023 | 271 | 29 | 0.8456 / 0.8181 | 299 |

*Raw note (no interpretation): umfloor seed 0 selected **epoch 5**, the earliest epoch the warmup rule
admits, on Val acc 0.8718 / roc 0.9207 (l.79).*

**MERGED floor 13150** (the prereg §2.1 floor; **secondary and non-binding after §3.4**) —

| seed | val-sel ep | val-sel acc / mF1 | line | final ep | final acc / mF1 | line |
|---|---|---|---|---|---|---|
| 0 | 20 | 0.8322 / 0.8023 | 220 | 29 | 0.8456 / 0.8181 | 302 |
| 1 | 26 | 0.8255 / 0.7956 | 275 | 29 | 0.8389 / 0.8113 | 303 |
| 2 | 19 | 0.8389 / 0.8065 | 207 | 29 | 0.8523 / 0.8226 | 298 |

## S5'.4 — PAIRED DELTAS, arm − UNMERGED floor (§3.4 binding pairing)

| protocol | seed | arm acc/mF1 | floor acc/mF1 | Δacc | ΔmF1 |
|---|---|---|---|---|---|
| val-sel | 0 | 0.8121 / 0.7679 | 0.7718 / 0.7259 | **+0.0403** | **+0.0420** |
| val-sel | 1 | 0.8389 / 0.8039 | 0.8121 / 0.7742 | **+0.0268** | **+0.0297** |
| val-sel | 2 | 0.8456 / 0.8107 | 0.8322 / 0.8023 | **+0.0134** | **+0.0084** |
| **val-sel** | **mean** | | | **+0.0268** (sd 0.0135) | **+0.0267** (sd 0.0170) |
| | *sign +* | | | *3/3* | *3/3* |
| final-ep | 0 | 0.8456 / 0.8107 | 0.8456 / 0.8181 | **+0.0000** | **−0.0074** |
| final-ep | 1 | 0.8456 / 0.8080 | 0.8255 / 0.7956 | **+0.0201** | **+0.0124** |
| final-ep | 2 | 0.8456 / 0.8107 | 0.8456 / 0.8181 | **+0.0000** | **−0.0074** |
| **final-ep** | **mean** | | | **+0.0067** (sd 0.0116) | **−0.0008** (sd 0.0114) |
| | *sign +* | | | *1/3* | *1/3* |

## S5'.5 — PAIRED DELTAS, arm − merged floor 13150 (secondary, non-binding)

| protocol | Δacc mean (sd) | ΔmF1 mean (sd) | sign + acc | sign + mF1 | per-seed Δacc |
|---|---|---|---|---|---|
| val-sel | **+0.0000** (0.0177) | **−0.0073** (0.0236) | 2/3 | 2/3 | −0.0201, +0.0134, +0.0067 |
| final-ep | **+0.0000** (0.0067) | **−0.0075** (0.0043) | 1/3 | 0/3 | +0.0000, +0.0067, −0.0067 |

## S5'.6 — FLOOR-vs-FLOOR: unmerged 13573 − merged 13150 (the merge-drift's own downstream size)

| protocol | Δacc mean | ΔmF1 mean | per-seed Δacc |
|---|---|---|---|
| val-sel | **−0.0268** | **−0.0340** | −0.0604, −0.0134, −0.0067 |
| final-ep | **−0.0067** | **−0.0067** | 0.0000, −0.0134, −0.0067 |

**Flagged for the verdict reviewer, stated as arithmetic only:** at val-sel the unmerged floor sits
**−0.0268** acc below the merged floor, and the arm sits **+0.0268** acc above the unmerged floor —
the same magnitude. Both facts are measured; their relationship is the reviewer's to weigh. The
floor-vs-floor gap is produced with **routing entirely absent** (same banked generic adapter, merged
vs unmerged forward only).

## S5'.7 — `KS-MOKA-0b` (job 13566 Stage A0), verbatim, all 6 cells

| split | stream | mean per-item cos | min per-item cos | ≥ 0.9999 ? |
|---|---|---|---|---|
| train (579) | img | 0.99984443 | 0.99894041 | **NO** |
| train | text | 0.99957055 | 0.99807644 | **NO** |
| dev_seen (78) | img | 0.99987048 | 0.99896044 | **NO** |
| dev_seen | text | **0.99954879** | 0.99750400 | **NO** |
| test_seen (149) | img | 0.99983090 | 0.99933851 | **NO** |
| test_seen | text | 0.99955094 | 0.99884427 | **NO** |

`WORST mean per-item cosine over all 6 (split × stream) = 0.99954879`; `>= 0.9999 ? False`
(`slurm/logs/moka_eh_13566.out:155-162`). **0 test-touch** — the probe reads no labels.

**NON-BINDING STRUCTURAL NOTE (recorded at coordinator request, for the verdict reviewer and any
paper text):** the **text** stream drifts ≈**3× further** from the merged reference than the **image**
stream — text means ≈0.99955 vs image means ≈0.99985, and text holds the worst per-item minima
(0.99750 on dev_seen vs 0.99894 for image). Measured on the **banked generic adapter with routing
entirely absent**, so it is a property of merged-vs-unmerged **bf16 accumulation order**, not of MokA.
The GPU-smoke 8-item rehearsal predicted both magnitude and ordering (img 0.99977 > text 0.99946).
**This note is non-binding and carries no pre-registered threshold.**

## S5'.8 — `KS-MOKA-2` — reported as a **NON-DEGENERACY FLOOR ONLY** (reviewer note **N1**)

`min 1.4039 | median 1.4170 | max 1.4292` over **196** layers, true even-sample median
(`logging/slurm/lora_sft_moka_13552.out:1994-1995`; `refine-logs/MOKA_KS2_routing_report.json`).

**N1 applied verbatim: this is NOT evidence that routing is real and must never be reported as such.**
N1 measured at freeze that two *independent Kaiming draws* at the deployed `A` shape already sit at
**1.4136**, while a *trained* `lora_A`'s total 3-epoch displacement is median **0.0506** / max
**0.1267**. The measured 1.4170 is therefore indistinguishable from the independent-draws value; the
check establishes only that the two down-projections did **not** collapse onto each other.
**Routing-activity evidence is `fallback_calls == 0` and `KS-MOKA-3`, not this number.**

Supporting `fallback_calls` evidence: GPU-smoke LEG 1 on the real `PeftModelForCausalLM` —
`hook_calls 314`, `routed_calls 77,224`, **`fallback_calls 0`** across 10 optimizer steps *and* 3 eval
loops; and job 13552 ran the full 3 epochs under `MOKA_STRICT=1` **without a single strict raise**.

## S5'.9 — `KS-MOKA-3` stream decomposition (CPU, $0, **train + dev_seen only, ZERO test-touch**)

Machinery: `scripts/analysis/encoder_swap_geometry.py` imported verbatim; movement rule transcribed
from `hatemm_lora_stream_decomp.py:232-237` (**MOVED** iff dAUC ≥ +0.010 train-LOO **and** ≥ +0.005 dev,
same sign; **FLAT** iff |dAUC| < 0.010 train-LOO). K=20. n_train 579, n_dev 78, id/label alignment
**EXACT**. Reported against **both** floors because §3.4 switched the pairing.

**(A) vs UNMERGED floor (§3.4 same-path pairing)** — `MOKA_KS3_stream_decomp_vs_unmerged.json`

| stream | floor trLOO | moka trLOO | Δ trLOO | floor dev | moka dev | Δ dev | mechanical label |
|---|---|---|---|---|---|---|---|
| img | 0.7124 | 0.7261 | **+0.0137** | 0.8307 | 0.8186 | **−0.0121** | **AMBIGUOUS** |
| text | 0.9280 | 0.9272 | **−0.0007** | 0.9279 | 0.9386 | **+0.0107** | **FLAT** |
| concat | 0.9137 | 0.9137 | +0.0000 | 0.9086 | 0.9229 | +0.0143 | FLAT |

**(B) vs MERGED floor (§3.7 literal wording)** — `MOKA_KS3_stream_decomp_vs_merged.json`

| stream | floor trLOO | moka trLOO | Δ trLOO | floor dev | moka dev | Δ dev | mechanical label |
|---|---|---|---|---|---|---|---|
| img | 0.7141 | 0.7261 | **+0.0120** | 0.8143 | 0.8186 | **+0.0043** | **AMBIGUOUS** |
| text | 0.9254 | 0.9272 | **+0.0018** | 0.9314 | 0.9386 | **+0.0071** | **FLAT** |
| concat | 0.9131 | 0.9137 | +0.0006 | 0.9086 | 0.9229 | +0.0143 | FLAT |

Under **both** floors the image stream is **AMBIGUOUS** (train-LOO clears +0.010 but the dev leg fails
the +0.005 same-sign requirement — marginally under (B), with opposite sign under (A)) and the text
stream is **FLAT**. **These are the mechanical rule labels only.** §3.7's three pre-declared readings
("text moved" / "image moved, head flat" / "neither moved") are the **independent reviewer's** to
apply; the executor does not select among them. Provenance cross-check: the merged floor's text
train-LOO AUC **0.9254** reproduces F45's published ZH value **0.925**.

## S5'.10 — Test-touch ledger

| item | budgeted | spent |
|---|---|---|
| arm head-seeds (job 13566 Stage B) | 3 | **3** |
| §3.4 contingent unmerged-floor head-seeds (job 13573) | +3 (F0.1 reservation, coordinator-ACKed) | **3** |
| `KS-MOKA-0b` | 0 | **0** |
| `KS-MOKA-3` | 0 | **0** |
| **total** | **6** | **6** |

**No unbudgeted test evaluation occurred.** `--force False` throughout; groups `RAC_video_moka` and
`RAC_video_moka_umfloor` were both fresh.
