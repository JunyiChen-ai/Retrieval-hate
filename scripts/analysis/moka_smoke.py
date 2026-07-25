#!/usr/bin/env python
"""MokA routed-LoRA CPU smoke — the KS-MOKA-0 machinery gate (refine-logs/MOKA_PREREG.md §4).

ZERO GPU. Run from the repo root inside conda `HateVideo`:

    python scripts/analysis/moka_smoke.py                # all CPU checks
    python scripts/analysis/moka_smoke.py --skip-mask    # skip the processor/tokenizer check

Checks (all must PASS before any SLURM submission):
  S1  install: `install_moka` converts every peft `lora.Linear` in place, adds `lora_A_v`, and
      registers the input_ids modality pre-hook.
  S2  IDENTITY CONTROL: with `lora_A_v` weights copied from `lora_A`, the routed forward must equal
      upstream PEFT's `Linear.forward` (fp32) for all-text, all-vision and mixed masks.
  S3  strict guard: a routed layer called with no mask RAISES (never a silent plain-LoRA null-op).
  S4  grad flow: after one backward both `lora_A` and `lora_A_v` AND the shared `lora_B` have
      non-zero grads; no `requires_grad=True` parameter has `grad is None` (MokA's D1 trap).
  S5  save/load round-trip: `lora_A_v` survives `save_pretrained` and is restored EXACTLY by
      `install_moka` + `load_moka_a_v`; a plain `from_pretrained` alone silently drops it.
  S6  merge guard: `merge_and_unload()` / `merge()` / `get_delta_weight()` RAISE.
  S7  parameter budget: the deployed 7-projection x 28-layer r=16 shape gives 58,490,880 MokA params
      vs 40,370,176 deployed = 1.4489x, with per-token rank 16 unchanged.
  S8  mask correctness on a REAL tokenized ZH SFT sample: `(input_ids==151655)|(input_ids==151656)`
      count must equal the processor's grid_thw arithmetic `sum(prod(grid_thw_i) // merge_size**2)`,
      and vision|text must cover the sequence exactly once.
  S9  DEPLOYED WRAPPER CLASS: the same machinery on a REAL `PeftModelForCausalLM`
      (`task_type=CAUSAL_LM`, the class BOTH production paths build) called the way the trainer and
      the extractor call it — `model(**inputs)`: the hook must fire, every routed layer must route,
      `fallback_calls == 0`, and the tied-`A_v` forward must equal upstream PEFT bit-exactly.
      S1-S8 run on the GENERIC `PeftModel` (no `task_type`) and are structurally blind to any
      defect that lives in the production wrapper's call chain — this is the check that closes it.
"""

import argparse
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src", "moka"))

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from peft.tuners.lora.layer import Linear as PeftLoraLinear  # noqa: E402
from peft.utils.save_and_load import get_peft_model_state_dict  # noqa: E402

from routed_lora import (  # noqa: E402
    IMAGE_PAD_ID,
    VIDEO_PAD_ID,
    MokaLinear,
    build_vision_mask,
    install_moka,
    load_moka_a_v,
    moka_param_report,
    moka_routing_report,
    moka_stats,
    reset_moka_stats,
    set_route_impl,
    _STASH,
)

FAILS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("  [{}] {} {}".format(tag, name, detail), flush=True)
    if not ok:
        FAILS.append(name)


class Toy(nn.Module):
    """Minimal stand-in for the Qwen decoder: an embedding + two targeted projections."""

    def __init__(self, vocab=151700, hid=32, out=48):
        super().__init__()
        self.embed = nn.Embedding(vocab, hid)
        self.q_proj = nn.Linear(hid, hid, bias=False)
        self.down_proj = nn.Linear(hid, out, bias=False)

    def forward(self, input_ids=None, **kw):
        h = self.embed(input_ids)
        return self.down_proj(self.q_proj(h))


def make_peft_toy(seed=0):
    torch.manual_seed(seed)
    toy = Toy().to(torch.float32)
    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "down_proj"], init_lora_weights=True,
    )
    return get_peft_model(toy, cfg)


def s1_s3_install_identity_strict():
    print("S1/S2/S3 — install, identity control, strict guard")
    m = make_peft_toy()
    n_lora = sum(1 for mod in m.modules() if isinstance(mod, PeftLoraLinear))
    n = install_moka(m, strict=True)
    check("S1.install-count", n == n_lora == 2, "converted {} of {}".format(n, n_lora))
    check("S1.class", all(isinstance(mod, MokaLinear) for mod in m.modules()
                          if isinstance(mod, PeftLoraLinear)), "")
    check("S1.A_v-present", all(set(mod.lora_A_v.keys()) == set(mod.lora_A.keys())
                                for mod in m.modules() if isinstance(mod, MokaLinear)), "")
    # The hook MUST sit on the OUTERMOST module (the one the trainer / extractor call via
    # `__call__`), never on `get_base_model()` — see routed_lora.install_moka and S9.  `__dict__`
    # is read directly because PeftModel forwards missing attributes down to the wrapped model,
    # so plain `getattr` cannot tell WHICH module owns the handle.
    check("S1.hook-on-outermost", "_moka_hook" in m.__dict__ and m._moka_hook is not None,
          "registered on {}".format(type(m).__name__))
    check("S1.hook-not-on-base", "_moka_hook" not in m.get_base_model().__dict__, "")
    check("S1.A_v-differs-from-A_t",
          all(not torch.equal(mod.lora_A_v["default"].weight, mod.lora_A["default"].weight)
              for mod in m.modules() if isinstance(mod, MokaLinear)),
          "(fresh independent Kaiming draw)")

    # --- S3: no mask -> raise (before any mask is stashed) ---
    _STASH.mask = None
    layer = [mod for mod in m.modules() if isinstance(mod, MokaLinear)][0]
    x = torch.randn(2, 7, layer.in_features)
    try:
        layer(x)
        check("S3.strict-raises", False, "no exception")
    except RuntimeError as e:
        check("S3.strict-raises", "modality mask absent" in str(e), "")

    # --- S2: tie A_v := A_t, then routed forward must equal plain PEFT forward ---
    for mod in m.modules():
        if isinstance(mod, MokaLinear):
            mod.lora_A_v["default"].weight.data.copy_(mod.lora_A["default"].weight.data)
            # give lora_B a non-trivial value: zero-init B would make every variant trivially equal
            nn.init.normal_(mod.lora_B["default"].weight, std=0.02)

    torch.manual_seed(1234)
    bsz, seq = 2, 11
    mods = [mod for mod in m.modules() if isinstance(mod, MokaLinear)]
    for label, mk in (
        ("all-text", lambda: torch.zeros(bsz, seq, dtype=torch.bool)),
        ("all-vision", lambda: torch.ones(bsz, seq, dtype=torch.bool)),
        ("mixed", lambda: torch.rand(bsz, seq) < 0.4),
    ):
        mask = mk()
        for j, mod in enumerate(mods):
            x = torch.randn(bsz, seq, mod.in_features)
            _STASH.mask = mask
            set_route_impl("dense")
            routed = mod(x)
            set_route_impl("gather")
            gathered = mod(x)
            set_route_impl("dense")
            mod.__class__ = PeftLoraLinear          # temporarily restore upstream forward
            plain = mod(x)
            mod.__class__ = MokaLinear
            d = (routed - plain).abs().max().item()
            dg = (routed - gathered).abs().max().item()
            check("S2.identity[{}][m{}]".format(label, j), d == 0.0, "max|delta| = {:.3e}".format(d))
            # cross-impl agreement: an indexing bug in EITHER form would show up far above 1e-5
            check("S2.dense-vs-gather[{}][m{}]".format(label, j), dg < 1e-5,
                  "max|delta| = {:.3e}".format(dg))
    _STASH.mask = None


def s4_grad_flow():
    print("S4 — grad flow (both A's + shared B; MokA's D1 dead-parameter trap)")
    m = make_peft_toy(seed=3)
    install_moka(m, strict=True)
    # PEFT zero-inits lora_B, and dL/dA = B^T(...) == 0 at step 0 for EVERY LoRA (routed or not).
    # Perturb B so the grad-flow check is made at a non-degenerate point (i.e. "after step 1").
    for mod in m.modules():
        if isinstance(mod, MokaLinear):
            nn.init.normal_(mod.lora_B["default"].weight, std=0.02)
    ids = torch.randint(0, 151000, (2, 9))
    ids[:, 2:5] = IMAGE_PAD_ID
    ids[0, 6] = VIDEO_PAD_ID
    reset_moka_stats()
    out = m(input_ids=ids)
    out.square().mean().backward()
    st = moka_stats()
    check("S4.hook-fired", st["hook_calls"] == 1 and st["fallback_calls"] == 0, str(st))
    check("S4.routed-calls", st["routed_calls"] == 2, str(st))
    dead = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    check("S4.no-dead-params", not dead, "dead={}".format(dead[:3]))
    for mod_name, mod in m.named_modules():
        if not isinstance(mod, MokaLinear):
            continue
        for key in ("lora_A", "lora_A_v", "lora_B"):
            g = getattr(mod, key)["default"].weight.grad
            check("S4.grad[{}][{}]".format(mod_name.split(".")[-1], key),
                  g is not None and float(g.abs().sum()) > 0.0,
                  "|grad|_1 = {:.4e}".format(float(g.abs().sum())) if g is not None else "None")
    rep = moka_param_report(m)
    check("S4.param-report", rep["lora_A_v"] == rep["lora_A_t"] and rep["lora_B"] > 0, str(rep))
    rows = moka_routing_report(m)
    check("S4.routing-report", len(rows) == 2 and all(r["rel_fro_diff"] > 0 for r in rows),
          "rel_fro_diff = {}".format([round(r["rel_fro_diff"], 4) for r in rows]))


def s5_s6_roundtrip_merge():
    print("S5/S6 — save/load round-trip + merge guard")
    m = make_peft_toy(seed=7)
    install_moka(m, strict=True)
    for mod in m.modules():
        if isinstance(mod, MokaLinear):
            nn.init.normal_(mod.lora_B["default"].weight, std=0.02)
    sd = get_peft_model_state_dict(m)
    av_keys = [k for k in sd if "lora_A_v" in k]
    check("S5.state-dict-keeps-A_v", len(av_keys) == 2, "keys={}".format(av_keys))

    with tempfile.TemporaryDirectory() as td:
        m.save_pretrained(td)
        check("S5.safetensors-written", os.path.isfile(os.path.join(td, "adapter_model.safetensors")), "")

        # (a) plain reload WITHOUT install_moka: A_v silently dropped (documents the trap)
        fresh = Toy().to(torch.float32)
        p_plain = PeftModel.from_pretrained(fresh, td)
        has_av = any(hasattr(mod, "lora_A_v") for mod in p_plain.modules())
        check("S5.plain-reload-drops-A_v", not has_av, "(this is exactly why --moka loads it explicitly)")

        # (b) correct path: from_pretrained -> install_moka -> load_moka_a_v
        fresh2 = Toy().to(torch.float32)
        p2 = PeftModel.from_pretrained(fresh2, td)
        install_moka(p2, strict=True)
        n_av = load_moka_a_v(p2, td)
        check("S5.load_moka_a_v-count", n_av == 2, "loaded {}".format(n_av))
        src = {n: mod for n, mod in m.named_modules() if isinstance(mod, MokaLinear)}
        dst = {n: mod for n, mod in p2.named_modules() if isinstance(mod, MokaLinear)}
        check("S5.same-modules", set(src) == set(dst), "")
        exact = all(
            torch.equal(src[k].lora_A_v["default"].weight, dst[k].lora_A_v["default"].weight)
            and torch.equal(src[k].lora_A["default"].weight, dst[k].lora_A["default"].weight)
            and torch.equal(src[k].lora_B["default"].weight, dst[k].lora_B["default"].weight)
            for k in src
        )
        check("S5.weights-bit-exact", exact, "")

        # (c) a GENERIC adapter must be REFUSED by load_moka_a_v
        gm = make_peft_toy(seed=11)
        with tempfile.TemporaryDirectory() as td2:
            gm.save_pretrained(td2)
            p3 = PeftModel.from_pretrained(Toy().to(torch.float32), td2)
            install_moka(p3, strict=True)
            try:
                load_moka_a_v(p3, td2)
                check("S5.generic-adapter-refused", False, "no exception")
            except RuntimeError as e:
                check("S5.generic-adapter-refused", "NO lora_A_v tensors" in str(e), "")

    # --- S6 merge guard ---
    for fn, name in ((lambda: m.merge_and_unload(), "merge_and_unload"),
                     (lambda: [mod for mod in m.modules() if isinstance(mod, MokaLinear)][0].merge(), "merge"),
                     (lambda: [mod for mod in m.modules() if isinstance(mod, MokaLinear)][0].get_delta_weight("default"),
                      "get_delta_weight")):
        try:
            fn()
            check("S6.{}-raises".format(name), False, "no exception")
        except RuntimeError:
            check("S6.{}-raises".format(name), True, "")


def s7_param_budget():
    print("S7 — parameter budget (deployed shape: 28 layers x 7 projections, r=16)")
    hid, inter, kv, r, layers = 3584, 18944, 512, 16, 28
    proj = [(hid, hid), (hid, kv), (hid, kv), (hid, hid), (hid, inter), (hid, inter), (inter, hid)]
    plain = sum(r * i + o * r for i, o in proj) * layers
    moka = sum(2 * r * i + o * r for i, o in proj) * layers
    check("S7.plain==banked-40,370,176", plain == 40370176, "{}".format(plain))
    check("S7.moka==58,490,880", moka == 58490880, "{}".format(moka))
    check("S7.ratio==1.4489", abs(moka / plain - 1.4489) < 5e-5, "{:.6f}x".format(moka / plain))


def s8_real_mask(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    print("S8 — modality mask on a REAL tokenized ZH SFT sample")
    rec_path = os.path.join(REPO, "data", "lora_sft", "MHC_zh", "train.json")
    if not os.path.isfile(rec_path):
        check("S8.data-present", False, rec_path)
        return
    rec = json.load(open(rec_path))[0]
    from PIL import Image
    from transformers import AutoProcessor

    imgs = [Image.open(p).convert("RGB") for p in rec["images"]]
    text = rec["messages"][0]["content"].replace(
        "<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    text = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(text)
    # 262144 == the deployed yaml's `image_max_pixels` (mhc_zh_qwen25vl_lora_sft.yaml:3). LLaMA-Factory
    # applies it by RESIZING THE PIL IMAGE before the processor
    # (llamafactory/data/mm_plugin.py:229-246,:358), not via the image processor's own max_pixels.
    import math as _math

    def _lf_resize(im, cap=262144):
        if im.width * im.height > cap:
            f = _math.sqrt(cap / (im.width * im.height))
            im = im.resize((int(im.width * f), int(im.height * f)))
        return im

    proc = AutoProcessor.from_pretrained(model_id)
    for tag, ims in (("deployed-cap", [_lf_resize(i) for i in imgs]), ("processor-default", imgs)):
        inputs = proc(text=[text], images=ims, return_tensors="pt")
        ids = inputs["input_ids"]
        mask = build_vision_mask(ids)
        grid = inputs["image_grid_thw"]
        merge = proc.image_processor.merge_size
        expect = int(sum(int(g.prod()) // (merge * merge) for g in grid))
        nv, nt = int(mask.sum()), int((~mask).sum())
        check("S8[{}].count-matches-grid_thw".format(tag), nv == expect,
              "mask {} vs grid_thw arithmetic {} (merge_size={}, {} images)".format(
                  nv, expect, merge, len(imgs)))
        check("S8[{}].partition-exact".format(tag), nv + nt == ids.numel(),
              "seq_len {} = vision {} + text {}  (vision share {:.1%})".format(
                  ids.numel(), nv, nt, nv / ids.numel()))
        check("S8[{}].ids-are-151655/151656".format(tag),
              set(ids[mask].unique().tolist()) <= {IMAGE_PAD_ID, VIDEO_PAD_ID},
              "unique = {}".format(sorted(set(ids[mask].unique().tolist()))))


def s9_production_wrapper_class():
    """S9 — the DEPLOYED PEFT wrapper class (`PeftModelForCausalLM`), CPU + offline.

    THE BLIND SPOT THIS CLOSES (codex gate 2026-07-26, P1-A). S1–S8 build their `LoraConfig`
    WITHOUT `task_type`, so `get_peft_model` returns the GENERIC `PeftModel`, whose `forward`
    reaches the base model through `__call__` (peft/peft_model.py:843-849) — a forward-pre hook
    registered anywhere in that chain fires. BOTH deployed paths instead set
    `task_type=CAUSAL_LM` (llamafactory/model/adapter.py:300-303 for job 1; the banked
    `adapter_config.json` `"task_type": "CAUSAL_LM"` for job 2), which dispatches to
    `PeftModelForCausalLM`, whose `forward` calls `self.base_model(...)` = `LoraModel.__call__`
    -> `BaseTuner.forward` -> `self.model.forward(*args, **kwargs)` (peft/tuners/tuners_utils.py:
    196-197) — a DIRECT `.forward()` that fires NO hook on the base model. A hook on
    `get_base_model()` is therefore dead on the production class while S1–S8 stay green.

    S9 instantiates that exact class over a tiny offline causal LM and asserts, through the
    trainer/extractor call shape `model(**inputs)`:  hook fires, every routed layer routes,
    `fallback_calls == 0`, and (with `lora_A_v` tied to `lora_A`) the wrapper's logits equal the
    plain-PEFT reference BIT-EXACTLY. It also probes the raw `.forward()` surface and asserts the
    no-silent-null-op invariant holds there too.
    """
    print("S9 — DEPLOYED PeftModelForCausalLM wrapper class (closes the P1-A blind spot)")
    from peft import TaskType  # noqa: PLC0415
    from peft.peft_model import PeftModelForCausalLM  # noqa: PLC0415
    from transformers import GPT2Config, GPT2LMHeadModel  # noqa: PLC0415

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=151700, n_layer=2, n_embd=32, n_head=2, n_positions=64)
    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["c_attn"], init_lora_weights=True,
    )
    m = get_peft_model(GPT2LMHeadModel(cfg).to(torch.float32).eval(), lcfg)
    check("S9.class-is-PeftModelForCausalLM", isinstance(m, PeftModelForCausalLM), type(m).__name__)

    for mod in m.modules():  # non-trivial B so the LoRA delta is not identically zero
        if isinstance(mod, PeftLoraLinear):
            nn.init.normal_(mod.lora_B["default"].weight, std=0.02)

    torch.manual_seed(7)
    ids = torch.randint(0, 50000, (1, 12))
    ids[:, 3:8] = IMAGE_PAD_ID
    ids[0, 9] = VIDEO_PAD_ID
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    with torch.no_grad():                      # reference: upstream PEFT, MokA not yet installed
        ref = m(**inputs).logits.clone()

    n = install_moka(m, strict=True)
    check("S9.install-count", n == 2, "routed {} lora.Linear layers".format(n))
    check("S9.hook-on-wrapper", "_moka_hook" in m.__dict__,
          "hook owner = {} (NOT get_base_model() = {})".format(
              type(m).__name__, type(m.get_base_model()).__name__))
    for mod in m.modules():                    # tie A_v := A_t -> routed == plain PEFT, exactly
        if isinstance(mod, MokaLinear):
            mod.lora_A_v["default"].weight.data.copy_(mod.lora_A["default"].weight.data)

    # --- the surface job 1 (Trainer.compute_loss) and job 2 (extractor:360) actually call ---
    _STASH.mask = None
    reset_moka_stats()
    with torch.no_grad():
        out = m(**inputs).logits
    st = moka_stats()
    check("S9.hook-fires-on-model(**inputs)", st["hook_calls"] > 0, str(st))
    check("S9.routed-calls", st["routed_calls"] == n, str(st))
    check("S9.no-fallback", st["fallback_calls"] == 0, str(st))
    d = (out - ref).abs().max().item()
    check("S9.identity-vs-plain-PEFT", d == 0.0, "max|delta| = {:.3e}".format(d))

    # --- raw `.forward()` surface (no `__call__`, so no hook can fire): must never silently
    # degrade to plain LoRA. With a clean stash it must RAISE under strict. ---
    _STASH.mask = None
    reset_moka_stats()
    try:
        with torch.no_grad():
            m.forward(**inputs)
        st2 = moka_stats()
        loud = st2["hook_calls"] > 0 and st2["fallback_calls"] == 0
        check("S9.direct-forward-no-silent-nullop", loud, "routed without raising: {}".format(st2))
    except RuntimeError as e:
        st2 = moka_stats()
        check("S9.direct-forward-no-silent-nullop",
              "modality mask absent" in str(e) and st2["routed_calls"] == 0,
              "raises (strict) instead of a silent plain-LoRA fallback: {}".format(st2))
    _STASH.mask = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mask", action="store_true", help="skip S8 (needs the local processor)")
    a = ap.parse_args()
    print("MokA CPU smoke — torch {}, peft {}".format(torch.__version__, __import__("peft").__version__))
    s1_s3_install_identity_strict()
    s4_grad_flow()
    s5_s6_roundtrip_merge()
    s7_param_budget()
    s9_production_wrapper_class()
    if not a.skip_mask:
        s8_real_mask()
    print("\n==== {} ====".format("ALL SMOKE CHECKS PASS" if not FAILS else "FAILURES: " + ", ".join(FAILS)))
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
