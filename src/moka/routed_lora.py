"""MokA modality-routed LoRA for Qwen2.5-VL (A-split only; shared B; NO cross-attention).

Adapted from GeWu-Lab/MokA (NeurIPS 2025), clone `external/baselines/MokA` @ commit `b28e834`,
file `VisualText/modified_peft/tuners/lora/layer.py:548-681` (the routed `forward`: gather/scatter
per modality through per-modality `lora_A`, then a SINGLE shared `lora_B`).  MokA is used under an
explicit USER RULING (2026-07-26): ungated, code may be used directly, credited in the paper.

WHAT IS PORTED (design authority: `refine-logs/MOKA_FORENSIC_RECON.md`, commit dbf30f1)
  * `layer.py:603-611` text tokens -> `lora_A['text']`, `:613-621` image tokens -> `lora_A['image']`,
    scattered back into one rank-space buffer;
  * `layer.py:655-657` the up-projection `lora_B` is SHARED (MokA hard-codes `lora_B['text']`).
WHAT IS DELIBERATELY *NOT* PORTED (recon §3.2, §1.4)
  * the question->visual cross-attention (`layer.py:627-653`) — second manipulated variable, untuned
    `attn_weight` whose own default is inconsistent (config 0.5 vs layer 1), per-sample Python loop;
  * MokA's dead `lora_B['image']` twin (recon D1: created, trained, never read — 27.5% of their LoRA
    params get zero gradient).  Here there is exactly ONE `lora_B`, so no dead parameter exists;
  * MokA's hard-coded `dtype=torch.bfloat16` on the adapters (recon D2) and their removed dtype casts
    (recon D7) — we follow upstream PEFT 0.14.0 dtype discipline instead;
  * MokA's modeling-file fork (they thread 3 positional masks through every projection).  We do not
    need it: Qwen2.5-VL `masked_scatter`s vision embeddings into the vision-pad positions IN PLACE
    (transformers 4.49 `modeling_qwen2_5_vl.py:1803-1809`), so a single mask computed once per batch
    from `input_ids` is positionally valid at the input of all 7 targeted projections in all 28
    decoder layers (recon §2.4).

DIVERGENCE FROM MokA THAT IS OURS AND IS LOAD-BEARING FOR THE IDENTITY CONTROL
  MokA pads both masks with False, so PAD positions receive no LoRA delta at all.  Here the text
  route is `~vision`, i.e. EVERY non-vision position (pad included) goes through `lora_A` — exactly
  how the deployed generic adapter treats those positions.  Consequence: `vision | ~vision` covers
  the sequence exactly once, so with `lora_A_v` weights tied to `lora_A` the routed forward is
  algebraically the plain PEFT forward (the KS-MOKA-0 identity control).  Our SFT/extraction both run
  `per_device_train_batch_size = 1`, so no pad position exists in practice anyway.

SHAPES / BUDGET (recon §3.3): `r_v = r_t = 16`, `alpha 32`, dropout 0.0, 7 projections x 28 layers.
Params 40,370,176 -> 58,490,880 = 1.4489x deployed; per-token rank and FLOPs are IDENTICAL (exactly
one `A` fires per token).
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
from peft.tuners.lora.layer import Linear as _PeftLoraLinear

# Qwen2.5-VL vision placeholder ids, read from the local tokenizer
# (`logging/lora/MHC_zh/added_tokens.json`): <|image_pad|> 151655, <|video_pad|> 151656.
IMAGE_PAD_ID = 151655
VIDEO_PAD_ID = 151656

# Fixed seed for the A_v Kaiming draw so the init is reproducible independently of global RNG state.
MOKA_INIT_SEED = 20260726

# Routing implementation.  MokA ships BOTH forms and the recon (§1.3) records that they are
# "mathematically identical given disjoint masks":
#   "gather" = VisualText `layer.py:603-621` (index_select the rows of each modality, scatter the
#              rank-space outputs back).  ONE `A` GEMM per modality on a row subset.
#   "dense"  = AudioVisualText `peft_hyper/tuners/lora.py:460-532` (run every `A` on the full
#              sequence, then select) -> ~2x the (tiny) `A` FLOPs, ZERO index/gather surface.
# DEFAULT = "dense", and that choice is LOAD-BEARING for the KS-MOKA-0 identity control: with
# `A_v` tied to `A_t`, "dense" reproduces upstream PEFT's forward BIT-EXACTLY (max|delta| == 0.0)
# because each `A` sees the identical full-sequence GEMM, whereas "gather" changes the GEMM's M
# dimension and therefore its BLAS blocking -> a measured fp32 residual of 5.96e-08 / 1.19e-07 on
# mixed masks (CPU, torch 2.6.0), which the recon's 0.0 threshold does not admit.  "gather" is kept
# so the smoke can cross-check the two formulations against each other (an indexing-bug guard).
_ROUTE_IMPL = os.environ.get("MOKA_ROUTE_IMPL", "dense")


class _Stash:
    """Module-level per-batch mask stash + instrumentation counters.

    IMPORTANT (recon §2.4): the mask is written by the forward-PRE hook and is NEVER cleared at
    end-of-forward, because gradient-checkpointing replays each decoder block during backward and
    the routed layers must still see the batch's mask.  It is overwritten once per batch.
    """

    mask: Optional[torch.Tensor] = None  # BoolTensor [B, S]; True = vision-pad position
    strict: bool = True                  # absent/mismatched mask -> raise instead of silent fallback
    routed_calls: int = 0
    fallback_calls: int = 0
    hook_calls: int = 0


_STASH = _Stash()


def set_route_impl(impl: str) -> str:
    """Select the routing formulation ('dense' = deployed/frozen, 'gather' = MokA VisualText)."""
    global _ROUTE_IMPL
    if impl not in ("dense", "gather"):
        raise ValueError("MokA: unknown MOKA_ROUTE_IMPL '{}' (expected dense|gather)".format(impl))
    _ROUTE_IMPL = impl
    return _ROUTE_IMPL


def moka_stats() -> dict:
    """Instrumentation readout (used by the smoke + the sbatch post-run print)."""
    return {
        "impl": _ROUTE_IMPL,
        "hook_calls": _STASH.hook_calls,
        "routed_calls": _STASH.routed_calls,
        "fallback_calls": _STASH.fallback_calls,
        "strict": _STASH.strict,
    }


def reset_moka_stats() -> None:
    _STASH.routed_calls = 0
    _STASH.fallback_calls = 0
    _STASH.hook_calls = 0


def build_vision_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """The modality mask, verbatim from the recon's locked decision (§2.4)."""
    return (input_ids == IMAGE_PAD_ID) | (input_ids == VIDEO_PAD_ID)


def _mask_pre_hook(module, args, kwargs):  # noqa: ANN001 - torch hook signature
    ids = kwargs.get("input_ids", None)
    if ids is None and args and torch.is_tensor(args[0]) and args[0].dtype in (torch.long, torch.int):
        ids = args[0]
    _STASH.mask = None if ids is None else build_vision_mask(ids)
    _STASH.hook_calls += 1
    return None


class MokaLinear(_PeftLoraLinear):
    """`peft.tuners.lora.layer.Linear` + a second down-projection `lora_A_v` for vision tokens.

    `lora_A_v` is registered under a name containing the `lora_` prefix on purpose, so that
    (a) `LoraModel._mark_only_adapters_as_trainable` (peft/tuners/lora/model.py:281-284, prefix
    `"lora_"`) leaves `requires_grad=True`, and (b) `get_peft_model_state_dict`
    (peft/utils/save_and_load.py:85) keeps the key -> it IS saved into `adapter_model.safetensors`.
    It is added to `adapter_layer_names` so `set_adapter` / `_move_adapter_to_device_of_base_layer`
    treat it exactly like `lora_A`.
    """

    adapter_layer_names = ("lora_A", "lora_A_v", "lora_B", "lora_embedding_A", "lora_embedding_B")

    # -- construction ---------------------------------------------------------------------------
    def moka_init(self, index: int) -> int:
        """Create `lora_A_v` mirroring every existing `lora_A` (Kaiming, PEFT `reset_lora_parameters`).

        `lora_B` stays zero-initialised, so `dW = 0` at step 0 exactly as standard LoRA.
        """
        if not hasattr(self, "lora_A_v"):
            self.lora_A_v = nn.ModuleDict({})
        made = 0
        for name, a_t in self.lora_A.items():
            if name in self.lora_A_v:
                continue
            # Everything that consumes RNG happens INSIDE fork_rng, so installing MokA leaves the
            # process-global RNG stream bit-identical to an un-patched run (nn.Linear's own
            # reset_parameters() would otherwise advance it).
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(MOKA_INIT_SEED + index + made)
                a_v = nn.Linear(a_t.in_features, a_t.out_features, bias=False)
                nn.init.kaiming_uniform_(a_v.weight, a=math.sqrt(5))  # PEFT layer.py:172, verbatim
            a_v.weight.data = a_v.weight.data.to(device=a_t.weight.device, dtype=a_t.weight.dtype)
            a_v.weight.requires_grad_(a_t.weight.requires_grad)
            self.lora_A_v[name] = a_v
            made += 1
        return made

    # -- merge guard (recon §3.5 item 6) --------------------------------------------------------
    def merge(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError(
            "MokA routed-LoRA has NO merged form: the delta is token-dependent (B@A_v@x on vision "
            "positions vs B@A_t@x elsewhere), so no single dW exists. Use the extractor's --moka "
            "path (unmerged adapter forward), never merge_and_unload()."
        )

    def unmerge(self, *args, **kwargs):  # noqa: ANN002, ANN003
        if not self.merged:
            return
        raise RuntimeError("MokA routed-LoRA cannot be unmerged (it is never merged).")

    def get_delta_weight(self, adapter):  # noqa: ANN001
        raise RuntimeError(
            "MokA routed-LoRA has no adapter-level delta weight (token-dependent routing)."
        )

    # -- forward --------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ANN002, ANN003
        # `adapter_names` is POPPED (upstream peft/tuners/lora/layer.py:598 does the same) so an
        # explicitly passed `adapter_names=None` cannot reach `self.base_layer(...)` at the routed
        # path below and TypeError out of `nn.Linear`.  It is re-inserted verbatim when it is not
        # None, so the mixed-batch delegation to upstream is unchanged.  No deployed call site
        # passes the kwarg at all, so this is inert for both jobs.
        adapter_names = kwargs.pop("adapter_names", None)
        if self.disable_adapters or self.merged or adapter_names is not None:
            if adapter_names is not None:
                kwargs["adapter_names"] = adapter_names
            return super().forward(x, *args, **kwargs)

        mask = _STASH.mask
        ok = (
            mask is not None
            and x.dim() == 3
            and mask.dim() == 2
            and mask.shape[0] == x.shape[0]
            and mask.shape[1] == x.shape[1]
        )
        if not ok:
            _STASH.fallback_calls += 1
            if _STASH.strict:
                raise RuntimeError(
                    "MokA: modality mask absent or shape-mismatched at a routed layer "
                    "(x={}, mask={}). Refusing to fall back to plain LoRA silently, which would "
                    "make the arm a null-op. Set MOKA_STRICT=0 to allow the plain fallback.".format(
                        tuple(x.shape), None if mask is None else tuple(mask.shape)
                    )
                )
            return super().forward(x, *args, **kwargs)

        _STASH.routed_calls += 1
        self._check_forward_args(x, *args, **kwargs)
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        bsz, seq = x.shape[0], x.shape[1]
        x_flat = x.reshape(-1, x.shape[-1])
        out_flat = result.reshape(-1, result.shape[-1])
        vis_flat = mask.reshape(-1).to(x.device)

        idx_v = idx_t = None
        if _ROUTE_IMPL == "gather":
            idx_v = vis_flat.nonzero(as_tuple=True)[0]
            idx_t = (~vis_flat).nonzero(as_tuple=True)[0]

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            a_t = self.lora_A[active_adapter]
            a_v = self.lora_A_v[active_adapter]
            lora_b = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            xin = x_flat.to(a_t.weight.dtype)
            if _ROUTE_IMPL == "gather":
                # MokA VisualText layer.py:600-621: one rank-space buffer, per-modality scatter.
                a_out = xin.new_zeros(xin.shape[0], self.r[active_adapter])
                if idx_t.numel() > 0:
                    a_out = a_out.index_put((idx_t,), a_t(dropout(xin.index_select(0, idx_t))))
                if idx_v.numel() > 0:
                    a_out = a_out.index_put((idx_v,), a_v(dropout(xin.index_select(0, idx_v))))
            else:
                # MokA AudioVisualText lora.py:460-532: both A's on the full sequence, then select.
                xd = dropout(xin)
                a_out = torch.where(vis_flat.unsqueeze(-1), a_v(xd), a_t(xd))
            # shared up-projection (MokA layer.py:657, `lora_B = self.lora_B['text']`).
            out_flat = out_flat + lora_b(a_out) * scaling

        return out_flat.reshape(bsz, seq, -1).to(torch_result_dtype)

    def __repr__(self) -> str:
        return "moka." + nn.Module.__repr__(self)


# ------------------------------------------------------------------------------------------------
# installation
# ------------------------------------------------------------------------------------------------
def install_moka(peft_model, strict: Optional[bool] = None, require_zero_dropout: bool = True) -> int:
    """Convert every `lora.Linear` of `peft_model` in place into a `MokaLinear` + install the hook.

    In-place class re-assignment (`obj.__class__ = MokaLinear`) is used rather than rebuilding and
    re-parenting modules: it preserves every existing submodule/parameter identity, so nothing that
    already holds a reference (optimizer groups are built later, but device maps / hooks are not)
    can go stale.  Returns the number of converted layers.
    """
    if strict is None:
        strict = os.environ.get("MOKA_STRICT", "1") != "0"
    _STASH.strict = bool(strict)

    n = 0
    for module in peft_model.modules():
        if isinstance(module, _PeftLoraLinear) and not isinstance(module, MokaLinear):
            if require_zero_dropout:
                for name, drop in module.lora_dropout.items():
                    if not isinstance(drop, nn.Identity):
                        raise RuntimeError(
                            "MokA install: lora_dropout for adapter '{}' is {} (not Identity). The "
                            "deployed recipe pins lora_dropout=0.0; a non-zero dropout draws "
                            "different RNG per modality group and voids the KS-MOKA-0 identity "
                            "control. Refusing to install.".format(name, type(drop).__name__)
                        )
            module.__class__ = MokaLinear
            module.moka_init(index=n * 8)
            n += 1
    if n == 0:
        raise RuntimeError("MokA install: found ZERO peft lora.Linear layers to route.")

    # -- modality-mask pre-hook: registered on the OUTERMOST module the caller actually invokes ---
    # `nn.Module` forward-PRE hooks fire inside `Module.__call__` ONLY.  Registering on
    # `peft_model.get_base_model()` does NOT work on the production wrapper: with
    # `task_type=CAUSAL_LM` (both deployed paths) `PeftModelForCausalLM.forward` calls
    # `self.base_model(...)` = `LoraModel.__call__` -> `BaseTuner.forward`
    # (peft/tuners/tuners_utils.py:196-197) -> `self.model.forward(*args, **kwargs)` — a DIRECT
    # `.forward()` call that bypasses the base model's `__call__` and therefore every hook on it
    # (measured pre-fix: hook_calls == 0, fallback_calls == 1, MOKA_STRICT raise on batch 1).
    # BOTH deployed call sites invoke THIS wrapper through `__call__`, so one registration covers
    # both:  job 1 = transformers `Trainer.compute_loss` `outputs = model(**inputs)`
    # (trainer.py:3759) and the eval loop (trainer.py:4525 / trainer_seq2seq.py:352;
    # `predict_with_generate` is OFF in the frozen yaml, so no `.generate()` surface exists);
    # job 2 = `src/utils/generate_VideoMLLM_embedding_lora_HF.py:360` `model(**inputs, ...)`.
    # The idempotence check reads `__dict__` rather than `getattr`, because PeftModel/LoraModel
    # forward missing attributes down to the wrapped model (peft_model.py:821-828,
    # tuners_utils.py:368-375) and would otherwise report a hook that is not on THIS module.
    if "_moka_hook" not in peft_model.__dict__:
        peft_model._moka_hook = peft_model.register_forward_pre_hook(_mask_pre_hook, with_kwargs=True)
    return n


def load_moka_a_v(peft_model, lora_dir: str, adapter_name: str = "default") -> int:
    """Explicitly load the `lora_A_v` tensors that `PeftModel.from_pretrained` silently dropped.

    `set_peft_model_state_dict` loads with `strict=False`, so when the MokaLinear class is not
    installed at load time the `lora_A_v` keys are discarded without warning (recon §3.5 item 4).
    The extraction path therefore does: `PeftModel.from_pretrained` -> `install_moka` -> this.
    Raises unless EVERY routed layer received its tensor and every checkpoint tensor was consumed.
    """
    from safetensors.torch import load_file

    path = os.path.join(lora_dir, "adapter_model.safetensors")
    if not os.path.isfile(path):
        raise FileNotFoundError("MokA: no adapter_model.safetensors in {}".format(lora_dir))
    sd = load_file(path)
    want = {k: v for k, v in sd.items() if ".lora_A_v." in k}
    if not want:
        raise RuntimeError(
            "MokA: adapter at {} contains NO lora_A_v tensors — it is a generic LoRA adapter, not a "
            "MokA adapter. Use --no_merge (plain unmerged) instead of --moka.".format(lora_dir)
        )

    named = dict(peft_model.named_modules())
    targets = {n for n, m in named.items() if isinstance(m, MokaLinear)}
    if not targets:
        raise RuntimeError("MokA: load_moka_a_v called before install_moka (no MokaLinear layers).")

    seen, n = set(), 0
    for key, val in want.items():
        mod_path = key[: key.index(".lora_A_v.")]
        if mod_path not in named:
            raise KeyError("MokA: checkpoint key '{}' has no module '{}'".format(key, mod_path))
        dst = named[mod_path].lora_A_v[adapter_name].weight
        if tuple(dst.shape) != tuple(val.shape):
            raise ValueError(
                "MokA: shape mismatch for {}: ckpt {} vs model {}".format(key, tuple(val.shape), tuple(dst.shape))
            )
        dst.data.copy_(val.to(device=dst.device, dtype=dst.dtype))
        seen.add(mod_path)
        n += 1
    missing = targets - seen
    if missing:
        raise RuntimeError(
            "MokA: {} routed layers received NO lora_A_v tensor (e.g. {}).".format(
                len(missing), sorted(missing)[:3]
            )
        )
    return n


# ------------------------------------------------------------------------------------------------
# $0 readouts
# ------------------------------------------------------------------------------------------------
def moka_param_report(peft_model) -> dict:
    """Trainable-parameter accounting (KS-MOKA-0 grad-flow / the 1.4489x disclosure)."""
    tot = a_t = a_v = b = 0
    for name, p in peft_model.named_parameters():
        if not p.requires_grad:
            continue
        tot += p.numel()
        if ".lora_A_v." in name:
            a_v += p.numel()
        elif ".lora_A." in name:
            a_t += p.numel()
        elif ".lora_B." in name:
            b += p.numel()
    return {"trainable_total": tot, "lora_A_t": a_t, "lora_A_v": a_v, "lora_B": b}


def moka_routing_report(peft_model, adapter_name: str = "default") -> list:
    """KS-MOKA-2 ("routing is real", $0): per-layer ||A_v - A_t||_F / ||A_t||_F."""
    rows = []
    for name, module in peft_model.named_modules():
        if not isinstance(module, MokaLinear) or adapter_name not in module.lora_A:
            continue
        w_t = module.lora_A[adapter_name].weight.detach().float()
        w_v = module.lora_A_v[adapter_name].weight.detach().float()
        den = torch.linalg.norm(w_t).item()
        rel = (torch.linalg.norm(w_v - w_t).item() / den) if den > 0 else float("nan")
        rows.append({"module": name, "rel_fro_diff": rel})
    return rows
