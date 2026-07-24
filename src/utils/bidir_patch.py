"""Bidirectional-attention surgery patch for the Qwen2.5-VL language decoder.

BIDIR_STAGE1_PREREG artifact A. CPU-importable; no weights changed at inference.

WHAT THIS DOES
--------------
The Qwen2.5-VL decoder (`Qwen2_5_VLModel`, i.e. `model.model` of a
`Qwen2_5_VLForConditionalGeneration`) hard-codes causal self-attention: every
`Qwen2_5_VLAttention.__init__` sets `self.is_causal = True`
(transformers 4.49.0 modeling_qwen2_5_vl.py:723) and there is NO config flag to
disable it. This module installs a monkey-patch that forces the decoder to attend
BIDIRECTIONALLY when harvesting embeddings (LLM2Vec / NV-Embed recipe), so a token
vector is computed with full past+future context rather than a cumulative
causal-prefix summary (the F35 pathology this cell attacks).

THE SDPA RE-CAUSALIZATION TRAP (the load-bearing correctness point)
-------------------------------------------------------------------
Our extraction path loads with `attn_implementation="sdpa"`. For SDPA, causality
comes from TWO places and you must defeat BOTH:
  (a) the 4D additive mask returned by `Qwen2_5_VLModel._update_causal_mask`
      (modeling_qwen2_5_vl.py:1244-1325), consumed as SDPA `attn_mask` (:995);
  (b) the fallback `is_causal = True if causal_mask is None and q_len > 1 else False`
      inside `Qwen2_5_VLSdpaAttention.forward` (:989).
For a single unpadded sample `_update_causal_mask` returns **None**
(via `_ignore_causal_mask_sdpa`, :1278), whereupon (b) sets `is_causal=True` and
SDPA applies causal masking INTERNALLY. So merely NULLING the mask leaves attention
silently causal. The fix is to force a NON-None all-zeros 4D additive mask, which
(i) makes :989 evaluate `is_causal=False` and (ii) adds a zero bias = attend
everywhere = bidirectional. `_bidir_update_causal_mask` returns exactly that.

FLASH CAVEAT: `Qwen2_5_VLFlashAttention2.forward` passes `is_causal=self.is_causal`
(:904) and uses the 4D mask only for padding, so the mask patch alone does NOT flip
flash. Our path is SDPA (asserted below); as belt-and-suspenders `apply_bidir_mask`
also sets `is_causal=False` on every decoder attention module (harmless for SDPA,
which recomputes is_causal locally, and correct for flash if it were ever used).

VISION UNTOUCHED: the patch binds ONLY to `model.model` (LLM decoder). The vision
tower (`model.visual`) already uses block-diagonal (cu_seqlens) full-within-window
attention (:265-269) and is not modified.
"""

import types

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention


def _bidir_update_causal_mask(
    self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions
):
    """Drop-in replacement for `Qwen2_5_VLModel._update_causal_mask`.

    Returns a NON-None all-zeros 4D additive mask (shape [bsz, 1, seq, seq]); folds
    2D padding masks (bsz=1 unpadded extraction => no-op). Because it always returns
    a non-None mask, `Qwen2_5_VLSdpaAttention.forward` sets `is_causal=False` and the
    zero additive bias imposes NO masking => fully bidirectional attention. It
    completely replaces `_update_causal_mask`, so `_ignore_causal_mask_sdpa` /
    `_unmask_unattended` never run (no interaction hazard).
    """
    dtype = input_tensor.dtype
    bsz, seq_len = input_tensor.shape[0], input_tensor.shape[1]
    mask = torch.zeros((bsz, 1, seq_len, seq_len), dtype=dtype, device=input_tensor.device)
    if attention_mask is not None and attention_mask.dim() == 2:  # fold padding (bsz=1 -> no-op)
        pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
        mask = mask + pad
    return mask


def apply_bidir_mask(model, *, assert_sdpa=True):
    """Install the bidirectional-attention patch on a Qwen2.5-VL model IN PLACE.

    `model` is a `Qwen2_5_VLForConditionalGeneration` (post `merge_and_unload` for
    the LoRA path); `model.model` is the `Qwen2_5_VLModel` decoder. Binds
    `_bidir_update_causal_mask` to the decoder instance and (defensively) clears
    `is_causal` on every decoder attention module. Asserts SDPA to prevent a silent
    flash fallback re-causalizing (the mask patch does not cover flash).
    Returns `model`.
    """
    decoder = model.model  # Qwen2_5_VLModel
    if assert_sdpa:
        impl = getattr(decoder.config, "_attn_implementation", None)
        assert impl == "sdpa", (
            "bidir patch requires attn_implementation='sdpa' (got {!r}); flash would "
            "silently re-causalize via is_causal=self.is_causal (modeling:904).".format(impl)
        )
    decoder._update_causal_mask = types.MethodType(_bidir_update_causal_mask, decoder)
    n = 0
    for m in decoder.modules():
        if isinstance(m, Qwen2_5_VLAttention):
            m.is_causal = False
            n += 1
    print(
        "[BIDIR] mask-flip patch installed on model.model; is_causal=False on {} decoder "
        "attention module(s); attention is now bidirectional.".format(n),
        flush=True,
    )
    return model


def bidir_self_test(seed=0, seq_len=6, eps_causal=1e-5, eps_bidir=1e-4, verbose=True):
    """CPU-only non-causality self-test on a tiny random Qwen2.5-VL decoder.

    Honest discriminator: perturb a FUTURE token (last position) and measure the
    change at an EARLY token (position 0). Under CAUSAL attention position 0 cannot
    see the future, so the change is ~0; under the BIDIR patch position 0 attends to
    the future, so the change is > 0. Also asserts the patched mask is a NON-None
    all-zeros tensor of shape [bsz, 1, seq, seq]. No trained weights needed: causal
    vs bidirectional is a structural property independent of the parameters.

    Returns a dict of measurements; raises AssertionError on failure.
    """
    from transformers import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    torch.manual_seed(seed)
    cfg = Qwen2_5_VLConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
        rope_scaling={"type": "mrope", "mrope_section": [4, 2, 2]},  # sums to head_dim/2 = 8
        _attn_implementation="sdpa",
    )
    m = Qwen2_5_VLModel(cfg).eval()
    H = cfg.hidden_size
    emb = torch.randn(1, seq_len, H)
    emb2 = emb.clone()
    emb2[0, seq_len - 1, :] = torch.randn(H)  # perturb the LAST (future) token

    @torch.no_grad()
    def run():
        h1 = m(inputs_embeds=emb, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
        h2 = m(inputs_embeds=emb2, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
        return h1, h2

    # 1) causal control (no patch)
    h1, h2 = run()
    d_causal_pos0 = (h1[0] - h2[0]).norm().item()          # early token, future perturbed
    d_causal_last = (h1[seq_len - 1] - h2[seq_len - 1]).norm().item()  # last token (sanity)

    # 2) bidir (patch the identical function apply_bidir_mask binds, onto this decoder)
    m._update_causal_mask = types.MethodType(_bidir_update_causal_mask, m)
    mk = m._update_causal_mask(None, emb, None, None, False)
    h1b, h2b = run()
    d_bidir_pos0 = (h1b[0] - h2b[0]).norm().item()

    ok_maskshape = tuple(mk.shape) == (1, 1, seq_len, seq_len)
    ok_maskzero = (mk.abs().max().item() == 0.0)
    ok_causal = d_causal_pos0 < eps_causal          # control genuinely causal
    ok_sanity = d_causal_last > eps_bidir           # perturbation is real
    ok_bidir = d_bidir_pos0 > eps_bidir             # patch makes it non-causal
    passed = ok_maskshape and ok_maskzero and ok_causal and ok_sanity and ok_bidir

    res = dict(
        d_causal_pos0=d_causal_pos0, d_causal_last=d_causal_last, d_bidir_pos0=d_bidir_pos0,
        mask_shape=tuple(mk.shape), mask_is_zero=ok_maskzero, passed=passed,
    )
    if verbose:
        print("[BIDIR self-test] patched mask shape={} all-zero={}".format(res["mask_shape"], ok_maskzero))
        print("[BIDIR self-test] d_causal(pos0, future perturbed) = {:.3e}  (expect < {:.0e})".format(d_causal_pos0, eps_causal))
        print("[BIDIR self-test] d_causal(last pos, sanity)        = {:.3e}  (expect > {:.0e})".format(d_causal_last, eps_bidir))
        print("[BIDIR self-test] d_bidir (pos0, future perturbed)  = {:.3e}  (expect > {:.0e})".format(d_bidir_pos0, eps_bidir))
        print("[BIDIR self-test] VERDICT: {}".format("PASS" if passed else "FAIL"))
    assert passed, "bidir self-test FAILED: {}".format(res)
    return res


if __name__ == "__main__":
    bidir_self_test()
