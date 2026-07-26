"""S1: bidirectional attention + LLM2Vec MEAN-POOL text readout, NO training.

MNTP staged plan stage S1 (refine-logs/MNTP_FORENSIC_RECON.md §6). This is a THIN
fork of generate_VideoMLLM_embedding_bidir_HF.py (itself a thin fork of the causal
extractor generate_VideoMLLM_embedding_lora_HF.py). It reuses read_gt / process_split
/ SPLIT_TO_OUTNAME / parse_args_sys VERBATIM from the causal extractor, so the ONLY
differences from the banked causal LoRA arm are:

  (1) the attention mask is flipped to bidirectional (src/utils/bidir_patch, frozen
      sha 36cedbac365b2b13c945adbe3437efdc61d8be15ecc85878eb9614225abe367b), applied
      POST-merge and BEFORE any forward pass -- identical to the F72 artifact-A2 runner;
  (2) the TEXT readout span.

THE ONE MANIPULATED READOUT (the whole point of S1)
---------------------------------------------------
Deployed text readout (causal extractor `_encode`, span="response", :386-400):
    start  = index of the LAST <|im_start|>  (the "<|im_start|>assistant\\n" header)
    pooled = last_hidden[start:].mean(0)
  => a mean over the trailing 3-4 assistant-header FORMAT tokens. This is an
     EOS-class / last-token readout: correct under causal attention (only the final
     tokens have seen the whole input), but it is exactly the pooling LLM2Vec's own
     ablation names as WORST under bidirectional attention.

S1 text readout (`_encode_s1_meanpool` below, span="response"):
    pooled = mean of last_hidden over ALL non-padding sequence positions, i.e. the
             span [0, seq_len) = every video-pad token + every instruction/title/
             transcript token + every chat-format token, INCLUDING the trailing
             "<|im_start|>assistant\\n" header.
  => LLM2Vec mean pooling ("works best with mean pooling"; the README default).
     At bsz=1 unpadded extraction the attention_mask is all-ones, so the masked mean
     is arithmetically identical to last_hidden.mean(0); the mask is honoured anyway
     so the operator is correct by construction rather than by assumption.
     Measured composition of that span (recon §1.5): 768 video tokens (constant,
     8 frames -> 4 temporal groups x 192 merged) + median 162.5 text tokens
     => median 930.5 positions, 82.5 % vision.

THE IMG READOUT IS BYTE-UNCHANGED. `_encode_s1_meanpool` DELEGATES span="prefix"
to the causal extractor's own `_encode` object (not a copy of it), so the img stream
runs the frozen function verbatim. Consequence, used as a free belt: this cache's
img_feats must reproduce the banked bidir cache's img_feats (same prompt, same span,
same weights, same topology) up to bf16 GPU nondeterminism.

Everything else is untouched: same adapters, same merge_and_unload, same SDPA, same
8 frames, same max_pixels, same instructions, same L2-norm, same cache contract.
The causal extractor's sha stays byte-unchanged, and so does the F72 bidir runner's.

Usage (S1 extracts train+val ONLY -- test is deliberately NOT touched at this stage):
  python src/utils/generate_VideoMLLM_embedding_bidir_meanpool_HF.py \
      --dataset MHC_zh --lora_dir logging/lora/MHC_zh --splits train,val \
      --out_model_tag Qwen2.5-VL-7B-Instruct-LoRA-bidir-meanpool_HF --device cuda
  python src/utils/generate_VideoMLLM_embedding_bidir_meanpool_HF.py \
      --dataset HateMM --lora_dir logging/lora/HateMM_curric --splits train,val \
      --out_model_tag Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-meanpool_HF --device cuda
"""

import os

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Verbatim machinery from the causal LoRA extractor (sibling module in src/utils/).
import generate_VideoMLLM_embedding_lora_HF as _causal
from generate_VideoMLLM_embedding_lora_HF import (
    SPLIT_TO_OUTNAME,
    parse_args_sys,
    process_split,
    read_gt,
)
from bidir_patch import apply_bidir_mask

# The frozen causal readout operator. span="prefix" (img) is delegated to it verbatim.
_encode_causal = _causal._encode


@torch.no_grad()
def _encode_s1_meanpool(frames, instruction, processor, model, device, max_pixels, span):
    """S1 readout. span="prefix" -> frozen causal operator; span="response" -> mean-pool.

    The span="response" branch is the ONLY manipulated code in the S1 arm. Its forward
    is a line-for-line copy of the causal `_encode` forward (same messages, same chat
    template, same processor call, same output_hidden_states=True / use_cache=False,
    same hidden_states[-1], same length assert, same .float() + L2-norm); only the two
    lines that select the pooling span are replaced by a masked mean over ALL positions.
    """
    if span == "prefix":
        # img_feats: run the frozen causal function object itself -- no copy, no drift.
        return _encode_causal(frames, instruction, processor, model, device, max_pixels, span)

    # ---- span == "response": LLM2Vec mean pooling over the WHOLE sequence ----
    messages = _causal._build_messages(frames, instruction)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=None,
        videos=[frames],
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1][0]  # [seq_len, D]
    input_ids = inputs["input_ids"][0]  # [seq_len]
    assert last_hidden.shape[0] == input_ids.numel(), (
        "hidden/input_ids length mismatch: {} vs {}".format(
            last_hidden.shape[0], input_ids.numel()
        )
    )

    # === THE ONLY MANIPULATED LINES vs the causal/F72-bidir arms ===
    # Deployed:  start = last <|im_start|>;  pooled = last_hidden[start:].mean(0)
    #            (trailing "<|im_start|>assistant\n" header = EOS-class readout)
    # S1:        pooled = mean over ALL non-padding positions, span [0, seq_len)
    #            (LLM2Vec mean pooling). bsz=1 unpadded => mask is all-ones => this
    #            equals last_hidden.mean(0); the mask is applied anyway for correctness.
    am = inputs.get("attention_mask", None)
    if am is None or bool(am[0].all()):
        # bsz=1 unpadded (the extraction regime): every position is real, so the mask
        # is a no-op. Use the SAME reduction call the frozen prefix readout uses
        # (last_hidden[...].mean(dim=0)) so the two streams are numerically parity by
        # construction -- same kernel, same bf16 accumulation, no extra rounding.
        pooled = last_hidden.mean(dim=0)
    else:
        m = am[0].to(last_hidden.dtype).unsqueeze(-1)  # [seq_len, 1]
        denom = m.sum().clamp(min=1)
        pooled = (last_hidden * m).sum(dim=0) / denom
    # ==============================================================

    pooled = pooled.float()
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
    return pooled.detach().cpu()


# Install the S1 readout so the VERBATIM-imported process_split (which resolves
# `_encode` as a module global of the causal extractor) dispatches through it.
_causal._encode = _encode_s1_meanpool


def main(args):
    # --- Guards run FIRST, before any filesystem or GPU side effect. Explicit raises,
    # --- not asserts, so python -O / PYTHONOPTIMIZE cannot disable them.
    if _causal._encode is not _encode_s1_meanpool:
        raise RuntimeError("S1 readout not installed; this arm would silently duplicate F72.")
    # A fork must never inherit the causal extractor's default out-tag and clobber a
    # banked cache. Every S1 cache carries 'meanpool' in its tag.
    if "meanpool" not in args.out_model_tag:
        raise RuntimeError(
            "--out_model_tag must contain 'meanpool' (got {!r}); refusing to write a "
            "cache whose tag does not identify the S1 readout.".format(args.out_model_tag)
        )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    # S1 is DEV-GATED: the held-out test split must not be extracted at this stage.
    if any(s == "test" or SPLIT_TO_OUTNAME.get(s, "").startswith("test") for s in splits):
        raise RuntimeError(
            "S1 is dev-only (ZERO test-touch): --splits must not contain 'test' "
            "(got {!r})".format(args.splits)
        )

    device = torch.device(args.device)

    out_dir = os.path.join(args.EXP_FOLDER, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("[S1] text readout = LLM2Vec mean pool over ALL non-padding positions "
          "[0, seq_len); img readout = frozen causal prefix-mean (delegated).", flush=True)

    print("Loading Qwen2.5-VL base model: {}".format(args.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # REQUIRED: apply_bidir_mask asserts sdpa (flash trap)
        device_map=None,
    )

    lora_dir = args.lora_dir.strip() if args.lora_dir else ""
    if lora_dir:
        if not os.path.isdir(lora_dir):
            raise FileNotFoundError(
                "--lora_dir '{}' is not a directory (expected a peft adapter dir "
                "with adapter_config.json + adapter_model.safetensors).".format(lora_dir)
            )
        from peft import PeftModel

        print("Attaching LoRA adapter from: {}".format(lora_dir), flush=True)
        model = PeftModel.from_pretrained(model, lora_dir)
        print("Merging LoRA adapter into base weights (merge_and_unload) ...", flush=True)
        model = model.merge_and_unload()
    else:
        print("No --lora_dir given; using FROZEN base model (bidir over frozen base).", flush=True)

    # POST-merge, PRE-forward: model is a plain Qwen2_5_VLForConditionalGeneration and
    # model.model is the SAME Qwen2_5_VLModel decoder instance; bind the bidir mask.
    apply_bidir_mask(model)  # asserts sdpa; forces non-None all-zeros mask; clears is_causal

    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)

    for split in splits:
        if split not in SPLIT_TO_OUTNAME:
            print("[WARN] split '{}' has no output-name mapping; skipping.".format(split))
            continue
        outname = SPLIT_TO_OUTNAME[split]
        gt_path = os.path.join(args.gt_dir, args.dataset, "{}.jsonl".format(split))
        if not os.path.exists(gt_path):
            print("[WARN] gt file not found, skipping split '{}': {}".format(split, gt_path))
            continue

        items = read_gt(gt_path)
        print(
            "Processing split '{}' ({} items) -> outname '{}'".format(split, len(items), outname),
            flush=True,
        )

        ids, img_feats, text_feats, labels_t, dv, dt, zero_guard = process_split(
            items, split, args, processor, model, device
        )

        save_obj = {
            "ids": [ids],
            "img_feats": img_feats,
            "text_feats": text_feats,
            "labels": labels_t,
        }
        out_path = os.path.join(out_dir, "{}_{}.pt".format(outname, args.out_model_tag))
        torch.save(save_obj, out_path)

        print(
            "Saved '{}': N={}, Dv={}, Dt={}, zero-vector videos={} -> {}".format(
                outname, len(ids), dv, dt, zero_guard, out_path
            ),
            flush=True,
        )


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    main(args)
