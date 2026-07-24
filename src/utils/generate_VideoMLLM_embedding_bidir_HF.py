"""Bidirectional-attention LoRA-Qwen2.5-VL embedding extractor.

BIDIR_STAGE1_PREREG artifact A2 (the runner that applies artifact A). This is a THIN
fork of generate_VideoMLLM_embedding_lora_HF.py: it REUSES that module's pooling
operator VERBATIM (imports read_gt / process_split / SPLIT_TO_OUTNAME, which in turn
use its _encode / load_video_frames / IMG_INSTRUCTION / TEXT_INSTRUCTION), so the ONLY
difference from the causal LoRA arm is the attention mask. It re-implements just the
~20-line main() so that, right after the LoRA merge and BEFORE any forward pass, it
installs the bidirectional-attention patch (src/utils/bidir_patch.apply_bidir_mask).

The banked causal extractor generate_VideoMLLM_embedding_lora_HF.py is NOT edited
(its sha is unchanged; every other provenance chain that pins it stays valid).

Usage mirrors the LoRA extractor, with LoRA-distinct out-tags carrying "-bidir":
  python src/utils/generate_VideoMLLM_embedding_bidir_HF.py \
      --dataset MHC_zh  --lora_dir logging/lora/MHC_zh \
      --out_model_tag Qwen2.5-VL-7B-Instruct-LoRA-bidir_HF --device cuda
  python src/utils/generate_VideoMLLM_embedding_bidir_HF.py \
      --dataset HateMM  --lora_dir logging/lora/HateMM_curric \
      --out_model_tag Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir_HF --device cuda
"""

import os

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Verbatim machinery from the causal LoRA extractor (sibling module in src/utils/).
from generate_VideoMLLM_embedding_lora_HF import (
    SPLIT_TO_OUTNAME,
    parse_args_sys,
    process_split,
    read_gt,
)
from bidir_patch import apply_bidir_mask


def main(args):
    device = torch.device(args.device)

    out_dir = os.path.join(args.EXP_FOLDER, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

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

    # === THE ONLY STEP THAT DIFFERS FROM THE CAUSAL LoRA EXTRACTOR ===
    # After the merge, model is the base Qwen2_5_VLForConditionalGeneration and
    # model.model is the SAME Qwen2_5_VLModel decoder instance; bind the bidir mask.
    apply_bidir_mask(model)  # asserts sdpa; forces non-None all-zeros mask; clears is_causal
    # =================================================================

    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

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
