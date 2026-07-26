"""S1b: bidirectional attention + TEXT-POSITIONS-ONLY mean-pool readout, NO training.

MNTP staged plan stage S1b (refine-logs/MNTP_S1_RECORD.md §6b, declared before this file
was written). Thin fork of the causal extractor in the F72 artifact-A2 pattern: it imports
read_gt / process_split / SPLIT_TO_OUTNAME / parse_args_sys VERBATIM and re-implements only
main() plus the ONE manipulated readout.

WHY S1b EXISTS
--------------
S1 (…-bidir-meanpool_HF) pooled ALL ~900 sequence positions. ~80 % of them are vision tokens
from the very same 8 frames the img stream pools, so the "text" vector collapsed onto the img
vector (measured cos 0.9273 / 0.9320) and stopped being a text readout at all. LLM2Vec
mean-pools sequences that are PURE TEXT. The faithful multimodal analogue pools TEXT POSITIONS
ONLY -- that is this arm, and it is the clean test of H2.

THE READOUT (selected by TOKEN ID, never by span arithmetic)
------------------------------------------------------------
    keep[i] == attention_mask[i] == 1            (non-padding; all-ones at bsz=1 unpadded)
            AND input_ids[i] != <|video_pad|>    (151656 -- the vision CONTENT positions)
            AND input_ids[i] != <|image_pad|>    (151655 -- defensive; this path passes videos)
    text_feats = L2( mean_{i: keep[i]} last_hidden[i] )

KEPT on purpose: <|vision_start|> / <|vision_end|> (exactly one each -- structural markers, not
content) and <|im_start|> / <|im_end|> chat formatting, plus every system/instruction/title/
transcript wordpiece. Decoding the kept positions reproduces the full prompt with only the video
content elided -- exactly the object LLM2Vec mean-pools.

Selecting by id (not by count) is deliberate: the vision-token COUNT is a function of the frame
geometry handed to the processor (measured 720 at the deployed max_pixels=151200; recon §1.5
reports 768 under its own dummy sizing), so it is NOT a constant across differently-shaped source
videos. An id mask is correct regardless.

THE IMG READOUT IS UNCHANGED -- span="prefix" delegates to the causal extractor's own `_encode`
FUNCTION OBJECT (not a copy), so it must reproduce the banked bidir img cache exactly. That is
retained as a belt.

Bidir patch: src/utils/bidir_patch.apply_bidir_mask, frozen sha
36cedbac365b2b13c945adbe3437efdc61d8be15ecc85878eb9614225abe367b, applied POST merge_and_unload
and BEFORE any forward pass.

Usage (train+dev ONLY -- test is never extracted at this stage; the guard below enforces it):
  python src/utils/generate_VideoMLLM_embedding_bidir_textpool_HF.py \
      --dataset HateMM --lora_dir logging/lora/HateMM_curric --splits train,val \
      --out_model_tag Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-textpool_HF --device cuda
"""

import json
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

# Token ids that mark VISION CONTENT positions. Resolved once, at model/processor setup.
_VISION_IDS = {"video_pad": None, "image_pad": None}

# Measured span decomposition over the REAL run (the record reports these, not dummy-frame
# estimates): one (n_total, n_vision, n_text) row per text-stream forward.
_SPAN_STATS = []


def resolve_vision_ids(processor):
    """Resolve the vision-content token ids from the deployed processor/tokenizer."""
    tok = processor.tokenizer
    vp = tok.convert_tokens_to_ids(processor.video_token)
    ip = tok.convert_tokens_to_ids(getattr(processor, "image_token", "<|image_pad|>"))
    if vp is None or vp < 0:
        raise RuntimeError("could not resolve <|video_pad|> id from the processor")
    _VISION_IDS["video_pad"] = int(vp)
    _VISION_IDS["image_pad"] = int(ip) if (ip is not None and ip >= 0) else None
    print("[S1b] vision-content token ids: video_pad={} image_pad={} (EXCLUDED from the text "
          "readout); vision_start/vision_end and chat-format tokens are KEPT.".format(
              _VISION_IDS["video_pad"], _VISION_IDS["image_pad"]), flush=True)


@torch.no_grad()
def _encode_s1b_textpool(frames, instruction, processor, model, device, max_pixels, span):
    """span="prefix" -> frozen causal operator; span="response" -> text-positions-only mean."""
    if span == "prefix":
        # img_feats: run the frozen causal function object itself -- no copy, no drift.
        return _encode_causal(frames, instruction, processor, model, device, max_pixels, span)

    # ---- span == "response": LLM2Vec mean pooling over TEXT POSITIONS ONLY ----
    # Checked BEFORE the forward: an unset mask must not cost a GPU forward to discover.
    if _VISION_IDS["video_pad"] is None:
        raise RuntimeError(
            "resolve_vision_ids() was never called; the vision mask is unset, so this readout "
            "would silently pool vision tokens and duplicate S1."
        )
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

    # === THE ONLY MANIPULATED LINES vs the causal / F72-bidir / S1 arms ===
    keep = torch.ones_like(input_ids, dtype=torch.bool)
    am = inputs.get("attention_mask", None)
    if am is not None:
        keep &= am[0].to(torch.bool)
    keep &= input_ids != _VISION_IDS["video_pad"]
    if _VISION_IDS["image_pad"] is not None:
        keep &= input_ids != _VISION_IDS["image_pad"]
    n_keep = int(keep.sum())
    if n_keep == 0:  # cannot happen (the prompt is always present); fail loudly if it does
        raise RuntimeError("text-position mask selected 0 positions")
    _SPAN_STATS.append((int(input_ids.numel()), int(input_ids.numel()) - n_keep, n_keep))
    pooled = last_hidden[keep].mean(dim=0)
    # =====================================================================

    pooled = pooled.float()
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
    return pooled.detach().cpu()


# Install the S1b readout so the VERBATIM-imported process_split (which resolves `_encode`
# as a module global of the causal extractor) dispatches through it.
_causal._encode = _encode_s1b_textpool


def _dump_span_stats(tag, out_dir):
    if not _SPAN_STATS:
        return
    import statistics as st
    tot = [r[0] for r in _SPAN_STATS]
    vis = [r[1] for r in _SPAN_STATS]
    txt = [r[2] for r in _SPAN_STATS]
    blob = {
        "n_items": len(_SPAN_STATS),
        "seq_len": {"median": st.median(tot), "min": min(tot), "max": max(tot)},
        "vision_positions": {"median": st.median(vis), "min": min(vis), "max": max(vis),
                             "constant": len(set(vis)) == 1},
        "text_positions": {"median": st.median(txt), "min": min(txt), "max": max(txt)},
        "vision_share_median": (st.median(vis) / st.median(tot)) if st.median(tot) else None,
    }
    print("[S1b] MEASURED span decomposition over the real run ({}): {}".format(tag, blob),
          flush=True)
    p = os.path.join(out_dir, "SPANSTATS_{}.json".format(tag))
    with open(p, "w") as fh:
        json.dump(blob, fh, indent=2)
    print("[S1b] wrote {}".format(p), flush=True)


def main(args):
    # --- Guards run FIRST, before any filesystem or GPU side effect. Explicit raises, not
    # --- asserts, so python -O / PYTHONOPTIMIZE cannot disable them.
    if _causal._encode is not _encode_s1b_textpool:
        raise RuntimeError("S1b readout not installed; this arm would silently duplicate F72.")
    if "textpool" not in args.out_model_tag:
        raise RuntimeError(
            "--out_model_tag must contain 'textpool' (got {!r}); refusing to write a cache "
            "whose tag does not identify the S1b readout.".format(args.out_model_tag)
        )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if any(s == "test" or SPLIT_TO_OUTNAME.get(s, "").startswith("test") for s in splits):
        raise RuntimeError(
            "S1b is dev-only (ZERO test-touch): --splits must not contain 'test' "
            "(got {!r})".format(args.splits)
        )
    # parse_args_sys is inherited verbatim, so it also exposes the MOKA cell's --no_merge
    # and --moka flags. This fork's main() always takes the deployed merge_and_unload path
    # and would SILENTLY IGNORE them -- a third behavioural divergence beyond the two this
    # arm declared. Refuse loudly instead of diverging quietly.
    for flag in ("no_merge", "moka"):
        if getattr(args, flag, False):
            raise RuntimeError(
                "--{} is not supported by the S1b arm: this fork always takes the deployed "
                "merge_and_unload path, and honouring it would add an undeclared third "
                "difference from the causal control.".format(flag.replace("_", "-"))
            )

    device = torch.device(args.device)

    out_dir = os.path.join(args.EXP_FOLDER, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("[S1b] text readout = mean over TEXT positions only (vision-content tokens EXCLUDED "
          "by id); img readout = frozen causal prefix-mean (delegated).", flush=True)

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

    # POST-merge, PRE-forward: bind the bidirectional mask to the decoder.
    apply_bidir_mask(model)  # asserts sdpa; forces non-None all-zeros mask; clears is_causal

    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    resolve_vision_ids(processor)

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

        del _SPAN_STATS[:]
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
        # Filename carries the OUT-TAG, so a later smoke run (tag …-smoke) can never
        # overwrite the full run's measured span decomposition.
        _dump_span_stats("{}_{}_{}".format(args.dataset, outname, args.out_model_tag), out_dir)


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    main(args)
