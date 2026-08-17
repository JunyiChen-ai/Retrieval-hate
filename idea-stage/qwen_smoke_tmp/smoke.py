#!/usr/bin/env python
"""Qwen2.5-VL-7B trainability smoke test on RTX 5090 (sm_120).

Replicates the historical LoRA-SFT config documented in
refine-logs/LORA_HATEMM_PREREG.md + refine-logs/FRAME_BUDGET_FORENSIC_RECON.md:
  8-frame multi-image ShareGPT, image_max_pixels=262144, cutoff_len=4096,
  lora_rank=16 alpha=32, targets q,k,v,o,gate,up,down, freeze_vision_tower=true,
  per_device_train_batch_size=1, gradient_checkpointing=true, bf16.

Measures a REAL forward + backward + optimizer step, reporting
torch.cuda.max_memory_allocated and seconds/step.
"""
import argparse, json, os, sys, time, glob, traceback

import torch

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
FRAMES_ROOT = "/home/jehc223/Retrieval-hate/data/lora_frames/HateMM"

INSTR = (
    "Analyze this video using the {n} frames and the transcript below. Decide whether "
    "the video is hateful. Answer with a single word: Yes or No.\n\nTranscript: {t}"
)


def build_batch(processor, n_frames, max_pixels, cutoff_len, batch_size, synthetic=False):
    from PIL import Image
    vids = sorted(os.listdir(FRAMES_ROOT))[:64] if os.path.isdir(FRAMES_ROOT) else []
    imgs_per_sample = []
    src = "synthetic"
    if vids and not synthetic:
        # cached frames are 8 per video; for 16-frame arm reuse two videos
        need_vids = (n_frames + 7) // 8
        for b in range(batch_size):
            imgs = []
            for k in range(need_vids):
                d = os.path.join(FRAMES_ROOT, vids[(b * need_vids + k) % len(vids)])
                for p in sorted(glob.glob(os.path.join(d, "frame_*.jpg")))[:8]:
                    imgs.append(Image.open(p).convert("RGB"))
            imgs_per_sample.append(imgs[:n_frames])
        src = "real:data/lora_frames/HateMM"
    else:
        for b in range(batch_size):
            imgs_per_sample.append(
                [Image.fromarray(
                    (torch.rand(480, 854, 3) * 255).byte().numpy()) for _ in range(n_frames)]
            )

    transcript = ("hello everyone welcome back to the channel today we are going to "
                  "talk about something very important " * 20)[:1500]
    texts, all_imgs = [], []
    for imgs in imgs_per_sample:
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": INSTR.format(n=n_frames, t=transcript)})
        msgs = [{"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": "Yes"}]}]
        texts.append(processor.apply_chat_template(msgs, tokenize=False))
        all_imgs.append(imgs)

    inputs = processor(
        text=texts, images=all_imgs, return_tensors="pt",
        padding=True, truncation=True, max_length=cutoff_len,
    )
    labels = inputs["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    labels[labels == pad_id] = -100
    inputs["labels"] = labels
    return inputs, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["qlora", "bf16"], required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--cutoff-len", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--lean", action="store_true",
                    help="qlora: skip prepare_model_for_kbit_training's fp32 upcast")
    ap.add_argument("--keep-layers", type=int, default=0,
                    help="truncate LM to first N decoder layers (memory-constrained probe); "
                         "0 = full 28 layers")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = {"mode": a.mode, "frames": a.frames, "batch_size": a.batch_size,
           "max_pixels": a.max_pixels, "cutoff_len": a.cutoff_len,
           "attn": a.attn, "grad_ckpt": not a.no_grad_ckpt}

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.cuda.reset_peak_memory_stats()
    free0, tot0 = torch.cuda.mem_get_info()
    res["gpu_free_at_start_gib"] = round(free0 / 2**30, 2)
    res["gpu_total_gib"] = round(tot0 / 2**30, 2)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=a.max_pixels)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    kw = dict(torch_dtype=torch.bfloat16, attn_implementation=a.attn, device_map={"": 0})
    if a.mode == "qlora":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    t0 = time.time()
    if a.keep_layers:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL_ID)
        tcfg = getattr(cfg, "text_config", cfg)
        tcfg.num_hidden_layers = a.keep_layers
        kw["config"] = cfg
        res["keep_layers"] = a.keep_layers
        res["NOTE"] = ("layer-truncated probe: %d of 28 LM decoder layers; "
                       "per-layer cost extrapolated to 28" % a.keep_layers)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **kw)
    res["load_seconds"] = round(time.time() - t0, 1)
    res["n_lm_layers_live"] = len(model.model.language_model.layers) if hasattr(
        model.model, "language_model") else -1
    res["weights_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    if a.mode == "qlora" and not a.lean:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=not a.no_grad_ckpt)
    elif a.mode == "qlora" and a.lean:
        # lean: skip the fp32 upcast of embed_tokens/lm_head/norms that
        # prepare_model_for_kbit_training does (~+2.2 GB); freeze base weights by hand.
        for p in model.parameters():
            p.requires_grad_(False)
        res["lean_no_fp32_upcast"] = True

    # freeze_vision_tower: true  -> LoRA only on the language model
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=r".*(language_model|model\.layers)\..*"
                       r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
    model = get_peft_model(model, lcfg)
    for n, p in model.named_parameters():
        if "visual" in n:
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    res["trainable_params_M"] = round(n_train / 1e6, 2)
    assert n_train > 0, "no trainable params -- target_modules regex matched nothing"

    if not a.no_grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()
    model.config.use_cache = False

    inputs, src = build_batch(processor, a.frames, a.max_pixels, a.cutoff_len,
                              a.batch_size, a.synthetic)
    res["input_source"] = src
    res["seq_len"] = int(inputs["input_ids"].shape[1])
    if "pixel_values" in inputs:
        res["vision_patches"] = int(inputs["pixel_values"].shape[0])
    inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}

    import bitsandbytes as bnb
    params = [p for p in model.parameters() if p.requires_grad]
    opt = (bnb.optim.PagedAdamW8bit(params, lr=1e-4) if a.mode == "qlora"
           else torch.optim.AdamW(params, lr=1e-4))

    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(a.steps):
        torch.cuda.synchronize(); t = time.time()
        out = model(**inputs)
        out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t
        times.append(dt)
        print(f"[step {i}] loss={out.loss.item():.4f} {dt:.2f}s "
              f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)

    res["loss"] = round(float(out.loss.item()), 4)
    res["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    res["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    res["sec_per_step_first"] = round(times[0], 3)
    res["sec_per_step_steady"] = round(sum(times[1:]) / max(1, len(times) - 1), 3)
    res["oom"] = False
    print("RESULT " + json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as e:
        print("RESULT " + json.dumps({"oom": True, "err": str(e)[:300],
              "argv": sys.argv[1:]}), flush=True)
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
