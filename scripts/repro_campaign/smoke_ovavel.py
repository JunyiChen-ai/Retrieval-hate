#!/usr/bin/env python
"""Smoke test: OV-AVEL training-free baseline (v0) on one of our videos.

The upstream entry point `proposed_method/ImageBind-main/baseline_v0_training_free.py`
is bound to the authors' OV-AVE dataset layout (a meta CSV, an annotation JSON and a
preprocessed 10-frame/10-second directory tree, all under /root/autodl-tmp). This driver
keeps the *method* — ImageBind audio/vision/text embeddings, per-second cosine similarity
to a category list, and the audio-argmax == vision-argmax agreement rule — and replaces
only the dataset plumbing, so the model can be pointed at one of our videos.

`compute_cross_modal_similarity` and `postprocess_simm` are copied verbatim from
baseline_v0_training_free.py; the only change is `.cuda()` -> `.to(device)` so the smoke
can run on CPU while the GPU is held by another job.
"""
import argparse
import os
import os.path as osp
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

OVAVEL_ROOT = "/home/jehc223/Retrieval-hate/third_party/OV-AVEL/proposed_method/ImageBind-main"


def compute_cross_modal_similarity(tensor_a, tensor_t):
    """verbatim from baseline_v0_training_free.py"""
    B, T, D = tensor_a.shape
    _, C, _ = tensor_t.shape
    tensor_a_expanded = tensor_a.unsqueeze(2).expand(B, T, C, D)
    tensor_t_expanded = tensor_t.unsqueeze(1).expand(B, T, C, D)
    cos_sim = F.cosine_similarity(tensor_a_expanded, tensor_t_expanded, dim=-1)
    return cos_sim


def postprocess_simm(simm_at, simm_vt, bg_class_id, device):
    """verbatim from baseline_v0_training_free.py, .cuda() -> .to(device)"""
    max_prob_at_idx = simm_at.max(dim=-1)[1]
    max_prob_vt_idx = simm_vt.max(dim=-1)[1]
    is_event_flag = (max_prob_at_idx == max_prob_vt_idx).float()
    B = is_event_flag.shape[0]
    C = simm_at.shape[-1]
    event_flag = torch.zeros([B, C + 1]).to(device)
    for i in range(B):
        if torch.all(is_event_flag[i] == 0):
            event_flag[i][bg_class_id] = 1
        else:
            nonzero_pos = torch.nonzero(is_event_flag[i], as_tuple=False).squeeze()
            if len(nonzero_pos.size()) == 0:
                category_id = max_prob_at_idx[i][nonzero_pos.item()]
            else:
                category_id = max_prob_at_idx[i][nonzero_pos[0].item()]
            event_flag[i][category_id] = 1
    return is_event_flag, event_flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/home/jehc223/data/HateClipSeg/videos/bit_0dcMcI6hYjhw.mp4")
    ap.add_argument("--wav", default="/home/jehc223/Retrieval-hate/data/AV2A_wav/HateClipSeg/bit_0dcMcI6hYjhw.wav")
    ap.add_argument("--seconds", type=int, default=10, help="OV-AVEL's fixed clip length")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.chdir(OVAVEL_ROOT)  # BPE_PATH in imagebind/data.py is relative
    sys.path.insert(0, OVAVEL_ROOT)
    import imagebind  # noqa: E402
    from imagebind import data as ib_data  # noqa: E402
    from imagebind.models import imagebind_model  # noqa: E402
    from imagebind.models.imagebind_model import ModalityType  # noqa: E402

    device = args.device
    tmp = tempfile.mkdtemp(prefix="ovavel_smoke_")

    # 1 frame per second, exactly what their preprocessed video/<vid>/ directory holds
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", args.video, "-t", str(args.seconds),
         "-vf", "fps=1", "-vsync", "0", osp.join(tmp, "%03d.jpg")],
        check=True,
    )
    frames = sorted(osp.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".jpg"))[: args.seconds]
    print(f"[smoke] {len(frames)} frames at 1 fps")

    wav = args.wav
    if not osp.exists(wav):
        wav = osp.join(tmp, "a.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args.video, "-ac", "2",
                        "-ar", "16000", wav], check=True)
    else:
        # their loader hardcodes a stereo zero-pad buffer; force 2ch/16k
        wav2 = osp.join(tmp, "a.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-ac", "2",
                        "-ar", "16000", wav2], check=True)
        wav = wav2

    # OV-AVEL's category list is the 67 AVE/VGGSound-style event names; for our corpus the
    # analogous open-vocabulary list is the campaign's hate categories plus a background name.
    text_list = [
        "hateful speech targeting a group of people",
        "racist content",
        "a violent attack",
        "a person talking to camera",
        "music",
        "speech",
    ]
    text_inputs = ib_data.load_and_transform_text(text_list, device)
    audio_inputs = ib_data.load_and_transform_audio_data([wav], device)  # [1, 10, 1, 128, 204]
    visual_inputs = ib_data.load_and_transform_vision_data(frames, device)  # [10, 3, 224, 224]
    print(f"[smoke] audio {tuple(audio_inputs.shape)} visual {tuple(visual_inputs.shape)} "
          f"text {tuple(text_inputs.shape)}")

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval().to(device)

    inputs = {
        ModalityType.TEXT: text_inputs.to(device),
        ModalityType.VISION: visual_inputs.unsqueeze(0).to(device),
        ModalityType.AUDIO: audio_inputs.squeeze(1).unsqueeze(0).to(device) if audio_inputs.dim() == 6
        else audio_inputs.to(device),
    }
    for k, v in inputs.items():
        print(f"[smoke] input {k}: {tuple(v.shape)}")
    with torch.no_grad():
        emb = model(inputs)
    a, v, t = emb["audio"], emb["vision"], emb["text"]
    print(f"[smoke] emb audio {tuple(a.shape)} vision {tuple(v.shape)} text {tuple(t.shape)}")

    bs = v.shape[0]
    t_rep = t.unsqueeze(0).repeat(bs, 1, 1)
    simm_at = compute_cross_modal_similarity(a, t_rep)
    simm_vt = compute_cross_modal_similarity(v, t_rep)
    is_event, event_flag = postprocess_simm(simm_at, simm_vt, len(text_list), device)
    print(f"[smoke] simm_at {tuple(simm_at.shape)} range "
          f"{simm_at.min().item():.4f}..{simm_at.max().item():.4f}")
    print(f"[smoke] simm_vt {tuple(simm_vt.shape)} range "
          f"{simm_vt.min().item():.4f}..{simm_vt.max().item():.4f}")
    print(f"[smoke] is_event_flag (per second) = {is_event[0].tolist()}")
    print(f"[smoke] event_flag argmax = {int(event_flag[0].argmax().item())} "
          f"(len(text_list)={len(text_list)} means background)")

    if args.out:
        np.savez(args.out,
                 is_event=is_event.cpu().numpy(),
                 simm_at=simm_at.cpu().numpy(),
                 simm_vt=simm_vt.cpu().numpy(),
                 rate=1.0)
        print(f"[smoke] wrote {args.out}")


if __name__ == "__main__":
    main()
