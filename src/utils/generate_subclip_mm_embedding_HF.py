import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.generate_VideoCLIP_embedding_HF import encode_text  # noqa: E402
from transformers import CLIPTokenizer, CLIPTextModel  # noqa: E402

# Multimodal (frame + per-window ASR text) sub-clip cache for the consensus
# E-step voting space (EXP_mm_segment_keys phase 2).
#
# W2 attribution: MHC-EN hate is predominantly speech/on-screen-text-carried;
# the purely-visual sub-clip keys vote noise. This script adds a per-window
# CLIP *text* embedding of the window's own Whisper transcript next to the
# existing per-window CLIP visual embedding.
#
#   input  1: existing sub-clip visual cache (NOT recomputed, bit-identical):
#             <EXP_FOLDER>/<DS>/{split}_subclipK<K>_<model>_HF.pt
#   input  2: segment-aligned ASR (generate_segment_asr_HF.py):
#             <asr_dir>/<DS>/{split}_asrK<K>_<asr_model>.jsonl
#   output : NEW cache file (never overwrites the visual cache):
#             <EXP_FOLDER>/<DS>/{split}_subclipK<K>_mm_<model>_HF.pt
#             = all keys of input 1, plus
#               "subclip_txt_feats"    [TotalSub, Dt] float32
#                   CLIP text pooler embedding of the window transcript,
#                   chunk-mean-pooled exactly like the whole-video text
#                   stream (encode_text); ZERO vector when the window has
#                   no transcript (or the video is missing from the ASR).
#               "subclip_txt_has_text" [TotalSub] bool
#               "asr_source"           str (the ASR jsonl consumed)
#
# Text encoding uses the SAME CLIP checkpoint as the visual stream, so the
# mm key is a two-channel (image | text) concat in the same model family as
# the whole-video memory keys. Window transcripts are short, so the 77-token
# CLIP limit that crippled video-level transcripts is no longer binding
# (encode_text still chunk-mean-pools the rare long window).

SPLIT_TO_OUTNAME = {
    "train": "train",
    "val": "dev_seen",
    "test": "test_seen",
}


def parse_args_sys(args_list=None):
    ap = argparse.ArgumentParser(
        description="Add per-window ASR CLIP-text embeddings to the sub-clip cache.")
    ap.add_argument("--dataset", type=str, default="MHC")
    ap.add_argument("--EXP_FOLDER", type=str, default="./data/CLIP_Embedding")
    ap.add_argument("--asr_dir", type=str, default="./data/ASR")
    ap.add_argument("--asr_model_tag", type=str, default="whisper-large-v3",
                    help="Tag inside the ASR jsonl filename.")
    ap.add_argument("--model", type=str,
                    default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--num_subclips", type=int, default=4)
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args(args_list)


def load_asr(path):
    recs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            recs[str(d["id"])] = d
    return recs


def main(args):
    device = torch.device(args.device)
    print("Loading CLIP text model: {}".format(args.model))
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    text_model = CLIPTextModel.from_pretrained(args.model)
    text_model.to(device).eval()
    dt = text_model.config.hidden_size

    model_tag = str(args.model).replace("/", "_")
    K = args.num_subclips

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in SPLIT_TO_OUTNAME:
            print("[WARN] split '{}' unmapped; skipping.".format(split))
            continue
        outname = SPLIT_TO_OUTNAME[split]
        vis_path = os.path.join(
            args.EXP_FOLDER, args.dataset,
            "{}_subclipK{}_{}_HF.pt".format(outname, K, model_tag))
        asr_path = os.path.join(
            args.asr_dir, args.dataset,
            "{}_asrK{}_{}.jsonl".format(outname, K, args.asr_model_tag))
        if not os.path.exists(vis_path):
            print("[WARN] missing visual sub-clip cache {}; skipping.".format(
                vis_path))
            continue
        if not os.path.exists(asr_path):
            print("[WARN] missing ASR jsonl {}; skipping.".format(asr_path))
            continue

        cache = torch.load(vis_path, map_location="cpu")
        video_ids = cache["video_ids"]
        S = cache["subclip_img_feats"].shape[0]
        assert S == len(video_ids) * K, \
            "sub-clip cache is not V*K -- unexpected layout"
        asr = load_asr(asr_path)

        txt_feats = torch.zeros(S, dt, dtype=torch.float32)
        has_text = torch.zeros(S, dtype=torch.bool)
        n_missing, n_windows_with_text = 0, 0

        with torch.no_grad():
            for v, vid in enumerate(video_ids):
                rec = asr.get(str(vid))
                if rec is None:
                    n_missing += 1
                    continue
                wt = rec.get("window_text") or []
                if len(wt) != K:
                    print("[WARN] {}: {} windows in ASR (expected {}); "
                          "truncating/padding.".format(vid, len(wt), K))
                    wt = (wt + [""] * K)[:K]
                for k in range(K):
                    t = (wt[k] or "").strip()
                    if not t:
                        continue
                    row = v * K + k
                    txt_feats[row] = encode_text(
                        t, tokenizer, text_model, device)
                    has_text[row] = True
                    n_windows_with_text += 1
                if (v + 1) % 100 == 0:
                    print("  [{}] {}/{} videos".format(
                        outname, v + 1, len(video_ids)))

        out = dict(cache)
        out["subclip_txt_feats"] = txt_feats
        out["subclip_txt_has_text"] = has_text
        out["asr_source"] = asr_path
        out_path = os.path.join(
            args.EXP_FOLDER, args.dataset,
            "{}_subclipK{}_mm_{}_HF.pt".format(outname, K, model_tag))
        torch.save(out, out_path)
        print("Saved '{}': S={}, windows-with-text={} ({:.1f}%), "
              "videos-missing-from-ASR={} -> {}".format(
                  outname, S, n_windows_with_text,
                  100.0 * n_windows_with_text / max(S, 1),
                  n_missing, out_path))


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    main(args)
