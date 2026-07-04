#!/usr/bin/env python3
"""
MultiHateLoc reimplementation -- Phase 1 feature extraction.

For each HateMM video, produce per-second (1 fps) aligned tri-modal features:
  - Video : ViT-B/16 CLS token           -> F_v [T, 768]
  - Audio : VGGish 0.96s embeddings       -> F_a [T, 128]  (linearly interp. to T)
  - Text  : sentence-wise Whisper+BERT     -> F_t [T, 768]

T = round(duration_seconds). Everything is stored fp16 in one .npz per video at
  data/multihateloc_feats/HateMM/<video_id>.npz  (keys: fv, fa, ft, T, dur, has_audio)
Whisper segments are dumped to  .../transcripts/<video_id>.json  for inspection.

Paper (Sun et al., MultiHateLoc, WWW'26, arXiv 2512.10408v3), Sec 3.1:
  F_v in R^{T x 768} (ViT-B/16), F_a in R^{T x 128} (VGGish, linearly interpolated
  to length T), F_t in R^{T x 768} (sentence-wise: Whisper transcribe -> split by
  sentence timestamps -> BERT per sentence -> expand to T within each interval).
"""
import argparse, glob, json, os, subprocess, sys, tempfile, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

FFMPEG = "/data/jehc223/miniconda3/envs/ExMRD/bin/ffmpeg"
FFPROBE = "/data/jehc223/miniconda3/envs/ExMRD/bin/ffprobe"
VIDEO_DIR = "/data/jehc223/HateMM/video"
OUT_DIR = "/data/jehc223/RGCL/data/multihateloc_feats/HateMM"
TRANS_DIR = os.path.join(OUT_DIR, "transcripts")


def probe_duration(mp4):
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", mp4], text=True).strip()
    return float(out)


def has_audio_stream(mp4):
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", mp4], text=True).strip()
    return len(out) > 0


# ----------------------------- Video (ViT) ------------------------------------
def load_vit(device):
    from transformers import ViTImageProcessor, ViTModel
    proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()
    return proc, model


def sample_frames_1fps(mp4, T):
    """Return list of PIL images, one per integer second [0..T-1] (first frame seen
    in each second bucket). Missing seconds are filled by repeating the last frame."""
    import av
    from PIL import Image
    frames = [None] * T
    try:
        container = av.open(mp4)
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            sec = int(frame.time)
            if 0 <= sec < T and frames[sec] is None:
                frames[sec] = frame.to_image().convert("RGB")
        container.close()
    except Exception as e:
        print(f"  [warn] av decode failed ({e}); ffmpeg fallback", flush=True)
    # fill gaps: forward-fill, then back-fill; if all None -> gray images
    last = None
    for i in range(T):
        if frames[i] is not None:
            last = frames[i]
        elif last is not None:
            frames[i] = last
    nxt = None
    for i in range(T - 1, -1, -1):
        if frames[i] is not None:
            nxt = frames[i]
        elif nxt is not None:
            frames[i] = nxt
    for i in range(T):
        if frames[i] is None:
            frames[i] = Image.new("RGB", (224, 224), (128, 128, 128))
    return frames


@torch.no_grad()
def encode_video(mp4, T, proc, model, device, bs=64):
    frames = sample_frames_1fps(mp4, T)
    feats = []
    for i in range(0, T, bs):
        batch = frames[i:i + bs]
        inp = proc(images=batch, return_tensors="pt").to(device)
        out = model(**inp).last_hidden_state[:, 0]  # CLS, [b,768]
        feats.append(out.float().cpu())
    return torch.cat(feats, 0).numpy()  # [T,768]


# ----------------------------- Audio (VGGish) ---------------------------------
def load_vggish(device):
    model = torch.hub.load("harritaylor/torchvggish", "vggish",
                           postprocess=False, preprocess=True, trust_repo=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def encode_audio(wav_path, T, model, device):
    """VGGish -> [n,128] (one row / 0.96s) -> linearly interpolate to [T,128]."""
    emb = model.forward(wav_path)  # tensor [n,128] on device (n>=1)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    emb = emb.float().cpu()  # [n,128]
    n = emb.shape[0]
    if n == T:
        return emb.numpy()
    # linear interpolation along time to length T
    x = emb.t().unsqueeze(0)  # [1,128,n]
    x = torch.nn.functional.interpolate(x, size=T, mode="linear", align_corners=True)
    return x.squeeze(0).t().numpy()  # [T,128]


# ----------------------------- Text (Whisper + BERT) --------------------------
def load_whisper(device):
    from transformers import pipeline
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base",
                   device=device, torch_dtype=dtype, chunk_length_s=30,
                   return_timestamps=True)
    return asr


def load_bert(device):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
    return tok, model


@torch.no_grad()
def encode_text(wav_path, T, asr, tok, bert, device):
    """Sentence-wise text embedding (paper Sec 3.1 / Fig 3):
    Whisper -> segments (start,end,text); BERT CLS per segment; expand to the
    segment's second range in F_t [T,768]. Silent seconds stay zero."""
    ft = np.zeros((T, 768), dtype=np.float32)
    try:
        import soundfile as sf
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        # feed raw array so the ASR pipeline does not shell out to ffmpeg
        res = asr({"raw": audio, "sampling_rate": int(sr)}, batch_size=8)
    except Exception as e:
        print(f"  [warn] whisper failed ({e}); zero text", flush=True)
        return ft, []
    chunks = res.get("chunks", []) or []
    segs = []
    texts, spans = [], []
    for ch in chunks:
        ts = ch.get("timestamp", (None, None))
        txt = (ch.get("text") or "").strip()
        if not txt:
            continue
        s, e = ts if ts else (None, None)
        if s is None:
            s = 0.0
        if e is None:
            e = min(s + 30.0, float(T))
        segs.append({"start": float(s), "end": float(e), "text": txt})
        texts.append(txt)
        spans.append((float(s), float(e)))
    if texts:
        # BERT CLS per sentence, batched
        embs = []
        for i in range(0, len(texts), 32):
            bt = texts[i:i + 32]
            enc = tok(bt, return_tensors="pt", padding=True, truncation=True,
                      max_length=128).to(device)
            cls = bert(**enc).last_hidden_state[:, 0].float().cpu().numpy()
            embs.append(cls)
        embs = np.concatenate(embs, 0)  # [S,768]
        for (s, e), emb in zip(spans, embs):
            a = max(0, int(np.floor(s)))
            b = min(T, int(np.ceil(e)))
            if b <= a:
                b = min(T, a + 1)
            ft[a:b] = emb
    return ft, segs


# ----------------------------- main -------------------------------------------
def list_video_ids():
    ids = [os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))]
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_t", type=int, default=0,
                    help="0 = no cap; else truncate T to this many seconds")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TRANS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} shard={args.shard}/{args.nshards}", flush=True)

    proc, vit = load_vit(device)
    vggish = load_vggish(device)
    asr = load_whisper(device)
    tok, bert = load_bert(device)
    print("all encoders loaded", flush=True)

    ids = list_video_ids()
    ids = [v for i, v in enumerate(ids) if i % args.nshards == args.shard]
    print(f"processing {len(ids)} videos in this shard", flush=True)

    done = err = 0
    for k, vid in enumerate(ids):
        out_npz = os.path.join(OUT_DIR, f"{vid}.npz")
        if os.path.exists(out_npz) and not args.overwrite:
            done += 1
            continue
        mp4 = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        try:
            dur = probe_duration(mp4)
            T = max(1, int(round(dur)))
            if args.max_t > 0:
                T = min(T, args.max_t)
            # audio -> wav
            has_aud = has_audio_stream(mp4)
            wav = None
            if has_aud:
                wav = tempfile.mktemp(suffix=".wav")
                subprocess.run([FFMPEG, "-y", "-i", mp4, "-ac", "1", "-ar", "16000",
                                "-vn", "-loglevel", "error", wav],
                               check=True)
            # video
            fv = encode_video(mp4, T, proc, vit, device)          # [T,768]
            # audio (robust: a VGGish failure must not discard video/text feats)
            if has_aud and os.path.exists(wav) and os.path.getsize(wav) > 44:
                try:
                    fa = encode_audio(wav, T, vggish, device)      # [T,128]
                except Exception as ae:
                    print(f"  [warn] VGGish failed on {vid} ({ae}); zero audio", flush=True)
                    fa = np.zeros((T, 128), dtype=np.float32)
                    has_aud = False
            else:
                fa = np.zeros((T, 128), dtype=np.float32)
                has_aud = False
            # text
            if has_aud:
                ft, segs = encode_text(wav, T, asr, tok, bert, device)  # [T,768]
            else:
                ft, segs = np.zeros((T, 768), dtype=np.float32), []
            if wav and os.path.exists(wav):
                os.remove(wav)

            np.savez_compressed(out_npz,
                                fv=fv.astype(np.float16),
                                fa=fa.astype(np.float16),
                                ft=ft.astype(np.float16),
                                T=np.int32(T), dur=np.float32(dur),
                                has_audio=np.int8(1 if has_aud else 0))
            with open(os.path.join(TRANS_DIR, f"{vid}.json"), "w") as f:
                json.dump({"video_id": vid, "duration": dur, "T": T,
                           "segments": segs}, f)
            done += 1
            if k % 20 == 0:
                print(f"[{k}/{len(ids)}] {vid} T={T} aud={has_aud} "
                      f"fv{fv.shape} fa{fa.shape} ft{ft.shape}", flush=True)
        except Exception as e:
            err += 1
            import traceback
            print(f"[ERR] {vid}: {e}\n{traceback.format_exc()}", flush=True)
    print(f"DONE shard {args.shard}: done={done} err={err}", flush=True)


if __name__ == "__main__":
    main()
