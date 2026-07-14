#!/usr/bin/env python3
"""S2S Stage-E extractor — per-frame-group Qwen2.5-VL vectors for HateMM + MHC-EN.

Pre-registration : research-wiki/experiments/exp-s2s-r3.md  (r1 APPROVED-WITH-AMENDMENTS)
Executable spec  : refine-logs/S2S_PROBE_DESIGN.md          (r1 §2/§3/§4/§6/§7)
Review           : refine-logs/S2S_PREREG_REVIEW.md         (A1-A5 + N1-N7)

WHAT THIS DOES
--------------
One frozen Qwen2.5-VL-7B forward per video (the SAME forward that produced the banked
`img_feats`, byte-parity per S2S_PROBE_DESIGN.md §2 — every forward-affecting knob is the
banked one BY IMPORT, not by copy). For each video it keeps a *set* of T temporal
frame-group vectors {g_t} (T = grid_t = num_frames//2), plus the exact algebraic pieces
{n_t, p_S, S, end, grid_thw} needed to reconstruct the banked pooled vector, and runs the
extraction-correctness gates INLINE, HALTing on any violation.

GATES (r1; HALT order matches prereg §7 gate 0-2)
-------------------------------------------------
  0a. TEMPORAL POSITIVE CONTROL (A1) — once at start, on a synthetic 4-colour-pair clip:
      each g_t must be nearest its intended temporal slab, and permuting the input pair
      order must permute {g_t} identically. The ONLY check that exercises the grouping.
  0b. GRID-CONSISTENCY GATE (A1) — per video, from the model's own video_grid_thw +
      spatial_merge_size: n_vis == grid_t*(grid_h//merge)*(grid_w//merge) (catches a wrong
      video_pad_id) AND (n_vis//T) == (grid_h//merge)*(grid_w//merge) (catches a wrong
      per-group size). Strictly stronger than `n_vis % T == 0`.
  1.  G-DECOMP (exact, float32) — L2norm((Sum_t n_t g_t + p_S)/end) == this forward's own
      float32 prefix-mean, max-abs <= 1e-5. Aggregate arithmetic ONLY; grouping-invariant
      (see the A1 correction in the design doc) — necessary, not sufficient. The frame set
      is certified by gates 0a/0b, not by this.
  2.  G-RECON (tolerance) — the fresh banked-formula pooled vector (computed by replicating
      the banked `_encode` prefix pooling line-for-line, bf16 path) vs the BANKED
      `img_feats[v]`: cosine >= 0.9999 AND max-abs <= 1e-3, per non-zero-guard video.

PRECISION NOTE (implementer resolution, flagged for code review)
----------------------------------------------------------------
G-decomp is an EXACT float32 identity: both the grouped reconstruction and its target
(`banked_formula_vec`) are computed from `prefix = last_hidden[:end].float()`, so the only
residual is float summation-order (<= 1e-5). G-recon instead compares against the banked
CACHE, which was produced by the banked bf16 pooling; to make the only difference vs the
cache be cross-run kernel drift (not an accumulation-dtype difference we introduce), the
fresh G-recon vector `grecon_vec` REPLICATES the banked line-for-line
(`last_hidden[:end].mean(0).float()` then L2-norm — generate_VideoMLLM_embedding_HF.py:303,
321-322). This is a strengthening of the §3 pseudocode (which reused the f32 vector for
both anchors); a reviewer should confirm the smoke's G-recon distribution is green.

RESUMABLE (idempotent). One atomic per-video shard under <outdir>/_shards/<outname>/<id>.pt.
On requeue a video whose shard exists AND passes a re-loaded integrity check is skipped; the
final per-split .pt is assembled from shards in gt order. NO forward is recomputed.

FAIL LOUD. No bare except wraps the forward or the gates (the A-line lesson: a swallowed OOM
once produced a garbage cache). The only broad except is around re-loading a possibly-corrupt
shard on resume, and it prints the exception and recomputes (the safe direction).

Zero-GPU beyond this one job; writes only the frame-set caches + gate logs.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ----------------------------------------------------------------------------
# Parity-by-import: reuse the banked extractor's samplers / message builder /
# instruction / gt reader VERBATIM (S2S_PROBE_DESIGN.md §3, §12 resolution 1).
# ----------------------------------------------------------------------------
REPO = "/data/jehc223/RGCL"
EXT_PATH = os.path.join(REPO, "src/utils/generate_VideoMLLM_embedding_HF.py")
_spec = importlib.util.spec_from_file_location("vmllm_ext", EXT_PATH)
ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ext)

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_PIXELS = 360 * 420  # 151200 — banked parity (generate_VideoMLLM_embedding_HF.py default)
NUM_FRAMES_DEFAULT = 8
TEMPORAL_PATCH = 2  # Qwen2.5-VL temporal_patch_size -> T = num_frames // 2

# Tolerances (prereg §7; S2S_PROBE_DESIGN.md §7 anchor table).
DECOMP_TOL = 1e-5
GRECON_COS_MIN = 0.9999
GRECON_MAXABS_MAX = 1e-3
SEED = 20260714


def _log(msg):
    print(msg, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_determinism():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ----------------------------------------------------------------------------
# Core: one frozen forward -> frame-set decomposition + inline gates 0b + 1 (+2).
# ----------------------------------------------------------------------------
@torch.no_grad()
def encode_frameset(frames, processor, model, device, banked_vec=None):
    """Run one frozen prefix forward over `frames`; return the frame set + gate metrics.

    Gates run INLINE: grid-consistency (A1, gate 0b) and G-decomp (gate 1) ALWAYS HALT on
    violation; G-recon (gate 2) runs only when `banked_vec` (the banked img_feats[v]) is
    supplied and HALTs on breach. Raises RuntimeError (never returns partial state).
    """
    messages = ext._build_messages(frames, ext.IMG_INSTRUCTION)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=[frames], return_tensors="pt")
    inputs = inputs.to(device)

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1][0]  # [seq_len, D] bf16
    input_ids = inputs["input_ids"][0]      # [seq_len]
    if last_hidden.shape[0] != input_ids.numel():
        raise RuntimeError(
            "hidden/input_ids length mismatch: {} vs {}".format(
                last_hidden.shape[0], input_ids.numel()
            )
        )

    # Span end = start of the assistant header (banked _encode span="prefix", :296-302).
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
    end = int(positions[-1].item()) if len(positions) > 0 else int(last_hidden.shape[0])
    end = max(end, 1)

    # --- G-recon reference: replicate the banked bf16 pooling line-for-line (:303,:321-322).
    grecon_pooled = last_hidden[:end].mean(dim=0)       # bf16 mean, banked path
    grecon_vec = F.normalize(grecon_pooled.float(), p=2, dim=0)  # [D]

    # --- float32 prefix for the EXACT G-decomp identity.
    prefix = last_hidden[:end].float()                  # [end, D] float32
    banked_formula_vec = F.normalize(prefix.mean(0), p=2, dim=0)  # f32 pooled (G-decomp target)

    # --- vision-token positions within the prefix span.
    video_pad_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    vis_mask = (input_ids == video_pad_id)
    vis_pos = vis_mask[:end].nonzero(as_tuple=True)[0]
    n_vis = int(vis_pos.numel())

    grid = inputs["video_grid_thw"][0]
    grid_t, grid_h, grid_w = int(grid[0].item()), int(grid[1].item()), int(grid[2].item())
    T = grid_t
    merge = int(model.config.vision_config.spatial_merge_size)

    # --- (r1: A1) HARD grid-consistency gate (0b) — strictly stronger than n_vis % T == 0.
    per_expected = (grid_h // merge) * (grid_w // merge)
    n_vis_expected = grid_t * per_expected
    if n_vis != n_vis_expected:
        raise RuntimeError(
            "[A1 grid gate] n_vis {} != grid count {} (grid_thw={}, merge={}) — wrong "
            "video_pad_id / vision-token boundary?".format(
                n_vis, n_vis_expected, (grid_t, grid_h, grid_w), merge
            )
        )
    if T <= 0 or n_vis % T != 0 or (n_vis // T) != per_expected:
        raise RuntimeError(
            "[A1 grid gate] per-group size mismatch: n_vis={} T={} per_expected={}".format(
                n_vis, T, per_expected
            )
        )
    per = n_vis // T

    # --- frame-group decomposition (temporal-major contiguous blocks; layout proof
    # modeling_qwen2_5_vl.py:466-505,529-534,560-562).
    g = []
    n_t = []
    for t in range(T):
        idx = vis_pos[t * per:(t + 1) * per]
        g.append(prefix[idx].mean(0))
        n_t.append(int(idx.numel()))
    g = torch.stack(g)                                  # [T, D] float32
    n_t_t = torch.tensor(n_t, dtype=torch.float32)      # [T]

    # --- non-vision prefix contribution.
    nonvis_pos = (~vis_mask[:end]).nonzero(as_tuple=True)[0]
    p_S = prefix[nonvis_pos].sum(0)                     # [D] float32
    S = int(nonvis_pos.numel())

    # --- (gate 1) G-decomp: exact float32 aggregate identity, HALT on residual > 1e-5.
    recon_mean = (g * n_t_t[:, None]).sum(0).add(p_S).div(float(end))
    decomp_res = float((F.normalize(recon_mean, p=2, dim=0) - banked_formula_vec).abs().max().item())
    if decomp_res > DECOMP_TOL:
        raise RuntimeError("[G-decomp] residual {:.3e} > {} (decomposition arithmetic bug)".format(
            decomp_res, DECOMP_TOL))

    # --- (gate 2) G-recon: banked-cache parity, HALT on breach (skip if no banked vec).
    grecon_cos = None
    grecon_maxabs = None
    if banked_vec is not None:
        bv = banked_vec.float()
        grecon_cos = float(F.cosine_similarity(grecon_vec, bv, dim=0).item())
        grecon_maxabs = float((grecon_vec - bv).abs().max().item())
        if grecon_cos < GRECON_COS_MIN or grecon_maxabs > GRECON_MAXABS_MAX:
            raise RuntimeError(
                "[G-recon] cos {:.6f} (min {}) / max-abs {:.3e} (max {}) — fresh forward "
                "!= banked cache".format(grecon_cos, GRECON_COS_MIN, grecon_maxabs, GRECON_MAXABS_MAX)
            )

    return {
        "g": g.detach().cpu(),                          # [T, D] float32 (fp16 on save)
        "n_t": n_t_t.detach().cpu(),                    # [T]
        "p_S": p_S.detach().cpu(),                      # [D] float32 (fp16 on save)
        "S": S,
        "end": int(end),
        "grid_thw": (grid_t, grid_h, grid_w),
        "T": T,
        "decomp_res": decomp_res,
        "grecon_cos": grecon_cos,
        "grecon_maxabs": grecon_maxabs,
    }


# ----------------------------------------------------------------------------
# (r1: A1) Temporal-structure positive control — gate 0a, HALT.
# ----------------------------------------------------------------------------
def _solid(color, size=336):
    return Image.new("RGB", (size, size), color)


def temporal_positive_control(processor, model, device):
    """Synthesise two 8-frame clips = 4 distinct solid-colour PAIRS in different orders;
    verify each g_t reflects its temporal slab and that permuting the input pair order
    permutes {g_t} by the SAME permutation. HALT on failure. (S2S_PROBE_DESIGN.md §3/§4.)
    """
    colours = [(220, 30, 30), (30, 200, 30), (30, 30, 220), (230, 210, 30)]  # R G B Y
    order_a = [0, 1, 2, 3]
    sigma = [2, 0, 3, 1]  # clip B's group j carries clip A's colour sigma[j]

    def clip(order):
        frames = []
        for c in order:
            frames.append(_solid(colours[c]))
            frames.append(_solid(colours[c]))
        return frames

    _log("[gate 0a] temporal positive control: encoding 2 synthetic 4-pair clips ...")
    ra = encode_frameset(clip(order_a), processor, model, device, banked_vec=None)
    rb = encode_frameset([f for c in sigma for f in (_solid(colours[c]), _solid(colours[c]))],
                         processor, model, device, banked_vec=None)
    if ra["T"] != 4 or rb["T"] != 4:
        raise RuntimeError("[gate 0a] expected T=4 for an 8-frame synthetic clip, got "
                           "{}/{}".format(ra["T"], rb["T"]))
    ga = F.normalize(ra["g"], p=2, dim=1)  # [4, D]
    gb = F.normalize(rb["g"], p=2, dim=1)  # [4, D]
    M = ga @ gb.t()                        # M[i, j] = cos(A_i, B_j)
    # B's group j must match A's group sigma[j] (that shares its colour).
    match = M.argmax(dim=0).tolist()       # for each B-group j -> best A-group i
    if match != sigma:
        raise RuntimeError(
            "[gate 0a] temporal assignment FAILED: argmax match {} != expected sigma {}. "
            "Cross-clip cosine matrix:\n{}".format(match, sigma, np.array2string(
                M.numpy(), precision=3))
        )
    # Within-clip distinctness: each group's best-other-group cosine must be < 1 - eps.
    Maa = ga @ ga.t()
    off = Maa - torch.eye(4) * 2.0  # suppress the diagonal
    if float(off.max().item()) > 0.999:
        raise RuntimeError("[gate 0a] synthetic groups not distinct: max off-diagonal "
                           "cosine {:.4f} > 0.999".format(float(off.max().item())))
    _log("[gate 0a] PASS: g_t assignment tracks the input pair order (match={}); groups "
         "distinct (max off-diag {:.3f}).".format(match, float(off.max().item())))


# ----------------------------------------------------------------------------
# Banked cache (for G-recon) — id -> img_feats vector.
# ----------------------------------------------------------------------------
def load_banked_imgfeats(dataset, outname):
    path = os.path.join(REPO, "data/CLIP_Embedding", dataset,
                        "{}_Qwen2.5-VL-7B-Instruct_HF.pt".format(outname))
    if not os.path.exists(path):
        raise RuntimeError("[G-recon] banked cache not found: {}".format(path))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    ids = obj["ids"][0]
    img = obj["img_feats"]
    return {str(i): img[k] for k, i in enumerate(ids)}


# ----------------------------------------------------------------------------
# Per-video shard I/O (resumable).
# ----------------------------------------------------------------------------
SHARD_KEYS = {"id", "g", "n_t", "p_S", "S", "end", "label", "grid_thw", "zero_guard",
              "decomp_res", "grecon_cos", "grecon_maxabs"}


def shard_ok(path, T_nominal):
    """Integrity check for resume: loadable, keys present, g shape correct, gate green."""
    if not os.path.exists(path):
        return False
    try:
        s = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # narrow purpose: a truncated/corrupt shard -> recompute (safe dir)
        _log("[resume] shard unreadable ({}): {!r} — recomputing".format(path, e))
        return False
    if not SHARD_KEYS.issubset(set(s.keys())):
        return False
    if bool(s["zero_guard"]):
        return True
    g = s["g"]
    if tuple(g.shape) != (int(s["grid_thw"][0]), 3584):
        return False
    if s["decomp_res"] is None or float(s["decomp_res"]) > DECOMP_TOL:
        return False
    return True


def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Split processing.
# ----------------------------------------------------------------------------
def process_split(dataset, split, outname, args, processor, model, device):
    gt_path = os.path.join(REPO, args.gt_dir, dataset, "{}.jsonl".format(split))
    if not os.path.exists(gt_path):
        _log("[WARN] gt not found, skipping split {}: {}".format(split, gt_path))
        return None
    items = ext.read_gt(gt_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    banked = None if args.limit else load_banked_imgfeats(dataset, outname)

    outdir = os.path.join(args.out_root, dataset, "frameset_qwen7b_{}f".format(args.num_frames))
    shard_dir = os.path.join(outdir, "_shards", outname)
    os.makedirs(shard_dir, exist_ok=True)

    T_nominal = args.num_frames // TEMPORAL_PATCH
    video_root = os.path.join(REPO, args.video_dir, dataset, "All")

    n_done = n_new = n_guard = 0
    for k, item in enumerate(items):
        vid = str(item["id"])
        shard_path = os.path.join(shard_dir, "{}.pt".format(vid))
        if shard_ok(shard_path, T_nominal):
            n_done += 1
            continue

        video_path = os.path.join(video_root, "{}.mp4".format(vid))
        frames, ok = ext.load_video_frames(video_path, args.num_frames)
        if not ok:
            n_guard += 1
            shard = {
                "id": vid,
                "g": torch.zeros(T_nominal, 3584, dtype=torch.float16),
                "n_t": torch.zeros(T_nominal, dtype=torch.float32),
                "p_S": torch.zeros(3584, dtype=torch.float16),
                "S": 0, "end": 0, "label": int(item["label"]),
                "grid_thw": (T_nominal, 0, 0), "zero_guard": True,
                "decomp_res": 0.0, "grecon_cos": None, "grecon_maxabs": None,
            }
            atomic_save(shard, shard_path)
            _log("  [{}/{}] {} ZERO-GUARD (undecodable) -> zero frame set".format(k + 1, len(items), vid))
            continue

        bvec = None if banked is None else banked.get(vid)
        r = encode_frameset(frames, processor, model, device, banked_vec=bvec)
        shard = {
            "id": vid,
            "g": r["g"].to(torch.float16),
            "n_t": r["n_t"],
            "p_S": r["p_S"].to(torch.float16),
            "S": int(r["S"]), "end": int(r["end"]), "label": int(item["label"]),
            "grid_thw": tuple(int(x) for x in r["grid_thw"]), "zero_guard": False,
            "decomp_res": float(r["decomp_res"]),
            "grecon_cos": (None if r["grecon_cos"] is None else float(r["grecon_cos"])),
            "grecon_maxabs": (None if r["grecon_maxabs"] is None else float(r["grecon_maxabs"])),
        }
        atomic_save(shard, shard_path)
        n_new += 1
        if (n_new % 20) == 0:
            _log("  [{}] {}/{} new (skipped {} done, {} guard); last decomp={:.2e} "
                 "grecon_cos={}".format(split, n_new, len(items), n_done, n_guard,
                                        r["decomp_res"], r["grecon_cos"]))

    # --- assemble the per-split .pt from shards, in gt order (banked contract).
    ids, gs, n_ts, p_Ss, Ss, ends, labels, grids, guards = [], [], [], [], [], [], [], [], []
    decomp_all, cos_all, maxabs_all = [], [], []
    for item in items:
        vid = str(item["id"])
        s = torch.load(os.path.join(shard_dir, "{}.pt".format(vid)), map_location="cpu",
                       weights_only=False)
        ids.append(vid)
        gs.append(s["g"])
        n_ts.append(s["n_t"])
        p_Ss.append(s["p_S"])
        Ss.append(int(s["S"]))
        ends.append(int(s["end"]))
        labels.append(int(s["label"]))
        grids.append([int(x) for x in s["grid_thw"]])
        guards.append(bool(s["zero_guard"]))
        decomp_all.append(float(s["decomp_res"]))
        if s["grecon_cos"] is not None:
            cos_all.append(float(s["grecon_cos"]))
            maxabs_all.append(float(s["grecon_maxabs"]))

    # Every g must share T for a clean [N, T, D] stack (fixed sampler -> constant T).
    T_set = sorted({int(g.shape[0]) for g in gs})
    if T_set != [T_nominal]:
        raise RuntimeError("[assemble] non-constant T across videos: {} (expected [{}])".format(
            T_set, T_nominal))

    save_obj = {
        "ids": [ids],                                       # banked contract: one sublist
        "g": torch.stack(gs).to(torch.float16),             # [N, T, D]
        "n_t": torch.stack(n_ts).to(torch.int16),           # [N, T]
        "p_S": torch.stack(p_Ss).to(torch.float16),         # [N, D]
        "S": torch.tensor(Ss, dtype=torch.int32),           # [N]
        "end": torch.tensor(ends, dtype=torch.int32),       # [N]
        "labels": torch.tensor(labels, dtype=torch.long),   # [N]
        "grid_thw": torch.tensor(grids, dtype=torch.int32), # [N, 3]
        "zero_guard": torch.tensor(guards, dtype=torch.bool),  # [N]
    }
    out_path = os.path.join(outdir, "{}_frameset.pt".format(outname))
    atomic_save(save_obj, out_path)

    gate = {
        "dataset": dataset, "split": split, "outname": outname, "N": len(ids),
        "T": T_nominal, "zero_guard_count": int(sum(guards)),
        "decomp_res_max": (max(decomp_all) if decomp_all else None),
        "grecon_cos_min": (min(cos_all) if cos_all else None),
        "grecon_maxabs_max": (max(maxabs_all) if maxabs_all else None),
        "grecon_n_checked": len(cos_all),
        "out_path": out_path,
    }
    with open(os.path.join(outdir, "{}_gatelog.json".format(outname)), "w") as f:
        json.dump(gate, f, indent=2)
    _log("[{}/{}] saved N={} T={} guards={} decomp_max={} grecon_cos_min={} "
         "grecon_maxabs_max={} -> {}".format(
             dataset, outname, gate["N"], gate["T"], gate["zero_guard_count"],
             gate["decomp_res_max"], gate["grecon_cos_min"], gate["grecon_maxabs_max"], out_path))
    return gate


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="S2S Stage-E frame-set extractor (frozen Qwen2.5-VL).")
    p.add_argument("--dataset", type=str, required=True, help="HateMM | MHC")
    p.add_argument("--splits", type=str, default="train,val,test")
    p.add_argument("--num_frames", type=int, default=NUM_FRAMES_DEFAULT)
    p.add_argument("--out_root", type=str, default=os.path.join(REPO, "data/CLIP_Embedding"),
                   help="Frame-set cache root (use a throwaway path for --limit smoke).")
    p.add_argument("--gt_dir", type=str, default="data/gt")
    p.add_argument("--video_dir", type=str, default="data/video")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, process only the first N per split AND skip G-recon (smoke).")
    return p.parse_args(argv)


def main():
    args = parse_args()
    set_determinism()
    self_sha = sha256_file(os.path.abspath(__file__))
    _log("=" * 78)
    _log("[S2S extract] script sha256 = {}".format(self_sha))
    _log("[S2S extract] config: dataset={} splits={} num_frames={} device={} limit={} "
         "out_root={}".format(args.dataset, args.splits, args.num_frames, args.device,
                              args.limit, args.out_root))
    _log("[S2S extract] model={} max_pixels={} dtype=bfloat16 attn=sdpa transformers={}".format(
        MODEL, MAX_PIXELS, __import__("transformers").__version__))
    _log("[S2S extract] parity-by-import from {} (sha256 {})".format(
        EXT_PATH, sha256_file(EXT_PATH)))
    _log("=" * 78)

    device = torch.device(args.device)
    _log("[S2S extract] loading model ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None
    )
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)

    # Gate 0a — temporal positive control (HALT before touching any real video).
    temporal_positive_control(processor, model, device)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    t0 = time.time()
    for split in splits:
        if split not in ext.SPLIT_TO_OUTNAME:
            _log("[WARN] split {} has no output-name mapping; skipping.".format(split))
            continue
        outname = ext.SPLIT_TO_OUTNAME[split]
        _log("[S2S extract] --- {} / {} (outname {}) ---".format(args.dataset, split, outname))
        process_split(args.dataset, split, outname, args, processor, model, device)
    _log("[S2S extract] DONE dataset={} in {:.1f}s".format(args.dataset, time.time() - t0))


if __name__ == "__main__":
    main()
