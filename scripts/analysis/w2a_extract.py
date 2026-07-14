#!/usr/bin/env python3
"""W2-A Stage-E' extractor — transcript-conditioned GROUNDED Qwen2.5-VL keys (HateMM + MHC-EN).

Pre-registration : research-wiki/experiments/exp-w2a-grounded.md  (r1 APPROVED-WITH-AMENDMENTS, cb59a94)
Forensic recon   : refine-logs/W2A_FORENSIC_RECON.md
Prereg review    : refine-logs/W2A_PREREG_REVIEW.md

WHAT THIS DOES
--------------
Per video, TWO frozen Qwen2.5-VL-7B forwards (bf16, sdpa, no_grad):

  GROUNDED forward   content = [{text: transcript}, {video: frames}, {text: IMG_INSTRUCTION}]
      transcript-FIRST so the (causal) vision tokens attend BACK to the transcript.
        grd     = L2(mean of the VISION-pad token hidden states)            [primary key]
        grd_pfx = L2(mean of the vision + trailing-instruction span)        [pooling sensitivity]

  IMG-CONTROL forward  content = [{video: frames}, {text: IMG_INSTRUCTION}]  (byte-identical to
      the banked img_feats recipe — the message builder + instruction are IMPORTED verbatim).
        ungrd_vis = L2(mean of the VISION-pad token hidden states)   [ungrounded-vision reference]
        img_recon = L2(mean of the banked PREFIX span [0:end])       [G-recon-IMG parity anchor]

The transcript-first message builder (`_build_grounded_messages`) and the vision-pad span pool are
the NOVEL code surfaces (RE-authored, not imported — the banked `_build_messages`/`_encode` hardcode
the video-first order and the prefix/response spans). Every OTHER forward-affecting knob (frame
sampler, decode helpers, gt reader, IMG_INSTRUCTION, split->outname map, max_pixels, dtype, attn) is
the banked one BY IMPORT / by copy of the banked constant.

GATES (prereg §4; K-numbers = §12 what-would-kill table)
--------------------------------------------------------
  0 (K0)  grid gate + vision-pad contiguity — n_vis == grid_t*(grid_h//merge)*(grid_w//merge) AND the
          vision-pad positions (input_ids == video_token_id) are a SINGLE contiguous block of that
          count, in BOTH forwards, identical mask logic. HALT on violation.
  1 (K1)  G-recon-IMG — img_recon vs banked img_feats[v]: cos >= 0.9999 AND max-abs <= 1e-3, every
          non-guard video (runs under --limit smoke too). HALT on breach.
  2 (K2)  grounding-LIVE — present-transcript-set MEDIAN cos(grd, ungrd_vis) >= 0.999 -> silent no-op
          -> probe VOID (recorded flag; the keys are still cached). tau_live + the empty-transcript
          branch cos are LOGGED DIAGNOSTICS, never HALT (r1 Amdt 4/6).
  3 (K3)  placebo — grd recomputed with a cross-video MISMATCHED (length-comparable) transcript must
          MOVE grd (median cos(grd, grd_placebo) < 0.999 over a >=50-video subset). >= 0.999 -> VOID
          (r1 Amdt 3). Within-video token-shuffle kept as a secondary diagnostic.
  4       length/parity — last_hidden.shape[0] == input_ids.numel() (banked preflight), both forwards;
          the M-RoPE vision-position offset (tokens before the first vision token) is logged.

Gate 0/1/4 HALT inline (fail-loud — NO bare except around a forward or a gate; the A-line lesson).
Gate 2/3 are aggregate (median over a set) -> computed at assembly, recorded as VOID flags + full
distributions in the gate log; the probe / verdict reviewer reads them.

Zero-guard: undecodable video -> zero grd/grd_pfx/ungrd_vis/img_recon (identical to banked). Empty
transcript -> NOT zero-guarded; the grounded forward runs with the "(none)" transcript block; the row's
grd ~ ungrd_vis by construction (mechanism vacuous) and is excluded from the grounding-live present-set
median (its cos is logged in the empty-set distribution instead).

RESUMABLE (idempotent): one atomic per-video shard under <out_root>/<ds>/grounded_qwen7b_<F>f/_shards/
<outname>/<id>.pt; the per-split .pt is assembled from shards in gt order. NO forward recomputed.

Zero GPU beyond this one job. Local GPU only (raw video, license-sensitive). `--self_test` runs the
message-builder + span-indexing logic on SYNTHETIC tensors with NO model / NO GPU.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Parity-by-import: reuse the banked extractor's samplers / gt reader / IMG
# instruction / video-first message builder / split map VERBATIM.
# ----------------------------------------------------------------------------
REPO = "/data/jehc223/RGCL"
EXT_PATH = os.path.join(REPO, "src/utils/generate_VideoMLLM_embedding_HF.py")
_spec = importlib.util.spec_from_file_location("vmllm_ext", EXT_PATH)
ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ext)

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_PIXELS = 360 * 420  # 151200 — banked parity (generate_VideoMLLM_embedding_HF.py default)
NUM_FRAMES_DEFAULT = 8
D_MODEL = 3584  # Qwen2.5-VL-7B hidden size

# --- pre-declared constants (hash-frozen with the script; prereg §16) ---
# GROUNDED transcript block text. Prereg §2/§4 message content = `{text: transcript}`; the team-lead
# directive + review shorthand `[{text:"(none)"}]` fix the empty case to "(none)". We use the RAW
# transcript (no "Transcript:" label) + this sentinel. [SPEC-AMBIGUITY 1 — see module docstring / the
# implementer note returned to the reviewer: §4's zero-guard prose writes the empty block as
# "Transcript: (none)"; we take the un-labelled "(none)" per §2/§4 message-content + team-lead.]
EMPTY_TRANSCRIPT_SENTINEL = "(none)"
GROUNDED_TRANSCRIPT_PREFIX = ""   # no "Transcript: " label (raw transcript block)

GRECON_COS_MIN = 0.9999           # gate 1 (K1)
GRECON_MAXABS_MAX = 1e-3          # gate 1 (K1)
GROUNDING_NOOP_VOID = 0.999       # gate 2 (K2) present-set MEDIAN cos(grd, ungrd_vis) >= -> VOID
PLACEBO_NOOP_VOID = 0.999         # gate 3 (K3) subset MEDIAN cos(grd, grd_placebo) >= -> VOID
PLACEBO_N = 50                    # gate 3 subset size (>=50 per prereg §4)
SEED = 20260715

GROUNDED_DIR_TMPL = "grounded_qwen7b_{}f"


def _log(msg):
    print(msg, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_determinism():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ----------------------------------------------------------------------------
# NOVEL surface 1 — the transcript-FIRST message builder (re-authored, not imported).
# ----------------------------------------------------------------------------
def _grounded_transcript_text(transcript):
    """The GROUNDED transcript content block: raw transcript, or the sentinel when empty."""
    t = transcript if transcript is not None else ""
    t = str(t)
    if not t.strip():
        return EMPTY_TRANSCRIPT_SENTINEL
    return GROUNDED_TRANSCRIPT_PREFIX + t


def _build_grounded_messages(frames, transcript_text):
    """One user turn ordered [transcript]->[video]->[IMG_INSTRUCTION] so the causal vision tokens
    attend BACK to the transcript. This is the mechanism under test (prereg §2)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": transcript_text},
                {"type": "video", "video": frames},
                {"type": "text", "text": ext.IMG_INSTRUCTION},
            ],
        }
    ]


# ----------------------------------------------------------------------------
# NOVEL surface 2 — vision-pad span location + gate-0 asserts + pools (pure tensor logic,
# so `--self_test` can drive it with no model).
# ----------------------------------------------------------------------------
def locate_spans(input_ids, video_pad_id, im_start_id):
    """Return (vis_pos, end). vis_pos = ascending indices of vision-pad tokens; end = start of the
    trailing assistant header (last <|im_start|>), i.e. the banked prefix boundary."""
    positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
    end = int(positions[-1].item()) if len(positions) > 0 else int(input_ids.numel())
    end = max(end, 1)
    vis_pos = (input_ids == video_pad_id).nonzero(as_tuple=True)[0]
    return vis_pos, end


def grid_contiguity_gate(vis_pos, grid_thw, merge, forward_name):
    """Gate 0 (K0): grid-count match + a single contiguous vision-pad block. HALT on violation.
    IDENTICAL logic for both forwards (prereg §4 gate 0, r1 Amdt 7)."""
    grid_t, grid_h, grid_w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    per_expected = (grid_h // merge) * (grid_w // merge)
    n_vis_expected = grid_t * per_expected
    n_vis = int(vis_pos.numel())
    if n_vis != n_vis_expected:
        raise RuntimeError(
            "[gate0 grid/{}] n_vis {} != grid count {} (grid_thw={}, merge={}) — wrong "
            "video_pad_id / vision boundary?".format(
                forward_name, n_vis, n_vis_expected, (grid_t, grid_h, grid_w), merge))
    if n_vis <= 0:
        raise RuntimeError("[gate0 grid/{}] no vision-pad tokens found".format(forward_name))
    lo = int(vis_pos.min().item())
    hi = int(vis_pos.max().item())
    if (hi - lo + 1) != n_vis:
        raise RuntimeError(
            "[gate0 contiguity/{}] vision-pad positions are NOT a single contiguous block: "
            "span [{}..{}] has {} slots but n_vis={} ".format(forward_name, lo, hi, hi - lo + 1, n_vis))
    return {"grid_thw": (grid_t, grid_h, grid_w), "n_vis": n_vis,
            "per_group": per_expected, "vis_lo": lo, "vis_hi": hi}


def _pool_slice_norm(last_hidden, end):
    """Banked PREFIX pool, replicated line-for-line (bf16 mean -> float -> L2). Used for img_recon so
    the ONLY residual vs the banked cache is cross-run kernel drift (S2S G-recon precedent)."""
    pooled = last_hidden[:end].mean(dim=0)
    return F.normalize(pooled.float(), p=2, dim=0)


def _pool_idx_norm(last_hidden, idx):
    """Pool over an ascending index tensor (bf16 mean -> float -> L2). Used for grd / grd_pfx /
    ungrd_vis so grd and ungrd_vis share an identical dtype path (their cos is the gate-2 metric)."""
    pooled = last_hidden[idx].mean(dim=0)
    return F.normalize(pooled.float(), p=2, dim=0)


def pool_grounded(last_hidden, vis_pos, end):
    """grd = mean over the vision-pad span; grd_pfx = mean over [first_vis .. end) = vision + the
    trailing IMG_INSTRUCTION (+ im_end), EXCLUDING the leading transcript (prereg §2/§4
    "vision+trailing-instruction span"). [SPEC-AMBIGUITY 2 — resolved to first_vis..end.]"""
    grd = _pool_idx_norm(last_hidden, vis_pos)
    first_vis = int(vis_pos.min().item())
    pfx_idx = torch.arange(first_vis, end, device=last_hidden.device)
    grd_pfx = _pool_idx_norm(last_hidden, pfx_idx)
    return grd, grd_pfx


def pool_control(last_hidden, vis_pos, end):
    """ungrd_vis = vision-pad pool (the ungrounded reference); img_recon = banked prefix pool."""
    ungrd_vis = _pool_idx_norm(last_hidden, vis_pos)
    img_recon = _pool_slice_norm(last_hidden, end)
    return ungrd_vis, img_recon


# ----------------------------------------------------------------------------
# Forward runner (shared by both forwards) — fail-loud, gate 0 + length-parity inline.
# ----------------------------------------------------------------------------
@torch.no_grad()
def _run_forward(messages, frames, processor, model, device, forward_name):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=[frames], return_tensors="pt")
    inputs = inputs.to(device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1][0]          # [seq, D] bf16
    input_ids = inputs["input_ids"][0]              # [seq]
    # gate 4 — length/parity (banked preflight): vision tokens are masked_scatter'd in place.
    if last_hidden.shape[0] != input_ids.numel():
        raise RuntimeError("[gate4 len-parity/{}] hidden {} != input_ids {}".format(
            forward_name, last_hidden.shape[0], input_ids.numel()))
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    video_pad_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    vis_pos, end = locate_spans(input_ids, video_pad_id, im_start_id)
    grid = inputs["video_grid_thw"][0]
    merge = int(model.config.vision_config.spatial_merge_size)
    gate0 = grid_contiguity_gate(vis_pos, grid, merge, forward_name)
    # M-RoPE vision-position offset diagnostic = tokens before the first vision token.
    rope_vis_offset = int(vis_pos.min().item())
    return last_hidden, vis_pos, end, gate0, rope_vis_offset


@torch.no_grad()
def encode_video(frames, transcript, processor, model, device, banked_vec):
    """Run BOTH forwards for one video; run gates 0/1/4 inline (HALT); return the four keys +
    per-video gate metrics. grounding_cos = cos(grd, ungrd_vis) (gate-2 raw material)."""
    # --- GROUNDED forward ---
    t_text = _grounded_transcript_text(transcript)
    g_hidden, g_vis, g_end, g_gate0, g_off = _run_forward(
        _build_grounded_messages(frames, t_text), frames, processor, model, device, "grounded")
    grd, grd_pfx = pool_grounded(g_hidden, g_vis, g_end)

    # --- IMG-CONTROL forward (byte-identical to banked img_feats; message builder IMPORTED) ---
    c_hidden, c_vis, c_end, c_gate0, c_off = _run_forward(
        ext._build_messages(frames, ext.IMG_INSTRUCTION), frames, processor, model, device,
        "img_control")
    ungrd_vis, img_recon = pool_control(c_hidden, c_vis, c_end)

    # --- gate 1 (K1) G-recon-IMG: fresh img_recon must reproduce the banked img_feats[v] ---
    grecon_cos = grecon_maxabs = None
    if banked_vec is not None:
        gv = img_recon.detach().cpu()
        bv = banked_vec.detach().float().cpu()
        grecon_cos = float(F.cosine_similarity(gv, bv, dim=0).item())
        grecon_maxabs = float((gv - bv).abs().max().item())
        if grecon_cos < GRECON_COS_MIN or grecon_maxabs > GRECON_MAXABS_MAX:
            raise RuntimeError(
                "[gate1 G-recon-IMG] cos {:.6f} (min {}) / max-abs {:.3e} (max {}) — fresh "
                "img-control forward != banked cache".format(
                    grecon_cos, GRECON_COS_MIN, grecon_maxabs, GRECON_MAXABS_MAX))

    grounding_cos = float(F.cosine_similarity(
        grd.detach().cpu(), ungrd_vis.detach().cpu(), dim=0).item())
    return {
        "grd": grd.detach().cpu(), "grd_pfx": grd_pfx.detach().cpu(),
        "ungrd_vis": ungrd_vis.detach().cpu(), "img_recon": img_recon.detach().cpu(),
        "grid_thw": g_gate0["grid_thw"], "img_grid_thw": c_gate0["grid_thw"],
        "grounding_cos": grounding_cos,
        "grecon_cos": grecon_cos, "grecon_maxabs": grecon_maxabs,
        "rope_vis_offset_grounded": g_off, "rope_vis_offset_control": c_off,
    }


@torch.no_grad()
def placebo_grd(frames, partner_transcript, processor, model, device):
    """Gate 3 (K3): grounded forward with video i's frames + a MISMATCHED partner transcript; pool
    grd (vision-pad span). Its cos with the true grd measures transcript-CONTENT sensitivity."""
    t_text = _grounded_transcript_text(partner_transcript)
    hidden, vis, end, _g0, _off = _run_forward(
        _build_grounded_messages(frames, t_text), frames, processor, model, device, "placebo")
    grd_p, _pfx = pool_grounded(hidden, vis, end)
    return grd_p.detach().cpu()


# ----------------------------------------------------------------------------
# Banked img_feats cache (gate 1 anchor).
# ----------------------------------------------------------------------------
def load_banked_imgfeats(dataset, outname):
    path = os.path.join(REPO, "data/CLIP_Embedding", dataset,
                        "{}_Qwen2.5-VL-7B-Instruct_HF.pt".format(outname))
    if not os.path.exists(path):
        raise RuntimeError("[gate1] banked cache not found: {}".format(path))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    ids = obj["ids"][0]
    img = obj["img_feats"]
    return {str(i): img[k] for k, i in enumerate(ids)}


# ----------------------------------------------------------------------------
# Placebo pairing (deterministic, length-comparable, drawn from the FULL split gt).
# ----------------------------------------------------------------------------
def build_placebo_partners(items):
    """For every present-transcript item, assign a length-comparable partner id (its neighbour in
    the char-length-sorted order; j != i). Deterministic. Returns {id: (partner_id, partner_text)}."""
    present = [(str(it["id"]), str(it.get("text") or "")) for it in items
               if str(it.get("text") or "").strip()]
    if len(present) < 2:
        return {}
    order = sorted(range(len(present)), key=lambda k: (len(present[k][1]), present[k][0]))
    partner = {}
    m = len(order)
    for rank, k in enumerate(order):
        vid = present[k][0]
        # successor in sorted order (cyclic); guaranteed distinct id because m >= 2.
        pk = order[(rank + 1) % m]
        if present[pk][0] == vid:
            pk = order[(rank - 1) % m]
        partner[vid] = (present[pk][0], present[pk][1])
    return partner


# ----------------------------------------------------------------------------
# Per-video shard I/O.
# ----------------------------------------------------------------------------
SHARD_KEYS = {"id", "grd", "grd_pfx", "ungrd_vis", "img_recon", "label", "grid_thw",
              "zero_guard", "empty_transcript", "grounding_cos", "grecon_cos", "grecon_maxabs",
              "placebo_partner_id", "placebo_cos", "rope_vis_offset_grounded",
              "rope_vis_offset_control", "transcript_len_chars"}


def shard_ok(path):
    if not os.path.exists(path):
        return False
    try:
        s = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # narrow: corrupt/truncated shard on resume -> recompute (safe direction)
        _log("[resume] shard unreadable ({}): {!r} — recomputing".format(path, e))
        return False
    if not SHARD_KEYS.issubset(set(s.keys())):
        return False
    if bool(s["zero_guard"]):
        return True
    if tuple(s["grd"].shape) != (D_MODEL,):
        return False
    if s["grecon_cos"] is None or float(s["grecon_cos"]) < GRECON_COS_MIN:
        return False
    if s["grecon_maxabs"] is None or float(s["grecon_maxabs"]) > GRECON_MAXABS_MAX:
        return False
    return True


def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _zero_shard(vid, label):
    z = torch.zeros(D_MODEL, dtype=torch.float16)
    return {
        "id": vid, "grd": z.clone(), "grd_pfx": z.clone(), "ungrd_vis": z.clone(),
        "img_recon": z.clone(), "label": int(label), "grid_thw": (0, 0, 0),
        "zero_guard": True, "empty_transcript": False, "grounding_cos": None,
        "grecon_cos": None, "grecon_maxabs": None, "placebo_partner_id": None,
        "placebo_cos": None, "rope_vis_offset_grounded": None,
        "rope_vis_offset_control": None, "transcript_len_chars": 0,
    }


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

    banked = load_banked_imgfeats(dataset, outname)
    partners = build_placebo_partners(items)

    grounded_dir = GROUNDED_DIR_TMPL.format(args.num_frames)
    outdir = os.path.join(args.out_root, dataset, grounded_dir)
    shard_dir = os.path.join(outdir, "_shards", outname)
    os.makedirs(shard_dir, exist_ok=True)
    video_root = os.path.join(REPO, args.video_dir, dataset, "All")

    # placebo subset = first PLACEBO_N present-transcript videos (gt order) actually processed.
    present_ids = [str(it["id"]) for it in items if str(it.get("text") or "").strip()]
    placebo_subset = set(present_ids[:PLACEBO_N]) if len(partners) >= 2 else set()

    n_done = n_new = n_guard = n_placebo = 0
    for k, item in enumerate(items):
        vid = str(item["id"])
        shard_path = os.path.join(shard_dir, "{}.pt".format(vid))
        if shard_ok(shard_path):
            n_done += 1
            continue

        transcript = str(item.get("text") or "")
        is_empty = not transcript.strip()
        video_path = os.path.join(video_root, "{}.mp4".format(vid))
        frames, ok = ext.load_video_frames(video_path, args.num_frames)
        if not ok:
            n_guard += 1
            atomic_save(_zero_shard(vid, item["label"]), shard_path)
            _log("  [{}/{}] {} ZERO-GUARD (undecodable) -> zero keys".format(k + 1, len(items), vid))
            continue

        bvec = banked.get(vid)  # gate 1 anchor (runs under --limit smoke too)
        r = encode_video(frames, transcript, processor, model, device, bvec)

        placebo_partner_id = placebo_cos = None
        if vid in placebo_subset and vid in partners:
            partner_id, partner_text = partners[vid]
            grd_p = placebo_grd(frames, partner_text, processor, model, device)
            placebo_cos = float(F.cosine_similarity(r["grd"], grd_p, dim=0).item())
            placebo_partner_id = partner_id
            n_placebo += 1

        shard = {
            "id": vid,
            "grd": r["grd"].to(torch.float16), "grd_pfx": r["grd_pfx"].to(torch.float16),
            "ungrd_vis": r["ungrd_vis"].to(torch.float16),
            "img_recon": r["img_recon"].to(torch.float16),
            "label": int(item["label"]),
            "grid_thw": tuple(int(x) for x in r["grid_thw"]),
            "zero_guard": False, "empty_transcript": bool(is_empty),
            "grounding_cos": float(r["grounding_cos"]),
            "grecon_cos": (None if r["grecon_cos"] is None else float(r["grecon_cos"])),
            "grecon_maxabs": (None if r["grecon_maxabs"] is None else float(r["grecon_maxabs"])),
            "placebo_partner_id": placebo_partner_id,
            "placebo_cos": (None if placebo_cos is None else float(placebo_cos)),
            "rope_vis_offset_grounded": int(r["rope_vis_offset_grounded"]),
            "rope_vis_offset_control": int(r["rope_vis_offset_control"]),
            "transcript_len_chars": len(transcript),
        }
        atomic_save(shard, shard_path)
        n_new += 1
        if (n_new % 20) == 0:
            _log("  [{}] {}/{} new (skip {} done, {} guard, {} placebo); last grecon_cos={} "
                 "grounding_cos={:.4f}".format(split, n_new, len(items), n_done, n_guard,
                                               n_placebo, r["grecon_cos"], r["grounding_cos"]))

    return assemble_split(dataset, split, outname, items, shard_dir, outdir, args)


def assemble_split(dataset, split, outname, items, shard_dir, outdir, args):
    ids, grd, grd_pfx, ungrd_vis, img_recon = [], [], [], [], []
    labels, grids, guards, empties, gcos = [], [], [], [], []
    grecon_cos, grecon_maxabs = [], []
    present_gcos, empty_gcos = [], []
    placebo_cos, placebo_rows = [], []
    rope_grounded, rope_control = [], []
    for item in items:
        vid = str(item["id"])
        s = torch.load(os.path.join(shard_dir, "{}.pt".format(vid)), map_location="cpu",
                       weights_only=False)
        ids.append(vid)
        grd.append(s["grd"]); grd_pfx.append(s["grd_pfx"])
        ungrd_vis.append(s["ungrd_vis"]); img_recon.append(s["img_recon"])
        labels.append(int(s["label"]))
        grids.append([int(x) for x in s["grid_thw"]])
        guards.append(bool(s["zero_guard"]))
        empties.append(bool(s["empty_transcript"]))
        gcos.append(None if s["grounding_cos"] is None else float(s["grounding_cos"]))
        if s["grecon_cos"] is not None:
            grecon_cos.append(float(s["grecon_cos"]))
            grecon_maxabs.append(float(s["grecon_maxabs"]))
        if (not s["zero_guard"]) and s["grounding_cos"] is not None:
            (empty_gcos if s["empty_transcript"] else present_gcos).append(float(s["grounding_cos"]))
        if s["placebo_cos"] is not None:
            placebo_cos.append(float(s["placebo_cos"]))
            placebo_rows.append({"id": vid, "partner": s["placebo_partner_id"],
                                 "cos": float(s["placebo_cos"])})
        if s["rope_vis_offset_grounded"] is not None:
            rope_grounded.append(int(s["rope_vis_offset_grounded"]))
            rope_control.append(int(s["rope_vis_offset_control"]))

    save_obj = {
        "ids": [ids],                                            # banked contract: one sublist
        "grd": torch.stack(grd).to(torch.float16),               # [N, D]
        "grd_pfx": torch.stack(grd_pfx).to(torch.float16),       # [N, D]
        "ungrd_vis": torch.stack(ungrd_vis).to(torch.float16),   # [N, D]
        "img_recon": torch.stack(img_recon).to(torch.float16),   # [N, D]
        "labels": torch.tensor(labels, dtype=torch.long),        # [N]
        "grid_thw": torch.tensor(grids, dtype=torch.int32),      # [N, 3]
        "zero_guard": torch.tensor(guards, dtype=torch.bool),    # [N]
        "empty_transcript": torch.tensor(empties, dtype=torch.bool),  # [N]
        "grounding_cos": torch.tensor([(-2.0 if c is None else c) for c in gcos],
                                      dtype=torch.float32),      # [N]; -2 sentinel = guard row
    }
    out_path = os.path.join(outdir, "{}_grounded.pt".format(outname))
    atomic_save(save_obj, out_path)

    def _stats(a):
        if not a:
            return None
        a = np.asarray(a, dtype=np.float64)
        return {"n": int(a.size), "median": float(np.median(a)), "mean": float(a.mean()),
                "min": float(a.min()), "max": float(a.max())}

    present_stats = _stats(present_gcos)
    empty_stats = _stats(empty_gcos)
    placebo_stats = _stats(placebo_cos)
    # tau_live (diagnostic only, r1 Amdt 4): median(present) - 0.5*(present - empty gap).
    tau_live = None
    if present_stats is not None:
        empty_med = empty_stats["median"] if empty_stats is not None else 1.0
        tau_live = float(present_stats["median"] - 0.5 * (present_stats["median"] - empty_med))
    grounding_void = bool(present_stats is not None and present_stats["median"] >= GROUNDING_NOOP_VOID)
    placebo_void = bool(placebo_stats is not None and placebo_stats["median"] >= PLACEBO_NOOP_VOID)

    gate = {
        "dataset": dataset, "split": split, "outname": outname, "N": len(ids),
        "zero_guard_count": int(sum(guards)), "empty_transcript_count": int(sum(empties)),
        # gate 1 (K1) G-recon-IMG
        "grecon_cos_min": (min(grecon_cos) if grecon_cos else None),
        "grecon_maxabs_max": (max(grecon_maxabs) if grecon_maxabs else None),
        "grecon_n_checked": len(grecon_cos),
        # gate 2 (K2) grounding-live (present-set median is the BINDING VOID; rest diagnostic)
        "grounding_present": present_stats,
        "grounding_empty_diag": empty_stats,
        "grounding_void_present_median_ge_%.3f" % GROUNDING_NOOP_VOID: grounding_void,
        "tau_live_diag": tau_live,
        # gate 3 (K3) placebo (subset median is the BINDING VOID)
        "placebo": placebo_stats,
        "placebo_void_median_ge_%.3f" % PLACEBO_NOOP_VOID: placebo_void,
        "placebo_rows": placebo_rows,
        # gate 4 diagnostic — M-RoPE vision-position offset
        "rope_vis_offset_grounded_median": (int(np.median(rope_grounded)) if rope_grounded else None),
        "rope_vis_offset_control_median": (int(np.median(rope_control)) if rope_control else None),
        "out_path": out_path,
    }
    with open(os.path.join(outdir, "{}_gatelog.json".format(outname)), "w") as f:
        json.dump(gate, f, indent=2)
    _log("[{}/{}] saved N={} guard={} empty={} grecon_cos_min={} grecon_maxabs_max={} "
         "grounding_present_median={} grounding_VOID={} placebo_median={} placebo_VOID={} -> {}".format(
             dataset, outname, gate["N"], gate["zero_guard_count"], gate["empty_transcript_count"],
             gate["grecon_cos_min"], gate["grecon_maxabs_max"],
             (None if present_stats is None else round(present_stats["median"], 4)), grounding_void,
             (None if placebo_stats is None else round(placebo_stats["median"], 4)), placebo_void,
             out_path))
    return gate


# ----------------------------------------------------------------------------
# CPU-only self-test — message builder + span-indexing logic on SYNTHETIC tensors (no model).
# ----------------------------------------------------------------------------
def self_test():
    _log("[self_test] message-builder + span-indexing on synthetic tensors (no GPU/model) ...")
    VID, IM, TXT = 900, 901, 902  # fake token ids: video-pad / <|im_start|> / other text

    # (a) message builders produce the pre-declared block order.
    gm = _build_grounded_messages(["<f>"], _grounded_transcript_text("hello world"))
    types = [c["type"] for c in gm[0]["content"]]
    assert types == ["text", "video", "text"], types
    assert gm[0]["content"][0]["text"] == "hello world"
    assert gm[0]["content"][2]["text"] == ext.IMG_INSTRUCTION
    assert _grounded_transcript_text("") == EMPTY_TRANSCRIPT_SENTINEL
    assert _grounded_transcript_text("   ") == EMPTY_TRANSCRIPT_SENTINEL
    cm = ext._build_messages(["<f>"], ext.IMG_INSTRUCTION)
    assert [c["type"] for c in cm[0]["content"]] == ["video", "text"]

    # (b) synthetic GROUNDED sequence: [sys IM ...text...][TXT transcript][VIS*nv][TXT instr][IM asst]
    nv = 6
    seq = [IM, TXT, TXT, TXT]          # system/user header + transcript block (positions 0..3)
    vis_lo = len(seq)
    seq += [VID] * nv                  # contiguous vision block
    seq += [TXT, TXT]                  # trailing IMG_INSTRUCTION
    end_expected = len(seq)            # assistant header starts here
    seq += [IM, TXT]                   # <|im_start|>assistant\n
    input_ids = torch.tensor(seq)
    hidden = torch.arange(len(seq) * 4, dtype=torch.float32).reshape(len(seq), 4)
    vis_pos, end = locate_spans(input_ids, VID, IM)
    assert end == end_expected, (end, end_expected)
    assert vis_pos.tolist() == list(range(vis_lo, vis_lo + nv)), vis_pos.tolist()
    # matching grid: grid_t=nv, h=w=merge -> per_group=(2//2)*(2//2)=1 -> count=nv.
    g0 = grid_contiguity_gate(vis_pos, (nv, 2, 2), 2, "self")
    assert g0["n_vis"] == nv and g0["vis_lo"] == vis_lo
    grd, grd_pfx = pool_grounded(hidden, vis_pos, end)
    exp_grd = F.normalize(hidden[vis_pos].mean(0).float(), p=2, dim=0)
    assert torch.allclose(grd, exp_grd, atol=1e-6)
    exp_pfx = F.normalize(hidden[torch.arange(vis_lo, end)].mean(0).float(), p=2, dim=0)
    assert torch.allclose(grd_pfx, exp_pfx, atol=1e-6)
    ungrd, img_recon = pool_control(hidden, vis_pos, end)
    assert torch.allclose(img_recon, F.normalize(hidden[:end].mean(0).float(), p=2, dim=0), atol=1e-6)
    assert torch.allclose(ungrd, exp_grd, atol=1e-6)

    # (c) gate-0 MUST raise on a non-contiguous vision block.
    bad = torch.tensor([IM, VID, TXT, VID, TXT, IM, TXT])
    bvis, bend = locate_spans(bad, VID, IM)
    raised = False
    try:
        grid_contiguity_gate(bvis, (2, 2, 2), 2, "self-bad")
    except RuntimeError:
        raised = True
    assert raised, "gate-0 failed to catch a non-contiguous vision block"

    # (d) gate-0 MUST raise on a grid-count mismatch.
    raised = False
    try:
        grid_contiguity_gate(vis_pos, (nv + 1, 2, 2), 2, "self-mismatch")
    except RuntimeError:
        raised = True
    assert raised, "gate-0 failed to catch a grid-count mismatch"

    # (e) placebo partner assignment: distinct, length-comparable.
    items = [{"id": "a", "text": "x"}, {"id": "b", "text": "xxxx"}, {"id": "c", "text": "xx"},
             {"id": "d", "text": ""}]
    partners = build_placebo_partners(items)
    assert set(partners.keys()) == {"a", "b", "c"}  # 'd' empty -> excluded
    for vid, (pid, _txt) in partners.items():
        assert pid != vid
    _log("[self_test] PASS — builders, span indexing, pools, gate-0 raises, placebo pairing all OK.")


# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="W2-A Stage-E' grounded extractor (frozen Qwen2.5-VL).")
    p.add_argument("--dataset", type=str, default=None, help="HateMM | MHC")
    p.add_argument("--splits", type=str, default="train,val,test")
    p.add_argument("--num_frames", type=int, default=NUM_FRAMES_DEFAULT)
    p.add_argument("--out_root", type=str, default=os.path.join(REPO, "data/CLIP_Embedding"),
                   help="Grounded-cache root (use a throwaway path for the --limit smoke).")
    p.add_argument("--gt_dir", type=str, default="data/gt")
    p.add_argument("--video_dir", type=str, default="data/video")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0, help="If >0, first N items per split (smoke).")
    p.add_argument("--self_test", action="store_true",
                   help="Run the CPU-only message-builder + span-indexing self-test and exit.")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.dataset:
        raise SystemExit("--dataset is required (HateMM | MHC) unless --self_test")

    set_determinism()
    self_sha = sha256_file(os.path.abspath(__file__))
    _log("=" * 78)
    _log("[W2-A extract] script sha256 = {}".format(self_sha))
    _log("[W2-A extract] config: dataset={} splits={} num_frames={} device={} limit={} out_root={}".format(
        args.dataset, args.splits, args.num_frames, args.device, args.limit, args.out_root))
    _log("[W2-A extract] model={} max_pixels={} dtype=bfloat16 attn=sdpa transformers={}".format(
        MODEL, MAX_PIXELS, __import__("transformers").__version__))
    _log("[W2-A extract] parity-by-import from {} (sha256 {})".format(EXT_PATH, sha256_file(EXT_PATH)))
    _log("[W2-A extract] grounded transcript block: raw text | empty->'{}' ; grd_pfx=[first_vis:end]".format(
        EMPTY_TRANSCRIPT_SENTINEL))
    _log("=" * 78)

    # CPU self-test always runs first (cheap, HALTs the job before model load on any logic bug).
    self_test()

    device = torch.device(args.device)
    _log("[W2-A extract] loading model ...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    t0 = time.time()
    for split in splits:
        if split not in ext.SPLIT_TO_OUTNAME:
            _log("[WARN] split {} has no output-name mapping; skipping.".format(split))
            continue
        outname = ext.SPLIT_TO_OUTNAME[split]
        _log("[W2-A extract] --- {} / {} (outname {}) ---".format(args.dataset, split, outname))
        process_split(args.dataset, split, outname, args, processor, model, device)
    _log("[W2-A extract] DONE dataset={} in {:.1f}s".format(args.dataset, time.time() - t0))


if __name__ == "__main__":
    main()
