#!/usr/bin/env python
"""C02 evidence-density view extractor -- TEXT STREAM ONLY, train + dev_seen only.

Record: refine-logs/C02_A0_V9_RECORD.md.  Registry authority:
TARGET_STATE.json::iteration_8_stage0_bounded_extraction_amendment.

REUSE, NOT REWRITE
    The encoder, the frame sampler, the chat template, the pooling span, the
    instruction constants and the prompt assembly are IMPORTED UNMODIFIED from the
    deployed extractor src/utils/generate_VideoMLLM_embedding_lora_HF.py, whose
    sha256 is asserted at start-up.  The only new numerical surface in this file is
    (a) which STRING goes into the transcript slot and (b) the output contract.
    If the deployed extractor changes, this script refuses to run.

WHY ONLY THE TEXT STREAM
    img_feats are produced from frames + a fixed instruction and never see the
    title or the transcript (generate_VideoMLLM_embedding_lora_HF.py:427-430), so
    they are invariant under every C02 view BY CONSTRUCTION.  Re-extracting them
    would only inject GPU non-determinism into a quantity that must be identical
    across arms.  The A0 arena therefore pairs the BANKED native img_feats with the
    per-view text_feats produced here.  NAT is re-extracted in this same session so
    every arm's floor and treatment come off one GPU, one driver and one process.

SPLIT SCOPE -- HARD
    Only gt splits `train` and `val` (-> cache names `train` and `dev_seen`) may be
    requested.  `test` is rejected by an assertion before anything is opened, the
    output path is refused if it contains a test-like token, and torch.load is
    wrapped by the same guard the head-space instrument uses.

OUTPUT (one file per dataset x split x view; never touches an existing cache)
    data/CLIP_Embedding/<DS>/{split}_{base_tag}-c02den-{VIEW}.pt
      {"ids": [[id, ...]], "text_feats": FloatTensor[N, D], "labels": LongTensor[N],
       "c02_view": VIEW, "c02_run_id": ..., "c02_base_tag": ...}
    The dict deliberately has NO "img_feats" key, so run_rac / load_feats_MHC
    cannot silently consume a view file as if it were a full cache.

DEGENERATE ORBITS
    For an item whose view string equals NAT (empty text, length guard, empty
    window; see src/utils/c02_density_views.py) the NAT vector is computed once and
    COPIED into that view slot, so the identity is bit-exact rather than merely
    close.  Videos that fail to decode take the deployed zero-vector guard and are
    zero in every view, exactly as they are zero in the banked cache.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

if not __debug__:
    raise SystemExit("REFUSING TO RUN: python -O strips the assert-based guards")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import c02_density_views as V  # noqa: E402
import generate_VideoMLLM_embedding_lora_HF as BASE  # noqa: E402

FROZEN_BASE_SHA256 = "75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399"
FROZEN_VIEWS_SHA256 = "44fbb00bf88ed1cbe7df2346d0961a172e8cfadd202af49d8b75f8ad7def3a52"

SPLIT_TO_OUTNAME = {"train": "train", "val": "dev_seen"}

# ---------------------------------------------------------------- GPU-budget guard
# Amendment condition (f) makes an over-budget run a PROTOCOL VIOLATION whose result is
# VOID.  Nothing in SLURM enforces it here (CLAUDE.md forbids --time), so the job
# enforces it itself.  The guard may only ever STOP work.  It never alters, truncates or
# partially writes a computed result: view caches and the manifest are written only after
# a whole split completes, so a breach leaves the in-progress dataset entirely unwritten
# and any already-completed dataset intact.  A breach record is published and the process
# exits with a distinct code.
BUDGET_EXIT_CODE = 5
BUDGET_ITEM_HEADROOM_FACTOR = 2.0   # require 2x the slowest item seen so far
BUDGET_MIN_ITEM_HEADROOM_S = 60.0   # ...and never less than this


class BudgetExceeded(Exception):
    pass


def budget_remaining(deadline_epoch):
    return float(deadline_epoch) - time.time()


def budget_check(deadline_epoch, max_item_s, where):
    """Fail-closed: refuse to START work that could carry the job past the deadline.

    The deadline already carries a wrapper-side safety margin below the hard cap; this
    adds a per-item headroom of max(2x slowest item so far, 60 s) so that a single
    unusually slow item cannot straddle it.
    """
    remaining = budget_remaining(deadline_epoch)
    need = max(BUDGET_ITEM_HEADROOM_FACTOR * float(max_item_s),
               BUDGET_MIN_ITEM_HEADROOM_S)
    if remaining < need:
        raise BudgetExceeded(
            "GPU budget guard at {}: {:.1f}s remain before the deadline but {:.1f}s of "
            "headroom are required (slowest item so far {:.1f}s)".format(
                where, remaining, need, max_item_s))
    return remaining
FORBIDDEN_TOKENS = ("test_seen", "/test", "test.jsonl", "test_")

_ORIG_TORCH_LOAD = torch.load


def _guarded_torch_load(f, *a, **kw):
    s = str(f)
    for tok in ("test_seen", "/test"):
        if tok in s:
            raise RuntimeError("TEST-SPLIT GUARD: refusing to open {}".format(s))
    return _ORIG_TORCH_LOAD(f, *a, **kw)


torch.load = _guarded_torch_load


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def assert_no_test_token(path):
    low = str(path).lower()
    for tok in FORBIDDEN_TOKENS:
        if tok in low:
            raise RuntimeError("TEST-PATH GUARD: refusing path {}".format(path))
    return path


def view_out_path(exp_folder, dataset, outname, base_tag, view):
    return os.path.join(exp_folder, dataset,
                        "{}_{}-c02den-{}.pt".format(outname, base_tag, view))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="C02 density-view text-stream extractor.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--base_tag", required=True,
                    help="the banked native cache tag this view set is paired with, "
                         "e.g. Qwen2.5-VL-7B-Instruct-LoRA-curric_HF")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--EXP_FOLDER", default="./data/CLIP_Embedding")
    ap.add_argument("--gt_dir", default="./data/gt")
    ap.add_argument("--video_dir", default="./data/video")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget_deadline_epoch", type=float, required=True,
                    help="absolute unix time after which no further work may START. "
                         "REQUIRED: amendment condition (f) voids an over-budget run, so "
                         "the guard is not optional and has no default.")
    return ap.parse_args(argv)


def build_text_prompt(transcript, title):
    """Byte-identical to the deployed assembly (lora extractor:438-442) with the
    deployed English defaults, differing only in the transcript string supplied."""
    return (BASE.TEXT_INSTRUCTION
            + "\n" + "Title: " + (title if title else "(none)")
            + "\n" + "Transcript: " + (transcript if transcript else "(none)"))


def main():
    a = parse_args()
    t_job0 = time.time()

    base_path = os.path.join(_HERE, "generate_VideoMLLM_embedding_lora_HF.py")
    views_path = os.path.join(_HERE, "c02_density_views.py")
    base_sha, views_sha = sha256_of(base_path), sha256_of(views_path)
    if FROZEN_BASE_SHA256 != "PENDING_FREEZE":
        assert base_sha == FROZEN_BASE_SHA256, "DEPLOYED EXTRACTOR CHANGED -- refusing"
    if FROZEN_VIEWS_SHA256 != "PENDING_FREEZE":
        assert views_sha == FROZEN_VIEWS_SHA256, "VIEW MODULE CHANGED -- refusing"

    selftest_cases = V.self_test()
    print("[c02den] view self-test PASS: {}".format(", ".join(selftest_cases)), flush=True)

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    for s in splits:
        if s not in SPLIT_TO_OUTNAME:
            raise RuntimeError("SPLIT GUARD: '{}' is not train/val".format(s))

    # EVERY output path is validated and proven absent BEFORE the 7B model is loaded, so
    # a name collision costs seconds rather than a full GPU pass.
    planned = []
    for sp in splits:
        gtp = assert_no_test_token(
            os.path.join(a.gt_dir, a.dataset, "{}.jsonl".format(sp)))
        if not os.path.exists(gtp):
            raise RuntimeError("missing ground truth: {}".format(gtp))
        for vn in V.VIEW_NAMES:
            op = assert_no_test_token(
                view_out_path(a.EXP_FOLDER, a.dataset, SPLIT_TO_OUTNAME[sp],
                              a.base_tag, vn))
            if os.path.exists(op):
                raise RuntimeError("NO-CLOBBER: {} already exists".format(op))
            planned.append(op)
    assert_no_test_token(a.manifest)
    if os.path.exists(a.manifest):
        raise RuntimeError("NO-CLOBBER: {}".format(a.manifest))
    print("[c02den] {} output paths validated absent".format(len(planned)), flush=True)

    budget_check(a.budget_deadline_epoch, 0.0, "before model load")
    print("[c02den] budget: {:.1f}s remain before the guard deadline".format(
        budget_remaining(a.budget_deadline_epoch)), flush=True)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    device = torch.device(a.device)
    print("[c02den] loading base {}".format(a.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    if not os.path.isdir(a.lora_dir):
        raise RuntimeError("lora_dir not a directory: {}".format(a.lora_dir))
    print("[c02den] attaching + merging LoRA {}".format(a.lora_dir), flush=True)
    model = PeftModel.from_pretrained(model, a.lora_dir)
    model = model.merge_and_unload()
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(a.model, max_pixels=a.max_pixels)

    out_dir = os.path.join(a.EXP_FOLDER, a.dataset)
    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        "schema_version": "c02_density_extract_manifest_v1",
        "run_id": a.run_id,
        "dataset": a.dataset,
        "base_tag": a.base_tag,
        "lora_dir": a.lora_dir,
        "model": a.model,
        "num_frames": a.num_frames,
        "max_pixels": a.max_pixels,
        "view_names": list(V.VIEW_NAMES),
        "k_windows": V.K_WINDOWS,
        "sep": V.SEP,
        "l_max": V.L_MAX,
        "source_sha256": {"deployed_extractor": base_sha, "view_module": views_sha,
                          "this_script": sha256_of(os.path.abspath(__file__))},
        "view_self_test_cases": selftest_cases,
        "splits": {},
        "test_contact": "NONE -- splits restricted to train/val, path guard, torch.load guard",
    }

    for split in splits:
        outname = SPLIT_TO_OUTNAME[split]
        gt_path = os.path.join(a.gt_dir, a.dataset, "{}.jsonl".format(split))
        assert_no_test_token(gt_path)
        items = BASE.read_gt(gt_path)
        video_root = os.path.join(a.video_dir, a.dataset, "All")
        d_model = model.config.hidden_size

        ids, labels = [], []
        feats = {name: [] for name in V.VIEW_NAMES}
        per_item, zero_guard = [], 0
        n_forward, n_copied = 0, 0
        t0 = time.time()
        max_item_s = 0.0

        for n, item in enumerate(items):
            # Fail-closed BEFORE any work on this item.  Nothing computed is discarded:
            # no view cache or manifest for this split has been written yet, so the split
            # is simply not produced.
            budget_check(a.budget_deadline_epoch, max_item_s,
                         "{}/{} item {}/{}".format(a.dataset, split, n + 1, len(items)))
            t_item = time.time()
            vid, title = item["id"], item.get("title", "")
            text = item.get("text", "")
            views, vmeta = V.build_views(text)
            for name in V.VIEW_NAMES:
                V.assert_subsequence(text, views[name])

            frames, ok = BASE.load_video_frames(
                os.path.join(video_root, "{}.mp4".format(vid)), a.num_frames)

            if not ok:
                zero_guard += 1
                zero = torch.zeros(d_model, dtype=torch.float32)
                for name in V.VIEW_NAMES:
                    feats[name].append(zero.clone())
                vmeta["video_ok"] = False
                vmeta["n_forward"] = 0
            else:
                vmeta["video_ok"] = True
                # one forward per DISTINCT view string; identical strings share a vector
                by_string, computed = {}, 0
                for name in V.VIEW_NAMES:
                    s = views[name]
                    if s not in by_string:
                        by_string[s] = BASE._encode(
                            frames, build_text_prompt(s, title), processor, model,
                            device, a.max_pixels, span="response")
                        computed += 1
                    else:
                        n_copied += 1
                    feats[name].append(by_string[s])
                n_forward += computed
                vmeta["n_forward"] = computed

            ids.append(vid)
            labels.append(item["label"])
            vmeta["id"] = vid
            per_item.append(vmeta)
            max_item_s = max(max_item_s, time.time() - t_item)

            if (n + 1) % 25 == 0:
                el = time.time() - t0
                print("  [{}/{}] {}/{} items, {:.1f}s elapsed, {} zero-guards".format(
                    a.dataset, split, n + 1, len(items), el, zero_guard), flush=True)

        labels_t = torch.tensor([int(x) for x in labels], dtype=torch.long)
        written = {}
        for name in V.VIEW_NAMES:
            stacked = torch.stack(feats[name], dim=0).float()
            path = view_out_path(a.EXP_FOLDER, a.dataset, outname, a.base_tag, name)
            assert_no_test_token(path)
            if os.path.exists(path):
                raise RuntimeError("NO-CLOBBER: {} exists".format(path))
            torch.save({"ids": [ids], "text_feats": stacked, "labels": labels_t,
                        "c02_view": name, "c02_run_id": a.run_id,
                        "c02_base_tag": a.base_tag}, path)
            written[name] = {"path": path, "sha256": sha256_of(path),
                             "shape": list(stacked.shape)}
            print("[c02den] wrote {} {}".format(path, list(stacked.shape)), flush=True)

        n_deg = sum(1 for m in per_item if m["degenerate"] != V.DEGEN_NONE)
        n_ident = sum(1 for m in per_item if len(m["identity_views"]) ==
                      len(V.NON_NATIVE_VIEWS))
        manifest["splits"][outname] = {
            "gt_path": gt_path, "n_items": len(ids), "d_model": int(d_model),
            "slowest_item_seconds": round(max_item_s, 2),
            "budget_seconds_remaining_at_split_end": round(
                budget_remaining(a.budget_deadline_epoch), 1),
            "zero_guard_videos": zero_guard,
            "n_text_forwards": n_forward, "n_vectors_copied": n_copied,
            "n_degenerate_items": n_deg,
            "n_full_identity_orbit_items": n_ident,
            "view_support": round(1.0 - (n_ident / float(len(ids))), 6) if ids else 0.0,
            "seconds": round(time.time() - t0, 1),
            "written": written,
            "per_item": per_item,
        }
        print("[c02den] {}/{}: n={} forwards={} copied={} degenerate={} "
              "full-identity={} view_support={:.4f} in {:.1f}s".format(
                  a.dataset, outname, len(ids), n_forward, n_copied, n_deg, n_ident,
                  manifest["splits"][outname]["view_support"],
                  manifest["splits"][outname]["seconds"]), flush=True)

    manifest["job_seconds"] = round(time.time() - t_job0, 1)
    assert_no_test_token(a.manifest)
    os.makedirs(os.path.dirname(a.manifest), exist_ok=True)
    if os.path.exists(a.manifest):
        raise RuntimeError("NO-CLOBBER: {}".format(a.manifest))
    tmp = a.manifest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, a.manifest)
    print("[c02den] manifest -> {}".format(a.manifest), flush=True)


def _publish_breach(a, exc):
    """Publish what the run had completed when the budget guard fired.

    Deliberately records only ACCOUNTING, never a partial scientific artifact: the view
    caches and the manifest for the in-progress dataset were not written and will not be,
    so the downstream A0 fails closed on their absence rather than reading a truncated
    bank.
    """
    rec = {"schema_version": "c02_density_budget_breach_v1",
           "run_id": a.run_id, "dataset": a.dataset,
           "reason": str(exc),
           "deadline_epoch": float(a.budget_deadline_epoch),
           "breached_at_epoch": time.time(),
           "manifest_written": os.path.exists(a.manifest),
           "view_caches_written_for_this_dataset": sorted(
               os.path.basename(q) for sp in SPLIT_TO_OUTNAME.values()
               for q in [view_out_path(a.EXP_FOLDER, a.dataset, sp, a.base_tag, vn)
                         for vn in V.VIEW_NAMES] if os.path.exists(q)),
           "note": "no partial view cache or manifest was written for the in-progress "
                   "split; a completed dataset, if any, is intact and its manifest is "
                   "present. The A0 fails closed on a missing manifest or view cache."}
    try:
        d = os.path.dirname(a.manifest)
        if d:
            os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "BUDGET_BREACH_{}.json".format(a.dataset))
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
        print("[c02den] BUDGET BREACH record -> {}".format(path), flush=True)
    except Exception as e:  # noqa: BLE001  never mask the breach with a reporting error
        print("[c02den] BUDGET BREACH (record could not be written: {})".format(
            repr(e)), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BudgetExceeded as _exc:
        print("[c02den] HALT_C02_DEN_GPU_BUDGET: {}".format(_exc), flush=True)
        try:
            _publish_breach(parse_args(), _exc)
        except SystemExit:
            raise
        except Exception as _e:  # noqa: BLE001
            print("[c02den] breach record failed: {}".format(repr(_e)), flush=True)
        raise SystemExit(BUDGET_EXIT_CODE)
