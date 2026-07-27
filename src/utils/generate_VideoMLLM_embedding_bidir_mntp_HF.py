"""S2a: bidirectional attention + published McGill MNTP LoRA TRANSPLANT, NO training.

MNTP staged plan stage S2a (refine-logs/MNTP_S1_RECORD.md §6d, declared before this file was
written). Thin fork in the F72 artifact-A2 pattern: imports read_gt / process_split /
SPLIT_TO_OUTNAME / parse_args_sys VERBATIM from the causal extractor and re-implements only
main(). **The readout is the DEPLOYED one, untouched** -- S1/S1b closed the readout route, so the
crater-comparable readout is the deployed EOS-class tail, which also keeps the frozen KS-MNTP-1
bars valid without reinterpretation. This file therefore adds exactly ONE thing to the F72 bidir
runner: a second adapter load + merge.

ARM (order DECLARED in §6d.3 = the CHEAP order, recon §4.2):
    base Qwen2.5-VL-7B -> PeftModel(task LoRA) -> merge_and_unload   [unchanged, all arms]
                       -> PeftModel(MNTP LoRA) -> merge_and_unload   [S2a ONLY]
                       -> apply_bidir_mask(model)                    [POST-merge, PRE-forward]
                       -> process_split(...)                         [imported VERBATIM]

WHAT THIS TESTS. S1/S1b showed the F72 crater is not a readout problem: under bidirectional
attention the vision/text mixture happens INSIDE attention, before any pooling. The surviving
hypothesis is the actual MNTP claim -- that bidirectional attention needs WEIGHT ADAPTATION. S2a
tests it at zero training cost.

THE TRANSPLANT RISK, STATED UP FRONT. The adapter is a low-rank delta fitted to
Qwen2.5-7B-Instruct's weight point. Qwen2.5-VL's trunk was initialised from Qwen2.5 and then
further trained during VL pretraining, so the base has drifted. The delta may carry the generic
"use bidirectional context" adaptation, or it may be noise at the new weight point.

BELTS IMPLEMENTED HERE (§6d.4):
  * WEIGHT-LEVEL TRANSPLANT PROOF (belt 1): decoder weights are snapshotted immediately before the
    MNTP merge and re-read after; they MUST change. This proves the transplant landed AT THE
    WEIGHTS, independently of any downstream metric, and catches a PEFT key-mismatch (the adapter
    was trained on Qwen2ForCausalLM, whose module paths must line up with the VL decoder's)
    directly rather than by inference. Aborts on no-change.
  * --nullop_lora: replaces the MNTP adapter with a FRESHLY-INITIALISED LoRA of identical shape.
    PEFT initialises lora_B to zeros, so this is a mathematical null-op and ONLY the merge PATH
    differs. Combined with --no_bidir this produces the same-path double-merge causal floor probe
    (belt 4 / recon §4.3), which guards against the F87 bf16 merge-drift artifact.
  * --no_bidir: skip the mask flip (used only by the floor probe above).

Usage:
  # S2a arm
  python src/utils/generate_VideoMLLM_embedding_bidir_mntp_HF.py \
      --dataset HateMM --lora_dir logging/lora/HateMM_curric --splits train,val \
      --mntp_dir <snapshot> --out_model_tag Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-mntp_HF
  # same-path double-merge causal floor probe (belt 4)
  python src/utils/generate_VideoMLLM_embedding_bidir_mntp_HF.py \
      --dataset HateMM --lora_dir logging/lora/HateMM_curric --splits train,val --limit 60 \
      --nullop_lora --no_bidir --out_model_tag Qwen2.5-VL-7B-Instruct-LoRA-curric-nullop2merge_HF
"""

import os
import sys

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

# Layers sampled for the weight-level transplant proof (belt 1). Spread across depth so a
# partial key match cannot pass by touching only early or only late layers.
_PROBE_SUFFIXES = [
    "layers.0.self_attn.q_proj.weight",
    "layers.13.mlp.down_proj.weight",
    "layers.27.self_attn.o_proj.weight",
    "layers.7.mlp.gate_proj.weight",
    "layers.21.self_attn.v_proj.weight",
]


def _snapshot_weights(model):
    out = {}
    for name, p in model.named_parameters():
        for suf in _PROBE_SUFFIXES:
            if name.endswith(suf):
                out[suf] = p.detach().clone()
    return out


def _assert_weights_changed(before, model, what):
    """Belt 1: prove the adapter landed AT THE WEIGHTS, not just that a file loaded.

    ALL probes must be found and ALL must change. Requiring only "at least one" would let a
    PARTIAL key match pass -- e.g. an adapter that binds early layers but not late ones.
    """
    after = _snapshot_weights(model)
    if len(before) != len(_PROBE_SUFFIXES):
        raise RuntimeError(
            "weight probe found {}/{} tensors BEFORE merge; the probe suffixes do not match this "
            "model's naming, so belt 1 cannot certify anything.".format(
                len(before), len(_PROBE_SUFFIXES))
        )
    missing = [k for k in before if k not in after]
    if missing:
        raise RuntimeError("weight probe lost tensors after merge: {}".format(missing))
    report, n_changed = [], 0
    for k, v0 in before.items():
        v1 = after[k]
        d = (v1.float() - v0.float()).abs()
        rel = float(d.max()) / (float(v0.float().abs().max()) + 1e-12)
        changed = float(d.max()) > 0.0
        n_changed += bool(changed)
        report.append("    {:44s} max|delta|={:.3e} rel={:.3e} changed={}".format(
            k, float(d.max()), rel, changed))
    print("[S2A] weight-level transplant proof ({}):".format(what), flush=True)
    for r in report:
        print(r, flush=True)
    print("[S2A]   {}/{} probed tensors changed".format(n_changed, len(before)), flush=True)
    return n_changed, len(before)


def _attach_second_adapter(model, mntp_dir, nullop):
    """Attach the second adapter to the BARE DECODER (`model.model`), then merge it back.

    TWO reasons this must bind `model.model` and not the outer
    `Qwen2_5_VLForConditionalGeneration` -- both verified empirically on this checkpoint:

    1. KEY ALIGNMENT. The McGill adapter was trained on a text-only Qwen2 stack and its saved
       keys are `base_model.model.layers.N...`. Wrapping the OUTER VL model yields PEFT keys
       `base_model.model.model.layers.N...` (one extra `.model`, because the outer model's
       decoder is at `.model`). PEFT loads non-strictly, so the mismatch does NOT raise -- it
       warns and leaves every `lora_B` at its zero init, i.e. the transplant silently becomes a
       no-op and the arm degenerates into a duplicate of F72. Binding `model.model` makes the
       decoder's own paths `layers.N...`, which matches the checkpoint exactly.
    2. SCOPE. Suffix-matching the 7 target names on the OUTER model hits 292 modules, 96 of them
       inside the VISION TOWER (`visual.blocks.N.mlp.{gate,up,down}_proj`) -- an undeclared change
       to a tower our SFT freezes. On `model.model` it hits exactly 196 = 28 layers x 7 modules,
       with zero vision modules.
    """
    from peft import PeftModel

    decoder = model.model  # Qwen2_5_VLModel; `model.visual` is a SIBLING and stays untouched
    if nullop:
        peft_dec = build_nullop_lora(decoder)
    else:
        print("Attaching MNTP LoRA transplant from: {}".format(mntp_dir), flush=True)
        peft_dec = PeftModel.from_pretrained(decoder, mntp_dir)

    # --- Scope + load verification, before the merge erases the evidence ---
    lora_mods, nonzero_B, vision_hits = 0, 0, 0
    for name, mod in peft_dec.named_modules():
        B = getattr(mod, "lora_B", None)
        if B is None or not hasattr(B, "keys"):
            continue
        lora_mods += 1
        if "visual" in name:
            vision_hits += 1
        for k in B.keys():
            if float(B[k].weight.detach().abs().max()) > 0.0:
                nonzero_B += 1
                break
    print("[S2A] adapter scope: {} LoRA-wrapped modules (expect 196 = 28x7), vision-tower "
          "modules wrapped: {} (expect 0); modules with NON-ZERO lora_B: {}".format(
              lora_mods, vision_hits, nonzero_B), flush=True)
    if vision_hits:
        raise RuntimeError(
            "adapter bound {} VISION-TOWER modules; scope violation (the SFT freezes the vision "
            "tower and §6d declares an LLM-trunk-only transplant).".format(vision_hits))
    if lora_mods != 196:
        raise RuntimeError(
            "adapter wrapped {} modules, expected 196 (28 layers x 7 projections).".format(
                lora_mods))
    if not nullop:
        # A freshly-INJECTED LoRA has lora_B == 0 by PEFT's init. A CORRECTLY LOADED trained
        # adapter has non-zero lora_B. So "all zero" is the exact signature of a silent key
        # mismatch -- the decisive check, and it needs no warning-parsing.
        if nonzero_B != 196:
            raise RuntimeError(
                "MNTP checkpoint did NOT load: only {}/196 modules have non-zero lora_B. PEFT "
                "loads non-strictly, so a module-path mismatch leaves lora_B at its zero init "
                "and the transplant becomes a silent no-op. ABORTING before any GPU forward."
                .format(nonzero_B))
    elif nonzero_B != 0:
        raise RuntimeError(
            "null-op arm expected ALL lora_B to be zero (PEFT zero-init) but {} were non-zero; "
            "the floor probe would not be a mathematical null-op.".format(nonzero_B))

    print("Merging second adapter (merge_and_unload) ...", flush=True)
    model.model = peft_dec.merge_and_unload()
    return model


def build_nullop_lora(model):
    """A freshly-initialised LoRA of the SAME shape as the MNTP adapter.

    PEFT initialises `lora_B` to zeros, so B@A == 0 and this adapter is a MATHEMATICAL null-op:
    merging it changes the weights by exactly zero in exact arithmetic. What it does NOT leave
    unchanged is the merge PATH -- the bf16 round-trip through merge_and_unload. That is precisely
    what belt 4 measures (recon §4.3, the F87 failure shape).
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=None,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    print("[S2A] attaching FRESH zero-init LoRA (r=16, alpha=32, same 7 targets) as a "
          "mathematical null-op; only the merge PATH differs.", flush=True)
    return get_peft_model(model, cfg)


def main(args):
    # --- Guards first, before any filesystem or GPU side effect (explicit raises, -O safe).
    if "mntp" not in args.out_model_tag and "nullop" not in args.out_model_tag:
        raise RuntimeError(
            "--out_model_tag must contain 'mntp' or 'nullop' (got {!r}); refusing to write a "
            "cache whose tag does not identify this arm.".format(args.out_model_tag)
        )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if any(s == "test" or SPLIT_TO_OUTNAME.get(s, "").startswith("test") for s in splits):
        raise RuntimeError(
            "S2a is dev-only (ZERO test-touch): --splits must not contain 'test' "
            "(got {!r})".format(args.splits)
        )
    for flag in ("no_merge", "moka"):
        if getattr(args, flag, False):
            raise RuntimeError(
                "--{} is not supported by the S2a arm: this fork always takes the deployed "
                "merge_and_unload path.".format(flag.replace("_", "-"))
            )
    nullop = bool(getattr(args, "nullop_lora", False))
    use_bidir = not bool(getattr(args, "no_bidir", False))
    mntp_dir = (getattr(args, "mntp_dir", "") or "").strip()
    if not nullop and not mntp_dir:
        raise RuntimeError("--mntp_dir is required unless --nullop_lora is given")
    if nullop and mntp_dir:
        raise RuntimeError("--nullop_lora and --mntp_dir are mutually exclusive")
    if not nullop and not os.path.isdir(mntp_dir):
        raise FileNotFoundError("--mntp_dir '{}' is not a directory".format(mntp_dir))
    # §6d declares exactly TWO configurations. Anything else is an undeclared arm, and the
    # out-tag must match the configuration so a cache can never be banked under a misleading
    # identity (e.g. a real MNTP run written under a '-nullop' tag).
    tag = args.out_model_tag
    if nullop:
        if use_bidir:
            raise RuntimeError(
                "undeclared combination: the null-op arm is the SAME-PATH CAUSAL floor probe and "
                "requires --no_bidir (§6d.4 belt 4)."
            )
        if "nullop" not in tag or "mntp" in tag:
            raise RuntimeError(
                "null-op floor arm must carry 'nullop' (and not 'mntp') in --out_model_tag; "
                "got {!r}.".format(tag))
    else:
        if not use_bidir:
            raise RuntimeError(
                "undeclared combination: the MNTP arm is declared WITH bidirectional attention "
                "(§6d.3); --no_bidir is only for the null-op floor probe."
            )
        if "mntp" not in tag or "nullop" in tag:
            raise RuntimeError(
                "MNTP arm must carry 'mntp' (and not 'nullop') in --out_model_tag; "
                "got {!r}.".format(tag))

    device = torch.device(args.device)
    out_dir = os.path.join(args.EXP_FOLDER, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("[S2A] arm: task-LoRA merge -> {} merge -> bidir {} -> DEPLOYED readout (unchanged)".format(
        "ZERO-INIT null-op LoRA" if nullop else "MNTP LoRA", "ON" if use_bidir else "OFF"),
        flush=True)

    print("Loading Qwen2.5-VL base model: {}".format(args.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # REQUIRED: apply_bidir_mask asserts sdpa (flash trap)
        device_map=None,
    )

    # ---- MERGE 1: the deployed task LoRA (unchanged, all arms) ----
    lora_dir = args.lora_dir.strip() if args.lora_dir else ""
    if lora_dir:
        if not os.path.isdir(lora_dir):
            raise FileNotFoundError("--lora_dir '{}' is not a directory".format(lora_dir))
        from peft import PeftModel

        print("Attaching LoRA adapter from: {}".format(lora_dir), flush=True)
        model = PeftModel.from_pretrained(model, lora_dir)
        print("Merging LoRA adapter into base weights (merge_and_unload) ...", flush=True)
        model = model.merge_and_unload()
    else:
        print("No --lora_dir given; using FROZEN base model.", flush=True)

    # ---- MERGE 2: the MNTP transplant (or the zero-init null-op for the floor probe) ----
    before = _snapshot_weights(model)
    model = _attach_second_adapter(model, mntp_dir, nullop)

    # BELT 1 — weight-level transplant proof.
    n_changed, n_probe = _assert_weights_changed(before, model, "nullop" if nullop else "MNTP")
    if nullop:
        # A zero-init LoRA is a mathematical null-op; any change here is pure merge-path drift,
        # which is exactly what belt 4 is quantifying. Report, never abort.
        print("[S2A] (null-op arm: {}/{} tensors moved => that movement IS the bf16 merge-path "
              "drift belt 4 measures)".format(n_changed, n_probe), flush=True)
    elif n_changed != n_probe:
        # §6d.4 belt 1 says the sampled tensors MUST change -- ALL of them. Accepting a partial
        # count would let a partially-applied delta through. A whole trained matrix delta
        # rounding away in bf16 is vanishingly unlikely, so if this ever fires it is a real
        # defect and must be investigated, not waved past.
        raise RuntimeError(
            "TRANSPLANT DID NOT FULLY LAND: only {}/{} probed decoder tensors changed after "
            "merging the MNTP adapter (§6d.4 belt 1 requires ALL). A zero count means PEFT "
            "matched no target modules (module-path mismatch between the Qwen2ForCausalLM the "
            "adapter was trained on and the VL decoder); a partial count means the delta applied "
            "unevenly across depth. Either way this arm cannot be interpreted. ABORTING before "
            "any GPU forward.".format(n_changed, n_probe)
        )

    if use_bidir:
        apply_bidir_mask(model)
    else:
        print("[S2A] --no_bidir: attention left CAUSAL (same-path floor probe).", flush=True)

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
        print("Processing split '{}' ({} items) -> outname '{}'".format(
            split, len(items), outname), flush=True)

        ids, img_feats, text_feats, labels_t, dv, dt, zero_guard = process_split(
            items, split, args, processor, model, device
        )

        save_obj = {"ids": [ids], "img_feats": img_feats,
                    "text_feats": text_feats, "labels": labels_t}
        out_path = os.path.join(out_dir, "{}_{}.pt".format(outname, args.out_model_tag))
        torch.save(save_obj, out_path)
        print("Saved '{}': N={}, Dv={}, Dt={}, zero-vector videos={} -> {}".format(
            outname, len(ids), dv, dt, zero_guard, out_path), flush=True)


def parse_args_s2a():
    """Deployed parser + the three S2a-only flags."""
    argv = list(sys.argv[1:])
    extra = {"mntp_dir": "", "nullop_lora": False, "no_bidir": False}
    keep = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--mntp_dir" or a.startswith("--mntp_dir="):
            if "=" in a:
                extra["mntp_dir"] = a.split("=", 1)[1]
                i += 1
            else:
                # Guard both ends: a trailing --mntp_dir would IndexError, and
                # `--mntp_dir --nullop_lora` would silently swallow the next flag as a path.
                if i + 1 >= len(argv):
                    raise RuntimeError("--mntp_dir requires a value")
                if argv[i + 1].startswith("--"):
                    raise RuntimeError(
                        "--mntp_dir was followed by {!r}, which looks like a flag, not a "
                        "path".format(argv[i + 1]))
                extra["mntp_dir"] = argv[i + 1]
                i += 2
            continue
        if a == "--nullop_lora":
            extra["nullop_lora"] = True; i += 1; continue
        if a == "--no_bidir":
            extra["no_bidir"] = True; i += 1; continue
        keep.append(a); i += 1
    args = parse_args_sys(keep)
    for k, v in extra.items():
        setattr(args, k, v)
    return args


if __name__ == "__main__":
    args = parse_args_s2a()
    print(args)
    main(args)
