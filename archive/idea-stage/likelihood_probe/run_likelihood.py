"""LIKELIHOOD PROBE -- runner.

For every (item, variant, pair) it computes the log-likelihood of the ENDORSING and of the
OPPOSING continuation under a local Qwen VL model, given the same context (frames + full
transcript). The model generates nothing: a single teacher-forced forward pass per
continuation, logits read only at the continuation positions.

  python run_likelihood.py --arm A1 --set eval
  python run_likelihood.py --arm A1 --set ctrl
  python run_likelihood.py --arm A1 --set smoke      # 3 TRAIN videos, prints raw logprobs only

Arms (frozen):
  A1  Qwen/Qwen2.5-VL-7B-Instruct   plain-text continuation      <- primary readout
  A2  Qwen/Qwen2.5-VL-7B-Instruct   chat-template continuation
  B1  Qwen/Qwen2-VL-7B-Instruct     plain-text continuation
  C1  Qwen/Qwen2-VL-7B  (base)      plain-text continuation      <- B1 - C1 = tuning contrast
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lp_common import (ROOT, MAX_PIXELS, CONT_PREFIX, context_text, ctrl_items,  # noqa: E402
                       ctrl_texts, eval_items, frames_of, load_texts, plan)

ARMS = {
    "A1": {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "fmt": "plain"},
    "A2": {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "fmt": "chat"},
    "B1": {"model": "Qwen/Qwen2-VL-7B-Instruct", "fmt": "plain"},
    "C1": {"model": "Qwen/Qwen2-VL-7B", "fmt": "plain"},
}
VIS = "<|vision_start|><|image_pad|><|vision_end|>"


class _NoLMHead(torch.nn.Module):
    """Stops the (seq x 152k) fp32 logits from ever being materialised. The real lm_head is
    applied by hand to the ~40 continuation positions we actually need."""

    def forward(self, x):
        return x[..., :1]


def free_vram_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used",
         "--format=csv,noheader,nounits"]).decode().strip().split("\n")[0]
    tot, used = [int(x) for x in out.split(",")]
    return tot - used


def load(arm, cpu_offload_gib=0):
    from transformers import AutoConfig, AutoProcessor
    mid = ARMS[arm]["model"]
    cfg = AutoConfig.from_pretrained(mid)
    if "qwen2_5_vl" in getattr(cfg, "model_type", ""):
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    else:
        from transformers import Qwen2VLForConditionalGeneration as Cls
    proc = AutoProcessor.from_pretrained(mid, max_pixels=MAX_PIXELS)
    kw = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if cpu_offload_gib:
        free = max(4, int(free_vram_mib() / 1024) - 3)
        kw.update(device_map="auto",
                  max_memory={0: f"{free}GiB", "cpu": f"{cpu_offload_gib}GiB"})
    else:
        kw.update(device_map={"": 0})
    model = Cls.from_pretrained(mid, **kw)
    model.eval()
    real_head = model.lm_head
    model.lm_head = _NoLMHead()
    return model, proc, real_head


def build_ctx(arm, proc, ds, frames, transcript):
    """Returns the processor output for the CONTEXT ONLY (frames + transcript + lead-in +
    'Pinned comment:'); the two continuations are appended as raw token ids afterwards."""
    body = context_text(ds, len(frames), transcript)
    if ARMS[arm]["fmt"] == "chat":
        content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": body}]
        text = proc.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=False, add_generation_prompt=True)
        text = text + CONT_PREFIX
    else:
        text = VIS * len(frames) + body + CONT_PREFIX
    return proc(text=[text], images=(frames or None), return_tensors="pt")


@torch.no_grad()
def score_one(model, proc, real_head, inputs, cont_ids_list, device):
    """Mean and sum token log-prob of each continuation, teacher-forced, sharing one context."""
    out = []
    ctx = inputs["input_ids"][0]
    for cont in cont_ids_list:
        ids = torch.cat([ctx, cont]).unsqueeze(0).to(device)
        kw = {"input_ids": ids,
              "attention_mask": torch.ones_like(ids)}
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs:
                v = inputs[k]
                kw[k] = v.to(device=device, dtype=model.dtype) if k == "pixel_values" \
                    else v.to(device)
        cap = {}

        def hook(_m, _i, o):
            cap["h"] = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]

        h = model.model.register_forward_hook(hook)
        try:
            model(**kw, use_cache=False, output_hidden_states=False)
        finally:
            h.remove()
        n_ctx, n_c = ctx.numel(), cont.numel()
        hs = cap["h"][0][n_ctx - 1: n_ctx + n_c - 1]              # predicts cont tokens
        logits = real_head(hs.to(real_head.weight.dtype)).float()
        lp = torch.log_softmax(logits, dim=-1)
        tok = cont.to(lp.device)
        vals = lp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        out.append({"n_tok": int(n_c), "sum_lp": float(vals.sum()),
                    "mean_lp": float(vals.mean()), "n_ctx": int(n_ctx)})
        del cap, hs, logits, lp
    return out


def run(a):
    dev = "cuda"
    if a.set == "eval":
        items, texts_fn, out_p = eval_items(), load_texts, f"lp_{a.arm}_eval.jsonl"
    elif a.set == "ctrl":
        items, texts_fn, out_p = ctrl_items(), ctrl_texts, f"lp_{a.arm}_ctrl.jsonl"
    else:
        items, texts_fn, out_p = ctrl_items()[:3], ctrl_texts, f"lp_{a.arm}_smoke.jsonl"
    reqs = plan(items, texts_fn)
    out_p = os.path.join(HERE, out_p)
    done = set()
    if os.path.exists(out_p) and not a.force:
        for line in open(out_p, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["dataset"], r["id"], r["variant"], r["pair"]))
    print(f"[{a.arm}/{a.set}] {len(reqs)} comparisons over {len(items)} items "
          f"({len(done)} cached)", flush=True)

    model, proc, real_head = load(a.arm, a.cpu_offload_gib)
    tok = proc.tokenizer
    print(f"[{a.arm}] loaded {ARMS[a.arm]['model']} fmt={ARMS[a.arm]['fmt']} "
          f"dev={next(model.parameters()).device}", flush=True)

    f = open(out_p, "a", encoding="utf-8")
    t0 = time.time()
    cur_key, ctx_inputs, cur_frames = None, None, None
    n_done = 0
    for i, r in enumerate(reqs):
        key = (r["dataset"], r["id"], r["variant"], r["pair"])
        if key in done:
            continue
        ik = (r["dataset"], r["id"])
        if ik != cur_key:
            cur_frames = frames_of(r["dataset"], r["id"])
            ctx_inputs = build_ctx(a.arm, proc, r["dataset"], cur_frames, r["transcript"])
            cur_key = ik
        conts = []
        for side in ("endorse_text", "oppose_text"):
            s = " " + r[side]
            conts.append(torch.tensor(tok(s, add_special_tokens=False)["input_ids"],
                                      dtype=torch.long))
        try:
            res = score_one(model, proc, real_head, ctx_inputs, conts, dev)
            err = None
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            res, err = None, f"OOM:{str(e)[:80]}"
        rec = {k: r[k] for k in ("dataset", "id", "group", "variant", "pair", "lang", "target")}
        rec.update({"arm": a.arm, "model": ARMS[a.arm]["model"], "fmt": ARMS[a.arm]["fmt"],
                    "n_frames": len(cur_frames),
                    "endorse_text": r["endorse_text"], "oppose_text": r["oppose_text"],
                    "endorse": res[0] if res else None, "oppose": res[1] if res else None,
                    "error": err})
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        n_done += 1
        if n_done % 25 == 0:
            el = time.time() - t0
            print(f"[{a.arm}/{a.set}] {i+1}/{len(reqs)} done={n_done} "
                  f"{el:.0f}s ({el/n_done:.2f}s/cmp) ctx={rec['endorse']['n_ctx'] if res else -1}",
                  flush=True)
        if a.set == "smoke":
            print(f"[smoke] {r['id']} {r['variant']} p{r['pair']} {r['lang']} "
                  f"E={res[0]['mean_lp']:.4f}({res[0]['n_tok']}t) "
                  f"O={res[1]['mean_lp']:.4f}({res[1]['n_tok']}t)", flush=True)
    f.close()
    print(f"[{a.arm}/{a.set}] wrote {out_p}; {n_done} new in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--set", required=True, choices=["eval", "ctrl", "smoke"])
    ap.add_argument("--cpu_offload_gib", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
