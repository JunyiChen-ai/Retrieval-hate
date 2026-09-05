#!/usr/bin/env python3
"""Full-cohort, label-free temporal-context measurements with one frozen Qwen."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import torch
from PIL import Image
from context_witness import VERSION, MODEL, ORDER, ATTRIBUTES, SYSTEM, read_measurement
from measurement_inputs import fixed_ids, window_transcripts
from utils.generate_subclip_embedding_HF import load_video_frames


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', choices=['hatemm', 'hateclipseg'], required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError('invalid shard')
    run = ROOT/args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    (run/'run.pid').write_text(str(os.getpid())+'\n')
    print(f'host={socket.gethostname()} code={VERSION}', flush=True)
    ds = {'hatemm': 'HateMM', 'hateclipseg': 'HateClipSeg'}[args.corpus]
    ids = fixed_ids(ROOT, args.corpus)[args.shard::args.shards]
    asr, missing = window_transcripts(ROOT/'data/ASR'/ds, 30, ids)
    cache = ROOT/'data/context_witness'/args.corpus
    cache.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), version=VERSION, model=MODEL, host=socket.gethostname(),
                  order=ORDER, attributes=ATTRIBUTES, system=SYSTEM, windows=30,
                  frames_per_target=4, frames_per_neighbor=2, max_pixels=151200,
                  answer_protocol='six autoregressive Yes/No lines, raw pre-processor logits',
                  expected_ids=ids, missing_asr=missing,
                  sampling='120 uniform endpoint-inclusive video frames; target4, previous last2, next first2',
                  command=' '.join(sys.argv))
    write_json(run/'config.json', config)
    (cache/f'PROVENANCE_shard{args.shard}.md').write_text(
        f'# Temporal-context observations\n\nDate: {datetime.now().isoformat()}\nHost: {socket.gethostname()}\n'
        f'Code: {VERSION}\nScript: scripts/analysis/extract_context_witness.py\nModel: {MODEL}\n'
        f'Inputs: data/video/{ds}/All; data/ASR/{ds} K30; results/reproduction/splits/{args.corpus}_*.txt (IDs only).\n'
        f'Full questions, missing transcripts, sampling and command: {run}/config.json\n'
        'No labels or GT spans consumed. Six answers are autoregressive, not independent marginal queries.\n'
        'Four modes processed as a batch by the same frozen model; no model ensemble.\n')
    (cache/'PROVENANCE.md').write_text('# Provenance\n\nSee PROVENANCE_shard*.md and the referenced full run configs.\n')
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, LogitsProcessor
    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=151200)
    processor.tokenizer.padding_side = 'left'
    tokens = [processor.tokenizer.encode(x, add_special_tokens=False) for x in ['No', 'Yes', '\n']]
    if any(len(x) != 1 for x in tokens) or len({x[0] for x in tokens}) != 3:
        raise ValueError(f'expected distinct single tokens No/Yes/newline: {tokens}')
    no, yes, newline = [x[0] for x in tokens]

    class AnswerGrammar(LogitsProcessor):
        def __init__(self, prefix):
            self.prefix = prefix

        def __call__(self, input_ids, scores):
            step = input_ids.shape[1]-self.prefix
            allowed = [no, yes] if step % 2 == 0 else [newline]
            # Do not mutate raw scores: generate.output_logits must stay raw.
            masked = torch.full_like(scores, -float('inf'))
            masked[:, allowed] = scores[:, allowed]
            return masked

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()

    @torch.inference_mode()
    def measure(frames, texts, before_available, after_available):
        prompts, videos = [], []
        for mode in ORDER:
            keep_target = mode in ['full', 'target_only']
            keep_context = mode in ['full', 'context_only']
            keep = [keep_context]*2 + [keep_target]*4 + [keep_context]*2
            supplied = [f if k else Image.new('RGB', f.size, (0, 0, 0)) for f, k in zip(frames, keep)]
            available = [keep_context and before_available, keep_target, keep_context and after_available]
            regions = []
            for name, indices, text, present in zip(['BEFORE', 'TARGET', 'AFTER'],
                    ['1-2', '3-6', '7-8'], texts, available):
                regions.append(f'{name}, frame positions {indices}: '+
                    (f'images supplied; transcript: {text.strip() or "(transcript absent)"}' if present
                     else 'content absent; black placeholder images; transcript absent'))
            query = '\n'.join(regions)+'\nAnswer these six TARGET questions in order:\n'+ '\n'.join(
                f'{i+1}. Does TARGET contain {attribute}?' for i, attribute in enumerate(ATTRIBUTES))
            messages = [dict(role='system', content=[dict(type='text', text=SYSTEM)]),
                        dict(role='user', content=[dict(type='video', video=supplied), dict(type='text', text=query)])]
            prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            videos.append(supplied)
        inputs = processor(text=prompts, videos=videos, padding=True, return_tensors='pt').to('cuda')
        prefix = inputs.input_ids.shape[1]
        output = model.generate(**inputs, max_new_tokens=11, do_sample=False, repetition_penalty=1.0,
            logits_processor=[AnswerGrammar(prefix)], output_logits=True, return_dict_in_generate=True)
        answers = output.sequences[:, prefix:]
        if answers.shape != (4, 11) or len(output.logits) != 11:
            raise RuntimeError('incomplete six-answer generation')
        if not torch.isin(answers[:, ::2], answers.new_tensor([no, yes])).all() or not (answers[:, 1::2] == newline).all():
            raise RuntimeError('invalid constrained answer format')
        pairs = torch.stack([output.logits[i][:, [no, yes]].float() for i in range(0, 11, 2)], 1)
        if not torch.isfinite(pairs).all():
            raise RuntimeError('raw logits are missing/nonfinite')
        lp = pairs.log_softmax(-1)
        return dict(log_odds=(pairs[..., 1]-pairs[..., 0]).cpu().tolist(),
                    entropy=(-(lp.exp()*lp).sum(-1)).cpu().tolist(),
                    answers=[['Yes' if token == yes else 'No' for token in row] for row in answers[:, ::2].cpu().tolist()])

    for n, vid in enumerate(ids):
        output = cache/'K30'/f'{vid}.json'
        if output.exists():
            read_measurement(output, vid)
            print(f'reused={n+1}/{len(ids)} video={vid}', flush=True)
            continue
        start = time.monotonic()
        video = ROOT/'data/video'/ds/'All'/f'{vid}.mp4'
        frames, ok = load_video_frames(str(video), 120)
        if not ok or len(frames) != 120:
            raise RuntimeError(f'video decode failed: {video}')
        windows = []
        for i in range(30):
            blank = lambda: Image.new('RGB', frames[0].size, (0, 0, 0))
            before = frames[4*i-2:4*i] if i else [blank(), blank()]
            after = frames[4*i+4:4*i+6] if i < 29 else [blank(), blank()]
            values = measure(before+frames[4*i:4*i+4]+after,
                [asr[vid][i-1] if i else '', asr[vid][i], asr[vid][i+1] if i < 29 else ''], i > 0, i < 29)
            windows.append(dict(index=i, relative_bounds=[i/30, (i+1)/30],
                                before_available=i > 0, after_available=i < 29, **values))
            print(f'{vid} window={i+1}/30 seconds={time.monotonic()-start:.1f}', flush=True)
        write_json(output, dict(version=VERSION, model=MODEL, id=vid, video=str(video),
            asr_missing=vid in missing, order=ORDER, attributes=ATTRIBUTES,
            answer_protocol=config['answer_protocol'], windows=windows))
        read_measurement(output, vid)
        print(f'completed={n+1}/{len(ids)} video={vid} seconds={time.monotonic()-start:.1f}', flush=True)
    write_json(run/'completion.json', dict(state='EXTRACTION_FINISHED', expected=len(ids),
                                          videos=len(ids), cache=str(cache)))


if __name__ == '__main__':
    main()
