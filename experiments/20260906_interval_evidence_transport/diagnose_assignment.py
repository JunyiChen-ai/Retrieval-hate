"""Read-only crop0 full-test activation diagnostics; no training or new AP/AUC."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import hier_evidence_common as common
import vlm_verdict
from interval_observation_data import EvalDataset
from model import Candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / 'runs'):
        parser.error('--out must be within repository runs/')
    print(f'host={socket.gethostname()}', flush=True)
    torch.set_num_threads(4)
    summary = json.loads((args.run / 'summary.json').read_text())
    if summary['ablation'] != 'full':
        parser.error('this diagnostic describes a full-model checkpoint only')
    coverage = json.loads((args.run / 'coverage.json').read_text())
    corpus = summary['corpus']
    ids = coverage['splits']['test']
    raw = {k: vlm_verdict.load_verdicts(corpus, k, video_ids=ids, strict=True) for k in [30, 4]}
    assert all(set(raw[k]) == set(ids) for k in [30, 4])
    observations = {v: np.concatenate([raw[k][v] for k in [30, 4]]) for v in ids}
    cache = common.ScaffoldCache(corpus, ids,
        lambda vid, snip, duration: np.zeros((len(snip), common.SCAF_DIM), dtype=np.float32))
    dataset = EvalDataset(corpus, ids, cache, verdicts=observations)
    model = Candidate(np.zeros(1920), np.ones(1920), summary['hparams']['dropout'])
    model.load_state_dict(torch.load(args.run / 'model.pth', map_location='cpu', weights_only=True))
    model.eval()
    capture = {}
    handles = [
        model.prior.register_forward_hook(lambda module, inputs, output: capture.update(prior=output.detach())),
        model.query.register_forward_hook(lambda module, inputs, output: capture.update(query=output.detach())),
        model.key.register_forward_hook(lambda module, inputs, output: capture.setdefault('keys', []).append(output.detach())),
    ]
    rows = []
    with torch.no_grad():
        for index, vid in enumerate(ids):
            capture.clear()
            visual, audio, _, _, returned_vid = dataset[index]
            assert vid == returned_vid
            visual, audio = visual[:1], audio[:1]  # All test videos, descriptive crop0 only.
            model(audio, visual, seq_len=None)
            bounds = audio[..., common.A_EXT_DIM:common.A_EXT_DIM + 2]
            overlap = (torch.minimum(bounds[:, None, :, 1], model.window_end[None, :, None]) -
                       torch.maximum(bounds[:, None, :, 0], model.window_start[None, :, None])).clamp_min(0)
            measure = overlap / overlap.sum(-1, keepdim=True)
            grade = audio[:, 0, common.A_EXT_DIM + 2:].long()
            log_prior = capture['prior'].log_softmax(-1)
            emission = model.emission()[model.scale_index][None]
            observed = emission.gather(-1, grade[:, :, None, None].expand(-1, -1, 4, 1)).squeeze(-1)
            posterior = (log_prior + observed).softmax(-1)
            prior = log_prior.exp()
            row = dict(video_id=vid,
                posterior_grade_probability=float(posterior.gather(-1, grade[..., None]).mean()),
                posterior_map_differs_from_grade=float((posterior.argmax(-1) != grade).float().mean()),
                posterior_vs_prior_tv=float((.5 * (posterior - prior).abs().sum(-1)).mean()),
                assignment={})
            assert len(capture['keys']) == 2
            for modality, key in zip(['audio_text', 'visual'], capture['keys']):
                affinity = capture['query'] @ key.transpose(1, 2) / (key.shape[-1] ** .5)
                log_a = (affinity + overlap.clamp_min(1e-12).log()).masked_fill(overlap <= 0, -torch.inf).log_softmax(-1)
                assignment = log_a.exp()
                kl_terms = torch.where(overlap > 0,
                    assignment * (log_a - measure.clamp_min(1e-12).log()), torch.zeros_like(assignment))
                row['assignment'][modality] = dict(
                    mean_tv_from_overlap=float((.5 * (assignment - measure).abs().sum(-1)).mean()),
                    mean_kl_from_overlap=float(kl_terms.sum(-1).mean()))
            rows.append(row)
    for handle in handles:
        handle.remove()
    keys = ['posterior_grade_probability', 'posterior_map_differs_from_grade', 'posterior_vs_prior_tv']
    means = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    means['assignment'] = {modality: {key: float(np.mean([row['assignment'][modality][key] for row in rows]))
        for key in ['mean_tv_from_overlap', 'mean_kl_from_overlap']} for modality in ['audio_text', 'visual']}
    report = dict(purpose='Full-test crop0 activation description, no GT, optimization, or metric recomputation',
        created_at=datetime.now().astimezone().isoformat(), command=[sys.executable, *sys.argv],
        code_description='2026-09-06 activation-only diagnostic using unmodified C9 forward', corpus=corpus, n_test_videos=len(rows),
        sources=[str(args.run / p) for p in ['model.pth', 'summary.json', 'coverage.json']],
        vlm_source='data/MLLM_scores/PROVENANCE.md', selected_epoch=summary['selected_epoch'],
        video_macro_means=means, per_video=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(n_test_videos=len(rows), **means)), flush=True)


if __name__ == '__main__':
    main()
