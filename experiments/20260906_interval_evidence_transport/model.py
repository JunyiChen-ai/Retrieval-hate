"""One ordinal-observation / interval-assignment / conditional-readout network."""
import math

import torch
from torch import nn
from torch.nn import functional as F

import hier_evidence_common as common


class Candidate(nn.Module):
    def __init__(self, mean, std, dropout, arm='full'):
        super().__init__()
        self.arm = arm
        hidden = 128
        self.register_buffer('content_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('content_std', torch.as_tensor(std, dtype=torch.float32))
        starts = [j / k for k in [30, 4] for j in range(k)]
        ends = [(j + 1) / k for k in [30, 4] for j in range(k)]
        self.register_buffer('window_start', torch.tensor(starts))
        self.register_buffer('window_end', torch.tensor(ends))
        self.register_buffer('scale_index', torch.tensor([0] * 30 + [1] * 4))
        distance = (torch.arange(4)[:, None] - torch.arange(4)[None, :]).float().square()
        self.register_buffer('ordinal_distance', distance)
        self.visual_proj = nn.Sequential(nn.Linear(1024, hidden), nn.GELU(), nn.Dropout(dropout))
        self.audio_proj = nn.Sequential(nn.Linear(896, hidden), nn.GELU(), nn.Dropout(dropout))
        self.prior = nn.Linear(2 * hidden, 4)
        self.noise_precision = nn.Parameter(torch.full((2,), math.log(math.expm1(1.))))
        self.categorical_emission = nn.Parameter(-distance[None].repeat(2, 1, 1))
        self.state_embedding = nn.Embedding(4, hidden)
        self.scale_embedding = nn.Embedding(2, hidden)
        self.query = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.LayerNorm(hidden))
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.message = nn.Linear(hidden, hidden)
        self.update_norm = nn.LayerNorm(hidden)
        self.update_ffn = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.output_norm = nn.LayerNorm(hidden)
        self.interval_update = nn.Linear(2 * hidden, hidden)
        self.interval_norm = nn.LayerNorm(hidden)
        # Exact factorization of a concatenated 512->128 affine map.
        self.read_content = nn.Linear(3 * hidden, hidden, bias=False)
        self.read_interval = nn.Linear(hidden, hidden)
        self.read_output = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.add_content = nn.Linear(3 * hidden, 1)
        self.add_interval = nn.Linear(hidden, 1, bias=False)

    def emission(self):
        logits = (self.categorical_emission if self.arm == 'categorical_noise' else
                  -F.softplus(self.noise_precision)[:, None, None] * self.ordinal_distance)
        return F.log_softmax(logits, -1)

    def _update(self, value, other_message):
        value = self.update_norm(value + self.message(other_message))
        return self.output_norm(value + self.update_ffn(value))

    def forward(self, audio, visual, seq_len=None):
        batch, steps = visual.shape[:2]
        bounds = audio[..., common.A_EXT_DIM:common.A_EXT_DIM + 2]
        valid = bounds[..., 1] > bounds[..., 0]
        if seq_len is not None:
            valid = valid & (torch.arange(steps, device=visual.device)[None] < seq_len.to(visual.device)[:, None])
        overlap = (torch.minimum(bounds[:, None, :, 1], self.window_end[None, :, None]) -
                   torch.maximum(bounds[:, None, :, 0], self.window_start[None, :, None])).clamp_min(0)
        overlap = overlap * valid[:, None]
        measure = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-12)
        content = torch.cat([visual, audio[..., :common.SCAF_OFFSET]], -1)
        content = (content - self.content_mean) / self.content_std
        v = self.visual_proj(content[..., :1024]) * valid[..., None]
        a = self.audio_proj(content[..., 1024:]) * valid[..., None]
        pooled_a, pooled_v = measure @ a, measure @ v
        prior_log = F.log_softmax(self.prior(torch.cat([pooled_a, pooled_v], -1)), -1)
        if self.arm == 'no_vlm':
            posterior = prior_log.exp()
            self.observation_loss = prior_log.sum() * 0
        else:
            grade = audio[:, 0, common.A_EXT_DIM + 2:].long()
            log_e = self.emission()[self.scale_index][None].expand(batch, -1, -1, -1)
            observed = log_e.gather(-1, grade[:, :, None, None].expand(-1, -1, 4, 1)).squeeze(-1)
            joint = prior_log + observed
            nll = -torch.logsumexp(joint, -1)
            self.observation_loss = .5 * (nll[:, :30].mean() + nll[:, 30:].mean())
            posterior = (F.one_hot(grade, 4).to(a.dtype) if self.arm == 'hard_observation'
                         else F.softmax(joint, -1))
        evidence = posterior @ self.state_embedding.weight + self.scale_embedding(self.scale_index)[None]
        q = self.query(torch.cat([pooled_a, pooled_v, evidence], -1))

        def assignment(value):
            if self.arm == 'uniform_assignment':
                return measure
            affinity = (q @ self.key(value).transpose(1, 2)) / math.sqrt(q.shape[-1])
            affinity = affinity + overlap.clamp_min(1e-12).log()
            return F.softmax(affinity.masked_fill(overlap <= 0, -torch.inf), -1)

        aa, av = assignment(a), assignment(v)
        ca, cv = aa @ a, av @ v
        ra = aa / aa.sum(1, keepdim=True).clamp_min(1e-12)
        rv = av / av.sum(1, keepdim=True).clamp_min(1e-12)
        a, v = self._update(a, ra.transpose(1, 2) @ cv), self._update(v, rv.transpose(1, 2) @ ca)
        a, v = a * valid[..., None], v * valid[..., None]
        interval = self.interval_norm(q + self.interval_update(torch.cat([ca, cv], -1)))
        r = .5 * (aa + av)
        r = r / r.sum(1, keepdim=True).clamp_min(1e-12)
        local = torch.cat([a, v, a * v], -1)
        if self.arm == 'additive_readout':
            logits = self.add_content(local).squeeze(-1) + (r * self.add_interval(interval)).sum(1)
            probability = logits.sigmoid() * valid
        else:
            projected_local = self.read_content(local)
            projected_interval = self.read_interval(interval)
            probability = visual.new_zeros((batch, steps))
            # First-layer content/interval projections are reused, not recomputed per pair.
            for j in range(34):
                conditional = self.read_output(projected_local + projected_interval[:, j, None]).squeeze(-1).sigmoid()
                probability = probability + r[:, j] * conditional
        bag = torch.stack([probability[b, valid[b]].topk(max(1, math.ceil(int(valid[b].sum()) / 16))).values.mean()
                           for b in range(batch)])
        logits = torch.logit(probability.clamp(1e-6, 1 - 1e-6))[..., None]
        return bag, logits, logits, logits, v, a

    def loss(self, output, label):
        video = F.binary_cross_entropy(output[0].clamp(1e-6, 1 - 1e-6), label)
        return video if self.arm == 'no_observation_loss' else video + self.observation_loss
