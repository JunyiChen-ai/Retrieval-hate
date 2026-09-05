"""Archived content model: normalized-time intensity and noisy aggregate supervision."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from hier_evidence_common import SCAF_OFFSET, A_EXT_DIM


class ResidualTemporal(nn.Module):
    def __init__(self, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(128, 128, 3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        return (x+self.dropout(F.gelu(self.conv(x.transpose(1, 2)).transpose(1, 2))))*mask[..., None]


class Candidate(nn.Module):
    def __init__(self, mean, std, false_positive, dropout=.2, arm='full'):
        super().__init__()
        self.arm = arm
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        self.project = nn.Linear(1920, 128)
        self.blocks = nn.ModuleList([ResidualTemporal(d, dropout) for d in [1, 2]])
        self.allocation = nn.Linear(128, 1)
        self.total = nn.Linear(128, 1)
        r = torch.as_tensor(false_positive, dtype=torch.float32).clamp(1e-6, 1-1e-6)
        self.noise_logit = nn.Parameter(torch.logit(r))
        self.sensitivity_gap = nn.Parameter(torch.full((2,), math.log(.9/.1)))

    def forward(self, audio, visual, seq_len=None):
        x = torch.cat([visual, audio[..., :SCAF_OFFSET]], -1)
        x = (x-self.mean)/self.std
        lengths = (torch.full((x.shape[0],), x.shape[1], device=x.device, dtype=torch.long)
                   if seq_len is None else torch.as_tensor(seq_len, device=x.device, dtype=torch.long))
        mask = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        # Only these two metadata columns are consumed; all teacher labels are
        # beyond this slice and are never features of the forward model.
        cells = audio[..., A_EXT_DIM:A_EXT_DIM+2]
        widths = (cells[..., 1]-cells[..., 0]).clamp_min(0)*mask
        valid = mask & (widths > 0)
        if not valid.any(1).all():
            raise ValueError('video without an integration interval')
        h = F.gelu(self.project(x))*mask[..., None]
        for block in self.blocks:
            h = block(h, mask)
        local = self.allocation(h).squeeze(-1)
        if self.arm == 'unfactorized':
            rate = F.softplus(local).clamp_min(1e-8)*valid
        else:
            pooled = (h*widths[..., None]).sum(1)/widths.sum(1, keepdim=True).clamp_min(1e-8)
            total = F.softplus(self.total(pooled).squeeze(-1)).clamp_min(1e-8)
            log_normalizer = torch.logsumexp((local+widths.clamp_min(1e-30).log()).masked_fill(~valid, -torch.inf), 1)
            log_rate = total.log()[:, None]+local-log_normalizer[:, None]
            rate = log_rate.exp()*valid
        mass = rate*widths
        logit = rate.clamp_min(1e-8).log().unsqueeze(-1)
        self._state = rate, mass, cells, valid
        video = -torch.expm1(-mass.sum(1))
        return video, logit, logit, logit, h, h

    def loss(self, labels, audio):
        rate, mass, cells, valid = self._state
        total = mass.sum(1).clamp_min(1e-8)
        # Exact stable Bernoulli NLL for a censored Poisson event.
        loss = (-labels*torch.log(-torch.expm1(-total))+(1-labels)*total).mean()
        if self.arm == 'no_vlm':
            return loss
        targets = audio[:, 0, A_EXT_DIM+2:A_EXT_DIM+36]
        offset = 0
        for j, k in enumerate([30, 4]):
            if self.arm == 'fine_only' and k == 4:
                break
            edges = torch.arange(k+1, device=rate.device, dtype=rate.dtype)/k
            overlap = (torch.minimum(cells[:, None, :, 1], edges[None, 1:, None])-
                       torch.maximum(cells[:, None, :, 0], edges[None, :-1, None])).clamp_min(0)*valid[:, None]
            if self.arm == 'topk_event':
                local = rate/(1+rate)
                member = overlap > 0
                counts = member.sum(-1)
                n_top = torch.div(counts+15, 16, rounding_mode='floor').clamp_min(1)
                sorted_scores = local[:, None].expand(-1, k, -1).masked_fill(~member, -torch.inf).sort(-1, descending=True).values
                selected = torch.arange(rate.shape[1], device=rate.device)[None, None] < n_top[..., None]
                event = torch.where(selected, sorted_scores, torch.zeros_like(sorted_scores)).sum(-1)/n_top
            else:
                window_mass = (overlap*rate[:, None]).sum(-1)
                event = -torch.expm1(-window_mass)
            if self.arm == 'hard_observation':
                observed = event
            else:
                r = self.noise_logit[j].sigmoid()
                q = r+(1-r)*self.sensitivity_gap[j].sigmoid()
                observed = r+(q-r)*event
            loss = loss+F.binary_cross_entropy(observed.clamp(1e-6, 1-1e-6), targets[:, offset:offset+k])
            offset += k
        return loss
