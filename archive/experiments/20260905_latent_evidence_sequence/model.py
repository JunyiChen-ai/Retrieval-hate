"""Joint observation/state model with normalized parallel log-semiring scans."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from hier_evidence_common import A_EXT_DIM, SCAF_OFFSET

NEG = -1e30  # Numerically zero mass; avoids gradients of logsumexp(all -inf).


def log_product(left, right):
    return torch.logsumexp(left.unsqueeze(-1) + right.unsqueeze(-3), dim=-2)


def prefix_products(matrices, reverse_product=False):
    """Inclusive associative scan, keeping each matrix normalized for precision.

    Returns log matrices up to a scalar plus that cumulative log scalar.
    Reverse products support a suffix pass after flipping the time axis.
    """
    scale = torch.logsumexp(matrices, dim=(-2, -1))
    current = matrices - scale[..., None, None]
    stride = 1
    while stride < current.shape[1]:
        left, right = current[:, :-stride], current[:, stride:]
        product = log_product(right, left) if reverse_product else log_product(left, right)
        norm = torch.logsumexp(product, dim=(-2, -1))
        next_scale = scale[:, :-stride] + scale[:, stride:] + norm
        current = torch.cat([current[:, :stride], product - norm[..., None, None]], dim=1)
        scale = torch.cat([scale[:, :stride], next_scale], dim=1)
        stride *= 2
    return current, scale


class Candidate(nn.Module):
    def __init__(self, mean, std, state_means, dropout=.2, ablation='full', max_seqlen=200):
        super().__init__()
        self.ablation = ablation
        self.dim = len(mean)
        self.register_buffer('obs_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('obs_std', torch.as_tensor(std, dtype=torch.float32))
        self.means = nn.Parameter(torch.as_tensor(state_means, dtype=torch.float32))
        raw = torch.eye(self.dim).repeat(2, 1, 1) * math.log(math.expm1(1 - 1e-4))
        self.raw_cholesky = nn.Parameter(raw)
        self.content = nn.Sequential(nn.Linear(1024 + SCAF_OFFSET, 128), nn.GELU(), nn.Dropout(dropout))
        self.temporal = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.transition = nn.Linear(384, 2)
        self.initial = nn.Linear(128, 1)
        self.static_logits = nn.Parameter(torch.tensor([-math.log(max_seqlen-1), 0.]))
        nn.init.zeros_(self.transition.weight)
        with torch.no_grad():
            self.transition.bias.copy_(self.static_logits)
        nn.init.zeros_(self.initial.weight)
        nn.init.constant_(self.initial.bias, -math.log(max_seqlen-1))

    def emission(self, observations):
        z = (observations - self.obs_mean) / self.obs_std
        diagonal = F.softplus(self.raw_cholesky.diagonal(dim1=-2, dim2=-1)) + 1e-4
        chol = torch.diag_embed(diagonal)
        if self.ablation != 'diagonal_emission':
            chol = chol + self.raw_cholesky.tril(-1)
        difference = z[:, :, None, :] - self.means[None, None, :, :]
        solved = torch.linalg.solve_triangular(chol[None, None], difference.unsqueeze(-1), upper=False).squeeze(-1)
        return -.5 * (solved.square().sum(-1) + self.dim * math.log(2 * math.pi)) - diagonal.log().sum(-1)

    def forward(self, f_a, f_v, seq_len=None):
        batch, length, _ = f_v.shape
        if f_a.shape[-1] != A_EXT_DIM + self.dim:
            raise ValueError('unexpected observation dimensions')
        if seq_len is None:
            seq_len = torch.full((batch,), length, dtype=torch.long, device=f_v.device)
        else:
            seq_len = seq_len.to(f_v.device)
        valid = torch.arange(length, device=f_v.device)[None, :] < seq_len[:, None]
        x = self.content(torch.cat([f_v, f_a[..., :SCAF_OFFSET]], -1))
        x = x * valid[..., None]
        if self.ablation != 'no_temporal_content':
            x = self.drop(F.gelu(self.temporal(x.transpose(1, 2)).transpose(1, 2)))
            x = x * valid[..., None]
        previous = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        rates = self.transition(torch.cat([previous, x, x-previous], -1))
        if self.ablation == 'static_transition':
            rates = self.static_logits.expand_as(rates)
        elif self.ablation == 'independent_state':
            # Rows identical: P(next=1) = sigmoid(first logit) from either state.
            rates = torch.stack([rates[..., 0], -rates[..., 0]], dim=-1)
        to_one, to_zero = rates.unbind(-1)
        a00, a01 = F.logsigmoid(-to_one), F.logsigmoid(to_one)
        a10, a11 = F.logsigmoid(to_zero), F.logsigmoid(-to_zero)
        zero = torch.full_like(a00, NEG)
        # States: never-positive background, positive, previously-positive background.
        trans = torch.stack([torch.stack([a00, a01, zero], -1),
                             torch.stack([zero, a11, a10], -1),
                             torch.stack([zero, a01, a00], -1)], -2)
        emission = self.emission(f_a[..., A_EXT_DIM:])
        emission = emission[..., [0, 1, 0]]
        matrices = trans + emission.unsqueeze(-2)
        initial = self.initial(x[:, 0]).squeeze(-1)
        first = torch.stack([F.logsigmoid(-initial), F.logsigmoid(initial),
                             torch.full_like(initial, NEG)], -1) + emission[:, 0]
        matrices = torch.cat([first[:, None, None, :].expand(-1, 1, 3, -1), matrices[:, 1:]], dim=1)
        identity = torch.full((3, 3), NEG, device=f_v.device, dtype=f_v.dtype)
        identity.fill_diagonal_(0.)
        matrices = torch.where(valid[..., None, None], matrices, identity)
        prefixes, scales = prefix_products(matrices)
        end = prefixes[:, -1, 0]
        end_norm = torch.logsumexp(end, -1)
        end = end - end_norm[:, None]
        self.event_log_probs = torch.stack([end[:, 0], torch.logsumexp(end[:, 1:], -1)], -1)
        self.observation_nll = -(scales[:, -1] + end_norm) / (seq_len * self.dim)
        if not self.training or self.ablation == 'event_to_topk':
            suffixes, _ = prefix_products(matrices.flip(1), reverse_product=True)
            suffixes = suffixes.flip(1)
            beta = torch.cat([torch.logsumexp(suffixes[:, 1:], dim=-1),
                              torch.zeros_like(end[:, None, :])], dim=1)
            posterior = (prefixes[:, :, 0, :] + beta).softmax(-1)[..., 1]
            self.posterior = posterior
            logits = torch.logit(posterior.clamp(1e-7, 1-1e-7))[..., None]
        else:
            self.posterior = None
            logits = torch.zeros((batch, length, 1), device=f_v.device)
        return self.event_log_probs[:, 1].exp(), logits, logits, logits, x, emission

    def loss(self, labels, seq_len):
        if self.ablation == 'event_to_topk':
            bags = torch.stack([self.posterior[i, :int(n)].topk(max(1, math.ceil(int(n)/16))).values.mean()
                                for i, n in enumerate(seq_len)])
            event = F.binary_cross_entropy(bags, labels, reduction='none')
        else:
            event = -self.event_log_probs.gather(1, labels.long()[:, None]).squeeze(1)
        return (event if self.ablation == 'no_observation_likelihood' else event + self.observation_nll).mean()
