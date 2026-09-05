"""Context-measured leave-one-position residuals and shared evidence deletion."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from hier_evidence_common import SCAF_OFFSET, A_EXT_DIM


class Candidate(nn.Module):
    def __init__(self, mean, std, dropout=.2, arm='full'):
        super().__init__()
        self.arm = arm
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        self.content = nn.Linear(1920, 128)
        self.evidence = nn.Linear(30, 128)
        self.temporal = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.reconstruct = nn.Linear(256, 1920)
        self.residual = nn.Linear(1920, 128)
        self.selector = nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.fuse = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(128, 1)

    def forward(self, audio, visual, seq_len=None):
        x = torch.cat([visual, audio[..., :SCAF_OFFSET]], -1)
        e = audio[..., A_EXT_DIM:]
        joined = (torch.cat([x, e], -1)-self.mean)/self.std
        x, e = joined[..., :1920], joined[..., 1920:]
        lengths = (torch.full((x.shape[0],), x.shape[1], device=x.device, dtype=torch.long)
                   if seq_len is None else torch.as_tensor(seq_len, device=x.device, dtype=torch.long))
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        h, v = F.gelu(self.content(x)), F.gelu(self.evidence(e))
        if self.arm != 'no_residual':
            packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            encoded, _ = self.temporal(packed)
            encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
            left, right = encoded[..., :128], encoded[..., 128:]
            if self.arm != 'visible_reconstruction':
                left = F.pad(left[:, :-1], (0, 0, 1, 0))
                right = F.pad(right[:, 1:], (0, 0, 0, 1))
            estimate = self.reconstruct(torch.cat([left, right], -1))
            residual = F.gelu(self.residual(x-estimate))
            per_token_mse = (x-estimate).square().mean(-1)
            reconstruction_loss = ((per_token_mse*mask).sum(1)/lengths).mean()
        else:
            residual = torch.zeros_like(h)
            reconstruction_loss = x.new_zeros(())
        z = self.selector(torch.cat([h, residual, v], -1))
        q = z.squeeze(-1).sigmoid()
        # Compute dropout-bearing token features once: all three calls see the
        # same token realization and share exactly the same classifier.
        tokens = self.fuse(torch.cat([h, v], -1))
        def classify(weight):
            weights = weight*mask
            pooled = (tokens*weights[..., None]).sum(1)/weights.sum(1, keepdim=True).clamp_min(1e-6)
            return self.classifier(pooled).squeeze(-1)
        full, kept, erased = classify(torch.ones_like(q)), classify(q), classify(1-q)
        self._terms = full, kept, erased, ((q*mask).sum(1)/lengths).mean(), reconstruction_loss
        # Shared evaluator uses only sigmoid(output[1]), averaged over 5 crops.
        return full.sigmoid(), z, z, z, h, v

    def loss(self, labels, lengths):
        full, kept, erased, sparsity, reconstruction = self._terms
        loss = F.binary_cross_entropy_with_logits(full, labels)+F.binary_cross_entropy_with_logits(kept, labels)
        if self.arm != 'no_deletion':
            loss = loss+F.binary_cross_entropy_with_logits(erased, torch.zeros_like(labels))
        if self.arm != 'no_sparsity':
            loss = loss+sparsity
        return loss+reconstruction
