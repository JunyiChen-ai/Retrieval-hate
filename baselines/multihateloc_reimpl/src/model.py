#!/usr/bin/env python3
"""MultiHateLoc model (reimplementation).

Faithful to Sun et al., "MultiHateLoc: Towards Temporal Localisation of Multimodal
Hate Content in Online Videos", WWW'26 (arXiv 2512.10408v3), Fig 2 and Sec 3.2-3.5.

Components:
  MA-TE      : per-modality Transformer temporal encoder (Eq 1).
  CM-Contrast: InfoNCE across modalities at matched (video,timestamp) (Eq 2).
  DCM-Fusion : Dynamic Modality Selection (Eq 3-4) + Cross-Modal Attention (Eq 5-7).
  MA-MIL     : per-modality + fused top-K frame heads (Eq 8-11) with adaptive K.
  losses     : L_total = L_MA-MIL + 0.1 L_smooth + 0.2 L_con  (Eq 12-13).

Hidden dim D and #layers are unspecified in the paper -> D=256, 1 MA-TE layer,
4 attention heads (paper Table 5 uses 4 heads for CMA). See README_deviations.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MATE(nn.Module):
    """Modality-aware temporal encoder: input proj to D, then one Transformer
    encoder layer (self-attention + FFN), Eq (1)."""
    def __init__(self, in_dim, d_model=256, heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff_mult * d_model,
            dropout=dropout, activation="relu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x, key_padding_mask):
        # x [B,T,in_dim]; key_padding_mask [B,T] True=pad
        h = self.proj(x)
        return self.enc(h, src_key_padding_mask=key_padding_mask)  # [B,T,D]


class MultiHateLoc(nn.Module):
    def __init__(self, d_model=256, heads=4, dropout=0.1):
        super().__init__()
        self.d = d_model
        self.mate_v = MATE(768, d_model, heads, dropout=dropout)
        self.mate_a = MATE(128, d_model, heads, dropout=dropout)
        self.mate_t = MATE(768, d_model, heads, dropout=dropout)

        # Dynamic Modality Selection: per-timestep scalar gate per modality (Eq 3)
        self.dms_v = nn.Linear(d_model, 1)
        self.dms_a = nn.Linear(d_model, 1)
        self.dms_t = nn.Linear(d_model, 1)

        # Cross-Modal Attention over concatenated (3D) features (Eq 5-7)
        self.cma_q = nn.Linear(3 * d_model, d_model)
        self.cma_k = nn.Linear(3 * d_model, d_model)
        self.cma_v = nn.Linear(3 * d_model, d_model)
        self.cma_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout,
                                              batch_first=True)

        # MA-MIL heads: per-modality frame scores (Eq 8) + fused frame score (Eq 9)
        self.cls_v = nn.Linear(d_model, 1)
        self.cls_a = nn.Linear(d_model, 1)
        self.cls_t = nn.Linear(d_model, 1)
        self.cls_fused = nn.Linear(d_model, 1)

    def forward(self, fv, fa, ft, mask):
        # mask [B,T] 1=valid; key_padding_mask expects True=pad
        kpm = (mask < 0.5)
        Fv = self.mate_v(fv, kpm)   # [B,T,D]
        Fa = self.mate_a(fa, kpm)
        Ft = self.mate_t(ft, kpm)

        # --- Dynamic Modality Selection (Eq 3-4) ---
        av = torch.sigmoid(self.dms_v(Fv))   # [B,T,1]
        aa = torch.sigmoid(self.dms_a(Fa))
        at = torch.sigmoid(self.dms_t(Ft))
        Wv, Wa, Wt = av * Fv, aa * Fa, at * Ft

        # --- Cross-Modal Attention (Eq 5-7) ---
        concat = torch.cat([Wv, Wa, Wt], dim=-1)         # [B,T,3D]
        q = self.cma_q(concat); k = self.cma_k(concat); v = self.cma_v(concat)
        fused, _ = self.cma_attn(q, k, v, key_padding_mask=kpm,
                                 need_weights=False)      # [B,T,D]

        # --- MA-MIL frame scores (Eq 8-9) ---
        p_v = torch.sigmoid(self.cls_v(Fv)).squeeze(-1)   # [B,T]
        p_a = torch.sigmoid(self.cls_a(Fa)).squeeze(-1)
        p_t = torch.sigmoid(self.cls_t(Ft)).squeeze(-1)
        p_fused = torch.sigmoid(self.cls_fused(fused)).squeeze(-1)  # localization out

        return {
            "p_fused": p_fused, "p_v": p_v, "p_a": p_a, "p_t": p_t,
            "Fv": Fv, "Fa": Fa, "Ft": Ft,
            "gate_v": av.squeeze(-1), "gate_a": aa.squeeze(-1),
            "gate_t": at.squeeze(-1),
        }


# ------------------------------- losses --------------------------------------
def topk_indices(prob, mask, k_div=3):
    """Adaptive top-K (paper Table 4): select top round(T_valid/k_div) frames
    (k_div=3 -> top 33%). Returns a boolean [B,T] selection mask over valid frames."""
    B, T = prob.shape
    sel = torch.zeros_like(mask)
    neg_inf = torch.finfo(prob.dtype).min
    masked = prob.masked_fill(mask < 0.5, neg_inf)
    lens = mask.sum(1).long()
    for i in range(B):
        Ti = int(lens[i].item())
        if Ti <= 0:
            continue
        ki = max(1, int(round(Ti / k_div)))
        ki = min(ki, Ti)
        idx = torch.topk(masked[i], ki).indices
        sel[i, idx] = 1.0
    return sel


def mil_bce(prob, y, mask, k_div=3, eps=1e-6):
    """Per-frame BCE over the adaptive top-K set against the video label (Eq 11,
    completed with the y=0 term). Averaged per video, then over the batch."""
    sel = topk_indices(prob, mask, k_div)               # [B,T]
    p = prob.clamp(eps, 1 - eps)
    bce = -(y.unsqueeze(1) * torch.log(p) +
            (1 - y).unsqueeze(1) * torch.log(1 - p))     # [B,T]
    num = (bce * sel).sum(1)
    den = sel.sum(1).clamp(min=1.0)
    return (num / den).mean(), sel


def video_score(prob, mask, k_div=3):
    """Video-level probability = mean of top-K fused frame probs (for AUC/F1/acc)."""
    sel = topk_indices(prob, mask, k_div)
    return (prob * sel).sum(1) / sel.sum(1).clamp(min=1.0)


def smoothness_loss(prob, mask):
    """L_smooth (Eq 12): mean squared successive difference over valid frames."""
    d = (prob[:, 1:] - prob[:, :-1]) ** 2                # [B,T-1]
    m = mask[:, 1:] * mask[:, :-1]
    return (d * m).sum() / m.sum().clamp(min=1.0)


def cm_contrast_loss(Fv, Fa, Ft, mask, tau=0.1):
    """CM-Contrast (Eq 2): InfoNCE across modality pairs. Positives = same
    (video,timestamp) across modalities; negatives = all other valid tokens in
    the batch. Computed over the flattened set of valid (video,time) tokens."""
    B, T, D = Fv.shape
    flat = mask.reshape(-1) > 0.5                        # [B*T]
    zv = F.normalize(Fv.reshape(-1, D)[flat], dim=-1, eps=1e-8)
    za = F.normalize(Fa.reshape(-1, D)[flat], dim=-1, eps=1e-8)
    zt = F.normalize(Ft.reshape(-1, D)[flat], dim=-1, eps=1e-8)
    N = zv.shape[0]
    if N < 2:
        return Fv.sum() * 0.0
    pairs = [(zv, za), (za, zt), (zv, zt)]
    losses = []
    device = Fv.device
    labels = torch.arange(N, device=device)
    for x, ynorm in pairs:
        logits = (x @ ynorm.t()) / tau                  # [N,N] anchor x candidate
        # symmetric InfoNCE: positive = matched index (same video+timestamp)
        losses.append(0.5 * (F.cross_entropy(logits, labels) +
                             F.cross_entropy(logits.t(), labels)))
    return torch.stack(losses).mean()


def total_loss(out, y, mask, k_div=3, lam_smooth=0.1, lam_con=0.2):
    """L_total = L_MA-MIL + 0.1 L_smooth + 0.2 L_con (Eq 13).
    L_MA-MIL = L_fused + sum_m w_m L_m, w_m = mean gate (detached) per modality."""
    l_fused, _ = mil_bce(out["p_fused"], y, mask, k_div)
    l_v, _ = mil_bce(out["p_v"], y, mask, k_div)
    l_a, _ = mil_bce(out["p_a"], y, mask, k_div)
    l_t, _ = mil_bce(out["p_t"], y, mask, k_div)
    # modality importance weights from DMS gates (Eq 10, detached scalar per batch)
    den = mask.sum().clamp(min=1.0)
    w_v = (out["gate_v"] * mask).sum().detach() / den
    w_a = (out["gate_a"] * mask).sum().detach() / den
    w_t = (out["gate_t"] * mask).sum().detach() / den
    l_mamil = l_fused + w_v * l_v + w_a * l_a + w_t * l_t

    l_smooth = smoothness_loss(out["p_fused"], mask)
    l_con = cm_contrast_loss(out["Fv"], out["Fa"], out["Ft"], mask)
    total = l_mamil + lam_smooth * l_smooth + lam_con * l_con
    return total, {"total": float(total), "mamil": float(l_mamil),
                   "fused": float(l_fused), "smooth": float(l_smooth),
                   "con": float(l_con)}
