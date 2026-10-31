"""Run-time check for README section 6: forward + backward time of the scale-aware attention with the bias table
gradient computed by the index lookup's own backward (seed-234 searches) and by the dense product (`_BucketBias`),
at the model size of the searches (hid 128, 4 heads) and training batch shapes. Also checks that both give the same
outputs and table gradient. Needs a GPU (submit with launch/sbatch_bench.sh)."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qtl import scale_cma


class LookupAttention(scale_cma.ScaleBiasAttention):
    """The seed-234 version: plain index lookup, autograd's index backward."""

    def forward(self, q, k, v, key_mask=None):
        orig = scale_cma._BucketBias.apply
        scale_cma._BucketBias.apply = lambda table, buckets: table[:, buckets]
        try:
            return super().forward(q, k, v, key_mask)
        finally:
            scale_cma._BucketBias.apply = orig


def run(attn, x, n):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        attn.zero_grad(set_to_none=True)
        attn(x, x, x).sum().backward()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    dev = "cuda"
    torch.manual_seed(0)
    new = scale_cma.ScaleBiasAttention(4, 128, True, dropout=0.0).to(dev)
    old = LookupAttention(4, 128, True, dropout=0.0).to(dev)
    old.load_state_dict(new.state_dict())
    with torch.no_grad():
        new.scale_bias.normal_()
        old.scale_bias.copy_(new.scale_bias)
    print(torch.cuda.get_device_name(0), torch.__version__)
    for B, T in ((32, 120), (32, 300), (8, 1000), (4, 2000)):
        x = torch.randn(B, T, 128, device=dev)
        o1, o2 = old(x, x, x), new(x, x, x)
        g1 = torch.autograd.grad(o1.square().sum(), old.scale_bias)[0]
        g2 = torch.autograd.grad(o2.square().sum(), new.scale_bias)[0]
        same = bool(torch.equal(o1, o2))
        rel = float((g1 - g2).abs().max() / g1.abs().max())
        run(old, x, 2), run(new, x, 2)
        t_old, t_new = run(old, x, 10), run(new, x, 10)
        print(f"B {B} T {T} | lookup {t_old * 1e3:.1f} ms | dense {t_new * 1e3:.1f} ms | same forward {same} "
              f"| table grad max rel diff {rel:.1e}", flush=True)


if __name__ == "__main__":
    main()
