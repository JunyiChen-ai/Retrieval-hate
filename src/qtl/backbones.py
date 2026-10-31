"""Model factory for loading finished trials of any content backbone (qtl.checkpoint.load_trial)."""
from qtl import inside_outside, scale_cma


def make_model(cfg):
    if cfg["backbone"] == "scale_cma":
        return scale_cma.make_model(cfg)
    return inside_outside.make_model(cfg)          # inside_outside or macil (r5)
