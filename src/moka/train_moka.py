#!/usr/bin/env python
"""MokA-routed LoRA SFT entry point — a wrapper around the DEPLOYED LLaMA-Factory entry.

Replicates `scripts/slurm/lora_sft.sbatch` step 2 (`cd $LF_ROOT && python src/train.py <yaml>`,
i.e. `llamafactory.train.tuner.run_exp()`) with EXACTLY ONE change: `get_peft_model` inside
`llamafactory.model.adapter` (imported at adapter.py:19, called at adapter.py:312) is monkey-patched
so the returned PeftModel has its `lora.Linear` layers converted to `MokaLinear` (modality-routed
`A`, shared `B`).  ZERO lines of the vendored LLaMA-Factory tree are edited.

The patch lands at adapter.py:312, i.e. BEFORE adapter.py:314-316 casts trainable params to fp32,
so `lora_A_v` is picked up by that cast exactly like `lora_A` / `lora_B`.

Usage (from $LF_ROOT, same cwd as the deployed invocation):
    python /data/jehc223/RGCL/src/moka/train_moka.py <config.yaml>
"""

import os
import sys

RGCL_ROOT = "/data/jehc223/RGCL"
LF_ROOT = os.path.join(RGCL_ROOT, "RA-HMD", "LLAMA-FACTORY-Ver202512")

# `llamafactory` is not pip-installed; the deployed entry works because `python src/train.py` puts
# $LF_ROOT/src on sys.path[0].  Reproduce that, and add THIS file's own directory (src/moka, which
# holds only routed_lora.py + train_moka.py) so `import routed_lora` cannot shadow anything.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LF_ROOT, "src"))

import llamafactory.model.adapter as lf_adapter  # noqa: E402
from routed_lora import install_moka, moka_param_report  # noqa: E402

_ORIG_GET_PEFT_MODEL = lf_adapter.get_peft_model


def _moka_get_peft_model(model, peft_config, *args, **kwargs):
    peft_model = _ORIG_GET_PEFT_MODEL(model, peft_config, *args, **kwargs)
    n = install_moka(peft_model)
    rep = moka_param_report(peft_model)
    print("[moka] routed {} lora.Linear layers -> MokaLinear (A_v added, B shared)".format(n), flush=True)
    print("[moka] trainable params: {}".format(rep), flush=True)
    return peft_model


lf_adapter.get_peft_model = _moka_get_peft_model

from llamafactory.train.tuner import run_exp  # noqa: E402


if __name__ == "__main__":
    os.chdir(LF_ROOT)  # the deployed invocation runs with cwd == $LF_ROOT
    print("[moka] patched llamafactory.model.adapter.get_peft_model; cwd={}".format(os.getcwd()), flush=True)
    run_exp()
