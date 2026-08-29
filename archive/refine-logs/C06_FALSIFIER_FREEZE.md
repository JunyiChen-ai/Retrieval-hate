# C06 $0 CPU falsifier — FREEZE

Date: 2026-08-05

| path | sha256 |
|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E2.md` | `a24188868272716ffcedcfc0dbb9769f5843f9fe34ebff66ecfc713e118337ae` |
| `scripts/analysis/c06_falsifier_arena.py` | `0ff7eedea2932d12303a7268ec0d4daa609f77c02b16b8b0bec1b82e1a692372` |
| `scripts/analysis/c06_falsifier_mint.py` | `299d04020489362558c6f4411fb702fad19abf50ef105eccd5321442842474ea` |
| `configs/c06/c06_falsifier.json` | `8196d163f39299273040dc8532a022433afb6c6289ca642b9360cdf99c6615db` |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `0cadfc7af7b132b1ba443311dec2f56ef960181e40e25fc3f134803f33ba0917` |

single submission authorized

sbatch re-frozen after job 13987 died in 1 s on `ModuleNotFoundError: No module named 'torch'` — the script activated no conda environment (CODE-R1 M-3, recorded and never wired). Environment-only; GATE-SHA had already passed 38/38 and no design, config or analysis file changed, so the four other digests above are unmoved.
