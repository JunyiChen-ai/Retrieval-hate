# Modal features-only offload (RGCL)

**Scope (binding).** ONLY derived float feature caches (`data/CLIP_Embedding/<ds>/*.pt`)
and label files (`data/gt/<ds>/*.jsonl|json|csv`) may leave the cluster. **Raw videos
never do** — `assert_uploadable` in `modal_probe_runner.py` fails loud on any media
extension or `video/` path before uploading a byte. Cloud numbers are **exploratory
triage only**; any pre-registered / paper number is re-run locally (G-repro rule).

**Commands**
1. User, once: `python3 -m modal setup` (browser auth → writes `~/.modal.toml`).
2. Validate token + billing + CPU/T4: `modal run scripts/cloud/modal_smoke.py`.
3. Push a dataset's caches + labels: `modal run scripts/cloud/modal_probe_runner.py::sync --dataset HateMM`.
4. Run a probe (CPU default; add `--gpu` for T4):
   `modal run scripts/cloud/modal_probe_runner.py::run --script src/run_rac.py --args "--path /root/data --dataset HateMM --model Qwen2.5-VL-7B-Instruct_HF --seed 0"`.

**Cost.** $30/mo free credits cover our probing load (~free). T4 ≈ $0.000164/s
(~$0.59/hr); a 25-s head-train ≈ $0.004. CPU worker is cheaper still.
