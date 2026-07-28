# Cloud GPU Feasibility for RGCL Hateful-Video Probing (2026-07-14)

**Author:** cloud-GPU feasibility scout (research only — no sign-ups, uploads, purchases, or code changes made)
**Question:** Can we offload our GPU work to an API-accessible cloud service with *minimal* code change, given (i) a local SLURM cluster that queues for hours and (ii) datasets that are **hateful-content research videos** under usage agreements?

**One-line answer:** Yes for the *cheap, low-value* half of the work (features-only head-training/probing), essentially for free on **Modal**. **No — do not — for the raw-video feature-extraction half**, which is both the GPU-heavy part *and* the part that trips every provider's Acceptable-Use Policy and the dataset licenses. The honest tension is that the job that needs the GPU is the sensitive one, and the job that is safe to offload barely needs a GPU.

---

## 0. Our workload, confirmed from the repo

Read from `scripts/slurm/*.sbatch` and `src/utils/generate_VideoMLLM_embedding_HF.py`:

| Class | Job | GPU need | Reads | Writes | Sensitivity |
|---|---|---|---|---|---|
| **(a) Extraction** | `generate_VideoMLLM_embedding_HF.py` — frozen Qwen2.5-VL-7B forward, 8 frames/video, mean-pool hidden states | 1–2 A100-h, real GPU | **raw `.mp4` videos** in `data/video/<ds>/All/` (~10–30 GB) + Qwen weights (16 GB, pulled from HF) | small `.pt` caches (3584-d floats), ~1–3 MB each; whole HateMM cache dir = 318 MB | **HIGH** — raw hateful videos |
| **(b) Head training** | `run_rac.py` (RGCL retrieval-contrastive head) on cached features | ~20–25 s/run, GPU barely used (`--Faiss_GPU False`) | `.pt` feature caches + `data/gt/*.jsonl` labels | metrics to log | **LOW** — 3584-d float vectors, no reconstructable content |
| **(c) Eval / probe** | localization scorers, cross-matrix evals | ~1 min | cached features | metrics | LOW |

**Two facts that drive the whole recommendation:**
1. The GPU is only genuinely needed for **(a)**, which requires **uploading raw hateful videos** — the AUP + license landmine.
2. **(b)/(c)** are what actually hurt today (fast jobs stuck behind a multi-hour SLURM queue), and they need **only float vectors** — no videos, no 16 GB weights, barely any GPU. The real pain is **queue latency on fast iterative probes**, not a compute shortfall.

So the win is: run **features-only (b)/(c) probes on an always-on cheap cloud box** to kill queue latency; **keep raw-video extraction (a) on academic infrastructure** (local SLURM or NeSI) for license/AUP reasons.

---

## 1. Shortlist (top 3) — one-liners

| Rank | Option | Integration delta | Cost (our jobs) | AUP verdict |
|---|---|---|---|---|
| **1** | **Modal** | ~15 lines (`@app.function(gpu=...)` wrapper around `run_rac.py`); or `subprocess` the exact sbatch command | (b)/(c): **~free** ($30/mo credits cover it); (a): $2.50–5/run on A100-80GB | **Features-only: clean.** Raw video: gray — bans "indecent/obscene", **no explicit hate ban**, no research carve-out → don't upload video |
| **2** | **SkyPilot** (on a backing cloud) | ~10-line YAML; `run:` block = your literal sbatch body → **true "zero code change"** | = whatever backing cloud charges (Lambda A100 $1.99/hr, RunPod A100 ~$1.2/hr) | Inherits the **backing cloud's** AUP → still features-only |
| **3** | **RunPod** | Pods (SSH, VM-like) small delta; Serverless needs a **Docker image** (bigger delta) | A100 ~$1.19–1.89/hr on-demand, per-second serverless | **Raw video: bad** — ToS bans "objectionable" + "offensive comments connected to race, national origin, gender…" |

**Runner-up / compliant home for (a):** **NeSI** (NZ eScience Infrastructure) — academic, University of Auckland is a principal investor, research-appropriate data governance, **no commercial AUP prohibiting hateful research content**. Catch: allocation runs on **quarterly Call-for-Applications cycles** and it is itself an HPC queue, so it does not solve "queue forever" quickly. It is the *license-clean* place to run raw-video extraction, not a latency fix.

**Disqualified:** **Vast.ai** (cheapest, but a marketplace of untrusted third-party hosts *and* ToS explicitly bans "racist, hateful… discriminatory" content — worst possible fit for this data); **Together / Replicate** (inference-only / Cog-packaged model hosting — cannot run our arbitrary training script); **Colab** (no real batch API).

---

## 2. Per-option detail

### 2.1 Modal — TOP PICK (features-only)
- **Integration delta.** Minimal. A working wrapper is ~15 lines and calls our **existing** `run_rac.py` unchanged:
  ```python
  # modal_headtrain.py — features-only; NO videos, NO 16GB Qwen weights needed for run_rac.py
  import modal, subprocess
  image = (modal.Image.debian_slim()
           .pip_install("torch", "faiss-cpu", "transformers==4.49", "numpy", "scikit-learn")
           .add_local_dir("src", "/root/src"))          # sync our code
  app = modal.App("rgcl-headtrain", image=image)
  feats = modal.Volume.from_name("rgcl-feats", create_if_missing=True)  # holds .pt caches + data/gt

  @app.function(gpu="T4", volumes={"/root/data": feats}, timeout=1800)
  def run(dataset="HateMM", model="Qwen2.5-VL-7B-Instruct_HF", seed=0):
      subprocess.run(["python", "src/run_rac.py", "--dataset", dataset,
                      "--model", model, "--seed", str(seed), "..."], cwd="/root", check=True)

  @app.local_entrypoint()
  def main(): run.remote()
  ```
  One-time: `modal volume put rgcl-feats ./data/CLIP_Embedding /data/CLIP_Embedding` (a few MB of floats) and the `data/gt` labels. Then `modal run modal_headtrain.py`. A T4 ($0.000164/s) or even CPU is plenty for a 25-s head-train; no A100 required for (b).
- **Data path.** `modal.Volume` is a persistent distributed FS; `modal volume put` uploads from laptop, `volumes={...}` mounts it. Crucially, for extraction you could `from_pretrained(...)` **inside** the function to pull the 16 GB Qwen weights **cloud-side straight from HF into the Volume — you never upload weights**. (Confirmed in Modal volumes docs.)
- **Cost.** Per-second billing; A100-80GB $0.000694/s ≈ **$2.50/hr**, A100-40GB ≈ $2.10/hr, H100 ≈ $3.95/hr, T4 ≈ $0.59/hr. Idle scale-to-zero. **$30/mo free credits.** Our (b)/(c) at ~10 runs/mo ≈ **$0** (inside credits); (a) if ever run ≈ $2.50–5/run.
- **AUP.** `modal.com/legal/terms`: "Prohibited Content" = illegal / IP-infringing / **"indecent or obscene"** / defamatory / malicious code. **No explicit ban on hateful or discriminatory content**, and no research carve-out. Narrowest of the four → **float vectors are clearly fine**; raw hateful video is a gray zone (avoid).
- **Availability.** Serverless, no queue; on-demand A100/H100 generally available 2026.
- **Setup burden (one-time):** create account, `pip install modal`, `modal token new` (browser auth), add a card for beyond-credits usage.

### 2.2 SkyPilot — the true "zero code change" path
- **Not a cloud** — an orchestrator that provisions on a backing account (Lambda / RunPod / AWS / GCP / Kubernetes / even Slurm). Supports existing GPU workloads with **no code changes**.
- **Integration delta.** ~10-line YAML whose `run:` block is literally the body of our sbatch:
  ```yaml
  # rgcl.sky.yaml
  resources: {accelerators: A100:1}
  workdir: .                                   # syncs the whole repo
  file_mounts: {/data/CLIP_Embedding: ./data/CLIP_Embedding}   # upload float caches only
  setup: |
    pip install torch faiss-cpu transformers==4.49 numpy scikit-learn
  run: |
    python src/run_rac.py --dataset HateMM --model Qwen2.5-VL-7B-Instruct_HF --seed 0 ...
  ```
  `sky launch -c rgcl rgcl.sky.yaml`; reuse with `sky exec rgcl rgcl.sky.yaml`; `sky down rgcl` to stop billing.
- **Data path.** `workdir`/`file_mounts` rsync local dirs up; can also mount cloud buckets.
- **Cost.** Backing-cloud rate while the VM is up (you pay hourly, so `sky down` between runs or keep one cheap dev box up and `sky exec`). No SkyPilot surcharge (open-source, Apache-2).
- **AUP.** Whatever the backing cloud's is (see Lambda/RunPod below) → still **features-only**.
- **Setup burden.** Higher: needs a configured **underlying** cloud account + credentials before `sky launch` works. Best choice **only if** the user wants to run the existing script verbatim and is willing to stand up one backing account (Lambda is the least-effort backer).

### 2.3 RunPod
- **Integration delta.** *Pods* = SSH into a VM, `git pull` + run (VM-like, small delta but manual). *Serverless* (per-second) needs the job packaged as a **Docker image + handler** → meaningfully more work than Modal's decorator.
- **Cost.** A100 on-demand ~$1.19–1.89/hr; serverless per-second; network volumes $0.07–0.14/GB-mo.
- **AUP.** `runpod.io/legal/terms-of-service`: content must not be "obscene, lewd, lascivious, filthy, violent, harassing… or otherwise objectionable" and must not include "offensive comments… connected to race, national origin, gender, sexual preference, or physical handicap." RunPod "may access, store, process, and use any of Your Content." → **raw hateful video plainly violates**; features-only is defensible but the platform is less friendly than Modal.

### 2.4 Lambda Cloud
- Raw **SSH VM** provider (cheap A100 $1.99/hr, H100 $3.29/hr, zero egress fees), best consumed **via SkyPilot** rather than directly (no decorator/API-line ergonomics of its own).
- **AUP.** `lambda.ai/legal/terms-of-service`: prohibits "distributing any offensive materials, including… obscene… indecent, or **hateful**… or discrimination based on race, sex, religion, nationality, disability, sexual orientation, or age." No research carve-out. Operative verb is "distributing," but relying on that distinction is risky → features-only.

### 2.5 HuggingFace Jobs (`hf jobs`)
- **Integration delta.** `hf jobs run --flavor a100-large <docker-image> <cmd>` or `hf jobs uv run script.py` — needs a **Docker image or uv-script** wrapper (moderate delta). Weights/datasets already in the HF ecosystem is a plus.
- **Cost.** Per-**minute** billing; T4 $0.40/hr up to multi-A100 $10–20/hr; needs a **paid HF plan (Pro/Team)** + token.
- **AUP note.** HF *hosts* hateful-content research datasets (Hateful Memes, MMHS) on the Hub, so the ecosystem is comparatively research-friendly — **but** we did not verify their Jobs AUP text verbatim, so treat as features-only too. Reasonable third choice if the team already lives in HF.

### 2.6 Vast.ai — DISQUALIFIED for this data
Cheapest on paper (A100 $0.29–0.80/hr, 4090 $0.29/hr) but (i) ToS bans content that is "unlawful, **racist, hateful**, abusive, libelous, obscene, or discriminatory," and (ii) it's a **marketplace of independent third-party hosts** with disclaimed data security ("Company shall not be held responsible for the acts or omissions of any Provider"). Uploading hateful research video to an unvetted stranger's machine is unacceptable on both license and AUP grounds. Even features-only, the trust model is wrong.

### 2.7 Together / Replicate — DISQUALIFIED
Inference-only (per-token API / Cog-packaged model hosting). Cannot run our arbitrary `run_rac.py` training loop. Not applicable.

### 2.8 NeSI (academic) — the compliant home for extraction (a)
- University of Auckland is a principal NeSI investor; 8× A100 available; PhD students eligible via **Institutional** or supervisor-sponsored **Merit** allocations. Data governance is research-appropriate → **no commercial AUP conflict** with hateful-content research, and data can stay in NZ under the dataset agreements.
- **Reality check:** allocations run on **quarterly Call-for-Applications** cycles (apply → reviewed the following month → 3–6-month preliminary allocation), and it is an HPC queue like the current cluster. So NeSI **does not fix queue latency quickly**; it is the *license-clean* venue for the sensitive raw-video extraction, worth pursuing in parallel but not a fast probing fix.

---

## 3. Recommendation

**Top pick: Modal, for features-only (b)/(c) probing.** Smallest code delta ("add a decorator"), per-second billing, $30/mo credits that likely cover our entire probing load, weights pull cloud-side, and the narrowest AUP of the commercial options. This directly kills the queue-latency pain on the exact jobs that hurt.

**Runner-up: SkyPilot on a Lambda backend**, if the team prefers to run the **existing sbatch command verbatim** (`run:` block) and doesn't mind standing up one backing cloud account.

**What stays local (or on NeSI): raw-video feature extraction (a).** Do not upload HateMM / MultiHateClip videos to any commercial GPU cloud. It is both the license-sensitive step and the one every provider's AUP restricts.

### Migration path — job (b) head-training / probing (the easy win)
1. One-time: `pip install modal`; `modal token new`; `modal volume create rgcl-feats`.
2. `modal volume put rgcl-feats ./data/CLIP_Embedding` + `data/gt` (a few MB of **float vectors + labels only** — no videos, no weights).
3. Add ~15-line `modal_headtrain.py` (§2.1) that `subprocess`-calls the **unchanged** `run_rac.py`.
4. `modal run modal_headtrain.py --dataset ... --seed ...` — result in seconds, no queue.
5. Sensitivity: the uploaded `.pt` are 3584-d L2-normed floats — not "content" under any provider's AUP, not reconstructable into video/images. This is the low-risk, high-relief path.

### Migration path — job (a) extraction (only if ever offloaded; not recommended to commercial cloud)
Same Modal wrapper but `gpu="A100"`, pull Qwen weights via `from_pretrained` cloud-side into the Volume (never upload 16 GB), and — the blocker — you would still have to upload raw videos, which we advise **against**. Keep (a) on local SLURM or pursue **NeSI** for it.

---

## 4. Honest risk list

1. **Dataset license / usage agreement (raw video).** HateMM and MultiHateClip are under research usage agreements; redistributing raw videos to a third-party cloud may breach them. **User's call to read the agreements** — but the safe default (features-only) sidesteps it entirely.
2. **Provider AUP (raw video).** Vast.ai/RunPod/Lambda explicitly restrict racist/hateful/objectionable/discriminatory content; Modal restricts indecent/obscene. **None carve out research or defensive analysis.** A provider can suspend at "sole discretion." → never upload raw hateful video; float vectors only.
3. **G-repro cross-hardware determinism (REAL protocol hazard).** Our protocol reports metrics to **4 decimal places** and makes pre-registered pass/fail calls on paired seed tests (e.g., "+0.03 acc on ≥2 datasets", "+0.04 substantial" bars). Different silicon/driver/CUDA/cuDNN/PyTorch build changes results at ~1e-3:
   - **Extraction** mean-pools bf16 hidden states and decodes frames (decord/PyAV) — kernel and library differences mean cloud-extracted features will **not** match local ones; a cloud feature cache is **not interchangeable** with a local one inside the same table.
   - **Head-training** drifts run-to-run beyond the 4th dp via cuDNN algo choice, non-deterministic reductions, and RNG-stream differences across GPU generations.
   → **Rule:** use cloud strictly for **exploratory triage** ("is this idea alive?"). Any number entering a paper table or a pre-registered decision must be produced on the **same hardware as the rest of the table** (the local A100), re-run locally for the official value. Pin torch/cuda/cudnn and set deterministic flags to reduce (not eliminate) drift. Never mix cloud- and locally-extracted caches in one comparison.
4. **Cost surprises.** Small here (probing ≈ free within credits), but: idle SkyPilot/Lambda VMs bill hourly until `sky down`; RunPod/HF storage volumes bill monthly; egress on some providers (Lambda = zero egress, a plus). Set a spend cap.
5. **Value/GPU mismatch.** The offloadable job (features-only head-training) barely needs a GPU; the real benefit is **latency, not compute**. Don't over-invest — a single small always-on box (T4/CPU) for probes is the right size.

---

## 5. What we need from the user to activate the top pick (Modal, features-only)

1. **Create a Modal account** and run `modal token new` (browser auth) — produces the API token.
2. **Add a payment card** for usage beyond the $30/mo free credits (probing likely stays within credits; a card is still required to activate).
3. **Confirm it is acceptable to upload the derived `.pt` float feature caches** (3584-d vectors, no reconstructable content) to Modal. (Our read: not "content" under Modal's AUP; low risk. Final call is the user's.)
4. **Explicit ruling that raw videos stay local / NeSI-only** — i.e., we will *not* upload HateMM/MHC `.mp4` to any commercial cloud.
5. **(Optional, parallel) Authorize applying to NeSI** for the raw-video extraction workload if we want a second, license-clean cluster for (a) — but expect quarterly-cycle turnaround, not an instant fix.

Dataset-license reading (item 1 of the risk list) is the user's call and is only needed if they ever want to reconsider offloading raw video; the features-only path does not require it.

---

## Sources
- Modal pricing: https://modal.com/pricing • GPU guide: https://modal.com/docs/guide/gpu • Volumes: https://modal.com/docs/guide/volumes • Terms: https://modal.com/legal/terms
- RunPod pricing: https://www.runpod.io/pricing • Terms of Service: https://www.runpod.io/legal/terms-of-service
- SkyPilot: https://github.com/skypilot-org/skypilot • Quickstart: https://docs.skypilot.co/en/latest/getting-started/quickstart.html
- Lambda pricing: https://lambda.ai/pricing • Terms/AUP: https://lambda.ai/legal/terms-of-service
- Vast.ai pricing: https://vast.ai/pricing • Terms: https://vast.ai/terms
- HuggingFace Jobs pricing: https://huggingface.co/docs/hub/jobs-pricing • Jobs guide: https://huggingface.co/docs/huggingface_hub/guides/jobs
- Together AI docs: https://docs.together.ai/docs/inference/overview
- NeSI allocations: https://docs.nesi.org.nz/General/NeSI_Policies/Allocation_classes/ • GPU rollout: https://www.nesi.org.nz/case-studies/tech-insights-behind-scenes-look-rolling-out-new-gpu-resources-nz-researchers
