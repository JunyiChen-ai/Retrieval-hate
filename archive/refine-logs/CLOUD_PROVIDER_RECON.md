# Cloud GPU provider recon — RGCL hateful-video project

**Prices verified on 2026-07-31** (NZ time; research session spanned 2026-07-30/31).
All numbers below are tagged with how they were obtained:

- **[LIVE]** — read directly from the vendor's own page/API *in this session*, through the
  university squid proxy.
- **[3P <date>]** — reported by a third party (aggregator/review site) as of the stated date;
  **not** vendor-confirmed. Treat as directional only.
- **[UNVERIFIED]** — could not be read this session; explicitly flagged, never quoted as fact.

Evidence rule applied throughout: no number appears here that was not read from a source in
this session. Where a vendor page rendered without its price table (Vast.ai marketing pages,
Hyperbolic), that is recorded as a failure, not backfilled from memory.

---

## 0. Two project-specific constraints that dominate the choice

Before any price comparison, two rules in this repo change what "move to cloud" can even mean:

1. **Raw video may never leave the cluster.** `scripts/cloud/modal_probe_runner.py:115`
   (`assert_uploadable`) fails loud on any media extension or `video/` path. The currently-held
   1-GPU **extraction** job reads raw video, so it **cannot** be moved to any cloud provider
   under the standing rule — not Modal, not RunPod, not Vast. Cloud can only take the
   *downstream* feature-space work. Lifting that rule is a user decision, not a procurement one.
2. **Cloud numbers are triage-only.** `CLAUDE.md` and `scripts/cloud/README.md` record a measured
   ~1.4 pp accuracy drift between banked local A100 and Modal T4 on the same seed/args, and forbid
   mixing cloud and local numbers in one table (G-repro discipline). Moving *formal 3-seed
   validation* to cloud does not just change the bill — it forces every comparison row in the
   affected table to be re-run on that same cloud hardware. The SLURM queue is currently acting as
   the **provenance anchor**, not merely as a compute source.

Consequence: the honest framing is *"cloud absorbs probes + feature-space training/inference; the
local queue keeps raw-video extraction and any number that enters a paper table."*

---

## 1. Proxy reachability — measured this session

All checks made from the login node through `http://squid.auckland.ac.nz:3128`.

### 1.1 HTTPS control planes (curl, single polite request each)

| Endpoint | HTTP status through squid | Reading |
|---|---|---|
| `https://api.modal.com` | 200 | reachable |
| `https://api.runpod.io/graphql` | 400 | reachable (400 = server answered) |
| `https://console.vast.ai/api/v0/` | 404 | reachable |
| `https://cloud.lambdalabs.com/api/v1/instance-types` | 401 | reachable (auth required) |
| `https://api.replicate.com/v1` | 401 | reachable |
| `https://api.together.xyz/v1/models` | 401 | reachable |
| `https://api.hyperbolic.xyz` | 200 | reachable |
| `https://api.paperspace.com/v1` | 404 | reachable |
| `https://api.coreweave.com` | 404 | reachable |

**Every provider's HTTPS API is reachable.** No provider is excluded on control-plane grounds.

### 1.2 Raw SSH through squid CONNECT — **works**

This was the open question in the brief. Measured by hand-writing `CONNECT host:port HTTP/1.1`
to squid:3128 and reading the first response line:

| Target | squid response |
|---|---|
| `ssh.runpod.io:22` | `HTTP/1.1 200 Connection established` → banner `SSH-2.0-Go` |
| `ssh1.vast.ai:22` | `HTTP/1.1 200 Connection established` → banner `SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.13` |
| `github.com:22` | `HTTP/1.1 200 Connection established` → banner `SSH-2.0-10bb30f` |
| `one.one.one.one:2053` (high port) | `HTTP/1.1 200 Connection established` |
| `one.one.one.one:853` | `HTTP/1.1 403 Forbidden` |
| `example.com:443`, `example.com:8080` | `200 Connection established` |

Interpretation: squid's `Safe_ports` ACL here permits **22, 443, 8080 and the 1025–65535 range**
(2053 established), and forbids off-list low ports (853 → explicit 403). Vast.ai and RunPod hand
out SSH on high ports on host IPs, which falls inside the permitted range.

**So the "raw-instance providers need SSH and SSH is blocked" worry is false.** Instance-model
providers are usable via:

```
Host vast-*
  ProxyCommand nc -X connect -x squid.auckland.ac.nz:3128 %h %p
```

Caveat: verified against the providers' *gateway* hosts and against an arbitrary high port on a
third-party host. Not verified against a live rented instance (that would require renting one,
which is out of scope for this research-only task).

### 1.3 Modal client deps (already known, re-stated)

`python-socks[asyncio]` (gRPC control plane) **and** `aiohttp-socks` (volume/blob upload path) are
both mandatory under squid — pinned in `scripts/cloud/requirements-cloud.txt`. Any rebuild of the
`HateVideo` env must reinstall both.

---

## 2. Per-provider findings

### 2.1 Modal — **[LIVE]** `https://modal.com/pricing`

Modal publishes **per-second** rates. Per-hour column is my arithmetic (×3600).

| GPU | $/sec [LIVE] | $/hr (derived) |
|---|---|---|
| B300 | 0.001972 | 7.0992 |
| B200 | 0.001736 | 6.2496 |
| H200 | 0.001261 | 4.5396 |
| H100 | 0.001097 | 3.9492 |
| RTX PRO 6000 | 0.000842 | 3.0312 |
| **A100 80 GB** | **0.000694** | **2.4984** |
| A100 40 GB | 0.000583 | 2.0988 |
| L40S | 0.000542 | 1.9512 |
| A10 | 0.000306 | 1.1016 |
| L4 | 0.000222 | 0.7992 |
| T4 | 0.000164 | 0.5904 |

Other **[LIVE]** facts:
- CPU `$0.0000131 / core / sec` (**minimum 0.125 cores per container**) = $0.04716/core/hr.
- Memory `$0.00000222 / GiB / sec` = $0.007992/GiB/hr.
- Volumes `$0.09 / GiB / mo`, footnoted **"includes 1 TiB / mo free"**.
- Plans: **Starter $0 base, $30/month free credits**; Team $250 base + $100/mo credits; Enterprise custom.
- Billing is per-second; the only stated minimum is the 0.125-core CPU floor. **No minimum
  duration or per-invocation minimum is stated.**
- **No egress/bandwidth line item appears anywhere on the pricing page** — I could not find a
  stated egress charge. Recorded as "not listed", not as "free".
- CPU/memory bundling: the page lists GPU, CPU and memory as **separate line items** and does
  **not** state that a GPU container includes free CPU/RAM. Cost estimates below therefore add a
  CPU+RAM term. If Modal in fact bundles some allocation, my Modal estimates are *over*-stated —
  which only strengthens the "stay on Modal" case, so the asymmetry is safe.

Cold start **[LIVE]** `https://modal.com/docs/guide/cold-start`: *"Containers boot in about one
second"*; default scaledown window **60 s** (configurable 2 s – 20 min); pre-baking tens-of-GB
models into the image *"can reduce boot times from minutes to seconds."*

Cross-check against our own bank: the README records T4 at `$0.000164/s` on 2026-07-14/15 — that
is **bit-identical** to today's published rate. Modal has not moved its price in ~2.5 weeks.

Free/academic credits:
- $30/mo Starter credits — **[LIVE]** on the pricing page.
- **Modal for Academics** (up to $25k) and Modal for Startups: the `/startups` page **[LIVE]**
  confirms a startups program (VC-funding gated — we do not qualify) and merely *mentions* that a
  "Modal for Academics" program exists in its FAQ, **with no amounts, eligibility, or status**.
  `modal.com/use-cases/academia` and `modal.com/docs/guide/free-credits` both **404**.
  Third-party reporting **[3P 2026]** claims up to **$25,000** for university faculty/postdocs/PhD
  students at accredited institutions, and separately that *"Modal's $10k academic credit program
  is currently paused"* with a mailing-list waitlist. **These are contradictory and unverified.**
  → **Action item, not a fact**: the exact academic-credit status must be confirmed by writing to
  Modal from the `@aucklanduni.ac.nz` address. A $25k grant would swamp every price difference in
  this document.

### 2.2 RunPod — **[LIVE]** `https://www.runpod.io/pricing`

Pods, on-demand $/hr:

| GPU | Community Cloud | Secure Cloud |
|---|---|---|
| RTX A6000 | 0.33 | 0.49 |
| **RTX 4090** | **0.34** | **0.69** |
| L40S | 0.79 | 0.99 |
| **A100 80 GB / A100 PCIe** | **1.19** | **1.39** |
| A100 SXM | 1.39 | 1.49 |
| H100 PCIe | 1.99 | 2.89 |
| H100 SXM | 2.69 | 2.99 |
| H200 | 3.59 | 4.39 |

Serverless, $/hr **[LIVE]** (`?tab=serverless`): B300 9.98 · B200 8.64 · H200 5.93 ·
RTX 6000 Pro 3.49 · H100 4.55 · **A100 80 GB 2.72** · L40/L40S/6000 Ada 1.75 · A6000/A40 1.22 ·
RTX 5090 1.58 · RTX PRO 4500 1.15 · RTX 4090 1.10 · L4/A5000/3090 0.69 · A4000-class 0.58.
(Note serverless A100 at $2.72/hr is **above** Modal's $2.4984/hr — RunPod's advantage is in
*pods*, not in its serverless tier.)

Storage **[LIVE]**: container disk $0.10/GB/mo · volume disk running $0.10/GB/mo · **volume disk
idle $0.20/GB/mo** · network storage $0.07/GB/mo (<1 TB), $0.05/GB/mo (>1 TB) · high-performance
network storage $0.14/GB/mo.

Billing granularity **[LIVE]**: the pricing page exposes both "Per hour" and "Per second" views.
Serverless docs **[LIVE]** `https://docs.runpod.io/serverless/pricing`: *"You're billed from when
a worker starts until it fully stops, rounded up to the nearest second"* — and you **do** pay for
container init / model load, and for the post-request idle window (default **5 s**).

Egress: **not stated on the pricing page**. **[3P 2026]** multiple reviewers state RunPod charges
no egress, with the caveat that *some Community Cloud hosts* apply their own network fees.
Recorded as third-party.

Reliability **[3P July 2026]**: consistent reviewer consensus that Community Cloud is
*"best-effort, no SLA"* — residential-internet hosts, unpredictable availability, and reported
failure mode of *"pods that fail to start while still billing."* Secure Cloud is described as
dedicated and materially more reliable. Recommendation in the reviews: Community for throwaway /
fault-tolerant batch; Secure when a lost run costs more than the $0.20/hr delta.

Connection **[LIVE]** `https://docs.runpod.io/pods/connect-to-a-pod`: web terminal, SSH,
JupyterLab and VS Code; HTTP services are exposed through
`https://<POD_ID>-8888.proxy.runpod.net` — i.e. **a pure-HTTPS path exists that needs no raw TCP
at all**. Combined with §1.2, RunPod is reachable two independent ways through squid.

### 2.3 Vast.ai — **[LIVE, queried directly]**

The marketing pages (`vast.ai/pricing`, `vast.ai/pricing/gpu/RTX-4090`, `.../A100-SXM4`) render
**without their price tables** through the proxy — I could not read a price off them, and am not
quoting one. Instead I queried the **public search API** through squid:

`PUT https://console.vast.ai/api/v0/search/asks/` — worked; returns live host offers.

Snapshot taken **2026-07-31**, filtering `rentable=true`, `num_gpus=1`, ordered by `dph_total`
(total $/hr incl. host markup). `n` = offers returned (capped at 64):

| Selection | type | n | min $/hr | median $/hr | max $/hr |
|---|---|---|---|---|---|
| **A100, 1×, ≥79 GB VRAM** | on-demand | 22 | **0.5496** | **1.0156** | 1.2889 |
| A100, 1×, ≥79 GB VRAM | interruptible (bid) | 19 | 0.1356 | 0.4696 | 1.0693 |
| A100 any VRAM, 1× | on-demand | 45 | 0.2681 | 0.8289 | 1.2889 |
| **RTX 4090, 1×, reliability ≥0.98** | on-demand | 64 | 0.1356 | **0.3413** | 0.4956 |
| RTX 4090, 1×, reliability ≥0.98 | bid | 64 | 0.1348 | 0.2895 | 0.6022 |
| RTX 5090, 1× | on-demand | 64 | 0.2129 | 0.4022 | — |
| RTX 5090, 1× | bid | 64 | 0.0844 | 0.2956 | — |
| L40S, 1× | on-demand | 6 | 0.4007 | 0.7693 | — |
| H100 (SXM/PCIe/NVL), 1×, ≥79 GB | on-demand | 17 | 1.4696 | 2.2696 | 5.4956 |
| H100, 1×, ≥79 GB | bid | 12 | 0.6022 | 1.4361 | 6.6689 |
| 2× A100 80 GB | on-demand | 14 | 1.3356 | 2.1087 | 3.4704 |

Also read live off the same offers: median host `reliability2` ≈ **0.99+** in every class;
median host **storage $0.20/GB/mo**; median **inet-down $0.0026/GB** (4090 pool) — i.e. bandwidth
is host-set and non-zero, unlike RunPod/Lambda.

Depth warning, straight from the counts: the **1× A100-80G pool is 22 offers deep and the 1× H100
pool is 17.** That is a thin market. The RTX 4090 / 5090 pools are 64+ (query cap). If a probe
needs 80 GB VRAM at a specific moment, Vast may simply not have a good host, and the median price
can move a lot day to day. This is a marketplace, not a price list.

Reliability **[3P 2026]**: reviewers converge on "renting from individuals; quality varies by host;
no SLA, no guaranteed uptime, support is community Discord"; standard advice is to filter
`reliability ≥ 0.95` for on-demand and `≥ 0.90` for interruptible. Interruptible instances are
reclaimed with *"a few minutes of warning."*

### 2.4 Lambda — **[LIVE]** `https://lambda.ai/pricing`

1× on-demand $/hr: B200 SXM6 **6.99** · H100 SXM **4.29** · H100 PCIe **3.29** · GH200 **2.29** ·
A100 SXM 40 GB **1.99** · A100 PCIe 40 GB **1.99** · A10 **1.29** · A6000 **1.09**.
8× tier per-GPU $/hr: B200 6.69 · H100 SXM 3.99 · **A100 SXM 80 GB 2.79** · A100 SXM 40 GB 1.99 ·
V100 0.79. 4× tier: B200 6.79 · H100 SXM 4.09 · A100 PCIe 40 GB 1.99 · A6000 1.09.

**Critical for us: there is no 1× A100-80 GB on-demand SKU.** The only 80 GB A100 rows are in the
8× node tier. Our workload is single-GPU; Lambda would force either a 40 GB A100 (too small for
7B-VL at 40–80 GB VRAM headroom) or an 8-GPU node.

Billing **[LIVE]**: *"Pay by the minute."* Egress **[LIVE]**: *"no egress fees."*
Storage: **[UNVERIFIED]** — no per-GB storage price appeared on the page I read.

### 2.5 Together AI — **[LIVE]** `https://www.together.ai/pricing`

GPU clusters, per GPU per hour, on-demand: **H100 $3.99 · H200 $5.99 · B200 $8.19**.
Reserved (7–180+ days): H100 $3.19–3.69 · H200 $3.99–4.99 · B200 $6.79–7.99.
Dedicated inference on-demand: H100 $5.49 · B200 $8.99.
Fine-tuning is **token-priced** (LoRA SFT $0.48–$2.90 / 1M tokens for ≤100B; **$4.00 minimum
charge per job**) — a managed-service abstraction, not a place to run our custom RGCL/RA-HMD
training loop.

**No A100 tier at all**, and the reserved tiers start at 7-day commitments. Wrong shape for
1.5-hour bursty single-GPU probes.

### 2.6 Replicate — **[LIVE]** `https://replicate.com/pricing`

T4 $0.000225/s (**$0.81/hr**) · L40S $0.000975/s (**$3.51/hr**) · **A100 80 GB $0.001400/s
($5.04/hr)** · H100 $0.001525/s (**$5.49/hr**).

Billing model **[LIVE], and this is the disqualifier**: for private models you *"pay for all the
time instances of the model are online: the time they spend setting up; the time they spend idle,
waiting for requests; and the time they spend active."* Only "fast booting fine-tunes" escape idle
billing. A100 at $5.04/hr **plus** paid idle is ~2× Modal and ~4× RunPod for our pattern.

### 2.7 CoreWeave — **[3P July 2026]** (pricing not self-serve readable)

**[3P]** H100 on-demand quoted around **$4.25/hr** per GPU (range ~$2.23–4.25 depending on node
config); A100 80 GB ≈ **$2.50/hr** normalised from 8-GPU nodes; A100 PCIe ≈ **$2.21/hr**.
Critically **[3P]**: *"There's no way to provision a single H100 or H200 from CoreWeave's current
pricing tier as a self-serve customer. The smallest unit is the 8-GPU HGX node."* Committed
discounts exist but rate tiers are unpublished and require an account manager.

→ **Structurally excluded**: 8-GPU node minimum + sales-gated onboarding vs our 1.5-GPU-hour bursts.

### 2.8 Paperspace / DigitalOcean — **[3P 2026-07-03]**

**[3P]** Paperspace public pricing: **A100 $3.18/hr**, **H100 $5.95/hr**; reserved 36-month commits
down to H100 $2.24/hr, A100-80G $1.15/hr. DigitalOcean GPU Droplets: 1× H100 ≈ **$3.39/hr**;
H200 on-demand $3.70/hr, spot $1.76/hr. Third party explicitly notes these are as of 03 Jul 2026
and may have moved. I did not verify any of these on a DigitalOcean-owned page.

On-demand A100 at ~$3.18/hr **[3P]** is above Modal's verified $2.4984/hr with none of Modal's
scale-to-zero benefit. No reason to pursue.

### 2.9 Google Colab / Cloud Run GPUs

**Colab [3P 2026]**: Pro **$9.99/mo**, Pro+ **$49.99/mo**; compute-unit burn ~1.76 CU/hr on T4 and
~15 CU/hr on A100; pay-as-you-go $9.99 per 100 CU ≈ 7 A100-hours. I could not verify these on a
Google-owned page this session.

Structural disqualifier regardless of price: Colab is a **notebook session**, not a job API. It has
no persistent volume for the feature caches, no scriptable submit-and-poll interface for 5 parallel
probes, and no reproducible container. It cannot host this project's harness.

**Cloud Run GPUs**: `cloud.google.com/run/pricing` **truncated on fetch through the proxy — could
not read the GPU table.** **[3P 2026]** reports Tier-1-region **L4 ≈ $0.0001867/GPU-s (≈$0.67/hr)**
without zonal redundancy, ≈$0.0002909/s with; per-second billing; **no GPU free tier**. Cloud Run
GPU support is L4-centric — **[3P]** L4's 24 GB does not cover the 40–80 GB VRAM requirement for
the 7B-VL extraction leg. Useful only for the small feature-space probes, at which point Modal's
L4 at a verified $0.7992/hr is comparable and already integrated.

### 2.10 Fireworks — **[LIVE]** `https://fireworks.ai/pricing`

On-demand GPU deployments, per-GPU-second billing, no startup charge: **H100 80 GB $7.00/hr ·
H200 141 GB $7.00/hr · B200 $10.00/hr · B300 $12.00/hr**. **No A100 SKU.** Fine-tuning is
token-priced (LoRA SFT $0.50/1M tokens ≤16B).

Fireworks is an inference-serving platform priced for production LLM serving. $7/hr H100 with no
A100 option is ~5× RunPod's A100 for a job that does not need an H100. Excluded.

### 2.11 Hyperbolic — **could not verify**

`api.hyperbolic.xyz` answers 200 through the proxy, but `hyperbolic.xyz/pricing` 301s to
`www.hyperbolic.ai/pricing` which returns **404**; the homepage **[LIVE]** states only that
*"On-demand GPUs are usage-based and billed by the hour"* and that H100/H200/B200 are offered,
with **prices visible only behind a dashboard login at app.hyperbolic.ai**. Docs URL also 404s.

**No Hyperbolic price is quoted in this document.** Two independent negatives: prices are
login-gated (so they cannot be audited before committing), and hourly — not per-second — billing
is the wrong granularity for 1.5-hour bursts. Not recommended without a login-verified quote.

### 2.12 SF Compute — **[LIVE homepage, partial]**

Homepage states **H100 ≈ $2.21 per GPU-hour (average, subject to market conditions)**; B300 coming
fall 2026. `sfcompute.com/pricing` redirects to an auth flow — **the actual pricing page is
login-gated**. CLI examples on the homepage buy in **node × days** increments (e.g. 32 nodes for
3 days). The product is a *futures market for cluster time* with VMs, bare metal and managed Slurm.

Wrong granularity: we need 1 GPU for 1.5 hours, not 32 nodes for 3 days. Also, it would reintroduce
exactly the Slurm-queue friction the user is trying to escape.

### 2.13 Others noted as of mid-2026 — **[3P July 2026], not verified**

**[3P]** Thunder Compute (A100 80 GB **$1.09/hr**, per-minute billing, VS Code extension),
Prime Intellect (per-second, on-demand + interruptible, **[3P]** "11 GPUs from $0.14/hr"),
DataCrunch/Verda (**[3P]** H100 $1.99/hr; rebranded to Verda Nov 2025), Spheron
(**[3P]** H100 spot $1.03/hr, on-demand $2.50/hr; A100-80G $1.07/hr on-demand, $0.60/hr spot),
JarvisLabs. **None of these was verified on a vendor page this session.** They are recorded so the
user knows the long tail exists, not as candidates — switching from a proven, proxy-validated
harness to an unverified vendor for a few hundred dollars a year is a bad trade (see §5).

---

## 3. Comparison table — the single-GPU 40–80 GB class we actually need

Sorted by verified 1× A100-80 GB-class on-demand $/hr. `—` = SKU does not exist for 1×.

| Provider | Model | 1× A100-80G $/hr | 1× H100 $/hr | 4090-class $/hr | Granularity | Persistent vol | Egress | Proxy | Verified |
|---|---|---|---|---|---|---|---|---|---|
| **Vast.ai** | marketplace instance | **0.5496 min / 1.0156 med** | 1.4696 min / 2.2696 med | 0.1356 min / 0.3413 med | per-second | host disk ~$0.20/GB/mo | host-set ~$0.0026/GB | SSH via CONNECT ✓ + HTTPS API ✓ | LIVE (API) |
| **RunPod (Community)** | instance/pod | **1.19** | 1.99 PCIe / 2.69 SXM | 0.34 | per-second | net vol $0.07/GB/mo | [3P] none | HTTPS proxy URLs ✓ + SSH ✓ | LIVE |
| **RunPod (Secure)** | instance/pod | **1.39** | 2.89 PCIe / 2.99 SXM | 0.69 | per-second | net vol $0.07/GB/mo | [3P] none | HTTPS ✓ + SSH ✓ | LIVE |
| Lambda | instance | — (8× only, $2.79/GPU) | 3.29 PCIe / 4.29 SXM | — | per-minute | not listed | none (stated) | HTTPS API ✓ | LIVE |
| **Modal** | serverless function | **2.4984** | 3.9492 | RTX PRO 6000 3.0312 | per-second | $0.09/GiB/mo, 1 TiB free | not listed | **HTTPS/gRPC ✓ (in production)** | LIVE |
| RunPod Serverless | serverless | 2.72 | 4.55 | 1.10 | per-second (rounded up) | net vol | [3P] none | HTTPS ✓ | LIVE |
| Paperspace | instance | 3.18 | 5.95 | — | [UNVERIFIED] | — | — | HTTPS ✓ | [3P 07-03] |
| Together | cluster | — (no A100) | 3.99 | — | hourly, 7d+ reserved tiers | — | — | HTTPS ✓ | LIVE |
| CoreWeave | 8-GPU node only | ~2.50 (normalised) | ~4.25 | — | node-level | — | — | HTTPS ✓ | [3P] |
| Replicate | serverless | 5.04 **+ paid idle** | 5.49 | — | per-second incl. setup+idle | model store | — | HTTPS ✓ | LIVE |
| Fireworks | inference platform | — (no A100) | 7.00 | — | per-GPU-second | — | — | HTTPS ✓ | LIVE |
| SF Compute | market, node×days | — | ~2.21 | — | node × days | — | — | HTTPS ✓ | LIVE (partial) |
| Hyperbolic | instance | **login-gated** | login-gated | login-gated | hourly (stated) | — | — | API 200 ✓ | **failed** |
| Colab | notebook | ~15 CU/hr | — | — | compute units | none | — | browser | [3P] |
| Cloud Run | serverless | — (L4 only) | — | L4 ≈0.67 | per-second | GCS | GCP egress | HTTPS ✓ | [3P] |

---

## 4. Cost arithmetic for this project's two canonical jobs

CPU/RAM adder for Modal assumes a modest container (2 cores + 16 GiB):
`2 × $0.0000131 + 16 × $0.00000222 = $0.00006172/s = $0.2222/hr`. Instance providers bundle
CPU/RAM in the quoted rate.

### 4.1 Typical probe — 1.5 GPU-h on A100-80G

| Provider | Compute | + overhead | **$/probe** |
|---|---|---|---|
| Vast.ai on-demand (median $1.0156/hr) | $1.523 | negligible | **$1.52** |
| Vast.ai on-demand (cheapest today $0.5496/hr) | $0.824 | negligible | **$0.82** |
| Vast.ai interruptible (median $0.4696/hr) | $0.704 | + restart risk | **$0.70** |
| RunPod Community ($1.19/hr) | $1.785 | — | **$1.79** |
| RunPod Secure ($1.39/hr) | $2.085 | — | **$2.09** |
| **Modal ($2.4984/hr)** | $3.748 | $0.333 CPU+RAM | **$4.08** (→ **$0.00** while inside the $30/mo credit) |
| RunPod Serverless ($2.72/hr) | $4.080 | — | **$4.08** |
| Replicate ($5.04/hr) | $7.560 | + boot + idle | **$7.56+** |

Reality check against our own bank: the probes this project actually runs are mostly *head-train
on cached features* — the README records **30 epochs in ~33 s on a T4**. At Modal's verified T4
rate that is **$0.0054**. The "1.5 GPU-h A100 probe" is the heavy tail, not the mode. **The $30/mo
Modal credit covers a very large number of the actual probes outright.**

### 4.2 Training run — 4 GPU-h A100-80G LoRA, ×3 seeds

| Provider | $/run (4 GPU-h) | $/3-seed set |
|---|---|---|
| Vast.ai on-demand median | $4.06 | $12.19 |
| RunPod Community | $4.76 | $14.28 |
| RunPod Secure | $5.56 | $16.68 |
| Lambda 1× A100 **40 GB** (VRAM may not fit) | $7.96 | $23.88 |
| **Modal** | $9.99 + $0.89 = **$10.88** | **$32.65** |
| Replicate | $20.16 + idle | $60.48+ |
| Fireworks / Together (H100 only) | $28.00 / $15.96 | $84.00 / $47.88 |

### 4.3 Annual delta — with the volume assumption stated

**Assumption (state it, it drives everything):** ~**500 A100-80G-equivalent GPU-hours/year** —
roughly 300 h of probing plus 200 h of feature-space training/inference. This is deliberately
generous: at the project's *observed* probe cost (33-second head trains) the true figure is far
lower, and the campaign log shows entire rounds costing ~16 GPU-h total.

| Provider | Annual compute | Storage (~10 GB caches) | Free credits | **Net/yr** |
|---|---|---|---|---|
| Vast.ai on-demand median | $507.80 | ~$24 | — | **~$532** |
| RunPod Community | $595.00 | $8.40 | — | **~$603** |
| RunPod Secure | $695.00 | $8.40 | — | **~$703** |
| **Modal** | $1,249.20 + $111 CPU/RAM | $0 (within 1 TiB free) | −$360 ($30×12) | **~$1,000** |

**Honest annual delta, Modal vs the cheapest verified alternative: roughly $470/year**
(Modal ~$1,000 vs Vast ~$532); vs RunPod Community ~$400/year; vs RunPod Secure ~$300/year.

At a **lower, more realistic** volume the ranking **inverts**. Take 150 A100-equivalent hours/year,
which is closer to the campaign's actual burn (whole rounds have cost ~16 GPU-h):

| Provider | Annual compute @150 h | Free credits | **Net/yr** |
|---|---|---|---|
| **Modal** | $374.76 + $33.33 CPU/RAM = $408.09 | −$360 | **~$48** |
| Vast.ai on-demand median | $152.34 | — | **~$152** (+ storage) |
| RunPod Community | $178.50 | — | **~$179** |
| RunPod Secure | $208.50 | — | **~$209** |

Because the $360/yr of Starter credits is a *fixed* subsidy, **Modal is the cheapest option
outright below the break-even, and only becomes the expensive option above it.** Solving
`2.7206·h − 360 = rate·h` gives the crossover in A100-equivalent hours/year:

- vs Vast.ai on-demand median ($1.0156/hr): **h ≈ 211**
- vs RunPod Community ($1.19/hr): **h ≈ 235**
- vs RunPod Secure ($1.39/hr): **h ≈ 271**

**The break-even is the number that matters, not the headline $/hr.** Under ~210 A100-hours a
year, Modal is simply the cheapest thing available to this project.

---

## 5. Recommendations

### 5.1 Lightweight probes → **stay on Modal**. Fallback: **RunPod pods**.

Modal wins on every axis that matters for a 0.5–2.5 GPU-h burst:
- Per-second billing, ~1 s container boot **[LIVE]**, scale-to-zero — **you pay for the 33 seconds
  the probe actually runs**, not for an instance you forgot to kill. On an instance provider a
  single forgotten A100 pod overnight (~$14–17) wipes out a month of savings.
- $30/mo free credits **[LIVE]** plausibly cover the entire real probe load.
- The harness exists, is debugged (two runner bugs already fixed), is proxy-validated end to end,
  and enforces the video guard. **Its price is $0 and it is already paid for.**
- 5 parallel probes need no capacity hunting, no SSH, no cleanup discipline.

$/probe (1.5 GPU-h A100): **Modal $4.08 gross, $0.00 net while under the monthly credit** ·
**RunPod Community $1.79 / Secure $2.09**.

### 5.2 Training + inference → **RunPod Secure Cloud** primary, **Vast.ai on-demand** fallback.

Once a job runs for hours, Modal's serverless premium stops buying anything — there is no cold
start to amortise and no idle to avoid. RunPod Secure A100-80G at **$1.39/hr [LIVE]** is **1.8×
cheaper than Modal's $2.4984/hr [LIVE]**, keeps per-second billing, offers a **$0.07/GB/mo network
volume** the feature caches can live on across runs, and reaches through squid **two independent
ways** (HTTPS `proxy.runpod.net` and SSH — both measured in §1).

$/run (4 GPU-h A100 LoRA): **RunPod Secure $5.56** · RunPod Community $4.76 ·
**Vast.ai on-demand median $4.06** · Modal $10.88.

Use **Secure**, not Community, for anything whose loss costs more than the $0.20/hr difference —
$0.80 per 4-hour run is cheap insurance against the reviewer-reported *"pod fails to start while
still billing"* failure mode **[3P July 2026]**.

Vast.ai as fallback, with **hard filters**: `reliability ≥ 0.98`, on-demand only, and check pool
depth first — the 1× A100-80 GB pool was **22 offers** today.

### 5.3 Is staying on Modal for everything defensible? **Yes.**

At the stated 500 A100-h/yr the total penalty is about **$470/year**. Below ~210 A100-h/yr there is
**no penalty at all** — the $30/mo credit makes Modal strictly cheaper than every alternative
(§4.3). Against that $470 worst case: a proven, proxy-validated, guard-enforcing harness, zero
idle-instance risk, and no new operational surface. **$470/year is not a reason to migrate — but
it is a reason to route the long jobs, and only the long jobs, elsewhere.** Hence the split
recommendation: probes stay, multi-hour training goes. If the year's cloud burn stays under ~210
A100-equivalent hours, staying on Modal for **everything** is not merely defensible, it is optimal.

The one thing that would change this calculus decisively is the **Modal for Academics** grant. If
it is live and we qualify, Modal becomes free for everything and the discussion ends. **That email
is worth more than this entire document** — see §2.1.

### 5.4 Migration cost — what moving training to RunPod actually requires

Low, and bounded. Nothing about the Modal path is discarded.

1. **Account + payment** (user action; not done, per instructions). RunPod is prepaid credit.
2. **Feature caches**: re-upload ~1.1 GB (HateMM 361.9 MB + MHC 361.0 MB + MHC_zh 385.6 MB, per
   the README's verified sync counts) to a RunPod **network volume** (~$0.08/mo at $0.07/GB/mo).
   One-time.
3. **Container image**: the Modal image spec already pins the exact dependency set
   (`torch 2.6.0+cu124`, `easydict/pandas/pillow/rank-bm25/torchmetrics/wandb`) — that translates
   to a Dockerfile or a `pip install` in the pod's start command almost mechanically. This is the
   single largest task and it is hours, not days.
4. **The upload guard must be ported.** `assert_uploadable` (`modal_probe_runner.py:115`) is
   currently the *only* enforcement of the raw-video ban. **A RunPod path with no equivalent guard
   would silently delete the project's data boundary.** Port `guard_reason`/`assert_uploadable`
   into whatever sync script pushes to the network volume, before the first byte moves. Non-negotiable.
5. **Proxy config**: `~/.ssh/config` `ProxyCommand nc -X connect -x squid.auckland.ac.nz:3128 %h %p`
   (§1.2 shows this path is open), or use the HTTPS `proxy.runpod.net` route and skip SSH entirely.
   RunPod's API is plain HTTPS at `api.runpod.io/graphql` — reachable (§1.1).
6. **Lifecycle discipline**: instances do **not** scale to zero. Every launch script must set a
   self-terminate on completion. This is the real ongoing cost of leaving serverless, and it is
   an operational habit, not a technical obstacle.
7. **Provenance**: any cloud training number inherits the triage-only rule (§0.2). If a cloud
   number is ever to enter a paper table, the *entire* table must be re-run on that same hardware.

### 5.5 What NOT to do

- **Do not move the held extraction job to any cloud.** It reads raw video; §0.1 forbids it. No
  provider choice fixes this — only a user ruling on the data boundary would.
- **Do not use Replicate** for this workload. $5.04/hr A100 **[LIVE]** *and* it bills setup and
  idle time **[LIVE]** — the two worst properties for bursty work, stacked.
- **Do not use Fireworks or Together** for custom training. No A100 SKU; Fireworks H100 $7.00/hr
  **[LIVE]**; Together's fine-tuning is a token-priced managed service that cannot run the RGCL
  loop, with a $4.00 per-job minimum **[LIVE]**.
- **Do not pursue CoreWeave or SF Compute.** 8-GPU node minimum **[3P]** and node×days market
  increments **[LIVE]** respectively. Both are wrong-granularity by an order of magnitude, and
  SF Compute would re-import Slurm-queue friction — the exact thing being escaped.
- **Do not sign up to Hyperbolic on the strength of a blog number.** Its prices are **login-gated**
  and I could not read one; it bills **hourly** **[LIVE homepage]**, which is wrong for 1.5-h jobs.
- **Do not treat Colab as infrastructure.** No persistent volume, no job API, no reproducible
  container — it cannot host the harness regardless of price.
- **Do not run Vast.ai *interruptible* for training.** Reclaimed with minutes of notice **[3P]**;
  a killed 4-hour run costs more than the $0.55/hr it saved. Interruptible is fine only for
  checkpointing jobs — which ours are not, at 4 GPU-h.
- **Do not use RunPod Community Cloud for anything whose loss matters.** Reviewer-reported
  *"pods that fail to start while still billing"* **[3P July 2026]**; no SLA.
- **Do not leave an instance running.** The entire cost advantage of leaving Modal is smaller than
  one forgotten A100 weekend (~$140 at $1.39/hr × 100 h).
- **Do not migrate the probe harness.** It works, it is proxy-validated, and it enforces the data
  boundary. Its replacement cost far exceeds the ~$400/year it "wastes".
- **Do not mix cloud and local numbers in one table**, and do not let a cloud training run become
  a paper number without re-running its whole comparison row on the same hardware (§0.2).

---

## 6. Sources

Read live through the squid proxy on **2026-07-31** unless noted.

- Modal pricing — https://modal.com/pricing
- Modal cold start guide — https://modal.com/docs/guide/cold-start
- Modal volumes guide — https://modal.com/docs/guide/volumes
- Modal startups page — https://modal.com/startups
- Modal free-credits / academia pages — https://modal.com/docs/guide/free-credits and https://modal.com/use-cases/academia (**both 404 on 2026-07-31**)
- RunPod pricing — https://www.runpod.io/pricing and https://www.runpod.io/pricing?tab=serverless
- RunPod serverless pricing docs — https://docs.runpod.io/serverless/pricing
- RunPod pod connection docs — https://docs.runpod.io/pods/connect-to-a-pod
- Vast.ai live offers — `PUT https://console.vast.ai/api/v0/search/asks/` (queried 2026-07-31; marketing pages https://vast.ai/pricing and https://vast.ai/pricing/gpu/RTX-4090 rendered **without** price tables)
- Lambda pricing — https://lambda.ai/pricing and https://lambda.ai/service/gpu-cloud
- Together AI pricing — https://www.together.ai/pricing
- Replicate pricing — https://replicate.com/pricing
- Fireworks pricing — https://fireworks.ai/pricing
- SF Compute — https://sfcompute.com/ (pricing page https://sfcompute.com/pricing is **auth-gated**)
- Hyperbolic — https://www.hyperbolic.ai/ (pricing page **404**; https://docs.hyperbolic.ai/docs/hyperbolic-pricing **404**)
- Google Cloud Run pricing — https://cloud.google.com/run/pricing (**GPU table truncated on fetch**)

Third-party, **[3P]**, dates as stated by the source:

- CoreWeave 8-GPU-node minimum and pricing — https://www.thundercompute.com/blog/coreweave-gpu-pricing-review ; https://gpucost.org/provider/coreweave (July 2026)
- Paperspace / DigitalOcean pricing — https://www.spheron.network/blog/digitalocean-gpu-pricing-2026/ ; https://computeprices.com/providers/paperspace (03 Jul 2026)
- RunPod reliability and egress — https://deploybase.ai/articles/runpod-review ; https://gigagpu.com/runpod-gpu-shortages-reliability/ ; https://www.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs (2026)
- Vast.ai reliability — https://www.aitooldiscovery.com/ai-infra/vast-ai-review ; https://gpuhosted.com/en/vast-ai-review/ (2026)
- Modal academic credits (contradictory) — https://grantedai.com/grants/modal-labs-startup-and-academics-programs-for-serverless-gpu-compute-cre-modal-labs-847dd6f7 ; https://www.thundercompute.com/blog/free-cloud-gpu-credits (2026)
- Cloud Run L4 GPU-second pricing — https://getdeploying.com/gpus/nvidia-l4 ; https://cloudchipr.com/blog/cloud-run-pricing (2026)
- Colab Pro / Pro+ pricing and compute units — https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025 ; https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/
- Long-tail providers (Thunder Compute, Prime Intellect, DataCrunch/Verda, Spheron) — https://www.thundercompute.com/blog/cheapest-cloud-gpu-providers ; https://gpufinder.dev/providers/primeintellect ; https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/ (July 2026)
