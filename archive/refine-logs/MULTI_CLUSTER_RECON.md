# Multi-Server SLURM Recon — can we submit to foscsmlprd02 / 03?

**Status:** `COMPLETE — READ-ONLY RECON, ZERO JOBS SUBMITTED`
**Date:** 2026-07-30 (Pacific/Auckland)
**Scope:** read-only infrastructure diagnosis. No `sbatch`, no `scancel`, no
`scontrol update`, no writes to any host, no port scans, no credential attempts.
Every command below is a query. Two other agents were operating the queue
concurrently; nothing in this recon touched it.

---

## Verdict in one line

`foscsmlprd01` is a **single-node SLURM cluster**. There is no second node, no
federation, no multi-cluster registration, and **no shared filesystem** — `/data`
is a local LVM/XFS volume on this box. "Submit where GPUs are free, keep assets
on 01" is **not achievable today by any user-side action**, and the storage
half of it cannot be achieved by IT either without exporting a network
filesystem. Separately, the recon found that the 8-hour queue waits are an
**administrative approval latency, not GPU scarcity** — which means more servers
may not fix the problem the user actually has.

---

## 1. Topology — one cluster, one node

### 1.1 The cluster is a single node

```
$ sinfo -N -l
Thu Jul 30 20:35:03 2026
NODELIST      NODES       PARTITION       STATE CPUS    S:C:T MEMORY TMP_DISK WEIGHT AVAIL_FE REASON
foscsmlprd01      1 slurmpartition*       mixed 256    2:64:2 100000        0      1   (null) none

$ sinfo
PARTITION       AVAIL  TIMELIMIT  NODES  STATE NODELIST
slurmpartition*    up   infinite      1    mix foscsmlprd01
```

One partition. One node. `foscsmlprd02` and `foscsmlprd03` are not in it.

### 1.2 Config confirms it

```
$ scontrol show config | grep -i -E 'ClusterName|AccountingStorage|Federation|SlurmctldHost'
AccountingStorageBackupHost = (null)
AccountingStorageEnforce = none
AccountingStorageHost   = localhost
AccountingStorageExternalHost = (null)
AccountingStorageType   = accounting_storage/slurmdbd
ClusterName             = machine-learning-group
FederationParameters    = (null)
SlurmctldHost[0]        = foscsmlprd01
SLURM_VERSION           = 25.11.4
```

`AccountingStorageHost = localhost` is decisive: the `slurmdbd` is on this box
and is not shared with anything. `AccountingStorageExternalHost = (null)`.

`/etc/slurm/slurm.conf`, compute-node section:

```
149:# COMPUTE NODES
150:#NodeName=foscsmlprd02-host CPUs=1  Sockets=1 State=UNKNOWN
151:NodeName=foscsmlprd01 Gres=gpu:a100:8 Sockets=2 CoresPerSocket=64 ThreadsPerCore=2 RealMemory=1000000
152:PartitionName=DEFAULT State=UP
153:PartitionName=slurmpartition Nodes=foscsmlprd01 Default=YES MaxTime=INFINITE MaxMemPerNode=512000 MaxCPUsPerNode=128
```

`/etc/slurm/gres.conf`:

```
AutoDetect=nvml
#NodeName=foscsmlprd02-gpu Name=gpu Count=1 Type=a100 File=/dev/nvidia1 Flags=nvidia_gpu_env
NodeName=foscsmlprd01 Name=gpu Count=8 Type=a100 File=/dev/nvidia[0-7] Flags=nvidia_gpu_env
```

**Both config files carry a commented-out `foscsmlprd02` stanza.** Someone
prepared, or at least sketched, adding `foscsmlprd02` to this same cluster and
then disabled it. The values (`CPUs=1 Sockets=1`, `Count=1 File=/dev/nvidia1`)
look like unedited template placeholders, so they are **not** evidence that
`foscsmlprd02` actually has one A100 — but the stanzas are excellent leverage
for an IT request, because they show the admin has already been down this road.

### 1.3 No federation

```
$ scontrol show federation
                                   <-- empty output, no federation configured

$ sacctmgr show federation
Federation    Cluster ID             Features     FedState
---------- ---------- -- -------------------- ------------
                                   <-- header only, zero rows
```

### 1.4 No multi-cluster registration

```
$ sacctmgr show clusters
   Cluster     ControlHost  ControlPort   RPC     Share GrpJobs ...
machine-l+       127.0.0.1         6817 11264         1

$ sinfo -M all
CLUSTER: machine-learning-group
PARTITION       AVAIL  TIMELIMIT  NODES  STATE NODELIST
slurmpartition*    up   infinite      1    mix foscsmlprd01

$ sinfo -M foscsmlprd02
sinfo: error: No cluster 'foscsmlprd02' known by database.
sinfo: error: 'foscsmlprd02' can't be reached now, or it is an invalid entry
       for --cluster.  Use 'sacctmgr list clusters' to see available clusters.
sinfo: fatal: Could not get cluster information
```

Exactly one cluster is registered, and its `ControlHost` is `127.0.0.1`. The
`-M` flag *is* compiled in and accepted syntactically, but there is nothing for
it to route to.

### 1.5 Consequence for node-targeting flags

Because the cluster has one node in one partition with no `AvailableFeatures`
(`scontrol show node foscsmlprd01` → `AvailableFeatures=(null)`), there is
**nothing for `--partition`, `--nodelist`, `--constraint` or `--exclude` to
select**. This is not a targeting problem with an easy answer; there is no
second target.

---

## 2. Are 02 and 03 separate SLURM clusters, or not SLURM at all?

**Could not determine.** SSH authentication is refused (§4), and I did not probe
service ports, per the read-only remit. What is established:

- They exist and resolve in DNS.
- They are in the same hostname series and the same `/24` as this box.
- They are network-reachable on SSH (a full SSH transport handshake completed).
- Nothing on this host reveals whether they run `slurmd`, `slurmctld`, or any
  scheduler, and nothing reveals whether they have GPUs.

**This is the single most important open unknown.** If `foscsmlprd02`/`03` are
storage, database or web servers, the entire plan is void. The user should
confirm with IT *before* investing in any of the routes below.

```
$ getent hosts foscsmlprd01 ; getent hosts foscsmlprd02 ; getent hosts foscsmlprd03
130.216.4.218   foscsmlprd01.its.auckland.ac.nz foscsmlprd01
130.216.4.219   foscsmlprd02.its.auckland.ac.nz
130.216.4.217   foscsmlprd03.its.auckland.ac.nz

$ getent hosts foscsmlprd04
NXDOMAIN (no 04)
```

The hostnames the team lead guessed are confirmed correct, and there are exactly
three. Note `/etc/hosts` (Puppet-managed) pins **only** `foscsmlprd01`; 02 and 03
come from DNS (`nameserver 130.216.191.1`, search domain `its.auckland.ac.nz`).

Both hosts are also present in the site-wide `/etc/ssh/ssh_known_hosts`
(731 lines, Puppet-managed):

```
$ grep -o 'foscsml[a-z0-9.-]*' /etc/ssh/ssh_known_hosts | sort -u
foscsmlprd01
foscsmlprd01.its.auckland.ac.nz
foscsmlprd02
foscsmlprd02.its.auckland.ac.nz
foscsmlprd03
foscsmlprd03.its.auckland.ac.nz
```

They are managed by the same Puppet infrastructure as this box. That makes it
likely they are siblings of some kind, but it does not establish GPUs.

---

## 3. Filesystem — the plan's blocking defect

**`/data` is node-local storage. It is not shared, and nothing on this host is.**

```
$ df -hT /data/jehc223
Filesystem            Type  Size  Used Avail Use% Mounted on
/dev/mapper/data-data xfs    14T   13T  1.8T  88% /data

$ df -hT $HOME          # HOME=/data/jehc223/home
/dev/mapper/data-data xfs    14T   13T  1.8T  88% /data

$ stat -f -c '%n : %T' /data/jehc223 $HOME /data
/data/jehc223 : xfs
/data/jehc223/home : xfs
/data : xfs

$ mount | grep -E 'nfs|gpfs|lustre|beegfs|cifs|smb|ceph'
                                   <-- zero matches
```

`/dev/mapper/data-data` is an LVM logical volume on local disk. `stat -f` reports
`xfs`, not a network filesystem type. Confirmed against `/etc/fstab`, which
contains **no** network mounts at all:

```
/dev/mapper/rhel-root   /       xfs  defaults        0 0
/dev/mapper/data-data   /data   xfs  defaults,usrquota        0 0
/dev/mapper/rhel-home   /home   xfs  defaults,usrquota,nodev,nosuid        0 0
/dev/mapper/rhel-tmp    /tmp    xfs  ...
/dev/mapper/rhel-var    /var    xfs  ...
```

There is no autofs either (`/etc/auto.master`: *No such file or directory*), no
`/scratch`, `/shared`, `/gpfs` or `/lustre`, and `nfs-client.target` is
`inactive`. `/data` hosts the home directories of ~312 users on this one box.

**Therefore: a job running on `foscsmlprd02` would not see `/data/jehc223` at
all.** The user's stated end state — "compute on 02/03, all assets stay on 01" —
is impossible without either a network filesystem or explicit data staging.

### 3.1 Quota headroom (relevant to any staging plan)

```
$ quota -s
Disk quotas for user jehc223 (uid 135258174):
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
/dev/mapper/data-data
                   288G    290G   3000G           1235k       0       0
/dev/mapper/rhel-home
                    36K  92160K    100M              13       0       0
```

The user is at **288G of a 290G soft quota** (3000G hard). Note the real `/home`
partition has a 100 MB hard limit — `$HOME` is redirected to `/data/jehc223/home`
precisely because `/home` is unusable for data. Any staging plan on 02/03 needs
a quota grant there too.

### 3.2 What would have to move (measured)

Minimum working set to run one RGCL GPU training job on a foreign host:

| Component | Size | Note |
|---|---|---|
| conda env `HateVideo` | 7.2 GiB | `/data/jehc223/miniconda3/envs/HateVideo` |
| `RGCL/data` feature caches | 4.85 GiB | CLIP_Embedding 2.1G, lora_frames 2.6G, audio 247M, ASR 27M |
| `RGCL/artifacts` | 1.97 GiB | sav_f0 1.7G dominates |
| `RGCL` src + scripts + configs | 0.16 GiB | |
| **Core subtotal** | **≈ 14.2 GiB** | one-time stage, then rsync deltas |
| HF hub cache | 30.2 GiB | only if the job instantiates an encoder from cache |
| `models/Molmo2-8B-bf16` | 16.1 GiB | only for Molmo2 work |
| **Worst case** | **≈ 60 GiB** | |

Raw video datasets — `Multihateclip` 27G, `HateMM` 9.6G, `HateClipSeg` 4.2G —
would **not** need to move for feature-space training, which is the relevant
regime for almost all current work.

### 3.3 Data-boundary check

`CLAUDE.md`'s hard rule is scoped to cloud: *"原始视频永不上云"* — raw videos
never go to the **cloud**, enforced by the hard block in
`scripts/cloud/modal_probe_runner.py`. `foscsmlprd02`/`03` are
university-owned, on-premises, same subnet, same Puppet estate — so this rule as
written does not prohibit staging to them. But the *spirit* of the rule and the
sbatch scripts' own convention (`apx_egemaps_extract.sbatch`: *"Raw videos stay
LOCAL; features-only output"*) both point the same way, and §3.2 shows raw video
does not need to move anyway. **Recommendation: stage derived features only,
keep raw video on 01.** That preserves the boundary and cuts the transfer by 40G.

---

## 4. Access to 02 / 03 — reachable, but no credentials

Both hosts complete a full SSH transport handshake, then reject us at auth:

```
$ ssh -o BatchMode=yes -o ConnectTimeout=5 -v foscsmlprd02.its.auckland.ac.nz hostname
debug1: Host 'foscsmlprd02.its.auckland.ac.nz' is known and matches the RSA host key.
debug1: Found key in /etc/ssh/ssh_known_hosts:643
...
Username accounts log in using your Token
debug1: Authentications that can continue: publickey,gssapi-keyex,gssapi-with-mic,password,keyboard-interactive
debug1: Next authentication method: gssapi-with-mic
debug1: Unspecified GSS failure.  Minor code may provide more information
No Kerberos credentials available (default cache: FILE:/tmp/krb5cc_135258174)
debug1: Next authentication method: publickey
debug1: Trying private key: /home/jehc223/.ssh/id_rsa
debug1: Trying private key: /home/jehc223/.ssh/id_dsa
debug1: No more authentication methods to try.
jehc223@foscsmlprd02.its.auckland.ac.nz: Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password,keyboard-interactive).
```

`foscsmlprd03` behaves identically (different host-key algorithms and a newer
OpenSSH, so the two boxes are not clones of each other).

Reading this precisely:

- **Network path is open.** TCP connect, key exchange and `SSH2_MSG_NEWKEYS` all
  succeeded. Not firewalled, not filtered. The squid proxy
  (`http_proxy=http://squid.auckland.ac.nz:3128`) is irrelevant here — SSH does
  not consult `http_proxy`, and these are direct on-campus connections.
- **We have no credentials of any kind.** `~/.ssh` contains only `known_hosts`:

  ```
  $ ls -la $HOME/.ssh
  -rw-r-----.  1 jehc223 5066  978 Aug 15  2025 known_hosts
  $ cat $HOME/.ssh/config
  cat: /data/jehc223/home/.ssh/config: No such file or directory
  ```

  No keypair exists, and no `~/.ssh/config`.
- **A path-mismatch trap.** SSH looked for keys in `/home/jehc223/.ssh/id_rsa`,
  because the passwd entry says `/home/jehc223`:

  ```
  $ getent passwd jehc223
  jehc223:x:135258174:5179:Junyi Chen:/home/jehc223:/bin/bash
  ```

  but the shell environment sets `HOME=/data/jehc223/home`. **Any key the user
  generates into `$HOME/.ssh` will be invisible to `ssh` unless they pass
  `-i` explicitly or write a `~/.ssh/config` with `IdentityFile`.** This will
  silently waste an afternoon if not anticipated. And `/home` has a 100 MB hard
  quota, so putting a key there is fine but nothing else.
- **Kerberos is a live alternative.** `gssapi-with-mic` is offered by both hosts
  and the realm is configured (`/etc/krb5.conf`: `default_realm =
  UOA.AUCKLAND.AC.NZ`, `kdc = krb5-prioritised.uoa.auckland.ac.nz:88`,
  `forwardable = true`). A `kinit` would very likely enable passwordless SSH to
  02/03 **if the account exists there**. This needs the user's own
  password/token, so it is theirs to try, not mine.
- The banner *"Username accounts log in using your Token"* indicates
  password/interactive login is token-gated (2FA). That is fine for a human
  logging in, but it means unattended `rsync`/remote-`sbatch` needs a key or a
  Kerberos ticket, not a password.

**Whether account `jehc223` even exists on 02/03 could not be determined.** The
`Permission denied` message is identical for "no such user" and "user exists,
no credentials", by design.

---

## 5. Load visibility across servers

**None exists today.** `sinfo -M all` and `squeue -M all` both return only
`machine-learning-group`. Without SSH or a shared `slurmdbd`, there is no way to
observe GPU state on 02/03 from here. The user's desired workflow — "submit to
whichever server has the most free GPUs" — has **no observation mechanism**, let
alone a submission mechanism.

What would provide it, cheapest first:
1. An SSH account on 02/03 → `ssh foscsmlprd02 nvidia-smi` / `sinfo`, wrappable
   in a 5-line poll script.
2. Registering the clusters in a shared `slurmdbd` → `sinfo -M all` works natively.
3. Adding 02/03 as nodes to *this* cluster → plain `sinfo` shows everything.

Local state at time of recon, for reference:

```
$ scontrol show node foscsmlprd01
   CfgTRES=cpu=256,mem=1000000M,billing=256,gres/gpu=8
   AllocTRES=cpu=24,mem=336G,gres/gpu=7
   State=MIXED

$ nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
0, NVIDIA A100-SXM4-80GB, 66147 MiB, 81920 MiB, 99 %
1, NVIDIA A100-SXM4-80GB, 77105 MiB, 81920 MiB, 0 %
2, NVIDIA A100-SXM4-80GB, 50803 MiB, 81920 MiB, 87 %
3, NVIDIA A100-SXM4-80GB, 57421 MiB, 81920 MiB, 99 %
4, NVIDIA A100-SXM4-80GB, 76651 MiB, 81920 MiB, 0 %
5, NVIDIA A100-SXM4-80GB, 0 MiB, 81920 MiB, 0 %
6, NVIDIA A100-SXM4-80GB, 6313 MiB, 81920 MiB, 39 %
7, NVIDIA A100-SXM4-80GB, 6247 MiB, 81920 MiB, 40 %
```

Eight A100-80GB, seven allocated, **GPU 5 completely idle**.

---

## 6. The finding that reframes the whole problem

**The queue wait is administrative approval, not GPU scarcity.**

`/etc/slurm/slurm.conf` sets `JobSubmitPlugins=lua`. The plugin is
`/etc/slurm/job_submit.lua` (world-readable), and it holds **every** job
unconditionally:

```lua
APPROVAL_MARKER = "PENDING_APPROVAL"

function slurm_job_submit(job_desc, part_list, submit_uid)
    job_desc.priority     = 0                 -- hold the job
    job_desc.admin_comment = APPROVAL_MARKER  -- tamper-proof marker
    slurm.log_info("job_submit: job from uid=%u held pending approval", submit_uid)
    return slurm.SUCCESS
end
```

The header comment states the release path explicitly:

```
-- Release path (admin only):
--   approve_jobs.py runs as root → job.release() → modify_uid == 0
--   → slurm_job_modify allows it.
```

`slurm_job_modify` rejects every non-root modification while the marker is
present, so the hold genuinely cannot be lifted by the user. The approver lives
in `/opt/csml-approval-agent` (`drwxr-x--- root root`, *Permission denied* — I
could not read it).

The direct evidence that this, not GPU supply, is the bottleneck: at recon time
**GPU 5 was completely idle and `AllocTRES` showed 7 of 8 GPUs used, while 15
jobs sat in `PENDING (JobHeldUser)`** — including three of the user's colleague's
`cap_msA_*` jobs each asking for a single GPU. A free GPU and held single-GPU
jobs coexisted. Nothing about adding servers 02/03 changes that, **if the same
approval agent gates them**.

Corollary finding: **the "16 CPU / 128 GB / 2 GPU" cap is not enforced anywhere
in readable SLURM configuration.**

```
$ scontrol show config | grep AccountingStorageEnforce
AccountingStorageEnforce = none

$ sacctmgr show qos format=Name,Priority,GrpTRES,MaxTRESPU,MaxJobsPU,MaxSubmitPU,Flags
      Name   Priority       GrpTRES     MaxTRESPU MaxJobsPU MaxSubmitPU                Flags
    normal          0

$ sacctmgr show assoc user=jehc223 format=Cluster,Account,User,Partition,GrpTRES,MaxTRES,MaxJobs,MaxSubmit,QOS
   Cluster    Account       User  Partition       GrpTRES       MaxTRES MaxJobs MaxSubmit                  QOS
                                   <-- header only, zero rows
```

Accounting enforcement is off, the `normal` QOS carries no limits, and the user
has no association limits. The only partition-level caps are per *node*, not per
user: `MaxMemPerNode=512000 MaxCPUsPerNode=128`. So the 16/128/2 figure is a
**site policy applied at approval time by the root-owned agent** — which I could
not read to verify. It is a human decision, and human decisions can be
renegotiated far more cheaply than filesystems can be re-architected.

---

## 7. Routes, assessed

| # | Route | Status | Blocker |
|---|---|---|---|
| R1 | `sbatch -M <cluster>` multi-cluster | **Dead today** | Only one cluster in `slurmdbd`; `AccountingStorageHost=localhost` |
| R2 | SLURM federation | **Dead today** | `scontrol show federation` empty; `FederationParameters=(null)` |
| R3 | Add 02/03 as nodes to *this* cluster | **Cleanest fix; admin-only** | Needs slurmd + munge key + ports 6817/6818 + **shared storage** |
| R4 | SSH + remote `sbatch` / rsync staging | **Plausible; needs account** | No credentials on 02/03; unknown whether they run SLURM or have GPUs |
| R5 | Site portal / OnDemand / k8s / submit host | **Does not exist** | See below |
| R6 | Raise the per-user cap / speed approval on 01 | **Best value, no infra work** | Pure policy ask |
| R7 | Modal cloud probes | **Already in use** | Triage-only by project rule (~1.4pt drift) |

**On R5** — there is no site HPC scaffolding of any kind:

```
$ module avail
------------------------ /usr/share/Modules/modulefiles ------------------------
dot  module-git  module-info  modules  null  use.own
```

Stock environment-modules with nothing installed — no Lmod, no EasyBuild, no
software stack. `/opt` holds only vendor agents (`azcmagent`, `nessus_agent`,
`puppetlabs`, `tenable`, `microsoft`, `nvidia`) plus `csml-approval-agent` and
`nvitop-thing`. `/etc/motd` is a Puppet-generated hardware banner with no user
documentation, and `/usr/local` has no site tooling. **No portal, no OnDemand,
no submit host, no documentation.** This is a lab box with SLURM bolted on, not
a managed HPC service.

---

## 8. What the user should do

### 8.1 Works right now, zero admin involvement

Nothing enables cross-server submission today. What *is* available:

- **Modal cloud probes**, already the project standard for triage — the existing
  and correct pressure valve. Unchanged by this recon.
- **Batch harder on 01.** Only 24 of 256 CPUs and 7 of 8 GPUs were allocated;
  CPU-only stages (`apx_egemaps_extract.sbatch`-style) contend with nobody. The
  scarce resource is approval events, not cores — so fewer, larger, multi-stage
  jobs beat many small ones under an approval-gated queue.

### 8.2 The user can do themselves

1. **Ask a colleague or the group's admin one question first:** *do
   `foscsmlprd02`/`03` have GPUs, and do I have an account?* This is a 30-second
   question that determines whether anything below is worth doing. Do not build
   before this is answered.
2. **Try Kerberos:** `kinit jehc223@UOA.AUCKLAND.AC.NZ` then
   `ssh foscsmlprd02.its.auckland.ac.nz hostname`. Both hosts offer
   `gssapi-with-mic` and the realm is configured. If an account exists, this
   works immediately with no admin involvement.
3. **Generate an SSH key — and mind the HOME trap.** `ssh` reads
   `/home/jehc223/.ssh` (the passwd home), *not* `$HOME=/data/jehc223/home`.
   Either put the key in `/home/jehc223/.ssh/` (that partition has 100 MB — ample
   for a key) or write `/home/jehc223/.ssh/config`:

   ```
   Host foscsmlprd02 foscsmlprd03
       HostName %h.its.auckland.ac.nz
       User jehc223
       IdentityFile /data/jehc223/home/.ssh/id_ed25519
       BatchMode yes
   ```

   Installing the public key on 02/03 still requires being able to log in there
   once (via token/Kerberos) or an admin doing it.

### 8.3 Requires IT / admin action

In priority order:

- **(A) Confirm what 02/03 are** and grant accounts if they are compute.
- **(B) Shared filesystem** — the true prerequisite. Either export `/data` from
  01 over NFS to 02/03, or provision a shared volume mounted at the same path on
  all three. **Without this, nothing else delivers the user's stated goal.**
- **(C) Add 02/03 as nodes to the existing `machine-learning-group` cluster**
  (the commented stanzas at `slurm.conf:150` and in `gres.conf` show this was
  already contemplated). Cleaner than federation or multi-cluster: one
  `slurmctld`, one `slurmdbd`, one `squeue`, and node-targeting flags start
  working. Requires slurmd, munge key distribution, and ports 6817/6818.
- **(D) If (B) is refused**, ask instead for the per-user cap on 01 to be raised
  or for a fast-approval lane for short jobs. Per §6 this is likely to deliver
  more throughput than (A)+(B)+(C) combined, at zero infrastructure cost.

### 8.4 Draft email to HPC/IT support

> **Subject:** foscsmlprd02/03 — GPU access and shared storage for the CSML SLURM cluster
>
> Hi,
>
> I'm a research student using `foscsmlprd01` under the `machine-learning-group`
> SLURM cluster (account `jehc223`, group `CSML_users`). GPU contention on 01 is
> currently the limiting factor on my project, and I'd like to ask about
> `foscsmlprd02` and `foscsmlprd03`.
>
> Three questions:
>
> 1. **Do `foscsmlprd02`/`03` have GPUs, and could I be given an account on
>    them?** I can currently reach both over SSH but have no credentials there.
>
> 2. **Could they be added as compute nodes to the existing
>    `machine-learning-group` cluster?** I noticed `/etc/slurm/slurm.conf` and
>    `/etc/slurm/gres.conf` already contain commented-out `foscsmlprd02` node
>    entries, so this may already have been considered. Adding them as nodes to
>    the one cluster would be simpler for users than SLURM multi-cluster or
>    federation — it needs no `slurmdbd` changes, and jobs could then be targeted
>    with ordinary `--nodelist`/`--constraint` flags.
>
> 3. **Would shared storage across the three servers be possible?** `/data` is
>    currently a local XFS volume on 01, so a job scheduled onto another node
>    would not see my datasets or environments. Either an NFS export of `/data`
>    from 01, or a shared volume mounted at the same path on all three, would
>    make multi-node scheduling work. Without it, adding nodes wouldn't help me,
>    since I'd have to copy ~15 GB of feature caches and conda environments to
>    each node for every job.
>
> If extending the cluster isn't feasible, a smaller alternative would help a
> lot: my current per-user limit is 2 GPUs, and jobs are held pending approval
> (I've waited around 8 hours today). I've seen jobs held while a GPU sat idle,
> so if the approval step could be made faster for short jobs — or the per-user
> GPU limit raised — that alone would resolve most of my throughput problem.
>
> Happy to provide job IDs or any other detail that helps.
>
> Thanks very much,
> Junyi Chen (`jehc223`)

---

## 9. Honest assessment of the user's plan

> *"Submit jobs to whichever server currently has the most free GPUs, while all
> assets stay on THIS server."*

**Achievable: no, not in its stated form, and not by any user-side action.**

The second clause is the killer. "Assets stay on 01" only works if 02/03 can
*read* 01's disk, which requires a network filesystem that does not exist and
that only IT can create. Every route reduces to one of two shapes:

- **IT exports shared storage** → then adding 02/03 as nodes (§8.3-C) makes the
  plan work exactly as the user imagines, with plain `sbatch`. Cost: entirely
  IT's, and a real infrastructure change they may decline.
- **No shared storage** → the plan becomes "assets are *mirrored* to 02/03",
  ~14 GiB of one-time staging plus rsync deltas, needing an account there and a
  quota grant. Workable, but it is a different plan, and it forfeits the
  single-source-of-truth property the project's reproducibility discipline
  (G-repro, frozen hashes, 4dp comparisons) depends on. Two copies of a feature
  cache is exactly the class of drift this project has spent months guarding
  against — it would need the same hash-verification ceremony applied to the
  mirror.

And both shapes are gated on an unverified premise: **that 02/03 have GPUs at
all.** No evidence on this host establishes that.

**Recommended sequence:** ask the one-line question in §8.2-1 first. If the
answer is "no GPUs" or "no account", the whole line is dead and costs nothing.
If it is "yes", send the §8.4 email and pursue (B)+(C) together, never (C) alone.
Meanwhile, treat §8.3-D as the higher-expected-value ask — §6 shows the project
is losing hours to an approval gate while a GPU sits idle, and that is a policy
problem, not a hardware one.

---

## 10. Limits of this recon

Stated explicitly, per project evidence discipline:

- **Could not determine** whether `foscsmlprd02`/`03` run SLURM. SSH auth
  refused; I did not probe service ports.
- **Could not determine** whether they have GPUs. The commented `gres.conf` line
  naming `a100` is a template placeholder, not evidence.
- **Could not determine** whether account `jehc223` exists on them. `Permission
  denied` is indistinguishable from "no such user".
- **Could not read** `/opt/csml-approval-agent` (root-only), so the 16/128/2 cap
  and the approval policy are inferred from `job_submit.lua` plus the absence of
  any SLURM-side enforcement, not read directly.
- **Could not read** `/etc/slurm/slurmdbd.conf` (mode `0600`, owner `slurm`) or
  `/usr/local/slurm_prolog.sh` (mode `0750`, owner `root`).
- Zero jobs submitted, cancelled, held or released. Zero configuration changed.
  Zero writes to any host other than this file.
