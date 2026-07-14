# CLAUDE.md

## 项目
这是一个 **hateful video detection** 研究项目:把 RGCL / RA-HMD(原本用于 hateful meme detection)适配到仇恨视频检测。

## GPU / 资源
- **所有 GPU / 计算任务必须通过 SLURM 提交**(登录节点即计算节点,非 SLURM 的计算进程会被回收)。
- 环境:`conda activate HateVideo`。
- 提交:`sbatch scripts/slurm/<name>.sbatch`,**不要设 `--time`**。每用户上限:16 CPU / 128 GB / 2 GPU。
- 作业初始通常是 `PENDING (JobHeldUser)` → **等自动放行即可**,不要强行释放。

## 云端探针(Modal)
- 所有探针 / 探索性 / 分诊实验(probing/triage)**一律上 Modal 云端跑**(`scripts/cloud/modal_probe_runner.py`,conda `HateVideo`,profile `jehc223`)——不要为这些排本机 SLURM 队列。
- 只有**正式验证**(预注册判决、3-seed 正式配对、要进论文表格的数字)才在本机 SLURM 排队用 GPU。
- 云端数字因跨硬件漂移(实测 ~1.4pt 量级,seed 噪声级)**只作分诊参考**,**永不**与本地数字混入同一张对比表;G-repro / 4dp 纪律只对本地数字有效。
- 数据边界:只允许上传派生特征缓存(`.pt` 浮点向量)与标签 JSON;**原始视频永不上云**(`modal_probe_runner.py` 的硬拦截不得移除)。
- 客户端依赖:squid 代理下 modal 需要 `python-socks[asyncio]` + `aiohttp-socks`(缺了会分别断 gRPC 控制面 / volume 上传面)——重建环境必装。
- **并行探针**:不同方向的 probing 可以(且应该)在云端同时多路并行——探针之间互不排他,不必等一个方向出结果再开下一个。
- **过线即排队**:某方向探针一过预注册杀开关,立即走正式流程(prereg→评审→冻结哈希)并**马上提交进本机 SLURM 队列排队**——排队时间与后续文书/其他探针并行,不留空转。仪式不减(单提纪律、独立评审照旧),但各阶段之间不留等待。
- **快杀**:kill 判定一出即关方向、立即换下一个候选;侦察(零 GPU forensic recon)对队列里每个候选提前做,保证弹药池永远有下一发。

## 权责声明(最重要)
- **主对话 = 你和我讨论、决策、汇报**,主对话本身**不执行任何杂活**。
- **一切杂活**(写代码、数据处理、提交与监控 SLURM、调试、跑实验)**一律交给 subagent 或 dynamic workflow 去做**。
- **主对话调用 subagent 只能用 Opus 4.8**(`model: opus`,即 `claude-opus-4-8`),**不得降级也不得升级**到其他模型。
