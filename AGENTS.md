# AGENT.md

## 项目
这是一个 **hateful video detection** 研究项目:把 RGCL / RA-HMD(原本用于 hateful meme detection)适配到仇恨视频检测。

## GPU / 资源
- **所有 GPU / 计算任务必须通过 SLURM 提交**(登录节点即计算节点,非 SLURM 的计算进程会被回收)。
- 环境:`conda activate HateVideo`。
- 提交:`sbatch scripts/slurm/<name>.sbatch`,**不要设 `--time`**。每用户上限:16 CPU / 128 GB / 2 GPU。
- 作业初始通常是 `PENDING (JobHeldUser)` → **等自动放行即可**,不要强行释放。

## 权责声明(最重要)
- **主对话 = 你和我讨论、决策、汇报**,主对话本身**不执行任何杂活**。
- **一切杂活**(写代码、数据处理、提交与监控 SLURM、调试、跑实验)**一律交给 subagent 或 dynamic workflow 去做**。开启subagent之后请等待它们的完成。
- **主对话调用 subagent 只能用 GPT-5.5 xhigh,**不得降级也不得升级**到其他模型。
