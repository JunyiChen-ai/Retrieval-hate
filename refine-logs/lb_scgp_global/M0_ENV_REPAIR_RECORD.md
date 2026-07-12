# M0 环境修复记录 — lb_scgp_global Run2-v2 依赖修复与全依赖审计

- **日期**: 2026-07-13
- **操作者**: Claude Opus 4.8(env-repair 角色)
- **范围**: 修复 `HateVideo` conda 环境缺失的 `jsonschema`;对 `lb_scgp_global_r2_run2_v2_*` 谱系做全依赖 + 外部命令 + 环境类前置检查审计。
- **纪律**: 未改动任何代码/配置/schema 文件;未提交任何 SLURM 作业。仅在登录节点做 pip 安装与 import 验证。
- **HateVideo 解释器**: `/data/jehc223/miniconda3/envs/HateVideo/bin/python` (Python 3.11.8)

---

## 1. 缺失确认证据

用 **HateVideo 的绝对解释器**(而非模糊的 `python`)确认:

```
$ /data/jehc223/miniconda3/envs/HateVideo/bin/python -c "import jsonschema"
ModuleNotFoundError: No module named 'jsonschema'
```

确认缺失。`jsonschema` 是硬依赖:
- `..._common.py:182`  `from jsonschema import Draft7Validator, RefResolver`(schema 校验,函数内惰性 import,失败即 `RuntimeError("jsonschema dependency unavailable; refusing to validate Run2-v2 payload")`)
- `..._independent_verify.py:167` 同款惰性 import,失败即 `RuntimeError("...independent verifier refuses PASS")`
- `..._validate.py:101` 预检 `python_dependency_check()` 仅做 `importlib.util.find_spec('jsonschema')` 存在性检查

## 2. 关键陷阱(这正是"谱系再死于缺包"的真正机理)

**`source activate HateVideo` 在本 shell 里静默失效**,当前活动环境其实是 **ExMRD (py3.12)**:

```
$ source activate HateVideo 2>/dev/null; echo $CONDA_PREFIX; which python; which pip
CONDA_PREFIX=/data/jehc223/miniconda3/envs/ExMRD
/data/jehc223/miniconda3/envs/ExMRD/bin/python
/data/jehc223/miniconda3/envs/ExMRD/bin/pip
```

后果:一次天真的 `pip install jsonschema` 会**装到错误的环境(ExMRD)**——下载的是 `rpds_py ... cp312` 轮子、"already satisfied" 指向 ExMRD 的 `python3.12/site-packages`——而 **HateVideo 依旧缺包**。我第一次的 `pip install` 正是掉进这个坑(装进了 ExMRD),已按下文回滚。

**正确做法(已采用)**:先 `source /data/jehc223/miniconda3/etc/profile.d/conda.sh` 再 `conda activate HateVideo`,或直接用 HateVideo 绝对解释器 `python -m pip`。**幸而 `require_slurm_run2()`(common.py:288)硬性检查 `CONDA_DEFAULT_ENV == "HateVideo"`,且 sbatch 第 12–13 行 `source conda.sh; conda activate HateVideo`——所以只要走 sbatch,`python` 与 `CONDA_DEFAULT_ENV` 都会正确落在 HateVideo,本次修复恰好落在 gate 所强制的那个环境里。**

## 3. 安装命令与输出摘录

用 HateVideo 绝对解释器安装(ABI 正确,cp311 轮子):

```
$ /data/jehc223/miniconda3/envs/HateVideo/bin/python -m pip install jsonschema
Collecting rpds-py>=0.25.0 (from jsonschema)
  Downloading rpds_py-2026.6.3-cp311-cp311-manylinux_2_17_x86_64...whl   # 注意: cp311, 匹配 py3.11
Requirement already satisfied: attrs>=22.2.0 in .../HateVideo/lib/python3.11/site-packages (26.1.0)
Requirement already satisfied: typing-extensions>=4.4.0 in .../HateVideo/lib/python3.11/site-packages (4.15.0)
Successfully installed jsonschema-4.26.0 jsonschema-specifications-2025.9.1 referencing-0.37.0 rpds-py-2026.6.3

$ /data/jehc223/miniconda3/envs/HateVideo/bin/python -c "import jsonschema, importlib.metadata as m; print(m.version('jsonschema'), jsonschema.__file__)"
4.26.0  /data/jehc223/miniconda3/envs/HateVideo/lib/python3.11/site-packages/jsonschema/__init__.py
```

**安装的 jsonschema 版本: 4.26.0**(依赖树见下)。集群有外网,pip 直接联网安装成功,未用本地 wheel。

**回滚 ExMRD 误装**(恢复该环境原状,只删本次新增的 4 个包,attrs/typing-extensions 是既有依赖未动):

```
$ /data/jehc223/miniconda3/envs/ExMRD/bin/python -m pip uninstall -y jsonschema jsonschema-specifications referencing rpds-py
Successfully uninstalled jsonschema-4.26.0 / -specifications-2025.9.1 / referencing-0.37.0 / rpds-py-2026.6.3
$ /data/jehc223/miniconda3/envs/ExMRD/bin/python -c "import importlib.util as u; print(u.find_spec('jsonschema') is not None)"
False   # ExMRD 已复原
```

## 4. 全依赖清单表(HateVideo 逐一验证)

第三方包用 HateVideo 绝对解释器 `import` 验证;标准库标注 stdlib 免验。

| 包 / 模块 | 类型 | 来源文件:行 | HateVideo 验证结果 |
|---|---|---|---|
| `numpy` (as np) | 第三方 | common:19 / producer:16 / independent_verify:20 | **OK 1.26.4** |
| `jsonschema` (`Draft7Validator`, `RefResolver`) | 第三方(本次修复) | common:182 / independent_verify:167(惰性);validate:101(存在性检查) | **OK 4.26.0**(见 §6 弃用告警) |
| `jsonschema.exceptions.SchemaError` | 第三方(随上) | common:183 / independent_verify:168 | **OK** |
| `jsonschema-specifications` | 传递依赖 | (jsonschema 拉入) | **OK 2025.9.1** |
| `referencing` | 传递依赖 | (jsonschema 拉入) | **OK 0.37.0** |
| `rpds-py` | 传递依赖(C 扩展,ABI 敏感) | (jsonschema 拉入) | **OK 2026.6.3 (cp311)** |
| `attrs` | 传递依赖(既有) | (jsonschema 拉入) | **OK 26.1.0** |
| `typing-extensions` | 传递依赖(既有) | (referencing 拉入) | **OK 4.15.0** |
| `lb_scgp_global_r2_run2_v2_common`(本地模块) | 本地 | producer:21 / validate:15(经 sys.path.insert) | **import OK**(`canonical_json` 等符号在) |
| `__future__` `hashlib` `json` `math` `os` `subprocess` `tempfile` `pathlib` `typing` `argparse` `copy` `sys` `importlib.util` | stdlib | 各文件顶部 | stdlib 免验 |

**审计结论: 唯一缺失包就是 `jsonschema`,未发现第二个缺失包。** numpy 早已存在;jsonschema 的全部传递依赖(含 ABI 敏感的 rpds-py cp311)均已就位。

补充稳健性验证(HateVideo 解释器):
- `python -m py_compile` 4 个 .py 文件 → **全部 OK**
- 惰性 import 精确复现 `from jsonschema import Draft7Validator, RefResolver; from jsonschema.exceptions import SchemaError` → **运行期 OK**

## 5. 外部命令清单(`command -v`)

| 命令 | 用途 / 来源 | 路径 |
|---|---|---|
| `jq` | wrapper:48/49/75;validate.py:144 预检 | `/usr/bin/jq` ✅ |
| `git` | validate.py:150 `git diff --check`;common `relevant_git_status` `git status` | `/usr/bin/git` ✅ |
| `bash` | wrapper shebang;validate.py:147 `bash -n` 语法检查 | `/usr/bin/bash` ✅ |
| `diff` | validate.py:150 `git diff --check`(git 内建调用) | `/usr/bin/diff` ✅ |
| `mktemp` | wrapper:59 建临时 validation json | `/usr/bin/mktemp` ✅ |
| `rm` | wrapper cleanup 清理 | `/usr/bin/rm` ✅ |
| `sbatch` | 作业提交 | `/usr/bin/sbatch` ✅ |
| `sha256sum` | **存在但非硬依赖**:脚本内 sha256 走 Python `hashlib`(`sha256_file`),源码里的 `"sha256"` 皆为 dict 键;仍验证其存在 | `/usr/bin/sha256sum` ✅ |
| `python` | wrapper/sbatch 内裸调用;validate 子进程用 `sys.executable`(自解释器,不受 PATH 影响) | 经 sbatch `conda activate HateVideo` 解析到 HateVideo(本 shell 未激活时裸 `python`=ExMRD,见 §2 陷阱) |

外部命令**无缺失**。

## 6. 其它"环境类"前置检查(validate.py)逐项确认

`require_slurm_run2()` 会拦截登录节点直跑,故不做端到端 dry-run(设计如此),改为逐项静态/手动确认其当前会过:

| validate.py 检查 | 类别 | 当前结论 |
|---|---|---|
| `require_slurm_run2()`: `SLURM_JOB_ID` 存在 + `CONDA_DEFAULT_ENV==HateVideo` + CPU=8/MEM=64G/GPU=0 | 环境变量 | sbatch 已设 `#SBATCH --cpus-per-task=8 --mem=64G`(无 GPU)且第 13 行 `conda activate HateVideo` → **SLURM 下会 PASS**(登录节点按设计拒跑) |
| `jq -e '.'` on config + EXPERIMENT_PLAN.machine.json + 3 schema + v1 contract_freeze | 输入文件存在 & 合法 JSON | config/payload/case/cert schema + v1 contract_freeze 全部 **jq OK** |
| `bash -n` wrapper + sbatch | 文件存在 & 语法 | 文件在;`bash -n` 由 validate 执行 |
| `py_compile` 4 个 .py | 语法 | **PASS**(§4) |
| `git diff --check` on 实现文件 + tracker | git 状态 / 尾随空白 | **clean(PASS)** |
| `scan_trailing_whitespace` | 尾随空白 | 与 git diff --check clean 一致 |
| `verify_run1_hashes` run1_frozen(10 文件) | 数据完整性 | 10/10 sha256 **全部匹配(PASS)** |
| `verify_run1_hashes` old_protected 快照(278 路径 manifest) | 数据完整性 | 未手动复现(由 `old_protected_hash_manifest()` 计算,非环境类,超出本次范围) |
| `no_clobber_check`:输出 artifact + .publish.lock 不得已存在 | 目录/状态 | 输出目录 `artifacts/lb_scgp_global/v2/m0/synth_kkt/` **不存在 → PASS**(无残留) |
| `resource_and_run_check`:config 字段匹配 | 配置一致性 | run_id / schema_id / artifact_path / `slurm={cpu:8,ram_gb:64,gpu:0,env:HateVideo,no_time_flag:true}` / `authorization.authorized_run_ids==[RUN2]` 经比对 **全部匹配** |

**结论:除 jsonschema(已修)外,所有环境类前置检查在 SLURM 下当前均会通过;无第二处会挡住 v2 的环境类问题。**

## 7. 前瞻性风险(未修,仅记录 — 本次不改代码)

`jsonschema.RefResolver` 自 **v4.18.0 起弃用**,当前 4.26.0 仍可 import 但会打印 `DeprecationWarning`("A future release will remove RefResolver")。含义:
- **默认告警设置下**:import 成功,仅 stderr 一行告警,不影响功能。
- **若设 `PYTHONWARNINGS=error` / `-W error`**:该告警升为异常 → 被 common:184 的 `except Exception` 捕获 → 触发 `RuntimeError("jsonschema dependency unavailable...")` → **fail-closed(拒绝校验,而非产出脏结果)**。当前 sbatch 未设 `PYTHONWARNINGS`,故无此风险。
- **若日后把 HateVideo 的 jsonschema 升级到移除 RefResolver 的版本**:惰性 import 直接 `ImportError` → 同样 fail-closed。

**建议(交用户/团队决定,本次不动代码)**:把 jsonschema 固定在含 RefResolver 的版本(如 `jsonschema<5` 或记录 `==4.26.0`),或后续将代码迁移到 `referencing` 库以彻底去除 RefResolver 依赖。

## 8. 变更足迹小结

- HateVideo:新增 `jsonschema 4.26.0` + 传递依赖(`jsonschema-specifications 2025.9.1`、`referencing 0.37.0`、`rpds-py 2026.6.3`;`attrs`/`typing-extensions` 既有未动)。
- ExMRD:第一次误装已**完整回滚**,恢复原状。
- 代码 / 配置 / schema / SLURM:**零改动、零提交**(`git status` 仅见既有 untracked 文件,无本人产生的改动)。
