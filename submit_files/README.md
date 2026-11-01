# TH-HPC6 5090 提交脚本

本目录由 GitHub 管理。脚本面向 `/fs6/home/zhongrsh/workspace/UnifiedGroupGen`，使用个人 Conda 环境 `unifiedgroupgen`、`GCC/12.2.0` 和 `CUDA/12.9`。

当前环境是 PyTorch `2.7.1+cu128`，四个 PyG 扩展在服务器 glibc 2.28 上用 CUDA 12.9 编译。不要直接用原始 CUDA 11.8 `environment.yml` 覆盖此环境。

## 上传 ZIP 后首次对齐

仅对已核对的服务器原始 `9d828670` 分支使用一次。先获取 GitHub 更新，再将对齐脚本读入检查目录：

```bash
(
set -e
module load proxy2/proxy2
cd /fs6/home/zhongrsh/workspace/UnifiedGroupGen
git fetch origin main
mkdir -p intermediate/checks/5090
git show origin/main:submit_files/align_uploaded_checkout.sh > intermediate/checks/5090/align_uploaded_checkout.sh
bash intermediate/checks/5090/align_uploaded_checkout.sh
)
```

脚本会先在同级 `UnifiedGroupGen_git_backup.*` 目录备份 `.git` 和 11 个已审计的差异文件，然后用 mixed reset 校准到 `6db52862`，只恢复这 11 个工作文件，再快进到 GitHub `main`。数据内容、`.env`、模型结果、Conda 环境和其他工作文件不由该恢复步骤覆盖；数据目录中仅更新 README。它不会执行 hard reset 或清理未跟踪文件。

## GPU 环境检查

`gpu_smoke.sbatch` 申请 1 张 5090、8 个 CPU 核，时限 10 分钟；通过 `gpu_smoke.py` 检查 FP32/BF16、四个 PyG 扩展及可微算子的梯度，不启动项目训练。

首次提交前创建日志目录，然后从项目根目录提交：

```bash
cd /fs6/home/zhongrsh/workspace/UnifiedGroupGen
mkdir -p intermediate/checks/5090
ugg_commit=$(git rev-parse HEAD)
UGG_GIT_COMMIT="$ugg_commit" yhbatch --export=ALL submit_files/gpu_smoke.sbatch
yhq
```

用返回的作业号替换 `JOBID` 查看日志：

```bash
tail -n 100 intermediate/checks/5090/gpu-smoke-JOBID.out
```

Git 提交号由登录节点通过环境变量传入，计算节点不需要安装 Git；直接提交而未传入时显示 `not recorded`，不阻止检查。作业会记录节点、提交时的 Git 提交号、Python 路径、GPU 和逐项检查结果。成功标志为 `ALL GPU CHECKS PASSED`。环境中的包不由作业脚本安装或升级。

## 后续更新

完成服务器首次 Git 基线校准后，按以下步骤更新；运行中的作业结束后再更新同一源码目录。

1. 本地修改并检查代码、配置和本目录脚本；`git add` 指定本次需要提交的路径，再提交、推送到 `main`。
2. 服务器执行 `git status --short --branch`，确认没有待保留的受跟踪文件改动，再在子 shell 中加载 `proxy2/proxy2` 并更新。
3. 查看更新的提交和脚本，使用 `yhbatch submit_files/<任务>.sbatch` 提交。

服务器 GitHub 访问已验证使用 `proxy2/proxy2`。更新命令如下，代理仅在子 shell 中生效：

```bash
(
set -e
module load proxy2/proxy2
cd /fs6/home/zhongrsh/workspace/UnifiedGroupGen
git pull --ff-only origin main
)
```

服务器自己的 `.env`、Conda 环境、`outputs/` 和 `intermediate/` 保留在服务器，不纳入脚本提交。仓库目前也跟踪部分训练数据：若以后在 GitHub 提交这些数据的变化，`git pull` 会尝试更新它们；只发布代码和脚本时，应明确选择提交路径。

本目录的检查不替代完整模型前后向、数据审计或正式训练验证。
