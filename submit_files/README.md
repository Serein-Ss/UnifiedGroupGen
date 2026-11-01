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

完成服务器首次 Git 基线校准后，按以下步骤更新。直接从工作目录运行的作业结束后再更新该目录的源码；下述 `hpc5090.py` 批次运行冻结快照，可以独立于后续代码拉取继续执行。

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

## 15 个旧模型评估和 4 个组成训练任务

入口为 `hpc5090.py`，调度脚本为 `hpc5090.sbatch`。所有作业仅申请 **5090 分区、1 GPU、8 CPU、64 GiB 内存**，不使用 CPU 分区，不安装或升级环境包。依据服务器手册的 `yhbatch`、`yhrun` 和 `--gres=gpu:1` 接口编写。

依赖和时限选项另参照 [Slurm sbatch 官方文档](https://slurm.schedmd.com/sbatch.html)。

四个新训练模型都没有性质目标，组成是独立输入：

| 实验名 | 训练数据 | source_domain | 组成训练 |
| --- | --- | --- | --- |
| mp20_u_comp_v2 | MP-20 | 无 | 50% 批内样本 |
| c2db51_u_comp_v2 | C2DB-51 | 无 | 50% 批内样本 |
| joint_u_comp_v2 | 联合 | 无 | 50% 批内样本 |
| joint_u_comp_source_control_v2 | 联合 | 保留 | 50% 批内样本 |

组成以每个样本的概率 0.5 提供，另 0.5 学习自由生成；不提供结构补全观测。四组使用现有默认 DescriptorTransformer/OrbitFlow、legacy prior、BF16 神经网络训练、FP32 几何、AdamW、学习率 3e-4、seed=42、验证间隔 5 epoch、最多 1000 epoch、早停耐心 50 epoch、梯度裁剪 10、坐标长度标度 3 Å。联合两组共用同一批大小和边预算。新旧模型直接比较还混入了组成学习等差异，**source_domain 的消融应比较新联合两组**。当前只有一个训练种子，不能据此声称统计稳健性。

从服务器项目根目录执行：

```bash
(
set -e
module load proxy2/proxy2
cd /fs6/home/zhongrsh/workspace/UnifiedGroupGen
git pull --ff-only origin main
python submit_files/hpc5090.py plan
python submit_files/hpc5090.py submit --name hpc5090_comp_v1
)
```

`plan` 不开始计算或提交；`submit` 一次提交 25 个作业：

1. 准备：核对本地公共预测器缓存、并行审计四组训练数据、核验旧 checkpoint 和可导入候选。
2. 校准：依赖准备，用真实 TRAIN 批次测试吞吐、最大边数批次和有限梯度；临时权重丢弃。
3. 四个训练：均依赖校准，互相独立。
4. 十五个旧模型生成与指标：均只依赖准备，可与训练排队竞争账号可用 GPU。
5. 四个新模型评估：各自只依赖对应训练。

作业时限依次为准备 4h、校准 2h、每个训练 72h、每个评估 24h；这是调度上限，不是耗时预测。依赖失败时后续作业取消，防止误用未完成的数据或模型。账号的并发 GPU 限额由调度器执行，不必手动等上一个模型结束再提交下一个。重复 `submit` 同一名称只补交尚未记录作业号的任务，不会自动重交已失败/取消的作业。

如果 GitHub 代理再次返回 503，可上传随本次发布生成的 `UnifiedGroupGen_update_hpc5090.bundle` 到服务器 workspace，先完成下面的离线更新，再执行 `plan` 和 `submit`。该增量包以服务器已验证的 `967276ef` 为基线：

```bash
(
set -e
cd /fs6/home/zhongrsh/workspace/UnifiedGroupGen
git bundle verify /fs6/home/zhongrsh/workspace/UnifiedGroupGen_update_hpc5090.bundle
git fetch /fs6/home/zhongrsh/workspace/UnifiedGroupGen_update_hpc5090.bundle refs/heads/main:refs/remotes/origin/main
git merge --ff-only origin/main
)
```

提交器用 `git archive` 固定本次提交的代码、配置和脚本；快照位于 `intermediate/hpc_runs/hpc5090_comp_v1/source/`。快照通过符号链接访问服务器的 `data/`、`outputs/`、`assets/`、`intermediate/`、`wandb/` 和 `.env`。后续代码更新可供下一个任务批次使用，不改变已排队任务的代码。共享的数据、环境和 `.env` 仍是服务器当前版本，运行期间保持不变。计算节点不调用 Git。

四个训练审计证书分别保存在本批次的 `audits/`；作业通过进程环境变量 `UGG_CAMPAIGN_NAME` 解析证书路径。新批次不会覆盖旧批次的证书或源码配置路径。

### 参数校准与预估

| 数据 | 待测 batch_size | 对应 edge_budget |
| --- | --- | --- |
| MP-20 / 联合 | 256 / 384 / 512 | 400000 / 500000 / 600000 |
| C2DB-51 | 768 / 1024 / 1536 | 200000 / 300000 / 400000 |

校准在 2 次热身后计时 4 个不同尺寸区间批次，并额外运行 2 个最大边数批次；保留 AdamW 状态，要求梯度有限且峰值保留显存不超过总显存的 85%，再从通过的候选中选择吞吐最高的一组。`edge_chunk_size=32768`，不启用梯度检查点。所有候选失败则阻止正式训练，不强行启动。校准不是完整 epoch 或泛化测试，后续仍需查看正式训练的显存和 epoch 时间。

旧日志里 MP-20 无性质模型约 8.73h/235 epoch，C2DB 约 0.78h/155 epoch，联合约 13.92h/225 epoch；这些是历史配置和硬件的观测。新 5090 训练先按 MP-20 **6–16h**、C2DB **1–4h**、每个联合 **10–24h** 规划，四组约 **27–68 GPU 小时**，不含排队、审计和评估。它们是粗预算，不是 5090 实测；校准结果写入 `profile.json`，正式 epoch 日志出来后再更新。较大批次也可能改变收敛 epoch 数。

评估保留原 `main15_local` 协议：64 步 Heun、FP32、seed=142；自由生成 256 个；CSP 256 个测试组成、每组成 20 候选；固定真实描述符每组成 1 候选；每个空间群/层群 5 个；联合数据平分组成/自由生成预算。条件旧模型再做 TRAIN 分位数性质目标自由生成，每目标 128 个，联合按域平分。CSP、固定真实描述符、自由生成、群约束结果分别报告。

神经积分批大小为 MP/联合 16、C2DB 32，component 检查为 32。每个评估作业同时最多运行 3 个单线程 CPU 指标进程；GPU 生成使用 3 个 CPU 线程，另留 CPU 给控制与加载。对于全部候选均已复用、无需新采样的旧模型，改为 6 个指标进程，账本导入使用 1 个线程。公共参考结构/性质缓存加跨进程文件锁，供同一批次的模型复用。CPU 指标较慢时 GPU 会等待或进入作业尾段；只有 GPU 分区权限时仍会产生这部分 GPU 占用。

旧诊断的生成约 0.5–1.7 秒/尝试，CFG=2 通常更慢。当前数据下原 15 模型协议计划 **107660 次尝试**；导入核验会写出 `legacy_budget.json` 中的精确 planned/retained/new/failed 数。可暂按剩余尝试数 × 0.5–1.7 秒估计**纯生成**，再加 CPU 指标和启动开销；packed 推理的 5090 加速比例尚未实测。新四模型共 **27178 次尝试**，对应纯生成粗预算约 4–13 GPU 小时。公共预测器参考缓存首次构建可能成为 CPU 瓶颈，因此不承诺这些时间覆盖完整指标。

### 历史结果、续跑与状态

旧 `outputs/<旧模型>/<原目录>/generate/diagnostic256_*` 和 `baseline256_*` 保留；本次生成/评估使用 `hpc5090_comp_v1_` 任务前缀。导入先核对 checkpoint 与已审计并随 Git 固定的 `configs/hpc5090_legacy_checkpoints.json` 哈希、账本索引、CIF 哈希、性质请求、case/candidate 槽位及请求群。失败候选同样导入，保留原始 seed 和来源，继续占用原分母。原目录缺少最初代码/选项身份收据这一限制会写入 import.json，不能将目录来源当作完整历史溯源。无法核验的来源记录原因后不导入，改在新目录按计划生成。

新原生结果保存内嵌 CIF、SHA256 和完整尝试账本。断点恢复只补缺失槽位，不将失败替换为成功，不修改旧文件。改模型、数据、指标实现或生成协议会拒绝继续写同一身份的结果。

```bash
cd /fs6/home/zhongrsh/workspace/UnifiedGroupGen
yhq
python submit_files/hpc5090.py status intermediate/hpc_runs/hpc5090_comp_v1/campaign.json
cat intermediate/hpc_runs/hpc5090_comp_v1/legacy_budget.json
cat intermediate/hpc_runs/hpc5090_comp_v1/profile.json
tail -n 60 intermediate/hpc_runs/hpc5090_comp_v1/logs/*.out
```

新模型位于 `outputs/<新模型>/hpc5090_comp_v1/model.best.pt`（最佳）和 `model.pt`（可恢复的原生 Lightning 最近状态），不会覆盖原 15 模型。每模型评估汇总位于 `intermediate/hpc_runs/hpc5090_comp_v1/evaluation/<模型>/summary.csv`；详细指标和样本账本在对应模型的 `evaluate/`、`generate/` 子目录。

如果作业失败，先查看 `logs/` 和 `state/`，确认原作业已经停止；可通过同一个冻结调度脚本重新提交对应阶段。训练自动从 `model.pt` 恢复，生成/指标核验已完成输出后跳过。依赖于失败任务的评估已经取消，需要另行重交；若修改了算法、输入或参数，则应以新代码、新批次名重新提交。

```bash
cd /fs6/home/zhongrsh/workspace/UnifiedGroupGen
ugg_run="$PWD/intermediate/hpc_runs/hpc5090_comp_v1"
UGG_GIT_COMMIT=$(python -c 'import json; print(json.load(open("intermediate/hpc_runs/hpc5090_comp_v1/campaign.json"))["revision"])') \
yhbatch --parsable --export=ALL --chdir="$ugg_run/source" --time=72:00:00 \
  --output="$ugg_run/logs/retry-train-mp20-%j.out" "$ugg_run/source/submit_files/hpc5090.sbatch" \
  --campaign "$ugg_run/campaign.json" --phase train --model mp20_u_comp_v2
```

当前接入的公共 MatGL 代理仅覆盖 formation_energy 和 band_gap；**energy_above_hull 不予评估**。C2DB 为探索性迁移，未证明预测准确度；公共预训练模型与测试集重叠也未审计。指标完成不代表结构已弛豫、热力学稳定或通过 DFT。

准备作业会先在 NaCl 体结构和 MoS2 层结构上实际运行两种公共预测器，任一前向失败都会阻止下游任务。本地 Windows / NumPy 1.26 探针已观察到 MatGL 0.9.2 构图中 `dtype=int` 与 Pymatgen 所需 int64 的位宽不匹配；没有修改本地环境或公共权重。Linux 环境的推理必须由服务器准备作业实测通过，不能用 GPU 算子检查或本地无预测器的回归测试替代。

仅在单独 Windows 测试进程中将该邻居输入转为 int64 后，岩盐 NaCl（SG225，8 原子）和 MoS2 的两种预测器均得到有限输出；这只是接口/数值检查，不是预测准确度或稳定性证据。

### CUDA 初始化失败后的恢复

若失败集中在某个节点，可临时排除该节点恢复本批次。以下 `node-5090-2` 是本次日志中四个 CUDA 初始化失败作业所在节点；这是作业级绕过措施，不修改节点或驱动配置。

```bash
cd /fs6/home/zhongrsh/workspace/UnifiedGroupGen
python submit_files/recover_hpc5090.py --campaign intermediate/hpc_runs/hpc5090_comp_v1/campaign.json --exclude node-5090-2
python submit_files/recover_hpc5090.py --campaign intermediate/hpc_runs/hpc5090_comp_v1/campaign.json --exclude node-5090-2 --submit
yhq
```

第一条预览恢复计划；第二条实际提交。入口核对本人队列，跳过仍在队列中的作业和成功/运行阶段，只重交已有 `failed` 标记的阶段。重交训练的后续评估会依赖新训练作业号。作业继续使用原 `campaign.json` 的源码快照、参数和输出目录，不重新训练已完成的 C2DB，也不修改运行中的 MP-20/联合训练。

本人仍在等待的本批次作业通过 `yhcontrol` 或 `scontrol` 增加节点排除，保留已有排除列表；不会改变运行中的作业或其他用户的节点设置。如果存在待更新的等待作业，但两个控制命令均不可用，脚本在提交前停止。原作业号备份到 `submitted.before_recovery.json`，新旧作业号对应关系保存到 `recovery.json`，`submitted.json` 更新为最新作业号，因此原 `status` 命令仍可使用。同一恢复记录中的作业不会因重复执行而再次提交；重试再次失败时先检查新日志。

### 指定组成设计

`design_composition.sbatch` 接受精确原子数 JSON、结构类型和独立输出目录；不提供性质条件，不固定真实群/描述符。优先使用去来源标签的三个新模型；source control 因需要来源上下文，不适用于这个简化模板。只有训练完成后才能提交。以下 Si2O4 是调用示例，不是已执行的实验：

```bash
cd /fs6/home/zhongrsh/workspace/UnifiedGroupGen
ugg_run="$PWD/intermediate/hpc_runs/hpc5090_comp_v1"
UGG_GIT_COMMIT=$(python -c 'import json; print(json.load(open("intermediate/hpc_runs/hpc5090_comp_v1/campaign.json"))["revision"])') \
yhbatch --parsable --export=ALL --chdir="$ugg_run/source" \
  --output="$ugg_run/logs/design-Si2O4-%j.out" "$ugg_run/source/submit_files/design_composition.sbatch" \
  "$PWD/outputs/joint_u_comp_v2/hpc5090_comp_v1/model.best.pt" '{"Si":2,"O":4}' space \
  "$PWD/outputs/joint_u_comp_v2/hpc5090_comp_v1/generate/design_Si2O4_seed142"
```

模板生成 256 次尝试、batch=16、64 步 Heun、FP32、guidance=0；不自动启动弛豫或物性计算。精确原子数与约化化学式不同：`{"Si":2,"O":4}` 必须是 6 原子；经验式模式另通过现有 `generation.formula` 和 `formula_units` 使用。元素必须在训练词表内、所选群轨道必须允许该组成，不保证每次采样都成功。
