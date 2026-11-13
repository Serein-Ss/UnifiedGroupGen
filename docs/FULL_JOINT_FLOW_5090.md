# 12个独立模型配置与5090首次完整训练

本批次准备12个joint_flow_v2模型，沿用已经修复的六随机流辅助路径、群操作硬约束和联合损失。不修改生成模型，不合并三种任务训练。默认提交仅包含MP-20＋EquiformerV2的DNG、CSP两个模型；其他10组只准备配置，需显式选择。

## 配置矩阵

| 数据集 | 主干 | DNG | 精确组分CSP | 带隙性质生成 |
|---|---|---|---|---|
| MP-20 | EquiformerV2 | [full_mp20_eqv2_dng](../configs/experiment/full_mp20_eqv2_dng.yaml) | [full_mp20_eqv2_csp](../configs/experiment/full_mp20_eqv2_csp.yaml) | [full_mp20_eqv2_property](../configs/experiment/full_mp20_eqv2_property.yaml) |
| C2DB-51 | EquiformerV2 | [full_c2db51_eqv2_dng](../configs/experiment/full_c2db51_eqv2_dng.yaml) | [full_c2db51_eqv2_csp](../configs/experiment/full_c2db51_eqv2_csp.yaml) | [full_c2db51_eqv2_property](../configs/experiment/full_c2db51_eqv2_property.yaml) |
| 两库混合 | EquiformerV2 | [full_joint_eqv2_dng](../configs/experiment/full_joint_eqv2_dng.yaml) | [full_joint_eqv2_csp](../configs/experiment/full_joint_eqv2_csp.yaml) | [full_joint_eqv2_property](../configs/experiment/full_joint_eqv2_property.yaml) |
| MP-20 | JointAtomGraph | [full_mp20_atomgraph_dng](../configs/experiment/full_mp20_atomgraph_dng.yaml) | [full_mp20_atomgraph_csp](../configs/experiment/full_mp20_atomgraph_csp.yaml) | [full_mp20_atomgraph_property](../configs/experiment/full_mp20_atomgraph_property.yaml) |

总表：[experiments_joint_flow_full.yaml](../configs/experiments_joint_flow_full.yaml)。新选择器不覆盖原来的flow_*、task_*、旧两阶段配置或历史输出。

DNG不提供性质或组分；CSP训练时可见准确的每胞元素计数，以库存受限的整轨道动作构造路径；property只使用实际存在的band_gap目标标签。所有配置chemical_set_probability=0，不混入限定元素集合任务。当前性质模块和cond_dropout=0沿用既有独立任务设置，不宣称新增CFG。任务类型、周期维度及群规范保持现有硬约束定义。

混合数据按相同划分自然拼接，不作1:1重采样，不加入source_domain/dataset条件。记录中的dataset仅用于标识、缓存键和审计，不作为神经条件。

无群输入沿用TRAIN频率初始化，CSP沿用库存可行性筛选；本批次不新增性质/组分条件下的群混合校准。性质模型所用的受控混合不自动等于完整数据条件分布，仍按既有数学说明解释。

| 数据 | TRAIN | VAL | TEST | 性质任务可用TRAIN带隙标签 |
|---|---:|---:|---:|---:|
| MP-20 | 27136 | 9047 | 9046 | 27136 |
| C2DB-51 | 10143 | 3381 | 3381 | 5206 |
| 混合 | 37279 | 12428 | 12427 | 32342 |

本地只读清点确认两库各自的(dataset,id)在划分间不重叠；这不等于新增的物理结构去重证明。缺失标签不填零，性质训练/验证只保留观测标签；DNG/CSP使用完整TRAIN/VAL。TEST不参与优化或最佳检查点选择，完整划分在服务器审计中仍被核对。

## 完整训练设置

- 全量可用TRAIN与VAL，200 epoch，无短测记录截断，joint_max_steps=-1。
- AdamW，学习率3e-4保持固定，梯度裁剪10，每epoch验证并按val/loss保存最佳检查点。当前联合训练不接入旧两阶段的学习率调度和早停，200 epoch是首轮完整基线预算，不保证已经收敛。
- 主干宽度128、4层；EquiformerV2为128球通道、lmax=4、mmax=2、原生8头，配置heads已标为实际8；JointAtomGraph沿用4头。原生Equiformer核心及其等变表示未修改。
- 事件预算96，最多192原子、52轨道；轨道预算回退使用修复后的独立时钟。
- BF16神经计算、FP32关键几何、融合AdamW；日志步频25；激活重计算默认关闭。
- 5090起始batch_size=8、edge_budget=40000。该预算是原子数平方的批量代理，不是实际周期边数或显存的严格上界；这是未做5090实测校准的起点，不是最优吞吐或不OOM的保证。

共同配置为[完整训练](../configs/train/joint_flow_full.yaml)和[5090执行配置](../configs/runtime/joint_flow_5090.yaml)。两主干的表示和参数量不同，本组对照用于检验同一方法在替换主干后是否适用，不宣称参数量相等。

## 服务器更新与默认提交

沿用仓库记录的TH-HPC6环境：个人unifiedgroupgen Conda、GCC/12.2.0、CUDA/12.9、PyTorch cu128。不要用本机cu118环境文件覆盖5090环境。模块与路径以服务器实际安装为准，本次没有连接服务器或启动实机作业。

在服务器登录节点执行，先确认没有需保留的受跟踪代码改动，再快进更新：

```bash
source /fs6/software/modules/bashrc
cd /fs6/home/zhongrsh/workspace/UnifiedGroupGen
git status --short --branch
(module load proxy2/proxy2 && git pull --ff-only origin main)
source /fs6/software/python/3.13_miniforge3-26.1.1/etc/profile.d/conda.sh
conda activate unifiedgroupgen

# 只打印计划，不构造或训练模型，不提交作业。
python submit_files/joint_flow5090.py plan

# 用户确认准备启动时才执行：恰好两个训练作业。
python submit_files/joint_flow5090.py submit --name jointflow_mp20_first_v1
yhq
```

提交器为每个模型申请5090分区1 GPU、8 CPU、64 GiB内存、72h。72h是单次调度上限，不是训练耗时预测。两个作业独立，账号并发限制由调度器执行；没有附加旧15模型评估、校准训练、其他10个模型、生成或DFT任务。

每个计算作业先运行对应配置的完整数据审计，证书通过后才调用正常Hydra/Lightning训练入口。审计与训练使用同一个批次私有data_audit覆盖；审计失败阻止训练，不跳过门禁。运行日志记录GPU和提交版本，脚本不安装或升级环境包。

`submit`通过git archive冻结代码、配置和提交脚本。冻结目录链接服务器自己的data、outputs、intermediate、assets、wandb和.env；不复制或改写已有权重、数据、环境。运行期间保持共享数据和环境不变，并保留冻结目录供恢复使用。新的代码提交可用于新批次，不替换已有批次的源码。

源文件必须已提交；未提交的运行代码会被拒绝，以免实际训练的版本和用户看到的工作树不同。提交器不自动reset、stash或清理任何文件。直接运行sbatch文件不替代提交器提供的冻结目录和manifest参数。

其他模型只在明确选择时提交，例如：

```bash
python submit_files/joint_flow5090.py plan --models full_joint_eqv2_property
python submit_files/joint_flow5090.py submit --name jointflow_joint_property_v1 --models full_joint_eqv2_property
```

## 输出与续训

首次默认批次的输出分别位于：

```text
outputs/full_mp20_eqv2_dng/jointflow_mp20_first_v1/
outputs/full_mp20_eqv2_csp/jointflow_mp20_first_v1/
```

每组包含model.pt（最终权重）、best.ckpt（按VAL损失选择的原生最佳权重）、last.ckpt（含优化器与循环状态的续训检查点）、model.history.jsonl及model.status.json。既有joint sample入口可直接读取best.ckpt；后续DNG/CSP评估应使用独立GPU作业和既定完整尝试分母，不将验证损失或示例结构当成材料性能结论。

批次目录intermediate/hpc_runs/jointflow_mp20_first_v1/中保存campaign.json、submitted.json、submission_history.jsonl、source/、audits/及logs/。日志名为ugg_模型名-作业号.out，可用tail查看；训练指标写在对应模型的history文件中。

重复普通submit不会重复提交已有作业号。若超时或失败，先用yhq确认原作业已经结束，再执行：

```bash
python submit_files/joint_flow5090.py submit --name jointflow_mp20_first_v1 --retry
```

它保留原冻结源码和选择，跳过已完成模型，重新审计未完成模型并自动从其last.ckpt恢复；worker文件锁阻止同一模型并发训练。没有last.ckpt但已有训练产物时拒绝覆盖，应检查原因并使用新批次名。更改模型集合、训练超参数或数据需创建新批次并重新审计；不能把旧批次恢复成另一个实验。

## 本地验证范围

本次本地没有构造训练模型、运行优化器或提交调度任务。仅进行12配置解析、数据清点、只读Hydra plan、Bash -n语法检查，以及模拟yhbatch/训练子进程的配置与控制流测试。验证记录见[准备状态](validation/full_flow_5090_20261113.json)。服务器软件、显存、完整训练耗时、收敛与材料性能仍需用户启动后实测。

本轮配置与控制流测试31项通过；默认两个模型的真实Hydra plan入口通过，提交脚本通过Git Bash的-n解析。测试中的yhbatch与训练子进程均为mock，未重复运行会启动短训的旧测试集。
