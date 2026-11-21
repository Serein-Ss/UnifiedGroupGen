# 12个独立模型配置与5090 FP32正式训练基准

本批次准备12个joint_flow_v2模型，沿用已经修复的六随机流辅助路径、群操作硬约束和联合损失。不修改生成模型，不合并三种任务训练。默认提交仅包含MP-20＋EquiformerV2的DNG、CSP两个模型；其他10组只准备配置，需显式选择。

## 配置矩阵

| 数据集 | 主干 | DNG | 精确组分CSP | 带隙性质生成 |
|---|---|---|---|---|
| MP-20 | EquiformerV2 | [full_mp20_eqv2_dng](../configs/experiment/full_mp20_eqv2_dng.yaml) | [full_mp20_eqv2_csp](../configs/experiment/full_mp20_eqv2_csp.yaml) | [full_mp20_eqv2_property](../configs/experiment/full_mp20_eqv2_property.yaml) |
| C2DB-51 | EquiformerV2 | [full_c2db51_eqv2_dng](../configs/experiment/full_c2db51_eqv2_dng.yaml) | [full_c2db51_eqv2_csp](../configs/experiment/full_c2db51_eqv2_csp.yaml) | [full_c2db51_eqv2_property](../configs/experiment/full_c2db51_eqv2_property.yaml) |
| 两库混合 | EquiformerV2 | [full_joint_eqv2_dng](../configs/experiment/full_joint_eqv2_dng.yaml) | [full_joint_eqv2_csp](../configs/experiment/full_joint_eqv2_csp.yaml) | [full_joint_eqv2_property](../configs/experiment/full_joint_eqv2_property.yaml) |
| MP-20 | JointAtomGraph | [full_mp20_atomgraph_dng](../configs/experiment/full_mp20_atomgraph_dng.yaml) | [full_mp20_atomgraph_csp](../configs/experiment/full_mp20_atomgraph_csp.yaml) | [full_mp20_atomgraph_property](../configs/experiment/full_mp20_atomgraph_property.yaml) |

总表：[experiments_joint_flow_full.yaml](../configs/experiments_joint_flow_full.yaml)。各选择器直接组合来源、模型、训练和运行配置，历史任务保存在原冻结批次。

DNG默认不提供性质或组分。CSP训练时可见准确的每胞元素计数，以库存受限的整轨道动作构造路径。表中的带隙实验为 DNG 加目标性质 CFG，观测标签参与拟合，训练条件丢弃率为 0.1。也可在 CSP 中显式选择 `conditioning=band_gap`。全表的 `chemical_set_probability` 为 0。

混合数据按相同划分自然拼接，不作1:1重采样，不加入source_domain/dataset条件。记录中的dataset仅用于标识、缓存键和审计，不作为神经条件。

无群输入沿用TRAIN频率初始化，CSP沿用库存可行性筛选；本批次不新增性质/组分条件下的群混合校准。性质模型所用的受控混合不自动等于完整数据条件分布，仍按既有数学说明解释。

| 数据 | TRAIN | VAL | TEST | 性质任务可用TRAIN带隙标签 |
|---|---:|---:|---:|---:|
| MP-20 | 27136 | 9047 | 9046 | 27136 |
| C2DB-51 | 10143 | 3381 | 3381 | 5206 |
| 混合 | 37279 | 12428 | 12427 | 32342 |

本地只读清点确认两库各自的(dataset,id)在划分间不重叠；这不等于新增的物理结构去重证明。缺失标签不填零，性质训练/验证只保留观测标签；DNG/CSP使用完整TRAIN/VAL。TEST不参与优化或最佳检查点选择，完整划分在服务器审计中仍被核对。

## 完整训练设置

- 全量可用TRAIN与VAL，最多1000 epoch，无短测记录截断，joint_max_steps=-1。1000是上限，早停可提前结束，不保证全局收敛或材料生成质量。
- AdamW，初始学习率3e-4，梯度裁剪10。每epoch计算完整VAL的联合损失val/loss及连续几何、元素、跳转等分项，以val/loss选择最佳检查点。
- ReduceLROnPlateau按验证损失衰减：factor=0.5、patience=10、绝对改进阈值1e-4、min_lr=1e-6。按PyTorch定义，超过10次无显著改善的验证后减半；批大小变化不自动线性放大学习率。
- 早停同样监控val/loss，连续60次验证无至少1e-4的改善后结束；非有限验证值触发停止检查。验证每epoch一次，因此耐心以epoch计。验证教师路径固定，验证阶段原生网络随机数固定为918并在结束后恢复训练随机状态，减少监控噪声；这不是确定性GPU计算的保证。
- 主干宽度128、4层；EquiformerV2为128球通道、lmax=4、mmax=2、原生8头，配置heads已标为实际8；JointAtomGraph沿用4头。原生Equiformer核心及其等变表示未修改。
- 事件预算96，最多192原子、52轨道；轨道预算回退使用修复后的独立时钟。
- 正式训练使用FP32（Lightning 32-true），关闭autocast与TF32；融合AdamW仍保留，参数、梯度和优化器状态使用FP32。日志步频25；激活重计算默认关闭。可选joint_flow_fast配置保留其原有BF16行为，不作为本轮基准。
- 每个5090作业先自动校准 `batch_size`，再按选中的固定结构数分批。直接启动训练入口时的默认批大小是 8。默认服务器提交路径必须有通过校验的校准记录才可启动训练。

共同配置为[完整训练](../configs/train/joint_flow_full.yaml)和[5090执行配置](../configs/runtime/joint_flow_5090.yaml)。两主干的表示和参数量不同，本组对照用于检验同一方法在替换主干后是否适用，不宣称参数量相等。

## 服务器自动校准

[校准程序](../submit_files/joint_flow5090_calibrate.py)仅允许在已分配的服务器作业内运行。DNG、CSP 分别校准，候选批大小为 1、2、4、8、16、32、64、128。训练使用 PyTorch 随机采样和固定大小组批，验证保持顺序。压力检查集中测试原子数较大的结构。

每个候选使用独立CUDA子进程与新初始化模型，执行2步预热、5步吞吐计时以及4步大结构/时间压力检查（t=0.05、0.75、0.95）。每候选11步、每模型最多88步临时前向/反向/优化器探测，**仅在用户启动服务器作业后执行**。从TRAIN中按实际批采样器选择代表性批次，包含目标尺寸较大的结构；不使用VAL/TEST选择批大小。计时包括CPU路径构造、传输、FP32前后向、裁剪和融合AdamW，预热和压力检查不计入吞吐。临时权重丢弃，正式训练在独立进程以seed=42重新初始化。

筛选要求梯度/权重有限，且峰值CUDA reserved显存不超过总显存85%和启动空闲显存减2 GiB两者的较小值。OOM或显存余量不足的候选淘汰，其他错误阻止正式训练；通过者按实测记录/秒选择，吞吐相同优先较小批次。所有候选失败也会阻止训练。有限探测只覆盖部分中间状态，正式训练仍需观察实际峰值显存。

校准结果保存到intermediate/hpc_runs/<批次名>/profiles/<模型名>.json，包含GPU、显存阈值、峰值、吞吐、实际批大小和中间原子数。对应profiles/<模型名>/attempt_*/保存输入、每候选JSON和日志。完整结果绑定冻结提交、配置哈希、TRAIN输入哈希和校准策略，重试直接复用。已有last.ckpt时若原校准记录缺失或不一致则拒绝重新校准，避免断点恢复时改变训练批次轨迹。

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
python submit_files/joint_flow5090.py submit --name jointflow_mp20_fp32_v2
yhq
```

提交器为每个模型申请5090分区1 GPU、8 CPU、64 GiB内存、72h。72h是单次调度上限，不是1000 epoch总耗时预测。两个作业独立，账号并发限制由调度器执行；校准在各自同一个训练作业内完成，没有额外校准作业，也不提交旧15模型评估、其他10个模型、生成或DFT任务。

每个计算作业依次执行完整数据审计、自动校准、正常Hydra/Lightning训练入口。审计、校准与训练使用同一个批次私有data_audit覆盖；审计失败阻止后续步骤。运行日志记录GPU、冻结提交和正式训练的batch/edge覆盖参数，脚本不安装或升级环境包。

`submit`通过git archive冻结代码、配置和提交脚本。冻结目录链接服务器自己的data、outputs、intermediate、assets、wandb和.env；不复制或改写已有权重、数据、环境。运行期间保持共享数据和环境不变，并保留冻结目录供恢复使用。新的代码提交可用于新批次，不替换已有批次的源码。

源文件必须已提交；未提交的运行代码会被拒绝，以免实际训练的版本和用户看到的工作树不同。提交器不自动reset、stash或清理任何文件。直接运行sbatch文件不替代提交器提供的冻结目录和manifest参数。

固定组批使用 `ugg_joint_flow_full_5090_v3`。新实验使用新的批次名，旧批次的精确续训使用各自的冻结源码和原校准记录。旧权重可用于生成评估，或通过 `initialize_from` 初始化一个新训练。

其他模型只在明确选择时提交，例如：

```bash
python submit_files/joint_flow5090.py plan --models full_joint_eqv2_property
python submit_files/joint_flow5090.py submit --name jointflow_joint_property_fp32_v2 --models full_joint_eqv2_property
```

## 输出与续训

首次默认批次的输出分别位于：

```text
outputs/full_mp20_eqv2_dng/jointflow_mp20_fp32_v2/
outputs/full_mp20_eqv2_csp/jointflow_mp20_fp32_v2/
```

每组包含model.pt（最终权重）、best.ckpt（按VAL损失选择的原生最佳权重）、last.ckpt（续训检查点）、model.history.jsonl及model.status.json。history记录联合/分项训练验证损失、当轮学习率和早停状态；最终status记录实际完成epoch、1000上限、停止原因及最佳验证损失。既有joint sample入口可直接读取best.ckpt；后续DNG/CSP评估应使用独立GPU作业和既定完整尝试分母，不将验证损失或示例结构当成材料性能结论。

新版见[统一输出规范](OUTPUT_LAYOUT.md)：v2状态增加模型构建、目标缓存准备、等待首批、训练、验证及保存阶段，生成补齐摘要/状态文件。GitHub代码更新不会改变已按a5079802冻结的任务。初始化阶段不是定时心跳，不能只凭文件更新时间判断卡住；已有服务器状态仍按冻结版本解释。

批次目录intermediate/hpc_runs/jointflow_mp20_fp32_v2/中保存campaign.json、submitted.json、submission_history.jsonl、source/、audits/、profiles/及logs/。日志名为ugg_模型名-作业号.out，可用tail查看。

重复普通 submit 不会重复提交已有作业号。`--retry` 仅用于当前批次、原冻结源码和原运行配置。先用 yhq 确认原作业已经结束，再用原批次名执行：

```bash
python submit_files/joint_flow5090.py submit --name jointflow_mp20_fp32_v2 --retry
```

它保留原冻结源码和选择，跳过已完成模型，重新审计未完成模型、复用原校准配置并从last.ckpt恢复；worker文件锁阻止同一模型并发训练。原生检查点保留优化器、学习率调度器、早停计数、循环和RNG状态；针对Lightning 2.5在epoch回调保存后才更新plateau调度器的顺序，记录尚未执行的验证指标，并在恢复时补执行一次，最终保存后不重复执行。没有last.ckpt但已有训练产物时拒绝覆盖，应检查原因并使用新批次名。更改模型集合、训练超参数或数据需创建新批次并重新审计；不能把旧批次恢复成另一个实验。

## 本地验证范围

本次本地未构造真实训练模型、未执行真实Trainer.fit、反向传播、optimizer.step、CUDA校准或调度提交。仅进行12配置解析、只读Hydra plan、Bash -n检查，以及模拟yhbatch/训练子进程的配置与控制流测试。调度器状态测试只为标量参数实例化优化器并更新学习率计数，不更新参数。验证记录见[FP32准备状态](validation/full_flow_fp32_5090_20261113.json)；[旧准备记录](validation/full_flow_5090_20261113.json)保留为200 epoch/BF16版本的历史快照。

本轮47项配置、校准控制流与状态测试通过，覆盖plateau衰减/恢复、早停计数恢复、FP32设置、验证RNG隔离、校准候选选择/输入映射、原校准复用及默认两作业门禁。默认两个模型的真实Hydra plan和Git Bash语法检查另行执行。未运行会启动短训的旧测试集；真实Lightning端到端续训、5090校准、吞吐、显存、实际收敛和材料性能需服务器启动后实测。

## MP-20 双主干对照

使用以下命令查看四个独立配置及批大小预检计划。

~~~bash
python -B submit_files/joint_flow5090.py plan --backbones
~~~

四组均采用随机初始化、MP-20 全量数据、seed 42、严格 FP32、batch size 4 和 AdamW。上限为 1000 轮，验证损失连续 30 轮无改善时早停。四组分别在 5090 上对 batch size 4 做显存和有限梯度检查。CSPNet 沿用速度测试中的 288 通道配置，EquiformerV2 沿用 128 通道和 8 个注意力头的全量配置。两种主干的参数量不同。

四组共用晶格形状、各向同性体积和轨道位点分项损失，以及显式群约束和整轨道规则。CSP 的组分排序使用训练集拟合的频次，也可通过 --csp-group-policy prior 运行原先的抽群方式。显式输入群编号或操作集时，直接使用指定条件。DNG 继续使用训练集群先验。

新作业从 Git 冻结源码，使用 UGG_PROJECT_ROOT 定位工作区及其中的 .env。MP-20 原始输入从 data/src/mp_20 读取，可按该路径复用已有原始 CSV。当前流程根据源数据和协议指纹选择 Tensor 缓存。旧版仓库中的原始 CSV 位于 data/scr/mp_20。

不带参数时仍显示两份原 EquiformerV2 配置。双主干计划需要显式选择 --backbones。计划命令仅输出配置和资源，校准与训练由明确提交的节点作业执行。
