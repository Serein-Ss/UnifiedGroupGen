# UnifiedGroupGen

基于群—轨道参数化和性质条件流匹配的材料结构生成研究实现。

**当前状态：可运行的研究代码，包括数据转换、离散描述生成、连续训练与采样、约束验证。尚未完成正式基准训练、性质命中率评测或稳定性验证。**

核心论点：把固定离散描述下的连续结构限制在群约束的合法参数域，将空间群、层群和平面群编译成相同的结构展开、随机先验、合法训练路径和切向更新接口。性质条件同时进入描述 Transformer 与几何流模型。

## 文档

- [完整研究方案](docs/RESEARCH_PLAN.md)：Introduction、Related Work、Preliminaries、Method、Experiments 和研究里程碑。
- [核心理论与证明](docs/THEORY.md)：群作用、零空间、晶格、条件纤维、流匹配、磁性扩展及成立边界。
- [实现对应关系](docs/IMPLEMENTATION.md)：与 CGDiT/newton、CrystalFlow 的关系、接口、数据协议和已实现范围。
- [验证记录](docs/VALIDATION.md)：本次实际执行结果与未完成项目。
- [正式训练前准备](docs/PRETRAINING_READINESS.md)：全数据状态、训练入口检查、运行命令与未满足条件。
- [参数量与容量对照](docs/MODEL_SCALE.md)：模型容量、参考工作计数边界与 GPU 实测。
- [文献与来源](docs/REFERENCES.md)。
- [Linux / RTX4090下载与完整训练](docs/REMOTE_TRAINING.md)：仓库内完整数据包、路径迁移、验证、启动、恢复、epoch与耗时。

## 服务器快速开始

仓库内已包含完整训练数据包（102,610条来源记录及联合划分），不依赖外部CGDiT工作目录。

```bash
git clone https://github.com/Serein-Ss/UnifiedGroupGen.git
cd UnifiedGroupGen
conda create -n unifiedgroupgen python=3.10 -y
conda activate unifiedgroupgen
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-remote.txt
python -m pip install -e ".[test]"
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python scripts/unpack_training_data.py
python scripts/verify_training_data.py
python -m pytest -q
mkdir -p artifacts/runs
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_remote.py --stage all > artifacts/runs/remote_driver.log 2>&1 &
```

优先训练二维/三维共享主模型可改为`--stage main`；主模型不依赖两个参考模型的权重。

## 目录

```text
UnifiedGroupGen/
  configs/                  MP20、MPTS52、C2DB、联合空间群/层群配置
  docs/                     研究方案、理论、实现说明、验证、文献
  examples/                 固定描述与固定晶格示例
  src/unifiedgroupgen/
    symmetry.py             群表、真实轨道自由度、相容 SPD 晶格、Φ_a
    constraints.py          共同约束零空间投影、仿射观测纤维
    conditioning.py         性质编码、缺失值、训练集缩放、CFG 空条件
    proposal.py             性质条件群/轨道/元素 Transformer、可完成前缀掩码
    flow.py                 CSPNet 风格消息网络、合法路径、FM、Heun 采样
    data.py                 原 CSV 转换、层群匹配、失败与坐标来源记录
    evaluation.py           群闭合、晶格残差、重数、距离、CIF 导出
    native_identifier.py    子进程隔离 SG/LG 识别，崩溃/超时记为未知
    spin.py                 共线单位磁矩的表示约束扩展
    training.py             图批处理、分项损失、AMP、last/best 与完整续训
    preflight.py            数据划分、有限性、来源、覆盖率与训练门槛
    joint.py                跨库材料分组划分、全部记录守恒与来源追溯
    cli.py                  prepare / preflight / train / sample / audit / parameters / summarize
  scripts/                  全量准备、容量比较与训练步成本测量
  tests/                    数学、模型和真实数据接口测试
  artifacts/                本地缓存、检查点和样本，默认不提交 Git
```

## 使用

在兼容的 Python/PyTorch 环境中，从本目录执行：

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/prepare_training.py --config configs/mp20_large.yaml --workers 4 --resume
python -m unifiedgroupgen.cli train --config configs/mp20_large.yaml --device cuda --output artifacts/runs/mp20_large_seed42/model.pt
python -m unifiedgroupgen.cli sample --checkpoint artifacts/runs/mp20_large_seed42/model.best.pt --kind space --count 100
```

性质条件逆向设计：

```powershell
python -m unifiedgroupgen.cli sample --checkpoint artifacts/runs/mp20_large_seed42/model.best.pt --properties '{"formation_energy": -1.0, "band_gap": 1.5}' --guidance 2 --count 100
```

固定组成：增加 `--composition Si2O4`。计数按模型采用的**常规晶胞**解释；不能直接把化学简式当成任意空间群下的常规晶胞组成。

固定描述及晶格补全：

```powershell
python -m unifiedgroupgen.cli sample --checkpoint artifacts/runs/mp20_large_seed42/model.best.pt --descriptor examples/descriptor.json --observations examples/observations.json --count 10
```

示例中的六个零值是相对于参考不变度量的六个对数晶格参数，生成三轴长度 3 Å 的 P-1 晶胞；不是把物理晶格长度设为零。

C2DB 的正式来源恢复配置为 `configs/c2db_official_local.yaml`，需要全部原始记录恢复成功后才能准备和训练。旧 `c2db_large.yaml` 保留用于未确认来源的诊断，会阻断正式训练。MPTS52 使用 `configs/mpts52_large.yaml`，正式识别容差为 0.01 Å，缓存独立保存。原 `mp20.yaml/mpts52.yaml/c2db.yaml` 为小模型诊断配置。所有缓存输出位于本目录，原数据不改写。

正式共享模型：先通过三个完整来源审计，再按材料分组重划分，最后从头训练。原 MP20/MPTS52 划分存在交叉重合，不能直接拼接。

```powershell
python scripts/prepare_joint.py --config configs/joint_space_layer_large.yaml
python -m unifiedgroupgen.cli train --config configs/joint_space_layer_large.yaml --device cuda --output artifacts/runs/joint_space_layer_large_seed42/model.pt
```

断点续训使用 `train ... --resume artifacts/runs/mp20_large_seed42/model.pt`，数据哈希、配置和缩放参数必须一致。`--epochs` 是总轮数。完整流程和状态见 [正式训练前准备](docs/PRETRAINING_READINESS.md)。

本机正式训练已使用 large + batch64 的 `mp20_local.yaml` / `mpts52_local.yaml` 顺序运行，入口是 `scripts/train_local.py`；实时状态在 `artifacts/runs/local_large_seed42/runner.json`。现有 cgdit 环境的 OpenMP 修复由 runner 在 Python 导入前设置。最新独立目录验证 **88 项测试通过**（含4项迁移测试）。共享主模型约 **18,030,806** 参数，保留计算来源条件并让 CFG 只移除性质条件。

已补充PyTorch 2.2旧AMP接口兼容：当前CUDA环境91项测试通过，实际PyTorch 2.2.1 CPU环境80项通过、11项CUDA测试跳过。服务器现有cgdit环境的安装、校验与完整训练命令见[复用cgdit说明](docs/REMOTE_TRAINING.md#复用已部署的cgdit环境)。

`scripts/complete_pipeline.py` 编排完整来源恢复、二维准备、共享划分、等待已有三维作业、共享训练及测试/生成评价；只在数据审计通过后启动共享 GPU 训练。C2DB 全部16905条来源恢复、v7全量转换与审计已通过，零拒绝；官网缺失的 `4Nb2S3-2` 已从公开原始ASE归档精确恢复。联合划分保留全部102610条输入，修复后的完整联合审计及实际训练入口核查均已通过。`recover_c2db.py --ase-db <原数据库路径>` 按确切 UID/ASE unique_id 恢复并验证来源，不删除缺失记录。

## 已实现与边界

| 项目 | 状态 |
|---|---|
| 230 SG / 80 LG / 17 PG 的共同编译接口 | 已实现，群表来源 PyXtal |
| 真实 0/1/2/3 维轨道参数与相容正定晶格 | 已实现 |
| 学习群、轨道实例、元素，不从经验群分布采样 | 已实现 |
| 组成、原子数上限、零维位置不可重复、可完成前缀 | 已实现；保证符号合法性 |
| 多性质编码、缺失标记、训练集统计、两阶段 CFG | 已实现 |
| 合法先验、周期自由坐标路径、非周期层高度、流匹配 | 已实现 |
| 固定晶格 / 部分自由坐标 / 一般仿射观测 | 已实现于固定描述和固定提升坐标图 |
| 共线单位磁矩 | 已实现约束构造与采样工具；未训练磁性生成模型 |
| 任意非线性性质的严格等值面生成 | 理论接口；未实现曲面训练/积分算法 |
| 真实结构的层群重新识别 | 已接入 spglib.get_layergroup；与指定操作闭合分别报告；80 个双轨道合法测试均精确回收目标 LG |
| 化学合理、低能量、凸包稳定、动力学稳定、性质命中 | 需要正式训练及外部计算验证 |

生成器保证包含指定群操作；结构可能具有更高对称性。轨道退化与原子碰撞单独报告。合法性掩码没有价态判定器，不能替代 CrysVCD 式化学验证。

## 数据注意事项

MP20/MPTS52 使用 PyXtal 的常规晶胞表示，因此展开后原子数可能超过原数据的 20/52。配置上限对应展开计数；评价时必须转回同一计数约定，不能直接与原始胞的统计混比。

本地 `c2db_51` 的 CIF 把笛卡尔坐标写入分数坐标字段，同时丢失完整晶胞朝向。正式流程恢复原始 cell/positions 并与 CSV 核对，再由 spglib 识别/标准化层群；缺失 lgnum 不会被直接设为 LG1。全部16905条已转换并审计通过。统一0.1 Å识别容差下，仅1条非理想standard positions需要额外轨道拟合，其编译结果通过严格闭合检查。25条已知层群标签发生重识别变化，94条缺失标签成功识别；原始标签、识别标签、坐标理想化和度量变化均记录。识别容差不是最终坐标位移或晶格应变的严格上界，各数据库能量定义以 source_domain 区分。

## 基础来源

本实现参考 [CGDiT/newton](https://github.com/Serein-Ss/test_CGDiT/tree/newton) 的性质编码、CFG、周期消息传递和轨道处理，以及 [CrystalFlow](https://github.com/ixsluo/CrystalFlow) 的晶格/坐标联合流匹配。代码采用独立小模块实现，无需导入父目录 CGDiT，也不依赖 torch-scatter 或 PyG。原检查点不能直接载入本模型；这是结构约束和参数维度改变后的新模型。
