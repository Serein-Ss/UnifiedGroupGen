> 2026-10-30后续职责整理见[CODE_ORGANIZATION.md](CODE_ORGANIZATION.md)，下文保留前一轮分层设计依据。

> 当前环境与命令以[ENVIRONMENT.md](ENVIRONMENT.md)、[核心库README](../unifiedgroupgen/README.md)为准；旧命令归档在tests/legacy。

# 核心包分层与统一训练入口

本次调整改变代码组织、配置组合和入口；默认网络参数名称、群约束数学、性质编码和损失定义保持原有行为。数据来源上下文的移除不属于本次重构，既有配置仍保留它。

## 目录职责

```text
unifiedgroupgen/
  run.py                     Hydra训练、断点恢复、预训练初始化入口
  common/                    路径、运行设置、检查点、随机状态、结构绘图
  group_constraints/         群—轨道编译器、合法空间、投影、自旋表示扩展
  conditional/               性质编码、化学硬约束、部分结构观测
  pl_data/                   结构编码、数据划分、边预算分批、Lightning DataModule
  pl_modules/
    backbones/               Transformer和图消息传递骨架
    descriptor.py            描述模型的可训练参数和前向传播
    orbitflow.py             几何特征、候选速度头、合法速度转换
    objectives/              描述交叉熵、连续流匹配、联合损失
    module.py                Lightning训练/验证/测试步
    callbacks.py             保存、历史、计时与状态回调
    trainer.py               Trainer、日志、原生恢复管理
    factory.py               从配置构造两个网络
  generation/                词表与合法性、描述采样、Heun连续采样、生成流水线
  evaluation/                统一CSP/从头生成/性质评价，保留历史协议
  tests/
    unit/                    自动化测试和工程集成检查
    catalog.py               运行前群目录检查
    preflight.py             训练数据门禁
    representation.py        群表示数值审计
    source_integrity.py      原始CIF到展开结构的独立比较
    gradients.py             梯度诊断
    native_identifier.py     隔离原生群识别
    reference_training.py    仅用于历史配对回归的手写循环参考
    reference_workflows/     旧训练调度的重现实验参考
    workflows/               检查、测速和短程对照实验
  environment.yml               依赖文件
```

`pl_data/workflows`与`evaluation/workflows`保留数据和评估任务。当前工作流直接通过模块入口调用，旧命令注册表与薄脚本只在tests/legacy中保留。旧扁平源文件已由对应分层文件替代；历史实验结果、模型、数据及其哈希不会为目录调整而改写。

`tests`中有两类不同用途：`unit`需要pytest，其他运行前检查也会在实际训练/生成中调用，不依赖pytest。不能因为它们位于tests就全部排除安装。

## 安装与配置

在项目根目录、已具备匹配GPU的PyTorch环境中安装：

```bash
python -m pip install --no-deps --no-build-isolation -e ./unifiedgroupgen
```

训练增加`hydra-core>=1.3,<1.4`和`omegaconf>=2.3,<2.4`依赖；如果环境缺少，单独安装，避免替换现有PyTorch。

主配置为`configs/run.yaml`；`configs/experiment/*.yaml`通过Hydra defaults引用原`configs/main15/*.yaml`，没有复制或改写原实验配置。命令行使用OmegaConf覆盖。Hydra配置组合参考官方文档：
https://hydra.cc/docs/1.3/advanced/defaults_list/

```bash
# 仅核对计划，不训练
python -m unifiedgroupgen.run experiment=mp20_u task=plan

# 直接文件入口等价
python unifiedgroupgen/run.py experiment=c2db51_u task=plan

# 显式确认之后才训练
python -m unifiedgroupgen.run experiment=mp20_u device=cuda

# 运行时参数覆盖
python -m unifiedgroupgen.run experiment=joint_u task=plan runtime.batch_size=256 runtime.edge_budget=400000

# 原生断点恢复；使用相同experiment，不修改审计数据、性质配置或运行参数
python -m unifiedgroupgen.run experiment=mp20_u device=cuda resume=outputs/mp20_u/YY-MM-DD-HH-mm-ss/last.ckpt

# 预训练参数初始化是新实验，不恢复原优化器和轮次
python -m unifiedgroupgen.run experiment=mp20_u device=cuda initialize_from=outputs/mp20_u/YY-MM-DD-HH-mm-ss/model.best.pt
```

`resume`与`initialize_from`互斥。历史手写训练的`.pt`可用于预训练权重初始化和推理，不能伪装成Lightning原生恢复。新原生`.ckpt`保存完整训练状态；`.pt`继续保存两个网络的推理权重，保持已有生成评估可读。

新运行在`HYDRA_JOBS/<experiment>/<YY-MM-DD-HH-mm-ss>/`保存权重、历史、计时、状态和`.hydra`配置。日期为本次真实启动时间，历史导入目录的日期含义不变。W&B沿用`.env`设置：1为在线，0为离线；显式`wandb.mode=disabled`可仅记录本地CSV。在线日志需要用户自己的可用登录。

MP20、C2DB、联合数据的新主实验默认沿用批准的256/400000、768/200000、256/400000运行参数，坐标参考长度3Å、损失除以9，不启用激活检查点，边分块32768。恢复运行优先读取原生检查点中的原运行参数；显式冲突会报错。

## 骨架、生成目标与采样的关系

骨架负责特征更新；模型负责特征与输出头；objective负责训练路径和损失；generation负责采样积分。消息传递层拆文件后仍注册在`flow.layers.*`，Transformer仍注册在`proposal.transformer.*`，避免仅因目录变化破坏旧权重名称。

为骨架消融预留的配置字段是`experiment.model.flow_backbone`和`experiment.model.descriptor_backbone`，接受Hydra `_target_`。替换消息层需实现`forward(h, edges, edge_index, degree, chunk_size)`；替换Transformer需接受现有mask接口。骨架参数必须由构造函数注册；新骨架改变权重形状时需要另行训练，不能声称与旧模型等价。

## 验证范围

需要分别检查模块导入、包安装、旧权重严格加载、固定混合SG/LG批次的输出/损失/梯度与采样一致性、Lightning训练与恢复、数据审计、CSP与从头生成评估。短程工程测试不替代正式收敛训练或物理稳定性验证。

完整测试入口：

```bash
python -m pytest -q
```

重构源码备份与验证证据由本次执行记录给出；不删除模型与实验输出，不连接服务器，也不自动提交正式实验。
