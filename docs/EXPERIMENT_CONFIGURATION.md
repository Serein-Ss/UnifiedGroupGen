# 分组配置与实验组合

训练使用Hydra的defaults组合配置。实验文件只选择数据、性质条件和运行环境，不再复制整套网络与训练参数。

2026-10-31增加的无来源新模型使用`experiment=new_<原名>`，默认`new_joint_u`；对应`experiments_v2.yaml`、`data/*_unified.yaml`、`learning/`和`prior/`。旧15套选择器保留用于历史模型复现。新骨架、共享性质掩码、组成/补全训练及先验组合见[MODEL_ABLATIONS_V2.md](MODEL_ABLATIONS_V2.md)，下面原15套说明属于兼容配置。

## 目录职责

```text
configs/
  run.yaml                         # 训练入口、路径、seed、恢复选项
  data/{mp20,c2db51,joint}.yaml      # 数据路径、编码约定、划分、来源上下文、审计
  model/default.yaml               # 模型容量上限，组合下面两个网络配置
  model/descriptor/default.yaml    # 描述Transformer
  model/orbitflow/default.yaml     # OrbitFlow
  conditioning/                    # none、三种单性质、multi_property
  optim/default.yaml               # 学习率与学习率调度耐心
  loss/default.yaml                # 各项权重、3 Å坐标尺度
  train/default.yaml               # 轮次、精度、梯度裁剪、验证与早停
  logging/wandb.yaml               # W&B项目与模式
  runtime/default.yaml             # 通用执行参数，从功能分组引用默认值
  runtime/rtx4090_crystal.yaml      # MP-20/联合：batch256、边预算400000
  runtime/rtx4090_layer.yaml        # C2DB：batch768、边预算200000
  experiment/*.yaml                # 15个轻量组合选择器
  experiments.yaml                # 实验有序目录，评估和数据检查也读取它
  generate.yaml + generation/      # 生成入口与任务选项
  evaluate.yaml + evaluation/      # 指标入口与计算协议
  evaluation_campaign.yaml
  evaluation_suite/                # 批量生成/评估预算与调度选项
  check_components.yaml
  component_check/                 # 模型组件检查
  property_predictor/、predictors/ # 可选外部性质预测器
  paths.yaml                       # 已有数据的逻辑路径挂载
```

根目录中的`*_verified_v1.yaml`等源数据配置是已有转换清单绑定的原始输入，保留其字节以核验来源；新的15套训练不再从它们继承网络和训练参数。`configs/main15`已移出活动配置区，原文件保存在`unifiedgroupgen/tests/legacy/configs/main15`，只用于历史测试与一致性比较。

## 一个实验怎样组成

例如`configs/experiment/joint_all.yaml`：

```yaml
# @package _global_
defaults:
  - override /data: joint
  - override /conditioning: multi_property
  - override /runtime: rtx4090_crystal
experiment_name: joint_all
```

`run.yaml`先选择各功能组的默认值，再由实验文件替换其中的组；命令行覆盖最后生效。所有模型共享`model`、`optim`、`loss`和`train`默认配置，避免修改15份文件。

数据配置中的`context_properties`单独记录必要上下文，`conditioning.properties`记录训练目标性质。通用加载函数将两者合并；联合数据还将`data.property_overrides.per_domain`应用到目标性质。当前联合模型仍保留`source_domain`，这次仅改变配置组织，没有移除来源标签或改变模型输入。

```bash
# 只查看，不训练
python -m unifiedgroupgen.run experiment=joint_all task=plan

# 替换性质组；查看单一带隙设置
python -m unifiedgroupgen.run experiment=joint_all conditioning=band_gap task=plan

# 学习率、网络深度、损失权重或运行参数的覆盖
python -m unifiedgroupgen.run experiment=mp20_u optim.learning_rate=0.0001 model.proposal_layers=6 task=plan
python -m unifiedgroupgen.run experiment=mp20_u loss.coordinate_length_scale=3 runtime.batch_size=128 task=plan

# 通用执行默认值；FP32便于CPU小型测试
python -m unifiedgroupgen.run experiment=mp20_u runtime=default train.precision=fp32 device=cpu task=plan
```

`optim`配置仍使用原有的原生AdamW（默认weight_decay=0.01）与ReduceLROnPlateau（factor=0.5），没有更换优化器。`lr_patience`以训练轮数计，在Lightning入口除以验证间隔后得到调度器的验证次数耐心。

## 审计定义与实际运行参数

历史数据审计的`config_sha256`绑定完整的扁平训练定义，而不只是数据路径。因此，`train`中的batch64、边预算200000，以及`model/orbitflow`中的重计算true、边分块4096，是保留的**审计定义默认值**。它们不是4090最终执行的参数；4090的实际参数由`runtime/rtx4090_*.yaml`覆盖。

| 设置 | MP-20与联合 | C2DB |
|---|---:|---:|
| 实际batch上限 | 256 | 768 |
| 实际边预算 | 400000 | 200000 |
| 边分块 | 32768 | 32768 |
| 激活重计算 | false | false |
| 验证间隔 | 5轮 | 5轮 |
| 坐标尺度 | 3 Å，坐标损失除以9 | 同左 |

`runtime/default.yaml`通过插值引用`train.validation_interval`和`loss.coordinate_length_scale`，后两项只有一处定义。训练把解析后的运行参数传给数据批处理和损失函数，并记录在检查点中。

`common/configuration.py::training_definition`将功能分组转换为原有扁平模型/数据接口；15套默认定义的字典、审计哈希和运行参数均与重构前比较。历史权重无需转换、数据审计无需重新签发。改变被审计的定义字段后仍需重新审计，不能绕过检查。可在`experiments.yaml`对应条目添加`overrides`，使数据审计和训练使用同一套自定义参数。

## 新增实验、恢复与记录

新增实验只需添加一个选择器，再按需要加入实验清单：

```yaml
experiments:
  my_experiment:
    experiment: my_experiment
    overrides: [optim.learning_rate=0.0001]
    # checkpoint: outputs/my_experiment/<date>/model.best.pt
```

加载器不包含15模型名单或数据集名称判断。独立的扁平YAML仍可通过清单中的`config`和`runtime`字段加载，供外部数据配置使用；项目内新实验统一使用Hydra选择器。

```bash
# 以下只有检查，不启动训练或采样
python -m unifiedgroupgen.pl_data.prepare --mode audit --experiments configs/experiments.yaml --verify-only
python -m scripts.evaluate_models evaluation_suite=main15_local evaluation_suite.action=plan evaluation_suite.output=intermediate/checks/main15_plan

# 需要训练时另行明确执行；这次重构未启动这些命令
python -m unifiedgroupgen.run experiment=mp20_u device=cuda
python -m unifiedgroupgen.run experiment=mp20_u resume=/absolute/path/model.pt
python -m unifiedgroupgen.run experiment=mp20_u initialize_from=/absolute/path/model.best.pt
```

恢复Lightning原生检查点时，未显式覆盖的运行参数沿用检查点；显式覆盖仍受原有兼容检查约束。`resume`与`initialize_from`互斥：前者恢复优化器等状态，后者只加载权重开始新实验。

训练配置通过W&B原生超参数记录及Lightning检查点保存，不另写`.hydra`或`provenance`配置快照。W&B模式可用`logging.mode=online/offline/disabled`覆盖；null沿用`.env`中的开关。

生成和指标计算继续使用独立入口：

```bash
python -m unifiedgroupgen.generate --cfg job
python -m unifiedgroupgen.evaluate --cfg job
```

它们分别读取`generation`与`evaluation`等任务组；批量评估通过同一个Hydra实验加载器得到定义，并以检查点保存的训练参数构建已训练模型。当前已停止的旧评估目录保留原源码身份和全部样本，不在新源码下自动续跑。

替换整个网络子配置时，可新增`model/descriptor/<name>.yaml`或`model/orbitflow/<name>.yaml`，再用Hydra的包定位语法选择：

```bash
python -m unifiedgroupgen.run experiment=mp20_u model/descriptor@model=my_transformer task=plan
python -m unifiedgroupgen.run experiment=mp20_u model/orbitflow@model=my_flow task=plan
```

这里两个子组的参数合并到同一个`model`节点，仍分别由各自文件维护；需要改变模型实现时，使用已有的可替换骨架配置接口，而不是只改YAML里的网络名称。
