# UnifiedGroupGen

通过群—轨道编译与流匹配生成三维晶体和二维材料。本仓库提供代码、配置、当前训练输入和测试；模型权重、运行日志、生成结构及评估结果不随代码提交。

## 安装与入口

```bash
conda env create -f unifiedgroupgen/environment.yml
conda activate unifiedgroupgen_cuda
python -m pip install --no-build-isolation -e ./unifiedgroupgen

# 查看组合配置，不启动训练。
ugg experiment=new_mp20_u task=plan
python -m unifiedgroupgen.generate --help
python -m unifiedgroupgen.evaluate --help
```

主要面向Linux CUDA服务器；Windows用于可选的小规模测试。CUDA环境、依赖版本与安装验证见[环境说明](docs/ENVIRONMENT.md)。本机路径在`.env`中设置，参考`.env.example`；实际`.env`不提交。

## 代码组织

- `unifiedgroupgen/common/`：配置、结构读写、可视化及共享函数。
- `unifiedgroupgen/group_constraints/`：群—轨道编译器、独立参数编码、展开及投影。
- `unifiedgroupgen/conditional/`：性质、组成与部分结构约束。
- `unifiedgroupgen/pl_data/`：数据编码、准备、批处理与Lightning数据接口。
- `unifiedgroupgen/pl_modules/`：描述Transformer、OrbitFlow、主干网络和损失。
- `unifiedgroupgen/generation/`：结构生成算法；配置和运行入口为`generate.py`。
- `unifiedgroupgen/evaluation/`：通用、CSP、DNG及性质条件指标；运行入口为`evaluate.py`。
- `unifiedgroupgen/run.py`：Lightning训练、恢复与预训练权重加载。
- `unifiedgroupgen/tests/`：测试与数据检查；必要参考代码仅用于兼容和回归比较。
- `configs/`：Hydra功能分组与实验选择器。
- `scripts/`、`assets/datasets/source/`：多模型评估、数据恢复及绘图源码。

用法见[实验配置](docs/EXPERIMENT_CONFIGURATION.md)和[生成与评估任务](docs/EVALUATION_TASKS.md)。

## 数据与实验配置

`data/dataset/full_verified_v1/`保存MP-20、MPTS-52和C2DB当前修复版的训练、验证和测试划分，以及转换清单与审计元数据。原始输入在`data/scr/`；C2DB必要恢复数据和来源资料在`data/scr/source_recovery/`。历史编码版本、重复划分和数据修复过程日志不随仓库发布，见[模型输入数据](data/dataset/README.md)。

现有15套配置为MP-20、C2DB及两者联合，各含基础模型、三个单性质模型及多性质模型。原选择器保留检查点兼容，其中联合配置含`source_domain`；新`new_*`选择器移除来源条件并统一性质可见掩码。配置存在不代表新方案已经训练或验证，主干、先验和条件任务的实现边界见[模型与消融配置](docs/MODEL_ABLATIONS_V2.md)。

C2DB编码使用0.1 Å识别容差，允许对称性理想化，不是逐原子严格无损编码。8,219条带隙标签缺失，缺失未填成0；性质训练按标签可见性处理。

## 测试与结果存放

```bash
python -m pytest -q
```

结果统一写入`outputs/<实验>/<日期>/`，训练文件与`generate/<任务>/`、`evaluate/<任务>/`按[结果目录规则](docs/OUTPUT_LAYOUT.md)组织。W&B记录配置与训练监督信息；临时检查、缓存放在`intermediate/`。这些目录均不纳入代码提交。

[代码组织](docs/CODE_ORGANIZATION.md) · [文件存放规则](docs/FILE_STORAGE.md) · [统一评估协议](docs/EVALUATION_PROTOCOL_V2.md) · [公开性质预测接口](docs/PUBLIC_PRETRAINED_PREDICTORS.md)
