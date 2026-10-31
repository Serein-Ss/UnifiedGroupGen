# UnifiedGroupGen代码库

从项目根目录创建环境并安装：

```bash
conda env create -f unifiedgroupgen/environment.yml
conda activate unifiedgroupgen_cuda
python -m pip install --no-build-isolation -e ./unifiedgroupgen
```

环境清单以Linux CUDA训练为主要用途，不注入操作系统环境变量。Windows小型测试是可选使用方式，见[环境说明](../docs/ENVIRONMENT.md)。

## 当前入口

新实验使用`experiment=new_<原名>`，默认`new_joint_u`；删除来源条件并共享两阶段性质可见掩码。旧15套选择器保留用于已有模型。官方候选主干、先验、批量生成和学习式补全见[新模型方案](../docs/MODEL_ABLATIONS_V2.md)。下面mp20_u命令属于旧方案兼容示例，不会把旧权重转换成新模型。

```bash
# 查看训练配置；不启动训练
ugg experiment=mp20_u task=plan
# 正式训练、恢复或权重初始化：只选择一种启动方式
ugg experiment=mp20_u device=cuda
ugg experiment=mp20_u resume=/absolute/path/last.ckpt
ugg experiment=mp20_u initialize_from=/absolute/path/model.best.pt
python -m unifiedgroupgen.generate --cfg job
python -m unifiedgroupgen.evaluate --cfg job
```

唯一控制台名称`ugg`由pyproject.toml的project.scripts生成，对应unifiedgroupgen.run:main，等价于`python -m unifiedgroupgen.run`，只负责训练与计划。生成和指标计算使用上面的独立入口。

## 数据与模型评估

```bash
python -m unifiedgroupgen.pl_data.prepare --config <source.yaml>
python -m unifiedgroupgen.pl_data.prepare --mode joint --config <joint.yaml>
python -m unifiedgroupgen.pl_data.prepare --mode audit --experiments configs/experiments.yaml --verify-only
python -m scripts.evaluate_models evaluation_suite=main15_local evaluation_suite.action=plan
```

具体数据、模型、性质条件、优化、损失、训练、日志和运行参数分组保存在configs中，experiment只选择组合。实验顺序由experiments.yaml维护；新增实验不需要修改Python名单或名称前缀规则。source_domain的现有建模行为没有在这次工程整理中改变。

## 职责

common保存公共函数、结构IO、结果输出和外部预测器；group_constraints保存编译和合法空间；conditional保存性质及硬约束；pl_data保存数据功能；pl_modules只保存模型框架、骨架、损失和权重读取；run.py仅负责训练；generate.py负责生成入口；evaluate.py负责完整指标计算；scripts/evaluate_models.py组织批量任务；generation只保留内存采样算法，配置/任务/保存/进度全部位于generate.py；evaluation只有通用、CSP、DNG、条件指标和utils五个模块。模型组件检查位于tests/components.py，旧命令只在tests/legacy保留。

[代码组织](../docs/CODE_ORGANIZATION.md) · [实验配置](../docs/EXPERIMENT_CONFIGURATION.md)

[统一生成/评估命令与配置](../docs/EVALUATION_TASKS.md)

公开MEGNet形成能/带隙预测及图适配见[公开预训练预测器](../docs/PUBLIC_PRETRAINED_PREDICTORS.md)，通过可选`pretrained` extra安装，不需要自行训练预测器。
