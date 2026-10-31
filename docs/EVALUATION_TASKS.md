# 独立生成与指标入口

训练使用 `run.py`（或 `ugg`），生成使用 `generate.py`，指标计算使用 `evaluate.py`。三者分别加载 `configs/run.yaml`、`generate.yaml`、`evaluate.yaml`，继续使用Hydra配置。指标模块本身不包含采样或任务调度。

## 指标目录

|文件|职责|
|---|---|
|`evaluation/common.py`|逐结构有效性、组成守恒、群操作残差、精确群一致性及共同汇总|
|`evaluation/csp.py`|指定组成的结构重建、两套匹配参数、首1/5/20个候选的Match@K及匹配结构误差|
|`evaluation/dng.py`|从头生成的唯一性、新颖性、VUN、指纹覆盖和结构分布比较|
|`evaluation/conditional.py`|可选的独立性质预测误差、目标命中、联合命中及性质分布距离|
|`evaluation/utils.py`|距离/几何、SMACT、指纹、匹配参数、Wasserstein/JSD/Fréchet等计算函数|
|`evaluation/resources/benchmark_comp_scaler.json`|原指标协议的成分指纹标准化常数，非模型权重|

这些模块没有命令行入口、模型加载、任务队列、绘图保存或预测器调用。`utils.py`保留分立的小函数，不把不同协议混为一个计算。原有独立最近邻覆盖与同一邻居联合覆盖继续分别输出；严格与DiffCSP匹配参数继续分别标注，二维使用明确标注的面积归一化适配。

`run.py`仅负责训练；`generate.py`加载生成配置和权重并调用`generation`；`evaluate.py`读取已有结构、计算指标、调用可选性质预测器并输出表图。`scripts/evaluate_models.py`仅承担多模型任务组织；`tests/components.py`负责组件检查；`common/structures.py`负责结构读写，`common/reports.py`负责图表输出。

参考[DiffCSP++的compute_metrics.py](https://github.com/jiaor17/DiffCSP-PP/blob/main/scripts/compute_metrics.py)中共享结构处理、RecEval与GenEval分工。这里保留二维处理和已有指标协议，没有复制其三维假设。

## 对现有结构计算指标

先在项目根目录安装新入口：

```bash
python -m pip install --no-deps --no-build-isolation -e ./unifiedgroupgen

python -m unifiedgroupgen.evaluate device=cpu \
  evaluation.task=dng evaluation.kind=space evaluation.dataset=mp_20 \
  evaluation.samples=/absolute/path/task/samples.jsonl \
  'evaluation.reference_files=[data/dataset/full_verified_v1/mp_20/test.jsonl]' \
  'evaluation.novelty_files=[data/dataset/full_verified_v1/mp_20/train.jsonl]' \
  evaluation.output=outputs/evaluations/my_dng
```

CSP改为`evaluation.task=csp`，每条记录必须保留`case_id`或`reference_id`。参考结构只用于对比，不作为DNG生成输入。请求文件可写`case_ids`、`attempts_per_case`并通过`evaluation.request=/absolute/path/request.json`绑定；也可直接用YAML映射提供request。单个测试例子生成20个候选后，共用同一序列的前1/5/20个候选，不重新选择有利样本；失败保留原槽位。

二维改为`evaluation.kind=layer`并替换对应数据路径。所有生成账本可内嵌CIF字符串，不需要逐结构CIF文件。

只导出感兴趣的一条结构时使用`python -m unifiedgroupgen.generate task=export_cif export.samples=/absolute/path/samples.jsonl export.index=3 export.output=outputs/selected.cif`。保留已有CIF字符串的原始字节与哈希，不改写生成账本，也不覆盖已有CIF。

指标输出`samples.csv`、`summary.json`、PNG及哈希manifest；CSP另有`cases.csv`和`csp_metrics.csv`。单次计算没有resume参数：已完成目录不覆盖，重新计算请选择新目录。批量调度的`skip_completed=true`只在核验输入、代码、参数及输出哈希后跳过已完成任务；中断的指标任务整体重算。参考指纹缓存只是避免重复计算相同真实结构的特征，不替代模型生成或指标计算。历史结构可以在原始CIF/账本哈希核验后复用，但历史指标不会被当成新协议结果。

## 可选性质评价

默认`evaluation.conditional.enabled=false`，不计算性质命中或性质分布。它不关闭组成/元素硬约束、群约束、观测残差或CSP匹配，这些仍属于基本结构评价。

已获得独立预测CSV时启用：

```bash
python -m unifiedgroupgen.evaluate device=cpu \
  evaluation.task=dng evaluation.kind=space evaluation.dataset=mp_20 \
  evaluation.samples=/absolute/path/task/samples.jsonl \
  evaluation.output=outputs/evaluations/my_conditional \
  evaluation.conditional.enabled=true \
  evaluation.assessments=/absolute/path/proxy/predictions.csv \
  evaluation.assessment_metadata=/absolute/path/proxy/evaluator.json \
  evaluation.reference_assessments=/absolute/path/proxy/reference_predictions.csv \
  'evaluation.reference_files=[data/dataset/full_verified_v1/mp_20/test.jsonl]'
```

目标与容差来自`request`或生成账本，预测值来自独立模型。目标值不输入预测器。缺失/失败预测按已启用指标的失败计入分母。没有预测器时不以生成条件充当预测值，也不声称性质控制已经验证。性质预测器配置与命令见[公开预测器](PUBLIC_PRETRAINED_PREDICTORS.md)。

## 结构生成

```bash
python -m unifiedgroupgen.generate device=cuda seed=142 \
  generation.checkpoint=/absolute/path/model.best.pt \
  generation.output=outputs/generation/my_samples generation.kind=space \
  generation.count=256 generation.steps=64
```

性质在`generation.properties`中指定，配合`generation.guidance`。组成、化学式、元素集合、固定群/描述和部分结构保留原硬约束逻辑，分别由`composition/formula/chemical_system/allowed_elements/required_elements/group/descriptor/scaffold/observations`配置。联合旧模型仍需其训练时使用的`source_domain`上下文；未在这次整理中移除。

## 模型组件检查

```bash
python -m unifiedgroupgen.tests.components device=cuda seed=142 \
  component_check.checkpoint=/absolute/path/model.best.pt \
  component_check.output=outputs/checks/my_components component_check.batch_size=8
```

使用完整、经审计的数据检查：描述Transformer的G/W/Z/STOP教师强制NLL和准确率，OrbitFlow在真实群描述及有噪声连续状态上的流匹配误差。输出逐条CSV、汇总JSON和组件图。这既不是组成→结构重建率，也不是从头生成质量。后两者必须通过实际采样再分别计算CSP/DNG指标。

## 从实验目录调度

```bash
# 默认只生成计划，不训练或生成结构
python -m scripts.evaluate_models device=cpu seed=142 \
  'evaluation_suite.models=[joint_u,joint_all]' \
  'evaluation_suite.tasks=[components,dng,csp]' \
  evaluation_suite.output=outputs/evaluations/joint_pair

# 检查计划之后，显式运行。只生成，不计算指标
python -m scripts.evaluate_models device=cuda seed=142 \
  'evaluation_suite.models=[joint_u,joint_all]' \
  'evaluation_suite.tasks=[components,dng,csp]' \
  evaluation_suite.output=outputs/evaluations/joint_pair \
  evaluation_suite.action=run evaluation_suite.stage=generate evaluation_suite.skip_completed=true
```

相同配置下再设`stage=metrics`，只复用已核验生成结果计算指标；`stage=all`顺序执行两阶段。`action=status`查看进度，`action=verify`核验生成账本与CIF哈希。`conditional_tasks=[dng,csp]`额外安排训练集分位数性质目标的采样；性质评分开关`conditional_enabled=true`单独控制。生成条件与性质评分是两件事。

默认CSP为完整测试集、20候选；`csp_cases>0`必须明确标为诊断子集。`per_group=5`启用逐空间群/层群采样。生成阶段遇到其他GPU计算进程会拒绝抢占；不会停止其他任务。

## 历史边界

旧命令只归档在`unifiedgroupgen/tests/legacy/evaluation_before_tasks`，用于历史复现和数值对比，当前代码不调用该目录。重构只调整组织与入口，不重训练、不替换权重、不重写历史结果。训练数据和历史权重保持原始字节。当前15套配置改为Hydra功能分组；原整套配置只归档在`unifiedgroupgen/tests/legacy/configs/main15`用于等价性核验，见[实验配置](EXPERIMENT_CONFIGURATION.md)。

## 此前指标目录整理的验证（历史记录）

完整回归初次为381通过、2失败；旧命令复用检查在源码稳定后独立复跑通过，历史指纹测试的文件引用从standard.py修正为utils.py后通过。合并完整回归、定向复跑及新增测试，覆盖当前385个测试，385通过，无遗漏或未解决失败；不是宣称又执行了一次全绿的完整回归。

其中新入口端到端检查9项通过；另有公式函数体AST一致性检查、严格/DiffCSP双协议及3D/2D的CSP/DNG数值对比、条件评价开关和完整分母测试。此前实际MEGNet权重经当时入口在TiO₂、MoS₂上验证通过，不作为生成模型质量或预测器精度的新证据。

15套配置原始字节、150个保留模型/历史/日志文件的SHA256不变；模型、训练函数和既有群约束实现AST不变。安装、ugg入口和pip check通过。详见历史工程记录（已清理）；原始检查输出位于intermediate/checks/evaluation_tasks。

## 本轮本地15模型评估预算

`evaluation_suite=main15_local`默认只列计划。诊断预算为每模型DNG256、指定组成256例×20候选、固定描述256例×1、每空间群/层群5次；联合模型普通任务的256预算在两来源各分128。性质模型另生成每个训练目标的10/50/90%训练集分位数，单目标每档128，联合模型每来源64；多性质模型再用训练集真实联合目标组合生成三档。仅条件DNG安排额外性质扫描，没有在本轮额外展开条件CSP。

组件检查使用完整测试集；CSP属于可复现的哈希抽样诊断子集，不能标成全测试集Match@K。20候选的同一序列分别计算Match@1/5/20；固定描述只有1候选，是更强输入信息的连续生成诊断。所有失败保留在原位置和分母内。

三个已评估基础模型的5404条结构和24854条组件记录已经逐文件核验。其已生成候选保留原始种子与CIF内容，只补齐缺少的候选；当前指标重新计算。已有候选的历史耗时不混入本轮采样速度。联合模型仍输入`source_domain`，本轮不宣称移除了来源条件。

公开MEGNet只提供形成能与带隙代理，未评估凸包能或物理稳定性。C2DB上的预测属于未经校准的探索性迁移。指标中若有`sampling_task`或`sampling_protocol`，它们仅说明读入结构来自什么任务、使用何种分母，并不会调用采样器。

只读查看本地进度（执行一次即退出，不持续轮询）：

```bash
python -m scripts.evaluate_models evaluation_suite.action=status evaluation_suite.output=intermediate/evaluation_campaigns/<批次>
# 完成后使用evaluation_suite.action=verify核验输出文件和内嵌CIF哈希
```

## 独立指标入口的本轮验证与执行

2026-10-31：28项首组检查、67项入口/指标/模型回归及最后15项针对改动的复跑均通过；这些套件有重叠，不累加为独立测试数量。真实RTX 3060上用保留检查点完成3次64步生成；公开MEGNet在TiO₂和MoS₂上的批量转换、原接口对齐及二维真空间隔不变性检查通过，新的evaluate.py可自动调用预测器后计算指标。上述检查不充当正式生成质量结果。

模型/训练函数和群约束的AST、15套配置以及150个保留模型/历史文件均核验未改变。验证清单：历史工程记录（已清理）。原始日志在`intermediate/checks/evaluate_entry`。

原15模型诊断已经按用户要求停止，其控制记录保存在历史工程记录（已清理）。原计划包含175任务、107660个候选，但没有全部完成，不能按计划数报告已得结果。已保存的样本归入对应模型`generate/diagnostic256_<任务>_seed142`，组件检查归入`evaluate/diagnostic256_components_seed142`。

新调度将计划、锁、跨模型进度和比较表放`intermediate/evaluation_campaigns/<批次>`；实际结构放在检查点旁`generate/<任务>/samples.jsonl`，指标在对应的`evaluate/<任务>/`。采样结束不等于指标完成。没有恢复旧任务，也没有创建持续监控。详见[结果目录规则](OUTPUT_LAYOUT.md)。

## 生成目录与入口分离（2026-10-31）

`generation/`只保留`descriptor.py`、`flow.py`、`rules.py`、`structure.py`和包标识。`structure.sample_orbits`接收模型与明确条件，返回编译空间；`structure.sample_geometry`接收连续模型与编译空间，返回独立状态及展开几何。它们不读YAML、检查点、测试集或观测文件，不选择批量任务，不创建目录或打印进度。

请求文件解析、组成/性质/固定描述/部分结构请求的组织、检查点加载、CSP参考选择、逐群循环、种子管理、文件写入、失败记录和已有候选核验全部在`generate.py`。原有Hydra配置和`python -m unifiedgroupgen.generate`命令保持不变。普通请求与批量请求现在统一写内嵌CIF的JSONL，包括逐个采样；只在显式导出指定结构时创建独立CIF。

本地15模型评估及其每小时检查按用户要求保持暂停。旧评估绑定重构前的代码哈希；源码快照位置记录在`intermediate/checks/generation_entry/migration.json`。恢复时必须使用该快照或另建新版本评估，不能跳过代码身份检查把两个版本混合。


生成入口分离的验证：109项回归和最后43项定向复跑均通过（套件有重叠，不相加）。配对覆盖3D/2D、CSP/DNG/固定描述/逐群、CFG、先验、独立参数观测与部分结构片段；相同权重和种子下状态、记录及CIF一致。重新安装、pip check和wheel独立导入通过。初次wheel只在Windows临时目录清理时遇到占用，改用intermediate内的构建目录后完成并核验了新模块与测试资源。验证清单为历史工程记录（已清理）。

原15模型评估和定时检查继续暂停。旧源码快照为历史工程记录（已清理），其全部原始运行代码已核验哈希；本轮结果目录的`source_refactor.json`记录恢复边界。新入口在写入前核对上层评估目录绑定的代码版本，拒绝混用。
