# 模型结果存放规则

`outputs`（或 `.env` 中指定的 `HYDRA_JOBS`）只保存模型训练、结构生成和指标评估结果。每次训练的目录如下：

```text
outputs/<实验名称>/<YY-MM-DD-HH-mm-ss>/
├── model.best.pt
├── model.pt
├── model.history.jsonl
├── model.status.json
├── train.log
├── generate/
│   └── <生成任务名称>/
│       ├── samples.jsonl
│       ├── summary.json
│       └── sampling.status.json
└── evaluate/
    └── <对应生成任务名称>/
        ├── samples.csv
        ├── summary.json
        └── *.png
```

生成与评估可保存少量任务配置、日志和校验记录；不为每个候选单独写文件。`samples.jsonl` 每行保留一个候选，包括失败记录；成功结构保留内嵌 CIF 字符串及 `cif_sha256`。仅在人工选中结构时，通过 `generate.py task=export_cif` 单独导出 CIF。

## 训练与恢复

新训练直接使用 Lightning `ModelCheckpoint`，将原生检查点的文件扩展名设置为 `.pt`，保存为 `model.best.pt` 与 `model.pt`。二者仍保留原生优化器、调度器、训练循环及必要恢复状态；并不是只有网络权重的导出文件。已有15套历史 `.pt` 不改写，加载器同时识别两种内部格式。历史检查点若不是原生 Lightning 格式，需要 `initialize_from` 初始化新训练，不能冒充原生断点恢复。

历史的15套日期目录为导入时间，不是训练开始时间；新训练默认按真实开始时间创建目录。目录中不再写 `provenance/`、`run.json`、`.hydra/`、CSV日志目录或独立梯度文件。

训练参数、完整解析配置、运行参数、数据划分哈希和损失使用 W&B 原生功能记录；W&B 文件写在 `WANDB_DIR`。在线与离线模式均支持。关闭 W&B 时仍保留五个训练文件和检查点内的必要配置，不另造配置快照。检查点中的 W&B run ID 用于续训关联同一运行。

```bash
python -m unifiedgroupgen.run experiment=new_joint_u device=cuda
python -m unifiedgroupgen.run experiment=new_joint_u device=cuda resume=/path/to/model.pt
```

## 单任务生成与指标

省略 `generation.output` 时，生成目录从指定的检查点位置推导；可以用 `generation.task_name` 明确区分不同预算、随机种子和性质目标，避免覆盖已有结果。

```bash
python -m unifiedgroupgen.generate generation.checkpoint=/path/to/model.best.pt generation.task_name=space_dng_seed142 generation.kind=space generation.count=256
python -m unifiedgroupgen.evaluate evaluation.samples=/path/to/generate/space_dng_seed142/samples.jsonl evaluation.task=dng evaluation.kind=space
```

旧联合模型仍需提供其原有来源上下文。上面的简例不代表可以省略旧模型必需的条件。指标的参考文件、条件预测器等由对应配置补充。

若样本位于 `<模型运行>/generate/<任务>/samples.jsonl`，省略 `evaluation.output` 时自动输出到 `<模型运行>/evaluate/<任务>`。外部样本仍可显式指定输入与输出位置。

## 多模型调度与可清理内容

`scripts/evaluate_models.py` 的计划、锁、总进度及跨模型比较放在 `intermediate/evaluation_campaigns/<批次>`；结构与逐任务指标仍归属检查点所在的模型目录。组件检查结果放在该模型 `evaluate/components` 中。调度目录不能被当成额外训练模型。

代码检查、环境检查、归档索引和迁移凭据放在 `intermediate`，不放进 `outputs`。论文成品图表放 `assets`。服务器 ZIP 导入工具也只向模型目录复制五种训练文件，导入清单保存在 `intermediate/training_imports`。

这次整理保留两个已有评估批次的区别：`baseline256_*_seed142` 对应已完成的三个基础模型评估，`diagnostic256_*_seed142` 对应后来停止的15模型诊断评估。迁移不会补做指标；未完成的结果仍然是未完成。
