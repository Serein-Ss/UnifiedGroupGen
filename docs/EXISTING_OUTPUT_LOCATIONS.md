# 当前结果位置

| 内容 | 位置 |
|---|---|
| 15套权重与训练日志 | outputs/<实验>/<日期>/下的五个训练文件 |
| 原始生成候选、内嵌CIF | outputs/<实验>/<日期>/generate/<任务>/samples.jsonl |
| 指标CSV、JSON与PNG | outputs/<实验>/<日期>/evaluate/<任务>/ |
| 已完成的基础模型评估 | 各模型下baseline256_*_seed142任务 |
| 已停止的15模型诊断产物 | 各模型下diagnostic256_*_seed142任务；不代表指标已完成 |
| 跨模型计划与进度 | intermediate/evaluation_campaigns/ |
| 当前模型索引与小型核验 | assets/results/main15/ 与 docs/validation/ |
| 2520轮合并训练表 | assets/training/main15/training_history.csv |
| 基础模型结果PNG和表 | assets/results/baselines_256_local_seed142_20261026/ |
| 数据审计与来源修复 | data/dataset/audits/ |

outputs不保存工程验证、独立索引或集中调度目录。论文成品图表继续放assets。见[结果目录规则](OUTPUT_LAYOUT.md)。
