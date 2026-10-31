# PyXtal与层群运行前检查

当前依赖固定为PyXtal 1.1.3。检查覆盖80个层群、477种单Wyckoff轨道的群操作闭合、轨道基底秩与群相容晶格。错误版本或错误目录应阻止运行；检查成功后在同一进程缓存，不进入每个训练batch。

这是一项数学/软件运行前检查，不是结构生成质量或物理稳定性。现有15套模型的数据门禁另行检查。

```bash
python -m unifiedgroupgen.tests.workflows.verify_symmetry_runtime --output intermediate/checks/symmetry_runtime.json
python -m unifiedgroupgen.pl_data.prepare --mode audit --experiments configs/experiments.yaml --verify-only
```

需要检查实际权重时，可为第一条命令加--checkpoint outputs/<实验>/<日期>/model.best.pt。该工具在CPU严格加载并检查声明数据，不会启动训练。请使用assets/results/main15/model_inventory.csv确认权重位置，日期为历史导入时间。

运行产生的兼容性报告放intermediate/checks，检查结束后可清理。训练依赖的永久数据审计保留在data/dataset/audits，不能随临时日志删除。较早兼容性报告与缓存已归档，不再用旧模型测试数字描述当前15套模型。
