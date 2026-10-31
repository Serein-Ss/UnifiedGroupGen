# 文件存放规则

| 位置 | 职责 | 实验后能否直接清理 |
|---|---|---|
| outputs/<实验>/<日期> | 模型、日志、正式生成与评价 | 否 |
| data/scr | 原始来源 | 否 |
| data/dataset | 编码后的模型输入及划分 | 否 |
| data/dataset/audits | 数据修复证据与训练门禁 | 否 |
| assets | 论文展示PNG与配套表 | 否 |
| docs/references/code_snapshots | 有来源标记的参考代码 | 不作为临时缓存 |
| intermediate/checks | 测试、兼容性、诊断及运行前检查输出 | 相关进程结束后可清理 |
| intermediate/cache | 可重建缓存 | 相关进程结束后可清理 |
| intermediate/staging | 临时打包、解包与转换 | 相关进程结束后可清理 |

artifacts物理目录已拆分取消。历史输入和检查点保留其原路径文本，通过configs/paths.yaml的显式映射定位到新位置；这保留审计身份与权重语义，不表示还存在artifacts目录。

生成结构统一位于outputs/<实验>/<日期>/generate/<任务>，指标位于同一模型的evaluate/<任务>。三个基础模型的已完成评价采用baseline256任务前缀，停止的15模型诊断采用diagnostic256前缀；旧的集中评价目录已经拆分。工程核验与索引放intermediate，完整规则见[结果目录规则](OUTPUT_LAYOUT.md)。旧结果隔离区现已删除。

新代码请直接使用新位置，不新增artifacts写入。历史冻结命令只在tests/legacy保留；并非所有历史命令都是当前入口。训练、生成和指标入口分别为python -m unifiedgroupgen.run、python -m unifiedgroupgen.generate、python -m unifiedgroupgen.evaluate；ugg命令对应训练入口。

## 只读验证与可清理输出

```bash
python -m unifiedgroupgen.pl_data.prepare --mode audit --experiments configs/experiments.yaml --verify-only
python -m unifiedgroupgen.tests.workflows.verify_symmetry_runtime --output intermediate/checks/symmetry_runtime.json
python -m pytest -q --basetemp intermediate/checks/pytest_run
```

最后一条在新建或可重置的测试目录运行；需保留多次检查时，为每次basetemp换一个名称。pytest缓存固定在intermediate/cache/pytest。这里只运行验证，不启动训练。
