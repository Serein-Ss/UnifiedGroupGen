# 当前联合模型评价入口

<!-- JOINT_THEORY_SCOPE -->
> 2026-11-12版本说明：现行方法为[单网络联合理论](THEORY.md)，实现对照见[IMPLEMENTATION.md](IMPLEMENTATION.md)。本页保留旧入口、旧配置或历史结果的操作契约；其两阶段步骤、参数量或数值不是新模型结果，未注明的新功能不自动继承。

> 本页joint指历史联合数据训练配置，不是新的离散／连续联合生成器。


`joint_u`为联合基础模型，`joint_all`为联合三性质CFG模型。评价分测试集模块诊断、指定组成CSP、固定描述诊断、从头生成、逐群生成，以及性质控制。两模型当前仍需source_domain=0/1上下文，不称为完全来源无条件。

```bash
# 只看参数与计划；不会启动生成或训练
python -m unifiedgroupgen.tests.legacy.evaluation_workflows.evaluate_joint_pair --help
python -m unifiedgroupgen.tests.legacy.evaluation_workflows.evaluate_joint_pair --suite discovery --plan
python -m unifiedgroupgen.tests.legacy.evaluation_workflows.evaluate_joint_pair --suite reconstruction --plan
```

确认计划后才显式使用--run。指定历史权重根目录时使用实际outputs中的位置，不能将导入日期视作训练开始时间。生成/性质/指标分阶段；--resume会核验输入、权重、代码和已有输出哈希。

生成结构默认每任务一个samples.jsonl，含CIF字符串、cif_sha256、请求和失败原因，不逐样本创建文件。选定结构通过`python -m unifiedgroupgen.run task=export_cif`配置samples、index、output后单独导出。当前任务入口见[EVALUATION_TASKS.md](EVALUATION_TASKS.md)，下文历史任务记录不作为新命令入口。

通用和任务指标见[UNIFIED_EVALUATION.md](UNIFIED_EVALUATION.md)。性质代理需要显式提供外部CGDiT源码、FE/BG权重和参考数据；二维跨域代理结果只是探索性评价，不当作已校准精度或物理稳定性。

清理前完整服务器操作记录与旧预测器inventory命令保存在[tests/legacy/docs](../unifiedgroupgen/tests/legacy/docs/JOINT_PAIR_EVALUATION.md)，不再作为当前命令。
