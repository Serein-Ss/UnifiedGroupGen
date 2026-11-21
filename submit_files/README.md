# 5090 提交入口

在项目目录中先配置 `.env` 和计算环境。以下命令仅查看计划。

```bash
python submit_files/joint_flow5090.py plan
```

默认选择 MP-20 EquiformerV2 的 DNG 和 CSP。也可用 `--models` 显式选择新实验。确认环境与计划后，用新的批次名提交。

```bash
python submit_files/joint_flow5090.py submit --name NEW_CAMPAIGN
```

每个作业读取并固定提交时的源码和 YAML，依次核验当前 Tensor 缓存、在所分配的 GPU 上校准批大小并训练。日志和校准记录位于 `intermediate/hpc_runs/<批次>/`，模型与训练结果位于 `outputs/<模型>/<批次>/`。

`--retry` 用于同一批次、冻结源码、缓存和运行配置下的原生续训。开始新实验时使用新批次名。历史批次迁移和专用续训脚本在 [归档](../unifiedgroupgen/tests/legacy_code/fresh_core_20261119/README.md) 中。
