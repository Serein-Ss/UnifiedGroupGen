# 来源核验修订版追加数据包

先确保已导入上一级datasets的原始完整数据包；此追加包仅提供修复后的源编码、新联合划分、审计与隔离清单，不重复打包原始CSV和官方恢复数据库。2个分片共66,474,910字节，53个受哈希保护文件，正式记录102,608条；原始102,610条中的2条MPTS52输入隔离。

```bash
python scripts/unpack_training_data.py --bundle datasets/source_integrity_v1
python scripts/verify_training_data.py
```

新配置为mp20_verified_v1、mpts52_verified_v1、c2db_verified_v1和joint_verified_v1。使用新输出目录从头训练，不接续旧检查点。完整说明见docs/SOURCE_INTEGRITY_AUDIT_20261022.md。原始数据、旧编码和旧权重保留。
