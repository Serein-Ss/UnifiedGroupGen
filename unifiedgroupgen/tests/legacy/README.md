# 历史命令与兼容归档

这里保留原CLI、命令注册与旧源码映射、旧三来源评估、历史控制器及其薄脚本。docs保存历史操作文档与清理前README。历史协议用于解释既有实验，不作为当前正式入口。

需要重放时从项目根目录显式调用，例如：

```bash
python -m unifiedgroupgen.tests.legacy.cli --help
python unifiedgroupgen/tests/legacy/scripts/evaluate_joint.py --plan
```

新训练、生成与统一评估仅使用run.py、generation.cli、evaluation.workflows。旧脚本的文件位置已迁移，归档文档中的原命令是历史记录；请使用这里对应薄入口或legacy.workflows。数学算法和历史指标协议仍保留可调用实现，已生成数据、哈希、失败记录不改写。
