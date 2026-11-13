# 目录整理说明

维护目录见[PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)。旧CLI、命令分发、历史调度及旧操作说明统一归档到unifiedgroupgen/tests/legacy。正式训练只走run.py的Hydra/Lightning入口，生成和指标用各自新模块。

数据位置读取兼容保留，避免改写已有配置、数据和权重；旧命令别名不再放在common中。原始数据、结果、失败记录和审计哈希没有清除或覆盖。

本次安装与功能验证见[VALIDATION.md](VALIDATION.md)及[ENVIRONMENT.md](ENVIRONMENT.md)。清理前的验证历史保存在tests/legacy/docs/REORGANIZATION_VALIDATION.md。
