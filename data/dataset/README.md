# 模型输入数据

- `full_verified_v1/`：当前正式修复版本；15套配置的MP-20、C2DB及两者联合读取其中的来源划分。
- `full_verified_v1/mpts_52/`：MPTS-52当前修复版；不是MP-20/C2DB的15套实验组成部分。
- `audits/`：当前数据检查所需元数据，不包含历史修复过程日志或重复完整数据。
- `test_fixtures/`：回归测试的固定结构数据，不是正式训练集。

数据文件保持原始字节，Git换行转换不作用于本目录。旧配置中的`artifacts/...`输入路径由`unifiedgroupgen.common.configuration.resolve_path`映射到当前目录，无需改写记录与哈希。

本地保留的其他历史编码或划分不随代码仓库发布。模型权重、训练日志、生成结构和指标写入`outputs/`，不放在数据目录，也不随仓库提交。
