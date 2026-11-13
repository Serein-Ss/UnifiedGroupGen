# 当前核验记录

当前仅保留15套正式模型。30份检查点的最新整理核验见[verification.json](validation/verification.json)，目录检查见[final_layout_check.json](validation/final_layout_check.json)，真实结构内嵌CIF回读见[inline_structure_readback.json](validation/inline_structure_readback.json)。这些是软件与存储验证，不是物理稳定性或性质控制证据。

旧模型、旧短程试验、重复归档及临时检查输出已清理。当前训练轨迹和生成评估留在各模型outputs目录，成品图表留在assets。三个基础模型评估的路径与检查点绑定见[当前结果来源](../assets/results/baselines_256_local_seed142_20261026/source_locations.json)。

需要新的软件检查时，可重新执行tests下的对应测试；运行前检查输出写入intermediate/checks，缓存写入intermediate/cache。不会恢复正式训练或评估任务。
