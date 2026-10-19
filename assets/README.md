# UnifiedGroupGen 当前图表与结果

更新日期：2026-10-19。最新服务器归档已核验；统一来源见 [current_snapshot.json](results/current_snapshot.json)。旧统计展示已移出当前图集，备份保存在 `artifacts/result_archive/pre_refresh_20261019/previous_presentations.zip`。所有当前图均为PNG。

## 当前结果

| 项目 | 已核验结果 |
|---|---|
| 联合训练 | 245轮完成并早停，49个真实验证点 |
| 最优检查点 | 第195轮，验证损失21.752631；与此前评估权重逐字节相同 |
| 训练／验证／测试 | 82,185／10,190／10,235；九份划分与新包哈希一致 |
| 统一表示 | 102,610条准备记录；2,280种单轨道目录通过；原始CIF重编码抽查48条 |
| 共享模型采样 | 三来源各两种模式，每组256次，共1,536次；新包七组评估输出与之前一致 |
| 原始精确群回收 | 1,526/1,536；标准CIF解析口径1,525/1,536 |
| 标准联合有效／VUN | 1,316/1,536；961/1,536 |

训练延长没有产生新的最优权重，也没有新增一轮采样；生成指标未冒充重新评估。性质控制、物理稳定性、公平基线、独立与共享训练对照仍待补。

## 图集入口

- [完整训练曲线](training/training_history.png) · [训练口径与CSV](training/README.md)
- [数据集图集](datasets/ATLAS_README.md) · [统计说明](datasets/README.md)：数据字节未变，统计值保持不变并重新核验来源。
- [数学构造验证](representation/README.md)
- [论文主图](paper/README.md) · [标准质量指标](paper/standard/main_results.csv)
- [当前结果与待补实验](results/RESULTS_PLAN.md)
- [总文章](../docs/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md)
- [本次更新与文件清单](../docs/RESULTS_UPDATE_20261019.md)

历史单来源64次采样图及旧PDF/SVG不再混入当前图集。对称性闭合、精确群、化学筛查、性质命中与稳定性属于不同证据。
