# UnifiedGroupGen 当前图表与结果

> **2026-10-22 来源一致性修订：** 68条旧编码需处理，66条修复、2条MPTS52隔离；正式数据102,608条，联合划分82,183/10,190/10,235。M1/M2/M3均需干净重训，以下旧训练与生成指标保留历史身份。详见[完整审计与迁移](../docs/SOURCE_INTEGRITY_AUDIT_20261022.md)。


更新日期：2026-10-19。最新服务器归档已核验；统一来源见 [current_snapshot.json](results/current_snapshot.json)。旧统计展示已移出当前图集，备份保存在 `artifacts/result_archive/pre_refresh_20261019/previous_presentations.zip`。所有当前图均为PNG。

## 历史模型结果（数据修订前）

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
- [数据集图集](datasets/ATLAS_README.md) · [统计说明](datasets/README.md)：来源一致性修订后重绘，正式记录102,608条；2条源输入隔离。
- [数学构造验证](representation/README.md)
- [论文主图](paper/README.md) · [标准质量指标](paper/standard/main_results.csv)
- [当前结果与待补实验](results/RESULTS_PLAN.md)
- [总文章](../docs/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md)
- [本次更新与文件清单](../docs/RESULTS_UPDATE_20261019.md)

历史单来源64次采样图及旧PDF/SVG不再混入当前图集。对称性闭合、精确群、化学筛查、性质命中与稳定性属于不同证据。
