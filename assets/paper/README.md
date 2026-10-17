# 论文图表输出

新论文统计图只保存PNG。运行 scripts/build_paper_results.py 后，本目录包含真实已完成评估对应的图、数据表和输入指纹。缺失的性质或物理计算不会生成模拟结果。

完整图1–6设计、各图应支持的结论和当前缺口见 [论文证据计划](../../docs/PAPER_EVIDENCE_PLAN.md)；命令见 [服务器执行流程](../../docs/PAPER_WORKFLOW.md)。现有数据图、表示检查和模型架构分别在 ../datasets、../representation、../architecture。

| 输出 | 内容 |
|---|---|
| 03_shared_generation_validity.png | 三来源无条件生成有效性和精确群回收，全部尝试分母 |
| 03_shared_generation_coverage.png | COV-P/R、有效唯一新颖产率，每个采样任务独立显示 |
| 03_shared_generation_by_dof.png | 按坐标自由度分层的联合有效率与数量 |
| 04_continuous_controls.png | 同描述/种子的合法先验、训练流、固定晶格对照 |
| 04_observation_residuals.png | 固定观测残差累计分布 |
| 05_condition_response.png | 独立性质值相对请求目标的中位数/IQR与CFG扫描 |
| 05_condition_valid_yield.png | 联合有效率与有效性质命中产率 |
| 06_physical_screening.png | 有兼容能量参考的弛豫后筛选与SUN产率 |
| 06_relaxation_symmetry.png | 弛豫前后均成功识别的样本中，回收群不变的比例与计数 |

本次仅建立输出目录和绘图程序；服务器正式联合生成结果尚未同步回本地，因此不以测试夹具填充这些论文图。
