# 当前联合模型训练曲线

2026-10-19新归档核验：训练已完成245轮并早停，共49个真实验证点；最佳为第195轮，验证总损失21.752630749。

![完整训练曲线](training_history.png)

[逐轮CSV](joint_training_history.csv)。曲线未经平滑，未验证轮次不补0；红色虚线与星号表示联合最优检查点，灰色点线表示早停完成轮次。分项最优与联合最优可能不同。训练和验证采用不同条件机制，损失不是材料质量或稳定性指标。

来源为 `assets/results/current_snapshot.json` 指向的新训练历史。旧单来源曲线和PDF/SVG已移入当前图集之外的历史备份。重新绘制：`python assets/training/source/build_result_figures.py`。
