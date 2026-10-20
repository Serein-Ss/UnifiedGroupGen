# 三模型训练与原始生成诊断

由 `scripts/build_gallery_model_comparison.py` 从最新服务器归档提取，2026-10-21整理。只重绘已有观测，不进行新训练或采样。

- `all_models_training_comparison.png`：总损失、描述Transformer、流匹配三子图；每图叠加M1/M2/M3训练和验证轨迹。750条完整epoch记录；真实验证点分别71/37/49。星号均为各模型总验证损失选定的最优轮次，不是每项各自最优。描述训练损失下降而验证损失后期上升，提示该模块存在泛化差距，不能仅凭总损失宣布所有模块收敛良好。
- `all_models_generation_diagnostics.png`：十组已有生成的原始精确群、0.6 Å距离初筛、轨道不重合检查。M1/M2每模式64次，M3每来源每模式256次。它不是标准联合有效率图。
- 两份CSV保留逐轮与逐组分母；`summary.json`列出源文件SHA256、测试来源和统计。

三套均为多性质CFG训练。M1/M2与M3使用不同数据划分、性质标签和样本预算，不能按原始损失或小样本通过率直接排名。M1/M2完整同协议SMACT、COV、VUN尚待补齐，M3标准指标另见 `assets/paper/standard` 对应结果。

所有新图仅导出PNG；原始素材保持不变。
