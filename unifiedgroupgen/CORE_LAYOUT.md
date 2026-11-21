# UnifiedGroupGen 核心代码

生产模型为 `joint_flow_v2`。EquiformerV2 是共享主干，JointAtomGraph 用于主干对照。DNG 与精确组分 CSP 独立配置，目标性质 CFG 可以叠加到任一任务。

| 位置 | 职责 |
|---|---|
| `common/` | 通用路径与环境、晶体解析和周期图算子、结构读写、哈希与进度显示 |
| `conditional/` | 标量目标性质编码、训练可见性、CFG 的空条件分支及任务规则 |
| `group_constraints/` | 群操作编译、Wyckoff 映射、轨道与晶格约束、合法事件及库存可行性 |
| `pl_data/cache.py` | 从 CSV 构建和复用当前 V3 Tensor 缓存，保存缓存配置 |
| `pl_data/dataset.py` 和 `datamodule.py` | 单结构 PyG Data、批处理、加载与同一运行的批次恢复 |
| `pl_data/unified/` | 从真实终点在线构造训练路径和监督 |
| `pl_modules/backbones/` | 网络主干与 EquiformerV2 依赖 |
| `pl_modules/unified/` | 共享网络、三部分联合损失、Lightning 训练和原生检查点 |
| `generation/` | 批量生成、连续积分、离散轨道事件、结构账本与轨迹 |
| `evaluation/` | 独立结构评估、CSP 匹配、性质预测器和报告 |
| `tests/unit/` | 当前实现的回归测试 |
| `tests/legacy_code/` | 旧源码和实验配置归档 |

## 参数归属

- `.env` 指定工作区、输出目录与 W&B 日志目录。
- `configs/sources/` 定义原始 CSV 的位置、列映射和对称性识别协议。`configs/data/` 用来选择单库或混合数据。
- `configs/model/` 定义模型规模和周期构图参数。`configs/train/` 定义训练目标、路径及验证策略。`configs/optim/`、`configs/conditioning/` 和 `configs/runtime/` 分别管理优化器、目标性质和设备执行参数。
- `configs/experiment/` 组合配置组并选择 DNG 或 CSP。`experiment_config.py` 将 Hydra 配置转换为模型使用的训练定义。
- `configs/evaluation/` 与 `configs/property_predictor/` 管理独立评估和性质预测设置。

## 从头运行

`data/process.py → pl_data/cache.py → CrystalDataset → DataModule → 联合中间态 → 共享模型`。默认从来源 CSV 选择或构建内容指纹匹配的缓存，使用训练集拟合元素词表、群频率和性质归一化。审计报告为显式选项。训练计划的 `resume` 和 `initialize_from` 默认为空。

`generate.py → generation/unified/` 使用用户明确指定的当前检查点。`evaluate.py → evaluation/` 使用用户明确指定的样本和参考数据。旧格式检查点、旧路径映射、JSONL 训练缓存迁移和历史批次修复入口已退出主线。当前格式检查点可在训练配置和数据身份相符时显式续训。

本轮移出的源码和配置保存在 `tests/legacy_code/fresh_core_20261119/`。历史运行文件仍在各自的原始位置。
