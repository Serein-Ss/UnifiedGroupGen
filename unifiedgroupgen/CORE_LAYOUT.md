# 核心库布局

本次只整理核心代码职责。配置、数据、权重、结果、网页和服务器冻结源码不迁移；不改变模型参数注册、数学路径或采样规则。

| 目录 | 职责 |
|---|---|
| common | 配置、路径、文件、哈希、精度、结构与通用显示工具 |
| conditional | 条件编码、可见性及任务规则；组分限制仍按硬任务约束处理 |
| evaluation | 结构指标、CSP匹配、性质代理及指标报告 |
| generation | two_stage与unified生成算法、轨迹和共享结果容器 |
| group_constraints | 唯一的操作、轨道、状态、合法变更与显式约束执行核心 |
| pl_data | 记录编码、目标恢复、共用教师程序、动态周期图和训练中间样本 |
| pl_modules | 一级共享层、主干和训练辅助；two_stage、unified两种架构 |
| tests | 既有测试及兼容回归；正式运行用的检查函数由对应业务目录实现 |

## 两种架构

- `pl_modules/two_stage`：DescriptorTransformer、OrbitFlow、训练目标和Lightning模块。
- `pl_modules/unified`：JointFlow、JointAtomGraph、周期Equiformer特征适配、联合目标和Lightning模块。
- `pl_modules/backbones/official` 保持原位置和原字节，包含许可证及数学常量；不重新包装主干或增加参数父层。
- `pl_modules/layers.py` 是共享图层；`training_utils.py` 保留精度策略及epoch边界规范化。

## 数据、约束和采样

`pl_data/targets.py` 只恢复端点，`programs.py` 是共用教师程序，`unified/path.py` 构造当前联合训练路径。`two_stage/path.py` 构造原几何插值。旧联合路径在 `unified/compat_path.py` 保留。

`group_constraints/flow_state.py` 保留当前状态与合法动作核，训练和生成共用。其 `advance` 只执行约束状态的确定性增量；Heun积分和事件时钟由 `generation/unified/sampling.py` 管理。

`pl_data/periodic_graph.py` 是运行时动态图构造，保留最终边向量的坐标与晶格梯度。预处理不意味着离线固定邻居或detach几何。

`generation/types.py` 定义共享JointSample，`trajectory.py` 保存观察轨迹。生成调度在 `generation/unified/runner.py`；根目录命令入口保持。

## 兼容性

旧 `joint_flow/*`、`pl_modules/descriptor.py`、`joint_training.py` 等位置保留模块转发，旧导入与新规范导入得到同一个模块对象；私有辅助名称和monkeypatch行为也保留。混合职责的 `joint_path`、`task_path` 由兼容模块导出拆分后的对象。

实际实现仅保留一份。原生检查点中的配置、张量键名、优化器、调度器、随机状态和循环协议不改写；旧模块路径仍可解析。正式模型通过原入口读取，不转换已有权重。

性质代理搬迁后的评估源码身份会自然改变，新的评估缓存不复用旧源码身份；历史报告和manifest不重写。公开代理注册表在原路径保留兼容资源，新增规范资源内容相同。

迁移索引见 `core_layout.json`。新测试不执行真实训练，CPU前向/自动微分只用于比较输出、损失及梯度；本次禁止调用Trainer.fit和优化器更新。
