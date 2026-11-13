# 参考DiffCSP++的职责重构验证（2026-10-30）

- 完整回归 **344 passed、0 failed**，741.37秒；285份实现/配置在测试期间哈希不变。
- 11个群约束、编码、网络、骨架和损失文件的函数/类AST保持一致，49份实验/启动YAML保持原字节。
- 固定SG/LG输入、初始化和随机数下73个数值/梯度条目精确一致；CSP/DNG四组逐样本与汇总语义一致，仅剔除计时和代码/文件身份字段。
- 整理后的独立wheel在无仓库源码搜索路径下完成生成→输出核验→续跑复用→CSV/JSON/PNG；三套真实基础权重严格加载通过。
- 新职责、清理范围、上游源码引用见[CODE_ORGANIZATION.md](CODE_ORGANIZATION.md)。机器结果见[本次verification.json](../outputs/tests/code_simplification/26-10-30-17-26-17/verification.json)，源码备份before.zip在同目录。
- 初次拆分的依赖遗漏、公共重导出恢复和修复后结果均保留日志。没有改原始数据/权重、启动正式训练、连接服务器、推送GitHub或发布网页。

以下为前一轮环境清理验证，测试数量和源码快照属于各自阶段。

---

# 环境与命令整理验证（2026-10-30）

已在独立Conda环境unifiedgroupgen完成安装与验证，保留原cgdit。唯一环境清单为unifiedgroupgen/environment.yml；唯一安装元数据为unifiedgroupgen/pyproject.toml。旧environment目录、根重复pyproject、requirements系列及constraints-tested已清理；旧CLI、命令分发、历史调度与操作说明归档到unifiedgroupgen/tests/legacy。

## 最终结果

- **341 passed，0 failed**，753秒；实际使用新Conda环境，包含CPU/CUDA、FP32/BF16、训练保存/恢复、损失与梯度、SG/LG编译、控制生成及统一指标。
- 测试期间295份源码/配置哈希不变；49份原实验与启动YAML保持原字节，模型类、骨架、群编译和投影数学保持原实现。
- 111个固定包版本核对通过，pip check无冲突。Python3.10.18；Torch运行/发行元数据均为2.7.1+cu118，CUDA11.8，RTX3060实际前向与反向通过；PyG扩展导入/运算、层群目录通过。
- Conda激活自动应用UTF-8，真实NaCl/SMACT化学检查通过。没有改化学阈值或SMACT数据表。
- MP20、C2DB、联合数据门禁通过；三套真实基础模型严格加载通过，没有重写数据或权重。
- 最终wheel独立安装后，从不含仓库源码的工作目录完成两项生成、哈希核验、续跑复用、CSV/JSON/PNG指标计算；不以这些随机小样本主张材料质量。
- 当前控制台只有ugg-train、ugg-generate、ugg-evaluate；当前代码不导入tests/legacy。
- 15个绘图/整理工具、三个示例和网页导出模块可导入。外部CGDiT生成性质、M3GNet及数据模块导入通过；未执行外部权重的性质精度评价。

## 安装中发现并处理的问题

旧pip缓存权限、审计临时路径过长、并行构建时过早清理目录，以及Windows GBK读取SMACT数据失败均保留原始日志。修复使用独立缓存、短临时目录、顺序构建/测试和Conda环境UTF-8变量；未修改全局缓存权限、系统长路径设置、GPU驱动或原cgdit环境。

## 证据与范围

完整机器报告、版本、XML、源码快照、安装日志和wheel位于 [本次验证目录](../tmp/environment_cleanup_20261030_160127/verification.json)。清理前源码备份为同目录before.zip；历史验证记录在tests/legacy/docs/VALIDATION.md。

本次没有连接服务器、启动正式训练、推送GitHub或发布网页。既有研究数据、权重、生成结果、失败记录和冻结实验快照保留。

安装与复查命令见[ENVIRONMENT.md](ENVIRONMENT.md)，当前训练/恢复见[LIGHTNING_TRAINING.md](LIGHTNING_TRAINING.md)。
